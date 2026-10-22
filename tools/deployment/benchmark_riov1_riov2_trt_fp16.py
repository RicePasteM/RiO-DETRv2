#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import onnx
import tensorrt as trt
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.core import YAMLConfig


SPECS = ('n', 's', 'm', 'l', 'x')
VERSIONS = ('riov1', 'riov2')


class DeployModel(nn.Module):
    def __init__(self, model, postprocessor):
        super().__init__()
        self.model = model.deploy()
        self.postprocessor = postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        return self.postprocessor(self.model(images), orig_target_sizes)


def config_path(version, spec):
    prefix = 'rio' if version == 'riov1' else 'riov2'
    return ROOT / 'configs' / 'rio-diorr' / f'{prefix}_hgnetv2_{spec}_diorr.yml'


def activate_zero_initialized_paths(model, seed):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    changed = []
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or not parameter.is_floating_point():
                continue
            if torch.count_nonzero(parameter).item() == 0:
                values = torch.randn(
                    parameter.shape,
                    dtype=parameter.dtype,
                    device='cpu',
                    generator=generator,
                )
                parameter.copy_(values.to(parameter.device) * 0.02)
                changed.append(name)
    return changed


def export_onnx(version, spec, output_path, image_size, simplify):
    cfg_path = config_path(version, spec)
    print(f'EXPORT {version}-{spec}: {cfg_path}', flush=True)
    cfg = YAMLConfig(str(cfg_path))
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    model = cfg.model
    changed = activate_zero_initialized_paths(model, 20260714)
    print(f'  activated_zero_initialized_tensors={len(changed)}', flush=True)

    wrapper = DeployModel(model, cfg.postprocessor).eval()
    images = torch.rand(1, 3, image_size, image_size)
    sizes = torch.tensor([[image_size, image_size]], dtype=torch.int64)
    with torch.no_grad():
        outputs = wrapper(images, sizes)
    assert len(outputs) == 3

    torch.onnx.export(
        wrapper,
        (images, sizes),
        str(output_path),
        input_names=['images', 'orig_target_sizes'],
        output_names=['labels', 'boxes', 'scores'],
        dynamic_axes=None,
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
    )
    model_proto = onnx.load(str(output_path))
    onnx.checker.check_model(model_proto)
    if simplify:
        try:
            import onnxsim
            simplified, valid = onnxsim.simplify(model_proto, check_n=1)
            if not valid:
                raise RuntimeError('onnxsim validation returned false')
            onnx.save(simplified, str(output_path))
            model_proto = simplified
            print('  onnx_simplified=true', flush=True)
        except Exception as exc:
            print(f'  onnx_simplified=false reason={exc}', flush=True)
    node_count = len(model_proto.graph.node)
    print(f'  onnx_nodes={node_count} size_mb={output_path.stat().st_size / 2**20:.1f}', flush=True)
    del wrapper, model, cfg, model_proto, images, sizes, outputs
    gc.collect()
    return node_count


def load_timing_cache(config, cache_path):
    data = cache_path.read_bytes() if cache_path.exists() else b''
    cache = config.create_timing_cache(data)
    config.set_timing_cache(cache, ignore_mismatch=False)


def save_timing_cache(config, cache_path):
    cache = config.get_timing_cache()
    cache_path.write_bytes(bytes(cache.serialize()))


def build_engine(onnx_path, engine_path, cache_path, workspace_gib, opt_level):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    if not builder.platform_has_fast_fp16:
        raise RuntimeError('TensorRT reports that fast FP16 is unavailable')
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = '\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f'ONNX parse failed:\n{errors}')

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    config.builder_optimization_level = opt_level
    load_timing_cache(config, cache_path)

    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        if tensor.dtype == trt.float32:
            tensor.dtype = trt.float16
            tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
    for index in range(network.num_outputs):
        tensor = network.get_output(index)
        if tensor.dtype == trt.float32:
            tensor.dtype = trt.float16
            tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)

    print(f'BUILD {engine_path.name}: opt_level={opt_level} workspace={workspace_gib}GiB', flush=True)
    started = time.time()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError('TensorRT engine build returned None')
    engine_path.write_bytes(bytes(plan))
    save_timing_cache(config, cache_path)
    build_seconds = time.time() - started
    print(
        f'  build_seconds={build_seconds:.1f} engine_mb={engine_path.stat().st_size / 2**20:.1f}',
        flush=True,
    )
    return build_seconds


def torch_dtype(dtype):
    mapping = {
        trt.float16: torch.float16,
        trt.float32: torch.float32,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.bool: torch.bool,
    }
    if dtype not in mapping:
        raise TypeError(f'Unsupported TensorRT dtype: {dtype}')
    return mapping[dtype]


def static_or_opt_shape(engine, name):
    shape = tuple(engine.get_tensor_shape(name))
    if -1 not in shape:
        return shape
    profile = engine.get_tensor_profile_shape(name, 0)
    return tuple(profile[1])


def benchmark_engine(engine_path, warmup, runs, repeats):
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f'Failed to deserialize {engine_path}')
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    tensors = {}

    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
            continue
        shape = static_or_opt_shape(engine, name)
        dtype = torch_dtype(engine.get_tensor_dtype(name))
        if dtype.is_floating_point:
            tensor = torch.randn(shape, dtype=dtype, device='cuda')
        else:
            tensor = torch.full(shape, 800, dtype=dtype, device='cuda')
        if -1 in tuple(engine.get_tensor_shape(name)):
            context.set_input_shape(name, shape)
        context.set_tensor_address(name, tensor.data_ptr())
        tensors[name] = tensor

    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
            continue
        shape = tuple(context.get_tensor_shape(name))
        if any(dim < 0 for dim in shape):
            raise RuntimeError(f'Unresolved output shape for {name}: {shape}')
        dtype = torch_dtype(engine.get_tensor_dtype(name))
        tensor = torch.empty(shape, dtype=dtype, device='cuda')
        context.set_tensor_address(name, tensor.data_ptr())
        tensors[name] = tensor

    image_dtype = str(tensors['images'].dtype).replace('torch.', '')
    if tensors['images'].dtype != torch.float16:
        raise RuntimeError(f'Expected FP16 image input, got {tensors["images"].dtype}')

    for _ in range(warmup):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('TensorRT warmup execution failed')
    stream.synchronize()

    latencies = []
    for repeat in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for _ in range(runs):
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError('TensorRT benchmark execution failed')
        end.record(stream)
        end.synchronize()
        latency = start.elapsed_time(end) / runs
        latencies.append(latency)
        print(f'  repeat={repeat + 1}/{repeats} latency_ms={latency:.5f}', flush=True)
        time.sleep(0.2)

    result = {
        'latency_ms_median': statistics.median(latencies),
        'latency_ms_mean': statistics.mean(latencies),
        'latency_ms_min': min(latencies),
        'latency_ms_max': max(latencies),
        'latency_ms_stdev': statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        'fps': 1000.0 / statistics.median(latencies),
        'repeat_latencies_ms': latencies,
        'image_input_dtype': image_dtype,
    }
    del tensors, context, engine, runtime
    gc.collect()
    torch.cuda.empty_cache()
    return result


def write_results(rows, output_dir, metadata):
    json_path = output_dir / 'results.json'
    csv_path = output_dir / 'results.csv'
    json_path.write_text(json.dumps({'metadata': metadata, 'results': rows}, indent=2))
    if rows:
        columns = [key for key in rows[0] if key != 'repeat_latencies_ms']
        with csv_path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row[key] for key in columns})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--image-size', type=int, default=800)
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--runs', type=int, default=1000)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--workspace-gib', type=int, default=8)
    parser.add_argument('--opt-level', type=int, default=5)
    parser.add_argument('--no-simplify', action='store_true')
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--only', nargs='*')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / 'timing.cache'
    selected = args.only or [f'{version}-{spec}' for spec in SPECS for version in VERSIONS]
    metadata = {
        'gpu': torch.cuda.get_device_name(0),
        'tensorrt': trt.__version__,
        'torch': torch.__version__,
        'image_shape': [1, 3, args.image_size, args.image_size],
        'warmup': args.warmup,
        'runs_per_repeat': args.runs,
        'repeats': args.repeats,
        'fp16_io': True,
        'cuda_graph': False,
        'weights': 'deterministic random; all-zero trainable tensors activated',
        'builder_optimization_level': args.opt_level,
        'workspace_gib': args.workspace_gib,
        'onnx_simplified': not args.no_simplify,
    }
    rows = []
    for item in selected:
        version, spec = item.split('-', 1)
        if version not in VERSIONS or spec not in SPECS:
            raise ValueError(f'Invalid model selection: {item}')
        onnx_path = args.output_dir / f'{item}_b1_{args.image_size}.onnx'
        engine_path = args.output_dir / f'{item}_b1_{args.image_size}_fp16.engine'
        node_count = None
        build_seconds = 0.0
        if args.rebuild or not engine_path.exists():
            node_count = export_onnx(
                version, spec, onnx_path, args.image_size, not args.no_simplify
            )
            build_seconds = build_engine(
                onnx_path,
                engine_path,
                cache_path,
                args.workspace_gib,
                args.opt_level,
            )
        elif onnx_path.exists():
            node_count = len(onnx.load(str(onnx_path)).graph.node)
        print(f'BENCHMARK {item}', flush=True)
        result = benchmark_engine(engine_path, args.warmup, args.runs, args.repeats)
        row = {
            'version': version,
            'spec': spec,
            'config': str(config_path(version, spec).relative_to(ROOT)),
            'onnx_nodes': node_count,
            'engine_mb': engine_path.stat().st_size / 2**20,
            'build_seconds': build_seconds,
            **result,
        }
        rows.append(row)
        write_results(rows, args.output_dir, metadata)
        print('RESULT=' + json.dumps(row), flush=True)

    print(f'WROTE {args.output_dir / "results.csv"}', flush=True)
    print(f'WROTE {args.output_dir / "results.json"}', flush=True)


if __name__ == '__main__':
    main()
