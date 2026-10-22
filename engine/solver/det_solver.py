"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import time
import json
import datetime
import math
import os

import torch

from ..misc import dist_utils, stats

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler

try:
    import swanlab
except ImportError:
    swanlab = None


class DetSolver(BaseSolver):

    def fit(self, ):
        self.train()
        args = self.cfg

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)
        # SwanLab init
        if args.use_swanlab and swanlab is not None and dist_utils.is_main_process():
            swanlab_config = {
                "epoches": args.epoches,
                "batch_size": args.batch_size,
                "checkpoint_freq": args.checkpoint_freq,
                "use_ema": args.use_ema,
                "use_amp": args.use_amp,
            }
            # Merge yaml config
            if hasattr(args, "yaml_cfg"):
                for k, v in args.yaml_cfg.items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        swanlab_config[k] = v

            experiment_name = args.swanlab_experiment
            if experiment_name is None and args.output_dir:
                experiment_name = os.path.basename(args.output_dir.rstrip("/"))

            swanlab_run_id = args.swanlab_run_id
            swanlab_resume = args.swanlab_resume
            swanlab_run_id_file = None
            if self.output_dir:
                swanlab_run_id_file = self.output_dir / "swanlab_run_id.txt"
                if swanlab_run_id is None and args.resume and swanlab_run_id_file.exists():
                    swanlab_run_id = swanlab_run_id_file.read_text().strip() or None
            if swanlab_run_id and args.resume and swanlab_resume is None:
                swanlab_resume = "must"

            swanlab_kwargs = {}
            if swanlab_run_id:
                swanlab_kwargs["id"] = swanlab_run_id
            if swanlab_resume:
                swanlab_kwargs["resume"] = swanlab_resume

            run = swanlab.init(
                project=args.swanlab_project,
                workspace=args.swanlab_workspace,
                experiment_name=experiment_name,
                config=swanlab_config,
                **swanlab_kwargs,
            )
            if swanlab_run_id_file is not None:
                current_run_id = getattr(run, "id", None)
                if current_run_id:
                    swanlab_run_id_file.write_text(str(current_run_id))

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch, total_epochs=args.epoches,
                                                warmup_iter=args.warmup_iter, flat_epochs=args.flat_epoch, no_aug_epochs=args.no_aug_epoch)
            self.self_lr_scheduler = True
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        primary_metric = args.yaml_cfg.get('primary_metric')
        if primary_metric is not None:
            print(f'Primary validation metric: {primary_metric}')

        def metric_scalar(stat_value):
            if isinstance(stat_value, (list, tuple)):
                return stat_value[0]
            if isinstance(stat_value, torch.Tensor):
                if stat_value.numel() > 1:
                    return stat_value[0].item()
                return stat_value.item()
            return stat_value

        top1 = 0
        best_stat = {'epoch': -1, }
        # evaluate again before resume training
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                output_dir=self.output_dir,
                current_epoch=self.last_epoch,
            )
            resume_metrics = {}
            for k, stat_value in test_stats.items():
                current_val = metric_scalar(stat_value)
                resume_metrics[k] = current_val
                best_stat['epoch'] = self.last_epoch
                best_stat[k] = current_val
                if primary_metric is None:
                    # Preserve the legacy behavior for existing configurations.
                    top1 = current_val
                if (args.use_swanlab and swanlab is not None
                        and dist_utils.is_main_process()
                        and isinstance(current_val, (int, float))):
                    swanlab.log({f'test_{k}': current_val}, step=self.last_epoch)
                print(f'best_stat: {best_stat}')

            if primary_metric is not None:
                if primary_metric not in resume_metrics:
                    raise KeyError(
                        f'Primary metric {primary_metric!r} is unavailable; '
                        f'evaluator returned {sorted(resume_metrics)}')
                top1 = resume_metrics[primary_metric]
                print(
                    f'Resumed primary metric {primary_metric}={top1:.6f} '
                    f'at epoch {self.last_epoch}')

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):
            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            train_stats, grad_percentages = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                teacher_model=self.teacher_model,
                output_dir=self.output_dir,
            )

            if not self.self_lr_scheduler:  # update by epoch 
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1
            distill_adaptive_enabled = hasattr(self.criterion, 'distill_adaptive_params') and \
                self.criterion.distill_adaptive_params and \
                self.criterion.distill_adaptive_params.get('enabled', False)
            if distill_adaptive_enabled:

                params = self.criterion.distill_adaptive_params
                default_weight = params.get('default_weight')

                current_weight = self.criterion.weight_dict.get('loss_distill', 0.0)
                new_weight = current_weight
                reason = 'unchanged'
                avg_percentage = 0.0

                # DDP gradients have already been reduced, so rank 0 keeps the
                # original RT-DETRv4 controller calculation. The resulting
                # scalar must then be broadcast because criterion is not DDP.
                if dist_utils.is_main_process():
                    avg_percentage = sum(grad_percentages) / len(grad_percentages) if grad_percentages else 0.0

                    if avg_percentage < 1e-6:
                        if default_weight is not None:
                            new_weight = default_weight
                            reason = 'reset_to_default_zero_grad'
                    elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if default_weight is not None:
                            new_weight = default_weight
                            reason = 'ema_phase_default'
                    else:
                        rho = params['rho']
                        delta = params['delta']
                        lower_bound = rho - delta
                        upper_bound = rho + delta
                        if not (lower_bound <= avg_percentage <= upper_bound):
                            target_percentage = upper_bound if avg_percentage < lower_bound else lower_bound
                            if current_weight > 1e-6:
                                p_current = avg_percentage / 100.0
                                p_target = target_percentage / 100.0
                                numerator = p_target * (1.0 - p_current)
                                denominator = p_current * (1.0 - p_target)
                                if abs(denominator) >= 1e-9:
                                    ratio = numerator / denominator
                                    ratio = max(ratio, 0.1)  # clamp non-positive to 0.1
                                    new_weight = current_weight * ratio
                                    new_weight = min(max(new_weight, current_weight / 10.0), current_weight * 10.0)
                                    reason = f'adjusted_to_{target_percentage:.2f}%'

                if dist_utils.is_dist_available_and_initialized():
                    synced_weight = torch.tensor([new_weight], dtype=torch.float64, device=self.device)
                    torch.distributed.broadcast(synced_weight, src=0)
                    new_weight = synced_weight.item()

                self.criterion.weight_dict['loss_distill'] = new_weight
                if dist_utils.is_main_process():
                    print(f"Epoch {epoch}: avg encoder grad {avg_percentage:.2f}% | "
                          f"distill {current_weight:.6f} -> {new_weight:.6f} "
                          f"({reason}, synced_to_all_ranks)")

            if self.output_dir:
                checkpoint_paths = []
                if epoch < self.train_dataloader.collate_fn.stop_epoch:
                    checkpoint_paths.append(self.output_dir / 'last.pth')

                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                output_dir=self.output_dir,
                current_epoch=epoch,
            )

            epoch_metrics = {}
            for k, stat_value in test_stats.items():
                current_val = metric_scalar(stat_value)
                epoch_metrics[k] = current_val

                if self.writer and dist_utils.is_main_process():
                    if isinstance(stat_value, (list, tuple)):
                        for i, v in enumerate(stat_value):
                            self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)
                    else:
                        self.writer.add_scalar(f'Test/{k}', current_val, epoch)

                if primary_metric is None:
                    # Legacy multi-metric checkpoint behavior remains opt-out.
                    if k in best_stat:
                        best_stat['epoch'] = epoch if current_val > best_stat[k] else best_stat['epoch']
                        best_stat[k] = max(best_stat[k], current_val)
                    else:
                        best_stat['epoch'] = epoch
                        best_stat[k] = current_val

                    if best_stat[k] > top1:
                        best_stat_print['epoch'] = epoch
                        top1 = best_stat[k]
                        if self.output_dir:
                            if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                                dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                            else:
                                dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')

                    best_stat_print[k] = max(best_stat[k], top1)
                    print(f'best_stat: {best_stat_print}')  # global best

                    if best_stat['epoch'] == epoch and self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            if current_val > top1:
                                top1 = current_val
                                dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                        else:
                            top1 = max(current_val, top1)
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')

                    elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        best_stat = {'epoch': -1, }
                        self.ema.decay -= 0.0001
                        self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                        print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            if primary_metric is not None:
                if primary_metric not in epoch_metrics:
                    raise KeyError(
                        f'Primary metric {primary_metric!r} is unavailable; '
                        f'evaluator returned {sorted(epoch_metrics)}')

                current_primary = epoch_metrics[primary_metric]
                improved = current_primary > top1
                for k, current_val in epoch_metrics.items():
                    best_stat[k] = max(best_stat.get(k, current_val), current_val)

                if improved:
                    top1 = current_primary
                    best_stat['epoch'] = epoch
                    if self.output_dir:
                        checkpoint_name = (
                            'best_stg2.pth'
                            if epoch >= self.train_dataloader.collate_fn.stop_epoch
                            else 'best_stg1.pth'
                        )
                        dist_utils.save_on_master(
                            self.state_dict(), self.output_dir / checkpoint_name)

                print(
                    f'primary_metric: {primary_metric} '
                    f'current={current_primary:.6f} best={top1:.6f} '
                    f'best_epoch={best_stat["epoch"]} improved={improved}')


            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # SwanLab logging
                if args.use_swanlab and swanlab is not None:
                    swanlab_log = {}
                    for k, v in log_stats.items():
                        if k == "epoch" or k == "n_parameters":
                            continue
                        if isinstance(v, (int, float)):
                            swanlab_log[k] = v
                    if swanlab_log:
                        swanlab.log(swanlab_log, step=epoch)

                # for evaluation logs
                if coco_evaluator is not None and hasattr(coco_evaluator, 'coco_eval'):
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        if args.use_swanlab and swanlab is not None and dist_utils.is_main_process():
            swanlab.finish()


    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device, output_dir=self.output_dir)

        if self.output_dir and hasattr(coco_evaluator, 'coco_eval'):
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return


    def state_dict(self):
        """State dict, train/eval"""
        state = {}
        state['date'] = datetime.datetime.now().isoformat()

        # For resume
        state['last_epoch'] = self.last_epoch

        for k, v in self.__dict__.items():
            if k == 'teacher_model':
                continue
            if hasattr(v, 'state_dict'):
                v = dist_utils.de_parallel(v)
                state[k] = v.state_dict()

        return state
