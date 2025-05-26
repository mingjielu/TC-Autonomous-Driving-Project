# https://github.com/open-mmlab/mmcv/blob/v1.7.1/mmcv/runner/hooks/hook.py
import os
import torch
# from mmengine.hooks import Hook
# from mmdet.registry import HOOKS
from mmcv.runner.hooks.hook import HOOKS, Hook


@HOOKS.register_module()
class MyProfilerHook(Hook):
    def __init__(self, log_dir='./profiler_logs', profile_step_start=1, profile_step_end=2, wait=None, warmup=None, active=None, repeat=1):
        self.log_dir = log_dir
        self.profile_step_start = profile_step_start
        self.profile_step_end = profile_step_end
        if wait is None:
            self.wait = max(profile_step_start-1, 0)
        else:
            self.wait = wait
        if warmup is None:
            self.warmup = 1 if profile_step_start > 0 else 0
        else:
            self.warmup = warmup
        if active is None:
            self.active = profile_step_end - profile_step_start
        else:
            self.active = active
        self.repeat = repeat
        self.profiler=None
        os.makedirs(log_dir, exist_ok=True)

    def before_run(self, runner):
        self.profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat
            ),
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.log_dir),
            record_shapes=True,
            with_stack=True,
            with_flops=True
        )
        self.profiler.__enter__()
        runner.logger.info(f"Init MyProfilerHook: {self.log_dir=}, {self.profile_step_start=}, {self.profile_step_end=}, {self.wait=}, {self.warmup=}, {self.active=}, {self.repeat=}")
        # self.profiler.start()

    def before_train_iter(self, runner):
        runner.logger.info(f'[MyProfilerHook] {runner.iter=}, Synchronize before training iter...')
        torch.cuda.synchronize()

    def after_train_iter(self, runner):
        assert self.profiler is not None
        torch.cuda.synchronize()
        self.profiler.step()
        if runner.iter == self.profile_step_end:
            self.profiler.__exit__(None, None, None)
            # self.profiler.stop()
            runner.logger.info(self.profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
            runner.logger.info(f"[MyProfilerHook] {runner.iter=}, end profiling")
            exit()

    def after_run(self, runner):
        self.profiler.__exit__(None, None, None)
        # self.profiler.stop()
