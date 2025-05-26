_base_ = [
    '../_base_/models/hv_pointpillars_fpn_nus.py',
    '../_base_/datasets/nus-3d.py', '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['my_hooks.profiler_hook'],
    allow_failed_imports=False
)

custom_hooks = [
    dict(
        type='MyProfilerHook',
        log_dir='./profiler_logs',
        profile_step_start=20,
        profile_step_end=22,
        priority='NORMAL'
    )
]
