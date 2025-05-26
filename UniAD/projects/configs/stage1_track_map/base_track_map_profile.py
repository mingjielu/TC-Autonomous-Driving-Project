_base_ = [
    './base_track_map.py',
]

custom_imports = dict(
    imports=['my_hooks.profiler_hook'],
    allow_failed_imports=False
)

custom_hooks = [
    dict(
        type='MyProfilerHook',
        log_dir='./profiler_logs',
        profile_step_start=60,
        profile_step_end=62,
        priority='NORMAL'
    )
]
