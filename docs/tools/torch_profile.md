```
    from torch.profiler import profile, record_function, ProfilerActivity
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=10, warmup=1, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./prof/trace_p1_base'),
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        profile_memory=True)
    prof.start()




    # training code ...

                optimizer.step()
                prof.step()
```