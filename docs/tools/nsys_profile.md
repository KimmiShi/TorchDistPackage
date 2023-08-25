
# nsys profile with cudaProfilerStart
```
nsys profile -o ~/work/timm-github/prof/nsysout -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi  --force-overwrite true
```

train code:
```
        if batch_idx>2:
            torch.cuda.cudart().cudaProfilerStart()
        if batch_idx>5:
            torch.cuda.cudart().cudaProfilerStop()
```