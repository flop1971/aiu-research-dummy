# Profiling Guide for torch-spyre

## Overview

This guide explains how to profile PyTorch model execution on the Spyre AIU
accelerator using the `torch_spyre.profiler` module.

## Basic Usage

```python
from torch_spyre.profiler import SpyreProfiler

with SpyreProfiler() as prof:
    output = model(input_tensor)

print(prof.summary())
```

## Granite Model Example

The following example profiles a Granite language model forward pass:

```python
import torch
from torch_spyre.profiler import SpyreProfiler

model = load_granite_model()
input_ids = torch.randint(0, 32000, (1, 512))

with SpyreProfiler() as prof:
    with torch.no_grad():
        logits = model(input_ids)

print(prof.summary())
```

## Platform Notes

### x86 Host
Standard profiling configuration applies.

### POWER/Z Host
On POWER10 and IBM z16 hosts, additional NUMA topology profiling
is available via the `numa_aware=True` flag:

```python
with SpyreProfiler(numa_aware=True) as prof:
    output = model(input_tensor)
```
