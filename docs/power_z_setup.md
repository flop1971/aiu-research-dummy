# POWER/Z Host Setup Guide

## Supported Platforms

torch-spyre supports the following host CPU architectures:

| Architecture | Status      | Notes                          |
|-------------|-------------|-------------------------------|
| x86_64      | Stable      | Primary development platform  |
| POWER10     | Beta        | Validated on select workloads |
| POWER9      | Experimental| CI runner available            |
| IBM z16     | Experimental| s390x, limited test coverage  |
| IBM z15     | Planned     | Roadmap item                  |

## Prerequisites

### POWER10 / POWER9

```bash
# Install POWER-compatible build dependencies
sudo dnf install gcc-toolset-13 cmake ninja-build
export CC=gcc CXXFLAGS="-mcpu=power10"
```

### IBM z16 / z15 (s390x)

```bash
# Install Z-compatible build dependencies
sudo apt-get install gcc-12 cmake ninja-build
export CC=gcc CXXFLAGS="-march=z16"
```

## Known Issues

- NUMA topology detection requires `numactl` on POWER hosts
- Endianness handling in tensor copy is validated on z16 only
- See label `platform: power-z` in the issue tracker for active work
