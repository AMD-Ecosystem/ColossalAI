# Setup

Requirements:
- 2.2 <= PyTorch < 2.14
- Python >= 3.10 and < 3.13
- CUDA >= 11.0
- [NVIDIA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus) >= 7.0 (V100/RTX20 and higher), or an AMD GPU with ROCm (see [AMD GPUs with ROCm](#amd-gpus-with-rocm))
- Linux OS

If you encounter any problem about installation, you may want to raise an [issue](https://github.com/hpcaitech/ColossalAI/issues/new/choose) in this repository.


## Download From PyPI

You can install Colossal-AI with

```shell
pip install colossalai
```

**Note: only Linux is supported for now**

If you want to build PyTorch extensions during installation, you can use the command below. Otherwise, the PyTorch extensions will be built during runtime.

```shell
BUILD_EXT=1 pip install --no-binary=colossalai --no-build-isolation colossalai
```


## Download From Source

> The version of Colossal-AI will be in line with the main branch of the repository. Feel free to raise an issue if you encounter any problem.

```shell
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI

# install dependency
pip install -r requirements/requirements.txt

# install colossalai
BUILD_EXT=1 pip install --no-build-isolation .
```

If you don't want to install and enable CUDA kernel fusion (compulsory installation when using fused optimizer), just don't specify the `BUILD_EXT`:

```shell
pip install .
```

For Users with CUDA 10.2, you can still build ColossalAI from source. However, you need to manually download the cub library and copy it to the corresponding directory.

```bash
# clone the repository
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI

# download the cub library
wget https://github.com/NVIDIA/cub/archive/refs/tags/1.8.0.zip
unzip 1.8.0.zip
cp -r cub-1.8.0/cub/ colossalai/kernel/cuda_native/csrc/kernels/include/

# install
BUILD_EXT=1 pip install --no-build-isolation .
```

## AMD GPUs with ROCm

Colossal-AI also runs on AMD GPUs supported by ROCm. Install a ROCm build of PyTorch first (see the [PyTorch installation guide](https://pytorch.org/get-started/locally/)), then install Colossal-AI from source against it:

```shell
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI
pip install -r requirements/requirements.txt
BUILD_EXT=1 PYTORCH_ROCM_ARCH=gfx942 pip install --no-build-isolation .
```

`PYTORCH_ROCM_ARCH` lists the GPU architectures to compile the kernels for, separated by semicolons, for example `gfx942` for the Instinct MI300 series or `gfx1100` for the Radeon RX 7900 series. Without it, the kernels are compiled for every architecture your PyTorch build supports. Without `BUILD_EXT=1`, they are compiled at first use, as on CUDA.

The FlashAttention-2 attention backend needs the ROCm build of `flash-attn`; see its [AMD ROCm instructions](https://github.com/Dao-AILab/flash-attention#amd-rocm-support). Without it, attention uses PyTorch's `scaled_dot_product_attention`.

<!-- doc-test-command: echo "installation.md does not need test" -->
