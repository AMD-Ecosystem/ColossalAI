import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from colossalai.accelerator import get_accelerator
from colossalai.quantization.fp8 import linear_fp8, scaled_mm_fp8_formats
from colossalai.quantization.fp8_hook import FP8Hook
from colossalai.tensor.colo_parameter import ColoParameter
from colossalai.tensor.param_op_hook import ColoParamOpHookManager
from colossalai.utils import get_current_device

REPLACED = False
TRIGGERED = False


def new_linear_fp8(x, w, bias=None):
    global TRIGGERED
    TRIGGERED = True
    return linear_fp8(x, w, bias)


class FP8TestHook(FP8Hook):
    def rewrite_op(self, func):
        func = super().rewrite_op(func)
        if func is linear_fp8:
            global REPLACED
            REPLACED = True
            return new_linear_fp8
        return func


# ROCm GPUs report AMD architecture numbers as their capability, so ask whether the device has an fp8 GEMM.
if torch.version.hip is not None:
    FP8_GEMM_UNSUPPORTED = scaled_mm_fp8_formats() is None
else:
    FP8_GEMM_UNSUPPORTED = get_accelerator().get_device_capability()[0] < 9

D_IN, D_OUT = 16, 32
B, S = 2, 64
DTYPE = torch.bfloat16


@pytest.mark.skipif(FP8_GEMM_UNSUPPORTED, reason="Test requires an fp8 GEMM (device capability >= 9.0 on CUDA)")
def test_fp8_hook():
    # create tensors
    w = nn.Parameter(torch.rand(D_OUT, D_IN, device=get_current_device(), dtype=DTYPE))
    x = torch.rand(B, S, D_IN, device=get_current_device(), dtype=DTYPE, requires_grad=True)
    w.__class__ = ColoParameter
    w.__init__(w, requires_grad=True)
    hook = FP8TestHook()
    with ColoParamOpHookManager.use_hooks(hook):
        o = F.linear(x, w)
    assert o.shape == (B, S, D_OUT)
    assert REPLACED
    assert TRIGGERED
