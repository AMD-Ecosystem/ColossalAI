import pytest
import torch
import torch.nn.functional as F

from colossalai.nn.layer.layernorm import MixedFusedLayerNorm

# fp32 statistics, so half-precision outputs differ from an fp32 reference by their own rounding.
TOLERANCES = {torch.float32: (1e-5, 1e-4), torch.float16: (1e-2, 1e-2), torch.bfloat16: (5e-2, 5e-2)}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden", [64, 100, 768, 1024, 5120])
@pytest.mark.parametrize("rows", [1, 7, 300])
def test_layernorm_kernel(dtype, hidden, rows):
    torch.manual_seed(0)
    layer = MixedFusedLayerNorm(hidden, device="cuda", dtype=dtype)
    with torch.no_grad():
        layer.weight.normal_(1.0, 0.1)
        layer.bias.normal_(0.0, 0.1)
    x = (torch.randn(rows, hidden, device="cuda") * 3 + 1).to(dtype).requires_grad_()
    grad = torch.randn(rows, hidden, device="cuda").to(dtype)

    out = layer(x)
    out.backward(grad)

    ref_x = x.detach().float().requires_grad_()
    ref_w = layer.weight.detach().float().requires_grad_()
    ref_b = layer.bias.detach().float().requires_grad_()
    ref = F.layer_norm(ref_x, (hidden,), ref_w, ref_b, layer.eps)
    ref.backward(grad.float())

    rtol, atol = TOLERANCES[dtype]
    torch.testing.assert_close(out.float(), ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(x.grad.float(), ref_x.grad, rtol=rtol, atol=atol)
    # Parameter gradients sum over rows; scale the tolerance with them.
    torch.testing.assert_close(layer.weight.grad.float(), ref_w.grad, rtol=rtol, atol=atol * rows)
    torch.testing.assert_close(layer.bias.grad.float(), ref_b.grad, rtol=rtol, atol=atol * rows)
