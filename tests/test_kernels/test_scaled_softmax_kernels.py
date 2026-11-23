import pytest
import torch

from colossalai.kernel.kernel_loader import ScaledMaskedSoftmaxLoader, ScaledUpperTriangleMaskedSoftmaxLoader

# Key lengths cover every power-of-two kernel instantiation the launchers pick, plus ragged lengths.
# Above 64 keys the kernels load four elements at a time, so the length must be a multiple of 4.
KEY_LENGTHS = [16, 17, 32, 33, 63, 64, 128, 132, 256, 512, 1000, 1024, 2044, 2048]
SCALE = 0.7


def softmax_reference(x, scale, masked):
    logits = (x.float() * scale).masked_fill(masked, -10000.0)
    return torch.softmax(logits, dim=-1)


def softmax_backward_reference(grad, probs, scale):
    grad, probs = grad.float(), probs.float()
    return scale * probs * (grad - (grad * probs).sum(dim=-1, keepdim=True))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sk", KEY_LENGTHS)
@pytest.mark.parametrize("pad_batches", [1, 2])
def test_scaled_masked_softmax(dtype, sk, pad_batches):
    torch.manual_seed(0)
    kernel = ScaledMaskedSoftmaxLoader().load()
    b, heads, sq = 2, 4, 16
    x = torch.randn(b, heads, sq, sk, device="cuda").mul(4).to(dtype)
    # 1 marks a masked position; every row keeps at least its first key.
    mask = torch.rand(pad_batches, 1, sq, sk, device="cuda") < 0.3
    mask[..., 0] = False
    probs = kernel.forward(x, mask.to(torch.uint8), SCALE)
    ref = softmax_reference(x, SCALE, mask)
    torch.testing.assert_close(probs.float(), ref, rtol=1e-2, atol=1e-3)

    grad = torch.randn_like(probs)
    # The backward kernel writes its result into the gradient tensor it is given.
    dx = kernel.backward(grad.clone(), probs, SCALE)
    torch.testing.assert_close(dx.float(), softmax_backward_reference(grad, probs, SCALE), rtol=2e-2, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seq", KEY_LENGTHS)
def test_scaled_upper_triang_masked_softmax(dtype, seq):
    torch.manual_seed(0)
    kernel = ScaledUpperTriangleMaskedSoftmaxLoader().load()
    attn_batches = 16
    x = torch.randn(attn_batches, seq, seq, device="cuda").mul(4).to(dtype)
    probs = kernel.forward(x, SCALE)
    causal = torch.ones(seq, seq, dtype=torch.bool, device="cuda").triu(1)
    ref = torch.softmax((x.float() * SCALE).masked_fill(causal, float("-inf")), dim=-1)
    torch.testing.assert_close(probs.float(), ref, rtol=1e-2, atol=1e-3)

    grad = torch.randn_like(probs)
    # The backward kernel writes its result into the gradient tensor it is given.
    dx = kernel.backward(grad.clone(), probs, SCALE)
    torch.testing.assert_close(dx.float(), softmax_backward_reference(grad, probs, SCALE), rtol=2e-2, atol=2e-3)
