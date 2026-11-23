import pytest
import torch

from colossalai.moe._operation import MoeCombine, MoeDispatch, moe_cumsum


def make_routing(s, e, c, k):
    # Top-k experts per token, capacity c per expert; tokens over capacity are dropped (mask 0).
    experts = torch.rand(s, e, device="cuda").argsort(dim=1)[:, :k].t().contiguous()
    mask = torch.zeros(k, s, dtype=torch.int32, device="cuda")
    dest = torch.zeros(k, s, dtype=torch.int32, device="cuda")
    used = torch.zeros(e, dtype=torch.int64, device="cuda")
    for i in range(k):
        onehot = torch.nn.functional.one_hot(experts[i], e)
        pos = (torch.cumsum(onehot, dim=0) - 1 + used)[torch.arange(s, device="cuda"), experts[i]]
        keep = pos < c
        mask[i] = keep.int()
        dest[i] = torch.where(keep, experts[i] * c + pos, torch.zeros_like(pos)).int()
        used += onehot.sum(dim=0)
    return mask, dest


def dispatch_reference(tokens, mask, dest, ec):
    out = tokens.new_zeros(ec, tokens.size(1))
    for i in range(mask.size(0)):
        keep = mask[i].bool()
        out = out.index_put((dest[i][keep].long(),), tokens[keep], accumulate=True)
    return out


def combine_reference(expert_tokens, logits, mask, dest, c):
    out = 0
    for i in range(mask.size(0)):
        idx = dest[i].long()
        weight = logits.gather(1, (idx // c)[:, None]) * mask[i][:, None]
        out = out + weight * expert_tokens[idx]
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("s", [4, 100, 1024, 2048, 3000, 4096])
def test_moe_cumsum(s):
    torch.manual_seed(0)
    inputs = (torch.rand(s, 8, device="cuda") < 0.3).int()
    if s <= 1024 or (s <= 2048 and s % 2 == 0) or s % 4 == 0:
        assert torch.equal(moe_cumsum(inputs, use_kernel=True), (torch.cumsum(inputs, dim=0) - 1).int())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("hidden", [128, 384, 768, 1536, 4096])
@pytest.mark.parametrize("k", [1, 2])
def test_moe_dispatch_combine(dtype, hidden, k):
    torch.manual_seed(0)
    s, e, c = 64, 4, 24
    ec = e * c
    mask, dest = make_routing(s, e, c, k)
    tokens = torch.randn(s, hidden, device="cuda", dtype=dtype, requires_grad=True)
    logits = torch.softmax(torch.randn(s, e, device="cuda"), dim=-1).requires_grad_()

    expert_in = MoeDispatch.apply(tokens, mask, dest, ec)
    expert_out = expert_in * 2 + 1
    out = MoeCombine.apply(expert_out, logits, mask, dest, ec)
    grad = torch.randn_like(out)
    out.backward(grad)

    ref_tokens = tokens.detach().float().requires_grad_()
    ref_logits = logits.detach().requires_grad_()
    ref_in = dispatch_reference(ref_tokens, mask, dest, ec)
    ref_out = combine_reference(ref_in * 2 + 1, ref_logits, mask, dest, c)
    ref_out.backward(grad.float())

    tol = dict(rtol=1e-5, atol=1e-5) if dtype == torch.float32 else dict(rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(expert_in.float(), ref_in, **tol)
    torch.testing.assert_close(out.float(), ref_out, **tol)
    torch.testing.assert_close(tokens.grad.float(), ref_tokens.grad, **tol)
    torch.testing.assert_close(logits.grad, ref_logits.grad, rtol=tol["rtol"], atol=tol["atol"] * hidden**0.5)
