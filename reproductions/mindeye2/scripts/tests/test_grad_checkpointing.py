#!/usr/bin/env python
"""Gradient equivalence with and without --grad_checkpointing.

Outputs: per parameter, ||g_ckpt - g_ref|| / ||g_ref|| (must be under 1e-4), plus a control row

Checkpointing is only a memory optimisation if the gradients are the ones the recipe prescribes, and a wrong
recomputation is silent: the run completes, the loss goes down, and the model is not the paper's. The three
components of the flag are tested on the real modules of models.py at reduced dimensions, in fp32, from a fixed
seed. The dropout inside BrainNetwork (p=.15 and .3) makes it a real test of the RNG handling of
torch.utils.checkpoint: use_reentrant=False replays the generator state. The control runs the reference twice: 0.

Usage (from src/, venv active):
    python tests/test_grad_checkpointing.py [--device cpu]
"""
import argparse
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required by use_deterministic_algorithms

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

TOL = 1e-4


def compare(ref, other):
    """max over parameters of ||g_other - g_ref|| / ||g_ref||, plus the worst offender"""
    worst, worst_name = 0.0, "-"
    missing = [n for n in ref if n not in other]
    assert not missing, f"parameters missing from the second run: {missing[:5]}"
    for name, g_ref in ref.items():
        g_oth = other[name]
        denom = g_ref.norm().item()
        num = (g_oth - g_ref).norm().item()
        rel = num / denom if denom > 0 else num
        if rel > worst:
            worst, worst_name = rel, name
    return worst, worst_name


def grads_of(module, run_fn, seed=0):
    module.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    loss = run_fn(module)
    loss.backward()
    return {n: p.grad.detach().clone() for n, p in module.named_parameters() if p.grad is not None}


def build_backbone(device):
    from models import BrainNetwork
    torch.manual_seed(1234)
    net = BrainNetwork(h=256, in_dim=256, out_dim=64 * 16, seq_len=1, n_blocks=4,
                       clip_size=64, blurry_recon=True, clip_scale=1).to(device).float()
    net.train()
    return net


def build_prior(device):
    from models import BrainDiffusionPrior, PriorNetwork
    torch.manual_seed(1234)
    net = PriorNetwork(dim=64, depth=6, dim_head=16, heads=4, causal=False,
                       num_tokens=16, learned_query_mode="pos_emb")
    prior = BrainDiffusionPrior(net=net, image_embed_dim=64, condition_on_text_encodings=False,
                                timesteps=100, cond_drop_prob=0.2, image_embed_scale=None)
    prior = prior.to(device).float()
    prior.train()
    return prior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tol", type=float, default=TOL)
    args = ap.parse_args()
    device = torch.device(args.device)

    # fp32 all the way, and no TF32: the point is to see a real disagreement if there is one,
    # not to have the reduced precision of the matmuls hide it (or fake it). Full determinism
    # too: without it, cuDNN's backward reduction order is nondeterministic and the control
    # row (reference vs itself) reports a spurious nonzero diff on near-zero-gradient params.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    print(f"device: {device} | tolerance: {args.tol:g}\n")

    cases = []

    # ---- backbone: the 4 mixer blocks, with dropout inside ----
    backbone = build_backbone(device)
    torch.manual_seed(7)
    x = torch.randn(6, 1, 256, device=device)

    def run_backbone(net):
        b, c, blur = net(x)
        # touch every head so every parameter gets a gradient
        return b.square().mean() + c.square().mean() + blur[0].square().mean() + blur[1].square().mean()

    backbone.ckpt_backbone = backbone.ckpt_blurry = False
    ref = grads_of(backbone, run_backbone)
    cases.append(("control (reference vs itself)", compare(ref, grads_of(backbone, run_backbone))))

    backbone.ckpt_backbone, backbone.ckpt_blurry = True, False
    cases.append(("--grad_checkpointing=backbone", compare(ref, grads_of(backbone, run_backbone))))

    backbone.ckpt_backbone, backbone.ckpt_blurry = False, True
    cases.append(("--grad_checkpointing=blurry", compare(ref, grads_of(backbone, run_backbone))))

    backbone.ckpt_backbone = backbone.ckpt_blurry = True
    cases.append(("backbone+blurry together", compare(ref, grads_of(backbone, run_backbone))))

    # ---- prior: the 6 transformer layers, the component that actually saves the memory ----
    prior = build_prior(device)
    torch.manual_seed(8)
    text_embed = torch.randn(6, 16, 64, device=device, requires_grad=True)
    image_embed = torch.randn(6, 16, 64, device=device)

    def run_prior(p):
        loss, _ = p(text_embed=text_embed, image_embed=image_embed)
        return loss

    prior.net.causal_transformer.grad_checkpointing = False
    ref_prior = grads_of(prior, run_prior)
    prior.net.causal_transformer.grad_checkpointing = True
    cases.append(("--grad_checkpointing=prior", compare(ref_prior, grads_of(prior, run_prior))))

    # ---- verdict ----
    print(f"{'case':<34}{'worst rel. diff':>18}   worst parameter")
    failed = False
    for name, (rel, param) in cases:
        flag = "" if rel <= args.tol else "   <-- FAIL"
        if rel > args.tol:
            failed = True
        print(f"{name:<34}{rel:>18.3e}   {param}{flag}")
    print()
    if failed:
        print(f"FAIL: at least one component changes the gradients by more than {args.tol:g}.")
        print("Do NOT train with that component checkpointed.")
        return 1
    print(f"PASS: every component reproduces the reference gradients within {args.tol:g}.")
    print("Gradient checkpointing is therefore a pure memory/time trade, not a deviation from")
    print("the authors' recipe, and does not need to be declared as one in the thesis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
