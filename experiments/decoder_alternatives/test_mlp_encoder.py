"""Unit tests for mlp_encoder.py: head shapes, the restandardize scale check, the O(1) size of the
three losses at random init, and that info_nce_loss is imported from decoder_training_utils
and not copied. No data is needed except for the infer_dim_low() test, which skips itself with a
message if data/targets/pca_zlow/ is missing.

Usage (from the project folder that holds data/):
    python experiments/decoder_alternatives/test_mlp_encoder.py
"""
import numpy as np
import torch

import decoder_training_utils
from mlp_encoder import (DIM_SEM, DIM_STRUCT, PCA_ZLOW_DIR, MLPEncoder, encoder_loss,
                          factor_l2_loss, infer_dim_low, info_nce_loss, sem_loss)
from trisect.ridge import restandardize_from_pred

torch.manual_seed(0)
np.random.seed(0)

DIM_LOW_SMALL = 64  # kept small in tests that don't need the real PCA basis, for speed


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name}: FAILED {detail}")
    print(f"  PASS {name}")


def test_output_shapes_per_head():
    model = MLPEncoder(subjects=(1,), dim_low=DIM_LOW_SMALL)
    from prepare_nsd_imagery_icnn import NUM_VOXELS  # noqa: E402 (same sys.path insert as the module)
    v = NUM_VOXELS[1]
    x = torch.randn(5, v)
    out = model(x, subj=1)
    check("sem shape", tuple(out["sem"].shape) == (5, DIM_SEM), out["sem"].shape)
    check("struct shape", tuple(out["struct"].shape) == (5, DIM_STRUCT), out["struct"].shape)
    check("low shape", tuple(out["low"].shape) == (5, DIM_LOW_SMALL), out["low"].shape)
    check("output finite", all(torch.isfinite(t).all() for t in out.values()))


def test_multisubject_adapters_independent_shared_trunk():
    model = MLPEncoder(subjects=(1, 2), dim_low=DIM_LOW_SMALL)
    check("adapter keys", set(model.adapters.keys()) == {"1", "2"}, list(model.adapters.keys()))
    from prepare_nsd_imagery_icnn import NUM_VOXELS
    for subj in (1, 2):
        x = torch.randn(3, NUM_VOXELS[subj])
        out = model(x, subj=subj)
        check(f"subj{subj} sem shape", tuple(out["sem"].shape) == (3, DIM_SEM))
        check(f"subj{subj} struct shape", tuple(out["struct"].shape) == (3, DIM_STRUCT))
        check(f"subj{subj} low shape", tuple(out["low"].shape) == (3, DIM_LOW_SMALL))


def test_restandardized_output_matches_target_scale():
    """A freshly initialized head does not output on the scale of the targets. Feed its output
    through trisect.ridge.restandardize_from_pred (reused, not reimplemented) against a synthetic
    training target with a deliberately different scale per dimension, and check that the result
    lands on that scale."""
    model = MLPEncoder(subjects=(1,), dim_low=DIM_LOW_SMALL)
    from prepare_nsd_imagery_icnn import NUM_VOXELS
    x = torch.randn(64, NUM_VOXELS[1])
    with torch.no_grad():
        pred = model(x, subj=1)

    rng = np.random.default_rng(0)
    for factor, dim in (("sem", DIM_SEM), ("struct", DIM_STRUCT), ("low", DIM_LOW_SMALL)):
        pred_np = pred[factor].numpy()
        raw_std = pred_np.std(axis=0)
        # synthetic training target on a deliberately different per-dimension scale, whitened-like
        # mean 0 but std varying 1..5 by dimension; nothing like a fresh nn.Linear's output scale
        train_target = rng.normal(loc=0.0, scale=np.linspace(1.0, 5.0, dim), size=(500, dim))
        restd = restandardize_from_pred(pred_np, train_target)
        restd_std = restd.std(axis=0)
        tgt_std = train_target.std(axis=0)
        check(f"{factor}: restandardized std matches target std",
              np.allclose(restd_std, tgt_std, rtol=1e-3, atol=1e-6),
              f"max abs diff {np.max(np.abs(restd_std - tgt_std)):.6f}")
        check(f"{factor}: raw output std was NOT already on target scale (restandardize is doing "
              f"real work, not a no-op)",
              not np.allclose(raw_std, tgt_std, rtol=0.2),
              f"raw_std range [{raw_std.min():.4f},{raw_std.max():.4f}] vs "
              f"tgt_std range [{tgt_std.min():.4f},{tgt_std.max():.4f}]")


def test_losses_are_o1_at_random_init():
    """Sanity on scale: with unit-variance targets (the scale of the whitened PCA bases), none of
    the three losses should be far from O(1) at random init. This catches a broadcasting or
    reduction bug (for example, summing over dimension instead of averaging) before a real
    training run does."""
    model = MLPEncoder(subjects=(1,), dim_low=DIM_LOW_SMALL)
    from prepare_nsd_imagery_icnn import NUM_VOXELS
    x = torch.randn(32, NUM_VOXELS[1])
    pred = model(x, subj=1)
    target = {
        "sem": torch.randn(32, DIM_SEM),
        "struct": torch.randn(32, DIM_STRUCT),
        "low": torch.randn(32, DIM_LOW_SMALL),
    }
    losses = encoder_loss(pred, target)
    check("sem loss in [0,2] (cosine bound)", 0.0 <= losses["sem"].item() <= 2.0, losses["sem"])
    check("struct loss O(1)", losses["struct"].item() < 20.0, losses["struct"])
    check("low loss O(1)", losses["low"].item() < 20.0, losses["low"])
    check("total finite", torch.isfinite(losses["total"]))
    check("ctr is nan when delta_ctr=0 (default, disabled)", torch.isnan(losses["ctr"]))


def test_sem_loss_zero_for_identical_input():
    x = torch.randn(4, 10)
    check("sem_loss(x,x) == 0", abs(sem_loss(x, x).item()) < 1e-5, sem_loss(x, x))


def test_factor_l2_loss_matches_mse():
    pred = torch.randn(6, 20)
    target = torch.randn(6, 20)
    expected = ((pred - target) ** 2).mean().item()
    check("factor_l2_loss matches mean squared error",
          abs(factor_l2_loss(pred, target).item() - expected) < 1e-5)


def test_ctr_disabled_below_min_batch_from_decoder_training_utils():
    b = decoder_training_utils.MIN_BATCH_FOR_CTR - 1
    pred = {"sem": torch.randn(b, DIM_SEM), "struct": torch.randn(b, DIM_STRUCT),
            "low": torch.randn(b, DIM_LOW_SMALL)}
    target = {k: torch.randn_like(v) for k, v in pred.items()}
    losses = encoder_loss(pred, target, delta_ctr=1.0)
    check(f"ctr disabled below MIN_BATCH_FOR_CTR ({decoder_training_utils.MIN_BATCH_FOR_CTR})",
          torch.isnan(losses["ctr"]))
    check("total excludes ctr when disabled",
          abs(losses["total"].item() - (losses["sem"] + losses["struct"] + losses["low"]).item())
          < 1e-5)


def test_ctr_enabled_at_min_batch_uses_info_nce_loss():
    b = decoder_training_utils.MIN_BATCH_FOR_CTR
    pred = {"sem": torch.randn(b, DIM_SEM), "struct": torch.randn(b, DIM_STRUCT),
            "low": torch.randn(b, DIM_LOW_SMALL)}
    target = {k: torch.randn_like(v) for k, v in pred.items()}
    delta = 0.5
    losses = encoder_loss(pred, target, delta_ctr=delta)
    expected_ctr = info_nce_loss(pred["sem"], target["sem"], 0.07)
    check("ctr term matches info_nce_loss(pred_sem, target_sem, default temp)",
          torch.allclose(losses["ctr"], expected_ctr))
    expected_total = losses["sem"] + losses["struct"] + losses["low"] + delta * expected_ctr
    check("total includes delta_ctr * ctr", torch.allclose(losses["total"], expected_total))


def test_info_nce_loss_is_imported_not_copied():
    check("mlp_encoder.info_nce_loss IS decoder_training_utils.info_nce_loss (same object, not a copy)",
          info_nce_loss is decoder_training_utils.info_nce_loss)


def test_infer_dim_low_reads_pca_basis():
    from pathlib import Path
    if not (Path(PCA_ZLOW_DIR) / "pca_basis.npz").exists():
        print(f"  SKIP test_infer_dim_low_reads_pca_basis ({PCA_ZLOW_DIR}/pca_basis.npz missing, "
              f"run fit_pca_zlow.py first)")
        return
    d = infer_dim_low()
    check("infer_dim_low returns a positive int", isinstance(d, int) and d > 0, d)
    model = MLPEncoder(subjects=(1,))  # dim_low=None -> reads the real basis
    check("model.dims['low'] matches infer_dim_low()", model.dims["low"] == d)


def main():
    tests = [
        test_output_shapes_per_head,
        test_multisubject_adapters_independent_shared_trunk,
        test_restandardized_output_matches_target_scale,
        test_losses_are_o1_at_random_init,
        test_sem_loss_zero_for_identical_input,
        test_factor_l2_loss_matches_mse,
        test_ctr_disabled_below_min_batch_from_decoder_training_utils,
        test_ctr_enabled_at_min_batch_uses_info_nce_loss,
        test_info_nce_loss_is_imported_not_copied,
        test_infer_dim_low_reads_pca_basis,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"  FAIL {t.__name__}\n{e}")

    print()
    if failures:
        print(f"{len(failures)}/{len(tests)} tests FAILED")
        raise SystemExit(1)
    print(f"All {len(tests)} tests PASSED")


if __name__ == "__main__":
    main()
