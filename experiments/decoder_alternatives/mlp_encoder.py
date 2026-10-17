"""MLPEncoder: a multi-subject nonlinear decoder from voxels to the three latent factors (Section 8.1.1).

Inputs:  voxel vectors (15724 / 14278 / 13039 / 12682 voxels for subjects 1 / 2 / 5 / 7)
Outputs: sem (256), struct (2048) and low (d_low) predictions, plus the losses for train_mlp_encoder.py

One linear adapter per subject, then a trunk and three heads shared by all subjects. The heads are
mean-only: no variance head, no product of experts, no KL term (with a single real expert, a product
of experts is only a shrinkage toward the prior). Trap: the raw output is on the model's own scale. Callers must pass it through
trisect.ridge.restandardize_from_pred() before comparing it with targets; this is not done in forward(),
because it needs the fit-split target moments that only train_mlp_encoder.py has.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent  # the folder that holds icnn/
sys.path.insert(0, str(_REPO_ROOT / "icnn" / "src"))
from prepare_nsd_imagery_icnn import NUM_VOXELS  # noqa: E402  (needs the sys.path insert above)

from trisect.pca import load_pca_basis  # noqa: E402
from decoder_training_utils import MIN_BATCH_FOR_CTR, info_nce_loss  # noqa: E402  (reused verbatim)

FACTORS = ("sem", "struct", "low")
DIM_SEM = 256      # img128 || txt128, same order as targets.h5 z_sem[:, 0/1, :]
DIM_STRUCT = 2048  # 8x8x32 flat, fixed by the shared P_struct basis (fit_pca_zstruct.py)
HIDDEN = 4096
N_TRUNK_LAYERS = 4
DROPOUT = 0.15
CTR_TEMP_DEFAULT = 0.07  # InfoNCE temperature

PCA_ZLOW_DIR = "data/targets/pca_zlow"


def infer_dim_low(pca_dir: str = PCA_ZLOW_DIR) -> int:
    """d_low read from the frozen PCA basis at runtime, never hardcoded: it was swept once and
    can change with the basis."""
    basis = load_pca_basis(pca_dir)
    return int(basis["components"].shape[0])


class TrunkBlock(nn.Module):
    """One of the 4 stacked trunk blocks: LayerNorm -> Linear(4096, 4096) -> GELU -> Dropout(0.15),
    wrapped in a residual. Same pre-LN residual shape as the attention blocks of adapter_model.py."""

    def __init__(self, dim: int = HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(F.gelu(self.linear(self.norm(x))))


class MLPEncoder(nn.Module):
    """Per-subject voxel adapter -> shared nonlinear trunk -> three mean-only heads
    (sem/struct/low). Multisubject: `subj` selects the adapter, everything after it is shared."""

    def __init__(self, subjects=(1, 2, 5, 7), dim_low: "int | None" = None,
                 hidden: int = HIDDEN, n_trunk_layers: int = N_TRUNK_LAYERS,
                 dropout: float = DROPOUT):
        super().__init__()
        if dim_low is None:
            dim_low = infer_dim_low()
        self.subjects = tuple(subjects)
        self.dims = {"sem": DIM_SEM, "struct": DIM_STRUCT, "low": dim_low}
        self.adapters = nn.ModuleDict({
            str(s): nn.Linear(NUM_VOXELS[s], hidden) for s in self.subjects
        })
        self.trunk = nn.Sequential(*[TrunkBlock(hidden, dropout) for _ in range(n_trunk_layers)])
        self.heads = nn.ModuleDict({
            "sem": nn.Linear(hidden, DIM_SEM),
            "struct": nn.Linear(hidden, DIM_STRUCT),
            "low": nn.Linear(hidden, dim_low),
        })

    def forward(self, voxels: torch.Tensor, subj: int) -> dict:
        """voxels: [B, V_subj]. Returns {factor: mu}, mean only (see the module docstring):
        callers restandardize before using this as a target-scale prediction."""
        h = self.adapters[str(subj)](voxels)
        h = self.trunk(h)
        return {f: self.heads[f](h) for f in FACTORS}


def sem_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity, averaged over the batch. z_sem is L2-normalized; cosine is the
    natural loss; there is no variance head that would justify a likelihood instead."""
    return (1 - F.cosine_similarity(pred, target, dim=-1)).mean()


def factor_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean per-dimension L2 for struct and low. The default MSE reduction of torch averages over
    batch and dimension. Both PCA bases are whitened; every target dimension has variance ~1
    and the three factor losses stay on a comparable scale without a weight to search."""
    return F.mse_loss(pred, target)


def encoder_loss(pred: dict, target: dict, delta_ctr: float = 0.0,
                  ctr_temp: float = CTR_TEMP_DEFAULT) -> dict:
    """Combined training loss: cosine on sem + mean-L2 on struct and low + an optional InfoNCE on
    sem only. The weight delta_ctr is a command-line argument of train_mlp_encoder.py, chosen on
    a held-out NSD-core split, not a default fixed here. InfoNCE needs at least MIN_BATCH_FOR_CTR
    distinct stimuli in the batch, or it switches itself off.

    Returns every term plus 'total'; callers and tests can inspect each piece and not only the
    sum."""
    losses = {
        "sem": sem_loss(pred["sem"], target["sem"]),
        "struct": factor_l2_loss(pred["struct"], target["struct"]),
        "low": factor_l2_loss(pred["low"], target["low"]),
    }
    total = losses["sem"] + losses["struct"] + losses["low"]
    batch_size = pred["sem"].shape[0]
    if delta_ctr > 0 and batch_size >= MIN_BATCH_FOR_CTR:
        losses["ctr"] = info_nce_loss(pred["sem"], target["sem"], ctr_temp)
        total = total + delta_ctr * losses["ctr"]
    else:
        losses["ctr"] = torch.tensor(float("nan"))
    losses["total"] = total
    return losses
