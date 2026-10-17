"""The adapter: (z_sem, z_struct) -> Versatile Diffusion's native cross-attention context.

Learned queries cross-attend to a small token sequence built from (z_sem, z_struct), then pass
through self-attention blocks (2x cross-attention, 2x self-attention). Two query sets (257 image, 77
text) share one context encoder, because VD's context is CLIP ViT-L/14 native, (257,768) + (77,768),
not the 16x1024 of the method description. The ~2.2k bottleneck comes from what is estimated from
the brain (z_sem 2x128 + z_struct 2048), not from the generator's interface width.

It is trained by regression against the true VD context (`train_adapter.py`), with no brain data and
no diffusion loss through the frozen generator. The widths `Z_SEM_DIM` and `Z_STRUCT_CHANNELS` are
constructor arguments; the width sweeps (`APsiMinimal(sem_dim=256)`, `struct_channels=64`) need
no edit to this file.
"""
import torch
import torch.nn as nn

CONTEXT_DIM = 768
Z_SEM_DIM = 128
Z_SEM_STREAMS = 2
Z_STRUCT_GRID = 8
Z_STRUCT_TOKENS = Z_STRUCT_GRID * Z_STRUCT_GRID  # 64
Z_STRUCT_CHANNELS = 32
IMAGE_CONTEXT_TOKENS = 257
TEXT_CONTEXT_TOKENS = 77


class ContextEncoder(nn.Module):
    """Projects (z_sem, z_struct) into a small token sequence the query decoders cross-attend to.

    2 semantic tokens (one per z_sem stream, img/txt) + 64 structural tokens (one per 8x8
    spatial position), each projected to CONTEXT_DIM and given a learned type/position
    embedding so the decoders can tell them apart; there is no other positional signal here.
    """

    def __init__(self, dim=CONTEXT_DIM, struct_channels=Z_STRUCT_CHANNELS, sem_dim=Z_SEM_DIM):
        super().__init__()
        self.sem_proj = nn.Linear(sem_dim, dim)
        self.sem_type_embed = nn.Parameter(torch.randn(Z_SEM_STREAMS, dim) * 0.02)
        self.struct_proj = nn.Linear(struct_channels, dim)
        self.struct_pos_embed = nn.Parameter(torch.randn(Z_STRUCT_TOKENS, dim) * 0.02)

    def forward(self, z_sem, z_struct):
        # z_sem: (B, 2, 128), z_struct: (B, 8, 8, 32)
        b = z_sem.shape[0]
        sem_tok = self.sem_proj(z_sem) + self.sem_type_embed.unsqueeze(0)  # (B,2,dim)
        struct_flat = z_struct.reshape(b, Z_STRUCT_TOKENS, -1)  # channels: whatever struct_proj expects
        struct_tok = self.struct_proj(struct_flat) + self.struct_pos_embed.unsqueeze(0)  # (B,64,dim)
        return torch.cat([sem_tok, struct_tok], dim=1)  # (B,66,dim)


class _CrossAttnBlock(nn.Module):
    """Pre-LN cross-attention: queries attend to a fixed context, plus an FFN."""

    def __init__(self, dim, heads):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, q, context):
        attn_out, _ = self.attn(self.norm_q(q), self.norm_kv(context), self.norm_kv(context), need_weights=False)
        q = q + attn_out
        q = q + self.mlp(self.norm_mlp(q))
        return q


class _SelfAttnBlock(nn.Module):
    """Pre-LN self-attention among the queries, plus an FFN."""

    def __init__(self, dim, heads):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, q):
        normed = self.norm_attn(q)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        q = q + attn_out
        q = q + self.mlp(self.norm_mlp(q))
        return q


class QueryDecoder(nn.Module):
    """Learned queries -> 2x cross-attention (to context) -> 2x self-attention, per SS5's spec."""

    def __init__(self, num_queries, dim=CONTEXT_DIM, heads=8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.cross1 = _CrossAttnBlock(dim, heads)
        self.cross2 = _CrossAttnBlock(dim, heads)
        self.self1 = _SelfAttnBlock(dim, heads)
        self.self2 = _SelfAttnBlock(dim, heads)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, context):
        b = context.shape[0]
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        q = self.cross1(q, context)
        q = self.cross2(q, context)
        q = self.self1(q)
        q = self.self2(q)
        return self.out_norm(q)


class APsiMinimal(nn.Module):
    """(z_sem, z_struct) -> (prompt_embeds (B,77,768), image_embeddings (B,257,768)).

    Both outputs are the UNCONDITIONAL-free (single, not CFG-doubled) predicted context --
    callers that need CFG must prepend the fixed negative half themselves
    (vd_context.encode_unconditional_context), same as the true-context path does
    internally before generate_from_context() is called.
    """

    def __init__(self, dim=CONTEXT_DIM, heads=8, struct_channels=Z_STRUCT_CHANNELS, sem_dim=Z_SEM_DIM):
        super().__init__()
        self.context_encoder = ContextEncoder(dim, struct_channels=struct_channels, sem_dim=sem_dim)
        self.text_decoder = QueryDecoder(TEXT_CONTEXT_TOKENS, dim, heads)
        self.image_decoder = QueryDecoder(IMAGE_CONTEXT_TOKENS, dim, heads)

    def forward(self, z_sem, z_struct):
        context = self.context_encoder(z_sem, z_struct)
        prompt_embeds = self.text_decoder(context)
        image_embeddings = self.image_decoder(context)
        return prompt_embeds, image_embeddings
