"""Helpers to drive Versatile Diffusion's dual-guided pipeline from raw context tensors.

`VersatileDiffusionDualGuidedPipeline.__call__` only takes a text and a PIL image, with no hook for a
precomputed (77,768) text or (257,768) image context. Two uses need that hook: the TRUE context
(`encode_true_context()`, the reference) and the adapter's PREDICTED context; `generate_from_context()`
takes either. `encode_true_context()` calls the pipeline's own `_encode_text_prompt` and
`_encode_image_prompt`, which define what "true context" means. `generate_from_context()` is steps 4-9
of `__call__` (diffusers 0.39.0) with the encoding replaced; `smoke_vd_context_roundtrip.py` checks it.

VDVAE init (Brain-Diffuser's `diff_str`): `init_image` (the 64x64 VDVAE output) plus `strength` gives
img2img-style partial denoising, as z_low needs. The default stays pure noise. `strength` is the fraction
of the schedule actually run (`t_enc` in Brain-Diffuser, `num_actual_steps` here): 1.0 equals the default
path, near 0 returns the init. The mechanism is re-expressed with this pipeline's primitives, because
Brain-Diffuser drives a hand-rolled `DDIMSampler_VD`.
"""
import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, VersatileDiffusionDualGuidedPipeline
from diffusers.models.transformers.dual_transformer_2d import DualTransformer2DModel
from diffusers.pipelines.deprecated.versatile_diffusion.modeling_text_unet import UNetFlatConditionModel
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPImageProcessor, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

MODEL_ID = "shi-labs/versatile-diffusion"

# Version-skew patch: CrossAttnDownBlock2D and its
# up/mid-block siblings in diffusers 0.39.0 unconditionally pass `encoder_attention_mask=...` to
# whichever attention module they hold, but `DualTransformer2DModel`, the class VD's
# `image_unet` uses for cross-attention (it mixes text and image conditioning), has not
# been touched since before that kwarg was introduced and its forward() does not accept it,
# raising `TypeError: DualTransformer2DModel.forward() got an unexpected keyword argument
# 'encoder_attention_mask'` on the very first denoising step. The value is always None on this
# path (no encoder_attention_mask is ever passed into image_unet); dropping it is safe; this
# is a diffusers-internal wiring gap for an unmaintained deprecated class, not a modeling choice.
_original_dual_transformer_forward = DualTransformer2DModel.forward


def _patched_dual_transformer_forward(
    self,
    hidden_states,
    encoder_hidden_states,
    timestep=None,
    attention_mask=None,
    cross_attention_kwargs=None,
    encoder_attention_mask=None,
    return_dict: bool = True,
):
    return _original_dual_transformer_forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep=timestep,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
        return_dict=return_dict,
    )


DualTransformer2DModel.forward = _patched_dual_transformer_forward


def load_pipeline(dtype=torch.float16, device="cuda"):
    """Load each component by hand instead of `VersatileDiffusionDualGuidedPipeline.from_pretrained`.

    The one-shot pipeline loader fails on this checkpoint: its `model_index.json` declares `text_unet: ["versatile_diffusion",
    "UNetFlatConditionModel"]`, pointing at a `versatile_diffusion.py` custom-code module that
    predates `UNetFlatConditionModel` being folded into diffusers proper; the module no longer
    exists in the repo and diffusers 0.39.0 no longer resolves it automatically. The weights
    themselves are fine (verified via `HfApi().list_repo_files`: `text_unet/config.json` +
    `text_unet/diffusion_pytorch_model.bin` are both present); only the pipeline-level `model_index.json`
    routing is stale. Loading every subfolder directly with its real diffusers/transformers class,
    then constructing the pipeline object ourselves, sidesteps that routing entirely.
    """
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    image_feature_extractor = CLIPImageProcessor.from_pretrained(MODEL_ID, subfolder="image_feature_extractor")
    text_encoder = CLIPTextModelWithProjection.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=dtype)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, subfolder="image_encoder", torch_dtype=dtype)
    image_unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="image_unet", torch_dtype=dtype)
    text_unet = UNetFlatConditionModel.from_pretrained(MODEL_ID, subfolder="text_unet", torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=dtype)
    scheduler = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    pipe = VersatileDiffusionDualGuidedPipeline(
        tokenizer=tokenizer,
        image_feature_extractor=image_feature_extractor,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        image_unet=image_unet,
        text_unet=text_unet,
        vae=vae,
        scheduler=scheduler,
    )
    pipe.remove_unused_weights()
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


@torch.no_grad()
def encode_true_context(pipe, pil_image, prompt, device, do_cfg=True):
    """Return (prompt_embeds, image_embeddings) exactly as pipe.__call__ would compute them.

    Shapes (before any CFG doubling): prompt_embeds (1,77,768), image_embeddings (1,257,768).
    With do_cfg=True (default, matches __call__'s guidance_scale > 1 path) both are doubled on
    the batch dim to (2,77,768) / (2,257,768), unconditional half first, ready to feed
    straight into generate_from_context() below without further concatenation.
    """
    prompt_embeds = pipe._encode_text_prompt([prompt], device, 1, do_cfg)
    image_embeddings = pipe._encode_image_prompt([pil_image], device, 1, do_cfg)
    return prompt_embeds, image_embeddings


@torch.no_grad()
def encode_unconditional_context(pipe, device):
    """Return VD's fixed negative/unconditional context: empty-string text, grey-image (0.5).

    `_encode_text_prompt`/`_encode_image_prompt` only expose this half bundled together with a
    positive half they compute themselves; there is no public hook to get just the negative
    half; this reimplements their negative branch verbatim (same tokenizer call, same
    `normalize_embeddings` closures, diffusers 0.39.0). Needed because the adapter
    (adapter_model.py) only predicts the POSITIVE half of the context; CFG-doubled generation still needs this fixed
    negative half prepended, exactly as generate_from_context() expects (batch dim 2, negative
    first): see encode_true_context()'s do_cfg=True output for the reference layout.

    Shapes: prompt_embeds (1,77,768), image_embeddings (1,257,768).
    """
    import numpy as np

    def normalize_text(encoder_output):
        embeds = pipe.text_encoder.text_projection(encoder_output.last_hidden_state)
        embeds_pooled = encoder_output.text_embeds
        return embeds / torch.norm(embeds_pooled.unsqueeze(1), dim=-1, keepdim=True)

    def normalize_image(encoder_output):
        embeds = pipe.image_encoder.vision_model.post_layernorm(encoder_output.last_hidden_state)
        embeds = pipe.image_encoder.visual_projection(embeds)
        embeds_pooled = embeds[:, 0:1]
        return embeds / torch.norm(embeds_pooled, dim=-1, keepdim=True)

    uncond_input = pipe.tokenizer(
        [""], padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    attention_mask = None
    if hasattr(pipe.text_encoder.config, "use_attention_mask") and pipe.text_encoder.config.use_attention_mask:
        attention_mask = uncond_input.attention_mask.to(device)
    neg_prompt_embeds = pipe.text_encoder(uncond_input.input_ids.to(device), attention_mask=attention_mask)
    neg_prompt_embeds = normalize_text(neg_prompt_embeds)

    uncond_image_input = pipe.image_feature_extractor(images=[np.zeros((512, 512, 3)) + 0.5], return_tensors="pt")
    pixel_values = uncond_image_input.pixel_values.to(device).to(pipe.image_encoder.dtype)
    neg_image_embeddings = pipe.image_encoder(pixel_values)
    neg_image_embeddings = normalize_image(neg_image_embeddings)

    return neg_prompt_embeds, neg_image_embeddings


@torch.no_grad()
def encode_init_image(pipe, image, height, width, device, generator=None):
    """Encode a pixel-space init image into VD's VAE latent space, scaled like fresh noise.

    `image` is whatever `pipe.image_processor.preprocess` accepts (PIL image or tensor), for
    the z_low path that is the VDVAE decoder's 64x64 output, upsampled to (height, width) by
    `preprocess` itself. Mirrors brain_diffuser's `net.autokl_encode(zin)`: same VAE, same
    [-1,1] normalization, but through diffusers' own preprocessing rather than the hand-rolled
    `regularize_image` in `versatilediffusion_reconstruct_images*.py`; it inherits whatever
    resize/interpolation this pipeline's image_processor actually expects.
    """
    image_tensor = pipe.image_processor.preprocess(image, height=height, width=width)
    image_tensor = image_tensor.to(device=device, dtype=pipe.vae.dtype)
    init_latents = pipe.vae.encode(image_tensor).latent_dist.sample(generator)
    return init_latents * pipe.vae.config.scaling_factor


def _img2img_timesteps(pipe, num_inference_steps, strength, device):
    """`strength` = fraction of the full schedule actually run (brain_diffuser's `diff_str`:
    `t_enc = int(strength * ddim_steps)`). strength=1.0 -> the full schedule, starting from pure
    noise, identical to the num_actual_steps == num_inference_steps case below.
    """
    assert 0.0 < strength <= 1.0, f"strength must be in (0, 1], got {strength}"
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    num_actual_steps = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - num_actual_steps, 0)
    return pipe.scheduler.timesteps[t_start:]


@torch.no_grad()
def generate_from_context(
    pipe,
    prompt_embeds,
    image_embeddings,
    text_to_image_strength=0.5,
    height=None,
    width=None,
    num_inference_steps=50,
    guidance_scale=7.5,
    generator=None,
    latents=None,
    init_image=None,
    strength=1.0,
    output_type="pil",
):
    """Run VD's denoising loop from precomputed context tensors.

    prompt_embeds / image_embeddings must already be CFG-doubled (batch dim = 2) if
    guidance_scale > 1, matching what encode_true_context(..., do_cfg=True) returns, or a
    future A_psi's CFG-doubled output. This is steps 4-9 of
    VersatileDiffusionDualGuidedPipeline.__call__ (diffusers 0.39.0), with step 3 (encoding)
    removed because the caller supplies its result directly.

    Two, mutually exclusive, ways to seed the denoising loop:
      - `latents=` (existing behaviour): a raw already-noised latent, or None for fresh Gaussian
        noise: passed straight through to `pipe.prepare_latents`, full `num_inference_steps`
        schedule. `strength` is ignored on this path.
      - `init_image=` (VDVAE init, see module docstring): a pixel-space image encoded through
        VD's VAE and forward-diffused to the noise level `strength` selects, then denoised for
        only the remaining `strength * num_inference_steps` steps. Do not pass `latents` too.
    """
    device = pipe._execution_device
    height = height or pipe.image_unet.config.sample_size * pipe.vae_scale_factor
    width = width or pipe.image_unet.config.sample_size * pipe.vae_scale_factor
    do_classifier_free_guidance = guidance_scale > 1.0

    dual_prompt_embeddings = torch.cat([prompt_embeds, image_embeddings], dim=1)
    prompt_types = ("text", "image")

    batch_size = 1
    if init_image is not None:
        assert latents is None, "pass either latents= or init_image=, not both"
        timesteps = _img2img_timesteps(pipe, num_inference_steps, strength, device)
        init_latents = encode_init_image(pipe, init_image, height, width, device, generator)
        noise = randn_tensor(init_latents.shape, generator=generator, device=device, dtype=init_latents.dtype)
        latents = pipe.scheduler.add_noise(init_latents, noise, timesteps[:1])
    else:
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps
        num_channels_latents = pipe.image_unet.config.in_channels
        latents = pipe.prepare_latents(
            batch_size, num_channels_latents, height, width, dual_prompt_embeddings.dtype, device, generator, latents
        )

    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta=0.0)
    pipe.set_transformer_params(text_to_image_strength, prompt_types)

    for t in pipe.progress_bar(timesteps):
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        noise_pred = pipe.image_unet(latent_model_input, t, encoder_hidden_states=dual_prompt_embeddings).sample

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

    if output_type == "latent":
        image = latents
    else:
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type=output_type)
