# Third-party notices

This repository holds the code written for the thesis, and, for the five reproduced models, only the
changes made to the original code (patches) and the files added to it. The original code is not copied,
with one exception (`trisect/vdvae_lib/`, below). Every licence below is MIT or BSD-3-Clause, except one
(Stable Diffusion v1, CreativeML Open RAIL-M), which governs one changed line.

The licence texts that have to travel with the code are in the folder where the code is used. The
licences of the other projects are linked here and are not copied.

| Project | Licence and copyright | How it is used here | Where |
|---|---|---|---|
| MindEye1, [`MedARC-AI/fMRI-reconstruction-NSD`](https://github.com/MedARC-AI/fMRI-reconstruction-NSD) | MIT, © 2022 MEDARC | Four patches and added scripts | `reproductions/mindeye1/` (`LICENSE.upstream`) |
| MindEye2, [`MedARC-AI/MindEyeV2`](https://github.com/MedARC-AI/MindEyeV2) | MIT, © 2023 MedARC | Two patches and added scripts | `reproductions/mindeye2/` (`LICENSE.upstream`) |
| MIRAGE, [`MedARC-AI/MIRAGE`](https://github.com/MedARC-AI/MIRAGE) (formerly `MindEye_Imagery`) | MIT, © 2023 MedARC | Transcription of its evaluation notebook, the evaluation code of the benchmark | `evaluation/` (the notice and the commit are in `evaluation/README.md`) |
| Brain-Diffuser, [`ozcelikfu/brain-diffuser`](https://github.com/ozcelikfu/brain-diffuser) | MIT, © 2023 Furkan Ozcelik | Four patches and added scripts | `reproductions/brain_diffuser/` (`LICENSE.upstream`) |
| Takagi and Nishimoto, [`yu-takagi/StableDiffusionReconstruction`](https://github.com/yu-takagi/StableDiffusionReconstruction) | MIT, © 2023 Yu Takagi | Nine patches and added scripts | `reproductions/takagi/` (`LICENSE.upstream`) |
| `KamitaniLab/feature-decoding`, [repository](https://github.com/KamitaniLab/feature-decoding) | MIT, © 2024 Kamitani Lab | Two configuration patches; scripts adapted from its examples | `reproductions/icnn/` (`LICENSE.upstream`) |
| VDVAE, [`openai/vdvae`](https://github.com/openai/vdvae) | MIT, © 2020 OpenAI | **Seven files copied** (`vae.py` is modified, as `vdvae_lib/README.md` says) | `trisect/vdvae_lib/` (`LICENSE.md`) |
| Versatile Diffusion, [`SHI-Labs/Versatile-Diffusion`](https://github.com/SHI-Labs/Versatile-Diffusion) | MIT, © 2022 SHI Labs | Not included. It is a dependency of Brain-Diffuser and of the method of Chapter 5, and comes with the Brain-Diffuser repository (its `versatile_diffusion/` folder) | Cited in `reproductions/brain_diffuser/README.md` |
| `bdpy`, [`KamitaniLab/bdpy`](https://github.com/KamitaniLab/bdpy) | MIT | Not included. Installed with pip | `reproductions/icnn/` |
| Stable Diffusion v2, [`Stability-AI/stablediffusion`](https://github.com/Stability-AI/stablediffusion) | MIT, © 2022 Stability AI | One patch of `ddpm.py` | `reproductions/takagi/patches/` (`LICENSE.stable-diffusion-v2.MIT`) |
| **Stable Diffusion v1**, [`CompVis/stable-diffusion`](https://github.com/CompVis/stable-diffusion) | **CreativeML Open RAIL-M** | One patch of `ddpm.py`, **one changed line**. No weights and no model are included. The use restrictions of the licence apply to the code the patch touches | `reproductions/takagi/patches/` (`LICENSE.stable-diffusion-v1.RAIL-M`) |
| BLIP, [`salesforce/BLIP`](https://github.com/salesforce/BLIP) | BSD-3-Clause, © 2022 Salesforce | Three patches | `reproductions/takagi/patches/` (`LICENSE.BLIP.BSD-3-Clause`) |
| `KamitaniLab/spurious_reconstruction` (Shirakawa et al.) | **No licence** (all rights reserved) | **Nothing is copied.** The thesis describes what its code does; no file here derives from it | `reproductions/icnn/README.md` ("What is not here") |
| `KamitaniLab/DeepImageReconstruction` | No licence | **Nothing is copied.** It is the original Python 2.7 and Caffe implementation, replaced by the port above | `reproductions/icnn/README.md` |

Licences were checked on 2026-10-02, on the `LICENSE` file of each repository and on the licence that
GitHub reports for it.

The data of the Natural Scenes Dataset and of NSD-Imagery, the model weights and the reconstructed images
are not part of this repository. `results/` holds only metrics computed on the reconstructions.
