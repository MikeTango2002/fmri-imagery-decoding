"""Brain captions for NSD-Imagery: decode_captions.py, one condition at a time.

Inputs:  the decoded BLIP scores of decoded_imagery/, per condition
Outputs: captions_brain_<cond>.csv

Copy of decode_captions.py with three changes: the score source, 6 stimuli instead of the hardcoded 982, and the
output name (otherwise the 5 conditions overwrite each other). Paths are anchored on __file__; it runs from any
cwd. generate_from_imageembeds keeps num_beams=3, max_length=20, min_length=5, repetition_penalty=1.5, the published
recipe; beam search is deterministic, and the 10 reconstructions of the protocol come from the diffusion noise.

Usage (venv: .venv_blip, Python 3.9, transformers 4.15):
    python decode_captions_imagery.py --subject subj01 --condition visioncomplex
"""
import torch
from models.blip import blip_decoder
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse, os

TAKAGI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

CONDITIONS = ["visionsimple", "visioncomplex",
              "imagerysimple", "imagerycomplex", "imageryconcepts"]


def generate_from_imageembeds(model, device, image_embeds, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
    if not sample:
        image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)

    image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(device)
    model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}

    prompt = [model.prompt]
    input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    input_ids[:,0] = model.tokenizer.bos_token_id
    input_ids = input_ids[:, :-1]

    if sample:
        #nucleus sampling
        outputs = model.text_decoder.generate(input_ids=input_ids,
                                              max_length=max_length,
                                              min_length=min_length,
                                              do_sample=True,
                                              top_p=top_p,
                                              num_return_sequences=1,
                                              eos_token_id=model.tokenizer.sep_token_id,
                                              pad_token_id=model.tokenizer.pad_token_id,
                                              repetition_penalty=1.1,
                                              **model_kwargs)
    else:
        #beam search
        outputs = model.text_decoder.generate(input_ids=input_ids,
                                              max_length=max_length,
                                              min_length=min_length,
                                              num_beams=num_beams,
                                              eos_token_id=model.tokenizer.sep_token_id,
                                              pad_token_id=model.tokenizer.pad_token_id,
                                              repetition_penalty=repetition_penalty,
                                              **model_kwargs)

    captions = []
    for output in outputs:
        caption = model.tokenizer.decode(output, skip_special_tokens=True)
        captions.append(caption[len(model.prompt):])
    return captions

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )
    parser.add_argument(
        "--condition",
        type=str,
        required=True,
        choices=CONDITIONS,
        help="NSD-Imagery condition, one of the 5 exported by prepare_nsd_imagery_takagi.py",
    )
    parser.add_argument(
        "--imagery_dir",
        type=str,
        default=os.path.join(TAKAGI_ROOT, 'decoded_imagery'),
        help="--out_dir of decode_imagery_features.py; also where the captions are written",
    )

    # Set parameters
    gpu = 0
    opt = parser.parse_args()
    subject = opt.subject
    condition = opt.condition
    norepeat = 1.5
    torch.cuda.set_device(gpu)
    image_size = 240
    savedir = os.path.join(opt.imagery_dir, subject, 'captions')
    os.makedirs(savedir, exist_ok=True)

    # Model Load
    model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth"
    model_local = os.path.join(TAKAGI_ROOT, 'cache', 'blip', 'model_base_capfilt_large.pth')
    if not os.path.exists(model_local):
        os.makedirs(os.path.dirname(model_local), exist_ok=True)
        import urllib.request
        print(f"Downloading BLIP checkpoint to {model_local} ...")
        urllib.request.urlretrieve(model_url, model_local)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model_decoder = blip_decoder(pretrained=model_local, image_size=image_size, vit="base")
    model_decoder.eval()
    model_decoder = model_decoder.to(device)

    # Load decoding results: the ridge prediction of the BLIP ViT embedding, decoded from
    # the imagery betas. No image ever enters here; the visual encoder is not called.
    roi = ['early', 'ventral', 'midventral', 'midlateral', 'lateral', 'parietal']
    scorepath = os.path.join(opt.imagery_dir, subject,
                             f'{subject}_{"_".join(roi)}_scores_blip_{condition}.npy')
    scores = np.load(scorepath)
    print(f'{scorepath}: {scores.shape}')

    # Caption generation
    captions_brain = []
    for imidx in tqdm(range(len(scores))):
        scores_test = torch.Tensor(scores[imidx,:].reshape(-1,768)).to(device).unsqueeze(0)
        print(scores_test.shape)
        caption = generate_from_imageembeds(model_decoder, device, scores_test,num_beams=3, max_length=20, min_length=5, repetition_penalty=norepeat)
        captions_brain.append(caption)
        print(f'{imidx:04}:  {caption}')

    df = pd.DataFrame(captions_brain)
    df.to_csv(f'{savedir}/captions_brain_{condition}.csv', sep='\t', header=False, index=False)

if __name__ == "__main__":
    main()
