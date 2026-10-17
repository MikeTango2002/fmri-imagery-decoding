"""Extract the evaluation features of the NSD-Imagery 12-stimulus pool (Takagi paths).

Inputs:  decoded/image-text-imagery/<subj>/<cond>/samples/{stim:05}_{sample:03}.png; imagery_stimuli_18.pt sliced 0:12
Outputs: <feats_dir>/<net>_<layer>.npy for the six evaluation networks

Port of the Brain-Diffuser extractor (eval_extract_features.py); only where the images live changes, and the five
models are measured by the same code. The 10 samples per stimulus come from one run of
diffusion_decoding_imagery.py, not 10 seeds: the same aggregation axis. Ground-truth rows 0..5 (simple) and 6..11
(complex) follow GT_SLICE of the generation script. `--bridge` extracts the shared1000 pilot (100 of the 982 test
images) to validate the harness before any imagery number is read.

Usage (from codes/utils/):
    CUDA_VISIBLE_DEVICES=0 python eval_extract_features_imagery.py --gt
    CUDA_VISIBLE_DEVICES=0 python eval_extract_features_imagery.py --subject subj01 --modality vision --sample 0
"""

import argparse
import os

import numpy as np
import torch
import torchvision.models as tvmodels
import torchvision.transforms as transforms
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import clip

NET_LIST = [
    ('inceptionv3', 'avgpool'),
    ('clip', 'final'),
    ('alexnet', 2),
    ('alexnet', 5),
    ('efficientnet', 'avgpool'),
    ('swav', 'avgpool'),
]


def pool_paths(recon_dir, subject, modality, sample):
    """The 12 reconstructions of one modality, simple first, in stimulus order."""
    paths = []
    for stype in ('simple', 'complex'):
        d = os.path.join(recon_dir, subject, '{}{}'.format(modality, stype), 'samples')
        paths += [os.path.join(d, '{:05}_{:03}.png'.format(i, sample)) for i in range(6)]
    return paths


def bridge_paths(bridge_dir, sample):
    """The shared1000 pilot: every index that has an `_org.png`, in index order.

    The pilot is 100 indices out of 982 and `--skip_existing` may have grown it; the
    count is read from disk instead of being hardcoded.
    """
    orgs = sorted(f for f in os.listdir(bridge_dir) if f.endswith('_org.png'))
    idxs = [int(f.split('_')[0]) for f in orgs]
    gt = [os.path.join(bridge_dir, '{:05}_org.png'.format(i)) for i in idxs]
    rec = [os.path.join(bridge_dir, '{:05}_{:03}.png'.format(i, sample)) for i in idxs]
    return gt, rec


def gt_images(stimuli_path):
    """The 12 ground-truth stimuli as PIL images, from the (18, 3, 425, 425) tensor.

    Same conversion as diffusion_decoding_imagery.py; these are bit-identical to the
    `_org.png` files it writes next to the samples.
    """
    stimuli = torch.load(stimuli_path)[0:12]
    assert tuple(stimuli.shape) == (12, 3, 425, 425), stimuli.shape
    out = []
    for t in stimuli:
        arr = (t.numpy() * 255.).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
        out.append(Image.fromarray(arr))
    return out


class ImageList(Dataset):
    """Upstream's batch_generator_external_images, accepting PIL images as well."""

    def __init__(self, images, net_name='clip'):
        self.images = images
        self.net_name = net_name
        if self.net_name == 'clip':
            self.normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711])
        else:
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])

    def __getitem__(self, idx):
        img = self.images[idx]
        if isinstance(img, str):
            img = Image.open(img)
        img = img.convert('RGB')
        img = T.functional.resize(img, (224, 224))
        img = T.functional.to_tensor(img).float()
        img = self.normalize(img)
        return img

    def __len__(self):
        return len(self.images)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--subject', type=str, default='subj01',
                        help='subj01 or subj02 or subj05 or subj07')
    parser.add_argument('--modality', type=str, default='imagery',
                        choices=['vision', 'imagery'])
    parser.add_argument('--sample', type=int, default=0,
                        help='which of the 10 samples of that stimulus (0..9)')
    parser.add_argument('--gt', action='store_true',
                        help='extract the ground truth of the 12-stimulus pool instead')
    parser.add_argument('--bridge', action='store_true',
                        help='shared1000 pilot instead of imagery')
    parser.add_argument('--method', type=str, default='text',
                        help='--bridge only: which decoded/image-<method>/ to read')
    parser.add_argument('--recon_dir', type=str, default='../../decoded/image-text-imagery',
                        help='--out_dir of diffusion_decoding_imagery.py')
    parser.add_argument('--stimuli_path', type=str,
                        default='../../nsd/nsddata_stimuli/stimuli/imagery_stimuli_18.pt')
    parser.add_argument('--feat_dir', type=str, default='../../eval_features_imagery')
    parser.add_argument('--clip_download_root', type=str, default='../../cache/clip')
    opt = parser.parse_args()

    if not os.path.isfile('make_subjmri.py'):
        raise SystemExit('run this from codes/utils/')

    images = None
    if opt.bridge:
        bridge_dir = os.path.join('../../decoded', 'image-{}'.format(opt.method),
                                  opt.subject, 'samples')
        gt, rec = bridge_paths(bridge_dir, opt.sample)
        if opt.gt:
            images, feats_dir = gt, os.path.join(
                opt.feat_dir, 'bridge', 'gt_{}_{}'.format(opt.subject, opt.method))
        else:
            images, feats_dir = rec, os.path.join(
                opt.feat_dir, 'bridge',
                '{}_{}_s{}'.format(opt.subject, opt.method, opt.sample))
        print('bridge pool: {} images from {}'.format(len(images), bridge_dir))
    elif opt.gt:
        images = gt_images(opt.stimuli_path)
        feats_dir = os.path.join(opt.feat_dir, 'gt12')
    else:
        assert opt.subject in ['subj01', 'subj02', 'subj05', 'subj07']
        images = pool_paths(opt.recon_dir, opt.subject, opt.modality, opt.sample)
        feats_dir = os.path.join(
            opt.feat_dir, opt.subject,
            '{}12_s{}'.format(opt.modality, opt.sample))

    missing = [p for p in images if isinstance(p, str) and not os.path.isfile(p)]
    if missing:
        raise SystemExit('missing {} image(s), first: {}'.format(len(missing), missing[0]))
    if not opt.bridge and len(images) != 12:
        raise SystemExit('pool has {} images, expected 12'.format(len(images)))

    os.makedirs(feats_dir, exist_ok=True)

    feat_list = []

    def fn(module, inputs, outputs):
        feat_list.append(outputs.cpu().numpy())

    device = 0  # single-GPU: select the physical GPU with CUDA_VISIBLE_DEVICES
    batchsize = 12

    for (net_name, layer) in NET_LIST:
        del feat_list[:]
        print(net_name, layer)
        loader = DataLoader(ImageList(images, net_name=net_name), batchsize, shuffle=False)

        if net_name == 'inceptionv3':
            net = tvmodels.inception_v3(pretrained=True)
            net.avgpool.register_forward_hook(fn)
        elif net_name == 'alexnet':
            net = tvmodels.alexnet(pretrained=True)
            if layer == 2:
                net.features[4].register_forward_hook(fn)
            elif layer == 5:
                net.features[11].register_forward_hook(fn)
        elif net_name == 'clip':
            model, _ = clip.load('ViT-L/14', device='cuda:{}'.format(device),
                                 download_root=opt.clip_download_root)
            net = model.visual
            net = net.to(torch.float32)
            net.register_forward_hook(fn)
        elif net_name == 'efficientnet':
            try:
                # what upstream writes; torchvision 0.15 rejects weights=True in some builds
                net = tvmodels.efficientnet_b1(weights=True)
            except (TypeError, ValueError):
                net = tvmodels.efficientnet_b1(
                    weights=tvmodels.EfficientNet_B1_Weights.IMAGENET1K_V1)
            net.avgpool.register_forward_hook(fn)
        elif net_name == 'swav':
            net = torch.hub.load('facebookresearch/swav:main', 'resnet50')
            net.avgpool.register_forward_hook(fn)
        net.eval()
        net.cuda(device)

        with torch.no_grad():
            for i, x in enumerate(loader):
                print(i * batchsize)
                _ = net(x.cuda(device))

        feats = np.concatenate(feat_list)
        file_name = '{}/{}_{}.npy'.format(feats_dir, net_name, layer)
        np.save(file_name, feats)
        print('saved {} {}'.format(file_name, feats.shape))


if __name__ == '__main__':
    main()
