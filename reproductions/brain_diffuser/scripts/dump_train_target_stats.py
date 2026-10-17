"""Dump the training-set moments that the NSD-Imagery predictions need, one small file per subject.

Inputs:  data/processed_data/subjNN/nsd_train_fmriavg_nsdgeneral_subN.npy, data/extracted_features/subjNN/*
Outputs: data/regression_weights/subjNN/train_target_stats.npz (about 5 MB): fmri (after /300), vdvae,
         cliptext and clipvision mean/std

Why: `imagery_regression_predict.py` maps every prediction back onto the mean and std of the training targets.
Those moments live only inside arrays of several GB; without them the predictions are on the wrong scale and
the reconstructions fall to chance with no error. Run once, before any intermediate file is deleted.

Usage (from the folder that contains data/):
    python dump_train_target_stats.py
"""
import numpy as np, os
for sub in (1, 2, 5, 7):
    d = 'data/regression_weights/subj{:02d}'.format(sub)
    tr = np.load('data/processed_data/subj{:02d}/nsd_train_fmriavg_nsdgeneral_sub{}.npy'.format(sub, sub))
    tr = tr / 300.0                       # same order as *_regression.py: /300, then z-score
    out = dict(fmri_mean=tr.mean(0), fmri_std=tr.std(0, ddof=1))
    del tr
    vd = np.load('data/extracted_features/subj{:02d}/nsd_vdvae_features_31l.npz'.format(sub))['train_latents']
    out['vdvae_mean'], out['vdvae_std'] = vd.mean(0), vd.std(0); del vd
    for name in ('cliptext', 'clipvision'):
        f = np.load('data/extracted_features/subj{:02d}/nsd_{}_train.npy'.format(sub, name))
        out[name + '_mean'], out[name + '_std'] = f.mean(0), f.std(0); del f
    np.savez(os.path.join(d, 'train_target_stats.npz'), **out)
    print(sub, 'ok', {k: v.shape for k, v in out.items()})
