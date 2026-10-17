"""Shared primitives for the TRISECT scripts.

Only functions that several scripts need live here (`l2_normalize`, `load_gt`, `load_pca_basis`,
`restandardize`, `layer_bounds`, `zscore_fmri`), so that a normalization recipe cannot drift between
copies. Script-specific code stays in its script, and the evaluation chain (`evaluation/`) is not
routed through this package.

Scripts import it as `from trisect.pca import load_pca_basis`. `fit_ridge.py` also exposes
`zscore_fmri`; sibling imports keep working.
"""
