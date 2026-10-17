# figures/

The scripts that draw figures of the thesis. They read reconstructions and metric files that already exist:
nothing is generated and no GPU is needed.

| File | Draws | Thesis |
|---|---|---|
| `make_shared1000_figure.py` | Reconstructions of 16 `shared1000` images by the five baselines and TRISECT, subject 1 | Figure 4.1 (a and b) |
| `make_imagery_figure.py` | Reconstructions of NSD-Imagery imagery trials by the six models, subject 1 | Figure 7.1 |
| `make_margin_figures.py` | The margins over measured chance, and the move from the vision to the imagery margin | Figure 4.4 of the script is Figure B.1 of Appendix B; Figure 4.3 is not in the text. It recomputes Table 7.2 and stops if a cell differs by more than 0.1 pp |

## Figure 4.1

```
python make_shared1000_figure.py
```

The script lists every file it reads. Before it draws, it checks that the ground truths exported by the models
are the same images (correlation of at least 0.99) and that the NSD identifiers of the iCNN files are those of
the TRISECT block set. The image files are named `fig4_2a_shared1000.png` and `fig4_2b_shared1000.png`, as when
the figure was numbered 4.2.

## Figure 7.1

The figure of the thesis was drawn with

```
python make_imagery_figure.py --stimuli 0,3,6,7,10,12,13,14 --trisect-samples 0,1,8,7,2,1,5,5
```

The rows are the stimuli 0 and 3 (simple), 6, 7 and 10 (complex), and 12, 13 and 14 (conceptual: the cue words
*stripes*, *zebra* and *mammal*). Brain-Diffuser, MindEye1, MindEye2, iCNN and Takagi et al. show their sample 0.
The TRISECT column shows, row by row, the samples 0, 1, 8, 7, 2, 1, 5 and 5 of its ten. Without the two options,
the script draws the stimuli 0, 3, 6, 7, 10, 12, 14, 16 and sample 0 everywhere.

The script prints the samples it used, and warns when they are not all sample 0: three of the baselines exist only
at sample 0; a choice of samples for TRISECT is therefore not comparable with their first draw.

The numbering of the stimuli is the one of NSD-Imagery: 0 to 5 simple, 6 to 11 complex, 12 to 17 conceptual. The
ground-truth files of the conceptual stimuli show their cue words: 12 *stripes*, 13 *zebra*, 14 *mammal*,
15 *yellow*, 16 *banana*, 17 *fruit*.

## Where the scripts look for their files

The scripts read their inputs from the layout of the project in which the figures were drawn, with the folder
that contains this repository as root. `make_margin_figures.py` reads the metric files from
`<root>/my_project/data/results/baselines/`, a copy of `results/baselines/` of this repository, and writes the
figures to `<root>/thesis/figures/`. The other two scripts read the reconstructions of the six models from the
folders listed in their path constants. The reconstruction images are not included in this repository: they are
produced by the generation scripts of `trisect/` and of `reproductions/`.
