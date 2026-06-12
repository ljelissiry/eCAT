# Fe/PhOH CV Example Dataset

This folder contains a small text-export cyclic voltammetry dataset used by the
community-facing eCAT tutorial notebooks. The files were selected from the
Fe/phenol CV example set because they are small, load reliably, and exercise
common eCAT workflows:

- loading a folder of instrument text exports,
- plotting one CV and multiple overlaid CVs,
- filtering by filename-derived metadata,
- grouping by scan rate or condition,
- extracting peak metrics,
- trying current-density, multiscan, and normalization examples,
- exporting processed tables and figures.

When source files had indexed repeat exports, the latest indexed export was
used and renamed to the clean, non-indexed filename in this folder. For example,
the packaged Ar electrolyte file comes from the `_2` source export.

The subset intentionally excludes binary exports and phenol concentrations
above 2.8 M so the notebooks remain portable and avoid the highest-concentration
examples. The multiscan example is a clean Fe-tpyPY2Me-only Ar export rather
than a high-phenol catalytic trace.

The files are suitable as public tutorial data, not as a curated mechanistic
benchmark. They have been renamed and lightly curated for portability. Advanced
analysis examples in the notebooks are marked as demonstrations of the workflow
rather than final scientific interpretation.
