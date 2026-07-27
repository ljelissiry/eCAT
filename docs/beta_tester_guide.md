# eCAT Beta Tester Guide

This guide is for friendly beta users trying eCAT on real electrochemistry data.
The most useful feedback is not whether every advanced analysis works perfectly;
it is where installation, file import, plotting, units, metadata, or analysis
outputs become confusing.

## What To Try

- Install eCAT in a clean environment.
- Load one folder of your own text exports with `get_data()`.
- Confirm that object metadata, units, gas, solvent, concentration, and scan rate look reasonable.
- Plot individual traces and one grouped/multiplot figure.
- Run one core CV analysis such as `peak_potential()` or `peak_current()`.
- Export one processed CSV and one figure.
- Try the docs without live help and note where you get stuck.
- If you are willing, upload or share a small representative text export that
  reproduces any parser or metadata problem.

## What Is In Scope

- CH, BASI, and EC-Lab text CV exports.
- Limited CH CA/CP and EC-Lab GCPL/CP loading/plotting.
- Folder loading, filtering, grouping, plotting, reference shifting, normalization, peak analysis, and export smoke behavior.

## What Is Out Of Scope

- Binary vendor files.
- Full CA/CP analysis validation.
- New feature requests unless they block core beta workflows.
- Manuscript-level scientific claims without human review.
- Broad API redesigns or workflow changes during this beta pass.

## How To Report Feedback

Use the eCAT Beta Bug / Feedback Report Google Form:

https://docs.google.com/forms/d/e/1FAIpQLSe5rFOeuQ_qoh5NpULyKKsGGnMWOvqXH011f3dyz3X8YR603g/viewform

The most useful reports include a tiny example file, the exact code run,
expected behavior, actual behavior, traceback, and whether the issue blocks your
workflow.
