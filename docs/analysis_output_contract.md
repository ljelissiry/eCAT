# Analysis Output Contract

Public eCAT analyses use the same notebook-facing report order whenever the
corresponding sections apply:

1. `<Analysis> Parameters`
2. `<Analysis> Equations`
3. `<Analysis> Summary`
4. `<Analysis> Data`, only when `print all=True`

An analysis may omit a section that has no meaningful content. It must not
move a later section ahead of an earlier one. Nested helper calls must not
print their own reports inside the parent report.

`print=False` suppresses the complete report. `pretty print=False` preserves
the same section names and order using plain-text tables and equations.

The default summary should be compact and decision-relevant. `print all=True`
may add a compact evidence or diagnostic table, but does not need to expose
every internal column. Complete numeric and audit data remain available on the
returned result through `.table`, `.fit_table`, `.summary`, and `.diagnostics`.

Display tables should:

- use `Parameter | Symbol | Value` for vertical parameter/setup tables when
  symbols appear in the corresponding equations;
- use `Metric | Value` for compact vertical summaries unless the analysis is
  naturally row-oriented, such as one row per CV or condition;
- keep units with values in vertical parameter tables or in column headers for
  row-oriented data;
- format electrochemical symbols consistently in rich notebook output;
- respect `sig figs` without changing stored numeric values;
- lead with the meaningful analysis variable, such as scan rate or
  concentration, plus other genuinely varying condition columns;
- omit `Name` when those visible context columns uniquely identify every row;
- include `Name` as the first column when duplicate visible contexts remain,
  including values that become equal after `sig figs` display formatting;
- identify aggregated rows with `Condition`, `Group`, or `Branch` rather than
  a single-object `Name`;
- remain left aligned with the package's shared table formatter.

Equation sections should show symbolic equations by default. They should not
repeat resolved user inputs, constants, or symbol definitions that already belong
in the Parameters table.

Returned numeric tables may retain object names even when the compact printed
table omits them. This preserves traceability for filtering and export without
repeating long filenames in routine notebook output.

Specialized fit reports may use their established setup/details and parameter
tables, but analyses that print parameters, equations, summaries, and data must
follow this ordering contract.
