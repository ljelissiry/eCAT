# Co/Zn/CO2 CV Example Data

Small cyclic-voltammetry subset for
`notebooks/10_equilibria_fitting.ipynb`.

Source experiment folder:

```text
Electrochemistry/CV/LJE-5-049
```

Included traces:

- Ar reference: MeCN, 0.1 M TBAPF6, 3 mM Fc, 1 mM Co(dmgH)2(py)Cl
- CO2 series: MeCN, 0.1 M TBAPF6, 3 mM Fc, 1 mM Co(dmgH)2(py)Cl,
  1 mM Zn(cyclen)(OTf)2, 2.8 M H2O, and 5, 10, 20, 40, 70, or 100 percent CO2

All included files are CHI text exports with a -1.3 V to 0.9 V scan window,
100 mV/s scan rate, and 3 segments. The quickstart references potentials to
Fc/Fc+ and trims segment 1 fitting inputs from -1 V vs Fc/Fc+ to the
lower-potential end of the trace.

The original source folder also contained CHI `.bin` files and an additional
`100%CO2` duplicate export. Those are intentionally omitted from this public
example subset.
