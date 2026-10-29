# Analysis equations and parameter definitions

These equations describe the eCAT 0.1.0b8 implementation. Option names are written as dictionary keys; object attributes and positional arguments are identified separately. Display units do not change the physical units required by a formula. F is the Faraday constant and R the gas constant. See [method sources](analysis_reading.md) for citations and applicability.

Extraction and display helpers without a separate physical equation are described in the [API reference](api_reference.md). The same definitions appear beside the corresponding functions in the Word guide.

## cv.peak_potential

- Ep = E at the selected local peak or requested exact potential.

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| E, i | Potential (V), current (A unless another signal axis is selected) | Selected data; segment / segments, x axis and y axis. |
| Ep | Peak potential (V); extraction only | guess potential / exact potential; peak prominence; noise window / noise polyorder. |

Peak location is an extraction algorithm, not a kinetic fit. Direction, smoothing, prominence and search choices can change which peak is selected. DPV has no CV sweep-pair interpretation.

Implementation: `src/ecat/objects.py: peak_potential`.

## dpv.peak_potential

- Ep = E at the selected local peak or requested exact potential.

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| E, i | Potential (V), current (A unless another signal axis is selected) | Selected data; segment / segments, x axis and y axis. |
| Ep | Peak potential (V); extraction only | guess potential / exact potential; peak prominence; noise window / noise polyorder. |

Peak location is an extraction algorithm, not a kinetic fit. Direction, smoothing, prominence and search choices can change which peak is selected. DPV has no CV sweep-pair interpretation.

Implementation: `src/ecat/objects.py: peak_potential`.

## cv.peak_current

- i_baseline(E) = a E + b
- ip = i(Ep) − i_baseline(Ep)

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| E, i | Potential (V), current (A unless another signal axis is selected) | Selected data; segment / segments, x axis and y axis. |
| Ep, ip | Peak potential (V), tangent-corrected peak current (A) | guess potential / exact potential; peak prominence; noise window / noise polyorder. |
| a, b | Baseline slope (A/V) and intercept (A) | tangent range / tangent potential / tangent min points. |

The baseline-corrected peak current differs from the raw current at Ep. peak_info also returns the width and half-peak quantities defined below.

Implementation: `src/ecat/objects.py: peak_current / peak_info`.

## cv.peak_info

- i_baseline(E) = a E + b
- ip = i(Ep) − i_baseline(Ep)

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| E, i | Potential (V), current (A unless another signal axis is selected) | Selected data; segment / segments, x axis and y axis. |
| Ep, ip | Peak potential (V), tangent-corrected peak current (A) | guess potential / exact potential; peak prominence; noise window / noise polyorder. |
| a, b | Baseline slope (A/V) and intercept (A) | tangent range / tangent potential / tangent min points. |

The baseline-corrected peak current differs from the raw current at Ep. peak_info also returns the width and half-peak quantities defined below.

Implementation: `src/ecat/objects.py: peak_current / peak_info`.

## cv.half_peak_potential

- i_corrected(E) = i(E) − (a E + b)
- i_corrected(Ecross) = f ip
- W = |Etrailing − Eleading|; half-peak shift = Ep − Ep/2

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| E, i | Potential (V), current (A unless another signal axis is selected) | Selected data; segment / segments, x axis and y axis. |
| Ep, ip | Peak potential (V), tangent-corrected peak current (A) | guess potential / exact potential; peak prominence; noise window / noise polyorder. |
| a, b | Baseline slope (A/V) and intercept (A) | tangent range / tangent potential / tangent min points. |
| f | Fraction of peak height (dimensionless) | level in peak_width; fixed 0.5 in half_peak_potential. |
| Eleading, Etrailing | Crossing potentials (V) | Crossings on either side of the selected peak; Ep/2 uses the leading side. |

peak_width interpolates the tangent-corrected crossings. half_peak_potential instead chooses the pre-peak sample nearest half the corrected peak current. Full width requires both crossings; unresolved trailing branches can make the width unavailable while the peak remains valid.

Implementation: `src/ecat/objects.py: half_peak_potential / peak_width`.

## cv.peak_width

- i_corrected(E) = i(E) − (a E + b)
- i_corrected(Ecross) = f ip
- W = |Etrailing − Eleading|; half-peak shift = Ep − Ep/2

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| E, i | Potential (V), current (A unless another signal axis is selected) | Selected data; segment / segments, x axis and y axis. |
| Ep, ip | Peak potential (V), tangent-corrected peak current (A) | guess potential / exact potential; peak prominence; noise window / noise polyorder. |
| a, b | Baseline slope (A/V) and intercept (A) | tangent range / tangent potential / tangent min points. |
| f | Fraction of peak height (dimensionless) | level in peak_width; fixed 0.5 in half_peak_potential. |
| Eleading, Etrailing | Crossing potentials (V) | Crossings on either side of the selected peak; Ep/2 uses the leading side. |

peak_width interpolates the tangent-corrected crossings. half_peak_potential instead chooses the pre-peak sample nearest half the corrected peak current. Full width requires both crossings; unresolved trailing branches can make the width unavailable while the peak remains valid.

Implementation: `src/ecat/objects.py: half_peak_potential / peak_width`.

## cv.half_wave_potential

- E1/2 = (Ep1 + Ep2)/2
- ΔEp = |Ep1 − Ep2|; ri = |ipa/ipc|

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| Ep1, Ep2 | Paired peak potentials (V) | segments and peak-selection options; peaks must belong to the intended wave. |
| ipa, ipc | Tangent-corrected anodic and cathodic peak currents (A) | peak-current baseline options, applied to the paired branches. |

This midpoint is an operational CV peak midpoint. It need not equal a formal potential or a catalytic half-wave potential. wave_info additionally reports branch peak widths and half-peak shifts.

Implementation: `src/ecat/objects.py: half_wave_potential / wave_info`.

## cv.wave_info

- E1/2 = (Ep1 + Ep2)/2
- ΔEp = |Ep1 − Ep2|; ri = |ipa/ipc|

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| Ep1, Ep2 | Paired peak potentials (V) | segments and peak-selection options; peaks must belong to the intended wave. |
| ipa, ipc | Tangent-corrected anodic and cathodic peak currents (A) | peak-current baseline options, applied to the paired branches. |

This midpoint is an operational CV peak midpoint. It need not equal a formal potential or a catalytic half-wave potential. wave_info additionally reports branch peak widths and half-peak shifts.

Implementation: `src/ecat/objects.py: half_wave_potential / wave_info`.

## cv.current_at_potential

- j = argmin |E_j − E*|; return the measured pair (E_j, i_j).

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| E* | Requested potential (V) | potential argument. |
| E_j, i_j | Selected-branch measurements | segment / segments; x axis / y axis. |

This helper selects the nearest measured sample; it does not interpolate. It returns None for a segment if the nearest point is more than three times |mean ΔE| from the request. Without a segment selection it reports every segment separately.

Implementation: `src/ecat/objects.py: current_at_potential`.

## ca.current_at_time

- i(t*) = linear interpolation of the current trace at t*.

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| t* | Requested time (s) | time argument or time option. |
| i | Selected current (A) | Raw current by default; corrected current / baseline correction options select the corrected trace. |

Time must lie within the measured range. Interpolation is in time, not potential.

Implementation: `src/ecat/objects.py: current_at_time`.

## ca.average_current

- i_avg = trapezoid integral of i(t) from ta to tb / (tb − ta)

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| ta, tb | Window endpoints (s) | time_range argument; time range or window option. |
| i | Selected current (A) | corrected current / baseline correction; boundary currents are interpolated. |

This is a time-weighted mean, not the arithmetic mean of samples. The window must have positive duration.

Implementation: `src/ecat/objects.py: average_current`.

## ca.rate_at_time

- electron flow = i/F
- molecular rate = |i| FE/(n F)
- TOF = molecular rate / Ncat

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| i | Current or time-weighted average current (A) | time or time range; same current selection as the corresponding current helper. |
| FE | Faradaic efficiency (fraction) | faradaic efficiency; conversion and validation follow the rate options. |
| n | Electrons per product molecule | num electrons / electron stoich. |
| Ncat | Catalyst amount (mol) | catalyst moles, or resolved catalyst concentration and volume. |
| F, R | Faraday constant (C/mol); gas constant (J/mol/K) | Package constants; no user option. |

Electron flow is signed; molecular rate and TOF use current magnitude. A molecular rate requires n, and TOF additionally requires catalyst amount. These conversions do not establish product identity or FE.

Implementation: `src/ecat/objects.py: _ca_rate_values_from_current`.

## ca.average_rate

- electron flow = i/F
- molecular rate = |i| FE/(n F)
- TOF = molecular rate / Ncat

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| i | Current or time-weighted average current (A) | time or time range; same current selection as the corresponding current helper. |
| FE | Faradaic efficiency (fraction) | faradaic efficiency; conversion and validation follow the rate options. |
| n | Electrons per product molecule | num electrons / electron stoich. |
| Ncat | Catalyst amount (mol) | catalyst moles, or resolved catalyst concentration and volume. |
| F, R | Faraday constant (C/mol); gas constant (J/mol/K) | Package constants; no user option. |

Electron flow is signed; molecular rate and TOF use current magnitude. A molecular rate requires n, and TOF additionally requires catalyst amount. These conversions do not establish product identity or FE.

Implementation: `src/ecat/objects.py: _ca_rate_values_from_current`.

## ca.charge

- Q_j = Σ(k=1…j) i_k (t_k − t_(k−1)); Q_0 = 0

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| Q | Cumulative signed charge (C) | Current and Time columns; baseline correction selects a corrected charge trace. |
| Qtarget | Target charge (C) | target charge, or target moles × target electrons × F. |
| F, R | Faraday constant (C/mol); gas constant (J/mol/K) | Package constants; no user option. |

The current implementation uses a right-endpoint sum, not trapezoidal quadrature. It assigns no charge to the initial sample. A baseline correction can change the charge materially.

Implementation: `src/ecat/objects.py: _charge_trace / charge`.

## ca.time_at_charge

- t* = linear interpolation of t against Q at Qtarget
- Qtarget = Ntarget n F (when a mole target is supplied)

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| Q | Cumulative charge (C) | Same right-endpoint sum as ca.charge; this helper uses the uncorrected charge trace. |
| Ntarget, n | Target amount (mol) and electrons per conversion | target moles / target electrons; alternatively pass charge or target charge in C. |
| F, R | Faraday constant (C/mol); gas constant (J/mol/K) | Package constants; no user option. |

The implementation calls np.interp with charge as the coordinate. Reliable inversion requires increasing charge; decreasing or nonmonotonic charge is not a general first-crossing calculation. A range check alone does not establish monotonicity.

Implementation: `src/ecat/objects.py: _time_at_charge_value / time_at_charge`.

## cp.cycle_info

- Capacity = |I| Δt × 1000/3600 (mAh)
- Energy = |I integral E(t) dt| × 1000/3600 (mWh)
- Coulombic efficiency = 100 Qdischarge/Qcharge
- Energy efficiency = 100 Wdischarge/Wcharge

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| I | Applied segment current (A) | Object cathodic_current / anodic_current metadata. |
| E, Δt | Measured potential (V); segment duration (s) | get_cycles segmentation. |
| Q, W | Discharge/charge capacity and energy | percent capacity normalizes each capacity series to its own first finite nonzero value. |

The calculation assumes constant current per segment. Energy uses trapezoidal potential integration. With percent capacity enabled, the current code computes coulombic efficiency from the separately normalized capacities; use absolute capacity mode for the physical charge ratio.

Implementation: `src/ecat/objects.py: cycle_info`.

## cp.cycling_plot

- Capacity = |I| Δt × 1000/3600 (mAh)
- Energy = |I integral E(t) dt| × 1000/3600 (mWh)
- Coulombic efficiency = 100 Qdischarge/Qcharge
- Energy efficiency = 100 Wdischarge/Wcharge

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| I | Applied segment current (A) | Object cathodic_current / anodic_current metadata. |
| E, Δt | Measured potential (V); segment duration (s) | get_cycles segmentation. |
| Q, W | Discharge/charge capacity and energy | percent capacity normalizes each capacity series to its own first finite nonzero value. |

The calculation assumes constant current per segment. Energy uses trapezoidal potential integration. With percent capacity enabled, the current code computes coulombic efficiency from the separately normalized capacities; use absolute capacity mode for the physical charge ratio.

Implementation: `src/ecat/objects.py: cycle_info`.

## dpv.fit_overlapping_peaks

- i(E) = b0 + b1 E + A1 exp[−(E−E1)^2/(2 σ1^2)] + A2 exp[−(E−E2)^2/(2 σ2^2)]

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| b0, b1 | Linear background intercept and slope | Fitted nuisance parameters. |
| A1, A2 | Gaussian amplitudes (signal units) | Fitted peak heights above the linear background. |
| E1, E2 | Peak centers (V) | guess potentials supplies initial values. |
| σ1, σ2 | Gaussian widths (V) | sigma guess / sigma min / sigma max; fit window selects points. |

Exactly two Gaussian peaks plus a linear background are fitted. This is a descriptive decomposition, not proof of two species or a mechanistic DPV model.

Implementation: `src/ecat/objects.py: _double_peak_model`.

## fit_model

- linear: y = m x + b
- power: y = A x^p; power_offset: y = b + A x^p
- exponential: y = b + A exp(k x)
- michaelis_menten: y = Vmax x/(Km + x)
- logistic: y = b + L/[1 + exp(−k(x − x0))]
- For transformed peak/rate fits: xfit = Tx(x), yfit = Ty(y); fit the model to these coordinates.

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| x, y | Coordinates in the selected units | x column / y column for fit_model; x axis / species for peak fits; x column / metric for fit_rate. |
| m, b, A, p, k, Vmax, Km, L, x0 | Fitted coefficients; units follow the chosen coordinates | fit model; fit init / fit bounds / fit params. |
| r, σ | Residual and supplied uncertainty | fit residual; fit sigma / fit absolute sigma; fit indices selects observations. |
| Tx, Ty | Transformations before fitting | x transform / y transform in peak/rate wrappers: identity, powers, sqrt, log10, ln or callable; transform mode and floors govern nonpositive inputs. |

fit_model fits its supplied coordinates directly and does not apply the peak/rate transformation layer. A plot-axis scale does not transform a fit. Coefficient units depend on the transformed coordinates; a descriptive regression is not a mechanistic rate law.

Implementation: `src/ecat/analysis_batch.py: _fit_model_equation / fit_model / _prepare_scatter_transforms`.

## fit_rate

- linear: y = m x + b
- power: y = A x^p; power_offset: y = b + A x^p
- exponential: y = b + A exp(k x)
- michaelis_menten: y = Vmax x/(Km + x)
- logistic: y = b + L/[1 + exp(−k(x − x0))]
- For transformed peak/rate fits: xfit = Tx(x), yfit = Ty(y); fit the model to these coordinates.

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| x, y | Coordinates in the selected units | x column / y column for fit_model; x axis / species for peak fits; x column / metric for fit_rate. |
| m, b, A, p, k, Vmax, Km, L, x0 | Fitted coefficients; units follow the chosen coordinates | fit model; fit init / fit bounds / fit params. |
| r, σ | Residual and supplied uncertainty | fit residual; fit sigma / fit absolute sigma; fit indices selects observations. |
| Tx, Ty | Transformations before fitting | x transform / y transform in peak/rate wrappers: identity, powers, sqrt, log10, ln or callable; transform mode and floors govern nonpositive inputs. |

fit_model fits its supplied coordinates directly and does not apply the peak/rate transformation layer. A plot-axis scale does not transform a fit. Coefficient units depend on the transformed coordinates; a descriptive regression is not a mechanistic rate law.

Implementation: `src/ecat/analysis_batch.py: _fit_model_equation / fit_model / _prepare_scatter_transforms`.

## fit_peak_current

- linear: y = m x + b
- power: y = A x^p; power_offset: y = b + A x^p
- exponential: y = b + A exp(k x)
- michaelis_menten: y = Vmax x/(Km + x)
- logistic: y = b + L/[1 + exp(−k(x − x0))]
- For transformed peak/rate fits: xfit = Tx(x), yfit = Ty(y); fit the model to these coordinates.

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| x, y | Coordinates in the selected units | x column / y column for fit_model; x axis / species for peak fits; x column / metric for fit_rate. |
| m, b, A, p, k, Vmax, Km, L, x0 | Fitted coefficients; units follow the chosen coordinates | fit model; fit init / fit bounds / fit params. |
| r, σ | Residual and supplied uncertainty | fit residual; fit sigma / fit absolute sigma; fit indices selects observations. |
| Tx, Ty | Transformations before fitting | x transform / y transform in peak/rate wrappers: identity, powers, sqrt, log10, ln or callable; transform mode and floors govern nonpositive inputs. |

fit_model fits its supplied coordinates directly and does not apply the peak/rate transformation layer. A plot-axis scale does not transform a fit. Coefficient units depend on the transformed coordinates; a descriptive regression is not a mechanistic rate law.

Implementation: `src/ecat/analysis_batch.py: _fit_model_equation / fit_model / _prepare_scatter_transforms`.

## fit_peak_potential

- linear: y = m x + b
- power: y = A x^p; power_offset: y = b + A x^p
- exponential: y = b + A exp(k x)
- michaelis_menten: y = Vmax x/(Km + x)
- logistic: y = b + L/[1 + exp(−k(x − x0))]
- For transformed peak/rate fits: xfit = Tx(x), yfit = Ty(y); fit the model to these coordinates.

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| x, y | Coordinates in the selected units | x column / y column for fit_model; x axis / species for peak fits; x column / metric for fit_rate. |
| m, b, A, p, k, Vmax, Km, L, x0 | Fitted coefficients; units follow the chosen coordinates | fit model; fit init / fit bounds / fit params. |
| r, σ | Residual and supplied uncertainty | fit residual; fit sigma / fit absolute sigma; fit indices selects observations. |
| Tx, Ty | Transformations before fitting | x transform / y transform in peak/rate wrappers: identity, powers, sqrt, log10, ln or callable; transform mode and floors govern nonpositive inputs. |

fit_model fits its supplied coordinates directly and does not apply the peak/rate transformation layer. A plot-axis scale does not transform a fit. Coefficient units depend on the transformed coordinates; a descriptive regression is not a mechanistic rate law.

Implementation: `src/ecat/analysis_batch.py: _fit_model_equation / fit_model / _prepare_scatter_transforms`.

## sevcik_analysis

- |ip| = 0.4463 n F S C sqrt(n F D v/(R T))
- Scan-rate series: D = R T/(n F)^3 × [m/(0.4463 S C)]^2
- Concentration series: D = R T/[(n F)^3 v S^2] × [mC/0.4463]^2

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| m, mC | Slope of ip vs sqrt(v), or ip vs C | fit indices chooses points. Slopes are converted to A and mol/cm³ before calculating D. |
| n | Electron count | num electrons. |
| S, T | Area (cm²), temperature (K) | First CV electrode_area and temperature metadata. |
| C | Selected species concentration (mol/cm³) | C / species or metadata; mol/L is divided by 1000. |
| v, D | Scan rate (V/s); derived diffusion coefficient (cm²/s) | CV scan_rate metadata; D is the result. |
| F, R | Faraday constant (C/mol); gas constant (J/mol/K) | Package constants; no user option. |

Assumes reversible electron transfer with planar semi-infinite diffusion. Use a fixed concentration for a scan-rate series, or a fixed scan rate for a concentration series. A nonzero intercept or nonlinear trend can signal background or model failure.

Implementation: `src/ecat/analysis_batch.py: _sevcik_analysis_payload / _format_sevcik_diffusion_equation`.

## nicholson_analysis

- z = 1000 n |Epa − Epc| (mV)
- Default: ψ = linear interpolation of Agarwal (2025), Table 4, at z.
- Optional Lavagnini: ψ = (−0.6288 + 0.0021 z)/(1 − 0.017 z)
- X = sqrt(R T/(π D n F v))
- ψ = k0 X (origin fit), or ψ = k0 X + b (linear fit)
- Per-point k0 = ψ/X

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| ψ, z | Dimensionless kinetic parameter; scaled peak separation (mV) | psi source = agarwal table (default), or lavagnini / empirical. |
| Epa, Epc, n | Paired peak potentials (V), electron count | segment / segments and peak-selection options; num electrons. |
| D | Diffusion coefficient (cm²/s) | D, required; one diffusion coefficient is used. |
| T, v | Temperature (K), scan rate (V/s) | CV temperature; fallback temperature when metadata absent. scan rate overrides CV metadata. |
| k0, b | Heterogeneous rate (cm/s), optional dimensionless intercept | fit model = origin or linear. Default is origin. |
| zmin, zmax | Accepted peak-separation range (mV) | nicholson delta ep min mv / max mv (61 / 212); exclude invalid delta ep. |
| F, R | Faraday constant (C/mol); gas constant (J/mol/K) | Package constants; no user option. |

The stored Agarwal table spans 61–214 mV; default rate-fit screening is 61–212 mV. Table interpolation returns NaN outside its domain, even if screening is relaxed. Lavagnini is an optional empirical approximation, not unrestricted extrapolation. Assumes the applicable simple quasireversible diffusion-controlled couple; uncompensated resistance or coupled chemistry can invalidate the interpretation.

Implementation: `src/ecat/analysis_batch.py: _nicholson_psi_from_source / nicholson_analysis`.

## reversibility_analysis

- z = 1000 n |Epa − Epc| (mV)
- Default: ψ = linear interpolation of Agarwal (2025), Table 4, at z.
- Optional Lavagnini: ψ = (−0.6288 + 0.0021 z)/(1 − 0.017 z)
- Bulk: Λ = sqrt(π) ψ; k0 = Λ sqrt(D n F v/(R T))
- Matsuda–Ayabe regions: reversible Λ ≥ 15; quasi-reversible 10^[−2(1+α)] < Λ < 15; irreversible Λ ≤ 10^[−2(1+α)].
- Chemical recovery: ri = |ipa/ipc|; di = |ri − 1|; fit di against log10(v).
- Surface: ip = n² F² S Γ v/(4 R T); effective ΔEp tolerance = max(configured tolerance, 3 × median potential increment).
- Surface Laviron region: n ΔEp > 200 mV; Epa = ma ln(v) + ba; Epc = mc ln(v) + bc.
- α = R T/(n F ma); β = −R T/(n F mc)
- A = R T/(α n F); B = R T/(β n F)
- ln(ks) = [A ln(α n F/(R T)) + B ln(β n F/(R T)) − (ba−bc)]/(A+B)

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| ψ, z | Dimensionless kinetic parameter; scaled peak separation (mV) | psi source = agarwal table (default), or lavagnini / empirical. |
| Epa, Epc, n | Paired peak potentials (V), electron count | segment / segments and peak-selection options; num electrons. |
| phase | Bulk diffusion or surface-confined model | phase = bulk or surface; the ψ source affects bulk analysis only. |
| α, β | Transfer coefficients (dimensionless) | Bulk alpha option sets the Matsuda–Ayabe boundary; surface α/β are independently inferred from slopes. |
| D, S, C, T, v | cm²/s, cm², mol/cm³, K, V/s | D or eligible Ševčík estimate; electrode area, C / species, temperature or shared metadata; CV scan_rate. |
| ri, di | Current-ratio recovery evidence | current ratio tolerance; min r2 controls trend acceptance. |
| Γ, ks | Surface coverage (mol/cm²); surface rate (s⁻¹) | Surface model; ks is distinct from bulk k0 in cm/s. |
| ma, mc, ba, bc, A, B | Surface slopes/intercepts and derived voltage scales | High-separation paired branch fits; at least two eligible scan rates. |
| Model agreement | Relative consistency of independent estimates | agreement tolerance; peak separation tolerance and min r2 control the surface assessment. |
| F, R | Faraday constant (C/mol); gas constant (J/mol/K) | Package constants; no user option. |

Replicate scans remain in the raw diagnostics. At each scan rate, the measured peak separations are averaged first and that mean is converted to ψ and Λ. This avoids silently dropping out-of-domain scans from one quantity while retaining them in another. Converted means have no reported ψ/Λ standard deviation; the raw separation spread is retained.

Both functions now share the same ψ converter and default. Reversibility adds its own eligibility gate (0.1 ≤ ψ ≤ 7 and z ≤ 212 mV). Values below 61 mV are treated as too reversible for a finite Nicholson rate, not assigned a measured infinite rate. Chemical recovery requires at least three finite ratios: all deviations within tolerance support recovery; mean deviation above tolerance or a reliable decrease toward unity flags coupled chemistry. These are eCAT decision rules, not proof of a unique reaction mechanism. Surface α and β are checked individually; the code does not enforce α+β=1.

Implementation: `src/ecat/reversibility.py: _nicholson_evidence / _chemical_reversibility_decision / _surface_laviron_estimate`.

## surface_coverage_analysis

- |ip| = m v + b; Γslope = 4 R T m/(n² F² S)
- Q = |trapezoid integral of [i(E) − i_baseline(E)] dE| / v
- Γcharge = Q/(n F S); loading = S Γ = Q/(n F)

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| Γ | Surface coverage (mol/cm²) | Derived separately from scan-rate slope and integrated charge. |
| m, b | Slope (A s/V) and intercept (A) | fit indices; min r2 reports fit quality. |
| n, S, T | Electron count, area (cm²), temperature (K) | num electrons; electrode area / temperature or shared metadata. |
| v, i_baseline | Scan rate (V/s), tangent baseline (A) | CV scan_rate; tangent and peak-selection options. |
| Integration limits | Potential interval (V) | integration range; agreement tolerance compares branch and method estimates. |
| F, R | Faraday constant (C/mol); gas constant (J/mol/K) | Package constants; no user option. |

Requires the reversible surface-confined model and a valid baseline. Q is the absolute net integral, not the integral of |i|. Coverage requires area; loading can remain available without area. Do not substitute bulk diffusion formulas.

Implementation: `src/ecat/reversibility.py: _surface_coverage_from_slope / _surface_charge_from_potential`.

## trumpet_analysis

- Epc = mc log10(v) + bc; Epa = ma log10(v) + ba
- xint = (bc − ba)/(ma − mc)
- α = −R T ln(10)/(2 F mc); β = R T ln(10)/(2 F ma)
- k0 = 10^(0.78 + xint/2) / sqrt(R T/(α F D))

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| mc, ma, bc, ba | Branch slopes (V/decade) and intercepts (V) | fit indices selects the fitted scan-rate interval; paired segments required. |
| v, xint | Scan rate (V/s); branch intersection in log10(V/s) | CV scan_rate metadata. |
| D, T | Diffusion coefficient (cm²/s), temperature (K) | D enables k0; temperature option or object metadata. |
| α, β, k0 | Derived transfer coefficients; rate (cm/s) | Returned by the implemented paired-branch asymptotic calculation. |
| F, R | Faraday constant (C/mol); gas constant (J/mol/K) | Package constants; no user option. |

These are the current standalone formulas, including their fixed factor 2 and constant 0.78; there is no num electrons control here. This is not the surface Laviron calculation in reversibility_analysis, which uses ln(v) and returns s⁻¹. Treat standalone kinetic estimates as model-dependent; Laviron is background for the separate surface interpretation, not a citation validating these exact formulas.

Implementation: `src/ecat/analysis_batch.py: _format_trumpet_equation / trumpet_analysis`.

## fowa

- xF = [1 + exp(−s n F (E − Eref)/(R T))]⁻¹; s = +1 for oxidation, −1 for reduction
- y = i/ip0; y = m xF + b
- EC′ apparent kobs = [0.4463 n m/(n′^σ)]² n F v/(R T)

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| E, Eref | Applied and reference potentials (V) | redox mode: half wave uses the non-catalytic reference midpoint; half peak uses the catalytic half-peak estimate; manual uses redox potential(s). |
| s | Reaction direction sign | reaction direction = auto, oxidation or reduction; auto uses the selected monotonic potential-sweep direction. |
| ip0 | Non-catalytic reference peak current (A) | ip0 / non catalytic current / non-catalytic cv(s). |
| n, n′, σ | Catalyst electron count, turnover electron count, pathway exponent | catalyst electrons (num electrons fallback), turnover electrons, sigma. |
| m, b | Fitted slope and intercept | fit basis / fit range; min fit points / min r2; selected foot region. |
| v, T, kobs | V/s, K, apparent rate s⁻¹ | CV scan_rate and temperature; mechanism / custom formula select conversion. |
| F, R | Faraday constant (C/mol); gas constant (J/mol/K) | Package constants; no user option. |

The displayed EC′ expression is mechanism-specific. The catalytic Ecat/2 and non-catalytic E1/2 need not coincide. A custom formula replaces the rate conversion and must document its own symbols. Automatic direction assumes the reaction follows the selected sweep direction; choose an explicit direction for unusual return-sweep chemistry. The half peak option is not guaranteed to give plateau-derived Ecat/2. To use a measured Ecat/2, supply it with manual mode. Fitted line display stays within the measured transformed-x span and the observed signed normalized-current envelope expanded to include zero. It never exceeds the measured current magnitude; clipping does not change regression coefficients.

Implementation: `src/ecat/analysis_batch.py: _format_fowa_kobs_equation / _fowa_formula_results`.

## plateau_current

- Direct: kobs = [|ilim|/(n F S C)]²/(D n′)
- Reference peak: kobs = [0.446 |ilim/ip0| sqrt(n F v0/(R T))]²/n′
- Reference slope: kobs = [0.446 |ilim/s0| sqrt(n F/(R T))]²/n′

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| ilim | Catalytic plateau current (A) | ilim / ic or extracted wave; plateau selection mode / plateau average method. |
| ip0, v0, s0 | Reference peak (A), its scan rate (V/s), slope of ip0 vs sqrt(v) | ip0; ip0 scan rate; ip0 sqrt scan rate slope; reference CV options. |
| n, n′ | Catalyst and turnover electron counts | catalyst electrons / turnover electrons. |
| D, S, C | cm²/s, cm², mol/cm³ | D; electrode area; C / c unit / species or metadata. |
| T, kobs | Temperature (K); apparent rate (s⁻¹) | temperature; formula mode selects direct, normalized or slope-normalized calculation. |
| F, R | Faraday constant (C/mol); gas constant (J/mol/K) | Package constants; no user option. |

These are the implemented plateau expressions, including coefficient 0.446; they are not identical to the FOWA conversion. require plateau / validate plateau and plateau slope tolerance assess selection, not mechanistic validity. A custom formula replaces this conversion.

Implementation: `src/ecat/analysis_batch.py: _format_plateau_kobs_equation`.

## normalize_current

- i_normalized = i/ip0

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| i, ip0 | Current and reference peak current in matching units | ip0 or reference cv(s) / reference index; reference guess potential selects the reference wave. |

The result is dimensionless. This wrapper copies CVs; the similarly named object method mutates its receiver. A reference peak must be nonzero.

Implementation: `src/ecat/analysis_cv.py: normalize_current`.

## scale_current

- i_scaled = s i; reference mode chooses s to match the selected reference current(s).

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| s | Multiplicative scale factor | scale, or reference cv(s) / reference index / reference mode. |

Scaling preserves current units but changes numerical current. It is distinct from a dimensionless i/ip0 column; retain scale provenance when interpreting quantitative rates.

Implementation: `src/ecat/analysis_cv.py: scale_current`.

## normalize

- θ = n F (E − E0)/(R T)
- Φ or χ = i/[n F S C sqrt(D n F v/(R T))]
- Optional homogeneous λ = khomo R T/(n F v)
- Optional heterogeneous ψ = k0/sqrt(π D n F v/(R T))

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| θ, Φ, χ | Dimensionless potential and current | mode selects normalization presentation; E0 requests potential normalization, while current-related inputs request current normalization. |
| n, E0 | Electron count; reference potential (V) | num electrons / n; E0. |
| D, S, C | cm²/s, cm², mol/cm³ | D; electrode area / area / s; C / c unit / species. |
| T, v | Temperature (K), scan rate (V/s) | temperature; scan rate / v; object metadata when not overridden. |
| khomo, k0 | Homogeneous rate (s⁻¹); heterogeneous rate (cm/s) | k homo or k0 adds the corresponding dimensionless kinetic parameter in the selected mode. |
| F, R | Faraday constant (C/mol); gas constant (J/mol/K) | Package constants; no user option. |

Use the centimeter-based unit combination shown here. Potential and current normalization do not validate a diffusion/kinetic model; homogeneous/heterogeneous modes use these same normalization scales.

Implementation: `src/ecat/analysis_cv.py: _format_cv_normalization_equation`.

## tafel_analysis

- TOF(η) = 2 TOFmax/[1 + exp(F(Ethermo − Eredox − η)/(R T))]
- y = log10(TOF)

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| TOFmax | Input rate (s⁻¹) | TOF_max positional argument; scalar or one per CV. |
| Ethermo, Eredox, η | Thermodynamic potential, redox potential, overpotential (V) | thermodynamic_potential / redox_potential arguments; overpotential range option. |
| T | Temperature (K) | CV temperature metadata. |
| F, R | Faraday constant (C/mol); gas constant (J/mol/K) | Package constants; no user option. |

This generates a molecular catalytic TOF curve; it does not fit a generic heterogeneous Tafel slope. The implemented expression approaches 2 × TOF_max at large positive η, despite the argument name. Do not interpret the input as the asymptotic value without checking the convention.

Implementation: `src/ecat/analysis_batch.py: _tafel_curve`.

## echem.spectrum

- Xk = Σ(n=0…N−1) wn xn exp(−i 2π k n/M)
- FFT amplitude: Ak = ck |Xk|/(N Gw); Gw = Σwn/N
- Welch density: Sxx(fk) = (ck/K) Σj |Xj,k|²/(fs Σwn²)

| Symbol | Meaning and units | Input or controlling option |
|---|---|---|
| xn, wn | Detrended signal and window weights | column, detrend, window; segments / time range select data. |
| N, M, K | Samples/window, FFT length, Welch window count | points per window; nfft; overlap controls Welch windows. |
| fs, fk | Sampling rate and frequency (Hz) | Derived from Time; resample / sampling tolerance govern irregular sampling. |
| ck | One-sided-bin multiplier | 1 at DC and Nyquist when present; 2 at other positive-frequency bins. |
| Ak, Sxx | Signal amplitude; signal²/Hz spectral density | method = fft or welch; frequency range / include dc select returned bins. |

Zero padding changes the frequency grid, not true frequency resolution. Welch density and FFT amplitude have different units and normalization. Nonstationary CV signals require care; a spectral peak alone does not identify a chemical timescale.

Implementation: `src/ecat/spectral.py: _spectrum_equation_bundle`.
