"""Public-loader regressions for filename and observed waveform metadata."""

import numpy as np
import pytest


def _nova_file(tmp_path, potential, time=None, name="nova.txt", header=()):
    time = np.arange(len(potential)) if time is None else time
    path = tmp_path / name
    path.write_text("\n".join([
        "NOVA ASCII export", "Technique: Cyclic Voltammetry", *header,
        "Time (s);WE(1).Potential (V);WE(1).Current (uA)",
        *(f"{t};{e};1" for t, e in zip(time, potential)),
    ]) + "\n")
    return path


@pytest.mark.parametrize("decimal", [",", "p", "."])
def test_filename_decimal_tokens_use_complete_numbers(ecat_module, tmp_path, decimal):
    path = _nova_file(tmp_path, [0, 0.1, 0.2], name=f"sample_0{decimal}5mMFc_0{decimal}1Vs.txt")
    obj = ecat_module.echem.from_file(path)
    assert type(obj).__name__ == "cv"
    assert obj.scan_rate == pytest.approx(0.1)
    assert obj.compounds == ["Fc"]
    assert obj.concentrations == ["0.5 mM"]


@pytest.mark.parametrize("suffix", ["", "_10mM_TBAPF6"])
def test_spaced_compound_survives_mixed_filename_tokens(ecat_module, tmp_path, suffix):
    path = _nova_file(tmp_path, [0, 0.1, 0.2], name=f"sample_1mM_ferrocene carboxylic acid{suffix}_100mVs.txt")
    obj = ecat_module.echem.from_file(path)
    assert obj.compounds == ["ferrocene carboxylic acid"] + (["TBAPF6"] if suffix else [])
    assert obj.concentrations == ["1 mM"] + (["10 mM"] if suffix else [])


def test_eclab_cycle_and_half_cycle_are_distinct_from_sweeps(ecat_module, tmp_path):
    path = tmp_path / "cycles.mpt"
    potential = [0, .1, .2, .2, .1, 0, 0, .1, .2, .1, 0]
    cycle = [1] * 7 + [2] * 4
    half = [0] * 4 + [1] * 3 + [2] * 2 + [3] * 2
    path.write_text("\n".join([
        "EC-Lab ASCII FILE", "Nb header lines : 5", "Cyclic Voltammetry", "nc cycles 2",
        "time/s\tEwe/V\tI/mA\tcycle number\thalf cycle",
        *(f"{i}\t{e}\t1\t{c}\t{h}" for i, (e, c, h) in enumerate(zip(potential, cycle, half))),
    ]) + "\n")
    parsed = ecat_module.echem.parse_file_to_result(path)
    obj = ecat_module.echem.from_file(path)
    assert parsed.technique == "CV"
    assert parsed.software == "EC-Lab"
    assert parsed.data["Cycle"].tolist() == cycle
    assert parsed.data["Half Cycle"].tolist() == half
    assert parsed.metadata["segments"] == obj.segments == 4


@pytest.mark.parametrize("potential, expected", [
    ([0, 0, .1, .2, .2, .1, 0, 0, .1], 3),
    ([0, .1, .2, .1, 0], 2),
    ([0, .1, .2], 1),
])
def test_nova_counts_actual_sweeps_including_partial_cycle(ecat_module, tmp_path, potential, expected):
    path = _nova_file(tmp_path, potential)
    obj = ecat_module.echem.from_file(path)
    assert obj.segments == expected


@pytest.mark.parametrize("potential, time", [
    ([0, .1, .2, .1, 0], [0, 1, 2, 3, 4]),
    ([0, .1, .25, .15, 0], [0, 1, 2.5, 3.5, 5]),
    ([0, 0, .1, .1, .2, .2, .1, .1, 0, 0], np.arange(10) / 2),
    ([0, 0, 0, .1, .2, .3, .3, .3, .2, .1, 0, 0], np.arange(12)),
])
def test_missing_scan_rate_inferred_per_ramp(ecat_module, tmp_path, potential, time):
    path = _nova_file(tmp_path, potential, time)
    parsed = ecat_module.echem.parse_file_to_result(path)
    obj = ecat_module.echem.from_file(path)
    assert parsed.units == {"Time": "s", "Potential": "V", "Current": "A"}
    assert parsed.metadata["scan_rate"] == obj.scan_rate == pytest.approx(.1)
    assert parsed.raw_metadata["scan_rate_source"] == "potential/time"
    assert any("inferred" in warning.lower() and "scan rate" in warning.lower() for warning in parsed.warnings)


@pytest.mark.parametrize("reverse_rate", [.2, .105])
def test_unequal_ramp_rates_are_unresolved(ecat_module, tmp_path, reverse_rate):
    interval = .1 / reverse_rate
    path = _nova_file(tmp_path, [0, .1, .2, .1, 0], [0, 1, 2, 2 + interval, 2 + 2 * interval])
    parsed = ecat_module.echem.parse_file_to_result(path)
    assert parsed.metadata["scan_rate"] is None
    assert any("unequal" in warning.lower() and "scan rate" in warning.lower() for warning in parsed.warnings)


@pytest.mark.parametrize("header, name, expected, source", [
    (["Scan rate: 0.5 V/s"], "nova.txt", .5, "header"),
    ([], "nova_250mVs.txt", .1, "potential/time"),
])
def test_scan_rate_source_priority(ecat_module, tmp_path, header, name, expected, source):
    path = _nova_file(tmp_path, [0, .1, .2, .1, 0], name=name, header=header)
    parsed = ecat_module.echem.parse_file_to_result(path)
    assert parsed.metadata["scan_rate"] == expected
    assert parsed.raw_metadata["scan_rate_source"] == source


def test_explicit_segments_are_preserved(ecat_module, tmp_path):
    path = _nova_file(tmp_path, [0, .1, .2], header=["Number of segments: 4"])
    parsed = ecat_module.echem.parse_file_to_result(path)
    assert parsed.metadata["segments"] == 4


def test_filename_scan_rate_can_precede_other_tokens(ecat_module, tmp_path):
    path = _nova_file(tmp_path, [0, .1, .2], name="0p25Vs_sample_0p5mM_propylamine.txt")
    obj = ecat_module.echem.from_file(path, {"print": False})
    assert obj.parse_result.raw_metadata["filename_scan_rate"] == .25
    assert obj.scan_rate == pytest.approx(.1)
    assert obj.compounds == ["propylamine"]
    assert obj.concentrations == ["0.5 mM"]


@pytest.mark.parametrize("time", [[0, 1, 1, 2, 3], [0, 1, .5, 2, 3]])
def test_invalid_time_order_does_not_produce_scan_rate(ecat_module, tmp_path, time):
    path = _nova_file(tmp_path, [0, .1, .2, .1, 0], time)
    parsed = ecat_module.echem.parse_file_to_result(path)
    assert parsed.metadata["scan_rate"] is None
    assert any("strictly increasing" in warning for warning in parsed.warnings)


def test_long_interior_hold_does_not_reduce_ramp_rate(ecat_module, tmp_path):
    potential = [0, .1, .2, .2, .2, .2, .3, .4, .5, .4, .3, .2, .1, 0]
    path = _nova_file(tmp_path, potential)
    parsed = ecat_module.echem.parse_file_to_result(path)
    assert parsed.metadata["segments"] == 2
    assert parsed.metadata["scan_rate"] == pytest.approx(.1)


def test_varying_rate_without_holds_is_unresolved(ecat_module, tmp_path):
    path = _nova_file(tmp_path, [0, .1, .2, .3, .4], [0, 1, 2, 4, 6])
    parsed = ecat_module.echem.parse_file_to_result(path)
    assert parsed.metadata["scan_rate"] is None
    assert any("vary within" in warning for warning in parsed.warnings)


def test_scan_rate_inference_converts_axes_to_canonical_units(ecat_module, tmp_path):
    path = tmp_path / "canonical.txt"
    path.write_text("Technique: Cyclic Voltammetry\nTime/ms\tPotential/mV\tCurrent/uA\n"
                    "0\t0\t1\n1000\t100\t1\n2000\t200\t1\n3000\t100\t1\n4000\t0\t1\n")
    parsed = ecat_module.echem.parse_file_to_result(path)
    assert parsed.metadata["scan_rate"] == pytest.approx(.1)
    assert parsed.units == {"Time": "s", "Potential": "V", "Current": "A"}


@pytest.mark.parametrize('token, expected', [('250mV_s', .25), ('0p5V_s', .5), ('0,5mV_s', .0005)])
def test_filename_underscore_scan_rate(ecat_module, tmp_path, token, expected):
    path = _nova_file(tmp_path, [0, .1, .2, .1, 0], name=f'sample_1mMFc_{token}_Ar.txt')
    obj = ecat_module.echem.from_file(path, {'print': False})
    assert obj.parse_result.raw_metadata["filename_scan_rate"] == pytest.approx(expected)
    assert obj.scan_rate == pytest.approx(.1)
    assert obj.compounds == ['Fc']


def _dense_program(tmp_path, *, name='program.txt', control=True, variable=False):
    # Synthetic ramps, modest measured-potential noise and shortened terminal
    # intervals. The control channel is distinct from the measured electrode.
    t = np.arange(201, dtype=float) * .1
    E = np.r_[np.arange(101) * .01, np.arange(99, -1, -1) * .01]
    if variable:
        t[50:] += (np.arange(151).clip(max=50)) * .1
    measured = E + .0004 * np.sin(np.arange(201) * 1.7)
    t[100] -= .012
    t[-1] -= .012
    path = tmp_path / name
    cols = ['time/s', 'Ewe/V', 'I/mA'] + (['control/V'] if control else [])
    rows = [f'{ti}\t{ei}\t1' + (f'\t{ci}' if control else '') for ti, ei, ci in zip(t, measured, E)]
    path.write_text('\n'.join(['Technique: Cyclic Voltammetry', '\t'.join(cols), *rows]) + '\n')
    return path, measured, E


@pytest.mark.parametrize('control', [True, False])
def test_dense_scan_rate_tolerates_voltage_noise_and_boundary_intervals(ecat_module, tmp_path, control):
    path, measured, program = _dense_program(tmp_path, control=control)
    parsed = ecat_module.echem.parse_file_to_result(path)
    assert parsed.metadata['scan_rate'] == pytest.approx(.1, rel=.001)
    np.testing.assert_allclose(parsed.data.Potential, measured)
    if control:
        np.testing.assert_allclose(parsed.data['Control Potential'], program)
        assert parsed.units['Control Potential'] == 'V'
        assert parsed.raw_metadata['scan_rate_source'] == 'control/time'
    else:
        assert parsed.raw_metadata['scan_rate_source'] == 'potential/time'


def test_dense_sustained_rate_change_is_not_smoothed_away(ecat_module, tmp_path):
    path, _, _ = _dense_program(tmp_path, variable=True)
    parsed = ecat_module.echem.parse_file_to_result(path)
    assert parsed.metadata['scan_rate'] is None
    assert any('vary within' in w for w in parsed.warnings)


def test_waveform_review_flag_uses_waveform_instead_of_filename(ecat_module, tmp_path, capsys):
    path, _, _ = _dense_program(tmp_path, name='program_250mV_s.txt')
    obj = ecat_module.echem.from_file(path, {'print': False})
    assert obj.scan_rate == pytest.approx(.1)
    raw = obj.parse_result.raw_metadata
    assert raw['scan_rate_source'] == 'control/time'
    assert raw['filename_scan_rate'] == .25
    assert raw['waveform_scan_rate'] == pytest.approx(.1)
    assert raw['scan_rate_review_required'] is True
    assert any('Review scan rate' in w for w in obj.parse_result.warnings)
    ecat_module.get_data({'folder path': str(tmp_path), 'print': True, 'pretty print': False, 'reference mode': 'none'})
    output = capsys.readouterr().out
    assert 'Review scan rate' in output
    assert '250 mV/s' in output and '100 mV/s' in output
    assert 'using control/time estimate' in output


def test_agreeing_waveform_has_no_review_flag(ecat_module, tmp_path):
    path, _, _ = _dense_program(tmp_path, name='program_100mV_s.txt')
    parsed = ecat_module.echem.parse_file_to_result(path)
    assert parsed.raw_metadata['scan_rate_review_required'] is False
    assert not any('Review scan rate' in w for w in parsed.warnings)


@pytest.mark.parametrize('potential', [
    [0, .1, .2, .2, .2, .1, 0, 0, .1],
    [0, .1, .2, .1, 0],
])
def test_segment_selection_plotting_and_lookup_share_parser_boundaries(ecat_module, tmp_path, potential):
    from ecat._cv_direction import split_cv_potential_segments
    path = _nova_file(tmp_path, potential)
    obj = ecat_module.echem.from_file(path, {'print': False})
    bounds = split_cv_potential_segments(potential)
    parts = obj._split_segment_arrays(np.asarray(potential), np.arange(len(potential)))
    assert obj.segments == len(bounds) == len(parts)
    assert ecat_module.count_segments(potential) == len(bounds)
    for b, part in zip(bounds, parts):
        start, stop, number = b['start'], b['stop'], b['segment']
        x, y = obj.analysis_segment_data({'segment': number})
        np.testing.assert_array_equal(x, potential[start:stop])
        np.testing.assert_array_equal(part['y'], np.arange(start, stop))
        for idx in range(start, stop):
            actual, sl, extent = obj.find_segment(idx=idx)
            assert actual == number - 1
            assert extent == (start, stop)


def test_control_voltage_does_not_replace_ca_control_current(ecat_module, tmp_path):
    path = tmp_path / 'ca.txt'
    path.write_text('Technique: Chronoamperometry\ntime/s\tEwe/V\tI/mA\tcontrol/mA\n'
                    '0\t.2\t1\t1\n1\t.2\t1\t1\n2\t.2\t1\t1\n')
    parsed = ecat_module.echem.parse_file_to_result(path)
    assert 'Control Potential' not in parsed.data
    assert parsed.data.Potential.tolist() == [.2, .2, .2]


def test_control_voltage_units_are_normalized(ecat_module, tmp_path):
    path = tmp_path / 'cv.txt'
    path.write_text('Technique: Cyclic Voltammetry\ntime/s\tEwe/V\tI/mA\tcontrol/mV\n'
                    '0\t0\t1\t0\n1\t.1\t1\t100\n2\t.2\t1\t200\n3\t.1\t1\t100\n4\t0\t1\t0\n')
    parsed = ecat_module.echem.parse_file_to_result(path)
    assert parsed.metadata['scan_rate'] == pytest.approx(.1)
    assert parsed.units['Control Potential'] == 'V'
    np.testing.assert_array_equal(parsed.data['Control Potential'], [0, .1, .2, .1, 0])


def test_unavailable_waveform_is_not_reported_as_verified(ecat_module, tmp_path):
    path = _nova_file(tmp_path, [0, .1, .2, .1, 0], time=[0, 1, 1, 2, 3], name='100mVs.txt')
    parsed = ecat_module.echem.parse_file_to_result(path)
    assert parsed.metadata['scan_rate'] == .1
    assert parsed.raw_metadata['waveform_scan_rate'] is None
    assert any('strictly increasing' in w for w in parsed.raw_metadata['waveform_rate_warnings'])


@pytest.mark.parametrize('token', ['DMA', 'DMAc', 'dma', 'DMAC', 'dimethylacetamide'])
def test_dimethylacetamide_solvent_aliases(ecat_module, tmp_path, token):
    path = _nova_file(tmp_path, [0, .1, .2, .1, 0], name=f'1mMFc_{token}_100mV_s.txt')
    obj = ecat_module.echem.from_file(path, {'print': False})
    assert obj.solvent == token
    assert obj.compounds == ['Fc']
    assert obj.concentrations == ['1 mM']


def test_dma_inside_compound_token_is_not_solvent(ecat_module, tmp_path):
    path = _nova_file(tmp_path, [0, .1, .2, .1, 0], name='1mM_DMAP_100mVs.txt')
    obj = ecat_module.echem.from_file(path, {'print': False})
    assert obj.solvent is None
    assert obj.compounds == ['DMAP']


@pytest.mark.parametrize('header', [
    ['Scan rate: 100 mV_s'], ['dE/dt 100', 'dE/dt unit mV_s'],
])
def test_header_underscore_scan_rate_unit(ecat_module, tmp_path, header):
    path = _nova_file(tmp_path, [0, .1, .2, .1, 0], header=header)
    parsed = ecat_module.echem.parse_file_to_result(path)
    assert parsed.metadata['scan_rate'] == .1
    assert parsed.raw_metadata['scan_rate_source'] == 'header'


def test_header_filename_conflict_also_sets_review_flag(ecat_module, tmp_path):
    path = _nova_file(tmp_path, [0, .1, .2, .1, 0], name='250mVs.txt', header=['Scan rate: 0.1 V/s'])
    with pytest.warns(UserWarning, match='using header value'):
        parsed = ecat_module.echem.parse_file_to_result(path)
    assert parsed.metadata['scan_rate'] == .1
    assert parsed.raw_metadata['scan_rate_review_required'] is True


@pytest.mark.parametrize('name', ['250mV_s.txt', '250mVs.txt'])
def test_filename_is_fallback_without_usable_time(ecat_module, tmp_path, name):
    path = tmp_path / name
    path.write_text('Technique: Cyclic Voltammetry\nPotential/V\tCurrent/A\n0\t0\n.1\t1e-6\n.2\t0\n.1\t-1e-6\n0\t0\n')
    obj = ecat_module.echem.from_file(path, {'print': False})
    assert obj.scan_rate == .25
    assert obj.parse_result.raw_metadata['scan_rate_source'] == 'filename'
    assert obj.parse_result.raw_metadata['waveform_scan_rate'] is None
