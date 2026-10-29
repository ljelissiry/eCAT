"""Released equilibrium input, update and fitting contracts (no custom backend)."""


from copy import deepcopy


import numpy as np


import pytest


@pytest.fixture
def sim(ecat_module):
    return ecat_module.simulation


def activity(sim, standard=1000.0, coverage=1.0, gamma=None):
    return sim._activity_config({'activity': {'standard_concentration': standard,
                                             'standard_coverage': coverage,
                                             'gamma': gamma or {}}})


def entry(sim, reaction='A+B=AB', value='2 M^-1', phase='bulk'):
    return {**sim._coerce_equilibrium_constant_fields(value), 'reaction': reaction,
            'phase': phase}


@pytest.mark.parametrize('standard, expected', [(1000.0, 2.0), (100.0, 0.2)])
def test_association_standard_state_regression(sim, standard, expected):
    item = entry(sim)
    parsed = sim._parse_equilibrium_reaction(item['reaction'])
    assert sim._dimensionless_equilibrium_constant(item, parsed, activity(sim, standard)) == pytest.approx(expected)
    assert sim._equilibrium_concentration_quotient(item, parsed, activity(sim, standard)) == pytest.approx(0.002)


def test_gamma_ratio_and_inverse_conversion(sim):
    item = entry(sim)
    parsed = sim._parse_equilibrium_reaction(item['reaction'])
    config = activity(sim, 100, gamma={'bulk': {'A': .5, 'B': 2, 'AB': 3}})
    assert sim._dimensionless_equilibrium_constant(item, parsed, config) == pytest.approx(.6)
    assert sim._equilibrium_concentration_quotient(item, parsed, config) == pytest.approx(.002)


def test_conflicting_unit_annotations_raise(sim):
    with pytest.raises(ValueError, match='[Cc]onflict'):
        sim._coerce_equilibrium_constant_fields('2 M^-1', {'K_unit': 'mM^-1'})


@pytest.mark.parametrize('standard', [1000, 100])
def test_legacy_numeric_k_stays_dimensionless(sim, standard):
    item = entry(sim, value=2)
    assert sim._dimensionless_equilibrium_constant(item, sim._parse_equilibrium_reaction(item['reaction']), activity(sim, standard)) == 2


def params(value='2 M^-1'):
    return {'concentrations': {'bulk': {'A': 10., 'B': 10., 'AB': 0.}},
            'activity': {'standard_concentration': 100.},
            'reactions': [{'reaction': 'A+B=AB', 'K': value, 'koff': 5.}]}


def test_preequilibrium_and_rates_share_physical_quotient(sim):
    prepared = sim._expand_parameter_model(params())
    c = prepared['concentrations']['bulk']
    # Independently computed positive root: x = .002*(10-x)^2.
    expected = (1.04 - np.sqrt(1.04**2 - 4*.002*.2))/(2*.002)
    assert c['AB'] == pytest.approx(expected, rel=1e-6)
    assert c['A'] + c['AB'] == pytest.approx(10)
    rates = prepared['_compiled']['reactions'][0]
    assert rates['kf']/rates['kb'] == pytest.approx(.002)
    report = prepared['_parameter_model']['equilibria'][0]
    assert report['K_dimensionless'] == pytest.approx(.2)
    assert report['logK'] == pytest.approx(np.log(.2))


def test_fit_init_bounds_vector_and_fixed_are_canonical(sim):
    original = params()
    program = sim.cv_program(0, E_low=-.2, points_per_segment=4)
    path = ('reactions', 0, 'K')
    spec = sim._normalize_fit_spec({'vary': ['reactions.0.K'],
          'init': {'reactions.0.K': '0.003 mM^-1'},
          'bounds': {'reactions.0.K': ['1 M^-1', '4 M^-1']}}, original, program)
    assert spec['base_params']['reactions'][0]['K'] == pytest.approx(.2)
    assert spec['entries'][path]['init'] == pytest.approx(.3)
    assert spec['entries'][path]['bounds'] == pytest.approx((.1, .4))
    updated = sim._params_from_fit_vector(spec['base_params'], spec, [np.log10(.35)])
    assert updated['reactions'][0]['K'] == pytest.approx(.35)
    assert updated['reactions'][0]['K_unit'] == 'dimensionless'
    derived = sim._expand_parameter_model(updated)['_parameter_model']['equilibria'][0]
    assert derived['K_concentration'] == pytest.approx(.0035)
    fixed = sim._normalize_fit_spec({'vary': [], 'fixed': {'reactions.0.K': '3 M^-1'}}, original, program)
    assert fixed['base_params']['reactions'][0]['K'] == pytest.approx(.3)
    assert original == params()


@pytest.mark.parametrize('reverse', [False, True])
def test_fixed_standard_gamma_and_kc_are_order_independent(sim, reverse):
    model = params()
    updates = [('reactions.0.K', '2 M^-1'), ('activity.standard_concentration', 200),
               ('activity.gamma.bulk.A', .5)]
    model['activity']['gamma'] = {'bulk': {'A': 1}}
    fit = {'vary': [], 'fixed': dict(reversed(updates) if reverse else updates)}
    fixed = sim._normalize_fit_spec(fit, model, sim.cv_program(0, E_low=-.2))
    assert fixed['base_params']['reactions'][0]['K'] == pytest.approx(.8)
    rates = sim._expand_parameter_model(fixed['base_params'])['_compiled']['reactions'][0]
    assert rates['kf']/rates['kb'] == pytest.approx(.002)


def test_group_per_cv_kc_bounds_use_each_dataset_context(sim):
    program = sim.cv_program(0, E_low=-.2)
    models = [params(), params()]
    models[1]['activity']['standard_concentration'] = 1000
    fit = {'vary': ['reactions.0.K'], 'init': {'reactions.0.K': '3 M^-1'},
           'bounds': {'reactions.0.K': ['1 M^-1', '4 M^-1']}}
    spec = sim._normalize_fit_spec(fit, models[0], program)
    group = sim._group_fit_spec(spec, [('reactions', 0, 'K')], models, [{'input': program}]*2)
    first, second = [group['entries'][path] for path in group['vary']]
    assert first['init'] == pytest.approx(.3)
    assert first['bounds'] == pytest.approx((.1, .4))
    assert second['init'] == pytest.approx(3)
    assert second['bounds'] == pytest.approx((1, 4))
    _, by_cv = sim._group_params_from_fit_vector(group, [np.log10(.3), np.log10(3)])
    for model in by_cv:
        rates = sim._expand_parameter_model(model)['_compiled']['reactions'][0]
        assert rates['kf']/rates['kb'] == pytest.approx(.003)
    with pytest.raises(ValueError, match='Shared equilibrium K'):
        sim._group_fit_spec(spec, [], models, [{'input': program}]*2)


def test_group_shared_dimensionless_k_across_standards(sim):
    program = sim.cv_program(0, E_low=-.2)
    models = [params(), params()]
    models[1]['activity']['standard_concentration'] = 1000
    fit = {'vary': ['reactions.0.K'], 'init': {'reactions.0.K': .3},
           'bounds': {'reactions.0.K': [.1, .4]}}
    spec = sim._normalize_fit_spec(fit, models[0], program)
    group = sim._group_fit_spec(spec, [], models, [{'input': program}]*2)
    _, by_cv = sim._group_params_from_fit_vector(group, [np.log10(.3)])
    assert [m['reactions'][0]['K'] for m in by_cv] == pytest.approx([.3, .3])


def test_numeric_k_and_rate_k_are_not_conflated(sim):
    model = {'reactions': [{'k': 2}]}
    sim._set_param_path(model, ('reactions', 0, 'k'), 3)
    assert model == {'reactions': [{'k': 3}]}
    with pytest.raises(ValueError, match='rate k'):
        sim._set_param_path(model, ('reactions', 0, 'k'), '2 M^-1')


class QuotientBackend:
    name = 'test-quotient'

    def simulate(self, input_obj, mechanism_spec, model, options):
        rate = model['_compiled']['reactions'][0]
        quotient = rate['kf']/rate['kb']
        return input_obj.E, quotient*(input_obj.E - .1), input_obj.t, self


def public_params():
    model = params()
    model['concentrations']['bulk'].update({'Ox': 1., 'Red': 0.})
    model['kinetics'] = [{'E0': -.1, 'k0': 1e-3, 'alpha': .5}]
    model['reactions'][0]['equilibrate'] = False
    return model


def test_public_fit_reports_canonical_k_and_preserves_input(sim, monkeypatch):
    monkeypatch.setattr(sim, 'get_backend', lambda name='electrokitty': QuotientBackend())
    program = sim.SimulatedCVInput(E=np.array([0., -.1, -.2]), t=np.array([0., 1., 2.]),
                                  i=.003*(np.array([0., -.1, -.2])-.1),
                                  metadata={'kind': 'cv_data', 'scan_rate': .1})
    original = public_params()
    before = deepcopy(original)
    result = sim.fit_cv(program, 'E(1):Ox=Red\nC:A+B=AB', original,
                        fit={'vary': ['reactions.0.K'],
                             'bounds': {'reactions.0.K': ['1 M^-1', '4 M^-1']}},
                        options={'plot': False, 'residual': 'direct', 'current sign': 1})
    assert original == before
    assert result.best_params['reactions'][0]['K'] == pytest.approx(.3, rel=1e-4)
    assert result.best_params['reactions'][0]['K_unit'] == 'dimensionless'
    derived = result.simulation_result.summary['parameter_model']['equilibria'][0]
    assert derived['K_concentration'] == pytest.approx(.003, rel=1e-4)


@pytest.mark.parametrize('reverse', [False, True])
def test_public_with_params_resolves_updates_after_standard(sim, monkeypatch, reverse):
    monkeypatch.setattr(sim, 'get_backend', lambda name='electrokitty': QuotientBackend())
    result = sim.simulate_cv(sim.cv_program(0, E_low=-.2, points_per_segment=4),
                            'E(1):Ox=Red\nC:A+B=AB', public_params(),
                            options={'plot': False})
    updates = [('reactions.0.K', '2 M^-1'), ('activity.standard_concentration', 200)]
    changed = result.with_params(set=dict(reversed(updates) if reverse else updates), options={'plot': False})
    assert changed.params['reactions'][0]['K'] == pytest.approx(.4)
    assert changed.summary['parameter_model']['equilibria'][0]['K_concentration'] == pytest.approx(.002)


def test_nested_update_clears_old_quotient_unit(sim):
    old = {'reactions': {'A+B=AB': {'K': 2., 'K_unit': 'M^-1'}}}
    updated = sim._deep_merge_dicts(old, {'reactions': {'A+B=AB': {'K': .4}}})
    item = updated['reactions']['A+B=AB']
    assert item.get('K_unit', 'dimensionless') == 'dimensionless'
    assert sim._dimensionless_equilibrium_constant(item, sim._parse_equilibrium_reaction('A+B=AB'), activity(sim, 100)) == pytest.approx(.4)


def test_shared_group_accepts_equivalent_numeric_and_string_k(sim):
    program = sim.cv_program(0, E_low=-.2)
    models = [params('2 M^-1'), params(.2)]
    spec = sim._normalize_fit_spec(['reactions.0.K'], models[0], program)
    group = sim._group_fit_spec(spec, [], models, [{'input': program}]*2)
    assert group['entries'][('reactions', 0, 'K')]['init'] == pytest.approx(.2)


@pytest.mark.parametrize('value,unit', [('2 M^-1', '1/M'), ('2 m^3/mol', 'm3/mol'), ('2 mM^-1', 'mM-1')])
def test_equivalent_unit_annotations_are_accepted(sim, value, unit):
    item = sim._coerce_equilibrium_constant_fields(value, {'K_unit': unit})
    assert item['K'] == 2


@pytest.mark.parametrize('path,value', [
    (('concentrations', 'bulk', 'K'), 0.),
    (('concentrations', 'bulk', 'K'), 2.),
    (('diffusion', 'K'), 1e-9),
    (('activity', 'gamma', 'bulk', 'K'), .8),
])
def test_species_named_k_keeps_ordinary_parameter_semantics(sim, path, value):
    model = {'concentrations': {'bulk': {'K': 1.}}, 'diffusion': {'K': 2e-9},
             'activity': {'gamma': {'bulk': {'K': 1.}}}}
    sim._set_param_path(model, path, value)
    container = sim._get_param_path(model, path[:-1])
    assert container == {'K': value}


def test_species_named_k_fit_allows_zero_init_and_bounds(sim):
    model = {'concentrations': {'bulk': {'K': 1.}}}
    fit = {'vary': ['concentrations.bulk.K'], 'init': {'concentrations.bulk.K': 0},
           'bounds': {'concentrations.bulk.K': [0, 2]}, 'transform': {'concentrations.bulk.K': 'linear'}}
    spec = sim._normalize_fit_spec(fit, model, sim.cv_program(0, E_low=-.2))
    assert spec['entries'][('concentrations', 'bulk', 'K')]['init'] == 0


@pytest.mark.parametrize('reverse', [False, True])
def test_fixed_gamma_for_species_k_precedes_equilibrium_conversion(sim, reverse):
    model = {'concentrations': {'bulk': {'K': 10., 'B': 10., 'KB': 0.}},
             'activity': {'standard_concentration': 100., 'gamma': {'bulk': {'K': 1.}}},
             'reactions': [{'reaction': 'K+B=KB', 'K': 2., 'koff': 5.}]}
    updates = [('reactions.0.K', '2 M^-1'), ('activity.gamma.bulk.K', .5)]
    spec = sim._normalize_fit_spec({'vary': [], 'fixed': dict(reversed(updates) if reverse else updates)},
                                  model, sim.cv_program(0, E_low=-.2))
    assert spec['base_params']['reactions'][0]['K'] == pytest.approx(.4)
    assert spec['base_params']['activity']['gamma']['bulk'] == {'K': .5}


def test_nested_species_k_update_does_not_strip_other_species(sim):
    old = {'concentrations': {'bulk': {'K': 1., 'K_unit': 2.}}}
    updated = sim._deep_merge_dicts(old, {'concentrations': {'bulk': {'K': 0.}}})
    assert updated == {'concentrations': {'bulk': {'K': 0., 'K_unit': 2.}}}


def test_shared_non_equilibrium_fit_keeps_first_dataset_bounds(sim):
    first = sim.SimulatedCVInput(E=np.array([-1., 0.]), t=np.array([0., 1.]))
    second = sim.SimulatedCVInput(E=np.array([0., 1.]), t=np.array([0., 1.]))
    models = [{'kinetics': [{'E0': -.5}]} for _ in range(2)]
    fit = {'vary': ['E0_0'], 'init': {'E0_0': -.5}}
    spec = sim._normalize_fit_spec(fit, models[0], first)
    group = sim._group_fit_spec(spec, [], models, [{'input': first}, {'input': second}])
    assert group['entries'][('kinetics', 0, 'E0')] == {'init': -.5, 'bounds': (-1., 0.), 'transform': 'linear'}


def test_per_cv_non_equilibrium_fit_keeps_shared_bounds_and_dataset_init(sim):
    first = sim.SimulatedCVInput(E=np.array([-1., 0.]), t=np.array([0., 1.]))
    second = sim.SimulatedCVInput(E=np.array([0., 1.]), t=np.array([0., 1.]))
    models = [{'kinetics': [{'E0': -.5}]}, {'kinetics': [{'E0': -.25}]}]
    spec = sim._normalize_fit_spec({'vary': ['E0_0']}, models[0], first)
    path = ('kinetics', 0, 'E0')
    group = sim._group_fit_spec(spec, [path], models, [{'input': first}, {'input': second}])
    assert group['entries'][('per_cv', 1, path)] == {'init': -.25, 'bounds': (-1., 0.), 'transform': 'linear'}


@pytest.mark.parametrize('unit,value', [('M^-1', 2.), ('mM^-1', .002), ('m^3/mol', .002), ('native', .002)])
@pytest.mark.parametrize('standard', [100., 1000.])
def test_supported_quotient_units_are_physically_equivalent(sim, unit, value, standard):
    item = {'reaction': 'A+B=AB', 'K': value, 'K_unit': unit, 'phase': 'bulk'}
    config = activity(sim, standard, gamma={'bulk': {'A': .5, 'B': 2., 'AB': 3.}})
    parsed = sim._parse_equilibrium_reaction(item['reaction'])
    assert sim._dimensionless_equilibrium_constant(item, parsed, config) == pytest.approx(.006*standard)
    assert sim._equilibrium_concentration_quotient(item, parsed, config) == pytest.approx(.002)


@pytest.mark.parametrize('unit', ['M^-1', 'mM^-1', 'm^3/mol'])
def test_named_inverse_concentration_unit_rejects_wrong_reaction_order(sim, unit):
    item = entry(sim, reaction='2A+B=AB', value='2 '+unit)
    with pytest.raises(ValueError, match='stoichiometry'):
        sim._dimensionless_equilibrium_constant(item, sim._parse_equilibrium_reaction(item['reaction']), activity(sim))


def test_native_quotient_retains_higher_order_support(sim):
    item = entry(sim, reaction='2A+B=AB', value='2e-6 native')
    config = activity(sim, 100.)
    parsed = sim._parse_equilibrium_reaction(item['reaction'])
    assert sim._dimensionless_equilibrium_constant(item, parsed, config) == pytest.approx(.02)
    assert sim._equilibrium_concentration_quotient(item, parsed, config) == pytest.approx(2e-6)


@pytest.mark.parametrize('value', [0, -1, float('inf'), float('nan')])
def test_fit_equilibrium_values_must_be_positive_and_finite(sim, value):
    with pytest.raises(ValueError, match='positive and finite'):
        sim._normalize_fit_spec({'vary': ['reactions.0.K'], 'init': {'reactions.0.K': value}},
                                params(), sim.cv_program(0, E_low=-.2))


def test_gamma_fitting_remains_supported_with_canonical_k(sim):
    model = params('2 M^-1')
    model['activity']['gamma'] = {'bulk': {'A': .5}}
    program = sim.cv_program(0, E_low=-.2)
    spec = sim._normalize_fit_spec(['reactions.0.K', 'activity.gamma.bulk.A'], model, program)
    assert spec['base_params']['reactions'][0]['K'] == pytest.approx(.4)
    updated = sim._params_from_fit_vector(spec['base_params'], spec, [np.log10(.4), np.log10(1.)])
    rate = sim._expand_parameter_model(updated)['_compiled']['reactions'][0]
    assert rate['kf']/rate['kb'] == pytest.approx(.004)
    all_spec = sim._normalize_fit_spec(None, model, program)
    assert ('activity', 'gamma', 'bulk', 'A') in all_spec['vary']


@pytest.mark.parametrize('key', ['K unit', 'K units', 'K_unit'])
def test_numeric_update_replaces_old_unit_annotation(sim, key):
    model = params(2.)
    model['reactions'][0][key] = 'M^-1'
    sim._set_param_path(model, ('reactions', 0, 'K'), .4)
    prepared = sim._expand_parameter_model(model)
    rate = prepared['_compiled']['reactions'][0]
    assert rate['kf']/rate['kb'] == pytest.approx(.004)
    assert model['reactions'][0]['K_unit'] == 'dimensionless'


def test_bare_numeric_fixed_k_is_not_reinterpreted_by_previous_units(sim):
    model = params('2 M^-1')
    fit = {'vary': [], 'fixed': {'reactions.0.K': .4}}
    spec = sim._normalize_fit_spec(fit, model, sim.cv_program(0, E_low=-.2))
    rate = sim._expand_parameter_model(spec['base_params'])['_compiled']['reactions'][0]
    assert rate['kf']/rate['kb'] == pytest.approx(.004)


@pytest.mark.parametrize('reverse', [False, True])
def test_fixed_context_updates_base_unit_bearing_k_even_when_k_not_fixed(sim, reverse):
    model = params('2 M^-1')
    model['activity']['gamma'] = {'bulk': {'A': 1.}}
    fixed = [('activity.standard_concentration', 200.), ('activity.gamma.bulk.A', .5)]
    spec = sim._normalize_fit_spec({'vary': ['reactions.0.K'], 'fixed': dict(reversed(fixed) if reverse else fixed)},
                                  model, sim.cv_program(0, E_low=-.2))
    assert spec['entries'][('reactions', 0, 'K')]['init'] == pytest.approx(.8)


def test_fit_unit_conversion_uses_fixed_reference_context(sim):
    # Unit-bearing entries are normalized once after fixed changes. They are
    # dimensionless optimizer values, not moving Kc bounds during gamma fitting.
    model = params('2 M^-1')
    model['activity']['gamma'] = {'bulk': {'A': 1.}}
    fit = {'vary': ['reactions.0.K', 'activity.gamma.bulk.A'],
           'init': {'reactions.0.K': '3 M^-1', 'activity.gamma.bulk.A': .5},
           'bounds': {'reactions.0.K': ['1 M^-1', '4 M^-1']}}
    spec = sim._normalize_fit_spec(fit, model, sim.cv_program(0, E_low=-.2))
    k_entry = spec['entries'][('reactions', 0, 'K')]
    assert k_entry['init'] == pytest.approx(.3)
    assert k_entry['bounds'] == pytest.approx((.1, .4))
    vector, _, _ = sim._fit_vectors(spec)
    updated = sim._params_from_fit_vector(spec['base_params'], spec, vector)
    rate = sim._expand_parameter_model(updated)['_compiled']['reactions'][0]
    assert rate['kf']/rate['kb'] == pytest.approx(.0015)


def test_fit_canonicalization_is_idempotent(sim):
    program = sim.cv_program(0, E_low=-.2)
    first = sim._normalize_fit_spec(['reactions.0.K'], params(), program)
    second = sim._normalize_fit_spec(['reactions.0.K'], first['base_params'], program)
    assert second['base_params'] == first['base_params']
    assert second['entries'] == first['entries']


@pytest.mark.parametrize('mode', ['path', 'nested'])
@pytest.mark.parametrize('value,quotient', [(.4, .004), ('0.003 mM^-1', .003)])
def test_public_updates_replace_unit_convention(sim, monkeypatch, mode, value, quotient):
    monkeypatch.setattr(sim, 'get_backend', lambda name='electrokitty': QuotientBackend())
    result = sim.simulate_cv(sim.cv_program(0, E_low=-.2, points_per_segment=4),
                            'E(1):Ox=Red\nC:A+B=AB', public_params(), options={'plot': False})
    before = deepcopy(result.input_params)
    if mode == 'path':
        changed = result.with_param('reactions.0.K', value, options={'plot': False})
    else:
        changed = result.with_params({'reactions': [{'K': value, 'koff': 5., 'equilibrate': False}]},
                                     options={'plot': False})
    assert changed.summary['parameter_model']['equilibria'][0]['K_concentration'] == pytest.approx(quotient)
    assert result.input_params == before


@pytest.mark.parametrize('per_cv', [True, False])
def test_public_group_fit_recovers_native_quotients(sim, monkeypatch, per_cv):
    monkeypatch.setattr(sim, 'get_backend', lambda name='electrokitty': QuotientBackend())
    models = [public_params(), public_params()]
    if per_cv:
        models[1]['activity']['standard_concentration'] = 1000.
    before = deepcopy(models)
    potential = np.array([0., -.1, -.2])
    program = sim.SimulatedCVInput(E=potential, t=np.array([0., 1., 2.]),
                                  i=.003*(potential-.1), metadata={'scan_rate': .1})
    datasets = [sim.simulate_cv(program, 'E(1):Ox=Red\nC:A+B=AB', m, options={'plot': False})
                for m in models]
    fit = {'vary': ['reactions.0.K'], 'bounds': {'reactions.0.K': ['1 M^-1', '4 M^-1']}}
    result = sim.fit_cvs(datasets, fit=fit,
                        per_cv=['reactions.0.K'] if per_cv else [],
                        options={'plot': False, 'print': False, 'progress': False,
                                 'residual': 'direct', 'current sign': 1})
    assert models == before
    for current in result.simulation_results:
        rate = current.params['_compiled']['reactions'][0]
        assert rate['kf']/rate['kb'] == pytest.approx(.003, rel=2e-4)
    assert [m['reactions'][0]['K'] for m in result.best_params_by_cv] == pytest.approx([.3, 3. if per_cv else .3], rel=2e-4)


@pytest.mark.parametrize('reaction,value', [('A+B+C=ABC', 1e308), ('ABC=A+B+C', 1e-320)])
def test_nonrepresentable_canonical_or_quotient_is_rejected(sim, reaction, value):
    item = entry(sim, reaction, value)
    parsed = sim._parse_equilibrium_reaction(reaction)
    # Dimensionless inputs may fit float64 while their implied native quotient
    # over/underflows. Never publish zero or infinity as a usable rate ratio.
    standard = 1e-100
    with pytest.raises(ValueError, match='representable'):
        sim._equilibrium_concentration_quotient(item, parsed, activity(sim, standard))


def test_public_unit_update_uses_replacement_mechanism_context(sim, monkeypatch):
    monkeypatch.setattr(sim, 'get_backend', lambda name='electrokitty': QuotientBackend())
    model = public_params()
    model['reactions'][0].pop('reaction')
    model['concentrations']['bulk'].update({'C': 10., 'AC': 0.})
    model['activity']['gamma'] = {'bulk': {'AC': 2.}}
    program = sim.cv_program(0, E_low=-.2, points_per_segment=4)
    result = sim.simulate_cv(program, 'E(1):Ox=Red\nC:A+B=AB', model, options={'plot': False})
    before = deepcopy(result.input_params)
    new_mechanism = 'E(1):Ox=Red\nC:A+C=AC'
    changed = result.with_param('reactions.0.K', '2 M^-1', mechanism=new_mechanism, options={'plot': False})
    fresh = sim.simulate_cv(program, new_mechanism, model, options={'plot': False})
    assert changed.summary['parameter_model']['equilibria'][0]['K_concentration'] == pytest.approx(.002)
    assert changed.summary['parameter_model']['equilibria'][0]['K_dimensionless'] == pytest.approx(.4)
    np.testing.assert_allclose(changed.data['Backend Current'], fresh.data['Backend Current'])
    assert result.input_params == before
