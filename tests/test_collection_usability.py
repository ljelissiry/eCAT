"""Public collection and display regressions using synthetic metadata."""
import copy
import pytest


def test_subfolder_display_does_not_rebase_on_notebook(ecat_module, monkeypatch, repo_root):
    from ecat.plotting import _display_folder_path_value
    monkeypatch.chdir(repo_root / 'notebooks')
    assert _display_folder_path_value('series/A') == 'series/A'
    assert _display_folder_path_value(r'series\A') == 'series/A'


def test_collection_key_case_and_paths(ecat_module, cv_factory):
    a, b = cv_factory(), cv_factory()
    a.name, b.name = 'Alpha', 'Beta'
    a.folderpath, b.folderpath = 'series/A', 'series/B'
    silent = {'print': False}
    assert ecat_module.filter([a, b], {'NAME': 'alpha'}, silent) == [a]
    assert ecat_module.filter([a, b], {'SubFolder': r'series\A'}, silent) == [a]
    assert ecat_module.sort([b, a], 'NAME', silent) == [a, b]
    assert ecat_module.group([a, b], 'Name', silent) == [[a], [b]]
    assert ecat_module.filter([a, b], ['NAME'], {'print':False}) == {'name': ['Alpha', 'Beta']}
    with pytest.raises(ValueError, match='Duplicate'):
        ecat_module.filter([a], {'name': 'Alpha', 'NAME': 'Beta'}, silent)


def test_electrode_keys_and_separate_reporting_reference(ecat_module, cv_factory):
    a, b, missing = cv_factory(), cv_factory(), cv_factory()
    a.name = 'cv_GCWE_ssLFPRE_PtCE_ACN'
    b.name = 'cv_PtWE_AgAgClRE_GCCE_MeCN'
    a.get_data_from_name(); b.get_data_from_name()
    a.reference_label = b.reference_label = 'Fc/Fc+'
    silent = {'print': False}
    assert ecat_module.filter([a,b,missing], {'WE':'gc', 'RE':'sslfp'}, silent) == [a]
    assert ecat_module.sort([missing,b,a], 'Working_Electrode', silent) == [a,b,missing]
    assert ecat_module.group([a,b], 'CE', silent) == [[a],[b]]
    from ecat.plotting import build_object_table
    table, _ = build_object_table([a,b,missing], {'columns':['WE','RE','CE']})
    assert any('Working Electrode' == col for col in table.columns)
    assert a.reference_label == 'Fc/Fc+'


@pytest.mark.parametrize('label', ['ACN','MeCN','acn','DMA','DMAc','water'])
def test_solvent_preserves_supplied_label(ecat_module, cv_factory, label):
    obj=cv_factory();obj.name=f'cv_{label}_100mVs';obj.get_data_from_name()
    assert obj.solvent == label


def test_solvent_alias_filter_and_explicit_grouping(ecat_module, cv_factory):
    a,b=cv_factory(),cv_factory()
    a.solvent,b.solvent='ACN','MeCN'
    silent={'print':False}
    assert ecat_module.filter([a,b], {'SOLVENT':'acetonitrile'},silent)==[a,b]
    assert ecat_module.group([a,b],'solvent',{**silent,'solvent grouping':'identity'})==[[a,b]]
    assert ecat_module.group([a,b],'solvent',{**silent,'solvent grouping':'label'})==[[a],[b]]
    assert (a.solvent,b.solvent)==('ACN','MeCN')


def test_window_auto_precision_and_candidate_pairs(ecat_module, cv_factory):
    a,b=cv_factory(),cv_factory()
    a.min_E,b.min_E=-1.00073,-1.0
    a.max_E=b.max_E=0.5
    before=copy.deepcopy(a.data)
    silent={'print':False}
    assert ecat_module.filter([a,b],{'Scan_Window':-1.0},silent)==[a,b]
    assert ecat_module.filter([a,b],{'scan window':[-1.,.5]},silent)==[a,b]
    assert ecat_module.filter([a,b],{'scan window':-1.0},{**silent,'sig figs':None})==[b]
    assert ecat_module.filter([a],['SCAN WINDOW'],{'print':False})=={'scan window':[[-1.00073,0.5]]}
    assert a.min_E == -1.00073
    assert a.data.equals(before)


def test_failed_filter_complete_candidates_option(ecat_module, cv_factory, capsys):
    objs=[cv_factory() for _ in range(16)]
    for i,o in enumerate(objs): o.name=f'item{i:02d}'
    ecat_module.filter(objs, {'name':'absent'}, {'pretty print':False})
    assert 'item15' not in capsys.readouterr().out
    ecat_module.filter(objs, {'name':'absent'}, {'pretty print':False,'candidate limit':None})
    assert 'item15' in capsys.readouterr().out
    assert len(ecat_module.filter(objs,['name'],{'print':False})['name'])==16


@pytest.mark.parametrize('value', [0,-1,1.5,True,float('nan')])
def test_invalid_candidate_limit(ecat_module,value):
    with pytest.raises(ValueError,match='candidate limit'):
        ecat_module.FilterOptions.from_options({'candidate limit':value})


def test_display_columns_forwarded_by_collections(ecat_module, cv_factory, monkeypatch):
    from ecat import collection
    from ecat.plotting import build_object_table
    obj=cv_factory();obj.working_electrode='GC'
    seen=[]
    def capture(objs,options):
        table,_=build_object_table(objs,options)
        seen.append(table)
    monkeypatch.setattr(collection,'show_objects',capture)
    ecat_module.filter([obj],{'NAME':obj.name},{'display columns':['WE']})
    ecat_module.sort([obj],'Name',{'display columns':['WE']})
    assert len(seen)==2
    assert all('Working Electrode' in t.columns for t in seen)


def test_group_summary_normalizes_keys(ecat_module,cv_factory):
    obj=cv_factory();obj.working_electrode='GC'
    result=ecat_module.group_summary([obj],group_keys='WE',options={'print':False})
    assert result['Working Electrode'].tolist()==['GC']


def test_subfolder_matching_root_and_dot_prefix(ecat_module,cv_factory):
    obj=cv_factory();obj.folderpath='series/A'
    assert ecat_module.filter([obj],{'subfolder':r'.\series\A' + '/' },{'print':False})==[obj]


def test_group_summary_keeps_all_solvent_labels(ecat_module,cv_factory):
    a,b=cv_factory(),cv_factory();a.solvent='ACN';b.solvent='MeCN'
    result=ecat_module.group_summary([a,b],group_keys='solvent',options={'print':False})
    assert len(result)==1
    assert 'ACN' in result['Solvent'].iloc[0] and 'MeCN' in result['Solvent'].iloc[0]
    result=ecat_module.group_summary([a,b],group_keys='solvent',options={'print':False,'solvent grouping':'label'})
    assert len(result)==2


@pytest.mark.parametrize('model',['ImportOptions','FilterOptions','SortGroupOptions'])
@pytest.mark.parametrize('value',[3,False,[1],{'WE':True}])
def test_display_columns_invalid_types(ecat_module,model,value):
    with pytest.raises(ValueError,match='display columns'):
        getattr(ecat_module,model).from_options({'display columns':value})


def test_filter_available_values_shortcut(ecat_module,cv_factory,capsys):
    a,b=cv_factory(),cv_factory();a.solvent='ACN';b.solvent='MeCN'
    a.reference_electrode=b.reference_electrode='ssLFP'
    silent={'print':False}
    expected={'solvent':['ACN','MeCN'], 'scan window':[[-0.25,0.25]], 'reference electrode':['ssLFP']}
    assert ecat_module.filter([a,b],['SOLVENT','scan_window','re'],silent)==expected
    all_values=ecat_module.filter([a,b], options=silent)
    assert ecat_module.filter([a,b],'available',silent)==all_values
    assert ecat_module.filter([a,b],options=silent)==all_values
    assert capsys.readouterr().out==''
    assert ecat_module.filter([a,b],{'solvent':'ACN'},silent)==[a,b]
    assert ecat_module.filter([a,b],{},silent)==[a,b]


def test_filter_available_values_print_limit_only(ecat_module,cv_factory,capsys):
    objs=[cv_factory() for _ in range(15)]
    for i, obj in enumerate(objs):
        obj.name = f'item{i:02d}'
    result=ecat_module.filter(objs,['name'],{'candidate limit':2})
    printed=capsys.readouterr().out
    assert len(result['name'])==15
    assert 'item01' in printed and 'item02' not in printed
    result=ecat_module.filter(objs,['name'],{'candidate limit':None})
    assert 'item14' in capsys.readouterr().out
    with pytest.raises(ValueError,match='Unknown'):
        ecat_module.filter(objs,['not a field'],{'print':False})


def test_filter_inspection_of_direct_file_without_import_root(ecat_module, tmp_path):
    path = tmp_path / '50mVs_ACN.txt'
    path.write_text('Oct. 22, 2026   12:00:00\nCyclic Voltammetry\n'
                    'Instrument Model: CHI760E\nScan Rate = 0.05\n'
                    'Potential/V,Current/A\n-0.1,0\n0.1,1e-6\n-0.1,0\n')
    obj = ecat_module.cv(path, {'print': False})
    assert not hasattr(obj, 'folderpath')
    values = ecat_module.filter([obj], options={'print': False})
    assert values['solvent'] == ['ACN']
    assert values['subfolder'] == [None]
