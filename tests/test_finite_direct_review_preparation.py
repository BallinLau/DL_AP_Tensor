import json
from pathlib import Path
import torch
import pytest
from analysis import prepare_finite_direct_outer4_review as review


def test_plan_preserves_four_outers_and_no_outer0_firm_retrain():
    stages=review.stage_plan()
    assert set(s['outer'] for s in stages)=={0,1,2,3}
    assert [s['stage'] for s in stages if s['outer']==0][0]=='reuse_verified_firm'
    for outer in (1,2,3):
        actual=[s['stage'] for s in stages if s['outer']==outer]
        assert actual[:4]==['finite_firm_bank','P0_PI','Q','BP_distillation']
        assert actual.index('FC1_0')<actual.index('macro_D1_refresh')<actual.index('FC1_1')
        assert actual.index('macro_D2_refresh')<actual.index('SDF_TRUE_ONLY_chunk0')
        assert actual[-2:]==['cross_block','outer_checkpoint']
    assert not any(s['completed'] for s in stages)


def test_fc1_only_checkpoint_composition(tmp_path,monkeypatch):
    def forbidden(*a,**k): raise AssertionError('review must not optimize')
    monkeypatch.setattr(torch.optim.Adam,'step',forbidden)
    firm={'w':torch.tensor([2.])}
    macro={'sdf_model.w':torch.tensor([3.]),'value_model.w':torch.tensor([4.]),
           'fc1_model.FC1_C.w':torch.tensor([5.]),'fc1_model.FC1_K.w':torch.tensor([6.])}
    updated={k:v+10 for k,v in macro.items()}
    old=tmp_path/'firm.pt'; new=tmp_path/'phi8.pt'
    torch.save(dict(models=dict(policy_value=firm,sdf_fc1=macro),git_head='old'),old)
    torch.save(dict(models=dict(policy_value=firm,sdf_fc1=updated),iteration=8),new)
    expected=dict(firm=review.hash_state(firm),
        sdf=review.hash_state(macro,'sdf_model.'),wealth=review.hash_state(macro,'value_model.'),
        fc1_c=review.hash_state(updated,'fc1_model.FC1_C.'),fc1_k=review.hash_state(updated,'fc1_model.FC1_K.'))
    (tmp_path/'metadata.json').write_text(json.dumps({'source_commit':review.SOURCE_COMMIT}))
    (tmp_path/'summary.json').write_text(json.dumps(dict(status='COMPLETED',MACRO_ITERATIONS_COMPLETED=8,
                                                       hashes_after=expected)))
    monkeypatch.setattr(review,'FIRM_HASH',review.hash_state(firm))
    monkeypatch.setattr(review,'FIRM_FILE_HASH',review.file_hash(old))
    result,manifest=review.combined_checkpoint(old,new,tmp_path)
    assert torch.equal(result['models']['sdf_fc1']['sdf_model.w'],macro['sdf_model.w'])
    assert torch.equal(result['models']['sdf_fc1']['value_model.w'],macro['value_model.w'])
    assert torch.equal(result['models']['sdf_fc1']['fc1_model.FC1_C.w'],updated['fc1_model.FC1_C.w'])
    assert not result['training_performed'] and not manifest['old_training_banks_imported']
    assert 'optimizers' not in result
    monkeypatch.setattr(review,'FIRM_FILE_HASH','wrong')
    with pytest.raises(RuntimeError,match='SHA-256'): review.combined_checkpoint(old,new,tmp_path)


def test_review_command_has_no_training_submission_path():
    import ast
    path=Path(review.__file__); tree=ast.parse(path.read_text())
    calls={ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)}
    assert not any(c.endswith('.step') or c.endswith('.backward') for c in calls)
    assert 'build_or_load_sampling' in calls
    assert not any(c in calls for c in ['subprocess.run','subprocess.Popen','os.system'])
