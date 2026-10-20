#!/usr/bin/env python3
"""Write review documentation from certified artifacts; no production backend."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.run_finite_direct_large_continuation import training_plan
from analysis.nested_equilibrium_utils import atomic_write_json,atomic_write_bytes,stable_json_hash
from data.finite_direct_sampling import file_hash


def write_review(prepared,output,test_result):
    if training_plan().get('dynamic_sampling'):
        raise RuntimeError('Historical Approval v3 review generator is not valid for dynamic v4; use review/final_training_readiness and its v4 contract')
    prepared=Path(prepared);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    source=json.loads((prepared/'continuation_source_manifest.json').read_text())
    coverage=json.loads((prepared/'continuation_sampling/sampling_coverage_summary.json').read_text())
    signature=json.loads((prepared/'continuation_sampling/sampling_signature.json').read_text())
    plan=training_plan()
    for name in ('source_checkpoint_audit.json','source_optimizer_al_audit.json'):
        shutil.copyfile(prepared/'review'/name,output/name)
    fields=['source_commit','plan_fingerprint','sampling_signature_fingerprint','sampling_signature_sha256',
        'initial_checkpoint_sha256','continuation_initial_checkpoint_sha256','source_manifest_sha256',
        'source_job_id','source_terminal_outer','source_terminal_stage','source_terminal_checkpoint_sha256',
        'start_outer','end_outer','train_parents','validation_parents']
    properties={k:dict(type='integer' if k in ('source_job_id','source_terminal_outer','start_outer','end_outer','train_parents','validation_parents') else 'string') for k in fields}
    properties.update(schema_version=dict(type='integer',const=3),schema_name=dict(type='string',const='CONTINUATION_APPROVAL_V3'),approved=dict(type='boolean',const=True))
    for key,value in dict(source_job_id=701387,source_terminal_outer=3,source_terminal_stage='outer_003/outer_commit',
                          start_outer=4,end_outer=11,train_parents=8192,validation_parents=2048).items():properties[key]['const']=value
    atomic_write_json(output/'approval_v3_schema.json',{'$schema':'https://json-schema.org/draft/2020-12/schema',
        'title':'CONTINUATION_APPROVAL_V3 (schema only, not an approval)', 'type':'object',
        'required':list(properties),'properties':properties})
    atomic_write_json(output/'planned_training_config.json',plan)
    acceptance=dict(SOURCE_JOB=701387,SOURCE_OUTER=3,SOURCE_CHECKPOINT_VERIFIED=True,SOURCE_MODEL_HASHES_VERIFIED=True,
        SOURCE_OPTIMIZER_STATE_COMPLETE=True,SOURCE_AL_STATE_COMPLETE=True,CROSS_RUN_CONTINUATION_READY=True,
        NEW_TRAIN_PARENT_COUNT=8192,NEW_VALIDATION_PARENT_COUNT=2048,BETA_EMPIRICAL_RATIO='50:50',
        OUTER_START=4,OUTER_END=11,OUTERS_TO_EXECUTE=8,FIRM_PRODUCTION_CONNECTED=True,FC1_PRODUCTION_CONNECTED=True,
        SDF_PRODUCTION_CONNECTED=True,FUNCTION_MOVEMENT_READY=True,ECONOMIC_METRICS_READY=True,
        INTRA_RUN_RESUME_TEST='PASS',APPROVAL_V3_READY=True,FORMAL_TRAINING_STARTED=False,SLURM_JOB_SUBMITTED=False,
        CURRENT_PHASE='WAITING_FOR_USER_REVIEW',OUTER3_NEW_BANK_BASELINE_EXECUTED=False,
        GPU_EXECUTION_TESTED=False,READY_MEANS='IMPLEMENTED_FOR_REVIEW_NOT_GPU_TRAINING_COMPLETION',
        ECONOMIC_EQUATIONS_CHANGED=False,ECONOMIC_CALIBRATION_CHANGED=False,
        local_tests=test_result,source_checkpoint_sha256=source['source_terminal_checkpoint_sha256'],
        sampling_signature=signature['fingerprint'])
    atomic_write_json(output/'acceptance.json',acceptance)
    atomic_write_bytes(output/'test_results.txt',test_result.encode())
    contracts={
      'continuation_contract.md':'''# Continuation contract

Initialization is CROSS_RUN_CONTINUATION_FROM_COMMITTED_OUTER3, not old-run resume.
Start with the authoritative StageJournal checkpoint_00423.pt from Job 701387.
The source terminal stage is outer_003/outer_commit. Eight NEW full Outers are
numbered outer_004 through outer_011 in real stage IDs, not relabeled 0-7.
No Outer 0-3 optimizer block is scheduled. Outer3/new-bank baseline is forward-only.
New sampling, source manifest, approval v3, datasets and journal identify a new run.
Default CLI is plan-only and does not construct a ProductionBackend.
Only explicit --execute-approved plus a fully matching approved v3 can train.
No approved=true artifact was created; no Slurm command was executed.
''',
      'cross_run_state_transfer.md':'''# Persistent learning state transfer

The certified source transaction supplies every model, optimizer (including
fc1_conditional), scheduler, AL multiplier/rho, global RNG and SDF runtime counters.
Actual source Adam states: FC1 6728 steps, SDF 12800 steps; global SDF epoch 400.
All optimizer moment tensors and original param-group fields are checked after load.
The Mac's Torch 2.10 adds decoupled_weight_decay metadata to THU Torch 2.5.1 Adam(W)
checkpoints. Only that extra key is permitted, and only if it agrees with the
existing optimizer class. Original LR, parameter order, step, exp_avg and
exp_avg_sq must remain identical. Other changes fail closed. No moments are reset.

P/Q Adam and LBFGS are stage-local in the existing solver; their local optimizer
restart at a new committed block is unchanged production semantics. Episode's
empty policy_value/fc2 optimizer states are copied, not invented as learned state.
The existing production BP stage's own optimizer creation rule is unchanged.

Old backend data, Firm-bank paths, FC1 progress and transaction completion list
are NOT restored into the new mutable backend. Source epoch/learning counters
are restored; previous committed Macro hashes seed the Outer4 boundary check.
No temporary weights-only pseudo-checkpoint is used to initialize production.
The minimal shared setup hook accepts certified model state, then full state
is restored and verified before any stage can execute.
''',
      'sampling_manifest.md':'''# Sampling provenance

8192 Train / 2048 Validation Parent, 200 firms/Parent, two child branches.
4096+4096 and 1024+1024 Beta/Empirical Parent (mixture unit is an economy).
Empirical templates are Job 699894 t0 RECORDED_INITIAL_PARENT_DISTRIBUTION,
not t>=1 on-policy captures. Full original templates: 4096 Train and 1024 Validation.
Each is used once in its own split; b is resampled by the original conditional
CDF-by-capital-quartile rule, not by copying full source firm state.
Template source file SHA f616821c7407230eef850a255ea1205a1df47d3b3e659d1b2820cfd810ebf1d3
was checked equal on Mac and THU. Source paths have no train/validation overlap.
Unchanged supports: alpha [1.8,3.2], beta [4.8,7.8], aggregate K [45,85].
New seed 202610204, new signature and fresh banks; old sampling is not overwritten.
Numeric Parent IDs are scoped by run_id/split/economic_time/branch; dataset
provenance explicitly includes run identity as well as all network/CRN hashes.
''',
      'sample_coverage.md':'# Actual constructed sampling coverage\n\n```json\n'+json.dumps(coverage,indent=2)+'\n```\n\nThese are CPU initial banks, not completed economic simulation/training rows.\n',
      'firm_macro_stage_order.md':'''# Exact production stage order

New-bank Outer3 Validation baseline -> initialize immutable probes (no optimization).
For EACH actual Outer 4..11:
outer_start -> finite-direct firm_bank -> three Firm inner rounds, each with
three _p_policy_outer blocks, three _q_picard_outer blocks and production BP
distillation -> D0 -> FC1 macro0 committed epochs -> D1 -> FC1 macro1 epochs ->
D2 -> SDF TRUE_ONLY chunk0 -> chunk1 -> POST_SDF_VALIDATION_RESIMULATION ->
cross_block -> fixed-function diagnostics -> committed outer_complete.

Existing production functions are called, not surrogate solvers. Firm conditional
banks use CURRENT Macro and verified refreshed Grid target. FC1 uses original
Episode._compute_fc1_only_loss; Signed AiO, wealth recursion and AL are unchanged.
Each module's before/after hashes enforce freezing. FC2/root/FAST guards remain.
Non-convergence is diagnostic-only; nonfinite/illegal required data is a hard error.
Budgets are unchanged apart from eight additional Outers and logical bank size.
''',
      'fresh_shock_rng_contract.md':'''# Fresh shocks and random state

The archived initial state preserves source bank eps [5120,16,1], source pair RNG,
global RNG, epoch=400 and optimizer step=12800. That old bank is not indexed as
8192/2048 new parents and is never doubled/repeated. After full learning-state
restoration, only active bank storage/capacity/key is invalidated for the new data.
At first SDF epoch 400, the unchanged production constructor creates a fresh bank
for current contiguous complete-pair indices (Train then disjoint Validation).
Production key is episode_id/global epoch/capacity/device/dtype, seed follows
the existing base_seed + 10000*episode_id + epoch rule, and pair reseeding uses
the existing seed+7919 rule. New namespace is the locked sampling signature.
Every committed chunk records actual range, key, bank version, epoch and RNG identity.

This is not bitwise continuation of a hypothetical unchanged old data stream.
Once the new run starts, its committed eps/pair/global RNG restore exactly.
CPU test verifies 10240 capacity at epoch400, no repeated-half bank, and exact
pair draws after fresh_state/restore_fresh roundtrip. CUDA RNG bytes are source-
audited but CUDA execution itself is intentionally not run during review.
''',
      'checkpoint_resume_contract.md':'''# New journal and disk retention

Fresh initialization imports certified persistent state into a NEW empty output.
It cannot copy the old latest.json or reuse the old writable journal.
--resume requires the new run's own transactions/latest.json; a nonempty fresh
output, wrong code/data/runtime contract or corrupted pointer fails closed.
P/Q/BP commit at block boundaries; FC1 at full epochs; SDF at chunks.
An uncommitted block restarts from its prior checkpoint; committed IDs never rerun.
Global/source counters are retained; new FC1 patience/data progress is independent.

Retention: KEEP_ALL_TRANSACTIONS_NO_AUTOMATIC_DELETION. About 890 journal files
plus eight completed Outer snapshots, before extra metadata. At current ~3MiB
terminal size this starts around 2.7GiB; growing result histories can increase
that materially. Budget 10-20GiB for checkpoints plus datasets, not a fixed cap.
No pruning is performed, and no Job701387 file is deleted. Future pruning needs
separate review and must preserve an atomically verified latest checkpoint.
''',
      'function_probe_contract.md':'''# Fixed function probes

Formal new-run baseline creates immutable, hash-bound Firm 7D inputs and complete
Validation macro parent/two-child records. No new baseline simulation has been
executed during review. It will run only behind approval v3.
Firm b,z,eta,i,x,hatcf,lnkf are unchanged across Outers; outputs P0/PI/Q and
bp0/bpI plus hard default/investment comparisons are saved. BP functions here are
HEAD functions (explicit label); actual Simulation still executes production Grid.
FC1 fixed (x_t,x_t1,Hatc_cal_t,LnK_cal_t) outputs are physical scale.
Wealth/M uses fixed true Parent and fixed common child x, current FC1 perceived
child, then existing value_model and sdf_model.get_M formula. Raw/used M are
separate and used M follows original Firm clamp. Additional TRUE-state wealth/M
on fixed realized children is separately named; it is not confused with Firm M.

Movement: RMS, MAE, abs P90/P99; binary flips; two-Outer distance; update-direction
cosine (undefined zero update is NOT_AVAILABLE). Distributions include mean,
variance, P90/P99 and clamp fraction. Probe/output hashes fail on changed inputs.
No Jacobian/spectral-radius experiment is introduced. CPU test loads actual
checkpoint and observed old probes, verifies production forward APIs and exact
repeatability/no model or RNG mutation with optimizer.step forbidden. Those
test fixtures are not represented as new-bank baseline or continuation results.
''',
      'economic_metrics_contract.md':'''# Economics and comparability

Reuse finite_direct_outer_metrics.py, Train/Validation x BETA/EMPIRICAL/ALL.
G is the saved normalized prediction residual, not a solved h* root; signed/raw
G scales are retained. Counts and valid fractions always accompany conditional
metrics. Gate-weighted default/investment rates are distinct from hard survivor
counts. No undefined economic label is imputed; failed rows remain in data audit.
FC1 one-child eligibility is not SDF two-child eligibility.
FC1 pre-fit and post-fit on unchanged labels are in fc1_fit_history.csv; D0/D1/D2
and post-SDF resimulation are distinct in outer_macro_economic_history.csv.
LnK level R2, Delta LnK R2, MAE and sample counts remain separate.

Historical Job701387 Outer0-3 metrics stay read-only in historical_baseline.
They have the OLD Validation bank. Only NEW-bank Outer3 baseline vs Outer4-11
share the new fixed Validation and CRN. Never concatenate old means and claim
same-sample improvement. All post-SDF economics come from a dedicated fresh
Validation-only cache with all current network hashes, no optimizer.
''',
    }
    for name,body in contracts.items():atomic_write_bytes(output/name,body.encode())
    common='''REPO_DIR=<THU_CLEAN_REVIEWED_CHECKOUT> REQUIRED_COMMIT=<REVIEWED_COMMIT> \\
RUN_ROOT=/home/fit/zhuyingz/WORK/LiuHao/cachedir/finite_direct_large_continuation_outer4_11_20261020_r1 \\
SAMPLING_DIR=<DEPLOYED_PREPARED>/continuation_sampling \\
INITIAL_CHECKPOINT=<DEPLOYED_PREPARED>/continuation_initial_checkpoint.pt \\
SOURCE_MANIFEST=<DEPLOYED_PREPARED>/continuation_source_manifest.json \\
APPROVAL_FILE=<FUTURE_USER_APPROVED_V3_FILE>'''
    commands='''PLANNED ONLY. Placeholders must be set after user approval; NOT EXECUTED.

Fresh CROSS_RUN_CONTINUATION (new empty run):
'''+common+''' RESUME=0 \\
/rmprog/slurm/v24.05.1/bin/sbatch <THU_CLEAN_REVIEWED_CHECKOUT>/slurm/run_finite_direct_large_continuation_outer4_11_a800.slurm

INTRA_RUN_RESUME (same new run and identical binding):
'''+common+''' RESUME=1 \\
/rmprog/slurm/v24.05.1/bin/sbatch <THU_CLEAN_REVIEWED_CHECKOUT>/slurm/run_finite_direct_large_continuation_outer4_11_a800.slurm

Exact Python CLI (fresh; append --resume only for the new journal):
python analysis/run_finite_direct_large_continuation.py \\
  --sampling-dir <DEPLOYED_PREPARED>/continuation_sampling \\
  --initial-checkpoint <DEPLOYED_PREPARED>/continuation_initial_checkpoint.pt \\
  --source-manifest <DEPLOYED_PREPARED>/continuation_source_manifest.json \\
  --output <NEW_RUN_ROOT> --device cuda:0 \\
  --parent-batch-size 128 --fit-parent-batch-size 128 --firm-batch-size 512 \\
  --execute-approved --approval <FUTURE_USER_APPROVED_V3_FILE>

No approved=true file exists. Original v2 approval is invalid for this run.
'''
    atomic_write_bytes(output/'planned_slurm_command.txt',commands.encode())
    changes=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)
    if not changes.strip():changes=subprocess.check_output(['git','show','--stat','--oneline','HEAD'],cwd=ROOT,text=True)
    atomic_write_bytes(output/'diff_summary.md',('''# Diff summary

New source certifier/learning-state codec, continuation sampler/preparer,
review-gated eight-Outer runner, fixed-input forward probes and histories,
Slurm plan, non-optimization tests, and review docs. Shared modifications are
limited to an optional initial-model injection in setup, optional full-learning
source hook in ProductionBackend, conditional run identity in cache provenance,
and dynamic Outer ticks in existing plots. No training equations or loss changed.

Implementation change set (pending worktree or final committed diff):
```text
'''+changes+'```\n').encode())
    body=f'''# Job 701387 大规模 Continuation：Outer 4–11 代码审查报告

## 中文审查摘要

本轮仅完成实现、CPU 采样与非优化验证，状态为 WAITING_FOR_USER_REVIEW。
未执行任何新的 Firm/FC1/SDF optimizer step，未提交 Slurm，未生成正式批准，未 push/merge。

1. **源状态已核验**：THU 的最后事务真实存在，指针与 SHA256 正确；来源确为
   Job701387 / commit160a4d1 / outer_003/outer_commit，五个模型实际哈希一致。
   StageJournal 与 outer_complete 内的学习状态逐项一致，不是只复制历史字符串。
2. **完整续训而非 weights-only**：FC1 动量/步数6728，SDF 动量/步数12800、epoch400，
   scheduler、AL、全局/配对 RNG 和旧 shock bank 完整提取。旧数据路径、缓存、patience
   和旧 completed 列表不作为新 run 的可写状态继承。P/Q 的块内局部优化器仍按生产规则创建。
3. **新采样已真实生成**：Train8192、Validation2048，每 Parent200家公司，Beta/Empirical
   严格50:50。原 t0 模板4096/1024全部可读且无跨 split 来源泄漏，模板复用与CDF fallback均为0。
   这仍是 RECORDED_INITIAL_PARENT_DISTRIBUTION，不能称为 t≥1 on-policy 模板。
4. **真正的新阶段编号**：先在获批后的新 run 内做无优化 Outer3/new-bank Validation baseline，
   然后从 Outer4 Firm 开始直到 Outer11；不会重新训练0–3，也不会从Outer2挑选最优权重。
   Firm→D0→FC1→D1→FC1→D2→SDF→Post-SDF Validation→cross-block顺序不变。
5. **经济方程和预算不变**：P/Q/BP、FC1 loss、Signed AiO/Wealth/AL均调用原实现；
   继续 Finite Direct，无 h* / FAST / Boundary-aware / FC2。只扩充逻辑Parent规模和新增八轮。
6. **固定函数评价已实现**：P0/PI/Q/BP头、FC1物理输出、真实Wealth/M公式及clamp分别保存；
   记录相邻/隔轮movement、翻转与方向cosine。BP头函数不冒充实际Grid执行动作。
7. **比较口径分开**：旧Job0–3留在历史目录；新银行Outer3 baseline与4–11共用新Validation/CRN。
   Post-SDF评价独立于D2缓存；失败覆盖率、完整Pair率随条件指标同时保存。
8. **审批与恢复已接通**：Approval v3绑定新commit/plan、采样、source manifest、完整续训状态及
   源Job/Outer/stage/checkpoint SHA；新StageJournal恢复已提交阶段，不重复优化。
9. **当前不代表GPU验收或训练完成**：新baseline和未来八轮尚未执行。真实A800运行、CUDA RNG
   流及正式经济结果必须在用户明确批准后检验。本轮不制造新的训练曲线。

测试结果：{test_result}

以下为详细接口、证据与运行计划；同目录 acceptance.json 为机器可读验收，
planned_slurm_command.txt 为尚未执行的真实 CLI 计划。初始状态和新采样另存于 prepared，
不纳入源码 commit；最终本地 commit 后会刷新 prepared manifest 的 new_code_commit。

Generated {datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()} (Asia/Shanghai).

## Status

Implementation and non-optimization verification only. CURRENT_PHASE =
WAITING_FOR_USER_REVIEW. No sbatch/srun, GPU training, real optimizer step,
automatic approval, push or merge occurred. This is not eight completed Outers.

## Source checkpoint (P0 completed first)

THU source exists. latest.json points to transactions/checkpoint_00423.pt.
Actual SHA256: {source['source_terminal_checkpoint_sha256']}.
Source commit is 160a4d18ab410dceb75b46f014dc33b5296526a2; terminal stage is
outer_003/outer_commit; completed indices are [0,1,2,3]. Models were loaded with
weights_only=True and a narrow numeric NumPy allowlist. No unsafe pickle fallback.
Five actual component hashes equal final_summary. Outer complete and journal
model/optimizer/AL/RNG/backend state digests agree despite different envelopes.
See source_checkpoint_audit.json and source_optimizer_al_audit.json for full hashes.

Persistent FC1 AdamW step=6728, SDF step=12800, SDF epoch=400. AL:
{json.dumps(source['source_al_state'],sort_keys=True)}.
Schedulers and RNG (global, paired and fresh bank) are complete. P/Q stage-local
optimizers follow existing block initialization; unused Episode optimizers remain
empty. No fake momentum is fabricated.

## Actual preparation

New CPU Parent banks exist: 8192 Train / 2048 Validation, each 200 firms.
Beta/Empirical 4096/4096 and 1024/1024. Unique templates 4096/1024, reuse=0,
conditional fallback=0, source leakage=0. Original Job699894 t0 source, not on-policy.
Template SHA matches original THU artifact. Supports and economic initialization
are unchanged. New sampling signature: {signature['fingerprint']}.
Continuation initial file SHA: {source['continuation_initial_checkpoint_sha256']}.
Original run and old sampling remain unchanged.

## Implementation and execution boundary

Explicit cross-run state extraction retains models, optimizers, scheduler/AL,
RNG and learning counters, rejects old mutable dataset/stage paths. New-run
resume uses a separate journal and strict current code/state/data contract.
Stage IDs really begin at Outer4 Firm after the no-optimization Outer3/new-bank
Validation baseline. Every outer reuses production P/Q/BP, D0→FC1→D1→FC1→D2→
SDF two chunks→post-SDF Validation→cross-block→commit. See stage/order contract.
Logical sample count is not GPU batch size. A800 plan is 1 GPU, 8CPU,128GB,7days,
with physical batches128/128/512. No GPU benchmark or production execution here.

New baseline/probes will be formed inside the approved run from its Validation
bank. Their interfaces are implemented and forward-tested, but no new-bank
macro baseline or future training curve is falsely supplied in this review.
Current validation is source/state transfer + CPU bank construction + targeted
unit/forward tests, not an end-to-end A800 training acceptance test.

## Tests and fixes found during local review

{test_result}

Additional finalization checks: bash -n for planned Slurm, plan-only CLI,
git diff --check. First stricter CPU restore check exposed Torch2.10's added
AdamW decoupled_weight_decay metadata versus verified THU source Torch2.5.1; original moments,
step counters and all old group fields matched exactly. The loader now checks
those exactly and admits only this class-consistent metadata key. It does not
relax state matching generally. Baseline-only CSV/plot paths were also tested
before any FC1/SDF result exists, fixing an absent-column bug.

CUDA RNG source bytes and envelopes match; source CUDA streams cannot be
executed on this Mac. Formal CUDA resume remains a post-approval runtime check.
No economic non-convergence threshold was relaxed. Raw M is not the unused
SDF MLP output; probes use actual wealth/FC1/SDF formula and separate used clamp.

## Histories, plots and interpretation

See function_probe_contract.md and economic_metrics_contract.md. RMS/MAE/P90/P99,
flips, two-Outer distance/cosine use immutable inputs. Actual M/Wealth and FC1
physical outputs have dedicated histories. Original economics and source grouping
remain in the existing metric module. Post-SDF Validation cannot reuse D2.
Historical old-bank Outer0-3 is separated from new-bank Outer3 baseline→4..11.
Default/investment gate rates are not hard survivor counts. Missing values remain
NOT_AVAILABLE; only real future observations generate training curves.

## Approval and remaining human action

approval_v3_schema.json is a schema, not approval. Future approval must bind new
clean commit/plan, new bank signature+file SHA, continuation initial+source manifest
SHA and source Job701387/Outer3/stage/checkpoint, start4/end11 and8192/2048.
It is path-independent and cannot reuse old v2 authorization. Commands are prepared
only, with deployment placeholders; see planned_slurm_command.txt.

The single local implementation commit is the repository HEAD returned at delivery.
Prepared artifact manifests are refreshed after that commit to record its exact
SHA; approval must use those finalized artifacts. No file embeds its own commit
hash recursively. No automatically approved artifact is generated.

## Acceptance

```json
{json.dumps(acceptance,indent=2)}
```
'''
    atomic_write_bytes(output/'implementation_report.md',body.encode())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--test-result',required=True)
    a=p.parse_args();write_review(a.prepared,a.output,a.test_result)
    # No commit embeds its own SHA recursively; finalized delivery artifacts
    # record clean HEAD after the single implementation commit.
