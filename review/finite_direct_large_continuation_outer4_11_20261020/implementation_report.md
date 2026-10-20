# Job 701387 大规模 Continuation：Outer 4–11 代码审查报告

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

测试结果：98 passed in targeted non-optimization suite; zero skipped. Exact pytest duration and fixture records are in test_results.xml. Suite covers continuation, outer4_runner/resume, approval_metrics, macro_contract, firm_bank, macro_production_adapter, sampling and review_preparation. Actual source/optimizer/AL and CPU forward-only fixtures exercised. No real optimizer steps, no Slurm. bash -n and git diff --check PASS.

以下为详细接口、证据与运行计划；同目录 acceptance.json 为机器可读验收，
planned_slurm_command.txt 为尚未执行的真实 CLI 计划。初始状态和新采样另存于 prepared，
不纳入源码 commit；最终本地 commit 后会刷新 prepared manifest 的 new_code_commit。

Generated 2026-10-20T21:34:26.403034+08:00 (Asia/Shanghai).

## Status

Implementation and non-optimization verification only. CURRENT_PHASE =
WAITING_FOR_USER_REVIEW. No sbatch/srun, GPU training, real optimizer step,
automatic approval, push or merge occurred. This is not eight completed Outers.

## Source checkpoint (P0 completed first)

THU source exists. latest.json points to transactions/checkpoint_00423.pt.
Actual SHA256: 6899b452c14a04a3a3b1837a8903dbb21249ec08d534293ef50046f99f2bc585.
Source commit is 160a4d18ab410dceb75b46f014dc33b5296526a2; terminal stage is
outer_003/outer_commit; completed indices are [0,1,2,3]. Models were loaded with
weights_only=True and a narrow numeric NumPy allowlist. No unsafe pickle fallback.
Five actual component hashes equal final_summary. Outer complete and journal
model/optimizer/AL/RNG/backend state digests agree despite different envelopes.
See source_checkpoint_audit.json and source_optimizer_al_audit.json for full hashes.

Persistent FC1 AdamW step=6728, SDF step=12800, SDF epoch=400. AL:
{"lambda_mean_high": 0.0, "lambda_mean_low": 0.07104423291275452, "lambda_var_high": 0.0, "rho": 2.0}.
Schedulers and RNG (global, paired and fresh bank) are complete. P/Q stage-local
optimizers follow existing block initialization; unused Episode optimizers remain
empty. No fake momentum is fabricated.

## Actual preparation

New CPU Parent banks exist: 8192 Train / 2048 Validation, each 200 firms.
Beta/Empirical 4096/4096 and 1024/1024. Unique templates 4096/1024, reuse=0,
conditional fallback=0, source leakage=0. Original Job699894 t0 source, not on-policy.
Template SHA matches original THU artifact. Supports and economic initialization
are unchanged. New sampling signature: 2a730d551131c6478fdb12ffb718d0a102082b3a370546606ef4b0356e41bd54.
Continuation initial file SHA: 6ff6b2979735734400eafc9149efe5caf969348bba6d0dcbffa95f10686ec6ad.
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

98 passed in targeted non-optimization suite; zero skipped. Exact pytest duration and fixture records are in test_results.xml. Suite covers continuation, outer4_runner/resume, approval_metrics, macro_contract, firm_bank, macro_production_adapter, sampling and review_preparation. Actual source/optimizer/AL and CPU forward-only fixtures exercised. No real optimizer steps, no Slurm. bash -n and git diff --check PASS.

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
{
  "SOURCE_JOB": 701387,
  "SOURCE_OUTER": 3,
  "SOURCE_CHECKPOINT_VERIFIED": true,
  "SOURCE_MODEL_HASHES_VERIFIED": true,
  "SOURCE_OPTIMIZER_STATE_COMPLETE": true,
  "SOURCE_AL_STATE_COMPLETE": true,
  "CROSS_RUN_CONTINUATION_READY": true,
  "NEW_TRAIN_PARENT_COUNT": 8192,
  "NEW_VALIDATION_PARENT_COUNT": 2048,
  "BETA_EMPIRICAL_RATIO": "50:50",
  "OUTER_START": 4,
  "OUTER_END": 11,
  "OUTERS_TO_EXECUTE": 8,
  "FIRM_PRODUCTION_CONNECTED": true,
  "FC1_PRODUCTION_CONNECTED": true,
  "SDF_PRODUCTION_CONNECTED": true,
  "FUNCTION_MOVEMENT_READY": true,
  "ECONOMIC_METRICS_READY": true,
  "INTRA_RUN_RESUME_TEST": "PASS",
  "APPROVAL_V3_READY": true,
  "FORMAL_TRAINING_STARTED": false,
  "SLURM_JOB_SUBMITTED": false,
  "CURRENT_PHASE": "WAITING_FOR_USER_REVIEW",
  "OUTER3_NEW_BANK_BASELINE_EXECUTED": false,
  "GPU_EXECUTION_TESTED": false,
  "READY_MEANS": "IMPLEMENTED_FOR_REVIEW_NOT_GPU_TRAINING_COMPLETION",
  "ECONOMIC_EQUATIONS_CHANGED": false,
  "ECONOMIC_CALIBRATION_CHANGED": false,
  "local_tests": "98 passed in targeted non-optimization suite; zero skipped. Exact pytest duration and fixture records are in test_results.xml. Suite covers continuation, outer4_runner/resume, approval_metrics, macro_contract, firm_bank, macro_production_adapter, sampling and review_preparation. Actual source/optimizer/AL and CPU forward-only fixtures exercised. No real optimizer steps, no Slurm. bash -n and git diff --check PASS.",
  "source_checkpoint_sha256": "6899b452c14a04a3a3b1837a8903dbb21249ec08d534293ef50046f99f2bc585",
  "sampling_signature": "2a730d551131c6478fdb12ffb718d0a102082b3a370546606ef4b0356e41bd54"
}
```
