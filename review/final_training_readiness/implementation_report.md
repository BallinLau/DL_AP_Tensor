# 7074dca 最终修复与 Outer4–11 训练准备

审查日期：2026-10-21（Asia/Shanghai）。基线：7074dca9b4e79dee52cdaacce8f3442315b40a11。
本报告和修正同一次本地提交；最终完整 SHA 见交付清单。没有训练、GPU测试、服务器访问、Slurm、批准实例、push或merge。

## 1. 本次最小修复

`analysis/run_finite_direct_large_continuation.py:main/history_init` 的批准留档路径由 `metadata/approval_v3.json` 改为 `metadata/approval_v4.json`。复制批准输入的原始字节，不重写JSON，不改变内容哈希与严格匹配机制。

新增测试直接提取并执行真实的嵌套 history_init 函数，而不是复制一段测试专用的 writer。测试只使用 approved=false 输入，验证v4留档字节一致且不生成v3路径；它不调用生产 backend 或优化器。

Slurm模板只增加留档位置注释，GPU/资源/CLI/预算不变。README同步迁移说明。

发现另一个历史引用：`analysis/build_finite_direct_continuation_review.py` 是旧v3报告生成器，它虽然不在正式Runner/准备链中，却导入当前training_plan，直接调用会为动态v4计划输出旧v3材料。现增加入口保护：遇到动态计划立即拒绝，不创建输出目录、不覆盖任何历史审计文件。没有重写其历史报告正文。

`analysis/finite_direct_continuation_state.py:approval_binding` 中保留的v3字段构造器是历史/共享基础；当前Runner明确导入dynamic_approval的v4包装器并只接受schema_version=4。保留这个共享构造器不等于允许v3执行。所有历史Job和旧review目录未修改。

## 2. 初始化与来源核验

仅对本地已保存的真实Job701387副本和现有初始化副本重新运行CPU认证。不是服务器实时核验，未重新部署。

- 原committed checkpoint：transactions/checkpoint_00423.pt，SHA `6899b452c14a04a3a3b1837a8903dbb21249ec08d534293ef50046f99f2bc585`。
- 终点：outer_003/outer_commit，completed=[0,1,2,3]。
- Journal checkpoint 与 outer_003/outer_complete.pt 的 transaction、fresh、state 内容一致。
- 初始完整learning与从真实终点重新提取的learning typed digest一致，不是仅权重初始化。
- Firm、FC1-C/K、SDF、Wealth 参数哈希全部通过。Persistent FC1 optimizer有16个state、step6728；SDF optimizer有8个state、step12800。
- SDF global epoch400，AL lambda_mean_low=0.07104423291275452，其余lambda为0，rho=2；scheduler、Python/NumPy/Torch/CUDA RNG及fresh shock/pair RNG均有完整指纹。
- Episode中未使用的policy_value/fc2 optimizer为空是原状态；P/Q优化器原本为stage-local，不能伪造旧momentum。

完整准确SHA与审计见 source_checkpoint_audit.json / checkpoint_source_audit.md。

现有候选初始化SHA是 `6ff6b2979735734400eafc9149efe5caf969348bba6d0dcbffa95f10686ec6ad`；其配套manifest仍绑定788fad7，且没有新initial_dynamic_pool.pt。因此**这套现有文件不能冒充已完成最终v4部署**。未来必须在新目录用最终提交重新准备、记录实际产物SHA，再申请新的内容绑定批准。

## 3. 新operator与动态契约

算子保持 `finite_direct_parent_exit_before_child_v2`，本次没有修改经济转移或采样算法。

Parent完整账本 → 保存原pre-transition Entry reference/资本桥接历史 → 原eta_t融资与bar_i资本/冲击展开 → 调用原Exit移除上一期default → Child Entry一次 → realized-b pre-decision capture → Child当前政策。

Outer4使用已锁定历史t0初始分布。之后每轮post-SDF Train-only source → EMPIRICAL_POOL_COMMIT → 下一轮outer_start才激活新Bank。Firm source与D0/D1/D2在同轮共用Bank。Train8192、Validation2048、每Parent200 slots、2 children、Beta/Empirical50:50；4096 Empirical精确2867Recent+1229Archive，Parent-lineage cluster权重；Validation/CRN固定。

模板用已按eta_t实现的b，不用head/target替代，不按当前Child事后成功筛选。旧operator v1缓存会因operator/Bank/Pool/模型/CRN来源不匹配被拒绝。

旧Job701387只是完整学习状态初始化与历史对照，不能直接证明修正operator下的经济收敛。下一次新run必须重新评价baseline、重建模拟数据。

## 4. 预算与执行边界

Outer4–11共8轮；Firm Inner3、Macro Inner2；P Adam200/LBFGS100、Q Picard3/fit300、BP1000；FC1每Macro Inner25–40（patience10）；SDF2×50=100epoch/Outer。物理批量Parent128、fit128、Firm512。

正式硬件仍为1×A80080GB。无Root Search、FAST、Boundary-aware、FC2、新消费/存活下界；P/Q、FC1 loss、SDF/Wealth、Signed AiO/AL和账本未改。

## 5. 断点恢复和授权

StageJournal本次零修改。恢复模型/optimizer/scheduler/AL/RNG/fresh和当前Outer/stage、Bank/Pool引用与数据来源。动态文件内容SHA和版本必须一致；已提交stage不重执行，未提交stage按RESTART_UNCOMMITTED_BLOCK重放。测试覆盖真实pool方法和Journal恢复，而非只检查字符串。

当前Runner先拒绝未批准/旧schema，再严格核对v4绑定。改变checkpoint、Sampling Signature、Pool/Train/Validation、source manifest或plan均使旧批准无效。新代码commit也必须重新批准。metadata/approval_v4.json是留档，不是自动批准机制。

## 6. 状态分层

最终定向CPU回归：191 passed、0 skip/xfail，30.47秒；1条既有测试故意触发的BP-head override warning。bash -n、真实Slurm参数与argparse一致性、git diff --check及review JSON解析均PASS。详细命令与范围见test_results.txt，机器结果见test_results.xml。

| 状态 | 本次结果 |
|---|---|
| 代码与本地CPU回归准备 | 完成，等待用户审查 |
| 本地真实source副本认证 | PASS |
| 本轮服务器artifact实时核验/部署 | 未执行 |
| 最终新commit的prepared/v4 Pool与SHA | 待未来部署准备 |
| 用户正式训练批准/approved实例 | 无 |
| Slurm/优化器/正式训练/GPU性能测试 | 未执行 |

未来命令仅见 planned_slurm_command.txt，**不得在本次执行**。最终停止状态 WAITING_FOR_USER_REVIEW。
