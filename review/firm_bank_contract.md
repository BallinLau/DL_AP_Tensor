# Firm Conditional Bank

- 原始7列严格来自同一生产节点：b,z,eta,i,x,hatcf,lnkf。
- context附带同节点聚合账本的hatc_cal/lnk_cal；不把forecast列改名为true列。
- identity保存path、firm_id、source time/branch、当前Firm/Macro hash。
- capture保存所有已有live输入行，当前节点是否合法另存eligible；不根据未来child筛选。
- conditional training当前取t0合法Parent，t≥1捕获仅为后续内生模板研究留证，不自动替换训练设计。
- Train与Validation原始path交集为空；抽样seed不同不构成独立性的替代证明。
- z/x/i冲击复用ConvergenceShockBank与production exogenous bundle；eta保留production exact expansion。
- raw M另存，P/Q共同used M为当前Macro、当前true-context下的raw M经原P common clamp处理。
- P/Q/bank含Macro hash，Macro更新后不能把旧bank当作新bank。
- BP adapter附加true macro context让Episode读取child M，不触发parent仅7列时的M=1逻辑。
- Firm期间Macro/FC2冻结；BP不改P/Q；P不改Q/BP；Q不改P/BP。
- 不保留旧Root-cleared bank，不人为拼接来自不同经济节点的微观/宏观列。
