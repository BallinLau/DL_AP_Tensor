# Sampling Provenance

正式Runner调用 `load_locked_sampling()`，读取现有签名、模板bank、两份schedule和initial bank；
不是只在prepare阶段调用新sampler。逐文件SHA、模板fingerprint、固定schedule及bank tensor hash必须匹配。

4096 Train=2048 Beta+2048 Empirical；1024 Validation=512+512，每Parent200企业。
Beta支持保持alpha[1.8,3.2]、beta[4.8,7.8]；同Parent不混拼两类企业。
Empirical来源仍为Job699894的真实t0 pre-decision bank。
标准语义 `RECORDED_INITIAL_PARENT_DISTRIBUTION`，不是t≥1内生分布。
现有旧包的原始verbose source label不篡改，新代码生成使用标准label。

资本四分位条件CDF；不足8条回退模板总体CDF，记录fallback；无jitter。
新Parent资本和其余合法初始状态独立生成；保留全局Parent/Firm identity，不整组复制来源企业。
Train/Validation来源path与template不重叠；当前选中2048/512个不同模板，复用率0。
b–K只保留条件分组关系，不声称完整保留copula或b–z/b–eta依赖。

默认capture现成t≥1输入状态，增加source outer/path/time/branch、b/K/z/eta与模型hash，
无额外模型执行，不筛选未来成功child；不自动转换成新模板、不更改已提交Parent bank。
后续内生模板若启用必须仅Train来源、独立版本fingerprint、固定Validation不变；本次不启用该升级。
