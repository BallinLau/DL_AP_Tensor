# 动态Empirical H50
Train Beta/Empirical=4096/4096不变；Recent/Archive=2867/1229。
源是POST_SDF_TRAIN_EMPIRICAL_SOURCE，Child Entry之后、当前Child policy之前的realized b/K。
模板按lineage、economic time、branch保存。只从已提交source manifest读入，验证每个shard hash。
Recent=上一Outer；Archive=锁定初始Train模板+更早已提交H50来源。抽样先lineage、再time、再可用snapshot，保留资本分位CDF。
不使用Validation、不按当前未来消费或违约成败选择模板，不使用bp head。
Train下一轮仅更新b/bp，其他初始微观字段及资本仍按原合同；宏观宽覆盖来自H50，不是假装初始bank已变为全状态动态模板。
