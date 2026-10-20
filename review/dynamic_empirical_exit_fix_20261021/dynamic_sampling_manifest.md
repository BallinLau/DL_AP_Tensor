# 动态抽样清单

Train8192 =4096固定设计Beta+4096动态Empirical；Validation2048=1024/1024固定。每Parent200初始企业槽位，两个宏观Child。

只改Empirical的b初始化；资本与其他冲击字段来自锁定的生产初始化，不复制历史完整企业状态。新 schedule记录parent_id、outer、template version/id、seed、source cluster/branch、Recent/Archive分类。

每轮文件位于 dynamic/outer_00k/：pool_<内容摘要>.pt、bank_<摘要>.pt、schedule_<摘要>.pt、bundle_<pool fingerprint>.json。bundle包含文件SHA，不以绝对路径作为内容身份。

宏观缓存同时绑定initial sampling signature、当前bank hash/version/pool、outer/stage、模型hash、经济配置与固定CRN。相同ParentID不表示跨Outer状态相同。
