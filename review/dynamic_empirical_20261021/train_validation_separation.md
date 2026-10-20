# Train / Validation 隔离

本次未修改任何 Bank。当前固定 Validation2048、Parent ID、CRN 保持原状。

未来 pool 只接受 Train lineage，入口检查与 archive/recent 合并均须拒绝 Validation。不能通过验证表现选择权重，不能从 POST_SDF_VALIDATION_RESIMULATION 提取训练模板。

固定 Validation 指标可跨 Outer 比较；变化的 Train 指标需携带 bank/pool hash，不能单凭其改善认定网络改善。未来动态泄漏与固定 Validation hash 测试均 NOT_RUN，不虚报通过。
