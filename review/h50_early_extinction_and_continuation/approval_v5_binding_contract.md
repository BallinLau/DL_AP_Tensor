# V5 新绑定：不是批准文件

schema_name 保留 CONTINUATION_APPROVAL_V5_HSTAR_H50；新增 continuation_contract=H50_CERTIFIED_OUTER11_AND_IMMUTABLE_BANK_V2。plan fingerprint 因来源、Bank 初始化、storage/sampling contract 改变。

必须绑定：新代码 commit、完整 plan fingerprint、sampling signature、locked 输入文件 SHA、source checkpoint SHA、最终 journal SHA、latest pointer SHA、source_job=702636、Outer11 pool/bank/schedule 三项 SHA、root config fingerprint、horizon=50、operator。

不同 Mac/HPC 路径可通过 source-artifact-dir 映射同名 source 文件；不能绕过内容 SHA。新目录内容不匹配时认证拒绝。旧 V5 缺少新字段/旧 plan 不能复用。旧 v4/v3 同样拒绝。

默认 plan-only；仅显式 execute-approved + 用户未来提供的有效批准才可能构造生产 backend。此轮没有创建 approved=true 文件，没有调用 execute-approved，没有提交模板 Slurm。
