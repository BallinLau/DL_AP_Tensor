# Approval v4 与留档

实际入口只接受 CONTINUATION_APPROVAL_V4_DYNAMIC_EMPIRICAL / schema_version4。缺少显式--execute-approved默认plan-only。旧v3即使approved字段为true也在读取checkpoint/构造backend前被拒绝。

新留档为metadata/approval_v4.json，原字节复制。Journal contract保存完整approval_binding。v3基础构造器只用于复用原有内容哈希字段，dynamic包装器覆盖schema并加入新字段；其实现未被本次改变。

必须严格匹配：exact新commit、plan fingerprint、initial checkpoint SHA、source manifest SHA、原Job701387 terminal SHA/Outer3/stage、sampling signature fingerprint及文件SHA、算法version/config/seed、initial pool SHA、initial Train SHA、固定Validation SHA、operator v2、50:50与70:30、Outer4–11及8192/2048规模。

未来pool不要求预知SHA，但算法/来源/seed规则已锁定，每轮journal提交实际SHA。不同pool/bank/CRN恢复必须拒绝。

本次没有创建任何approved=true文件。future批准必须在部署实际artifact重新哈希后另行获得，不能借用7074dca、788fad7、旧Job或本地测试对象的授权。
