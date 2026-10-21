# Checkpoint 存储实测

对同一个真实 Outer11 learning state，分别序列化旧 h50_train_bank inline 结构和新 reference 结构，未构造或更新 optimizer。

| 项 | 实际 bytes |
|---|---:|
| 源完整 learning checkpoint | 12,821,336 |
| learning + inline Train Bank | 160,239,905 |
| learning + Train Bank reference | 12,822,117 |
| 单次避免重复写 | 147,417,788 |
| Bank+schedule 不可变包装，一次/版本 | 150,097,309 |

拟议 H50 有449个阶段；同体积假设下可避免约66.19GB重复写，尚未扣除每轮新Bank一次性文件与新增历史。不是未来磁盘总量的保证。

参考路径携带实际 file SHA、typed digest、Outer、Bank/Pool版本；篡改和错误Outer测试拒绝。原 StageJournal 原子提交/恢复未改。仍有模型、optimizer、fresh shock bank、history/probe等主要存储，不宣称整体checkpoint全部变小到零成本。

测量输出 JSON 为 checkpoint_storage_audit.json；测量用的大 pt 不进入审查 ZIP。
