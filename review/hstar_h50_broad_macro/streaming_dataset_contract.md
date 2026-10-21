# Streaming store
一个physical batch×一个economic time一个atomic torch shard；内容含rows、compact captures、templates和下一主干state。
manifest保存所有shard SHA；sidecar只发布完整atomic文件。若atomic rename后、sidecar前中断，需safe-load和精确provenance/row count核验才恢复sidecar。
没有一次性819200行大GPU张量，也不构建全部企业历史DataFrame。
ShardStore逐个读取并核验；Macro subset bounded reservoir；Firm两遍索引采样。
raw失败行和计划无效后缀都保留，不补零消费/资本/log。
