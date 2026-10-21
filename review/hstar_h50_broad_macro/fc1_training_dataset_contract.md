# FC1 H50数据
X=(x_t,x_child,h_cal_t,lnK_cal_t)；Y=(h_cal_child,lnK_cal_child)。
沿用select_training_rows；Parent失效的edge不会被Child成功洗白。
所有原始Child记录按时间分shard保存，训练索引仅从合法记录产生。
拟预算Train8192、Validation2048个parent-time上限，每阶段锁定subset；不是把409600 pairs全量送进每epoch。
非空Early/Middle/Late分配显式quota，再按lineage/macro-bin加权reservoir；空时间层不补造数据。
R²/MAE按时间与二维宏观状态保存。总体有效率按8192*50或2048*50计算，不复用一期分母。
Train/Validation按初始lineage隔离；固定Validation起点及CRN，允许模型改变造成有效轨迹变化。
