# 真实初始根可行性
来源：Job702636 Outer11；固定Validation前4条，200企业/Parent。
真实模型，不是合成policy；CPU Grid使用既有 require_cuda=False reference选项，保留生产shock seed和child count。
4/4严格初始根通过真实账本回放；大致 h*=[-0.955403,-0.807478,-1.070035,-0.989166]。
这是小样本PASS，不是8192 Parent总体可行性估计；没有运行A800。
完整root audit及生成源码SHA见 real_cpu_probe_entry_complete/，后续最终复核输出以交付说明为准。
先前少量CPU总流程约5.7秒（包含原模型已迅速失效的50期计划后缀），不能外推为全存活50期吞吐。
Mean/root solve time、operator evaluations：见原始initial_hstar_audit.csv；不把batch全局评估次数误称每Parent独立次数。
REAL_ROOT_FEASIBILITY=PASS_SMALL_CPU_SAMPLE
FULL_BANK_ROOT_FEASIBILITY=NOT_MEASURED
GPU_PERFORMANCE=NOT_MEASURED
