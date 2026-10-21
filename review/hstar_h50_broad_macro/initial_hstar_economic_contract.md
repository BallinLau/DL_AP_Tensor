# 初始严格 h-star
发生在每次数据阶段 t0：初始企业横截面/共同冲击已锁定、尚未执行 t0政策。t0维持不Entry。
K0=sum(alive * firm_mass * K)，lnkf=log K0。只求h，不求二维(h,lnK)。
复用 production STRICT_GLOBAL；G=exp(h)*Kledger-Craw。10 coarse、20 refinement、4 rounds、normalized tolerance1e-5。
候选根必须重新调用完整经济算子；求根终点插值只提案，不是接受证据。Production process_node 独立回放 K/C、G、lnK，并执行适用的真实Grid BP。
无根/跳跃/非有限值会保留 controller failure/root records，停止当前数据阶段，不替换随机Parent、不选最小残差冒充根。
缓存绑定初始state全tensor hash、所有相关模型hash、经济配置、CRN、root config、adapter source SHA。改变模型/配置拒绝缓存。
t>=1只有FC1预测，无root、FAST或boundary-aware调用。
当前t0账本有定义但政策后零存活时，可以尝试原下一期Entry；训练eligibility不因此被改成true。
