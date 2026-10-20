# Checkpoint来源：本地认证 PASS，服务器未重验

Source Job701387，原代码160a4d18ab410dceb75b46f014dc33b5296526a2，terminal Outer3。
原服务器路径（仅身份说明，本轮未连接）：
/home/fit/zhuyingz/WORK/LiuHao/cachedir/full_nested_finite_direct_outer4_20261020_160a4d1_r1

本地认证 actual transactions/latest.json→checkpoint_00423.pt，SHA6899b452c14a04a3a3b1837a8903dbb21249ec08d534293ef50046f99f2bc585；outer_complete SHA2ba3ef5a859df102e1c7b33923885406cb357de564861a0b61a527f8acaab369。真实learning/optimizer/AL/fresh内容一致。

现有候选：

| 内容 | 当前本地准确SHA |
|---|---|
| continuation_initial_checkpoint.pt | 6ff6b2979735734400eafc9149efe5caf969348bba6d0dcbffa95f10686ec6ad |
| continuation_source_manifest.json | 2807bbd6aee623c5e880d48d9e2968adbf0404ee5370073179159eb35785aa5e |
| sampling_signature.json文件 | f08cdec937092c44d747da9cef7791d473133942f6a1a4458dc9b72dafc94db1 |
| train_initial_bank.pt | bfc2c57de3ced55c13a1dc9f492db3715b201ac6bb942bd127903adf8b924222 |
| validation_initial_bank.pt | dbc5d6b2de8a9234335a1a85a97dd148360a72d0ac2b4824e11ed5742ba24263 |

Sampling fingerprint：2a730d551131c6478fdb12ffb718d0a102082b3a370546606ef4b0356e41bd54。

现有manifest仍绑定788fad7；initial_dynamic_pool.pt不存在。以上是已核验候选字节，不是最终新部署的批准清单。不得把旧manifest简单当作新训练就绪。

认证包含Firm/FC1-C/K/SDF/Wealth参数、Persistent FC1 16states/step6728、SDF8states/step12800、scheduler、AL、Python/NumPy/Torch/CUDA RNG、fresh bank/pair RNG。SDF epoch400，lambda_mean_low0.07104423291275452、其他lambda0、rho2。完整精确指纹在source_checkpoint_audit.json。

未执行CUDA RNG流或GPU恢复测试，不能把CPU认证写成服务器训练恢复完成。未来应在新目录重新prepare并核对本报告候选与实际部署产物SHA；任何SHA变化都必须反映在新v4批准中。旧结果不是新operator的收敛证据。
