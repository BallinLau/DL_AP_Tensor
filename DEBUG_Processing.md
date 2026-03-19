## Debug 过程文件

### episode.py

- episode 0 
  - [X] sdf \& FC1 训练检查
    - [X] 数据生成模块检查  
      - 多个branches（>=2）
      - 每一个path 生成的公司数量比较少（n_company = 2 左右）
      - 时间戳为 \(t - t+1^{0}...t+1^{branch}\)
      - batch 生成 为 parent - children
    - [X] loss 计算模块检查
      - 该阶段有SDF的AIOloss -> 目前没有计算AIO，而是把每个child的loss算出来，然后求Norm1
      - SDF的一阶矩和二阶矩loss，没有FC1相关loss
  - [X] policy \& value 训练检查
    - [X] 数据生成模块检查
      - 生成的数据需要包含 b z eta i x hatcf lnkf
      - parent 的 是随机抽样
      - children 是基于 parent 变化，注意 b 怎么考虑的
      - AI: 1，判断一下 _predict_macro 与 现在的 sdf_fc1.py 里的模型是不是兼容的，如果不兼容，以sdf_fc1.py为基准进行修改；（判断没问题）
    - [ ] loss 模块检查
  - [ ] 训练测试
    - [ ] 写一个ipynb文件，查看FC2 训练使用的数据，以及观察该数据是否能够用于对SDF FC1 的二次训练？ 训练流程参考/DL-AP/experiments/run_multi_episode.py，也就是说查看当episode = 0 时，训练流程走到FC2的时候，这里生成的数据格式以及该数据是否能用于训练SDF FC1 （能否用于代替self.df_sdf = sampler.build_sdf_fc1_df()来训练sdffc1？）
      - 不能直接用，1，FC2产生的df，branch 从 -1 开始，而 sdf 从0开始，这一点需要统一 2， 在FC2之后，训练sfcfc1需要有真实的lnk 和hatc 这一点也需要在sdfloss中标记一下
      - FC2要想输出能进入sdf的训练，需要确保
    - [ ] sample class 在 simulate 模式下，生成的数据有问题，详细检查一下，目前觉得的问题时 bari，也就是K似乎没有更新？