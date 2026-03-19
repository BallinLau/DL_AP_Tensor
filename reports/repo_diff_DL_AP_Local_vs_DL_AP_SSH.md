# 仓库差异报告：DL_AP_Local vs DL_AP_SSH

生成时间：2026-03-11 17:18:56 CST

## 1. 对比对象
- A（本地仓库快照）: `BallinLau/DL_AP_Local` @ `origin/main`（2d9009a6aaf95ca41bc24bf0910db09acd87fd19）
- B（服务器仓库快照）: `BallinLau/DL_AP_SSH` @ `sshrepo/main`（061b281dc0f3dea7a12fdb208e8f78af7364b0ac）

说明：以下统计按 `git diff A B`（方向 A -> B）生成。

## 2. 总体结论
- A 文件总数：76
- B 文件总数：121
- 仅 A 存在文件数：5
- 仅 B 存在文件数：50
- 共同路径但内容不同文件数：13
- 总差异文件数（A 与 B）：68
- 代码行变更摘要： 68 files changed, 3000412 insertions(+), 791 deletions(-)

## 3. 详细清单（已落盘）
- 仅 A 存在：`reports/local_only_files.txt`
- 仅 B 存在：`reports/ssh_only_files.txt`
- 共同路径内容有差异：`reports/common_changed_files.txt`
- 全量变更状态（含新增/删除/重命名）：`reports/name_status_local_to_ssh.txt`
- 全量增删行统计：`reports/numstat_local_to_ssh.tsv`
- 变更规模 Top20：`reports/top20_churn_local_to_ssh.txt`

## 4. 解读建议
- 若你关心“本地新增了什么”：先看 `local_only_files.txt`。
- 若你关心“服务器版本新增了什么”：看 `ssh_only_files.txt`。
- 若你关心“同名文件代码差异”：看 `common_changed_files.txt`，再对单文件执行 `git diff origin/main sshrepo/main -- <path>`。

## 5. 关键差异解读（人工摘要）
- `DL_AP_SSH` 相比 `DL_AP_Local` 额外包含大量训练产物：
  - `checkpoints/*.pt`
  - `data/outputs/*.pkl`
  - `simulated_firm_data_episode_*.csv`
- 变更规模统计中“300万+ 插入行”主要由上述 CSV 文件主导，不代表核心源码发生了同等规模改动。
- 共同路径中有实质源码差异的关键文件包括：
  - `training/episode.py`
  - `losses/p0_loss.py`
  - `losses/pi_loss.py`
  - `data/simulate_ts.py`
  - `losses/FC2losspipe.py`
  - `training/trainer.py`
- `DL_AP_Local` 独有内容主要是本地分析与审计文档，以及 `skills/update-docs` 相关文件。
