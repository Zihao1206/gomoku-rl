# 五子棋 / 井字棋 强化学习（从零实现）

一个**从零开始、循序渐进**的强化学习（Reinforcement Learning）教学项目：
从最基础的**表格型 Q-Learning** 起步，目标一路走到神经网络（DQN）与 AlphaZero 思路。

棋盘大小和「几子连珠算赢」都是参数：
`3×3 连三`（= 井字棋，当前已训练）→ `15×15 连五`（= 标准五子棋，远期目标）。

## 环境

- Python 3.11
- `numpy`（核心）；`matplotlib` / `tqdm` / `torch`（后续阶段用）

## 文件说明

| 文件 | 作用 |
|------|------|
| `gomoku_env.py` | 棋盘**环境**：`reset` / `step` / 胜负判断 / 渲染 |
| `agents.py` | 三个**智能体**：`RandomAgent`（随机）、`QAgent`（Q-Learning）、`HumanAgent`（人类键盘输入）|
| `play.py` | **对弈循环** `play_game` + 随机对随机统计 |
| `train.py` | **自我对弈训练** + 评估，训练完保存 Q 表 |
| `play_human.py` | **人机对战**：跟训练好的 AI 下一盘 |

## 快速开始

```bash
# 1) 训练（3×3，约 20 秒，生成 q_table_3x3.pkl）
python train.py

# 2) 跟你训练好的 AI 下一盘（井字棋它已是完美玩家，你最多逼平）
python play_human.py
```

## 路线图

- [x] 阶段 1：棋盘环境
- [x] 阶段 2：随机 Agent + 对弈循环
- [x] 阶段 3：表格型 Q-Learning（3×3 自我对弈，胜率 58% → 98.6%，**0 负**）
- [x] 阶段 4：放大棋盘，逼近表格法的天花板（`experiment_scaling.py`：Q 表涨到 800 万条、3/4 局面没见过 → 印证需要神经网络）
- [ ] 阶段 5：DQN（用神经网络代替 Q 表）
- [ ] 阶段 6：MCTS + 策略/价值网络（AlphaZero 思路）

> 这是一个注重「把每一步讲清楚」的学习项目。
