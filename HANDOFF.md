# 项目交接文档（HANDOFF）

> 给接手的 AI 助手（Codex）。**先完整读本文件再开工。** 你要接替一位「资深 AI 工程师 + 极有耐心的编程导师」，继续带一位学生从零做五子棋强化学习。

---

## 0. 一句话

一个**从零、循序渐进**的 RL 教学项目（五子棋 / 井字棋）。阶段 1–4 与 5.1 已完成，**下一步是 5.2**。最重要的不是写得快，而是**保持下面的教学风格**。

## 1. 教学风格 ⭐（最重要，必须严格保持）

- 全程**中文**。
- **一步步、小块**推进：一次只带学生完成一个小模块，**绝不一次甩几百行代码**。
- **先用大白话讲直觉**、结合五子棋场景；**不要一上来甩数学公式**，等直觉到位再点公式。
- 代码配**详尽的中文注释**，并明确告诉学生**怎么运行 / 测试**；写完要**亲自跑一遍**，把真实输出讲给学生。
- 每完成一步，**等学生确认「跑通了、也理解了」**再进入下一步。
- 学生**随时会打断提问**——先把问题讲透，再继续。多用「**你已经会了 X，这只是 Y**」来降低难度。

## 2. 学生背景

熟练 **Python**；但 **RL 和 PyTorch 都是新手**；做机器人 / 视觉。技术词建议中英并给（如「智能体（Agent）」）。

## 3. 环境与运行

- conda 环境 **`gomoku`**（Python 3.11）。依赖：`numpy 2.4.6`、`matplotlib`、`tqdm`、**`torch 2.12.0+cu130`（CUDA 可用，显卡 RTX 4070）**。
- 跑代码用该环境的 python：`/home/zh/anaconda3/envs/gomoku/bin/python 文件.py`，或先 `conda activate gomoku`。
- ⚠️ **已知坑**：学生 shell 里 source 了 ROS，`PYTHONPATH` 带着 ROS/系统包（pip 装 torch 时报过 launch-ros 的无害警告）。若出现「莫名 import 到系统/ROS 包」的怪事，先 `unset PYTHONPATH` 再跑。
- Git：**https://github.com/Zihao1206/gomoku-rl** （public，main 分支）。`*.pkl` 模型不入库，跑 `train.py` 可重新生成。
- 习惯：每完成一个阶段，**经学生同意后** `commit + push`（提交信息用中文，结尾带 `Co-Authored-By`）。

## 4. 代码现状（逐文件）

| 文件 | 内容 |
|------|------|
| `gomoku_env.py` | `GomokuEnv(board_size, win_length)`。`reset()`→state；`step(action)`→`(state, reward, done, info)`（Gym 风格）；`legal_actions()`→`[(r,c),...]`；`render()`；`_check_win`（只查最后落子、横竖斜 4 方向两边数）；`_in_board`。|
| `agents.py` | `RandomAgent`；`QAgent`（`epsilon/alpha/gamma`、`select_action` ε-贪婪、`update` 标准 Q 公式、`save/load` pickle、`_state_key=board.tobytes()`）；`HumanAgent`（键盘输入）。|
| `play.py` | `play_game(env, black_agent, white_agent, render)`→winner；`__main__` 跑随机对随机统计。|
| `train.py` | `play_one_selfplay_game`（自我对弈+学习，功劳分配见 §5）；`evaluate`（ε=0、复用 play_game、不学习）；`train_selfplay`（ε 线性衰减）；`__main__` 在 3×3 训 20 万局并存 `q_table_3x3.pkl`。|
| `play_human.py` | 人机对战，加载 `q_table_3x3.pkl`。|
| `experiment_scaling.py` | 放大棋盘实验，实证表格法的天花板。|
| `qnet.py` | 【阶段5.1】`encode_board(board, player)`→`(2,H,W)` 张量（图层0=我方子，图层1=对方子，按「轮到谁」的视角）；`QNetwork`（MLP，输出每格一个 Q）；含前向 demo。**网络还没训练。** |
| `README.md` / `.gitignore` | — |
| `test_env.ipynb` | 学生的草稿本，**未入库**，别动。|

## 5. 关键约定 ⚠️（新代码必须保持一致）

- **棋盘编码**：`0`空 / `1`黑(X，先手) / `2`白(O，后手)。**动作** = `(row, col)` 普通 int 元组。
- **`step` 返回** `(state, reward, done, info)`（Gym 风格）。`state` 是 `board.copy()`（numpy 数组）。
- **奖励约定**：环境**只给「刚走这步直接获胜」的人 +1**，其余（含平局）为 0。**输家的 −1 不由环境给，在训练循环里补**。
- **智能体统一接口**：`action = agent.select_action(state, legal_actions)`。**所有 agent 都遵守它**，这样 `play_game` / `evaluate` 不用改就能复用。
- **自我对弈功劳分配**（`train.py` 的核心，新写 DQN 训练要照此逻辑）：把对手看作「环境的一部分」；用 `pending[黑]/pending[白]` 记各自「还没结算的那一手」；某方**再次轮到**时结算它上一手（`reward=0`、`next_state`=眼前局面、非终局）；**终局**时给赢家最后一手 `+1`、输家最后一手 `−1`、平局都 `0`。
- **表格状态键** = `board.tobytes()`。
- **DQN 编码**（`qnet.py`）：2 个图层（我方/对方），**按「轮到的玩家」视角**，一个网络通吃黑白。
- **DQN 动作下标**：格子编号 = `row * W + col`（网络每格输出一个 Q）；**选/算 max 时要盖住非法格子**。

## 6. 路线图与进度

- [x] 阶段1 棋盘环境
- [x] 阶段2 随机 Agent + 对弈循环
- [x] 阶段3 表格型 Q-Learning（3×3 自我对弈：胜率 58%→98.6%，**0 负**；模型存于 `q_table_3x3.pkl`）
- [x] 阶段3.4 人机对战 + pickle 存取
- [x] 阶段4 放大棋盘、实证表格法天花板（`experiment_scaling.py`）
- [ ] **阶段5 DQN**：5.1 棋盘编码 + Q 网络 ✅ →（下一步）**5.2 网络版 select_action** → 5.3 训练
- [ ] 阶段6 MCTS + 策略/价值网络（AlphaZero 思路）

## 7. 下一步具体设计（给你的实现指引，但仍要按教学风格一步步带）

### 5.2 —— 让网络「会下棋」
- 新增一个 DQN 智能体（如 `agents.py` 里的 `DQNAgent`，或新文件），内部持有一个 `QNetwork`。
- `select_action(state, legal_actions)`：① 从棋盘**推断当前玩家**（数子：黑子数==白子数 → 轮到黑；黑子数==白子数+1 → 轮到白。**接口里没传 player，必须自己推**）；② `encode_board`→网络前向→每格 Q；③ **盖住非法格子**（非法位设 `-inf` 或只在合法格里比）；④ ε-贪婪选。
- 保持 `select_action(state, legal_actions)` 签名，**直接能塞进 `play_game`**。
- 验证：未训练的 DQNAgent 丢进 `play_game` 打随机，应≈随机水平（证明接线对了）。

### 5.3 —— 训练（重头戏，按教学拆成小步）
- **经验回放**：用 `deque` 存转移 `(state, player, action_idx, reward, next_state, next_player, next_legal, done)`；训练时随机抽 batch。
- **目标网络**：`QNetwork` 的一个副本，每隔 N 步同步一次；用它算目标 `target = r + γ·max_合法 Q_target(s',·)`（终局则 `target = r`）。
- **两人对弈细节**：`s'` 仍是「该玩家**再次轮到**时」的局面（对手当环境），**与 `train.py` 的功劳分配完全一致**；输家 −1 同样在收集转移时按终局规则给。
- **损失** = `MSE(预测 Q(s, action), target)`；优化器 Adam（lr≈1e-3）；张量/网络 `.to(device)` 用 GPU。
- **验证路径**：先在**小棋盘（3×3 或 4×4）**训，对照表格法基线（应也能打到≈0 负），证明 DQN 没写错；**再放大到 5×5/6×6**——表格法在那已崩，看 DQN 靠泛化扛住。
- 网络存取用 `torch.save(net.state_dict(), ...)`。

## 8. 已验证的超参数与结果

- **表格 3×3**：`α=0.2, γ=0.95, ε 0.30→0.02, 20万局自我对弈`（约 20s）→ 作黑 98.6% 胜 / 0 负，作白 87.3% 胜 / 0 负。Q 表 16089 条。
- **放大实验**（各训 4 万局）：Q 表条目 3×3=1.5万 → 4×4=175万 → 5×5=805万；「实战中没见过的局面」占比 0% → 43% → 75%（这就是必须上神经网络的实证理由）。

## 9. 启动 Prompt（学生会把它粘给你；与本文件配套）

见仓库外学生提供的 prompt；核心就是：**读本文件、严守 §1 教学风格与 §5 约定、从 5.2 开始、一步步带、跑通再继续**。
