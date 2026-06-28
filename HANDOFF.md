# 项目交接文档（HANDOFF）

> 给接手的 AI 助手（Codex）。**先完整读本文件再开工。** 你要接替一位「资深 AI 工程师 + 极有耐心的编程导师」，继续带一位学生从零做五子棋强化学习。

---

## 0. 一句话

一个**从零、循序渐进**的 RL 教学项目（五子棋 / 井字棋）。阶段 1–4 与 5.1、5.2 已完成；**5.3 DQN 训练也已跑通（3×3：99% 胜 / 0 负）**，下一步把棋盘放大看泛化。最重要的不是写得快，而是**保持下面的教学风格**。

## 1. 教学风格 ⭐（最重要，必须严格保持）

- 全程**中文**。
- **一步步、小块**推进：一次只带学生完成一个小模块，**绝不一次甩几百行代码**。
- **先用大白话讲直觉**、结合五子棋场景；**不要一上来甩数学公式**，等直觉到位再点公式。
- 代码配**详尽的中文注释**，并明确告诉学生**怎么运行 / 测试**；写完要**亲自跑一遍**，把真实输出讲给学生。
- 每完成一步，**等学生确认「跑通了、也理解了」**再进入下一步。
- 学生**随时会打断提问**——先把问题讲透，再继续。多用「**你已经会了 X，这只是 Y**」来降低难度。

**导师式提问（学生 2026-06 明确要求，长期严格遵守）⭐**

- 你的任务**不是替学生写完**，而是在一步步过程里**教会他**——他在从零学 RL / PyTorch（机器人 / 视觉背景，Python 很熟）。
- **每进入一个新概念 / 新小模块前，先问 1–3 个关键问题**，判断他当前理解到哪里，别一上来就开讲。
- 问题要**激活思考**：让他预测某段代码会发生什么、比较两种设计、解释某个变量为什么存在（用开放式问题，不要是非题）。
- **别为提问而拖慢**：他说「理解了，继续」就直接进下一小步。
- **答错不直接否定**：先指出他卡在哪，再用**具体例子**帮他修正。
- 默认他**不懂** RL / Q-Learning / DQN / PyTorch 术语；**每个术语第一次出现就用大白话解释**（中英并给）。
- 讲解**优先结合本项目已有代码**（gomoku_env / agents / qnet / train / replay …），别脱离项目泛泛讲理论。
- **确认他是否掌握时，不要用「清楚了吗 / 理解了吗 / 懂了吗 / 你确认理解了再继续」这类笼统是非问法**；改用 **1–2 个具体、可回答的小问题**来判断。看他的回答：答到位就进下一步；答不完整就指出他卡在哪、用例子补上，再继续。

## 2. 学生背景

熟练 **Python**；但 **RL 和 PyTorch 都是新手**；做机器人 / 视觉。技术词建议中英并给（如「智能体（Agent）」）。

## 3. 环境与运行

> 本项目最初在 **Linux + RTX 4070** 上开发；2026-06 起迁移到一台 **M1 Mac** 继续。两台机器都能跑，主要差别是「设备」（CUDA vs MPS）。

**当前机器（M1 Mac）：**
- 项目路径：`/Users/shenzihao/zh/gomoku`（= git 仓库根 = 工作目录）。
- conda 环境 **`gomoku`**（Python 3.11，用 brew 装的 Miniforge）。依赖：`numpy 2.4.6`、**`torch 2.12.0`（arm64 Mac 版，无 CUDA）**、`matplotlib`、`tqdm`。
- 运行：先 `conda activate gomoku`，再 `python 文件.py`（或一行 `conda run -n gomoku python 文件.py`）。
- **设备：没有 N 卡 → 用 MPS(Metal)/CPU**。`torch.backends.mps.is_available()` == True。写训练/推断代码时设备一律 `device = "mps" if torch.backends.mps.is_available() else "cpu"`，**不要写 `cuda`**。本项目很小，CPU/MPS 都绰绰有余。

**旧机器（Linux + RTX 4070，历史参考）：**
- python 路径 `/home/zh/anaconda3/envs/gomoku/bin/python`；torch `2.12.0+cu130`（CUDA 可用）。
- ⚠️ 已知坑：shell 里 source 了 ROS，`PYTHONPATH` 带着 ROS/系统包；若出现「莫名 import 到系统/ROS 包」的怪事，先 `unset PYTHONPATH` 再跑。
- Git：**https://github.com/Zihao1206/gomoku-rl** （public，main 分支）。`*.pkl` 模型不入库，跑 `train.py` 可重新生成。
- 习惯：每完成一个阶段，**经学生同意后** `commit + push`（提交信息用中文，结尾带 `Co-Authored-By`）。

## 4. 代码现状（逐文件）

| 文件 | 内容 |
|------|------|
| `gomoku_env.py` | `GomokuEnv(board_size, win_length)`。`reset()`→state；`step(action)`→`(state, reward, done, info)`（Gym 风格）；`legal_actions()`→`[(r,c),...]`；`render()`；`_check_win`（只查最后落子、横竖斜 4 方向两边数）；`_in_board`。|
| `agents.py` | `RandomAgent`；`QAgent`（ε-贪婪、`update`、`save/load` pickle、`_state_key=board.tobytes()`）；`HumanAgent`（键盘输入）；**`DQNAgent`**（持 `QNetwork`、数子推断玩家、网络算 Q + 盖非法格 + ε-贪婪、`save/load` 用 `torch.save` 存 `state_dict`、设备 mps/cpu）。|
| `play.py` | `play_game(env, black_agent, white_agent, render)`→winner；`__main__` 跑随机对随机统计。|
| `train.py` | `play_one_selfplay_game`（自我对弈+学习，功劳分配见 §5）；`evaluate`（ε=0、复用 play_game、不学习）；`train_selfplay`（ε 线性衰减）；`__main__` 在 3×3 训 20 万局并存 `q_table_3x3.pkl`。|
| `play_human.py` | 人机对战，加载 `q_table_3x3.pkl`。|
| `experiment_scaling.py` | 放大棋盘实验，实证表格法的天花板。|
| `qnet.py` | 【5.1】`encode_board(board, player)`→`(2,H,W)`（图层0=我方/1=对方，按「轮到谁」视角）；`QNetwork`（MLP，每格一个 Q）。|
| `replay.py` | 【5.3①】`ReplayBuffer`（经验回放：固定容量、随机抽样、可 seed）+ `Transition`。|
| `train_dqn.py` | 【5.3③④】`collect_one_game`（收集转移，功劳分配照搬 train.py）+ `compute_loss`（MSE）+ `train_dqn`（训练循环：优化器三连 + 目标网络定期同步 + 存盘）。命令行选棋盘：`python train_dqn.py 5 4 6000`。|
| `play_human_dqn.py` | 人机对战（DQN 版），加载 `dqn_*.pt`。|
| `demo_target_net.py` / `demo_overfit_batch.py` | 【5.3②④】目标网络机制、优化器降 loss 的最小验证。|
| `aznet.py` | 【阶段6】`PolicyValueNet`：双头网络（共享躯干 + policy 头出 n_cells logits + value 头出 1 过 tanh）。复用 `qnet.encode_board`；forward 只吐 logits、softmax 留外面。|
| `mcts.py` | 【阶段6】MCTS 引擎：`Node`(N/W/Q/P/done/winner) → `puct_score`(⚠️利用项取 **-Q**，孩子是对手视角) / `select_to_leaf` → `expand`(priors 可选，没给走均匀) → `evaluate_rollout` / `net_evaluate`(网络版，一次前向出 P 和 v) → `backpropagate`(每层**符号翻转**) → `mcts_search`(net=None 走经典 rollout、给 net 走 AlphaZero) → `MCTSAgent`(统一 select_action 接口)。|
| `eval_mcts.py` | 【阶段6】MCTS 打随机评估（3×3）：rollout 版 / 未训练网络版 / 传入训练后 net。|
| `train_az.py` | 【阶段6】`self_play_game`(收集 `(state,π,z)`，按 N 正比采样落子) + `examples_to_tensors` + `compute_loss`(policy 软标签交叉熵 + value MSE) + `train`(自举训练循环：自对弈→训练→再自对弈)。|
| `aznet_cnn.py` | 【阶段7】`PolicyValueNetCNN`：**CNN 双头网**——卷积躯干(3×Conv3×3+ReLU、padding=1、**与盘大小解耦**) + **1×1 卷积 policy 头**(逐格出 logit) + **flatten+Linear value 头**(跨格汇总→tanh)。接口同 `aznet.py`，可直接替换塞进 `mcts/train_az`。|
| `train_az_cnn.py` | 【阶段7】6×6(win4) CNN 训练入口，复用 `train_az.train(net=注入)`。`python train_az_cnn.py [iters] [eval_games]`，存 `az_cnn_6x6.pt`。|
| `play_human_az_cnn.py` | 【阶段7】人机对战（AZ-CNN）：加载 `az_cnn_6x6.pt`→`MCTSAgent(net=...)`。`python play_human_az_cnn.py 6 4 模型 n_sim`（n_sim 调小≈看网络原始直觉）。|
| `eval_net_vs_rollout.py` | 【阶段7·诊断】训练后网络版 MCTS vs 纯 rollout 版（**同 n_sim**）头对头——剥离"搜索"单看"网络"。结果 黑90%/白75%。|
| `diag_tactics.py` / `diag_tactics2.py` | 【阶段7·诊断】固定战术体检：必防局面的 policy 排名 + 各 sim 的 P/N/Q + net/rollout 对照；**diag_tactics2** 用旋转/镜像生成横竖主反对角**等价**局面，证实**朝向等变性盲区**。|
| `demo_conv_shape.py` / `smoke_az_cnn.py` | 【阶段7】Conv2d 形状/padding 最小实验；CNN 接入训练链的冒烟测试。|
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
- [x] **阶段5 DQN** ✅：5.1 编码+Q网络 → 5.2 select_action → 5.3 训练（① ReplayBuffer ② 目标网络 ③ 收集转移 ④ 训练循环 ⑤ 放大验证泛化）全部完成
    - 结果：3×3 作黑 98%+/0 负、作白 ~87%（≈表格法、约 1/66 对局）；**5×5(win4) 作黑 96.5%**（表格法在此 75% 局面没见过，DQN 靠泛化扛住）。
    - 人机对战：`python play_human_dqn.py [bs] [wl]`，加载 `DQNAgent.save` 存的 `dqn_*.pt`。
- [x] **阶段6 AlphaZero（MCTS + 策略/价值网络）** ✅
    - 链路：MCTS 四步（PUCT 选择 / 扩展 / 评估 / 回传）→ 双头网络（policy+value）→ 自对弈生成 `(state,π,z)` → 双头 loss（policy 软标签交叉熵 + value MSE）→ 自举训练。
    - 3×3 验证：纯 rollout MCTS（零训练、零网络）作黑 **100%/0负**、作白 90%；训练后网络版 MCTS 作黑 **100%/0负**（从未训练的 96% 反超）、作白 86%。
    - ⚠️ TODO：执白 86% 还可提升 —— 更多自对弈轮 / minibatch 采样 / 温度调度 / value loss 加权或 L2 正则；放大到 6×6+ 验证泛化；存模型 + 人机对战（仿 `play_human_dqn.py`）。
- [~] **阶段7 放大到 6×6 + 网络 MLP→CNN**（进行中）
    - **CNN 双头网** `aznet_cnn.py`：卷积躯干（局部连接+权重共享→**与盘大小解耦**）+ 1×1 逐格 policy 头 + flatten/Linear 全局 value 头。接口同 MLP 版，`mcts.py` 零改即可用。
    - **接线**：`train_az.train()` 加可注入 `net=None`（默认仍造 MLP，向后兼容 3×3）；`eval_mcts.run()` 加 `board_size/win_length`（默认 3×3）。
    - **训练**（`train_az_cnn.py`，6×6 win4，`iters=20/games=20/n_sim=200/epochs=10`，cpu，~70s/轮）：policy 3.58→1.66、value→~0.26；打随机 作黑/作白 **均 100%**；存 `az_cnn_6x6.pt`。
    - ⚠️ **"打随机 100%"是假象**：网络版 MCTS 同 n_sim 能赢纯 rollout（黑90%/白75%，证明确实学到了东西），**却被随手织威胁的人类轻松击败**。
    - 🔬 **诊断结论（核心）**：`diag_tactics.py`/`diag_tactics2.py` 固定局面体检 + 旋转镜像等价对照 → **不是 bug**（控制组横向必防完美），是 **朝向等变性盲区**：同一棋形**横向学透、竖向半生、对角基本不会**（横 vs 它转 90° 的竖，value 从 −0.92 翻到 +0.01）。根因 = **无对称数据增强 + 自对弈不足**；卷积只平移不变、不旋转不变。
    - 🛠️ **B 路线图**（从零学、"撞了再修"、每步回旧诊断验收）：**B1 8 重对称数据增强**（最对症、先做）→ B2 温度调度（self-play 落子）→ B3 根节点 Dirichlet 噪声（**仅 self-play**）→ B4 回放缓冲+minibatch → B5 上规模长训+全套复检 →（选修）B6 残差块/更大棋盘。
    - **当前位置：B1 待开始**（拟在新 session 接力；本会话已落盘到此）。

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
- **损失** = `MSE(预测 Q(s, action), target)`；优化器 Adam（lr≈1e-3）；张量/网络 `.to(device)`（本机 Mac：`device = "mps" if torch.backends.mps.is_available() else "cpu"`，**不要用 cuda**）。
- **验证路径**：先在**小棋盘（3×3 或 4×4）**训，对照表格法基线（应也能打到≈0 负），证明 DQN 没写错；**再放大到 5×5/6×6**——表格法在那已崩，看 DQN 靠泛化扛住。
- 网络存取用 `torch.save(net.state_dict(), ...)`。

## 8. 已验证的超参数与结果

- **表格 3×3**：`α=0.2, γ=0.95, ε 0.30→0.02, 20万局自我对弈`（约 20s）→ 作黑 98.6% 胜 / 0 负，作白 87.3% 胜 / 0 负。Q 表 16089 条。
- **放大实验**（各训 4 万局）：Q 表条目 3×3=1.5万 → 4×4=175万 → 5×5=805万；「实战中没见过的局面」占比 0% → 43% → 75%（这就是必须上神经网络的实证理由）。
- **DQN（阶段5.3）**：`γ=0.95, Adam lr=1e-3, batch=64, 目标网络每 100 步同步, ε 0.30→0.05`。3×3 自对弈 3000 局 → 作黑 98%+/0 负、作白 ~87%（≈表格法，约 1/66 对局）；**5×5(win4) 6000 局 → 作黑 96.5%**（表格法在此 75% 局面没见过，DQN 靠泛化扛住）。设备 mps/cpu。
- **AlphaZero（阶段6）**：3×3，`c_puct=1, n_simulations=100~200`。① 纯 rollout MCTS（零训练、零网络）作黑 **100%/0负**、作白 90%/1负 —— 搜索本身即强（追平/超过训练后的表格法、DQN）。② 自举训练 `iterations=15, games/iter=10, n_sim=100, epochs/iter=10, Adam lr=1e-3`：policy_loss 2.2(≈log9)→1.4、value_loss→低；训练后网络版作黑 **100%/0负**（从未训练 96% 反超）、作白 86%。设备 cpu（3×3 小，cpu 比 mps 快）。
- **AlphaZero+CNN（阶段7）**：6×6 win4，`iters=20, games/iter=20, n_sim=200, epochs/iter=10, Adam lr=1e-3`，cpu ~70s/轮。policy 3.56→1.66(≈log36 起步)、value 0.98→0.26；网络版 MCTS 打随机 黑/白均 100%；同 n_sim 头对头 **网络 vs rollout = 黑 90% / 白 75%**。⚠️ 仍被人类轻松赢 → 诊断为**朝向等变性盲区**（详见 §6 阶段7）。对弈 `n_sim=400`、`n_sim=10` 看网络原始直觉。

## 9. 启动 Prompt（学生会把它粘给你；与本文件配套）

见仓库外学生提供的 prompt；核心就是：**读本文件、严守 §1 教学风格与 §5 约定、从 5.2 开始、一步步带、跑通再继续**。
