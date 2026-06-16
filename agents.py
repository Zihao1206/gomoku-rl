"""
智能体（Agent）—— 第二、三阶段

Agent 就是"做决定的玩家大脑"：看一眼局面，从合法动作里挑一个落子。

约定：所有 Agent 都实现同一个方法（这是它们共同的"接口 / 契约"）：

    action = agent.select_action(state, legal_actions)
        state          当前棋盘（numpy 数组），即"状态 State"
        legal_actions  当前所有能下的位置列表 [(r, c), ...]
        返回值          选中的一个动作 (row, col)

因为接口统一，RandomAgent / QAgent / HumanAgent 都能直接塞进 play_game，对弈循环一行都不用改。
"""

import random
import pickle
from collections import defaultdict

import torch                              # 阶段5 起：DQN 要用神经网络
from qnet import encode_board, QNetwork   # 复用 5.1 写好的「棋盘编码 + Q 网络」


class RandomAgent:
    """随机玩家：完全不看局面，从所有合法动作里瞎挑一个。"""

    def __init__(self, name="随机玩家", seed=None):
        self.name = name
        # 每个 agent 自带一个随机数发生器；传了 seed 就能复现同样的对局
        self.rng = random.Random(seed)

    def select_action(self, state, legal_actions):
        # 随机玩家不看棋盘，所以 state 用不上；但接口保持统一
        return self.rng.choice(legal_actions)


class QAgent:
    """
    Q-Learning 玩家：随身带一张"打分本"(Q 表)，挑分最高的位置下，
    并以小概率 ε 随机探索（ε-贪婪）；每走一步用 update 复盘修正 Q 表。
    """

    def __init__(self, name="Q学习玩家", epsilon=0.1, alpha=0.1, gamma=0.9, seed=None):
        self.name = name
        self.epsilon = epsilon              # 探索概率：这个比例的步数会随机乱下去探索
        self.alpha = alpha                  # 学习率：每次把旧分数往"新看法"挪多少 (0~1)
        self.gamma = gamma                  # 折扣因子：未来的好处打几折 (0~1)
        self.rng = random.Random(seed)
        # Q 表：键是 (局面, 动作)，值是分数 Q。
        # 用 defaultdict(float)：没记录过的 (局面, 动作) 自动当 0.0，省去"判断在不在表里"
        self.Q = defaultdict(float)

    def _state_key(self, state):
        """
        把棋盘(numpy 数组)转成能当"字典键"的东西。
        numpy 数组本身不可哈希(不能直接当键)，.tobytes() 把它压成一串唯一的字节；
        同样的局面 → 同样的字节 → 指向 Q 表里同一条记录。
        """
        return state.tobytes()

    def select_action(self, state, legal_actions):
        """ε-贪婪地选一个动作。"""
        # —— 探索：以 epsilon 的概率，随便挑一个合法位置 ——
        if self.rng.random() < self.epsilon:
            return self.rng.choice(legal_actions)

        # —— 利用：挑当前 Q 分数最高的位置 ——
        key = self._state_key(state)
        q_values = [self.Q[(key, a)] for a in legal_actions]   # 每个合法动作各自的分数
        best_q = max(q_values)

        # 可能好几个动作并列最高分(尤其训练初期全是 0)，
        # 就从这些并列最高的里随机挑一个 —— 避免老是固定挑第一个而产生偏置
        best_actions = [a for a, q in zip(legal_actions, q_values) if q == best_q]
        return self.rng.choice(best_actions)

    def update(self, state, action, reward, next_state, next_legal_actions, done):
        """
        学习一步：把 Q(state, action) 朝"实际看到的结果"挪一点点。

        公式：  Q(s,a) ← Q(s,a) + α · [ r + γ · max Q(s',·) − Q(s,a) ]
                          旧分数        学习率   当场  折扣  新局面最高分  旧分数
        """
        key = self._state_key(state)
        old_q = self.Q[(key, action)]                  # 旧分数 Q(s,a)

        if done:
            # 终局：没有"下一步"了，未来价值 = 0，新看法就只剩当场奖励 r
            future = 0.0
        else:
            # 还没结束：看新局面 s' 里所有合法动作中最高的分（"走到 s' 后还能多好"）
            nkey = self._state_key(next_state)
            future = max(self.Q[(nkey, a)] for a in next_legal_actions)

        target = reward + self.gamma * future          # 新看法（目标值）
        # 把旧分数朝新看法挪 α 那么多
        self.Q[(key, action)] = old_q + self.alpha * (target - old_q)

    def save(self, path):
        """把学到的 Q 表存到磁盘(pickle)，下次直接 load，不用重训。"""
        with open(path, "wb") as f:
            pickle.dump(dict(self.Q), f)               # 存成普通 dict 更稳

    def load(self, path):
        """从磁盘加载 Q 表；装回 defaultdict，没见过的局面仍默认 0 分。"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.Q = defaultdict(float, data)


class HumanAgent:
    """人类玩家：从键盘输入坐标落子。接口和别的 agent 一样，能直接塞进 play_game。"""

    def __init__(self, name="你"):
        self.name = name

    def select_action(self, state, legal_actions):
        while True:                                    # 一直问到给出合法落子为止
            try:
                raw = input("  请落子（格式：行 列，例如  1 2 ；输 q 退出）: ").strip().lower()
            except EOFError:
                raise SystemExit("\n（输入已结束，退出对局）")
            if raw in ("q", "quit", "exit"):
                raise SystemExit("（你退出了对局）")
            try:
                parts = raw.replace(",", " ").split()
                move = (int(parts[0]), int(parts[1]))
            except (ValueError, IndexError):
                print("  ⚠️ 格式不对，请输入两个数字，比如  1 2")
                continue
            if move not in legal_actions:
                print("  ⚠️ 这个位置不能下（越界或已被占），换一个")
                continue
            return move


class DQNAgent:
    """
    DQN 玩家（阶段 5.2）：随身带一个 Q 网络(QNetwork)。
    看一眼棋盘 → 网络给【每个格子】打一个分(Q) → 在合法格子里按 ε-贪婪挑一个落子。

    和 QAgent 最大的区别：
        · QAgent 用「查表」记分 —— 没见过的局面只能抓瞎；
        · DQNAgent 用「神经网络」算分 —— 能对没见过的局面做泛化推断。
    ⚠️ 本阶段网络【还没训练】，所以打分是乱的、棋力≈随机。这一步只为把「接线」搭通：
       证明它能正确地"判断该谁下 → 算分 → 盖住非法格 → 选点"，能直接塞进 play_game。

    接口仍是 select_action(state, legal_actions)，和别的 agent 完全一致。
    """

    def __init__(self, board_size, name="DQN玩家", epsilon=0.1,
                 net=None, device=None, seed=None):
        self.name = name
        self.board_size = board_size
        self.epsilon = epsilon                 # ε-贪婪的探索概率
        self.rng = random.Random(seed)         # 自带随机源，传 seed 可复现

        # —— 设备：有苹果 MPS(Metal) 就用 MPS，否则用 CPU ——
        # （这台 Mac 没有 N 卡，所以【不写 cuda】；本项目很小，CPU/MPS 都够快）
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)

        # 持有一个 Q 网络；不传就新建一个（此刻是随机初始化、未训练的）
        self.net = net if net is not None else QNetwork(board_size)
        self.net.to(self.device)
        self.net.eval()                        # 推断模式：本阶段只用来选动作，不训练

    @staticmethod
    def _infer_player(state):
        """
        接口里没给「现在轮到谁」，得自己从棋盘【数子】推断（黑 1 先手、白 2 后手）：
            黑子数 == 白子数      → 轮到黑(1)   （刚开局 0==0 也成立）
            黑子数 == 白子数 + 1  → 轮到白(2)
            其它                 → 不可能出现的合法局面 → 直接报错，把 bug 挡在早处
        （正常交替落子下，黑子要么和白子一样多、要么只多一个；
          若出现「白比黑多」或「黑比白多 2+」，一定是上游哪里把棋盘传错了。）
        """
        black = int((state == 1).sum())        # 1 = 黑子
        white = int((state == 2).sum())        # 2 = 白子
        if black == white:
            return 1                           # 轮到黑
        if black == white + 1:
            return 2                           # 轮到白
        raise ValueError(
            f"非法局面：黑子 {black} 个、白子 {white} 个，无法推断该谁下"
            f"（黑白子数应相等，或黑只比白多 1）。"
        )

    def select_action(self, state, legal_actions):
        """ε-贪婪：以 epsilon 概率随机合法走；否则选「合法格里 Q 最高」的那个。"""
        # —— 探索：以 epsilon 概率，随便挑一个合法位置 ——
        if self.rng.random() < self.epsilon:
            return self.rng.choice(legal_actions)

        # —— 利用：让网络给每个格子打分，再【只在合法格里】挑最高 ——
        player = self._infer_player(state)         # ① 先搞清楚此刻该谁下
        x = encode_board(state, player)            # ② 编码成 (2, H, W)：我方/对方两张图
        x = x.unsqueeze(0).to(self.device)         #    加 batch 维 → (1, 2, H, W) 并搬到设备

        with torch.no_grad():                      # 只前向、不求梯度（推断更快更省内存）
            q = self.net(x)[0].cpu()               # ③ → (n_cells,) 搬回 CPU 方便逐格取值

        # ④ 屏蔽非法落子：只遍历【合法动作】，绝不会选到已占/越界的格
        W = self.board_size
        best_q = None
        best_actions = []
        for (r, c) in legal_actions:
            qa = q[r * W + c].item()               # 格子编号 = row * W + col
            if best_q is None or qa > best_q:
                best_q = qa
                best_actions = [(r, c)]
            elif qa == best_q:
                best_actions.append((r, c))        # 并列最高，先都收着

        # ⑤ 多个并列最高就随机挑一个（和 QAgent 同款，避免老固定选第一个的偏置）
        return self.rng.choice(best_actions)
