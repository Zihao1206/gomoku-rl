"""
阶段6 (AlphaZero) —— MCTS 第①块：搜索树的节点 Node

一个 Node = 搜索树上的一个「局面」。它要能撑起四件事，缺啥补啥：
    ① 算自己的 PUCT 分（选择用）  → 需要 N、P（探索项要用），父节点的总访问数
    ② 被扩展出孩子（扩展用）      → 需要知道自己是什么局面 state，才能列合法走法
    ③ 回传时更新（回传用）        → 需要 N、W，以及能往上爬的 parent
    ④ 找到孩子和父亲             → children（往下）、parent（往上）

注：Q 不单独存，它永远 = W / N，用 @property 现算 —— 少一个"回传后忘同步 Q"的 bug 点。
"""

import math
import random

import numpy as np
import torch

from gomoku_env import GomokuEnv
from qnet import encode_board


class Node:
    def __init__(self, state, parent=None, prior=0.0):
        # —— 树结构：往上、往下 ——
        self.parent = parent       # 父节点（根节点为 None）；回传时顺着它一路往上走
        self.children = {}         # 孩子：dict，键 = 动作 (r, c)，值 = 子 Node；选择时在这里挑

        # —— 这个节点代表的局面 ——
        self.state = state         # 棋盘 board.copy()（小棋盘直接存最省心；扩展/评估都要用它）

        # —— MCTS 统计量（你列的 N / W / P，Q 现算）——
        self.P = prior             # 先验概率：父节点扩展时，由 policy 分给"走到我这一步"的 P
        self.N = 0                 # 访问次数（visit count）
        self.W = 0.0               # 累计价值（历次回传灌进来的 v 之和）

        # —— 终局信息（扩展子节点时，由环境 step 的结果填入）——
        self.done = False          # 这个局面是不是终局（有人赢 / 下满平局）
        self.winner = None         # 终局赢家：1黑 / 2白；None = 非终局 或 平局

    @property
    def Q(self):
        """平均价值 = 累计价值 / 访问次数；还没访问过(N=0)时先当 0，顺便躲开除零。"""
        return self.W / self.N if self.N > 0 else 0.0

    def is_leaf(self):
        """没有孩子 = 还没被扩展过 = 叶子。选择阶段会一直往下走到叶子为止。"""
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def puct_score(self, c_puct):
        """
        本节点（作为父亲的某个孩子）的 PUCT 选择分，【从父亲视角】看：
            score = -Q  +  c · P · sqrt(N_父) / (1 + N)
                    利用       ──── 探索（先验 P 加权，随自己 N 增大而衰减）────
        ⚠️ 利用项是 -Q 不是 +Q：self.Q 存的是【本节点(对手)视角】的价值（回传时 v
           一层层翻符号灌下来的）。父亲要挑"把对手坑得最惨"的，即 -Q 最大 的孩子。
        N_父 = 父节点访问次数：父亲被访问越多、自己越没被选，探索项越大。
        （父亲还没被访问时 sqrt(0)=0、探索项为 0，只剩 -Q；实际选择中父亲总先被走到、N≥1。）
        """
        N_parent = self.parent.N
        explore = c_puct * self.P * math.sqrt(N_parent) / (1 + self.N)
        return -self.Q + explore


def select_child(node, c_puct):
    """在 node 的所有孩子里挑 PUCT 分最高的，返回 (动作, 子节点)。"""
    return max(node.children.items(), key=lambda kv: kv[1].puct_score(c_puct))


def select_to_leaf(root, c_puct):
    """从根出发，每个岔口都按 PUCT 选最高的往下走，直到落到一个叶子，返回这个叶子。"""
    node = root
    while not node.is_leaf():
        _action, node = select_child(node, c_puct)
    return node


def _infer_player(state):
    """数子推断「轮到谁」：黑==白 → 轮到黑(1)；黑==白+1 → 轮到白(2)。（沿用全项目约定）"""
    black = int((state == 1).sum())
    white = int((state == 2).sum())
    if black == white:
        return 1
    if black == white + 1:
        return 2
    raise ValueError("非法局面：黑 %d 白 %d，无法推断该谁下" % (black, white))


def _step_from(state, action, board_size, win_length):
    """
    从任意局面 state 出发走一步 action —— 复用 GomokuEnv 现成的落子 / 判胜逻辑，
    不重写五子棋规则。返回 (next_state, done, winner)。
    做法：临时造个 env，把它的局面塞成 state、推断出该谁下，再 step 一次。
    """
    env = GomokuEnv(board_size, win_length)
    env.board = state.copy()
    env.current_player = _infer_player(state)
    env.done = False
    env.winner = None
    next_state, _reward, done, _info = env.step(action)
    return next_state, done, env.winner


def expand(node, board_size, win_length, priors=None):
    """
    扩展：把 node（一个非终局叶子）的每个合法走法都挂成子节点。
        · 子节点 state = 从 node 局面走那一步后的新棋盘
        · 先验 P：给了 priors（policy 网络输出）就用它；没给就退回均匀 1/n
          （均匀 → PUCT 退化成 UCT、只看 Q/N，正是 rollout 版的行为）。
        · 顺手记录子节点是否终局（done/winner），评估终局节点时直接用。
    """
    legal = [(int(r), int(c)) for r, c in np.argwhere(node.state == 0)]
    uniform = 1.0 / len(legal)                     # 没给 priors 时退回均匀
    for action in legal:
        next_state, done, winner = _step_from(node.state, action, board_size, win_length)
        p = priors[action] if priors is not None else uniform
        child = Node(state=next_state, parent=node, prior=p)
        child.done = done
        child.winner = winner
        node.children[action] = child


def evaluate_rollout(node, board_size, win_length, rng):
    """
    评估叶子 node 的价值 v，【站在"node 局面轮到谁下"的视角】：+1 赢 / -1 输 / 0 平。
        · 终局叶子：不用模拟。终局＝对手上一步刚走赢了 → 轮到你时你已经输 → v=-1；平局 v=0。
        · 非终局叶子：从 node 局面起，双方【随机】走到底（rollout），看这位玩家赢没赢。
    rng 是外部传进来的随机源（random.Random），方便复现。
    """
    leaf_player = _infer_player(node.state)

    if node.done:
        # 轮到下的这位，正是被对手将死的输家（平局除外）
        return 0.0 if node.winner is None else -1.0

    # —— rollout：把局面塞进临时 env，从 leaf_player 开始随机摸到终局 ——
    env = GomokuEnv(board_size, win_length)
    env.board = node.state.copy()
    env.current_player = leaf_player
    env.done = False
    env.winner = None
    while not env.done:
        env.step(rng.choice(env.legal_actions()))

    if env.winner is None:
        return 0.0                                  # 平局
    return 1.0 if env.winner == leaf_player else -1.0


def backpropagate(leaf, v):
    """
    把叶子算出的价值 v 沿【原路】一路灌回根：每往上爬一层，符号翻转一次（换手＝换视角）。
        站在叶子视角是 v；父亲是对手视角，拿 -v；再上一层又翻回 +v…… 如此交替。
        每个经过的节点：N += 1、W += (当前这层视角的 v)。
    一路爬到根（parent is None）为止——根的 N 就是"总模拟次数"，下次选择当 N_父 用。
    """
    node = leaf
    value = v
    while node is not None:
        node.N += 1
        node.W += value
        value = -value           # 往上一层，换手，符号翻转
        node = node.parent


def net_evaluate(state, legal_actions, net, device):
    """
    用双头网络对一个局面【前向一次】，同时拿到扩展用的先验 P 和评估用的价值 v：
        priors : dict，合法动作 → 先验概率（只在合法 logit 上 softmax，等于盖住非法格）
        value  : 标量，站"该局面轮到谁"视角的赢面 v ∈ [-1, 1]
    底层就是 DQNAgent.select_action 那套：encode_board → 前向 → 盖非法格；
    只是这里一次前向同时取了 policy 和 value 两个头（aznet 双头网络的好处）。
    """
    player = _infer_player(state)
    x = encode_board(state, player).unsqueeze(0).to(device)      # (1, 2, H, W)
    with torch.no_grad():
        logits, value = net(x)
    logits = logits[0]                                           # (n_cells,)
    W = state.shape[1]
    idx = [r * W + c for (r, c) in legal_actions]                # 合法格的一维下标
    probs = torch.softmax(logits[idx], dim=0)                    # 只在合法格上 softmax
    priors = {a: float(probs[i]) for i, a in enumerate(legal_actions)}
    return priors, float(value.item())


def mcts_search(root_state, n_simulations, board_size, win_length, c_puct, rng,
                net=None, device="cpu"):
    """
    一次完整的 MCTS 决策：从 root_state 跑 n_simulations 次模拟，
    返回 (最佳动作, 根节点)。最佳 = 根下【访问次数 N 最大】的孩子。

    两套评估二选一：
        · net=None → rollout 版：均匀先验扩展 + 随机摸到底估 v（经典 MCTS）
        · net 给了 → 网络版：net_evaluate 一次前向，priors 当先验、value 当 v（AlphaZero）
    每次模拟 = 选择 →（终局?定值 : 扩展+评估）→ 回传。
    """
    root = Node(state=root_state.copy())
    for _ in range(n_simulations):
        leaf = select_to_leaf(root, c_puct)                          # ① 选择
        if leaf.done:                                                # 终局：不靠网络/rollout
            v = 0.0 if leaf.winner is None else -1.0
        elif net is None:                                            # rollout 版
            expand(leaf, board_size, win_length)
            v = evaluate_rollout(leaf, board_size, win_length, rng)
        else:                                                        # 网络版（AlphaZero）
            legal = [(int(r), int(c)) for r, c in np.argwhere(leaf.state == 0)]
            priors, v = net_evaluate(leaf.state, legal, net, device)
            expand(leaf, board_size, win_length, priors=priors)
        backpropagate(leaf, v)                                       # ⑤ 回传
    best_action = max(root.children.items(), key=lambda kv: kv[1].N)[0]
    return best_action, root


class MCTSAgent:
    """
    把 MCTS 包成和别的 agent 一样的统一接口：select_action(state, legal_actions)
    内部跑一次 mcts_search，返回访问次数最高的落子。能直接塞进 play_game。
    这是【纯 rollout 版】（还没接神经网络），但已经会算棋。第⑦块再把网络接进来。
    """

    def __init__(self, board_size, win_length, n_simulations=200,
                 c_puct=1.0, name="MCTS", seed=None, net=None, device="cpu"):
        self.board_size = board_size
        self.win_length = win_length
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.name = name
        self.rng = random.Random(seed)
        self.net = net               # None → rollout 版；给了网络 → AlphaZero 版
        self.device = device

    def select_action(self, state, legal_actions):
        # legal_actions 用不上（mcts_search 内部自己从 state 算合法走法），但接口保持统一
        action, _ = mcts_search(state, self.n_simulations, self.board_size,
                                 self.win_length, self.c_puct, self.rng,
                                 net=self.net, device=self.device)
        return action


if __name__ == "__main__":
    from gomoku_env import GomokuEnv

    env = GomokuEnv(board_size=3, win_length=3)
    root_state = env.reset()
    root = Node(state=root_state)                      # 建根：无父、无先验

    # —— 手动挂两个孩子，模拟"扩展"出两步：(1,1) 和 (0,0)，先验先随便给 ——
    # （真正的扩展/回传是后面几步的事；这里只为单独测 Node 这个数据结构对不对）
    for action, p in [((1, 1), 0.5), ((0, 0), 0.5)]:
        child = Node(state=root_state.copy(), parent=root, prior=p)
        root.children[action] = child

    # —— 模拟一次真实回传：(1,1) 这条线，叶子(对手视角)评估为输 v=-1，沿路灌上来 ——
    #    回传翻符号：叶子(白)视角 -1 → 根(黑)视角 +1。这正是下面 PUCT 取 -Q 的由来。
    backpropagate(root.children[(1, 1)], -1.0)

    print("根   : N=%d  Q=%.2f  is_leaf=%s  is_root=%s"
          % (root.N, root.Q, root.is_leaf(), root.is_root()))
    for a, c in root.children.items():
        print("孩子 %s: P=%.2f  N=%d  W=%.1f  Q=%.2f  is_leaf=%s"
              % (a, c.P, c.N, c.W, c.Q, c.is_leaf()))

    # —— 第②块：PUCT 选择 ——
    c_puct = 1.0
    print("\nPUCT 打分 (c=%.1f, N_父=%d):" % (c_puct, root.N))
    for a, ch in root.children.items():
        print("  %s: score=%.3f" % (a, ch.puct_score(c_puct)))
    a_sel, _ = select_child(root, c_puct)
    print("select_child  → 选中", a_sel)
    leaf = select_to_leaf(root, c_puct)
    print("select_to_leaf → 落到的叶子就是 (1,1) 那个孩子? %s, is_leaf=%s"
          % (leaf is root.children[(1, 1)], leaf.is_leaf()))

    # —— 第③块：扩展 ——
    print("\n--- 扩展测试 ---")
    # (a) 空棋盘根：扩展应挂出 9 个孩子、先验均匀 1/9、都非终局
    fresh = Node(state=GomokuEnv(3, 3).reset())
    expand(fresh, board_size=3, win_length=3)
    print("空根扩展出 %d 个孩子；某孩子 P=%.3f；有终局孩子? %s"
          % (len(fresh.children),
             next(iter(fresh.children.values())).P,
             any(c.done for c in fresh.children.values())))
    # (b) 黑差一步连成第 0 行：扩展后 (0,2) 这个孩子应是终局、黑胜
    e = GomokuEnv(3, 3); e.reset()
    for mv in [(0, 0), (1, 0), (0, 1), (1, 1)]:    # X O X O，轮到黑，黑走 (0,2) 即三连
        e.step(mv)
    near = Node(state=e._get_state())
    expand(near, 3, 3)
    win_child = near.children[(0, 2)]
    print("黑走 (0,2) 的子节点：done=%s winner=%s（1=黑）"
          % (win_child.done, win_child.winner))

    # —— 第④块：评估（rollout）——
    print("\n--- 评估测试 ---")
    import random
    rng = random.Random(0)
    # (a) 终局叶子（黑已赢、轮到白）：白是输家 → v 应 = -1
    print("终局叶子(黑已赢, 轮到白) 的 v =", evaluate_rollout(win_child, 3, 3, rng))
    # (b) 非终局叶子（near：轮到黑、黑差一步就能赢）：单次 + 500 次平均，站黑视角应明显 > 0
    one = evaluate_rollout(near, 3, 3, rng)
    avg = sum(evaluate_rollout(near, 3, 3, rng) for _ in range(500)) / 500
    print("非终局叶子(黑大优, 轮到黑)：单次 v=%.0f；500 次平均 v=%.3f（应明显>0）" % (one, avg))

    # —— 第⑤块：回传 ——
    print("\n--- 回传测试 ---")
    s = GomokuEnv(3, 3).reset()
    r = Node(state=s)                          # 根
    mid = Node(state=s.copy(), parent=r)       # 中间层（对手在这里做决定）
    r.children[(0, 0)] = mid
    leaf2 = Node(state=s.copy(), parent=mid)   # 叶
    mid.children[(1, 1)] = leaf2
    backpropagate(leaf2, 1.0)                  # 叶视角 v=+1，往上灌
    print("叶 : N=%d W=%+.0f  (应 N=1 W=+1)" % (leaf2.N, leaf2.W))
    print("中 : N=%d W=%+.0f  (对手层，应 N=1 W=-1)" % (mid.N, mid.W))
    print("根 : N=%d W=%+.0f  (与叶同方，应 N=1 W=+1)" % (r.N, r.W))

    # —— 第⑥块：完整 search ——
    print("\n--- search 测试（near：黑差一步赢，应锁定制胜手 (0,2)）---")
    best, root_s = mcts_search(near.state, n_simulations=200,
                               board_size=3, win_length=3, c_puct=1.0, rng=rng)
    print("MCTS 落子：", best, " （期望 (0,2)）")
    print("各孩子按访问次数排序：")
    for a, ch in sorted(root_s.children.items(), key=lambda kv: -kv[1].N):
        print("  %s : N=%3d  对黑价值 -Q=%+.2f" % (a, ch.N, -ch.Q))

    # —— 第⑦块：把（未训练的）双头网络接进来，先看一次前向给的 P 和 v ——
    print("\n--- 网络接入测试（网络【未训练】，应 P≈均匀、v≈乱、近 0）---")
    from aznet import PolicyValueNet
    net = PolicyValueNet(board_size=3)
    net.eval()
    legal = [(int(r), int(c)) for r, c in np.argwhere(near.state == 0)]
    priors, v = net_evaluate(near.state, legal, net, device="cpu")
    print("near 合法走法的先验 P（均匀应≈1/%d=%.3f）：" % (len(legal), 1 / len(legal)))
    for a, p in priors.items():
        print("  %s: P=%.3f" % (a, p))
    print("near 的 value v = %+.3f  （未训练 → 该≈乱、近 0）" % v)
