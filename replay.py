"""
经验回放池（Replay Buffer）—— 阶段 5.3 的第①块

DQN 和 Q 表学的目标一样（都想把 Q(s,a) 拟合到 r + γ·max Q(s',·)），
但神经网络「改一次参数会牵动所有局面」，没法像查表那样即时精确改一格；
而且「同一局连续几步」高度相关，直接拿来训练会学偏。

解药就是这个盒子：
  · 先把每一步的「转移(transition)」攒进来；
  · 训练时【随机抽一批】出来做梯度下降。
这样样本被打散、更像独立抽题，训练更稳；一份经历还能反复抽用，更省数据。

本文件只负责「存 + 随机抽」，不碰网络、不算损失——那是后面几步的事。
"""

import random
from collections import deque, namedtuple


# 一条「转移 / 经历」= 在 state（此刻轮到 player）下走了 action_idx，
# 拿到 reward，进入 next_state（轮到 next_player，合法动作 next_legal），done 表示是否终局。
#   · action_idx 是「格子编号 = row * W + col」（和 qnet / DQNAgent 的约定一致）
#   · 字段顺序严格照 HANDOFF §5.3 定的那套，后面收集/训练都按这个来
Transition = namedtuple(
    "Transition",
    ["state", "player", "action_idx", "reward",
     "next_state", "next_player", "next_legal", "done"],
)


class ReplayBuffer:
    """固定容量的经验池：满了之后自动挤掉最旧的，只留最近 capacity 条。"""

    def __init__(self, capacity=10000, seed=None):
        # deque(maxlen=N)：从一端 append，满了会自动从另一端挤掉最旧的，
        # 天然就是「只保留最近 N 条」，不用自己手写删除逻辑。
        self.buf = deque(maxlen=capacity)
        # 自带随机源（和 RandomAgent / QAgent 一个风格）：传 seed 就能复现同样的抽样
        self.rng = random.Random(seed)

    def push(self, state, player, action_idx, reward,
             next_state, next_player, next_legal, done):
        """存一条转移（字段含义见上面 Transition 的注释）。"""
        self.buf.append(Transition(state, player, action_idx, reward,
                                    next_state, next_player, next_legal, done))

    def sample(self, batch_size):
        """随机抽 batch_size 条（不放回）。返回一个 Transition 列表。"""
        return self.rng.sample(self.buf, batch_size)

    def __len__(self):
        """当前存了多少条——训练前常用它判断「攒够一批没」。"""
        return len(self.buf)


# ———————————————————————————————————————————————
# 直接运行本文件（python replay.py）跑下面这段自测
# ———————————————————————————————————————————————
if __name__ == "__main__":
    import numpy as np

    print("===== ReplayBuffer 自测 =====")
    buf = ReplayBuffer(capacity=5, seed=0)   # 容量设小演示「挤掉旧的」；seed=0 让抽样可复现
    print("刚建好，空池长度:", len(buf), "（应为 0）")

    # 塞 7 条假转移：capacity=5，所以最后只会留下「最近 5 条」
    for i in range(7):
        fake = np.zeros((3, 3), dtype=np.int8)     # 假的棋盘，纯占位
        buf.push(state=fake, player=1, action_idx=i, reward=0.0,
                 next_state=fake, next_player=2, next_legal=[(0, 0)], done=False)

    print("塞了 7 条后长度:", len(buf), "（容量 5，应为 5）")
    print("池中各条的 action_idx:", [t.action_idx for t in buf.buf],
          "（应是最近 5 条 → 2,3,4,5,6）")

    # 随机抽一批，并演示「具名字段」取值很直观
    batch = buf.sample(3)
    print("\n随机抽 3 条（seed=0，每次运行结果都一样）：")
    for t in batch:
        print(f"  action_idx={t.action_idx}  reward={t.reward}  done={t.done}")

    print("\n✅ ReplayBuffer 基本功能正常：能存、会自动丢旧、能复现地随机抽批。")
