"""
评估：纯 rollout 版 MCTS 打随机，验证 MCTS 引擎作为【完整棋手】的棋力（3×3）。
（还没接神经网络——只有"选择/扩展/rollout/回传"四步搜索，就已经会算棋了。）
"""

from collections import Counter

from gomoku_env import GomokuEnv
from agents import RandomAgent
from play import play_game
from mcts import MCTSAgent


def run(n, mcts_is_black, n_sim=200, net=None, board_size=3, win_length=3):
    env = GomokuEnv(board_size, win_length)
    mcts = MCTSAgent(board_size, win_length, n_simulations=n_sim, seed=0, net=net, device="cpu")
    rand = RandomAgent("随机", seed=1)
    black, white = (mcts, rand) if mcts_is_black else (rand, mcts)
    tally = Counter()
    for _ in range(n):
        tally[play_game(env, black, white)] += 1
    return tally   # 键：1=黑胜 / 2=白胜 / None=平


if __name__ == "__main__":
    N = 50
    print("=== 纯 rollout MCTS(200 模拟/步)  vs  随机，3×3，各 %d 局 ===\n" % N)

    t = run(N, mcts_is_black=True)
    print("MCTS 执黑：胜 %2d (%.0f%%) / 负 %2d / 平 %2d"
          % (t[1], 100 * t[1] / N, t[2], t[None]))

    t = run(N, mcts_is_black=False)
    print("MCTS 执白：胜 %2d (%.0f%%) / 负 %2d / 平 %2d"
          % (t[2], 100 * t[2] / N, t[1], t[None]))

    # —— 对照：接上【未训练】网络（value 在乱猜）→ 应明显弱于 rollout 版 ——
    from aznet import PolicyValueNet
    net = PolicyValueNet(3)
    net.eval()
    t = run(N, mcts_is_black=True, net=net)
    print("\n未训练网络版 MCTS 执黑：胜 %2d (%.0f%%) / 负 %2d / 平 %2d   ← 对比 rollout 的 100%%"
          % (t[1], 100 * t[1] / N, t[2], t[None]))
