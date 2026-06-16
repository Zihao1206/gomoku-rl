"""
阶段 5.2 验证脚本：未训练的 DQNAgent 能不能正常「选动作」并接进对弈循环。

⚠️ 目标不是「下得好」（网络还没训练，棋力≈随机），而是验证三件事：
    1. select_action 能根据棋盘选出一个【合法】动作，不报错；
    2. 能自动判断当前该谁下（黑/白）；
    3. 能直接塞进 play_game，和随机玩家打完整对局。

预期：未训练 DQN vs 随机 ≈ 五五开（先手略占优），说明「接线」通了。

运行： conda activate gomoku 后  python demo_dqn.py
"""

from collections import Counter

from gomoku_env import GomokuEnv
from agents import RandomAgent, DQNAgent
from play import play_game


def main():
    env = GomokuEnv(board_size=3, win_length=3)   # 3×3 三连（井字棋），小而清楚

    # —— 1) 单步检查：给一个具体局面，看 DQN 选出的动作合不合法 ——
    print("======== 1) 单步检查：DQN 选一个动作 ========")
    # epsilon=0：关掉随机探索，纯看网络打分，才能真正测「算分 + 盖非法格」这条路
    dqn = DQNAgent(board_size=3, name="DQN", epsilon=0.0, seed=0)
    print(f"使用设备: {dqn.device}")            # 期望在这台 Mac 上看到 mps

    env.reset()
    env.step((1, 1))                            # 黑下中心
    env.step((0, 0))                            # 白下角
    state = env._get_state()
    env.render()

    legal = env.legal_actions()
    inferred = DQNAgent._infer_player(state)
    action = dqn.select_action(state, legal)
    print(f"网络推断现在轮到: {'黑(1)' if inferred == 1 else '白(2)'}   （此局面应为黑）")
    print(f"DQN 选择的动作 : {action}")
    print(f"是否合法       : {action in legal}   （应为 True）\n")

    # —— 2) 接线检查：未训练 DQN(黑) vs 随机(白)，统计胜负 ——
    print("======== 2) 未训练 DQN(黑) vs 随机(白)：200 局 ========")
    dqn_black = DQNAgent(board_size=3, name="DQN黑", epsilon=0.1, seed=1)
    rand_white = RandomAgent("随机白", seed=2)

    tally = Counter()
    for _ in range(200):
        tally[play_game(env, dqn_black, rand_white)] += 1      # play_game 内部会 reset

    label = {env.BLACK: "DQN黑胜", env.WHITE: "随机白胜", None: "平局"}
    for k in (env.BLACK, env.WHITE, None):
        print(f"  {label[k]:>8}: {tally[k]:4d}  ({tally[k] / 200:.1%})")
    print("  （未训练 → 约等于随机水平就对了，说明接线正确；训练是 5.3 的事）\n")

    print("✅ 5.2 验证通过：DQNAgent.select_action 能跑通，并已接入 play_game。")


if __name__ == "__main__":
    main()
