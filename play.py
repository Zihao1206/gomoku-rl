"""
对弈循环（Game Loop）—— 第二阶段

把第一课那张"Agent ⇄ Environment 心跳图"用代码真正跑起来：
让两个 agent 从开局自动下到分胜负。本阶段还没有"学习"，纯粹是搭框架。
"""

from collections import Counter

from gomoku_env import GomokuEnv
from agents import RandomAgent


def play_game(env, black_agent, white_agent, render=False):
    """
    让两个 agent 下完整的一局，返回赢家：env.BLACK / env.WHITE / None(平局)。

    这个 while 循环就是 RL 的"心跳"：
        看状态 → agent 选动作 → env.step 给出新状态+奖励 → 再看 → ... 直到结束
    """
    state = env.reset()          # ① 开新局，拿到初始状态
    done = False
    if render:
        env.render()

    while not done:              # 没结束就一直下
        # ② 看现在轮到谁（黑还是白），就让对应的 agent 来决定这一手
        if env.current_player == env.BLACK:
            action = black_agent.select_action(state, env.legal_actions())
        else:
            action = white_agent.select_action(state, env.legal_actions())

        # ③ 把动作交给环境，推进一步，拿回新状态/奖励/是否结束
        state, reward, done, info = env.step(action)

        if render:
            env.render()

    return env.winner            # ④ 一局结束，返回赢家


if __name__ == "__main__":
    env = GomokuEnv(board_size=3, win_length=3)      # 3×3 三连（井字棋）看得清楚
    black = RandomAgent("黑·随机", seed=0)
    white = RandomAgent("白·随机", seed=1)

    # —— 1) 完整看一局随机 vs 随机 ——
    print("================ 随机 vs 随机：完整一局 ================\n")
    winner = play_game(env, black, white, render=True)

    # —— 2) 连下很多局，统计胜负分布（体会"先手优势"）——
    n = 2000
    tally = Counter()
    for _ in range(n):
        tally[play_game(env, black, white, render=False)] += 1

    label = {env.BLACK: "黑(先手)胜", env.WHITE: "白(后手)胜", None: "平局"}
    print(f"================ 随机对随机 {n} 局统计 ================")
    for key in (env.BLACK, env.WHITE, None):
        cnt = tally[key]
        print(f"{label[key]:>10}: {cnt:5d}  ({cnt / n:.1%})")
