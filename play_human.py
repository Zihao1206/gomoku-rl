"""
人机对战 —— 跟你训练好的 Q-Learning AI 下一盘井字棋(3×3 三连胜)。

用法：
    1. 先运行  python train.py   生成 q_table_3x3.pkl
    2. 再运行  python play_human.py   开战
"""

import os

from gomoku_env import GomokuEnv
from agents import QAgent, HumanAgent
from play import play_game

BS, WL = 3, 3
MODEL = "q_table_3x3.pkl"


def main():
    if not os.path.exists(MODEL):
        print(f"找不到模型 {MODEL}，请先运行：  python train.py")
        return

    ai = QAgent(name="AI", epsilon=0.0)    # ε=0：只用本事，绝不乱下
    ai.load(MODEL)
    human = HumanAgent()
    env = GomokuEnv(BS, WL)

    print("=" * 44)
    print(" 跟你的 AI 下井字棋（3×3 三连胜），坐标从 0 开始")
    print(" 它是个完美玩家，你最多只能逼平、赢不了它 😏")
    print("=" * 44)

    side = input("你执黑(先手 X) 还是 白(后手 O)？ 输 b/w [默认 b]: ").strip().lower()

    # HumanAgent 和 AI 都遵守同一接口，所以直接复用 play_game，循环一行没改
    if side == "w":
        winner = play_game(env, ai, human, render=True)     # AI 执黑 / 你执白
        human_color = env.WHITE
    else:
        winner = play_game(env, human, ai, render=True)     # 你执黑 / AI 执白
        human_color = env.BLACK

    if winner is None:
        print("结果：平局！你成功逼平了完美 AI 👏")
    elif winner == human_color:
        print("结果：你居然赢了？！那说明 AI 还没练到位，快回来告诉我 😄")
    else:
        print("结果：AI 赢了，再来一局？")


if __name__ == "__main__":
    main()
