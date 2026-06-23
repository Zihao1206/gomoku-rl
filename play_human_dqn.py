"""
人机对战（DQN 版）—— 跟你训练好的 DQN 模型下一盘。

用法：
    1. 先训练并保存模型：
         python train_dqn.py              # 3×3(win3)，存 dqn_3x3.pt
         python train_dqn.py 5 4 6000     # 5×5(win4)，存 dqn_5x5.pt
    2. 再开战：
         python play_human_dqn.py         # 默认 3×3
         python play_human_dqn.py 5 4     # 5×5
"""

import os
import sys

from gomoku_env import GomokuEnv
from agents import DQNAgent, HumanAgent
from play import play_game

BS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
WL = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MODEL = sys.argv[3] if len(sys.argv) > 3 else f"dqn_{BS}x{BS}.pt"


def main():
    if not os.path.exists(MODEL):
        print(f"找不到模型 {MODEL}，请先训练：  python train_dqn.py {BS} {WL}")
        return

    # ε=0：只用本事、绝不乱下；device 用 cpu（人机对战一次只算一步，无所谓快慢）
    ai = DQNAgent(board_size=BS, name="AI", epsilon=0.0, device="cpu")
    ai.load(MODEL)
    human = HumanAgent()
    env = GomokuEnv(BS, WL)

    print("=" * 48)
    print(f" 跟你训练的 DQN 下 {BS}×{BS}（{WL} 连胜），坐标从 0 开始")
    print(" 落子格式：行 列，例如  1 2 ；输 q 退出")
    print("=" * 48)

    side = input("你执黑(先手 X) 还是 白(后手 O)？ 输 b/w [默认 b]: ").strip().lower()

    # HumanAgent 和 DQNAgent 都遵守同一接口，直接复用 play_game
    if side == "w":
        winner = play_game(env, ai, human, render=True)     # AI 执黑 / 你执白
        human_color = env.WHITE
    else:
        winner = play_game(env, human, ai, render=True)     # 你执黑 / AI 执白
        human_color = env.BLACK

    if winner is None:
        print("结果：平局！")
    elif winner == human_color:
        print("结果：你赢了 DQN！🎉")
    else:
        print("结果：DQN 赢了，再来一局？")


if __name__ == "__main__":
    main()
