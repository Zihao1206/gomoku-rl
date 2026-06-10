"""
训练循环（自我对弈）—— 第三阶段 3.3

让 QAgent 自己跟自己下成千上万局，每步用 update 复盘学习；
训练完，拉它去打 RandomAgent，看胜率能不能碾压随机基准线。

核心难点是"两人对弈的功劳分配"：
  · 把对手看成环境的一部分 → 某一方的"下一步状态"是它【再次轮到】时的局面；
  · 一局结束时，赢家最后一手 +1、输家最后一手 −1、平局都 0。
"""

from collections import Counter

from gomoku_env import GomokuEnv
from agents import QAgent, RandomAgent
from play import play_game


def play_one_selfplay_game(env, agent):
    """一局自我对弈，过程中让 agent 学习。返回赢家(BLACK/WHITE/None)。"""
    state = env.reset()
    BLACK, WHITE = env.BLACK, env.WHITE
    # pending：各方"已经落子、但还没结算"的那一手 (state, action)
    pending = {BLACK: None, WHITE: None}

    done = False
    while not done:
        p = env.current_player
        legal = env.legal_actions()

        # —— 结算 p 的上一手 ——
        # p 又轮到了，说明它上一手之后对手回应了、且没分胜负 →
        # 那一手的 reward=0、"下一步局面"就是眼前这个 state、非终局
        if pending[p] is not None:
            ps, pa = pending[p]
            agent.update(ps, pa, 0.0, state, legal, done=False)

        # —— p 选一步并落子 ——
        action = agent.select_action(state, legal)
        next_state, reward, done, _ = env.step(action)
        pending[p] = (state, action)     # 记下这一手，等下次轮到 p 时再结算
        state = next_state

    # —— 终局结算：给"赢家/平局的最后一手"和"对手的最后一手"各补一刀 ——
    # 赢棋时 env.step 不换手，所以 current_player 仍是最后落子者(=赢家)
    last = env.current_player
    other = WHITE if last == BLACK else BLACK
    winner = env.winner

    if winner is None:                   # 平局
        r_last, r_other = 0.0, 0.0
    else:                                # last 赢了，other 输了
        r_last, r_other = 1.0, -1.0

    ls, la = pending[last]
    agent.update(ls, la, r_last, None, [], done=True)   # 终局 → next_state 用不上
    if pending[other] is not None:
        os_, oa = pending[other]
        agent.update(os_, oa, r_other, None, [], done=True)

    return winner


def evaluate(agent, opponent, n_games=2000, board_size=3, win_length=3, agent_is_black=True):
    """评估：agent 不探索(ε=0)、不学习，纯凭本事打 opponent。返回 {win/lose/draw: 次数}。"""
    env = GomokuEnv(board_size, win_length)
    saved_eps = agent.epsilon
    agent.epsilon = 0.0                  # 评估时只利用、不探索
    tally = Counter()
    for _ in range(n_games):
        if agent_is_black:
            w = play_game(env, agent, opponent)      # play_game 只调 select_action，不学习
            tally["win" if w == env.BLACK else "lose" if w == env.WHITE else "draw"] += 1
        else:
            w = play_game(env, opponent, agent)
            tally["win" if w == env.WHITE else "lose" if w == env.BLACK else "draw"] += 1
    agent.epsilon = saved_eps
    return tally


def train_selfplay(agent, n_games, board_size=3, win_length=3,
                   eps_start=0.30, eps_end=0.02, eval_every=0, eval_opponent=None):
    """自我对弈训练 n_games 局；ε 从 eps_start 线性衰减到 eps_end（先广撒网、后凭本事）。"""
    env = GomokuEnv(board_size, win_length)
    for i in range(1, n_games + 1):
        agent.epsilon = eps_start + (eps_end - eps_start) * (i / n_games)
        play_one_selfplay_game(env, agent)

        if eval_every and eval_opponent is not None and i % eval_every == 0:
            t = evaluate(agent, eval_opponent, 1000, board_size, win_length, agent_is_black=True)
            tot = sum(t.values())
            print(f"  已训练 {i:>6} 局 | 作黑：胜 {t['win']/tot:5.1%}  负 {t['lose']/tot:5.1%}  平 {t['draw']/tot:5.1%}")


if __name__ == "__main__":
    BS, WL = 3, 3
    agent = QAgent(name="Q", epsilon=0.3, alpha=0.2, gamma=0.95, seed=0)
    rand = RandomAgent(seed=123)

    print("===== 训练前（应≈随机基准 58%）=====")
    t0 = evaluate(agent, rand, 2000, BS, WL, agent_is_black=True)
    print("  作黑:", dict(t0))

    N = 200000
    print(f"\n===== 自我对弈训练 {N} 局（看胜率往上爬）=====")
    train_selfplay(agent, N, BS, WL, eps_start=0.30, eps_end=0.02,
                   eval_every=N // 5, eval_opponent=rand)

    print("\n===== 训练后 =====")
    tb = evaluate(agent, rand, 3000, BS, WL, agent_is_black=True)
    tw = evaluate(agent, rand, 3000, BS, WL, agent_is_black=False)
    print(f"  作黑(先手): 胜 {tb['win']/3000:.1%}  负 {tb['lose']/3000:.1%}  平 {tb['draw']/3000:.1%}")
    print(f"  作白(后手): 胜 {tw['win']/3000:.1%}  负 {tw['lose']/3000:.1%}  平 {tw['draw']/3000:.1%}")
    print(f"  学到的局面数(Q 表大小): {len(agent.Q)}")

    agent.save("q_table_3x3.pkl")
    print("  已保存 Q 表到 q_table_3x3.pkl（下次人机对战直接加载，不用重训）")
