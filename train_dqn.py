"""
DQN 训练器 —— 阶段 5.3

· collect_one_game（③）：自对弈，把每一步打包成 Transition 存进 ReplayBuffer；
  功劳分配【完全照搬】train.py。
· compute_loss（④a）：把一批转移算成一个 MSE 损失。
· train_dqn（④b）：完整训练循环 = 边自对弈攒数据 + 每步抽新批做"优化器三连"
  + 每隔 N 步同步 target + 定期评估；可选 save_path 训练完存盘。

命令行：
    python train_dqn.py                 # 默认 3×3(win3)，3000 局，存 dqn_3x3.pt
    python train_dqn.py 5 4 6000        # 5×5(win4)，6000 局，存 dqn_5x5.pt
"""

from gomoku_env import GomokuEnv
from agents import DQNAgent, RandomAgent
from replay import ReplayBuffer
from play import play_game
from train import evaluate           # 复用表格阶段的评估（统一接口的好处）

import torch
from qnet import encode_board, QNetwork


def collect_one_game(env, agent, buffer):
    """自对弈一局，把每一步打包成 Transition 存进 buffer。返回赢家。"""
    state = env.reset()
    BLACK, WHITE = env.BLACK, env.WHITE
    W = env.board_size
    pending = {BLACK: None, WHITE: None}     # 各方「已落子、还没存进 buffer」的那一手 (state, action_idx)

    done = False
    while not done:
        p = env.current_player
        legal = env.legal_actions()

        # 结算 p 的上一手：p 又轮到了 → reward=0、非终局、next_state=眼前局面、next_player=p 自己
        if pending[p] is not None:
            ps, pa = pending[p]
            buffer.push(ps, p, pa, 0.0, state, p, legal, False)

        action = agent.select_action(state, legal)
        a_idx = action[0] * W + action[1]
        next_state, reward, done, _ = env.step(action)
        pending[p] = (state, a_idx)
        state = next_state

    # 终局结算：赢家/平局最后一手 + 对手最后一手，各补一条（done=True）
    last = env.current_player
    other = WHITE if last == BLACK else BLACK
    winner = env.winner
    r_last, r_other = (0.0, 0.0) if winner is None else (1.0, -1.0)

    ls, la = pending[last]
    buffer.push(ls, last, la, r_last, None, last, [], True)
    if pending[other] is not None:
        os_, oa = pending[other]
        buffer.push(os_, other, oa, r_other, None, other, [], True)
    return winner


def compute_loss(online_net, target_net, batch, gamma, device):
    """
    对一批转移算 MSE 损失（教学版用循环，易懂；以后要快可向量化）。
      pred  = Q_online(s)[a]                       —— 带梯度，要被训练
      target = r                                    （终局）
             = r + γ·max_合法 Q_target(s')          （非终局；只在合法格里取 max）
        target 全程 no_grad → 钉死的常数
      loss = mean( (pred - target)^2 )
    """
    W = batch[0].state.shape[1]
    preds, targets = [], []
    for tr in batch:
        x = encode_board(tr.state, tr.player).unsqueeze(0).to(device)
        preds.append(online_net(x)[0][tr.action_idx])         # 带梯度

        if tr.done:
            target = tr.reward
        else:
            xn = encode_board(tr.next_state, tr.next_player).unsqueeze(0).to(device)
            with torch.no_grad():
                qn = target_net(xn)[0]
            legal_idx = [r * W + c for (r, c) in tr.next_legal]
            target = tr.reward + gamma * max(qn[i].item() for i in legal_idx)
        targets.append(target)

    preds = torch.stack(preds)
    targets = torch.tensor(targets, dtype=torch.float32, device=device)
    return ((preds - targets) ** 2).mean()


def train_dqn(agent, n_games, board_size=3, win_length=3,
              gamma=0.95, lr=1e-3, batch_size=64,
              capacity=20000, warmup=500, sync_every=100,
              eps_start=0.30, eps_end=0.05,
              eval_every=0, eval_opponent=None, save_path=None):
    """完整 DQN 自对弈训练。save_path 给了就在训练完存盘。返回训练好的网络。"""
    env = GomokuEnv(board_size, win_length)
    device = agent.device

    online_net = agent.net                          # agent 内部的网络 = 在线网络（被训练）
    online_net.train()
    target_net = QNetwork(board_size).to(device)    # 目标网络 = online 的冻结副本
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(online_net.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=capacity, seed=0)

    train_steps = 0
    for i in range(1, n_games + 1):
        agent.epsilon = eps_start + (eps_end - eps_start) * (i / n_games)   # ε 线性衰减
        collect_one_game(env, agent, buffer)        # 自对弈一局，攒转移

        if len(buffer) < warmup:                    # warm-up：没攒够先别训
            continue

        batch = buffer.sample(batch_size)           # 每步重新抽一批
        loss = compute_loss(online_net, target_net, batch, gamma, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_steps += 1

        if train_steps % sync_every == 0:           # 每隔 N 步同步 target
            target_net.load_state_dict(online_net.state_dict())

        if eval_every and eval_opponent is not None and i % eval_every == 0:
            t = evaluate(agent, eval_opponent, 1000, board_size, win_length, agent_is_black=True)
            tot = sum(t.values())
            print(f"  局 {i:>5} | 训练步 {train_steps:>5} | loss {loss.item():.4f} | "
                  f"作黑 胜 {t['win']/tot:5.1%}  负 {t['lose']/tot:5.1%}  平 {t['draw']/tot:5.1%}")
            online_net.train()

    if save_path:
        agent.save(save_path)
        print(f"  ✅ 模型已保存到 {save_path}")
    return online_net


if __name__ == "__main__":
    import sys
    BS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    WL = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    path = f"dqn_{BS}x{BS}.pt"

    # 小棋盘小网络：CPU 往往比 MPS 还快（MPS 调度开销在小张量上占比大）
    agent = DQNAgent(board_size=BS, name="DQN", device="cpu", seed=0)
    rand = RandomAgent(seed=123)

    print(f"===== 训练 {BS}x{BS}(win{WL}) DQN，自对弈 {N} 局 =====")
    train_dqn(agent, n_games=N, board_size=BS, win_length=WL,
              eval_every=max(1, N // 4), eval_opponent=rand, save_path=path)

    print("\n===== 训练后 vs 随机（黑白分开看）=====")
    b = evaluate(agent, rand, 2000, BS, WL, agent_is_black=True);  nb = sum(b.values())
    w = evaluate(agent, rand, 2000, BS, WL, agent_is_black=False); nw = sum(w.values())
    print(f"  作黑 胜 {b['win']/nb:.1%}  负 {b['lose']/nb:.1%}  平 {b['draw']/nb:.1%}")
    print(f"  作白 胜 {w['win']/nw:.1%}  负 {w['lose']/nw:.1%}  平 {w['draw']/nw:.1%}")
    print(f"\n模型在 {path}，现在可以开战： python play_human_dqn.py {BS} {WL}")
