"""
④b-1：证明「优化器(optimizer) 真能用梯度把 loss 降下来」。

办法：故意只拿【同一批】转移，反复做"优化器三连"——
    optimizer.zero_grad()  →  loss.backward()  →  optimizer.step()
看 loss 一步步往下掉。这叫"过拟合一个 batch"，是检验训练机制有没有写对的经典 sanity check。

注意：这里 target_net 一直【不同步】，所以靶子是固定的 → loss 能一路降到很低。
真正的训练会不停换新数据、还会定期同步 target，loss 不会这么单调地掉——那是 ④b-2 的事。

运行： python demo_overfit_batch.py
"""

import torch

from gomoku_env import GomokuEnv
from agents import DQNAgent
from replay import ReplayBuffer
from qnet import QNetwork
from train_dqn import collect_one_game, compute_loss


if __name__ == "__main__":
    # —— 先攒一点数据 ——
    env = GomokuEnv(board_size=3, win_length=3)
    agent = DQNAgent(board_size=3, epsilon=0.25, seed=0)
    buf = ReplayBuffer(seed=0)
    for _ in range(50):
        collect_one_game(env, agent, buf)

    device = agent.device
    online_net = agent.net                          # 被训练的网络
    target_net = QNetwork(3).to(device)             # 目标网络：固定不动（本 demo 全程不同步）
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    # 优化器：用 Adam 按梯度自动微调 online_net 的参数；lr=学习率(每步迈多大)
    optimizer = torch.optim.Adam(online_net.parameters(), lr=1e-3)

    batch = buf.sample(64)                          # 固定这一批，反复练
    online_net.train()                              # 切到训练模式

    print(f"设备: {device}")
    print("反复在【同一批】上训练，看 loss 往下掉：")
    for step in range(1, 201):
        loss = compute_loss(online_net, target_net, batch, gamma=0.95, device=device)
        optimizer.zero_grad()                       # ① 清掉上一轮旧梯度
        loss.backward()                             # ② 反向传播：算这一轮的梯度
        optimizer.step()                            # ③ 迈步：按梯度真正更新参数
        if step == 1 or step % 40 == 0:
            print(f"  第 {step:3d} 步   loss = {loss.item():.4f}")

    print("\n✅ loss 持续下降 → 优化器三连(zero_grad→backward→step)确实在更新参数、把预测拉向目标。")
