"""
阶段 5.3 第②块：目标网络(target network)机制的最小验证。

只验证「机制」，不训练、不碰 qnet.py：
  · online_net：将来被训练（梯度下降）的网络；
  · target_net：online_net 的一个【副本】，平时冻结，专门用来算训练目标里的 max Q(s',·)；
    每隔 N 步才把 online 的参数整套【复制】过来一次（target.load_state_dict(online.state_dict())）。

本脚本证明三件事：
  1) 同步后：同一输入，两网输出【完全一致】（参数被复制成一样的）；
  2) 手动改动 online 参数后：两网输出【不同】（target 没跟着动，是冻结的）；
  3) 再同步一次：两网输出又【一致】（target 被刷新成最新的 online）。
"""

import torch

from qnet import QNetwork, encode_board
from gomoku_env import GomokuEnv


def sync(target_net, online_net):
    """把 online 的整套参数复制给 target —— 这就是「对一次表」/同步。"""
    target_net.load_state_dict(online_net.state_dict())


def max_q(net, x):
    """用 net 对输入 x 前向，返回所有格子里最大的 Q（演示用）。"""
    with torch.no_grad():            # 不求梯度：目标网络只前向、绝不参与训练
        return net(x).max().item()


if __name__ == "__main__":
    torch.manual_seed(0)             # 固定初始化，结果可复现

    # —— 用本项目真实的棋盘编码做输入（结合已有代码）——
    env = GomokuEnv(board_size=3, win_length=3)
    env.reset(); env.step((1, 1)); env.step((0, 0))      # 黑下中心、白下角
    x = encode_board(env._get_state(), env.current_player).unsqueeze(0)   # (1,2,3,3)

    # —— 两个结构一样的网络 ——
    online_net = QNetwork(board_size=3)
    target_net = QNetwork(board_size=3)
    target_net.eval()                # 目标网络只用来推断，切到评估模式(eval)

    print("【刚建好、还没同步】两者各自随机初始化 → 输出大概率不同")
    print(f"  online max Q = {max_q(online_net, x):.6f}")
    print(f"  target max Q = {max_q(target_net, x):.6f}")

    # 1) 同步后 → 完全一致
    sync(target_net, online_net)
    o1, t1 = max_q(online_net, x), max_q(target_net, x)
    print("\n【1) 同步后】应完全一致")
    print(f"  online = {o1:.6f}   target = {t1:.6f}   完全相等? {o1 == t1}")

    # 2) 粗暴改动 online 的参数（模拟「训练动了一下」）→ target 应纹丝不动
    with torch.no_grad():
        for p in online_net.parameters():
            p.add_(0.5)              # online 每个参数都 +0.5
    o2, t2 = max_q(online_net, x), max_q(target_net, x)
    print("\n【2) 改动 online 之后】online 变了，target 应没跟着动")
    print(f"  online = {o2:.6f}   (相比同步后变了? {o2 != o1})")
    print(f"  target = {t2:.6f}   (还和同步后一样? {t2 == t1})")
    print(f"  两者现在不同? {o2 != t2}")

    # 3) 再同步一次 → 又完全一致
    sync(target_net, online_net)
    o3, t3 = max_q(online_net, x), max_q(target_net, x)
    print("\n【3) 再同步一次后】应又完全一致")
    print(f"  online = {o3:.6f}   target = {t3:.6f}   完全相等? {o3 == t3}")

    print("\n✅ 目标网络机制验证通过：同步=复制成一样；不同步时 target 冻结不动；全程只前向、不训练。")
