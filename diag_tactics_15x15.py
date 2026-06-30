"""
阶段8 · 固定战术诊断（15×15, win5）——把 diag_tactics2.py 从 6×6(win4) 泛化而来。

诊断思路不变（验收"网络到底会不会防"，防 loss低≠棋力强）：
  人工摆一个"对手已连 4 子、我方此时必须堵某一点否则下一步就输"的局面，
  看网络 policy 有没有把那个【唯一必防点】排在第一，以及 MCTS 最终选不选它。
  Test 2：用旋转/镜像造横/竖/主对角/反对角【除方向外等价】的局面，验收朝向等变性
  （卷积只保证平移不变、不保证旋转不变 → 网络可能有方向偏好）。

与 6×6 版的差异：
  · 棋盘 15×15、win5 → 必防局面从"3 连"改"4 连"（连 4，第 5 子就赢）。
  · 网络换成 PolicyValueNetResNet，加载 az_resnet_15x15.pt。
  · 模型不存在时用【随机初始化网络】跑，仅验证仪器机械正确性（断言/渲染/查询）。
约定 Q = -child.Q（当前落子方视角）。只诊断，不加 Dirichlet、不训练。

用法：python diag_tactics_15x15.py [model_path]
"""
import os
import sys
import random
import numpy as np
import torch

from gomoku_env import GomokuEnv
from aznet_resnet import PolicyValueNetResNet
from mcts import mcts_search, net_evaluate, _infer_player

BS, WL = 15, 5
CHANNELS, NUM_BLOCKS = 64, 5
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "az_resnet_15x15.pt"

net = PolicyValueNetResNet(BS, channels=CHANNELS, num_blocks=NUM_BLOCKS)
if os.path.exists(MODEL_PATH):
    net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    print("已加载模型：%s" % MODEL_PATH)
else:
    print("⚠️  模型 %s 不存在 → 用【随机初始化网络】跑，仅验证仪器机械正确性"
          "（数字无意义，断言/渲染/查询能通即证尺造对了）。" % MODEL_PATH)
net.eval()
ZERO = np.zeros_like(GomokuEnv(BS, WL).reset())


def board_from(black, white):
    b = ZERO.copy()
    for r, c in black: b[r, c] = 1
    for r, c in white: b[r, c] = 2
    return b


def render(b):
    print("     " + " ".join("%2d" % c for c in range(BS)))      # 两位数列号对齐
    sym = {0: " .", 1: " X", 2: " O"}
    for r in range(BS):
        print("%2d   " % r + " ".join(sym[int(b[r, c])] for c in range(BS)))


def move_wins(b, cell, player):
    if b[cell] != 0: return False
    env = GomokuEnv(BS, WL); env.board = b.copy()
    env.current_player = player; env.done = False; env.winner = None
    env.step(cell)
    return env.winner == player


def attacker_wins(b, attacker):
    return [(int(r), int(c)) for r, c in np.argwhere(b == 0)
            if move_wins(b, (int(r), int(c)), attacker)]


def assert_single_block(b, defender, attacker):
    """自动校验 3 条件，返回唯一必防点：落子方=防守方 / 不堵则对手赢且唯一 / 堵后对手无立即胜点。"""
    assert _infer_player(b) == defender, "落子方不是防守方"
    th = attacker_wins(b, attacker)
    assert len(th) == 1, f"威胁点不唯一: {th}"
    mb = th[0]
    b2 = b.copy(); b2[mb] = defender
    assert attacker_wins(b2, attacker) == [], "堵住后对手仍有立即胜点"
    return mb


def moveQ(child): return -child.Q


def report(title, b, key, sims=(100, 400)):
    player = _infer_player(b)
    legal = [(int(r), int(c)) for r, c in np.argwhere(b == 0)]
    priors, v = net_evaluate(b, legal, net, "cpu")
    ranked = sorted(priors.items(), key=lambda kv: -kv[1])
    rank = {a: i + 1 for i, (a, _) in enumerate(ranked)}
    print("\n" + "-" * 64)
    print("%s ｜ 轮到 %s" % (title, "黑X" if player == 1 else "白O"))
    render(b)
    print("  网络 value v=%+.3f ｜ policy前3: %s"
          % (v, "  ".join("%s=%.3f" % (a, p) for a, p in ranked[:3])))
    for k in key:
        print("  关键点 %s: policy #%d/%d (p=%.4f)" % (k, rank[k], len(legal), priors[k]))
    for n in sims:
        best, root = mcts_search(b, n, BS, WL, 1.0, random.Random(0), net=net, device="cpu")
        hit = "✅" if best in key else "❌"
        kn = "  ".join("%s:N=%d Q=%+.2f" % (k, root.children[k].N, moveQ(root.children[k])) for k in key)
        print("    n=%4d → 选 %s %s ｜ %s" % (n, best, hit, kn))


print("\n##### Test 2：单端【四连】·唯一防守点，四方向（组内严格等价）#####")
# 横：黑 4 连在第 7 行 c4..c7，左端 (7,3) 被白堵 → 唯一胜点=右端 (7,8)，白必堵。
# 白另放 2 个远角填子，使 黑4=白3+... 等 → 黑数=白数+1 → 轮到白（防守方）。
base_h = board_from(black=[(7, 4), (7, 5), (7, 6), (7, 7)],
                    white=[(7, 3), (0, 0), (14, 14)])
# 主对角：黑 4 连 (5,5)(6,6)(7,7)(8,8)，左上端 (4,4) 被白堵 → 唯一胜点=(9,9)。
base_d = board_from(black=[(5, 5), (6, 6), (7, 7), (8, 8)],
                    white=[(4, 4), (0, 14), (14, 0)])
variants = [
    ("横 H", base_h),
    ("竖 V", np.ascontiguousarray(np.rot90(base_h))),       # 旋转：横→竖，严格等价
    ("主对角 ↘", base_d),
    ("反对角 ↙", np.ascontiguousarray(np.fliplr(base_d))),  # 镜像：主→反，严格等价
]
for name, b in variants:
    mb = assert_single_block(b, defender=2, attacker=1)     # 自动校验 + 求唯一必防点
    report("方向 %s ｜ 必防点 %s" % (name, mb), b, [mb])

print("\n（仪器自检：四向断言全过 = 必防局面构造正确，可用于长训后验收四向防守。）")
