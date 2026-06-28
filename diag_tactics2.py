"""
固定战术诊断 · 第二版（更严谨）。回应两点修正：
  (1) 两头空的"活三"已必败，堵任一头只是延迟失败 → 不能证明谁对谁错。
      真正该查的是【上一手为何允许活三形成】→ Test 1：把实战各回退一手。
  (2) 方向盲区要用"除方向外等价"的局面证。但棋盘对称群里 {横,竖} 与 {主对角,反对角}
      是两条不同轨道（横向步距 (0,1)、对角步距 (1,1)，几何本就不同），无法跨组完全等价。
      故 Test 2：组内(横≡竖、主≡反)用旋转/镜像严格等价，跨组(轴向 vs 对角)做对比。
      （卷积只保证平移不变，不保证旋转不变 → 网络可能有方向偏好。）

每个构造局面自动断言：必防点唯一且合法 / 不堵则对手下一步赢 / 堵住后对手无立即胜点。
只诊断，不加 Dirichlet、不训练。 约定 Q = -child.Q（当前落子方视角）。
"""
import random
import numpy as np
import torch

from gomoku_env import GomokuEnv
from aznet_cnn import PolicyValueNetCNN
from mcts import mcts_search, net_evaluate, _infer_player

BS, WL = 6, 4
net = PolicyValueNetCNN(BS)
net.load_state_dict(torch.load("az_cnn_6x6.pt", map_location="cpu"))
net.eval()
ZERO = np.zeros_like(GomokuEnv(BS, WL).reset())


def board_from(black, white):
    b = ZERO.copy()
    for r, c in black: b[r, c] = 1
    for r, c in white: b[r, c] = 2
    return b


def replay(moves):
    env = GomokuEnv(BS, WL); s = env.reset()
    for mv in moves:
        s, _, done, _ = env.step(mv)
        assert not done, f"setup {mv} 意外终局"
    return s


def render(b):
    print("    " + " ".join(str(c) for c in range(BS)))
    sym = {0: ".", 1: "X", 2: "O"}
    for r in range(BS):
        print(" %d  " % r + " ".join(sym[int(b[r, c])] for c in range(BS)))


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
    """自动校验 3 条件，并返回唯一必防点。"""
    assert _infer_player(b) == defender, "落子方不是防守方"
    th = attacker_wins(b, attacker)
    assert len(th) == 1, f"威胁点不唯一: {th}"          # 不堵则对手下一步赢，且唯一
    mb = th[0]
    b2 = b.copy(); b2[mb] = defender
    assert attacker_wins(b2, attacker) == [], "堵住后对手仍有立即胜点"
    return mb


def moveQ(child): return -child.Q


def report(title, b, key, sims=(100, 400, 1600)):
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
        print("  关键点 %s: policy #%d/%d (p=%.3f)" % (k, rank[k], len(legal), priors[k]))
    for n in sims:
        best, root = mcts_search(b, n, BS, WL, 1.0, random.Random(0), net=net, device="cpu")
        hit = "✅" if best in key else "❌"
        kn = "  ".join("%s:N=%d Q=%+.2f" % (k, root.children[k].N, moveQ(root.children[k])) for k in key)
        print("    n=%4d → 选 %s %s ｜ %s" % (n, best, hit, kn))


print("##### Test 1：把实战回退一手，能否阻止对手活三成型 #####")
# 局面1 回退到 AI 上一手：黑活两 (2,2)(3,3)，白应占 (1,1) 或 (4,4) 阻断对角
report("局面1-回退（白防, 阻断点 (1,1)/(4,4)）",
       replay([(3, 3), (2, 3), (2, 2)]), [(1, 1), (4, 4)])
# 局面2 回退到 AI 上一手：白活两 (1,3)(2,2)，黑应占 (3,1)/(0,4)/(4,0) 阻断反对角
report("局面2-回退（黑防, 阻断点 (3,1)/(0,4)/(4,0)）",
       replay([(2, 3), (2, 2), (3, 3), (1, 3), (4, 3), (5, 3)]), [(3, 1), (0, 4), (4, 0)])

print("\n\n##### Test 2：单端三连·唯一防守点，四方向（组内严格等价）#####")
base_h = board_from(black=[(2, 1), (2, 2), (2, 3)], white=[(2, 0), (5, 5)])
base_d = board_from(black=[(1, 1), (2, 2), (3, 3)], white=[(0, 0), (5, 0)])
variants = [
    ("横 H", base_h),
    ("竖 V", np.ascontiguousarray(np.rot90(base_h))),       # 旋转：横→竖，严格等价
    ("主对角 ↘", base_d),
    ("反对角 ↙", np.ascontiguousarray(np.fliplr(base_d))),  # 镜像：主→反，严格等价
]
for name, b in variants:
    mb = assert_single_block(b, defender=2, attacker=1)     # 自动校验 + 求唯一必防点
    report("方向 %s ｜ 必防点 %s" % (name, mb), b, [mb])
