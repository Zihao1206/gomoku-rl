"""
固定战术诊断：把"人赢 AI"的两个真实局面 + 一个干净控制局面冻结下来逐个体检，
定位"AI 为什么不防守"到底卡在哪个环节（对应 4 个假设）：
    H1 网络没学会威胁  → 看网络 raw policy 给"必防点"的排名/先验
    H2 先验太低没被搜到 → 看 100→1600 模拟下，必防点 N 是否随预算上来、最终是否选它
    H3 value 误判      → 看根局面网络给的 v、以及最佳手 Q 是否正确逼近 -1（已输）
    H4 MCTS 局部 bug   → 和【纯 rollout 版】同局面对照：rollout 看得见(Q→-1)而网络看不见 → 锅在网络

约定：Q 一律换成【当前落子方视角】= -child.Q（+1 这步对我好 / -1 对我糟）。
控制组是"单边三连、必防点唯一 (2,4) 且堵了能救"——有明确正解，最能定性。
"""
import random
import numpy as np
import torch

from gomoku_env import GomokuEnv
from aznet_cnn import PolicyValueNetCNN
from mcts import mcts_search, net_evaluate, _infer_player

BS, WL = 6, 4
SIMS = [100, 200, 400, 1600]
net = PolicyValueNetCNN(BS)
net.load_state_dict(torch.load("az_cnn_6x6.pt", map_location="cpu"))
net.eval()


def replay(moves):
    """用真引擎重演到目标局面，顺便保证局面合法、且中途没人提前赢。"""
    env = GomokuEnv(BS, WL)
    s = env.reset()
    for mv in moves:
        s, _, done, _ = env.step(mv)
        assert not done, f"setup 落子 {mv} 意外终局"
    return s


def render(state):
    print("    " + " ".join(str(c) for c in range(BS)))
    sym = {0: ".", 1: "X", 2: "O"}
    for r in range(BS):
        print(" %d  " % r + " ".join(sym[int(state[r, c])] for c in range(BS)))


def moveQ(child):
    return -child.Q                      # 换成"当前落子方视角"


def dump_topk(root, threats, k=5):
    items = sorted(root.children.items(), key=lambda kv: -kv[1].N)[:k]
    for a, ch in items:
        mark = "  <<必防点" if a in threats else ""
        print("        %s  P=%.3f N=%4d Q=%+.2f%s" % (a, ch.P, ch.N, moveQ(ch), mark))


def diagnose(name, state, threats):
    player = _infer_player(state)
    who = "黑X" if player == 1 else "白O"
    legal = [(int(r), int(c)) for r, c in np.argwhere(state == 0)]
    print("\n" + "=" * 72)
    print("%s ｜ 轮到 %s 防守 ｜ 必防点 %s" % (name, who, threats))
    render(state)

    # —— H1/H3：网络直觉（与搜索无关）——
    priors, v = net_evaluate(state, legal, net, "cpu")
    ranked = sorted(priors.items(), key=lambda kv: -kv[1])
    rank_of = {a: i + 1 for i, (a, _) in enumerate(ranked)}
    print("\n[网络直觉] 根局面 value v = %+.3f   (若已输，应≈ -1)" % v)
    print("           policy 前3： " +
          " ， ".join("%s p=%.3f" % (a, p) for a, p in ranked[:3]))
    for t in threats:
        print("           必防点 %s ： 排名 #%d/%d，先验 p=%.3f"
              % (t, rank_of[t], len(legal), priors[t]))

    # —— H2/H4：搜索表，网络版 vs rollout 版，逐 sim ——
    for tag, use_net in [("网络版", net), ("rollout版", None)]:
        print("\n[%s]" % tag)
        for n in SIMS:
            best, root = mcts_search(state, n, BS, WL, 1.0, random.Random(0),
                                     net=use_net, device="cpu")
            blocked = "✅防" if best in threats else "❌没防"
            tline = "  ".join("%s:N=%d Q=%+.2f" % (t, root.children[t].N, moveQ(root.children[t]))
                              for t in threats)
            print("  n_sim=%4d → 选 %s %s  bestQ=%+.2f ｜ 必防点 %s"
                  % (n, best, blocked, moveQ(root.children[best]), tline))
            if tag == "网络版" and n == SIMS[-1]:
                print("     ↑ 该 sim 下访问最多的前5手 (P/N/Q)：")
                dump_topk(root, threats)


# 局面1：你执黑那局，白(AI)该防黑的对角活三 (2,2)(3,3)(4,4) → 必防点 (1,1)/(5,5)
diagnose("局面1（实战·白AI该防）",
         replay([(3, 3), (2, 3), (2, 2), (1, 2), (4, 4)]), [(1, 1), (5, 5)])

# 局面2：你执白那局，黑(AI)该防白的反对角活三 (1,3)(2,2)(3,1) → 必防点 (0,4)/(4,0)
diagnose("局面2（实战·黑AI该防）",
         replay([(2, 3), (2, 2), (3, 3), (1, 3), (4, 3), (5, 3), (1, 2), (3, 1)]),
         [(0, 4), (4, 0)])

# 控制组：单边三连，必防点唯一 (2,4)，堵了能救（最干净的正解）
diagnose("控制组（单边三连·正解唯一 (2,4)）",
         replay([(2, 1), (2, 0), (2, 2), (0, 0), (2, 3)]), [(2, 4)])
