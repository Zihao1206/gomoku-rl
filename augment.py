"""
B1：对称性数据增强。利用棋盘的 8 重对称（D4：4 旋转 × 2 镜像）把每条训练样本扩成 8 条，
强制网络对所有朝向一视同仁 —— 正面修阶段7 诊断出的"朝向等变性盲区"。

核心：
  · state 与 π 都是【空间量·等变 equivariant】，必须用【完全相同】的变换同步旋转/镜像；
    π 长 n_cells，先 reshape 成 (H,W) 跟着变、再 flatten 回去 → 保证逐格对齐。
  · z 是【标量胜负·不变 invariant】，转/翻都不动。
"""
import numpy as np


def augment_example(state, pi, z, board_size):
    """一条 (state, π, z) → 8 条对称等价样本 [(state', π', z), ...]。"""
    pi2d = np.asarray(pi, dtype=np.float32).reshape(board_size, board_size)
    out = []
    for k in range(4):                       # 旋转 0/90/180/270
        for flip in (False, True):           # 各自再镜像与否 → 共 8（D4 全部元素）
            s = np.rot90(state, k)
            p = np.rot90(pi2d, k)            # 对 π 做【与棋盘相同】的旋转
            if flip:
                s = np.fliplr(s)
                p = np.fliplr(p)            # 镜像也同步
            out.append((np.ascontiguousarray(s), p.flatten().copy(), z))
    return out


if __name__ == "__main__":
    BS = 6
    # 探针：棋盘在 (1,2) 放一颗子，π 的尖峰也设在【同一格】(1,2)。
    # 若 π 正确地跟着棋盘变换，则每个对称版本里"子的位置"必等于"π 尖峰的位置"。
    state = np.zeros((BS, BS), dtype=np.int8); state[1, 2] = 1
    pi = np.zeros(BS * BS, dtype=np.float32); pi[1 * BS + 2] = 1.0

    aug = augment_example(state, pi, z=1.0, board_size=BS)
    print("扩出样本数:", len(aug), "(应 8)")

    seen = set()
    for i, (s, p, z) in enumerate(aug):
        stone = tuple(int(x) for x in np.argwhere(s == 1)[0])
        peak = tuple(int(x) for x in np.unravel_index(int(p.argmax()), (BS, BS)))
        assert stone == peak, f"第{i}个: 子在{stone} 但 π 尖峰在{peak} —— π 没跟着棋盘变!"
        assert abs(float(p.sum()) - 1.0) < 1e-6, "π 不再是概率分布"
        assert z == 1.0, "z 不应改变"
        seen.add(s.tobytes())
        print("  样本%d: 子%s == π尖峰%s ✅  z=%.0f" % (i, stone, peak, z))
    print("8 个版本互不相同:", len(seen) == 8, "(子在 (1,2) 非对称点，应 True)")
    print("\n✅ 通过：state 与 π 同步变换、z 不变、覆盖 8 重对称")
