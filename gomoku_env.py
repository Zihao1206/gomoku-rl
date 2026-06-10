"""
五子棋环境（Environment）—— 第一阶段 1b：加上裁判（胜负判断）

在 1a（棋盘骨架）的基础上，这一版给棋盘装上了"裁判"：
每次落子后会检测是否五子连珠，并据此决定输赢、发奖励、结束这一局。

设计要点：棋盘大小 board_size 和"几子连珠算赢" win_length 都是参数：
    GomokuEnv(board_size=3,  win_length=3)  → 井字棋（教学用，最简单）
    GomokuEnv(board_size=15, win_length=5)  → 标准五子棋
"""

import numpy as np


class GomokuEnv:
    # —— 用整数表示每个格子的三种状态（类常量，全大写是惯例）——
    EMPTY = 0   # 空格
    BLACK = 1   # 黑子（先手）
    WHITE = 2   # 白子（后手）

    def __init__(self, board_size: int = 15, win_length: int = 5):
        self.board_size = board_size    # 棋盘边长（几乘几）
        self.win_length = win_length    # 多少子连成一条线算赢
        self.reset()                    # 一创建对象就初始化好一局

    def reset(self):
        """
        开始新的一局：棋盘清空、轮到黑棋先走。返回初始状态（State）。
        RL 里每开一局（一个 Episode）都要先调用 reset()。
        """
        # board_size × board_size 的二维数组，全 0（= 全是空格）
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = self.BLACK   # 黑棋先手（五子棋惯例：黑先）
        self.done = False                  # 这一局是否已经结束
        self.winner = None                 # 谁赢了；None = 还没结果 或 平局
        return self._get_state()

    def _get_state(self):
        """
        返回当前状态（State）= 棋盘的一份【拷贝】。
        用 .copy() 是怕外面不小心改动了我们内部的棋盘。
        （方法名前的下划线 _ 是 Python 习惯：表示"内部使用，外人别乱碰"）
        """
        return self.board.copy()

    def legal_actions(self):
        """
        返回当前所有合法动作（Action）= 所有空格坐标的列表，
        每个动作是一个 (row, col) 元组。
        """
        # np.argwhere 找出所有 == EMPTY 的位置，返回形如 [[r,c],[r,c],...]
        positions = np.argwhere(self.board == self.EMPTY)
        # 转成普通 Python int（打印干净；以后当字典的键也更省心）
        return [(int(r), int(c)) for r, c in positions]

    def _in_board(self, row: int, col: int) -> bool:
        """坐标 (row, col) 是否还在棋盘范围之内。"""
        return 0 <= row < self.board_size and 0 <= col < self.board_size

    def _check_win(self, row, col, player):
        """
        判断"刚落在 (row, col) 的 player 这颗子"有没有凑成 win_length 连珠。

        🔑 关键思路：只需检查【最后落下的这一颗】！
        因为这步之前还没人赢，所以真要赢，必然是这颗子促成的——
        于是我们不用扫描整个棋盘，只看这颗子周围，又快又简单。

        每颗子有 4 条可能的连珠方向，每条都要朝正、反【两边】各数一数：
            横 —  (0, 1)      竖 |  (1, 0)
            斜 ↘ (1, 1)       斜 ↗ (1, -1)
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1  # 先把刚落的这颗子算上

            # ① 朝 (dr, dc) 方向数：只要还在盘内、且是同色子，就一直往前走
            r, c = row + dr, col + dc
            while self._in_board(r, c) and self.board[r, c] == player:
                count += 1
                r, c = r + dr, c + dc

            # ② 再朝相反方向 (-dr, -dc) 数
            r, c = row - dr, col - dc
            while self._in_board(r, c) and self.board[r, c] == player:
                count += 1
                r, c = r - dr, c - dc

            # 这条线上连续同色子数够了 → 赢
            if count >= self.win_length:
                return True

        return False  # 4 个方向都没连够 → 还没赢

    def step(self, action):
        """
        核心方法：让"当前玩家"在 action=(row, col) 落子，把游戏推进一步。
        这是 RL 交互循环的心脏：
            新状态, 奖励, 是否结束, 附加信息 = env.step(动作)

        关于奖励(reward)的约定 —— 站在【刚落子的这位玩家】的角度：
            · 这一步直接连成 win_length 子获胜  → +1.0
            · 其它情况（棋局继续 / 平局）        →  0.0
        注：'输棋方该拿 -1' 这件事，留到以后训练 Agent 时在外层处理；
            环境本身只如实报告"刚走这一步赢了没"，保持简单。
        """
        row, col = action

        # —— 安全检查：挡住三种非法落子，否则以后很难 debug ——
        if self.done:
            raise ValueError("这一局已经结束，请先 reset() 再下。")
        if not self._in_board(row, col):
            raise ValueError(f"动作 {action} 超出棋盘范围 [0, {self.board_size - 1}]。")
        if self.board[row, col] != self.EMPTY:
            raise ValueError(f"位置 {action} 已经有子了，不能重复落子。")

        # 先记住"现在是谁在走"（下面可能要换手，先存起来）
        player = self.current_player

        # —— 真正落子 ——
        self.board[row, col] = player

        # —— 裁判登场，判定这一步的结果 ——
        reward = 0.0
        if self._check_win(row, col, player):       # ① 这一步直接赢了
            self.done = True
            self.winner = player
            reward = 1.0
        elif len(self.legal_actions()) == 0:        # ② 盘满了还没人赢 → 平局
            self.done = True
            self.winner = None
        else:                                       # ③ 棋局继续 → 轮到对手走
            self.current_player = (
                self.WHITE if player == self.BLACK else self.BLACK
            )

        info = {}   # 预留口袋：以后往里塞点调试信息
        return self._get_state(), reward, self.done, info

    def render(self):
        """把棋盘打印到屏幕，方便人眼观察。  . 空 / X 黑 / O 白 """
        symbols = {self.EMPTY: ".", self.BLACK: "X", self.WHITE: "O"}
        # 第一行：列号表头（每列占 2 字符宽，右对齐）
        print("    " + "".join(f"{c:>2}" for c in range(self.board_size)))
        # 每一行：行号 + 这一行所有格子的符号
        for r in range(self.board_size):
            cells = "".join(f"{symbols[v]:>2}" for v in self.board[r])
            print(f"{r:>2}  {cells}")
        # 末尾：棋局没结束就提示轮到谁；结束了就报结果
        if not self.done:
            who = "X(黑)" if self.current_player == self.BLACK else "O(白)"
            print(f"轮到: {who}\n")
        elif self.winner is None:
            print("棋局结束：平局\n")
        else:
            w = "X(黑)" if self.winner == self.BLACK else "O(白)"
            print(f"棋局结束：{w} 获胜！🎉\n")


# ———————————————————————————————————————————————
# 直接运行本文件（python gomoku_env.py）时，跑下面这段小演示来自测
# ———————————————————————————————————————————————
if __name__ == "__main__":
    # 用最小的 3×3、三连即胜（其实就是井字棋）来演示胜负判断
    env = GomokuEnv(board_size=3, win_length=3)

    print("【演示：X(黑) 在第 0 行连成三子获胜】\n")
    env.render()

    # 交替落子：X 走 (0,0)(0,1)(0,2) 连成第 0 行；O 在中间瞎走
    moves = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)]
    for mv in moves:
        state, reward, done, info = env.step(mv)
        print(f"→ 落子 {mv}    reward={reward}   done={done}")
        env.render()
        if done:                # 分出胜负就停
            break
