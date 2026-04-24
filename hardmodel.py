import random
import numpy as np


class HardSetCoverTask:
    def __init__(self, ground_size=200, budget=100, seed=42):
        self.ground_set = list(range(ground_size))
        self.budget = budget

        # 内部状态
        self.costs = {}
        self.coverage = {}

        # 特征全集 (Universe)
        num_features = 5000

        random.seed(seed)
        np.random.seed(seed)

        # ==========================================
        # 陷阱组 1：贪心诱饵 (Greedy Baits)
        # 数量多，体积小，初始密度极高，但互相重叠极严重
        # ==========================================
        for i in range(0, ground_size - 10):
            # 基础成本随机，但偏小
            base_cost = random.randint(3, 8)
            self.costs[i] = base_cost

            # 覆盖点数严格正比于成本，强行制造 10.0 的虚假高密度
            num_points = int(base_cost * 10.0)

            # 诱饵元素大量覆盖 0~1000 这个“公共拥挤区”
            public_points = random.sample(range(0, 1000), num_points - 2)
            # 加一点独有特征防止密度衰减过快
            private_points = random.sample(range(1000, num_features), 2)

            self.coverage[i] = set(public_points) | set(private_points)

        # ==========================================
        # 陷阱组 2：真正的全局最优 (The Hidden Optima)
        # 体积大，刚好能填满 Budget，绝对零重叠
        # 但初始密度故意设为 9.8，导致第一轮贪心绝对不会选它们
        # ==========================================
        opt_cost = budget // 5  # 需要 5 个完美拼图填满背包
        for i in range(ground_size - 10, ground_size):
            self.costs[i] = opt_cost
            num_points = int(opt_cost * 9.8)  # 密度 9.8，低于诱饵的 10.0

            # 给它们分配绝对私有的特征区间，保证边际收益永不衰减！
            start_idx = 1000 + i * 200
            self.coverage[i] = set(range(start_idx, start_idx + num_points))

    # --- 以下方法无缝对接你的 EfficientBFS 接口 ---

    def cost_of_singleton(self, e):
        return self.costs[e]

    def cost_of_set(self, s):
        return sum(self.costs[e] for e in s)

    def objective(self, s):
        if not s:
            return 0.0
        # 子模函数：集合覆盖的并集大小
        covered = set()
        for e in s:
            covered.update(self.coverage[e])
        return float(len(covered))

    def marginal_gain(self, e, s):
        if not s:
            return float(len(self.coverage[e]))

        covered_by_s = set()
        for existing_e in s:
            covered_by_s.update(self.coverage[existing_e])

        # 增量收益 = 新元素覆盖点 中 减去 已经被覆盖的点
        new_points = self.coverage[e] - covered_by_s
        return float(len(new_points))