import heapq
import time

from base_task import BaseTask


# Lazy optimizer for ub2
# class LazySlicingOptimizer:
#     def __init__(self, model):
#         self.model: BaseTask = model
#         # store {element: (last_calculated_gain)}
#         self.memo = {}
#         self.version_counter = 0
#         # max heap
#         self.h = []
#         self.base = []
#
#     def density(self, ele, base):
#         return self.model.marginal_gain(ele, list(base)) / self.model.cost_of_singleton(ele)
#
#     def build(self, base, remaining):
#         self.base = base
#         for ele in remaining:
#             heapq.heappush(self.h, (-self.density(ele, base), ele))
#
#     def update_base(self, base):
#         self.base = base
#
#     def solve(self, remaining_set, budget):
#         self.version_counter += 1
#         current_base = set(self.base)
#
#         # 1.Initialize the max heap
#         # storage format: (-priority_density, element_id)
#         delta = 0
#         cur_cost = 0
#         processed_elements = []
#         while self.h and cur_cost < budget:
#             # 1. Pop the element with the highest upper bound
#             _, best_ele = heapq.heappop(self.h)
#
#             # 1.5 ignore the element if it has been removed
#             if best_ele not in remaining_set:
#                 continue
#
#             # 2. Calculate its actual density
#             actual_density = self.density(best_ele, self.base)
#             # 3. Compare the actual density with the upper bound of the second-highest element
#             while self.h and self.h[0][1] not in remaining_set:
#                 heapq.heappop(self.h)
#
#             if not self.h or actual_density >= -self.h[0][0]:
#                 # Best ele is actually the best one. Process it.
#                 if actual_density == 0:
#                     break
#
#                 current_density = self.density(best_ele, current_base)
#
#                 s_value = current_density / actual_density
#                 additive_cost = s_value * self.model.cost_of_singleton(best_ele)
#
#                 if cur_cost + additive_cost <= budget:
#                     delta += actual_density * additive_cost
#                     cur_cost += additive_cost
#                     current_base.add(best_ele)
#
#                     # Store to push back later
#                     processed_elements.append((-actual_density, best_ele))
#                 else:
#                     additive_cost = budget - cur_cost
#                     delta += actual_density * additive_cost
#                     processed_elements.append((-actual_density, best_ele))
#                     break
#             else:
#                 heapq.heappush(self.h, (-actual_density, best_ele))
#
#         # 4. Push processed elements back to heap
#         for ele in processed_elements:
#             heapq.heappush(self.h, ele)
#
#         return delta
class LazySlicingOptimizer:
    def __init__(self, model):
        self.model = model
        self.h = []
        self.base = []

    def build(self, base, remaining):
        self.base = list(base)
        for ele in remaining:
            cost = self.model.cost_of_singleton(ele)
            gain = self.model.marginal_gain(ele, self.base)
            density = gain / cost if cost > 0 else 0
            heapq.heappush(self.h, (-density, ele))

    def update_base(self, base):
        self.base = list(base)

    def solve(self, remaining_set, budget):
        # 使用 list 避免 set 到 list 的 O(N) 转换开销
        current_base = list(self.base)

        delta = 0
        cur_cost = 0
        processed_elements = []

        while self.h and cur_cost < budget:
            _, best_ele = heapq.heappop(self.h)

            if best_ele not in remaining_set:
                continue

            # 提前拉取成本，避免重复计算
            cost_best = self.model.cost_of_singleton(best_ele)
            actual_gain = self.model.marginal_gain(best_ele, self.base)
            actual_density = actual_gain / cost_best if cost_best > 0 else 0

            while self.h and self.h[0][1] not in remaining_set:
                heapq.heappop(self.h)

            if not self.h or actual_density >= -self.h[0][0]:
                if actual_density == 0:
                    break

                current_gain = self.model.marginal_gain(best_ele, current_base)

                # S-value 数学展开: (current_gain / actual_gain)
                additive_cost = (current_gain / actual_gain) * cost_best if actual_gain != 0 else 0

                if cur_cost + additive_cost <= budget:
                    # 数学等价代换：彻底省略实际密度与附加成本的乘法运算
                    delta += current_gain
                    cur_cost += additive_cost
                    current_base.append(best_ele)  # list 的 append 是 O(1)
                    processed_elements.append((-actual_density, best_ele))
                else:
                    additive_cost = budget - cur_cost
                    delta += actual_density * additive_cost
                    processed_elements.append((-actual_density, best_ele))
                    break
            else:
                heapq.heappush(self.h, (-actual_density, best_ele))

        for ele in processed_elements:
            heapq.heappush(self.h, ele)

        return delta

# Lazy optimizer for ub0
class LazyPlainOptimizer:
    def __init__(self, model):
        self.model: BaseTask = model
        # store {element: (last_calculated_gain)}
        self.memo = {}
        self.version_counter = 0
        # max heap
        self.h = []
        self.base = []

    def density(self, ele, base):
        return self.model.marginal_gain(ele, list(base)) / self.model.cost_of_singleton(ele)

    def build(self, base, remaining):
        self.base = base
        for ele in remaining:
            heapq.heappush(self.h, (-self.density(ele, base), ele))

    def update_base(self, base):
        self.base = base

    def solve(self, remaining_set, budget):
        self.version_counter += 1
        current_base = set(self.base)

        # 1.Initialize the max heap
        # storage format: (-priority_density, element_id)
        delta = 0
        cur_cost = 0
        processed_elements = []
        while self.h and cur_cost < budget:
            # 1. Pop the element with the highest upper bound
            _, best_ele = heapq.heappop(self.h)

            # 1.5 ignore the element if it has been removed
            if best_ele not in remaining_set:
                continue

            # 2. Calculate its actual density
            actual_density = self.density(best_ele, self.base)
            # 3. Compare the actual density with the upper bound of the second-highest element
            while self.h and self.h[0][1] not in remaining_set:
                heapq.heappop(self.h)

            if not self.h or actual_density >= -self.h[0][0]:
                # Best ele is actually the best one. Process it.
                if actual_density == 0:
                    break

                additive_cost = self.model.cost_of_singleton(best_ele)
                if cur_cost + additive_cost <= budget:
                    delta += actual_density * additive_cost
                    cur_cost += additive_cost
                    current_base.add(best_ele)

                    # Store to push back later
                    processed_elements.append((-actual_density, best_ele))
                else:
                    additive_cost = budget - cur_cost
                    delta += actual_density * additive_cost

                    processed_elements.append((-actual_density, best_ele))
                    break
            else:
                heapq.heappush(self.h, (-actual_density, best_ele))

        # 4. Push processed elements back to heap
        for ele in processed_elements:
            heapq.heappush(self.h, ele)

        return delta
