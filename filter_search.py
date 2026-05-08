import copy
import heapq
import multiprocessing
import time
from functools import total_ordering

import acclerated_upper_bounds
from OptimalAlg import OptimalAlg
from base_task import BaseTask
from MaxHeap import MaxHeap, HeapObj, EfficientBFSHeapObj, BranchAndBoundNode, SimpleMaxHeap, BaseHeapObj
from data_dependent_upperbound import marginal_delta_version7, marginal_delta, marginal_delta_m, marginal_delta_m_acc, \
    marginal_delta_random_budget, marginal_delta_version7_random_budget, marginal_delta_m_acc_random_budget, \
    marginal_delta_dom_random_budget
from optimizer import DominantOptimizer
import math
import random

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# Tree-mode logging helpers (canonical set formatting, etc.).
import runlog as runlog_mod



@total_ordering
class AugmentedValue:
    def __init__(self, av, v):
        self.av = av
        self.v = v

    def __eq__(self, other):
        return self.av == other.av and self.v == other.v

    def __lt__(self, other):
        if self.av < other.av:
            return True
        if self.av == other.av and self.v < other.v:
            return True
        return False


@total_ordering
class AugmentedFSValue:
    def __init__(self, inner_v, lbd_v, sort_v):
        self.inner_v = inner_v
        self.lbd_v = lbd_v
        self.sort_v = sort_v

    def __eq__(self, other):
        return self.lbd_v == other.lbd_v and self.sort_v == other.sort_v

    def __lt__(self, other):
        if self.lbd_v < other.lbd_v:
            return True
        if self.lbd_v == other.lbd_v and self.sort_v < other.sort_v:
            return True
        return False


@total_ordering
class RefinedBFSValue:
    def __init__(self, inner_v, lbd_v, sort_v):
        self.inner_v = inner_v
        self.lbd_v = lbd_v
        self.sort_v = sort_v

    def __eq__(self, other):
        return self.lbd_v == other.lbd_v and self.sort_v == other.sort_v

    def __lt__(self, other):
        if self.lbd_v < other.lbd_v:
            return True
        if self.lbd_v == other.lbd_v and self.sort_v < other.sort_v:
            return True
        return False

class FS(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)

        self.closed_list = []
        self.heap = MaxHeap()
        self.f = None
        self.h = None

        self.lbd = -1
        self.augmentation = False

        self.E = set()

    def build(self):
        self.closed_list.clear()
        self.heap.clear()
        self.lbd = -1
        self.augmentation = False

        self.f = self.model.objective
        if self.opt == 'ub0':
            self.h = self.h_ub0
        elif self.opt == 'ub1':
            self.h = self.h_ub1
        elif self.opt == 'ub2':
            self.h = self.h_ub2
        elif self.opt == 'ub0+':
            self.h = self.h_ub0
            self.augmentation = True
        elif self.opt == 'ub1+':
            self.h = self.h_ub1
            self.augmentation = True
        elif self.opt == 'ub2+':
            self.h = self.h_ub2
            self.augmentation = True

    # the heuristic function
    def h_ub0(self, S):
        delta, _ = marginal_delta_random_budget(set(S), set(self.model.ground_set) - set(S), self.model,
                                                budget=self.model.budget - self.model.cost_of_set(S))
        return delta

    # the heuristic function
    def h_ub2(self, S):
        delta, _ = marginal_delta_version7_random_budget(set(S), set(self.model.ground_set) - set(S), self.model,
                                                         budget=self.model.budget - self.model.cost_of_set(S))
        return delta

    # the heuristic function
    def h_ub1(self, S):
        # delta, _ = marginal_delta_m_acc(set(S), set(self.model.ground_set) - set(S), self.model)
        delta, _ = marginal_delta_m_acc_random_budget(set(S), set(self.model.ground_set) - set(S), self.model,
                                                      budget=self.model.budget - self.model.cost_of_set(S))
        return delta

    def is_on_the_edge(self, S):
        base_cost = self.model.cost_of_set(S)
        for ele in set(self.model.ground_set) - S:
            if base_cost + self.model.cost_of_singleton(ele) <= self.model.budget:
                return False
        return True

    def g(self, S):
        if self.is_on_the_edge(S):
            return self.f(list(S)), self.f(list(S))

        current_h = self.f(list(S)) + self.alpha * self.h(S)
        normal_h = current_h

        if self.augmentation:
            if self.lbd < 0 or current_h < self.lbd:
                self.lbd = current_h
            else:
                current_h = self.lbd

        return current_h, normal_h

    def push_heap(self, S):
        aug_v, v = self.g(S)
        t = AugmentedValue(aug_v, v)
        self.heap.push(HeapObj(S, t))

    def optimize(self):
        start_time = time.time()

        ret = {
        }

        s = None

        node_count = 0
        explored_node_count = 0

        self.push_heap(set())
        while self.heap.size() > 0:
            obj = self.heap.pop()
            s, v = obj.s, obj.v

            # print(f"s:{s}, v:{v}, lbd:{self.lbd}")
            node_count += 1
            if self.is_on_the_edge(s):
                stop_time = time.time()
                ret['S'] = s
                ret['c(S)'] = self.model.cost_of_set(s)
                ret['f(S)'] = self.model.objective(s)
                ret['time'] = stop_time - start_time
                ret['node_count'] = node_count
                ret['explored_node_count'] = explored_node_count
                return ret

            if s not in self.closed_list:
                self.closed_list.append(s)

                for ele in set(self.model.ground_set) - s:
                    s_plus = s | {ele}
                    if self.model.cost_of_set(s_plus) <= self.model.budget:
                        explored_node_count += 1
                        self.push_heap(s_plus)

        stop_time = time.time()
        ret['S'] = s
        ret['c(S)'] = self.model.cost_of_set(s)
        ret['f(S)'] = self.model.objective(s)
        ret['time'] = stop_time - start_time
        ret['node_count'] = node_count
        ret['explored_node_count'] = explored_node_count
        print(f"return from fallback")

        return ret


# class AugmentedFS(OptimalAlg):
#     def __init__(self, model: BaseTask):
#         super().__init__(model)
#
#         self.closed_list = []
#         self.heap = MaxHeap()
#
#         self.g = None
#         self.inner_h = None
#
#         self.lbd = -1
#         self.augmentation = False
#
#         self.E = set()
#
#     def build(self):
#         self.closed_list.clear()
#         self.heap.clear()
#         self.lbd = -1
#
#         self.g = self.model.objective
#
#         if self.opt == 'ub0':
#             self.inner_h = self.h_ub0
#         elif self.opt == 'ub1':
#             self.inner_h = self.h_ub1
#         elif self.opt == 'ub2':
#             self.inner_h = self.h_ub2
#
#     # the heuristic function
#     def h_ub0(self, S):
#         delta, _ = marginal_delta_random_budget(set(S), set(self.model.ground_set) - set(S), self.model, budget=self.model.budget - self.model.cost_of_set(S))
#         return delta
#
#     # the heuristic function
#     def h_ub2(self, S):
#         delta, _ = marginal_delta_version7_random_budget(set(S), set(self.model.ground_set) - set(S), self.model, budget=self.model.budget - self.model.cost_of_set(S))
#         return delta
#
#     # the heuristic function
#     def h_ub1(self, S):
#         # delta, _ = marginal_delta_m_acc(set(S), set(self.model.ground_set) - set(S), self.model)
#         delta, _ = marginal_delta_m_acc_random_budget(set(S), set(self.model.ground_set) - set(S), self.model, budget=self.model.budget - self.model.cost_of_set(S))
#         return delta
#
#     def is_on_the_edge(self, S):
#         base_cost = self.model.cost_of_set(S)
#         for ele in set(self.model.ground_set) - S:
#             if base_cost + self.model.cost_of_singleton(ele) <= self.model.budget:
#                 return False
#         return True
#
#     def h(self, S):
#         if self.is_on_the_edge(S):
#             return 0
#         return self.inner_h(S)
#
#     def f(self, node: HeapObj):
#         return min(self.g(node.s) + self.alpha * self.h(node.s), node.lbd)
#
#     def push_heap(self, S, parent_lbd):
#         current_lbd = min(self.g(S) + self.alpha * self.h(S), parent_lbd)
#
#         t = AugmentedValue(current_lbd, self.g(S))
#
#         self.heap.push(HeapObj(S, t, current_lbd))
#
#     def optimize(self):
#         start_time = time.time()
#
#         ret = {
#         }
#
#         s = None
#
#         node_count = 0
#
#         self.push_heap(set(), self.alpha * self.h(set()))
#
#         while self.heap.size() > 0:
#             obj = self.heap.pop()
#             s, v, parent_lbd = obj.s, obj.v, obj.lbd
#
#             # print(f"s:{s}, v:{v}, lbd:{self.lbd}")
#
#             node_count += 1
#
#             if self.is_on_the_edge(s):
#                 stop_time = time.time()
#                 ret['S'] = s
#                 ret['c(S)'] = self.model.cost_of_set(s)
#                 ret['f(S)'] = self.model.objective(s)
#                 ret['time'] = stop_time - start_time
#                 ret['node_count'] = node_count
#                 return ret
#
#             if s not in self.closed_list:
#                 self.closed_list.append(s)
#
#                 for ele in set(self.model.ground_set) - s:
#                     s_plus = s | {ele}
#                     if self.model.cost_of_set(s_plus) <= self.model.budget:
#                         self.push_heap(s_plus, parent_lbd)
#
#         stop_time = time.time()
#         ret['S'] = s
#         ret['c(S)'] = self.model.cost_of_set(s)
#         ret['f(S)'] = self.model.objective(s)
#         ret['time'] = stop_time - start_time
#         ret['node_count'] = node_count
#         print(f"return from fallback")
#
#         return ret

# focal filter search
# class FFS(OptimalAlg):
#     def __init__(self, model: BaseTask):
#         super().__init__(model)
#
#         self.closed_list = []
#         self.heap = MaxHeap()
#         self.focal = []
#         self.f = None
#         self.h = None
#
#         self.lbd = -1
#         self.augmentation = False
#
#         self.E = set()
#
#     def build(self):
#         self.closed_list.clear()
#         self.heap.clear()
#         self.focal.clear()
#         self.lbd = -1
#         self.augmentation = False
#
#         self.f = self.model.objective
#         if self.opt == 'ub0':
#             self.h = self.h_ub0
#         elif self.opt == 'ub1':
#             self.h = self.h_ub1
#         elif self.opt == 'ub2':
#             self.h = self.h_ub2
#         elif self.opt == 'ub0+':
#             self.h = self.h_ub0
#             self.augmentation = True
#         elif self.opt == 'ub1+':
#             self.h = self.h_ub1
#             self.augmentation = True
#         elif self.opt == 'ub2+':
#             self.h = self.h_ub2
#             self.augmentation = True
#
#     # the heuristic function
#     def h_ub0(self, S):
#         delta, _ = marginal_delta_random_budget(set(S), set(self.model.ground_set) - set(S), self.model, budget=self.model.budget - self.model.cost_of_set(S))
#         return delta
#
#     # the heuristic function
#     def h_ub2(self, S):
#         delta, _ = marginal_delta_version7_random_budget(set(S), set(self.model.ground_set) - set(S), self.model, budget=self.model.budget - self.model.cost_of_set(S))
#         return delta
#
#     # the heuristic function
#     def h_ub1(self, S):
#         # delta, _ = marginal_delta_m_acc(set(S), set(self.model.ground_set) - set(S), self.model)
#         delta, _ = marginal_delta_m_acc_random_budget(set(S), set(self.model.ground_set) - set(S), self.model, budget=self.model.budget - self.model.cost_of_set(S))
#         return delta
#
#     def is_on_the_edge(self, S):
#         base_cost = self.model.cost_of_set(S)
#         for ele in set(self.model.ground_set) - S:
#             if base_cost + self.model.cost_of_singleton(ele) <= self.model.budget:
#                 return False
#         return True
#
#     def g(self, S):
#         if self.is_on_the_edge(S):
#             return self.f(list(S)), self.f(list(S))
#
#         current_h = self.f(list(S)) + self.alpha * self.h(S)
#         normal_h = current_h
#
#         if self.augmentation:
#             if self.lbd < 0 or current_h < self.lbd:
#                 self.lbd = current_h
#             else:
#                 current_h = self.lbd
#
#         return current_h, normal_h
#
#     def push_heap(self, S):
#         aug_v, v = self.g(S)
#         t = AugmentedValue(aug_v, v)
#         self.heap.push(HeapObj(S, t))
#
#     def wrap(self, s):
#         aug_v, v = self.g(s)
#         t = AugmentedValue(aug_v, v)
#         obj = HeapObj(s, t)
#         return obj
#
#     def max_focal(self):
#         max_obj = None
#         for obj in self.focal:
#             if max_obj is None or max_obj < obj:
#                 max_obj = obj
#         return max_obj
#
#     def update_upper_bound(self, old_bound, new_bound):
#         for obj in self.heap.h:
#             if old_bound > obj.v.v >= new_bound and obj not in self.focal:
#                 self.focal.append(obj)
#
#     def optimize(self):
#         start_time = time.time()
#
#         ret = {
#         }
#
#         s = None
#
#         node_count = 0
#         explored_node_count = 0
#
#         obj = self.wrap(set())
#         self.heap.push(obj)
#         self.focal.append(obj)
#         f_max = 0
#
#         while self.heap.size() > 0:
#             obj = self.max_focal()
#             f_max = self.heap.top().v
#
#             self.focal.remove(obj)
#             self.heap.remove(obj)
#
#             s, v = obj.s, obj.v
#             # print(f"s:{s}, v:{v}, lbd:{self.lbd}")
#             node_count += 1
#             if self.is_on_the_edge(s):
#                 stop_time = time.time()
#                 ret['S'] = s
#                 ret['c(S)'] = self.model.cost_of_set(s)
#                 ret['f(S)'] = self.model.objective(s)
#                 ret['time'] = stop_time - start_time
#                 ret['node_count'] = node_count
#                 ret['explored_node_count'] = explored_node_count
#                 return ret
#
#             if s not in self.closed_list:
#                 self.closed_list.append(s)
#
#                 for ele in set(self.model.ground_set) - s:
#                     s_plus = s | {ele}
#                     if self.model.cost_of_set(s_plus) <= self.model.budget:
#                         explored_node_count += 1
#                         obj = self.wrap(s_plus)
#                         self.heap.push(obj)
#                         if obj.v.v >= self.alpha * f_max.v:
#                             self.focal.append(obj)
#
#             if self.heap.size() > 0 and f_max > self.heap.top().v:
#                 self.update_upper_bound(self.alpha * f_max.v, self.alpha * self.heap.top().v.v)
#
#
#         stop_time = time.time()
#         ret['S'] = s
#         ret['c(S)'] = self.model.cost_of_set(s)
#         ret['f(S)'] = self.model.objective(s)
#         ret['time'] = stop_time - start_time
#         ret['node_count'] = node_count
#         ret['explored_node_count'] = explored_node_count
#         print(f"return from fallback")
#
#         return ret
#

class AugmentedFS(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)
        self.max_heap = MaxHeap()
        self.inner_h = None
        self.d = None

    def build(self):
        self.max_heap.clear()

    def set_h(self, heuristic):
        if heuristic == 'ub0':
            self.inner_h = self.h_ub0
        elif heuristic == 'ub2':
            self.inner_h = self.h_ub2
        elif heuristic == 'ub4':
            self.inner_h = self.h_ub4

    def density_for_set(self, n):
        if self.model.cost_of_set(list(n)) == 0:
            return 0
        return self.g(n) / self.model.cost_of_set(list(n))

    def set_d(self, sorting):
        if sorting == 'g':
            self.d = self.g
        elif sorting == 'b':
            self.d = lambda x: self.model.cost_of_set(list(x))
        elif sorting == 'd':
            self.d = self.density_for_set

    def g(self, n):
        return self.model.objective(list(n))

    def is_on_the_edge(self, n):
        base_cost = self.model.cost_of_set(n)
        for ele in set(self.model.ground_set) - set(n):
            if base_cost + self.model.cost_of_singleton(ele) <= self.model.budget:
                return False
        return True

    def h(self, n):
        if self.is_on_the_edge(n):
            return 0
        return self.inner_h(n)

    def f(self, n):
        return self.g(n) + self.alpha * self.h(n)

    def h_ub0(self, n):
        delta, _ = marginal_delta_random_budget(set(n), set(self.model.ground_set) - set(n), self.model,
                                                budget=self.model.budget - self.model.cost_of_set(n))
        return delta

    def h_ub2(self, n):
        delta, _ = marginal_delta_version7_random_budget(set(n), set(self.model.ground_set) - set(n), self.model,
                                                         budget=self.model.budget - self.model.cost_of_set(n))
        return delta

    def h_ub4(self, n):
        opt = DominantOptimizer()
        opt.setModel(self.model)
        opt.setBase(n)
        opt.build()
        delta = opt.optimize()['delta']

        return delta

    def push_heap(self, n, inherited_value):
        max_value = 0
        if len(n) > 0:
            max_value = max(n)

        v = AugmentedFSValue(self.f(n), inherited_value, self.d(n), min(self.f(n), inherited_value), max_value)
        node = HeapObj(n, v)
        self.max_heap.push(node)

    def optimize(self):
        start_time = time.time()
        root = []
        self.push_heap(root, self.f(root))

        sol = None
        node_count = 0
        while self.max_heap.size() > 0:
            node = self.max_heap.pop()
            node_count += 1
            s = node.s
            v = node.v
            max_idx = v.max_idx
            if self.h(s) == 0:
                sol = s
                break

            for i in set(self.model.ground_set) - set(s):
                if i > max_idx and self.model.cost_of_set(s) + self.model.cost_of_singleton(i) <= self.model.budget:
                    inherited_value = min(self.f(list(set(s) | {i})), v.inherited_v)
                    self.push_heap(list(set(s) | {i}), inherited_value)

        stop_time = time.time()

        assert sol is not None, "No solution found."

        ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
               'time': stop_time - start_time, 'node_count': node_count}

        return ret


class BestAugmentedFS(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)
        self.max_heap = MaxHeap()
        self.inner_h = None
        self.f = None
        self.d = None
        self.use_alpha = False

    def build(self):
        self.max_heap.clear()
        if self.use_alpha:
            self.f = self.f_with_alpha
        else:
            self.f = self.f_without_alpha

    def set_h(self, heuristic):
        if heuristic == 'ub0':
            self.inner_h = self.h_ub0
        elif heuristic == 'ub2':
            self.inner_h = self.h_ub2
        elif heuristic == 'ub4':
            self.inner_h = self.h_ub4

    def density_for_set(self, n):
        if self.model.cost_of_set(list(n)) == 0:
            return 0
        return self.g(n) / self.model.cost_of_set(list(n))

    def set_d(self, sorting):
        if sorting == 'g':
            self.d = self.g
        elif sorting == 'b':
            self.d = lambda x: self.model.cost_of_set(list(x))
        elif sorting == 'd':
            self.d = self.density_for_set

    def g(self, n):
        return self.model.objective(list(n))

    def is_on_the_edge(self, n):
        base_cost = self.model.cost_of_set(n)
        for ele in set(self.model.ground_set) - set(n):
            if base_cost + self.model.cost_of_singleton(ele) <= self.model.budget:
                return False
        return True

    def h(self, n):
        if self.is_on_the_edge(n):
            return 0
        return self.inner_h(n)

    def f_with_alpha(self, n):
        return self.g(n) + self.alpha * self.h(n)

    def f_without_alpha(self, n):
        return self.g(n) + self.h(n)

    def h_ub0(self, n):
        delta, _ = marginal_delta_random_budget(set(n), set(self.model.ground_set) - set(n), self.model,
                                                budget=self.model.budget - self.model.cost_of_set(n))
        return delta

    def h_ub2(self, n):
        delta, _ = marginal_delta_version7_random_budget(set(n), set(self.model.ground_set) - set(n), self.model,
                                                         budget=self.model.budget - self.model.cost_of_set(n))
        return delta

    def h_ub4(self, n):
        opt = DominantOptimizer()
        opt.setModel(self.model)
        opt.setBase(n)
        opt.build()
        delta = opt.optimize()['delta']

        return delta

    def push_heap(self, n, inherited_value):
        max_value = 0
        if len(n) > 0:
            max_value = max(n)

        v = AugmentedFSValue(self.f(n), inherited_value, self.d(n), min(self.f(n), inherited_value), max_value)
        node = HeapObj(n, v)
        self.max_heap.push(node)

    def greedy_add(self, base):
        sol = set(base)
        remaining_elements = set(self.model.ground_set) - set(base)
        cur_cost = self.model.cost_of_set(list(sol))

        while len(remaining_elements):
            u, max_density = None, -1.
            for e in remaining_elements:
                # e is an object
                ds = self.model.density(e, list(sol))
                if u is None or ds > max_density:
                    u, max_density = e, ds
            assert u is not None
            if cur_cost + self.model.cost_of_singleton(u) <= self.model.budget:
                # satisfy the knapsack constraint
                sol.add(u)
                cur_cost += self.model.cost_of_singleton(u)

            remaining_elements.remove(u)
            # filter out violating elements
            to_remove = set()
            for v in remaining_elements:
                if self.model.cost_of_singleton(v) + cur_cost > self.model.budget:
                    to_remove.add(v)
            remaining_elements -= to_remove

        # find the maximum singleton
        v_star, v_star_fv = None, float('-inf')
        for e in set(self.model.ground_set) - set(base):
            if self.model.cost_of_singleton(e) > self.model.budget - self.model.cost_of_set(list(base)):
                # filter out singleton whose cost is larger than budget
                continue
            fv = self.model.objective(list(set(base) | {e}))
            if fv > v_star_fv:
                v_star, v_star_fv = e, fv

        sol_fv = self.model.objective(list(sol))

        if v_star_fv > sol_fv:
            return list(set(base) | {v_star})
        else:
            return list(sol)

    def optimize(self):
        start_time = time.time()
        root = []
        self.push_heap(root, self.f(root))

        g_upper = self.f(root)
        s_max = self.greedy_add(root)
        s_max_v = self.g(s_max)
        sol = s_max

        node_count = 0
        open_list_count = 1

        while self.max_heap.size() > 0:
            node = self.max_heap.pop()
            node_count += 1
            s = node.s
            v = node.v
            max_idx = v.max_idx

            g_upper = min(g_upper, v.outlook_v)

            s_final = self.greedy_add(s)
            if self.g(s_final) > self.g(s_max):
                s_max = s_final
                s_max_v = self.g(s_max)

            if self.use_alpha:
                if self.g(s_max) >= g_upper:
                    sol = s_max
                    break
            else:
                if self.g(s_max) >= self.alpha * g_upper:
                    sol = s_max
                    break

            for i in set(self.model.ground_set) - set(s):
                if i > max_idx and self.model.cost_of_set(s) + self.model.cost_of_singleton(i) <= self.model.budget:
                    final_v = self.f_without_alpha(list(set(s) | {i}))
                    if final_v >= s_max_v:
                        inherited_value = min(self.f(list(set(s) | {i})), v.inherited_v)
                        self.push_heap(list(set(s) | {i}), inherited_value)
                        open_list_count += 1

        stop_time = time.time()

        assert sol is not None, "No solution found."

        ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
               'time': stop_time - start_time, 'node_count': node_count, "open_list_count": open_list_count}

        return ret


class BestAugmentedMoreFS(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)
        self.max_heap = MaxHeap()
        self.inner_h = None
        self.f = None
        self.d = None
        self.lbd = None
        self.use_alpha = False
        self.pushing_back = True
        self.ground_size = 0

    def build(self):
        self.max_heap.clear()
        self.ground_size = len(self.model.ground_set)

        if self.use_alpha:
            self.f = self.f_with_alpha
        else:
            self.f = self.f_without_alpha

    # def set_h(self, heuristic):
    #     if heuristic == 'ub0':
    #         self.inner_h = self.h_ub0
    #         self.lbd = self.lbd0
    #     elif heuristic == 'ub2':
    #         self.inner_h = self.h_ub2
    #         self.lbd = self.lbd2
    #     elif heuristic == 'dom':
    #         self.inner_h = self.h_dom
    #         self.lbd = self.lbd_dom
    #
    # def density_for_set(self, n):
    #     if self.model.cost_of_set(list(n)) == 0:
    #         return 0
    #     return self.g(n) / self.model.cost_of_set(list(n))
    #
    # def set_d(self, sorting):
    #     if sorting == 'g':
    #         self.d = self.g
    #     elif sorting == 'b':
    #         self.d = lambda x: self.model.cost_of_set(list(x))
    #     elif sorting == 'd':
    #         self.d = self.density_for_set
    #
    # def g(self, n):
    #     return self.model.objective(list(n))
    #
    # def is_on_the_edge(self, n):
    #     base_cost = self.model.cost_of_set(n)
    #     for ele in set(self.model.ground_set) - set(n):
    #         if base_cost + self.model.cost_of_singleton(ele) <= self.model.budget:
    #             return False
    #     return True
    #
    # def h(self, n):
    #     if self.is_on_the_edge(n):
    #         return 0
    #     return self.inner_h(n)
    #
    # def f_with_alpha(self, n):
    #     return self.g(n) + self.alpha * self.h(n)
    #
    # def f_without_alpha(self, n):
    #     return self.g(n) + self.h(n)
    #
    # def h_ub0(self, n):
    #     delta, _ = marginal_delta_random_budget(set(n), set(self.model.ground_set) - set(n), self.model,
    #                                             budget=self.model.budget - self.model.cost_of_set(n))
    #     return delta
    #
    # def h_ub2(self, n):
    #     delta, _ = marginal_delta_version7_random_budget(set(n), set(self.model.ground_set) - set(n), self.model,
    #                                                      budget=self.model.budget - self.model.cost_of_set(n))
    #     return delta
    #
    # def h_dom(self, n):
    #     delta, _ = marginal_delta_dom_random_budget(set(n), set(self.model.ground_set) - set(n), self.model,
    #                                                 budget=self.model.budget - self.model.cost_of_set(n))
    #     return delta
    #
    # def lbd0(self, base, budget):
    #     delta, _ = marginal_delta_random_budget(set(base), set(self.model.ground_set) - set(base), self.model,
    #                                             budget=budget)
    #     return delta
    #
    # def lbd2(self, base, budget):
    #     delta, _ = marginal_delta_version7_random_budget(set(base), set(self.model.ground_set) - set(base), self.model,
    #                                                      budget=budget)
    #     return delta
    #
    # def lbd_dom(self, base, budget):
    #     delta, _ = marginal_delta_dom_random_budget(set(base), set(self.model.ground_set) - set(base), self.model,
    #                                                 budget=budget)
    #     return delta
    #
    # def h_ub4(self, n):
    #     opt = DominantOptimizer()
    #     opt.setModel(self.model)
    #     opt.setBase(n)
    #     opt.build()
    #     delta = opt.optimize()['delta']
    #
    #     return delta

    def push_heap(self, s, lbd_v, visited=False, candidate=None, w=None, s_max_v=0):
        max_idx = 0
        if len(s) > 0:
            max_idx = max(s)

        node = HeapObj(s, candidate=candidate, w=w, max_idx=max_idx, visited=visited)
        new_g = self.g(node)
        new_h = self.h(node)
        final_v = new_g + new_h

        v = None
        if final_v >= s_max_v:
            if self.use_alpha:
                lbd_v = min(new_g + self.alpha * new_h, lbd_v)
                v = AugmentedFSValue(new_g + self.alpha * new_h, lbd_v, self.d(s))
            else:
                lbd_v = min(new_g + new_h, lbd_v)
                v = AugmentedFSValue(new_g + new_h, lbd_v, self.d(s))

            node.v = v
            self.max_heap.push(node)

            return node

        return None

    def push_root(self):
        root = HeapObj([], candidate=self.model.ground_set, w=self.model.budget, visited=True, max_idx=0)
        v = AugmentedFSValue(self.f(root), self.f(root), self.d(root.s))
        root.v = v
        root.cost = 0

        f_upper = self.f(root)
        s_max, f_local = self.greedy_add(root.s)
        f_upper = min(f_upper, f_local)

        self.max_heap.push(root)

        return root, f_upper, s_max

    def greedy_add(self, base):
        sol = set(base)
        base_cost = self.model.cost_of_set(list(sol))
        remaining_elements = set(self.model.ground_set) - set(base)
        cur_cost = self.model.cost_of_set(list(sol))

        f_local = None
        while len(remaining_elements):
            u, max_density = None, -1.
            for e in remaining_elements:
                # e is an object
                ds = self.model.density(e, list(sol))
                if u is None or ds > max_density:
                    u, max_density = e, ds
            assert u is not None
            if cur_cost + self.model.cost_of_singleton(u) <= self.model.budget:
                # satisfy the knapsack constraint
                sol.add(u)
                cur_cost += self.model.cost_of_singleton(u)

            f_temp = self.g(sol) + self.lbd(base=sol, candidate=set(self.model.ground_set) - set(sol),
                                            budget=self.model.budget - base_cost)
            if f_local is None or f_temp < f_local:
                f_local = f_temp

            remaining_elements.remove(u)
            # filter out violating elements
            to_remove = set()
            for v in remaining_elements:
                if self.model.cost_of_singleton(v) + cur_cost > self.model.budget:
                    to_remove.add(v)
            remaining_elements -= to_remove

        # find the maximum singleton
        v_star, v_star_fv = None, float('-inf')
        for e in set(self.model.ground_set) - set(base):
            if self.model.cost_of_singleton(e) > self.model.budget - self.model.cost_of_set(list(base)):
                # filter out singleton whose cost is larger than budget
                continue
            fv = self.model.objective(list(set(base) | {e}))
            if fv > v_star_fv:
                v_star, v_star_fv = e, fv

        sol_fv = self.model.objective(list(sol))

        if v_star_fv > sol_fv:
            return list(set(base) | {v_star}), f_local
        else:
            return list(sol), f_local

    def optimize(self):
        start_time = time.time()
        # set root node as visited
        root, f_upper, s_max = self.push_root()

        # check if s_max now is an optimal solution
        if self.g(s_max) >= self.alpha * f_upper:
            # print(f"here, g:{self.g(s_max)}, f:{f_upper}")
            sol = s_max
            stop_time = time.time()

            ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
                   'time': stop_time - start_time, 'node_count': 1, "open_list_count": 1,
                   "push_back_count": 0}

            return ret

        push_back_count = 0
        s_max_v = self.g(s_max)
        sol = s_max

        node_count = 0
        open_list_count = 1

        time_for_stage_0 = 0
        time_for_stage_1 = 0
        time_for_stage_2 = 0
        time_for_stage_3 = 0
        children_count = 0

        while self.max_heap.size() > 0:
            t0 = time.time()

            node = self.max_heap.pop()
            node_count += 1
            s = node.s
            v = node.v
            max_idx = node.max_idx
            f_upper = min(f_upper, v.lbd_v)

            t1 = time.time()

            time_for_stage_0 += t1 - t0

            if not node.visited:
                s_final, f_local = self.greedy_add(s)
                if self.g(s_final) > self.g(s_max):
                    s_max = s_final
                    s_max_v = self.g(s_max)

            t2 = time.time()

            time_for_stage_1 += t2 - t1

            if self.use_alpha:
                if self.g(s_max) >= f_upper:
                    sol = s_max
                    break
            else:
                if self.g(s_max) >= self.alpha * f_upper:
                    # print(f"here, g:{self.g(s_max)}, f:{f_upper}")
                    sol = s_max
                    break

            if not node.visited and self.pushing_back:
                if f_local < v.lbd_v:
                    push_back_count += 1
                    self.push_heap(s, f_local, True)
                    continue

            t3 = time.time()

            time_for_stage_2 += t3 - t2

            # change this to [max_idx + 1, n]
            # save the cost of s, the g value of s, and the h value of s as properties of its node
            for i in range(max_idx + 1, self.ground_size):
                children_count += 1
                if self.model.cost_of_set(node.s) + self.model.cost_of_singleton(i) <= self.model.budget:
                    self.push_heap(list(set(s) | {i}), node.v.lbd_v, candidate=set(node.candidate) - {i},
                                   w=node.budget - self.model.cost_of_singleton(i))

            t4 = time.time()

            time_for_stage_3 += t4 - t3

        stop_time = time.time()

        assert sol is not None, "No solution found."

        ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
               'time': stop_time - start_time, 'node_count': node_count, "open_list_count": open_list_count,
               "push_back_count": push_back_count, "stg0": time_for_stage_0, "stg1": time_for_stage_1,
               "stg2": time_for_stage_2, "stg3": time_for_stage_3,
               'children_count': children_count}

        return ret


class EfficientBranchAndBound(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)
        self.lb_star = None
        self.s_star = None
        self.node_count = 0
        self.basic_mode = False
        self.get_children = None

        self.children_count = 0

        self.start_time = 0
        self.TLE = False
        self.time_limit = 5000

        self.ub_global = 0

        # When True, dump per-node / per-child trace via ``print``. Default off because
        # those statements dominate runtime on non-trivial problems (each line is also
        # tee'd to disk by ``compute_efficient.TeeLogger``).
        self.verbose = False

        # Unified per-run event logger; overridden by ``compute_efficient.py``.
        # ``runlog.verbose`` controls high-frequency events (``NODE_POP`` / ``NODE_BRANCH`` /
        # ``NODE_PRUNE``); independent of :attr:`verbose`, which keeps the legacy plain
        # ``print`` trace.
        from runlog import NullRunLogger as _NullRL
        self.runlog = _NullRL()

    def set_h(self, heuristic):
        # EBB does not use the node heuristic ``self.lbd`` / ``self.inner_h`` anywhere.
        # Keeping the method as a no-op so the algorithm factory can call it uniformly
        # without us silently picking up the base class' ``self.inner_h`` / ``self.lbd``
        # for an algorithm that never reads them.
        return

    def build(self):
        self.lb_star = 0
        self.s_star = []
        self.node_count = 0
        if self.basic_mode:
            self.get_children = self.get_children_basic
        else:
            self.get_children = self.get_children_advance

    def greedy_add(self, t):
        """
        Lazy-greedy fill of ``t``. Returns ``(sol, f_local, c, g_prefix)`` where
        ``g_prefix[i] = self.g(t.s ∪ c[:i])`` so :meth:`get_children_advance` can
        avoid recomputing the prefix objective with another ``model.objective`` call
        for every child (#7).
        """
        def density(ele, base_set):
            return self.model.marginal_gain(ele, list(base_set)) / self.model.cost_of_singleton(ele)

        base = t.s
        sol = set(base)
        remaining_elements = set(t.candidate)
        cur_cost = self.model.cost_of_set(list(sol))

        opt = acclerated_upper_bounds.LazyPlainOptimizer(self.model)
        opt.build(base=base, remaining=remaining_elements)

        g_base = self.g(sol)
        # ``g_prefix[i]`` is the objective at ``t.s ∪ c[:i]``. Initially we only know
        # ``g_prefix[0] = g(t.s)`` and append as elements are committed in greedy order.
        g_prefix = [g_base]

        f_local = g_base + opt.solve(remaining_elements, t.budget)
        c = []

        h = []
        for e in remaining_elements:
            if time.time() - self.start_time > self.time_limit:
                self.TLE = True
                return list(sol), f_local, c, g_prefix
            heapq.heappush(h, (-density(e, base), e))

        while h:
            if time.time() - self.start_time > self.time_limit:
                self.TLE = True
                return list(sol), f_local, c, g_prefix
            _, u = heapq.heappop(h)
            if cur_cost + self.model.cost_of_singleton(u) > t.budget:
                remaining_elements.remove(u)
                continue

            actual_density = density(u, list(sol))

            if not h or actual_density >= -h[0][0]:
                sol.add(u)
                c.append(u)
                remaining_elements.remove(u)
                opt.update_base(sol)
                g_sol = self.g(sol)
                g_prefix.append(g_sol)
                f_temp = g_sol + opt.solve(remaining_set=set(t.candidate) - set(sol),
                                           budget=t.budget)
                if f_local is None or f_temp < f_local:
                    f_local = f_temp
                cur_cost += self.model.cost_of_singleton(u)
            else:
                heapq.heappush(h, (-actual_density, u))

        return list(sol), f_local, c, g_prefix

    def get_children_advance(self, t: BranchAndBoundNode, c, g_prefix=None):
        """
        Build child branches by complementary exclusion of each ``c[i]``.

        ``g_prefix`` (optional, computed by :meth:`greedy_add`) caches
        ``self.model.objective(s ∪ c[:i])`` for ``i = 0..len(c)``; passing it avoids
        ``len(c) + 1`` extra ``model.objective`` calls per node (#7). When omitted
        (legacy callers), we fall back to recomputing — same behavior as before.
        """
        children = []
        s = t.s
        parent_tree_id = int(getattr(t, "tree_id", 0))

        # 1. Instantiate and build the Lazy Optimizer
        opt = acclerated_upper_bounds.LazyPlainOptimizer(self.model)
        opt.build(base=set(s), remaining=set(t.candidate))

        for i in range(0, len(c)):
            base_set = set(s) | set(c[:i])
            remaining_set = set(t.candidate) - set(c[:i + 1])
            budget_i = t.budget - self.model.cost_of_set(c[:i])
            temp = BranchAndBoundNode(list(base_set), list(remaining_set), budget_i)

            # 2. Update base incrementally (O(1) overhead) and solve lazily
            opt.update_base(base_set)
            upper_bound_delta = opt.solve(remaining_set, budget_i)

            if g_prefix is not None and i < len(g_prefix):
                current_f = g_prefix[i]
            else:
                current_f = self.model.objective(list(base_set))

            node_ub = current_f + upper_bound_delta
            if self.alpha * node_ub > self.lb_star:
                if self.verbose:
                    print(f"      [BRANCH exclude={c[i]}] s={list(base_set)} ub={node_ub:.4f}")
                temp.tree_id = self.runlog.next_node_id()
                self.runlog.node_push(
                    parent_id=parent_tree_id,
                    id=int(temp.tree_id),
                    s=runlog_mod.fmt_set(temp.s),
                    ub=float(node_ub),
                    status="pushed",
                    excluded=int(c[i]),
                )
                children.append(temp)
            else:
                if self.verbose:
                    print(f"      [PRUNED exclude={c[i]}] s={list(base_set)} ub={node_ub:.4f} <= lb={self.lb_star:.4f}")
                pre_id = self.runlog.next_node_id()
                self.runlog.node_push(
                    parent_id=parent_tree_id,
                    id=int(pre_id),
                    s=runlog_mod.fmt_set(base_set),
                    ub=float(node_ub),
                    status="pruned_pre",
                    excluded=int(c[i]),
                )
                self.runlog.node_prune(
                    reason="child_ub",
                    ub=float(node_ub),
                    lb=float(self.lb_star),
                    id=int(pre_id),
                )

        # 3. Process the final child (including all elements of c)
        base_set_final = set(s) | set(c)
        remaining_set_final = set(t.candidate) - set(c)
        budget_final = t.budget - self.model.cost_of_set(c)

        temp = BranchAndBoundNode(list(base_set_final), list(remaining_set_final), budget_final)

        opt.update_base(base_set_final)
        upper_bound_delta_final = opt.solve(remaining_set_final, budget_final)
        if g_prefix is not None and len(c) < len(g_prefix):
            current_f_final = g_prefix[len(c)]
        else:
            current_f_final = self.model.objective(list(base_set_final))

        node_ub_final = current_f_final + upper_bound_delta_final
        if node_ub_final > self.lb_star:
            temp.tree_id = self.runlog.next_node_id()
            self.runlog.node_push(
                parent_id=parent_tree_id,
                id=int(temp.tree_id),
                s=runlog_mod.fmt_set(temp.s),
                ub=float(node_ub_final),
                status="pushed",
                excluded=-1,
            )
            children.append(temp)
        else:
            pre_id = self.runlog.next_node_id()
            self.runlog.node_push(
                parent_id=parent_tree_id,
                id=int(pre_id),
                s=runlog_mod.fmt_set(base_set_final),
                ub=float(node_ub_final),
                status="pruned_pre",
                excluded=-1,
            )
            self.runlog.node_prune(
                reason="child_ub",
                ub=float(node_ub_final),
                lb=float(self.lb_star),
                id=int(pre_id),
            )

        return children

    def get_children_basic(self, t: BranchAndBoundNode, c, g_prefix=None):
        # ``g_prefix`` accepted for signature parity with :meth:`get_children_advance`;
        # this variant doesn't use cached objectives.
        del g_prefix
        children = []
        s = t.s
        tc = list(t.cost)

        tc.sort(key=lambda x: self.g_over([x], s) / self.model.cost_of_singleton(x), reverse=True)

        for i in range(0, len(tc)):
            temp = BranchAndBoundNode(list(set(s) | {tc[i]}), list(set(t.cost) - set(tc[:i + 1])),
                                      t.budget - self.model.cost_of_singleton(tc[i]))
            children.append(temp)

        return children


    def bab_stack(self, initial_node: BranchAndBoundNode):
        # Iterative DFS over the branch-and-bound tree. Replaces the (deleted) recursive
        # ``bab`` method which was unused.
        # Tree-mode bookkeeping: the seed ``initial_node`` is the root (id = 0); its
        # children get monotonic ids from ``runlog.next_node_id()``.
        initial_node.tree_id = 0
        stack = [initial_node]
        while stack:
            t = stack.pop()

            if time.time() - self.start_time > self.time_limit:
                self.TLE = True
                return

            self.node_count += 1

            popped_tree_id = int(getattr(t, "tree_id", 0))
            self.runlog.node_pop(
                node=int(self.node_count),
                open=int(len(stack)),
                s_size=len(t.s),
                cost=float(self.model.cost_of_set(t.s)),
                budget=float(self.model.budget),
                ub=float(self.lb_star) if self.lb_star is not None else float("inf"),
                id=popped_tree_id,
                s=runlog_mod.fmt_set(t.s),
            )

            if self.verbose:
                print(
                    f"\n[POP] node={self.node_count} stack={len(stack)} "
                    f"cost={self.model.cost_of_set(t.s):.2f}/{self.model.budget}"
                )
                print(f"   s={t.s}")

            # 1. Pruning / base-case checks
            if len(t.candidate) == 0:
                self.runlog.node_prune(reason="empty_candidate", ub=0.0,
                                       lb=float(self.lb_star) if self.lb_star is not None else 0.0,
                                       id=popped_tree_id)
                continue

            if self.is_on_the_edge(t):
                self.runlog.node_prune(reason="edge", ub=0.0,
                                       lb=float(self.lb_star) if self.lb_star is not None else 0.0,
                                       id=popped_tree_id)
                continue

            # 2. Local greedy search and lower-bound update
            s_primal, f_local, c, g_prefix = self.greedy_add(t)

            if self.TLE:
                return

            primal_val = self.g(s_primal)
            if primal_val > self.lb_star:
                prev_lb = float(self.lb_star) if self.lb_star is not None else 0.0
                if self.verbose:
                    print(
                        f"   [LB+] new_lb={primal_val:.4f} prev={prev_lb:.4f}"
                    )
                self.lb_star = primal_val
                self.s_star = s_primal
                self.runlog.incumbent(
                    f_s=float(primal_val),
                    prev=prev_lb,
                    set_size=len(s_primal),
                    s=runlog_mod.fmt_set(s_primal),
                )

            if self.verbose:
                print(
                    f"   [EVAL] heuristic_seq_len={len(c)} f_local={f_local:.4f} "
                    f"lb={self.lb_star:.4f}"
                )

            # 3. Upper-bound pruning
            if self.alpha * f_local <= self.lb_star:
                if self.verbose:
                    print(f"   [KILLED] alpha*f_local <= lb")
                self.runlog.node_prune(reason="alpha_lb", ub=float(f_local),
                                       lb=float(self.lb_star),
                                       id=popped_tree_id)
                continue

            if self.verbose:
                print(f"   [SURVIVED] expanding children")

            # 4. Branching (reuse cached ``g_prefix`` to avoid recomputing objectives)
            children = self.get_children(t, c, g_prefix=g_prefix)

            self.children_count += len(children)
            if children:
                self.runlog.node_branch(children=len(children), strategy="ebb",
                                        parent_id=popped_tree_id)

            # Push in reverse so the first ``children`` entry is popped first (DFS).
            for t_i in reversed(children):
                stack.append(t_i)

    def optimize(self):
        self.start_time = time.time()
        # Bootstrap incumbent reported once via the unified format (lb_star starts at 0).
        self.runlog.greedy_done(
            f_s=float(self.lb_star or 0.0),
            set_size=len(self.s_star or []),
            s=runlog_mod.fmt_set(self.s_star or []),
        )
        self.bab_stack(BranchAndBoundNode(self.s_star, self.model.ground_set, self.model.budget))
        stop_time = time.time()

        ret = {
            'S': self.s_star,
            'c(S)': self.model.cost_of_set(self.s_star),
            'f(S)': self.model.objective(self.s_star),
            'time': stop_time - self.start_time,
            'node_count': self.node_count,
            'TLE': self.TLE,
            'children_count': self.children_count
        }

        self.runlog.run_end(
            status="TLE" if self.TLE else "OK",
            f_s=float(ret['f(S)']),
            c_s=float(ret['c(S)']),
            nodes=int(ret['node_count']),
            time_s=float(ret['time']),
            children=int(ret['children_count']),
            final_s=runlog_mod.fmt_set(self.s_star or []),
        )
        return ret


class EfficientBFS(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)
        self.max_heap = None
        self.inner_h = None
        self.f = None
        self.d = None
        self.lbd = None
        self.use_alpha = False
        self.pushing_back = True
        self.ground_size = 0
        self.heap_class = 'tradition'

    def build(self):
        if self.heap_class == 'tradition':
            self.max_heap = MaxHeap()
        elif self.heap_class == 'simple':
            self.max_heap = SimpleMaxHeap()

        self.max_heap.clear()
        self.ground_size = len(self.model.ground_set)

        if self.use_alpha:
            self.f = self.f_with_alpha
        else:
            self.f = self.f_without_alpha

    def push_heap(self, s, lbd_v, visited=False, first_child=False, heuristic_sequence=None, candidate=None, w=None,
                  s_max_v=0):
        max_idx = 0
        if len(s) > 0:
            max_idx = max(s)

        node = EfficientBFSHeapObj(s, candidate=candidate, w=w, visited=visited, first_child=first_child,
                                   heuristic_sequence=heuristic_sequence, max_idx=max_idx)
        node.cost = self.model.cost_of_set(s)

        new_g = self.g(node)
        new_h = self.h(node)
        final_v = new_g + new_h

        v = None
        # print(f"examing node {node.s}, {final_v}, budget:{w}, candidate:{set(self.model.ground_set) - set(candidate)}")
        if final_v >= s_max_v:
            if self.use_alpha:
                lbd_v = min(new_g + self.alpha * new_h, lbd_v)
                v = RefinedBFSValue(new_g + self.alpha * new_h, lbd_v, self.d(s))
            else:
                lbd_v = min(new_g + new_h, lbd_v)
                v = RefinedBFSValue(new_g + new_h, lbd_v, self.d(s))

            node.v = v

            self.max_heap.push(node)
            # print(f"child {node.s}, {node.v.lbd_v} pushed")

            return node

        return None

    def greedy_add(self, node):
        def density(ele, base_set):
            return self.model.marginal_gain(ele, list(base_set)) / self.model.cost_of_singleton(ele)

        base = node.s
        candidate = node.candidate
        budget = node.budget

        sol = set(base)
        remaining_elements = set(candidate)
        cur_cost = 0  # Tracks the cost of elements added ON TOP of the base

        # Initialize the Lazy Optimizer
        opt = acclerated_upper_bounds.LazySlicingOptimizer(self.model)
        opt.build(base=base, remaining=remaining_elements)

        # Initial upper bound
        # Assuming node.budget represents the remaining capacity for the candidate set
        f_local = self.g(sol) + opt.solve(remaining_elements, budget)
        heuristic_sequence = []

        # 1. Initialize max-heap for the outer greedy loop
        h = []
        tie_breaker = 0
        for e in remaining_elements:
            heapq.heappush(h, (-density(e, sol), tie_breaker, e))
            tie_breaker += 1

        while h:

            # 2. Pop the element with the highest upper-bound density
            neg_ds, _, u = heapq.heappop(h)

            # 3. Lazy Budget Check: Discard instantly if it no longer fits the knapsack
            cost_u = self.model.cost_of_singleton(u)
            if cur_cost + cost_u > budget:
                continue

            # 4. Evaluate actual density against the dynamically updating solution set
            actual_ds = density(u, sol)

            # 5. Clean up stale/violating elements at the top of the heap
            while h:
                top_e = h[0][2]
                if cur_cost + self.model.cost_of_singleton(top_e) > budget:
                    heapq.heappop(h)
                else:
                    break

            # 6. Check the lazy condition
            if not h or actual_ds >= -h[0][0]:
                # u is the true maximum element. Process it.
                sol.add(u)
                heuristic_sequence.append(u)
                cur_cost += cost_u

                # --- LAZY OPTIMIZER UPPER BOUND ---
                # Update base seamlessly without rebuilding the inner heap
                opt.update_base(sol)

                # Ensure the optimizer strictly uses the remaining budget (budget)
                remaining_for_opt = set(node.candidate) - sol
                f_temp = self.g(sol) + opt.solve(remaining_for_opt, budget)
                if f_local is None or f_temp < f_local:
                    f_local = f_temp

            else:
                # Push the element back into the heap with its newly calculated density
                heapq.heappush(h, (-actual_ds, tie_breaker, u))
                tie_breaker += 1

        return list(sol), f_local, heuristic_sequence

    def push_root(self):
        root = EfficientBFSHeapObj([], candidate=self.model.ground_set, w=self.model.budget, visited=True, max_idx=0)
        root.cost = 0

        f_upper = self.f(root)
        s_max, f_local, heuristic_sequence = self.greedy_add(root)
        # f_upper = min(f_upper, f_local)

        # v = RefinedBFSValue(self.f(root), min(f_local, f_upper), self.d(root.s))
        v = RefinedBFSValue(self.f(root), f_upper, self.d(root.s))
        root.v = v
        root.heuristic_sequence = heuristic_sequence

        self.max_heap.push(root)

        return root, f_upper, heuristic_sequence, s_max

    def optimize(self):
        start_time = time.time()

        root, f_upper, heuristic_sequence, s_max = self.push_root()

        # check if s_max now is an optimal solution
        if self.g(s_max) >= self.alpha * f_upper:
            # print(f"here, g:{self.g(s_max)}, f:{f_upper}")
            sol = s_max
            stop_time = time.time()
            ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
                   'time': stop_time - start_time, 'node_count': 1, "open_list_count": 1,
                   "push_back_count": 0}
            return ret

        push_back_count = 0
        sol = s_max

        node_count = 0
        open_list_count = 1

        time_for_stage_0 = 0
        time_for_stage_1 = 0
        time_for_stage_2 = 0
        time_for_stage_3 = 0
        children_count = 0

        while self.max_heap.size() > 0:
            t0 = time.time()

            node: EfficientBFSHeapObj = self.max_heap.pop()

            node_count += 1
            s = node.s
            v = node.v

            print(f"new node popped:{s}, {v.lbd_v}")

            t1 = time.time()

            time_for_stage_0 += t1 - t0

            f_local, heuristic_sequence = node.v.lbd_v, None
            if not node.visited and not node.first_child:
                s_final, f_local, heuristic_sequence = self.greedy_add(node)
                # node.v.lbd_v = min(node.v.lbd_v, f_local)
                f_upper = min(f_upper, node.v.lbd_v)

                if self.g(s_final) > self.g(s_max):
                    s_max = s_final
                    print(f"s_max updated:{s_max}, early pruning:{min(node.v.lbd_v, f_local) * self.alpha}")

                if min(node.v.lbd_v, f_local) * self.alpha <= self.g(s_max):
                    continue

                if self.use_alpha:
                    if self.g(s_max) >= f_upper:
                        sol = s_max
                        break
                else:
                    if self.g(s_max) >= self.alpha * f_upper:
                        sol = s_max
                        break

            if node.visited or node.first_child:
                heuristic_sequence = node.heuristic_sequence

            t2 = time.time()

            time_for_stage_1 += t2 - t1

            if not node.visited and self.pushing_back:
                if f_local < v.lbd_v:
                    push_back_count += 1
                    node.v.lbd_v = f_local
                    self.max_heap.push(node)
                    continue

            t3 = time.time()

            time_for_stage_2 += t3 - t2

            if self.is_on_the_edge(node):
                continue

            # push first child
            first_ele = heuristic_sequence[0]
            new_candidate = list(set(node.candidate) - {first_ele})
            # new_lbd = min(node.v.lbd_v, f_local)
            new_lbd = node.v.lbd_v

            print(f"current lb_star:{self.g(s_max)}, first ele:{first_ele}, density:{self.model.density(first_ele,s)}")
            if node.cost + self.model.cost_of_singleton(first_ele) <= self.model.budget:
                open_list_count += 1

                new_heuristic_sequence = copy.deepcopy(heuristic_sequence)
                new_heuristic_sequence.pop(0)

                self.push_heap(s=list(set(s) | {first_ele}), lbd_v=new_lbd, first_child=True,
                               heuristic_sequence=new_heuristic_sequence,
                               candidate=new_candidate,
                               w=node.budget - self.model.cost_of_singleton(first_ele), s_max_v=self.g(s_max))

            # push second child
            self.push_heap(s=s, lbd_v=new_lbd, first_child=False,
                           candidate=new_candidate, w=node.budget, s_max_v=self.g(s_max))
            open_list_count += 1

            t4 = time.time()

            time_for_stage_3 += t4 - t3

        stop_time = time.time()

        assert sol is not None, "No solution found."

        ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
               'time': stop_time - start_time, 'node_count': node_count, "open_list_count": open_list_count,
               "push_back_count": push_back_count, "stg0": time_for_stage_0, "stg1": time_for_stage_1,
               "stg2": time_for_stage_2, "stg3": time_for_stage_3,
               'children_count': children_count}
        return ret


class BFSTC(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)
        self.max_heap = None
        self.f = self.f_with_alpha
        # NB: do NOT shadow ``self.h`` with ``None`` here. The base class ``OptimalAlg.h``
        # provides the ``is_on_the_edge`` guard so the early-exit ``self.h(node) == 0``
        # check below actually fires; setting ``self.h = self.inner_h`` would skip that.
        self.heap_class = 'simple'

        self.start_time = 0
        self.time_limit = 5000
        self.TLE = False

        # Unified per-run event logger; overridden by ``compute_efficient.py``.
        # See :mod:`runlog` for the event schema.
        from runlog import NullRunLogger as _NullRL
        self.runlog = _NullRL()

    def build(self):
        if self.heap_class == 'tradition':
            self.max_heap = MaxHeap()
        elif self.heap_class == 'simple':
            self.max_heap = SimpleMaxHeap()

        self.max_heap.clear()

    def greedy_add_plain(self, s):
        base = s
        candidate = set(self.model.ground_set) - set(base)
        budget = self.model.budget

        sol = set(base)
        remaining_elements = set(candidate)
        cur_cost = self.model.cost_of_set(list(sol))

        while len(remaining_elements):
            u, max_density = None, -1.
            for e in remaining_elements:
                # e is an object
                ds = self.model.density(e, list(sol))
                if u is None or ds > max_density:
                    u, max_density = e, ds

            assert u is not None

            if cur_cost + self.model.cost_of_singleton(u) <= budget:
                # satisfy the knapsack constraint
                sol.add(u)
                cur_cost += self.model.cost_of_singleton(u)

            remaining_elements.remove(u)
            # filter out violating elements
            to_remove = set()
            for v in remaining_elements:
                if self.model.cost_of_singleton(v) + cur_cost > budget:
                    to_remove.add(v)
            remaining_elements -= to_remove

        return list(sol)

    def greedy_add(self, s):
        base = s
        candidate = set(self.model.ground_set) - set(base)
        budget = self.model.budget

        sol = set(base)
        remaining_elements = set(candidate)
        cur_cost = self.model.cost_of_set(list(sol))

        # 1. Initialize max-heap with initial densities
        h = []
        tie_breaker = 0
        for e in remaining_elements:
            heapq.heappush(h, (-self.model.density(e, list(sol)), tie_breaker, e))
            tie_breaker += 1

        while h:
            # 2. Pop the element with the highest upper-bound density
            _, _, u = heapq.heappop(h)
            if time.time() - self.start_time > self.time_limit:
                self.TLE = True
                break
            cost_u = self.model.cost_of_singleton(u)

            # 3. Lazy budget check: discard instantly if it exceeds remaining budget
            if cur_cost + cost_u > budget:
                continue

            # 4. Evaluate actual density against the dynamically updating solution set
            actual_density = self.model.density(u, list(sol))

            # 5. Clean up stale/violating elements at the top of the heap
            while h:
                top_e = h[0][2]
                if cur_cost + self.model.cost_of_singleton(top_e) > budget:
                    heapq.heappop(h)
                else:
                    break

            # 6. Check the lazy condition
            if not h or actual_density >= -h[0][0]:
                sol.add(u)
                cur_cost += cost_u
            else:
                heapq.heappush(h, (-actual_density, tie_breaker, u))
                tie_breaker += 1

        return list(sol)

    def optimize(self):
        self.start_time = time.time()

        root = BaseHeapObj([], candidate=self.model.ground_set, budget=self.model.budget)
        root.v = self.f(root)
        # Tree-mode bookkeeping: root id = 0; children get monotonic ids from
        # ``runlog.next_node_id()``. Stored on the heap node so the matching
        # ``NODE_POP`` event can correlate back to its earlier ``NODE_PUSH``.
        root.tree_id = 0

        s_max = self.greedy_add([])
        # Bootstrap incumbent reported once via the unified format.
        f_initial = self.model.objective(list(s_max)) if s_max else 0.0
        self.runlog.greedy_done(f_s=float(f_initial), set_size=len(s_max))
        if self.TLE:
            # Preserve the partial greedy result on TLE instead of returning ``S=[]``;
            # ``greedy_add`` returns ``list(sol)`` even after timing out, and that's
            # the best feasible solution we have at this point.
            ret = {'S': list(s_max),
                   'c(S)': self.model.cost_of_set(list(s_max)),
                   'f(S)': self.model.objective(list(s_max)) if s_max else 0,
                   'TLE': self.TLE,
                   'time': time.time() - self.start_time,
                   'node_count': 0, "open_list_count": 0}

            self.runlog.run_end(
                status="TLE",
                f_s=float(ret["f(S)"]),
                c_s=float(ret["c(S)"]),
                nodes=int(ret["node_count"]),
                time_s=float(ret["time"]),
                final_s=runlog_mod.fmt_set(s_max),
            )
            return ret

        # Sound upper bound on OPT: g(node) + h(node).
        # The previous formulation ``f(node)/alpha = g/alpha + h`` was a valid but loose
        # bound (alpha < 1 inflates ``g``), which delayed the alpha-pruning trigger.
        g_upper = self.g(root) + self.h(root)
        self.max_heap.push(root)

        sol = s_max
        node_count = 0
        open_list_count = 0
        # Once the alpha-feasible sol = s_max has been certified, stop popping; further
        # pops could overwrite ``sol`` via the ``self.h(node) == 0`` branch.
        sol_certified = False
        while self.max_heap.size() > 0 and not sol_certified:
            node: BaseHeapObj = self.max_heap.pop()
            node_count += 1

            self.runlog.node_pop(
                node=int(node_count),
                open=int(self.max_heap.size()),
                s_size=len(node.s),
                cost=float(self.model.cost_of_set(node.s)),
                budget=float(self.model.budget),
                ub=float(self.g(node) + self.h(node)),
                id=int(getattr(node, "tree_id", 0)),
                s=runlog_mod.fmt_set(node.s),
            )

            if time.time() - self.start_time > self.time_limit:
                self.TLE = True
                break

            if self.h(node) == 0:
                sol = node.s
                break

            g_upper = min(g_upper, self.g(node) + self.h(node))

            children_pushed = 0
            parent_tree_id = int(getattr(node, "tree_id", 0))
            for i in node.candidate:
                if self.model.cost_of_singleton(i) <= node.budget:
                    s_final = self.greedy_add(set(node.s) | {i})
                    if self.TLE:
                        break

                    if self.g(s_max) < self.g(s_final):
                        prev_lb = float(self.g(s_max))
                        s_max = s_final
                        self.runlog.incumbent(
                            f_s=float(self.g(s_max)),
                            prev=prev_lb,
                            set_size=len(s_max),
                            s=runlog_mod.fmt_set(s_max),
                        )

                    if self.g(s_max) >= self.alpha * g_upper:
                        sol = s_max
                        sol_certified = True
                        break

                    new_node = BaseHeapObj(set(node.s) | {i}, candidate=set(node.candidate) - {i}, budget=node.budget - self.model.cost_of_singleton(i))
                    new_node.v = self.f(new_node)
                    new_node.tree_id = self.runlog.next_node_id()
                    self.runlog.node_push(
                        parent_id=parent_tree_id,
                        id=int(new_node.tree_id),
                        s=runlog_mod.fmt_set(new_node.s),
                        ub=float(self.g(new_node) + self.h(new_node)),
                        status="pushed",
                        depth=len(new_node.s),
                    )

                    self.max_heap.push(new_node)
                    open_list_count += 1
                    children_pushed += 1

            if children_pushed:
                self.runlog.node_branch(children=children_pushed, strategy="bfstc")

        stop_time = time.time()

        assert sol is not None, "No solution found."
        ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol), 'TLE': self.TLE,
               'time': stop_time - self.start_time, 'node_count': node_count, "open_list_count": open_list_count}
        self.runlog.run_end(
            status="TLE" if self.TLE else "OK",
            f_s=float(ret["f(S)"]),
            c_s=float(ret["c(S)"]),
            nodes=int(ret["node_count"]),
            time_s=float(ret["time"]),
            final_s=runlog_mod.fmt_set(sol),
        )
        return ret


class EfficientBFSNoInherit(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)
        self.max_heap = None
        self.inner_h = None
        self.f = None
        self.d = None
        self.lbd = None
        self.use_alpha = False
        self.pushing_back = True
        self.ground_size = 0
        self.heap_class = 'tradition'

    def build(self):
        if self.heap_class == 'tradition':
            self.max_heap = MaxHeap()
        elif self.heap_class == 'simple':
            self.max_heap = SimpleMaxHeap()

        self.max_heap.clear()
        self.ground_size = len(self.model.ground_set)

        if self.use_alpha:
            self.f = self.f_with_alpha
        else:
            self.f = self.f_without_alpha

    def push_heap(self, s, lbd_v, visited=False, first_child=False, heuristic_sequence=None, candidate=None, w=None,
                  s_max_v=0):
        max_idx = 0
        if len(s) > 0:
            max_idx = max(s)

        node = EfficientBFSHeapObj(s, candidate=candidate, w=w, visited=visited, first_child=first_child,
                                   heuristic_sequence=heuristic_sequence, max_idx=max_idx)
        node.cost = self.model.cost_of_set(s)

        new_g = self.g(node)
        new_h = self.h(node)
        final_v = new_g + new_h

        if final_v >= s_max_v:
            if self.use_alpha:
                node.v = new_g + self.alpha * new_h
            else:
                node.v = new_g + new_h

            self.max_heap.push(node)

            return node

        return None

    def greedy_add(self, node: EfficientBFSHeapObj):
        base = node.s
        candidate = node.candidate
        budget = node.budget

        sol = set(base)
        remaining_elements = set(candidate)
        cur_cost = 0

        # f_local = None
        heuristic_sequence = []
        # print(f"//")
        while len(remaining_elements):
            u, max_density = None, -1.
            for e in remaining_elements:
                # e is an object
                ds = self.model.density(e, list(sol))
                if u is None or ds > max_density:
                    u, max_density = e, ds

            assert u is not None

            if cur_cost + self.model.cost_of_singleton(u) <= budget:
                # satisfy the knapsack constraint
                sol.add(u)
                heuristic_sequence.append(u)
                cur_cost += self.model.cost_of_singleton(u)

            # f_temp = self.g(sol) + self.lbd(base=sol, candidate=set(node.candidate) - set(sol), budget=budget)
            # if f_local is None or f_temp < f_local:
            #     f_local = f_temp
            #     print(f"base:{base}, sol:{sol}, lbd:{f_temp}, c:{len(candidate)}, budget:{budget}")

            remaining_elements.remove(u)
            # filter out violating elements
            to_remove = set()
            for v in remaining_elements:
                if self.model.cost_of_singleton(v) + cur_cost > budget:
                    to_remove.add(v)
            remaining_elements -= to_remove

        return list(sol), heuristic_sequence

    def push_root(self):
        root = EfficientBFSHeapObj([], candidate=self.model.ground_set, w=self.model.budget, visited=True, max_idx=0)
        root.cost = 0

        f_upper = self.f(root)
        s_max, heuristic_sequence = self.greedy_add(root)

        root.v = f_upper
        root.heuristic_sequence = heuristic_sequence

        self.max_heap.push(root)

        return root, f_upper, heuristic_sequence, s_max

    def optimize(self):
        start_time = time.time()

        root, f_upper, heuristic_sequence, s_max = self.push_root()

        # check if s_max now is an optimal solution
        if self.g(s_max) >= self.alpha * f_upper:
            # print(f"here, g:{self.g(s_max)}, f:{f_upper}")
            sol = s_max
            stop_time = time.time()
            ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
                   'time': stop_time - start_time, 'node_count': 1, "open_list_count": 1,
                   "push_back_count": 0}

            return ret

        push_back_count = 0
        sol = s_max

        node_count = 0
        open_list_count = 1

        time_for_stage_0 = 0
        time_for_stage_1 = 0
        time_for_stage_2 = 0
        time_for_stage_3 = 0
        children_count = 0

        while self.max_heap.size() > 0:
            t0 = time.time()

            node: EfficientBFSHeapObj = self.max_heap.pop()
            node_count += 1
            s = node.s
            v = node.v

            t1 = time.time()

            time_for_stage_0 += t1 - t0

            f_local, heuristic_sequence = 0., None
            if not node.visited and not node.first_child:
                s_final, heuristic_sequence = self.greedy_add(node)
                # node.v.lbd_v = min(node.v.lbd_v, f_local)
                f_upper = min(f_upper, v)

                if self.g(s_final) > self.g(s_max):
                    s_max = s_final

                if self.use_alpha:
                    if self.g(s_max) >= f_upper:
                        sol = s_max
                        break
                else:
                    if self.g(s_max) >= self.alpha * f_upper:
                        # print(f"here, s:{s_max}, g:{self.g(s_max)}, f:{f_upper}")
                        sol = s_max
                        break

            if node.visited or node.first_child:
                heuristic_sequence = node.heuristic_sequence

            t2 = time.time()

            time_for_stage_1 += t2 - t1

            if not node.visited and self.pushing_back:
                if f_local < v:
                    push_back_count += 1
                    node.v.lbd_v = f_local
                    self.max_heap.push(node)
                    continue

            t3 = time.time()

            time_for_stage_2 += t3 - t2

            # print(f"vis:{node.visited}, f:{node.first_child}, s:{node.s}, is_on_the_edge:{self.is_on_the_edge(node)}, hs:{heuristic_sequence}, c:{len(node.candidate)}, c:{self.model.cost_of_set(node.s)}, w:{node.budget}")

            if self.is_on_the_edge(node):
                continue

            # for i in node.candidate:
            #     if self.model.cost_of_singleton(i) + self.model.cost_of_set(node.s) <= node.budget:
            #         print(f"i:{i}")

            # push first child
            first_ele = heuristic_sequence[0]
            new_candidate = list(set(node.candidate) - {first_ele})
            if node.cost + self.model.cost_of_singleton(first_ele) <= self.model.budget:
                open_list_count += 1

                new_heuristic_sequence = copy.deepcopy(heuristic_sequence)
                new_heuristic_sequence.pop(0)

                self.push_heap(s=list(set(s) | {first_ele}), lbd_v=v, first_child=True,
                               heuristic_sequence=new_heuristic_sequence,
                               candidate=new_candidate,
                               w=node.budget - self.model.cost_of_singleton(first_ele))

            # push second child
            self.push_heap(s=s, lbd_v=v, first_child=False,
                           candidate=new_candidate, w=node.budget)
            open_list_count += 1

            t4 = time.time()

            time_for_stage_3 += t4 - t3

        stop_time = time.time()

        assert sol is not None, "No solution found."

        ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
               'time': stop_time - start_time, 'node_count': node_count, "open_list_count": open_list_count,
               "push_back_count": push_back_count, "stg0": time_for_stage_0, "stg1": time_for_stage_1,
               "stg2": time_for_stage_2, "stg3": time_for_stage_3,
               'children_count': children_count}

        return ret


class InheritBFS(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)
        self.max_heap = None
        self.inner_h = None
        self.f = None
        self.d = None
        self.lbd = None
        self.use_alpha = False
        self.pushing_back = True
        self.ground_size = 0
        self.heap_class = 'tradition'

    def build(self):
        if self.heap_class == 'tradition':
            self.max_heap = MaxHeap()
        elif self.heap_class == 'simple':
            self.max_heap = SimpleMaxHeap()

        self.max_heap.clear()
        self.ground_size = len(self.model.ground_set)

        if self.use_alpha:
            self.f = self.f_with_alpha
        else:
            self.f = self.f_without_alpha

    def push_heap(self, s, lbd_v, visited=False, candidate=None, w=None,
                  s_max_v=0):
        max_idx = 0
        if len(s) > 0:
            max_idx = max(s)

        node = HeapObj(s, candidate=candidate, w=w, visited=visited, max_idx=max_idx)
        node.cost = self.model.cost_of_set(s)

        new_g = self.g(node)
        new_h = self.h(node)
        final_v = new_g + new_h

        v = None
        if final_v >= s_max_v:
            if self.use_alpha:
                lbd_v = min(new_g + self.alpha * new_h, lbd_v)
                v = RefinedBFSValue(new_g + self.alpha * new_h, lbd_v, self.d(s))
            else:
                lbd_v = min(new_g + new_h, lbd_v)
                v = RefinedBFSValue(new_g + new_h, lbd_v, self.d(s))

            node.v = v

            self.max_heap.push(node)

            return node

        return None

    def greedy_add(self, node: EfficientBFSHeapObj):
        base = node.s
        candidate = node.candidate
        budget = node.budget

        sol = set(base)
        remaining_elements = set(candidate)
        cur_cost = 0

        f_local = None
        # print(f"//")
        while len(remaining_elements):
            u, max_density = None, -1.
            for e in remaining_elements:
                # e is an object
                ds = self.model.density(e, list(sol))
                if u is None or ds > max_density:
                    u, max_density = e, ds

            assert u is not None

            if cur_cost + self.model.cost_of_singleton(u) <= budget:
                # satisfy the knapsack constraint
                sol.add(u)
                cur_cost += self.model.cost_of_singleton(u)

            f_temp = self.g(sol) + self.lbd(base=sol, candidate=set(node.candidate) - set(sol), budget=budget)
            if f_local is None or f_temp < f_local:
                f_local = f_temp
                # print(f"base:{base}, sol:{sol}, lbd:{f_temp}, c:{len(candidate)}, budget:{budget}")

            remaining_elements.remove(u)
            # filter out violating elements
            to_remove = set()
            for v in remaining_elements:
                if self.model.cost_of_singleton(v) + cur_cost > budget:
                    to_remove.add(v)
            remaining_elements -= to_remove

        return list(sol), f_local

    def push_root(self):
        root = EfficientBFSHeapObj([], candidate=self.model.ground_set, w=self.model.budget, visited=True, max_idx=0)
        root.cost = 0

        f_upper = self.f(root)
        s_max, f_local = self.greedy_add(root)
        # f_upper = min(f_upper, f_local)

        v = RefinedBFSValue(self.f(root), min(f_upper, f_local), self.d(root.s))
        root.v = v

        self.max_heap.push(root)

        return root, f_upper, s_max

    def optimize(self):
        start_time = time.time()

        root, f_upper, s_max = self.push_root()

        # check if s_max now is an optimal solution
        if self.g(s_max) >= self.alpha * f_upper:
            # print(f"here, g:{self.g(s_max)}, f:{f_upper}")
            sol = s_max
            stop_time = time.time()
            ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
                   'time': stop_time - start_time, 'node_count': 1, "open_list_count": 1,
                   "push_back_count": 0}

            return ret

        push_back_count = 0
        sol = s_max

        node_count = 0
        open_list_count = 1

        time_for_stage_0 = 0
        time_for_stage_1 = 0
        time_for_stage_2 = 0
        time_for_stage_3 = 0
        children_count = 0

        while self.max_heap.size() > 0:
            t0 = time.time()

            node: EfficientBFSHeapObj = self.max_heap.pop()
            node_count += 1
            s = node.s
            v = node.v

            t1 = time.time()

            time_for_stage_0 += t1 - t0

            f_local = node.v.lbd_v
            if not node.visited:
                s_final, f_local = self.greedy_add(node)
                # node.v.lbd_v = min(node.v.lbd_v, f_local)
                f_upper = min(f_upper, node.v.lbd_v)

                if self.g(s_final) > self.g(s_max):
                    s_max = s_final

                if self.use_alpha:
                    if self.g(s_max) >= f_upper:
                        sol = s_max
                        break
                else:
                    if self.g(s_max) >= self.alpha * f_upper:
                        # print(f"here, s:{s_max}, g:{self.g(s_max)}, f:{f_upper}")
                        sol = s_max
                        break

            t2 = time.time()

            time_for_stage_1 += t2 - t1

            t3 = time.time()

            time_for_stage_2 += t3 - t2

            # print(f"vis:{node.visited}, f:{node.first_child}, s:{node.s}, is_on_the_edge:{self.is_on_the_edge(node)}, hs:{heuristic_sequence}, c:{len(node.candidate)}, c:{self.model.cost_of_set(node.s)}, w:{node.budget}")

            if self.is_on_the_edge(node):
                continue

            # for i in node.candidate:
            #     if self.model.cost_of_singleton(i) + self.model.cost_of_set(node.s) <= node.budget:
            #         print(f"i:{i}")

            # # push first child
            # first_ele = heuristic_sequence[0]
            # new_candidate = list(set(node.candidate) - {first_ele})
            # if node.cost + self.model.cost_of_singleton(first_ele) <= self.model.budget:
            #     open_list_count += 1
            #
            #     new_heuristic_sequence = copy.deepcopy(heuristic_sequence)
            #     new_heuristic_sequence.pop(0)
            #
            #     self.push_heap(s=list(set(s) | {first_ele}), lbd_v=node.v.lbd_v, first_child=True,
            #                    heuristic_sequence=new_heuristic_sequence,
            #                    candidate=new_candidate,
            #                    w=node.budget - self.model.cost_of_singleton(first_ele))
            #
            # # push second child
            # self.push_heap(s=s, lbd_v=node.v.lbd_v, first_child=False,
            #                candidate=new_candidate, w=node.budget)
            # open_list_count += 1

            new_lbd = min(f_local, node.v.lbd_v)
            for i in node.candidate:
                if i > node.max_idx and node.cost + self.model.cost_of_singleton(i) <= self.model.budget:
                    open_list_count += 1
                    self.push_heap(s=list(set(s) | {i}), lbd_v=new_lbd, candidate=set(node.candidate) - {i}, w = node.budget- self.model.cost_of_singleton(i))

            t4 = time.time()

            time_for_stage_3 += t4 - t3

        stop_time = time.time()

        assert sol is not None, "No solution found."

        ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
               'time': stop_time - start_time, 'node_count': node_count, "open_list_count": open_list_count,
               "push_back_count": push_back_count, "stg0": time_for_stage_0, "stg1": time_for_stage_1,
               "stg2": time_for_stage_2, "stg3": time_for_stage_3,
               'children_count': children_count}

        return ret


# class AnytimeEfficientBFSNoInherit(OptimalAlg):
#     def __init__(self, model: BaseTask):
#         super().__init__(model)
#         self.max_heap = None
#         self.inner_h = None
#         self.f = None
#         self.d = None
#         self.lbd = None
#         self.pushing_back = True
#         self.ground_size = 0
#         self.heap_class = 'tradition'
#
#         self.running_time = 0
#
#     def build(self):
#         if self.heap_class == 'tradition':
#             self.max_heap = MaxHeap()
#         elif self.heap_class == 'simple':
#             self.max_heap = SimpleMaxHeap()
#
#         self.max_heap.clear()
#         self.ground_size = len(self.model.ground_set)
#
#         self.f = self.f_without_alpha
#         self.alpha = 0
#
#     def push_heap(self, s, lbd_v, visited=False, first_child=False, heuristic_sequence=None, candidate=None, w=None,
#                   s_max_v=0):
#         max_idx = 0
#         if len(s) > 0:
#             max_idx = max(s)
#
#         node = EfficientBFSHeapObj(s, candidate=candidate, w=w, visited=visited, first_child=first_child,
#                                    heuristic_sequence=heuristic_sequence, max_idx=max_idx)
#         node.cost = self.model.cost_of_set(s)
#
#         new_g = self.g(node)
#         new_h = self.h(node)
#         final_v = new_g + new_h
#         node.v = final_v
#
#         # self.max_heap.push(node)
#         # return node
#
#         if final_v >= s_max_v:
#             node.v = new_g + new_h
#
#             self.max_heap.push(node)
#
#             return node
#
#         return None
#
#     def greedy_add(self, node: EfficientBFSHeapObj):
#         base = node.s
#         candidate = node.candidate
#         budget = node.budget
#
#         sol = set(base)
#         remaining_elements = set(candidate)
#         cur_cost = 0
#
#         # f_local = None
#         heuristic_sequence = []
#         # print(f"//")
#         while len(remaining_elements):
#             u, max_density = None, -1.
#             for e in remaining_elements:
#                 # e is an object
#                 ds = self.model.density(e, list(sol))
#                 if u is None or ds > max_density:
#                     u, max_density = e, ds
#
#             assert u is not None
#
#             if cur_cost + self.model.cost_of_singleton(u) <= budget:
#                 # satisfy the knapsack constraint
#                 sol.add(u)
#                 heuristic_sequence.append(u)
#                 cur_cost += self.model.cost_of_singleton(u)
#
#             # f_temp = self.g(sol) + self.lbd(base=sol, candidate=set(node.candidate) - set(sol), budget=budget)
#             # if f_local is None or f_temp < f_local:
#             #     f_local = f_temp
#             #     print(f"base:{base}, sol:{sol}, lbd:{f_temp}, c:{len(candidate)}, budget:{budget}")
#
#             remaining_elements.remove(u)
#             # filter out violating elements
#             to_remove = set()
#             for v in remaining_elements:
#                 if self.model.cost_of_singleton(v) + cur_cost > budget:
#                     to_remove.add(v)
#             remaining_elements -= to_remove
#
#         return list(sol), heuristic_sequence
#
#     def push_root(self):
#         root = EfficientBFSHeapObj([], candidate=self.model.ground_set, w=self.model.budget, visited=True, max_idx=0)
#         root.cost = 0
#
#         f_upper = self.f(root)
#         s_max, heuristic_sequence = self.greedy_add(root)
#
#         root.v = f_upper
#         root.heuristic_sequence = heuristic_sequence
#
#         self.max_heap.push(root)
#
#         return root, f_upper, heuristic_sequence, s_max
#
#     def optimize(self):
#         start_time = time.time()
#         root, f_upper, heuristic_sequence, s_max = self.push_root()
#         sol = s_max
#
#         node_count = 1
#         open_list_count = 1
#
#         if self.g(s_max) >= f_upper:
#             self.alpha = 1.0
#             stop_time = time.time()
#
#             ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
#                    'alpha': self.alpha,
#                    'time': stop_time - start_time, 'node_count': node_count, "open_list_count": open_list_count}
#             return ret
#
#         while self.max_heap.size() > 0:
#             node: EfficientBFSHeapObj = self.max_heap.pop()
#             node_count += 1
#             s = node.s
#             v = node.v
#
#             if self.g(s_max) >= f_upper:
#                 self.alpha = 1.0
#                 break
#
#             f_local, heuristic_sequence = 0., None
#             if not node.visited and not node.first_child:
#                 s_final, heuristic_sequence = self.greedy_add(node)
#                 # node.v.lbd_v = min(node.v.lbd_v, f_local)
#                 f_upper = min(f_upper, v)
#
#                 if self.g(s_final) > self.g(s_max):
#                     s_max = s_final
#
#                 if self.g(s_max) >= f_upper:
#                     self.alpha = 1.0
#                     break
#
#                 if self.g(s_max) / f_upper > self.alpha:
#                     sol = s_max
#                     self.alpha = self.g(s_max) / f_upper
#
#             if node.visited or node.first_child:
#                 heuristic_sequence = node.heuristic_sequence
#
#             # print(f"vis:{node.visited}, f:{node.first_child}, s:{node.s}, is_on_the_edge:{self.is_on_the_edge(node)}, hs:{heuristic_sequence}, c:{len(node.candidate)}, c:{self.model.cost_of_set(node.s)}, w:{node.budget}")
#
#             if self.is_on_the_edge(node):
#                 continue
#
#             # for i in node.candidate:
#             #     if self.model.cost_of_singleton(i) + self.model.cost_of_set(node.s) <= node.budget:
#             #         print(f"i:{i}")
#
#             # push first child


#             first_ele = heuristic_sequence[0]
#             new_candidate = list(set(node.candidate) - {first_ele})
#             if node.cost + self.model.cost_of_singleton(first_ele) <= self.model.budget:
#                 open_list_count += 1
#
#                 new_heuristic_sequence = copy.deepcopy(heuristic_sequence)
#                 new_heuristic_sequence.pop(0)
#
#                 self.push_heap(s=list(set(s) | {first_ele}), lbd_v=v, first_child=True,
#                                heuristic_sequence=new_heuristic_sequence,
#                                candidate=new_candidate,
#                                w=node.budget - self.model.cost_of_singleton(first_ele))
#
#             # push second child
#             self.push_heap(s=s, lbd_v=v, first_child=False,
#                            candidate=new_candidate, w=node.budget)
#             open_list_count += 1
#
#             stop_time = time.time()
#             if stop_time - start_time > self.running_time:
#                 ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol), 'alpha': self.alpha,
#                        'time': stop_time - start_time, 'node_count': node_count, "open_list_count": open_list_count}
#                 return ret
#
#         assert sol is not None, "No solution found."
#
#         stop_time = time.time()
#         ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol), 'alpha': self.alpha,
#                'time': stop_time - start_time, 'node_count': node_count, "open_list_count": open_list_count}
#         return ret

class AnytimeEfficientBFSNoInherit(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)
        self.max_heap = None
        self.inner_h = None
        self.f = None
        self.d = None
        self.lbd = None
        self.use_alpha = False
        self.pushing_back = False
        self.ground_size = 0
        self.heap_class = 'tradition'

        self.start_time = 0.0
        self.running_time = 0.0
        self.report_interval = 0.0
        self.next_report_time = 0.0
        self.terminated = False

        self.af_plot = []
        self.alpha = 0.0

    def build(self):
        if self.heap_class == 'tradition':
            self.max_heap = MaxHeap()
        elif self.heap_class == 'simple':
            self.max_heap = SimpleMaxHeap()

        self.max_heap.clear()
        self.ground_size = len(self.model.ground_set)

        if self.use_alpha:
            self.f = self.f_with_alpha
        else:
            self.f = self.f_without_alpha

        self.af_plot.clear()
        self.alpha = 0.0
        self.terminated = False

    def push_heap(self, s, lbd_v, visited=False, first_child=False, heuristic_sequence=None, candidate=None, w=None,
                  s_max_v=0):
        max_idx = 0
        if len(s) > 0:
            max_idx = max(s)

        node = EfficientBFSHeapObj(s, candidate=candidate, w=w, visited=visited, first_child=first_child,
                                   heuristic_sequence=heuristic_sequence, max_idx=max_idx)
        node.cost = self.model.cost_of_set(s)

        new_g = self.g(node)
        new_h = self.h(node)
        final_v = new_g + new_h

        if final_v >= s_max_v:
            if self.use_alpha:
                node.v = new_g + self.alpha * new_h
            else:
                node.v = new_g + new_h

            self.max_heap.push(node)

            return node

        return None

    def greedy_add(self, node: EfficientBFSHeapObj):
        base = node.s
        candidate = node.candidate
        budget = node.budget

        sol = set(base)
        remaining_elements = set(candidate)
        cur_cost = 0

        # f_local = None
        heuristic_sequence = []
        # print(f"//")
        while len(remaining_elements):
            elapsed = time.time() - self.start_time

            while elapsed >= self.next_report_time:
                self.af_plot.append(float(self.alpha))
                self.next_report_time += self.report_interval

            if elapsed > self.running_time:
                self.af_plot.append(float(self.alpha))
                self.terminated = True
                break

            u, max_density = None, -1.
            for e in remaining_elements:
                # e is an object
                ds = self.model.density(e, list(sol))
                if u is None or ds > max_density:
                    u, max_density = e, ds

            assert u is not None

            if cur_cost + self.model.cost_of_singleton(u) <= budget:
                # satisfy the knapsack constraint
                sol.add(u)
                heuristic_sequence.append(u)
                cur_cost += self.model.cost_of_singleton(u)

            # f_temp = self.g(sol) + self.lbd(base=sol, candidate=set(node.candidate) - set(sol), budget=budget)
            # if f_local is None or f_temp < f_local:
            #     f_local = f_temp
            #     print(f"base:{base}, sol:{sol}, lbd:{f_temp}, c:{len(candidate)}, budget:{budget}")

            remaining_elements.remove(u)
            # filter out violating elements
            to_remove = set()
            for v in remaining_elements:
                if self.model.cost_of_singleton(v) + cur_cost > budget:
                    to_remove.add(v)
            remaining_elements -= to_remove

        return list(sol), heuristic_sequence

    def push_root(self):
        root = EfficientBFSHeapObj([], candidate=self.model.ground_set, w=self.model.budget, visited=True, max_idx=0)
        root.cost = 0

        f_upper = self.f(root)
        s_max, heuristic_sequence = self.greedy_add(root)

        root.v = f_upper
        root.heuristic_sequence = heuristic_sequence

        self.max_heap.push(root)

        return root, f_upper, heuristic_sequence, s_max

    def optimize(self):
        self.start_time = time.time()
        self.next_report_time = self.report_interval
        self.terminated = False
        self.alpha = 0.0

        root, f_upper, heuristic_sequence, s_max = self.push_root()

        if f_upper > 0:
            self.alpha = self.g(s_max) / f_upper
        else:
            self.alpha = 1.0

        # check if s_max now is an optimal solution
        early_stop = False
        if self.g(s_max) >= f_upper:
            early_stop = True

        if early_stop:
            sol = s_max
            self.af_plot.append(float(self.alpha))
            stop_time = time.time()
            ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
                   'alpha': self.alpha, 'report': self.af_plot,
                   'time': stop_time - self.start_time, 'node_count': 1, "open_list_count": 1,
                   "push_back_count": 0, "stg0": 0, "stg1": 0, "stg2": 0, "stg3": 0, "children_count": 0}

            return ret

        push_back_count = 0
        sol = s_max

        node_count = 0
        open_list_count = 1

        time_for_stage_0 = 0
        time_for_stage_1 = 0
        time_for_stage_2 = 0
        time_for_stage_3 = 0
        children_count = 0

        while self.max_heap.size() > 0:
            elapsed = time.time() - self.start_time

            while elapsed >= self.next_report_time:
                self.af_plot.append(float(self.alpha))
                self.next_report_time += self.report_interval

            if elapsed > self.running_time or self.terminated:
                self.af_plot.append(float(self.alpha))
                break

            t0 = time.time()

            node: EfficientBFSHeapObj = self.max_heap.pop()
            node_count += 1
            s = node.s
            v = node.v

            t1 = time.time()

            time_for_stage_0 += t1 - t0

            f_local, heuristic_sequence = 0., None
            if not node.visited and not node.first_child:
                s_final, heuristic_sequence = self.greedy_add(node)

                if self.terminated:
                    break

                # node.v.lbd_v = min(node.v.lbd_v, f_local)
                f_upper = min(f_upper, v)

                if self.g(s_final) > self.g(s_max):
                    s_max = s_final

                if f_upper > 0:
                    self.alpha = self.g(s_max) / f_upper
                else:
                    self.alpha = 1.0

                if self.g(s_max) >= f_upper:
                    # print(f"here, s:{s_max}, g:{self.g(s_max)}, f:{f_upper}")
                    self.alpha = 1.0
                    sol = s_max
                    self.terminated = True
                    break

            if node.visited or node.first_child:
                heuristic_sequence = node.heuristic_sequence

            t2 = time.time()

            time_for_stage_1 += t2 - t1

            t3 = time.time()

            time_for_stage_2 += t3 - t2

            # print(f"vis:{node.visited}, f:{node.first_child}, s:{node.s}, is_on_the_edge:{self.is_on_the_edge(node)}, hs:{heuristic_sequence}, c:{len(node.candidate)}, c:{self.model.cost_of_set(node.s)}, w:{node.budget}")

            if self.is_on_the_edge(node):
                continue

            # for i in node.candidate:
            #     if self.model.cost_of_singleton(i) + self.model.cost_of_set(node.s) <= node.budget:
            #         print(f"i:{i}")

            # push first child
            first_ele = heuristic_sequence[0]
            new_candidate = list(set(node.candidate) - {first_ele})
            if node.cost + self.model.cost_of_singleton(first_ele) <= self.model.budget:
                open_list_count += 1

                new_heuristic_sequence = copy.deepcopy(heuristic_sequence)
                new_heuristic_sequence.pop(0)

                self.push_heap(s=list(set(s) | {first_ele}), lbd_v=v, first_child=True,
                               heuristic_sequence=new_heuristic_sequence,
                               candidate=new_candidate,
                               w=node.budget - self.model.cost_of_singleton(first_ele))

            # push second child
            self.push_heap(s=s, lbd_v=v, first_child=False,
                           candidate=new_candidate, w=node.budget)
            open_list_count += 1

            t4 = time.time()

            time_for_stage_3 += t4 - t3

        stop_time = time.time()

        assert sol is not None, "No solution found."

        ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
               'alpha': self.alpha, 'report': self.af_plot,
               'time': stop_time - self.start_time, 'node_count': node_count, "open_list_count": open_list_count,
               "push_back_count": push_back_count, "stg0": time_for_stage_0, "stg1": time_for_stage_1,
               "stg2": time_for_stage_2, "stg3": time_for_stage_3,
               'children_count': children_count}

        return ret


class AnytimeEfficientBFS(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)
        self.max_heap = None
        self.inner_h = None
        self.f = None
        self.d = None
        self.lbd = None
        self.use_alpha = False
        self.pushing_back = True
        self.ground_size = 0
        self.heap_class = 'tradition'

        self.start_time = 0
        self.running_time = 0
        self.report_interval = 0
        self.next_report_time = 0
        self.terminated = False

        self.af_plot = []
        self.report_mode = 'utility'
        self.s_max = []

    def report(self):
        # if self.report_mode == 'alpha':
        #     self.af_plot.append(float(self.alpha))
        # elif self.report_mode == 'utility':
        #     self.af_plot.append(float(self.g(list(self.s_max))))

        self.af_plot.append((float(self.g(list(self.s_max))), self.alpha))

    def build(self):
        if self.heap_class == 'tradition':
            self.max_heap = MaxHeap()
        elif self.heap_class == 'simple':
            self.max_heap = SimpleMaxHeap()

        self.max_heap.clear()
        self.af_plot.clear()
        self.ground_size = len(self.model.ground_set)

        if self.use_alpha:
            self.f = self.f_with_alpha
        else:
            self.f = self.f_without_alpha

    def push_heap(self, s, lbd_v, visited=False, first_child=False, heuristic_sequence=None, candidate=None, w=None,
                  s_max_v=0):
        max_idx = 0
        if len(s) > 0:
            max_idx = max(s)

        node = EfficientBFSHeapObj(s, candidate=candidate, w=w, visited=visited, first_child=first_child,
                                   heuristic_sequence=heuristic_sequence, max_idx=max_idx)
        node.cost = self.model.cost_of_set(s)

        new_g = self.g(node)
        new_h = self.h(node)
        final_v = new_g + new_h

        v = None
        if final_v >= s_max_v:
            if self.use_alpha:
                lbd_v = min(new_g + self.alpha * new_h, lbd_v)
                v = RefinedBFSValue(new_g + self.alpha * new_h, lbd_v, self.d(s))
            else:
                lbd_v = min(new_g + new_h, lbd_v)
                v = RefinedBFSValue(new_g + new_h, lbd_v, self.d(s))

            node.v = v

            self.max_heap.push(node)

            return node

        return None

    # def greedy_add_plain(self, node: EfficientBFSHeapObj):
    #     base = node.s
    #     candidate = node.candidate
    #     budget = node.budget
    #     base_cost = self.model.cost_of_set(base)
    #
    #     sol = set(base)
    #     remaining_elements = set(candidate)
    #     cur_cost = 0
    #
    #     f_local = None
    #     heuristic_sequence = []
    #
    #     # print(f"//")
    #     while len(remaining_elements):
    #         elapsed = time.time() - self.start_time
    #
    #         if self.report_interval > 0:
    #             while elapsed >= self.next_report_time:
    #                 self.af_plot.append(float(self.alpha))
    #                 self.next_report_time += self.report_interval
    #
    #         if elapsed > self.running_time:
    #             self.af_plot.append(float(self.alpha))
    #             self.terminated = True
    #             break
    #
    #         u, max_density = None, -1.
    #         for e in remaining_elements:
    #             # e is an object
    #             ds = self.model.density(e, list(sol))
    #             if u is None or ds > max_density:
    #                 u, max_density = e, ds
    #
    #         assert u is not None
    #
    #         if cur_cost + self.model.cost_of_singleton(u) <= budget:
    #             # satisfy the knapsack constraint
    #             sol.add(u)
    #             heuristic_sequence.append(u)
    #             cur_cost += self.model.cost_of_singleton(u)
    #
    #         f_temp = self.g(sol) + self.lbd(base=sol, candidate=set(node.candidate) - set(sol), budget=budget)
    #         if f_local is None or f_temp < f_local:
    #             f_local = f_temp
    #             # print(f"base:{base}, sol:{sol}, lbd:{f_temp}, c:{len(candidate)}, budget:{budget}")
    #
    #         remaining_elements.remove(u)
    #         # filter out violating elements
    #         to_remove = set()
    #         for v in remaining_elements:
    #             if self.model.cost_of_singleton(v) + cur_cost > budget:
    #                 to_remove.add(v)
    #         remaining_elements -= to_remove
    #
    #     return list(sol), f_local, heuristic_sequence

    def greedy_add(self, node):
        def density(ele, base_set):
            return self.model.marginal_gain(ele, list(base_set)) / self.model.cost_of_singleton(ele)

        base = node.s
        candidate = node.candidate
        budget = node.budget

        sol = set(base)
        remaining_elements = set(candidate)
        cur_cost = 0  # Tracks the cost of elements added ON TOP of the base

        # Initialize the Lazy Optimizer
        opt = acclerated_upper_bounds.LazySlicingOptimizer(self.model)
        opt.build(base=base, remaining=remaining_elements)

        # Initial upper bound
        # Assuming node.budget represents the remaining capacity for the candidate set
        f_local = self.g(sol) + opt.solve(remaining_elements, budget)
        heuristic_sequence = []

        # 1. Initialize max-heap for the outer greedy loop
        h = []
        tie_breaker = 0
        for e in remaining_elements:
            heapq.heappush(h, (-density(e, sol), tie_breaker, e))
            tie_breaker += 1

        while h:
            # --- Time Tracking ---
            elapsed = time.time() - self.start_time

            if self.report_interval > 0:
                while elapsed >= self.next_report_time:
                    self.report()
                    self.next_report_time += self.report_interval

            if elapsed > self.running_time:
                self.report()
                self.terminated = True
                break

            # 2. Pop the element with the highest upper-bound density
            neg_ds, _, u = heapq.heappop(h)

            # 3. Lazy Budget Check: Discard instantly if it no longer fits the knapsack
            cost_u = self.model.cost_of_singleton(u)
            if cur_cost + cost_u > budget:
                continue

            # 4. Evaluate actual density against the dynamically updating solution set
            actual_ds = density(u, sol)

            # 5. Clean up stale/violating elements at the top of the heap
            while h:
                top_e = h[0][2]
                if cur_cost + self.model.cost_of_singleton(top_e) > budget:
                    heapq.heappop(h)
                else:
                    break

            # 6. Check the lazy condition
            if not h or actual_ds >= -h[0][0]:
                # u is the true maximum element. Process it.
                sol.add(u)
                heuristic_sequence.append(u)
                cur_cost += cost_u

                # --- LAZY OPTIMIZER UPPER BOUND ---
                # Update base seamlessly without rebuilding the inner heap
                opt.update_base(sol)

                # Ensure the optimizer strictly uses the remaining budget (budget)
                remaining_for_opt = set(node.candidate) - sol
                f_temp = self.g(sol) + opt.solve(remaining_for_opt, budget)
                if f_local is None or f_temp < f_local:
                    f_local = f_temp

            else:
                # Push the element back into the heap with its newly calculated density
                heapq.heappush(h, (-actual_ds, tie_breaker, u))
                tie_breaker += 1

        return list(sol), f_local, heuristic_sequence

    def push_root(self):
        root = EfficientBFSHeapObj([], candidate=self.model.ground_set, w=self.model.budget, visited=True, max_idx=0)
        root.cost = 0

        t0 = time.time()
        s_max, f_local, heuristic_sequence = self.greedy_add(root)
        # print(f"t:{time.time()-t0}")
        f_upper = f_local

        # v = RefinedBFSValue(self.f(root), min(f_local, f_upper), self.d(root.s))
        v = RefinedBFSValue(self.f(root), f_upper, self.d(root.s))
        root.v = v
        root.heuristic_sequence = heuristic_sequence

        self.max_heap.push(root)

        return root, f_upper, heuristic_sequence, s_max

    def optimize(self):
        self.start_time = time.time()
        self.next_report_time = self.report_interval

        t0 = time.time()
        root, f_upper, heuristic_sequence, self.s_max = self.push_root()
        # check if s_max now is an optimal solution
        if self.g(self.s_max) >= f_upper:
            # print(f"here, g v:{self.g(self.s_max)}, f:{f_upper}, g:{time.time() - t0}")
            sol = self.s_max
            self.alpha = 1.0
            self.report()
            stop_time = time.time()

            ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),'alpha': self.alpha, 'report': self.af_plot,
                   'time': stop_time - self.start_time,  'node_count': 1, "open_list_count": 1,
                   "push_back_count": 0}

            return ret

        push_back_count = 0
        sol = self.s_max

        node_count = 0
        open_list_count = 1

        while self.max_heap.size() > 0:
            elapsed = time.time() - self.start_time
            while elapsed > self.next_report_time:
                self.report()
                self.next_report_time += self.report_interval
            if elapsed > self.running_time:
                self.report()
                ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol), 'alpha': self.alpha, 'report': self.af_plot,
                       'time': elapsed, 'node_count': node_count, "open_list_count": open_list_count}
                return ret

            node: EfficientBFSHeapObj = self.max_heap.pop()
            node_count += 1
            s = node.s
            v = node.v

            if self.g(self.s_max) >= f_upper:
                self.alpha = 1.0
                self.report()
                break

            f_local, heuristic_sequence = node.v.lbd_v, None
            if not node.visited and not node.first_child:
                s_final, f_local, heuristic_sequence = self.greedy_add(node)
                if self.terminated:
                    ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
                           'alpha': self.alpha, 'report': self.af_plot,
                           'time': elapsed, 'node_count': node_count,
                           "open_list_count": open_list_count}
                    return ret

                # node.v.lbd_v = min(node.v.lbd_v, f_local)
                f_upper = min(f_upper, node.v.lbd_v)

                if self.g(s_final) > self.g(self.s_max):
                    self.s_max = s_final

                if self.g(self.s_max) >= f_upper:
                    self.alpha = 1.0
                    self.report()
                    break

                if self.g(self.s_max) / f_upper > self.alpha:
                    sol = self.s_max
                    self.alpha = self.g(self.s_max) / f_upper

            # report_interval_timer += time.time() - loop_timer
            # if report_interval_timer >= self.report_interval:
            #     report_interval_timer = 0
            #     self.af_plot.append(float(self.alpha))
            # loop_timer = time.time()
            #
            # if loop_timer - start_time > self.running_time:
            #     self.af_plot.append(float(self.alpha))
            #     ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol), 'alpha': self.alpha,'report': self.af_plot,
            #            'time': loop_timer - start_time, 'node_count': node_count, "open_list_count": open_list_count}
            #     return ret

            if node.visited or node.first_child:
                heuristic_sequence = node.heuristic_sequence

            if not node.visited and self.pushing_back:
                if f_local < v.lbd_v:
                    push_back_count += 1
                    node.v.lbd_v = f_local
                    self.max_heap.push(node)
                    continue

            # print(f"vis:{node.visited}, f:{node.first_child}, s:{node.s}, is_on_the_edge:{self.is_on_the_edge(node)}, hs:{heuristic_sequence}, c:{len(node.candidate)}, c:{self.model.cost_of_set(node.s)}, w:{node.budget}")

            if self.is_on_the_edge(node):
                continue

            # for i in node.candidate:
            #     if self.model.cost_of_singleton(i) + self.model.cost_of_set(node.s) <= node.budget:
            #         print(f"i:{i}")

            # push first child
            first_ele = heuristic_sequence[0]
            new_candidate = list(set(node.candidate) - {first_ele})
            new_lbd = min(node.v.lbd_v, f_local) if f_local is not None else node.v.lbd_v
            # new_lbd = node.v.lbd_v
            if node.cost + self.model.cost_of_singleton(first_ele) <= self.model.budget:
                open_list_count += 1

                new_heuristic_sequence = copy.deepcopy(heuristic_sequence)
                new_heuristic_sequence.pop(0)

                self.push_heap(s=list(set(s) | {first_ele}), lbd_v=new_lbd, first_child=True,
                               heuristic_sequence=new_heuristic_sequence,
                               candidate=new_candidate,
                               w=node.budget - self.model.cost_of_singleton(first_ele))

            # push second child
            self.push_heap(s=s, lbd_v=new_lbd, first_child=False,
                           candidate=new_candidate, w=node.budget)
            open_list_count += 1

        assert sol is not None, "No solution found."

        if self.max_heap.size() == 0 and not self.terminated:
            self.alpha = 1.0
            if len(self.af_plot) == 0 or self.af_plot[-1] != 1.0:
                self.report()

        stop_time = time.time()
        ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol), 'alpha': self.alpha,'report':self.af_plot,
               'time': stop_time - self.start_time, 'node_count': node_count, "open_list_count": open_list_count}

        return ret


class AnytimeEfficientBranchAndBound(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)

        self.lb_star = None
        self.s_star = None
        self.lbd = None
        self.node_count = 0
        self.basic_mode = False
        self.get_children = None

        self.af_plot = []
        self.report_mode = 'utility'

        self.running_time = 0.0

        self.report_interval = 0.0
        self.next_report_time = 0

        self.start_time = 0.0

        self.ret = None
        self.terminated = False

        self.children_count = 0

    def set_h(self, heuristic):
        if heuristic == 'ub0':
            self.lbd = self.lbd0
        elif heuristic == 'ub2':
            self.lbd = self.lbd2
        elif heuristic == 'dom':
            self.lbd = self.lbd_dom

    def report(self):
        if self.report_mode == 'alpha':
            self.af_plot.append(float(self.alpha))
        elif self.report_mode == 'utility':
            self.af_plot.append(float(self.lb_star))

    def build(self):
        self.lb_star = 0
        self.s_star = []
        self.node_count = 0
        self.af_plot = []
        self.alpha = 0.0

        if self.basic_mode:
            self.get_children = self.get_children_basic
        else:
            self.get_children = self.get_children_advance

    # def greedy_add(self, t):
    #     base = t.s
    #     sol = set(base)
    #     base_cost = self.model.cost_of_set(list(sol))
    #     remaining_elements = set(t.candidate)
    #     cur_cost = self.model.cost_of_set(list(sol))
    #
    #     opt = acclerated_upper_bounds.LazyOptimizer(self.model)
    #     opt.build(base=base, remaining=remaining_elements)
    #
    #     f_local = self.g(sol) + opt.solve(remaining_elements, self.model.budget - base_cost)
    #
    #     c = []
    #     while len(remaining_elements):
    #         # --- Insert Time Tracking Here ---
    #         elapsed = time.time() - self.start_time
    #         if self.report_interval > 0:
    #             while elapsed >= self.next_report_time:
    #                 self.af_plot.append(float(self.alpha))
    #                 self.next_report_time += self.report_interval
    #
    #         if elapsed > self.running_time:
    #             self.af_plot.append(float(self.alpha))
    #             self.terminated = True
    #             break  # Exit the heavy loop early
    #
    #         u, max_density = None, -1.
    #         for e in remaining_elements:
    #             # e is an object
    #             ds = self.model.density(e, list(sol))
    #             if u is None or ds > max_density:
    #                 u, max_density = e, ds
    #         assert u is not None
    #         if cur_cost + self.model.cost_of_singleton(u) <= self.model.budget:
    #             # satisfy the knapsack constraint
    #             sol.add(u)
    #             c.append(u)
    #             cur_cost += self.model.cost_of_singleton(u)
    #
    #         remaining_elements.remove(u)
    #         # filter out violating elements
    #         to_remove = set()
    #         for v in remaining_elements:
    #             if self.model.cost_of_singleton(v) + cur_cost > self.model.budget:
    #                 to_remove.add(v)
    #         remaining_elements -= to_remove
    #
    #         t0 = time.time()
    #         opt.update_base(sol)
    #         f_temp = self.g(sol) + opt.solve(remaining_set=set(t.candidate) - set(sol), budget=self.model.budget - base_cost)
    #         print(f"checkpoint:{time.time() - t0}")
    #
    #         if f_local is None or f_temp < f_local:
    #             f_local = f_temp
    #
    #     return list(sol), f_local, c

    def greedy_add(self, t):
        def density(ele, base_set):
            return self.model.marginal_gain(ele, list(base_set)) / self.model.cost_of_singleton(ele)

        base = t.s
        sol = set(base)
        base_cost = self.model.cost_of_set(list(sol))
        remaining_elements = set(t.candidate)
        cur_cost = self.model.cost_of_set(list(sol))

        opt = acclerated_upper_bounds.LazyPlainOptimizer(self.model)
        opt.build(base=base, remaining=remaining_elements)

        f_local = self.g(sol) + opt.solve(remaining_elements, t.budget)
        c = []

        # 1. Initialize the max heap for outer greedy loop
        h = []
        for e in remaining_elements:
            heapq.heappush(h, (-density(e, base), e))

        while h:
            # --- Insert Time Tracking Here ---
            elapsed = time.time() - self.start_time
            if self.report_interval > 0:
                while elapsed >= self.next_report_time:
                    self.report()
                    self.next_report_time += self.report_interval

            if elapsed > self.running_time:
                self.report()
                self.terminated = True
                break  # Exit the heavy loop early

            _, u = heapq.heappop(h)
            if cur_cost + self.model.cost_of_singleton(u) > t.budget:
                # u does not satisfy the knapsack constraint
                remaining_elements.remove(u)
                continue

            # 2. Evaluate the actual density
            actual_density = density(u, list(sol))

            if not h or actual_density >= -h[0][0]:
                sol.add(u)
                c.append(u)
                remaining_elements.remove(u)
                opt.update_base(sol)
                f_temp = self.g(sol) + opt.solve(remaining_set=set(t.candidate) - set(sol),
                                                 budget=t.budget)
                if f_local is None or f_temp < f_local:
                    f_local = f_temp
                cur_cost += self.model.cost_of_singleton(u)
            else:
                heapq.heappush(h, (-actual_density, u))

        return list(sol), f_local, c

    def get_children_basic(self, t: BranchAndBoundNode, c):
        children = []
        s = t.s
        tc = list(t.cost)

        tc.sort(key=lambda x: self.g_over([x], s) / self.model.cost_of_singleton(x), reverse=True)

        for i in range(0, len(tc)):
            temp = BranchAndBoundNode(list(set(s) | {tc[i]}), list(set(t.cost) - set(tc[:i + 1])),
                                      t.budget - self.model.cost_of_singleton(tc[i]))
            children.append(temp)

        return children

    def get_children_advance(self, t: BranchAndBoundNode, c):
        children = []
        s = t.s

        # 1. Instantiate and build the Lazy Optimizer ONCE for this node
        opt = acclerated_upper_bounds.LazyPlainOptimizer(self.model)
        opt.build(base=set(s), remaining=set(t.candidate))

        for i in range(0, len(c)):
            # --- Time Tracking Check to Prevent Hanging ---
            elapsed = time.time() - self.start_time
            if elapsed > self.running_time:
                self.terminated = True
                return children

            base_set = set(s) | set(c[:i])
            remaining_set = set(t.candidate) - set(c[:i + 1])
            budget_i = t.budget - self.model.cost_of_set(c[:i])

            temp = BranchAndBoundNode(list(base_set), list(remaining_set), budget_i)

            # 2. Update base incrementally (O(1) overhead) and solve lazily
            opt.update_base(base_set)
            upper_bound_delta = opt.solve(remaining_set, budget_i)

            current_f = self.model.objective(list(base_set))

            if current_f + upper_bound_delta > self.lb_star:
                children.append(temp)

        # 3. Process the final child (including all elements of c)
        base_set_final = set(s) | set(c)
        remaining_set_final = set(t.candidate) - set(c)
        budget_final = t.budget - self.model.cost_of_set(c)

        temp = BranchAndBoundNode(list(base_set_final), list(remaining_set_final), budget_final)

        opt.update_base(base_set_final)
        upper_bound_delta_final = opt.solve(remaining_set_final, budget_final)
        current_f_final = self.model.objective(list(base_set_final))

        if current_f_final + upper_bound_delta_final > self.lb_star:
            children.append(temp)

        return children

    def bab(self, t: BranchAndBoundNode):
        if self.terminated:
            print(f"stop here")
            return

        elapsed = time.time() - self.start_time
        if elapsed > self.running_time:
            self.report()
            self.terminated = True
            return

        while elapsed >= self.next_report_time:
            self.report()
            self.next_report_time += self.report_interval

        self.node_count = self.node_count + 1

        # print(f"reach here")
        if len(t.candidate) == 0:
            # print(f"no candidate here:{time.time() - self.start_time}")
            print("here 0")
            self.report()
            return

        if self.is_on_the_edge(t):
            # print(f"on the edge here:{time.time() - self.start_time}")
            print("here 1")
            self.report()
            return

        t0 = time.time()
        s_primal, f_local, c = self.greedy_add(t)
        print(f"g:{time.time() - t0}")
        if self.terminated:
            # print(f"terminate here:{time.time() - self.start_time}")
            print("here 2")
            return

        if self.g(s_primal) > self.lb_star:
            self.lb_star = self.g(s_primal)
            self.s_star = s_primal

        # print(f"greedy time:{time.time() - t0},ub:{f_local}, s_primal:{s_primal}, v:{self.g(s_primal)},  lb_star:{self.lb_star}, alpha:{self.alpha}")

        ub = f_local
        if self.lb_star >= ub:
            self.report()
            print("here 3")
            return

        # if self.alpha * ub <= self.lb_star:
        #     self.alpha = self.lb_star/ub

        children = self.get_children(t, c)

        if self.terminated:
            self.report()
            return

        for t_i in children:
            if not self.terminated:
                self.bab(t_i)

        self.children_count += len(children)

    def optimize(self):
        self.start_time = time.time()
        self.next_report_time = self.report_interval

        self.bab(BranchAndBoundNode(self.s_star, self.model.ground_set, self.model.budget))

        stop_time = time.time()
        ret = {
            'S': self.s_star,
            'c(S)': self.model.cost_of_set(self.s_star),
            'f(S)': self.model.objective(self.s_star),
            'alpha': self.alpha,
            'report': self.af_plot,
            'time': stop_time - self.start_time,
            'node_count': self.node_count,
            'children_count': self.children_count
        }

        return ret


class MCTSNode:
    """
    A node in the Monte Carlo Search Tree, adapted for the Knapsack Constraint.
    """

    def __init__(self, state, parent=None):
        # The state is now a tuple: (current_solution_set, remaining_candidates_list, budget_remaining)
        self.state = state
        self.parent = parent
        self.children = []

        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        """ A node is fully expanded if all possible legal moves have been explored. """
        # In our binary model, this means 2 children if "take" is possible, or 1 if only "discard" is.
        _, remaining_r, budget_rem = self.state
        if not remaining_r:
            return True  # No more candidates

        element_to_decide = remaining_r[0]
        cost_of_element = MCTSNode.model.cost_of_singleton(element_to_decide)

        if cost_of_element > budget_rem:
            # "Take" is impossible, so it's fully expanded if the "discard" child exists.
            return len(self.children) == 1
        else:
            # "Take" is possible, so it's fully expanded if both children exist.
            return len(self.children) == 2

    def is_on_the_edge(self):
        _, remaining_r, budget_rem = self.state
        if not remaining_r:
            return True  # No candidates left

        for item in remaining_r:
            if MCTSNode.model.cost_of_singleton(item) <= budget_rem:
                return False  # Found an item that can fit, so not on the edge

        return True


class AnytimeAugmentedMCTS(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)
        self.start_time = 0.0
        self.running_time = 0.0
        self.report_interval = 0.0
        self.next_report_time = 0.0
        self.terminated = False
        self.epsilon = 0.1

        self.af_plot = []
        self.alpha = 0.0

        self.exploration_constant = 1.414
        self.root = None
        self.s_max = []
        self.g_s_max = 0.0

    def build(self):
        self.af_plot.clear()
        self.alpha = 0.0
        self.terminated = False
        MCTSNode.model = self.model

        # Sort candidates by a heuristic, e.g., density (value/cost)
        initial_candidates = sorted(
            list(self.model.ground_set),
            key=lambda e: (self.model.objective([e]) + self.lbd2({e}, list(set(self.model.ground_set) - {e}), self.model.budget - self.model.cost_of_singleton(e))) / self.model.cost_of_singleton(e) if self.model.cost_of_singleton(
                e) > 0 else float('inf'),
            reverse=True
        )

        root_state = (set(), initial_candidates, self.model.budget)
        self.root = MCTSNode(root_state)

        # Initialize s_max with a simple greedy solution respecting knapsack constraint.
        self.s_max = self._initial_greedy_solution()
        self.g_s_max = self.g(self.s_max)

    def _initial_greedy_solution(self):
        """ A simple greedy packing based on the sorted candidates. """
        sol = []
        current_cost = 0
        _, initial_candidates, _ = self.root.state
        for item in initial_candidates:
            cost = self.model.cost_of_singleton(item)
            if current_cost + cost <= self.model.budget:
                sol.append(item)
                current_cost += cost
        return sol

    def _select(self, node):
        current_node = node
        while not current_node.is_on_the_edge():
            if not current_node.is_fully_expanded():
                return current_node
            current_node = self._best_child_ucb(current_node)
        return current_node

    def _expand(self, node):
        """
        Expands a node by creating EXACTLY ONE new child node.
        It follows a fixed order: first try to create "take", then "discard".
        """
        if node.is_on_the_edge():  # Safety check
            return node

        current_s, remaining_r, budget_rem = node.state
        element_to_decide = remaining_r[0]
        next_r = remaining_r[1:]
        cost_of_element = self.model.cost_of_singleton(element_to_decide)

        # --- Logic to create just one child ---
        # 1. Check if the "take" action is possible and if its child has been created.
        take_possible = cost_of_element <= budget_rem
        take_child_exists = False
        if take_possible:
            for child in node.children:
                # A "take" child is identified by having one more element in its solution set
                if len(child.state[0]) > len(current_s):
                    take_child_exists = True
                    break

        if take_possible and not take_child_exists:
            # If "take" is possible and not yet created, create it.
            take_s = current_s.union({element_to_decide})
            take_state = (take_s, next_r, budget_rem - cost_of_element)
            new_child = MCTSNode(take_state, parent=node)
            node.children.append(new_child)
            return new_child  # Return the newly created "take" node
        else:
            discard_s = current_s
            discard_state = (discard_s, next_r, budget_rem)

            # We must explicitly check if the discard child already exists.
            discard_child_exists = False
            for child in node.children:
                # A "discard" child is identified by having the same number of elements
                if len(child.state[0]) == len(node.state[0]):
                    discard_child_exists = True
                    break

            if not discard_child_exists:
                discard_s = node.state[0]
                new_child = MCTSNode(discard_state, parent=node)
                node.children.append(new_child)
                return new_child
            else:
                return self._best_child_ucb(node)


    def _simulate(self, node):
        current_s, remaining_r, budget_rem = node.state
        sim_s = set(current_s)

        # Randomly try to pack remaining items
        candidates_to_try = list(remaining_r)
        random.shuffle(candidates_to_try)

        for item in candidates_to_try:
            cost = self.model.cost_of_singleton(item)
            if cost <= budget_rem:
                sim_s.add(item)
                budget_rem -= cost

        reward = self.g(list(sim_s))
        if reward > self.g_s_max:
            self.s_max = list(sim_s)
            self.g_s_max = reward
        return reward

    def _greedy_simulate(self, node):
        # --- 1. Initialization ---
        current_s, remaining_r, budget_rem = node.state
        sim_s = set(current_s)

        # We need a mutable list of candidates for this strategy
        candidates = list(remaining_r)

        # --- 2. Iteratively build the solution ---
        while candidates:
            min_cost_remaining = float('inf')
            possible_to_add = False
            for item in candidates:
                cost = self.model.cost_of_singleton(item)
                if cost <= budget_rem:
                    possible_to_add = True
                    min_cost_remaining = min(min_cost_remaining, cost)

            if not possible_to_add:
                # If no single remaining item can fit, the simulation for this path is done.
                break

            # --- 2a. ε-Greedy Decision ---
            if random.random() < self.epsilon:
                # --- Exploration ---
                # Select a random valid item (one that fits the budget)
                fittable_candidates = [c for c in candidates if self.model.cost_of_singleton(c) <= budget_rem]
                if not fittable_candidates:
                    break
                element_to_add = random.choice(fittable_candidates)
            else:
                # --- Exploitation ---
                # Find the best valid item according to marginal gain (or density for knapsack)
                best_element = None
                max_density = -float('inf')

                for e in candidates:
                    cost = self.model.cost_of_singleton(e)
                    if cost <= budget_rem:
                        marginal_density = (self.g(list(sim_s.union({e}))) - self.g(list(sim_s)))/self.model.cost_of_singleton(e)

                        if marginal_density > max_density:
                            max_density = marginal_density
                            best_element = e

                if best_element is None:
                    break

                element_to_add = best_element

            # --- 2b. Update State for this Simulation ---
            cost_to_add = self.model.cost_of_singleton(element_to_add)
            sim_s.add(element_to_add)
            budget_rem -= cost_to_add
            candidates.remove(element_to_add)

        # --- 3. Evaluate and update global best ---
        reward = self.g(list(sim_s))

        if reward > self.g_s_max:
            self.s_max = list(sim_s)
            self.g_s_max = reward

        return reward

    def _backpropagate(self, node, reward):
        current_node = node
        while current_node is not None:
            current_node.visits += 1
            current_node.value += reward
            current_node = current_node.parent

    def _best_child_ucb(self, node):
        best_score = -float('inf')
        best_child = None
        for child in node.children:
            if child.visits == 0:
                return child
            exploit_term = child.value / child.visits
            explore_term = self.exploration_constant * math.sqrt(math.log(node.visits) / child.visits)
            score = exploit_term + explore_term
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _get_best_solution_path(self):
        current_node = self.root
        while not current_node.is_on_the_edge():
            if not current_node.children:
                break
            current_node = max(current_node.children,
                               key=lambda c: (c.visits, c.value / c.visits if c.visits > 0 else 0))
        solution_set, _, _ = current_node.state
        return list(solution_set)

    def optimize(self):
        self.start_time = time.time()
        self.next_report_time = self.report_interval
        self.terminated = False
        iteration_count = 0

        # --- Reporting Logic Change ---
        while True:
            elapsed = time.time() - self.start_time
            if elapsed > self.running_time:
                self.terminated = True
                break

            # --- Reporting Logic Change ---
            # At each report interval, record the best function value found so far.
            while elapsed >= self.next_report_time:
                # self.g_s_max is continuously updated by the _simulate method
                self.af_plot.append(float(self.g_s_max))
                self.next_report_time += self.report_interval

            # --- MCTS Core Loop ---
            leaf_node = self._select(self.root)

            child_to_simulate = leaf_node
            if not leaf_node.is_fully_expanded() and not leaf_node.is_on_the_edge():
                child_to_simulate = self._expand(leaf_node)

            reward = self._greedy_simulate(child_to_simulate)  # _simulate updates self.g_s_max
            self._backpropagate(child_to_simulate, reward)
            iteration_count += 1

        stop_time = time.time()

        # --- Final Solution Extraction ---
        sol = self.s_max
        final_solution_value = self.g(sol)

        # --- Reporting Logic Change ---
        if not self.af_plot or self.af_plot[-1] != final_solution_value:
            self.af_plot.append(float(final_solution_value))

        # The return dictionary now reflects the changes.
        ret = {'S': sol,
               'c(S)': self.model.cost_of_set(sol),
               'f(S)': self.model.objective(sol),
               'alpha': "N/A - Reporting Function Value",  # Clearly state we're not using alpha
               'report': self.af_plot,
               'time': stop_time - self.start_time,
               'node_count': iteration_count,
               "open_list_count": "N/A for MCTS"}

        return ret


class AnytimeMCTS(OptimalAlg):
    """
    A baseline, un-augmented Anytime MCTS algorithm for submodular maximization
    with a knapsack constraint.
    """

    def __init__(self, model: BaseTask):
        super().__init__(model)
        self.start_time = 0.0
        self.running_time = 0.0
        self.report_interval = 0.0
        self.next_report_time = 0.0
        self.terminated = False

        self.af_plot = []

        self.exploration_constant = 1.414  # C in the UCB1 formula
        self.root = None
        self.s_max = []  # Best solution found so far across all simulations
        self.g_s_max = 0.0  # Value of the best solution

    def build(self):
        """ Prepares the algorithm for a new run without any specialized heuristics. """
        self.af_plot.clear()
        self.terminated = False
        MCTSNode.model = self.model

        # --- NO HEURISTIC EXPANSION ---
        # Candidates are taken in their original, arbitrary order.
        # We convert to a list to ensure a consistent processing order.
        initial_candidates = list(self.model.ground_set)

        root_state = (set(), initial_candidates, self.model.budget)
        self.root = MCTSNode(root_state)

        # Initialize s_max with an empty set.
        self.s_max = []
        self.g_s_max = self.g(self.s_max)

    def _select(self, node):
        """ Phase 1: Selection. Traverses the tree to find a node to expand. """
        current_node = node
        while not current_node.is_on_the_edge():
            if not current_node.is_fully_expanded():
                return current_node
            current_node = self._best_child_ucb(current_node)
        return current_node

    def _expand(self, node):
        """
        Phase 2: Expansion. Expands a node by creating EXACTLY ONE new child.
        This corrected version avoids creating duplicate children.
        """
        current_s, remaining_r, budget_rem = node.state
        element_to_decide = remaining_r[0]
        next_r = remaining_r[1:]
        cost_of_element = self.model.cost_of_singleton(element_to_decide)

        # Check if the "take" branch is a possibility
        take_possible = cost_of_element <= budget_rem

        take_child_exists = False
        if take_possible:
            for child in node.children:
                if len(child.state[0]) > len(current_s):
                    take_child_exists = True
                    break

        if take_possible and not take_child_exists:
            # Create the "take" child if possible and not already present
            take_s = current_s.union({element_to_decide})
            take_state = (take_s, next_r, budget_rem - cost_of_element)
            new_child = MCTSNode(take_state, parent=node)
            node.children.append(new_child)
            return new_child
        else:
            discard_s = current_s
            discard_state = (discard_s, next_r, budget_rem)
            # We must explicitly check if the discard child already exists.
            discard_child_exists = False
            for child in node.children:
                # A "discard" child is identified by having the same number of elements
                if len(child.state[0]) == len(node.state[0]):
                    discard_child_exists = True
                    break

            if not discard_child_exists:
                discard_s = node.state[0]
                new_child = MCTSNode(discard_state, parent=node)
                node.children.append(new_child)
                return new_child
            else:
                return self._best_child_ucb(node)

    def _simulate(self, node):
        """ Phase 3: Simulation. Performs a purely random rollout. """
        current_s, remaining_r, budget_rem = node.state
        sim_s = set(current_s)

        # Randomly try to pack remaining items.
        candidates_to_try = list(remaining_r)
        random.shuffle(candidates_to_try)

        for item in candidates_to_try:
            cost = self.model.cost_of_singleton(item)
            if cost <= budget_rem:
                sim_s.add(item)
                budget_rem -= cost

        reward = self.g(list(sim_s))

        # Update the best-known solution if this random one is better.
        if reward > self.g_s_max:
            self.s_max = list(sim_s)
            self.g_s_max = reward

        return reward

    def _backpropagate(self, node, reward):
        """ Phase 4: Backpropagation. Updates statistics up the tree. """
        current_node = node
        while current_node is not None:
            current_node.visits += 1
            current_node.value += reward
            current_node = current_node.parent

    def _best_child_ucb(self, node):
        """ Selects the best child of a node using the UCB1 formula. """
        best_score = -float('inf')
        best_child = None
        for child in node.children:
            if child.visits == 0:
                return child

            exploit_term = child.value / child.visits
            explore_term = self.exploration_constant * math.sqrt(math.log(node.visits) / child.visits)
            score = exploit_term + explore_term

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _get_best_solution_path(self):
        """ Extracts the solution by following the most visited path from the root. """
        current_node = self.root
        while not current_node.is_on_the_edge():
            if not current_node.children:
                break
            # Tie-breaking: choose based on visits, then by average value.
            current_node = max(current_node.children,
                               key=lambda c: (c.visits, c.value / c.visits if c.visits > 0 else 0))

        solution_set, _, _ = current_node.state
        return list(solution_set)

    def optimize(self):
        """ Main optimization loop. """
        self.start_time = time.time()
        self.next_report_time = self.report_interval
        self.terminated = False
        iteration_count = 0

        while True:
            elapsed = time.time() - self.start_time
            if elapsed > self.running_time:
                self.terminated = True
                break

            # At each report interval, record the best function value found so far.
            while self.report_interval > 0 and elapsed >= self.next_report_time:
                self.af_plot.append(float(self.g_s_max))
                self.next_report_time += self.report_interval

            # --- MCTS Core Loop ---
            node_to_process = self._select(self.root)

            child_to_simulate = node_to_process
            if not node_to_process.is_fully_expanded() and not node_to_process.is_on_the_edge():
                child_to_simulate = self._expand(node_to_process)

            reward = self._simulate(child_to_simulate)
            self._backpropagate(child_to_simulate, reward)
            iteration_count += 1

        stop_time = time.time()

        # --- Final Solution Extraction ---
        # Compare the most robust path with the best solution found in any rollout.
        sol = self.s_max

        final_solution_value = self.g(sol)

        # Add the final, best solution value to the report.
        if self.report_interval > 0:
            if not self.af_plot or self.af_plot[-1] != final_solution_value:
                self.af_plot.append(float(final_solution_value))

        ret = {'S': sol,
               'c(S)': self.model.cost_of_set(sol),
               'f(S)': self.model.objective(sol),
               'alpha': "N/A - Reporting Function Value",
               'report': self.af_plot,
               'time': stop_time - self.start_time,
               'node_count': iteration_count,
               "open_list_count": "N/A for MCTS"}

        return ret


class AnytimeBFSTC(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)
        self.max_heap = None

        self.f = self.f_with_alpha
        self.h = None

        self.heap_class = 'simple'

        self.report_mode = ''
        self.running_time = 0.0
        self.report_interval = 0.0
        self.start_time = 0.0

        self.af_report = []

    def report(self):
        if self.report_mode == 'alpha':
            self.af_report.append(self.alpha)
        elif self.report_mode == 'utility':
            self.af_report.append(self.s_m)

    def build(self):
        if self.heap_class == 'tradition':
            self.max_heap = MaxHeap()
        elif self.heap_class == 'simple':
            self.max_heap = SimpleMaxHeap()

        self.max_heap.clear()
        self.h = self.inner_h

    def greedy_add(self, s):
        base = s
        candidate = set(self.model.ground_set) - set(base)
        budget = self.model.budget

        sol = set(base)
        remaining_elements = set(candidate)
        cur_cost = self.model.cost_of_set(list(sol))

        while len(remaining_elements):
            u, max_density = None, -1.
            for e in remaining_elements:
                # e is an object
                ds = self.model.density(e, list(sol))
                if u is None or ds > max_density:
                    u, max_density = e, ds

            assert u is not None

            if cur_cost + self.model.cost_of_singleton(u) <= budget:
                # satisfy the knapsack constraint
                sol.add(u)
                cur_cost += self.model.cost_of_singleton(u)

            remaining_elements.remove(u)
            # filter out violating elements
            to_remove = set()
            for v in remaining_elements:
                if self.model.cost_of_singleton(v) + cur_cost > budget:
                    to_remove.add(v)
            remaining_elements -= to_remove

        return list(sol)

    def optimize(self):
        start_time = time.time()

        root = BaseHeapObj([], candidate=self.model.ground_set, budget=self.model.budget)
        root.v = self.f(root)

        s_max = self.greedy_add([])
        g_upper = self.h(root)
        self.max_heap.push(root)

        sol = s_max
        node_count = 0
        open_list_count = 0
        while self.max_heap.size() > 0:
            node: BaseHeapObj = self.max_heap.pop()
            node_count += 1

            if self.h(node) == 0:
                sol = node.s
                break

            g_upper = min(g_upper, self.f(node) / self.alpha)

            for i in node.candidate:
                if self.model.cost_of_singleton(i) <= node.budget:
                    s_final = self.greedy_add(set(node.s) | {i})
                    if self.g(s_max) < self.g(s_final):
                        s_max = s_final

                    if self.g(s_max) / g_upper >= self.alpha:
                        sol = s_max
                        break

                    new_node = BaseHeapObj(set(node.s) | {i}, candidate=set(node.candidate) - {i}, budget=node.budget - self.model.cost_of_singleton(i))
                    new_node.v = self.f(new_node)

                    self.max_heap.push(new_node)
                    open_list_count += 1

        stop_time = time.time()

        assert sol is not None, "No solution found."

        ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
               'time': stop_time - start_time, 'node_count': node_count, "open_list_count": open_list_count}

        return ret

class ILP(OptimalAlg):
    def __init__(self, model):
        super().__init__(model)
        self.elements = list(self.model.ground_set)
        self.n = len(self.elements)

    def optimize(self):
        start_time = time.time()

        # Setup MILP variables: y_0, y_1, ..., y_{n-1}, eta
        # Objective: Maximize eta -> Minimize -eta
        c = np.zeros(self.n + 1)
        c[self.n] = -1.0

        # Integrality: 1 for integer (binary), 0 for continuous (eta)
        integrality = np.ones(self.n + 1)
        integrality[self.n] = 0

        # Bounds: y_i in [0, 1], eta in (-inf, inf)
        lb = np.zeros(self.n + 1)
        lb[self.n] = -np.inf
        ub = np.ones(self.n + 1)
        ub[self.n] = np.inf
        bounds = Bounds(lb, ub)

        # Base constraints matrix (A) and upper bounds (b_ub)
        A = []
        b_ub = []

        # 1. Knapsack/Budget constraint: sum(cost_j * y_j) <= budget
        A_budget = np.zeros(self.n + 1)
        for idx, e in enumerate(self.elements):
            A_budget[idx] = self.model.cost_of_singleton(e)
        A.append(A_budget)
        b_ub.append(self.model.budget)

        # Initialize constraint pool with an empty set
        Q = [set()]
        node_count = 0

        while True:
            # Process the newest set added to Q and build its linear constraint
            S_latest = Q[-1]
            z_S = self.model.objective(list(S_latest))

            # Constraint: eta - sum_{j in N \ S} rho_j(S) * y_j <= z(S)
            A_row = np.zeros(self.n + 1)
            A_row[self.n] = 1.0  # for eta

            for idx, e in enumerate(self.elements):
                if e not in S_latest:
                    S_union_e = list(S_latest) + [e]
                    rho = self.model.objective(S_union_e) - z_S
                    A_row[idx] = -rho  # Move to LHS: -rho_j

            A.append(A_row)
            b_ub.append(z_S)

            # Solve the relaxed MILP
            constraints = LinearConstraint(A, ub=b_ub)
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            node_count += 1

            if not res.success:
                raise ValueError(f"MILP solver failed: {res.message}")

            # Extract results
            eta_p = -res.fun
            y = res.x[:self.n]

            # Map binary vector back to a subset R^p
            R_p = set(self.elements[i] for i in range(self.n) if y[i] > 0.5)
            z_R_p = self.model.objective(list(R_p))

            # Termination check: if the upper bound matches the actual value
            if self.alpha * eta_p <= z_R_p + 1e-6:  # 1e-6 tolerance for floating-point inaccuracies
                sol = list(R_p)
                break

            # Otherwise, add R^p to the pool and generate a new constraint next iteration
            Q.append(R_p)

        stop_time = time.time()

        ret = {
            'S': sol,
            'c(S)': self.model.cost_of_set(sol),
            'f(S)': self.model.objective(sol),
            'time': stop_time - start_time,
            'node_count': node_count,
            'open_list_count': len(Q),
            'push_back_count': 0
        }

        return ret


class ILP(OptimalAlg):
    def __init__(self, model):
        super().__init__(model)
        self.elements = list(self.model.ground_set)
        self.n = len(self.elements)

        self.start_time = 0
        self.time_limit = 10000
        self.TLE = False

    def optimize(self):
        self.start_time = time.time()

        # Setup MILP variables: y_0, y_1, ..., y_{n-1}, eta
        # Objective: Maximize eta -> Minimize -eta
        c = np.zeros(self.n + 1)
        c[self.n] = -1.0

        # Integrality: 1 for integer (binary), 0 for continuous (eta)
        integrality = np.ones(self.n + 1)
        integrality[self.n] = 0

        # Bounds: y_i in [0, 1], eta in (-inf, inf)
        lb = np.zeros(self.n + 1)
        lb[self.n] = -np.inf
        ub = np.ones(self.n + 1)
        ub[self.n] = np.inf
        bounds = Bounds(lb, ub)

        # Base constraints matrix (A) and upper bounds (b_ub)
        A = []
        b_ub = []

        # 1. Knapsack/Budget constraint: sum(cost_j * y_j) <= budget
        A_budget = np.zeros(self.n + 1)
        for idx, e in enumerate(self.elements):
            A_budget[idx] = self.model.cost_of_singleton(e)
        A.append(A_budget)
        b_ub.append(self.model.budget)

        # Initialize constraint pool with an empty set
        Q = [set()]
        node_count = 0

        sol = []
        while True:
            if time.time() - self.start_time > self.time_limit:
                self.TLE = True
                break

            # Process the newest set added to Q and build its linear constraint
            S_latest = Q[-1]
            z_S = self.model.objective(list(S_latest))

            # Constraint: eta - sum_{j in N \ S} rho_j(S) * y_j <= z(S)
            A_row = np.zeros(self.n + 1)
            A_row[self.n] = 1.0  # for eta

            for idx, e in enumerate(self.elements):
                if e not in S_latest:
                    S_union_e = list(S_latest) + [e]
                    rho = self.model.objective(S_union_e) - z_S
                    A_row[idx] = -rho  # Move to LHS: -rho_j

            A.append(A_row)
            b_ub.append(z_S)

            # Solve the relaxed MILP
            constraints = LinearConstraint(A, ub=b_ub)
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            node_count += 1

            if not res.success:
                raise ValueError(f"MILP solver failed: {res.message}")

            # Extract results
            eta_p = -res.fun
            y = res.x[:self.n]

            # Map binary vector back to a subset R^p
            R_p = set(self.elements[i] for i in range(self.n) if y[i] > 0.5)
            z_R_p = self.model.objective(list(R_p))

            # Termination check: if the upper bound matches the actual value
            if self.alpha * eta_p <= z_R_p + 1e-6:  # 1e-6 tolerance for floating-point inaccuracies
                sol = list(R_p)
                break

            # Otherwise, add R^p to the pool and generate a new constraint next iteration
            Q.append(R_p)

        stop_time = time.time()

        ret = {
            'S': sol,
            'c(S)': self.model.cost_of_set(sol),
            'f(S)': self.model.objective(sol),
            'time': stop_time - self.start_time,
            'TLE': self.TLE,
            'node_count': node_count,
            'open_list_count': len(Q),
            'push_back_count': 0
        }

        return ret