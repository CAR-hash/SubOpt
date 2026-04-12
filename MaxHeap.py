import heapq

class BaseHeapObj(object):
    def __init__(self, s, v=None, candidate=None, budget=None):
        self.s = s
        self.candidate = candidate
        self.v = v
        self.budget = budget

    def __lt__(self, other):
        return self.v > other.v

    def __eq__(self, other):
        return self.v == other.v

    def __str__(self):
        return f"{self.s}, {self.v}"


class HeapObj(BaseHeapObj):
    def __init__(self, s, v=None, cost=None, candidate=None, w=None, max_idx=0, visited=False):
        super().__init__(s, v, candidate, w)
        self.cost = cost
        self.max_idx = max_idx
        self.visited = visited

    def __lt__(self, other):
        return self.v > other.v

    def __eq__(self, other):
        return self.v == other.v

    def __str__(self):
        return f"{self.s}, {self.v}"


class EfficientBFSHeapObj(HeapObj):
    def __init__(self, s, v=None, cost=None, candidate=None, w=0, first_child=False, heuristic_sequence=None, max_idx=0,
                 visited=False, depth=0, forbidden_sets=None):
        super().__init__(s, v, cost, candidate=candidate, w=w, max_idx=max_idx, visited=visited)

        self.first_child = first_child
        self.heuristic_sequence = heuristic_sequence
        self.depth = depth
        self.forbidden_sets = forbidden_sets if forbidden_sets is not None else []

class BranchAndBoundNode(HeapObj):
    # s: current set
    # c: candidate set
    # w: remaining budget
    def __init__(self, s, c, w):
        super().__init__(s=s, candidate=c, w=w)


class MaxHeap(object):
    def __init__(self):
        self.h = []

    def push(self, v):
        heapq.heappush(self.h, v)

    def pop(self):
        return heapq.heappop(self.h)

    def remove(self, ele):
        try:
            self.h.remove(ele)
            heapq.heapify(self.h)
        except ValueError:
            print(f"Element {ele} not found in heap.")

    def top(self):
        return self.h[0]

    def __getitem__(self, item):
        assert item < len(self.h)
        return self.h[item]

    def clear(self):
        self.h = []

    def size(self):
        return len(self.h)


class ComplexMaxHeap(object):
    def __init__(self):
        self.elements = []
        self.h = []

        self.current_size = 0

    @staticmethod
    def find_parent(idx):
        return (idx-1)//2

    def push(self, node):
        ele_idx = len(self.elements)
        self.elements.append(node)
        self.h.append(ele_idx)

        idx_idx = self.current_size
        while idx_idx > 0:
            parent_idx_idx = self.find_parent(idx_idx)
            if self.elements[self.h[parent_idx_idx]].v < self.elements[self.h[idx_idx]].v:
                self.h[parent_idx_idx], self.h[idx_idx] = self.h[idx_idx], self.h[parent_idx_idx]
                idx_idx = parent_idx_idx
            else:
                break

        self.current_size = self.current_size + 1

    def pop(self):
        assert len(self.h) > 0, "Heap is empty."
        idx_idx = self.current_size - 1
        ele = self.elements[self.h[0]]
        self.h[0] = self.h[idx_idx]

        self.h.pop(idx_idx)
        self.current_size = self.current_size - 1

        idx_idx = 0
        while 2 * idx_idx + 1 < self.current_size:
            large_child = idx_idx * 2 + 1
            small_child = idx_idx * 2 + 2

            if large_child < self.current_size and small_child < self.current_size and self.elements[self.h[large_child]].v < self.elements[self.h[small_child]].v:
                large_child += 1

            if self.elements[self.h[large_child]].v <= self.elements[self.h[idx_idx]].v:
                break

            self.h[large_child], self.h[idx_idx] = self.h[idx_idx], self.h[large_child]

            idx_idx = large_child

        return ele

    def remove(self, ele):
        pass

    def top(self):
        assert len(self.h) > 0, "Heap is empty."

        return self.elements[self.h[0]]

    def __getitem__(self, item):
        assert item < len(self.h)
        return self.h[item]

    def clear(self):
        self.h.clear()
        self.elements.clear()
        self.current_size = 0

    def size(self):
        return self.current_size


class SimpleMaxHeap(object):
    def __init__(self):
        self.h = []

    @staticmethod
    def find_parent(idx):
        return (idx-1)//2

    def push(self, node):
        self.h.append(node)
        idx = len(self.h) - 1

        while idx > 0:
            parent_idx = self.find_parent(idx)
            if self.h[parent_idx].v < self.h[idx].v:
                self.h[parent_idx], self.h[idx] = self.h[idx], self.h[parent_idx]
                idx = parent_idx
            else:
                break

    def pop(self):
        assert len(self.h) > 0, "Heap is empty."
        idx = len(self.h) - 1

        ele = self.h[0]
        self.h[0] = self.h[idx]
        self.h.pop(idx)

        current_size = len(self.h)

        idx = 0
        while 2 * idx + 1 < current_size:
            large_child = idx * 2 + 1
            small_child = idx * 2 + 2

            if small_child < current_size and self.h[large_child].v < self.h[small_child].v:
                large_child += 1

            if self.h[large_child].v <= self.h[idx].v:
                break

            self.h[large_child], self.h[idx] = self.h[idx], self.h[large_child]

            idx = large_child

        return ele

    def remove(self, ele):
        pass

    def top(self):
        assert len(self.h) > 0, "Heap is empty."
        return self.h[0]

    def __getitem__(self, item):
        assert item < len(self.h)
        return self.h[item]

    def clear(self):
        self.h.clear()

    def size(self):
        return len(self.h)
