import argparse
import os
import pickle
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed  # 引入多进程库

import efficient_bfs
import model_factory


# 1. 把单次实验封装成一个函数
def run_single_experiment(task, num, seed, budget, heuristic, alpha, sorting, branching, root_dir):
    random.seed(seed)
    model = model_factory.model_factory(task, num, seed, budget, knap=True)

    alg = efficient_bfs.EfficientBFS(model)
    alg.use_alpha = True
    alg.alpha = alpha
    alg.set_d(sorting)
    alg.configure_upper_bound(heuristic)
    alg.branching_strategy = branching

    alg.build()
    res = alg.optimize()

    # 保存结果
    save_dir = os.path.join(root_dir, task, str(num), str(seed))
    os.makedirs(save_dir, exist_ok=True)
    filename = f"EfficientBFS-{branching}-{heuristic}-{sorting}-{budget}-{alpha}-{model.__class__.__name__}.pckl"
    save_path = os.path.join(save_dir, filename)
    with open(save_path, "wb") as wrt:
        pickle.dump(res, wrt)

    return f"Seed {seed}, Budget {budget:4.1f} | Nodes: {res.get('node_count', -1):<6d} | Time: {res.get('time', 0.0):.2f}s"


if __name__ == "__main__":
    # ... (此处省略 argparse 代码) ...
    parser = argparse.ArgumentParser(description="Run EfficientBFS Experiments")
    parser.add_argument("-t", "--task", default='sensor', help="task name")
    parser.add_argument("-n", "--num", type=int, default=100, help="size of the ground set")
    parser.add_argument("-a", "--archive", default="94", help="archive index")
    parser.add_argument("-hf", "--heuristic", default='ub2', help="the heuristic function")
    parser.add_argument("-aa", "--alpha", type=float, default=0.8, help="the approximation factor")
    parser.add_argument("-d", "--sorting", default='d', help="the sorting function for breaking ties")

    # 新增：明确暴露 branching_strategy 以供对比实验调用
    parser.add_argument("-bs", "--branching", default='traditional',
                        choices=['traditional', 'density_gap'],
                        help="the branching strategy to use")
    args = parser.parse_args()

    # 将 seed 也放入参数，方便以后写 bash 脚本跑多线程
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--stop_seed", type=int, default=1)

    bds = np.linspace(start=6, stop=10, num=5)
    root_dir = os.path.join("./result", f"archive-{args.archive}")

    # 构建所有需要跑的任务参数列表
    tasks = []
    for seed in range(0, 2):  # 假设跑 5 个 seed
        for budget in bds:
            tasks.append(
                (args.task, args.num, seed, budget, args.heuristic, args.alpha, args.sorting, args.branching, root_dir))

    print(f"🚀 开始并行实验，总任务数: {len(tasks)}")

    # 2. 启动多进程池 (max_workers 根据你的 CPU 核心数来定)
    max_cores = os.cpu_count() - 2  # 留两个核防止电脑卡死

    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        # 提交任务
        futures = [executor.submit(run_single_experiment, *t) for t in tasks]

        # 监控进度
        for future in as_completed(futures):
            try:
                result_msg = future.result()
                print(f"✅ 完成 -> {result_msg}")
            except Exception as e:
                print(f"❌ 任务崩溃: {e}")