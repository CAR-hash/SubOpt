import argparse
import os
import pickle
import random

import numpy as np

import efficient_bfs
import model_factory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Run Degenerated Algorithms")
    parser.add_argument("-t", "--task", default='sensor')
    parser.add_argument("-n", "--num", type=int, default=100)
    parser.add_argument("-a", "--archive", default="27")
    parser.add_argument("-aa", "--alpha", type=float, default=0.8)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--stop_seed", type=int, default=1)
    args = parser.parse_args()

    # --- 定义实验对照组 ---
    # 每个配置包含：名称标签, 分支策略, 上界类型, 是否开启继承
    experiments = [
        {"tag": "Standard", "bs": "traditional", "ub": "ub2", "inh": True},
        # {"tag": "NoInherit", "bs": "traditional", "ub": "ub2", "inh": False},
        # {"tag": "PlainUB", "bs": "traditional", "ub": "ub0", "inh": True},
        # {"tag": "NaiveBS", "bs": "naive", "ub": "ub2", "inh": True},
    ]

    # 设置 Budget 范围
    bds = np.linspace(start=12, stop=15, num=4)
    root_dir = os.path.join("./result", f"archive-{args.archive}")

    for seed in range(args.start_seed, args.stop_seed):
        for budget in bds:
            print(f"\n🚀 Seed: {seed} | Budget: {budget:.1f}")

            for exp in experiments:
                # 1. 严格重置随机环境
                random.seed(seed)

                # 2. 实例化模型与算法
                model = model_factory.model_factory(args.task, args.num, seed, budget, knap=True)
                alg = efficient_bfs.EfficientBFS(model)

                # 3. 注入对照组参数
                alg.branching_strategy = exp["bs"]
                alg.ub_type = exp["ub"]  # 对应你修改后的 ub_type
                alg.inherit_bounds = exp["inh"]  # 对应你修改后的继承开关

                # 4. 其他常规配置
                alg.alpha = args.alpha
                alg.set_h(heuristic=exp["ub"])  # 同步修改启发式函数
                alg.setOpt(exp["ub"])
                alg.set_d('d')

                # 5. 执行
                alg.build()
                res = alg.optimize()

                # 6. 打印对比摘要
                node_cnt = res.get('node_count', -1)
                time_cost = res.get('time', 0.0)
                print(
                    f"  [{exp['tag']:<10}] Nodes: {node_cnt:<6} | Time: {time_cost:.2f}s | f(S): {res.get('f(S)', 0):.2f}")

                # 7. 存储结果（文件名包含对照组标签）
                save_dir = os.path.join(root_dir, args.task, str(args.num), str(seed))
                os.makedirs(save_dir, exist_ok=True)

                filename = f"Exp-{exp['tag']}-{exp['ub']}-{budget}-{args.alpha}.pckl"
                with open(os.path.join(save_dir, filename), "wb") as f:
                    pickle.dump(res, f)

    print("\n✅ 所有批量实验已完成！")

