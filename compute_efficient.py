import argparse
import os
import pickle
import random
import sys

import testlogger

import numpy as np

import efficient_bfs
import model_factory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EfficientBFS Experiments with Multiple Strategies")
    parser.add_argument("-t", "--task", default='sensor', help="task name")
    parser.add_argument("-n", "--num", type=int, default=100, help="size of the ground set")
    parser.add_argument("-a", "--archive", default="27", help="archive index")
    parser.add_argument("-hf", "--heuristic", default='ub2', help="the heuristic function")
    parser.add_argument("-aa", "--alpha", type=float, default=0.8, help="the approximation factor")
    parser.add_argument("-d", "--sorting", default='d', help="the sorting function for breaking ties")

    # 修改：支持传入一个列表，如果不传则默认跑完全部 4 种策略
    parser.add_argument("-bs", "--branching", nargs='+',
                        default=['traditional', 'density_gap', 'look_ahead'],
                        help="list of branching strategies to test (space-separated)")

    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--stop_seed", type=int, default=1)
    parser.add_argument("-ls", "--local-search",action='store_true',help='enable local search hybrid')
    # 增加一个开关参数
    parser.add_argument("-dive", "--use_dive", action="store_true", help="enable Dive-and-Bound (DFS-BFS Hybrid)")
    # 不使用界限继承的开关（传入该参数则关闭继承）
    parser.add_argument("-ni", "--no_inherit", action="store_true", help="disable bound inheritance mechanism")
    # 注入级联机制
    parser.add_argument("-ca", "--cascade", action="store_true", help="enable the cascade mechanism")

    args = parser.parse_args()
    assert args.heuristic in ['ub0', 'ub1', 'ub2', 'ub0+', 'ub1+', 'ub2+', 'ub4', 'dom']

    interval = 1
    num_points = 10
    start_point = 6
    end_point = start_point + (num_points - 1) * interval
    bds = np.linspace(start=start_point, stop=end_point, num=num_points)

    root_dir = os.path.join("./result", f"archive-{args.archive}")

    # 打印本次实验要跑的所有策略
    print(f"🚀 Starting Experiments | Task: {args.task} | Strategies to test: {args.branching}")

    for seed in range(args.start_seed, args.stop_seed):
        for budget in bds:

            print(f"\n--- Testing Seed: {seed:02d} | Budget: {budget:4.1f} ---")

            # 最内层循环：遍历所有指定的分支策略
            for strategy in args.branching:
                log_dir = f"./result/archive-{args.archive}/{args.task}/"
                os.makedirs(log_dir, exist_ok=True)
                sys.stdout = testlogger.TeeLogger(os.path.join(log_dir, f"{strategy}_{budget}_{args.heuristic}_log.txt"))

                # 💡 极其重要：随机数种子必须在这里重置！
                # 确保同一个 seed+budget 下，不管跑哪个策略，底层的随机生成序列完全一致
                random.seed(seed)

                # 💡 极其重要：模型和算法必须在策略循环内全新实例化，防止状态污染
                model = model_factory.model_factory(args.task, args.num, seed, budget, knap=True)
                alg = efficient_bfs.EfficientBFS(model)
                # alg = efficient_bfs.EarlyEfficientBFS(model)

                # 参数配置
                alg.use_alpha = True
                alg.local_search = args.local_search
                alg.alpha = args.alpha
                alg.set_d(args.sorting)
                alg.set_h(heuristic=args.heuristic)
                alg.setOpt(args.heuristic)
                alg.inherit_bounds = not args.no_inherit
                # ... 在实例化 alg 之后 ...
                alg.use_dive = args.use_dive
                alg.ub_type = args.heuristic

                # 同样地，把开启了下潜的策略在文件名上标记出来，防止文件覆盖
                strategy_name_for_file = strategy
                if args.local_search:
                    strategy_name_for_file += "_LS"
                if args.use_dive:
                    strategy_name_for_file += "_Dive"
                if args.no_inherit:
                    strategy_name_for_file += "_NoInh"  # 标识未开启继承

                # 注入当前循环到的分支策略
                alg.branching_strategy = strategy

                # 初始化与执行
                alg.build()
                res = alg.optimize()

                # 打印单次策略的结果
                node_cnt = res.get('node_count', -1)
                time_cost = res.get('time', 0.0)
                function_val = res.get('f(S)', 0.0)
                print(f"trigger count:{res.get('probing_trigger_count', 0)}, depth_list:{res.get('probing_trigger_depth_list', [])}, max_depth:{res.get('max_depth', 0)}")
                if not res.get('TLE', False):
                    print(f"  ✅ Strategy: {strategy_name_for_file:<14s} | f(S):{function_val} | Nodes: {node_cnt:<6d} | Time: {time_cost:.2f}s")
                else:
                    print(f"  ❌ Strategy: {strategy_name_for_file:<14s} | f(S):{function_val} | Nodes: {node_cnt:<6d} | TLE: {time_cost:.2f}s")

                # 文件保存
                save_dir = os.path.join(root_dir, args.task, str(args.num), str(seed))
                os.makedirs(save_dir, exist_ok=True)

                filename = "EfficientBFS-{}-{}-{}-{}-{}-{}.pckl".format(
                    strategy_name_for_file,
                    args.heuristic,
                    args.sorting,
                    budget,
                    args.alpha,
                    model.__class__.__name__
                )
                save_path = os.path.join(save_dir, filename)

                with open(save_path, "wb") as wrt:
                    pickle.dump(res, wrt)

    print("\n🎉 All experiments completed!")