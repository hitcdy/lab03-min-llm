# -*- coding: utf-8 -*-
"""
实验作业三：批量运行基线 + 表 5 的 6 组对照实验（严格顺序执行，CPU 只占 4 线程）。

每组只改一个变量，其余与基线一致：
  baseline  基线：2 层 / 128 维 / 4 头 / T=128 / lr=1e-3 / 2000 步（按指南任务一）
  exp1a/b   学习率     1e-2 / 1e-4
  exp2a/b   层数       1 / 4
  exp3a/b   嵌入维度   64（2 头）/ 256（8 头）
  exp4      上下文长度 32
  exp5      位置编码   去掉（--no_pos）
  exp6      采样温度   见 sampling_hw3.py（只改采样参数，不重训）

结果统一写入 results_lab03.json（可中断续跑，已完成的组自动跳过）。
用法：python run_experiments_hw3.py [--groups baseline exp1a ...] [--force]
"""
import argparse
import json
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results_lab03.json")
PY = sys.executable

# 每组：名称、用途说明、额外的命令行参数（未列出的参数保持基线值）
GROUPS = {
    "baseline": dict(dim="—", note="组0 基线：2 层 / 128 维 / 4 头 / T=128，lr=1e-3，2000 步",
                     args=["--iters", "2000", "--save_model"]),
    "exp1a": dict(dim="学习率 1e-2", note="组1a lr=1e-2（观察是否发散）",
                  args=["--iters", "1000", "--lr", "1e-2"]),
    "exp1b": dict(dim="学习率 1e-4", note="组1b lr=1e-4（收敛更慢）",
                  args=["--iters", "1000", "--lr", "1e-4"]),
    "exp2a": dict(dim="层数 1", note="组2a n_layer=1",
                  args=["--iters", "1000", "--n_layer", "1"]),
    "exp2b": dict(dim="层数 4", note="组2b n_layer=4",
                  args=["--iters", "1000", "--n_layer", "4"]),
    "exp3a": dict(dim="嵌入 64（2 头）", note="组3a n_embd=64, n_head=2",
                  args=["--iters", "1000", "--n_embd", "64", "--n_head", "2"]),
    "exp3b": dict(dim="嵌入 256（8 头）", note="组3b n_embd=256, n_head=8",
                  args=["--iters", "1000", "--n_embd", "256", "--n_head", "8"]),
    "exp4": dict(dim="上下文 32", note="组4 block_size=32",
                 args=["--iters", "1000", "--block_size", "32"]),
    "exp5": dict(dim="位置编码 去掉", note="组5 --no_pos",
                 args=["--iters", "1000", "--no_pos"]),
}


def load_results():
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as f:
            return json.load(f)
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", default=["all"])
    ap.add_argument("--force", action="store_true", help="忽略已有结果重跑")
    args = ap.parse_args()

    names = list(GROUPS) if args.groups == ["all"] else args.groups
    results = load_results()
    for tag in names:
        cfg = GROUPS[tag]
        if tag in results and not args.force:
            print(f"[{tag}] 已有结果，跳过（--force 可重跑）", flush=True)
            continue
        print(f"\n=== [{tag}] {cfg['note']} 开始（{time.strftime('%H:%M:%S')}）", flush=True)
        t0 = time.time()
        cmd = [PY, os.path.join(BASE, "min_llm.py"), "--max_new_tokens", "120"] + cfg["args"]
        r = subprocess.run(cmd, cwd=BASE)
        if r.returncode != 0:
            print(f"[{tag}] 运行失败（退出码 {r.returncode}），跳过", flush=True)
            continue
        results = load_results()
        if tag in results:
            results[tag]["dim"] = cfg["dim"]
            with open(RESULTS, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            rec = results[tag]
            print(f"[{tag}] 完成：参数量 {rec['params']:,} | 初始 loss {rec['init_loss']} "
                  f"| 最终 loss {rec['final_loss']} | 耗时 {rec['time_min']} min "
                  f"| 墙钟 {time.time() - t0:.0f}s", flush=True)

    results = load_results()
    print("\n=== 已完成组 ===")
    print(f"{'组':<9}{'维度':<18}{'参数量':>10}{'初始loss':>10}{'最终loss':>10}{'耗时(min)':>10}")
    for tag in [t for t in GROUPS if t in results]:
        r = results[tag]
        print(f"{tag:<9}{r.get('dim', r['tag']):<18}{r['params']:>10,}"
              f"{r['init_loss']:>10.4f}{r['final_loss']:>10.4f}{r['time_min']:>10.2f}")


if __name__ == "__main__":
    main()
