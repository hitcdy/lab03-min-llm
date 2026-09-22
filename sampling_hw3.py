# -*- coding: utf-8 -*-
"""
实验作业三 · 任务五：采样实验（固定基线模型，只改采样参数）。
- 表 5 的 5 组：temperature ∈ {1.0, 0.5, 1.5} × top_k=20，以及 top_k ∈ {5, 不限制}
- 每组用提示词「春」「月」各生成 2 段，统计多样性与重复度：
    distinct-1 / distinct-2（越大越多样）、重复 3-gram 比例（越大越重复）
- 进阶任务(b)：禁止重复 3-gram 采样，对比前后效果
输出：sampling_hw3.json、result_sampling.png（图 7）、result_norepeat.png（图 8）、
      generated_sampling.txt（生成原文，供报告表 5 摘录）
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import min_llm as ml

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(BASE, "model_L2_E128_lr0.001.pt")
OUT_JSON = os.path.join(BASE, "sampling_hw3.json")
PROMPTS = ["春", "月"]
N_SAMPLE = 2
MAX_NEW = 120

# (组号, temperature, top_k（None = 不限制）, 说明)
SETTINGS = [
    (1, 1.0, 20, "基线采样"),
    (2, 0.5, 20, "低温：更确定"),
    (3, 1.5, 20, "高温：更多样"),
    (4, 1.0, 5, "极窄候选"),
    (5, 1.0, None, "不限制候选"),
]


def load_model(path=MODEL):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = ml.MiniGPT(cfg["vocab_size"], cfg["n_embd"], cfg["n_head"],
                       cfg["n_layer"], cfg["block_size"], use_pos=cfg["use_pos"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tok = ml.CharTokenizer("".join(ckpt["stoi"].keys()))
    return model, tok, cfg


def gen(model, tok, prompt, temperature, top_k, no_repeat_ngram=0):
    ids = torch.tensor([tok.encode(prompt)])
    out = model.generate(ids, MAX_NEW, temperature, top_k, no_repeat_ngram)
    return tok.decode(out[0].tolist())


def metrics(texts):
    joined = "\n".join(texts)
    return dict(distinct1=round(ml.distinct_n(joined, 1), 3),
                distinct2=round(ml.distinct_n(joined, 2), 3),
                repeat3=round(ml.repeat_ratio(joined, 3), 3),
                n_grams=len(joined))


def main():
    model, tok, cfg = load_model()
    print(f"载入基线模型：{cfg} | 线程数 {torch.get_num_threads()}")

    records, lines = [], []
    for gid, temp, top_k, note in SETTINGS:
        texts = []
        for p in PROMPTS:
            for _ in range(N_SAMPLE):
                texts.append(gen(model, tok, p, temp, top_k))
        m = metrics(texts)
        records.append(dict(group=gid, temperature=temp,
                            top_k=("不限制" if top_k is None else top_k),
                            note=note, texts=texts, **m))
        print(f"[采样 {gid}] T={temp} top_k={top_k} → distinct-1 {m['distinct1']} "
              f"distinct-2 {m['distinct2']} 重复3-gram {m['repeat3']}", flush=True)
        lines.append(f"# 组{gid}  temperature={temp}  top_k={top_k}  {note}")
        for p, t in zip([x for x in PROMPTS for _ in range(N_SAMPLE)], texts):
            lines.append(f"「{p}」{t}")
        lines.append("")

    # 进阶任务(b)：禁止重复 3-gram
    nr_texts = []
    for p in PROMPTS:
        for _ in range(N_SAMPLE):
            nr_texts.append(gen(model, tok, p, 1.0, 20, no_repeat_ngram=3))
    nr = dict(temperature=1.0, top_k=20, no_repeat_ngram=3, texts=nr_texts, **metrics(nr_texts))
    print(f"[进阶] 禁止重复 3-gram → distinct-1 {nr['distinct1']} distinct-2 {nr['distinct2']} "
          f"重复3-gram {nr['repeat3']}", flush=True)
    lines.append("# 进阶：temperature=1.0 top_k=20 no_repeat_ngram=3")
    for p, t in zip([x for x in PROMPTS for _ in range(N_SAMPLE)], nr_texts):
        lines.append(f"「{p}」{t}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dict(settings=records, norepeat=nr), f, ensure_ascii=False, indent=2)
    with open(os.path.join(BASE, "generated_sampling.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("已保存 sampling_hw3.json 与 generated_sampling.txt")

    # 图 7：多样性与重复度对比
    labels = [f"T={r['temperature']}\ntop_k={r['top_k']}" for r in records]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    x = range(len(records))
    a1.bar([i - 0.2 for i in x], [r["distinct1"] for r in records], width=0.4,
           label="distinct-1", color="#1f77b4")
    a1.bar([i + 0.2 for i in x], [r["distinct2"] for r in records], width=0.4,
           label="distinct-2", color="#ff7f0e")
    a1.set_xticks(list(x)); a1.set_xticklabels(labels, fontsize=8)
    a1.set_ylabel("不重复 n-gram 比例"); a1.set_title("多样性：越大越不重复")
    a1.legend(fontsize=8); a1.grid(True, ls="--", alpha=0.3, axis="y")
    a2.bar(list(x), [r["repeat3"] for r in records], color="#d62728")
    a2.axhline(nr["repeat3"], ls="--", color="#2ca02c",
               label=f"禁止重复 3-gram（{nr['repeat3']}）")
    a2.set_xticks(list(x)); a2.set_xticklabels(labels, fontsize=8)
    a2.set_ylabel("重复 3-gram 比例"); a2.set_title("重复度：越小越不循环")
    a2.legend(fontsize=8); a2.grid(True, ls="--", alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, "result_sampling.png"), dpi=150)
    plt.close(fig)
    print("已保存 result_sampling.png")

    # 图 8：进阶（重复惩罚）前后对比
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    vals = [records[0]["repeat3"], nr["repeat3"]]
    ax.bar(["基线采样", "禁止重复 3-gram"], vals, color=["#d62728", "#2ca02c"])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("重复 3-gram 比例"); ax.set_title("进阶：重复惩罚的效果")
    ax.grid(True, ls="--", alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, "result_norepeat.png"), dpi=150)
    plt.close(fig)
    print("已保存 result_norepeat.png")


if __name__ == "__main__":
    main()
