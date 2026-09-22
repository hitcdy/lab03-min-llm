# -*- coding: utf-8 -*-
"""
手搓最小 LLM —— 使用 CPU 训练（实验作业三）
对应教材第 5 章《Transformer 模型》：字符级分词 + 最小 GPT + 自回归采样。

单文件、零依赖外部数据：
  python min_llm.py                      # 基线：2000 步
  python min_llm.py --iters 800          # 快速跑通
  python min_llm.py --n_layer 1 --n_embd 64
  python min_llm.py --no_pos             # 去掉位置编码（exp5）
  python min_llm.py --temperature 1.5    # 采样温度（exp6）

CPU 线程默认限制为 4（给系统留余量）：LAB03_THREADS=4 python min_llm.py ...
"""
import argparse
import json
import math
import os
import random
import time

# 限制 CPU 线程，避免训练占满整机（可用环境变量 LAB03_THREADS 调整）
THREADS = int(os.environ.get("LAB03_THREADS", "4"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, str(THREADS))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(THREADS)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE_DIR, "results_lab03.json")


# ============ 1. 语料：内置唐诗（corpus_poems.txt），可追加自备语料 ============
def build_corpus(extra_file=None):
    """拼接语料；同目录下的 corpus_extra.txt（自备语料）会自动追加。"""
    path = os.path.join(BASE_DIR, "corpus_poems.txt")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if extra_file:
        p = extra_file if os.path.isabs(extra_file) else os.path.join(BASE_DIR, extra_file)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                text = text + "\n" + f.read()
            print(f"已追加自备语料 {p}")
    return text


# ============ 2. 字符级分词器 ============
class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


# ============ 3. 最小 GPT ============
class CausalSelfAttention(nn.Module):
    """带因果掩码的多头自注意力（教材 5.4 节）。张量形状：(B, T, C)。"""

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)      # Q/K/V 一次算出
        self.proj = nn.Linear(n_embd, n_embd)          # 输出投影 W_O
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)：先拆头再转置
        q = q.view(B, T, self.n_head, -1).transpose(1, 2)
        k = k.view(B, T, self.n_head, -1).transpose(1, 2)
        v = v.view(B, T, self.n_head, -1).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))     # 缩放点积
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))   # 因果掩码
        att = F.softmax(att, dim=-1)
        y = att @ v                                                  # 加权求和
        y = y.transpose(1, 2).contiguous().view(B, T, C)              # 拼接多头
        return self.proj(y)


class Block(nn.Module):
    """Pre-Norm Transformer 块：x + Attn(LN(x))，x + MLP(LN(x))。"""

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=2,
                 block_size=128, use_pos=True):
        super().__init__()
        self.block_size = block_size
        self.use_pos = use_pos
        self.tok_emb = nn.Embedding(vocab_size, n_embd)                     # 词嵌入
        self.pos_emb = nn.Embedding(block_size, n_embd) if use_pos else None  # 可学习位置编码
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight        # 权重共享，只计一次参数
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """小方差初始化：权重 N(0, 0.02)，偏置清零 → 初始 loss ≈ ln V。"""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)                                  # (B, T, C)
        if self.pos_emb is not None:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)                          # 词向量 + 位置向量
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=1.0, top_k=None,
                 no_repeat_ngram=0):
        """自回归生成：每步预测下一个字符并拼接回输入。"""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]        # 截断到上下文窗口
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, -1:]
                logits[logits < kth] = float("-inf")     # 只保留 top-k
            if no_repeat_ngram and idx.size(1) >= no_repeat_ngram:
                # 进阶任务(b)：禁止出现过的 n-gram，抑制“循环背诵”
                prefix = tuple(idx[0, -(no_repeat_ngram - 1):].tolist())
                banned = set()
                seq = idx[0].tolist()
                for i in range(len(seq) - no_repeat_ngram + 1):
                    if tuple(seq[i:i + no_repeat_ngram - 1]) == prefix:
                        banned.add(seq[i + no_repeat_ngram - 1])
                if banned:
                    logits[:, list(banned)] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)        # 按概率采样
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ============ 4. 数据批：随机截取 (block_size+1) 的片段 ============
def get_batch(data, block_size, batch_size, device="cpu"):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])   # 标签错开一位
    return x.to(device), y.to(device)


# ============ 5. 训练 / 绘图 / 采样指标 ============
def train(model, data, iters, block_size, batch_size, lr, log_every=100):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses, t0 = [], time.time()
    for step in range(1, iters + 1):
        xb, yb = get_batch(data, block_size, batch_size)
        _, loss = model(xb, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if log_every and (step % log_every == 0 or step == 1):
            speed = step / (time.time() - t0)
            print(f"  step {step:5d}/{iters} | loss {loss.item():.4f} | "
                  f"{speed:.2f} it/s | 预计剩余 {(iters - step) / speed:.0f}s", flush=True)
    return losses, time.time() - t0


def save_curve(losses, path, title):
    plt.figure(figsize=(7, 4))
    plt.plot(losses, linewidth=0.8, color="#1f77b4")
    plt.xlabel("Iteration")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(title)
    plt.grid(True, ls="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def distinct_n(text, n):
    """distinct-n：不重复 n-gram 占全部 n-gram 的比例（越大越多样）。"""
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


def repeat_ratio(text, n=3):
    """重复率：非首次出现的 n-gram 占比（越大越重复）。"""
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    if not grams:
        return 0.0
    return 1 - len(set(grams)) / len(grams)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def config_tag(args):
    """曲线命名与指南一致：L2_E128_lr0.001 / no_pos；非默认上下文长度带上 T。"""
    if args.no_pos:
        return "no_pos"
    t = f"_T{args.block_size}" if args.block_size != 128 else ""
    return f"L{args.n_layer}_E{args.n_embd}{t}_lr{args.lr:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--n_embd", type=int, default=128)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_layer", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=20, help="0 表示不限制（= 词表大小）")
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--no_pos", action="store_true", help="去掉位置编码（exp5）")
    ap.add_argument("--no_repeat_ngram", type=int, default=0, help="进阶：禁止重复的 n-gram")
    ap.add_argument("--extra_corpus", type=str, default="corpus_extra.txt",
                    help="自备补充语料文件（可选，放在同目录）")
    ap.add_argument("--save_model", action="store_true", help="保存基线模型权重 model_<tag>.pt")
    ap.add_argument("--results", type=str, default=None, help="结果写入的 json（默认 results_lab03.json）")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"PyTorch {torch.__version__} | 设备: cpu | 线程数: {torch.get_num_threads()}")
    text = build_corpus(args.extra_corpus)
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"语料 {len(text)} 字符 | 词表大小 V = {tok.vocab_size}")

    model = MiniGPT(tok.vocab_size, args.n_embd, args.n_head, args.n_layer,
                    args.block_size, use_pos=not args.no_pos)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,} | 配置: n_layer={args.n_layer} n_head={args.n_head} "
          f"n_embd={args.n_embd} block={args.block_size} "
          f"pos={'有' if not args.no_pos else '无'}")

    losses, train_time = train(model, data, args.iters, args.block_size,
                               args.batch_size, args.lr)
    tag = config_tag(args)
    init_loss, final_loss = losses[0], losses[-1]
    print(f"训练完成：耗时 {train_time / 60:.1f} 分钟 | 初始 loss {init_loss:.4f} "
          f"| 最终 loss {final_loss:.4f}（前 100 步均值 {sum(losses[:100]) / 100:.4f}）")

    curve = os.path.join(BASE_DIR, f"loss_curve_{tag}.png")
    save_curve(losses, curve, f"Training Loss (n_layer={args.n_layer}, n_embd={args.n_embd}, "
                              f"lr={args.lr}, pos={'off' if args.no_pos else 'on'})")
    print(f"已保存 {os.path.basename(curve)}")

    if args.save_model:
        mp = os.path.join(BASE_DIR, f"model_{tag}.pt")
        torch.save({"state_dict": model.state_dict(),
                    "config": dict(vocab_size=tok.vocab_size, n_embd=args.n_embd,
                                   n_head=args.n_head, n_layer=args.n_layer,
                                   block_size=args.block_size, use_pos=not args.no_pos),
                    "stoi": tok.stoi}, mp)
        print(f"已保存模型权重 {os.path.basename(mp)}")

    # 采样：提示词「春」「月」各 2 段
    samples = {}
    top_k = None if args.top_k in (0, -1) else args.top_k
    for prompt in ["春", "月"]:
        outs = []
        for _ in range(2):
            ids = model.generate(torch.tensor([tok.encode(prompt)]),
                                 args.max_new_tokens, args.temperature, top_k,
                                 args.no_repeat_ngram)
            outs.append(tok.decode(ids[0].tolist()))
        samples[prompt] = outs
        print(f"生成示例（temperature={args.temperature}, top_k={args.top_k}, 「{prompt}」）：")
        for s in outs:
            print("   " + s.replace("\n", " "))

    gen_path = os.path.join(BASE_DIR, f"generated_{tag}.txt")
    with open(gen_path, "w", encoding="utf-8") as f:
        for prompt, outs in samples.items():
            f.write(f"# 提示词「{prompt}」 temperature={args.temperature} "
                    f"top_k={args.top_k} no_repeat_ngram={args.no_repeat_ngram}\n")
            for s in outs:
                f.write(s + "\n")
    print(f"已保存 {os.path.basename(gen_path)}")

    rec = dict(tag=tag, iters=args.iters, batch_size=args.batch_size,
               block_size=args.block_size, n_embd=args.n_embd, n_head=args.n_head,
               n_layer=args.n_layer, lr=args.lr, seed=args.seed, use_pos=not args.no_pos,
               no_repeat_ngram=args.no_repeat_ngram,
               vocab=tok.vocab_size, n_chars=len(text), params=n_params,
               init_loss=round(init_loss, 4), final_loss=round(final_loss, 4),
               loss_first100=round(sum(losses[:100]) / 100, 4),
               loss_head=round(sum(losses[:200]) / len(losses[:200]), 4),
               time_min=round(train_time / 60, 2), curve=os.path.basename(curve),
               samples=samples, temperature=args.temperature, top_k=args.top_k)
    path = args.results or RESULTS
    allrec = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            allrec = json.load(f)
    allrec[tag] = rec
    save_json(path, allrec)
    print(f"已写入 {os.path.basename(path)} :: {tag}")


if __name__ == "__main__":
    main()
