# -*- coding: utf-8 -*-
"""打印最小 GPT 各模块参数量，用于与手算结果逐项核对（报告表 3 与附录）。"""
import os

import torch

os.environ.setdefault("LAB03_THREADS", "2")
import min_llm as ml

torch.set_num_threads(2)

text = ml.build_corpus(None)
tok = ml.CharTokenizer(text)
m = ml.MiniGPT(tok.vocab_size, 128, 4, 2, 128, use_pos=True)

total = sum(p.numel() for p in m.parameters())
print(f"语料 {len(text)} 字符 | 词表 V = {tok.vocab_size}")
print(f"sum(p.numel() for p in model.parameters()) = {total:,}")
print()
print("按模块拆解：")
for name, mod in m.named_children():
    if isinstance(mod, torch.nn.Embedding):
        print(f"  {name:<8} {sum(p.numel() for p in mod.parameters()):>9,}  "
              f"shape={tuple(mod.weight.shape)}")
    elif isinstance(mod, torch.nn.ModuleList):
        for i, blk in enumerate(mod):
            print(f"  blocks[{i}] {sum(p.numel() for p in blk.parameters()):>9,}")
            for n2, m2 in blk.named_children():
                if isinstance(m2, torch.nn.Sequential):
                    print(f"      {n2:<6} {sum(p.numel() for p in m2.parameters()):>9,}")
                else:
                    print(f"      {n2:<6} {sum(p.numel() for p in m2.parameters()):>9,}")
    else:
        print(f"  {name:<8} {sum(p.numel() for p in mod.parameters()):>9,}")

print()
print("注：head.weight 与 tok_emb.weight 是同一个张量（权重绑定），逐模块相加得 592,128；")
print("    model.parameters() 自动去重，所以真实参数量是 502,656。")

# 手算三项与实际的差值来源
V, C, T, L = tok.vocab_size, 128, 128, 2
hand = V * C + T * C + L * 12 * C * C
ln = 5 * 2 * C                      # 4 个块内 LayerNorm + 1 个 ln_f，每个 (gamma, beta) 共 2C
bias_attn = L * (3 * C + C)         # qkv 偏置 3C + 输出投影偏置 C
bias_mlp = L * (4 * C + C)          # MLP 第一层偏置 4C + 第二层偏置 C
print()
print(f"手算三项：{V}×{C} + {T}×{C} + {L}×12×{C}² = {V*C:,} + {T*C:,} + {L*12*C*C:,} = {hand:,}")
print(f"代码打印：{total:,}   差值 {total-hand:,}（{(total-hand)/total*100:.2f}%）")
print(f"差值来源：LayerNorm {ln:,} + 注意力偏置 {bias_attn:,} + MLP 偏置 {bias_mlp:,} "
      f"= {ln+bias_attn+bias_mlp:,}")
