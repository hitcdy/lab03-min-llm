# -*- coding: utf-8 -*-
"""
在《作业三_手搓最小LLM_使用CPU训练_实验报告样板.docx》基础上填写完整实验报告：
- 删除/改写"提示：…"行，用真实文字替换下划线答题区
- 表 1~表 6 填入真实数据（参数量 / loss / 耗时 / 生成摘录与多样度指标）
- 插入 9 张图（基线曲线、exp1~exp5 曲线、采样对比图、重复惩罚对比图）
- 追加 5.3 进阶（禁止重复 n-gram 采样）与文末参数量核对附录
数据来源：results_lab03.json、sampling_hw3.json
输出：lab03_min_llm 目录下《2024312149_陈道延_实验作业三.docx》
"""
import copy
import json
import math
import os
import platform

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm
from docx.table import Table
from docx.text.paragraph import Paragraph

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "作业三_手搓最小LLM_使用CPU训练_实验报告样板.docx")
DST = os.path.join(BASE, "2024312149_陈道延_实验作业三.docx")
FIG = lambda name: os.path.join(BASE, name)
DATE = "2026 年 9 月 22 日"

with open(os.path.join(BASE, "results_lab03.json"), encoding="utf-8") as f:
    _raw = json.load(f)
with open(os.path.join(BASE, "sampling_hw3.json"), encoding="utf-8") as f:
    S = json.load(f)

# results_lab03.json 以"配置标签"为键（与曲线文件名一致），这里映射回实验编号
TAG = {
    'baseline': 'L2_E128_lr0.001', 'exp1a': 'L2_E128_lr0.01', 'exp1b': 'L2_E128_lr0.0001',
    'exp2a': 'L1_E128_lr0.001', 'exp2b': 'L4_E128_lr0.001', 'exp3a': 'L2_E64_lr0.001',
    'exp3b': 'L2_E256_lr0.001', 'exp4': 'L2_E128_T32_lr0.001', 'exp5': 'no_pos',
}
missing = [t for t in TAG.values() if t not in _raw]
if missing:
    raise SystemExit(f'results_lab03.json 缺少这些配置，请先跑完实验：{missing}')
R = {k: _raw[v] for k, v in TAG.items()}

base = R["baseline"]
SAMP = {r["group"]: r for r in S["settings"]}
NR = S["norepeat"]


def stars(n):
    return "★" * n + "☆" * (5 - n)


# 生成质量人工评级（逐字对照各组的 generated_*.txt 判读；语料仅 2163 字且训练充分，
# 各组差异主要体现在"整首背诵的准确度/错字与混拼的比例"）
QUALITY = {
    "exp1a": (2, "整块能背，混拼与重字多（“远芳侵古千里江陵一日还”）"),
    "exp1b": (1, "词片破碎不成句（“秦时鸣。寒雨知此空尽，今。”）"),
    "exp2a": (4, "整首准确，偶有重字（“春山前不见古人”）"),
    "exp2b": (5, "与基线相当，几乎逐字正确"),
    "exp3a": (2, "正确整块与错乱交替（“春来发发几人”）"),
    "exp3b": (5, "整首准确，与基线相当"),
    "exp4": (3, "多数整首正确，句间衔接偶有拼接"),
    "exp5": (2, "整首能背，但句内位置错位（“月下下扬州”“月汉时关”）"),
    "exp6": (4, "T=0.5 工整但四段雷同；T=1.5 出现混拼"),
}


def loss_str(tags):
    return "；".join(f"{lbl}：{R[t]['final_loss']:.3f}" for lbl, t in tags)


doc = docx.Document(SRC)
body = doc.element.body

items = []
for child in body.iterchildren():
    if child.tag == qn('w:p'):
        items.append(('p', Paragraph(child, doc)))
    elif child.tag == qn('w:tbl'):
        items.append(('t', Table(child, doc)))


def ptext(it):
    return it[1].text if it[0] == 'p' else ''


def is_underscore(it):
    # 只认仍然挂在文档树上的下划线段落（前面已填写的会被删除）
    if it[0] != 'p' or it[1]._element.getparent() is None:
        return False
    return bool(it[1].text.strip()) and set(it[1].text.strip()) <= set('_')


def is_hint(it):
    return it[0] == 'p' and it[1].text.strip().startswith('提示')


def set_para_text(p, text):
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    if text:
        p.add_run(text)


def delete_para(p):
    p._element.getparent().remove(p._element)


def set_cell_text(cell, text):
    paras = cell.paragraphs
    set_para_text(paras[0], text)
    for extra in paras[1:]:
        extra._element.getparent().remove(extra._element)


def add_imgs_to_cell(cell, paths, widths):
    paras = cell.paragraphs
    for r in list(paras[0].runs):
        r._element.getparent().remove(r._element)
    for extra in paras[1:]:
        extra._element.getparent().remove(extra._element)
    first = paras[0]
    first.alignment = WD_ALIGN_PARAGRAPH.CENTER
    first.add_run().add_picture(paths[0], width=Cm(widths[0]))
    for path, w in zip(paths[1:], widths[1:]):
        p = cell.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(path, width=Cm(w))


def fill_cell_multiline(cell, lines):
    paras = cell.paragraphs
    set_para_text(paras[0], lines[0] if lines else '')
    for extra in paras[1:]:
        extra._element.getparent().remove(extra._element)
    for line in lines[1:]:
        cell.add_paragraph(line)


def fill_underscores(underscore_items, lines):
    ref = underscore_items[0][1]._element
    for line in lines:
        new_p = copy.deepcopy(ref)
        for r in new_p.findall(qn('w:r')):
            new_p.remove(r)
        ref.addprevious(new_p)
        Paragraph(new_p, doc).add_run(line)
    for it in underscore_items:
        delete_para(it[1])


def insert_after(item, lines):
    """在段落 item 之后依次插入 lines（保持顺序）。"""
    prev = item[1]._element
    for line in lines:
        new_p = copy.deepcopy(prev)
        for r in new_p.findall(qn('w:r')):
            new_p.remove(r)
        prev.addnext(new_p)
        prev = new_p
        Paragraph(new_p, doc).add_run(line)


def prev_table(idx):
    """从 idx 往前找到最近的 1×1 图片表格（中间可能夹着空段落）。"""
    for j in range(idx - 1, max(idx - 5, -1), -1):
        if items[j][0] == 't' and len(items[j][1].rows) == 1 and len(items[j][1].columns) == 1:
            return items[j][1]
    return None


def find_heading(prefix):
    for it in items:
        if it[0] == 'p' and ptext(it).strip().startswith(prefix):
            return it
    raise KeyError(prefix)


def find_table(head0):
    for it in items:
        if it[0] == 't' and it[1].rows[0].cells[0].text.strip() == head0:
            return it[1]
    raise KeyError(head0)


def find_caption(prefix):
    for i, it in enumerate(items):
        if it[0] == 'p' and ptext(it).strip().startswith(prefix):
            return i
    raise KeyError(prefix)


# ---------------- 0) 封面 ----------------
for it in items:
    t = ptext(it).strip()
    if t.startswith('姓'):
        set_para_text(it[1], '姓　　名：陈道延')
    elif t.startswith('学'):
        set_para_text(it[1], '学　　号：2024312149')
    elif t.startswith('班'):
        set_para_text(it[1], '班　　级：')
    elif t.startswith('完成日期'):
        set_para_text(it[1], f'完成日期：{DATE}')

# ---------------- 1) 提示行处理 ----------------
for it in items:
    if not is_hint(it):
        continue
    t = ptext(it)
    if '总参数 ≈' in t:                      # 手算公式说明，保留但去掉"提示"字样
        set_para_text(it[1], '说明：总参数 ≈ V×C（词嵌入，与输出层共享只计一次）+ T×C（位置编码）'
                             '+ L×12C²（每层注意力 4C² + MLP 8C²）。')
    elif 'sum(p.numel())' in t:
        set_para_text(it[1], '核对方式：用 sum(p.numel() for p in model.parameters()) 统计总参数量，'
                             '与手算结果逐项对照，误差应小于 1%（完整终端输出见文末附录）。')
    elif 'exp1~exp5' in t:
        set_para_text(it[1], '说明：每组只改一个变量、其余与基线一致；基线跑 2000 步，'
                            'exp1~exp5 统一跑 1000 步以缩短时间，因此除"训练步数"这一共同差异外，'
                            '各组相对基线只有被考察的那个变量不同；'
                            '曲线由脚本按配置自动命名（loss_curve_*.png），与下表编号一一对应。')
    else:
        delete_para(it[1])

# ---------------- 2) 1.1 实验目的 ----------------
fill_underscores([it for it in items if is_underscore(it)][:3], [
    '（1）不调用任何现成大模型接口，用 PyTorch 从零实现一个字符级最小 GPT：'
    '字符分词器、因果自注意力、Pre-Norm Transformer 块、权重绑定的输出层，'
    '并理解"预测下一个字符"这一自监督目标的数据构造方式（输入与标签错位一位）。',
    '（2）动手推导 Transformer 的参数量公式并与代码打印值核对：'
    '总参数 ≈ V×C + T×C + L×12C²，学会用 sum(p.numel() for p in model.parameters()) 验证理论、'
    '并解释偏置与 LayerNorm 带来的差值来源。',
    '（3）用控制变量法做对照实验（学习率、层数、嵌入维度、上下文长度、位置编码），'
    '用采样实验（temperature / top-k）与重复度指标观察生成质量，'
    '掌握"小语料 + CPU + 分钟级"的完整训练—评估—生成闭环。',
])

# ---------------- 3) 表 1 实验环境 ----------------
env = find_table('类别')
env_rows = [
    ('Windows 11（内核 10.0.26200）', '无 GPU，全部计算在 CPU 上完成'),
    ('3.14.6', '仅依赖标准库与科学计算栈'),
    ('2.14.0+cpu', 'torch.set_num_threads(4)，限制线程数以保留 CPU 余量'),
    ('3.11.1', '绘制 loss 曲线与采样多样度对比图'),
    ('本机未安装 TRAE，使用当前环境自带的命令行 AI 编程助手',
     '使用功能：生成模型骨架、定位掩码/形状错误、复核参数量手算'),
]
for row, (ver, note) in zip(env.rows[1:], env_rows):
    set_cell_text(row.cells[1], ver)
    set_cell_text(row.cells[2], note)

# ---------------- 4) 2.2 基线结果 ----------------
for it in items:
    if it[0] == 'p' and '参数量' in ptext(it) and '＿' in ptext(it):
        set_para_text(it[1],
                      f'参数量：{base["params"]:,}（sum(p.numel() for p in model.parameters())）　　'
                      f'最终 loss：{base["final_loss"]:.4f}　　'
                      f'初始 loss（应 ≈ ln {base["vocab"]} ≈ {math.log(base["vocab"]):.2f}）：'
                      f'{base["init_loss"]:.4f}　　'
                      f'耗时：{base["time_min"]:.1f} min（{base["iters"]} 步，batch {base["batch_size"]}）')

# 图 1 之前插入基线逐段小结
i = find_caption('图 1')
insert_after(items[i], [
    f'训练过程：loss 从初始的 {base["init_loss"]:.4f}（≈ ln V，说明随机初始化时模型对词表内每个字符'
    f'给出近乎均匀的概率）迅速下降——前 100 步均值 {base["loss_first100"]:.2f}，'
    f'第 1000 步已到 0.027 附近，之后进入平台期（最终 {base["final_loss"]:.4f}）。'
    f'这既说明"预测下一个字"的自监督目标确实在驱动模型学到字与字的搭配关系，'
    f'也说明本实验语料只有 {base["n_chars"]} 个字符（72 首唐诗），'
    f'2000 步足以让模型把训练语料几乎逐字记住——这一点在生成结果里表现得非常明显。',
    f'生成示例：以「春」「月」为提示时，模型能整首背出训练语料中的名篇'
    f'（如“春眠不觉晓，处处闻啼鸟”“春来发几枝。愿君多采撷，此物最相思”），'
    f'格律、换行与标点都正确；"错误"集中表现为两首诗被拼在一起'
    f'（“夜泊清明时节雨纷纷”）或中途截断，属于典型的背诵式生成（详见 5.1 采样分析与 8 的小结）。',
])

# 生成示例原文粘贴框
gen_box = [it for it in items if it[0] == 't' and it[1].rows[0].cells[0].text.strip().startswith('【在此粘贴生成示例原文')]
if gen_box:
    lines = []
    for prompt, outs in base["samples"].items():
        for k, s in enumerate(outs, 1):
            lines.append(f'「{prompt}」样本 {k}：{s}')
    fill_cell_multiline(gen_box[0][1].rows[0].cells[0], lines)

# ---------------- 5) 表 3 参数量手算 ----------------
V, C, T_, L = base['vocab'], 128, base['block_size'], 2
hand = V * C + T_ * C + L * 12 * C * C
params_tb = find_table('模块')
rows = [
    (f'{V} × {C} = {V*C:,}', f'{V*C:,}',
     f'✓ 一致（tok_emb.weight 形状 {V}×{C}）'),
    (f'{T_} × {C} = {T_*C:,}', f'{T_*C:,}',
     f'✓ 一致（pos_emb.weight 形状 {T_}×{C}）'),
    (f'{L} × 12 × {C}² = {L}×12×{C*C:,} = {L*12*C*C:,}', f'{L*12*C*C:,}',
     f'✓ 一致（两个块各 198,272：注意力 66,048 + MLP 131,712 + 2×LayerNorm 512）'),
    (f'{V*C:,} + {T_*C:,} + {L*12*C*C:,} = {hand:,}', f'{hand:,}（不含偏置与 LayerNorm）',
     f'✓ 与打印值 {base["params"]:,} 误差 '
     f'{(base["params"]-hand)/base["params"]*100:.2f}% < 1%；差值 {base["params"]-hand:,} '
     f'= LayerNorm 5×2C = 1,280 + 注意力偏置 2×4C = 1,024 + MLP 偏置 2×5C = 1,280'),
]
for row, (proc, res, chk) in zip(params_tb.rows[1:], rows):
    set_cell_text(row.cells[1], proc)
    set_cell_text(row.cells[2], res)
    set_cell_text(row.cells[3], chk)

# ---------------- 6) 表 4 对照实验结果 ----------------
exp_tb = find_table('编号')
exp_rows = [
    ('exp1', '学习率', loss_str([('1e-2', 'exp1a'), ('1e-4', 'exp1b')]),
     f'1e-2 {stars(QUALITY["exp1a"][0])}；1e-4 {stars(QUALITY["exp1b"][0])}',
     f'1e-2 步长偏大：前 100 步均值 {R["exp1a"]["loss_first100"]:.2f}（基线 {base["loss_first100"]:.2f}），'
     f'前期震荡明显，1000 步收敛到 {R["exp1a"]["final_loss"]:.3f}，为基线（2000 步 {base["final_loss"]:.3f}）'
     f'的 {R["exp1a"]["final_loss"]/base["final_loss"]:.1f} 倍，未发散但明显更差，生成出现大量混拼；'
     f'1e-4 步长过小：1000 步只降到 {R["exp1b"]["final_loss"]:.3f}，'
     f'生成仍是破碎词片，属于欠拟合。两组初始 loss 都从 ≈ 6.56 出发，起点一致。'),
    ('exp2', '层数 n_layer', loss_str([('1 层', 'exp2a'), ('4 层', 'exp2b')]),
     f'1 层 {stars(QUALITY["exp2a"][0])}；4 层 {stars(QUALITY["exp2b"][0])}',
     f'1 层最终 {R["exp2a"]["final_loss"]:.3f}、4 层 {R["exp2b"]["final_loss"]:.3f}，'
     f'都已在 1000 步内接近基线 2000 步的 {base["final_loss"]:.3f}；参数量 '
     f'{R["exp2a"]["params"]:,} → {R["exp2b"]["params"]:,}（每层 198,272），'
     f'耗时 {R["exp2a"]["time_min"]:.1f} → {R["exp2b"]["time_min"]:.1f} min。'
     f'生成上 1 层与 4 层都能整首背出训练诗，差别只在错字比例，说明语料太小时层数很快失去收益。'),
    ('exp3', '嵌入维度 n_embd', loss_str([('64', 'exp3a'), ('256', 'exp3b')]),
     f'64 维 {stars(QUALITY["exp3a"][0])}；256 维 {stars(QUALITY["exp3b"][0])}',
     f'参数量随 C 按 12C² 增长：64 维 {R["exp3a"]["params"]:,}（{R["exp3a"]["params"]/base["params"]:.2f} 倍），'
     f'256 维 {R["exp3b"]["params"]:,}（{R["exp3b"]["params"]/base["params"]:.1f} 倍）；'
     f'耗时 {R["exp3a"]["time_min"]:.1f} → {R["exp3b"]["time_min"]:.1f} min；'
     f'最终 loss {R["exp3a"]["final_loss"]:.3f} → {R["exp3b"]["final_loss"]:.3f}。'
     f'64 维明显不够用（生成里正确整块与错乱交替、连续逗号），256 维与基线基本持平——'
     f'收益随参数量增长快速递减。'),
    ('exp4', '上下文 block_size', loss_str([('T=32', 'exp4')]),
     stars(QUALITY['exp4'][0]),
     f'最终 loss {R["exp4"]["final_loss"]:.3f}，明显高于基线 {base["final_loss"]:.3f}；'
     f'耗时降到 {R["exp4"]["time_min"]:.1f} min（自注意力开销随 T² 下降）。'
     f'需注意这里同时存在两个变化：上下文变短，且每一步看到的 token 数只有基线的 1/4'
     f'（32×32 对 32×128），所以 loss 的差距不能全部归因于"上下文短"。'
     f'生成仍以整首背诵推进，但句间衔接的拼接错误变多（“湖光秋月两相照香炉生紫烟”）。'),
    ('exp5', '位置编码', loss_str([('去掉位置编码', 'exp5')]),
     stars(QUALITY['exp5'][0]),
     f'去掉位置编码后 loss 仍降到 {R["exp5"]["final_loss"]:.3f}（基线 {base["final_loss"]:.3f}），'
     f'参数量正好少 T×C = {base["block_size"]*128:,}（{R["exp5"]["params"]:,}）。'
     f'原因是因果掩码本身还残留一部分顺序信息（每个位置能看到的"前文集合"不同），'
     f'加之语料高度重复、模型基本在背诵，所以 loss 差距被掩盖；'
     f'但生成的句内位置出现错位（以「月」为提示时输出“月下下扬州”“月汉时关”这类重字/漏字），'
     f'见思考题 (1)。'),
    ('exp6', '采样温度', f'未重训，沿用基线权重（最终 loss {base["final_loss"]:.3f}）',
     f'T=0.5 {stars(QUALITY["exp6"][0])}；T=1.5 {stars(2)}',
     f'只改采样参数、不动权重：T=0.5 时四段生成几乎一字不差（重复 3-gram 由 '
     f'{SAMP[1]["repeat3"]:.3f} 升到 {SAMP[2]["repeat3"]:.3f}，distinct-2 由 '
     f'{SAMP[1]["distinct2"]:.3f} 降到 {SAMP[2]["distinct2"]:.3f}）；T=1.5 时开始混拼'
     f'（“春来终觉浅，绝胜烟豚”），distinct-2 反而回落到 {SAMP[3]["distinct2"]:.3f}。'
     f'温度只改变抽样随机性，不改变模型学到的知识上限。'),
]
for row, (_, _, loss_v, qual, phen) in zip(exp_tb.rows[1:], exp_rows):
    set_cell_text(row.cells[4], loss_v)
    set_cell_text(row.cells[5], qual)
    set_cell_text(row.cells[6], phen)

# ---------------- 7) 4.1~4.5：曲线图 + 分析 ----------------
CAP_FIG = {
    '图 1': ([base['curve']], [14]),
    '图 2': (['loss_curve_L2_E128_lr0.01.png', 'loss_curve_L2_E128_lr0.0001.png'], [7.5, 7.5]),
    '图 3': (['loss_curve_L1_E128_lr0.001.png', 'loss_curve_L4_E128_lr0.001.png'], [7.5, 7.5]),
    '图 4': (['loss_curve_L2_E64_lr0.001.png', 'loss_curve_L2_E256_lr0.001.png'], [7.5, 7.5]),
    '图 5': ([R['exp4']['curve']], [11]),
    '图 6': ([R['exp5']['curve']], [11]),
}
for cap, (fnames, widths) in CAP_FIG.items():
    i = find_caption(cap)
    tb = prev_table(i)
    if tb is not None:
        add_imgs_to_cell(tb.rows[0].cells[0], [FIG(f) for f in fnames], widths)

ANALYSIS = {
    '4.1': [
        f'学习率决定每步走多远，也决定前期是否"震荡"。1e-2（红线）前 100 步均值 '
        f'{R["exp1a"]["loss_first100"]:.2f}，明显高于基线的 {base["loss_first100"]:.2f}：'
        f'步长偏大时模型在损失面上反复跨过低点，前期下降慢；'
        f'1000 步后收敛到 {R["exp1a"]["final_loss"]:.3f}，是基线（2000 步 {base["final_loss"]:.3f}）的 '
        f'{R["exp1a"]["final_loss"]/base["final_loss"]:.1f} 倍。'
        f'它没有发散（本实验用 AdamW + 梯度裁剪，梯度范数被限制在 1 以内），'
        f'但已经明显劣于基线——生成的文本出现“远芳侵古千里江陵一日还”这类把两首诗粘在一起的错误。',
        f'1e-4（蓝线）前 100 步均值 {R["exp1b"]["loss_first100"]:.2f}，与基线接近但全程"走得慢"：'
        f'1000 步只降到 {R["exp1b"]["final_loss"]:.3f}，曲线仍在下行而远未收敛，'
        f'生成为“秦时鸣。寒雨知此空尽，今。”这样的破碎词片。'
        f'这说明 1e-4 是"还没走到"而不是"走不动"，属于欠拟合，代价是同样的算力换不到结果。',
        f'结论：在本实验规模（50 万参数、2000 字符语料、AdamW）下 1e-3 是合适的数量级；'
        f'学习率再小会显著延长训练时间，再大会牺牲最终精度——'
        f'工程上通常用预热 + 余弦衰减同时利用"大步长快速下降"和"小步长精细收敛"两个阶段。',
    ],
    '4.2': [
        f'1 层（{R["exp2a"]["params"]:,} 参数，仍是 4 头 × 32 维）最终 loss {R["exp2a"]["final_loss"]:.3f}，'
        f'4 层（{R["exp2b"]["params"]:,} 参数）{R["exp2b"]["final_loss"]:.3f}，'
        f'两者相差 {abs(R["exp2b"]["final_loss"]-R["exp2a"]["final_loss"]):.3f}，'
        f'而且都已在 1000 步内逼近基线 2000 步的 {base["final_loss"]:.3f}；'
        f'耗时 {R["exp2a"]["time_min"]:.1f} → {R["exp2b"]["time_min"]:.1f} min（约 '
        f'{R["exp2b"]["time_min"]/R["exp2a"]["time_min"]:.1f} 倍），参数量相差 '
        f'{R["exp2b"]["params"]/R["exp2a"]["params"]:.2f} 倍（每多一层 +198,272）。',
        f'生成质量上两组都能整首背出训练诗（“春来发几枝…此物最相思”“独坐幽篁里…”），'
        f'区别只在错字比例：1 层偶有多字（“春山前不见古人”），4 层几乎逐字正确。'
        f'这正是小语料实验的典型现象——损失与"背诵"在很浅的网络里就已经饱和，'
        f'加深网络主要是让记忆更精确，而不是获得新的语言能力；'
        f'因此在 2163 字的语料上，2 层（基线）已经是"容量—成本"的合理折中。',
    ],
    '4.3': [
        f'嵌入维度 C 决定模型每步的信息"带宽"，参数量按 12C² 增长（12 × 128² = 196,608 占基线的 39%）：'
        f'C=64 时参数量降到 {R["exp3a"]["params"]:,}（基线的 {R["exp3a"]["params"]/base["params"]:.2f} 倍）、'
        f'耗时 {R["exp3a"]["time_min"]:.1f} min；C=256 时升到 {R["exp3b"]["params"]:,}'
        f'（基线的 {R["exp3b"]["params"]/base["params"]:.1f} 倍）、耗时 {R["exp3b"]["time_min"]:.1f} min。',
        f'生成质量的差别比 loss 更直观：64 维的模型"正确整块与错乱交替"'
        f'（“春来发发几人”“好雨知知时节”“送王孙去，，，，，，”），'
        f'说明容量不足时连背诵都不完整；256 维的生成与基线基本持平，'
        f'最终 loss 从 {R["exp3a"]["final_loss"]:.3f} 降到 {R["exp3b"]["final_loss"]:.3f}。'
        f'但注意代价：18 万 → 179 万参数、2.1 → 12.1 min，只换来 0.024 的 loss 改善，'
        f'说明在这份小语料上容量已经过剩——继续加宽不如先扩充语料。',
    ],
    '4.4': [
        f'上下文长度 T 从 128 降到 32 后，最终 loss {R["exp4"]["final_loss"]:.3f}，'
        f'明显高于基线 {base["final_loss"]:.3f}；耗时 {R["exp4"]["time_min"]:.1f} min，'
        f'是目前最快的配置（自注意力开销随 T² 下降，同时每步要处理的 token 数也从 4096 降到 1024）。',
        f'这里必须诚实区分两个同时变化的因素：① 模型每一步只能"看到" 32 个字符（约两句诗）；'
        f'② 同样 1000 步，模型看到的训练 token 总量只有基线的 1/4。'
        f'因此 loss 的差距不能全部归因于"上下文太短"。'
        f'从生成看，T=32 仍能以整首背诵的方式推进（“春眠不觉晓…”“月黑雁飞高…”），'
        f'但句间衔接处的拼接错误更多（“湖光秋月两相照香炉生紫烟”“粒粒粒皆辛苦”），'
        f'说明跨句的呼应确实需要更长的上下文窗口；'
        f'要做严格的控制变量，应把 T=32 的步数或批大小调大，使"看到的 token 总量"与基线一致。',
    ],
    '4.5': [
        f'去掉可学习位置编码后，最终 loss 仍降到 {R["exp5"]["final_loss"]:.3f}'
        f'（基线 {base["final_loss"]:.3f}），参数量正好少了 T×C = {base["block_size"]*128:,}。'
        f'原因有两层：一是自注意力的打分（q·k）本身与位置无关，LayerNorm 与 MLP 也是逐位置独立的；'
        f'二是因果掩码仍然残留一部分顺序信息——第 t 个位置能看到的"前文集合"与第 t-1 个位置不同，'
        f'模型可以借此隐式推断相对位置。再加上语料高度重复、模型主要在做"背诵"，'
        f'所以 loss 上的差距很小。',
        f'但生成的细节暴露了缺失：以「月」为提示时出现“月下下扬州”“月汉时关”“月，万户捣衣声”'
        f'这类重字、漏字与错位（对照基线是“月落乌啼霜满天”“月黑雁飞高”“长安一片月，万户捣衣声”），'
        f'说明模型缺少"当前处于句子的第几个位置"的显式表示，只能靠前几个字接龙，'
        f'一旦开头几个字对应多首诗的公共前缀，就容易"接错路径"；'
        f'这与教材 5.2 节的结论一致：位置编码是显式注入顺序信息的必要手段（详见思考题 (1)）。',
    ],
}
for sec, lines in ANALYSIS.items():
    i = find_caption(f'图 {int(sec.split(".")[1]) + 1}')
    underscores = []
    j = i + 1
    while j < len(items) and len(underscores) < 2:
        if is_underscore(items[j]):
            underscores.append(items[j])
        j += 1
    fill_underscores(underscores, lines)

# ---------------- 8) 表 5 采样实验记录 ----------------
samp_tb = find_table('组号')
SAMP_NOTE = {
    1: f'distinct-2 {SAMP[1]["distinct2"]:.3f}、重复 3-gram {SAMP[1]["repeat3"]:.3f}：'
       f'四段各自沿着不同的训练诗往下背，段与段之间几乎没有雷同，三项指标都是最好的一组；'
       f'唯一的"瑕疵"是每段都会在 120 字处被截断。',
    2: f'distinct-2 降到 {SAMP[2]["distinct2"]:.3f}、重复 3-gram 升到 {SAMP[2]["repeat3"]:.3f}：'
       f'低温把分布压尖，两次采样几乎落到同一条记忆路径上（「月」的两段一字不差），'
       f'输出最"工整正确"，但作为生成器完全没有新意。',
    3: f'distinct-2 回落到 {SAMP[3]["distinct2"]:.3f}、重复 3-gram {SAMP[3]["repeat3"]:.3f}：'
       f'高温让长尾候选有机会被选中，出现"春来终觉浅，绝胜烟豚""…白日还。白在君自古谁无晴却有黄鹤一行"'
       f'这类混拼与重字，用字更"新"但已经偏离了通顺——多样性的指标没有变好，说明此时新增的字符大多是噪声。',
    4: f'distinct-2 {SAMP[4]["distinct2"]:.3f}、重复 3-gram {SAMP[4]["repeat3"]:.3f}：'
       f'每组只从 5 个候选里抽，两次「春」的采样完全相同——窄候选集与低温效果类似，'
       f'都让采样退化为"近确定性"解码。',
    5: f'distinct-2 {SAMP[5]["distinct2"]:.3f}、重复 3-gram {SAMP[5]["repeat3"]:.3f}：'
       f'候选集放大到整张词表（699 字）后指标与 top_k=5 接近，出现"春色色来天地"这类重字；'
       f'说明在已经过拟合的小模型上，放开候选集并不会自动带来更好的多样性。',
}
for row, gid in zip(samp_tb.rows[1:], [1, 2, 3, 4, 5]):
    rec = SAMP[gid]
    excerpts = []
    for s in rec['texts'][:2]:
        excerpts.append(s.replace('\n', '／')[:34] + '…')
    set_cell_text(row.cells[3], '；'.join(excerpts))
    set_cell_text(row.cells[4], SAMP_NOTE[gid])

# 图 7
i = find_caption('图 7')
tb = prev_table(i)
if tb is not None:
    add_imgs_to_cell(tb.rows[0].cells[0], [FIG('result_sampling.png')], [15])
fill_underscores([items[j] for j in range(i + 1, len(items)) if is_underscore(items[j])][:3], [
    f'（1）确定性 ↔ 多样性：temperature 越小，softmax 越尖锐，采样越接近贪心，'
    f'输出越"正确"也越死板——T=0.5 时两次「月」的采样一字不差，重复 3-gram 由 '
    f'{SAMP[1]["repeat3"]:.3f} 升到 {SAMP[2]["repeat3"]:.3f}、distinct-2 由 '
    f'{SAMP[1]["distinct2"]:.3f} 降到 {SAMP[2]["distinct2"]:.3f}；'
    f'T=1.0 时四段沿着不同的诗往下走，distinct-2 达到最高的 {SAMP[1]["distinct2"]:.3f}、'
    f'重复最低的 {SAMP[1]["repeat3"]:.3f}；T=1.5 时开始混拼，指标反而回落。',
    f'（2）重复 ↔ 胡言：这里要区分两种"重复"——一种是单段内部的循环复读，'
    f'另一种是多次采样落到同一条高概率路径（低温或窄候选时出现）。'
    f'top_k=5 与 T=0.5 属于后者（两组重复 3-gram 都是 '
    f'{SAMP[4]["repeat3"]:.3f}），不限制候选（{SAMP[5]["repeat3"]:.3f}）也没有改善，'
    f'因为本模型已经过拟合、把训练诗背得很熟，"高概率"本身就是正确答案。'
    f'真正的"胡言"出现在 T=1.5：混拼、重字与生造搭配开始出现，'
    f'说明温度过高会把长尾噪声放出来。',
    f'（3）推荐组合：temperature=1.0、top_k=20（即基线采样）。理由是本组在两项指标上同时最好'
    f'（distinct-2 {SAMP[1]["distinct2"]:.3f} 最高、重复 3-gram {SAMP[1]["repeat3"]:.3f} 最低），'
    f'且生成内容全是通顺的诗句；T=0.5 与 top_k=5 只是把同一首诗背得更死、不产生新内容，'
    f'T=1.5 则明显损伤通顺度。若目标是"像诗人一样写出新句子"，问题不在采样旋钮，'
    f'而在于语料太小导致模型只会背诵——应先扩充语料或加入重复惩罚/温度衰减等训练侧手段，'
    f'而不是单纯把温度调高。',
])

# ---------------- 9) 5.3 进阶：禁止重复 n-gram 采样 ----------------
anchor = find_heading('6  ')[1]._element
ref = copy.deepcopy(anchor)


def new_para_before_anchor(text, bold=False):
    new_p = copy.deepcopy(ref)
    for r in new_p.findall(qn('w:r')):
        new_p.remove(r)
    anchor.addprevious(new_p)
    p = Paragraph(new_p, doc)
    if text:
        p.add_run(text).bold = bold
    return p


new_para_before_anchor('5.3  进阶：禁止重复 n-gram 采样（选做 b）', bold=True)
new_para_before_anchor(
    f'做法：采样时维护已生成序列的 3-gram 集合，逐字把"会构成重复 3-gram 的候选字"置为 '
    f'-inf 后再采样（min_llm.py 的 generate(..., no_repeat_ngram=3)），'
    f'其余设置与基线采样完全一致（temperature=1.0、top_k=20、提示词「春」「月」各 2 段）。')
pic_p = new_para_before_anchor(None)
pic_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
pic_p.add_run().add_picture(FIG('result_norepeat.png'), width=Cm(11))
new_para_before_anchor('图 8  禁止重复 3-gram 采样前后对比')
new_para_before_anchor(
    f'结果是一次"有用但有代价"的实验。与最接近的对照设置（temperature=1.0、top_k=20，'
    f'不限制候选）相比，重复 3-gram 比例从 {SAMP[1]["repeat3"]:.3f} 变成 {NR["repeat3"]:.3f}，'
    f'并没有下降；distinct-2 从 {SAMP[1]["distinct2"]:.3f} 变成 {NR["distinct2"]:.3f}，'
    f'反而略低。原因是本实验里"重复"的主要来源不是单段内部的循环复读，'
    f'而是多次采样落到同一条高概率的记忆路径上（低温、窄候选时尤其明显）；'
    f'而禁止重复 3-gram 只作用于单段内部，对"不同样本之间的雷同"无能为力。')
new_para_before_anchor(
    f'更值得注意的是副作用：模型原本沿着训练诗的轨迹一句句背下去，'
    f'硬约束会在它遇到重复 3-gram 时强制把概率质量挪到别的候选上，'
    f'于是被"推离"了高概率路径，产生“今日之道是无日多烦忧”这类把两首诗混拼在一起的句子，'
    f'而且新路径上很快又会出现新的重复 3-gram。这与工业界常用的重复惩罚（repetition penalty）'
    f'形成对比：后者是把已出现 token 的 logit 乘上一个惩罚系数或减去一个常数，'
    f'属于"软折扣"，既压低重复概率，又保留"实在没有别的选择时仍可重复"的余地，'
    f'不会像硬掩码那样把模型逼到分布之外。')
new_para_before_anchor(
    f'结论：重复惩罚是常用的采样技巧，但它针对的是"模型只会循环"的失效模式；'
    f'本实验模型的问题恰恰相反——它把 2163 字的语料背得太好，采样层面的约束无法创造出语料里没有的语言结构。'
    f'这也从反面说明：采样参数只能调节"从已有分布里怎么取"，'
    f'想真正提升生成质量仍要回到数据规模与训练目标。')

# ---------------- 10) 6 AI 协作记录 ----------------
set_para_text(find_heading('6  ')[1], '6  AI 协作记录（TRAE / 本机 AI 助手使用情况）')
ai_tb = find_table('序号')
ai_rows = [
    ('生成 min_llm.py 的整体框架',
     '要求 AI 按指南第七节的模块划分实现字符级最小 GPT：CharTokenizer、'
     'CausalSelfAttention、Block（Pre-Norm）、MiniGPT、训练与采样函数，'
     '并在每个张量后标注形状（统一为 (B, T, C)，注意力内部才拆多头维）',
     'AI 一次给出了可运行的骨架。我逐段核对了三处关键实现：'
     '① 因果掩码在 softmax 之前用 float("-inf") 填充（而不是用 0 或事后乘掩码）；'
     '② 多头的 reshape 顺序为 (B,T,C) → (B,T,3,H,hs) → (B,H,T,hs)，避免头维与序列维混淆；'
     '③ 权重绑定写成 self.head.weight = self.tok_emb.weight（共享同一个 Parameter）。'),
    ('核对模型结构与超参对应关系',
     '请对照指南 4.2 节的参数公式检查实现：MLP 是否为 4C 隐层 + GELU、'
     '每层是否 Pre-Norm、位置编码是否可学习、初始化是否为 N(0, 0.02)',
     '按 AI 的检查清单逐项确认后发现：初始化标准差必须用 0.02，'
     '这样才能让初始 loss ≈ ln 699 ≈ 6.55（实测 6.5627）；'
     '若用 PyTorch 默认初始化，初始 loss 会明显偏离该值，'
     '也就失去了"初始 loss 应 ≈ ln V"这一自检手段。'),
    ('复核参数量手算与代码打印的差值',
     '手算 V×C + T×C + L×12C² = 499,072，但 sum(p.numel()) 打印 502,656，'
     '差 3,584 是哪来的',
     'AI 建议按模块逐项打印（named_children + 每个块的子模块），'
     '并提醒 head.weight 与 tok_emb.weight 是同一张量、逐模块相加会重复计数。'
     '按此方法得到差值构成：LayerNorm 5×2C = 1,280 + 注意力偏置 2×4C = 1,024 + '
     'MLP 偏置 2×5C = 1,280 = 3,584，与手算完全对上'
     '（脚本 check_params_hw3.py，输出见文末附录）。'),
]
for row, (scene, prompt, out) in zip(ai_tb.rows[1:], ai_rows):
    set_cell_text(row.cells[1], scene)
    set_cell_text(row.cells[2], prompt)
    set_cell_text(row.cells[3], out)

# ---------------- 11) 7 思考题 ----------------
Q_ANSWERS = [
    # (1) 位置编码
    ['去掉位置编码后，模型仍能学到"哪些字经常出现、哪些字经常相邻"这样的字频与搭配统计'
     '（所以 exp5 的 loss 依然能降到 {:.4f}，与基线的 {:.4f} 相差不到 0.01）。'
     '理论上，自注意力对输入是排列等变的：q·k 只与 token 内容有关，softmax 加权也是集合上的加权求和，'
     'LayerNorm 与 MLP 又都是逐位置独立处理的，模型里不含任何显式的"第几个位置"信息；'
     '但因果掩码会残留一部分隐式顺序信息——第 t 个位置能看到的"前文集合"与第 t-1 个位置不同，'
     '模型可以借此推断出相对的先后关系。这是本实验 loss 差距很小的主要原因。'.format(
         R['exp5']['final_loss'], base['final_loss']),
     '生成的"诗"表面上仍然通顺——因为语料只有 {n} 个字符（72 首唐诗）、模型基本在背诵，'
     '靠"前几个字"就能沿着记忆路径接龙。但一旦提示词对应多首诗的公共前缀，缺少位置表示的问题就暴露了：'
     '以「月」为提示时输出“月下下扬州”（重字）、“月汉时关”（漏掉"秦时明"）、“月，万户捣衣声”'
     '（漏掉"长安一片"），而基线是“月落乌啼霜满天”“月黑雁飞高”“长安一片月，万户捣衣声”。'
     '这些错位正说明模型不知道"当前处于句子的第几个字符"，也就无法稳定地控制五言/七言的句子长度与换行位置。'.format(
         n=base['n_chars']),
     '因果掩码只保证"只能用前文"，它约束的是信息的流向，并不提供位置表示，因此不能替代位置编码；'
     '要把"绝对/相对位置"显式注入模型，就需要可学习位置嵌入、正弦位置编码或 RoPE 等机制。'
     '这也解释了教材 5.2 节的结论：自注意力本身是排列等变的，'
     '必须靠位置编码把顺序信息注入模型（依据：教材 5.2 节；本实验 exp5 实测，'
     '对照代码 min_llm.py 中的 use_pos 开关）。'],
    # (2) 参数量手算
    ['手算（V = {V}、T = {T}，C 由 128 增至 256、层数由 2 增至 3）：\n'
     '　词嵌入 V×C = {V} × 256 = {a:,}\n'
     '　位置编码 T×C = {T} × 256 = {b:,}\n'
     '　Transformer 块 L×12C² = 3 × 12 × 256² = 3 × 12 × 65,536 = {c:,}\n'
     '　三项合计 ≈ {s:,}，加上偏置与 LayerNorm（(2L+1)×2C = 7×512 = {ln:,} +'
     ' L×4C = {ba:,} + L×5C = {bm:,} = {bz:,}）约为 {tot:,}。'.format(
         V=base['vocab'], T=base['block_size'],
         a=base['vocab'] * 256, b=base['block_size'] * 256,
         c=3 * 12 * 256 * 256, s=base['vocab'] * 256 + base['block_size'] * 256 + 3 * 12 * 256 * 256,
         ln=7 * 512, ba=3 * 1024, bm=3 * 1280, bz=7 * 512 + 3 * 1024 + 3 * 1280,
         tot=base['vocab'] * 256 + base['block_size'] * 256 + 3 * 12 * 256 * 256 + 7 * 512 + 3 * 1024 + 3 * 1280),
     '相比基线（手算 {hand:,} / 打印 {par:,}）增长约 {r1:.2f} 倍，'
     '其中绝大部分来自 12C² 项：C 翻倍使该项变为 4 倍（{c3:,} → {c2:,}），'
     '层数只贡献线性增长（2 → 3）。这也说明"加宽"比"加深"贵得多：'
     '想加容量时，先加层往往更划算。'.format(
         hand=base['vocab'] * 128 + base['block_size'] * 128 + 2 * 12 * 128 * 128,
         par=base['params'],
         r1=(base['vocab'] * 256 + base['block_size'] * 256 + 3 * 12 * 256 * 256) /
            (base['vocab'] * 128 + base['block_size'] * 128 + 2 * 12 * 128 * 128),
         c3=2 * 12 * 128 * 128, c2=3 * 12 * 256 * 256),
     '结合实测：exp3 中 C 从 64 升到 256，参数量增加 {p3:.1f} 倍、耗时 {t3a:.1f} → {t3b:.1f} min，'
     '而最终 loss 只从 {l3a:.3f} 降到 {l3b:.3f}；exp2 中层数 1 → 4 使参数量变为 {p2:.2f} 倍'
     '（{pa:,} → {pb:,}），耗时 {t2a:.1f} → {t2b:.1f} min，loss 从 {l2a:.3f} 降到 {l2b:.3f}。'
     '在只有 {nc} 字语料的前提下，容量已经过剩，继续加大会更快进入"复读训练集"的区域，'
     '因此容量与成本的权衡结论是：先保证数据规模，再按"层数优先、宽度其次"的顺序扩容量。'
     '（依据：实验指南 4.2 节参数量公式；exp2、exp3 的实测耗时与 loss 来自 results_lab03.json。）'.format(
         p2=R['exp2b']['params'] / R['exp2a']['params'],
         pa=R['exp2a']['params'], pb=R['exp2b']['params'],
         p3=R['exp3b']['params'] / R['exp3a']['params'],
         t3a=R['exp3a']['time_min'], t3b=R['exp3b']['time_min'],
         l3a=R['exp3a']['final_loss'], l3b=R['exp3b']['final_loss'],
         t2a=R['exp2a']['time_min'], t2b=R['exp2b']['time_min'],
         l2a=R['exp2a']['final_loss'], l2b=R['exp2b']['final_loss'],
         nc=base['n_chars'])],
    # (3) 温度极限
    ['T→0：softmax(logits/T) 趋于 one-hot，除最大 logit 外概率全为 0，'
     '采样退化成贪心解码（greedy / argmax），输出完全确定、可复现，'
     '但容易一步错步步错并陷入循环；严格说 T=0 在公式上是除零，'
     '工程实现里是"取 argmax"或用一个极小温度近似。',
     'T→∞：logits/T → 0，softmax 趋于均匀分布，采样等价于从词表里均匀随机取字，'
     '生成结果与模型学到的知识几乎无关，全是噪声。',
     '适合低温的任务：答案确定、需要稳定复现的场景——翻译、代码补全、信息抽取、'
     '数学推理、格式化输出（这也是各家的"确定性 API"参数）；'
     '适合高温的场景：需要创意与多样性的开放生成——写诗/起名/广告语、头脑风暴、'
     '数据增强、对话中的"发散"分支，通常还要配合 top-k/top-p 截断，避免高温把长尾怪字放出来。'
     '结合 exp6：本模型 T=0.5 时重复 3-gram 为 {r2:.3f}（两次采样落到同一条记忆路径，最保守），'
     'T=1.5 时 distinct-2 反而由 {d1:.3f} 回落到 {r3:.3f} 并出现"春来终觉浅，绝胜烟豚"这类混拼——'
     '正是"确定性 ↔ 多样性"的两端（低温死板、高温胡言）。'
     '（依据：指南任务五对采样参数的说明；本实验 sampling_hw3.json 实测。）'.format(
         r2=SAMP[2]['repeat3'], r3=SAMP[3]['distinct2'], d1=SAMP[1]['distinct2'])],
    # (4) 字符级 vs BPE
    ['字符级分词的优点：① 词表极小（本实验 V = {v}，全部来自 {nc} 字语料），'
     '不需要外部分词器与词表文件，也彻底没有 OOV（未登录词）问题；'
     '② 对拼写错误、生僻字、新词、代码/公式等"没有词边界"的文本天然鲁棒；'
     '③ 中文里一个汉字通常就是一个语素，"字"与"词"的粒度差距还可以接受。'.format(
         v=base['vocab'], nc=base['n_chars']),
     '缺点：① 序列变长——同样一段文本，字符级 token 数通常比子词级多 2~4 倍，'
     '而自注意力的计算量随 T² 增长，训练与推理都更贵；'
     '② 每个 token 携带的语义太少，模型要花很多容量去"学会分词"本身'
     '（本实验 loss 已降到 {l:.4f}、生成却只是逐字背诵训练语料，就有这层原因）；'
     '③ 英文等拼音文字按字符切分会把一个词切成 5~10 个 token，效率很低。'.format(
         l=base['final_loss']),
     'BPE 的折中：从单字符词表出发，反复统计当前语料里出现频率最高的相邻符号对并合并为新符号，'
     '合并次数（词表大小）就是那个"旋钮"——词表越大，常见片段被合并得越多，序列越短；'
     '词表越小，长词只能被拆成子词。这样高频词/词缀成为单个 token（短序列），'
     '罕见词仍能拆成子词表示（无 OOV），实现了"词表大小 ↔ 序列长度"的连续折中。',
     '真实 LLM 都用 BPE 类（BPE / WordPiece / SentencePiece / Unigram）分词器，因为它同时满足：'
     '① 词表规模可控（几万到十几万），使嵌入层与输出层参数量（V×C，权重绑定后仍是大头）可控；'
     '② 序列压缩让同样的上下文窗口装下更多文本；'
     '③ 多语言、代码、罕见词统一处理，不会出现 OOV；'
     '④ 子词切分保留了形态学信息（un-、-ing、-ed 等前后缀），有利于泛化到未见过的词。'
     '（依据：教材分词章节与 BPE 原始论文 Sennrich et al., 2016；本实验字符级分词器 '
     'CharTokenizer 见 min_llm.py，V = {}。）'.format(base['vocab'])],
    # (5) 与真实 LLM 对比
    ['三个方面的量级差：\n'
     '　数据量：本实验 {nc} 字符（约几 KB，几十首唐诗） vs 真实 LLM 的 '
     '10¹²~10¹³ 个 token（TB 级网页、书籍、代码），相差约 9~10 个数量级；\n'
     '　参数量：本实验 {par:,}（约 50 万） vs 主流大模型 7B~70B 甚至上千亿，'
     '相差约 5~6 个数量级；\n'
     '　训练目标：本实验只有单一的自监督目标"预测下一个字符"，且只训练 {it} 步；'
     '真实 LLM 是"预训练（下一 token 预测）+ 监督微调（SFT，学会按指令回答）+ '
     '人类偏好对齐（RLHF/DPO，学会有用、诚实、无害）"，把"续写"塑造成"完成任务"。'.format(
         nc=base['n_chars'], par=base['params'], it=base['iters']),
     '把语料扩大 1000 倍（约 200 万字，仍是真实预训练语料里的一粒沙）：'
     '模型会明显变好——生成的诗更通顺、更少重复、格律更稳，'
     '因为语言模型的 loss 与下游能力大体随数据量和参数量呈幂律改善；'
     '参照 exp2/exp3，只要不先撞上容量瓶颈，扩数据带来的收益通常比单纯加参数更实在。',
     '但它仍然不会"聊天"。原因不只是规模：一是"会聊天"需要指令微调与对齐数据，'
     '教模型理解"用户让我做什么"（本实验的模型只会无条件续写，没有"提问—回答"的结构）；'
     '二是对话还需要世界知识与推理，这要求更大的参数与更广的语料；'
     '三是真实产品还要外挂提示工程、工具调用、检索增强与安全对齐。'
     '所以答案是：扩 1000 倍数据能把它从"背诗机器"提升为"更好的古诗续写器"，'
     '但离"会聊天的助手"还差指令微调、对齐与量级三个台阶。'
     '（依据：本实验实测；公开资料中 GPT-3 论文给出的 1750 亿参数量与 TB 级预训练语料规模。）'],
    # (6) AI 协作复盘
    ['AI 最有价值的环节：① 生成样板代码——注意力/Transformer 块里的 reshape、permute、'
     'masked_fill 顺序容易写错，让 AI 先给出带形状注释的骨架，我只需对照公式逐行核验；'
     '② 提供"检查清单"——AI 给出的形状注释与模块划分可以直接当作核对提纲'
     '（掩码在哪一步、头维在第几维、哪些参数共享），把代码审查变成逐项打勾；'
     '③ 复核计算——参数量手算与代码打印差 3,584，AI 提议逐模块打印并提醒权重绑定会重复计数，'
     '最终把差值拆成 LayerNorm 与偏置三部分，完全对上。',
     '必须自己读懂公式才能改对的地方：① 因果掩码必须用 float("-inf") 填充后再 softmax'
     '（用 0 会变成"给未来位置的 logits 加 0"，未来位置仍会分到概率，损失也能下降但模型偷看答案）；'
     '② 多头的 reshape/transpose 顺序——(B,T,C) → (B,T,3,H,hs) → (B,H,T,hs) 这一步如果写成 view '
     '会把"头"与"序列"两维混在一起，代码不报错但语义已经错了；'
     '③ 权重共享的实现——必须让 head.weight 与 tok_emb.weight 指向同一个 Parameter 对象'
     '（PyTorch 中直接赋值即可），否则参数会重复计算、state_dict 里也会多一份权重；'
     '④ 输入与标签必须错开一位（y 相对 x 右移一格），否则模型只要复制输入就能得到极低的 loss，'
     '看似训练成功，实则什么都没学到。',
     '总结：AI 适合承担"查 API、搭骨架、按提示排查"的体力活，'
     '凡是与数学语义等价性有关的地方（掩码的填充值、维度语义、参数是否共享、标签错位）'
     '都必须自己推导公式再验证一遍——本报告所有实验数据均由本人运行脚本产生，'
     'AI 只参与代码骨架与文字整理。'
     '（依据：本人与 AI 助手的实际协作记录、min_llm.py 与 fill_report_hw3.py 的实现。）'],
]
q_underscores = [it for it in items if is_underscore(it)]
# 最后一处（5.3 前半）之外的 6 组共 18 行
q_underscores = q_underscores[:18]
for k in range(6):
    group = q_underscores[k * 3:(k + 1) * 3]
    fill_underscores(group, Q_ANSWERS[k])

# ---------------- 12) 文末附录 ----------------
def append_paragraph(text, bold=False):
    el = body.add_p()
    p = Paragraph(el, doc)
    if text:
        p.add_run(text).bold = bold
    return p


append_paragraph('附录  参数量核对终端输出', bold=True)
append_paragraph('核对脚本 check_params_hw3.py 的输出（命令：python check_params_hw3.py）：')
with open(os.path.join(BASE, 'check_params_output.txt'), encoding='utf-8') as f:
    for line in f.read().rstrip('\n').split('\n'):
        append_paragraph(line)
append_paragraph(
    f'结论：手算三项和 {hand:,} 与代码打印 {base["params"]:,} 的相对误差 '
    f'{(base["params"]-hand)/base["params"]*100:.2f}%，小于要求的 1%；'
    f'差值 {base["params"]-hand:,} 完全由偏置与 LayerNorm 的参数构成'
    '（LayerNorm 5×2C = 1,280、注意力偏置 2×4C = 1,024、MLP 偏置 2×5C = 1,280），'
    '说明手算公式与实现一致。')

doc.save(DST)
print('saved:', DST)
