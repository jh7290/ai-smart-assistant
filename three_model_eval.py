"""
📌 AI Evaluator — Step 6：多模型横向评测（三模型版）
===============================================
目标：让【三个真实模型】（DeepSeek / Qwen / Doubao 可配置）回答同一批问题，
用 LLM 当裁判横向对比，量化"谁更好、好在哪"，并检测裁判的"位置偏置"。

这就是业界"模型竞技场"(Chatbot Arena / MT-Bench)的简化实现：
    1. 同一份评测集喂给多个模型
    2. 成对比较(Pairwise) + 绝对打分(Scoring) 两种评测方式
    3. 每个模型对交换位置消除/检测位置偏置(Position Bias)
    4. 统计胜率(Win Rate) + 输出报告

面试可讲点：
    - "我做过多模型横向评测：三模型两两对比 + 胜率统计 + 绝对打分"
    - "我处理过 LLM-as-a-Judge 的位置偏置，做了位置交换一致性校验"

运行：
    python step6_multimodel_eval.py
    结果保存到 data/multimodel_report.md 与 data/multimodel_result.json
"""

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Windows 控制台默认 GBK，打印 emoji 会崩；统一走 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "eval_dataset.json"
RESULT_PATH = BASE_DIR / "data" / "multimodel_result.json"
REPORT_PATH = BASE_DIR / "data" / "multimodel_report.md"

# ---------- 模型配置：模型 A / B / C ----------
# A 默认 DeepSeek；B 默认通义千问 Qwen；C 默认豆包 Doubao
# 每个模型都支持 OpenAI 兼容接口，可在 .env 里覆盖为任意模型
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

MODEL_B_API_KEY = os.getenv("MODEL_B_API_KEY", "")
MODEL_B_BASE_URL = os.getenv("MODEL_B_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_B_NAME = os.getenv("MODEL_B_NAME", "Qwen")
MODEL_B_MODEL = os.getenv("MODEL_B_MODEL", "qwen-plus")

MODEL_C_API_KEY = os.getenv("MODEL_C_API_KEY", "")
MODEL_C_BASE_URL = os.getenv("MODEL_C_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
MODEL_C_NAME = os.getenv("MODEL_C_NAME", "Doubao")
MODEL_C_MODEL = os.getenv("MODEL_C_MODEL", "doubao-seed-1-6")

MODELS = [
    {
        "name": "DeepSeek",
        "api_key": DEEPSEEK_API_KEY,
        "base_url": DEEPSEEK_BASE_URL,
        "model": "deepseek-chat",
        "configured": bool(DEEPSEEK_API_KEY),
        # 未配置 API Key 时，用评测集里的高质量参考回答充当该模型输出（离线演示）
        "fallback": "answer",
    },
    {
        "name": MODEL_B_NAME,
        "api_key": MODEL_B_API_KEY,
        "base_url": MODEL_B_BASE_URL,
        "model": MODEL_B_MODEL,
        "configured": bool(MODEL_B_API_KEY and MODEL_B_BASE_URL),
        # 未配置时用评测集里的弱基线回答，模拟"强模型 vs 弱基线"的对比
        "fallback": "baseline_answer",
    },
    {
        "name": MODEL_C_NAME,
        "api_key": MODEL_C_API_KEY,
        "base_url": MODEL_C_BASE_URL,
        "model": MODEL_C_MODEL,
        "configured": bool(MODEL_C_API_KEY and MODEL_C_BASE_URL),
        # 未配置时也回退到弱基线回答，保证离线也能跑通演示
        "fallback": "baseline_answer",
    },
]

JUDGE_CONFIGURED = bool(DEEPSEEK_API_KEY)


def load_dataset() -> list[dict]:
    """加载外部评测集（与 step3 共用同一份数据，体现"评测集资产化"）"""
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def generate_answer(model: dict, item: dict) -> str:
    """让某个模型回答问题；未配置时回退到评测集内的回答（用于离线演示）"""
    if not model["configured"]:
        return item[model["fallback"]]

    client = OpenAI(api_key=model["api_key"], base_url=model["base_url"])
    response = client.chat.completions.create(
        model=model["model"],
        messages=[
            {"role": "system", "content": "你是 AI 助手，请直接、准确、完整地回答用户问题。"},
            {"role": "user", "content": item["question"]},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def pairwise_judge(question: str, first: str, second: str) -> dict:
    """成对比较：判断 first 与 second 谁更好（返回 winner: first/second/tie）"""
    system_prompt = """你是公正的 AI 评测裁判。请比较两个回答，判断哪个更好。
评判维度：准确性、完整性、清晰度、实用性。
只输出 JSON：{"winner": "first/second/tie", "reason": "一句话理由"}"""

    user_prompt = f"""问题：{question}

【回答1】{first}

【回答2】{second}

哪个更好？"""

    if not JUDGE_CONFIGURED:
        # 离线演示：简单启发式——更长、有分点、标点更丰富的回答更好
        def strength(text: str) -> float:
            points = len(re.findall(r"[。；;.!?]", text)) + text.count("1.") + text.count("2.") + text.count("3.")
            return len(text) + points * 20

        s1, s2 = strength(first), strength(second)
        if abs(s1 - s2) < 20:
            return {"winner": "tie", "reason": "两个回答质量接近"}
        return {"winner": "first" if s1 > s2 else "second", "reason": "内容更完整、结构更清晰"}

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.2,
        max_tokens=512,
    )
    return _parse_json(response.choices[0].message.content, {"winner": "unknown", "reason": "解析失败"})


def scoring_judge(question: str, answer: str) -> dict:
    """绝对打分：给单个回答在 4 个维度上打分，取平均分"""
    system_prompt = """你是 AI 评测专家。从以下维度评分（每项1-10）：
accuracy 准确性、completeness 完整性、clarity 清晰度、usefulness 实用性。
输出 JSON：{"accuracy": 分, "completeness": 分, "clarity": 分, "usefulness": 分, "total": 平均分, "summary": "评价"}"""

    if not JUDGE_CONFIGURED:
        base = _mock_score(answer)
        return {"accuracy": base, "completeness": base, "clarity": base, "usefulness": base,
                "total": base, "summary": "离线演示评分"}

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": f"问题：{question}\n回答：{answer}\n请评分。"}],
        temperature=0.2,
        max_tokens=1024,
    )
    return _parse_json(response.choices[0].message.content, {"total": 0, "summary": "解析失败"})


def _parse_json(raw: str, default: dict) -> dict:
    try:
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        return json.loads(raw.strip())
    except Exception:
        return default


def _mock_score(answer: str) -> int:
    """离线演示：按回答信息量估一个 4-9 分的演示分数（配真实 Key 后由 LLM 裁判打分）"""
    length = len(answer)
    if length < 25:
        return 4
    if length < 60:
        return 6
    if length < 100:
        return 7
    if length < 160:
        return 8
    return 9


def to_winner(judge: dict, first_name: str, second_name: str, swapped: bool = False) -> str:
    """把裁判结果归一化成模型名或 'tie'（兼容正序/反序调用）"""
    winner = judge.get("winner")
    if winner == "tie":
        return "tie"
    if swapped:
        # 反序调用中：first=second_name, second=first_name
        if winner == "first":
            return second_name
        if winner == "second":
            return first_name
        return "tie"
    if winner == "first":
        return first_name
    if winner == "second":
        return second_name
    return "tie"


def process_item(item: dict, names: list[str], pairs: list[tuple[str, str]]):
    """评测单条数据：生成回答 + 两两比较 + 绝对打分（可多线程并行）"""
    answers = {m["name"]: generate_answer(m, item) for m in MODELS}
    item_pairwise, item_bias = [], []
    item_scoring = {n: [] for n in names}
    item_pair_results = []

    # 成对比较：每个模型对都跑正序 + 反序，反序用于检测位置偏置
    for first_name, second_name in pairs:
        judge_ab = pairwise_judge(item["question"], answers[first_name], answers[second_name])
        judge_ba = pairwise_judge(item["question"], answers[second_name], answers[first_name])
        win_normal = to_winner(judge_ab, first_name, second_name, swapped=False)
        win_swapped = to_winner(judge_ba, first_name, second_name, swapped=True)

        item_pairwise.append({
            "id": item["id"], "category": item["category"], "question": item["question"],
            "pair": f"{first_name} vs {second_name}",
            "winner": win_normal,
            "reason": judge_ab.get("reason", ""),
        })
        item_bias.append({
            "id": item["id"], "question": item["question"][:30],
            "pair": f"{first_name} vs {second_name}",
            "normal": win_normal,
            "swapped": win_swapped,
            "consistent": win_normal == win_swapped,
        })

        item_pair_results.append(f"{first_name} vs {second_name}: {win_normal}")

    # 绝对打分：每个模型独立评分
    for m in MODELS:
        item_scoring[m["name"]].append(scoring_judge(item["question"], answers[m["name"]]))

    return item_pairwise, item_bias, item_scoring, item_pair_results


def main() -> None:
    dataset = load_dataset()
    names = [m["name"] for m in MODELS]
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]

    print("=" * 64)
    print(f"  📊 多模型横向评测：{' / '.join(names)}")
    print("=" * 64)

    # 1. 并行逐题评测（每条数据独立，可多线程加速）
    max_workers = min(8, len(dataset))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda item: process_item(item, names, pairs), dataset))

    pairwise, bias_checks = [], []
    scoring = {n: [] for n in names}
    for item, item_result in zip(dataset, results):
        item_pairwise, item_bias, item_scoring, item_pair_results = item_result
        pairwise.extend(item_pairwise)
        bias_checks.extend(item_bias)
        for n in names:
            scoring[n].extend(item_scoring[n])
        print(f"  #{item['id']:<2} [{item['category']}] " + " | ".join(item_pair_results))

    # 2. 统计每个模型的胜/负/平场（只统计该模型参与的比较）
    model_stats = {n: {"wins": 0, "losses": 0, "ties": 0} for n in names}
    for p in pairwise:
        a, b = p["pair"].split(" vs ", 1)
        if p["winner"] == "tie":
            model_stats[a]["ties"] += 1
            model_stats[b]["ties"] += 1
        elif p["winner"] == a:
            model_stats[a]["wins"] += 1
            model_stats[b]["losses"] += 1
        else:
            model_stats[b]["wins"] += 1
            model_stats[a]["losses"] += 1

    # 3. 位置偏置一致性
    consistent = sum(1 for b in bias_checks if b["consistent"])
    bias_rate = consistent / len(bias_checks)

    # 4. 绝对打分平均值
    avg_score = {}
    for n in names:
        totals = [s.get("total", 0) for s in scoring[n]]
        avg_score[n] = round(sum(totals) / len(totals), 2)

    # 5. 结果落盘（JSON 结构：win_rate 按模型名记录胜/负/平场）
    win_rate = {}
    for n in names:
        s = model_stats[n]
        participated = s["wins"] + s["losses"] + s["ties"]
        win_rate[n] = {
            "wins": s["wins"],
            "losses": s["losses"],
            "ties": s["ties"],
            "total": participated,
            "rate": round(s["wins"] / participated, 2) if participated else 0,
        }

    result = {
        "models": names,
        "pairs": [f"{a} vs {b}" for a, b in pairs],
        "win_rate": win_rate,
        "position_bias": {"consistent": consistent, "total": len(bias_checks), "rate": round(bias_rate, 2)},
        "avg_score": avg_score,
        "pairwise": pairwise,
        "bias_checks": bias_checks,
        "scoring": scoring,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # 6. 生成 Markdown 报告
    lines = [
        "# 多模型横向评测报告",
        "",
        f"- 评测模型：{' / '.join(names)}",
        f"- 成对比较：{', '.join(result['pairs'])}",
        f"- 评测集规模：{len(dataset)} 条（{', '.join(sorted({p['category'] for p in pairwise}))}）",
        f"- 评测方式：成对比较(Pairwise) + 绝对打分(Scoring)",
        "",
        "## 一、胜场统计（Win Rate）",
        "",
        "| 模型 | 胜场 | 负场 | 平局 | 参与比较 | 胜率 |",
        "|------|------|------|------|------|------|",
    ]
    for n in names:
        w = win_rate[n]
        lines.append(f"| {n} | {w['wins']} | {w['losses']} | {w['ties']} | {w['total']} | {w['rate']*100:.0f}% |")
    lines += [
        "",
        "## 二、绝对打分（平均分 / 10）",
        "",
        "| 模型 | 平均分 |",
        "|------|--------|",
    ]
    for n in names:
        lines.append(f"| {n} | {avg_score[n]} |")
    lines += [
        "",
        "## 三、位置偏置检测（交换顺序后裁判是否一致）",
        "",
        f"- 一致性：{consistent}/{len(bias_checks)} = {bias_rate*100:.0f}%",
        "- 说明：若一致率低，说明裁判存在位置偏置，需要固定顺序或多次交换取平均。",
        "",
        "## 四、逐题结果",
        "",
    ]
    for p in pairwise:
        lines.append(f"- #{p['id']} [{p['category']}] `{p['pair']}` → **{p['winner']}**（{p['reason']}）")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 64)
    for n in names:
        s = win_rate[n]
        print(f"  🏆 {n}: 胜 {s['wins']} / 负 {s['losses']} / 平 {s['ties']} / 参与 {s['total']}（胜率 {s['rate']*100:.0f}%）")
    avg_str = " | ".join(f"{n} {avg_score[n]}" for n in names)
    print(f"  📈 平均分：{avg_str}")
    print(f"  🎯 位置一致性：{consistent}/{len(bias_checks)}（{bias_rate*100:.0f}%）")
    print(f"  📄 报告已保存：{REPORT_PATH.relative_to(BASE_DIR)}")
    print("=" * 64)


if __name__ == "__main__":
    main()
