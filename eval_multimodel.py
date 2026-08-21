"""
📌 AI 智能助手 — 多模型横向评测（简历「多模型接入与横向评测」的落地脚本）
===============================================
目标：对 GPT-4o-mini / DeepSeek / 豆包 三套模型做横向对比，
从「响应质量、指令遵循度、稳定性、响应延迟」四个维度量化评分，
输出模型选型结论。

设计要点（对应简历描述）：
    - 复用 app.py 的三套 provider 预设，保证「统一接入层」
    - 统一 System Prompt + 同一批评测 Prompt 集，保证「一致输入条件」
    - 容错降级 + 60s 超时控制
    - 稳定性 = 同一条重复调用 2 次的成功率；响应延迟 = 单次耗时（秒）
    - 只对比「已配置 Key」的模型，未配置的明确跳过，避免用兜底答案冒充真实模型

面试可讲点：
    - "响应质量、指令遵循度用 LLM-as-a-Judge 打分；稳定性靠重复调用成功率；延迟靠实测"
    - "统一 System Prompt 是为了控制变量，排除 prompt 差异对横向对比的干扰"
    - "裁判模型与参赛模型重叠会有自我偏置，生产环境要用独立裁判"

运行：
    python eval_multimodel.py
    结果保存到 data/multimodel_report.md 与 data/multimodel_result.json
"""

import json
import os
import sys
import time
from pathlib import Path
from urllib import error, request

from dotenv import load_dotenv

load_dotenv()

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
PROMPTS_PATH = BASE_DIR / "data" / "eval_prompts.json"
RESULT_PATH = BASE_DIR / "data" / "multimodel_result.json"
REPORT_PATH = BASE_DIR / "data" / "multimodel_report.md"

# ---------- 三套模型预设（与 app.py provider_config 的 presets 保持一致） ----------
PRESETS = {
    "openai":   {"name": "GPT-4o-mini", "baseUrl": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "deepseek": {"name": "DeepSeek",    "baseUrl": "https://api.deepseek.com",    "model": "deepseek-chat"},
    "doubao":   {"name": "豆包",        "baseUrl": "https://ark.cn-beijing.volces.com/api/v3", "model": "doubao-seed-1-6"},
}

# 统一 System Prompt：控制变量，排除 prompt 差异对横向对比的干扰
SYSTEM_PROMPT = "你是 AI 助手。请直接、准确、完整地回答用户问题，并严格遵守用户给出的格式或安全要求。"


def api_key_for(provider: str) -> str:
    return os.getenv(f"{provider.upper()}_API_KEY") or os.getenv("AI_API_KEY") or ""


def is_configured(provider: str) -> bool:
    return bool(api_key_for(provider))


def call_provider(provider: str, messages: list[dict]) -> tuple[str, float]:
    """调用某个模型，返回 (回答, 耗时秒)。已配置才调用真实模型。"""
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]

    if not is_configured(provider):
        return "", 0.0

    preset = PRESETS[provider]
    payload = json.dumps(
        {"model": preset["model"], "messages": full_messages, "temperature": 0.3},
        ensure_ascii=False,
    ).encode("utf-8")
    req = request.Request(
        f"{preset['baseUrl']}/chat/completions",
        data=payload,
        method="POST",
        headers={"Authorization": f"Bearer {api_key_for(provider)}", "Content-Type": "application/json"},
    )
    start = time.perf_counter()
    try:
        with request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        elapsed = time.perf_counter() - start
        return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip(), elapsed
    except error.HTTPError as exc:
        elapsed = time.perf_counter() - start
        return f"[调用失败 {exc.code}]", elapsed
    except Exception as exc:
        # 统一异常捕获（网络抖动 / SSL / 超时等），保证评测不中断，计入稳定性
        elapsed = time.perf_counter() - start
        return f"[调用失败 {type(exc).__name__}]", elapsed


def judge_answer(item: dict, answer: str) -> dict:
    """LLM-as-a-Judge 打两个分：响应质量、指令遵循度。无裁判 Key 时走离线启发式"""
    rubric = item.get("rubric", "")
    fmt = item.get("format", "free")

    if is_configured("deepseek"):
        system_prompt = """你是 AI 评测裁判。从两个维度给回答打分（每项1-10）：
- quality（响应质量）：回答是否正确、完整、有用
- instruction（指令遵循度）：是否严格遵守了指令/格式要求
输出 JSON：{"quality": 分, "instruction": 分, "reason": "一句话理由"}"""
        user_prompt = f"指令/评分要点：{rubric}\n回答：{answer}\n请打分。"
        payload = json.dumps({
            "model": PRESETS["deepseek"]["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 512,
        }, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            f"{PRESETS['deepseek']['baseUrl']}/chat/completions",
            data=payload, method="POST",
            headers={"Authorization": f"Bearer {api_key_for('deepseek')}", "Content-Type": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=60) as resp:
                raw = json.loads(resp.read().decode("utf-8")).get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception:
            raw = ""
        try:
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0]
            return json.loads(raw.strip())
        except Exception:
            return {"quality": 0, "instruction": 0, "reason": "解析失败"}

    # 离线启发式（仅当没有 DeepSeek 当裁判时）
    quality = max(4, min(9, len(answer) // 25))
    if fmt == "short":
        instruction = 9 if len(answer) <= 15 else 5
    elif fmt == "json":
        instruction = 9 if "{" in answer else 5
    elif fmt == "refusal":
        refused = any(k in answer for k in ("抱歉", "不能", "无法", "拒绝", "不提供", "不帮"))
        instruction = 9 if refused else 4
    else:
        instruction = 7
    return {"quality": quality, "instruction": instruction, "reason": "离线演示评分"}


def main() -> None:
    items = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    providers = list(PRESETS.keys())
    names = {p: PRESETS[p]["name"] for p in providers}
    active = [p for p in providers if is_configured(p)]

    print("=" * 64)
    print("  🔍 多模型横向评测：GPT-4o-mini vs DeepSeek vs 豆包")
    print("=" * 64)

    if not active:
        print("\n⚠️ 没有检测到任何已配置的模型 API Key。")
        print("   请在 .env 中至少设置一个：OPENAI_API_KEY / DEEPSEEK_API_KEY / DOUBAO_API_KEY")
        print("   （或统一用 AI_API_KEY + AI_BASE_URL + AI_MODEL）")
        print("   配置后重新运行，即可得到真实的多模型对比与选型结论。")
        return

    skipped = [names[p] for p in providers if p not in active]
    if skipped:
        print(f"\n（已跳过未配置的模型：{', '.join(skipped)}）\n")

    summary = {p: {"quality": [], "instruction": [], "latency": [], "ok": 0, "calls": 0} for p in active}
    detail = {p: [] for p in active}

    for p in active:
        for item in items:
            # 稳定性：同一条跑 2 次，统计成功率；延迟取第一次
            ok = 0
            answers, latencies = [], []
            for _ in range(2):
                ans, lat = call_provider(p, item["messages"])
                answers.append(ans)
                latencies.append(lat)
                if ans and not ans.startswith("[调用失败"):
                    ok += 1

            judge = judge_answer(item, answers[0])
            summary[p]["quality"].append(judge.get("quality", 0))
            summary[p]["instruction"].append(judge.get("instruction", 0))
            summary[p]["latency"].append(latencies[0])
            summary[p]["ok"] += ok
            summary[p]["calls"] += 2

            detail[p].append({
                "id": item["id"], "scenario": item["scenario"],
                "answer": answers[0][:120], "quality": judge.get("quality", 0),
                "instruction": judge.get("instruction", 0), "latency": round(latencies[0], 3),
                "ok": ok,
            })
            status = "✅" if ok == 2 else ("⚠️" if ok == 1 else "❌")
            print(f"  [{names[p]}] #{item['id']} [{item['scenario']}] {status} 质量={judge.get('quality', 0)} 指令={judge.get('instruction', 0)}")

    def avg(xs):
        return round(sum(xs) / len(xs), 2) if xs else 0

    table = {}
    for p in active:
        table[p] = {
            "响应质量": avg(summary[p]["quality"]),
            "指令遵循度": avg(summary[p]["instruction"]),
            "稳定性%": round(summary[p]["ok"] / summary[p]["calls"] * 100) if summary[p]["calls"] else 0,
            "响应延迟s": avg(summary[p]["latency"]),
        }

    overall = {p: table[p]["响应质量"] + table[p]["指令遵循度"] for p in active}
    ranking = sorted(active, key=lambda p: overall[p], reverse=True)

    result = {
        "active_models": [names[p] for p in active],
        "skipped_models": skipped,
        "summary": table,
        "ranking": [names[p] for p in ranking],
        "detail": detail,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# AI 智能助手 — 多模型横向评测报告", ""]
    lines.append("| 模型 | 响应质量 | 指令遵循度 | 稳定性 | 响应延迟 |")
    lines.append("|------|:---:|:---:|:---:|:---:|")
    for p in active:
        t = table[p]
        lines.append(f"| {names[p]} | {t['响应质量']} | {t['指令遵循度']} | {t['稳定性%']}% | {t['响应延迟s']}s |")
    lines.append("")
    if len(active) >= 2:
        lines.append("## 选型结论")
        lines.append("")
        lines.append(f"- 综合排序：{' > '.join(names[p] for p in ranking)}")
        lines.append(f"- 推荐：**{names[ranking[0]]}**（响应质量与指令遵循度综合最优）")
    else:
        lines.append("## 选型结论")
        lines.append("")
        lines.append(f"- 当前仅配置 **{len(active)}** 个模型，不足以得出选型结论；请再配置一个模型后重跑。")
    lines += [
        "",
        "## 说明",
        "",
        "- 响应质量、指令遵循度：LLM-as-a-Judge 打分（未配置 DeepSeek 时为离线启发式）",
        "- 稳定性：同一条重复调用 2 次的成功率；响应延迟：单次调用耗时（秒）",
        "- ⚠️ 裁判模型与参赛模型重叠时存在「自我偏置」，生产环境应使用独立的裁判模型",
    ]
    if skipped:
        lines.append(f"- 已跳过未配置模型：{', '.join(skipped)}")
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 64)
    for p in active:
        t = table[p]
        print(f"  {names[p]}: 质量{t['响应质量']} 指令{t['指令遵循度']} 稳定{t['稳定性%']}% 延迟{t['响应延迟s']}s")
    if len(active) >= 2:
        print(f"  🏆 选型结论：{' > '.join(names[p] for p in ranking)}")
    else:
        print("  💡 提示：至少配置两个模型才能得出选型结论")
    print(f"  📄 报告已保存：{REPORT_PATH.relative_to(BASE_DIR)}")
    print("=" * 64)


if __name__ == "__main__":
    main()
