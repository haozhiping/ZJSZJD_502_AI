"""
查询优化器模块 - 评估用户查询并选择合适的检索优化策略

支持策略:
  1. 查询重写 (Rewrite) - 使查询更具体详细，提升检索精度
  2. 查询扩展 (Step-back) - 生成更宽泛的查询，获取更广泛上下文
  3. 子查询分解 (Decompose) - 将复杂查询拆解为多个简单子问题

策略选择基于启发式规则，无需额外LLM调用即可完成评估。
"""
import os
import re
from openai import OpenAI


def _get_client():
    """获取 DashScope OpenAI 兼容客户端"""
    return OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


# ========== 1. 查询重写 ==========

def rewrite_query(original_query: str, model: str = "qwen-plus") -> str:
    """
    将用户查询改写得更具体、更详细，补充关键词和概念以提升检索精度。

    Args:
        original_query: 用户原始查询
        model: 用于重写的 LLM 模型

    Returns:
        改写后的查询字符串
    """
    client = _get_client()
    system_prompt = (
        "你是一名擅长优化检索查询的AI助手。"
        "你的任务是将用户的查询改写得更加具体、详细，"
        "并包含有助于检索准确信息的相关术语和概念。"
    )
    user_prompt = f"""
请将下列查询改写为更具体、更详细的表达，补充相关的关键词和概念，以便更好地检索到准确的信息。

原始查询：{original_query}

改写后的查询："""

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


# ========== 2. 查询扩展（后退一步） ==========

def generate_step_back_query(original_query: str, model: str = "qwen-plus") -> str:
    """
    生成更宽泛、更通用的"后退一步"查询，以检索更广泛的背景上下文。

    Args:
        original_query: 用户原始查询
        model: 用于生成的 LLM 模型

    Returns:
        后退一步的查询字符串
    """
    client = _get_client()
    system_prompt = (
        "你是一名擅长检索策略的AI助手。"
        "你的任务是将具体的用户查询改写为更宽泛、更通用的问题，"
        "以便检索到相关的背景信息和更广泛的上下文。"
    )
    user_prompt = f"""
请将下列查询改写为更宽泛、更通用的问题，以便有助于检索相关的背景信息和更广泛的上下文。

原始查询：{original_query}

后退一步的查询："""

    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


# ========== 3. 子查询分解 ==========

def decompose_query(
    original_query: str, num_subqueries: int = 3, model: str = "qwen-plus"
) -> list:
    """
    将复杂的用户查询分解为多个更简单、聚焦的子问题。

    Args:
        original_query: 用户原始查询
        num_subqueries: 生成的子查询数量
        model: 用于分解的 LLM 模型

    Returns:
        子查询字符串列表
    """
    client = _get_client()
    system_prompt = (
        "你是一名擅长将复杂问题拆解为简单子问题的AI助手。"
        "你的任务是把复杂的用户查询分解为若干个更简单、聚焦不同方面的子问题，"
        "所有子问题的答案合起来可以完整回答原始问题。"
    )
    user_prompt = f"""
请将下列复杂查询拆解为 {num_subqueries} 个更简单的子问题。每个子问题应关注原始问题的不同方面。

原始查询：{original_query}

请生成 {num_subqueries} 个子问题，每行一个，格式如下：
1. [第一个子问题]
2. [第二个子问题]
以此类推……"""

    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content.strip()
    lines = content.split("\n")
    sub_queries = []
    for line in lines:
        if line.strip() and any(
            line.strip().startswith(f"{i}.") for i in range(1, 10)
        ):
            query = line.strip()
            query = query[query.find(".") + 1 :].strip()
            sub_queries.append(query)

    return sub_queries if sub_queries else [original_query]


# ========== 4. 查询评估与策略选择 ==========

# 标记复杂查询的关键词/模式
_MULTI_QUESTION_MARKERS = [
    "还有", "以及", "另外", "此外", "同时",
    "第一", "第二", "第三",
    "首先", "其次", "最后", "然后",
    "一方面", "另一方面",
]

# 标记宽泛查询的模式
_VAGUE_PATTERNS = [
    "是什么", "有哪些", "怎么样", "如何", "为什么",
    "介绍一下", "讲讲", "说说", "什么是",
]

# 标记过于具体/技术化的模式
_SPECIFIC_PATTERNS = [
    "版本", "参数", "配置", "命令", "函数",
    "API", "端口", "报错", "错误码", "--",
]


def evaluate_query(query: str) -> dict:
    """
    基于启发式规则评估查询特征，选择合适的优化策略。

    评估维度:
      - 复杂度: 多问句、并列结构
      - 宽泛度: 短且包含开放式疑问词
      - 具体度: 包含技术参数、配置等细节

    Returns:
        {
            "strategy": "rewrite" | "step_back+rewrite" | "decompose",
            "reason": "选择该策略的原因说明",
            "query_length": int,
            "multi_question_score": int,
        }
    """
    query = query.strip()
    length = len(query)

    # 统计问号数量（多问题标记）
    question_mark_count = query.count("？") + query.count("?")
    # 统计并列标记
    conjunction_count = sum(1 for m in _MULTI_QUESTION_MARKERS if m in query)
    multi_score = question_mark_count + conjunction_count

    # 判断是否宽泛
    is_vague = any(p in query for p in _VAGUE_PATTERNS) and length < 30

    # 判断是否过于具体
    is_specific = any(p in query for p in _SPECIFIC_PATTERNS)

    # ---- 决策树 ----
    if multi_score >= 2 or (question_mark_count >= 2 and length > 50):
        strategy = "decompose"
        reason = f"检测到{multi_score}个多问题标记（问号{question_mark_count}个，并列词{conjunction_count}个），查询较长({length}字)，使用子查询分解将复杂问题拆解为子问题分别检索。"
    elif is_specific:
        strategy = "step_back+rewrite"
        reason = f"查询包含具体技术细节（长度{length}字），先后退一步获取更广泛的背景上下文，再重写查询提升精度。"
    elif is_vague:
        strategy = "rewrite"
        reason = f"查询较短({length}字)且宽泛，使用查询重写补充关键词和概念，使查询更具体详细。"
    elif length < 50:
        strategy = "rewrite+step_back"
        reason = f"查询长度适中({length}字)，同时进行查询重写和后退一步扩展，兼顾检索精度和上下文广度。"
    else:
        strategy = "rewrite"
        reason = f"查询长度{length}字，使用查询重写优化检索精度。"

    return {
        "strategy": strategy,
        "reason": reason,
        "query_length": length,
        "multi_question_score": multi_score,
    }


# ========== 5. 统一入口：执行优化策略 ==========

def optimize_query(query: str) -> dict:
    """
    评估查询并执行最优的检索优化策略。

    Returns:
        {
            "original_query": str,           # 原始查询
            "strategy": str,                 # 使用的策略标识
            "strategy_reason": str,          # 策略选择原因
            "optimized_queries": [str],      # 优化后的查询列表（用于向量检索）
            "strategy_details": {            # 各策略的详细输出
                "rewrite": {"rewritten_query": str} | None,
                "step_back": {"step_back_query": str} | None,
                "decompose": {"sub_queries": [str]} | None,
            }
        }
    """
    evaluation = evaluate_query(query)
    strategy = evaluation["strategy"]

    result = {
        "original_query": query,
        "strategy": strategy,
        "strategy_reason": evaluation["reason"],
        "optimized_queries": [],
        "strategy_details": {},
    }

    if strategy == "decompose":
        # 子查询分解：拆成多个子问题分别检索
        sub_queries = decompose_query(query)
        result["optimized_queries"] = sub_queries
        result["strategy_details"]["decompose"] = {
            "description": "将复杂查询拆解为多个子问题分别检索，合并所有结果后回答",
            "sub_queries": sub_queries,
        }

    elif strategy == "step_back+rewrite":
        # 先后退一步，再重写
        step_back = generate_step_back_query(query)
        rewritten = rewrite_query(query)
        result["optimized_queries"] = [step_back, rewritten, query]
        result["strategy_details"]["step_back"] = {
            "description": "生成更宽泛的查询以获取更广泛的上下文",
            "step_back_query": step_back,
        }
        result["strategy_details"]["rewrite"] = {
            "description": "将查询改写得更具体详细以提升检索精度",
            "rewritten_query": rewritten,
        }

    elif strategy == "rewrite+step_back":
        # 同时重写和后退一步
        rewritten = rewrite_query(query)
        step_back = generate_step_back_query(query)
        result["optimized_queries"] = [rewritten, step_back, query]
        result["strategy_details"]["rewrite"] = {
            "description": "将查询改写得更具体详细以提升检索精度",
            "rewritten_query": rewritten,
        }
        result["strategy_details"]["step_back"] = {
            "description": "生成更宽泛的查询以获取更广泛的上下文",
            "step_back_query": step_back,
        }

    else:  # "rewrite"
        rewritten = rewrite_query(query)
        result["optimized_queries"] = [rewritten, query]
        result["strategy_details"]["rewrite"] = {
            "description": "将查询改写得更具体详细以提升检索精度",
            "rewritten_query": rewritten,
        }

    return result
