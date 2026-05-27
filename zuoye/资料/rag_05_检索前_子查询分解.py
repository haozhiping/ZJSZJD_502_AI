import os

from openai import OpenAI


def decompose_query(original_query, num_subqueries=4, model="qwen-plus"):
    """
    将复杂的查询分解为更简单的子查询。

    参数:
        original_query (str): 原始的复杂查询
        num_subqueries (int): 生成的子查询数量
        model (str): 用于查询分解的模型

    返回:
        List[str]: 一个包含简单子查询的列表
    """
    # 中文系统提示，指导AI助手的行为
    system_prompt = "你是一名擅长将复杂问题拆解为简单子问题的AI助手。你的任务是把复杂的用户查询分解为若干个更简单、聚焦不同方面的子问题，所有子问题的答案合起来可以完整回答原始问题。"

    # 中文用户提示，包含待分解的原始查询
    user_prompt = f"""
        请将下列复杂查询拆解为 {num_subqueries} 个更简单的子问题。每个子问题应关注原始问题的不同方面。

        原始查询：{original_query}

        请生成 {num_subqueries} 个子问题，每行一个，格式如下：
        1. [第一个子问题]
        2. [第二个子问题]
        以此类推……
        """

    # 使用指定的模型生成子查询
    client = OpenAI(
        api_key=os.getenv('DASHSCOPE_API_KEY'),  # 替换为你的API Key
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,  # 稍高的温度以获得一些变化
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 处理回复以提取子查询
    content = response.choices[0].message.content.strip()

    # 使用简单的解析方法提取编号的查询
    lines = content.split("\n")
    sub_queries = []

    for line in lines:
        if line.strip() and any(line.strip().startswith(f"{i}.") for i in range(1, 10)):
            # 移除编号和前导空格
            query = line.strip()
            query = query[query.find(".") + 1:].strip()
            sub_queries.append(query)

    return sub_queries


query_str = "大模型的应用场景有哪些？"
print(decompose_query(query_str))

# 打印结果： ['大模型在自然语言处理领域有哪些典型应用场景？', '大模型在计算机视觉和多模态任务中有哪些实际应用？', '大模型在行业垂直领域（如医疗、金融、教育）中解决了哪些具体问题？', '大模型在软件开发、科研辅助等生产力工具场景中有哪些代表性用途？']
