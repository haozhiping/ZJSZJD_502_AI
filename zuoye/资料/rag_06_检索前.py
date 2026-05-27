from openai import OpenAI  # OpenAI API 客户端库
import os

# 初始化 OpenAI 客户端，设置基础 URL 和 API 密钥
client = OpenAI(
    api_key=os.getenv('DASHSCOPE_API_KEY'),  # 替换为你的API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
)


# 1. 查询重写
# 该技术通过使查询更加具体和详细来提高检索的精确性。
def rewrite_query(original_query, model="qwen-plus"):
    """
    根据给定的原始查询重新编写查询，使其更加具体和详细，以便更好地检索信息。
    参数:
        original_query (str): 原始用户查询
        model (str): 用于查询重写的模型名称

    返回:
        str: 重写后的查询
    """
    # 中文系统提示，指导AI助手的行为
    system_prompt = "你是一名擅长优化检索查询的AI助手。你的任务是将用户的查询改写得更加具体、详细，并包含有助于检索准确信息的相关术语和概念。"
    # 中文用户提示，包含需要被重写的原始查询
    user_prompt = f"""
                    请将下列查询改写为更具体、更详细的表达，补充相关的关键词和概念，以便更好地检索到准确的信息。

                    原始查询：{original_query}

                    改写后的查询：
                  """

    # 使用指定的模型生成重写后的查询
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,  # 低温度值以确保输出确定性
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 返回重写后的查询，并去除任何前导或尾随空白字符
    return response.choices[0].message.content.strip()

# 2. 查询扩展
def generate_step_back_query(original_query, model="qwen-plus"):
    """
    生成一个更通用的“后退一步”查询，以检索更广泛的上下文。

    参数:
        original_query (str): 原始用户查询
        model (str): 用于生成后退一步查询的模型

    返回:
        str: 后退一步查询
    """
    # 中文系统提示，指导AI助手的行为
    system_prompt = "你是一名擅长检索策略的AI助手。你的任务是将具体的用户查询改写为更宽泛、更通用的问题，以便检索到相关的背景信息和更广泛的上下文。"

    # 中文用户提示，包含需要被泛化的原始查询
    user_prompt = f"""
                        请将下列查询改写为更宽泛、更通用的问题，以便有助于检索相关的背景信息和更广泛的上下文。

                        原始查询：{original_query}

                    """
    client = OpenAI(
        api_key=os.getenv('DASHSCOPE_API_KEY'),  # 替换为你的API Key
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
    )

    # 使用指定的模型生成后退一步查询
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,  # 稍高的温度以获得一些变化
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 返回后退一步查询，并去除任何前导/尾随空白字符
    return response.choices[0].message.content.strip()

# 3. 子查询分解
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


# 示例查询
original_query = "中国政府在AI上有哪些政策?"
print("Original Query:", original_query)

# 查询重写
rewritten_query = rewrite_query(original_query)
print("\n1. Rewritten Query:")
print(rewritten_query)

# 后退提示生成
step_back_query = generate_step_back_query(original_query)
print("\n2. Step-back Query:")
print(step_back_query)

# 子查询分解
sub_queries = decompose_query(original_query, num_subqueries=4)
print("\n3. Sub-queries:")
for i, query in enumerate(sub_queries, 1):
    print(f"   {i}. {query}")