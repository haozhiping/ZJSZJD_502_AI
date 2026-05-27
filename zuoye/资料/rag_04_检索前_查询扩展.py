import os

from openai import OpenAI


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

                        后退一步的查询：
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


res = generate_step_back_query("大模型有哪些应用场景")

print(res)

# 打印结果：后退一步的查询：人工智能模型（尤其是大规模预训练模型）在现实世界中有哪些典型应用领域和实际落地场景？