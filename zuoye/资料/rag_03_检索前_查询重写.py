# 1、环境设置
# 首先，我们导入必要的库。
from openai import OpenAI  # OpenAI API 客户端库

## 2、设置OpenAI API客户端
## 我们初始化OpenAI客户端以生成嵌入向量和回复。
import os

# 初始化 OpenAI 客户端，设置基础 URL 和 API 密钥
client = OpenAI(
    api_key=os.getenv('DASHSCOPE_API_KEY'),  # 替换为你的API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
)


# 3. 查询重写
# 该技术通过使查询更加具体和详细来提高检索的精确性。
#
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


query_str = "大模型的应用场景有哪些？"
print(f"对{query_str}进行查询重写后的结果为：{rewrite_query(query_str)}")
# 打印结果： “大型语言模型（LLM）在不同行业和实际业务场景中的典型应用案例有哪些？请重点涵盖自然语言处理（NLP）任务（如智能客服、机器翻译、文本摘要、情感分析、问答系统）、内容生成（如营销文案、新闻稿、代码生成、多模态内容创作）、企业服务（如知识管理、智能办公助手、合同审查、合规风控）、教育科技（如个性化辅导、自动批改、学习路径推荐）、医疗健康（如医学文献理解、临床决策支持、患者问诊辅助）、金融领域（如智能投顾、反欺诈分析、财报解读）等方向，并说明各场景中所依赖的关键技术能力（如指令微调、RAG、Agent架构、多步推理、工具调用）及当前落地面临的挑战（如幻觉、可解释性、数据隐私、算力成本）。”
