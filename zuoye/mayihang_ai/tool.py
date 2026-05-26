import os
from chromadb.utils import embedding_functions
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from openai import OpenAI

# 一、文档加载和分割
def extract_text_from_pdf(filename, page_numbers=None, min_line_length=1):
    """从 PDF 文件中（按指定页码）提取文字"""
    paragraphs = []
    buffer = ''
    full_text = ''
    # 提取全部文本
    for i, page_layout in enumerate(extract_pages(filename)):
        # 如果指定了页码范围，跳过范围外的页
        if page_numbers is not None and i not in page_numbers:
            continue
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                full_text += element.get_text() + '\n'
    # 按空行分隔，将文本重新组织成段落
    lines = full_text.split('\n')
    for text in lines:
        if len(text) >= min_line_length:
            buffer += (' ' + text) if not text.endswith('-') else text.strip('-')
        elif buffer:
            paragraphs.append(buffer)
            buffer = ''
    if buffer:
        paragraphs.append(buffer)
    return paragraphs

# 二、自定义嵌入函数
qwen_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model_name="text-embedding-v3",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 封装提示词模板
prompt_template_tianwen = """
    你是一个国家天文台的专家，你的任务是用给定的信息来回答用户的问题。
    给定信息：
    {context}
    用户查询：
    {query}
    如果给定的信息不包含用户查询的答案，只需回答“我无法回答你的问题”
    请永远不要输出给定信息中未包含的答案。
"""

prompt_template_docker = """
    你是一个docker容器技术的专家，你的任务是用给定的信息来回答用户的问题。
    给定信息：
    {context}
    用户查询：
    {query}
    如果给定的信息不包含用户查询的答案，只需回答“我无法回答你的问题”
    请永远不要输出给定信息中未包含的答案。
"""

prompt_templates ={
    "tianwen":prompt_template_tianwen,
    "docker":prompt_template_docker
}


# 三、构建提示词
def build_prompt(prompt_template, **kwargs):
    print("kwargs",kwargs)
    """将 Prompt 模板赋值"""
    inputs = {}
    for k, v in kwargs.items():
        if isinstance(v, list) and all(isinstance(elem, str) for elem in v):
            val = '\n\n'.join(v)
        else:
            val = v
        inputs[k] = val
    return prompt_template.format(**inputs)


# 四、调用大模型
# 1）、调用任意模型
def load_llm(model_name,base_url,api_key,prompt,max_tokens=1024):
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    # 2）、调用
    response = client.chat.completions.create(
        model=model_name,
        messages=prompt,
        stream=False,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

# 2）、调用千问的模型
def load_Dashcope_llm(model_name,api_key,prompt:str, max_tokens=1024):
    api_key = os.getenv("DASHSCOPE_API_KEY",api_key)
    client = OpenAI(
        api_key= api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    # 2）、调用
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        stream=False,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def load_Dashcope_llm02(model_name, api_key, prompt: str):
    api_key = os.getenv("DASHSCOPE_API_KEY", api_key)
    return load_llm(
        model_name=model_name,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
        prompt=[
            {"role": "user", "content": prompt}
        ]
    )