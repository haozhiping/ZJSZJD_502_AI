import os
from VectorDB import VectorDBConnector
from tool import extract_text_from_pdf, qwen_ef, build_prompt, prompt_templates, \
    load_Dashcope_llm

# ================= 配置区 =================
DB_PATH = "./chroma_data"
COLLECTION_NAME = "enterprise_kb"
MODEL_NAME = "qwen-plus"
# ==========================================

# 1. 初始化向量数据库连接 (使用 tool.py 中封装的 qwen_ef 作为嵌入函数)
db_connector = VectorDBConnector(
    dbpath=DB_PATH,
    collection_name=COLLECTION_NAME,
    embedding_fn=qwen_ef
)


def process_and_store_pdf(file_path: str) -> int:
    """
    处理上传的 PDF，提取文字并分块存入 ChromaDB
    返回存入的知识切片(chunk)数量
    1. 调用 tool.py 读取 PDF 中的文字。
    2. 调用 VectorDB.py 将文字切片、向量化并存入 ChromaDB 数据库。
    """
    # 提取 PDF 为段落列表
    paragraphs = extract_text_from_pdf(file_path)

    if not paragraphs:
        return 0

    # 存入向量数据库，根据 VectorDB.py 的逻辑，传递 model_family 决定分批处理策略
    db_connector.add_documents(documents=paragraphs, model_family="Qwen")

    return len(paragraphs)


def chat_with_rag(query: str) -> tuple[str, list]:
    """
    执行 RAG 检索问答：检索知识库 -> 构建提示词 -> 请求大模型 -> 返回 (回答文本, 引用来源列表)
    1. 接收用户的提问，调用 VectorDB.py 在数据库中搜索最相关的 3 个段落。
    2. 如果没搜到，直接告诉用户“不知道”。
    3. 如果搜到了，调用 tool.py 将“检索到的段落”和“用户的问题”按照特定的 Prompt 模板组装起来。
    4. 调用 tool.py 里的千问大模型，拿到最终回答，并带上参考来源返回给 main.py。
    """
    # 1. 检索向量库 (Top 3)
    search_results = db_connector.search(query, top_n=3)

    # ChromaDB 的 query 结果默认是在列表中嵌套列表
    documents = search_results["documents"][0] if search_results.get("documents") else []
    distances = search_results["distances"][0] if search_results.get("distances") else []

    # 如果没有检索到任何相关内容
    if not documents:
        return "本地知识库中未检索到相关内容，我无法回答你的问题。", []

    # 2. 构建 Prompt
    # 将检索到的段落拼接为上下文
    context_text = "\n\n".join(documents)
    prompt = build_prompt(
        prompt_templates["docker"],
        context=context_text,
        query=query
    )

    # 3. 调用千问大模型
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("未检测到 DASHSCOPE_API_KEY 环境变量，请先配置。")

    answer = load_Dashcope_llm(model_name=MODEL_NAME, api_key=api_key, prompt=prompt)

    # 4. 组装溯源数据结构以供前端展示
    sources = []
    for doc, dist in zip(documents, distances):
        sources.append({
            "content": doc,
            "distance": float(dist)
        })

    return answer, sources


def get_db_stats() -> dict:
    """
    获取当前向量数据库的统计信息
    """
    total_chunks = db_connector.collection.count()
    return {
        "collection_name": COLLECTION_NAME,
        "total_chunks": total_chunks
    }
