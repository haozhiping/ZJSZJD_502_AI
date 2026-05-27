import os
from VectorDB import VectorDBConnector
from tool import extract_text_from_pdf, qwen_ef, build_prompt, prompt_templates, \
    load_Dashcope_llm
from query_optimizer import optimize_query

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
    """
    paragraphs = extract_text_from_pdf(file_path)

    if not paragraphs:
        return 0

    db_connector.add_documents(documents=paragraphs, model_family="Qwen")

    return len(paragraphs)


def _search_and_collect(query: str, top_n: int = 3) -> list[dict]:
    """检索向量库并返回统一格式的结果列表"""
    search_results = db_connector.search(query, top_n=top_n)
    documents = search_results["documents"][0] if search_results.get("documents") else []
    distances = search_results["distances"][0] if search_results.get("distances") else []

    results = []
    for doc, dist in zip(documents, distances):
        results.append({"content": doc, "distance": float(dist)})
    return results


def _merge_deduplicate_results(all_results: list[dict], top_n: int = 10) -> list[dict]:
    """合并并去重多个检索结果，按距离升序排列（越小越相似）"""
    seen = set()
    unique = []
    for r in sorted(all_results, key=lambda x: x["distance"]):
        # 用内容前80个字符作为去重指纹
        fingerprint = r["content"][:80]
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(r)
    return unique[:top_n]


def chat_with_rag(query: str) -> tuple[str, list, dict]:
    """
    执行 RAG 检索问答：
      1. 评估查询并选择优化策略（重写/扩展/分解）
      2. 用优化后的查询检索向量库
      3. 合并去重检索结果
      4. 构建 Prompt 请求大模型
      5. 返回 (回答文本, 引用来源列表, 优化策略信息)

    Returns:
        (answer, sources, optimization_info)
    """
    # 1. 查询优化
    opt_result = optimize_query(query)
    optimized_queries = opt_result["optimized_queries"]

    # 2. 对每个优化后的查询分别检索
    all_results = []
    for q in optimized_queries:
        results = _search_and_collect(q, top_n=3)
        all_results.extend(results)

    # 3. 合并去重，取 Top 8 作为上下文
    merged_results = _merge_deduplicate_results(all_results, top_n=8)

    if not merged_results:
        return (
            "本地知识库中未检索到相关内容，我无法回答你的问题。",
            [],
            opt_result,
        )

    # 4. 构建 Prompt
    documents = [r["content"] for r in merged_results]
    context_text = "\n\n".join(documents)
    prompt = build_prompt(
        prompt_templates["docker"],
        context=context_text,
        query=query,
    )

    # 5. 调用千问大模型
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("未检测到 DASHSCOPE_API_KEY 环境变量，请先配置。")

    answer = load_Dashcope_llm(model_name=MODEL_NAME, api_key=api_key, prompt=prompt)

    # 6. 组装溯源数据
    sources = []
    for i, r in enumerate(merged_results):
        sources.append({
            "content": r["content"],
            "distance": r["distance"],
            "index": i + 1,
        })

    return answer, sources, opt_result


def get_db_stats() -> dict:
    """
    获取当前向量数据库的统计信息
    """
    total_chunks = db_connector.collection.count()
    return {
        "collection_name": COLLECTION_NAME,
        "total_chunks": total_chunks
    }
