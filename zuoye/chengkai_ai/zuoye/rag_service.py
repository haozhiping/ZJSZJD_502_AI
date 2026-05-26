import os
import math
from typing import List, Dict, Any, Optional
from chromadb.utils import embedding_functions
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from openai import OpenAI
import chromadb


# ==================== 一、文档加载和分割 ====================
def extract_text_from_pdf(filename: str, page_numbers: Optional[List[int]] = None, min_line_length: int = 10) -> List[
    str]:
    """从 PDF 文件中提取文字并智能分段"""
    paragraphs = []
    buffer = ''
    full_text = ''

    # 提取全部文本
    for i, page_layout in enumerate(extract_pages(filename)):
        if page_numbers is not None and i not in page_numbers:
            continue
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                full_text += element.get_text() + '\n'

    # 按空行分隔成原始段落
    lines = full_text.split('\n')
    raw_paragraphs = []
    current_para = []

    for line in lines:
        line = line.strip()
        if len(line) >= min_line_length:
            current_para.append(line)
        elif current_para:
            # 遇到空行或过短的行，保存当前段落
            raw_paragraphs.append(' '.join(current_para))
            current_para = []

    if current_para:
        raw_paragraphs.append(' '.join(current_para))

    # 智能合并：将短句合并到前一个段落（避免语义碎片化）
    merged_paragraphs = []
    for para in raw_paragraphs:
        para = para.strip()
        # 如果段落很短（<100字符）且已有其他段落，则合并到上一个
        if len(para) < 100 and merged_paragraphs and not para.startswith(('第', '章', '节', '1.', '2.', '3.')):
            merged_paragraphs[-1] += ' ' + para
        else:
            # 过滤掉纯数字、纯标点等无意义内容
            if len(para) >= 30 and any(c.isalpha() for c in para):
                merged_paragraphs.append(para)

    print(f"PDF 提取完成，原始段落 {len(raw_paragraphs)} 个，合并后 {len(merged_paragraphs)} 个有效段落")
    return merged_paragraphs


# ==================== 二、嵌入函数配置 ====================
def get_qwen_embedding_function():
    """获取通义千问 Embedding 函数"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("环境变量 DASHSCOPE_API_KEY 未设置")

    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-v3",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


# 初始化 Embedding 函数
qwen_ef = get_qwen_embedding_function()

# ==================== 三、提示词模板 ====================
prompt_template_product = """
    你是一个销售人员，你的任务是用给定的信息来回答用户的问题。
    给定信息：
    {context}
    用户查询：
    {query}
    如果给定的信息不包含用户查询的答案，只需回答"我无法回答你的问题"
    请永远不要输出给定信息中未包含的答案。
"""

prompt_templates = {
    "product": prompt_template_product
}


# ==================== 四、构建提示词 ====================
def build_prompt(prompt_template: str, **kwargs) -> str:
    """将 Prompt 模板赋值"""
    print("kwargs", kwargs)
    inputs = {}
    for k, v in kwargs.items():
        if isinstance(v, list) and all(isinstance(elem, str) for elem in v):
            val = '\n\n'.join(v)
        else:
            val = v
        inputs[k] = val
    return prompt_template.format(**inputs)

# ==================== 六、向量数据库操作类 ====================
class VectorDBConnector:
    """向量数据库连接器，提供文档添加和检索功能"""

    # 不同模型家族的批次大小限制
    BATCH_CONFIG = {
        "Qwen": 10,
        "ZhiPu": 5,
        "Default": None
    }

    def __init__(self, db_path: str, collection_name: str, embedding_fn=None):
        """
        初始化向量数据库

        Args:
            db_path: 数据库持久化路径
            collection_name: 集合名称
            embedding_fn: 嵌入函数，默认为 Qwen embedding
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_fn = embedding_fn or qwen_ef

        # 创建或连接 ChromaDB
        chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = chroma_client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents: List[str], model_family: str = "Qwen") -> int:
        """
        向 collection 中添加文档与向量（支持分批处理）

        Args:
            documents: 文档列表
            model_family: 模型家族名称（用于确定批次大小）

        Returns:
            添加的文档总数
        """
        batch_size = self.BATCH_CONFIG.get(model_family, 10)

        if batch_size is not None:
            print(f"进入了分批处理的分支，批次大小: {batch_size}")
            total_added = 0

            for i in range(math.ceil(len(documents) / batch_size)):
                batch_docs = documents[i * batch_size:(i + 1) * batch_size]
                batch_ids = [f"id_{total_added + j}" for j in range(len(batch_docs))]

                self.collection.add(
                    embeddings=self.embedding_fn(batch_docs),
                    documents=batch_docs,
                    ids=batch_ids
                )

                total_added += len(batch_docs)
                print(f"已添加批次 {i + 1}，当前集合总记录数: {self.collection.count()}")

            return total_added
        else:
            # 不需要分批处理
            self.collection.add(
                embeddings=self.embedding_fn(documents),
                documents=documents,
                ids=[f"id{i}" for i in range(len(documents))]
            )
            return len(documents)

    def search(self, query: str, top_n: int = 2) -> Dict[str, Any]:
        """
        检索向量数据库

        Args:
            query: 查询文本
            top_n: 返回最相关的 N 个结果

        Returns:
            包含 embeddings、documents、distances 的字典
        """
        results = self.collection.query(
            query_embeddings=self.embedding_fn([query]),
            n_results=top_n,
            include=["embeddings", "documents", "distances"]
        )
        return results

    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        return {
            "collection_name": self.collection_name,
            "total_chunks": self.collection.count(),
            "db_path": self.db_path
        }


# ==================== 七、RAG 服务主类 ====================
class RAGService:
    """RAG 问答服务"""

    def __init__(self, db_path: str = "./chroma_data", collection_name: str = "default",
                 model_name: str = "qwen-plus", api_key: str = None):
        """
        初始化 RAG 服务

        Args:
            db_path: 数据库路径
            collection_name: 集合名称
            model_name: LLM 模型名称
            api_key: API Key（可选，优先使用环境变量）
        """
        self.db_connector = VectorDBConnector(db_path, collection_name, qwen_ef)
        self.model_name = model_name
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def upload_and_process_pdf(self, file_path: str, model_family: str = "Qwen") -> Dict[str, Any]:
        """
        上传并处理 PDF 文件

        Args:
            file_path: PDF 文件路径
            model_family: 模型家族

        Returns:
            包含文件名、切片数量等信息的字典
        """
        # 提取文本
        paragraphs = extract_text_from_pdf(file_path)

        if not paragraphs:
            raise ValueError("PDF 文件中未提取到有效文本")

        # 添加到向量数据库
        chunks_count = self.db_connector.add_documents(paragraphs, model_family)

        filename = os.path.basename(file_path)

        return {
            "filename": filename,
            "chunks_count": chunks_count,
            "total_paragraphs": len(paragraphs)
        }

    def query(self, user_query: str, top_n: int = 5, template_name: str = "product") -> Dict[str, Any]:
        """
        执行 RAG 问答

        Args:
            user_query: 用户查询
            top_n: 检索相关文档数量
            template_name: 提示词模板名称

        Returns:
            包含答案和来源的字典
        """
        # 1. 检索相关文档
        search_results = self.db_connector.search(user_query, top_n)

        # 调试：打印检索结果
        print(f"\n=== 检索结果调试 ===")
        print(f"查询: {user_query}")
        print(f"检索到的文档数量: {len(search_results.get('documents', [[]])[0]) if search_results else 0}")
        if search_results and search_results['documents']:
            for i, doc in enumerate(search_results['documents'][0]):
                print(
                    f"文档 {i + 1} (距离: {search_results['distances'][0][i] if search_results['distances'] else 'N/A'}):")
                print(f"  内容预览: {doc[:100]}...")
        print(f"==================\n")

        # 2. 构建提示词
        prompt_template = prompt_templates.get(template_name, prompt_template_product)

        # 过滤掉距离过大的文档（不相关的）
        filtered_docs = []
        if search_results['documents']:
            docs = search_results['documents'][0]
            dists = search_results['distances'][0] if search_results['distances'] else []

            # 只保留距离 < 1.5 的文档（可根据实际情况调整阈值）
            valid_pairs = [(d, dist) for d, dist in zip(docs, dists) if dist < 1.5]

            if valid_pairs:
                # 按距离从小到大排序（越相似越靠前）
                valid_pairs.sort(key=lambda x: x[1])
                filtered_docs = [p[0] for p in valid_pairs[:top_n]]
            else:
                filtered_docs = docs[:top_n]

        prompt = build_prompt(
            prompt_template,
            context=filtered_docs,
            query=user_query
        )

        # 调试：打印 prompt
        print(f"\n=== Prompt 预览 ===")
        print(f"Prompt 长度: {len(prompt)} 字符")
        print(f"Prompt 内容:\n{prompt[:500]}...\n")

        # 3. 调用 LLM
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        answer = response.choices[0].message.content

        # 4. 格式化来源
        sources = []
        for idx, doc in enumerate(filtered_docs):
            dist = search_results['distances'][0][idx] if search_results['distances'] else 0
            sources.append({
                "index": idx + 1,
                "content": doc,
                "distance": float(dist)
            })

        return {
            "answer": answer,
            "sources": sources,
            "query": user_query
        }
