import os
import time
import math
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

# ==========================================
# 辅助函数 1：从 PDF 提取文本段落
# ==========================================
def extract_text_from_pdf(filename, min_line_length=1):
    """
    从 PDF 文件中提取文字并划分为段落。
    这是日常作业的基础组件，直接引入使用。
    """
    paragraphs = []
    buffer = ''
    full_text = ''
    
    # 提取全部文本
    for page_layout in extract_pages(filename):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                full_text += element.get_text() + '\n'
                
    # 按空行分隔，将文本重新组织成段落
    lines = full_text.split('\n')
    for text in lines:
        if len(text) >= min_line_length:
            buffer += (' ' + text) if not text.endswith('-') else text.strip('-')
        elif buffer:
            paragraphs.append(buffer.strip())
            buffer = ''
    if buffer:
        paragraphs.append(buffer.strip())
    return paragraphs

# ==========================================
# 辅助函数 2：构建提示词
# ==========================================
def build_prompt(prompt_template, **kwargs):
    """
    将 Prompt 模板中的变量填充进去。
    """
    inputs = {}
    for k, v in kwargs.items():
        if isinstance(v, list) and all(isinstance(elem, str) for elem in v):
            val = '\n\n'.join(v)
        else:
            val = v
        inputs[k] = val
    return prompt_template.format(**inputs)


# ==========================================
# 核心业务类：RAGService
# ==========================================
class RAGService:
    def __init__(self, db_path="./chroma_data_zuoye", collection_name="enterprise_brain"):
        """
        初始化 RAG 服务，建立 ChromaDB 连接。
        """
        self.db_path = db_path
        self.collection_name = collection_name
        
        # 1. 初始化通义千问的自定义向量化函数 (OpenAIEmbeddingFunction)
        # TODO: 从环境变量获取 DASHSCOPE_API_KEY，并初始化 qwen_ef
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            print("⚠️ 警告: 未找到 DASHSCOPE_API_KEY 环境变量，请配置，否则向量化和 LLM 无法工作！")
            
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.api_key,
            model_name="text-embedding-v3",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        # 2. 连接/创建 ChromaDB PersistentClient
        # TODO: 初始化 Chroma 客户端，获取或创建名为 collection_name 的 collection
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name
        )

    def process_and_add_pdf(self, pdf_path):
        """
        处理 PDF 并将其写入 ChromaDB。
        返回成功解析的片段段落数量。
        """
        # 1. 提取段落
        paragraphs = extract_text_from_pdf(pdf_path)
        if not paragraphs:
            return 0
            
        # 2. 生成唯一的 IDs，防止重复上传冲突
        # TODO: 生成一个与段落数相等的 ids 列表，保证每个 ID 唯一。
        ids = [f"doc_{int(time.time())}_{i}" for i in range(len(paragraphs))]
        
        # 3. 将文档分批添加到 ChromaDB 中（支持分批处理）
        # TODO: 编写将段落批量写入 self.collection 的代码
        batch_size = 10
        for i in range(math.ceil(len(paragraphs) / batch_size)):
            batch_paragraphs = paragraphs[i * batch_size : (i + 1) * batch_size]
            batch_ids = ids[i * batch_size : (i + 1) * batch_size]
            
            self.collection.add(
                embeddings=self.embedding_fn(batch_paragraphs),
                documents=batch_paragraphs,
                ids=batch_ids
            )
            
        return len(paragraphs)

    def search_and_qa(self, query: str, top_n: int = 4):
        """
        根据用户提问进行检索并调用大模型生成回答。
        返回包含回答内容以及参考资料来源的字典：
        {
           "answer": "模型的回答",
           "sources": [{"content": "片段1", "distance": 0.12}, ...]
        }
        """
        # 1. 检索向量数据库
        # TODO: 调用 self.collection.query，传入用户查询的向量化结果，返回前 top_n 个最相关的文档和距离
        results = self.collection.query(
            query_embeddings=self.embedding_fn([query]),
            n_results=top_n,
            include=["documents", "distances"]
        )
        
        # 2. 提取检索出来的文本片段和距离
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        sources = []
        for doc, dist in zip(documents, distances):
            sources.append({
                "content": doc,
                "distance": float(dist)
            })
            
        # 3. 组装 Prompt
        # TODO: 定义一个 Prompt 模板，并使用 build_prompt 进行拼接
        prompt_template = """
        你是一个助手。请严格根据以下给定的参考信息回答用户的问题。
        
        【参考信息】：
        {context}
        
        【用户问题】：
        {query}
        
        请根据给定的参考信息，给出专业、准确、条理清晰的回答。如果参考信息中无法得出答案，请回答“我无法在现有的知识库中找到相关信息”。不要编造任何事实。
        """
        
        context_str = "\n\n".join(documents)
        prompt = build_prompt(prompt_template, context=context_str, query=query)
        
        # 4. 调用通义千问大模型进行回答
        # TODO: 初始化 OpenAI 客户端连接通义千问兼容模式接口，并获取回答
        client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        
        answer = response.choices[0].message.content
        
        return {
            "answer": answer,
            "sources": sources
        }

    def get_db_stats(self):
        """
        获取当前数据库的状态（如当前集合中的总切片数）。
        """
        # TODO: 返回当前集合的 count 数量
        total_chunks = self.collection.count()
        return {
            "total_chunks": total_chunks,
            "collection_name": self.collection_name
        }
