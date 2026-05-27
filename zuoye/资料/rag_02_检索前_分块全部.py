import re
import jieba
import numpy as np
import requests
import json
import warnings
import os

# ====================== 你的原有代码（分块/分词/断句）======================
jieba.initialize()
warnings.filterwarnings("ignore", category=UserWarning, module="jieba")

_CHINESE_STOPWORDS = set("""
的 了 和 是 就 都 而 及 与 着 或 一个 没有 我们 你们 他们 它们 自己 这 那 这些 那些
此 彼 这里 那里 哪里 什么 谁 哪 怎么 怎样 如何 为什么 因为 所以 因此 如果 但是
虽然 然而 那么 这样 那样 之 其 该 各 每 某 些 上 下 里 中 内 外 前 后 左 右 间 边
面 头 部 方 面 种 类 个 位 件 条 项 份 把 只 支 张 本 台 部 套 种 层 片 块 段 篇 章
""".split())

def to_keywords(input_string, stopwords=None):
    if stopwords is None:
        stopwords = _CHINESE_STOPWORDS
    word_tokens = jieba.cut_for_search(input_string)
    filtered = []
    for w in word_tokens:
        w = w.strip()
        if (len(w) > 1 or w.encode('utf-8').isalpha()) and w not in stopwords and not re.match(r'^[\W\d_]+$', w):
            filtered.append(w)
    return ' '.join(filtered)

_SENTENCE_END_PATTERN = re.compile(r'(?<=[。！？；?!…])\s*')

def sent_tokenize(input_string, min_length=10):
    if not input_string or not input_string.strip():
        return []
    text = input_string.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    raw_sentences = _SENTENCE_END_PATTERN.split(text)
    sentences = []
    for s in raw_sentences:
        s = s.strip()
        if len(s) >= min_length:
            sentences.append(s)
        elif s and sentences:
            sentences[-1] += s
    return sentences

def split_text(paragraphs, chunk_size=300, overlap_size=100, min_line_length=10, by_token=False):
    if isinstance(paragraphs, str):
        paragraphs = [paragraphs]
    sentences = []
    for p in paragraphs:
        if isinstance(p, list):
            for item in p:
                sentences.extend(sent_tokenize(str(item), min_line_length))
        else:
            sentences.extend(sent_tokenize(str(p), min_line_length))
    if not sentences:
        return []

    def _length(text):
        if by_token:
            return len(jieba.lcut(text))
        return len(text)

    chunks = []
    i = 0
    while i < len(sentences):
        current_chunk = sentences[i]
        overlap = []
        overlap_len = 0
        prev = i - 1
        while prev >= 0:
            sent = sentences[prev]
            sent_len = _length(sent)
            if overlap_len + sent_len <= overlap_size:
                overlap.insert(0, sent)
                overlap_len += sent_len + 1
                prev -= 1
            else:
                break
        if overlap:
            current_chunk = ' '.join(overlap) + ' ' + current_chunk
        next_idx = i + 1
        while next_idx < len(sentences):
            candidate = current_chunk + ' ' + sentences[next_idx]
            if _length(candidate) <= chunk_size:
                current_chunk = candidate
                next_idx += 1
            else:
                break
        chunks.append(current_chunk.strip())
        advance = max(1, next_idx - i - len(overlap))
        i = i + advance if overlap else next_idx
    return chunks

def semantic_split(text, max_chunk_size=300, overlap_size=100):
    paragraphs = re.split(r'\n\s*\n', text.strip())
    all_chunks = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chunk_size:
            all_chunks.append(para)
        else:
            all_chunks.extend(split_text([para], max_chunk_size, overlap_size))
    return all_chunks

# ====================== 千问模型调用（核心）======================
# 阿里云 千问 API 密钥（请替换为你自己的）
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')


def qwen_embedding(text):
    """千问文本向量 - 使用OpenAI兼容格式"""
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
    headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "text-embedding-v3",
        "input": text,
        "encoding_format": "float"
    }

    resp = requests.post(url, json=data, headers=headers)
    result = resp.json()

    # OpenAI兼容格式的响应结构
    return np.array(result["data"][0]["embedding"])


def qwen_chat(prompt):
    """千问对话生成"""
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "qwen-turbo",
        "input": {"messages": [{"role": "user", "content": prompt}]},
        "parameters": {"temperature": 0.1}
    }
    resp = requests.post(url, json=data, headers=headers)
    return resp.json()["output"]["text"]

# ====================== 最简向量库 ======================
class SimpleVectorDB:
    def __init__(self):
        self.chunks = []
        self.vectors = []

    def add(self, chunk):
        self.chunks.append(chunk)
        self.vectors.append(qwen_embedding(chunk))

    def search(self, query, top_k=2):
        q_vec = qwen_embedding(query)
        scores = [np.dot(q_vec, vec) for vec in self.vectors]
        ranked = sorted(zip(scores, self.chunks), key=lambda x: -x[0])
        return [c for s, c in ranked[:top_k]]

# ====================== RAG 主流程 ======================
def rag_qa(question, knowledge_text):
    # 1. 用你的代码做文本分块（这就是你的代码的作用）
    chunks = semantic_split(knowledge_text, max_chunk_size=300, overlap_size=80)

    # 2. 存入向量库
    db = SimpleVectorDB()
    for c in chunks:
        db.add(c)

    # 3. 检索相关块
    related_chunks = db.search(question)
    context = "\n".join(related_chunks)

    # 4. 构造 Prompt 给千问
    prompt = f"""你是问答助手，请根据下面的参考资料回答问题，不要编造。
参考资料：
{context}

问题：{question}
"""
    return qwen_chat(prompt)

# ====================== 测试 ======================
if __name__ == "__main__":
    # 你的知识库文本
    test_knowledge = """
    自然语言处理是人工智能领域的重要方向。它研究如何让计算机理解人类语言。

    深度学习在 NLP 中取得了巨大突破。Transformer 架构成为了主流。
    大语言模型如 GPT、BERT 等被广泛应用。它们能够完成翻译、摘要、问答等任务。

    检索增强生成（RAG）是一种新兴技术。它将外部知识库与大模型结合，减少幻觉问题。
    """

    # 提问
    q = "RAG 技术能解决什么问题？"

    # RAG 回答
    answer = rag_qa(q, test_knowledge)
    print("问题：", q)
    print("回答：", answer)