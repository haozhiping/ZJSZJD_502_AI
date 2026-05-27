import re
import jieba
jieba.initialize()  # 预加载词典，就不会在控制台中出现警告
import warnings

# 忽略 jieba 的 pkg_resources 警告
warnings.filterwarnings("ignore", category=UserWarning, module="jieba")

# ========== 1. 更完善的中文停用词 ==========

_CHINESE_STOPWORDS = set("""
的 了 和 是 就 都 而 及 与 着 或 一个 没有 我们 你们 他们 它们 自己 这 那 这些 那些
此 彼 这里 那里 哪里 什么 谁 哪 怎么 怎样 如何 为什么 因为 所以 因此 如果 但是
虽然 然而 那么 这样 那样 之 其 该 各 每 某 些 上 下 里 中 内 外 前 后 左 右 间 边
面 头 部 方 面 种 类 个 位 件 条 项 份 把 只 支 张 本 台 部 套 种 层 片 块 段 篇 章
节 回 集 期 卷 册 本 号 码 页 行 字 句 词 项 款 条 则 点 种 类 般 样 式 型 化 性
学 家 员 者 人 士 师 生 工 农 商 官 兵 警 民 众 群 党 派 系 组织 机构 单位 部门
公司 企业 集团 厂 店 所 院 校 社 会 国 省 市 县 区 乡 镇 村 街 路 号
""".split())


def to_keywords(input_string, stopwords=None):
    """将句子转成检索关键词序列"""
    if stopwords is None:
        stopwords = _CHINESE_STOPWORDS

    # 搜索引擎模式分词，召回更全
    word_tokens = jieba.cut_for_search(input_string)

    # 过滤：停用词 + 纯标点 + 纯数字 + 单字（可选）
    filtered = []
    for w in word_tokens:
        w = w.strip()
        if (len(w) > 1 or w.encode('utf-8').isalpha()) \
                and w not in stopwords \
                and not re.match(r'^[\W\d_]+$', w):
            filtered.append(w)

    return ' '.join(filtered)


# ========== 2. 更精确的断句 ==========

_SENTENCE_END_PATTERN = re.compile(r'(?<=[。！？；?!…])\s*')

def sent_tokenize(input_string, min_length=10):
    """
    按标点断句，支持省略号、过滤过短句子
    """
    if not input_string or not input_string.strip():
        return []

    # 先规范化：统一换行、去除多余空格
    text = input_string.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)

    # 按结束标点切分
    raw_sentences = _SENTENCE_END_PATTERN.split(text)

    sentences = []
    for s in raw_sentences:
        s = s.strip()
        if len(s) >= min_length:
            sentences.append(s)
        elif s and sentences:
            # 短句合并到前一句（保持语义连贯）
            sentences[-1] += s

    return sentences


# ========== 3. 核心优化：按 token/字符 双保险分块 ==========

def split_text(paragraphs, chunk_size=300, overlap_size=100,
               min_line_length=10, by_token=False):
    """
    交叠分块，支持按字符数或估算 token 数切分

    参数：
        paragraphs: 段落列表（字符串或字符串列表）
        chunk_size: 每个 chunk 的最大长度
        overlap_size: 相邻 chunk 的重叠长度
        min_line_length: 句子最小长度
        by_token: 是否按 token 估算（中文 1 字 ≈ 1 token，英文按空格分词）
    """

    # 统一输入格式
    if isinstance(paragraphs, str):
        paragraphs = [paragraphs]

    # 提取所有句子
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
        """计算长度：字符数 或 估算 token 数"""
        if by_token:
            # 简单估算：中文按字，英文按空格分词
            return len(jieba.lcut(text))  # 更准但慢；或简单用 len(text.split())
        return len(text)

    chunks = []
    i = 0

    while i < len(sentences):
        # ========== 当前 chunk 从第 i 句开始 ==========
        current_chunk = sentences[i]

        # ========== 向前找重叠（优化：确保一定有 overlap） ==========
        overlap = []
        overlap_len = 0
        prev = i - 1

        while prev >= 0:
            sent = sentences[prev]
            sent_len = _length(sent)

            # 优先整句纳入，如果单句就超了则跳过
            if overlap_len + sent_len <= overlap_size:
                overlap.insert(0, sent)
                overlap_len += sent_len + 1  # +1 为连接空格
                prev -= 1
            else:
                break

        # 组装：overlap + 当前句
        if overlap:
            current_chunk = ' '.join(overlap) + ' ' + current_chunk

        # ========== 向后扩展 chunk ==========
        next_idx = i + 1
        while next_idx < len(sentences):
            candidate = current_chunk + ' ' + sentences[next_idx]
            if _length(candidate) <= chunk_size:
                current_chunk = candidate
                next_idx += 1
            else:
                break

        chunks.append(current_chunk.strip())

        # 下一个 chunk 的起始位置
        # 优化：确保前进，避免死循环
        advance = max(1, next_idx - i - len(overlap))
        i = i + advance if overlap else next_idx

    return chunks


# ========== 4. 进阶：语义感知的分块（可选） ==========

def semantic_split(text, max_chunk_size=300, overlap_size=100):
    """
    尝试按语义段落切分（通过空行、标题等结构特征），
    再对每个语义块做交叠细切分
    """
    # 按空行分割语义段落
    paragraphs = re.split(r'\n\s*\n', text.strip())

    all_chunks = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 如果段落本身在限制内，直接作为一个 chunk
        if len(para) <= max_chunk_size:
            all_chunks.append(para)
        else:
            # 段落太长，再用句子级细切分
            all_chunks.extend(
                split_text([para], chunk_size=max_chunk_size, overlap_size=overlap_size)
            )

    return all_chunks


# ========== 测试 ==========

if __name__ == "__main__":
    test_text = """
    自然语言处理是人工智能领域的重要方向。它研究如何让计算机理解人类语言。

    深度学习在 NLP 中取得了巨大突破。Transformer 架构成为了主流。
    大语言模型如 GPT、BERT 等被广泛应用。它们能够完成翻译、摘要、问答等任务。

    检索增强生成（RAG）是一种新兴技术。它将外部知识库与大模型结合，减少幻觉问题。
    """

    print("=" * 50)
    print("【基础分块】")
    chunks = split_text(test_text, chunk_size=80, overlap_size=30)
    for idx, c in enumerate(chunks, 1):
        print(f"\n--- Chunk {idx} ({len(c)}字) ---")
        print(c[:100] + "..." if len(c) > 100 else c)

    print("\n" + "=" * 50)
    print("【语义分块】")
    semantic_chunks = semantic_split(test_text, max_chunk_size=80, overlap_size=30)
    for idx, c in enumerate(semantic_chunks, 1):
        print(f"\n--- Chunk {idx} ({len(c)}字) ---")
        print(c[:100] + "..." if len(c) > 100 else c)

    print("\n" + "=" * 50)
    print("【关键词提取】")
    kw = to_keywords("自然语言处理是人工智能的重要方向，深度学习取得了巨大突破。")
    print(kw)