import collections
import math

import chromadb


# 1、定义一个类：向量数据库的相关操作（集合的相关操作）
class VectorDBConnector:
    def __init__(self,dbpath, collection_name, embedding_fn):
        # 存储在内存中
        # chroma_client = chromadb.Client(Settings(allow_reset=True))
        chroma_client = chromadb.PersistentClient(path=rf"{dbpath}")
        # 创建一个 collection
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name)
        self.embedding_fn = embedding_fn

    # 给向量数据库添加文档（内部自动向量化）
    def add_documents(self, documents,model_family):
        """向 collection 中添加文档与向量"""


        # 需要考虑不同模型家族是否需要分批处理
        # 第一种情况：只区分是否需要分批处理（假定只要分配，必然是10个段落）
        # batch_model_family = ["Qwen", "ZhiPu"]
        # if model_family in batch_model_family:
        #     for i in range(math.ceil(len(documents) / 10)):
        #         self.collection.add(
        #             embeddings=self.embedding_fn(documents),  # 每个文档的向量
        #             documents=documents[i * 10:(i + 1) * 10],  # 文档的原文
        #             ids=[f"id{i}" for i in range(len(documents))]  # 每个文档的 id
        #         )
        # else:
        #     self.collection.add(
        #         embeddings=self.embedding_fn(documents),  # 每个文档的向量
        #         documents=documents,  # 文档的原文
        #         ids=[f"id{i}" for i in range(len(documents))]  # 每个文档的 id
        #     )

        # 第二种情况：模型家族需要分批处理，并且分批段落数不一样
        batch_model_family = {
            "Qwen":10,
            "ZhiPu":5
        }
        num = batch_model_family[model_family]
        if num  is not None:
            print("进入了分批处理的分支，num",10)
            for i in range(math.ceil(len(documents) / num)):
                """向 collection 中添加文档与向量"""
                self.collection.add(
                    embeddings=self.embedding_fn(documents[i * num:(i + 1) * num]),  # 每个文档的向量
                    documents=documents[i * num:(i + 1) * num],  # 文档的原文
                    ids =  [f"id{i*10+j}" for j in range(len(documents[i * num:(i + 1) * num]))]
                )

                print("一共有多少条记录：", self.collection.count())
        else:
            self.collection.add(
                embeddings=self.embedding_fn(documents),  # 每个文档的向量
                documents=documents,  # 文档的原文
                ids=[f"id{i}" for i in range(len(documents))]  # 每个文档的 id
            )

    # 检索
    def search(self, query, top_n):
        """检索向量数据库"""
        results = self.collection.query(
            query_embeddings=self.embedding_fn([query]),
            n_results=top_n,
            include=["embeddings", "documents", "distances"] # 希望返回的结果包含特定的内容（此处：返回的是向量、文档原文、距离）
        )
        return results