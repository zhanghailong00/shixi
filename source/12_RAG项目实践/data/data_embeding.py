"""
我们将 FAQ 数据格式化成 json 数据后，接下来就要转成向量数据并存储到向量数据库中，此处以 redis 为例，操作内容包括：

使用 向量化模型（Embedding Model，如 BGE、OpenAI Embedding） 将文档片段转换为向量表示。
存储至向量数据库（如 Milvus、Weaviate、Redis Vector、Faiss），支持高效的相似度搜索。
"""
import json
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore
import os
import dotenv
dotenv.load_dotenv(override=True)

def insert_faq(texts, meta_data):
    """
    将FAQ文本数据插入到Redis向量存储中
    
    Args:
        texts (list): 包含问题文本的列表
        meta_data (list): 包含每个问题对应元数据的列表，每个元素为字典格式
        
    Returns:
        None
    """
    # 配置Redis连接参数和索引名称
    config = RedisConfig(
        index_name="faq",
        redis_url="redis://localhost:6379",
    )
    # 初始化 Embedding 模型
    embedding = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.getenv("QWEN_API_KEY"),
    )
    # 创建Redis向量存储实例
    vector_store = RedisVectorStore(embedding, config=config)
    vector_store.add_texts(texts=texts, metadatas=meta_data)


def insert_from_file(file_path):
    """
    从JSON文件中读取FAQ数据并插入到向量存储中
    
    Args:
        file_path (str): 包含FAQ数据的JSON文件路径
        
    Returns:
        None
    """
    with open(file_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    texts = []
    meta_data = []
    # 解析文档数据，提取问题文本和元数据
    for doc in docs:
        texts.append(doc["question"])
        meta_data.append({
            "answer": doc["answer"],
            "category": doc["category"],
            "source": doc["source"]
        })
    insert_faq(texts, meta_data)


if __name__ == "__main__":
    # 程序入口：先创建索引再批量插入数据
    insert_from_file("faq.json")