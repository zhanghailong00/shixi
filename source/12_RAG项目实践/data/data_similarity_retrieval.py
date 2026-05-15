from langchain_ollama import OllamaEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
import os
import dotenv
dotenv.load_dotenv(override=True)

def search_question(question):
    # 初始化 Embedding 模型
    embedding = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.getenv("QWEN_API_KEY"),
    )

    # 配置Redis连接参数和索引名称
    config = RedisConfig(
        index_name="faq",
        redis_url="redis://localhost:6379",
    )

    # 创建Redis向量存储实例
    vector_store = RedisVectorStore(embedding, config=config)


    # 创建检索器，进行数据检索
    retriever = vector_store.as_retriever()
    documents = retriever.invoke(question)

    for document in documents:
        print(document.page_content)
        print(document.metadata)
        print("=================================")

if __name__ == "__main__":
    # 程序入口：先创建索引再批量插入数据
    search_question("在线支付取消订单后钱怎么返还给我呢")