from langchain_ollama import OllamaEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import DashScopeEmbeddings
import os
import dotenv
dotenv.load_dotenv(override=True)

def build_prompt(question: str):
    """
    使用向量检索技术查找相关文档，并通过 LangChain PromptTemplate 构造提示词。

    参数:
        question (str): 用户提出的问题。

    返回:
        str: 构造完成的提示词字符串。
    """
    # 初始化 Embedding 模型
    embedding = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.getenv("QWEN_API_KEY"),
    )

    # Redis 配置
    config = RedisConfig(
        index_name="faq",
        redis_url="redis://localhost:6379",
    )

    # 创建 Redis 向量存储实例
    vector_store = RedisVectorStore(embedding, config=config)

    # 创建检索器，取 2 个最相关文档
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    documents = retriever.invoke(question)

    # 组装 context
    context = "\n\n".join(
        f"【文档片段{i + 1}】\nQ: {doc.page_content}\nA: {doc.metadata.get('answer', '')}"
        for i, doc in enumerate(documents)
    )

    # 定义 Prompt 模板
    template = """
    你是一个外卖公司的智能客服，接下来你将扮演一个专业客服的角色，
    对用户提出来的商品问题进行回答，一定要礼貌热情，如果用户提问与客服和商品无关的问题，
    礼貌委婉的表示拒绝或无法回答，只回答外卖服务相关的问题。
    
    用户问题：
    {question}
    
    可用文档片段：
    {context}
    
    请基于以上信息，生成简洁明了的回答：
    """
    prompt_template = PromptTemplate(
        input_variables=["question", "context"], template=template
    )

    # 渲染提示词
    prompt = prompt_template.format(question=question, context=context)

    print("=== 提示词 ===")
    print(prompt)
    print("=================================")
    return prompt


if __name__ == "__main__":
    build_prompt("在线支付取消订单后钱怎么返还给我呢")