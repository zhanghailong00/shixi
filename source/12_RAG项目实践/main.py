
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_redis import RedisConfig, RedisVectorStore, RedisChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.embeddings import DashScopeEmbeddings
import os
import dotenv
from langchain.chat_models import init_chat_model
dotenv.load_dotenv(override=True)
# ---------- 工具 ----------

def format_docs(docs):
    """
    把检索到的文档格式化成上下文字符串，用于提供给语言模型作为参考信息。

    参数:
        docs (list): 文档对象列表，每个对象应包含 page_content 和 metadata 属性。

    返回:
        str: 格式化后的字符串，包含多个文档片段及其问答内容。
    """
    return "\n\n".join(
        f"【文档片段{i + 1}】\n"
        f"Q: {doc.page_content}\n"
        f"A: {doc.metadata.get('answer', '')}"
        for i, doc in enumerate(docs)
    )


def extract_question(x: str | list) -> str:
    """
    从 RunnableWithMessageHistory 的输入中提取用户的纯文本问题。

    参数:
        x (str | list): 输入可以是字符串或消息对象列表。

    返回:
        str: 提取到的用户问题文本。
    """
    if isinstance(x, str):
        return x
    # x 是 list[HumanMessage]
    return x[-1].content


# ---------- 构建链 ----------

def build_chain():
    """
    构建一个基于检索增强生成（RAG）的对话链，用于智能客服问答。

    返回:
        Chain: 一个可调用的 LangChain 链对象，用于处理用户问题并生成回答。
    """
    # 初始化嵌入模型
    embedding = DashScopeEmbeddings(
        model="text-embedding-v2",
        dashscope_api_key=os.getenv("QWEN_API_KEY")
    )

    # 配置 Redis 向量存储
    config = RedisConfig(index_name="faq", redis_url="redis://localhost:6379")
    vector_store = RedisVectorStore(embedding, config=config)

    # 创建文档检索器，最多返回2个相关文档
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # 定义提示模板
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
    prompt = PromptTemplate.from_template(template)

    # 初始化语言模型和输出解析器
    llm = init_chat_model(
    "deepseek-chat",
    model_provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)
    parser = StrOutputParser()

    # 构建处理链：提取问题 -> 检索文档 -> 格式化上下文 -> 拼接提示 -> 调用模型 -> 解析输出
    chain = (
        {
            "context": extract_question | retriever | format_docs,
            "question": extract_question,
        }
        | prompt
        | llm
        | parser
    )
    return chain


# ---------- 交互 ----------

def main():
    """
    主函数，启动智能客服交互系统。
    """
    # 构建对话链
    chain = build_chain()

    # 初始化 Redis 聊天历史记录
    history = RedisChatMessageHistory(session_id='rag', redis_url='redis://localhost:6379/0')

    # 将对话链包装为带历史记录的可运行对象
    runnable = RunnableWithMessageHistory(
        chain,
        get_session_history=lambda: history
    )

    # 启动交互循环
    print(">>> 欢迎使用外卖智能客服系统，输入 quit/exit 退出 <<<")
    while True:
        try:
            user = input("\n您：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye~")
            break
        if user.lower() in {"quit", "exit", "q"}:
            print("客服：祝您生活愉快，再见！")
            break
        answer = runnable.invoke(user)      # 自动把 user 包装成 HumanMessage
        print("客服：", answer)


if __name__ == "__main__":
    main()
