import asyncio
import json
from typing import Any, Dict
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from loguru import logger
import os
from langchain.chat_models import init_chat_model

# 加载 .env 文件中的环境变量，override=True 表示覆盖已存在的变量
load_dotenv(override=True)

checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "user-001"}}


def load_servers(file_path: str = "mcp.json") -> Dict[str, Any]:
    """
    从指定的 JSON 文件中加载 MCP 服务器配置。

    参数:
        file_path (str): 配置文件路径，默认为 "mcp.json"

    返回:
        Dict[str, Any]: 包含 MCP 服务器配置的字典，若文件中没有 "mcpServers" 键则返回空字典
    """
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        return data.get("mcpServers", {})


async def run_chat_loop() -> None:
    """
    启动并运行一个基于 MCP 工具的聊天代理循环。

    该函数会：
    1. 加载 MCP 服务器配置；
    2. 初始化 MCP 客户端并获取工具；
    3. 创建基于 langchain 的语言模型和代理；
    4. 启动命令行聊天循环；
    5. 在退出时清理资源。

    返回:
        None
    """
    # 1️ 加载服务器配置
    servers_cfg = load_servers()

    # 2️ 初始化 MCP 客户端并获取工具
    mcp_client = MultiServerMCPClient(servers_cfg)
    tools = await mcp_client.get_tools()
    logger.info(f"✅ 已加载 {len(tools)} 个 MCP 工具： {[t.name for t in tools]}")

    # 3 初始化语言模型
    llm = init_chat_model(
        "deepseek-chat",
        model_kwargs={"temperature": 0.7, "max_tokens": 2048}
    )
    # 4 构建LangGraph Agent
    prompt = """
    你是一个智能体，可以调用以下函数：
    1. get_weather(city: str) —— 获取指定地点的天气
    2. fetch(url: str) —— 请求指定 URL 并返回内容网页的内容
    
    请根据用户的自然语言请求，判断是否需要调用函数，并严格按照函数输入格式返回调用指令。
    如果不需要调用函数，就直接回答。
    """
    agent = create_react_agent(model=llm, prompt=prompt, tools=tools, checkpointer=checkpointer)
    # 5. CLI聊天
    logger.info("\n🤖 MCP Agent 已启动，输入 'quit' 退出")
    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() == "quit":
            break
        try:
            result = await agent.ainvoke({"messages": [("user", user_input)]}, config)
            print(f"\nAI: {result['messages'][-1].content}")
        except Exception as exc:
            logger.error(f"\n⚠️  出错: {exc}")

    # 6. 退出会话
    logger.info("🧹 已退出会话，Bye!")


if __name__ == "__main__":
    # 启动异步事件循环并运行聊天代理
    asyncio.run(run_chat_loop())
