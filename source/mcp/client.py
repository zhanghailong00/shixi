import asyncio
import json
import os
from dotenv import load_dotenv
from loguru import logger
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model

load_dotenv(override=True)


# =========================
# 读取 MCP 配置
# =========================
def load_servers(file_path="mcp.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)["mcpServers"]


# =========================
# 主程序
# =========================
async def main():

    # 1️⃣ 加载 MCP Server配置
    servers = load_servers()

    # 2️⃣ 创建 MCP Client
    client = MultiServerMCPClient(servers)

    # 3️⃣ 获取 MCP Tools（自动发现）
    tools = await client.get_tools()
    logger.info(f"🧩 已加载工具: {[t.name for t in tools]}")

    # 4️⃣ 初始化 LLM（DeepSeek在线模型）
    llm = init_chat_model(
        "deepseek-chat",
        model_provider="deepseek",
        api_key=os.getenv("DEEPSEEK_API_KEY")  
    )

    # 5️⃣ 创建 Agent（🔥关键替换点）
    agent = create_react_agent(llm, tools)

    # 6️⃣ CLI循环
    logger.info("🤖 MCP Agent 已启动，输入 quit 退出")

    while True:
        query = input("\n你: ").strip()

        if query.lower() == "quit":
            break

        try:
            # 🚀 LangGraph调用方式
            result = await agent.ainvoke({
                "messages": [
                    ("user", query)
                ]
            })

            # 最终输出
            final_msg = result["messages"][-1].content
            print("\nAI:", final_msg)

        except Exception as e:
            logger.error(f"❌ 错误: {e}")


# =========================
# 启动入口
# =========================
if __name__ == "__main__":
    asyncio.run(main())