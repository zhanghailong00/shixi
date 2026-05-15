import os
import asyncio
import dotenv

from playwright.async_api import async_playwright

from langchain_community.agent_toolkits import PlayWrightBrowserToolkit

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

# =========================
# 加载环境变量
# =========================
dotenv.load_dotenv()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")


async def main():

    # =========================
    # 启动 Playwright
    # =========================
    playwright = await async_playwright().start()

    # 启动 Chromium 浏览器
    browser = await playwright.chromium.launch(
    channel="chrome",
    headless=False
    )

    # =========================
    # 创建 Toolkit
    # =========================
    toolkit = PlayWrightBrowserToolkit.from_browser(
        async_browser=browser
    )

    # 获取工具
    tools = toolkit.get_tools()

    print("\n===== 已加载工具 =====")

    for tool in tools:
        print(tool.name)

    # =========================
    # 初始化 DeepSeek
    # =========================
    llm = init_chat_model(
        "deepseek-chat",
        model_provider="deepseek",
        api_key=deepseek_api_key
    )

    # =========================
    # 创建 Agent
    # =========================
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "你是一个网页分析助手。"
            "你可以访问网页并总结网页内容。"
        )
    )

    # =========================
    # 用户输入
    # =========================
    user_input = (
        "访问这个网站："
        "https://langchain-doc.cn/v1/python/langchain/agents.html "
        "并帮我总结网站内容"
    )

    print("\n===== 开始执行 =====\n")

    # =========================
    # Agent调用
    # =========================
    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        }
    )

    # =========================
    # 输出结果
    # =========================
    final_msg = result["messages"][-1]

    print("\n===== AI总结 =====\n")

    print(final_msg.content)

    # =========================
    # 关闭浏览器
    # =========================
    await browser.close()
    await playwright.stop()


# =========================
# 程序入口
# =========================
if __name__ == "__main__":
    asyncio.run(main())