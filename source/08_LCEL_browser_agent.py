import os
import asyncio
from datetime import datetime

import dotenv

# =========================
# Playwright 原生异步API
# =========================
from playwright.async_api import async_playwright

# =========================
# LangChain Browser Toolkit
# =========================
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit

# =========================
# 2026 推荐统一模型入口
# =========================
from langchain.chat_models import init_chat_model

# =========================
# 2026 推荐 Agent API
# =========================
from langchain.agents import create_agent

# =========================
# LangChain Chain 组件
# =========================
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# =========================
# 加载环境变量
# =========================
dotenv.load_dotenv(override=True)

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")


# =========================
# 网站总结函数
# =========================
async def summarize_website(url: str) -> str:
    """
    访问指定网站并返回内容总结。

    参数:
        url (str): 要访问和总结的网页URL。

    返回:
        str: 网页正文内容总结。
    """

    try:

        # =========================
        # 启动 Playwright
        # =========================
        playwright = await async_playwright().start()

        # =========================
        # 启动 Chrome 浏览器
        # 使用本地Chrome，避免 playwright install 下载失败
        # =========================
        browser = await playwright.chromium.launch(
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            headless=False
        )

        # =========================
        # 创建 Browser Toolkit
        # =========================
        toolkit = PlayWrightBrowserToolkit.from_browser(
            async_browser=browser
        )

        # 获取浏览器工具
        tools = toolkit.get_tools()

        print("\n===== 已加载 Browser Tools =====")

        for tool in tools:
            print(tool.name)

        # =========================
        # 初始化 DeepSeek 在线模型
        # =========================
        llm = init_chat_model(
            "deepseek-chat",
            model_provider="deepseek",
            api_key=deepseek_api_key
        )

        # =========================
        # 创建 Browser Agent
        # =========================
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=(
                "你是一个网页分析助手。"
                "你可以自动访问网页、读取网页内容并总结正文。"
                "请忽略评论区、版权信息、友情链接等无关内容。"
            )
        )

        # =========================
        # 构造用户任务
        # =========================
        user_input = (
            f"访问这个网站：{url} "
            f"并帮我详细总结网页正文内容，"
            f"不要总结评论区、版权信息、友情链接等内容。"
        )

        print("\n===== Browser Agent 开始执行 =====\n")

        # =========================
        # Agent执行
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

        # 获取最终结果
        final_msg = result["messages"][-1].content

        print("\n===== 网站总结完成 =====\n")

        # =========================
        # 关闭浏览器
        # =========================
        await browser.close()
        await playwright.stop()

        return final_msg

    except Exception as e:

        return f"网站访问失败: {str(e)}"


# =========================
# 保存 Markdown 文件
# =========================
def save_file(summary: str) -> str:
    """
    将文本内容保存为 md 文件。

    参数:
        summary (str): 需要保存的内容。

    返回:
        str: 保存成功的文件路径。
    """

    # 生成文件名
    filename = f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    # 写入 Markdown 文件
    with open(filename, "w", encoding="utf-8") as f:

        f.write("# 网页内容总结\n\n")

        f.write(summary)

    return filename


# =========================
# 格式化总结内容
# =========================
async def format_summary(summary: str) -> str:
    """
    对网站总结内容进行 Markdown 美化。

    参数:
        summary (str): 原始总结内容。

    返回:
        str: 格式化后的 Markdown 内容。
    """

    # 初始化格式化模型
    llm = init_chat_model(
        "deepseek-chat",
        model_provider="deepseek",
        api_key=deepseek_api_key
    )

    # Prompt模板
    prompt = ChatPromptTemplate.from_template(
        """
请优化以下网站总结内容，使其更适合 Markdown 文档格式。

要求：
1. 添加标题与小节
2. 使用 Markdown 语法
3. 内容结构清晰
4. 保留核心内容
5. 不要添加虚构信息

原始总结内容：
{summary}

优化后的 Markdown 内容：
"""
    )

    # 输出解析器
    parser = StrOutputParser()

    # 构建Chain
    chain = prompt | llm | parser

    # 执行Chain
    result = await chain.ainvoke(
        {
            "summary": summary
        }
    )

    return result


# =========================
# 主流程
# =========================
async def main():

    # 目标网站
    url = "https://langchain-doc.cn/v1/python/langchain/agents.html"

    print("\n===== 开始网站分析 =====\n")

    # 1️⃣ Browser Agent 总结网站
    summary = await summarize_website(url)

    print("\n===== 原始总结 =====\n")

    print(summary)

    # 2️⃣ LLM优化Markdown格式
    formatted_summary = await format_summary(summary)

    print("\n===== Markdown格式优化完成 =====\n")

    # 3️⃣ 保存文件
    filename = save_file(formatted_summary)

    print(f"\n✅ 文件已保存: {filename}")


# =========================
# 程序入口
# =========================
if __name__ == "__main__":

    asyncio.run(main())