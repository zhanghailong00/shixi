import asyncio
import dotenv
from loguru import logger
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain.agents import create_react_agent, create_openai_tools_agent, AgentExecutor
from llm import llm
from state import State
async def recommend_node(state: State) -> State:
    dotenv.load_dotenv()
    # 1️⃣ 加载服务器配置
    mcp_client = MultiServerMCPClient({
        "howtocook-mcp": {
            "transport": "sse",
            "url": "https://dashscope.aliyuncs.com/api/v1/mcps/how-to-cook/sse",
            "headers": {
                "Authorization": "Bearer " + dotenv.get_key(".env", "BAILIAN_API_KEY"),
            }
        }
    })
    # 2️⃣ 初始化 MCP 客户端并获取工具
    tools = await mcp_client.get_tools()
    logger.info(f"✅ 已加载 {len(tools)} 个 MCP 工具： {[t.name for t in tools]}")
    # 3️⃣ 初始化语言模型、提示模板和代理执行器
    prompt = hub.pull("hwchase17/openai-tools-agent")
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    logger.info("执行美食推荐agent")
    result = await agent_executor.ainvoke({"input": state.messages})
    # logger.info(f"美食推荐结果: {result}")
    return State(messages=state.messages + [AIMessage(f"{result['output']}")],
                 type="recommend", phase="gather")