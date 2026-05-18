# 导入：用LangGraph的create_react_agent
from langgraph.prebuilt import create_react_agent
import os
import dotenv
from tools import get_weather, write_file
from langchain.chat_models import init_chat_model

dotenv.load_dotenv(override=True)

# 初始化大语言模型
llm = init_chat_model(
    "deepseek-chat",
    model_provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 工具列表
tools = [get_weather, write_file]

# 创建智能体
agent = create_react_agent(model=llm, tools=tools)

# 测试运行
if __name__ == "__main__":
    response = agent.invoke({
        "messages": [("user", "查询北京的天气，并且把结果写入文件")]
    })

    print("最终结果：", response["messages"][-1].content)