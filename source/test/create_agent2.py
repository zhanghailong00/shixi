#流式输出
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

# 工具函数
def search_web(query):
    """搜索网络信息"""
def analyze_data(data):
    """分析数据内容"""
def send_email(content):
    """发送邮件通知"""

# 创建Agent
agent = create_agent(
    model="deepseek:deepseek-chat",
    tools=[search_web, analyze_data, send_email],
    system_prompt="你是一个有帮助的研究助理。"
)

print("AI 回复：", end="", flush=True)

# 🔥 适配 LangGraph 底层的流式输出（唯一正确写法）
for state in agent.stream(
    input={"messages": [{"role": "user", "content": "研究 AI 安全趋势"}]},
    stream_mode="values"  # 核心参数！必须加！
):
    # 读取状态中的最后一条消息
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        # 只打印AI的回复内容
        if isinstance(last_msg, AIMessage) and last_msg.content:
            print(last_msg.content, end="", flush=True)

print()
