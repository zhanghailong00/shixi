"""
create_agent源码
1. ReAct 实现
底层：LangGraph 状态图
结构：模型节点（推理） + 工具节点（行动） + 条件循环
流程：推理 → 行动 → 再推理 → 结束
2. 中间件
类型：6 种生命周期钩子
实现：链式拦截，劫持 Agent 所有行为
用途：日志、权限、扩展、监控
3. 记忆 & 持久化
checkpointer：单轮对话记忆（存对话历史）
store：全局持久化（存长期数据）
载体：messages 列表
一句话总结
create_agent = LangGraph 驱动的 ReAct 智能体 + 中间件扩展 + 状态持久化
"""


# 1. 加载环境变量（必备！）
from dotenv import load_dotenv
load_dotenv()
# 作用：读取你项目根目录的 .env 文件，自动拿到 DEEPSEEK_API_KEY
# 没有这两行，模型会报错找不到密钥，官方文档省略了这行（因为是环境配置）

# 2. 导入核心函数：LangChain官方提供的「一键创建Agent」工具
from langchain.agents import create_agent

# 3. 定义Agent可以使用的工具（3个工具函数）
# 🔥 强制要求：每个函数必须加 """文档注释"""，否则直接报错（新版规则）
def search_web(query):
    """搜索网络信息"""  # 注释=工具的功能说明，AI会看这个来决定用不用工具
def analyze_data(data):
    """分析数据内容"""
def send_email(content):
    """发送邮件通知"""

# 4. 创建智能Agent（1:1 复刻你的官方文档）
agent = create_agent(
    model="deepseek:deepseek-chat",    # 模型：格式固定为 厂商:模型名称
    tools=[search_web, analyze_data, send_email],  # 给Agent绑定3个工具
    system_prompt="你是一个有帮助的研究助理。"  # 设定AI的角色
)

# 5. 调用Agent，给它发送问题（完全照搬文档写法）
result = agent.invoke({
    "messages": [  # 输入：对话历史列表（用户问题放在这里）
        {"role": "user", "content": "研究 AI 安全趋势"}
    ]
})

# 6. 🔥 核心：取出并打印最终结果（重点解释！）
print(result["messages"][-1].content)