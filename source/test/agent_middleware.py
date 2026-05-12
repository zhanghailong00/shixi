"""
本文件演示了如何使用LangChain的Agent Middleware（智能体中间件）来增强智能体的功能和安全性。
Agent Middleware 是一种强大的机制，可以在智能体的生命周期中插入自定义逻辑，类似于Web开发中的中间件概念。
通过使用中间件，我们可以在智能体执行工具调用、生成回复等关键步骤之前或之后进行拦截和处理，从而实现日志记录、权限控制、输入输出过滤等功能。
在这个示例中，我们将创建一个简单的智能体，并为其添加三个官方。
"""

# ===================== 环境配置 =====================
from dotenv import load_dotenv
import os

load_dotenv(override=True)

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

# ===================== 模型 =====================
from langchain_deepseek import ChatDeepSeek

model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    api_key=deepseek_api_key,
)

# ===================== Tool =====================
from langchain_core.tools import tool


@tool
def read_email() -> str:
    """
    读取邮件内容
    """

    return """
    用户姓名：张三
    手机号：13800138000
    邮箱：test@example.com
    """


@tool
def send_email(content: str) -> str:
    """
    发送邮件
    """

    return f"邮件发送成功：{content}"


# ===================== Middleware =====================
# 注意：
# 某些 langchain 版本中 middleware 仍属于实验API

from langchain.agents.middleware import (
    PIIMiddleware,
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
)

# ===================== 创建 Agent =====================
from langchain.agents import create_agent

agent = create_agent(
    model=model,

    tools=[read_email, send_email],

    middleware=[

    # 邮箱脱敏
    PIIMiddleware(
        pii_type="email",
        strategy="redact",
    ),

    # URL脱敏（可选）
    PIIMiddleware(
        pii_type="url",
        strategy="redact",
    ),

    # 自动摘要
    SummarizationMiddleware(
        model=model,
        max_tokens_before_summary=500,
    ),

    # HITL
    HumanInTheLoopMiddleware(
        interrupt_on={
            "send_email": {
                "allowed_decisions": [
                    "approve",
                    "edit",
                    "reject"
                ]
            }
        }
    ),
]
)

# ===================== 运行测试 =====================
if __name__ == "__main__":

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "读取邮件，然后发送邮件告诉用户已收到信息"
                }
            ]
        }
    )

    print("\n========== Agent 输出 ==========\n")

    print(result)