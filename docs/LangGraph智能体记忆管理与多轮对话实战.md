# LangGraph智能体记忆管理与多轮对话实战

> 写在前面：做AI Agent开发时，多轮对话的记忆管理是个绕不开的问题。本文从短期记忆、长期记忆、消息裁剪、消息总结四个维度，结合LangGraph的底层API，手把手带你实现一个"记得住事"的智能体。

---

## 一、记忆管理的四种方案

在开始写代码之前，先理清楚LangGraph中处理记忆的几种思路：

| 方案 | 载体 | 适用场景 | 优缺点 |
|------|------|----------|--------|
| **短期记忆** | Checkpointer（MemorySaver/RedisSaver） | 同一线程内的连续对话 | 简单直接，但受上下文窗口限制 |
| **长期记忆** | BaseStore（InMemoryStore/RedisStore） | 跨线程、跨会话的用户画像 | 支持向量检索，但需要手动管理 |
| **消息裁剪** | trim_messages | 快速丢弃旧消息 | 简单可控，但会丢失信息 |
| **消息总结** | LLM生成摘要 | 长对话压缩 | 保留关键信息，但依赖摘要质量 |

---

## 二、短期记忆：让Agent记住刚才聊了啥

短期记忆是最基础的，用`Checkpointer`就能搞定。核心思路是：**同一个thread_id就是同一个会话**。

### 2.1 预构建Agent方式

最简单的方式，几行代码就能跑起来：

```python
import dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
import os

dotenv.load_dotenv()

# 初始化模型
llm = init_chat_model(
    "deepseek-chat", 
    model_provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 关键：传入checkpointer
checkpointer = InMemorySaver()
agent = create_agent(model=llm, tools=[], checkpointer=checkpointer)

# 同一个thread_id = 同一个会话
config = {"configurable": {"thread_id": "user-001"}}

# 第一轮：自我介绍
msg1 = agent.invoke({"messages": [("user", "你好，我叫小蠡，喜欢学习。")]}, config)
msg1["messages"][-1].pretty_print()

# 第二轮：Agent能记住你说的话
msg2 = agent.invoke({"messages": [("user", "我叫什么？我喜欢做什么？")]}, config)
msg2["messages"][-1].pretty_print()
```

**运行结果：**

```
# 第一轮回复
你好，小蠡！很高兴认识你～「蠡」这个字很有深意呢...

# 第二轮回复 - Agent记住了！
根据我们刚刚的对话——
**你叫**：小蠡
**你喜欢**：学习
```

### 2.2 底层API方式

如果需要更灵活的控制，可以用StateGraph自己搭建：

```python
from typing import TypedDict, Annotated
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
import dotenv

dotenv.load_dotenv(override=True)

# 定义状态结构
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 构建图
graph_builder = StateGraph(State)

llm = init_chat_model(
    "deepseek-chat",
    model_provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# 传入checkpointer
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# 多轮对话
config = {"configurable": {"thread_id": "chat-1"}}

msg1 = graph.invoke({"messages": ["你好，我叫二狗，喜欢学习。"]}, config=config)
msg1["messages"][-1].pretty_print()

msg2 = graph.invoke({"messages": ["我叫什么？我喜欢做什么？"]}, config=config)
msg2["messages"][-1].pretty_print()
```

**输出：**

```
# 第一轮
你好，二狗！很高兴认识你，爱学习的你听起来就是个有趣的灵魂。

# 第二轮 - 记住了！
你叫二狗，喜欢学习～ 看来我得记牢这个"学霸认证"啦！
```

---

## 三、长期记忆：跨会话也能认出你

短期记忆有个局限：换了个thread_id就"失忆"了。长期记忆通过`BaseStore`实现**跨线程记忆**，用户换个会话Agent也能认出他。

### 核心实现

```python
import uuid
from typing import TypedDict, Annotated
from langchain_core.runnables import RunnableConfig
from langgraph.constants import END, START
from langgraph.graph import StateGraph, MessagesState, add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langchain.chat_models import init_chat_model
import dotenv

dotenv.load_dotenv()

model = init_chat_model(
    "deepseek-chat",    
    model_provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

class State(TypedDict):
    messages: Annotated[list, add_messages]


def save_memory(store: BaseStore, user_id: str, content: str):
    """保存记忆到Store"""
    namespace = ("memories", user_id)
    store.put(namespace, str(uuid.uuid4()), {"data": content})


def recall_memories(store: BaseStore, user_id: str, query: str, limit: int = 5):
    """从Store中检索相关记忆"""
    namespace = ("memories", user_id)
    memories = store.search(namespace, query=query, limit=limit)
    return [m.value["data"] for m in memories]


def chatbot(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    
    # 检索历史记忆
    query = state["messages"][-1].content
    related_memories = recall_memories(store, user_id, query)
    
    # 把记忆注入系统提示
    system_msg = (
        "你是一个友好的聊天助手。\n"
        f"以下是关于用户的记忆:\n{chr(10).join(related_memories) if related_memories else '暂无'}"
    )
    
    # 保存当前消息到记忆
    save_memory(store, user_id, query)
    
    response = model.invoke(
        [{"role": "system", "content": system_msg}] + state["messages"]
    )
    return {"messages": response}


# 构建图
builder = StateGraph(State)
builder.add_node(chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# 同时使用Checkpointer和Store
checkpointer = InMemorySaver()
store = InMemoryStore()

graph = builder.compile(checkpointer=checkpointer, store=store)


# 测试：两个不同的thread，但同一个user_id
config1 = {"configurable": {"thread_id": "1", "user_id": "1"}}
msg1 = graph.invoke({"messages": [{"role": "user", "content": "我叫钢蛋，喜欢学习。"}]}, config1)
print("第一次回复：")
msg1["messages"][-1].pretty_print()

# 换了thread_id，但user_id相同，Agent还能记得
config2 = {"configurable": {"thread_id": "2", "user_id": "1"}}
msg2 = graph.invoke({"messages": [{"role": "user", "content": "我叫什么？我喜欢做什么？"}]}, config2)
print("第二次回复：")
msg2["messages"][-1].pretty_print()
```

**输出：**

```
第一次回复：
你好，崔亮！很高兴认识你。你提到喜欢学习，这真是一个很棒的兴趣！

第二次回复：
你叫钢蛋，你喜欢学习。
```

关键点：虽然thread_id从"1"变成了"2"，但因为user_id相同，Agent还是记住了用户信息。

---

## 四、消息裁剪：简单粗暴的上下文控制

当对话太长时，最直接的办法就是**砍掉旧消息**。LangGraph提供了`trim_messages`函数来实现这个功能。

```python
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
import dotenv

dotenv.load_dotenv()

model = init_chat_model(
    "deepseek-chat",
    model_provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)


def pre_model_hook(state):
    """
    钩子函数：在调用模型前裁剪消息
    """
    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",           # 保留最新的
        token_counter=count_tokens_approximately,
        max_tokens=300,            # 最多300个token
        start_on="human",          # 从人类消息开始裁剪
        end_on=("human", "tool"),  # 裁剪到人类消息或工具消息
    )
    return {"llm_input_messages": trimmed_messages}


checkpointer = InMemorySaver()
agent = create_react_agent(
    model,
    tools=[],
    pre_model_hook=pre_model_hook,
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "user-001"}}

# 第一轮
msg1 = agent.invoke({"messages": [("user", "你好，我叫迪迦")]}, config)
msg1["messages"][-1].pretty_print()

# 发送多轮消息
like_list = ['唱', '跳', 'rap', '篮球']
for like in like_list:
    agent.invoke({"messages": [("user", f"我喜欢做的事是：{like}")]}, config)

# 查询记忆 - 由于裁剪，可能丢失早期信息
msg2 = agent.invoke({"messages": [("user", "我叫什么？我喜欢做的事是什么？")]}, config)
msg2["messages"][-1].pretty_print()
```

**裁剪策略说明：**

| 参数 | 说明 |
|------|------|
| `strategy="last"` | 保留最新的消息 |
| `max_tokens=300` | 限制总token数 |
| `start_on="human"` | 裁剪时从人类消息开始 |
| `end_on=("human", "tool")` | 裁剪到人类消息或工具消息为止 |

**优缺点：**
- ✅ 简单直接，保证上下文不超限
- ❌ 容易丢失重要信息（比如早期介绍的名字）

---

## 五、消息总结：智能压缩历史

消息总结比裁剪更聪明，它用LLM把旧消息**压缩成摘要**，既控制了token数，又保留了关键信息。

### 核心思路

1. 设定阈值（比如消息数 > 6）
2. 超过阈值时，把旧消息喂给LLM生成摘要
3. 保留：摘要 + 最近几条消息

### 完整实现

```python
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    "deepseek-chat",
    model_provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def create_summary(model, messages):
    """用LLM生成对话摘要"""
    summary_prompt = [
        SystemMessage(content="请将以下对话历史压缩成一段简短的摘要，保留关键信息（如用户姓名、偏好等）。"),
        *messages
    ]
    response = model.invoke(summary_prompt)
    return response.content


# 阈值：消息数超过6条触发总结
summary_threshold = 6


def summary_node(state: State):
    messages = state["messages"]
    msg_count = len(messages)
    
    print(f"\n[总结节点] 当前消息数: {msg_count}")
    
    if msg_count > summary_threshold:
        print(f"触发总结！消息数 {msg_count} > 阈值 {summary_threshold}")
        
        # 保留最近2条，其余总结
        recent_messages = messages[-2:]
        old_messages = messages[:-2]
        
        # 生成摘要
        summary_content = create_summary(model, old_messages)
        print(f"摘要: {summary_content[:100]}...")
        
        # 构建新消息列表
        summarized_messages = [
            SystemMessage(content=f"【历史对话摘要】{summary_content}"),
            *recent_messages
        ]
        
        print(f"消息数变化: {msg_count} -> {len(summarized_messages)}")
        return {"messages": summarized_messages}
    else:
        print(f"未达到阈值，继续累积")
        return state


def chat_node(state: State):
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content="你是一个友好的AI助手。")] + messages
    
    response = model.invoke(messages)
    return {"messages": [response]}


def should_summarize(state: State):
    msg_count = len(state["messages"])
    return "summary" if msg_count > summary_threshold else "chat"


# 构建图
graph_builder = StateGraph(State)
graph_builder.add_node("summary", summary_node)
graph_builder.add_node("chat", chat_node)
graph_builder.add_edge(START, "summary")
graph_builder.add_conditional_edges("summary", should_summarize, ["summary", "chat"])
graph_builder.add_edge("chat", END)

memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)


# 测试
config = {"configurable": {"thread_id": "user-003"}}

# 第一轮
graph.invoke({"messages": [("user", "你好，我叫小蠡，喜欢学习。")]}, config)

# 多轮对话，触发总结
like_list = ['唱', '跳', 'rap', '篮球', '看电影', '打游戏']
for like in like_list:
    graph.invoke({"messages": [("user", f"我喜欢做的事是：{like}")]}, config)

# 查询记忆 - 总结后是否还记得？
msg_final = graph.invoke({"messages": [("user", "我叫什么？我喜欢做的事是什么？")]}, config)
print("\n最终回复:")
print(msg_final["messages"][-1].content)
```

**运行输出：**

```
[总结节点] 当前消息数: 3
未达到阈值，继续累积

[总结节点] 当前消息数: 5
未达到阈值，继续累积

[总结节点] 当前消息数: 7
触发总结！消息数 7 > 阈值 6
摘要: 用户叫小蠡，喜欢学习，还喜欢唱、跳、rap、篮球...
消息数变化: 7 -> 3

最终回复:
你叫小蠡，喜欢学习，喜欢的活动包括唱、跳、rap、篮球...
```

---

## 六、总结对比

| 方案 | 实现复杂度 | 信息保留度 | 适用场景 |
|------|------------|------------|----------|
| 短期记忆 | ⭐ | ⭐⭐⭐ | 简单多轮对话 |
| 长期记忆 | ⭐⭐⭐ | ⭐⭐⭐ | 用户画像、个性化服务 |
| 消息裁剪 | ⭐ | ⭐ | 快速原型、token敏感场景 |
| 消息总结 | ⭐⭐ | ⭐⭐⭐⭐ | 长对话、复杂任务 |

**实际项目建议：**

- 短期对话为主 → 用Checkpointer就够了
- 需要跨会话记忆 → 加上BaseStore
- 对话很长 → 消息总结 + 短期记忆配合使用
- Token预算紧张 → 消息裁剪兜底

---

## 七、踩坑记录

1. **Checkpointer和Store的区别** - Checkpointer保存的是图的运行状态（短期记忆），Store保存的是用户数据（长期记忆），两者要配合使用

2. **消息裁剪的边界** - `start_on="human"`和`end_on`参数很重要，否则可能裁剪掉不完整的消息对

3. **总结阈值的设置** - 太小会频繁总结影响性能，太大又起不到压缩作用，建议根据实际对话长度调整

4. **Store的命名空间** - 用`("memories", user_id)`这样的结构可以实现用户隔离，避免数据串线

---

> **源码获取**：本文所有代码均可在项目仓库的 `05_智能体记忆管理与多轮对话方法.ipynb` 中找到，欢迎star~

如果觉得有帮助，点个赞再走呗 👍
