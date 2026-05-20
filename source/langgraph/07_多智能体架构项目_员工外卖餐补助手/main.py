import asyncio
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from customer_service import customer_service_node
from recommend import recommend_node
from reimburse import reimburse_node
from state import State
from supervisor import supervisor_node

builder = StateGraph(State)
builder.add_node("supervisor_node", supervisor_node)
builder.add_node("recommend_node", recommend_node)
builder.add_node("customer_service_node", customer_service_node)
builder.add_node("reimburse_node", reimburse_node)


# ===== 流程控制 =====
def estimate(state: State) -> str | None:
    if state.phase == "dispatch":  # 任务分发阶段
        if state.type == "recommend":
            return "recommend_node"
        elif state.type == "customer_service":
            return "customer_service_node"
        elif state.type == "reimburse":
            return "reimburse_node"
        return None
    else:  # 结果汇总阶段
        return "END"  # 汇总完成 → END


# START → supervisor(dispatch)
builder.add_edge(START, "supervisor_node")

# supervisor(dispatch) → Agent / supervisor(gather) → END
builder.add_conditional_edges("supervisor_node", estimate, {
    "recommend_node": "recommend_node",
    "customer_service_node": "customer_service_node",
    "reimburse_node": "reimburse_node",
    "END": END
})

# Agent → supervisor(gather)
builder.add_edge("recommend_node", "supervisor_node")
builder.add_edge("customer_service_node", "supervisor_node")
builder.add_edge("reimburse_node", "supervisor_node")

# ===== 编译 & 测试 =====
graph = builder.compile()
graph.get_graph().draw_png('./graph.png')


# 使用异步方式调用
async def main():
    # content1 = await graph.ainvoke({"messages": ["我今天中午该吃什么啊"]})
    # content1["messages"][-1].pretty_print()

    # content2 = await graph.ainvoke({"messages": ["我点的外卖不想要了，我要退钱"]})
    # content2["messages"][-1].pretty_print()
    content3 = await graph.ainvoke({"messages": ["我要报销这张餐补发票"]})
    content3["messages"][-1].pretty_print()

# 运行异步主函数
asyncio.run(main())
