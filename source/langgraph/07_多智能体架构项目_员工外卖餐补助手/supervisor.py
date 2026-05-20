from typing import TypedDict, Annotated, Optional
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from loguru import logger
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm import chat_model
from state import State

def supervisor_node(state: State) -> State | None:
    logger.info(f"[supervisor_node] 当前阶段: {state.phase}, State: {state}")

    if state.phase == "dispatch":
        # 分发阶段 -> 分类问题
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的客服助手，专门负责对用户提出的问题进行分类，并将任务分发给其他Agent执行。
                            如果用户的问题和食谱推荐有关，那就返回recommend。
                            如果用户的问题和外卖问题有关，那就返回customer_service。
                            如果用户的问题和报销发票有关，那就返回reimburse_extract。
                            除了上述选项外，不要返回其他的内容。"""),
            ("human", "用户提出的问题是：{question}")
        ])
        parser = StrOutputParser()
        chain = chat_prompt | chat_model | parser
        task_type = chain.invoke({"question": state.messages})
        logger.info(f"问题分类结果: {task_type}")
        return State(messages=state.messages, type=task_type.strip(), phase="dispatch")
    elif state.phase == "gather":
        # 汇总阶段 -> 整理子Agent结果
        return State(messages=state.messages, type="summary", phase="done")
    else:
        return None
