from typing import List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from apps.agent_flow.src.schemas.schema_agent import Message, MessageType
from libs.python.utils.graph_rag_utils.format_output_resources import RESOURCES_TITLE


async def simple_format_dialog_content(dialog_content: List[Message]) -> str:
    messages = "\n".join(
        [
            f"{d_c.type_message.value}: {d_c.content.strip()}"
            for d_c in dialog_content
            if len(d_c.content.strip()) != 0
        ]
    )
    return messages


async def format_dialog_content_to_lc(
    dialog_content: List[Message],
) -> List[BaseMessage]:
    messages = [
        (
            HumanMessage(content=d_c.content.strip())
            if d_c.type_message == MessageType.HUMAN
            else AIMessage(
                content=await remove_resources_from_agent_answer(
                    agent_answer=d_c.content.strip(),
                    string_from_delete=[
                        RESOURCES_TITLE,
                        "This information is supported by",
                        "This is supported by",
                    ],
                )
            )
        )
        for d_c in dialog_content
        if len(d_c.content.strip()) != 0
    ]

    return messages


async def remove_resources_from_agent_answer(
    agent_answer: str, string_from_delete: List[str]
):
    for sfd in string_from_delete:
        if sfd in agent_answer:
            # Find the starting index of string_from_delete in agent_answer
            start_index = agent_answer.find(sfd)
            # Slice the agent_answer up to the start of string_from_delete
            agent_answer = agent_answer[:start_index]

    return agent_answer
