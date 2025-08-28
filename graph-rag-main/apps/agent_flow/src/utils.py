from typing import List, Tuple

from langchain_core.messages import BaseMessage

from apps.agent_flow.src.cat_main_agent.format_messages import (
    format_dialog_content_to_lc,
    # simple_format_dialog_content,
)
from apps.agent_flow.src.schemas.schema_agent import (
    MessageType,
    Message,
)
from libs.python.utils.logger import logger


async def convert_dialogs_to_chat_history_lc(
    dialog_content: List[Message], k: int
) -> Tuple[str, List[BaseMessage]]:
    input_message = ""
    chat_history = []
    if k < 0:
        logger.info(
            f"Slice was set as negative: {k}; "
            f"So dialog_content would be as []; SET k AS POSITIVE!;"
        )
    len_dialog_content_prev = len(dialog_content)
    dialog_content = dialog_content[-k:] if k > 0 else []

    logger.info(
        f"Slice dialog_content: dialog_content = dialog_content[-k:]; "
        f"len from {len_dialog_content_prev} to {len(dialog_content)});"
        # f"Token count: {}"
    )
    if len(dialog_content) != 0:
        if dialog_content[-1].type_message == MessageType.HUMAN:
            input_message = dialog_content[-1].content
            chat_history = await format_dialog_content_to_lc(dialog_content[:-1])
            # chat_history_string = await simple_format_dialog_content(
            #     dialog_content[:-1]
            # )
        else:
            # chat_history_string = await simple_format_dialog_content(dialog_content)
            chat_history = await format_dialog_content_to_lc(dialog_content)
    return input_message, chat_history
