from typing import List

from pydantic import BaseModel, Field


class References(BaseModel):
    """
    !!!GENERAL RULES FOR `ids_reports` OR `ids_entities` or `ids_relationships` Fields!!!

    1. !!!REMEMBER!!! References is obligatory if ---Data tables--- exists!!! IF ---Data tables--- does not exist then do not provide `References`!!!
    2. !!!If you don't know the answer, just say so, Do not make anything up! AND SET `ids_reports` OR `ids_entities` or `ids_relationships` as empty List!!!
    3. DO NOT MIX record `IDS` between datasets!!!
    4. DO NOT INCLUDE TO References record `id` THAT JUST MENTION IN THE CONTENT FROM ANOTHER DATASET FIELD.
    For example, DO NOT INCLUDE record `id` TO `ids_entities` and `ids_relationships` from `content` column from <dataset name> = -----Reports-----!!!
    """

    ids_reports: List[int] = Field(
        description="""
List of record `id` of `Reports` table. 
`id` -- First column `id` in <dataset name> = -----Reports-----. 
USE `id` ONLY FROM FIRST COLUMN in <dataset name> = -----Reports-----!!!
"""
    )
    ids_entities: List[int] = Field(
        description="""
List of record `id` of `Entities` table. 
`id` -- First column `id` in <dataset name> = -----Entities-----. 
USE `id` ONLY FROM FIRST COLUMN in <dataset name> = -----Entities-----!!!
"""
    )
    ids_relationships: List[int] = Field(
        description="""
List of record `id` of `Relationships` table. 
`id` -- First column `id` in <dataset name> = -----Relationships-----. 
USE `id` ONLY FROM FIRST COLUMN in <dataset name> = -----Relationships-----!!!
"""
    )


class MainAgentFinishOutput(BaseModel):
    """
    Use this as FINISH AGENT ANSWER!!!

    Main Agent Output. This output should have 2 Fields:
    1. `agent_answer` <- This is main Agent answer there you provide AI Message Turn in Chat History.
    2. `references` <- References that prove your `agent_answer` based on data with `references`.
    """

    agent_answer: str = Field(
        description="Main Agent answer there you provide `AI Message Turn` in `Chat History`."
    )
    references: References = Field(
        description="""
Your Response (Main Agent answer) MUST BE supported by data (references), YOU should list their data references. 
"""
    )


def main_agent_finish_output(agent_answer, references):
    return agent_answer, references


async def amain_agent_finish_output(agent_answer, references):
    return agent_answer, references
