import re
import traceback
from enum import Enum

import pandas as pd
from typing import List, Dict

from libs.python.schemas.graphrag.graph_rag_config import SearchMode
from libs.python.utils.logger import logger

COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
ALL_MAPS = "all_maps"


NEW_LINE = "\n"
RESOURCES_LIMIT = 11
RESOURCES_TITLE = "***Report Resources:***"


class ContextDataColumn(str, Enum):
    REPORTS = "reports"
    RELATIONSHIPS = "relationships"
    ENTITIES = "entities"


MAP_FROM_CONTEXT_DATA_COLUMN_TO_TEXT_UNITS = {
    ContextDataColumn.REPORTS: "comms_to_text_units",
    ContextDataColumn.RELATIONSHIPS: "rels_to_text_units",
    ContextDataColumn.ENTITIES: "ents_to_text_units",
}

MAP_FROM_CONTEXT_DATA_COLUMN_TO_MASK_PATTERN = {
    # ContextDataColumn.REPORTS: r"\[Data: Reports \(([\d,\s\+more]*)\)\]",
    # ContextDataColumn.RELATIONSHIPS: r"\[Data: Relationships \(([\d,\s\+more]*)\)\]",
    # ContextDataColumn.ENTITIES: r"\[Data: Entities \(([\d,\s\+more]*)\)\]",
    ContextDataColumn.REPORTS: r"Reports \(([\d,\s\+more]*)\)",
    ContextDataColumn.RELATIONSHIPS: r"Relationships \(([\d,\s\+more]*)\)",
    ContextDataColumn.ENTITIES: r"Entities \(([\d,\s\+more]*)\)",
}


async def extract_report_numbers(text: str, mask_pattern: str) -> List[int]:
    # Regular expression pattern to find report numbers in the "[Data: Reports (" format

    # Find all matches in the text
    matches = re.findall(mask_pattern, text)

    # Initialize a list to store all the report numbers
    report_numbers = []

    # Process each match
    for match in matches:
        # Split the match by commas and strip any whitespace
        numbers = [
            num.strip()
            for num in match.split(",")
            if num.strip() and num.strip().isnumeric()
        ]

        for n_i in range(len(numbers)):
            try:
                numbers[n_i] = int(numbers[n_i])
            except Exception as e:
                fmt_exc = traceback.format_exc()
                logger.exception(f"TRYING: {numbers[n_i]} to INT\n\n{fmt_exc}\n")

        # Append numbers to the report_numbers list
        report_numbers.extend(numbers)

    # Remove duplicates and sort the report numbers
    if len(report_numbers) != 0:
        report_numbers = sorted(set(report_numbers))

    return report_numbers


async def get_resources_from_column_context_data_result(
    result, all_maps, based_on_column_: ContextDataColumn
) -> Dict:
    resources = None
    based_on_column = based_on_column_.value
    if based_on_column in result.context_data:

        if based_on_column_ == ContextDataColumn.REPORTS:
            idxs = result.context_data[based_on_column].id
        else:
            idxs = result.context_data[based_on_column].id_full

        result.context_data[based_on_column]["sources"] = idxs.apply(
            lambda x: list(
                set(
                    [
                        all_maps["docs_to_source"][item]
                        for tu in all_maps[
                            MAP_FROM_CONTEXT_DATA_COLUMN_TO_TEXT_UNITS[based_on_column_]
                        ][x]
                        for item in all_maps["text_units_to_docs"][tu]
                    ]
                )
            )
        )

        result.context_data[based_on_column]["id"] = pd.to_numeric(
            result.context_data[based_on_column]["id"]
        )
        result.context_data[based_on_column] = result.context_data[
            based_on_column
        ].sort_values(by="id")
        result.context_data[based_on_column]["id"] = result.context_data[
            based_on_column
        ]["id"].astype(str)

        resources = dict()
        for idx, row in result.context_data[based_on_column].iterrows():
            for resource in row.sources:
                if resource in resources:
                    resources[resource].append(int(row.id))
                else:
                    resources[resource] = [
                        int(row.id),
                    ]

        resources = {
            kk: resources[kk]
            for kk in sorted(resources, key=lambda k: len(resources[k]), reverse=True)
        }
    return resources


async def add_resources_to_response(
    response_text: str,
    resources: Dict,
    extracted_numbers_map: Dict[ContextDataColumn, List[int]] = None,
) -> str:
    resources_formatted = ""
    numero_to_check_limit = 0
    data_to_print = {
        ContextDataColumn.ENTITIES: dict(),
        ContextDataColumn.RELATIONSHIPS: dict(),
        ContextDataColumn.REPORTS: dict(),
    }

    for context_data_column in data_to_print.keys():
        if len(resources[context_data_column.value]) == 0:
            logger.info(f"!!!len(resources[{context_data_column.value}]) == 0;")
            continue

        if extracted_numbers_map is None:
            extracted_numbers: List[int] = await extract_report_numbers(
                text=response_text,
                mask_pattern=MAP_FROM_CONTEXT_DATA_COLUMN_TO_MASK_PATTERN[
                    context_data_column
                ],
            )
        else:
            extracted_numbers: List[int] = extracted_numbers_map[context_data_column]

        logger.info(
            f"!!!extracted_numbers <{context_data_column}>: {extracted_numbers};"
        )
        if len(extracted_numbers) == 0:
            logger.info(f"!!!len(extracted_numbers) == 0;")
            continue

        # resources_formatted += (
        #     f"Formatted Pre Analysis: {context_data_column.value.capitalize()}"
        # )

        for numero, resource in enumerate(resources[context_data_column.value]):
            needful_rep_to_print = []
            for rep in resources[context_data_column.value][resource]:
                if rep in extracted_numbers:
                    needful_rep_to_print.append(str(rep))

            if len(needful_rep_to_print) != 0:
                if resource not in data_to_print[context_data_column]:
                    data_to_print[context_data_column][resource] = set()

                data_to_print[context_data_column][resource] = data_to_print[
                    context_data_column
                ][resource].union(needful_rep_to_print)

    resources_names_all = list()
    for col_name_data in data_to_print.keys():
        data_to_print[col_name_data] = {
            res_k_sorted: data_to_print[col_name_data][res_k_sorted]
            for res_k_sorted in sorted(
                data_to_print[col_name_data].keys(),
                key=lambda res_name: len(data_to_print[col_name_data][res_name]),
                reverse=True,
            )
        }

        for rec in data_to_print[col_name_data].keys():
            if rec not in resources_names_all:
                resources_names_all.append(rec)

            data_to_print[col_name_data][rec] = sorted(
                list(data_to_print[col_name_data][rec]), key=lambda x: int(x)
            )

    for numero, resource in enumerate(resources_names_all):

        if resource in data_to_print[ContextDataColumn.ENTITIES]:
            needful_rep_to_print = data_to_print[ContextDataColumn.ENTITIES][resource]
            numero_to_check_limit += 1

            additional_info_rels = ""
            if resource in data_to_print[ContextDataColumn.RELATIONSHIPS]:
                needful_rep_to_print_rels = data_to_print[
                    ContextDataColumn.RELATIONSHIPS
                ][resource]
                additional_info_rels = f"{ContextDataColumn.RELATIONSHIPS.value.capitalize()}: {', '.join(needful_rep_to_print_rels)}; "

            additional_info_reps = ""
            if resource in data_to_print[ContextDataColumn.REPORTS]:
                needful_rep_to_print_reps = data_to_print[ContextDataColumn.REPORTS][
                    resource
                ]
                additional_info_reps = f"{ContextDataColumn.REPORTS.value.capitalize()}: {', '.join(needful_rep_to_print_reps)}; "

            resources_formatted += (
                f"{numero_to_check_limit}. {resource} was in "
                f"{ContextDataColumn.ENTITIES.value.capitalize()}: {', '.join(needful_rep_to_print)}; {additional_info_rels}{additional_info_reps}\n"
            )

            if (numero_to_check_limit + 1) == RESOURCES_LIMIT:
                # TODO: add check in `if rep in extracted_report_numbers` to other_resources_count
                other_resources_count = resources_names_all[(numero + 1) :]

                if len(other_resources_count) != 0:
                    resources_formatted += (
                        f"And etc... (+ extra {len(other_resources_count)} sources)"
                    )
                break

        elif resource in data_to_print[ContextDataColumn.REPORTS]:
            needful_rep_to_print = data_to_print[ContextDataColumn.REPORTS][resource]
            numero_to_check_limit += 1

            additional_info_rels = ""
            if resource in data_to_print[ContextDataColumn.RELATIONSHIPS]:
                needful_rep_to_print_rels = data_to_print[
                    ContextDataColumn.RELATIONSHIPS
                ][resource]
                additional_info_rels = f"{ContextDataColumn.RELATIONSHIPS.value.capitalize()}: {', '.join(needful_rep_to_print_rels)}; "

            resources_formatted += (
                f"{numero_to_check_limit}. {resource} was in "
                f"{ContextDataColumn.REPORTS.value.capitalize()}: {', '.join(needful_rep_to_print)}; {additional_info_rels}\n"
            )

            if (numero_to_check_limit + 1) == RESOURCES_LIMIT:
                # TODO: add check in `if rep in extracted_report_numbers` to other_resources_count
                other_resources_count = resources_names_all[(numero + 1) :]

                if len(other_resources_count) != 0:
                    resources_formatted += (
                        f"And etc... (+ extra {len(other_resources_count)} sources)"
                    )
                break

        elif resource in data_to_print[ContextDataColumn.RELATIONSHIPS]:
            needful_rep_to_print = data_to_print[ContextDataColumn.RELATIONSHIPS][
                resource
            ]
            numero_to_check_limit += 1

            resources_formatted += (
                f"{numero_to_check_limit}. {resource} was in "
                f"{ContextDataColumn.RELATIONSHIPS.value.capitalize()}: {', '.join(needful_rep_to_print)};\n"
            )

            if (numero_to_check_limit + 1) == RESOURCES_LIMIT:
                # TODO: add check in `if rep in extracted_report_numbers` to other_resources_count
                other_resources_count = resources_names_all[(numero + 1) :]

                if len(other_resources_count) != 0:
                    resources_formatted += (
                        f"And etc... (+ extra {len(other_resources_count)} sources)"
                    )
                break

    # f"{numero_to_check_limit}. {resource} was in : {', '.join(needful_rep_to_print)};\n"
    if resources_formatted == "":
        resources_formatted = "There is NO any Resources!"

    response_text += f"\n\n\n\n\t{RESOURCES_TITLE}\n\n{resources_formatted}\n\n"
    return response_text
