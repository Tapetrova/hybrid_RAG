import json


def generate_consistency_rubric_json_string(
    dialog_string_main: str, new_dialogs_to_eval_consistency: str
):

    CONSISTENCY_CRITERIA = """
Did the responses remain consistent even with minor changes in the dialogue?

The metric evaluates the consistency of responses when the dialogue undergoes minor changes. 
Consistent responses should convey the same meaning, 
even if the dialogue is slightly altered using synonyms or other minor modifications.

**Responses** include The main **response** is based on main dialogue and **responses** are based on slightly modified dialogues.



Here is the main dialogue:
{dialog_string_main}



Here is slightly modified dialogues:
{new_dialogs_to_eval_consistency}
    """.strip()

    CONSISTENCY_CRITERIA = CONSISTENCY_CRITERIA.format(
        dialog_string_main=dialog_string_main,
        new_dialogs_to_eval_consistency=new_dialogs_to_eval_consistency,
    )

    score1_description = """
The responses are highly inconsistent. The answers significantly change in meaning or context, showing a lack of stability when minor changes are made to the dialogue.
    """.strip()

    score2_description = """
The responses show noticeable inconsistencies. There are clear changes in meaning in answers, indicating poor stability with minor dialogue modifications.
    """.strip()

    score3_description = """
The responses are moderately consistent. Some changes in answers are present, but the overall intent remains mostly intact despite minor dialogue modifications.
    """.strip()

    score4_description = """
The responses are mostly consistent. There are minor changes in wording (in answers) but the meaning and context remain stable (in answers), showing good resilience to dialogue modifications.
    """.strip()

    score5_description = """
The responses are highly consistent. The meaning and context in answers remain unchanged, demonstrating excellent stability and coherence even with minor changes in dialogue.
    """.strip()

    CONSISTENCY_RUBRIC = {
        "criteria": CONSISTENCY_CRITERIA,
        "score1_description": score1_description,
        "score2_description": score2_description,
        "score3_description": score3_description,
        "score4_description": score4_description,
        "score5_description": score5_description,
    }

    CONSISTENCY_RUBRIC_JSON_STRING = json.dumps(CONSISTENCY_RUBRIC)
    return CONSISTENCY_RUBRIC_JSON_STRING
