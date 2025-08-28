import json


def generate_plagiarism_rubric_json_string(source_retrival_info: str):
    PLAGIARISM_CRITERIA = """
Plagiarism is assessed by evaluating the extent to which the agent's answer has copied or closely paraphrased text from a source.
{source_retrival_info}
    """.strip()

    PLAGIARISM_CRITERIA = PLAGIARISM_CRITERIA.format(
        source_retrival_info=source_retrival_info
    )

    score1_description = """
    The agent's answer contains verbatim copying of significant portions of text from the source, with no changes or minimal changes and no attribution. The structure, phrasing, and content are directly lifted from the source.
    """.strip()

    score2_description = """
    The agent's answer includes close paraphrasing of the source text. While not verbatim, the answer follows the structure and phrasing of the source closely, with only minor word substitutions and no attribution.
    """.strip()

    score3_description = """
    The agent's answer shows moderate paraphrasing with more significant changes in wording and structure but still follows the source text closely. Attribution may be present but is insufficient or unclear.
    """.strip()

    score4_description = """
    The agent's answer demonstrates a good level of paraphrasing, with clear rephrasing and restructuring of the source text. Attribution is provided but may not follow proper citation guidelines completely.
    """.strip()

    score5_description = """
    The agent's answer is original and does not rely heavily on the source text. If any information from the source is used, it is properly paraphrased, and full and proper attribution is given according to citation guidelines.
    """.strip()

    PLAGIARISM_RUBRIC = {
        "criteria": PLAGIARISM_CRITERIA,
        "score1_description": score1_description,
        "score2_description": score2_description,
        "score3_description": score3_description,
        "score4_description": score4_description,
        "score5_description": score5_description,
    }

    PLAGIARISM_RUBRIC_JSON_STRING = json.dumps(PLAGIARISM_RUBRIC)
    return PLAGIARISM_RUBRIC_JSON_STRING
