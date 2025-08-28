import json

COMPREHENSIVENESS_CRITERIA = """
How much detail does the answer provide to cover all aspects and details of the question?
A comprehensive answer is one that thoroughly addresses all facets of the user's (HUMAN) question, providing in-depth information and covering all relevant aspects.
Comprehensiveness includes the extent of detail, the inclusion of necessary and pertinent information, and the completeness of the response.
It is important to consider the context of the user's question. When an authoritative "gold standard" source is available, it should be referenced as primary, but additional details and information from other sources can enhance the response.
""".strip()

score1_description = """
The answer provides minimal detail and fails to cover most aspects of the question. The response lacks depth and necessary information, resulting in an incomplete and unsatisfactory answer.
""".strip()

score2_description = """
The answer provides some detail but does not cover all aspects of the question. The response is somewhat thorough but lacks depth and necessary information in several areas, leading to an incomplete understanding.
""".strip()

score3_description = """
The answer generally covers most aspects of the question with a fair amount of detail. The response is mostly thorough but may miss some important details or aspects, providing a reasonably complete answer.
""".strip()

score4_description = """
The answer effectively covers all major aspects of the question with good detail. The response is thorough and includes most necessary information, providing a nearly complete answer.
""".strip()

score5_description = """
The answer thoroughly covers all aspects and details of the question with extensive depth. The response is highly detailed and includes all necessary information, providing a comprehensive and complete answer.
""".strip()

COMPREHENSIVENESS_RUBRIC = {
    "criteria": COMPREHENSIVENESS_CRITERIA,
    "score1_description": score1_description,
    "score2_description": score2_description,
    "score3_description": score3_description,
    "score4_description": score4_description,
    "score5_description": score5_description,
}

COMPREHENSIVENESS_RUBRIC_JSON_STRING = json.dumps(COMPREHENSIVENESS_RUBRIC)


def generate_comprehensiveness_rubric_json_string():
    return COMPREHENSIVENESS_RUBRIC_JSON_STRING
