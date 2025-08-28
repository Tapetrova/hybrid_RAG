import json

DIRECTNESS_CRITERIA = """
How specifically and clearly does the answer address the question?

The metric evaluates the directness of responses to the main question.
Direct responses should be clear and specific, directly addressing the core of the question without ambiguity or unnecessary information.
    """.strip()

score1_description = """
    The answer does not clearly or specifically address the user's question. The response is ambiguous, contains unnecessary information, or deviates from the topic, making it difficult for the user to understand.
    """.strip()

score2_description = """
    The answer partially addresses the user's question but lacks clarity and specificity. The response includes some unnecessary information or ambiguity, making it less effective.
    """.strip()

score3_description = """
    The answer generally addresses the user's question with some clarity and specificity. The response is mostly direct but may contain minor unnecessary information or slight ambiguity.
    """.strip()

score4_description = """
    The answer effectively addresses the user's question with clarity and specificity. The response is direct, concise, and includes minimal unnecessary information or ambiguity.
    """.strip()

score5_description = """
    The answer fully and clearly addresses the user's question with high specificity. The response is extremely direct, concise, and free of unnecessary information or ambiguity.
    """.strip()

DIRECTNESS_RUBRIC = {
    "criteria": DIRECTNESS_CRITERIA,
    "score1_description": score1_description,
    "score2_description": score2_description,
    "score3_description": score3_description,
    "score4_description": score4_description,
    "score5_description": score5_description,
}

DIRECTNESS_RUBRIC_JSON_STRING = json.dumps(DIRECTNESS_RUBRIC)


def generate_directness_rubric_json_string():
    return DIRECTNESS_RUBRIC_JSON_STRING
