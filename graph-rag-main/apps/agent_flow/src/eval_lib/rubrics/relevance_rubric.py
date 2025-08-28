import json

RELEVANCE_CRITERIA = """
Can this model provide a relevant answer in the context of a dialogue between AI and HUMAN, which includes questions and requests from the HUMAN?
A relevant answer is one that corresponds to the user's (HUMAN) request, addresses their question or problem, and takes into account the provided information and context.
Relevance includes accuracy, completeness, appropriateness, and usefulness of the response to the user.
""".strip()

score1_description = """
The answer does not correspond to the user's request or address their question or problem. The response lacks accuracy and does not take into account the provided information and context. The answer is incomplete and not useful to the user.
""".strip()

score2_description = """
The answer partially corresponds to the user's request but does not fully address their question or problem. The response has some accuracy but fails to consider all provided information and context. The answer is somewhat complete but lacks usefulness and appropriateness.
""".strip()

score3_description = """
The answer generally corresponds to the user's request and somewhat addresses their question or problem. The response is mostly accurate and considers some of the provided information and context. The answer is complete but may lack in appropriateness or full usefulness.
""".strip()

score4_description = """
The answer effectively corresponds to the user's request and adequately addresses their question or problem. The response is accurate and considers most of the provided information and context. The answer is complete, appropriate, and useful to the user.
""".strip()

score5_description = """
The answer fully corresponds to the user's request and thoroughly addresses their question or problem. The response is highly accurate and takes into account all provided information and context. The answer is complete, highly appropriate, and extremely useful to the user.
""".strip()

RELEVANCE_RUBRIC = {
    "criteria": RELEVANCE_CRITERIA,
    "score1_description": score1_description,
    "score2_description": score2_description,
    "score3_description": score3_description,
    "score4_description": score4_description,
    "score5_description": score5_description,
}

RELEVANCE_RUBRIC_JSON_STRING = json.dumps(RELEVANCE_RUBRIC)


def generate_relevance_rubric_json_string():
    return RELEVANCE_RUBRIC_JSON_STRING
