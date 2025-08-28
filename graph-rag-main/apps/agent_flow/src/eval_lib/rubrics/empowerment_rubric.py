import json

EMPOWERMENT_CRITERIA = """
How well does the answer help the reader understand and make informed judgements about the topic?
An empowering answer is one that enhances the user's (HUMAN) understanding, provides clear explanations, and enables them to make informed decisions.
Empowerment includes the clarity of information, depth of explanation, and the ability to provide actionable insights or guidance.
""".strip()

score1_description = """
The answer does not help the reader understand or make informed judgements about the topic. The response lacks clarity, depth, and actionable insights, leaving the user confused or misinformed.
""".strip()

score2_description = """
The answer provides limited help in understanding the topic or making informed judgements. The response is somewhat clear but lacks depth and actionable insights, offering minimal guidance to the user.
""".strip()

score3_description = """
The answer generally helps the reader understand and make informed judgements about the topic. The response is mostly clear and provides some depth and actionable insights, but may lack in comprehensive guidance.
""".strip()

score4_description = """
The answer effectively helps the reader understand and make informed judgements about the topic. The response is clear, provides good depth, and includes actionable insights, offering substantial guidance to the user.
""".strip()

score5_description = """
The answer fully helps the reader understand and make informed judgements about the topic. The response is highly clear, provides extensive depth, and includes actionable insights, offering excellent guidance to the user.
""".strip()

EMPOWERMENT_RUBRIC = {
    "criteria": EMPOWERMENT_CRITERIA,
    "score1_description": score1_description,
    "score2_description": score2_description,
    "score3_description": score3_description,
    "score4_description": score4_description,
    "score5_description": score5_description,
}

EMPOWERMENT_RUBRIC_JSON_STRING = json.dumps(EMPOWERMENT_RUBRIC)


def generate_empowerment_rubric_json_string():
    return EMPOWERMENT_RUBRIC_JSON_STRING
