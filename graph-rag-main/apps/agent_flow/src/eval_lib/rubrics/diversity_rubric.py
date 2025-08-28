import json

DIVERSITY_CRITERIA = """
How varied and rich is the answer in providing different perspectives and insights on the question?
A diverse answer is one that includes multiple viewpoints, various examples, and a wide range of insights relevant to the user's (HUMAN) question.
Diversity includes the breadth of information, the inclusion of different perspectives, and the richness of examples provided in the response.
It is important to consider the context of the user's question. When an authoritative "gold standard" source is available, it should be referenced as primary, but additional perspectives can enhance the response.
""".strip()

score1_description = """
The answer does not provide varied or rich perspectives on the question. The response is narrow, lacking different viewpoints, examples, and insights, making it limited in scope.
""".strip()

score2_description = """
The answer provides limited diversity in perspectives and insights. The response includes some viewpoints and examples but lacks breadth and richness, offering a narrow understanding of the topic.
""".strip()

score3_description = """
The answer generally provides varied perspectives and insights on the question. The response includes multiple viewpoints and examples, but may lack comprehensive diversity and richness.
""".strip()

score4_description = """
The answer effectively provides varied and rich perspectives on the question. The response includes a good range of viewpoints, examples, and insights, offering a broad understanding of the topic.
""".strip()

score5_description = """
The answer fully provides varied and rich perspectives on the question. The response includes a wide range of viewpoints, numerous examples, and deep insights, offering an extensive and comprehensive understanding of the topic.
""".strip()

DIVERSITY_RUBRIC = {
    "criteria": DIVERSITY_CRITERIA,
    "score1_description": score1_description,
    "score2_description": score2_description,
    "score3_description": score3_description,
    "score4_description": score4_description,
    "score5_description": score5_description,
}

DIVERSITY_RUBRIC_JSON_STRING = json.dumps(DIVERSITY_RUBRIC)


def generate_diversity_rubric_json_string():
    return DIVERSITY_RUBRIC_JSON_STRING
