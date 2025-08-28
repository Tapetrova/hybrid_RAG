import asyncio
import json

from tqdm import tqdm

from apps.agent_flow.src.eval_lib.plagiarism_calc import (
    calculate_similarity,
)

input_file_path = "apps/agent_flow/tests/all_dialogs_dataset.json"
with open(input_file_path, "r") as file:
    data = json.load(file)

# Iterate over each dialog in the dataset
results = []
for dialog_key, dialog in tqdm(data.items(), desc="Processing dialogs"):
    agent_answer = dialog["response"]["agent_answer"]
    source_content = dialog["source_retrival_info"]

    score, additional_metrics = asyncio.run(
        calculate_similarity(
            agent_answer=agent_answer,
            sources_content=source_content,
            chunk_size=10,
            overlap_size=4,
            dimensions=10000,
        )
    )
    print(
        f"score={score}, additional_metrics={json.dumps(additional_metrics, indent=2)}\n{'='*100}"
    )
