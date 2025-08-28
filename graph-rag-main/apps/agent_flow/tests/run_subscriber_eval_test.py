import os
import json
import uuid
import redis
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from libs.python.utils.request_utils import send_request

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6380))
REDIS_PASSWORD_LLM = os.getenv("REDIS_PASSWORD_LLM", "redispassword")
REDIS_DB_PUB_SUB = int(os.getenv("REDIS_DB_PUB_SUB", 2))

ENDPOINT = "http://127.0.0.1:8096/evaluation/score_calc_pipeline"


def message_handler(message):
    try:
        print(
            f"Received message: \n{json.dumps(json.loads(message['data'].decode('utf8')), indent=2, ensure_ascii=False)}\n\n{'='*100}"
        )
    except Exception as e:
        print(f"Error handling message: {e}")


def listen_for_messages(pubsub, channel):
    pubsub.subscribe(**{channel: message_handler})
    print(f"Subscribed to {channel}")

    for message in pubsub.listen():
        if message["type"] == "message":
            event_message_data = json.loads(message["data"])
            print(f"Received message: {event_message_data}")

            if event_message_data.get("status") == "SUCCESS":
                break

    pubsub.unsubscribe(channel)
    print(f"Unsubscribed from {channel}")


def send_eval_request(data):
    response = send_request(data=data, endpoint=ENDPOINT)
    response_dict = response.json()
    print(
        f"response_dict.get('pubsub_channel_name'): {response_dict.get('pubsub_channel_name')}"
    )


def subscriber():
    client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB_PUB_SUB,
        password=REDIS_PASSWORD_LLM,
    )

    path2request_body = "test_payload_request.json"
    with open(path2request_body) as f:
        dataset = json.load(f)

    for data_name, data in dataset.items():
        print(f"process data_name: {data_name}")

        data["eval_score_cfg"] = {
            "llm_judge_config": {
                "batch_size_to_eval": 5,
                "times_to_eval": 2,
                "max_tokens": 1024,
                "repetition_penalty": 1.03,
                "best_of": 1,
                "temperature": 0.8,
                "top_p": 0.9,
                # "max_tokens": 1124,
                # "repetition_penalty": 1.02,
                # "best_of": 2,
                # "temperature": 0.2,
                # "top_p": 0.8,
            },
            # "eval_metric_toggle": {
            #     "accuracy_compare_to_fact_knowledge_db": False,
            #     "comprehensiveness": False,
            #     "consistency": False,
            #     "directness": False,
            #     "diversity": False,
            #     "empowerment": False,
            #     "plagiarism": False,
            #     "relevance": False,
            # },
            "eval_metric_toggle": {
                "accuracy_compare_to_factKDB": True,
                "comprehensiveness": True,
                "consistency": True,
                "directness": True,
                "diversity": True,
                "empowerment": True,
                "plagiarism": True,
                "relevance": True,
            },
        }

        data["as_task"] = True
        data["pubsub_channel_name"] = f"pubsub_channel_name_test_{str(uuid.uuid4())}"
        channel = data["pubsub_channel_name"]
        print(f"!!!CHANNEL: {channel}!!!")
        pubsub = client.pubsub()

        with ThreadPoolExecutor() as executor:
            future_listener = executor.submit(listen_for_messages, pubsub, channel)
            future_request = executor.submit(send_eval_request, data)

            future_listener.result()
            future_request.result()
        break


if __name__ == "__main__":
    subscriber()
