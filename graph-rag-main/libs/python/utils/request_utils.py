import json

import requests

# from tenacity import retry, stop_after_attempt, wait_fixed

from libs.python.utils.logger import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

RETRIES = 5
BACKOFF_FACTOR = 0.3
TIMEOUT = 2048
# async def asend_request(data, endpoint: str):
#     response = requests.post(endpoint, json=data)
#     if response.status_code == 200:
#         logger.info(
#             f"Sent request to {endpoint}: {json.dumps(response.json(), indent=2)}"
#         )
#         return response
#     else:
#         logger.error(
#             f"Error: Unable to send request to {endpoint}, "
#             f"status code {response.status_code}; {response.content}"
#         )


# @retry(stop=stop_after_attempt(10), wait=wait_fixed(3))
async def asend_request(
    data, endpoint: str, retries=RETRIES, backoff_factor=BACKOFF_FACTOR
):
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST", "GET", "PUT"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    try:
        response = session.post(endpoint, json=data, timeout=TIMEOUT)
        if response.status_code == 200:
            logger.info(f"Sent request to {endpoint}")
            logger.info(response.json())
            return response
        else:
            logger.error(
                f"Error: Unable to send request to {endpoint}, "
                f"status code {response.status_code}; {response.content}"
            )
    except requests.exceptions.Timeout:
        logger.error(f"Error: Request to {endpoint} timed out after {retries} retries.")
    except requests.exceptions.RequestException as e:
        logger.error(
            f"Error: An error occurred while sending request to {endpoint}: {e}"
        )


# def send_request(data, endpoint: str):
#     response = requests.post(endpoint, json=data)
#     if response.status_code == 200:
#         logger.info(
#             f"Sent request to {endpoint}: {json.dumps(response.json(), indent=2)}"
#         )
#         return response
#     else:
#         logger.error(
#             f"Error: Unable to send request to {endpoint}, "
#             f"status code {response.status_code}; {json.dumps(response.json())}"
#         )


# @retry(stop=stop_after_attempt(10), wait=wait_fixed(3))
def send_request(data, endpoint: str, retries=RETRIES, backoff_factor=BACKOFF_FACTOR):
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST", "GET", "PUT"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    try:
        response = session.post(endpoint, json=data, timeout=TIMEOUT)
        if response.status_code == 200:
            logger.info(f"Sent request to {endpoint}:")
            logger.info(response.json())
            return response
        else:
            logger.error(
                f"Error: Unable to send request to {endpoint}, "
                f"status code {response.status_code}; {json.dumps(response.json())}"
            )
    except requests.exceptions.Timeout:
        logger.error(f"Error: Request to {endpoint} timed out after {retries} retries.")
    except requests.exceptions.RequestException as e:
        logger.error(
            f"Error: An error occurred while sending request to {endpoint}: {e}"
        )
