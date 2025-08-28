import unittest
import requests
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# gino off logs
logging.getLogger("gino.engine._SAEngine").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


class TestKnowledgeManagerEndpoints(unittest.TestCase):
    BASE_URL = "http://localhost:8098"

    def test_record_content(self):
        """Test the record_content endpoint in the building router."""
        url = f"{self.BASE_URL}/building/record_content"
        data = {
            "knowledge_content": [
                {"text": "Example text", "src": "Example source"},
                {"text": "Example text1", "src": "Example source1"},
                {"text": "Example text2", "src": "Example source2"},
                {"text": "Example text3", "src": "Example source3"},
            ]
        }
        response = requests.post(url, json=data)
        logger.info(
            f"Testing /record_content: status {response.status_code}, response {response.json()}"
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue("tasks" in response.json())

    def test_vector_retrieve_1(self):
        """Test the vector_retrieve endpoint in the retrievement router."""
        url = f"{self.BASE_URL}/retrievement/vector_retrieve"
        data = {
            "natural_query": "What's the typical charging time for the BMW i3 and Audi A3 Sportback e-tron?",
            "score_threshold": 0.88,
            "user_id": "123",
            "session_id": "session123",
            "dialog_id": "dialog123",
            "top_k_url": 10,
            "country": "gb",
            "locale": "en",
            "domain_filter": [],
            "use_output_from_verif_as_content": True,
            "llm_model": "gpt-3.5-turbo-16k",
        }
        response = requests.post(url, json=data)
        logger.info(
            f"Testing /vector_retrieve: status {response.status_code}, response {response.json()}"
        )

        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json().get("knowledge_content"), list)

    def test_vector_retrieve_2(self):
        """Test the vector_retrieve endpoint in the retrievement router."""
        url = f"{self.BASE_URL}/retrievement/vector_retrieve"
        data = {
            "natural_query": "What's the average mileage I can expect from a used BMW i3?",
            "score_threshold": 0.88,
            "user_id": "123",
            "session_id": "session123",
            "dialog_id": "dialog123",
            "top_k_url": 10,
            "country": "gb",
            "locale": "en",
            "domain_filter": [],
            "use_output_from_verif_as_content": True,
            "llm_model": "gpt-3.5-turbo-16k",
        }
        response = requests.post(url, json=data)
        logger.info(
            f"Testing /vector_retrieve: status {response.status_code}, response {response.json()}"
        )

        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json().get("knowledge_content"), list)

    def test_vector_retrieve_3(self):
        """Test the vector_retrieve endpoint in the retrievement router."""
        url = f"{self.BASE_URL}/retrievement/vector_retrieve"
        data = {
            "natural_query": "What's the average mileage I can expect from a used BMW i3?",
            "score_threshold": 0.88,
            "user_id": "123",
            "session_id": "session123",
            "dialog_id": "dialog123",
            "top_k_url": 10,
            "country": "gb",
            "locale": "en",
            "domain_filter": [],
            "use_output_from_verif_as_content": True,
            "llm_model": "gpt-3.5-turbo-16k",
        }
        response = requests.post(url, json=data)
        logger.info(
            f"Testing /vector_retrieve: status {response.status_code}, response {response.json()}"
        )

        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json().get("knowledge_content"), list)

    def test_vector_retrieve_4(self):
        """Test the vector_retrieve endpoint in the retrievement router."""
        url = f"{self.BASE_URL}/retrievement/vector_retrieve"
        data = {
            "natural_query": "What's the real-world fuel efficiency of the Audi A3 Sportback e-tron in hybrid mode?",
            "score_threshold": 0.88,
            "user_id": "123",
            "session_id": "session123",
            "dialog_id": "dialog123",
            "top_k_url": 10,
            "country": "gb",
            "locale": "en",
            "domain_filter": [],
            "use_output_from_verif_as_content": True,
            "llm_model": "gpt-3.5-turbo-16k",
        }
        response = requests.post(url, json=data)
        logger.info(
            f"Testing /vector_retrieve: status {response.status_code}, response {response.json()}"
        )

        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json().get("knowledge_content"), list)


if __name__ == "__main__":
    unittest.main()
