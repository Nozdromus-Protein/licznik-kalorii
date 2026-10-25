import unittest
from unittest.mock import patch, Mock, AsyncMock

from fastapi.testclient import TestClient
import main


class APIContractSmokeTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def test_health_and_openapi(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        paths = self.client.get("/openapi.json").json()["paths"]
        for path in ("/analyze-meal", "/chat", "/ai/chat", "/v1/translate",
                     "/v1/prices/barcode/{gtin}", "/v1/recipes/analyze"):
            self.assertIn(path, paths)

    def test_translate_with_mocked_upstream(self):
        response = Mock(status_code=200)
        response.json.return_value = {"data": {"translations": [{"translatedText": "apple"}]}}
        with patch.dict(main.os.environ, {"GOOGLE_TRANSLATOR_API_KEY": "test"}), \
             patch.object(main.requests, "post", return_value=response):
            main._translation_cache.clear()
            result = self.client.post("/v1/translate", json={"text": "jabłko", "target": "en"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["translation"], "apple")

    def test_prices_with_mocked_upstream(self):
        response = Mock(status_code=200)
        response.json.return_value = {"items": [{"price": 2.5, "currency": "EUR"}]}
        with patch.object(main.requests, "get", return_value=response):
            result = self.client.get("/v1/prices/barcode/12345678")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["offers"][0]["price"], 2.5)

    def test_chat_aliases_with_mocked_ai(self):
        with patch.object(main, "generate_chat_json", new=AsyncMock(return_value={"reply": "Test"})):
            for path in ("/chat", "/ai/chat"):
                result = self.client.post(path, json={"message": "Test"})
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json()["reply"], "Test")

    def test_validation_does_not_call_ai(self):
        self.assertEqual(self.client.post("/analyze-meal").status_code, 422)
        self.assertEqual(self.client.post("/v1/recipes/analyze", data={"description": ""}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
