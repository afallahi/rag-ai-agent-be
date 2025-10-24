import boto3
import json
from main.config import Config


class BedrockIntentDetector:
    def __init__(self):
        self.client = boto3.client("bedrock-runtime", region_name=Config.AWS_REGION)
        self.model_id = Config.BEDROCK_MODEL_ID

        self.intents = [
            "greeting",
            "thanks",
            "goodbye",
            "help",
            "chitchat",
            "question",
            "unclear",
        ]

    def detect(self, text: str) -> str:
        prompt = (
            f"Classify the following user message into one of these intents: {', '.join(self.intents)}.\n\n"
            f"User Message: {text.strip()}\n\nIntent:"
        )

        try:
            body = json.dumps({
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 20,
                "temperature": 0,
            })

            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json"
            )

            result_body = json.loads(response["body"].read())
            text_output = result_body["content"][0]["text"].strip().lower()

            # Normalize and validate
            if text_output not in self.intents:
                text_output = "unclear"

            return text_output

        except Exception as e:
            print(f"[WARN] Bedrock intent detection failed: {e}")
            return "unclear"
