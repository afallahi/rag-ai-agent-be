import boto3
import json
from main.config import Config
from main.llm.base import LLMBase

class BedrockClient(LLMBase):
    def __init__(self, model_id: str = Config.BEDROCK_MODEL_ID, region: str = Config.BEDROCK_REGION):
        self.model_id = model_id
        self.region = region
        self.client = boto3.client("bedrock-runtime", region_name=region)


    def is_running(self) -> bool:
        try:
            # Run a minimal test prompt to check Bedrock connectivity
            test_prompt = "Hello"
            input_data = {
                "prompt": test_prompt,
                "max_tokens": 5,
                "temperature": 0.1,
                "top_p": 0.5
            }

            response = self.client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(input_data)
            )
            return response['ResponseMetadata']['HTTPStatusCode'] == 200
        except Exception as e:
            print(f"[ERROR] Bedrock health check failed: {e}")
            return False


    def generate_answer(self, prompt: str) -> str:
        try:
            if self.model_id.startswith("amazon.titan-text"):
                body = {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "temperature": 0.5,
                        "maxTokenCount": 500,
                        "topP": 0.9
                    }
                }
            else:
                raise NotImplementedError(f"Model '{self.model_id}' is not yet supported.")

            response = self.client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body)
            )

            result = json.loads(response['body'].read())
            return result.get("results", [{}])[0].get("outputText", "No response")

        except NotImplementedError as ne:
            print(f"[ERROR] Unsupported model: {ne}")
            return "This model is not currently supported in the implementation."
        except Exception as e:
            print(f"[ERROR] Bedrock generation failed: {e}")
            return "LLM error: could not generate response"
