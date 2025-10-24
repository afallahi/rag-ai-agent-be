import boto3
import json
from main.config import Config
from main.llm.base import LLMBase

class BedrockClient(LLMBase):
    def __init__(self, model_id: str = Config.BEDROCK_MODEL_ID, region: str = Config.BEDROCK_REGION):
        self.provider = "bedrock"
        self.model_id = model_id
        self.region = region
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self._use_converse = False  # whether to use converse() instead of invoke_model()

        model_info = self._get_model_info()
        if model_info:
            inference_types = model_info.get("inferenceTypesSupported", [])
            if "ON_DEMAND" not in inference_types:
                raise RuntimeError(
                    f"Model '{self.model_id}' only works with an inference profile. "
                    "Choose another model for on-demand use."
                )
            if model_id.startswith("anthropic.claude"):
                self._use_converse = True
            elif model_id.startswith("amazon.titan-text"):
                self._use_converse = False
            else:
                raise NotImplementedError(f"Bedrock model '{model_id}' is not supported by this client.")
        else:
            # Fallback: assume Titan unless it's claude
            if model_id.startswith("anthropic.claude"):
                self._use_converse = True


    def _get_model_info(self):
        """Fetches metadata about the current model to detect inference type requirements. """
        try:
            bedrock_client  = boto3.client("bedrock", region_name=self.region)
            response = bedrock_client.list_foundation_models()
            summaries = response.get("modelSummaries", [])
            for summary in summaries:
                if summary.get("modelId") == self.model_id:
                    return summary
            return None
        except Exception as e:
            print(f"[WARN] Could not fetch model metadata: {e}")
            return None
        
    def _invoke(self, prompt: str, max_tokens: int = 500) -> str:
        """Internal method to invoke the model."""
        if self._use_converse:
            # Claude / INFERENCE_PROFILE
            messages = [{"role": "user", "content": [{"text": prompt}]}]
            try:
                response = self.client.converse(
                    modelId=self.model_id,
                    messages=messages,
                    inferenceConfig={"maxTokens": max_tokens, "temperature": 0.5, "topP": 0.9}
                )
                return response["output"]["message"]["content"][0].get("text", "No response")
            except Exception as e:
                if "on-demand throughput isn’t supported" in str(e):
                    raise RuntimeError(
                        f"Model '{self.model_id}' requires an inference profile. "
                        "You cannot use it with on-demand throughput."
                    ) from e
                raise
        else:
            # Titan / invoke_model
            body = {
                "inputText": prompt,
                "textGenerationConfig": {"temperature": 0.5, "maxTokenCount": max_tokens, "topP": 0.9}
            }
            try:
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body)
                )
                result = json.loads(response["body"].read())
                return result.get("results", [{}])[0].get("outputText", "No response")
            except Exception as e:
                raise RuntimeError(f"Bedrock invocation failed: {e}") from e


    def is_running(self) -> bool:
        """Quick connectivity test for the configured model."""
        try:
            _ = self._invoke("Hello", max_tokens=5)
            return True
        except Exception as e:
            print(f"[ERROR] Bedrock health check failed: {e}")
            return False
    

    def generate_answer(self, prompt: str) -> str:
        """Generate an answer from the selected model."""
        try:
            return self._invoke(prompt, max_tokens=500)
        except Exception as e:
            print(f"[ERROR] Bedrock generation failed: {e}")
            return "LLM error: could not generate response"
