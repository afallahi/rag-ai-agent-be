class IntentDetectorBase:
    def detect(self, text: str) -> str:
        """Return a classified intent (string)."""
        raise NotImplementedError("Subclasses must implement detect()")
