import cleantext
import torch
import transformers
import os
import joblib


class Inference:
    def __init__(self, model_path: str):
        """
        Initializes the Inference for sentiment analysis by loading a pre-trained model and tokenizer.

        Args:
            model_path (str): The file path (absolute or relative) to the directory containing the serialized
                              sentiment analysis model and tokenizer. The model is
                              expected to be stored as 'sentiment_model.joblib' and
                              the tokenizer as 'sentiment_tokenizer.joblib'.

        Attributes:
            model: The loaded sentiment analysis model used for predictions.
            tokenizer: The loaded tokenizer used to preprocess the input text before
                       passing it to the model.
        """
        self.model = joblib.load(os.path.join(model_path, "sentiment_model.joblib"))
        self.tokenizer = joblib.load(os.path.join(model_path, "sentiment_tokenizer.joblib"))

    def predict(self, text: str) -> str:
        """
        Predicts the sentiment of a given input text.

        Args:
            text (str): The input text to analyze for sentiment.

        Returns:
            str: The predicted sentiment label ('positive' or 'negative')
                 based on the input text.

        Steps:
            1. Clean data
            2. Preprocesses the input text using the tokenizer.
            3. Passes the preprocessed text to the sentiment model for prediction.
            4. Converts the model output (logits) into a human-readable sentiment label.
        """
        with torch.no_grad():
            inputs = self._preprocess(text)
            logits = self.model(**inputs).logits

        prediction = self._postprocess(logits)
        return prediction

    def _postprocess(self, logits: torch.Tensor) -> str:
        predicted_label = logits.argmax().item()
        return "positive" if predicted_label == 1 else "negative"

    def _preprocess(self, text: str) -> transformers.tokenization_utils_base.BatchEncoding:
        text = self._clean_text(text)
        tokens = self._tokenize(text)
        return tokens

    def _tokenize(self, text: str) -> transformers.tokenization_utils_base.BatchEncoding:
        return self.tokenizer(text, return_tensors="pt")

    @staticmethod
    def _clean_text(text: str) -> str:
        return cleantext.clean(
            text,
            to_ascii=False,
            lower=True,
            no_emoji=True,
            no_urls=True,
            no_emails=True,
            no_phone_numbers=True,
            replace_with_url=" ",
            replace_with_email=" ",
            replace_with_phone_number=" ",
        )