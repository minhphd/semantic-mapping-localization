import os
import time
import configparser
from io import BytesIO
import base64

import numpy as np
from PIL import Image
from groq import Groq
from openai import OpenAI


try:
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), '..', 'configs', 'api_keys.ini'))
except Exception as e:
    config = configparser.ConfigParser()


class GroqModel():
    """Groq-hosted VLM (Llama-4 family)."""

    def __init__(self, model):
        if model not in {
            'meta-llama/llama-4-scout-17b-16e-instruct',
            'meta-llama/llama-4-maverick-17b-128e-instruct',
        }:
            raise ValueError(f"Unsupported vision model: {model}")
        self.model = model
        if 'groq_api' not in config['API_KEYS']:
            self.client = None
            print("Warning: Groq API key not found. GroqModel will not work.")
        else:
            self.client = Groq(api_key=config['API_KEYS']['groq_api'])

    def __call__(self, input, image=None):
        content = [{"type": "text", "text": input}]
        if image is not None:
            buffered = BytesIO()
            Image.fromarray(image).save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_str}"},
            })

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": content}],
                model=self.model,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Error during Groq API call: {e}")
            time.sleep(60)
            return self.__call__(input, image)


class OpenaiEmbedding():
    """OpenAI text embedding models."""

    def __init__(self, model):
        if model not in {'text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002'}:
            raise ValueError(f"Unsupported embedding model: {model}")
        self.model = model
        self.client = OpenAI(api_key=config['API_KEYS']['openai_api'])

    def __call__(self, input):
        """
        Parameters
        ----------
        input : str | list[str]

        Returns
        -------
        np.ndarray  shape (D,) for single string, (N, D) for list
        """
        single = isinstance(input, str)
        if single:
            input = [input]

        response = self.client.embeddings.create(model=self.model, input=input)
        embeddings = np.array(
            [item.embedding for item in response.data],
            dtype=np.float32,
        )
        return embeddings[0] if single else embeddings


class OpenaiModel():
    """OpenAI GPT VLM."""

    def __init__(self, model):
        if model not in {'gpt-5.1', 'gpt-5-mini', 'gpt-4.1-mini'}:
            raise ValueError(f"Unsupported vision model: {model}")
        self.model = model
        self.client = OpenAI(api_key=config['API_KEYS']['openai_api'])

    def __call__(self, input, image=None):
        content = [{"type": "input_text", "text": input}]
        if image is not None:
            buffered = BytesIO()
            Image.fromarray(image).save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{img_str}",
            })

        try:
            chat_completion = self.client.responses.create(
                input=[{"role": "user", "content": content}],
                model=self.model,
            )
        except Exception as e:
            print(f"Error during OpenAI API call: {e}")

        return chat_completion.output_text
