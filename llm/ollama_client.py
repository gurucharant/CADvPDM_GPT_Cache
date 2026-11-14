import requests


class OllamaClient:

    def __init__(
            self,
            model="llama3.2:3b",
            base_url="http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url

    def generate(
            self,
            prompt,
            temperature=0.1
    ):

        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        response = requests.post(
            url,
            json=payload,
            timeout=180
        )

        response.raise_for_status()

        result = response.json()

        return result["response"]


if __name__ == "__main__":

    client = OllamaClient()

    answer = client.generate(
        "Explain PLM in two sentences."
    )

    print(answer)