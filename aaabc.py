import os, time, json, traceback
from typing import Optional, List
import openai
import pdb

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://trschatsc.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_API_KEY"] = "0b00962cd666462287f7baf216b2641f"

openai.api_type = "azure"
openai.api_base = "https://trschatsc.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "0b00962cd666462287f7baf216b2641f"

class ChatGPT():
    history_len: int = 10
    def __init__(self, history_len: int = 10):
        self.history_len = history_len

    def generatorAnswer(self, prompt: str,
                        historys: List[List[str]] = [],
                        streaming: bool = False,
                        **kwargs):
        query = prompt

        messages = [
            {"role": "system", "content": "你是一个导游机器人"},
        ]
        messages.append({"role":"user", "content": query})

        if kwargs:
            temperature = kwargs['temperature']
        else:
            temperature = 0.9

        responses = openai.ChatCompletion.create(
            engine="gpt-turbo",
            messages=messages,
            stream=streaming,
            temperature=temperature,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.6,
            timeout=60
        )
        output_text = responses['choices'][0]['message']['content']
        print(output_text)
    
if __name__ == "__main__":
    chatGPT = ChatGPT()
    chatGPT.generatorAnswer('请介绍一下北京的名胜古迹')
            
