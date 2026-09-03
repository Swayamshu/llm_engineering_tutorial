from openai import OpenAI
import gradio as gr

OLLAMA_BASE_URL = "http://localhost:11434/v1"
API_KEY = "ollama"
MODEL = "gemma4:latest"

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=API_KEY)

system_message = "You are a helpful assistant"

def chat(message, history):
    history = [{"role":h["role"], "content":h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]

    response = client.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content

def streamChat(message, history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]

    stream = client.chat.completions.create(model=MODEL, messages=messages, stream=True)
    response = ""

    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response


gr.ChatInterface(fn=streamChat).launch()

