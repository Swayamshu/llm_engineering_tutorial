import json
from openai import OpenAI
import gradio as gr

OLLAMA_BASE_URL = "http://localhost:11434/v1"
API_KEY = "ollama"
MODEL = "gemma4:latest"

client = OpenAI(api_key=API_KEY, base_url=OLLAMA_BASE_URL)

system_message = """
You are a helpful assistant for an Airline called FlightAI.
Give short, courteous answers, no more than 1 sentence.
Always be accurate. If you don't know the answer, say so.
"""

# Tool Calling Section

ticket_prices = {
    "london": "$599",
    "paris": "$699",
    "tokyo": "$499"
}

def get_ticket_price(destination_city):
    print(f"Tool called for {destination_city} city")

    price = ticket_prices.get(destination_city.lower(), "Unknown City")
    return f"The price for a flight to {destination_city} is {price}"


# LLM recognises JSON format perfectly for Tools or any other special information
# We can describe the Tool to the LLM in JSON format, as shown below
price_function = {
    "name": "get_ticket_price",
    "description": "Get price of a ticket to the destination city from current city.",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to."
            }
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}

# List of Tools for the LLM
tools = [
    {"type": "function", "function": price_function}
]


# Gradio chat callback function
def chat(message, history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]

    response = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)

    while response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        response = handle_tool_calls(message)

        messages.append(message)
        messages.append(response)

        # for message in messages:
        #     print(message)
        response = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)

    return response.choices[0].message.content

# Handle Tool Call method will basically look for LLM asking for inputs from the tools and trigger them manually
def handle_tool_calls(message):
    responses = []

    for tool_call in message.tool_calls:
        if tool_call.function.name == "get_ticket_price":
            arguements = json.loads(tool_call.function.arguments)
            city = arguements.get("destination_city")
            
            price_details = get_ticket_price(city)

            responses.append({
                "role": "tool",
                "content": price_details,
                "tool_call_id": tool_call.id
            })

    return responses


# Gradio Chat interface
gr.ChatInterface(fn=chat).launch()

