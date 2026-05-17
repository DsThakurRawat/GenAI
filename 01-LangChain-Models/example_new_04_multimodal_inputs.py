"""
Topic: Multimodal Inputs (Vision)
What it does: Passes an image alongside text instructions to a Vision-capable Chat Model.

Why use it:
- Allows the model to "see" and reason about charts, screenshots, or photos.
- Modern Chat Models accept a list of content blocks (text or image_url) inside a HumanMessage.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# We must use a model that supports vision, like gpt-4o or gpt-4-turbo
model = ChatOpenAI(model="gpt-4o-mini")

# We construct a HumanMessage where the content is a list containing both 
# text and an image URL (or a base64 encoded string).
message = HumanMessage(
    content=[
        {"type": "text", "text": "What is in this image? Be specific."},
        {
            "type": "image_url",
            "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wis-black-cobble.jpg/2560px-Gfp-wis-black-cobble.jpg"},
        },
    ]
)

print("--- Sending Multimodal Request ---")
print("Analyzing image from Wikipedia...")

response = model.invoke([message])

print("\n--- Vision Analysis ---")
print(response.content)
