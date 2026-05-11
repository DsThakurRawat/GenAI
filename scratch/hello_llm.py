import os
from google import genai
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found in .env file.")
    print("Please add your key to the .env file.")
    exit(1)

# Initialize the client
client = genai.Client(api_key=api_key)

def main():
    print("--- GenAI Hello World (New SDK) ---")
    prompt = "Explain Generative AI in one sentence to a software engineer."
    
    print(f"Prompt: {prompt}")
    print("Generating response...")
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        print(f"\nResponse: {response.text}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
