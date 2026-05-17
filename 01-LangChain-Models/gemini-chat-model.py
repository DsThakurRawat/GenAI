from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from src.core.logger import logger
from src.core.config import settings

# CONCEPTUAL: Gemini's Unique Features
"""
Gemini (by Google) has a few unique characteristics in the ecosystem:
1. Native Multimodality: It can process images, video, and audio in the same prompt.
2. Safety Filters: Google is very strict about safety; you often need to configure 
   'Safety Settings' if you get blocked responses.
3. Model Types:
   - Gemini 1.5 Flash: Fast, cost-effective (Great for most tasks).
   - Gemini 1.5 Pro: Massive context window (up to 2 Million tokens!) and better reasoning.
"""

# Initialize Gemini Chat Model
# Note: Google uses 'GOOGLE_API_KEY'
chat_model = ChatGoogleGenerativeAI(
    google_api_key=settings.GOOGLE_API_KEY,
    
    # PARAMETER: model
    # IMPACT: 'gemini-1.5-flash' is optimized for speed/cost. 'gemini-1.5-pro' for complex reasoning.
    model="gemini-1.5-flash",
    
    # PARAMETER: temperature
    # IMPACT: 0.0 for structured output (highly recommended for the itinerary example below).
    temperature=0.7,
    
    # PARAMETER: max_output_tokens
    # IMPACT: Limits response length (Gemini supports massive outputs up to 8k+ depending on region).
    # max_output_tokens=2048,
    
    # PARAMETER: safety_settings
    # IMPACT: Configures how strict Google is about filtering content.
    # safety_settings={...} 
)

from pydantic import BaseModel, Field

# CONCEPTUAL: Gemini Structured Output
# Gemini 1.5 Flash/Pro are highly reliable for data extraction.
class TravelItinerary(BaseModel):
    destination: str = Field(description="The city and country")
    top_activities: list[str] = Field(description="3 nature-focused activities")
    best_time_to_visit: str = Field(description="The ideal month/season")

# Create a structured version of Gemini
structured_gemini = chat_model.with_structured_output(TravelItinerary)

def run_gemini_demo():
    try:
        location = "Kerala, India"
        logger.info(f"Generating Itinerary for: {location}")

        # This returns a TravelItinerary object
        itinerary = structured_gemini.invoke(f"Plan a nature trip to {location}")
        
        logger.success("Itinerary generated successfully.")
        print(f"\n--- Trip to {itinerary.destination} ---")
        print(f"Best Time: {itinerary.best_time_to_visit}")
        print("Top Activities:")
        for activity in itinerary.top_activities:
            print(f" - {activity}")

    except Exception as e:
        logger.error(f"Gemini API Error: {e}")
        print("\n[TIP] Make sure GOOGLE_API_KEY is set in your .env file.")
        print("[TIP] You can get a free key at https://aistudio.google.com/")

if __name__ == "__main__":
    run_gemini_demo()
