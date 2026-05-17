import streamlit as st
import logging
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import load_prompt, ChatPromptTemplate
from dotenv import load_dotenv
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class SummaryRequest(BaseModel):
    paper_input: str
    style_input: str
    length_input: str

def run_app():
    st.set_page_config(page_title="Research Assistant", page_icon="🔬")
    st.header('🔬 Research Paper Summarizer')
    st.subheader('Powered by LangChain & Gemini')

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    model_name = st.sidebar.selectbox("Select Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
    temp = st.sidebar.slider("Temperature", 0.0, 2.0, 0.7)

    # Main Input section
    paper_input = st.selectbox(
        "Select Research Paper Name", 
        ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"]
    )

    style_input = st.selectbox(
        "Select Explanation Style", 
        ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
    ) 

    length_input = st.selectbox(
        "Select Explanation Length", 
        ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
    )

    if st.button('🚀 Summarize'):
        try:
            # Validate input using Pydantic
            req = SummaryRequest(
                paper_input=paper_input,
                style_input=style_input,
                length_input=length_input
            )
            logger.info(f"Processing request for: {req.paper_input}")

            # Initialize Model
            model = ChatGoogleGenerativeAI(model=model_name, temperature=temp)

            # Load Template
            # We try to load the local JSON if it exists, otherwise use a fallback
            if os.path.exists('template.json'):
                try:
                    template = load_prompt('template.json')
                    logger.info("Loaded template from template.json")
                except Exception as e:
                    logger.warning(f"Could not load template.json: {e}. Using fallback.")
                    template = ChatPromptTemplate.from_template("Summarize {paper_input} in {style_input} style and {length_input} length.")
            else:
                template = ChatPromptTemplate.from_template("Summarize {paper_input} in {style_input} style and {length_input} length.")

            # Run Chain using LCEL (LangChain Expression Language)
            with st.spinner('Analyzing paper and generating summary...'):
                chain = template | model
                result = chain.invoke(req.dict())
                
                st.markdown("### Summary")
                st.write(result.content)
                logger.info("Summary generated successfully.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            logger.error(f"App error: {e}")

if __name__ == "__main__":
    run_app()