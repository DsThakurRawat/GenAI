import logging
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from dotenv import load_dotenv

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

"""
###############################################################################
### INTERMEDIATE: MULTI-QUERY RETRIEVER ###
###############################################################################

1. THE PROBLEM:
   - Users often ask questions in vague or "non-optimal" ways.
   - Example: "How to stay fit?" might miss a document that says 
     "Daily exercise routines for health."

2. THE SOLUTION:
   - MultiQueryRetriever uses an LLM to generate 3-5 variations of the query.
   - It runs all of them against the vector store.
   - It combines and deduplicates the results.

3. ADVANTAGE:
   - It overcomes the limitations of purely distance-based vector search 
     by looking at the question from different perspectives.
###############################################################################
"""

def multi_query_demo():
    try:
        embeddings = OpenAIEmbeddings()
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        # 1. Setup sample data
        docs = [
            Document(page_content="Walking for 30 minutes a day significantly improves cardiovascular health."),
            Document(page_content="Eating leafy greens provides essential vitamins and fiber."),
            Document(page_content="Consistent strength training builds long-term bone density.")
        ]
        vector_store = Chroma.from_documents(docs, embeddings)

        # 2. Initialize MultiQueryRetriever
        logger.info("Initializing MultiQueryRetriever...")
        retriever = MultiQueryRetriever.from_llm(
            retriever=vector_store.as_retriever(),
            llm=llm
        )

        # 3. Search
        question = "How can I maintain my health?"
        logger.info(f"Original Question: {question}")
        
        # This will trigger the LLM to rewrite the query
        # You can see the variations in the logs if you enable DEBUG logging
        results = retriever.invoke(question)

        print("\n" + "="*30)
        print("MULTI-QUERY RESULTS")
        print("="*30)
        print(f"Retrieved {len(results)} relevant documents by expanding the query.")
        for i, doc in enumerate(results):
            print(f"{i+1}. {doc.page_content}")

    except Exception as e:
        logger.error(f"Multi-query retrieval failed: {str(e)}")

if __name__ == "__main__":
    multi_query_demo()
