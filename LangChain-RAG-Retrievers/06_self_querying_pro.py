import logging
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_core.documents import Document
from dotenv import load_dotenv

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

"""
###############################################################################
### EXPERT: SELF-QUERY RETRIEVER ###
###############################################################################

1. THE PROBLEM:
   - Vector search is great for similarity, but terrible for specific constraints.
   - Example: "Find movies about aliens made after 2010". 
   - Vector search might find an alien movie from 1980 because it's semantically 
     similar, ignoring the "after 2010" part.

2. THE SOLUTION:
   - SelfQueryRetriever uses an LLM to parse the query into:
     a) A semantic search string ("aliens").
     b) A structured filter ({ "year": { "$gt": 2010 } }).

3. REQUIREMENT:
   - You must define the "Metadata Field Info" so the LLM knows which 
     fields are available for filtering.
###############################################################################
"""

def self_query_demo():
    try:
        embeddings = OpenAIEmbeddings()
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        # 1. Setup sample data with rich metadata
        docs = [
            Document(page_content="A movie about friendly aliens.", metadata={"year": 2012, "rating": 8.5, "genre": "Sci-Fi"}),
            Document(page_content="A scary movie about space monsters.", metadata={"year": 1979, "rating": 9.2, "genre": "Horror"}),
            Document(page_content="A documentary about the stars.", metadata={"year": 2021, "rating": 7.0, "genre": "Documentary"})
        ]
        vectorstore = Chroma.from_documents(docs, embeddings)

        # 2. Define Metadata Field Information
        # This tells the LLM what 'year', 'rating', and 'genre' are.
        metadata_field_info = [
            AttributeInfo(name="genre", description="The genre of the movie", type="string"),
            AttributeInfo(name="year", description="The year the movie was released", type="integer"),
            AttributeInfo(name="rating", description="A 1-10 rating for the movie", type="float"),
        ]
        document_content_description = "Brief summary of a movie"

        # 3. Create Retriever
        logger.info("Initializing SelfQueryRetriever...")
        retriever = SelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=vectorstore,
            document_contents=document_content_description,
            metadata_field_info=metadata_field_info,
            verbose=True # See the generated query in terminal
        )

        # 4. Search with a complex constraint
        query = "Show me Sci-Fi movies released after 2010"
        logger.info(f"Query: {query}")
        
        results = retriever.invoke(query)

        print("\n" + "="*30)
        print("SELF-QUERY RESULTS")
        print("="*30)
        for doc in results:
            print(f"Movie: {doc.page_content} | Metadata: {doc.metadata}")

    except Exception as e:
        logger.error(f"Self-querying failed: {str(e)}")

if __name__ == "__main__":
    self_query_demo()
