from langchain import embeddings
from langchain import vectorstores as vs
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_TYPE_LIST = ["HF", "OPENAI"]
VECTORSTORE_TYPE_LIST = ["FAISS", "CHROMA"]

def get_embedding(embedding_type="HF", OPENAI_KEY=None):
    
    embedding_model = None
    embedding_type = embedding_type.upper() if embedding_type.upper() in EMBEDDING_TYPE_LIST else EMBEDDING_TYPE_LIST[0]

    if embedding_type == "HF":
        embedding_model = embeddings.HuggingFaceEmbeddings()

    elif embedding_type == "OPENAI":
        if OPENAI_KEY:
            embedding_model = embeddings.OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
        else:
            print("OPENAI API KEY Required")

    return embedding_model
    
def store_and_get_vector_db(text_data, documents, vectorstore_type = "FAISS", embedding_type="HF", OPENAI_KEY=None):
    embedding_model = get_embedding(embedding_type=embedding_type, OPENAI_KEY=OPENAI_KEY)
    vectorstore_type = vectorstore_type.upper() if vectorstore_type.upper() in VECTORSTORE_TYPE_LIST else VECTORSTORE_TYPE_LIST[0]
    if vectorstore_type == "FAISS":
        vector_db = vs.FAISS
    elif vectorstore_type == "CHROMA":
        vector_db = vs.Chroma
    db = None
    if text_data:
        try:
            db = vector_db.from_texts(text_data, embedding_model)
        except Exception as e:
            print(f"Error in vector db from_texts: {e}")
            db = None

    elif documents:
        try:
            db = vector_db.from_documents(documents, embedding_model)
        except Exception as e:
            print(f"Error in vector db from_documents: {e}")
            db = None

    return db

def get_similar_docs(db, question, with_score=False):
    similar_docs = None
    if db is not None:
        if with_score:
            similar_docs = db.similarity_search_with_relevance_scores(question)
        else:
            similar_docs = db.similarity_search(question)
    return similar_docs
