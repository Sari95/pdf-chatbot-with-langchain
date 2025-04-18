# chatbot_core.py: This module defines the core logic of the PDFchatbot.
# It builds a RAG pipeline using LangChain.

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain

def build_qa_chain(pdf_path="example.pdf"):
    loader = PyPDFLoader(pdf_path) # Loads the PDF
    documents = loader.load()[1:]  # Skip page 1 (element 0)

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100) # Generates chunks of the document
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # Generates vector embeddings for each chunk
    db = FAISS.from_documents(docs, embeddings) # Stores the chunks in a FAISS vector db for similarity search
    retriever = db.as_retriever() # Create a retriever to find relevant chunks based on a question

    llm = ChatOllama(model="mistral") # Combines the retriever with mistral
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain # The function 'qa_chain()' returns a ready-to-use question-answering chain
