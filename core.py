from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

"""
This module defines the core logic of the PDFchatbot.

It builds a RAG pipeline using LangChain.
"""


def build_qa_chain(pdf_path="example.pdf"):
    loader = PyPDFLoader(pdf_path)  # Loads the PDF
    documents = loader.load()[1:]   # Skip page 1 (element 0)

    # Generates chunks of the document
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    # Generates vector embeddings for each chunk
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Stores the chunks in a FAISS vector db for similarity search
    db = FAISS.from_documents(docs, embeddings)
    # Create a retriever to find relevant chunks based on a question
    retriever = db.as_retriever()

    # Combines the retriever with mistral
    llm = ChatOllama(model="mistral")
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=True)

    # The function 'qa_chain()' returns a ready-to-use question-answering chain
    return qa_chain
