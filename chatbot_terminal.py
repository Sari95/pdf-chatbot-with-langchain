# main.py: This file contains the code to interact with the PDF chatbot
# The file is mainly intended for testing, debugging or if no web interface is needed
# The chatbot uses a RAG pipeline that is defined in chatbot_core.py

from chatbot_core import build_qa_chain # Imports the RAG pipeline builder from chatbot_core.py

qa_chain = build_qa_chain("example.pdf") #Builds the QA chain using a local PDF file
chat_history = [] #Initializes an empty list to store the chat history

print("🧠 PDF-Chatbot started! Enter ‘exit’ to quit.") # Prints the welcome message to the terminal

# Starts a loop to allow the user to ask questions continuously
while True:
    query = input("\n❓ Your questions: ")
    # Breaks the loop if the user types 'exit' or 'quit'
    if query.lower() in ["exit", "quit"]:
        print("👋 Chat finished.")
        break

    # Get the answer from the QA chain (LLM + Retriever) and prints the answer to the terminal
    result = qa_chain({"question": query, "chat_history": chat_history})
    print("\n💬 Answer:", result["answer"])
    chat_history.append((query, result["answer"])) #Saves the Q&A pair in the chat history
    print("\n🔍 Source – Document snippet:") #Shows a snippet from the source document that is used
    print(result["source_documents"][0].page_content[:300])
