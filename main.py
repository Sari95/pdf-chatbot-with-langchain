#!/usr/bin/env python

from core import build_qa_chain

"""
This file contains the code to interact with the PDF chatbot

The file is mainly intended for testing, debugging or if no web interface is needed

The chatbot uses a RAG pipeline that is defined in chatbot_core.py
"""


def setup_qa_chain(pdf_path="example.pdf"):
    """
    Build QA chain from the specified PDF file

    Args:
        pdf_path: Path to the PDF file to load

    Returns:
        The QA chain object
    """
    return build_qa_chain(pdf_path)


def chat_loop(qa_chain):
    """
    Run the interactive chat loop with the user

    Args:
        qa_chain: The QA chain to use for answering questions
    """
    # Initializes an empty list to store the chat history
    chat_history = []

    # Prints the welcome message to the terminal
    print("🧠 PDF-Chatbot started! Enter 'exit' to quit.")

    # Starts a loop to allow the user to ask questions continuously
    while True:
        query = input("\n❓ Your questions: ")
        # Breaks the loop if the user types 'exit' or 'quit'
        if query.lower() in ["exit", "quit"]:
            print("👋 Chat finished.")
            break

        # Get the answer from the QA chain (LLM + Retriever) and prints the answer to the terminal
        result = qa_chain.invoke({"question": query, "chat_history": chat_history})
        print("\n💬 Answer:", result["answer"])

        # Saves the Q&A pair in the chat history
        chat_history.append((query, result["answer"]))

        # Shows a snippet from the source document that is used
        print("\n🔍 Source – Document snippet:")
        print(result["source_documents"][0].page_content[:300])


def main():
    try:
        qa_chain = setup_qa_chain()
        chat_loop(qa_chain)
    except KeyboardInterrupt:
        print("\n\n👋 Bye!")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
