"""
Streamlit Chat Interface for DeepSeek R1 Distill RAG System

This module provides a web-based chat interface using Streamlit that allows users
to interact with the RAG (Retrieval Augmented Generation) system powered by 
DeepSeek R1 and SmolAgents.

The interface provides:
- Interactive chat with document-based Q&A
- Chat history management
- Sidebar with information and controls
- Real-time responses using the RAG pipeline

Author: AI-CIS Project
Dependencies: streamlit, r1_smolagent_rag module
"""

import streamlit as st
from r1_smolagent_rag import primary_agent

def init_chat_history():
    """
    Initialize the chat history in Streamlit session state.
    
    Creates an empty list to store chat messages if it doesn't exist.
    This ensures chat history persists across Streamlit reruns.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []

def display_chat_history():
    """
    Display all previous chat messages in the Streamlit interface.
    
    Iterates through the session state messages and displays each one
    with appropriate chat message styling (user/assistant roles).
    Uses Streamlit's chat_message component for proper formatting.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input(prompt: str):
    """
    Process user input and generate AI response using the RAG system.
    
    This function:
    1. Adds the user's message to chat history
    2. Displays the user message in the chat interface
    3. Calls the primary_agent (RAG system) to generate a response
    4. Displays the AI response with a loading spinner
    5. Adds the AI response to chat history
    
    Args:
        prompt (str): The user's question or input text
    """
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response using the RAG system
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):  # Show loading indicator
            # Call the SmolAgents RAG system to process the query
            response = primary_agent.run(prompt, reset=False)
            st.markdown(response)
            # Add AI response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

def display_sidebar():
    """
    Create and display the sidebar with information and controls.
    
    The sidebar contains:
    - Information about how the RAG system works
    - Clear chat history button for resetting conversations
    - Educational content about the RAG process
    """
    with st.sidebar:
        st.title("About")
        st.markdown("""
        This Q&A bot uses RAG (Retrieval Augmented Generation) to answer questions about your documents.
        
        **The RAG Process:**
        1. **Document Ingestion**: PDFs are processed and stored in vector database
        2. **Query Processing**: Your question is converted to embeddings
        3. **Similarity Search**: Most relevant document chunks are retrieved
        4. **Reasoning**: DeepSeek R1 model generates answer based on context
        5. **Response**: Final answer is presented with reasoning traces
        
        **Models Used:**
        - **Reasoning Model**: DeepSeek R1 (for deep thinking and analysis)
        - **Tool Model**: Llama 3.1 (for tool calling and coordination)
        - **Embeddings**: all-mpnet-base-v2 (for semantic similarity)
        """)
        
        # Button to clear chat history
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()  # Refresh the interface

def main():
    """
    Main function that sets up and runs the Streamlit application.
    
    This function:
    1. Configures the Streamlit page settings
    2. Sets up the page title and layout
    3. Initializes the chat history
    4. Displays the chat interface components
    5. Handles user input through the chat input widget
    
    The interface uses Streamlit's chat components for a modern,
    ChatGPT-like user experience.
    """
    # Configure Streamlit page settings
    st.set_page_config(
        page_title="DeepSeek R1 RAG Q&A Bot", 
        layout="wide",
        page_icon="🤖"
    )
    
    # Main page title and description
    st.title("🤖 DeepSeek R1 Document Q&A Bot")
    st.markdown("""
    Ask questions about your documents and get intelligent answers powered by 
    **DeepSeek R1** reasoning and **RAG** (Retrieval Augmented Generation).
    """)

    # Initialize chat history for the session
    init_chat_history()

    # Display existing chat messages
    display_chat_history()
    
    # Display sidebar with information and controls
    display_sidebar()

    # Chat input widget - handles user input
    if prompt := st.chat_input("Ask a question about your documents..."):
        handle_user_input(prompt)

if __name__ == "__main__":
    """
    Entry point for the Streamlit application.
    
    When this script is run directly (not imported), it starts the
    Streamlit web interface for the RAG Q&A system.
    
    To run this application:
    1. Ensure all dependencies are installed (pip install -r requirements.txt)
    2. Make sure the vector database is created (run ingest_pdfs.py first)
    3. Run: streamlit run streamlit.py
    4. Open your browser to the provided URL (usually http://localhost:8501)
    """
    main()
