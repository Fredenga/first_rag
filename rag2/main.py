import streamlit as st
import tempfile

def process_file():
    # Ensure "messages" is initialized as an empty list if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    # Process the uploaded files
    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

def display_messages():
    # Display the stored messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def process_input():
    # Process user input and generate response
    if prompt := st.chat_input("How can I help?"):
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add the user message to session state
        st.session_state["messages"].append({"role": "user", "content": prompt})
    
    # Generate assistant response
    response = generate_response()
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add the assistant's response to session state
    st.session_state["messages"].append({"role": "assistant", "content": response})

def generate_response():
    return "Hello, how can I help you today..."

def main():
    st.title("Document")

    # Initialize "messages" in session state if not already initialized
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # File uploader
    st.file_uploader(
        "Upload the document",
        type=["pdf"],
        key="file_uploader",
        on_change=process_file,
        label_visibility="collapsed",
        accept_multiple_files=True
    )
    
    # Display existing messages
    display_messages()

    # Process user input and generate a response
    process_input()

if __name__ == "__main__":
    main()
