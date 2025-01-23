import streamlit as st

def process_file():
    pass


def display_messages():
    for messages in st.session_state:
        with st.chat_message(messages["role"]):
            st.markdown(messages["content"])
        


def process_input():
    if prompt := st.chat_input("How can I help?"):
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.append({"role": "user", "content": prompt})


def main():
    st.title("Document")

    if len(st.session_state) == 0:
        st.session_state = []

    st.file_uploader(
        "Uploade the document",
        type=["pdf"],
        key="file_uploader",
        on_change=process_file,
        label_visibility="collapsed",
        accept_multiple_files=True
    )
    display_messages()
    process_input()

if __name__ == "__main__":
    main()