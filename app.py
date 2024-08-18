import streamlit as st
from rag_system import RAGSystem

def main():
    st.title("Customer Support System For Tymeline")

    # Initialize the RAG system
    @st.cache_resource
    def load_rag_system():
        rag = RAGSystem('nitishpawar11/finetuned-llama-2-7b-customer-support')
        rag.process_and_store()
        return rag

    rag = load_rag_system()

    # Create a text input for the user's query
    user_query = st.text_input("Enter your question:", "")

    if st.button("Get Answer"):
        if user_query:
            with st.spinner("Generating answer..."):
                answer = rag.answer_query(user_query)
                st.write("Answer:", answer)
        else:
            st.warning("Please enter a question.")

    st.sidebar.header("About")
    st.sidebar.info(
        "This app uses a Retrieval-Augmented Generation (RAG) system "
        "with a fine-tuned LLaMA 2 model for customer support. "
        "Enter your question in the text box and click 'Get Answer' to receive a response."
    )

if __name__ == "__main__":
    main()