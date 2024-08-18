# Chatbot-for-Customer-Support

A customer support chatbot powered by a fine-tuned language model and Retrieval-Augmented Generation (RAG) system. This chatbot is trained on a mixed dataset and can answer queries based on website content and PDF documents.

## Description

This project implements a customer support chatbot using a Large Language Model (LLM) fine-tuned on a mixed dataset. The dataset combines samples from lmsys/lmsys-chat-1m and bitext/Bitext-customer-support-llm-chatbot-training-dataset. The fine-tuned model is uploaded to Hugging Face, and a RAG system is implemented to answer user queries based on website content and PDF documents.

## Main Features

1. Content Loading: Ability to load content from PDFs and websites.
2. Text Processing: Splits documents into manageable chunks for processing.
3. Embedding Generation: Creates embeddings for text chunks using Hugging Face models.
4. Vector Storage: Stores embeddings in a Chroma vector database for efficient retrieval.
5. Query Processing: Uses a fine-tuned LLaMA 2 model to generate responses to user queries.
6. Web Interface: Provides a Streamlit-based web interface for easy interaction with the chatbot.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/Chatbot-for-Customer-Support.git
   cd Chatbot-for-Customer-Support
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Enter your question in the text input field and click "Get Answer" to receive a response from the chatbot.

## Dependencies

All required dependencies are listed in the `requirements.txt` file. Key dependencies include:

- streamlit
- langchain
- transformers
- beautifulsoup4
- PyPDF2
- requests
- chromadb

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.


## Acknowledgements

- This project uses the LLaMA 2-7B model fine-tuned for customer support tasks.
- Dataset sources: lmsys/lmsys-chat-1m and bitext/Bitext-customer-support-llm-chatbot-training-dataset
