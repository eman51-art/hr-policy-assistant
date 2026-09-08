# HR Policy Assistant — RAG

## About the Project

**HR Policy Assistant** is an AI-powered web application built with **Streamlit** that allows users to upload an HR Policy PDF and ask questions about its contents.

The application uses **Retrieval-Augmented Generation (RAG)** to provide answers based on the information available in the uploaded document.

Instead of asking the AI model to answer from its general knowledge, the application first searches the uploaded HR Policy for relevant information and then provides that information to the AI model as context. This helps the assistant give answers that are grounded in the actual HR Policy document.

## Features

* 📄 Upload HR Policy PDF directly through the web interface
* 🔎 Extract text from PDF using PyMuPDF
* ✂️ Split document text into smaller overlapping chunks
* 🧠 Generate semantic embeddings using Sentence Transformers
* ⚡ Store and search embeddings using FAISS
* 🎯 Retrieve the most relevant policy sections
* 🤖 Generate answers using Groq and `openai/gpt-oss-20b`
* 💬 Simple and user-friendly Streamlit interface
* 🚫 Avoid making up information when the answer is not available in the uploaded policy

## How It Works

The application follows a **Retrieval-Augmented Generation (RAG)** pipeline.

### 1. Upload PDF
The user uploads an HR Policy PDF through the Streamlit interface.

### 2. Extract Text
**PyMuPDF** reads the PDF and extracts the text from its pages.

### 3. Create Text Chunks
The extracted text is divided into smaller chunks, created with some **overlap** so important information isn't lost when a piece of text is split between two chunks.

### 4. Generate Embeddings
Each text chunk is converted into a numerical representation called an **embedding** using the Sentence Transformer model `all-MiniLM-L6-v2`. These embeddings represent the meaning of the text.

### 5. Create FAISS Index
The embeddings are stored in a **FAISS index**, which allows the application to quickly search for text chunks that are semantically similar to a user's question.

### 6. Ask a Question
The user enters a question about the HR Policy, e.g. *"How many annual leaves are employees allowed?"*

### 7. Retrieve Relevant Information
The question is converted into an embedding and searched against the FAISS index. The application retrieves the most relevant sections from the uploaded HR Policy.

### 8. Generate the Answer
The retrieved policy sections are provided to the **Groq LLM** (`openai/gpt-oss-20b`) along with the user's question. The model is instructed to use the retrieved policy information as context and not invent information that isn't present in the document.

### 9. Display the Answer
The final answer is displayed to the user through the Streamlit interface.

## RAG Workflow

```text
HR Policy PDF
     ↓
Text Extraction
     ↓
Text Chunking
     ↓
Sentence Transformer
(all-MiniLM-L6-v2)
     ↓
Text Embeddings
     ↓
FAISS Index
     ↓
User Question
     ↓
Question Embedding
     ↓
FAISS Similarity Search
     ↓
Relevant Policy Chunks
     ↓
Groq LLM
(openai/gpt-oss-20b)
     ↓
Final Answer
```

## Technologies Used

| Technology            | Purpose                  |
| ---------------------- | ------------------------ |
| Python                | Application development  |
| Streamlit             | Web interface            |
| PyMuPDF               | PDF text extraction      |
| Sentence Transformers | Generate text embeddings |
| FAISS                 | Similarity search        |
| Groq                  | LLM API                  |
| `openai/gpt-oss-20b`  | Generate answers         |

## Project Structure

```text
HR-Policy-Assistant/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## API Key Setup

This app needs a Groq API key to work. **Never** put your API key inside `app.py` or commit it to GitHub.

Add it as a Streamlit Cloud Secret instead:

1. Go to your app's **Settings → Secrets** on Streamlit Cloud
2. Add:

```toml
GROQ_API_KEY = "your_api_key_here"
```

3. Save — that's it.

## Example Questions

Users can ask questions such as:

* What is the annual leave policy?
* How many sick leaves are allowed?
* What is the company's working hours?
* What is the maternity leave policy?
* What is the company's remote work policy?
* What are the rules for employee resignation?

The assistant retrieves the relevant information from the uploaded HR Policy before generating the answer.

## Key Benefit

The main goal of the HR Policy Assistant is to make large HR policy documents easier to search and understand. Instead of manually reading the entire document, employees can upload the policy and ask questions in natural language to quickly find the relevant information.
