# BookLens 📚

BookLens is a PDF-based RAG (Retrieval-Augmented Generation) Question Answering chatbot built using LangChain, ChromaDB, and MistralAI LLM.  
It allows users to upload PDF documents and ask natural language questions, with answers generated directly from the document content.

---

## Features

- PDF document ingestion
- Text chunking and embeddings
- Vector database storage using ChromaDB
- Context-aware question answering
- Retrieval-Augmented Generation (RAG)
- Local document querying
- Built with LangChain

---

## Tech Stack

- Python
- LangChain
- ChromaDB
- HuggingFace Embeddings
- MistralAI
- PyPDF
- Chroma Vector Store
- Streamlit

---

## Project Structure

```bash
.
├── app.py
├── main.py
├── create_database.py
├── requirements.txt
├── document_loaders/
├── chroma_db/
└── .env
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ys09123/BookLens.git
cd BookLens
```

### 2. Create virtual environment

```bash
uv venv
```

### 3. Activate virtual environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / Mac

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
uv pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the root directory:

```
HF_TOKEN=your_huggingface_token
```
and

```
MISTRAL_KEY=your_api_key
```

---

## Run the Project

### Create Vector Database

```bash
python create_database.py
```

### Start the Application

```bash
streamlit run app.py
```

---

## How It Works

1. PDF documents are loaded and processed
2. Text is split into chunks
3. Embeddings are generated
4. Embeddings are stored in ChromaDB
5. User queries are matched against relevant chunks
6. LLM generates answers using retrieved context

---

## Future Improvements

- Multiple PDF uploads
- Chat history memory
- Authentication
- Hybrid search
- Docker deployment

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## License

This project is licensed under the MIT License.

---

## Author
Yash.
