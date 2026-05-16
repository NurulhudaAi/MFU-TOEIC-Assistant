# 🏢 MFU TOEIC Assistant (RAG System)

An AI-powered assistant designed to provide accurate information regarding **TOEIC examination regulations** at **Mae Fah Luang University (MFU)**. This project utilizes Retrieval-Augmented Generation (RAG) to ensure responses are grounded strictly in official university guidelines.

### 📖 Overview
The **MFU TOEIC Assistant** helps students and the general public navigate the complexities of test registration, required documents, and campus-specific rules. By leveraging the **Llama-3.3-70b** model via Groq, the system provides bilingual support (Thai/English) with high accuracy.

### ✨ Key Features
* **Bilingual Q&A**: Automatically detects and handles queries in both Thai and English.
* **MFU Specific Logic**: Includes details on the **785+ score exemption** for the Communicative English course.
* **Real-time Retrieval**: Uses a FAISS vector database to fetch relevant context before generating an answer.
* **User-Friendly Interface**: Accessible via a terminal-based CLI or a Streamlit web application.

### 🛠️ Technical Stack
* **LLM**: Llama-3.3-70b-versatile (via Groq Cloud API).
* **Framework**: LangChain (0.2+).
* **Vector Database**: FAISS (Facebook AI Similarity Search).
* **Embeddings**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
* **Frontend**: Streamlit / Gradio.

### 🚀 Getting Started

#### Prerequisites
* Python 3.10+
* A **Groq API Key** (Set in Streamlit Secrets or Environment Variables)

#### Installation
1. **Clone the repository**:
   ```bash
   git clone TOEIC_RAG_QA_Bilingual
   cd mfu-toeic-assistant

   pip install -r requirements.txt

   streamlit run app.py

### 👥 Contributors (Group 5)
* **Nurulhuda Adam Ishaq** 
* **Jiranya Arun** 
* **Chanisara Pathrugsa** 
* **Phloiphailin Khampuk** 
* **Chaw Chaw Latt** 
* **Yoon Thedar Cho** 

---

### ⚠️ Disclaimer
This assistant is a student project and should be used for informational purposes only. Please refer to the official [MFU TOEIC website](https://toeic.mfu.ac.th) for legally binding regulations.

