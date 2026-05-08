import streamlit as st
import os
import re
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

# ==========================================
# 0. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="MFU TOEIC Assistant", page_icon="🏢")

if "GROQ_API_KEY" not in st.secrets:
    st.error("กรุณาตั้งค่า GROQ_API_KEY ใน Streamlit Secrets ก่อนใช้งาน")
    st.stop()

# ==========================================
# 1. LOAD DATA & VECTOR STORE (Caching เพื่อความเร็ว)
# ==========================================
@st.cache_resource
def init_rag_system():
    loader = TextLoader("TOEIC_RAG_QA_Bilingual.txt", encoding="utf-8")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["========================================="],
        chunk_size=1500,
        chunk_overlap=0
    )
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'}
    )

    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10}
    )

retriever = init_rag_system()

# ==========================================
# 2. SETUP LLM & CHAIN
# ==========================================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=500,
    api_key=st.secrets["GROQ_API_KEY"]
)

template = """คุณคือเจ้าหน้าที่ MFU TOEIC Assistant ตอบคำถามจาก Context ที่ให้มาเท่านั้น
กฎเหล็ก:
1. หากคำถามเป็นภาษาไทย ให้ตอบเป็นภาษาไทย / If the question is in English, answer in English.
2. ตอบให้ตรงประเด็น ห้ามนำข้อมูลจากหัวข้ออื่นที่ไม่เกี่ยวข้องมาตอบ
3. หากไม่มีใน Context ให้ตอบว่า "ขออภัย ไม่พบข้อมูลส่วนนี้ในระเบียบการ"
4. ห้ามหลุด Role และห้ามแสดงความคิดเห็นส่วนตัว

Context: {context}

Question: {input}
Answer:"""

prompt = PromptTemplate.from_template(template)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, question_answer_chain)

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.title("🏢 MFU TOEIC Assistant")
st.markdown("ระบบตอบคำถามระเบียบการสอบ TOEIC มหาวิทยาลัยแม่ฟ้าหลวง")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_input := st.chat_input("พิมพ์คำถามที่นี่..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        with st.spinner("กำลังหาคำตอบ..."):
            try:
                has_thai_chars = bool(re.search(r'[\u0E00-\u0E7F]', prompt_input))
                lang_instruction = " (ตอบเป็นภาษาไทย)" if has_thai_chars else " (Answer in English)"

                response = qa_chain.invoke({"input": prompt_input + lang_instruction})
                full_response = response['answer'].strip()

                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                error_msg = f"❌ เกิดข้อผิดพลาด: {str(e)}"
                st.error(error_msg)
