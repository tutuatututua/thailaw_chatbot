from operator import itemgetter
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import os

# =========================================================
# Streamlit Setup
# =========================================================

st.set_page_config(page_title="Thai Law Chatbot", page_icon="⚖️", layout="wide")
st.title("⚖️ Thai Law Chatbot – RAG with Continuous Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================================================
# Load RAG Components (cached)
# =========================================================
@st.cache_resource(show_spinner=True)
def load_rag():

    # ----- Embeddings -----
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # ----- Vector DB -----
    db = Chroma(
        collection_name="langchain",
        persist_directory="chroma_db",
        embedding_function=embeddings,
    )

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )

    # ----- LLM -----
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    # llm = ChatOllama(model="deepseek-r1:8b", temperature=0.2)

    # ======================================================
    # QUESTION REWRITER → Convert user Thai to legal Thai
    # ======================================================
    rewrite_prompt = ChatPromptTemplate.from_template("""
    แปลงประโยคต่อไปนี้ให้เป็นภาษากฎหมายทางการของไทย
    เน้นศัพท์กฎหมายและสำนวนทางราชการ ใช้ในการค้นหาข้อมูลกฎหมาย

    ประโยค: {question}
    
    ให้ตอบเฉพาะประโยคที่แปลงแล้วเท่านั้น
    """)
    question_rewriter = rewrite_prompt | llm | StrOutputParser()

    # ======================================================
    # Final Prompt
    # ======================================================
    prompt = ChatPromptTemplate.from_template("""
    คุณคือผู้เชี่ยวชาญด้านกฎหมายไทย ตอบคำถามโดยอ้างอิงจากข้อมูลที่ให้ไป
    หากไม่พบข้อมูล ให้ตอบว่า "ฉันไม่พบข้อมูลที่เกี่ยวข้อง"

    ประวัติการสนทนาก่อนหน้า:
    {history}

    --------------------
    ข้อมูล:
    {context}
    --------------------

    คำถาม: {question}
    """)

    # ======================================================
    # RAG chain
    # ======================================================
    rag_chain = (
        {
            "context": itemgetter("rewritten") | retriever,
            "question": itemgetter("rewritten"),
            "history": itemgetter("history")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever, question_rewriter


def get_history_text():
    history_lines = []
    for msg in st.session_state.messages[-10:]:   # last 10 messages
        role = "ผู้ใช้" if msg["role"] == "user" else "ผู้ช่วย"
        history_lines.append(f"{role}: {msg['content']}")
    return "\n".join(history_lines)


rag_chain, retriever, question_rewriter = load_rag()

# =========================================================
# Ask Function
# =========================================================
def ask(question: str):
    history_text = get_history_text()

    # Rewrite question → legal Thai
    rewritten = question_rewriter.invoke({"question": question})

    # Run RAG using rewritten version
    answer = rag_chain.invoke({
        "rewritten": rewritten,
        "history": history_text
    })

    # 3️⃣ Retrieve based on rewritten version
    sources = retriever.invoke(rewritten)

    return answer, sources, rewritten


# =========================================================
# Display Previous Chat Messages
# =========================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =========================================================
# Chat Input
# =========================================================
user_input = st.chat_input("พิมพ์คำถามเกี่ยวกับกฎหมายไทย...")

if user_input:

    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run RAG
    answer, sources, rewritten = ask(user_input)

    # Format sources
    source_list = "\n".join(
        f"- {doc.metadata.get('title', 'ไม่ทราบแหล่งที่มา')}"
        for doc in sources
    )

    final_answer = (
        f"**🔎 คำถามที่ปรับให้อยู่ในรูปภาษากฎหมาย:**\n{rewritten}\n\n"
        + answer
        + "\n\n---\n**แหล่งอ้างอิง:**\n"
        + source_list
    )

    # Store assistant message
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
    with st.chat_message("assistant"):
        st.markdown(final_answer)
