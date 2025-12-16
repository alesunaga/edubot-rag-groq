import streamlit as st
import google.generativeai as genai
from groq import Groq
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import os
import csv
import pandas as pd
import numpy as np
from datetime import datetime
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EduBot RAG - Multi-Provider",
    page_icon="🤖",
    layout="wide"
)

# --- RESOURCE CACHING ---
@st.cache_resource
def load_embedding_model():
    # Local model for embeddings (Runs on CPU, no cost/quota)
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- DIDACTIC CONSTANTS & DESCRIPTIONS ---
ROLES_PROMPTS = {
    "Custom": "You are a helpful and patient tutor. Answer questions based on the provided context.",
    "Learning Instructor": "Your goal is to guide and scaffold. Do not answer immediately; help the student construct knowledge.",
    "Learning Partner": "Act as a peer. Ask questions to check understanding and give feedback.",
    "Learning Assistant": "Support and simplify. Use simple language and short sentences."
}

# Descriptions to help teachers choose
ROLE_DESCRIPTIONS = {
    "Custom": "Standard helpful behavior. Good for general queries.",
    "Learning Instructor": "**The Guide:** Best for introducing new topics. It breaks down complex concepts into steps (Scaffolding) and avoids giving direct answers, encouraging the student to think.",
    "Learning Partner": "**The Peer:** Best for review and practice. It adopts a Socratic method, asking questions back to the student to verify understanding and correct misconceptions.",
    "Learning Assistant": "**The Supporter:** Best for inclusion and differentiation. It simplifies language, summarizes long texts, and helps students with reading difficulties or language barriers."
}

# Safety Prompt (Strict Mode)
STRICT_SYSTEM_PROMPT = """
CRITICAL INSTRUCTION: You are a specialized RAG bot.
You must answer the user's question ONLY using the information provided in the "CONTEXT" section below.
If the answer cannot be found in the context, or if the context is empty, you MUST refuse to answer.
IMPORTANT: The refusal message must be in the target language defined below (e.g., if Portuguese, say "Desculpe, não encontrei essa informação no material.").
Do not use your own outside knowledge. Do not answer questions about general topics (like cooking, history, geography) unless they are explicitly in the context.
"""

LOG_FILE = "chat_logs.csv"

# --- HELPER FUNCTIONS ---
def log_interaction(role, question, answer, model, language, method):
    file_exists = os.path.isfile(LOG_FILE)
    try:
        with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Timestamp", "Role", "Language", "Model", "Method", "Question", "Answer_Length"])
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), role, language, model, method, question, len(answer)])
    except:
        pass

def split_text_into_chunks(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

# --- MODEL FETCHING FUNCTIONS ---

def get_google_models(api_key):
    try:
        genai.configure(api_key=api_key)
        model_list = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                model_list.append(m.name)
        # Prioritize Flash and Pro models
        model_list.sort(key=lambda x: ('flash' not in x, 'pro' not in x, x))
        return model_list
    except:
        return []

def get_groq_models(api_key):
    try:
        client = Groq(api_key=api_key)
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        # Filter for Llama/Mixtral and sort by newness
        text_models = [m for m in model_ids if "llama" in m or "mixtral" in m]
        text_models.sort(key=lambda x: (not x.startswith('llama-3.3'), not x.startswith('llama-3.1'), x))
        return text_models
    except:
        return []

# --- CORE AI FUNCTIONS ---

def get_local_embeddings(chunks):
    model = load_embedding_model()
    embeddings = model.encode(chunks)
    return embeddings

def get_ai_response(provider, api_key, model_name, prompt):
    try:
        if provider == "Google Gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
            
        elif provider == "Groq (Llama 3)":
            client = Groq(api_key=api_key)
            completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_name,
                temperature=0.3
            )
            return completion.choices[0].message.content
            
    except Exception as e:
        return f"AI Error ({provider}): {str(e)}"

# --- SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "rag_chunks" not in st.session_state: st.session_state.rag_chunks = []
if "rag_embeddings" not in st.session_state: st.session_state.rag_embeddings = []
if "simple_text" not in st.session_state: st.session_state.simple_text = ""
if "system_instruction" not in st.session_state: st.session_state.system_instruction = ROLES_PROMPTS["Custom"]
if "api_key" not in st.session_state: st.session_state.api_key = ""
if "provider" not in st.session_state: st.session_state.provider = "Google Gemini"
if "rag_mode" not in st.session_state: st.session_state.rag_mode = "Simple (Context)"
if "target_language" not in st.session_state: st.session_state.target_language = "English"
if "available_models" not in st.session_state: st.session_state.available_models = []
if "strict_mode" not in st.session_state: st.session_state.strict_mode = True

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ AI Configuration")
    role_view = st.selectbox("View Mode", ["Student", "Teacher"])
    
    # PROVIDER SELECTION
    new_provider = st.selectbox("AI Provider", ["Google Gemini", "Groq (Llama 3)"])
    
    # Reset models if provider changes
    if new_provider != st.session_state.provider:
        st.session_state.provider = new_provider
        st.session_state.available_models = []
        st.session_state.api_key = "" 
        st.rerun()
    
    # Dynamic Label
    label = "Gemini API Key" if st.session_state.provider == "Google Gemini" else "Groq API Key"
    user_api_key = st.text_input(label, type="password", value=st.session_state.api_key)
    
    if user_api_key:
        st.session_state.api_key = user_api_key
        
        # Fetch models dynamically
        if not st.session_state.available_models:
            with st.spinner(f"Connecting to {st.session_state.provider}..."):
                if st.session_state.provider == "Google Gemini":
                    models = get_google_models(user_api_key)
                else:
                    models = get_groq_models(user_api_key)
                
                if models:
                    st.session_state.available_models = models
                    st.success(f"Connected! {len(models)} models found.")
                else:
                    st.error("Invalid Key or Connection Error.")
    
    # Model Selector
    selected_model = ""
    if st.session_state.available_models:
        selected_model = st.selectbox("AI Model", st.session_state.available_models)
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- TEACHER DASHBOARD ---
if role_view == "Teacher":
    st.title(f"👨‍🏫 Teacher Dashboard ({st.session_state.provider})")
    
    tab1, tab2 = st.tabs(["📚 Material & Didactics", "📊 Analytics"])
    
    with tab1:
        st.subheader("1. Knowledge Base")
        st.session_state.rag_mode = st.radio("Reading Method:", 
            ["Simple (Short Text)", "Advanced (Books/Courseware)"],
            help="Simple: Fast, no quota usage. Advanced: Uses Local Embeddings (CPU).")
        
        uploaded_file = st.file_uploader("Upload PDF/TXT", type=['txt', 'pdf'])
        
        if uploaded_file and st.button("📥 Process File"):
            raw_text = ""
            if uploaded_file.type == "application/pdf":
                reader = PdfReader(uploaded_file)
                for page in reader.pages: raw_text += page.extract_text()
            else:
                raw_text = uploaded_file.getvalue().decode("utf-8")
            
            if st.session_state.rag_mode == "Simple (Short Text)":
                st.session_state.simple_text = raw_text
                st.session_state.rag_chunks = []
                st.success(f"✅ Text loaded in Simple Mode ({len(raw_text)} characters).")
            else:
                with st.spinner("Processing Local Embeddings (CPU)..."):
                    chunks = split_text_into_chunks(raw_text)
                    embeddings = get_local_embeddings(chunks)
                    st.session_state.rag_chunks = chunks
                    st.session_state.rag_embeddings = np.array(embeddings)
                    st.session_state.simple_text = ""
                    st.success(f"✅ Processing complete! {len(chunks)} fragments indexed.")

        st.divider()
        st.subheader("2. Pedagogical Strategy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            role = st.selectbox("Bot Role", list(ROLES_PROMPTS.keys()))
            # Show the educational description of the selected role!
            st.info(ROLE_DESCRIPTIONS[role])
            
        with col2:
            lang = st.selectbox("Response Language", ["English", "Portuguese", "Spanish", "German"])
            st.caption("The bot will answer in this language, regardless of the student's input language.")
        
        st.session_state.strict_mode = st.checkbox("🔒 Strict Mode (Prevent Hallucinations)", value=st.session_state.strict_mode)

        if st.button("Save Configuration"):
            st.session_state.system_instruction = ROLES_PROMPTS[role]
            st.session_state.target_language = lang
            st.success("Configuration saved!")

    with tab2:
        if os.path.exists(LOG_FILE):
            st.dataframe(pd.read_csv(LOG_FILE).tail(15))

# --- STUDENT INTERFACE ---
elif role_view == "Student":
    st.title("🎓 EduBot")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
        
    if prompt := st.chat_input("Type your question..."):
        if not selected_model:
            st.error("⚠️ Teacher: Please configure the API Key in the sidebar first.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        # Context Retrieval
        context = ""
        method = "None"
        
        if st.session_state.simple_text:
            context = st.session_state.simple_text
            method = "Simple"
        elif st.session_state.rag_chunks:
            with st.spinner("Consulting material..."):
                model_emb = load_embedding_model()
                q_emb = model_emb.encode([prompt])
                scores = np.dot(st.session_state.rag_embeddings, q_emb.T).flatten()
                top_indices = np.argsort(scores)[::-1][:3] # Top 3 chunks
                relevant = [st.session_state.rag_chunks[i] for i in top_indices]
                context = "\n...\n".join(relevant)
                method = "Vector (Local)"
        
        if not context:
            st.warning("The teacher has not uploaded the study material yet.")
            st.stop()
            
        # Prompt Construction
        strict_instruction = STRICT_SYSTEM_PROMPT if st.session_state.strict_mode else ""
        
        full_prompt = f"""
        {strict_instruction}
        DIDACTIC GOAL: {st.session_state.system_instruction}
        
        CONTEXT:
        {context}
        
        QUESTION: {prompt}
        
        OUTPUT INSTRUCTION: Answer strictly in {st.session_state.target_language}.
        """
        
        with st.chat_message("assistant"):
            with st.spinner(f"Thinking ({st.session_state.provider})..."):
                response = get_ai_response(
                    st.session_state.provider, 
                    st.session_state.api_key, 
                    selected_model, 
                    full_prompt
                )
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                log_interaction("Student", prompt, response, selected_model, st.session_state.target_language, method)
