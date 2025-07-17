import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TORCH_HOME'] = '/tmp/torch'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers'

import streamlit as st
import hashlib
import json
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# --- Authentication Configuration ---
USERS = dict(st.secrets["users"])

def hash_password(password):
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def check_password():
    """Returns True if the user has entered a correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        username = st.session_state["username"]
        password = st.session_state["password"]
        
        if username in USERS and hash_password(password) == USERS[username]:
            st.session_state["password_correct"] = True
            st.session_state["authenticated_user"] = username
            del st.session_state["password"]  # Don't store password
            del st.session_state["username"]  # Don't store username
        else:
            st.session_state["password_correct"] = False

    # Return True if password is validated
    if st.session_state.get("password_correct", False):
        return True

    # Show login form
    st.markdown("## 🔐 Login")
    st.text_input("Username", key="username")
    st.text_input("Password", type="password", key="password")
    st.button("Login", on_click=password_entered)
    
    if "password_correct" in st.session_state:
        if not st.session_state["password_correct"]:
            st.error("😕 Username or password incorrect")
    
    return False


# --- Main App ---
if check_password():
    
    # Logout button
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Show welcome message
    st.sidebar.success(f"Welcome, {st.session_state.get('authenticated_user', 'User')}!")
    
    # --- Your Original App Code Below ---
    
    # Custom CSS
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        .main {
            background: linear-gradient(135deg, #f5f1eb 0%, #f8f6f0 100%);
            font-family: 'Inter', sans-serif;
        }
        
        /* Hide default streamlit elements */
        .stDeployButton {display: none;}
        header[data-testid="stHeader"] {display: none;}
        .stMainBlockContainer {padding-top: 2rem;}
        
        /* Custom Title */
        .custom-title {
            font-family: 'Inter', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            color: #2d2d2d;
            text-align: center;
            margin-bottom: 2rem;
            letter-spacing: -0.02em;
        }
        
        /* Button Container */
        .button-container {
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
            justify-content: center;
        }
        
        /* Custom Buttons */
        .custom-button {
            background: linear-gradient(135deg, #d4c4b0 0%, #c9b899 100%);
            border: none;
            border-radius: 12px;
            padding: 0.8rem 2rem;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            font-size: 1rem;
            color: #2d2d2d;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
            min-width: 160px;
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .custom-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(0, 0, 0, 0.12);
            background: linear-gradient(135deg, #c9b899 0%, #bfa888 100%);
        }
        
        .custom-button:active {
            transform: translateY(0);
        }
        
        /* Input Container */
        .input-container {
            background: white;
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
            margin-bottom: 2rem;
            border: 1px solid #e8e2d8;
        }
        
        /* Custom Input */
        .stTextInput > div > div > input {
            font-family: 'Inter', sans-serif;
            font-size: 1.1rem;
            border: 2px solid #e8e2d8;
            border-radius: 12px;
            padding: 1rem;
            background: #fefdfb;
            color: #2d2d2d;
            transition: all 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #d4c4b0;
            box-shadow: 0 0 0 3px rgba(212, 196, 176, 0.15);
            outline: none;
        }
        
        /* Input Label */
        .stTextInput > label {
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            color: #2d2d2d;
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
        }
        
        /* Output Container */
        .output-container {
            background: white;
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
            border: 1px solid #e8e2d8;
            margin-top: 2rem;
        }
        
        /* Output Text */
        .output-text {
            font-family: 'Inter', sans-serif;
            font-size: 1.1rem;
            line-height: 1.7;
            color: #2d2d2d;
            white-space: pre-wrap;
        }
        
        /* Warning Messages */
        .stAlert {
            border-radius: 12px;
            font-family: 'Inter', sans-serif;
            border: none;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        }
        
        /* Chat History */
        .chat-message {
            background: #fefdfb;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border-left: 4px solid #d4c4b0;
            font-family: 'Inter', sans-serif;
            line-height: 1.6;
            color: #2d2d2d;
        }
        
        .chat-message.user {
            background: #f5f3f0;
            border-left-color: #a69483;
        }
        
        .chat-message.assistant {
            background: white;
            border-left-color: #d4c4b0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        
        /* Section Headers */
        .section-header {
            font-family: 'Inter', sans-serif;
            font-size: 1.3rem;
            font-weight: 600;
            color: #2d2d2d;
            margin-bottom: 1rem;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

    # --- Setup ---
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    torch.cuda.is_available = lambda: False
    torch.set_default_device('cpu')

    @st.cache_resource
    def load_model():
        return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device='cpu')

    model = load_model()

    # Load index + metadata
    index = faiss.read_index("data/posts_index.faiss")
    with open("data/posts_metadata.json", "r", encoding="utf-8") as f:
        posts_data = json.load(f)

    about_the_user = "Você é uma criadora de conteúdo dona de um estúdio de design, focada em identidade visual e postagens para Instagram."

    # --- Init Session State ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "system", "content": about_the_user}
        ]
    if "new_post_mode" not in st.session_state:
        st.session_state.new_post_mode = True

    # --- Title ---
    st.markdown('<h1 class="custom-title">Criador de Posts para Instagram</h1>', unsafe_allow_html=True)

    # --- User Input ---
    user_input = st.text_input("Digite o tema do post ou um ajuste no anterior", key="user_input")

    # --- Buttons ---
    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        button_col1, button_col2 = st.columns(2)
        
        with button_col1:
            if st.button("Novo Post", key="new_post", help="Criar um novo post"):
                st.session_state.chat_history = [
                    {"role": "system", "content": about_the_user}
                ]
                st.session_state.new_post_mode = True
        
        with button_col2:
            send_button = st.button("Enviar", key="send", help="Enviar mensagem")

    # --- Process Input ---
    if send_button:
        if user_input.strip() == "":
            st.warning("Por favor, digite algo.")
            
        else:
            if st.session_state.new_post_mode:
                # Novo post: roda RAG
                query_embedding = model.encode([user_input])[0]
                D, I = index.search(np.array([query_embedding]).astype("float32"), k=3)
                examples = [posts_data[i] for i in I[0]]

                # Monta prompt com exemplos
                rag_prompt = "Baseado nos exemplos abaixo, crie uma nova legenda sobre: " + user_input + "\n\n"
                for i, ex in enumerate(examples, 1):
                    rag_prompt += f"Exemplo {i}:\nTítulo: {ex['name']}\nLegenda: {ex['desc']}\n\n"
                rag_prompt += "Nova legenda:"

                st.session_state.chat_history.append({"role": "user", "content": rag_prompt})
                st.session_state.new_post_mode = False  # volta ao modo "conversando"
            else:
                # Ajuste: só adiciona o input
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                

            # Chamada ao modelo
            response = client.chat.completions.create(
                model="gpt-4o",
                # model="gpt-3.5-turbo", 
                messages=st.session_state.chat_history,
                temperature=0.8,
                max_tokens=500
            )

            output = response.choices[0].message.content
            st.session_state.chat_history.append({"role": "assistant", "content": output})

            # Mostra resposta
            st.markdown(f'<div class="output-text">{output}</div>', unsafe_allow_html=True)



    # --- Chat History (Optional Display) ---
    if len(st.session_state.chat_history) > 1:
        st.markdown('<div class="section-header">Histórico da Conversa</div>', unsafe_allow_html=True)
        
        for i, message in enumerate(st.session_state.chat_history[1:], 1):  # Skip system message
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user"><strong>Você:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            elif message["role"] == "assistant":
                st.markdown(f'<div class="chat-message assistant"><strong>Assistente:</strong> {message["content"]}</div>', unsafe_allow_html=True)