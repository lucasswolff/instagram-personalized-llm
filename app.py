import streamlit as st
from openai import OpenAI
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


## TO DO:
## - melhor rag_prompt. Algo como: seja criativa, fale de forma espontânea e conecte com o público feminino empreendedor 
## - melhorar system content
## - Fazer deploy do app na web e add login
##     - https://chatgpt.com/c/686fa4e8-9d40-8012-9417-69f42088e72d
## - Melhorar a interface do app



# --- Setup ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

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
    st.session_state.new_post_mode = False

# --- Title & Buttons ---
st.title("📲 Criador de Posts para o Instagram")

if st.button("🆕 Novo Post"):
    st.session_state.chat_history = [
        {"role": "system", "content": about_the_user}
    ]
    st.session_state.new_post_mode = True

# --- User Input ---
user_input = st.text_input("Digite o tema do post ou um ajuste no anterior")

if st.button("✍️ Enviar"):
    if user_input.strip() == "":
        st.warning("Por favor, digite algo.")
        
    else:
        if st.session_state.new_post_mode:
            # Novo post: roda RAG
            query_embedding = model.encode([user_input])[0]
            D, I = index.search(np.array([query_embedding]).astype("float32"), k=5)
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
            messages=st.session_state.chat_history,
            temperature=0.8,
            max_tokens=300
        )

        output = response.choices[0].message.content
        st.session_state.chat_history.append({"role": "assistant", "content": output})

        # Mostra resposta
        st.markdown(output)