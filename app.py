import streamlit as st
from openai import OpenAI
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


## TO DO:
## - melhor rag_prompt. Algo como: seja criativa, fale de forma espontânea e conecte com o público feminino empreendedor 
## - melhorar system content
## - algumentar o numero de posts parecidos (k=5/10 ?)
## - Fazer deploy do app na web e add login
##     - https://chatgpt.com/c/686fa4e8-9d40-8012-9417-69f42088e72d
## - Permitir que o app mantenha o contexto da conversa (e.g. user pede para mudar uma frase)
##     - Opção para começar um novo post/continuar nesse
## - Melhorar a interface do app
## - testar aqui e excluir copy



# --- Setup ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Load index + metadata
index = faiss.read_index("data/posts_index.faiss")
with open("data/posts_metadata.json", "r", encoding="utf-8") as f:
    posts_data = json.load(f)

# --- App UI ---
st.title("📲 Criador de Posts para o Instagram")
user_input = st.text_input("Digite o tema do post")

if st.button("Gerar legenda"):
    if user_input.strip() == "":
        st.warning("Por favor, digite um tema.")
    else:
        # Embed the query
        query_embedding = model.encode([user_input])[0]
        D, I = index.search(np.array([query_embedding]).astype("float32"), k=3)
        examples = [posts_data[i] for i in I[0]]

        # Build RAG prompt
        rag_prompt = "Baseado nos exemplos abaixo, crie uma nova legenda sobre: " + user_input + "\n\n"
        for i, ex in enumerate(examples, 1):
            rag_prompt += f"Exemplo {i}:\nTítulo: {ex['name']}\nLegenda: {ex['desc']}\n\n"
        rag_prompt += "Nova legenda:"

        # LLM call
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Você é uma criadora de conteúdo dona de um estúdio de design, focada em identidade visual e postagens para Instagram."},
                {"role": "user", "content": rag_prompt}
            ],
            temperature=0.8,
            max_tokens=300
        )

        output = response.choices[0].message.content
        st.success("Legenda gerada:")
        st.markdown(output)
