import streamlit as st
import openai
import joblib

# Use your OpenAI key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load trained model
model = joblib.load("star_rating_model_balanced.pkl")

# Generate embedding
def get_embedding(text, model_name="text-embedding-3-small"):
    response = openai.embeddings.create(input=text, model=model_name)
    return response.data[0].embedding

# UI
st.title("⭐ Predicción de Calificación de Reseña")
st.write("Escribe una reseña en español para predecir una calificación de 1 a 5 estrellas.")

review = st.text_area("✍️ Reseña", height=150)

if st.button("Predecir"):
    if review.strip():
        with st.spinner("Analizando..."):
            embedding = get_embedding(review)
            prediction = model.predict([embedding])[0]
            st.success(f"🌟 Predicción: **{int(prediction)} estrellas**")
    else:
        st.warning("Por favor escribe una reseña.")