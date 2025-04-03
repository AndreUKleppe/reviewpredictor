{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b9f777ff-486a-4fd8-835c-e4517b8e1c2e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# streamlit_app.py\n",
    "\n",
    "import streamlit as st\n",
    "import openai\n",
    "import joblib\n",
    "import numpy as np\n",
    "\n",
    "# 🌐 Set your OpenAI key (use secrets in Streamlit Cloud)\n",
    "openai.api_key = st.secrets[\"OPENAI_API_KEY\"]\n",
    "\n",
    "# Load trained model\n",
    "model = joblib.load(\"star_rating_model_balanced.pkl\")\n",
    "\n",
    "# Embedding function\n",
    "def get_embedding(text, model_name=\"text-embedding-3-small\"):\n",
    "    response = openai.embeddings.create(input=text, model=model_name)\n",
    "    return response.data[0].embedding\n",
    "\n",
    "# UI\n",
    "st.title(\"⭐ Predicción de Calificación de Reseña\")\n",
    "st.markdown(\"Escribe una reseña en español y el modelo predecirá la calificación de 1 a 5 estrellas.\")\n",
    "\n",
    "user_input = st.text_area(\"✍️ Reseña del producto\", height=150)\n",
    "\n",
    "if st.button(\"🔮 Predecir calificación\"):\n",
    "    if user_input.strip():\n",
    "        with st.spinner(\"Procesando...\"):\n",
    "            embedding = get_embedding(user_input)\n",
    "            prediction = model.predict([embedding])[0]\n",
    "            st.success(f\"🌟 Predicción: **{int(prediction)} estrellas**\")\n",
    "    else:\n",
    "        st.warning(\"Por favor, escribe una reseña.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
