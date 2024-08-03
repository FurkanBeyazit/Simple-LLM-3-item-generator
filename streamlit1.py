import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# TinyLlama model ve tokenizer'ı yükleyin
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Pipeline'ı oluşturun
pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=150,
    pad_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.2
)

# Streamlit Uygulaması
st.title("Top 3 Brand")
st.write("Top 3 Brand of the Product")

# Kullanıcıdan mutfak türü seçimi al
cuisine_options = ["Watch", "Art", "Sport Car","Fashion bag",]
selected_cuisine = st.selectbox("Select a cuisine type:", cuisine_options)

# Kullanıcıdan "Get Recommendations" butonuna basmasını bekle
if st.button("Check "):
    # Kullanıcıdan prompt oluştur
    prompt = f"Most expensive {selected_cuisine} brands top 3  "

    # Modelden yanıt al
    response = pipe(prompt)

    # Sonuçları yazdır
    st.write("Top-3:")
    st.write(response[0]['generated_text'])
