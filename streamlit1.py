import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=150,
    pad_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.2
)

st.title("Top 3 Brand")
st.write("Top 3 Brand of the Product")


options = ["Watch", "Art", "Sport Car","Fashion bag",]
selected = st.selectbox("Select a type:",options)


if st.button("Check "):
   
    prompt = f"Most expensive {selected} brands top 3  "

    
    response = pipe(prompt)

    
    st.write("Top-3:")
    st.write(response[0]['generated_text'])
