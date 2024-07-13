import streamlit as st
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")

# Setting the title of the Streamlit application
st.title('MY LLM 🤖')

# Defining a function to generate a response using the GPT-Neo model
def generate_response(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    st.info(response_text)

# Creating a form in the Streamlit app for user input
with st.form('my_form'):
    # Adding a text area for user input with a default prompt
    text = st.text_area('Enter text:', '')
    # Adding a submit button for the form
    submitted = st.form_submit_button('Submit')
    # If the form is submitted, generate a response
    if submitted:
        generate_response(text)
