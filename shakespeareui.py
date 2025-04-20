import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import base64

# Load model and tokenizer
model_name = "aadia1234/shakespeare-to-modern"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Function to encode local image to base64
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Load images
bg_base64 = get_base64("background.jpg")  # Parchment background
logo_base64 = get_base64("logo.png")      # Logo

# Inject CSS for styling
st.markdown(f"""
    <style>
    * {{
        font-family: Georgia, serif !important;
    }}
    .stApp {{
        background-image: url("data:image/jpeg;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #000;
    }}
    /* Styling the header area (including the top ribbon) */
    header {{
        background-color: rgba(194, 178, 128, 0.75) !important;  /* Parchment color */
        border-bottom: 2px solid #C2B280 !important;  /* Optional: add a bottom border */
    }}
    .header {{
        display: flex;
        align-items: center;
        gap: 10px;
        background-color: rgba(194, 178, 128, 0.75); /* Parchment shade */
        padding: 0.5rem 1rem;
        border-radius: 12px;
        margin-top: 10px;
    }}
    .header img {{
        height: 50px;
        border-radius: 8px;
    }}
    .quote {{
        font-style: italic;
        font-size: 1.3rem;
        text-align: center;
        color: #3e3e3e;
        margin: 1.5rem 0 0.5rem 0;
    }}
    textarea, .stTextArea textarea {{
        background-color: rgba(255, 248, 220, 0.75) !important;
        border-radius: 10px !important;
        color: #000 !important;
    }}
    .stButton > button {{
        background-color: rgba(255, 248, 220, 0.7) !important;
        color: #000 !important;
        border: 2px solid #d1c7a1 !important;
        border-radius: 8px !important;
    }}
    .stButton > button:hover {{
        background-color: rgba(255, 248, 220, 1) !important;
    }}
    .translated-box {{
        margin-top: 10px;
        background-color: rgba(255, 248, 220, 0.6);
        padding: 15px;
        border-radius: 10px;
        font-size: 1.1rem;
    }}
    </style>
""", unsafe_allow_html=True)

# 🖼️ Header with logo and title
st.markdown(f"""
    <div class="header">
        <img src="data:image/png;base64,{logo_base64}" alt="logo">
        <h2>Shakespeare Translator</h2>
    </div>
""", unsafe_allow_html=True)

# ✒️ Shakespeare Quote (separate from containers)
st.markdown('<p class="quote">"All that glisters is not gold." – William Shakespeare</p>', unsafe_allow_html=True)

# 🎭 Translator section
st.markdown("### Convert Shakespearean English to Modern English:")

user_input = st.text_area("Enter Shakespearean text", height=150)

if st.button("Translate"):
    if not user_input.strip():
        st.warning("Pray, enter some text!")
    else:
        with st.spinner("Summoning the Bard’s essence..."):
            input_text = f"translate Shakespeare to Modern English: {user_input}"
            inputs = tokenizer(input_text, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=100)
            modern_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.success("Behold! Thy translation is ready:")
        st.markdown(f'<div class="translated-box">{modern_text}</div>', unsafe_allow_html=True)
