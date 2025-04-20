# 🧠 Shakespearean to Modern English Translator

This project is a Natural Language Processing (NLP) application that translates text from Shakespearean English into Modern English using a transformer-based language model. It features a clean, themed Streamlit-based UI and integrates a Hugging Face pre-trained model for accurate and stylistically fluent translation.

---
![WhatsApp Image 2025-04-20 at 23 10 21_48ad7803](https://github.com/user-attachments/assets/eb8a95e4-9aec-45e5-8489-189e4945342b)


## ✨ Features

- 🔄 Translates complex Shakespearean English to fluent Modern English
- 🤖 Uses Hugging Face model: `aadia1234/shakespeare-to-modern`
- 🖼️ Themed UI with parchment-style design and Shakespeare quotes
- 💻 Built entirely in Python with Streamlit for easy frontend deployment
- ⚙️ Optional GPT-2 based custom model training using LoRA and 4-bit quantization

---

## 📁 Project Structure

```bash
.
├── streamlitui.py                  # Streamlit UI code
├── background.jpg          # Parchment background image
├── logo.png                # Logo for the UI
├── output/                 # Folder for saving trained models (optional)
├── train.py          # GPT-2 model training with LoRA + 4-bit config
       # Python dependencies
└── README.md               # Project overview
```

## 🎯 Objective
Make Shakespearean English accessible and enjoyable for modern audiences

Demonstrate the power of transfer learning and stylistic translation in NLP

Provide a smooth and engaging user experience via Streamlit

Experiment with efficient model training using LoRA and 4-bit quantization (optional)

## 🧪 Example Usage
Input (Shakespearean):

“Thou art as wise as thou art beautiful.”

Output (Modern):

“You are as wise as you are beautiful.”

## 📚 Dataset
Name: Shakespearean and Modern English Conversational Dataset

Link: View on Hugging Face https://huggingface.co/datasets/Roudranil/shakespearean-and-modern-english-conversational-dataset

Description: A parallel corpus of over 2,000 Shakespearean and Modern English sentence pairs. It is ideal for training and evaluating style transfer models in NLP. Each entry includes a translated_dialog (Modern English) and og_response (Shakespearean English).

## 📖 Bibliography
Jhamtani, H. et al. (2017) – Shakespearizing Modern Language Using Copy-Enriched Sequence-to-Sequence Models

Sancheti, A. et al. (2020) – Reinforced Rewards Framework for Text Style Transfer

MIT Press (2021) – Deep Learning for Text Style Transfer: A Survey

Roudranil (2022) – Shakespearean and Modern English Conversational Dataset

Wolf, T. et al. (2020) – Transformers: State-of-the-Art NLP

Hu, E. J. et al. (2021) – LoRA: Low-Rank Adaptation of LLMs

Streamlit Docs (2023) – Build Fast Frontends for ML

## 📌 Acknowledgments
Hugging Face for their pre-trained models and dataset tools

Streamlit for enabling beautiful UIs without frontend complexity

Our institution and mentor for guidance and resources

