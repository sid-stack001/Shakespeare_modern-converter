from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "aadia1234/shakespeare-to-modern"  # This is the original T5 version
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Inference
input_text = "translate Shakespeare to Modern English: Fetch me that flower; the herb I showed thee once"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
