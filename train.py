import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch

# 1. Load dataset
splits = {'train': 'data/train.csv', 'test': 'data/test.csv'}
df = pd.read_csv("hf://datasets/Roudranil/shakespearean-and-modern-english-conversational-dataset/" + splits["train"])
df.columns = [col.strip().lower() for col in df.columns]  # Clean column names

# Optional: reduce data for quick test
# df = df.sample(100).reset_index(drop=True)

# 2. Format dataset as prompt-style
print("\n📝 Formatting dataset...")
def format_example(example):
    return {
        "text": f"Translate to Shakespearean:\nModern: {example['translated_dialog']}\nShakespeare: {example['og_response']}"
    }

formatted_data = df.apply(format_example, axis=1)
dataset = Dataset.from_pandas(pd.DataFrame(formatted_data.tolist()))

# 3. Load tokenizer
print("\n🔡 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token

# 4. Tokenize
print("\n🔄 Tokenizing...")
def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize)

# 5. Load GPT-2 in 4-bit mode
print("\n📦 Loading GPT-2 in 4-bit mode...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained("gpt2", quantization_config=bnb_config, device_map="auto")

# 6. Apply LoRA
print("\n⚙️ Applying LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 7. Training arguments
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    save_strategy="epoch",
    logging_steps=5,
    fp16=True,
    report_to="none",
)

# 8. Train the model
print("\n🚀 Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# 9. Save model
print("\n💾 Saving model...")
model.save_pretrained("./output")
tokenizer.save_pretrained("./output")

print("\n✅ Training complete and model saved at ./output")
