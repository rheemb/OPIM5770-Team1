import os
import re
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments
import warnings
import PyPDF2
from datasets import Dataset

warnings.filterwarnings("ignore")

# Function to extract text from each page of a PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    # Append text only if it is not part of the Table of Contents or References
                    if not is_toc_or_references(page_text):
                        text += page_text + "\n"  # Add a newline to separate pages
                else:
                    print(f"Warning: No text found on page {page_num + 1} of {pdf_path}")
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

# Function to check if a text section is likely a Table of Contents or References
def is_toc_or_references(text):
    # Check for common indicators of Table of Contents and References sections
    toc_keywords = ["table of contents", "contents"]
    ref_keywords = ["references", "bibliography"]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in toc_keywords + ref_keywords)

# Function to split text by sections while retaining document structure
def split_text_by_sections(text):
    sections = re.split(r'(?i)(\n\s*INTRODUCTION|\n\s*CONCLUSION|\n\s*[A-Z]+\s*\n)', text)
    structured_text = [section.strip() for section in sections if section.strip()]
    return structured_text

# Function to extract text from all PDFs and structure it
def extract_and_structure_text_from_pdfs(directory_path):
    all_texts = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            text = extract_text_from_pdf(pdf_path)
            structured_text = split_text_by_sections(text)
            all_texts.extend(structured_text)
    return all_texts

# Example usage to extract and structure text
pdf_directory_path = "/root/Capstone Chatbots/mis_mod/data/Docs"
all_texts = extract_and_structure_text_from_pdfs(pdf_directory_path)

# Prepare training data with sections
training_data = [{"text": section} for section in all_texts]

# Convert structured data into Hugging Face Dataset format
dataset = Dataset.from_dict({"text": [data["text"] for data in training_data]})

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("/root/Capstone Chatbots/models/gpt-neo")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=1024)

# Tokenize dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Add labels for causal language modeling
def add_labels(examples):
    examples['labels'] = examples['input_ids'].copy()
    return examples

tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model = GPTNeoForCausalLM.from_pretrained("/root/Capstone Chatbots/models/gpt-neo").to(device)
model.resize_token_embeddings(len(tokenizer))

# Set up training arguments
training_args = TrainingArguments(
    output_dir="/root/Capstone Chatbots/models/GPTNeoTrain/noutput",
    evaluation_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="/root/Capstone Chatbots/models/GPTNeoTrain/Logs",
    logging_steps=1000,
    save_steps=1000,
    remove_unused_columns=False
)

# Initialize and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
