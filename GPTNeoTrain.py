import time
import os
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import warnings
warnings.filterwarnings("ignore")

import PyPDF2

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()  # Extract text from each page
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

# Function to filter and extract text only from PDF files in a directory
def extract_text_from_pdfs_in_directory(directory_path):
    all_texts = []
    # Loop through the directory to find PDF files
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):  # Only process files with .pdf extension
            pdf_path = os.path.join(directory_path, filename)
            # print(f"Processing PDF: {filename}")
            text = extract_text_from_pdf(pdf_path)
            all_texts.append(text)
    return all_texts

# Example usage to extract text from multiple PDFs in a directory
pdf_directory_path = "/root/Capstone Chatbots/mis_mod/data/Docs"

# Extract text from each PDF
all_texts = extract_text_from_pdfs_in_directory(pdf_directory_path)

# Optionally, you can print or store the extracted text
for i, text in enumerate(all_texts):
     print(f"Extracted text from PDF {i+1}:")
     print(text[:500])  # Print first 500 characters of each PDF for a quick look

from datasets import Dataset
import re

# Split text based on paragraph boundaries (two newlines or more)
def split_text_on_paragraph_boundaries(text):
    # Use regex to split on paragraphs (two or more newlines)
    paragraphs = re.split(r'\n{2,}', text)
    return paragraphs

# Apply the function to all extracted texts
split_texts = [split_text_on_paragraph_boundaries(text) for text in all_texts]

# Flatten the list into a single sequence of paragraphs
flattened_paragraphs = [paragraph.strip() for sublist in split_texts for paragraph in sublist if paragraph.strip()]

# Create training data with paragraphs
training_data = [{"text": paragraph} for paragraph in flattened_paragraphs]

# Check the first few paragraphs for review
# print(training_data[:3])

# Convert the structured paragraphs into a Hugging Face Dataset
dataset = Dataset.from_dict({"text": [data["text"] for data in training_data]})

from transformers import GPT2Tokenizer

# Load the GPT-Neo tokenizer (GPT-Neo uses GPT-2 tokenizer)
tokenizer = GPT2Tokenizer.from_pretrained("/root/Capstone Chatbots/models/gpt-neo")

# Add padding token if necessary (this ensures compatibility with padding requirements)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenization function
def tokenize_function(examples):
    # Ensure padding to max length (512 in this case), truncating longer sequences
    return tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=512
    )

# Apply tokenization across the entire dataset, in batched mode
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Function to add labels (same as input_ids for causal language modeling)
def add_labels(examples):
    # Copy 'input_ids' to 'labels' for causal language modeling tasks
    examples['labels'] = examples['input_ids'].copy()
    return examples

# Apply the function to the dataset to add 'labels'
tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the GPT-Neo model and move it to GPU if available
model = GPTNeoForCausalLM.from_pretrained("/root/Capstone Chatbots/models/gpt-neo").to(device)

# Resize embeddings to accommodate any new tokens added to the tokenizer
model.resize_token_embeddings(len(tokenizer))

from transformers import GPTNeoForCausalLM, Trainer, TrainingArguments
# Set training arguments with `remove_unused_columns=False`
training_args = TrainingArguments(
    output_dir="/root/Capstone Chatbots/models/GPTNeoTrain/Output",  # Directory to save the model
    evaluation_strategy="no",  # No evaluation as you want to use the full dataset
    #max_steps=15,  # Short training run for testing; adjust as necessary
    learning_rate=5e-5,
    per_device_train_batch_size=2,  # Batch size per device (GPU/CPU)
    per_device_eval_batch_size=2,
    num_train_epochs=2,  # If max_steps is given, num_train_epochs will be ignored
    weight_decay=0.01,
    logging_dir="/root/Capstone Chatbots/models/GPTNeoTrain/Logs",  # Directory for logs
    logging_steps=1000,
    save_steps=1000,  # Save model every 1000 steps
    remove_unused_columns=False  # This ensures the 'text' column isn't dropped
)

# Initialize the Trainer
trainer = Trainer(
    model=model,  # The model to be trained
    args=training_args,  # Training arguments
    train_dataset=tokenized_dataset,  # The training dataset
)

# Start fine-tuning the model
trainer.train()
