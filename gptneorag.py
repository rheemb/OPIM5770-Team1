import os
import re
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch

# Step 1: Load FAISS index and chunks
faiss_index_path = "/root/Capstone Chatbots/mis_mod/gptneoragreq/faiss_index.bin"
chunks_path = "/root/Capstone Chatbots/mis_mod/gptneoragreq/chunks.pkl"

index = faiss.read_index(faiss_index_path)
with open(chunks_path, "rb") as f:
    all_chunks = pickle.load(f)

print(f"FAISS index loaded with {index.ntotal} chunks.")

# Step 2: Load the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 3: Load the GPT-Neo model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = GPT2Tokenizer.from_pretrained("/root/Capstone Chatbots/models/gpt-neo")
model = GPTNeoForCausalLM.from_pretrained(
    "/root/Capstone Chatbots/models/GPTNeoTrain/Output/checkpoint-32"
).to(device)

# Add padding token if necessary
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

# Step 4: Retrieve relevant chunks with FAISS and reranking
def retrieve_relevant_chunks(query, k=3, keywords=['cryptocurrency']):
    """Retrieve top-k relevant chunks and rerank using keywords."""
    query_vector = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vector, min(k + 7, index.ntotal))

    relevant_chunks = []
    for idx in indices[0]:
        if idx < len(all_chunks):
            chunk = all_chunks[idx]
            if any(keyword.lower() in chunk.lower() for keyword in keywords):
                relevant_chunks.append((chunk, distances[0][indices[0].tolist().index(idx)]))

    if not relevant_chunks:
        relevant_chunks = [(all_chunks[idx], distances[0][i]) 
                           for i, idx in enumerate(indices[0]) if idx < len(all_chunks)]

    relevant_chunks = sorted(relevant_chunks, key=lambda x: x[1])
    return [chunk for chunk, _ in relevant_chunks[:k]]

# Step 5: Clean context to avoid repetition
def clean_context(context):
    return " ".join(context.split()).strip()

# Step 6: Semantic filtering to clean generated response
def clean_generated_response(response, query):
    """Remove extra spaces, repeated or contradictory sentences."""
    response = " ".join(response.split()).strip()
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', response)]

    valid_sentences = []
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)

    for sentence in sentences:
        sentence_embedding = embedding_model.encode(sentence, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(query_embedding, sentence_embedding).item()

        if similarity > 0.7:  # Keep relevant sentences
            valid_sentences.append(sentence)

    match = re.search(r"(.*?)([.!?])\s*$", " ".join(valid_sentences))
    if match:
        cleaned_response = match.group(1) + match.group(2)
    else:
        cleaned_response = " ".join(valid_sentences)

    return cleaned_response

# Step 7: Generate response using GPT-Neo
def generate_response_with_rag(query):
    """Generate a response using the fine-tuned GPT-Neo model."""
    relevant_chunks = retrieve_relevant_chunks(query, k=1)
    cleaned_context = clean_context(" ".join(relevant_chunks))

    inputs = tokenizer(
        f"Context: {cleaned_context}\n\nQuestion: {query}\nAnswer:",
        return_tensors="pt", truncation=True, padding=True, max_length=768
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=120,
        no_repeat_ngram_size=3,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return clean_generated_response(response, query)

# Step 8: Interactive query loop
while True:
    try:
        query = input("Enter your question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        response = generate_response_with_rag(query)
        print(f"Response: {response}")
    except Exception as e:
        print(f"An error occurred: {e}")
