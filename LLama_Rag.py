import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv
from peft import PeftModel, LoraConfig, get_peft_model
from langchain.llms.base import LLM
from typing import Any, Optional, List, Dict
from pydantic import Field
import logging
from trl import setup_chat_format

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()

# Environment variables or configuration
BASE_MODEL_PATH = os.path.join(os.getcwd(), "Models")
FINE_TUNED_MODEL_PATH = os.path.join(os.getcwd(), "Models")

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

def load_fine_tuned_model():
    logger.info(f"Loading the fine-tuned model from {FINE_TUNED_MODEL_PATH}...")
    try:
        torch_dtype = torch.float16
        attn_implementation = "eager"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation=attn_implementation
        )

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        model, tokenizer = setup_chat_format(model, tokenizer)

        # LoRA config
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
        )
        model = get_peft_model(model, peft_config)

        # Load the fine-tuned weights
        model = PeftModel.from_pretrained(model, FINE_TUNED_MODEL_PATH)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
        
        logger.info("Fine-tuned model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading fine-tuned model: {e}")
        raise e

class LocalLLM(LLM):
    model: Any = Field(default=None)
    tokenizer: Any = Field(default=None)

    def __init__(self, model, tokenizer):
        super().__init__(model=model, tokenizer=tokenizer)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        chat_input = [{"role": "user", "content": prompt}]
        formatted_input = self.tokenizer.apply_chat_template(chat_input, tokenize=False)
        inputs = self.tokenizer(formatted_input, return_tensors="pt", padding=True, truncation=True, max_length=250).to(DEVICE)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process the response
        sentences = response.split('.')
        if len(sentences) > 5:
            response = '. '.join(sentences[:5]) + '.'
        
        return response.strip()

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"name": "LocalLLM"}

    @property
    def _llm_type(self) -> str:
        return "custom"

def chat(user_input: str) -> str:
    try:
        logger.info(f"Received input: {user_input}")
        
        # Format the input as per the chat template
        chat_input = [{"role": "user", "content": user_input}]
        formatted_input = tokenizer.apply_chat_template(chat_input, tokenize=False)
        
        # Use the LocalLLM to generate a response
        response = local_llm(formatted_input)
        
        logger.info(f"Model response: {response}")
        
        # Ensure the response ends with a full stop
        response = response.strip()
        if not response.endswith('.'):
            response += '.'
        
        return response
    except Exception as e:
        logger.error(f"Error in chat function: {str(e)}")
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    logger.info("Initializing the application...")
    model, tokenizer = load_fine_tuned_model()
    local_llm = LocalLLM(model, tokenizer)
    logger.info("Model loaded and ready for chat.")

    # Example usage
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            break
        response = chat(user_input)
        print(f"AI: {response}")

    logger.info("Chat session ended.")