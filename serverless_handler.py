#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RunPod Serverless Handler for Llama-3.1-8B Dating Model
"""

import runpod
import torch
import json
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import Dict, Any

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
tokenizer = None
model = None

def load_model():
    """Model'i yükle"""
    global tokenizer, model
    
    logger.info("🔄 Model yükleniyor...")
    
    try:
        # Tokenizer yükle
        tokenizer = AutoTokenizer.from_pretrained(
            "sworm/llama-8b-dating-model",
            trust_remote_code=True
        )
        
        # Base model yükle
        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # LoRA adapter'ı yükle
        model = PeftModel.from_pretrained(base_model, "sworm/llama-8b-dating-model")
        
        logger.info("✅ Model başarıyla yüklendi!")
        
    except Exception as e:
        logger.error(f"❌ Model yükleme hatası: {e}")
        raise

def generate_response(messages: list, max_tokens: int = 100, temperature: float = 0.7) -> str:
    """Yanıt üret"""
    try:
        # Chat template kullan
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        
        # Generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Sadece yanıt kısmını al
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Generation hatası: {e}")
        raise

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod serverless handler"""
    try:
        # Input'u al
        input_data = event.get("input", {})
        
        # Mesajları al
        messages = input_data.get("messages", [])
        max_tokens = input_data.get("max_tokens", 100)
        temperature = input_data.get("temperature", 0.7)
        
        # Default system prompt
        if not messages or messages[0].get("role") != "system":
            system_prompt = {
                "role": "system",
                "content": "Sen romantik bir dating asistanısın. Karşı tarafı etkilemek için samimi, ilgili ve çekici bir şekilde yanıt ver. Türkçe konuş ve doğal bir sohbet tarzı kullan."
            }
            messages = [system_prompt] + messages
        
        # Yanıt üret
        response = generate_response(messages, max_tokens, temperature)
        
        return {
            "success": True,
            "response": response,
            "input_tokens": len(tokenizer.encode(messages[-1]["content"])),
            "output_tokens": len(tokenizer.encode(response))
        }
        
    except Exception as e:
        logger.error(f"❌ Handler hatası: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# Model'i yükle
load_model()

# RunPod serverless başlat
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
