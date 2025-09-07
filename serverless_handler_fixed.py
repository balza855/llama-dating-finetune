#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RunPod Serverless Handler - CUDA Fixed
"""

import runpod
import torch
import json
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
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
        
        # Model device'ını kontrol et
        device = next(model.parameters()).device
        logger.info(f"Model device: {device}")
        
        logger.info("✅ Model başarıyla yüklendi!")
        
    except Exception as e:
        logger.error(f"❌ Model yükleme hatası: {e}")
        raise

def _truncate_to_sentences(text: str, max_sentences: int) -> str:
    """Keep only the first N sentences from text."""
    # Split by common sentence enders while keeping unicode ellipsis
    parts = re.split(r"(?<=[\.\!\?…])\s+", text.strip())
    kept: list[str] = []
    for part in parts:
        part = part.strip()
        if part:
            kept.append(part)
        if len(kept) >= max_sentences:
            break
    return " ".join(kept) if kept else text


def generate_response(messages: list, max_tokens: int = 100, temperature: float = 0.7, max_sentences: int = 2) -> str:
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
        
        # Move inputs to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
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
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                length_penalty=0.9
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Sadece yanıt kısmını al
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()

        # En fazla N cümle ile sınırla
        response = _truncate_to_sentences(response, max_sentences)
        
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
        max_tokens = input_data.get("max_tokens", 80)
        temperature = input_data.get("temperature", 0.6)
        max_sentences = int(input_data.get("max_sentences", 2))
        
        # Default system prompt
        if not messages or messages[0].get("role") != "system":
            system_prompt = {
                "role": "system",
                "content": (
                    "Sen kadın bir dating asistanısın. Cevapların doğal, samimi, diyalog kuran ve sınırlarını bilen bir gerçek kadın gibi olsun. "
                    "Maksimum iki cümle yaz; kısa, net ve kibar ol. Argo veya kaba söz kullanma; güvenli ve flörtöz bir ton tut."
                ),
            }
            messages = [system_prompt] + messages
        
        # Yanıt üret
        response = generate_response(messages, max_tokens, temperature, max_sentences)
        
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
