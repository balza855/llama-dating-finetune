#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RunPod Simple Llama-3.1-8B Fine-tuning - Gradient Hatası Düzeltilmiş
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
# import wandb  # W&B'yi tamamen devre dışı bırak
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SimpleConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir: str = "./llama_8b_dating_model"
    hf_dataset_name: str = "sworm/datingassistant"
    data_file: str = "training_data.jsonl"
    lora_r: int = 16  # LoRA rank azalt
    lora_alpha: int = 32  # LoRA alpha azalt
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    num_train_epochs: int = 2  # Epoch azalt
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-4  # Learning rate artır
    max_length: int = 512  # Max length daha da azalt
    fp16: bool = True
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 10

class SimpleLlamaFineTuner:
    def __init__(self, config: SimpleConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    def setup_wandb(self):
        # W&B'yi tamamen devre dışı bırak
        logger.info("⚠️  Weights & Biases devre dışı - training logs console'da görünecek")
        return
    
    def load_tokenizer(self):
        logger.info("🔄 Tokenizer yükleniyor...")
        
        # Hugging Face token kontrol et
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            logger.error("❌ HUGGING_FACE_HUB_TOKEN bulunamadı!")
            logger.info("🔑 Hugging Face token'ını ayarla:")
            logger.info("export HUGGING_FACE_HUB_TOKEN=your_token_here")
            raise ValueError("Hugging Face token gerekli!")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            token=hf_token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("✅ Tokenizer yüklendi")
    
    def load_model(self):
        logger.info("🔄 Model yükleniyor...")
        
        # Hugging Face token al
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        
        # 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Model yükle
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            token=hf_token
        )
        
        # LoRA config
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # LoRA uygula
        self.model = get_peft_model(self.model, lora_config)
        
        # Model'i training mode'a al
        self.model.train()
        
        # Tüm LoRA parametrelerini aktif et
        for name, param in self.model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
                logger.info(f"✅ {name} - requires_grad: {param.requires_grad}")
        
        # Trainable parametreleri kontrol et
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
        
        # Gradient kontrolü
        if trainable_params == 0:
            logger.error("❌ Hiç trainable parametre yok!")
            raise ValueError("Model parametreleri dondurulmuş!")
        
        logger.info("✅ Model yüklendi ve LoRA uygulandı")
    
    def download_training_data(self):
        logger.info("📥 Training data indiriliyor...")
        try:
            from huggingface_hub import hf_hub_download
            hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
            data_path = hf_hub_download(
                repo_id=self.config.hf_dataset_name,
                filename=self.config.data_file,
                repo_type="dataset",
                token=hf_token
            )
            logger.info(f"✅ Training data indirildi: {data_path}")
            return data_path
        except Exception as e:
            logger.error(f"❌ Training data indirilemedi: {e}")
            raise
    
    def load_dataset(self, data_path: str):
        logger.info("🔄 Dataset yükleniyor...")
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        logger.info(f"✅ Toplam {len(data)} örnek yüklendi")
        
        # Train/Val split (90/10)
        val_size = int(len(data) * 0.1)
        train_data = data[val_size:]
        val_data = data[:val_size]
        
        logger.info(f"📊 Training: {len(train_data)} örnek")
        logger.info(f"📊 Validation: {len(val_data)} örnek")
        
        return Dataset.from_list(train_data), Dataset.from_list(val_data)
    
    def tokenize_function(self, examples):
        """Dataset'i tokenize et - düzeltilmiş versiyon"""
        texts = []
        for messages in examples["messages"]:
            # Chat template kullan
            text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            texts.append(text)
        
        # Tokenize with proper padding
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors=None
        )
        
        # Labels ekle
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def setup_trainer(self, train_dataset, val_dataset):
        logger.info("🔄 Trainer setup ediliyor...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            
            # Optimization
            fp16=self.config.fp16,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,  # Worker sayısını 0 yap
            
            # Logging
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=2,
            
            # Evaluation
            eval_strategy="steps",
            save_strategy="steps",
            
            # Memory optimizations
            gradient_checkpointing=False,  # Gradient checkpointing'i kapat
            optim="adamw_torch",
            
            # RunPod optimizations
            remove_unused_columns=False,
            report_to=None,  # W&B'yi tamamen devre dışı bırak
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        logger.info("✅ Trainer setup edildi")
    
    def train(self):
        logger.info("🚀 Fine-tuning başlıyor...")
        try:
            # Training başlat
            self.trainer.train()
            
            # Model kaydet
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            logger.info("✅ Fine-tuning tamamlandı!")
            logger.info(f"📁 Model kaydedildi: {self.config.output_dir}")
            
        except Exception as e:
            logger.error(f"❌ Training sırasında hata: {str(e)}")
            raise
    
    def run(self):
        try:
            # Setup
            self.setup_wandb()
            self.load_tokenizer()
            self.load_model()
            
            # Training data indir
            data_path = self.download_training_data()
            
            # Dataset yükle
            train_dataset, val_dataset = self.load_dataset(data_path)
            
            # Tokenize datasets
            train_dataset = train_dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=train_dataset.column_names
            )
            val_dataset = val_dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=val_dataset.column_names
            )
            
            # Trainer setup
            self.setup_trainer(train_dataset, val_dataset)
            
            # Training
            self.train()
            
        except Exception as e:
            logger.error(f"❌ Hata: {str(e)}")
            raise
        finally:
            # W&B'yi tamamen devre dışı bıraktık
            pass

def main():
    config = SimpleConfig()
    fine_tuner = SimpleLlamaFineTuner(config)
    fine_tuner.run()

if __name__ == "__main__":
    main()