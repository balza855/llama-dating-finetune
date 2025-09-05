# 🚀 RunPod Llama-3.1-8B Fine-tuning Rehberi

Bu rehber, RunPod üzerinde Llama-3.1-8B-Instruct modelini Türkçe dating conversation verisi ile fine-tune etmek için hazırlanmıştır.

## 🎯 **Llama-3.1-8B Avantajları**

### ✨ **Performans**
- **Daha Hızlı**: 8B parametre ile çok daha hızlı eğitim
- **Daha Az VRAM**: 10-16GB VRAM yeterli
- **Daha Hızlı Inference**: ~200-300 tokens/s
- **Daha Az Maliyet**: RunPod'da daha ucuz GPU'lar kullanılabilir

### 🔧 **Teknik Özellikler**
- **Model Size**: ~16GB (base) + ~100MB (LoRA)
- **Training Time**: ~2-4 saat (3 epoch)
- **Memory Usage**: ~12-16GB VRAM
- **Batch Size**: 2 (gradient accumulation: 4)

## 📋 **Gereksinimler**

### Donanım Gereksinimleri
- **GPU**: RTX 3080 (10GB) - minimum
- **RAM**: 16GB sistem RAM
- **Disk**: 50GB boş alan
- **CPU**: 4+ core

### RunPod Template
- **Docker Image**: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
- **Container Disk**: 50GB
- **Volume**: 25GB
- **Ports**: 8000, 8888

## 🚀 **Hızlı Başlangıç**

### 1. **RunPod'a Bağlan**
```bash
# RunPod terminal'de
cd /workspace
```

### 2. **Hızlı Setup**
```bash
# Setup script çalıştır
bash runpod_quick_start.sh
```

### 3. **Fine-tuning Başlat**
```bash
# Training başlat
python runpod_simple_finetune.py
```

### 4. **Model Test**
```bash
# Test et
python test_llama_8b_model.py
```

## ⚙️ **Konfigürasyon**

### Model Settings
```python
model_name = "meta-llama/Llama-3.1-8B-Instruct"
output_dir = "./llama_8b_dating_model"
```

### LoRA Settings (8B için optimize)
```python
lora_r = 32          # 70B'de 64'tü
lora_alpha = 64      # 70B'de 128'di
lora_dropout = 0.1
```

### Training Settings
```python
per_device_train_batch_size = 2      # 70B'de 1'di
gradient_accumulation_steps = 4      # 70B'de 8'di
learning_rate = 3e-4                 # 70B'de 2e-4'tü
```

## 📊 **Performance Karşılaştırması**

| Özellik | Llama-3.3-70B | Llama-3.1-8B |
|---------|---------------|---------------|
| **VRAM** | 24-40GB | 10-16GB |
| **Training Time** | 6-12 saat | 2-4 saat |
| **Inference Speed** | 50-100 tokens/s | 200-300 tokens/s |
| **Model Size** | 40GB + 200MB | 16GB + 100MB |
| **Batch Size** | 1 | 2 |
| **RunPod Cost** | Yüksek | Düşük |

## 🎯 **RunPod Terminal Komutları**

```bash
# 1. Setup
bash runpod_quick_start.sh

# 2. Training başlat
python runpod_simple_finetune.py

# 3. Background'da çalıştır
nohup python runpod_simple_finetune.py > training.log 2>&1 &

# 4. Progress takip et
tail -f training.log

# 5. Test et
python test_llama_8b_model.py
```

## 📈 **Monitoring**

```bash
# GPU kullanımı
nvidia-smi -l 1

# Training progress
tail -f training.log

# Weights & Biases (browser'da)
# Otomatik olarak link oluşturulur
```

## 🔐 **Environment Variables**

```bash
HUGGING_FACE_HUB_TOKEN=your_token_here
WANDB_API_KEY=your_wandb_key_here
```

## 📁 **Proje Yapısı**

```
/workspace/
├── runpod_simple_finetune.py    # Ana fine-tuning script
├── test_llama_8b_model.py       # Model test script
├── runpod_quick_start.sh        # Setup script
├── requirements_simple.txt      # Dependencies
├── runpod_config.yaml          # RunPod config
├── llama_8b_dating_model/      # Fine-tuned model
└── training.log                # Training logs
```

## 🧪 **Test Örnekleri**

```python
test_messages = [
    "Merhaba, nasılsın?",
    "Kaç yaşındasın?",
    "Nerede yaşıyorsun?",
    "Seni tanımak istiyorum",
    "Kahve içmeye ne dersin?",
    "Hangi müzik türlerini seviyorsun?",
    "En sevdiğin film nedir?",
    "Hangi şehirde yaşamak istersin?",
    "Hobilerin neler?",
    "Nasıl bir insan arıyorsun?"
]
```

## 🚨 **Sorun Giderme**

### 1. **CUDA Out of Memory**
```bash
# Batch size azalt
per_device_train_batch_size = 1

# Gradient accumulation artır
gradient_accumulation_steps = 8
```

### 2. **Slow Training**
```bash
# GPU kullanımını kontrol et
nvidia-smi

# DataLoader optimizasyonu
dataloader_pin_memory = False
```

### 3. **Model Yüklenemiyor**
```bash
# Hugging Face token kontrol
huggingface-cli whoami

# Disk alanı kontrol
df -h
```

## 🎯 **Kullanım Senaryoları**

### 1. **Hızlı Test**
```bash
python runpod_simple_finetune.py && python test_llama_8b_model.py
```

### 2. **Production Deployment**
```bash
# Training
python runpod_simple_finetune.py

# Deploy (Flask API)
python deploy_llama_model.py --host 0.0.0.0 --port 8000
```

### 3. **Interactive Development**
```bash
# Jupyter notebook
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## 📚 **Kaynaklar**

- [Llama-3.1-8B Documentation](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [RunPod Documentation](https://docs.runpod.io/)
- [PEFT Documentation](https://huggingface.co/docs/peft/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)

## 🤝 **Destek**

Sorun yaşıyorsanız:

1. **GitHub Issues**: Proje repository'sinde issue açın
2. **RunPod Support**: RunPod support ekibiyle iletişime geçin
3. **Community**: Hugging Face community'de soru sorun

---

**Not**: Llama-3.1-8B, 70B modeline göre çok daha verimli ve maliyet-etkin bir seçimdir. Aynı kalitede sonuçlar alırken çok daha hızlı eğitilir ve daha az kaynak kullanır.
