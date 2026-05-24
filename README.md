<div align="center">
  <h1>🚀 EvoNet-Studio Pro</h1>
  <p><b>The Ultimate Open-Source AI Fine-Tuning & LLMOps Lifecycle Platform</b></p>

  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Framework-vietnamese--ai-red" alt="Framework">
  <img src="https://img.shields.io/badge/Accelerated_by-Unsloth-pink" alt="Unsloth">
</div>

---

EvoNet-Studio Pro is a cutting-edge, web-based (Gradio) interface designed to make Supervised Fine-Tuning (SFT), Reinforcement Learning (RLHF), and Model Deployment accessible, fast, and secure. Built on top of the robust [vietnamese-ai](https://github.com/phonghhd/vietnamese-ai) framework and optimized with **Unsloth**, this studio allows you to build, align, test, and deploy state-of-the-art Large Language Models (LLMs) and Vision-Language Models (VLMs) right from your local machine.

## ✨ Super Features

### 1. ⚖️ Alignment Tuning (RLHF)
Move beyond basic SFT. EvoNet-Studio supports the holy trinity of AI alignment:
- **DPO** (Direct Preference Optimization)
- **ORPO** (Odds Ratio Preference Optimization) - *Highly VRAM efficient!*
- **KTO** (Kahneman-Tversky Optimization)

### 2. 🤖 Agentic & VLM Fine-Tuning
- **Agent Tuning:** Train models to use external APIs and Tools via standard ReAct or function-calling formats (ChatML, Llama-3-Instruct).
- **Vision-Language Models (VLM):** Built-in toggle to fine-tune multi-modal models like LLaVA or Qwen-VL using `FastVisionModel`.

### 3. 🧬 Data Synthesis (Knowledge Distillation)
Stop scraping data! Use the integrated **Google Gemini API** (or OpenAI/Anthropic/Local LLMs) to dynamically generate high-quality `.jsonl` datasets (Alpaca/ChatML format) based on your custom prompt.

### 4. 📚 RAG Studio
Test your fine-tuned model's ability to extract facts from private documents:
- Upload `.pdf` or `.txt` files.
- The built-in Local Vector Database (`sentence-transformers`) chunks and embeds the document.
- Chat with your model securely offline and verify its RAG capabilities.

### 5. 📉 LLM-as-a-Judge Evaluation
Ditch traditional Perplexity metrics. Use Gemini, GPT-4, or Claude to automatically grade your model's responses on a scale of 1-10 to ensure logical accuracy and prevent hallucinations.

### 6. ⚡ 1-Click Deployment (vLLM & Fallback)
Once trained, deploy your model as a production-grade API server (`http://localhost:8000/v1/chat/completions`) with a single click.
- **vLLM Engine:** For those with powerful GPUs, deploy with PagedAttention for maximum throughput.
- **FastAPI Engine:** Native CPU/GPU fallback using standard Transformers.

### 7. 📦 Model Merging & Hub Push
Merge your trained LoRA adapters directly into the Base Model and instantly push the standalone model to the **HuggingFace Hub** to share with the world.

---

## ⚙️ Installation

We provide an automated setup script that creates an optimized virtual environment:

```bash
# 1. Clone the repository
git clone https://github.com/phonghhd/EvoNet-Studio.git
cd EvoNet-Studio

# 2. Run the setup script
bash setup.sh

# 3. Activate the environment
source venv/bin/activate

# 4. Start the Studio!
python3 app.py
```

*The Gradio interface will automatically open at `http://localhost:7860`.*

---

## 🖥️ Hybrid Architecture (CPU / GPU)

EvoNet-Studio intelligently detects your hardware capabilities:
- **GPU Present:** The system will dynamically load `Unsloth` kernels to perform 4-bit quantization fine-tuning, resulting in 2x faster training and 70% less VRAM usage.
- **CPU Only:** The system gracefully falls back to native HuggingFace `transformers` and `trl`, allowing you to develop, test, and run the entire pipeline even without a dedicated graphics card.

## 🤝 Contributing

Contributions are welcome! If you'd like to add support for new Alignment algorithms, better UI elements, or new Deployment engines, feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
