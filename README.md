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

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20375290-blue?style=flat-square)](https://doi.org/10.5281/zenodo.20375290)

[Kaggle Notebook](https://www.kaggle.com/code/phonglucian/no-code-evonet-studio-faster-llm-fine-tuning)

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

### 5. 📉 LLM-as-a-Judge & LLM Arena (DPO Synthesis)
Ditch traditional Perplexity metrics. Use our built-in **LLM Arena** to blindly test two models side-by-side. Your votes are automatically saved as a high-quality DPO dataset (`outputs/arena_dpo_dataset.jsonl`). You can also use Gemini, GPT-4, or Claude to automatically grade your model's responses.

### 6. ✨ Auto-Prompt Optimizer (DSPy Style)
Stop struggling with prompt engineering! Enter a basic idea (e.g., "A customer service bot") and our built-in Auto-Prompt Optimizer will use advanced Meta-Prompting to generate a highly structured, professional System Prompt (Role, Objective, Rules, Format) instantly.

### 7. 🧹 Data Cleaning & Preprocessing Studio
Garbage in, garbage out! Use the built-in Data Cleaner to instantly analyze your raw `.jsonl` datasets. It automatically detects and removes duplicate records, identifies missing keys, strips out overly short responses, and estimates total tokens to ensure your dataset is perfectly healthy before wasting VRAM on training.

### 8. 📊 Automated Academic Benchmark Suite
Stop guessing if your model is smart! We integrated `lm-eval` (HuggingFace Evaluation Harness) directly into EvoNet-Studio. Select your fine-tuned model, choose tests like **MMLU, GSM8k, or HellaSwag**, and run a rigorous academic benchmark locally to get objective scores.

### 9. 🔄 Data Flywheel (Closed-Loop Telemetry)
The Holy Grail of AI: A closed feedback loop! When deploying the model via our Native API, all chats are logged. External apps can send Thumbs Up/Down feedback to the `/v1/feedback` endpoint. Inside the **Telemetry Studio**, you can view bad responses, correct them manually, and instantly push them to your DPO Dataset to make your next model version smarter!

### 10. ⚡ 1-Click Deployment (vLLM & Fallback)
Once trained, deploy your model as a production-grade API server (`http://localhost:8000/v1/chat/completions`) with a single click.
- **vLLM Engine:** For those with powerful GPUs, deploy with PagedAttention for maximum throughput.
- **FastAPI Engine:** Native CPU/GPU fallback using standard Transformers (supports Production Telemetry).

### 11. 🧩 Multi-LoRA MoE Router (LoRAX) & LoRA Manager
Never suffer from Catastrophic Forgetting again! Instead of overwriting your base model, train isolated LoRA experts (e.g., Code, Math, Law). In the API Deployment tab, load your Base Model and pass a JSON dictionary of your LoRA adapters. The Native Server acts as a **Mixture-of-Experts (MoE) Router**, dynamically analyzing incoming prompts and hot-swapping the correct LoRA adapter in 0.01s before generating a response. Serve 50 experts from a single GPU without OOM!
You can easily manage all your trained adapters using the dedicated **LoRA Manager** Tab.

### 12. 🔄 Dynamic HuggingFace Hub Fetcher
EvoNet-Studio is future-proof. Click the "Fetch Latest Models" button to instantly query the HuggingFace API and pull down the top trending models (e.g., latest Llama, Qwen, Gemma variants from Unsloth) straight into your training dropdowns.

### 13. 📦 Model Merging, GGUF Export & Ollama Push
Merge your trained LoRA adapters directly into the Base Model and instantly push the standalone model to the **HuggingFace Hub** to share with the world. You can also export the model to **GGUF** format and push it directly to **Ollama** with a single click to run on local CPU/Edge devices!

### 14. 📚 Continued Pre-Training (CPT)
Train your model on massive amounts of raw text (e.g., medical textbooks, legal documents) without formatting them into Q&A. The new CPT mode allows you to inject deep domain knowledge natively into the base model before you fine-tune it for conversation.

### 15. ☁️ Cloud Workspace Sync (Backup & Restore)
Never lose your precious datasets or fine-tuned weights! The Cloud Sync tab allows you to compress your entire `outputs/` folder and securely back it up to a Private HuggingFace Dataset repository. Switch computers? Just click "Restore Workspace" to pull everything down and pick up exactly where you left off.

---

## 💼 Enterprise Features (Paid Version Only)

For corporate environments and heavy workloads, the **EvoNet-Studio Enterprise Edition** includes:
1. **API Key Gateway & Rate Limiting:** Prevent D-DoS and control token usage with internal API keys.
2. **SSO & OAuth2 Integration:** 1-Click login via Google/Microsoft Azure AD for entire teams.
3. **Multi-Node Distributed Training:** Seamlessly enable DeepSpeed ZeRO-3 to distribute heavy training across multiple physical GPU nodes.
4. **GPU Cost Dashboard:** Real-time tracking of GPU uptime and estimated compute costs (in USD).
5. **Canary & A/B Deployments:** Safely route a percentage of API traffic to a Challenger Model to test in production.
6. **Hardware Lock & Auto PII Masking:** Compliance and security for sensitive data.

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

Bạn có thể khởi động EvoNet-Studio Pro bằng nhiều cách tùy theo nhu cầu bảo mật:

**Chạy Cục bộ (Local - Khuyên dùng cho cá nhân):**
```bash
python3 app.py
```
*(Chỉ truy cập được từ máy tính của bạn thông qua `http://localhost:7860`)*

**Chạy trên Cloud/Kaggle (Cần Public Link):**
```bash
python3 app.py --share
```
*(Tạo đường link `.gradio.live`. Lưu ý: Ai có link cũng có thể vào được!)*

**Chạy trên Cloud với Mật khẩu Bảo mật (Khuyên dùng):**
```bash
python3 app.py --share --auth admin:matkhau123
```
*(Tạo link Public nhưng bị khóa bởi màn hình Đăng nhập. Thay đổi `admin:matkhau123` thành tài khoản và mật khẩu của bạn).*

---

## 🐳 Docker Deployment (Enterprise / Cloud)

For enterprise users looking to deploy EvoNet-Studio on cloud instances (AWS EC2, GCP, Azure) or local servers, we provide a production-ready Docker setup.

```bash
# 1. Clone the repository
git clone https://github.com/phonghhd/EvoNet-Studio.git
cd EvoNet-Studio

# 2. Deploy with Docker Compose (requires NVIDIA Container Toolkit)
docker-compose up -d --build
```

**What this does:**
- Automatically builds an isolated container using `pytorch:2.2.1-cuda12.1` as the base image.
- Passes GPU passthrough to the container (`capabilities: [gpu]`).
- Exposes port `7860` for the Studio UI and port `8000` for the API Deployment server.
- Mounts local `/outputs` and `/dataset` directories to prevent data loss when the container restarts.

---

## 🖥️ Hybrid Architecture (CPU / GPU)

EvoNet-Studio intelligently detects your hardware capabilities:
- **GPU Present:** The system will dynamically load `Unsloth` kernels to perform 4-bit quantization fine-tuning, resulting in 2x faster training and 70% less VRAM usage.
- **CPU Only:** The system gracefully falls back to native HuggingFace `transformers` and `trl`, allowing you to develop, test, and run the entire pipeline even without a dedicated graphics card.

## 📚 Official Documentation

Tài liệu hướng dẫn chi tiết dành cho các phiên bản của EvoNet-Studio:
- [📖 Hướng Dẫn Bản Open-Source Miễn Phí (Github)](docs/Open_Source_Guide.md)
- [🏢 Tài Liệu Vận Hành Bản Enterprise (Doanh Nghiệp)](docs/Enterprise_Manual.md)

## 🚀 Future Roadmap

EvoNet-Studio is constantly evolving. In the upcoming Q3/Q4 releases, we are planning to introduce:
- **Apple Silicon (MPS) Optimization:** Enhanced hardware acceleration for MacOS users without NVIDIA cards.
- **Llama.cpp WebUI Integration:** Connect directly to local model management interfaces for smoother execution.
- **Multi-Tenant Architecture (Enterprise):** True SaaS-ready architecture allowing a single installation to serve multiple isolated departments.
- **Advanced RAG Evaluation Suites (Enterprise):** Built-in RAG measurement tools (RAGAS, TruLens) directly on the dashboard to score AI factual accuracy.

## 🤝 Contributing

Contributions are welcome! If you'd like to add support for new Alignment algorithms, better UI elements, or new Deployment engines, feel free to submit a Pull Request.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 💖 Support & Donate (Ủng hộ dự án)

Nếu bạn thấy **EvoNet-Studio Pro** hữu ích cho công việc, học tập của bạn, hãy cân nhắc ủng hộ (donate) để tiếp thêm động lực cho chúng tôi duy trì và phát triển thêm nhiều siêu tính năng mới trong tương lai!

Mọi sự đóng góp dù nhỏ nhất đều là nguồn động viên vô giá đối với những người làm mã nguồn mở.

- **Ngân hàng:** MB BANK
- **Số tài khoản:** 0948622661
- **Tên người thụ hưởng:** HUYNH DUONG PHONG
- **Nội dung:** Donate EvoNet Studio

<img src="QR.jpg" alt="QR Code Donate" width="200"/>

Cảm ơn bạn rất nhiều vì đã đồng hành cùng sự phát triển của Trí tuệ Nhân tạo Việt Nam! 🇻🇳
