# Hướng dẫn Sử dụng EvoNet-Studio (Bản Open-Core Miễn Phí)

Chào mừng bạn đến với **EvoNet-Studio**, một hệ sinh thái AI (LLMOps) đa năng dành cho việc tinh chỉnh (Fine-tuning), RAG, và triển khai các mô hình ngôn ngữ lớn (LLMs). Tài liệu này hướng dẫn chi tiết cách sử dụng phiên bản Open-Source (miễn phí) trên GitHub.

## 1. Cài đặt Môi trường
EvoNet-Studio yêu cầu Python 3.10+ và hệ điều hành Linux (khuyến nghị Ubuntu).
Môi trường ảo (virtual environment) là bắt buộc để tránh xung đột thư viện.

```bash
# 1. Clone mã nguồn
git clone https://github.com/phonghhd/EvoNet-Studio.git
cd EvoNet-Studio

# 2. Tạo và kích hoạt môi trường ảo
python3 -m venv venv
source venv/bin/activate

# 3. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

## 2. Khởi chạy Giao diện
Sau khi cài đặt xong, bạn khởi động giao diện người dùng (Gradio) bằng lệnh:
```bash
python3 app.py
```
Giao diện sẽ chạy ở địa chỉ `http://127.0.0.1:7860`. Nếu bạn chạy trên máy chủ đám mây (VPS/Colab), bạn có thể thêm tham số `--share`:
```bash
python3 app.py --share
```

## 3. Các Tính năng Chính
EvoNet-Studio chia làm nhiều Tab công cụ. Dưới đây là hướng dẫn sử dụng các công cụ quan trọng nhất.

### 3.1. 🧹 Data Cleaner (Chẩn đoán & Dọn dẹp dữ liệu)
Tính năng này giúp bạn loại bỏ dữ liệu rác trước khi huấn luyện (tiết kiệm VRAM và thời gian).
1. Chuẩn bị file `dataset.jsonl` (định dạng ChatML: `{"messages": [{"role": "user", "content": "..."}, ...]}`).
2. Nhập đường dẫn file vào hệ thống và bấm **🔍 Analyze Dataset**. Hệ thống sẽ kiểm tra cú pháp và tìm các dòng trùng lặp.
3. Nếu phát hiện trùng lặp, bấm **✨ Clean & Export Dataset** để xuất ra một file dữ liệu sạch mới.

### 3.2. 🧠 SFT Training (Tinh chỉnh có giám sát)
Tinh chỉnh LLM theo phong cách hỏi-đáp.
1. Trong Tab SFT, chọn Mô hình (Base Model), ví dụ: `unsloth/llama-3-8b-bnb-4bit`.
2. Điền đường dẫn tới file dữ liệu đã làm sạch.
3. Chỉnh tham số: Bạn có thể nhấn **✨ Auto-Suggest Hyperparameters** để hệ thống tự động điền Epochs, Batch Size, Learning Rate dựa vào kích thước file dữ liệu.
4. Nhấn **🚀 Start SFT Training**. Quá trình huấn luyện sẽ bắt đầu và log sẽ hiện ra theo thời gian thực.
Mô hình sau khi huấn luyện sẽ nằm trong thư mục `outputs/`.

### 3.3. ⚔️ DPO / RLHF Alignment (Huấn luyện DPO)
Giúp mô hình học cách ứng xử giống con người hơn thông qua dữ liệu phân cực (Tốt/Xấu).
1. Bạn cần chuẩn bị file dữ liệu có cấu trúc 3 cột: `prompt` (câu hỏi), `chosen` (câu trả lời đúng), `rejected` (câu trả lời sai).
2. Nếu bạn chưa có dữ liệu, hãy dùng Tab **LLM Arena** để bình chọn các câu trả lời. Hệ thống sẽ tự động tổng hợp ra file `outputs/arena_dpo_dataset.jsonl`.
3. Tải file này lên Tab DPO Alignment và nhấn **🚀 Start DPO Alignment**.

### 3.4. 🤖 RAG Studio (Tìm kiếm Vector)
Xây dựng mô hình Chatbot có khả năng đọc tài liệu nội bộ (PDF/TXT).
1. Tại Tab RAG, nhập đường dẫn file PDF (ví dụ: `tailieu_congty.pdf`).
2. Bấm **Build Vector DB**.
3. Quay lại Tab Chat Inference để bắt đầu trò chuyện. AI sẽ sử dụng Vector DB vừa xây dựng để cung cấp thông tin chính xác.

### 3.5. 💬 Giao diện Trò chuyện Nâng cao (Explainable Chat)
Khi chat với mô hình tại Tab Chat Inference, bạn có thể tích chọn **⚡ Show Generation Speed (Tokens/sec)**. Hệ thống sẽ tự động đo lường và hiển thị tốc độ sinh từ của mô hình (tokens/second) ở cuối mỗi câu trả lời, giúp bạn đánh giá hiệu năng cục bộ.

### 3.6. 🗂️ LoRA Manager
Tính năng quản lý các trọng số LoRA sau khi huấn luyện:
- Chuyển sang Tab LoRA Manager và nhấn **Scan for Adapters**.
- Hệ thống sẽ liệt kê tất cả các adapter đang có trong thư mục `outputs/` kèm theo dung lượng (MB).

### 3.7. 📦 Export GGUF & Triển khai Ollama
Sau khi hợp nhất LoRA vào Base Model, bạn có thể xuất mô hình để chạy trên các thiết bị yếu:
- Ở Tab Export & Hub, nhập đường dẫn mô hình và chọn định dạng lượng tử (ví dụ: `q4_k_m`).
- Nhấn **Export to GGUF**.
- Sau đó, bạn có thể điền tên mô hình và nhấn **Push to Ollama** để hệ thống tự động đưa mô hình vào Ollama (yêu cầu máy tính đã cài đặt phần mềm Ollama). Mở Terminal lên và gõ `ollama run <tên-mô-hình>` để trò chuyện trực tiếp!

### 3.8. 🖥️ System Monitor (Giám sát Tài nguyên)
Tab này cho phép bạn theo dõi tình trạng tiêu thụ RAM, CPU, và VRAM của GPU theo thời gian thực. Hãy thường xuyên kiểm tra Tab này trong lúc Training để đảm bảo không bị quá tải bộ nhớ (Out of Memory - OOM).

## 4. Cấu trúc Thư mục
- `evonet_studio/ui/`: Chứa mã nguồn của từng Tab giao diện (SFT, DPO, RAG,...).
- `evonet_studio/engine.py`: Xử lý toàn bộ logic liên lạc với HuggingFace, Unsloth và PyTorch.
- `outputs/`: Nơi lưu trữ các mô hình sau khi huấn luyện xong.
- `qa_tests/`: Thư mục chứa các kịch bản kiểm thử (Pytest/Unittest) để đảm bảo chất lượng hệ thống.

## 5. Lộ trình Phát triển (Product Roadmap - Open Core)
Để mang đến công cụ AI mã nguồn mở mạnh mẽ nhất cho cộng đồng, chúng tôi đang có kế hoạch phát triển các tính năng sau trong tương lai:
- **Tối ưu hóa Apple Silicon (MPS):** Hỗ trợ tăng tốc phần cứng tốt hơn cho người dùng MacOS (M1/M2/M3) không có card NVIDIA.
- **Tích hợp Llama.cpp WebUI:** Kết nối trực tiếp Giao diện quản lý model để chạy local mượt mà hơn.
- **Thư viện Template Cộng đồng:** Kho giao diện Prompt Template có sẵn (System Prompts, RLHF guidelines) được cộng đồng đóng góp.
- **Auto-Evaluation Metrics Cấp Cao:** Đưa vào các chỉ số BLEU, ROUGE, BERTScore trực tiếp trên UI để người dùng so sánh chất lượng model.

## 6. Hỗ trợ & Đóng góp
Nếu bạn gặp vấn đề hoặc muốn bổ sung tính năng mới cho bản Open-Core, xin vui lòng tạo Issue hoặc Pull Request trên trang GitHub của chúng tôi. 

Cảm ơn bạn đã tin dùng EvoNet-Studio!

---
*© 2026 EvoNet Framework. Phiên bản Open-Source.*
