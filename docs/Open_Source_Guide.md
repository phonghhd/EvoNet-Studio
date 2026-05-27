# Hướng dẫn Sử dụng EvoNet-Studio (Bản Open-Core Miễn Phí)

Chào mừng bạn đến với **EvoNet-Studio**, một hệ sinh thái AI (LLMOps) đa năng dành cho việc tinh chỉnh (Fine-tuning), RAG, và triển khai các mô hình ngôn ngữ lớn (LLMs). Tài liệu này hướng dẫn chi tiết cách sử dụng phiên bản Open-Source (miễn phí) trên GitHub.

## 1. Cài đặt Môi trường
EvoNet-Studio yêu cầu Python 3.10+ và hệ điều hành Linux (khuyến nghị Ubuntu).
Môi trường ảo (virtual environment) là bắt buộc để tránh xung đột thư viện.

```bash
# 1. Clone mã nguồn
git clone https://github.com/your-username/EvoNet-Studio.git
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
3. Chỉnh tham số: Epochs, Batch Size, Learning Rate.
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

### 3.5. 🖥️ System Monitor (Giám sát Tài nguyên)
Tab này cho phép bạn theo dõi tình trạng tiêu thụ RAM, CPU, và VRAM của GPU theo thời gian thực. Hãy thường xuyên kiểm tra Tab này trong lúc Training để đảm bảo không bị quá tải bộ nhớ (Out of Memory - OOM).

## 4. Cấu trúc Thư mục
- `evonet_studio/ui/`: Chứa mã nguồn của từng Tab giao diện (SFT, DPO, RAG,...).
- `evonet_studio/engine.py`: Xử lý toàn bộ logic liên lạc với HuggingFace, Unsloth và PyTorch.
- `outputs/`: Nơi lưu trữ các mô hình sau khi huấn luyện xong.
- `qa_tests/`: Thư mục chứa các kịch bản kiểm thử (Pytest) để đảm bảo chất lượng hệ thống.

## Hỗ trợ & Đóng góp
Nếu bạn gặp vấn đề hoặc muốn bổ sung tính năng mới cho bản Open-Core, xin vui lòng tạo Issue hoặc Pull Request trên trang GitHub của chúng tôi. 
Cảm ơn bạn đã tin dùng EvoNet-Studio!
