# 📚 EvoNet-Studio Documentation Hub

Chào mừng bạn đến với Thư viện Tài liệu chính thức của **EvoNet-Studio**. Tùy thuộc vào phiên bản bạn đang sử dụng, vui lòng chọn tài liệu phù hợp dưới đây:

## 📖 Dành cho Người Dùng Miễn Phí (Open-Source)
Phiên bản Open-Core được thiết kế cho Cộng đồng Nghiên cứu, Sinh viên và Lập trình viên độc lập. Bạn có thể tự do clone mã nguồn, huấn luyện (Fine-tuning), thiết kế mô hình RAG (Retrieval-Augmented Generation) và sử dụng Agentic AI.
👉 **[Xem Hướng Dẫn Sử Dụng Bản Open-Source (Miễn Phí)](Open_Source_Guide.md)**

## 🏢 Dành cho Doanh Nghiệp (Enterprise Edition)
Phiên bản Enterprise (Thương mại) là một giải pháp đóng gói hoàn chỉnh (Black-box), hỗ trợ bảo mật dữ liệu tuyệt đối (Auto PII Masking), phân quyền người dùng (RBAC), và khóa phần cứng (Hardware Lock) để bảo vệ tài sản trí tuệ. 
👉 **[Xem Sách Hướng Dẫn Vận Hành Hệ Thống Doanh Nghiệp](Enterprise_Manual.md)**

---
### Các Thành Phần Kiến Trúc Quan Trọng

Nếu bạn là Lập trình viên muốn tìm hiểu sâu về kiến trúc đằng sau EvoNet-Studio:
1. **Vietnamese AI Framework:** Core lõi xử lý được lập trình bằng Python & Rust. Hỗ trợ Unsloth (Fine-tuning siêu tốc), MemGPT (Bộ nhớ dài hạn cho AI Agent), và vLLM (Inference tốc độ cao).
2. **DePIN Architecture:** Cho phép triển khai AI thành mô hình phi tập trung. Các Node tính toán có thể giao tiếp với nhau bằng giao thức gRPC.
3. **Data Flywheel:** Cơ chế tự động ghi nhận hội thoại của người dùng (Prompt) và những đánh giá tốt/xấu (Chosen/Rejected) để sau đó hệ thống tự động Fine-tune cải thiện mô hình bằng thuật toán DPO/ORPO.

*Cảm ơn bạn đã đồng hành cùng EvoNet AI.*
