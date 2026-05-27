# Hướng dẫn Vận hành EvoNet-Studio Enterprise

Tài liệu này dành cho Khách hàng Doanh nghiệp (Enterprise Users) đã mua bản quyền EvoNet-Studio.

## 1. Yêu cầu Hệ thống
- Hệ điều hành: Linux (Ubuntu 22.04 khuyến nghị)
- GPU: Tối thiểu 1x NVIDIA GPU (hỗ trợ CUDA 12.1)
- RAM: Tối thiểu 32GB
- Storage: Tối thiểu 100GB SSD
- Phần mềm: Docker & NVIDIA Container Toolkit đã được cài đặt.

## 2. Hướng dẫn Khởi chạy
Sau khi thanh toán thành công, bạn sẽ nhận được một file nén (ví dụ: `TenCongTy_EvoNet_Enterprise.zip`). Bên trong có chứa file `license.key`.

1. **Tải Docker Image:**
   Tải file `.tar` của phần mềm về từ đường link do nhà phát triển cung cấp. Sau đó load vào Docker:
   ```bash
   docker load -i evonet_enterprise_v1.tar
   ```

2. **Chạy Phần mềm (Mount License Key):**
   Bạn bắt buộc phải đặt file `license.key` cùng thư mục mà bạn chuẩn bị gõ lệnh khởi chạy.
   ```bash
   docker run -d \
     --gpus all \
     -p 8080:8080 \
     -v $(pwd)/license.key:/app/license.key \
     --name evonet-enterprise \
     evonet-enterprise:v1
   ```

3. **Truy cập Giao diện:**
   Mở trình duyệt web và truy cập `http://<IP_MAY_CHU_CUA_BAN>:8080`
   Hệ thống sẽ yêu cầu đăng nhập. Vui lòng liên hệ Admin để nhận tài khoản (mặc định ban đầu có thể là `admin` / `admin_password`).

## 3. Các Tính năng Độc quyền (Enterprise)
- **Auto PII Masking:** Khi xử lý dữ liệu trong Tab Data Cleaner, tính năng PII Masking tự động ẩn danh thẻ tín dụng, số điện thoại để tuân thủ luật bảo mật.
- **Hardware Lock:** File license.key của bạn chỉ hoạt động trên Máy chủ có địa chỉ MAC trùng khớp với lúc bạn đăng ký mua. Việc sao chép file sang máy khác sẽ khiến phần mềm văng lỗi và từ chối khởi động.
- **Quản lý API Key & Rate Limiting:** Thiết lập hạn mức truy cập API cho từng người dùng, ngăn chặn tình trạng D-DoS hoặc spam token từ nội bộ.
- **SSO (Single Sign-On):** Hỗ trợ đăng nhập nhanh qua Google/Microsoft Azure AD OAuth2, loại bỏ việc quản lý hàng trăm tài khoản phân tán.
- **Huấn luyện Phân tán Đa Máy Chủ (Multi-Node):** Kích hoạt DeepSpeed ZeRO-3 chỉ bằng một click, tự động ghép nối sức mạnh từ nhiều máy chủ GPU vật lý.
- **Dashboard Chi Phí GPU:** Ước tính và theo dõi chi phí điện toán quy ra USD (ví dụ: \$3/giờ cho A100) trên thời gian thực.
- **Canary & A/B Testing Deployment:** Cho phép phân chia lưu lượng (Traffic Split) giữa Model cũ và Model mới để kiểm thử A/B không rủi ro.

## 4. Kiểm thử Tự động (QA Testing)
Phiên bản Enterprise đi kèm bộ kiểm thử nội bộ cực kỳ khắt khe:
- Chạy thử lệnh: `venv/bin/pytest qa_tests/ -v`
- Bộ test sẽ rà quét tính ổn định của License Manager, thuật toán ẩn danh PII Masking, và chức năng thanh toán trực tuyến (SaaS Portal).

## 5. Cổng kết nối (API Gateway)
Phiên bản Doanh nghiệp cho phép triển khai Mô hình AI thành dạng API (chuẩn OpenAI) cho các dự án nội bộ.
- Trong Tab "1-Click Deployment", bạn có thể tích hợp **vLLM** (Engine tốc độ cao) để phục vụ hàng ngàn request đồng thời.
- Để cấp phép truy cập API cho bên thứ 3, hãy cấu hình file `.env` hoặc truyền API keys thông qua giao diện Portal.

## 6. Xử lý Sự cố (Troubleshooting)
- **Lỗi "Hardware Lock Violation":** Bạn đã mang file key sang một máy chủ khác. Vui lòng liên hệ đội ngũ EvoNet để mua thêm Node mới.
- **Lỗi "License expired":** Key của bạn đã hết hạn. Vui lòng truy cập trang bán hàng để gia hạn hợp đồng.
- **Lỗi Cạn kiệt VRAM (OOM):** Mở tính năng **PagedAttention** trong phần cài đặt khởi chạy Server hoặc sử dụng kỹ thuật lượng tử hóa (Quantization) xuống `4-bit`.

## 7. Lộ trình Phát triển (Product Roadmap)
EvoNet-Studio Enterprise không ngừng nâng cấp để đáp ứng quy mô tập đoàn. Các tính năng sắp ra mắt trong bản cập nhật tới:
- **Advanced Audit Logs:** Nhật ký truy vết toàn diện, ghi nhận chi tiết mọi hành động của người dùng (Who, What, When, Where) trên hệ thống để phục vụ điều tra và tuân thủ bảo mật.
- **Resource Quota Management:** Cho phép Admin giới hạn ngân sách GPU (GPU hours) hoặc số lượng Token tối đa cho từng phòng ban hoặc cá nhân.
- **Model & Data Registry:** Hệ thống quản lý phiên bản chuyên sâu (Versioning), giúp truy xuất chính xác phiên bản Dataset nào đã tạo ra phiên bản Model nào.
- **Multi-Tenant Architecture:** Kiến trúc Đa khách thuê thực thụ (SaaS-ready), cho phép một bản cài đặt phục vụ độc lập nhiều phòng ban mà không chia sẻ tài nguyên dữ liệu.
- **Active Directory/LDAP Sync:** Đồng bộ hóa danh bạ người dùng nội bộ tự động để quản trị phân quyền tập trung.
- **Continuous Pre-Training (CPT) Cluster Scheduling:** Lập lịch huấn luyện tự động, tích hợp với SLURM/Ray để phân bổ hàng Terabyte dữ liệu chạy định kỳ.
- **Advanced RAG Evaluation Suites:** Bổ sung các công cụ đo lường RAG (RAGAS, TruLens) trực tiếp vào bảng điều khiển để chấm điểm độ trung thực của AI.

---
*© 2026 EvoNet Framework. Protected by Cython Engine.*