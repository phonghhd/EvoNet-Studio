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
- **Role-Based Access Control (RBAC):** Nhân viên bình thường chỉ có quyền Xem (Viewer) và chat với AI, không được quyền nhấn nút "Start Training" (tốn tài nguyên GPU). Quyền này chỉ dành cho tài khoản có Role là `admin`.

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
Để không ngừng gia tăng giá trị cho các Khách hàng Doanh nghiệp, EvoNet-Studio đang nghiên cứu và sẽ sớm ra mắt các tính năng cấp cao sau trong các bản cập nhật sắp tới (Q3/Q4):

- **Single Sign-On (SSO) & Azure AD Integration:** Hỗ trợ đăng nhập một chạm thông qua tài khoản nội bộ của doanh nghiệp (Google Workspace, Microsoft Entra ID).
- **Advanced Audit Logs:** Nhật ký truy vết toàn diện, ghi nhận chi tiết mọi hành động của người dùng (Who, What, When, Where) trên hệ thống để phục vụ điều tra và tuân thủ bảo mật.
- **Resource Quota Management:** Cho phép Admin giới hạn ngân sách GPU (GPU hours) hoặc số lượng Token tối đa cho từng phòng ban hoặc cá nhân.
- **Model & Data Registry:** Hệ thống quản lý phiên bản chuyên sâu (Versioning), giúp truy xuất chính xác phiên bản Dataset nào đã tạo ra phiên bản Model nào.

---
*© 2026 EvoNet Framework. Protected by Cython Engine.*
