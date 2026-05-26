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

## 4. Xử lý Sự cố (Troubleshooting)
- **Lỗi "Hardware Lock Violation":** Bạn đã mang file key sang một máy chủ khác. Vui lòng liên hệ đội ngũ EvoNet để mua thêm Node mới.
- **Lỗi "License expired":** Key của bạn đã hết hạn 1 năm. Vui lòng truy cập trang bán hàng để gia hạn hợp đồng.

---
*© 2026 EvoNet Framework. Protected by Cython Engine.*
