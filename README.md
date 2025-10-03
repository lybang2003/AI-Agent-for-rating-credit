# AI-Agent-for-rating-credit

## Hướng dẫn cài đặt môi trường và chạy ứng dụng

### 1. Tạo môi trường ảo (khuyến nghị dùng Python >=3.10)

```bash
python -m venv .venv
```

### 2. Kích hoạt môi trường ảo
- **Windows (cmd):**
  ```cmd
  .venv\Scripts\activate
  ```
- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Linux/Mac:**
  ```bash
  source .venv/bin/activate
  ```

### 3. Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

### 4. Chạy ứng dụng Streamlit

```bash
streamlit run streamlit_app.py
```

### 5. Lưu ý
- Không commit thư mục `.venv` và các file dữ liệu lớn lên GitHub.
- Nếu gặp lỗi về thiếu thư viện, hãy kiểm tra lại file `requirements.txt` hoặc cài đặt bổ sung.


