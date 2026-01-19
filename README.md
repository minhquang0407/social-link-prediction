# 🕸️ Phân tích Mạng xã hội (Wikidata) & Dự đoán Mối liên kết bằng AI

> **Dự án Khoa học Dữ liệu End-to-End**: Từ thu thập dữ liệu đồ thị tri thức (Wikidata) đến triển khai mô hình AI dự đoán liên kết (Link Prediction) và ứng dụng tương tác.

---

## 🚀 Giới thiệu (Overview)

Dự án này xây dựng một hệ thống phân tích mạng xã hội của những người nổi tiếng và các thực thể liên quan (như trường học, đảng phái, nơi làm việc...). Hệ thống sử dụng dữ liệu thực tế từ **Wikidata**, mô hình hóa dưới dạng đồ thị (Graph), và áp dụng các kỹ thuật **Học Sâu trên Đồ thị (Graph Neural Networks - GNN)** để dự đoán các mối quan hệ tiềm năng chưa được khai phá.

### Mục tiêu chính:
1.  **Xây dựng Cơ sở dữ liệu Đồ thị**: Thu thập và làm sạch dữ liệu quan hệ phức tạp từ Wikidata thông qua SPARQL.
2.  **Phân tích Mạng lưới ("Sáu Bậc Xa cách")**: Tìm đường đi ngắn nhất kết nối hai nhân vật bất kỳ.
3.  **Dự đoán Liên kết (AI/ML)**: Sử dụng mô hình GNN (GraphSAGE/HGT) để dự đoán xác suất tồn tại mối quan hệ giữa hai thực thể.
4.  **Trực quan hóa**: Cung cấp giao diện web trực quan để khám phá và tương tác với dữ liệu.

---

## ✨ Tính năng Cốt Lõi (Key Features)

*   **🔍 Tìm kiếm thông minh**: Hỗ trợ tìm kiếm mờ (Fuzzy Search) tên nhân vật nhanh chóng.
*   **✈️ Đường đi ngắn nhất**: Minh chứng lý thuyết "Sáu bậc xa cách" (Six Degrees of Separation) với thuật toán BFS tối ưu.
*   **🔮 AI Dự đoán**:
    *   Sử dụng **PyTorch Geometric** với kiến trúc **GraphSAGE** và **HGT (Heterogeneous Graph Transformer)**.
    *   Hỗ trợ xử lý đồ thị dị thể (Heterogeneous Graph) với nhiều loại node (Người, Tổ chức...) và edge (Vợ chồng, Đồng nghiệp, Học tại...).
*   **📊 Dashboard Phân tích**: Thống kê quy mô đồ thị, phân phối bậc (degree distribution), và các metrics mạng lưới.
*   **🌐 Giao diện Streamlit**: Tương tác mượt mà, trực quan hóa đồ thị với PyVis.

---

## 🛠️ Công nghệ Sử dụng (Tech Stack)

| Lĩnh vực | Công nghệ / Thư viện |
| :--- | :--- |
| **Ngôn ngữ** | Python 3.9+ |
| **Ứng dụng Web** | [Streamlit](https://streamlit.io/) |
| **Thu thập dữ liệu** | SPARQLWrapper (Wikidata API), Pandas |
| **Xử lý đồ thị** | Igraph |
| **AI/Deep Learning** | PyTorch, PyTorch Geometric (PyG) |
| **Lưu trữ dữ liệu** | Pickle, JSON |


---

## 📂 Cấu trúc Dự án (Project Structure)

```
Social-Link-Prediction/
├── application/            # Logic ứng dụng (Service Layer)
├── config/                 # Cấu hình hệ thống (Settings, Paths)
├── core/                   # Các thuật toán cốt lõi (BFS, Search)
├── data_output/            # Dữ liệu đầu ra (Graph, Model checkpoints)
├── data_pipeline/          # Pipeline thu thập & xử lý dữ liệu (ETL)
├── infrastructure/         # Tầng giao tiếp dữ liệu & Repositories
├── presentation/           # Giao diện người dùng (Streamlit UI)
├── scripts/                # Scripts chạy rời (CLI)
│   ├── etl_run.py          # Script chạy ETL
│   └── train_model.py      # Script huấn luyện AI
├── main.py                 # Điểm khởi chạy ứng dụng (Entry Point)
├── requirements.txt        # Danh sách thư viện phụ thuộc
└── README.md               # Tài liệu dự án
```

---

## ⚙️ Hướng dẫn Cài đặt (Installation)

### 1. Yêu cầu
*   Python 3.9+
*   Git

### 2. Tải kho chứa (Clone Repository)
```bash
git clone https://github.com/minhquang0407/Social-Link-Prediction.git
cd Social-Link-Prediction
```

### 3. Thiết lập môi trường ảo (Khuyến nghị)
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Cài đặt thư viện
```bash
pip install -r requirements.txt
```
*Lưu ý: Đối với `torch` và `torch_geometric`, nếu gặp lỗi, vui lòng tham khảo trang chủ PyTorch để cài phiên bản phù hợp với CUDA của máy bạn.*

---

## 🏃 Hướng dẫn Sử dụng (Usage)

Dự án cung cấp file `main.py` đóng vai trò là entry point cho mọi tác vụ.

### 1. Chạy quy trình ETL (Thu thập dữ liệu)
Thu thập dữ liệu từ Wikidata và xây dựng đồ thị:
```bash
python main.py --etl
```
*Quá trình này có thể mất nhiều thời gian tùy thuộc vào tốc độ mạng và giới hạn API.*

### 2. Huấn luyện Mô hình AI
Huấn luyện mô hình dự đoán liên kết trên dữ liệu:
```bash
python main.py --train
```
*Quá trình training sẽ sử dụng GPU nếu có (CUDA) và lưu model vào thư mục `data_output`.*

### 3. Chạy Ứng dụng Web (Streamlit)

```bash
streamlit run main.py
```
---

## 👥 Đội ngũ Thực hiện

*   **Nguyễn Quốc Anh Quân**: Kỹ sư Wikidata (Extractor / SPARQL).
*   **Đinh Nhật Tân**: Kỹ sư Đồ thị & AI (Transformer / AI Lead).
*   **Nguyễn Minh Quang**: Kỹ sư Module & Ứng dụng (Loader / App Lead)

---

## 📜 Giấy phép
Dự án được thực hiện cho mục đích học tập và nghiên cứu. Dữ liệu thuộc về [Wikidata](https://www.wikidata.org/).
