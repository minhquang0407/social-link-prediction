import streamlit as st
import sys
import subprocess
from pathlib import Path

from config.settings import GRAPH_PATH, MODEL_PATH, PYG_DATA_PATH, MAPPING_PATH, NODES_DATA_PATH
from infrastructure.repositories import PickleGraphRepository, ModelRepository, PyGDataRepository
from application import AIService, AnalysisService
from core.logic import RapidFuzzySearch
from core.ai.gnn_architecture import LinkPredictionModel
from core.ai.predicter import Predictor
import pandas as pd
import torch

# --- 1. HÀM LOAD TÀI NGUYÊN VÀ LẮP RÁP SERVICE (BOOTSTRAP) ---

@st.cache_resource(show_spinner="Đang khởi động hệ thống...")
def bootstrap_services():
    """
    Hàm này chạy 1 lần duy nhất để lắp ráp và cache các Services đã nạp dữ liệu.
    """
    print("LOG: Bắt đầu quá trình Bootstrap hệ thống...")

    # --- INFRASTRUCTURE (Tầng 1) ---
    graph_repo = PickleGraphRepository(GRAPH_PATH)
    model_repo = ModelRepository(MODEL_PATH)
    feature_repo = PyGDataRepository(PYG_DATA_PATH, MAPPING_PATH)

    # 1. Load Graph
    G_full = graph_repo.load_graph()
    if G_full is None:
        print("LỖI: Không tải được đồ thị G_full.gpickle.")
        return None, None

    # 2. Lấy Search Engine & Fuzzy Search
    try:
        df_nodes = pd.read_parquet(NODES_DATA_PATH)
        search_engine = RapidFuzzySearch(df_nodes)
    except Exception as e:
        print(f"LỖI: Không tải được df_nodes từ {NODES_DATA_PATH}. Chi tiết: {e}")
        return None, None

    # 3. Load Model AI & PyG Data
    hetero_data = feature_repo.load_data()
    if hetero_data is None:
        print("LỖI: Không tải được hetero_data.pt.")
        return None, None

    try:
        metadata = hetero_data.metadata()
        # Khởi tạo kiến trúc mô hình rỗng
        model_arch = LinkPredictionModel(hidden_channels=256, out_channels=128, metadata=metadata)
        # Nạp trọng số từ file model.pt
        ai_model = model_repo.load_model(model=model_arch)
    except Exception as e:
        print(f"LỖI: Không khởi tạo hoặc tải được mô hình AI. Chi tiết: {e}")
        return None, None

    # --- APPLICATION (Tầng 2: Lắp ráp các Service) ---
    try:
        # Khởi tạo bộ máy dự đoán Link Predictor
        predictor = Predictor(model=ai_model, data=hetero_data)
        
        # Lắp ráp các Service
        analysis_service = AnalysisService(G_full, search_engine)
        ai_service = AIService(model=ai_model, embeddings=predictor.embeddings, engine=search_engine, predictor=predictor)
    except Exception as e:
        print(f"LỖI: Không lắp ráp được các dịch vụ. Chi tiết: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    print("LOG: Hệ thống Services đã được lắp ráp thành công.")
    return analysis_service, ai_service


def run_web_app():
    """Lắp ráp Services và khởi chạy giao diện AppRunner."""
    analysis_service, ai_service = bootstrap_services()

    if analysis_service and ai_service:
        from presentation import AppRunner
        app = AppRunner(analysis_service, ai_service)
        app.run()
    else:
        st.set_page_config(layout="wide", page_title="Social Network Analysis")
        st.title("⚠️ Lỗi Hệ thống")
        st.error("Không thể khởi động ứng dụng. Vui lòng kiểm tra file dữ liệu `G_full.gpickle` và chạy lại.")


# --- 3. HÀM CHẠY CÁC LỆNH TIỆN ÍCH (CLI) ---

def run_cli_command(command):
    """
    Xử lý các lệnh CLI: etl, train.
    Đây là nơi gọi các scripts/run_etl.py (dùng subprocess)
    """
    if command == "etl":
        print("BẮT ĐẦU: Chạy quy trình thu thập và xử lý dữ liệu (ETL)...")
        subprocess.run(["python", "scripts/etl_run.py"], check=True)
    elif command == "train":
        print("BẮT ĐẦU: Chạy quy trình huấn luyện AI...")
        subprocess.run(["python", "scripts/train_model.py"], check=True)
    else:
        print("\nSử dụng: python main.py [COMMAND]")
        print("COMMANDS:")
        print("  --etl     : Chạy quy trình thu thập và xử lý dữ liệu.")
        print("  --train   : Huấn luyện mô hình GNN (Sử dụng dữ liệu G_full đã có).")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith('--'):
        command = sys.argv[1].lstrip('--')
        run_cli_command(command)
    else:
        # Nếu không có tham số CLI, chạy Web App
        run_web_app()