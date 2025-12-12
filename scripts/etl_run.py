import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from config.settings import GRAPH_PATH, RAW_JSON_DIR
from infrastructure.pipelines import GraphTransformer, WikidataExtractor
from infrastructure.repositories import PickleGraphRepository


def run_etl_pipeline():
    # Khởi động Extractor, thu thập dữ liệu
    extractor = WikidataExtractor(user_agent= 'quangminh222@gmail.com')
    success = extractor.fetch_all_relationships()
    if not success:
        print('Failed to fetch data from Wikidata.')
        return
    # Khởi động Transformer, chuyển đổi dữ liệu đã thu thập
    transformer = GraphTransformer()
    relationship_graph = transformer.run_transformer(RAW_JSON_DIR)
    # Lưu đồ thị đã chuyển
    repo = PickleGraphRepository(GRAPH_PATH)
    success = repo.save_graph(relationship_graph)

    if success:
        print("✅ ETL Pipeline hoàn tất thành công!")
    else:
        print("❌ Lưu thất bại!")


if __name__ == "__main__":
    run_etl_pipeline()