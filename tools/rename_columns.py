import pandas as pd
import re


def to_snake_case(name: str) -> str:
    """
    Chuyển đổi chuỗi từ camelCase/PascalCase/Space-separated sang snake_case.
    """
    # Bước 1: Xử lý các cụm viết tắt và ký tự hoa liên tiếp
    # Ví dụ: personSubType -> person_Sub_Type
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)

    # Bước 2: Tách giữa ký tự thường/số và ký tự hoa
    # Ví dụ: person_Sub_Type -> person_sub_type
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    # Bước 3: Thay thế khoảng trắng hoặc gạch ngang thành gạch dưới
    return s2.replace(" ", "_").replace("-", "_")


edges.columns = [to_snake_case(col) for col in edges.columns]

