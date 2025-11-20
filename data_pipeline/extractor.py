import sys
import json
import os
import time
from datetime import datetime
from SPARQLWrapper import SPARQLWrapper, JSON
from data_pipeline.queries import QueryTemplates as qt
class WikidataExtractor:

    def __init__(self, user_agent):
        if not user_agent:
            raise ValueError("User-Agent là bắt buộc để truy vấn Wikidata.")

        self.endpoint_url = "https://query.wikidata.org/sparql"
        self.sparql = SPARQLWrapper(self.endpoint_url)
        self.sparql.agent = user_agent
        self.sparql.setReturnFormat(JSON)
        self.sparql.setTimeout(300) 

    def _run_basic_query(self, base_query: str, page_size=10000) -> list:

        page = 1
        all_bindings = []
        json_head = None
        retry_count = 0
        offset_num = 0
        max_retries = 30
        current_page_size = page_size 

        while True:
            paginated_query = base_query + f"\nLIMIT {current_page_size}\nOFFSET {offset_num}"
            self.sparql.setQuery(paginated_query)
            
            print(f"{datetime.now().strftime('%H:%M:%S')} Đang lấy trang {page}, {current_page_size}/page...", end="", flush=True)
            
            try:
                start_time = time.monotonic()
                
                # Truy vấn và decode
                response = self.sparql.query()
                raw_data_bytes = response.response.read()
                cleaned_data_string = raw_data_bytes.decode('utf-8', errors='ignore') # Dùng 'ignore' cho an toàn
                results = json.loads(cleaned_data_string)
                
		
                end_time = time.monotonic()
                duration = end_time - start_time
                
                if json_head is None:
                    json_head = results["head"]
                bindings = results["results"]["bindings"]
                
                retry_count = 0
         
                all_bindings.extend(bindings)
                
                if not bindings or len(bindings) < current_page_size:
                    print(f"\n-------> Đã lấy hết! Tổng {len(all_bindings)}")
                    break

                print(f" OK! Lấy {len(bindings)}, mất {int(duration)}s", end="\n", flush=True)
                
                page += 1
                offset_num += current_page_size
                time.sleep(1)

            except Exception as e:
                print(f"\n!!! LỖI KHI ĐANG TRUY VẤN (offset {offset_num}): {e}", file=sys.stderr)
                retry_count += 1
                
                if retry_count > max_retries:
                    print(f"    Đã thử lại {max_retries} lần thất bại. TỪ BỎ truy vấn này.", file=sys.stderr)
                    break
                else:
                    if retry_count % 5 == 0 and retry_count > 0:
                        sleep_time = 60 * (retry_count // 5)
                    else:
                        sleep_time = 5 * retry_count
                    
                    if current_page_size > 2000:
                        current_page_size -= 2000
                    
                    print(f"    Đang thử lại (lần {retry_count}/{max_retries}) sau {sleep_time}s với {current_page_size}/page...", file=sys.stderr)
                    time.sleep(sleep_time)
                    
        return all_bindings

    def _create_intervals(self, start_val, end_val, step=10) -> list:
        intervals = []
        current_start = start_val
        while current_start < end_val:
            current_end = current_start + step
            if current_end > end_val:
                current_end = end_val
            intervals.append((current_start, current_end))
            current_start = current_end
        return intervals


    def _run_paginated_query(self, start, end, base_query, page_size) -> list:
        """
        Chạy query theo từng khoảng thời gian (Intervals).
        """
        all_bindings = []
        intervals = self._create_intervals(start, end + 1)
        
        print(f"=========== BẮT ĐẦU CHẠY THEO KHOẢNG {start}-{end} ===========")
        start_time = datetime.now()
        
        for start_year, end_year in intervals:
            print(f"\n--- KỶ NGUYÊN {start_year}-{end_year} ---")
            
            year_filter_str = f"FILTER(YEAR(?person_dob) > {start_year} && YEAR(?person_dob) <= {end_year})"
            era_query = base_query.replace("##YEAR_FILTER_HOOK##", year_filter_str)
            
            binding = self._run_basic_query(era_query, page_size)
            all_bindings.extend(binding)
            
            print(f"--- KẾT THÚC {start_year}-{end_year}. (Tổng tích lũy: {len(all_bindings)}) ---")
            
        end_time = datetime.now()
        print(f"========== TỔNG KẾT: {len(all_bindings)} kết quả, Thời gian: {end_time - start_time} ========== ")

        return all_bindings

    def _save_data(self, all_bindings, name, output_dir="data"):
        output_filename = os.path.join(output_dir, f"raw_data_{name}.json")
        
        final_json_output = {
            "head": {"vars": []}, # (Chúng ta chấp nhận vars rỗng hoặc có thể cải thiện sau)
            "results": {"bindings": all_bindings}
        }
        
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(final_json_output, f, ensure_ascii=False, indent=2)
            print(f"\n==> ĐÃ LƯU: {len(all_bindings)} kết quả vào file '{output_filename}'")
        except IOError as e:
            print(f"\n!!! LỖI KHI LƯU FILE: {e}", file=sys.stderr)
        
        return output_filename

    def fetch_all_relationships(self, relationship_queries, start, end, output_dir="data"):
        os.makedirs(output_dir, exist_ok=True)
        all_data = {}
        base_query = qt.BASE

        for name, (snippet, page_size) in relationship_queries.items():
            
            print(f"\n\n################ STARTING JOB: {name} ################")
            full_query = base_query.replace("##FIND_HOOK##", snippet)
            all_bindings = self._run_paginated_query(start, end, full_query, page_size)
            
            output_filename = self._save_data(all_bindings, name, output_dir)

            try:
                with open(output_filename, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    all_data[name] = loaded_data["results"]["bindings"]
            except Exception:
                all_data[name] = []
            
            time.sleep(5)

        print("\n*** HOÀN TẤT TẤT CẢ TRUY VẤN! ***")
        return all_data

if __name__ == "__main__":
    YOUR_USER_AGENT = "minhquang04072005@gmail.com"
    OUTPUT_DIR = "data_output"


  
    queries_to_run = qt.get_all_queries()

    extractor = WikidataExtractor(user_agent=YOUR_USER_AGENT)
    extractor.fetch_all_relationships(queries_to_run, 1800, 2020, OUTPUT_DIR)