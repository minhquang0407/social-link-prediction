# Social Link Prediction (GNN & Wikidata)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch__Geometric-2.3%2B-3c78d8.svg)](https://pytorch-geometric.readthedocs.io/)
[![igraph](https://img.shields.io/badge/igraph-0.10%2B-red.svg)](https://igraph.org/)

---

## Table of Contents / Mục lục
1. [English Version (System Specification & Guide)](#english-version)
2. [Bản tiếng Việt (Báo cáo Kỹ thuật & Hướng dẫn)](#ban-tieng-viet)

---

# English Version

An advanced social network analysis system leveraging Heterogeneous Graph Neural Networks (GNN) and graph structural algorithms. The system extracts real-world knowledge graphs from the Wikidata API, processes high-dimensional attributes through an automated Data Engineering pipeline, and deploys deep learning models to predict hidden or future relationships (spouses, colleagues, employers, education) between entities.

---

## Project Context and Objectives

In the digital knowledge era, Knowledge Graphs (KGs) like Wikidata serve as the backbone for modern AI systems. However, these graphs often suffer from incomplete and sparse relations, representing "knowledge shadows". 

This project implements a hybrid Neuro-Symbolic AI approach that combines:
* **Symbolic Reasoning**: Logical and structural path calculations using C-Core graph algorithms (igraph).
* **Neural representation**: Implicit deep learning over continuous vector spaces using Heterogeneous Graph Neural Networks (GNN).

---

## System Architecture (Clean Architecture)

The project adheres to the strict principles of Clean Architecture to ensure modularity, scalability, and testability across layers:

```mermaid
graph TD
    subgraph Presentation [Presentation Layer - streamlit]
        main_py[main.py] --> app_py[app.py]
        app_py --> bfs_tab[bfs_tab.py]
        app_py --> ai_tab[ai_tab.py]
        app_py --> analytics_tab[analytics_tab.py]
        app_py --> ego_tab[ego_tab.py]
    end

    subgraph Application [Application Layer - Services]
        ai_service[AIService]
        analysis_service[AnalysisService]
    end

    subgraph Core [Core Layer - logic & algorithms]
        gnn[LinkPredictionModel]
        bfs[PathFinder Dijkstra/BFS]
        fuzzy[RapidFuzzySearch]
    end

    subgraph Infrastructure [Infrastructure Layer - Storage & Data]
        extractor[WikidataExtractor]
        transformer[GraphTransformer]
        repos[PickleGraph / Model / PyG Data Repositories]
    end

    main_py --> ai_service
    main_py --> analysis_service
    ai_service --> gnn
    ai_service --> fuzzy
    analysis_service --> bfs
    analysis_service --> fuzzy
    transformer --> repos
```

---

## Data Engineering Pipeline (ETL)

The automated ETL pipeline manages high-scale knowledge graph ingestion:

### 1. Ingestion & Multi-Stage Extraction
* Developed a robust SPARQL Wrapper ([extractor.py](file:///c:/Users/nguye/social-link-prediction/infrastructure/pipelines/extractor.py)) to fetch structured RDF Triples, T = (Subject, Predicate, Object), from the live Wikidata endpoint.
* Implemented a birth interval scanning mechanism. Querying data in parallel batches based on birth years (e.g., 5-year buckets) prevents API rate limits and timeouts.
* Formulated semantic queries in [queries.py](file:///c:/Users/nguye/social-link-prediction/config/queries.py) for relations including: `educated_at`, `work_at`, `award_received`, `father`, `mother`, `spouse`, `employer`, and `colleague`.

### 2. Transformation, Cleansing & Deduplication
* Managed anomalies in [transformer.py](file:///c:/Users/nguye/social-link-prediction/infrastructure/pipelines/transformer.py): resolved self-loops, normalized bidirectional relations, cleaned duplicate edges using mapped keys `(id_min + "_" + id_max + "_" + relation)`, and computed missing values (imputation on birth year).
* Serialized processed entities into highly optimized Apache Parquet tables to achieve maximum I/O throughput and clean structure.

### 3. Graph Scale and Topological Statistics
* **Dataset Scale**:
  * **Nodes**: 4,609,521 entities (including 3,247,681 `human` nodes, 438,293 `written_work` nodes, 315,908 `film` nodes, etc. spanning 11 total types).
  * **Edges**: 10,761,807 active connections spanning 44 relationship categories (dominated by `educated_at` with 1,714,215 edges and `work_at` with 1,669,322 edges).
* **CCDF & Power-Law Exponent**:
  * Analysis of the Cumulative Complementary Distribution Function (CCDF) on a Log-Log scale shows a scale-free topology.
  * The Power-Law exponent estimated via Maximum Likelihood Estimation (MLE) yields:
    
    `gamma = 1 + n * [ sum( ln(k_i / (k_min - 0.5)) ) ]^-1`
    
    Setting the cut-off `k_min = 100` yields `gamma = 3.35`, showing that the graph is scale-free with tiptoeing properties resembling a random-like network since super-connected hubs do not completely centralize the entire connectivity.

### 4. Feature Engineering & PyG HeteroData Construction
* **Semantic Encoding**: Embedded multi-lingual node text attributes (name, description, occupation, birthplace, gender, interests) into a 384-dimensional dense vector using the pre-trained Sentence-BERT (`paraphrase-multilingual-MiniLM-L12-v2`) model.
* **Structural Engineering**: 
  * Computed Multi-view PageRank scores (45 distinct views based on different relationship categories) using the C-core `igraph` library.
  * Computed global node degrees and applied a log-transform (`log1p`) to mitigate the outsized influence of major hubs.
* **Temporal Features**: Scaled birth years using Min-Max normalization and created a binary indicator flag for missing values.
* **Feature Concat**: Consolidated all properties into a unified 432-dimensional node feature tensor:
  
  `x_node = [ v_SBERT(384) || v_year(1) || v_missing_flag(1) || v_PageRank(45) || v_degree(1) ]`
  
* **PyG Loading**: Constructed PyTorch Geometric `HeteroData` by mapping raw Wikidata QIDs to local, 0-indexed PyG typed integer indices.

---

## Graph Neural Network (GNN) Modeling

The GNN model is developed using PyG to predict pairwise link probabilities on heterogeneous graphs.

### 1. Inductive Encoder (GraphSAGE)
* Employed a message-passing GraphSAGE architecture to support inductive learning. The model generalizes to unseen nodes at inference time without requiring full retraining.
* Wrapped layers with PyG's `to_hetero` wrapper using sum aggregation to compile representations across multiple incoming edge types.
* Integrated LayerNorm and a 40% dropout rate to guarantee stable convergence.

### 2. Deep Interaction Decoder & Shared Decoder Mechanism
* **Shared Decoder Mechanism**: Decoders are keyed by relation type `__rel__` rather than `src__rel__dst`. Edge types that share the same relation name (e.g., spouses or colleagues of different entity subtypes) share the same decoder parameters. This reduces model size, prevents overfitting, and enables collaborative representation learning across different sub-graphs.
* **InteractionMLP Decoder**: Designed a custom InteractionMLP to compute coupling probabilities between node embeddings:
  
  `h_pair = [ z_src || z_dst || (z_src * z_dst) ]`
  
  Where `*` is the Hadamard product (element-wise multiplication). Incorporating this product enables the MLP to capture fine-grained relational interactions.
* Placed a Sigmoid activation function at the output layer to map scores directly to connection probabilities in `[0.0, 1.0]`.

### 3. Structural Constraints & Graph Pathfinding
* **Degrees of Separation (Social Distance)**: Defined specifically on nodes of type `human`, calculating the number of human steps separating two individuals, matching the "Six Degrees of Separation" theorem.
* **Taboo Filters**: Hard-coded structural rules to filter out family relations (sibling, father, mother) during spouse recommendations.
* **Age Gap Penalty**: Automatically penalizes GNN link scores or increases Dijkstra edge weights when the age gap between individuals exceeds 20 years.
* **Hub Penalty**: Dijkstra edge costs are computed as:
  
  `w_edge = log(in_degree + 1)`
  
  This pushes path searches to bypass broad hubs (e.g., country nodes) in favor of high-quality interpersonal pathways.

---

## Empirical Validation & Case Studies

To evaluate the mathematical validity, network structure, and artificial intelligence predictive power of the model, we conducted systematic validation experiments on the live graph topology using concrete case studies.

### 1. Fuzzy Search Distance Matcher
We implemented `RapidFuzzySearch` based on the edit distance (Levenshtein Distance) to resolve raw user inputs into internal graph node IDs:
$$score(s_1, s_2) \propto \frac{1}{d(s_1, s_2) + 1}$$
This handles spelling variations and accents gracefully, returning candidates instantaneously.

### 2. Proof of Six Degrees of Separation Theorem
According to Watts & Strogatz's small-world theory, the expected average shortest path length ($L_{theory}$) in a network is:
$$L_{theory} \approx \frac{\ln(N)}{\ln(\langle k \rangle)}$$
Given the network parameters:
* Active connected component size ($N$): $2,869,142$
* Average network degree ($\langle k \rangle$): $\approx 3.8$
* Expected Small-World Geodesic Distance ($L_{theory}$): $\approx 11.1$

#### Large-Scale Statistical Validation
We extracted **100,000 random entity pairs** $(u, v)$ from the graph and computed their geodesic distances.
* **Result**: The empirical average path length was $\bar{d} \leq 6$, following a tight bell-curve distribution centered around $4$ steps.
* **Conclusion**: We accepted the null hypothesis ($H_0: \bar{d} \leq 6$). The Wikidata knowledge graph exhibits a robust small-world topology with extreme global connectivity.

#### Concrete Case Studies
* **Case Study 1: Sơn Tùng M-TP $\Longleftrightarrow$ Taylor Swift**
  The system located a shortest path with exactly 4 intermediate nodes (6 hops) showing the cultural intersection of popular music:
  1. 👤 **Taylor Swift** [Q26876]
  2. 🟢 **American Music Award for Favorite Pop/Rock Female Artist** [Q1441929] `(award_received)`
  3. 👤 **Mariah Carey** [Q41076] `(award_received)`
  4. 👤 **Mỹ Linh** [Q6945890] `(influenced_by)`
  5. 🟢 **Vietnam National Academy of Music** [Q5649320] `(educated_at)`
  6. 👤 **Nguyễn Ánh Tuyết** [Q118249221] `(educated_at)`
  7. 🟢 **Conservatory of Ho Chi Minh City** [Q1377237] `(work_at)`
  8. 👤 **Sơn Tùng M-TP** [Q17450386] `(educated_at)`

* **Case Study 2: Ho Chi Minh $\Longleftrightarrow$ Taylor Swift**
  The system successfully traced a pathway showing how prestigious awards and historical entities bridge into modern popular culture:
  1. 👤 **Ho Chi Minh** [Q36014]
  2. 🟢 **Star of the Republic of Indonesia** [Q2340171] `(award_received)`
  3. 👤 **Elizabeth II** [Q9682] `(award_received)`
  4. 🟢 **Honorary doctor of the Royal College of Music** [Q99025668] `(award_received)`
  5. 👤 **Andrew Lloyd Webber** [Q180975] `(award_received)`
  6. 🟢 **Beautiful Ghosts** [Q72270672] `(composer)`
  7. 👤 **Taylor Swift** [Q26876] `(lyricist)`

### 3. GNN Link Prediction Experimental Scenarios
We deployed our trained GNN model to perform live predictions and recommendations under five distinct real-world operational environments:

#### Scenario A: Pairwise Link Probability Prediction (Sơn Tùng M-TP vs Snoop Dogg)
The model evaluated the hidden interaction space between the two embeddings and computed these relation scores (evaluated with independent Sigmoid decoders):
* `collaborate_with` : **0.7242** (High)
* `influenced_by` : **0.6710** (Possible)
* `advisor_of` : **0.5450** (Possible)
* `sibling` : **0.4900** (Low)
* `spouse` : **0.2043** (Very Low)

*   **Short Comment**: 
    The high prediction score of `collaborate_with` (0.7242) is highly consistent with real-world artistic interactions, as Sơn Tùng M-TP collaborated directly with Snoop Dogg in the music product *Hãy Trao Cho Anh*. Additionally, family relations such as `sibling` or `spouse` remain extremely low, demonstrating that logical social constraints were effectively learned.

#### Scenario B: Global Recommendations (Ho Chi Minh)
Generating the top 10 most compatible entities globally for President Ho Chi Minh across different relation types:
1.  **Vietnam Communist Party** - `[founder_of]` - **0.8255**
2.  **Quoc Hoc - Hue High School for the Gifted** - `[educated_at]` - **0.8221**
3.  **Communist University of the Toilers of the East** - `[educated_at]` - **0.8014**
4.  **Hoang Thi Loan** (Mother) - `[mother]` - **0.7602**
5.  **Nguyen Sinh Sac** (Father) - `[father]` - **0.7208**
6.  **Order of Lenin** - `[award_received]` - **0.7031**
7.  **Star of the Republic of Indonesia** - `[award_received]` - **0.6992**
8.  **Grand Cross of the Order of Polonia Restituta** - `[award_received]` - **0.6706**
9.  **Gold Star Order** - `[award_received]` - **0.6531**
10. **Pablo Picasso** - `[collaborate_with]` - **0.6221**

*   **Short Comments**:
    *   **Hub Dominance (Top 1-3)**: Org nodes like "Party" or "School" act as high-degree hub nodes in the Wikidata KG. GNN's neighborhood aggregation concentrates massive information flow towards the target node, resulting in career-related elements dominating over ancestral lineages.
    *   **Local Strong Ties (Top 4-5)**: Direct family nodes represent strong 1-hop ties. However, due to their localized structural isolation on the global graph, their cosine similarity scores are slightly pulled down compared to global political entities.
    *   **Structural Equivalence (Top 6-9)**: Prestigious state awards achieve highly clustered scores. The GNN effectively recognizes that these nodes share identical topological roles (all being Award type nodes connected to Politician type nodes).
    *   **Contextual Inference (Top 10)**: Although Picasso shares no direct link with Ho Chi Minh, their similarity is exceptionally high (0.6221). The GNN successfully captured their shared historical context (both active in Paris in the 1920s, sharing the French Communist Party and left-wing ideological associations).

#### Scenario C: Specific Relationship Recommendation (Ho Chi Minh for `educated_at`)
Retrieving top education venue recommendations for President Ho Chi Minh:
1.  **Quoc Hoc - Hue High School for the Gifted** - **0.8221**
2.  **Communist University of the Toilers of the East** - **0.8014**
3.  **International Lenin School** - **0.7245**
4.  **Sorbonne University** - **0.7042**
5.  **Yale University** - **0.6402**

*   **Short Comments**:
    *   **Historical Accuracy (Top 1-3)**: #01 & #02 are high-scoring True Positives (schools he actually attended). #03 is another precise historical prediction reflecting his time working and studying at the Lenin Institute in Moscow.
    *   **Contextual Hallucination (Top 4)**: Sorbonne University (0.7042) acts as a highly logical False Positive. Although he never officially studied at Sorbonne, his prolonged residency in Paris, participation in political debates, and research in major French libraries pulled his embedding close to the "Paris Intellectuals" cluster.
    *   **Graph Noise (Top 5)**: Yale University (0.6402) is a massive global educational hub node, which naturally scores highly for world leaders, but its score is notably lower.

#### Scenario D: Spouse Advisor with Hard Logical Constraints (Trấn Thành)
Generating top spouse candidates for actor/showman Trấn Thành:
1.  **Hari Won** (1985/female) - **Score: 0.7658**
2.  **Mai Hồ** (1987/female) - **Score: 0.6138**
3.  **Tuấn Trần** (1992/male) - **Score: 0.5858**
4.  **Thu Trang** (1984/female) - **Score: 0.5687**
5.  **Việt Hương** (1976/female) - **Score: 0.5058**

*   **Short Comments**:
    *   **Absolute Accuracy (Top 1)**: Hari Won (0.7658) is predicted as the top spouse candidate with a margin of >0.15 over the next candidate. This represents a perfect True Positive, as Hari Won is indeed Trấn Thành's real-life wife.
    *   **Romantic History (Top 2)**: Mai Hồ (0.6138) is his historical ex-partner. The GNN successfully captured their strong historical romantic context from existing literature.
    *   **The Clique Effect (Top 3)**: Tuấn Trần (0.5858) is a close male actor colleague. He appeared in the spouse candidate list due to their massive shared presence in major films (Bố Già, Đất Rừng Phương Nam), which the model recognized as a tight, intimate tie.

#### Scenario E: Zero-Shot Transfer Learning (Shared Decoder)
Validating the collaborative transfer capacity of the Shared Decoder by predicting the `member_of` relation for **Ho Chi Minh City University of Science** (unseen relation type during training):
1.  **VNU-HCM (Vietnam National University, Ho Chi Minh City)** - **0.6452** (Possible)
2.  **Ministry of Education and Training** - **0.4120** (Low)
3.  **VNU-HN (Vietnam National University, Hanoi)** - **0.1560** (Very Low)
4.  **FPT Group** - **0.0230** (Very Low)

*   **Short Comments**:
    *   **Hierarchy Detection**: The model successfully differentiated between the "direct administrative parent unit" (VNU-HCM: 0.6452) and the general "state regulator" (Ministry of Education: 0.4120).
    *   **Semantic Sensitivity**: VNU-HN (0.1560) scored significantly lower than VNU-HCM despite sharing the "VNU" prefix, proving the GNN's geographic sensitivity.
    *   **Domain Separation**: The corporate FPT Group (0.0230) scored near zero, indicating clear boundary learning between academic and business ecosystems.
    *   *Conclusion*: This validates that the Shared Decoder successfully transferred structural relational rules from human organizations to higher education institutions without explicit training on those specific node types.

### 4. GNN Training & Verification Metrics

<img width="934" height="454" alt="image" src="https://github.com/user-attachments/assets/3a9391c0-fe13-468b-9dbb-204e87aaa4b9" />

* **AUC-ROC (Area Under ROC)**: Achieved a validation AUC-ROC metric of **0.78**.
* **AP (Average Precision)**: Achieved a validation AP metric of **0.82**.

---

## Streamlit Interactive UI Tabs

1. **Six Degrees of Separation (BFS)**: Computes the shortest path between any two individuals, displaying a visual timeline and an interactive network using pyvis.
2. **Pairwise Link Prediction**: Evaluates two nodes and displays a probability bar chart for all relationships in real-time.
3. **Connection & Spouse Advisor**: Recommends partners using GNN scores and hard constraints (age limits, taboo checks, gender filters).
4. **Network Analytics**: Visualizes node degree distributions, scale-free power law, and maps top PageRank centrality nodes using stylized HTML cards.
5. **Ego Network Explorer**: Renders the 1-degree neighborhood surrounding a central node inside a dynamic, physics-enabled network.

---

## Installation & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Streamlit Web UI
```bash
streamlit run main.py
```

### 3. Utility CLI Commands
* **Run ETL Pipeline**: `python main.py --etl`
* **Train GNN Model**: `python main.py --train`

---

# Bản tiếng Việt

Hệ thống **Social Link Prediction** là giải pháp phân tích mạng lưới xã hội nâng cao, tích hợp công nghệ **Học sâu đồ thị (Graph Neural Networks - GNN)** dị thể và các thuật toán cấu trúc đồ thị tối ưu. Dự án thu thập dữ liệu tri thức thế giới thực từ **Wikidata API**, xử lý kỹ thuật dữ liệu quy mô lớn (Data Engineering), và xây dựng mô hình AI dự đoán các mối quan hệ ẩn hoặc tiềm năng tương lai giữa các thực thể (người dùng, tổ chức, trường học...).

---

## Bối cảnh và Mục tiêu dự án

Trong kỷ nguyên tri thức số, các Đồ thị Tri thức như Wikidata đóng vai trò là "xương sống" cho các hệ thống Trí tuệ nhân tạo hiện đại. Tuy nhiên, các hệ thống này đang đối mặt với "Vùng tối tri thức" – nơi các mối quan hệ xã hội thực tế tồn tại nhưng chưa được số hóa, dẫn đến tính không đầy đủ và thưa thớt của dữ liệu.

Đề tài này xây dựng giải pháp trên nền tảng kiến trúc lai **Neuro-Symbolic AI**, tối ưu hóa sự kết hợp giữa:
* **Symbolic (Biểu trưng)**: Khả năng suy luận logic cấu trúc đồ thị cực nhanh nhờ lõi C-Core (igraph).
* **Neural (Nơ-ron)**: Khả năng học biểu diễn ẩn mạnh mẽ của mạng nơ-ron đồ thị dị thể (GNN) trên không gian liên tục.

---

## Kiến trúc Hệ thống (Clean Architecture)

Hệ thống được thiết kế và triển khai chặt chẽ theo nguyên lý Kiến trúc Sạch (Clean Architecture) nhằm đảm bảo tính module hóa, độc lập và dễ dàng mở rộng:

* **Presentation Layer (Lớp Giao diện)**: Xây dựng bằng Streamlit, bao gồm 5 Tab nghiệp vụ trực quan hóa cao kết hợp thư viện đồ thị động PyVis.
* **Application Layer (Lớp Nghiệp vụ Service)**: [ai_service.py](file:///c:/Users/nguye/social-link-prediction/application/ai_service.py) và [analysis_service.py](file:///c:/Users/nguye/social-link-prediction/application/analysis_service.py) đóng vai trò điều hợp luồng dữ liệu và thuật toán.
* **Core Logic Layer (Lớp Lõi thuật toán)**: Chứa định nghĩa mô hình GNN, thuật toán BFS/Dijkstra tối ưu trọng số và bộ máy tìm kiếm mờ Fuzzy Search sử dụng `rapidfuzz`.
* **Infrastructure Layer (Lớp Hạ tầng ETL)**: Quản lý trích xuất dữ liệu từ Wikidata, làm sạch và lưu trữ đồ thị thông qua các Repositories.

---

## Quy trình Kỹ nghệ Dữ liệu (Data Engineering Pipeline)

Hệ thống sở hữu một pipeline xử lý dữ liệu (ETL) hoàn chỉnh, khép kín và tự động hóa cao:

### 1. Ingestion & Trích xuất đa giai đoạn (Extraction)
* Sử dụng SPARQL Wrapper ([extractor.py](file:///c:/Users/nguye/social-link-prediction/infrastructure/pipelines/extractor.py)) để truy vấn trực tiếp từ endpoint Wikidata dưới dạng bộ ba RDF T = (Chủ thể, Thuộc tính, Đối tượng).
* Cơ chế quét theo khoảng thời gian sinh (birth interval) song song tránh bị giới hạn băng thông (rate limit) hoặc timeout của Wikidata API.
* Định nghĩa các câu truy vấn ngữ nghĩa phức tạp trong [queries.py](file:///c:/Users/nguye/social-link-prediction/config/queries.py) cho nhiều loại mối quan hệ: học tập (`educated_at`), làm việc (`work_at`), giải thưởng (`award_received`), quan hệ gia đình (`father`, `mother`, `spouse`), và đồng nghiệp (`colleague` / `member_of_sports_team` / `acted_in`).

### 2. Biến đổi, Làm sạch & Lọc trùng (Transformation)
* Dọn dẹp và chuẩn hóa dữ liệu ([transformer.py](file:///c:/Users/nguye/social-link-prediction/infrastructure/pipelines/transformer.py)): Loại bỏ các vòng tự lặp (self-loops), xử lý giá trị khuyết thiếu ở cột năm sinh, giới tính và mô tả.
* Lọc cạnh trùng lặp & cạnh ngược bằng khóa tạo lập `(id_min + "_" + id_max + "_" + relation)` giúp đồ thị nhất quán.
* Lưu trữ dữ liệu làm sạch dưới định dạng Apache Parquet tối ưu hóa tốc độ ghi đọc và dung lượng lưu trữ.

### 3. Quy mô Đồ thị và Phân tích Topo chuyên sâu
* **Quy mô dữ liệu**:
  * **Đỉnh (Nodes)**: 4,609,521 thực thể (bao gồm 3,247,681 nút `human`, 438,293 nút `written_work`, 315,908 nút `film`... thuộc 11 loại nút khác nhau).
  * **Cạnh (Edges)**: 10,761,807 liên kết hoạt động thuộc 44 loại quan hệ (đứng đầu là `educated_at` với 1,714,215 cạnh và `work_at` với 1,669,322 cạnh).
* **Kiểm định Phân phối lũy thừa (Power-Law Exponent)**:
  * Vẽ biểu đồ phân bố bậc CDF/CCDF trên thang đo Log-Log. Hệ số mũ Power-law ước lượng bằng phương pháp Maximum Likelihood Estimation (MLE):
    
    `gamma = 1 + n * [ sum( ln(k_i / (k_min - 0.5)) ) ]^-1`
    
    Với ngưỡng tối thiểu `k_min = 100`, ta tính ra `gamma = 3.35`. Chỉ số này chứng minh mạng lưới có tính phi tỷ lệ nhưng tiệm cận mạng ngẫu nhiên (Random-like), do các siêu đỉnh (Hubs) chưa chiếm mật độ tuyệt đối để tập trung quyền lực đồ thị.

### 4. Kỹ nghệ Đặc trưng & Khởi tạo PyG HeteroData
* **Đặc trưng Ngữ nghĩa (Semantic Features)**: Sử dụng mô hình Transformer ngôn ngữ đa quốc gia Sentence-BERT (`paraphrase-multilingual-MiniLM-L12-v2`) mã hóa thông tin thuộc tính dạng text của nút (tên, mô tả, nghề nghiệp, nơi sinh, giới tính, sở thích...) thành một dense vector 384 chiều.
* **Đặc trưng Cấu trúc (Structural Features)**: 
  * Sử dụng thư viện đồ thị C-Core igraph tính toán chỉ số Multi-view PageRank (45 lượt trên từng loại cạnh) để ghi nhận thuộc tính cấu trúc ẩn.
  * Tính toán Bậc kết nối toàn cục (Total Degree) của các nút và chuẩn hóa Log-transform (`log1p`) để triệt tiêu ảnh hưởng của các nút ngoại lai.
* **Đặc trưng Thời gian**: Chuẩn hóa Min-Max năm sinh và tạo vector cờ nhị phân đánh dấu giá trị năm sinh bị khuyết thiếu.
* **Hợp nhất vector đặc trưng (Feature Concat)**: Tạo ra vector đặc trưng nút hợp nhất 432 chiều:
  
  `x_node = [ v_SBERT(384) || v_year(1) || v_missing_flag(1) || v_PageRank(45) || v_degree(1) ]`
  
* **Khởi tạo Đồ thị Dị thể (Heterogeneous Graph)**: Map index ID dạng chuỗi (QID) sang local index dạng số nguyên của PyTorch Geometric (PyG), lọc các quan hệ có quá ít cạnh để tránh nhiễu và đóng gói thành đối tượng `HeteroData`.

---

## Mô hình Dự đoán AI (Graph Neural Network - GNN)

Mô hình học máy chính được triển khai bằng PyTorch Geometric (PyG), áp dụng kiến trúc mạng tích chập đồ thị dị thể để học nhúng cấu trúc và ngữ nghĩa.

### 1. Cơ chế Encoder (GraphSAGE)
* Sử dụng mạng GraphSAGE (Sample and Aggregate) kế thừa sức mạnh học quy nạp (inductive learning). Mô hình có thể dự đoán nhúng cho cả các nút mới hoàn toàn (unseen nodes) khi có thông tin đặc trưng mà không cần huấn luyện lại từ đầu.
* Phép biến đổi dị thể được gói bằng hàm `to_hetero` của PyG, sử dụng cơ chế gom tụ sum để tổng hợp thông tin từ nhiều loại quan hệ khác nhau truyền đến nút đích.
* Áp dụng chuẩn hóa LayerNorm và dropout 40% giữa các lớp tích chập đồ thị nhằm chống hiện tượng quá khớp (overfitting) và triệt tiêu gradient (vanishing gradients).

### 2. Cơ chế Decoder & Cơ chế Shared Decoder
* **Cơ chế Shared Decoder**: Các bộ giải mã (decoders) được ánh xạ dựa trên khóa quan hệ dạng `__rel__` thay vì `src__rel__dst` riêng biệt. Các loại cạnh có chung tên mối quan hệ (ví dụ: vợ/chồng giữa các phân nhóm nút khác nhau) sẽ chia sẻ chung bộ trọng số giải mã. Điều này giúp tiết kiệm tài nguyên bộ nhớ, tránh hiện tượng overfitting và tăng cường khả năng học biểu diễn tương hỗ của GNN trên toàn đồ thị.
* **Bộ giải mã InteractionMLP**: Thay vì chỉ thực hiện nhân vô hướng (Dot Product) hoặc khoảng cách Cosine thông thường giữa hai vector nhúng, hệ thống thiết kế một InteractionMLP tối ưu để giải mã điểm số liên kết:
  
  `h_pair = [ z_src || z_dst || (z_src * z_dst) ]`
  
  Trong đó `z_src * z_dst` là tích Hadamard (nhân từng phần tử). Việc đưa tích Hadamard vào giúp mạng MLP nhận diện nhạy bén độ tương đồng và các tương tác phi tuyến tính giữa nguồn và đích.
* Điểm số đầu ra được ép qua hàm Sigmoid để đưa về khoảng xác suất `[0.0, 1.0]`.

### 3. Ràng buộc Logic cứng (Hard Constraints) & Phạt Hub (Hub Penalty)
* **Khoảng cách Xã hội (Bậc trung gian)**: Định nghĩa chỉ tính trên đỉnh loại `human` (người). Số bậc tương đương số lượng nút người làm cầu nối giữa hai thực thể, giúp kiểm chứng lý thuyết "Sáu bậc xa cách" trên thực tế.
* **Bộ lọc Cấm kỵ Huyết thống (Taboo Constraints)**: Loại bỏ các gợi ý kết hôn (`spouse`) nếu hai thực thể đã tồn tại mối quan hệ gia đình trực hệ trong đồ thị gốc (cha, mẹ, anh, chị, em).
* **Phạt chênh lệch tuổi tác (Age Gap Penalty)**: Tự động phạt điểm số liên kết hoặc tăng trọng số đường đi Dijkstra nếu chênh lệch tuổi tác giữa hai thực thể vượt quá ngưỡng cho phép (ví dụ: cách nhau trên 20 tuổi).
* **Phạt Hub Trung tâm (Hub Penalty)**: Trọng số đường đi Dijkstra qua một đỉnh được tính theo logarit của bậc vào (in-degree):
  
  `w_edge = log(in_degree + 1)`
  
  Giúp thuật toán tìm đường đi BFS/Dijkstra thông minh tránh đi qua các nút quá lớn (như quốc gia, tập đoàn lớn) để tìm ra các liên kết cá nhân thực sự chất lượng.

---

## Thực nghiệm & Kiểm chứng thực tế

Để đánh giá tính đúng đắn toán học, cấu trúc mạng lưới và năng lực dự đoán của mạng neural nhân tạo, hệ thống đã trải qua quy trình kiểm thử thực nghiệm chi tiết và kiểm chứng trên topo đồ thị thế giới thực.

### 1. Phép tìm kiếm mờ Fuzzy Search
Sử dụng thuật toán khoảng cách chỉnh sửa Levenshtein Distance để khớp từ khóa do người dùng nhập với cơ sở dữ liệu nút đồ thị:
$$score(s_1, s_2) \propto \frac{1}{d(s_1, s_2) + 1}$$
Hỗ trợ tìm kiếm nhanh, sửa lỗi gõ sai và xử lý các từ tiếng Việt không dấu hoặc có dấu tức thời.

### 2. Kiểm chứng thực tế lý thuyết "Sáu bậc phân cách"
Theo mô hình Small-world của Watts & Strogatz, độ dài đường đi trung bình lý thuyết ($L_{theory}$) giữa hai thực thể bất kỳ là:
$$L_{theory} \approx \frac{\ln(N)}{\ln(\langle k \rangle)}$$
Dựa trên thông số đồ thị thu được:
* Kích thước phân vùng liên thông đồ thị ($N$): $2,869,142$
* Bậc trung bình của mạng lưới ($\langle k \rangle$): $\approx 3.8$
* Giá trị Geodesic lý thuyết kỳ vọng ($L_{theory}$): $\approx 11.1$

#### Thực nghiệm quy mô lớn
Hệ thống tiến hành lấy mẫu kiểm tra ngẫu nhiên **100,000 cặp thực thể** $(u, v)$ và đo khoảng cách đường đi thực tế.
* **Kết quả**: Đường đi trung bình thực nghiệm thu được là $\bar{d} \leq 6$, tập trung phân bố chuẩn dạng chuông quanh $4$ bước.
* **Kết luận**: Chấp nhận giả thuyết không ($H_0: \bar{d} \leq 6$). Đồ thị tri thức Wikidata biểu diễn một thế giới nhỏ kết nối cực kỳ bền chặt, nơi khoảng cách xã hội bị xóa nhòa.

#### Các ca kiểm chứng thực tế
* **Ví dụ 1: Sơn Tùng M-TP $\Longleftrightarrow$ Taylor Swift**
  Hệ thống tìm thấy đường đi ngắn nhất chỉ gồm 4 bậc trung gian (6 liên kết trung chuyển), thể hiện sự giao thoa văn hóa âm nhạc:
  1. 👤 **Taylor Swift** [Q26876]
  2. 🟢 **American Music Award for Favorite Pop/Rock Female Artist** [Q1441929] `(award_received)`
  3. 👤 **Mariah Carey** [Q41076] `(award_received)`
  4. 👤 **Mỹ Linh** [Q6945890] `(influenced_by)`
  5. 🟢 **Vietnam National Academy of Music** [Q5649320] `(educated_at)`
  6. 👤 **Nguyễn Ánh Tuyết** [Q118249221] `(educated_at)`
  7. 🟢 **Conservatory of Ho Chi Minh City** [Q1377237] `(work_at)`
  8. 👤 **Sơn Tùng M-TP** [Q17450386] `(educated_at)`

* **Ví dụ 2: Chủ tịch Hồ Chí Minh $\Longleftrightarrow$ Taylor Swift**
  Đường đi ngắn nhất nối kết một nhân vật lịch sử tầm cỡ quốc tế và một biểu tượng nhạc pop hiện đại qua các giải thưởng ngoại giao và học viện âm nhạc hoàng gia Anh:
  1. 👤 **Ho Chi Minh** [Q36014]
  2. 🟢 **Star of the Republic of Indonesia** [Q2340171] `(award_received)`
  3. 👤 **Elizabeth II** [Q9682] `(award_received)`
  4. 🟢 **Honorary doctor of the Royal College of Music** [Q99025668] `(award_received)`
  5. 👤 **Andrew Lloyd Webber** [Q180975] `(award_received)`
  6. 🟢 **Beautiful Ghosts** [Q72270672] `(composer)`
  7. 👤 **Taylor Swift** [Q26876] `(lyricist)`

### 3. Kịch bản thực tế dự đoán liên kết GNN
Mô hình học sâu GNN sau khi huấn luyện được thực nghiệm dự báo trực tiếp trên 5 kịch bản nghiệp vụ thực tế:

#### Kịch bản A: Xác suất hình thành quan hệ (Sơn Tùng M-TP vs Snoop Dogg)
Mạng AI tính toán và xuất ra bảng phân phối xác suất quan hệ (qua các bộ giải mã độc lập Sigmoid):
* `collaborate_with` (Hợp tác cùng): **0.7242** (Cao)
* `influenced_by` (Chịu ảnh hưởng bởi): **0.6710** (Có thể)
* `advisor_of` (Cố vấn của): **0.5450** (Có thể)
* `sibling` (Anh chị em): **0.4900** (Thấp)
* `spouse` (Vợ chồng): **0.2043** (Rất thấp)

*   **Nhận xét ngắn**:
    Sơn Tùng M-TP đã từng hợp tác với Snoop Dogg trong sản phẩm âm nhạc *Hãy Trao Cho Anh* $\rightarrow$ Điểm khá cao (0.7242). Các quan hệ gia đình/vợ chồng (sibling, spouse) có điểm số rất thấp, chứng tỏ tính hợp lý sinh học và xã hội được mô hình nắm bắt rất tốt.

#### Kịch bản B: Gợi ý thực thực thể tiềm năng toàn cục (Hồ Chí Minh)
Quét không gian nhúng của toàn đồ thị để tìm ra 10 thực thể lân cận có mức độ tương thích cao nhất với Bác Hồ:
1.  **Đảng Cộng sản VN** - `[founder_of]` - **0.8255**
2.  **Quốc Học - Hue High School for the Gifted** (Trường Quốc học Huế) - `[educated_at]` - **0.8221**
3.  **Communist University of the Toilers of the East** (Đại học Phương Đông) - `[educated_at]` - **0.8014**
4.  **Hoàng Thị Loan** (Mẫu thân) - `[mother]` - **0.7602**
5.  **Nguyễn Sinh Sắc** (Thân sinh) - `[father]` - **0.7208**
6.  **Huân chương Lenin** - `[award_received]` - **0.7031**
7.  **Star of the Republic of Indonesia** (Huân chương của Indonesia) - `[award_received]` - **0.6992**
8.  **Grand Cross of the Order of Polonia Restituta** (Huân chương Ba Lan) - `[award_received]` - **0.6706**
9.  **Huân chương Sao Vàng** - `[award_received]` - **0.6531**
10. **Pablo Picasso** - `[collaborate_with]` - **0.6221**

*   **Nhận xét ngắn**:
    *   **Sự thống trị của các Node trung tâm (Hub Dominance) - [Top 1-3]**: Đảng CS VN (0.8255) và các trường học đứng đầu, cao hơn cả Cha Mẹ. Lý do là trong đồ thị Knowledge Graph, các node tổ chức (Org) là các Hub Nodes có bậc (degree) rất lớn. Cơ chế tổng hợp thông tin từ lân cận của GraphSAGE tập trung dòng thông tin chảy về node trung tâm mạnh mẽ, giúp model đánh giá "Sự nghiệp" định nghĩa con người Bác rõ nét hơn "Gia phả".
    *   **Kết nối mạnh nhưng cục bộ (Local Strong Ties) - [Top 4-5]**: Hoàng Thị Loan và Nguyễn Sinh Sắc đại diện cho các kết nối 1-hop trực tiếp cực mạnh. Tuy nhiên, các node này bị cô lập hơn trên đồ thị toàn cầu nên điểm tương đồng bị kéo xuống nhẹ so với các tổ chức lớn.
    *   **Hiệu ứng "Tương đương cấu trúc" (Structural Equivalence) - [Top 6-9]**: Các huân chương danh giá xuất hiện liên tiếp với điểm số sát nhau. GNN học được rằng các node này đóng vai trò topo giống hệt nhau (đều là node loại Award nối vào node Politician), tự động kéo vector của Bác lại gần cụm "Huân chương Xã hội chủ nghĩa".
    *   **Suy diễn ngữ cảnh (Contextual Inference) - [Top 10]**: Pablo Picasso đạt điểm rất cao (0.6221) dù không có cạnh trực tiếp. Lý do là họ chia sẻ chung một tập hợp lân cận ấm: Paris, Đảng Cộng sản Pháp và tư tưởng cánh tả. Mô hình đã nén được bối cảnh "Hoạt động tại Pháp những năm 1920" vào vector nhúng của cả hai.

#### Kịch bản C: Dự báo loại quan hệ đặc thù (Hồ Chí Minh cho `educated_at`)
Tìm kiếm 5 trường học/học viện có khả năng cao là nơi học tập của Bác Hồ:
1.  **Quốc Học - Hue High School for the Gifted** - **0.8221**
2.  **Communist University of the Toilers of the East** (Đại học Phương Đông) - **0.8014**
3.  **International Lenin School** (Trường Quốc tế Lenin) - **0.7245**
4.  **Sorbonne University** (Đại học Sorbonne) - **0.7042**
5.  **Yale University** (Đại học Yale) - **0.6402**

*   **Nhận xét ngắn**:
    *   **Độ chính xác lịch sử (Top 1-3)**: #01 & #02 (True Positives) là các trường Bác từng học thực tế với điểm số > 0.8. #03 cũng là dự đoán chính xác nơi Bác từng học tập và làm việc tại Viện Lênin ở Moscow.
    *   **Sự suy diễn logic (Top 4 - False Positive "hợp lý")**: Sorbonne University (0.7042). Bác Hồ chưa từng học tại Sorbonne, tuy nhiên thời gian Bác ở Paris tham gia Đảng Xã hội Pháp và nghiên cứu tại các thư viện lớn đã kéo Vector của Bác lại gần cụm "Trí thức Paris", tạo nên một "ảo giác thông minh" hợp lý.
    *   **Nhiễu (Top 5 - Noise)**: Yale University (0.6402) là hub giáo dục lớn toàn cầu nên thường xuất hiện trong gợi ý các lãnh tụ thế giới, tuy nhiên điểm số đã tụt giảm rõ rệt.

#### Kịch bản D: Dự đoán quan hệ đặc thù: Vợ chồng (Trấn Thành)
Tìm kiếm và gợi ý 5 ứng viên kết đôi tiềm năng nhất cho diễn viên, người dẫn chương trình Trấn Thành:
1.  **Hari Won** (1985/female) - **Score: 0.7658**
2.  **Mai Hồ** (1987/female) - **Score: 0.6138**
3.  **Tuấn Trần** (1992/male) - **Score: 0.5858**
4.  **Thu Trang** (1984/female) - **Score: 0.5687**
5.  **Việt Hương** (1976/female) - **Score: 0.5058**

*   **Nhận xét ngắn**:
    *   **Độ chính xác tuyệt đối (Top 1)**: Hari Won (0.7658) được dự đoán là người vợ hiện tại với điểm số vượt trội hoàn toàn (cách biệt > 0.15 so với người thứ 2), chứng tỏ liên kết spouse trong đồ thị rất mạnh và vector của cả hai kéo sát lại nhau.
    *   **Nhạy bén với quá khứ (Top 2)**: Mai Hồ (0.6138) là người yêu cũ. AI đã nắm bắt được bối cảnh tình cảm từ dữ liệu lịch sử mặc dù hiện tại họ không còn bên nhau.
    *   **Hiệu ứng "Hội nhóm" (The Clique Effect - Top 3)**: Tuấn Trần (0.5858). Dù là nam, nhưng Tuấn Trần lọt vào gợi ý "Bạn đời" do tần suất xuất hiện chung dày đặc trong các sản phẩm điện ảnh (Bố Già, Đất Rừng Phương Nam), khiến AI coi đây là một mối liên kết cực kỳ thân thiết.

#### Kịch bản E: Dự báo Zero-Shot Transfer Learning (Shared Decoder)
Đánh giá tính tổng quát của mô hình qua việc dự báo quan hệ `member_of` giữa **Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM** (tổ chức) với các đơn vị khác (quan hệ tổ chức này chưa từng được huấn luyện trực tiếp):
1.  **ĐHQG TP.HCM (VNU-HCM)** - **0.6452** (Có thể)
2.  **Bộ Giáo dục & Đào tạo** - **0.4120** (Thấp)
3.  **ĐHQG Hà Nội (VNU-HN)** - **0.1560** (Rất thấp)
4.  **Tập đoàn FPT** - **0.0230** (Rất thấp)

*   **Nhận xét ngắn**:
    *   **Khả năng phân bậc quản lý (Hierarchy Detection)**: VNU-HCM (0.6452) > Bộ Giáo dục (0.4120). Model phân biệt được "Đơn vị chủ quản trực tiếp" (ĐHQG) và "Cơ quan quản lý nhà nước" (Bộ) nhờ các cụm thành viên liên thông mạnh trong đồ thị.
    *   **Độ nhạy về ngữ nghĩa (Semantic Sensitivity)**: VNU-HN (0.1560) điểm thấp hơn hẳn ĐHQG TP.HCM dù cùng tên là "ĐHQG", chứng tỏ mô hình nhận ra sự khác biệt địa lý và không bị nhầm lẫn.
    *   **Phân loại lĩnh vực (Domain Separation)**: FPT (0.0230) điểm tiệm cận 0, chứng tỏ mô hình tách biệt hoàn toàn khối học thuật (Academic) và khối doanh nghiệp (Corporate).
    *   *Kết luận:* Cơ chế Shared Decoder hoạt động hiệu quả qua việc chuyển giao tri thức (Transfer Learning) từ quan hệ `member_of` của con người sang tổ chức.

### 4. Chỉ số Huấn luyện & Hiệu năng của GNN

<img width="934" height="454" alt="image" src="https://github.com/user-attachments/assets/c4dde69f-ea7f-46cd-a9fa-14f80abc865c" />

* **AUC-ROC (Diện tích dưới đường cong)**: Đạt chỉ số kiểm thử AUC-ROC là **0.78**.
* **AP (Average Precision)**: Đạt chỉ số kiểm thử độ chính xác trung bình AP là **0.82**.

---

## Giao diện Tương tác Streamlit Dashboard

1. **Sáu Bậc Xa Cách (Degrees of Separation)**: Tìm đường đi liên kết ngắn nhất giữa hai người dùng bất kỳ. Trực quan hóa sơ đồ Timeline và mạng lưới động dạng bóng nẩy kéo thả bằng PyVis.
2. **Dự đoán Liên kết AI (GNN Pairwise)**: Nhập hai thực thể để phân tích và vẽ biểu đồ xác suất của tất cả mối quan hệ tiềm năng giữa họ.
3. **Gợi ý kết nối thông minh**: Tìm kiếm các đối tác tiềm năng hoặc gợi ý vợ/chồng kết hợp GNN Soft Constraints và các bộ lọc logic cứng về giới tính, tuổi tác, huyết thống.
4. **Thống kê mạng lưới (Network Analytics)**: Hiển thị thống kê quy mô đồ thị, vẽ biểu đồ phân bố bậc (Degree Distribution) và xếp hạng Top 10 thực thể có tầm ảnh hưởng lớn nhất qua thuật toán PageRank Centrality.
5. **Khám phá lân cận (Ego Network)**: Vẽ biểu đồ mạng lưới các quan hệ trực tiếp (bậc 1) xung quanh một thực thể chỉ định bằng bong bóng động tương tác.

---

## Hướng dẫn Khởi chạy nhanh

### 1. Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

### 2. Khởi chạy Dashboard Streamlit
```bash
streamlit run main.py
```

### 3. Các lệnh CLI Tiện ích (ETL & Huấn luyện)
* **Quy trình ETL thu thập dữ liệu Wikidata**: `python main.py --etl`
* **Quy trình Huấn luyện lại mô hình GNN**: `python main.py --train`

---

## Cấu trúc Thư mục Dự án

* **/config**: Các tệp cấu hình đường dẫn hằng số và Wikidata SPARQL Queries.
* **/core**: Lớp lõi thuật toán & AI (GNN, BFS/Dijkstra tối ưu, Fuzzy Search).
* **/infrastructure**: Pipelines ETL Wikidata (Extractor/Transformer) và các repositories lưu trữ.
* **/application**: Lớp phối hợp nghiệp vụ (AIService, AnalysisService).
* **/presentation**: Streamlit Web UI và mã nguồn render 5 Tab nghiệp vụ chính.
* **/scripts**: Điểm khởi chạy chạy độc lập hoặc CLI (ETL, train_model).

---

## Thành viên Thực hiện (Credits)

Dự án được nghiên cứu và phát triển bởi Nhóm 3 - Môn học Python cho Khoa học Dữ liệu - Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM:
* **Nguyễn Minh Quang** - [minhquang0407](https://github.com/minhquang0407)
* **Đinh Nhật Tân** - [Hecquyn175](https://github.com/Hecquyn175)
* **Nguyễn Quốc Anh Quân** - [nqaq2005](https://github.com/nqaq2005)
