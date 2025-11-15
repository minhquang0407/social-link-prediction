# Wikidata Relationship Explorer and Predictor

This project is a Python-based tool for extracting, analyzing, and predicting relationships between entities from Wikidata. It uses a combination of data pipeline engineering, graph analytics, and machine learning to build a comprehensive relationship graph and uncover potential connections.

## Key Features

*   **Data Extraction:** Fetches data from Wikidata using SPARQL queries.
*   **Graph Construction:** Builds a `networkx` graph from the extracted data.
*   **Graph Analytics:** Provides a suite of tools for analyzing the graph, including:
    *   Pathfinding between two entities.
    *   Ego network analysis.
    *   Calculation of various network statistics (centrality, community detection, etc.).
*   **Link Prediction:** Utilizes a RandomForest model to predict potential new relationships in the graph.
*   **Explainable AI (XAI):** Provides insights into the features that are most important for the model's predictions.

## Project Structure

```
.
├── data_output/
│   ├── raw/
│   │   └── ... (raw JSON data from Wikidata)
│   └── full_graph.gpickle
├── data_pipeline/
│   ├── __init__.py
│   ├── extractor.py
│   └── transformer.py
├── models/
│   └── link_prediction_model.pkl
├── src/
│   ├── __init__.py
│   ├── ai_model.py
│   ├── analytics_engine.py
│   └── app.py
├── .gitignore
├── README.md
└── requirements.txt
```

### Directory Descriptions

*   **`data_output/`**: This directory is the default location for all generated data.
    *   `data_output/raw/`:  Stores the raw data extracted from Wikidata in JSON format.
    *   `data_output/full_graph.gpickle`: The final processed `networkx` graph object.
*   **`data_pipeline/`**: Contains the ETL (Extract, Transform, Load) scripts.
    *   `extractor.py`:  Defines the `WikidataExtractor` class, which is responsible for fetching data from Wikidata via SPARQL queries.
    *   `transformer.py`: Defines the `GraphTransformer` class, which takes the raw data and transforms it into a `networkx` graph.
*   **`models/`**:  This directory is used to store trained machine learning models.
*   **`src/`**:  Contains the core logic of the application.
    *   `ai_model.py`: Defines the `AIModel` class, which handles everything related to machine learning, including training, prediction, and saving/loading models.
    *   `analytics_engine.py`: Defines the `AnalyticsEngine` class, which provides various graph analysis functionalities.
    *   `app.py`:  (Currently empty) Intended to be the main entry point for a future interactive application.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The project is designed to be run as a series of scripts. Here is a typical workflow:

### 1. Data Extraction

First, you need to fetch the data from Wikidata. This is done using the `WikidataExtractor` class in `data_pipeline/extractor.py`.

*   **Define your SPARQL queries:** You will need to create a dictionary of SPARQL queries to fetch the desired relationships.
*   **Run the extractor:**

```python
# Example usage of the extractor
from data_pipeline.extractor import WikidataExtractor

# Define your SPARQL queries
# (This is just an example, you should create your own queries)
relationship_queries = {
    "family": "SELECT ?person ?spouse WHERE { ... }",
    "education": "SELECT ?person ?school WHERE { ... }",
}

# Initialize the extractor
extractor = WikidataExtractor(user_agent="YourAppName/1.0")

# Fetch the data and save it to the data_output/raw directory
extractor.fetch_all_relationships(relationship_queries, "data_output/raw")

```

### 2. Graph Transformation

Once you have the raw data, you can build the graph using the `GraphTransformer` class in `data_pipeline/transformer.py`.

```python
# Example usage of the transformer
from data_pipeline.transformer import GraphTransformer

# Define the paths to your raw data files
raw_files_dict = {
    "family": "data_output/raw/family.json",
    "education": "data_output/raw/education.json",
    # Add all your other raw data files here
}

# Initialize the transformer
transformer = GraphTransformer()

# Build the graph
G_full = transformer.build_full_graph(raw_files_dict)

# Save the graph
transformer.save_graph("data_output/full_graph.gpickle")
```

### 3. Graph Analysis

With the graph built, you can perform various analyses using the `AnalyticsEngine` class in `src/analytics_engine.py`.

```python
import networkx as nx
from src.analytics_engine import AnalyticsEngine

# Load the graph
G_full = nx.read_gpickle("data_output/full_graph.gpickle")

# Initialize the analytics engine
engine = AnalyticsEngine(G_full)

# Example: Find a path between two people
path = engine.find_path("Person A", "Person B")
print(f"The path between Person A and Person B is: {path}")

# Example: Get the ego network of a person
ego_network = engine.get_ego_network("Person A")
print(f"The ego network of Person A has {len(ego_network.nodes)} nodes.")

# Example: Calculate offline statistics
# (This may take a long time to run)
engine.calculate_offline_stats()
print("Offline statistics calculated:", engine.analytics_results)
```

### 4. AI Model Training and Prediction

Finally, you can train and use the AI model for link prediction using the `AIModel` class in `src/ai_model.py`.

```python
import networkx as nx
from src.ai_model import AIModel

# Load the graph
G_full = nx.read_gpickle("data_output/full_graph.gpickle")

# Initialize the AI model
ai = AIModel(G_full)

# Create training data
ai.create_training_data()

# Train the model
report = ai.train()
print(report)

# Save the trained model
ai.save_model("models/link_prediction_model.pkl")

# Load a pre-trained model
# ai.load_model("models/link_prediction_model.pkl")

# Predict top partners for a person
top_partners = ai.predict_top_partners("person_id_here")
print(f"Top 10 predicted partners: {top_partners}")

```
