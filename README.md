# Social Link Prediction

This README provides a technical overview of the Social Link Prediction project, detailing its architecture, features, and performance metrics. It is designed to be a professional guide for developers and researchers to understand the system's capabilities.

### Project Overview

The Social Link Prediction system leverages Graph Neural Networks (GNN) to analyze social network topologies and node attributes. Its primary objective is to predict the probability of future or hidden connections between entities within a network.

### System Architecture

The project follows a modular, layered architecture to ensure maintainability and scalability:

* **Infrastructure Layer**: Handles data extraction, transformation (ETL), and persistence through specialized repositories for graphs, features, and models.
* **Core Logic Layer**: Contains the fundamental algorithms, including Breadth-First Search (BFS) for connectivity analysis, fuzzy search for entity matching, and GNN architectures (GraphSAGE/GCN).
* **Application Layer**: Orchestrates AI and analysis services, bridging the gap between raw data processing and user-facing requirements.
* **Presentation Layer**: A Streamlit-based dashboard providing real-time visualization and interaction with the prediction models.

### Key Features

* **Graph Deep Learning**: Implementation of inductive learning using GNNs to handle dynamic graphs and unseen nodes.
* **Automated ETL Pipeline**: Standardized process for ingesting raw data and converting it into graph-compatible formats.
* **Structural Analysis**: Traditional graph metrics integrated with modern deep learning for hybrid link prediction.
* **Interactive Dashboard**: Real-time prediction interface with visualization of node neighborhoods and connection probabilities.

### Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt

```


2. Launch the application:
```bash
streamlit run main.py

```



### Training Results

The following metrics represent the model's performance on the validation dataset:

| Metric | Value |
| --- | --- |
| Accuracy | 0.76 |
| AP | 0.84 |
| AUC-ROC | 0.76 |

**Training Curves**

<img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/c2ea2d05-57e5-4daf-bb1a-0cb4f2037ec0" />


### Demo Graph Visualization

The system evaluates node similarity and structural proximity to predict links. Below is a text-based representation of a predicted connection:

```text
       [ User A ] ----------- [ User B ]
           |                     |
           |                     |
       [ User C ] - - - - - - [ User D ]
           |          ^          |
           |   (Link Predicted)  |
       [ User E ] ----------- [ User F ]

```

**Technical Logic**: In this scenario, while User C and User D are not currently connected, the model identifies a high likelihood of connection (e.g., 0.88) based on their shared neighbors (A, B, E, F) and similar node embeddings.

### Directory Structure

* **/core/ai**: Model definitions (GNN), training logic, and data processors.
* **/infrastructure**: Database repositories and ETL pipeline components (Extractor/Transformer).
* **/presentation**: Streamlit application files and UI components.
* **/scripts**: Entry points for model training and ETL execution.
* **/tools**: Utility scripts for data cleaning and batch processing.
  
### Credits
* Developed by:
* **Nguyen Minh Quang** - University of Science, VNU. https://github.com/minhquang0407
* **Dinh Nhat Tan** - University of Science, VNU. https://github.com/Hecquyn175
* **Nguyen Quoc Anh Quan** - University of Science, VNU. https://github.com/nqaq2005 
---
