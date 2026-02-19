
## Brain Cell Atlas: for language genes

An interactive visualization platform for single-cell RNA-seq analysis of the Human Middle Temporal Gyrus (MTG). This project runs as a split-stack application with a high-performance **FastAPI** backend and a responsive **Streamlit** frontend.

## Project Architecture

- **Backend (`deploy/api`)**: FastAPI server that handles data processing, serves the machine learning model, and manages the `.h5ad` dataset. It provides REST endpoints for gene expression, dimensionality reduction coordinates (UMAP), and cell type predictions.
- **Frontend (`deploy/dashboard`)**: Streamlit dashboard that consumes the API to create interactive visualizations (UMAP plots, dot plots, heatmaps) without loading heavy datasets directly into the user's browser.
- **Data**: The application uses a dataset (`human_mtg_brain_atlas_final.h5ad`) and a pre-trained classifier (`cell_type_classifier.pkl`), which are automatically downloaded from a secure source upon the first launch of the API.

##  Reproducible Workflow

Follow these steps to set up and run the application locally. This workflow ensures all dependencies and environment variables are correctly configured.

### Prerequisites
- Linux, macOS, or WSL
- Python 3.9 or higher
- `pip` (Python package installer)

---

### Step 1: Clone the Repository


git clone <your-repo-url>
cd brain-cell-atlas

### Step 2: Set up the Backend (API)



1.  Navigate to the API directory
2.  Create and activate a virtual environment
3.  Install dependencies present in requirements.txt
4.  Start the API server
       uvicorn app:app --host 0.0.0.0 --port 8000 --reload

    
    > **Note:** On the first run, the API will automatically download the required dataset (~500MB) and model files. Watch the terminal logs for "Download complete" and "API Ready!".

    *Keep this terminal window open.*

### Step 3: Set up the Frontend (Dashboard)

Open a new terminal window/tab to run the dashboard.
Run the Streamlit app:
    
    command: streamlit run app.py


    The dashboard should automatically open in your browser at `http://localhost:8501`.

---

##  Usage Guide

### Dashboard (`http://localhost:8501`)
- **The Atlas**: Overview of the dataset, UMAP projection of all cells, and cell type distribution.
- **Gene Inspector**: Search for specific genes (e.g., `SLC17A7`, `GAD1`) to visualize their expression patterns across cell types.
- **Model Performance**: Metrics for the cell type classifier, including confusion matrix and feature importance.

### API Documentation (`http://localhost:8000/docs`)
- Access the interactive Swagger UI to test API endpoints directly.
- Monitor the health of the service at `GET /healthz`.

## Project Structure

```text
.
├── deploy/
│   ├── api/                 # FastAPI Backend
│   │   ├── app.py           # Main application entry point
│   │   └── requirements.txt # Backend dependencies
│   └── dashboard/           # Streamlit Frontend
│       ├── app.py           # Dashboard interface code
│       └── requirements.txt # Frontend dependencies
├── notebooks/               # Jupyter notebooks for analysis and model training
├── data/                    # Local data storage (populated on run)
└── results/                 # Model artifacts and outputs
```

##  Development

To modify the codebase:
1.  **API Changes**: Edit `deploy/api/app.py`. The server will auto-reload if you used the `--reload` flag.
2.  **Dashboard Changes**: Edit `deploy/dashboard/app.py`. Refresh the browser to see changes.
