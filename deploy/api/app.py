"""
Brain Cell Atlas API - FastAPI Backend
=====================================
Production-grade REST API for single-cell RNA-seq analysis
of the Human Middle Temporal Gyrus (MTG) dataset.

Endpoints:
- GET /           : Health check
- GET /genes      : List all available genes
- GET /cell_types : Cell type distribution
- GET /umap_data  : UMAP coordinates for all cells
- GET /gene_expression/{gene_symbol} : Mean expression by cell type
- POST /predict_random : Predict cell type for random cell
- GET /model_performance : Pre-computed classifier metrics
- GET /pc_importance : PCA feature importance
- GET /language_genes_analysis : Language gene expression
- GET /gene_umap_overlay/{gene_symbol} : UMAP with expression colors
- GET /marker_genes_table : Top markers per cell type
- GET /gene_dotplot/{gene_symbol} : Dot plot data
- GET /qc_dashboard : Quality control metrics
- GET /pca_variance : PCA variance explained
- GET /cell_type_profile/{cell_type} : Cell type details
"""

import logging
import requests
from pathlib import Path
import os

def download_from_gdrive(file_id: str, destination: str):
    """Download file from Google Drive if it doesn't exist locally."""
    if os.path.exists(destination):
        logger.info(f"✅ Using cached file: {destination}")
        return
    
    logger.info(f"📥 Downloading from Google Drive to {destination}...")
    
    # Create directory if needed
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Google Drive direct download URL
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    session = requests.Session()
    response = session.get(url, stream=True)
    
    # Handle large file confirmation
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            params = {'id': file_id, 'confirm': value}
            response = session.get(url, params=params, stream=True)
            break
    
    # Save file
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    
    logger.info(f"✅ Download complete: {destination}")

# =============================================================================
# App Lifespan - Resource Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load resources once at startup, clean up on shutdown.
    Downloads large data files from Google Drive if needed.
    """
    global adata, model, gene_set
    
    logger.info("🚀 Starting Brain Cell Atlas API...")
    
    try:
        # ------------------------------------------------------------------
        # GOOGLE DRIVE CONFIGURATION
        # Replace these IDs with your actual Google Drive File IDs
        # ------------------------------------------------------------------
        DATA_FILE_ID = "data/human_mtg_brain_atlas_final.h5ad"
        MODEL_FILE_ID = "/mnt/c/Users/sneha/BrainCellAtlas/results/models/cell_type_classifier.pkl"
        # ------------------------------------------------------------------
        
        # Local cache directory (ephemeral on Render)
        cache_dir = Path("/tmp/brain_atlas_cache")
        data_path = cache_dir / "human_mtg_brain_atlas_final.h5ad"
        model_path = cache_dir / "cell_type_classifier.pkl"
        
        # Download files if they don't exist
        # Only attempt download if IDs are set, otherwise fall back to local relative path
        if "PUT_YOUR" not in DATA_FILE_ID:
            download_from_gdrive(DATA_FILE_ID, str(data_path))
            download_from_gdrive(MODEL_FILE_ID, str(model_path))
            load_path_data = str(data_path)
            load_path_model = str(model_path)
        else:
            # Fallback for local development if files are present
            logger.warning("⚠️ Google Drive IDs not set. Checking local paths...")
            load_path_data = "../../data/human_mtg_brain_atlas_final.h5ad"
            load_path_model = "../../results/models/cell_type_classifier.pkl"

        logger.info(f"📂 Loading data from: {load_path_data}")
        adata = sc.read_h5ad(load_path_data)
        
        logger.info(f"🤖 Loading model from: {load_path_model}")
        model = joblib.load(load_path_model)
        
        # Create fast lookup set for gene validation (O(1) search)
        gene_set = set(adata.var_names.str.upper())
        
        logger.info(f"✅ Loaded {adata.n_obs:,} cells, {adata.n_vars:,} genes")
        logger.info(f"✅ Cell types: {adata.obs['cell_type'].unique().tolist()}")
        logger.info("🟢 API Ready!")
        
        yield
        
    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        raise RuntimeError(f"Missing required file: {e}")
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise
    finally:
        logger.info("🛑 Shutting down API...")


# =============================================================================
# FastAPI App Configuration
# =============================================================================

app = FastAPI(
    title="Brain Cell Atlas API",
    description="REST API for Human MTG single-cell RNA-seq analysis",
    version="1.0.0",
    lifespan=lifespan
)

# CRITICAL: Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Existing Endpoints
# =============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """API health check endpoint."""
    try:
        return HealthResponse(
            status="online",
            n_cells=int(adata.n_obs),
            n_genes=int(adata.n_vars),
            cell_types=[str(ct) for ct in adata.obs['cell_type'].unique() if pd.notna(ct)]
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/genes", response_model=GeneListResponse)
async def get_genes():
    """Return list of all available gene symbols."""
    try:
        return GeneListResponse(
            genes=adata.var_names.tolist(),
            total=int(adata.n_vars)
        )
    except Exception as e:
        logger.error(f"Failed to get genes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cell_types", response_model=CellTypeDistResponse)
async def get_cell_types():
    """Return cell type distribution with counts and percentages."""
    try:
        counts = adata.obs['cell_type'].value_counts()
        total = len(adata)
        
        distribution = [
            CellTypeDistItem(
                cell_type=str(ct),
                count=int(count),
                percentage=round(count / total * 100, 2)
            )
            for ct, count in counts.items()
        ]
        
        return CellTypeDistResponse(
            distribution=distribution,
            total_cells=total
        )
    except Exception as e:
        logger.error(f"Failed to get cell types: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/umap_data", response_model=List[UMAPPoint])
async def get_umap_data():
    """Return UMAP coordinates for all cells."""
    try:
        umap_coords = adata.obsm['X_umap']
        cell_types = adata.obs['cell_type'].values
        
        data = [
            UMAPPoint(
                x=round(float(umap_coords[i, 0]), 4),
                y=round(float(umap_coords[i, 1]), 4),
                cell_type=str(cell_types[i])
            )
            for i in range(len(adata))
        ]
        
        return data
    except Exception as e:
        logger.error(f"Failed to get UMAP data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gene_expression/{gene_symbol}", response_model=GeneExpressionResponse)
async def get_gene_expression(gene_symbol: str):
    """Return mean expression of a gene across all cell types."""
    try:
        gene_upper = gene_symbol.upper()
        
        if gene_upper not in gene_set:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Gene not found",
                    "requested": gene_symbol,
                    "suggestion": "Use GET /genes to see available genes"
                }
            )
        
        matching_genes = [g for g in adata.var_names if g.upper() == gene_upper]
        gene_name = matching_genes[0]
        gene_idx = adata.var_names.get_loc(gene_name)
        
        expression_data = []
        for cell_type in adata.obs['cell_type'].unique():
            if pd.isna(cell_type):
                continue
            mask = adata.obs['cell_type'] == cell_type
            
            subset = adata[mask, gene_idx].X
            if hasattr(subset, 'toarray'):
                subset = subset.toarray()
            mean_expr = float(np.nanmean(subset))
            if np.isnan(mean_expr):
                mean_expr = 0.0
            
            expression_data.append(
                GeneExpressionItem(
                    cell_type=str(cell_type),
                    mean_expression=round(mean_expr, 4)
                )
            )
        
        expression_data.sort(key=lambda x: x.mean_expression, reverse=True)
        
        return GeneExpressionResponse(
            gene=gene_name,
            expression=expression_data
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get gene expression for {gene_symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_random", response_model=PredictionResponse)
async def predict_random():
    """Select a random cell and predict its cell type."""
    try:
        cell_idx = np.random.randint(0, adata.n_obs)
        pca_features = adata.obsm['X_pca'][cell_idx, :50]
        true_label = str(adata.obs['cell_type'].iloc[cell_idx])
        predicted_label = model.predict([pca_features])[0]
        
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba([pca_features])[0]
            confidence = float(np.max(probas))
            
            top_3_indices = np.argsort(probas)[-3:][::-1]
            top_3 = {
                str(model.classes_[idx]): round(float(probas[idx]), 4)
                for idx in top_3_indices
            }
        else:
            confidence = 1.0
            top_3 = {str(predicted_label): 1.0}
        
        return PredictionResponse(
            cell_index=int(cell_idx),
            true_label=true_label,
            predicted_label=str(predicted_label),
            confidence=confidence,
            top_3_predictions=top_3,
            pca_features=[round(float(x), 4) for x in pca_features[:10]]
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# NEW Endpoints
# =============================================================================

@app.get("/model_performance")
async def get_model_performance():
    """Return pre-computed model performance metrics."""
    return {
        "overall_accuracy": 0.985,
        "test_size": 1498,
        "per_class_metrics": CLASS_METRICS,
        "confusion_matrix": CONFUSION_MATRIX,
        "class_order": CLASS_ORDER
    }


@app.get("/pc_importance")
async def get_pc_importance():
    """Return feature importance from the trained model."""
    try:
        importances = model.feature_importances_
        
        all_importances = [
            {"pc": i, "importance": float(importances[i])}
            for i in range(len(importances))
        ]
        
        top_10 = sorted(
            all_importances,
            key=lambda x: x['importance'],
            reverse=True
        )[:10]
        
        return {
            "importances": all_importances,
            "top_10": top_10
        }
    except Exception as e:
        logger.error(f"Failed to get PC importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/language_genes_analysis")
async def get_language_genes_analysis():
    """Return analysis of language-related genes."""
    try:
        results = []
        
        # Get raw gene names for lookup
        raw_var_names = adata.raw.var_names if adata.raw is not None else adata.var_names
        raw_var_names_upper = {g.upper(): g for g in raw_var_names}
        
        for gene in LANGUAGE_GENES:
            gene_upper = gene.upper()
            
            if gene_upper not in raw_var_names_upper:
                continue
            
            actual_gene_name = raw_var_names_upper[gene_upper]
            gene_idx = raw_var_names.get_loc(actual_gene_name)
            
            by_celltype = []
            for ct in adata.obs['cell_type'].unique():
                if pd.isna(ct):
                    continue
                mask = adata.obs['cell_type'] == ct
                
                if adata.raw is not None:
                    expr = adata.raw[mask, gene_idx].X
                else:
                    expr = adata[mask, gene_idx].X
                    
                if hasattr(expr, 'toarray'):
                    expr = expr.toarray().flatten()
                
                mean_expr = float(np.nanmean(expr))
                if np.isnan(mean_expr):
                    mean_expr = 0.0
                
                by_celltype.append({
                    "cell_type": str(ct),
                    "mean_expression": round(mean_expr, 4)
                })
            
            results.append({
                "gene": actual_gene_name,
                "expression_by_celltype": by_celltype
            })
        
        return {
            "language_genes": results,
            "comparison": {
                "pca_accuracy": 0.985,
                "language_only_accuracy": 0.347,
                "n_features_pca": 50,
                "n_features_language": len(LANGUAGE_GENES)
            }
        }
    except Exception as e:
        logger.error(f"Failed to get language genes analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gene_umap_overlay/{gene_symbol}")
async def gene_umap_overlay(gene_symbol: str):
    """Return UMAP coordinates with per-cell gene expression."""
    try:
        gene_upper = gene_symbol.upper()
        
        raw_var_names = adata.raw.var_names if adata.raw is not None else adata.var_names
        matching_genes = [g for g in raw_var_names if g.upper() == gene_upper]
        
        if not matching_genes:
            raise HTTPException(404, f"Gene '{gene_symbol}' not found")
        
        gene_name = matching_genes[0]
        gene_idx = raw_var_names.get_loc(gene_name)
        
        if adata.raw is not None:
            expr = adata.raw[:, gene_idx].X
        else:
            expr = adata[:, gene_idx].X
            
        if hasattr(expr, 'toarray'):
            expr = expr.toarray().flatten()
        
        umap_coords = adata.obsm['X_umap']
        
        data = []
        for i in range(len(adata)):
            expr_val = float(expr[i])
            if np.isnan(expr_val):
                expr_val = 0.0
            
            data.append({
                "x": round(float(umap_coords[i, 0]), 4),
                "y": round(float(umap_coords[i, 1]), 4),
                "expression": round(expr_val, 4),
                "cell_type": str(adata.obs['cell_type'].iloc[i])
            })
        
        return {
            "gene": gene_name,
            "data": data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get gene UMAP overlay for {gene_symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/marker_genes_table")
async def get_marker_genes_table():
    """Return top marker genes per cell type from rank_genes_groups."""
    try:
        result = adata.uns['rank_genes_groups']
        
        markers = {}
        for cluster_id, cell_type in CLUSTER_ANNOTATIONS.items():
            try:
                genes = result['names'][cluster_id][:10]
                scores = result['scores'][cluster_id][:10]
                
                markers[cell_type] = [
                    {
                        "gene": str(genes[i]),
                        "score": round(float(scores[i]), 2),
                        "rank": i + 1
                    }
                    for i in range(len(genes))
                ]
            except (KeyError, IndexError) as e:
                logger.warning(f"Could not get markers for {cell_type}: {e}")
                markers[cell_type] = []
        
        return markers
    except Exception as e:
        logger.error(f"Failed to get marker genes table: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gene_dotplot/{gene_symbol}")
async def gene_dotplot(gene_symbol: str):
    """Return dot plot data: % expressing and mean expression per cell type."""
    try:
        gene_upper = gene_symbol.upper()
        
        raw_var_names = adata.raw.var_names if adata.raw is not None else adata.var_names
        matching_genes = [g for g in raw_var_names if g.upper() == gene_upper]
        
        if not matching_genes:
            raise HTTPException(404, f"Gene '{gene_symbol}' not found")
        
        gene_name = matching_genes[0]
        gene_idx = raw_var_names.get_loc(gene_name)
        
        results = []
        for ct in adata.obs['cell_type'].unique():
            if pd.isna(ct):
                continue
            
            mask = adata.obs['cell_type'] == ct
            
            if adata.raw is not None:
                expr = adata.raw[mask, gene_idx].X
            else:
                expr = adata[mask, gene_idx].X
                
            if hasattr(expr, 'toarray'):
                expr = expr.toarray().flatten()
            
            # % expressing (expr > 0)
            pct = float((expr > 0).sum() / len(expr) * 100)
            
            # Mean in expressing cells only
            expr_positive = expr[expr > 0]
            mean_expr = float(np.mean(expr_positive)) if len(expr_positive) > 0 else 0.0
            
            results.append({
                "cell_type": str(ct),
                "pct_expressed": round(pct, 2),
                "mean_expression": round(mean_expr, 4)
            })
        
        return {
            "gene": gene_name,
            "data": results
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get gene dotplot for {gene_symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/qc_dashboard")
async def get_qc_dashboard():
    """Return quality control metrics for all cells."""
    try:
        return {
            "filtering_summary": {
                "original_cells": 15928,
                "cells_after_qc": int(adata.n_obs),
                "removed_cells": 15928 - int(adata.n_obs),
                "original_genes": 50281,
                "hvg_selected": 3000,
                "removal_percentage": round((15928 - int(adata.n_obs)) / 15928 * 100, 1)
            },
            "per_cell_qc": {
                "n_genes": [int(x) for x in adata.obs['n_genes_by_counts'].tolist()],
                "total_counts": [float(x) for x in adata.obs['total_counts'].tolist()],
                "pct_mt": [float(x) for x in adata.obs['pct_counts_mt'].tolist()],
                "cell_types": [str(ct) if pd.notna(ct) else "Unknown" 
                              for ct in adata.obs['cell_type'].tolist()]
            },
            "thresholds_used": {
                "min_genes": 500,
                "max_genes": 6000,
                "max_mt_pct": 10
            },
            "qc_stats": {
                "mean_genes_per_cell": round(float(adata.obs['n_genes_by_counts'].mean()), 1),
                "median_genes_per_cell": round(float(adata.obs['n_genes_by_counts'].median()), 1),
                "mean_counts_per_cell": round(float(adata.obs['total_counts'].mean()), 1),
                "mean_mt_pct": round(float(adata.obs['pct_counts_mt'].mean()), 2)
            }
        }
    except Exception as e:
        logger.error(f"Failed to get QC dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pca_variance")
async def get_pca_variance():
    """Return PCA variance explained."""
    try:
        variance_ratio = adata.uns['pca']['variance_ratio'][:30]
        cumulative = np.cumsum(variance_ratio)
        
        pcs_for_90 = int(np.where(cumulative >= 0.9)[0][0] + 1) if (cumulative >= 0.9).any() else 30
        
        return {
            "variance_ratio": [round(float(v), 6) for v in variance_ratio],
            "cumulative_variance": [round(float(v), 4) for v in cumulative],
            "pcs_for_90_pct": pcs_for_90,
            "total_pcs": 50
        }
    except Exception as e:
        logger.error(f"Failed to get PCA variance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cell_type_profile/{cell_type}")
async def get_cell_type_profile(cell_type: str):
    """Return detailed profile for a specific cell type."""
    try:
        mask = adata.obs['cell_type'] == cell_type
        
        if mask.sum() == 0:
            raise HTTPException(404, f"Cell type '{cell_type}' not found")
        
        return {
            "cell_type": cell_type,
            "n_cells": int(mask.sum()),
            "percentage": round(float(mask.sum() / len(adata) * 100), 2),
            "canonical_markers": CANONICAL_MARKERS.get(cell_type, []),
            "qc_stats": {
                "mean_genes": round(float(adata.obs[mask]['n_genes_by_counts'].mean()), 1),
                "mean_counts": round(float(adata.obs[mask]['total_counts'].mean()), 1),
                "mean_mt_pct": round(float(adata.obs[mask]['pct_counts_mt'].mean()), 2)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cell type profile for {cell_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
