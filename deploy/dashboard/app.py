"""
Brain Cell Atlas Dashboard - Streamlit Frontend
================================================
Research-grade dashboard for single-cell RNA-seq analysis
of the Human Middle Temporal Gyrus (MTG) dataset.

Pages:
- 🗺️ The Atlas: Dataset overview and UMAP visualization
- 🔬 Gene Inspector: Gene expression analysis
- 🤖 Model Performance: Classifier metrics and confusion matrix
- 🗣️ Language Genes: Language disorder gene analysis
- 📊 Data Quality: QC metrics and filtering summary

CRITICAL: This frontend uses thin-client architecture.
All data is fetched from the FastAPI backend via REST API.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Brain Cell Atlas",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_URL = "http://localhost:8000"

# =============================================================================
# Cell Type Colors
# =============================================================================

CELL_TYPE_COLORS: Dict[str, str] = {
    'Excitatory Neurons L2/3': '#ff6b6b',    # Coral red
    'Excitatory Neurons L5/6': '#ee5a6f',    # Rose red
    'Inhibitory Neurons': '#4ecdc4',         # Teal
    'Astrocytes': '#45b7d1',                 # Sky blue
    'Oligodendrocytes': '#f9ca24',           # Yellow
    'OPCs': '#ff9ff3',                       # Pink
    'Microglia': '#95e1d3',                  # Mint
    'Endothelial cells': '#a29bfe'           # Periwinkle
}

# =============================================================================
# CSS Styling - Dark Research Theme
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ===== CSS VARIABLES ===== */
:root {
    --bg-primary: #0a0a0a;
    --bg-secondary: #1a1a1a;
    --bg-tertiary: #2a2a2a;
    --accent-primary: #d4968c;
    --accent-hover: #e0a89e;
    --text-primary: #f5f5f5;
    --text-secondary: #a0a0a0;
    --text-muted: #707070;
    --border-default: #3a3a3a;
}

/* ===== GLOBAL THEME ===== */
.stApp {
    background: #0a0a0a;
    color: #f5f5f5;
}

/* ===== HIDE STREAMLIT BRANDING ===== */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.viewerBadge_container__1QSob {visibility: hidden;}

/* ===== TYPOGRAPHY ===== */
h1, h2, h3 {
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: #f5f5f5 !important;
    font-weight: 600 !important;
}

h1 { font-size: 2rem !important; }
h2 { font-size: 1.5rem !important; }
h3 { font-size: 1.25rem !important; }

/* Target specific text elements to avoid breaking icons */
p, .stMarkdown, .stButton, .stMetricLabel, .stMetricValue, label {
    font-family: 'Inter', -apple-system, sans-serif !important;
}

/* Base color but NO global font override that breaks icons */
.stApp {
    color: #f5f5f5;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: #1a1a1a;
    border-right: 1px solid #3a3a3a;
}

section[data-testid="stSidebar"] .stRadio label {
    color: #f5f5f5 !important;
    padding: 0.75rem 1rem;
    border-radius: 8px;
    transition: all 0.2s;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    background: #2a2a2a;
    color: #d4968c !important;
}

/* ===== METRIC CARDS ===== */
[data-testid="stMetric"] {
    background: #1a1a1a;
    padding: 1.25rem;
    border-radius: 12px;
    border: 1px solid #3a3a3a;
}

[data-testid="stMetricLabel"] {
    color: #a0a0a0 !important;
    font-size: 0.875rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

[data-testid="stMetricValue"] {
    color: #d4968c !important;
    font-size: 1.75rem;
    font-weight: 600;
}

/* ===== BUTTONS ===== */
.stButton > button {
    background: #d4968c !important;
    color: #0a0a0a !important;
    border: none !important;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s;
}

.stButton > button:hover {
    background: #e0a89e !important;
    transform: translateY(-1px);
}

/* ===== DATA TABLES ===== */
.stDataFrame {
    border: 1px solid #3a3a3a;
    border-radius: 8px;
    overflow: hidden;
}

/* ===== PLOTLY CHARTS ===== */
.js-plotly-plot {
    border-radius: 12px;
    border: 1px solid #3a3a3a;
}

/* ===== SELECT BOXES ===== */
.stSelectbox > div > div {
    background: #1a1a1a !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 8px;
    color: #f5f5f5 !important;
}

/* ===== EXPANDERS ===== */
.streamlit-expanderHeader {
    background: #1a1a1a;
    border: 1px solid #3a3a3a;
    border-radius: 8px;
    color: #f5f5f5 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ===== INFO/SUCCESS/ERROR BOXES ===== */
.stAlert {
    border-radius: 8px;
    border-left: 4px solid #d4968c;
    background: #1a1a1a;
}

/* ===== CUSTOM METRIC CARD ===== */
.metric-card {
    background: #1a1a1a;
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #3a3a3a;
    text-align: center;
    transition: all 0.2s;
}

.metric-card:hover {
    border-color: #d4968c;
}

.metric-value {
    font-family: 'Inter', sans-serif;
    font-size: 2rem;
    font-weight: 600;
    color: #d4968c;
    margin-bottom: 0.5rem;
}

.metric-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: #a0a0a0;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ===== RESULT CARDS ===== */
.result-card {
    background: #1a1a1a;
    border: 2px solid #3a3a3a;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}

.result-card-correct {
    border-color: #4caf50;
}

.result-card-wrong {
    border-color: #f44336;
}

.result-title {
    font-size: 0.75rem;
    color: #707070;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.75rem;
}

.result-value {
    font-size: 1.1rem;
    font-weight: 600;
    color: #f5f5f5;
}

.result-subtitle {
    font-size: 0.85rem;
    color: #a0a0a0;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# API Helper Function
# =============================================================================

def api_request(endpoint: str, method: str = "GET", data: dict = None) -> Optional[dict]:
    """Wrapper for API requests with error handling."""
    try:
        url = f"{API_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.ConnectionError:
        st.error("⚠️ **Backend Offline** - The FastAPI server is not running")
        st.markdown("""
        **Start the backend first:**
        ```bash
        cd deploy/api
        uvicorn app:app --host 0.0.0.0 --port 8000 --reload
        ```
        """)
        st.stop()
    
    except requests.exceptions.HTTPError as e:
        try:
            error_detail = e.response.json().get('detail', str(e))
        except:
            error_detail = str(e)
        st.error(f"⚠️ **API Error:** {error_detail}")
        return None
    
    except Exception as e:
        st.error(f"⚠️ **Unexpected Error:** {str(e)}")
        return None


# =============================================================================
# Sidebar Navigation
# =============================================================================

with st.sidebar:
    st.markdown("# Brain Cell Atlas")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["The Atlas", "Gene Inspector", "Model Performance", 
         "Language Genes", "Data Quality"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Dataset info
    st.markdown("### Dataset Info")
    health = api_request("/")
    
    if health:
        st.markdown(f"""
        <div style="background: #1a1a1a; padding: 1rem; border-radius: 8px; border: 1px solid #3a3a3a;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem;">
                <span style="color: #a0a0a0;">Cells</span>
                <span style="color: #f5f5f5; font-weight: 600;">{health['n_cells']:,}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem;">
                <span style="color: #a0a0a0;">Genes</span>
                <span style="color: #d4968c; font-weight: 600;">{health['n_genes']:,}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #a0a0a0;">Cell Types</span>
                <span style="color: #f5f5f5; font-weight: 600;">{len(health['cell_types'])}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #707070; font-size: 0.75rem;">
        Human MTG Atlas v1.0<br>
        Language Disorder Research
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Page 1: The Atlas
# =============================================================================

def page_atlas():
    """Render the Atlas overview page."""
    
    st.markdown("# The Atlas")
    st.markdown("### Human Middle Temporal Gyrus — Single-Cell Transcriptomics")
    
    health = api_request("/")
    if not health:
        return
    
    # Metric cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{health['n_cells']:,}</div>
            <div class="metric-label">Total Cells</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{health['n_genes']:,}</div>
            <div class="metric-label">Genes Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(health['cell_types'])}</div>
            <div class="metric-label">Cell Types</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Cell type distribution
    st.markdown("---")
    st.subheader("Cell Type Distribution")
    
    cell_data = api_request("/cell_types")
    
    if cell_data:
        df_dist = pd.DataFrame(cell_data['distribution'])
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(
                df_dist.style.format({'percentage': '{:.2f}%', 'count': '{:,}'}),
                use_container_width=True,
                height=350
            )
        
        with col2:
            fig = go.Figure(data=go.Bar(
                x=df_dist['count'],
                y=df_dist['cell_type'],
                orientation='h',
                marker=dict(
                    color=[CELL_TYPE_COLORS.get(ct, '#d4968c') for ct in df_dist['cell_type']],
                    line=dict(color='#3a3a3a', width=1)
                ),
                hovertemplate='<b>%{y}</b><br>Count: %{x:,}<extra></extra>'
            ))
            
            fig.update_layout(
                plot_bgcolor='#0a0a0a',
                paper_bgcolor='#0a0a0a',
                font=dict(family='Inter, sans-serif', color='#f5f5f5', size=12),
                xaxis=dict(title="Cell Count", gridcolor='#2a2a2a', zerolinecolor='#3a3a3a'),
                yaxis=dict(title="", gridcolor='#2a2a2a'),
                height=350,
                margin=dict(l=20, r=20, t=20, b=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # UMAP Visualization
    st.markdown("---")
    st.subheader("UMAP Projection")
    
    @st.cache_data(ttl=3600)
    def load_umap():
        return api_request("/umap_data")
    
    with st.spinner("Loading UMAP coordinates..."):
        umap_data = load_umap()
    
    if umap_data:
        df_umap = pd.DataFrame(umap_data)
        
        fig = go.Figure()
        
        for cell_type in df_umap['cell_type'].unique():
            subset = df_umap[df_umap['cell_type'] == cell_type]
            
            fig.add_trace(go.Scattergl(
                x=subset['x'],
                y=subset['y'],
                mode='markers',
                name=cell_type,
                marker=dict(
                    size=4,
                    color=CELL_TYPE_COLORS.get(cell_type, '#d4968c'),
                    opacity=0.8
                ),
                hovertemplate='<b>%{fullData.name}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            plot_bgcolor='#0a0a0a',
            paper_bgcolor='#0a0a0a',
            font=dict(family='Inter, sans-serif', color='#f5f5f5', size=12),
            xaxis=dict(title="UMAP 1", gridcolor='#2a2a2a', zerolinecolor='#3a3a3a'),
            yaxis=dict(title="UMAP 2", gridcolor='#2a2a2a', zerolinecolor='#3a3a3a'),
            height=700,
            hovermode='closest',
            legend=dict(
                bgcolor='#1a1a1a',
                bordercolor='#3a3a3a',
                borderwidth=1,
                font=dict(size=11, color='#f5f5f5')
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"📍 Displaying **{len(df_umap):,}** cells across **{df_umap['cell_type'].nunique()}** cell types")
    
    # Marker genes section
    st.markdown("---")
    st.subheader("Top Marker Genes by Cell Type")
    
    marker_data = api_request("/marker_genes_table")
    
    if marker_data:
        for cell_type in sorted(marker_data.keys()):
            if marker_data[cell_type]:
                with st.expander(f"**{cell_type}** - Top 10 Markers"):
                    df_markers = pd.DataFrame(marker_data[cell_type])
                    st.dataframe(
                        df_markers[['rank', 'gene', 'score']].style.format({'score': '{:.2f}'}),
                        use_container_width=True,
                        hide_index=True
                    )


# =============================================================================
# Page 2: Gene Inspector
# =============================================================================

def page_gene_inspector():
    """Render the Gene Inspector page."""
    
    st.markdown("# Gene Inspector")
    st.markdown("### Explore gene expression patterns across cell types")
    
    @st.cache_data(ttl=3600)
    def load_genes():
        data = api_request("/genes")
        return data['genes'] if data else []
    
    genes = load_genes()
    
    if not genes:
        st.error("Failed to load gene list from API")
        return
    
    # Gene selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_gene = st.selectbox(
            "Search for a gene:",
            options=genes,
            index=0,
            help="Type to search through available genes"
        )
    
    with col2:
        st.write("")
        st.write("")
        analyze_btn = st.button("ANALYZE", type="primary", use_container_width=True)
    
    # Suggested genes
    with st.expander("Suggested Marker Genes"):
        st.markdown("""
        **Neurons:** SLC17A7, GAD1, GAD2, PVALB, SST, VIP  
        **Glial:** GFAP, AQP4, MBP, PLP1, OLIG2  
        **Other:** PDGFRA, CX3CR1, CLDN5, PECAM1
        """)
    
    if analyze_btn and selected_gene:
        with st.spinner(f"Fetching data for **{selected_gene}**..."):
            gene_data = api_request(f"/gene_expression/{selected_gene}")
            dotplot_data = api_request(f"/gene_dotplot/{selected_gene}")
        
        if gene_data:
            st.success(f"Showing expression for **{gene_data['gene']}**")
            
            df_expr = pd.DataFrame(gene_data['expression'])
            
            # Visualization selector
            viz_type = st.radio(
                "Visualization:",
                ["Bar Chart", "Dot Plot", "UMAP Overlay", "Table"],
                horizontal=True,
                label_visibility="collapsed"
            )
            
            if viz_type == "Bar Chart":
                fig = go.Figure(data=go.Bar(
                    x=df_expr['cell_type'],
                    y=df_expr['mean_expression'],
                    marker=dict(
                        color=[CELL_TYPE_COLORS.get(ct, '#d4968c') for ct in df_expr['cell_type']],
                        line=dict(color='#3a3a3a', width=1)
                    ),
                    hovertemplate='<b>%{x}</b><br>Expression: %{y:.4f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=dict(
                        text=f"Expression of {gene_data['gene']} Across Cell Types",
                        font=dict(family='Inter, sans-serif', size=16, color='#f5f5f5')
                    ),
                    plot_bgcolor='#0a0a0a',
                    paper_bgcolor='#0a0a0a',
                    font=dict(family='Inter, sans-serif', color='#f5f5f5'),
                    xaxis=dict(title="Cell Type", gridcolor='#2a2a2a', tickangle=-45),
                    yaxis=dict(title="Mean Expression (log-normalized)", gridcolor='#2a2a2a'),
                    height=500,
                    margin=dict(b=120)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Dot Plot" and dotplot_data:
                dot_df = pd.DataFrame(dotplot_data['data'])
                
                # Scale sizes for visibility
                max_pct = max(dot_df['pct_expressed'].max(), 1)
                scaled_sizes = [max(5, (p / max_pct) * 50) for p in dot_df['pct_expressed']]
                
                fig = go.Figure(data=go.Scatter(
                    x=dot_df['cell_type'],
                    y=[1] * len(dot_df),
                    mode='markers',
                    marker=dict(
                        size=scaled_sizes,
                        color=dot_df['mean_expression'],
                        colorscale=[[0, '#1a1a1a'], [0.5, '#d4968c'], [1, '#ff6b6b']],
                        showscale=True,
                        colorbar=dict(title="Mean Expr", titlefont=dict(color='#f5f5f5'), tickfont=dict(color='#f5f5f5'))
                    ),
                    text=[f"{ct}<br>Expressed: {pct:.1f}%<br>Mean: {expr:.3f}" 
                          for ct, pct, expr in zip(dot_df['cell_type'], dot_df['pct_expressed'], dot_df['mean_expression'])],
                    hovertemplate='%{text}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=dict(text=f"Dot Plot: {gene_data['gene']}", font=dict(color='#f5f5f5')),
                    plot_bgcolor='#0a0a0a',
                    paper_bgcolor='#0a0a0a',
                    font=dict(family='Inter, sans-serif', color='#f5f5f5'),
                    xaxis=dict(title="Cell Type", tickangle=-45, gridcolor='#2a2a2a'),
                    yaxis=dict(visible=False),
                    height=300,
                    margin=dict(b=120, t=60)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.caption("**Dot size** = % cells expressing | **Color** = mean expression level")
            
            elif viz_type == "UMAP Overlay":
                with st.spinner("Loading UMAP overlay..."):
                    umap_expr = api_request(f"/gene_umap_overlay/{selected_gene}")
                
                if umap_expr:
                    fig = go.Figure(data=go.Scattergl(
                        x=[d['x'] for d in umap_expr['data']],
                        y=[d['y'] for d in umap_expr['data']],
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=[d['expression'] for d in umap_expr['data']],
                            colorscale=[[0, '#1a1a1a'], [0.5, '#d4968c'], [1, '#ff6b6b']],
                            showscale=True,
                            colorbar=dict(title=f"{selected_gene}", titlefont=dict(color='#f5f5f5'), tickfont=dict(color='#f5f5f5'))
                        ),
                        hovertemplate='Expression: %{marker.color:.3f}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title=dict(text=f"{selected_gene} Expression on UMAP", font=dict(color='#f5f5f5')),
                        plot_bgcolor='#0a0a0a',
                        paper_bgcolor='#0a0a0a',
                        font=dict(family='Inter, sans-serif', color='#f5f5f5'),
                        xaxis=dict(title="UMAP 1", gridcolor='#2a2a2a'),
                        yaxis=dict(title="UMAP 2", gridcolor='#2a2a2a'),
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            else:  # Table
                st.dataframe(
                    df_expr.style.format({'mean_expression': '{:.4f}'})
                           .background_gradient(subset=['mean_expression'], cmap='OrRd'),
                    use_container_width=True
                )
            
            # Download
            csv = df_expr.to_csv(index=False)
            st.download_button(
                label="Download Data (CSV)",
                data=csv,
                file_name=f"{selected_gene}_expression.csv",
                mime="text/csv"
            )


# =============================================================================
# Page 3: Model Performance
# =============================================================================

def page_model_performance():
    """Render Model Performance page."""
    
    st.markdown("# Model Performance")
    st.markdown("### Random Forest Classifier — Cell Type Prediction")
    
    perf = api_request("/model_performance")
    if not perf:
        return
    
    # Overall metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{perf['overall_accuracy']:.1%}")
    with col2:
        st.metric("Test Cells", f"{perf['test_size']:,}")
    with col3:
        st.metric("Features", "50 PCs")
    
    st.markdown("---")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    
    cm = np.array(perf['confusion_matrix'])
    class_names = perf['class_order']
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale=[[0, '#1a1a1a'], [0.5, '#d4968c'], [1, '#ff6b6b']],
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 10, "color": "#f5f5f5"},
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        plot_bgcolor='#0a0a0a',
        paper_bgcolor='#0a0a0a',
        font=dict(family='Inter, sans-serif', color='#f5f5f5'),
        xaxis=dict(title="Predicted Label", tickangle=-45),
        yaxis=dict(title="True Label"),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Per-class metrics
    st.markdown("---")
    st.subheader("Per-Class Metrics")
    
    metrics_data = []
    for cell_type, metrics in perf['per_class_metrics'].items():
        metrics_data.append({
            'Cell Type': cell_type,
            'Precision': f"{metrics['precision']:.3f}",
            'Recall': f"{metrics['recall']:.3f}",
            'F1-Score': f"{metrics['f1']:.3f}",
            'Support': metrics['support']
        })
    
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
    
    # PC Feature Importance
    st.markdown("---")
    st.subheader("Top 10 Principal Components by Importance")
    
    pc_data = api_request("/pc_importance")
    
    if pc_data:
        top10 = pc_data['top_10']
        
        fig = go.Figure(data=go.Bar(
            x=[f"PC{d['pc']+1}" for d in top10],
            y=[d['importance'] for d in top10],
            marker_color='#d4968c'
        ))
        
        fig.update_layout(
            plot_bgcolor='#0a0a0a',
            paper_bgcolor='#0a0a0a',
            font=dict(family='Inter, sans-serif', color='#f5f5f5'),
            xaxis=dict(title="Principal Component", gridcolor='#2a2a2a'),
            yaxis=dict(title="Feature Importance", gridcolor='#2a2a2a'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Random prediction demo
    st.markdown("---")
    st.subheader("Random Cell Prediction Demo")
    
    if st.button("PICK RANDOM CELL & PREDICT", type="primary"):
        with st.spinner("Analyzing..."):
            result = api_request("/predict_random", method="POST")
        
        if result:
            is_correct = result['true_label'] == result['predicted_label']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-title">TRUE LABEL</div>
                    <div class="result-value">{result['true_label']}</div>
                    <div class="result-subtitle">Cell Index: {result['cell_index']:,}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                status_class = "result-card-correct" if is_correct else "result-card-wrong"
                status_icon = "CORRECT" if is_correct else "MISMATCH"
                
                st.markdown(f"""
                <div class="result-card {status_class}">
                    <div class="result-title">PREDICTED</div>
                    <div class="result-value">{result['predicted_label']}</div>
                    <div class="result-subtitle">{status_icon} | Confidence: {result['confidence']:.1%}</div>
                </div>
                """, unsafe_allow_html=True)


# =============================================================================
# Page 4: Language Genes
# =============================================================================

def page_language_genes():
    """Render Language Genes analysis page."""
    
    st.markdown("# Language & Dyslexia Genes")
    st.markdown("### Analysis of language disorder-associated genes")
    
    lang_data = api_request("/language_genes_analysis")
    
    if not lang_data:
        return
    
    # Key finding
    st.info(f"""
    **Key Finding:** These {lang_data['comparison']['n_features_language']} language-related genes 
    alone achieve **{lang_data['comparison']['language_only_accuracy']:.1%}** classification accuracy, 
    compared to **{lang_data['comparison']['pca_accuracy']:.1%}** using {lang_data['comparison']['n_features_pca']} 
    PCA components from the full transcriptome.
    
    This suggests language genes show cell-type-specific expression patterns but are insufficient 
    alone for cell type classification.
    """)
    
    # Comparison metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("PCA Accuracy (50 features)", f"{lang_data['comparison']['pca_accuracy']:.1%}")
    with col2:
        st.metric("Language Genes Only (8 features)", f"{lang_data['comparison']['language_only_accuracy']:.1%}")
    
    st.markdown("---")
    st.subheader("Gene Expression by Cell Type")
    
    # Gene cards
    if lang_data['language_genes']:
        for gene_info in lang_data['language_genes']:
            with st.expander(f"{gene_info['gene']}"):
                expr_df = pd.DataFrame(gene_info['expression_by_celltype'])
                
                fig = go.Figure(data=go.Bar(
                    x=expr_df['cell_type'],
                    y=expr_df['mean_expression'],
                    marker=dict(
                        color=[CELL_TYPE_COLORS.get(ct, '#d4968c') for ct in expr_df['cell_type']]
                    )
                ))
                
                fig.update_layout(
                    plot_bgcolor='#0a0a0a',
                    paper_bgcolor='#0a0a0a',
                    font=dict(family='Inter, sans-serif', color='#f5f5f5'),
                    xaxis=dict(title="Cell Type", tickangle=-45, gridcolor='#2a2a2a'),
                    yaxis=dict(title="Mean Expression", gridcolor='#2a2a2a'),
                    height=350,
                    margin=dict(b=100)
                )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No language genes found in the dataset raw counts.")


# =============================================================================
# Page 5: Data Quality
# =============================================================================

def page_data_quality():
    """Render Data Quality page."""
    
    st.markdown("# Data Quality")
    st.markdown("### Quality control and filtering metrics")
    
    qc_data = api_request("/qc_dashboard")
    
    if not qc_data:
        return
    
    # Summary cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Original Cells", f"{qc_data['filtering_summary']['original_cells']:,}")
    with col2:
        st.metric("After QC", f"{qc_data['filtering_summary']['cells_after_qc']:,}",
                  delta=f"-{qc_data['filtering_summary']['removed_cells']:,}")
    with col3:
        st.metric("Original Genes", f"{qc_data['filtering_summary']['original_genes']:,}")
    with col4:
        st.metric("HVG Selected", f"{qc_data['filtering_summary']['hvg_selected']:,}")
    
    st.markdown("---")
    
    # Thresholds
    st.subheader("Filtering Thresholds Used")
    st.code(f"""
Minimum genes per cell: {qc_data['thresholds_used']['min_genes']}
Maximum genes per cell: {qc_data['thresholds_used']['max_genes']}
Maximum mitochondrial %: {qc_data['thresholds_used']['max_mt_pct']}%
    """)
    
    # QC stats
    st.subheader("QC Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Genes/Cell", f"{qc_data['qc_stats']['mean_genes_per_cell']:,.0f}")
    with col2:
        st.metric("Median Genes/Cell", f"{qc_data['qc_stats']['median_genes_per_cell']:,.0f}")
    with col3:
        st.metric("Mean Counts/Cell", f"{qc_data['qc_stats']['mean_counts_per_cell']:,.0f}")
    with col4:
        st.metric("Mean MT %", f"{qc_data['qc_stats']['mean_mt_pct']:.2f}%")
    
    st.markdown("---")
    
    # QC scatter plot
    st.subheader("Quality Control Metrics per Cell")
    
    # Sample if too many points for performance
    n_points = len(qc_data['per_cell_qc']['n_genes'])
    sample_size = min(5000, n_points)
    
    if n_points > sample_size:
        indices = np.random.choice(n_points, sample_size, replace=False)
        n_genes = [qc_data['per_cell_qc']['n_genes'][i] for i in indices]
        total_counts = [qc_data['per_cell_qc']['total_counts'][i] for i in indices]
        pct_mt = [qc_data['per_cell_qc']['pct_mt'][i] for i in indices]
        cell_types = [qc_data['per_cell_qc']['cell_types'][i] for i in indices]
        st.caption(f"Showing {sample_size:,} of {n_points:,} cells for performance")
    else:
        n_genes = qc_data['per_cell_qc']['n_genes']
        total_counts = qc_data['per_cell_qc']['total_counts']
        pct_mt = qc_data['per_cell_qc']['pct_mt']
        cell_types = qc_data['per_cell_qc']['cell_types']
    
    fig = go.Figure(data=go.Scattergl(
        x=n_genes,
        y=total_counts,
        mode='markers',
        marker=dict(
            size=3,
            color=pct_mt,
            colorscale=[[0, '#4caf50'], [0.5, '#ff9800'], [1, '#f44336']],
            showscale=True,
            colorbar=dict(title="% MT", titlefont=dict(color='#f5f5f5'), tickfont=dict(color='#f5f5f5'))
        ),
        text=cell_types,
        hovertemplate='Genes: %{x}<br>Counts: %{y}<br>MT: %{marker.color:.1f}%<br>Type: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        plot_bgcolor='#0a0a0a',
        paper_bgcolor='#0a0a0a',
        font=dict(family='Inter, sans-serif', color='#f5f5f5'),
        xaxis=dict(title="Number of Genes", gridcolor='#2a2a2a'),
        yaxis=dict(title="Total Counts", gridcolor='#2a2a2a'),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # PCA variance
    st.markdown("---")
    st.subheader("PCA Variance Explained")
    
    pca_data = api_request("/pca_variance")
    
    if pca_data:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=[f"PC{i+1}" for i in range(len(pca_data['variance_ratio']))],
            y=pca_data['variance_ratio'],
            name="Individual",
            marker_color='#d4968c'
        ))
        
        fig.add_trace(go.Scatter(
            x=[f"PC{i+1}" for i in range(len(pca_data['cumulative_variance']))],
            y=pca_data['cumulative_variance'],
            name="Cumulative",
            mode='lines+markers',
            line=dict(color='#45b7d1', width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            plot_bgcolor='#0a0a0a',
            paper_bgcolor='#0a0a0a',
            font=dict(family='Inter, sans-serif', color='#f5f5f5'),
            xaxis=dict(title="Principal Component", gridcolor='#2a2a2a', tickangle=-45),
            yaxis=dict(title="Variance Ratio", gridcolor='#2a2a2a'),
            legend=dict(bgcolor='#1a1a1a', bordercolor='#3a3a3a'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"**{pca_data['pcs_for_90_pct']}** PCs needed for 90% variance explained")


# =============================================================================
# Page Routing
# =============================================================================

if page == "The Atlas":
    page_atlas()
elif page == "Gene Inspector":
    page_gene_inspector()
elif page == "Model Performance":
    page_model_performance()
elif page == "Language Genes":
    page_language_genes()
elif page == "Data Quality":
    page_data_quality()
