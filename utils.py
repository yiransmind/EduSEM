import pandas as pd
import plotly.graph_objects as go
from semopy import model_visualization as vis


def load_data(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def plot_path_diagram(model, title="SEM Path Diagram"):
    G = vis.plot(model)
    fig = go.Figure()
    for edge in G.edges(data=True):
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines+markers',
                                 text=[edge[2]['label']], hoverinfo='text'))
    fig.update_layout(title_text=title, showlegend=False)
    return fig
