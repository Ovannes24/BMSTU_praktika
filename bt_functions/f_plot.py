import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

def plot_arrows(fig, df, up_down):
    if up_down == 'up':
        ay=90
        color='green'
    else:
        ay=-90
        color='red'
        
    for i, j in zip(df.Close, df.index):
        fig.add_annotation(x=j, y=i, xref="x", yref="y", showarrow=True, arrowhead=1, arrowsize=1, arrowwidth=2,
            arrowcolor=color,
            ax=0,
            ay=ay,  
        )
    
    return fig