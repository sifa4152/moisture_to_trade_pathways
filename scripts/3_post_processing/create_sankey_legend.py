import plotly.graph_objects as go

def create_sankey_legend_5x5():
    """
    Standalone script to generate a 5x5 annotated Sankey legend.
    5 Sources (A) -> 1 Mediator (B) -> 5 Destinations (C)
    """

    # 1. LOCAL COLOR DEFINITIONS
    COLOR_SOURCE_REF = "#E8A279" # Orange (used in text to explain other plots)
    COLOR_DEST_REF = "#628398"   # Blue (used in text to explain other plots)
    
    NODE_DARK = "#131313"  # Mediator
    NODE_MID = "#333333"   # Standard countries
    NODE_LIGHT = "#B0B0B0" # Others
    LINK_COLOR = "#DCDCDC" # Standard link color (gray)

    # 2. DUMMY DATA STRUCTURE (5 Sources, 1 Mediator, 5 Destinations)
    # Indices: 
    # 0-4: Sources | 5: Mediator | 6-10: Destinations
    node_labels = [
        "Source A₁", "Source A₂", "Source A₃", "Source A₄", "Others", # 0-4
        "<b>Mediator (B)</b>",                                         # 5
        "Dest. C₁", "Dest. C₂", "Dest. C₃", "Dest. C₄", "Others "     # 6-10
    ]
    
    # Define flows (Source, Target, Value)
    links = [
        # Left Side: Sources -> Mediator
        (0, 5, 30), (1, 5, 15), (2, 5, 10), (3, 5, 5), (4, 5, 40),
        # Right Side: Mediator -> Destinations
        (5, 6, 25), (5, 7, 20), (5, 8, 15), (5, 9, 15), (5, 10, 25)
    ]
    
    sources, targets, values = zip(*links)

    # 3. NODE STYLING
    node_colors = [
        NODE_MID, NODE_MID, NODE_MID, NODE_MID, NODE_LIGHT, # Sources
        NODE_DARK,                                          # Mediator
        NODE_MID, NODE_MID, NODE_MID, NODE_MID, NODE_LIGHT  # Destinations
    ]

    # 4. CREATE FIGURE
    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(
            pad=12, thickness=15,
            color=node_colors,
            line=dict(width=0),
            # Manual coordinates: Left=0.01, Middle=0.5, Right=0.99
            x=[0.01]*5 + [0.5] + [0.99]*5,
            # Vertical distribution (0.1 to 0.9)
            y=[0.1, 0.3, 0.5, 0.7, 0.9, 0.5, 0.1, 0.3, 0.5, 0.7, 0.9]
        ),
        link=dict(
            source=sources, target=targets, value=values,
            color=LINK_COLOR
        )
    )) #            label=node_labels,

    # 5. LAYOUT & ANNOTATIONS
    width_px  = 70 / 25.4 * 72
    height_px = 65 / 25.4 * 72
    fig.update_layout(
        title=dict(
            text="", 
            x=0.5, y=0.97, font=dict(size=12)
        ),
        font=dict(size=10, family="Arial", color="black"),
        width=width_px, height=height_px,
        margin=dict(l=7, r=7, t=10, b=5),
        paper_bgcolor="white"
    ) #<b>How to read the Sankey diagrams</b>

    return fig

if __name__ == "__main__":
    fig = create_sankey_legend_5x5()
    fig.show()
    # Save as PDF
    output_name = "/p/projects/open/simon/bgwater/papers/paper_III/workflow/output/version_2026-03-26/figures/sankeys/fig5b_legend_base.pdf"
    # fig.write_image(output_name)
    
    print(f"Legend successfully created: {output_name}")
    