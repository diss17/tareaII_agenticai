import sys
from pathlib import Path

# Añadir raíz del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graph.builder import build_graph

# Construir el grafo compilado
graph = build_graph()

# Exportar como PNG
output_dir = Path(__file__).parent / "exports"
output_dir.mkdir(exist_ok=True)

png_path = output_dir / "graph_diagram.png"
png_data = graph.get_graph().draw_mermaid_png()
with open(png_path, "wb") as f:
    f.write(png_data)

print(f"Grafo exportado a: {png_path}")
