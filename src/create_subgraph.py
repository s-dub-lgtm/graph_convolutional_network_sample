# %%
import dgl

# %%
g = dgl.DGLGraph()
g.add_nodes(5)
g.add_edges([0, 1, 2, 3, 4], [1, 2, 3, 4, 0])
sg = g.subgraph([0, 1, 4])
