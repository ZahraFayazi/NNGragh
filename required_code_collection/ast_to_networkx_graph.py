import networkx as nx

def add_to_graph(graph, node, parent=None):
    if node is not None:
        graph.add_node(node.number, label=node.value)
        if parent is not None:
            graph.add_edge(parent.number, node.number)
        for child in node.children:
            add_to_graph(graph, child, node)

def transform_ast_to_networkx(ast_root_node):
    graph = nx.DiGraph()
    add_to_graph(graph, ast_root_node)
    return graph

def show_ast(ast_root_node):
    from matplotlib import pyplot as plt
    from networkx.drawing.nx_pydot import graphviz_layout

    graph = transform_ast_to_networkx(ast_root_node)
    pos = graphviz_layout(graph, prog="dot")
    nx.draw(
        graph,
        pos,
        node_size=500,
        labels=nx.get_node_attributes(graph, "label"),
        alpha=0.5,
        node_color="cyan",
        with_labels=True
    )
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()