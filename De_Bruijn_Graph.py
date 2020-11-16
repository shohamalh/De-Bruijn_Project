import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import write_dot
import os
from PIL import Image

class GraphVisualization:
    def __init__(self):
        self.edges = []
        self.nodes = set()
        self.number_of_nodes = 0
        self.number_of_edges = 0

    def init_graph(self, st, k):
        for i in range(len(st) - k + 1):
            self.edges.append((st[i:i + k - 1], st[i + 1:i + k]))
            self.number_of_edges += 1
            self.nodes.add(st[i:i + k - 1])
            self.nodes.add(st[i + 1:i + k])
            self.number_of_nodes += 2
        self.number_of_nodes = len(self.nodes)
        self.number_of_edges = len(self.edges)

    def visualize(self, st, k):
        self.init_graph(st, k)
        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(node)
        for src, dst in self.edges:
            G.add_edge(src, dst)

        G.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}
        G.graph['graph'] = {'scale': '3'}

        nx.nx_pydot.write_dot(G,'graph.dot')
        os.system('dot -Tpng graph.dot > De-Bruijn_graph.png')
        graph = Image.open('De-Bruijn_graph.png')
        graph.show()

def de_bruijn(n) -> str:
    alphabet = list(map(str,range(2)))
    a = [0]*2*n
    seq = []
    def db(t, p):
        if t > n:
            if n % p == 0:
                seq.extend(a[1 : p + 1])
        else:
            a[t] = a[t-p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, 2):
                a[t] = j
                db(t+1, t)
    db(1,1)
    return "".join(alphabet[i] for i in seq)

if __name__ == '__main__':
    de_bruijn_three = '0001110100'
    de_bruijn_four = '0000100110101111000'
    G = GraphVisualization()
    G.visualize(de_bruijn_three, 3)
    print ("There are " + str(G.number_of_nodes) + " nodes in G.")
    print ("There are " + str(G.number_of_edges) + " edges in G.")