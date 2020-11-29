import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import write_dot
import os
from PIL import Image


def binaryToDecimal(binary) -> int:
    binary1 = binary
    decimal, i, n = 0, 0, 0
    while (binary != 0):
        dec = binary % 10
        decimal = decimal + dec * pow(2, i)
        binary = binary // 10
        i += 1
    return decimal


class DeBruijnGraph:
    def __init__(self):
        self.n = 0
        self.N = 0
        self.edges = []
        self.nodes = set()
        self.number_of_nodes = 0
        self.number_of_edges = 0

    def init_graph(self, st, n):
        for i in range(len(st) - n + 1):
            self.edges.append((st[i:i + n - 1], st[i + 1:i + n]))
            self.number_of_edges += 1
            self.nodes.add(st[i:i + n - 1])
            self.nodes.add(st[i + 1:i + n])
            self.number_of_nodes += 2
        self.number_of_nodes = len(self.nodes)
        self.number_of_edges = len(self.edges)
        self.n = n
        self.N = pow(2, n)

    def visualize(self, st, n):
        self.init_graph(st, n)
        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(node)
        for src, dst in self.edges:
            G.add_edge(src, dst)

        G.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}
        G.graph['graph'] = {'scale': '3'}

        nx.nx_pydot.write_dot(G, 'graph.dot')
        os.system('dot -Tpng graph.dot > De-Bruijn_graph.png')
        graph = Image.open('De-Bruijn_graph.png')
        graph.show()

        # the function int(number, base) converts bases.

    """
    This method receives a binary number and executes the exchange function.
    """

    # i is an n-1 bit input
    def exchange(self, i):  # i = 10
        x = binaryToDecimal(i)
        print(x)

    def __str__(self):
        res = 'There are ' + str(G.number_of_nodes) + ' nodes and ' + str(G.number_of_edges) + ' edges in G.'
        return res


"""
This methods generates a De-Bruijn sequence of any length. Not required.

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
"""

if __name__ == '__main__':
    n = int(input('Please enter an input number between 3 and 5.\n'))
    while n not in [3, 4, 5]:
        n = int(input('Invalid input. Please enter a valid number.\n'))
    if n == 3:
        sequence = '0001011100'
    elif n == 4:
        sequence = '0000100110101111000'
    else:
        sequence = '000001000110010100111010110111110000'

    G = DeBruijnGraph()
    G.exchange(10)
    G.visualize(sequence, n)
    print(G)
