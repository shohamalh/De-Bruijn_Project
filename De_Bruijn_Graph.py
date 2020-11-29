import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import write_dot
import os
from PIL import Image


class DeBruijnGraph:
    def __init__(self, seq, n):
        self.n = n
        self.N = pow(2,n)
        self.G = nx.DiGraph()
        for i in range(len(seq) - n + 1):
            self.G.add_edge(seq[i:i + n - 1], seq[i + 1:i + n]) #adds nodes and edges

    def print_graph(self):
        self.G.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}
        self.G.graph['graph'] = {'scale': '3'}

        nx.nx_pydot.write_dot(self.G, 'graph.dot')
        os.system('dot -Tpng graph.dot > De-Bruijn_graph.png')
        graph = Image.open('De-Bruijn_graph.png')
        graph.show()

        # the function int(number, base) converts bases.

    """
    This method receives a binary number and executes the exchange function.
    """

    # i is an n-1 bit input
    def exchange(self, i) -> nx.DiGraph():
        x = int(i, 2)  # i
        y = x + 1  # i + N/2
        z = x * 2  # 2i
        w = x * 2 + 1  # 2i +1

        # need to find the correct nodes and edges and then remove the edges and creates new ones in a new graph

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

    G = DeBruijnGraph(sequence, n)
    G.print_graph()
    exit(0)
    G.exchange('10')
    G.visualize(sequence, n)
    print(G)
