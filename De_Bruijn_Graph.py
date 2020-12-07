import matplotlib.pyplot as plt
import toyplot
import toyplot.browser
import networkx as nx
import os
from PIL import Image
from graphviz import Digraph


class Graph:
    """
    Wrapper for De-Bruijn Graph and Line-Graph
    """

    def __init__(self):
        self.G = nx.DiGraph()

    def create_line_graph(self):
        """
        This method creates the line graph from any given (Di)Graph.
        :rtype: object
        """
        g_line_graph = Graph()
        g_line_graph.G = nx.DiGraph(nx.line_graph(self.G))
        return g_line_graph

    def plot(self, filename='graph'):
        f = Digraph(filename=filename + '.gv')  # Digraph (name, filename to save)
        f.attr(rankdir='LR', size='8,5')  # horizontal and not vertical
        # assume no labels, just simple printing
        f.attr('node', shape='circle')
        for e in self.G.edges():
            f.edge(str(e[0]), str(e[1]), label=None) # can add labels = name that WILL BE DISPLAYED on the edges
        f.view()


class DeBruijnGraph(Graph):
    def __init__(self, sequence, k):
        """
        :param sequence: the de-bruijn sequence
        :param n: the length of the binary representation
        """
        super().__init__()
        self.n = k  # number of bits
        self.N = 2 ** k  # number of nodes
        kmers = self.get_kmers_from_sequence(sequence, k)
        edges = self.get_edges_from_kmers(kmers)
        self.G = nx.DiGraph(edges)
        # note: we no longer add ids to edges, we need to find the actual nodes.
        # when we need to print the edges, we will generate them manually.
        # it may be slower, need to think about it.
        # keeping this for legacy in case we need it.
        """
        for i in range(0, len(seq)):
            u = seq[i:i + n]
            v = seq[i + 1:i + n + 1]
            u_of_edge = str(u)
            v_of_edge = str(v)
            e = seq[i:i + n]
            self.G.add_node(u_of_edge, id=int(u_of_edge, 2))
            self.G.add_node(v_of_edge, id=int(v_of_edge, 2))
            self.G.add_edge(u_of_edge, v_of_edge, id=int(str(e), 2), id_b=str(e))
        """

    @staticmethod
    def get_kmers_from_sequence(sequence, k):
        """
        Returns dictionary with keys representing all possible kmers in a sequence
        and values counting their occurrence in the sequence.
        """
        # dict to store kmers
        kmers = {}

        # count how many times each occurred in this sequence (treated as cyclic)
        for i in range(0, len(sequence)):
            kmer = sequence[i:i + k]

            # skip kmers at end of sequence
            if len(kmer) != k:
                continue

            # count occurrence of this kmer in sequence
            if kmer in kmers:
                kmers[kmer] += 1
            else:
                kmers[kmer] = 1

        return kmers

    @staticmethod
    def get_edges_from_kmers(kmers):  # todo: add labels
        """
        Every possible (k-1)mer (n-1 suffix and prefix of kmers) is assigned
        to a node, and we connect one node to another if the (k-1)mer overlaps
        another. Nodes are (k-1)mers, edges are kmers.
        """
        # store edges as tuples in a set
        edges = set()
        # compare each (k-1)mer
        for k1 in kmers:
            for k2 in kmers:
                if k1 != k2:
                    # if they overlap then add to edges
                    if k1[1:] == k2[:-1]:
                        edges.add((k1[:-1], k2[:-1]))
                    if k1[:-1] == k2[1:]:
                        edges.add((k2[:-1], k1[:-1]))
                    # todo: add edge label

        return edges

    def exchange(self, i):
        """
        This method receives a binary number and executes the exchange function.
        :rtype: object
        :param i: first n-1 bits of node to perform the exchange operation on.
        """
        N = self.N
        # to get binary representation with literals for multiplication: bin(int(i,2))
        # decimal representations
        x1 = int(i, 2)  # i
        x2 = int (x1 + N / 2)  # i+N/2
        y1 = (2 * x1) % N  # 2i
        y2 = (2 * x1 + 1) % N  # 2i+1
        z1 = (2 * y1) % N  # 4i
        z2 = (2 * y1 + 1) % N  # 4i+1
        z3 = (2 * y2) % N  # 4i+2
        z4 = (2 * y2 + 1) % N  # 4i+3
        # binary representations, removing the literal
        x1_b = bin(x1)[2:]
        x2_b = bin(x2)[2:]
        y1_b = bin(y1)[2:]
        y2_b = bin(y2)[2:]
        z1_b = bin(z1)[2:]
        z2_b = bin(z2)[2:]
        z3_b = bin(z3)[2:]
        z4_b = bin(z4)[2:]
        # we now find the edges themselves.

        # first we remove 2i->4i and add 2i->4i+2.
        edge_to_remove = (y1_b, z1_b)
        self.G.remove_edge(*edge_to_remove)
        # next we add 2i->4i+2.  # todo: problem, how do we define names for 2i->4i+2? It's not De-Bruijn like.
        edge_to_add = (y1, z3)
        self.G.add_edge(*edge_to_add)

        # next we remove 2i+1->4i+2.
        edge_to_remove = (y2, z3)
        self.G.remove_edge(*edge_to_remove)
        # finally we add 2i+1->4i.
        edge_to_add = (y2, z1)
        self.G.add_edge(*edge_to_add)

    def exchange_two(self, i1, i2):
        """
        performs two exchange operations on given bits i1 and i2.
        :rtype: object
        """
        self.exchange(i1)
        self.exchange(i2)

    @property
    def __str__(self):
        res = 'There are ' + str(self.G.number_of_nodes) + ' nodes and ' \
              + str(self.G.number_of_edges) + ' edges in G.'
        return res


if __name__ == '__main__':
    n = 4
    # n = int(input('Please enter an input number between 3 and 5.\n'))
    seq = ''
    while n not in [2, 3, 4, 5]:
        n = int(input('Invalid input. Please enter a valid number.\n'))
    if n == 2:
        seq = '00110'
    elif n == 3:
        seq = '0001011100'
    elif n == 4:
        seq = '0000100110101111000'
    elif n == 5:
        seq = '000001000110010100111010110111110000'
    elif n == 6:
        seq = '000000100001100010100011100100101100110100111101010111011011111100000'

    DBG = DeBruijnGraph(seq, n)
    # DBG.plot_graph()
    #DBG.plot('DBG')
    line_graph = DBG.create_line_graph()
    #line_graph.plot('line graph before')
    DBG.exchange('10')
    line_graph = DBG.create_line_graph()
    line_graph.plot_graph('line graph after')
