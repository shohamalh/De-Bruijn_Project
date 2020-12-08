import networkx as nx
from graphviz import Digraph


def create_de_bruijn_sequence(k) -> str:
    """
    An implementation of the FKM algorithm for generating the de Bruijn
    sequence containing all 2-ary strings of length n, as described in
    "Combinatorial Generation" by Frank Ruskey.
    NOTE: this creates cyclic versions of the string.
    """
    _ = int(2)
    alphabet = list(map(str, range(2)))
    n = k + 1
    a = [0] * 2 * n
    sequence = []

    def db(t, p):
        if t > n:
            if n % p == 0:
                sequence.extend(a[1: p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, 2):
                a[t] = j
                db(t + 1, t)

    db(1, 1)
    return "".join(alphabet[i] for i in sequence)


class Graph:
    """
    Wrapper for De-Bruijn Graph and Line-Graph
    """

    def __init__(self):
        self.n = 0
        self.k = 0
        self.N = 0
        self.G = nx.DiGraph()
        # remember to change to False after exchanging edges or creating the line graph.
        self.is_de_bruijn = False
        self.is_line_graph = False

    def create_line_graph(self):
        """
        This method creates the line graph from any given (Di)Graph.
        :rtype: object
        """
        g_line_graph = Graph()
        g_line_graph.G = nx.DiGraph(nx.line_graph(self.G))
        g_line_graph.n = self.n  # just for the tuples
        g_line_graph.is_de_bruijn = False
        g_line_graph.is_line_graph = True
        return g_line_graph

    def plot(self, filename='graph'):
        # format(14, '08b') turns 14 to binary with 8 bits, with 0 padding: 00001110
        f = Digraph(filename=filename + '.gv')  # Digraph (name, filename to save)
        f.attr(rankdir='LR', size='8,5')  # horizontal and not vertical
        # assume no labels, just simple printing
        f.attr('node', shape='circle')
        for e in self.G.edges():
            if not self.is_line_graph:
                formatted_u = format(e[0], '0' + str(self.n) + 'b')
                formatted_v = format(e[1], '0' + str(self.n) + 'b')
            else: # is a line graph
                formatted_u_0 = format(e[0][0], '0' + str(self.n) + 'b')
                formatted_u_1 = format(e[0][1], '0' + str(self.n) + 'b')
                formatted_v_0 = format(e[1][0], '0' + str(self.n) + 'b')
                formatted_v_1 = format(e[1][1], '0' + str(self.n) + 'b')
                formatted_u = formatted_u_0 + ', ' + formatted_u_1
                formatted_v = formatted_v_0 + ', ' + formatted_v_1
            f.edge(formatted_u, formatted_v, label=None)  # can add labels = name that WILL BE DISPLAYED on the edges
        f.view()


class DeBruijnGraph(Graph):
    def __init__(self, sequence, n):
        """
        :param sequence: the de-bruijn sequence
        :param n: the length of the binary representation
        """
        super().__init__()
        self.n = n  # number of bits of each node
        self.k = n + 1
        self.N = 2 ** (self.n)  # number of nodes
        kmers = self.get_kmers_from_sequence(sequence, self.k)
        edges = self.get_edges_from_kmers(kmers)
        self.G = nx.DiGraph(edges)
        self.is_de_bruijn = True
        self.is_line_graph = False
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
                kmer += sequence[:(k - len(kmer))]

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
                    # making the str an int in the first place
                    if k1[1:] == k2[:-1]:
                        # edges.add((k1[:-1], k2[:-1]))
                        edges.add((int(k1[:-1], 2), int(k2[:-1], 2)))
                    if k1[:-1] == k2[1:]:
                        # edges.add((k2[:-1], k1[:-1]))
                        edges.add((int(k2[:-1], 2), int(k1[:-1], 2)))
                    # todo: add edge label

        return edges

    def exchange(self, i: str) -> None:
        """
        This method receives a binary number and executes the exchange function.
        :param i: first n-1 bits of node to perform the exchange operation on.
        """
        # check that there were given n-1 bits
        assert (isinstance(i, str))
        if len(i) != self.n - 1:
            raise ValueError

        N = self.N
        # we convert the binary representation to decimal
        x1 = int(i, 2)  # i
        x2 = int('1' + i, 2)  # i+N/2
        y1 = (2 * x1) % N  # 2i
        y2 = (2 * x1 + 1) % N  # 2i+1
        z1 = (2 * y1) % N  # 4i
        z2 = (2 * y1 + 1) % N  # 4i+1
        z3 = (2 * y2) % N  # 4i+2
        z4 = (2 * y2 + 1) % N  # 4i+3
        # we now find the edges themselves.

        # first we remove 2i->4i and 2i+1->4i+2
        # work around since G doesn't save the nodes with 0b literal - iterating over all edges and comparing.
        for e in self.G.edges():
            if (e == (y1, z1)) or (e == (y2, z3)):
                self.G.remove_edge(*e)
                break

        # next we add 2i->4i+2 and 2i+1->4i  # todo: problem, how do we define names for 2i->4i+2? It's not De-Bruijn like.
        edge_to_add = (y1, z3)
        self.G.add_edge(*edge_to_add)
        edge_to_add = (y2, z1)
        self.G.add_edge(*edge_to_add)
        # not a de-bruijn graph anymore
        self.is_de_bruijn = False

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
    print('Please enter the sequence kmer length.\n')
    while True:
        n = input()
        try:
            n = int(n)
        except:
            print('Invalid input. Please enter a valid number.')
            continue
        if n < 1:
            print('Invalid input. Please enter a valid number.')
            continue
        break

    seq = create_de_bruijn_sequence(n)
    DBG = DeBruijnGraph(seq, n)
    # DBG.plot('DBG')

    line_graph = DBG.create_line_graph()
    # line_graph.plot('line graph before')

    DBG.exchange('10')
    # DBG.plot('DBG after exchange')

    # line_graph = DBG.create_line_graph()
    # line_graph.plot('line graph after')
