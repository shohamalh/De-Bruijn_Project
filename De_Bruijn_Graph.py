import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import write_dot
import os
from PIL import Image


class DeBruijnGraph:
    # self.n is actually n-1, for length 3 send 4, for length 4 send 5, for length 5 send 6
    def __init__(self, seq, n):
        self.n = n
        self.N = pow(2, n)
        self.G = nx.DiGraph()
        for i in range(len(seq) - n + 1):
            u = seq[i:i + n - 1]
            v = seq[i + 1:i + n]
            u_of_edge = str(u)
            v_of_edge = str(v)
            e = seq[i:i + n]
            self.G.add_node(u_of_edge, id=int(u_of_edge, 2))
            self.G.add_node(v_of_edge, id=int(v_of_edge, 2))
            self.G.add_edge(u_of_edge, v_of_edge, id=int(str(e), 2), id_b=str(e))
        print("done init")

    def print(self):  # print labels, but no self-loops
        G = self.G
        pos = nx.spring_layout(G)
        plt.figure()
        nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos=pos)
        edge_labels = nx.get_edge_attributes(G, 'id_b')  # key is edge, pls check for your case
        formatted_edge_labels = {(elem[0], elem[1]): edge_labels[elem] for elem in
                                 edge_labels}  # use this to modify the tuple keyed dict if it has > 2 elements, else ignore
        nx.draw_networkx_edge_labels(G, pos, edge_labels=formatted_edge_labels, font_color='red')
        plt.show()

    def print_graph(self):  # creates a png and opens it, no edge labels
        self.G.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}
        self.G.graph['graph'] = {'scale': '3'}
        nx.nx_pydot.write_dot(self.G, 'graph.dot')
        os.system('dot -Tpng graph.dot > De-Bruijn_graph.png')
        graph = Image.open('De-Bruijn_graph.png')
        graph.show()

    """
    This method receives a binary number and executes the exchange function.
    """

    # i is an n-1 bit input
    def exchange(self, i):
        n = self.n
        # decimal representations
        x1 = int(str(i), 2)  # i
        x2 = x1 + self.N / 2  # i+N/2
        y1 = (2 * x1) % n  # 2i
        y2 = (2 * x1 + 1) % n  # 2i+1
        z1 = (2 * y1) % n  # 4i
        z2 = (2 * y1 + 1) % n  # 4i+1
        z3 = (2 * y2) % n  # 4i+2
        z4 = (2 * y2 + 1) % n  # 4i+3

        edge_to_remove_number = int(int((i + '0'), 2))  # todo: do we need % n for the numbers?
        edge_to_remove = [(u, v) for u, v, e in self.G.edges.data() if e['id'] == edge_to_remove_number][0]
        print('first we remove 2i->4i and add 2i->4i+2.')
        self.G.remove_edge(*edge_to_remove)

        print('next we add 2i->4i+2.')  # todo: problem, how do we define names for 2i->4i+2? It's not De-Bruijn like.
        # edge_to_add_number = int(int((i + '0'), 2))
        edge_to_add = (y1, z3)
        # self.G.add_edge(edge_to_add, id=todo:)

        print('next we remove 2i+1->4i+2.')
        edge_to_remove_number = int(int((i + '1'), 2))
        edge_to_remove = [(u, v) for u, v, e in self.G.edges.data() if e['id'] == edge_to_remove_number][0]
        self.G.remove_edge(*edge_to_remove)
        print('finally we add 2i+1->4i.')
        # edge_to_add_number = int(int((todo:)))
        edge_to_add = (y2, z1)
        # self.G.add_edge(edge_to_add, id=todo:)
        # need to find the correct nodes and edges and then remove the edges and creates new ones in a new graph
        print("done exchange")

    def exchange_two(self, i1, i2):
        G.exchange(i1)
        G.exchange(i2)

    def __str__(self):
        res = 'There are ' + (self.G.number_of_nodes) + ' nodes and ' \
              + str(self.G.number_of_edges) + ' edges in G.'
        return res


if __name__ == '__main__':
    n = 4
    #    n = int(input('Please enter an input number between 3 and 5.\n'))
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
    G.exchange('10')
    exit(0)
    print(G)
