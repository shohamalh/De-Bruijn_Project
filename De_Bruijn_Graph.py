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
        self.n = 0  # length of node representation
        self.input_n = 0  # original length of node representation to be used in L(G)
        self.k = 0  # length of edge representation
        self.N = 0  # number of Nodes
        self.E = 0  # number of Edges
        self.G = nx.DiGraph()

    def plot(self, filename='graph'):
        f = Digraph(filename=filename + '.gv')  # Digraph (name, filename to save)
        f.attr(rankdir='LR', size='8,5')  # horizontal and not vertical
        f.attr('node', shape='circle')
        edge_label = None
        for e in self.G.edges():  # choses an edge from the _____ graph every iteration
            if isinstance(self, DeBruijnGraph):  # if the graph is debruijn
                u = bin(e[0])[2:].zfill(self.n)  #
                v = bin(e[1])[2:].zfill(self.n)
                edge_label = u + v[-1]  # creates the edge label by combining the names of the vertices
            elif isinstance(self, LineGraph):  # is a line graph
                u0 = bin(e[0][0])[2:].zfill(self.n - 1)
                u1 = bin(e[0][1])[2:].zfill(self.n - 1)
                v0 = bin(e[1][0])[2:].zfill(self.n - 1)
                v1 = bin(e[1][1])[2:].zfill(self.n - 1)
                u = u0 + u1[-1]
                v = v0 + v1[-1]
            f.edge(u, v, label=edge_label)  # can add labels = name that WILL BE DISPLAYED on the edges
        f.view()

    def gen_strings(self, n, arr, i):
        if i == n:
            return

        # First assign "0" at ith position
        # and try for all other permutations
        # for remaining positions
        arr[i] = 0
        # [1 ,0, 0, 0]
        self.gen_strings(n, arr, i + 1)

        # And then assign "1" at ith position
        # and try for all other permutations
        # for remaining positions
        arr[i] = 1
        self.gen_strings(n, arr, i + 1)


class LineGraph(Graph):
    def __init__(self, g: Graph):
        """
        This method creates the line graph from any given (Di)Graph.
        :rtype: object
        """
        super().__init__()
        self.G = nx.DiGraph(nx.line_graph(g.G))
        self.input_n = g.input_n
        self.n = g.n + 1  # length of node representation
        self.k = self.n + 1  # length of edge representation
        self.N = self.G.number_of_nodes()
        self.E = self.G.number_of_edges()
        self.quadruples = []
        self._find_quadruples()  # 1. find all quadruples.
        self.decompositions = []
        self._find_decompositions()  # 2. find all decompositions.
        """
        decompositions_with_circles is a list of tuples:
        (current_decomposition_with_circles, circles_in_each_length_in_decomposition).
        current_decomposition_with_circles is a list of all the circles in the decomposition.
        circles_in_each_length_in_decomposition is a list in size |E|, and the [i] place has the number of the 
        circles with the length of i+1 (the first place is [0] - has the number of circles with length 1)
        """
        self.decompositions_with_circles = []
        """
        unique_circles_in_each_length is a list in size |E|, and the [i] place is a Set that has all the 
        circles (a circle is a list) with the length of i+1 (the first place is [0] - has all the circles with length 1)
        ***but the circles in each set are sorted by sort function, so they are not in the order of the circle!!!!!***
        """
        # self.unique_circles_in_each_length = [None] * self.E
        self.unique_circles_in_each_length = [[] for _ in range(int(self.E / 2))]
        self._find_decompositions_circles()  # 3. find all circles in decompositions.

    # private methods

    def _create_quadruple(self, edge) -> list:
        """
        # v1 -> u1  a
        #   X    \b,   /c
        # v2 -> u2  d
        @param edge:
        @return: quadruple of a,b,c,d as a list
        """

        a = edge  # v1->u1
        v1 = a[0]
        u1 = a[1]
        b, c = (None, None)

        for e in self.G.edges():
            if e[0] == v1 and e[1] != u1:  # find the other edge coming from v1
                b = e
                u2 = e[1]
            if e[0] != v1 and e[1] == u1:  # find the other edge entering u2
                c = e
                v2 = e[0]

        if b is None or c is None:
            raise ValueError
        d = (v2, u2)
        return [a, b, c, d]

    def _find_quadruples(self) -> None:  # return a list of all the quadruples
        """
        This method finds all the quadruples and inserts them to a list.
        @rtype: None
        """
        for e in self.G.edges():
            quad = self._create_quadruple(e)
            tmp = self.quadruples.copy()
            t = [q for q in tmp if set(quad) == set(q)]  # maybe there is a better way?
            # we try to find all the matching quadruples to tmp. If it is empty, we add tmp.
            if 0 == len(t):
                self.quadruples.append(quad)

    def _find_decompositions(self) -> None:
        """
        in normal graph, the edges and the vertices have names. in the line graph, only the vertices have names,
        and the names are the names of the edges they came from.
        the number of vertices in the line graph is double the number of vertices in the normal graph because there are 2
         outgoing edges from each vertix in the original one.
        the number quadruples in the line graph, is half the number of vertixes in the line graph -
        so the number of quadruples in the line graph equal to the number
        of the edges in the original graph (n) = half of line_graph's number of vertices.
        :return:
        number of decompositions = 2**(2**self.n)
        """
        # quad = [a b c d] and we need to choose a+d or b+c
        for i in range(0, 2 ** (2 ** self.input_n)):  # index of the decomposition from 0 to 65,536
            bin_i = bin(i)[2:].zfill(2 ** self.input_n)
            current_decomposition = []  # the i-th decomposition
            for quad, j in zip(self.quadruples, bin_i):  # 2^n iterations
                if '0' == j:
                    current_decomposition.append(quad[0])
                    current_decomposition.append(quad[3])
                else:
                    current_decomposition.append(quad[1])
                    current_decomposition.append(quad[2])
            self.decompositions.append(current_decomposition)

    def _is_circle_in_list(self, circle_list, circle) -> bool:
        for i in circle_list:
            if set(i).__eq__(set(circle)):
                return True
        return False

    def _find_decompositions_circles(self) -> None:
        """
        choosing 1 decomposition at a time. decomposition is made out of edges: tuples of (v1,v2), so choosing the first
         edge for the current circle, running on all the edges and every time i add 1 to a circle, remove it from the
          list of edges left. then go to the next circle. until no more edges are left.
        :return:
        sdfas
        """
        for decomp in self.decompositions:  # choosing 1 decomposition
            current_decomposition_with_circles = []
            circles_in_each_length_in_decomposition = [0] * len(decomp)  # creating a list with 0's the size of |E|
            current_decomposition = decomp.copy()  # copying the current decomposition so i can remove edges
            while len(current_decomposition) > 0:  # as long as i didnt finish mapping the edges into circles
                current_circle = [current_decomposition[0]]  # starting to find the circle of the current first edge
                current_decomposition.pop(0)  # removing the edge after adding it to the circle
                current_circle_length = 1
                added_an_edge_to_circle = True
                while added_an_edge_to_circle:  # stop looking for the next edge in circle if we didnt found an edge now
                    added_an_edge_to_circle = False
                    for edge in current_decomposition:
                        if current_circle[-1][1] == edge[0]:
                            current_circle.append(edge)  # add the next edge to the circle
                            current_decomposition.remove(edge)  # remove the edge from list because we added to circle
                            current_circle_length += 1  # increase the length of the current circle in 1
                            added_an_edge_to_circle = True
                            break  # we have found the next edge in the current circle
                # we finished finding the current circle
                current_decomposition_with_circles.append(current_circle)
                # add 1 to the num of circles this length
                circles_in_each_length_in_decomposition[current_circle_length - 1] += 1
                # adds the circle in sorted mode so the set of the specific length will not add it if its already there:
                if not self._is_circle_in_list \
                            (self.unique_circles_in_each_length[current_circle_length - 1], current_circle):
                    self.unique_circles_in_each_length[current_circle_length - 1].append(current_circle)
            # we finished finding all the circles for the current decomposition
            t = (current_decomposition_with_circles, circles_in_each_length_in_decomposition)
            self.decompositions_with_circles.append(t)
        print("done")

    def _all_circles_in_length(self, length) -> int:
        return len(self.unique_circles_in_each_length[length - 1])

    def _print_circle_in_binary(self, circle: list) -> None:  # line_graph printing
        for e in circle:
            u0 = bin(e[0][0])[2:].zfill(self.n - 1)
            u1 = bin(e[0][1])[2:].zfill(self.n - 1)
            u = u0 + u1[-1]
            print(u + ' -->', end=' ')

    # public methods

    def number_of_decompositions_with_only_one_circle(self) -> int:
        counter = 0
        for curr_decomposition_with_circles in self.decompositions_with_circles:
            if curr_decomposition_with_circles[1][-1] == 1:  # if it has a circle in size |E|
                counter += 1
        return counter

    def print_all_circles_in_length(self, length) -> None:
        print("The number of unique circles with length ", length, "is ", self._all_circles_in_length(length),
              "\n And they are:")
        for circle in self.unique_circles_in_each_length[length - 1]:
            self._print_circle_in_binary(circle)

    def print_all_decompositions_circles_sizes(self) -> None:
        for decomp, i in zip(self.decompositions_with_circles, range(len(self.decompositions_with_circles))):
            print("Decomposition #" + str(i + 1).zfill(len(str(2 ** 2 ** self.input_n))) + ": ",
                  print(*decomp[1], sep=", "))

            # Todo: print the dcompositions accordint to the list "input_decompositions_numbers"
            #  from self.decompositions_with_circles[0]
            # Todo: print in an organized manner both print parts in this function
            #  (first one maybe in a table, second maybe show graph)

    def print_decomposition(self, i) -> None:
        print("The circles in decomposition number " + str(i) + " are:")
        for circle in self.decompositions_with_circles[i - 1][0]:
            print(*circle)

    def print_specific_decompositions(self) -> None:
        input_decompositions_numbers = [int(item) for item in input(
            "Please enter, in the next line, all the numbers of the decompositions you wish to print: ").split()]
        for i in input_decompositions_numbers:
            self.print_decomposition(i)


class DeBruijnGraph(Graph):
    def __init__(self, sequence, n):
        """
        :param sequence: the de-bruijn sequence
        :param n: the length of the binary representation
        """
        super().__init__()
        self.n = n  # number of bits of each node
        self.input_n = n
        self.k = n + 1
        self.N = 2 ** self.n  # number of nodes
        kmers = self.get_kmers_from_sequence(sequence, self.k)
        edges = self.get_edges_from_kmers(kmers)
        self.G = nx.DiGraph(edges)

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
        :param i: last n-1 bits of node to perform the exchange operation on.
        """
        # check that there were given n-1 bits
        assert (isinstance(i, str))
        if len(i) != self.n - 1:
            raise ValueError

        N = self.N
        # we convert the binary representation to decimal
        x1 = int(i, 2)  # i '0i'
        # x2 = int('1' + i, 2)  # i+N/2 '1i'
        y1 = (2 * x1) % N  # 2i
        y2 = (2 * x1 + 1) % N  # 2i+1
        z1 = (2 * y1) % N  # 4i
        # z2 = (2 * y1 + 1) % N  # 4i+1
        z3 = (2 * y2) % N  # 4i+2
        # z4 = (2 * y2 + 1) % N  # 4i+3
        # we now find the edges themselves.

        # first we remove 2i->4i and 2i+1->4i+2
        # work around since G doesn't save the nodes with 0b literal - iterating over all edges and comparing.
        for e in self.G.edges():
            if e == (y1, z1):
                # todo: store edge ID
                self.G.remove_edge(*e)
                break
        for e in self.G.edges():
            if e == (y2, z3):
                # todo: store edge ID
                self.G.remove_edge(*e)
                break

        # next we add 2i->4i+2 and 2i+1->4i  # todo: problem, how do we define names for 2i->4i+2? It's not De-Bruijn like.
        edge_to_add = (y1, z3)
        self.G.add_edge(*edge_to_add, id='hello')  # todo: add the id saved
        edge_to_add = (y2, z1)
        self.G.add_edge(*edge_to_add, id='bye')  # todo: add the id saved

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
    n = 2
    seq = create_de_bruijn_sequence(n)
    DBG = DeBruijnGraph(seq, n)
    # DBG.plot('DBG')
    line_graph = LineGraph(DBG)
    # line_graph.plot('line graph before exchange on DBG.')
    print("number_of_decompositions_with_only_one_circle")
    print(line_graph.print_specific_decompositions())
    exit(1)
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

    exit(0)
    DBG.exchange('1')
    line_graph = LineGraph(DBG)
    line_graph.plot('line graph after exchange on DBG.')
