# Custom Graph error
class GraphError(Exception): pass


class GraphSolution:
    """
    Graph Class ADT
    """

    class Edge:
        """
        Class representing an Edge in the Graph
        """
        __slots__ = ['source', 'destination']

        def __init__(self, source, destination):
            """
            DO NOT EDIT THIS METHOD!
            Class representing an Edge in a graph
            :param source: Vertex where this edge originates
            :param destination: ID of Vertex where this edge ends
            """
            self.source = source
            self.destination = destination

        def __eq__(self, other):
            return self.source == other.source and self.destination == other.destination

        def __repr__(self):
            return f"Source: {self.source} Destination: {self.destination}"

            __str__ = __repr__

    class Path:
        """
        Class representing a Path through the Graph
        """
        __slots__ = ['vertices']

        def __init__(self, vertices=[]):
            """
            DO NOT EDIT THIS METHOD!
            Class representing a path in a graph
            :param vertices: Ordered list of vertices that compose the path
            """
            self.vertices = vertices

        def __eq__(self, other):
            return self.vertices == other.vertices

        def __repr__(self):
            return f"Path: {' -> '.join([str(v) for v in self.vertices])}\n"

        __str__ = __repr__

        def add_vertex(self, vertex):
            """
            Adds vertex ID to list of vertices.
            :param vertex: ID to add.
            :return: Nothing.
            """
            vx = GraphSolution.Vertex(vertex)
            self.vertices.append(vx.ID)

        def remove_vertex(self):
            """
            Removes vertex most recently added to list.
            :return: Nothing
            """
            if len(self.vertices):
                self.vertices.pop()

        def last_vertex(self):
            """
            Retrieves the vertex at the end of the path.
            :return: ID of last vertex in path.
            """
            return self.vertices[len(self.vertices) - 1]

        def is_empty(self):
            """
            Indicates if the path is empty or not.
            :return bool: True if empty, false otherwise.
            """
            return not len(self.vertices)

    class Vertex:
        """
        Class representing a Vertex in the Graph
        """
        __slots__ = ['ID', 'edges', 'pred_vertex', 'distance']

        def __init__(self, ID):
            """
            Class representing a vertex in the graph
            :param ID : Unique ID of this vertex
            """
            self.ID = ID
            self.edges = []
            self.pred_vertex = 0
            self.distance = 0

        def __repr__(self):
            return f"Vertex: {self.ID}"

        __str__ = __repr__

        def __eq__(self, other):
            """
            DO NOT EDIT THIS METHOD
            :param other: Vertex to compare
            :return: Bool, True if same, otherwise False
            """
            if self.ID == other.ID :
                if len(self.edges) == len(other.edges):
                    edges = set((edge.source.ID, edge.destination) for edge in self.edges)
                    difference = [e for e in other.edges if (e.source.ID, e.destination) not in edges]
                    if len(difference) > 0:
                        return False
                    return True

        def add_edge(self, destination):
            """
            Adds a vertex to the list of adjacent edges.
            :param destination: ID of vertex to add as an edge.
            :return: Nothing.
            """
            ex = GraphSolution.Edge(self, destination)
            self.edges.append(ex)

        def degree(self):
            """
            Returns degree (no. of edges) of vertex.
            :return: Nothing.
            """
            return len(self.edges)

        def get_edge(self, destination):
            """
            Retrieve edge with vertex ID == destination.
            :param destination: ID of the vertex to retrieve.
            :return: None if vertex does not have this edge.
            """
            for ex in range(len(self.edges)):
                if self.edges[ex].destination == destination:
                    return self.edges[ex]
            return None

        def get_edges(self):
            """
            Retrieve the list of edges for this vertex.
            :return: list of edges.
            """
            return self.edges

    def __init__(self, size=0):
        """
        DO NOT EDIT THIS METHOD
        Construct a random DAG
        :param size: Number of vertices
        :param adj_list: List of vertices.
        """
        self.adj_list = {}
        self.size = size

    def __eq__(self, other):
        """
        DO NOT EDIT THIS METHOD
        Determines if 2 graphs are Identical
        :param other: Graph Object
        :return: Bool, True if Graph objects are equal
        """
        if len(self.adj_list) == len(other.adj_list):
            for key in self.adj_list.items():
                if key in other.adj_list:
                    if not self.adj_list[key] == other.adj_list[key]:
                        return False
                else:
                    return False
            return True
        return False

    def get_vertex(self, ID):
        """
        Find and return requested vertex from adj_list.
        :param ID: ID of vertex to return
        :return: vertex ID, none if doesnt exist
        """
        if self.adj_list[ID]:
            return self.adj_list[ID]
        return None

    def construct_edge(self, data):
        """
        Construct a graph obj using file if filename is
        provided, otherwise construct it randomly.
        :return: Nothing
        """
        src = GraphSolution.Vertex(data[0])
        dest = GraphSolution.Vertex(data[1])
        if src.ID in self.adj_list:
            self.adj_list[src.ID].add_edge(data[1])
        else:
            self.adj_list[src.ID] = src
            self.adj_list[src.ID].add_edge(data[1])
        if dest.ID not in self.adj_list:
            self.adj_list[dest.ID] = dest

    def bellman_ford(self):
        # Initialize all vertex distances to infinity and
        # and predecessor vertices to None.
        for k, current_vertex in self.adj_list.items():
            current_vertex.distance = float('inf')  # Infinity
            current_vertex.pred_vertex = None

        for k, first_vertex in self.adj_list.items():
            first_vertex.distance = 0   # start at very first vertex
            break

        # Main loop is executed |V|-1 times to guarantee minimum distances.
        for i in range(len(self.adj_list) - 1):
            # The main loop.
            for k, current_vertex in self.adj_list.items():
                for e in range(len(current_vertex.edges)):
                    alternative_path_distance = current_vertex.distance
                    adj_vertex = self.get_vertex(current_vertex.edges[e].destination)
                    # If shorter path from start_vertex to adj_vertex is found,
                    # update adj_vertex's distance and predecessor
                    if alternative_path_distance < adj_vertex.distance:
                        adj_vertex.distance = alternative_path_distance
                        adj_vertex.pred_vertex = current_vertex
        return True

    def get_shortest_path(self, startID, endID):
        valid = self.get_vertex(startID)
        if valid is None:
            raise GraphError("Invalid ID")
        valid = self.get_vertex(endID)
        if valid is None:
            raise GraphError("Invalid ID")
        # call bellman ford
        self.bellman_ford()
        # Start from end_vertex and build the path backwards.
        p = GraphSolution.Path()
        start_vertex = self.get_vertex(startID)
        end_vertex = self.get_vertex(endID)
        backwards = []
        current_vertex = end_vertex
        backwards.append(current_vertex)
        distance = 0.0
        if start_vertex is None:
            return p  # empty return path
        while current_vertex.ID is not start_vertex.ID:
            current_vertex = current_vertex.pred_vertex
            if current_vertex is None:
                return p # empty return path
            backwards.append(current_vertex)
        for item in range((len(backwards)-1), 0, -1):
            p.vertices.append(backwards[item])
        p.vertices.append(backwards[0])
        return p