from pqdict import PQDict
import pandas as pd
import networkx as nx
import time


# 1. Implement the Bellman-Ford algorithm
def bellman_ford(G, weight, start, end=None):
    inf = float('inf')
    nodes = list(G.nodes())
    edges = G.edges()
    path = []

    # getting indices of start and end nodes' in nodes list
    start_index = nodes.index(start, 0, len(nodes))
    end_index = nodes.index(end, 0, len(nodes))

    # lists for holding weights of edges and predecessors of nodes
    dist = list()
    pred = list()

    # initialize lists
    for i in range(0, len(nodes)):
        dist.append(inf)
        pred.append(nodes[i])

    # edge weight from node to itself is 0
    dist[start_index - 1] = 0

    # Implementing BELLMAN_FORD algorithm
    for _ in nodes:
        for [u, v] in edges:
            v_index = nodes.index(v, 0, len(nodes))
            u_index = nodes.index(u, 0, len(nodes))
            if dist[v_index - 1] > (dist[u_index - 1] + G.edges[u, v][weight]):
                dist[v_index - 1] = dist[u_index - 1] + G.edges[u, v][weight]
                pred[v_index - 1] = u

    # if there is no path to end
    if dist[end_index] == inf:
        return
        pass
    # we start to appending nodes to path with destination(end)
    path.append(end)

    # if did not reach start point, add  predecessor to path
    while start not in path:
        temp = nodes.index(path[-1], 0, len(nodes))
        path.append(pred[temp - 1])

    # reverse path, because we found path from destination to start. But we need vice versa
    path.reverse()
    return path


# 2. Implement the Dijkstra’s algorithm
def dijkstra(G, weight, start, end=None):
    inf = float('inf')
    # mapping of nodes to their dist from start
    dist = {start: 0}
    # priority queue for tracking min shortest path
    queue = PQDict(dist)
    # mapping of nodes to their direct candidates
    candidates = {}
    # unvisited nodes
    unvisited_nodes = set(G.nodes())

    # while there is unvisited node
    while unvisited_nodes:
        (v, d) = queue.popitem()
        dist[v] = d
        unvisited_nodes.remove(v)
        if v == end: break

        for w in G[v]:
            if w in unvisited_nodes:
                d = dist[v] + G[v][w][weight]
                if d < queue.get(w, inf):
                    queue[w] = d
                    candidates[w] = v

    node = end
    path = [node]
    while node != start:
        node = candidates[node]
        path.append(node)
    path.reverse()
    return path


class Airports:
    data = pd.read_csv('airports.csv')

    G = nx.from_pandas_edgelist(data, source='Origin', target='Dest', edge_attr=True, create_using=nx.DiGraph)

    print("=====================================")

    # 3. Find the shortest path with respect to the distance (column Distance of
    # the dataset) from ’CRP’ to ’BOI’ and vice versa with both Bellman-Ford
    # and Dijkstra’s algorithms; compare their performance

    print("First case: respect to DISTANCE ")
    print("=====================================")

    print("Shortest path from CRP to BOI:")

    start_time = time.time()
    print("Shortest path with Bellman-Ford Algorithm: ", bellman_ford(G, 'Distance', 'CRP', 'BOI'))
    end_time = time.time()
    time_for_bellman = end_time - start_time

    start_time = time.time()
    print("Shortest path with Dijkstra Algorithm: ", dijkstra(G, 'Distance', 'CRP', 'BOI'))
    end_time = time.time()
    time_for_dijkstra = end_time - start_time
    print("=====================================")
    print("Shortest path from BOI to CRP:")
    print("Shortest path with Bellman-Ford Algorithm: ", bellman_ford(G, 'Distance', 'BOI', 'CRP'))
    print("Shortest path with Dijkstra Algorithm: ", dijkstra(G, 'Distance', 'BOI', 'CRP'))
    print("=====================================")

    # 4. Find the shortest path with respect to the time (column AirTime of the dataset)
    # from ’CRP’ to ’BOI’ and vice versa with both Bellman-Ford and Dijkstra’s algorithms
    # Compare their performance

    print("=====================================")

    print("Second case: respect to AIR TIME ")
    print("=====================================")

    print("Shortest path from CRP to BOI:")
    print("Shortest path with Bellman-Ford Algorithm: ", bellman_ford(G, 'AirTime', 'CRP', 'BOI'))
    print("Shortest path with Dijkstra Algorithm: ", dijkstra(G, 'AirTime', 'CRP', 'BOI'))

    print("=====================================")
    print("Shortest path from BOI to CRP:")
    print("Shortest path with Bellman-Ford Algorithm: ", bellman_ford(G, 'AirTime', 'BOI', 'CRP'))
    # print("Shortest path with Dijkstra Algorithm: ", dijkstra(G, 'AirTime', 'BOI', 'CRP')) #in this case, there is 'pqdict is empty' error
    print("=====================================")
    print("CONCLUSION: According to the algorithms, after comparing the run times, it was determined that\n"
          "            Dijkstra is faster than Bellman-Ford approximately 100 times")
    print("=====================================")
    print("Results:")
    print("Bellman-Ford run time: ", time_for_bellman.__round__(5))
    print("Dijkstra run time:     ", time_for_dijkstra.__round__(5))
    print("=====================================")
