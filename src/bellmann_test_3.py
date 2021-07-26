from collections import deque
from networkx.utils import generate_unique_node
import networkx as nx
import pandas as pd


def negative_edge_cycle(G, weight='weight'):

    newnode = generate_unique_node()
    G.add_edges_from([(newnode, n) for n in G])

    try:
        pred, dist, negative_cycle_end = bellman_ford_tree(G, newnode, weight)

        if negative_cycle_end:
            nodes = []
            negative_cycle = True
            end = negative_cycle_end
            while True:
                nodes.insert(0, end)
                if nodes.count(end) > 1:
                    end_index = nodes[1:].index(end) + 2
                    nodes = nodes[:end_index]
                    break
                end = pred[end]
            length = sum(
                G[u][v].get(weight, 1) for (u, v) in zip(nodes, nodes[1:])
            )
        else:
            nodes = None
            negative_cycle = False
            length = None

        return length, nodes, negative_cycle
    finally:
        G.remove_node(newnode)


def bellman_ford(G, source, target, weight='weight'):

    # Get shortest path tree
    pred, dist, negative_cycle_end = bellman_ford_tree(G, source, weight)

    nodes = []

    if negative_cycle_end:
        negative_cycle = True
        end = negative_cycle_end
        while True:
            nodes.insert(0, end)
            if nodes.count(end) > 1:
                end_index = nodes[1:].index(end) + 2
                nodes = nodes[:end_index]
                break
            end = pred[end]
    else:
        negative_cycle = False
        end = target
        while True:
            nodes.insert(0, end)
            # If end has no predecessor
            if pred.get(end, None) is None:
                # If end is not s, then there is no s-t path
                if end != source:
                    nodes = []
                break
            end = pred[end]
    #
    # if nodes:
    #     length = sum(
    #         G[u][v].get(weight, 1) for (u, v) in zip(nodes, nodes[1:])
    #     )
    # else:
    #     length = float('inf')

    return  nodes


def bellman_ford_tree(G, source, weight='weight'):

    if source not in G:
        raise KeyError("Node %s is not found in the graph" % source)

    dist = {source: 0}
    pred = {source: None}

    return _bellman_ford_relaxation(G, pred, dist, [source], weight)


def _bellman_ford_relaxation(G, pred, dist, source, weight):

    if G.is_multigraph():
        def get_weight(edge_dict):
            return min(eattr.get(weight, 1) for eattr in edge_dict.values())
    else:
        def get_weight(edge_dict):
            return edge_dict.get(weight, 1)

    G_succ = G.succ if G.is_directed() else G.adj
    inf = float('inf')
    n = len(G)

    count = {}
    q = deque(source)
    in_q = set(source)
    while q:
        u = q.popleft()
        in_q.remove(u)

        # Skip relaxations if the predecessor of u is in the queue.
        if pred[u] not in in_q:
            dist_u = dist[u]

            for v, e in G_succ[u].items():
                dist_v = dist_u + get_weight(e)

                if dist_v < dist.get(v, inf):
                    dist[v] = dist_v
                    pred[v] = u

                    if v not in in_q:
                        q.append(v)
                        in_q.add(v)
                        count_v = count.get(v, 0) + 1

                        if count_v == n:
                            return dist

                        count[v] = count_v

    return  dist



class Airports:
    data = pd.read_csv('airports.csv')

    G = nx.from_pandas_edgelist(data, source='Origin', target='Dest', edge_attr=True, create_using=nx.DiGraph)

    print("=====================================")
    print("AIRPORTS:")
    print("=====================================")

    # 3.2.2. Find the shortest path with respect to the distance (column Distance of
    # the dataset) from ’CRP’ to ’BOI’ and vice versa
    print("=====================================")

    print("First case: respect to distance ")
    print("Shortest path from CRP to BOI:")
    print("1 - Output of My Function::")
    print(bellman_ford(G, 'CRP', 'BOI', 'Distance'))
    print(nx.bellman_ford_path(G, 'CRP', 'BOI', 'Distance'))



