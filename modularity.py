"""
Module for calculating Newman modularity of a graph given community assignments.
"""

import networkx as nx

def calculate_newman_modularity(G, communities):
    import networkx as nx
    from networkx.algorithms import community

    weight_key = 'strength'

    if weight_key not in next(iter(G.edges(data=True)))[2]:
        for u, v, data in G.edges(data=True):
            d = float(data.get('weight', 1.0))        # distance
            data['strength'] = 1.0 / (d )        # strong-tie
    
    unique = set(communities.values()) - {-1}
    partition = [{n for n,c in communities.items() if c==cid} for cid in unique]

    return community.modularity(G, partition, weight=weight_key)

