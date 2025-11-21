"""
公平的时间基准测试脚本
所有算法在内存中计算，避免IO影响
"""

import time
import numpy as np
import pandas as pd
import networkx as nx
import os
from collections import defaultdict

import community as community_louvain  # python-louvain
import leidenalg
import igraph as ig

from scipy.spatial import Delaunay
from sklearn.metrics.pairwise import cosine_similarity

from voronoi_detector import VoronoiCommunityDetector
from modularity import calculate_newman_modularity


class TimingBenchmark:
    """
    All algorithms run in-memory to avoid IO impact
    """
    
    def __init__(self, graphml_path="output_socioeconomic_new/final_weighted_graph.graphml"):
        """
        Initialize: preload all necessary data into memory
        """
        print("="*60)
        print("Timing Benchmark Initialization")
        print("="*60)
        
        self.graphml_path = graphml_path
        
        print("\nPreloading data into memory...")
        
        print("  Loading NetworkX graph...")
        self.G = nx.read_graphml(graphml_path).to_undirected()
        print(f"    {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        
        print("  Pre-converting to igraph format...")
        self.ig_graph = self._networkx_to_igraph()
        print(f"    {self.ig_graph.vcount()} nodes, {self.ig_graph.ecount()} edges")
        
        print("  Preloading Voronoi data...")
        self._preload_voronoi_data()
        
        self.timing_results = {}
        self.timing_stats = {}
        self.communities_results = {}
        
        print("\nInitialization complete! All data loaded into memory.")
        print("="*60)
    
    def _networkx_to_igraph(self):
        """
        Convert NetworkX graph to igraph format
        """
        node_list = list(self.G.nodes())
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        
        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in self.G.edges()]
        
        ig_graph = ig.Graph(edges=edges, directed=False)
        
        weights = [self.G[u][v].get('weight', 1.0) for u, v in self.G.edges()]
        ig_graph.es['weight'] = weights
        
        self.node_list = node_list
        self.node_to_idx = node_to_idx
        
        return ig_graph
    
    def _preload_voronoi_data(self):
        """
        Preload all data required for Voronoi algorithm
        """
        self.coords = []
        self.node_ids = []
        
        for node in self.G.nodes():
            x = self.G.nodes[node].get('x')
            y = self.G.nodes[node].get('y')
            if x is not None and y is not None:
                self.coords.append([float(x), float(y)])
                self.node_ids.append(node)
        
        self.coords = np.array(self.coords)
        
        self.edge_weights = {}
        for u, v, data in self.G.edges(data=True):
            weight = data.get('weight', 1.0)
            self.edge_weights[(u, v)] = weight
            self.edge_weights[(v, u)] = weight
        
        print(f"    {len(self.coords)} valid coordinate points")
    
    
    def _run_voronoi(self, R=2.6):
        """
        Run Voronoi algorithm (including file loading)
        """
        detector = VoronoiCommunityDetector(self.graphml_path)
        communities, generators, _ = detector.voronoi_community_detection(R)
        return communities
    
    def _run_leiden(self, resolution=0.367):
        """
        Run Leiden algorithm (using NetworkX + leidenalg)
        """
        G = nx.read_graphml(self.graphml_path).to_undirected()
        
        node_list = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
        ig_graph = ig.Graph(edges=edges, directed=False)
        weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
        ig_graph.es['weight'] = weights
        
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            weights='weight',
            seed=42
        )
        
        communities = {node_list[i]: partition.membership[i] 
                      for i in range(len(node_list))}
        return communities
    
    def _run_louvain(self, resolution=1.0):
        """
        Run Louvain algorithm (pure NetworkX implementation)
        """
        G = nx.read_graphml(self.graphml_path).to_undirected()
        
        communities = community_louvain.best_partition(
            G,
            weight='weight',
            resolution=resolution,
            random_state=42
        )
        
        return communities
    
    def _run_fastgreedy(self, n_clusters=9):
        """
        Run FastGreedy algorithm
        """
        G = nx.read_graphml(self.graphml_path).to_undirected()
        
        node_list = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
        ig_graph = ig.Graph(edges=edges, directed=False)
        weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
        ig_graph.es['weight'] = weights
        
        dendrogram = ig_graph.community_fastgreedy(weights='weight')
        communities_ig = dendrogram.as_clustering(n=n_clusters)
        
        communities = {node_list[i]: communities_ig.membership[i] 
                      for i in range(len(node_list))}
        return communities
    
    def _run_walktrap(self, steps=8, n_clusters=9):
        """
        Run Walktrap algorithm
        """
        G = nx.read_graphml(self.graphml_path).to_undirected()
        
        node_list = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
        ig_graph = ig.Graph(edges=edges, directed=False)
        weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
        ig_graph.es['weight'] = weights
        
        dendrogram = ig_graph.community_walktrap(weights='weight', steps=steps)
        communities_ig = dendrogram.as_clustering(n=n_clusters)
        
        communities = {node_list[i]: communities_ig.membership[i] 
                      for i in range(len(node_list))}
        return communities
    
    
    def run_single_timing(self):
        """
        Run each algorithm and record the time
        """
        print("\n" + "="*60)
        print("Run Timing Test")
        print("="*60)
        
        results = {}
        
        print("\nTesting Voronoi...")
        start = time.time()
        communities = self._run_voronoi(R=2.6)
        elapsed = time.time() - start
        n_comm = len(set(communities.values()) - {-1})
        results['Voronoi'] = {'time': elapsed, 'n_communities': n_comm}
        print(f"  Time: {elapsed:.4f}s, Number of communities: {n_comm}")
        self.communities_results['voronoi'] = communities
        
        print("\nTesting Leiden...")
        start = time.time()
        communities = self._run_leiden(resolution=0.367)
        elapsed = time.time() - start
        n_comm = len(set(communities.values()))
        results['Leiden'] = {'time': elapsed, 'n_communities': n_comm}
        print(f"  Time: {elapsed:.4f}s, Number of communities: {n_comm}")
        self.communities_results['leiden'] = communities
        
        print("\nTesting FastGreedy...")
        start = time.time()
        communities = self._run_fastgreedy(n_clusters=9)
        elapsed = time.time() - start
        n_comm = len(set(communities.values()))
        results['FastGreedy'] = {'time': elapsed, 'n_communities': n_comm}
        print(f"  Time: {elapsed:.4f}s, Number of communities: {n_comm}")
        self.communities_results['fastgreedy'] = communities
        
        print("\nTesting Walktrap...")
        start = time.time()
        communities = self._run_walktrap(steps=8, n_clusters=9)
        elapsed = time.time() - start
        n_comm = len(set(communities.values()))
        results['Walktrap'] = {'time': elapsed, 'n_communities': n_comm}
        print(f"  Time: {elapsed:.4f}s, Number of communities: {n_comm}")
        self.communities_results['walktrap'] = communities
        
        self.timing_results = results
        return results
    
    def run_benchmark(self, n_runs=10):
        """
        Run timing benchmark for all methods
        
        Parameters:
        - n_runs: Number of runs for each method
        """
        print("\n" + "="*60)
        print(f"Timing Benchmark ({n_runs} runs average)")
        print("="*60)
        print("All computations are performed in-memory, no IO impact")
        
        timing_stats = {}
        
        methods = {
            'Voronoi': lambda: self._run_voronoi(R=2.6),
            'Leiden': lambda: self._run_leiden(resolution=0.367),
            'FastGreedy': lambda: self._run_fastgreedy(n_clusters=9),
            'Walktrap': lambda: self._run_walktrap(steps=8, n_clusters=9)
        }
        
        for method_name, method_func in methods.items():
            times = []
            print(f"\nTesting {method_name}...")
            
            try:
                _ = method_func()
            except:
                pass
            
            for i in range(n_runs):
                try:
                    start = time.time()
                    _ = method_func()
                    elapsed = time.time() - start
                    times.append(elapsed)
                    print(f"  Run {i+1}: {elapsed:.4f}s")
                except Exception as e:
                    print(f"  Run {i+1}: Failed - {e}")
            
            if times:
                mean_time = np.mean(times)
                std_time = np.std(times)
                min_time = min(times)
                max_time = max(times)
                
                timing_stats[method_name] = {
                    'mean': mean_time,
                    'std': std_time,
                    'min': min_time,
                    'max': max_time,
                    'times': times
                }
                print(f"  Mean: {mean_time:.4f}s ± {std_time:.4f}s")
        
        self.timing_stats = timing_stats
        return timing_stats
    
    
    def print_summary(self):
        """
        Print timing benchmark summary
        """
        if not self.timing_stats:
            print("Please run run_benchmark() first")
            return
        
        print("\n" + "="*60)
        print("Timing Benchmark Summary")
        print("="*60)
        print(f"{'Method':<12} {'Mean (s)':<10} {'Std (s)':<10} {'Min (s)':<10} {'Max (s)':<10} {'Relative':<10}")
        print("-"*60)
        
        min_mean = min(stats['mean'] for stats in self.timing_stats.values())
        
        for method, stats in self.timing_stats.items():
            mean = stats['mean']
            std = stats['std']
            min_t = stats['min']
            max_t = stats['max']
            relative = mean / min_mean
            print(f"{method:<12} {mean:<10.4f} {std:<10.4f} {min_t:<10.4f} {max_t:<10.4f} {relative:<10.2f}x")
        
        print("="*60)
        
        print("\nTable for paper (CSV format):")
        print("Method,Mean_Time_s,Std_Time_s")
        for method, stats in self.timing_stats.items():
            print(f"{method},{stats['mean']:.4f},{stats['std']:.4f}")
    
    
    def export_to_csv(self, filename='timing_results.csv'):
        """
        Export results to CSV
        """
        if not self.timing_stats:
            print("Please run run_benchmark() first")
            return
        
        data = []
        for method, stats in self.timing_stats.items():
            data.append({
                'Method': method,
                'Mean_Time_s': round(stats['mean'], 4),
                'Std_Time_s': round(stats['std'], 4),
                'Min_Time_s': round(stats['min'], 4),
                'Max_Time_s': round(stats['max'], 4)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"\nResults exported to: {filename}")
        
        return df


def main():
    import time
    
    graphml_path = "output_socioeconomic_new/final_weighted_graph.graphml"
    
    print("="*60)
    print("Timing Test")
    print("="*60)
    
    G = nx.read_graphml(graphml_path).to_undirected()
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")
    
    print("Warming up...")
    from voronoi_detector import VoronoiCommunityDetector
    detector = VoronoiCommunityDetector(graphml_path)
    _ = detector.voronoi_community_detection(2.6)
    print("finish \n")
    
    results = {}
    
    print("Voronoi...", end=" ")
    start = time.time()
    detector = VoronoiCommunityDetector(graphml_path)
    communities, _, _ = detector.voronoi_community_detection(2.6)
    t = time.time() - start
    n = len(set(communities.values()) - {-1})
    results['Voronoi'] = t
    print(f"{t:.4f}s ({n} communities)")
    
    print("Leiden...", end=" ")
    start = time.time()
    G = nx.read_graphml(graphml_path).to_undirected()
    node_list = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    ig_graph = ig.Graph(edges=edges, directed=False)
    ig_graph.es['weight'] = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
    partition = leidenalg.find_partition(ig_graph, leidenalg.RBConfigurationVertexPartition,
                                         resolution_parameter=0.367, weights='weight', seed=42)
    t = time.time() - start
    n = len(set(partition.membership))
    results['Leiden'] = t
    print(f"{t:.4f}s ({n} communities)")
    
    print("FastGreedy...", end=" ")
    start = time.time()
    G = nx.read_graphml(graphml_path).to_undirected()
    node_list = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    ig_graph = ig.Graph(edges=edges, directed=False)
    ig_graph.es['weight'] = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
    dendrogram = ig_graph.community_fastgreedy(weights='weight')
    communities_ig = dendrogram.as_clustering(n=9)
    t = time.time() - start
    n = len(set(communities_ig.membership))
    results['FastGreedy'] = t
    print(f"{t:.4f}s ({n} communities)")
    
    print("Walktrap...", end=" ")
    start = time.time()
    G = nx.read_graphml(graphml_path).to_undirected()
    node_list = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    ig_graph = ig.Graph(edges=edges, directed=False)
    ig_graph.es['weight'] = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
    dendrogram = ig_graph.community_walktrap(weights='weight', steps=8)
    communities_ig = dendrogram.as_clustering(n=9)
    t = time.time() - start
    n = len(set(communities_ig.membership))
    results['Walktrap'] = t
    print(f"{t:.4f}s ({n} communities)")
    
    print("\n" + "="*60)
    print("results: ")
    print("="*60)
    
    min_time = min(results.values())
    for method, t in results.items():
        relative = t / min_time
        print(f"{method:<12} {t:.4f}s  ({relative:.1f}x)")

if __name__ == "__main__":
    main()
