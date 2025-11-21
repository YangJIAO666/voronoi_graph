import pandas as pd
import numpy as np
import networkx as nx
import os
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import community as community_louvain  # python-louvain
import leidenalg 
import igraph as ig
import networkx.algorithms.community as nx_community


from voronoi_detector import VoronoiCommunityDetector

from modularity import calculate_newman_modularity

class TargetedCommunityEvaluation:
    """
    evaluation_fixed_communities.py
    only evaluate: IFR, Cosine Similarity, Modularity, Normalized Inequality
    Compare 4 methods: Voronoi, Leiden, Walktrap, FastGreedy
    """
    
    def __init__(self, graphml_path=r"output_socioeconomic_new\final_weighted_graph.graphml"):
        """
        Initialize the evaluator
        """
        print("="*60)
        print("Initializing evaluator for 9 fixed communities")
        print("="*60)
        
        self.graphml_path = graphml_path
        self.G = self._load_graph()
        self.communities_results = {}  # Store community results from different methods
        self.evaluation_results = {}   # Store evaluation results
        self.target_communities = (9, 9)  # Target number of communities: fixed at 9
        
    def _load_graph(self):
        """
        Load GraphML file
        """
        if not os.path.exists(self.graphml_path):
            raise FileNotFoundError(f"GraphML file not found: {self.graphml_path}")
            
        print(f"Loading GraphML file: {self.graphml_path}")
        
        G = nx.read_graphml(self.graphml_path)
        G = G.to_undirected()
        
        print(f"Successfully loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def run_voronoi_detection_fixed(self):
        """
        Using fixed parameter R=2.6 to run Voronoi method (corresponding to 9 communities)
        """
        print(f"\nVoronoi method using fixed parameter R=2.6 (corresponding to 9 communities)...")
        
        detector = VoronoiCommunityDetector(self.graphml_path)
        fixed_radius = 2.6
        
        try:
            communities, generators, R = detector.voronoi_community_detection(fixed_radius)
            n_communities = len(set(communities.values()) - {-1})
            
            modularity = calculate_newman_modularity(self.G, communities)
            
            print(f"  R={fixed_radius:.4f}: {n_communities} communities, modularity={modularity:.4f}")
            
            self.communities_results['voronoi'] = {
                'communities': communities,
                'modularity': modularity,
                'n_communities': n_communities,
                'radius': fixed_radius
            }
            
            print(f"Voronoi result: R={fixed_radius:.4f}, {n_communities} communities, modularity={modularity:.4f}")
            return communities
            
        except Exception as e:
            print(f"Voronoi method failed: {e}")
            return {}

    
    def run_leiden_method_search(self):
        """
        Search for the best parameter in Leiden method within the target community number range
        """
        print(f"\nSearching for the best parameter in Leiden method to obtain {self.target_communities[0]} communities...")
        
        resolution_range = np.linspace(0.1, 0.5, 4)
        best_result = None
        best_modularity = -1
        
        for resolution in resolution_range:
            try:
                ig_graph = self._networkx_to_igraph()
                
                partition = leidenalg.find_partition(
                    ig_graph,
                    leidenalg.RBConfigurationVertexPartition,
                    resolution_parameter=resolution,
                    weights='weight',
                    seed=42
                )
                
                communities = {}
                node_list = list(self.G.nodes())
                for i, community_id in enumerate(partition.membership):
                    communities[node_list[i]] = community_id
                
                n_communities = len(set(communities.values()))
                
                if n_communities == self.target_communities[0]:
                    modularity = self._calculate_modularity_networkx(communities)
                    
                    print(f"  Resolution={resolution:.3f}: {n_communities} communities, modularity={modularity:.4f}")
                    
                    if modularity > best_modularity:
                        best_modularity = modularity
                        best_result = {
                            'communities': communities,
                            'modularity': modularity,
                            'n_communities': n_communities,
                            'resolution': resolution
                        }
            except Exception as e:
                continue
        
        if best_result:
            self.communities_results['leiden'] = best_result
            print(f"Leiden best result: Resolution={best_result['resolution']:.3f}, {best_result['n_communities']} communities, modularity={best_result['modularity']:.4f}")
            return best_result['communities']
        else:
            print("Leiden method did not find a suitable result")
            return {}
    
    def run_walktrap_method_search(self):
        """
        Search for the best parameter in Walktrap method within the target community number range
        """
        print(f"\nSearching for the best parameter in Walktrap method to obtain {self.target_communities[0]} communities...")
        
        steps_range = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        best_result = None
        best_modularity = -1
        
        for steps in steps_range:
            try:
                ig_graph = self._networkx_to_igraph()
                
                dendrogram = ig_graph.community_walktrap(weights='weight', steps=steps)
                
                for n_clusters in [self.target_communities[0]]:
                    try:
                        communities_ig = dendrogram.as_clustering(n=n_clusters)
                        
                        node_list = list(self.G.nodes())
                        communities = {node_list[i]: communities_ig.membership[i] for i in range(len(node_list))}
                        
                        modularity = self._calculate_modularity_networkx(communities)
                        
                        print(f"  Steps={steps}, Clusters={n_clusters}: modularity={modularity:.4f}")
                        
                        if modularity > best_modularity:
                            best_modularity = modularity
                            best_result = {
                                'communities': communities,
                                'modularity': modularity,
                                'n_communities': n_clusters,
                                'steps': steps
                            }
                    except Exception as e:
                        continue
                        
            except Exception as e:
                continue
        
        if best_result:
            self.communities_results['walktrap'] = best_result
            print(f"Walktrap best result: Steps={best_result['steps']}, {best_result['n_communities']} communities, modularity={best_result['modularity']:.4f}")
            return best_result['communities']
        else:
            print("Walktrap method did not find a suitable result")
            return {}
    
    def run_fastgreedy_method_search(self):
        """
        Search for the best parameter in FastGreedy method within the target community number range
        """
        print(f"\nSearching for the best parameter in FastGreedy method to obtain {self.target_communities[0]} communities...")
        
        best_result = None
        best_modularity = -1
        
        try:
            ig_graph = self._networkx_to_igraph()
            dendrogram = ig_graph.community_fastgreedy(weights='weight')
            
            for n_clusters in [self.target_communities[0]]:
                try:
                    communities_ig = dendrogram.as_clustering(n=n_clusters)
                    
                    node_list = list(self.G.nodes())
                    communities = {node_list[i]: communities_ig.membership[i] for i in range(len(node_list))}
                    
                    modularity = self._calculate_modularity_networkx(communities)
                    
                    print(f"  Clusters={n_clusters}: modularity={modularity:.4f}")
                    
                    if modularity > best_modularity:
                        best_modularity = modularity
                        best_result = {
                            'communities': communities,
                            'modularity': modularity,
                            'n_communities': n_clusters
                        }
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"FastGreedy method execution failed: {e}")
        
        if best_result:
            self.communities_results['fastgreedy'] = best_result
            print(f"FastGreedy best result: {best_result['n_communities']} communities, modularity={best_result['modularity']:.4f}")
            return best_result['communities']
        else:
            print("FastGreedy method did not find a suitable result")
            return {}
    
    def _networkx_to_igraph(self):
        """
        Convert NetworkX graph to igraph format
        Convert weights: distance semantics → similarity semantics
        """
        ig_graph = ig.Graph()
        
        node_list = list(self.G.nodes())
        ig_graph.add_vertices(len(node_list))
        
        edges = []
        strength = []
        node_to_index = {node: i for i, node in enumerate(node_list)}
        
        for u, v, data in self.G.edges(data=True):
            edges.append((node_to_index[u], node_to_index[v]))
            distance_weight = data.get('weight', 1.0)
            similarity_weight = 1.0 / (distance_weight)
            strength.append(similarity_weight)
        
        ig_graph.add_edges(edges)
        ig_graph.es['weight'] = strength
        
        return ig_graph
    
    def _calculate_modularity_networkx(self, communities):

       
       
        modularity = calculate_newman_modularity(self.G, communities)
        return modularity
    def calculate_intra_flow_ratio(self, communities, method_name):
        """
        Calculate Intra-Flow Ratio
        Using similarity weights: high similarity within communities = high IFR = good
        """
        print(f"Calculating {method_name} Intra-Flow Ratio...")
        
        total_internal_weight = 0
        total_weight = 0
        
        valid_communities = set(communities.values()) - {-1}
        
        for u, v, data in self.G.edges(data=True):
            distance_weight = float(data.get('weight', 1.0))
            similarity_weight = 1.0 / (distance_weight)
            
            total_weight += similarity_weight
            
            if (u in communities and v in communities and 
                communities[u] == communities[v] and 
                communities[u] in valid_communities):
                total_internal_weight += similarity_weight
        
        global_ifr = total_internal_weight / total_weight if total_weight > 0 else 0
        
        print(f"{method_name} Intra-Flow Ratio: {global_ifr:.4f}")
        return global_ifr
    
    def calculate_normalized_inequality(self, communities, method_name):
        """
        Calculate Normalized Inequality
        """
        print(f"Calculating {method_name} Normalized Inequality...")
        
        community_sizes = []
        valid_communities = set(communities.values()) - {-1}
        
        for comm_id in valid_communities:
            size = sum(1 for cid in communities.values() if cid == comm_id)
            community_sizes.append(size)
        
        if len(community_sizes) <= 1:
            return 0.0
        
        mean_size = np.mean(community_sizes)
        std_size = np.std(community_sizes)
        
        cv = std_size / mean_size if mean_size > 0 else 0
        max_possible_cv = np.sqrt(len(valid_communities) - 1)
        normalized_inequality = min(cv / max_possible_cv, 1.0) if max_possible_cv > 0 else 0
        
        print(f"{method_name} Normalized Inequality: {normalized_inequality:.4f}")
        return normalized_inequality
    
    def calculate_community_distinction(self, communities, method_name):
        """Community Distinction
        calculate_cosine_similarity
        Calculate Cosine Similarity
        Feature vector includes: brand distribution + geographic features + socio-economic features
        """
        print(f"Calculating {method_name} Community Distinction...")

        valid_communities = sorted(set(communities.values()) - {-1})
        n_communities = len(valid_communities)

        if n_communities <= 1:
            return 0.0

        community_features = []
        all_brands = sorted(set(self.G.nodes[n].get('brand', 'unknown') for n in self.G.nodes()))

        for comm_id in valid_communities:
            comm_nodes = [n for n, cid in communities.items() if cid == comm_id]
            
            if len(comm_nodes) == 0:
                continue
                
            brand_counts = {b: 0 for b in all_brands}
            for node in comm_nodes:
                brand = self.G.nodes[node].get('brand', 'unknown')
                brand_counts[brand] += 1
            
            total_nodes = len(comm_nodes)
            brand_vector = [brand_counts[b] / total_nodes for b in all_brands]
            
            coords = []
            for node in comm_nodes:
                x = float(self.G.nodes[node].get('x', 0))
                y = float(self.G.nodes[node].get('y', 0))
                coords.append([x, y])
            
            coords = np.array(coords)
            centroid_x = coords[:, 0].mean()
            centroid_y = coords[:, 1].mean()
            
            geo_spread = np.sqrt(((coords[:, 0] - centroid_x) ** 2 + 
                                (coords[:, 1] - centroid_y) ** 2)).mean()
            
            pop_d_total = 0
            family_inc_total = 0
            family_out_total = 0
            night_total = 0
            
            for node in comm_nodes:
                pop_d_total += float(self.G.nodes[node].get('pop_d', 0))
                family_inc_total += float(self.G.nodes[node].get('family_inc', 0))
                family_out_total += float(self.G.nodes[node].get('family_out', 0))
                night_total += float(self.G.nodes[node].get('night', 0))
            
            avg_pop_d = pop_d_total / total_nodes
            avg_family_inc = family_inc_total / total_nodes
            avg_family_out = family_out_total / total_nodes
            avg_night = night_total / total_nodes
            
            feature_vector = (brand_vector + 
                            [centroid_x, centroid_y, geo_spread] + 
                            [avg_pop_d, avg_family_inc, avg_family_out, avg_night])
            
            community_features.append(feature_vector)

        if len(community_features) <= 1:
            return 0.0

        community_features = np.array(community_features)
        
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        community_features_scaled = scaler.fit_transform(community_features)
        
        similarity_matrix = cosine_similarity(community_features_scaled)
        
        mask = np.ones_like(similarity_matrix, dtype=bool)
        np.fill_diagonal(mask, False)
        avg_similarity = similarity_matrix[mask].mean()
        
        print(f"{method_name} Cosine Similarity: {avg_similarity:.4f}")
        return 1-avg_similarity
    
    
    
    def evaluate_all_methods(self):
        """
        Evaluate all methods, Voronoi uses fixed parameter R=2.6938, other methods search for the best parameters
        """
        print(f"Starting evaluation of all methods, target number of communities: {self.target_communities[0]}...")
        
        self.run_voronoi_detection_fixed()
        self.run_leiden_method_search()
        self.run_walktrap_method_search()
        self.run_fastgreedy_method_search()
        
        for method, result in self.communities_results.items():
            if 'communities' not in result:
                continue
                
            communities = result['communities']
            
            print(f"\n{'='*40}")
            print(f"Evaluating method: {method.upper()}")
            print(f"{'='*40}")
            
            ifr = self.calculate_intra_flow_ratio(communities, method)
            inequality = self.calculate_normalized_inequality(communities, method)
            community_distinction = self.calculate_community_distinction(communities, method)
            modularity = result['modularity']
            
            self.evaluation_results[method] = {
                'intra_flow_ratio': ifr,
                'normalized_inequality': inequality,
                'community_distinction': community_distinction,
                'modularity': modularity,
                'n_communities': result['n_communities']
            }
        
        return self.evaluation_results
    
    def create_evaluation_plots(self, output_dir="results_9communities_evaluation"):
        """
        Create comparison plots for the four evaluation metrics
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.evaluation_results:
            print("Please run evaluate_all_methods() first!")
            return
        
        methods = list(self.evaluation_results.keys())
        method_labels = [m.capitalize() for m in methods]
        
        ifr_values = [self.evaluation_results[m]['intra_flow_ratio'] for m in methods]
        inequality_values = [self.evaluation_results[m]['normalized_inequality'] for m in methods]
        community_distinction_values = [self.evaluation_results[m]['community_distinction'] for m in methods]
        modularity_values = [self.evaluation_results[m]['modularity'] for m in methods]
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
        
        bars1 = axes[0, 0].bar(method_labels, ifr_values, color=colors, alpha=0.8)
        axes[0, 0].set_title('Intra-Flow Ratio Comparison (9 Communities)', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Intra-Flow Ratio')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        for bar, value in zip(bars1, ifr_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        bars2 = axes[0, 1].bar(method_labels, inequality_values, color=colors, alpha=0.8)
        axes[0, 1].set_title('Normalized Inequality Comparison (9 Communities)', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Normalized Inequality')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        for bar, value in zip(bars2, inequality_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        bars3 = axes[1, 0].bar(method_labels, community_distinction_values, color=colors, alpha=0.8)
        axes[1, 0].set_title('Community Distinction Comparison (9 Communities)', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Community Distinction')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        for bar, value in zip(bars3, community_distinction_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        bars4 = axes[1, 1].bar(method_labels, modularity_values, color=colors, alpha=0.8)
        axes[1, 1].set_title('Modularity Comparison (8 Communities)', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Modularity')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        for bar, value in zip(bars4, modularity_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        
        print(f"Evaluation comparison plots saved to {output_dir}/evaluation_comparison_9communities.png")
    
    def save_evaluation_results(self, output_dir="results_9communities_evaluation"):
        """
        Save evaluation results to CSV file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.evaluation_results:
            print("Please run evaluate_all_methods() first!")
            return
        
        summary_data = []
        for method, results in self.evaluation_results.items():
            param_info = ""
            if method in self.communities_results:
                result = self.communities_results[method]
                if 'radius' in result:
                    param_info = f"R={result['radius']:.4f}"
                elif 'resolution' in result:
                    param_info = f"Res={result['resolution']:.3f}"
                elif 'steps' in result:
                    param_info = f"Steps={result['steps']}"
            
            summary_data.append({
                'Method': method.capitalize(),
                'Parameters': param_info,
                'Number_of_Communities': results['n_communities'],
                'Intra_Flow_Ratio': results['intra_flow_ratio'],
                'Normalized_Inequality': results['normalized_inequality'],
                'Community_Distinction': results['community_distinction'],
                'Modularity': results['modularity']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Modularity', ascending=False)
        summary_df.to_csv(f"{output_dir}/evaluation_results_9communities.csv", index=False)
        
        print(f"Evaluation results table saved to {output_dir}/evaluation_results_9communities.csv")
        return summary_df
    
    def print_evaluation_summary(self):
        """
        Print evaluation summary
        """
        if not self.evaluation_results:
            print("Please run evaluate_all_methods() first!")
            return
        
        print("\n" + "="*100)
        print(f"Community Detection Methods Evaluation Summary (Fixed Number of Communities: {self.target_communities[0]})")
        print("="*100)
        
        print(f"{'Method':<12} {'Parameters':<15} {'Number of Communities':<8} {'IFR':<8} {'Inequality':<10} {'Community Distinction':<12} {'Modularity':<8}")
        print("-" * 120)
        
        sorted_methods = sorted(self.evaluation_results.items(), 
                              key=lambda x: x[1]['modularity'], reverse=True)
        
        for method, results in sorted_methods:
            param_info = ""
            if method in self.communities_results:
                result = self.communities_results[method]
                if 'radius' in result:
                    param_info = f"R={result['radius']:.4f}"
                elif 'resolution' in result:
                    param_info = f"Res={result['resolution']:.3f}"
                elif 'steps' in result:
                    param_info = f"Steps={result['steps']}"
            
            print(f"{method.capitalize():<12} "
                  f"{param_info:<15} "
                  f"{results['n_communities']:<8} "
                  f"{results['intra_flow_ratio']:<8.3f} "
                  f"{results['normalized_inequality']:<10.3f} "
                  f"{results['community_distinction']:<12.3f} "
                  f"{results['modularity']:<8.3f}")
        
        print("\nMetric Descriptions:")
        print("- IFR: Intra-flow ratio within communities, higher is better")
        print("- Inequality: Degree of inequality in community sizes, lower is better")  
        print("- Community Distinction: Degree of distinction between communities, higher is better")
        print("- Modularity: Traditional modularity metric, higher is better")


def main():
    """
    Main function: Evaluation for 9 communities
    """
    print("Starting evaluation of community detection methods for 9 communities...")
    
    evaluator = TargetedCommunityEvaluation()
    
    results = evaluator.evaluate_all_methods()
    
    evaluator.print_evaluation_summary()
    
    evaluator.create_evaluation_plots()
    
    summary_df = evaluator.save_evaluation_results()
    
    print("\nEvaluation for 9 communities completed! Results have been saved to the results_9communities_evaluation/ directory")
    
    return evaluator, summary_df


if __name__ == "__main__":
    main()
