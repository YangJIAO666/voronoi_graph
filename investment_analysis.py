import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from collections import defaultdict, Counter
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import seaborn as sns

from voronoi_detector import VoronoiCommunityDetector

class CommunityBoundaryAnalyzer:
    """
    Class for analyzing community boundaries in a graph using Voronoi-based community detection.
    """
    
    def __init__(self, graphml_path, optimal_R=None):
        """
        Initialize the community boundary analyzer.
        
        Parameters:
        graphml_path: Path to the GraphML file
        optimal_R: Optional R value, if not provided the default is used
        """
        self.graphml_path = graphml_path
        self.optimal_R = optimal_R
        self.detector = None
        self.communities = {}
        self.generators = []
        self.boundary_edges = []
        self.boundary_nodes = set()
        
        self._run_community_detection()
        
    def _run_community_detection(self):
        """Run Voronoi community detection"""
        
        self.detector = VoronoiCommunityDetector(self.graphml_path)
        
        self.detector.calculate_weighted_local_density()
        
        if self.optimal_R is None:
            self.optimal_R = 2.6  # Use the value from your previous code
            
        
        communities, generators, R = self.detector.voronoi_community_detection(self.optimal_R)
        
        self.communities = communities
        self.generators = generators
        
        
    def identify_boundary_edges(self):
        """Identify edges that cross community boundaries"""
        
        self.boundary_edges = []
        G = self.detector.G
        
        for u, v in G.edges():
            u_community = self.communities.get(u, -1)
            v_community = self.communities.get(v, -1)
            
            if u_community != v_community and u_community != -1 and v_community != -1:
                edge_data = G[u][v]
                
                self.boundary_edges.append({
                    'node1': u,
                    'node2': v,
                    'community1': u_community,
                    'community2': v_community,
                    'distance': float(edge_data.get('weight', 1.0)),
                    'weight': float(edge_data.get('weight', 1.0)),
                    'similarity': 1.0 / (float(edge_data.get('weight', 1.0)) + 0.01),
                    'brand_similarity': edge_data.get('brand_similarity', 1.0),
                    'node1_brand': G.nodes[u].get('brand', ''),
                    'node2_brand': G.nodes[v].get('brand', ''),
                    'node1_x': float(G.nodes[u].get('x', 0)),
                    'node1_y': float(G.nodes[u].get('y', 0)),
                    'node2_x': float(G.nodes[v].get('x', 0)),
                    'node2_y': float(G.nodes[v].get('y', 0))
                })
                
                self.boundary_nodes.add(u)
                self.boundary_nodes.add(v)
        
        
        return self.boundary_edges
    
    def analyze_boundary_opportunities(self):
        """Analyze boundary opportunities"""
        if not self.boundary_edges:
            self.identify_boundary_edges()
            
        
        community_pairs = defaultdict(list)
        for edge in self.boundary_edges:
            pair = tuple(sorted([edge['community1'], edge['community2']]))
            community_pairs[pair].append(edge)
        
        boundary_analysis = []
        for pair, edges in community_pairs.items():
            comm1, comm2 = pair
            
            total_weight = sum(edge['weight'] for edge in edges)
            avg_weight = total_weight / len(edges)
            min_weight = min(edge['weight'] for edge in edges)
            
            brand_combinations = Counter()
            for edge in edges:
                brand_pair = tuple(sorted([edge['node1_brand'], edge['node2_brand']]))
                brand_combinations[brand_pair] += 1
            
            boundary_strength = len(edges) / avg_weight if avg_weight > 0 else 0
            
            boundary_analysis.append({
                'community_pair': f"{comm1}-{comm2}",
                'community1': comm1,
                'community2': comm2,
                'edge_count': len(edges),
                'total_weight': total_weight,
                'avg_weight': avg_weight,
                'min_weight': min_weight,
                'boundary_strength': boundary_strength,
                'top_brand_combination': brand_combinations.most_common(1)[0] if brand_combinations else ('', 0),
                'brand_diversity': len(brand_combinations)
            })
        
        boundary_analysis.sort(key=lambda x: x['boundary_strength'], reverse=True)
        
        return boundary_analysis
    
    def visualize_boundary_opportunities(self, output_dir="results_boundary"):
        """Visualize boundary opportunities"""
        if not self.boundary_edges:
            self.identify_boundary_edges()
            
        os.makedirs(output_dir, exist_ok=True)
        
        G = self.detector.G
        
        pos = {}
        for node in G.nodes():
            pos[node] = (float(G.nodes[node]['x']), float(G.nodes[node]['y']))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        unique_communities = set(self.communities.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_communities)))
        color_map = {comm_id: colors[i] for i, comm_id in enumerate(sorted(unique_communities))}
        
        for comm_id in unique_communities:
            if comm_id < 0:
                continue
            comm_nodes = [node for node, cid in self.communities.items() if cid == comm_id]
            nx.draw_networkx_nodes(G, pos, 
                                 nodelist=comm_nodes,
                                 node_color=[color_map[comm_id]] * len(comm_nodes),
                                 node_size=80,
                                 ax=ax1,
                                 alpha=0.8)
        
        normal_edges = [(u, v) for u, v in G.edges() 
                       if self.communities.get(u, -1) == self.communities.get(v, -1)]
        nx.draw_networkx_edges(G, pos, 
                             edgelist=normal_edges,
                             width=0.5, 
                             alpha=0.3, 
                             edge_color='gray',
                             ax=ax1)
        
        boundary_edge_list = [(edge['node1'], edge['node2']) for edge in self.boundary_edges]
        if boundary_edge_list:
            nx.draw_networkx_edges(G, pos, 
                                 edgelist=boundary_edge_list,
                                 width=2.0, 
                                 alpha=0.8, 
                                 edge_color='red',
                                 ax=ax1)
        
        if self.boundary_nodes:
            nx.draw_networkx_nodes(G, pos, 
                                 nodelist=list(self.boundary_nodes),
                                 node_color='red',
                                 node_size=120,
                                 node_shape='s',
                                 ax=ax1,
                                 alpha=0.7)
        
        
        ax1.set_title("Community Boundaries and Inter-Community Connections", fontsize=14)
        ax1.axis('off')
        
        community_patches = [mpatches.Patch(color=color_map[comm_id], 
                                          label=f'Community {comm_id}') 
                           for comm_id in sorted(unique_communities) if comm_id >= 0]
        boundary_patch = mpatches.Patch(color='red', label='Boundary Edges')
        ax1.legend(handles=community_patches + [boundary_patch], 
                  loc='upper right', fontsize=9)
        
        boundary_analysis = self.analyze_boundary_opportunities()
        
        if boundary_analysis:
            communities_list = sorted(unique_communities)
            n_communities = len(communities_list)
            boundary_matrix = np.zeros((n_communities, n_communities))
            
            for analysis in boundary_analysis:
                comm1 = analysis['community1']
                comm2 = analysis['community2']
                strength = analysis['boundary_strength']
                
                if comm1 in communities_list and comm2 in communities_list:
                    i = communities_list.index(comm1)
                    j = communities_list.index(comm2)
                    boundary_matrix[i, j] = strength
                    boundary_matrix[j, i] = strength
            
            im = ax2.imshow(boundary_matrix, cmap='Reds', aspect='auto')
            
            ax2.set_xticks(range(n_communities))
            ax2.set_yticks(range(n_communities))
            ax2.set_xticklabels([f'C{c}' for c in communities_list])
            ax2.set_yticklabels([f'C{c}' for c in communities_list])
            
            for i in range(n_communities):
                for j in range(n_communities):
                    if boundary_matrix[i, j] > 0:
                        ax2.text(j, i, f'{boundary_matrix[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=10)
            
            ax2.set_title("Boundary Strength Between Communities", fontsize=14)
            ax2.set_xlabel("Community ID")
            ax2.set_ylabel("Community ID")
            
            cbar = plt.colorbar(im, ax=ax2)
            cbar.set_label("Boundary Strength", rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/boundary_opportunities_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        
    def generate_boundary_opportunities_map(self, output_dir="results_boundary"):
        """Generate a detailed map of boundary opportunities"""
        if not self.boundary_edges:
            self.identify_boundary_edges()
            
        os.makedirs(output_dir, exist_ok=True)
        
        boundary_analysis = self.analyze_boundary_opportunities()
        
        G = self.detector.G
        
        pos = {}
        for node in G.nodes():
            pos[node] = (float(G.nodes[node]['x']), float(G.nodes[node]['y']))
        
        plt.figure(figsize=(16, 12))
        
        unique_communities = set(self.communities.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_communities)))
        color_map = {comm_id: colors[i] for i, comm_id in enumerate(sorted(unique_communities))}
        
        for comm_id in unique_communities:
            if comm_id < 0:
                continue
            comm_nodes = [node for node, cid in self.communities.items() if cid == comm_id]
            if comm_nodes:
                nx.draw_networkx_nodes(G, pos, 
                                     nodelist=comm_nodes,
                                     node_color=[color_map[comm_id]] * len(comm_nodes),
                                     node_size=60,
                                     alpha=0.7,
                                     label=f'Community {comm_id}')
        
        if boundary_analysis:
            strengths = [analysis['boundary_strength'] for analysis in boundary_analysis]
            min_strength = min(strengths) if strengths else 0
            max_strength = max(strengths) if strengths else 1
            
            for edge in self.boundary_edges:
                comm_pair = tuple(sorted([edge['community1'], edge['community2']]))
                
                edge_strength = 0
                for analysis in boundary_analysis:
                    if (analysis['community1'], analysis['community2']) == comm_pair or \
                       (analysis['community2'], analysis['community1']) == comm_pair:
                        edge_strength = analysis['boundary_strength']
                        break
                
                if max_strength > min_strength:
                    intensity = (edge_strength - min_strength) / (max_strength - min_strength)
                else:
                    intensity = 0.5
                
                plt.plot([edge['node1_x'], edge['node2_x']], 
                        [edge['node1_y'], edge['node2_y']], 
                        color=plt.cm.Reds(0.3 + 0.7 * intensity),
                        linewidth=1.0 + 2.0 * intensity,
                        alpha=0.8)
        
        high_opportunity_nodes = set()
        if boundary_analysis:
            top_opportunities = boundary_analysis[:3]
            for analysis in top_opportunities:
                comm1, comm2 = analysis['community1'], analysis['community2']
                for edge in self.boundary_edges:
                    if (edge['community1'] == comm1 and edge['community2'] == comm2) or \
                       (edge['community1'] == comm2 and edge['community2'] == comm1):
                        high_opportunity_nodes.add(edge['node1'])
                        high_opportunity_nodes.add(edge['node2'])
        
        if high_opportunity_nodes:
            nx.draw_networkx_nodes(G, pos, 
                                 nodelist=list(high_opportunity_nodes),
                                 node_color='gold',
                                 node_size=150,
                                 node_shape='*',
                                 alpha=0.9,
                                 edgecolors='black',
                                 linewidths=2)
        
        
        plt.title("Community Boundary Opportunities Map\n(Golden stars indicate high-opportunity boundary locations)", 
                 fontsize=16, pad=20)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        
        community_patches = [mpatches.Patch(color=color_map[comm_id], 
                                          label=f'Community {comm_id}') 
                           for comm_id in sorted(unique_communities) if comm_id >= 0]
        
        opportunity_patch = mpatches.Patch(color='gold', label='High Opportunity Locations (★)')
        
        plt.legend(handles=community_patches + [opportunity_patch], 
                  loc='upper right', fontsize=10, bbox_to_anchor=(1.15, 1))
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/boundary_opportunities_map.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        
    def save_boundary_analysis_results(self, output_dir="results_boundary"):
        """Save boundary analysis results"""
        if not self.boundary_edges:
            self.identify_boundary_edges()
            
        os.makedirs(output_dir, exist_ok=True)
        
        boundary_df = pd.DataFrame(self.boundary_edges)
        boundary_df.to_csv(f"{output_dir}/boundary_edges_detailed.csv", index=False)
        
        boundary_analysis = self.analyze_boundary_opportunities()
        analysis_df = pd.DataFrame(boundary_analysis)
        analysis_df.to_csv(f"{output_dir}/boundary_opportunities_analysis.csv", index=False)
        
        community_assignments = []
        for node, comm_id in self.communities.items():
            node_data = self.detector.G.nodes[node]
            assignment_data = {
                'node_id': node,
                'store_id': node_data.get('id', ''),
                'brand': node_data.get('brand', ''),
                'region': node_data.get('region', ''),
                'longitude': node_data.get('x', 0.0),
                'latitude': node_data.get('y', 0.0),
                'community_id': comm_id,
                'is_generator': node in self.generators,
                'is_boundary_node': node in self.boundary_nodes
            }
            community_assignments.append(assignment_data)

        assignments_df = pd.DataFrame(community_assignments)
        assignments_df.to_csv(f"{output_dir}/community_assignments_with_boundary.csv", index=False)
        
        
        return boundary_analysis

def main():
    """Main function"""
    graphml_path = r"output_socioeconomic_new/final_weighted_graph.graphml"
    
    optimal_R = 2.6
    
    analyzer = CommunityBoundaryAnalyzer(graphml_path, optimal_R)
    
    boundary_analysis = analyzer.analyze_boundary_opportunities()
    
    modularity = analyzer.detector.calculate_modularity(analyzer.communities)
    
    
    total_edges = analyzer.detector.G.number_of_edges()
    ratio = (len(analyzer.boundary_edges) / total_edges * 100) if total_edges > 0 else 0
    
    
    for analysis in boundary_analysis:
        pair = analysis['community_pair']
        edge_count = analysis['edge_count']
        strength = analysis['boundary_strength']
        brand_combo = f"{analysis['top_brand_combination'][0][0]} & {analysis['top_brand_combination'][0][1]}" if analysis['top_brand_combination'][0] else 'N/A'
        diversity = analysis['brand_diversity']
        
    
    analyzer.visualize_boundary_opportunities()
    analyzer.generate_boundary_opportunities_map()
    
    boundary_analysis_results = analyzer.save_boundary_analysis_results()
    
    if boundary_analysis:
        strengths = [x['boundary_strength'] for x in boundary_analysis]
        avg_strength = np.mean(strengths)
        
        high_threshold = avg_strength * 1.5
        medium_threshold = avg_strength * 0.5
        
        
        avg_edges_per_pair = np.mean([x['edge_count'] for x in boundary_analysis])
    

if __name__ == "__main__":
    main()
