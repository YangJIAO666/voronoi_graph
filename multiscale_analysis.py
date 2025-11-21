import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram

from voronoi_detector import VoronoiCommunityDetector


def run_for_r(detector: VoronoiCommunityDetector, r: float, outdir: str) -> Dict[str, int]:
    """Run community detection for a given R and output map + heatmap + detailed CSV."""
    communities, generators, _ = detector.voronoi_community_detection(r)
    detector.optimal_R = r
    detector.communities = communities
    detector.generators = generators

    detector.visualize_community_detection_result(output_dir=outdir)

    df = pd.DataFrame({
        "community": list(communities.values()),
        "brand": [detector.G.nodes[n]["brand"] for n in communities]
    })

    heat = pd.crosstab(df["community"], df["brand"])

    save_detailed_community_csv(detector, communities, generators, r, outdir)

    return communities


def save_detailed_community_csv(detector: VoronoiCommunityDetector, 
                               communities: Dict[str, int], 
                               generators: List, 
                               r: float, 
                               outdir: str) -> None:
    """
    Save detailed community information to a CSV file
    Includes: node_id, community, brand, lat, long
    
    Parameters:
    detector: VoronoiCommunityDetector instance
    communities: Community assignment dictionary {node: community ID}
    generators: List of generator nodes
    r: Current R value
    outdir: Output directory
    """
    detailed_data = []
    
    for node_id, community_id in communities.items():
        node_data = detector.G.nodes[node_id]
        
        detailed_info = {
            'node_id': node_id,
            'community': community_id,
            'brand': node_data.get('brand', ''),
            'lat': float(node_data.get('y', 0.0)),  # y corresponds to latitude
            'long': float(node_data.get('x', 0.0))  # x corresponds to longitude
        }
        
        detailed_data.append(detailed_info)
    
    df = pd.DataFrame(detailed_data)
    df = df.sort_values(['community', 'node_id'])
    
    csv_filename = os.path.join(outdir, f"community_details_R{r}.csv")
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    




def save_multiscale_summary(comm_by_r: Dict[float, Dict[str, int]], rs: List[float], outdir: str) -> None:

    summary_data = []
    
    for r in sorted(rs):
        communities = comm_by_r[r]
        unique_communities = set(communities.values()) - {-1}
        
        comm_sizes = {}
        for node, comm_id in communities.items():
            if comm_id >= 0:
                comm_sizes[comm_id] = comm_sizes.get(comm_id, 0) + 1
        
        summary_info = {
            'R_value': r,
            'total_communities': len(unique_communities),
            'total_nodes': len([c for c in communities.values() if c >= 0]),
            'largest_community_size': max(comm_sizes.values()) if comm_sizes else 0,
            'smallest_community_size': min(comm_sizes.values()) if comm_sizes else 0,
            'average_community_size': np.mean(list(comm_sizes.values())) if comm_sizes else 0,
            'community_sizes': str(sorted(comm_sizes.values(), reverse=True))
        }
        
        summary_data.append(summary_info)
    
    summary_df = pd.DataFrame(summary_data)
    summary_filename = os.path.join(outdir, "multiscale_summary.csv")
    summary_df.to_csv(summary_filename, index=False, encoding='utf-8')
    



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphml", default="output_socioeconomic_new/final_weighted_graph.graphml", help="GraphML path")
    parser.add_argument("--r", type=float, nargs="+", default=[5.11, 2.6, 2], help="List of R values")
    parser.add_argument("--outdir", default="multiscale_results", help="Output directory")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    detector = VoronoiCommunityDetector(args.graphml)

    comm_by_r: Dict[float, Dict[str, int]] = {}
    for r in args.r:
        subdir = os.path.join(args.outdir, f"R{r}")
        os.makedirs(subdir, exist_ok=True)
        comm_by_r[r] = run_for_r(detector, r, subdir)

    save_multiscale_summary(comm_by_r, args.r, args.outdir)



if __name__ == "__main__":
    main()
