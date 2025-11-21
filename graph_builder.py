import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import networkx as nx
import os
import math
import csv
from matplotlib.colors import to_rgba

def calculate_haversine_distance(lat1, lon1, lat2, lon2, unit='km'):

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    earth_radius_km = 6371.0
    
    distance_km = earth_radius_km * c
    
    if unit.lower() == 'km':
        return distance_km
    elif unit.lower() == 'm':
        return distance_km * 1000
    else:
        raise ValueError("Unsupported unit. Use 'km' or 'm'.")

def edge_clustering_coefficient(graph, node1, node2):
    """
    Edge Clustering Coefficient
    based on Radicchi et al. (2004)
    """
    k_i = graph.degree(node1)
    k_j = graph.degree(node2)
    
    common_neighbors = set(graph.neighbors(node1)) & set(graph.neighbors(node2))
    z_ij = len(common_neighbors)
    
    max_possible_triangles = min(k_i - 1, k_j - 1)
    
    if max_possible_triangles <= 0:
        return 0.01  # Changed to 0.01 to avoid division by zero
    
    ecc = (z_ij + 1) / max_possible_triangles
    ecc = max(0.0, min(ecc, 1.0))
    
    return ecc

def calculate_brand_distance(brand1, brand2):
    return 0.5 if brand1 == brand2 else 1.0

def prepare_socioeconomic_pca(region_data):
    """
    Use PCA to process socioeconomic data and normalize the PCA coordinates
    
    Parameters:
    region_data - DataFrame containing regional socioeconomic data
    
    Returns:
    pca_model - Trained PCA model
    X_pca_normalized - PCA transformed and normalized coordinates
    region_to_idx - Mapping from region name to index
    """
    
    features = ['pop_d', 'family_inc', 'family_out', 'night']
    X = region_data[features].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA selected {pca.n_components_} principal components")
    print(f"Cumulative explained variance: {pca.explained_variance_ratio_.sum():.1%}")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {ratio:.1%}")
    
    X_pca_min = X_pca.min(axis=0)
    X_pca_max = X_pca.max(axis=0)
    X_pca_range = X_pca_max - X_pca_min
    X_pca_range[X_pca_range == 0] = 1.0
    X_pca_normalized = (X_pca - X_pca_min) / X_pca_range

    region_to_idx = {region: i for i, region in enumerate(region_data['DUN'].values)}
    
    pca_model = {
        'scaler': scaler,
        'pca': pca,
        'X_pca': X_pca_normalized,  
        'X_pca_raw': X_pca,         
        'pca_min': X_pca_min,       
        'pca_max': X_pca_max,
        'region_to_idx': region_to_idx
    }
    return pca_model


def calculate_socioeconomic_distance_pca(region1, region2, pca_model):
    """
    Calculate socioeconomic distance between two regions using PCA coordinates
    """
    if region1 == region2:
        return 0.0
    
    X_pca = pca_model['X_pca']
    region_to_idx = pca_model['region_to_idx']
    
    idx1 = region_to_idx.get(region1)
    idx2 = region_to_idx.get(region2)
    
    if idx1 is None or idx2 is None:
        return 1.0 
    
    pca_coord1 = X_pca[idx1]
    pca_coord2 = X_pca[idx2]
    
    distance = np.linalg.norm(pca_coord1 - pca_coord2)
    
    return distance

def load_data(store_file_path):
    """Load convenience store data"""
    try:
        if store_file_path.endswith('.xls') or store_file_path.endswith('.xlsx'):
            store_df = pd.read_excel(store_file_path)
        else:
            store_df = pd.read_csv(store_file_path)
        
        lat_col = 'latitude'
        lon_col = 'longitude'
        id_col = 'FID'
        brand_col = 'brand'
        region_col = 'DUN'  
        
        required_cols = [lat_col, lon_col, id_col, brand_col, region_col, 
                         'night', 'pop_d', 'family_inc', 'family_out']
        missing_cols = [col for col in required_cols if col not in store_df.columns]
        
        if missing_cols:
            return None, None, None, None, None, None, None
        
        region_df = store_df[[region_col, 'night', 'pop_d', 'family_inc', 'family_out']].drop_duplicates()
        return store_df, region_df, lat_col, lon_col, id_col, brand_col, region_col
    
    except Exception as e:
        return None, None, None, None, None, None, None

def create_graph(store_df, region_df, lat_col, lon_col, id_col, brand_col, region_col, max_distance=5.0):
    """Create initial graph based on Delaunay triangulation and filter edges exceeding the distance threshold"""
    coords = store_df[[lon_col, lat_col]].values
    
    tri = Delaunay(coords)
    
    G = nx.Graph()
    
    for i, row in store_df.iterrows():
        G.add_node(i, 
                  x=float(row[lon_col]), 
                  y=float(row[lat_col]), 
                  id=str(row[id_col]),
                  brand=str(row[brand_col]),
                  region=str(row[region_col]))
    
    edge_set = set()
    filtered_edges = 0
    total_potential_edges = 0
    
    for simplex in tri.simplices:
        for i in range(3):
            node1_idx = simplex[i]
            node2_idx = simplex[(i + 1) % 3]
            
            edge = tuple(sorted([node1_idx, node2_idx]))
            if edge not in edge_set:
                total_potential_edges += 1
                
                lat1 = store_df.iloc[node1_idx][lat_col]
                lon1 = store_df.iloc[node1_idx][lon_col]
                lat2 = store_df.iloc[node2_idx][lat_col]
                lon2 = store_df.iloc[node2_idx][lon_col]
                
                distance = calculate_haversine_distance(lat1, lon1, lat2, lon2)
                
                if distance <= max_distance:
                    edge_set.add(edge)
                    G.add_edge(node1_idx, node2_idx)
                else:
                    filtered_edges += 1
    print(f"Graph created: {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Filtered out {filtered_edges} edges exceeding {max_distance} km (accounting for {filtered_edges/max(1, total_potential_edges)*100:.1f}% of potential edges)")
    
    return G, coords, tri

def calculate_edge_attributes(G, store_df, region_df, lat_col, lon_col, region_col, 
                              alpha=0.4, beta=0.3, gamma=0.3, distance_unit='km'):
    """
    Calculate various edge attributes using the new weighting formula
    
    New formula: l_ij = (α·d_geo_norm + β·d_brand + γ·d_socio_pca) / ecc
    
    Parameters:
    alpha - Geographic distance weight (default 0.4)
    beta - Brand distance weight (default 0.3)
    gamma - Socioeconomic distance weight (default 0.3)
    """
    
    pca_model = prepare_socioeconomic_pca(region_df)
    
    distance_data = []
    geo_distances = []
    
    for node1, node2 in G.edges():
        lat1 = G.nodes[node1]['y']
        lon1 = G.nodes[node1]['x']
        lat2 = G.nodes[node2]['y']
        lon2 = G.nodes[node2]['x']
        
        distance = calculate_haversine_distance(lat1, lon1, lat2, lon2, unit=distance_unit)
        geo_distances.append(distance)
        G[node1][node2]['distance'] = distance
        
        distance_data.append({
            'node1': node1,
            'node2': node2,
            'distance': distance,
            'distance_unit': distance_unit
        })
    
    max_geo_distance = max(geo_distances) if geo_distances else 1.0
    G.graph['max_geo_distance'] = max_geo_distance
    brand_distance_data = []
    
    for node1, node2 in G.edges():
        brand1 = G.nodes[node1]['brand']
        brand2 = G.nodes[node2]['brand']
        
        brand_dist = calculate_brand_distance(brand1, brand2)
        G[node1][node2]['brand_distance'] = brand_dist
        
        brand_distance_data.append({
            'node1': node1,
            'node2': node2,
            'node1_brand': brand1,
            'node2_brand': brand2,
            'brand_distance': brand_dist
        })
    
    socioeconomic_data = []
    socio_distances = []
    
    for node1, node2 in G.edges():
        region1 = G.nodes[node1]['region']
        region2 = G.nodes[node2]['region']
        
        socio_dist = calculate_socioeconomic_distance_pca(region1, region2, pca_model)
        socio_distances.append(socio_dist)
        G[node1][node2]['socioeconomic_distance'] = socio_dist
        
        socioeconomic_data.append({
            'node1': node1,
            'node2': node2,
            'node1_region': region1,
            'node2_region': region2,
            'socioeconomic_distance_pca': socio_dist
        })
    
    max_socio_distance = max(socio_distances) if socio_distances else 1.0
    G.graph['max_socio_distance'] = max_socio_distance
    ecc_data = []
    
    for node1, node2 in G.edges():
        ecc = edge_clustering_coefficient(G, node1, node2)
        G[node1][node2]['ecc'] = ecc
        ecc_data.append({
            'node1': node1,
            'node2': node2,
            'node1_brand': G.nodes[node1]['brand'],
            'node2_brand': G.nodes[node2]['brand'],
            'ecc': ecc
        })
    
    weight_data = []
    
    for node1, node2 in G.edges():
        d_geo = G[node1][node2]['distance']
        d_brand = G[node1][node2]['brand_distance']
        d_socio = G[node1][node2]['socioeconomic_distance']
        ecc = G[node1][node2]['ecc']
        
        d_geo_norm = d_geo / max_geo_distance
        
        d_socio_norm = d_socio / max_socio_distance if max_socio_distance > 0 else 0
        
        edge_weight = (alpha * d_geo_norm + beta * d_brand + gamma * d_socio_norm) / max(ecc, 0.01)
        
        G[node1][node2]['weight'] = edge_weight
        G[node1][node2]['d_geo_norm'] = d_geo_norm      
        G[node1][node2]['d_socio_norm'] = d_socio_norm
        weight_data.append({
            'node1': node1,
            'node2': node2,
            'd_geo': d_geo,
            'd_geo_norm': d_geo_norm,
            'd_brand': d_brand,
            'd_socio_pca': d_socio,
            'd_socio_norm': d_socio_norm,
            'ecc': ecc,
            'combined_weight': edge_weight,
            'formula': f'({alpha}*{d_geo_norm:.3f} + {beta}*{d_brand:.3f} + {gamma}*{d_socio_norm:.3f}) / {ecc:.3f}'
        })
    
    weights = [data['combined_weight'] for data in weight_data]
    
    return G, distance_data, brand_distance_data, socioeconomic_data, ecc_data, weight_data, pca_model

def save_results(output_dir, G, distance_data, brand_distance_data, socioeconomic_data, 
                ecc_data, weight_data, pca_model):
    """Save results to specified output directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    initial_graph_path = os.path.join(output_dir, "initial_graph.graphml")
    nx.write_graphml(G, initial_graph_path)
    
    distance_path = os.path.join(output_dir, "distances.csv")
    pd.DataFrame(distance_data).to_csv(distance_path, index=False, encoding='utf-8')
    
    brand_distance_path = os.path.join(output_dir, "brand_distance.csv")
    pd.DataFrame(brand_distance_data).to_csv(brand_distance_path, index=False, encoding='utf-8')
    
    socioeconomic_path = os.path.join(output_dir, "socioeconomic_distance_pca.csv")
    pd.DataFrame(socioeconomic_data).to_csv(socioeconomic_path, index=False, encoding='utf-8')
    
    ecc_path = os.path.join(output_dir, "edge_clustering_coefficients.csv")
    pd.DataFrame(ecc_data).to_csv(ecc_path, index=False, encoding='utf-8')
    
    weight_path = os.path.join(output_dir, "combined_weights.csv")
    pd.DataFrame(weight_data).to_csv(weight_path, index=False, encoding='utf-8')
    
    pca_info_path = os.path.join(output_dir, "pca_info.txt")
    with open(pca_info_path, 'w', encoding='utf-8') as f:
        f.write("PCA Analysis Report\n")
        f.write("="*50 + "\n")
        f.write(f"Number of principal components: {pca_model['pca'].n_components_}\n")
        f.write(f"Cumulative explained variance: {pca_model['pca'].explained_variance_ratio_.sum():.4f}\n\n")
        f.write("Explained variance ratio of each principal component:\n")
        for i, ratio in enumerate(pca_model['pca'].explained_variance_ratio_):
            f.write(f"  PC{i+1}: {ratio:.4f}\n")
    
    final_graph_path = os.path.join(output_dir, "final_weighted_graph.graphml")
    nx.write_graphml(G, final_graph_path)

def visualize_network(G, coords, tri, brand_col, output_dir, distance_unit='km'):
    """Create enhanced network visualization without displaying distance numbers on edges"""
    fig, ax = plt.subplots(figsize=(14, 12), dpi=100)
    pos = {i: (coords[i][0], coords[i][1]) for i in range(len(coords))}
    
    brands = sorted(list(set(nx.get_node_attributes(G, 'brand').values())))
    colormap = plt.cm.tab20
    brand_colors = {brand: colormap(i % 20) for i, brand in enumerate(brands)}
    
    for brand in brands:
        brand_nodes = [n for n, d in G.nodes(data=True) if d['brand'] == brand]
        
        if len(brand_nodes) > 0:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=brand_nodes,
                node_color=[brand_colors[brand]],
                node_size=60,
                label=f"{brand} ({len(brand_nodes)})",
                alpha=0.8
            )
    
    distances = [G[u][v]['distance'] for u, v in G.edges()]
    min_dist = min(distances)
    max_dist = max(distances)
    
    for u, v in G.edges():
        dist = G[u][v]['distance']
        normalized_width = 2.0 * (1 - (dist - min_dist) / (max_dist - min_dist)) + 0.5
        
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            width=normalized_width,
            alpha=0.6,
            edge_color='gray'
        )
    
    unit_text = "KM" if distance_unit == 'km' else "m"
    plt.title(f"Johor Bahru Convenience Store Graph ", 
              fontsize=18, pad=20)
    plt.legend(scatterpoints=1, loc='upper right', fontsize=12)
    
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # North arrow
    plt.annotate('N', xy=(0.02, 0.98), xycoords='axes fraction', fontsize=14, weight='bold')
    plt.annotate('↑', xy=(0.02, 0.95), xycoords='axes fraction', fontsize=14, weight='bold')

    
    lon_min = min(coords[:, 0])
    lon_max = max(coords[:, 0])
    mid_lat = np.mean(coords[:, 1])
    one_km_lon = 1 / (111.32 * np.cos(np.radians(mid_lat)))
    
    map_width_km = calculate_haversine_distance(mid_lat, lon_min, mid_lat, lon_max, unit='km')
    
    if map_width_km > 20:
        scale_km = 5.0
    elif map_width_km > 10:
        scale_km = 2.0
    elif map_width_km > 5:
        scale_km = 1.0
    else:
        scale_km = 0.5
    
    scale_x_left = 0.1
    scale_x_right = scale_x_left + (scale_km * one_km_lon / (lon_max - lon_min))
    scale_y = 0.05
    
    plt.plot([scale_x_left, scale_x_right], [scale_y, scale_y], 'k-', 
             transform=plt.gca().transAxes, linewidth=2)
    plt.text(scale_x_left, scale_y - 0.01, '0', transform=plt.gca().transAxes, 
             horizontalalignment='center', verticalalignment='top')
    plt.text(scale_x_right, scale_y - 0.01, f'{scale_km} {unit_text}', transform=plt.gca().transAxes, 
             horizontalalignment='center', verticalalignment='top')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'convenience_store_network_new_formula.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Network visualization saved to {output_path}")
    plt.close()

def main():

    store_file_path = r"finalstore.csv"
    store_df, region_df, lat_col, lon_col, id_col, brand_col, region_col = load_data(store_file_path)
    if store_df is None:
        return
    distance_unit = 'km'
    max_distance = 6.0
    alpha = 0.69  # Geographic distance weight
    beta = 0.12   # Brand distance weight
    gamma = 0.19  # Socioeconomic distance weight
    
    G, coords, tri = create_graph(store_df, region_df, lat_col, lon_col, id_col, brand_col, region_col, max_distance)
    
    G, distance_data, brand_distance_data, socioeconomic_data, ecc_data, weight_data, pca_model = calculate_edge_attributes(
        G, store_df, region_df, lat_col, lon_col, region_col, alpha, beta, gamma, distance_unit
    )
    
    output_dir = "output_socioeconomic_new"
    save_results(output_dir, G, distance_data, brand_distance_data, socioeconomic_data, 
                ecc_data, weight_data, pca_model)
    
    visualize_network(G, coords, tri, brand_col, output_dir, distance_unit)
    
    
    return G

if __name__ == "__main__":
    main()
