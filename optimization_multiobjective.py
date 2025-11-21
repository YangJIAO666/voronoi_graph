"""
Multi-objective Bayesian Optimization for Voronoi-based Community Detection
Objectives:
    1. Modularity (maximize)
    2. 1 - Inequality (maximize)
    3. Community Distinction (maximize)
    4. IFR - Intra-Flow Ratio (maximize)
"""

import os
import time
import json
import numpy as np
import networkx as nx
import optuna
from collections import defaultdict
import heapq

from modularity import calculate_newman_modularity


class VoronoiMultiObjectiveOptimizer:
    def __init__(self, graphml_path,
                 R_min=1.5, R_max=6.0,
                 target_communities=(7, 9),
                 n_trials=100,
                 output_dir="opt_results",
                 seed=42):
        """
        初始化多目标优化器
        
        参数:
        graphml_path: GraphML文件路径
        R_min: R参数最小值
        R_max: R参数最大值
        target_communities: 目标社区数量范围 (min, max)
        n_trials: 优化迭代次数
        output_dir: 输出目录
        seed: 随机种子
        """
        self.base_path = graphml_path
        self.base_G = nx.read_graphml(graphml_path).to_undirected()

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.R_min, self.R_max = R_min, R_max
        self.target_min, self.target_max = target_communities
        self.n_trials = n_trials
        self.seed = seed

        self.edge_info = {}
        geo_list = []
        socio_list = []

        for u, v, d in self.base_G.edges(data=True):
            d_geo = float(d.get("distance", 0.0))
            d_brand = float(d.get("brand_distance", 1.0))
            d_socio = float(d.get("socioeconomic_distance", 0.0))
            ecc = float(d.get("ecc", 0.01))

            self.edge_info[(u, v)] = {
                "d_geo": d_geo,
                "d_brand": d_brand,
                "d_socio": d_socio,
                "ecc": ecc
            }
            self.edge_info[(v, u)] = self.edge_info[(u, v)]

            geo_list.append(d_geo)
            socio_list.append(d_socio)

        self.max_geo = max(geo_list) if geo_list else 1.0
        self.max_socio = max(socio_list) if socio_list else 1.0
        if self.max_geo == 0: self.max_geo = 1.0
        if self.max_socio == 0: self.max_socio = 1.0

        self.node_brand = {n: self.base_G.nodes[n].get("brand", "unknown") for n in self.base_G.nodes()}
        


    def compute_edge_weights(self, alpha, beta, gamma):
        """根据α, β, γ计算边权重"""
        G = nx.Graph()

        for n, data in self.base_G.nodes(data=True):
            G.add_node(n, **data)

        edge_lengths = {}

        for u, v in self.base_G.edges():
            info = self.edge_info[(u, v)]

            d_geo_norm = info["d_geo"] / self.max_geo
            d_socio_norm = info["d_socio"]
            ecc = max(info["ecc"], 0.01)

            w = (alpha * d_geo_norm +
                 beta * info["d_brand"] +
                 gamma * d_socio_norm) / ecc

            G.add_edge(u, v, weight=w)

            edge_lengths[(u, v)] = w
            edge_lengths[(v, u)] = w

        return G, edge_lengths


    def calculate_local_relative_density(self, G):
        """计算局部相对密度"""
        density = {}
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if not neighbors:
                density[node] = 0.0
                continue

            m = 0  # 内部边
            k = 0  # 外部边

            for i, u in enumerate(neighbors):
                for v in neighbors[i+1:]:
                    if G.has_edge(u, v):
                        m += 1

            for u in neighbors:
                for v in G.nodes():
                    if v != node and v not in neighbors:
                        if G.has_edge(u, v):
                            k += 1

            density[node] = 0.0 if (m + k) == 0 else m / (m + k)

        return density

    def calculate_weighted_local_density(self, G):
        """计算加权局部相对密度"""
        base = self.calculate_local_relative_density(G)
        weighted = {}

        for node in G.nodes():
            s = sum(1.0 / G[node][nb].get("weight", 1.0) for nb in G.neighbors(node))
            weighted[node] = base[node] * s

        return weighted

    def dijkstra_with_cutoff(self, G, edge_lengths, source, radius):
        """带截断的Dijkstra算法"""
        dist = {n: float("inf") for n in G.nodes()}
        dist[source] = 0.0
        pq = [(0.0, source)]
        visited = set()

        while pq:
            d, node = heapq.heappop(pq)
            if node in visited or d > radius:
                continue

            visited.add(node)

            for nb in G.neighbors(node):
                nd = d + edge_lengths[(node, nb)]
                if nd < dist[nb]:
                    dist[nb] = nd
                    heapq.heappush(pq, (nd, nb))

        return visited

    def choose_generators(self, G, edge_lengths, R):
        """选择生成器节点"""
        density = self.calculate_weighted_local_density(G)
        sorted_nodes = sorted(density.items(), key=lambda x: x[1], reverse=True)

        generators = []
        excluded = set()

        for node, _ in sorted_nodes:
            if node not in excluded:
                generators.append(node)
                excluded |= self.dijkstra_with_cutoff(G, edge_lengths, node, R)
                if len(excluded) >= G.number_of_nodes():
                    break

        return generators


    def assign_communities(self, G, edge_lengths, generators):
        """分配社区"""
        gen_index = {g: i for i, g in enumerate(generators)}
        best = {n: (float("inf"), None) for n in G.nodes()}

        for g in generators:
            dist = {n: float("inf") for n in G.nodes()}
            dist[g] = 0.0
            pq = [(0.0, g)]

            while pq:
                d, node = heapq.heappop(pq)
                if d > dist[node]:
                    continue
                for nb in G.neighbors(node):
                    nd = d + edge_lengths[(node, nb)]
                    if nd < dist[nb]:
                        dist[nb] = nd
                        heapq.heappush(pq, (nd, nb))

            gi = gen_index[g]
            for n in G.nodes():
                if dist[n] < best[n][0]:
                    best[n] = (dist[n], gi)

        communities = {n: gi for n, (d, gi) in best.items()}
        return communities


    def calculate_inequality(self, communities):
        """计算社区大小不平等度"""
        labels = set(communities.values()) - {-1}
        sizes = [sum(1 for v in communities.values() if v == lab)
                 for lab in labels]

        if len(sizes) <= 1:
            return 0.0

        mean = np.mean(sizes)
        std = np.std(sizes)
        cv = std / mean if mean > 0 else 0.0
        max_cv = np.sqrt(len(sizes) - 1)
        return cv / max_cv if max_cv > 0 else 0.0

    def calculate_community_distinction(self, G, communities):
        """
        计算社区区分度（基于品牌组成的差异）
        值越高表示社区之间的品牌组成差异越大
        """
        labels = sorted(set(communities.values()) - {-1})
        
        if len(labels) <= 1:
            return 0.0
        
        all_brands = sorted(set(self.node_brand.values()))
        
        community_vectors = {}
        for lab in labels:
            nodes = [n for n in G.nodes() if communities[n] == lab]
            total = max(1, len(nodes))
            
            brand_counts = defaultdict(int)
            for n in nodes:
                brand_counts[self.node_brand[n]] += 1
            
            vector = np.array([brand_counts[b] / total for b in all_brands])
            community_vectors[lab] = vector
        
        distinctions = []
        for i, lab1 in enumerate(labels):
            for lab2 in labels[i+1:]:
                vec1 = community_vectors[lab1]
                vec2 = community_vectors[lab2]
                
                dist = np.linalg.norm(vec1 - vec2)
                distinctions.append(dist)
        
        return float(np.mean(distinctions)) if distinctions else 0.0

    def calculate_ifr(self, G, communities):
        """
        计算Intra-Flow Ratio (IFR)
        社区内部边权重占总边权重的比例
        """
        total_weight = 0.0
        intra_weight = 0.0
        
        for u, v, data in G.edges(data=True):
            weight = float(data.get('weight', 1.0))
            similarity = 1.0 / (weight + 0.01)
            
            total_weight += similarity
            
            if communities.get(u, -1) == communities.get(v, -1) and communities.get(u, -1) != -1:
                intra_weight += similarity
        
        return intra_weight / total_weight if total_weight > 0 else 0.0


    def evaluate(self, alpha, beta, gamma, R):
        """评估给定参数的社区检测结果"""
        G, edge_lengths = self.compute_edge_weights(alpha, beta, gamma)

        generators = self.choose_generators(G, edge_lengths, R)
        if len(generators) == 0:
            return {
                "modularity": 0.0,
                "inequality": 1.0,
                "community_distinction": 0.0,
                "ifr": 0.0,
                "n_comm": 0
            }

        communities = self.assign_communities(G, edge_lengths, generators)
        n_comm = len(set(communities.values()) - {-1})

        modularity = calculate_newman_modularity(G, communities)

        if not (self.target_min <= n_comm <= self.target_max):
            return {
                "modularity": modularity,
                "inequality": 1.0,
                "community_distinction": 0.0,
                "ifr": 0.0,
                "n_comm": n_comm
            }

        inequality = self.calculate_inequality(communities)
        community_distinction = self.calculate_community_distinction(G, communities)
        ifr = self.calculate_ifr(G, communities)

        return {
            "modularity": modularity,
            "inequality": inequality,
            "community_distinction": community_distinction,
            "ifr": ifr,
            "n_comm": n_comm
        }


    def objective(self, trial):
        """多目标优化的目标函数"""
        raw = np.random.dirichlet([1, 1, 1])
        alpha = 0.1 + 0.7 * raw[0]
        beta  = 0.1 + 0.7 * raw[1]
        gamma = 0.1 + 0.7 * raw[2]

        R = trial.suggest_float("R", self.R_min, self.R_max)

        res = self.evaluate(alpha, beta, gamma, R)

        trial.set_user_attr("alpha", alpha)
        trial.set_user_attr("beta", beta)
        trial.set_user_attr("gamma", gamma)
        trial.set_user_attr("R", R)

        for k, v in res.items():
            trial.set_user_attr(k, v)

        return (
            res["modularity"],           # 最大化
            1 - res["inequality"],       # 最大化 (1-不平等度)
            res["community_distinction"], # 最大化
            res["ifr"]                   # 最大化
        )


    def optimize(self, n_trials=None, timeout=None):
        """运行多目标优化"""
        n_trials = n_trials or self.n_trials

        sampler = optuna.samplers.NSGAIISampler(seed=self.seed)

        study = optuna.create_study(
            directions=["maximize", "maximize", "maximize", "maximize"],
            sampler=sampler
        )

        study.optimize(self.objective, n_trials=n_trials,
                       timeout=timeout,
                       show_progress_bar=True)

        pareto = []
        for t in study.best_trials:
            p = {
                "trial": t.number,
                "modularity": t.values[0],
                "one_minus_inequality": t.values[1],
                "community_distinction": t.values[2],
                "ifr": t.values[3],
                "alpha": t.user_attrs["alpha"],
                "beta": t.user_attrs["beta"],
                "gamma": t.user_attrs["gamma"],
                "R": t.user_attrs["R"],
                "n_comm": t.user_attrs["n_comm"],
            }
            pareto.append(p)

        pareto_path = os.path.join(self.output_dir, "pareto_front.json")
        with open(pareto_path, "w") as f:
            json.dump(pareto, f, indent=2)

        df = study.trials_dataframe()
        csv_path = os.path.join(self.output_dir, "all_trials.csv")
        df.to_csv(csv_path, index=False)


        return study, pareto

    def get_best_for_target_communities(self, pareto, target_n):
        """
        从Pareto front中获取指定社区数量的最佳解
        
        参数:
        pareto: Pareto front列表
        target_n: 目标社区数量
        
        返回:
        最佳解的字典，如果没有找到则返回None
        """
        candidates = [p for p in pareto if p["n_comm"] == target_n]
        
        if not candidates:
            return None
        
        best = max(candidates, key=lambda x: x["modularity"])
        return best


def run_optimization(graphml_path, 
                     R_min=1.5, 
                     R_max=12.0,
                     target_communities=(3, 15),
                     n_trials=500,
                     output_dir="opt_results",
                     seed=42):
    """
    便捷函数：运行多目标优化
    
    参数:
    graphml_path: GraphML文件路径
    R_min: R参数最小值
    R_max: R参数最大值
    target_communities: 目标社区数量范围
    n_trials: 优化迭代次数
    output_dir: 输出目录
    seed: 随机种子
    
    返回:
    study: Optuna study对象
    pareto: Pareto front列表
    """
    optimizer = VoronoiMultiObjectiveOptimizer(
        graphml_path=graphml_path,
        R_min=R_min,
        R_max=R_max,
        target_communities=target_communities,
        n_trials=n_trials,
        output_dir=output_dir,
        seed=seed
    )
    
    study, pareto = optimizer.optimize(n_trials=n_trials)
    
    return study, pareto, optimizer


def print_pareto_summary(pareto):
    """打印Pareto front摘要"""
    
    by_n_comm = defaultdict(list)
    for p in pareto:
        by_n_comm[p["n_comm"]].append(p)
    
    for n_comm in sorted(by_n_comm.keys()):
        solutions = by_n_comm[n_comm]
        best = max(solutions, key=lambda x: x["modularity"])
        


if __name__ == "__main__":
    graph_path = "output_socioeconomic_new/final_weighted_graph.graphml"
    
    study, pareto, optimizer = run_optimization(
        graphml_path=graph_path,
        R_min=1.5,
        R_max=12.0,
        target_communities=(3, 20),
        n_trials=500,
        output_dir="opt_results_revised",
        seed=42
    )
    
    print_pareto_summary(pareto)