import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import community as community_louvain

def load_network():
    """加载网络数据"""
    return nx.read_gexf("reddit_network.gexf")

def analyze_conflict_propagation(G):
    """分析冲突传播路径"""
    # 获取社区划分
    communities = community_louvain.best_partition(G.to_undirected())
    
    # 1. 分析每个社区的冲突特征
    community_stats = defaultdict(lambda: {
        'out_conflicts': 0,  # 作为源头的冲突数
        'in_conflicts': 0,   # 作为目标的冲突数
        'total_out_links': 0,
        'total_in_links': 0,
        'nodes': set(),
        'conflict_ratio': 0.0
    })
    
    # 2. 分析社区间的冲突传播
    inter_community_conflicts = defaultdict(lambda: defaultdict(int))
    
    # 3. 分析桥接节点
    bridge_nodes = defaultdict(int)
    
    # 遍历所有边
    for u, v, data in G.edges(data=True):
        source_comm = communities[u]
        target_comm = communities[v]
        
        # 更新社区统计
        community_stats[source_comm]['out_conflicts'] += data['neg_label_ratio'] * data['weight']
        community_stats[source_comm]['total_out_links'] += data['weight']
        community_stats[source_comm]['nodes'].add(u)
        
        community_stats[target_comm]['in_conflicts'] += data['neg_label_ratio'] * data['weight']
        community_stats[target_comm]['total_in_links'] += data['weight']
        community_stats[target_comm]['nodes'].add(v)
        
        # 如果是跨社区边，记录冲突传播
        if source_comm != target_comm:
            inter_community_conflicts[source_comm][target_comm] += data['neg_label_ratio'] * data['weight']
            
            # 更新桥接节点统计
            bridge_nodes[u] += data['neg_label_ratio'] * data['weight']
            bridge_nodes[v] += data['neg_label_ratio'] * data['weight']
    
    # 计算每个社区的冲突比例
    for comm_id, stats in community_stats.items():
        if stats['total_out_links'] > 0:
            stats['out_conflict_ratio'] = stats['out_conflicts'] / stats['total_out_links']
        if stats['total_in_links'] > 0:
            stats['in_conflict_ratio'] = stats['in_conflicts'] / stats['total_in_links']
    
    return community_stats, inter_community_conflicts, bridge_nodes

def visualize_conflict_network(G, community_stats, inter_community_conflicts, communities, output_file="conflict_network.png"):
    """可视化冲突网络"""
    plt.figure(figsize=(20, 20))
    
    # 创建社区级别的图
    comm_graph = nx.DiGraph()
    
    # 添加节点（社区）
    for comm_id, stats in community_stats.items():
        comm_graph.add_node(comm_id,
                          size=len(stats['nodes']),
                          out_conflict_ratio=stats['out_conflict_ratio'],
                          in_conflict_ratio=stats['in_conflict_ratio'])
    
    # 添加边（社区间的冲突传播）
    for source_comm, targets in inter_community_conflicts.items():
        for target_comm, conflict_weight in targets.items():
            if conflict_weight > 0:
                comm_graph.add_edge(source_comm, target_comm, weight=conflict_weight)
    
    # 计算布局
    pos = nx.spring_layout(comm_graph, k=2, iterations=100)
    
    # 绘制节点
    node_sizes = [data['size'] * 100 for _, data in comm_graph.nodes(data=True)]
    node_colors = [data['out_conflict_ratio'] for _, data in comm_graph.nodes(data=True)]
    
    nx.draw_networkx_nodes(comm_graph, pos,
                          node_size=node_sizes,
                          node_color=node_colors,
                          cmap=plt.cm.Reds,
                          alpha=0.8)
    
    # 绘制边
    edge_weights = [data['weight'] for _, _, data in comm_graph.edges(data=True)]
    max_weight = max(edge_weights) if edge_weights else 1
    
    nx.draw_networkx_edges(comm_graph, pos,
                          width=[w/max_weight * 5 for w in edge_weights],
                          edge_color='gray',
                          alpha=0.4)
    
    # 添加标签
    labels = {}
    for node, data in comm_graph.nodes(data=True):
        # 找出该社区中最重要的节点
        community_nodes = [n for n, comm in communities.items() if comm == node]
        if community_nodes:
            pagerank = nx.pagerank(G.subgraph(community_nodes), weight='weight')
            top_node = max(pagerank.items(), key=lambda x: x[1])[0]
            labels[node] = f"{top_node}\n({data['out_conflict_ratio']:.2f})"
    
    nx.draw_networkx_labels(comm_graph, pos, labels, font_size=8)
    
    plt.title("Reddit社区冲突传播网络", fontsize=16)
    plt.axis('off')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("加载网络数据...")
    G = load_network()
    
    print("分析冲突传播路径...")
    communities = community_louvain.best_partition(G.to_undirected())
    community_stats, inter_community_conflicts, bridge_nodes = analyze_conflict_propagation(G)
    
    # 1. 找出冲突频发源头社区
    print("\n=== 冲突频发源头社区 ===")
    conflict_sources = sorted(
        [(comm_id, stats['out_conflict_ratio']) 
         for comm_id, stats in community_stats.items()],
        key=lambda x: x[1],
        reverse=True
    )[:10]
    for comm_id, ratio in conflict_sources:
        print(f"社区 {comm_id}: 冲突输出比例 {ratio:.2%}")
    
    # 2. 找出高频受害者社区
    print("\n=== 高频受害者社区 ===")
    conflict_targets = sorted(
        [(comm_id, stats['in_conflict_ratio']) 
         for comm_id, stats in community_stats.items()],
        key=lambda x: x[1],
        reverse=True
    )[:10]
    for comm_id, ratio in conflict_targets:
        print(f"社区 {comm_id}: 冲突输入比例 {ratio:.2%}")
    
    # 3. 找出重要的桥接节点
    print("\n=== 重要的桥接节点 ===")
    top_bridges = sorted(bridge_nodes.items(), key=lambda x: x[1], reverse=True)[:10]
    for node, score in top_bridges:
        print(f"{node}: 桥接分数 {score:.2f}")
    
    # 4. 可视化冲突网络
    print("\n生成冲突网络可视化...")
    visualize_conflict_network(G, community_stats, inter_community_conflicts, communities)
    print("冲突网络可视化已保存为 'conflict_network.png'")

if __name__ == "__main__":
    main() 