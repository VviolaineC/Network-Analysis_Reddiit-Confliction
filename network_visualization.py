import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import community as community_louvain
import matplotlib.colors as mcolors

def load_network():
    """加载网络数据"""
    return nx.read_gexf("reddit_network.gexf")

def create_subgraph(G, top_n=1000):
    """创建包含最重要节点的子图"""
    # 计算PageRank中心性
    pagerank = nx.pagerank(G, weight='weight')
    
    # 选择PageRank值最高的top_n个节点
    top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_nodes = [node for node, _ in top_nodes]
    
    # 创建子图
    subgraph = G.subgraph(top_nodes)
    return subgraph

def visualize_network(G, output_file="reddit_network.png"):
    """可视化网络"""
    plt.figure(figsize=(25, 25))
    
    # 计算节点大小（基于度数）
    degrees = dict(G.degree())
    max_degree = max(degrees.values())
    node_sizes = [degrees[node] * 1000 / max_degree for node in G.nodes()]
    
    # 计算节点颜色（基于社区）
    communities = community_louvain.best_partition(G.to_undirected())
    community_colors = [communities[node] for node in G.nodes()]
    
    # 创建自定义颜色映射
    cmap = plt.cm.get_cmap('tab20', len(set(communities.values())))
    
    # 计算边的权重
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights)
    
    # 绘制网络
    pos = nx.spring_layout(G, k=2, iterations=100)
    
    # 绘制边
    nx.draw_networkx_edges(G, pos,
                          width=[w/max_weight * 3 for w in edge_weights],
                          alpha=0.2,
                          edge_color='gray')
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, 
                          node_size=node_sizes,
                          node_color=community_colors,
                          cmap=cmap,
                          alpha=0.8)
    
    # 添加标签（只显示度数最高的20个节点）
    top_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:20]
    labels = {node: node for node, _ in top_degrees}
    nx.draw_networkx_labels(G, pos, labels, 
                           font_size=12,
                           font_weight='bold',
                           font_color='black')
    
    # 添加图例
    community_counts = Counter(communities.values())
    top_communities = community_counts.most_common(20)
    
    # 创建图例
    legend_elements = []
    for comm_id, count in top_communities:
        # 找出该社区中最重要的节点
        community_nodes = [node for node, comm in communities.items() if comm == comm_id]
        community_subgraph = G.subgraph(community_nodes)
        pagerank = nx.pagerank(community_subgraph, weight='weight')
        top_node = max(pagerank.items(), key=lambda x: x[1])[0]
        
        color = cmap(comm_id)
        legend_elements.append(plt.Line2D([0], [0], 
                                        marker='o', 
                                        color='w', 
                                        label=f'社区 {comm_id}: {top_node} ({count}节点)',
                                        markerfacecolor=color,
                                        markersize=10))
    
    plt.legend(handles=legend_elements, 
              loc='center left', 
              bbox_to_anchor=(1, 0.5),
              title="社区分布")
    
    plt.title("Reddit Subreddit Network (Top 1000 Nodes by PageRank)", 
              fontsize=16, pad=20)
    plt.axis('off')
    
    # 调整布局以容纳图例
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_communities(G):
    """分析社区结构"""
    # 转换为无向图进行社区检测
    G_undirected = G.to_undirected()
    communities = community_louvain.best_partition(G_undirected)
    
    # 统计社区大小
    community_sizes = Counter(communities.values())
    print("\n=== 社区结构分析 ===")
    print(f"社区数量: {len(community_sizes)}")
    print("\n最大的20个社区:")
    for comm_id, size in community_sizes.most_common(20):
        # 找出该社区中最重要的节点
        community_nodes = [node for node, comm in communities.items() if comm == comm_id]
        community_subgraph = G.subgraph(community_nodes)
        pagerank = nx.pagerank(community_subgraph, weight='weight')
        top_node = max(pagerank.items(), key=lambda x: x[1])[0]
        print(f"社区 {comm_id}: {size} 个节点, 代表节点: {top_node}")

def main():
    print("加载网络数据...")
    G = load_network()
    
    print("创建子图...")
    subgraph = create_subgraph(G, top_n=1000)
    
    print("分析社区结构...")
    analyze_communities(subgraph)
    
    print("生成可视化...")
    visualize_network(subgraph)
    print("网络可视化已保存为 'reddit_network.png'")
    
    # 保存社区信息
    communities = community_louvain.best_partition(subgraph.to_undirected())
    with open("community_info.txt", "w") as f:
        for node, comm_id in communities.items():
            f.write(f"{node}\t{comm_id}\n")
    print("社区信息已保存为 'community_info.txt'")

if __name__ == "__main__":
    main() 