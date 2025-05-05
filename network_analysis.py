import pandas as pd
import networkx as nx
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def load_processed_data():
    """加载处理好的数据"""
    df = pd.read_csv('processed_reddit_data.csv')
    # 确保时间戳列被正确解析
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
    return df

def create_subreddit_network(df):
    """构建subreddit网络图"""
    # 创建有向图
    G = nx.DiGraph()
    
    # 1. 首先统计每个subreddit的基本信息
    subreddit_stats = defaultdict(lambda: {
        'out_degree': 0,
        'in_degree': 0,
        'total_posts': 0,
        'negative_posts': 0,
        'first_seen': pd.Timestamp.max,
        'last_seen': pd.Timestamp.min,
        'source_count': 0,
        'target_count': 0
    })
    
    # 2. 统计边的信息
    edge_stats = defaultdict(lambda: {
        'weight': 0,
        'negative_count': 0,
        'first_link_time': pd.Timestamp.max,
        'last_link_time': pd.Timestamp.min
    })
    
    # 处理每一行数据
    print("处理数据中...")
    total_rows = len(df)
    for idx, row in df.iterrows():
        if idx % 100000 == 0:
            print(f"已处理 {idx}/{total_rows} 行...")
            
        source = row['SOURCE_SUBREDDIT']
        target = row['TARGET_SUBREDDIT']
        timestamp = row['TIMESTAMP']
        is_negative = row['POST_LABEL'] == -1
        
        # 跳过无效的时间戳
        if pd.isna(timestamp):
            continue
        
        # 更新节点统计信息
        for sub, role in [(source, 'source'), (target, 'target')]:
            subreddit_stats[sub]['total_posts'] += 1
            if role == 'source':
                subreddit_stats[sub]['out_degree'] += 1
                subreddit_stats[sub]['source_count'] += 1
            else:
                subreddit_stats[sub]['in_degree'] += 1
                subreddit_stats[sub]['target_count'] += 1
            
            if is_negative:
                subreddit_stats[sub]['negative_posts'] += 1
            
            subreddit_stats[sub]['first_seen'] = min(subreddit_stats[sub]['first_seen'], timestamp)
            subreddit_stats[sub]['last_seen'] = max(subreddit_stats[sub]['last_seen'], timestamp)
        
        # 更新边的统计信息
        edge = (source, target)
        edge_stats[edge]['weight'] += 1
        if is_negative:
            edge_stats[edge]['negative_count'] += 1
        edge_stats[edge]['first_link_time'] = min(edge_stats[edge]['first_link_time'], timestamp)
        edge_stats[edge]['last_link_time'] = max(edge_stats[edge]['last_link_time'], timestamp)
    
    print("构建网络...")
    # 3. 添加节点和边到图中
    for subreddit, stats in subreddit_stats.items():
        # 计算节点的主要类型（source或target）
        primary_type = 'source' if stats['source_count'] > stats['target_count'] else 'target'
        # 计算负面帖子比例
        neg_ratio = stats['negative_posts'] / stats['total_posts'] if stats['total_posts'] > 0 else 0
        
        # 将时间戳转换为字符串格式
        first_seen_str = stats['first_seen'].strftime('%Y-%m-%d %H:%M:%S') if stats['first_seen'] != pd.Timestamp.max else ''
        last_seen_str = stats['last_seen'].strftime('%Y-%m-%d %H:%M:%S') if stats['last_seen'] != pd.Timestamp.min else ''
        
        G.add_node(subreddit, 
                  out_degree=stats['out_degree'],
                  in_degree=stats['in_degree'],
                  total_posts=stats['total_posts'],
                  negative_ratio=float(neg_ratio),  # 确保是原生Python类型
                  primary_type=primary_type,
                  first_seen=first_seen_str,
                  last_seen=last_seen_str)
    
    # 添加边
    for (source, target), stats in edge_stats.items():
        # 将时间戳转换为字符串格式
        first_link_str = stats['first_link_time'].strftime('%Y-%m-%d %H:%M:%S') if stats['first_link_time'] != pd.Timestamp.max else ''
        last_link_str = stats['last_link_time'].strftime('%Y-%m-%d %H:%M:%S') if stats['last_link_time'] != pd.Timestamp.min else ''
        
        G.add_edge(source, target,
                  weight=stats['weight'],
                  neg_label_ratio=float(stats['negative_count'] / stats['weight']),  # 确保是原生Python类型
                  first_link_time=first_link_str,
                  last_link_time=last_link_str)
    
    return G

def analyze_network(G):
    """分析网络的基本特征"""
    print("\n=== 网络基本信息 ===")
    print(f"节点数量: {G.number_of_nodes()}")
    print(f"边的数量: {G.number_of_edges()}")
    
    # 计算基本网络指标
    print("\n=== 网络统计 ===")
    density = nx.density(G)
    print(f"网络密度: {density:.6f}")
    
    # 度分布统计
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    
    print("\n=== 度分布统计 ===")
    print("入度统计:")
    print(f"  最小值: {min(in_degrees)}")
    print(f"  最大值: {max(in_degrees)}")
    print(f"  平均值: {sum(in_degrees)/len(in_degrees):.2f}")
    
    print("\n出度统计:")
    print(f"  最小值: {min(out_degrees)}")
    print(f"  最大值: {max(out_degrees)}")
    print(f"  平均值: {sum(out_degrees)/len(out_degrees):.2f}")
    
    # 边权重统计
    weights = [d['weight'] for (u, v, d) in G.edges(data=True)]
    print("\n=== 边权重统计 ===")
    print(f"最小权重: {min(weights)}")
    print(f"最大权重: {max(weights)}")
    print(f"平均权重: {sum(weights)/len(weights):.2f}")
    
    # 找出最活跃的社区
    print("\n=== 最活跃的社区（基于总度数）===")
    total_degrees = {node: G.in_degree(node) + G.out_degree(node) 
                    for node in G.nodes()}
    top_communities = sorted(total_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    for comm, degree in top_communities:
        print(f"{comm}: {degree}")
    
    # 计算并显示最具争议性的社区（负面比例最高的）
    print("\n=== 最具争议性的社区（基于负面比例）===")
    controversial_communities = sorted(
        [(node, data['negative_ratio']) 
         for node, data in G.nodes(data=True)
         if data['total_posts'] >= 10],  # 只考虑有至少10个帖子的社区
        key=lambda x: x[1],
        reverse=True
    )[:10]
    for comm, ratio in controversial_communities:
        print(f"{comm}: {ratio:.2%}")

def main():
    # 加载数据
    print("加载数据...")
    df = load_processed_data()
    
    # 构建网络
    print("构建网络...")
    G = create_subreddit_network(df)
    
    # 分析网络
    print("分析网络...")
    analyze_network(G)
    
    # 保存网络数据（可以用于Gephi等工具进行可视化）
    print("\n保存网络数据...")
    nx.write_gexf(G, "reddit_network.gexf")
    print("网络数据已保存为 'reddit_network.gexf'")
    
    # 同时保存为GraphML格式（作为备份）
    nx.write_graphml(G, "reddit_network.graphml")
    print("网络数据也保存为 'reddit_network.graphml'")

if __name__ == "__main__":
    main() 