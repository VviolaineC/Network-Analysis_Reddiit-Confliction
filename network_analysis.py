import pandas as pd
import networkx as nx
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def load_processed_data():
    """Load processed data"""
    df = pd.read_csv('processed_reddit_data.csv')
    # Ensure timestamp column is correctly parsed
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
    return df

def create_subreddit_network(df):
    """Build subreddit network graph"""
    # Create directed graph
    G = nx.DiGraph()
    
    # 1. First, collect basic information for each subreddit
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
    
    # 2. Collect edge information
    edge_stats = defaultdict(lambda: {
        'weight': 0,
        'negative_count': 0,
        'first_link_time': pd.Timestamp.max,
        'last_link_time': pd.Timestamp.min
    })
    
    # Process each row of data
    print("Processing data...")
    total_rows = len(df)
    for idx, row in df.iterrows():
        if idx % 100000 == 0:
            print(f"Processed {idx}/{total_rows} rows...")
            
        source = row['SOURCE_SUBREDDIT']
        target = row['TARGET_SUBREDDIT']
        timestamp = row['TIMESTAMP']
        is_negative = row['POST_LABEL'] == -1
        
        # Skip invalid timestamps
        if pd.isna(timestamp):
            continue
        
        # Update node statistics
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
        
        # Update edge statistics
        edge = (source, target)
        edge_stats[edge]['weight'] += 1
        if is_negative:
            edge_stats[edge]['negative_count'] += 1
        edge_stats[edge]['first_link_time'] = min(edge_stats[edge]['first_link_time'], timestamp)
        edge_stats[edge]['last_link_time'] = max(edge_stats[edge]['last_link_time'], timestamp)
    
    print("Building network...")
    # 3. Add nodes and edges to the graph
    for subreddit, stats in subreddit_stats.items():
        # Calculate primary type of node (source or target)
        primary_type = 'source' if stats['source_count'] > stats['target_count'] else 'target'
        # Calculate negative post ratio
        neg_ratio = stats['negative_posts'] / stats['total_posts'] if stats['total_posts'] > 0 else 0
        
        # Convert timestamps to string format
        first_seen_str = stats['first_seen'].strftime('%Y-%m-%d %H:%M:%S') if stats['first_seen'] != pd.Timestamp.max else ''
        last_seen_str = stats['last_seen'].strftime('%Y-%m-%d %H:%M:%S') if stats['last_seen'] != pd.Timestamp.min else ''
        
        G.add_node(subreddit, 
                  out_degree=stats['out_degree'],
                  in_degree=stats['in_degree'],
                  total_posts=stats['total_posts'],
                  negative_ratio=float(neg_ratio),  # Ensure native Python type
                  primary_type=primary_type,
                  first_seen=first_seen_str,
                  last_seen=last_seen_str)
    
    # Add edges
    for (source, target), stats in edge_stats.items():
        # Convert timestamps to string format
        first_link_str = stats['first_link_time'].strftime('%Y-%m-%d %H:%M:%S') if stats['first_link_time'] != pd.Timestamp.max else ''
        last_link_str = stats['last_link_time'].strftime('%Y-%m-%d %H:%M:%S') if stats['last_link_time'] != pd.Timestamp.min else ''
        
        G.add_edge(source, target,
                  weight=stats['weight'],
                  neg_label_ratio=float(stats['negative_count'] / stats['weight']),  # Ensure native Python type
                  first_link_time=first_link_str,
                  last_link_time=last_link_str)
    
    return G

def analyze_network(G):
    """Analyze basic network characteristics"""
    print("\n=== Basic Network Information ===")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    
    # Calculate basic network metrics
    print("\n=== Network Statistics ===")
    density = nx.density(G)
    print(f"Network density: {density:.6f}")
    
    # Degree distribution statistics
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    
    print("\n=== Degree Distribution Statistics ===")
    print("In-degree statistics:")
    print(f"  Minimum: {min(in_degrees)}")
    print(f"  Maximum: {max(in_degrees)}")
    print(f"  Average: {sum(in_degrees)/len(in_degrees):.2f}")
    
    print("\nOut-degree statistics:")
    print(f"  Minimum: {min(out_degrees)}")
    print(f"  Maximum: {max(out_degrees)}")
    print(f"  Average: {sum(out_degrees)/len(out_degrees):.2f}")
    
    # Edge weight statistics
    weights = [d['weight'] for (u, v, d) in G.edges(data=True)]
    print("\n=== Edge Weight Statistics ===")
    print(f"Minimum weight: {min(weights)}")
    print(f"Maximum weight: {max(weights)}")
    print(f"Average weight: {sum(weights)/len(weights):.2f}")
    
    # Find most active communities
    print("\n=== Most Active Communities (Based on Total Degree) ===")
    total_degrees = {node: G.in_degree(node) + G.out_degree(node) 
                    for node in G.nodes()}
    top_communities = sorted(total_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    for comm, degree in top_communities:
        print(f"{comm}: {degree}")
    
    # Calculate and display most controversial communities (highest negative ratio)
    print("\n=== Most Controversial Communities (Based on Negative Ratio) ===")
    controversial_communities = sorted(
        [(node, data['negative_ratio']) 
         for node, data in G.nodes(data=True)
         if data['total_posts'] >= 10],  # Only consider communities with at least 10 posts
        key=lambda x: x[1],
        reverse=True
    )[:10]
    for comm, ratio in controversial_communities:
        print(f"{comm}: {ratio:.2%}")

def main():
    # Load data
    print("Loading data...")
    df = load_processed_data()
    
    # Build network
    print("Building network...")
    G = create_subreddit_network(df)
    
    # Analyze network
    print("Analyzing network...")
    analyze_network(G)
    
    # Save network data (can be used for visualization in tools like Gephi)
    print("\nSaving network data...")
    nx.write_gexf(G, "reddit_network.gexf")
    print("Network data saved as 'reddit_network.gexf'")
    
    # Also save in GraphML format (as backup)
    nx.write_graphml(G, "reddit_network.graphml")
    print("Network data also saved as 'reddit_network.graphml'")

if __name__ == "__main__":
    main() 
