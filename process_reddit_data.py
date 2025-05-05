import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# 定义列名
columns = [
    'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'POST_ID', 'TIMESTAMP', 
    'POST_LABEL', 'POST_PROPERTIES'
]

# 加载数据
def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t', names=columns, low_memory=False)
    return df

# 处理POST_PROPERTIES列
def process_post_properties(df):
    # 将POST_PROPERTIES拆分为单独的列
    properties = df['POST_PROPERTIES'].str.split(',', expand=True)
    properties = properties.apply(pd.to_numeric, errors='coerce')
    
    # 为每个属性创建有意义的列名
    property_names = [
        'num_chars', 'num_chars_no_space', 'frac_alpha', 'frac_digits',
        'frac_uppercase', 'frac_whitespace', 'frac_special', 'num_words',
        'num_unique_words', 'num_long_words', 'avg_word_length',
        'num_stopwords', 'frac_stopwords', 'num_sentences',
        'num_long_sentences', 'avg_chars_per_sentence',
        'avg_words_per_sentence', 'readability_index',
        'sentiment_positive', 'sentiment_negative', 'sentiment_compound'
    ]
    
    # 添加LIWC特征名称
    liwc_features = [
        'LIWC_Funct', 'LIWC_Pronoun', 'LIWC_Ppron', 'LIWC_I', 'LIWC_We',
        'LIWC_You', 'LIWC_SheHe', 'LIWC_They', 'LIWC_Ipron', 'LIWC_Article',
        'LIWC_Verbs', 'LIWC_AuxVb', 'LIWC_Past', 'LIWC_Present', 'LIWC_Future',
        'LIWC_Adverbs', 'LIWC_Prep', 'LIWC_Conj', 'LIWC_Negate', 'LIWC_Quant',
        'LIWC_Numbers', 'LIWC_Swear', 'LIWC_Social', 'LIWC_Family',
        'LIWC_Friends', 'LIWC_Humans', 'LIWC_Affect', 'LIWC_Posemo',
        'LIWC_Negemo', 'LIWC_Anx', 'LIWC_Anger', 'LIWC_Sad', 'LIWC_CogMech',
        'LIWC_Insight', 'LIWC_Cause', 'LIWC_Discrep', 'LIWC_Tentat',
        'LIWC_Certain', 'LIWC_Inhib', 'LIWC_Incl', 'LIWC_Excl', 'LIWC_Percept',
        'LIWC_See', 'LIWC_Hear', 'LIWC_Feel', 'LIWC_Bio', 'LIWC_Body',
        'LIWC_Health', 'LIWC_Sexual', 'LIWC_Ingest', 'LIWC_Relativ',
        'LIWC_Motion', 'LIWC_Space', 'LIWC_Time', 'LIWC_Work', 'LIWC_Achiev',
        'LIWC_Leisure', 'LIWC_Home', 'LIWC_Money', 'LIWC_Relig', 'LIWC_Death',
        'LIWC_Assent', 'LIWC_Dissent', 'LIWC_Nonflu', 'LIWC_Filler'
    ]
    
    property_names.extend(liwc_features)
    
    # 确保我们有足够的列名
    if len(property_names) != properties.shape[1]:
        print(f"Warning: Number of property names ({len(property_names)}) does not match number of columns ({properties.shape[1]})")
    
    # 重命名列
    properties.columns = property_names[:properties.shape[1]]
    
    # 合并回原始数据框
    df = pd.concat([df.drop('POST_PROPERTIES', axis=1), properties], axis=1)
    
    return df

# 基本数据分析
def analyze_data(df, source_name):
    print(f"\n分析 {source_name} 数据集:")
    print(f"总行数: {len(df)}")
    print(f"子版块数量: {df['SOURCE_SUBREDDIT'].nunique()}")
    print(f"目标子版块数量: {df['TARGET_SUBREDDIT'].nunique()}")
    
    # 清理标签数据
    df['POST_LABEL'] = pd.to_numeric(df['POST_LABEL'], errors='coerce')
    df = df.dropna(subset=['POST_LABEL'])
    df['POST_LABEL'] = df['POST_LABEL'].astype(int)
    
    print("\n标签分布:")
    print(df['POST_LABEL'].value_counts())
    
    # 时间范围
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
    df = df.dropna(subset=['TIMESTAMP'])
    print(f"\n时间范围: {df['TIMESTAMP'].min()} 到 {df['TIMESTAMP'].max()}")

def main():
    # 加载数据
    body_df = load_data("soc-redditHyperlinks-body.tsv")
    title_df = load_data("soc-redditHyperlinks-title.tsv")
    
    # 处理数据
    body_df = process_post_properties(body_df)
    title_df = process_post_properties(title_df)
    
    # 添加数据源标识
    body_df['SOURCE_TYPE'] = 'body'
    title_df['SOURCE_TYPE'] = 'title'
    
    # 合并数据集
    combined_df = pd.concat([body_df, title_df], ignore_index=True)
    
    # 分析数据
    analyze_data(body_df, "Body")
    analyze_data(title_df, "Title")
    analyze_data(combined_df, "Combined")
    
    # 保存处理后的数据
    combined_df.to_csv('processed_reddit_data.csv', index=False)
    print("\n处理后的数据已保存到 'processed_reddit_data.csv'")

if __name__ == "__main__":
    main() 