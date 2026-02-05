import pandas as pd
import numpy as np
import random
from collections import defaultdict
import os

class HNLVoterModel:
    def __init__(self, nodes, hyperedges, influence_weights, q=1.5, latency_dist='normal', mu=5, sigma=2, lambda_param=None, consensus_threshold=0.8):
        self.nodes = nodes
        self.hyperedges = hyperedges
        self.influence_weights = influence_weights
        self.q = q
        self.consensus_threshold = consensus_threshold
        self.opinions = np.random.choice([0, 1], size=len(nodes))  # Default random opinions
        self.last_change_time = np.zeros(len(nodes))
        self.time_step = 0
        self.history = []
        n = len(nodes)
        if latency_dist == 'normal':
            self.latencies = np.maximum(1, np.random.normal(mu, sigma, n)).astype(int)
        elif latency_dist == 'exponential':
            if lambda_param is None:
                lambda_param = 1 / mu if mu > 0 else 1
            self.latencies = np.maximum(1, np.random.exponential(1 / lambda_param, n)).astype(int)
        else:
            raise ValueError("latency_dist must be 'normal' or 'exponential'")

    def update_opinions(self):
        new_opinions = self.opinions.copy()
        for hyperedge in self.hyperedges:
            hyperedge = np.array(hyperedge)
            mask0 = (self.opinions[hyperedge] == 0)
            mask1 = (self.opinions[hyperedge] == 1)
            s0 = np.sum(self.influence_weights[hyperedge] * mask0)
            s1 = np.sum(self.influence_weights[hyperedge] * mask1)
            denom = s0**self.q + s1**self.q
            p1 = s1**self.q / denom if denom > 0 else 0.5
            for node in hyperedge:
                if self.time_step - self.last_change_time[node] >= self.latencies[node]:
                    new_opinion = 1 if random.random() < p1 else 0
                    if new_opinion != self.opinions[node]:
                        new_opinions[node] = new_opinion
                        self.last_change_time[node] = self.time_step
        self.opinions = new_opinions

    def run_simulation(self, max_steps=100):
        self.history = []
        prev_opinions = self.opinions.copy()
        for self.time_step in range(1, max_steps + 1):
            self.update_opinions()
            cascade_size = np.sum(self.opinions != prev_opinions) / len(self.nodes)
            prev_opinions = self.opinions.copy()
            n0 = np.sum(self.opinions == 0)
            n1 = np.sum(self.opinions == 1)
            diversity = 1 - max(n0, n1) / len(self.nodes)
            oscillation = np.mean(self.opinions)
            self.history.append({"diversity": diversity, "cascade_size": cascade_size, "oscillation": oscillation})
            majority = max(n0, n1) / len(self.nodes)
            if majority >= self.consensus_threshold:
                break
        return self.history

    def plot_metrics(self):
        import matplotlib.pyplot as plt
        diversity = [h["diversity"] for h in self.history]
        cascade_size = [h["cascade_size"] for h in self.history]
        oscillation = [h["oscillation"] for h in self.history]
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        axs[0].plot(diversity)
        axs[0].set_title("Opinion Diversity over Time")
        axs[1].plot(cascade_size)
        axs[1].set_title("Cascade Size over Time")
        axs[2].plot(oscillation)
        axs[2].set_title("Oscillatory Behavior (Average Opinion) over Time")
        plt.tight_layout()
        plt.show()

def load_data(file_path, format_type=None, sentiment_col='Sentiment', user_col='User', hashtags_col='Hashtags', retweets_col='Retweets', likes_col='Likes'):
    if format_type is None:
        _, ext = os.path.splitext(file_path)
        format_type = ext.lstrip('.').lower()

    if format_type == 'csv':
        df = pd.read_csv(file_path)
    elif format_type == 'parquet':
        df = pd.read_parquet(file_path)
    elif format_type == 'json':
        df = pd.read_json(file_path, lines=True)
    else:
        raise ValueError("Unsupported format. Supported: csv, parquet, json")

    nodes = df[user_col].unique()
    n = len(nodes)
    node_to_index = {user: i for i, user in enumerate(nodes)}

    hashtag_groups = defaultdict(list)
    for _, row in df.iterrows():
        if pd.isna(row[hashtags_col]):
            continue
        hashtags = str(row[hashtags_col]).split()
        for h in hashtags:
            hashtag_groups[h].append(node_to_index[row[user_col]])
    hyperedges = [group for group in hashtag_groups.values() if len(group) > 1]

    df['Engagement'] = df[retweets_col].fillna(0) + df[likes_col].fillna(0)
    engagement = df.groupby(user_col)['Engagement'].mean()
    influence_weights = engagement.reindex(nodes).fillna(0.1).values / engagement.max()

    def map_sentiment(s):
        return 1 if isinstance(s, str) and s.lower() == 'positive' else 0

    user_opinion = df.groupby(user_col)[sentiment_col].first().apply(map_sentiment)
    opinions = user_opinion.reindex(nodes).fillna(0).values.astype(int)

    return nodes, hyperedges, influence_weights, opinions

def run_model_from_file(file_path, format_type=None, q=1.5, latency_dist='normal', mu=5, sigma=2, lambda_param=None, consensus_threshold=0.8, max_steps=100):
    nodes, hyperedges, influence_weights, opinions = load_data(file_path, format_type)
    model = HNLVoterModel(nodes, hyperedges, influence_weights, q=q, latency_dist=latency_dist, mu=mu, sigma=sigma, lambda_param=lambda_param, consensus_threshold=consensus_threshold)
    model.opinions = opinions
    history = model.run_simulation(max_steps=max_steps)
    model.plot_metrics()
    return history