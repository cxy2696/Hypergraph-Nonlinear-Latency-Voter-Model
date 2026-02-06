import numpy as np
import random
from model import HNLVoterModel  # Import the class

def simulate_hnl_voter_random(n=100, num_hyperedges=200, hyperedge_size_range=(3, 6), q=1.5, latency_dist='normal', mu=5, sigma=2, lambda_param=None, consensus_threshold=0.8, max_steps=100):
    nodes = np.arange(n)
    hyperedges = [random.sample(range(n), random.randint(*hyperedge_size_range)) for _ in range(num_hyperedges)]
    influence_weights = np.random.normal(1, 0.2, n)
    influence_weights = np.maximum(0.1, influence_weights)
    model = HNLVoterModel(nodes, hyperedges, influence_weights, q=q, latency_dist=latency_dist, mu=mu, sigma=sigma, lambda_param=lambda_param, consensus_threshold=consensus_threshold)
    history = model.run_simulation(max_steps=max_steps)
    model.plot_metrics()
    return history

# Example
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    simulate_hnl_voter_random()
