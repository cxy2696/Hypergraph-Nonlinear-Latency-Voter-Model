from model import HNLVoterModel, load_data

def simulate_hnl_voter_real_world(file_path='sentimentdataset.csv', format_type='csv', q=1.5, latency_dist='normal', mu=5, sigma=2, lambda_param=None, consensus_threshold=0.8, max_steps=100):
    nodes, hyperedges, influence_weights, opinions = load_data(file_path, format_type)
    model = HNLVoterModel(nodes, hyperedges, influence_weights, q=q, latency_dist=latency_dist, mu=mu, sigma=sigma, lambda_param=lambda_param, consensus_threshold=consensus_threshold)
    model.opinions = opinions
    history = model.run_simulation(max_steps=max_steps)
    model.plot_metrics()
    return history

# Example
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    simulate_hnl_voter_real_world()
