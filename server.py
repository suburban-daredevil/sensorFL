import flwr as fl
from flwr.server.strategy import FedAvg

# fraction_fit : float, optional
# Fraction of clients used during training. In case `min_fit_clients`
# is larger than `fraction_fit * available_clients`, `min_fit_clients`
# will still be sampled. Defaults to 1.0.

# fraction_evaluate : float, optional
# Fraction of clients used during validation. In case `min_evaluate_clients`
# is larger than `fraction_evaluate * available_clients`,
# `min_evaluate_clients` will still be sampled. Defaults to 1.0.

"""Aggregate evaluation losses using weighted average."""

strategy = FedAvg(
        fraction_fit = 1.0,
        fraction_evaluate = 1.0,
        min_available_clients = 2,
    )

config = fl.server.ServerConfig(num_rounds=3)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    strategy = strategy,
    config = config,
)