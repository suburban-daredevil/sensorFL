import flwr as fl
from flwr.server.strategy import FedAvg

strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
    )

fl.server.start_server(
    server_address="0.0.0.0:8080",
    strategy = strategy,
)