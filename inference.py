import torch
import torch.nn as nn
import json
import random
import time
from zeus.monitor import ZeusMonitor

print("CUDA:", torch.cuda.is_available())
monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()], approx_instant_energy=True)

ACTIVATION_FUNCTIONS = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'leaky_relu': nn.LeakyReLU,
    'elu': nn.ELU,
    'gelu': nn.GELU,
    'selu': nn.SELU
}

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, activation_fn_name):
        super(MLP, self).__init__()

        if activation_fn_name not in ACTIVATION_FUNCTIONS:
            raise ValueError(f"Unsupported activation function: {activation_fn_name}")

        activation_fn = ACTIVATION_FUNCTIONS[activation_fn_name]()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation_fn)

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(ACTIVATION_FUNCTIONS[activation_fn_name]())

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    

with open("configs.json", "r") as f:
    cfgs = json.load(f)

results = []
for i, cfg in enumerate(cfgs):
    model = MLP(**cfg["model"]).cuda()
    x = torch.rand([cfg["batch_size"], cfg["model"]["input_dim"]]).cuda()
    y = model(x)

    monitor.begin_window("step")
    model(x)
    result = monitor.end_window("step")
    results.append({
        "config": cfg,
        "time": result.time,
        "energy": result.gpu_energy[0]
    })

    print(f"Done {i}")

with open("output.json", "w") as f:
    json.dump(results, f)