import torch.nn as nn
import json
import random

ACTIVATION_FUNCTIONS = {
    0: nn.ReLU,
    1: nn.Tanh,
    2: nn.Sigmoid,
    3: nn.LeakyReLU,
}

def generate_random_config():
    while True:
        hidden_dim = random.randint(1, 2048)
        num_layers = random.randint(1, 200)

        if 1e7 < hidden_dim * hidden_dim * num_layers < 1e9:
            break


    config = {
        "model": {
            "input_dim": random.randint(1, 10000),
            "output_dim": random.randint(1, 10000),
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "activation_fn_name":  random.choice(list(ACTIVATION_FUNCTIONS.keys()))

        },
        "batch_size": random.randint(1, 100),
        "expect_param_count": hidden_dim * hidden_dim * num_layers
    }
    return config

num_configs = 1000
cfgs_json = []
for i in range(num_configs):
    cfg = generate_random_config()
    cfgs_json.append(cfg)

with open("configs.json", "w") as f:
    json.dump(cfgs_json, f)

# model = MLP(**cfg["model"])
# x = torch.rand([cfg["batch_size"], cfg["model"]["input_dim"]])
# y = model(x)
# print(y.shape)