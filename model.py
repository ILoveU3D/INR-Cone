import torch
import tinycudann as tcnn
from options import beijingVolumeSize

encoding_config = {
    "otype": "Grid",
    "type": "Hash",
    "n_levels": 8, "n_features_per_level": 8,
    "log2_hash_map_size": 22,
    "base_resolution": 3,
    "per_level_scale": 1.95*4,
    "interpolation": "Linear"
}
network_config = {
    "otype": "CutlassMLP",
    "activation": "ReLU",
    "output_activation": "ReLU",
    "n_neurons": 64, "n_hidden_layers": 1
}
model = torch.nn.ModuleList()
for s in range(beijingVolumeSize[2]):
    model.append(tcnn.NetworkWithInputEncoding(
        n_input_dims=2, n_output_dims=1,
        encoding_config=encoding_config, network_config=network_config
    ))



