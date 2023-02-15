import torch
import numpy as np
from torch.distributions.categorical import Categorical


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def make_transformer_model(params):
    encoder_layer = torch.nn.TransformerEncoderLayer(
        d_model=params.transformer.d_model, nhead=params.transformer.nhead
    )
    transformer_model = torch.nn.TransformerEncoder(
        encoder_layer,
        num_layers=params.transformer.num_layers,
        norm=params.transformer.norm,
        enable_nested_tensor=params.transformer.enable_nested_tensor,
    )
    return transformer_model


def make_model(params, input_size, output_size):
    models_list = []
    transformer_model = make_transformer_model(params)
    if params.mlps.encoder_output != 0 and params.mlps.decoder_input != 0:
        assert (
            params.mlps.encoder_output == params.mlps.decoder_input
        ), "the encoder output and decoder input must have the same size"
    if params.mlps.encoder_output != 0:
        encoder = MLP(
            input_size, params.mlps.encoder_hidden, params.mlps.encoder_output
        )
        models_list.append(encoder)
    models_list.append(transformer_model)
    if params.mlps.decoder_input != 0:
        decoder = MLP(
            params.mlps.decoder_input, params.mlps.decoder_hidden, output_size
        )
        models_list.append(decoder)
    return torch.nn.Sequential(*models_list)


class Mediator(torch.nn.Module):
    def __init__(self, num_actions=2, num_input=1, params=None, fix_input_output=False):
        super().__init__()
        if fix_input_output:
            self.network = make_model(params, num_input, num_actions)
        else:
            self.network = make_transformer_model(params)

    def forward(self, x):
        logits = self.network(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return (action, probs.log_prob(action), probs.entropy(), None)
