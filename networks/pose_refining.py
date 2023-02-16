import math

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
            self,
            frame_size,
            latent_size=32,
            hidden_size=256,
            num_condition_frames=1,
            num_future_frames=1
    ):
        super().__init__()

        input_size = frame_size * (num_future_frames + num_condition_frames)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(frame_size + hidden_size, hidden_size)
        self.m = nn.Linear(frame_size + hidden_size, latent_size)
        self.v = nn.Linear(frame_size + hidden_size, latent_size)

    def encode(self, x, c):
        h1 = nn.ELU()(self.fc1(torch.cat([x, c], dim=1)))
        h2 = nn.ELU()(self.fc2(torch.cat([x, h1], dim=1)))
        s = torch.cat([x, h2], dim=1)
        return self.m(s), self.v(s)

    def forward(self, x, c):
        m, v = self.encode(x, c)
        std = torch.exp(0.5 * v)
        eps = torch.randn_like(std)
        z = m + eps * std
        return z, m, v


class MixedDecoder(nn.Module):
    def __init__(
            self,
            frame_size,
            latent_size=32,
            hidden_size=256,
            num_condition_frames=1,
            num_future_frames=1,
            num_experts=6,
            gate_hidden_size=64,
    ):
        super().__init__()

        input_size = latent_size + frame_size * num_condition_frames
        inter_size = latent_size + hidden_size
        output_size = num_future_frames * frame_size

        self.fc1 = (
            nn.Parameter(torch.empty(num_experts, input_size, hidden_size)),
            nn.Parameter(torch.empty(num_experts, hidden_size)),
            nn.ELU(),
        )

        self.fc2 = (
            nn.Parameter(torch.empty(num_experts, inter_size, hidden_size)),
            nn.Parameter(torch.empty(num_experts, hidden_size)),
            nn.ELU(),
        )

        self.fc3 = (
            nn.Parameter(torch.empty(num_experts, inter_size, output_size)),
            nn.Parameter(torch.empty(num_experts, output_size)),
            None,
        )

        for index, (weight, bias, _) in enumerate([self.fc1, self.fc2, self.fc3]):
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(bias, -bound, bound)
            self.register_parameter(f"w{index}", weight)
            self.register_parameter(f"b{index}", bias)

        self.gate = nn.Sequential(
            nn.Linear(input_size, gate_hidden_size), nn.ELU(),
            nn.Linear(gate_hidden_size, gate_hidden_size), nn.ELU(),
            nn.Linear(gate_hidden_size, num_experts),
        )

    def forward(self, z, c):
        coefficients = nn.Softmax(dim=1)(self.gate(torch.cat([z, c], dim=1)))
        output = c
        for (weight, bias, activation) in [self.fc1, self.fc2, self.fc3]:
            flat_weight = weight.flatten(start_dim=1, end_dim=2)
            mixed_weight = torch.matmul(coefficients, flat_weight).view(
                coefficients.shape[0], *weight.shape[1:3]
            )

            layer_input = torch.cat((z, output), dim=1).unsqueeze(1)
            mixed_bias = torch.matmul(coefficients, bias).unsqueeze(1)
            layer_output = torch.baddbmm(mixed_bias, layer_input, mixed_weight).squeeze(1)
            output = activation(layer_output) if activation is not None else layer_output
        return output


class PoseMixtureVAE(nn.Module):
    def __init__(
        self,
        normalization,
        frame_size,
        latent_size=32,
        hidden_size=256,
        num_condition_frames=1,
        num_future_frames=1,
        num_experts=6
    ):
        super().__init__()

        self.data_max = normalization["max"]
        self.data_min = normalization["min"]
        self.num_future_frames = num_future_frames
        args = (
            frame_size,
            latent_size,
            hidden_size,
            num_condition_frames,
            num_future_frames,
        )

        self.encoder = Encoder(*args)
        self.decoder = MixedDecoder(*args, num_experts)

    def normalize(self, t):
        return 2 * (t - self.data_min) / (self.data_max - self.data_min) - 1

    def denormalize(self, t):
        return (t + 1) * (self.data_max - self.data_min) / 2 + self.data_min

    def encode(self, x, c):
        _, m, v = self.encoder(x, c)
        return m, v

    def forward(self, x, c):
        z, m, v = self.encoder(x, c)
        output = self.decoder(z, c)
        return output, m, v

    def sample(self, z, c):
        return self.decoder(z, c)
