import torch
from torch import nn

import math


class SegformerDecoder(nn.Module):
    def __init__(self, input_dim, hidden_size, num_hidden_states, dropout_prob=0.1, nclasses=19):
        super().__init__()

        self.linear_c = nn.ModuleList([nn.Linear(input_dim, hidden_size) for _ in range(num_hidden_states)])

        self.linear_fuse = nn.Conv2d(
            in_channels=hidden_size * num_hidden_states,
            out_channels=hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Conv2d(hidden_size, nclasses, kernel_size=1)

        self.apply(self._init_weights)

    
    def _init_weights(self, module, std=0.02):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
                

    def forward(self, encoder_hidden_states):
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            encoder_hidden_state = encoder_hidden_state[:,1:,:]

            encoder_hidden_state = mlp(encoder_hidden_state)

            height = width = int(math.sqrt(encoder_hidden_state.shape[1]))
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
            # upsample
            # encoder_hidden_state = F.interpolate(encoder_hidden_state, size=(128,128), mode="bilinear", align_corners=False)

            all_hidden_states += (encoder_hidden_state,)

        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        return logits
