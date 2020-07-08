import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding_layer = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout = dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # input must be [sentence length, batch size]
        # embedded = [sentence length, batch size, embedding dim]
        embedded_input = self.dropout(self.embedding_layer(input))

        #output = [sentence length, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        output, (hidden, cell) = self.rnn(embedded_input)

        return hidden, cell