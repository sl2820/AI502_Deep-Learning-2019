import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding_process = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout = dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # input must be: [sentence length, batch size]
        # embedded will be: [sentence length, batch size, embedding dim]
        embedded_input = self.dropout(self.embedding_process(input))

        #output will be: [sentence length, batch size, hid dim * n directions]
        #hidden states: [n layers * n directions, batch size, hid dim]
        #cells: [n layers * n directions, batch size, hid dim]
        output, (hidden_state, cells) = self.rnn(embedded_input)

        return hidden_state, cells


class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding_process = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout = dropout)
        self.final_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden_state, cells):
        # Input Variable Dimensions
        # input: [batch size]
        # hidden_stete: [n layers, batch size, hid dim]
        # cells: [n layers, batch size, hid dim]

        # input = [1, batch size]
        input = input.unsqueeze(0)

        # embeddid_input: [1, batch size, embedding dim]
        embedded_input = self.dropout(self.embedding_process(input))

        #output: [sent len = 1, batch size, hid dim]
        #hidden_state: [n layers, batch size, hid dim]
        #cell: [n layers, batch size, hid dim]
        output, (hidden_state, cells) = self.rnn(embedded_input, (hidden_state, cells))

        #prediction: [batch size, output dim]
        prediction = self.final_out(output.squeeze(0))

        return prediction, hidden_state, cells