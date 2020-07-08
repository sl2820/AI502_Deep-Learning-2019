import torch
import torch.nn as nn
import random


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding_process = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers)

    def forward(self, input):
        # input must be: [sentence length, batch size]
        # embedded will be: [sentence length, batch size, embedding dim]
        embedded_input = self.embedding_process(input)

        #output will be: [sentence length, batch size, hid dim * n directions]
        #hidden states: [n layers * n directions, batch size, hid dim]
        #cells: [n layers * n directions, batch size, hid dim]
        output, (hidden_state, cells) = self.rnn(embedded_input)

        return hidden_state, cells


class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding_process = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers)
        self.final_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden_state, cells):
        # Input Variable Dimensions
        # input: [batch size]
        # hidden_stete: [n layers, batch size, hid dim]
        # cells: [n layers, batch size, hid dim]

        # input = [1, batch size]
        input = input.unsqueeze(0)

        # embeddid_input: [1, batch size, embedding dim]
        embedded_input = self.embedding_process(input)

        #output: [sent len = 1, batch size, hid dim]
        #hidden_state: [n layers, batch size, hid dim]
        #cell: [n layers, batch size, hid dim]
        output, (hidden_state, cells) = self.rnn(embedded_input, (hidden_state, cells))

        #prediction: [batch size, output dim]
        prediction = self.final_out(output.squeeze(0))

        return prediction, hidden_state, cells


class Encoder_with_dorpout(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers):
        super().__init__()
        self.drop_val = 0.5
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding_process = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout = self.drop_val)
        self.dropout = nn.Dropout(self.drop_val)

    def forward(self, input):
        # input must be: [sentence length, batch size]
        # embedded will be: [sentence length, batch size, embedding dim]
        embedded_input = self.dropout(self.embedding_process(input))

        #output will be: [sentence length, batch size, hid dim * n directions]
        #hidden states: [n layers * n directions, batch size, hid dim]
        #cells: [n layers * n directions, batch size, hid dim]
        output, (hidden_state, cells) = self.rnn(embedded_input)

        return hidden_state, cells


class Decoder_with_dropout(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers):
        super().__init__()
        self.drop_val = 0.5
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding_process = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=self.drop_val)
        self.final_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(self.drop_val)

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



class seq2seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        if encoder.hidden_dim != decoder.hidden_dim:
            print("This Code might not run because")
            print("Hidden dimesions are not equal between Encoder and Decoder")
        if encoder.n_layers != decoder.n_layers:
            print("This Code might not run because")
            print("Numbers of Layers are not equal between Encoder and Decoder")

    def forward(self, source, target):
        batch_size = target.shape[1]
        max_length = target.shape[0]
        target_vocab = self.decoder.output_dim

        output_list = torch.zeros(max_length, batch_size, target_vocab).to(self.device)
        hidden_state, cells = self.encoder(source)

        input = target[0,:]
        for text in range(1, max_length):
            output, hidden_state, cells = self.decoder(input, hidden_state, cells)
            output_list[text] = output
            top_1 = output.argmax(1)


        return output_list