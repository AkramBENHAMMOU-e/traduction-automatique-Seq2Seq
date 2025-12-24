import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .data_preprocessing import PAD_token


class EncoderRNN(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_vocab_size, embedding_dim, padding_idx=PAD_token)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, src, src_lengths):
        """
        src: (src_len, batch_size)
        src_lengths: list[int] of length batch_size
        """
        embedded = self.embedding(src)  # (src_len, batch_size, embedding_dim)

        packed = pack_padded_sequence(embedded, src_lengths, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(packed_outputs)  # (src_len, batch_size, hidden_size)

        return outputs, (hidden, cell)


class DecoderRNN(nn.Module):
    def __init__(self, output_vocab_size, embedding_dim, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.output_vocab_size = output_vocab_size

        self.embedding = nn.Embedding(output_vocab_size, embedding_dim, padding_idx=PAD_token)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_size, output_vocab_size)

    def forward(self, input_step, hidden, cell):
        """
        input_step: (batch_size,)  - token ids for current time step
        hidden, cell: LSTM states from previous step
        """
        input_step = input_step.unsqueeze(0)  # (1, batch_size)
        embedded = self.embedding(input_step)  # (1, batch_size, embedding_dim)

        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        predictions = self.fc_out(outputs.squeeze(0))  # (batch_size, vocab_size)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        """
        src: (src_len, batch_size)
        src_lengths: list[int]
        trg: (trg_len, batch_size)
        """
        batch_size = trg.size(1)
        trg_len = trg.size(0)
        vocab_size = self.decoder.output_vocab_size

        outputs = torch.zeros(trg_len, batch_size, vocab_size, device=self.device)

        _, (hidden, cell) = self.encoder(src, src_lengths)

        # first input to the decoder is the <SOS> token for each sentence
        input_step = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_step, hidden, cell)
            outputs[t] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)

            input_step = trg[t] if teacher_force else top1

        return outputs

