import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .data_preprocessing import PAD_token


class EncoderRNN(nn.Module):
    def __init__(
        self,
        input_vocab_size,
        embedding_dim,
        hidden_size,
        num_layers=1,
        dropout=0.1,
        bidirectional=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(input_vocab_size, embedding_dim, padding_idx=PAD_token)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
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


class Attention(nn.Module):
    def __init__(self, enc_output_dim, dec_hidden_size):
        super().__init__()
        self.enc_proj = nn.Linear(enc_output_dim, dec_hidden_size, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, src_lengths):
        """
        decoder_hidden: (batch_size, dec_hidden_size)
        encoder_outputs: (src_len, batch_size, enc_output_dim)
        src_lengths: list[int] length=batch_size
        """
        src_len, batch_size, _ = encoder_outputs.shape
        enc_proj = self.enc_proj(encoder_outputs)  # (src_len, batch, dec_hidden)

        scores = torch.einsum("bd,sbd->bs", decoder_hidden, enc_proj)  # (batch, src_len)

        if src_lengths is not None:
            lengths = torch.as_tensor(src_lengths, device=encoder_outputs.device)
            positions = torch.arange(src_len, device=encoder_outputs.device).unsqueeze(0)
            mask = positions < lengths.unsqueeze(1)  # (batch, src_len)
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=1)  # (batch, src_len)
        context = torch.einsum("bs,sbe->be", attn_weights, encoder_outputs)  # (batch, enc_output_dim)
        return context, attn_weights


class AttnDecoderRNN(nn.Module):
    def __init__(
        self,
        output_vocab_size,
        embedding_dim,
        hidden_size,
        enc_output_dim,
        num_layers=1,
        dropout=0.1,
    ):
        super().__init__()
        self.output_vocab_size = output_vocab_size
        self.hidden_size = hidden_size
        self.enc_output_dim = enc_output_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_vocab_size, embedding_dim, padding_idx=PAD_token)
        self.embedding_dropout = nn.Dropout(dropout)

        self.attention = Attention(enc_output_dim=enc_output_dim, dec_hidden_size=hidden_size)
        self.lstm = nn.LSTM(
            embedding_dim + enc_output_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc_out = nn.Linear(hidden_size + enc_output_dim, output_vocab_size)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, input_step, hidden, cell, encoder_outputs, src_lengths):
        """
        input_step: (batch_size,)
        hidden, cell: (num_layers, batch_size, hidden_size)
        encoder_outputs: (src_len, batch_size, enc_output_dim)
        """
        input_step = input_step.unsqueeze(0)  # (1, batch)
        embedded = self.embedding(input_step)  # (1, batch, emb)
        embedded = self.embedding_dropout(embedded)

        dec_hidden_last = hidden[-1]  # (batch, hidden)
        context, _ = self.attention(dec_hidden_last, encoder_outputs, src_lengths)  # (batch, enc_dim)
        context = context.unsqueeze(0)  # (1, batch, enc_dim)

        lstm_input = torch.cat([embedded, context], dim=2)  # (1, batch, emb+enc_dim)
        outputs, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        outputs = outputs.squeeze(0)  # (batch, hidden)
        context = context.squeeze(0)  # (batch, enc_dim)
        logits = self.fc_out(self.out_dropout(torch.cat([outputs, context], dim=1)))  # (batch, vocab)
        return logits, hidden, cell


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


class Seq2SeqAttn(nn.Module):
    def __init__(self, encoder, decoder, device, enc_bidirectional=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.enc_bidirectional = enc_bidirectional

        enc_hidden_size = encoder.hidden_size
        enc_out_dim = enc_hidden_size * (2 if enc_bidirectional else 1)

        self.init_hidden = nn.Linear(enc_out_dim, decoder.hidden_size)
        self.init_cell = nn.Linear(enc_out_dim, decoder.hidden_size)

    def _init_dec_state(self, enc_state):
        hidden, cell = enc_state
        if self.enc_bidirectional:
            layer_start = (self.encoder.num_layers - 1) * 2
            h_fwd = hidden[layer_start]  # (batch, hid)
            h_bwd = hidden[layer_start + 1]
            c_fwd = cell[layer_start]
            c_bwd = cell[layer_start + 1]
            h_cat = torch.cat([h_fwd, h_bwd], dim=1)
            c_cat = torch.cat([c_fwd, c_bwd], dim=1)
        else:
            h_cat = hidden[-1]
            c_cat = cell[-1]

        h0 = torch.tanh(self.init_hidden(h_cat)).unsqueeze(0)
        c0 = torch.tanh(self.init_cell(c_cat)).unsqueeze(0)
        if self.decoder.num_layers > 1:
            h0 = h0.repeat(self.decoder.num_layers, 1, 1)
            c0 = c0.repeat(self.decoder.num_layers, 1, 1)
        return h0, c0

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.size(1)
        trg_len = trg.size(0)
        vocab_size = self.decoder.output_vocab_size

        outputs = torch.zeros(trg_len, batch_size, vocab_size, device=self.device)

        encoder_outputs, enc_state = self.encoder(src, src_lengths)
        hidden, cell = self._init_dec_state(enc_state)

        input_step = trg[0, :]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(
                input_step, hidden, cell, encoder_outputs, src_lengths
            )
            outputs[t] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_step = trg[t] if teacher_force else top1

        return outputs
