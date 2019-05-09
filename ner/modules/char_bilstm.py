__author__ = "liuwei"

"""
the character-based bilstm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ner.functions.utils import reverse_padded_sequence


class Char_BiLSTM(nn.Module):
    def __init__(self, data):
        super(Char_BiLSTM, self).__init__()
        print("Build character-based BiLSTM...")

        self.gpu = data.HP_gpu
        self.embedding_dim = data.char_emb_dim
        self.hidden_dim = data.HP_hidden_dim
        self.drop = nn.Dropout(data.HP_dropout)
        self.droplstm = nn.Dropout(data.HP_dropout)

        self.lstm_layer = data.HP_lstm_layer

        self.f_char_lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.lstm_layer,
                                     batch_first=True, bidirectional=False)
        self.b_char_lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.lstm_layer,
                                   batch_first=True, bidirectional=False)


        if self.gpu:
            self.drop = self.drop.cuda()
            self.droplstm = self.droplstm.cuda()
            # self.char_embedding = self.char_embedding.cuda()
            self.f_char_lstm = self.f_char_lstm.cuda()
            self.b_char_lstm = self.b_char_lstm.cuda()
            # self.hidden2tag = self.hidden2tag.cuda()


    def get_lstm_features(self, char_inputs, char_seq_length):
        """
        Args:
            char_inputs: [batch_size, sent_len]
            char_seq_length: [batch_size, 1]

        """
        f_inputs, b_inputs = char_inputs
        f_inputs = self.drop(f_inputs)
        b_inputs = self.drop(b_inputs)


        f_lstm_out, f_lstm_hidden = self.f_char_lstm(f_inputs)
        b_lstm_out, b_lstm_hidden = self.b_char_lstm(b_inputs)

        f_lstm_out = self.droplstm(f_lstm_out)
        b_lstm_out = self.droplstm(b_lstm_out)


        return f_lstm_out, b_lstm_out


    def forward(self, inputs, char_seq_length):
        f_lstm_out, b_lstm_out = self.get_lstm_features(inputs, char_seq_length)

        lengths = list(map(int, char_seq_length))
        rb_lstm_out = reverse_padded_sequence(b_lstm_out, lengths)

        lstm_out = torch.cat((f_lstm_out, rb_lstm_out), dim=-1)


        return lstm_out, (f_lstm_out, b_lstm_out)