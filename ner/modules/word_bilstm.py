__author__ = "liuwei"

"""
the word-based bilstm

just same as the character-based lstm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ner.functions.utils import reverse_padded_sequence


class Word_BiLSTM(nn.Module):
    def __init__(self, data):
        super(Word_BiLSTM, self).__init__()
        print("Build word-based BiLSTM...")

        self.gpu = data.HP_gpu
        self.embedding_dim = data.word_emb_dim
        self.hidden_dim = data.HP_hidden_dim
        self.drop = nn.Dropout(data.HP_dropout)
        self.droplstm = nn.Dropout(data.HP_dropout)

        self.lstm_layer = data.HP_lstm_layer

        self.f_word_lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.lstm_layer,
                                     batch_first=True, bidirectional=False)
        self.b_word_lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.lstm_layer,
                                   batch_first=True, bidirectional=False)


        if self.gpu:
            self.drop = self.drop.cuda()
            self.droplstm = self.droplstm.cuda()
            # self.char_embedding = self.char_embedding.cuda()
            self.f_word_lstm = self.f_word_lstm.cuda()
            self.b_word_lstm = self.b_word_lstm.cuda()
            # self.hidden2tag = self.hidden2tag.cuda()


    def get_lstm_features(self, word_inputs, word_seq_length):
        """
        Args:
            word_inputs: [batch_size, sent_len]
            word_seq_length: [batch_size, 1]

        """
        f_inputs, b_inputs = word_inputs
        f_inputs = self.drop(f_inputs)
        b_inputs = self.drop(b_inputs)


        f_lstm_out, f_lstm_hidden = self.f_word_lstm(f_inputs)
        b_lstm_out, b_lstm_hidden = self.b_word_lstm(b_inputs)

        f_lstm_out = self.droplstm(f_lstm_out)
        b_lstm_out = self.droplstm(b_lstm_out)


        return f_lstm_out, b_lstm_out


    def forward(self, inputs, word_seq_length):
        f_lstm_out, b_lstm_out = self.get_lstm_features(inputs, word_seq_length)

        lengths = list(map(int, word_seq_length))
        rb_lstm_out = reverse_padded_sequence(b_lstm_out, lengths)

        lstm_out = torch.cat((f_lstm_out, rb_lstm_out), dim=-1)


        return lstm_out, (f_lstm_out, b_lstm_out)