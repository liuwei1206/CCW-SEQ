__author__ = "liuwei"

"""
A bilstm use for the sentences contain gaz msg
"""

import numpy as np
import torch
import torch.nn as nn

from ner.functions.utils import reverse_padded_sequence

class Gaz_BiLSTM(torch.nn.Module):
    def __init__(self, data, input_size, hidden_size):
        print("Build the Gaz bilstm...")
        super(Gaz_BiLSTM, self).__init__()

        self.gpu = data.HP_gpu
        self.batch_size = data.HP_batch_size
        self.drop = nn.Dropout(data.HP_dropout)
        self.droplstm = nn.Dropout(data.HP_dropout)

        self.f_lstm = nn.LSTM(input_size, hidden_size, num_layers=data.HP_lstm_layer, batch_first=True)
        self.b_lstm = nn.LSTM(input_size, hidden_size, num_layers=data.HP_lstm_layer, batch_first=True)

        if self.gpu:
            self.drop = self.drop.cuda()
            self.droplstm = self.droplstm.cuda()
            self.f_lstm = self.f_lstm.cuda()
            self.b_lstm = self.b_lstm.cuda()

    def weight_init_lstm(self):
        """
        use the trained weight to init the bilstm
        """
        if not self.use:
            return

        weight_file = "data/weights.hdf5"
        if not os.path.exists(weight_file):
            return

        ## for bilstm
        f_weight_ih_l0 = load_weights(weight_file, 'Char_BiLSTM_CRF', 'lstm.char_lstm.weight_ih_l0')
        f_weight_hh_l0 = load_weights(weight_file, 'Char_BiLSTM_CRF', 'lstm.char_lstm.weight_hh_l0')
        f_bias_ih_l0 = load_weights(weight_file, 'Char_BiLSTM_CRF', 'lstm.char_lstm.bias_ih_l0')
        f_bias_hh_l0 = load_weights(weight_file, 'Char_BiLSTM_CRF', 'lstm.char_lstm.bias_hh_l0')

        b_weight_ih_l0 = load_weights(weight_file, 'Char_BiLSTM_CRF', 'lstm.char_lstm.weight_ih_l0_reverse')
        b_weight_hh_l0 = load_weights(weight_file, 'Char_BiLSTM_CRF', 'lstm.char_lstm.weight_hh_l0_reverse')
        b_bias_ih_l0 = load_weights(weight_file, 'Char_BiLSTM_CRF', 'lstm.char_lstm.bias_ih_l0_reverse')
        b_bias_hh_l0 = load_weights(weight_file, 'Char_BiLSTM_CRF', 'lstm.char_lstm.bias_hh_l0_reverse')


        char_lstm_weight_ih_l0 = getattr(self.f_lstm, 'weight_ih_l0')
        char_lstm_weight_ih_l0.data[:, :] = f_weight_ih_l0
        char_lstm_weight_ih_l0.requires_grad = False
        char_lstm_weight_hh_l0 = getattr(self.f_lstm, 'weight_hh_l0')
        char_lstm_weight_hh_l0.data[:, :] = f_weight_hh_l0
        char_lstm_weight_hh_l0.requires_grad = False
        char_lstm_bias_ih_l0 = getattr(self.f_lstm, 'bias_ih_l0')
        char_lstm_bias_ih_l0.data[:] = f_bias_ih_l0
        char_lstm_bias_hh_l0 = getattr(self.f_lstm, 'bias_hh_l0')
        char_lstm_bias_hh_l0.data[:] = f_bias_hh_l0

        b_char_lstm_weight_ih_l0 = getattr(self.b_lstm, 'weight_ih_l0')
        b_char_lstm_weight_ih_l0.data[:, :] = b_weight_ih_l0
        b_char_lstm_weight_hh_l0 = getattr(self.b_lstm, 'weight_hh_l0')
        b_char_lstm_weight_hh_l0.data[:, :] = b_weight_hh_l0
        b_char_lstm_bias_ih_l0 = getattr(self.b_lstm, 'bias_ih_l0')
        b_char_lstm_bias_ih_l0.data[:] = b_bias_ih_l0
        b_char_lstm_bias_hh_l0 = getattr(self.b_lstm, 'bias_hh_l0')
        b_char_lstm_bias_hh_l0.data[:] = b_bias_hh_l0


    def get_lstm_features(self, inputs, word_seq_length):
        """
        get the output of bilstm. Note that inputs is forward and backward inputs
        Args:
            inputs: a tuple, each item size is [batch_size, sent_len, dim]
            word_seq_length: a [batch_size] tensor
        """
        # lengths = list(map(int, word_seq_length))
        f_inputs, b_inputs = inputs
        f_inputs = self.drop(f_inputs)
        b_inputs = self.drop(b_inputs)

        f_lstm_out, f_hidden = self.f_lstm(f_inputs)
        b_lstm_out, b_hidden = self.b_lstm(b_inputs)
        # b_lstm_out = reverse_padded_sequence(b_lstm_out, lengths)

        f_lstm_out = self.droplstm(f_lstm_out)
        b_lstm_out = self.droplstm(b_lstm_out)

        return f_lstm_out, b_lstm_out


    def forward(self, inputs, word_seq_length):
        """
        """
        f_lstm_out, b_lstm_out = self.get_lstm_features(inputs, word_seq_length)

        lengths = list(map(int, word_seq_length))
        rb_lstm_out = reverse_padded_sequence(b_lstm_out, lengths)

        lstm_out = torch.cat((f_lstm_out, rb_lstm_out), dim=-1)

        return lstm_out, (f_lstm_out, b_lstm_out)