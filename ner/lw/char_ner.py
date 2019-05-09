__author__ = " liuwei"

"""
character-based biLSTM-CRF model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ner.functions.utils import random_embedding
from ner.functions.utils import reverse_padded_sequence
from ner.modules.char_bilstm import Char_BiLSTM
from ner.model.crf import CRF

class Char_NER(nn.Module):
    def __init__(self, data):
        super(Char_NER, self).__init__()
        print("Build Character-based BiLSTM-CRF....")

        self.gpu = data.HP_gpu
        self.embedding_dim = data.char_emb_dim
        self.hidden_dim = data.HP_hidden_dim

        self.char_embedding = nn.Embedding(data.char_alphabet.size(), self.embedding_dim)

        self.lstm = Char_BiLSTM(data)
        self.hidden2tag = nn.Linear(self.hidden_dim * 2, data.label_alphabet_size + 2)
        self.crf = CRF(data.label_alphabet_size, self.gpu)


        if data.pretrain_char_embedding is not None:
            self.char_embedding.weight.data.copy_(
                torch.from_numpy(data.pretrain_char_embedding)
            )
        else:
            self.char_embedding.weight.data.copy_(
                random_embedding(data.char_alphabet.size(), self.embedding_dim)
            )

        if self.gpu:
            self.char_embedding = self.char_embedding.cuda()
            self.hidden2tag = self.hidden2tag.cuda()


    def neg_log_likelihood_loss(self, char_inputs, char_seq_lengths, batch_label, mask):
        batch_size = char_inputs.size(0)
        sent_len = char_inputs.size(1)

        lengths = list(map(int, char_seq_lengths))
        reverse_char_inputs = reverse_padded_sequence(char_inputs, lengths)

        char_emb = self.char_embedding(char_inputs)
        reverse_char_emb = self.char_embedding(reverse_char_inputs)

        lstm_out, _ = self.lstm((char_emb, reverse_char_emb), char_seq_lengths)

        # softmax
        outs = self.hidden2tag(lstm_out)

        # CRF
        ## crf and loss
        loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
        _, tag_seq = self.crf._viterbi_decode(outs, mask)

        return loss, tag_seq


    def forward(self, char_inputs, char_seq_lengths, mask):
        batch_size = char_inputs.size(0)
        sent_len = char_inputs.size(1)

        lengths = list(map(int, char_seq_lengths))
        reverse_char_inputs = reverse_padded_sequence(char_inputs, lengths)

        char_emb = self.char_embedding(char_inputs)
        reverse_char_emb = self.char_embedding(reverse_char_inputs)

        lstm_out, _ = self.lstm((char_emb, reverse_char_emb), char_seq_lengths)

        # softmax
        outs = self.hidden2tag(lstm_out)

        # CRF
        _, tag_seq = self.crf._viterbi_decode(outs, mask)

        return tag_seq
