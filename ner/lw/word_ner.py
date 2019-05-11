__author__ = " liuwei"

"""
word-based biLSTM-CRF model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ner.functions.utils import random_embedding
from ner.functions.utils import reverse_padded_sequence
from ner.modules.word_bilstm import Word_BiLSTM
from ner.model.crf import CRF

class Word_NER(nn.Module):
    def __init__(self, data):
        super(Word_NER, self).__init__()
        print("Build Character-based BiLSTM-CRF....")

        self.gpu = data.HP_gpu
        self.embedding_dim = data.word_emb_dim
        self.hidden_dim = data.HP_hidden_dim

        self.word_embedding = nn.Embedding(data.word_alphabet.size(), self.embedding_dim)

        self.lstm = Word_BiLSTM(data)
        self.hidden2tag = nn.Linear(self.hidden_dim * 2, data.label_alphabet_size + 2)
        self.crf = CRF(data.label_alphabet_size, self.gpu)


        if data.pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(
                torch.from_numpy(data.pretrain_word_embedding)
            )
        else:
            self.word_embedding.weight.data.copy_(
                random_embedding(data.word_alphabet.size(), self.embedding_dim)
            )

        if self.gpu:
            self.word_embedding = self.word_embedding.cuda()
            self.hidden2tag = self.hidden2tag.cuda()


    def neg_log_likelihood_loss(self, word_inputs, word_seq_lengths, batch_label, mask):
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)

        lengths = list(map(int, word_seq_lengths))
        reverse_word_inputs = reverse_padded_sequence(word_inputs, lengths)

        word_emb = self.word_embedding(word_inputs)
        reverse_word_emb = self.word_embedding(reverse_word_inputs)

        lstm_out, _ = self.lstm((word_emb, reverse_word_emb), word_seq_lengths)

        # softmax
        outs = self.hidden2tag(lstm_out)

        # CRF
        ## crf and loss
        loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
        _, tag_seq = self.crf._viterbi_decode(outs, mask)

        return loss, tag_seq


    def forward(self, word_inputs, word_seq_lengths, mask):
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)

        lengths = list(map(int, word_seq_lengths))
        reverse_word_inputs = reverse_padded_sequence(word_inputs, lengths)

        word_emb = self.word_embedding(word_inputs)
        reverse_word_emb = self.word_embedding(reverse_word_inputs)

        lstm_out, _ = self.lstm((word_emb, reverse_word_emb), word_seq_lengths)

        # softmax
        outs = self.hidden2tag(lstm_out)

        # CRF
        _, tag_seq = self.crf._viterbi_decode(outs, mask)

        return tag_seq
