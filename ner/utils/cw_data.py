# -*- coding: utf-8 -*-
# @Author: Liu Wei
# @Date:   2019-05-09


"""
This data is for character-based and word-based models

"""

import sys
import numpy as np
import pickle
from ner.utils.alphabet import Alphabet
from ner.utils.functions import *
from ner.utils.gazetteer import Gazetteer

# from ner.seg.jieba import sent_to_words


START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"
NULLKEY = "-null-"


class Data:
    def __init__(self):
        self.MAX_SENTENCE_LENGTH = 250
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = True
        self.norm_char_emb = True
        self.norm_word_emb = True
        self.norm_biword_emb = True
        self.norm_gaz_emb = False
        self.word_alphabet = Alphabet('word')
        self.biword_alphabet = Alphabet('biword')
        self.char_alphabet = Alphabet('character')
        self.label_alphabet = Alphabet('label', True)
        self.gaz_lower = False
        self.gaz = Gazetteer(self.gaz_lower)
        self.gaz_alphabet = Alphabet('gaz')
        self.HP_fix_gaz_emb = False
        self.HP_use_gaz = True

        self.tagScheme = "NoSeg"
        self.char_features = "LSTM"

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []

        ## for evaluate word ner labels
        self.train_char_labels = []
        self.dev_char_labels = []
        self.test_char_labels = []

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []
        self.use_bigram = True
        self.word_emb_dim = 200
        self.biword_emb_dim = 50
        self.char_emb_dim = 50
        self.gaz_emb_dim = 200
        self.gaz_dropout = 0.5
        self.pretrain_word_embedding = None
        self.pretrain_biword_embedding = None
        self.pretrain_char_embedding = None
        self.label_size = 0
        self.word_alphabet_size = 0
        self.biword_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0
        ### hyperparameters
        self.HP_iteration = 100
        self.HP_batch_size = 10
        self.HP_char_hidden_dim = 100
        self.HP_hidden_dim = 100
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True
        self.HP_use_char = False
        self.HP_gpu = False
        self.HP_lr = 0.015
        self.HP_lr_decay = 0.05
        self.HP_clip = 5.0
        self.HP_momentum = 0

        self.unknow_index = {}

    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     Tag          scheme: %s" % (self.tagScheme))
        print("     MAX SENTENCE LENGTH: %s" % (self.MAX_SENTENCE_LENGTH))
        print("     MAX   WORD   LENGTH: %s" % (self.MAX_WORD_LENGTH))
        print("     Number   normalized: %s" % (self.number_normalized))
        print("     Use          bigram: %s" % (self.use_bigram))
        print("     Word  alphabet size: %s" % (self.word_alphabet_size))
        print("     Biword alphabet size: %s" % (self.biword_alphabet_size))
        print("     Char  alphabet size: %s" % (self.char_alphabet_size))
        print("     Gaz   alphabet size: %s" % (self.gaz_alphabet.size()))
        print("     Label alphabet size: %s" % (self.label_alphabet_size))
        print("     Word embedding size: %s" % (self.word_emb_dim))
        print("     Biword embedding size: %s" % (self.biword_emb_dim))
        print("     Char embedding size: %s" % (self.char_emb_dim))
        print("     Gaz embedding size: %s" % (self.gaz_emb_dim))
        print("     Norm     word   emb: %s" % (self.norm_word_emb))
        print("     Norm     biword emb: %s" % (self.norm_biword_emb))
        print("     Norm     gaz    emb: %s" % (self.norm_gaz_emb))
        print("     Norm   gaz  dropout: %s" % (self.gaz_dropout))
        print("     Train instance number: %s" % (len(self.train_texts)))
        print("     Dev   instance number: %s" % (len(self.dev_texts)))
        print("     Test  instance number: %s" % (len(self.test_texts)))
        print("     Raw   instance number: %s" % (len(self.raw_texts)))
        print("     Hyperpara  iteration: %s" % (self.HP_iteration))
        print("     Hyperpara  batch size: %s" % (self.HP_batch_size))
        print("     Hyperpara          lr: %s" % (self.HP_lr))
        print("     Hyperpara    lr_decay: %s" % (self.HP_lr_decay))
        print("     Hyperpara     HP_clip: %s" % (self.HP_clip))
        print("     Hyperpara    momentum: %s" % (self.HP_momentum))
        print("     Hyperpara  hidden_dim: %s" % (self.HP_hidden_dim))
        print("     Hyperpara     dropout: %s" % (self.HP_dropout))
        print("     Hyperpara  lstm_layer: %s" % (self.HP_lstm_layer))
        print("     Hyperpara      bilstm: %s" % (self.HP_bilstm))
        print("     Hyperpara         GPU: %s" % (self.HP_gpu))
        print("     Hyperpara     use_gaz: %s" % (self.HP_use_gaz))
        print("     Hyperpara fix gaz emb: %s" % (self.HP_fix_gaz_emb))
        print("     Hyperpara    use_char: %s" % (self.HP_use_char))
        if self.HP_use_char:
            print("             Char_features: %s" % (self.char_features))
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def refresh_label_alphabet(self, input_file):
        old_size = self.label_alphabet_size
        self.label_alphabet.clear(True)
        in_lines = open(input_file, 'r').readlines()
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split()
                label = pairs[-1]
                self.label_alphabet.add(label)
        self.label_alphabet_size = self.label_alphabet.size()
        startS = False
        startB = False
        for label, _ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"
        self.fix_alphabet()
        print("Refresh label alphabet finished: old:%s -> new:%s" % (old_size, self.label_alphabet_size))

    def build_char_alphabet(self, input_file):
        """
        init character alphabet and label alphabet
        """
        with open(input_file, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            for idx in range(len(lines)):
                line = lines[idx]
                if len(line) > 2:
                    pairs = line.strip().split()
                    char = pairs[0]
                    if self.number_normalized:
                        char = normalize_word(char)
                    label = pairs[-1]
                    self.label_alphabet.add(label)
                    self.char_alphabet.add(char)

                    if idx < len(lines) - 1 and len(lines[idx+1]) > 2:
                        bichar = char + lines[idx+1].strip().split()[0]
                    else:
                        bichar = char + NULLKEY
                    self.biword_alphabet.add(bichar)
        self.char_alphabet_size = self.char_alphabet.size()
        self.biword_alphabet_size = self.biword_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()

        # BMES or BIO ?
        startS = False
        startB = False
        for label, _ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"

    def build_word_alphabet(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if len(line) > 2:
                    pairs = line.strip().split()
                    word = pairs[0]
                    if self.number_normalized:
                        word = normalize_word(word)
                    label = pairs[-1]

                    self.label_alphabet.add(label)
                    self.word_alphabet.add(word)
        self.label_alphabet_size = self.label_alphabet.size()
        self.word_alphabet_size = self.word_alphabet.size()

        # BMES or BIO ?
        startS = False
        startB = False
        for label, _ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"


    def fix_alphabet(self):
        self.word_alphabet.close()
        self.biword_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close()
        self.gaz_alphabet.close()

    def build_char_pretrain_emb(self, emb_path):
        print("build character pretrain emb...")
        self.pretrain_char_embedding, self.char_emb_dim = build_pretrain_embedding(emb_path, self.char_alphabet,
                                                                                   self.char_emb_dim,
                                                                                   self.norm_char_emb)

    def build_biword_pretrain_emb(self, emb_path):
        print("build biword pretrain emb...")
        self.pretrain_biword_embedding, self.biword_emb_dim = build_pretrain_embedding(emb_path, self.biword_alphabet,
                                                                                       self.biword_emb_dim,
                                                                                       self.norm_biword_emb)

    def build_small_word_pretrain_emb(self, emb_path):
        print("build small word pretrain emb...")
        self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(emb_path, self.word_alphabet,
                                                                                   self.word_emb_dim,
                                                                                   self.norm_word_emb)

    def build_word_pretrain_emb(self, emb_path=None):
        print("build word pretrain emb...")
        self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding_for_gaz(self.word_alphabet,
                                                                                         embedding_dir='data/small_embeddings',
                                                                                         embedding_name='embed',
                                                                                         file_num=2,
                                                                                         embedd_dim=self.word_emb_dim,
                                                                                         norm=self.norm_word_emb)

    def generate_instance_char(self, input_file, name):
        """
        each instance include:
            chars, bichars, labels
            char_ids, bichar_ids, label_ids
        """
        self.fix_alphabet()

        if name == "train":
            self.train_texts, self.train_Ids = read_instance_char(input_file, self.char_alphabet,
                                                                  self.biword_alphabet, self.label_alphabet,
                                                                  self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance_char(input_file, self.char_alphabet,
                                                                  self.biword_alphabet, self.label_alphabet,
                                                                  self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance_char(input_file, self.char_alphabet,
                                                                  self.biword_alphabet, self.label_alphabet,
                                                                  self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_instance_char(input_file, self.char_alphabet,
                                                                  self.biword_alphabet, self.label_alphabet,
                                                                  self.number_normalized, self.MAX_SENTENCE_LENGTH)

    def generate_instance_word(self, input_file, name):
        self.fix_alphabet()

        if name == "train":
            self.train_texts, self.train_Ids = read_instance_word(input_file, self.word_alphabet,
                                                                  self.label_alphabet, self.number_normalized,
                                                                  self.MAX_SENTENCE_LENGTH)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance_word(input_file, self.word_alphabet,
                                                                  self.label_alphabet, self.number_normalized,
                                                                  self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance_word(input_file, self.word_alphabet,
                                                                  self.label_alphabet, self.number_normalized,
                                                                  self.MAX_SENTENCE_LENGTH)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_instance_word(input_file, self.word_alphabet,
                                                                  self.label_alphabet, self.number_normalized,
                                                                  self.MAX_SENTENCE_LENGTH)

    def generate_instance_with_gaz_no_char(self, input_file, name):
        """
        every instance include:
            words, biwords, gazs, labels
            word_Ids, biword_Ids, gazs_Ids, label_Ids
        """
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_instance_with_gaz_no_char(input_file, self.gaz,
                                                                                   self.word_alphabet,
                                                                                   self.biword_alphabet,
                                                                                   self.gaz_alphabet,
                                                                                   self.label_alphabet,
                                                                                   self.number_normalized,
                                                                                   self.MAX_SENTENCE_LENGTH, self.unknow_index)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance_with_gaz_no_char(input_file, self.gaz, self.word_alphabet,
                                                                               self.biword_alphabet, self.gaz_alphabet,
                                                                               self.label_alphabet,
                                                                               self.number_normalized,
                                                                               self.MAX_SENTENCE_LENGTH, self.unknow_index)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance_with_gaz_no_char(input_file, self.gaz,
                                                                                 self.word_alphabet,
                                                                                 self.biword_alphabet,
                                                                                 self.gaz_alphabet, self.label_alphabet,
                                                                                 self.number_normalized,
                                                                                 self.MAX_SENTENCE_LENGTH, self.unknow_index)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_instance_with_gaz_no_char_ls(input_file, self.gaz,
                                                                                    self.word_alphabet,
                                                                                    self.biword_alphabet,
                                                                                    self.gaz_alphabet,
                                                                                    self.label_alphabet,
                                                                                    self.number_normalized,
                                                                                    self.MAX_SENTENCE_LENGTH)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s" % (name))

    def write_decoded_results(self, output_file, predict_results, name):
        fout = open(output_file, 'w')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
            content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert (sent_num == len(content_list))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx])
            for idy in range(sent_length):
                ## content_list[idx] is a list with [word, char, label]
                fout.write(content_list[idx][0][idy].encode('utf-8') + " " + predict_results[idx][idy] + '\n')

            fout.write('\n')
        fout.close()
        print("Predict %s result has been written into file. %s" % (name, output_file))


    def get_char_label(self, char_file, name):
        """
        get char file to evaluate the result of word-based NER model
        Args:
            char_file:
            name:
        """
        if name == 'dev':
            self.dev_char_labels = read_char_file(char_file)
        elif name == 'test':
            self.test_char_labels = read_char_file(char_file)
        elif name == 'train':
            self.train_char_labels = read_char_file(char_file)



