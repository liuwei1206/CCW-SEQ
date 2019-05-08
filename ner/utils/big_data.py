# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-14 17:34:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-01-29 15:26:51

"""
big_data.py is quite different from data.py, in this file, we use a more big embedding file
for the NER task. But due to memory limitation, we should use a lots of small embedding files
to implement the code. The diffirent mainly including:
1. build the gaz tree, here, we use vocab.txt to build it
    build_gaz_file()

2. get gaz alphabet. with the help of gaz tree
    build_gaz_alphabet()

3. get intances with gaz
    generate_instance_with_gaz_no_char()

4. load pretrained word embeddings
    build_word_pretrain_emb() for gaz

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
        self.MAX_SENTENCE_LENGTH = 5000
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = True
        self.norm_word_emb = True
        self.norm_biword_emb = True
        self.norm_gaz_emb = False
        self.word_alphabet = Alphabet('word')
        self.biword_alphabet = Alphabet('biword')
        self.char_alphabet = Alphabet('character')
        # self.word_alphabet.add(START)
        # self.word_alphabet.add(UNKNOWN)
        # self.char_alphabet.add(START)
        # self.char_alphabet.add(UNKNOWN)
        # self.char_alphabet.add(PADDING)
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

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []
        self.use_bigram = True
        self.word_emb_dim = 100
        self.biword_emb_dim = 50
        self.char_emb_dim = 30
        self.gaz_emb_dim = 200
        self.gaz_dropout = 0.5
        self.pretrain_word_embedding = None
        self.pretrain_biword_embedding = None
        self.pretrain_gaz_embedding = None
        self.label_size = 0
        self.word_alphabet_size = 0
        self.biword_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0
        ### hyperparameters
        self.HP_iteration = 100
        self.HP_batch_size = 10
        self.HP_char_hidden_dim = 100
        self.HP_hidden_dim = 200
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True
        self.HP_use_char = False
        self.HP_gpu = False
        self.HP_lr = 0.0035
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

    def build_alphabet(self, input_file):
        in_lines = open(input_file, 'r').readlines()
        for idx in range(len(in_lines)):
            line = in_lines[idx]
            if len(line) > 2:
                pairs = line.strip().split()
                word = pairs[0]
                if self.number_normalized:
                    word = normalize_word(word)
                label = pairs[-1]
                self.label_alphabet.add(label)
                self.word_alphabet.add(word)
                if idx < len(in_lines) - 1 and len(in_lines[idx + 1]) > 2:
                    biword = word + in_lines[idx + 1].strip().split()[0]
                else:
                    biword = word + NULLKEY
                self.biword_alphabet.add(biword)
                for char in word:
                    self.char_alphabet.add(char)
        self.word_alphabet_size = self.word_alphabet.size()
        self.biword_alphabet_size = self.biword_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
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


    def build_gaz_file(self, vocab_file='data/vocab.txt', train_seg='data/train_seg_vocab.txt'):
        """
        we use a vocab.txt file to build gaz file, we do not use embedding file, because it too large.
        """
        # vocab = set()

        with open(vocab_file, 'r') as f:
            for line in f:
                line = line.strip()

                if line:
                    self.gaz.insert(line, 'one_source')


        print("Load gaz file: ", vocab_file, " total size:", self.gaz.size())


    def build_gaz_alphabet(self, input_file):
        """
        based on the train, dev, test file, we only save the seb-sequence word that my be appear
        """
        in_lines = open(input_file, 'r').readlines()
        word_list = []
        for line in in_lines:
            if len(line) > 3:
                word = line.split()[0]
                if self.number_normalized:
                    word = normalize_word(word)
                word_list.append(word)
            else:
                ### a black line
                w_length = len(word_list)
                ## Travser from [0: n], [1: n] to [n-1: n]
                for idx in range(w_length):
                    matched_entity = self.gaz.enumerateMatchList(word_list[idx:])
                    for entity in matched_entity:
                        # print entity, self.gaz.searchId(entity),self.gaz.searchType(entity)
                        self.gaz_alphabet.add(entity)

                word_list = []
        print("gaz alphabet size:", self.gaz_alphabet.size())


    def add_segs(self, file='data/seg_vocab.txt'):
        """
        we add the seg to gaz after generate the gaz_alphabet
        """
        # a 'unknow' word for word not in embeddings
        self.gaz_alphabet.add("unknow")
        self.unknow_index['unknow'] = self.gaz_alphabet.next_index - 1
        for i in range(1, 9):
            self.gaz_alphabet.add("unknow" + str(i))
            self.unknow_index['unknow' + str(i)] = self.gaz_alphabet.next_index - 1

        with open(file, 'r') as f:
            words = f.readlines()

            for word in words:
                word = word.strip()

                if word:
                    self.gaz.insert(word, 'one_source')

        print('total gaz size after add seg: ', self.gaz.size())



    def fix_alphabet(self):
        self.word_alphabet.close()
        self.biword_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close()
        self.gaz_alphabet.close()

    def build_word_pretrain_emb(self, emb_path):
        print("build word pretrain emb...")
        self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(emb_path, self.word_alphabet,
                                                                                   self.word_emb_dim,
                                                                                   self.norm_word_emb)

    def build_biword_pretrain_emb(self, emb_path):
        print("build biword pretrain emb...")
        self.pretrain_biword_embedding, self.biword_emb_dim = build_pretrain_embedding(emb_path, self.biword_alphabet,
                                                                                       self.biword_emb_dim,
                                                                                       self.norm_biword_emb)

    def build_gaz_pretrain_emb(self, emb_path):
        print("build gaz pretrain emb...")
        self.pretrain_gaz_embedding, self.gaz_emb_dim = build_pretrain_embedding_for_gaz(self.gaz_alphabet,
                                                                                         embedding_dir='data/small_embeddings',
                                                                                         embedding_name='embed',
                                                                                         file_num=30,
                                                                                         embedd_dim=self.gaz_emb_dim,
                                                                                         norm=self.norm_gaz_emb)

        # self.pretrain_gaz_embedding, self.gaz_emb_dim = build_pretrain_embedding(emb_path, self.gaz_alphabet,
        #                                                                          self.gaz_emb_dim, self.norm_gaz_emb)



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

    def create_two_more_labels(self):
        """
        as we divided the ner to three sub-processes, so we need create labels
        for the total three precess.
        Step1: is a word is entity, a two class label
        Step2: a word is begin of entity, middle entity, end of entity or other,
               a five class label
        Step3: a NER problem

        return: add two more labels for Step1, and Step2

        for BMES schema
            for Step1:
                NO-ENTITY: 0
                ENTITY:    1

            for Step2:
                NO-ENTITY: 0
                B-ENTITY:  1
                M-ENTITY:  2
                E-ENTITY:  3
                S-ENTITY:  4

        same to BIO schema
            for Step1:
                NO-ENTITY: 0
                NO-ENTITY: 1

            for step2:
                NO-ENTITY: 0
                B-ENTITY: 1
                I-ENTITY: 2

        """

        if self.tagScheme == "BMES":
            ## for train
            train_label_texts = [texts[-1] for texts in self.train_texts]
            train_label_ids = [ids[-1] for ids in self.train_Ids]
            train_size = len(train_label_texts)
            assert train_size == len(train_label_ids)

            self.three_text_id = {}
            self.three_text_id['NO-ENTITY'] = 0
            self.three_labels = ['NO-ENTITY']

            label_length = {"PER": 2, "GPE": 2, "ORG": 5, "LOC": 3}

            id_num = 1
            inner_num = 0

            for i in range(train_size):
                one_label_texts = []
                two_label_texts = []
                three_label_texts = []
                one_label_ids = []
                two_label_ids = []
                three_label_ids = []

                for t_text, t_id in zip(train_label_texts[i], train_label_ids[i]):
                    t_text = t_text.upper()

                    if "PER" in t_text:
                        now_label = "PER"
                    elif "GPE" in t_text:
                        now_label = "GPE"
                    elif "ORG" in t_text:
                        now_label = "ORG"
                    elif "LOC" in t_text:
                        now_label = "LOC"

                    if t_text == 'O':
                        one_label_texts.append('NO-ENTITY')
                        two_label_texts.append('NO-ENTITY')
                        three_label_texts.append('NO-ENTITY')
                        one_label_ids.append(0)
                        two_label_ids.append(0)
                        three_label_ids.append(0)

                        inner_num = 0

                    elif "B-" in t_text:
                        one_label_texts.append("ENTITY")
                        one_label_ids.append(1)

                        two_label_texts.append("B-ENTITY")
                        two_label_ids.append(1)

                        if now_label not in self.three_text_id:
                            self.three_text_id[now_label] = id_num
                            self.three_labels.append(now_label)
                            id_num += 1
                        three_label_texts.append(now_label)
                        three_label_ids.append(self.three_text_id[now_label])

                        inner_num = 0

                    elif "M-" in t_text or "I-" in t_text:
                        one_label_texts.append("ENTITY")
                        one_label_ids.append(1)

                        two_label_texts.append("M-ENTITY")
                        two_label_ids.append(2)

                        if now_label not in self.three_text_id:
                            self.three_text_id[now_label] = id_num
                            self.three_labels.append(now_label)
                            id_num += 1
                        three_label_texts.append(now_label)
                        three_label_ids.append(self.three_text_id[now_label])


                    elif "E-" in t_text:
                        one_label_texts.append("ENTITY")
                        one_label_ids.append(1)

                        two_label_texts.append("E-ENTITY")
                        two_label_ids.append(3)

                        if now_label not in self.three_text_id:
                            self.three_text_id[now_label] = id_num
                            self.three_labels.append(now_label)
                            id_num += 1
                        three_label_texts.append(now_label)
                        three_label_ids.append(self.three_text_id[now_label])

                        inner_num = 0

                    elif "S-" in t_text:
                        one_label_texts.append("ENTITY")
                        one_label_ids.append(1)

                        two_label_texts.append("S-ENTITY")
                        two_label_ids.append(4)

                        if now_label not in self.three_text_id:
                            self.three_text_id[now_label] = id_num
                            self.three_labels.append(now_label)
                            id_num += 1
                        three_label_texts.append(now_label)
                        three_label_ids.append(self.three_text_id[now_label])

                        inner_num = 0

                # add the new label to train_texts and train_Ids
                self.train_texts[i].insert(-1, one_label_texts)
                self.train_texts[i].insert(-1, two_label_texts)
                self.train_texts[i].insert(-1, three_label_texts)
                self.train_Ids[i].insert(-1, one_label_ids)
                self.train_Ids[i].insert(-1, two_label_ids)
                self.train_Ids[i].insert(-1, three_label_ids)
            del train_label_texts
            del train_label_ids

            ## for dev
            dev_label_texts = [texts[-1] for texts in self.dev_texts]
            dev_label_ids = [ids[-1] for ids in self.dev_Ids]
            dev_size = len(dev_label_texts)
            assert dev_size == len(dev_label_ids)

            inner_num = 0

            for i in range(dev_size):
                one_label_texts = []
                two_label_texts = []
                three_label_texts = []
                one_label_ids = []
                two_label_ids = []
                three_label_ids = []
                for d_text, d_id in zip(dev_label_texts[i], dev_label_ids[i]):
                    d_text = d_text.upper()

                    if "PER" in d_text:
                        now_label = "PER"
                    elif "GPE" in d_text:
                        now_label = "GPE"
                    elif "ORG" in d_text:
                        now_label = "ORG"
                    elif "LOC" in d_text:
                        now_label = "LOC"

                    if d_text == 'O':
                        one_label_texts.append('NO-ENTITY')
                        two_label_texts.append('NO-ENTITY')
                        three_label_texts.append('NO-ENTITY')
                        one_label_ids.append(0)
                        two_label_ids.append(0)
                        three_label_ids.append(0)

                        inner_num = 0

                    elif "B-" in d_text:
                        one_label_texts.append("ENTITY")
                        one_label_ids.append(1)

                        two_label_texts.append("B-ENTITY")
                        two_label_ids.append(1)

                        if now_label not in self.three_text_id:
                            self.three_text_id[now_label] = id_num
                            self.three_labels.append(now_label)
                            id_num += 1
                        three_label_texts.append(now_label)
                        three_label_ids.append(self.three_text_id[now_label])

                        inner_num = 0

                    elif "M-" in d_text or "I-" in d_text:
                        one_label_texts.append("ENTITY")
                        one_label_ids.append(1)

                        two_label_texts.append("M-ENTITY")
                        two_label_ids.append(2)

                        if inner_num < label_length[now_label]:
                            inner_num += 1

                        if now_label not in self.three_text_id:
                            self.three_text_id[now_label] = id_num
                            self.three_labels.append(now_label)
                            id_num += 1
                        three_label_texts.append(now_label)
                        three_label_ids.append(self.three_text_id[now_label])

                    elif "E-" in d_text:
                        one_label_texts.append("ENTITY")
                        one_label_ids.append(1)

                        two_label_texts.append("E-ENTITY")
                        two_label_ids.append(3)

                        if now_label not in self.three_text_id:
                            self.three_text_id[now_label] = id_num
                            self.three_labels.append(now_label)
                            id_num += 1
                        three_label_texts.append(now_label)
                        three_label_ids.append(self.three_text_id[now_label])

                        inner_num = 0

                    elif "S-" in d_text:
                        one_label_texts.append("ENTITY")
                        one_label_ids.append(1)

                        two_label_texts.append("S-ENTITY")
                        two_label_ids.append(4)

                        if now_label not in self.three_text_id:
                            self.three_text_id[now_label] = id_num
                            self.three_labels.append(now_label)
                            id_num += 1
                        three_label_texts.append(now_label)
                        three_label_ids.append(self.three_text_id[now_label])

                        inner_num = 0

                # add the new label to train_texts and train_Ids
                self.dev_texts[i].insert(-1, one_label_texts)
                self.dev_texts[i].insert(-1, two_label_texts)
                self.dev_texts[i].insert(-1, three_label_texts)
                self.dev_Ids[i].insert(-1, one_label_ids)
                self.dev_Ids[i].insert(-1, two_label_ids)
                self.dev_Ids[i].insert(-1, three_label_ids)
            del dev_label_texts
            del dev_label_ids

            ## for test
            test_label_texts = [texts[-1] for texts in self.test_texts]
            test_label_ids = [ids[-1] for ids in self.test_Ids]
            test_size = len(test_label_texts)
            assert test_size == len(test_label_ids)

            inner_num = 0

            for i in range(test_size):
                one_label_texts = []
                two_label_texts = []
                three_label_texts = []
                one_label_ids = []
                two_label_ids = []
                three_label_ids = []
                for t_text, t_id in zip(test_label_texts[i], test_label_ids[i]):
                    t_text = t_text.upper()

                    if "PER" in d_text:
                        now_label = "PER"
                    elif "GPE" in d_text:
                        now_label = "GPE"
                    elif "ORG" in d_text:
                        now_label = "ORG"
                    elif "LOC" in d_text:
                        now_label = "LOC"

                    if t_text == 'O':
                        one_label_texts.append('NO-ENTITY')
                        two_label_texts.append('NO-ENTITY')
                        three_label_texts.append('NO-ENTITY')
                        one_label_ids.append(0)
                        two_label_ids.append(0)
                        three_label_ids.append(0)

                        inner_num = 0

                    elif "B-" in t_text:
                        one_label_texts.append("ENTITY")
                        one_label_ids.append(1)

                        two_label_texts.append("B-ENTITY")
                        two_label_ids.append(1)

                        if now_label not in self.three_text_id:
                            self.three_text_id[now_label] = id_num
                            self.three_labels.append(now_label)
                            id_num += 1
                        three_label_texts.append(now_label)
                        three_label_ids.append(self.three_text_id[now_label])

                        inner_num = 0

                    elif "M-" in t_text or "I-" in t_text:
                        one_label_texts.append("ENTITY")
                        one_label_ids.append(1)

                        two_label_texts.append("M-ENTITY")
                        two_label_ids.append(2)

                        if now_label not in self.three_text_id:
                            self.three_text_id[now_label] = id_num
                            self.three_labels.append(now_label)
                            id_num += 1
                        three_label_texts.append(now_label)
                        three_label_ids.append(self.three_text_id[now_label])


                    elif "E-" in t_text:
                        one_label_texts.append("ENTITY")
                        one_label_ids.append(1)

                        two_label_texts.append("E-ENTITY")
                        two_label_ids.append(3)

                        if now_label not in self.three_text_id:
                            self.three_text_id[now_label] = id_num
                            self.three_labels.append(now_label)
                            id_num += 1
                        three_label_texts.append(now_label)
                        three_label_ids.append(self.three_text_id[now_label])

                        inner_num = 0

                    elif "S-" in t_text:
                        one_label_texts.append("ENTITY")
                        one_label_ids.append(1)

                        two_label_texts.append("S-ENTITY")
                        two_label_ids.append(4)

                        if now_label not in self.three_text_id:
                            self.three_text_id[now_label] = id_num
                            self.three_labels.append(now_label)
                            id_num += 1
                        three_label_texts.append(now_label)
                        three_label_ids.append(self.three_text_id[now_label])

                        inner_num = 0

                # add the new label to train_texts and train_Ids
                self.test_texts[i].insert(-1, one_label_texts)
                self.test_texts[i].insert(-1, two_label_texts)
                self.test_texts[i].insert(-1, three_label_texts)
                self.test_Ids[i].insert(-1, one_label_ids)
                self.test_Ids[i].insert(-1, two_label_ids)
                self.test_Ids[i].insert(-1, three_label_ids)
            del test_label_texts
            del test_label_ids

            # for key in self.three_text_id:
            #     self.three_labels.append(key)
            #     # self.three_labels[self.three_text_id[key]] = key

        elif self.tagScheme == "BIO":
            ## for train
            train_label_texts = [texts[-1] for texts in self.train_texts]
            train_label_ids = [ids[-1] for ids in self.train_Ids]
            train_size = len(train_label_texts)
            assert train_size == len(train_label_ids)

            for i in range(train_size):
                one_label_texts = []
                two_label_texts = []
                three_label_texts = []
                one_label_ids = []
                two_label_ids = []
                three_label_ids = []
                for t_text, t_id in zip(train_label_texts[i], train_label_ids[i]):
                    t_text = t_text.upper()
                    if t_text == 'O':
                        one_label_texts.append('NO-ENTITY')
                        two_label_texts.append('NO-ENTITY')
                        three_label_texts.append('NO-ENTITY')
                        one_label_ids.append(0)
                        two_label_ids.append(0)
                        three_label_ids.append(0)
                    elif "B-" in t_text:
                        one_label_texts.append("ENTITY")
                        one_label_ids.append(1)

                        two_label_texts.append("B-ENTITY")
                        two_label_ids.append(1)

                        if t_text.split('-')[1] not in self.three_text_id:
                            self.three_text_id[t_text.split('-')[1]] = id_num
                            id_num += 1
                        three_label_texts.append(t_text.split('-')[1])
                        three_label_ids.append(self.three_text_id[t_text.split('-')[1]])

                    elif "I-" in t_text:
                        one_label_texts.append("ENTITY")
                        one_label_ids.append(1)

                        two_label_texts.append("I-ENTITY")
                        two_label_ids.append(2)

                        if t_text.split('-')[1] not in self.three_text_id:
                            self.three_text_id[t_text.split('-')[1]] = id_num
                            id_num += 1
                        three_label_texts.append(t_text.split('-')[1])
                        three_label_ids.append(self.three_text_id[t_text.split('-')[1]])

                # add the new label to train_texts and train_Ids
                self.train_texts[i].insert(-1, one_label_texts)
                self.train_texts[i].insert(-1, two_label_texts)
                self.train_texts[i].insert(-1, three_label_texts)
                self.train_Ids[i].insert(-1, one_label_ids)
                self.train_Ids[i].insert(-1, two_label_ids)
                self.train_Ids[i].insert(-1, three_label_ids)
            del train_label_texts
            del train_label_ids

            ## for dev
            dev_label_texts = [texts[-1] for texts in self.dev_texts]
            dev_label_ids = [ids[-1] for ids in self.dev_Ids]
            dev_size = len(dev_label_texts)
            assert dev_size == len(dev_label_ids)

            for i in range(dev_size):
                one_label_texts = []
                two_label_texts = []
                three_label_texts = []
                one_label_ids = []
                two_label_ids = []
                three_label_ids = []
                for d_text, d_id in zip(dev_label_texts[i], dev_label_ids[i]):
                    d_text = d_text.upper()
                    if d_text == 'O':
                        one_label_texts.append('NO-ENTITY')
                        two_label_texts.append('NO-ENTITY')
                        three_label_texts.append('NO-ENTITY')
                        one_label_ids.append(0)
                        two_label_ids.append(0)
                        three_label_ids.append(0)
                    elif "B-" in d_text:
                        one_label_texts.append("ENTITY")
                        one_label_ids.append(1)

                        two_label_texts.append("B-ENTITY")
                        two_label_ids.append(1)

                        three_label_texts.append(d_text.split('-')[1])
                        three_label_ids.append(self.three_text_id[d_text.split('-')[1]])

                    elif "I-" in d_text:
                        one_label_texts.append("ENTITY")
                        one_label_ids.append(1)

                        two_label_texts.append("I-ENTITY")
                        two_label_ids.append(2)

                        three_label_texts.append(d_text.split('-')[1])
                        three_label_ids.append(self.three_text_id[d_text.split('-')[1]])

                # add the new label to train_texts and train_Ids
                self.dev_texts[i].insert(-1, one_label_texts)
                self.dev_texts[i].insert(-1, two_label_texts)
                self.dev_texts[i].insert(-1, three_label_texts)
                self.dev_Ids[i].insert(-1, one_label_ids)
                self.dev_Ids[i].insert(-1, two_label_ids)
                self.dev_Ids[i].insert(-1, three_label_ids)
            del dev_label_texts
            del dev_label_ids

            ## for test
            test_label_texts = [texts[-1] for texts in self.test_texts]
            test_label_ids = [ids[-1] for ids in self.test_Ids]
            test_size = len(test_label_texts)
            assert test_size == len(test_label_ids)

            for i in range(test_size):
                one_label_texts = []
                two_label_texts = []
                three_label_texts = []
                one_label_ids = []
                two_label_ids = []
                three_label_ids = []
                for t_text, t_id in zip(test_label_texts[i], test_label_ids[i]):
                    t_text = t_text.upper()
                    if t_text == 'O':
                        one_label_texts.append('NO-ENTITY')
                        two_label_texts.append('NO-ENTITY')
                        three_label_texts.append('NO-ENTITY')
                        one_label_ids.append(0)
                        two_label_ids.append(0)
                        three_label_ids.append(0)
                    elif "B-" in t_text:
                        one_label_texts.append("ENTITY")
                        one_label_ids.append(1)

                        two_label_texts.append("B-ENTITY")
                        two_label_ids.append(1)

                        three_label_texts.append(t_text.split('-')[1])
                        three_label_ids.append(self.three_text_id[t_text.split('-')[1]])

                    elif "I-" in t_text:
                        one_label_texts.append("ENTITY")
                        one_label_ids.append(1)

                        two_label_texts.append("I-ENTITY")
                        two_label_ids.append(2)

                        three_label_texts.append(t_text.split('-')[1])
                        three_label_ids.append(self.three_text_id[t_text.split('-')[1]])

                # add the new label to train_texts and train_Ids
                self.test_texts[i].insert(-1, one_label_texts)
                self.test_texts[i].insert(-1, two_label_texts)
                self.test_texts[i].insert(-1, three_label_texts)
                self.test_Ids[i].insert(-1, one_label_ids)
                self.test_Ids[i].insert(-1, two_label_ids)
                self.test_Ids[i].insert(-1, three_label_ids)
            del test_label_texts
            del test_label_ids

            for key in self.three_text_id:
                self.three_labels.append(key)





