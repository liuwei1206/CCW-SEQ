# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:23:06
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-05-12 22:09:37
import sys
import numpy as np
from ner.utils.alphabet import Alphabet
NULLKEY = "-null-"
def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_instance_char(input_file, char_alphabet, bichar_alphabet, label_alphabet, number_normalized, max_sent_length):
    """
    read instance with, character, bichar, label, no gaz
    Args:
        input_file: the input file path
        char_alphabet: character
        bichar_alphabet: bichar
        label_alphabet: labels
        number_normalized:
        max_sent_length:
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        in_lines = f.readlines()
        instence_texts = []
        instence_Ids = []
        chars = []
        bichars = []
        labels = []
        char_Ids = []
        bichar_Ids = []
        label_Ids = []
        for idx in range(len(in_lines)):
            line = in_lines[idx]
            if len(line) > 2:
                pairs = line.strip().split()
                char = pairs[0]
                if number_normalized:
                    char = normalize_word(char)
                label = pairs[-1]

                if idx < len(in_lines) - 1 and len(in_lines[idx+1]) > 2:
                    bichar = char + in_lines[idx+1].strip().split()[0]
                else:
                    bichar = char + NULLKEY

                chars.append(char)
                bichars.append(bichar)
                labels.append(label)
                char_Ids.append(char_alphabet.get_index(char))
                bichar_Ids.append(bichar_alphabet.get_index(bichar))
                label_Ids.append(label_alphabet.get_index(label))

            else:
                # black line
                if (max_sent_length < 0) or (len(chars) < max_sent_length):
                    instence_texts.append([chars, bichars, labels])
                    instence_Ids.append([char_Ids, bichar_Ids, label_Ids])

                chars = []
                bichars = []
                labels = []
                char_Ids = []
                bichar_Ids = []
                label_Ids = []
        return instence_texts, instence_Ids

def read_instance_word(input_file, word_alphabet, label_alphabet, number_normalized, max_sent_length):
    """
    read instance with: word, label, no char, no gaz
    Args:
        input_file:
        word_alphabet:
        label_alphabet:
        number_normalized:
        max_sent_length:
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        in_lines = f.readlines()
        instence_texts = []
        instence_Ids = []
        words = []
        labels = []
        word_Ids = []
        label_Ids = []
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split()
                word = pairs[0]
                if number_normalized:
                    word = normalize_word(word)
                label = pairs[-1]

                word_Id = word_alphabet.get_index(word)
                label_Id = label_alphabet.get_index(label)

                words.append(word)
                labels.append(label)
                word_Ids.append(word_Id)
                label_Ids.append(label_Id)

            else:
                if max_sent_length < 0 or (len(words) < max_sent_length):
                    instence_texts.append([words, labels])
                    instence_Ids.append([word_Ids, label_Ids])

                words = []
                labels = []
                word_Ids = []
                label_Ids = []

        return instence_texts, instence_Ids

def read_char_file(input_file):
    """
    read char file, to get string of character and label
    :param input_file:
    :return:
    """
    chars = []
    labels = []
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        now_chars = []
        now_labels = []
        for line in lines:
            if len(line.strip()) < 1:
                chars.append(now_chars)
                labels.append(now_labels)

                now_chars = []
                now_labels = []
            else:
                items = line.strip().split()

                now_chars.append(items[0])
                now_labels.append(items[1])

    return [chars, labels]

def read_instance_with_gaz_no_char(input_file, gaz, word_alphabet, biword_alphabet, gaz_alphabet, label_alphabet, number_normalized, max_sent_length, unknow_index=None):
    """
    read instance with, word, biword, gaz, lable, no char
    Args:
        input_file: the input file path
        gaz: the gaz obj
        word_alphabet: word
        biword_alphabet: biword
        gaz_alphabet: gaz
        label_alphabet: label
        number_normalized: true or false
        max_sent_length: the max length
    """
    in_lines = open(input_file, 'r', encoding='utf-8').readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    biwords = []
    labels = []
    word_Ids = []
    biword_Ids = []
    label_Ids = []
    for idx in range(len(in_lines)):
        line = in_lines[idx]
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0]
            if number_normalized:
                word = normalize_word(word)
            label = pairs[-1]
            if idx < len(in_lines) - 1 and len(in_lines[idx+1]) > 2:
                biword = word + in_lines[idx+1].strip().split()[0]
            else:
                biword = word + NULLKEY
            biwords.append(biword)
            words.append(word)
            labels.append(label)
            word_Ids.append(word_alphabet.get_index(word))
            biword_Ids.append(biword_alphabet.get_index(biword))
            label_Ids.append(label_alphabet.get_index(label))
        else:
            if ((max_sent_length < 0) or (len(words) < max_sent_length)) and (len(words) > 0):
                gazs = []
                gaz_Ids = []
                gazs_length = []
                w_length = len(words)

                reverse_gazs = [[] for i in range(w_length)]
                reverse_gaz_Ids = [[] for i in range(w_length)]

                # assign sub-sequence to every chinese letter
                flag = [0 for f in range(w_length)]
                pre_flag = 0
                for i in range(w_length):
                    matched_list = gaz.enumerateMatchList(words[i: ])

                    if len(matched_list) > 0:
                        f_len = len(matched_list[0])
                        if (flag[i] == 1 or len(matched_list) > 1) and len(matched_list[-1]) == 1:
                            matched_list = matched_list[:-1]
                        for f_pos in range(pre_flag, i+f_len):
                            flag[f_pos] = 1
                        pre_flag = i + f_len

                    matched_length = [len(a) for a in matched_list]
                    gazs.append(matched_list)
                    matched_Id = []
                    for entity in matched_list:
                        index_id = gaz_alphabet.get_index(entity)
                        if index_id > 1:
                            matched_Id.append(index_id)
                        else:
                            w_name = "unknow" + str(len(entity))
                            if w_name in unknow_index:
                                matched_Id.append(unknow_index[w_name])
                            else:
                                matched_Id.append(unknow_index['unknow'])

                    if matched_Id:
                        # gaz_Ids.append([matched_Id, matched_length])
                        gaz_Ids.append(matched_Id)
                        gazs_length.append(matched_length)
                    else:
                        gaz_Ids.append([])
                        gazs_length.append([])

                # for i in range(w_length-1, -1, -1):
                for i in range(w_length):
                    now_pos_gaz = gazs[i]
                    now_pos_gaz_Id = gaz_Ids[i]
                    now_pos_gaz_len = gazs_length[i]

                    ## Traversing it
                    l = len(now_pos_gaz)
                    assert len(now_pos_gaz) == len(now_pos_gaz_Id)
                    for j in range(l):
                        width = now_pos_gaz_len[j]
                        end_char_pos = i + width - 1

                        reverse_gazs[end_char_pos].append(now_pos_gaz[j])
                        reverse_gaz_Ids[end_char_pos].append(now_pos_gaz_Id[j])


                instence_texts.append([words, biwords, gazs, reverse_gazs, labels])
                instence_Ids.append([word_Ids, biword_Ids, gaz_Ids, reverse_gaz_Ids, label_Ids])
            words = []
            biwords = []
            labels = []
            word_Ids = []
            biword_Ids = []
            label_Ids = []
            gazs = []
            reverse_gazs = []
            gaz_Ids = []
            reverse_gaz_Ids = []
    return instence_texts, instence_Ids


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):    
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0

    ## we should also init the index 0
    pretrain_emb[0, :] = np.random.uniform(-scale, scale, [1, embedd_dim])

    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/word_alphabet.size()))
    return pretrain_emb, embedd_dim

def build_pretrain_embedding_for_gaz(word_alphabet, embedding_dir="data/small_embeddings", embedding_name="embed", file_num=100, embedd_dim=200, norm=False):
    embedd_dict = dict()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    pretrained_size = 0

    ## we should also init the index 0
    pretrain_emb[0, :] = np.random.uniform(-scale, scale, [1, embedd_dim])

    ## flag can prevent repeated initialization
    flag = [1 for i in range(word_alphabet.size())]
    flag[0] = 0

    for pos in range(1, file_num+1):
        embed_file = embedding_dir + "/" + embedding_name + str(pos)

        if embed_file != None:
            embedd_dict, embedd_dim = load_pretrain_emb(embed_file)

        for word, index in word_alphabet.iteritems():
            if word in embedd_dict and flag[index]:
                if norm:
                    pretrain_emb[index, :] = norm2one(embedd_dict[word])
                else:
                    pretrain_emb[index, :] = embedd_dict[word]
                perfect_match += 1
                case_match += 1

                flag[index] = 0

        pretrained_size += len(embedd_dict)
        embedd_dict = None
        embed_file = None

    for word, index in word_alphabet.iteritems():
        if flag[index]:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1

    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/word_alphabet.size()))
    return pretrain_emb, embedd_dim

       
def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square

def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                if (embedd_dim + 1) != len(tokens):
                    print(embedd_dim, len(tokens))
                    continue
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict, embedd_dim

if __name__ == '__main__':
    a = np.arange(9.0)
    print(a)
    print(norm2one(a))
