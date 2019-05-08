# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:11:08
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-07-06 11:08:27

import time
import sys
import argparse
import random
import copy
import torch
import gc
import pickle
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
import re
import os

from ner.utils.metric import get_ner_fmeasure
# from ner.model.bilstmcrf import BiLSTM_CRF as SeqModel
# from ner.lw.cw_ner import CW_NER as SeqModel
from ner.lw.big_cw_ner import Big_CW_NER as SeqModel
from ner.utils.big_data import Data
from tensorboardX import SummaryWriter
from ner.lw.tbx_writer import TensorboardWriter
from ner.functions.save_res import *


seed_num = 100
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

# set logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
BASIC_FORMAT = "%(asctime)s:%(levelname)s: %(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler() # 输出到控制台的handler
chlr.setFormatter(formatter)
logger.addHandler(chlr)


# for checkpoint
max_model_num = 50
old_model_paths = []

# tensorboard writer
log_dir = "data/log"

train_log = SummaryWriter(os.path.join(log_dir, "train"))
validation_log = SummaryWriter(os.path.join(log_dir, "validation"))
test_log = SummaryWriter(os.path.join(log_dir, "test"))
tensorboard = TensorboardWriter(train_log, validation_log, test_log)


def data_initialization(data, gaz_file, train_file, dev_file, test_file):
    data.build_alphabet(train_file)
    data.build_alphabet(dev_file)
    # data.build_alphabet(test_file)
    data.build_gaz_file()
    data.build_gaz_alphabet(train_file)
    data.build_gaz_alphabet(dev_file)
    # data.build_gaz_alphabet(test_file)
    data.fix_alphabet()
    return data


def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """

    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        # print "p:",pred, pred_tag.tolist()
        # print "g:", gold, gold_tag.tolist()
        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1 - decay_rate) ** epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def save_model(epoch, state, models_dir):
    """
    save the model state
    Args:
        epoch: the number of epoch
        state: [model_state, training_state]
        models_dir: the dir to save model
    """
    if models_dir is not None:
        model_path = os.path.join(models_dir, "model_state_epoch_{}.th".format(epoch))
        train_path = os.path.join(models_dir, "training_state_epoch_{}.th".format(epoch))

        model_state, training_state = state
        torch.save(model_state, model_path)
        torch.save(training_state, train_path)

        if max_model_num > 0:
            old_model_paths.append([model_path, train_path])
            if len(old_model_paths) > max_model_num:
                paths_to_remove = old_model_paths.pop(0)

                for fname in paths_to_remove:
                    os.remove(fname)

def find_last_model(models_dir):
    """
    find the lastes checkpoint file
    Args:
         models_dir: the dir save models
    """
    epoch_num = 30
    return  models_dir + "/model_state_epoch_{}.th".format(epoch_num), models_dir + "/training_state_epoch_{}.th".format(epoch_num)

    if models_dir is None:
        return None

    saved_models_path = os.listdir(models_dir)
    saved_models_path = [x for x in saved_models_path if 'model_state_epoch' in x]

    if len(saved_models_path) == 0:
        return None

    found_epochs = [
        re.search("model_state_epoch_([0-9]+).th", x).group(1)
        for x in saved_models_path
    ]
    int_epochs = [int(epoch) for epoch in found_epochs]
    print("len: ", len(int_epochs))
    last_epoch = sorted(int_epochs, reverse=True)[0]
    epoch_to_load = "{}".format(last_epoch)

    model_path = os.path.join(models_dir, "model_state_epoch_{}.th".format(epoch_to_load))
    training_state_path = os.path.join(models_dir, "training_state_epoch_{}.th".format(epoch_to_load))

    return model_path, training_state_path


def restore_model(models_dir):
    """
    restore the lastes checkpoint file
    """
    lastest_checkpoint = find_last_model(models_dir)

    if lastest_checkpoint is None:
        return None
    else:
        model_path, training_state_path = lastest_checkpoint

        model_state = torch.load(model_path)
        training_state = torch.load(training_state_path)

        return (model_state, training_state)


def evaluate(data, model, name):
    if name == "train":
        instances = data.train_Ids
        texts = data.train_texts
    elif name == "dev":
        instances = data.dev_Ids
        texts = data.dev_texts
    elif name == 'test':
        instances = data.test_Ids
        texts = data.test_texts
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print("Error: wrong evaluate name,", name)
    right_token = 0
    whole_token = 0
    pred_results = []
    gold_results = []
    ## set model in eval model
    model.eval()
    batch_size = 4
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num // batch_size + 1
    for batch_id in range(total_batch):
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        gaz_list, reverse_gaz_list, batch_word, batch_biword, batch_wordlen, batch_wordrecover, batch_label, mask = batchify_with_label(instance, data.HP_gpu, True)
        tag_seq = model(gaz_list, reverse_gaz_list, batch_word, batch_wordlen, mask)
        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances) / decode_time

    if type == '-':
        print('overall result of: {}'.format(name))
    elif type == ".NAM":
        print('NE result of: {}'.format(name))
    elif type == ".NOM":
        print('NM result of: {}'.format(name))
    
    save_gold_pred(texts, pred_results, gold_results, name)
    
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    print("time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (decode_time, speed, acc, p, r, f))
    return speed, acc, p, r, f, pred_results


def batchify_with_label(input_batch_list, gpu, volatile_flag=False):
    """
        input: list of words, chars and labels, various length. [[words,biwords,chars,gaz, labels],[words,biwords,chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    biwords = [sent[1] for sent in input_batch_list]
    gazs = [sent[2] for sent in input_batch_list]
    reverse_gazs = [sent[3] for sent in input_batch_list]
    labels = [sent[4] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max()
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    biword_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len))).byte()
    for idx, (seq, biseq, label, seqlen) in enumerate(zip(words, biwords, labels, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        biword_seq_tensor[idx, :seqlen] = torch.LongTensor(biseq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1] * int(seqlen))
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    biword_seq_tensor = biword_seq_tensor[word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]

    _, word_seq_recover = word_perm_idx.sort(0, descending=False)

    gaz_list = [gazs[i] for i in word_perm_idx]
    reverse_gaz_list = [reverse_gazs[i] for i in word_perm_idx]

    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        biword_seq_tensor = biword_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        mask = mask.cuda()

        # gaz_seq_tensor = gaz_seq_tensor.cuda()
        # gaz_seq_length = gaz_seq_length.cuda()
        # gaz_mask_tensor = gaz_mask_tensor.cuda()

    return gaz_list, reverse_gaz_list, word_seq_tensor, biword_seq_tensor, word_seq_lengths, word_seq_recover, label_seq_tensor, mask


def get_data_model(data, save_model_dir, seg=True):
    print("restore the model from checkpoint...")

    data.show_data_summary()
    model = SeqModel(data, type=5)
    print("finished built model.")

    loss_function = nn.NLLLoss()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.SGD(parameters, lr=data.HP_lr, momentum=data.HP_momentum)
    best_dev = -1
    data.HP_iteration = 100

    ## here we should restore the model
    state = restore_model(save_model_dir)
    epoch = 0
    if state is not None:
        model_state = state[0]
        training_state = state[1]

        model.load_state_dict(model_state)
        optimizer.load_state_dict(training_state['optimizer'])
        epoch = int(training_state['epoch'])

    batch_size = 8  ## current only support batch size = 1 to compulate and accumulate to data.HP_batch_size update weights

    return data, model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with bi-directional LSTM-CRF')
    parser.add_argument('--embedding', help='Embedding for words', default='None')
    parser.add_argument('--status', choices=['train', 'test', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--savemodel', default="data/model/msra")
    parser.add_argument('--savedset', help='Dir of saved data setting', default="data/save.dset")
    # parser.add_argument('--train', default="data/note4/train.char.bmes")
    # parser.add_argument('--dev', default="data/note4/dev.char.bmes")
    # parser.add_argument('--test', default="data/note4/test.char.bmes")
    # parser.add_argument('--train', default="data/resume/train.char.bmes")
    # parser.add_argument('--dev', default="data/resume/dev.char.bmes")
    # parser.add_argument('--test', default="data/resume/test.char.bmes")
    # parser.add_argument('--train', default="data/weibo/train.char.bmes")
    # parser.add_argument('--dev', default="data/weibo/dev.char.bmes")
    # parser.add_argument('--test', default="data/weibo/test.char.bmes")
    # parser.add_argument('--train', default="data/demo.train.char")
    # parser.add_argument('--dev', default="data/demo.dev.char")
    # parser.add_argument('--test', default="data/demo.test.char")
    parser.add_argument('--train', default="data/msra/train.char.bmes")
    parser.add_argument('--dev', default="data/msra/test.char.bmes")
    parser.add_argument('--test', default="data/weibo/test.char.bmes")
    parser.add_argument('--seg', default="True")
    parser.add_argument('--extendalphabet', default="True")
    parser.add_argument('--raw')
    parser.add_argument('--loadmodel')
    parser.add_argument('--output')
    args = parser.parse_args()

    train_file = args.train
    dev_file = args.dev
    test_file = args.test
    raw_file = args.raw
    model_dir = args.loadmodel
    dset_dir = args.savedset
    output_file = args.output
    if args.seg.lower() == "true":
        seg = True
    else:
        seg = False
    status = args.status.lower()

    save_model_dir = args.savemodel
    gpu = torch.cuda.is_available()

    char_emb = "data/gigaword_chn.all.a2b.uni.ite50.vec"
    bichar_emb = None
    gaz_file = "data/ctb.50d.vec"
    # gaz_file = None
    # char_emb = None
    # bichar_emb = None

    print("CuDNN:", torch.backends.cudnn.enabled)
    # gpu = False
    print("GPU available:", gpu)
    print("Status:", status)
    print("Seg: ", seg)
    print("Train file:", train_file)
    print("Dev file:", dev_file)
    print("Test file:", test_file)
    print("Raw file:", raw_file)
    print("Char emb:", char_emb)
    print("Bichar emb:", bichar_emb)
    print("Gaz file:", gaz_file)
    if status == 'train':
        print("Model saved to:", save_model_dir)
    sys.stdout.flush()

    if status == 'train':
        data = Data()
        data.HP_gpu = gpu
        data.HP_use_char = False
        data.HP_batch_size = 1
        data.use_bigram = False
        data.gaz_dropout = 0.5
        data.norm_gaz_emb = False
        data.HP_fix_gaz_emb = False
        data_initialization(data, gaz_file, train_file, dev_file, test_file)
        data.generate_instance_with_gaz_no_char(train_file, 'train')
        data.generate_instance_with_gaz_no_char(dev_file, 'dev')
        # data.generate_instance_with_gaz_no_char(test_file, 'test')
        data.build_word_pretrain_emb(char_emb)
        # data.build_biword_pretrain_emb(bichar_emb)
        data.build_gaz_pretrain_emb(gaz_file)


        # begin evaludate
        # first get model, second do evaluate_save
        data, model = get_data_model(data, save_model_dir, seg)

        # do evaludate train
        # evaluate(data, model, "train")


        # do evaludate dev
        evaluate(data, model, "dev")

        # do evaluate test
        # evaluate(data, model, "test")



