__author__ = "liuwei"

"""
metric for auto-word and gold-word
"""

def get_auto_word_ner_measure(golden_lists, word_tags, predict_lists):
    """
    Args:
        golden_lists: a tuple, (chars, char_labels)
        word_tags: assigned ner tag by convert from char golden label
        predict_lists: a tuple, (words, predict_labels)
    """
    chars, char_labels = golden_lists
    words, pred_labels = predict_lists

    sent_num = len(chars)
    print(sent_num, len(words))
    assert sent_num == len(words)
    print(sent_num, len(pred_labels))
    assert sent_num == len(pred_labels)

    # for p, r, f1
    golden_full = []
    predict_full = []
    right_full = []
    # for acc
    right_tag = 0
    all_tag = 0

    for idx in range(0, sent_num):
        char = chars[idx]
        char_label = char_labels[idx]
        word = words[idx][0]
        word_pred = pred_labels[idx]
        word_tag = word_tags[idx]

        ## get accuracy, maybe not precious, but just very similar
        for idy in range(len(word_tag)):
            if word_pred[idy] == word_tag[idy]:
                right_tag += 1
        all_tag += len(word_tag)

        ## for p, r, f; we need to get the right predict, full predict, total_gold
        gold_matrix = get_ner_auto_word(char, char_label)
        pred_matrix = get_ner_auto_word(word, word_pred)

        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner

    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)

    # calculate p, r, f
    if predict_num == 0:
        precision = -1
    else:
        precision = (right_num + 0.0) / predict_num

    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num + 0.0) / golden_num

    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)

    accuracy = (right_tag + 0.0) / all_tag
    print("gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num)
    return accuracy, precision, recall, f_measure


def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string


def get_ner_auto_word(items, item_labels):
    """
    items and item_labels maybe: 1. words and word_labels; 2. characters and character_labels
    Args:
        items:
        item_labels:
    """
    list_len = len(items)
    assert list_len == len(item_labels)

    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'

    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []

    now_len = 0
    for i in range(0, list_len):
        current_label = item_labels[i].upper()

        ## a ner only can be add into list when 'S-' or 'E-', if 'B-' then start
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(now_len-1))

            whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(now_len)
            index_tag = current_label.replace(begin_label, "", 1)

            now_len += len(items[i])

        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(now_len-1))

            whole_tag = current_label.replace(single_label, "", 1) + '[' + str(now_len)
            now_len += len(items[i])
            tag_list.append(whole_tag + ',' + str(now_len-1))

            whole_tag = ""
            index_tag = ""

        elif end_label in current_label:
            now_len += len(items[i])
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(now_len-1))

            whole_tag = ""
            index_tag = ""
        else:
            now_len += len(items[i])
            continue

    if (whole_tag != '') and (index_tag != ''):
        tag_list.append(whole_tag + ',' + str(now_len - 1))

    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)

    return stand_matrix



def get_gold_word_ner_measure(golden_lists, predict_lists):
    """
    train, dev, and test all have gold label, and a label just cover a single word,
    so we don't need to take B, M, E, S into account
    Args:
        golden_lists:
        predict_lists:
    """
    sent_num = len(golden_lists)
    assert sent_num == len(predict_lists)

    # for accuracy
    right_tag = 0
    pred_tag = 0
    all_tag = 0

    for idx in range(0, sent_num):
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]

        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1

        all_tag += len(golden_list)
        pred_tag += len(predict_list)

    # p, r, f
    if pred_tag == 0:
        precision = -1
    else:
        precision = (right_tag + 0.0) / pred_tag

    if all_tag == 0:
        recall = -1
    else:
        recall = (right_tag + 0.0) / all_tag

    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)

    accuracy = (right_tag + 0.0) / all_tag

    print("gold_num = ", all_tag, " pred_num = ", pred_tag, " right_num = ", right_tag)
    return accuracy, precision, recall, f_measure



