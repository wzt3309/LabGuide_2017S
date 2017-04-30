# -*- coding: UTF-8 -*-
"""
对quora 数据进行处理；
"""

import re
import codecs
import collections
import numpy as np
from keras.preprocessing.sequence import pad_sequences


def tokenize(sent):
    """ 切句子
    """
    return [x.strip().lower() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_quoar_dul_data(data_file):
    """解析quoar文件的内容
    """
    ques_pairs = []
    with codecs.open(data_file, 'rb') as fi:
        fi.readline()
        for line in fi:
            if len(line) == 0:
                continue
            splits = line.split('\t')
            if len(splits) < 6:
                continue
            pid = splits[0]
            ques1 = splits[3]
            ques2 = splits[4]
            is_dul = int(splits[5])

            ques1_token = tokenize(ques1)
            ques2_token = tokenize(ques2)

            ques_pairs.append((pid, ques1_token, ques2_token, is_dul))
    return ques_pairs


def build_vocab(ques_pairs):
    """ 建立token->id 索引
    """
    wordcounts = collections.Counter()
    for pair in ques_pairs:
        for w in pair[1]:
            wordcounts[w] += 1
        for w in pair[2]:
            wordcounts[w] += 1
    # 将两个ques合并，形成一个没有重复单词的单词列表
    words = [wordcount[0] for wordcount in wordcounts.most_common()]
    # 将每个单词都标上序号，从1开始
    word2idx = {w: i + 1 for i, w in enumerate(words)}

    return word2idx


def get_seq_maxlen(ques_pairs):
    """计算最大句子长度
    """
    max_ques1_len = max([len(pair[1]) for pair in ques_pairs])
    max_ques2_len = max([len(pair[2]) for pair in ques_pairs])
    max_len = max(max_ques1_len, max_ques2_len)

    return max_len


def vectorize_ques_pair(ques_pairs, word2idx, seq_maxlen):
    """ 对question pair进行id向量化
    """
    pids = []
    x_ques1 = []
    x_ques2 = []
    y = []
    for pair in ques_pairs:
        pids.append(pair[0])
        # 每个问题的单词由单词表中的序号代替——句子转化成数组
        # 所有问题1形成一个矩阵
        x_ques1.append([word2idx[w] for w in pair[1]])
        # 所有问题2形成一个矩阵
        x_ques2.append([word2idx[w] for w in pair[2]])
        # 将每对问题的结果转换成numpy中的array，加快运算
        y.append((np.array([0, 1]) if pair[3] == 1 else np.array([1, 0])))

    pids = np.array(pids)
    # 由于每个问题长度不同，为了形成每行都等长的矩阵，需要补0操作
    x_ques1 = pad_sequences(x_ques1, maxlen=seq_maxlen)
    x_ques2 = pad_sequences(x_ques2, maxlen=seq_maxlen)
    # 形成每对问题是否相同的，结果矩阵
    y = np.array(y)

    return pids, x_ques1, x_ques2, y


def pred_save(out_file, y_preds, y_trues, ids):
    """ 输出预测结果
    """
    with open(out_file, 'w') as fo:
        for i in range(len(y_preds)):
            pred = y_preds[i]
            truelabel = y_trues[i][1]
            pid = ids[i]
            fo.write("%s, %s, %s\n" % (str(pid), str(pred), str(truelabel)))
