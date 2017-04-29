# -*- coding: UTF-8 -*-
"""
该模块对quora 重复文档进行word2vector 预训练
"""
from gensim.models import word2vec
from os.path import join, exists, split
import os


def train_word2vec(ques_pairs, num_features=100, min_word_count=1, context=5):
    """ 对ques_pairs 进行word2vec预训练
    """
    model_dir = '../data/word2vec_models'
    model_name = "{:d}features_{:d}minwords_{:d}context".format(
        num_features, min_word_count, context)
    model_name = join(model_dir, model_name)
    embedding_model = None
    if exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
        print 'Loading existing Word2Vec mode \'%s\'' % split(model_name)[-1]
    else:
        sentences = []
        for pair in ques_pairs:
            sentences.append(pair[1])
            sentences.append(pair[2])

        embedding_model = word2vec.Word2Vec(sentences,
                                            size=num_features,
                                            min_count=min_word_count,
                                            workers=2,
                                            window=context)

        embedding_model.init_sims(replace=True)

        if not exists(model_dir):
            os.mkdir(model_dir)
        print 'Saving Word2Vec mode \'%s\'' % split(model_name)[-1]
        embedding_model.save(model_name)

    return embedding_model
