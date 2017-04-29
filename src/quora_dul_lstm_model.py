# -*- coding: UTF-8 -*-
"""
该模块对quora 重复问题进行识别，采用lstm模型；
"""

# from gensim.models import Word2Vec
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Merge, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.model_selection import KFold
import numpy as np
import os

import data_process
import word2vec_pretrain

EMBED_DIM = 64
HIDDEN_DIM = 100
BATCH_SIZE = 32
NBR_EPOCHS = 1

MODEL_DIR = "../data/"


def model(data_file):
    """ 使用lstm模型，判断一对问题是否是重复的
    """

    # 数据预处理，十折交叉法
    print "data pre-processing"
    ques_pairs = data_process.parse_quoar_dul_data(data_file)[0:500]
    word2idx = data_process.build_vocab(ques_pairs)
    vocab_size = len(word2idx) + 1
    seq_maxlen = data_process.get_seq_maxlen(ques_pairs)
    pids, x_ques1, x_ques2, y = data_process.vectorize_ques_pair(
        ques_pairs, word2idx, seq_maxlen)

    # 计算embeding 初始weight；
    w2v_embedding_model = word2vec_pretrain.train_word2vec(
        ques_pairs, num_features=EMBED_DIM, min_word_count=1, context=5)
    embedding_weights = np.zeros((vocab_size, EMBED_DIM))
    for word, index in word2idx.iteritems():
        if word in w2v_embedding_model:
            embedding_weights[index, :] = w2v_embedding_model[word]
        else:
            embedding_weights[index, :] = np.random.uniform(
                -0.25, 0.25, w2v_embedding_model.vector_size)
    # 建立模型；
    print("Building model...")
    ques1_enc = Sequential()
    ques1_enc.add(
        Embedding(
            output_dim=EMBED_DIM,
            input_dim=vocab_size,
            weights=[embedding_weights],
            mask_zero=True))
    ques1_enc.add(
        LSTM(
            HIDDEN_DIM,
            input_shape=(EMBED_DIM, seq_maxlen),
            return_sequences=False))
    ques1_enc.add(Dropout(0.3))

    ques2_enc = Sequential()
    ques2_enc.add(
        Embedding(
            output_dim=EMBED_DIM,
            input_dim=vocab_size,
            weights=[embedding_weights],
            mask_zero=True))
    ques2_enc.add(
        LSTM(
            HIDDEN_DIM,
            input_shape=(EMBED_DIM, seq_maxlen),
            return_sequences=False))
    ques2_enc.add(Dropout(0.3))

    model = Sequential()
    model.add(Merge([ques1_enc, ques2_enc], mode="sum"))
    model.add(Dense(2, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"])

    kf = KFold(n_splits=10)
    i = 0
    sum_acc = 0.0
    for train_index, test_index in kf.split(x_ques1):
        i += 1
        print "TRAIN-TEST: %d" % i
        x_ques1train, x_ques1test = x_ques1[train_index], x_ques1[test_index]
        x_ques2train, x_ques2test = x_ques2[train_index], x_ques2[test_index]
        ytrain, ytest = y[train_index], y[test_index]
        pidstrain, pidstest = pids[train_index], pids[test_index]

        print(x_ques1train.shape, x_ques1test.shape, x_ques2train.shape,
              x_ques2test.shape, ytrain.shape, ytest.shape, pidstrain.shape,
              pidstest.shape)

        print("Training...")
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "quora_dul_best_lstm.hdf5"),
            verbose=1,
            save_best_only=True)
        model.fit(
            [x_ques1train, x_ques2train],
            ytrain,
            batch_size=BATCH_SIZE,
            epochs=NBR_EPOCHS,
            validation_split=0.1,
            verbose=2,
            callbacks=[checkpoint])

        # predict
        print("predict...")
        y_test_pred = model.predict_classes(
            [x_ques1test, x_ques2test], batch_size=BATCH_SIZE)
        data_process.pred_save("../data/y_test_{:d}.pred".format(i),
                               y_test_pred, ytest, pidstest)

        print("Evaluation...")
        loss, acc = model.evaluate(
            [x_ques1test, x_ques2test], ytest, batch_size=BATCH_SIZE)
        print("Test loss/accuracy final model = %.4f, %.4f" % (loss, acc))

        model.save_weights(
            os.path.join(MODEL_DIR, "quora_dul_lstm-final.hdf5"))
        with open(os.path.join(MODEL_DIR, "quora_dul_lstm.json"),
                  "wb") as fjson:
            fjson.write(model.to_json())

        model.load_weights(filepath=os.path.join(MODEL_DIR,
                                                 "quora_dul_best_lstm.hdf5"))
        loss, acc = model.evaluate(
            [x_ques1test, x_ques2test], ytest, batch_size=BATCH_SIZE)
        print("Test loss/accuracy best model = %.4f, %.4f" % (loss, acc))
        sum_acc += acc
    print "After all the result acc:", sum_acc / 10


if __name__ == '__main__':
    model("../data/quora_duplicate_questions.tsv")
