from keras.layers.recurrent import *
from keras.layers import *
from keras.layers.core import *
from keras.models import *
import numpy as np
from keras.preprocessing.sequence import *

import sys

sys.path.append('../../')

from LSTM.src.WordEmbeddingLayer import *


ACTION = {0:'NONE', 1:'NEG', 2:'POS'}
class DeepQNetwork(object):
    def __init__(self,input_dim,output_dim,hidden_dim,exploration_probability):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.exploration_probability = exploration_probability
        self.QTable = {}
        self.DiscountFactor = 0.8
        self.max_len = 100

    def build_model(self):


        self.QModel = Sequential()
        self.QModel.add(LSTM(output_dim=self.hidden_dim, input_length=self.max_len, input_dim=self.input_dim, return_sequences=True))
        self.QModel.add(LSTM(output_dim=self.output_dim, input_length=self.max_len, input_dim=self.hidden_dim, return_sequences=True))
        #self.QModel.add(Activation('sigmoid'))

        self.QModel.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])





    def get_Qvalue(self,stateId,action):
        if (stateId,action) in self.QTable.keys():
            return self.QTable[(stateId,action)]
        else:
            self.QTable[(stateId,action)] = np.random.rand()

    def select_action(self,state):
        rand = np.random.rand(100)
        if rand < self.exploration_probability:
            return self.explor()
        else:
            return np.argmax(self.QModel.predict(state))



    def learning_loop(self,embedded_sentence,sentiment):

        #vocab_representation = WordEmbeddingLayer()
        #vocab_representation.load_filtered_embedding("../data/filtered_glove.840B.300d")

        #embedded_sentence = np.asarray(vocab_representation.embed(sentence)[0])

        padded_embedded_sentence = np.zeros((self.max_len,self.input_dim))

        for i in np.arange(len(embedded_sentence)):
            padded_embedded_sentence[i,:] = embedded_sentence[i]

        estimated_qz = self.QModel.predict(np.asarray([padded_embedded_sentence]))[0]

        estimated_qz = estimated_qz - 0.5
        #print("sentence length: "+str(len(embedded_sentence)))
        #print("predicted sentiment is: "+ACTION[np.argmax(estimated_qz[len(embedded_sentence) - 1])]+" "+str(np.argmax(estimated_qz[len(embedded_sentence) - 1]) == target))
        rewards = np.zeros_like(estimated_qz)
        rewards[len(embedded_sentence) - 1,:] = -100
        rewards[len(embedded_sentence) - 1][sentiment] = 100
        updating_qvals = np.copy(estimated_qz)
        u_rewards = []
        updating_qvals[len(embedded_sentence) - 1][sentiment] = rewards[len(embedded_sentence) - 1][sentiment]
        best_actions = []
        i = len(embedded_sentence) - 2
        while i >= 0:
            best_next_actionn = np.argmax(rewards[i+1])
            for k in ACTION.keys():
                rewards[i][k] = estimated_qz[i][k] + self.DiscountFactor * rewards[i+1][best_next_actionn]

            best_action = np.argmax(rewards[i])
            updating_qvals[i][best_action] = rewards[i][best_action]
            best_actions.append(best_action)
            u_rewards.append(rewards[i][best_action])
            i -= 1


        padded_targets = np.zeros((self.max_len, self.output_dim))
        for i in np.arange(len(embedded_sentence)):
            padded_targets[i, :] = updating_qvals[i]
        self.QModel.train_on_batch(np.asarray([padded_embedded_sentence]),np.asarray([padded_targets]))

import random


if __name__ == '__main__':
    dqn = DeepQNetwork(input_dim=300,output_dim=3,hidden_dim=300,exploration_probability=0.3)
    dqn.build_model()



    embedded_train, train_labels = WordEmbeddingLayer.load_embedded_data(path="../data/", name="train",
                                                                         representation="glove.840B.300d")
    c = list(zip(embedded_train,train_labels))
    random.shuffle(c)
    embedded_train, train_labels = zip(*c)
    embedded_train, train_labels = embedded_train, train_labels

    binary_embedded_train = []
    binary_train_labels = []
    for i in np.arange(len(embedded_train)):
        if np.argmax(train_labels[i]) != 1:
            binary_embedded_train.append(embedded_train[i])
            m = (np.argmax(train_labels[i]) // 2) + 1
            binary_train_labels.append(m)

    embedded_test, test_labels = WordEmbeddingLayer.load_embedded_data(path="../data/", name="test",
                                                                         representation="glove.840B.300d")
    embedded_test, test_labels = embedded_test, test_labels

    binary_embedded_test = []
    binary_test_labels = []
    for i in np.arange(len(embedded_test)):
        if np.argmax(train_labels[i]) != 1:
            binary_embedded_test.append(embedded_test[i])
            m = (np.argmax(test_labels[i]) // 2) + 1
            binary_test_labels.append(m)


    padded_embedded_test = []

    for i in np.arange(len(binary_embedded_test)):
        padded_embedded_sentence = np.zeros((dqn.max_len, dqn.input_dim))
        padded_label = np.zeros((dqn.max_len, dqn.output_dim))
        for j in np.arange(len(binary_embedded_test[i])):
            padded_embedded_sentence[j, :] = binary_embedded_test[i][j]


        padded_embedded_test.append(padded_embedded_sentence)

    for i in np.arange(1000):

        for embededed_sentence,target in zip(embedded_train,binary_train_labels):
            dqn.learning_loop(embededed_sentence,target)


        targets = dqn.QModel.predict(np.asarray(padded_embedded_test),batch_size=1)
        targets = np.argmax(targets[:,-1,:],axis=1)

        accuracy = sum(x == 0 for x in (targets - binary_train_labels)) / len(targets)
        print(accuracy)








