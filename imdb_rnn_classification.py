#!/usr/bin/env python
# coding: utf-8

# Import TensorFlow and TensorFlow Eager
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb

tfe.enable_eager_execution()

# 25000 条训练数据和 25000 条测试数据
vocab_size = 10000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

# Dict，键为词，值为词对应的标号
word_index = imdb.get_word_index()
# Dict，键为序号，值为词. 注意0是一个特殊的键，代表Unknown word
index_word = dict([(value, key) for (key, value) in word_index.items()])
# print("LEN:", len(set(index_word.keys())))
# print(index_word[int(2)])

# We decode the review; note that our indices were offset by 3
# because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
decoded_review = ' '.join([index_word.get(i - 3, '?') for i in train_data[10]])
# print(train_data[10])
# print(decoded_review)

# 全部句子截断或延长为80个词长的句子
from keras.preprocessing import sequence
maxlen = 80
train_data = sequence.pad_sequences(train_data, maxlen=maxlen)
test_data = sequence.pad_sequences(test_data, maxlen=maxlen)
assert train_data.shape == test_data.shape == (25000, 80)


class RNNModel(tf.keras.Model):
    def __init__(self, 
                 vocabulary_size=None,        # 词典大小
                 embedding_size=500,          # the size of the word embedding
                 cell_size=500,                # RNN cell size
                 dense_size1=800,              # the size of the dense layer
                 dense_size2=100,              # the size of the dense layer
                 num_classes=2):              # 2-分类
        """ 
            embedding_size: the size of the word embedding.
            cell_size: RNN cell size.
            dense_size: the size of the dense layer.
            num_classes: the number of labels in the network.
            vocabulary_size: the size of the word vocabulary.
            rnn_cell: string, either 'lstm' or 'ugrnn'.
            device: string, 'cpu:n' or 'gpu:n' (n can vary). Default, 'cpu:0'.
        """
        super(RNNModel, self).__init__()
        
        self.cell_size = cell_size
        self.dense_size2 = dense_size2
        w_initializer = tf.contrib.layers.xavier_initializer()
        b_initializer = tf.zeros_initializer()
        
        # Initialize weights for word embeddings 
        self.embeddings = tf.keras.layers.Embedding(vocabulary_size, embedding_size, 
                    embeddings_initializer=w_initializer)
        
        # Dense layer initialization
        self.dense_layer1 = tf.keras.layers.Dense(dense_size1, activation=tf.nn.relu, 
                    kernel_initializer=w_initializer)
        self.dense_layer2 = tf.keras.layers.Dense(dense_size2, activation=tf.nn.relu, 
                    kernel_initializer=w_initializer)

        # Predictions layer initialization
        self.pred_layer = tf.keras.layers.Dense(num_classes, activation=None, 
                    kernel_initializer=w_initializer)
        
        # LSTM cell
        self.rnn_cell_1 = tf.nn.rnn_cell.LSTMCell(cell_size)
        self.rnn_cell_2 = tf.nn.rnn_cell.LSTMCell(cell_size)
        
    def predict(self, X, seq_length, is_training):
        """ X: 2D tensor of shape (batch_size, time_steps).
            seq_length: the length of each sequence in the batch.
            is_training: Boolean. Either the network is predicting in training mode or not.
        """
        # Get the number of samples within a batch
        num_samples = tf.shape(X)[0]
        
        # Get the embedding of each word in the sequence
        X = self.embeddings(X)
        
        # Unstack the embeddings
        X = tf.unstack(X, axis=1)
        
        # RNN layer 1
        state1 = self.rnn_cell_1.zero_state(num_samples, dtype=tf.float32)
        outputs1 = []
        for input_step in X:
            output1, state1 = self.rnn_cell_1(input_step, state1)
            outputs1.append(output1)
        outputs1 = tf.layers.dropout(outputs1, rate=0.3, training=is_training)

        # RNN layer 2
        state2 = self.rnn_cell_2.zero_state(num_samples, dtype=tf.float32)
        for input_step in outputs1:
            output, state2 = self.rnn_cell_2(input_step, state2)

        # Add dropout for regularization
        dropped_output = tf.layers.dropout(output, rate=0.3, training=is_training)

        # dense1
        dense1 = self.dense_layer1(dropped_output)
        # 
        dense1_drop = tf.layers.dropout(dense1, rate=0.3, training=is_training)
        # 
        dense2 = self.dense_layer2(dense1_drop)
        assert dense2.shape == (num_samples.numpy(), self.dense_size2)

        # Compute the unnormalized log probabilities
        logits = self.pred_layer(dense2)
        return logits
    
    def loss_fn(self, X, y, seq_length, is_training):
        """ Defines the loss function used during 
            training.         
        """
        preds = self.predict(X, seq_length, is_training)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=preds)
        return loss
    
    def grads_fn(self, X, y, seq_length, is_training):
        with tfe.GradientTape() as tape:
            loss = self.loss_fn(X, y, seq_length, is_training)
        return loss, tape.gradient(loss, self.variables)

    def acc_fn(self, X, y, seq_length, is_training):
        preds = self.predict(X, seq_length, is_training).numpy()
        acc = np.sum(np.argmax(preds, axis=1) == y.numpy(), dtype=np.float32) / X.numpy().shape[0]
        return acc
        

def train(model, dataset, test_data, test_labels, 
          checkpoint, checkpoint_prefix, optimizer, epoches=10):
    best_loss = 1e5
    for epoch in range(epoches):
        # losses = []
        # for (batch, (inp, targ)) in enumerate(dataset):
        #     loss, gradients = model.grads_fn(inp, targ, seq_length=80, is_training=True)
        #     gradients, _ = tf.clip_by_global_norm(gradients, 0.25)    # clip梯度
        #     optimizer.apply_gradients(zip(gradients, model.variables))
        #     print(loss.numpy(), end=', ', flush=True)
        #     losses.append(loss.numpy())
        # print("\nEpoch=", epoch, "\n训练集平均损失:", np.mean(losses))
        # eval
        test_data, test_labels = tf.convert_to_tensor(test_data), tf.convert_to_tensor(test_labels)
        eval_loss = model.loss_fn(test_data, test_labels, seq_length=80, is_training=False)
        eval_acc = model.acc_fn(test_data, test_labels, seq_length=80, is_training=False)
        print("验证集损失:", eval_loss.numpy())
        print("验证集准确率:", eval_acc*100, "%")

        # save
        if best_loss > eval_loss:
            best_loss = eval_loss
            checkpoint.save(file_prefix=checkpoint_prefix)
        print("------\n\n")

# model
learning_rate = tf.Variable(1e-5, name="learning_rate")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
model = RNNModel(vocabulary_size=vocab_size)

# checkpoint
checkpoint_dir = 'models_checkpoints/ImdbRNN/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, 
                learning_rate=learning_rate, model=model)

# data
dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).shuffle(25000)
dataset = dataset.batch(250, drop_remainder=True)

# 训练
train(model, dataset, test_data[:500], test_labels[:500], checkpoint, 
     checkpoint_prefix, optimizer, epoches=30)


# Visualizing RNN cell activations
# The part of this tutorial has been inspired by the work of 
# Karpathy in "Visualizing and understanding recurrent neural networks" ([link here](https://arxiv.org/abs/1506.02078)). 
# We will use the library seaborn for plotting the heatmap. You can get 


# import seaborn as sns

# checkpoint.restore(os.path.join(checkpoint_dir, "ckpt-10"))

# def VisualizeRNN(model, X):
#     ''' Function to return the tanh of the cell state at each timestep.

#         Args:
#             model: trained RNN model.
#             X: indexed review of shape (1, sequence_length).
            
#         Returns:
#             tanh(cell_states): the tanh of the memory cell at each timestep.       
#     '''
    
#     # Initialize LSTM cell state with zeros
#     state = model.rnn_cell_1.zero_state(1, dtype=tf.float32)
    
#     # Get the embedding of each word in the sequence
#     embedded_words = model.embeddings(X)

#     # Unstack the embeddings
#     unstacked_embeddings = tf.unstack(embedded_words, axis=1)

#     # Iterate through each timestep and append its cell state
#     cell_states = []
#     for input_step in unstacked_embeddings:
#         _, state = model.rnn_cell_1(input_step, state)
#         cell_states.append(state[0])
        
#     # Stack cell_states to (batch_size, time_steps, cell_size)
#     cell_states = tf.stack(cell_states, axis=1)
#     return tf.tanh(cell_states)


# def test()

# dummy_review = "very recommended to everyone very recommended to everyone very recommended to everyone very recommended to everyone very good good good"
# y = 1

# print("\n评论:\n", dummy_review)
# print("极性:\n", y, "\n\n")

# indexed_review = np.array([[(word_index.get(w, -1) + 3) for w in dummy_review.split()]])
# X = sequence.pad_sequences(indexed_review, maxlen=maxlen)
# assert X.shape == (1, 80)

# # predict
# predict = model.predict(tf.convert_to_tensor(X), seq_length=80, is_training=False)[0]
# print("预测结果: \n", predict)
# result = ["Bad", "Good"]
# print("预测极性: \n", result[np.argmax(predict)])

# # Get the cell states
# cell_states = VisualizeRNN(model, tf.convert_to_tensor(X))
# cell_states = cell_states.numpy()[:, -1*indexed_review[0].shape[0]:, :]
# print(cell_states.shape)

# # Plot activations of the first 10 units in the cell (the cell has 64 units in total)
# plt.figure(figsize = (16,8))
# sns.heatmap(cell_states[0,:,:10].T, 
#             xticklabels=dummy_review.split(),
#             cmap='RdBu')
# plt.savefig("rnn-heatmap.jpg")
