######################################################################################
# A script to test a recurrent neural network (LSTM) on a dataset of sentences.The
# LSTM should learn how to interprete the next word(s) from a sequence of input words
######################################################################################

# Imports
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time

# Global variables
trainData = 'test_input_data.txt'
modelDir = 'models/'
numInputs = 4
numHidden = 128
numRnnCells = 3
keepProb = 1.0

# Function to read the text data from a file
def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [content[i].split() for i in range(len(content))]
    content = np.array(content)
    content = np.reshape(content, [-1, ])
    return content

# Function to build the dataset of words
def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

# Function to create an RNN cell with dropout
def createRNNCell():
    rnnCell = rnn.BasicLSTMCell(numHidden)   
    rnnDropout = rnn.DropoutWrapper(rnnCell,input_keep_prob=keepProb, output_keep_prob=keepProb)
    return rnnDropout

# Function to define the RNN structure
def RNN(x, weights, biases):
    x = tf.reshape(x, [-1, numInputs])
    x = tf.split(x,numInputs,1)

    rnnNet = rnn.MultiRNNCell([createRNNCell() for _ in range(numRnnCells)])

    outputs, states = rnn.static_rnn(rnnNet, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']



# Load up the training data
print("Loading training data...")
training_data = read_data(trainData)
print("Loaded training data")
dictionary, reverse_dictionary = build_dataset(training_data)
vocab_size = len(dictionary)

# TF placeholders
x = tf.placeholder("float", [None, numInputs, 1])
y = tf.placeholder("float", [None, vocab_size])

# RNN output node weights and biases
weights = {'out': tf.Variable(tf.random_normal([numHidden, vocab_size]))}
biases = {'out': tf.Variable(tf.random_normal([vocab_size]))}

# Create the RNN here
pred = RNN(x, weights, biases)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Create the saver
saver = tf.train.Saver(tf.trainable_variables())

# Launch the graph
with tf.Session() as session:

    # Load the variables from a checkpoint file
    saver.restore(session, modelDir + 'lstm.ckpt')
    print("Model loaded from file")

    while True:
        prompt = "%s words: " % numInputs
        sentence = input(prompt)
        sentence = sentence.strip()
        words = sentence.split(' ')
        if len(words) != numInputs:
            continue
        try:
            symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
            for i in range(32):
                keys = np.reshape(np.array(symbols_in_keys), [-1, numInputs, 1])
                onehot_pred = session.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sentence = "%s %s" % (sentence,reverse_dictionary[onehot_pred_index])
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(onehot_pred_index)
            print(sentence)
        except:
            print("Word not in dictionary")