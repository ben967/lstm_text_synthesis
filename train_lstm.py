######################################################################################
# A script to train a recurrent neural network (LSTM) on a dataset of sentences.The
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
trainData = 'input_data.txt'
modelDir = 'models/'
learningRate = 0.001
numIterations = 50000
displayStep = 1000
numInputs = 3
numHidden = 512

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

# Define the RNN structure
def RNN(x, weights, biases):
    # reshape to [1, numInputs]
    x = tf.reshape(x, [-1, numInputs])
    # Generate a numInputs-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,numInputs,1)
    # 2-layer LSTM, each layer has numHidden units.
    # Average Accuracy= 95.20% at 50k iter
    
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(numHidden),rnn.BasicLSTMCell(numHidden)])
    # 1-layer LSTM with numHidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.BasicLSTMCell(numHidden)
    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    # there are numInputs outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']
pred = RNN(x, weights, biases)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learningRate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Create the saver
saver = tf.train.Saver(tf.trainable_variables())

# Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0
    offset = random.randint(0,numInputs+1)
    end_offset = numInputs + 1
    acc_total = 0
    loss_total = 0
    while step < numIterations:
        
        # Generate a minibatch. Add some randomness on selection process.
        if offset > (len(training_data)-end_offset):
            offset = random.randint(0, numInputs+1)
        symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset+numInputs) ]
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, numInputs, 1])
        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
        symbols_out_onehot[dictionary[str(training_data[offset+numInputs])]] = 1.0
        symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])
        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
        loss_total += loss
        acc_total += acc
        if (step+1) % displayStep == 0:
            print("Iter= " + str(step+1) + ", Average Loss= " + \
                  "{:.6f}".format(loss_total/displayStep) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100*acc_total/displayStep))
            acc_total = 0
            loss_total = 0
            symbols_in = [training_data[i] for i in range(offset, offset + numInputs)]
            symbols_out = training_data[offset + numInputs]
            symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
            print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
        step += 1
        offset += (numInputs+1)

    # Save the variables to disk.
    savePath = saver.save(session, modelDir + 'lstm.ckpt')
    print("Model saved in file: %s" % savePath)