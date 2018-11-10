from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pyspark import SparkContext, SparkConf
import gzip
import os
import sys
import tensorflow.python.platform
import numpy as np
from six.moves import urllib
from six.moves import xrange  
import tensorflow as tf
import itertools

def map_fun(i):
    with tf.Graph().as_default() as g:
        hello = tf.constant('Hello, TensorFlow!', name="hello_constant")
    with tf.Session() as sess:
          return sess.run(hello)

conf = (SparkConf().setMaster("local").setAppName("CNN").set("spark.storage.memory", "1g"))

spc = SparkContext(conf = conf)
rdd = spc.parallelize(range(10))
rdd.map(map_fun).collect()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def maybe_download(filename):
    if not os.path.exists(WORK_DIRECTORY):
        os.mkdir(WORK_DIRECTORY)
        filepath = os.path.join(WORK_DIRECTORY, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath

def extract_data(filename, num_images):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    return data


def extract_labels(filename, num_images):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
      # Convert to dense 1-hot representation.
    return (np.arange(NUM_LABELS) == labels[:, None]).astype(np.float32)


def fake_data(num_images):
    data = np.ndarray(
          shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
          dtype=np.float32)
    labels = np.zeros(shape=(num_images, NUM_LABELS), dtype=np.float32)
    for image in xrange(num_images):
        label = image % 2
        data[image, :, :, 0] = label - 0.5
        labels[image, label] = 1.0
    return data, labels


def error_rate(predictions, labels):
    return 100.0 - (
          100.0 *
          np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
          predictions.shape[0])

train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

train_data = extract_data(train_data_filename, 42000)
train_labels = extract_labels(train_labels_filename, 42000)
test_data = extract_data(test_data_filename, 28000)
test_labels = extract_labels(test_labels_filename, 28000)

# Generate a validation set.
validation_data = train_data[:VALIDATION_SIZE, :, :, :]
validation_labels = train_labels[:VALIDATION_SIZE]
train_data = train_data[VALIDATION_SIZE:, :, :, :]
train_labels = train_labels[VALIDATION_SIZE:]
num_epochs = NUM_EPOCHS
train_size = train_labels.shape[0]

#Distributed implementation
class ConvNet(object): pass

def create_graph(base_learning_rate = 0.01, decay_rate = 0.95, conv1_size=32, conv2_size=64, fc1_size=512):
        # training step using the {feed_dict} argument to the Run() call below.
        train_data_node = tf.placeholder(
          tf.float32,
          shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        train_labels_node = tf.placeholder(tf.float32,
                                         shape=(BATCH_SIZE, NUM_LABELS))
        validation_data_node = tf.constant(validation_data)
        test_data_node = tf.constant(test_data)

		# {tf.initialize_all_variables().run()}
        conv1_weights = tf.Variable(
          tf.truncated_normal([5, 5, NUM_CHANNELS, conv1_size],  # 5x5 filter, depth 32.
                              stddev=0.1,
                              seed=SEED))
        conv1_biases = tf.Variable(tf.zeros([conv1_size]))
        
        conv2_weights = tf.Variable(
          tf.truncated_normal([5, 5, conv1_size, conv2_size],
                              stddev=0.1,
                              seed=SEED))
        conv2_biases = tf.Variable(tf.constant(0.1, shape=[conv2_size]))
        
        fc1_weights = tf.Variable(  # fully connected, depth 512.
            tf.truncated_normal(
              [IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * conv2_size, fc1_size], stddev=0.1, seed=SEED))
        fc1_biases = tf.Variable(tf.constant(0.1, shape=[fc1_size]))
        
        fc2_weights = tf.Variable(
          tf.truncated_normal([fc1_size, NUM_LABELS], stddev=0.1, seed=SEED))
        fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

def model(data, train=False):
        conv = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv = tf.nn.conv2d(pool,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(
            pool,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        
		hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        
		if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
            return tf.matmul(hidden, fc2_weights) + fc2_biases

        
		logits = model(train_data_node, True)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
          logits, train_labels_node))

        
		regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                      tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
        
		loss += 5e-4 * regularizers

		batch = tf.Variable(0)
        
		learning_rate = tf.train.exponential_decay(
          base_learning_rate,                # Base learning rate.
          batch * BATCH_SIZE,  # Current index into the dataset.
          train_size,          # Decay step.
          decay_rate,                # Decay rate.
          staircase=True)
        
		optimizer = tf.train.MomentumOptimizer(learning_rate,
                                             0.9).minimize(loss,
                                                           global_step=batch)


        
		train_prediction = tf.nn.softmax(logits)
        
		validation_prediction = tf.nn.softmax(model(validation_data_node))
        test_prediction = tf.nn.softmax(model(test_data_node))

        res = ConvNet()
        res.train_prediction = train_prediction
        res.optimizer = optimizer
        res.loss = loss
        res.learning_rate = learning_rate
        res.validation_prediction = validation_prediction
        res.test_prediction = test_prediction
        res.train_data_node = train_data_node
        res.train_labels_node = train_labels_node
        return res

train_data_bc = spc.broadcast(train_data)
train_labels_bc = spc.broadcast(train_labels)

def run(base_learning_rate, decay_rate, fc1_size):
    train_data = train_data_bc.value
    train_labels = train_labels_bc.value
    res = {}
    res['base_learning_rate'] = base_learning_rate
    res['decay_rate'] = decay_rate
    res['fc1_size'] = fc1_size
    res['minibatch_loss'] = 100.0
    res['test_error'] = 100.0
    res['validation_error'] = 100.0
    # Training may fail to converge, or even diverge; guard against that.
    try:
        with tf.Session() as s:
          
		  graph = create_graph(base_learning_rate, decay_rate, fc1_size=fc1_size)
          
		  tf.initialize_all_variables().run()
          
		  for step in xrange(num_epochs * train_size // BATCH_SIZE):
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), :, :, :]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            feed_dict = {graph.train_data_node: batch_data,
                         graph.train_labels_node: batch_labels}
            # Run the graph and fetch some of the nodes.
            _, l, lr, predictions = s.run(
                [graph.optimizer, graph.loss, graph.learning_rate, graph.train_prediction],
                feed_dict=feed_dict)
            res['minibatch_loss'] = l
            res['test_error'] = error_rate(graph.test_prediction.eval(), test_labels)
            res['validation_error'] = error_rate(graph.validation_prediction.eval(), validation_labels)
        return res
    except Exception as e:
        pass
    return res

base_learning_rates = [float(x) for x in np.logspace(-3, -1, num=10, base=10.0)]
decay_rates = [0.95]
fc1_sizes = [64, 128, 256, 512, 1024]
all_experiments = list(itertools.product(base_learning_rates, decay_rates, fc1_sizes))
print(len(all_experiments))

len(all_experiments)

num_nodes = 4
n = max(2, int(len(all_experiments) // num_nodes))
grouped_experiments = [all_experiments[i:i+n] for i in range(0, len(all_experiments), n)]
all_exps_rdd = spc.parallelize(grouped_experiments, numSlices=len(grouped_experiments))
results = all_exps_rdd.flatMap(lambda z: [run(*y) for y in z]).collect()

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(sum(map(ord, "aesthetics")))
import pandas as pd

df = pd.DataFrame(results)
df

