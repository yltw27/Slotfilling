"""
Ref:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py
https://liusida.github.io/2016/11/16/study-lstm/
https://github.com/HadoopIt/rnn-nlu

y1: intent
y2: label/entity

"""
import tensorflow as tf
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from tensorflow.contrib.rnn import static_bidirectional_rnn
from datetime import datetime
import os

start = datetime.now()
ckpt_folder = start.strftime("%Y-%m-%d")

try:
    os.mkdir('../ckpt/'+ckpt_folder)
except FileExistsError:
    pass

tf.reset_default_graph()

# Training Parameters
learning_rate_intent = 0.00075  # for intent classification
learning_rate_label = 0.0025  # for label classification
# learning_rate = 0.00075
# loss_magnification = 3

training_steps = 100
batch_size_train = 50
# dropout_keep_prob = 1.0

# Network Parameters
num_input = 50
num_hidden = 128
num_classes_intent = 22
num_classes_label = 122

# data processing
# df = pd.read_table('../data/ATIS/atis-2.dev.w-intent.iob', names=['utterance', 'label'])
df_train = pd.read_table('../data/ATIS/atis.train.w-intent.iob', names=['utterance', 'label'])
df_test = pd.read_table('../data/ATIS/atis.test.w-intent.iob', names=['utterance', 'label'])

intent_dict = pd.read_csv('../data/ATIS/intent_dict.csv', header=0)
label_dict = pd.read_csv('../data/ATIS/label_dict.csv', header=0)

#model = models.KeyedVectors.load_word2vec_format('../word2vec/en.model.bin', binary=True)  # old
word_vectors = KeyedVectors.load_word2vec_format("../word2vec/en.model.bin", binary=True)
# print('word2vec model loaded')


# data processing: x
def seq2vec(seq):
    seq_list = seq.split(' ')[1:-1]  # remove BOS, EOS
    sentence = []
    for i in range(len(seq_list)):
        # TODO: find a better way to solve OOV
        try:
            sentence.append(word_vectors.get_vector(seq_list[i]))
        except KeyError:
            sentence.append(np.random.randint(low=3, size=300))
    length = len(sentence)
    if length < num_input:
        for x in range(num_input - length):
            sentence.append(np.zeros((300,)))
    return np.vstack(sentence)  # return a numpy array, shape=(num_input, 300)


def generate_batch(df, batch_size=batch_size_train, train=False):
    num_samples = df.shape[0]
    batch_num = int(np.ceil(num_samples / batch_size))
    label = df['label'].tolist()

    # data processing: y2 (label/entity)
    y2_temp = np.zeros((num_samples, num_input, num_classes_label))
    for i in range(len(label)):
        temp = label[i].split(' ')[:-1]
        if temp[0] == '':
            temp = temp[1:]
        if train:
            temp_idx = [label_dict.loc[label_dict['label'] == x]['idx'].values[0]-1 for x in temp]
        else:
            temp_idx = [label_dict.loc[label_dict['label'] == x]['idx'].values - 1 for x in temp]
        if len(temp_idx) < num_input:
            for a in range(num_input-len(temp_idx)):
                temp_idx.append(0)
        for j in range(len(temp_idx)):
            y2_temp[i][j][temp_idx[j]] = 1
    y2 = y2_temp

    batch_y2 = []
    head = 0
    tail = head + batch_size
    for i in range(batch_num):
        batch_y2.append(y2[head:tail])
        head += batch_size
        tail += batch_size
    print(batch_y2[0].shape)

    # data processing: y1 (intent)
    y1 = []
    for i in range(len(label)):
        temp = label[i].split(' ')[-1]
        if train:
            y1.append(intent_dict.loc[intent_dict['intent'] == temp]['idx'].values[0]-1)
        else:
            y1.append(intent_dict.loc[intent_dict['intent'] == temp]['idx'].values - 1)

    def one_hot(num, classes):
        a = np.zeros(classes)
        a[num-1] = 1
        return a

    batch_y1_500 = []
    for i in range(len(y2)):
        batch_y1_500.append(one_hot(y1[i], num_classes_intent))
    batch_y1_500 = np.array(batch_y1_500)  # (500, 16)
    batch_y1 = []
    head = 0
    tail = head + batch_size
    for i in range(batch_num):
        batch_y1.append(batch_y1_500[head:tail])
        head += batch_size
        tail += batch_size
    batch_y1 = np.array(batch_y1)
    print(batch_y1[0].shape)

    ut2vec = []
    ut = df['utterance'].tolist()
    for i in range(len(ut)):
        vec = seq2vec(ut[i])
        ut2vec.append(vec)
    ut2vec = np.array(ut2vec)  # (500, 50, 300)
    batch_x = []
    head = 0
    tail = head + batch_size
    for i in range(batch_num):
        batch_x.append(ut2vec[head:tail])
        head += batch_size
        tail += batch_size
    batch_x = np.array(batch_x)  # (16, 50, 300) * 32

    if train:
        print('Training data prepared')
    else:
        print('Testing data prepared')

    return num_samples, batch_num, batch_x, batch_y1, batch_y2

# num_samples, batch_num, batch_x, batch_y1, batch_y2 = generate_batch(df)
num_samples_train, batch_num_train, batch_x_train, batch_y1_train, batch_y2_train = generate_batch(df_train, train=True)
num_samples_test, batch_num_test, batch_x_test, batch_y1_test, batch_y2_test = generate_batch(df_test, batch_size=893)
dp = datetime.now()

# tf Graph input
x = tf.placeholder("float", [None, num_input, 300], name='x')
y1 = tf.placeholder("float", [None, num_classes_intent], name='y1')  # y1: intent
y2 = tf.placeholder("float", [None, num_input, num_classes_label], name='y2')  # y2: label/entity


# Define weights
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.046875)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    # initial = tf.constant(0.0, shape=shape)
    initial = tf.truncated_normal(shape, stddev=0.046875)
    return tf.Variable(initial, name=name)


def encoder_birnn(x):
    x = tf.unstack(x, num_input, 1)
    # Forward direction cell
    cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # Backward direction cell
    cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0)

    encoder_outputs, encoder_state_fw, encoder_state_bw = static_bidirectional_rnn(cell_fw, cell_bw, x,
                                                                                   dtype=tf.float32)
    # here we simply use the state from last layer as the encoder state
    state_fw = encoder_state_fw[-1]
    state_bw = encoder_state_bw[-1]
    encoder_state = tf.concat([tf.concat(state_fw, 1),
                               tf.concat(state_bw, 1)], 1)
    top_states = [tf.reshape(e, [-1, 1, 2*num_hidden]) for e in encoder_outputs]
    attention_states = tf.concat(top_states, 1)
    return encoder_outputs, encoder_state, attention_states

encoder_outputs, encoder_state, attention_states = encoder_birnn(x)


def decoder_intent(x):
    w = weight_variable(shape=[2 * num_hidden, num_classes_intent], name='w_intent')
    b = bias_variable(shape=[num_classes_intent], name='b_intent')
    result = tf.nn.relu(tf.matmul(x, w) + b)
    return result

logits = decoder_intent(encoder_state)

# Define loss and optimizer
loss_op1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                         logits=logits, labels=y1))
optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate_intent,
                                    beta1=0.9,
                                    beta2=0.999,
                                    epsilon=1e-4,)
train_op1 = optimizer1.minimize(loss_op1)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred1 = tf.equal(tf.argmax(logits, 1, name='intent_pred'), tf.argmax(y1, 1))
accuracy1 = tf.reduce_mean(tf.cast(correct_pred1, tf.float32))


def decoder_labels(encoder_outputs, encoder_state, attention_states):
    cell = tf.nn.rnn_cell.LSTMCell(num_hidden*2, forget_bias=1.0)

    # TODO: attention states

    outputs, state = tf.nn.static_rnn(cell, encoder_outputs, initial_state=None, dtype=tf.float32)
    # output: (?, 256) * 50

    w = weight_variable(shape=[2 * num_hidden, num_classes_label], name='w_label')
    b = bias_variable(shape=[num_classes_label], name='b_label')

    # Transform the list into a 3D tensor with dimensions (n_timesteps, batch_size, N_HIDDEN)
    seq_outputs = tf.stack(outputs)

    def pred_fn(current_output):
        return tf.matmul(current_output, w) + b

    # Use tf.map_fn to apply pred_fn to each tensor in outputs, along dimension 0 (timestep dimension)
    pred = tf.map_fn(pred_fn, seq_outputs)
    result = tf.transpose(pred, [1, 0, 2])

    return result

batch_seq_cls = decoder_labels(encoder_outputs, encoder_state, attention_states)

loss_op2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=batch_seq_cls, labels=y2))
# joint_loss = tf.add(tf.multiply(loss_op1, loss_magnification), loss_op2)

# Define loss and optimizer
optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate_label,
                                    beta1=0.9,
                                    beta2=0.999,
                                    epsilon=1e-4,)
train_op2 = optimizer2.minimize(loss_op2)

# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
#                                    beta1=0.9,
#                                    beta2=0.999,
#                                    epsilon=1e-4,)
# train_op = optimizer.minimize(joint_loss)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred2 = tf.equal(tf.argmax(batch_seq_cls, 2, name='seq_pred'), tf.argmax(y2, 2))
accuracy2 = tf.reduce_mean(tf.cast(correct_pred2, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    saver = tf.train.Saver(max_to_keep=5)
    for step in range(1, training_steps+1):
        loss_ep1, acc_ep1, loss_ep2, acc_ep2 = [], [], [], []
        for idx in range(batch_num_train):
            input_x, input_y1, input_y2 = batch_x_train[idx], batch_y1_train[idx], batch_y2_train[idx]
            sess.run(train_op1, feed_dict={x: input_x, y1: input_y1})
            sess.run(train_op2, feed_dict={x: input_x, y2: input_y2})
            # sess.run(train_op, feed_dict={x: input_x, y1: input_y1, y2: input_y2})

            # Calculate batch loss and accuracy
            loss1, acc1 = sess.run([loss_op1, accuracy1], feed_dict={x: input_x, y1: input_y1})
            loss2, acc2 = sess.run([loss_op2, accuracy2], feed_dict={x: input_x, y2: input_y2})

            loss_ep1.append(loss1)
            acc_ep1.append(acc1)
            loss_ep2.append(loss2)
            acc_ep2.append(acc2)

        # Save the variables to disk
        save_path = saver.save(sess, '../ckpt/' + ckpt_folder + "/model.ckpt." +
                               datetime.now().strftime("%Y-%m-%d-%H-%M"),
                               global_step=step)

        loss_avg1 = sum(loss_ep1) / len(loss_ep1)
        acc_avg1 = sum(acc_ep1) / len(acc_ep1)
        loss_avg2 = sum(loss_ep2) / len(loss_ep2)
        acc_avg2 = sum(acc_ep2) / len(acc_ep2)

        test_x, test_y1, test_y2 = batch_x_test[0], batch_y1_test[0], batch_y2_test[0]
        test_acc1 = sess.run(accuracy1, feed_dict={x: test_x, y1: test_y1})
        test_acc2 = sess.run(accuracy2, feed_dict={x: test_x, y2: test_y2})

        print("Step:", step)
        print("[Intent]       Training Loss= " +
              "{:.4f}".format(loss_avg1) + ", Training Accuracy= " +
              "{:.3f}".format(acc_avg1) + ", Testing Accuracy= " +
              "{:.3f}".format(test_acc1))
        print("[Slot-filling] Training Loss= " +
              "{:.4f}".format(loss_avg2) + ", Training Accuracy= " +
              "{:.3f}".format(acc_avg2) + ", Testing Accuracy= " +
              "{:.3f}".format(test_acc2))
        print('Checkpoint:', save_path)
        print("=" * 70)

        # simple early stop
        if loss_avg1 < 0.1 and loss_avg2 < 0.1:
            print('Early Stop at step {}'.format(step))
            break

print('Total Samples', num_samples_train)
end = datetime.now()
print('Data Processing:', str(dp - start))
print('Training Duration:', str(end-start))

