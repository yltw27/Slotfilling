import tensorflow as tf
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
from datetime import datetime

s = datetime.now()

checkpoint_path = '../ckpt/2017-09-05/'
target_ckpt_file = 'model.ckpt.2017-09-05-21-38-100'

intent_dict = pd.read_csv('../data/ATIS/intent_dict.csv', header=0)
label_dict = pd.read_csv('../data/ATIS/label_dict.csv', header=0)
print('dictionaries loaded')

word_vectors = KeyedVectors.load_word2vec_format("../word2vec/en.model.bin", binary=True)
print('word2vec model loaded')


# data processing: input_x
def seq2vec(seq):
    num_input = 50
    seq_list = seq.split(' ')
    sentence = []
    for i in range(len(seq_list)):
        try:
            sentence.append(word_vectors.get_vector(seq_list[i]))
        except KeyError:
            sentence.append(np.random.randint(low=3, size=300))
    length = len(sentence)
    if length < num_input:
        for x in range(num_input - length):
            sentence.append(np.zeros((300,)))
    return np.reshape(np.vstack(sentence), [1, 50, 300])  # return a numpy array, shape=(num_input, 300)


def inference(data):
    x = graph.get_tensor_by_name("x:0")  # (?, 50, 300)
    y1 = graph.get_tensor_by_name("y1:0")  # (?, 22)
    y2 = graph.get_tensor_by_name('y2:0')  # (?, 50, 122)

    b = np.zeros((1, 22))
    c = np.zeros((1, 50, 122))

    feed_dict_labels = {x: data, y1: b}
    feed_dict_intent = {x: data, y2: c}

    intent_pred = graph.get_tensor_by_name("intent_pred:0")
    label_pred = graph.get_tensor_by_name("seq_pred:0")

    result_intent = sess.run(intent_pred, feed_dict_intent)
    result_labels = sess.run(label_pred, feed_dict_labels)[0]

    intent = intent_dict.loc[intent_dict['idx'] == result_intent[0]+2]['intent'].values[0]

    labels = ''
    for i in range(len(result_labels)):
        if result_labels[i] == 0:
            break
        temp = label_dict.loc[label_dict['idx'] == result_labels[i]+1]['label'].values[0]
        labels += temp+' '

    return intent, labels[2:]


def run():
    query = input('>>>Query: ')
    start_point = datetime.now()
    intent, result_labels = inference(seq2vec(query))
    end_point = datetime.now()
    print('Intent: {}\nEntity: {}\ninference time: {}'.format(intent, result_labels, str(end_point - start_point)))
    print('=' * 70)

if __name__ == '__main__':
    sess = tf.Session()
    saver = tf.train.import_meta_graph(checkpoint_path + target_ckpt_file + '.meta')
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
    graph = tf.get_default_graph()
    e = datetime.now()
    print('running time before inference: {}'.format(str(e - s)))

    while True:
        run()
