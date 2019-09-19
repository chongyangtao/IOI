import os
import pickle
from collections import defaultdict
import logging
import time
import numpy as np
from random import shuffle
import codecs
import concurrent.futures
from datetime import datetime
import tensorflow as tf

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    '''Pads each sequence to the same length:
    the length of the longest sequence.

    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.

    Supports post-padding and pre-padding (default).

    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.

    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

def load_file(input_file, word2idx_file, isshuffle = True):
    word2idx = pickle.load(open(word2idx_file, 'rb'))
    revs = []
    response_set = []
    with open(input_file, 'r') as f: 
        for k, line in enumerate(f):
            parts = line.strip().split("\t")
            label = parts[0]
            context = parts[1:-1] # multi-turn
            if len(context) > 10:
                    context = context[-10:]

            response = parts[-1]
            data = {"y": label, "c": context, "r": response}
            revs.append(data)
            response_set.append(response)
    print("Processed dataset with %d context-response pairs " % (len(revs)))
    if isshuffle == True:
        shuffle(revs)

    return revs, response_set, word2idx


def get_word_idx_from_sent(sent, word_idx_map, max_word_len=50):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    token_ids = [word_idx_map.get(word, 0) for word in sent.split()]
    x = pad_sequences([token_ids], padding='post', maxlen=max_word_len)[0]
    x_len = min(len(token_ids), max_word_len)

    return x, x_len

def get_word_idx_from_sent_msg(sents, word_idx_map, max_turn=10, max_word_len=50):
    word_turns = []
    word_lens = []

    for sent in sents:
        words = sent.split()
        token_ids = [word_idx_map.get(word, 0) for word in words]
        x = pad_sequences([token_ids], padding='post', maxlen=max_word_len)[0]
        x_mask = pad_sequences([len(token_ids)*[1]], padding='post', maxlen=max_word_len)[0]
        
        word_turns.append(x)
        word_lens.append(min(len(words), max_word_len))

    word_turns_new = np.zeros([max_turn, max_word_len], dtype=np.int32) 
    word_lens_new = np.zeros([max_turn], dtype=np.int32) 

    if len(word_turns) <= max_turn:
        word_turns_new[-len(word_turns):]= word_turns
        word_lens_new[-len(word_turns):] = word_lens
       
    if len(word_turns) > max_turn:
        word_turns_new[:] = word_turns[len(word_turns)-max_turn:len(word_turns)]
        word_lens_new[:] = word_lens[len(word_turns)-max_turn:len(word_turns)]


    # print("sents: ", sents)
    # print("word_turns_new: ", word_turns_new)
    # print("word_lens_new: ", word_lens_new)
    # print("\n")
    # time.sleep(20)

    return word_turns_new, word_lens_new, len(sents)




def build_records(data_file, word2idx_file, records_name, max_turn=10, max_utterance_len=50, isshuffle=False, max_mum=100000000):
    revs, response_set, word2idx= load_file(data_file, word2idx_file, isshuffle)
    print("load data done ...")
    writer = tf.python_io.TFRecordWriter(records_name)
    for k, rev in enumerate(revs):
        context, context_len, turn = get_word_idx_from_sent_msg(rev["c"], word2idx, max_turn, max_utterance_len)
        response, response_len = get_word_idx_from_sent(rev['r'], word2idx, max_utterance_len)
        y_label = int(rev["y"]) 
        features = {
            'context': tf.train.Feature(bytes_list=tf.train.BytesList(value=[context.tostring()])),
            'context_len': tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_len.tostring()])),
            'response': tf.train.Feature(bytes_list=tf.train.BytesList(value=[response.tostring()])),
            'response_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[response_len])),
            'turn': tf.train.Feature(int64_list=tf.train.Int64List(value=[turn])),
            'y_label': tf.train.Feature(int64_list=tf.train.Int64List(value=[y_label]))   
  
        }

        tf_features = tf.train.Features(feature=features)
        tf_example = tf.train.Example(features=tf_features)
        tf_serialized = tf_example.SerializeToString()
        writer.write(tf_serialized)
        if((k+1)%10000==0):
            print('Write {} examples to {}'.format(k+1, records_name))
        if (k+1)>=max_mum:
            break
    writer.close()

def get_record_parser(FLAGS):
    def _parser(example_proto):
        dics = {
            'context': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'context_len': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'response': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'response_len': tf.FixedLenFeature(shape=[], dtype=tf.int64),
            'turn': tf.FixedLenFeature(shape=[], dtype=tf.int64),
            'y_label': tf.FixedLenFeature(shape=[], dtype=tf.int64)
        }

        parsed_example = tf.parse_single_example(serialized=example_proto,features=dics)

        context = tf.reshape(tf.decode_raw(parsed_example["context"], tf.int32), [FLAGS.max_turn, FLAGS.max_utterance_len])
        context_len = tf.reshape(tf.decode_raw(parsed_example["context_len"], tf.int32), [FLAGS.max_turn])
        response = tf.reshape(tf.decode_raw(parsed_example["response"], tf.int32), [FLAGS.max_utterance_len])
        response_len = parsed_example["response_len"]
        y_label = parsed_example["y_label"]
        turn = parsed_example["turn"]

        return context, context_len, response, response_len, turn, y_label
    return _parser


def get_batch_dataset(record_file, parser, batch_size, num_threads, capacity, is_test=False):
    num_threads = tf.constant(num_threads, dtype=tf.int32)
    
    if is_test:
        dataset = tf.data.TFRecordDataset(record_file).map(parser, num_parallel_calls=num_threads).repeat(1).batch(batch_size) 
    else:
        dataset = tf.data.TFRecordDataset(record_file).map(parser, num_parallel_calls=num_threads).shuffle(capacity).repeat().batch(batch_size)
    return dataset

def process_word2vec(word2vec_file, emb_size,  total_words=10000000,  out_dict_file='word_dict.pkl', out_emb_file='word_emb_matrix.pkl'):
    word_dict = dict()
    vectors = []
    zero_vec = list(np.zeros(emb_size, dtype=float))
    vectors.append(zero_vec) # for pad
    vectors.append(zero_vec) # for unk
    word_dict['<pad>'] = 0
    word_dict['<unk>'] = 1

    with open(word2vec_file, 'r', encoding = "ISO-8859-1") as f:  
        # there exits an useless line in word2vec 
        lines = f.readlines()[1:]  
        for i, line in enumerate(lines):
            line = line.rstrip().split(' ')
            word_dict[line[0]] = i + 2

            if len(list(map(float, line[1:]))) !=emb_size:
                print(word2vec_file, i, '_%s_'%line[0], len(list(map(float, line[1:]))))

            vectors.append(list(map(float, line[1:])))
            if i > total_words:
                break
        
    with open(out_emb_file, 'wb') as f:
        pickle.dump(vectors, f) # 
    with open(out_dict_file, 'wb') as f:
        pickle.dump(word_dict, f)


if __name__ == "__main__":
    emb_size = 200
    data_path = 'data/ubuntu'

    process_word2vec(os.path.join(data_path, 'ubuntu.200d.word2vec'), \
                        emb_size, \
                        out_dict_file=os.path.join(data_path, 'word_dict.pkl'), \
                        out_emb_file=os.path.join(data_path, 'word_emb_matrix.pkl'))
    

    if 0:
        build_records(os.path.join(data_path, 'train.txt'), 
                        os.path.join(data_path, 'word_dict.pkl'),
                        os.path.join(data_path, 'train.small.tfrecords'), isshuffle=True, max_mum=20000)
        build_records(os.path.join(data_path, 'valid.txt'), 
                        os.path.join(data_path, 'word_dict.pkl'),
                        os.path.join(data_path, 'valid.small.tfrecords'), max_mum=10000)

        build_records(os.path.join(data_path, 'test.txt'), 
                        os.path.join(data_path, 'word_dict.pkl'),
                        os.path.join(data_path, 'test.char.small.tfrecords'), max_mum=10000)
    else:
        build_records(os.path.join(data_path, 'train.txt'), 
                        os.path.join(data_path, 'word_dict.pkl'),
                        os.path.join(data_path, 'train.tfrecords'), isshuffle=True)
        build_records(os.path.join(data_path, 'valid.txt'), 
                        os.path.join(data_path, 'word_dict.pkl'),
                        os.path.join(data_path, 'valid.tfrecords'))

        build_records(os.path.join(data_path, 'test.txt'), 
                        os.path.join(data_path, 'word_dict.pkl'),
                        os.path.join(data_path, 'test.tfrecords'))

