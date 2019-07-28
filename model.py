import os, sys
import tensorflow as tf
import numpy as np
from layer import *


class model(object):
    def __init__(self, batch, FLAGS, pretrained_word_embeddings=None):
       
        embed_dim = FLAGS.embed_dim
        vocab_size = FLAGS.vocab_size
        hidden_dim = FLAGS.hidden_dim
        max_turn = FLAGS.max_turn
        max_word_len = FLAGS.max_utterance_len

        self.is_training = tf.placeholder(tf.bool, [])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.context, self.context_len, self.response, self.response_len, self.turn, self.target = batch.get_next()

        self.context_mask = tf.cast(tf.not_equal(self.context, 0), tf.float32)
        self.response_mask = tf.cast(tf.not_equal(self.response, 0), tf.float32)
        self.turn = tf.cast( self.turn, tf.int32)
        self.target = tf.cast( self.target, tf.int32)

        self.expand_response_mask = tf.tile(tf.expand_dims(self.response_mask, 1), [1, max_turn, 1]) 
        self.expand_response_mask = tf.reshape(self.expand_response_mask, [-1, max_word_len])  
        self.parall_context_mask = tf.reshape(self.context_mask, [-1, max_word_len])  

        self.y_pred = 0.0
        self.loss = 0.0
        self.loss_list = []

        with tf.variable_scope("word_embeddings"):
            word_embeddings = tf.get_variable('word_embeddings_v', shape=(vocab_size, embed_dim), dtype=tf.float32, trainable=True)  
            if pretrained_word_embeddings is not None:
                self.embedding_init = word_embeddings.assign(pretrained_word_embeddings)

            self.context_embeddings = tf.nn.embedding_lookup(word_embeddings, self.context)  
            self.response_embeddings = tf.nn.embedding_lookup(word_embeddings, self.response)  
            self.context_embeddings = tf.layers.dropout(self.context_embeddings, rate=1.0-self.dropout_keep_prob)
            self.response_embeddings = tf.layers.dropout(self.response_embeddings, rate=1.0-self.dropout_keep_prob)
            self.context_embeddings = tf.multiply(self.context_embeddings, tf.expand_dims(self.context_mask, axis=-1))  
            self.response_embeddings = tf.multiply(self.response_embeddings, tf.expand_dims(self.response_mask, axis=-1)) 


        self.expand_response_embeddings = tf.tile(tf.expand_dims(self.response_embeddings, 1), [1, max_turn, 1, 1]) 
        self.expand_response_embeddings = tf.reshape(self.expand_response_embeddings, [-1, max_word_len, embed_dim]) 
        self.parall_context_embeddings = tf.reshape(self.context_embeddings, [-1, max_word_len, embed_dim])
        context_rep, response_rep = self.parall_context_embeddings, self.expand_response_embeddings

        losses_list = []
        y_pred_list = []
        logits_list=[]
        for k in range(FLAGS.num_layer):
            inter_feat_collection = []
            with tf.variable_scope('dense_interaction_{}'.format(k)): 
                # get the self rep
                context_self_rep = self_attention(context_rep, context_rep, embed_dim, 
                                                    query_masks=self.parall_context_mask, 
                                                    key_masks=self.parall_context_mask, 
                                                    num_blocks=1, num_heads=1, 
                                                    dropout_rate=1.0-self.dropout_keep_prob,
                                                    use_residual=True, use_feed=True, 
                                                    scope='context_self_attention')[1]  # [batch*turn, len_utt, embed_dim, 2]
                response_self_rep = self_attention(response_rep, response_rep, embed_dim, 
                                                    query_masks=self.expand_response_mask, 
                                                    key_masks=self.expand_response_mask, 
                                                    num_blocks=1, num_heads=1, 
                                                    dropout_rate=1.0-self.dropout_keep_prob, 
                                                    use_residual=True, use_feed=True, 
                                                    scope='response_self_attention')[1]  # [batch*turn, len_res, embed_dims, 2]

                # get the attended rep
                context_cross_rep = self_attention(context_rep, response_rep, embed_dim, 
                                                    query_masks=self.parall_context_mask, 
                                                    key_masks=self.expand_response_mask, 
                                                    num_blocks=1, num_heads=1, 
                                                    dropout_rate=1.0-self.dropout_keep_prob, 
                                                    use_residual=True, use_feed=True, 
                                                    scope='context_cross_attention')[1]  # [batch*turn, len_utt, embed_dim]

                response_cross_rep = self_attention(response_rep, context_rep, embed_dim, 
                                                    query_masks=self.expand_response_mask, 
                                                    key_masks=self.parall_context_mask, 
                                                    num_blocks=1, num_heads=1, 
                                                    dropout_rate=1.0-self.dropout_keep_prob, 
                                                    use_residual=True, use_feed=True, 
                                                    scope='response_cross_attention')[1]  # [batch*turn, len_res, embed_dim]


                context_inter_feat_multi = tf.multiply(context_rep, context_cross_rep)
                response_inter_feat_multi = tf.multiply(response_rep, response_cross_rep)


                context_concat_rep = tf.concat([context_rep, context_self_rep, context_cross_rep, context_inter_feat_multi], axis=-1) 
                response_concat_rep = tf.concat([response_rep, response_self_rep, response_cross_rep, response_inter_feat_multi], axis=-1)


                context_concat_dense_rep = tf.layers.dense(context_concat_rep, embed_dim, activation=tf.nn.relu, use_bias=True, 
                                                                    name='context_dense1') 
                context_concat_dense_rep = tf.layers.dropout(context_concat_dense_rep, rate=1.0-self.dropout_keep_prob)

                response_concat_dense_rep = tf.layers.dense(response_concat_rep, embed_dim,  activation=tf.nn.relu, use_bias=True, 
                                                                    name='response_dense1') 
                response_concat_dense_rep = tf.layers.dropout(response_concat_dense_rep, rate=1.0-self.dropout_keep_prob)

              
                inter_feat = tf.matmul(context_rep, tf.transpose(response_rep, perm=[0, 2, 1])) / tf.sqrt(tf.to_float(embed_dim))
                inter_feat_self = tf.matmul(context_self_rep, tf.transpose(response_self_rep, perm=[0, 2, 1])) / tf.sqrt(tf.to_float(embed_dim))
                inter_feat_cross = tf.matmul(context_cross_rep, tf.transpose(response_cross_rep, perm=[0, 2, 1])) / tf.sqrt(tf.to_float(embed_dim))


                inter_feat_collection.append(inter_feat)
                inter_feat_collection.append(inter_feat_self)
                inter_feat_collection.append(inter_feat_cross)

                if k==0:
                    context_rep = tf.add(context_rep, context_concat_dense_rep)
                    response_rep = tf.add(response_rep, response_concat_dense_rep)
                else:
                    context_rep = tf.add_n([self.parall_context_embeddings, context_rep, context_concat_dense_rep])
                    response_rep = tf.add_n([self.expand_response_embeddings, response_rep, response_concat_dense_rep])

                context_rep = normalize(context_rep, scope='layer_context_normalize') 
                response_rep = normalize(response_rep, scope='layer_response_normalize') 

                context_rep = tf.multiply(context_rep, tf.expand_dims(self.parall_context_mask, axis=-1))
                response_rep = tf.multiply(response_rep, tf.expand_dims(self.expand_response_mask, axis=-1))

                matching_feat = tf.stack(inter_feat_collection, axis=-1)
                matrix_trans = tf.reshape(matching_feat, [-1, max_turn, max_word_len, max_word_len, len(inter_feat_collection)])  # embed_dim

            with tf.variable_scope('CRNN_{}'.format(k)): 
                conv1 = tf.layers.conv2d(matching_feat, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                            activation=tf.nn.relu, name='conv1')
                pool1 = tf.layers.max_pooling2d(conv1, (3, 3), strides=(3, 3), padding='same', name='max_pooling1')
                
                conv2 = tf.layers.conv2d(pool1, filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                            activation=tf.nn.relu, name='conv2')
                pool2 = tf.layers.max_pooling2d(conv2, (3, 3), strides=(3, 3), padding='same', name='max_pooling2')                    
                flatten = tf.contrib.layers.flatten(pool2)
                flatten = tf.layers.dropout(flatten, rate=1.0-self.dropout_keep_prob)

                matching_vector = tf.layers.dense(flatten, embed_dim,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      activation=tf.tanh, name='dense_feat') 
                matching_vector = tf.reshape(matching_vector, [-1, max_turn, embed_dim]) 

                final_gru_cell = tf.contrib.rnn.GRUCell(embed_dim, kernel_initializer=tf.orthogonal_initializer())
                _, last_hidden = tf.nn.dynamic_rnn(final_gru_cell, matching_vector, dtype=tf.float32, scope='final_GRU')  # TODO: check time_major
                logits = tf.layers.dense(last_hidden, 2, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='final_v')

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=logits)
            loss = tf.reduce_mean(tf.clip_by_value(loss, -FLAGS.clip_value, FLAGS.clip_value))
            y_pred = tf.nn.softmax(logits)
            
            losses_list.append(loss) 
            y_pred_list.append(y_pred) 
            logits_list.append(logits)

        if FLAGS.use_loss_decay:
            self.loss =sum([((idx+1)/float(FLAGS.num_layer))*item for idx, item in enumerate(losses_list)])
        else:
            self.loss = sum(losses_list)
        self.loss_list = losses_list

        self.y_pred = sum(y_pred_list)

        if FLAGS.use_globalLoss:
            logits_sum = tf.add_n(logits_list)
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=logits_sum)
            self.loss = tf.reduce_mean(tf.clip_by_value(self.loss, -FLAGS.clip_value, FLAGS.clip_value))
            self.loss_list = [self.loss]
            self.y_pred = tf.nn.softmax(logits_sum) 


        self.correct = tf.equal(tf.cast(tf.argmax(self.y_pred, axis=1), tf.int32), tf.to_int32(self.target))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, 'float'))
