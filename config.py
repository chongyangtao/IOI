import tensorflow as tf
def get_args():

    tf.flags.DEFINE_string('data_path', '../ubuntu', 'Path to dataset. ')
    
    tf.flags.DEFINE_boolean("auto_gpu", True, "Automatically select gpu")
    tf.flags.DEFINE_integer("num_threads", 4, "Dimensionality of embedding")
    tf.flags.DEFINE_integer("capacity", 15000, "Dimensionality of embedding")


    tf.flags.DEFINE_string("model_pattern", 'ioi', "use which model")
    tf.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')

    tf.flags.DEFINE_integer("num_layer", 7, "Number of stacked layers")
    tf.flags.DEFINE_boolean("init_dict", True, "use initial word2vec")
    tf.flags.DEFINE_integer("vocab_size", 5000000, "Size of vocabulary")
    tf.flags.DEFINE_integer("max_turn", 10, "Max length of turn")
    tf.flags.DEFINE_integer("max_utterance_len", 50, "Max length of word") 

    tf.flags.DEFINE_integer("embed_dim", 200, "Dimensionality of embedding")
    tf.flags.DEFINE_integer("hidden_dim", 200, "Dimensionality of rnn")
    tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability") 


    tf.flags.DEFINE_string('optimizer', 'adam', 'Which optimization method to use') # adam 0.001  adadelta
    tf.flags.DEFINE_float('lr', 0.0005, 'learning rate')
    tf.flags.DEFINE_boolean("lr_decay", True, "Allow reload the model")
    tf.flags.DEFINE_float('decay_rate', 0.9, 'decay rate')  
    tf.flags.DEFINE_integer('decay_steps', 2000, 'decay steps')  
    tf.flags.DEFINE_float('lr_minimal', 0.00005, 'minimal learning rate') # 0.00002


    tf.flags.DEFINE_boolean('use_globalLoss', False, 'l2 loss rate') 
    tf.flags.DEFINE_boolean('use_loss_decay', False, 'l2 loss rate') 
    tf.flags.DEFINE_float('clip_value', 10.0, 'clip_value')


    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 20, "Batch Size")
    tf.flags.DEFINE_integer("num_epochs", 200000, "Number of training epochs")
    tf.flags.DEFINE_integer("print_every", 50, "Print the results after this many steps")
    tf.flags.DEFINE_integer("valid_every", 2000, "Evaluate model on dev set after this many steps")
    tf.flags.DEFINE_integer("test_every", 2000, "Testing model after this many step")
    tf.flags.DEFINE_integer("checkpoint_every", 2000, "Save model after this many step")

    tf.flags.DEFINE_boolean("reload_model", False, "Allow reload the model")
    tf.flags.DEFINE_string('log_root', 'logs_tmp/', 'Root directory for all logging.')


    # Misc Parameters
    tf.flags.DEFINE_integer('gpu',10 , 'Which GPU to use')
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags() 
    return FLAGS

