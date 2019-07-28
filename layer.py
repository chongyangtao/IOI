import tensorflow as tf


initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)
initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                             mode='FAN_IN',
                                                             uniform=False,
                                                             dtype=tf.float32)
regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7)

def linear(args, output_size, bias=False):
    total_arg_size = 0
    shapes = [arg.get_shape() for arg in args]
    for shape in shapes:
        if shape[-1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[-1].value
    # Set data type
    dtype = args[0].dtype

    # Compute linear part
    _scope = tf.get_variable_scope()
    with tf.variable_scope(_scope) as outer_scope:
        W = tf.get_variable('W', [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
        logits = tf.einsum('aij,jk->aik', args[0], W) #TODO: check
        #logits = tf.matmul(args[0], W)
    else:
        logits = tf.einsum('aij,jk->aik', tf.concat(args, -1), W)
        #logits = tf.matmul(tf.concat(args, -1), W)
    if not bias:
        return logits
    b = tf.get_variable('b',
                        [output_size],
                        dtype=dtype,
                        initializer=tf.constant_initializer(0.0, dtype=dtype))
    #return tf.sigmoid(tf.nn.bias_add(logits, b))
    return tf.nn.bias_add(logits, b)


def masked_softmax(scores, mask):
    """
    Used to calculcate a softmax score with true sequence length (without padding), rather than max-sequence length.
    Input shape: (batch_size, len_res, len_utt).
    mask parameter: Tensor of shape (batch_size, len_utt). Such a mask is given by the length() function.
    return shape: [batch_size, len_res, len_utt]
    """
    numerator = tf.exp(tf.subtract(scores, tf.reduce_max(scores, 2, keep_dims=True))) * tf.expand_dims(mask, axis=1)
    denominator = tf.reduce_sum(numerator, 2, keep_dims=True)

    # weights = tf.div(numerator, denominator)
    # weights = tf.div(numerator, denominator+1e-3)
    weights = tf.div(numerator + 1e-5 / mask.get_shape()[-1].value, denominator + 1e-5)
    return weights



def normalize(inputs, axis=None, epsilon=1e-8, scope='normalize', reuse=None):
    '''Add layer normalization.
    Args:
        x: a tensor
        axis: the dimensions to normalize

    Returns:
        a tensor the same shape as x.

    Raises:
    '''
    with tf.variable_scope(scope, reuse=reuse):
        if axis is None:
            axis = [-1]
        shape = [inputs.shape[i] for i in axis]

        scale = tf.get_variable(name='scale', shape=shape, dtype=tf.float32, initializer=tf.ones_initializer())
        bias = tf.get_variable(name='bias', shape=shape, dtype=tf.float32, initializer=tf.zeros_initializer())

        mean = tf.reduce_mean(inputs, axis=axis, keep_dims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=axis, keep_dims=True)
        # mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True) # more fast

        norm = (inputs-mean) * tf.rsqrt(variance + epsilon)
        return scale * norm + bias


def conv(inputs, output_size, kernel_size = [1,2,3,4], bias = None, activation = None, name = "conv", isNormalize= False, reuse = None):
    with tf.variable_scope(name, reuse = reuse):
        conv_features = []
        shapes = inputs.shape.as_list()
        for k in kernel_size:
            filter_shape = [k, shapes[-1], output_size]
            bias_shape = [1,1,output_size]
            strides = 1
            kernel_ = tf.get_variable("kernel_%s"%k,
                            filter_shape,
                            dtype = tf.float32,
                            regularizer=regularizer,
                            initializer = initializer())
            feature = tf.nn.conv1d(inputs, kernel_, strides, "SAME")
            if bias:
                feature += tf.get_variable("bias_%s"%k,
                            bias_shape,
                            regularizer=regularizer,
                            initializer = tf.zeros_initializer())
            if activation is not None:
                feature = activation(feature)
            conv_features.append(feature)
        output = tf.concat(conv_features, axis=-1)
        if isNormalize:
            output = normalize(output, 1e-8, "normalize", reuse) 
        return output



def multihead_attention(queries, keys, query_masks=None, key_masks=None, num_units=None, num_heads=8, dropout_rate=0,
                            is_training=True, causality=False, scope="multihead_attention", reuse=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu, name="dense_q")  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu, name="dense_k")  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu, name="dense_v")  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        if key_masks is None:
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        if query_masks is None:
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = normalize(outputs)  # (N, T_q, C)


        # Feed Forward
        outputs = feedforward(outputs, num_units=[num_units, num_units], scope='feed_forward')


    return outputs


def self_attention(queries, keys, num_units, query_masks=None, key_masks=None, num_blocks=6, num_heads=1, 
                        dropout_rate=0, causality=False, use_linear=False, use_residual=True, use_feed=True, 
                        attention_flag='full', is_training=False, scope=None, reuse=None, queries_hist=None, keys_hist=None):
    '''Applies singlehead attention.
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    hiddens =[]
    hiddens.append(queries)
    with tf.variable_scope(scope or "self_attention", reuse=reuse):        
        # Linear projections
        if use_linear:
            queries = tf.layers.dense(queries, num_units, activation=tf.nn.relu, name="dense_q")  # (N, T_q, C)
            keys = tf.layers.dense(keys, num_units, activation=tf.nn.relu, name="dense_k")  # (N, T_k, C)
            values = tf.layers.dense(keys, num_units, activation=tf.nn.relu, name="dense_v")  # (N, T_k, C)
        else:
            values = keys

        if attention_flag=='dot':
            if queries_hist==None:
                outputs = tf.matmul(queries, tf.transpose(keys, [0, 2, 1]))  # (N, T_q, T_k)
            else:
                outputs = tf.matmul(tf.concat([queries, queries_hist], axis=-1), tf.transpose(tf.concat([keys, keys_hist], axis=-1), [0, 2, 1]))  # (N, T_q, T_k)
        else:
            if queries_hist==None:
                outputs = full_attention(queries, keys) # fully aware attention
            else:
                outputs = full_attention(tf.concat([queries, queries_hist], axis=-1), tf.concat([keys, keys_hist], axis=-1)) # fully aware attention

        # Scale
        scale = tf.maximum(1.0, keys.get_shape().as_list()[-1] ** 0.5)
        outputs = outputs / scale

        # Key Masking
        if key_masks is None:
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (N, T_q, T_k)

        # For mask_softmax activation
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (N, T_q, T_k)
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        if query_masks is None:
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, values)  # ( h*N, T_q, C/h)

        if use_residual:
            # Residual connection
            outputs += queries
            # Normalize
            outputs = normalize(outputs)  # (N, T_q, C)

        # Feed Forward
        if use_feed:
            outputs = feedforward(outputs, num_units=[num_units, num_units], scope='feed_forward')

        hiddens.append(outputs)
    return hiddens # tf.stack(encs, axis=-1)


def feedforward(inputs, num_units=[2048, 512], scope="feed_forward", use_dense=True, reuse=None):
    '''Point-wise feed forward net.
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        if use_dense:
            outputs = tf.layers.dense(inputs, num_units[0], activation = tf.nn.relu, 
                                            # kernel_initializer = tf.contrib.keras.initializers.he_normal(), 
                                            use_bias=True, name="dense_1")  # (N, T_q, C)
            outputs = tf.layers.dense(outputs, num_units[1], activation=None, 
                                            # kernel_initializer = tf.contrib.layers.xavier_initializer(), 
                                            use_bias=True, name="dense_2")  # (N, T_q, C)  
        else:          
            params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            
            params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                      "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
        
        # Residual connection
        outputs += inputs
        # Normalize
        outputs = normalize(outputs)
            
    return outputs



def cross_attention(query, key, dim, scope="cross_attention", reuse=None):
    '''Cross interaction. 
    Args:
      query: [batch, len1, dim] 
      key: [batch, len2, dim]
      scope: Optional scope for `variable_scope`.

    Returns:
      A 3d tensor with the same shape and dtype as query
    '''

    with tf.variable_scope(scope, reuse=reuse):
        Weight = tf.get_variable('Weight', shape=(dim, dim), dtype=tf.float32)
        matrix = tf.einsum('aij,jk->aik', query, Weight) # [?, len1, dim]
        # [?, len1, dim] * [?, len2, dim] ->[?, len1, len2]
        matrix = tf.einsum('aij,akj->aik', matrix, key)

        # [?, len2, dim] * [?, len1, len2] -> [?, len1, dim]
        matrix_feature = tf.einsum('aij,aki->akj', key, tf.nn.tanh(matrix))
        # [batch, len1, dim]
        cross_att = tf.multiply(query, matrix_feature)
    return cross_att


def coattention_nnsubmulti(utterance, response, utterance_mask, scope="co_attention", reuse=None):
    '''Point-wise interaction. (NNSUBMULTI)
    Args:
      utterance: [turns, len_utt, dim] 
      response: [turns, len_res, dim]
      utterance_mask: [turns, len_utt] 
      scope: Optional scope for `variable_scope`.

    Returns:
      A 3d tensor with the same shape and dtype as response
    '''

    with tf.variable_scope(scope, reuse=reuse):
        dim = utterance.get_shape().as_list()[-1]

        weight = tf.get_variable('Weight', shape=[dim, dim], dtype=tf.float32)
        e_utterance = tf.einsum('aij,jk->aik', utterance, weight)
        # [batch, len_res, dim] * [batch, len_utterance, dim]T -> [batch, len_res, len_utterance]
        a_matrix = tf.matmul(response, tf.transpose(e_utterance, perm=[0,2,1]))
        
        reponse_atten = tf.matmul(masked_softmax(a_matrix, utterance_mask), utterance)

        feature_mul = tf.multiply(reponse_atten, response)
        feature_sub = tf.subtract(reponse_atten, response)
        feature_last = tf.layers.dense(tf.concat([feature_mul, feature_sub], axis=-1), dim, use_bias=True, 
                                                activation=tf.nn.relu, reuse=reuse)     # [batch*turn, len, dim]
    return feature_last



def full_attention(utt_how, resp_how, dim=None, scope="full_attention", reuse=None):
    '''Fully aware attention
    Args:
      utt_how: [batch, len_utt, dim] 
      resp_how: [batch, len_res, dim]
      scope: Optional scope for `variable_scope`.

    Returns:
      A 3d tensor with the same shape and dtype as response
    '''
    if dim==None:
        dim = utt_how.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=reuse):
        U = tf.get_variable('Weight_U', shape=[dim, dim], dtype=tf.float32)

        I = tf.eye(dim)
        D = tf.get_variable('Weight_D', shape=[dim, dim], dtype=tf.float32)
        D = tf.multiply(D, I)  # restrict to diagonal

        f1 = tf.nn.relu(tf.einsum('aij,jk->aik', utt_how, U), name='utt_how_relu') # [batch, len_utt, dim]
        f2 = tf.nn.relu(tf.einsum('aij,jk->aik', resp_how, U), name='resp_how_relu') # [batch, len_res, dim]
        S = tf.einsum('aij,jk->aik', f1, D)  # [batch, len_utt, dim]
        S = tf.einsum('aij,akj->aik', S, f2) # [batch, len_utt,len_res]
        # S = tf.nn.softmax(S, dim=-1)
    return S


def bilinear_attention(utt_how, resp_how, dim, scope="full_attention", reuse=None):
    '''Bilinear  attention
    Args:
      utt_how: [batch, len_utt, dim] 
      resp_how: [batch, len_res, dim]
      scope: Optional scope for `variable_scope`.

    Returns:
      A 3d tensor with the same shape and dtype as response
    '''

    with tf.variable_scope(scope, reuse=reuse):
        U = tf.get_variable('Weight_U', shape=[dim, dim], dtype=tf.float32)
        T = tf.einsum('aij,jk->aik', utt_how, U) # [batch, len_utt, dim]
        S = tf.einsum('aij,akj->aik', T, resp_how) # [batch, len_utt,len_res]
        # S = tf.nn.softmax(S, dim=-1)
    return S


def symmetric_attention(utt_how, resp_how, dim, scope="full_attention", reuse=None):
    '''Bilinear  attention
    Args:
      utt_how: [batch, len_utt, dim] 
      resp_how: [batch, len_res, dim]
      scope: Optional scope for `variable_scope`.

    Returns:
      A 3d tensor with the same shape and dtype as response
    '''

    with tf.variable_scope(scope, reuse=reuse):
        U = tf.get_variable('Weight_U', shape=[dim, dim], dtype=tf.float32)
        I = tf.eye(dim)
        D = tf.get_variable('Weight_D', shape=[dim, dim], dtype=tf.float32)
        D = tf.multiply(D, I)  # restrict to diagonal
        
        f1 = tf.einsum('aij,jk->aik', utt_how, U) # [batch, len_utt, dim]
        f2 = tf.einsum('aij,jk->aik', resp_how, U) # [batch, len_res, dim]

        T = tf.einsum('aij,jk->aik', utt_how, U) # [batch, len_utt, dim]
        S = tf.einsum('aij,akj->aik', T, resp_how) # [batch, len_utt,len_res]
    return S

