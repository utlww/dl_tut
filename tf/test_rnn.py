import tensorflow as tf
from tensorflow.python.layers import core as layers_core

import iterator_utils


def get_device_str(device_id, num_gpus):
    "device string for multi-gpu setup. e.g. if num_gpus=3, device_id can be (0, 1, 2)."
    if num_gpus == 0:
        return "/cpu:0"
    res = "/gpu:%d" % (device_id % num_gpus)
    return res


def _single_cell(unit_type, num_units, forget_bias, dropout,
                 residual_connection=False, device_str=None):

    if unit_type == "lstm":
        single_cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=forget_bias)
    elif unit_type == "gru":
        single_cell = tf.contrib.rnn.GRUCell(num_units)
    else:
        raise ValueError("Unknown unit type %s!" % unit_type)

    if dropout > 0.0:
        single_cell = tf.contrib.rnn.DropoutWrapper(cell=single_cell, input_keep_prob=1.0-dropout)

    if residual_connection:
        single_cell = tf.contrib.rnn.ResidualWrapper(single_cell)

    if device_str:
        single_cell = tf.contrib.rnn.DeviceWrapper(single_cell, device_str)

    return single_cell


def _cell_list(unit_type, num_units, num_layers, num_residual_layers,
               forget_bias, dropout, mode, num_gpus, base_gpu=0):
    """Create a list of RNN cells. Multi-GPU."""

    cell_list = []
    for i in range(num_layers):
        dropout = dropout if mode==tf.contrib.learn.ModeKeys.TRAIN else 0.0
        single_cell = _single_cell(
            unit_type=unit_type,
           num_units=num_units,
           forget_bias=forget_bias,
           dropout=dropout,
           residual_connection=(i >= num_layers-num_residual_layers),
           device_str=get_device_str(base_gpu+i, num_gpus)
        )
        cell_list.append(single_cell)
    return cell_list


def create_rnn_cel(unit_type, num_units, num_layers, num_residual_layers,
                    forget_bias, dropout, mode, num_gpus, base_gpu=0):
    """Create multi-layer RNN cell.
    """
    cell_list = _cell_list(unit_type, num_units, num_layers, num_residual_layers,
                           forget_bias, dropout, mode, num_gpus, base_gpu)
    if(len(cell_list) == 1): # single layer
        return cell_list[0]
    else:   # multi layer
        return tf.contrib.rnn.MultiRNNCell(cell_list)


def create_emb_for_encoder_and_decoder(share_vocab,
                                       src_vocab_size,
                                       tgt_vocab_size,
                                       src_embed_size,
                                       tgt_embed_size,
                                       dtype=tf.float32,
                                       scope=None):
    with tf.variable_scope(scope or "embeddings", dtype=dtype) as scope:
        with tf.variable_scope("encoder"):
            embedding_encoder = tf.get_variable(
                name="embedding_encoder",
                shape=(src_vocab_size, src_embed_size),
                dtype=dtype)
        with tf.variable_scope("decoder"):
            embedding_decoder = tf.get_variable(
                name="embedding_decoder",
                shape=(tgt_vocab_size, tgt_embed_size),
                dtype=dtype)
        return embedding_encoder, embedding_decoder


class BaseModel:
    """basic seq2seq model."""

    def __init__(self, hparams, mode, iterator,
                 source_vocab_table, target_vocab_table,
                 reverse_target_vocab_table=None,
                 scope=None):
        assert isinstance(iterator, iterator_utils.BatchedInput)
        self.iterator = iterator
        self.mode = mode
        self.src_vocab_table = source_vocab_table
        self.tgt_vocab_table = target_vocab_table

        self.hparams = hparams
        self.scope = scope
        self.src_vocab_size = hparams.src_vocab_size
        self.tgt_vocab_size = hparams.tgt_vocab_size
        self.num_layers = hparams.num_layers
        self.num_gpus = hparams.num_gpus
        self.time_major = hparams.time_major

        # set initializer
        initializer = tf.random_uniform_initializer(
            -hparams.init_weight,
            hparams.init_weight,
            seed=hparams.random_seed)
        tf.get_variable_scope().set_initializer(initializer)

        # Embeddings
        self.init_embeddings()
        # ?? bug
        self.batch_size = tf.size(self.iterator.source_sequence_length)

        # projection
        with tf.variable_scope(scope or "build_network"):
            with tf.variable_scope("decoder/output_projection"):
                self.output_layer = layers_core.Dense(
                    hparams.tgt_vocab_size, use_bias=False, name="output_projection")

        # train graph
        res = self.build_graph()


    def build_graph(self):
        "build encoder and decoder."


    def init_embeddings(self):
        "init embedding vars."
        hparams = self.hparams
        scope = self.scope
        self.embedding_encoder, self.embedding_decoder = (
            create_emb_for_encoder_and_decoder(
                share_vocab=hparams.share_vocab,
                src_vocab_size=self.src_vocab_size,
                tgt_vocab_size=self.tgt_vocab_size,
                src_embed_size=hparams.num_units,
                tgt_embed_size=hparams.num_units,
                scope=scope
            ))
