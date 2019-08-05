#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input, Subtract
from keras.models import Model, model_from_json
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard
from keras import callbacks
from keras import backend as KBack
import tensorflow as tf
from time import time

from dynamicgem.embedding.dynamic_graph_embedding import DynamicGraphEmbedding
from dynamicgem.utils.dnn_utils import *

class DynAE(DynamicGraphEmbedding):
    """Dynamic AutoEncoder
    
    DynAE is a dynamic graph embedding algorithm which also takes different
    timestep graph with varying lookback to be considered in embedding the nodes 
    using the autoencoder. 

    Args:
        d (int): dimension of the embedding
        beta (float): penalty parameter in matrix B of 2nd order objective
        n_prev_graphs (int): Lookback (number of previous graphs to be considered) for the dynamic graph embedding
        nu1 (float): L1-reg hyperparameter
        nu2 (float): L2-reg hyperparameter
        K (float): number of hidden layers in encoder/decoder
        rho (float): bounding ratio for number of units in consecutive layers (< 1)
        n_units (list) : vector of length K-1 containing #units in hidden layers of encoder/decoder, not including the units in the embedding layer
        n_iter (int): number of sgd iterations for first embedding (const)
        xeta (float): sgd step size parameter
        n_batch (int): minibatch size for SGD
        modelfile (str): Files containing previous encoder and decoder models
        weightfile (str): Files containing previous encoder and decoder weights
    
    Examples:
        >>> from dynamicgem.embedding.dynAE import DynAE
        >>> from dynamicgem.graph_generation import dynamic_SBM_graph
        >>> node_num = 1000
        >>> community_num = 2
        >>> node_change_num = 10
        >>> length =5
        >>> dynamic_sbm_series = dynamic_SBM_graph.get_community_diminish_series_v2(node_num,
                                                                                community_num,
                                                                                length,
                                                                                1,
                                                                                node_change_num)
        >>> embedding = DynAE(d=dim_emb,
                        beta=5,
                        n_prev_graphs=lookback,
                        nu1=1e-6,
                        nu2=1e-6,
                        n_units=[500, 300, ],
                        rho=0.3,
                        n_iter=epochs,
                        xeta=args.learningrate,
                        n_batch=args.batch,
                        modelfile=['./intermediate/enc_model.json', './intermediate/dec_model.json'],
                        weightfile=['./intermediate/enc_weights.hdf5', './intermediate/dec_weights.hdf5'],
                        savefilesuffix="testing")

        >>> graphs = [g[0] for g in dynamic_sbm_series]
        >>> embs = []

        >>> for temp_var in range(length):
        >>>         emb, _ = embedding.learn_embeddings(graphs[temp_var])
        >>>         embs.append(emb)
    """

    def __init__(self, d, *hyper_dict, **kwargs):
        self._d = d
        hyper_params = {
            'method_name': 'dynAE',
            'actfn': 'relu',
            'modelfile': None,
            'weightfile': None,
            'savefilesuffix': None
        }
        hyper_params.update(kwargs)
        for key in hyper_params.keys():
            self.__setattr__('_%s' % key, hyper_params[key])
        for dictionary in hyper_dict:
            for key in dictionary:
                self.__setattr__('_%s' % key, dictionary[key])

    def get_method_name(self):
        """Function to return the method name.
            
           Returns:
                String: Name of the method.
        """
        return self._method_name

    def get_method_summary(self):
        """Function to return the summary of the algorithm. 
           
           Returns:
                String: Method summary
        """
        return '%s_%d' % (self._method_name, self._d)

    def learn_embeddings(self, graphs):
        """Learns the embedding of the nodes.
           
           Attributes:
               graph (Object): Networkx Graph Object

            Returns:
                List: Node embeddings and time taken by the algorithm
        """
        self._node_num = graphs[0].number_of_nodes()
        t1 = time()
        ###################################
        # TensorFlow wizardry
        config = tf.ConfigProto()
        # Don't pre-allocate memory; allocate as-needed
        config.gpu_options.allow_growth = True
        # Only allow a total of half the GPU memory to be allocated
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        # Create a session with the above options specified.
        KBack.tensorflow_backend.set_session(tf.Session(config=config))
        ###################################
        # Generate encoder, decoder and autoencoder
        self._num_iter = self._n_iter
        self._encoder = get_encoder(self._node_num, self._d,
                                    self._n_units,
                                    self._nu1, self._nu2,
                                    self._actfn)
        self._decoder = get_decoder(self._node_num, self._d,
                                    self._n_units,
                                    self._nu1, self._nu2,
                                    self._actfn)
        self._autoencoder = get_autoencoder(self._encoder, self._decoder)

        # Initialize self._model
        # Input
        x_in = Input(shape=(self._node_num,), name='x_in')
        x_pred = Input(
            shape=(self._node_num,),
            name='x_pred'
        )
        # Process inputs
        [x_hat, y] = self._autoencoder(x_in)
        # Outputs
        x_diff = Subtract()([x_hat, x_in])

        # Objectives
        def weighted_mse_x(y_true, y_pred):
            """ Hack: This fn doesn't accept additional arguments.
                      We use y_true to pass them.
                y_pred: Contains x_hat - x_pred
                y_true: Contains b
            """
            return KBack.sum(
                KBack.square(y_pred * y_true[:, 0:self._node_num]),
                axis=-1
            )

        # Model
        self._model = Model(input=[x_in, x_pred], output=x_diff)
        sgd = SGD(lr=self._xeta, decay=1e-5, momentum=0.99, nesterov=True)
        adam = Adam(lr=self._xeta, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self._model.compile(optimizer=sgd, loss=weighted_mse_x)

        # tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        # pdb.set_trace()
        history = self._model.fit_generator(
            generator=batch_generator_dynae(
                graphs,
                self._beta,
                self._n_batch,
                self._n_prev_graphs,
                True
            ),
            nb_epoch=self._num_iter,
            samples_per_epoch=(
                                      graphs[0].number_of_nodes() * self._n_prev_graphs
                              ) // self._n_batch,
            verbose=1
            # callbacks=[tensorboard]
        )
        loss = history.history['loss']
        # Get embedding for all points
        if loss[0] == np.inf or np.isnan(loss[0]):
            print('Model diverged. Assigning random embeddings')
            self._Y = np.random.randn(self._node_num, self._d)
        else:
            self._Y, self._next_adj = model_batch_predictor_dynae(
                self._autoencoder,
                graphs[len(graphs) - self._n_prev_graphs:],
                self._n_batch
            )
        t2 = time()
        # Save the autoencoder and its weights
        if self._weightfile is not None:
            saveweights(self._encoder, self._weightfile[0])
            saveweights(self._decoder, self._weightfile[1])
        if self._modelfile is not None:
            savemodel(self._encoder, self._modelfile[0])
            savemodel(self._decoder, self._modelfile[1])
        if self._savefilesuffix is not None:
            saveweights(self._encoder,
                        'encoder_weights_' + self._savefilesuffix + '.hdf5')
            saveweights(self._decoder,
                        'decoder_weights_' + self._savefilesuffix + '.hdf5')
            savemodel(self._encoder,
                      'encoder_model_' + self._savefilesuffix + '.json')
            savemodel(self._decoder,
                      'decoder_model_' + self._savefilesuffix + '.json')
            # Save the embedding
            np.savetxt('embedding_' + self._savefilesuffix + '.txt',
                       self._Y)
            np.savetxt('next_pred_' + self._savefilesuffix + '.txt',
                       self._next_adj)
        return self._Y, (t2 - t1)

    def get_embeddings(self):
        """Function to return the embeddings"""
        return self._Y

    def get_edge_weight(self, i, j, embed=None, filesuffix=None):
        """Function to get edge weight.
           
            Attributes:
              i (int): source node for the edge.
              j (int): target node for the edge.
              embed (Matrix): Embedding values of all the nodes.
              filesuffix (str): File suffix to be used to load the embedding.

            Returns:
                Float: Weight of the given edge.
        """
        if embed is None:
            if filesuffix is None:
                embed = self._Y
            else:
                embed = np.loadtxt('embedding_' + filesuffix + '.txt')
        if i == j:
            return 0
        else:
            S_hat = self.get_reconst_from_embed(embed[(i, j), :], filesuffix)
            return (S_hat[i, j] + S_hat[j, i]) / 2

    def get_reconstructed_adj(self, embed=None, node_l=None, filesuffix=None):
        """Function to reconstruct the adjacency list for the given node.
           
            Attributes:
              node_l (int): node for which the adjacency list will be created.
              embed (Matrix): Embedding values of all the nodes.
              filesuffix (str): File suffix to be used to load the embedding.

            Returns:
                List : Adjacency list of the given node.
        """
        if embed is None:
            if filesuffix is None:
                embed = self._Y
            else:
                embed = np.loadtxt('embedding_' + filesuffix + '.txt')
        S_hat = self.get_reconst_from_embed(embed, filesuffix)
        return graphify(S_hat)

    def get_reconst_from_embed(self, embed, filesuffix=None):
        """Function to reconstruct the graph from the embedding.
           
            Attributes:
              node_l (int): node for which the adjacency list will be created.
              embed (Matrix): Embedding values of all the nodes.
              filesuffix (str): File suffix to be used to load the embedding.

            Returns:
                List: REconstructed graph for the given nodes.
        """
        if filesuffix is None:
            return self._decoder.predict(embed, batch_size=self._n_batch)
        else:
            try:
                decoder = model_from_json(open('./intermediate/decoder_model_' + filesuffix + '.json').read())
            except:
                print('Error reading file: {0}. Cannot load previous model'.format(
                    'decoder_model_' + filesuffix + '.json'))
                exit()
            try:
                decoder.load_weights('./intermediate/decoder_weights_' + filesuffix + '.hdf5')
            except:
                print('Error reading file: {0}. Cannot load previous weights'.format(
                    'decoder_weights_' + filesuffix + '.hdf5'))
                exit()
            return decoder.predict(embed, batch_size=self._n_batch)

    def predict_next_adj(self, node_l=None):
        """Function to predict the next adjacency for the given node.
           
            Attributes:
              node_l (int): node for which the adjacency list will be created.

            Returns:
                List: Reconstructed adjancey list.
        """
        if node_l is not None:
            # pdb.set_trace()
            return self._next_adj[node_l]
        else:
            return self._next_adj
