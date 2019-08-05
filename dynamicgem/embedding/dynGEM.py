#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from keras.layers import Input, Lambda, merge, Subtract
from keras.models import Model, model_from_json
from keras.optimizers import SGD, Adam
from keras import backend as KBack
import tensorflow as tf
from time import time
import networkx as nx

from dynamicgem.embedding.dynamic_graph_embedding import DynamicGraphEmbedding
from dynamicgem.utils import graph_util, plot_util
from dynamicgem.evaluation import visualize_embedding as viz
from dynamicgem.utils.sdne_utils import *
from dynamicgem.graph_generation import SBM_graph
from dynamicgem.evaluation import evaluate_graph_reconstruction as gr

class DynGEM(DynamicGraphEmbedding):
    """Structural Deep Network Embedding
    
    DynSDNE (also DynGEM) perfomr the dynamic network embedding
    while utilizing Structural Deep Network Embedding (SDNE) with dynamically evolving graphs as input.

    Args:
        d (int): dimension of the embedding
        beta (float): penalty parameter in matrix B of 2nd order objective
        n_prev_graphs (int): Lookback (number of previous graphs to be considered) for the dynamic graph embedding
        nu1 (float): L1-reg hyperparameter
        nu2 (float): L2-reg hyperparameter
        K (float): number of hidden layers in encoder/decoder
        rho (float): bounding ratio for number of units in consecutive layers (< 1)
        n_aeunits  (list) = List of embedding dimension for auto encoder layers
        n_lstmunits= List of embedding dimension for lstm layers
        n_iter (int): number of sgd iterations for first embedding (const)
        xeta (float): sgd step size parameter
        n_batch (int): minibatch size for SGD
        modelfile (str): Files containing previous encoder and decoder models
        weightfile (str): Files containing previous encoder and decoder weights
    
    Examples:
        >>> from dynamicgem.embedding.dynSDNE import DynSDNE
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
        >>> graphs = [g[0] for g in dynamic_sbm_series]
        >>> embedding = DynSDNE(d=128, beta=5, alpha=0, nu1=1e-6, nu2=1e-6, K=3,
                     n_units=[500, 300], n_iter=20, xeta=0.01,
                     n_batch=500,
                     modelfile=['./intermediate/enc_model.json',
                                './intermediate/dec_model.json'],
                     weightfile=['./intermediate/enc_weights.hdf5',
                                 './intermediate/dec_weights.hdf5'])
        >>> embedding.learn_embedding(graph=graphs._graph, edge_f=None,
                              is_weighted=True, no_python=True)
    """
    def __init__(self, *hyper_dict, **kwargs):
        """ Initialize the DynSDNE class"""
        hyper_params = {
            'method_name': 'dynSDNE',
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

    def learn_embedding(self, graph=None, edge_f=None,
                        is_weighted=False, no_python=False):
        """Learns the embedding of the nodes.
           
           Attributes:
               graph (Object): Networkx Graph Object

            Returns:
                List: Node embeddings and time taken by the algorithm
        """

        ###################################
        # TensorFlow wizardry
        config = tf.ConfigProto()
         
        # Don't pre-allocate memory; allocate as-needed
        config.gpu_options.allow_growth = True
         
        # Only allow a total of half the GPU memory to be allocated
        config.gpu_options.per_process_gpu_memory_fraction = 0.25
         
        # Create a session with the above options specified.
        KBack.tensorflow_backend.set_session(tf.Session(config=config))
        ###################################

        if not graph and not edge_f:
            raise Exception('graph/edge_f needed')
        if not graph:
            graph = graph_util.loadGraphFromEdgeListTxt(edge_f)
        S = nx.to_scipy_sparse_matrix(graph)
        t1 = time()
        S = (S + S.T) / 2
        self._node_num = graph.number_of_nodes()

        # Generate encoder, decoder and autoencoder
        self._num_iter = self._n_iter
        # If cannot use previous step information, initialize new models
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
        x_in = Input(shape=(2 * self._node_num,), name='x_in')
        x1 = Lambda(
            lambda x: x[:, 0:self._node_num],
            output_shape=(self._node_num,)
        )(x_in)
        x2 = Lambda(
            lambda x: x[:, self._node_num:2 * self._node_num],
            output_shape=(self._node_num,)
        )(x_in)
        # Process inputs
        [x_hat1, y1] = self._autoencoder(x1)
        [x_hat2, y2] = self._autoencoder(x2)
        # Outputs
        x_diff1 = Subtract()([x_hat1, x1])
        x_diff2 = Subtract()([x_hat2, x2]) 
        y_diff =  Subtract()([y2, y1]) 

        # Objectives
        def weighted_mse_x(y_true, y_pred):
            ''' Hack: This fn doesn't accept additional arguments.
                      We use y_true to pass them.
                y_pred: Contains x_hat - x
                y_true: Contains [b, deg]
            '''
            return KBack.sum(
                KBack.square(y_pred * y_true[:, 0:self._node_num]),
                axis=-1) / y_true[:, self._node_num]

        def weighted_mse_y(y_true, y_pred):
            ''' Hack: This fn doesn't accept additional arguments.
                      We use y_true to pass them.
            y_pred: Contains y2 - y1
            y_true: Contains s12
            '''
            min_batch_size = KBack.shape(y_true)[0]
            return KBack.reshape(
                KBack.sum(KBack.square(y_pred), axis=-1),
                [min_batch_size, 1]
            ) * y_true

        # Model
        self._model = Model(input=x_in, output=[x_diff1, x_diff2, y_diff])
        sgd = SGD(lr=self._xeta, decay=1e-5, momentum=0.99, nesterov=True)
        # adam = Adam(lr=self._xeta, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self._model.compile(
            optimizer=sgd,
            loss=[weighted_mse_x, weighted_mse_x, weighted_mse_y],
            loss_weights=[1, 1, self._alpha]
        )

        self._model.fit_generator(
            generator=batch_generator_sdne(S, self._beta, self._n_batch, True),
            nb_epoch=self._num_iter,
            samples_per_epoch=S.nonzero()[0].shape[0] // self._n_batch,
            verbose=1
        )
        # Get embedding for all points
        self._Y = model_batch_predictor(self._autoencoder, S, self._n_batch)
        t2 = time()
        # Save the autoencoder and its weights
        if self._weightfile is not None:
            saveweights(self._encoder, self._weightfile[0])
            saveweights(self._decoder, self._weightfile[1])
        if self._modelfile is not None:
            savemodel(self._encoder, self._modelfile[0])
            savemodel(self._decoder, self._modelfile[1])
        if self._savefilesuffix is not None:
            saveweights(
                self._encoder,
                'encoder_weights_' + self._savefilesuffix + '.hdf5'
            )
            saveweights(
                self._decoder,
                'decoder_weights_' + self._savefilesuffix + '.hdf5'
            )
            savemodel(
                self._encoder,
                'encoder_model_' + self._savefilesuffix + '.json'
            )
            savemodel(
                self._decoder,
                'decoder_model_' + self._savefilesuffix + '.json'
            )
            # Save the embedding
            np.savetxt('embedding_' + self._savefilesuffix + '.txt', self._Y)
        return self._Y, (t2 - t1)

    def get_embedding(self, filesuffix=None):
        """Function to return the embeddings"""
        return self._Y if filesuffix is None else np.loadtxt(
            'embedding_' + filesuffix + '.txt'
        )

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
        S_hat = self.get_reconst_from_embed(embed, node_l, filesuffix)
        return graphify(S_hat)

    def get_reconst_from_embed(self, embed, node_l=None, filesuffix=None):
        """Function to reconstruct the graph from the embedding.
           
            Attributes:
              node_l (int): node for which the adjacency list will be created.
              embed (Matrix): Embedding values of all the nodes.
              filesuffix (str): File suffix to be used to load the embedding.

            Returns:
                List: REconstructed graph for the given nodes.
        """
        if filesuffix is None:
            if node_l is not None:
                return self._decoder.predict(
                    embed,
                    batch_size=self._n_batch)[:, node_l]
            else:
                return self._decoder.predict(embed, batch_size=self._n_batch)
        else:
            try:
                decoder = model_from_json(
                    open('decoder_model_' + filesuffix + '.json').read()
                )
            except:
                print('Error reading file: {0}. Cannot load previous model'.format('decoder_model_'+filesuffix+'.json'))
                exit()
            try:
                decoder.load_weights('decoder_weights_' + filesuffix + '.hdf5')
            except:
                print('Error reading file: {0}. Cannot load previous weights'.format('decoder_weights_'+filesuffix+'.hdf5'))
                exit()
            if node_l is not None:
                return decoder.predict(embed, batch_size=self._n_batch)[:, node_l]
            else:
                return decoder.predict(embed, batch_size=self._n_batch)

    
