#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import matplotlib.pyplot as plt
import numpy as np
from time import time

from dynamicgem.embedding.static_graph_embedding import StaticGraphEmbedding
from dynamicgem.visualization import plot_dynamic_sbm_embedding
from dynamicgem.utils import graph_util, plot_util, dataprep_util
from dynamicgem.utils.sdne_utils import *
from dynamicgem.utils import timers_utils as tu



class incSVD(StaticGraphEmbedding):
    """Incremental Singular Value Decomposition
    
    Utilizes the incremental SVD decomposition to acquire 
    the embedding of the nodes. 

    Args:
        Args:
            K (int): dimension of the embedding
            theta (float): threshold for rerun
            datafile (str): location of the data file
            length (int) : total timesteps of the data
            nodemigraiton (int): number of nodes to migrate for sbm_cd datatype
            resultdir (str): directory to save the result
            datatype (str): sbm_cd, enron, academia, hep, AS

    Examples:
        >>> from dynamicgem.embedding.incrementalSVD import incSVD
        >>> from dynamicgem.graph_generation import dynamic_SBM_graph
        >>> node_num = 100
        >>> community_num = 2
        >>> node_change_num = 2
        >>> length =5
        >>> resultdir='./results_link_all'
        >>> dynamic_sbm_series = dynamic_SBM_graph.get_community_diminish_series_v2(node_num,
                                                                                community_num,
                                                                                length,
                                                                                1,
                                                                                node_change_num)
        >>> graphs = [g[0] for g in dynamic_sbm_series]

        >>> datafile = dataprep_util.prep_input_TIMERS(graphs, length, args.testDataType)
    
        >>> embedding = incSVD(K=16,
                           Theta=0.5,
                           datafile=datafile,
                           length=length,
                           nodemigration=node_change_num,
                           resultdir=resultdir,
                           datatype='sbm_cd'
                           )
        >>> outdir_tmp = './output'
        >>> if not os.path.exists(outdir_tmp):
        >>>     os.mkdir(outdir_tmp)
        >>> outdir_tmp = outdir_tmp + '/sbm_cd'
        >>> if not os.path.exists(outdir_tmp):
        >>>    os.mkdir(outdir_tmp)
        >>> if not os.path.exists(outdir_tmp + '/incrementalSVD'):
        >>>    os.mkdir(outdir_tmp + '/incrementalSVD')
        >>>  embedding.learn_embedding()
        >>>  outdir = resultdir
        >>>  if not os.path.exists(outdir):
        >>>     os.mkdir(outdir)
        >>>  outdir = outdir + '/' + args.testDataType
        >>>  if not os.path.exists(outdir):
        >>>      os.mkdir(outdir)
        >>>  embedding.get_embedding(outdir_tmp, 'incrementalSVD')
             # embedding.plotresults()  
        >>>  outdir1 = outdir + '/incrementalSVD'
        >>>  if not os.path.exists(outdir1):
        >>>      os.mkdir(outdir1)
        >>>  lp.expstaticLP_TIMERS(dynamic_sbm_series,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'nm' + str(args.nodemigration) + '_l' + str(length) + '_emb' + str(int(dim_emb)),
                                  )

    """
    def __init__(self,  *hyper_dict, **kwargs):

        hyper_params = {
            'method_name': 'TIMERS',
            'modelfile': None,
            'weightfile': None,
            'savefilesuffix': None

        }
        # pdb.set_trace()
        hyper_params.update(kwargs)
        for key in hyper_params.keys():
            self.__setattr__('_%s' % key, hyper_params[key])
        # for dictionary in hyper_dict:
        #     for key in dictionary:
        #         self.__setattr__('_%s' % key, dictionary[key])

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
        return '%s' % (self._method_name)

    def learn_embedding(self, graph=None):
        """Learns the embedding of the nodes.
           
           Attributes:
               graph (Object): Networkx Graph Object

            Returns:
                List: Node embeddings and time taken by the algorithm
        """
        timers = tu.incrementalSVD(self._datafile, self._K // 2, self._Theta, self._datatype)

    def plotresults(self, dynamic_sbm_series):
        """Function to plot the results"""
        plt.figure()
        plt.clf()
        plot_dynamic_sbm_embedding.plot_dynamic_sbm_embedding_v2(self._X[-5:-1], dynamic_sbm_series[-5:])

        resultdir = self._resultdir + '/' + self._datatype
        if not os.path.exists(resultdir):
            os.mkdir(resultdir)

        resultdir = resultdir + '/' + self._method
        if not os.path.exists(resultdir):
            os.mkdir(resultdir)

        #         plt.savefig('./'+resultdir+'/V_'+self._method+'_nm'+str(self._nodemigration)+'_l'+str(self._length)+'_theta'+str(theta)+'_emb'+str(self._K*2)+'.pdf',bbox_inches='tight',dpi=600)
        plt.show()
        # plt.close()  

    def get_embedding(self, outdir_tmp, method):
        """Function to return the embeddings"""
        self._outdir_tmp = outdir_tmp
        self._method = method
        self._X = dataprep_util.getemb_TIMERS(self._outdir_tmp, int(self._length), int(self._K // 2), self._method)
        return self._X

    def get_edge_weight(self, t, i, j):
        """Function to get edge weight.
           
            Attributes:
              i (int): source node for the edge.
              j (int): target node for the edge.
              embed (Matrix): Embedding values of all the nodes.
              filesuffix (str): File suffix to be used to load the embedding.

            Returns:
                Float: Weight of the given edge.
        """
        try:
            return np.dot(self._X[t][i, :int(self._K // 2)], self._X[t][j, int(self._K // 2):])
        except Exception as e:
            print(e.message, e.args)
            pdb.set_trace()

    def get_reconstructed_adj(self, t, X=None, node_l=None):
        """Function to reconstruct the adjacency list for the given node.
           
            Attributes:
              node_l (int): node for which the adjacency list will be created.
              embed (Matrix): Embedding values of all the nodes.
              filesuffix (str): File suffix to be used to load the embedding.

            Returns:
                List : Adjacency list of the given node.
        """
        if X is not None:
            node_num = X.shape[0]
            # self._X = X
        else:
            node_num = self._node_num
        adj_mtx_r = np.zeros((node_num, node_num))
        for v_i in range(node_num):
            for v_j in range(node_num):
                if v_i == v_j:
                    continue
                adj_mtx_r[v_i, v_j] = self.get_edge_weight(t, v_i, v_j)
        return adj_mtx_r

    def predict_next_adj(self, t, node_l=None):
        """Function to predict the next adjacency for the given node.
           
            Attributes:
              node_l (int): node for which the adjacency list will be created.

            Returns:
                List: Reconstructed adjancey list.
        """
        if node_l is not None:
            return self.get_reconstructed_adj(t, node_l)
        else:
            return self.get_reconstructed_adj(t)

