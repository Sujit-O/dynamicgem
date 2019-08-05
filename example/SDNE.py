#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

disp_avlbl = True
import os
if os.name == 'posix' and 'DISPLAY' not in os.environ:
    disp_avlbl = False
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import scipy.io as sio
from argparse import ArgumentParser
import time

from dynamicgem.embedding.sdne_dynamic import SDNE
from dynamicgem.utils import graph_util
from dynamicgem.visualization import plot_dynamic_sbm_embedding
from dynamicgem.graph_generation import dynamic_SBM_graph
from dynamicgem.evaluation import evaluate_link_prediction as lp
from dynamicgem.evaluation import visualize_embedding as viz

if __name__ == '__main__':
    parser = ArgumentParser(description='Learns node embeddings for a sequence of graph snapshots')
    parser.add_argument('-t', '--testDataType', default='sbm_cd', type=str,help='Type of data to test the code')
    args = parser.parse_args()

    if args.testDataType == 'sbm_rp':
        node_num = 10000
        community_num = 500
        node_change_num = 100
        length = 2
        dynamic_sbm_series = dynamic_SBM_graph.get_random_perturbation_series(node_num, community_num, length, node_change_num)
        dynamic_embedding = SDNE(d=100, beta=5, alpha=1e-5, nu1=1e-6, nu2=1e-6, K=3, n_units=[500, 300,], rho=0.3, n_iter=30, n_iter_subs=5, xeta=0.01, n_batch=500, modelfile=['./intermediate/enc_model.json', './intermediate/dec_model.json'], weightfile=['./intermediate/enc_weights.hdf5', './intermediate/dec_weights.hdf5'], node_frac=1, n_walks_per_node=10, len_rw=2)
        dynamic_embedding.learn_embeddings([g[0] for g in dynamic_sbm_series], False, subsample=False)
        plot_dynamic_sbm_embedding.plot_dynamic_sbm_embedding(dynamic_embedding.get_embeddings(), dynamic_sbm_series)
        plt.savefig('result/visualization_sdne_rp.png')
        plt.show()
    elif args.testDataType == 'sbm_cd':
        node_num = 100
        community_num = 2
        node_change_num = 2
        length = 5
        dynamic_sbm_series = dynamic_SBM_graph.get_community_diminish_series_v2(node_num, community_num, length, 1, node_change_num)
        dynamic_embedding = SDNE(d=16, beta=5, alpha=1e-5, nu1=1e-6, nu2=1e-6, K=3, n_units=[500, 300,], rho=0.3, n_iter=2, n_iter_subs=5, xeta=0.01, n_batch=50, modelfile=['./intermediate/enc_model.json', './intermediate/dec_model.json'], weightfile=['./intermediate/enc_weights.hdf5', './intermediate/dec_weights.hdf5'], node_frac=1, n_walks_per_node=10, len_rw=2)
        embs= []
        graphs = [g[0] for g in dynamic_sbm_series]
        for temp_var in range(length):
                emb= dynamic_embedding.learn_embedding(graphs[temp_var])
                embs.append(emb)
        viz.plot_static_sbm_embedding(embs[-4:], list(dynamic_sbm_series)[-4:])
    else:
        dynamic_graph_series = graph_util.loadRealGraphSeries('data/real/hep-th/month_', 1, 5)
        dynamic_embedding = SDNE(d=100, beta=2, alpha=1e-6, nu1=1e-5, nu2=1e-5, K=3, n_units=[400, 250,], rho=0.3, n_iter=100, n_iter_subs=30, xeta=0.001, n_batch=500, modelfile=['./intermediate/enc_model.json', './intermediate/dec_model.json'], weightfile=['./intermediate/enc_weights.hdf5', './intermediate/dec_weights.hdf5'])
        dynamic_embedding.learn_embeddings(dynamic_graph_series, False)