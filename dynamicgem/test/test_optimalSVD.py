#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for testing optimal SVD
"""
import os
from dynamicgem.utils import dataprep_util
from dynamicgem.embedding.optimalSVD import optimalSVD
from dynamicgem.graph_generation import dynamic_SBM_graph as sbm
from dynamicgem.evaluation import visualize_embedding as viz
from dynamicgem.evaluation import evaluate_link_prediction as lp
from time import time


def test_optimalSVD():
    # Parameters for Stochastic block model graph
    # Todal of 1000 nodes
    node_num = 100
    # Test with two communities
    community_num = 2
    # At each iteration migrate 10 nodes from one community to the another
    node_change_num = 2
    # Length of total time steps the graph will dynamically change
    length = 7
    # output directory for result
    outdir = './output'
    intr = './intermediate'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    if not os.path.exists(intr):
        os.mkdir(intr)
    testDataType = 'sbm_cd'
    # Generate the dynamic graph
    dynamic_sbm_series = list(sbm.get_community_diminish_series_v2(node_num,
                                                                   community_num,
                                                                   length,
                                                                   1,  # comminity ID to perturb
                                                                   node_change_num))
    graphs = [g[0] for g in dynamic_sbm_series]
    # parameters for the dynamic embedding
    # dimension of the embedding
    dim_emb = 8

    # TIMERS
    datafile = dataprep_util.prep_input_TIMERS(graphs, length, testDataType)
    embedding = optimalSVD(K=dim_emb,
                       Theta=0.5,
                       datafile=datafile,
                       length=length,
                       nodemigration=node_change_num,
                       resultdir=outdir,
                       datatype=testDataType)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outdir_tmp = outdir + '/sbm_cd'
    if not os.path.exists(outdir_tmp):
        os.mkdir(outdir_tmp)
    if not os.path.exists(outdir_tmp + '/optimalSVD'):
        os.mkdir(outdir_tmp + '/optimalSVD')

    t1 = time()
    embedding.learn_embedding()
    embedding.get_embedding(outdir_tmp, 'optimalSVD')
    print(embedding._method_name + ':\n\tTraining time: %f' % (time() - t1))
    lp.expstaticLP_TIMERS(dynamic_sbm_series,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir + '/',
                                  'nm' + str(node_change_num) + '_l' + str(length) + '_emb' + str(int(dim_emb)),
                                  )


if __name__ == '__main__':
    test_TIMERS()
