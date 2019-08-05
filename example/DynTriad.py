'''
=============================
Example Code for DynamicTRIAD
=============================
'''

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

import sys
import tensorflow as tf
import argparse
import operator
import time
import os
import importlib
import pdb
import random
import networkx as nx

from dynamicgem.embedding.dynamicTriad import dynamicTriad
from dynamicgem.utils import graph_util, plot_util, dataprep_util
from dynamicgem.evaluation import visualize_embedding as viz
from dynamicgem.utils.sdne_utils import *
from dynamicgem.graph_generation import dynamic_SBM_graph
from dynamicgem.utils.dynamictriad_utils import *
import dynamicgem.utils.dynamictriad_utils.dataset.dataset_utils as du
import dynamicgem.utils.dynamictriad_utils.algorithm.embutils as eu
from dynamicgem.evaluation import evaluate_link_prediction as lp


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Learns static node embeddings')
    parser.add_argument('-t', '--testDataType',
                        default='sbm_cd',
                        type=str,
                        help='Type of data to test the code')
    parser.add_argument('-nm', '--nodemigration',
                        default=2,
                        type=int,
                        help='number of nodes to migrate')
    parser.add_argument('-iter', '--niters',
                        type=int,
                        help="number of optimization iterations",
                        default=2)
    parser.add_argument('-m', '--starttime',
                        type=str,
                        help=argparse.SUPPRESS,
                        default=0)
    parser.add_argument('-d', '--datafile',
                        type=str,
                        help='input directory name')
    parser.add_argument('-b', '--batchsize',
                        type=int,
                        help="batchsize for training",
                        default=100)
    parser.add_argument('-n', '--nsteps',
                        type=int,
                        help="number of time steps",
                        default=4)
    parser.add_argument('-K', '--embdim',
                        type=int,
                        help="number of embedding dimensions",
                        default=32)
    parser.add_argument('-l', '--stepsize',
                        type=int,
                        help="size of of a time steps",
                        default=1)
    parser.add_argument('-s', '--stepstride',
                        type=int,
                        help="interval between two time steps",
                        default=1)
    parser.add_argument('-o', '--outdir',
                        type=str,
                        default='./output',
                        help="output directory name")
    parser.add_argument('-rd', '--resultdir',
                        type=str,
                        default='./results_link_all',
                        help="result directory name")
    parser.add_argument('--lr',
                        type=float,
                        help="initial learning rate",
                        default=0.1)
    parser.add_argument('--beta-smooth',
                        type=float,
                        default=0.1,
                        help="coefficients for smooth component")
    parser.add_argument('--beta-triad',
                        type=float,
                        default=0.1,
                        help="coefficients for triad component")
    parser.add_argument('--negdup',
                        type=int,
                        help="neg/pos ratio during sampling",
                        default=1)
    parser.add_argument('--datasetmod',
                        type=str,
                        default='dynamicgem.utils.dynamictriad_utils.dataset.adjlist',
                        help='module name for dataset loading',
                        )
    parser.add_argument('--validation',
                        type=str,
                        default='link_reconstruction',
                        help=', '.join(list(sorted(set(du.TestSampler.tasks) & set(eu.Validator.tasks)))))
    parser.add_argument('-te', '--test',
                        type=str,
                        nargs='+',
                        default='link_predict',
                        help='type of test, (node_classify, node_predict, link_classify, link_predict, '
                             'changed_link_classify, changed_link_predict, all)')
    parser.add_argument('--classifier',
                        type=str,
                        default='lr',
                        help='lr, svm')
    parser.add_argument('--repeat',
                        type=int,
                        default=1,
                        help='number of times to repeat experiment')
    parser.add_argument('-sm', '--samples',
                        default=5000,
                        type=int,
                        help='samples for test data')
    args = parser.parse_args()
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    args.embdir = args.outdir + '/dynTriad/' + args.testDataType
    args.cachefn = '/tmp/' + args.testDataType
    args.beta = [args.beta_smooth, args.beta_triad]
    # some fixed arguments in published code
    args.pretrain_size = args.nsteps
    args.trainmod = 'dynamicgem.utils.dynamictriad_utils.algorithm.dynamic_triad'
    args.sampling_args = {}
    args.debug = False
    args.scale = 1

    if args.validation not in du.TestSampler.tasks:
        raise NotImplementedError("Validation task {} not supported in TestSampler".format(args.validation))
    if args.validation not in eu.Validator.tasks:
        raise NotImplementedError("Validation task {} not supported in Validator".format(args.validation))

    print("running with options: ", args.__dict__)

    epochs = args.niters
    length = args.nsteps

    if args.testDataType == 'sbm_cd':
        node_num = 200
        community_num = 2
        node_change_num = args.nodemigration
        dynamic_sbm_series = dynamic_SBM_graph.get_community_diminish_series_v2(node_num,
                                                                                community_num,
                                                                                length,
                                                                                1,
                                                                                node_change_num)
        graphs = [g[0] for g in dynamic_sbm_series]

        datafile = dataprep_util.prep_input_dynTriad(graphs, length, args.testDataType)

        embedding = dynamicTriad(niters=args.niters,
                                 starttime=args.starttime,
                                 datafile=datafile,
                                 batchsize=args.batchsize,
                                 nsteps=args.nsteps,
                                 embdim=args.embdim,
                                 stepsize=args.stepsize,
                                 stepstride=args.stepstride,
                                 outdir=args.outdir,
                                 cachefn=args.cachefn,
                                 lr=args.lr,
                                 beta=args.beta,
                                 negdup=args.negdup,
                                 datasetmod=args.datasetmod,
                                 trainmod=args.trainmod,
                                 pretrain_size=args.pretrain_size,
                                 sampling_args=args.sampling_args,
                                 validation=args.validation,
                                 datatype=args.testDataType,
                                 scale=args.scale,
                                 classifier=args.classifier,
                                 debug=args.debug,
                                 test=args.test,
                                 repeat=args.repeat,
                                 resultdir=args.resultdir,
                                 testDataType=args.testDataType,
                                 clname='lr',
                                 node_num=node_num )

        embedding.learn_embedding()
        embedding.get_embedding()
        # embedding.plotresults(dynamic_sbm_series)

        outdir = args.resultdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + args.testDataType
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + 'dynTRIAD'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        lp.expstaticLP_TRIAD(dynamic_sbm_series,
                             graphs,
                             embedding,
                             1,
                             outdir + '/',
                             'nm' + str(args.nodemigration) + '_l' + str(args.nsteps) + '_emb' + str(args.embdim),
                             )


    elif args.testDataType == 'academic':
        print("datatype:", args.testDataType)

        sample = args.samples
        if not os.path.exists('./test_data/academic/pickle'):
            os.mkdir('./test_data/academic/pickle')
            graphs, length = dataprep_util.get_graph_academic('./test_data/academic/adjlist')
            for i in range(length):
                nx.write_gpickle(graphs[i], './test_data/academic/pickle/' + str(i))
        else:
            length = len(os.listdir('./test_data/academic/pickle'))
            graphs = []
            for i in range(length):
                graphs.append(nx.read_gpickle('./test_data/academic/pickle/' + str(i)))

        G_cen = nx.degree_centrality(graphs[29])  # graph 29 in academia has highest number of edges
        G_cen = sorted(G_cen.items(), key=operator.itemgetter(1), reverse=True)
        node_l = []
        i = 0
        while i < sample:
            node_l.append(G_cen[i][0])
            i += 1
        # pdb.set_trace()
        # node_l = np.random.choice(range(graphs[29].number_of_nodes()), 5000, replace=False)
        # print(node_l)
        for i in range(length):
            graphs[i] = graph_util.sample_graph_nodes(graphs[i], node_l)
        # pdb.set_trace()
        graphs = graphs[-args.nsteps:]
        datafile = dataprep_util.prep_input_dynTriad(graphs, args.nsteps, args.testDataType)

        embedding = dynamicTriad(niters=args.niters,
                                 starttime=args.starttime,
                                 datafile=datafile,
                                 batchsize=args.batchsize,
                                 nsteps=args.nsteps,
                                 embdim=args.embdim,
                                 stepsize=args.stepsize,
                                 stepstride=args.stepstride,
                                 outdir=args.outdir,
                                 cachefn=args.cachefn,
                                 lr=args.lr,
                                 beta=args.beta,
                                 negdup=args.negdup,
                                 datasetmod=args.datasetmod,
                                 trainmod=args.trainmod,
                                 pretrain_size=args.pretrain_size,
                                 sampling_args=args.sampling_args,
                                 validation=args.validation,
                                 datatype=args.testDataType,
                                 scale=args.scale,
                                 classifier=args.classifier,
                                 debug=args.debug,
                                 test=args.test,
                                 repeat=args.repeat,
                                 resultdir=args.resultdir,
                                 testDataType=args.testDataType,
                                 clname='lr',
                                 node_num=sample

                                 )
        embedding.learn_embedding()
        embedding.get_embedding()

        outdir = args.resultdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + args.testDataType
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        outdir = outdir + '/dynTriad'
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        lp.expstaticLP_TRIAD(None,
                             graphs,
                             embedding,
                             1,
                             outdir + '/',
                             'l' + str(args.nsteps) + '_emb' + str(args.embdim) + '_samples' + str(sample),
                             n_sample_nodes=sample
                             )


    elif args.testDataType == 'hep':
        print("datatype:", args.testDataType)

        if not os.path.exists('./test_data/hep/pickle'):
            os.mkdir('./test_data/hep/pickle')
            files = [file for file in os.listdir('./test_data/hep/hep-th') if '.gpickle' in file]
            length = len(files)
            graphs = []
            for i in range(length):
                G = nx.read_gpickle('./test_data/hep/hep-th/month_' + str(i + 1) + '_graph.gpickle')

                graphs.append(G)
            total_nodes = graphs[-1].number_of_nodes()

            for i in range(length):
                for j in range(total_nodes):
                    if j not in graphs[i].nodes():
                        graphs[i].add_node(j)

            for i in range(length):
                nx.write_gpickle(graphs[i], './test_data/hep/pickle/' + str(i))
        else:
            length = len(os.listdir('./test_data/hep/pickle'))
            graphs = []
            for i in range(length):
                graphs.append(nx.read_gpickle('./test_data/hep/pickle/' + str(i)))

        # pdb.set_trace()            
        sample = args.samples
        G_cen = nx.degree_centrality(graphs[-1])  # graph 29 in academia has highest number of edges
        G_cen = sorted(G_cen.items(), key=operator.itemgetter(1), reverse=True)
        node_l = []
        i = 0
        while i < sample:
            node_l.append(G_cen[i][0])
            i += 1
        for i in range(length):
            graphs[i] = graph_util.sample_graph_nodes(graphs[i], node_l)

        graphs = graphs[-args.nsteps:]
        datafile = dataprep_util.prep_input_dynTriad(graphs, args.nsteps, args.testDataType)

        embedding = dynamicTriad(niters=args.niters,
                                 starttime=args.starttime,
                                 datafile=datafile,
                                 batchsize=args.batchsize,
                                 nsteps=args.nsteps,
                                 embdim=args.embdim,
                                 stepsize=args.stepsize,
                                 stepstride=args.stepstride,
                                 outdir=args.outdir,
                                 cachefn=args.cachefn,
                                 lr=args.lr,
                                 beta=args.beta,
                                 negdup=args.negdup,
                                 datasetmod=args.datasetmod,
                                 trainmod=args.trainmod,
                                 pretrain_size=args.pretrain_size,
                                 sampling_args=args.sampling_args,
                                 validation=args.validation,
                                 datatype=args.testDataType,
                                 scale=args.scale,
                                 classifier=args.classifier,
                                 debug=args.debug,
                                 test=args.test,
                                 repeat=args.repeat,
                                 resultdir=args.resultdir,
                                 testDataType=args.testDataType,
                                 clname='lr',
                                 node_num=sample

                                 )
        embedding.learn_embedding()
        embedding.get_embedding()

        outdir = args.resultdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + args.testDataType
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        outdir = outdir + '/dynTriad'
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        lp.expstaticLP_TRIAD(None,
                             graphs,
                             embedding,
                             1,
                             outdir + '/',
                             'l' + str(args.nsteps) + '_emb' + str(args.embdim) + '_samples' + str(sample),
                             n_sample_nodes=sample
                             )


    elif args.testDataType == 'AS':
        print("datatype:", args.testDataType)

        files = [file for file in os.listdir('./test_data/AS/as-733') if '.gpickle' in file]
        length = len(files)
        graphs = []

        for i in range(length):
            G = nx.read_gpickle('./test_data/AS/as-733/month_' + str(i + 1) + '_graph.gpickle')
            graphs.append(G)

        sample = args.samples
        G_cen = nx.degree_centrality(graphs[-1])  # graph 29 in academia has highest number of edges
        G_cen = sorted(G_cen.items(), key=operator.itemgetter(1), reverse=True)
        node_l = []
        i = 0
        while i < sample:
            node_l.append(G_cen[i][0])
            i += 1
        for i in range(length):
            graphs[i] = graph_util.sample_graph_nodes(graphs[i], node_l)

        graphs = graphs[-args.nsteps:]
        datafile = dataprep_util.prep_input_dynTriad(graphs, args.nsteps, args.testDataType)

        embedding = dynamicTriad(niters=args.niters,
                                 starttime=args.starttime,
                                 datafile=datafile,
                                 batchsize=args.batchsize,
                                 nsteps=args.nsteps,
                                 embdim=args.embdim,
                                 stepsize=args.stepsize,
                                 stepstride=args.stepstride,
                                 outdir=args.outdir,
                                 cachefn=args.cachefn,
                                 lr=args.lr,
                                 beta=args.beta,
                                 negdup=args.negdup,
                                 datasetmod=args.datasetmod,
                                 trainmod=args.trainmod,
                                 pretrain_size=args.pretrain_size,
                                 sampling_args=args.sampling_args,
                                 validation=args.validation,
                                 datatype=args.testDataType,
                                 scale=args.scale,
                                 classifier=args.classifier,
                                 debug=args.debug,
                                 test=args.test,
                                 repeat=args.repeat,
                                 resultdir=args.resultdir,
                                 testDataType=args.testDataType,
                                 clname='lr',
                                 node_num=sample

                                 )

        embedding.learn_embedding()
        embedding.get_embedding()

        outdir = args.resultdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + args.testDataType
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        outdir = outdir + '/dynTriad'
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        lp.expstaticLP_TRIAD(None,
                             graphs,
                             embedding,
                             1,
                             outdir + '/',
                             'l' + str(args.nsteps) + '_emb' + str(args.embdim) + '_samples' + str(sample),
                             n_sample_nodes=sample
                             )

    elif args.testDataType == 'enron':
        print("datatype:", args.testDataType)

        files = [file for file in os.listdir('./test_data/enron') if 'month' in file]
        length = len(files)
        graphsall = []

        for i in range(length):
            G = nx.read_gpickle('./test_data/enron/month_' + str(i + 1) + '_graph.gpickle')
            graphsall.append(G)

        sample = graphsall[0].number_of_nodes()
        graphs = graphsall[-args.nsteps:]
        datafile = dataprep_util.prep_input_dynTriad(graphs, args.nsteps, args.testDataType)
        # pdb.set_trace()

        embedding = dynamicTriad(niters=args.niters,
                                 starttime=args.starttime,
                                 datafile=datafile,
                                 batchsize=100,
                                 nsteps=args.nsteps,
                                 embdim=args.embdim,
                                 stepsize=args.stepsize,
                                 stepstride=args.stepstride,
                                 outdir=args.outdir,
                                 cachefn=args.cachefn,
                                 lr=args.lr,
                                 beta=args.beta,
                                 negdup=args.negdup,
                                 datasetmod=args.datasetmod,
                                 trainmod=args.trainmod,
                                 pretrain_size=args.pretrain_size,
                                 sampling_args=args.sampling_args,
                                 validation=args.validation,
                                 datatype=args.testDataType,
                                 scale=args.scale,
                                 classifier=args.classifier,
                                 debug=args.debug,
                                 test=args.test,
                                 repeat=args.repeat,
                                 resultdir=args.resultdir,
                                 testDataType=args.testDataType,
                                 clname='lr',
                                 node_num=sample

                                 )

        embedding.learn_embedding()
        embedding.get_embedding()

        outdir = args.resultdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + args.testDataType
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        outdir = outdir + '/dynTriad'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        lp.expstaticLP_TRIAD(None,
                             graphs,
                             embedding,
                             1,
                             outdir + '/',
                             'l' + str(args.nsteps) + '_emb' + str(args.embdim) + '_samples' + str(sample),
                             n_sample_nodes=sample
                             )
