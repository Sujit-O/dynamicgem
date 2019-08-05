#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
disp_avlbl = True
if os.name == 'posix' and 'DISPLAY' not in os.environ:
    disp_avlbl = False
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import networkx as nx
import operator
from time import time
from argparse import ArgumentParser

from dynamicgem.embedding.TIMERS import TIMERS
from dynamicgem.utils import graph_util, plot_util, dataprep_util
from dynamicgem.evaluation import visualize_embedding as viz
from dynamicgem.utils.sdne_utils import *
from dynamicgem.evaluation import evaluate_graph_reconstruction as gr
from dynamicgem.evaluation import evaluate_link_prediction as lp
from dynamicgem.graph_generation import dynamic_SBM_graph


if __name__ == '__main__':

    parser = ArgumentParser(description='Learns static node embeddings')
    parser.add_argument('-t', '--testDataType',
                        default='sbm_cd',
                        type=str,
                        help='Type of data to test the code')
    parser.add_argument('-l', '--timelength',
                        default=5,
                        type=int,
                        help='Number of time series graph to generate')
    parser.add_argument('-nm', '--nodemigration',
                        default=5,
                        type=int,
                        help='number of nodes to migrate')
    parser.add_argument('-emb', '--embeddimension',
                        default=16,
                        type=float,
                        help='embedding dimension')
    parser.add_argument('-theta', '--theta',
                        default=0.5,  # 0.17
                        type=float,
                        help='a threshold for re-run SVD')
    parser.add_argument('-rdir', '--resultdir',
                        default='./results_link_all',  # 0.17
                        type=str,
                        help='directory for storing results')
    parser.add_argument('-sm', '--samples',
                        default=10,
                        type=int,
                        help='samples for test data')
    parser.add_argument('-exp', '--exp',
                        default='lp',
                        type=str,
                        help='experiments (lp, emb)')

    args = parser.parse_args()
    dim_emb = args.embeddimension
    length = args.timelength
    theta = args.theta
    sample = args.samples

    if args.testDataType == 'sbm_cd':
        node_num = 100
        community_num = 2
        node_change_num = args.nodemigration
        dynamic_sbm_series = dynamic_SBM_graph.get_community_diminish_series_v2(node_num,
                                                                                community_num,
                                                                                length,
                                                                                1,
                                                                                node_change_num)
        graphs = [g[0] for g in dynamic_sbm_series]

        datafile = dataprep_util.prep_input_TIMERS(graphs, length, args.testDataType)

        embedding = TIMERS(K=dim_emb,
                           Theta=theta,
                           datafile=datafile,
                           length=length,
                           nodemigration=args.nodemigration,
                           resultdir=args.resultdir,
                           datatype=args.testDataType
                           )
        outdir_tmp = './output'
        if not os.path.exists(outdir_tmp):
            os.mkdir(outdir_tmp)
        outdir_tmp = outdir_tmp + '/sbm_cd'
        if not os.path.exists(outdir_tmp):
            os.mkdir(outdir_tmp)
        if not os.path.exists(outdir_tmp + '/incrementalSVD'):
            os.mkdir(outdir_tmp + '/incrementalSVD')
        if not os.path.exists(outdir_tmp + '/rerunSVD'):
            os.mkdir(outdir_tmp + '/rerunSVD')
        if not os.path.exists(outdir_tmp + '/optimalSVD'):
            os.mkdir(outdir_tmp + '/optimalSVD')

        if args.exp == 'emb':
            print('plotting embedding not implemented!')

        if args.exp == 'lp':
            embedding.learn_embedding()

            outdir = args.resultdir
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            outdir = outdir + '/' + args.testDataType
            if not os.path.exists(outdir):
                os.mkdir(outdir)

            embedding.get_embedding(outdir_tmp, 'incrementalSVD')
            # embedding.plotresults()  
            outdir1 = outdir + '/incrementalSVD'
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(dynamic_sbm_series,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'nm' + str(args.nodemigration) + '_l' + str(length) + '_emb' + str(int(dim_emb)),
                                  )

            embedding.get_embedding(outdir_tmp, 'rerunSVD')
            outdir1 = outdir + '/rerunSVD'
            # embedding.plotresults()
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(dynamic_sbm_series,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'nm' + str(args.nodemigration) + '_l' + str(length) + '_emb' + str(int(dim_emb)),
                                  )

            embedding.get_embedding(outdir_tmp, 'optimalSVD')
            # embedding.plotresults()
            outdir1 = outdir + '/optimalSVD'
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(dynamic_sbm_series,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'nm' + str(args.nodemigration) + '_l' + str(length) + '_emb' + str(int(dim_emb)),
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
        graphs = graphs[-args.timelength:]

        datafile = dataprep_util.prep_input_TIMERS(graphs, args.timelength, args.testDataType)

        embedding = TIMERS(K=dim_emb,
                           Theta=theta,
                           datafile=datafile,
                           length=args.timelength,
                           nodemigration=args.nodemigration,
                           resultdir=args.resultdir,
                           datatype=args.testDataType
                           )
        outdir_tmp = './output'
        if not os.path.exists(outdir_tmp):
            os.mkdir(outdir_tmp)
        outdir_tmp = outdir_tmp + '/' + args.testDataType
        if not os.path.exists(outdir_tmp):
            os.mkdir(outdir_tmp)
        if not os.path.exists(outdir_tmp + '/incrementalSVD'):
            os.mkdir(outdir_tmp + '/incrementalSVD')
        if not os.path.exists(outdir_tmp + '/rerunSVD'):
            os.mkdir(outdir_tmp + '/rerunSVD')
        if not os.path.exists(outdir_tmp + '/optimalSVD'):
            os.mkdir(outdir_tmp + '/optimalSVD')

        if args.exp == 'emb':
            print('plotting embedding not implemented!')

        if args.exp == 'lp':
            embedding.learn_embedding()

            outdir = args.resultdir
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            outdir = outdir + '/' + args.testDataType
            if not os.path.exists(outdir):
                os.mkdir(outdir)

            embedding.get_embedding(outdir_tmp, 'incrementalSVD')
            # embedding.plotresults()  
            outdir1 = outdir + '/incrementalSVD'
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(None,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'l' + str(args.timelength) + '_emb' + str(int(dim_emb)) + '_samples' + str(sample),
                                  n_sample_nodes=sample
                                  )

            embedding.get_embedding(outdir_tmp, 'rerunSVD')
            outdir1 = outdir + '/rerunSVD'
            # embedding.plotresults()
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(None,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'l' + str(args.timelength) + '_emb' + str(int(dim_emb)) + '_samples' + str(sample),
                                  n_sample_nodes=sample
                                  )

            embedding.get_embedding(outdir_tmp, 'optimalSVD')
            # embedding.plotresults()
            outdir1 = outdir + '/optimalSVD'
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(None,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'l' + str(args.timelength) + '_emb' + str(int(dim_emb)) + '_samples' + str(sample),
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

        graphs = graphs[-args.timelength:]

        datafile = dataprep_util.prep_input_TIMERS(graphs, args.timelength, args.testDataType)

        embedding = TIMERS(K=dim_emb,
                           Theta=theta,
                           datafile=datafile,
                           length=args.timelength,
                           nodemigration=args.nodemigration,
                           resultdir=args.resultdir,
                           datatype=args.testDataType
                           )
        outdir_tmp = './output'
        if not os.path.exists(outdir_tmp):
            os.mkdir(outdir_tmp)
        outdir_tmp = outdir_tmp + '/' + args.testDataType
        if not os.path.exists(outdir_tmp):
            os.mkdir(outdir_tmp)
        if not os.path.exists(outdir_tmp + '/incrementalSVD'):
            os.mkdir(outdir_tmp + '/incrementalSVD')
        if not os.path.exists(outdir_tmp + '/rerunSVD'):
            os.mkdir(outdir_tmp + '/rerunSVD')
        if not os.path.exists(outdir_tmp + '/optimalSVD'):
            os.mkdir(outdir_tmp + '/optimalSVD')

        if args.exp == 'emb':
            print('plotting embedding not implemented!')

        if args.exp == 'lp':
            embedding.learn_embedding()

            outdir = args.resultdir
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            outdir = outdir + '/' + args.testDataType
            if not os.path.exists(outdir):
                os.mkdir(outdir)

            embedding.get_embedding(outdir_tmp, 'incrementalSVD')
            # embedding.plotresults()  
            outdir1 = outdir + '/incrementalSVD'
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(None,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'l' + str(args.timelength) + '_emb' + str(int(dim_emb)) + '_samples' + str(sample),
                                  n_sample_nodes=sample
                                  )

            embedding.get_embedding(outdir_tmp, 'rerunSVD')
            outdir1 = outdir + '/rerunSVD'
            # embedding.plotresults()
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(None,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'l' + str(args.timelength) + '_emb' + str(int(dim_emb)) + '_samples' + str(sample),
                                  n_sample_nodes=sample
                                  )

            embedding.get_embedding(outdir_tmp, 'optimalSVD')
            # embedding.plotresults()
            outdir1 = outdir + '/optimalSVD'
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(None,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'l' + str(args.timelength) + '_emb' + str(int(dim_emb)) + '_samples' + str(sample),
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

        graphs = graphs[-args.timelength:]

        datafile = dataprep_util.prep_input_TIMERS(graphs, args.timelength, args.testDataType)

        embedding = TIMERS(K=dim_emb,
                           Theta=theta,
                           datafile=datafile,
                           length=args.timelength,
                           nodemigration=args.nodemigration,
                           resultdir=args.resultdir,
                           datatype=args.testDataType
                           )
        outdir_tmp = './output'
        if not os.path.exists(outdir_tmp):
            os.mkdir(outdir_tmp)
        outdir_tmp = outdir_tmp + '/' + args.testDataType
        if not os.path.exists(outdir_tmp):
            os.mkdir(outdir_tmp)
        if not os.path.exists(outdir_tmp + '/incrementalSVD'):
            os.mkdir(outdir_tmp + '/incrementalSVD')
        if not os.path.exists(outdir_tmp + '/rerunSVD'):
            os.mkdir(outdir_tmp + '/rerunSVD')
        if not os.path.exists(outdir_tmp + '/optimalSVD'):
            os.mkdir(outdir_tmp + '/optimalSVD')

        if args.exp == 'emb':
            print('plotting embedding not implemented!')

        if args.exp == 'lp':
            embedding.learn_embedding()

            outdir = args.resultdir
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            outdir = outdir + '/' + args.testDataType
            if not os.path.exists(outdir):
                os.mkdir(outdir)

            embedding.get_embedding(outdir_tmp, 'incrementalSVD')
            # embedding.plotresults()  
            outdir1 = outdir + '/incrementalSVD'
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(None,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'l' + str(args.timelength) + '_emb' + str(int(dim_emb)) + '_samples' + str(sample),
                                  n_sample_nodes=sample
                                  )

            embedding.get_embedding(outdir_tmp, 'rerunSVD')
            outdir1 = outdir + '/rerunSVD'
            # embedding.plotresults()
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(None,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'l' + str(args.timelength) + '_emb' + str(int(dim_emb)) + '_samples' + str(sample),
                                  n_sample_nodes=sample
                                  )

            embedding.get_embedding(outdir_tmp, 'optimalSVD')
            # embedding.plotresults()
            outdir1 = outdir + '/optimalSVD'
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(None,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'l' + str(args.timelength) + '_emb' + str(int(dim_emb)) + '_samples' + str(sample),
                                  n_sample_nodes=sample
                                  )

    elif args.testDataType == 'enron':
        print("datatype:", args.testDataType)

        files = [file for file in os.listdir('./test_data/enron') if 'month' in file]
        length = len(files)
        # print(length)
        graphsall = []

        for i in range(length):
            G = nx.read_gpickle('./test_data/enron/month_' + str(i + 1) + '_graph.gpickle')
            graphsall.append(G)

        sample = graphsall[0].number_of_nodes()
        graphs = graphsall[-args.timelength:]
        # pdb.set_trace()
        datafile = dataprep_util.prep_input_TIMERS(graphs, args.timelength, args.testDataType)

        embedding = TIMERS(K=dim_emb,
                           Theta=theta,
                           datafile=datafile,
                           length=args.timelength,
                           nodemigration=args.nodemigration,
                           resultdir=args.resultdir,
                           datatype=args.testDataType
                           )
        outdir_tmp = './output'
        if not os.path.exists(outdir_tmp):
            os.mkdir(outdir_tmp)
        outdir_tmp = outdir_tmp + '/' + args.testDataType
        if not os.path.exists(outdir_tmp):
            os.mkdir(outdir_tmp)
        if not os.path.exists(outdir_tmp + '/incrementalSVD'):
            os.mkdir(outdir_tmp + '/incrementalSVD')
        if not os.path.exists(outdir_tmp + '/rerunSVD'):
            os.mkdir(outdir_tmp + '/rerunSVD')
        if not os.path.exists(outdir_tmp + '/optimalSVD'):
            os.mkdir(outdir_tmp + '/optimalSVD')

        if args.exp == 'emb':
            print('plotting embedding not implemented!')

        if args.exp == 'lp':
            embedding.learn_embedding()

            outdir = args.resultdir
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            outdir = outdir + '/' + args.testDataType
            if not os.path.exists(outdir):
                os.mkdir(outdir)

            embedding.get_embedding(outdir_tmp, 'incrementalSVD')
            # embedding.plotresults()  
            outdir1 = outdir + '/incrementalSVD'
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(None,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'l' + str(args.timelength) + '_emb' + str(int(dim_emb)) + '_samples' + str(sample),
                                  n_sample_nodes=sample
                                  )

            embedding.get_embedding(outdir_tmp, 'rerunSVD')
            outdir1 = outdir + '/rerunSVD'
            # embedding.plotresults()
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(None,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'l' + str(args.timelength) + '_emb' + str(int(dim_emb)) + '_samples' + str(sample),
                                  n_sample_nodes=sample
                                  )

            embedding.get_embedding(outdir_tmp, 'optimalSVD')
            # embedding.plotresults()
            outdir1 = outdir + '/optimalSVD'
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(None,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'l' + str(args.timelength) + '_emb' + str(int(dim_emb)) + '_samples' + str(sample),
                                  n_sample_nodes=sample
                                  )
