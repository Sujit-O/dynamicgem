disp_avlbl = True
import os
if os.name == 'posix' and 'DISPLAY' not in os.environ:
    disp_avlbl = False
    import matplotlib

    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from joblib import Parallel, delayed

import keras.regularizers as Reg
from argparse import ArgumentParser
from time import time
import operator

from dynamicgem.embeddings.dynAE import DynAE
from dynamicgem.utils import plot_util, graph_util, dataprep_util
from dynamicgem.visualization import plot_dynamic_sbm_embedding
from dynamicgem.graph_generation import dynamic_SBM_graph
from dynamicgem.evaluation import evaluate_link_prediction, evaluate_graph_reconstruction
from dynamicgem.utils.dnn_utils import *


if __name__ == '__main__':
    parser = ArgumentParser(description='Learns node embeddings for a sequence of graph snapshots')
    parser.add_argument('-t', '--testDataType',
                        default='sbm_cd',
                        type=str,
                        help='Type of data to test the code')
    parser.add_argument('-c', '--criteria',
                        default='degree',
                        type=str,
                        help='Node Migration criteria')
    parser.add_argument('-rc', '--criteria_r',
                        default=1,
                        type=int,
                        help='Take highest centrality measure to perform node migration')
    parser.add_argument('-l', '--timelength',
                        default=5,
                        type=int,
                        help='Number of time series graph to generate')
    parser.add_argument('-lb', '--lookback',
                        default=2,
                        type=int,
                        help='number of lookbacks')
    parser.add_argument('-eta', '--learningrate',
                        default=1e-4,
                        type=float,
                        help='learning rate')
    parser.add_argument('-bs', '--batch',
                        default=100,
                        type=int,
                        help='batch size')
    parser.add_argument('-nm', '--nodemigration',
                        default=2,
                        type=int,
                        help='number of nodes to migrate')
    parser.add_argument('-iter', '--epochs',
                        default=2,
                        type=int,
                        help='number of epochs')
    parser.add_argument('-emb', '--embeddimension',
                        default=16,
                        type=int,
                        help='embedding dimension')
    parser.add_argument('-rd', '--resultdir',
                        type=str,
                        default='./results_link_all',
                        help="result directory name")
    parser.add_argument('-sm', '--samples',
                        default=10,
                        type=int,
                        help='samples for test data')
    parser.add_argument('-ht', '--hypertest',
                        default=0,
                        type=int,
                        help='hyper test')
    parser.add_argument('-exp', '--exp',
                        default='lp',
                        type=str,
                        help='experiments (lp, emb)')

    args = parser.parse_args()
    epochs = args.epochs
    dim_emb = args.embeddimension
    lookback = args.lookback
    length = args.timelength

    if not os.path.exists('./intermediate'):
          os.mkdir('./intermediate')
          
    if length < lookback + 5:
        length = lookback + 5
    if args.testDataType == 'sbm_rp':
        node_num = 10000
        community_num = 500
        node_change_num = 100
        dynamic_sbm_series = dynamic_SBM_graph.get_random_perturbation_series(node_num,
                                                                              community_num,
                                                                              length,
                                                                              node_change_num)
        dynamic_embedding = DynAE(
            d=100,
            beta=5,
            n_prev_graphs=lookback,
            nu1=1e-6,
            nu2=1e-6,
            n_units=[500, 300, ],
            rho=0.3,
            n_iter=1000,
            xeta=0.005,
            n_batch=500,
            modelfile=['./intermediate/enc_model.json', './intermediate/dec_model.json'],
            weightfile=['./intermediate/enc_weights.hdf5', './intermediate/dec_weights.hdf5'],
        )
        dynamic_embedding.learn_embeddings([g[0] for g in dynamic_sbm_series])
        plt.clf()
        plot_dynamic_sbm_embedding.plot_dynamic_sbm_embedding(dynamic_embedding.get_embeddings(),
                                                              dynamic_sbm_series)
        plt.savefig('result/visualization_DynRNN_rp.png')
        plt.show()

    elif args.testDataType == 'sbm_cd':
        node_num = 100
        community_num = 2
        node_change_num = args.nodemigration
        dynamic_sbm_series = dynamic_SBM_graph.get_community_diminish_series_v2(node_num,
                                                                                community_num,
                                                                                length,
                                                                                1,  # communitiy to dimisnish
                                                                                node_change_num
                                                                                )
        dynamic_embedding = DynAE(
            d=dim_emb,
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
            savefilesuffix="testing"
        )
        graphs = [g[0] for g in dynamic_sbm_series]
        embs = []

        outdir = args.resultdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + args.testDataType
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        outdir = outdir + '/dynAE'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if args.exp == 'emb':
            result = Parallel(n_jobs=4)(delayed(dynamic_embedding.learn_embeddings)(graphs[:temp_var]) for temp_var in
                                        range(lookback + 1, length + 1))
            for i in range(len(result)):
                embs.append(np.asarray(result[i][0]))

            for temp_var in range(lookback + 1, length + 1):
                emb, _ = dynamic_embedding.learn_embeddings(graphs[:temp_var])
                embs.append(emb)

            plt.figure()
            plt.clf()
            plot_dynamic_sbm_embedding.plot_dynamic_sbm_embedding_v2(embs[-5:-1], dynamic_sbm_series[-5:])
            plt.savefig('./' + outdir + '/V_DynAE_nm' + str(args.nodemigration) + '_l' + str(length) + '_epoch' + str(
                epochs) + '_emb' + str(dim_emb) + '.pdf', bbox_inches='tight', dpi=600)
            plt.show()

        if args.hypertest == 1:
            fname = 'epoch' + str(args.epochs) + '_bs' + str(args.batch) + '_lb' + str(args.lookback) + '_eta' + str(
                args.learningrate) + '_emb' + str(args.embeddimension)
        else:
            fname = 'nm' + str(args.nodemigration) + '_l' + str(length) + '_emb' + str(dim_emb)

        if args.exp == 'lp':
            evaluate_link_prediction.expLP(
                graphs,
                dynamic_embedding,
                1,
                outdir + '/',
                fname,
            )

    elif args.testDataType == 'academic':
        print("datatype:", args.testDataType)

        dynamic_embedding = DynAE(
            d=dim_emb,
            beta=5,
            n_prev_graphs=lookback,
            nu1=1e-6,
            nu2=1e-6,
            n_units=[500, 300, ],
            rho=0.3,
            n_iter=epochs,
            xeta=1e-5,
            n_batch=100,
            modelfile=['./intermediate/enc_modelacdm.json', './intermediate/dec_modelacdm.json'],
            weightfile=['./intermediate/enc_weightsacdm.hdf5', './intermediate/dec_weightsacdm.hdf5'],
            savefilesuffix="testingacdm"
        )

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
        for i in range(length):
            graphs[i] = graph_util.sample_graph_nodes(graphs[i], node_l)

        outdir = args.resultdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + args.testDataType
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        outdir = outdir + '/dynAE'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if args.exp == 'emb':
            print('plotting embedding not implemented!')

        if args.exp == 'lp':
            evaluate_link_prediction.expLP(graphs[-args.timelength:],
                                           dynamic_embedding,
                                           1,
                                           outdir + '/',
                                           'lb_' + str(lookback) + '_l' + str(args.timelength) + '_emb' + str(
                                               dim_emb) + '_samples' + str(sample),
                                           n_sample_nodes=sample
                                           )

    elif args.testDataType == 'hep':
        print("datatype:", args.testDataType)
        dynamic_embedding = DynAE(
            d=dim_emb,
            beta=5,
            n_prev_graphs=lookback,
            nu1=1e-6,
            nu2=1e-6,
            n_units=[500, 300, ],
            rho=0.3,
            n_iter=epochs,
            xeta=1e-8,
            n_batch=int(args.samples / 10),
            modelfile=['./intermediate/enc_modelhep.json', './intermediate/dec_modelhep.json'],
            weightfile=['./intermediate/enc_weightshep.hdf5', './intermediate/dec_weightshep.hdf5'],
            savefilesuffix="testinghep"
        )

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

        outdir = args.resultdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + args.testDataType
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        outdir = outdir + '/dynAE'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if args.exp == 'emb':
            print('plotting embedding not implemented!')

        if args.exp == 'lp':
            evaluate_link_prediction.expLP(graphs[-args.timelength:],
                                           dynamic_embedding,
                                           1,
                                           outdir + '/',
                                           'lb_' + str(lookback) + '_l' + str(args.timelength) + '_emb' + str(
                                               dim_emb) + '_samples' + str(sample),
                                           n_sample_nodes=sample
                                           )

    elif args.testDataType == 'AS':
        print("datatype:", args.testDataType)
        dynamic_embedding = DynAE(
            d=dim_emb,
            beta=5,
            n_prev_graphs=lookback,
            nu1=1e-6,
            nu2=1e-6,
            n_units=[500, 300, ],
            rho=0.3,
            n_iter=epochs,
            xeta=1e-5,
            n_batch=int(args.samples / 10),
            modelfile=['./intermediate/enc_modelAS.json', './intermediate/dec_modelAS.json'],
            weightfile=['./intermediate/enc_weightsAS.hdf5', './intermediate/dec_weightsAS.hdf5'],
            savefilesuffix="testingAS"
        )

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

        outdir = args.resultdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + args.testDataType
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        outdir = outdir + '/dynAE'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if args.exp == 'emb':
            print('plotting embedding not implemented!')

        if args.exp == 'lp':
            evaluate_link_prediction.expLP(graphs[-args.timelength:],
                                           dynamic_embedding,
                                           1,
                                           outdir + '/',
                                           'lb_' + str(lookback) + '_l' + str(args.timelength) + '_emb' + str(
                                               dim_emb) + '_samples' + str(sample),
                                           n_sample_nodes=sample
                                           )

    elif args.testDataType == 'enron':
        print("datatype:", args.testDataType)

        dynamic_embedding = DynAE(
            d=dim_emb,
            beta=5,
            n_prev_graphs=lookback,
            nu1=1e-6,
            nu2=1e-6,
            n_units=[500, 300, ],
            rho=0.3,
            n_iter=epochs,
            xeta=1e-8,
            n_batch=20,
            modelfile=['./intermediate/enc_modelenron.json', './intermediate/dec_modelenron.json'],
            weightfile=['./intermediate/enc_weightsenron.hdf5', './intermediate/dec_weightsenron.hdf5'],
            savefilesuffix="testingAS"
        )

        files = [file for file in os.listdir('./test_data/enron') if 'week' in file]
        length = len(files)
        graphs = []

        for i in range(length):
            G = nx.read_gpickle('./test_data/enron/week_' + str(i) + '_graph.gpickle')
            graphs.append(G)

        sample = graphs[0].number_of_nodes()
        print(sample)

        outdir = args.resultdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + args.testDataType
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        outdir = outdir + '/dynAE'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if args.exp == 'emb':
            print('plotting embedding not implemented!')

        if args.exp == 'lp':
            evaluate_link_prediction.expLP(graphs[-args.timelength:],
                                           dynamic_embedding,
                                           1,
                                           outdir + '/',
                                           'lb_' + str(lookback) + '_l' + str(args.timelength) + '_emb' + str(
                                               dim_emb) + '_samples' + str(sample),
                                           n_sample_nodes=sample
                                           )
