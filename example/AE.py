import os
import networkx as nx
disp_avlbl = True
if os.name == 'posix' and 'DISPLAY' not in os.environ:
    disp_avlbl = False
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from argparse import ArgumentParser
import sys
import pdb
from joblib import Parallel, delayed
import operator
from time import time

from dynamicgem.embedding.ae_static import AE
from dynamicgem.utils import graph_util, dataprep_util
from dynamicgem.evaluation import visualize_embedding as viz
from dynamicgem.utils.sdne_utils import *
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
    parser.add_argument('-iter', '--epochs',
                        default=2,
                        type=int,
                        help='number of epochs')
    parser.add_argument('-emb', '--embeddimension',
                        default=16,
                        type=int,
                        help='embedding dimension')
    parser.add_argument('-sm', '--samples',
                        default=10,
                        type=int,
                        help='samples for test data')
    parser.add_argument('-exp', '--exp',
                        default='lp',
                        type=str,
                        help='experiments (lp, emb)')
    parser.add_argument('-rd', '--resultdir',
                        type=str,
                        default='./results_link_all',
                        help="result directory name")

    args = parser.parse_args()
    epochs = args.epochs
    dim_emb = args.embeddimension
    length = args.timelength

    if not os.path.exists('./intermediate'):
          os.mkdir('./intermediate')
          
    if args.testDataType == 'sbm_cd':
        node_num = 100
        community_num = 2
        node_change_num = args.nodemigration
        dynamic_sbm_series = dynamic_SBM_graph.get_community_diminish_series_v2(node_num,
                                                                                community_num,
                                                                                length,
                                                                                1,
                                                                                node_change_num)
        
        embedding = AE(d=dim_emb,
                       beta=5,
                       nu1=1e-6,
                       nu2=1e-6,
                       K=3,
                       n_units=[500, 300, ],
                       n_iter=epochs,
                       xeta=1e-4,
                       n_batch=100,
                       modelfile=['./intermediate/AE_enc_modelsbm.json',
                                  './intermediate/AE_dec_modelsbm.json'],
                       weightfile=['./intermediate/AE_enc_weightssbm.hdf5',
                                   './intermediate/AE_dec_weightssbm.hdf5'])

        graphs = [g[0] for g in dynamic_sbm_series]
        embs = []

        outdir = args.resultdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + args.testDataType
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        outdir = outdir + '/staticAE'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if args.exp == 'emb':
            for temp_var in range(length):
                emb, _ = embedding.learn_embeddings(graphs[temp_var])
                embs.append(emb)

            result = Parallel(n_jobs=4)(
                delayed(embedding.learn_embeddings)(graphs[temp_var]) for temp_var in range(length))
            for i in range(len(result)):
                embs.append(np.asarray(result[i][0]))

            plt.figure()
            plt.clf()
            viz.plot_static_sbm_embedding(embs[-4:], dynamic_sbm_series[-4:])

            plt.savefig('./' + outdir + '/V_AE_nm' + str(args.nodemigration) + '_l' + str(length) + '_epoch' + str(
                epochs) + '_emb' + str(dim_emb) + '.pdf', bbox_inches='tight', dpi=600)
            plt.show()
            plt.close()

        if args.exp == 'lp':
            lp.expstaticLP(dynamic_sbm_series,
                           graphs,
                           embedding,
                           1,
                           outdir + '/',
                           'nm' + str(args.nodemigration) + '_l' + str(length) + '_emb' + str(dim_emb),
                           )

    elif args.testDataType == 'academic':
        print("datatype:", args.testDataType)

        embedding = AE(d=dim_emb,
                       beta=5,
                       nu1=1e-6,
                       nu2=1e-6,
                       K=3,
                       n_units=[500, 300, ],
                       n_iter=epochs,
                       xeta=1e-4,
                       n_batch=1000,
                       modelfile=['./intermediate/enc_modelacdm.json',
                                  './intermediate/dec_modelacdm.json'],
                       weightfile=['./intermediate/enc_weightsacdm.hdf5',
                                   './intermediate/dec_weightsacdm.hdf5'])

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

        outdir = outdir + '/staticAE'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if args.exp == 'emb':
            print('plotting embedding not implemented!')

        if args.exp == 'lp':
            lp.expstaticLP(None,
                           graphs[-args.timelength:],
                           embedding,
                           1,
                           outdir + '/',
                           'l' + str(args.timelength) + '_emb' + str(dim_emb) + '_samples' + str(sample),
                           n_sample_nodes=graphs[i].number_of_nodes()
                           )

    elif args.testDataType == 'hep':
        print("datatype:", args.testDataType)
        embedding = AE(d=dim_emb,
                       beta=5,
                       nu1=1e-6,
                       nu2=1e-6,
                       K=3,
                       n_units=[500, 300, ],
                       n_iter=epochs,
                       xeta=1e-4,
                       n_batch=1000,
                       modelfile=['./intermediate/enc_modelhep.json',
                                  './intermediate/dec_modelhep.json'],
                       weightfile=['./intermediate/enc_weightshep.hdf5',
                                   './intermediate/dec_weightshep.hdf5'])

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

        outdir = outdir + '/staticAE'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if args.exp == 'emb':
            print('plotting embedding not implemented!')

        if args.exp == 'lp':
            lp.expstaticLP(None,
                           graphs[-args.timelength:],
                           embedding,
                           1,
                           outdir + '/',
                           'l' + str(args.timelength) + '_emb' + str(dim_emb) + '_samples' + str(sample),
                           n_sample_nodes=graphs[i].number_of_nodes()
                           )

    elif args.testDataType == 'AS':
        print("datatype:", args.testDataType)
        embedding = AE(d=dim_emb,
                       beta=5,
                       nu1=1e-6,
                       nu2=1e-6,
                       K=3,
                       n_units=[500, 300, ],
                       n_iter=epochs,
                       xeta=1e-4,
                       n_batch=1000,
                       modelfile=['./intermediate/enc_modelAS.json',
                                  './intermediate/dec_modelAS.json'],
                       weightfile=['./intermediate/enc_weightsAS.hdf5',
                                   './intermediate/dec_weightsAS.hdf5'])

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

        outdir = outdir + '/staticAE'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if args.exp == 'emb':
            print('plotting embedding not implemented!')

        if args.exp == 'lp':
            lp.expstaticLP(None,
                           graphs[-args.timelength:],
                           embedding,
                           1,
                           outdir + '/',
                           'l' + str(args.timelength) + '_emb' + str(dim_emb) + '_samples' + str(sample),
                           n_sample_nodes=graphs[i].number_of_nodes()
                           )

    elif args.testDataType == 'enron':
        print("datatype:", args.testDataType)
        embedding = AE(d=dim_emb,
                       beta=5,
                       nu1=1e-6,
                       nu2=1e-6,
                       K=3,
                       n_units=[500, 300, ],
                       n_iter=epochs,
                       xeta=1e-8,
                       n_batch=20,
                       modelfile=['./intermediate/enc_modelAS.json',
                                  './intermediate/dec_modelAS.json'],
                       weightfile=['./intermediate/enc_weightsAS.hdf5',
                                   './intermediate/dec_weightsAS.hdf5'])

        files = [file for file in os.listdir('./test_data/enron') if '.gpickle' in file if 'month' in file]
        length = len(files)
        graphsall = []

        for i in range(length):
            G = nx.read_gpickle('./test_data/enron/month_' + str(i + 1) + '_graph.gpickle')
            graphsall.append(G)

        outdir = args.resultdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + args.testDataType
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        outdir = outdir + '/staticAE'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if args.exp == 'emb':
            print('plotting embedding not implemented!')

        if args.exp == 'lp':
            sample = graphsall[0].number_of_nodes()
            graphs = graphsall[-args.timelength:]
            lp.expstaticLP(None,
                           graphs,
                           embedding,
                           1,
                           outdir + '/',
                           'l' + str(args.timelength) + '_emb' + str(dim_emb) + '_samples' + str(sample),
                           n_sample_nodes=sample
                           )