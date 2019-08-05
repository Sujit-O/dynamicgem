#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
import operator
import time
import os
import importlib
import random

from six.moves import cPickle
from keras import backend as KBack
from os.path import isfile
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.linear_model import LogisticRegression

try:
    from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
except ImportError:
    from sklearn.cross_validation import cross_val_score, KFold, StratifiedKFold
from sklearn import svm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


from dynamicgem.embedding.static_graph_embedding import StaticGraphEmbedding
from dynamicgem.utils import graph_util, plot_util, dataprep_util
from dynamicgem.evaluation import visualize_embedding as viz
from dynamicgem.utils.sdne_utils import *
from dynamicgem.graph_generation import dynamic_SBM_graph
from dynamicgem.utils.dynamictriad_utils import *
import dynamicgem.utils.dynamictriad_utils.dataset.dataset_utils as du
import dynamicgem.utils.dynamictriad_utils.algorithm.embutils as eu
from dynamicgem.evaluation import evaluate_link_prediction as lp


class dynamicTriad(StaticGraphEmbedding):
    """ Dynamic Triad Closure based embedding
        
        DynamicTriad preserves both structural informa- tion 
        and evolution patterns of a given network. The general 
        idea of our approach is to impose triad, which is a group 
        of three vertices and is one of the basic units of networks.

        Args:
            niters (int): Number of iteration to run the algorithm
            starttime (int): start time for the graph step
            datafile (str) : The file for the input graph
            batchsize (int): batch size for training the algorithm
            nsteps (int) : total number of steps in the temporal graph
            embdim (int): embedding dimension
            stepsize (int): step size for the graph
            stepstride (int): stride to consider for temporal stride
            outdir (str): The output directory to store the result
            cachefn (str): Directory to cache the temporary data
            lr (float): Learning rate for the algorithm
            beta (float): coefficients for triad component
            negdup (float): neg/pos ratio during sampling
            datasetmod (str): module name for dataset loading
            trainmod (str): module name for training model 
            pretrain_size (int): size of the  graph for pre-training 
            sampling_args (int): sampling size
            validation (list): link_reconstruction validation data
            datatype (str): type of network data 
            scale (int): scaling 
            classifier (str): type of classifier to be used
            debug (bool): debugging flag
            test (bool): type of test to perform
            repeat (int): Number of times to repeat the learning
            resultdir (str): directory to store the result
            testDataType (str): type of test data
            clname (str) : classifier type 
            node_num (int): number of nodes
        
        Examples:
            >>> from dynamicgem.embedding.dynamicTriad import dynamicTriad
            >>> from dynamicgem.graph_generation import dynamic_SBM_graph
            >>> node_num = 200
            >>> community_num = 2
            >>> node_change_num = 2
            >>> length =5
            >>> dynamic_sbm_series = dynamic_SBM_graph.get_community_diminish_series_v2(node_num,
                                                                                    community_num,
                                                                                    length,
                                                                                    1,
                                                                                    node_change_num)
            >>> graphs = [g[0] for g in dynamic_sbm_series]

            >>> datafile = dataprep_util.prep_input_dynTriad(graphs, length, args.testDataType)

            >>> embedding = dynamicTriad(niters=10,
                                     starttime=0,
                                     datafile=datafile,
                                     batchsize=10,
                                     nsteps=5,
                                     embdim=16,
                                     stepsize=1,
                                     stepstride=1,
                                     outdir='./output',
                                     cachefn='./tmp',
                                     lr=0.001,
                                     beta=0.1,
                                     negdup=1,
                                     datasetmod='dynamicgem.utils.dynamictriad_utils.dataset.adjlist',
                                     trainmod='dynamicgem.utils.dynamictriad_utils.algorithm.dynamic_triad',
                                     pretrain_size=4,
                                     sampling_args={},
                                     validation='link_reconstruction',
                                     datatype='sbm_cd',
                                     scale=1,
                                     classifier='lr',
                                     debug=False,
                                     test='link_predict',
                                     repeat=1,
                                     resultdir='./results_link_all',
                                     testDataType='sbm_cd',
                                     clname='lr',
                                     node_num=node_num )

            >>> embedding.learn_embedding()
            >>> embedding.get_embedding()
            >>> outdir = args.resultdir
            >>> if not os.path.exists(outdir):
            >>>     os.mkdir(outdir)
            >>> outdir = outdir + '/' + args.testDataType
            >>> if not os.path.exists(outdir):
            >>>     os.mkdir(outdir)
            >>> outdir = outdir + '/' + 'dynTRIAD'
            >>> if not os.path.exists(outdir):
             >>>    os.mkdir(outdir)

            >>> lp.expstaticLP_TRIAD(dynamic_sbm_series,
                                 graphs,
                                 embedding,
                                 1,
                                 outdir + '/',
                                 'nm' + str(args.nodemigration) + '_l' + str(args.nsteps) + '_emb' + str(args.embdim),
                                 )
    """

    def __init__(self, *hyper_dict, **kwargs):
        hyper_params = {
            'method_name': 'Dynamic TRIAD',
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
        self.clf = self.__make_classifier()
        self._model = None
        # self._clname='lr'       

    def __make_classifier(self):
        """Function to initialize the classifier"""
        class_weight = 'balanced'

        if self._clname == 'svm':
            return svm.SVC(kernel='linear', class_weight=class_weight)
        elif self._clname == 'lr':
            return LogisticRegression(class_weight=class_weight)
        else:
            raise NotImplementedError()

    def load_trainmod(self, modname):
        """Function to load the training module"""
        mod = importlib.import_module(modname)
        return getattr(mod, 'Model')

    def load_datamod(self, modname):
        """Function to load the dataset module"""
        mod = importlib.import_module(modname)
        return getattr(mod, 'Dataset')

    def load_or_update_cache(self, ds, cachefn):
        """Function to either update or load the cache"""
        if cachefn is None:
            return
        cachefn += '.cache'
        if isfile(cachefn + '.args'):
            args = cPickle.load(open(cachefn + '.args', 'r'))
            try:
                ds.load_cache(args, lambda: cPickle.load(open(cachefn, 'r')))
                print("Data loaded from cache file {}".format(cachefn))
                return
            except (ValueError, EOFError) as e:
                print("Failed to load cache file {}: {}".format(cachefn, e.message))

        # update cache
        print("updating cache file for prefix {}".format(cachefn))
        ar, args = ds.cache()
        cPickle.dump(args, open(cachefn + '.args', 'w'))
        cPickle.dump(ar, open(cachefn, 'w'))
        print("cache file {} updated".format(cachefn))

    def export(self, vertices, data, outdir):
        """function to export the data"""
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + self._datatype
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/dynTriad'
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        for i in range(len(data)):
            assert len(vertices) == len(data[i]), (len(vertices), len(data[i]))
            fn = "{}/{}.out".format(outdir, i)
            fh = open(fn, 'w')
            for j in range(len(vertices)):
                print("{} {}".format(vertices[j], ' '.join(["{:.3f}".format(d) for d in data[i][j]])), file=fh)
            fh.close()

    def load_embedding(self, fn, vs):
        """Function to load the embedding"""
        data = open(fn, 'r').read().rstrip('\n').split('\n')
        emb = {}
        for line in data:
            fields = line.split()
            emb[fields[0]] = [float(e) for e in fields[1:]]
        # it is possible that the output order differs from :param vs: given different node_type,
        # so we have to reorder the embedding according to :param vs:
        emb = [emb[str(v)] for v in vs]

        return np.vstack(emb)

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
        return '%s_%d' % (self._method_name)

    def learn_embedding(self):
        """Learns the embedding of the nodes.
           
            Returns:
                List: Node embeddings and time taken by the algorithm
        """
        # TensorFlow wizardry
        config = tf.ConfigProto()

        # Don't pre-allocate memory; allocate as-needed
        config.gpu_options.allow_growth = True

        # Only allow a total of half the GPU memory to be allocated
        config.gpu_options.per_process_gpu_memory_fraction = 0.2

        # Create a session to pass the above configuration
        sess = tf.Session(config=config)

        # Create a tensorflow debugger wrapper
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess) 

        # Create a session with the above options specified.
        KBack.tensorflow_backend.set_session(sess)

        TrainModel = self.load_trainmod(self._trainmod)
        Dataset = self.load_datamod(self._datasetmod)

        ds = Dataset(self._datafile, self._starttime, self._nsteps, stepsize=self._stepsize,
                     stepstride=self._stepstride)
        #         self.load_or_update_cache(ds, self._cachefn)
        # dsargs = {'datafile': self._datafile, 'starttime': self._starttime, 'nsteps': self._nsteps,
        #           'stepsize': self._stepsize, 'stepstride': self._stepstride, 'datasetmod': self._datasetmod}
        tm = TrainModel(ds, pretrain_size=self._pretrain_size, embdim=self._embdim, beta=self._beta,
                        lr=self._lr, batchsize=self._batchsize, sampling_args=self._sampling_args)

        edgecnt = [g.number_of_edges() for g in ds.gtgraphs]
        k_edgecnt = sum(edgecnt[:self._pretrain_size])
        print("{} edges in pretraining graphs".format(k_edgecnt))

        if self._pretrain_size > 0:
            initstep = int(ds.time2step(self._starttime))
            tm.pretrain_begin(initstep, initstep + self._pretrain_size)

            print("generating validation set")
            validargs = tm.dataset.sample_test_data(self._validation, initstep, initstep + self._pretrain_size,
                                                    size=10000)
            # print(validargs)
            print("{} validation samples generated".format(len(validargs[0])))

            max_val, max_idx, maxmodel = -1, 0, None

            # for early stopping
            start_time = time.time()
            scores = []
            for i in range(self._niters):
                tm.pretrain_begin_iteration()

                epoch_loss = 0
                for batidx, bat in enumerate(tm.batches(self._batchsize)):
                    inputs = tm.make_pretrain_input(bat)
                    l = tm.pretrain['lossfunc'](inputs)
                    if isinstance(l, (list, tuple)):
                        l = l[0]
                    epoch_loss += l
                    print("\repoch {}: {:.0%} completed, cur loss: {:.3f}".format(i, float(batidx * self._batchsize)
                                                                                  / tm.sample_size(), l.flat[0]),
                          end='')
                    sys.stdout.flush()
                tm.pretrain_end_iteration()

                print(" training completed, total loss {}".format(epoch_loss), end='')

                # without validation, the model exists only after I iterations
                if self._validation != 'none':
                    val_score = tm.validate(self._validation, *validargs)

                    if val_score > max_val:
                        max_val = val_score
                        max_idx = i
                        maxmodel = tm.save_model()
                    print(", validation score {:.3f}".format(val_score))
                else:
                    max_idx, max_val = i, epoch_loss
                    # maxmodel is not saved here in order to save time
                    print("")

                if self._validation != 'none':
                    scores.append(val_score)
                    if max_val > 0 and i - max_idx > 5:
                        break

            print("best validation score at itr {}: {}".format(max_idx, max_val))
            print("{} seconds elapsed for pretraining".format(time.time() - start_time))
            # lastmodel = tm.save_model()  # for debug
            print("saving output to {}".format(self._outdir))
            tm.restore_model(maxmodel)
            tm.pretrain_end()
            self.export(list(tm.dataset.mygraphs['any'].nodes()), tm.export(), self._outdir)

        # online training disabled
        startstep = int(tm.dataset.time2step(self._starttime))
        for y in range(startstep + self._pretrain_size, startstep + self._nsteps):
            raise NotImplementedError()

    def get_embedding(self):
        """Function to return the embeddings"""
        self._X = dataprep_util.getemb_dynTriad(self._outdir + '/' + self._testDataType + '/dynTriad', self._nsteps,
                                                self._embdim)
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
            feat = np.fabs(self._X[t][i, :] - self._X[t][j, :])
            return self._model.predict(np.reshape(feat, [1, -1]))[0]
        except:
            pdb.set_trace()

    def get_reconstructed_adj(self, t, X=None, node_l=None):
        """Function to reconstruct the adjacency list for the given node.
           
            Attributes:
              node_l (int): node for which the adjacency list will be created.
              X (Matrix): Embedding values of all the nodes.
              t (int): Time step

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

    def sample_link_reconstruction(self, g, sample_nodes=None, negdup=1):
        """Function to sample the link reconstruction"""
        pos = []
        # assert not g.is_directed()
        # for g in graphs:
        for e in g.edges():
            if int(e[0]) > int(e[1]):
                # check symmetric
                names = list(g.nodes())
                assert g.edges(e[0], e[1]), "{}: {} {}".format(names[e[0]],
                                                               names[e[1]])
                continue
            pos.append([int(e[0]), int(e[1])])
        pos = np.vstack(pos).astype('int32')

        neg = []
        vsize = len(g.nodes())
        nodenames = list(g.nodes())
        for i in range(negdup):
            for p in pos:
                src, tgt = p
                # g = self.mygraphs[tm + intv]
                assert g.out_degree(nodenames[src]) < vsize - 1 or g.out_degree(nodenames[tgt]) < vsize - 1, \
                    "We do not expect any node to connect to all other nodes"

                while True:
                    if random.randint(0, 1) == 0:  # replace source
                        # cur_range = negrange[tm][tgt]
                        # new_src = cur_range[random.randint(0, len(cur_range) - 1)]
                        new_src = random.randint(0, vsize - 1)
                        if not g.has_edge(nodenames[new_src], nodenames[tgt]):
                            neg.append([new_src, tgt])
                            break
                    else:  # replace target
                        # cur_range = negrange[tm][src]
                        # new_tgt = cur_range[random.randint(0, len(cur_range) - 1)]
                        new_tgt = random.randint(0, vsize - 1)
                        if not g.has_edge(nodenames[src], nodenames[new_tgt]):
                            neg.append([src, new_tgt])
                            break
        neg = np.vstack(neg).astype('int32')

        lbs = np.concatenate((np.ones(len(pos)), -np.ones(len(neg))))
        return np.concatenate((pos, neg), axis=0), lbs

    class ResultPresenter(object):
        """result presenter class"""
        def __init__(self):
            self.f1, self.prec, self.rec, self.acc = [], [], [], []

        def add_result(self, res):
            self.prec.extend(res[0])
            self.rec.extend(res[1])
            self.f1.extend(res[2])
            self.acc.extend(res[3])

        def show_result(self):
            print("precision mean: {} std: {}".format(np.mean(self.prec), np.std(self.prec)))
            print("recall mean: {} std: {}".format(np.mean(self.rec), np.std(self.rec)))
            print("f1 mean: {} std: {}".format(np.mean(self.f1), np.std(self.f1)))
            print("accuracy mean: {} std: {}".format(np.mean(self.acc), np.std(self.acc)))

    def __classify(self, feat, lbs):
        sm = None

        poscnt, negcnt = np.sum(lbs == 1), np.sum(lbs == -1)
        print("classifying with pos:neg = {}:{}".format(poscnt, negcnt))

        try:
            cv = StratifiedKFold(n_splits=5, shuffle=True)
            parts = cv.split(feat, lbs)
        except TypeError:
            cv = StratifiedKFold(lbs, n_folds=5, shuffle=True)
            parts = cv

        f1, prec, rec, acc = [], [], [], []
        for tr, te in parts:
            if sm is not None:
                x, y = sm.fit_sample(feat[tr], lbs[tr])
                # x, y = feat[tr], lbs[tr]
            else:
                x, y = feat[tr], lbs[tr]
            model = self.clf.fit(x, y)
            p = model.predict(feat[te])
            # self._model=model
            # if self.debug:
            #     print("results:", p, lbs[te])
            # print(p,np.shape(p))
            f1.append(f1_score(lbs[te], p))
            prec.append(precision_score(lbs[te], p))
            rec.append(recall_score(lbs[te], p))
            acc.append(accuracy_score(lbs[te], p))
        # idx = np.random.permutation(len(lbs))
        # x,y = feat[idx], lbs[idx]
        # self._model=self.clf.fit(x, y)    
        return prec, rec, f1, acc

    def link_predict(self, g, t, intv=0, repeat=1):
        """Function to perform link prediction"""
        samp, lbs = self.sample_link_reconstruction(g, sample_nodes=None, negdup=1)
        # pdb.set_trace()
        # TODO: different feature generation method might be used here
        try:
            feat = np.fabs(self._X[t][samp[:, 0]] - self._X[t][samp[:, 1]])
        except:
            pdb.set_trace()
        print("feature shape {}".format(feat.shape))

        # rp = self.ResultPresenter()
        # for i in range(repeat):
        #     res = self.__classify(feat, lbs)
        #     rp.add_result(res)
        # rp.show_result()

        idx = np.random.permutation(len(lbs))
        x, y = feat[idx], lbs[idx]
        self._model = self.clf.fit(x, y)

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

    def plotresults(self, dynamic_sbm_series):
        """Function to plot the result"""
        plt.figure()
        plt.clf()
        viz.plot_static_sbm_embedding(self._X[-4:], dynamic_sbm_series[-4:])
