'''
============================
Example Code for DynGEM
============================
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
import matplotlib.pyplot as plt

from dynamicgem.embedding.dynGEM import DynGEM
from dynamicgem.graph_generation import SBM_graph
from dynamicgem.utils import graph_util, plot_util
from dynamicgem.graph_generation import SBM_graph
from dynamicgem.evaluation import evaluate_graph_reconstruction as gr
from time import time

if __name__ == '__main__':
    my_graph = SBM_graph.SBMGraph(100, 2)
    my_graph.sample_graph()
    node_colors = plot_util.get_node_color(my_graph._node_community)
    t1 = time()
    embedding = DynGEM(d=8, beta=5, alpha=0, nu1=1e-6, nu2=1e-6, K=3,
                     n_units=[64, 16], n_iter=2, xeta=0.01,
                     n_batch=50,
                     modelfile=['./intermediate/enc_model.json',
                                './intermediate/dec_model.json'],
                     weightfile=['./intermediate/enc_weights.hdf5',
                                 './intermediate/dec_weights.hdf5'])
    embedding.learn_embedding(graph=my_graph._graph, edge_f=None,
                              is_weighted=True, no_python=True)
    print('SDNE:\n\tTraining time: %f' % (time() - t1))
    MAP, prec_curv, err, err_baseline = \
        gr.evaluateStaticGraphReconstruction(
            my_graph._graph,
            embedding,
            embedding.get_embedding(),
            None
        )
    print(MAP)
    print(prec_curv[:10])