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

from dynamicgem.embedding.graphFac_dynamic import GraphFactorization
from dynamicgem.visualization import plot_dynamic_sbm_embedding
from dynamicgem.graph_generation import dynamic_SBM_graph


if __name__ == '__main__':
    node_num = 100
    community_num = 2
    node_change_num = 2
    length = 5
    dynamic_sbm_series = dynamic_SBM_graph.get_community_diminish_series_v2(node_num, community_num, length,1,
                                                                          node_change_num)

    dynamic_embeddings = GraphFactorization(16, 10, 10, 5 * 10 ** -2, 1.0, 1.0)
    # pdb.set_trace()
    dynamic_embeddings.learn_embeddings([g[0] for g in dynamic_sbm_series])

    plot_dynamic_sbm_embedding.plot_dynamic_sbm_embedding(dynamic_embeddings.get_embeddings(), list(dynamic_sbm_series))
    plt.show()
