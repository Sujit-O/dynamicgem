Implemented Algorithms
-----------------------

Dynamic graph embedding algorithms aim to capture the dynamics of the network and its evolution. These methods are useful to predict the future behavior of the network, such as future connections within a network. The problem can be defined formally as follows.

Consider a weighted graph $G(V, E)$, with $V$ and $E$ as the set of vertices and edges respectively.
Given an evolution of graph :math:`\mathcal{G} = \lbrace G_1, .., G_T\rbrace`, where :math:`G_t` represents the state of graph at time :math:`t`, a dynamic graph embedding method aims to represent each node $v$ in a series of low-dimensional vector space :math:`y_{v_1}, \ldots y_{v_t}` by learning mappings :math:`f_t: \{V_1, \ldots, V_t, E_1, \ldots E_t\} \rightarrow \mathbb{R}^d` and :math:`y_{v_i} = f_i(v_1, \ldots, v_i, E_1, \ldots E_i)`.
The methods differ in the definition of :math:`f_t` and the properties of the network preserved by :math:`f_t`.

There are various existing state of the art methods trying to solve this problem that we have incorporated and included them in this python package including: 

* `Optimal SVD`_: This method decomposes adjacency matrix of the graph at each time step using Singular Value Decomposition (SVD) to represent each node using the $d$ largest singular values.

* `Incremental SVD`_: This method utilizes a perturbation matrix capturing the dynamics of the graphs along with performing additive modification on the SVD.

* `Rerun SVD`_: This method uses incremental SVD to create the dynamic graph embedding. In addition to that, it uses a tolerance threshold to restart the optimal SVD calculations and avoid deviation in incremental graph embedding.

* `Dynamic TRIAD`_: This method utilizes the triadic closure process to generate a graph embedding that preserves structural and evolution patterns of the graph.

* AEalign_: This method uses deep auto-encoder to embed each node in the graph and aligns the embeddings at different time steps using a rotation matrix.

* dynGEM_: This method utilizes deep auto-encoders to incrementally generate embedding of a dynamic graph at snapshot :math:`t`.

* dyngraph2vecAE_: This method models the interconnection of nodes within and across time using multiple fully connected layers.

* dyngraph2vecRNN_: This method uses sparsely connected Long Short Term Memory (LSTM) networks to learn the embedding.

* dyngraph2vecAERNN_: This method uses a fully connected encoder to initially acquire low dimensional hidden representation and feeds this representation into LSTMs to capture network dynamics.


.. _Optimal SVD: https://www.kdd.org/kdd2016/papers/files/rfp0184-ouA.pdf
.. _Incremental SVD: https://www.merl.com/publications/docs/TR2006-059.pdf
.. _Rerun SVD: https://arxiv.org/abs/1711.09541
.. _Dynamic TRIAD: https://github.com/luckiezhou/DynamicTriad/blob/master/README.md
.. _AEalign: https://arxiv.org/abs/1805.11273
.. _dynGEM: https://arxiv.org/abs/1805.11273
.. _dyngraph2vecAE: https://arxiv.org/abs/1809.02657 
.. _dyngraph2vecRNN:  https://arxiv.org/abs/1809.02657
.. _dyngraph2vecAERNN: https://arxiv.org/abs/1809.02657
