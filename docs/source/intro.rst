Introduction
===============

Graph embedding methods aim to represent each node of a graph in a low-dimensional vector space while preserving certain graph's properties. Such  methods  have been used to tackle many real-world tasks, e.g.,  friend recommendation in social networks, genome classification in biology networks, and visualizing topics in research using collaboration networks.

More recently, much attention has been devoted to extending static embedding techniques to capture graph evolution. Applications include  temporal link prediction, and understanding the evolution dynamics of network communities. Most methods aim to efficiently update the embedding of the graph at each time step using information from previous embedding and from changes in the graph. Some methods also capture the temporal patterns of the evolution in the learned embedding, leading to improved link prediction performance.

In this library, we present an easy-to-use toolkit of state-of-the-art dynamic graph embedding  methods.  **dynamicgem** implements methods which can handle the evolution of networks over time. Further, we provide a comprehensive framework to evaluate the methods by providing support for four tasks on dynamic networks: graph reconstruction, static and temporal link prediction, node classification, and temporal visualization. For each task, our framework includes multiple evaluation metrics to quantify the performance of the methods. We further share synthetic and real networks for evaluation. Thus, our library is an end-to-end framework to experiment with dynamic graph embedding.  

