from __future__ import print_function

import itertools
from collections import defaultdict
import numpy as np
import dynamicgem.utils.dynamictriad_utils.utils
try:
    from itertools import izip
except:
    izip = zip    

__gtutils_debug = True


def type2python(tp):
    if tp == 'string':
        return str
    elif tp in ['short', 'int', 'long', 'long long', 'int16_t', 'int32_t', 'int64_t']:
        return int
    elif tp in ['double', 'float']:
        return float
    else:
        raise TypeError("Unknown type {}".format(tp))


def python2type(tp):
    if tp == int:
        return 'long'
    elif tp == str:
        return 'string'
    elif tp == float:
        return 'double'
    else:
        raise TypeError("Unsupported python type {}".format(tp))


def graph_summary(g):
    return "nsize: {}, esize: {}, weight_type: {}, name_type: {}, directed: {}". \
        format(g.num_vertices(), g.num_edges(), g.ep['weight'].python_value_type(), g.vp['name'].python_value_type(), g.is_directed())


def load_mygraph(fn, directed=True, nodename=None, nametype='string', convert_to=None):
    if nodename is not None:
        raise NotImplementedError("given nodename is not supported in load_mygraph")

    data = utils.open_datasrc(fn).read().split('\n')[1:]  # skip the first line counting number of vertices
    if data[-1] == '' and len(data) % 2 == 1:  # try to fix length problem
        data = data[:-1]
    assert len(data) % 2 == 0, "{} {}".format(len(data), str(data))

    vertices = data[::2]
    if nametype == 'string':
        vertices = [v[v.find('@') + 1:] for v in vertices]
    elist = data[1::2]

    def str2elist(s):
        arr = s.split()[1:]  # discard edge cnt at the beginning of this line
        if nametype == 'string':
            evertices = [v[v.find('@') + 1:] for v in arr[::2]]
        else:
            evertices = arr[::2]
        return izip(evertices, arr[1::2])

    vid2elist = utils.KeyDefaultDict(lambda x: str2elist(elist[x]))
    return load_mygraph_core(vertices, vid2elist, directed=directed, nametype=nametype, weighttype='float',
                             convert_to=convert_to, check=True)

# load mygraph format, which is always directed
# if directed=False, the directed mygraph format is converted to non-directed if possible
def load_graph(fn, fmt='mygraph', directed=None, nodename=None, nametype='string', convert_to=None):
    if fmt == 'mygraph':
        return load_mygraph(fn, directed=directed, nodename=nodename, nametype=nametype, convert_to=convert_to)
    elif fmt == 'edgelist':
        return load_edge_list(fn, directed=directed, nodename=nodename, nametype=nametype, convert_to=convert_to)
    else:
        raise NotImplementedError


def save_graph(g, fn, fmt='adjlist', weight=None):
    if fmt == 'adjlist':
        save_adjlist(g, fn, weight=weight)
    elif fmt == 'edgelist':
        save_edgelist(g, fn, weight=weight)
    elif fmt == 'TNE':
        save_TNE(g, fn, weight=weight)
    else:
        raise RuntimeError("Unkonwn graph format {}".format(fmt))


def save_adjlist(g, fn, weight=None):
    fh = open(fn, 'w')
    nodeidx = list(sorted([int(v) for v in g.vertices()]))
    for i in nodeidx:
        if g.is_directed():
            nbrs = [int(n) for n in g.vertex(i).out_neighbours()]
        else:
            nbrs = [int(n) for n in g.vertex(i).all_neighbours()]
        if weight is None:
            strnbr = ' '.join([str(n) for n in nbrs])
        else:
            w = [weight[g.edge(i, n)] for n in nbrs]
            assert len(nbrs) == len(w)
            strnbr = ' '.join([str(e) for e in itertools.chain.from_iterable(zip(nbrs, w))])
        print("{} {}".format(i, strnbr), file=fh)
    fh.close()


def save_edgelist(g, fn, weight=None):
    fh = open(fn, 'w')
    # if we don't care about order
    for e in g.edges():
        if weight is None:
            print("{} {}".format(int(e.source()), int(e.target())), file=fh)
        else:
            print("{} {} {}".format(int(e.source()), int(e.target()), weight[e]), file=fh)
    fh.close()


# TNE format is an undirected format defined here
# https://github.com/linhongseba/Temporal-Network-Embedding
def save_TNE(g, fn, weight=None):
    assert not g.is_directed()
    
    fh = open(fn, 'w')
    # in order to speed up edge access
    edge_cache = {}
    
    if weight is None:  # in this format, a weight must be given
        weight = defaultdict(lambda x: 1.0)

    for e in g.edges():
        isrc, itgt = int(e.source()), int(e.target())
        # isrc, itgt = min(isrc, itgt), max(isrc, itgt)
        edge_cache[(isrc, itgt)] = weight[e]
        edge_cache[(itgt, isrc)] = weight[e]
    # w = None

    print(g.num_vertices(), file=fh)
    for i in range(g.num_vertices()):
        outnbr = [int(v) for v in list(g.vertex(i).out_neighbours())]
        #if len(outnbr) == 0:  # for debug
        #    continue
        outnbr = list(sorted(outnbr))  # in ascending order
        w = [edge_cache[(i, v)] for v in outnbr]
        fields = ['{},{}'.format(i, len(outnbr))] + ["{},{}".format(a, b) for a, b in zip(outnbr, w)]
        print(':'.join(fields), file=fh)
    fh.close()
