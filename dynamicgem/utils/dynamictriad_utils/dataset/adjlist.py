from __future__ import print_function

import networkx as nx

from dynamicgem.utils.dynamictriad_utils.dataset.dataset_utils import DatasetBase
from dynamicgem.utils.dynamictriad_utils import mygraph_utils as mgutils


class Dataset(DatasetBase):
    @property
    def inittime(self):
        return 0 

    def __init__(self, datafn, localtime, nsteps, stepsize, stepstride, offset=0, dataname=None):
        self.datafn = datafn
        self.__datadir = datafn 

        DatasetBase.__init__(self, datafn, localtime, nsteps, stepsize, stepstride, offset, dataname)

        self.vertices = None

    @property
    def name(self):
        return "adjlist"

    # required by Timeline
    def _time2unit(self, tm):
        return int(float(tm))

    def _unit2time(self, unit):
        return str(unit)

    def __check_vertices(self, vs):
        
        assert len(vs) == len(self.vertices), (len(vs), len(self.vertices))
        for i in range(len(vs)):
            assert vs[i] == self.vertices[i], (i, vs[i], self.vertices[i])

    # required by DyanmicGraph
    def _load_unit_graph(self, tm):
        tm = self._time2unit(tm)
        fn = "{}/{}".format(self.__datadir, tm)
        g = mgutils.load_adjlist(fn)
        if self.vertices is None:
            self.vertices = list(g.nodes())
            self.vertices.sort()
        else:
            try:
                nodes= list(g.nodes())
                nodes.sort()
                self.__check_vertices(nodes)  # ensure all graphs share a same set of vertices
            except AssertionError as e:
                if hasattr(e, 'message'):
                    msg = e.message
                else:
                    msg = e
                import pdb
                pdb.set_trace()
                raise RuntimeError("Vertices in graph file {} are not compatible with files already loaded: {}"
                                   .format(fn, msg))
        return g

    def _merge_unit_graphs(self, graphs, curstep):
        curunit = self._time2unit(self.step2time(curstep))
        print("merging graph from year {} to {}".format(curunit, curunit + self.stepsize - 1))

        ret = nx.DiGraph()
        for g in graphs:
            ret = nx.compose(ret,g)

        return ret

    def archive(self, name=None):
        ar = super(Dataset, self).archive()
        return ar

    def load_archive(self, ar, copy=False, name=None):
        super(Dataset, self).load_archive(ar, copy=copy)


