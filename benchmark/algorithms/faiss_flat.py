from __future__ import absolute_import
import numpy as np
#import sys
#sys.path.append("install/lib-faiss")  # noqa
import numpy
import sklearn.preprocessing
import ctypes
import faiss
from faiss.contrib.ondisk import merge_ondisk
import os
import time
from benchmark.algorithms.base import BaseANN
from benchmark.datasets import DATASETS

def knn_search_batched(index, xq, k, bs):
    D, I = [], []
    for i0 in range(0, len(xq), bs):
        Di, Ii = index.search(xq[i0:i0 + bs], k)
        D.append(Di)
        I.append(Ii)
    return np.vstack(D), np.vstack(I)


class Faiss(BaseANN):
    def query(self, X, n):
        if self._metric == 'angular':
            X /= numpy.linalg.norm(X)
        self.res = self.index.search(X.astype(numpy.float32), n)

    def get_results(self):
        D, I = self.res
        return I
    #        res = []
    #        for i in range(len(D)):
    #            r = []
    #            for l, d in zip(L[i], D[i]):
    #                if l != -1:
    #                    r.append(l)
    #            res.append(r)
    #        return res

class FaissFlat(Faiss):
    def __init__(self, metric, n_list):
        self._n_list = n_list
        self._metric = metric

    def index_name(self, name):
        return f"data/flat_{name}_{self._n_list}_{self._metric}"

    def fit(self, dataset):
        ds = DATASETS[dataset]()
        d = ds.d
        metric_type = (
            faiss.METRIC_L2 if ds.distance() == "euclidean" else
            faiss.METRIC_INNER_PRODUCT if ds.distance() in ("ip", "angular") else
            1 / 0
        )
        index = faiss.index_factory(d, "Flat", metric_type)

        t0 = time.time()
        add_bs = 10000000
        i0 = 0
        for xblock in ds.get_dataset_iterator(bs=add_bs):
            i1 = i0 + len(xblock)
            print("  adding %d:%d / %d [%.3f s, RSS %d kiB] " % (
                i0, i1, ds.nb, time.time() - t0,
                faiss.get_mem_usage_kb()))
            index.add(xblock)
            i0 = i1

        print("  add in %.3f s" % (time.time() - t0))
        print('storing')
        faiss.write_index(index, self.index_name(dataset))

        self.index = index
        self.ps = faiss.ParameterSpace()
        self.ps.initialize(self.index)

    def load_index(self, dataset):
        if not os.path.exists(self.index_name(dataset)):
            return False

        print("Loading index")
        self.index = faiss.read_index(self.index_name(dataset))
        self.ps = faiss.ParameterSpace()
        self.ps.initialize(self.index)
        return True

    def set_query_arguments(self, n_probe):
        return

    def get_additional(self):
        return {}

    def __str__(self):
        return 'FaissFlatMem'


