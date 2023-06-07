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


class FaissIVFFlatDisk(Faiss):
    def __init__(self, metric, n_list):
        self._n_list = n_list
        self._metric = metric

    def index_name(self, name):
        return f"data/ivf_flat_disk_{name}_{self._n_list}_{self._metric}"

    def fit(self, dataset):
        ds = DATASETS[dataset]()
        d = ds.d
        metric_type = (
                faiss.METRIC_L2 if ds.distance() == "euclidean" else
                faiss.METRIC_INNER_PRODUCT if ds.distance() in ("ip", "angular") else
                1/0
        )
        maxtrain = 100 * self._n_list
        print("setting maxtrain to %d" % maxtrain)
        # train on dataset
        print(f"getting first {maxtrain} dataset vectors for training")
        xt2 = next(ds.get_dataset_iterator(bs=maxtrain))

        self.quantizer = faiss.IndexFlatL2(xt2.shape[1])
        index = faiss.IndexIVFFlat(
            self.quantizer, xt2.shape[1], self._n_list, faiss.METRIC_L2)
        index.verbose = True
        index.quantizer.verbose = True
        index.cp.verbose = True
        index.train(xt2)
        print("writing trained index")
        faiss.write_index(index, self.index_name(dataset) + "_trained.index")
        print("adding vectors")

        t0 = time.time()
        add_bs = 10000000
        i0 = 0
        bno = 0
        for xblock in ds.get_dataset_iterator(bs=add_bs):
            index = faiss.read_index(self.index_name(dataset) + "_trained.index")
            i1 = i0 + len(xblock)
            print("  adding %d:%d / %d [%.3f s, RSS %d kiB] " % (
                i0, i1, ds.nb, time.time() - t0,
                faiss.get_mem_usage_kb()))
            index.add_with_ids(xblock, np.arange(i0, i1))

            print(f"storing block {bno} index")
            faiss.write_index(index, self.index_name(dataset) + "_block_%d.index" % bno)
            i0 = i1
            bno += 1

        print("  add in %.3f s" % (time.time() - t0))
        print('merging indexes')
        index = faiss.read_index(self.index_name(dataset) + "_trained.index")

        block_fnames = [
            self.index_name(dataset) + "_block_%d.index" % t
            for t in range(bno)
        ]
        merge_ondisk(index, block_fnames, self.index_name(dataset) + "_merged_index.ivfdata")
        print("storing merged index")
        faiss.write_index(index, self.index_name(dataset))

        self.index = index

    def load_index(self, dataset):
        if not os.path.exists(self.index_name(dataset)):
            return False

        print("Loading index")
        self.index = faiss.read_index(self.index_name(dataset))
        return True

    def set_query_arguments(self, n_probe):
        faiss.cvar.indexIVF_stats.reset()
        self._n_probe = n_probe
        self.index.nprobe = self._n_probe

    def get_additional(self):
        return {"dist_comps": faiss.cvar.indexIVF_stats.ndis +      # noqa
                faiss.cvar.indexIVF_stats.nq * self._n_list}

    def __str__(self):
        return 'FaissIVFFlatDisk(n_list=%d, n_probe=%d)' % (self._n_list,
                                                    self._n_probe)
