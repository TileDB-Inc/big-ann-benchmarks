from __future__ import absolute_import
import numpy
import os
from benchmark.algorithms.base import BaseANN
from benchmark.datasets import DATASETS
from tiledb.vector_search.ingestion import ingest, IVFFlatIndex
import numpy as np
import multiprocessing


class TileDBIVFFlat(BaseANN):
    def __init__(self, metric, n_list):
        self._n_list = n_list
        self._metric = metric

    def index_name(self, name):
        return f"data/tiledb_{name}_{self._n_list}_{self._metric}"

    def query(self, X, n):
        if self._metric == 'angular':
            raise NotImplementedError()
        self.res =np.transpose(self.index.query(np.transpose(X), k=n, nthreads=multiprocessing.cpu_count(), nprobe=int(self._n_probe)))

    def get_results(self):
        return self.res

    def fit(self, dataset):
        if DATASETS[dataset]().dtype == "uint8":
            source_type = "U8BIN"
        elif DATASETS[dataset]().dtype == "float32":
            source_type = "F32BIN"

        source_uri = DATASETS[dataset]().get_dataset_fn()
        self.index = ingest(index_type="IVF_FLAT",
                       array_uri=self.index_name(dataset),
                       source_uri=source_uri,
                       source_type=source_type,
                       size=DATASETS[dataset]().nb,
                       training_sample_size=100 * self._n_list,
                       partitions=self._n_list)

    def load_index(self, dataset):
        if not os.path.exists(self.index_name(dataset)):
            return False

        self.index = IVFFlatIndex(self.index_name(dataset))
        return True

    def set_query_arguments(self, n_probe):
        self._n_probe = n_probe

    def get_additional(self):
        return {}

    def __str__(self):
        return 'TileDBIVFFlat(n_list=%d)' % self._n_list
