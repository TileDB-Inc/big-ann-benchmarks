from __future__ import absolute_import
import numpy
import os
from benchmark.algorithms.base import BaseANN
from benchmark.datasets import DATASETS
from tiledb.vector_search.ingestion import ingest
from tiledb.vector_search.index import FlatIndex
import numpy as np
import multiprocessing


class TileDBFlat(BaseANN):
    def __init__(self, metric, arg):
        self._metric = metric

    def index_name(self, name):
        return f"data/tiledb_flat_{name}_{self._metric}"

    def query(self, X, n):
        if self._metric == 'angular':
            raise NotImplementedError()
        self.res =np.transpose(self.index.query(np.transpose(X), k=n, nthreads=multiprocessing.cpu_count()))

    def get_results(self):
        return self.res

    def fit(self, dataset):
        if DATASETS[dataset]().dtype == "uint8":
            source_type = "U8BIN"
        elif DATASETS[dataset]().dtype == "float32":
            source_type = "F32BIN"

        source_uri = DATASETS[dataset]().get_dataset_fn()
        self.index = ingest(index_type="FLAT",
                       array_uri=self.index_name(dataset),
                       source_uri=source_uri,
                       source_type=source_type,
                       size=DATASETS[dataset]().nb)

    def load_index(self, dataset):
        if not os.path.exists(self.index_name(dataset)):
            return False

        self.index = FlatIndex(self.index_name(dataset))
        return True

    def set_query_arguments(self):
        return

    def get_additional(self):
        return {}

    def __str__(self):
        return 'TileDBFlat'
