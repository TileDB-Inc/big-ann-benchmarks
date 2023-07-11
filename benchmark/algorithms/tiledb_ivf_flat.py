from __future__ import absolute_import
import numpy
import os
import tiledb
from benchmark.algorithms.base import BaseANN
from benchmark.datasets import DATASETS
from tiledb.vector_search.ingestion import ingest
from tiledb.vector_search.index import IVFFlatIndex
import numpy as np
import multiprocessing


class TileDBIVFFlat(BaseANN):
    def __init__(self, metric, index_params):
        self._index_params = index_params
        self._uri_prefix = index_params.get("uri_prefix", "file://"+os.getcwd()+"/data")
        self._config = index_params.get("config", {})
        self._n_list = int(index_params.get("n_list", "-1"))
        self._mem_budget = int(index_params.get("mem_budget", "-1"))
        self._metric = metric
        try:
            tiledb.default_ctx(self._config)
        except tiledb.TileDBError:
            pass

    def index_name(self, name):
        return f"{self._uri_prefix}/tiledb_ivf_flat_{name}_{self._n_list}_{self._metric}"

    def query(self, X, n):
        try:
            tiledb.default_ctx(self._config)
        except tiledb.TileDBError:
            pass
        if self._metric == 'angular':
            raise NotImplementedError()
        self.res =np.transpose(self.index.query(np.transpose(X), k=n, nthreads=multiprocessing.cpu_count(), nprobe=self._n_probe, use_nuv_implementation=self._nuv))

    def get_results(self):
        return self.res

    def fit(self, dataset):
        if DATASETS[dataset]().dtype == "uint8":
            source_type = "U8BIN"
        elif DATASETS[dataset]().dtype == "float32":
            source_type = "F32BIN"
        maxtrain = min(50 * self._n_list, DATASETS[dataset]().nb)
        source_uri = DATASETS[dataset]().get_dataset_fn()
        print(self._config)
        self.index = ingest(index_type="IVF_FLAT",
                       array_uri=self.index_name(dataset),
                       source_uri=source_uri,
                       source_type=source_type,
                       size=DATASETS[dataset]().nb,
                       training_sample_size=maxtrain,
                       partitions=self._n_list,
                       input_vectors_per_work_item=100000000,
                       config=self._config
                     )

    def load_index(self, dataset):
        print(self._config)
        try:
            tiledb.default_ctx(self._config)
        except tiledb.TileDBError:
            pass
        vfs = tiledb.VFS()
        if not vfs.is_dir(self.index_name(dataset)):
            return False
        if DATASETS[dataset]().dtype == "uint8":
            self.index = IVFFlatIndex(self.index_name(dataset), np.uint8, memory_budget=self._mem_budget)
        elif DATASETS[dataset]().dtype == "float32":
            self.index = IVFFlatIndex(self.index_name(dataset), np.float32, memory_budget=self._mem_budget)
        return True

    def set_query_arguments(self, query_params):
        print(query_params)
        self._query_params = query_params
        self._n_probe = int(query_params.get("n_probe", "1"))
        self._nuv = query_params.get("nuv", "False") == "True"
        print(self._nuv)

    def get_additional(self):
        return {}

    def __str__(self):
        return 'TileDBIVFFlat(n_list=%d, n_probe=%d, mem_budget=%d, nuv=%d)' % (self._n_list, self._n_probe, self._mem_budget, self._nuv)
