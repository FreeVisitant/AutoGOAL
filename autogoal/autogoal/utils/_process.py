import multiprocessing
import warnings
import psutil
import signal
import os
import logging
import platform
from numpy.core._exceptions import _ArrayMemoryError
import dill
# from autogoal.kb import Pipeline
from pathlib import Path
import sys

if platform.system() == "Linux":
    import resource

TEMPORARY_DATA_PATH = Path.home() / ".autogoal" / "automl"

def ensure_temporary_data_path():
    global TEMPORARY_DATA_PATH
    os.makedirs(TEMPORARY_DATA_PATH, exist_ok=True)
    
def delete_temporary_data_path():
    os.remove(TEMPORARY_DATA_PATH)
    os.removedirs(TEMPORARY_DATA_PATH)

from autogoal.utils import Mb

logger = logging.getLogger("autogoal")

IS_MP_CUDA_INITIALIZED = False
def initialize_cuda_multiprocessing():
    try:
        import torch.multiprocessing as mp
        global IS_MP_CUDA_INITIALIZED
        if not IS_MP_CUDA_INITIALIZED:
            mp.set_start_method('spawn', force=True)
            print("initialized multiprocessing")
            IS_MP_CUDA_INITIALIZED = True
    except:
        return

def is_cuda_multiprocessing_enabled():
    import torch.multiprocessing as mp
    return mp.get_start_method() == 'spawn'

class RestrictedWorker:
    def __init__(self, function, timeout: int, memory: int):
        self.function = function
        self.timeout = timeout
        self.memory = memory
        signal.signal(signal.SIGXCPU, alarm_handler)

    def _restrict(self):
        if platform.system() == "Linux":
            msoft, mhard = resource.getrlimit(resource.RLIMIT_DATA)
            csoft, chard = resource.getrlimit(resource.RLIMIT_CPU)
            used_memory = self.get_used_memory()

            if self.memory and self.memory > (used_memory + 500 * Mb):
                # memory may be restricted
                
                self.memory = min(self.memory, sys.maxsize)
                resource.setrlimit(resource.RLIMIT_DATA, (self.memory, mhard))
            else:
                warnings.warn("Cannot restrict memory")

            if self.timeout:
                # time may be restricted
                resource.setrlimit(resource.RLIMIT_CPU, (self.timeout, chard))
            else:
                warnings.warn("Cannot restrict cpu time")

    def _restricted_function(self, result_bucket, *args, **kwargs):
        try:
            self._restrict()
            result = self.function(*args, **kwargs)
            result_bucket["result"] = result
        except _ArrayMemoryError as e:
            result_bucket["result"] = _ArrayMemoryError(e.shape, e.dtype)
        except Exception as e:
            result_bucket["result"] = e

    def run_restricted(self, *args, **kwargs):
        """
        Executes a given function with restricted amount of
        CPU time and RAM memory usage
        """
        manager = multiprocessing.Manager()
        result_bucket = manager.dict()

        rprocess = multiprocessing.Process(
            target=self._restricted_function, args=[result_bucket, args, kwargs]
        )

        rprocess.start()
        rprocess.join(timeout=self.timeout)

        if rprocess.is_alive():
            rprocess.terminate()
            raise TimeoutError(
                f"Process took more than {self.timeout} seconds to complete and has been terminated."
            )

        result = result_bucket["result"]

        if isinstance(result, Exception):  # Exception ocurred
            raise result

        return result

    def get_used_memory(self):
        """
        returns the amount of memory being used by the current process
        """
        process = psutil.Process(os.getpid())
        return process.memory_info().rss

    def __call__(self, *args, **kwargs):
        return self.run_restricted(*args, **kwargs)


def alarm_handler(*args):
    raise TimeoutError("process %d got to time limit" % os.getpid())


class RestrictedWorkerByJoin(RestrictedWorker):
    def __init__(self, function, timeout: int, memory: int):
        self.function = function
        self.timeout = timeout
        self.memory = memory

    def _restrict(self):
        if platform.system() == "Linux":
            _, mhard = resource.getrlimit(resource.RLIMIT_AS)
            used_memory = self.get_used_memory()

            if self.memory is None:
                return

            if self.memory > (used_memory + 50 * Mb):
                # memory may be restricted
                self.memory = min(self.memory, sys.maxsize)
                logger.info("💻 Restricting memory to %s" % self.memory)
                try: 
                    resource.setrlimit(resource.RLIMIT_DATA, (self.memory, mhard))
                except Exception as e:
                    logger.info("💻 Failed to restrict memory to %s" % self.memory)
                    raise e
            else:
                raise Exception (
                    "Cannot restrict memory to %s < %i"
                    % (self.memory, used_memory + 50 * Mb)
                )

    def run_restricted(self, *args, **kwargs):
        """
        Executes a given function with restricted amount of
        CPU time and RAM memory usage
        """
        manager = multiprocessing.Manager()
        result_bucket = manager.dict()

        rprocess = multiprocessing.Process(
            target=self._restricted_function, args=[result_bucket, *args], kwargs=kwargs
        )

        rprocess.start()
        rprocess.join(self.timeout)

        if rprocess.exitcode == 0:
            result = result_bucket["result"]
        else:
            rprocess.terminate()
            raise TimeoutError(
                f"Exceded allowed time for execution. Any restricted function should end its excution in a timespan of {self.timeout} seconds."
            )

        if isinstance(result, Exception):  # Exception ocurred
            raise result

        return result

class RestrictedWorkerDiskSerializableByJoin(RestrictedWorkerByJoin):
    def __init__(self, function, timeout: int, memory: int):
        self.function = dill.dumps(function)
        self.timeout = timeout
        self.memory = memory
    
    def _restricted_function(self, result_bucket, *args, **kwargs):
        try:
            self._restrict()
            
            from autogoal.kb import Pipeline
            algorithms, types = Pipeline.load_algorithms(TEMPORARY_DATA_PATH)
            pipeline = Pipeline(algorithms, types)
            
            function = dill.loads(self.function)
            result = function(pipeline)
            result_bucket["result"] = result
        except _ArrayMemoryError as e:
            result_bucket["result"] = _ArrayMemoryError(e.shape, e.dtype)
        except Exception as e:
            result_bucket["result"] = e
    
    def run_restricted(self, pipeline, *args, **kwargs):
        """
        Executes a given function with restricted amount of
        CPU time and RAM memory usage
        """
        manager = multiprocessing.Manager()
        result_bucket = manager.dict()

        # ensure the directory to where the pipeline is going 
        # to be exported exists 
        ensure_temporary_data_path()
        
        global TEMPORARY_DATA_PATH
        pipeline.save_algorithms(TEMPORARY_DATA_PATH)

        rprocess = multiprocessing.Process(
            target=self._restricted_function, args=[result_bucket, TEMPORARY_DATA_PATH, *args], kwargs=kwargs
        )

        rprocess.start()
        rprocess.join(self.timeout)

        if rprocess.exitcode == 0:
            result = result_bucket["result"]
        else:
            rprocess.terminate()
            raise TimeoutError(
                f"Exceded allowed time for execution. Any restricted function should end its excution in a timespan of {self.timeout} seconds."
            )

        if isinstance(result, Exception):  # Exception ocurred
            raise result

        # load trained pipeline
        from autogoal.kb import Pipeline
        algorithms, _ = Pipeline.load_algorithms(TEMPORARY_DATA_PATH)
        pipeline.algorithms = algorithms
        
        # delete all generated temp files
        delete_temporary_data_path()
        return result


class RestrictedWorkerWithState(RestrictedWorkerByJoin):
    def __init__(self, function, timeout: int, memory: int):
        self.function = function
        self.timeout = timeout
        self.memory = memory

    def _restricted_function(self, result_bucket, arguments_bucket, *args, **kwargs):
        try:
            instance = arguments_bucket["instance"]
            self._restrict()
            result = self.function(instance, *args, **kwargs)
            result_bucket["result"] = result
            result_bucket["instance"] = instance
        except _ArrayMemoryError as e:
            result_bucket["result"] = _ArrayMemoryError(e.shape, e.dtype)
        except Exception as e:
            result_bucket["result"] = e

    def run_restricted(self, instance, *args, **kwargs):
        """
        Executes a given function with restricted amount of
        CPU time and RAM memory usage
        """
        manager = multiprocessing.Manager()
        result_bucket = manager.dict()
        arguments_bucket = manager.dict(
            {
                "instance": instance,
            }
        )

        rprocess = multiprocessing.Process(
            target=self._restricted_function,
            args=[result_bucket, arguments_bucket, *args],
            kwargs=kwargs,
        )

        rprocess.start()
        rprocess.join(self.timeout)

        if rprocess.exitcode == 0:
            result = result_bucket.get("result"), result_bucket.get("instance")
        else:
            rprocess.kill()
            raise TimeoutError(
                f"Exceded allowed time for execution. Any restricted function should end its excution in a timespan of {self.timeout} seconds."
            )

        if isinstance(result, Exception):  # Exception ocurred
            raise result

        return result
