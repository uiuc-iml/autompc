from abc import ABC, abstractmethod
from pathlib import Path
import pickle
import os
import uuid


class DataStore(ABC):
    """A DataStore is responsible for wrapping and unwrapping data to provide
    to parallel workers."""
    @abstractmethod
    def wrap(self,data):
        return data

class WrappedData(ABC):
    @abstractmethod
    def unwrap(self):
        pass
    @abstractmethod
    def cleanup(self):
        pass

class MemoryWrappedData(WrappedData):
    def __init__(self, data):
        self._data = data

    def unwrap(self):
        return self._data

    def cleanup(self):
        pass

class MemoryDataStore(DataStore):
    """A DataStore that just stores data in memory.  Used when workers have access
    to shared memory."""
    def __init__(self):
        pass
        
    def wrap(self, data):
        if isinstance(data, MemoryWrappedData):
            return data
        else:
            return MemoryWrappedData(data)


class FileWrappedData(WrappedData):
    def __init__(self, filename, do_cleanup=True):
        self._filename = filename
        self._do_cleanup = do_cleanup

    def unwrap(self):
        with open(self._filename, "rb") as f:
            return pickle.load(f)

    def cleanup(self):
        if not self._do_cleanup: return
        if os.path.exists(self._filename):
            os.remove(self._filename)


class FileDataStore(DataStore):
    """A DataStore that pickles and unpickles data.  Used when workers have access
    to a shared filesystem but not shared memory.
    
    If do_cleanup=True, items are deleted after temporary use.  Otherwise, they
    are left in the filesystem.
    """
    def __init__(self, data_dir, do_cleanup=True):
        self._data_dir = Path(data_dir)
        self._do_cleanup = do_cleanup
    
    def set_dir(self, data_dir):
        self._data_dir = Path(data_dir)

    def wrap(self, data):
        if isinstance(data, FileWrappedData):
            return data
        else:
            object_id = uuid.uuid4().hex
            filename = self._data_dir / f"{object_id}.pkl"
            with open(filename, "wb") as f:
                pickle.dump(data, f)
            return FileWrappedData(filename,self._do_cleanup)

