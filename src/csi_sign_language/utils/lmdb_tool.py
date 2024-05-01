import lmdb
import numpy as np
import pickle

def store_numpy_array(env, key, array):
    # Serialize NumPy array to bytes
    array_bytes = array.tobytes()

    # Open a transaction and store the array in the LMDB database
    with env.begin(write=True) as txn:
        txn.put(key.encode('utf-8'), array_bytes)

def retrieve_numpy_array(env, key, shape, dtype):
    # Open a transaction and retrieve the array from the LMDB database
    with env.begin() as txn:
        array_bytes = txn.get(key.encode('utf-8'))

    # Deserialize bytes to NumPy array
    array = np.frombuffer(array_bytes, dtype=dtype).reshape(shape)
    return array

def store_data(env, key, value):
    with env.begin(write=True) as txn:
        key = key.encode('utf-8')
        value = pickle.dumps(value)
        txn.put(key, value)

def retrieve_data(env, key):
    with env.begin() as txn:
        key = key.encode('utf-8')
        value = txn.get(key)
        if value is not None:
            return pickle.loads(value)
        else:
            raise Exception('key not exist in lmdb dataset')
            
