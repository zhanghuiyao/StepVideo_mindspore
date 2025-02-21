import mindspore as ms
from mindspore import Tensor
import numpy as np
import threading
import pickle

lock = threading.Lock()

_tensor = np.random.randn(1000, 1000)


def run(self):
        with lock:
            # try:
            #
            # except Exception as e:
            #     print("Caught Exception: ", e)
            #     return Response(e)
            

            _tensor = Tensor(_tensor).to(ms.bfloat16)
            _tensor *= 2.0

            response = pickle.dumps(_tensor.to(ms.float32).asnumpy())

            return response


run()
