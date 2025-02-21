import mindspore as ms
import numpy as np
import threading


lock = threading.Lock()

_tensor = ms.Tensor

def get(self):
        with lock:
            # try:
            #
            # except Exception as e:
            #     print("Caught Exception: ", e)
            #     return Response(e)
            
            feature = pickle.loads(request.get_data())
            feature['api'] = 'vae'
        
            feature = {k:v for k, v in feature.items() if v is not None}
            video_latents = self.vae_pipeline.decode(**feature)

            response = pickle.dumps(video_latents)

            return Response(response)

