

## installation

```shell
pip install git+https://github.com/mindspore-lab/mindone.git
```


## (testing) run inference

```shell
# run vae/captioner server with single-card (Ascend910*)
ASCEND_RT_VISIBLE_DEVICES=0 python api/call_remote_server.py --model_dir where_you_download_dir &


# run main process with multi-cards (Ascend910*)
parallel=4
url='127.0.0.1'
model_dir=where_you_download_dir

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 msrun --bind_core=True --worker_num=$parallel --local_worker_num=$parallel --master_port=9000 --log_dir=outputs/parallel_logs python \
run_parallel.py --model_dir $model_dir --vae_url $url --caption_url $url  --ulysses_degree $parallel --prompt "一名宇航员在月球上发现一块石碑，上面印有“MindSpore”字样，闪闪发光" --infer_steps 50  --cfg_scale 9.0 --time_shift 13.0
```
