
## gbox11传文件到zj和amx上
```
cd /data3/liangmin.wu/workspace/opencompass

sudo scp -r -i /home/yechen/id_ecdsa_zj * hbuser1@124.152.91.205:/share/home/hbuser1/liangmwu/opencompass

scp -r -P 9090 * root@218.91.113.234:/data/liangmin.wu/opencompass
```

## 传模型文件
```
紫金云到gbox11
scp -r -P 30022 -i /share/home/hbuser1/.ssh/id_rsa bloom-7b-sft-smart-73500 liangmin.wu@121.46.232.162:/mnt/nfs/algo/wlm

amx到gbox11
scp -r -P 30022 tigerbot-7b-sft-simple-6700 liangmin.wu@121.46.232.162:/mnt/nfs/algo/wlm

gbox11到紫金云
sudo scp -r -i /home/yechen/id_ecdsa_zj checkpoint-28900 hbuser1@124.152.91.205:/share/home/hbuser1/liangmwu/models/

gbox11到amx
scp -r -P 9090 checkpoint-28900 root@218.91.113.234:/data/liangmin.wu/models/

gbox11到gbox8共享nas盘:/mnt/nfs/algo/wlm，其对应gbox7：/mnt/nfs-algo/wlm
```

## 启动评测任务
```
zj:
export MODULEPATH=/share/home/deploy/apps/modulefiles:$MODULEPATH
module load anaconda/2021.11
module load cuda/11.6.0

amx:
module load anaconda/2021.11
module load cuda/11.6
module load gcc/9.4

conda activate test

11: cd /data3/liangmin.wu/workspace/opencompass
zj: cd /share/home/hbuser1/liangmwu/opencompass
amx: cd /data/liangmin.wu/opencompass

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run.py configs/eval_llama_7b.py --max-partition-size 2000

opencompass文档地址：https://opencompass.readthedocs.io/en/latest/
```