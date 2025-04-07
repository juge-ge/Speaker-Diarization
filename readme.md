# 使用指南
## 1.pip install requirements 
   下载相关依赖
## 2.下载相关模型 
 speech_campplus_sv_zh-cn_16k-common
 speech_fsmn_vad_zh-cn-16k-common-pytorch
 speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
 punc_ct-transformer_zh-cn-common-vocab272727-pytorch 
  这些模型在魔塔社区中下载好
## 3.代码中修改模型路径的问题（自己在AudioSeparation.py 修改，我这里是硬编码的所以需要修改）
## 4.服务器部署
### 4.1进入虚拟环境    source /home/data/jdssy_liy/my_venv/bin/activate
### 4.2启动该程序   uvicorn main:app --host 0.0.0.0 --port 8090运行命令
### 4.3使用结束后退出该环境   deactivate 
