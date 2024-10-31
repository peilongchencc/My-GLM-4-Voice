# My-GLM-4-Voice

ubuntu 系统下 [GLM-4-Voice](https://github.com/THUDM/GLM-4-Voice) 部署经验分享。

- [My-GLM-4-Voice](#my-glm-4-voice)
  - [笔者配置:](#笔者配置)
  - [程序运行:](#程序运行)
    - [代码拉取:](#代码拉取)
    - [创建/激活虚拟环境:](#创建激活虚拟环境)
    - [安装依赖项(很耗时):](#安装依赖项很耗时)
      - [修改torch版本:](#修改torch版本)
      - [常规依赖项安装:](#常规依赖项安装)
      - [安装音频处理辅助库](#安装音频处理辅助库)
  - [模型下载:](#模型下载)
  - [主程序运行:](#主程序运行)
    - [启动模型服务:](#启动模型服务)
    - [启动 web 服务:](#启动-web-服务)
    - [测试效果:](#测试效果)
  - [附录: 笔者的环境](#附录-笔者的环境)


## 笔者配置:

| 系统          | 显卡              | CUDA Version | 模型初始化显存占用 |
|---------------|-------------------|--------------|--------------|
| Ubuntu 22.04  | A100-PCIE-40GB * 1 | 12.2         | 18745MiB    |


## 程序运行:

### 代码拉取:

```bash
git clone --recurse-submodules https://github.com/THUDM/GLM-4-Voice
cd GLM-4-Voice
```

🚨笔者于2024-10-31拉取的GLM-4-Voice代码。

### 创建/激活虚拟环境:

```bash
conda create -n glm_voice python==3.11
conda activate glm_voice
```

### 安装依赖项(很耗时):

#### 修改torch版本:

GLM-4-Voice中`requirements.txt`指定的torch版本为 `torch 2.3.0`，已知 `torch 2.3.0` 在卷积操作时无法正确使用cuDNN，报错信息如下:

```log
/home/vipuser/miniconda3/envs/glm_voice/lib/python3.10/site-packages/torch/nn/modules/conv.py:306: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
  return F.conv1d(input, weight, bias, self.stride,
```

pytorch官方github相关issue可查看[这里](https://github.com/pytorch/pytorch/issues/121834)。

解决方案为去除版本号，让pip自动查找符合依赖的版本:

```txt
torch
torchaudio
```

#### 常规依赖项安装:

```bash
pip install -r requirements.txt
pip install accelerate
# 以不安装依赖的方式安装matcha-tts(作者少写了这个库)
pip install --no-deps matcha-tts
```

如果你使用的是Linux系统(例如ubuntu 22.04)，运行`pip install -r requirements.txt`后终端提示:

```log
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple/
Ignoring onnxruntime: markers 'sys_platform == "darwin" or sys_platform == "windows"' don't match your environment
```

不必惊慌，这表示这个依赖项只在 sys_platform 为 darwin（macOS）或 windows 时安装。`onnxruntime-gpu==1.16.0; sys_platform == 'linux'`会自动帮你安装Linux所需的onnx配置。

#### 安装音频处理辅助库

```bash
sudo apt update
apt install ffmpeg
# 查看安装的 ffmpeg 版本
ffmpeg -version
```

如果不安装`ffmpeg`，运行`web_demo.py`会提示缺少`ffprobe`而引发报错。

> ffprobe 是 FFmpeg 工具包的一部分，专门用于分析媒体文件的元数据，而 ffmpeg 是用于实际转换或处理音视频文件的工具。


## 模型下载:

由于笔者使用Git LFS从HF和MS均拉取不到模型，故采用利用代码从MS拉取模型。方式如下:

```bash
cd model_dl
# 下载GLM-4-Voice-Tokenizer
python dl_glm_voice_tokenizer.py
# 下载GLM-4-Voice-Decoder
python dl_glm_voice_decoder.py
# 下载GLM-4-Voice-9B
python dl_glm_voice.py
```

模型均保存在 `~/.cache/modelscope/hub/ZhipuAI/` 目录下。


## 主程序运行:

### 启动模型服务:

```bash
python model_server.py --model-path ~/.cache/modelscope/hub/ZhipuAI/glm-4-voice-9b
```

### 启动 web 服务:

```bash
python web_demo.py --tokenizer-path ~/.cache/modelscope/hub/ZhipuAI/glm-4-voice-tokenizer --flow-path ~/.cache/modelscope/hub/ZhipuAI/glm-4-voice-decoder --model-path ~/.cache/modelscope/hub/ZhipuAI/glm-4-voice-9b
```

### 测试效果:

- 上传`example_wav`文件夹下文件，点击Submit。
- 点击🎙️录制音频，点击Submit。(生成录制文件需要一定的时间)

web效果图如下:

> 音频含义是笔者自己标注的。

![](./docs/wav_test.png)

终端效果图如下:

![](./docs/终端效果.png)


## 附录: 笔者的环境

这里贴一下笔者按照上面的讲解运行程序后，使用 `pip freeze > requirements.txt` 得到的结果:

```txt
absl-py==2.1.0
accelerate==1.0.1
addict==2.4.0
aiofiles==23.2.1
aiohappyeyeballs==2.4.3
aiohttp==3.10.10
aiosignal==1.3.1
aliyun-python-sdk-core==2.16.0
aliyun-python-sdk-kms==2.16.5
annotated-types==0.7.0
antlr4-python3-runtime==4.9.3
anyio==4.6.2.post1
attrs==24.2.0
audioread==3.0.1
beautifulsoup4==4.12.3
cachetools==5.5.0
certifi==2024.8.30
cffi==1.17.1
charset-normalizer==3.4.0
click==8.1.7
coloredlogs==15.0.1
conformer==0.3.2
contourpy==1.3.0
crcmod==1.7
cryptography==43.0.3
cycler==0.12.1
Cython==3.0.11
datasets==2.18.0
decorator==5.1.1
deepspeed==0.14.2
diffusers==0.27.2
dill==0.3.8
einops==0.8.0
fastapi==0.115.3
fastapi-cli==0.0.4
ffmpy==0.4.0
filelock==3.16.1
flatbuffers==24.3.25
fonttools==4.54.1
frozenlist==1.5.0
fsspec==2024.2.0
gast==0.6.0
gdown==5.1.0
google-auth==2.35.0
google-auth-oauthlib==1.0.0
gradio==5.3.0
gradio_client==1.4.2
grpcio==1.57.0
grpcio-tools==1.57.0
h11==0.14.0
hjson==3.1.0
httpcore==1.0.6
httpx==0.27.2
huggingface-hub==0.25.2
humanfriendly==10.0
hydra-core==1.3.2
HyperPyYAML==1.2.2
idna==3.10
importlib_metadata==8.5.0
importlib_resources==6.4.5
inflect==7.3.1
Jinja2==3.1.4
jmespath==0.10.0
joblib==1.4.2
kiwisolver==1.4.7
lazy_loader==0.4
librosa==0.10.2
lightning==2.2.4
lightning-utilities==0.11.8
llvmlite==0.43.0
Markdown==3.7
markdown-it-py==3.0.0
MarkupSafe==2.1.5
matcha-tts==0.0.7.0
matplotlib==3.7.5
mdurl==0.1.2
modelscope==1.15.0
more-itertools==10.5.0
mpmath==1.3.0
msgpack==1.1.0
multidict==6.1.0
multiprocess==0.70.16
networkx==3.1
ninja==1.11.1.1
numba==0.60.0
numpy==1.24.4
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==8.9.2.26
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-nccl-cu12==2.20.5
nvidia-nvjitlink-cu12==12.6.77
nvidia-nvtx-cu12==12.1.105
oauthlib==3.2.2
omegaconf==2.3.0
onnxruntime-gpu==1.16.0
openai-whisper==20231117
orjson==3.10.10
oss2==2.19.1
packaging==24.1
pandas==2.2.3
pillow==10.4.0
platformdirs==4.3.6
pooch==1.8.2
propcache==0.2.0
protobuf==4.25.0
psutil==6.1.0
py-cpuinfo==9.0.0
pyarrow==18.0.0
pyarrow-hotfix==0.6
pyasn1==0.6.1
pyasn1_modules==0.4.1
pycparser==2.22
pycryptodome==3.21.0
pydantic==2.7.0
pydantic_core==2.18.1
pydub==0.25.1
Pygments==2.18.0
pynini==2.1.5
pynvml==11.5.3
pyparsing==3.2.0
PySocks==1.7.1
python-dateutil==2.9.0.post0
python-multipart==0.0.16
pytorch-lightning==2.4.0
pytz==2024.2
PyYAML==6.0.2
regex==2024.9.11
requests==2.32.3
requests-oauthlib==2.0.0
rich==13.7.1
rsa==4.9
ruamel.yaml==0.18.6
ruamel.yaml.clib==0.2.12
ruff==0.7.1
safetensors==0.4.5
scikit-learn==1.5.2
scipy==1.14.1
semantic-version==2.10.0
shellingham==1.5.4
simplejson==3.19.3
six==1.16.0
sniffio==1.3.1
sortedcontainers==2.4.0
soundfile==0.12.1
soupsieve==2.6
soxr==0.5.0.post1
starlette==0.41.2
sympy==1.13.3
tensorboard==2.14.0
tensorboard-data-server==0.7.2
threadpoolctl==3.5.0
tiktoken==0.8.0
tokenizers==0.19.1
tomli==2.0.2
tomlkit==0.12.0
torch==2.3.1
torchaudio==2.3.1
torchmetrics==1.5.1
tqdm==4.66.6
transformers==4.44.1
triton==2.3.1
typeguard==4.4.0
typer==0.12.5
typing_extensions==4.12.2
tzdata==2024.2
urllib3==2.2.3
uvicorn==0.32.0
websockets==12.0
Werkzeug==3.0.6
WeTextProcessing==1.0.3
wget==3.2
xxhash==3.5.0
yapf==0.40.2
yarl==1.17.1
zipp==3.20.2
```