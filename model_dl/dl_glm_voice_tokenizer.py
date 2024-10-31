"""
Description: GLM-4-Voice 的 speech tokenizer 部分。通过在 Whisper 的 encoder 部分增加 vector quantization 进行训练，将连续的语音输入转化为离散的 token。每秒音频转化为 12.5 个离散 token。
Notes: 
"""
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('ZhipuAI/glm-4-voice-tokenizer')