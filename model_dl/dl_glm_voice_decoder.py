"""
Description: GLM-4-Voice 的 speech decoder 部分。GLM-4-Voice-Decoder 是基于 CosyVoice 重新训练的支持流式推理的语音解码器，将离散化的语音 token 转化为连续的语音输出。最少只需要 10 个音频 token 即可开始生成，降低对话延迟。
Notes: 
"""
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('ZhipuAI/glm-4-voice-decoder')