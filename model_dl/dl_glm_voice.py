"""
Description: GLM-4-Voice 的 LLM 部分。GLM-4-Voice-9B 在 GLM-4-9B 的基础上进行语音模态的预训练和对齐，从而能够理解和生成离散化的语音。
Notes: 
"""
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('ZhipuAI/glm-4-voice-9b')