import time
import io, os, sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/AcademiCodec'.format(ROOT_DIR))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

import requests
from pydub import AudioSegment

import numpy as np
from flask import Flask, request, Response,send_from_directory
import torch
import torchaudio

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import ffmpeg

from flask_cors import CORS
from flask import make_response

import shutil

import json

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B')

default_voices = cosyvoice.list_available_spks()

spk_custom = []
for name in os.listdir(f"{ROOT_DIR}/voices/"):
    print(name.replace(".pt", ""))
    spk_custom.append(name.replace(".pt", ""))

print("默认音色", default_voices)
print("自定义音色", spk_custom)

app = Flask(__name__)

CORS(app, cors_allowed_origins="*")
# CORS(app, supports_credentials=True)

def process_audio(tts_speeches, sample_rate=22050, format="wav"):
    """处理音频数据并返回响应"""
    buffer = io.BytesIO()
    audio_data = torch.concat(tts_speeches, dim=1)
    torchaudio.save(buffer, audio_data, sample_rate, format=format)
    buffer.seek(0)
    return buffer

def create_audio_response(buffer, format="wav"):
    """创建音频响应"""
    if format == "wav":
        return Response(buffer.read(), mimetype="audio/wav")
    else:
        response = make_response(buffer.read())
        response.headers['Content-Type'] = f'audio/{format}'
        response.headers['Content-Disposition'] = f'attachment; filename=sound.{format}'
        return response

def load_voice_data(speaker):
    """加载语音数据"""
    voice_path = f"{ROOT_DIR}/voices/{speaker}.pt"
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not os.path.exists(voice_path):
            return None
        voice_data = torch.load(voice_path, map_location=device)
        return voice_data.get('audio_ref')
    except Exception as e:
        raise ValueError(f"加载音色文件失败: {e}")

@app.route("/", methods=['GET', 'POST'])
@app.route("/tts", methods=['GET', 'POST'])
def tts():
    # 获取参数
    params = request.get_json() if request.method == 'POST' else request.args
    text = params.get('text')
    speaker = params.get('speaker')
    instruct = params.get('instruct')
    streaming = int(params.get('streaming', 0))
    speed = float(params.get('speed', 1.0))

    # 验证必要参数
    if not text or not speaker:
        return {"error": "文本和角色名不能为空"}, 400

    # 处理 instruct 模式
    if instruct:
        prompt_speech_16k = load_voice_data(speaker)
        if prompt_speech_16k is None:
            return {"error": "预训练音色文件中缺少audio_ref数据！"}, 500
        
        inference_func = lambda: cosyvoice.inference_instruct2(
            text, instruct, prompt_speech_16k, stream=bool(streaming), speed=speed
        )
    else:
        inference_func = lambda: cosyvoice.inference_sft(
            text, speaker, stream=bool(streaming), speed=speed
        )

    # 处理流式输出
    if streaming:
        def generate():
            for _, i in enumerate(inference_func()):
                buffer = process_audio([i['tts_speech']], format="ogg")
                yield buffer.read()
        
        response = make_response(generate())
        response.headers.update({
            'Content-Type': 'audio/ogg',
            'Content-Disposition': 'attachment; filename=sound.ogg'
        })
        return response
    
    # 处理非流式输出
    tts_speeches = [i['tts_speech'] for _, i in enumerate(inference_func())]
    buffer = process_audio(tts_speeches, format="wav")
    return create_audio_response(buffer)


@app.route("/speakers", methods=['GET', 'POST'])
def speakers():

    voices = []

    for x in default_voices:
        voices.append({"name":x,"voice_id":x})

    for name in os.listdir("voices"):
        name = name.replace(".pt","")
        voices.append({"name":name,"voice_id":name})

    response = app.response_class(
        response=json.dumps(voices),
        status=200,
        mimetype='application/json'
    )
    return response

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9880)
