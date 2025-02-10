# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import shutil
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
try:
    shutil.copy2('spk2info.pt', 'pretrained_models/CosyVoice2-0.5B/spk2info.pt')
except Exception as e:
    logging.warning(f'复制文件失败: {e}')

inference_mode_list = ['预训练音色', '自然语言控制', '3s极速复刻', '跨语种复刻']
instruct_dict = {'预训练音色': '1. 选择预训练音色\n2. 点击生成音频按钮',
                 '3s极速复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3. 点击生成音频按钮\n4. (可选)保存音色模型',
                 '跨语种复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 点击生成音频按钮\n3. (可选)保存音色模型',
                 '自然语言控制': '1. 选择预训练音色\n2. 输入instruct文本\n3. 点击生成音频按钮'}
stream_mode_list = [('否', False), ('是', True)]
max_val = 0.8

def refresh_sft_spk():
    """刷新音色选择列表
    
    Returns:
        dict: 包含更新后的音色选项的字典
    """
    # 获取自定义音色
    files = [(entry.name, entry.stat().st_mtime) for entry in os.scandir(f"{ROOT_DIR}/voices")]
    files.sort(key=lambda x: x[1], reverse=True) # 按时间排序

    # 添加预训练音色
    choices = [f[0].replace(".pt", "") for f in files] + cosyvoice.list_available_spks()

    if not choices:
        choices = ['']

    return {"choices": choices, "__type__": "update"}


def refresh_prompt_wav():
    """刷新音频选择列表
    
    Returns:
        dict: 包含更新后的音频选项的字典
    """
    files = [(entry.name, entry.stat().st_mtime) for entry in os.scandir(f"{ROOT_DIR}/audios")]
    files.sort(key=lambda x: x[1], reverse=True)  # 按时间排序
    choices = ["请选择参考音频或者自己上传"] + [f[0] for f in files]

    if not choices:
        choices = ['']

    return {"choices": choices, "__type__": "update"}


def change_prompt_wav(audio_path):
    full_path = f"{ROOT_DIR}/audios/{audio_path}"

    if not os.path.exists(full_path):
        logging.warning(f"音频文件不存在: {full_path}")
        return None

    return full_path


def change_stream(stream):
    audio_output_stream = gr.update(visible=stream)
    audio_output_normal = gr.update(visible=not stream)
    return [audio_output_stream, audio_output_normal]


def save_name(name):

    if not name or name == "":
        gr.Info("音色名称不能为空")
        return False

    shutil.copyfile(f"{ROOT_DIR}/output.pt",f"{ROOT_DIR}/voices/{name}.pt")
    gr.Info("音色保存成功,存放位置为voices目录")


def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }


def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech


def change_instruction(mode_checkbox_group):
    sft_dropdown_visible = mode_checkbox_group in ['预训练音色', '自然语言控制']
    save_btn_visible = mode_checkbox_group in ['3s极速复刻']
    return (instruct_dict[mode_checkbox_group], 
            gr.update(visible=sft_dropdown_visible),
            gr.update(visible=save_btn_visible))

def prompt_wav_recognition(prompt_wav):
    if prompt_wav is None:
        return ''

    res = asr_model.generate(input=prompt_wav,
                             language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
                             use_itn=True,
    )

    text = res[0]["text"].split('|>')[-1]
    return text

def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None

    # if instruct mode, please make sure that model is iic/CosyVoice-300M-Instruct and not cross_lingual mode
    if mode_checkbox_group in ['自然语言控制']:
        if not cosyvoice.is_05b and cosyvoice.instruct is False:
            gr.Warning('您正在使用自然语言控制模式, {}模型不支持此模式, 请使用iic/CosyVoice-300M-Instruct模型'.format(args.model_dir))
            yield (cosyvoice.sample_rate, default_data), None
        if instruct_text == '':
            gr.Warning('您正在使用自然语言控制模式, 请输入instruct文本')
            yield (cosyvoice.sample_rate, default_data), None
        if prompt_wav is not None or prompt_text != '':
            gr.Info('您正在使用自然语言控制模式, prompt音频/prompt文本会被忽略')
    # if cross_lingual mode, please make sure that model is iic/CosyVoice-300M and tts_text prompt_text are different language
    if mode_checkbox_group in ['跨语种复刻']:
        if not cosyvoice.is_05b and cosyvoice.instruct is True:
            gr.Warning('您正在使用跨语种复刻模式, {}模型不支持此模式, 请使用iic/CosyVoice-300M模型'.format(args.model_dir))
            yield (cosyvoice.sample_rate, default_data), None
        if instruct_text != '':
            gr.Info('您正在使用跨语种复刻模式, instruct文本会被忽略')
        if prompt_wav is None:
            gr.Warning('您正在使用跨语种复刻模式, 请提供prompt音频')
            yield (cosyvoice.sample_rate, default_data), None
        gr.Info('您正在使用跨语种复刻模式, 请确保合成文本和prompt文本为不同语言')
    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if mode_checkbox_group in ['3s极速复刻', '跨语种复刻']:
        if prompt_wav is None:
            gr.Warning('prompt音频为空，您是否忘记输入prompt音频？')
            yield (cosyvoice.sample_rate, default_data), None
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning('prompt音频采样率{}低于{}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            yield (cosyvoice.sample_rate, default_data), None
    # sft mode only use sft_dropdown
    if mode_checkbox_group in ['预训练音色']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('您正在使用预训练音色模式，prompt文本/prompt音频/instruct文本会被忽略！')
        if sft_dropdown == '':
            gr.Warning('没有可用的预训练音色！')
            yield (cosyvoice.sample_rate, default_data), None
    # zero_shot mode only use prompt_wav prompt text
    if mode_checkbox_group in ['3s极速复刻']:
        if prompt_text == '':
            gr.Warning('prompt文本为空，您是否忘记输入prompt文本？')
            yield (cosyvoice.sample_rate, default_data), None
        if instruct_text != '':
            gr.Info('您正在使用3s极速复刻模式，预训练音色/instruct文本会被忽略！')

    if mode_checkbox_group == '预训练音色':
        logging.info('get sft inference request')
        set_all_random_seed(seed)

        tts_speeches = []
        for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed):
            tts_speeches.append(i['tts_speech'])
            if stream:
                yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten()), None
            
        if not stream:
            audio_data = torch.concat(tts_speeches, dim=1)
            yield None, (cosyvoice.sample_rate, audio_data.numpy().flatten())

    elif mode_checkbox_group == '3s极速复刻':
        logging.info('get zero_shot inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)

        tts_speeches = []
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed):
            tts_speeches.append(i['tts_speech'])
            if stream:
                yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten()), None

        if not stream:
            audio_data = torch.concat(tts_speeches, dim=1)
            yield None, (cosyvoice.sample_rate, audio_data.numpy().flatten())
        
    elif mode_checkbox_group == '跨语种复刻':
        logging.info('get cross_lingual inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)

        tts_speeches = []
        for i in cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=stream, speed=speed):
            tts_speeches.append(i['tts_speech'])
            if stream:
                yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten()), None

        if not stream:
            audio_data = torch.concat(tts_speeches, dim=1)
            yield None, (cosyvoice.sample_rate, audio_data.numpy().flatten())
    else:
        logging.info('get instruct inference request')
        set_all_random_seed(seed)
    
        if sft_dropdown == '':
            gr.Warning('没有选择预训练音色！')
            yield (cosyvoice.sample_rate, default_data), None
            return
        
        # 从预训练音色文件中加载prompt_speech_16k
        voice_path = f"{ROOT_DIR}/voices/{sft_dropdown}.pt"
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            voice_data = torch.load(voice_path, map_location=device) if os.path.exists(voice_path) else None
            prompt_speech_16k = voice_data.get('audio_ref') if voice_data else None
        except Exception as e:
            logging.error(f"加载音色文件失败: {e}")
            prompt_speech_16k = None
        
        if prompt_speech_16k is None:
            gr.Warning('预训练音色文件中缺少prompt_speech数据！')
            yield (cosyvoice.sample_rate, default_data), None
            return

        tts_speeches = []
        for i in cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k, stream=stream, speed=speed):
            tts_speeches.append(i['tts_speech'])
            if stream:
                yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten()), None

        if not stream:
            audio_data = torch.concat(tts_speeches, dim=1)
            yield None, (cosyvoice.sample_rate, audio_data.numpy().flatten())

def main():
    with gr.Blocks() as demo:
        gr.Markdown("### 代码库 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) \
                    预训练模型 [CosyVoice2-0.5B](https://www.modelscope.cn/models/iic/CosyVoice2-0.5B) \
                    [CosyVoice-300M](https://www.modelscope.cn/models/iic/CosyVoice-300M) \
                    [CosyVoice-300M-Instruct](https://www.modelscope.cn/models/iic/CosyVoice-300M-Instruct) \
                    [CosyVoice-300M-SFT](https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT)")
        gr.Markdown("#### 请输入需要合成的文本，选择推理模式，并按照提示步骤进行操作")

        tts_text = gr.Textbox(label="输入合成文本", lines=1, value="CosyVoice迎来全面升级，提供更准、更稳、更快、 更好的语音生成能力。CosyVoice is undergoing a comprehensive upgrade, providing more accurate, stable, faster, and better voice generation capabilities.")
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='选择推理模式', value=inference_mode_list[0])
            instruction_text = gr.Text(label="操作步骤", value=instruct_dict[inference_mode_list[0]], scale=0.5)
            sft_dropdown = gr.Dropdown(choices=sft_spk, label='选择预训练音色', value=sft_spk[0], scale=0.25)
            refresh_new_button = gr.Button("刷新音色")
            refresh_new_button.click(fn=refresh_sft_spk, inputs=[], outputs=[sft_dropdown])
            with gr.Column(scale=0.25):
                stream = gr.Radio(choices=stream_mode_list, label='是否流式推理', value=stream_mode_list[0][1])
                speed = gr.Number(value=1, label="速度调节(仅支持非流式推理)", minimum=0.5, maximum=2.0, step=0.1)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="随机推理种子")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='选择prompt音频文件，注意采样率不低于16khz')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='录制prompt音频文件')
            wavs_dropdown = gr.Dropdown(label="参考音频列表", choices=reference_wavs, value="请选择参考音频或者自己上传", interactive=True)
            refresh_button = gr.Button("刷新参考音频")
            refresh_button.click(fn=refresh_prompt_wav, inputs=[], outputs=[wavs_dropdown])

        prompt_text = gr.Textbox(label="输入prompt文本", lines=1, placeholder="请输入prompt文本，支持自动识别，您可以自行修正识别结果...", value='')
        instruct_text = gr.Textbox(label="输入instruct文本", lines=1, placeholder="请输入instruct文本.例如:用四川话说这句话。", value='')
        wavs_dropdown.change(change_prompt_wav, inputs=[wavs_dropdown], outputs=[prompt_wav_upload])

        with gr.Row(visible=False) as save_spk_btn:  # 默认隐藏
            new_name = gr.Textbox(label="输入新的音色名称", lines=1, placeholder="输入新的音色名称.", value='', scale=2)
            save_button = gr.Button(value="保存音色模型", scale=1)
            save_button.click(save_name, inputs=[new_name])

        generate_button = gr.Button("生成音频")

        # 创建两个音频输出组件,一个用于流式一个用于非流式
        with gr.Row() as audio_outputs:
            audio_output_stream = gr.Audio(
                label="合成音频(流式)", 
                value=None,
                streaming=True,
                autoplay=True,
                show_label=True,
                show_download_button=True,
                visible=False
            )
            audio_output_normal = gr.Audio(
                label="合成音频(非流式)",
                value=None, 
                streaming=False,
                autoplay=True,
                show_label=True,
                show_download_button=True,
                visible=True
            )

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(
            generate_audio,
            inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, 
                   prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed],
            outputs=[audio_output_stream, audio_output_normal]
        )
        mode_checkbox_group.change(fn=change_instruction, 
                                 inputs=[mode_checkbox_group], 
                                 outputs=[instruction_text, sft_dropdown, save_spk_btn])
        prompt_wav_upload.change(fn=prompt_wav_recognition, inputs=[prompt_wav_upload], outputs=[prompt_text])
        prompt_wav_record.change(fn=prompt_wav_recognition, inputs=[prompt_wav_record], outputs=[prompt_text])
        stream.change(
            fn=change_stream, 
            inputs=[stream], 
            outputs=[audio_output_stream, audio_output_normal]
        )
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port, inbrowser=args.open)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8080)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    parser.add_argument('--open', action='store_true', help='open in browser')
    args = parser.parse_args()
    try:
        cosyvoice = CosyVoice(args.model_dir)
    except Exception:
        try:
            cosyvoice = CosyVoice2(args.model_dir)
        except Exception:
            raise TypeError('no valid model_type!')

    sft_spk = refresh_sft_spk()['choices']
    reference_wavs = refresh_prompt_wav()['choices']

    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)

    model_dir = "iic/SenseVoiceSmall"
    asr_model = AutoModel(
        model=model_dir,
        disable_update=True,
        log_level='DEBUG',
        device="cuda:0")
    main()
