<p align="center">
  <a href="https://github.com/Akshay090/svg-banners" target="_blank">
    <img alt="SVG Banners" src="https://svg-banners.vercel.app/api?type=origin&text1=CosyVoice🤠&text2=Text-to-Speech%20💖%20Large%20Language%20Model&width=800&height=210">
  </a>
</p>
<p align="center">
  <a href="https://count.getloli.com" target="_blank">
    <img alt="CosyVoice2-Ex" src="https://count.getloli.com/@CosyVoice2-Ex.github?name=CosyVoice2-Ex.github&theme=3d-num&padding=7&offset=0&align=top&scale=1&pixelated=1&darkmode=auto">
  </a>
</p>

# CosyVoice2-Ex
CosyVoice2 功能扩充（预训练音色/3s极速复刻/自然语言控制/自动识别/音色保存/API），支持 Windows / MacOS

Demo: [Modelscope](https://www.modelscope.cn/studios/journey0ad/CosyVoice2-Ex)

## 启动

### Windows
提供有Windows可用的一键包，首次运行会自动下载模型文件

解压后双击打开 `运行-CosyVoice2-Ex.bat` 即可运行

[>>> 点击下载 <<<](https://github.com/journey-ad/CosyVoice2-Ex/releases/latest)

### macOS
须通过conda环境运行，参考 https://docs.conda.io/en/latest/miniconda.html

```sh
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

python webui.py --port 8080 --open
```

## 接口地址

[示例指令](./示例指令.html)
```sh
python api.py

http://localhost:9880/?text=春日清晨，老街深处飘来阵阵豆香。三代传承的手艺，将金黄的豆浆熬制成最纯粹的味道。一碗温热的豆腐脑，不仅是早餐，更是儿时难忘的记忆，是岁月沉淀的生活智慧。&speaker=舌尖上的中国

http://localhost:9880/?text=hello%20hello~%20[breath]%20听得到吗？%20きこえていますか？%20初次见面，请多关照呀！%20这里是嘉然Diana，大家也可以叫我<strong>蒂娜</strong>%20是你们最甜甜甜的小草莓&speaker=嘉然&instruct=慢速，用可爱的语气说
```

## Credits

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [CosyVoice_For_Windows](https://github.com/v3ucn/CosyVoice_For_Windows)
- [Fish Audio](https://fish.audio)