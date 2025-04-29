```bash
sudo apt-get update && sudo apt-get install git-lfs cbm ffmpeg
git clone https://huggingface.co/spaces/svjack/FantasyTalking && cd FantasyTalking
pip install -r requirements.txt

#### 512 image 极短 音频
python infer.py  --image_path ganyu.png \
 --audio_path assets/audios/woman.wav \
 --prompt "In this vibrant anime-style digital illustration, GANYU, a character with light blue hair and red horns, is depicted sitting at a café table. She holds a white coffee cup with a logo, steam rising from it, and a plate with a cake and fruit. She wears a white shirt and a black apron. The background shows a cozy café with wooden walls, potted plants, and soft lighting. The overall atmosphere is warm and inviting." \
 --prompt_cfg_scale 5.0 --audio_cfg_scale 5.0 --num_persistent_param_in_dit 7000000000
```

```python
#https://huggingface.co/datasets/simon3000/genshin-voice
#pip install soundfile datasets

from datasets import load_dataset
import soundfile as sf
import os

# Load the dataset
dataset = load_dataset('simon3000/genshin-voice', split='train', streaming=True)

# Filter the dataset for Chinese voices of Ganyu with transcriptions
chinese_ganyu = dataset.filter(lambda voice: voice['language'] == 'Chinese' and voice['speaker'] == 'Ganyu' and voice['transcription'] != '')

# Create a folder to store the audio and transcription files
ganyu_folder = 'genshin_impact_ganyu_audio_sample'
os.makedirs(ganyu_folder, exist_ok=True)

# Process each voice in the filtered dataset
for i, voice in enumerate(chinese_ganyu):
  audio_path = os.path.join(ganyu_folder, f'{i}_audio.wav')  # Path to save the audio file
  transcription_path = os.path.join(ganyu_folder, f'{i}_audio.txt')  # Path to save the transcription file

  # Save the audio file
  sf.write(audio_path, voice['audio']['array'], voice['audio']['sampling_rate'])

  # Save the transcription file
  with open(transcription_path, 'w') as transcription_file:
    transcription_file.write(voice['transcription'])

  print(f'{i} done')  # Print the progress

```

[中文阅读](./README_zh.md)
# FantasyTalking: Realistic Talking Portrait Generation via Coherent Motion Synthesis

[![Home Page](https://img.shields.io/badge/Project-<Website>-blue.svg)](https://fantasy-amap.github.io/fantasy-talking/)
[![arXiv](https://img.shields.io/badge/Arxiv-2504.04842-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2504.04842)
[![hf_paper](https://img.shields.io/badge/🤗-Paper%20In%20HF-red.svg)](https://huggingface.co/papers/2504.04842)

## 🔥 Latest News!!
* April 29, 2025: Our work is merged to [ComfyUI-Wan](https://github.com/kijai/ComfyUI-WanVideoWrapper) ! Thank [kijai](https://github.com/kijai) for the update 👏!
* April 28, 2025: We released the inference code and model weights for audio conditions.


<!-- ![Fig.1](https://github.com/Fantasy-AMAP/fantasy-talking/blob/main/assert/fig0_1_0.png) -->


## Quickstart
### 🛠️Installation

Clone the repo:

```
git clone https://github.com/Fantasy-AMAP/fantasy-talking.git
cd fantasy-talking
```

Install dependencies:
```
# Ensure torch >= 2.0.0
pip install -r requirements.txt
# Optional to install flash_attn to accelerate attention computation
pip install flash_attn
```

### 🧱Model Download
| Models        |                       Download Link                                           |    Notes                      |
| --------------|-------------------------------------------------------------------------------|-------------------------------|
| Wan2.1-I2V-14B-720P  |      🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P)    🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-720P)     | Base model
| Wav2Vec |      🤗 [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)    🤖 [ModelScope](https://modelscope.cn/models/AI-ModelScope/wav2vec2-base-960h)      | Audio encoder
| FantasyTalking model      |      🤗 [Huggingface](https://huggingface.co/acvlab/FantasyTalking/)     🤖 [ModelScope](https://www.modelscope.cn/models/amap_cvlab/FantasyTalking/)         | Our audio condition weights

Download models using huggingface-cli:
``` sh
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./models/Wan2.1-I2V-14B-720P
huggingface-cli download facebook/wav2vec2-base-960h --local-dir ./models/wav2vec2-base-960h
huggingface-cli download acvlab/FantasyTalking fantasytalking_model.ckpt --local-dir ./models
```

Download models using modelscope-cli:
``` sh
pip install modelscope
modelscope download Wan-AI/Wan2.1-I2V-14B-720P --local_dir ./models/Wan2.1-I2V-14B-720P
modelscope download AI-ModelScope/wav2vec2-base-960h --local_dir ./models/wav2vec2-base-960h
modelscope download amap_cvlab/FantasyTalking   fantasytalking_model.ckpt  --local_dir ./models
```

### 🔑 Inference
``` sh
python infer.py  --image_path ./assets/images/woman.png --audio_path ./assets/audios/woman.wav
```
You can control the character's behavior through the prompt. **The recommended range for prompt and audio cfg is [3-7]. You can increase the audio cfg to achieve more consistent lip-sync.**
``` sh
python infer.py  --image_path ./assets/images/woman.png --audio_path ./assets/audios/woman.wav --prompt "The person is speaking enthusiastically, with their hands continuously waving." --prompt_cfg_scale 5.0 --audio_cfg_scale 5.0
```

We present a detailed table here. The model is tested on a single A100.(512x512, 81 frames).

|`torch_dtype`|`num_persistent_param_in_dit`|Speed|Required VRAM|
|-|-|-|-|
|torch.bfloat16|None (unlimited)|15.5s/it|40G|
|torch.bfloat16|7*10**9 (7B)|32.8s/it|20G|
|torch.bfloat16|0|42.6s/it|5G|

### Gradio Demo
We construct an [online demo](https://huggingface.co/spaces/acvlab/FantasyTalking) in Huggingface.
For the local gradio demo, you can run:
``` sh
pip install gradio spaces
python app.py
```

## 🧩 Community Works
We ❤️ contributions from the open-source community! If your work has improved FantasyTalking, please inform us.
Or you can directly e-mail [frank.jf@alibaba-inc.com](mailto://frank.jf@alibaba-inc.com). We are happy to reference your project for everyone's convenience.

## 🔗Citation
If you find this repository useful, please consider giving a star ⭐ and citation
```
@article{wang2025fantasytalking,
   title={FantasyTalking: Realistic Talking Portrait Generation via Coherent Motion Synthesis},
   author={Wang, Mengchao and Wang, Qiang and Jiang, Fan and Fan, Yaqi and Zhang, Yunpeng and Qi, Yonggang and Zhao, Kun and Xu, Mu},
   journal={arXiv preprint arXiv:2504.04842},
   year={2025}
 }
```

## Acknowledgments
Thanks to [Wan2.1](https://github.com/Wan-Video/Wan2.1), [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), and [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) for open-sourcing their models and code, which provided valuable references and support for this project. Their contributions to the open-source community are truly appreciated.
