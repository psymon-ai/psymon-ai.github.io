---
title: LLaMa Fine-tuning
author: psymon
pubDatetime: 2023-06-20T19:47:59Z
modDatetime: 2023-06-20T19:47:59Z
slug: llama-fine-tuning
featured: false
draft: false
tags:
  - LLaMa
  - Local AI
  - AI
  - Fine-tuning
description: This is the example description of the example post.
---

## Table of contents

## 들어가는 글

안녕하세요. 올해 LLaMa 유출 사건 이후 로컬 Ai에 관심이 생겨 여러 시도를 했습니다.
그간 시행 착오를 정리할 겸 블로그에 공유합니다.


## LLaMa Fine-tuning
가장 먼저 시도한건 Alpaca 모델을 한국어로 학습하는 것이었습니다. 처음에는 Alpaca 원본 데이터를 번역할 생각이었는데 이미 번역 + 학습까지 마친 _KoAlpaca_ 모델이 있다는 걸 알고 KoAlpaca 데이터셋 + [나무위키 데이터](https://huggingface.co/datasets/psymon/namuwiki_alpaca_dataset)를 결합해 Runpod에서 A100 * 4대로 학습했습니다. 

2023년 4월 23일 가득한 기대를 안고 Golani 라고 이름 붙인 모델을 실행했으나 결과는 기대에 많이 못 미쳤습니다. 여러 문제가 있었는데 가장 큰 문제는 한국어를 너무 못한다는 점이었습니다. 나중에 안 사실이지만 원본 LLaMa 모델은 한국어 데이터가 빈약해 단순 Fine-tuning 으로는 한국어 능력을 발휘하기 어려웠습니다. 결국 고라니 모델은 잠시 보류하고 다른 방법을 찾아야 했습니다.

![base-model.png](@/assets/images/base-model.png)

_고라니의 한국어 실력_



## lit-llama를 이용해 Base-model 만들기
한국어 기반으로 학습한 Polyglot 모델을 파인튜닝 해야하나 고민하던 중. [RedPajama](https://www.together.ai/blog/redpajama?utm_source=pytorchkr&ref=pytorchkr)에 대한 글을 발견했습니다. 흥미로운 점은 학습에 사용한 1TB의 학습 데이터를 공개한 것입니다. 아니나 다를까 데이터 공개 직후 RedPajama 데이터를 이용한 오픈 라이선스 LLaMa모델 [OpenLLaMa](https://github.com/openlm-research/open_llama) 프로젝트가 진행되는 것을 보았습니다. 

그때 떠올렸습니다. LLaMa 모델과 동일한 구조를 가져가면서 데이터만 한국어 90% 영어 10%인 모델이 있다면 어떨까. 물론 이는 개인의 범주를 넘는 일입니다. 그래도 일단 되는데까지 해보기로 마음 먹고 한국어 데이터를 모으기 시작했습니다.

이후 손닿는대로 한국어 데이터를 수집했는데 역시나 1TB는 무리였습니다. 약 50GB 정도 데이터를 모은 뒤 결과가 어찌되든 한 번 학습을 돌려보기로 했습니다. 데이터는 RedPajama 형식에 맞게 수정하고 학습 코드는 LLaMa 원본 코드에서 GPL 라이센스를 제거한 [Lit-LLaMa](https://github.com/Lightning-AI/lit-llama)를 사용했습니다. 

### 한국어 토크나이저 문제
문제는 토크나이저였는데 Lit-LLaMa모델 학습용 토크나이저는 한국어 데이터를 고려하지 않았기에 한국어 데이터를 처리하기 부적합했습니다. 그렇다고 polyglot용 토크나이저를 사용하자니 데이터만 한국어인 LLaMa 모델을 만든다는 처음 취지와 맞지 않았습니다. 결국 LLaMa에서 사용하는 Sentencepiece를 직접 수집한 데이터로 다시 학습시켜 사용했습니다. 이게 최선인지는 잘 모르겠습니다. 혹시 따라해보실 분은 이 코드를 사용하시면 됩니다.

```python
import sentencepiece as spm
 from pathlib import Path
 
 paths = [str(x) for x in Path('토크나이저 학습용 데이터 디렉토리 경로').glob("*.txt")]
 corpus = ",".join(paths)
 prefix = "golani"
 vocab_size = 32000-7 # 사용자 정의 토큰을 위해 -7
 spm.SentencePieceTrainer.train(
     f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" + 
     " --model_type=bpe" +
     " --max_sentence_length=999999" + # 문장 최대 길이
     " --pad_id=0 --pad_piece=<pad>" + # pad (0)
     " --unk_id=1 --unk_piece=<unk>" + # unknown (1)
     " --bos_id=2 --bos_piece=<s>" + # begin of sequence (2)
     " --eos_id=3 --eos_piece=</s>" + # end of sequence (3)
     " --byte_fallback=true" + # add byte_fallback for unk tokens
     " --user_defined_symbols=<sep>,<cls>,<mask>"
 ) # 사용자 정의 토큰
```

### 학습 환경과 몇 가지 실수
모델 학습에 상당한 시간이 소요될 것이기에 AWS EC2 환경에서 V100 * 4 대를 이용해 학습했습니다. 이 과정에서 몇 가지 실수가 있었는데 첫 번째는 Out of memory 오류를 해결하려고 **모델 layer와 head를 절반으로 줄인 것**입니다. LLaMa 7B 모델을 학습하기에는 VRam이 부족하여 3B모델이라도 만들어보려던 것인데 단순하게 절반값을 설정했더니 다른 3B 모델과 호환이 안되는 독자규격이 되어버렸습니다. 때문에 hf모델로 변환, ggml 변환, 양자화 등 모든 과정에 큰 번거로움이 뒤따랐습니다.

두번째는 **학습이 중간에 실패할 경우 이어서 학습하는 코드를 구현하지 않은 것**입니다. 16시간 정도 학습한 데이터를 2번 날려먹었습니다. 결국 AWS 크레딧을 거의 다 소진하여 학습을 중단했고 2,700 step을 돌린 미완성의 무언가만 남았습니다. 그래도 애정을 갖고 lit-golani라고 이름 짓고 실행해봤습니다.

![lit-llama.png](@/assets/images/lit-llama.png)

_신이 된 73세 조태호씨_

![lit-llama2.png](@/assets/images/lit-llama2.png)

_의도치 않은 공포영화 도입부_

![lit-llama3.png](@/assets/images/lit-llama3.png)

_직장 내공 남용 금지법_

보다시피 문장이 엉망이지만 생각보단 나쁘지 않다고 느꼈습니다. **50GB** 데이터로 **2700step** 돌린 원본 모델이 그래도 한국어 흉내는 냈으니 말입니다. LLaMa 원본 모델 한국어 실력이랑 좀 비슷한 것 같기도 합니다.  

결과물을 보고 나니 더 욕심이 생겼습니다. 더 많은 데이터와 더 좋은 하드웨어가 있다면 소형 한국어 LLaMa base 모델을 만드는 것도 가능해 보였습니다. 그래서 더 많은 데이터를 모으고 부족한 부분은 RedPajama 1TB 데이터를 번역해야겠다고 생각하게 됐습니다. 이 시점에서 고민이 생겼는데 1TB라는 용량은 번역 프로그램으로 돌리기엔 상당한 시간과 비용이 필요했던 까닭입니다.


## polyglot-QLora로 개인 번역기 만들기
그때 우연히 흥미로운 글을 발견했습니다. QLoRa를 이용하면 Colab에서 Polyglot-12.8B 모델을 학습할 수 있다는 놀라운 내용이었습니다. 게다가 이미 한국어 능력이 뛰어난 polyglot 모델이기에 영-한 번역기로 충분하겠다고 판단하고 바로 도전해봤습니다.

원본 코드는 [여기](https://colab.research.google.com/gist/Beomi/a3032e4eaa33b86fdf8de1f47f15a647/2023_05_26_bnb_4bit_koalpaca_v1_1a_on_polyglot_ko_12_8b.ipynb)에서 제가 사용한 코드는 [여기](https://colab.research.google.com/drive/1k5NFJSZfJfqIuM93LInMDy2a4QpvvONh?usp=sharing)에서 확인할 수 있습니다. 데이터는 Ai-hub의 [영-한 번역 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=71265)를 사용했습니다. Ai-hub 정책상 데이터를 직접 공유할 수는 없지만 사이트에서 다운받으시고 아래 코드 실행하시면 제가 사용한 데이터 셋과 동일합니다. 

```python
import json

 # 디렉토리 경로와 파일명
 directory = "데이터 다운로드 위치/"
 file_name = "일상생활및구어체_영한_valid_set.json"

 # JSON 파일 열기
 with open(directory + file_name, "r", encoding="utf-8") as file:
     data = json.load(file)

 converted_data = []
 for item in data["data"]:
     converted_item = {
         "instruction": "주어진 문장이 한국어일 경우 영어로, 영어일 경우 한국어로 번역하시오.",
         "input": item["en"],
         "output": item["ko"]
     }
     converted_data.append(converted_item)

 output_file_name = "clean_" + file_name
 with open(directory + output_file_name, "w", encoding="utf-8") as output_file:
     json.dump(converted_data, output_file, ensure_ascii=False, indent=4)
```

Colab Pro 환경에서 A100으로 약 8시간 정도 학습했습니다. 학습이 완료되면 [여기](https://colab.research.google.com/drive/1clhrGya8300bSRvVkN9I5s8-F6txnDhy?usp=sharing)서 실행해 볼 수 있습니다. 번역 결과 자체는 생각보다 준수했습니다. 문제는 한 번 실행에 4분 정도 소요된다는 점이었습니다. 실행 속도를 높이기 위해 ggml로 변환했습니다. ggml 변환을 위해 먼저 QLora 파일과 원본 모델을 하나로 병합합니다. 

```python
import sys
 import torch
 from peft import PeftModel
 from transformers import AutoModelForCausalLM
 
 # Based on https://github.com/tloen/alpaca-lora/blob/main/export_hf_checkpoint.py
 # Note that this does NOT guard against no-op merges. I would suggest testing the output.
 
 if len(sys.argv) != 4:
     print("Usage: python export_hf_checkpoint.py <source> <lora> <dest>")
     exit(1)
 
 source_path = sys.argv[1]
 lora_path = sys.argv[2]
 dest_path = sys.argv[3]
 
 base_model = AutoModelForCausalLM.from_pretrained(
     source_path,
     load_in_8bit=False,
     torch_dtype=torch.float16,
     device_map={"": "cpu"},
     trust_remote_code=True,
 )
 
 lora_model = PeftModel.from_pretrained(
     base_model,
     lora_path,
     device_map={"": "cpu"},
     torch_dtype=torch.float16,
 )
 
 # merge weights - new merging method from peft
 lora_model = lora_model.merge_and_unload()
 lora_model.train(False)
 
 lora_model_sd = lora_model.state_dict()
 deloreanized_sd = {
     k.replace("base_model.model.", ""): v
     for k, v in lora_model_sd.items()
     if "lora" not in k
 }
 
 base_model.save_pretrained(
     dest_path, state_dict=deloreanized_sd, max_shard_size="1024MB"
 )
```

저장한 파일을 아래 명령어로 실행하면 됩니다.

```bash
python 파일명.py <원본 모델 위치> <Lora 파일 위치> <저장할 경로>
```
polyglot-12.8b 모델을 병합하는 경우 16GB ram으로는 부족합니다. 저는 ram이 충분한 runpod을 하나 생성해서 진행했습니다. 병합이 끝났다면 이제 ggml로 변환을 진행합니다. 

```python
import sys
 import struct
 import json
 import numpy as np
 
 from transformers import AutoModelForCausalLM, AutoTokenizer
 
 # output in the same directory as the model
 dir_model = "병합 파일이 있는 디렉토리"
 fname_out = "ggml파일 저장할 디렉토리/ggml-model-f16.bin"
 ftype = 1
 
 with open(f"{dir_model}/config.json", "r", encoding="utf-8") as f:
     hparams = json.load(f)
     print(f"open susseced! {dir_model}/config.json")
 
 tokenizer = AutoTokenizer.from_pretrained(dir_model)
 print("load susseced! tokenizer")
 model = AutoModelForCausalLM.from_pretrained(dir_model, low_cpu_mem_usage=True)
 print("load susseced! model")
 
 list_vars = model.state_dict()
 for name in list_vars.keys():
     print(name, list_vars[name].shape, list_vars[name].dtype)
 
 fout = open(fname_out, "wb")
 
 print(hparams)
 
 fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex
 fout.write(struct.pack("i", hparams["vocab_size"]))
 fout.write(struct.pack("i", hparams["max_position_embeddings"]))
 fout.write(struct.pack("i", hparams["hidden_size"]))
 fout.write(struct.pack("i", hparams["num_attention_heads"]))
 fout.write(struct.pack("i", hparams["num_hidden_layers"]))
 fout.write(struct.pack("i", int(hparams["rotary_pct"]*(hparams["hidden_size"]//hparams["num_attention_heads"]))))
 fout.write(struct.pack("i", hparams["use_parallel_residual"] if "use_parallel_residual" in hparams else True))
 fout.write(struct.pack("i", ftype))
 
 # TODO: temporary hack to not deal with implementing the tokenizer
 for i in range(hparams["vocab_size"]):
     text = tokenizer.decode([i]).encode('utf-8')
     fout.write(struct.pack("i", len(text)))
     fout.write(text)
 
 for name in list_vars.keys():
     data = list_vars[name].squeeze().numpy()
     print("Processing variable: " + name + " with shape: ", data.shape)
 
     # we don't need these
     if name.endswith(".attention.masked_bias") or     \
        name.endswith(".attention.bias") or \
        name.endswith(".attention.rotary_emb.inv_freq"):
         print("  Skipping variable: " + name)
         continue
 
     n_dims = len(data.shape)
 
     # ftype == 0 -> float32, ftype == 1 -> float16
     ftype_cur = 0
     if ftype != 0:
         if name[-7:] == ".weight" and n_dims == 2:
             print("  Converting to float16")
             data = data.astype(np.float16)
             ftype_cur = 1
         else:
             print("  Converting to float32")
             data = data.astype(np.float32)
             ftype_cur = 0
     else:
         if data.dtype != np.float32:
             print("  Converting to float32")
             data = data.astype(np.float32)
             ftype_cur = 0
 
     # header
     str = name.encode('utf-8')
     fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
     for i in range(n_dims):
         fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
     fout.write(str)
 
     # data
     data.tofile(fout)
 
 fout.close()
 
 print("Done. Output file: " + fname_out)
 print("")
```

주의점은 병합한 파일이 있는 디렉토리 안에 원본 모델에서 복사한 3개 파일을 넣어줘야 합니다. 

* tokenizer.json
* tokenizer_config.json
* special_tokens_map.json

여기까지 마쳤다면 이제 양자화만 진행하면 됩니다. 양자화 방법은 ggml 레포지토리 GPT-NeoX 예제를 따라하시면 됩니다. 만약 모든 과정이 번거롭다면 제가 올려놓은 파일을 받아서 사용하시면 됩니다. [12.8b-q4_0](https://huggingface.co/psymon/ggml-polyglot-12.8b-translate-q4_0), [12.8b-q5_1](https://huggingface.co/psymon/ggml-polyglot-12.8b-translate-q5_1).

ggml을 통해 실행할 수 있습니다. 실행 프롬프트는 아래와 같습니다. -t 는 실행하는 환경 cpu 코어 수와 동일하게 설정하고, -b 는 메모리가 허용하는 한도내에서 크게 잡으시면 됩니다.

```bash
./bin/gpt-neox -m ../model/ggml-polyglot-translate-q4_0.bin -p "### 명령어: 주어진 문장이 한국어일 경우 영어로, 영어일 경우 한국어로 번역하시오.

### 원문: 번역할 영어 문장

### 번역:" -t 8 -b 16 --temp 0.7 --top_k 40 --top_p 0.1
```

LLaMa 논문의 첫 문단을 번역해 보겠습니다.

![trans.png](@/assets/images/trans.png)

Colab으로 8시간만에 만든 모델치고는 번역 품질이 괜찮습니다. 하지만 매번 터미널 환경을 사용하자니 편의성이 떨어집니다. llama.cpp나 webui에서 실행할 수 있으면 좋겠지만 polyglot기반 모델은 두 프로그램에서 돌리는게 쉽지 않습니다. 그래서 gradio로 간단한 ui를 하나 만들었습니다. 코드는 [링크](https://colab.research.google.com/drive/1mVGLi8OtAHGEOrL93O-wwrY7hwoApmTu?usp=sharing)를 참조. 

실행하면 아래와 같은 심플한 UI가 생성됩니다. 4bit 양자화 후에도 12.8b 모델은 실행에 약 30초가 소요됩니다. 추론 시간을 줄이려 시도해봤지만 실패했습니다. 

![trans2.png](@/assets/images/trans2.png)

위 과정을 본인이 원하는 데이터로 바꾸면 Polyglot 모델을 쉽고 다양하게 활용할 수 있을 것입니다. 
<br /><br />
<br /><br />


