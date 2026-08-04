---
title: Mistral-7B ARC-Challenge - 모델 오답으로 다시 학습하기
author: psymon
pubDatetime: 2026-07-20T13:11:20+09:00
modDatetime: 2026-07-20T12:11:20+09:00
slug: mistral-7b-arc-sft-mio-revalidation
featured: true
draft: false
tags:
  - LLM
  - Fine-tuning
keywords:
  - Mistral-7B
  - ARC-Challenge
  - QLoRA
  - SFT
  - MIO
  - DPO
  - hard negative
  - preference learning
  - lm-evaluation-harness
  - LLM evaluation
description: Mistral-7B에 과학 QA를 SFT한 뒤 모델이 높은 점수를 준 오답을 preference 학습에 활용했다. 높은 점수를 얻은 뒤 대조 실험과 재현성을 다시 검증한 기록.
---

## Table of contents

## 들어가는 글

최근 공개 모델을 직접 학습하고 평가하는 실험을 이어가고 있습니다. 벤치마크 점수는 모델끼리 비교하기 편하지만, 숫자가 왜 올랐는지 설명하는 일은 생각보다 어렵습니다. 관련 지식이 늘어서인지, 평가 형식에 익숙해져서인지, 학습과 평가가 우연히 잘 맞았는지 점수 하나만으로는 알기 어렵기 때문입니다.

이번에는 `Mistral-7B-v0.1`과 ARC-Challenge를 골랐습니다. 7B 모델은 개인 환경에서도 여러 학습 방법을 비교해 볼 수 있고, ARC-Challenge는 정답과 오답이 분명해 학습 전후 변화를 살펴보기 좋았습니다. 단순히 점수를 얼마나 높일 수 있는지보다, 어떤 학습 신호가 실제로 점수를 바꾸는지 확인해 보고 싶었습니다.

처음 세운 가설은 단순했습니다. ARC-Challenge가 초등·중등 과학을 다루는 만큼, ARC와 OpenBookQA, QASC, SciQ를 한데 모아 가르치면 점수가 자연스럽게 오를 것 같았습니다.

실제로 점수는 올랐습니다. 다만 기대 만큼은 아니었습니다. 틀린 문항을 살펴보니 모델이 지식을 몰라서 틀린 건 아니었습니다. 정답과 비슷한 오답을 판별하는 능력이 부족했습니다.

이 결과를 보며 정답을 아는 것만으로 충분하지 않다고 생각했습니다. 그럴듯한 오답까지 가려낼 수 있어야 합니다. 여기서 질문을 조금 바꿔 봤습니다.

> 정답을 한 번 더 가르치기보다  
> 모델이 정답처럼 높은 점수를 준 오답을 함께 보여주면 어떨까?

이 질문을 확인하기 위해 학습을 두 단계로 나눴습니다.

1. 과학 QA로 completion-only SFT를 한다.
2. SFT 모델이 가장 그럴듯하다고 본 오답을 골라 preference learning을 한다.

실험 결과는 예상보다 컸습니다. Validation `acc_norm`은 57.53%에서 72.91%까지 올랐고, 마지막 Test에서는 76.54%가 나왔습니다.

그런데 점수가 높아질수록 확인해야 할 질문도 늘었습니다.

> 이 개선은 정말 hard negative 때문일까?  
> MIO가 DPO보다 나았던 걸까?  
> seed가 바뀌어도 같은 결과가 나올까?

처음에는 점수를 높인 방법만 정리할 생각이었습니다. 대조 실험을 하나씩 더하면서 처음 설명만으로는 부족하다는 걸 알게 됐습니다. 이번 글에는 모델을 학습한 과정뿐 아니라, 높은 점수를 다시 의심하고 검증한 과정까지 함께 담았습니다.

## ARC-Challenge는 어떤 능력을 평가할까

ARC(AI2 Reasoning Challenge)[^arc-paper]는 과학 객관식 benchmark입니다. 그중 ARC-Challenge에는 단순한 단어 맞추기보다 여러 추론 단계를 거쳐야 풀 수 있는 문제가 모여 있습니다.

평가에는 EleutherAI의 `lm-evaluation-harness` 0.4.12를 사용했습니다. 25-shot 문맥을 주고, 모델이 각 선택지에 매긴 log-likelihood를 비교합니다.

프롬프트 형식은 단순합니다.

```text
Question: {question}
Answer:
```

여기서 중요한 건 모델이 `A`, `B`, `C`, `D` 같은 위치 기호를 맞히는 게 아니라는 점입니다. 평가기는 각 선택지 문장을 completion으로 하나씩 붙여 likelihood를 비교합니다.

예를 들어 질문과 선택지가 다음과 같다고 해보겠습니다.

```text
Question: What is a worldwide increase in temperature called?
Answer:
```

```text
 A decrease in rainfall
 Global warming
 A change in seasons
 Soil erosion
```

선택지마다 likelihood를 구한 뒤 값이 가장 높은 답을 고릅니다. 이번 실험에서는 raw `acc`와 `acc_norm`을 모두 기록했고, 모델을 고를 때는 `acc_norm`을 기준으로 삼았습니다. 선택지 길이가 다르면 짧은 답이 유리할 수 있어 completion 길이에 맞춰 likelihood를 보정한 값입니다.

이 평가는 모델이 자유롭게 답을 생성하는 능력을 재지 않습니다. 평가기가 정해 둔 답 가운데 가장 그럴듯한 선택지를 찾는 closed-set 평가입니다. 따라서 이후에 나오는 점수도 일반 질의응답이나 free-form 생성 능력까지 좋아졌다는 뜻으로 해석할 수는 없습니다.

## 첫 번째 가설: 과학 문제를 더 많이 가르치자

첫 단계는 과학 QA를 이용한 SFT였습니다. 데이터는 모두 공개 train split에서 가져왔습니다.

| 데이터셋      | 필터링 후 문항 | 역할                 |
| ------------- | -------------: | -------------------- |
| ARC-Challenge |          1,117 | SFT, preference 후보 |
| ARC-Easy      |          2,241 | SFT, preference 후보 |
| OpenBookQA    |          4,826 | SFT, preference 후보 |
| QASC          |          7,514 | SFT, preference 후보 |
| SciQ          |         11,596 | SFT 전용             |
| **합계**      |     **27,294** |                      |

원본 후보는 28,140문항이었습니다. 여러 데이터셋에 겹쳐 들어간 질문이 있어 정규화한 stem을 기준으로 train 중복 843문항을 제거했습니다. ARC-Challenge validation과 같은 질문 2문항, 유사도 0.94 이상인 질문 1문항도 제외했습니다. 이 단계에서는 Test를 열지 않았고, 중복 검사와 모델 선택에는 validation만 사용했습니다.

모든 데이터를 같은 형식으로 바꿨습니다.

```text
prompt     = "Question: {question}\nAnswer:"
completion = " {correct answer text}"
```

질문에는 선택지를 넣지 않았습니다. 모델이 answer position을 외우는 대신 정답 문장을 completion하도록 했고, loss도 prompt token이 아닌 정답 completion에만 걸었습니다.

Completion-only loss를 쓴다고 해서 모델이 질문을 읽지 않는 건 아닙니다. 질문은 정답을 예측하는 문맥으로 그대로 들어가고, loss를 계산할 때만 prompt token을 제외합니다. 전체 문장에 loss를 걸면 이미 입력으로 주어진 질문과 고정된 템플릿을 다시 예측하는 데에도 gradient가 쓰입니다. 이번 SFT에서 배우게 하고 싶었던 건 질문을 복원하는 능력이 아니라, 질문에 맞는 답을 이어 쓰는 능력이었습니다.

학습 설정은 다음과 같습니다.

| 항목                        | 값                          |
| --------------------------- | --------------------------- |
| Base model                  | `mistralai/Mistral-7B-v0.1` |
| Quantization                | 4-bit NF4 QLoRA             |
| Compute dtype               | BF16                        |
| LoRA target                 | 모든 linear projection      |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05              |
| Learning rate               | `1e-4`                      |
| Effective batch             | 16                          |
| Epoch / optimizer step      | 1 / 1,706                   |
| Max length                  | 384                         |

학습 메모리를 줄이기 위해 QLoRA[^qlora]를 사용했습니다. 다만 비교 평가는 quantized 모델로 하지 않았습니다. 각 adapter를 FP16 parent에 safe merge한 뒤 Base, SFT, preference 모델을 모두 같은 FP16 설정에서 평가했습니다.

## SFT만으로는 충분하지 않았다

Validation 결과는 다음과 같았습니다.

| 모델                | `acc_norm` |
| ------------------- | ---------: |
| Mistral-7B Base     |     57.53% |
| Completion-only SFT |     60.54% |

모델 선택 지표인 `acc_norm`은 3.01%p 올랐습니다. 과학 QA 27,294문항을 더 보여준 것에 비하면 아쉬운 결과였습니다.

이 지점에서 단순히 정답을 더 많이 학습하는 것만으로 충분한지 의문이 들었습니다. 모델이 정답 문장에 높은 확률을 주는 일과, 정답과 닮은 오답을 가려내는 일은 조금 다른 문제일 수 있습니다. 그래서 SFT 모델이 각 오답에 얼마나 높은 점수를 주는지 직접 살펴봤습니다.

## 모델이 높은 점수를 준 오답 찾기

ARC-Challenge, ARC-Easy, OpenBookQA, QASC의 train 질문 15,698문항을 다시 꺼내 모든 선택지를 SFT 모델로 채점했습니다. 선택지 점수는 answer-token log-likelihood 합을 문자 수로 나눠 계산했고, 각 질문에서 점수가 가장 높은 오답을 찾았습니다.

```text
prompt   = "Question: {question}\nAnswer:"
chosen   = " {correct answer text}"
rejected = " {high-scoring wrong answer text}"
```

질문마다 오답을 최대 세 개 골랐고, 모두 47,087개의 preference pair를 얻었습니다. 15,698문항 가운데 3,081문항에선 적어도 하나의 오답이 정답보다 높은 점수를 받았습니다. 나머지 문항도 오답 중 모델이 가장 그럴듯하다고 본 선택지를 골랐습니다.

여기서 `hard negative`는 반드시 정답보다 점수가 높은 오답을 뜻하지 않습니다. 질문마다 모델이 가장 높은 점수를 준 오답을 가리킵니다.

SciQ는 preference pair에서 제외했습니다. 정규화 과정에서 answer position shortcut을 배울 수 있고, 전체 55%를 동일 데이터셋으로 채우는 것에 대한 우려 때문입니다.

이렇게 고른 pair는 정답과 모델이 높은 점수를 준 오답이 함께 들어갑니다. 모델이 실제로 고른 오답을 다음 학습에 다시 활용한 셈입니다.

## MIO로 정답과 오답 점수 조정하기

Preference 단계에는 MIO(Mutual Information Optimization)[^mio-paper]를 사용했습니다. SFT가 정답 문장의 확률을 높이는 단계였다면, 다음 단계에서는 정답과 모델이 고른 오답의 차이를 직접 학습하고 싶었습니다. MIO는 policy가 chosen과 rejected를 reference보다 얼마나 높이거나 낮췄는지 따로 볼 수 있어 이 가설과 잘 맞았습니다.

물론 개념이 잘 맞는다는 이유만으로 MIO가 다른 objective보다 낫다고 말할 수는 없습니다. 당시에는 가설을 가장 강하게 반영할 수 있는 방법이라고 판단해 선택했고, 이후 DPO 대조군을 추가해 실제 차이가 있는지 따로 확인했습니다.

먼저 SFT 모델을 FP16으로 merge해 frozen reference로 고정하고, 같은 checkpoint 위에 새 LoRA policy를 붙였습니다. Chosen과 rejected completion에 대해서는 reference 대비 policy의 길이 정규화 log-ratio를 계산했습니다.

$$
z_w = \overline{\log \pi_\theta(y_w|x)}
      - \overline{\log \pi_{ref}(y_w|x)}
$$

$$
z_l = \overline{\log \pi_\theta(y_l|x)}
      - \overline{\log \pi_{ref}(y_l|x)}
$$

이번 실험에서 사용한 pairwise loss는 다음과 같습니다.

$$
\mathcal{L}_{MIO}
= -\log \sigma(\beta z_w)
- \frac{1}{2}\log \sigma(-\beta z_w)
- \frac{1}{2}\log \sigma(-\beta z_l)
$$

수식만 보면 복잡해 보이지만, 목적은 세 가지입니다.

- 정답에 매긴 점수를 SFT reference보다 높인다.
- 모델이 높은 점수를 준 오답에 매긴 점수를 reference보다 낮춘다.
- 정답 점수가 제한 없이 커지는 것은 억제한다.

Completion log-probability는 EOS를 포함한 token 평균으로 계산했습니다. 답변 길이가 beta의 의미를 크게 바꾸지 않도록 하기 위한 선택입니다.

Policy와 reference는 같은 SFT checkpoint에서 시작합니다. 따라서 초기에는 `z_w = z_l = 0`이어야 하고, 구현이 맞다면 loss는 `ln(4) = 1.386294`가 나와야 합니다. 실제 학습에 앞서 이 값과 policy/reference identity를 gate로 확인했습니다.

학습 설정은 다음과 같습니다.

| 항목                        | 값            |
| --------------------------- | ------------- |
| Preference pair             | 47,087        |
| Beta                        | 0.5           |
| Learning rate               | `7.5e-6`      |
| LoRA rank / alpha / dropout | 16 / 32 / 0.0 |
| Effective batch             | 32            |
| Epoch / optimizer step      | 1 / 1,472     |
| Max length                  | 256           |

학습은 Google Colab의 NVIDIA A100-SXM4 40GB 한 장에서 진행했습니다. Frozen reference forward를 `no_grad`로 먼저 실행하고 activation을 해제한 뒤 policy forward를 계산했습니다. 이렇게 하면 두 모델의 activation graph를 동시에 들고 있지 않아도 같은 objective를 계산할 수 있습니다.

## Validation과 Test 결과

MIO adapter를 같은 FP16 SFT parent에 merge하고, 299개 validation 문항을 다시 평가했습니다.

| 단계      | `acc_norm` |
| --------- | ---------: |
| Base      |     57.53% |
| SFT       |     60.54% |
| SFT + MIO | **72.91%** |

SFT만으로는 3.01%p 올랐지만, preference 단계를 거치면서 12.37%p가 더 올랐습니다.

Test를 열기 전 후보 모델과 평가 조건을 고정했습니다. Base, SFT, MIO가 같은 validation 문항과 prompt, target을 평가했는지 확인하고, data hash와 adapter hash를 candidate lock에 기록했습니다.

이어서 ARC-Challenge Test 1,172문항을 한 번 평가했습니다.

| split              |  문항 |  `acc` | `acc_norm` |
| ------------------ | ----: | -----: | ---------: |
| ARC-Challenge Test | 1,172 | 74.74% | **76.54%** |

처음에는 여기까지가 실험의 결론이라고 생각했습니다. 모델도 Hugging Face에 공개했습니다.

- [psymon/mistral-7b-mio-arc-fp16](https://huggingface.co/psymon/mistral-7b-mio-arc-fp16)

Standalone FP16 weight와 tokenizer를 함께 올렸기 때문에 추론에는 PEFT나 bitsandbytes가 필요하지 않습니다.

## Hard negative가 정말 중요했을까

높은 점수를 얻고 나니 다음 질문은 각 요소별 기여도를 분리하는 것이었습니다. 첫 실험에는 SFT reference, model-scored hard negative, token 평균 log-probability, MIO가 한꺼번에 들어갔습니다. 전체 절차가 좋아졌다는 사실은 알 수 있었지만, 어느 요소가 얼마나 기여했는지는 알 수 없었습니다.

이를 확인하기 위해 QASC의 negative selection만 uniform으로 바꾼 대조군을 만들었습니다.

QASC는 질문마다 오답이 7개입니다. 기존 hard 조건에서는 SFT 모델 점수가 높은 오답 세 개를 사용했습니다. 대조군에서는 고정된 SHA-256 기반 seed로 오답 세 개를 균일하게 골랐습니다. QASC 7,514개 그룹 중 7,489개 그룹의 rejected 구성이 바뀌었고, 나머지 데이터셋 pair는 그대로 유지했습니다.

| seed 42 설정                |  `acc` | `acc_norm` |
| --------------------------- | -----: | ---------: |
| MIO + hard negative         | 70.90% | **72.91%** |
| MIO + QASC uniform negative | 70.90% |     72.58% |

방향만 보면 hard negative가 조금 높았습니다. 하지만 validation 299문항 중 한 문항 차이였고, 모든 source의 negative를 uniform으로 바꾼 것도 아니며 비교 seed도 하나뿐입니다. 이 결과만으로 hard negative의 우위를 주장하기는 어렵습니다.

처음에는 hard negative가 점수 상승의 핵심일 거라고 생각했습니다. 대조군을 추가한 뒤에는 결론을 더 좁게 잡아야 했습니다.

> 모델이 고른 오답을 사용한 전체 절차는 잘 작동했다.  
> 다만 hard selection이 uniform selection보다 확실히 낫다는 증거는 아직 부족하다.

## MIO가 DPO보다 나았을까

MIO 자체의 기여를 보기 위해 DPO[^dpo-paper] 대조군도 학습했습니다.

초기 policy와 reference가 같을 때 MIO와 DPO의 chosen/rejected raw gradient가 같아지도록 beta를 맞췄습니다. MIO는 `beta=0.5`, DPO는 `beta=0.25`를 사용했습니다. 초기 gradient cosine similarity와 norm ratio가 모두 1.0인지 확인한 뒤 같은 hard-negative pair로 학습했습니다.

| seed 42 설정        |      `acc` | `acc_norm` |
| ------------------- | ---------: | ---------: |
| MIO + hard negative | **70.90%** | **72.91%** |
| DPO + hard negative |     69.23% | **72.91%** |

Raw accuracy는 MIO가 높았지만, 처음부터 고정한 선택 지표 `acc_norm`은 정확히 같았습니다. 적어도 이 validation과 설정에서는 MIO가 DPO보다 낫다고 말할 수 없습니다.

이 결과는 MIO를 선택한 이유와 실험으로 입증한 내용을 구분하게 해줬습니다. MIO는 정답과 오답의 점수 차이를 reference 기준으로 조절하고 싶어서 선택했습니다. 실제로 확인한 건 특정 objective의 우위가 아니라 preference optimization 단계 전체의 효과였습니다.

## 정답을 더 보여준 효과는 아니었을까

Preference 단계에서는 chosen answer도 다시 봅니다. 어쩌면 모델이 좋아진 이유는 rejected signal 때문이 아니라, 정답을 한 번 더 학습했기 때문일 수 있습니다.

이를 확인하기 위해 rejected signal이 없는 positive-only 대조군을 만들었습니다. MIO의 초기 chosen raw-gradient coefficient에 맞춰 정답 completion만 추가 학습했습니다.

| seed 42 설정         | `acc_norm` |
| -------------------- | ---------: |
| SFT                  |     60.54% |
| Chosen-only 추가 SFT |     64.21% |
| MIO + hard negative  | **72.91%** |

정답을 다시 보여주는 것만으로도 3.68%p 올랐습니다. 다만 MIO와는 8.70%p 차이가 남았습니다. 적어도 preference 단계의 큰 상승을 정답 재노출만으로 설명하기는 어려웠습니다.

이 대조군도 완전하지는 않습니다. 초기 chosen gradient coefficient만 맞췄고, MIO의 chosen gradient는 학습 중 비선형으로 달라집니다. 두 모델의 차이를 rejected term만의 순수한 효과라고 해석할 수는 없습니다. 이번 실험에서 확인한 범위는 더 좁습니다.

> 단순한 chosen-only 추가 학습만으로는 MIO-hard의 상승을 재현하지 못했다.

## Seed가 바뀌어도 반복됐을까

첫 모델은 seed 42 한 번으로 학습했습니다. Validation 299문항에서 정답이 42문항 늘어난 건 큰 변화지만, seed가 바뀌어도 같은 절차가 유지되는지는 알 수 없었습니다.

그래서 SFT부터 hard-negative scoring, MIO까지 다시 실행하도록 보완 실험을 구성했습니다. 완료된 결과는 다음과 같습니다.

| seed | SFT `acc_norm` | MIO-hard `acc_norm` |     변화 |
| ---: | -------------: | ------------------: | -------: |
|   42 |         60.54% |              72.91% | +12.37%p |
|   43 |         59.53% |              72.58% | +13.04%p |
|   44 |         59.87% |              학습 중 오류로 폐기 |        - |

Seed 43에서는 SFT를 처음부터 다시 학습하고, 새 SFT 모델로 모든 선택지를 재채점해 47,087개 hard pair를 다시 만들었습니다. MIO 결과는 72.58%였습니다. Seed 42와 비교하면 한 문항 차이입니다.

두 개의 완전한 seed에서는 비슷한 개선이 반복됐습니다. SFT 점수는 조금 달랐지만, MIO 뒤에는 72%대 중후반으로 모였습니다.

Seed 44는 SFT 평가까지 완료했지만 MIO 단계는 새 Colab runtime에서 실행 상태가 끊겨 마치지 못했습니다. 

> 끝까지 실행한 seed 두 개에서 큰 개선이 반복됐다.  

조금 덜 완성된 문장이지만, 현재 실험 상태를 가장 정확하게 설명합니다.

## ARC가 좋아지면 모델 전체도 좋아질까

ARC Test를 마친 뒤 목표 밖 능력도 확인했습니다. Base와 최종 Hugging Face checkpoint를 같은 조건으로 비교했습니다.

| Benchmark            | Metric          |   Base | 최종 모델 |        변화 |
| -------------------- | --------------- | -----: | --------: | ----------: |
| HellaSwag            | `acc_norm`      | 81.17% |    85.43% |     +4.26%p |
| PIQA                 | `acc_norm`      | 82.48% |    85.75% |     +3.26%p |
| WinoGrande           | `acc`           | 75.37% |    80.03% |     +4.66%p |
| MMLU Humanities      | `acc`           | 56.43% |    53.18% |     -3.25%p |
| MMLU Social Sciences | `acc`           | 73.81% |    70.46% |     -3.35%p |
| WikiText             | word perplexity | 8.0848 |    8.7194 | +7.85% 악화 |

자연어 선택지를 평가하는 HellaSwag, PIQA, WinoGrande는 모두 좋아졌습니다. 반면 MMLU 두 분야와 WikiText는 나빠졌습니다.

MMLU는 정답 문장이 아니라 `A`, `B`, `C`, `D` 위치 기호의 likelihood를 비교합니다. 위치별로 보면 `B`, `D`가 정답인 문항에서 하락 폭이 더 컸습니다. 과학 QA를 answer text completion으로 학습하면서 label calibration이 달라졌을 가능성이 있습니다.

그렇다고 모든 하락을 answer format 차이로 설명할 수는 없습니다. WikiText는 선택지 기호를 사용하지 않는데도 perplexity가 나빠졌습니다. 도메인에 맞추는 과정에서 일반 언어 모델링 능력 일부가 손상됐을 가능성도 남습니다.

이 결과를 한마디로 benchmark overfitting이라고 부르기도 어렵습니다. HellaSwag, PIQA, WinoGrande는 오히려 좋아졌기 때문입니다. 이번 학습은 과학 QA에 모델을 특화했고, answer text completion이라는 출력 형식에도 영향을 줬습니다. 공개 학습 데이터와 ARC 사이의 중복 가능성은 또 다른 문제라서 뒤에서 따로 확인했습니다.

결국 최종 모델은 범용 Mistral-7B의 상위 호환 모델이 아닙니다.

> 과학·상식 객관식 answer-text discrimination에 특화된 모델이다.

Hugging Face 모델 카드에서도 chat이나 일반 instruction following 모델이 아니라 domain-specialized completion model이라고 명시했습니다.

## 데이터 중복은 없었을까

높은 Test 점수를 얻은 뒤에는 공개 corpus와 Test 사이 문자열 중복도 따로 확인했습니다.

Test ID가 같은 문항은 없었고, 질문과 전체 선택지가 완전히 같은 MCQ도 없었습니다. 다만 normalized question stem이 정확히 같은 문항은 9개, 문자열 유사도 0.90 이상인 문항은 18개 있었습니다. 이 중 13개는 MIO preference source와 대응했습니다.

18개 문항을 모두 제외하면 Test `acc_norm`은 76.54%에서 76.43%가 됐습니다. 차이는 -0.11%p였습니다.

이 결과는 18개의 lexical match만으로 높은 점수가 나온 것은 아니라는 점을 보여줍니다. 그렇다고 semantic overlap까지 없다는 뜻은 아닙니다. 표현이 크게 다른 paraphrase나 같은 과학 사실을 묻는 문제는 문자열 검사로 잡기 어렵습니다.

공개 benchmark와 공개 학습 데이터를 함께 쓰는 이상 이 위험을 완전히 제거하기는 어렵습니다. 할 수 있는 일은 데이터 경계를 먼저 고정하고, 발견한 overlap과 점수 변화를 함께 공개하는 것입니다.

## 이번 실험에서 확인한 결과

처음에는 결론이 단순해 보였습니다.

> Hard negative를 사용한 MIO가 ARC-Challenge를 크게 개선했다.

추가 검증을 마친 뒤에는 같은 문장을 그대로 쓰기 어려워졌습니다. 확인한 내용은 다음과 같습니다.

- Completion-only science SFT는 validation `acc_norm`을 57.53%에서 60.54%로 올렸습니다.
- Preference 단계를 거치자 72.91%까지 올랐습니다.
- 단순 chosen-only 추가 학습은 64.21%에 그쳐 preference 단계의 상승을 설명하지 못했습니다.
- SFT부터 다시 실행한 seed 43에서도 MIO-hard가 72.58%를 기록했습니다.
- Test `acc_norm`은 76.54%였습니다.

부분적으로만 확인한 내용도 있습니다.

- Hard negative는 QASC uniform 대조군보다 한 문항 높았습니다.
- 두 개의 완전한 seed에서 전체 절차가 비슷한 결과를 냈습니다.
- Test의 lexical overlap 18문항을 제외해도 초기 점수는 거의 유지됐습니다.

아직 확인하지 못한 내용은 더 중요합니다.

- MIO가 DPO보다 낫다는 증거는 없습니다. `acc_norm`은 같았습니다.
- Hard negative의 독립적인 우위는 입증하지 못했습니다.
- Seed 44 MIO가 없어 3-seed 재현성은 완료되지 않았습니다.
- 문자열 검사만으로 semantic contamination을 배제할 수 없습니다.
- MIO, hard negative, SFT reference와 길이 보정 사이의 interaction을 모두 분리하지 못했습니다.

현재 가장 정확한 결론은 이렇습니다.

> 과학 QA를 SFT한 모델이 높은 점수를 준 오답으로 preference pair를 만들고 reference-relative preference optimization을 적용한 전체 절차는 ARC-Challenge validation에서 큰 개선을 보였다. 이 개선은 두 개의 완전한 seed에서 반복됐고 chosen-only 학습만으로는 재현되지 않았다. 다만 hard negative와 MIO 각각의 독립적인 우위는 아직 확인되지 않았다.

## 점수보다 평가 절차가 더 어려웠다

처음 목표는 benchmark 점수를 높이는 일이었습니다. 실험을 마친 뒤 더 오래 남은 건 평가 절차였습니다. Test를 언제 열었는지, adapter와 parent가 정말 같은 계보인지, 높은 점수가 특정 중복 문항 때문은 아닌지 확인하는 일은 학습 코드를 짜는 일만큼 중요했습니다.

처음 숫자만 그대로 밀어붙였다면 글은 훨씬 단순했을 겁니다.

SFT로 조금 올랐고, MIO로 크게 올랐고, Test에서 76.54%를 얻었다.

하지만 그렇게 쓰면 가장 많이 배운 부분이 빠집니다. 높은 점수를 얻은 뒤에는 그 점수를 지키는 것보다 의심하는 편이 더 어렵습니다. 대조군을 추가하면 처음의 설명이 약해질 수 있고, 평가 코드를 다시 보면 이미 공개한 숫자의 전제가 바뀔 수도 있습니다.

이번에도 그랬습니다.

MIO가 DPO보다 좋다는 결론은 남지 않았습니다. Hard negative가 핵심이라는 주장도 약해졌습니다. 

그래도 중요한 결과는 남았습니다. 정답을 더 보여주는 것만으로는 부족했고, 모델이 높은 점수를 준 오답을 함께 학습한 preference 단계에서 큰 차이가 났습니다. 두 번의 완전한 학습에서 비슷한 결과가 나왔습니다. 동시에 도메인 특화가 다른 능력을 손상할 수 있다는 점도 확인했습니다.

처음에는 Mistral-7B의 ARC 점수를 얼마나 높일 수 있는지가 궁금했습니다.

지금은 조금 다른 질문이 남았습니다.

> 모델이 좋아졌다고 말하려면, 어디까지 확인해야 할까?


[^arc-paper]: Peter Clark et al., [Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge](https://arxiv.org/abs/1803.05457), 2018.

[^qlora]: Tim Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314), 2023.

[^mio-paper]: Xin Lv et al., [The Hidden Link Between RLHF and Contrastive Learning](https://arxiv.org/abs/2506.22578), ICLR 2026.

[^dpo-paper]: Rafael Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290), NeurIPS 2023.
