---
title: 토크나이저, 언어 모델의 보이지 않는 관문
author: psymon
pubDatetime: 2026-03-28T11:21:00Z
modDatetime: 2026-03-28T11:21:00Z
slug: tokenizer-the-invisible-gate
featured: true
draft: false
tags:
  - LLM
  - Tokenizer
  - NLP
  - BPE
description: 언어 모델이 어떻게 세상을 읽는지 그리고 그 읽는 방식이 언어 모델 능력에 어떤 흔적을 남기는지에 관한 긴 이야기.
---

## Table of contents

## 들어가며 - Low level에서 시작하는 이유

언어 모델을 공부하다 보면 쉽게 간과하는 부분이 있다. 바로 토크나이저다. 트랜스포머 구조를 이해하고, 어텐션 메커니즘을 머릿속에 그릴 수 있고, RLHF가 무엇인지 설명할 수 있는데도, 정작 모델이 ‘hello’라는 문자열을 어떻게 숫자로 바꾸는지는 흐릿하게 알고 있는 경우가 많다. 나 역시 그랬다. 대강 BPE라는 알고리즘이 있고 단어를 조각으로 쪼개는 무언가라는 정도의 이해로 꽤 오랜 시간을 보냈다.

그러다 한국어 모델을 직접 만지기 시작하면서 뭔가 이상하다는 것을 느꼈다. 한국어는 영어에 비해 두 배 이상 많은 토큰이 필요했다. 영어로 물으면 2초 만에 끝날 응답이 한국어로는 5초 가까이 걸렸다. 비용도 정확히 그만큼 더 들었다. 이 현상의 뿌리에는 우리가 쉽게 지나치는 가장 낮은 단계 컴포넌트, 바로 **토크나이저**(Tokenizer)가 있다.

토크나이저는 언어 모델의 보이지 않는 관문이다. 모든 입력이 이곳을 통해 들어가고 모든 출력이 이곳을 통해 나온다. 따라서 이 관문을 어떻게 설계하느냐에 따라 모델이 언어를 다루는 방식과 연산 비용이 달라지고, 심지어는 최종 성능까지 달라진다. 이 글은 바로 그 관문에 대한 이야기다.

분량이 조금 길다. 토크나이저를 처음 접하는 사람부터 이미 BPE를 구현해 본 사람까지 모두 읽을 수 있도록 썼다. 각 섹션은 독립적으로 읽어도 무방하니 필요한 곳만 골라 읽어도 좋다. 다만 가능하면 처음부터 끝까지 읽어보길 권한다. 중간중간 연결되는 지점들이 있기 때문이다.

## 토크나이저란 무엇인가

언어 모델은 숫자만 이해한다. 더 정확히 말하자면 벡터만 이해한다. 'hello'라는 문자열이 모델에 들어가려면 어떤 식으로든 숫자로 변환해야 한다. 이 변환을 담당하는 것이 토크나이저다.

변환은 크게 두 단계로 나뉜다. 첫 번째는 **토큰화**(tokenization)다. 문자열을 의미 있는 단위로 쪼개는 작업이다. 'hello world'라는 문자열이 `['hello', ' world']`가 되거나, `['he', 'llo', ' wor', 'ld']`가 되거나, `['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']`가 될 수 있다. 어떻게 쪼갤 것인지는 토크나이저의 설계에 따라 달라진다.

두 번째는 **인코딩**(encoding)이다. 쪼갠 토큰들을 정수 ID로 매핑하는 작업이다. `['hello', ' world']`가 `[31373, 995]`와 같은 숫자 배열이 된다. 이 숫자들은 모델 내부에서 임베딩 벡터로 변환되어 트랜스포머 레이어로 들어간다.

정리하면 토크나이저는 이런 파이프라인을 가진다.

```
원시 텍스트 → [전처리] → [분절] → [토큰 ID 매핑] → 정수 시퀀스
"hello world"                                   [31373, 995]
```

반대 방향도 있다. 모델이 출력한 토큰 ID 시퀀스를 다시 사람이 읽을 수 있는 문자열로 되돌리는 **디코딩(decoding)** 과정이다. 이상적으로는 인코딩과 디코딩이 완벽하게 역함수 관계여야 하지만 실제로는 그렇지 않은 경우가 꽤 많다. 이 부분은 뒤에서 다시 다루겠다.

한 가지 흥미로운 점은 토크나이저가 **데이터로부터 학습하여 만드는 컴포넌트**라는 것이다. 모델 가중치를 학습시키듯, 토크나이저 어휘집과 분절 규칙도 말뭉치로부터 만든다. 다만 이 과정은 신경망 학습과는 전혀 다른 방식으로 진행된다. 대부분은 통계적 규칙 기반 알고리즘으로 어휘집을 구성하며, 한 번 만든 뒤에는 일반적으로 다시 학습시키지 않는다. 즉 토크나이저는 **모델 가중치와 독립적으로 고정하는 구성 요소**다. 이 점이 여러 흥미로운 문제를 낳는다는 사실도 뒤에서 살펴볼 것이다.

## 왜 토크나이저가 필요한가 - 역사적 맥락

토크나이저가 지금의 형태를 갖추기까지는 꽤 긴 여정이 있었다. 그 여정을 간단히 훑어보면 현재 설계를 이해하는 데 도움이 된다.

### 단어(Word) 수준 토큰화의 한계

NLP 초기에는 단어(word) 단위로 토큰화하는 것이 자연스러운 선택이었다. 공백으로 문장을 쪼개고 각 단어에 고유한 ID를 부여하는 방식이다. 이 접근은 직관적이지만 심각한 문제가 있다.

첫 번째는 **어휘집 크기 폭발**이다. 영어만 해도 단어의 수는 수십만 개에 달하고, 굴절(inflection)이나 파생(derivation)까지 고려하면 수백만 개로 불어난다. 한국어처럼 교착어 계열의 언어는 더 심각하다. '먹다'라는 동사 하나에서 '먹는다', '먹었다', '먹겠다', '먹어야겠다', '먹을까'와 같은 수많은 활용형이 파생된다. 이들을 모두 별개의 토큰으로 다루면 어휘집이 과도하게 커진다.

두 번째는 **미등록어(OOV, Out-of-Vocabulary) 문제**다. 학습 데이터에 없던 단어가 추론 시점에 등장하면 모델은 이를 처리할 방법이 없다. 새로운 고유명사, 신조어, 오타까지 포함하면 이 문제는 실전에서 치명적이다. 보통 `<UNK>`라는 특수 토큰으로 대체하는데, 이렇게 되면 모델이 그 토큰에서 의미 있는 정보를 얻지 못한다.

세 번째는 **형태론적 정보의 손실**이다. 'unhappiness'라는 단어를 하나의 토큰으로 취급하면 'un-', 'happy', '-ness'라는 형태소의 조합이라는 정보가 사라진다. 모델은 'happiness'와 'unhappiness'의 관계를 순수하게 벡터 공간에서만 학습해야 한다.

### 문자(Char) 수준 토큰화의 한계

단어 수준 토큰화의 대안으로 문자(character) 단위 토큰화가 있다. 이 방식은 어휘집이 매우 작고(영어라면 알파벳 26자 + 특수문자) 동시에 미등록어 문제가 완전히 사라진다. 어떤 문자열이든 문자로 쪼갤 수 있기 때문이다.

하지만 문자 단위 토큰화는 치명적 단점이 있다. 시퀀스 길이가 폭발적으로 늘어난다는 것이다. 'hello'라는 단어 하나가 5개의 토큰이 된다. 트랜스포머의 어텐션 연산 복잡도는 시퀀스 길이에 대해 $O(n^2)$이기 때문에, 시퀀스 길이가 5배가 되면 연산량이 25배가 된다. 실용적이지 않다.

게다가 문자 단위에서는 의미 학습이 어렵다. 'h'라는 문자 하나가 가진 의미는 맥락에 따라 너무 다양해서 모델이 의미를 학습하기 위해 매우 긴 문맥을 봐야 한다.

### 절충 - 서브워드 토큰화의 등장

단어는 너무 크고, 문자는 너무 작다. 그래서 등장한 것이 **서브워드(subword)** 단위 토큰화다. 자주 등장하는 단어는 하나의 토큰으로, 드물게 등장하는 단어는 더 작은 조각으로 쪼개는 방식이다. 'unhappiness'는 `['un', 'happiness']` 혹은 `['un', 'happy', 'ness']`로 쪼개진다. 미등록어도 어떤 식으로든 기존의 서브워드 조합으로 표현할 수 있다.

이 아이디어 자체는 1994년 데이터 압축 알고리즘으로 제안된 **BPE(Byte Pair Encoding)**[^BPE]에서 출발했다. 이것이 NLP에 본격적으로 도입된 것은 2016년 Sennrich et al.의 논문 <cite>Neural Machine Translation of Rare Words with Subword Units</cite>에서였다. 이후 BERT의 WordPiece, GPT의 BPE, T5의 Unigram 등 다양한 변주가 등장하며 오늘날 언어 모델의 기본 토크나이저 설계로 자리 잡았다.

## 토크나이저의 종류와 내부 동작

이제 각 서브워드 토크나이저가 어떻게 동작하는지 자세히 살펴보자. 먼저 짚어 둘 점이 하나 있다. 뒤에서 "BPE, WordPiece, Unigram LM, SentencePiece, tiktoken"을 차례로 설명하지만, 앞의 셋은 어휘집 학습 **알고리즘**이고 뒤의 둘은 이 알고리즘을 담아 내는 **구현체**다. 

### BPE — 가장 널리 쓰이는 알고리즘

BPE의 학습 알고리즘은 간단하다. 의사코드(pseudo-code)로 표현하면 다음과 같다.

1. 모든 단어를 문자 단위로 분해한다.
2. 가장 빈번하게 인접 등장하는 문자 쌍을 찾는다.
3. 그 쌍을 새로운 토큰으로 병합(merge)한다.
4. 원하는 어휘집 크기에 도달할 때까지 2~3을 반복한다.

예를 들어 말뭉치가 `low`, `lower`, `newest`, `widest` 네 단어로만 구성되어 있다고 하자. 초기 상태는 문자 단위 분해다.

```
l o w _        (5회 등장)
l o w e r _    (2회)
n e w e s t _  (6회)
w i d e s t _  (3회)
```

여기서 `_`는 단어 경계를 나타내는 특수 문자다(Sennrich et al. 원 논문에서는 `</w>`를 쓰며, 구현체마다 표기가 다르다. 여기서는 가독성을 위해 `_`로 쓴다). 가장 빈번한 문자 쌍을 찾아보면 `e s`가 9회(`newest` 6 + `widest` 3)로 최다다. 이를 병합하여 `es`라는 새 토큰을 만든다.

```
l o w _
l o w e r _
n e w es t _
w i d es t _
```

다음으로 `es t`가 9회로 최다다. 이를 병합하여 `est`를 만든다. 이 과정을 반복하며 어휘집을 확장한다.

추론 시에는 학습 때 찾아낸 병합 규칙을 순서대로 적용한다. 새로운 단어 `lowest`가 들어오면 먼저 문자로 분해한 뒤, 1번 병합 규칙부터 차례로 적용해 최종적으로 `['low', 'est']` 같은 결과를 얻는다.

이 단순한 알고리즘이 강력한 이유는 빈도 기반이라는 점에 있다. 자주 등장하는 문자열은 자연스럽게 하나의 토큰으로 묶이고, 드문 문자열은 작은 조각으로 남는다. 결과적으로 효율과 유연성의 균형이 잡힌다.

### WordPiece — BERT 계열의 선택

WordPiece는 BERT와 그 후속 모델들이 채택한 알고리즘이다. BPE와 매우 유사하지만 병합 기준이 다르다. BPE가 단순 빈도를 쓰는 반면, WordPiece는 **가능도**(likelihood) 기반 점수를 쓴다.

$$
\text{score}(x, y) = \frac{\text{freq}(xy)}{\text{freq}(x) \cdot \text{freq}(y)}
$$

즉 두 토큰 $x$, $y$를 병합했을 때의 빈도를 각각의 빈도의 곱으로 나눈 값이다. 이 값이 크다는 것은 $x$와 $y$가 독립적으로 나타나기보다 함께 나타나는 경향이 강하다는 뜻이다. 이 기준은 본래 말뭉치의 언어 모델 가능도를 최대화하는 병합을 찾는 과정에서 유도되며, 로그를 취하면 상호정보량(pointwise mutual information, PMI) 형태가 된다. BPE의 단순 빈도 기준과 비교하면, WordPiece는 "얼마나 자주 붙어 나오는가"뿐 아니라 "각자 얼마나 자주 등장하는가"까지 함께 고려하는 셈이다.

실무적 차이로는 WordPiece가 단어 내부의 서브워드를 `##`이라는 접두사로 표시한다는 점이 있다. `playing`이 `['play', '##ing']`으로 토큰화된다. 이는 디토큰화 시 원래 문자열로 정확히 복원하는 데 도움이 된다.

### Unigram Language Model — 확률적 관점

Kudo가 2018년 논문에서 제안한 Unigram LM은 앞선 두 알고리즘과는 접근이 완전히 다르다. BPE와 WordPiece가 작은 어휘집에서 출발해 병합으로 확장하는 '바텀업' 방식이라면, Unigram은 처음에 매우 큰 후보 어휘집을 만들어 놓고 하나씩 제거하는 '탑다운' 방식이다.

학습 과정은 다음과 같다.

1. 매우 큰 초기 어휘집을 만든다. 실무에서는 모든 부분 문자열을 나열하지 않고, Suffix Array나 빈도 컷오프 같은 휴리스틱으로 후보를 추린다.
2. EM(expectation-maximization) 알고리즘으로 각 서브워드의 확률 $p(x_i)$를 추정한다.
3. 전체 말뭉치의 가능도를 가장 덜 감소시키는 서브워드들을 어휘집에서 제거한다.
4. 원하는 크기에 도달할 때까지 2~3을 반복한다.

Unigram의 매력은 하나의 문자열을 여러 방식으로 토큰화할 수 있다는 점에 있다. 예를 들어 `hello`를 `['hello']`, `['he', 'llo']`, `['hel', 'lo']` 중 어느 방식으로 쪼개든 확률 모델에서 각각의 가능도를 계산할 수 있다. Kudo는 같은 논문에서 이 성질을 이용한 **서브워드 정규화**(subword regularization)라는 데이터 증강 기법도 함께 제안했다. 학습 시 매번 다른 토큰화를 샘플링하여 모델이 특정 분절에 갇히지 않도록 하는 방법이다.

### SentencePiece — 언어 독립적 구현

SentencePiece는 알고리즘이라기보다 구글이 개발한 구현체다. BPE와 Unigram을 모두 지원한다. SentencePiece의 핵심 특징은 언어 독립적이라는 점이다.

기존 BPE 구현은 대부분 공백을 기준으로 단어를 나눈 뒤 서브워드를 찾는 방식이었다. 이는 영어처럼 공백으로 단어가 구분되는 언어에서는 자연스럽지만, 중국어나 일본어처럼 공백이 없는 언어에서는 어려움을 겪는다. 한국어처럼 공백이 있지만 형태소 경계와 공백이 일치하지 않는 언어도 까다롭다.

SentencePiece는 이 문제를 단순하고 우아하게 해결한다. 공백 자체를 하나의 문자로 취급하는 것이다. 정확히는 공백을 `▁`(U+2581)이라는 특수 문자로 치환한다. `Hello world`는 `▁Hello▁world`가 되고, 이 문자열 전체를 대상으로 BPE 혹은 Unigram을 수행한다. 이렇게 하면 공백이 없는 언어든, 복잡한 형태론을 가진 언어든 동일하게 처리할 수 있다.

또 다른 장점은 **완전한 가역성**(reversibility)이다. 일반 BPE는 공백을 별도 전처리로 떼어 내기 때문에 토큰화 후 디토큰화하면 원래 문자열과 미묘하게 달라질 수 있다. SentencePiece는 공백을 `▁` 문자로 취급하므로 토큰을 이어 붙이고 `▁`를 공백으로 되돌리는 것만으로 원래 문자열을 정확히 복원할 수 있다.

이런 장점 덕에 LLaMA, LLaMA2, T5, XLNet, ALBERT 등 많은 현대 모델이 SentencePiece를 기반으로 한다.

### tiktoken — OpenAI의 선택

OpenAI는 자체 BPE 구현체인 tiktoken을 만들어 사용한다. tiktoken의 가장 큰 특징은 바이트 수준(byte-level) BPE라는 점이다. 문자가 아닌 UTF-8 바이트를 최소 단위로 다룬다. 이렇게 하면 이론상 어떤 유니코드 문자열도 깨지지 않고 토큰화할 수 있다. 이모지, 희귀 언어, 수식 기호 무엇이든 바이트 단위로 표현 가능하기 때문이다. 초기 어휘집은 256개의 바이트에서 시작한다.

또 다른 특징은 **pre-tokenizer regex**의 존재다. tiktoken은 BPE를 수행하기 전에 정규식으로 텍스트를 미리 쪼갠다. 이는 "어떤 문자 쌍이 서로 병합될 수 있는가"를 제약하는 역할을 한다. 예를 들어 공백과 알파벳은 같은 토큰 안에 올 수 있지만, 숫자와 알파벳은 같은 토큰이 될 수 없도록 막을 수 있다. 이 pre-tokenizer 설계는 모델마다 조금씩 다른데, r50k_base와 p50k_base(GPT-3 계열)로 시작해 cl100k_base(GPT-3.5/4), o200k_base(GPT-4o)로 이어지며 tiktoken 계보에서 계속 진화해 왔다.

LLaMA 3도 tiktoken과 호환되는 BPE 규격으로 전환했으며, EXAONE 등 여러 한국어 LLM도 유사한 접근을 취한다. 다만 각 모델의 pre-tokenizer 정규식은 서로 다르며, 이 차이가 한국어 토큰화 효율에 상당한 영향을 미친다.

## 파이썬으로 구현해보는 BPE

이론만으로는 감이 오지 않을 수 있으니, 최소 구현을 통해 BPE를 직접 체험해 보자. 다음은 교육 목적 파이썬 BPE 토크나이저다.

```python
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

class SimpleBPE:
    def __init__(self):
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}

    def _get_word_freqs(self, corpus: List[str]) -> Dict[Tuple[str, ...], int]:
        """말뭉치에서 단어 빈도를 추출하고 문자 단위로 분해"""
        freqs: Dict[Tuple[str, ...], int] = Counter()
        for text in corpus:
            for word in text.split():
                # 단어 끝에 </w> 기호를 붙여 단어 경계를 표시
                chars = tuple(list(word) + ["</w>"])
                freqs[chars] += 1
        return freqs

    def _get_pair_stats(self, word_freqs: Dict[Tuple[str, ...], int]) -> Counter:
        """인접한 문자 쌍의 빈도를 계산"""
        pairs = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    def _merge_pair(
        self,
        pair: Tuple[str, str],
        word_freqs: Dict[Tuple[str, ...], int]
    ) -> Dict[Tuple[str, ...], int]:
        """가장 빈번한 쌍을 하나의 토큰으로 병합"""
        new_freqs = {}
        merged = pair[0] + pair[1]
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                # 현재 위치와 다음 위치가 병합 대상 쌍과 일치하면 합침
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_freqs[tuple(new_word)] = freq
        return new_freqs

    def train(self, corpus: List[str], num_merges: int = 100):
        """말뭉치로부터 BPE 규칙을 학습"""
        word_freqs = self._get_word_freqs(corpus)

        for i in range(num_merges):
            pairs = self._get_pair_stats(word_freqs)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            word_freqs = self._merge_pair(best_pair, word_freqs)
            self.merges.append(best_pair)

        # 최종 어휘집 구축
        vocab_set = set()
        for word in word_freqs:
            for token in word:
                vocab_set.add(token)
        self.vocab = {token: idx for idx, token in enumerate(sorted(vocab_set))}

    def tokenize(self, text: str) -> List[str]:
        """학습된 규칙을 적용하여 새 텍스트를 토큰화"""
        result = []
        for word in text.split():
            tokens = list(word) + ["</w>"]
            # 학습된 순서대로 병합 규칙을 적용
            for pair in self.merges:
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                        new_tokens.append(pair[0] + pair[1])
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
            result.extend(tokens)
        return result


# 사용 예시
corpus = [
    "low lower lowest",
    "new newer newest",
    "wide wider widest",
    "slow slower slowest",
]

bpe = SimpleBPE()
bpe.train(corpus, num_merges=20)

print("학습된 병합 규칙:")
for i, merge in enumerate(bpe.merges[:10]):
    print(f"  {i+1}. {merge[0]} + {merge[1]} -> {merge[0]+merge[1]}")

print("\n토큰화 결과:")
print(bpe.tokenize("lowest newer"))
```

이 구현은 매우 단순하지만 BPE의 핵심은 담고 있다. 실제로 실행해 보면 학습 말뭉치에서 가장 빈번한 'lo', 'st' 등이 먼저 병합 규칙으로 등장하는 것을 확인할 수 있다.

프로덕션 수준 구현은 이보다 훨씬 복잡하다. 우선 성능 최적화가 필요하다. 위 구현은 매 병합마다 전체 말뭉치를 다시 순회하는데, 실제 구현(예: Hugging Face의 `tokenizers` 라이브러리)은 우선순위 큐를 사용해 변화가 있는 부분만 업데이트한다. 또한 바이트 수준 처리, 정규식 기반 pre-tokenization, 캐싱, 멀티스레딩 등이 추가된다. 하지만 알고리즘의 핵심은 이 50줄 남짓한 코드에 모두 담겨 있다.

SentencePiece나 tiktoken을 사용하는 예시도 보고 넘어가자. 실무에서는 보통 이런 라이브러리를 사용한다.

```python
# tiktoken 사용 예시 (OpenAI 모델 호환)
import tiktoken

# GPT-4o 토크나이저 로드
enc = tiktoken.get_encoding("o200k_base")

text = "안녕하세요, 오늘은 토크나이저에 대해 공부해 봅시다."
tokens = enc.encode(text)
print(f"토큰 수: {len(tokens)}")
print(f"토큰 ID: {tokens}")
print(f"각 토큰 디코딩: {[enc.decode([t]) for t in tokens]}")

# 디코딩 - 완전 가역
print(f"복원: {enc.decode(tokens)}")
```

```python
# SentencePiece로 직접 학습하기
import sentencepiece as spm

# 학습용 말뭉치 파일 준비 후
spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=32000,  # 말뭉치 크기에 따라 조정
    model_type="bpe",  # 혹은 "unigram"
    character_coverage=0.9995,  # 한국어는 0.9995 권장
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
)

# 학습된 토크나이저 로드 및 사용
sp = spm.SentencePieceProcessor()
sp.load("my_tokenizer.model")

text = "토크나이저 학습 예제입니다."
pieces = sp.encode_as_pieces(text)
ids = sp.encode_as_ids(text)
print(pieces)  # ['▁토크나이저', '▁학습', '▁예제', '입니다', '.']
print(ids)
```

## 한국어 토크나이저의 특수한 사정

앞서 살펴봤듯 한국어 토큰화는 영어보다 효율이 떨어진다. 이 문제는 단순히 '한국어가 어려운 언어'라서 생기는 것이 아니라 여러 구조적 원인이 얽혀 있다.

### 바이트 수준 BPE와 한글

대부분의 현대 LLM은 바이트 수준 BPE를 사용한다. 그런데 한글 음절(U+AC00~U+D7A3, 가-힣)은 UTF-8로 인코딩할 때 한 글자당 3바이트를 차지한다. 영어 알파벳이 1바이트인 것과 대조적이다. '가'라는 한 글자가 `0xEA 0xB0 0x80`이라는 세 바이트의 시퀀스가 된다.

학습 말뭉치에 영어가 압도적으로 많은 토크나이저의 경우(대부분의 글로벌 LLM이 그렇다), 한글 병합이 충분히 일어나지 않는다. 결과적으로 많은 한글 문자가 2~3개의 바이트 토큰으로 쪼개진다. '안녕하세요'라는 다섯 글자가 10개 이상의 토큰이 되는 경우도 흔하다.

이것이 왜 문제인가.

첫째, **비용(cost)**. LLM API는 토큰 단위로 과금한다. 같은 내용을 한국어로 주고받으면 영어로 했을 때보다 2~3배 비싼 요금을 내야 한다.

둘째, **지연(latency)**. 추론 속도는 생성할 토큰 수에 비례한다. 한국어 응답이 체감상 느린 이유다.

셋째, **문맥 윈도우(context window) 소모**. 컨텍스트 길이가 128K 토큰인 모델이라도, 한국어로는 실질적으로 40~50K 문자 정도밖에 담지 못한다.

이 문제는 Petrov et al.의 2023년 연구 *Language Model Tokenizers Introduce Unfairness Between Languages*에서 체계적으로 분석했다. 저자들은 다국어 토크나이저의 'tokenization fertility'(문자당 토큰 수)를 언어별로 측정했고, 일부 언어는 영어 대비 최대 15배의 토큰을 소모한다는 결과를 보고했다. 이는 단순히 효율의 문제를 넘어 언어 간 형평성(equity)의 문제이기도 하다.

### 교착어 특성과 형태소

한국어는 교착어다. 어근에 조사와 어미가 붙어 의미를 확장한다. `'학교에서는'`은 `'학교'` + `'에서'` + `'는'`이라는 세 형태소의 조합이다. 이상적인 토크나이저라면 이러한 형태소 경계를 존중해야 한다.

하지만 통계 기반 BPE는 형태소 정보를 알지 못한다. `'에서는'`이 자주 등장하면 하나의 토큰이 될 것이고, `'학교에'`가 자주 등장하면 또 그것이 한 토큰이 될 것이다. 결과적으로 같은 형태소가 서로 다른 토큰 경계로 쪼개지는 일이 빈번하다.

이 문제를 완화하기 위해 전처리 단계에 형태소 분석기를 결합하는 방식이 있다. Mecab-ko, KoNLPy 계열 도구로 먼저 형태소 경계를 분석한 뒤, BPE를 학습시켜 병합이 형태소 경계를 넘지 않도록 유도하는 방식이다. 최근 한국어 특화 LLM 대다수가 이 진영이다. **EXAONE 3.0**은 교착어 특성을 반영해 한국어 말뭉치를 MeCab으로 pre-tokenize한 뒤 vocab 102,400의 BBPE(byte-level BPE)를 학습시켰고[^exaone3], **HyperCLOVA X**도 vocab 100,000의 morpheme-aware byte-level BPE를 사용한다[^hcx]. 이 방식은 형태소 경계를 존중한다는 장점이 있지만 분석기 오류가 학습에 전파될 수 있고, 구현에 따라서는 추론 시점까지 형태소 분석기 의존성이 이어지기도 한다.

다른 방향의 접근도 있다. 형태소 분석 없이 한국어 말뭉치 비중을 크게 높여 BPE가 자연스럽게 한국어 패턴을 학습하도록 하는 방식이다. 대표 사례는 **Polyglot-Ko**로, 전처리 후 863GB(처리 전 1.2TB) 한국어 데이터로 vocab 30,003의 BPE를 학습했다[^polyglot]. 카카오브레인의 **KoGPT**도 유사한 접근이다[^kogpt]. 순수 데이터 기반이므로 분석기 오류가 없고 딥러닝 친화적이지만 어휘집이 한국어에 많이 할애되어야 하므로 다국어 지원 시 Trade-off가 생긴다.

최근에는 제3의 방향도 등장했다. 형태소 분석기 없이 vocab 크기를 과감히 키우고, **SuperBPE**[^superbpe]처럼 공백을 넘는 병합을 허용하는 기법으로 한국어 토큰 효율을 개선하는 접근이다. 2026년 공개된 **K-EXAONE**[^kexaone]은 기존 EXAONE 계열의 vocab 100K를 150K로 재설계하면서 SuperBPE 전략을 도입해 superword 토큰이 전체 어휘의 약 20%를 차지하도록 했고, 이를 통해 평균 약 30%의 토큰 효율 개선을 보고했다.


### Pre-tokenizer regex의 미묘한 차이

GPT-4o의 o200k_base와 LLaMA 3의 정규식을 비교해 보면, o200k_base는 `\p{L}`을 대소문자 카테고리(`\p{Lu}`, `\p{Ll}` 등)로 쪼개 라틴 계열의 CamelCase나 분음기호 조합을 더 정밀하게 포착한다. 반면 LLaMA 3는 단일 `\p{L}+`를 쓰는 더 단순한 정규식을 유지한다. 그러나 한글은 대소문자 구분이 없는 `\p{Lo}` 카테고리에 속하므로, o200k_base의 이 세분화는 한글 처리에 사실상 영향을 주지 않는다 — 두 정규식 모두 한글을 별도로 취급하지 않는다는 점에서 동일하다.

한국어에 영향을 주는 변수는 따로 있다. 선행 공백을 토큰에 붙이는 방식, 숫자를 몇 자리 단위로 끊는지, 그리고 `\p{Lo}` 연속 문자열을 한 덩어리로 묶을지 잘게 쪼갤지 같은 선택이다. 이 설계들이 vocab 할당과 결합하여 최종적으로 한국어 토큰화 효율을 결정한다.

이런 설계 선택은 동일한 모델 크기와 학습 데이터라도 **실제 서비스 성능(토큰당 바이트, 추론 비용, 한국어 다운스트림 품질)에** 상당한 영향을 미친다. 그래서 한국어 LLM을 개발할 때는 토크나이저 설계를 매우 신중하게 해야 한다. 한 번 확정한 토크나이저는 이후 모델 가중치와 묶여 수정하기 어렵기 때문이다.

## 토크나이저를 어떻게 평가할 것인가

토크나이저의 성능을 평가하는 것은 생각보다 까다롭다. 여러 지표가 있고 각기 측정하는 속성이 다르다. 게다가 토크나이저의 "좋음"은 결국 언어 모델의 학습·추론 품질에 간접적으로 드러나는 것이라, 토크나이저만 따로 떼어 점수를 매기는 일은 언제나 부분적일 수밖에 없다.

### Fertility - 단위당 토큰 수

가장 직관적인 지표는 **fertility**다. 텍스트 한 단위를 토큰화했을 때 몇 개의 토큰이 나오는지를 나타낸다. 영어권에서는 단어당 토큰 수(tokens/word)로, 다국어 비교에서는 문자당 토큰 수로 쓴다.

$$
\text{fertility} = \frac{\text{토큰 수}}{\text{단어 수 혹은 문자 수}}
$$

낮을수록 같은 내용을 더 적은 토큰으로 표현한다는 뜻이다. 계산이 쉽고 직관적이어서 비용·지연·문맥 효율을 논할 때 가장 먼저 보는 수치다.

fertility는 **그 자체로는 품질 지표가 아니다**. 어휘집을 키우면 fertility는 내려가지만 임베딩 행렬이 선형으로 커져 메모리가 늘고 희귀 토큰은 제대로 학습되지 않는다. fertility가 낮다고 해서 좋은 토크나이저인 것은 아니라는 뜻이다.

### 압축률과 정보밀도

fertility의 단점을 보완하는 것이 **BPB**(bits-per-byte) 혹은 **BPC**(bits-per-character)다. 이는 "모델이 한 문자(혹은 바이트)를 예측하는 데 몇 비트의 정보가 필요한가"를 나타낸다. 낮을수록 모델이 텍스트를 잘 압축한다는 뜻이다.

$$
\text{bpb} = \frac{-\sum_{i=1}^{N} \log_2 p(t_i \mid t_{<i})}{B}
$$

분자는 토큰 $t_1, \dots, t_N$에 대한 모델의 log-likelihood 합이고, 분모 $B$는 원 텍스트의 바이트 수다. 분자는 토큰 공간에서 계산하지만 분모는 토크나이저와 무관한 바이트 수라는 점이 핵심이다.

이 비대칭 덕분에 BPB는 **토크나이저가 달라도 공정하게 비교 가능**하다. Perplexity를 그대로 비교하면 토크나이저가 다른 순간 값 자체가 왜곡되는데, BPB는 원 텍스트라는 공통 기준으로 정규화하니 그 문제를 피한다. 이를 강조한 것이 Rust et al.의 2021년 연구 *How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models*다. 저자들은 토크나이저가 다운스트림 성능에 미치는 영향을 정량화하기 위해 BPB를 핵심 지표로 썼다.

단, BPB도 만능은 아니다. BPB가 재는 것은 모델의 압축 능력이지 다운스트림 태스크(QA, 추론, 코드 생성) 성능이 아니다. 최근 여러 연구는 BPB와 실제 태스크 성능의 상관이 완벽하지 않다는 것을 지적해 왔다. 토크나이저의 압축 성능과 실제 언어모델의 성능이 비례하는 않는다는 것이다. "좋은 토크나이저"를 탐색하려는 연구에서 가장 골치 아픈 부분이 바로 이 지점이다.

### 형태소 경계 보존율

언어학적 관점에서는 **형태소 경계를 얼마나 잘 보존하는가**도 중요한 지표다. 형태소 분석기 출력을 기준으로 삼고, 토크나이저가 만든 경계와의 일치율을 precision/recall로 계산한다.

한국어, 튀르키예어, 핀란드어처럼 형태론적으로 풍부한 언어에서 특히 중요하다. 다만 '좋은' 경계가 무엇인지는 논쟁의 여지가 있다. 형태소가 항상 의미 단위와 일치하는 것은 아니고, 딥러닝 모델은 언어학적 경계와 무관한 효율적 표현을 찾아낼 때도 많다는 증거도 쌓여 왔다(Bostrom & Durrett 2020 등). 해석 가능성에는 유리하지만, 다운스트림 성능과의 직접적 인과는 연구마다 결론이 엇갈린다.

### Intrinsic vs Extrinsic 평가

위의 지표들은 모두 **내재적(intrinsic) 평가**에 속한다. 토크나이저만 단독으로 평가하는 것이다. 반면 **외재적(extrinsic) 평가**는 토크나이저를 실제 모델에 탑재하고 다운스트림 태스크에서의 성능을 측정한다.

외재적 평가는 중요하지만 평가를 위한 비용이 크다. LLM을 매번 처음부터 학습해야 하기 때문이다. 그래서 실무에서는 intrinsic 지표로 후보를 걸러낸 후, 유망한 후보만 소규모 학습으로 검증하는 방식이 일반적이다.

## 토크나이저가 LLM에 미치는 영향

토크나이저가 모델 가중치와 독립적인 컴포넌트라는 사실 때문에 많은 사람이 이를 부수적인 것으로 취급한다. 하지만 토크나이저는 모델의 여러 핵심 동작에 깊은 영향을 미친다.

### 숫자 계산 능력과 토큰화

가장 흥미로운 사례는 **숫자 토큰화**다. 초기 LLM이 산술 계산에 약했다는 것은 유명한 사실인데, 그 원인 중 하나가 토크나이저에 있다.

GPT-2의 BPE는 숫자를 자릿값 구조와 무관하게 병합했다. 말뭉치에 `2023`이 자주 등장했다면 `['2023']`이 한 토큰이 되고, `1234`는 `['12', '34']`가 되는 식이다. 같은 네 자릿수라도 숫자마다 분절이 다르고 자릿값 경계가 보존되지 않는다. 모델 입장에서는 같은 연산(예: 덧셈의 자리 올림)을 학습하기 위해 서로 다른 토큰 조합 수백 가지를 별개로 익혀야 했다.

이 문제를 해결하기 위해 최근 모델들은 숫자 분절을 고정한다. LLaMA 1/2, Mistral, Gemma는 각 숫자를 개별 토큰으로 처리하고, GPT-4 계열과 LLaMA 3는 3자릿수 단위로 묶는 방식을 쓴다. 어느 쪽이든 자릿값 구조가 보존된다는 점이 중요하며, 이 작은 변경만으로도 산술 과제에서 상당한 성능 향상을 얻었다.[^arith].

### Glitch tokens - 학습되지 않은 토큰

2023년 초 GPT 계열에서 발견된 유명한 버그가 있다. `SolidGoldMagikarp`, `rrawerrer`, `petertodd` 같은 이상한 토큰들이 어휘집에 존재했는데, 이들을 입력하면 모델이 기괴한 반응을 보였다. 말을 더듬거나, 전혀 관련 없는 주제로 이탈하거나, 오류를 일으켰다[^glitch].

원인은 이랬다. 토크나이저 학습에 사용된 말뭉치(Reddit 스크래핑 데이터 등)에서 이 문자열들이 높은 빈도로 등장해 하나의 토큰으로 병합됐다. 그런데 이후 언어 모델 학습 단계에서는 필터링 파이프라인이 바뀌거나 해당 데이터 소스가 배제되면서 이 토큰들이 거의 등장하지 않았다. 결과적으로 모델은 이 토큰에 대응하는 임베딩 벡터를 제대로 학습하지 못했고, 무작위에 가까운 반응을 하게 됐다.

이 사례는 토크나이저와 모델의 데이터 불일치가 얼마나 미묘한 버그를 만들 수 있는지 보여 준다. 토크나이저를 설계할 때는 이후 쓰일 학습 말뭉치의 분포와 필터링 파이프라인까지 함께 고려해야 한다.

### 어휘집 크기의 선택

토크나이저의 어휘집 크기는 여러 트레이드오프를 만든다.

**큰 어휘집의 장점**: fertility가 낮아져 시퀀스 길이가 짧아지고, 추론 속도가 빨라지며, 같은 문맥 길이에 더 많은 내용을 담을 수 있다.

**큰 어휘집의 단점**: 임베딩 레이어와 최종 softmax 레이어의 파라미터가 커진다. 이 두 레이어는 각각 $V \times d$ 크기이며(weight tying으로 공유하기도 한다), 어휘집 크기 $V$가 10만에서 20만으로 두 배가 되면 해당 메모리도 두 배가 된다. 또한 학습 시 각 토큰당 노출되는 예시 수가 줄어들어 드문 토큰은 충분히 학습되지 않을 수 있다.

현대 LLM의 어휘집 크기는 32K에서 출발해 최근에는 100K~256K 범위를 사용한다. LLaMA 1의 32K, LLaMA 3의 128K, GPT-4o의 200K, K-EXAONE의 150K, Gemma 2의 256K처럼 최근 추세는 분명 크기를 키우고있다. 그러나 마냥 커질 수는 없다. 다국어 지원과 긴 문맥 요구가 상한을 키우는 만큼 임베딩 메모리와 드문 토큰 학습 부담이 하한을 붙잡고 있다.

### 프롬프트 인젝션과 보안

토크나이저의 특이한 동작이 보안 이슈로 이어지기도 한다. 대표적인 것이 **시각적으로는 동일하지만 토큰 ID가 다른 문자열**을 이용한 우회 공격이다. 일반 공백 대신 non-breaking space를, 일반 알파벳 대신 유사한 키릴 문자를 섞으면 사람 눈에는 같은 텍스트지만 모델은 완전히 다른 토큰 시퀀스로 받아들인다. 이를 이용해 콘텐츠 필터를 우회할 수 있다. 엄밀히 말하면 이건 유니코드 정규화 단계의 이슈지만, 토크나이저 파이프라인이 그 정규화를 어떻게 처리하느냐에 따라 취약성이 결정되므로 사실상 토크나이저 설계 문제로 분류된다.

앞서 다룬 glitch token도 보안 관점에서 재해석할 수 있다. 학습되지 않은 토큰을 입력에 섞으면 모델이 예측 불가능하게 반응하므로, 안전 장치를 우회하거나 의도하지 않은 출력을 유도하는 수단이 될 수 있다.

이런 사례들은 토크나이저 설계가 단순히 성능 문제가 아니라 시스템 신뢰성과 안전성의 문제이기도 함을 보여 준다.

## 최신 연구 동향

이 분야는 최근 빠르게 움직이고 있다. 몇 가지 주목할 만한 연구 방향을 정리해 본다.

### SuperBPE - 단어 경계를 넘어서

2025년 Liu et al.이 발표한 **SuperBPE**는 기존 BPE의 단어 경계 제약을 없앤 변형이다[^superbpe]. 일반 BPE는 pre-tokenizer가 단어 단위로 먼저 쪼개기 때문에 서로 다른 단어에 속한 문자 쌍은 병합되지 않는다. `of the`라는 자주 등장하는 구문조차 두 개의 토큰으로 남는다.

SuperBPE는 학습 과정을 두 단계로 나눈다. 1단계에서는 일반 BPE처럼 단어 내부에서만 병합을 수행하고, 2단계에서는 단어 경계 제약을 풀고 **교차 단어(cross-word) 병합**을 허용한다. 결과적으로 `of_the`, `in_the` 같은 고빈도 구문이 단일 토큰이 된다.

저자들이 8B 규모로 직접 사전학습해 비교한 실험에서 SuperBPE는 30개 다운스트림 태스크 평균 +4.0%p(MMLU에서 +8.2%p)의 절대적 성능 향상을 보였고, 동시에 추론 시 컴퓨팅을 약 27% 절감했다. 이 결과가 의미 있는 이유는 fertility 감소가 단순히 "토큰 효율" 지표만 개선하는 게 아니라 다운스트림 성능에도 전이된다는 점을 명시적으로 보였기 때문이다. 한국어 맥락에서는 앞서 언급한 K-EXAONE이 SuperBPE를 채택한 대표 사례다.

### Tekken - Mistral의 다국어 토크나이저

Mistral이 2024년 NeMo 공개와 함께 발표한 **Tekken**은 tiktoken 기반이지만 다국어, 특히 유럽 언어와 비라틴 문자 언어에 대한 효율을 크게 개선했다[^tekken]. 100개 이상의 언어로 학습했으며, **이전 Mistral의 SentencePiece 대비** 소스 코드·중국어·이탈리아어·프랑스어·독일어·스페인어·러시아어에서 약 30%, 한국어에서 약 2배, 아랍어에서 약 3배 효율이 개선되었다. 핵심은 언어별 비중을 세심하게 조정한 학습 말뭉치와 pre-tokenizer 정규식 튜닝이다.

### 토크나이저 없는 모델 - Byte-level Transformers

한편에서는 **토크나이저를 아예 없애려는 시도**도 꾸준히 이어지고 있다. ByT5[^byt5], MegaByte[^megabyte], MambaByte[^mambabyte] 같은 모델은 UTF-8 바이트 수준에서 직접 동작한다. 토크나이저로 인한 편향과 불평등을 제거할 수 있지만, **시퀀스 길이 폭발**이라는 오래된 문제가 다시 등장한다. 한글 음절 하나가 3바이트니까, 같은 내용을 바이트 시퀀스로 표현하면 토큰 시퀀스보다 훨씬 길어진다.

이를 해결하기 위한 아이디어도 다양하다. MegaByte는 바이트를 고정 크기 패치로 묶어 계층적 트랜스포머를 구성하고, Mamba 계열은 상태공간 모델로 어텐션의 $O(n^2)$ 병목을 회피한다. 2024년 Meta가 발표한 **Byte Latent Transformer**(BLT)는 한 걸음 더 나아가, 바이트 엔트로피에 따라 패치 경계를 **동적으로** 결정한다[^blt]. "예측하기 쉬운 구간은 길게, 어려운 구간은 짧게" 패치를 잡는 발상이다. BLT는 LLaMA 3와 같은 학습 예산에서 비슷하거나 더 나은 스케일링을 보였고, 노이즈가 포함된 입력에서는 토큰 기반 모델을 크게 앞섰다. 아직 대형 모델에서 BPE 기반 접근을 완전히 대체하지는 못했지만, 장기적으로 유력한 방향이다.

### Pre-tokenizer의 결정적 영향 - Wegmann et al. 2025

또 하나 주목할 만한 연구가 **Wegmann et al. 2025**다[^wegmann]. 이들은 BERT base를 여러 조건에서 직접 사전학습해, 토크나이저 설계의 어느 축이 다운스트림 성능에 가장 큰 영향을 미치는지 체계적으로 비교했다. 비교 축은 세 가지였다. 학습 말뭉치, 어휘집 크기, 그리고 pre-tokenizer.

결론은 흥미롭다. **Pre-tokenizer의 선택이 가장 큰 영향을 미치며, 그 영향은 말뭉치나 어휘집 크기보다 크다.** 또한 태스크 성격에 따라 최적의 토크나이저가 다르다는 점도 확인되었다 — 의미 중심 태스크(NLI 등)에서는 GPT-2 풍의 더 공격적인 pre-tokenizer가 유리하지만, 형태 민감 태스크(저자 식별, 방언 분류 등)에서는 LLaMA 3 풍의 더 보수적인 pre-tokenizer와 큰 어휘집이 유리하다는 것이다. 저자들은 이 결과를 바탕으로 Rényi entropy 같은 기존 intrinsic 지표보다 다운스트림 성능과 상관이 높은 새로운 proxy(토큰 존재 기반 로지스틱 회귀)도 제안했다.

이 연구가 흥미로운 이유는 앞서 "BPB와 다운스트림 성능의 상관이 완벽하지 않다"고 했던 지점에 대한 부분적 해답을 제시하기 때문이다. 적어도 **pre-tokenizer라는 축 하나만 놓고 보면** 설계 선택이 성능에 체계적으로 반영된다는 것이다.

이런 관점은 새로운 토크나이저를 설계할 때 유용하다. "BPE 변형을 하나 더 만든다"가 아니라 "pre-tokenizer·어휘집 학습 알고리즘·어휘집 크기"를 각각 명시적 선택으로 다룰 수 있게 해 주기 때문이다. 개인적으로 한국어 토크나이저 작업에서 이 연구의 영향을 많이 받았다.

## 실무 선택 가이드

지금까지의 내용을 실무 관점에서 정리해 본다. 새로운 프로젝트에서 토크나이저를 선택하거나 설계할 때 고려할 지점들이다.

### 기존 모델을 파인튜닝하는 경우

**기본은 원래 토크나이저를 그대로 쓰는 것이다.** 토크나이저를 바꾸면 임베딩 레이어를 다시 학습해야 하고, 이는 사실상 사전학습 일부를 재현하는 일이 된다.

예외는 세 가지다. 첫째, **어휘집 확장(vocab extension)**. 도메인 특화 용어나 한국어 토큰 몇백~몇천 개를 추가하는 정도는 실무에서 흔히 하는 일이다. 단, 새 토큰의 임베딩 초기화에 주의를 기울여야 하며(평균 임베딩, 서브워드 합성 등), 추가한 토큰이 실제로 학습되도록 충분한 continued pre-training이 필요하다. 둘째, **어휘집 교체(vocab swap)**. ZeTT(Minixhofer et al. 2024) 같은 zero-shot transfer 기법이 등장하면서 완전한 교체도 현실적 선택지가 됐다. 셋째, **충분한 재학습 예산이 있는 경우**. Dagan et al.(2024)의 경험적 결과에 따르면 약 50B 토큰 규모의 continued pre-training으로 토크나이저 교체의 영향을 거의 회복할 수 있다[^dagan].

### 새 모델을 처음부터 학습하는 경우

대상 언어와 도메인 분포를 명확히 정하고, 그에 맞는 말뭉치로 SentencePiece BPE나 tiktoken 스타일 BBPE를 새로 학습한다.

**어휘집 크기**는 최근 지형을 반영해 결정한다. 영어 단일 모델이라면 32K~64K로도 충분하지만, 다국어 혹은 한국어 중심이라면 최소 64K, 실전에서는 **100K~200K**가 보통이다(K-EXAONE 150K, GPT-4o 200K, Gemma 2 256K). 어휘집이 크다고 무조건 좋은 건 아니고, 드문 토큰의 표현 학습과 임베딩 메모리라는 상한이 있다는 점은 앞 섹션에서 다룬 대로다.

**한국어 비중**은 토크나이저 학습 말뭉치에서 경험적으로 높게 잡는 게 좋다. 영어가 다수인 혼합 말뭉치에서는 한국어 병합이 충분히 일어나지 않아 fertility가 악화된다. 공개된 정확한 수치는 없지만, 내 경험에서는 한국어 중심 모델이면 학습 말뭉치의 한국어 비중을 상당히 높여야 의미 있는 한국어 fertility가 나온다. 한편 morpheme-aware 방식을 택할지(EXAONE·HyperCLOVA X 계열) 순수 데이터 기반으로 갈지(Polyglot-Ko 계열) 혹은 vocab을 과감히 키워 SuperBPE 스타일로 풀지(K-EXAONE)는 앞선 섹션들의 트레이드오프를 바탕으로 판단한다.

### 다국어 모델을 개발하는 경우

각 언어의 fertility를 측정하고 격차를 확인한다. 단, **말뭉치 비중만으로는 fertility 균형이 잡히지 않는다**는 점을 유념해야 한다. Petrov et al. 2023이 보였듯 언어의 타입론적 특성과 pre-tokenizer 설계가 말뭉치 비중보다 큰 영향을 미치는 경우가 많다[^petrov]. 실무에서 쓰이는 완화책으로는 언어별 부록 어휘집(language-specific tail), superword 토큰의 언어별 할당(K-EXAONE의 2:3:1 비율처럼), parallel tokenizer 같은 접근들이 있다.

### 평가 파이프라인

여러 후보를 비교할 때는 단계적으로 거른다.

1. **1차 스크리닝(intrinsic)**: 대상 언어·도메인 말뭉치에 대한 fertility, BPB, coverage(바이트 폴백 비율 등)를 측정한다. 모든 후보를 빠르게 훑는 단계다.
2. **2차 구조 검증**: 형태소 경계 보존율, 숫자·코드·이모지 같은 특수 도메인의 분절 품질, dead token 추정치.
3. **3차 proxy LM 검증**: 125M~1B 규모의 소형 LM을 각 후보 토크나이저로 학습시켜 다운스트림 몇 개를 재 본다. 비용이 크지만, 내재적 지표만으로 고를 때의 위험을 줄여 준다.
4. **최종 sanity check**: full-scale 학습 직전에 가역성과 특수 케이스 처리를 한 번 더 검증한다.

### 가역성과 정규화 검사

어떤 토크나이저를 선택하든 인코딩-디코딩 왕복이 **완전한 항등 함수**인지 반드시 검증해야 한다. 유니코드 정규화, 공백 처리, 특수 토큰 처리 등에서 미묘한 손실이 생기는 경우가 많고, 이는 실제 서비스에서 "사용자 입력이 살짝 바뀌어 출력된다"는 버그로 이어진다.

특히 **유니코드 정규화 방식**(NFC vs NFKC)의 선택은 도메인에 따라 결과가 크게 달라진다. NFKC는 호환성 기반 정규화라 위첨자/아래첨자/특수 기호를 평범한 형태로 "뭉개는" 반면, NFC는 이런 의미 구분을 보존한다. K-EXAONE이 STEM·코드 도메인 성능을 위해 NFKC에서 NFC로 전환한 것이 대표적 사례다. 한국어 단독으로 보면 한글 조합형/완성형 처리(NFD vs NFC) 역시 토큰화 결과에 영향을 준다.

## 마무리 — Low level에서 보이는 것들

언어 모델을 개발하다 보면 화려한 아키텍처나 거대한 학습 규모에 시선을 빼앗기기 쉽다. 하지만 모델이 세상을 읽는 가장 낮은 층위, 그 관문이 어떻게 생겼느냐가 결국 그 위에 쌓이는 모든 것을 제약한다. 동일한 LLM임에도 영어보다 한국어로 대화할 때 답답한 이유도, 숫자 계산에 유독 약한 모델이 있는 이유도, 동일한 아키텍처라도 언어에 따라 성능이 달라지는 이유도 결국 이 낮은 층위의 문제에서 출발한다.

토크나이저는 화려하지 않다. 논문 한 편으로 만들기 어렵고, 눈에 띄는 주목을 받지도 않는다. 하지만 바로 그 안에 **모델의 품질·효율·비용·안전성을 결정하는 수많은 변수**가 숨어 있다. 이 글이 그 안을 들여다보려는 누군가에게 작은 도움이 되었으면 한다.

다음 글에서는 내가 지금 작업 중인 한국어 토크나이저의 구체적인 설계 선택을 다뤄 보려 한다. pre-tokenizer 정규식을 어떻게 짰는지, 어휘집 크기를 얼마로 정했고 왜 그렇게 정했는지, 형태소 분석기 연동을 놓고 어떤 고민을 했는지, 평가 프로토콜을 어떻게 세웠는지 같은 이야기들이다. 이론만으로는 보이지 않는 부분이 거기에 있다.


<br /><br />
<br /><br />

-----

[^bpe]: Gage, P. *A New Algorithm for Data Compression*. The C Users Journal, 1994. 
[^exaone3]: LG AI Research. *EXAONE 3.0 7.8B Instruction Tuned Language Model*. arXiv:2408.03541, 2024. <https://arxiv.org/abs/2408.03541>
[^hcx]: Yoo, K. M. et al. (NAVER Cloud HyperCLOVA X Team). *HyperCLOVA X Technical Report*. arXiv:2404.01954, 2024. <https://arxiv.org/abs/2404.01954>
[^polyglot]: Ko, H. et al. *A Technical Report for Polyglot-Ko: Open-Source Large-Scale Korean Language Models*. arXiv:2306.02254, 2023. <https://arxiv.org/abs/2306.02254>
[^kogpt]: Kim, I., Han, G., Ham, J., & Baek, W. *KoGPT: KakaoBrain Korean(hangul) Generative Pre-trained Transformer*. 2021. <https://github.com/kakaobrain/kogpt> 
[^kexaone]: LG AI Research. *K-EXAONE Technical Report*. arXiv:2601.01739, 2026. <https://arxiv.org/abs/2601.01739>
[^superbpe]: Liu, A. et al. *SuperBPE: Space Travel for Language Models*. arXiv:2503.13423, 2025 (COLM 2025). <https://arxiv.org/abs/2503.13423>
[^arith]: Singh, A., & Strouse, D. *Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs*. arXiv:2402.14903, 2024. <https://arxiv.org/abs/2402.14903>
[^glitch]: Rumbelow, J., & Watkins, M. *SolidGoldMagikarp (plus, prompt generation)*. LessWrong, 2023. <https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation>
[^tekken]: Mistral AI. *Mistral NeMo*. 2024-07. <https://mistral.ai/news/mistral-nemo>
[^byt5]: Xue, L. et al. *ByT5: Towards a Token-free Future with Pre-trained Byte-to-Byte Models*. TACL, 2022 (arXiv:2105.13626). <https://arxiv.org/abs/2105.13626>
[^megabyte]: Yu, L. et al. *MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers*. NeurIPS 2023 (arXiv:2305.07185). <https://arxiv.org/abs/2305.07185>
[^mambabyte]: Wang, J. et al. *MambaByte: Token-free Selective State Space Model*. arXiv:2401.13660, 2024 (COLM 2024). <https://arxiv.org/abs/2401.13660>
[^blt]: Pagnoni, A. et al. *Byte Latent Transformer: Patches Scale Better Than Tokens*. arXiv:2412.09871, 2024. <https://arxiv.org/abs/2412.09871>
[^wegmann]: Wegmann, A., Nguyen, D., & Jurgens, D. *Tokenization is Sensitive to Language Variation*. Findings of ACL 2025 (arXiv:2502.15343). <https://arxiv.org/abs/2502.15343>
[^dagan]: Dagan, G., Synnaeve, G., & Rozière, B. *Getting the Most out of Your Tokenizer for Pre-training and Domain Adaptation*. ICML 2024. <https://arxiv.org/abs/2402.01035>
[^petrov]: Petrov, A., La Malfa, E., Torr, P. H. S., & Bibi, A. *Language Model Tokenizers Introduce Unfairness Between Languages*. NeurIPS 2023 (arXiv:2305.15425). <https://arxiv.org/abs/2305.15425>