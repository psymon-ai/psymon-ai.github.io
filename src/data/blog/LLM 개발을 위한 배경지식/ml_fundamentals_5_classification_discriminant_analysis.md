---
title: 5. 분류 II - 판별 분석과 베이즈 분류기
author: psymon
pubDatetime: 2026-04-10T09:44:11Z
modDatetime: 2026-04-10T09:44:11Z
slug: ml-fundamentals-5-classification-discriminant-analysis
featured: true
draft: false
tags:
  - Theory
  - Machine Learning
keywords:
  - 머신러닝
  - 분류
  - 베이즈 분류기
  - 판별 분석
  - QDA
  - LDA
  - 나이브 베이즈
  - 생성 모델
  - 혼동 행렬
  - ROC
  - AUC
description: 베이즈 정리로 분류 문제를 바라본다. 조건부 확률에서 출발해 QDA, LDA, 나이브 베이즈의 수식을 단계별로 풀고, 정확도만으로는 부족한 분류 평가 지표까지 정리한다.
---

## Table of contents

## 들어가는 글

지난 글에선 **로지스틱 회귀(Logistic Regression)**를 배웠다. 로지스틱 회귀는 입력 $x$가 주어졌을 때 정답이 1일 확률을 바로 계산했다.

$$ P(y=1 \mid x) $$

예를 들어 카드 지출액이 주어졌을 때, 채무 불이행(default)할 확률을 직접 예측하는 식이다. 이런 접근을 **판별 모델(discriminative model)**이라 부른다. 판별 모델은 클래스가 어떻게 생겼는지 전체 분포를 배우기보다 클래스 사이를 나누는 경계에 집중한다.

이번에는 다른 방식으로 분류를 바라본다.

> "각 클래스 데이터가 원래 어떤 모양으로 생겼는지 먼저 배운 다음, 새 데이터가 어느 클래스에서 나왔을 가능성이 큰지 판단하면 어떨까?"

이 방식은 먼저 클래스별 데이터 분포를 생각한다. 예를 들어 default가 난 고객들의 카드 지출액은 어떤 분포를 따르는지, default가 나지 않은 고객들의 카드 지출액은 어떤 분포를 따르는지 따로 본다. 그다음 새 고객의 카드 지출액이 어느 쪽 분포에서 더 자연스러운지 비교한다.

이런 접근을 **생성 모델(generative model)**이라 부른다. 클래스별로 데이터가 "생성되는 방식"을 모델링하기 때문이다.

이 관점은 분류 문제를 조금 다르게 본다. 로지스틱 회귀가 클래스 사이 경계를 직접 배우는 방법이라면, 생성 모델은 각 클래스가 차지하는 영역의 모양을 먼저 그린 뒤 새 데이터가 어느 쪽에서 나왔을 가능성이 큰지 판단한다.

참고로, 여기서 말하는 생성 모델은 ChatGPT 같은 LLM을 말하는 게 아니다. LLM에서 생성 모델은 보통 다음 토큰을 생성하는 모델을 뜻한다. 반면 이 글에서 말하는 생성 모델은 **분류를 위해 각 클래스의 데이터 분포를 모델링하는 통계적 접근**이다.

## 먼저 확률 표기부터

수식이 나오기 전에 확률 표기부터 정리하자. 이 글에서 가장 많이 볼 표기는 다음이다.

$$ P(y \mid x) $$

가운데 세로줄 $\mid$는 "~가 주어졌을 때"라고 읽는다. 따라서 $P(y \mid x)$는 **$x$를 알고 있을 때 $y$일 확률**이다.

예를 들어 $x$가 "카드 지출액이 2,000달러"라는 정보이고, $y$가 "default가 난다"라는 사건이라면 다음처럼 읽는다.

$$ P(y=\text{default} \mid x=\text{spending 2000}) $$

뜻은 "카드 지출액이 2,000달러라는 사실을 알고 있을 때, default가 날 확률"이다.

조건부 확률의 기본 공식은 다음과 같다.

$$ P(A \mid B) = \cfrac{P(A \cap B)}{P(B)} $$

말로 풀면 이렇다.

1. $B$가 일어난 경우만 모아 놓는다.
2. 그중에서 $A$도 같이 일어난 비율을 본다.

예를 들어 전체 100명 중 카드 지출액이 높은 사람이 20명이고, 그 20명 중 default가 난 사람이 8명이라면 다음과 같다.

$$ P(\text{default} \mid \text{high spending}) = \cfrac{8}{20} = 0.4 $$

조건부 확률은 "전체 중 몇 명?"이 아니라, **이미 어떤 조건을 만족한 사람들 중 몇 명?**을 묻는다. 베이즈 정리와 분류 모델은 대부분 이 표기법을 사용한다.

## 베이즈 분류기

분류 모델은 결국 결정을 내린다. 이메일을 스팸함으로 보낼지, 환자에게 추가 검사를 권할지, 카드 고객을 위험 고객으로 볼지 등의 문제를 다룬다.

결정에는 비용이 따른다. 정상 메일을 스팸으로 보내는 비용과 스팸 메일을 놓치는 비용은 다르다. 암 환자를 정상으로 판단하는 비용과 정상인을 암으로 오진하는 비용도 다르다.

이 비용을 수학적으로 표현한 것이 **손실 함수(loss function)**다.

$$ \mathcal{L}(y, f(x)) $$

$y$는 진짜 정답이고, $f(x)$는 모델의 예측이다. 손실 함수는 "정답이 $y$인데 모델이 $f(x)$라고 예측했을 때 얼마나 손해인가?"를 숫자로 나타낸다.

가장 단순한 손실은 **0-1 손실(0-1 loss)**이다.

$$
\mathcal{L}(y, \hat{y}) =
\begin{cases}
0 & \text{if } y = \hat{y} \\
1 & \text{if } y \ne \hat{y}
\end{cases}
$$

맞히면 0점, 틀리면 1점 벌점이라고 보면 된다.

이제 입력 $x$ 하나가 들어왔다고 하자. 이진 분류라서 정답은 $+1$ 또는 $-1$ 둘 중 하나다. 우리는 $+1$로 예측할지, $-1$로 예측할지 골라야 한다.

먼저 $+1$로 예측한다고 해보자.

- 실제 정답이 $+1$이면 맞혔으니 손실은 0이다.
- 실제 정답이 $-1$이면 틀렸으니 손실은 1이다.

따라서 $+1$로 예측했을 때 평균 손실은 다음과 같다.

$$ 0 \cdot P(y=+1 \mid x) + 1 \cdot P(y=-1 \mid x) $$

앞 항은 0이므로 사라진다.

$$ P(y=-1 \mid x) $$

즉, $+1$로 예측했을 때의 손실은 **사실은 $-1$일 확률**이다.

이번에는 $-1$로 예측한다고 해보자.

- 실제 정답이 $-1$이면 맞혔으니 손실은 0이다.
- 실제 정답이 $+1$이면 틀렸으니 손실은 1이다.

평균 손실은 다음과 같다.

$$ 1 \cdot P(y=+1 \mid x) + 0 \cdot P(y=-1 \mid x) = P(y=+1 \mid x) $$

따라서 결정 규칙은 아주 단순하다.

- $+1$일 확률이 더 크면 $+1$을 고른다.
- $-1$일 확률이 더 크면 $-1$을 고른다.

식으로 쓰면 다음과 같다.

$$
f^*(x) =
\begin{cases}
+1 & \text{if } P(y=+1 \mid x) > P(y=-1 \mid x) \\
-1 & \text{otherwise}
\end{cases}
$$

이렇게 **사후 확률이 가장 큰 클래스를 고르는 규칙**이 0-1 손실에서의 베이즈 분류기다. 사후 확률은 데이터를 본 뒤의 클래스 확률이라는 뜻이다.

![ml-5-1.png](@/assets/images/ml-5-1.png)
*베이즈 분류기는 평균 손실, 즉 리스크를 최소화하는 이상적인 분류 규칙이다.*

같은 규칙을 로그 비율로도 쓸 수 있다.

$$ f^*(x) = \operatorname{sign}\left(\log \cfrac{P(y=+1 \mid x)}{P(y=-1 \mid x)}\right) $$

이 식이 낯설게 보일 수 있으니 천천히 읽어보자.

분수 안쪽은 두 확률의 비율이다.

$$ \cfrac{P(y=+1 \mid x)}{P(y=-1 \mid x)} $$

분자가 더 크면 이 비율은 1보다 크다. 예를 들어 $0.8 / 0.2 = 4$다. 분모가 더 크면 비율은 1보다 작다. 예를 들어 $0.2 / 0.8 = 0.25$다.

로그 함수는 다음 성질을 가진다.

- $\log(1) = 0$
- $a > 1$이면 $\log(a) > 0$
- $0 < a < 1$이면 $\log(a) < 0$

따라서 로그 비율이 양수면 $+1$ 쪽 확률이 더 크고, 음수면 $-1$ 쪽 확률이 더 크다. 마지막의 $\operatorname{sign}$은 부호만 보는 함수다. 양수면 $+1$, 음수면 $-1$을 내놓는다.

복잡해 보이지만 결국 같은 말이다.

> 두 클래스의 확률을 비교해서 더 큰 쪽을 고른다.

이제 남은 문제는 하나다. $P(y \mid x)$를 어떻게 알 수 있을까?

로지스틱 회귀는 이 값을 직접 모델링했다. 이번 글의 모델들은 **베이즈 정리**로 우회한다.

## 베이즈 정리

베이즈 정리는 조건부 확률 공식에서 바로 나온다. 먼저 조건부 확률을 두 방향으로 써보자.

$$ P(y \mid x) = \cfrac{P(x \cap y)}{P(x)} $$

또 다른 방향으로 쓰면 다음과 같다.

$$ P(x \mid y) = \cfrac{P(x \cap y)}{P(y)} $$

두 번째 식에서 양변에 $P(y)$를 곱하면 다음이 된다.

$$ P(x \cap y) = P(x \mid y)P(y) $$

이 값을 첫 번째 식의 $P(x \cap y)$ 자리에 넣는다.

$$ P(y \mid x) = \cfrac{P(x \mid y)P(y)}{P(x)} $$

이것이 **베이즈 정리(Bayes' theorem)**다.

신용카드 default 예시로 각 항을 구체적으로 보자. 여기서는 다음처럼 두 사건을 정하자.

- $y$: 고객이 default를 낸다.
- $x$: 고객의 카드 지출액이 1,900~2,100달러 구간에 있다.

정확히 2,000달러처럼 한 점으로 잡으면 연속값에서는 확률을 다루기 애매하므로, 여기서는 작은 구간으로 생각하자.

전체 고객이 10,000명이고, 그중 default를 낸 고객이 333명이라고 하자. 또 카드 지출액이 1,900~2,100달러 구간인 고객이 500명이고, 그 500명 중 default를 낸 고객이 120명이라고 하자.

이때 각 항의 의미는 다음과 같다.

**$P(y \mid x)$: 사후 확률(Posterior)**
데이터 $x$를 본 뒤, 클래스가 $y$일 확률이다. 예시에서는 "카드 지출액이 1,900~2,100달러라는 사실을 알고 있을 때, 이 고객이 default를 낼 확률"이다. 위 숫자로는 해당 구간 고객 500명 중 120명이 default를 냈으므로 $120/500 = 0.24$다. 우리가 최종적으로 알고 싶은 값이다.

**$P(x \mid y)$: 가능도(Likelihood)**
클래스가 $y$라고 가정했을 때, 이런 데이터 $x$가 나올 확률이다. 예시에서는 "default를 낸 고객들 중 카드 지출액이 1,900~2,100달러 구간에 있을 확률"이다. default 고객 333명 중 120명이 이 구간에 있으므로 $120/333 \approx 0.36$이다.

**$P(y)$: 사전 확률(Prior)**
데이터를 보기 전에 클래스 $y$가 나올 기본 확률이다. 예시에서는 고객의 카드 지출액을 보기 전에, 아무 고객이나 한 명 뽑았을 때 default를 낼 확률이다. 전체 10,000명 중 333명이 default를 냈으므로 $333/10000 = 0.0333$이다.

**$P(x)$: 증거(Evidence)**
전체 데이터에서 $x$가 나타날 확률이다. 예시에서는 아무 고객이나 한 명 뽑았을 때 카드 지출액이 1,900~2,100달러 구간에 있을 확률이다. 전체 10,000명 중 500명이 이 구간에 있으므로 $500/10000 = 0.05$다.

베이즈 정리에 넣으면 다음처럼 사후 확률이 나온다.

$$
P(y \mid x)
= \cfrac{P(x \mid y)P(y)}{P(x)}
= \cfrac{0.36 \times 0.0333}{0.05}
\approx 0.24
$$

즉, 카드 지출액이 높은 구간에 있다는 정보를 보고 나면 default 확률을 3.33%에서 24%로 업데이트하게 된다. 이것이 베이즈 정리의 핵심이다. **관측한 데이터 $x$를 근거로, $y$일 확률을 갱신한다.**

하지만 분류에선 여기서 한 걸음 더 나아가야 한다. 고객이 default를 낼 가능성만 보는 것이 아니라, default를 내지 않을 가능성도 비교해야 한다. 이후 더 그럴듯한 쪽을 최종 예측으로 고른다.

위 예시를 이진 분류 표기로 쓰면 다음과 같다.

- $y=+1$: default
- $y=-1$: default 아님
- $x$: 카드 지출액이 1,900~2,100달러 구간에 있음

그러면 비교해야 할 두 값은 다음이다.

$$ P(y=+1 \mid x) = \cfrac{P(x \mid y=+1)P(y=+1)}{P(x)} $$

$$ P(y=-1 \mid x) = \cfrac{P(x \mid y=-1)P(y=-1)}{P(x)} $$

두 식의 분모는 똑같이 $P(x)$다. 같은 $x$를 두고 클래스를 비교하기 때문이다. 분모가 같으면 대소 비교에는 영향을 주지 않는다. 따라서 분자만 비교한다.

$$ P(x \mid y=+1)P(y=+1) \quad \text{vs.} \quad P(x \mid y=-1)P(y=-1) $$

이를 짧게 쓰면 다음과 같다.

$$ P(y \mid x) \propto P(x \mid y)P(y) $$

$\propto$는 "비례한다"는 뜻이다. 정확한 확률값을 구하려면 나중에 전체 합이 1이 되도록 나눠줘야 하지만, 어느 클래스가 더 큰지 고르는 건 분자만 비교해도 된다.

![ml-5-5.png](@/assets/images/ml-5-5.png)
*베이즈 정리는 직접 알기 어려운 사후 확률을 가능도와 사전 확률로 바꾼다.*

이제 같은 생각을 간단한 코드로 옮겨보자. 이번에는 신용카드가 아니라 메일 분류 예시다. 어떤 메일에 `free`와 `meeting`이라는 단어가 들어 있다. 스팸 메일과 정상 메일에서 각 단어가 나올 확률을 이미 추정해 두었다고 하자.

```python
import math

priors = {"spam": 0.3, "normal": 0.7}
likelihoods = {
    "spam": {"free": 0.80, "meeting": 0.10},
    "normal": {"free": 0.05, "meeting": 0.60},
}
tokens = ["free", "meeting"]

scores = {}
for label in priors:
    score = math.log(priors[label])
    for token in tokens:
        score += math.log(likelihoods[label][token])
    scores[label] = score

best = max(scores, key=scores.get)

for label, score in scores.items():
    print(f"{label:6s}: log-score={score:.3f}")
print("prediction:", best)
```
```bash
spam  : log-score=-3.730
normal: log-score=-3.863
prediction: spam
```

여기서 로그를 쓴 이유는 계산 안정성 때문이다. 확률은 보통 0과 1 사이의 작은 수다. 컴퓨터 계산에서는 작은 수를 여러 번 곱하면 값이 0으로 사라질 수 있다. 그래서 곱셈을 덧셈으로 바꾸기 위해 로그를 취한다.

$$ \log(ab) = \log a + \log b $$

대소 관계는 그대로 보존된다. 로그는 단조 증가 함수이기 때문이다. 즉, $a > b$이면 $\log a > \log b$다. 그래서 확률 곱을 직접 비교하지 않고 로그 점수를 비교해도 같은 결론이 나온다.

## 생성 모델과 판별 모델

이제 두 분류 접근을 분명히 나눌 수 있다.

**판별 모델(discriminative model)**은 $P(y \mid x)$를 직접 학습한다. 입력이 주어졌을 때 클래스가 무엇인지 바로 맞히려 한다. 로지스틱 회귀가 여기에 속한다.

**생성 모델(generative model)**은 $P(y)$와 $P(x \mid y)$를 먼저 학습한다. 클래스별 데이터 분포를 배운 뒤, 베이즈 정리로 $P(y \mid x)$를 계산한다.

![ml-5-6.png](@/assets/images/ml-5-6.png)
*판별 모델은 결정 경계를 직접 배우고, 생성 모델은 클래스별 데이터 분포를 먼저 배운다.*

생성 모델의 핵심 질문은 이것이다.

> 클래스가 정해졌을 때, 데이터 $x$는 어떤 분포에서 나왔다고 볼 것인가?

가장 먼저 볼 답은 **정규 분포(Gaussian distribution)**다.

## 정규 분포로 클래스 모양 표현하기

정규 분포는 종 모양의 분포다. 평균 근처에 데이터가 많이 모이고, 평균에서 멀어질수록 데이터가 적어진다.

1차원 정규 분포의 확률 밀도 함수는 다음과 같다.

$$
\mathcal{N}(x;\mu,\sigma^2)
= \cfrac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(-\cfrac{(x-\mu)^2}{2\sigma^2}\right)
$$

기호가 많지만 각 부분의 역할은 어렵지 않다.

**$\mu$**는 평균이다. 분포의 중심 위치를 정한다.

**$\sigma^2$**는 분산이다. 데이터가 얼마나 넓게 퍼지는지 정한다. 분산이 작으면 좁고 뾰족한 분포가 되고, 분산이 크면 넓게 퍼진 분포가 된다.

**$(x-\mu)^2$**는 $x$가 평균에서 얼마나 멀리 떨어져 있는지 나타낸다. 제곱이므로 왼쪽으로 멀어져도, 오른쪽으로 멀어져도 값이 커진다.

여기까지는 $x$가 숫자 하나인 경우다. 그런데 현실 데이터는 보통 피처가 여러 개다. 예를 들어 고객 한 명을 설명할 때 카드 지출액만 보는 것이 아니라, 소득, 나이, 연체 이력 같은 값을 함께 본다.

이때 각 피처가 얼마나 퍼져 있는지는 여전히 **분산(variance)**으로 볼 수 있다. 하지만 피처가 여러 개면 한 가지를 더 봐야 한다. 두 피처가 **어떻게 함께 움직이는가**다. 이걸 나타내는 값이 **공분산(covariance)**이다.

예를 들어 두 피처를 이렇게 두자.

- $x_1$: 카드 지출액
- $x_2$: 연체 일수

카드 지출액이 평균보다 큰 고객들이 연체 일수도 평균보다 큰 경향이 있다면, 두 값은 같은 방향으로 움직인다. 이때 공분산은 양수가 된다.

반대로 카드 지출액이 평균보다 큰 고객일수록 연체 일수는 평균보다 작은 경향이 있다면, 두 값은 반대 방향으로 움직인다. 이때 공분산은 음수가 된다.

둘 사이에 뚜렷한 관계가 없다면 공분산은 0에 가까워진다.

공분산 공식은 다음처럼 생겼다.

$$
\operatorname{Cov}(x_1, x_2)
= \cfrac{1}{n}\sum_{i=1}^{n}(x_{i1}-\mu_1)(x_{i2}-\mu_2)
$$

공식 안의 곱셈이 핵심이다. 어떤 고객이 두 피처 모두 평균보다 크면 두 괄호가 모두 양수라 곱이 양수다. 두 피처 모두 평균보다 작아도 두 괄호가 모두 음수라 곱은 다시 양수다. 즉, 두 값이 같은 방향으로 움직이면 양수가 쌓인다. 한쪽은 평균보다 크고 다른 쪽은 평균보다 작으면 곱이 음수가 된다.

피처가 여러 개일 때는 이런 분산과 공분산을 표처럼 모아 둔다. 이것을 **공분산 행렬(covariance matrix)**이라고 부른다. 대각선에는 각 피처의 분산이 들어가고, 대각선 바깥에는 피처끼리의 공분산이 들어간다.

$$
\Sigma =
\begin{bmatrix}
\operatorname{Var}(x_1) & \operatorname{Cov}(x_1, x_2) \\
\operatorname{Cov}(x_2, x_1) & \operatorname{Var}(x_2)
\end{bmatrix}
$$

여기서 곧바로 행렬 수식으로 들어가면 설명이 너무 무거워진다. 그래서 먼저 1차원, 즉 피처가 하나뿐인 경우부터 보자. 1차원에서는 공분산 행렬까지 갈 필요가 없고, 분산 $\sigma^2$ 하나만 있으면 된다.

판별 분석은 먼저 클래스별로 이런 1차원 정규 분포가 있다고 가정한다.

$$ x \mid y=k \sim \mathcal{N}(\mu_k, \sigma_k^2) $$

"클래스가 $k$라면, $x$는 평균 $\mu_k$, 분산 $\sigma_k^2$인 정규 분포에서 나온다"는 뜻이다.

이제 이 단순한 1차원 식으로 QDA와 LDA의 핵심 차이를 먼저 이해하자. 다차원에서는 뒤에서 $\sigma^2$ 자리에 공분산 행렬 $\Sigma$가 들어간다고 확장하면 된다.

![ml-5-8.png](@/assets/images/ml-5-8.png)
*각 클래스의 평균과 분산을 데이터에서 추정한 뒤, 새 데이터가 어느 분포에서 더 자연스러운지 비교한다.*

이 1차원 가정에서 QDA와 LDA가 나온다. 둘의 차이는 분산을 어떻게 보느냐에 있다.

- QDA: 클래스마다 분산이 다를 수 있다.
- LDA: 모든 클래스가 같은 분산을 가진다고 본다.

먼저 QDA부터 보자.

## QDA - 분산이 다르면 이차식이 남는다

**QDA(Quadratic Discriminant Analysis)**는 클래스마다 다른 분산을 허용한다.

이진 분류에서 두 클래스가 있다고 하자.

$$ y \in \{+1, -1\} $$

각 클래스의 분포는 다음처럼 둔다.

$$ x \mid y=+1 \sim \mathcal{N}(\mu_1, \sigma_1^2) $$

$$ x \mid y=-1 \sim \mathcal{N}(\mu_{-1}, \sigma_{-1}^2) $$

베이즈 분류기는 두 사후 확률을 비교한다.

$$ P(y=+1 \mid x) \quad \text{vs.} \quad P(y=-1 \mid x) $$

베이즈 정리를 쓰면 분모 $P(x)$는 양쪽에서 같으므로, 다음 둘만 비교하면 된다.

$$ P(x \mid y=+1)P(y=+1) $$

$$ P(x \mid y=-1)P(y=-1) $$

클래스 $+1$의 사전 확률을 $\alpha$라고 두자. 그러면 클래스 $-1$의 사전 확률은 $1-\alpha$다.

$$ P(y=+1)=\alpha $$

$$ P(y=-1)=1-\alpha $$

우리는 다음 로그 비율의 부호를 보면 된다.

$$
\log
\cfrac{
\mathcal{N}(x;\mu_1,\sigma_1^2)\alpha
}{
\mathcal{N}(x;\mu_{-1},\sigma_{-1}^2)(1-\alpha)
}
$$

이 식을 천천히 풀어보자. 사용할 로그 성질은 세 가지다.

$$ \log \cfrac{a}{b} = \log a - \log b $$

$$ \log(ab) = \log a + \log b $$

$$ \log(e^u) = u $$

먼저 분수의 로그를 빼기로 바꾼다.

$$
\log \mathcal{N}(x;\mu_1,\sigma_1^2)
+ \log \alpha
- \log \mathcal{N}(x;\mu_{-1},\sigma_{-1}^2)
- \log(1-\alpha)
$$

이제 정규 분포의 로그를 계산한다.

$$
\log \mathcal{N}(x;\mu,\sigma^2)
=
\log \cfrac{1}{\sqrt{2\pi\sigma^2}}
+ \log \exp\left(-\cfrac{(x-\mu)^2}{2\sigma^2}\right)
$$

두 번째 항은 $\log(e^u)=u$이므로 바로 내려온다.

$$
\log \mathcal{N}(x;\mu,\sigma^2)
=
-\cfrac{1}{2}\log(2\pi\sigma^2)
-\cfrac{(x-\mu)^2}{2\sigma^2}
$$

이 값을 양쪽 클래스에 대입하면 다음 모양이 된다.

$$
-\cfrac{1}{2}\log\cfrac{\sigma_1^2}{\sigma_{-1}^2}
-\cfrac{(x-\mu_1)^2}{2\sigma_1^2}
+\cfrac{(x-\mu_{-1})^2}{2\sigma_{-1}^2}
+\log\cfrac{\alpha}{1-\alpha}
$$

$2\pi$가 사라진 이유는 양쪽에 똑같이 들어 있어서 빼면 0이 되기 때문이다.

여기서 핵심은 제곱 항이다.

$$ (x-\mu_1)^2, \quad (x-\mu_{-1})^2 $$

제곱식을 전개하면 다음과 같다.

$$ (x-\mu)^2 = x^2 - 2\mu x + \mu^2 $$

QDA에서는 분산 $\sigma_1^2$와 $\sigma_{-1}^2$가 다를 수 있다. 그러면 $x^2$ 앞에 붙는 계수가 서로 달라진다. 따라서 두 식을 빼도 $x^2$ 항이 완전히 사라지지 않는다.

결정 경계에 $x^2$ 항이 남으므로 경계는 **이차식(quadratic)**이 된다. 그래서 이름이 QDA다.

![ml-5-13.png](@/assets/images/ml-5-13.png)
*QDA는 클래스마다 다른 분산과 공분산을 허용하므로 곡선 형태의 결정 경계를 만들 수 있다.*

간단한 코드로 QDA와 LDA의 차이를 살펴보자. QDA는 클래스별 분산을 그대로 쓰고, LDA는 공통 분산을 쓴다. LDA의 수식은 다음 절에서 자세히 전개한다.

```python
import math

def mean(values):
    return sum(values) / len(values)

def variance(values, mu):
    return sum((x - mu) ** 2 for x in values) / len(values)

def log_gaussian(x, mu, var):
    return -0.5 * math.log(2 * math.pi * var) - ((x - mu) ** 2) / (2 * var)

groups = {
    "safe": [720, 810, 890, 940, 1010, 1080],
    "risk": [1320, 1500, 1710, 1970, 2310, 2760],
}

stats = {}
total = sum(len(v) for v in groups.values())
for label, values in groups.items():
    mu = mean(values)
    var = variance(values, mu)
    prior = len(values) / total
    stats[label] = {"mu": mu, "var": var, "prior": prior}

pooled_var = sum(len(v) * stats[k]["var"] for k, v in groups.items()) / total

def predict_qda(x):
    scores = {}
    for label, s in stats.items():
        scores[label] = math.log(s["prior"]) + log_gaussian(x, s["mu"], s["var"])
    return max(scores, key=scores.get)

def predict_lda(x):
    scores = {}
    for label, s in stats.items():
        scores[label] = math.log(s["prior"]) + log_gaussian(x, s["mu"], pooled_var)
    return max(scores, key=scores.get)

for label, s in stats.items():
    print(f"{label}: mean={s['mu']:.1f}, var={s['var']:.1f}")
print(f"pooled var={pooled_var:.1f}")

for x in [1000, 1500, 2200]:
    print(f"spending={x}: QDA={predict_qda(x)}, LDA={predict_lda(x)}")
```
```bash
safe: mean=908.3, var=14380.6
risk: mean=1928.3, var=240047.2
pooled var=127213.9
spending=1000: QDA=safe, LDA=safe
spending=1500: QDA=risk, LDA=risk
spending=2200: QDA=risk, LDA=risk
```

QDA는 유연하다. 클래스마다 퍼지는 모양이 달라도 받아들인다. 대신 추정해야 할 값이 많다. 데이터가 적으면 평균과 분산을 잘못 추정하기 쉽고, 모델이 흔들릴 수 있다.

## LDA - 분산이 같으면 직선이 된다

**LDA(Linear Discriminant Analysis)**는 QDA에서 한 가지를 더 단순하게 본다.

> 두 클래스의 분산이 같다.

1차원에서는 다음처럼 쓴다.

$$ \sigma_1^2 = \sigma_{-1}^2 = \sigma^2 $$

QDA에서 봤던 로그 비율 식을 다시 가져오자.

$$
-\cfrac{1}{2}\log\cfrac{\sigma_1^2}{\sigma_{-1}^2}
-\cfrac{(x-\mu_1)^2}{2\sigma_1^2}
+\cfrac{(x-\mu_{-1})^2}{2\sigma_{-1}^2}
+\log\cfrac{\alpha}{1-\alpha}
$$

이제 두 분산 $\sigma_1^2$와 $\sigma_{-1}^2$가 같다고 하자.

첫 번째 항은 이렇게 된다.

$$ -\cfrac{1}{2}\log\cfrac{\sigma^2}{\sigma^2} = -\cfrac{1}{2}\log 1 = 0 $$

분산 비율이 1이므로 사라진다.

남은 제곱 항을 보자.

$$
-\cfrac{(x-\mu_1)^2}{2\sigma^2}
+\cfrac{(x-\mu_{-1})^2}{2\sigma^2}
$$

분모가 같으니 하나로 묶을 수 있다.

$$
\cfrac{(x-\mu_{-1})^2 - (x-\mu_1)^2}{2\sigma^2}
$$

이제 제곱식을 전개한다.

$$ (x-\mu_{-1})^2 = x^2 - 2\mu_{-1}x + \mu_{-1}^2 $$

$$ (x-\mu_1)^2 = x^2 - 2\mu_1x + \mu_1^2 $$

두 식을 빼면 다음과 같다.

$$
(x-\mu_{-1})^2 - (x-\mu_1)^2
$$

$$
= (x^2 - 2\mu_{-1}x + \mu_{-1}^2)
- (x^2 - 2\mu_1x + \mu_1^2)
$$

괄호를 풀면 $x^2$ 항과 $-x^2$ 항이 만난다.

$$
= x^2 - 2\mu_{-1}x + \mu_{-1}^2
- x^2 + 2\mu_1x - \mu_1^2
$$

$x^2$ 항이 사라진다.

$$
= 2(\mu_1-\mu_{-1})x + \mu_{-1}^2 - \mu_1^2
$$

이제 전체 식은 $x$에 대한 일차식이 된다.

$$
\cfrac{\mu_1-\mu_{-1}}{\sigma^2}x
- \cfrac{\mu_1^2-\mu_{-1}^2}{2\sigma^2}
+ \log\cfrac{\alpha}{1-\alpha}
$$

복잡해 보이지만 중요한 결론은 하나다.

> 분산이 같다고 가정하면 $x^2$ 항이 상쇄되어 사라진다.

그래서 결정 경계가 직선이 된다. 이 때문에 이름이 **Linear Discriminant Analysis**다.

QDA와 LDA의 차이는 결국 여기서 갈린다.

- QDA: 분산이 달라도 된다. $x^2$ 항이 남는다. 경계가 곡선이 될 수 있다.
- LDA: 분산이 같다고 본다. $x^2$ 항이 사라진다. 경계가 직선이 된다.

LDA는 QDA보다 덜 유연하다. 대신 추정해야 할 값이 적으므로 데이터가 적을 때 더 안정적이다. 머신러닝에서 자주 만나는 트레이드오프다. 모델이 유연할수록 더 많은 데이터가 필요하고, 모델이 단순할수록 적은 데이터로도 버틸 수 있다.

## 다차원으로 확장하기

지금까지 $x$를 숫자가 하나인 1차원으로 봤다. 현실에선 피처가 여러 개다. 예를 들어 카드 고객을 분류한다면 카드 지출액, 소득, 나이, 결제 이력 같은 값이 함께 들어갈 수 있다.

피처가 여러 개이면 $x$는 벡터가 된다.

$$ x = (x_1, x_2, \cdots, x_p) $$

1차원에서 평균은 숫자 하나였지만, 다차원에서는 평균도 벡터가 된다.

$$ \mu_k = (\mu_{k1}, \mu_{k2}, \cdots, \mu_{kp}) $$

분산은 공분산 행렬로 바뀐다.

$$ \Sigma_k $$

공분산 행렬은 피처들이 각각 얼마나 퍼지는지뿐 아니라, 피처들이 서로 어떻게 함께 움직이는지도 담는다. 예를 들어 카드 지출액이 높을수록 연체 일수도 늘어나는 경향이 있다면 두 피처는 함께 움직인다. 이런 관계를 공분산이 표현한다.

다차원 QDA의 점수는 다음과 같다.

$$
\delta_k(x)
= -\cfrac{1}{2}\log|\Sigma_k|
- \cfrac{1}{2}(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)
+ \log \pi_k
$$

행렬 기호가 낯설면, 지금은 다음 정도로 이해해도 충분하다.

**$\mu_k$**는 클래스 $k$의 중심이다.

**$\Sigma_k$**는 클래스 $k$가 어떤 모양으로 퍼져 있는지 나타낸다.

**$(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)$**는 $x$가 클래스 중심에서 얼마나 멀리 있는지를 공분산까지 고려해 계산한 거리다.

LDA는 모든 클래스가 같은 공분산 $\Sigma$를 쓴다. 그러면 1차원에서 $x^2$ 항이 사라졌듯이, 다차원에서도 이차 항이 클래스 비교에서 사라진다. 남는 점수는 $x$에 대한 선형식이다.

$$
\delta_k(x)
= x^T\Sigma^{-1}\mu_k
- \cfrac{1}{2}\mu_k^T\Sigma^{-1}\mu_k
+ \log \pi_k
$$

![ml-5-14.png](@/assets/images/ml-5-14.png)
*다차원에서도 원리는 같다. 평균 벡터와 공분산 행렬을 추정해 클래스별 점수를 계산한다.*

## 클래스가 3개 이상이면

클래스가 3개 이상이면 어떻게 될까? 원리는 크게 다르지 않다. 각 클래스마다 점수 $\delta_k(x)$를 계산하고, 가장 큰 점수를 가진 클래스를 고르면 된다.

$$ \hat{y} = \underset{k}{\operatorname{argmax}} \, \delta_k(x) $$

$\operatorname{argmax}$는 "값을 가장 크게 만드는 $k$를 고르라"는 뜻이다.

점수를 확률처럼 바꾸고 싶으면 소프트맥스를 쓴다.

$$ P(y=k \mid x) = \cfrac{\exp(\delta_k(x))}{\sum_j \exp(\delta_j(x))} $$

이 식은 지난 글에서 본 소프트맥스와 같은 모양이다. 차이는 점수 $\delta_k(x)$가 어디서 왔느냐다. 로지스틱 회귀에서는 점수를 직접 학습했고, LDA/QDA에서는 정규 분포 가정에서 점수가 나왔다.

![ml-5-17.png](@/assets/images/ml-5-17.png)
*클래스가 세 개 이상이어도 각 클래스의 점수를 계산하고 가장 큰 값을 고르는 구조는 같다.*

## LDA와 로지스틱 회귀의 관계

LDA와 로지스틱 회귀는 출발점이 다르다.

로지스틱 회귀는 $P(y \mid x)$를 직접 모델링한다. 입력 $x$가 주어졌을 때, 클래스가 무엇일지 바로 계산한다.

반면 LDA는 $P(x \mid y)$와 $P(y)$를 먼저 모델링한다. 즉, 각 클래스의 데이터가 어떤 정규 분포에서 나오는지 먼저 가정하고, 베이즈 정리로 $P(y \mid x)$를 구한다.

출발점은 다르지만, 이진 분류에서는 두 모델의 식이 같은 모양으로 정리된다. 왜 그런지 보자.

앞에서 LDA의 클래스별 점수를 다음처럼 썼다.

$$
\delta_k(x)
= x^T\Sigma^{-1}\mu_k
- \cfrac{1}{2}\mu_k^T\Sigma^{-1}\mu_k
+ \log \pi_k
$$

이 점수는 "입력 $x$가 클래스 $k$에 속할 그럴듯함"을 나타내는 값이다. 이진 분류라면 클래스가 두 개뿐이므로, 클래스 1의 점수와 클래스 0의 점수를 비교하면 된다.

$$ \delta_1(x) \quad \text{vs.} \quad \delta_0(x) $$

둘 중 어느 쪽이 큰지만 보면 되므로, 두 점수의 차이를 보자.

$$
\delta_1(x)-\delta_0(x)
$$

식을 그대로 빼면 다음과 같다.

$$
\delta_1(x)-\delta_0(x)
= x^T\Sigma^{-1}(\mu_1-\mu_0)
- \cfrac{1}{2}\left(\mu_1^T\Sigma^{-1}\mu_1-\mu_0^T\Sigma^{-1}\mu_0\right)
+ \log \cfrac{\pi_1}{\pi_0}
$$

길어 보이지만 $x$가 들어 있는 부분은 첫 항뿐이다.

$$ x^T\Sigma^{-1}(\mu_1-\mu_0) $$

나머지 두 항은 $\mu_0$, $\mu_1$, $\Sigma$, $\pi_0$, $\pi_1$로만 이루어져 있다. 학습이 끝나면 모두 고정된 숫자다. 따라서 상수항으로 묶을 수 있다.

그래서 전체를 이렇게 쓸 수 있다.

$$
\delta_1(x)-\delta_0(x) = \beta_0 + \beta^Tx
$$

여기서 $\beta^Tx$는 입력 $x$에 대한 선형식이고, $\beta_0$는 절편이다.

이 점수 차이는 두 클래스의 사후 확률 비율, 즉 로그 오즈와 같은 역할을 한다. 그래서 다음과 같은 형태가 나온다.

$$ \log \cfrac{P(y=1 \mid x)}{P(y=0 \mid x)} = \beta_0 + \beta^Tx $$

지난 글에서 본 로지스틱 회귀의 로짓과 같은 모양이다. 로지스틱 회귀도 "로그 오즈가 입력 $x$에 대해 선형"이라고 가정했다. LDA는 정규 분포와 공통 공분산 가정에서 출발했는데, 정리하고 보니 같은 선형 로짓 형태에 도착한 것이다.

그래서 LDA와 로지스틱 회귀는 비슷한 선형 결정 경계를 만들 수 있다.

하지만 두 모델이 같은 것은 아니다.

**로지스틱 회귀**는 클래스별 데이터가 정규 분포인지 관심이 없다. 분류 경계만 잘 배우면 된다.

**LDA**는 클래스별 데이터가 정규 분포이고, 공분산이 같다는 가정을 둔다. 이 가정이 맞으면 데이터가 적을 때도 안정적일 수 있다. 가정이 크게 틀리면 성능이 떨어질 수 있다.

![ml-5-19.png](@/assets/images/ml-5-19.png)
*LDA와 로지스틱 회귀는 비슷한 선형 로짓 형태를 만들 수 있지만, 그 형태에 도달하는 방식이 다르다.*

## 나이브 베이즈

이제 다른 생성 모델을 보자. **나이브 베이즈(Naive Bayes)**다.

QDA와 LDA는 피처가 함께 어떻게 움직이는지를 공분산으로 다룬다. 피처가 많아지면 이 공분산을 추정하기 어려워진다.

나이브 베이즈는 여기서 아주 과감한 가정을 한다.

> 클래스가 주어졌다면, 피처들은 서로 독립이다.

이 문장을 천천히 읽어보자.

"피처들이 서로 독립"이라는 말은 한 피처를 알아도 다른 피처에 대한 정보가 늘어나지 않는다는 뜻이다. 예를 들어 이메일 분류에서 `무료`라는 단어가 나온 것과 `증정`이라는 단어가 나온 것이 서로 독립이라고 보는 식이다.

현실에선 보통 틀린 가정이다. `무료`가 나온 메일에는 `증정`이나 `가입` 같은 단어도 같이 나올 가능성이 높다. 그래서 "나이브", 즉 현실의 관계를 너무 안일하게 단순화한다는 이름이 붙었다.

수식으로는 다음과 같다.

$$ P(x \mid y) = P(x_1, x_2, \cdots, x_d \mid y) $$

원래는 피처 전체가 함께 나올 확률을 알아야 한다. 하지만 조건부 독립을 가정하면 곱으로 나눌 수 있다.

$$ P(x \mid y) = \prod_{j=1}^{d} P(x_j \mid y) $$

$\prod$는 곱셈 기호다. $\sum$이 여러 값을 더하라는 뜻이라면, $\prod$는 여러 값을 곱하라는 뜻이다.

피처가 세 개라면 다음과 같은 말이다.

$$ P(x_1, x_2, x_3 \mid y) = P(x_1 \mid y)P(x_2 \mid y)P(x_3 \mid y) $$

복잡한 전체 확률을 피처별 확률의 곱으로 바꿨다. 이 덕분에 계산이 매우 쉬워진다.

![ml-5-23.png](@/assets/images/ml-5-23.png)
*나이브 베이즈는 조건부 독립 가정 덕분에 고차원 가능도를 피처별 가능도의 곱으로 단순화한다.*

흥미로운 점은 틀린 가정에도 불구하고 나이브 베이즈가 꽤 잘 작동한다는 것이다. 분류에선 확률값이 완벽히 정확할 필요는 없다. 어느 클래스 점수가 더 큰지만 잘 맞으면 된다. 특히 텍스트 분류에서는 단어별 신호가 강해서, 이 안일한 가정만으로도 꽤 좋은 베이스라인이 된다.

![ml-5-24.png](@/assets/images/ml-5-24.png)
*피처 독립 가정은 대개 틀리다. 그래도 분류 경계의 대소 관계만 잘 맞으면 실전에서는 유용할 수 있다.*

나이브 베이즈의 장점은 명확하다.

- 학습이 매우 빠르다.
- 데이터가 적어도 동작한다.
- 피처가 많아도 계산이 단순하다.
- 스팸 필터, 문서 분류, 감성 분석의 강력한 베이스라인이 된다.

## Play Golf 예시

카테고리형 피처에서 나이브 베이즈가 어떻게 작동하는지 작은 예제로 보자. 14일 동안 날씨와 골프 여부를 기록한 데이터가 있다.

내일 날씨가 다음과 같을 때 골프를 칠지 예측해보자.

$$ x = (\text{Sunny}, \text{Hot}, \text{High}, \text{Weak}) $$

나이브 베이즈는 `Yes`와 `No`에 대해 각각 다음 값을 계산한다.

$$
P(\text{Yes})
P(\text{Sunny}\mid\text{Yes})
P(\text{Hot}\mid\text{Yes})
P(\text{High}\mid\text{Yes})
P(\text{Weak}\mid\text{Yes})
$$

$$
P(\text{No})
P(\text{Sunny}\mid\text{No})
P(\text{Hot}\mid\text{No})
P(\text{High}\mid\text{No})
P(\text{Weak}\mid\text{No})
$$

각 항은 어렵지 않다. 예를 들어 $P(\text{Sunny}\mid\text{Yes})$는 "골프를 친 날들 중 날씨가 Sunny였던 비율"이다.

아래 코드는 같은 계산을 라플라스 스무딩과 로그 확률로 구현한다. 라플라스 스무딩은 바로 다음 절에서 설명한다. 지금은 "0이 되는 확률을 막기 위해 작은 가상 카운트를 더한다" 정도로만 보면 된다.

```python
import math

data = [
    ("Sunny", "Hot", "High", "Weak", "No"),
    ("Sunny", "Hot", "High", "Strong", "No"),
    ("Overcast", "Hot", "High", "Weak", "Yes"),
    ("Rain", "Mild", "High", "Weak", "Yes"),
    ("Rain", "Cool", "Normal", "Weak", "Yes"),
    ("Rain", "Cool", "Normal", "Strong", "No"),
    ("Overcast", "Cool", "Normal", "Strong", "Yes"),
    ("Sunny", "Mild", "High", "Weak", "No"),
    ("Sunny", "Cool", "Normal", "Weak", "Yes"),
    ("Rain", "Mild", "Normal", "Weak", "Yes"),
    ("Sunny", "Mild", "Normal", "Strong", "Yes"),
    ("Overcast", "Mild", "High", "Strong", "Yes"),
    ("Overcast", "Hot", "Normal", "Weak", "Yes"),
    ("Rain", "Mild", "High", "Strong", "No"),
]

features = ["Sunny", "Hot", "High", "Weak"]
labels = sorted({row[-1] for row in data})
values_by_col = [sorted({row[i] for row in data}) for i in range(4)]
alpha = 1

log_scores = {}
for label in labels:
    rows = [row for row in data if row[-1] == label]
    score = math.log(len(rows) / len(data))

    for i, value in enumerate(features):
        count = sum(1 for row in rows if row[i] == value)
        prob = (count + alpha) / (len(rows) + alpha * len(values_by_col[i]))
        score += math.log(prob)

    log_scores[label] = score

max_score = max(log_scores.values())
weights = {k: math.exp(v - max_score) for k, v in log_scores.items()}
normalizer = sum(weights.values())
posteriors = {k: weights[k] / normalizer for k in weights}

for label in labels:
    print(f"{label:3s}: log-score={log_scores[label]:.3f}, posterior={posteriors[label]:.3f}")
print("prediction:", max(posteriors, key=posteriors.get))
```
```bash
No : log-score=-3.887, posterior=0.688
Yes: log-score=-4.678, posterior=0.312
prediction: No
```

이 예시에서는 `No`의 점수가 더 크다. 따라서 해당 조건에서는 골프를 치지 않는 쪽으로 예측한다.


## 나이브 베이즈의 두 가지 실전 문제

나이브 베이즈를 구현할 때 반드시 챙겨야 할 문제가 두 가지 있다.

첫째, **zero-frequency 문제**다. 학습 데이터에서 어떤 클래스와 피처 값의 조합이 한 번도 등장하지 않으면 확률이 0이 된다.

예를 들어 `Yes`인 날 중 `Snow`가 한 번도 없었다면 다음 값은 0이다.

$$ P(\text{Snow} \mid \text{Yes}) = 0 $$

나이브 베이즈는 확률을 곱한다. 곱셈에선 하나라도 0이면 전체가 0이 된다.

$$ 0.7 \times 0.5 \times 0 = 0 $$

이러면 나머지 피처가 아무리 강한 신호를 줘도 전체 점수가 0이 되어 버린다.

해결책은 **라플라스 스무딩(Laplace smoothing)**이다.

$$ P(x_j=v \mid y=k) = \cfrac{\operatorname{count}(x_j=v, y=k) + \alpha}{\operatorname{count}(y=k) + \alpha |V_j|} $$

식이 길지만 하는 일은 단순하다.

분자에는 해당 값이 나온 횟수를 넣는다. 여기에 $\alpha$를 더한다. 보통 $\alpha=1$을 쓴다. 그러면 한 번도 안 나온 값도 최소 1번 나온 것처럼 처리된다.

분모에는 클래스 $k$에 속한 데이터 수를 넣는다. 그리고 가능한 값의 개수 $|V_j|$만큼 $\alpha$를 더해 전체 확률 합이 1이 되도록 맞춘다.

둘째, **수치 언더플로우(numerical underflow)**다. 작은 확률을 많이 곱하면 컴퓨터가 너무 작은 값을 표현하지 못해 0으로 처리할 수 있다.

해결책은 로그 확률이다.

$$ \log P(y \mid x) = \log P(y) + \sum_{j=1}^{d}\log P(x_j \mid y) + C $$

$C$는 모든 클래스에 공통으로 들어가는 상수라 비교할 때는 신경 쓰지 않아도 된다. 그래서 보통 `log_prior + log_likelihood`만 비교한다.

![ml-5-27.png](@/assets/images/ml-5-27.png)
*스무딩과 로그 확률은 나이브 베이즈를 실제로 쓸 때 빠뜨리면 안 되는 기본 장치다.*

## 분류 모델 평가

모델을 만들었으면 평가해야 한다. 여기서 가장 흔한 함정은 **정확도(accuracy)**만 보는 것이다.

예를 들어 희귀병 진단 모델을 만든다고 하자. 전체 인구의 1%만 병에 걸린다. 이때 모든 사람을 정상이라고 예측하는 모델의 정확도는 99%다. 정확도만 보면 훌륭해 보인다. 하지만 이 모델은 환자를 단 한 명도 찾지 못한다. 분류 문제에서는 어떤 종류의 오류를 내는지가 정확도만큼 중요하다.

## 혼동 행렬

이진 분류 결과는 네 가지로 나뉜다. 이를 혼동 행렬(Confusion Matrix)로 표현하면 아래와 같다.

| 실제 \ 예측 | 양성 | 음성 |
|---|---:|---:|
| 양성 | True Positive (TP) | False Negative (FN) |
| 음성 | False Positive (FP) | True Negative (TN) |

앞의 단어는 예측이 맞았는지 나타낸다. 뒤의 단어는 모델이 예측한 라벨을 나타낸다.

**False Positive**는 모델이 양성이라고 말했지만 틀린 경우다. 정상 메일을 스팸이라고 판단하는 일이 여기에 해당한다.

**False Negative**는 모델이 음성이라고 말했지만 틀린 경우다. 암 환자를 정상이라고 판단하는 일이 여기에 해당한다.

![ml-5-20.png](@/assets/images/ml-5-20.png)
*정확도 하나로는 False Positive와 False Negative의 비용 차이를 볼 수 없다.*

신용카드 default 예시를 보자. 만 명의 고객 중 실제 default 고객은 333명이다. 모델의 결과가 다음과 같다고 하자.

| 실제 \ 예측 | No | Yes |
|---|---:|---:|
| No | 9,644 | 23 |
| Yes | 252 | 81 |

전체 정확도는 높다. 하지만 default 고객 333명 중 모델이 잡은 사람은 81명뿐이다.

```python
tp, tn, fp, fn = 81, 9644, 23, 252

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
fpr = fp / (fp + tn)
fnr = fn / (fn + tp)

print(f"accuracy : {accuracy:.3%}")
print(f"precision: {precision:.3%}")
print(f"recall   : {recall:.3%}")
print(f"f1       : {f1:.3%}")
print(f"FPR/FNR  : {fpr:.3%} / {fnr:.3%}")
```
```bash
accuracy : 97.250%
precision: 77.885%
recall   : 24.324%
f1       : 37.071%
FPR/FNR  : 0.238% / 75.676%
```

정확도는 97.25%다. 하지만 재현율은 24.3%에 불과하다. 진짜 default 고객 네 명 중 세 명을 놓친다. 신용카드 회사 입장에서는 쓸모없는 모델이다.

각 지표는 서로 다른 질문에 답한다.

**정확도(Accuracy)**는 전체 중 맞힌 비율이다.

$$ \text{Accuracy} = \cfrac{TP + TN}{TP + TN + FP + FN} $$

**정밀도(Precision)**는 양성이라고 예측한 것 중 실제 양성의 비율이다.

$$ \text{Precision} = \cfrac{TP}{TP + FP} $$

정밀도가 낮으면 양성이라고 예측한 것 중 틀린 예측이 많다는 뜻이다. 스팸 필터에서는 정밀도가 중요하다. 정상 메일을 스팸으로 보내면 안 된다.

**재현율(Recall)**은 실제 양성 중 모델이 찾아낸 비율이다.

$$ \text{Recall} = \cfrac{TP}{TP + FN} $$

재현율이 낮으면 놓치는 것이 많다는 뜻이다. 암 진단이나 사기 탐지처럼 놓치는 비용이 큰 문제에서는 재현율이 중요하다.

**F1 스코어**는 정밀도와 재현율의 조화 평균이다.

$$ F1 = 2 \cdot \cfrac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}} $$

두 값이 모두 높아야 F1도 높다. 한쪽이 0이면 F1도 0이다.

## 임계값과 트레이드오프

확률 모델은 보통 $P(y=1 \mid x)$ 같은 점수를 낸다. 이 점수를 실제 클래스로 바꾸려면 임계값이 필요하다.

$$
\hat{y} =
\begin{cases}
1 & \text{if } P(y=1 \mid x) \ge t \\
0 & \text{otherwise}
\end{cases}
$$

임계값 $t$를 낮추면 더 많은 샘플을 양성으로 분류한다. 재현율은 올라가지만 False Positive도 늘어난다. 임계값을 높이면 확실한 것만 양성으로 분류한다. 정밀도는 올라갈 수 있지만 놓치는 양성이 늘어난다.

![ml-5-21.png](@/assets/images/ml-5-21.png)
*임계값을 바꾸면 False Positive와 False Negative의 균형이 달라진다.*

작은 예시로 확인해보자.

```python
scores = [0.95, 0.82, 0.71, 0.63, 0.55, 0.44, 0.37, 0.22, 0.10, 0.03]
y_true = [1, 0, 1, 1, 0, 0, 1, 0, 0, 0]

def counts_at(threshold):
    preds = [1 if s >= threshold else 0 for s in scores]
    tp = sum(p == 1 and y == 1 for p, y in zip(preds, y_true))
    fp = sum(p == 1 and y == 0 for p, y in zip(preds, y_true))
    fn = sum(p == 0 and y == 1 for p, y in zip(preds, y_true))
    tn = sum(p == 0 and y == 0 for p, y in zip(preds, y_true))

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return tp, fp, tn, fn, precision, recall

for threshold in [0.7, 0.5, 0.3]:
    tp, fp, tn, fn, precision, recall = counts_at(threshold)
    print(f"t={threshold:.1f}: TP={tp}, FP={fp}, FN={fn}, precision={precision:.2f}, recall={recall:.2f}")
```
```bash
t=0.7: TP=2, FP=1, FN=2, precision=0.67, recall=0.50
t=0.5: TP=3, FP=2, FN=1, precision=0.60, recall=0.75
t=0.3: TP=4, FP=3, FN=0, precision=0.57, recall=1.00
```

임계값을 낮출수록 더 많은 양성을 잡아내지만, 잘못 양성으로 분류하는 샘플도 늘어난다. 그래서 좋은 분류 모델을 만드는 일은 모델 학습만으로 끝나지 않는다. 서비스 목적에 맞게 임계값을 정해야 한다.

## ROC와 AUC

임계값 하나를 정해 놓고 보는 대신, 모든 임계값에서의 성능을 한 번에 보는 방법이 있다. **ROC 곡선(Receiver Operating Characteristic Curve)**이다.

ROC 곡선은 임계값을 움직이면서 다음 두 값을 그린다.

$$ TPR = \cfrac{TP}{TP + FN} $$

$$ FPR = \cfrac{FP}{FP + TN} $$

TPR은 재현율과 같다. FPR은 실제 음성 중 양성으로 잘못 분류한 비율이다. 좋은 모델은 TPR이 높고 FPR이 낮다. 그래서 ROC 곡선이 왼쪽 위에 가까울수록 좋다.

곡선 아래 면적을 **AUC(Area Under the Curve)**라고 부른다. AUC는 임의로 뽑은 양성 샘플이 임의로 뽑은 음성 샘플보다 더 높은 점수를 받을 확률로 해석할 수도 있다. 1에 가까울수록 좋고, 0.5면 무작위 예측과 비슷하다.

![ml-5-22.png](@/assets/images/ml-5-22.png)
*ROC 곡선은 임계값을 하나로 고정하지 않고 모델의 순위화 능력을 본다.*

다만 클래스 불균형이 심하면 ROC가 지나치게 좋아 보일 수 있다. 음성이 압도적으로 많으면 FPR이 작게 유지되기 쉽기 때문이다. 이런 경우에는 **정밀도-재현율 곡선(PR curve)**을 함께 보는 편이 낫다.

## 정리와 다음 글 예고

이번 글에서는 분류 문제를 베이즈 관점에서 다시 봤다.

**베이즈 분류기**는 평균 손실을 최소화하는 이상적 분류 규칙이다. 0-1 손실에서는 사후 확률이 가장 큰 클래스를 고르면 된다.

**베이즈 정리**는 $P(y \mid x)$를 $P(x \mid y)P(y)$로 바꿔 생각한다. 분류에선 같은 $x$를 두고 비교하므로 분모 $P(x)$는 대소 비교에서 사라진다.

**QDA**는 클래스마다 다른 분산과 공분산을 허용한다. 그래서 $x^2$ 항이 남고, 결정 경계가 곡선이 될 수 있다.

**LDA**는 모든 클래스가 같은 분산과 공분산을 가진다고 가정한다. 이 가정 덕분에 $x^2$ 항이 상쇄되어 결정 경계가 직선이 된다.

**나이브 베이즈**는 클래스가 주어졌을 때 피처들이 독립이라고 가정한다. 가정은 대개 틀리지만, 피처가 많은 텍스트 분류에서 빠르고 강한 베이스라인이 된다.

**분류 평가는 정확도만으로 부족하다.** 혼동 행렬, 정밀도, 재현율, F1, ROC, AUC를 함께 봐야 모델이 어떤 종류의 실수를 하는지 알 수 있다.

다음 글은 모델이 너무 자유로울 때 생기는 문제, 즉 **과적합(overfitting)**과 그 해결책을 다룬다. 데이터에 지나치게 맞춘 모델은 훈련 데이터에서는 좋아 보이지만 새 데이터에서는 쉽게 무너진다. 이제 이 문제를 줄이는 정규화와, 새 데이터에서도 잘 작동하는 일반화의 관점으로 넘어간다.

<br><br>
<br><br>

-----
이미지 출처: [[ML/DL] Lecture 6. Classification II (Discriminant Analysis)](https://www.youtube.com/watch?v=MCDCDgF11Kg&list=PL0E_1UqNACXA5u65LBjzFCAVSZ4xuBWqj&index=6)
