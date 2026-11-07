
# YOLO11 완전 기초 정리

> Python과 인공지능을 처음 공부하는 사람을 위한 YOLO11 학습 노트

## 목차

1. YOLO란?
2. YOLO11이란?
3. YOLO11의 작동 원리
4. 객체 탐지의 핵심 개념
5. YOLO11이 수행할 수 있는 작업
6. YOLO11 모델 종류
7. YOLO11과 다른 인공지능 모델의 차이
8. Google Colab에서 YOLO11 사용하기
9. YOLO11 코드 상세 해설
10. 직접 촬영한 사진 분석하기
11. YOLO11 모델 학습시키기
12. YOLO11 성능 평가 방법
13. YOLO11의 장점과 한계
14. 활용 분야와 프로젝트 아이디어
15. 핵심 요약 및 참고 자료

---

# 1. YOLO란?

## 1.1 YOLO의 정의

**YOLO(You Only Look Once)**는 이미지나 영상 속 물체를 탐지하는 데 사용되는 딥러닝 기반 컴퓨터 비전 모델 계열이다.

YOLO는 이미지 속에서 물체의 종류와 위치를 예측할 수 있다.

예를 들어 사진에 사람, 자동차, 강아지가 있다면 다음 정보를 얻을 수 있다.

- 어떤 물체가 존재하는가?
- 물체가 어디에 있는가?
- 탐지 결과의 신뢰도는 얼마인가?

### 예시

**입력 이미지**

```text
사진
 ├── 사람
 ├── 자동차
 └── 강아지
```

**YOLO의 출력**

```text
Person : 0.95
Car    : 0.91
Dog    : 0.88
```

위 숫자는 각 탐지 결과의 신뢰도 점수를 의미한다.

실제 YOLO는 물체의 종류뿐 아니라 위치를 나타내는 좌표도 출력한다.

## 1.2 YOLO라는 이름의 의미

YOLO는 **You Only Look Once**의 약자이다.

초기 YOLO 모델은 이미지를 한 번의 신경망 처리 과정으로 분석하여 객체의 위치와 종류를 예측하는 방식을 제안했다.

이러한 구조는 빠른 객체 탐지를 가능하게 한다.

단, 실제 모델 내부에서는 여러 계층과 연산이 수행된다.

## 1.3 컴퓨터 비전이란?

**Computer Vision(컴퓨터 비전)**은 컴퓨터가 이미지와 영상에서 정보를 추출하고 이해하도록 만드는 인공지능 분야이다.

대표적인 작업은 다음과 같다.

| 작업 | 설명 |
|---|---|
| Image Classification | 이미지 전체를 분류 |
| Object Detection | 물체의 종류와 위치 탐지 |
| Segmentation | 물체가 차지하는 픽셀 영역 구분 |
| Pose Estimation | 사람 등의 관절 및 자세 추정 |
| Object Tracking | 영상 속 물체의 움직임 추적 |

YOLO11은 이러한 컴퓨터 비전 작업 중 여러 종류를 지원한다.

---

# 2. YOLO11이란?

## 2.1 YOLO11의 정의

**YOLO11**은 Ultralytics에서 2024년 9월 공개한 YOLO 모델 계열이다.

객체 탐지뿐 아니라 이미지 분할, 자세 추정, 이미지 분류 등의 작업을 지원한다.

주요 특징은 다음과 같다.

1. 빠른 객체 탐지
2. 다양한 모델 크기 제공
3. 여러 컴퓨터 비전 작업 지원
4. Python을 이용한 간편한 사용
5. 사전 학습된 모델 제공
6. 사용자 데이터로 추가 학습 가능

## 2.2 사전 학습된 모델이란?

**Pretrained Model(사전 학습 모델)**은 이미 많은 데이터를 사용하여 학습한 모델이다.

예를 들어 `yolo11n.pt`는 사전 학습된 YOLO11 객체 탐지 모델의 가중치 파일이다.

이 파일을 사용하면 처음부터 모델을 학습시키지 않고도 물체를 탐지할 수 있다.

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
```

### 코드 설명

- `from ultralytics import YOLO`
  - Ultralytics 라이브러리에서 YOLO 클래스를 가져온다.

- `model = YOLO("yolo11n.pt")`
  - YOLO11 nano 모델을 불러온다.
  - 모델 파일이 없다면 자동으로 다운로드될 수 있다.

### `.pt` 파일이란?

`.pt`는 PyTorch에서 사용하는 모델 저장 형식이다.

YOLO의 `.pt` 파일에는 학습된 가중치 등의 모델 정보가 포함될 수 있다.

---

# 3. YOLO11의 작동 원리

YOLO11은 입력 이미지에서 특징을 추출하고 이를 분석하여 물체의 위치와 종류를 예측한다.

전체 과정은 다음과 같이 이해할 수 있다.

```text
입력 이미지
    ↓
이미지 전처리
    ↓
Backbone
    ↓
Neck
    ↓
Detection Head
    ↓
예측 결과 후처리
    ↓
최종 객체 탐지 결과
```

## 3.1 이미지 전처리

입력된 이미지를 모델이 처리할 수 있는 형태로 변환한다.

대표적인 과정은 다음과 같다.

- 이미지 크기 조정
- 픽셀 값 정규화
- 텐서 변환
- 필요한 경우 패딩 적용

## 3.2 Backbone

**Backbone은 이미지의 특징을 추출하는 부분**이다.

예를 들어 사람을 인식할 때 다음과 같은 시각적 특징을 분석한다.

- 선과 모서리
- 색상과 질감
- 형태와 구조
- 복잡한 시각적 패턴

신경망의 여러 계층을 거치면서 점차 복잡한 특징을 추출한다.

## 3.3 Neck

**Neck은 Backbone에서 추출한 특징을 결합하는 부분**이다.

이미지에는 크기가 서로 다른 물체가 존재한다.

예를 들어 가까이 있는 자동차는 크게 보이지만 멀리 있는 자동차는 작게 보인다.

Neck은 서로 다른 해상도의 특징 정보를 결합하여 다양한 크기의 물체를 탐지하는 데 도움을 준다.

## 3.4 Detection Head

**Detection Head는 추출된 특징을 바탕으로 최종 예측을 수행하는 부분**이다.

대표적으로 다음 정보를 예측한다.

- 물체의 위치
- 물체의 종류
- 탐지 신뢰도

## 3.5 후처리

모델은 동일한 물체에 대해 여러 후보 박스를 예측할 수 있다.

이때 중복된 예측을 정리하는 과정이 필요하다.

대표적인 방법이 **NMS(Non-Maximum Suppression)**이다.

NMS는 겹치는 탐지 후보 중 적절한 박스를 선택하여 중복 탐지를 줄인다.

---

# 4. 객체 탐지의 핵심 개념

## 4.1 Bounding Box

**Bounding Box(바운딩 박스)**는 이미지에서 물체의 위치를 나타내는 사각형이다.

```text
+-------------------------+
|                         |
|    +-------------+      |
|    |             |      |
|    |    DOG      |      |
|    |             |      |
|    +-------------+      |
|                         |
+-------------------------+
```

일반적으로 다음과 같은 좌표로 표현한다.

```text
(x1, y1, x2, y2)
```

- `x1`: 왼쪽 위 x좌표
- `y1`: 왼쪽 위 y좌표
- `x2`: 오른쪽 아래 x좌표
- `y2`: 오른쪽 아래 y좌표

이미지 좌표계에서는 일반적으로 왼쪽 위가 원점 `(0, 0)`이다.

## 4.2 Class

**Class(클래스)**는 탐지 대상의 종류를 의미한다.

예시:

```text
0 : person
1 : bicycle
2 : car
3 : motorcycle
```

클래스 번호는 사용하는 데이터셋의 정의에 따라 달라질 수 있다.

## 4.3 Confidence

**Confidence(신뢰도)**는 모델이 출력한 탐지 결과의 점수이다.

예시:

```text
person : 0.97
car    : 0.83
dog    : 0.65
```

일반적으로 점수가 높을수록 모델이 해당 탐지 결과에 더 높은 확신을 가지고 있음을 나타낸다.

단, 신뢰도 점수가 실제 정답 확률과 정확히 일치하는 것은 아니다.

## 4.4 IoU

**IoU(Intersection over Union)**는 두 영역이 얼마나 겹치는지를 나타내는 지표이다.

공식은 다음과 같다.

```text
IoU = 교집합 면적 / 합집합 면적
```

예를 들어 두 박스의 교집합 면적이 60이고 합집합 면적이 100이라면 다음과 같다.

```text
IoU = 60 / 100 = 0.6
```

IoU 값은 0에서 1 사이이다.

- 0: 두 영역이 겹치지 않음
- 1: 두 영역이 완전히 일치함

IoU는 객체 탐지 평가와 NMS 등의 과정에서 활용된다.

---

# 5. YOLO11이 수행할 수 있는 작업

YOLO11은 다양한 컴퓨터 비전 작업을 지원한다.

## 5.1 Object Detection

**객체 탐지**

이미지 속 물체의 위치와 종류를 예측한다.

예시:

- 사람 탐지
- 자동차 탐지
- 동물 탐지

사용 모델:

```python
model = YOLO("yolo11n.pt")
```

## 5.2 Instance Segmentation

**인스턴스 분할**

물체를 사각형으로 표시하는 것을 넘어 물체가 차지하는 픽셀 영역을 구분한다.

예를 들어 사람의 몸 형태를 따라 영역을 표시할 수 있다.

사용 모델:

```python
model = YOLO("yolo11n-seg.pt")
```

## 5.3 Pose Estimation

**자세 추정**

사람의 주요 관절 위치 등을 예측한다.

예시:

- 어깨
- 팔꿈치
- 손목
- 무릎
- 발목

사용 모델:

```python
model = YOLO("yolo11n-pose.pt")
```

## 5.4 Image Classification

**이미지 분류**

이미지 전체가 어떤 종류에 해당하는지 예측한다.

예를 들어 사진이 고양이인지 강아지인지 분류할 수 있다.

사용 모델:

```python
model = YOLO("yolo11n-cls.pt")
```

## 5.5 Oriented Bounding Box

**회전 바운딩 박스(OBB)**

물체의 방향에 맞게 회전된 사각형으로 위치를 표현한다.

예시:

- 항공 사진의 비행기
- 기울어진 물체
- 회전된 제품

사용 모델:

```python
model = YOLO("yolo11n-obb.pt")
```

---

# 6. YOLO11 모델 종류

YOLO11은 모델 크기에 따라 다섯 가지 대표적인 버전이 존재한다.

| 모델 | 크기 | 특징 |
|---|---|---|
| YOLO11n | Nano | 가장 작고 가벼움 |
| YOLO11s | Small | 비교적 가벼움 |
| YOLO11m | Medium | 중간 규모 |
| YOLO11l | Large | 큰 모델 |
| YOLO11x | Extra Large | 가장 큰 모델 |

### 모델 선택 기준

- `n`: 빠른 실습과 제한된 하드웨어에 적합
- `s`: 속도와 성능 사이의 균형을 고려할 때 사용
- `m`: 더 높은 성능이 필요한 경우 고려
- `l`: 충분한 연산 자원이 있는 경우 사용
- `x`: 높은 성능을 우선하고 연산 자원이 충분한 경우 고려

일반적으로 모델이 커질수록 연산량과 메모리 요구량이 증가한다.

다만 정확도와 속도는 데이터셋 및 실행 장치에 따라 달라질 수 있다.

**초보자에게는 `yolo11n.pt`를 추천한다.**

---

# 7. YOLO11과 다른 인공지능 모델의 차이

## 7.1 일반적인 이미지 분류 모델

이미지 전체에 하나의 분류 결과를 예측한다.

예시:

```text
입력: 강아지 사진

출력: dog
```

## 7.2 객체 탐지 모델

이미지 안에 있는 여러 물체의 위치와 종류를 예측한다.

예시:

```text
입력: 사람과 강아지가 있는 사진

출력:
person : 위치 + 신뢰도
dog    : 위치 + 신뢰도
```

## 7.3 YOLO와 생성형 AI의 차이

| 구분 | YOLO11 | 생성형 AI |
|---|---|---|
| 주요 목적 | 이미지 속 물체 인식 | 새로운 콘텐츠 생성 |
| 대표 입력 | 이미지, 영상 | 텍스트, 이미지 등 |
| 대표 출력 | 클래스, 위치, 신뢰도 | 텍스트, 이미지 등 |
| 활용 예시 | 로봇 물체 인식 | 챗봇, 이미지 생성 |

YOLO11은 주로 이미지 속 대상을 찾아내는 데 특화된 모델이다.

---

# 8. Google Colab에서 YOLO11 사용하기

## 8.1 Google Colab이란?

Google Colab은 웹 브라우저에서 Python 코드를 작성하고 실행할 수 있는 클라우드 기반 개발 환경이다.

별도의 복잡한 설치 없이 YOLO11을 실습할 수 있다.

공식 사이트:

https://colab.research.google.com/

## 8.2 실습 준비

1. Google Colab 접속
2. 새 노트북 생성
3. 코드 셀 추가
4. 아래 코드를 순서대로 실행

### STEP 1. Ultralytics 설치

```python
!pip install -U ultralytics
```

`pip`는 Python 패키지를 설치하는 도구이다.

Colab에서 `!`를 붙이면 셸 명령어를 실행할 수 있다.

### STEP 2. YOLO 라이브러리 불러오기

```python
from ultralytics import YOLO
```

### STEP 3. YOLO11 모델 생성

```python
model = YOLO("yolo11n.pt")
```

여기서는 YOLO11 Nano 객체 탐지 모델을 사용한다.

### STEP 4. 이미지 분석

```python
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg"
)
```

`source`는 분석할 이미지의 경로 또는 URL을 의미한다.

### STEP 5. 탐지 결과 출력

```python
from IPython.display import display
from PIL import Image

display(
    Image.fromarray(
        results[0].plot()[..., ::-1]
    )
)
```

이 코드는 탐지된 물체가 표시된 이미지를 Colab 화면에 출력한다.

---

# 9. YOLO11 코드 상세 해설

다음은 앞에서 사용한 코드를 하나로 합친 예제이다.

```python
# YOLO 라이브러리 불러오기
from ultralytics import YOLO

# 이미지 출력에 필요한 라이브러리
from IPython.display import display
from PIL import Image

# YOLO11 모델 생성
model = YOLO("yolo11n.pt")

# 이미지 분석
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    conf=0.25
)

# 결과 이미지 출력
annotated_image = results[0].plot()

display(
    Image.fromarray(
        annotated_image[..., ::-1]
    )
)
```

## 9.1 model이란?

```python
model = YOLO("yolo11n.pt")
```

`model`은 불러온 YOLO 모델을 저장하는 변수이다.

이후 `model`을 사용하여 이미지를 분석할 수 있다.

## 9.2 predict()란?

```python
results = model.predict(source="image.jpg")
```

`predict()`는 모델에 입력 데이터를 전달하여 예측을 수행하는 메서드이다.

## 9.3 results란?

```python
results = model.predict(...)
```

`results`에는 모델의 예측 결과가 저장된다.

여기에는 다음과 같은 정보가 포함될 수 있다.

- 탐지된 물체의 위치
- 클래스 번호
- 신뢰도
- 결과 이미지 관련 정보

## 9.4 conf란?

```python
conf=0.25
```

`conf`는 탐지 결과를 표시하기 위한 최소 신뢰도 임계값이다.

예를 들어 `conf=0.25`라면 일반적으로 신뢰도 점수가 0.25보다 낮은 탐지 후보를 제외한다.

값을 높이면 탐지 결과 수가 줄어들 수 있다.

반대로 값을 낮추면 더 많은 후보가 표시될 수 있지만 잘못된 탐지도 늘어날 수 있다.

## 9.5 plot()이란?

```python
results[0].plot()
```

`plot()`은 예측 결과를 이미지 위에 시각화하는 메서드이다.

바운딩 박스와 클래스 이름 등을 표시할 수 있다.

## 9.6 [..., ::-1]의 의미

```python
annotated_image[..., ::-1]
```

YOLO의 `plot()` 결과는 일반적으로 BGR 채널 순서의 NumPy 배열이다.

PIL 이미지로 올바르게 표시하기 위해 RGB 순서로 변환하는 코드이다.

---

# 10. 직접 촬영한 사진 분석하기

Google Colab에 직접 사진을 업로드하여 분석할 수 있다.

## STEP 1. 사진 업로드

```python
from google.colab import files

uploaded = files.upload()
```

코드를 실행하면 컴퓨터에 저장된 파일을 업로드할 수 있다.

## STEP 2. YOLO 모델 불러오기

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
```

## STEP 3. 업로드한 사진 분석

```python
filename = next(iter(uploaded))

results = model.predict(
    source=filename,
    conf=0.25
)
```

`filename`에는 업로드한 첫 번째 파일의 이름이 저장된다.

## STEP 4. 결과 확인

```python
from IPython.display import display
from PIL import Image

display(
    Image.fromarray(
        results[0].plot()[..., ::-1]
    )
)
```

### 주의사항

사전 학습된 모델은 학습 과정에서 배운 클래스만 인식할 수 있다.

예를 들어 특정 전자부품이나 직접 제작한 물체를 인식시키려면 별도의 데이터셋으로 학습해야 할 수 있다.

---

# 11. YOLO11 모델 학습시키기

## 11.1 모델 학습이란?

모델 학습(Training)은 데이터를 이용하여 신경망의 가중치를 조정하는 과정이다.

YOLO11은 사전 학습된 모델을 바탕으로 새로운 데이터에 맞게 추가 학습할 수 있다.

이를 **Fine-tuning(파인튜닝)**이라고 한다.

## 11.2 학습에 필요한 데이터

객체 탐지 모델을 학습하려면 일반적으로 다음 자료가 필요하다.

1. 학습 이미지
2. 이미지 속 물체의 위치 정보
3. 물체의 클래스 정보
4. 데이터셋 설정 파일

## 11.3 Annotation이란?

**Annotation(어노테이션)**은 이미지에 정답 정보를 표시하는 작업이다.

객체 탐지에서는 이미지 속 물체의 위치와 종류를 기록한다.

예를 들어 강아지 사진에서 강아지의 위치를 사각형으로 표시하고 `dog` 클래스를 지정할 수 있다.

## 11.4 YOLO 라벨 형식

일반적인 YOLO 객체 탐지 라벨은 다음과 같다.

```text
class_id x_center y_center width height
```

예시:

```text
0 0.50 0.50 0.30 0.40
```

각 값의 의미는 다음과 같다.

| 값 | 의미 |
|---|---|
| class_id | 클래스 번호 |
| x_center | 박스 중심의 x좌표 |
| y_center | 박스 중심의 y좌표 |
| width | 박스 너비 |
| height | 박스 높이 |

좌표와 크기는 일반적으로 이미지 크기를 기준으로 0~1 사이로 정규화한다.

## 11.5 데이터셋 폴더 구조

```text
dataset/
│
├── images/
│   ├── train/
│   └── val/
│
├── labels/
│   ├── train/
│   └── val/
│
└── data.yaml
```

- `images/train`: 학습 이미지
- `images/val`: 검증 이미지
- `labels/train`: 학습용 정답 라벨
- `labels/val`: 검증용 정답 라벨
- `data.yaml`: 데이터셋 설정 파일

## 11.6 data.yaml 파일

예를 들어 사람과 자동차를 탐지하도록 학습한다면 다음과 같이 설정할 수 있다.

```yaml
path: /content/dataset

train: images/train
val: images/val

names:
  0: person
  1: car
```

## 11.7 학습 코드

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="/content/dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
```

### 주요 매개변수

| 매개변수 | 설명 |
|---|---|
| data | 데이터셋 설정 파일 경로 |
| epochs | 전체 학습 데이터를 반복 학습하는 횟수 |
| imgsz | 학습 이미지 크기 |
| batch | 한 번에 처리하는 이미지 수 |

위 코드는 데이터셋이 준비되어 있어야 정상적으로 실행된다.

메모리가 부족하면 `batch` 값을 줄여야 할 수 있다.

## 11.8 Epoch란?

**Epoch(에포크)**는 전체 학습 데이터를 한 번 학습하는 과정을 의미한다.

예를 들어 다음과 같다.

```text
Epoch 1  → 전체 데이터 한 번 학습
Epoch 2  → 전체 데이터 두 번째 학습
Epoch 3  → 전체 데이터 세 번째 학습
...
Epoch 50 → 전체 데이터 50번째 학습
```

Epoch가 많다고 무조건 좋은 것은 아니다.

너무 오래 학습하면 학습 데이터에 지나치게 맞춰지는 **과적합(Overfitting)**이 발생할 수 있다.

## 11.9 학습된 모델 사용

학습이 완료되면 일반적으로 다음과 같은 가중치 파일이 생성된다.

```text
best.pt
last.pt
```

- `best.pt`: 검증 성능을 기준으로 선택된 모델
- `last.pt`: 마지막 학습 시점의 모델

학습된 모델은 다음과 같이 불러올 수 있다.

```python
model = YOLO("best.pt")

results = model.predict("test.jpg")
```

실제 파일 경로는 학습 결과가 저장된 위치에 맞게 지정해야 한다.

---

# 12. YOLO11 성능 평가 방법

객체 탐지 모델의 성능을 평가할 때는 다양한 지표를 사용한다.

## 12.1 Precision

**정밀도(Precision)**는 모델이 탐지했다고 판단한 결과 중 실제로 올바른 결과의 비율이다.

```text
Precision = TP / (TP + FP)
```

- TP: 올바르게 탐지한 경우
- FP: 잘못 탐지한 경우

정밀도가 높을수록 잘못된 탐지가 적은 경향이 있다.

## 12.2 Recall

**재현율(Recall)**은 실제 존재하는 정답 물체 중 모델이 올바르게 탐지한 비율이다.

```text
Recall = TP / (TP + FN)
```

- TP: 올바르게 탐지한 경우
- FN: 존재하는 물체를 놓친 경우

재현율이 높을수록 실제 물체를 놓치는 경우가 적은 경향이 있다.

## 12.3 mAP

**mAP(mean Average Precision)**는 객체 탐지 모델의 성능을 평가하는 대표적인 지표이다.

클래스별 Average Precision(AP)을 평균하여 계산한다.

대표적으로 다음 지표를 사용한다.

- `mAP50`: IoU 임계값 0.50을 기준으로 평가
- `mAP50-95`: IoU 0.50부터 0.95까지 여러 임계값에서 평가한 결과의 평균

일반적으로 값이 높을수록 탐지 성능이 좋다고 평가한다.

## 12.4 모델 검증 코드

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

metrics = model.val()
```

`val()`은 검증 데이터셋을 사용하여 모델 성능을 평가한다.

사용하는 모델과 데이터셋 설정에 따라 평가 결과가 달라진다.

---

# 13. YOLO11의 장점과 한계

## 13.1 장점

### 빠른 처리 속도

YOLO 계열은 효율적인 객체 탐지를 목표로 설계되어 실시간 영상 처리에 활용할 수 있다.

### 다양한 모델 크기

하드웨어 성능에 따라 작은 모델부터 큰 모델까지 선택할 수 있다.

### 사용하기 쉬운 Python API

Ultralytics 라이브러리를 사용하면 비교적 적은 코드로 모델을 실행할 수 있다.

### 다양한 작업 지원

객체 탐지뿐 아니라 분할, 자세 추정, 분류 등을 지원한다.

### 사용자 데이터 학습

직접 수집한 이미지와 라벨을 사용하여 특정 물체를 탐지하도록 학습시킬 수 있다.

## 13.2 한계

### 학습하지 않은 물체 인식의 어려움

기본 모델이 학습하지 않은 특정 물체는 제대로 인식하지 못할 수 있다.

### 조명과 촬영 환경의 영향

어두운 환경, 강한 역광, 흐린 이미지에서는 성능이 저하될 수 있다.

### 작은 물체 탐지의 어려움

이미지에서 매우 작게 나타나는 물체는 탐지하기 어려울 수 있다.

### 물체가 가려진 경우

다른 물체에 가려져 있는 경우 탐지 정확도가 떨어질 수 있다.

### 하드웨어 제약

큰 모델은 많은 메모리와 연산 성능을 요구할 수 있다.

---

# 14. YOLO11 활용 분야와 프로젝트 아이디어

## 14.1 자율주행

카메라 영상을 분석하여 주변 물체를 탐지할 수 있다.

예시:

- 자동차
- 보행자
- 자전거
- 교통 관련 물체

단, 실제 자율주행에는 객체 탐지 외에도 거리 추정, 경로 계획, 제어 및 안전 검증 등이 필요하다.

## 14.2 로봇공학

로봇이 주변 환경을 인식하는 데 활용할 수 있다.

예시:

- 사람 인식 로봇
- 물체를 찾는 로봇
- 물체를 따라 움직이는 로봇
- 특정 물체를 분류하는 로봇

## 14.3 스마트 카메라

카메라 영상에서 특정 물체를 탐지하여 알림이나 자동화 기능을 구현할 수 있다.

## 14.4 제조업

제품의 위치를 탐지하거나 특정 결함을 인식하는 시스템에 활용할 수 있다.

단, 결함 탐지에는 해당 결함에 맞는 데이터셋과 추가 학습이 필요할 수 있다.

## 14.5 초보자 추천 프로젝트

### 프로젝트 1. 사진 속 물체 찾기

**난이도: 쉬움**

목표:
- 사진을 업로드한다.
- YOLO11로 물체를 탐지한다.
- 탐지 결과를 화면에 출력한다.

필요 기술:
- Python 기초
- Google Colab
- YOLO11

### 프로젝트 2. 웹캠 물체 탐지

**난이도: 보통**

목표:
- 웹캠 영상을 입력받는다.
- 영상 속 사람이나 물체를 탐지한다.
- 실시간으로 결과를 표시한다.

필요 기술:
- Python
- OpenCV
- YOLO11

참고로 Google Colab에서는 로컬 웹캠을 일반적인 Python 프로그램처럼 직접 다루기 어려울 수 있다.

실시간 웹캠 프로젝트는 VS Code 등의 로컬 Python 환경에서 진행하는 것이 편리하다.

### 프로젝트 3. 로봇 물체 인식

**난이도: 보통~어려움**

목표:
- 로봇에 카메라를 연결한다.
- YOLO11로 특정 물체를 인식한다.
- 탐지 결과에 따라 로봇을 움직인다.

필요 기술:
- Python
- YOLO11
- 카메라 처리
- 로봇 제어
- 하드웨어 통신

---

# 15. 핵심 요약

## 반드시 기억할 개념

| 개념 | 핵심 의미 |
|---|---|
| YOLO | 빠른 객체 탐지를 위한 딥러닝 모델 계열 |
| YOLO11 | Ultralytics에서 공개한 YOLO 모델 계열 |
| Object Detection | 물체의 종류와 위치를 예측하는 작업 |
| Bounding Box | 물체의 위치를 나타내는 사각형 |
| Class | 탐지한 물체의 종류 |
| Confidence | 탐지 결과의 신뢰도 점수 |
| IoU | 두 영역의 겹침 정도 |
| Backbone | 이미지 특징 추출 |
| Neck | 여러 특징 정보 결합 |
| Detection Head | 물체의 위치와 종류 예측 |
| Pretrained Model | 미리 학습된 모델 |
| Fine-tuning | 사전 학습 모델을 추가 학습하는 과정 |
| Epoch | 전체 학습 데이터의 반복 학습 횟수 |
| Precision | 탐지 결과 중 올바른 결과의 비율 |
| Recall | 실제 물체 중 올바르게 탐지한 비율 |
| mAP | 객체 탐지 성능 평가 지표 |

## YOLO11 기본 코드

```python
from ultralytics import YOLO

# 모델 불러오기
model = YOLO("yolo11n.pt")

# 이미지 분석
results = model.predict("image.jpg")

# 결과 확인
results[0].show()
```

위 코드는 `image.jpg` 파일이 현재 실행 환경에 존재해야 정상적으로 작동한다.

## 추천 학습 순서

1. Python 기초 문법 익히기
2. Google Colab 사용법 익히기
3. YOLO11 사전 학습 모델 실행
4. 직접 촬영한 사진 분석
5. Bounding Box와 Confidence 이해
6. OpenCV 기초 학습
7. 직접 데이터셋 제작
8. YOLO11 모델 파인튜닝
9. 웹캠 및 로봇 프로젝트에 적용

---

# 참고 자료

- [Ultralytics YOLO11 공식 문서](https://docs.ultralytics.com/models/yolo11/)
- [Ultralytics Python 사용법](https://docs.ultralytics.com/usage/python/)
- [Ultralytics 객체 탐지 가이드](https://docs.ultralytics.com/tasks/detect/)
- [Ultralytics 모델 학습 가이드](https://docs.ultralytics.com/modes/train/)
- [Google Colab](https://colab.research.google.com/)
- [PyTorch 공식 사이트](https://pytorch.org/)

---

**학습 목표:** YOLO11의 기본 원리를 이해하고, Python과 Google Colab을 이용하여 이미지 속 물체를 탐지하며, 이후 직접 제작한 데이터셋으로 모델을 학습시킬 수 있도록 한다.
