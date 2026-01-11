## 5. 합성곱 신경망(CNN)
### 5.1 합성곱 신경망의 기본 개념과 구조
- 기존 신경망(DNN)은 이미지가 커질수록 학습량이 지나치게 증가 -> 계산 효율 하락
- CNN은 2D의 구조를 유지, 파라미터 공유 -> 효율성 증가
- CNN의 특징
  - 지역적 연결성 : 각 이미지의 일부 영역에만 연결 -> 특정 부분의 특징을 효과적으로 파악 (not 전체)
  - 파라미터 공유 : 각 레이어에 동일한 필터가 적용 -> 학습할 파라미터 수 감소
  - 평행 이동 불변성 : 객체 위치가 달라져도 동일 특징 감지
### 5.2 합성곱 신경망의 주요 구성 요소
1. 합성곱 레이어
  - 필터(커널) : 가중치 행렬
 ![743E548E-6D76-4350-9438-7C1A68C55967_4_5005_c](https://github.com/user-attachments/assets/1161c026-dd37-4fdb-8f94-5367dca5213b)
(출처 : https://wiki1.kr/index.php?title=합성곱_신경망&mobileaction=toggle_view_desktop)
  - 필터의 동작 원리 : 밝기가 급격히 바뀌는 경계에서 큰 값이 생성
  - 스트라이드 : 필터가 이동하는 간격 n이라고 했을 때 n 픽셀씩 이동
  - 패딩 : 입력 이미지 가장자리에 추가되는 픽셀
  - 출력 특성 맵의 크기 : 출력크기 = (입력크기- 필터크기十2X패딩)/스트라이드十1
2. 활성화 함수와 비선형
  - ReLU

| 활성화 함수 | 수식 (텍스트) | 특징 | 장점 | 단점 / 주의점 | 목적 |
|------------|--------------|------|------|--------------|------|
| Leaky ReLU | f(x) = max(0.01x, x) | 음수 구간에 작은 기울기 | Dying ReLU 완화 | 기울기 값 수동 설정 | ReLU 개선 |
| ELU | x (x > 0), α(e^x - 1) (x ≤ 0) | 음수 구간이 부드러운 곡선 | 평균 활성화 0 근처 | 계산량 증가 | 학습 안정화 |
| SELU | Scaled ELU | 자기 정규화 특성 | 깊은 네트워크 안정 | 조건 제한 있음 | 깊은 모델 |
| Swish | f(x) = x * sigmoid(x) | 부드러운 비단조 함수 | 성능 우수 | 계산량 증가 | 성능 최적화 |

- 풀링 레이어
  - 최대 풀링 : 윈도우 내 가장 큰 값 선택
  - 평균 풀링 : 윈도우 내 값들의 평균
  - 풀링의 효과 : 파라미터 수 감소, 과적합 방지, 수용 영역 확대
  - 풀링 레이어 설계시 주의 사항 : 풀링 레이어가 너무 많으면 정보 손실 위험, 스트라이드는 풀링의 크기와 동일하게 설정
- 파이토치를 이용한 합성곱 신경망 구현

| 레이어            | 역할    | 설명                                                  | 주요 파라미터                                                                                                                                  |
| -------------- | ----- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `nn.Conv2d`    | 특징 추출 | 이미지와 같은 2차원 입력에 여러 개의 커널(필터)을 적용하여 에지, 패턴 등의 특징을 추출 | `in_channels` : 입력 특징 맵 채널 수<br>`out_channels` : 출력 특징 맵 채널 수(필터 개수)<br>`kernel_size` : 커널 크기<br>`stride` : 이동 간격<br>`padding` : 가장자리 패딩 |
| `nn.MaxPool2d` | 특징 축소 | 특징 맵에서 가장 큰 값만 선택하여 공간 크기를 줄이고 중요한 정보만 유지           | `kernel_size` : 풀링 윈도우 크기<br>`stride` : 이동 간격                                                                                            |

- 배치 정규화 : 미니 배치의 활성화 값을 정규화 하여 학습 속도를 높임
- 드롭아웃 : `nn.Dropout2d`, `nn .Dropout`
3. CIFAR-10 분류기
- CIFAR-10 : 10개의 클래스의 32x32 컬러 이미지 6만장으로 구성된 데이터셋 (train: 5만 test: 1만)

```python
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# 데이터 전처리 정의
# ===============================

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # 랜덤 크롭 (데이터 증강)
    transforms.RandomHorizontalFlip(),          # 랜덤 수평 뒤집기 (데이터 증강)
    transforms.ToTensor(),                      # PIL 이미지 → Tensor
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),               # 평균 (RGB)
        (0.2470, 0.2435, 0.2616)                # 표준편차 (RGB)
    )
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

# ===============================
# 데이터셋 불러오기
# ===============================

print("CIFAR-10 데이터셋 로드 중...")

try:
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_test
    )

    # ===============================
    # 데이터로더 생성
    # ===============================

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=0
    )

    print("CIFAR-10 데이터셋 로드 완료")
    print(f"훈련 데이터: {len(train_dataset)}개")
    print(f"테스트 데이터: {len(test_dataset)}개")

except Exception as e:
    print(f"데이터셋 로드 중 오류 발생: {e}")
    print("인터넷 연결 또는 데이터셋 경로를 확인하세요.")

# ===============================
# 클래스 이름
# ===============================

classes = (
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)

# ===============================
# 이미지 시각화 함수
# ===============================

def imshow(img):
    img = img / 2 + 0.5              # 정규화 해제
    npimg = img.numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()`
```
`RandomCrop` + `HorizontalFlip` : 이미지를 무작위로 자르고 좌우 반전 -> 데이터 증강 효과 -> 학습 데이터 수 증가

```python
# 더 다양한 데이터 증강 기법 적용
transform_train_augmented = transforms.Compose([
    transforms.RandomCrop(32, padding=4),          # 랜덤 크롭
    transforms.RandomHorizontalFlip(),              # 랜덤 수평 뒤집기
    transforms.RandomRotation(15),                  # 랜덤 회전 (±15도)
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),                                              # 색상 변화
    transforms.ToTensor(),                          # Tensor 변환
    transforms.Normalize(
        (0.4914, 0.4822
```
### 5.3 고급 합성곱 신경망 아키텍처
- VGG 네트워크
1. 작은 필터 크기를 반복 -> 큰 필터와 같은 효과

  - why?
    
    1.  큰 필터 한 번이면 한 번의 비선형 (복잡한 학습 한계)/작은 필터 여러 장이면 여러 번의 비선형 변환으로 표현력이 강해짐 
    2. 적은 파라미터로 큰 필터의 표현 효과를 나타낼 수 있음
```
python
# 파라미터 계산 예시: 입력 채널 = 64, 출력 채널 = 64
# 5x5 필터 한 장
params_5x5 = 64 * 64 * 5 * 5
print("5x5 필터 한 장 파라미터 수:", params_5x5)  # 81920

# 3x3 필터 두 장
params_3x3_2 = 64 * 64 * 3 * 3 * 2
print("3x3 필터 두 장 파라미터 수:", params_3x3_2)  # 36864
```
2. 깊은 네트워크 구조로 복잡한 특징 학습이 가능함
3. 풀링으로만 공간 차원 축소: 공간 차원의 축소는 모두 풀링 레이어를 통해 이뤄지며， 합성곱 층의 스트라이드는 1로 설정
   
   - VGG에서는 Conv 층마다 stride=1 → 출력 크기 거의 입력 크기와 같음 -> 이미지 공간 정보 손실 최소화 + 특징맵(어떤 위치에 어떤 특징이 있는지 숫자로 표시)의 세밀한 위치 정보 유지
   - 공간 차원 축소는 풀링(지역에서 가장 중요한 값만 뽑음)으로만 -> 위치 정보의 손실은 줄이고 중요한 특징만 남겨 효율 증대

### 5.4 ResNet 구조
- 잔차 학습(Residual Learning)
  - 깊은 네트워크를 학습할 때 생기는 기울기 소실 문제 완화
- 직접 연결
  - 입력을 직접 출력에 더하는 연결
  - 신경망에서 한 층의 출력을 다음 층으로 보내는 것과 별도로, 더 뒤쪽 층에도 그대로 전달하는 연결
 
```
python
  입력 x ---> [층1] ---> [층2] ---> 출력
    |________________________|
          (Skip Connection)
```
  -> 중간 출력을 나중 층으로 직접 전달해서, 신호가 사라지거나 변형되는 걸 막는 연결
*pooling의 개념적 반대는 upsampling이 맞지만 upsampling이 사라진 정보를 되돌릴 수 없기에 반대 연산은 아님
### 5.5 합성곱 신경망의 성능 최적화
1. 주요 가중치 초기화 방법
- He(Kaiming) 초기화 : ReLU 활성화 함수와 함께 사용하기 좋음
- Xavier 초기화 : 시그모이드나 tanh 활성화 함수와 함께 사용하기 좋음
- 정규 분포 초기화 : 평균과 표준편차를 지정하여 초기화
2. 최적화 알고리즘 수렴 분석
- SGD with Momentum: 수렴은 느리지만 안정적이며, 하이퍼파라미터 튜닝 시 일반화 성능이 우수함
- Adam: 초기 수렴이 빠르고 튜닝이 쉽지만, 경우에 따라 SGD보다 일반화 성능이 낮을 수 있음
- AdamW: Adam에 올바른 가중치 감쇠를 적용한 방법으로, 빠른 수렴과 뛰어난 일반화 성능을 동시에 제공
3. 모델 압축 기법
- 가지치기 : 중요도가 낮은 가중치나 필터 제거
- 필터 가지치기 : 전체 필터를 제거해 모델 구조 자체를 간소화 (단, 중요한 특징을 담당하는 필터는 제거하면 안 됨)
```
python
print("\nApplying pruning...")

for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        prune.l1_unstructured(
            module,
            name='weight',
            amount=0.3
        )
```

4. 양자화
- 모델의 가중치와 활성화 값을 정밀도가 낮은 데이터 타입으로 변환
- 모델 크기 감소 -> 추론 속도 향상 -> 메모리 사용량 감소
5. 지식 증류
- 큰 모델의 지식을 작은 모델로 전달하는 기법
- 더 작은 모델로 유사한 성능을 달성
### 5.6 합성곱 신경망의 시각화와 해석
- 합성곱 신경망의 중간층(Conv, ReLU, Pool 등에서 나오는 feature map) 활성화를 시각화 -> 모델이 학습하는 특징을 이해할 수 있음
<img width="593" height="390" alt="스크린샷 2026-01-11 오후 11 34 52" src="https://github.com/user-attachments/assets/9a6d0015-13c5-4756-b763-dfbe09129925" />
-> conv1에서 3으로 갈 수록 점점 더 고수준의 학습 특징이 포착됨
- 필터 패턴 분석 : 필터(가중치)를 시각화하면 각 필터가 어떤 패턴을 감지하도록 학습되었는지 알 수 있음
1. 모델 해석 기법
- CAM
  - 합성곱 신경망이 분류 결과를 도출할 때 이미지의 어떤 부분에 주목했는지 시각화
  - "모델이 특정 클래스를 판단할 때 이미지의 어디를 봤는지”를 히트맵으로 보여주는 방법
  - 전역 평균 풀링(GAP)을 사용하는 네트워크에서만 사용 가능 -> CAM은 “클래스 점수 = 각 특징맵의 선형 결합” 구조여야 하는데, GAP가 있어야 그 구조가 성립하기 때문
  - GAP은 특징맵 전체를 평균애서 채널당 숫자 1개로 줄이는 연산
  <img width="536" height="219" alt="스크린샷 2026-01-11 오후 11 38 27" src="https://github.com/user-attachments/assets/056e6b22-9c30-4c35-8b96-68f8a388b6b1" />
- Grad-CAM
  - CAM의 일반화 버전으로 네트워크 아키텍처에 상관 없이 적용 가능
  - 출력 클래스에 대한 기울기를 사용해서 중요 영역을 계산
  <img width="534" height="212" alt="스크린샷 2026-01-11 오후 11 40 20" src="https://github.com/user-attachments/assets/fe7fe379-bb61-498a-98de-a022ab912e62" />
