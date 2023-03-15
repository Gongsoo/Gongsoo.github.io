---
title: Machine Learning from Disastor [Pytorch]
date: 2023-03-15 21:52:53 +0900
categories: [pytorch, study]
tags: [pytorch, machine learning]     # TAG names should always be lowercase
pin: True
---
# [Python Study] Pytorch를 이용한 Titanic 생존자 예측 Model 연습(1)

2023.03.15

고영수

# 서론

### ‘Machine learning from disastor’

최근에 재개봉한 Titanic을 영화관에서 관람했다. 25년된 영화였지만, 촌스럽지 않았으며 세 시간이 어떻게 지나가는지 모르게 몰입했다. 정말 큰 사고였고, 인재였으며 안전에 대해 다시 한 번 생각해볼 수 있었다. Titanic data를 이용해  machine learning을 연습하는 것을 알고 있었지만, 이런 숭고한 제목이 있는지는 몰랐다.

이번에 연습해 볼 것은 pandas의 Dataframe 사용과, 어떻게 Data를 가다듬어 Train set을 만들지이다. model과 accuracy는 크게 고려하지 않겠지만 Train set에 따라 정확도가 어떻게 바뀌는지를 확인해볼 것이다. 

첫 번째는 data를 추가하지 않고, 결측치만을 채워서 Train set을 만들고 가장 기초적인 Linear model을 통해 학습시킬 것이다.

코드 작성과 훈련은 colab으로, accuracy는 Dacon의 [타이타닉 생존 예측 경진대회](https://dacon.io/competitions/open/235539/overview/description) 기준으로 측정하겠다.

# Module import

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from google.colab import drive
drive.mount('/content/drive')
```

# Load Dataset

```python
cd "/content/drive/MyDrive/"
```

```python
!unzip -qq "/content/drive/MyDrive/타이타닉.zip"
```

```python
!mkdir titanic_pytorch
```

```python
import shutil
shutil.move('/content/drive/MyDrive/train.csv','/content/drive/MyDrive/titanic_pytorch/train.csv')
shutil.move('/content/drive/MyDrive/test.csv','/content/drive/MyDrive/titanic_pytorch/test.csv')
shutil.move('/content/drive/MyDrive/submission.csv','/content/drive/MyDrive/titanic_pytorch/submission.csv')
shutil.move('/content/drive/MyDrive/타이타닉.zip','/content/drive/MyDrive/titanic_pytorch/titanic_data.zip')
```

```python
train = pd.read_csv('/content/drive/MyDrive/titanic_pytorch/train.csv')
test = pd.read_csv('/content/drive/MyDrive/titanic_pytorch/test.csv')
sub = pd.read_csv('/content/drive/MyDrive/titanic_pytorch/submission.csv')
```

# Data Pre-processing

우선 Dataset을 보면 결측치가 많은 Cabin, 생존과 크게 상관없어보니는 Ticket열은 제거해주는게 좋을 것 같다.

```python
train = train.drop('Ticket',axis = 1)
train = train.drop('Cabin',axis = 1)
test = test.drop('Ticket',axis = 1)
test = test.drop('Cabin',axis = 1)
```

Embarked는 결측치가 세개밖에 없기 때문에 가장 많은 정착지였던 Southampton으로 채워주었다. 그리고 test data에 Fare에 결측치는 20으로 적당하게 채워주었다.

```python
train = train.fillna({'Embarked':'S'})
test = test.fillna({'Embarked':'S','Fare':20})
```

Age 결측치는 이름의 Mr, Miss, Mrs 등을 근거로 적당히 채워보았다.

```python
age_map = {'Mr.':20,'Miss.':15,'Mrs.':20,'Ms.':10,'Mme.':30,'Mlle.':15,'Dona.':30,'Countess.':30,'Sir.':25,'Lady.':20,'Jonkheer.':20,'Don.':20,'Col.':35,'Major.':30,'Capt.':23,'Dr.':30,'Master.':25,'Rev.':35}

for t in [train] :
  for i in range(len(t['Age'])) :
    if np.isnan(t['Age'][i]) :
      for j in age_map :
        if j in t['Name'][i] :
          t['Age'][i] = age_map[j]
for t in [test] :
  for i in range(len(t['Age'])) :
    if np.isnan(t['Age'][i]) :
      for j in age_map :
        if j in t['Name'][i] :
          t['Age'][i] = age_map[j]
```

마지막으로 data input은 숫자여야지 훈련이 가능하다. 그래서 성별은 male : 1, female : 2로, 정착지는 S : 1, Q : 2, C : 3으로 바꿔주었다.

```python
em_map = {'S':1,'Q':2,'C':3}
for t in [train] :
  t['Embarked'] = t['Embarked'].map(em_map)
for t in [test] :
  t['Embarked'] = t['Embarked'].map(em_map)

s_map = {'male':1, 'female':2}
for t in [train] :
  t['Sex'] = t['Sex'].map(s_map)
for t in [test] :
  t['Sex'] = t['Sex'].map(s_map)
```

이제 Name, PassengerId는 필요 없으니 두 개열을 없애보자.

```python
train = train.drop(['Name'], axis=1)
test = test.drop(['Name'], axis=1)
train = train.drop(['PassengerId'], axis=1)
test = test.drop(['PassengerId'], axis=1)
```

훈련 시킬 데이터 x와 정답 label을 나눠보자

```python
x = train.drop(['Survived'],axis=1)
y = train['Survived']
```

일단 편의를 위해 scale을 해보았다. 데이터 양이 많은것 같지도 않고 모델도 단순하게 제작할 거라 안해도 될거같음. 그래서 해야하는 건가? 흠,,

```python
scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

scaler.fit(test)
test_scaled = scaler.transform(test)
```

# Set model

거창한거 없이 간단하게 linear모델을 만들어보자. input은 7, output은 0또는 1이 나와야하므로 Sigmoid를 모델의 마지막에 추가해줬습니다.

```python
class LiModel(nn.Module) :
  def __init__(self):
    super(LiModel,self).__init__()
    self.layer1 = nn.Sequential(
        nn.Linear(7,1),
        nn.Sigmoid()
    )
    
  
  def forward(self, x) :
    x = self.layer1(x)
    return x
model = LiModel()
model
```

dataloader를 만들어 보겠습니다. 

```python
train_x = torch.Tensor(x_scaled).float()
train_y = torch.Tensor(y).float()
test_x = torch.Tensor(test.values).float()

train_dataset = TensorDataset(train_x,train_y)
train_data_loader = DataLoader(train_dataset,batch_size = 64, shuffle = True,drop_last = True)
```

# Train

이제 훈련을 시켜봅시다. optimizer는 Adam, learing rate는 0.01, loss는 binary cross entropy로 정했습니다. epoch는 500번 하면서 loss가 가장 낮을 때 모델을 저장하도록 했습니다.

```python
epochs = 500
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
train_loss_min = np.inf
for epoch in range(epochs) :
  train_loss = 0
  correct = 0
  for x,y in train_data_loader :
    
    model.train()
    optimizer.zero_grad()
    pred = model(x).view(-1)
    loss = torch.nn.functional.binary_cross_entropy(pred,y)

    
    loss.backward()
    optimizer.step()
    output = pred >= torch.FloatTensor([0.5])
    correct += output.float() == y
    train_loss += loss.item() * len(x)
  train_loss = train_loss / len(train_x)
  if (epoch + 1) % 20 == 0:
      print('Epoch {}/{}, Prediction : {}/{}, Cost : {}'.format(epoch+1, epochs, sum(correct), len(train_x), train_loss))
  if train_loss < train_loss_min:
      print('=*=*=*= Loss decreased ({:10f} ===> {:10f}). Saving the model! =*=*=*='.format(train_loss_min, train_loss))
      torch.save(model.state_dict(), 'model.pt')
      train_loss_min = train_loss
```

loss가 0.4정도에서 저장됐네요. (결과는 github에서 확인 가능합니다.)

# Prediction & submit

```python
model.eval()
with torch.no_grad():
  test_y_pred = model(test_x)
```

eval()은 Dropout이나 batchnorm등을 비활성 시키고 with torch.no_grad()는 gradient 트래킹을 안한다고 합니다. 뭐 지금은 크게 상관없겠지만요.

```python
sub['Survived'] = torch.Tensor.detach(test_y_pred)
sub.to_csv('베이스 라인 .csv', index = False)
```

이대로 제출 해봅시당.

![타이타닉1](https://user-images.githubusercontent.com/121560522/225315991-b70da6c9-6f14-4b80-8f66-f6b367408e69.png)

에.. 좋은 결과는 나오지 않았네요. 다음번에는 데이터를 요리조리 만져서 더 나은 결과를 만들어 보겠습니다.

# Reference

[https://velog.io/@hyungraelee/Titanic-Machine-Learning-from-Disaster-Pytorch](https://velog.io/@hyungraelee/Titanic-Machine-Learning-from-Disaster-Pytorch)
[https://m.blog.naver.com/khm159/221792363035](https://m.blog.naver.com/khm159/221792363035)