---
title: Data Type(Numbers, Strings)[Python]
date: 2023-02-07 21:52:53 +0900
categories: [python, study]
tags: [python, data type]     # TAG names should always be lowercase
pin: True
---

# Python의 자료형(숫자형, 문자열)
Python 스터디로 [점프 투 파이썬](https://wikidocs.net/11)을 2주에 걸쳐 정리하기로 했다. 이 post에서는 2장에 해당하는 자료형 중 숫자형과 문자열에 대해 정리한다. 점프 투 파이썬을 기반으로 실제로 공부하면서 궁금했던 것을 추가하면서 작성 할 예정이다.

## 숫자형
말 그대로 숫자 형태로 이루어진 자료형으로 정수(int), 실수(float), 2진수(bin) 등이 있다.
#### 정수형과 실수형
정수형과 실수형은 우리가 지금 까지 배워온 숫자처럼 사용할 수 있다.
```
> 123
> 1.35
```
그러고 지수표현방식으로도 표현 할 수 있다.

```
> 8.855e-12
> 4.56e10
```
### 진수표현
python에서는 2진수, 8진수, 16진수 함수인 bin(), oct(), hex()를 지원한다. 이 때 변환 된 data의 data type은 추후에 공부할 string으로 바뀐다

```
> bin(10)
0b1010
> oct(10)
0o12
> hex(10)
0xa
> print(type(bin(10)))
<class 'str'>
```
반대로 2, 8, 16진수를 10진수로 변환 할 때는 int()함수를 사용하면 된다.

### 연산기호
기본적으로 더하기, 빼기, 곱하기, 나누기 연산이 가능하다.(이때 곱하기는 *, 나누기는 /를 사용한다. 그러고 /로 연산시 float으로 data type이 바뀐다.)
이외에도 나머지를 반환하는 %, 제곱연산 **, 몫을 반환하는 //를 주로 사용한다.

```
> 7%4
3
> 2**10
1024
> 5//2
2
``` 
______________
## 문자열(String)
쉽게 생각하면 ' ', " " 혹은 """ """로 둘러 쌓여 있는 거의 모든 자료형을 문자열이라고 한다. 이 때 '123'같이 숫자가 따옴표로 묶여있는 것도 문자열이니 헷갈리지 말자.

Python에서 문자열을 표현하는데 세가지 방법이나 있는 이유는 문자열 안에 따옴표가 쓰일 수 있기 때문이다.
```
> a = 'Han's bakery'
SyntaxError: invalid syntax
```
위와 같은 경우는 제일 바깥에 있는 따옴표를 큰 따옴표로 바꿔줌으로써 SyntaxError를 방지할 수 있다. (또한 안에 있는 따옴표 앞에 \를 넣는 방법도 있다.)

### 문자열의 연산
Python의 가장 큰 특징으로 사용법은 다음과 같다.
#### 문자열 더하기
```
> a = 'Py'
> b = 'thon'
> a + b
'Python'
```
#### 문자열 곱하기

```
> a = 'Python'
> a * 2
'PythonPython'
```
#### 문자열 길이 구하기
len함수를 이용해 구할 수 있다. len함수는 숫자형 자료형 이외의 자료형의 길이를 구할 수 있다.
```
> a = 'Python'
> len(a)
6
```

### 인덱싱, 슬라이싱
예제를 통해 인덱싱과 슬라이싱에 대해 알아보자

```
> alphabet = 'abcdefghijklmnopqrstuvwxyz'
> alphabet[2]
'c'
> alphabet[1:4]
'bcd'
> alphabet[:5]
'abcde'
> alphabet[10:]
'klmnopqrstuvwxyz'
> alphabet[:]
'abcdefghijklmnopqrstuvwxyz'
```
alphabet이라는 변수에 a~z까지 순서대로 적힌 문자열 자료형을 저장했다.
그 뒤 alphabet[2]를 입력하자 'c'라는 문자를 반환했다. c는 세 번째에 있는 문자이다.
str[i]는 문자열 자료형의 **i+1**번째에 있는 문자를 반환한다. i+1번째 인 이유는 파이썬이 숫자를 **0**부터 세기 때문이다.
이렇게 str[숫자]는 문자열 안의 특정 값을 뽑아내는 역할을 하고, 이를 **인덱싱**이라고 한다.
('숫자'자리에는 음수값이 들어가도 된다. 이때는 문자열 자료형을 **뒤에서**부터 읽기 시작한다. 위의 예를 들면 alphabet[-1]은 'z'를 반환한다.)
비슷하게 alphabet[1:4]는 'bcd'를 반환했다. 인덱싱과 같은 원리로 문자열의 두 번째, 세 번째, 네 번째 문자를 반환했다.
여기서 주의할 점은 alphabet[4]인 e는 반환되지 않는 다는 것이다. 즉 str[n:m]은 str[n]~str[m-1]까지의 문자를 반환한다.
n대신 빈칸을 넣으면 왼쪽 처음부터 반환하고, m대신 빈칸을 넣으면 오른쪽 끝까지 반환한다. 둘 다 비워두면 모든 문자를 반환한다.

**주의**
String은 List와 다르게 인덱싱 한 문자를 바꿀 수 없다. (replace 함수로 변환 가능하긴 하다.)
```
> a = 'AABA'
> a[2] = 'A'
TypeError: 'str' object does not support item assignment
```

### Formatting
문자열 자료형에 변수를 넣는다고 생각하면 된다. 바로 예제를 확인해보자

```
> a = 'I live on the {}th floor'
> print(a.format(4))
I live on the 4th floor
> print(a.format(5))
I live on the 5th floor
```
이처럼 숫자를 바꿔가면서 출력할 수 있고, 필요 시 중괄호를 여러개 사용할 수 도 있다.
또 {} 안에 ':.4f' 등을 넣어 소수표현이나 공백 채우기 등 작업을 할 수 있다. 

### 문자열 관련 함수
#### count(str)
문자열 중 문자 개수를 반환하는 함수
```
> a = "hobby"  
> a.count('b') 
2
```
#### find(str), index(str)
문자열 중 'str'이 가장 처음 나온 위치를 반환한다. find와 index의 가장 큰 차이점은 문자열 중 찾는 문자가 없으면 find는 -1을 반환하고, index는 오류가 발생한다.
```
> a = 'hello'
> a.find('l')
2
> a.index('l')
2
> a.find('x')
-1
> a.index('x')
ValueError: substring not found
```
#### upper(), lower()
upper()은 소문자를 대문자로, lower()은 대문자를 소문자로 변환해서 반환한다.
(기존 변수에 저장되어있는 문자열은 보존된다.)
```
> a = 'pYthOn'
> a.upper()
'PYTHON'
> a.lower()
'python'
> a
'pYthOn'
```
#### strip()
양 끝의 공백을 지운다. 괄호안에 str을 넣으면 양 끝에 있는 str을 제거한다.
```
> a = '  hello           '
> a.strip()
'hello'
```
```
> a = 'xxhelloxx'
> a.strip('x')
'hello'
```
#### replace(old_str, new_str)
문자열의 old_str을 new_str로 치환해서 반환한다.

```
> a = 'hello'
> a.replace('l','x')
'hexxo'
```

#### split()
괄호 안의 문자를 기준으로 문자열을 나눠서 **List**로 반환한다. 괄호 안이 비어있으면 공백으로 나눈다.
```
> a = "Life is too short"  
> a.split()
['Life', 'is', 'too', 'short']
> b = "a:b:c:d"
> b.split(':')
['a', 'b', 'c', 'd']
```