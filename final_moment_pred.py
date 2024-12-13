import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


# 물체의 형상에 따른 관성모멘트 계산 함수 정의
def calculate_inertia(moment_type, mass, size):
    if moment_type == "sphere":  # 구
        return (2/5) * mass * (size**2)
    elif moment_type == "disk":  # 원판
        return (1/2) * mass * (size**2)
    elif moment_type == "rod_center":  # 막대 (중심을 축으로 회전)
        return (1/12) * mass * (size**2)
    elif moment_type == "rod_end":  # 막대 (끝을 축으로 회전)
        return (1/3) * mass * (size**2)
    
#데이터 생성(gpt에게 랜덤 데이터를 생성하는 방법을 물어봄)
data = []
moment_types = ["sphere", "disk", "rod_center", "rod_end"]

for i in range(1000): #1000개 데이터 생성
    moment_type = np.random.choice(moment_types) #moment_type 리스트에서 하나 선택
    mass = round(random.uniform(0.1, 10.0), 5) #질량 0.1kg~10kg미만 소수점 다섯째자리까지
    size = round(random.uniform(0.1, 2.0), 5) #반지름(막대의 경우 길이) 0.1m~2m미만
    #실행마다 다른 데이터가 생성됨
    
    #각 데이터마다 관성모멘트 계산
    inertia = calculate_inertia(moment_type, mass, size) 

    data.append({"type": moment_type, "mass": mass, "size": size, "inertia": inertia})

inertia_data = pd.DataFrame(data)

#데이터를 확인할 수 있도록 csv 파일로 저장
inertia_data.to_csv("inertia_data.csv", index=False) 

#입력과 출력 분리
x = inertia_data[["mass", "size"]]
y = inertia_data["inertia"]

#학습 데이터와 테스트 데이터로 분리
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=0) 

#knn 모델 생성 및 훈련
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(x_train, y_train)

#테스트 데이터로 예측
prediction = knn.predict(x_test)
print("테스트 세트에 대한 예측값:\n {}".format(prediction))

#결정계수 R^2 출력
print("테스트 세트의 정확도: {:.2f}".format(knn.score(x_test, y_test))) #정확도는 대체로 0.6~0.8 사이 값이 나옴
