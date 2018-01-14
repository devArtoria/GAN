# WGAN 정리

## 1. WGAN 정리
아직은 잘 안봤지만 거리(?) 의 정의를 새로 바꿔서 분포들의 모습을 측정하는 방법을 다르게 한다는 뜻인것같은데 지난번에 본  KL Divergence 이런거랑 연관되는 부분이겠지. 보니까 이거의 연장이 BEGAN까지 간다던데  
추가바람~

2018/1/14
KL, JS Divergence는 원 데이터와 G 생성 데이터의 확률 분포를 엄격하게 체크하며 이게 GAN의 판별자 학습에 장애가 된다.  
따라서 유연하며 수렴에 포커스를 맞춘 다른 metric 이 필요한 것이다. (사실 정확히 말하면 KL과 JS는 거리의 조건 4가지 중에서 두가지인가 만족 못해서 유사 metric 이라 카더라)  
그리고 그 메트릭이 바로 Wassertein distance.  
Wassertein distance란 어차피 수식은 LaTex 까먹어서 못적는다. 
P, Q에 대한 모든 결합확률분포들 중에서 d(X, Y)의 기댓값을 가장 작게 추정한 값을 의미한다. 그리고 이것을 2차원으로 매핑해 거리를 계산해서 전체 확률분포를 대국적으로 보는 분포수렴한다는것 같네...  
P. S. soft한 metric이 무슨뜻일까... 대국적으로 보는 그걸 말하는건가??



## 2. WGAN 참고자료
1. [Wassertein 거리 설명](https://rosinality.github.io/2017/04/wasserstein-%EA%B1%B0%EB%A6%AC/)
2. [WGAN의 수학 설명](https://www.slideshare.net/ssuser7e10e4/wasserstein-gan-i)
3. [아마도 못읽을 원 논문](https://arxiv.org/abs/1701.07875)
4. 