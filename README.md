### UST ETRI 스쿨 연구인턴십 과제

- 기간: 2025.01.06 ~ 2025.02.13
- 주제: Harmonic-NAS 기반 멀티모달 네트워크의 하이퍼파라미터 최적화 및 성능 분석

📅 주차별 활동 내용


1주차: Ubuntu 22.04 환경 기반, 딥러닝 환경 설정(CUDA, cuDNN, NVIDIA 드라이버 등등 설치) & Neural Architecture Search 기초 개념 정립

2주차: 선행 논문 연구(MM-NAS, BM-NAS, MFAS 등등), Harmonic-NAS 프레임워크 이해하기

3주차: MM-IMDB, AV-MNIST 등등 멀티모달 데이터셋 기반으로 실험 진행

4,5주차: 데이터 증강 및 하이퍼 파라미터 변경 후 인사이트 도출

👨‍🔬실험 결과(WandB 프레임워크 활용)
1. Sample 1: 데이터 증강만 시킨 것
https://wandb.ai/junylee00-chonnam-national-university/search_algo_harmnas?nw=nwuserjunylee00
2. Sample 2: Search Algorithm 하이퍼 파라미터 Sweeping
https://wandb.ai/junylee00-chonnam-national-university/search_algo_harmnas_with_sweep?nw=nwuserjunylee00
3. Sample 3: 2번 결과 토대로 Best-MM Model => 앞에서 찾은 최적의 신경망 탐색 구조를 평가!!
https://wandb.ai/junylee00-chonnam-national-university/best_mm_model_harmnas?nw=nwuserjunylee00


