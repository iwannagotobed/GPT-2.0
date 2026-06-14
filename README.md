# 미니 GPT-2

GPT-2의 핵심 구조(Transformer Decoder)를 PyTorch로 직접 구현한 nanoGPT 규모의 미니 언어모델.
한국어 위키문헌의 공개 고전소설을 문자 단위로 학습해 다음 토큰을 예측하고 텍스트를 생성한다.

## 데이터셋

- 저작권 보호기간이 끝난 김동인·현진건·나도향·최서해·이상의 소설
- 한국어 위키문헌의 `저자:` 역링크와 소설 분류를 이용해 약 100만 자 수집
- 작품 단위 train/validation 분리로 같은 작품의 문장이 양쪽에 섞이는 것을 방지
- 수집한 문서의 URL, revision ID, 문자 수, 분할은 `data/sources.json`에 기록
- 이후 `prepare.py` 실행은 고정 revision을 사용하며, `--refresh`로만 작품을 다시 검색

## 구조

```
config.py      하이퍼파라미터(GPTConfig/TrainConfig) + device 자동 선택
tokenizer.py   CharTokenizer (문자 단위 encode/decode)
data/prepare.py  공개 고전소설 수집·정제 → train.bin / val.bin / meta.pkl
model.py       CausalSelfAttention, FeedForward, Block, GPT
utils.py       get_batch, estimate_loss, perplexity, 체크포인트 I/O
train.py       학습 루프 + train/val loss 로깅 + 체크포인트
generate.py    체크포인트 로드 → temperature 샘플링 생성
```

## 사용법

```bash
# 1) 환경
pip install -r requirements.txt

# 2) 데이터 준비 (고정된 sources.json revision 재현 + 인코딩)
python data/prepare.py

# 작품을 다시 검색해 100만 자 manifest 갱신
python data/prepare.py --refresh

# 3) 학습 (device는 cuda > mps > cpu 자동 선택)
python train.py

# 4) 생성
python generate.py --prompt "그는" --max_new_tokens 500 --temperature 0.8
```

짧은 학습 파이프라인 검증에는 CLI 오버라이드를 사용할 수 있다.

```bash
python train.py --max-iters 10 --eval-interval 5 --eval-iters 2 \
  --checkpoint /tmp/mini-gpt2-smoke.pt --log /tmp/mini-gpt2-smoke.csv
```

## 하이퍼파라미터

기본값은 Colab 무료 티어/CPU에서도 돌도록 작게 설정했다 (`config.py`에서 조정).

| 항목 | 기본값 |
|------|--------|
| n_layer | 4 |
| n_embd | 256 |
| n_head | 4 |
| block_size | 128 |
| batch_size | 32 |
| max_iters | 5000 |
| lr | 3e-4 |

## 평가

- **정량**: train/val loss 곡선(`loss_log.csv`), Perplexity = exp(val loss)
- **정성**: 학습 전/후 생성 텍스트 비교, temperature 변화에 따른 다양성
- **재현성**: `data/sources.json`의 revision ID와 작품 단위 split 검증
