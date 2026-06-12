# 미니 GPT-2

GPT-2의 핵심 구조(Transformer Decoder)를 PyTorch로 직접 구현한 nanoGPT 규모의 미니 언어모델.
Tiny Shakespeare 데이터를 문자 단위로 학습해 다음 토큰을 예측하고 텍스트를 생성한다.

## 구조

```
config.py      하이퍼파라미터(GPTConfig/TrainConfig) + device 자동 선택
tokenizer.py   CharTokenizer (문자 단위 encode/decode)
data/prepare.py  데이터 다운로드 → train.bin / val.bin / meta.pkl
model.py       CausalSelfAttention, FeedForward, Block, GPT
utils.py       get_batch, estimate_loss, perplexity, 체크포인트 I/O
train.py       학습 루프 + train/val loss 로깅 + 체크포인트
generate.py    체크포인트 로드 → temperature 샘플링 생성
```

## 사용법

```bash
# 1) 환경
pip install -r requirements.txt

# 2) 데이터 준비 (Tiny Shakespeare 다운로드 + 인코딩)
python data/prepare.py

# 3) 학습 (device는 cuda > mps > cpu 자동 선택)
python train.py

# 4) 생성
python generate.py --prompt "ROMEO:" --max_new_tokens 500 --temperature 0.8
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