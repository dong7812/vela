# Vela

**AI SDK that acts before you ask**

로컬 LLM(Ollama) 기반으로 대화 상태를 실시간 감지하고, 사용자가 묻기 전에 먼저 대화를 이끄는 오픈소스 Python SDK.

---

## Table of Contents

- [기존 챗봇과의 차이](#기존-챗봇과의-차이)
- [동작 원리](#동작-원리)
  - [PRIMA — 개입 판단 엔진](#prima--개입-판단-엔진)
  - [WFC — 대화 공간 관리](#wfc--대화-공간-관리)
  - [대화 상태 감지](#대화-상태-감지)
  - [피드백 루프](#피드백-루프)
- [설치](#설치)
- [빠른 시작](#빠른-시작)
- [SDK 레퍼런스](#sdk-레퍼런스)
- [커스텀 LLM 연결](#커스텀-llm-연결)
- [프로젝트 구조](#프로젝트-구조)
- [참조 논문](#참조-논문)
- [기여하기](#기여하기)
- [라이선스](#라이선스)

---

## 기존 챗봇과의 차이

| | ChatGPT / Claude | **Vela** |
|---|---|---|
| 대화 시작 | 사용자가 먼저 | 문서 로드 시 자동 분석 후 에이전트가 먼저 발화 |
| 개입 여부 | 없음 | PRIMA 점수가 임계값 초과 시에만 선택적 개입 |
| 개입 전략 | 없음 | ESConv 8가지 전략 중 신호 패턴에 맞는 것 선택 |
| 논의 흐름 | 사용자 주도 | WFC로 대화 공간 구성 후 entropy 순서로 주제 유도 |
| 다음 질문 | 없음 | 개입 임계값 미달 시 선제 질문 3개 자동 생성 |
| 효과 측정 | 없음 | 메시지 길이 변화 + 상태 전이로 암묵적 outcome 수집 |

```
[사용자] 로컬 스토리지를 활용하려고 합니다.

[🌊 WFC · 이메일 전송 방식]
  그렇다면 연락 폼 이메일 전송 방식을 살펴봐야 할 것 같습니다.
  백엔드 없이 이메일을 보내려면 외부 서비스가 필요한데,
  생각해두신 게 있나요?              ← INFORMATION + WFC 연계

[사용자] 모르겠어요...

[🪞 PRIMA · REFLECTION]
  지금 어느 부분에서 막히신 건지 같이 정리해봐요.
  백엔드가 없다는 제약 안에서 선택지가 좁혀지는 것 때문에
  막히신 건가요?                     ← stagnation 감지 → REFLECTION
```

---

## 동작 원리

### PRIMA — 개입 판단 엔진

**Proactive Response with Initiative and Multi-signal Analysis**

매 사용자 턴마다 5가지 신호를 계산해 개입 점수를 구합니다.
LLM 호출 없이 임계값(0.38)을 초과할 때만 전략을 선택하고 발화합니다.

```
score = 0.35 × stagnation       (대화 맴돔 정도, state 기반)
      + 0.25 × confusion        (혼란/막힘, 의문형 구분 감지)
      + 0.20 × coverage_gap     (WFC 미논의 비율)
      + 0.12 × engagement_decay (TTR 기반 참여도 하락)
      + 0.08 × initiative_debt  (연속 반응 횟수 누적)

score ≥ 0.38  → 개입
score < 0.38  → 선제 질문 3개 생성 (fallback)
```

**가중치 근거**: ESConv 테이블 4 ablation (Liu et al. 2021) — stagnation이 단일 예측 변수 중 가장 강함.

**개입 전략 선택 — ESConv 8가지 전략 (Liu et al., ACL 2021)**

Hill의 Helping Skills Theory에서 실증적으로 도출된 분류 체계를 적용합니다.
Deng et al. (IJCAI 2023)의 3가지 능동 대화 모드로 신호-전략을 매핑합니다.

| 모드 | 전략 | 트리거 조건 |
|---|---|---|
| Clarification | `QUESTION` | initiative_debt 누적 (≥ 0.6) |
| Clarification | `RESTATEMENT` | confusion ≥ 0.5 (하드 트리거) |
| Target-guided | `REFLECTION` | stagnation 0.3–0.7, 2턴 지속 |
| Target-guided | `INFORMATION` | coverage_gap ≥ 0.5, WFC 셀 연계 |
| Non-collaborative | `AFFIRMATION` | engagement 소폭 하락 + debt ≥ 0.6 |
| Non-collaborative | `SUGGESTION` | engagement ≥ 0.5 급락 |
| Non-collaborative | `REFRAME` | stagnation ≥ 1.0 (STUCK, 즉시 발화) |
| Non-collaborative | `SELF_DISCLOSURE` | 그 외 |

**하드 트리거**: STUCK(stagnation=1.0) → REFRAME 즉시 발화. LOOPING(stagnation=0.7) → 2턴 지속 후 REFLECTION (Bohus & Rudnicky 2005).

---

### WFC — 대화 공간 관리

문서 로드 또는 첫 대화 후 LLM이 핵심 논의 주제 셀을 생성합니다.
Wave Function Collapse 알고리즘으로 어떤 주제를 먼저 꺼낼지 결정합니다.

- 각 셀은 `entropy` 값을 가집니다. 낮을수록 먼저 논의할 주제
- 한 셀이 논의되면 관련 셀의 entropy가 감소합니다 (constraint propagation)
- 코사인 유사도 < 0.15인 셀은 LLM 환각으로 판단해 자동 제거합니다 (Reimers & Gurevych 2019)
- PRIMA가 `INFORMATION` 전략을 선택하면 WFC 다음 셀을 꺼내 발화합니다

```
대화 공간 (2/6 탐색됨)
✅  ~~백엔드 구조~~
✅  ~~이메일 전송 방식~~
▶️  포트폴리오 접근 범위   ← entropy 최저, 다음 주제
○   상태 관리 전략
○   배포 옵션
○   보안 고려사항
```

---

### 대화 상태 감지

최근 N턴의 사용자 발화를 임베딩해 코사인 유사도로 상태를 판단합니다.
PRIMA의 stagnation 신호 입력값으로 사용됩니다.

| 상태 | 유사도 범위 | stagnation | 의미 |
|---|---|---|---|
| `EXPLORING` | < 0.60 | 0.0 | 새로운 주제 탐색 중 |
| `DEEPENING` | 0.60–0.85 | 0.3 | 주제가 깊어지는 중 |
| `LOOPING` | 0.85–0.95 | 0.7 | 같은 자리 맴돔 → 개입 필요 |
| `STUCK` | ≥ 0.95 | 1.0 | 완전히 막힘 → 즉시 개입 |

---

### 피드백 루프

Vela를 실행하면 `.vela_feedback.jsonl`에 PRIMA 개입 결과가 자동으로 누적됩니다.
데이터 수집 흐름:

```
PRIMA 개입 발생
  → intervention 레코드 저장 (전략 타입, 점수, 신호값, 상태)

다음 사용자 턴
  → outcome 레코드 자동 저장 (메시지 길이 변화, 상태 전이)

👍/👎 클릭
  → rating 레코드 즉시 저장
```

암묵적 성공 판정 기준 (Murray & Levesque 2003):
- `length_ratio > 1.3` — 다음 메시지가 30% 이상 길어짐 (참여도 회복)
- `state_improved` — LOOPING/STUCK → EXPLORING/DEEPENING 전이
- `explicit_rating == 1` — 사용자 👍 클릭

사이드바 피드백 패널에서 전략 타입별 성공률을 실시간으로 확인할 수 있습니다.

---

## 설치

**요구사항**: Python 3.10+, [Ollama](https://ollama.com)

```bash
git clone https://github.com/dong7812/vela.git
cd vela

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Ollama 모델 준비 (최초 1회)
ollama pull qwen2.5:3b
```

**RAM 가이드**

| 모델 | 필요 RAM | 비고 |
|---|---|---|
| `qwen2.5:3b` (기본값) | ~2 GB | 권장 |
| `llama3.2:3b` | ~2 GB | 대안 |
| `gemma2:2b` | ~1.5 GB | 최소 사양 |

---

## 빠른 시작

```bash
# Ollama 서버 시작 (별도 터미널)
ollama serve

# Streamlit UI 실행
streamlit run vela/ui/app.py
```

브라우저에서 `http://localhost:8501` 접속. 사이드바에서 파일을 업로드하거나 바로 대화를 시작합니다.

**Claude API 사용 시**: 사이드바 → LLM 설정 → Claude API 선택 후 API Key 입력.

---

## SDK 레퍼런스

### VelaAgent

```python
from vela import VelaAgent

agent = VelaAgent()

# 문서 로드 + 자동 분석 + WFC 대화 공간 초기화
agent.load_document("requirements.pdf")
analysis = agent.analyze_document()
print(analysis)

# 대화 (비스트리밍)
response, state, decision = agent.chat("질문 내용")

print(f"상태: {state}")                        # EXPLORING / DEEPENING / LOOPING / STUCK
print(f"PRIMA 점수: {decision.score:.2f}")
print(f"개입 여부: {decision.should_intervene}")

if decision.should_intervene:
    if decision.initiative_type.value == "INFORMATION" and agent.get_wfc_next():
        msg = agent.wfc_proactive()              # WFC 다음 주제 발화
    else:
        msg = agent.prima_intervene(decision.initiative_type)   # ESConv 전략 발화
else:
    questions = agent.suggest_questions()        # 선제 질문 3개 (fallback)
```

### 스트리밍 (Streamlit / UI 연동)

```python
# prepare_chat → stream → finalize_chat 순서로 호출
messages, system, state = agent.prepare_chat(user_input)

for token in agent._llm.chat_stream(messages, system):
    print(token, end="", flush=True)

decision = agent.finalize_chat(full_response, state)

if decision.should_intervene:
    for token in agent.prima_intervene_stream(decision.initiative_type):
        print(token, end="", flush=True)
```

### WFC 셀 조회

```python
cells = agent.get_wfc_cells()          # 전체 셀 목록
next_cell = agent.get_wfc_next()       # 다음 논의할 셀 (entropy 최저)

for cell in cells:
    print(cell.topic, cell.state, cell.entropy)
```

### 피드백 통계

```python
summary = agent.feedback.get_summary()
# {
#   "REFRAME":    {"count": 5, "success_rate": 0.80, "avg_length_ratio": 1.52},
#   "REFLECTION": {"count": 3, "success_rate": 0.67, "avg_length_ratio": 1.21},
#   ...
# }
```

---

## 커스텀 LLM 연결

`BaseLLM`을 상속해 `chat()`, `chat_stream()`, `is_available()`을 구현합니다.

```python
from typing import Iterator
from vela.llm.base import BaseLLM

class MyLLM(BaseLLM):
    def chat(self, messages: list[dict], system: str = "") -> str:
        # 동기 응답 반환
        ...

    def chat_stream(self, messages: list[dict], system: str = "") -> Iterator[str]:
        # 토큰 단위로 yield
        ...

    def is_available(self) -> bool:
        ...

agent = VelaAgent(llm=MyLLM())
```

OpenAI, Gemini 등 어떤 LLM이든 위 인터페이스만 맞추면 연결됩니다.

---

## 프로젝트 구조

```
vela/
├── agent.py              # VelaAgent — 전체 파이프라인 진입점
├── core/
│   ├── prima.py          # PRIMA 엔진 (개입 판단 + ESConv 전략 선택)
│   ├── wfc.py            # WFC 엔진 (ConversationWFC, ConversationCell)
│   ├── feedback.py       # FeedbackLogger (JSONL 기반 자동 outcome 수집)
│   ├── embedder.py       # 임베딩 + 코사인 유사도 (sentence-transformers)
│   ├── context.py        # 대화 컨텍스트 윈도우 관리
│   └── state.py          # 대화 상태 감지 (StateDetector)
├── rag/
│   ├── loader.py         # 문서 로드 + 청킹 (txt, md, pdf)
│   └── retriever.py      # ChromaDB 로컬 벡터 저장 + 검색
├── llm/
│   ├── base.py           # BaseLLM 추상 인터페이스
│   ├── ollama.py         # Ollama 구현체
│   └── claude.py         # Claude API 구현체
└── ui/
    └── app.py            # Streamlit UI
```

**로컬 데이터** (`.gitignore`로 제외):

| 경로 | 내용 |
|---|---|
| `.vela_db/` | ChromaDB 벡터 인덱스 |
| `.vela_feedback.jsonl` | PRIMA 개입 outcome 누적 로그 |

---

## 참조 논문

| 논문 | 적용 |
|---|---|
| Horvitz (1999) *Mixed-Initiative Interaction*, CHI | PRIMA 개입 임계값 — E[utility(act)] > E[utility(wait)] |
| Liu et al. (2021) *ESConv*, ACL | InitiativeType 8가지 전략 분류 체계 |
| Deng et al. (2023) *Survey on Proactive Dialogue Systems*, IJCAI | 3모드 신호-전략 매핑 구조 |
| Deng, Liao et al. (2023) *Prompting LLMs for Proactive Dialogues*, EMNLP | 전략별 시스템 프롬프트 분리 설계 근거 |
| Bohus & Rudnicky (2005) *Error Handling in Conversational Systems* | 일시적 stagnation vs. 지속 stagnation 구분 |
| Murray & Levesque (2003) *Expressing key points and managing convergence* | 응답 길이 30% 증가 = 참여도 회복 지표 |
| Richards (1987) *Type-Token Ratio* | engagement_decay 신호 — TTR 기반 어휘 다양성 |
| Sacks, Schegloff & Jefferson (1974) *Turn-taking in Conversation* | confusion 신호 — 의문형 vs. 평서형 구분 |
| Reimers & Gurevych (2019) *Sentence-BERT*, EMNLP | WFC 셀 품질 필터 임계값 (cosine < 0.15) |

---

## 기여하기

PR과 이슈 모두 환영합니다.

```bash
# 개발 환경 셋업
git clone https://github.com/dong7812/vela.git
cd vela
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

**우선순위 높은 기여 항목**

- [ ] PRIMA 가중치 자동 튜닝 — `.vela_feedback.jsonl` 데이터 기반 RL 또는 Bayesian 최적화
- [ ] WFC entropy 학습 기반 초기화 — 대화 이력으로부터 우선순위 학습
- [ ] OpenAI / Gemini LLM 구현체 추가
- [ ] 영어 문서화 및 i18n 지원
- [ ] PRIMA 신호 단위 테스트

**코드 컨벤션**

- 타입 힌트 필수
- 새 신호/상수에는 근거 논문 출처 주석 추가
- 새 LLM 추가 시 반드시 `BaseLLM` 상속

---

## 라이선스

MIT © [dong7812](https://github.com/dong7812)
