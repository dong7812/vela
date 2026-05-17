# Vela

**범용 능동형 AI SDK**  
사용자가 묻기 전에, 데이터가 먼저 말한다.

---

## 왜 능동형 AI인가

ChatGPT를 비롯한 대부분의 생성형 AI는 **반응형**이다.  
사용자가 먼저 물어야 답한다. 사용자가 뭘 물어야 할지 모르면, AI는 침묵한다.

Vela는 반대로 작동한다.  
사용자가 앱을 열지 않아도, 백그라운드에서 데이터를 지켜보다가 **먼저 말을 건다.**

| 도메인 | Vela가 감지하는 것 | Vela가 먼저 하는 것 |
|---|---|---|
| 팬덤 | 며칠째 접속 없음 + 컴백 임박 | "오늘 밤 12시 컴백이에요. 알림 설정할까요?" |
| 소상공인 | 평소보다 매출 40% 하락 | "오늘 매출이 많이 낮아요. 확인해보셨나요?" |
| 헬스케어 | 수면·활동 패턴 3일 연속 변화 | "요즘 수면이 줄었어요. 괜찮으세요?" |
| 대화 AI | 대화 흐름이 같은 자리를 맴돎 | "혹시 이 부분이 막히시나요?" |

도메인은 바꿔도, 구조는 하나다.  
**팬덤은 첫 번째 케이스일 뿐 — Vela는 어떤 도메인에도 붙는 SDK다.**

---

## Table of Contents

- [3-Layer Pipeline](#3-layer-pipeline)
- [설치](#설치)
- [빠른 시작](#빠른-시작)
- [SDK 레퍼런스](#sdk-레퍼런스)
- [새 도메인 추가하기](#새-도메인-추가하기)
- [커스텀 LLM 연결](#커스텀-llm-연결)
- [프로젝트 구조](#프로젝트-구조)
- [참조 논문](#참조-논문)
- [기여하기](#기여하기)
- [라이선스](#라이선스)

---

## 3-Layer Pipeline

모든 도메인이 공유하는 실행 구조. 도메인마다 **Layer 01·02 구현체만 교체**하면 됩니다.

```
입력: current(현재 데이터 스냅샷) + history(과거 데이터)
                    │
          ┌─────────▼──────────┐
          │   Layer 01         │  로컬 연산만. LLM 없음. $0
          │   Signal Detector  │
          │                    │
          │  baseline = compute_baseline(history)
          │  signal   = detect(current, baseline)
          │                    │
          │  signal.score < threshold  ──→  None 반환 (종료)
          │  signal.score ≥ threshold  ──→  Layer 02 진행
          └─────────┬──────────┘
                    │ ~5%만 통과
          ┌─────────▼──────────┐
          │   Layer 02         │  단순 케이스: 조건 분기($0)
          │   Intent           │  복합 케이스: Claude Haiku (선택)
          │   Classifier       │
          │                    │
          │  intent = classify(signal)
          │  → type:   WARNING / OPPORTUNITY / INFO
          │  → intent: NEED_WARNING / NEED_ACTION / NEED_INFO
          └─────────┬──────────┘
                    │
          ┌─────────▼──────────┐
          │   Layer 03         │  Claude Sonnet
          │   Output Generator │
          │                    │
          │  result = generate(signal, intent, domain_prompt)
          │  → message: 사용자가 바로 읽을 수 있는 한두 문장
          │  → reason:  개입 이유
          └─────────┬──────────┘
                    │
              OutputResult
```

`DomainPlugin`은 Layer 01·02 구현체 + 도메인 프롬프트를 묶는 컨테이너입니다.

```python
class DomainPlugin(ABC):
    def get_signal_detector(self)   -> BaseSignalDetector:    ...  # Layer 01
    def get_intent_classifier(self) -> BaseIntentClassifier:  ...  # Layer 02
    def get_output_prompt(self)     -> str:                   ...  # Layer 03 시스템 프롬프트
```

---

## 설치

**요구사항**: Python 3.10+

```bash
git clone https://github.com/dong7812/vela.git
cd vela

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

로컬 LLM 사용 시 [Ollama](https://ollama.com) 추가 필요:

```bash
ollama pull qwen2.5:3b
```

| 모델 | 필요 RAM | 비고 |
|---|---|---|
| `qwen2.5:3b` (기본값) | ~2 GB | 권장 |
| `llama3.2:3b` | ~2 GB | 대안 |
| `gemma2:2b` | ~1.5 GB | 최소 사양 |

Claude API를 사용하면 Ollama 없이도 동작합니다.

---

## 빠른 시작

### Streamlit UI (대화 도메인)

```bash
ollama serve                       # 별도 터미널
streamlit run vela/ui/app.py
```

`http://localhost:8501` 접속. 사이드바에서 파일 업로드 또는 바로 대화 시작.  
Claude API 사용 시: 사이드바 → LLM 설정 → Claude API → API Key 입력.

### SDK (팬덤 도메인)

```python
from vela import VelaAgent, FandomDomainPlugin
from vela.llm.claude import ClaudeLLM

agent = VelaAgent(
    plugin=FandomDomainPlugin(),
    llm=ClaudeLLM(api_key="sk-ant-..."),
)

history = [{"activity_count": 12, "session_hour": 21}] * 14  # 과거 14일

today = {
    "activity_count": 1,
    "content_skip_rate": 0.9,
    "session_hour": 4,
    "comeback_imminent": True,
    "purchase_signal": 0.85,
}

result = agent.run(current=today, history=history)
if result:
    print(result.message)   # "오늘 밤 12시 컴백이에요. 알림 설정할까요?"
    print(result.reason)    # "컴백 임박"
```

---

## SDK 레퍼런스

### 대화 도메인 — VelaAgent

```python
from vela import VelaAgent

agent = VelaAgent()   # 기본값: Ollama

# 문서 로드 → WFC 초기화
agent.load_document("requirements.pdf")

# 비스트리밍 분석
analysis = agent.analyze_document()

# 스트리밍 분석 (UI용) — WFC 초기화는 완료 후 별도 호출
for token in agent.analyze_document_stream():
    print(token, end="", flush=True)
agent.init_wfc_from_document()

# 대화 (비스트리밍)
response, state, decision = agent.chat("질문 내용")
# state    → EXPLORING / DEEPENING / LOOPING / STUCK
# decision → should_intervene, initiative_type, score

if decision.should_intervene:
    if decision.initiative_type.value == "INFORMATION" and agent.get_wfc_next():
        msg = agent.wfc_proactive()                            # WFC 다음 주제 발화
    else:
        msg = agent.prima_intervene(decision.initiative_type)  # ESConv 전략 발화
else:
    questions = agent.suggest_questions()                      # 선제 질문 3개 fallback

# 대화 (스트리밍 — UI용)
messages, system, state = agent.prepare_chat(user_input)
accumulated = []
for token in agent._llm.chat_stream(messages, system):
    accumulated.append(token)
    print(token, end="", flush=True)
decision = agent.finalize_chat("".join(accumulated), state)

# 스트리밍 개입 발화
for token in agent.prima_intervene_stream(decision.initiative_type):
    print(token, end="", flush=True)

for token in agent.wfc_proactive_stream():
    print(token, end="", flush=True)

# WFC 상태 조회
cells    = agent.get_wfc_cells()   # 전체 셀 목록
next_one = agent.get_wfc_next()    # entropy 최저 셀

# PRIMA 피드백 통계
summary = agent.feedback.get_summary()
# {"REFRAME": {"count": 5, "success_rate": 0.80, ...}, ...}

agent.reset()  # 대화 초기화
```

### 범용 파이프라인 — DomainPlugin 주입

```python
from vela import VelaAgent, FandomDomainPlugin

agent = VelaAgent(plugin=FandomDomainPlugin(), llm=my_llm)
result = agent.run(current=today_data, history=past_14_days)
# result = None          → 개입 불필요 (score < threshold, LLM 비용 $0)
# result = OutputResult  → message / reason 담긴 인사이트
```

`VelaPipeline`을 직접 사용하면 `VelaAgent` 없이도 실행할 수 있습니다.

```python
from vela import VelaPipeline, FandomDomainPlugin

pipeline = VelaPipeline(plugin=FandomDomainPlugin(), llm=my_llm)
result = pipeline.run(current=today_data, history=history)
```

### 백그라운드 실행 — VelaScheduler

```python
import asyncio
from vela import VelaScheduler, FandomDomainPlugin
from vela.llm.claude import ClaudeLLM

scheduler = VelaScheduler(
    plugin=FandomDomainPlugin(),
    llm=ClaudeLLM(api_key="sk-ant-..."),
    interval_minutes=30,
)

async def fetch_users() -> dict[str, dict]:
    return {"user_001": {...}, "user_002": {...}}

asyncio.run(scheduler.run_loop(user_data_fn=fetch_users))
```

---

## 새 도메인 추가하기

세 가지만 구현합니다: `BaseSignalDetector` → `BaseIntentClassifier` → `DomainPlugin`.

### Step 1 — Layer 01: Signal Detector

```python
from vela import BaseSignalDetector, SignalResult

class BizOwnerSignalDetector(BaseSignalDetector):
    THRESHOLD = 0.55

    def compute_baseline(self, history: list[dict]) -> dict:
        # LLM 호출 금지. 로컬 연산만.
        avg = sum(h["daily_revenue"] for h in history) / len(history)
        return {"avg_revenue": avg}

    def detect(self, current: dict, baseline: dict) -> SignalResult:
        # LLM 호출 금지. 로컬 연산만.
        dev = abs(current["daily_revenue"] - baseline["avg_revenue"])
        score = min(1.0, dev / max(1, baseline["avg_revenue"]))
        return SignalResult(
            score=score,
            should_act=score >= self.THRESHOLD,
            trigger=f"revenue_deviation score={score:.2f}",
            raw={"deviation": dev, **current},
        )
```

### Step 2 — Layer 02: Intent Classifier

```python
from vela import BaseIntentClassifier, IntentResult, SignalResult

class BizOwnerIntentClassifier(BaseIntentClassifier):
    def classify(self, signal: SignalResult) -> IntentResult:
        raw = signal.raw

        # 단순 케이스 — LLM 없음
        if raw["daily_revenue"] < raw.get("break_even", float("inf")):
            return IntentResult(
                type="WARNING", intent="NEED_WARNING",
                urgency=0.90, confidence=0.85, reason="손익분기점 미달",
            )
        if raw.get("top_item_out_of_stock"):
            return IntentResult(
                type="OPPORTUNITY", intent="NEED_ACTION",
                urgency=0.75, confidence=0.90, reason="베스트셀러 재고 부족",
            )

        # 복합 케이스 — 기본 분류로 fallback (LLM 연동은 도메인에서 직접 구현)
        # 이유: 매출 감소 + 재고 부족 + 외부 이벤트 동시 발생 → 조건 분기로 커버 불가
        return IntentResult(
            type="INFO", intent="NEED_INFO",
            urgency=signal.score, confidence=0.60, reason="복합 신호 감지",
        )
```

### Step 3 — DomainPlugin 조합

```python
from vela.domain.base import DomainPlugin

class BizOwnerDomainPlugin(DomainPlugin):
    @property
    def domain_name(self) -> str:
        return "bizowner"

    def get_signal_detector(self):
        return BizOwnerSignalDetector()

    def get_intent_classifier(self):
        return BizOwnerIntentClassifier()

    def get_output_prompt(self) -> str:
        return "소상공인 맥락에서 오늘 바로 실행할 수 있는 조치를 한두 문장으로."

# 사용
agent = VelaAgent(plugin=BizOwnerDomainPlugin(), llm=my_llm)
result = agent.run(current=today_sales, history=past_30_days)
```

---

## 커스텀 LLM 연결

```python
from typing import Iterator
from vela.llm.base import BaseLLM

class MyLLM(BaseLLM):
    def chat(self, messages: list[dict], system: str = "") -> str: ...
    def chat_stream(self, messages: list[dict], system: str = "") -> Iterator[str]: ...
    def is_available(self) -> bool: ...

agent = VelaAgent(llm=MyLLM())
```

---

## 프로젝트 구조

```
vela/
├── agent.py                  # VelaAgent — 진입점, DomainPlugin 주입
├── core/
│   ├── signal_detector.py    # Layer 01 추상 인터페이스 + SignalResult
│   ├── intent_classifier.py  # Layer 02 추상 인터페이스 + IntentResult
│   ├── output_generator.py   # Layer 03 LLM 출력 생성 + OutputResult
│   ├── pipeline.py           # VelaPipeline — 3레이어 실행기
│   ├── scheduler.py          # VelaScheduler — 백그라운드 주기 실행
│   ├── feedback.py           # FeedbackLogger — JSONL 기반 outcome 수집
│   ├── prima.py              # PRIMAEngine — 개입 점수 계산 + ESConv 전략 선택
│   ├── wfc.py                # ConversationWFC — entropy 기반 대화 공간 관리
│   ├── state.py              # StateDetector — 코사인 유사도 기반 상태 감지
│   ├── embedder.py           # sentence-transformers 임베딩 + 유사도
│   └── context.py            # ContextWindow — 대화 히스토리 관리
├── domain/
│   ├── base.py               # DomainPlugin ABC
│   ├── conversation/
│   │   ├── plugin.py         # ConversationDomainPlugin
│   │   │                     #   PrimaSignalDetector: StateDetector+WFC+PRIMA → SignalResult
│   │   │                     #   WfcIntentClassifier: InitiativeType → IntentResult
│   │   ├── esconv.py         # ESConv 8가지 전략 프롬프트
│   │   ├── prima.py          # core.prima re-export
│   │   └── wfc.py            # core.wfc re-export
│   └── fandom/
│       ├── plugin.py         # FandomDomainPlugin
│       ├── detector.py       # FanSignalDetector — 행동 편차 점수화
│       └── classifier.py     # FanIntentClassifier — 조건 분기 의도 분류
├── llm/
│   ├── base.py               # BaseLLM ABC
│   ├── ollama.py             # Ollama 구현체
│   └── claude.py             # Claude API 구현체
├── rag/
│   ├── loader.py             # 문서 청킹 (txt, md, pdf)
│   └── retriever.py          # ChromaDB 벡터 저장 + 검색
└── ui/
    └── app.py                # Streamlit UI (대화 도메인)
```

**로컬 데이터** (`.gitignore` 제외):

| 경로 | 내용 |
|---|---|
| `.vela_db/` | ChromaDB 벡터 인덱스 |
| `.vela_feedback.jsonl` | PRIMA 개입 outcome 누적 로그 |

---

## 참조 논문

| 논문 | 적용 |
|---|---|
| Horvitz (1999) *Mixed-Initiative Interaction*, CHI | PRIMA 임계값 — E[utility(act)] > E[utility(wait)] |
| Liu et al. (2021) *ESConv*, ACL | InitiativeType 8가지 전략 + 신호 가중치 |
| Deng et al. (2023) *Survey on Proactive Dialogue Systems*, IJCAI | 3모드 신호-전략 매핑 구조 |
| Deng, Liao et al. (2023) *Prompting LLMs for Proactive Dialogues*, EMNLP | 전략별 시스템 프롬프트 분리 |
| Bohus & Rudnicky (2005) *Error Handling in Conversational Systems* | stagnation 일시 vs. 지속 구분 (2턴 기준) |
| Murray & Levesque (2003) *Expressing Key Points and Managing Convergence* | 응답 길이 30% 증가 = 참여도 회복 지표 |
| Richards (1987) *Type-Token Ratio* | engagement_decay — TTR 기반 어휘 다양성 |
| Sacks, Schegloff & Jefferson (1974) *Turn-taking in Conversation* | confusion — 의문형 위치 감지 |
| Reimers & Gurevych (2019) *Sentence-BERT*, EMNLP | WFC 셀 품질 필터 (cosine < 0.15) |

---

## 기여하기

```bash
git clone https://github.com/dong7812/vela.git
cd vela
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

**우선순위 항목**

- [ ] 새 도메인 구현 — `bizowner`, `health`, `ecommerce` 등
- [ ] `FanIntentClassifier` 복합 케이스 Claude Haiku 연동
- [ ] PRIMA 가중치 자동 튜닝 — `.vela_feedback.jsonl` 기반 Bayesian 최적화
- [ ] VelaScheduler 푸시 알림 연동 (FCM, APNs)
- [ ] OpenAI / Gemini / Bedrock LLM 구현체
- [ ] Layer 01·02 단위 테스트

**코드 규칙**

- 타입 힌트 필수
- Layer 01 구현에서 LLM 호출 금지
- Layer 02 LLM 호출 시 이유 주석 필수 (`# 이유: ...`)
- 새 LLM은 반드시 `BaseLLM` 상속

---

## 라이선스

MIT © [dong7812](https://github.com/dong7812)
