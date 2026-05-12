# Vela

**실제 데이터 기반 능동형 AI SDK**  
사용자가 묻기 전에, 데이터가 먼저 말한다.

```
기존 Vela  →  감정/대화 패턴 감지 (주관적, 수치화 어려움)
Vela v3    →  실제 데이터 편차 감지 (객관적, 수치화 가능)
```

---

## Table of Contents

- [기존 SDK와의 차이](#기존-sdk와의-차이)
- [3-Layer Pipeline](#3-layer-pipeline)
- [대화 도메인 — 실제 흐름](#대화-도메인--실제-흐름)
- [팬덤 도메인 — 실제 흐름](#팬덤-도메인--실제-흐름)
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

## 기존 SDK와의 차이

| | LangChain / LlamaIndex | **Vela v3** |
|---|---|---|
| 개입 방식 | 사용자 호출 시에만 반응 | 데이터 편차 감지 시 능동 개입 |
| 적용 범위 | 고정된 RAG/Agent 파이프라인 | DomainPlugin으로 임의 도메인 확장 |
| LLM 비용 | 모든 요청 LLM 호출 | Layer 01(로컬)에서 ~95% 차단, LLM은 마지막 수단 |
| 도메인 신호 | 텍스트 쿼리만 | 행동 데이터·수치 편차·시계열 이상 등 |
| 대화 도메인 | 지원 없음 | PRIMA + WFC + ESConv 내장 |
| 백그라운드 | 없음 | VelaScheduler — 사용자가 앱을 열지 않아도 실행 |

---

## 3-Layer Pipeline

모든 도메인이 공유하는 실행 구조입니다. 도메인마다 **Layer 01·02 구현체만 교체**하면 됩니다.

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
          │   Intent           │  복합 케이스: Claude Haiku
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

**`DomainPlugin`** 은 Layer 01·02 구현체 + 도메인 프롬프트를 묶는 컨테이너입니다.

```python
class DomainPlugin(ABC):
    def get_signal_detector(self)  -> BaseSignalDetector:   ...  # Layer 01
    def get_intent_classifier(self) -> BaseIntentClassifier: ...  # Layer 02
    def get_output_prompt(self)    -> str:                   ...  # Layer 03 시스템 프롬프트
```

---

## 대화 도메인 — 실제 흐름

대화 도메인은 **WFC + PRIMA + ESConv** 세 엔진이 협력합니다.  
사용자 발화가 들어올 때마다 아래 순서로 실행됩니다.

```
사용자: "로컬 스토리지를 쓰려는데 어떻게 하면 될까요?"
                    │
    ┌───────────────▼──────────────────────────────┐
    │  Layer 01 · PrimaSignalDetector              │
    │                                              │
    │  ① 대화 상태 감지 (StateDetector)            │
    │     최근 3턴 발화를 임베딩 → 코사인 유사도   │
    │     avg_similarity = 0.72 → DEEPENING        │
    │     stagnation = 0.3                         │
    │                                              │
    │  ② WFC coverage_gap 계산                     │
    │     전체 셀 5개 중 논의된 셀 1개             │
    │     coverage_gap = (5-1)/5 = 0.80            │
    │                                              │
    │  ③ 기타 신호                                 │
    │     confusion        = 0.0  (의문형 없음)    │
    │     engagement_decay = 0.15 (발화 충분)      │
    │     initiative_debt  = 0.40 (3턴째 반응 중)  │
    │                                              │
    │  ④ PRIMA 점수 계산                           │
    │     score = 0.35×0.3 + 0.25×0.0             │
    │           + 0.20×0.80 + 0.12×0.15           │
    │           + 0.08×0.40                       │
    │           = 0.105+0+0.16+0.018+0.032 = 0.315│
    │                                              │
    │     0.315 < 0.38 (threshold)                 │
    │     → should_act = False → Layer 02 진행 안함│
    │     → fallback: 선제 질문 3개 생성           │
    └──────────────────────────────────────────────┘

사용자: "모르겠어요... 왜 이게 이렇게 복잡하죠?"
                    │
    ┌───────────────▼──────────────────────────────┐
    │  Layer 01 · PrimaSignalDetector              │
    │                                              │
    │  ① 대화 상태: avg_similarity = 0.91 → LOOPING│
    │     stagnation = 0.7                         │
    │                                              │
    │  ② WFC coverage_gap = 0.80 (여전히 4개 남음) │
    │                                              │
    │  ③ confusion = 0.7  ("왜" + "?" 감지)        │
    │     engagement_decay = 0.55 (짧은 발화)      │
    │     initiative_debt  = 0.80 (4턴째)          │
    │                                              │
    │  ④ score = 0.35×0.7 + 0.25×0.7              │
    │          + 0.20×0.80 + 0.12×0.55            │
    │          + 0.08×0.80                        │
    │          = 0.245+0.175+0.16+0.066+0.064     │
    │          = 0.710                            │
    │                                              │
    │     0.710 ≥ 0.38 → should_act = True         │
    └───────────────┬──────────────────────────────┘
                    │
    ┌───────────────▼──────────────────────────────┐
    │  Layer 02 · WfcIntentClassifier              │
    │                                              │
    │  PRIMA 결정: confusion=0.7 → RESTATEMENT     │
    │  (confusion ≥ 0.5 하드 트리거)               │
    │                                              │
    │  → type=INFO, intent=NEED_INFO, urgency=0.50 │
    └───────────────┬──────────────────────────────┘
                    │
    ┌───────────────▼──────────────────────────────┐
    │  Layer 03 · OutputGenerator (LLM)            │
    │                                              │
    │  system: ESConv RESTATEMENT 프롬프트          │
    │  → "제가 이해한 바로는, 백엔드 없이 이메일을  │
    │     보내는 방법을 찾고 계신 거죠?            │
    │     어떤 부분이 제일 막히시나요?"            │
    └──────────────────────────────────────────────┘
```

### PRIMA가 INFORMATION을 선택하면 WFC가 개입

PRIMA가 `INFORMATION` 전략을 선택하고 미논의 WFC 셀이 남아 있으면,  
`VelaAgent`는 `wfc_proactive()`를 호출해 entropy 최저 셀을 꺼냅니다.

```
PRIMA → INFORMATION + WFC 셀 남음
                    │
    WFC: 현재 대화 공간 (1/5 논의됨)
    ✅  ~~백엔드 구조~~
    ▶️  이메일 전송 방식  ← entropy=0.12 (최저, 다음 주제)
    ○   포트폴리오 접근 범위
    ○   상태 관리 전략
    ○   보안 고려사항
                    │
    Layer 03: "이메일 전송을 살펴봐야 할 것 같아요.
              백엔드 없이 보내려면 외부 서비스가 필요한데,
              생각해두신 게 있나요?"
```

### 대화 상태 → stagnation 값

| 상태 | 유사도 범위 | stagnation | PRIMA 반응 |
|---|---|---|---|
| `EXPLORING` | < 0.60 | 0.0 | 개입 없음 |
| `DEEPENING` | 0.60–0.85 | 0.3 | coverage_gap 높으면 개입 가능 |
| `LOOPING` | 0.85–0.95 | 0.7 | 2턴 지속 시 REFLECTION |
| `STUCK` | ≥ 0.95 | 1.0 | 즉시 REFRAME |

### ESConv 전략 선택 기준

| 전략 | 조건 |
|---|---|
| `REFRAME` | stagnation ≥ 1.0 (STUCK 즉시 발화) |
| `REFLECTION` | stagnation ≥ 0.7, 2턴 연속 (Bohus & Rudnicky 2005) |
| `RESTATEMENT` | confusion ≥ 0.5 하드 트리거 |
| `INFORMATION` | coverage_gap ≥ 0.5 + WFC 셀 연계 |
| `SUGGESTION` | engagement_decay ≥ 0.5 급락 |
| `AFFIRMATION` | engagement 소폭 하락 + debt ≥ 0.6 |
| `QUESTION` | initiative_debt ≥ 0.6 |
| `SELF_DISCLOSURE` | 그 외 soft-score 통과 케이스 |

---

## 팬덤 도메인 — 실제 흐름

```
사용자 "팬_A"의 오늘 행동 데이터:
  activity_count   = 1   (평소 12회)
  content_skip_rate = 0.9 (공식 영상 90% 건너뜀)
  session_hour     = 4   (평소 새벽 2시, 오늘 새벽 4시)
  comeback_imminent = True
  purchase_signal  = 0.85
                    │
    ┌───────────────▼──────────────────────────────┐
    │  Layer 01 · FanSignalDetector                │
    │                                              │
    │  baseline (과거 14일 평균):                   │
    │    avg_activity    = 12.0                    │
    │    avg_session_hour = 2.0                    │
    │                                              │
    │  편차 계산:                                   │
    │    activity_deviation  = |1-12|/12  = 0.917  │
    │    content_skip_rate   =            0.900    │
    │    session_time_shift  = |4-2|/12   = 0.167  │
    │    purchase_signal     =            0.850    │
    │                                              │
    │  score = 0.40×0.917 + 0.30×0.900            │
    │        + 0.20×0.167 + 0.10×0.850            │
    │        = 0.367+0.270+0.033+0.085 = 0.755    │
    │                                              │
    │  0.755 ≥ 0.50 (threshold) → should_act=True  │
    └───────────────┬──────────────────────────────┘
                    │
    ┌───────────────▼──────────────────────────────┐
    │  Layer 02 · FanIntentClassifier              │
    │                                              │
    │  comeback_imminent = True                    │
    │  → 첫 번째 조건 즉시 매칭 (LLM 없음)         │
    │  → OPPORTUNITY / NEED_ACTION / urgency=0.95  │
    └───────────────┬──────────────────────────────┘
                    │
    ┌───────────────▼──────────────────────────────┐
    │  Layer 03 · OutputGenerator (LLM)            │
    │                                              │
    │  system: FandomDomainPlugin.get_output_prompt│
    │  → "오늘 밤 12시 컴백이에요. 알림 설정할까요?"│
    └──────────────────────────────────────────────┘
```

Layer 02 조건 분기 순서 (위에서 매칭되면 LLM 호출 없이 즉시 반환):

```python
if comeback_imminent:                                 → OPPORTUNITY urgency=0.95
if activity_deviation > 0.7 and days_inactive >= 3:  → WARNING     urgency=0.85
if purchase_signal > 0.6:                            → OPPORTUNITY urgency=0.80
if is_new_fan:                                       → INFO        urgency=0.50
# 위 조건 모두 미해당 → 복합 케이스 → Claude Haiku 호출
```

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

# 문서 로드 → 자동 분석 → WFC 초기화
agent.load_document("requirements.pdf")
analysis = agent.analyze_document()

# 대화 (비스트리밍)
response, state, decision = agent.chat("질문 내용")
# state    → EXPLORING / DEEPENING / LOOPING / STUCK
# decision → should_intervene, initiative_type, score, signals

if decision.should_intervene:
    if decision.initiative_type.value == "INFORMATION" and agent.get_wfc_next():
        msg = agent.wfc_proactive()                        # WFC 다음 주제 발화
    else:
        msg = agent.prima_intervene(decision.initiative_type)  # ESConv 전략 발화
else:
    questions = agent.suggest_questions()                  # 선제 질문 3개 fallback

# 스트리밍 (UI 연동)
messages, system, state = agent.prepare_chat(user_input)
for token in agent._llm.chat_stream(messages, system):
    print(token, end="", flush=True)
decision = agent.finalize_chat(full_response, state)

# WFC 상태 조회
cells    = agent.get_wfc_cells()   # 전체 셀 목록
next_one = agent.get_wfc_next()    # entropy 최저 셀

# PRIMA 피드백 통계
summary = agent.feedback.get_summary()
# {"REFRAME": {"count": 5, "success_rate": 0.80, "avg_length_ratio": 1.52}, ...}
```

### 범용 파이프라인 — DomainPlugin 주입

`VelaAgent(plugin=...)` 로 어떤 도메인이든 같은 인터페이스로 실행합니다.

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

사용자가 앱을 열지 않아도 주기적으로 파이프라인을 실행합니다.

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

        # 복합 케이스 — Claude Haiku 호출
        # 이유: 매출 감소 + 재고 부족 + 날씨 이벤트 동시 발생 → 조건 분기로 커버 불가
        return self._llm_classify(signal)
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
- [ ] `FanIntentClassifier` 복합 케이스 LLM 연동 완성
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
