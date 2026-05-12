# Vela

**실제 데이터 기반 능동형 AI SDK**
사용자가 묻기 전에, 데이터가 먼저 말한다.

기존 Vela(v1)는 대화 감정 패턴을 감지하는 단일 도메인 SDK였다.  
**v3는 임의의 실데이터 편차를 감지하는 범용 SDK로 전환한다.**  
감정 수치화의 한계를 버리고, 객관적으로 측정 가능한 데이터만 다룬다.

```
기존 Vela  →  감정/대화 패턴 감지 (주관적, 수치화 어려움)
Vela v3    →  실제 데이터 편차 감지 (객관적, 수치화 가능)
```

---

## Table of Contents

- [기존 SDK와의 차이](#기존-sdk와의-차이)
- [아키텍처 — 3-Layer Pipeline](#아키텍처--3-layer-pipeline)
- [내장 도메인](#내장-도메인)
  - [대화 도메인 (Conversation)](#대화-도메인-conversation)
  - [팬덤 도메인 (Fandom)](#팬덤-도메인-fandom)
- [설치](#설치)
- [빠른 시작](#빠른-시작)
- [SDK 레퍼런스](#sdk-레퍼런스)
  - [대화 도메인 — VelaAgent](#대화-도메인--velaagent)
  - [범용 파이프라인 — DomainPlugin 주입](#범용-파이프라인--domainplugin-주입)
  - [백그라운드 실행 — VelaScheduler](#백그라운드-실행--velascheduler)
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
| 비용 제어 | 모든 요청 LLM 호출 | Layer 01에서 ~5%만 통과, 나머지 $0 |
| 도메인 신호 | 텍스트 쿼리만 | 행동 데이터, 수치 편차, 시계열 이상 등 |
| 대화 도메인 | 지원 없음 | PRIMA + WFC + ESConv 통합 |
| 스케줄링 | 없음 | VelaScheduler — 백그라운드 주기 실행 |

---

## 아키텍처 — 3-Layer Pipeline

모든 도메인은 동일한 3단계 파이프라인을 통과합니다.

```
Layer 01 · Signal Detector     (로컬 연산, 비용 $0)
  → 도메인 데이터로 베이스라인 계산
  → 현재 데이터와 편차 점수화 (0.0 ~ 1.0)
  → score > threshold 일 때만 Layer 02 진행
  → 전체 이벤트의 ~5%만 통과

Layer 02 · Intent Classifier   (로컬 우선, 복합 케이스만 LLM)
  → 단순 케이스: 조건 분기로 즉시 분류 (LLM 없음)
  → 복합 케이스: Claude Haiku 호출
  → WARNING / OPPORTUNITY / INFO
     + NEED_ACTION / NEED_INFO / NEED_WARNING

Layer 03 · Output Generator    (LLM)
  → Layer 01·02 결과 합성
  → 사용자가 읽고 바로 행동 가능한 형태로 출력
```

**DomainPlugin 인터페이스**로 도메인마다 Layer 01·02를 교체합니다.  
Layer 03과 파이프라인 실행기는 공용이므로 추가 구현이 필요 없습니다.

```python
class DomainPlugin(ABC):
    @property
    @abstractmethod
    def domain_name(self) -> str: ...          # 'conversation' | 'fandom' | ...

    @abstractmethod
    def get_signal_detector(self) -> BaseSignalDetector: ...   # Layer 01

    @abstractmethod
    def get_intent_classifier(self) -> BaseIntentClassifier: ...  # Layer 02

    def get_output_prompt(self) -> str: ...    # Layer 03 시스템 프롬프트
```

---

## 내장 도메인

### 대화 도메인 (Conversation)

기존 Vela v1의 대화 상태 감지 + PRIMA + WFC + ESConv를 모두 포함합니다.  
하위 호환 — `VelaAgent()`를 기존처럼 그대로 사용하면 됩니다.

#### PRIMA — 개입 판단 엔진

매 사용자 턴마다 5가지 신호를 계산해 개입 점수를 구합니다.  
LLM 호출 없이 임계값(0.38)을 초과할 때만 전략을 선택하고 발화합니다.

```
score = 0.35 × stagnation       (대화 맴돔 정도, 대화 상태 기반)
      + 0.25 × confusion        (혼란/막힘, 의문형 구분 감지)
      + 0.20 × coverage_gap     (WFC 미논의 비율)
      + 0.12 × engagement_decay (TTR 기반 참여도 하락)
      + 0.08 × initiative_debt  (연속 반응 횟수 누적)

score ≥ 0.38  → 개입 (ESConv 전략 선택)
score < 0.38  → 선제 질문 3개 생성 (fallback)
```

**ESConv 8가지 개입 전략** (Liu et al., ACL 2021 / Deng et al., IJCAI 2023):

| 모드 | 전략 | 트리거 |
|---|---|---|
| Clarification | `QUESTION` | initiative_debt ≥ 0.6 |
| Clarification | `RESTATEMENT` | confusion ≥ 0.5 (하드 트리거) |
| Target-guided | `REFLECTION` | stagnation 0.3–0.7, 2턴 지속 |
| Target-guided | `INFORMATION` | coverage_gap ≥ 0.5, WFC 셀 연계 |
| Non-collaborative | `AFFIRMATION` | engagement 소폭 하락 + debt ≥ 0.6 |
| Non-collaborative | `SUGGESTION` | engagement ≥ 0.5 급락 |
| Non-collaborative | `REFRAME` | stagnation ≥ 1.0 즉시 발화 |
| Non-collaborative | `SELF_DISCLOSURE` | 그 외 |

#### WFC — 대화 공간 관리

문서 로드 또는 첫 대화 후 핵심 논의 주제를 셀로 구성합니다.  
Wave Function Collapse 알고리즘으로 entropy가 가장 낮은 주제를 우선 꺼냅니다.

```
대화 공간 (2/5 탐색됨)
✅  ~~백엔드 구조~~
✅  ~~이메일 전송 방식~~
▶️  포트폴리오 접근 범위   ← entropy 최저, 다음 주제
○   상태 관리 전략
○   보안 고려사항
```

#### 대화 상태 감지

최근 N턴의 발화를 임베딩해 코사인 유사도로 상태를 판단합니다.

| 상태 | 유사도 | stagnation | 의미 |
|---|---|---|---|
| `EXPLORING` | < 0.60 | 0.0 | 새로운 주제 탐색 중 |
| `DEEPENING` | 0.60–0.85 | 0.3 | 주제가 깊어지는 중 |
| `LOOPING` | 0.85–0.95 | 0.7 | 같은 자리 맴돔 → 개입 필요 |
| `STUCK` | ≥ 0.95 | 1.0 | 완전히 막힘 → 즉시 개입 |

---

### 팬덤 도메인 (Fandom)

팬 행동 데이터의 편차로 이탈 위험·구매 기회·컴백 알림을 능동 감지합니다.

#### Layer 01 신호 가중치

```
score = 0.40 × activity_deviation   (평소 접속/활동 대비 편차)
      + 0.30 × content_skip_rate    (공식 콘텐츠 미소비율)
      + 0.20 × session_time_shift   (접속 시간대 변화)
      + 0.10 × purchase_signal      (티켓/굿즈 페이지 반복 조회)

THRESHOLD = 0.50  # 단일 신호 오탐 방지, 복합 신호 2개 이상 기준
```

#### Layer 02 의도 분류 (조건 분기 우선)

```python
if comeback_imminent:
    → OPPORTUNITY / NEED_ACTION  urgency=0.95  # "오늘 밤 12시 컴백이에요."
if activity_deviation > 0.7 and days_inactive >= 3:
    → WARNING / NEED_WARNING     urgency=0.85  # "오래됐죠? 최근 직캠 올라왔어요."
if purchase_signal > 0.6:
    → OPPORTUNITY / NEED_ACTION  urgency=0.80  # "티켓 오픈 30분 전이에요."
if is_new_fan:
    → INFO / NEED_INFO           urgency=0.50  # "[아티스트] 입문, 이것부터 보세요."
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
ollama serve           # 별도 터미널에서 실행
streamlit run vela/ui/app.py
```

`http://localhost:8501` 접속. 사이드바에서 파일 업로드 또는 바로 대화 시작.  
Claude API 사용 시: 사이드바 → LLM 설정 → Claude API 선택 후 API Key 입력.

### SDK (팬덤 도메인 예시)

```python
from vela import VelaAgent, FandomDomainPlugin
from vela.llm.claude import ClaudeLLM

plugin = FandomDomainPlugin()
agent = VelaAgent(plugin=plugin, llm=ClaudeLLM(api_key="sk-ant-..."))

# 과거 14일치 행동 데이터
history = [
    {"activity_count": 12, "session_hour": 21} for _ in range(14)
]

# 오늘 데이터 — 컴백 임박 + 활동 급감
today = {
    "activity_count": 2,
    "session_hour": 3,
    "comeback_imminent": True,
    "purchase_signal": 0.8,
}

result = agent.run(current=today, history=history)
if result:
    print(result.message)
    # → "오늘 밤 12시 컴백이에요. 알림 설정할까요?"
```

---

## SDK 레퍼런스

### 대화 도메인 — VelaAgent

기존 v1 API를 그대로 유지합니다.

```python
from vela import VelaAgent

agent = VelaAgent()

# 문서 로드 + 분석 + WFC 초기화
agent.load_document("requirements.pdf")
analysis = agent.analyze_document()

# 대화 (비스트리밍)
response, state, decision = agent.chat("질문 내용")
print(f"상태: {state}")                         # EXPLORING / DEEPENING / LOOPING / STUCK
print(f"PRIMA 점수: {decision.score:.2f}")
print(f"개입 여부: {decision.should_intervene}")

if decision.should_intervene:
    if decision.initiative_type.value == "INFORMATION" and agent.get_wfc_next():
        msg = agent.wfc_proactive()              # WFC 다음 주제 발화
    else:
        msg = agent.prima_intervene(decision.initiative_type)
else:
    questions = agent.suggest_questions()        # 선제 질문 3개 (fallback)

# 스트리밍 (UI 연동)
messages, system, state = agent.prepare_chat(user_input)
for token in agent._llm.chat_stream(messages, system):
    print(token, end="", flush=True)
decision = agent.finalize_chat(full_response, state)

# WFC 셀 조회
cells = agent.get_wfc_cells()
next_cell = agent.get_wfc_next()

# 피드백 통계
summary = agent.feedback.get_summary()
# {"REFRAME": {"count": 5, "success_rate": 0.80, "avg_length_ratio": 1.52}, ...}
```

---

### 범용 파이프라인 — DomainPlugin 주입

`VelaAgent(plugin=...)` 형태로 임의 도메인 파이프라인을 실행합니다.

```python
from vela import VelaAgent, FandomDomainPlugin, ConversationDomainPlugin

# 팬덤 도메인
agent = VelaAgent(plugin=FandomDomainPlugin(), llm=my_llm)
result = agent.run(current=today_data, history=past_14_days)

# 대화 도메인을 파이프라인으로 실행
agent = VelaAgent(plugin=ConversationDomainPlugin(), llm=my_llm)
result = agent.run(current={
    "user_turns": context.get_user_turns(),
    "wfc_total": len(wfc_cells),
    "wfc_collapsed": collapsed_count,
})

if result:
    print(result.message)   # 생성된 인사이트
    print(result.reason)    # 개입 이유
```

VelaPipeline을 직접 사용하면 agent 없이도 파이프라인을 실행할 수 있습니다.

```python
from vela import VelaPipeline, FandomDomainPlugin

pipeline = VelaPipeline(plugin=FandomDomainPlugin(), llm=my_llm)
result = pipeline.run(current=today_data, history=history)
```

---

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

async def fetch_all_users() -> dict[str, dict]:
    # DB에서 활성 사용자 + 오늘 행동 데이터 조회
    return {"user_001": {...}, "user_002": {...}}

async def main():
    async for user_id, result in ...:
        pass

asyncio.run(scheduler.run_loop(user_data_fn=fetch_all_users))
```

---

### 새 도메인 추가하기

`DomainPlugin`, `BaseSignalDetector`, `BaseIntentClassifier` 세 가지를 구현합니다.

#### 1. Layer 01 — Signal Detector

```python
from vela import BaseSignalDetector, SignalResult

class BizOwnerSignalDetector(BaseSignalDetector):
    THRESHOLD = 0.55   # 도메인 특성에 맞게 조정

    def compute_baseline(self, history: list[dict]) -> dict:
        # 과거 데이터로 정상 범위 계산 (최소 2주치 권장)
        avg_revenue = sum(h["daily_revenue"] for h in history) / len(history)
        return {"avg_revenue": avg_revenue}

    def detect(self, current: dict, baseline: dict) -> SignalResult:
        # LLM 호출 절대 금지 — 로컬 연산만
        deviation = abs(current["daily_revenue"] - baseline["avg_revenue"])
        score = min(1.0, deviation / baseline["avg_revenue"])
        return SignalResult(
            score=score,
            should_act=score >= self.THRESHOLD,
            trigger=f"revenue_deviation (score={score:.2f})",
            raw={"deviation": deviation, **current},
        )
```

#### 2. Layer 02 — Intent Classifier

```python
from vela import BaseIntentClassifier, IntentResult, SignalResult

class BizOwnerIntentClassifier(BaseIntentClassifier):
    def classify(self, signal: SignalResult) -> IntentResult:
        raw = signal.raw

        # 단순 케이스 — 조건 분기, LLM 없음
        if raw.get("daily_revenue", 0) < raw.get("break_even", float("inf")):
            return IntentResult(
                type="WARNING", intent="NEED_WARNING",
                urgency=0.90, confidence=0.85,
                reason="손익분기점 미달",
            )
        if raw.get("top_item_out_of_stock"):
            return IntentResult(
                type="OPPORTUNITY", intent="NEED_ACTION",
                urgency=0.75, confidence=0.90,
                reason="베스트셀러 재고 부족",
            )

        # 복합 케이스 — LLM 호출 (이유: 여러 신호 동시 발생으로 분기 커버 불가)
        return self._llm_classify(signal)
```

#### 3. DomainPlugin 조합

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
        return "소상공인 맥락에서 오늘 바로 실행할 수 있는 조치를 한두 문장으로 생성하라."

# 사용
agent = VelaAgent(plugin=BizOwnerDomainPlugin(), llm=my_llm)
result = agent.run(current=today_sales, history=past_30_days)
```

---

### 커스텀 LLM 연결

`BaseLLM`을 상속해 `chat()`, `chat_stream()`, `is_available()`을 구현합니다.

```python
from typing import Iterator
from vela.llm.base import BaseLLM

class MyLLM(BaseLLM):
    def chat(self, messages: list[dict], system: str = "") -> str:
        ...

    def chat_stream(self, messages: list[dict], system: str = "") -> Iterator[str]:
        yield from ...

    def is_available(self) -> bool:
        ...

agent = VelaAgent(llm=MyLLM())
```

OpenAI, Gemini, Bedrock 등 어떤 LLM이든 위 인터페이스만 맞추면 연결됩니다.

---

## 프로젝트 구조

```
vela/
├── agent.py                  # VelaAgent — 진입점, DomainPlugin 주입
├── core/
│   ├── signal_detector.py    # Layer 01 추상 인터페이스 + SignalResult
│   ├── intent_classifier.py  # Layer 02 추상 인터페이스 + IntentResult
│   ├── output_generator.py   # Layer 03 LLM 출력 생성기 + OutputResult
│   ├── pipeline.py           # VelaPipeline — 3레이어 실행기
│   ├── scheduler.py          # VelaScheduler — 백그라운드 능동 트리거
│   ├── feedback.py           # FeedbackLogger — JSONL 기반 outcome 수집
│   ├── prima.py              # PRIMA 엔진 (개입 판단 + ESConv 전략 선택)
│   ├── wfc.py                # WFC 엔진 (ConversationWFC, ConversationCell)
│   ├── state.py              # 대화 상태 감지 (StateDetector)
│   ├── embedder.py           # 임베딩 + 코사인 유사도 (sentence-transformers)
│   └── context.py            # 대화 컨텍스트 윈도우
├── domain/
│   ├── base.py               # DomainPlugin 추상 인터페이스
│   ├── conversation/         # 대화 도메인 (v1 하위 호환)
│   │   ├── plugin.py         # ConversationDomainPlugin
│   │   ├── esconv.py         # ESConv 전략 프롬프트 (8가지)
│   │   ├── prima.py          # core.prima re-export
│   │   └── wfc.py            # core.wfc re-export
│   └── fandom/               # 팬덤 도메인 (첫 번째 실도메인)
│       ├── plugin.py         # FandomDomainPlugin
│       ├── detector.py       # FanSignalDetector
│       └── classifier.py     # FanIntentClassifier
├── llm/
│   ├── base.py               # BaseLLM 추상 인터페이스
│   ├── ollama.py             # Ollama 구현체
│   └── claude.py             # Claude API 구현체
├── rag/
│   ├── loader.py             # 문서 로드 + 청킹 (txt, md, pdf)
│   └── retriever.py          # ChromaDB 로컬 벡터 저장 + 검색
└── ui/
    └── app.py                # Streamlit UI (대화 도메인)
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
| Horvitz (1999) *Mixed-Initiative Interaction*, CHI | PRIMA 임계값 — E[utility(act)] > E[utility(wait)] |
| Liu et al. (2021) *ESConv*, ACL | InitiativeType 8가지 전략 분류 체계 + 신호 가중치 |
| Deng et al. (2023) *Survey on Proactive Dialogue Systems*, IJCAI | 3모드 신호-전략 매핑 구조 |
| Deng, Liao et al. (2023) *Prompting LLMs for Proactive Dialogues*, EMNLP | 전략별 시스템 프롬프트 분리 설계 근거 |
| Bohus & Rudnicky (2005) *Error Handling in Conversational Systems* | stagnation 일시 vs. 지속 구분 (2턴 기준) |
| Murray & Levesque (2003) *Expressing Key Points and Managing Convergence* | 응답 길이 30% 증가 = 참여도 회복 지표 |
| Richards (1987) *Type-Token Ratio* | engagement_decay — TTR 기반 어휘 다양성 |
| Sacks, Schegloff & Jefferson (1974) *Turn-taking in Conversation* | confusion — 의문형 vs. 평서형 위치 감지 |
| Reimers & Gurevych (2019) *Sentence-BERT*, EMNLP | WFC 셀 품질 필터 임계값 (cosine < 0.15) |

---

## 기여하기

PR과 이슈 모두 환영합니다.

```bash
git clone https://github.com/dong7812/vela.git
cd vela
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

**우선순위 높은 기여 항목**

- [ ] 새 도메인 추가 — `bizowner`, `health`, `ecommerce` 등
- [ ] `FanIntentClassifier` 복합 케이스 LLM 연동 완성
- [ ] PRIMA 가중치 자동 튜닝 — `.vela_feedback.jsonl` 기반 RL/Bayesian 최적화
- [ ] `VelaScheduler` 푸시 알림 연동 (FCM, APNs)
- [ ] OpenAI / Gemini / Bedrock LLM 구현체 추가
- [ ] 영어 문서화 및 i18n 지원
- [ ] Layer 01·02 단위 테스트

**코드 컨벤션**

- 타입 힌트 필수 (Python 3.10+)
- Layer 01 구현에서 LLM 호출 금지
- Layer 02 LLM 호출 시 이유 주석 필수
- 새 LLM 추가 시 반드시 `BaseLLM` 상속

---

## 라이선스

MIT © [dong7812](https://github.com/dong7812)
