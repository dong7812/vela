# Vela — Project Context (v3)

## 한 줄 설명

**실제 데이터 기반 능동형 AI SDK**
사용자가 묻기 전에, 데이터가 먼저 말한다.

기존 Vela(v1)는 대화 감정 패턴 기반이었다.
v3는 **실제 도메인 데이터의 편차**를 기반으로 능동 개입한다.
감정 수치화의 한계를 버리고, 객관적으로 측정 가능한 데이터만 다룬다.

---

## 핵심 철학

```
기존 Vela  →  감정/대화 패턴 감지 (주관적, 수치화 어려움)
Vela v3    →  실제 데이터 편차 감지 (객관적, 수치화 가능)
```

**변한 것**: 신호 소스 — 감정 추론 → 데이터 이상 감지
**변하지 않은 것**: PRIMA 철학 — 임계값 기반 선택적 개입, LLM은 마지막 수단

---

## 3레이어 파이프라인

```
Layer 01 · Signal Detector     (로컬, 비용 $0)
  → 도메인 데이터로 베이스라인 계산
  → 현재 데이터와 편차 점수화
  → score > threshold 일 때만 Layer 02 호출
  → 전체 이벤트의 ~5%만 통과

Layer 02 · Intent Classifier   (로컬 우선, 복합 케이스만 LLM)
  → 단순 케이스: 조건 분기로 의도 분류 (LLM 없음)
  → 복합 케이스: Claude Haiku 호출
  → WARNING / OPPORTUNITY / INFO + NEED_ACTION / NEED_INFO / NEED_WARNING

Layer 03 · Output Generator    (Claude Sonnet)
  → Layer 01·02 결과 합성
  → message + action + reason 구조로 생성
  → 사용자가 읽고 바로 행동 가능한 형태
```

---

## 프로젝트 구조

```
vela/
├── core/
│   ├── signal_detector.py    # Layer 01 추상 인터페이스 + BaseSignalDetector
│   ├── intent_classifier.py  # Layer 02 추상 인터페이스 + BaseIntentClassifier
│   ├── output_generator.py   # Layer 03 Claude Sonnet 연동
│   ├── pipeline.py           # 3레이어 파이프라인 실행기
│   ├── scheduler.py          # 능동 트리거 스케줄러
│   └── feedback.py           # 개입 결과 수집 (기존 유지)
├── domain/
│   ├── base.py               # DomainPlugin 추상 인터페이스
│   ├── conversation/         # 기존 Vela 대화 도메인 (하위호환)
│   │   ├── plugin.py         # ConversationDomainPlugin
│   │   ├── prima.py          # PRIMA 엔진 (기존 코드 유지)
│   │   ├── wfc.py            # WFC 엔진 (기존 코드 유지)
│   │   └── esconv.py         # ESConv 전략 (기존 코드 유지)
│   └── fandom/               # 팬덤 도메인 (신규, 첫 번째 실도메인)
│       ├── plugin.py         # FandomDomainPlugin
│       ├── detector.py       # 팬 행동 데이터 Signal Detector
│       └── classifier.py     # 팬 유형 Intent Classifier
├── llm/
│   ├── base.py               # BaseLLM 인터페이스 (기존 유지)
│   ├── ollama.py             # Ollama 구현체 (기존 유지)
│   └── claude.py             # Claude API 구현체 (기존 유지)
├── agent.py                  # VelaAgent 진입점 (DomainPlugin 주입)
└── ui/
    └── app.py                # Streamlit UI (기존 유지)
```

---

## 핵심 인터페이스

### DomainPlugin

```python
from abc import ABC, abstractmethod

class DomainPlugin(ABC):
    """새 도메인 추가 시 이것만 구현하면 된다."""

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """도메인 식별자. 예: 'conversation', 'fandom', 'bizowner'"""
        ...

    @abstractmethod
    def get_signal_detector(self) -> "BaseSignalDetector":
        """Layer 01 구현체 반환"""
        ...

    @abstractmethod
    def get_intent_classifier(self) -> "BaseIntentClassifier":
        """Layer 02 구현체 반환"""
        ...

    def get_output_prompt(self) -> str:
        """Layer 03 시스템 프롬프트. 기본값 제공, 도메인별 오버라이드 가능."""
        return "사용자가 바로 행동할 수 있는 인사이트를 message, action, reason 구조로 생성하라."
```

### BaseSignalDetector (Layer 01)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class SignalResult:
    score: float        # 0.0 ~ 1.0, 편차 점수
    should_act: bool    # threshold 초과 여부
    trigger: str        # 감지된 신호 설명 (로깅용)
    raw: dict           # 원본 신호값 (Layer 02 입력)

class BaseSignalDetector(ABC):

    THRESHOLD: float = 0.50  # 도메인별 오버라이드 가능

    @abstractmethod
    def compute_baseline(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        """
        과거 데이터로 베이스라인 계산.
        최소 2주치 데이터 필요. 부족하면 should_act=False 반환.
        """
        ...

    @abstractmethod
    def detect(self, current: dict[str, Any], baseline: dict[str, Any]) -> SignalResult:
        """
        현재 데이터와 베이스라인 편차 점수화.
        LLM 호출 절대 금지. 로컬 연산만.
        """
        ...
```

### BaseIntentClassifier (Layer 02)

```python
from dataclasses import dataclass

@dataclass
class IntentResult:
    type: str           # "WARNING" | "OPPORTUNITY" | "INFO"
    intent: str         # "NEED_ACTION" | "NEED_INFO" | "NEED_WARNING"
    urgency: float      # 0.0 ~ 1.0
    confidence: float   # 낮으면 Layer 03에서 보수적으로 처리
    reason: str

class BaseIntentClassifier(ABC):

    @abstractmethod
    def classify(self, signal: SignalResult) -> IntentResult:
        """
        단순 케이스: 조건 분기로 처리 (LLM 없음).
        복합 케이스만 Claude Haiku 호출.
        LLM 호출 시 반드시 주석으로 이유 명시.
        """
        ...
```

---

## 기존 Vela 대화 도메인 래핑

기존 코드를 건드리지 않는다. ConversationDomainPlugin이 감싼다.

```python
# domain/conversation/plugin.py
class ConversationDomainPlugin(DomainPlugin):

    @property
    def domain_name(self) -> str:
        return "conversation"

    def get_signal_detector(self) -> BaseSignalDetector:
        return PrimaSignalDetector()
        # 기존 PRIMA score → SignalResult 변환
        # should_act = score >= 0.38 (기존 임계값 유지)

    def get_intent_classifier(self) -> BaseIntentClassifier:
        return WfcIntentClassifier()
        # 기존 WFC + ESConv 8가지 전략 선택을 래핑
```

---

## 팬덤 도메인 설계 (첫 번째 실도메인)

### Layer 01 신호 가중치

```python
# domain/fandom/detector.py
score = (
    0.40 * activity_deviation    # 평소 접속/활동 대비 편차
  + 0.30 * content_skip_rate     # 공식 콘텐츠 미소비율
  + 0.20 * session_time_shift    # 접속 시간대 변화
  + 0.10 * purchase_signal       # 티켓/굿즈 페이지 반복 조회
)
THRESHOLD = 0.50
# 근거: 단일 신호 오탐 방지. 복합 신호 2개 이상 기준.
```

### Layer 02 의도 분류 (조건 분기 우선)

```python
# domain/fandom/classifier.py
def classify(self, signal: SignalResult) -> IntentResult:
    raw = signal.raw

    # 단순 케이스 — LLM 없음
    if raw["comeback_imminent"]:
        return IntentResult(type="OPPORTUNITY", intent="NEED_ACTION", urgency=0.95, ...)
    if raw["activity_deviation"] > 0.7 and raw["days_inactive"] >= 3:
        return IntentResult(type="WARNING", intent="NEED_WARNING", urgency=0.85, ...)
    if raw["purchase_signal"] > 0.6:
        return IntentResult(type="OPPORTUNITY", intent="NEED_ACTION", urgency=0.80, ...)
    if raw["is_new_fan"]:
        return IntentResult(type="INFO", intent="NEED_INFO", urgency=0.50, ...)

    # 복합 케이스 — Claude Haiku 호출
    # 이유: 신규 팬 + 구매 의향 높음 + 활동 감소 동시 발생 → 조건 분기로 커버 불가
    return self._llm_classify(signal)
```

### Layer 03 출력 예시

```
OPPORTUNITY(컴백)  → "오늘 밤 12시 컴백이에요. 알림 설정할까요?"
WARNING(이탈위험)  → "오래됐죠? [멤버] 최근 직캠 올라왔어요."
OPPORTUNITY(구매)  → "티켓 오픈 30분 전이에요."
INFO(신규팬)       → "[아티스트] 입문, 이것부터 보세요."
```

---

## 스케줄러

능동성의 핵심. 사용자가 앱을 열지 않아도 백그라운드에서 파이프라인 실행.

```python
# core/scheduler.py
class VelaScheduler:
    def __init__(self, plugin: DomainPlugin, interval_minutes: int = 30):
        self.pipeline = VelaPipeline(plugin)

    async def run_once(self, user_id: str, data: dict) -> OutputResult | None:
        """
        1. Signal Detector 실행
        2. threshold 미달 → None 반환 (종료, 비용 $0)
        3. threshold 초과 → Intent Classifier → Output Generator
        4. 결과 반환 (푸시 알림 or 인앱 카드)
        """
        ...

    async def run_loop(self):
        """주기적으로 모든 활성 사용자에 대해 run_once 실행"""
        ...
```

---

## LLM 설정

```python
# Layer 02 복합 케이스 (빠르고 저렴하게)
CLASSIFIER_MODEL = "claude-haiku-4-5-20251001"

# Layer 03 출력 생성 (품질 우선)
GENERATOR_MODEL = "claude-sonnet-4-6"

# 로컬 fallback (Ollama)
LOCAL_MODEL = "qwen2.5:3b"
```

---

## 코드 규칙

- Python 타입 힌트 항상 사용
- 클래스: PascalCase, 함수/변수: snake_case
- **Layer 01 LLM 호출 절대 금지** — 로컬 연산만
- **Layer 02 LLM은 조건 분기로 커버 안 될 때만** — 주석으로 이유 명시
- 새 신호/가중치 추가 시 근거 주석 필수
- 모든 Layer 입출력은 dataclass로 타입 정의
- 새 도메인 = domain/{name}/ 디렉토리 + DomainPlugin 구현

---

## 구현 우선순위

```
1순위 — vela-core 추상화
  [ ] core/signal_detector.py      BaseSignalDetector 인터페이스
  [ ] core/intent_classifier.py    BaseIntentClassifier 인터페이스
  [ ] core/pipeline.py             3레이어 파이프라인 실행기
  [ ] domain/base.py               DomainPlugin 인터페이스
  [ ] domain/conversation/plugin.py 기존 PRIMA+WFC 래핑 (하위호환 확인)
  [ ] core/scheduler.py            VelaScheduler 기본 구현

2순위 — 팬덤 도메인
  [ ] domain/fandom/detector.py    FandomSignalDetector
  [ ] domain/fandom/classifier.py  FandomIntentClassifier
  [ ] domain/fandom/plugin.py      FandomDomainPlugin
  [ ] core/output_generator.py     Claude Sonnet 연동

3순위 — 고도화
  [ ] PRIMA 가중치 자동 튜닝 (feedback.jsonl 기반)
  [ ] Layer 02 Haiku 복합 케이스 처리
  [ ] OpenAI / Gemini LLM 구현체
  [ ] 영어 문서화
```

---

## 주의사항

- 베이스라인은 최소 2주치 데이터 후 감지 시작. 부족 시 should_act=False 강제.
- 오탐 방지: raw 신호 2개 이상 동시 발생 기준 권장.
- 기존 Vela 대화 기능은 ConversationDomainPlugin으로 완전 하위호환.
- vela-core는 단독 패키지로 배포 가능하게 설계. 실서비스는 별도 레포.
