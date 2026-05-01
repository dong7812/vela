# 🌌 Vela

**AI SDK that acts before you ask**

로컬 LLM 기반으로 대화 상태를 감지하고, 사용자가 묻기 전에 먼저 대화를 이끄는 오픈소스 Python SDK.

---

## 기존 챗봇과 무엇이 다른가?

| | ChatGPT / Claude | **Vela** |
|---|---|---|
| 대화 시작 | 사용자가 먼저 | **문서 로드 시 Vela가 먼저 분석** |
| 개입 시점 | 없음 | **PRIMA 점수가 임계값 초과 시에만 개입** |
| 개입 방식 | 없음 | **ESConv 8가지 전략 중 신호에 맞는 것 선택** |
| 논의 흐름 | 사용자 주도 | **WFC로 대화 공간 구성 → entropy 순서로 이끌어감** |
| 다음 질문 | 없음 | **선제 질문 버튼 자동 생성** |

```
[사용자] 로컬 스토리지를 활용하려고 합니다.

[🌊 WFC ] 그렇다면 '연락 폼 이메일 전송 방식'을 살펴봐야 할 것 같습니다.
          백엔드 없이 이메일을 보내려면 외부 서비스가 필요한데,
          생각해두신 게 있나요?           ← target-guided: INFORMATION

[사용자] 모르겠어요...

[🪞 Vela] 지금 어느 부분에서 막히신 건지 같이 정리해봐요.
          '백엔드가 없다'는 제약 안에서 선택지가 좁혀지는 것 때문에
          막히신 건가요?                  ← REFLECTION (stagnation 감지)
```

---

## 핵심 알고리즘

### 1. PRIMA — 언제, 어떻게 개입할지 결정

**논문 기반 multi-signal 점수 계산:**

```
score = 0.30 × stagnation       ← Horvitz 1999 (Mixed-Initiative Interaction)
      + 0.20 × engagement_decay ← Liu et al. 2021 (ESConv engagement proxies)
      + 0.20 × confusion        ← Deng et al. 2023 (Clarification mode signals)
      + 0.15 × coverage_gap     ← WFC 미논의 비율
      + 0.15 × initiative_debt  ← Horvitz 1999 (대기 비용)

if score > 0.42 → 개입 (LLM 호출 없이 계산)
```

**개입 전략 선택 — ESConv 8가지 전략 (Liu et al., ACL 2021):**

Hill의 Helping Skills Theory에서 실증적으로 도출된 분류 체계를 적용.
Deng et al. (IJCAI 2023) 의 3가지 능동 대화 모드에 따라 신호 패턴과 전략을 매핑.

| 모드 | 전략 | 트리거 신호 |
|---|---|---|
| Clarification | `QUESTION` | initiative_debt 누적 |
| Clarification | `RESTATEMENT` | confusion 높음 |
| Target-guided | `REFLECTION` | stagnation 0.3~0.7 |
| Target-guided | `INFORMATION` | coverage_gap 높음 → **WFC 셀 연계** |
| Non-collaborative | `AFFIRMATION` | engagement 소폭 하락 |
| Non-collaborative | `SUGGESTION` | engagement 급락 |
| Non-collaborative | `REFRAME` | stagnation ≥ 0.7 (STUCK/LOOPING) |
| Non-collaborative | `SELF_DISCLOSURE` | 기타 (AI 관점 공유) |

### 2. WFC 기반 대화 공간

대화가 시작되면 LLM이 맥락을 분석해 논의해야 할 주제 셀을 생성합니다. Wave Function Collapse(WFC) 알고리즘으로 어떤 주제를 먼저 꺼낼지 결정합니다.

- 각 셀은 `entropy` 값을 가집니다. 낮을수록 먼저 논의할 주제
- 한 셀이 논의되면 관련 셀의 entropy가 갱신됩니다 (constraint propagation)
- PRIMA가 `INFORMATION` 전략을 선택하면 WFC 다음 셀을 꺼내 발화

```
대화 공간 (2/6 탐색됨)
✅  ~~백엔드 구조~~
✅  ~~이메일 전송 방식~~
▶️  포트폴리오 접근 범위   ← entropy 최저, 다음 주제
○   상태 관리 전략
○   배포 옵션
○   보안 고려사항
```

### 3. 대화 상태 감지

최근 N턴의 발화를 임베딩해 코사인 유사도로 대화 상태를 판단합니다. PRIMA의 stagnation 신호로 입력됩니다.

```
EXPLORING  → stagnation 0.0   (정상 탐색)
DEEPENING  → stagnation 0.3   (주제 심화)
LOOPING    → stagnation 0.7   → REFRAME 또는 REFLECTION 개입
STUCK      → stagnation 1.0   → REFRAME 즉시 개입
```

### 4. 능동 후속 흐름

```
chat() 호출
  → stagnation / confusion / engagement_decay / coverage_gap / debt 계산
  → score > 0.42?
      Yes → 전략 선택 (PRIMA._select_type)
              INFORMATION + WFC 셀 있음? → wfc_proactive() (WFC 주제 특화 발화)
              그 외                      → prima_intervene(strategy)
      No  → suggest_questions() (선제 질문 버튼 3개)
```

---

## 설치

```bash
# 1. 저장소 클론
git clone https://github.com/dong7812/vela.git
cd vela

# 2. 가상환경 생성 및 의존성 설치
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Ollama 설치 및 모델 준비
# https://ollama.com 에서 Ollama 설치 후:
ollama pull qwen2.5:3b
```

> **RAM 가이드**
> | 모델 | 필요 RAM | 비고 |
> |---|---|---|
> | `qwen2.5:3b` (기본값) | ~2GB | 권장 |
> | `llama3.2:3b` | ~2GB | 대안 |
> | `gemma2:2b` | ~1.5GB | 최소 사양 |

---

## 실행

```bash
# Ollama 서버 시작 (별도 터미널)
ollama serve

# Vela UI 실행
streamlit run vela/ui/app.py
```

---

## SDK 사용법

```python
from vela import VelaAgent

agent = VelaAgent()

# 문서 로드 → 자동 분석 + WFC 대화 공간 초기화
agent.load_document("requirements.pdf")
analysis = agent.analyze_document()

# 대화 — PRIMA 판단이 포함된 3-tuple 반환
response, state, decision = agent.chat("질문 내용")

print(f"상태: {state}")                      # EXPLORING / DEEPENING / LOOPING / STUCK
print(f"PRIMA 점수: {decision.score:.2f}")
print(f"개입 여부: {decision.should_intervene}")

if decision.should_intervene:
    msg = agent.prima_intervene(decision.initiative_type)  # ESConv 전략 기반 발화
    # 또는 WFC 연계:
    wfc_msg = agent.wfc_proactive()   # INFORMATION 전략 + WFC 다음 셀
else:
    questions = agent.suggest_questions()  # fallback: 선제 질문 3개
```

### 커스텀 LLM 연결

```python
from vela.llm.base import BaseLLM

class MyLLM(BaseLLM):
    def chat(self, messages: list[dict], system: str = "") -> str:
        ...
    def is_available(self) -> bool:
        ...

agent = VelaAgent(llm=MyLLM())
```

---

## 프로젝트 구조

```
vela/
├── agent.py              # VelaAgent — 전체 파이프라인 진입점
├── core/
│   ├── prima.py          # PRIMA 엔진 (개입 판단 + ESConv 전략 선택)
│   ├── wfc.py            # WFC 엔진 (ConversationWFC, ConversationCell)
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
├── ui/
│   └── app.py            # Streamlit UI
└── examples/
    └── requirements_bot/ # 요구사항 명세서 능동 분석 데모
```

---

## 참조 논문

| 논문 | 적용 |
|---|---|
| Horvitz (1999) Mixed-Initiative Interaction, CHI | PRIMA 개입 임계값 설계 |
| Liu et al. (2021) ESConv, ACL | InitiativeType 8가지 전략 분류 |
| Deng et al. (2023) Survey on Proactive Dialogue, IJCAI | 3모드 신호-전략 매핑 구조 |
| Deng, Liao et al. (2023) Prompting LLMs for Proactive Dialogues, EMNLP | 전략별 시스템 프롬프트 분리 설계 근거 |

---

## 기술 스택

- **LLM 백엔드**: [Ollama](https://ollama.com) (로컬, 기본 모델: `qwen2.5:3b`) / Claude API
- **임베딩**: [sentence-transformers](https://www.sbert.net) (`all-MiniLM-L6-v2`, 로컬)
- **벡터 DB**: [ChromaDB](https://www.trychroma.com) (로컬 persist)
- **UI**: [Streamlit](https://streamlit.io)
- **언어**: Python 3.10+

---

## 기여하기

PR 환영합니다.

- [ ] 스트리밍 응답 UI 반영
- [ ] PRIMA 가중치 자동 튜닝 (RL 또는 사용자 피드백 기반)
- [ ] WFC entropy 학습 기반 초기화
- [ ] 영어 문서화

```bash
pip install -e .
```

---

## 라이선스

MIT
