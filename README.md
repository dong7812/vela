# 🌌 Vela

**AI SDK that acts before you ask**

로컬 LLM 기반으로 임베딩 유사도로 대화 상태를 감지하고, 사용자가 묻기 전에 먼저 대화를 이끄는 오픈소스 Python SDK.

---

## 기존 챗봇과 무엇이 다른가?

| | ChatGPT / Claude | **Vela** |
|---|---|---|
| 대화 시작 | 사용자가 먼저 | **Vela가 먼저** (문서 로드 시 자동 분석) |
| 논의 흐름 | 사용자 주도 | **WFC로 대화 공간 구성 → entropy 순서로 이끌어감** |
| 반복/막힘 | 없음 | **LOOPING/STUCK 감지 → 즉시 방향 전환** |
| 다음 질문 | 없음 | **선제 질문 버튼 자동 생성** |

```
[사용자가 문서 업로드]

[📄 Vela] 문서를 분석했습니다. '인증 없는 접근 범위'가 모호하고,
          요구사항 3번과 7번이 충돌 가능성이 있습니다.

[🌊 WFC ] 먼저 '백엔드 없는 정적 파일 구조'에 대해 이야기해봅시다.
          배포 환경에서 상태 관리는 어떻게 처리하실 계획인가요?

[사용자] 로컬 스토리지를 활용하려고 합니다.

[🌊 WFC ] 그렇다면 다음 주제인 '연락 폼 이메일 전송 방식'을 살펴볼게요.
          백엔드 없이 이메일을 보내려면 외부 서비스가 필요한데, 생각해두신 게 있나요?
```

---

## 핵심 개념

### 1. WFC 기반 대화 공간

대화가 시작되면 LLM이 맥락을 분석해 논의해야 할 주제 셀을 생성합니다. 이후 WFC(Wave Function Collapse) 알고리즘으로 어떤 주제를 먼저 꺼낼지 결정합니다.

- 각 셀은 `entropy` 값을 가집니다. 낮을수록 먼저 논의할 주제
- 한 셀이 논의되면 관련 셀의 entropy가 갱신됩니다 (constraint propagation)
- 사용자가 어떤 주제를 언급했는지 임베딩 유사도로 감지해 자동으로 셀을 collapse

```
대화 공간 (2/6 탐색됨)
✅  ~~백엔드 구조~~
✅  ~~이메일 전송 방식~~
▶️  포트폴리오 접근 범위   ← entropy 최저, 다음 주제
○   상태 관리 전략
○   배포 옵션
○   보안 고려사항
```

### 2. 대화 상태 감지

최근 N턴의 발화를 임베딩해 코사인 유사도로 대화 상태를 판단합니다. WFC보다 우선순위가 높아 대화가 나빠지면 즉시 개입합니다.

```
EXPLORING  → 새로운 주제 탐색 중       (WFC가 다음 셀 제시)
DEEPENING  → 주제 심화 중             (WFC가 다음 셀 제시)
LOOPING    → 같은 말 반복 중          → 능동 개입 (WFC 우선순위 밀림)
STUCK      → 완전히 막힘             → 즉시 방향 전환
```

### 3. 능동 후속 우선순위

```
매 응답 후:
  1순위 LOOPING/STUCK  → proactive_nudge()  방향 전환 개입
  2순위 WFC 셀 남음    → wfc_proactive()    다음 주제 선제 제시
  3순위 WFC 소진       → suggest_questions() 선제 질문 버튼 3개
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

# 요구사항 분석 데모
streamlit run examples/requirements_bot/app.py
```

---

## SDK 사용법

```python
from vela import VelaAgent

agent = VelaAgent()

# 문서 로드 → 자동 분석 + WFC 대화 공간 초기화
agent.load_document("requirements.pdf")
analysis = agent.analyze_document()  # 사용자 입력 없이 먼저 발화 + WFC 초기화

# 대화 (매 턴마다 WFC 셀 자동 감지 및 collapse)
response, state = agent.chat("질문 내용")
print(f"상태: {state}")  # EXPLORING / DEEPENING / LOOPING / STUCK

# 능동 후속 — 우선순위대로 호출
nudge = agent.proactive_nudge(state)     # LOOPING/STUCK이면 개입, 아니면 None
wfc   = agent.wfc_proactive()            # 다음 WFC 셀 선제 발화, 소진 시 None
qs    = agent.suggest_questions()        # fallback: 선제 질문 3개
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
│   ├── wfc.py            # WFC 엔진 (ConversationWFC, ConversationCell)
│   ├── embedder.py       # 임베딩 + 코사인 유사도 (sentence-transformers)
│   ├── context.py        # 대화 컨텍스트 윈도우 관리
│   └── state.py          # 대화 상태 감지 (StateDetector)
├── rag/
│   ├── loader.py         # 문서 로드 + 청킹 (txt, md, pdf)
│   └── retriever.py      # ChromaDB 로컬 벡터 저장 + 검색
├── llm/
│   ├── base.py           # BaseLLM 추상 인터페이스
│   └── ollama.py         # Ollama 구현체
├── ui/
│   └── app.py            # Streamlit UI
└── examples/
    └── requirements_bot/ # 요구사항 명세서 능동 분석 데모
```

---

## 기술 스택

- **LLM 백엔드**: [Ollama](https://ollama.com) (로컬, 기본 모델: `qwen2.5:3b`)
- **임베딩**: [sentence-transformers](https://www.sbert.net) (`all-MiniLM-L6-v2`, 로컬)
- **벡터 DB**: [ChromaDB](https://www.trychroma.com) (로컬 persist)
- **UI**: [Streamlit](https://streamlit.io)
- **언어**: Python 3.10+

---

## 기여하기

PR 환영합니다. 아래 항목을 우선적으로 기여해주세요:

- [ ] OpenAI / Claude API LLM 구현체 추가 (`llm/` 하위에 `BaseLLM` 상속)
- [ ] 스트리밍 응답 UI 반영
- [ ] WFC entropy 튜닝 가이드
- [ ] 영어 문서화

```bash
# 개발 환경 설정
pip install -e .
```

---

## 라이선스

MIT
