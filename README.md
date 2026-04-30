# 🌌 Vela

**AI SDK that acts before you ask**

로컬 LLM 기반으로 임베딩 유사도로 대화 상태를 감지하고, 사용자가 묻기 전에 먼저 대화를 이끄는 오픈소스 Python SDK.

---

## 기존 챗봇과 무엇이 다른가?

| | ChatGPT / Claude | **Vela** |
|---|---|---|
| 대화 시작 | 사용자가 먼저 | **Vela가 먼저** (문서 로드 시 자동 분석) |
| 반복 감지 | 없음 | **LOOPING/STUCK 상태 감지 → 즉시 개입** |
| 논의 흐름 | 사용자 주도 | **Agenda 자동 생성 → LLM이 순서대로 이끌어감** |
| 다음 질문 | 없음 | **선제 질문 버튼 자동 생성** |

```
[사용자가 문서 업로드]

[🎯 Vela] 문서를 분석했습니다. 주요 요구사항 3가지를 발견했으며,
          '인증 없는 접근 범위'가 모호합니다. 먼저 이 부분을 명확히 할까요?

[📋 Vela] 논의 계획 1번 — 백엔드 없는 정적 파일 구조에 대해 얘기해봅시다.
          배포 환경에서 상태 관리는 어떻게 처리하실 계획인가요?

[사용자] ...
```

---

## 핵심 개념: 대화 상태 감지

최근 N턴의 발화를 임베딩해 코사인 유사도로 대화 상태를 실시간 판단합니다.

```
EXPLORING  → 새로운 주제 탐색 중       (정상)
DEEPENING  → 주제 심화 중             (정상)
LOOPING    → 같은 말 반복 중          → 능동 개입
STUCK      → 완전히 막힘             → 즉시 방향 전환
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
ollama pull qwen2.5:3b   # RAM 8GB 이상 권장 (최소 4GB)
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

# 문서 로드 → 자동 분석 + 논의 계획 생성
agent.load_document("requirements.pdf")
analysis = agent.analyze_document()   # 사용자 입력 없이 먼저 발화
agenda   = agent.generate_agenda()    # ['항목1', '항목2', ...]
first    = agent.advance_agenda()     # 첫 번째 주제로 대화 시작

# 대화
response, state = agent.chat("질문 내용")
print(f"상태: {state}")  # EXPLORING / DEEPENING / LOOPING / STUCK

# 상태 기반 능동 개입
nudge = agent.proactive_nudge(state)   # LOOPING/STUCK이면 개입 메시지, 아니면 None

# 선제 질문 생성 (Agenda 완료 후)
questions = agent.suggest_questions()  # ['질문1', '질문2', '질문3']
```

### 커스텀 LLM 연결

```python
from vela.llm.base import BaseLLM

class MyLLM(BaseLLM):
    def chat(self, messages: list[dict], system: str = "") -> str:
        ...
    def embed(self, text: str) -> list[float]:
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
- [ ] 상태 감지 임계값 튜닝 가이드
- [ ] 영어 문서화

```bash
# 개발 환경 설정
pip install -e .
```

---

## 라이선스

MIT
