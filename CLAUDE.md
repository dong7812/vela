# Vela — Project Context

## 한 줄 설명
AI SDK that acts before you ask
로컬 LLM(Ollama) 기반으로 맥락을 감지해 능동적으로 대화를 이끄는 오픈소스 Python SDK.

## 프로젝트 구조
```
vela/
├── agent.py              # VelaAgent — 전체 파이프라인 진입점
├── core/
│   ├── embedder.py       # 임베딩 변환 + 유사도 계산 (sentence-transformers)
│   ├── context.py        # 대화 맥락 윈도우 관리 (ContextWindow, Turn)
│   └── state.py          # 대화 상태 감지 + 능동 트리거 (StateDetector)
├── rag/
│   ├── loader.py         # 문서 로드 + 청킹 (txt, md, pdf)
│   └── retriever.py      # ChromaDB 로컬 벡터 저장 + 검색
├── llm/
│   ├── base.py           # BaseLLM 추상 인터페이스
│   └── ollama.py         # Ollama 구현체
├── ui/
│   └── app.py            # Streamlit UI
└── examples/
    └── requirements_bot/ # 요구사항 명세서 능동 분석 챗봇 데모
```

## 기술 스택
- **LLM 백엔드**: Ollama (로컬, 기본 모델: llama3)
- **임베딩**: sentence-transformers (all-MiniLM-L6-v2, 로컬)
- **벡터 DB**: ChromaDB (로컬 persist)
- **UI**: Streamlit
- **언어**: Python 3.10+

## 핵심 개념

### 대화 상태 (ConversationState)
```
EXPLORING  → 주제 탐색 중 (정상)
DEEPENING  → 주제 깊어지는 중 (정상)
LOOPING    → 맴돌고 있음 → 능동 개입 필요
STUCK      → 완전히 막힘 → 즉시 방향 전환
```

### 상태 감지 로직
- 최근 N턴의 사용자 발화를 임베딩 변환
- 턴 간 코사인 유사도 계산
- 평균 유사도 임계값으로 상태 판단
  - 0.95 이상 → STUCK
  - 0.85 이상 → LOOPING
  - 0.60 이상 → DEEPENING
  - 그 이하 → EXPLORING

### 파이프라인 흐름
```
사용자 입력
  → ContextWindow에 추가
  → RAG 검색 (ChromaDB)
  → StateDetector로 상태 판단
  → 상태에 따라 시스템 프롬프트 분기
  → Ollama로 응답 생성
  → ContextWindow에 응답 추가
```

## 개발 규칙

### 코드 스타일
- Python 타입 힌트 항상 사용
- 클래스: PascalCase, 함수/변수: snake_case
- 모듈별 단일 책임 원칙 준수
- 추상 클래스(BaseLLM)로 LLM 교체 가능하게 유지

### LLM 확장
- 새 LLM 추가 시 반드시 `llm/base.py`의 `BaseLLM` 상속
- `chat()`, `embed()`, `is_available()` 구현 필수

### RAG 확장
- 새 파일 형식 추가 시 `rag/loader.py`의 `load_document()`에만 추가
- chunk_size 기본값: 500 words, overlap: 50 words

### 상태 임계값 조정
- `core/state.py` 상단 상수에서만 조정
  - LOOP_THRESHOLD = 0.85
  - STUCK_THRESHOLD = 0.95
  - MIN_TURNS_TO_JUDGE = 3

## 실행 명령어
```bash
# 의존성 설치
pip install -r requirements.txt

# Ollama 모델 준비 (최초 1회)
ollama pull llama3

# UI 실행
streamlit run vela/ui/app.py

# 기본 사용
from vela import VelaAgent
agent = VelaAgent()
agent.load_document("my_doc.pdf")
response, state = agent.chat("질문 내용")
```

## 주의사항
- Ollama가 로컬에서 실행 중이어야 함 (http://localhost:11434)
- `is_available()` 호출로 사전 체크 권장
- ChromaDB는 `.vela_db/` 디렉토리에 로컬 persist
- 비개발자 대상 UI이므로 Streamlit 에러 메시지는 한국어로

## 다음 구현 목표
- [ ] examples/requirements_bot 데모 완성
- [ ] LOOPING/STUCK 상태별 능동 개입 프롬프트 고도화
- [ ] 스트리밍 응답 UI 반영
- [ ] Ollama 외 LLM 추가 (OpenAI, Claude API)
- [ ] 설치 스크립트 (비개발자용)
