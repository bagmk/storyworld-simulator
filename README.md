# Storyworld Simulator

에피소드 설정 파일을 넣으면, 캐릭터 상호작용을 시뮬레이션하고, 장면으로 압축한 뒤, 읽을 수 있는 한국어 소설 챕터로 생성하고, Discord 안에서 리뷰와 재실행까지 이어 가는 파이프라인입니다.

이 저장소의 핵심 진입점은 `!novel-daily`입니다.  
처음 보는 사람도 이 명령 하나만 이해하면 전체 구조를 빠르게 파악할 수 있습니다.

![Discord sample](Discord_sample.png)

* * *

## 한눈에 보기

이 프로젝트는 단순한 텍스트 생성기가 아닙니다.

- 에피소드 설정(YAML)을 읽고
- 캐릭터 상호작용을 시뮬레이션하고
- raw turn log를 장면 단위로 압축한 뒤
- 한국어 소설 챕터로 생성하고
- 자동 리뷰와 사용자 피드백을 반영해
- 스토리 설정 또는 파라미터까지 다시 수정하는

**“생성 → 리뷰 → 개선 → 재실행” 루프**를 Discord 안에서 운영합니다.

핵심적으로는 이렇게 이해하면 됩니다.

> 설정 파일 하나를 넣으면 초고를 만들고, Discord에서 읽고, 고치고, 다음 화로 넘길 수 있는 시스템

* * *

## 왜 이 구조를 쓰는가

LLM으로 소설을 바로 생성하면 보통 이런 문제가 생깁니다.

- 같은 내용을 조금씩 반복함
- 설정은 맞는데 소설처럼 잘 읽히지 않음
- 인물의 대화와 반응이 비슷해짐
- 기술 설명이 과해져 장면이 멈춤
- 장편에서 앞 화의 단서와 리듬이 쉽게 무너짐

이 프로젝트는 그 문제를 **단계 분리**로 다룹니다.

1. 먼저 **시뮬레이션**으로 “무슨 일이 벌어졌는가”를 생성합니다.
2. 다음으로 **scene distillation**으로 반복적인 turn log를 장면으로 압축합니다.
3. 마지막으로 **prose generation**으로 읽히는 문장으로 바꿉니다.
4. 그리고 **polishing + feedback loop**로 문체, 분량, 앵커, 가독성을 정리합니다.

즉,

**사건 생성**과 **문장 생성**을 분리해서 장편 소설 품질을 더 안정적으로 관리하는 구조입니다.

* * *

## 전체 파이프라인

```text
episode.yaml
  -> Config / Guardian checks
  -> SimulationOrchestrator
      -> DirectorAI guardrails
      -> character turn loop
      -> memory / clue / continuity updates
  -> SceneDistiller
      -> repeated turn compression
      -> beat mapping
      -> scene summaries
  -> ProseGenerator
      -> scene-by-scene prose
      -> transition generation
      -> chapter assembly
  -> ChapterPolisher
      -> bounded polish
      -> anchor coverage correction
      -> reader feedback pass
  -> Discord review / user feedback
  -> story fix
  -> rerun
```

이 구조의 장점은, 문제가 생겼을 때 **어느 단계가 원인인지 분리해서 볼 수 있다**는 점입니다.

- 시뮬레이션이 이상한가?
- 장면 압축이 너무 거친가?
- 산문 생성이 과하게 설명적인가?
- 폴리싱이 지나치게 손을 대는가?

각 문제를 서로 다른 층에서 다룰 수 있습니다.

* * *

## 가장 중요한 명령: `!novel-daily`

처음 쓰는 사람은 다른 스크립트보다 이 명령부터 이해하는 것이 가장 좋습니다.

`!novel-daily`는 아래 일을 한 번에 처리합니다.

1. 에피소드 설정 검사
2. 캐릭터 시뮬레이션 실행
3. 챕터 생성
4. 품질 리뷰 수행
5. 사용자 피드백 수집
6. 상태 저장 및 다음 실행 준비

즉, **“한 화를 돌리고, 읽고, 고치고, 다음 화로 넘길지 결정하는 전체 작업 흐름”**이 Discord 안에서 돌아갑니다.

* * *

## 실제 Discord 흐름 예시

```text
User
!novel-daily 1

Bot
리뷰 등급을 골라주세요.
1 또는 mini      — 빠르고 저렴
2 또는 premium   — 정밀, 고품질

User
2

Bot
▶️ !novel-daily ep01_academic_presentation 시작
리뷰 등급: premium
진행 상황 확인: !status | 중단: !stop

Bot
[GUARDIAN] Config 검수 시작
...
```

사용자 입장에서는 이렇게 느껴집니다.

- 진행 상황이 Discord에 실시간으로 보인다
- 생성된 챕터를 바로 읽을 수 있다
- 마음에 안 드는 부분을 바로 수정 요청할 수 있다
- 수정 후 다시 돌려보는 루프까지 연결된다

* * *

## 입력과 출력 예시

### 입력: 에피소드 설정 파일

```yaml
episode:
  id: ep01_conference_shadow
  summary: >
    제네바 컨퍼런스 센터에서 모레노 교수의 초전도 큐비트 발표가 열린다.
    수민은 발표를 들으며 자신의 보상 회로 아이디어가 실제 시스템 수준에서
    통할 수 있다는 가능성을 처음으로 실감한다.
    발표장 곳곳에는 이 연구를 단순한 학문적 성과가 아니라 전략 자산으로
    평가하는 듯한 인물들이 보인다.
```

### 출력: 실제 소설 문장

> 2041년 5월 21일, 수민은 그날의 공기가 바뀌는 순간을 또렷하게 감지했다.
>
> 모레노의 목소리가 높은 슬라이드와 함께 잔향처럼 흘러나왔다. 수민은 통로 쪽에 비스듬히 기대어 서서 화면을 훑었다. 숫자 하나가 아니었다. 발표가 건드린 문제의 결이 그가 지난 몇 해 붙들고 있던 것과 정확히 맞닿아 있었다.
>
> 청중 가운데 어떤 이들의 시선이 달랐다. 박수 대신 계산표를 보는 눈이었다. 표준적 경외가 아니라 적용 가능성을 재는 얼굴들이었다.

즉, 이 저장소는 설정을 단순히 늘려 쓰는 것이 아니라, **시뮬레이션과 장면 구조화를 거쳐 읽을 수 있는 챕터로 변환**합니다.

* * *

## 빠른 시작

### 준비물

- Python 3.10+
- OpenAI API 키
- Discord 서버 1개
- Discord 봇 토큰 1개 이상

필수 환경 변수:

- `OPENAI_API_KEY`
- `DISCORD_BOT_TOKEN`

선택 환경 변수:

- `DISCORD_BOT_TOKEN2`
- `DISCORD_BOT_TOKEN3`
- `DISCORD_BOT_TOKEN4`
- `DISCORD_ALERT_CHANNEL_ID`

처음에는 **봇 1개만으로 시작하는 것을 권장**합니다.

### 설치

#### 1) 프로젝트 폴더로 이동

```bash
cd /path/to/storyworld-simulator
```

#### 2) 가상환경 생성

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### 3) 패키지 설치

```bash
pip install -r requirements.txt
```

### `.env` 설정

```env
OPENAI_API_KEY="sk-..."
DISCORD_BOT_TOKEN="your-main-bot-token"
DISCORD_BOT_TOKEN2=""
DISCORD_BOT_TOKEN3=""
DISCORD_BOT_TOKEN4=""
DISCORD_ALERT_CHANNEL_ID=""
```

### 첫 실행

```bash
python tools/discord_loop_bot.py
```

그다음 Discord 채널에서:

```text
!novel-daily 1
```

또는 episode key를 알고 있으면:

```text
!novel-daily ep01_academic_presentation
```

옵션 예시:

```text
!novel-daily ep01_academic_presentation --target-words 3500 --budget 4.0 --protagonist kim_sumin --review-tier premium
```

* * *

## 자주 쓰는 명령어

### 메인 실행

```text
!novel-daily 1
!novel-daily ep01_academic_presentation
```

### 상태 확인

```text
!status
```

### 중단 및 재시작

```text
!stop       # 현재 채널 파이프라인 중단
!reboot     # 봇 프로세스 재시작
!shutdown   # 봇 종료
```

### 리뷰 후 선택

리뷰가 끝나면 보통 아래 셋 중 하나를 고릅니다.

#### 1) 스토리 수정

```text
1 수민이 너무 수동적이야. 초반부터 더 적극적으로 움직였으면 좋겠어.
```

#### 2) 최적화 추가

```text
2
```

#### 3) 다음 화 진행 (그만두기)

```text
3
```

* * *

## 프로젝트 구조

```text
simulate.py                 # 단일 에피소드 시뮬레이션
generate_chapter.py         # 챕터 생성 엔트리
trial_simulate.py           # 반복 실험용 실행기

config/                     # 세계관 / 캐릭터 / 에피소드 설정
src/novel_writer/           # 코어 로직
tools/daily_pipeline.py     # !novel-daily 핵심 파이프라인
tools/discord_loop_bot.py   # Discord 봇 엔트리포인트
tests/                      # 테스트
output/                     # 생성 결과물
data/                       # 상태 / policy / DB 관련 데이터
```

처음 보는 사람은 아래 파일부터 보는 것이 가장 좋습니다.

1. `tools/discord_loop_bot.py`
2. `tools/daily_pipeline.py`
3. `simulate.py`

그다음 핵심 로직은 이 순서로 보면 이해가 빠릅니다.

1. `src/novel_writer/orchestrator.py`
2. `src/novel_writer/director.py`
3. `src/novel_writer/scene_distiller.py`
4. `src/novel_writer/prose_generator.py`
5. `src/novel_writer/polisher.py`

* * *

## 핵심 컴포넌트 설명

### 1) SimulationOrchestrator

에피소드의 turn-based 상호작용을 실제로 굴리는 엔진입니다.

역할:

- 활성 캐릭터 선택
- 턴별 컨텍스트 구성
- LLM 액션 생성
- Director 검수 통과 여부 판단
- world state 적용
- memory / clue / interaction 저장
- 완료 조건 체크

즉, **“이번 화에서 실제로 무슨 일이 벌어졌는가”**를 만듭니다.

### 2) DirectorAI

시뮬레이션이 엉뚱한 방향으로 새지 않도록 막는 감독 레이어입니다.

예를 들어 이런 것을 관리합니다.

- invariant check
- knowledge leak check
- storyline alignment
- active cast selection
- clue injection
- completion / resolution check

즉, **자유 생성은 허용하지만 스토리 가드레일은 유지**합니다.

### 3) SceneDistiller

시뮬레이션 turn log를 소설용 장면으로 압축합니다.

이 단계가 중요한 이유는, turn log는 “사건 기록”에는 좋지만 그대로는 소설 초고로 쓰기 어렵기 때문입니다.

Distiller는 다음을 수행합니다.

- protagonist perspective filtering
- scene boundary detection
- repeated turn compression
- key dialogue / actions / discoveries 추출
- YAML beat와 cross-reference
- distilled scene list 생성

즉, **30~60개 이상의 raw turn을 6~12개의 서사 장면으로 정리**합니다.

### 4) ProseGenerator

압축된 장면을 실제 소설 문장으로 바꿉니다.

주요 특징:

- scene-by-scene prose generation
- beat-aware context 사용
- transition generation
- chapter assembly
- anchor preservation
- chapter polishing 연동

중요한 점은, 이 경로가 예전의 `novel_generator.py`처럼 raw turn log에서 바로 소설을 만드는 방식보다 **더 압축되고 더 통제된 입력**을 사용한다는 것입니다.

### 5) ChapterPolisher

최종 장의 문장을 정리하는 bounded polishing 단계입니다.

이 단계는 의도적으로 제한되어 있습니다.

- broad polish pass
- anchor coverage correction pass
- reader feedback final pass
- deterministic cleanup / normalization

즉, **장면 구조를 뒤엎지 않고 문장을 다듬는 역할**에 집중합니다.

### 6) Runtime Policy (`rl_policy.py`)

튜닝 가능한 런타임 파라미터를 프롬프트 밖으로 분리해 둔 레이어입니다.

예:

- scene target min / max
- distiller temperature
- prose scene / transition / polish temperature
- history handling
- fallback / guard 관련 값

즉, “프롬프트를 계속 수동 수정”하는 대신 **조정 가능한 정책값으로 실험**할 수 있게 설계되어 있습니다.

* * *

## 이 프로젝트를 이해할 때 중요한 포인트

### 이 프로젝트는 “좋은 프롬프트 하나”가 아니다

핵심은 좋은 프롬프트 하나가 아니라, **반복 가능한 제작 루프**입니다.

### 이 프로젝트는 “사건 생성”과 “문장 생성”을 분리한다

- 시뮬레이션 = 사건/상호작용 생성
- distillation = 장면 구조화
- prose generation = 읽히는 문장 생성
- polishing = 최종 품질 보정

### Discord는 단순 채팅창이 아니라 운영 인터페이스다

Discord 안에서:

- 실행하고
- 상태를 보고
- 결과를 읽고
- 피드백을 주고
- 재실행을 결정합니다

즉, 사용자 경험의 중심은 CLI가 아니라 **Discord 운영 루프**입니다.

* * *

## 처음 기여하거나 분석할 때 추천 순서

### 사용 흐름부터 이해하고 싶다면

1. `README.md`
2. `tools/discord_loop_bot.py`
3. `tools/daily_pipeline.py`

### 시뮬레이션 구조를 보고 싶다면

1. `src/novel_writer/orchestrator.py`
2. `src/novel_writer/director.py`

### 소설 생성 품질 문제를 보고 싶다면

1. `src/novel_writer/scene_distiller.py`
2. `src/novel_writer/prose_generator.py`
3. `src/novel_writer/polisher.py`

### 튜닝 / 실험 포인트를 보고 싶다면

1. `rl_policy.py`
2. review / feedback 관련 코드
3. tests / trial runner

* * *

## 자주 막히는 문제

### 봇은 켜졌는데 `!novel-daily`에 반응이 없다

확인할 것:

- Discord bot의 `MESSAGE CONTENT INTENT`가 켜져 있는지
- 봇이 해당 채널을 읽을 권한이 있는지
- 올바른 서버 / 채널에서 테스트 중인지

### `Set DISCORD_BOT_TOKEN in .env` 오류

원인:

- `.env` 파일이 없거나
- `DISCORD_BOT_TOKEN` 값이 비어 있음

### `Set OPENAI_API_KEY in .env` 오류

원인:

- API 키가 없거나
- `OPENAI_API_KEY` 값이 비어 있음

* * *

## 현재 권장 사용 방식

현재 구조상 권장 경로는 다음과 같습니다.

- **Discord에서 `!novel-daily`로 실행**
- **scene distillation + prose generation 경로 사용**
- **feedback loop를 통해 스토리 수정 또는 추가 최적화 후 재실행**

즉, 이 저장소는 단순한 챕터 생성 스크립트 모음이 아니라, **연재형 소설을 운영하고 개선하기 위한 제작 시스템**으로 보는 것이 가장 정확합니다.

* * *

## 한 줄 요약

**Storyworld Simulator는 에피소드 설정 파일을 입력으로 받아, 캐릭터 시뮬레이션과 장면 압축을 거쳐 읽을 수 있는 소설 챕터를 만들고, Discord 안에서 리뷰와 재실행까지 이어 주는 AI 소설 제작 파이프라인입니다.**

* * *

## API 사용 지도

파이프라인 1회 실행 시 어떤 단계에서 어떤 모델을 사용하는지 정리한 참조 문서입니다.

### 티어 선택 로직

실행 시 `review_tier`를 `premium`(기본) 또는 `mini`로 지정하면 일부 모델이 달라집니다.

| 구분 | mini 티어 | premium 티어 (기본) |
|---|---|---|
| 글쓰기 모델 | `gpt-4.1-mini` | `gpt-5-mini` |
| 리뷰 승격 | `gpt-5-mini` 유지 | `gpt-4o` 로 승격 |
| 피드백 파싱 | `gpt-4o-mini` 유지 | `gpt-4o` 로 승격 |

### 단계별 모델 배정

| 단계 / 역할 | premium 티어 | mini 티어 |
|---|---|---|
| **Guardian** 컨텍스트 분석 + 브리핑 | `gpt-5.4` | `gpt-5.4` |
| **Simulator** 에이전트 턴 + 디렉터 체크 | `gpt-4o-mini` | `gpt-4o-mini` |
| **글쓰기 전체** (씬 증류·산문·전환·폴리싱) | `gpt-5-mini` | `gpt-4.1-mini` |
| **Optuna 채점** GPT (8항목 × 3회/trial) | `gpt-4o-mini` | `gpt-4o-mini` |
| **Optuna 채점** Claude 앙상블 (편향 보정) | `claude-haiku-4-5-20251001` | `claude-haiku-4-5-20251001` |
| **리뷰** (베이스라인·사이클·최종) | `gpt-4o` | `gpt-5-mini` / `gpt-4o-mini` |
| **피드백 파싱** → story_state 업데이트 | `gpt-4o` | `gpt-4o-mini` |
| **Meitner 봇** 플랜 생성·코드베이스 질의 | `gpt-5-mini` | `gpt-4o-mini` |

> Guardian 캐시 히트 시 GPT 호출 없음 (YAML 해시 기반 캐시).

### 1회 실행 시 API 호출 횟수 추정 (outer 3회 기준)

| API | 호출 수 (최대) |
|---|---|
| `gpt-5.4` (Guardian) | **1회** (캐시 미스 시) |
| `gpt-4o-mini` (Optuna 채점) | **225회** (25 trials × 3runs × 3outer) |
| `claude-haiku-4-5-20251001` (앙상블) | **225회** (동일) |
| `gpt-5-mini` / `gpt-4.1-mini` (글쓰기) | **75회** (25 × 3outer) + 리뷰 3–4회 |
