# Novel Writer

에피소드 설정 파일을 넣으면, **소설 초고를 만들고 Discord 안에서 계속 개선해 나가는** 파이프라인입니다.

이 저장소의 핵심 진입점은 `!novel-daily`입니다.  
처음 보는 사람도 이 명령 하나만 이해하면 전체 구조를 빠르게 파악할 수 있습니다.

![Discord sample](Discord_sample.png)

---

## 한눈에 보기

이 프로젝트는 단순한 텍스트 생성기가 아닙니다.

- 에피소드 설정(YAML)을 읽고
- 캐릭터 상호작용을 시뮬레이션하고
- 장면을 압축한 뒤
- 한국어 소설 챕터로 생성하고
- 자동 리뷰와 사용자 피드백을 받아
- 코드 또는 스토리 설정까지 다시 수정하는

**“생성 → 리뷰 → 개선 → 재실행” 루프**를 Discord 안에서 돌립니다.

핵심적으로는 이렇게 이해하면 됩니다.

> **설정 파일 하나를 넣으면 초고를 만들고, Discord에서 읽고, 고치고, 다음 화로 넘길 수 있는 시스템**

---

## 왜 흥미로운가

보통 소설 초고를 다듬으려면 이런 작업을 따로 해야 합니다.

1. 설정 파일 확인
2. 시뮬레이션 실행
3. 챕터 생성
4. 품질 리뷰
5. 피드백 정리
6. 코드 또는 스토리 수정
7. 다시 실행

`!novel-daily`는 이 과정을 하나의 루프로 묶습니다.

```text
episode.yaml
  -> Config Guardian
  -> 시뮬레이션
  -> 챕터 생성
  -> 자동 리뷰
  -> 리포트 생성
  -> Discord 피드백 수집
  -> 코드 수정 또는 스토리 수정
  -> 자동 재실행
```

사용자 입장에서는 이렇게 느껴집니다.

- 진행 상황이 Discord에 실시간으로 보인다
- 생성된 챕터를 바로 읽을 수 있다
- 마음에 안 드는 부분을 바로 수정 요청할 수 있다
- 수정 후 다시 돌려보는 루프까지 연결된다

---

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

즉, 이 저장소는 설정을 단순히 늘려 쓰는 것이 아니라,
**시뮬레이션과 장면 구조화를 거쳐 읽을 수 있는 챕터로 변환**합니다.

---

## 가장 중요한 명령: `!novel-daily`

처음 쓰는 사람은 다른 스크립트보다 이 명령부터 이해하는 것이 가장 좋습니다.

`!novel-daily`는 아래 일을 한 번에 처리합니다.

1. 에피소드 설정 검사
2. 캐릭터 시뮬레이션 실행
3. 챕터 생성
4. 품질 리뷰 및 자동 개선
5. 사용자 피드백 반영
6. 상태 저장 및 다음 실행 준비

즉,
**“한 화를 돌리고, 읽고, 고치고, 다음으로 넘길지 결정하는 전체 작업 흐름”**이 Discord 안에서 돌아갑니다.

---

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

Bot
[SIM] 시뮬레이션 시작

Bot
[SIM] Turn 7/28

Bot
[CHAPTER] 챕터 생성 중

Bot
[AUTO] AI 자동 개선 루프 1/20 시작

Bot
[REVIEW] 자동 검수 완료
긴장감: 8/10
문체: 8/10
인과성: 7/10

Bot
개선 방향을 선택해주세요.
1 코드 수정
2 스토리 수정
3 다음으로

User
1 초반 대사가 조금 딱딱하고 설명투야. 긴장감을 더 빨리 올려줘.

Bot
[CHOICE] 코드 수정 선택 — Codex Fixer 실행 중...

Bot
[CHAPTER] 수정된 코드로 챕터 재생성 중...

Bot
[DONE] 코드 수정 완료. 같은 화 재시도: !novel-daily ep01_academic_presentation
```

이 예시에서 핵심은 세 가지입니다.

- 사용자는 Discord에서 명령만 입력하면 됩니다.
- 봇은 실행, 리뷰, 개선, 재실행을 하나의 흐름으로 이어줍니다.
- 마지막에는 “코드 수정 / 스토리 수정 / 다음 화 진행”을 다시 선택할 수 있습니다.

진행 중 상태가 궁금하면 `!status`로 확인할 수 있습니다.

```text
User
!status

Bot
현재 파이프라인 상태: 챕터 생성 중
경과 시간: 6분 12초
세션 누적 토큰: ...
세션 누적 비용: ...
```

---

## 빠른 시작

정말 처음이라면 아래 순서만 따라오면 됩니다.

1. Python 환경 준비
2. OpenAI API 키 발급
3. Discord 봇 생성 및 서버 초대
4. `.env` 설정
5. `python tools/discord_loop_bot.py` 실행
6. Discord 채널에서 `!novel-daily 1` 입력

---

## 준비물

- Python 3.10 이상
- OpenAI API 키
- Discord 서버 1개
- Discord 봇 토큰 1개 이상

반드시 필요한 값:

- `OPENAI_API_KEY`
- `DISCORD_BOT_TOKEN`

---

## 설치

### 1) 프로젝트 폴더로 이동

```bash
cd "/Users/saesunkim/Documents/Novel Writter - 2026"
```

### 2) 가상환경 생성

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 3) 패키지 설치

```bash
pip install -r requirements.txt
```

---

## `.env` 설정

먼저 예제 파일을 복사합니다.

```bash
cp .env.example .env
```

그다음 `.env`를 아래처럼 채웁니다.

```env
OPENAI_API_KEY="sk-..."
DISCORD_BOT_TOKEN="your-main-bot-token"
DISCORD_BOT_TOKEN2=""
DISCORD_BOT_TOKEN3=""
DISCORD_BOT_TOKEN4=""
```

설명:

- `OPENAI_API_KEY`: 필수
- `DISCORD_BOT_TOKEN`: 필수
- `DISCORD_BOT_TOKEN2/3/4`: 선택

처음에는 **봇 하나만** 설정하는 것을 권장합니다.

---

## Discord 봇 설정

이 단계에서 가장 자주 막히는 포인트는 두 가지입니다.

1. 봇을 서버에 초대하지 않음
2. `MESSAGE CONTENT INTENT`를 켜지 않음

### 1) Application 생성

- `https://discord.com/developers/applications` 접속
- `New Application` 클릭
- 예: `Novel Writer Daily`

### 2) Bot 생성

- 왼쪽 메뉴 `Bot`
- `Add Bot`

### 3) 토큰 발급

- 같은 `Bot` 화면에서 토큰 확인
- `.env`의 `DISCORD_BOT_TOKEN`에 입력

### 4) Message Content Intent 켜기

이 프로젝트는 슬래시 커맨드가 아니라 일반 메시지 명령을 읽습니다.
예:

- `!novel-daily 1`
- `!status`
- `!stop`

따라서 `message.content`를 읽을 수 있도록 반드시 아래를 켜야 합니다.

- `Bot` 메뉴
- `Privileged Gateway Intents`
- `MESSAGE CONTENT INTENT` 활성화

이걸 빠뜨리면 봇은 접속되지만 명령을 읽지 못합니다.

### 5) 서버에 초대하기

`OAuth2 -> URL Generator`에서:

- Scope: `bot`
- 권장 권한:
  - View Channels
  - Send Messages
  - Read Message History
  - Add Reactions
  - Attach Files
  - Create Public Threads
  - Send Messages in Threads

---

## 첫 실행

가장 쉬운 첫 명령은 이것입니다.

```text
!novel-daily 1
```

에피소드 키를 알고 있다면 이렇게도 가능합니다.

```text
!novel-daily ep01_academic_presentation
```

옵션까지 함께 주는 예:

```text
!novel-daily ep01_academic_presentation --target-words 3500 --budget 4.0 --protagonist kim_sumin --review-tier premium
```

자주 쓰는 옵션:

- `--target-words 3500`: 목표 분량
- `--budget 4.0`: 실행 예산
- `--protagonist kim_sumin`: 주인공 ID
- `--review-tier mini|premium`: 리뷰 방식

---

## 리뷰 티어

`--review-tier`를 생략하면 Discord에서 먼저 물어봅니다.

```text
1 또는 mini      -> 빠르고 저렴
2 또는 premium   -> 더 정밀
```

### 모델 구성

| 단계 | mini | premium |
|---|---|---|
| 시뮬레이션 — 에이전트 턴 | gpt-4o-mini | gpt-4o-mini |
| 시뮬레이션 — 디렉터/산문 | gpt-5-mini | gpt-5-mini |
| 챕터 생성 — 기본 구성 | gpt-4o-mini | gpt-4o-mini |
| 챕터 생성 — 산문 생성 | gpt-4.1-mini | gpt-5-mini |
| Guardian 분석 | gpt-4o-mini | gpt-4o |
| Quality Reviewer | gpt-4o-mini | gpt-4o |
| AI 루프 리뷰 | gpt-5-mini | gpt-4o |
| Codex 코드 수정 | Codex CLI (`gpt-5.4-mini`) | Codex CLI (config 기본값) |
| Regen 판단 LLM | gpt-4o-mini | gpt-4o |
| Feedback 파싱 LLM | gpt-4o-mini | gpt-4o |

메모:

- mini 티어는 비용을 줄이면서도 long-context 품질을 확보하도록 조정되어 있습니다.
- premium 티어는 리뷰와 판단 품질을 더 높게 가져갑니다.

---

## 실행 중 무슨 일이 일어나는가

`!novel-daily`를 실행하면 대략 아래 순서로 움직입니다.

```text
episode config
  -> Config Guardian
  -> Simulator
  -> Chapter Generator
  -> Auto Review / Auto Improve Loop
  -> Final Review
  -> 사용자 선택:
       1 코드 수정
       2 스토리 수정
       3 다음으로 진행
  -> story_state.json 업데이트
```

실행 중에는 Discord 채널과 쓰레드에 단계별 메시지가 올라오고,
완료 후에는 챕터 파일과 리포트 이미지도 받을 수 있습니다.

### 단계별 봇 역할

| 봇 | 토큰 | 담당 메시지 |
|---|---|---|
| Simulator | TOKEN1 | `[SIM]`, `[CHAPTER]` |
| Reader | TOKEN2 | `[GUARDIAN]`, `[REVIEW]`, `[AUTO] 📊 리뷰 결과` |
| Programmer | TOKEN3 | `[AUTO]` 루프, `[FIXER]` |
| Manager | TOKEN4 | `[START]`, `[MANAGER]`, `[CHOICE]`, `[WAIT]`, `[DONE]` |

`TOKEN2/3/4`를 따로 설정하지 않으면 모두 메인 봇이 대신 보냅니다.

---

## 사용자가 최종적으로 할 수 있는 선택

리뷰가 끝나면 보통 세 가지 중 하나를 고르게 됩니다.

### 1) 코드 수정

예:

```text
1 초반 대사가 너무 딱딱해. 문장이 자주 끊겨.
```

이 경우:

- Codex가 코드 쪽을 수정합니다.
- 챕터를 다시 생성합니다.
- 필요하면 자동 재실행을 이어갑니다.

### 2) 스토리 수정

예:

```text
2 수민이 너무 수동적이야. 초반부터 더 적극적으로 움직였으면 좋겠어.
```

이 경우:

- 에피소드 YAML을 수정합니다.
- 수정 후 자동으로 YAML 문법 검수가 실행됩니다.

### 3) 다음으로 진행

예:

```text
3
```

이 경우:

- 현재 결과를 승인합니다.
- 다음 에피소드로 넘어갈 준비를 합니다.

---

## 자주 쓰는 명령어

### `!novel-daily`

메인 명령입니다.

```text
!novel-daily 1
!novel-daily ep01_academic_presentation
!novel-daily ep01_academic_presentation --target-words 3200 --budget 4.0 --review-tier premium
```

### `!status`

현재 파이프라인 상태를 확인합니다.

- 현재 단계
- 경과 시간
- 누적 토큰
- 누적 비용
- 백그라운드 프로세스 상태

### `!stop`

현재 파이프라인 중단 요청을 보냅니다.
중단 직후 지금까지의 실행 결과를 요약한 이미지가 자동 생성됩니다.

### `!chapter`

가장 최근 생성된 챕터 파일을 다시 받아옵니다.

### `!benchmark [episode_key]`

지금까지 실행된 품질 점수 추이를 정리해서 보여줍니다.

```text
!benchmark
!benchmark ep01
```

### `!meitner <질문>`

저장소 구조를 질문하는 도우미 명령입니다.

```text
!meitner daily pipeline이 어디서 시작돼?
```

### `!approve <req_id>` / `!reject <req_id>`

고급 기능입니다. 승인 대기 중인 설정 변경 요청을 처리할 때 사용합니다.

---

## 생성되는 파일들

`!novel-daily`를 돌리면 보통 아래 경로에 실행 결과가 저장됩니다.

```text
output/daily/YYYYMMDD_<episode_key>/HHMMSS/
```

예:

```text
output/daily/20260320_ep01_academic_presentation/013820/
```

이 안에는 다음과 같은 파일들이 생길 수 있습니다.

- `config_check.md`
- `*_simulation.json`
- `*_debug.log`
- `*_chapter.txt`
- `scorecard.txt`
- `auto_review_cycle*.json`
- `pipeline_report.png`

상태 파일:

- `data/story_state.json`
- `data/pending_config_changes.json`

---

## 결과 이미지 예시

이 프로젝트는 텍스트만 주지 않고 실행 결과를 요약한 이미지도 생성할 수 있습니다.

예를 들어 `pipeline_report.png`에는 이런 내용이 들어갈 수 있습니다.

- 사이클별 품질 점수 추이
- 단계별 비용
- 단계별 소요 시간

![Sample pipeline report](docs/assets/readme-pipeline-report.png)

---

## Discord 없이 로컬에서만 테스트하기

Discord를 붙이기 전에 로컬에서 내부 로직만 점검할 수도 있습니다.

### 단일 시뮬레이션

```bash
python simulate.py \
  --episode config/episodes/ep05_unexpected_visitors.yaml \
  --characters config/characters.yaml \
  --world config/world_facts.yaml \
  --storyline config/storyline.yaml \
  --budget 5.0
```

### 챕터 생성

```bash
python generate_chapter.py \
  --episode ep05_unexpected_visitors \
  --episode-config config/episodes/ep05_unexpected_visitors.yaml \
  --protagonist kim_sumin \
  --protagonist-name "Kim Sumin" \
  --words 2000
```

### Daily pipeline만 실행

```bash
python tools/daily_pipeline.py --episode ep01_academic_presentation --no-discord
```

---

## 저장소 구조

```text
simulate.py                 # 단일 에피소드 시뮬레이션
generate_chapter.py         # 챕터 생성
trial_simulate.py           # 반복 실험용 실행기

config/                     # 세계관, 캐릭터, 에피소드 설정
src/novel_writer/           # 코어 로직
tools/daily_pipeline.py     # !novel-daily 핵심 파이프라인
tools/discord_loop_bot.py   # Discord 봇 엔트리포인트
tests/                      # 테스트
output/                     # 실행 산출물
data/                       # 상태 파일
```

처음 보는 사람은 아래 세 파일만 먼저 보면 흐름을 파악하기 쉽습니다.

- `tools/discord_loop_bot.py`
- `tools/daily_pipeline.py`
- `simulate.py`

---

## 자주 막히는 문제

### 봇은 켜졌는데 `!novel-daily`에 반응이 없다

가장 흔한 원인:

- `MESSAGE CONTENT INTENT`를 켜지 않음
- 봇이 채널 메시지를 볼 권한이 없음
- 잘못된 채널에서 테스트 중

### `Set DISCORD_BOT_TOKEN in .env` 오류

원인:

- `.env` 파일이 없거나
- `DISCORD_BOT_TOKEN` 값이 비어 있음

해결:

```bash
cp .env.example .env
```

후에 토큰을 정확히 넣습니다.

### `Set OPENAI_API_KEY in .env` 오류

원인:

- API 키가 없거나
- `OPENAI_API_KEY`가 비어 있음

해결:

- OpenAI API 키를 발급받아 `.env`에 넣습니다.

### 파일 업로드가 안 된다

가능성:

- `Attach Files` 권한 없음

### 쓰레드 생성이 안 된다

가능성:

- `Create Public Threads`
- `Send Messages in Threads`

권한 중 하나 이상이 없음

### 같은 채널에서 상태가 꼬이는 것 같다

이유:

- 이 봇은 채널 단위로 상태를 관리합니다.

권장:

- 실험을 분리하려면 채널도 분리하세요.

---

## 고급 내부 동작

이 섹션은 처음 읽는 사람에게 꼭 필요하지는 않지만,
이 프로젝트가 단순 생성기가 아니라 **자기개선형 파이프라인**이라는 점을 보여줍니다.

### AUTO 루프

자동 개선 루프는 최대 20사이클까지 돌아갑니다.

- 리뷰 점수 계산
- Manager 분석
- Codex 코드 수정
- 챕터 재생성
- 점수 비교

평균 점수가 8.5 이상이면 조기 종료됩니다.

### Codex Fixer

Codex Fixer가 수정한 내용은 실제 소스 파일에 반영됩니다.
예를 들어 아래 파일들이 직접 수정될 수 있습니다.

- `prose_generator.py`
- `scene_distiller.py`
- `director.py`

따라서 `git status`로 변경 내역을 확인하고,
마음에 드는 결과가 나오면 직접 `git commit`으로 기록하는 것이 좋습니다.

### 5사이클마다 심층 회고

매니저는 단순히 현재 점수만 보는 것이 아니라,
사이클 간 점수 변화와 코드 변경 이력을 함께 추적합니다.

5사이클마다:

1. 누적 코드 diff 분석
2. 점수 이력 정리
3. 강한 수정 방향 제시

를 수행해 다음 Codex 수정 방향을 더 강하게 제한합니다.

즉, 같은 실패를 반복하기보다
**“어떤 종류의 수정이 실제로 점수 향상과 연결됐는가”**를 학습해 가는 구조입니다.

---

## 처음이라면 이렇게 시작하는 것을 권장

1. `.env`에 `OPENAI_API_KEY`와 `DISCORD_BOT_TOKEN`만 넣기
2. 봇 1개만 만들어 서버에 초대하기
3. `MESSAGE CONTENT INTENT` 켜기
4. `python tools/discord_loop_bot.py` 실행하기
5. Discord에서 `!novel-daily 1` 입력하기
6. 잘 돌면 그다음에 `premium`, 다중 봇 토큰, 세부 운영 방식으로 확장하기

처음부터 reviewer / fixer / manager 봇을 모두 분리하면 설정 포인트가 많아집니다.
먼저 단일 봇으로 한 번 성공시키고, 그다음 확장하는 편이 훨씬 쉽습니다.

---

## 최근 변경사항

### 2026-03-20

#### 리뷰 티어 단순화 — `codex` 폐지

ChatGPT Pro 한도 도달 시 파이프라인 중간 실패 문제로 `codex` 티어를 폐지했습니다.
이제 리뷰 티어는 `mini` / `premium` 두 가지만 있습니다.
코드 수정용 Codex CLI는 두 티어 모두에서 그대로 사용됩니다.

#### mini 티어 모델 최적화

| 단계 | 변경 전 | 변경 후 | 이유 |
|---|---|---|---|
| 챕터 산문 생성 | gpt-4o-mini | gpt-4.1-mini | long-context 성능 향상, 비용 절감 |
| AI 루프 리뷰 | gpt-4o-mini | gpt-5-mini | 문학적 판단 품질 유지 → 사이클 수 감소 |
| Codex 코드 수정 | gpt-5.1-codex-mini | gpt-5.4-mini | 더 안정적인 모델로 교체 |

#### Discord 429 Rate Limit 재시도

`[FIXER] ⚙️` 메시지가 Simulator / Programmer 봇을 번갈아 출력하던 문제를 수정했습니다.
이제 `retry_after` 대기 후 최대 3회 재시도합니다.

#### 봇 라우팅 정확도 개선

| 메시지 | 변경 전 | 변경 후 |
|---|---|---|
| `[FIXER] ⚙️` 수정 과정 | Simulator + Programmer 혼용 | Programmer(TOKEN3) 전용 |
| `[AUTO] 📊 AI 리뷰 결과` | Programmer | Reader(TOKEN2) |
| `[AUTO] 🚀 루프 시작` | 앵커+쓰레드 | 채널 직접 메시지 |
| `[GUARDIAN] 💸` 비용 | 채널 직접 | Reviewer 봇 guardian 쓰레드 |
| `[RESET] ♻️` | Simulator | Programmer 봇 + 쓰레드 |
| `[AUTO] 📊` ✅ 완료 반응 | 누락 | 정상 처리 |

#### `!benchmark` 명령어 추가

`output/daily/`의 모든 실행 결과를 스캔해 점수 추이 테이블과 차트를 Discord에 전송합니다.

```text
!benchmark
!benchmark ep01
```

#### `[AUTO] 📊` 점수 이모지

평균 점수 기반 이모지가 붙습니다.

- `🏆` 목표 달성
- `🌟` 7.5+
- `🎯` 6.5+
- `🧪` 5.5+
- `📉` 그 이하

#### 동적 씬 수 자동 계산

| 목표 단어수 | 최소 씬 수 |
|---|---|
| 3,500단어 | 4장면 |
| 6,000단어 | 6장면 |
| 8,000단어 | 8장면 |

#### 스토리 수정 후 YAML 자동 검수

스토리 수정(선택 2) 요청 시 `config/episodes/` 전체를 자동 검수합니다.
단순 문법 오류는 자동 수정하고, 복잡한 오류는 파일명과 내용을 알려줍니다.

#### Discord 로그 클린 출력

Discord에는 Python 내부 로그 대신 한국어 요약 메시지만 표시됩니다.

```text
이전: 10:07:40 [INFO] generate_chapter: Scene data → /Users/.../ep01_scenes.json
이후: [CHAPTER] 🧩 Stage 1 | 씬 분석 완료
```
