# Storyworld Simulator

YAML 기반 세계관/인물/에피소드를 바탕으로 시뮬레이션 로그를 만들고, 이를 소설 챕터로 변환하는 프로젝트입니다.

## What This Does

- `config/*.yaml`로 스토리/세계관/인물 정의
- `simulate.py` 또는 `trial_simulate.py`로 턴 기반 상호작용 생성
- `generate_chapter.py`로 DB 로그를 장면 압축 후 prose 챕터 생성
- `tools/*`로 품질 점검/개선 루프 실행

## Project Layout

```text
simulate.py                 # 단일 에피소드 실행 CLI
trial_simulate.py           # Trial-and-Learn 실행 CLI
generate_chapter.py         # 챕터 생성 CLI

src/novel_writer/           # 코어 엔진 패키지
  config_loader.py
  database.py
  director.py
  llm_client.py
  models.py
  orchestrator.py
  prose_generator.py
  scene_distiller.py
  trial_runner.py

tools/                      # 품질 도구
  quality_analyzer.py
  quality_adaptive_generator.py
  compare_quality.py
  run_quality_pipeline.sh

tests/                      # 테스트/검증 스크립트
config/                     # YAML 설정
output/                     # 생성 산출물
reports/                    # 분석/리포트 문서
examples/Good_example/      # 레퍼런스 챕터
```

## Requirements

- Python 3.10+
- OpenAI API Key

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
```

## Quick Start

### 1) Single Episode Simulation

```bash
python simulate.py \
  --episode config/episodes/ep05_unexpected_visitors.yaml \
  --characters config/characters.yaml \
  --world config/world_facts.yaml \
  --storyline config/storyline.yaml \
  --budget 5.0
```

### 2) Generate Chapter

```bash
python generate_chapter.py \
  --episode ep05_unexpected_visitors \
  --episode-config config/episodes/ep05_unexpected_visitors.yaml \
  --protagonist kim_sumin \
  --protagonist-name "Kim Sumin" \
  --words 2000
```

### 3) Trial-and-Learn (Optional)

```bash
python trial_simulate.py \
  --episode config/episodes/ep05_unexpected_visitors.yaml \
  --characters config/characters.yaml \
  --world config/world_facts.yaml \
  --storyline config/storyline.yaml \
  --max-trials 5 \
  --budget 15.0
```

### 4) Quality Pipeline (Optional)

```bash
bash tools/run_quality_pipeline.sh ep05_unexpected_visitors
```

## Typical Flow

1. `simulate.py` 또는 `trial_simulate.py` 실행
2. `output/<episode_id>_simulation.json`, `output/<episode_id>_debug.log` 생성
3. `generate_chapter.py` 실행
4. `output/<episode_id>_scenes.json`, `output/<episode_id>_chapter.md` 생성
5. 필요 시 `tools/quality_analyzer.py`로 품질 분석

## Key Concepts

### Director Guardrails

`src/novel_writer/director.py`에서 매 턴 다음을 검사합니다.

- 캐릭터 invariant 위반
- knowledge leak
- storyline/milestone alignment
- clue injection 필요 여부
- episode 종료 조건 충족 여부

### Casting Priority

1. `episode.characters` 명시값
2. 에피소드 텍스트에서 명시적으로 언급된 인물
3. 이전 에피소드 캐스팅 승계
4. 최소 fallback cast

## CLI Reference

### `simulate.py`

- 목적: 단일 에피소드 실행
- 주요 인자:
  - `--episode` `--characters` (필수)
  - `--world` `--storyline`
  - `--model` `--premium`
  - `--budget` `--output` `--db`

### `trial_simulate.py`

- 목적: 실패 원인 학습 + 재시도
- 주요 인자:
  - `--max-trials`
  - `--success-threshold`
- 산출물:
  - `output/<episode_id>_trial_summary.json`
  - `output/<episode_id>_trialN_simulation.json`
  - `output/<episode_id>_trialN_debug.log`

### `generate_chapter.py`

- 목적: DB 상호작용 로그 -> 장면 압축 -> 챕터 생성
- 주요 인자:
  - `--episode` `--episode-config`
  - `--protagonist` `--protagonist-name`
  - `--words` `--scenes` `--style`

## Output Files

- `output/<episode_id>_simulation.json`
- `output/<episode_id>_debug.log`
- `output/<episode_id>_scenes.json`
- `output/<episode_id>_chapter.md`
- `output/<episode_id>_trial_summary.json` (trial 사용 시)

## Notes

- 스토리 데이터는 코드가 아니라 `config/*.yaml`에서 관리됩니다.
- 품질 실험 스크립트는 `tools/`에 모아두었습니다.
- 코어 로직은 `src/novel_writer` 패키지에 통합되어 있습니다.
