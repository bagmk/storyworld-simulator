# 코드 수정 후 전체 검수 프롬프트

> 코드를 수정한 뒤 아래 프롬프트를 Claude Code에 그대로 붙여넣으면 됩니다.

---

## 사용법

Claude Code에서 아래 한 줄 입력:

```
REVIEW_CHECKLIST.md 파일 안의 검수 프롬프트를 실행해줘
```

---

## 검수 프롬프트 (복사해서 사용)

```
다음 24가지 항목을 순서대로 전부 검수해줘.
코드를 직접 읽고 확인해야 하며, 추측으로 답하지 마.
문제가 있으면 파일명:라인번호와 함께 구체적으로 알려줘.

─────────────────────────────────────────────────────────
[사전 체크] 변경 파일 분석 → 우선 검수 항목 선정
─────────────────────────────────────────────────────────

본 검수를 시작하기 전에 먼저 아래를 실행해줘.

1. `git diff --name-only HEAD` 로 이번에 수정된 파일 목록을 확인해줘.

2. 수정된 파일을 아래 매핑표와 대조해서 관련 항목 번호를 추려줘:

   수정된 파일                         → 우선 검수 항목
   ──────────────────────────────────────────────────────
   discord_loop_bot.py                 → 1, 2, 3, 4, 5, 13, 16, 24
   daily_pipeline.py                   → 1, 3, 4, 7, 9, 10, 14, 23
   inline_optimizer.py                 → 1, 7, 9, 11, 12, 15
   src/novel_writer/prose_generator.py → 7, 8, 14
   src/novel_writer/scene_distiller.py → 7, 8, 14
   src/novel_writer/polisher.py        → 7, 8, 14
   src/novel_writer/director.py        → 7, 8, 14
   src/novel_writer/llm_client.py      → 9, 11
   tools/optuna_multi_study.py         → 12
   tools/optuna_prose_test.py          → 12
   data/rl_policy.json                 → 7
   tests/test_reader_feedback_guards.py → 14
   README.md                           → 6

3. 결과를 이렇게 출력해줘:
   "변경된 파일: X개
    우선 검수 항목: N, N, N, ... (총 N개)
    나머지 항목: N, N, N, ... (총 N개)"

4. **우선 검수 항목을 먼저 실행**하고, 결과를 출력해줘.
   우선 항목에서 ❌ 크래시 위험이 발견되면 즉시 보고하고 나머지 항목 계속 진행해줘.

5. 우선 검수가 끝난 뒤 나머지 항목을 순서대로 실행해줘.

─────────────────────────────────────────────────────────

─────────────────────────────────────────────────────────
항목 1. 디스코드 명령어 데이터 연동 확인
─────────────────────────────────────────────────────────

아래 각 명령어가 읽는 데이터 소스를 확인하고,
현재 파이프라인(daily_pipeline.py, inline_optimizer.py)이
실제로 그 소스에 데이터를 쓰고 있는지 검증해줘.

- !status     → DAILY_STATUS, DAILY_PROCESS_INFO, DAILY_SESSION_METRICS
- !usage      → DAILY_SESSION_METRICS (spent_usd, token counts)
- !benchmark  → data/cycle_score_log.jsonl 또는 SESSION_BENCHMARK_LOG
- !parameter  → data/rl_policy.json + data/cycle_score_log.jsonl
                GROUPS에 정의된 파라미터 키가 rl_policy.json에 모두 존재하는지
                rl_policy.json에 있는데 GROUPS에 없는 키가 있는지

확인 포인트:
- 데이터를 쓰는 키 이름과 읽는 키 이름이 일치하는가
- None 또는 빈 dict일 때 크래시 없이 처리되는가

─────────────────────────────────────────────────────────
항목 2. Discord 쓰레드 생성 및 라우팅 확인
─────────────────────────────────────────────────────────

discord_loop_bot.py에서:
- anchor_threads 딕셔너리에 정의된 키 목록 확인
- _anchor_key_for_text() 함수가 어떤 텍스트를 어떤 키로 분류하는지 확인
- _thread_route_for_text() 함수가 어떤 텍스트를 어떤 쓰레드로 라우팅하는지 확인
- daily_pipeline.py에서 notify()로 전송하는 주요 메시지들이
  위 라우팅 함수와 잘 매칭되는지 확인
- 쓰레드가 생성되지 않은 채로 라우팅 시도해서 NoneType 에러가 날 수 있는 경로가 있는지 확인

─────────────────────────────────────────────────────────
항목 3. !novel-daily 전체 통합 및 충돌 확인
─────────────────────────────────────────────────────────

discord_loop_bot.py → run_daily_pipeline() 호출부를 읽고:
- 호출 인자(파라미터명, 타입, 순서)가 daily_pipeline.py 함수 시그니처와 완벽히 일치하는지
- outer_max_cycles 입력 → 파싱 → 전달 → 실제 사용까지 전 경로 추적
- review_tier 값이 mini/premium 이외의 값이 들어올 경우 처리되는지
- 같은 채널에서 !novel-daily 중복 실행 시 충돌 방지 로직이 있는지
- stop_event가 올바르게 초기화되고 재사용되는지

─────────────────────────────────────────────────────────
항목 4. !novel-daily 실행 플랜 메시지 정확성 확인
─────────────────────────────────────────────────────────

daily_pipeline.py에서 [AUTO] 📋 학습 계획 브리핑 메시지를 찾아서:
- Phase A trial 수가 AUTO_BATCH_TRIALS 상수와 일치하는지
- Phase B 설명(AI 리뷰 + Factor Analysis)이 실제 코드 흐름과 일치하는지
- Phase C 설명(코드 수정 최대 N회)이 AUTO_INNER_MAX_CYCLES와 일치하는지
- outer cycle 수가 사용자 입력값(_outer_max)으로 정확히 표시되는지
- 총 최대 챕터 생성 계산식이 실제 루프 구조와 일치하는지
- YAML 자동 피드백(Fix E) 같은 추가 단계가 있다면 플랜에 포함되어 있는지

─────────────────────────────────────────────────────────
항목 5. 봇 시작 시 명령어 안내 메시지 최신화 확인
─────────────────────────────────────────────────────────

discord_loop_bot.py에서 봇 시작 시 전송하는 _cmd_guide 메시지를 찾아서:
- 현재 구현된 모든 CMD_ 상수와 비교해서 누락된 명령어가 있는지
- 존재하지 않는 명령어가 안내에 포함되어 있는지
- 각 명령어 설명이 실제 동작과 일치하는지
- !novel-daily 옵션 목록(--target-words, --budget, --protagonist,
  --review-tier, --outer-cycles)이 실제 파싱 코드와 일치하는지

─────────────────────────────────────────────────────────
항목 6. README 최신화 확인
─────────────────────────────────────────────────────────

README.md를 읽고 실제 코드와 비교해서:
- 파이프라인 흐름도(episode.yaml → ... → 챕터 생성)가 현재 코드 순서와 일치하는지
  (SimulationOrchestrator → SceneDistiller → ProseGenerator → ChapterPolisher 포함)
- 모델 테이블의 모델명이 llm_client.py MODEL_PRICING 또는 실제 호출 코드와 일치하는지
- AUTO 루프 설명(Phase A trial 수, outer/inner 최대 횟수)이 daily_pipeline.py 상수와 일치하는지
- 자동 최적화 파이프라인 설명이 현재 2-스터디 구조(study_sim, study_prose)와 일치하는지
- 명령어 목록(자주 쓰는 명령어 섹션)에 !reboot, !shutdown이 포함되어 있는지
- 생성되는 파일 경로가 실제 output 구조와 일치하는지
- Codex CLI 상태(현재 비활성 → GPT fallback)가 정확히 반영되어 있는지

─────────────────────────────────────────────────────────
항목 7. rl_policy.json 키 동기화 확인
─────────────────────────────────────────────────────────

src/novel_writer/ 하위 파일(prose_generator.py, scene_distiller.py,
polisher.py, director.py)과 daily_pipeline.py, inline_optimizer.py에서
runtime_policy 또는 rl_policy를 통해 읽는 모든 키를 수집하고:
- 수집한 키가 data/rl_policy.json에 모두 존재하는지
- 코드에서 읽는데 rl_policy.json에 없는 키가 있으면 KeyError 위험 → 즉시 보고
- 새로 추가된 키에 기본값(default)이 코드 내에 있는지(.get("key", default) 형태인지)
- rl_policy.json의 version 필드가 변경 이력과 맞는지

─────────────────────────────────────────────────────────
항목 8. SceneDistiller → ProseGenerator 인터페이스 확인
─────────────────────────────────────────────────────────

scene_distiller.py의 DistilledScene dataclass 필드 목록을 확인하고:
- prose_generator.py에서 DistilledScene 객체의 필드를 접근하는 모든 곳을 찾아서
  실제 필드명과 일치하는지 확인
- polisher.py, daily_pipeline.py, generate_chapter.py에서
  distilled scene JSON을 파싱할 때 존재하지 않는 키에 접근하는 곳이 있는지
- DistilledScene을 dict로 변환하거나 JSON으로 직렬화하는 코드가
  모든 필드를 올바르게 포함하는지

─────────────────────────────────────────────────────────
항목 9. LLM budget 소진 처리 확인
─────────────────────────────────────────────────────────

inline_optimizer.py와 daily_pipeline.py에서:
- 트라이얼 도중 LLMClient budget 초과 시 어떤 예외가 발생하는지 확인
- 그 예외가 trial 단위로 catch되어 나머지 trial은 계속 실행되는지,
  아니면 전체 배치가 중단되는지 확인
- budget 초과로 trial 결과가 None일 때 스코어 집계(평균 계산 등)에서
  ZeroDivisionError 또는 TypeError가 발생하지 않는지 확인
- 모든 trial이 budget 초과로 실패했을 때의 fallback 처리가 있는지

─────────────────────────────────────────────────────────
항목 10. Git 상태 및 auto-commit 충돌 확인
─────────────────────────────────────────────────────────

daily_pipeline.py에서 GPT fixer가 git commit을 수행하는 코드를 찾아서:
- 파이프라인 시작 전 git working tree가 clean한지 확인하는 로직이 있는지
- uncommitted changes가 있을 때 auto-commit이 의도치 않은 파일을 포함하지 않는지
- git commit 실패 시(hook 오류, 권한 오류 등) 파이프라인이 크래시 없이 계속되는지
- rollback 로직이 있다면 git reset/restore가 의도한 파일만 대상으로 하는지

─────────────────────────────────────────────────────────
항목 11. 앙상블 스코어 fallback 확인
─────────────────────────────────────────────────────────

inline_optimizer.py 또는 daily_pipeline.py에서 Claude Haiku 앙상블 스코어링 코드를 찾아서:
- ANTHROPIC_API_KEY가 없을 때 except/fallback 분기가 있는지
- fallback 시 GPT 단독 점수로 대체되는지, 아니면 0점 처리되는지
- Claude API 호출 실패(타임아웃, 네트워크 오류)가 trial 전체를 죽이지 않는지
- 앙상블 가중치(GPT:Claude 비율)가 fallback 시에도 분모가 0이 되지 않는지

─────────────────────────────────────────────────────────
항목 12. Optuna SQLite study 누적 및 warmup 오염 확인
─────────────────────────────────────────────────────────

inline_optimizer.py와 optuna_multi_study.py에서:
- study SQLite 파일 경로 패턴 확인 (data/optuna_mini_{episode_id}.db 등)
- warmup으로 주입하는 policy_score_log.jsonl 레코드에서
  현재 파라미터 공간에 없는 구버전 키가 포함될 경우 Optuna가 에러 없이 처리하는지
- study를 load_if_exists=True로 열 때 이전 trial의 파라미터 범위가
  현재 search space와 충돌하면 어떻게 되는지 확인
- 오래된 .db 파일이 자동 정리되는 로직이 있는지

─────────────────────────────────────────────────────────
항목 13. Discord rate limit 처리 확인
─────────────────────────────────────────────────────────

discord_loop_bot.py에서 _send_text_with_token 및 _rest_send_text_return_message_id를 찾아서:
- Discord API 429(rate limit) 응답 시 재시도 로직이 있는지
- Phase A에서 trial 결과를 빠르게 연속 전송할 때 메시지 유실 가능성이 있는지
- asyncio.sleep을 사용한 rate limit 방지 간격이 충분한지 (Discord: 5msg/5sec 기준)
- 전송 실패 시 예외를 삼키고 파이프라인은 계속 진행되는지

─────────────────────────────────────────────────────────
항목 14. 테스트 커버리지 확인
─────────────────────────────────────────────────────────

tests/test_reader_feedback_guards.py를 읽고:
- 테스트가 import하는 모듈(prose_generator, scene_distiller, director, polisher)의
  실제 함수명/클래스명이 현재 소스와 일치하는지
- 최근 수정된 함수나 클래스 중 테스트에서 호출하지 않는 것이 있는지
- 테스트가 실행 가능한지 (import 경로, fixture, mock 대상이 현재 코드와 맞는지)
- daily_pipeline.py에서 검수 레이어로 실행하는 regression test 명령이
  실제 테스트 파일 경로와 일치하는지

─────────────────────────────────────────────────────────
항목 15. 로그 파일 크기 및 파싱 성능 확인
─────────────────────────────────────────────────────────

discord_loop_bot.py에서 !parameter, !benchmark 명령어 처리 코드를 찾아서:
- data/cycle_score_log.jsonl 전체를 매번 읽는지, 아니면 최근 N개만 읽는지 확인
- 파일이 클 경우(수천 줄 이상) Discord 응답이 타임아웃되지 않도록
  asyncio.to_thread 또는 비동기 처리가 되어 있는지
- policy_score_log.jsonl도 동일하게 크기 제한 없이 전부 읽는지
- 읽은 데이터가 Discord 2000자 제한을 초과할 때 분할 처리가 올바른지

─────────────────────────────────────────────────────────
항목 16. 다중 채널 동시 실행 격리 확인
─────────────────────────────────────────────────────────

discord_loop_bot.py에서 채널별 상태 딕셔너리를 모두 찾아서:
- DAILY_FEEDBACK_QUEUES, DAILY_STOP_EVENTS, DAILY_STATUS,
  DAILY_PROCESS_INFO, DAILY_SESSION_METRICS, DAILY_CHAPTER_PATHS 등이
  모두 channel_id를 키로 격리되어 있는지
- 두 채널이 동시에 !novel-daily를 실행할 때 공유 전역 변수가
  서로 덮어쓰지 않는지 (특히 anchor_threads, anchor_messages 등)
- !stop이 특정 채널에서만 작동하고 다른 채널 파이프라인에는 영향 없는지
- 채널 A의 파이프라인이 종료된 후 상태가 올바르게 정리(cleanup)되는지

─────────────────────────────────────────────────────────
항목 17. plan_approval 단계 타임아웃 및 상태 잔류 확인
─────────────────────────────────────────────────────────

discord_loop_bot.py에서 DAILY_PENDING_REVIEW_TIER를 찾아서:
- stage == "plan_approval" 상태에서 사용자가 아무 응답도 안 할 때
  자동으로 만료되는 로직(타임아웃 또는 TTL)이 있는지 확인
- 타임아웃 없으면: 해당 채널에서 이후 !novel-daily가 다시 동작 가능한지 확인
  (pending_review_tier 체크 로직이 먼저 실행되어 새 명령을 막는지)
- plan_approval 상태에서 다른 사용자(user_id 불일치)가 1/2를 입력하면 어떻게 처리되는지
- !stop, !reboot 같은 다른 명령어가 plan_approval 상태를 초기화하는지

─────────────────────────────────────────────────────────
항목 18. !emotion 명령어 연동 및 안내 메시지 확인
─────────────────────────────────────────────────────────

discord_loop_bot.py에서 CMD_EMOTION과 _make_emotion_chart를 찾아서:
- _cmd_guide (봇 시작 시 안내 메시지)에 !emotion이 포함되어 있는지 확인
- !emotion 호출 시 episode_key를 DAILY_EPISODE_KEYS에서 가져오는데,
  파이프라인 미실행 상태에서 None이 될 경우 glob 범위가 너무 넓어져
  엉뚱한 에피소드 씬 파일이 선택될 수 있는지 확인
- _make_emotion_chart 내부의 fallback glob("**/*scenes*.json")이
  output/ 전체를 탐색할 때 성능 문제가 없는지
- matplotlib import 실패 시 예외를 사용자에게 메시지로 전달하는지

─────────────────────────────────────────────────────────
항목 19. cost_estimator.py 연동 확인
─────────────────────────────────────────────────────────

tools/cost_estimator.py와 discord_loop_bot.py의 _build_plan_preview를 찾아서:
- _build_plan_preview 내부에서 format_cost_estimate_for_plan import가
  실패할 때 fallback 메시지가 표시되는지 확인
- log_cycle_score() 호출 시 cost_tracker 인자가 실제로 전달되는지
  (daily_pipeline.py:3342 근처 호출부)
- cycle_score_log.jsonl에 cost_usd가 없는 구버전 레코드가 있을 때
  estimate_cost_per_cycle()이 fallback_estimate()를 올바르게 반환하는지
- 첫 실행(레코드 0개) 상태에서 플랜 미리보기의 비용 안내가 fallback 수치임을
  사용자에게 명확히 표시하는지

─────────────────────────────────────────────────────────
항목 20. GPT Fixer 전역 Lock 다중 채널 영향 확인
─────────────────────────────────────────────────────────

daily_pipeline.py에서 _get_codex_fixer_lock()과 _CODEX_FIXER_LOCK을 찾아서:
- 이 lock이 전역(모든 채널 공유)인지, 채널별로 분리되는지 확인
- 두 채널이 동시에 Phase C(GPT Fixer)에 진입할 때 한 채널이
  다른 채널의 fixer 완료를 기다려야 하는지
- lock 대기 중 stop_event가 set되어도 대기에서 빠져나올 수 있는지
  (asyncio.wait_for 또는 stop_event 체크 로직 확인)
- lock 획득 후 예외 발생 시 lock이 정상 해제되는지 (async with 구조인지)

─────────────────────────────────────────────────────────
항목 21. Fix D 감정 필드 역직렬화 경로 확인
─────────────────────────────────────────────────────────

generate_chapter.py에서 씬 JSON을 읽어 DistilledScene을 재구성하는 코드를 찾아서:
- emotion_trajectory, tension_peaks, relationship_delta 필드가 없는
  구버전 *_scenes.json 파일을 읽을 때 KeyError 없이 기본값으로 처리되는지
- _load_precomputed_scenes() 또는 유사 함수에서 .get("emotion_trajectory", []) 형태인지,
  아니면 직접 인덱싱([key])인지 확인
- polisher.py나 다른 파일에서 DistilledScene JSON을 읽는 경로가 있다면 동일하게 확인
- to_dict()가 새 필드를 포함하지만 구버전 JSON 파싱 시 빠진 필드를
  DistilledScene 생성자에서 dataclass default로 처리하는지

─────────────────────────────────────────────────────────
항목 23. GPT Fixer 전역 Lock stop_event 대응 확인
─────────────────────────────────────────────────────────

daily_pipeline.py에서 _get_codex_fixer_lock()과 _CODEX_FIXER_LOCK을 찾아서:
- lock 대기(async with fixer_lock) 중 stop_event가 set되어도 lock 획득 대기가 취소되지 않는지 확인
  (현재 async with는 stop_event를 모름 — !stop 입력해도 fixer 대기 채널은 멈추지 않음)
- asyncio.wait_for 또는 stop_event 체크 루프로 대기를 중단하는 로직이 있는지 확인
- 없다면: lock 대기 최대 시간(timeout)을 설정해서 무한 대기를 방지하는지 확인
- lock 획득 후 예외 발생 시 async with 구조로 lock이 정상 해제되는지 확인

─────────────────────────────────────────────────────────
항목 24. cycle_score_log.jsonl 동기 읽기 블로킹 확인
─────────────────────────────────────────────────────────

discord_loop_bot.py에서 !parameter, !benchmark 명령 핸들러를 찾아서:
- cycle_score_log.jsonl, policy_score_log.jsonl 읽기가 메인 asyncio 루프에서 동기로 실행되는지 확인
  (open().read() 형태면 파일이 크면 Discord 이벤트 루프 블로킹 가능)
- asyncio.to_thread 또는 loop.run_in_executor로 감싸여 있는지 확인
- 파일이 없을 때(첫 실행) FileNotFoundError 처리가 있는지 확인
- 읽은 데이터가 Discord 2000자 제한을 초과할 때 청크 분할 처리가 올바른지 확인

─────────────────────────────────────────────────────────
항목 22. 체크리스트 자가 업데이트 (항상 마지막에 실행)
─────────────────────────────────────────────────────────

위 1~23개 항목 검수를 마친 뒤, 아래를 수행해줘.

1. `git diff HEAD` 또는 수정된 파일 목록을 확인해서
   이번 코드 변경에서 새로 등장한 개념·컴포넌트·데이터 흐름을 파악해줘.

2. 현재 REVIEW_CHECKLIST.md의 항목 1~24과 비교해서
   아래 기준으로 새 검수 항목이 필요한지 판단해줘:

   - 새로운 명령어(CMD_*)가 추가됐는가?
     → 해당 명령어의 데이터 연동, 쓰레드 라우팅, 안내 메시지 반영 여부 검수 항목 필요
   - 새로운 파이프라인 단계(함수, 클래스)가 추가됐는가?
     → 해당 단계의 인터페이스(입출력 필드), 에러 처리, 플랜 메시지 반영 여부 검수 항목 필요
   - 새로운 데이터 파일(*.json, *.jsonl, *.db)이 추가됐는가?
     → 해당 파일의 읽기/쓰기 키 일치 여부, 크기 관리 검수 항목 필요
   - 새로운 외부 API 연동(새 모델, 새 서비스)이 추가됐는가?
     → fallback 처리, rate limit, 비용 추적 검수 항목 필요
   - 기존 항목 중 이번 변경으로 검수 범위가 달라진 항목이 있는가?
     → 해당 항목의 확인 포인트를 업데이트할 필요가 있으면 명시

3. 새로 추가할 항목이 있으면 아래 형식으로 제안해줘:

   제안 항목 N. <항목 이름>
   배경: <왜 이번 변경으로 이 검수가 필요해졌는지>
   확인 포인트:
   - ...
   - ...

4. 제안 항목이 1개 이상이면 반드시 이렇게 말해줘:
   "REVIEW_CHECKLIST.md에 아래 항목을 추가할까요?"
   → 사용자가 "응" 또는 "추가해줘"라고 하면 REVIEW_CHECKLIST.md를 직접 수정해서
     항목 번호를 이어서 추가하고, 상단 "다음 N가지 항목" 숫자도 함께 업데이트해줘.

5. 추가할 항목이 없으면:
   "이번 변경에서 새로운 검수 항목은 필요하지 않습니다." 라고 말해줘.

─────────────────────────────────────────────────────────
검수 결과 형식
─────────────────────────────────────────────────────────

각 항목마다:
✅ 정상 — 확인된 내용 한 줄 요약
⚠️ 불일치 — 파일명:라인번호 + 구체적 문제 설명 + 수정 필요 여부
❌ 크래시 위험 — 파일명:라인번호 + 즉시 수정 필요

마지막에 수정이 필요한 항목만 모아서 우선순위 순으로 정리해줘.
```
