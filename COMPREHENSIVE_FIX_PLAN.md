# 전체 소설 일관성 수정 플랜

## 📖 스토리 구조 개요

**총 49 에피소드 / 6막 구조**
- Act 1 (Setup): EP01-03 (3편)
- Act 2-3 (Discovery): EP04-14 (11편)
- Act 4-5 (Technical/Weaponization): EP15-30 (16편)
- Act 5-6 (Crisis): EP31-44 (14편)
- Act 6 (Climax): EP45-47 (3편)
- Act 7 (Resolution): EP48-49 (2편)

**타임라인**: 2041년 5월 ~ 2043년 3월 (약 2년)

---

## 🚨 발견된 일관성 문제 (우선순위별)

### Priority 1: CRITICAL - 타임라인 및 핵심 서사 누락

#### 1.1 Alex Moreno 납치/죽음 서사 (🔴 긴급)

**문제:**
- **설정**: "ep5_kidnap_moreno" (2043-03-15) - Act 4
- **실제 EP01-10**: Moreno는 EP01 학회 후 사라짐, 납치 장면 없음
- **스토리상 위치**: EP05가 아니라 실제 ep5_kidnap_moreno beat

**혼란 원인:**
```yaml
# storyline.yaml 구조
episodes:
  - episode_id: "ep0"  # 실제 에피소드: ep01-04
  - episode_id: "ep1"  # 실제 에피소드: ep05-15
  - episode_id: "ep5"  # 실제 에피소드: ep39-40?
```

**실제 에피소드 매핑:**
- `episode_id: "ep0"` → 실제 파일: `ep01`, `ep02`, `ep03`, `ep04`
- `episode_id: "ep1"` → 실제 파일: `ep05`~`ep15`?
- Moreno 납치 = `episode_id: "ep5"` → 실제 에피소드 번호 확인 필요

**수정 플랜:**
1. 실제 에피소드 파일 목록 확인
2. Storyline의 episode_id와 실제 파일명 매핑표 작성
3. Moreno 납치 에피소드가 존재하는지 확인
4. 없으면 생성, 있으면 내용 검증

---

#### 1.2 Elena USB 전달 서사 (🔴 긴급)

**문제:**
- **설정**: "ep3_elena_link" - Elena가 USB로 내부 증거 전달
- **실제 EP01-10**: Elena는 EP04, EP07 등장하지만 USB 전달 없음
- **Storyline**: `episode_id: "ep3"` (Act 3) = "Elena's USB"

**실제 에피소드 매핑 추정:**
- `episode_id: "ep3"` → 실제 파일 `ep??`

**수정 플랜:**
1. `episode_id: "ep3"` 해당하는 실제 파일 찾기
2. Elena USB 전달 장면 있는지 확인
3. 없으면 해당 에피소드에 beat 추가

---

#### 1.3 FBI/CIA 등장 타이밍 (🟡 중요)

**문제:**
- **FBI Sophia**: 설정상 `ep2_first_contact_fbi` (Act 3)인데 EP01-10에 미등장
- **CIA Ethan**: 설정상 `ep4_CIA_overture` (Act 4)인데 EP01-10에 미등장

**수정 플랜:**
1. Storyline의 `episode_id: "ep2"`, `episode_id: "ep4"` 실제 파일 확인
2. 해당 에피소드 생성 여부 확인
3. 미생성 시 우선순위 낮춤 (아직 도달 안 한 부분일 수 있음)

---

### Priority 2: HIGH - 캐릭터 역할 혼동

#### 2.1 Ben Clarke - NSA 거리 확보 (🟡 중요)

**현재 문제:**
```yaml
EP03 (ep03_ben_encounter.yaml):
"NSA가 자금을 댔어. 밀러라는 요원 알지? 그놈이 우리 팀에 붙어 있어."
```

**수정 방향:**
```yaml
수정 후:
"NSA 공동 펀딩이야. DARPA 계약의 일부라서 감독관이 배정됐어.
 밀러라는 요원인데... 솔직히 불편해. 우리 회의에도 자주 와."
```

**추가 장면 (라이벌 관계 강화):**
```yaml
# EP03에 추가할 회상 장면
flashback:
  "벤을 보는 순간, MIT 세미나실이 떠올랐다.
   Moreno 교수 앞에서 위상 안정성 문제를 놓고 3시간 논쟁했던 날.
   결국 벤이 먼저 해법을 찾았고, 나는... 인정할 수밖에 없었다.

   그리고 지금. 벤은 DARPA 팀 리더고, 나는 비자 걱정을 하고 있다."

# 대화 추가
ben_dialogue:
  "Moreno 교수님 요즘 어떠신가? 내가 DARPA로 간다고 했을 때
   실망하셨지. '순수 연구의 배신'이라고까지 하셨어.
   근데 넌 어때? 아직도 교수님 방식을 따르고 있어?"
```

**파일 수정:**
- `config/episodes/ep03_ben_encounter.yaml`
- Summary 섹션 수정
- Clue에 Moreno 관련 추가

---

#### 2.2 Elena Ramirez - 출입 기록 모순 (🟡 중요)

**현재 문제:**
```yaml
EP07 언급:
"Elena의 배지가 3일간 기록 안 됐는데 사무실 문은 열림"
```

**수정 방향 (Option A - 설명 추가):**
```yaml
수정:
"Elena의 배지가 3일간 기록 안 됐다. 하지만 사무실 도어록은 열렸다.
 수민이 요나스에게 물었다. '배지 없이 어떻게?'
 요나스가 어깨를 으쓱했다. '마스터 키카드. 관리자는 로그를 남기지 않아도 돼.'"
```

**수정 방향 (Option B - Elena의 의도 강조):**
```yaml
수정:
"Elena는 의도적으로 자기 배지를 쓰지 않았다.
 누군가의 마스터 키를 빌렸거나, 시스템을 우회했을 것이다.
 그녀는 systems/security engineering 전문가니까.
 문제는... 왜 자기 흔적을 지우려 했는가?"
```

**파일 수정:**
- `config/episodes/ep07_*.yaml` 찾기 및 수정

---

#### 2.3 Agent Miller - 이름 철자 불일치 구현 (🟢 낮음)

**현재 문제:**
- 설정: "Christan/Christian 불일치"가 tradecraft tell
- 실제: 모두 "Christian"으로 일관됨

**수정 방향:**
```yaml
Option A - 에피소드에서 불일치 구현:
  EP02: "Christian Miller" (명함)
  EP07: "Christan Miller" (이메일) ← 철자 다름
  EP08: "Christian Miller" (공식 문서)

  수민의 발견:
  "Miller의 이메일을 다시 봤다. 서명이 'Christan Miller'였다.
   i가 빠졌다. 명함에는 'Christian'이었는데.
   오타? 아니면 의도적? NSA 요원이 자기 이름 철자를 틀릴 리 없다."

Option B - 설정 수정 (tell 제거):
  characters.yaml에서 "inconsistent spelling tell" 설명 삭제
```

**권장:** Option A (스토리 요소로 활용)

---

### Priority 3: MEDIUM - 관계 및 동기 명확화

#### 3.1 Greyshore 정보 출처 (🟡 중요)

**현재 문제:**
- Ben이 Greyshore를 언급 (EP03)
- Greyshore는 cartel 자금 채널인데, Ben이 왜 아는가?

**수정 방향:**
```yaml
Option A - Ben은 Greyshore를 잘 모름:
  EP03 수정:
  ben: "뒷면에 'Ask about Greyshore'라고 적었어.
        정확히는 나도 몰라. DARPA 브리핑에서 한 번 나온 이름인데,
        특수 펀딩 채널이라고만 들었어. 네가 궁금하면 Miller한테 물어봐."

Option B - Ben이 의도적으로 경고:
  ben: "Greyshore... 공식적으로는 모르는 게 좋아.
        근데 네가 QuantumFront 자금 흐름 이상하다고 느꼈다면,
        그 이름 기억해둬. 언젠가 필요할 거야."

  (Ben의 동기: 수민이 위험을 깨닫고 자기 팀으로 오길 바람)
```

**권장:** Option B (Ben의 복잡한 동기 표현)

---

#### 3.2 Carlos-Sumin 관계 긴장도 (✅ 일관적)

**검증 결과:** 일관적으로 잘 표현됨
- Initial relationship: -0.6
- EP04, EP09, EP10에서 압박/레버리지 사용
- 수정 불필요

---

#### 3.3 El Patrón 등장 타이밍 (✅ 일관적)

**검증 결과:**
- 설정: `ep4_el_patrons_eye` 이전 얼굴 노출 금지
- 실제: EP06 음성만, EP09-10 간접 언급
- **BUT**: Storyline의 `episode_id: "ep4"` ≠ 실제 파일 `ep04`
- 실제 대면은 나중 에피소드일 가능성

**확인 필요:**
- `episode_id: "ep4"` 해당하는 실제 파일 찾기

---

## 📋 수정 작업 플랜

### Phase 1: 타임라인 및 매핑 정리 (🔴 최우선)

#### Task 1.1: Episode ID 매핑표 작성
```bash
목표: storyline.yaml의 episode_id와 실제 파일 매핑 확인

실행:
1. config/episodes/ 전체 파일 목록
2. 각 파일의 episode_id 필드 확인
3. storyline.yaml의 episode_id와 대조

출력: EPISODE_MAPPING.md
```

#### Task 1.2: 누락된 핵심 에피소드 확인
```bash
확인 항목:
- ep5_kidnap_moreno (Moreno 납치) - 존재 여부
- ep3_elena_link (Elena USB 전달) - 존재 여부
- ep2_first_contact_fbi (FBI Sophia) - 존재 여부
- ep4_CIA_overture (CIA Ethan) - 존재 여부

누락 시:
- 우선순위 1: Moreno 납치, Elena USB
- 우선순위 2: FBI/CIA 등장 (Act 3-4에서 중요)
```

---

### Phase 2: 캐릭터 일관성 수정 (🟡 높음)

#### Task 2.1: Ben Clarke 수정
```yaml
파일: config/episodes/ep03_ben_encounter.yaml

수정 1 - Summary 섹션:
  old: "NSA가 자금을 댔어. 밀러라는 요원 알지? 그놈이 우리 팀에 붙어 있어."
  new: "NSA 공동 펀딩이야. 계약 조건의 일부라서 감독관이 배정됐어.
        밀러라는 요원인데... 솔직히 불편해. 우리 회의에도 자주 와.
        친한 건 아니야. 그냥 업무상 알고 지내는 정도."

수정 2 - Moreno 관련 회상 추가:
  location: summary 첫 부분
  content:
    "벤을 보는 순간, MIT 시절이 떠올랐다. Moreno 교수 세미나실.
     우리 둘만 남아 위상 안정성 문제를 놓고 3시간 동안 논쟁했던 날.
     결국 벤이 먼저 해법을 찾았고, Moreno는 '우아한 접근'이라고 칭찬했다.
     그날 밤 나는 연구실에 남아 다른 방법을 찾았지만,
     벤의 방법이 더 실용적이라는 걸 인정할 수밖에 없었다."

수정 3 - Clue 추가:
  id: "clue_ep03_rivalry_moreno"
  content:
    "벤이 잠깐 감상적인 표정을 지었다. 'Moreno 교수님 요즘 어떠신가?
     내가 DARPA 쪽으로 간다고 했을 때 실망하셨지.
     근데 지금 보면... 내 선택이 맞았던 것 같아. 넌 어때?
     아직도 교수님 방식을 따르고 있어?'

     그 질문에는 도전이 섞여 있었다. '누가 Moreno의 진정한 계승자인가.'"

수정 4 - Greyshore 언급:
  old: "Ask about 'Greyshore'"
  new: "Ask about 'Greyshore' — DARPA 브리핑에서 들었던 이름. 특수 펀딩 채널.
        네가 QuantumFront 자금 이상하다고 느꼈다면, 기억해둬."
```

#### Task 2.2: Elena Ramirez 수정
```yaml
파일: ep07 해당 파일 (찾기 필요)

수정 - 배지 기록 설명:
  old: "Elena의 배지가 3일간 기록 안 됨"
  new: "Elena의 배지가 3일간 기록 안 됐다. 하지만 사무실 도어록은 열렸다.
        그녀는 의도적으로 자기 흔적을 지우고 있었다.
        Systems engineering 전문가니까... 가능하겠지.
        문제는 왜? 누구를 피하고 있는가?"
```

#### Task 2.3: Agent Miller 수정 (선택적)
```yaml
파일: config/episodes/ep02_nsa_funding.yaml, ep07_*.yaml

Option A - 이름 불일치 구현:
  EP02: "Christian Miller" (명함)
  EP07: "Christan Miller" (이메일 서명)
  EP08: "Christian Miller" (공식 문서)

  수민의 발견 (EP08):
  "Miller의 과거 이메일을 다시 봤다. 서명이 'Christan'이었다.
   i가 빠졌다. 명함과 공식 문서에는 'Christian'인데.
   NSA 요원이 자기 이름을 틀릴 리 없다. 의도적인 신호? 여러 정체성?"

Option B - 설정 수정:
  characters.yaml에서 tell 설명 제거

권장: Option A
```

---

### Phase 3: Director AI 가이드 강화 (🟢 예방)

#### Task 3.1: Character Constraints 추가
```yaml
파일: config/storyline.yaml 또는 별도 character_rules.yaml

추가:
character_constraints:
  ben_clarke:
    - "Ben is NOT an NSA employee - he receives government funding"
    - "Relationship with Miller: professional/monitored, not friendly"
    - "Ben's motivation: prove pragmatic path > idealistic path (vs Sumin)"
    - "Moreno connection: both were mentees, competitive relationship"
    - "Greyshore knowledge: limited, from DARPA briefing only"

  elena_ramirez:
    - "Elena is systems/security expert - can bypass access logs"
    - "USB delivery happens in episode_id ep3 (Elena's USB)"
    - "Death gate: not before ep6_betrayal_elena_death"

  alex_moreno:
    - "Kidnap happens in episode_id ep5"
    - "Before kidnap: can only appear in flashbacks or distant communication"
    - "Relationship with Sumin: 0.95 (filial bond)"
    - "Relationship with Ben: 0.35 (disappointed by pragmatic choice)"

  agent_christan_miller:
    - "Name spelling varies: 'Christian' (official), 'Christan' (informal)"
    - "Not friends with Ben - monitors DARPA contractors"
    - "Uses coercion through legal gray zones"
```

#### Task 3.2: Episode Generation Rules
```yaml
파일: director.py 또는 config/director_rules.yaml

추가:
generation_rules:
  character_consistency:
    - "Check character bio before generating dialogue"
    - "Validate relationships match initial_relationships values"
    - "Ensure character doesn't know information outside their access level"

  timeline_enforcement:
    - "Check director_control_model flags before mentioning events"
    - "Moreno kidnap: check moreno_kidnapped flag"
    - "Elena death: check elena_dead flag"
    - "RSA break: check rsa_broken_real flag"
```

---

### Phase 4: 재생성 및 검증 (🔵 실행)

#### Task 4.1: 수정된 에피소드 재생성
```bash
# Ben Clarke 관련 수정 후
python3 quality_adaptive_generator.py \
  --episode-id ep03_ben_encounter \
  --episode-config config/episodes/ep03_ben_encounter.yaml \
  --protagonist kim_sumin \
  --target-words 800 --scenes 3 \
  --target-score 0.80 --max-iterations 1

# Elena 관련 수정 후 (파일 확인 후)
python3 quality_adaptive_generator.py \
  --episode-id ep07_* \
  --episode-config config/episodes/ep07_*.yaml \
  ...
```

#### Task 4.2: 일관성 재검증
```bash
# 재생성된 챕터 검증
python3 quality_analyzer.py output/ep03_ben_encounter_chapter.md

# 캐릭터 일관성 자동 체크 (스크립트 작성 필요)
python3 verify_character_consistency.py \
  --character ben_clarke \
  --episodes ep03
```

---

## 🎯 실행 우선순위

### Week 1: 긴급 (P1)
1. ✅ Episode ID 매핑표 작성
2. ✅ Moreno 납치 에피소드 존재 확인
3. ✅ Elena USB 에피소드 존재 확인
4. ⏳ 누락 시 해당 에피소드 생성 또는 타임라인 조정

### Week 2: 중요 (P2)
5. ⏳ Ben Clarke 수정 (EP03)
6. ⏳ Elena 배지 문제 수정 (EP07)
7. ⏳ Greyshore 정보 출처 명확화
8. ⏳ 재생성 및 품질 검증

### Week 3: 강화 (P3)
9. ⏳ Agent Miller 이름 불일치 구현
10. ⏳ Director AI 규칙 추가
11. ⏳ Character constraints 문서화
12. ⏳ 전체 에피소드 일관성 재검증

---

## 📊 예상 수정 범위

| 항목 | 파일 수 | 난이도 | 예상 시간 |
|------|--------|--------|----------|
| Episode 매핑 조사 | 1 (새 문서) | 중 | 1h |
| Ben Clarke 수정 | 1 (ep03) | 중 | 2h |
| Elena 수정 | 1 (ep07?) | 중 | 2h |
| Miller 이름 수정 | 3 (ep02, 07, 08?) | 낮 | 1h |
| Director 규칙 추가 | 2 (storyline, director) | 중 | 3h |
| 재생성 및 검증 | 3+ episodes | 높 | 4h |
| **합계** | **~11 files** | **중-높** | **13h** |

---

## 🔍 다음 단계

1. **즉시 실행 필요:**
   - Episode ID 매핑표 작성
   - Moreno/Elena 핵심 에피소드 존재 확인

2. **사용자 결정 필요:**
   - Moreno 납치 에피소드 미생성 시 → 지금 생성 vs 나중 생성?
   - Ben Clarke 수정 방향 → Option A (거리 확보) vs Option B (관계 재설정)?
   - Miller 이름 불일치 → 구현 vs 설정 수정?

3. **자동화 가능:**
   - 캐릭터 일관성 검증 스크립트
   - Episode 재생성 배치 처리

---

**준비 완료. 다음 지시를 기다립니다.**
