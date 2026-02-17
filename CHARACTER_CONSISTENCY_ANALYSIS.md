# 캐릭터 일관성 분석 리포트

## 🎯 사용자 지적 사항

**벤 클라크(Ben Clarke)**가:
- 설정: "수민의 라이벌" (학술 경쟁자, 같은 멘토 출신)
- 실제: "NSA 멤버"처럼 묘사됨

---

## 📊 캐릭터 설정 vs 실제 묘사 비교

### Ben Clarke - 설정 (characters.yaml)

```yaml
id: "ben_clarke"
name: "Ben Clarke"
role: "supporting"
bio: |
  An American physicist and Sumin's long-running rival, first forged in elite academic competition
  (the "we were measured against each other" kind). Both men came up under Dr. Alex Moreno's orbit
  (students/mentees in different cohorts), turning Moreno into both a moral benchmark and a contested legacy.
  Ben chose the pragmatic arc: government contracts, industry credibility, and a clean narrative of success.
```

**핵심 정체성:**
- 물리학자 (physicist)
- 수민의 라이벌 (rival)
- Moreno의 제자 (같은 멘토, 다른 코호트)
- **선택**: 정부 계약, 산업 신뢰도, "클린한 성공 경로"

**직업:**
- Government contracts (정부 계약)
- Industry credibility (산업 신뢰도)
- **NOT an NSA employee** - 계약자/협력자일 수는 있지만 요원은 아님

---

### Ben Clarke - 실제 묘사 (EP03 config + chapter)

#### EP03 설정 (ep03_ben_encounter.yaml)

```yaml
summary: |
  벤 클라크가 그곳에 앉아 있었다...
  벤은 이제 프로젝트 리더였다. 그의 명함: 'Ben Clark, Quantum Packaging Team Lead,
  Advanced Microelectronics Research Division (AMRD)'.

  벤이 목소리를 낮췄다. "우린 초전도체 패키징에 특화했어. 그리고..." 벤이 주변을 둘러봤다.
  "NSA가 자금을 댔어. 밀러라는 요원 알지? 그놈이 우리 팀에 붙어 있어."
```

**문제점 1: 직접적인 NSA 연결**
- "NSA가 자금을 댔어" - OK (펀딩 받는 것)
- **"그놈이 우리 팀에 붙어 있어"** - ⚠️ 벤이 NSA 내부자처럼 말함

#### 생성된 챕터 (ep03_ben_encounter_iter1_chapter.md)

Line 15:
```
"DARPA 계약서에 자금 지원과 NSA 공동 펀딩이라 적혀있어요."
```
- ✅ OK: 펀딩 받는 것

Line 55:
```
나는 조용히 물었다. "밀러는 그 과정을 어떻게 돕는 건가요?"
벤이 잠깐 웃으며 답했다. "밀러는 좋은 사람이야. 신뢰할 수 있어."
```
- ⚠️ 문제: 벤이 밀러(NSA 요원)를 개인적으로 잘 아는 것처럼 묘사

Line 63:
```
멀리서 밀러의 시선이 느껴졌다. 그의 눈빛은 계산기처럼 차갑고 정확하게 반짝였다.
```
- ⚠️ 문제: 밀러가 벤과 수민의 대화를 감시하는 장면 - 벤이 NSA 작전의 일부처럼 보임

---

## 🔍 발견된 일관성 문제

### 1. Ben Clarke의 정체성 혼란

| 측면 | 설정 | 실제 묘사 | 문제 |
|------|------|-----------|------|
| 직업 | 물리학자, AMRD 팀 리더 | NSA 협력자/내부자? | ⚠️ 경계 모호 |
| Miller와 관계 | (설정 없음) | "우리 팀에 붙어 있어" | ⚠️ 너무 친밀 |
| NSA 연관 | 펀딩 수혜자 | 작전 참여자처럼 보임 | ⚠️ 역할 혼동 |
| Sumin과 관계 | 라이벌 (경쟁자) | 리크루터/협박자 | ⚠️ 라이벌 요소 약화 |

### 2. 라이벌 관계가 제대로 표현되지 않음

**설정에서 기대되는 요소:**
- "Elite academic competition" (엘리트 학술 경쟁)
- "We were measured against each other" (서로 비교당함)
- "Moreno into both a moral benchmark and contested legacy" (모레노가 도덕적 기준이자 경쟁 대상)

**실제 묘사:**
- ❌ 학술적 경쟁 요소 없음
- ❌ 과거 경쟁 에피소드 언급 없음
- ❌ Moreno를 두고 경쟁했던 관계 묘사 없음
- ✅ "채용 제안" 장면만 있음

### 3. 타 에피소드에서의 혼란

**EP04 (lab_whispers):**
```
모레노 "위상 드리프트 보상 회로", 벤의 메모 "Greyshore", 요나스의 노트북 "Phase-Guard Protocol"
```
- 벤이 Greyshore와 연결 - 이건 cartel/NSA 백채널 암시
- **문제**: 벤이 단순 리크루터가 아니라 정보 작전의 노드처럼 보임

**EP06 (patrons_doctrine):**
```
Benefactor. 8월부터 10월까지 $800,000의 미지의 자금.
```
- Benefactor가 cartel인데, NSA 펀딩과 혼재

**EP07 (between_faction):**
```
모레노? 벤? 카를로스? 요나스? 아니면 모두? 수민은 이제 누구를 신뢰할 수 없었다.
```
- 벤이 적대 세력 리스트에 포함 - 라이벌이 아닌 "적"으로 격상

---

## 🎭 캐릭터별 일관성 체크

### Kim Sumin (김수민) - ✅ 일관성 OK
- 설정: Protagonist, quantum physicist, vulnerable to leverage
- 묘사: 일관적으로 압박받는 과학자로 표현

### Elena Ramirez - ✅ 일관성 OK
- 설정: Cartel 내부자, 수민의 협력자
- 묘사: 일관적

### Carlos Reyes - ✅ 일관성 OK
- 설정: Cartel operator, leverage specialist
- 묘사: 압박과 레버리지 사용

### Alex Moreno - ⚠️ 부분적 문제
- 설정: Mentor, 도덕적 기준점
- 묘사: 언급은 많지만 실제 등장/상호작용 부족
- **Ben과의 관계**: 둘 다 Moreno 제자인데 이 연결 약함

### Ben Clarke - ❌ 심각한 일관성 문제
#### 설정 vs 묘사 차이:

| 요소 | 설정 | 묘사 | 괴리도 |
|------|------|------|--------|
| 정체성 | 라이벌 물리학자 | NSA 협력자 | ⚠️⚠️⚠️ |
| 관계 | 경쟁자 | 리크루터 | ⚠️⚠️ |
| 동기 | "Prove his choices were right" | NSA 작전 수행? | ⚠️⚠️ |
| Moreno 연결 | 같은 멘토 출신 | 거의 언급 없음 | ⚠️⚠️ |
| 목표 | "Pull Sumin into legitimate pipeline" | 비자/협박 사용 | ⚠️ |

### Agent Christian Miller - ⚠️ 부분적 문제
- 설정: NSA operator, 독립적 행동
- 묘사: Ben과 너무 긴밀한 협력 → Ben을 NSA 내부자처럼 보이게 만듦

### Yonas Mehret - ✅ 일관성 OK
- 설정: Ambiguous loyalties
- 묘사: 일관적으로 애매모호

---

## 🚨 핵심 문제점

### 1. Ben Clarke의 역할 혼동

**의도된 설정:**
```
벤 = 수민의 라이벌 + "클린한 길"의 유혹자
- 같은 Moreno 밑에서 공부
- 벤은 정부 계약 선택 → 성공
- 수민은 순수 연구 선택 → 위기
- 벤의 제안 = "내 길이 맞았지?" 증명
```

**실제 묘사:**
```
벤 = NSA 작전의 협력자/도구
- Miller와 긴밀한 관계
- Greyshore 정보 제공
- 수민 리크루팅 작전의 일부
- 학술 경쟁 요소 부재
```

### 2. NSA vs Cartel vs Ben의 경계 불명확

현재 스토리에서:
- NSA (Miller) ← **Ben** → DARPA
- Cartel (Carlos) ← **Benefactor** → Greyshore
- **Ben이 Greyshore를 언급** → Ben이 cartel 연결?

**혼란 포인트:**
- Ben이 NSA 펀딩을 받는 것: ✅ OK
- Ben이 Miller를 잘 아는 것: ⚠️ 경계선
- Ben이 Greyshore를 아는 것: ❌ 문제 (cartel 정보를 왜 알아?)
- Miller가 Ben-Sumin 대화 감시: ❌ 문제 (Ben을 NSA 요원처럼 만듦)

### 3. 라이벌 관계의 부재

**부족한 요소:**
- 과거 경쟁 에피소드 (논문 경쟁, 학회 발표, Moreno의 인정 경쟁 등)
- 학술적 긴장감 ("네가 이 문제를 먼저 풀었지만, 내가 먼저 상용화했어")
- 감정적 복잡성 (존중 + 질투 + 경쟁심)

**현재 묘사:**
- 일방적 리크루팅
- 힘의 불균형 (Ben이 우위)
- 감정적 긴장 부족

---

## 📋 수정 필요 항목

### 우선순위 1 (긴급): Ben Clarke 정체성 명확화

#### A. characters.yaml 수정 필요 여부
**현재 설정 자체는 OK**, 다만 명확화 필요:

```yaml
ben_clarke:
  clarifications:
    - "NOT an NSA employee - receives government funding"
    - "Relationship with Miller: professional/transactional, not friendly"
    - "Knows about Greyshore through DARPA briefing, not insider knowledge"
    - "Primary motivation: prove his pragmatic path was right vs Sumin's idealism"
```

#### B. 에피소드 설정 수정 필요

**ep03_ben_encounter.yaml:**

변경 전:
```yaml
"NSA가 자금을 댔어. 밀러라는 요원 알지? 그놈이 우리 팀에 붙어 있어."
```

변경 후:
```yaml
"NSA 공동 펀딩이야. 계약 조건의 일부지. 밀러라는 요원이 감독관으로 배정됐어."
"친한 건 아니야. 그냥 업무상 알고 지내는 정도. 근데 그 사람 통해서 네 상황도 들었어."
```

**차이:**
- "우리 팀에 붙어 있어" → "감독관으로 배정" (거리감 유지)
- 친밀도 하향 조정
- Ben이 NSA 내부자가 아니라 계약 수혜자임을 명확히

### 우선순위 2: 라이벌 관계 강화

#### EP03에 추가할 요소:

```yaml
summary 추가:
  "벤을 보는 순간, MIT 시절이 떠올랐다. Moreno 교수 세미나실.
   우리 둘만 남아 칠판 앞에서 위상 안정성 문제를 놓고 3시간 동안 논쟁했던 날.
   결국 벤이 먼저 해법을 찾았고, Moreno는 '우아한 접근'이라고 칭찬했다.
   그날 밤 나는 연구실에 남아 다른 방법을 찾았지만,
   벤의 방법이 더 실용적이라는 걸 인정할 수밖에 없었다.

   그리고 지금. 벤은 DARPA 팀 리더고, 나는... 비자 걱정을 하고 있다.
   누가 옳았는지는 명확해 보였다."

clue 추가:
  - id: "clue_ep03_rivalry_moreno"
    content: |
      벤이 잠깐 감상적인 표정을 지었다. "Moreno 교수님 요즘 어떠신가?
      내가 DARPA 쪽으로 간다고 했을 때 실망하셨지. '순수 연구의 배신'이라고까지 하셨어.
      근데 지금 보면... 내 선택이 맞았던 것 같아. 넌 어때? 아직도 교수님 방식을 따르고 있어?"

      그 질문에는 도전이 섞여 있었다. "누가 Moreno의 진정한 계승자인가"라는.
```

### 우선순위 3: Greyshore 정보 출처 명확화

**문제:** Ben이 왜 Greyshore를 아는가?

**해결 옵션:**

**Option A - Ben은 Greyshore를 잘 모름 (추천)**
```yaml
ep03 수정:
  "뒷면에 손으로 쓴 글씨. 'ben.clark@amrd.gov / Ask about Greyshore if you need'.

  수민: "Greyshore가 뭐죠?"
  벤: "정확히는 나도 몰라. DARPA 브리핑에서 한 번 나온 이름인데,
       특수 펀딩 채널이라고만 들었어. 네가 궁금하면 Miller한테 물어봐.
       나는 계약 라인만 알고, 저쪽 디테일은 몰라."
```

**Option B - Ben은 의도적으로 정보 흘림**
```yaml
ep03 수정:
  벤이 목소리를 낮췄다. "Greyshore... 공식적으로는 모르는 게 좋아.
  근데 네가 QuantumFront 쪽 자금 흐름 이상하다고 느꼈다면,
  그 이름 기억해둬. 언젠가 필요할 거야."

  (Ben의 동기: 수민이 QuantumFront의 위험을 깨닫고 자기 팀으로 오길 바람)
```

### 우선순위 4: Miller-Ben 관계 재정의

**현재 문제:** 너무 긴밀해 보임

**수정 방향:**
```yaml
ep03 추가 장면:
  Miller가 멀리서 지켜보는 장면에서:

  "Miller의 시선이 느껴졌다. 벤도 그걸 눈치챘는지 미간을 찌푸렸다.
   '저 사람 때문에 자유롭게 말하기 힘들어. DARPA 감독관이라는 게
   생각보다 답답해. 내 팀 회의에도 자주 오거든.'

   벤의 불편함이 진짜였다. 그는 Miller의 도구가 아니라,
   Miller에게 감시받는 계약자였다."
```

---

## 🎯 플래닝: 수정 우선순위

### Phase 1: 긴급 수정 (Ben Clarke 정체성)

1. **ep03_ben_encounter.yaml 수정**
   - Ben-Miller 관계 거리감 확보
   - Greyshore 언급 맥락 수정
   - 라이벌 관계 회상 추가

2. **ep04_lab_whispers.yaml 검토**
   - Ben 언급 부분 일관성 확인

3. **ep39_ben_offer.yaml 사전 검토**
   - 후반부 Ben 등장 에피소드 일관성 확보

### Phase 2: 관계 강화 (Rivalry dynamics)

1. **새 clue 추가: Moreno 연결**
   - EP03에 Moreno 언급 대화 추가
   - 과거 경쟁 에피소드 플래시백

2. **Ben의 동기 명확화**
   - "수민 리크루팅" 이유: 라이벌을 이기고 싶은 욕구
   - "Prove my path was right" 요소 강화

### Phase 3: 전체 일관성 검증

1. **캐릭터 관계 매트릭스 재검토**
   - Ben ↔ Miller: 계약자 ↔ 감독관
   - Ben ↔ Sumin: 라이벌 + 리크루터
   - Ben ↔ Moreno: 제자 + "배신자"

2. **정보 흐름 검증**
   - Ben이 아는 정보: DARPA 계약, 공식 펀딩
   - Ben이 모르는 정보: Cartel 디테일, Greyshore 세부사항
   - Ben이 추측하는 정보: QuantumFront의 위험성

---

## 📝 요약

### 발견된 문제
1. ⚠️⚠️⚠️ **Ben Clarke가 NSA 내부자처럼 묘사됨** (설정: 라이벌 물리학자)
2. ⚠️⚠️ **라이벌 관계 거의 표현 안 됨** (경쟁 요소 부재)
3. ⚠️⚠️ **Greyshore 정보를 Ben이 아는 것이 부자연스러움**
4. ⚠️ **Miller-Ben 관계가 너무 친밀함** (감독자-피감독자여야 함)

### 수정 필요 파일
- `config/episodes/ep03_ben_encounter.yaml` (높음)
- `config/episodes/ep04_lab_whispers.yaml` (중간)
- `config/episodes/ep39_ben_offer.yaml` (예방)
- `config/characters.yaml` (명확화만 필요, 설정 자체는 OK)

### 권장 조치
1. **코드 수정 불필요** - 에피소드 설정(YAML) 수정으로 해결 가능
2. **재생성 필요** - EP03, EP04 (수정 후)
3. **Director AI 가이드 추가** - "Ben is NOT NSA member" 명시

---

**다음 단계:** 위 분석을 바탕으로 수정 작업을 진행할지 결정 필요.
