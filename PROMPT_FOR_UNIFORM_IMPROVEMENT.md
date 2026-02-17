# AI에게 요청할 프롬프트: 전체 에피소드 균일 품질 향상

## 📋 배경 상황 설명용 프롬프트

```
현재 상황:
- EP01-04를 반복 개선했지만, EP04만 성공(0.845, B등급)하고 나머지는 실패했습니다
- 문제: 시뮬레이션 데이터의 대화량이 에피소드마다 달라서, 일부는 대화가 거의 없음
- 목표: 모든 에피소드(EP01-04)가 균일하게 0.800+ (B등급) 달성

현재 결과:
- EP01: 0.784 (C) - 대화 12.27%
- EP02: 0.776 (C) - 대화 1.44% ← 심각
- EP03: 0.762 (C) - 대화 22.37% (대화는 성공)
- EP04: 0.845 (B) - 대화 21.15% ✓

요구사항:
1. 에피소드를 제이하고 전체적인 코드를 프롬프트를 수정해서
2. 시뮬레이션 데이터에 대화가 없어도
3. AI가 자동으로 적절한 대화를 창작하도록 만들어주세요
4. 모든 에피소드에서 일관되게 작동해야 합니다

핵심 제약:
- 에피소드 설정 파일은 건드리지 말 것
- 대화 비율: 20-25% 필수
- 문장 길이: 11-15단어 (한국어)
- 단락 구조: 2-3문장
```

---

## 🎯 구체적 작업 요청 프롬프트

### 프롬프트 1: 전체 분석 및 진단

```
다음 파일들을 분석해주세요:

1. 전체 시템 코드(lines 226-310)
   - 현재 시스템 프롬프트와 태스크 프롬프트 확인

2. 생성된 4개 에피소드의 대화 분석:
   - output/ep01_academic_presentation_iter1_chapter.md
   - output/ep02_nsa_funding_iter1_chapter.md
   - output/ep03_ben_encounter_iter1_chapter.md
   - output/ep04_lab_whispers_iter1_chapter.md

질문:
1. EP04는 왜 성공하고 EP02는 실패했나요? (대화 21.15% vs 1.44%)
2. 시뮬레이션 데이터의 대화량 차이를 무시하고, AI가 항상 대화를 창작하도록 만들려면?
3. 모든 에피소드에서 균일하게 작동하는 프롬프트 전략은?

분석 후 수정방안검토
```

---

### 프롬프트 2: 강력한 대화 생성 프롬프트 작성

```
prose_generator.py의 시스템 프롬프트(lines 226-247)를 다음 조건으로 재작성해주세요:

필수 요구사항:
1. "시뮬레이션 데이터에 대화가 없어도 반드시 창작하라"는 명시적 지시
2. 대화 생성이 선택이 아닌 "의무"임을 강조
3. 최소 대화 개수 명시 (예: 5-7개 교환)
4. 시각적으로 눈에 띄는 강조 (━━━ 박스, ⚠️ 이모지 등)
5. 좋은 예시 4-5개 (다양한 상황의 대화)
6. 나쁜 예시 3-4개 (간접화법, 인용부호 없음)

프롬프트 구조:
- 첫 줄: 강력한 경고문
- 대화 형식: 구체적 예시
- 문장 구조: 11-15단어
- 단락 구조: 2-3문장
- 검증 지시: "인용부호 개수 세기"

출력: 완전한 시스템 프롬프트 코드 (Python f-string 형식)
```

---

### 프롬프트 3: 태스크 프롬프트 개선

```
prose_generator.py의 태스크 프롬프트(lines 282-309)를 개선해주세요:

현재 문제:
- "Key dialogue: {dialogue_text}" 부분이 비어있으면 AI가 대화를 건너뜀
- 대화 배치 패턴이 권장사항일 뿐, 강제되지 않음

개선 방향:
1. 시뮬레이션 대화가 비어있어도 작동하는 로직
2. 대화 생성을 "체크리스트" 형식으로 변경:
   ```
   필수 대화 생성 체크리스트:
   [ ] 1번째 대화 교환 (2-3 문장)
   [ ] 2번째 대화 교환 (2-3 문장)
   [ ] 3번째 대화 교환 (2-3 문장)
   ...
   ```
3. 대화 주제 힌트 제공:
   - 캐릭터 간 관계/갈등
   - 정보 교환
   - 감정 표현
4. 검증 단계 추가: "생성 후 인용부호 개수 세기"

출력: 완전한 태스크 프롬프트 코드
```

---

### 프롬프트 4: 대화 창작 가이드 추가

```
prose_generator.py에 "대화 창작 가이드"를 추가해주세요.

목적: 시뮬레이션에 대화가 없어도, AI가 자연스러운 대화를 만들 수 있도록

포함 내용:
1. 장면별 대화 주제:
   - 학술 발표 → 질문/답변, 전문용어 설명
   - 비밀 제안 → 조건 협상, 의심 표현
   - 실험실 조사 → 데이터 토론, 우려 공유

2. 대화 패턴:
   - 정보 전달형: "데이터를 보니..." → "어떻게 해석하죠?" → "내 생각엔..."
   - 갈등형: "이건 위험해" → "선택의 여지가 없어" → "하지만..."
   - 신뢰 구축형: "믿어도 될까?" → "증거가 있어" → "알겠어"

3. 캐릭터별 말투:
   - 수민: 신중, 질문 많음, 전문적
   - 벤: 설득적, 실용적
   - 카를로스: 경계적, 짧은 답변

위치: 시스템 프롬프트 또는 별도 섹션
출력: 추가할 코드
```

---

### 프롬프트 5: 전체 수정 및 테스트

```
최종 작업:

1. 위의 개선사항을 모두 적용하여 에피소드를 제외한 모든 코드를 수정
2. 다음 순서로 재생성:
   ```bash
   python3 quality_adaptive_generator.py \
     --episode-id ep02_nsa_funding \
     --episode-config config/episodes/ep02_nsa_funding.yaml \
     --protagonist kim_sumin \
     --target-words 800 --scenes 3 \
     --target-score 0.80 --max-iterations 1
   ```
3. 품질 측정:
   ```bash
   python3 quality_analyzer.py output/ep02_nsa_funding_iter1_chapter.md
   ```
4. 대화 비율 확인:
   ```bash
   grep -c '"' output/ep02_nsa_funding_iter1_chapter.md
   ```

성공 기준:
- EP02 점수: 0.800+ (B등급)
- 대화 비율: 18-25%
- 인용부호: 10개 이상

EP02가 성공하면, 동일한 방법으로 EP01, EP03 재생성하여 균일성 확인.

최종 목표:
- 모든 에피소드 평균: 0.800+
- 표준편차: 0.05 이하 (균일성)
```

---

## 🎯 핵심 전략 요약

### 1. 대화 생성 강제화
```python
# 현재 (약한 권장)
"Dialogue: 3-5 QUOTED exchanges (20% of total)"

# 개선 (강력한 의무)
"⚠️ MANDATORY: You MUST create dialogue even if source has none"
"MINIMUM 20% dialogue is NON-NEGOTIABLE"
"Rejection criteria: Less than 5 quoted exchanges"
```

### 2. 대화 주제 자동 제안
```python
# Characters present에서 자동 생성
characters = ["Kim Sumin", "Ben Hartley", "Dr. Moreno"]
dialogue_topics = f"""
DIALOGUE TOPICS (create conversations about):
1. {characters[0]} questioning {characters[1]} about proposal details
2. {characters[1]} explaining technical aspects to {characters[0]}
3. Tension/conflict between {characters[0]} and {characters[2]}
"""
```

### 3. 검증 메커니즘
```python
# 프롬프트에 추가
"BEFORE SUBMITTING YOUR TEXT:
1. Count quotation marks: Should be {num_dialogues * 2}+
2. Calculate dialogue %: Count words in quotes / total words
3. If below 20%, ADD MORE DIALOGUE NOW"
```

---

## 📊 예상 결과

개선 후 목표:
```
Episode  | 현재 점수 | 목표 점수 | 대화 % | 상태
---------|----------|----------|--------|------
EP01     | 0.784    | 0.820+   | 20%+   | ✅
EP02     | 0.776    | 0.820+   | 20%+   | ✅
EP03     | 0.762    | 0.820+   | 20%+   | ✅
EP04     | 0.845    | 0.840+   | 20%+   | ✅

평균:     | 0.792    | 0.825+   | 20%+   | ✅
표준편차: | 0.034    | <0.020   |        | 균일성 ✅
```

