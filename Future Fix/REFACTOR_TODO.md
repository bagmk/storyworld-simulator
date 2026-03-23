# 나중에 할 리팩토링 2개

## 1. `_split_korean_sentences` 중복 제거

**현황:** `scene_distiller.py`와 `prose_generator.py`에 동일한 staticmethod가 복사되어 있음.

**작업 내용:**
1. `src/novel_writer/scene_state.py` 맨 아래에 아래 함수를 추가:
   ```python
   def split_korean_sentences(text: str, min_chars: int = 220, min_tokens: int = 12) -> list[str]:
       """Split Korean prose into sentences for structural analysis."""
       import re
       _RE = re.compile(r'(?<=[.!?…])\s+|(?<=[다요죠]\.)\s+')
       ...  # scene_distiller.py:2790 또는 prose_generator.py:4433 의 본문 그대로 복사
   ```
2. `scene_distiller.py`에서 `_split_korean_sentences` staticmethod 삭제, `split_korean_sentences` import 추가
3. `prose_generator.py`에서 동일하게 처리
4. 두 파일에서 `self._split_korean_sentences(...)` → `split_korean_sentences(...)` 호출로 변경
5. 임포트 테스트 + 실제 함수 결과 비교 확인

**주의:** 테스트가 없으므로 변경 전후 동일한 텍스트로 출력 비교 필수.

---

## 2. `reader_feedback` 중복 캐시 제거

**현황:** `polisher.py`, `prose_generator.py`, `scene_distiller.py` 생성자에서 동일 패턴 반복:
```python
self.reader_profile: ReaderProfile = build_reader_profile(reader_feedback)
self.reader_feedback = self.reader_profile.as_dict()  # ← 이게 중복
```
`self.reader_feedback`은 항상 `self.reader_profile.as_dict()`와 동일한 값.

**작업 내용:**
1. 각 파일에서 `self.reader_feedback = ...` 줄 삭제
2. `self.reader_feedback`을 읽는 모든 곳을 찾아 `self.reader_profile.as_dict()`로 교체
   ```bash
   grep -n "self\.reader_feedback" src/novel_writer/polisher.py
   grep -n "self\.reader_feedback" src/novel_writer/prose_generator.py
   grep -n "self\.reader_feedback" src/novel_writer/scene_distiller.py
   ```
3. `orchestrator.py`, `director.py`도 동일 패턴 있으면 함께 처리
4. 임포트 테스트 + smoke test 확인

**주의:** `self.reader_feedback`이 외부에서 직접 읽히는지 확인 필요 (타 모듈에서 `.reader_feedback` 접근하면 AttributeError).
