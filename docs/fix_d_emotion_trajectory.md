# Fix D — Emotion Trajectory Preservation

## 목적

`SceneDistiller`가 raw turns를 압축할 때 감정 곡선(`emotion_trajectory`), 긴장 피크(`tension_peaks`), 관계 변화(`relationship_delta`)를 유실하지 않고 `DistilledScene`에 보존.
`ProseGenerator`가 이 데이터를 LLM 프롬프트에 주입해 문장 수준의 감정 일관성을 확보한다.

---

## 1. DistilledScene 확장 (`src/novel_writer/scene_distiller.py`)

### 변경 위치: `DistilledScene` dataclass (현재 line ~34)

```python
@dataclass
class DistilledScene:
    # ... 기존 필드들 ...

    # Fix D: 감정 궤적 필드 추가
    emotion_trajectory: dict[str, list[float]] = field(default_factory=dict)
    # 예: {"Sumin": [0.2, 0.4, 0.7, 0.9]}
    # 장면 내 각 주요 전환점에서의 감정 강도 (0.0=중립, 1.0=극강)

    tension_peaks: list[int] = field(default_factory=list)
    # 긴장감이 최고조에 달한 turn 인덱스 목록
    # 예: [5, 12] → turn 5와 12가 장면의 피크

    relationship_delta: dict[str, float] = field(default_factory=dict)
    # 장면 전후 캐릭터 관계 변화량
    # 예: {"Sumin→Moreno": +0.3, "Sumin→Miller": -0.1}
```

`to_dict()` 메서드에도 세 필드 추가:
```python
def to_dict(self) -> dict:
    return {
        # ... 기존 ...
        "emotion_trajectory": self.emotion_trajectory,
        "tension_peaks": self.tension_peaks,
        "relationship_delta": self.relationship_delta,
    }
```

---

## 2. SceneDistiller.distill() 확장

### 변경 위치: `_llm_distill()` 메서드 (현재 line ~222)

LLM distillation 프롬프트에 감정 추출 지시 추가:

```python
# 기존 지시 블록 끝에 추가:
extra = (
    "\n\nEach scene object must also include these emotion fields:\n"
    '  "emotion_trajectory": {{"CharName": [float, ...]}},  '
    "// list of 2-4 emotion intensity values (0.0=neutral, 1.0=peak) at key turns\n"
    '  "tension_peaks": [int, ...],  '
    "// turn indices (relative to turn_range start) where tension peaks\n"
    '  "relationship_delta": {{"A→B": float, ...}}  '
    "// net relationship change for each character pair in this scene\n"
)
```

### 변경 위치: `_parse_distilled_scenes()` (JSON 파싱 메서드)

파싱 후 각 `DistilledScene` 생성 시 세 필드 채우기:

```python
scene = DistilledScene(
    # ... 기존 필드 ...
    emotion_trajectory=raw.get("emotion_trajectory", {}),
    tension_peaks=[int(t) for t in raw.get("tension_peaks", [])],
    relationship_delta={k: float(v) for k, v in raw.get("relationship_delta", {}).items()},
)
```

### Fallback: `_fallback_chunk()` 메서드

fallback으로 생성된 씬에는 빈 값으로 초기화:
```python
emotion_trajectory={},
tension_peaks=[],
relationship_delta={},
```

---

## 3. ProseGenerator 감정 프롬프트 주입

### 변경 위치: `_build_scene_prompt()` (현재 line ~?)

각 씬의 prose 생성 프롬프트에 감정 데이터 주입:

```python
def _build_emotion_context(scene: DistilledScene) -> str:
    """감정 궤적 데이터를 한국어 프롬프트 블록으로 변환."""
    lines = []
    if scene.emotion_trajectory:
        for char, vals in scene.emotion_trajectory.items():
            if len(vals) >= 2:
                start_v, end_v = vals[0], vals[-1]
                direction = "상승" if end_v > start_v else "하강" if end_v < start_v else "유지"
                lines.append(f"- {char}의 감정: {start_v:.1f}→{end_v:.1f} ({direction})")
    if scene.tension_peaks:
        lines.append(f"- 긴장 피크 위치: 장면 {scene.tension_peaks} 전환점")
    if scene.relationship_delta:
        for pair, delta in scene.relationship_delta.items():
            sign = "+" if delta >= 0 else ""
            lines.append(f"- 관계 변화 {pair}: {sign}{delta:.2f}")
    if not lines:
        return ""
    return "【감정 궤적】\n" + "\n".join(lines)
```

프롬프트 빌더에서 호출:
```python
emotion_ctx = _build_emotion_context(scene)
if emotion_ctx:
    prompt += f"\n\n{emotion_ctx}\n이 감정 곡선이 문장의 밀도와 긴장감에 반영되도록 하라."
```

---

## 4. 구현 시 주의사항

1. **기존 씬 캐시 호환성**: `DistilledScene`은 dataclass이므로 기존 캐시된 JSON을 역직렬화할 때 세 필드가 없으면 `field(default_factory=...)` 덕분에 자동으로 빈값이 들어간다. 별도 마이그레이션 불필요.

2. **LLM 미반환 처리**: distiller LLM이 emotion 필드를 누락할 수 있다. `raw.get("emotion_trajectory", {})` 패턴으로 graceful fallback 처리.

3. **프롬프트 길이**: 감정 블록은 최대 5줄로 제한. scene이 많으면 token 비용 증가에 주의.

4. **테스트**: `tests/test_reader_feedback_guards.py`에 감정 필드 보존 테스트 추가 권장:
   ```python
   def test_emotion_trajectory_preserved():
       scene = DistilledScene(..., emotion_trajectory={"A": [0.2, 0.8]}, ...)
       d = scene.to_dict()
       assert d["emotion_trajectory"] == {"A": [0.2, 0.8]}
   ```

---

## 5. 구현 순서

1. `DistilledScene` dataclass 확장 (필드 추가 + to_dict 업데이트)
2. `_parse_distilled_scenes()` 파싱 로직 업데이트
3. `_llm_distill()` 프롬프트 지시 추가
4. `_fallback_chunk()` 빈값 초기화
5. `ProseGenerator._build_emotion_context()` 헬퍼 추가
6. `ProseGenerator._build_scene_prompt()` 주입 연결
7. 단위 테스트 추가
