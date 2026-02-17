"""
Prose Generator for the AI Story Simulation Engine.

Generates literary-quality prose from distilled scenes + original YAML beats.
This replaces the old novel_generator.py pipeline that worked from raw turn logs.

Key differences from novel_generator.py:
  - Reads original YAML beats directly (not just simulation output)
  - Works from DistilledScene objects (compressed, deduplicated)
  - Generates prose per-scene with beat-aware context
  - Single coherent LLM call per scene (not per-chunk of turns)
  - Controls tone, pacing, and episode position explicitly

Pipeline:
  1. Receive list of DistilledScene objects + episode config
  2. For each scene: generate literary prose grounded in YAML beats + distilled content
  3. Generate internal monologue transitions between scenes
  4. Combine into a single chapter
  5. Polish for consistency and target word count
  6. Write final chapter as Markdown
"""

from __future__ import annotations
import json
import logging
import re
from pathlib import Path
from typing import Optional

from .llm_client import LLMClient
from .scene_distiller import DistilledScene

logger = logging.getLogger(__name__)


class ProseGenerator:
    """
    Generates literary prose from distilled scenes and YAML beat definitions.

    Parameters
    ----------
    llm : LLMClient
        Used with premium model for narrative generation.
    episode_config : dict
        Original YAML episode configuration.
    output_dir : str
        Directory where chapter .md files are saved.
    """

    def __init__(
        self,
        llm: LLMClient,
        episode_config: dict,
        output_dir: str = "output",
    ) -> None:
        self.llm = llm
        self.episode_config = episode_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public: Generate Chapter
    # ------------------------------------------------------------------ #

    def generate_chapter(
        self,
        scenes: list[DistilledScene],
        protagonist_name: str = "Kim Sumin",
        style: str = "first_person",
        target_words: int = 3500,
    ) -> str:
        """
        Generate a full novel chapter from distilled scenes.

        Returns path to the generated .md file.
        """
        episode_id = self.episode_config.get("id", "unknown")
        logger.info(
            "Generating chapter for %s: %d scenes, target %d words",
            episode_id, len(scenes), target_words,
        )

        # Calculate per-scene word budget
        words_per_scene = target_words // max(len(scenes), 1)
        # Climax scenes get 30% more, opening/resolution 15% less
        scene_budgets = self._calculate_scene_budgets(scenes, target_words)

        # Build episode-level context
        episode_context = self._build_episode_context(protagonist_name)

        # Generate title
        title = self._generate_title(scenes, episode_context)

        # Generate prose for each scene
        prose_sections: list[str] = []
        for i, scene in enumerate(scenes):
            prev_section = prose_sections[-1] if prose_sections else None
            section = self._generate_scene_prose(
                scene=scene,
                scene_index=i,
                total_scenes=len(scenes),
                episode_context=episode_context,
                protagonist_name=protagonist_name,
                style=style,
                word_budget=scene_budgets[i],
                prev_section_tail=prev_section[-300:] if prev_section else None,
            )
            prose_sections.append(section)
            logger.info(
                "Scene %d/%d '%s': %d words",
                i + 1, len(scenes), scene.title, len(section.split()),
            )

        # Generate transitions between scenes
        combined = self._combine_with_transitions(
            prose_sections, scenes, episode_context, style,
        )

        # Polish
        final = self._polish(combined, target_words, style)

        # Write
        out_path = self.output_dir / f"{episode_id}_chapter.md"
        self._write_chapter(out_path, title, final, episode_id, scenes)

        word_count = len(final.split())
        logger.info("Chapter written: %s (%d words)", out_path, word_count)
        return str(out_path)

    # ------------------------------------------------------------------ #
    # Episode Context
    # ------------------------------------------------------------------ #

    def _build_episode_context(self, protagonist_name: str) -> dict:
        """Build rich episode context from YAML config."""
        ep = self.episode_config
        ep_id = str(ep.get("id", ""))

        # Extract episode number
        ep_num = 0
        for part in ep_id.split("_"):
            digits = "".join(c for c in part if c.isdigit())
            if digits:
                ep_num = int(digits)
                break

        pacing = ep.get("pacing", "normal")
        pacing_tone = {
            "slow": "contemplative, observational, rich in sensory detail and internal reflection",
            "normal": "balanced between action and reflection, natural rhythm",
            "tense": "tight, anxious, every detail feels loaded with significance",
            "fast": "urgent, compressed, events cascade without time to process",
        }.get(pacing, "balanced")

        # Beat summaries
        clues = ep.get("introduced_clues", [])
        beat_descriptions = [c.get("content", "") for c in clues if isinstance(c, dict)]

        return {
            "episode_number": ep_num,
            "total_episodes": 49,
            "location": ep.get("location", ""),
            "date": ep.get("date", ""),
            "summary": ep.get("summary", ""),
            "pacing": pacing,
            "pacing_tone": pacing_tone,
            "protagonist": protagonist_name,
            "beat_descriptions": beat_descriptions,
            "recommended_length": ep.get("recommended_length", 3500),
        }

    def _load_quality_feedback(self, episode_id: str) -> str:
        """
        Load reinforcement feedback generated by quality_adaptive_generator.
        Returns formatted text block or empty string.
        """
        safe_episode_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", episode_id or "unknown")
        feedback_path = self.output_dir / f"quality_feedback_{safe_episode_id}.json"
        if not feedback_path.exists():
            return ""

        try:
            payload = json.loads(feedback_path.read_text(encoding="utf-8"))
        except Exception:
            return ""

        target_episode = payload.get("episode_id")
        if target_episode and target_episode != episode_id:
            return ""

        weak_metrics = payload.get("weak_metrics", [])
        directives = payload.get("directives", [])
        if not weak_metrics and not directives:
            return ""

        lines = ["REINFORCEMENT FEEDBACK FROM PREVIOUS ITERATION:"]
        if weak_metrics:
            lines.append(f"- Weak metrics: {', '.join(weak_metrics)}")
        for d in directives:
            lines.append(f"- {d}")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Scene Prose Generation
    # ------------------------------------------------------------------ #

    def _generate_scene_prose(
        self,
        scene: DistilledScene,
        scene_index: int,
        total_scenes: int,
        episode_context: dict,
        protagonist_name: str,
        style: str,
        word_budget: int,
        prev_section_tail: Optional[str] = None,
    ) -> str:
        """Generate literary prose for one distilled scene."""
        pov = "first person" if style == "first_person" else "third person close"
        ep = episode_context
        episode_id = str(self.episode_config.get("id", ""))
        reinforcement_feedback = self._load_quality_feedback(episode_id)

        # Format scene content
        has_source_dialogue = bool(scene.key_dialogue)
        dialogue_text = "\n".join(
            f"  {d.get('speaker', '?')}: \"{d.get('line', '')}\""
            for d in scene.key_dialogue
        ) or "  (no key dialogue in simulation data)"

        actions_text = "\n".join(
            f"  - {a}" for a in scene.key_actions
        ) or "  (no key actions)"

        discoveries_text = "\n".join(
            f"  - {d}" for d in scene.discoveries
        ) or "  (no discoveries)"

        # Beat context — what YAML says should happen
        beat_context = ""
        if scene.beat_references:
            matched_beats = [
                b for b in ep["beat_descriptions"]
                if any(ref in str(b) for ref in scene.beat_references)
            ]
            if matched_beats:
                beat_context = "Original story beats for this scene:\n" + \
                    "\n".join(f"  - {b}" for b in matched_beats)

        # Position description
        if scene_index == 0:
            position = "OPENING — establish atmosphere, setting, and protagonist's state of mind"
        elif scene_index == total_scenes - 1:
            position = "CLOSING — bring threads together, leave resonance, hint at what's next"
        elif scene.pacing == "climax":
            position = "CLIMAX — this is the pivotal moment; give it full emotional weight"
        else:
            position = f"MIDDLE (scene {scene_index + 1}/{total_scenes}) — develop naturally"

        # Calculate targets - keep dialogue explicit and measurable
        dialogue_words = int(word_budget * 0.22)  # 22% dialogue target
        num_dialogues = max(5, dialogue_words // 14)  # ~14 words per exchange, minimum 5 exchanges

        # Dialogue creation guidance that works even when source dialogue is empty
        chars = scene.characters_present or [protagonist_name]
        c1 = chars[0] if len(chars) > 0 else protagonist_name
        c2 = chars[1] if len(chars) > 1 else "상대 인물"
        c3 = chars[2] if len(chars) > 2 else "제3 인물"

        title_lower = scene.title.lower()
        if any(k in title_lower for k in ["발표", "presentation", "academic"]):
            scene_dialogue_topics = (
                f"  - 질문/답변: {c1}이 핵심 수치와 한계를 설명하고 {c2}가 검증을 요구\n"
                f"  - 전문용어 해설: 모델/실험 조건을 풀어 말해 오해를 줄이기\n"
                f"  - 압박 대응: 공격적인 질문에 침착하게 근거를 제시"
            )
        elif any(k in title_lower for k in ["funding", "offer", "proposal", "제안"]):
            scene_dialogue_topics = (
                f"  - 조건 협상: {c1}과 {c2}가 자금/기한/통제권을 두고 밀고 당기기\n"
                f"  - 의심 표현: {c1}이 리스크와 숨은 의도를 따져 묻기\n"
                f"  - 정보 교환: {c2}가 숫자와 조건으로 설득하고 {c1}이 반문"
            )
        elif any(k in title_lower for k in ["lab", "cryo", "investigation", "조사", "실험실"]):
            scene_dialogue_topics = (
                f"  - 데이터 토론: 로그/인보이스/장비 수량 불일치를 짚기\n"
                f"  - 우려 공유: {c1}이 위반 가능성을 제기하고 {c2}가 방어\n"
                f"  - 긴장 고조: {c3}가 개입해 책임 범위를 흐리거나 압박"
            )
        else:
            scene_dialogue_topics = (
                f"  - 관계/갈등: {c1}과 {c2}의 신뢰 또는 충돌을 드러내기\n"
                f"  - 정보 교환: 장면 목표 달성에 필요한 사실을 대사로 전달\n"
                f"  - 감정 표현: 두려움/의심/결심을 직접 발화로 보여주기"
            )

        dialogue_checklist = "\n".join(
            f"[ ] {i}번째 대화 교환 (2-3문장, 직접 인용 1개 이상)"
            for i in range(1, num_dialogues + 1)
        )

        source_dialogue_rule = (
            "Simulation dialogue exists: reuse key intent but rewrite naturally."
            if has_source_dialogue
            else "Simulation dialogue is empty: you MUST invent natural dialogue from actions/discoveries."
        )

        system = (
            f"⚠️ MANDATORY DIALOGUE RULE: EVEN IF SOURCE HAS NO DIALOGUE, YOU MUST CREATE IT.\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"You are writing psychological thriller in Korean literary style (김영하).\n"
            f"If dialogue quota is missed, output is invalid.\n\n"
            f"{reinforcement_feedback + chr(10) + chr(10) if reinforcement_feedback else ''}"
            f"NON-NEGOTIABLE CONSTRAINTS:\n"
            f"• Sentence length: 11-15 words (Korean spacing count).\n"
            f"• Paragraph structure: 2-3 sentences per paragraph, then blank line.\n"
            f"• Dialogue ratio: 20-23% of total words.\n"
            f"• Dialogue count: at least {num_dialogues} quoted exchanges.\n"
            f"• Dialogue is mandatory, not optional.\n\n"
            f"GOOD DIALOGUE EXAMPLES (direct quotes required):\n"
            f"✓ \"이 제안은 신뢰할 수 있나요?\" 나는 숨을 고르며 물었다.\n"
            f"✓ 벤이 낮게 말했다. \"우리는 현실적이어야 해. 선택은 불가피해.\"\n"
            f"✓ \"데이터를 보니 편차가 커요.\" 수민이 그래프를 짚으며 말했다.\n"
            f"✓ 카를로스가 눈을 좁혔다. \"지금은 질문보다 실행이 먼저야.\"\n"
            f"✓ \"증거를 보여줘요.\" 내가 말하자 벤이 태블릿을 내밀었다.\n\n"
            f"BAD EXAMPLES (forbidden):\n"
            f"✗ 벤이 현실적이어야 한다고 말했다. (indirect speech)\n"
            f"✗ 나는 신뢰할 수 있는지 물었다. (no quotes)\n"
            f"✗ 그는 위험하다고 경고했다. (no spoken line)\n"
            f"✗ 대화 없이 설명만 이어졌다. (dialogue quota fail)\n\n"
            f"DIALOGUE CREATION GUIDE:\n"
            f"Scene-tailored topics:\n{scene_dialogue_topics}\n\n"
            f"Patterns you can apply:\n"
            f"  - 정보 전달형: \"데이터를 보니...\" → \"어떻게 해석하죠?\" → \"내 생각엔...\"\n"
            f"  - 갈등형: \"이건 위험해\" → \"선택의 여지가 없어\" → \"하지만...\"\n"
            f"  - 신뢰 구축형: \"믿어도 될까?\" → \"증거가 있어\" → \"알겠어\"\n\n"
            f"Voice hints:\n"
            f"  - 수민: 신중, 질문 많음, 전문적 어휘.\n"
            f"  - 벤: 설득적, 실용적, 결론 지향.\n"
            f"  - 카를로스: 경계적, 짧고 단단한 문장.\n\n"
            f"SELF-CHECK BEFORE FINAL OUTPUT:\n"
            f"1) Count quotation marks (\") >= {num_dialogues * 2}\n"
            f"2) Ensure quoted words are 20-23% of total\n"
            f"3) If below target, add more direct dialogue now."
        )

        continuity = ""
        if prev_section_tail:
            continuity = (
                f"\n## Previous Section Ending\n"
                f"...{prev_section_tail}\n"
                f"Continue naturally from this point.\n"
            )

        prompt = (
            f"## Episode Context\n"
            f"Episode {ep['episode_number']}/{ep['total_episodes']}\n"
            f"Location: {ep['location']}\n"
            f"Pacing: {ep['pacing']} — {ep['pacing_tone']}\n"
            f"Protagonist: {ep['protagonist']}\n\n"
            f"## Scene: {scene.title}\n"
            f"Position: {position}\n"
            f"Emotional arc: {scene.emotional_arc}\n"
            f"Location: {scene.location}\n"
            f"Characters: {', '.join(scene.characters_present)}\n\n"
            f"## Essential Content (from simulation)\n"
            f"Key dialogue:\n{dialogue_text}\n"
            f"Dialogue source status: {source_dialogue_rule}\n\n"
            f"Key actions:\n{actions_text}\n\n"
            f"Discoveries/revelations:\n{discoveries_text}\n\n"
            f"Scene summary: {scene.narrative_summary}\n\n"
        )

        if beat_context:
            prompt += f"## Original Story Beats\n{beat_context}\n\n"

        prompt += (
            f"{continuity}"
            f"## Task: Write {word_budget} words\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚠️  CRITICAL REQUIREMENT: DIALOGUE WITH QUOTATION MARKS\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"You MUST include AT LEAST {num_dialogues} dialogue exchanges.\n"
            f"{source_dialogue_rule}\n"
            f"Each exchange MUST use quotation marks: \"대사 내용\"\n\n"
            f"CORRECT FORMAT:\n"
            f"  벤이 낮게 말했다. \"우리는 현실적이어야 해.\"\n"
            f"  \"정말 신뢰할 수 있나요?\" 나는 물었다.\n"
            f"  카를로스가 대답했다. \"물론이지. 계약서를 봐.\"\n\n"
            f"WRONG FORMAT (DO NOT USE):\n"
            f"  ✗ 벤이 현실적이어야 한다고 말했다. (NO QUOTES!)\n"
            f"  ✗ 나는 신뢰에 대해 물었다. (NO QUOTES!)\n\n"
            f"WORD COUNT BREAKDOWN:\n"
            f"• Narrative/Description: ~{word_budget - dialogue_words} words (78%)\n"
            f"• Dialogue (QUOTED): ~{dialogue_words} words (22%) = {num_dialogues}+ exchanges\n\n"
            f"MANDATORY DIALOGUE CHECKLIST (complete all):\n"
            f"{dialogue_checklist}\n\n"
            f"DIALOGUE TOPIC HINTS:\n"
            f"• Relationship/conflict between present characters\n"
            f"• Information exchange that moves plot forward\n"
            f"• Emotional expression under pressure\n\n"
            f"DIALOGUE PLACEMENT PATTERN (enforced):\n"
            f"1. Action paragraph (2-3 sentences)\n"
            f"2. Dialogue paragraph (2-3 exchanges)\n"
            f"3. Action paragraph (2-3 sentences)\n"
            f"4. Dialogue paragraph (2-3 exchanges)\n"
            f"Repeat until quota is satisfied.\n\n"
            f"SENTENCE LENGTH: 11-15 words each. Use -고/-며/-면서 to combine clauses.\n\n"
            f"FINAL VERIFICATION (required):\n"
            f"1) Count quotation marks: {num_dialogues * 2}+ required\n"
            f"2) Estimate dialogue ratio: 20-23% required\n"
            f"3) If requirement fails, revise before finishing."
        )

        # Korean long-form prose often needs a larger token budget than English
        # for the same word target; keep a higher ceiling to avoid truncation.
        scene_max_tokens = min(4800, max(1800, word_budget * 5))

        return self.llm.chat(
            [{"role": "user", "content": prompt}],
            system=system,
            purpose="prose_scene_gen",
            use_premium=True,
            temperature=0.75,
            max_tokens=scene_max_tokens,
        )

    # ------------------------------------------------------------------ #
    # Transitions
    # ------------------------------------------------------------------ #

    def _combine_with_transitions(
        self,
        sections: list[str],
        scenes: list[DistilledScene],
        episode_context: dict,
        style: str,
    ) -> str:
        """Combine prose sections with internal monologue transitions."""
        if len(sections) <= 1:
            return sections[0] if sections else ""

        pov = "first person" if style == "first_person" else "third person close"
        parts = [sections[0]]

        for i in range(1, len(sections)):
            prev_scene = scenes[i - 1] if i - 1 < len(scenes) else None
            next_scene = scenes[i] if i < len(scenes) else None

            bridge = self._generate_transition(
                prev_tail=sections[i - 1][-300:],
                next_head=sections[i][:300],
                prev_scene=prev_scene,
                next_scene=next_scene,
                pov=pov,
            )
            parts.append(bridge)
            parts.append(sections[i])

        return "\n\n".join(parts)

    def _generate_transition(
        self,
        prev_tail: str,
        next_head: str,
        prev_scene: Optional[DistilledScene],
        next_scene: Optional[DistilledScene],
        pov: str,
    ) -> str:
        """Generate a 2-4 sentence transition between scenes."""
        prev_title = prev_scene.title if prev_scene else "previous scene"
        next_title = next_scene.title if next_scene else "next scene"

        prompt = (
            f"Write a 2-4 sentence {pov} internal monologue or brief scene transition.\n\n"
            f"Leaving: '{prev_title}'\n"
            f"...{prev_tail}\n\n"
            f"Entering: '{next_title}'\n"
            f"{next_head}...\n\n"
            f"The transition should feel like a natural breath between moments — "
            f"a thought, a physical movement, a shift in attention. "
            f"Write in Korean. Write ONLY the transition text."
        )
        return self.llm.chat(
            [{"role": "user", "content": prompt}],
            purpose="prose_transition",
            use_premium=True,
            temperature=0.7,
            max_tokens=200,
        )

    # ------------------------------------------------------------------ #
    # Polish
    # ------------------------------------------------------------------ #

    def _polish(self, text: str, target_words: int, style: str) -> str:
        """Final consistency and word count pass."""
        current = len(text.split())
        pov = "first person" if style == "first_person" else "third person close"

        if current < target_words * 0.7:
            instruction = (
                f"The chapter is {current} words but should be ~{target_words}. "
                f"Expand with additional sensory detail, deeper internal reflection, "
                f"and richer scene-setting. Do NOT add new plot events."
            )
        elif current > target_words * 1.4:
            instruction = (
                f"The chapter is {current} words but should be ~{target_words}. "
                f"Tighten the prose: remove repetition, merge redundant descriptions, "
                f"cut filler phrases. Preserve all key events and dialogue."
            )
        else:
            instruction = (
                f"The chapter is {current} words (target: ~{target_words}). "
                f"Do a final review for: consistent {pov} voice, smooth flow, "
                f"no abrupt tonal shifts. Make only minor improvements."
            )

        prompt = (
            f"{instruction}\n\n"
            f"Also ensure:\n"
            f"- Consistent {pov} voice throughout (Korean)\n"
            f"- No simulation artifacts (turn numbers, metadata, labels)\n"
            f"- Paragraphs should usually contain 2-4 sentences\n"
            f"- Average sentence length should stay around 9-16 words\n"
            f"- Dialogue ratio should stay near 20-23%\n"
            f"- Natural paragraph breaks at emotional beats\n"
            f"- No identical phrases or descriptions repeated\n\n"
            f"Full chapter text:\n\n{text}"
        )

        polished = self.llm.chat(
            [{"role": "user", "content": prompt}],
            purpose="prose_polish",
            use_premium=True,
            temperature=0.4,
            max_tokens=min(16000, max(6000, target_words * 5)),
        )
        return self._normalize_paragraphs(polished)

    def _normalize_paragraphs(self, text: str) -> str:
        """
        Post-process overly long paragraphs so analyzer-facing structure stays stable.
        Keeps content intact and only adjusts paragraph breaks.
        """
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        normalized: list[str] = []

        for block in blocks:
            if block.startswith("#") or block.startswith("*") or block.startswith("---"):
                normalized.append(block)
                continue

            sentences = [
                s.strip()
                for s in re.split(r'(?<=[.!?])\s+', block)
                if s.strip()
            ]

            if len(sentences) <= 4:
                normalized.append(" ".join(sentences) if sentences else block)
                continue

            # Split long blocks into 3-sentence chunks.
            for i in range(0, len(sentences), 3):
                chunk = " ".join(sentences[i:i + 3]).strip()
                if chunk:
                    normalized.append(chunk)

        return "\n\n".join(normalized)

    # ------------------------------------------------------------------ #
    # Scene Word Budget
    # ------------------------------------------------------------------ #

    def _calculate_scene_budgets(
        self, scenes: list[DistilledScene], target_words: int
    ) -> list[int]:
        """Distribute word budget based on scene pacing."""
        if not scenes:
            return []

        weights = []
        for s in scenes:
            if s.pacing == "climax":
                weights.append(1.3)
            elif s.pacing in ("opening", "resolution"):
                weights.append(0.85)
            else:
                weights.append(1.0)

        total_weight = sum(weights)
        # Reserve ~15% for transitions
        prose_budget = int(target_words * 0.85)
        budgets = [int(prose_budget * w / total_weight) for w in weights]
        return budgets

    # ------------------------------------------------------------------ #
    # Title
    # ------------------------------------------------------------------ #

    def _generate_title(
        self, scenes: list[DistilledScene], episode_context: dict
    ) -> str:
        """Generate a literary chapter title."""
        summaries = " / ".join(s.title for s in scenes[:4])
        prompt = (
            f"Suggest a literary chapter title (3-8 words, Korean) for:\n"
            f"Episode {episode_context['episode_number']}: {episode_context['summary'][:200]}\n"
            f"Scenes: {summaries}\n"
            f"Reply with only the title, no quotes or punctuation."
        )
        return self.llm.chat(
            [{"role": "user", "content": prompt}],
            purpose="prose_title",
            use_premium=False,
            temperature=0.9,
            max_tokens=30,
        ).strip()

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #

    def _write_chapter(
        self,
        path: Path,
        title: str,
        content: str,
        episode_id: str,
        scenes: list[DistilledScene],
    ) -> None:
        """Write the chapter Markdown file."""
        scene_summary = "\n".join(
            f"  {i + 1}. {s.title} ({s.pacing})"
            for i, s in enumerate(scenes)
        )
        header = (
            f"# {title}\n\n"
            f"*Episode: {episode_id}*\n\n"
            f"---\n\n"
        )
        footer = (
            f"\n\n---\n\n"
            f"*Scene structure:*\n{scene_summary}\n"
        )
        path.write_text(header + content + footer, encoding="utf-8")
