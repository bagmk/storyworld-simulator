You are editing YAML episode config files in the repository.

Target directory:
- config/episodes/

Scope:
- Scan all episode config files under config/episodes/
- EXCLUDE episode 1 / ep01 from modification
- Modify all other episode YAML files in place
- Keep YAML valid and readable

Goal:
Normalize later episode configs so they are closer to the improved ep01 style:
1. clues are shorter and more scene-safe
2. optional phases are added where clearly inferable
3. clue-to-phase mapping is added where clearly inferable
4. locations are made more structurally useful
5. unsupported speculative fields are NOT added unless already consumed by the codebase

IMPORTANT:
- Do NOT rewrite story meaning
- Do NOT invent new plot events
- Do NOT hardcode ep01-specific story content into other episodes
- Preserve each episode’s original narrative arc, clues, and intent
- Keep changes conservative and production-safe

==================================================
PRIMARY TRANSFORMATION RULES
==================================================

For each episode file except ep01:

1. KEEP these top-level fields unless clearly broken:
- episode.id
- date
- location
- summary
- introduced_clues
- resolved
- recommended_length
- pacing
- max_turns

2. REWRITE summary lightly, only if needed:
- preserve original meaning
- remove excessive repetition
- keep it readable and scene-oriented
- do not over-compress if summary contains important structural transitions
- keep major time jumps or location jumps explicit if already present

3. SHORTEN every clue content block
Current problem:
Many clue content blocks are written like miniature prose paragraphs, which encourages replay, repetition, and scene duplication.

Rewrite each clue’s `content` so that it becomes:
- 2 to 4 short lines max
- concrete
- observational
- scene-safe
- centered on facts, actions, visible artifacts, direct dialogue anchors, or state changes
- NOT padded with repeated mood narration or abstract commentary

Good clue content should preserve:
- what is seen / said / shown / handed over / discovered
- one key line if needed
- one concrete object, number, or signal if important

Bad clue content to reduce:
- long emotional explanation
- repeated gravitas words
- interpretive over-narration
- mini-novel paragraphs
- repeated “this was dangerous / heavy / meaningful” phrasing

4. PRESERVE trigger / inject_threshold / inject_method unless they are obviously malformed
- only lightly normalize wording if needed
- do not aggressively redesign trigger semantics in this pass

==================================================
PHASE SUPPORT RULES
==================================================

Add `phases:` only when the episode clearly contains distinct scene blocks or time/location stages.

Examples of when to add phases:
- conference hall -> corridor -> lounge
- office -> lab -> meeting room
- “two weeks later” / “later that night” / “next morning”
- public scene -> private follow-up
- field site -> return to HQ
- event venue -> home/lab/office aftermath

If phases are added:
- keep them lightweight
- 2 to 5 phases usually
- each phase should have:
  - id
  - location
  - description

Use generic phase ids derived from the episode itself, for example:
- corridor_encounter
- post_event_lounge
- lab_review
- night_followup
- office_aftermath
- cryolab_investigation

Do NOT force phases if the episode is truly single-scene.

==================================================
ALLOWED_PHASE RULES
==================================================

If phases are added and a clue clearly belongs to one phase, add:
- allowed_phase: <phase_id>

Only assign allowed_phase when it is reasonably clear from:
- summary
- clue content
- obvious time/location markers
- clue order

Do NOT assign allowed_phase if phase membership is too ambiguous.

==================================================
LOCATION NORMALIZATION RULES
==================================================

Improve location usefulness conservatively.

1. Keep top-level episode.location, but lightly normalize if it is too vague or bloated.
2. If phases are added, give each phase a specific location string.
3. Do NOT add unsupported fields like per-clue location unless already present in the schema.
4. Do NOT invent room names that are not supported by the episode text.
5. If a major time jump exists, reflect it in phases rather than stuffing all locations into one line.

==================================================
FIELDS TO AVOID ADDING UNLESS ALREADY SUPPORTED
==================================================

Do NOT add speculative new fields everywhere just because they seem useful.

In particular:
- Do NOT add `state_delta` unless the codebase already consumes it
- Do NOT add `scene_constraints` unless already supported in the repo
- Do NOT add custom metadata blobs unless clearly used by code

Preferred safe additions:
- phases
- allowed_phase

Only add fields that are likely to remain harmless and useful.

==================================================
CLUE REWRITE STYLE GUIDE
==================================================

When rewriting clue `content`:

Prefer:
- short factual lines
- visible artifacts
- one anchored quoted line if important
- clear distinction between characters’ narrative functions
- concrete numbers, labels, records, objects, IDs, brief asks/offers

Avoid:
- heavy mood interpretation
- repeated abstract theme words
- repeated “danger / burden / choice / pressure / future” language
- long physiological narration
- long explanatory commentary
- multiple emotional conclusions inside one clue

If two clues in the same episode perform almost the same function:
- keep both if both are structurally important
- but differentiate them more clearly through concise content
- avoid making both sound like the same warning paraphrase

==================================================
SPECIAL HANDLING FOR MULTI-TIMELINE OR TIME-JUMP EPISODES
==================================================

If an episode contains an explicit jump like:
- two weeks later
- later that night
- next morning
- back at HQ
then reflect that in phases.

Do not flatten pre-jump and post-jump material into one undifferentiated scene block.

==================================================
OUTPUT REQUIREMENTS
==================================================

Modify the files in place.

After editing, provide:
1. A short summary of which files were changed
2. For each file:
   - whether phases were added
   - whether allowed_phase was added to clues
   - whether clues were shortened
3. A note listing any files where phase inference was too ambiguous to apply safely
4. A note listing any unsupported fields you intentionally did NOT add (for example state_delta)

==================================================
QUALITY BAR
==================================================

Your edits should make the YAML:
- easier for the simulation to use
- less likely to cause repeated scene replay
- less likely to over-script miniature prose inside clues
- more structurally legible
- still faithful to the original episode intent

Do not over-engineer.
Be conservative, clear, and consistent.새로 고