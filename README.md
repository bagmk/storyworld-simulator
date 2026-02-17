# AI Story Simulation Engine

A Python-based simulation system where AI agents interact as story characters. Story structure and character definitions live in external YAML files — no hardcoded content. Agent interactions are recorded in a database and translated into polished novel prose.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key
export OPENAI_API_KEY=sk-...

# 3. Run a simulation episode
python simulate.py \
    --episode  config/episodes/ep01.yaml \
    --characters config/characters.yaml \
    --world    config/world_facts.yaml \
    --budget   3.00

# 4. Generate the novel chapter
python generate_novel.py \
    --episode  ep01_the_anomaly \
    --protagonist dr_reyes \
    --words    3500
```

---

## Project Structure

```
ai_story_sim/
├── simulate.py          # CLI: run a simulation episode
├── generate_novel.py    # CLI: convert episode to novel chapter
│
├── models.py            # Core data classes (Agent, Memory, WorldState, etc.)
├── config_loader.py     # YAML config parsers
├── database.py          # SQLite persistence layer
├── llm_client.py        # OpenAI wrapper with budget tracking
├── director.py          # Director AI (invariants, clue injection, knowledge control)
├── orchestrator.py      # Turn-based simulation loop
├── novel_generator.py   # Narrative prose generation pipeline
│
├── config/
│   ├── characters.yaml       # Character definitions
│   ├── world_facts.yaml      # Hidden / discoverable / public facts
│   └── episodes/
│       └── ep01.yaml         # Episode beat config
│
├── data/
│   └── simulation.db         # SQLite database (auto-created)
│
└── output/
    ├── ep01_simulation.json  # Full interaction log
    ├── ep01_chapter.md       # Generated novel chapter
    └── ep01_debug.log        # Director AI intervention log
```

---

## Configuration Files

### `config/characters.yaml`

Define every character in your story world:

```yaml
characters:
  - id: "alice"
    name: "Alice Vance"
    role: "protagonist"          # protagonist / supporting / background
    bio: "Background and personality description..."
    invariants:
      - "Never betrays her sister, no matter the cost"
      - "Always tells the truth to people she trusts"
    goals:
      - "Find the missing research files"
      - "Expose the cover-up before the deadline"
    initial_relationships:
      bob: 0.7                   # -1.0 (hostile) to 1.0 (deep trust)
      carol: -0.3
```

### `config/episodes/ep01.yaml`

Define scene beats, required clues, and pacing for one episode:

```yaml
episode:
  id: "ep01_scene_name"
  date: "2043-03-15T09:00"
  location: "Research Lab — Building 7"
  summary: "Scene description visible to all agents..."

  introduced_clues:
    - id: "clue_001"
      content: "The access log shows a midnight login by a deactivated account"
      trigger: "agent reviews the terminal"
      inject_threshold: 0.5     # Director forces it at 50% episode progress if missed
      inject_method: "environmental_cue"  # environmental_cue / npc_question / object_placement

  resolved:
    - "Previous mystery thread is addressed"

  recommended_length: 3500     # Target words for novel chapter
  pacing: "tense"              # slow / normal / tense / fast
```

### `config/world_facts.yaml`

Define the three tiers of knowledge:

```yaml
world_facts:
  hidden:           # Director-only. Agents must NEVER reveal these.
    - content: "The true identity of the antagonist is..."

  discoverable:     # Agents can learn these through appropriate actions.
    - id: "clue_001"
      content: "The midnight login used account ID OSE-2040"
      trigger: "agent checks access logs"

  public:           # All agents know these from the start.
    - "The year is 2043"
    - "The lab is in a coastal city"
```

---

## How It Works

### Simulation Engine

The engine runs a **turn-based loop**:

1. **Agent selected** (round-robin across active agents)
2. **Context built** — world state, recent interactions, agent's filtered memory, relationship scores
3. **Agent generates** action/dialogue/inner-thought via LLM
4. **Director evaluates**:
   - Invariant check: does this violate core character rules?
   - Knowledge leak check: does this reveal hidden facts the agent shouldn't know?
   - If either fails → regenerate with correction prompt (up to 3 attempts)
5. **World state updated**
6. **All agent memories updated** (perspective-filtered)
7. **Clue discoveries tracked** → logged to database
8. **Director checks** whether clue injection is needed this turn
9. **Completion check** — Director can end early when the beat is complete (after minimum turns) or continue until max turns

### Director AI

The Director is a separate LLM call (using the premium model) that:

- **Enforces invariants** — "this character would never say that"
- **Prevents knowledge leakage** — "this character doesn't know that yet"  
- **Guards storyline alignment** — keeps turns in the current episode/milestone lane
- **Injects clues** — fires a scene event when a required clue hasn't surfaced
- **Tracks pacing** — nudges the story toward key beats at appropriate moments

All Director interventions are written to `output/<episode_id>_debug.log`.

### Novel Generator

After an episode completes, the novel generator:

1. Loads all interactions from the database
2. Filters to the protagonist's perspective (removes inner thoughts of other characters)
3. Analyses the emotional arc
4. Identifies key moments (high emotion, director events, discoveries)
5. Generates prose section by section using the premium model
6. Bridges sections with internal monologue
7. Polishes the full chapter for consistent voice and target word count
8. Writes `output/<episode_id>_chapter.md`

---

## CLI Reference

### `simulate.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--episode` | *(required)* | Path to episode YAML |
| `--characters` | *(required)* | Path to characters YAML |
| `--world` | `config/world_facts.yaml` | Path to world_facts YAML |
| `--storyline` | `config/storyline.yaml` | Path to long-arc storyline YAML (optional guardrails) |
| `--model` | `gpt-4o-mini` | Model for agent turns |
| `--premium` | `gpt-4o` | Model for Director AI |
| `--budget` | `3.00` | USD cap for the episode |
| `--output` | `output/` | Output directory |
| `--db` | `data/simulation.db` | SQLite database path |
| `--debug` | off | Enable verbose logging |

### `generate_novel.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--episode` | *(required)* | Episode ID (from YAML `id` field) |
| `--protagonist` | *(required)* | Agent ID for POV |
| `--model` | `gpt-4o-mini` | Default model |
| `--premium` | `gpt-4o` | Premium model for narrative gen |
| `--budget` | `3.00` | USD cap |
| `--words` | `3500` | Target chapter word count |
| `--style` | `first_person` | `first_person` or `third_person_close` |
| `--output` | `output/` | Output directory |
| `--db` | `data/simulation.db` | SQLite database path |

---

## Budget Optimisation

The system uses **two model tiers**:

- **Cheap model** (`gpt-4o-mini`) — all agent turns (~80% of calls)
- **Premium model** (`gpt-4o`) — Director AI checks, narrative generation, arc analysis

Typical episode costs:

| Episode Length | Approx. Cost |
|---------------|-------------|
| 30-turn short | ~$0.15–0.40 |
| 60-turn standard | ~$0.40–0.90 |
| Novel chapter generation | ~$0.50–1.20 |

---

## Database Schema

The SQLite database at `data/simulation.db` stores everything:

| Table | Contents |
|-------|----------|
| `agents` | Character personas and roles |
| `episodes` | Episode configs and run status |
| `interactions` | Every turn's dialogue/action |
| `emotions` | Emotional state per turn per agent |
| `relationships` | Relationship value history |
| `world_states` | World state snapshots |
| `persona_deltas` | Personality evolution events |
| `clues` | Clue introduction tracking |
| `agent_knowledge` | What each agent has discovered |

All tables support full querying — filter by agent, episode, turn, emotion type, etc.

---

## Design Principles

- **Content-agnostic** — no hardcoded story details anywhere in engine code
- **YAML-driven** — all story content in external config files
- **Knowledge-separated** — agents cannot access hidden facts
- **Invariant-enforced** — characters stay consistent across the episode
- **Clue-tracked** — required story beats are guaranteed to surface
- **Budget-aware** — cost-optimised model tiering
- **Debuggable** — full Director intervention log
- **Modular** — simulation and novel generation are fully independent

---

## Requirements

- Python 3.10+
- OpenAI API key with access to GPT-4o / GPT-4o-mini
- See `requirements.txt` for Python packages
