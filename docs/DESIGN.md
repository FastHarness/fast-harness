# fast-harness — design and implementation

## 1. Goal and non-goals

The goal: when the same task is executed many times, gradually replace some of the
expensive teacher reviews with verified experience, while keeping the ability to detect
errors, correct them in place, and recover to a previously verified sub-goal. The intuition
that "the first few rounds are for learning, later rounds need fewer reviews" is a *trend*,
not a fixed schedule baked into the code.

This package does **not** train the policy, does **not** add a second learning model, does
**not** rewrite the task instruction, does **not** copy the simulator's code, and does
**not** roll back physical state. It also does not claim the teacher's visual judgement is
always correct.

Responsibilities are split four ways:

| Role | Owns | Does not own |
|---|---|---|
| Backend | observations, fresh policy proposals, execution, ACK, native success | how often the teacher is called |
| Reviewer | the previous chunk's result, the current stage, the next chunk's intent and actions | scheduling the long loop |
| Memory / Scheduler | accumulating experience, matching stages, deciding review frequency | inferring a grasp from joint values |
| Recovery / Harness | recovery state transitions, per-chunk scheduling, audits | bypassing backend action limits |

## 2. Why the main loop is driven from Python

In a purely teacher-driven controller the teacher itself issues `infer` and `execute`.
Even if you ask it to "look less often", each tool call still involves a model generation,
so you cannot guarantee fewer calls.

Here the main loop is driven from Python: each chunk first fetches a fresh proposal, then
decides whether to call the reviewer. The auto (no-review) branch never touches the model
service — it constructs an explicit response tagged `harness_auto`, reusing the current
`request_id`, the original action prefix and the original validator; the unexecuted tail of
the old proposal is discarded.

An auto response reports execution state and intent as `uncertain`; it never fabricates that
the teacher observed an aligned or successful outcome. Before the first control step the
state can only be `not_started`. All edits / end-effector targets must come from the reviewer
and pass through the real `failed` / `misaligned` gates.

## 3. Data contract

The dataclasses in `types.py` *are* the protocol; they do not depend on the simulator:

| Type | Key fields | Invariant |
|---|---|---|
| Observation | episode_id, step, features, payload, terminal, native_success | step never regresses within an episode; the native result is stored independently |
| Proposal | request_id, observation, payload, max_steps, warnings, blocked | regenerated for every new observation; never replayed |
| Transition | before, after, stage, mode, source, steps | describes the actual ACK, not a planned trajectory |
| Review | stage, last_outcome, response, checkpoint, recovery_status | the review of the last chunk's real result and the next proposal are kept separate |
| Checkpoint | id, observation, stage, goal, prerequisites | a sub-goal confirmed *after* execution, not a state snapshot |

`last_outcome` is `ok / error / unknown`. Approving the next chunk of actions is not enough to
return `ok` for the previous chunk; with no execution history it must be `unknown`. A
checkpoint additionally requires an explicit, re-reachable goal and its prerequisites. The
response for a terminal observation must be null — no further actions are produced or executed.

## 4. The experience store and stage recognition

### 4.1 Two kinds of history

- **Per-episode trajectory** — a JSONL log keeps the actual action source, reviews,
  checkpoints, ACKs and timings. Checkpoints are only used for recovery *within the current
  episode*.
- **Cross-episode experience** — a SQLite store keeps stage samples, confirmed outcomes,
  error counts and recovery evidence, used to schedule reviews in later episodes. Coordinates
  from another episode are never reused directly as a checkpoint in this one.

Nearby situations that share a stage and error type are merged into one count (default merge
radius 0.04); two observations are not required to be pixel-identical.

Database tables:

| Table | Contents |
|---|---|
| metadata | schema version (currently 1) |
| examples | recent observation-feature samples per stage, at most 128 |
| stats | decayed success / failure counts, total confirmed reviews, clean streak |
| errors | error situations, type, count, evidence, recovery count and note |
| events | already-processed review outcomes, de-duplicated by episode/step/kind |

A namespace isolates the task, weights, policy config, model and situation scope; a live run
also folds in the reviewer protocol, an endpoint digest, the effort and (for Anthropic)
max_tokens, so experience from differently-configured reviewers is never mixed. The core lets
you pass a custom namespace; the live command derives a digest from identity parameters.
Changing the layout semantics must change the `scope` — the environment distribution is never
assumed to be unchanged automatically.

### 4.2 Matching by resemblance, not by physical step N

The teacher gives the current stage a stable short name, e.g. `approach-bottle`. On a later
run the current features are matched against already-reviewed stage samples; the physical
step is used only for ordering, audits and budgets.

Default RoboDojo features: three camera views at 8×8 RGB each, normalized to [0,1], plus a
14-dimensional normalized robot state — 590 dimensions in total. The three cameras carry
equal weight; joints map to [-π, π] and are clipped, the gripper stays in [0,1].

Distance is the root-mean-square distance `d(x,y) = sqrt(mean((x-y)^2))`. Per stage, the
nearest sample decides:

- nearest distance above the threshold → unfamiliar situation → review;
- the two closest stages too close together → ambiguous stage → review;
- match to a different stage → review;
- cold start, no samples → review, and let the teacher establish the stage samples.

This is a transparent, low-cost starter implementation, **not** a semantic visual encoder. A
similar background, tiny object details, or an occluded gripper can all cause a wrong match;
fixed-layout validation is not a proof of layout generalization.

### 4.3 Success accounting must be conservative

Only the **last student chunk that an actual review confirms** is scored a success or a
failure. Intermediate skipped (unreviewed) chunks are not retroactively credited; a successful
end-effector / edit recovery does not raise the policy's reliability either. Student actions
during recovery do not add to ordinary execution reliability.

For example, if A/B/C run back-to-back and only C is reviewed, only C's result is updated —
A and B are not batch-credited. The reviewer does receive a summary of the unreviewed chunks,
but that is not proof of completion.

When the next proposal is judged `misaligned`, it is recorded separately as an *intent* error,
not folded into some executed chunk's failure; it raises the review frequency and triggers a
recovery assessment. All review outcomes are de-duplicated by episode/step/kind, so recovery
re-planning is never double-counted.

## 5. The dynamic review function

Each stage keeps exponentially-decayed evidence (default `decay = 0.98`):

```text
new success:  S ← decay × S + 1,  F ← decay × F
new failure:  S ← decay × S,      F ← decay × F + 1
unknown / unreviewed / correction-success: no VLA success sample added
```

With `n = S + F` and `p = S / n`, compute a conservative reliability score `L` in Wilson form
(`z = 1.96`):

```text
L   = [p + z²/(2n) − z·sqrt(p(1−p)/n + z²/(4n²))] / [1 + z²/n]
gap = 1 + floor((max_interval − 1) × L²)
```

Below `min_samples`, or before the post-error clean-success cooldown is satisfied, `gap` is
forced to 1.

Note: because the samples are decayed, correlated and selectively reviewed, `L` is a
**conservative scheduling score** — it must not be read as a statistically proven 95% safety
lower bound on the robot.

Default parameters and priority:

| Parameter | Default | Meaning |
|---|---:|---|
| max_interval | 6 | at most this many chunks between reviews |
| min_samples | 4 | confirmed results required before a stage may lower its frequency |
| cooldown_successes | 3 | consecutive confirmed successes required after an error |
| max_unreviewed_steps | 60 | hard cap on physical steps since the last review; auto chunks are clipped to the remainder |
| audit_probability | 0.05 | random spot-audit even when skipping is allowed (seeded, reproducible) |
| match_threshold / ambiguity_margin | 0.06 / 0.01 | feature-match / stage-ambiguity thresholds |
| error_radius | 0.04 | tighten review near a known error |

At the same 100% success rate, one success and one hundred successes give different intervals.
Reviewed chunks are also clipped by the physical-step cap; the review event records the
original decision, and `execution_requested` records the prefix actually taken. A new error
resets the cooldown immediately; the neighbourhood of a known error requires at least
`3 × cooldown_successes` consecutive confirmed successes before the normal interval resumes,
without ever deleting the error record.

Unfamiliar situations, stage changes, a previous `uncertain`, active recovery, a monitor alert
and the step cap all take priority over `gap`. So the real review frequency is not determined
by the success rate alone; a random re-audit can also make one round slightly heavier than the
last.

## 6. Who monitors when the teacher is not called

Every chunk still has a real new observation and a fresh policy proposal. The default checks:

- action dimension, numeric finiteness, gripper range → an invalid one blocks execution
  outright;
- large joint jumps between adjacent actions, or from the measured state to the first action
  → trigger a review;
- the robot state barely changing across several chunks → trigger a review;
- unfamiliar / ambiguous / near-known-error matches, and the longest-unreviewed cap → trigger
  a review.

These rules **cannot** reliably detect a missed grasp, the wrong object, or a dropped object.
Detection latency is bounded at best by the audit interval; a temporary absence of alerts must
not be recorded as a success. For a new robot, start with an all-review baseline, then
calibrate the thresholds against the measured detection latency.

## 7. The recovery state machine

```text
normal
  └─ observed error / next-chunk intent error → local
       ├─ recovery succeeds → normal
       └─ two attempts fail → return_latest
            ├─ return fails → select
            └─ return succeeds → after_return
                 ├─ recovery succeeds → normal
                 └─ two attempts fail → select
                      ├─ no reachable point → exhausted
                      └─ teacher picks an untried historical point → return_selected
                           ├─ return fails → select
                           └─ return succeeds → after_return
```

### What one attempt is

One attempt may contain several 1–5 step end-effector / edit calls; each has a new
observation, a new proposal and a teacher review. An explicit `succeeded` / `failed` ends the
current attempt; running `continue` up to the default 8 chunks also counts as one failed
attempt. If a fresh policy proposal can self-recover, student actions are still allowed — a
`failed` / `misaligned` is never fabricated just to force an end-effector move.

The first failure does not roll back immediately; the second escalates. A return is itself a
closed-loop goal — success only enters `after_return`, it does not mean the whole fault is
fixed. `after_return` re-points the goal at the original failed task intent, rather than
re-proving that the historical point was reached. Every recovery action after a fault is
reviewed, and a re-plan is requested after the state changes, to avoid executing actions
generated against a stale goal.

The total recovery budget defaults to 64 chunks, accumulated across the whole episode — it is
not reset indefinitely by "recovered, then failed again". The total control-chunk budget
defaults to 1000. No exhausted budget may be disguised as a native task failure.

### Checkpoint eligibility

A checkpoint must be in the current episode, before the error, and confirmed after execution.
The most-recent point is chosen by actual step; a teacher choice must belong to the explicit
candidate set. Return targets already tried during the current fault are excluded, to avoid an
infinite loop between two historical points; a new fault after a successful recovery re-allows
those points but keeps the total recovery budget. By default only the most recent 64
checkpoints of the current episode are retained (configurable). A historical point only
describes the conditions at that time — whether it is still reachable now must be re-observed;
there is no object-state rewind.

For example, if the object has already fallen out of the workspace, then even if the arm pose
from the old checkpoint is still reachable, the grasp state cannot be assumed recovered. The
correct outcome is to report the goal unreachable, not to replay the record backwards.

## 8. Requests, records and cost

The reviewer (Responses or Anthropic Messages) sends the current observation, the previous
chunk's before/after, the current policy forward-kinematics, a bounded summary of unreviewed
chunks, the stage and error memory, and the recovery context. It attaches only the current
three images, the three images from before the last chunk's execution, and the image of the
active recovery goal — it does not re-upload the whole image history every time.

A request is an explicit, single-shot context call with no hidden thread. It uses a strict JSON
schema and validates the `request_id`, fields, numbers, action ranges, gate and terminal
contract. The endpoint and model are given explicitly; the key is read only from an environment
variable; redirects are forbidden, the environment proxy is off by default, and there is no
automatic retry. A failed response never writes credentials or the raw HTTP error body.

When the object field set does not match, a `ReviewSchemaError` is raised; the live failure
event, `probe.json` and CLI stderr record only `schema_path`, `missing_keys` and `extra_count`.
Both the path and the missing fields come from the local schema — no field values, unknown key
names or raw response are recorded. A `ReviewConstraintError` likewise carries only a fixed
local rule code and schema path, never the actual value, unknown field name or raw reply. That
diagnostic covers type/enum/range, quaternion, gate, execution-result consistency, checkpoint
and protocol envelope; all original validation predicates are unchanged.

The reviewed output is an outer `Review`; the action's request_id/mode/steps/reason/edit/
target/assessment must be nested inside `response`. The error envelope rejects a malformed
output rather than silently promoting top-level fields to patch it.

Responses uses `json_schema` output; Anthropic Messages carries the same `Review` through a
single `submit_review` tool, then runs the same strict local validation. It does not rely on a
beta strict-tool mode and does not treat plain text as fallback JSON; truncated, wrong-tool and
malformed responses are rejected. The core reviewer protocol is fixed — any other agent must
implement this robot-review contract and cannot act directly, bypassing the main loop.

The `probe` command sends a single review over a synthetic state and an optional gray PNG; it
creates no robot backend and executes no action. `effort` is passed through to the matching
protocol field verbatim, with no guarantee the model/gateway supports it.

For a Responses service with no separate cache-write billing, if the price is per million
tokens, the cost is computable offline:

```text
cost = [(input − cached) × input_price
        + cached × cached_input_price
        + output × output_price] / 1_000_000
```

Cached input is a subset of input, and reasoning tokens are usually a subset of output — they
must not be added twice. Anthropic native input excludes cache read/write; this implementation
sums the three raw counts into a total input only when all three are present, and lists
`cache_creation_input_tokens` separately. If any required usage field is missing, the cost is
unknown and is not filled with zero.

## 9. Verification and real experiments

The offline tests cover experience persistence and isolation, dynamic frequency, unknown /
ambiguous / error cooldown, recovery escalation and budgets, termination, freshness,
single-shot action calls, usage, strict output, and the image / authentication boundaries. The
toy fault scripts are explicit, deterministic scenarios — they do not stand in for real
physical recovery.

A real experiment should keep two controls: an all-review baseline of the *same* reviewer and
backend, as the scheduling comparison. Fix the task, layout, policy, model, budget and logging;
use a separate experience store per condition; repeat over several random seeds. Report native
success rate, failure / incomplete, teacher calls, real tokens, error-detection latency,
recovery actions and total wall-clock.

If the native task has already ended but the terminal re-check fails, keep `complete` /
`native_success` and record `status=audit_failed` separately — a success must not be erased
into unknown.

### Note on gripper evidence

`gripper_opening_command` is a continuous command (0 closed, 1 open); it is not a measured
gripper width, contact force or grasp-success label. An upstream client may derive a
descriptive `gripper_closed` from a `< 0.5` threshold, but the student execution keeps the
continuous value; a non-zero 0.33 must not be read directly as "did not close / missed the
grasp". A review should describe the command and the RGB object-motion evidence separately, and
return `unknown` when uncertain.

## 10. RoboDojo integration contract

The core package is simulator-agnostic: `types.py` is the whole protocol. `robodojo.py`
adapts one concrete simulator — [RoboDojo](https://github.com/robodojo-benchmark/RoboDojo) —
by translating its observation payload into `Observation`/`Proposal` and back. It only reads
the whitelisted `PUBLIC_FIELDS` of the payload, and models a 14-DoF bimanual robot with the two
gripper indices (6 and 13) dropped from the matched joint vector, and three RGB cameras
(`cam_high`, `cam_left_wrist`, `cam_right_wrist`).

`fast-harness live` binds two caller-provided runtime classes, imported from the module named
by `--runtime-module` (and optionally `--upstream` to put its parent on `sys.path`):

| Class | Constructor | Used for |
|---|---|---|
| `PolicyClient` | `PolicyClient(port, checkpoint)` | the action-chunk policy server; exposes `.metadata` (`checkpoint_sha256`, `config`) and `.close()` |
| `RoboDojoTools` | `RoboDojoTools(rollout_dir, task, policy, sim_port=, seed=, max_decisions=)` | the simulator client; exposes a `.sim.request(name)` RPC channel used for read-only keepalive |

RoboDojo itself is released as an eval-only client under its own non-commercial research
license; the policy server is a separate component you supply. This package neither vendors
nor requires either one — it defines the contract above and leaves the runtime to the caller,
so any RoboDojo-compatible client and policy server can be plugged in. The fully offline
`demo` command needs none of this.
