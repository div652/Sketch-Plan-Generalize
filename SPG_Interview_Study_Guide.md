# Sketch–Plan–Generalize (SPG) — Interview Study Guide

**Paper:** *Sketch-Plan-Generalize: Learning and Planning with Neuro-Symbolic Programmatic Representations for Inductive Spatial Concepts*
**Authors:** Kalithasan, Sachdeva, Singh, Bindal, Tuli, Panjeta, Vora, Aggarwal, Paul, Singla (IIT Delhi)
**Venue:** Programmatic Representations for Agent Learning Workshop, ICML 2025 (PMLR 267). arXiv 2404.07774 (v1 Apr 2024 → v3 Jun 2025).

> Use this alongside the highlighted PDF. Colors there: **yellow** = method/key ideas, **green** = results/evidence, **blue** = problem framing & formal setup, **pink** = limitations (of prior work and of this paper) & future work.

---

## 0. The 30-second pitch (memorize this)

A robot should learn a *new spatial concept* (e.g., "staircase") from just a few human demonstrations, and then build it at **any size** and reuse it to learn **even more complex** concepts. SPG does this by learning the concept as a small **Python program** (a neuro-symbolic representation). The trick: instead of searching directly in the huge space of programs (intractable, and you can't grade a half-written program against a demo), SPG factors the problem into **Sketch → Plan → Generalize**: an LLM guesses the function *signature*, an **MCTS search over grounded robot actions** finds the concrete build sequence (gradeable against the demo via a dense IoU reward), and an LLM **abstracts** that sequence into a general program that gets added to a reusable library. Result: far better inductive (out-of-distribution) generalization than LLM-only or purely-neural baselines, plus continual, compositional learning.

---

## 1. Motivation — why this paper, and why then

For robots to collaborate with people, they must pick up *personalized* concepts from a handful of demonstrations. The paper argues a good concept representation needs **four properties** (highlighted blue on p.1):

1. **(a) Grounded in demonstrations** — reflect the human's actual intent, not just the model's prior world knowledge.
2. **(b) Inductive generalization** — from a few towers of fixed heights, infer how to build a tower of *arbitrary* height.
3. **(c) Hierarchical composition** — express complex concepts as compositions of simpler ones (a staircase = a sequence of towers of increasing height; a pyramid = rows of decreasing size).
4. **(d) Modifiability** — easily fold in new constraints ("no green block next to a blue or yellow one in the staircase").

### 1.1 Where the four properties come from

They are the authors' **own desiderata**, but the *reason* those particular properties matter is borrowed from cognitive science. The framing sentence — forming inductive representations of novel grounded concepts and applying them beyond training is "a hallmark of human intelligence" — is anchored to two citations:

- **Tenenbaum et al., 2011** ("How to grow a mind," *Science*): humans learn rich concepts from very sparse data using structured, compositional, abstract representations.
- **Chollet, 2019** ("On the measure of intelligence," the ARC paper): intelligence = efficient generalization to novel tasks from few examples, not memorized skill.

So the list isn't lifted from an external taxonomy; it's the authors' distillation of "what human-like concept learning looks like," specialized to robot learning. It plays a double role: (i) a yardstick to show each prior family fails ≥1 property, and (ii) a preview of SPG's selling points, since each property maps to a component — **(a) grounding ↔ the demo-guided Plan stage, (b) induction ↔ programs with loops, (c) hierarchy ↔ the concept library, (d) modifiability ↔ the constraint bifurcation in §6.** They double as the paper's evaluation criteria.

### 1.2 Property (a) in depth — *grounded in demonstrations, not priors*

> *"it should be grounded directly in human demonstrations, reflecting the human's intent rather than relying solely on prior world knowledge."*

The whole property turns on **one question: where does the robot's understanding of the concept actually come from?** Two competing sources:

- **Prior world knowledge** — what a pretrained model already "knows." An LLM has seen the word "tower" countless times and carries a built-in notion of it.
- **The demonstration** — what *this specific human just physically did*: the concrete pick-and-place actions and the resulting 3D configuration.

Property (a) insists the representation be pinned to the **second** source. Word by word:

- *"grounded directly in human demonstrations"* — the symbol "tower" must be tied to the robot's own perception and action (which blocks, which 3D positions, which motions), derived from the demo, not from a textual/internet idea of the word.
- *"reflecting the human's intent"* — the human may mean something idiosyncratic (their staircase ascends left-to-right; their usage requires alternating colors). Capture *their* version, not the average-of-the-internet version.
- *"rather than relying **solely** on prior world knowledge"* — note **"solely."** It does *not* forbid priors. SPG uses GPT-4's priors for the cheap parts (guessing a signature, writing clean code). What's forbidden is the *content* of the concept being **determined only by** priors; the demo must be the dominant source of what's actually learned.

**Why it matters concretely:**
- **Faithfulness / personalization** — a prior-reliant model produces the generic tower even when the human demonstrated something different; it never really "reads" the demo.
- **Linguistically novel concepts** — some concepts have *no* useful prior. The sharpest test is **Dataset II**, where "tower" is renamed **"rewot."** A prior-reliant model has nothing to fall back on; a demo-grounded one can still learn it. Empirically GPT-4/GPT-4V collapse to ~0 on complex structures here while SPG holds — making Dataset II essentially a **direct test of property (a)**, exposing that the baselines were leaning on the *name*.

**Common confusion:** property (a) is about the **source** of the concept (demo vs. prior), *not* about size — size is property (b). "Learn the tower faithfully from this demo" = (a); "now build it at height 10" = (b).

**One-liner — how SPG achieves (a):** In the Plan stage, SPG searches for an action sequence whose built result matches the human's demonstrated keyframes via a dense IoU reward, so the concept is fixed by *what the human actually built* rather than the LLM's prior notion of the word (and the visual grounder ties symbols like `filter(green, dice)` to the actual objects in the scene).

### 1.3 Context at the time (2022–2024) — what the LLM/VLM "robot programmers" were doing

This is the paradigm SPG reacts to. **Shared idea:** use a pretrained **LLM as the high-level planner/brain** that turns a language task into a sequence of robot actions. Two flavors:

- **LLM-as-step-planner** — SayCan (Ahn 2022), Inner Monologue (Huang 2022b), Zero-Shot Planners (Huang 2022a). The LLM emits steps ("pick up the sponge, go to the sink…") mapped onto a fixed menu of skills. *SayCan's twist:* multiply the LLM's "**say**" score (is this skill relevant?) by a learned value function's "**can**" score (can the robot do it from this state?), grounding suggestions in affordances. *Inner Monologue* adds closed-loop text feedback to re-plan after failures.
- **LLM-as-code-writer** — ProgPrompt (Singh 2022) and Code as Policies (Liang 2023), the direct precursors of SPG's "plans-as-programs." The LLM literally **writes Python** calling perception APIs (`get_obj_pos`, `detect`) and control primitives (`move_to`, `pick`, `place`), with loops/conditionals/arithmetic; few-shot prompted, and executing the code drives the arm.

**How were they "learning to take actions and guide the arm"?** The key nuance: for *planning*, they mostly **weren't learning** (no gradient updates). The **LLM is frozen and prompted** (zero/few-shot, in-context); its contribution is task knowledge and common sense from pretraining. The arm is actually driven by three *other* pieces:

1. A fixed library of **pretrained low-level skills** (`pick`, `place`, `move_to`), each trained separately (imitation/RL) or implemented as classical motion planners/scripts. The LLM never moves joints — it *sequences* these skills.
2. **Perception APIs** (detectors, pose estimators) that feed object positions into the LLM's code — the grounding hooks.
3. **Grounding mechanisms** like SayCan's value functions to keep choices physically feasible.

Pipeline: *language task → (frozen, prompted LLM) → sequence of skill calls / Python → executed by pretrained skills + perception → arm moves.*

**Where SPG's critique bites:** these systems **translate known tasks into action sequences using priors**; they don't *learn a new concept's representation from a demonstration*. Show them a demo of a never-named structure and there's no mechanism to faithfully absorb it — they pattern-match to priors, don't induce a reusable, size-general program from watching, and (reasoning over language, not physics) can emit physically ungrounded plans. That's the failure of property (a) and of learning "linguistically novel" concepts.

**Balance / nuance to raise:** SPG doesn't discard this paradigm — it *uses* it. The **Sketch** and **Generalize** stages are LLM code-generation, and the inference-time planning in §6 is explicitly Code-as-Policies-style prompting. SPG's move is to insert a **demonstration-guided grounded search (Plan)** in the middle, so the concept gets pinned to the demo *before* the LLM abstracts it into reusable code.

**The other two families (for completeness):** neuro-symbolic *program induction* (DreamCoder, LILO/Grand et al.) and purely-neural rearrangement (StructDiffusion) were strong but in different ways — see §2. No single approach hit all four properties at once; SPG's pitch is to combine the **code-generation strength of LLMs** with **grounded neural search guided by demonstrations**.

---

## 2. What was broken in prior approaches (the gap SPG fills)

Frame this as **three families**, each failing a different property. This is a very common interview question.

| Family | Examples | What it got right | Why it failed |
|---|---|---|---|
| **LLM/VLM planners** | Code as Policies, ProgPrompt, SayCan | World knowledge, fluent code gen | Struggle to learn *new*, *linguistically novel* concepts from demos; **over-rely on prior knowledge** → poor faithfulness & poor inductive generalization; can output **physically ungrounded** plans |
| **Purely neural** | StructDiffusion | Learn directly from demos, noise-resilient | **Don't model symbolic induction** and **lack modularity** → can't generalize past training sizes, can't reuse learned concepts |
| **Neuro-symbolic / program search** | DreamCoder, LILO | Best generalization of the three; programs are inductive & reusable | **Enumerative search is intractable** in rich program spaces (e.g., Python); crucially, **search can't be guided by demonstrations** because a *partial* program can't be executed/graded |

**The key diagnosis (say this in the interview):** the third family is the right *representation* (programs generalize and compose), but searching *in program space* is both intractable and **demonstration-blind**. SPG's insight is to move the expensive search out of program space and into the space of **grounded actions**, where a partially-built structure *can* be graded against the demonstration with a dense reward — then let an LLM do the final jump from a concrete action trace to a general program.

---

## 3. Core idea & novelty (the one thing to remember)

**Factor inductive concept learning into three stages so the search happens where it can be guided:**

- **Sketch** *(LLM, cheap)* — from the instruction Λ, produce the program **signature**: concept name + arguments, e.g. `Staircase(steps=4, filter(cyan, legos))`. Ground it on the scene with a visual module.
- **Plan** *(MCTS, the workhorse)* — search over **grounded action sequences** (move a "head", place objects, or call macro-actions for already-known concepts) to maximize how well the built structure matches the demonstration (dense **IoU reward**). This is gradeable at every step, unlike a partial program.
- **Generalize** *(LLM, cheap)* — hand the concrete grounded plan + signature to GPT-4, which **abstracts** it into a general Python program with loops/recursion. Add it to the **library L** for reuse.

**Why this is novel / clever:**
- It **decouples** "what's the abstraction" (LLM, sketch + generalize) from "what's the physically-grounded realization" (neural+symbolic search). The hard combinatorial part is guided by demonstrations via dense reward, which prior program-search methods couldn't do.
- The library makes learning **compositional and continual**: once you know `tower`, the search can use a `Make_Tower(size)` **macro-action**, so a staircase is discovered as "tower, move right, bigger tower, …" rather than dozens of primitive placements. Shorter, modular plans are also the ones LLMs can actually generalize (see §6 gotcha).
- It explicitly enforces **physical plausibility** during search (the structures must be buildable), unlike 2D drawing-program work where you can "draw a tower top-down" — physically impossible to build that way.

---

## 4. The formal framework (be ready to explain at a whiteboard)

**Setup (blue on p.3):** a goal-conditioned MDP ⟨S, A, T, g, R, γ⟩. The agent already has *primitive* visual concepts (green, dice) and *action* concepts (left, right, move-and-place) — these populate the initial library **L** as executable function calls. A plan = a program of function calls. Demonstrations D = (language Λ, keyframes S₁…S_g).

**Inductive spatial concept (the definition to quote):** a structure is *inductive* if its construction can be described **recursively** using a smaller instance of itself **or** as a **composition** of simpler structures. A partial order on L (`C depends on C̃` if `C̃` is a substructure of `C`) gives **structural complexity**: staircase depends on tower; X depends on diagonals.

**Recursive construction (Eq. 1)** — `h(C_k, n, p)` = three terms composed:
- **Induction (I):** build `C_k` of size *n* from `C_k` of size *n−1* (toggled by λ∈{0,1}).
- **Composition (C):** build `C_k` from previously-known concepts `C_{k'}` with `k'<k`.
- **Base (B):** primitive actions (e.g., a `move-top` then place) — the recursion's base case.

> Intuition: a tower of *n* = tower of *(n−1)* + place one block on top (pure induction). A staircase = composition of towers of increasing height (composition). A single block placement = base case.

**Learning objective (Eq. 2)** — Bayesian MAP over programs:
`P(H | Λ, S₁…S_g) ∝ P(S₁…S_g | Λ, H) · P(H | Λ)`.
- **Likelihood:** does executing program H reproduce the demonstrated frames?
- **Prior:** regularizer **P(H) ∝ |H|^(−α)** → *prefer short programs* (this is what pushes toward modular, looping, reusable code rather than long flat action lists).
So `H* = argmin [ Loss(frames, Exec(H,Λ)) + α·log|H| ]`. Exact inference is intractable → approximate by **search**, but the search is done over *grounded plans*, not raw programs (Eq. 3: `Sketch → Plan → Generalize`). After learning, **L ← L ∪ H***.

---

## 5. The method in depth

### 5.1 Sketch
GPT-4 with in-context examples turns Λ into a **tree of nested function calls** (the signature), e.g. `Staircase(steps=4, filter(cyan, legos))`. A **visual grounding module** then instantiates it on the scene. The grounder has three parts: a **ResNet-34** visual feature extractor; a **concept-embedding module** that learns *disentangled* embeddings for visual attributes (green, dice); and an **executor** with primitives like `filter`. Grounding `filter(green, dice)` returns the ordered list of green dice → `Tower(height=3, ObjectSet)`.

### 5.2 Plan (the heart)
**The "head" abstraction:** placement is reasoned about with a **head** — a 3D cuboidal bounding box (x1,y1,x2,y2,depth) representing "where I'm considering placing next." Moving the head ≈ the robot mentally scanning placement locations.

**Primitive actions A_p:**
- `move_head(direction)` — a **neural operator** (trained on pick-and-place data) that nudges the head (left/right/front/back/top).
- `keep_at_head(objects)` — place the chosen object at the head.
- `store_head()` / `reset_head()` — push/pop head positions on a stack (to return to useful spots).

**Macro/compound actions A_c:** every learned concept `<cpt>` gets a `Make_<cpt>(size)` macro-action that executes its program. So `A = A_p ∪ A_c`.

**Reward:** **IoU** between the achieved state and the expected demo state, given for macro-actions and `keep_at_head`; all else gets 0. Dense, intermediate rewards (the demo's intermediate frames are available) make the search well-guided.

**Two design pillars (memorize the L / P split):**
- **Modularity = "+L":** allow the search to call learned-concept macro-actions. Discounted IoU rewards make a macro *preferred* over the equivalent primitive sequence — e.g., `Make_Tower(3)` scores `1+1+1` whereas the 3-step primitive sequence scores `1 + γ²·1 + γ⁴·1` (γ=0.95). This realizes the short-program prior and yields **concise, modular, generalizable** plans.
- **Scalability = "+P":** as L grows, the action space balloons. Train a **reactive policy π_neural** that, given current and next-expected state, predicts the single best *primitive* action `a*_p`. During search, only expand `A_c ∪ {a*_p}` instead of all of A_p → branching factor drops from `N·|A_c| + |A_p|` to **`|A_c| + 1`**.

**MCTS tweaks (Appendix A.3):** *no Monte-Carlo rollouts* (the reward isn't sparse, so per-leaf simulation is unnecessary); **off-policy Q-learning backups** `Q(s,a)=r+γV(s')`, `V(s)=max_a Q`.

The full method = **MCTS+L+P** (a.k.a. MCTS+P+L).

### 5.3 Generalize
GPT-4 distills the grounded plan into a general Python program. Two robustness moves:
- **Top-k plans, not one:** keep the k highest-IoU plans, generalize each, re-execute, and keep the program with the best IoU (k=5 in experiments; performance flat from k=5→20).
- **Multiple demonstrations:** independently sketch+plan each of k demos, then ask GPT-4 to infer *one* abstraction over all of them — and explicitly tell it *"some plans may be partially wrong"* → robustness to noisy/partly-wrong traces. (3 demos per concept in the main results.)

### 5.4 Worked example (the staircase — highlighted yellow in Appendix A.5)
- **Parse:** `Staircase(height=3, objects=filter(red, blocks))`
- **Ground:** `filter(red, blocks) = [1,2,3,4,5,6]`
- **Plan (from keyframes):** `Tower(1,[1..6]), move_head(right), Tower(2,[2..6]), move_head(right), Tower(3,[4,5,6])`
- **Generalize:**
  ```python
  def staircase(height, objects):
      for i in range(height):
          tower(height=i, objects)
          move_head(right)
  ```
This single example crystallizes the whole pipeline — point to it in the interview.

---

## 6. A subtle but important point (great to raise unprompted)

**Not all correct plans are equally generalizable by an LLM.** Appendix C.3–C.4 shows two *same-length, both-correct* plans for a pyramid/tower: GPT-4 cleanly turns the **modular** one (built from `row(...)` calls) into a looped program, but **fails** to generalize the one written purely in primitives. This is the deeper justification for the "+L" macro-actions and the short-program prior: the system is biased toward plans that are not just short, but **abstractable**. (This connects modularity to LLM generalization in a concrete way — interviewers like this.)

---

## 7. Adapting to new instructions & constraints (Section 6 + D.5)

- **Goal grounding:** a parser splits an instruction into *concepts* + *constraints*. Missing concept → request a human demo → run SPG → append to L.
- **Constraint resolution is modularized away from structure learning.** An LLM decides if explicit constraint solving is needed; if so, constraints are isolated and handed to the right solver:
  - **Z3 (SMT/CSP) solver** for hard discrete constraints, e.g. λ₁ = "staircase of size 5 where each block matches the color of the block to its left, but never the block on top" (Fig. 6).
  - **GPT-4 directly** for simpler constraints expressible with library calls, e.g. λ₂ "tower of green dice the same height as the existing white tower" (uses a `find_size` helper) or λ₃ "tower of 6 alternating blue/red" (Fig. 7).
- **Rationale:** LLMs alone are bad at spatial grounding, visibility, occlusion, and physical stability, so *bifurcating* structure-learning from constraint-solving beats asking one LLM to do both.
- **Goal-conditioned planning (D.5):** with learned concepts, do a **forward search** over abstract scene-graph states. Fig. 8 shows it (a) fixing an adversarial start by unstacking/restacking a faulty tower, and (b) completing a staircase from a pre-built row **by a construction method it never saw during learning** — evidence the representation supports flexible planning, not just replay.
  - Bonus detail: they argue **why hand-coding PDDL is hard here** — classic blocks-world PDDL assumes only `onTop`, but these structures need `onRight`, `onFront`, etc., and the same action yields different post-conditions depending on state. They instead learn preconditions (`is-clear`, `will-not-be-floating` via an on-table classifier) and actions (`place-random` via a VAE, `move`).

---

## 8. Experiments — what was tested and the headline numbers

**Corpus:** simulated robot arm on a tabletop with an RGB-D sensor. **15 structure types**, **3 demos each**, up to 20 objects/scene. Simple = Row, Column, Tower, their inversions, four Diagonals. Complex = X, Staircase, Inverted-Staircase, Pyramid, Arch-Bridge, Boundary. Structures inspired by Silver et al. 2019 (staircase/enclosure), Collins et al. 2024 / RAMP (boundaries), Lake et al. 2015 (arch-bridge, X).

**Three datasets:**
- **I / II:** sizes ∈ [3,5]. **II reverses concept names** (tower → "rewot") to kill the LLM's prior knowledge and force reliance on demos.
- **III:** *larger* sizes than training → the OOD / inductive-generalization test.

**Baselines:** purely-neural **SD** and **SD+G** (StructDiffusion, with/without a perfect grounder); program-emitting **GPT-4** (text scene-graph input, *assumes no distractors*) and **GPT-4V** (image input). (CodeLlama-70B tried, much worse, dropped.)

**Metrics:** **Program Accuracy** (binary, human-judged: did it build the structure fully?), **IoU** (2D bbox overlap), **MSE** (bbox + depth).

### The four research questions and the numbers that matter

**Q1 — In-distribution accuracy (Tables 1–2):**
- Program accuracy — **SPG 1.00 / 0.83** (simple/complex); GPT-4 0.78 / **0.00**; GPT-4V 0.33 / **0.00**.
- Pre-trained models score **zero on complex** structures. LLM beats VLM on simple but is worse on complex (text descriptions of complex spatial relations are weak). Purely-neural beats foundation models on complex but still trails SPG.

**Q2 — OOD / inductive generalization (Table 3, the most important result):** relative decrease (R.D.) in IoU going from in-dist → larger sizes:
- **SPG: only 7.27% (simple) / 5.74% (complex)** — barely degrades.
- SD+G: **63.25% / 74.72%** — purely neural collapses OOD.
- GPT-4: 12.61% / **53.87%**; GPT-4V: 23.33% / 41.64% — LLMs can't emit inductively-generalizing programs for complex structures.

**Q3 — Robustness & efficiency:**
- *Reliance on priors (Table 4, reversed names):* SPG 0.88 / 0.78; GPT-4 0.67 / **0.00**; GPT-4V 0.23 / **0.00**. SPG's drop on simple structures (12%) is smaller than GPT-4 (14%) and far smaller than GPT-4V (30%) → SPG learns from *demos*, not the name.
- *Ablations (Table 5):* replace MCTS with an LLM planner (**SPG-M+LMP**) → 0.55 / 0.16; even GPT-4V with 5-sample reranking + ground-truth sub-program teacher-forcing (**GPT-4V+VRF**) → 0.66 / 0.16. Both far below SPG → **the MCTS search is doing essential work**.
- *MCTS variants (Fig. 5):* **MCTS+L+P reaches 0.933 program accuracy in ~4,000 expansions**, vs **~512,000** for the no-pruning variant → the neural pruner is a ~100× efficiency win. Library-using ("+L") variants exceed 0.6; the no-library greedy variant plateaus low → the library enables a richer concept class.
- *Compute:* 36h pretraining; best concept-learning run ~12 min; inference < 2 min. (3× NVIDIA A40.)

**Q4 — Constraints & planning:** qualitative successes on λ₁/λ₂/λ₃ and on adversarial/assistive planning (Figs. 6–8), as in §7.

**One-line takeaway:** *SPG matches or beats every baseline in-distribution and dramatically wins out-of-distribution and under name-reversal, and ablations show both the symbolic search and the concept library are necessary.*

---

## 9. Limitations

**Stated by the authors (pink, conclusion):**
1. Relies on **perfect demonstrations** (no real demo noise).
2. Assumes **full observability** of all objects (no occlusion/partial views).
3. **Simulation only** — no real-robot results.
4. (From D.4) The reactive policy can't yet emit `reset_head`/`store_head`, and handling a growing `A_c` is left to future work.

**Additional critical points (your own analysis — strong to volunteer):**
- **LLM dependence & error compounding:** Sketch and Generalize both lean on GPT-4. A wrong signature or a botched abstraction propagates; the top-k re-ranking mitigates but doesn't eliminate this.
- **Hand-designed primitive vocabulary:** the head, the move/keep/store/reset primitives, and the IoU reward are engineered; the "discovery" is of compositions over a fixed primitive set, not of the primitives themselves.
- **Pruning asymmetry:** π_neural prunes only primitive actions A_p; the macro-action space A_c still grows with the library, so MCTS branching can still degrade as more concepts accumulate.
- **Evaluation scale & subjectivity:** 15 structures, 3 demos each, binary *human-judged* program accuracy; 2D-IoU ignores some 3D error. Small, somewhat subjective.
- **Idealized physics:** pick-and-place is clean; no grasp failure, collisions, or controller noise.
- **Constraint pipeline is somewhat bolted-on:** routing to Z3/LLM/neural solvers is effective but not learned end-to-end.

---

## 10. Future directions
- Noisy / imperfect demonstrations (the multi-demo "some plans may be wrong" prompt is a first step).
- **Reasoning under partial observability / beliefs** (drop the full-observability assumption).
- **Interleaving planning and execution** (closed-loop, replan on failure).
- Real-robot deployment.
- A reactive policy / network architecture that scales with a growing macro-action library and supports stack primitives.
- (Natural extensions) richer/learned primitives, learned constraint routing, perception robustness.

---

## 11. Where it sits in the literature (for "how does this relate to X?")
- **vs. LLM robot-coders** (Code as Policies, ProgPrompt, SayCan): SPG *learns new grounded concepts from demos* rather than only translating known tasks to code; demonstrably less reliant on the concept's name/prior.
- **vs. program induction** (DreamCoder, LILO/Grand et al.): same "concepts-as-programs, build a library" spirit, but SPG searches **grounded actions guided by dense demo reward** instead of enumerating programs — tractable in a Python-scale space and demonstration-driven.
- **vs. purely-neural rearrangement** (StructDiffusion): SPG adds explicit symbolic induction + modular reuse, which is exactly what gives the OOD generalization the neural baseline lacks.
- **Builds on** the authors' own grounding/concept-learning line (Kalithasan et al. 2023/2024; Mao et al. neuro-symbolic concept learner) for the visual grounder and primitive action semantics.

---

## 12. Likely interview questions + crisp answers

- **Why search grounded actions instead of programs directly?** Because a partial *program* can't be executed and graded against a demo, so demonstrations can't guide program-space search; a partial *grounded plan* can be graded with intermediate IoU rewards. You also avoid Python's intractable program space.
- **What does the LLM do vs. the search?** LLM = abstraction (signature + plan→program). Search = grounded, physically-plausible realization. Ablations (Table 5) show swapping the search for an LLM planner tanks performance, so the search is load-bearing.
- **What makes generalization "inductive"?** The learned program contains loops/recursion over a `size` argument, so the *same* program builds size-3 or size-10; OOD R.D. of ~6–7% vs 60–75% for the neural baseline is the evidence.
- **Why a concept library / macro-actions?** Compositionality + continual learning: complex concepts are found as compositions of learned ones, plans stay short (short-program prior), and short *modular* plans are the ones LLMs can actually abstract (C.3).
- **What's the role of the IoU reward and the "head"?** Dense IoU vs. the demo frames guides MCTS; the head is the 3D placement cursor that turns "where to put the next block" into a small set of `move_head`/`keep_at_head` decisions.
- **How does it handle constraints?** Separate the structure (learned program) from constraints (Z3 CSP / LLM / neural solver), because LLMs are weak at spatial/physical reasoning — modularizing beats one monolithic LLM call.
- **Biggest weakness?** Perfect demos + full observability + sim-only, plus reliance on GPT-4 for sketch/generalize. Then mention error-compounding and the A_c-growth pruning gap.
- **What is MCTS+L+P vs the ablations?** L = library macro-actions, P = neural pruning of primitives. Full = both. "−P" (no pruning) needs ~512k vs 4k expansions; "−L" (no library) plateaus low because it can't compose.

---

## 13. Glossary / quick reference
- **SPG** — Sketch, Plan, Generalize.
- **Sketch (H\*_S)** — the LLM-produced grounded function signature.
- **Plan (H\*_P)** — the MCTS-found grounded action sequence.
- **Generalize (H\*_G = H\*)** — the abstracted Python program, added to L.
- **L** — concept library (executable function calls; updated continually).
- **head** — 3D bbox "placement cursor"; moved by a neural `move_head`.
- **A_p / A_c** — primitive / compound(macro) actions; `A = A_p ∪ A_c`.
- **π_neural** — reactive policy that prunes primitives during search.
- **IoU reward** — overlap between built and demonstrated states.
- **Datasets I/II/III** — in-dist / name-reversed / larger-size(OOD).
- **R.D.** — relative decrease in IoU from in-dist to OOD (lower = better generalization).
- **Key numbers:** OOD R.D. **7.27/5.74%** (SPG) vs **63.25/74.72%** (SD+G); efficiency **0.933 @ 4k** vs **512k** expansions; in-dist program accuracy **1.00/0.83** vs **0.00** complex for GPT-4/4V.

---

## 14. Verbal summaries to rehearse

**~2 minutes:** State the goal (learn reusable, size-generalizing spatial concepts from a few demos), the gap (LLMs over-rely on priors and aren't grounded; neural nets don't generalize/compose; program search is intractable and demo-blind), the idea (Sketch→Plan→Generalize moves the guided search into grounded-action space where dense IoU rewards work, then abstracts to a library program), and the punchline (near-flat OOD degradation and strong name-reversal robustness, with ablations proving both the search and the library matter).

**~5 minutes:** Add the formal framing (inductive concept = recursion/composition; Bayesian MAP with a short-program prior), the Plan internals (head, primitives, macro-actions, the discounted-reward bias toward macros, neural pruning), the worked staircase example, the constraint bifurcation with Z3, and the concrete tables (1.00/0.83 in-dist; 7.27/5.74 vs 63–75 OOD; 4k vs 512k expansions; SPG-M+LMP and GPT-4V+VRF ablations). Close with limitations (perfect demos, full observability, sim-only) and future work (noise, beliefs, closed-loop, real robot).
