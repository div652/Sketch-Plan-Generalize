# Sketch–Plan–Generalize (SPG) — Interview Study Guide

**Paper:** *Sketch-Plan-Generalize: Learning and Planning with Neuro-Symbolic Programmatic Representations for Inductive Spatial Concepts*
**Authors:** Kalithasan, Sachdeva, Singh, Bindal, Tuli, Panjeta, Vora, Aggarwal, Paul, Singla (IIT Delhi)
**Venue:** Programmatic Representations for Agent Learning Workshop, ICML 2025 (PMLR 267). arXiv 2404.07774 (v1 Apr 2024 → v3 Jun 2025).

> Use this alongside the highlighted PDF. Colors there: **yellow** = method/key ideas, **green** = results/evidence, **blue** = problem framing & formal setup, **pink** = limitations (of prior work and of this paper) & future work.

> **v2 additions:** §5.0 how the primitives are pretrained (NSRM/NSCL); expanded §5.2 dense-reward mechanics, demonstration anatomy, 3D coordinates, and `store_head`/`reset_head`; §9 the append-only limitation + the masking clarification; NSRM promoted to a direct-predecessor bullet in §11; new Q&A and glossary entries.

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

**Why "demonstration-blind"?** Guidance needs a score you can compute on a *partial* solution, so at each fork you steer toward the option closer to the demo. A **half-written program isn't runnable** — it produces nothing to compare until it's complete, so demonstrations have nothing to push against during the search. (Worse, even a *complete* program scores discontinuously: a one-token change can flip it from perfect to a crash, so the scoring landscape is jagged with no gradient.) You can only run finished programs and check pass/fail — which filters end-products but can't prune *during* construction. A **grounded action sequence is executable incrementally**: after every action there's a real partial state on the table, so you get a dense, smooth **IoU-vs-keyframe** reward at each step (place block 1 right → reward up; block 2 → up again). That per-step signal is exactly what lets the demonstration guide the search. So SPG searches actions (gradeable, demo-guided), then lets the LLM *write* the program — nobody has to *search* program space.

> **One-liner — why program space is intractable but action space is tractable:** Program space is unbounded and compositional (arbitrary loops, recursion, nesting → astronomically many candidates) *and* gives no partial-credit signal, so you can't prune as you go; the grounded-action space is small and finite per step (a handful of `move_head`/`keep_at_head`/macro choices) *and* yields a dense IoU reward after every action, so demonstrations prune it down to a guided, tractable walk.

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

### 5.0 The pretrained substrate — where the primitives come from (NSRM / NSCL)
SPG **assumes** it already has grounded visual concepts (red, cube, dice), relations (left, right, front), and single-step action semantics — Sec. 3 says these "can be acquired… via (Kalithasan et al., 2023 [**NSRM**, ICRA'23]; Mao et al., 2019 [**NSCL**])." That precursor work is the machinery underneath SPG's alphabet; SPG then learns the *grammar* (inductive programs) on top. The unifying principle in both: **no labels on concepts and no intermediate supervision — everything is learned from (instruction, initial scene, final scene) triples by "distant" supervision.**

- **Colors / shapes (object-level concepts):** each is a **neural embedding**. Execution is **quasi-symbolic and differentiable** (NSCL): an object set is a *probability vector* (entry *i* = P(object *i* ∈ set); `Filter(set, concept)` compares the concept embedding to each object's visual feature and updates that vector. No "this is red" label — selecting the wrong object moves the wrong block → wrong final scene → high bbox loss → gradients push the embedding toward the truly-red object.
- **Actions (`MovRight`/`MovTop` → SPG's `move_head`):** NSRM's **Action Simulator** network maps `(action concept, object-to-move location, reference location) → target location`, trained on `L_act = Σ‖pred−true‖ + β(1−IoU)`. "MovRight" *becomes* "place-to-the-right" because that's the only meaning reproducing the demonstrated positions. SPG's `move_head(dir)` is this same learned operator, repurposed to move the abstract head cursor (Sec. 5.2 says it's trained "similar to (Kalithasan et al., 2023)").
- **Parser (language → program):** there are **no gold programs**, so the Language Reasoner is trained with **REINFORCE**, reward = **−L_act** (programs yielding good final scenes get reinforced); mean action-loss is subtracted as a **baseline** to tame variance. Everything else trains by backprop; only the discrete-program output uses RL.
- **Curriculum:** *NSRM* — (i) single-step over object features, (ii) freeze action simulator, train spatial relations, (iii) fine-tune the multi-step splitter. *SPG (App. A.2/B)* — t0 visual attributes → t1 action concepts (long-range pick-and-place) → t2 inductive concepts (tower) → t3 interspersed new attributes (chocolate); the **5k "twin-tower"** pre-training corpus learns `move_head`, base attributes, and π_neural *before* any Sketch–Plan–Generalize.
- **Supervision actually increases from NSRM → SPG:** NSRM learns single-step actions from **only initial+final** scenes; SPG's *concept* learning needs **intermediate keyframes** for the dense IoU search. (SPG's inference-time planner in §7 reverts to initial+final-style scene-graph goals.)

> **Mental model:** *NSRM/NSCL learns the alphabet (concepts, relations, single-step actions) from initial→final pairs via distant supervision — embeddings + action-simulator by backprop, parser by REINFORCE, on a curriculum. SPG assumes that alphabet and learns the grammar (inductive, composable programs) via Sketch–Plan–Generalize.*

### 5.1 Sketch
GPT-4 with in-context examples turns Λ into a **tree of nested function calls** (the signature), e.g. `Staircase(steps=4, filter(cyan, legos))`. A **visual grounding module** then instantiates it on the scene. The grounder has three parts: a **ResNet-34** visual feature extractor; a **concept-embedding module** that learns *disentangled* embeddings for visual attributes (green, dice); and an **executor** with primitives like `filter`. Grounding `filter(green, dice)` returns the ordered list of green dice → `Tower(height=3, ObjectSet)`.

> **Concept — grounding & visual grounding.** A *symbol* (`"green"`, `tower`, `objects`) is just a token with no meaning on its own. **Grounding** = binding that symbol to the real thing it refers to — pixels, 3D coordinates, robot actions (the classic *symbol grounding problem*, Harnad 1990; defining "tower" with more words is circular, grounding breaks the circle by tying it to non-symbolic content). **Visual grounding** is the flavor that resolves language symbols against an *image/scene*: "given the word `green` and this camera view, which objects are meant?" Concretely, an ungrounded sketch `Tower(3, filter(green, dice))` is just a *description*; grounding resolves it to `Tower(3, [4,5,6])` — specific real blocks in *this* scene — which is now executable. SPG grounds along two axes: the **visual grounder** grounds *objects/attributes* in what the camera sees, and the **IoU-reward Plan search** grounds the *concept's structure* in what the human actually built — together pinning the symbol to reality (this is the machinery behind property (a)). The text-only GPT-4 baseline sidesteps visual grounding entirely (fed pre-computed `left(a,b)` relations, *assumes no distractors*) — part of why it's brittle.

> **Runtime learning of a new visual concept (+ gumbel-softmax).** The grounder isn't frozen — a brand-new attribute can be learned *at inference time* by backprop (Appendix D.2, Fig. 18). E.g. "tower of **chocolate** cubes": the grounder spots `chocolate` as unknown, **randomly initializes a new embedding**, runs the *known* `tower` program forward to a predicted scene, computes MSE+IoU loss vs. the demo's final scene, and backprops — **freezing everything except the new embedding** (disentanglement → learns `chocolate` without forgetting `magenta`, Fig. 17). *Problem:* the forward pass must "pick which block to place," but a discrete `argmax` over objects has no gradient → severs the learning path. *Fix:* **gumbel-softmax + masking** — `filter` scores → softmax gives a differentiable distribution over blocks; Gumbel noise + temperature τ make it behave like a near-one-hot *sample* (acts discrete, stays differentiable; anneal τ → harder); masking zeros out invalid/already-placed blocks. Forward pass acts like a hard pick; backward pass still passes gradients into the embedding (hard argmax at inference). *Same trick elsewhere:* discrete latents in VAEs, hard attention, DARTS (architecture search), discrete RL actions/tokens, mixture-of-experts routing (Jang et al. 2017). **Caveat (one unknown at a time):** SPG assumes exactly *one* new concept per episode — it can learn a new color (known structure) *or* a new structure (known grounding), not both at once, because embedding-learning needs a known program for the gradient to flow through and structure-search needs working grounding for a meaningful reward (chicken-and-egg). The curriculum (Fig. 10) is engineered to always introduce one unknown at a time.

### 5.2 Plan (the heart)
**The "head" abstraction:** placement is reasoned about with a **head** — a 3D cuboidal bounding box (x1,y1,x2,y2,depth) representing "where I'm considering placing next." Moving the head ≈ the robot mentally scanning placement locations.

**Primitive actions A_p:**
- `move_head(direction)` — a **neural operator** (trained on pick-and-place data) that nudges the head (left/right/front/back/top).
- `keep_at_head(objects)` — place the chosen object at the head.
- `store_head()` / `reset_head()` — a **stack-backed cursor**: `store_head` pushes the current head position onto a stack; `reset_head` pops and jumps back. Needed because the head is *one* cursor but many structures **branch** or must **return to an earlier reference point** after building a sub-part. Closest analogy: `pushd`/`popd`, OpenGL matrix push/pop, or **saving/restoring the turtle position in LOGO/L-systems** when drawing a branching tree. Example (pyramid from primitives, C.3): lay a row moving right, then `reset_head` to recover the row's *start* before going up for the next row — instead of spelling out "move left N times, up once." Payoff: enables branching structures, keeps plans **short** (short-program prior) and **size-agnostic** (no hard-coded back-tracking counts). *Caveat:* π_neural is **not** trained to emit `store_head`/`reset_head` (App. D.4) — they exist in the full search but are under-supported in the pruned search.

**Macro/compound actions A_c:** every learned concept `<cpt>` gets a `Make_<cpt>(size)` macro-action that executes its program. So `A = A_p ∪ A_c`.

**Reward — dense IoU, and how it's *aligned* to the demo (a common interview drill):**
- **Reward fires on *placement events*, not search steps.** Only `keep_at_head` and macro-actions are rewarded; `move_head`/`store_head`/`reset_head` give **0**. So the reward stream is driven by "I just placed an object," and alignment to the demo is by **placement count**, not step count — the several `move_head`s between placements are invisible to the reward (this dissolves the "next keyframe vs. one a few steps ahead?" confusion).
- **Matching is per-object, by identity.** When an object is placed, reward = **IoU between *that object's* resulting bbox and the *same object's* bbox in the demonstration**, not "whole current scene vs. keyframe N." The grounder produces an *ordered* `ObjectSet` and `keep_at_head` consumes it first-in-first-out (then removes it), and object identity is tracked across scenes by data association — so the k-th object the search places is the k-th placed in the demo.
- **Append-only makes the target well-defined.** Because a placed block is never moved again, each object has exactly *one* target — the spot it holds in the keyframe right after placement, which persists to the final scene. So an object placed *early* already has a known target; that's the "dense" part (you don't need the full structure finished to score block 1).
- **Worked decode of the discounting** (`Make_Tower(3)` = `1+1+1` vs. primitives `1 + γ²·1 + γ⁴·1`, γ=0.95): `keep`→place block1, IoU 1, γ⁰=**1**; `move_top`→**0**; `keep`→block2, γ²=**γ²**; `move_top`→**0**; `keep`→block3, γ⁴=**γ⁴**. The macro collapses 3 placements into one timestep → undiscounted 1+1+1, the thumb on the scale toward macros/short programs.
- **The dense signal replaces rollouts.** MCTS uses **off-policy Q-learning backups** (`Q(sₜ,a)=rₜ+γV(sₜ₊₁)`, `V(sₜ)=maxₐQ`) and **skips Monte-Carlo simulation** entirely — App. A.3 says rollouts are unnecessary precisely "because our reward is not completely sparse and the intermediate IoU rewards… guide the search effectively."

> **Demonstration anatomy (size-4 staircase).** Don't conflate three counts: **# demonstrations** = 3 in the main results (1 in the Fig. 2 example) — you show full staircase build(s), not separate tower demos; **structural decomposition** = **4 towers** (heights 1,2,3,4); **physical placements / keyframes** = 1+2+3+4 = **10 blocks**. So one demo of a size-4 staircase contains 10 placements organized as 4 towers — the search *explains* the 10-block trace as 4 towers via the macro rewards. (App. A.5's example uses height 3 = 6 blocks, `filter(red,blocks)=[1..6]`.)

> **Coordinates: world/3D, not image pixels.** The head and state are **3D bounding boxes with depth** `(x1,y1,x2,y2,d)` (Table 7); the construction loss is MSE over bboxes **+ center depth**. The RGB image only feeds the *grounder* (which object is "green dice"); spatial reasoning and placement happen in metric 3D box space. Minor rigor tension: they reason in 3D but report **2D-IoU**, which slightly under-measures vertical/depth error.

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
1. Relies on **perfect demonstrations** (no real demo noise). Note this is the deliberate *cost* of the design, not an oversight: heavy dependence on the demonstration is exactly what property (a) asks for (ground the concept in what the human did, not in priors), and the dense per-keyframe IoU reward is what delivers it — but with no prior to fall back on, a noisy/erroneous demo directly corrupts the reward and the search will faithfully reproduce the human's mistakes. Faithfulness and robustness-to-bad-demos are in tension; SPG buys the former and lists noisy demonstrations as future work.
2. Assumes **full observability** of all objects (no occlusion/partial views).
3. **Simulation only** — no real-robot results.
4. (From D.4) The reactive policy can't yet emit `reset_head`/`store_head`, and handling a growing `A_c` is left to future work.

**Additional critical points (your own analysis — strong to volunteer):**
- **Monotonic / append-only construction (a limitation beyond the three they list).** The learning pipeline's actions only ever *place* (`keep_at_head`); there is **no un-stack / remove primitive**, and the "remove placed object from the candidate list" bookkeeping hard-codes that each block is placed once and never touched again. So any concept whose *demonstration* requires temporarily displacing an already-placed block to reach another — e.g. lift the red cube to free the black one, then re-stack — **cannot be learned**. *Important contrast:* the **inference-time planner** (Sec. 6 / Fig. 8) *can* unstack/restack to fix an adversarial start — but that's the **goal-reaching planner** (richer `move(rel,a,b)` actions + scene-graph + preconditions), shown only **qualitatively**, and it *plans to realize a known structure* rather than *inducing a new concept* from such a demo. So the capability exists in the system but not in the concept-learning loop.
- **"Masking placed objects" is not reward hacking.** Removing an already-placed block from the selectable set is a **correct physical constraint** (a block on the table genuinely can't be picked-and-placed again) and doesn't touch the reward — IoU is still scored honestly. What it *rests on* are the real assumptions: **full observability** (a placed block is always seen) and **perfect, idealized placement**. So the right label for the concern is "idealized-environment assumptions," not "reward hacking."
- **LLM dependence & error compounding:** Sketch and Generalize both lean on GPT-4. A wrong signature or a botched abstraction propagates; the top-k re-ranking mitigates but doesn't eliminate this.
- **Hand-designed primitive vocabulary:** the head, the move/keep/store/reset primitives, and the IoU reward are engineered; the "discovery" is of compositions over a fixed primitive set, not of the primitives themselves. (And those primitives must be *pretrained* first — see §5.0 — so SPG inherits NSRM's assumptions too.)
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
- **Non-monotonic construction** — add an un-stack/remove primitive so concepts whose demos require re-arranging already-placed blocks can be *learned*, not just planned for.
- (Natural extensions) richer/learned primitives, learned constraint routing, perception robustness.

---

## 11. Where it sits in the literature (for "how does this relate to X?")
- **vs. LLM robot-coders** (Code as Policies, ProgPrompt, SayCan): SPG *learns new grounded concepts from demos* rather than only translating known tasks to code; demonstrably less reliant on the concept's name/prior.
- **vs. program induction** (DreamCoder, LILO/Grand et al.): same "concepts-as-programs, build a library" spirit, but SPG searches **grounded actions guided by dense demo reward** instead of enumerating programs — tractable in a Python-scale space and demonstration-driven.
- **vs. purely-neural rearrangement** (StructDiffusion): SPG adds explicit symbolic induction + modular reuse, which is exactly what gives the OOD generalization the neural baseline lacks.
- **Builds directly on NSRM (Kalithasan et al., ICRA 2023) + NSCL (Mao et al. 2019).** NSRM is the **direct predecessor** (same lab/authors): it learns grounded *single-step* manipulation programs (colors, shapes, relations, actions) from (instruction, initial scene, final scene) triples — embeddings + action-simulator by backprop, parser by REINFORCE, no intermediate supervision. SPG **assumes** that pretrained alphabet and adds the new piece: *inductive, looped, composable* programs learned via demonstration-guided MCTS (see §5.0). Concretely, SPG's `move_head`, `filter`, and visual attributes are NSRM/NSCL machinery; SPG's contribution is the Sketch–Plan–Generalize loop and the concept library on top. Code: github.com/dair-iitd/NSRM.

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
- **How is the dense reward aligned to the demo — which keyframe?** It isn't aligned by step or by a fixed look-ahead. Only *placement* actions are rewarded, so alignment is by **placement count**: after k objects placed, score each against the **same object's** target bbox in the demo (per-object, by identity, via the ordered ObjectSet + cross-scene data association). Append-only guarantees each object has a single well-defined target. The off-policy Q-backup then propagates these up the tree (no rollouts).
- **How are the primitives (colors/shapes/actions) trained?** Via NSRM/NSCL from initial+final scene pairs (distant supervision): concept embeddings + action-simulator by backprop on a bbox+IoU loss, the parser by REINFORCE (reward = −action loss), on a curriculum. SPG pretrains these on 5k twin-tower demos, then learns inductive programs on top.
- **Biggest *unstated* limitation?** Monotonic/append-only construction — no un-stack primitive, so concepts whose demo needs re-arranging already-placed blocks can't be learned (the planner can unstack to *reach* goals, but that's not concept learning).

---

## 13. Glossary / quick reference
- **SPG** — Sketch, Plan, Generalize.
- **Sketch (H\*_S)** — the LLM-produced grounded function signature.
- **Plan (H\*_P)** — the MCTS-found grounded action sequence.
- **Generalize (H\*_G = H\*)** — the abstracted Python program, added to L.
- **L** — concept library (executable function calls; updated continually).
- **head** — 3D bbox "placement cursor"; moved by a neural `move_head`.
- **store_head / reset_head** — push/pop the head position on a stack (turtle-graphics-style save/restore); enables branching structures and keeps plans short & size-agnostic. (π_neural can't emit them — App. D.4.)
- **append-only / monotonic construction** — each block is placed once and never moved again (no un-stack primitive); makes each object's reward target well-defined, but blocks learning concepts whose demo needs re-arrangement.
- **A_p / A_c** — primitive / compound(macro) actions; `A = A_p ∪ A_c`.
- **π_neural** — reactive policy that prunes primitives during search.
- **NSRM / NSCL** — the predecessor (Kalithasan 2023) / its basis (Mao 2019) that pretrain SPG's primitives (colors, shapes, relations, `move_head`) from initial+final scenes via distant supervision; SPG learns inductive programs on top.
- **reward alignment** — dense IoU is given only on placements, matched per-object by identity (ordered ObjectSet), aligned by placement count not step count.
- **grounding** — binding a symbol to the real thing it denotes (pixels, 3D coords, actions); breaks the circular symbol-grounding problem.
- **visual grounding** — grounding language symbols against an image: "which objects does `filter(green, dice)` mean in *this* scene?" (ResNet-34 features + disentangled attribute embeddings + `filter` executor).
- **gumbel-softmax + masking** — makes the discrete "pick which block" choice differentiable so a new attribute embedding (e.g. `chocolate`) can be learned at runtime by backprop; masking removes invalid blocks. (One unknown per episode: grounding-learning needs a known program; structure-learning needs working grounding.)
- **IoU reward** — overlap between built and demonstrated states.
- **Datasets I/II/III** — in-dist / name-reversed / larger-size(OOD).
- **R.D.** — relative decrease in IoU from in-dist to OOD (lower = better generalization).
- **Key numbers:** OOD R.D. **7.27/5.74%** (SPG) vs **63.25/74.72%** (SD+G); efficiency **0.933 @ 4k** vs **512k** expansions; in-dist program accuracy **1.00/0.83** vs **0.00** complex for GPT-4/4V.

---

## 14. Verbal summaries to rehearse

**~2 minutes:** State the goal (learn reusable, size-generalizing spatial concepts from a few demos), the gap (LLMs over-rely on priors and aren't grounded; neural nets don't generalize/compose; program search is intractable and demo-blind), the idea (Sketch→Plan→Generalize moves the guided search into grounded-action space where dense IoU rewards work, then abstracts to a library program), and the punchline (near-flat OOD degradation and strong name-reversal robustness, with ablations proving both the search and the library matter).

**~5 minutes:** Add the formal framing (inductive concept = recursion/composition; Bayesian MAP with a short-program prior), the Plan internals (head, primitives, macro-actions, the discounted-reward bias toward macros, neural pruning), the worked staircase example, the constraint bifurcation with Z3, and the concrete tables (1.00/0.83 in-dist; 7.27/5.74 vs 63–75 OOD; 4k vs 512k expansions; SPG-M+LMP and GPT-4V+VRF ablations). Close with limitations (perfect demos, full observability, sim-only) and future work (noise, beliefs, closed-loop, real robot).
