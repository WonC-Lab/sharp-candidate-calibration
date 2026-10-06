# T3.5 verdict — sharp identification versus headline novelty

**Decision: STOP after T3.5.** No manuscript rewrite, new PDF, or experiment is authorized by this note. The resulting theory is sharper and corrects a T3 inequality direction, but the two-headline threshold for the intended top optimization/OR tier is not yet met.

| Result | Exact characterization / identified set? | Converse? | Sharpness construction? | General beyond CFLP/portfolio? | Classification | Headline decision |
|---|---|---|---|---|---|---|
| Quality–support/Pareto frontier (Theorem A) | Yes: nondominated strict-improvement budgets and generalized inverse | Yes | Yes, by attained first-entry solutions | Yes | SUPPORTING; standard order-theoretic mechanism | No |
| Known-profile bounded-solver first-pass set (Theorem B) | Yes: every integer in a finite interval, or a tail plus $+\infty$ | Yes | Yes: explicit error sequence and MIO auxiliary-variable realization | Yes | SHARP PARTIAL IDENTIFICATION; proof short | Candidate, but not independently deep enough |
| Known-profile uncertain $\eta$/$\delta$ (equation 4) | Yes: per-budget membership, possibly holes | Yes | Yes, explicit plateau MIO | Yes | SUPPORTING extension of Theorem B | No separate headline |
| Unknown-profile proxy-label exactness set | Yes under stated broad profile class | Yes | Yes, step-profile MIO construction | Yes | SHARP but intentionally weak information model | Supporting |
| Joint observed-incumbent/valid-bound set (Theorem C) | Yes: exact per-budget membership, potentially nonconvex set, and sharp endpoint hull | Yes | Yes: observationally equivalent 0–1 MILPs and minimax width | Yes | Strongest HEADLINE candidate, subject to broader partial-identification priority check | Yes, conditionally |
| Uniform operational exactness criterion $z\le t<\Delta$ | Yes, necessary and sufficient over independent bounded-error class | Yes | Both boundary failures constructed | Yes | Sharp but algebraic consequence | Supporting |
| Unit-lattice operational region | Worst-case sharp sufficient region | Boundary counterexamples | Yes | Yes | Supporting integrality corollary | No |

## Answers to the required questions

**Q1 — Two strongest theorems.** Theorem C, the joint scalar-observation identified set with unknown full optimum and minimax lower bound, is first. Theorem B, the complete known-profile first-pass set under bounded solver error, is second. They are related: C deepens B's information model rather than providing an independent second mathematical mechanism.

**Q2 — Converses.** Yes within their declared information classes. Theorem B constructs an admissible error sequence for every listed budget and excludes every unlisted budget. Theorem C gives necessary-and-sufficient membership conditions for each budget; the splice construction proves sufficiency, and monotone envelopes prove necessity. These converses do not assert that every cost is realizable within a **fixed** CFLP or portfolio formulation.

**Q3 — Sharpness.** Yes, with explicit limits. Theorem B's endpoints and intermediate budgets are realized by a finite MIO with a global auxiliary binary. Theorem C's nonconvex set and both endpoint budgets are realized by observationally equivalent 0–1 MILPs that share the same scalar incumbent/bound/error-cap data. The interval minimax width follows for rules restricted to those scalar observations. If a rule also sees the complete full-universe formulation, the indistinguishability premise fails; no such broader minimax claim is made.

**Q4 — Subsumption.** The spectrum's $Q(0)$ is the exactness budget. The Pareto/support result subsumes exact support entry. Shifting thresholds by $\delta$ or $\eta$ gives oracle reference and ideal certificate labels; Theorem B adds finite solver error and missing-pass possibilities; Theorem C allows the full optimum and $g$ themselves to be unknown. The chain subsumes the T3 identities logically, but not the empirical validation work.

**Q5 — Genuine difference from closest MIO literature.** [Clarke–Stellato 2025](https://arxiv.org/html/2512.03547v1) bound a learned hierarchical solution's suboptimality; [Clarke–Stellato 2026](https://arxiv.org/html/2602.01476v2) calibrate early termination from solver state; [Wada et al. 2026](https://arxiv.org/abs/2608.01871) derive deterministic safe binary fixing from full-model information. None of the inspected primary texts gave this **specific** identified set for minimum nested candidate dimension under bounded incumbent error and uncertain full-optimum/bound information. This is a narrow finding, not a claim of firstness. More broadly, [Manski–Tamer 2002](https://onlinelibrary.wiley.com/doi/10.1111/1468-0262.00294) and a [2026 sharp monotone-threshold paper](https://arxiv.org/html/2608.08405v1) show that monotone partial identification and threshold inversion themselves are established techniques.

**Q6 — Hostile optimization theorist's likely view.** Theorem C is an exact, testable information result with a nonconvex identified set and minimax construction, so it is more than a rewritten definition. However, its proof is still a finite monotone-envelope argument; its relevance relies on withholding the full model formulation and knowing a uniform solver-error cap that may be unavailable under timeouts. A theorist could reasonably judge it a well-executed specialization of classical partial identification, not a breakthrough in MIO. Theorem B alone would likely be regarded as elementary first-passage arithmetic.

**Q7 — Ready for T4/T5?** **No for the declared top-tier objective.** The current theoretical novelty remains insufficient for the intended journal tier. Exactly one result (Theorem C) is a plausible nontrivial headline candidate, and its broader priority and operational information assumptions require further scrutiny. A second independent result would need to exploit concrete MIO structure to compute or tightly bound $Q$/$\Delta$ from realistic partial solves or valid global bounds, not merely restate the same monotone first-passage geometry. No T4/T5 action was taken.

## Corrections and open risks

1. The inherited T3 inverse-envelope inequality had reversed directions; it has been corrected in `T3_THEORY_NOTE.md` and re-proved in the T3.5 note.
2. Unknown $\eta$/$\delta$ can produce a **noncontiguous** identified set. Reporting only endpoint intervals as “exact” would be false.
3. The shorthand $\epsilon_{\rm ref}+\delta_U+z<h$ does not prevent finite-solver delay; uniform operational equality instead needs $z\le\epsilon_{\rm ref}+\delta_L$ and $\epsilon_{\rm ref}+\delta_U<\Delta$ (or the stronger lattice-safe upper condition $<h$).
4. A missing feasible incumbent violates the finite uniform-error premise and can make $+\infty$ attainable even when a full mathematical candidate exists.
5. Theorem C is a deterministic information statement about scalar records, not a computational complexity lower bound and not a probabilistic confidence set. A statistical uncertainty layer would require separate calibration and coverage analysis.
