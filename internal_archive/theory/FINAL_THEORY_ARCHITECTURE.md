# T4 — Final theory architecture (locked proposal, 2026-10-06)

This is a theory-to-evidence architecture, not a rewritten manuscript or a new experimental protocol. The unit of analysis is a fixed nested candidate family \(C_\theta(b_0)\subseteq\cdots\subseteq C_\theta(N)\), with attained minimization values \(J_B\), \(s=J_N=J^*\), and \(g_B=J_B-s\). Define the **Candidate Quality-Budget Spectrum** by
\[
Q_C(\theta;\tau)=\min\{B\in\{b_0,\ldots,N\}:g_B\le\tau\},\qquad \tau\ge0.
\]
The exactness budget is \(B^\dagger_C=Q_C(0)\). Infeasible prefixes have \(J_B=+\infty\); all finite-envelope statements below apply after the feasible floor \(b_0\). The candidate ranking and family are fixed, not optimized by this theory.

## The single headline result

**Headline theorem — Sharp Identification of the Candidate Quality Budget from Solver Records.** The information class contains only globally valid scalar records \(L_B\le J_B\le U_B\), plus \(L_*\le s\le U_*\), and nesting/monotonicity. It does not contain the full formulations, cross-budget cuts, or a solver trace. Absorb the full records into \(l_N=\max(L_N,L_*)\), \(u_N=\min(U_N,U_*)\), with \(l_B=L_B,u_B=U_B\) elsewhere. Put
\[
a_B=\max_{j\ge B}l_j,\qquad c_B=\min_{i\le B}u_i,\qquad S=[a_N,c_N].
\]
Assuming compatible finite records (\(a_B\le c_B\) for all \(B\)), the **exact**, potentially noninterval identified set is
\[
\mathcal Q_\tau=\{b_0:S\cap[a_{b_0}-\tau,\infty)\ne\varnothing\}
\;\cup\;
\{r>b_0:S\cap[a_r-\tau,c_{r-1}-\tau)\ne\varnothing\}.
\]
Every admitted budget is realized by a monotone upper-before/lower-after splice of feasible profiles; every excluded budget violates a necessary first-pass inequality. Coordinate envelopes, endpoint bounds, and a finite-MIO observational-equivalence construction establish sharpness **only for this scalar-record information class**. A fixed CFLP or MIQP formulation can impose additional restrictions. The set can have holes (T3.5 Example E7 gives \(\{1,3\}\)); its endpoint hull is not generally the identified set. T3.5 Theorem C is the uniform-error special case \(l_B=u_B-z\) with its full bound. T3.6 D1–D3 are **one unified theorem and corollaries**, not separate discoveries.

**Corollary — Necessary and Sufficient Point Identification.** \(\mathcal Q_\tau=\{q\}\) exactly when
\[
(q=N\ \text{or}\ c_q-a_N\le\tau)\quad\text{and}\quad
(q=b_0\ \text{or}\ a_{q-1}-c_N>\tau).
\]
The first condition universally passes \(q\), the second universally fails its predecessor. At \(\tau=0\) this identifies \(B^\dagger\); it does not demand exact solution of every prefix. If bounds remain loose, the theorem returns a set, not a guessed point. This is deterministic partial identification, not a confidence set or an algorithm with a runtime guarantee.

**Corollary — Integer-Lattice Localization.** If all *optimal values* \(J_B\) lie in a proven common lattice \(\alpha+h\mathbb Z\), valid bounds can be conservatively mapped to lattice endpoints by ceiling the lower and flooring the upper, after accounting for numerical validity; then apply the same theorem with \(\lfloor\tau/h\rfloor\). For integer-data CFLP the mathematical optimum lattice has \(h=1\). This does **not** mean every feasible continuous-flow solution has integer cost. Raw floating solver bounds are not automatically outward-certified. The existing integer-oracle code uses `UB-LB < 0.95` and `round(UB)` but does not implement a separately audited outward-rounding interval; T5 must present that as a numerical qualification, not as an exact-arithmetic proof.

## Supporting result ledger

| Result / inherited source | Final rank | Reason and disposition |
|---|---|---|
| Quality-support first-entry identity, \(Q(\tau)=\min\{\operatorname{entry}_C(\operatorname{supp}_{ent}v):f(v)\le J^*+\tau\}\) | MAIN THEOREM (short) | Gives structural meaning to the spectrum; state once, including ties, all optimal supports, and feasibility floor. |
| Spectrum/Pareto strict-improvement steps (T3.5 A) | COROLLARY | Generalized inverse of a monotone frontier; no independent novelty claim. |
| Exactness \(Q(0)\), \(\tau\)-optimal support, \(\Delta_C\) separation | COROLLARY / DEFINITION | Fold into quality-support statement, not duplicate exactness theorems. |
| Reference-proxy translation \(Q(\epsilon_{ref}+\delta)\), certification translation \(Q(\epsilon_{cert}-\eta)\) when defined | PROPOSITION | Algebraic tolerance shifts; state assumptions and distinguish reference error from LP looseness. |
| Finite incumbent-error first-pass sets (T3.5 B), unknown-error examples | LEMMA / APPENDIX ONLY | Special information models subsumed by headline; keep counterexamples for strict boundaries and missing incumbents. |
| T3.5 C plus T3.6 D1–D3 | HEADLINE THEOREM + point-ID and lattice COROLLARIES | One monotone-envelope and sharp-splice mechanism. Do not count as two pillars. |
| Solver-specific three-way PASS/FAIL/UNRESOLVED and feasible-profile algorithm | PROPOSITION / APPENDIX ONLY | Operational implementation of headline, not a new algorithmic complexity guarantee. |
| Supplied-optimum NP-completeness R1 | PROPOSITION (main text), proof APPENDIX | Even with correct \(J^*\), a supplied optimum, fixed ranking, feasible prefixes, and \(N-k=1\), earlier optimal-prefix existence is NP-complete. Hardness is restricted-prefix alternative-optimum feasibility, not the ordered-index search. |
| R2 CFLP specialization | APPENDIX ONLY | Set-cover encoding; does not justify approximation hardness for prefix ratio. |
| LP / min-cost flow / matroid tractability, exact-value-oracle search | REMARK | Classical polynomial subproblem and ordered-search boundary. |
| Split conformal calibration of required dimension | MAIN THEOREM only as an application guarantee | Established conformal rank principle; never a claimed new conformal theorem. Exact coverage requires genuine/conservative exactness labels, whereas the Phase B real-valued labels target reference fidelity. |
| Geometric expansion and union-bound statistical transfer | LEMMA / REMARK | Structural accounting and elementary probability, not headline contributions. |

The three strongest supporting mathematical results are (i) quality-support first-entry/Pareto geometry, (ii) proxy/certificate translation with finite-error qualification, and (iii) supplied-optimum R1 with classical tractable boundaries. Solver-bound point-ID is the most important corollary of the headline, not an independent supporting pillar.

## Claim boundaries and evidence interface

The theory identifies an information-limited object. Even if the full MIO is available, solving it or adding model-specific cuts may shrink the identified set. No fast computation of all \(J_B\), no optimal adaptive query schedule, and no new NP-hardness mechanism are claimed. The scalar-envelope mathematics is related to established partial identification and monotone-function intervals; originality is the specific candidate-quality object and its solver-record integration, subject to specialist priority review.

Empirical observability on locked artifacts is asymmetric. Phase B stores full-MIP incumbent/bound, full LP bound, and only reached candidate stages; exact-oracle files store \(J^*\) and \(B^\dagger\) labels but not a complete \(J_B\) vector. Thus one can test the LP certification barrier and exactness-label coverage, not plot a full \(g_B\) or \(Q(\tau)\) staircase or retrospectively count D3 early identification. Re-solving the same seeded instances would be a provenance-preserving **re-analysis**, not a new statistical sample, if environment, ranking, solver and budgets were frozen and hashes/derivations recorded. It is not necessary in T4 and was not done. The manuscript should report the evidence gaps rather than fabricate trajectories.

## Paper identity and ordered titles

Primary identity: a theory-of-information paper about the **minimum candidate dimension for optimization quality**, with conformal calibration and computational certification as distinct layers. Ranking (English manuscript titles):

1. *Minimum Candidate Dimension for Mixed-Integer Optimization: Sharp Identification and Solver Certification*.
2. *Candidate Quality Budgets for Repeated Mixed-Integer Optimization: Sharp Identification, Calibration, and Certification*.
3. *Quality–Dimension Frontiers in Repeated Mixed-Integer Optimization*.
4. *Sharp Identification of Optimization Quality Budgets from Incomplete Solver Records*.
5. *Candidate Quality-Budget Spectra for Repeated Mixed-Integer Optimization*.

The first title names the mathematical object and the differentiating information theorem without implying a new conformal principle or proven speedup.
