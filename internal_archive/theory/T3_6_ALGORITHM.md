# T3.6 — Bound-driven quality-budget certification algorithm

All arithmetic below uses **valid global** minimization bounds, not local-node values or a solver's rounded display. A reported incumbent is an upper bound only after feasibility is verified. Keep the full solve as the (N)-th candidate solve, intersecting its bounds with any separate full-model records.

## Exact information state

At iteration (t), retain all current $L_B(t),U_B(t),L_*(t),U_*(t)$. On refinement, replace a lower bound by its maximum with earlier lower bounds and an upper bound by its minimum with earlier upper bounds. Compute (a(t),c(t),S(t)) and the exact set $\mathcal Q_\tau(t)$ from (D4)–(D5). Inconsistent records $(a_B>c_B)$ indicate an invalid numerical certificate, incompatible models, or an implementation error; do not report a budget. If some budget lacks a feasible incumbent, set $U_B=+\infty$, and if a globally valid lower bound is absent, set $L_B=-\infty$. A finite full incumbent is needed for a useful failure test.

Define `FAIL(B)` for $B<N$ by $a_B-c_N>\tau$; define `PASS(B)` by $c_B-a_N\le\tau$, with `PASS(N)` unconditional. Each is an exact *uniform* classification over the scalar-record class. If neither is true, return `UNRESOLVED`, never `PASS`. Under valid records, `FAIL(B)` propagates to all earlier budgets and `PASS(B)` to all later budgets. The exact set can nevertheless have holes, so preserve $\mathcal Q_\tau$, not only its endpoint hull.

## Adaptive procedure

```text
Input: nested MIOs B=b0,...,N; threshold tau>=0; valid initial bounds
repeat:
    absorb new valid full/candidate bounds; compute a,c,S and exact Qset
    if Qset is empty: stop with invalid-record diagnostic
    if Qset={q}: return q with the D8 predecessor-failure/pass witnesses
    B_low=min(Qset); B_high=max(Qset)
    if full optimum uncertainty is the common obstacle, refine the full MIO
    else choose an unresolved B near the median of Qset or a current boundary
         refine its candidate MIO, retaining all valid interim bounds
    if no query/refinement can be completed: return Qset as unresolved
```

The selection rule is a practical heuristic, not a proven optimal policy. One may choose the budget with the smallest estimated certification effort rather than the median. A single full-bound improvement can sharpen every candidate gap; a candidate improvement usually acts through a suffix/prefix envelope. Warm starts may transfer feasible incumbents upward in the nested family, but any transferred solution must be checked in the destination formulation.

**Invariant and safety.** Every actual objective profile belongs to the current feasible class, so its true $Q(\tau)$ belongs to the current $\mathcal Q_\tau$. Valid tightening makes the class and identified set shrink. If the algorithm returns (q), (D8) proves (q) passes and its predecessor fails for every compatible profile; hence no false sufficiency or minimality certificate is possible under the validity assumptions. An incomplete MIO run can still produce useful certificates. A timeout alone is not a bound.

**Finite termination.** If the full and each queried candidate MIO can be solved to *mathematically exact* objective bounds in finite work, a fallback that solves the full problem and binary-searches exact candidate predicates terminates. After the full optimum is known, at most $\lceil\log_2(N-b_0+1)\rceil$ exact candidate-predicate queries identify the first passing budget, up to a conventional endpoint/rounding adjustment. Without exact finite solves or an explicit positive separation from the threshold, no finite termination follows merely from asymptotic bound convergence. At an equality $J_B-s=\tau$, upper and lower records may approach the common value forever without a finite `PASS` witness. An objective lattice may restore finite stopping once bounds are narrower than the relevant lattice spacing; this still depends on the solver producing valid bounds of that width in finite time.

**Query-complexity limit.** If an exact query returns only the monotone Boolean predicate $J_B-s\le\tau$, classical binary search is worst-case optimal up to constants: there are $N-b_0+1$ possible first-pass positions, requiring at least that many leaves and worst-case depth at least $\lceil\log_2(N-b_0+1)\rceil$ in a binary decision tree. This counting argument applies to that *restricted Boolean oracle*, not to a rich primal-dual record that may reveal several budgets at once. Three-way partial queries can return `UNRESOLVED` repeatedly, so there is no finite worst-case query/refinement bound without a progress or cost model. Conversely a strong full/candidate certificate can identify $Q$ in fewer queries than binary search. No claimed novelty or optimality attaches to midpoint, doubling, or threshold search here.

For unknown (N), exponential/doubling search first brackets a monotone transition, followed by binary search; that is classical. Online bidding optimizes a different cost objective when increasing budgets must be purchased. Parametric search has still another oracle structure. None supplies an automatic new result for partial MIO bound refinement, whose per-query time and information gain are instance dependent.

The claim here is operational correctness and exact scalar-record stopping, not an improved complexity theorem or a scheduler competitive ratio. An observational lower bound is given separately in `T3_6_SHARPNESS.md`.
