# T3.6 verdict — useful certification, no independent second headline

**Decision: STOP after T3.6.** Five T3.6 theory/audit files were produced. No master-manuscript file was changed, no experiment was run, and no portfolio safe-screening result was repackaged. The scalar solver-bound result is exact and operationally useful, but the hard two-independent-headline requirement is **not met**.

## Answers to the required questions

**Q1. Is there a genuinely independent second headline theorem?** No. Candidate D is a sharp generalization of T3.5 Theorem C from a common incumbent-error cap to heterogeneous per-budget primal/dual intervals, plus separate full-model bounds. Its mathematical engine is still monotone envelopes and a first-passage splice. The one searched fallback, perturbation stability, also reduces to a value-bound threshold shift.

**Q2. What is its exact mathematical statement?** Theorem D2: after endpoint intersection, set $a_B=\max_{j\ge B}l_j$, $c_B=\min_{i\le B}u_i$, $S=[a_N,c_N]$. Then $b_0$ is attainable iff $S\cap[a_{b_0}-\tau,\infty)\ne\varnothing$; each $r>b_0$ is attainable iff $S\cap[a_r-\tau,c_{r-1}-\tau)\ne\varnothing$. Theorem D3: $Q(\tau)=q$ is point-identified iff ($q=N$ or $c_q-a_N\le\tau$) and ($q=b_0$ or $a_{q-1}-c_N>\tau$). The exactness-budget case is $\tau=0$. These statements concern the scalar-record information class, not all deductions from the full formulations.

**Q3. Does it have a converse?** Yes within that class. Every budget satisfying the membership inequality is realized by the explicit upper-before/lower-after feasible-profile splice; every excluded budget violates a necessary first-pass inequality. The singleton criteria are necessary and sufficient because the candidate and predecessor gap extrema are attained by feasible profiles.

**Q4. Is it sharp?** Yes relative to scalar records: finite 0–1 MIO constructions realize the compatible profiles and identical valid bound reports. The set can be noncontiguous, e.g. $\{1,3\}$. Boundary examples show neither singleton inequality can be dropped. This is not sharpness for a fixed CFLP/MIQP formulation or a prescribed solver trace.

**Q5. Does it exploit information unavailable to Theorem C?** Operationally yes: heterogeneous candidate-specific dual bounds and a full upper bound yield `FAIL(B)` as well as `PASS(B)`; Theorem C's observation model does not contain these general records. Mathematically no independent mechanism appears. Both the new inequalities and converse are the same envelope argument under a richer interval vector.

**Q6. Can it compute or point-identify $Q(\tau)$ from realistic partial MIP solves?** Sometimes. Valid interim full and candidate bounds can satisfy D8 before every prefix is optimized. For unit-lattice integer CFLP, outward-rounded bounds can certify exactness even when displayed raw gaps remain nonzero but below one. If bounds remain loose, the exact identified set is reported rather than a guessed point. No finite stopping guarantee exists from arbitrary partial runs; exact solves provide a classical fallback. No new empirical claim about how often D8 is met is made.

**Q7. Would a hostile optimization theorist view it as more than interval arithmetic?** Probably not as a separate top-tier headline. The nonconvex set, sharp converse, and two-sided certificates are mathematically correct and practically informative, but the hostile description “intersected solver intervals and used monotonicity” is essentially accurate. There is no demonstrated optimal adaptive strategy, nontrivial runtime lower bound, or class-specific separation theorem. The information lower bound is explicitly confined to scalar reports.

**Q8. Is the master manuscript ready for T4/T5?** No under the stated two-headline objective. T3.5 Theorem C remains one plausible headline candidate; T3.6 supplies a useful supporting certification result but no independent second pillar. It also does not validate operational performance because empirical runs were prohibited. Do not start T4/T5 or rewrite claims without a new, separately justified result or a revised publication target.

## Release constraints and residual risks

1. Every bound used in a mathematical certificate must be globally valid for the same exact objective/feasible set; numerical display values require conservative outward handling. A time limit, incumbent alone, relative gap, or local node relaxation is not enough for every test.
2. An (N)-budget full solve can itself be expensive. The algorithm's policy has no proven cost-optimality; three-way statuses give no query bound without a refinement-progress model.
3. The finite-MIO realizations preserve scalar bound reports, not full formulations or solver internal trajectories. A more informed algorithm may obtain stronger deductions.
4. The lattice specialization requires an exact common objective lattice or a proven gap-spacing property. Floating-point CFLP coefficients, tolerances, and scaling can invalidate a naive unit-lattice claim.
5. Perturbation direction A was investigated and rejected as a second headline after a targeted primary-source search. No claim was made about alternatives B or C.
