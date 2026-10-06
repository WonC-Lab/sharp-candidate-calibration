# T3.7 — Computational complexity of minimum candidate dimension

This note separates the difficulty of the full optimization, the restricted-prefix subproblem, and the ordered search over prefixes. All complexity claims use deterministic polynomial-time many-one reductions unless otherwise stated. The hardness core is a bounded 0–1 linear minimization model with rational coefficients in binary, a permutation of its designated entity binaries supplied explicitly, and prefixes formed by fixing all later entity binaries to zero. General MIO formulations require their own feasibility/bit-complexity assumptions; the results below are not silently extended to arbitrary nonlinear or real-number encodings.

## Formal problems and information models

Let $F$ be a nonempty finite feasible set, $F_B=\{v\in F:\operatorname{supp}_{\rm ent}(v)\subseteq C(B)\}$, and $C(B)$ the first $B$ entities in a fixed supplied order. Assume every tested $F_B$ is nonempty when $J_B$ is used; otherwise set $J_B=+\infty$. For objective $f$, define $J_B=\min_{v\in F_B}f(v)$, $J^*=J_N$, $g_B=J_B-J^*$, and $Q(\tau)=\min\{B:g_B\le\tau\}$ for rational $\tau\ge0$. EXACTNESS-BUDGET asks whether $Q(0)\le k$; QUALITY-BUDGET asks whether $Q(\tau)\le k$; QUALITY-BUDGET-VALUE outputs $Q(\tau)$; FULL-SPECTRUM outputs the finite staircase $\tau\mapsto Q(\tau)$ as its positive breakpoints and first-passing budget on each interval.

Model A receives only the full instance, candidate ranking, $\tau$, and $k$; $J^*$ is unknown. Model B additionally receives a **promised correct** value $J^*$. Model C additionally receives a **promised globally optimal** feasible vector $v^*$. These are promise-input formulations unless optimality is independently certified. The reductions R1 and R2 use nonnegative objectives and exhibit a zero-cost $v^*$, so the promises are trivially checkable there.

For bounded binary linear models, Model B/C decision problems lie in NP under the promise: a feasible prefix-$k$ binary vector of cost at most $J^*+\tau$ is a polynomial witness. R1 establishes NP-completeness even for constant coefficients, $J^*=0$, and supplied $v^*$. Model A is at least NP-hard by the same restriction. For binary rational linear models with finite optima, an NP oracle can compute the rational optimum after a polynomial-bit common-denominator scaling and binary search over a polynomial-bit objective range, then test the prefix. Hence the Model-A decision problems are in $P^{NP}$ for that encoding; no $P^{NP}$-completeness or sharper class is claimed.

The support-entry identity is exact:
$$
Q(\tau)=\min\{\operatorname{entry}_C(\operatorname{supp}_{\rm ent}(v)):
v\in F,\ f(v)\le J^*+\tau\}.
\tag{C1}
$$
For $\tau=0$, supplying *one* optimum only supplies an upper bound on this minimum; alternative optima can enter earlier. R1 makes deciding their existence NP-complete. Yet this is not the same objective as minimum support **cardinality**: with order $(1,2,3)$ and optimal supports $\{1,2\}$ and $\{3\}$, the earliest prefix is 2, while the minimum cardinality is 1. The mathematical distinction is rank-maximum versus support count; the complexity reduction is nevertheless a standard restriction-feasibility gadget, not a demonstrated new source of hardness.

## Parameters and approximation

R1 has $N-k=1$, so parameterizing by the number $d$ of excluded ranked entities is para-NP-hard: an $f(d)\operatorname{poly}(|I|)$ algorithm would solve 3SAT in polynomial time at $d=1$. The same conclusion applies to R2 for input-dependent $\tau$. R1 does not show W-hardness or fixed-parameter tractability in the budget $k$ or support size: in the reduction $k$ grows with the SAT instance. No treewidth claim is made.

If there are $p$ binary variables total and all remaining variables form a rational LP when those binaries are fixed, exhaustive enumeration of the $2^p$ assignments followed by polynomial-time LP optimization computes every $J_B$ and $Q$ in $2^p\operatorname{poly}(|I|,N)$ time. Thus the problem is FPT in total binary dimension $p$ under this standard mixed-binary model. This is the familiar fixed-integer-dimension boundary, not a new candidate-budget algorithm. The approximation ratio $\widehat Q/Q$, subject to $g_{\widehat Q}\le\tau$, is not structurally stable under insertion of inert ranked entities; R3 explains why no meaningful ratio barrier is retained.

## Entire spectrum and exact-value oracle

There are at most $N-b_0$ positive breakpoints: a breakpoint requires a strict decrease in the $N-b_0$ adjacent candidate values. This upper bound is attained. For example, use one demand and $N$ ranked facilities, require at least one to open, give them strictly decreasing nonnegative opening costs $c_{b_0}>\cdots>c_N=0$, and zero assignment costs. Then $J_B=c_B$ and every preterminal budget contributes a distinct positive breakpoint. Equivalently, the finite 0–1 option construction in T3.5 E6 realizes any rational nonincreasing profile. Hence the count is tight but elementary.

Given the exact vector $(J_{b_0},\ldots,J_N)$, one scan groups equal-value plateaus and outputs every breakpoint and its first budget in $O(N-b_0+1)$ arithmetic operations, apart from output bit cost. With an exact candidate-value oracle and supplied $J^*$, one fixed $Q(\tau)$ needs at most $\lceil\log_2(N-b_0+1)\rceil$ midpoint queries after the endpoint $N$ is known. A matching Boolean-information lower bound follows on step profiles $J_B=1$ for $B<q$, $J_B=0$ for $B\ge q$: the numeric oracle reveals only one comparison bit per query, and there are $N-b_0+1$ possible $q$. This is classical ordered search, not novel MIO complexity.

For FULL-SPECTRUM, if arbitrary rational strictly decreasing profiles are allowed and $J_N$ is supplied, every nonterminal $J_B$ must be queried in the worst case to output all **numeric** breakpoints exactly: an unqueried value can be perturbed strictly between its neighbors without changing any queried response, but it changes a breakpoint. Thus the exact-value oracle requires $N-b_0$ queries in that model, matching the output size. This is not a computational runtime lower bound for a solver with access to model coefficients.

## Complexity conclusion

The full-optimum objection is defeated by R1: even a certified value and one optimum do not identify the earliest optimal prefix. But the remaining difficulty is precisely a hard restricted-prefix feasibility/optimization problem. The ordered budget search itself is logarithmic under an exact-value oracle. No independent complexity mechanism, meaningful inapproximability threshold, or nontrivial parameterized dichotomy emerged.
