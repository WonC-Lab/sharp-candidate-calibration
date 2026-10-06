# T3.7 — Reductions for minimum ranked-prefix quality budget

All problems here are minimization problems with a fixed, explicitly supplied ranking. A candidate prefix fixes all later entity binaries to zero. A supplied optimum is a feasible vector together with its objective value; its global optimality is promised in the general decision problem and independently verifiable in the reductions below.

## R1. Alternative-optimum hardness with a known optimal solution

**Theorem R1.** EXACTNESS-BUDGET and QUALITY-BUDGET at the fixed tolerance $\tau=1$ are NP-complete on a restricted class of 0–1 linear programs, even when (i) the full optimum $J^*=0$ is supplied, (ii) a globally optimal solution is supplied, (iii) every candidate prefix is feasible, (iv) objective and constraint coefficients have constant magnitude, and (v) only one entity is excluded at the tested prefix. Consequently, computing $B^\dagger$ or $Q(1)$ is NP-hard under the same conditions. This is strong NP-hardness in the standard bounded-coefficient sense.

**Reduction.** Let $\phi$ be a 3-CNF formula in variables $x_1,\ldots,x_n$. Introduce binary $y,p,z$ and the $x_i$. For each clause $C$, let $T_C(x)$ be the sum of its three literal truth indicators, using $x_i$ for a positive literal and $1-x_i$ for a negative literal. Form the 0–1 MILP
$$
\min 2y
\quad\text{subject to}\quad
y+z+T_C(x)\ge1\quad(C\in\phi),\qquad
p\ge1-y-z,\quad p\le1-y,\quad p\le1-z.
\tag{R1}
$$
The entity order is $(y,x_1,\ldots,x_n,p,z)$, with candidate budgets $B=1,\ldots,N=n+3$ and tested $k=n+2=N-1$. At prefix $B$, entities after position $B$ are fixed to zero. The assignment $y=1$, all other variables zero, is feasible in every prefix, so every $J_B$ is finite. The supplied full solution $v^*$ has $z=1$ and all other variables zero; it has value zero. Because the objective is nonnegative, $J^*=0$ is certified without solving anything.

At any prefix excluding $z$, an objective-zero solution must have $y=z=0$. Then the marker constraints force $p=1$, so no such solution exists before budget $k$; the clause constraints become exactly $T_C(x)\ge1$ for every $C$. Therefore
$$
\phi\text{ satisfiable}\ \Longrightarrow\ Q(0)=Q(1)=k,\qquad
\phi\text{ unsatisfiable}\ \Longrightarrow\ Q(0)=Q(1)=N.
\tag{R2}
$$
In the unsatisfiable case every proper prefix has only objective value 2, while the full problem has value 0. In the satisfiable case the first objective-zero prefix is exactly $k$ because $p$ is required. The construction has $n+3$ binaries and $|\phi|+3$ constraints; literal expansion uses coefficients and constants of magnitude at most three. It is computable in polynomial time. Membership in NP on this restricted class follows by guessing a prefix-$k$ binary feasible vector of objective at most the supplied $J^*+\tau$ and checking the linear inequalities. Thus both decision statements are NP-complete on the restriction. Since the supplied zero-cost vector plus nonnegative objective certifies optimality, this is not merely NP-completeness of recognizing a promised $J^*$.

**Interpretation limit.** The theorem excludes the explanation “finding the full optimum is hard”: that optimum and a proof of its value are provided. It does *not* exclude the explanation “optimization or feasibility inside the restricted prefix is hard.” Indeed the prefix-$k$ test is the original 3SAT instance encoded as a 0–1 feasible subsystem. This is a variable-fixing/alternative-optimum reduction, not evidence of a new complexity mechanism specific to searching over budget indices.

## R2. CFLP specialization at input-dependent tolerance

**Theorem R2.** For nonmetric, unit-demand CFLP with uniform capacity at least the number of clients, QUALITY-BUDGET is NP-complete when $\tau$ is part of the input, even with a supplied full optimum of value zero and one supplied global optimal solution, all prefixes feasible, and only one facility excluded at the tested prefix.

**Reduction.** Start with SET COVER: universe $E$ of $m$ elements, sets $S_1,\ldots,S_s$, and integer $K\ge1$. Make one unit-demand client for each element. Facilities are a fallback $f_0$, one facility $f_i$ for each set, and a final universal facility $z$. All have capacity $m$. Opening costs are $K+1$, $1$, and $0$, respectively. The fallback and $z$ have zero assignment cost to every client. Set facility $f_i$ has assignment cost zero to client $e\in S_i$ and $K+1$ otherwise. The ranking is $(f_0,f_1,\ldots,f_s,z)$, so $k=s+1=N-1$, and the threshold is $\tau=K$. Every prefix contains $f_0$ and is feasible. Opening $z$ alone and assigning every client to it is the supplied full optimum of value zero, certified by nonnegative costs.

At prefix $k$, any solution of cost at most $K$ cannot open $f_0$ or make a noncovered assignment, each costing $K+1$. Thus its opened set facilities cover all clients and their count is at most $K$. Conversely a cover by at most $K$ sets yields a prefix solution of cost at most $K$. Therefore $Q(K)\le k$ iff the SET COVER instance is YES. We may assume $K\le s$ (larger $K$ gives a trivial YES whenever the union covers), so all numerical costs are polynomially bounded; the reduction gives strong NP-hardness. Construction size and coefficient bit lengths are polynomial, and a facility-opening/assignment vector is an NP witness. This proves the restricted NP-completeness. The costs are explicitly **nonmetric**; no metric-CFLP claim follows. The threshold depends on $K$; R2 does not establish CFLP hardness for a fixed positive tolerance or for $B^\dagger=Q(0)$. It is a direct Set Cover cost encoding and is not promoted as novel facility-location hardness.

## R3. Why a dimension approximation barrier is not retained

R1 has a one-index YES/NO gap ($k$ versus $k+1$), insufficient for a meaningful multiplicative inapproximability claim. Arbitrarily many inert ranked entities can be inserted between $p$ and $z$, turning the NO value into a large $N$ without changing the actual optimization task. Such padding can manufacture any desired ratio gap in the *index* while adding no computational structure. Therefore no Set-Cover-style logarithmic approximation threshold, no constant-factor lower bound, and no practical approximation claim is inferred from these reductions. A useful approximation theory would need a nonpadding restriction or a dimension measure invariant under inert ranked entities.
