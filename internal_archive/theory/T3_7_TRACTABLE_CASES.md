# T3.7 — Tractable boundaries for ranked-prefix quality budget

The following are exact polynomial-time results, not approximation statements. They clarify which structural features remove the restricted-prefix MIO difficulty. The ranking is fixed and explicitly given throughout.

## T1. Rational linear-programming families

**Theorem T1.** Let $F=\{x\in\mathbb R^n:Ax\le b,\ x\ge0\}$ be a nonempty rational polyhedron with finite attained minimum of rational linear objective $c^\top x$. Let $C(B)$ be a fixed prefix of coordinates, and define $F_B$ by $x_i=0$ for all $i\notin C(B)$. Then QUALITY-BUDGET-VALUE, EXACTNESS-BUDGET, and FULL-SPECTRUM are computable in polynomial time in the binary input length and $n$, assuming each needed restricted optimum is finite or infeasibility is reported.

**Proof.** A rational LP algorithm computes the full optimum $s$ and every restricted optimum $J_B$ in polynomial time. Compare $J_B-s$ to the rational $\tau$ and scan for the first pass; scanning adjacent values also emits all strict-improvement breakpoints. Alternatively, once $s$ is known, a single QUALITY-BUDGET predicate at $B$ is LP feasibility of $Ax\le b$, $x\ge0$, $x_i=0$ for $i\notin C(B)$, and $c^\top x\le s+\tau$. This makes earliest support entry on an LP optimal face polynomial even though minimum **cardinality** of a feasible support can be difficult in other formulations. The proof uses no integer variables or fixed charges. □

## T2. Minimum-cost network flow with ranked arcs

**Theorem T2.** Consider a rational capacitated single-commodity minimum-cost flow instance with fixed supplies/demands and a fixed ranking of optional arcs. Arc capacities, conservation equations, and linear arc costs are retained; the budget-$B$ problem simply fixes flow on later optional arcs to zero. If the full instance is feasible and bounded, its entire quality spectrum is computable in polynomial time.

**Proof.** Each prefix is again a minimum-cost flow instance on a subnetwork, with possible infeasibility at small $B$. Polynomial minimum-cost flow algorithms solve the full network and each of the at most $N-b_0+1$ subnetworks. Compute gaps and scan as in T1. If a candidate is infeasible, treat its cost as $+\infty$; monotone feasibility and finite full feasibility ensure a first finite passage for every finite $\tau$. The repeated solves give a polynomial, though not necessarily strongly polynomial, bound in the chosen rational encoding. □

This class retains shared flow-conservation constraints and capacities, so it is more than a separable sum. It loses CFLP's binary opening/fixed-charge decisions, the feature used by the Set Cover reduction R2. Polynomial solvability of each candidate, not a new method for searching the index, explains tractability.

## T3. Weighted matroid bases under prefix restriction

**Theorem T3.** Let a matroid of rank $r$ be given by a polynomial-time independence oracle on a ground set with rational element weights and a fixed ranking. At budget $B$ choose a minimum-weight basis of the full matroid contained in the first $B$ elements, if one exists. Then $Q(\tau)$ and the full spectrum are polynomial-time computable in the oracle/input model.

**Proof.** Compute the rank and, when feasible, a minimum-weight basis of each restriction to a prefix using the standard weighted matroid greedy algorithm, stopping at rank $r$. The greedy basis theorem gives its exact weight. There are at most $N$ prefixes, hence a polynomial number of independence calls and arithmetic operations. Compare each weight to the full-basis weight plus $\tau$. □

This result includes minimum spanning-tree support over ranked edges in the graphic-matroid case. A prefix may contain a low-weight spanning tree or none; both are decided by greedy independence tests. The result is standard matroid optimization, not a complexity novelty claim.

## T4. An assignment-separable facility-location limit

If facility opening costs are identically zero, capacities do not bind, and every client may be assigned independently to any allowed ranked facility, then $J_B=\sum_j d_j\min_{i\le B}c_{ij}$. Maintaining each client's running minimum computes all candidate values in $O(Nm)$ comparisons for $m$ clients; one scan returns $Q$ and all breakpoints. This boundary is much easier than CFLP with fixed charges and should not be used to suggest the experimental CFLP is tractable.

The preceding cases do not contradict R1/R2: in those reductions an integral switch or fixed-charge choice encodes SAT/SET COVER inside a single prefix test.
