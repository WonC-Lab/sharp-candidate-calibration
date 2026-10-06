# T3.6 — Solver-bound localization of minimum quality budget

This is a research audit, not manuscript text. Let $I=\{b_0,\ldots,N\}$, $\tau\ge0$, and let $J_B$ be attained minima for nested minimization problems, so $J_{b_0}\ge\cdots\ge J_N=s=J^*$. The information model consists **only** of valid scalar records $L_B\le J_B\le U_B$ and $L_*\le s\le U_*$; it excludes the full model coefficients, cross-budget cuts, and other solver-state information. Missing bounds may be represented by infinite endpoints, provided all constructions below remain finite where needed. The records are assumed jointly feasible.

## Theorem D1 (sharp feasible-profile envelopes)

Absorb the full bounds at (N): set $l_B=L_B,u_B=U_B$ for $B<N$, $l_N=\max(L_N,L_*)$, and $u_N=\min(U_N,U_*)$. Define

\[
a_B=\max_{j\ge B}l_j,\qquad c_B=\min_{i\le B}u_i,\qquad S=[a_N,c_N]. \tag{D1}
\]

The complete feasible-profile class is
\[
\mathcal F=\{j:j_B\ge j_{B+1},\ l_B\le j_B\le u_B\ (B\in I)\}. \tag{D2}
\]
It is nonempty if and only if $a_B\le c_B$ for every (B). Both (a) and (c) are members of $\mathcal F$, and every $j\in\mathcal F$ obeys $a_B\le j_B\le c_B$. Thus (D1) is coordinatewise sharp. In particular the sharp full-optimum interval is (S), not merely $[L_*,U_*]$. The suffix direction for lower bounds follows from $J_B\ge J_j\ge L_j$ for $j\ge B$; the prefix direction for upper bounds follows from $J_B\le J_i\le U_i$ for $i\le B$.

For fixed $s\in S$, the two simultaneously feasible profiles are
\[
m_B(s)=\max\{a_B,s\}\ (B<N),\quad m_N(s)=s;\qquad
M_B(s)=c_B\ (B<N),\quad M_N(s)=s. \tag{D3}
\]
Every feasible profile with terminal value (s) lies between them. To verify feasibility, (a,c) are nonincreasing, $a_B\le c_B$, and $s\le c_N\le c_B$; $m_B(s)\le c_B\le u_B$, while $M_B(s)\ge a_B\ge l_B$. The terminal values satisfy $l_N\le s\le u_N$. No independence of per-budget observations is assumed.

## Theorem D2 (exact identified set and converse)

Define $Q_j(\tau)=\min\{B:j_B-s\le\tau\}$. Its exact scalar-record identified set $\mathcal Q_\tau=\{Q_j(\tau):j\in\mathcal F\}$ satisfies
\[
b_0\in\mathcal Q_\tau\ \Longleftrightarrow\ S\cap[a_{b_0}-\tau,\infty)\ne\varnothing, \tag{D4}
\]
and, for $r>b_0$,
\[
r\in\mathcal Q_\tau\ \Longleftrightarrow\ S\cap[a_r-\tau,c_{r-1}-\tau)\ne\varnothing. \tag{D5}
\]
For every admitted (r), choose (s) in the displayed intersection and set $j_B=M_B(s)$ for $B<r$, $j_B=m_B(s)$ for $B\ge r$. The splice is nonincreasing: $M_{r-1}(s)=c_{r-1}\ge m_r(s)$, because $c_{r-1}\ge c_r\ge a_r,s$. Earlier budgets fail, as $c_{r-1}>s+\tau$, and (r) passes, as $a_r\le s+\tau$. For $r=b_0$, use (m(s)). Conversely, an actual first pass at (r) forces $a_r\le j_r\le s+\tau$ and, if $r>b_0$, $c_{r-1}\ge j_{r-1}>s+\tau$. These yield (D4)–(D5). The strict inequality at the predecessor is indispensable.

The extreme attainable budgets are
\[
q_{\min}=Q_{m(c_N)}(\tau),\qquad q_{\max}=Q_{M(a_N)}(\tau), \tag{D6}
\]
but $\mathcal Q_\tau$ need not contain every integer between them. At $B<N$, the sharp possible gap range has endpoints
\[
\min_{j\in\mathcal F}(j_B-s)=\max(0,a_B-c_N),\qquad
\max_{j\in\mathcal F}(j_B-s)=c_B-a_N. \tag{D7}
\]
At (N), both endpoints equal zero. Equation (D7) follows by taking $m(c_N)$ and $M(a_N)$, respectively. These are jointly realizable *profiles*, not independent choices of bounds at each budget.

## Corollary D3 (point identification and exactness)

For $q\in I$, the following are equivalent: (i) $\mathcal Q_\tau=\{q\}$; (ii) (q) is universally sufficient and, if $q>b_0$, its predecessor is universally insufficient; (iii)
\[
\bigl(q=N\ \text{or}\ c_q-a_N\le\tau\bigr)
\quad\text{and}\quad
\bigl(q=b_0\ \text{or}\ a_{q-1}-c_N>\tau\bigr). \tag{D8}
\]
The $q=N$ clause reflects the identity $J_N=s$, even with loose full records. For $B<N$, $c_B-a_N\le\tau$ is the necessary-and-sufficient **uniform sufficiency** test; $a_B-c_N>\tau$ is the necessary-and-sufficient **uniform insufficiency** test. If neither holds, both statuses occur among scalar-record-compatible profiles. Monotonicity means testing only the predecessor is necessary and sufficient for all earlier budgets. The equivalence follows from the sharp extrema (D7), not only one-way weak duality. At $\tau=0$, (D8) is the exact scalar-record condition for $B^\dagger=q$. It does not require every candidate MIP to have been solved exactly.

If all objectives lie on a **known common** lattice $\alpha+h\mathbb Z$, intersect each raw interval with the lattice first: $l_B^h=\alpha+h\lceil(l_B-\alpha)/h\rceil$, $u_B^h=\alpha+h\lfloor(u_B-\alpha)/h\rfloor$, including the full intersection at (N), then recompute $a^h,c^h,S^h$. Theorems D1–D3 remain exact for the lattice-valued sequence class: all constructed values and splices lie on the lattice. If only the spacing of objective *differences* is known, without the offset, safe sufficient tests are $c_q-a_N<h(\lfloor\tau/h\rfloor+1)$ and $a_{q-1}-c_N>h\lfloor\tau/h\rfloor$. These are not asserted necessary without the offset or additional arithmetic information.

### Integer-parametric CFLP

For an integer-valued CFLP objective, take $h=1,\alpha=0$, use integer-rounded bounds, and replace $\tau$ by $k=\lfloor\tau\rfloor$. The minimum quality dimension is (q) exactly when $q=N$ or $c_q^h-a_N^h\le k$, and $q=b_0$ or $a_{q-1}^h-c_N^h>k$. In particular, $B^\dagger=q$ can be proved from **partial** solves if the predecessor's best valid lower bound exceeds the full upper envelope and the candidate's upper envelope equals the full lower envelope after integer rounding. Raw positive gaps narrower than one can suffice: if $c_q-a_N<1$, the nonnegative integer gap must be zero; a strictly positive $a_{q-1}-c_N$ proves the predecessor's gap at least one. This certifies the *minimum dimension*, not just an objective value at one candidate set. Solver numerical tolerances require outward rounding of purported valid bounds before applying these exact statements.

### Relation to T3.5

Theorem C of `T3_5_PARTIAL_IDENTIFICATION.md` has $l_B=u_B-z$, $u_B=u_B$, and a full lower bound, and proves precisely the fixed-(s) monotone-envelope/splice principle. D1–D3 allow heterogeneous candidate dual bounds and explicit full primal/dual records; operationally they add useful failure certificates. Their proof, however, is **the same mathematical mechanism**. Candidate D is a sharp generalization of C, not an independent second headline theorem. The sharpness is relative to scalar records; model-specific cuts, reduced-cost structure, and common solver-tree information can shrink the class further.
