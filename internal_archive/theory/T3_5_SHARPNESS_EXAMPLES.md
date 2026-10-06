# T3.5 — Complete finite MIO sharpness constructions

All examples are analytic constructions, not new empirical instances. Unless stated otherwise, $y_i\in\{0,1\}$, $\sum_i y_i\le1$, $C(B)=\{1,\ldots,B\}$, and excluded $y_i$ are fixed to zero. A global auxiliary variable is not subject to entity-prefix fixing. Each table reports mathematical optima, not solver-reported floating-point statuses.

## E1. Frontier, first-entry support, plateau, and sharp $\Delta$

Take $N=3$, $F=\{y\in\{0,1\}^3:\sum_i y_i\le1\}$, $f(y)=4-2y_1-y_2-4y_3$, and $b_0=1$. The null point costs 4; options 1, 2, 3 cost 2, 3, 0. Thus

| Budget $B$ | $C(B)$ | $J_B$ | $g_B=J_B-J^*$ | $Q(\tau)$ transition |
|---:|---|---:|---:|---|
| 1 | $\{1\}$ | 2 | 2 | $Q(\tau)=1$ at $\tau\ge2$ |
| 2 | $\{1,2\}$ | 2 | 2 | Never a first-pass budget |
| 3 | $\{1,2,3\}$ | 0 | 0 | $Q(\tau)=3$ for $0\le\tau<2$ |

Here $J^*=0$, $B^\dagger=3$, $\Delta=2$, and the nondominated frontier is $\{(1,2),(3,0)\}$; $(2,2)$ is dominated. The jump at $\tau=2=\Delta$ is from 3 directly to 1. The objective at support $\{2\}$ is worse than the existing support $\{1\}$, so a plateau need not be caused by multiple near-optimal supports. At $\tau=2$, $\Omega^\tau$ contains supports $\{1\}$ and $\{3\}$; the first-entry minimum selects budget 1. Oracle reference with $J_{\rm ref}=0$, $\epsilon_{\rm ref}=2$ has label 1, nonexact; ideal certificate with $L=-1$ ($\eta=1$), $\epsilon_{\rm cert}=3$ also has label 1. Decreasing either effective threshold below 2 moves the label to 3. This proves strictness of both translation boundaries.

## E2. Exact first-pass set for bounded solver error

Take $N=3$, $F=\{(y,w):y\in\{0,1\}^3,\ \sum_i y_i\le1,\ w\in\{0,1\}\}$, and $f(y,w)=3-y_1-2y_2-3y_3+2w$. The global binary $w$ supplies a feasible incumbent exactly 2 above every restricted optimum. The minima are $J=(2,1,0)$, $J^*=0$, $g=(2,1,0)$, $B^\dagger=3$, and $\Delta=1$. At $t=2$, $z=2$, $Q(t)=1$, $Q(t-z)=Q(0)=3$.

| Feasible solver-error sequence $\zeta$ | Observed residual $g+\zeta$ | First pass $R_U(2)$ |
|---|---|---:|
| $(0,0,0)$ | $(2,1,0)$ | 1 |
| $(2,0,0)$ | $(4,1,0)$ | 2 |
| $(2,2,0)$ | $(4,3,0)$ | 3 |

Every integer in the identified set $\{1,2,3\}$ is realized by a **feasible** incumbent of this MIO. With $t=1<z=2$ and $\zeta=(2,2,2)$, the residual is $(4,3,2)$ and no budget passes, so $R_U(1)=+\infty$ even though $Q(1)=2$. With $\zeta=(2,0,0)$ at $t=1$, $R_U(1)=2$; with $\zeta=(2,2,0)$ it is 3. Hence the exact set in this regime is $\{2,3,+\infty\}$. The error sequence may be nonmonotone without invalidating the first-pass characterization.

## E3. Unknown reference or lower-bound quality can create holes

Use E1's $g=(2,2,0)$ and $Q(\tau)=3$ for $0\le\tau<2$, $Q(\tau)=1$ for $\tau\ge2$. For oracle reference tolerance $\epsilon_{\rm ref}=0$, let $\delta\in[0,2]$. To make every reference value physically feasible while preserving the integer optima, add a global continuous variable $x\in[0,2]$ of objective cost $x$ and take the full reference as the option-3 solution with $x=\delta$. The candidate minima remain $(2,2,0)$. Then $R_{\rm ref}=Q(\delta)$ is 3 when $\delta<2$ and 1 when $\delta=2$; budget 2 is impossible. The exact identified set is $\{1,3\}$, not the endpoint hull $[1,3]\cap\mathbb Z$.

For certification on the same gap profile, take $\epsilon_{\rm cert}=2$, $z=0$, and $\eta\in[0,2]$, so the valid full lower bound is $L=-\eta$ and effective threshold $t=2-\eta\in[0,2]$. The ideal certificate label is again only 1 or 3, never 2. Here $J^*=0$, $B^\dagger=3$, $\Delta=2$; at $\eta=0$, the nonexact budget 1 passes the certificate tolerance, whereas at any $\eta>0$, the label is 3. This demonstrates why uncertain-comparator endpoint intervals need not be exact identified sets.

## E4. Finite solver error changes the exactness identification boundary

Let $N=2$, $F=\{(y,w):y\in\{0,1\}^2,\ y_1+y_2\le1,\ w\in\{0,1\}\}$, $f(y,w)=1-y_2+0.2w$. Prefixes are $C(1)=\{1\}$ and $C(2)=\{1,2\}$. Then $J=(1,0)$, $J^*=0$, $g=(1,0)$, $B^\dagger=2$, $\Delta=1$; **both optimal objective values lie on the unit lattice**, although nonoptimal incumbent values need not. Set $J_{\rm ref}=0$ ($\delta=0$), $\epsilon_{\rm ref}=0.1$, and solver-error cap $z=0.2$. The suggested condition $\epsilon_{\rm ref}+\delta_U+z<1$ holds ($0.3<1$), but choosing $\zeta_2=0.2$ gives observed full-budget residual $0.2>0.1$ and **no reference-fidelity pass**. The correct uniform condition requires $z\le\epsilon_{\rm ref}+\delta_L$, which fails here. This is a failure of operational point identification, not of the mathematical unit-lattice separation result.

For the opposite boundary, use the same $g=(1,0)$, $z=0$, and $\epsilon_{\rm ref}+\delta=1=\Delta$. Then budget 1 passes and is nonexact. For a certificate, take $L=-0.4$, $\eta=0.4$, $\epsilon_{\rm cert}=1.4=\eta+\Delta$; budget 1 passes exactly at the nonexact threshold. If instead $\epsilon_{\rm cert}<\eta$, even the true optimum at budget 2 fails against this fixed bound.

## E5. Proxy label alone: same observation, different exactness budget

Let $N=2$, $F=\{y\in\{0,1\}^2:y_1+y_2\le1\}$, $C(1)=\{1\}$, $C(2)=U$, $J_{\rm ref}=0$, and $\epsilon_{\rm ref}=0$. Compare $f_A(y)=1-y_1+2y_2$ and $f_B(y)=1-y_1-2y_2$. Both candidate-restricted problems at budget 1 are identical and return $J_1=0$, so both operational oracle reference labels are 1. In A, $J_A^*=0$, $g_A=(0,0)$, $Q_A(0)=B_A^\dagger=1$, $\Delta_A=+\infty$, $\delta_A=0$. In B, $J_B^*=-1$, $g_B=(1,0)$, $Q_B(0)=B_B^\dagger=2$, $\Delta_B=1$, $\delta_B=1$, and its effective tolerance equals $\Delta_B$. Thus reference label 1 alone cannot reveal exactness when $\delta$ may range over $[0,1]$. This is observational ambiguity, not a claim that the full formulations are identical.

For a **finite-solver** label with threshold $t=0.1$, $z=0.2$, and $N=3$, a true exactness budget before the observed label can be masked by error $0.2>t$ on the earlier budgets, whereas a true exactness budget after the observed label can coexist with a positive candidate gap $0.1$ at the observed budget. Both constructions use the step-gap MIO in E6, with the global auxiliary binary of cost $0.2$ for delayed passes. Hence the fixed-$t$ identified set can span the entire budget range when $0<t<z$.

## E6. General realization of a nonincreasing gap profile

Let $I=\{b_0,\ldots,N\}$ and choose any finite nonincreasing profile $g_{b_0}\ge\cdots\ge g_N=0$. Introduce one binary selectable option $y_k$ for each $k\in I$ with $\sum_{k\in I}y_k=1$, and cost $\sum_k g_k y_k$. Pad with $b_0-1$ mandatory null entities to make $|C(B)|=B$, and let $C(B)$ contain options $k\le B$. Then the full optimum is 0, while the candidate optimum is $\min_{k\le B}g_k=g_B$. A global binary $w$ of cost $z$ permits feasible solver-error endpoints 0 and $z$ at every budget without changing $g$. Thus every step-profile used to prove the fixed-label identification table and Theorem B has a genuine finite 0–1 MIO realization. The construction is deliberately broad; it establishes sharpness of an information bound, not realism of a specific CFLP or portfolio structure.

## E7. Observed incumbent vector with a **nonconvex** exact identified set

Take budgets $1,2,3$, observed feasible incumbent values $u=(2,3,2)$, error cap $z=1$, and valid full lower bound $L=1$. Formula (6) of the theory note gives $s\in[s_-,s_+]=[1,2]$, $d=(2,2,1)$, and prefix upper envelope $a=(2,2,2)$. At quality tolerance $\tau=0$:

| Candidate true objective sequence $J$ | Full optimum $s$ | Solver errors $u-J$ | Exactness budget $Q_J(0)$ |
|---|---:|---|---:|
| $(2,2,2)$ | 2 | $(0,1,0)$ | 1 |
| $(2,2,1)$ | 1 | $(0,1,1)$ | 3 |

Both are nonincreasing and observationally consistent. There is **no** consistent sequence with $Q_J(0)=2$: exactness at 2 would require $J_2=s$ and $J_1>s$. But $u_2-z=2$ forces $J_2\ge2$, while $u_1=2$ forces $J_1\le2$; consequently $J_1>s=J_2$ is impossible. The sharp set is $\{1,3\}$, whereas its minimal covering interval is $[1,3]$. The generic 0–1 construction of Theorem C realizes both rows as distinct MIOs with the **same scalar observations** $(u,L,z)$, proving the minimax width lower bound $3-1=2$ for interval-valued reports. A set-valued rule can report the sharper set $\{1,3\}$.

## E8. Boundary and missing-incumbent cases

With $b_0=N$, $J_N=J^*$ and $Q(\tau)=N$ for every $\tau\ge0$; the frontier has one point and $\Delta=+\infty$. If $t<0$, no $g_B+\zeta_B\le t$ is possible. If $t=z$, even the worst error at $N$ passes by weak inequality, so $+\infty$ leaves the identified set exactly at that boundary. If $t<z$, choosing $\zeta_N=z$ can prevent even a full-budget pass. If the solver returns no feasible incumbent at $N$, model it as $\zeta_N=+\infty$; a finite uniform cap does not hold, and a forced fallback label $N$ must not be described as a certificate.
