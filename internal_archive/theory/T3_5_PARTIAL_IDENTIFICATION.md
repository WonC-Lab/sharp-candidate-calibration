# T3.5 — Sharp partial identification of minimum candidate dimension

This is a mathematical research note, not manuscript text. All sharpness assertions specify an information class. A finite interval's endpoint hull is **not** automatically its exact identified set. The standing setting is a finite nested candidate family indexed by $I=\{b_0,\ldots,N\}$, with $C(N)=U$, attained restricted optima $J_B$, full optimum $s=J_N=J^*$, and $g_B=J_B-s\ge0$ nonincreasing. Define $Q(t)=\min\{B:g_B\le t\}$ for $t\ge0$, and $Q(t)=+\infty$ for $t<0$. A first pass over an empty set is $+\infty$. All tolerances are absolute objective gaps unless explicitly noted.

## 0. Classification of the inherited T3 results

| T3 result | Mathematical type | Sharpness/depth judgment |
|---|---|---|
| $g_B\ge0$, $g_N=0$, monotonicity | STANDARD CONSEQUENCE | Exact but immediate from nested feasible sets. |
| $\tau$-optimal support-entry identity | EXACT CHARACTERIZATION | Exact converse under attainment; useful, but a short restriction equivalence. |
| $Q(\tau)=B^\dagger\iff\tau<\Delta$ | EXACT CHARACTERIZATION; SHARP RESULT | Equality boundary proved; finite first-passage arithmetic, not independently deep. |
| Oracle reference and ideal certificate translations | EXACT CHARACTERIZATION; SHARP RESULT | Both directions follow by threshold shifts. |
| Finite-solver lower/upper sandwich | ONE-WAY BOUND as originally stated; PARTIAL IDENTIFICATION RESULT after attainable-set proof | Exact set requires a specified error information class and incumbent realizability. |
| $U_B-L=g_B+\eta+\zeta_B$ and lower-bound barrier | STANDARD CONSEQUENCE | Weak duality and additive decomposition. |
| Operational-to-exact coverage | ONE-WAY BOUND; STANDARD CONSEQUENCE | Union bound; not a new conformal theorem. |
| Lattice corollary | SHARP RESULT as a sufficient lattice threshold; STANDARD CONSEQUENCE | Sharp worst-case gap $h$, but integrality itself is standard. |
| Candidate-only impossibility | SHARP within its broad class; STANDARD CONSEQUENCE | Elementary indistinguishability, not a headline. |

**Correction to T3.** The inverse-envelope inequality was written in the wrong direction in an earlier draft. For $g_{\min}\le g\le g_{\max}$, the correct order is $Q_{g_{\min}}(t)\le Q_g(t)\le Q_{g_{\max}}(t)$. The prior T3 note has been corrected; all results below use this order.

## 1. Quality–support duality and the complete finite frontier

For each full feasible solution $v$, let $S(v)$ be its active entity support and $d(v)=\operatorname{entry}_{\mathcal C}(S(v)):=\min\{B\in I:S(v)\subseteq C(B)\}$. Let $\Omega^\tau=\{v\in F:f(v)\le s+\tau\}$. Exact null-block restriction semantics and attained restricted optima give

$$Q(\tau)=\min_{v\in\Omega^\tau}d(v).\tag{1}$$

**Proof.** $g_B\le\tau$ iff an attained $v\in F(B)$ has $f(v)\le s+\tau$; $v\in F(B)$ iff $d(v)\le B$. Taking first budgets proves (1). □

Define the attainable dimension–degradation set $\mathcal A=\{(d(v),f(v)-s):v\in F\}$, with both coordinates minimized. Let $\mathcal P$ be its nondominated subset. Let $p_1=b_0<p_2<\cdots<p_m=B^\dagger$ be the first budget and every subsequent **strict** improvement budget, i.e. $g_{p_{j}}<g_{p_j-1}$ for $j>1$. If $b_0=B^\dagger$, then $m=1$.

**Theorem A (frontier/inverse equivalence).**

1. $\mathcal P=\{(p_j,g_{p_j}):j=1,\ldots,m\}$; every such point is realized by a feasible solution whose support first enters at exactly $p_j$.
2. $g_B=\min\{z:(d,z)\in\mathcal A,\ d\le B\}$, and $Q(\tau)=\min\{d:(d,z)\in\mathcal P,\ z\le\tau\}$. Thus $Q$ is the generalized inverse of the finite candidate-dimension/quality Pareto frontier.
3. The **complete set of positive breakpoints** of $Q$ is $\{g_{p_j}:j<m\}$ (possibly empty). At $\tau=g_{p_j}>0$, $Q$ jumps from $p_{j+1}$ immediately below the breakpoint to $p_j$ at the breakpoint; the jump size is $p_{j+1}-p_j$. Repeated gap values on a budget plateau are *not* separate breakpoints.

**Proof.** Every $v$ with $d(v)=B$ has $f(v)-s\ge g_B$. If $B$ lies in a plateau, its pair is dominated by the first budget of the same plateau. At a strict improvement $p_j$, an attained minimizer at $p_j$ cannot have entry $<p_j$, since that would yield the same value at an earlier budget. Its pair $(p_j,g_{p_j})$ is nondominated: every earlier budget has strictly larger gap, and every later budget has larger dimension. These observations prove item 1 and the lower-envelope identity in item 2; equation (1) yields its inverse identity. As $\tau$ increases from 0, the first eligible frontier point moves backward through the ordered list, crossing exactly each positive $g_{p_j}$; weak inequality includes the breakpoint itself, proving item 3. □

Multiple near-optimal supports may map to the same frontier point, and a plateau may instead arise because no new support improves the objective; multiplicity is neither necessary nor sufficient for a plateau. The frontier uniquely determines $Q$, and $Q$ determines the **nondominated** frontier points and their gaps via its jumps, but neither object determines all dominated solutions or support multiplicities. The theorem generalizes the exact-support case, yet is order-theoretic once (1) is known. **Status: SUPPORTING, not a defensible independent headline.**

## 2. Exact first-pass identified set under bounded solver error

Fix the full profile $g$, a threshold $t\in\mathbb R$, and a cap $z\ge0$. In the abstract bounded-error information model $\mathfrak Z=[0,z]^I$, let $R(t;\zeta)=\min\{B\in I:g_B+\zeta_B\le t\}$. An actual fixed MIO may not have feasible incumbents at every cost $J_B+\zeta_B$; the abstract result is sharp over the class of MIOs with such dominated feasible incumbents, not automatically within each fixed formulation.

**Theorem B (complete first-pass identified set).** If $t<0$, the set of possible $R$ is $\{+\infty\}$. If $0\le t<z$, it is
$$\{Q(t),Q(t)+1,\ldots,N,+\infty\}.\tag{2a}$$
If $t\ge z$, it is
$$\{Q(t),Q(t)+1,\ldots,Q(t-z)\}.\tag{2b}$$
These statements include $b_0=0$, $b_0=N$, repeated gaps, score ties after a fixed ranking rule, and nonmonotone $\zeta_B$. No gap-plateau assumption is needed.

**Proof.** If $t<0$, $g_B+\zeta_B\ge0$ everywhere. For $t\ge0$, an observed pass implies $g_B\le t$, so $R\ge Q(t)$. If $t\ge z$, $g_{Q(t-z)}+\zeta_{Q(t-z)}\le t$, so $R\le Q(t-z)$. For any integer $r$ between these bounds, set $\zeta_B=z$ for $B<r$ and $\zeta_B=0$ for $B\ge r$. Every predecessor of $r$ fails because either $B<Q(t)$ and $g_B>t$, or $B<Q(t-z)$ and $g_B>t-z$. At $r$, $g_r\le t$, so it passes. If $0\le t<z$, the same construction realizes every $r\ge Q(t)$: every predecessor with $g_B=0$ still fails since $z>t$. Taking $\zeta_B=z$ for every $B$ produces no pass, since $g_B+z>t$ everywhere. □

The extra $+\infty$ in (2a) is essential. If a feasible incumbent is **missing** at an arbitrary budget, represent its error as $+\infty$; then no uniform finite upper endpoint survives. With unrestricted missingness and $t\ge0$, the identified set expands to $\{Q(t),\ldots,N,+\infty\}$, provided other budgets may return exact incumbents. A full-budget fallback label $N$ is not a passing certificate. For a fixed MIO that restricts feasible incumbent values, (2a,b) are outer bounds; realizability of every integer is established by an independent binary global variable with objective cost $z$, which permits errors 0 and $z$ at each budget (see examples).

For a known profile, the certification residual is $U_B-L=g_B+\eta+\zeta_B$, so substitute $t=\epsilon-\eta$ into (2a,b). This gives the complete phase diagram: **Regime I** $\epsilon<\eta$ gives only $+\infty$; **Regime II** $\eta\le\epsilon<\eta+z$ gives all budgets from $Q(\epsilon-\eta)$ through $N$ plus $+\infty$; **Regime III** $\epsilon\ge\eta+z$ gives the finite interval in (2b). A different transition occurs when $t$ crosses $\Delta$: the oracle first pass changes from $B^\dagger$ to a nonexact prefix. Uniformly over all $\zeta\in[0,z]^I$,
$$R(t;\zeta)=B^\dagger\ \text{for every admissible }\zeta
\quad\Longleftrightarrow\quad z\le t<\Delta.\tag{3}$$
The lower inequality prevents solver-induced delay; the upper prevents a nonexact early pass. Both are necessary in the unrestricted error information model, with the usual $\Delta=+\infty$ convention. In particular, when $z=0$, (3) reduces to $0\le t<\Delta$.

**Novelty judgment.** The exact attainable set is stronger than the T3 sandwich. Its converse and constructive sharpness hold over a declared information model; it is not a generic claim that every actual solver can realize all costs.

## 3. Interval-valued bound or reference quality: exact sets, not just endpoints

Now retain the **known full gap profile** $g$. Let $t$ vary in a closed interval $T=[t_-,t_+]$, independently of the abstract solver errors $\zeta_B\in[0,z]$. For a finite budget $r>b_0$, define $g_{r-1}$ as the preceding gap. The exact attainable set is

$$\mathcal I_R(T,z)=\left\{r\in I:\exists t\in T\cap[0,\infty),\quad g_r\le t<g_{r-1}+z\right\}\cup\left\{b_0:\exists t\in T,\ t\ge g_{b_0}\right\}\cup\left\{+\infty:\exists t\in T,\ t<z\right\}.\tag{4}$$

The $+\infty$ term also covers $t<0$. To prove the finite-budget criterion, choose $\zeta_r=0$ to make $r$ pass and $\zeta_B=z$ at every earlier $B$; monotonicity means the smallest preceding gap is $g_{r-1}$. Conversely, if $r$ is first, $g_r\le t$ and $g_{r-1}+z>t$. For $r=b_0$ there is no predecessor. The no-pass criterion follows from $g_N=0$ and choosing all errors $z$.

For **oracle reference fidelity**, set $z=0$ and $T_{\rm ref}=[\epsilon_{\rm ref}+\delta_L,\epsilon_{\rm ref}+\delta_U]$, with $0\le\delta_L\le\delta_U$. The exact identified set is (4) without $+\infty$; its endpoint hull is
$$\big[Q(\epsilon_{\rm ref}+\delta_U),\ Q(\epsilon_{\rm ref}+\delta_L)\big]\cap\mathbb Z,\tag{5}$$
but the set can have holes. For example $g=(2,2,0)$ and $T_{\rm ref}=[0,2]$ give labels $\{1,3\}$, never 2. The endpoint hull is therefore not itself an exact identified set. For **finite-incumbent reference fidelity**, keep $z>0$ and the same $T_{\rm ref}$ in (4).

For **certification**, set $T_{\rm cert}=[\epsilon-\eta_U,\epsilon-\eta_L]$, where $0\le\eta_L\le\eta_U$. Formula (4) is its exact attainable set. If $\epsilon<\eta_L$, all $t<0$ and certification is impossible. If $\epsilon\ge\eta_U+z$, every allowed $t\ge z$ and all outcomes are finite; the smallest and largest attainable budgets are $Q(\epsilon-\eta_L)$ and $Q(\epsilon-\eta_U-z)$, respectively, yet intermediate budgets can be absent when $z=0$ or due to a plateau. If $\epsilon<\eta_U+z$, $+\infty$ is attainable. A **guaranteed** certificate at some budget for all allowed $\eta,\zeta$ exists iff $\epsilon\ge\eta_U+z$. This is a worst-case necessary-and-sufficient statement, not a prediction of a particular solver run.

The $\eta$ and $\delta$ intervals are sensitivity parameters **conditional on a known $g$**. In an actual deployment $g$ and $J^*$ are often unknown, so the interval parameters are coupled to unknown objectives; independently varying $g$ and $\eta$ may overstate physical possibilities. Section 5 treats a joint observation model.

## 4. Proxy label alone and point identification of exactness

If $g$ is already known, $B^\dagger=Q(0)$ is trivially known; asking whether a proxy label identifies it would be vacuous. Here the observer knows the first-pass label $r$, tolerance information, and error cap, but **not** the gap profile. Work over all nonincreasing nonnegative finite profiles with $g_N=0$. For a fixed effective threshold $t\ge0$ and a finite observed first-pass $r$, the exact possible values of $B^\dagger$ are:

| Error regime | $t=0$ | $t>0$ |
|---|---|---|
| $z=0$ | $\{r\}$ | $\{r,r+1,\ldots,N\}$ |
| $0<z\le t$ | impossible at $t=0$ | $\{r,r+1,\ldots,N\}$ |
| $z>t$ | $\{b_0,\ldots,r\}$ | $\{b_0,\ldots,N\}$ |

The row $z=0,t=0$ is the exact-oracle label. The table follows from two facts: a pass at $r$ means $g_r\le t$; an exact predecessor $B^\dagger<r$ would necessarily pass whenever $z\le t$. Conversely all permitted values can be realized by finite decreasing step profiles: place a positive gap at budgets before the desired exactness index, place it at most $t$ from $r$ onward when exactness is later, and use error $z$ on earlier budgets to delay passes when $z>t$. Explicit 0–1 MIO realizations are in the sharpness file. An uncertain threshold interval yields the **union** of these fixed-$t$ sets over $t\in T\cap[0,\infty)$; do not replace the union by a convenient sufficient condition.

For an **oracle reference label**, $z=0$ and $t=\epsilon_{\rm ref}+\delta$. It point-identifies $B^\dagger$ at $r$ for every admissible $\delta$ iff $r=N$ or the admissible threshold set contains only $0$; otherwise the identified set is $\{r,\ldots,N\}$. If an objective lattice of spacing $h$ is known, replace “only $0$” by $t_+<h$ (because no positive gap below $h$ exists), again with $r=N$ as a boundary exception. Without a lattice or a lower bound on $\Delta$, an arbitrarily small positive tolerance permits nonexact early passage.

For a **finite-solver operational label**, the fixed-$t$ table is the exact general-profile identified set in the abstract error model; for a known interval of $\eta$ or $\delta$, take its union over $t$. If only a **missing-pass fallback label** $N$ is observed, rather than a genuine pass at $N$, the table does not apply: the fallback carries no certificate or fidelity event, and $B^\dagger$ can be anywhere in $I$ absent additional information.

## 5. Joint identification from observed incumbent values and a valid lower bound

This section gives the non-vacuous deployment information model. Observe a feasible-incumbent objective $u_B$ at **every** budget $B\in I$, a valid full-universe lower bound $L$, and a uniform unknown candidate-solver error $0\le u_B-J_B\le z<+\infty$. Do **not** assume $J^*$, $\eta$, $\delta$, or $g$ known. The observation consists only of these scalar values, not the entire full formulation; otherwise an unlimited solver could recover $J^*$ and the information problem would change. Assume the data are consistent with at least one nonincreasing objective sequence $J_B$ and $J_N=s\ge L$.

Define
$$s_-:=\max\{L,u_N-z\},\qquad s_+:=\min_{B\in I}u_B,\qquad
d_B:=\max_{j\in I:j\ge B}(u_j-z),\qquad a_B:=\min_{i\in I:i\le B}u_i.\tag{6}$$
Consistency entails $s_-\le s_+$ and $d_B\le a_B$ for all $B<N$. For any $s\in[s_-,s_+]$, set
$$m_B(s)=\max\{s,d_B\},\qquad
M_B(s)=\begin{cases}a_B,&B<N,\\s,&B=N.\end{cases}\tag{7}$$
Then $m(s)$ and $M(s)$ are nonincreasing objective sequences satisfying $u_B-z\le J_B\le u_B$, $J_N=s$, and $L\le s$. Every consistent sequence lies pointwise between them. The proof is the suffix lower-envelope and prefix upper-envelope inequalities; consistency ensures the envelopes also satisfy the opposite per-budget bounds. Define $Q_J(\tau)=\min\{B:J_B-s\le\tau\}$ for $\tau\ge0$.

**Theorem C (sharp scalar-observation identified set).** A budget $r=b_0$ belongs to the exact identified set for $Q_J(\tau)$ iff
$$[s_-,s_+]\cap[d_{b_0}-\tau,\infty)\ne\varnothing.\tag{8a}$$
For $r>b_0$, it belongs iff
$$[s_-,s_+]\cap[d_r-\tau,\ a_{r-1}-\tau)\ne\varnothing.\tag{8b}$$
The strict upper endpoint is essential: the predecessor must fail its weak quality threshold. The smallest and largest attainable budgets are respectively
$$q_{\min}=Q_{m(s_+)}(\tau),\qquad q_{\max}=Q_{M(s_-)}(\tau),\tag{9}$$
where $Q_{m(s)}$ and $Q_{M(s)}$ use baseline $s$. Equations (8), not merely the hull $[q_{\min},q_{\max}]\cap\mathbb Z$, define the sharp set.

**Proof.** Fix $s$. The earliest possible first pass is achieved by $m(s)$ and the latest by $M(s)$. A first pass exactly at $r>b_0$ is feasible iff some consistent profile has $J_{r-1}>s+\tau$ and $J_r\le s+\tau$. Necessity gives $M_{r-1}(s)=a_{r-1}>s+\tau$ and $m_r(s)=\max(s,d_r)\le s+\tau$, equivalent to (8b). For sufficiency choose $J_B=M_B(s)$ before $r$ and $J_B=m_B(s)$ from $r$ onward; the splice is nonincreasing because $M_{r-1}(s)\ge m_r(s)$, and it satisfies all observational intervals. The $r=b_0$ case has only the pass condition. As $s$ increases, both envelopes' gaps relative to $s$ weakly decrease, giving the extremal budgets in (9). Each envelope is itself feasible, proving endpoint sharpness. □

**MIO realizability and minimax corollary.** For every observationally consistent monotone sequence $J_B=j_B$ and $s=j_N$, construct a finite 0–1 MILP with one selectable option $k$ for each budget $k\in I$, $\sum_{k\in I}y_k=1$, and an auxiliary binary $w_k\le y_k$. Give its objective $\sum_k[j_k y_k+(u_k-j_k)w_k]$. Pad with $b_0-1$ mandatory null entities if necessary so that the top-$B$ family has cardinality $B$. At budget $B$, only options $k\le B$ are available, hence $J(B)=\min_{k\le B}j_k=j_B$; choosing $k=B$ and $w_B=1$ gives feasible incumbent value $u_B$. All constructed MILPs produce **identical scalar observations** $(u_B,L,z)$, while profiles at the two endpoints of (9) attain $q_{\min}$ and $q_{\max}$. Therefore any interval-valued rule that sees only $(u_B,L,z)$ and is uniformly valid over this class must have width at least $q_{\max}-q_{\min}$; $[q_{\min},q_{\max}]$ achieves that minimum width. A set-valued rule can report the possibly smaller exact set (8). This is a deterministic information lower bound, not a runtime lower bound or a statement about an observer given full model coefficients.

If $u_N$ is absent, $s$ may lack a finite upper bound and these formulas must be rederived; one must not use (9) with a missing full-budget incumbent. If only the valid lower bound and $z$ are known, without the vector $u_B$, $Q$ is generally not nontrivially identified.

## 6. Exactness and lattice specialization

The separation identity remains $Q(\tau)=B^\dagger\iff0\le\tau<\Delta$, with $\Delta=+\infty$ if $B^\dagger=b_0$. For a known gap profile and unknown error $z$, equation (3) is the **necessary-and-sufficient uniform operational equality** criterion. With interval reference quality $\delta\in[\delta_L,\delta_U]$, it becomes
$$z\le\epsilon_{\rm ref}+\delta_L\quad\text{and}\quad\epsilon_{\rm ref}+\delta_U<\Delta.\tag{10}$$
With interval lower-bound looseness $\eta\in[\eta_L,\eta_U]$, it becomes
$$\eta_U+z\le\epsilon_{\rm cert}\quad\text{and}\quad\epsilon_{\rm cert}-\eta_L<\Delta.\tag{11}$$
These are both necessary and sufficient for the observed first-pass budget to equal $B^\dagger$ **for every independently admissible parameter/error configuration** in the known-profile information model. They need not be necessary for one realized solver run, and physically coupled uncertainty can shrink the class.

If every candidate optimum and $s$ lie on a common lattice $a+h\mathbb Z$, then $\Delta\ge h$ whenever $\Delta<+\infty$. Thus replacing $\Delta$ by $h$ on the right of (10)–(11) gives **uniform sufficient** point-identification regions. For unit lattice,
$$z\le\epsilon_{\rm ref}+\delta_L,\quad \epsilon_{\rm ref}+\delta_U<1;\qquad
\eta_U+z\le\epsilon_{\rm cert},\quad\epsilon_{\rm cert}-\eta_L<1.\tag{12}$$
These are worst-case sharp over unit-lattice gap profiles. The proposed shortcut $\epsilon_{\rm ref}+\delta_U+z<1$ is **not sufficient** for an operational label to identify exactness: it prevents an early nonexact pass but does not prevent solver error from delaying the first pass. The error cap belongs on the **lower effective-tolerance** side, $z\le\epsilon_{\rm ref}+\delta_L$, not added to the upper separation side. Conversely, a particular run may have $\zeta_{B^\dagger}=0$ and identify exactness even when the uniform condition fails.

For a finite observed label $r$ under an unknown unit-lattice profile, the fixed-$t$ table in Section 4 sharpens: if $0\le t<1$, a pass at $r$ forces $B^\dagger\le r$; if also $z\le t$, exactness is point-identified at $r$. If $t\ge1$, a nonexact prefix with gap 1 may pass. This is mathematical objective lattice reasoning; a floating-point solver status is not an exact-integrality certificate.

## 7. Interpretation and limitations

Theorem A is a complete frontier representation but is mainly generalized-inverse order theory. Theorem B adds an exact attainable set for operational first-pass budgets, including $+\infty$. Theorem C is the strongest new candidate: it jointly treats unknown full optimum, incumbent errors, a valid lower bound, monotone restricted objectives, sharp membership of every budget, and a scalar-observation minimax lower bound. Its relevance depends on a **restricted information model**. If the entire full-universe MIO formulation or exact $J(N)$ is available, the information set changes and the same minimax claim is inappropriate. No theorem here establishes a new fast algorithm for computing $Q$ or $\Delta$ from a full MIO.
