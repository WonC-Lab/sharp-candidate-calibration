# Phase T3: a sharp theory of candidate-dimension quality

**Status:** mathematical development only. No claim below has been inserted into `manuscript/master.tex`; no new experiment has been run. `HEADLINE` denotes a possible organizing result, not an established priority claim or an assertion of top-journal depth. T2 is a hard novelty constraint: conformal MIO quality, early stopping, learned reduction, safe screening, and warm starts are prior art.

## 1. Model and primitive [STANDARD]

Fix an instance $\theta$, a finite entity universe $U$ of size $N$, and a nested family $C_\theta(B)$ for consecutive integer budgets $B\in\{b_0,\ldots,N\}$, where $C_\theta(N)=U$. Excluded entity blocks are fixed to a null state. Let $F_\theta(B)$ be precisely the full feasible points whose active support is contained in $C_\theta(B)$. Assume the full minimum $J^*_\theta$ is finite and attained, $b_0=B_{\min,\mathcal C}(\theta)$ is the first feasible budget, and every feasible restricted minimum $J_\theta(B)$ is attained. Nesting implies feasibility at all $B\ge b_0$. Set

$$g_\theta(B)=J_\theta(B)-J^*_\theta,\qquad Q_\mathcal C(\theta;\tau)=\min\{B\in\{b_0,\ldots,N\}:g_\theta(B)\le\tau\}\quad(\tau\ge0).$$

For algebra with negative effective tolerances, extend $Q_\mathcal C(\theta;t)=+\infty$ for $t<0$. A minimum over an empty operational pass set is also $+\infty$; replacing it by $N$ is an explicit *fallback label*, not a successful certificate. Since $F(B)\subseteq F(B+1)\subseteq F(N)=F$, $g(B)\ge0$, $g(N)=0$, and $g$ is nonincreasing. These are standard consequences of nesting, not claimed novelties. Write $B^\dagger=Q(0)$.

The attainment and exact null-block restriction assumptions are **technical only for the support-entry converse**; nestedness and $C(N)=U$ are **sufficient and structurally necessary for the stated monotone/full-budget conclusions**, but particular conclusions can hold accidentally without them. Feasible-prefix convention is **technical only**; alternatively set $J(B)=+\infty$ before $b_0$.

## 2. Near-optimal support entry [SUPPORTING]

For $\tau\ge0$, let $\Omega_\theta^\tau=\{v\in F_\theta:f_\theta(v)\le J^*_\theta+\tau\}$ and $\mathcal S_\tau(\theta)=\{S(v):v\in\Omega_\theta^\tau\}$.

**Theorem 1 (exact support-entry characterization).** Under the preceding assumptions,
$$Q_\mathcal C(\theta;\tau)=\min_{S\in\mathcal S_\tau(\theta)}\min\{B\in\{b_0,\ldots,N\}:S\subseteq C_\theta(B)\}.\tag{1}$$

**Proof.** For any feasible $B$, attainment and the restriction semantics give the equivalence
$$g(B)\le\tau\ \Longleftrightarrow\ \exists v\in F(B):f(v)\le J^*+\tau\ \Longleftrightarrow\ \exists v\in\Omega^\tau:S(v)\subseteq C(B).$$
Taking the first budget on both sides yields (1). The full optimum makes $\Omega^\tau$ nonempty, and $C(N)=U$ ensures every support eventually enters. □

For a deterministic ranking of nonmandatory entities $U\setminus M$, let $r(i)$ be the one-based rank, $C(B)=M\cup\{i:r(i)\le B-|M|\}$, and $d_\tau=\min_{S\in\mathcal S_\tau}\max_{i\in S\setminus M}r(i)$, with $\max\varnothing=0$. Then
$$Q(\tau)=\max\{b_0,|M|+d_\tau\}.\tag{2}$$
No uniqueness of a global optimum or $\tau$-optimal support is required. Multiple points with the same support but different objectives do not cause an error: $\mathcal S_\tau$ contains that support if **at least one** representative is $\tau$-optimal. At $\tau=0$, (1) recovers the existing exact support-entry result; above $\max_B g(B)$, $Q(\tau)=b_0$. Tie scores need a fixed deterministic ranking. Formula (2) follows directly from the first entry time of each support.

**Depth audit.** This is a useful and general exact characterization, but the proof is a three-step attainment equivalence. It subsumes the old exact theorem and applies beyond CFLP/portfolio, yet it is a routine tolerance extension rather than a demonstrably deep new optimization theorem. Keep as the structural center, but do not present it as one of two independent breakthroughs without a stronger consequence.

## 3. Separation margin and sharp equivalence [SUPPORTING]

Define $\Delta=\min_{b_0\le B<B^\dagger}g(B)$ if $B^\dagger>b_0$, and $\Delta=+\infty$ otherwise. In the first case, $\Delta=g(B^\dagger-1)>0$ by finite nesting and the definition of the first exact budget. In the second, $Q(\tau)=b_0$ for every $\tau\ge0$.

**Proposition 2 (sharp first transition).** For every $\tau\ge0$,
$$Q(\tau)=B^\dagger\quad\Longleftrightarrow\quad \tau<\Delta.\tag{3}$$

**Proof.** If $\tau<\Delta$, no earlier feasible budget has gap at most $\tau$, while $B^\dagger$ does. Conversely, if $\Delta<+\infty$ and $\tau\ge\Delta$, budget $B^\dagger-1$ passes, so $Q(\tau)<B^\dagger$. For $\Delta=+\infty$, $B^\dagger=b_0$ and every finite $\tau\ge0$ gives $Q(\tau)=b_0$. □

The condition $0\le\tau<\Delta$ is **both necessary and sufficient** within this finite attained model, not merely sufficient. The strict boundary is unavoidable. $Q$ is a nonincreasing, right-continuous step function with changes among its finite gap values; this is **STANDARD** finite first-passage structure.

## 4. Quality-spectrum translation phase diagram [HEADLINE candidate, integrated result]

Let a feasible full reference have $J_{\rm ref}=J^*+\delta$, $\delta\ge0$, and define its **oracle** first-pass budget $R_{\rm ref}=\min\{B:J(B)-J_{\rm ref}\le\epsilon_{\rm ref}\}$ for $\epsilon_{\rm ref}\ge0$. Let a valid full-universe lower bound be $L=J^*-\eta$, $\eta\ge0$, and define the **ideal exact-candidate** certificate budget $R_{\rm cert}=\min\{B:J(B)-L\le\epsilon_{\rm cert}\}$.

**Theorem 3 (exact translations and phase transitions).**
$$R_{\rm ref}=Q(\epsilon_{\rm ref}+\delta),\qquad R_{\rm cert}=Q(\epsilon_{\rm cert}-\eta).\tag{4}$$
Consequently,
$$R_{\rm ref}=B^\dagger\quad\Longleftrightarrow\quad\epsilon_{\rm ref}+\delta<\Delta,\tag{5}$$
and
$$R_{\rm cert}=B^\dagger\quad\Longleftrightarrow\quad\eta\le\epsilon_{\rm cert}<\eta+\Delta.\tag{6}$$
If $\epsilon_{\rm cert}<\eta$, $R_{\rm cert}=+\infty$ and no feasible incumbent, even a global optimum, can pass. At $\epsilon_{\rm cert}=\eta$, $R_{\rm cert}=B^\dagger$. At $\epsilon_{\rm cert}=\eta+\Delta<+\infty$, a nonexact prefix passes and $R_{\rm cert}<B^\dagger$.

**Proof.** $J(B)-J_{\rm ref}=g(B)-\delta$ and $J(B)-L=g(B)+\eta$. Their threshold inequalities are exactly $g(B)\le\epsilon_{\rm ref}+\delta$ and $g(B)\le\epsilon_{\rm cert}-\eta$. Equation (4) follows, including the $+\infty$ convention for negative effective tolerance. Equation (3) gives both directions of (5) and (6), with nonnegative effective tolerance in (6). For any feasible incumbent $U\ge J^*$, $U-L\ge\eta$, proving the barrier. □

Validity of $L$ and feasibility of the reference are **necessary for the nonnegative-error interpretation**, although the algebraic equality could be reparameterized even for invalid comparators. The strict upper thresholds in (5)–(6) are **both necessary and sufficient**. These formulas give an exact phase diagram and resolve the manuscript's separate-budget ambiguity, but their proof is elementary algebra plus Proposition 2. Novelty lies, at most, in identifying the shared candidate-quality spectrum as the primitive and using the phase diagram to distinguish structural approximation from certification information.

## 5. Finite-solver partial identification [HEADLINE candidate]

Let $U_B=J(B)+\zeta_B$ be a feasible candidate incumbent, with $0\le\zeta_B\le\bar\zeta$ for every feasible budget. For effective tolerance $t$ define
$$\widetilde Q(t)=\min\{B:g(B)+\zeta_B\le t\}.\tag{7}$$
The observed reference and certificate budgets are respectively $\widetilde Q(\epsilon_{\rm ref}+\delta)$ and $\widetilde Q(\epsilon_{\rm cert}-\eta)$, provided all needed incumbents exist. In particular,
$$U_B-L=g(B)+\eta+\zeta_B.\tag{8}$$

**Theorem 4 (sharp identified set under a bounded-error information model).** Treat the error sequence as unknown apart from $0\le\zeta_B\le\bar\zeta$; a particular MIO may restrict which error values correspond to actual feasible incumbents. If $t\ge\bar\zeta$, set $q_-=Q(t)$ and $q_+=Q(t-\bar\zeta)$. Then
$$q_-\le\widetilde Q(t)\le q_+,\tag{9}$$
and, in that bounded-error information model, **every integer** $k\in\{q_-,\ldots,q_+\}$ is attained by some admissible $\{\zeta_B\}$. In particular the endpoints are sharp; no gap-plateau assumption is needed. These are also sharp **over a class of MIOs**: one may add an independent feasible binary auxiliary variable of cost $\bar\zeta$, allowing both error endpoints at every candidate budget. For a fixed MIO without such dominated incumbents, only a subset of integers may be physically attainable; (9) remains valid. If $t<\bar\zeta$, the right endpoint $Q(t-\bar\zeta)=+\infty$ is only a vacuous bound, and an operational pass need not exist even at $N$.

**Proof.** An observed pass implies $g(B)\le t$, hence $q_-\le\widetilde Q(t)$. At $q_+$, $g(q_+)\le t-\bar\zeta$, so every allowable error passes and $\widetilde Q(t)\le q_+$. To realize an arbitrary $k$ between them, set $\zeta_B=\bar\zeta$ for $B<k$ and $\zeta_B=0$ for $B\ge k$. For $B<q_-$, $g(B)>t$; for $q_-\le B<k\le q_+$, $B<q_+$ implies $g(B)>t-\bar\zeta$. Thus no $B<k$ passes. Since $k\ge q_-$ and $g$ is nonincreasing, $g(k)\le t$ and $k$ passes. □

For certification with $t=\epsilon_{\rm cert}-\eta\ge\bar\zeta$ the **exact** possible observed-budget interval is
$$\left[Q(\epsilon_{\rm cert}-\eta),\ Q(\epsilon_{\rm cert}-\eta-\bar\zeta)\right]\cap\mathbb Z.\tag{10}$$
Thus improving the ranking changes $g$, strengthening the global lower bound reduces $\eta$, and improving candidate solves reduces $\bar\zeta$; they are not interchangeable. The uniform error assumption is **sufficient** for (9), not necessary for a particular realization; nonnegativity of error is **necessary** for the universal lower inequality. $t\ge\bar\zeta$ is **sufficient** for finite universal upper identification and is worst-case necessary because $g(N)=0$ and $\zeta_N$ may equal $\bar\zeta$.

**Inverse identification when the true optimum is known.** The previous theorem fixes $g$ and varies solver error. Suppose instead the numerical incumbent residuals $h_B=U_B-J^*$ are observed and $J^*$ is independently known, but $g$ is unknown. Work over the information class of monotone gap profiles, not a fixed fully known MIO formulation. Assume the observations are consistent with at least one nonincreasing $g$ with $g(N)=0$ and $0\le h_B-g(B)\le\bar\zeta$. Define
$$g_{\min}(B)=\max_{j\ge B}(h_j-\bar\zeta)_+,\qquad
g_{\max}(B)=\begin{cases}\min_{i\le B}h_i,&B<N,\\0,&B=N.\end{cases}\tag{11}$$
Both profiles are feasible **within that monotone-profile information class**, nonincreasing, and pointwise bracket every consistent $g$. Indeed any feasible $g(B)\ge g(j)\ge h_j-\bar\zeta$ for $j\ge B$, and $g(B)\le g(i)\le h_i$ for $i\le B$. Consistency supplies the remaining per-budget bounds for the two envelopes. Hence the possible true quality budget has sharp endpoints
$$Q_{g_{\min}}(t)\le Q_g(t)\le Q_{g_{\max}}(t).\tag{12}$$
The endpoints are attained by the two envelope profiles in the information class; a fixed MIO's feasibility structure may narrow them, and (12) does **not** assert that every intermediate integer is attainable without further checking. If only a lower bound $L$ is known, rather than $J^*$, the residuals $h_B$ are themselves unknown and (11) cannot be computed. A lower bound plus $\bar\zeta$ alone does **not** identify $Q$; additional full-optimum information or a tight interval for $J^*$ is required. This qualification is essential to the question's “given only” wording.

**Depth audit.** The exact attainable set and monotonicity-adjusted inverse envelopes are stronger than a two-sided inequality and may have utility outside the application. Still, the construction is elementary order geometry. This is the strongest formal candidate in T3, but its originality relative to general partial-identification/isotonic-bound literature is **unverified**.

## 6. Proxy-to-exact nonidentification [SUPPORTING]

Equations (3) and (5) imply that below $\Delta$ the oracle proxy budget equals $B^\dagger$, while at or above $\Delta$ it **strictly precedes** $B^\dagger$ whenever a feasible nonexact predecessor exists. This is an exact dichotomy, not merely a “may” statement, for the fixed instance and oracle proxy. Yet the observed proxy *label alone* does not reveal which side of the divide holds, because it does not reveal $\Delta$ or $J^*$. The complete two-instance construction is in `T3_COUNTEREXAMPLES.md`. If finite solver errors are present, crossing $\Delta$ does **not** force the *observed* first pass earlier: a large $\zeta_B$ can mask the nonexact oracle pass. Candidate-only versus full-information distinctions must therefore remain explicit.

## 7. Candidate-only certification impossibility [STANDARD / not headline]

**Proposition 5 (restricted-information minimax obstruction).** Fix a proper candidate $C$ and an instance class containing, for every $K>0$, the pair of two-binary MILPs constructed in `T3_COUNTEREXAMPLES.md`. Their complete candidate-restricted formulations and all candidate-generated data are identical, but the candidate optimum's full gaps are $0$ and $K$. A sound rule whose output depends only on candidate-generated data must report a gap upper bound at least $K$ for the second member. As $K$ is unbounded, no **finite** uniform upper certificate is sound over the class. On a class capped at $K_0$, no candidate-only bound strictly less than $K_0$ is uniformly sound for that pair.

**Proof.** Indistinguishable restricted observations force the same output on both instances. Soundness on the second requires an upper bound at least its gap $K$, for each $K$ in the class. □

“Uniform” means valid for every instance in the stated class; “nontrivial” means finite for the unbounded class or below the known worst-case $K_0$ for a bounded class. This does not exclude randomized *distributional* guarantees, nor certification with valid global bounds, safe screening, or restrictions on excluded entities. A targeted primary-source search found no identical theorem statement, but generic no-free-lunch/indistinguishability reasoning is established; see [Igel–Toussaint's optimization no-free-lunch discussion](https://arxiv.org/abs/cs/0303032). The obstruction is **not** marketed as novel or deep.

## 8. Operational-to-exact coverage [SUPPORTING; standard probability step]

Suppose an exchangeable operational label $R(\theta)$ and a correctly constructed split-conformal upper budget $\widehat B_\alpha$ satisfy $\Pr\{R_{\rm new}\le\widehat B_\alpha\}\ge1-\alpha$. Let $E$ imply $B^\dagger_{\rm new}\le R_{\rm new}$ and satisfy $\Pr(E)\ge1-\beta$. Then
$$\Pr\{B^\dagger_{\rm new}\le\widehat B_\alpha\}\ge1-\alpha-\beta,\qquad
\Pr\{J(C(\widehat B_\alpha))=J^*\}\ge1-\alpha-\beta.\tag{13}$$
**Proof.** On $E\cap\{R_{\rm new}\le\widehat B_\alpha\}$, nesting yields $B^\dagger_{\rm new}\le\widehat B_\alpha$; the union bound controls the complement. □

For oracle reference labels, $E=\{\epsilon_{\rm ref}+\delta<\Delta\}$ gives equality of proxy and exact budgets. For finite-incumbent reference labels, $\widetilde Q(t)\ge Q(t)$, so the same event is sufficient for conservativeness. If no budget passes, setting $R=N$ is an exactness-conservative *fallback label*, not a reference-fidelity claim. A valid certified-pass label is conservative for exactness when $\epsilon_{\rm cert}-\eta<\Delta$; if no pass, use the same explicit fallback convention. Exchangeability and calibration independence are **sufficient** for the quoted conformal premise, not necessary for every possible procedure. The condition $E$ is **sufficient** for (13), not necessary in general. The union bound is **STANDARD**; any innovation lies in the deterministic separation/translation condition, not in conformal prediction itself. No conditional-on-instance coverage is asserted.

## 9. Lattice and optional tools

**Corollary [SUPPORTING].** If all feasible $J(B)$ and $J^*$ lie in one lattice $a+h\mathbb Z$ with $h>0$, every positive candidate gap is at least $h$, so $\Delta\ge h$ or $+\infty$. Thus $Q(\tau)=B^\dagger$ for $0\le\tau<h$; $R_{\rm ref}=B^\dagger$ if $\epsilon_{\rm ref}+\delta<h$; $R_{\rm cert}=B^\dagger$ if $\eta\le\epsilon_{\rm cert}<\eta+h$. With finite solver error, $\widetilde Q(t)=B^\dagger$ is guaranteed if $0\le t<h$ **and** $\zeta_{B^\dagger}\le t$; $t\ge\bar\zeta$ is a stronger uniform sufficient condition. If $t<h$ but error prevents passing at $B^\dagger$, the observed label can be later or absent. Unit-lattice CFLP is an instance only after proving the continuous transportation subproblem has an integral optimal extreme point and objective coefficients are integral. These mathematical statements do not themselves validate numerical MIP tolerances.

**Ranking perturbation [STANDARD].** To guarantee *every* top-ranked prefix through $B_0$, it is enough, and in a worst-case tie-free perturbation sense necessary, that each original adjacent optional-score gap at boundaries $1,\ldots,B_0-|M|$ exceed $2\varepsilon$. For stability of a *single* budget $k$, only the boundary gap at $k-|M|$ need exceed $2\varepsilon$. The induced $Q(\tau)$ is stable if its critical prefix and all prior-prefix membership relevant to its first-entry proof remain fixed. This is a one-line ordering observation; exclude as a theorem.

**Simultaneous finite-grid frontier [STANDARD].** For a prespecified finite $T$, calibrating the scalar max residual $R_j=\max_{\tau\in T}[Q_j(\tau)-\hat b(X_j,\tau)]$ with the conservative split-conformal order statistic yields simultaneous marginal coverage for all $\tau\in T$. This is standard max-score conformal logic, not a new conformal principle. Exclude from the headline spine.

## 10. Necessary-versus-sufficient audit

| Condition | Classification | Precise role |
|---|---|---|
| $0\le\tau<\Delta$ for $Q(\tau)=B^\dagger$ | **BOTH** | Exact finite-budget equivalence (3). |
| $\epsilon_{\rm ref}+\delta<\Delta$ for oracle reference/exact equality | **BOTH** | Exact equivalence (5), feasible reference assumed. |
| $\eta\le\epsilon_{\rm cert}<\eta+\Delta$ for ideal certificate/exact equality | **BOTH** | Exact equivalence (6), valid full bound assumed. |
| $\eta>\epsilon_{\rm cert}$ for fixed-bound impossibility | **BOTH** for impossibility of certifying a true global optimum against this fixed bound | At equality, the optimum passes. Some nonoptimal incumbents may still fail when $\eta\le\epsilon_{\rm cert}$. |
| Attained restricted optima | **TECHNICAL ONLY** for (1) | Without attainment at $\tau=0$, an infimum can pass without any attaining support. |
| $0\le\zeta_B\le\bar\zeta$ uniformly | **SUFFICIENT** for (9), sharp in the worst case | Per-budget weaker bounds yield budget-specific envelopes. |
| $t\ge\bar\zeta$ | **BOTH** for a finite universal upper budget over all allowable errors | Not necessary for a particular observed solve. |
| Separation event $E$ of probability $1-\beta$ | **SUFFICIENT**, not necessary | Enables marginal exactness transfer via a union bound. |
| Objective lattice spacing $h$ | **SUFFICIENT**, not necessary | Guarantees $\Delta\ge h$; nonlattice data can also have a positive $\Delta$. |
| Candidate-only information in Proposition 5 | **NECESSARY** for that obstruction's indistinguishability | Any informative full-universe datum can distinguish the constructed pair. |

## 11. T3 verdict and acceptance test

The requested headline gate requires at least three of nine properties. The integrated translation result meets properties 3 (exact characterization), 4 (converse), 6 (resolves the proxy/exactness gap), 7 (subsumes prior separate budget claims), and 8 (applies beyond the two examples); property 9 remains a **bounded T2 comparison**, not a verified priority result. The finite-solver result meets 1 (not definition-only), 3 (exact attainable set), 4 (inverse/end-point converse), 5 (sharp constructions), 6 (explains observed operational inflation), and 8 (applies to any nested restricted optimization); its relation to broader isotonic partial-identification literature remains unresolved. The support-entry identity meets 3, 7, and 8 formally, but fails the **substantive depth judgment** because its proof is an immediate attainment equivalence. The numeric gate is therefore necessary, not sufficient, for promotion.

| Result | Status | Novelty level | Proof complete? | Converse? | Sharp? | Closest literature threat | Keep in headline theory? |
|---|---|---|---|---|---|---|---|
| $\tau$-optimal support entry | SUPPORTING (initial HEADLINE candidate) | Integrated specialization, originality unproven | Yes | Yes | Exact equality | Learned reduction, predict-and-search | Structural spine, but not sold as deep theorem |
| Separation-margin equivalence | SUPPORTING | Elementary finite first passage | Yes | Yes | Equality boundary | General nested optimization | Yes as lemma/proposition |
| Quality-spectrum translation phase diagram | HEADLINE candidate | Potentially new integration; algebra elementary | Yes | Yes | Exact thresholds | Clarke–Stellato 2025/26; Wada 2026 bounds | Yes, with restrained novelty language |
| Finite-solver identified set and inverse envelopes | HEADLINE candidate | Strongest technical result; priority unverified | Yes | Yes for attainable set | Every forward integer and inverse endpoints attained | General partial identification / isotonic bounds | Yes, conditional on further literature check |
| Candidate-only impossibility | STANDARD | Generic indistinguishability | Yes | Minimax obstruction only | No-free-lunch / safe screening | No; brief motivation only |
| Coverage transfer | SUPPORTING | Standard union bound using new deterministic condition | Yes | No | $1-\alpha-\beta$ worst-case sharp | Conformal MIO quality papers | No |
| Integer-lattice corollary | SUPPORTING | Standard discreteness | Yes | No | Unit boundary | Integer programming integrality | No |
| Ranking stability | STANDARD | Ordering fact | Yes | Worst-case gap threshold | $2\varepsilon$ | Ranking robustness | No |
| Simultaneous frontier calibration | STANDARD | Max-score conformal | Yes | No | Standard rank sharpness | Conformal prediction | No |

**Q1.** Not yet. There are two *plausible organizing headline candidates*, but neither is established as a genuinely new, nontrivial optimization-theory result after specialist comparison; the translation proof is mostly algebra, and the partial-identification envelopes may be a known order-theoretic device.

**Q2.** The finite-solver identified-set theorem is strongest: it determines *every attainable first-pass budget*, not just an inequality, and supplies sharp inverse envelopes when $J^*$ is known.

**Q3.** Monotonicity, ranking stability, and candidate-only impossibility are too elementary to headline. The conjecture that crossing $\Delta$ forces an *observed finite-solver* proxy to precede exactness is false when solver error masks the pass. The claim that a lower bound plus an error cap alone identifies $Q$ is false without full-optimum information.

**Q4.** No firm top-optimization/OR novelty moat yet. The integrated object is distinct from the closest T2 papers, but distinctness is not the same as mathematical depth.

**Q5.** A deeper theorem must exploit additional MIO structure to give a nontrivial, computable characterization or bound on $\Delta$/$Q$ without solving every reduced problem or knowing $J^*$; alternatively, establish a genuinely new minimax/instance-dependent identification frontier under realistic global-bound information. The present T3 results do not close that gap.
