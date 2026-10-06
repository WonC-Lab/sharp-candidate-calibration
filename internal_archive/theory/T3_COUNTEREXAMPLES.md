# Phase T3: explicit counterexamples and edge-case attacks

These are hand-constructed mathematical test instances, **not** additions to the locked empirical sample. In every binary example, $y_i=1$ means entity $i$ is active, excluded variables are fixed to zero, and the candidate family is the stated nested prefix.

## CE1. Sharp separation boundary and an explicit quality spectrum

Let $N=2$, $F=\{y\in\{0,1\}^2:y_1+y_2\le1\}$, $f(y)=5-2y_1-5y_2$, and $C(0)=\varnothing$, $C(1)=\{1\}$, $C(2)=\{1,2\}$. All three prefixes are feasible. Their optima are $J(0)=5$, $J(1)=3$, $J(2)=0=J^*$, so $g=(5,3,0)$, $B^\dagger=2$, and $\Delta=3$. Thus $Q(\tau)=2$ for $0\le\tau<3$, $Q(\tau)=1$ for $3\le\tau<5$, and $Q(\tau)=0$ for $\tau\ge5$. In particular $Q(\Delta)=1<2$: strictness in the separation criterion is necessary. Taking $J_{\rm ref}=0$ and $\epsilon_{\rm ref}=3$ gives a nonexact reference-pass budget 1. Taking $L=-2$ ($\eta=2$) and $\epsilon_{\rm cert}=5=\eta+\Delta$ gives the same nonexact ideal certificate budget 1.

## CE2. Same observed reference proxy, different exactness

Let $N=2$, $F=\{y\in\{0,1\}^2:y_1+y_2\le1\}$, $C(1)=\{1\}$, $C(2)=\{1,2\}$, $b_0=1$, and choose the feasible full reference $y=(1,0)$ of value $J_{\rm ref}=0$, with $\epsilon_{\rm ref}=0$. Define two objectives:

$$f_A(y)=1-y_1+2y_2,\qquad f_B(y)=1-y_1-2y_2.$$

Inside $C(1)$ both restricted formulations, feasible sets, objectives, and reference incumbent are identical: $y_2=0$, $J_A(1)=J_B(1)=0$. In A the full optimum is 0, hence $g_A=(0,0)$, $B_A^\dagger=1$, $\Delta_A=+\infty$, and $Q_A(0)=1$. In B the excluded point $(0,1)$ has objective $-1$, hence $g_B=(1,0)$, $B_B^\dagger=2$, $\Delta_B=1$, $\delta_B=J_{\rm ref}-J_B^*=1$, and $Q_B(\epsilon_{\rm ref}+\delta_B)=Q_B(1)=1$. Both operational reference labels equal 1, but only A is exact there. The proxy observation, even together with the entire candidate-restricted problem and reference value, does not identify exactness when its effective tolerance reaches the separation boundary. This does not contradict the sharp *instance-wise* identity: B's effective tolerance is exactly $\Delta_B$.

## CE3. Nonunique optima and multiple near-optimal supports

Let $N=2$, $F=\{y\in\{0,1\}^2:y_1+y_2\le1\}$, $f(y)=1-y_1-y_2$, and prefixes $C(0)=\varnothing$, $C(1)=\{1\}$, $C(2)=\{1,2\}$. Both supports $\{1\}$ and $\{2\}$ are globally optimal, but the first enters at budget 1, so $B^\dagger=1$ rather than 2. Here $g=(1,0,0)$ and $\Delta=1$. The support-entry minimum must range over *all* optimal supports; a single solver-selected optimum can overestimate first entry.

For support-equivalent points with different costs, augment the same binary feasible set by $x\in\{0,1\}$ with $x\le y_1$ and objective $f(y,x)=4-y_1-4y_2-x$. Points $(1,0,0)$ and $(1,0,1)$ share support $\{1\}$ but cost 3 and 2; $(0,1,0)$ is optimal at 0. At $\tau=2$, support $\{1\}$ is in $\mathcal S_\tau$ because **one** representative has cost 2. Defining the support family from an arbitrary representative would be wrong.

## CE4. Attainment cannot be silently omitted

This example deliberately uses a nonclosed continuous feasible set, outside standard closed bounded MILP assumptions. Let $N=2$, $C(1)=\{1\}$, $C(2)=\{1,2\}$, and full feasible points consist of $(y_1,y_2,x)$ with either $(1,0,x)$ for $0<x\le1$, or $(0,1,0)$. Let $f=x$. The full optimum $0$ is attained at support $\{2\}$. The restricted problem at budget 1 has infimum 0 but no minimizer. If one defines $J(1)$ as an infimum, $Q(0)=1$, while no globally optimal support enters until budget 2. The support-entry identity fails; attained restricted optima (or a modified strict-tolerance formulation) are required.

## CE5. Infeasible prefixes, mandatory blocks, and endpoints

Let $N=2$, $F=\{y\in\{0,1\}^2:y_1+y_2\ge1\}$ with prefixes $C(0)=\varnothing$, $C(1)=\{1\}$, $C(2)=U$. Budget 0 is infeasible, so $b_0=1$; it must not be included as a finite-gap breakpoint. If instead a mandatory set is $M=\{1\}$, $C(1)=M$ and $C(2)=U$, the top-rank formula counts $|M|$ in total budget. If $b_0=N$, there is no earlier feasible prefix, $B^\dagger=N$, $\Delta=+\infty$, and $Q(\tau)=N$ for every finite $\tau$. If $B^\dagger=b_0$ at any size, the same $+\infty$ convention applies.

## CE6. The finite-solver sandwich is sharp at every integer

Let $N=3$, $F=\{(y,z):y\in\{0,1\}^3,\ z\in\{0,1\},\ y_1+y_2+y_3\le1\}$, $f(y,z)=3-y_1-2y_2-3y_3+2z$, and $C(B)=\{1,\ldots,B\}$ for $B=1,2,3$; $z$ is a global auxiliary binary not subject to candidate fixing. The restricted optima are $(2,1,0)$, so $g=(2,1,0)$. At every budget, choosing $z=0$ or $z=1$ gives feasible incumbents with error 0 or 2. Take $t=2$ and $\bar\zeta=2$. Then $Q(t)=1$, $Q(t-\bar\zeta)=Q(0)=3$. The **physically feasible** error sequences $\zeta=(0,0,0)$, $(2,0,0)$, and $(2,2,0)$ yield observed first-pass budgets $1,2,3$, respectively. Thus both endpoints and every intermediate budget in the stated interval are attained in this MIO. Reverse inequality directions are false: the middle sequence has $\widetilde Q(2)=2>Q(2)=1$. A different fixed MIO might not possess all dominated incumbent objective values, so the general exact-interval claim is relative to the bounded-error information model, with sharpness over the displayed MIO class.

When $t<\bar\zeta$, even $B=N$ can fail: use the same profile with $t=1$, $\bar\zeta=2$, and $\zeta_B=2$ at all budgets. Since $g(N)+\zeta_N=2>1$, $\widetilde Q(1)=+\infty$ despite $Q(1)=2$.

## CE7. Nonmonotone observed residuals and inverse envelopes

With $g=(2,1,0)$ and $\zeta=(0,2,0)$, the observed residual sequence is $(2,3,0)$: nested candidates do **not** imply monotone finite-solver incumbents when independent time limits/solves are used. A binary search over observed pass/fail values is invalid without an incumbent-preservation policy.

For inverse identification, take $N=3$, observed $h=U-J^*=(2,2,1)$, and $\bar\zeta=1$. Both $g_{\min}=(1,1,0)$ and $g_{\max}=(2,2,0)$ satisfy the nesting and error constraints and yield the same observed $h$ with different $\zeta$. At tolerance 1, their true quality budgets are $Q_{g_{\min}}(1)=1$ and $Q_{g_{\max}}(1)=3$. The inverse envelope endpoints cannot be tightened from these observations alone.

## CE8. A lower bound and solver-error cap alone do not identify $Q$

Retain the identical candidate-restricted model from CE2, with observed candidate incumbent 0 at budget 1 and an error cap $\bar\zeta=0$. Give both instances the same valid lower bound $L=-1$. In A, $J_A^*=0$ and $Q_A(0)=1$; in B, $J_B^*=-1$ and $Q_B(0)=2$. Thus even exact candidate solves plus the same valid lower bound and candidate observations do not identify the exactness budget. Knowledge of $J^*$, a tight interval, or excluded-entity information is needed. The full-budget incumbent would distinguish them; the counterexample concerns the proposed “only a lower bound and error cap” information set.

## CE9. Candidate-only certification obstruction

Let $N=2$, $F=\{y\in\{0,1\}^2:y_1+y_2\le1\}$, and proper candidate $C=\{1\}$. For any $K>0$, define $f_A(y)=y_2$ and $f_{B,K}(y)=-Ky_2$. The candidate restriction $y_2=0$ is literally the same MILP in both: same variables, constraints, objective, optimum 0, candidate dual data, and every possible candidate solver transcript. In A the candidate incumbent 0 is globally optimal; in B the excluded point $(0,1)$ has cost $-K$, so its global gap is $K$. A candidate-only rule that reports a finite gap bound $G$ on this common transcript is unsound whenever $K>G$. If an instance class imposes $K\le K_0$, the minimax sound bound is at least $K_0$; a bound below that is impossible. A global lower bound or safe-screening fact can break the indistinguishability. The theorem is correct but reflects ordinary information insufficiency, not established new mathematics.

## CE10. Coverage transfer cannot assume independent failures

Let $U\sim\mathrm{Unif}[0,1]$, define failure of proxy-label coverage on $[0,\alpha)$ and failure of the conservativeness event on $[\alpha,\alpha+\beta)$, for $\alpha+\beta\le1$. The failures are disjoint, so the joint success probability is exactly $1-\alpha-\beta$, not $(1-\alpha)(1-\beta)$. Such dependence is compatible with the two marginal premises. Conversely, proxy coverage alone gives no exactness coverage: CE2 can occur on every test instance, with $R_{\rm proxy}=1<B^\dagger=2$ even if $R_{\rm proxy}\le\widehat B$ holds surely at $\widehat B=1$.

## Rejected overstatements

1. “$\tau\le\Delta$ preserves exactness” is false (CE1).
2. “An oracle proxy at or above $\Delta$ may *or may not* precede exactness” is inaccurate for a fixed instance with a feasible nonexact predecessor: it **must** precede; only finite-solver error can mask it.
3. “The observed solver budget is monotone in $B$” is false (CE7).
4. “$t<\bar\zeta$ merely weakens the upper bound but still ensures a full-budget pass” is false (CE6).
5. “A valid lower bound plus candidate-error cap identifies the true quality spectrum” is false (CE8).
6. “Candidate-only impossibility covers restricted instance classes with a known finite omitted-benefit bound” is false as stated; CE9 yields a finite minimax bound there.
