# T3.6 — Finite MIO sharpness and information limits

The relevant information class is the vector of **valid scalar** primal/dual records and nestedness, not complete solver logs or full model coefficients. All examples have minimization objectives and $\tau=0$; adding a positive prefix cost and shifting the threshold gives analogous nonzero-tolerance examples. Bounds may be weak but must be valid.

## A nonconvex identified set

Let $I=\{1,2,3\}$, $L=(1,2,1)$, $U=(2,3,2)$, and full bounds $[L_*,U_*]=[1,2]$. Then $a=(2,2,1)$, $c=(2,2,2)$, $S=[1,2]$. Two compatible profiles are
\[
j^{(A)}=(2,2,2),\quad Q_{j^{(A)}}(0)=1;\qquad
j^{(B)}=(2,2,1),\quad Q_{j^{(B)}}(0)=3.
\]
Budget 2 is impossible: $j_2=s$ would force $s\ge2$, whereas $j_1>s$ would force $j_1>2$, contradicting $U_1=2$. Hence the exact identified set is $\{1,3\}$, not its hull $\{1,2,3\}$. D5 gives the same exclusion because $[a_2,c_1)=[2,2)$ is empty. This is a sharp converse, though the identical phenomenon already appears in Theorem C's coarser information class.

## Each side of the singleton test is necessary

Take $I=\{1,2,3\}$, $s=1$ known exactly.

1. **Relax strict predecessor failure.** Let $L=(1,1,1)$, $U=(2,1,1)$. Candidate 2 passes in every compatible profile, but $j=(1,1,1)$ has $Q(0)=1$, while $j=(2,1,1)$ has $Q(0)=2$. The predecessor's lower gap is zero, not strictly positive. Replacing $>0$ by $\ge0$ is invalid.
2. **Relax current-budget sufficiency.** Let $L=(2,1,1)$, $U=(2,2,1)$. Candidate 1 fails in every compatible profile, but $j=(2,1,1)$ has $Q(0)=2$, while $j=(2,2,1)$ has $Q(0)=3$. The candidate's upper gap is one, not zero. Without the pass inequality, $q=2$ is not point-identified.

The endpoint exceptions in (D8) are also necessary: (N) passes regardless of a loose interval $[L_N,U_N]$, and $b_0$ has no predecessor. These counterexamples show both inequalities cannot be discarded or made weak at the exactness boundary; they do not establish a new computational separation theorem.

## Finite 0–1 realization of every admissible profile

Given any finite nonincreasing $j_B$ and scalar upper records $U_B\ge j_B$, introduce binary choices $y_k$ with $\sum_{k\in I}y_k=1$ and binaries $w_k\le y_k$. At budget (B), impose $y_k=w_k=0$ for $k>B$, and minimize
\[
\sum_{k\in I}\bigl[j_k y_k+(U_k-j_k)w_k\bigr]. \tag{S1}
\]
Because $j_k\ge j_B$ for $k\le B$, the restricted optimum equals $j_B$. Choosing $y_B=w_B=1$ realizes a feasible incumbent of value $U_B$ for that candidate MIO. If a separate full incumbent $U_*\ge s$ must also be realized, add a full-only dominated binary option of cost $U_*$; it does not change any optimum. Every $L_B\le j_B$ and $L_*\le s$ is a valid lower numerical bound. Thus the two profiles in each example can be represented by distinct finite 0–1 MILPs with exactly the **same reported scalar records** and different (Q). One may pad mandatory null entities to obtain any requested starting candidate cardinality.

This construction establishes sharpness over a class of MIO formulations and possible valid bound reports. It does **not** prove that a specified solver, presolve setting, branch-and-bound tree, or runtime will emit those same numbers for both formulations. Nor does it make the formulations indistinguishable to an algorithm allowed to inspect their coefficients. Claiming more would exceed the observation model.

## Adaptive observational lower bound

At any adaptive time (t), suppose the accumulated scalar records give $|\mathcal Q_\tau(t)|>1$. Choose distinct $r_1,r_2\in\mathcal Q_\tau(t)$. Theorem D2 constructs two compatible profiles with these respective first-pass budgets; (S1) realizes both by finite 0–1 MIOs with the same accumulated valid scalar records. A deterministic rule seeing only those records and past budget choices has the same information transcript at time (t), so it cannot correctly announce a unique (Q) in both worlds. The same holds for a randomized rule claiming error-free identification on every world: its output distribution cannot be a point mass at both distinct labels. This is an adversarial *record-information* limit, almost definitional once the sharp identified set is known. It is neither a runtime lower bound nor a lower bound for an algorithm seeing the complete MIO models or obtaining richer branch-and-bound certificates.

If future solver behavior is part of the observation, indistinguishability must be re-established for that larger transcript; the current construction makes no claim about it.
