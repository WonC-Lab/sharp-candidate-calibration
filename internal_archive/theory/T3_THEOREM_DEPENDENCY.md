# T3 theorem architecture and dependency graph

```text
Finite entity-block universe + null-state restriction + nested candidates
    |
    +-- feasible-budget floor, attainment, full candidate at N
    |       |
    |       +--> monotone candidate-gap profile g(B) [STANDARD]
    |               |
    |               +--> quality-budget spectrum Q(tau) [primitive]
    |               |       |
    |               |       +--> tau-optimal support-entry equality [SUPPORTING]
    |               |       |       +--> ranked tau-support depth [corollary]
    |               |       |       +--> exactness budget B† = Q(0)
    |               |       |
    |               |       +--> first positive separation Delta [SUPPORTING]
    |               |               |
    |               |               +--> sharp reference translation [HEADLINE candidate]
    |               |               +--> sharp ideal-certificate translation [HEADLINE candidate]
    |               |               +--> proxy-conservativeness event
    |               |                       +--> standard union-bound coverage transfer
    |               |
    |               +--> finite-solver error profile zeta(B)
    |                       |
    |                       +--> every attainable observed first-pass budget
    |                       |    in [Q(t), Q(t-zeta_bar)] [HEADLINE candidate]
    |                       +--> inverse monotone-envelope identification
    |                       |    when J* is independently known [HEADLINE candidate]
    |                       +--> certificate decomposition g + eta + zeta
    |
    +-- valid full-universe lower bound L
            |
            +--> eta = J* - L >= 0
            +--> fixed-bound impossibility when eta > epsilon_cert
            +--> certified budget via effective tolerance epsilon_cert - eta

Proper candidate with no excluded-entity information
    +--> indistinguishable full instances [STANDARD obstruction]
            +--> need valid global bound, safe screening, structural exclusion,
                 or full-model search for deterministic global certification
```

## Reading order for a future approved rewrite

1. Define the restriction semantics, feasibility floor, and candidate quality profile.
2. Introduce $Q(\tau)$ and prove support entry once; present exactness as $Q(0)$, not a separate theorem.
3. Define $\Delta$ and give the sharp iff first-transition statement.
4. Present reference and valid-lower-bound proxies as translations of one spectrum, emphasizing their different signs and the $\eta>\epsilon$ barrier.
5. Add finite-incumbent error, the sharp identified interval, and the $g+\eta+\zeta$ decomposition.
6. Only then introduce split-conformal calibration of a budget label and the secondary operational-to-exact transfer.
7. Keep geometric recovery, perspective relaxation, and application-specific integer integrality in later sections or appendices, not as the spine's novelty claims.

## Non-independence of proposed results

The support-entry theorem and separation margin both derive from the same finite nested sequence. The translation phase diagram is Proposition 2 under shifted tolerances. The lower-bound barrier is the zero-gap endpoint of that diagram and weak duality. Coverage transfer is a union bound after deterministic proxy conservativeness. The finite-solver identified interval uses nesting plus bounded additive error and is the only T3 result with a substantive attainable-set construction. The apparent count of theorem statements must **not** be interpreted as a count of independent deep contributions.

## Scientific gate

The current architecture is coherent, exact, and useful for explaining the locked negative certification findings, but it does not yet establish two independently deep mathematical theorems. The required next theoretical advance is a computationally informative, model-structural bound on $Q$ or $\Delta$ (or a genuinely new sharp identification result under realistic full-bound data), not an additional corollary of the same finite first-passage identity. No T4–T7 work should be inferred from this diagram.
