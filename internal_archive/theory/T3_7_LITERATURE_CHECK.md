# T3.7 — Targeted complexity and novelty audit

This is a targeted, non-exhaustive audit of the exact mechanism used by R1/R2. It does not assert that no earlier paper states the same ranked-prefix formulation. Each comparison distinguishes a proved fact from an inference about novelty.

## Classical hardness and sparsity

* [Karp (1972), “Reducibility Among Combinatorial Problems”](https://www2.seas.gwu.edu/~simhaweb/champalg/tsp/papers/KarpOriginal.pdf) includes SET COVERING among foundational NP-complete problems. R2 is a direct facility-location encoding of this source; no new CFLP approximation or metric hardness is inferred.
* [Natarajan (1995), “Sparse Approximate Solutions to Linear Systems”](https://epubs.siam.org/doi/abs/10.1137/S0097539792240406) proves NP-hardness of sparse linear-system solutions. This is related to the support-entry interpretation, but it optimizes support **size**, not the maximum fixed rank occupied by a support. A short counterexample in the complexity note proves those objectives need not coincide.
* [Stephan's work on cardinality-constrained combinatorial optimization](https://webdoc.sub.gwdg.de/ebook/dissts/Berlin_TU/Stephan2009.pdf) treats minimum/maximum-cardinality variants and weight transformations. It reinforces that secondary optimization over a solution set is an established perspective; it is not cited as having the exact R1 theorem.

## Alternative optima, variable fixing, and oracle search

* [Kilby, Slaney, Thiebaux, and Walsh (2005), “Backbones and Backdoors in Satisfiability”](https://cdn.aaai.org/AAAI/2005/AAAI05-217.pdf) studies literals fixed across satisfying assignments and the hardness of backbone computation. [Marques-Silva, Janota, and Lynce, “On Computing Backbones of Propositional Theories”](https://sat.inesc-id.pt/~mikolas/ecai10.pdf) develops backbone algorithms after satisfiability information. The R1 condition “is there an optimal solution with the late switch fixed to zero?” is the existential complement of a fixed-in-all-optima question; this is a close, established variable-fixing paradigm, though the cited papers do not supply R1's exact MILP statement.
* [Marques-Silva et al., “Computing Minimal Sets on Propositional Formulae I”](https://arxiv.org/abs/1402.3011) treats minimal sets and SAT-oracle complexity, including backbone-related tasks. The related [query-complexity paper](https://doi.org/10.1016/J.ARTINT.2016.01.002) further underscores that selecting a minimal set from a monotone predicate has prior theory. T3.7's exact-value-oracle binary search and full-spectrum linear-query observation should not be presented as fresh search complexity.
* [A 2025 study of multiple ILP optima](https://pmc.ncbi.nlm.nih.gov/articles/PMC11970949/) addresses enumeration/sampling of alternative optimal integer solutions. Its task is different from minimum prefix entry, but it confirms that alternative optima are an active, pre-existing object of study.

## Polynomial and parameterized baselines

* [Lenstra (1983), “Integer Programming with a Fixed Number of Variables”](https://pub.math.leidenuniv.nl/~lenstrahw/PUBLICATIONS/1983i/art.pdf) is the classical fixed-integer-dimension baseline. T3.7's $2^p$ binary enumeration/LP bound is simpler and not novel.
* [Goldberg and Tarjan, “Solving Minimum-Cost Flow Problems”](https://www.cs.princeton.edu/techreports/1987/081.pdf) gives polynomial algorithms for min-cost flow; T2's polynomial candidate spectrum follows by repeated ordinary flow solves.
* [Karp (1972)](https://www2.seas.gwu.edu/~simhaweb/champalg/tsp/papers/KarpOriginal.pdf) and [Feige (1998), “A Threshold of ln n for Approximating Set Cover”](https://disco.ethz.ch/alumni/pascalv/refs/ds_1998_feige.pdf) set important approximation context. R2 preserves a *cost-threshold decision* but not the optimum candidate-dimension ratio, so Feige's logarithmic hardness cannot be transferred. A logarithmic prefix-index inapproximability claim would be unsupported.

## Hostile novelty test

The literal statement “earliest ranked prefix among global optima” differs from sparsest-optimum cardinality, and R1 decisively survives supplied $J^*$ and $v^*$. However, its proof sets one late switch to zero and asks whether a restricted SAT subsystem admits a zero-cost optimum. That is an established type of alternative-solution/variable-fixing query. The fixed ranking adds an interpretation and a marker locating the first possible budget, not a new reduction principle. A reviewer could therefore accept R1 as a correct supporting theorem while rejecting it as an independent top-tier complexity pillar.

No source inspected was found to state R1 verbatim. That narrow observation is **not** a firstness claim. Stronger positioning would require a nontrivial approximation-preserving reduction, a genuine parameterized dichotomy in a natural model, a class-specific theorem not inherited from restricted subproblem hardness, or a new algorithmic separation. None is established here.
