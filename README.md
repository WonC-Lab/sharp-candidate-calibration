# Learning-Augmented Multi-Period Portfolio Optimization

Research-grade framework for **Certified Learning-Augmented Multi-Period Portfolio Rebalancing with Mixed-Integer Transaction Costs and Liquidity Constraints**.

## Core Philosophy

> **Machine learning does not replace optimization; it identifies promising combinatorial structure so that a certified mathematical optimization solver can solve repeated portfolio-rebalancing problems substantially more efficiently.**

- **ML is advisory**: Predicts candidate active sets (support) and initial binary structures (warm starts).
- **Solver is authoritative**: Gurobi MIQP guarantees 100% constraint satisfaction and provable bounds.

## Project Structure

```text
learning_augmented_portfolio/
├── pyproject.toml
├── README.md
├── src/
│   └── learning_augmented_portfolio/
│       ├── optimization/
│       │   ├── variables.py           # Dataclasses and parameter specifications
│       │   └── single_period_qp.py    # Phase 1: Canonical Convex QP formulation
│       └── solver/
│           └── gurobi_solver.py       # Solver configurations and status handling
├── tests/
│   └── test_qp.py                     # Comprehensive 10-point unit test suite
└── experiments/
    └── exp01_qp_validation.py        # Minimal synthetic benchmark validation
```

## Phase 1 Formulation (Convex QP)

$$\begin{aligned}
\min_{w, u, q, c, \kappa} \quad & \frac{\gamma}{2} w^\top \Sigma w - \mu^\top w - r_f c + \kappa \\
\text{s.t.} \quad
& w - u = \bar{w} \\
& \mathbf{1}^\top w + c + \kappa = 1 \\
& \kappa - a^\top q = 0 \\
& q - u \ge 0, \quad q + u \ge 0 \\
& \sum_{i=1}^n q_i \le \tau_{\max} \\
& E_g^{\min} \le \sum_{i \in \mathcal{S}_g} w_i \le E_g^{\max}, \quad \forall g \\
& 0 \le w_i \le U_i, \quad 0 \le q_i \le L_i, \quad c \ge c_{\min}
\end{aligned}$$
