"""
Nested Candidate Recovery Loop and Integrated Optimizer.
Implements:
  - Fast Mode (latency-minimized reduced solve without certificate overhead)
  - Certified Mode (nested candidate expansion guided by full relaxation certificates)
  - Eventual Full-Universe Fallback (C_t(B_m) = {0, ..., N-1})
  - Inter-stage Warm Starting (reusing stage j incumbent to initialize stage j+1)
  - Strictly avoids double-counting relaxation costs
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from ..dataset.instance import OptimizationInstance
from ..learning.warm_start import MIPStartData, WarmStartType
from ..optimization.reduced_miqp import MultiPeriodReducedMIQPOptimizer
from ..optimization.relaxation import MultiPeriodRelaxedQPOptimizer
from ..optimization.support_restriction import CandidateUniverse
from ..optimization.variables import (
    MultiPeriodMarketState,
    MultiPeriodOptimizationResult,
    MultiPeriodPortfolioParameters,
    SolverConfig,
)
from .candidate_sets import generate_nested_candidate_sequence
from .certificate import (
    CertificateSource,
    FullProblemCertificate,
    compute_full_problem_certificate,
)
from .feasibility import FullProblemFeasibilityChecker


@dataclass
class StageTelemetry:
    """Telemetry for a single stage in the recovery sequence."""
    stage_index: int
    budget: int
    effective_size: int
    is_reduced_feasible: bool
    is_full_feasible: bool
    objective_value: Optional[float]
    certificate_gap: Optional[float]
    relative_certificate_gap: Optional[float]
    tolerance_met: bool
    certificate_source: CertificateSource
    relaxation_certified: bool
    full_mip_certified: bool
    trigger_reason: str
    stage_time: float
    solver_time: float
    node_count: int


@dataclass
class RecoveryLoopResult:
    """Overall output container for nested recovery optimization."""
    instance_id: str
    fast_mode: bool
    is_certified: bool                     # Relaxation-certified
    relaxation_certified: bool
    full_mip_certified: bool
    any_valid_certificate: bool
    certificate_source: CertificateSource
    final_objective: Optional[float]
    lower_bound: Optional[float]           # Backward-compatible LB^{relax}
    lower_bound_relax: Optional[float]
    lower_bound_best: Optional[float]
    full_mip_bound: Optional[float]
    upper_bound: Optional[float]
    final_certificate_gap: Optional[float] # Relaxation gap
    relaxation_certificate_gap: Optional[float]
    full_mip_certificate_gap: Optional[float]
    relative_certificate_gap: Optional[float]
    stages_executed: int
    initial_budget: int
    final_budget: int
    initial_reduction: float
    final_reduction: float
    full_fallback_occurred: bool
    total_runtime: float
    relaxation_time: float
    final_solution: Optional[MultiPeriodOptimizationResult]
    stage_telemetry: List[StageTelemetry] = field(default_factory=list)


class NestedRecoveryOptimizer:
    """
    Executes the nested candidate recovery loop:
      C_t(B_1) subset C_t(B_2) subset ... subset C_t(B_m) = {0, ..., N - 1}
    guaranteeing mathematically valid certification and eventual full-universe fallback.
    """

    def __init__(
        self,
        params: MultiPeriodPortfolioParameters,
        solver_config: Optional[SolverConfig] = None,
        epsilon_cert: float = 1e-4,
        use_recovery_warm_starts: bool = True,
    ) -> None:
        self.params = params
        self.solver_config = solver_config or SolverConfig()
        self.epsilon_cert = epsilon_cert
        self.use_recovery_warm_starts = use_recovery_warm_starts

    def solve(
        self,
        instance: OptimizationInstance,
        scores: np.ndarray,
        budgets: Sequence[int],
        initial_warm_start: Optional[MIPStartData] = None,
        precomputed_lower_bound: Optional[float] = None,
        precomputed_relaxation_time: float = 0.0,
        fast_mode: bool = False,
    ) -> RecoveryLoopResult:
        """
        Execute recovery optimization on instance.
        """
        t0_total = time.perf_counter()
        inst_id = instance.metadata.instance_id
        current_w = instance.actual_state.normalized_weights
        N = instance.market_state.n_assets

        # 1. Lower Bound calculation or reuse
        lb_val = precomputed_lower_bound
        t_relax = precomputed_relaxation_time

        if not fast_mode and lb_val is None:
            # Must compute full relaxation lower bound
            t0_relax = time.perf_counter()
            relaxed_opt = MultiPeriodRelaxedQPOptimizer(self.params, self.solver_config)
            rel_res = relaxed_opt.solve(instance.market_state)
            t_relax = time.perf_counter() - t0_relax
            if rel_res.is_feasible:
                lb_val = rel_res.objective_value
            else:
                lb_val = float("-inf")

        # 2. Build nested candidate sequence
        cand_list = generate_nested_candidate_sequence(
            scores=scores,
            current_weights=current_w,
            budgets=budgets,
        )

        stage_history: List[StageTelemetry] = []
        current_start = initial_warm_start
        final_sol: Optional[MultiPeriodOptimizationResult] = None
        final_cert: Optional[FullProblemCertificate] = None
        full_fallback = False

        checker = FullProblemFeasibilityChecker(self.params, instance.market_state)

        for j, cand in enumerate(cand_list):
            t0_stage = time.perf_counter()
            reduced_opt = MultiPeriodReducedMIQPOptimizer(self.params, self.solver_config)
            red_exec = reduced_opt.solve(
                full_state=instance.market_state,
                candidate=cand,
                expand_solution=True,
                warm_start=current_start,
            )
            t_stage = time.perf_counter() - t0_stage

            lifted_sol = red_exec.optimization_result
            final_sol = lifted_sol
            audit = checker.audit_solution(lifted_sol)

            trigger_recovery = False
            trigger_reason = "OK"
            abs_gap = None
            rel_gap = None
            tol_met = False
            cert_source = CertificateSource.NONE
            relax_cert = False
            mip_cert = False
            is_full = (budgets[j] >= N)

            if not audit.is_feasible:
                trigger_recovery = True
                trigger_reason = f"Lifted solution infeasible: {audit.violations[:1]}"
            else:
                ub_val = audit.objective_value
                if not fast_mode and lb_val is not None:
                    cert = compute_full_problem_certificate(
                        instance_id=inst_id,
                        lower_bound=lb_val,
                        upper_bound=ub_val,
                        reduced_mip_gap=lifted_sol.mip_gap,
                        candidate_budget=budgets[j],
                        candidate_size=cand.candidate_size,
                        epsilon_cert=self.epsilon_cert,
                        full_mip_bound=lifted_sol.best_bound if is_full else None,
                        full_mip_gap=lifted_sol.mip_gap if is_full else None,
                        epsilon_mip=self.solver_config.mip_gap,
                        is_full_universe=is_full,
                    )
                    final_cert = cert
                    abs_gap = cert.relaxation_certificate_gap
                    rel_gap = cert.relative_gap
                    cert_source = cert.certificate_source
                    relax_cert = cert.relaxation_certified
                    mip_cert = cert.full_mip_certified
                    tol_met = cert.any_valid_certificate

                    # Trigger recovery if relaxation certificate not met (unless already at full universe)
                    if not relax_cert and not is_full:
                        trigger_recovery = True
                        trigger_reason = f"Relaxation certificate gap {abs_gap:.2e} > epsilon {self.epsilon_cert:.2e}"
                else:
                    # Fast mode: valid solution found, terminate immediately
                    tol_met = True

            telemetry = StageTelemetry(
                stage_index=j,
                budget=budgets[j],
                effective_size=cand.candidate_size,
                is_reduced_feasible=lifted_sol.is_feasible,
                is_full_feasible=audit.is_feasible,
                objective_value=audit.objective_value if audit.is_feasible else None,
                certificate_gap=abs_gap,
                relative_certificate_gap=rel_gap,
                tolerance_met=tol_met,
                certificate_source=cert_source,
                relaxation_certified=relax_cert,
                full_mip_certified=mip_cert,
                trigger_reason=trigger_reason,
                stage_time=t_stage,
                solver_time=red_exec.solver_time,
                node_count=red_exec.node_count or 0,
            )
            stage_history.append(telemetry)

            if fast_mode:
                break

            if not trigger_recovery:
                # Early stop achieved via relaxation certificate or full universe completion!
                break

            if j < len(cand_list) - 1:
                # Inter-stage warm start: carry forward feasible lifted solution
                if self.use_recovery_warm_starts and audit.is_feasible:
                    current_start = MIPStartData(
                        start_type=WarmStartType.PREVIOUS_SOLUTION,
                        y_start=lifted_sol.holding_indicators,
                        z_start=lifted_sol.trade_indicators,
                        x_start=lifted_sol.post_trade_holdings,
                        b_start=lifted_sol.post_trade_cash,
                        is_complete=True,
                    )
                else:
                    current_start = initial_warm_start

        total_time = (time.perf_counter() - t0_total)

        first_stage = stage_history[0]
        last_stage = stage_history[-1]
        full_fallback = (last_stage.budget >= N and first_stage.budget < N)

        final_source = final_cert.certificate_source if final_cert else CertificateSource.NONE
        relax_certified = final_cert.relaxation_certified if final_cert else False
        full_mip_certified = final_cert.full_mip_certified if final_cert else False
        any_valid_cert = final_cert.any_valid_certificate if final_cert else False

        return RecoveryLoopResult(
            instance_id=inst_id,
            fast_mode=fast_mode,
            is_certified=relax_certified,
            relaxation_certified=relax_certified,
            full_mip_certified=full_mip_certified,
            any_valid_certificate=any_valid_cert,
            certificate_source=final_source,
            final_objective=last_stage.objective_value,
            lower_bound=lb_val,
            lower_bound_relax=lb_val,
            lower_bound_best=final_cert.lower_bound_best if final_cert else lb_val,
            full_mip_bound=final_cert.full_mip_bound if final_cert else None,
            upper_bound=last_stage.objective_value,
            final_certificate_gap=last_stage.certificate_gap,
            relaxation_certificate_gap=final_cert.relaxation_certificate_gap if final_cert else None,
            full_mip_certificate_gap=final_cert.full_mip_certificate_gap if final_cert else None,
            relative_certificate_gap=last_stage.relative_certificate_gap,
            stages_executed=len(stage_history),
            initial_budget=first_stage.budget,
            final_budget=last_stage.budget,
            initial_reduction=1.0 - float(first_stage.effective_size) / float(N),
            final_reduction=1.0 - float(last_stage.effective_size) / float(N),
            full_fallback_occurred=full_fallback,
            total_runtime=total_time,
            relaxation_time=t_relax,
            final_solution=final_sol,
            stage_telemetry=stage_history,
        )
