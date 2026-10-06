"""
Scalability Experiment Runner Module.
Executes paired multi-method evaluations on identical synthetic instances:
  - M0: Full MIQP
  - M1: Fast MarketOnly Candidate MIQP
  - M2: ContinuousRelaxation Ranking + Reduced MIQP
  - M3: Best Learned Candidate Method
  - M4: Reduced MIQP + Warm Start
  - M5: Certified Recovery (Full relaxation certificate + nested recovery)
  - M6: Oracle-Support Diagnostic (OFFLINE ONLY - not deployable)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import gurobipy as gp
import numpy as np

from ..dataset.instance import OptimizationInstance
from ..evaluation.scalability_metrics import (
    E2ETimerBreakdown,
    compute_objective_quality,
    compute_speedup,
    compute_runtime_reduction,
)
from ..learning.baselines import (
    ContinuousRelaxationYBaseline,
    ExpectedReturnBaseline,
)
from ..learning.candidate_set import construct_candidate_universe
from ..learning.warm_start import MIPStartData, WarmStartType
from ..optimization.multi_period_miqp import MultiPeriodMIQPOptimizer
from ..optimization.reduced_miqp import MultiPeriodReducedMIQPOptimizer
from ..optimization.relaxation import MultiPeriodRelaxedQPOptimizer
from ..optimization.support_restriction import CandidateUniverse
from ..optimization.variables import MultiPeriodOptimizationResult, SolverConfig
from ..recovery.certificate import CertificateSource
from ..recovery.recovery_loop import NestedRecoveryOptimizer
from .anytime import AnytimeCallback, AnytimeTrajectory
from .failure import SolverExitStatus, classify_solver_outcome


@dataclass
class MethodSolveResult:
    """Detailed execution telemetry for a single method solve on an instance."""
    method_name: str
    instance_id: str
    n_assets: int
    horizon: int
    candidate_size: int
    status: SolverExitStatus
    objective_value: Optional[float]
    best_bound: Optional[float]
    mip_gap: Optional[float]
    node_count: int
    work: float
    timers: E2ETimerBreakdown
    anytime_trajectory: Optional[AnytimeTrajectory] = None
    root_relaxation_time: Optional[float] = None
    root_relaxation_obj: Optional[float] = None
    recovery_stages: int = 1
    certificate_source: str = "NONE"
    relaxation_certified: bool = False
    full_mip_certified: bool = False
    any_valid_certificate: bool = False
    relaxation_cert_gap: Optional[float] = None
    is_deployable: bool = True  # False for M6 Oracle diagnostic
    uses_teacher_information: bool = False
    holding_indicators: Optional[np.ndarray] = None


class ScalabilityRunner:
    """
    Orchestrates paired multi-method execution on a given OptimizationInstance.
    """

    def __init__(self, solver_config: Optional[SolverConfig] = None) -> None:
        self.solver_config = solver_config or SolverConfig(time_limit=15.0, mip_gap=1e-4, threads=1)

    def solve_full_miqp(
        self,
        instance: OptimizationInstance,
        record_anytime: bool = True,
    ) -> MethodSolveResult:
        """M0: Full MIQP baseline."""
        inst = instance
        N = inst.market_state.n_assets
        H = inst.params.horizon

        timers = E2ETimerBreakdown()
        cb = AnytimeCallback(target_tol=self.solver_config.mip_gap) if record_anytime else None

        t0_solver = time.perf_counter()
        opt = MultiPeriodMIQPOptimizer(inst.params, self.solver_config)

        try:
            if cb is not None:
                cb.start()
            sol = opt.solve(inst.market_state, callback=cb)
            timers.t_solver_sec = time.perf_counter() - t0_solver

            status = classify_solver_outcome(
                is_optimal=sol.is_optimal,
                is_feasible=sol.is_feasible,
                has_incumbent=sol.objective_value is not None,
                gurobi_status=sol.status,
            )

            return MethodSolveResult(
                method_name="Full_MIQP",
                instance_id=inst.metadata.instance_id,
                n_assets=N,
                horizon=H,
                candidate_size=N,
                status=status,
                objective_value=sol.objective_value,
                best_bound=sol.best_bound,
                mip_gap=sol.mip_gap,
                node_count=sol.node_count or 0,
                work=sol.work or 0.0,
                timers=timers,
                anytime_trajectory=cb.trajectory if cb else None,
                certificate_source=CertificateSource.FULL_MIP_BOUND.value if sol.is_optimal else "NONE",
                full_mip_certified=sol.is_optimal,
                any_valid_certificate=sol.is_optimal,
                holding_indicators=sol.holding_indicators,
            )
        except gp.GurobiError as e:
            timers.t_solver_sec = time.perf_counter() - t0_solver
            status = classify_solver_outcome(False, False, False, 0, str(e))
            return MethodSolveResult(
                method_name="Full_MIQP",
                instance_id=inst.metadata.instance_id,
                n_assets=N,
                horizon=H,
                candidate_size=N,
                status=status,
                objective_value=None,
                best_bound=None,
                mip_gap=None,
                node_count=0,
                work=0.0,
                timers=timers,
            )

    def solve_fast_market_only(
        self,
        instance: OptimizationInstance,
        candidate_budget_B: int,
        scores: Optional[np.ndarray] = None,
    ) -> MethodSolveResult:
        """M1: Fast MarketOnly Candidate MIQP (no relaxation certificate)."""
        inst = instance
        N = inst.market_state.n_assets
        H = inst.params.horizon
        timers = E2ETimerBreakdown()

        # 1. Ranking scores (e.g. Expected Return heuristic if scores not provided)
        t0_rank = time.perf_counter()
        if scores is None:
            ranker = ExpectedReturnBaseline()
            scores = ranker.predict_instance_scores(inst)
        timers.t_ranking_sec = time.perf_counter() - t0_rank

        # 2. Candidate set construction
        t0_cand = time.perf_counter()
        n_held = int(np.sum(inst.actual_state.normalized_weights > 1e-6))
        eff_B = max(candidate_budget_B, n_held)
        cand = construct_candidate_universe(
            scores=scores,
            budget_B=eff_B,
            current_weights=inst.actual_state.normalized_weights,
            use_safe_augmentation=True,
        )
        timers.t_build_sec += (time.perf_counter() - t0_cand)

        # 3. Physically sliced solve
        t0_solver = time.perf_counter()
        red_opt = MultiPeriodReducedMIQPOptimizer(inst.params, self.solver_config)
        try:
            red_exec = red_opt.solve(inst.market_state, cand, expand_solution=True)
            timers.t_build_sec += red_exec.build_time
            timers.t_solver_sec = red_exec.solver_time

            sol = red_exec.optimization_result
            status = classify_solver_outcome(
                is_optimal=sol.is_optimal,
                is_feasible=sol.is_feasible,
                has_incumbent=sol.objective_value is not None,
                gurobi_status=sol.status,
            )
            return MethodSolveResult(
                method_name="Fast_MarketOnly_MIQP",
                instance_id=inst.metadata.instance_id,
                n_assets=N,
                horizon=H,
                candidate_size=cand.candidate_size,
                status=status,
                objective_value=sol.objective_value,
                best_bound=sol.best_bound,
                mip_gap=sol.mip_gap,
                node_count=sol.node_count or 0,
                work=sol.work or 0.0,
                timers=timers,
            )
        except gp.GurobiError as e:
            timers.t_solver_sec = time.perf_counter() - t0_solver
            status = classify_solver_outcome(False, False, False, 0, str(e))
            return MethodSolveResult(
                method_name="Fast_MarketOnly_MIQP",
                instance_id=inst.metadata.instance_id,
                n_assets=N,
                horizon=H,
                candidate_size=cand.candidate_size,
                status=status,
                objective_value=None,
                best_bound=None,
                mip_gap=None,
                node_count=0,
                work=0.0,
                timers=timers,
            )

    def solve_continuous_relaxation_ranking(
        self,
        instance: OptimizationInstance,
        candidate_budget_B: int,
    ) -> MethodSolveResult:
        """M2: ContinuousRelaxation Ranking + Reduced MIQP (includes relaxation timer)."""
        inst = instance
        N = inst.market_state.n_assets
        H = inst.params.horizon
        timers = E2ETimerBreakdown()

        # 1. Full continuous relaxation solve
        t0_relax = time.perf_counter()
        rel_opt = MultiPeriodRelaxedQPOptimizer(inst.params, self.solver_config)
        try:
            rel_sol = rel_opt.solve(inst.market_state)
            timers.t_relax_sec = time.perf_counter() - t0_relax
        except gp.GurobiError as e:
            timers.t_relax_sec = time.perf_counter() - t0_relax
            status = classify_solver_outcome(False, False, False, 0, str(e))
            return MethodSolveResult(
                method_name="ContinuousRelaxation_MIQP",
                instance_id=inst.metadata.instance_id,
                n_assets=N,
                horizon=H,
                candidate_size=candidate_budget_B,
                status=status,
                objective_value=None,
                best_bound=None,
                mip_gap=None,
                node_count=0,
                work=0.0,
                timers=timers,
            )

        # 2. Score via ContinuousRelaxationYBaseline
        t0_rank = time.perf_counter()
        inst_copy = OptimizationInstance(
            metadata=inst.metadata,
            actual_state=inst.actual_state,
            params=inst.params,
            market_state=inst.market_state,
            relaxation=rel_sol,
        )
        ranker = ContinuousRelaxationYBaseline()
        scores = ranker.predict_instance_scores(inst_copy)
        timers.t_ranking_sec = time.perf_counter() - t0_rank

        # 3. Candidate universe
        t0_cand = time.perf_counter()
        n_held = int(np.sum(inst.actual_state.normalized_weights > 1e-6))
        eff_B = max(candidate_budget_B, n_held)
        cand = construct_candidate_universe(
            scores=scores,
            budget_B=eff_B,
            current_weights=inst.actual_state.normalized_weights,
            use_safe_augmentation=True,
        )
        timers.t_build_sec += (time.perf_counter() - t0_cand)

        # 4. Physically sliced reduced MIQP
        t0_solver = time.perf_counter()
        red_opt = MultiPeriodReducedMIQPOptimizer(inst.params, self.solver_config)
        try:
            red_exec = red_opt.solve(inst.market_state, cand, expand_solution=True)
            timers.t_build_sec += red_exec.build_time
            timers.t_solver_sec = red_exec.solver_time

            sol = red_exec.optimization_result
            status = classify_solver_outcome(
                is_optimal=sol.is_optimal,
                is_feasible=sol.is_feasible,
                has_incumbent=sol.objective_value is not None,
                gurobi_status=sol.status,
            )

            # Check relaxation gap
            gap = (sol.objective_value - rel_sol.objective_value) if (sol.is_feasible and rel_sol.is_feasible) else None
            is_rel_cert = bool(gap is not None and gap <= 1e-4)

            return MethodSolveResult(
                method_name="ContinuousRelaxation_MIQP",
                instance_id=inst.metadata.instance_id,
                n_assets=N,
                horizon=H,
                candidate_size=cand.candidate_size,
                status=status,
                objective_value=sol.objective_value,
                best_bound=sol.best_bound,
                mip_gap=sol.mip_gap,
                node_count=sol.node_count or 0,
                work=sol.work or 0.0,
                timers=timers,
                root_relaxation_time=timers.t_relax_sec,
                root_relaxation_obj=rel_sol.objective_value,
                certificate_source=CertificateSource.FULL_RELAXATION.value if is_rel_cert else "NONE",
                relaxation_certified=is_rel_cert,
                any_valid_certificate=is_rel_cert,
                relaxation_cert_gap=gap,
            )
        except gp.GurobiError as e:
            timers.t_solver_sec = time.perf_counter() - t0_solver
            status = classify_solver_outcome(False, False, False, 0, str(e))
            return MethodSolveResult(
                method_name="ContinuousRelaxation_MIQP",
                instance_id=inst.metadata.instance_id,
                n_assets=N,
                horizon=H,
                candidate_size=cand.candidate_size,
                status=status,
                objective_value=None,
                best_bound=None,
                mip_gap=None,
                node_count=0,
                work=0.0,
                timers=timers,
            )

    def solve_certified_recovery(
        self,
        instance: OptimizationInstance,
        nested_budgets: List[int],
        scores: Optional[np.ndarray] = None,
    ) -> MethodSolveResult:
        """M5: Certified Recovery Loop (relaxation certification + nested expansion)."""
        inst = instance
        N = inst.market_state.n_assets
        H = inst.params.horizon
        timers = E2ETimerBreakdown()

        t0_rank = time.perf_counter()
        if scores is None:
            ranker = ExpectedReturnBaseline()
            scores = ranker.predict_instance_scores(inst)
        timers.t_ranking_sec = time.perf_counter() - t0_rank

        t0_rec = time.perf_counter()
        rec_opt = NestedRecoveryOptimizer(
            params=inst.params,
            solver_config=self.solver_config,
            epsilon_cert=1e-4,
            use_recovery_warm_starts=True,
        )

        n_held = int(np.sum(inst.actual_state.normalized_weights > 1e-6))
        clamped_budgets = [max(b, n_held) for b in nested_budgets]
        clamped_budgets = sorted(list(set(clamped_budgets)))

        try:
            rec_res = rec_opt.solve(
                instance=inst,
                scores=scores,
                budgets=clamped_budgets,
                fast_mode=False,
            )
            timers.t_solver_sec = time.perf_counter() - t0_rec
            timers.t_relax_sec = rec_res.relaxation_time

            sol = rec_res.final_solution
            status = classify_solver_outcome(
                is_optimal=(sol.is_optimal if sol else False),
                is_feasible=(sol.is_feasible if sol else False),
                has_incumbent=(rec_res.final_objective is not None),
                gurobi_status=(sol.status if sol else 2),
            )
            cert_src = rec_res.certificate_source.value if hasattr(rec_res.certificate_source, "value") else str(rec_res.certificate_source)

            return MethodSolveResult(
                method_name="Certified_Recovery",
                instance_id=inst.metadata.instance_id,
                n_assets=N,
                horizon=H,
                candidate_size=rec_res.final_budget,
                status=status,
                objective_value=rec_res.final_objective,
                best_bound=rec_res.lower_bound_best,
                mip_gap=(sol.mip_gap if sol else 0.0),
                node_count=sum(st.node_count for st in rec_res.stage_telemetry),
                work=(sol.work if sol else 0.0),
                timers=timers,
                recovery_stages=rec_res.stages_executed,
                certificate_source=cert_src,
                relaxation_certified=rec_res.relaxation_certified,
                full_mip_certified=rec_res.full_mip_certified,
                any_valid_certificate=rec_res.any_valid_certificate,
                relaxation_cert_gap=rec_res.relaxation_certificate_gap,
            )
        except gp.GurobiError as e:
            timers.t_solver_sec = time.perf_counter() - t0_rec
            status = classify_solver_outcome(False, False, False, 0, str(e))
            return MethodSolveResult(
                method_name="Certified_Recovery",
                instance_id=inst.metadata.instance_id,
                n_assets=N,
                horizon=H,
                candidate_size=nested_budgets[0],
                status=status,
                objective_value=None,
                best_bound=None,
                mip_gap=None,
                node_count=0,
                work=0.0,
                timers=timers,
            )

    def solve_oracle_support(
        self,
        instance: OptimizationInstance,
        teacher_support_or_y: Optional[Union[set[int], np.ndarray]] = None,
        teacher_support: Optional[set[int]] = None,
    ) -> MethodSolveResult:
        """
        M6: Oracle-Support Reduced MIQP.
        OFFLINE DIAGNOSTIC ONLY. Perfect candidate prediction bound.
        Explicitly marked is_deployable=False, uses_teacher_information=True.

        Constructs true oracle candidate universe:
            S_t^* = {i : exists h, y^*_{i,h} > 0.5} (or passed support set)
            M_t = {i : w_bar_{i,t} > 1e-6}
            C_t^{oracle} = S_t^* union M_t
        Never truncates or bounds |C_t^{oracle}| smaller than |S_t^* union M_t|.
        """
        inst = instance
        N = inst.market_state.n_assets
        H = inst.params.horizon
        timers = E2ETimerBreakdown()

        # Handle backward-compatible argument
        target_input = teacher_support_or_y if teacher_support_or_y is not None else teacher_support
        if target_input is None:
            target_input = set()

        # 1. Extract true teacher horizon-union support S_t^*
        if isinstance(target_input, np.ndarray):
            # y has shape (H, N)
            union_active = np.any(target_input > 0.5, axis=0)
            teacher_support_s = set(np.where(union_active)[0].tolist())
        else:
            teacher_support_s = set(target_input)

        # 2. Extract mandatory currently held assets M_t
        held_indices = set(np.where(inst.actual_state.normalized_weights > 1e-6)[0].tolist())

        # 3. Construct exact Oracle Universe: C_t^{oracle} = S_t^* union M_t
        oracle_indices = teacher_support_s | held_indices
        if not oracle_indices:
            oracle_indices = {0}

        cand = CandidateUniverse.from_indices(oracle_indices, full_universe_size=N)

        t0_solver = time.perf_counter()
        red_opt = MultiPeriodReducedMIQPOptimizer(inst.params, self.solver_config)
        try:
            red_exec = red_opt.solve(inst.market_state, cand, expand_solution=True)
            timers.t_build_sec = red_exec.build_time
            timers.t_solver_sec = red_exec.solver_time

            sol = red_exec.optimization_result
            status = classify_solver_outcome(
                is_optimal=sol.is_optimal,
                is_feasible=sol.is_feasible,
                has_incumbent=sol.objective_value is not None,
                gurobi_status=sol.status,
            )
            return MethodSolveResult(
                method_name="Oracle_Diagnostic",
                instance_id=inst.metadata.instance_id,
                n_assets=N,
                horizon=H,
                candidate_size=cand.candidate_size,
                status=status,
                objective_value=sol.objective_value,
                best_bound=sol.best_bound,
                mip_gap=sol.mip_gap,
                node_count=sol.node_count or 0,
                work=sol.work or 0.0,
                timers=timers,
                is_deployable=False,  # CRITICAL: Offline diagnostic only
                uses_teacher_information=True,
            )
        except gp.GurobiError as e:
            timers.t_solver_sec = time.perf_counter() - t0_solver
            status = classify_solver_outcome(False, False, False, 0, str(e))
            return MethodSolveResult(
                method_name="Oracle_Diagnostic",
                instance_id=inst.metadata.instance_id,
                n_assets=N,
                horizon=H,
                candidate_size=cand.candidate_size,
                status=status,
                objective_value=None,
                best_bound=None,
                mip_gap=None,
                node_count=0,
                work=0.0,
                timers=timers,
                is_deployable=False,
                uses_teacher_information=True,
            )
