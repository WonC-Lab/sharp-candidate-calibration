import sys

def update_manuscript():
    with open('manuscript/main.tex', 'r', encoding='utf-8') as f:
        text = f.read()

    # 1. Preamble math commands
    old_preamble_cmd = r"""\newcommand{\LBrelax}{\mathrm{LB}^{\mathrm{relax}}}"""
    new_preamble_cmd = r"""\newcommand{\LBfullrelax}{\mathrm{LB}_{\mathrm{full}}^{\mathrm{relax}}}
\newcommand{\LBrelax}{\mathrm{LB}_{\mathrm{full}}^{\mathrm{relax}}}"""
    if old_preamble_cmd in text:
        text = text.replace(old_preamble_cmd, new_preamble_cmd, 1)

    # 2. Section 8 replacement via exact string markers
    sec8_start_marker = r"\section{Controlled Computational Experiments \& Theory Validation}"
    sec9_start_marker = r"\section{Real-Market Operational Validation}"

    idx8 = text.find(sec8_start_marker)
    idx9 = text.find(sec9_start_marker)

    if idx8 != -1 and idx9 != -1:
        new_sec8_content = r"""\section{Controlled Computational Experiments \& Theory Validation}
\label{sec:controlled_exp}

We validate the theoretical framework across controlled combinatorial facility location benchmarks and multi-period portfolio MIQPs. All reported runtimes and speedup metrics are programmatically verified against raw solver logs under machine-precision arithmetic identities.

\paragraph{Primary vs.\ Secondary Speedup Conventions.}
Throughout this section, aggregate computational efficiency is reported primarily via the \emph{Ratio of Means}:
\begin{equation}
    \mathrm{Speedup}_{\mathrm{E2E}} \coloneqq \frac{\overline{T}_{\mathrm{full}}}{\overline{T}_{\mathrm{total}}} = \frac{\frac{1}{K}\sum_{i=1}^K T_{\mathrm{full},i}}{\frac{1}{K}\sum_{i=1}^K T_{\mathrm{total},i}},
\end{equation}
which measures overall system throughput across a workload. For completeness, we also report the \emph{Mean of Ratios} ($\frac{1}{K}\sum_{i=1}^K \frac{T_{\mathrm{full},i}}{T_{\mathrm{total},i}}$) in summary tables as a secondary metric of instance-level speedup distribution.

\subsection{Computational Scaling and Candidate Compression in CFLP}
\label{sec:results_cflp_scaling}

Table~\ref{tab:scaled_cflp_audited} documents runtime scaling, candidate space compression, and recovery behavior across facility universes up to $N=50$ (1{,}550 variables, 80 constraints). For each scale, candidate selection was executed using conformal calibration initialized at nominal 80\% coverage ($\alpha = 0.20$). To eliminate ambiguity, we distinguish the calibration-stage full-universe selection rate ($\mathrm{InitialFullRate} \coloneqq \mathbb{P}(B_1 = N)$) from the recovery-driven fallback rate ($\mathrm{RecoveryToFullRate} \coloneqq \mathbb{P}(B_1 < N \text{ and } B_{\mathrm{term}} = N)$).

\begin{table}[htbp]
\centering
\small
\caption{Computational scaling, candidate compression, and certified recovery metrics across facility universes $N \in \{8, 20, 30, 50\}$. All runtimes are in milliseconds averaged across test instances. Ratio of Means denotes $\overline{T}_{\mathrm{full}}/\overline{T}_{\mathrm{total}}$, while Mean of Ratios denotes $\frac{1}{K}\sum (T_{\mathrm{full},i}/T_{\mathrm{total},i})$.}
\label{tab:scaled_cflp_audited}
\resizebox{\textwidth}{!}{%
\begin{tabular}{rccccccccccc}
\toprule
$N$ & Vars & Constrs & Mean $B/N$ & Med $B/N$ & Q90 $B/N$ & InitialFull & RecoveryToFull & $T_{\mathrm{full}}$ (ms) & $T_{\mathrm{cand}}$ (ms) & $T_{\mathrm{total}}$ (ms) & Speedup (Ratio of Means) \\
\midrule
 8  &  128 &  23 & 1.000 & 1.000 & 1.000 &   0.0\% &  0.0\% &  10.44 &  10.33 &  10.47 & 0.997$\times$ (MeanOfRatios: 0.995$\times$) \\
20  &  420 &  40 & 0.850 & 0.850 & 0.850 &   0.0\% & 25.0\% &  45.85 &  32.08 &  42.37 & 1.082$\times$ (MeanOfRatios: 1.307$\times$) \\
30  &  780 &  55 & 0.867 & 0.867 & 0.867 &   0.0\% & 25.0\% &  74.87 &  67.82 &  87.83 & 0.852$\times$ (MeanOfRatios: 0.947$\times$) \\
50  & 1550 &  80 & 0.840 & 0.840 & 0.840 &   0.0\% &  0.0\% & 133.47 & 112.90 & 113.09 & 1.180$\times$ (MeanOfRatios: 1.160$\times$) \\
\bottomrule
\end{tabular}%
}
\end{table}

At small scale ($N=8$), the full combinatorial problem is solved in $10.44$\,ms, rendering feature extraction and ranking overhead comparable to candidate solve gains ($T_{\mathrm{total}} = 10.47$\,ms, $0.997\times$ speedup). As the universe expands to $N=20$, the pipeline achieves candidate dimension compression to $B/N = 0.850$, reducing candidate solver time from $45.85$\,ms to $32.08$\,ms. On instances certified without fallback, instance-level speedups reach up to $1.49\times$, yielding a mean of per-instance speedup ratios of $1.307\times$. However, on 25\% of instances, certified recovery was triggered, adding an average recovery overhead of $10.10$\,ms and yielding an aggregate throughput speedup (ratio of means) of $1.082\times$.

Importantly, our empirical scaling reveals a non-monotonic runtime profile at intermediate scale. At $N=30$, while initial candidate solves achieved a modest acceleration from $74.87$\,ms to $67.82$\,ms, 25\% of instances failed Stage~1 certification and executed multiple expansion steps along the geometric ladder. This recovery process incurred an average overhead of $T_{\mathrm{recovery}} = 19.86$\,ms, resulting in an end-to-end slowdown ($\overline{T}_{\mathrm{full}}/\overline{T}_{\mathrm{total}} = 0.852\times$). As problem scale expands further to $N=50$, where branch-and-bound combinatorial complexity dominates overheads, candidate reduction achieves $B/N = 0.840$ with zero recovery fallbacks, delivering an aggregate end-to-end speedup of $1.180\times$ (solver-only speedup of $1.182\times$).

\subsection{Operational Compression and Speedup Frontiers: Fast Mode vs.\ Certified Mode}
\label{sec:results_frontiers}

Table~\ref{tab:target_specific_frontier} reports the operational trade-off between nominal calibration confidence $1-\alpha$ and candidate compression across certification tolerances $\varepsilon_{\mathrm{cert}} \in \{10^{-4}, 5\times 10^{-4}, 10^{-3}\}$. To avoid conflating runtime semantics, we strictly distinguish:
\begin{itemize}[noitemsep]
  \item \textbf{True Fast Mode}: Calibrated purely against reference fidelity ($J_\theta(\mathcal{C}_\theta(B)) - J_\theta^{\mathrm{ref}} \le \varepsilon_{\mathrm{ref}}$ with $\varepsilon_{\mathrm{ref}} = 10^{-4}$). The solver terminates immediately after the first candidate solve without evaluating continuous relaxation bounds or executing recovery ladders. Its runtime is mathematically independent of $\varepsilon_{\mathrm{cert}}$.
  \item \textbf{Certified Mode}: Calibrated against certified suboptimality ($\mathrm{UB} - \LBfullrelax \le \varepsilon_{\mathrm{cert}}$) and backed by the geometric certified recovery ladder.
\end{itemize}

\begin{table}[htbp]
\centering
\small
\caption{Operational candidate compression and speedup frontiers under True Fast Mode (reference fidelity $\varepsilon_{\mathrm{ref}}=10^{-4}$) versus Certified Mode ($\varepsilon_{\mathrm{cert}}$ termination). Evaluated on $N=10$ benchmark instances. All speedups report the primary Ratio of Means.}
\label{tab:target_specific_frontier}
\resizebox{\textwidth}{!}{%
\begin{tabular}{cccccccccccc}
\toprule
\multirow{2}{*}{$\varepsilon_{\mathrm{cert}}$} & \multirow{2}{*}{$\alpha$} & Nominal & \multicolumn{4}{c}{True Fast Mode (Reference Target)} & \multicolumn{4}{c}{Certified Mode ($\varepsilon_{\mathrm{cert}}$ Target)} \\
\cmidrule(lr){4-7} \cmidrule(lr){8-11}
 & & Conf. & Coverage & Mean $B/N$ & InitialFull & Speedup & Coverage & Mean $B/N$ & RecoveryToFull & Speedup \\
\midrule
\multirow{3}{*}{$1.0 \times 10^{-4}$} & 0.20 & 80\% & 80.0\% & 0.80 &   0.0\% & 1.47$\times$ &  80.0\% & 0.80 &   0.0\% & 1.25$\times$ \\
                                      & 0.10 & 90\% & 95.0\% & 0.90 &   0.0\% & 1.17$\times$ & 100.0\% & 1.00 & 100.0\% & 1.10$\times$ \\
                                      & 0.05 & 95\% & 100.0\% & 1.00 & 100.0\% & 1.03$\times$ & 100.0\% & 1.00 & 100.0\% & 1.08$\times$ \\
\midrule
\multirow{3}{*}{$5.0 \times 10^{-4}$} & 0.20 & 80\% & 80.0\% & 0.80 &   0.0\% & 1.47$\times$ &  80.0\% & 0.80 &   0.0\% & 1.75$\times$ \\
                                      & 0.10 & 90\% & 95.0\% & 0.90 &   0.0\% & 1.17$\times$ & 100.0\% & 1.00 & 100.0\% & 1.05$\times$ \\
                                      & 0.05 & 95\% & 100.0\% & 1.00 & 100.0\% & 1.03$\times$ & 100.0\% & 1.00 & 100.0\% & 1.03$\times$ \\
\midrule
\multirow{3}{*}{$1.0 \times 10^{-3}$} & 0.20 & 80\% & 80.0\% & 0.80 &   0.0\% & 1.47$\times$ &  80.0\% & 0.80 &   0.0\% & 1.42$\times$ \\
                                      & 0.10 & 90\% & 95.0\% & 0.90 &   0.0\% & 1.17$\times$ & 100.0\% & 1.00 & 100.0\% & 0.95$\times$ \\
                                      & 0.05 & 95\% & 100.0\% & 1.00 & 100.0\% & 1.03$\times$ & 100.0\% & 1.00 & 100.0\% & 1.00$\times$ \\
\bottomrule
\end{tabular}%
}
\end{table}

The empirical frontier demonstrates a fundamental structural insight:
Under True Fast Mode, candidate budgets scale smoothly with confidence, achieving $B/N = 0.80$ at 80\% confidence ($1.47\times$ speedup) and $B/N = 0.90$ at 90\% confidence ($1.17\times$ speedup). 

In contrast, under Certified Mode, the pipeline exhibits a sharp phase transition. While useful dimension compression survives at 80\% confidence ($B/N = 0.80$, 0\% fallback), elevating nominal confidence to 90\% and 95\% forces the conformal calibration quantile to the universe boundary, collapsing compression entirely ($B/N = 1.00$, 100\% fallback rate). In this regime, achieving non-asymptotic mathematical certificates at extreme confidence requires evaluating the full combinatorial universe.

\subsection{Integer-Lattice-Certified Oracle Exactness on CFLP}
\label{sec:results_exact_oracle}

To audit whether conformal coverage guarantees hold against genuine mathematical ground truth rather than heuristics, we deployed the integer-parametric CFLP benchmark established in Proposition~\ref{prop:cflp_integrality} (Appendix~\ref{app:proof_cflp_integrality}). Because all optimal objectives are strictly integer-valued, any feasible upper bound satisfying:
\begin{equation}
    \mathrm{UB}_{\mathrm{int}} - \mathrm{LB} < 1 - \tau_{\mathrm{num}}
\end{equation}
provides an integer-lattice objective certificate under the documented solver-bound and numerical safety tolerance $\tau_{\mathrm{num}} = 0.05$ (implementation threshold $0.95$, reflecting a conservative numerical margin against floating-point inaccuracies).

\begin{table}[htbp]
\centering
\small
\caption{Integer-lattice-certified exact-oracle conformal exactness audit across $M=30$ independent replications on integer-parametric CFLP ($N=10$). The table strictly distinguishes the 95\% bootstrap confidence interval for the Monte Carlo mean coverage from the empirical replication-level quantiles ($Q_{05}, Q_{50}, Q_{95}$).}
\label{tab:exact_oracle_audited}
\resizebox{\textwidth}{!}{%
\begin{tabular}{cccccccccc}
\toprule
$\alpha$ & Target & Mean Coverage & Replication SD & SE of Mean & 95\% Bootstrap CI of Mean & $Q_{05}$ & $Q_{50}$ & $Q_{95}$ & Mean $B/N$ \\
\midrule
0.20 & 80.0\% & \textbf{89.00\%} & 15.17\% & 2.77\% & \textbf{[83.33\%, 94.00\%]} & 60.0\% & 100.0\% & 100.0\% & 92.5\% \\
0.10 & 90.0\% & \textbf{98.00\%} &  8.05\% & 1.47\% & \textbf{[94.67\%, 100.0\%]} & 89.0\% & 100.0\% & 100.0\% & 99.2\% \\
0.05 & 95.0\% & \textbf{100.00\%} &  0.00\% & 0.00\% & \textbf{[100.0\%, 100.0\%]} & 100.0\% & 100.0\% & 100.0\% & 100.0\% \\
\bottomrule
\end{tabular}%
}
\end{table}

Table~\ref{tab:exact_oracle_audited} details the results across $M=30$ independent calibration/test replications. Observed integer-lattice-certified oracle coverage was consistent with and conservative relative to the finite-sample guarantee under the exchangeable benchmark design. At nominal 80\% confidence ($\alpha = 0.20$), empirical oracle coverage achieved a mean of $89.00\%$ ($\mathrm{SE} = 2.77\%$), with a 95\% bootstrap confidence interval of $[83.33\%, 94.00\%]$. The lower confidence limit strictly exceeds the nominal 80\% target, empirically corroborating the conservative validity guaranteed by Theorem~\ref{thm:adaptive_exactness}. At $\alpha = 0.10$ and $\alpha = 0.05$, mean coverage reached $98.00\%$ (bootstrap CI $[94.67\%, 100.0\%]$) and $100.00\%$, respectively.

\subsection{Operational Candidate-Budget Efficiency Across Ranking Methods}
\label{sec:results_ranking}

Table~\ref{tab:ranking_efficiency} compares the operational candidate-budget efficiency across ranking methods. To preserve strict scientific accuracy, we distinguish proxy budgets from true active supports: $B^{\dagger,\mathrm{ref}}/N$ measures candidate dimension required to achieve one-sided reference fidelity ($J_\theta(\mathcal{C}_\theta(B)) - J_\theta^{\mathrm{ref}} \le 10^{-4}$), while $B^{\dagger,\mathrm{cert}}/N$ measures dimension required to satisfy certified tolerance ($\mathrm{UB} - \LBfullrelax \le 10^{-4}$). Smaller operational budgets indicate that the ranking exposes high-quality candidate entities earlier in the sequence.

\begin{table}[htbp]
\centering
\small
\caption{Operational candidate-budget efficiency across ranking methods ($N=10$ benchmark instances). Smaller operational budgets indicate that the ranking exposes high-quality candidate entities earlier in the sequence.}
\label{tab:ranking_efficiency}
\begin{tabular}{llccccc}
\toprule
Ranking Method & Metric & Mean & Median & Q75 & Q90 & Max \\
\midrule
\multirow{2}{*}{Efficiency Ranking (Ours)} & $B^{\dagger,\mathrm{ref}} / N$  & \textbf{0.625} & \textbf{0.600} & 0.700 & 0.805 & 0.950 \\
                                           & $B^{\dagger,\mathrm{cert}} / N$ & \textbf{0.629} & \textbf{0.600} & 0.700 & 0.805 & 1.000 \\
\midrule
\multirow{2}{*}{Random Ranking}            & $B^{\dagger,\mathrm{ref}} / N$  & 0.932 & 0.950 & 1.000 & 1.000 & 1.000 \\
                                           & $B^{\dagger,\mathrm{cert}} / N$ & 0.932 & 0.950 & 1.000 & 1.000 & 1.000 \\
\midrule
\multirow{2}{*}{Reverse Ranking}           & $B^{\dagger,\mathrm{ref}} / N$  & 0.999 & 1.000 & 1.000 & 1.000 & 1.000 \\
                                           & $B^{\dagger,\mathrm{cert}} / N$ & 0.999 & 1.000 & 1.000 & 1.000 & 1.000 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Perspective Relaxation: Bound Tightening vs.\ Wall-Clock Overhead}
\label{sec:results_perspective}

Table~\ref{tab:perspective_breakeven_audited} evaluates the end-to-end computational efficacy of perspective-strengthened continuous relaxation bounds against basic continuous relaxations across multi-period ($H=3, N=5$) and single-period ($H=1, N \in \{10, 20\}$) MIQPs.

\begin{table}[htbp]
\centering
\small
\caption{Audited perspective relaxation tightening, continuous solve overhead, and end-to-end wall-clock runtimes. Evaluated across multi-period and single-period benchmark regimes. All speedups report the primary Ratio of Means.}
\label{tab:perspective_breakeven_audited}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccccc}
\toprule
Regime & $\varepsilon_{\mathrm{cert}}$ & $\Delta \mathrm{LB}$ & Overhead (ms) & Stage1Cert (Basic vs Persp) & RecoveryToFull (Basic vs Persp) & $T_{\mathrm{E2E}}^{\mathrm{basic}}$ & $T_{\mathrm{E2E}}^{\mathrm{persp}}$ & Speedup$_{\mathrm{persp}}$ & Break-Even? \\
\midrule
Regime A ($H=3, N=5$) & $10^{-4}$ & $1.77 \times 10^{-5}$ & $+1.65$ &   0.0\% vs   0.0\% & 100.0\% vs 100.0\% & 23.97 & 25.63 & 0.936$\times$ & \textbf{No} ($<1.0$) \\
Regime A ($H=3, N=5$) & $5\times 10^{-4}$ & $1.77 \times 10^{-5}$ & $+1.65$ &  11.1\% vs  11.1\% &  88.9\% vs  88.9\% & 23.97 & 25.63 & 0.936$\times$ & \textbf{No} ($<1.0$) \\
Regime A ($H=3, N=5$) & $10^{-3}$ & $1.77 \times 10^{-5}$ & $+1.65$ &  44.4\% vs  44.4\% &  55.6\% vs  55.6\% & 23.97 & 25.63 & 0.936$\times$ & \textbf{No} ($<1.0$) \\
\midrule
Regime B ($H=1, N=10$) & $10^{-4}$ & $3.97 \times 10^{-5}$ & $+1.64$ &  56.0\% vs  64.0\% &  44.0\% vs  36.0\% &  7.89 &  9.52 & 0.828$\times$ & \textbf{No} ($<1.0$) \\
Regime B ($H=1, N=10$) & $5\times 10^{-4}$ & $3.97 \times 10^{-5}$ & $+1.64$ &  72.0\% vs  76.0\% &  28.0\% vs  24.0\% &  7.89 &  9.52 & 0.828$\times$ & \textbf{No} ($<1.0$) \\
Regime B ($H=1, N=10$) & $10^{-3}$ & $3.97 \times 10^{-5}$ & $+1.64$ &  76.0\% vs  76.0\% &  24.0\% vs  24.0\% &  7.89 &  9.52 & 0.828$\times$ & \textbf{No} ($<1.0$) \\
\midrule
Regime B ($H=1, N=20$) & $10^{-4}$ & $4.54 \times 10^{-6}$ & $+4.22$ &  96.0\% vs  96.0\% &   4.0\% vs   4.0\% & 14.12 & 18.33 & 0.770$\times$ & \textbf{No} ($<1.0$) \\
Regime B ($H=1, N=20$) & $5\times 10^{-4}$ & $4.54 \times 10^{-6}$ & $+4.22$ &  96.0\% vs  96.0\% &   4.0\% vs   4.0\% & 14.12 & 18.33 & 0.770$\times$ & \textbf{No} ($<1.0$) \\
Regime B ($H=1, N=20$) & $10^{-3}$ & $4.54 \times 10^{-6}$ & $+4.22$ &  96.0\% vs  96.0\% &   4.0\% vs   4.0\% & 14.12 & 18.33 & 0.770$\times$ & \textbf{No} ($<1.0$) \\
\bottomrule
\end{tabular}%
}
\end{table}

While perspective reformulation strictly tightens the continuous lower bound ($\Delta \mathrm{LB} \in [4.5\times 10^{-6}, 3.97\times 10^{-5}]$), solving the rotated second-order cone reformulation adds $+1.64$\,ms to $+4.22$\,ms of continuous relaxation solve overhead. In these small-to-moderate MIQPs, integer branch-and-bound solve times are on the order of 7--24\,ms; consequently, the continuous relaxation overhead exceeds the computational savings achieved by early termination. Across all evaluated regimes, perspective speedups range from $0.770\times$ to $0.936\times$, demonstrating that break-even is not achieved at these dimensional scales.

\subsection{Synthetic Portfolio MIQP Finite-Sample Conformal Audit}
\label{sec:synthetic_portfolio_audit}

Table~\ref{tab:conformal_coverage} validates split-conformal calibration on $N=10, H=3$ synthetic portfolio instances across $\alpha \in \{0.20, 0.10, 0.05\}$.

\begin{table}[htbp]
\centering
\small
\caption{Finite-sample split-conformal exactness coverage audit on synthetic portfolio test instances ($N=10, H=3, n_{\mathrm{cal}}=12, n_{\mathrm{test}}=12$). Target exactness coverage is $1-\alpha$.}
\label{tab:conformal_coverage}
\resizebox{\textwidth}{!}{%
\begin{tabular}{ccccccr}
\toprule
$\alpha$ & Target Cov. & $k_{\mathrm{cal}} / n_{\mathrm{cal}}$ & Residual $q_\alpha$ & Method & Test Coverage & Mean Budget $B$ ($B/N$) \\
\midrule
\multirow{2}{*}{0.20} & \multirow{2}{*}{$\ge 80.0\%$} & \multirow{2}{*}{11 / 12} & \multirow{2}{*}{$-0.89$} & Adaptive $B_\alpha(X)$ & \textbf{91.7\%} & 3.50 (35.0\%) \\
                      &                                &                          &                          & Marginal $\widehat{B}_\alpha$ & \textbf{91.7\%} & 3.00 (30.0\%) \\
\midrule
\multirow{2}{*}{0.10} & \multirow{2}{*}{$\ge 90.0\%$} & \multirow{2}{*}{12 / 12} & \multirow{2}{*}{$+0.11$} & Adaptive $B_\alpha(X)$ & \textbf{100.0\%} & 4.00 (40.0\%) \\
                      &                                &                          &                          & Marginal $\widehat{B}_\alpha$ & \textbf{100.0\%} & 4.00 (40.0\%) \\
\midrule
\multirow{2}{*}{0.05} & \multirow{2}{*}{$\ge 95.0\%$} & \multirow{2}{*}{13 / 12} & \multirow{2}{*}{$+\infty$} & Adaptive $B_\alpha(X)$ & \textbf{100.0\%} & 10.00 (100.0\%) \\
                      &                                &                          & (InitialFull)            & Marginal $\widehat{B}_\alpha$ & \textbf{100.0\%} & 10.00 (100.0\%) \\
\bottomrule
\end{tabular}%
}
\end{table}

Observed test coverage strictly satisfies $\mathbb{P} \ge 1-\alpha$ across all evaluated $\alpha$. For $\alpha = 0.20$, adaptive calibration attains $91.7\%$ coverage using on average only $3.50$ assets ($35\%$ of universe). For $\alpha = 0.05$, because $k_{\mathrm{cal}} = 13 > 12$, the safe fallback rule automatically triggers $B = N$ ($\mathrm{InitialFullRate} = 100\%$), guaranteeing exactness as proved in Theorem~\ref{thm:adaptive_exactness}. Furthermore, across all instances with geometric ladder expansion ($B_1 = 3, r = 1.5$), observed recovery stages $\ell^*$ satisfied $\ell^* \le 1 + \lceil \log_r(\Bdag/B_1) \rceil_+$ on 100\% of instances (Theorem~\ref{thm:recovery_depth}), and cumulative dimensions strictly satisfied $\sum B_\ell < \frac{r^2}{r-1} \Bdag + \ell^*$ on 100\% of instances (Theorem~\ref{thm:structural_work}).

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{artifacts/plots/fig_empirical_frontier.png}
\caption{Audited empirical frontier across benchmark studies. \textbf{(A)}~Nominal confidence vs.\ calibrated candidate budget ratio $B/N$. \textbf{(B)}~Nominal confidence vs.\ full-universe fallback rates, showing the sharp phase transition to full fallback at 90--95\% confidence. \textbf{(C)}~Problem scale $N$ vs.\ end-to-end speedup in CFLP, visibly documenting the non-monotonic recovery slowdown at $N=30$ ($0.852\times$) and recovery at $N=50$ ($1.180\times$). \textbf{(D)}~Certification tolerance $\varepsilon_{\mathrm{cert}}$ vs.\ perspective E2E speedup, illustrating that continuous relaxation solve overhead prevents wall-clock break-even ($<1.0\times$) across tested scales.}
\label{fig:empirical_frontier_4panel}
\end{figure}

"""
        text = text[:idx8] + new_sec8_content + text[idx9:]
        print('Replaced Section 8 successfully!')
    else:
        print(f'Error: Could not locate Section 8 markers (idx8={idx8}, idx9={idx9})')

    with open('manuscript/main.tex', 'w', encoding='utf-8') as f:
        f.write(text)

    print('Finished writing main.tex.')

if __name__ == '__main__':
    update_manuscript()
