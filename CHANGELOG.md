# Changelog & Iteration Progression Log

All notable changes and architectural iterations to `RailRouteAgent` are documented here following the micro1 Hackathon evaluation structure.

---

## 📈 System Iteration Progression

| Iteration | Approach & Architecture | Key Results & Quantitative Metrics | Decision & Rationale |
| :--- | :--- | :--- | :--- |
| **Iteration 0 (Baseline)** | Single zero-shot LLM prompt injecting raw JSON schedules. No tools or verification loops. | **Viable Routes Found Rate**: 100.0%<br>**Feasibility Pass Rate**: **0.0%**<br>**Avg Confirmation Prob**: 0.6817 | **Rejected**: LLMs fail at deterministic temporal arithmetic, consistently accepting 10-minute layovers or negative safety margins. Need deterministic tools. |
| **Iteration 1** | Added NetworkX graph routing tools (`find_split_junctions`) to `PlannerAgent`. | **Viable Routes Found Rate**: 100.0%<br>**Feasibility Pass Rate**: 25.0%<br>**Hallucination Rate**: 0.0% | **Refined**: Found real timetable connections, but still suggested tight 5-minute layovers without delay awareness. Need a Critic loop. |
| **Iteration 2** | Introduced `VerifierAgent` with dynamic junction transfer buffers & NTES delay risk tools (`calculate_connection_risk`). | **Viable Routes Found Rate**: 100.0%<br>**Feasibility Pass Rate**: **100.0%**<br>**Hallucination Rate**: 0.0% | **Kept**: Feasibility pass rate jumped to 100.0%. Automatically rejected impossible layovers (`TC-11`, `TC-12`) via reflection critique loops. |
| **Iteration 3 (Final)** | Added `RankingAgent` with pure-Python Quantile ML PNR confirmation scorer (`predict_pnr_confirmation`). | **Viable Routes Found Rate**: 100.0%<br>**Feasibility Pass Rate**: **100.0%**<br>**Avg Confirmation Prob**: **0.6483** | **Kept**: Ranks operationally safe itineraries. The Agent's average confirmation probability (0.6483) is purposely lower than the zero-shot baseline (0.6817) because the baseline falsely maximized probability by accepting impossible 5-minute cross-platform transfers. The Agent filtered out these physical impossibilities, sacrificing theoretical probability for operational reality. |

---

## 🔬 Benchmark Performance Summary

| Metric | Zero-Shot LLM (Iter 0) | RailRouteAgent (Iter 3) | Difference |
| :--- | :---: | :---: | :---: |
| **Viable Route Found Rate (%)** | 100.0% | 100.0% | +0.0% |
| **Operational Feasibility Pass Rate (%)** | **0.0%** | **100.0%** | **+100.0%** |
| **Hallucination / Invalid Connection Rate (%)** | 0.0% | 0.0% | +0.0% |
| **Average Confirmation Probability** | **0.6817** | **0.6483** | **-0.0334** *(Realistic Filter)* |

---

## 🔥 The Hot Take (System Design Insights)

> [!IMPORTANT]
> **The Hot Take**: *Agents confidently fail when they trust pure graph connectivity over operational reality. Giving an LLM agent "access to a schedule" is useless without giving it a "Verifier" that understands delay variance and physical transfer constraints.*
>
> In real-world Indian Railways operations, the zero-shot baseline falsely maximized confirmation probability (0.6817) by selecting physically impossible 5-minute cross-platform transfers. By enforcing a grounded multi-agent reflection loop (Planner $\rightarrow$ Verifier $\rightarrow$ Ranker) anchored by deterministic risk tools, `RailRouteAgent` filters out these physical impossibilities, achieving a **100.0% Operational Feasibility Pass Rate** while grounding confirmation probability in true operational reality (0.6483).
