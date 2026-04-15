# References and upstream attribution

This file is part of the **public reproduction** bundle. It points readers to **ANDES** (the simulator used for trajectory generation in the associated study) and to **representative work by Prof. Spyros Chatzivasileiadis** and collaborators in machine learning and data-driven power systems—listed here for **scholarly context** and **credit**. Inclusion does **not** imply that those authors endorse or maintain this repository.

**This study’s code and frozen splits:** **GitHub** — [`https://github.com/albert8943/unified-physics-informed-neural-surrogate-smib-tsa-cct-benchmark`](https://github.com/albert8943/unified-physics-informed-neural-surrogate-smib-tsa-cct-benchmark); **Zenodo Tier 1** — [`https://doi.org/10.5281/zenodo.19562416`](https://doi.org/10.5281/zenodo.19562416).

---

## ANDES (simulation software)

**ANDES** is an open-source toolbox for symbolic power-system modeling and time-domain simulation.

| Resource | Link |
|----------|------|
| **Source repository** | [https://github.com/CURENT/andes](https://github.com/CURENT/andes) |
| **Documentation** | [https://docs.andes.app/en/latest/index.html](https://docs.andes.app/en/latest/index.html) |

**Primary academic reference for the ANDES framework:**

- H. Cui, F. Li, and K. Tomsovic, “Hybrid Symbolic-Numeric Framework for Power System Modeling and Analysis,” *IEEE Transactions on Power Systems*, vol. 36, no. 2, pp. 1373–1384, Mar. 2021. DOI: [10.1109/TPWRS.2020.3017019](https://doi.org/10.1109/TPWRS.2020.3017019) — IEEE Xplore: [link](https://ieeexplore.ieee.org/document/9169830).

Path A in `reproduction_steps.md` uses ANDES as described in the companion IEEE Access manuscript and in `dependency_versions.md` (version / patch pins).

---

## Prof. Spyros Chatzivasileiadis (DTU) — official profiles

Prof. Chatzivasileiadis leads research at the intersection of **power systems**, **optimization**, and **machine learning**. Official pages:

- **DTU (English):** [https://www.dtu.dk/english/person/spyros-chatzivasileiadis?id=113161](https://www.dtu.dk/english/person/spyros-chatzivasileiadis?id=113161)
- **DTU ORBIT (publications):** [https://orbit.dtu.dk/en/persons/spyros-chatzivasileiadis/](https://orbit.dtu.dk/en/persons/spyros-chatzivasileiadis/)

---

## Representative publications (examples cited in the manuscript bibliography)

These entries illustrate the **wider literature** on data-driven security assessment and machine learning in power systems; the companion paper’s `access.bib` may list additional sources.

1. S. Chatzivasileiadis, A. Venzke, J. Stiasny, and G. S. Misyris, “Machine Learning in Power Systems: Is It Time to Trust It?,” *IEEE Power and Energy Magazine*, vol. 20, no. 3, pp. 32–41, May 2022. DOI: [10.1109/MPE.2022.3150810](https://doi.org/10.1109/MPE.2022.3150810) — [IEEE Xplore](https://ieeexplore.ieee.org/document/9761145).

2. F. Thams, A. Venzke, R. Eriksson, and S. Chatzivasileiadis, “Efficient Database Generation for Data-Driven Security Assessment of Power Systems,” *IEEE Transactions on Power Systems*, vol. 35, no. 1, pp. 30–41, Jan. 2020. DOI: [10.1109/TPWRS.2018.2890769](https://doi.org/10.1109/TPWRS.2018.2890769) — [IEEE Xplore](https://ieeexplore.ieee.org/document/8600355).

3. A. Venzke, D. K. Molzahn, and S. Chatzivasileiadis, “Efficient Creation of Datasets for Data-Driven Power System Applications,” *Electric Power Systems Research*, vol. 190, 106614, 2021. DOI: [10.1016/j.epsr.2020.106614](https://doi.org/10.1016/j.epsr.2020.106614) — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0378779620304181).

4. P. Ellinas, I. Karampinis, I. Ventura Nadal, R. Nellikkath, J. Vorwerk, and S. Chatzivasileiadis, “Physics-informed machine learning for power system dynamics: A framework incorporating trustworthiness,” *Sustainable Energy, Grids and Networks*, vol. 43, 101818, 2025. DOI: [10.1016/j.segan.2025.101818](https://doi.org/10.1016/j.segan.2025.101818) — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352467725002000).

For the **full** reference list as formatted in the article, see the IEEE Access submission materials (not all are duplicated here).
