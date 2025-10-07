# TAG-K: Tail-Averaged Greedy Kaczmarz for Computationally Efficient and Performant Online Inertial Parameter Estimation 

**Title (ZH)**: TAG-K: 基于尾平均贪婪Kaczmarz的高效高性能在线惯性参数估计 

**Authors**: Shuo Sha, Anupam Bhakta, Zhenyuan Jiang, Kevin Qiu, Ishaan Mahajan, Gabriel Bravo, Brian Plancher  

**Link**: [PDF](https://arxiv.org/pdf/2510.04839)  

**Abstract**: Accurate online inertial parameter estimation is essential for adaptive robotic control, enabling real-time adjustment to payload changes, environmental interactions, and system wear. Traditional methods such as Recursive Least Squares (RLS) and the Kalman Filter (KF) often struggle to track abrupt parameter shifts or incur high computational costs, limiting their effectiveness in dynamic environments and for computationally constrained robotic systems. As such, we introduce TAG-K, a lightweight extension of the Kaczmarz method that combines greedy randomized row selection for rapid convergence with tail averaging for robustness under noise and inconsistency. This design enables fast, stable parameter adaptation while retaining the low per-iteration complexity inherent to the Kaczmarz framework. We evaluate TAG-K in synthetic benchmarks and quadrotor tracking tasks against RLS, KF, and other Kaczmarz variants. TAG-K achieves 1.5x-1.9x faster solve times on laptop-class CPUs and 4.8x-20.7x faster solve times on embedded microcontrollers. More importantly, these speedups are paired with improved resilience to measurement noise and a 25% reduction in estimation error, leading to nearly 2x better end-to-end tracking performance. 

**Abstract (ZH)**: 准确的在线惯性参数估计对于自适应机器人控制至关重要，能够实现实时调整载荷变化、环境交互和系统磨损。传统方法如递推最小二乘法（RLS）和卡尔曼滤波器（KF）往往难以追踪参数的突变或产生高计算成本，限制了它们在动态环境和计算受限的机器人系统中的有效性。因此，我们引入了TAG-K，这是一种轻量级的Kaczmarz方法扩展，结合了贪婪随机行选择以实现快速收敛，并使用尾端平均以增强抗噪和一致性能力。这种设计能够实现快速稳定地参数适应，同时保留Kaczmarz框架固有的每迭代低复杂度。我们通过合成基准和旋翼无人机跟踪任务将TAG-K与RLS、KF以及其他Kaczmarz变种进行了对比评估。在笔记本级别CPU上，TAG-K的求解时间快1.5至1.9倍，在嵌入式微控制器上快4.8至20.7倍。更重要的是，这些加速与对测量噪声的改进抵抗力和25%的估计误差降低相结合，导致端到端跟踪性能提高了近2倍。 

---
# PAD-TRO: Projection-Augmented Diffusion for Direct Trajectory Optimization 

**Title (ZH)**: PAD-TRO: 投影增强扩散直接轨迹优化 

**Authors**: Jushan Chen, Santiago Paternain  

**Link**: [PDF](https://arxiv.org/pdf/2510.04436)  

**Abstract**: Recently, diffusion models have gained popularity and attention in trajectory optimization due to their capability of modeling multi-modal probability distributions. However, addressing nonlinear equality constraints, i.e, dynamic feasi- bility, remains a great challenge in diffusion-based trajectory optimization. Recent diffusion-based trajectory optimization frameworks rely on a single-shooting style approach where the denoised control sequence is applied to forward propagate the dynamical system, which cannot explicitly enforce constraints on the states and frequently leads to sub-optimal solutions. In this work, we propose a novel direct trajectory optimization approach via model-based diffusion, which directly generates a sequence of states. To ensure dynamic feasibility, we propose a gradient-free projection mechanism that is incorporated into the reverse diffusion process. Our results show that, compared to a recent state-of-the-art baseline, our approach leads to zero dynamic feasibility error and approximately 4x higher success rate in a quadrotor waypoint navigation scenario involving dense static obstacles. 

**Abstract (ZH)**: 基于模型的扩散模型在动态可行性的直接轨迹优化方法 

---
# HEHA: Hierarchical Planning for Heterogeneous Multi-Robot Exploration of Unknown Environments 

**Title (ZH)**: HEHA：异质多机器人未知环境分级规划探索方法 

**Authors**: Longrui Yang, Yiyu Wang, Jingfan Tang, Yunpeng Lv, Shizhe Zhao, Chao Cao, Zhongqiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2510.04161)  

**Abstract**: This paper considers the path planning problem for autonomous exploration of an unknown environment using multiple heterogeneous robots such as drones, wheeled, and legged robots, which have different capabilities to traverse complex terrains. A key challenge there is to intelligently allocate the robots to the unknown areas to be explored and determine the visiting order of those spaces subject to traversablity constraints, which leads to a large scale constrained optimization problem that needs to be quickly and iteratively solved every time when new space are explored. To address the challenge, we propose HEHA (Hierarchical Exploration with Heterogeneous Agents) by leveraging a recent hierarchical method that decompose the exploration into global planning and local planning. The major contribution in HEHA is its global planning, where we propose a new routing algorithm PEAF (Partial Anytime Focal search) that can quickly find bounded sub-optimal solutions to minimize the maximum path length among the agents subject to traversability constraints. Additionally, the local planner in HEHA also considers heterogeneity to avoid repeated and duplicated exploration among the robots. The experimental results show that, our HEHA can reduce up to 30% of the exploration time than the baselines. 

**Abstract (ZH)**: 基于异构代理的分层探索路径规划方法HEHA 

---
# Robust Permissive Controller Synthesis for Interval MDPs 

**Title (ZH)**: 区间MDP中鲁棒许可控制器综合 

**Authors**: Khang Vo Huynh, David Parker, Lu Feng  

**Link**: [PDF](https://arxiv.org/pdf/2510.03481)  

**Abstract**: We address the problem of robust permissive controller synthesis for robots operating under uncertain dynamics, modeled as Interval Markov Decision Processes (IMDPs). IMDPs generalize standard MDPs by allowing transition probabilities to vary within intervals, capturing epistemic uncertainty from sensing noise, actuation imprecision, and coarse system abstractions-common in robotics. Traditional controller synthesis typically yields a single deterministic strategy, limiting adaptability. In contrast, permissive controllers (multi-strategies) allow multiple actions per state, enabling runtime flexibility and resilience. However, prior work on permissive controller synthesis generally assumes exact transition probabilities, which is unrealistic in many robotic applications. We present the first framework for robust permissive controller synthesis on IMDPs, guaranteeing that all strategies compliant with the synthesized multi-strategy satisfy reachability or reward-based specifications under all admissible transitions. We formulate the problem as mixed-integer linear programs (MILPs) and propose two encodings: a baseline vertex-enumeration method and a scalable duality-based method that avoids explicit enumeration. Experiments on four benchmark domains show that both methods synthesize robust, maximally permissive controllers and scale to large IMDPs with up to hundreds of thousands of states. 

**Abstract (ZH)**: 我们针对机器人在不确定动力学条件下操作的鲁棒容错控制器综合问题，将其建模为区间马尔可夫决策过程（IMDPs）。我们提出了一种框架，用于在IMDPs上进行鲁棒容错控制器综合，确保所有与合成的多策略兼容的策略，在所有容许的转换下都能满足可达性或基于奖励的规范。我们将问题形式化为混合整数线性规划（MILPs），并提出两种编码方法：基于顶点枚举的基本方法和一种避免显式枚举的可扩展对偶方法。实验结果表明，这两种方法都能合成出鲁棒性和容错性最高的控制器，并适用于多达数十万个状态的大型IMDPs。 

---
# Efficient Probabilistic Planning with Maximum-Coverage Distributionally Robust Backward Reachable Trees 

**Title (ZH)**: 高效的最大覆盖分布鲁棒逆向可达树概率规划 

**Authors**: Alex Rose, Naman Aggarwal, Christopher Jewison, Jonathan P. How  

**Link**: [PDF](https://arxiv.org/pdf/2510.04807)  

**Abstract**: This paper presents a new multi-query motion planning algorithm for linear Gaussian systems with the goal of reaching a Euclidean ball with high probability. We develop a new formulation for ball-shaped ambiguity sets of Gaussian distributions and leverage it to develop a distributionally robust belief roadmap construction algorithm. This algorithm synthe- sizes robust controllers which are certified to be safe for maximal size ball-shaped ambiguity sets of Gaussian distributions. Our algorithm achieves better coverage than the maximal coverage algorithm for planning over Gaussian distributions [1], and we identify mild conditions under which our algorithm achieves strictly better coverage. For the special case of no process noise or state constraints, we formally prove that our algorithm achieves maximal coverage. In addition, we present a second multi-query motion planning algorithm for linear Gaussian systems with the goal of reaching a region parameterized by the Minkowski sum of an ellipsoid and a Euclidean ball with high probability. This algorithm plans over ellipsoidal sets of maximal size ball-shaped ambiguity sets of Gaussian distributions, and provably achieves equal or better coverage than the best-known algorithm for planning over ellipsoidal ambiguity sets of Gaussian distributions [2]. We demonstrate the efficacy of both methods in a wide range of conditions via extensive simulation experiments. 

**Abstract (ZH)**: 一种针对线性高斯系统的新多查询运动规划算法：以高概率到达欧几里得球体 

---
# RAP: 3D Rasterization Augmented End-to-End Planning 

**Title (ZH)**: RAP: 3D 光栅化增强端到端规划 

**Authors**: Lan Feng, Yang Gao, Eloi Zablocki, Quanyi Li, Wuyang Li, Sichao Liu, Matthieu Cord, Alexandre Alahi  

**Link**: [PDF](https://arxiv.org/pdf/2510.04333)  

**Abstract**: Imitation learning for end-to-end driving trains policies only on expert demonstrations. Once deployed in a closed loop, such policies lack recovery data: small mistakes cannot be corrected and quickly compound into failures. A promising direction is to generate alternative viewpoints and trajectories beyond the logged path. Prior work explores photorealistic digital twins via neural rendering or game engines, but these methods are prohibitively slow and costly, and thus mainly used for evaluation. In this work, we argue that photorealism is unnecessary for training end-to-end planners. What matters is semantic fidelity and scalability: driving depends on geometry and dynamics, not textures or lighting. Motivated by this, we propose 3D Rasterization, which replaces costly rendering with lightweight rasterization of annotated primitives, enabling augmentations such as counterfactual recovery maneuvers and cross-agent view synthesis. To transfer these synthetic views effectively to real-world deployment, we introduce a Raster-to-Real feature-space alignment that bridges the sim-to-real gap. Together, these components form Rasterization Augmented Planning (RAP), a scalable data augmentation pipeline for planning. RAP achieves state-of-the-art closed-loop robustness and long-tail generalization, ranking first on four major benchmarks: NAVSIM v1/v2, Waymo Open Dataset Vision-based E2E Driving, and Bench2Drive. Our results show that lightweight rasterization with feature alignment suffices to scale E2E training, offering a practical alternative to photorealistic rendering. Project page: this https URL. 

**Abstract (ZH)**: 基于模仿学习的端到端驾驶训练仅依赖于专家演示。部署成闭环后，这类策略缺乏恢复数据：小错误无法纠正并迅速复合成失败。一种有前景的方向是生成超越记录路径的替代视点和轨迹。先前的工作通过神经渲染或游戏引擎探索逼真的数字孪生，但这些方法耗时且成本高昂，因此主要用于评估。在本文中，我们argue逼真度并不是训练端到端规划器所需要的。重要的是语义保真度和扩展性：驾驶依赖于几何和动力学，而非纹理或照明。受此启发，我们提出了3D栅格化，用轻量级的标注原语的栅格化替代昂贵的渲染，使counterfactual恢复机动和跨agent视图合成成为可能。为有效地将这些合成视图转移到实际部署中，我们引入了一种栅格化到现实的特征空间对齐，以弥合仿真到现实的差距。这些组件共同构成了栅格化增强规划（RAP），这是一种扩展性的规划数据增强流水线。RAP在闭环稳健性和长尾泛化方面达到了最佳效果，在四个主要基准测试中排名第一：NASIM v1/v2、Waymo开放数据集基于视觉的端到端驾驶以及Bench2Drive。我们的结果显示，带有特征对齐的轻量级栅格化足以扩展端到端训练，为逼真渲染提供了一个实用的替代方案。项目页面：this https URL。 

---
# A KL-regularization framework for learning to plan with adaptive priors 

**Title (ZH)**: 一种带有自适应先验的计划学习KL正则化框架 

**Authors**: Álvaro Serra-Gomez, Daniel Jarne Ornia, Dhruva Tirumala, Thomas Moerland  

**Link**: [PDF](https://arxiv.org/pdf/2510.04280)  

**Abstract**: Effective exploration remains a central challenge in model-based reinforcement learning (MBRL), particularly in high-dimensional continuous control tasks where sample efficiency is crucial. A prominent line of recent work leverages learned policies as proposal distributions for Model-Predictive Path Integral (MPPI) planning. Initial approaches update the sampling policy independently of the planner distribution, typically maximizing a learned value function with deterministic policy gradient and entropy regularization. However, because the states encountered during training depend on the MPPI planner, aligning the sampling policy with the planner improves the accuracy of value estimation and long-term performance. To this end, recent methods update the sampling policy by minimizing KL divergence to the planner distribution or by introducing planner-guided regularization into the policy update. In this work, we unify these MPPI-based reinforcement learning methods under a single framework by introducing Policy Optimization-Model Predictive Control (PO-MPC), a family of KL-regularized MBRL methods that integrate the planner's action distribution as a prior in policy optimization. By aligning the learned policy with the planner's behavior, PO-MPC allows more flexibility in the policy updates to trade off Return maximization and KL divergence minimization. We clarify how prior approaches emerge as special cases of this family, and we explore previously unstudied variations. Our experiments show that these extended configurations yield significant performance improvements, advancing the state of the art in MPPI-based RL. 

**Abstract (ZH)**: 基于模型的强化学习中有效的探索仍然是一个核心挑战，特别是在高维度连续控制任务中，样本效率至关重要。近期的一项主要研究方向是利用学习到的策略作为Model-Predictive Path Integral (MPPI) 规划的提议分布。早期的方法独立地更新采样策略和规划分布，通常通过确定性策略梯度和熵正则化最大化学习的价值函数。然而，由于训练过程中遇到的状态依赖于MPPI规划器，使采样策略与规划器对齐可以提高价值估计的准确性及长期性能。为此，近期的方法通过最小化KL散度到规划分布或在策略更新中引入规划器指导的正则化来更新采样策略。在本文中，我们通过引入Policy Optimization-Model Predictive Control (PO-MPC) 方法，将这些基于MPPI的强化学习方法统一到一个框架下，PO-MPC是一种KL正则化的基于模型的强化学习方法，将规划器的动作分布作为策略优化中的先验。通过使学习到的策略与规划器的行为对齐，PO-MPC在策略更新中提供更多灵活性，以权衡回报最大化和KL散度最小化。我们明确了先前方法作为该家族的特例，并探索了未被研究的变体。实验结果显示，这些扩展配置显著提高了性能，推动了基于MPPI的强化学习技术的发展。 

---
# Agile Tradespace Exploration for Space Rendezvous Mission Design via Transformers 

**Title (ZH)**: 基于Transformer的敏捷 tradespace 探索方法在空间对接任务设计中的应用 

**Authors**: Yuji Takubo, Daniele Gammelli, Marco Pavone, Simone D'Amico  

**Link**: [PDF](https://arxiv.org/pdf/2510.03544)  

**Abstract**: Spacecraft rendezvous enables on-orbit servicing, debris removal, and crewed docking, forming the foundation for a scalable space economy. Designing such missions requires rapid exploration of the tradespace between control cost and flight time across multiple candidate targets. However, multi-objective optimization in this setting is challenging, as the underlying constraints are often highly nonconvex, and mission designers must balance accuracy (e.g., solving the full problem) with efficiency (e.g., convex relaxations), slowing iteration and limiting design agility. To address these challenges, this paper proposes an AI-powered framework that enables agile mission design for a wide range of Earth orbit rendezvous scenarios. Given the orbital information of the target spacecraft, boundary conditions, and a range of flight times, this work proposes a Transformer-based architecture that generates, in a single parallelized inference step, a set of near-Pareto optimal trajectories across varying flight times, thereby enabling rapid mission trade studies. The model is further extended to accommodate variable flight times and perturbed orbital dynamics, supporting realistic multi-objective trade-offs. Validation on chance-constrained rendezvous problems with passive safety constraints demonstrates that the model generalizes across both flight times and dynamics, consistently providing high-quality initial guesses that converge to superior solutions in fewer iterations. Moreover, the framework efficiently approximates the Pareto front, achieving runtimes comparable to convex relaxation by exploiting parallelized inference. Together, these results position the proposed framework as a practical surrogate for nonconvex trajectory generation and mark an important step toward AI-driven trajectory design for accelerating preliminary mission planning in real-world rendezvous applications. 

**Abstract (ZH)**: 基于AI的航天器交会智能设计框架：支持可扩展太空经济的敏捷任务规划 

---
# Real-Time Threaded Houbara Detection and Segmentation for Wildlife Conservation using Mobile Platforms 

**Title (ZH)**: 基于移动平台的实时线程化aho猎隼检测与分割用于野生动物保护 

**Authors**: Lyes Saad Saoud, Loic Lesobre, Enrico Sorato, Irfan Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2510.03501)  

**Abstract**: Real-time animal detection and segmentation in natural environments are vital for wildlife conservation, enabling non-invasive monitoring through remote camera streams. However, these tasks remain challenging due to limited computational resources and the cryptic appearance of many species. We propose a mobile-optimized two-stage deep learning framework that integrates a Threading Detection Model (TDM) to parallelize YOLOv10-based detection and MobileSAM-based segmentation. Unlike prior YOLO+SAM pipelines, our approach improves real-time performance by reducing latency through threading. YOLOv10 handles detection while MobileSAM performs lightweight segmentation, both executed concurrently for efficient resource use. On the cryptic Houbara Bustard, a conservation-priority species, our model achieves mAP50 of 0.9627, mAP75 of 0.7731, mAP95 of 0.7178, and a MobileSAM mIoU of 0.7421. YOLOv10 operates at 43.7 ms per frame, confirming real-time readiness. We introduce a curated Houbara dataset of 40,000 annotated images to support model training and evaluation across diverse conditions. The code and dataset used in this study are publicly available on GitHub at this https URL. For interactive demos and additional resources, visit this https URL. 

**Abstract (ZH)**: 自然环境中实时动物检测与分割对于野生动物保护至关重要，能通过远程相机流实现非侵入式监测。然而，这些任务由于计算资源有限和许多物种隐蔽的外观而具有挑战性。我们提出了一种针对移动设备优化的两阶段深度学习框架，结合了线程检测模型（TDM）以并行化基于YOLOv10的检测和基于MobileSAM的分割。与之前的YOLO+SAM管道不同，我们的方法通过降低延迟来提高实时性能。YOLOv10负责检测，MobileSAM执行轻量级分割，两者并发执行以高效利用资源。对于隐蔽的厚颈鸨这一保护优先物种，我们的模型实现了mAP50为0.9627、mAP75为0.7731、mAP95为0.7178，并且MobileSAM的mIoU为0.7421。YOLOv10每帧运行时间为43.7毫秒，证实了实时性能。我们提供了一个包含40,000张标注图像的厚颈鸨数据集，以支持模型训练和评估。本文中使用的代码和数据集在GitHub上公开，网址为：这个https URL。有关交互式演示和额外资源，请访问这个https URL。 

---
# Adaptive Cruise Control in Autonomous Vehicles: Challenges, Gaps, Comprehensive Review, and, Future Directions 

**Title (ZH)**: 自动驾驶车辆的自适应巡航控制：挑战、缺口、综合综述及未来方向 

**Authors**: Shradha Bavalatti, Yash Kangralkar, Santosh Pattar, Veena P Badiger  

**Link**: [PDF](https://arxiv.org/pdf/2510.03300)  

**Abstract**: The development of Autonomous Vehicles (AVs) has redefined the way of transportation by eliminating the need for human intervention in driving. This revolution is fueled by rapid advancements in adaptive cruise control (ACC), which make AVs capable of interpreting their surroundings and responding intelligently. While AVs offer significant advantages, such as enhanced safety and improved traffic efficiency, they also face several challenges that need to be addressed. Existing survey papers often lack a comprehensive analysis of these challenges and their potential solutions. Our paper stands out by meticulously identifying these gaps in current ACC research and offering impactful future directions to guide researchers in designing next-generation ACC systems. Our survey provides a detailed and systematic review, addressing the limitations of previous studies and proposing innovative approaches to achieve sustainable and fault-resilient urban transportation. 

**Abstract (ZH)**: 自主车辆的发展通过消除驾驶过程中的人为干预重新定义了交通方式。这种革命是由自适应巡航控制（ACC）的迅速进步推动的，使自主车辆能够解读其环境并做出智能响应。虽然自主车辆带来了诸如增强的安全性和提高的交通效率等优势，但它们也面临着需要解决的若干挑战。现有的综述论文往往缺乏对这些挑战及其潜在解决方案的全面分析。我们的论文通过细致地识别当前ACC研究中的空白区域，并提出具有影响的未来发展方向，以指导研究人员设计下一代ACC系统，脱颖而出。我们的综述提供了详细和系统的回顾，解决了先前研究的局限性，并提出了创新的方法以实现可持续和容错的城市交通。 

---
# Look-ahead Reasoning with a Learned Model in Imperfect Information Games 

**Title (ZH)**: 带有学习模型的展望推理在不完美信息游戏中 

**Authors**: Ondřej Kubíček, Viliam Lisý  

**Link**: [PDF](https://arxiv.org/pdf/2510.05048)  

**Abstract**: Test-time reasoning significantly enhances pre-trained AI agents' performance. However, it requires an explicit environment model, often unavailable or overly complex in real-world scenarios. While MuZero enables effective model learning for search in perfect information games, extending this paradigm to imperfect information games presents substantial challenges due to more nuanced look-ahead reasoning techniques and large number of states relevant for individual decisions. This paper introduces an algorithm LAMIR that learns an abstracted model of an imperfect information game directly from the agent-environment interaction. During test time, this trained model is used to perform look-ahead reasoning. The learned abstraction limits the size of each subgame to a manageable size, making theoretically principled look-ahead reasoning tractable even in games where previous methods could not scale. We empirically demonstrate that with sufficient capacity, LAMIR learns the exact underlying game structure, and with limited capacity, it still learns a valuable abstraction, which improves game playing performance of the pre-trained agents even in large games. 

**Abstract (ZH)**: Test-time reasoning显著增强预训练AI代理的性能，但需要显式的环境模型，这在现实世界场景中往往不可用或过于复杂。虽然MuZero在完美信息游戏中有效学习模型，将其范式扩展到不完美信息游戏由于更复杂的前瞻推理技术和大量相关状态，面临重大挑战。本文介绍了一种算法LAMIR，可以从代理与环境的交互中直接学习不完美信息游戏的抽象模型。测试时，训练后的模型用于进行前瞻推理。学习到的抽象将每个子游戏的规模限制在可管理的范围内，即使在以前的方法无法扩展的游戏环境中，也使理论上原则性的前瞻推理变得可行。我们通过实验证明，LAMIR在足够的能力下学习到游戏的确切结构，在有限的能力下仍能学习有价值的观点，并提高预训练代理在大型游戏中的表现。 

---
# Safe and Compliant Cross-Market Trade Execution via Constrained RL and Zero-Knowledge Audits 

**Title (ZH)**: 通过受限RL和零知识审计实现安全合规的跨市场交易执行 

**Authors**: Ailiya Borjigin, Cong He  

**Link**: [PDF](https://arxiv.org/pdf/2510.04952)  

**Abstract**: We present a cross-market algorithmic trading system that balances execution quality with rigorous compliance enforcement. The architecture comprises a high-level planner, a reinforcement learning execution agent, and an independent compliance agent. We formulate trade execution as a constrained Markov decision process with hard constraints on participation limits, price bands, and self-trading avoidance. The execution agent is trained with proximal policy optimization, while a runtime action-shield projects any unsafe action into a feasible set. To support auditability without exposing proprietary signals, we add a zero-knowledge compliance audit layer that produces cryptographic proofs that all actions satisfied the constraints. We evaluate in a multi-venue, ABIDES-based simulator and compare against standard baselines (e.g., TWAP, VWAP). The learned policy reduces implementation shortfall and variance while exhibiting no observed constraint violations across stress scenarios including elevated latency, partial fills, compliance module toggling, and varying constraint limits. We report effects at the 95% confidence level using paired t-tests and examine tail risk via CVaR. We situate the work at the intersection of optimal execution, safe reinforcement learning, regulatory technology, and verifiable AI, and discuss ethical considerations, limitations (e.g., modeling assumptions and computational overhead), and paths to real-world deployment. 

**Abstract (ZH)**: 一种平衡执行质量与严格合规 enforcement 的跨市场算法交易系统 

---
# Human Behavior Atlas: Benchmarking Unified Psychological and Social Behavior Understanding 

**Title (ZH)**: 人类行为地图：统一心理与社会行为理解的基准测试 

**Authors**: Keane Ong, Wei Dai, Carol Li, Dewei Feng, Hengzhi Li, Jingyao Wu, Jiaee Cheong, Rui Mao, Gianmarco Mengaldo, Erik Cambria, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04899)  

**Abstract**: Using intelligent systems to perceive psychological and social behaviors, that is, the underlying affective, cognitive, and pathological states that are manifested through observable behaviors and social interactions, remains a challenge due to their complex, multifaceted, and personalized nature. Existing work tackling these dimensions through specialized datasets and single-task systems often miss opportunities for scalability, cross-task transfer, and broader generalization. To address this gap, we curate Human Behavior Atlas, a unified benchmark of diverse behavioral tasks designed to support the development of unified models for understanding psychological and social behaviors. Human Behavior Atlas comprises over 100,000 samples spanning text, audio, and visual modalities, covering tasks on affective states, cognitive states, pathologies, and social processes. Our unification efforts can reduce redundancy and cost, enable training to scale efficiently across tasks, and enhance generalization of behavioral features across domains. On Human Behavior Atlas, we train three models: OmniSapiens-7B SFT, OmniSapiens-7B BAM, and OmniSapiens-7B RL. We show that training on Human Behavior Atlas enables models to consistently outperform existing multimodal LLMs across diverse behavioral tasks. Pretraining on Human Behavior Atlas also improves transfer to novel behavioral datasets; with the targeted use of behavioral descriptors yielding meaningful performance gains. 

**Abstract (ZH)**: 使用智能系统感知心理和社会行为：人类行为图谱的构建与应用 

---
# Video Game Level Design as a Multi-Agent Reinforcement Learning Problem 

**Title (ZH)**: 视频游戏关卡设计作为一种多智能体强化学习问题 

**Authors**: Sam Earle, Zehua Jiang, Eugene Vinitsky, Julian Togelius  

**Link**: [PDF](https://arxiv.org/pdf/2510.04862)  

**Abstract**: Procedural Content Generation via Reinforcement Learning (PCGRL) offers a method for training controllable level designer agents without the need for human datasets, using metrics that serve as proxies for level quality as rewards. Existing PCGRL research focuses on single generator agents, but are bottlenecked by the need to frequently recalculate heuristics of level quality and the agent's need to navigate around potentially large maps. By framing level generation as a multi-agent problem, we mitigate the efficiency bottleneck of single-agent PCGRL by reducing the number of reward calculations relative to the number of agent actions. We also find that multi-agent level generators are better able to generalize to out-of-distribution map shapes, which we argue is due to the generators' learning more local, modular design policies. We conclude that treating content generation as a distributed, multi-agent task is beneficial for generating functional artifacts at scale. 

**Abstract (ZH)**: 基于强化学习的程序化内容生成（PCGRL）提供了一种在无需人类数据集的情况下训练可控关卡设计代理的方法，使用作为关卡质量代理的指标作为奖励。现有PCGRL研究集中在单个生成器代理上，但受限于频繁重新计算关卡质量的启发式以及代理需要在可能非常大的地图中导航的需求。通过将关卡生成问题框架化为多代理问题，我们通过减少奖励计算次数相对于代理动作次数的比例，缓解了单代理PCGRL的效率瓶颈。我们还发现，多代理关卡生成器能够更好地泛化到分布外的地图形状，我们认为这是由于生成器学习到了更多局部的、模块化的设计策略。我们得出结论，将内容生成视为分布式的多代理任务有助于大规模生成功能性成果。 

---
# Hybrid-Balance GFlowNet for Solving Vehicle Routing Problems 

**Title (ZH)**: 混合平衡GFlowNet解决车辆 routing 问题 

**Authors**: Ni Zhang, Zhiguang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2510.04792)  

**Abstract**: Existing GFlowNet-based methods for vehicle routing problems (VRPs) typically employ Trajectory Balance (TB) to achieve global optimization but often neglect important aspects of local optimization. While Detailed Balance (DB) addresses local optimization more effectively, it alone falls short in solving VRPs, which inherently require holistic trajectory optimization. To address these limitations, we introduce the Hybrid-Balance GFlowNet (HBG) framework, which uniquely integrates TB and DB in a principled and adaptive manner by aligning their intrinsically complementary strengths. Additionally, we propose a specialized inference strategy for depot-centric scenarios like the Capacitated Vehicle Routing Problem (CVRP), leveraging the depot node's greater flexibility in selecting successors. Despite this specialization, HBG maintains broad applicability, extending effectively to problems without explicit depots, such as the Traveling Salesman Problem (TSP). We evaluate HBG by integrating it into two established GFlowNet-based solvers, i.e., AGFN and GFACS, and demonstrate consistent and significant improvements across both CVRP and TSP, underscoring the enhanced solution quality and generalization afforded by our approach. 

**Abstract (ZH)**: 基于GFlowNet的方法在车辆路线问题（VRPs）中的现有研究通常使用轨迹平衡（TB）以实现全局优化，但往往会忽视局部优化的重要方面。虽然详细平衡（DB）更有效地处理局部优化，但它单独解决VRPs时仍存在不足，因为VRPs本质上要求全面的轨迹优化。为了解决这些局限性，我们提出了混合平衡GFlowNet（HBG）框架，该框架以原则性和自适应的方式独特地结合了TB和DB的固有互补优势。此外，我们还提出了一种专门的推理策略，用于以配送中心为中心的情景，如 capacitated vehicle routing problem (CVRP)，利用配送中心节点在选择后继者方面的更大灵活性。尽管有所专化，HBG仍然保持广泛的适用性，有效地扩展到诸如旅行商问题（TSP）等没有明确配送中心的问题。我们通过将HBG集成到两个现有的GFlowNet基于的求解器（AGFN和GFACS）中来评估HBG，并在CVRP和TSP上展示了其一致且显著的改进，突显了我们方法提供的增强解决方案质量和泛化能力。 

---
# Watch and Learn: Learning to Use Computers from Online Videos 

**Title (ZH)**: 看并学习：从在线视频中学习使用计算机 

**Authors**: Chan Hee Song, Yiwen Song, Palash Goyal, Yu Su, Oriana Riva, Hamid Palangi, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2510.04673)  

**Abstract**: Computer use agents (CUAs) need to plan task workflows grounded in diverse, ever-changing applications and environments, but learning is hindered by the scarcity of large-scale, high-quality training data in the target application. Existing datasets are domain-specific, static, and costly to annotate, while current synthetic data generation methods often yield simplistic or misaligned task demonstrations. To address these limitations, we introduce Watch & Learn (W&L), a framework that converts human demonstration videos readily available on the Internet into executable UI trajectories at scale. Instead of directly generating trajectories or relying on ad hoc reasoning heuristics, we cast the problem as an inverse dynamics objective: predicting the user's action from consecutive screen states. This formulation reduces manual engineering, is easier to learn, and generalizes more robustly across applications. Concretely, we develop an inverse dynamics labeling pipeline with task-aware video retrieval, generate over 53k high-quality trajectories from raw web videos, and demonstrate that these trajectories improve CUAs both as in-context demonstrations and as supervised training data. On the challenging OSWorld benchmark, UI trajectories extracted with W&L consistently enhance both general-purpose and state-of-the-art frameworks in-context, and deliver stronger gains for open-source models under supervised training. These results highlight web-scale human demonstration videos as a practical and scalable foundation for advancing CUAs towards real-world deployment. 

**Abstract (ZH)**: 基于观察与学习的计算机使用代理框架（Watch & Learn）：大规模生成可执行的用户界面轨迹 

---
# Perfect AI Mimicry and the Epistemology of Consciousness: A Solipsistic Dilemma 

**Title (ZH)**: 完美的AI模拟与意识的 epistemology : 一种唯我论困境 

**Authors**: Shurui Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.04588)  

**Abstract**: Rapid advances in artificial intelligence necessitate a re-examination of the epistemological foundations upon which we attribute consciousness. As AI systems increasingly mimic human behavior and interaction with high fidelity, the concept of a "perfect mimic"-an entity empirically indistinguishable from a human through observation and interaction-shifts from hypothetical to technologically plausible. This paper argues that such developments pose a fundamental challenge to the consistency of our mind-recognition practices. Consciousness attributions rely heavily, if not exclusively, on empirical evidence derived from behavior and interaction. If a perfect mimic provides evidence identical to that of humans, any refusal to grant it equivalent epistemic status must invoke inaccessible factors, such as qualia, substrate requirements, or origin. Selectively invoking such factors risks a debilitating dilemma: either we undermine the rational basis for attributing consciousness to others (epistemological solipsism), or we accept inconsistent reasoning. I contend that epistemic consistency demands we ascribe the same status to empirically indistinguishable entities, regardless of metaphysical assumptions. The perfect mimic thus acts as an epistemic mirror, forcing critical reflection on the assumptions underlying intersubjective recognition in light of advancing AI. This analysis carries significant implications for theories of consciousness and ethical frameworks concerning artificial agents. 

**Abstract (ZH)**: 快速发展的人工智能 necessitates a re-examination of the epistemological foundations upon which we attribute consciousness。 

---
# Strongly Solving 2048 4x3 

**Title (ZH)**: 强求解2048游戏的4x3版本 

**Authors**: Tomoyuki Kaneko, Shuhei Yamashita  

**Link**: [PDF](https://arxiv.org/pdf/2510.04580)  

**Abstract**: 2048 is a stochastic single-player game involving 16 cells on a 4 by 4 grid, where a player chooses a direction among up, down, left, and right to obtain a score by merging two tiles with the same number located in neighboring cells along the chosen direction. This paper presents that a variant 2048-4x3 12 cells on a 4 by 3 board, one row smaller than the original, has been strongly solved. In this variant, the expected score achieved by an optimal strategy is about $50724.26$ for the most common initial states: ones with two tiles of number 2. The numbers of reachable states and afterstates are identified to be $1,152,817,492,752$ and $739,648,886,170$, respectively. The key technique is to partition state space by the sum of tile numbers on a board, which we call the age of a state. An age is invariant between a state and its successive afterstate after any valid action and is increased two or four by stochastic response from the environment. Therefore, we can partition state space by ages and enumerate all (after)states of an age depending only on states with the recent ages. Similarly, we can identify (after)state values by going along with ages in decreasing order. 

**Abstract (ZH)**: 2048-4x3：一个具有12个细胞的4x3板的单人博弈的强解 

---
# Aria: An Agent For Retrieval and Iterative Auto-Formalization via Dependency Graph 

**Title (ZH)**: Aria: 一种基于依赖图检索和迭代自动形式化的代理 

**Authors**: Hanyu Wang, Ruohan Xie, Yutong Wang, Guoxiong Gao, Xintao Yu, Bin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2510.04520)  

**Abstract**: Accurate auto-formalization of theorem statements is essential for advancing automated discovery and verification of research-level mathematics, yet remains a major bottleneck for LLMs due to hallucinations, semantic mismatches, and their inability to synthesize new definitions. To tackle these issues, we present Aria (Agent for Retrieval and Iterative Autoformalization), a system for conjecture-level formalization in Lean that emulates human expert reasoning via a two-phase Graph-of-Thought process: recursively decomposing statements into a dependency graph and then constructing formalizations from grounded concepts. To ensure semantic correctness, we introduce AriaScorer, a checker that retrieves definitions from Mathlib for term-level grounding, enabling rigorous and reliable verification. We evaluate Aria on diverse benchmarks. On ProofNet, it achieves 91.6% compilation success rate and 68.5% final accuracy, surpassing previous methods. On FATE-X, a suite of challenging algebra problems from research literature, it outperforms the best baseline with 44.0% vs. 24.0% final accuracy. On a dataset of homological conjectures, Aria reaches 42.9% final accuracy while all other models score 0%. 

**Abstract (ZH)**: 准确的自动形式化是推进研究级数学的自动发现与验证的关键，但由于幻觉、语义不匹配以及合成新定义的能力不足，这对大型语言模型仍然是一大瓶颈。为解决这些问题，我们提出了Aria（推理与迭代自动形式化代理），一种通过递归分解语句为依赖图，然后从基础概念构建形式化的Lean系统，以模仿人类专家推理。为确保语义正确性，我们引入了AriaScorer，这是一种检查器，从Mathlib检索定义进行术语级别接地，实现严格的可靠验证。我们在多种基准上评估了Aria。在ProofNet上，它实现了91.6%的编译成功率和68.5%的最终准确率，超过了之前的方法。在FATE-X上，这是一个来自研究文献的一系列具有挑战性的代数问题套件，它以44.0%的最终准确率超过了最佳基线的24.0%。在同调猜想数据集中，Aria 达到了42.9%的最终准确率，而其他所有模型均为0%。 

---
# Impatient Users Confuse AI Agents: High-fidelity Simulations of Human Traits for Testing Agents 

**Title (ZH)**: 用户缺乏耐心使AI代理困惑：用于测试代理的高保真人类特质模拟 

**Authors**: Muyu He, Anand Kumar, Tsach Mackey, Meghana Rajeev, James Zou, Nazneen Rajani  

**Link**: [PDF](https://arxiv.org/pdf/2510.04491)  

**Abstract**: Despite rapid progress in building conversational AI agents, robustness is still largely untested. Small shifts in user behavior, such as being more impatient, incoherent, or skeptical, can cause sharp drops in agent performance, revealing how brittle current AI agents are. Today's benchmarks fail to capture this fragility: agents may perform well under standard evaluations but degrade spectacularly in more realistic and varied settings. We address this robustness testing gap by introducing TraitBasis, a lightweight, model-agnostic method for systematically stress testing AI agents. TraitBasis learns directions in activation space corresponding to steerable user traits (e.g., impatience or incoherence), which can be controlled, scaled, composed, and applied at inference time without any fine-tuning or extra data. Using TraitBasis, we extend $\tau$-Bench to $\tau$-Trait, where user behaviors are altered via controlled trait vectors. We observe on average a 2%-30% performance degradation on $\tau$-Trait across frontier models, highlighting the lack of robustness of current AI agents to variations in user behavior. Together, these results highlight both the critical role of robustness testing and the promise of TraitBasis as a simple, data-efficient, and compositional tool. By powering simulation-driven stress tests and training loops, TraitBasis opens the door to building AI agents that remain reliable in the unpredictable dynamics of real-world human interactions. We have open-sourced $\tau$-Trai across four domains: airline, retail, telecom, and telehealth, so the community can systematically QA their agents under realistic, behaviorally diverse intents and trait scenarios: this https URL. 

**Abstract (ZH)**: 尽管在构建对话AI代理方面取得了快速进展，但其鲁棒性仍然没有得到充分测试。用户行为的小幅变化，如更加急躁、不连贯或怀疑，都可能导致代理性能急剧下降，揭示当前AI代理的脆弱性。现有的基准未能捕捉到这种脆弱性：代理可能在标准评估中表现良好，但在更现实和多变的环境中表现会大幅下降。为此，我们通过引入TraitBasis，一种轻量级、模型无关的方法，系统地对AI代理进行压力测试来填补这一鲁棒性测试的空白。TraitBasis学习与可调控用户特性（如急躁或不连贯）对应的激活空间方向，这些特性可以在推理时控制、缩放、组合和应用，无需微调或额外数据。使用TraitBasis，我们将$\tau$-Bench扩展为$\tau$-Trait，通过控制特性向量改变用户行为。结果显示，前沿模型在$\tau$-Trait上的性能平均下降2%-30%，突显出当前AI代理在用户行为变化方面的鲁棒性不足。这些结果强调了鲁棒性测试的至关重要性，并展示了TraitBasis作为一种简单、数据高效且可组合工具的潜力。通过驱动模拟驱动的压力测试和训练循环，TraitBasis为构建能够在现实世界人类互动的不可预测动态中保持可靠性的AI代理打开了大门。我们已在四个领域（航空、零售、电信和远程医疗）开源了$\tau$-Trait，社区可以使用它系统地对代理在现实的、行为多样的意图和特性场景下进行QA：this https URL。 

---
# On Continuous Optimization for Constraint Satisfaction Problems 

**Title (ZH)**: 连续优化在约束满足问题中的应用 

**Authors**: Yunuo Cen, Zixuan Wang, Jintao Zhang, Zhiwei Zhang, Xuanyao Fong  

**Link**: [PDF](https://arxiv.org/pdf/2510.04480)  

**Abstract**: Constraint satisfaction problems (CSPs) are fundamental in mathematics, physics, and theoretical computer science. While conflict-driven clause learning Boolean Satisfiability (SAT) solvers have achieved remarkable success and become the mainstream approach for Boolean satisfiability, recent advances show that modern continuous local search (CLS) solvers can achieve highly competitive results on certain classes of SAT problems. Motivated by these advances, we extend the CLS framework from Boolean SAT to general CSP with finite-domain variables and expressive constraints. We present FourierCSP, a continuous optimization framework that generalizes the Walsh-Fourier transform to CSP, allowing for transforming versatile constraints to compact multilinear polynomials, thereby avoiding the need for auxiliary variables and memory-intensive encodings. Our approach leverages efficient evaluation and differentiation of the objective via circuit-output probability and employs a projected gradient optimization method with theoretical guarantees. Empirical results on benchmark suites demonstrate that FourierCSP is scalable and competitive, significantly broadening the class of problems that can be efficiently solved by CLS techniques. 

**Abstract (ZH)**: 约束满足问题（CSPs）在数学、物理学和理论计算机科学中是基础性的。尽管基于冲突驱动的_clause学习（CDCL）的布尔可满足性（SAT）求解器取得了显著成功并已成为布尔可满足性的主流方法，但最近的研究表明，现代连续局部搜索（CLS）求解器在某些类型的SAT问题上可以取得高度竞争的结果。受这些进展的启发，我们将CLS框架从布尔SAT扩展到具有有限域变量和表达性约束的通用CSP。我们提出了FourierCSP，这是一种连续优化框架，将Walsh-傅里叶变换推广到CSP，允许将多样化的约束转换为紧凑的多线性多项式，从而避免使用辅助变量和内存密集型编码。我们的方法利用电路输出概率高效评估和求解目标函数的梯度，并采用具有理论保证的投影梯度优化方法。基准测试集上的实验结果表明，FourierCSP是可扩展且竞争力强的，显著扩展了可以高效解决的CSP问题类别。 

---
# DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization 

**Title (ZH)**: DRPO：通过解耦奖励策略优化进行高效推理 

**Authors**: Gang Li, Yan Chen, Ming Lin, Tianbao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04474)  

**Abstract**: Recent large reasoning models (LRMs) driven by reinforcement learning algorithms (e.g., GRPO) have achieved remarkable performance on challenging reasoning tasks. However, these models suffer from overthinking, generating unnecessarily long and redundant reasoning even for simple questions, which substantially increases computational cost and response latency. While existing methods incorporate length rewards to GRPO to promote concise reasoning, they incur significant performance degradation. We identify the root cause: when rewards for correct but long rollouts are penalized, GRPO's group-relative advantage function can assign them negative advantages, actively discouraging valid reasoning. To overcome this, we propose Decoupled Reward Policy Optimization (DRPO), a novel framework that decouples the length-based learning signal of correct rollouts from incorrect ones. DRPO ensures that reward signals for correct rollouts are normalized solely within the positive group, shielding them from interference by negative samples. The DRPO's objective is grounded in integrating an optimized positive data distribution, which maximizes length-based rewards under a KL regularization, into a discriminative objective. We derive a closed-form solution for this distribution, enabling efficient computation of the objective and its gradients using only on-policy data and importance weighting. Of independent interest, this formulation is general and can incorporate other preference rewards of positive data beyond length. Experiments on mathematical reasoning tasks demonstrate DRPO's significant superiority over six efficient reasoning baselines. Notably, with a 1.5B model, our method achieves 77\% length reduction with only 1.1\% performance loss on simple questions like GSM8k dataset, while the follow-up baseline sacrifices 4.3\% for 68\% length reduction. 

**Abstract (ZH)**: Recent Large Reasoning Models Driven by Reinforcement Learning Algorithms (e.g., GRPO) for Efficient Reasoning Tasks: Decoupled Reward Policy Optimization (DRPO) for Reducing Overthinking 

---
# Utility-Learning Tension in Self-Modifying Agents 

**Title (ZH)**: 自我修改代理的效用学习张力 

**Authors**: Charles L. Wang, Keir Dorchen, Peter Jin  

**Link**: [PDF](https://arxiv.org/pdf/2510.04399)  

**Abstract**: As systems trend toward superintelligence, a natural modeling premise is that agents can self-improve along every facet of their own design. We formalize this with a five-axis decomposition and a decision layer, separating incentives from learning behavior and analyzing axes in isolation. Our central result identifies and introduces a sharp utility--learning tension, the structural conflict in self-modifying systems whereby utility-driven changes that improve immediate or expected performance can also erode the statistical preconditions for reliable learning and generalization. Our findings show that distribution-free guarantees are preserved iff the policy-reachable model family is uniformly capacity-bounded; when capacity can grow without limit, utility-rational self-changes can render learnable tasks unlearnable. Under standard assumptions common in practice, these axes reduce to the same capacity criterion, yielding a single boundary for safe self-modification. Numerical experiments across several axes validate the theory by comparing destructive utility policies against our proposed two-gate policies that preserve learnability. 

**Abstract (ZH)**: 随着系统向超智能发展，一个自然的建模假设是代理可以在设计的各个方面自我改进。我们通过五轴分解和决策层形式化这一假设，将激励与学习行为分离，并分别分析各个维度。我们的主要成果是识别并引入了一种尖锐的效用-学习张力，这是一种自我修改系统中的结构冲突，其中由效用驱动的改进即时或预期性能的变化，也可能侵蚀可靠学习和泛化的统计前提条件。研究发现，在且仅在策略可达到的模型家族具有统一的容量限制时，无分布保证被保留在；当容量可以无限制增长时，效用理性自我变更可以使可学习的任务变得不可学习。在实践中常见的标准假设下，这些维度归结为同一容量标准，提供了一个自我修改的安全边界。通过对多个维度的数值实验，通过将破坏性效用政策与我们提出的双门控政策进行比较，验证了理论，后者保留了可学习性。 

---
# Closing the Loop: Coordinating Inventory and Recommendation via Deep Reinforcement Learning on Multiple Timescales 

**Title (ZH)**: 闭环控制：通过多时间尺度深度强化学习协调库存与推荐 

**Authors**: Jinyang Jiang, Jinhui Han, Yijie Peng, Ying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04272)  

**Abstract**: Effective cross-functional coordination is essential for enhancing firm-wide profitability, particularly in the face of growing organizational complexity and scale. Recent advances in artificial intelligence, especially in reinforcement learning (RL), offer promising avenues to address this fundamental challenge. This paper proposes a unified multi-agent RL framework tailored for joint optimization across distinct functional modules, exemplified via coordinating inventory replenishment and personalized product recommendation. We first develop an integrated theoretical model to capture the intricate interplay between these functions and derive analytical benchmarks that characterize optimal coordination. The analysis reveals synchronized adjustment patterns across products and over time, highlighting the importance of coordinated decision-making. Leveraging these insights, we design a novel multi-timescale multi-agent RL architecture that decomposes policy components according to departmental functions and assigns distinct learning speeds based on task complexity and responsiveness. Our model-free multi-agent design improves scalability and deployment flexibility, while multi-timescale updates enhance convergence stability and adaptability across heterogeneous decisions. We further establish the asymptotic convergence of the proposed algorithm. Extensive simulation experiments demonstrate that the proposed approach significantly improves profitability relative to siloed decision-making frameworks, while the behaviors of the trained RL agents align closely with the managerial insights from our theoretical model. Taken together, this work provides a scalable, interpretable RL-based solution to enable effective cross-functional coordination in complex business settings. 

**Abstract (ZH)**: 有效跨功能协调对于增强企业整体盈利能力至关重要，尤其是面对日益增长的组织复杂性和规模。近年来，尤其是强化学习（RL）的进展为解决这一根本挑战提供了有希望的途径。本文提出了一种针对不同功能模块联合优化的统一多Agent RL框架，通过协调库存补充和个性化产品推荐进行阐述。我们首先开发了一个综合的理论模型来捕捉这些功能之间的复杂相互作用，并推导出表征最优协调的分析基准。分析揭示了产品和服务之间同步调整的模式，突显了协调决策的重要性。利用这些见解，我们设计了一种新颖的多时间尺度多Agent RL架构，根据部门功能分解策略成分，并基于任务复杂性和响应性分配不同的学习速度。我们的无模型多Agent设计增强了可扩展性和部署灵活性，而多时间尺度更新增强了跨异质决策的收敛稳定性和适应性。进一步建立了所提算法的渐近收敛性。广泛的模拟实验表明，所提出的方法相对于孤立决策框架显著提高了盈利能力，而训练的RL代理行为与我们理论模型的管理洞察高度一致。总体而言，本文为复杂商业环境中的有效跨功能协调提供了一个可扩展且可解释的RL基解决方案。 

---
# Open Agent Specification (Agent Spec) Technical Report 

**Title (ZH)**: 开放代理规范（代理规范）技术报告 

**Authors**: Yassine Benajiba, Cesare Bernardis, Vladislav Blinov, Paul Cayet, Hassan Chafi, Abderrahim Fathan, Louis Faucon, Damien Hilloulin, Sungpack Hong, Ingo Kossyk, Rhicheek Patra, Sujith Ravi, Jonas Schweizer, Jyotika Singh, Shailender Singh, Xuelin Situ, Weiyi Sun, Jerry Xu, Ying Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04173)  

**Abstract**: Open Agent Specification (Agent Spec) is a declarative language that allows AI agents and their workflows to be defined in a way that is compatible across different AI frameworks, promoting portability and interoperability within AI Agent frameworks.
Agent Spec aims to resolve the challenges of fragmented agent development by providing a common unified specification that allows AI agents to be designed once and deployed across various frameworks, improving interoperability and reusability, and reducing redundant development efforts. Additionally, Agent Spec facilitates development tools and portability, allowing AI agents to be defined independently of their execution environment and enabling teams to exchange solutions without implementation-specific limitations.
Agent Spec benefits four key groups: (i) Agent developers, who gain access to a superset of reusable components and design patterns, enabling them to leverage a broader range of functionalities; (ii) Agent framework and tool developers, who can use Agent Spec as an interchange format and therefore benefit from the support of other frameworks as well as other tools; (iii) Researchers, who can achieve reproducible results and comparability, facilitating more reliable and consistent outcomes; (iv) Enterprises, which benefit from faster prototype-to-deployment, increased productivity, as well as greater scalability and maintainability for their AI agent solutions. This technical report provides an overview of the technical foundations of Agent Spec, including motivation, benefits, and future developments. 

**Abstract (ZH)**: 开源代理规范（Agent Spec）是一种声明性语言，允许AI代理及其工作流以跨不同AI框架兼容的方式被定义，促进AI代理框架内的便捷性和互操作性。 

---
# Internal states before wait modulate reasoning patterns 

**Title (ZH)**: 等待前的内部状态调节推理模式 

**Authors**: Dmitrii Troitskii, Koyena Pal, Chris Wendler, Callum Stuart McDougall, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2510.04128)  

**Abstract**: Prior work has shown that a significant driver of performance in reasoning models is their ability to reason and self-correct. A distinctive marker in these reasoning traces is the token wait, which often signals reasoning behavior such as backtracking. Despite being such a complex behavior, little is understood of exactly why models do or do not decide to reason in this particular manner, which limits our understanding of what makes a reasoning model so effective. In this work, we address the question whether model's latents preceding wait tokens contain relevant information for modulating the subsequent reasoning process. We train crosscoders at multiple layers of DeepSeek-R1-Distill-Llama-8B and its base version, and introduce a latent attribution technique in the crosscoder setting. We locate a small set of features relevant for promoting/suppressing wait tokens' probabilities. Finally, through a targeted series of experiments analyzing max activating examples and causal interventions, we show that many of our identified features indeed are relevant for the reasoning process and give rise to different types of reasoning patterns such as restarting from the beginning, recalling prior knowledge, expressing uncertainty, and double-checking. 

**Abstract (ZH)**: 先前的研究表明，推理模型的性能显著取决于其推理和自我纠正的能力。这些推理路径中的一个独特标志是等待标记（token wait），它通常表明了诸如回溯等推理行为。尽管这是一个复杂的行为，但对于模型为何或为何不以这种方式推理的具体原因还知之甚少，这限制了我们对哪些因素使推理模型如此有效这一理解。在本工作中，我们探讨了模型等待标记之前的潜在特征是否包含调节后续推理过程的相关信息。我们对DeepSeek-R1-Distill-Llama-8B及其基版本的多个层级进行了交叉编码器训练，并引入了一种在交叉编码器设置中的潜在特征归因技术。我们定位到一组促进或抑制等待标记概率的相关特征。最后，通过分析最大激活示例和因果干预的一系列针对性实验，我们证明了我们识别出的许多特征确实对推理过程具有相关性，并导致了不同的推理模式，如重新开始、调用先前的知识、表达不确定性以及复查。 

---
# WebRenderBench: Enhancing Web Interface Generation through Layout-Style Consistency and Reinforcement Learning 

**Title (ZH)**: WebRenderBench：通过布局-样式一致性与强化学习提升Web界面生成 

**Authors**: Peichao Lai, Jinhui Zhuang, Kexuan Zhang, Ningchang Xiong, Shengjie Wang, Yanwei Xu, Chong Chen, Yilei Wang, Bin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2510.04097)  

**Abstract**: Automating the conversion of UI images into web code is a critical task for front-end development and rapid prototyping. Advances in multimodal large language models (MLLMs) have made WebUI-to-Code increasingly feasible, yet existing benchmarks remain limited in data diversity and evaluation reliability. To address these issues, we present WebRenderBench, a large-scale benchmark of 22.5k webpages collected from real-world portal sites, offering greater diversity, complexity, and realism than prior benchmarks. We further propose a novel evaluation metric that measures layout and style consistency from the final rendered pages. Unlike vision-based methods that rely on costly LLM reasoning or structure-based comparisons vulnerable to noise and asymmetry, our approach enables more efficient, objective, and reliable UI quality assessment. Finally, we introduce the Automated Layout and Style Inspection Agent (ALISA), which integrates this metric into reinforcement learning as a reward signal to enhance training on crawled asymmetric webpages. Experiments show that ALISA significantly boosts generation performance, achieving state-of-the-art results across multiple metrics. 

**Abstract (ZH)**: 自动化将UI图像转换为网页代码是前端开发和快速原型设计中的关键任务。多模态大型语言模型的进步使得WebUI-to-Code越来越可行，但现有基准在数据多样性和评估可靠性方面仍然有限。为了解决这些问题，我们介绍了WebRenderBench，这是一个由22500个网页组成的大规模基准，这些网页来自实际门户站点，提供了比先前基准更多的多样性和现实性。进一步提出了一种新的评估指标，用于衡量最终渲染页面的布局和样式一致性。与依赖昂贵的LLM推理或易受噪声和不对称影响的结构比较方法不同，我们的方法能够更高效、客观和可靠地评估UI质量。最后，我们引入了自动化布局和样式检查代理（ALISA），将该指标整合到强化学习中作为奖励信号，以增强抓取的不对称网页上的训练。实验结果表明，ALISA显著提升了生成性能，在多个指标上达到了最先进的结果。 

---
# Moral Anchor System: A Predictive Framework for AI Value Alignment and Drift Prevention 

**Title (ZH)**: 道德锚定系统：一种预测性AI价值对齐与偏移预防框架 

**Authors**: Santhosh Kumar Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2510.04073)  

**Abstract**: The rise of artificial intelligence (AI) as super-capable assistants has transformed productivity and decision-making across domains. Yet, this integration raises critical concerns about value alignment - ensuring AI behaviors remain consistent with human ethics and intentions. A key risk is value drift, where AI systems deviate from aligned values due to evolving contexts, learning dynamics, or unintended optimizations, potentially leading to inefficiencies or ethical breaches. We propose the Moral Anchor System (MAS), a novel framework to detect, predict, and mitigate value drift in AI agents. MAS combines real-time Bayesian inference for monitoring value states, LSTM networks for forecasting drift, and a human-centric governance layer for adaptive interventions. It emphasizes low-latency responses (<20 ms) to prevent breaches, while reducing false positives and alert fatigue via supervised fine-tuning with human feedback. Our hypothesis: integrating probabilistic drift detection, predictive analytics, and adaptive governance can reduce value drift incidents by 80 percent or more in simulations, maintaining high detection accuracy (85 percent) and low false positive rates (0.08 post-adaptation). Rigorous experiments with goal-misaligned agents validate MAS's scalability and responsiveness. MAS's originality lies in its predictive and adaptive nature, contrasting static alignment methods. Contributions include: (1) MAS architecture for AI integration; (2) empirical results prioritizing speed and usability; (3) cross-domain applicability insights; and (4) open-source code for replication. 

**Abstract (ZH)**: 人工智能（AI）作为超级能力助手的崛起已 transforming生产力和决策领域。然而，这种整合引发了关于价值对齐的重要关切——确保AI行为与人类伦理和意图保持一致。一个关键风险是价值偏移，即由于环境变化、学习动态或意外优化，AI系统可能偏离对齐价值观，导致效率低下或伦理违规。我们提出道德锚系统（MAS），这是一种新颖的框架，用于检测、预测和减轻AI代理的价值偏移。MAS结合了实时贝叶斯推理进行价值状态监控，LSTM网络进行偏移预测，以及以人为中心的治理层进行适应性干预。它强调低于20毫秒的低延迟响应，以防止违规行为，同时通过监督微调和人类反馈减少误报和警报疲劳。我们的假设：结合概率偏移检测、预测分析和适应性治理可以在模拟中将价值偏移事件减少80％或更多，保持高检测准确性（85％）和低误报率（在适应后为0.08）。严格的实验验证了MAS在目标错配代理中的可扩展性和响应性。MAS的独特性在于其预测和适应性，与静态对齐方法形成对比。贡献包括：（1）AI集成的MAS架构；（2）优先考虑速度和易用性的实证结果；（3）跨领域的适用性见解；（4）开源代码以供复制。 

---
# A global log for medical AI 

**Title (ZH)**: 医疗AI全球日志 

**Authors**: Ayush Noori, Adam Rodman, Alan Karthikesalingam, Bilal A. Mateen, Christopher A. Longhurst, Daniel Yang, Dave deBronkart, Gauden Galea, Harold F. Wolf III, Jacob Waxman, Joshua C. Mandel, Juliana Rotich, Kenneth D. Mandl, Maryam Mustafa, Melissa Miles, Nigam H. Shah, Peter Lee, Robert Korom, Scott Mahoney, Seth Hain, Tien Yin Wong, Trevor Mundel, Vivek Natarajan, Noa Dagan, David A. Clifton, Ran D. Balicer, Isaac S. Kohane, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2510.04033)  

**Abstract**: Modern computer systems often rely on syslog, a simple, universal protocol that records every critical event across heterogeneous infrastructure. However, healthcare's rapidly growing clinical AI stack has no equivalent. As hospitals rush to pilot large language models and other AI-based clinical decision support tools, we still lack a standard way to record how, when, by whom, and for whom these AI models are used. Without that transparency and visibility, it is challenging to measure real-world performance and outcomes, detect adverse events, or correct bias or dataset drift. In the spirit of syslog, we introduce MedLog, a protocol for event-level logging of clinical AI. Any time an AI model is invoked to interact with a human, interface with another algorithm, or act independently, a MedLog record is created. This record consists of nine core fields: header, model, user, target, inputs, artifacts, outputs, outcomes, and feedback, providing a structured and consistent record of model activity. To encourage early adoption, especially in low-resource settings, and minimize the data footprint, MedLog supports risk-based sampling, lifecycle-aware retention policies, and write-behind caching; detailed traces for complex, agentic, or multi-stage workflows can also be captured under MedLog. MedLog can catalyze the development of new databases and software to store and analyze MedLog records. Realizing this vision would enable continuous surveillance, auditing, and iterative improvement of medical AI, laying the foundation for a new form of digital epidemiology. 

**Abstract (ZH)**: 现代计算机系统通常依赖syslog这一简单且通用的协议，用于记录异构基础设施中所有关键事件。然而，随着医疗保健领域临床AI堆栈的迅速增长，我们缺乏相应的等效工具。随着医院急忙进行大型语言模型和其他基于AI的临床决策支持工具的试点，我们仍然缺乏一种标准方法来记录这些AI模型何时、何地、以及为谁被使用。缺乏这种透明度和可见性使得难以衡量实际表现和结果、检测不良事件或纠正偏差或数据集漂移。怀着syslog的精神，我们介绍MedLog，这是一种用于记录临床AI事件的协议。每当AI模型被调用来与人类交互、与其他算法接口，或独立行动时，就会创建一个MedLog记录。该记录包括九个核心字段：头信息、模型、用户、目标、输入、产物、输出、结果和反馈，从而提供了一个结构化且一致的模型活动记录。为了鼓励早期采用，特别是在资源有限的环境中，MedLog支持基于风险的抽样、生命周期意识的保留策略，以及写后缓存；复杂的、代理驱动的或多阶段的工作流的详细追踪也可以在MedLog下捕获。MedLog可以促进新数据库和软件的开发，用于存储和分析MedLog记录。实现这一愿景将使持续监控、审计和迭代改进医疗AI成为可能，为新的数字流行病学奠定基础。 

---
# Kantian-Utilitarian XAI: Meta-Explained 

**Title (ZH)**: 康德-功利主义XAI：元解释 

**Authors**: Zahra Atf, Peter R. Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2510.03892)  

**Abstract**: We present a gamified explainable AI (XAI) system for ethically aware consumer decision-making in the coffee domain. Each session comprises six rounds with three options per round. Two symbolic engines provide real-time reasons: a Kantian module flags rule violations (e.g., child labor, deforestation risk without shade certification, opaque supply chains, unsafe decaf), and a utilitarian module scores options via multi-criteria aggregation over normalized attributes (price, carbon, water, transparency, farmer income share, taste/freshness, packaging, convenience). A meta-explainer with a regret bound (0.2) highlights Kantian--utilitarian (mis)alignment and switches to a deontically clean, near-parity option when welfare loss is small. We release a structured configuration (attribute schema, certification map, weights, rule set), a policy trace for auditability, and an interactive UI. 

**Abstract (ZH)**: 我们提出了一种游戏化解释性人工智能（XAI）系统，用于咖啡领域中的伦理意识消费者决策。每个会话包括六轮，每轮有三个选项。两个符号引擎提供实时原因：一个康德模块标记规则违反（例如，童工、缺乏阴凉认证的采伐风险、不透明的供应链、不安全的脱咖啡因），一个功利主义模块通过归一化属性的多准则聚合为选项打分（价格、碳足迹、水资源使用、透明度、农民收入份额、风味/新鲜度、包装材料、便捷性）。一个元解释器带有后悔上限（0.2），突出康德主义-功利主义（不）一致，并在福利损失较小的情况下切换到一个ontology上洁净、接近对等的选择。我们发布了结构化的配置（属性模式、认证图谱、权重、规则集）、审计跟踪的政策策略以及交互式用户界面。 

---
# Spatial CAPTCHA: Generatively Benchmarking Spatial Reasoning for Human-Machine Differentiation 

**Title (ZH)**: 空间CAPTCHA：生成性评估人类与机器的空间推理差异 

**Authors**: Arina Kharlamova, Bowei He, Chen Ma, Xue Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03863)  

**Abstract**: Online services rely on CAPTCHAs as a first line of defense against automated abuse, yet recent advances in multi-modal large language models (MLLMs) have eroded the effectiveness of conventional designs that focus on text recognition or 2D image understanding. To address this challenge, we present Spatial CAPTCHA, a novel human-verification framework that leverages fundamental differences in spatial reasoning between humans and MLLMs. Unlike existing CAPTCHAs which rely on low-level perception tasks that are vulnerable to modern AI, Spatial CAPTCHA generates dynamic questions requiring geometric reasoning, perspective-taking, occlusion handling, and mental rotation. These skills are intuitive for humans but difficult for state-of-the-art (SOTA) AI systems. The system employs a procedural generation pipeline with constraint-based difficulty control, automated correctness verification, and human-in-the-loop validation to ensure scalability, robustness, and adaptability. Evaluation on a corresponding benchmark, Spatial-CAPTCHA-Bench, demonstrates that humans vastly outperform 10 state-of-the-art MLLMs, with the best model achieving only 31.0% Pass@1 accuracy. Furthermore, we compare Spatial CAPTCHA with Google reCAPTCHA, which confirms its effectiveness as both a security mechanism and a diagnostic tool for spatial reasoning in AI. 

**Abstract (ZH)**: 基于空间推理的Spatial CAPTCHA：一种克服现代大模型挑战的人机验证框架 

---
# The Hidden Game Problem 

**Title (ZH)**: 隐藏的比赛问题 

**Authors**: Gon Buzaglo, Noah Golowich, Elad Hazan  

**Link**: [PDF](https://arxiv.org/pdf/2510.03845)  

**Abstract**: This paper investigates a class of games with large strategy spaces, motivated by challenges in AI alignment and language games. We introduce the hidden game problem, where for each player, an unknown subset of strategies consistently yields higher rewards compared to the rest. The central question is whether efficient regret minimization algorithms can be designed to discover and exploit such hidden structures, leading to equilibrium in these subgames while maintaining rationality in general. We answer this question affirmatively by developing a composition of regret minimization techniques that achieve optimal external and swap regret bounds. Our approach ensures rapid convergence to correlated equilibria in hidden subgames, leveraging the hidden game structure for improved computational efficiency. 

**Abstract (ZH)**: 基于AI对齐和语言游戏挑战的大型策略空间博弈研究：隐藏博弈问题及其高效 regrets 优化算法 

---
# Towards Policy-Compliant Agents: Learning Efficient Guardrails For Policy Violation Detection 

**Title (ZH)**: 符合政策合规的代理：学习高效的政策违规检测边界 

**Authors**: Xiaofei Wen, Wenjie Jacky Mo, Yanan Xie, Peng Qi, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.03485)  

**Abstract**: Autonomous web agents need to operate under externally imposed or human-specified policies while generating long-horizon trajectories. However, little work has examined whether these trajectories comply with such policies, or whether policy violations persist across different contexts such as domains (e.g., shopping or coding websites) and subdomains (e.g., product search and order management in shopping). To address this gap, we introduce PolicyGuardBench, a benchmark of about 60k examples for detecting policy violations in agent trajectories. From diverse agent runs, we generate a broad set of policies and create both within subdomain and cross subdomain pairings with violation labels. In addition to full-trajectory evaluation, PolicyGuardBench also includes a prefix-based violation detection task where models must anticipate policy violations from truncated trajectory prefixes rather than complete sequences. Using this dataset, we train PolicyGuard-4B, a lightweight guardrail model that delivers strong detection accuracy across all tasks while keeping inference efficient. Notably, PolicyGuard-4B generalizes across domains and preserves high accuracy on unseen settings. Together, PolicyGuardBench and PolicyGuard-4B provide the first comprehensive framework for studying policy compliance in web agent trajectories, and show that accurate and generalizable guardrails are feasible at small scales. 

**Abstract (ZH)**: 基于PolicyGuardBench的Web代理政策遵守综合框架 

---
# A Qualitative Comparative Evaluation of Cognitive and Generative Theories 

**Title (ZH)**: 认知与生成理论的定性比较评价 

**Authors**: Paul S. Rosenbloom  

**Link**: [PDF](https://arxiv.org/pdf/2510.03453)  

**Abstract**: Evaluation is a critical activity associated with any theory. Yet this has proven to be an exceptionally challenging activity for theories based on cognitive architectures. For an overlapping set of reasons, evaluation can also be challenging for theories based on generative neural architectures. This dual challenge is approached here by leveraging a broad perspective on theory evaluation to yield a wide-ranging, albeit qualitative, comparison of whole-mind-oriented cognitive and generative architectures and the full systems that are based on these architectures. 

**Abstract (ZH)**: 基于认知架构和生成神经架构的理论评估是一项严峻挑战，本文通过广泛理论评估视角，提供了认知和生成架构及其所支持的完整系统之间的广泛且定性的比较。 

---
# ContraGen: A Multi-Agent Generation Framework for Enterprise Contradictions Detection 

**Title (ZH)**: ContraGen：一种企业矛盾检测的多Agent生成框架 

**Authors**: Ananya Mantravadi, Shivali Dalmia, Abhishek Mukherji, Nand Dave, Anudha Mittal  

**Link**: [PDF](https://arxiv.org/pdf/2510.03418)  

**Abstract**: Retrieval-Augmented Generation (RAG) integrates LLMs with external sources, offering advanced capabilities for information access and decision-making. However, contradictions in retrieved evidence can result in inconsistent or untrustworthy outputs, which is especially problematic in enterprise settings where compliance, governance, and accountability are critical. Existing benchmarks for contradiction detection are limited to sentence-level analysis and do not capture the complexity of enterprise documents such as contracts, financial filings, compliance reports, or policy manuals. To address this limitation, we propose ContraGen, a contradiction-aware benchmark framework tailored to enterprise domain. The framework generates synthetic enterprise-style documents with embedded contradictions, enabling systematic evaluation of both intra-document and cross-document consistency. Automated contradiction mining is combined with human-in-the-loop validation to ensure high accuracy. Our contributions include generating realistic enterprise documents, modeling a taxonomy of contradiction types common in business processes, enabling controlled creation of self- and pairwise contradictions, developing a contradiction-aware retrieval evaluation pipeline and embedding human oversight to reflect domain-specific judgment complexity. This work establishes a foundation for more trustworthy and accountable RAG systems in enterprise information-seeking applications, where detecting and resolving contradictions is essential for reducing risk and ensuring compliance. 

**Abstract (ZH)**: Retrieval-Augmented Generation中的矛盾感知基准框架：ContraGen 

---
# Refined Iterated Pareto Greedy for Energy-aware Hybrid Flowshop Scheduling with Blocking Constraints 

**Title (ZH)**: 改进的迭代帕累托贪婪算法用于带有阻塞约束的能量感知混合流水车间调度 

**Authors**: Ahmed Missaoui, Cemalettin Ozturk, Barry O'Sullivan  

**Link**: [PDF](https://arxiv.org/pdf/2510.03377)  

**Abstract**: The scarcity of non-renewable energy sources, geopolitical problems in its supply, increasing prices, and the impact of climate change, force the global economy to develop more energy-efficient solutions for their operations. The Manufacturing sector is not excluded from this challenge as one of the largest consumers of energy. Energy-efficient scheduling is a method that attracts manufacturing companies to reduce their consumption as it can be quickly deployed and can show impact immediately. In this study, the hybrid flow shop scheduling problem with blocking constraint (BHFS) is investigated in which we seek to minimize the latest completion time (i.e. makespan) and overall energy consumption, a typical manufacturing setting across many industries from automotive to pharmaceutical. Energy consumption and the latest completion time of customer orders are usually conflicting objectives. Therefore, we first formulate the problem as a novel multi-objective mixed integer programming (MIP) model and propose an augmented epsilon-constraint method for finding the Pareto-optimal solutions. Also, an effective multi-objective metaheuristic algorithm. Refined Iterated Pareto Greedy (RIPG), is developed to solve large instances in reasonable time. Our proposed methods are benchmarked using small, medium, and large-size instances to evaluate their efficiency. Two well-known algorithms are adopted for comparing our novel approaches. The computational results show the effectiveness of our method. 

**Abstract (ZH)**: 非可再生能量资源稀缺、供给中的地缘政治问题、不断上升的价格以及气候影响迫使全球经济开发更高效的能量解决方案。制造部门并非不受这一挑战的影响，因其是最大的能量消费者之一。能量高效调度是一种吸引制造企业减少能耗的方法，因为它可以快速部署并能立即显示效果。在本研究中，我们探讨了带有阻塞约束的混合流水车间调度问题（BHFS），旨在最小化最晚完工时间和总体能耗，这是众多行业从汽车到制药的典型制造环境。能量消耗和客户订单的最晚完工时间通常是一个相互矛盾的目标。因此，我们首先将问题形式化为一种新型多目标混合整数规划（MIP）模型，并提出了一种扩展的ε约束方法来寻找帕累托最优解。同时，我们开发了一种有效的多目标元启发式算法——精炼迭代帕累托贪婪算法（RIPG），以在合理时间内解决大型实例。我们使用小、中、大型实例来评估所提出方法的效率，并采用两种知名算法进行比较。计算结果表明了我们方法的有效性。 

---
# TopInG: Topologically Interpretable Graph Learning via Persistent Rationale Filtration 

**Title (ZH)**: TopInG: 基于持久性解析过滤的拓扑可解释图学习 

**Authors**: Cheng Xin, Fan Xu, Xin Ding, Jie Gao, Jiaxin Ding  

**Link**: [PDF](https://arxiv.org/pdf/2510.05102)  

**Abstract**: Graph Neural Networks (GNNs) have shown remarkable success across various scientific fields, yet their adoption in critical decision-making is often hindered by a lack of interpretability. Recently, intrinsically interpretable GNNs have been studied to provide insights into model predictions by identifying rationale substructures in graphs. However, existing methods face challenges when the underlying rationale subgraphs are complex and varied. In this work, we propose TopInG: Topologically Interpretable Graph Learning, a novel topological framework that leverages persistent homology to identify persistent rationale subgraphs. TopInG employs a rationale filtration learning approach to model an autoregressive generation process of rationale subgraphs, and introduces a self-adjusted topological constraint, termed topological discrepancy, to enforce a persistent topological distinction between rationale subgraphs and irrelevant counterparts. We provide theoretical guarantees that our loss function is uniquely optimized by the ground truth under specific conditions. Extensive experiments demonstrate TopInG's effectiveness in tackling key challenges, such as handling variform rationale subgraphs, balancing predictive performance with interpretability, and mitigating spurious correlations. Results show that our approach improves upon state-of-the-art methods on both predictive accuracy and interpretation quality. 

**Abstract (ZH)**: 拓扑可解释图学习：基于持久同调的TopInG 

---
# Learning to Interpret Weight Differences in Language Models 

**Title (ZH)**: 学习解释语言模型中的权重差异 

**Authors**: Avichal Goel, Yoon Kim, Nir Shavit, Tony T. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05092)  

**Abstract**: Finetuning (pretrained) language models is a standard approach for updating their internal parametric knowledge and specializing them to new tasks and domains. However, the corresponding model weight changes ("weight diffs") are not generally interpretable. While inspecting the finetuning dataset can give a sense of how the model might have changed, these datasets are often not publicly available or are too large to work with directly. Towards the goal of comprehensively understanding weight diffs in natural language, we introduce Diff Interpretation Tuning (DIT), a method that trains models to describe their own finetuning-induced modifications. Our approach uses synthetic, labeled weight diffs to train a DIT adapter, which can be applied to a compatible finetuned model to make it describe how it has changed. We demonstrate in two proof-of-concept settings (reporting hidden behaviors and summarizing finetuned knowledge) that our method enables models to describe their finetuning-induced modifications using accurate natural language descriptions. 

**Abstract (ZH)**: fine-tuning 预训练语言模型是更新其内部参数知识并将模型专门化于新任务和领域的一种标准方法。然而，相应的模型权重变化（“权重差异”）通常不具备解释性。虽然可以检查 fine-tuning 数据集来了解模型可能的变化，但这些数据集往往无法公开，或者太大而无法直接处理。为了全面理解自然语言中的权重差异，我们引入了差分解释微调（DIT）方法，该方法训练模型描述自己的 fine-tuning 引起的修改。我们的方法使用合成的、标记的权重差异来训练一个 DIT 调整器，该调整器可以应用于兼容的 fine-tuned 模型，使其能够描述自身的变化。我们在两种概念验证设置中（报告隐藏行为和总结 fine-tuning 知识）展示了我们的方法能够使模型使用准确的自然语言描述来描述 fine-tuning 引起的修改。 

---
# HybridFlow: Quantification of Aleatoric and Epistemic Uncertainty with a Single Hybrid Model 

**Title (ZH)**: HybridFlow：单一混合模型在定量评估 aleatoric 不确定性和 epistemic 不确定性方面的应用 

**Authors**: Peter Van Katwyk, Karianne J. Bergen  

**Link**: [PDF](https://arxiv.org/pdf/2510.05054)  

**Abstract**: Uncertainty quantification is critical for ensuring robustness in high-stakes machine learning applications. We introduce HybridFlow, a modular hybrid architecture that unifies the modeling of aleatoric and epistemic uncertainty by combining a Conditional Masked Autoregressive normalizing flow for estimating aleatoric uncertainty with a flexible probabilistic predictor for epistemic uncertainty. The framework supports integration with any probabilistic model class, allowing users to easily adapt HybridFlow to existing architectures without sacrificing predictive performance. HybridFlow improves upon previous uncertainty quantification frameworks across a range of regression tasks, such as depth estimation, a collection of regression benchmarks, and a scientific case study of ice sheet emulation. We also provide empirical results of the quantified uncertainty, showing that the uncertainty quantified by HybridFlow is calibrated and better aligns with model error than existing methods for quantifying aleatoric and epistemic uncertainty. HybridFlow addresses a key challenge in Bayesian deep learning, unifying aleatoric and epistemic uncertainty modeling in a single robust framework. 

**Abstract (ZH)**: 高风险机器学习应用中，不确定性量化对于确保鲁棒性至关重要。我们引入了HybridFlow，这是一种模块化的混合架构，通过结合条件掩码自回归归一化流来估算aleatoric不确定性以及灵活的概率预测器来估算epistemic不确定性，从而统一建模这两种不确定性。该框架支持与任何概率模型类的集成，使用户能够轻松适应现有架构而不牺牲预测性能。HybridFlow在深度估计等多种回归任务、一系列回归基准测试以及一个关于冰盖模拟的科学案例研究中，改进了之前的各种不确定性量化框架。此外，我们提供了量化的不确定性结果，表明HybridFlow量化出的不确定性是校准良好的，并且更好地与模型错误相一致，超过了现有方法的量化效果。HybridFlow解决了贝叶斯深度学习中的一个关键挑战，即在单个稳健的框架中统一建模aleatoric和epistemic不确定性。 

---
# Graph-Aware Diffusion for Signal Generation 

**Title (ZH)**: 图感知扩散信号生成 

**Authors**: Sergio Rozada, Vimal K. B., Andrea Cavallo, Antonio G. Marques, Hadi Jamali-Rad, Elvin Isufi  

**Link**: [PDF](https://arxiv.org/pdf/2510.05036)  

**Abstract**: We study the problem of generating graph signals from unknown distributions defined over given graphs, relevant to domains such as recommender systems or sensor networks. Our approach builds on generative diffusion models, which are well established in vision and graph generation but remain underexplored for graph signals. Existing methods lack generality, either ignoring the graph structure in the forward process or designing graph-aware mechanisms tailored to specific domains. We adopt a forward process that incorporates the graph through the heat equation. Rather than relying on the standard formulation, we consider a time-warped coefficient to mitigate the exponential decay of the drift term, yielding a graph-aware generative diffusion model (GAD). We analyze its forward dynamics, proving convergence to a Gaussian Markov random field with covariance parametrized by the graph Laplacian, and interpret the backward dynamics as a sequence of graph-signal denoising problems. Finally, we demonstrate the advantages of GAD on synthetic data, real traffic speed measurements, and a temperature sensor network. 

**Abstract (ZH)**: 我们研究了从给定图上未知分布生成图信号的问题，这与推荐系统或传感器网络等领域相关。我们的方法基于生成性扩散模型，这类模型在视觉和图生成领域已有很好的应用，但在图信号生成方面仍处于探索阶段。现有方法缺乏通用性，要么在正向过程中忽视了图结构，要么为特定领域设计了图感知机制。我们采用通过热方程引入图的正向过程。不同于传统的公式化方法，我们考虑了一个时间扭曲系数来缓解漂移项的指数衰减，从而构建了一个图感知生成扩散模型（GAD）。我们分析了其正向动力学，证明其收敛到以图拉普拉斯矩阵参数化协方差的高斯马尔可夫随机场，并将反向动力学解释为一系列图信号去噪问题。最后，我们在合成数据、实际交通速度测量以及温度传感器网络上展示了GAD的优势。 

---
# Rethinking Langevin Thompson Sampling from A Stochastic Approximation Perspective 

**Title (ZH)**: 从随机逼近视角重新审视 Langevin 汀伯特采样 

**Authors**: Weixin Wang, Haoyang Zheng, Guang Lin, Wei Deng, Pan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.05023)  

**Abstract**: Most existing approximate Thompson Sampling (TS) algorithms for multi-armed bandits use Stochastic Gradient Langevin Dynamics (SGLD) or its variants in each round to sample from the posterior, relaxing the need for conjugacy assumptions between priors and reward distributions in vanilla TS. However, they often require approximating a different posterior distribution in different round of the bandit problem. This requires tricky, round-specific tuning of hyperparameters such as dynamic learning rates, causing challenges in both theoretical analysis and practical implementation. To alleviate this non-stationarity, we introduce TS-SA, which incorporates stochastic approximation (SA) within the TS framework. In each round, TS-SA constructs a posterior approximation only using the most recent reward(s), performs a Langevin Monte Carlo (LMC) update, and applies an SA step to average noisy proposals over time. This can be interpreted as approximating a stationary posterior target throughout the entire algorithm, which further yields a fixed step-size, a unified convergence analysis framework, and improved posterior estimates through temporal averaging. We establish near-optimal regret bounds for TS-SA, with a simplified and more intuitive theoretical analysis enabled by interpreting the entire algorithm as a simulation of a stationary SGLD process. Our empirical results demonstrate that even a single-step Langevin update with certain warm-up outperforms existing methods substantially on bandit tasks. 

**Abstract (ZH)**: TS-SA: Incorporating Stochastic Approximation within the Thompson Sampling Framework 

---
# AWARE, Beyond Sentence Boundaries: A Contextual Transformer Framework for Identifying Cultural Capital in STEM Narratives 

**Title (ZH)**: AWARE，超越句子边界：一种在STEM叙事中识别文化资本的上下文Transformer框架 

**Authors**: Khalid Mehtab Khan, Anagha Kulkarni  

**Link**: [PDF](https://arxiv.org/pdf/2510.04983)  

**Abstract**: Identifying cultural capital (CC) themes in student reflections can offer valuable insights that help foster equitable learning environments in classrooms. However, themes such as aspirational goals or family support are often woven into narratives, rather than appearing as direct keywords. This makes them difficult to detect for standard NLP models that process sentences in isolation. The core challenge stems from a lack of awareness, as standard models are pre-trained on general corpora, leaving them blind to the domain-specific language and narrative context inherent to the data. To address this, we introduce AWARE, a framework that systematically attempts to improve a transformer model's awareness for this nuanced task. AWARE has three core components: 1) Domain Awareness, adapting the model's vocabulary to the linguistic style of student reflections; 2) Context Awareness, generating sentence embeddings that are aware of the full essay context; and 3) Class Overlap Awareness, employing a multi-label strategy to recognize the coexistence of themes in a single sentence. Our results show that by making the model explicitly aware of the properties of the input, AWARE outperforms a strong baseline by 2.1 percentage points in Macro-F1 and shows considerable improvements across all themes. This work provides a robust and generalizable methodology for any text classification task in which meaning depends on the context of the narrative. 

**Abstract (ZH)**: 识别学生反思中的文化资本主题可以为促进公平的学习环境提供宝贵的见解。然而，诸如抱负目标或家庭支持等主题往往融入叙事中，而非直接作为关键词出现。这使得标准的自然语言处理模型难以检测。核心挑战在于模型缺乏领域意识，因为标准模型预训练于一般语料库，无法识别领域特定的语言和叙事背景。为此，我们提出了AWARE框架，系统性地提高模型对这一细腻任务的意识。AWARE有三个核心组成部分：1）领域意识，调整模型词汇以适应学生反思的语言风格；2）上下文意识，生成感知全文背景的句子嵌入；3）类别重叠意识，采用多标签策略识别单句中存在的多个主题。我们的结果显示，通过使模型明确意识到输入的特性，AWARE在宏F1分数上比强 baseline 高出2.1个百分点，并且在所有主题上均表现出显著改进。这项工作提供了任何依赖叙事上下文的文本分类任务的稳健且可泛化的方法论。 

---
# Embracing Discrete Search: A Reasonable Approach to Causal Structure Learning 

**Title (ZH)**: 拥抱离散搜索：一种因果结构学习的合理方法 

**Authors**: Marcel Wienöbst, Leonard Henckel, Sebastian Weichwald  

**Link**: [PDF](https://arxiv.org/pdf/2510.04970)  

**Abstract**: We present FLOP (Fast Learning of Order and Parents), a score-based causal discovery algorithm for linear models. It pairs fast parent selection with iterative Cholesky-based score updates, cutting run-times over prior algorithms. This makes it feasible to fully embrace discrete search, enabling iterated local search with principled order initialization to find graphs with scores at or close to the global optimum. The resulting structures are highly accurate across benchmarks, with near-perfect recovery in standard settings. This performance calls for revisiting discrete search over graphs as a reasonable approach to causal discovery. 

**Abstract (ZH)**: 基于评分的快速顺序和祖先学习：一种线性模型的因果发现算法 

---
# MuFFIN: Multifaceted Pronunciation Feedback Model with Interactive Hierarchical Neural Modeling 

**Title (ZH)**: 多面发音反馈模型：交互式分层神经建模 

**Authors**: Bi-Cheng Yan, Ming-Kang Tsai, Berlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.04956)  

**Abstract**: Computer-assisted pronunciation training (CAPT) manages to facilitate second-language (L2) learners to practice pronunciation skills by offering timely and instructive feedback. To examine pronunciation proficiency from multiple facets, existing methods for CAPT broadly fall into two categories: mispronunciation detection and diagnosis (MDD) as well as automatic pronunciation assessment (APA). The former aims to pinpoint phonetic pronunciation errors and provide diagnostic feedback, while the latter seeks instead to quantify pronunciation proficiency pertaining to various aspects. Despite the natural complementarity between MDD and APA, researchers and practitioners, however, often treat them as independent tasks with disparate modeling paradigms. In light of this, we in this paper first introduce MuFFIN, a Multi-Faceted pronunciation Feedback model with an Interactive hierarchical Neural architecture, to jointly address the tasks of MDD and APA. To better capture the nuanced distinctions between phonemes in the feature space, a novel phoneme-contrastive ordinal regularization mechanism is then put forward to optimize the proposed model to generate more phoneme-discriminative features while factoring in the ordinality of the aspect scores. In addition, to address the intricate data imbalance problem in MDD, we design a simple yet effective training objective, which is specifically tailored to perturb the outputs of a phoneme classifier with the phoneme-specific variations, so as to better render the distribution of predicted phonemes meanwhile considering their mispronunciation characteristics. A series of experiments conducted on the Speechocean762 benchmark dataset demonstrates the efficacy of our method in relation to several cutting-edge baselines, showing state-of-the-art performance on both the APA and MDD tasks. 

**Abstract (ZH)**: 计算机辅助发音训练中的多维度反馈模型：MuFFIN及其在发音错误检测与诊断及自动发音评估中的应用 

---
# Feasibility-Aware Decision-Focused Learning for Predicting Parameters in the Constraints 

**Title (ZH)**: 面向可行性的决策聚焦学习：预测约束条件下的参数 

**Authors**: Jayanta Mandi, Marianne Defresne, Senne Berden, Tias Guns  

**Link**: [PDF](https://arxiv.org/pdf/2510.04951)  

**Abstract**: When some parameters of a constrained optimization problem (COP) are uncertain, this gives rise to a predict-then-optimize (PtO) problem, comprising two stages -- the prediction of the unknown parameters from contextual information and the subsequent optimization using those predicted parameters. Decision-focused learning (DFL) implements the first stage by training a machine learning (ML) model to optimize the quality of the decisions made using the predicted parameters. When parameters in the constraints of a COP are predicted, the predicted parameters can lead to infeasible solutions. Therefore, it is important to simultaneously manage both feasibility and decision quality. We develop a DFL framework for predicting constraint parameters in a generic COP. While prior works typically assume that the underlying optimization problem is a linear program (LP) or integer linear program (ILP), our approach makes no such assumption. We derive two novel loss functions based on maximum likelihood estimation (MLE): the first one penalizes infeasibility (by penalizing when the predicted parameters lead to infeasible solutions), and the second one penalizes suboptimal decisions (by penalizing when the true optimal solution is infeasible under the predicted parameters). We introduce a single tunable parameter to form a weighted average of the two losses, allowing decision-makers to balance suboptimality and feasibility. We experimentally demonstrate that adjusting this parameter provides a decision-maker the control over the trade-off between the two. Moreover, across several COP instances, we find that for a single value of the tunable parameter, our method matches the performance of the existing baselines on suboptimality and feasibility. 

**Abstract (ZH)**: 在约束优化问题参数不确定时的预测-优化问题及决策聚焦学习框架 

---
# A First Context-Free Grammar Applied to Nawatl Corpora Augmentation 

**Title (ZH)**: 一种初步的应用于纳瓦特尔语语料库扩充的上下文无关文法 

**Authors**: Juan-José Guzmán-Landa, Juan-Manuel Torres-Moreno, Miguel Figueroa-Saavedra, Ligia Quintana-Torres, Martha-Lorena Avendaño-Garrido, Graham Ranger  

**Link**: [PDF](https://arxiv.org/pdf/2510.04945)  

**Abstract**: In this article we introduce a context-free grammar (CFG) for the Nawatl language. Nawatl (or Nahuatl) is an Amerindian language of the $\pi$-language type, i.e. a language with few digital resources, in which the corpora available for machine learning are virtually non-existent. The objective here is to generate a significant number of grammatically correct artificial sentences, in order to increase the corpora available for language model training. We want to show that a grammar enables us significantly to expand a corpus in Nawatl which we call $\pi$-\textsc{yalli}. The corpus, thus enriched, enables us to train algorithms such as FastText and to evaluate them on sentence-level semantic tasks. Preliminary results show that by using the grammar, comparative improvements are achieved over some LLMs. However, it is observed that to achieve more significant improvement, grammars that model the Nawatl language even more effectively are required. 

**Abstract (ZH)**: 本文介绍了纳瓦特尔语的上下文自由文法。纳瓦特尔语（纳乔特尔语或纳赫都特尔语）是一种美洲原住民语言，属于$\pi$语言类型，即资源稀缺的语言，其中可用于机器学习的语料库几乎不存在。本文旨在生成大量语法正确的合成句子，以增加用于语言模型训练的语料库。我们希望表明，文法能够显著扩展纳瓦特尔语$\pi$-\textsc{yalli}语料库。通过丰富后的语料库能够用于训练如FastText等算法，并在句子级语义任务上评估它们。初步结果显示，通过使用文法，相对于一些预训练语言模型(LLMs)，取得了比较明显的改进。然而，观察到要实现更为显著的改进，需要更有效地建模纳瓦特尔语言的文法。 

---
# Unsupervised Active Learning via Natural Feature Progressive Framework 

**Title (ZH)**: 无监督主动学习基于自然特征渐进框架 

**Authors**: Yuxi Liu, Catherine Lalman, Yimin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04939)  

**Abstract**: The effectiveness of modern deep learning models is predicated on the availability of large-scale, human-annotated datasets, a process that is notoriously expensive and time-consuming. While Active Learning (AL) offers a strategic solution by labeling only the most informative and representative data, its iterative nature still necessitates significant human involvement. Unsupervised Active Learning (UAL) presents an alternative by shifting the annotation burden to a single, post-selection step. Unfortunately, prevailing UAL methods struggle to achieve state-of-the-art performance. These approaches typically rely on local, gradient-based scoring for sample importance estimation, which not only makes them vulnerable to ambiguous and noisy data but also hinders their capacity to select samples that adequately represent the full data distribution. Moreover, their use of shallow, one-shot linear selection falls short of a true UAL paradigm. In this paper, we propose the Natural Feature Progressive Framework (NFPF), a UAL method that revolutionizes how sample importance is measured. At its core, NFPF employs a Specific Feature Learning Machine (SFLM) to effectively quantify each sample's contribution to model performance. We further utilize the SFLM to define a powerful Reconstruction Difference metric for initial sample selection. Our comprehensive experiments show that NFPF significantly outperforms all established UAL methods and achieves performance on par with supervised AL methods on vision datasets. Detailed ablation studies and qualitative visualizations provide compelling evidence for NFPF's superior performance, enhanced robustness, and improved data distribution coverage. 

**Abstract (ZH)**: 现代深度学习模型的有效性依赖于大规模的人工标注数据集，这一过程既昂贵又耗时。主动学习（AL）通过仅标注最有信息性和代表性的数据，提供了一种战略性的解决方案，但其迭代性质仍然需要大量的人类参与。无监督主动学习（UAL）通过将注释负担转移到单一的选择步骤之后，提供了一种替代方案。然而，现有UAL方法难以达到最先进的性能。这些方法通常依赖于局部、基于梯度的得分来进行样本重要性估计，这不仅使它们容易受到模糊和噪声数据的影响，也限制了它们选择能够充分代表数据分布的样本的能力。此外，它们采用浅层的一次性线性选择方法未能真正体现UAL的范式。在本文中，我们提出了自然特征渐进行因子方法（NFPF），这是一种UAL方法，重新定义了样本重要性度量的方式。NFPF的核心在于使用特定特征学习机（SFLM）有效量化每个样本对模型性能的贡献。我们进一步利用SFLM定义了强大的重构差异度量方法，用于初始样本选择。我们的全面实验表明，NFPF显著优于所有现有UAL方法，并在视觉数据集上达到了与监督AL方法相当的性能。详细的消融研究和定性可视化提供了NFPF在性能、鲁棒性及数据分布覆盖方面的优越性的有力证据。 

---
# ONNX-Net: Towards Universal Representations and Instant Performance Prediction for Neural Architectures 

**Title (ZH)**: ONNX-Net：通往通用表示和神经架构即时性能预测的道路 

**Authors**: Shiwen Qin, Alexander Auras, Shay B. Cohen, Elliot J. Crowley, Michael Moeller, Linus Ericsson, Jovita Lukasik  

**Link**: [PDF](https://arxiv.org/pdf/2510.04938)  

**Abstract**: Neural architecture search (NAS) automates the design process of high-performing architectures, but remains bottlenecked by expensive performance evaluation. Most existing studies that achieve faster evaluation are mostly tied to cell-based search spaces and graph encodings tailored to those individual search spaces, limiting their flexibility and scalability when applied to more expressive search spaces. In this work, we aim to close the gap of individual search space restrictions and search space dependent network representations. We present ONNX-Bench, a benchmark consisting of a collection of neural networks in a unified format based on ONNX files. ONNX-Bench includes all open-source NAS-bench-based neural networks, resulting in a total size of more than 600k {architecture, accuracy} pairs. This benchmark allows creating a shared neural network representation, ONNX-Net, able to represent any neural architecture using natural language descriptions acting as an input to a performance predictor. This text-based encoding can accommodate arbitrary layer types, operation parameters, and heterogeneous topologies, enabling a single surrogate to generalise across all neural architectures rather than being confined to cell-based search spaces. Experiments show strong zero-shot performance across disparate search spaces using only a small amount of pretraining samples, enabling the unprecedented ability to evaluate any neural network architecture instantly. 

**Abstract (ZH)**: 基于ONNX的神经架构基准（ONNX-Bench）：统一格式下的神经网络表示与性能预测 

---
# AURA Score: A Metric For Holistic Audio Question Answering Evaluation 

**Title (ZH)**: AURA评分：全方位音频问答评估指标 

**Authors**: Satvik Dixit, Soham Deshmukh, Bhiksha Raj  

**Link**: [PDF](https://arxiv.org/pdf/2510.04934)  

**Abstract**: Audio Question Answering (AQA) is a key task for evaluating Audio-Language Models (ALMs), yet assessing open-ended responses remains challenging. Existing metrics used for AQA such as BLEU, METEOR and BERTScore, mostly adapted from NLP and audio captioning, rely on surface similarity and fail to account for question context, reasoning, and partial correctness. To address the gap in literature, we make three contributions in this work. First, we introduce AQEval to enable systematic benchmarking of AQA metrics. It is the first benchmark of its kind, consisting of 10k model responses annotated by multiple humans for their correctness and relevance. Second, we conduct a comprehensive analysis of existing AQA metrics on AQEval, highlighting weak correlation with human judgment, especially for longer answers. Third, we propose a new metric - AURA score, to better evaluate open-ended model responses. On AQEval, AURA achieves state-of-the-art correlation with human ratings, significantly outperforming all baselines. Through this work, we aim to highlight the limitations of current AQA evaluation methods and motivate better metrics. We release both the AQEval benchmark and the AURA metric to support future research in holistic AQA evaluation. 

**Abstract (ZH)**: 音频问答（AQA）是评估音频语言模型（ALMs）的关键任务，但评估开放性回答仍具挑战性。现有的AQA评估指标如BLEU、METEOR和BERTScore主要源自NLP和音频字幕领域，依赖表面相似性，未能考虑到问题背景、推理和部分正确性。为弥补文献中的这一空白，我们在本文中做出了三项贡献。首先，我们引入AQEval以实现AQA评估指标的系统基准测试，这是首个包含10,000个由多人标注正确性和相关性的模型回答的基准。其次，我们在AQEval上对现有AQA评估指标进行了全面分析，特别指出这些指标与人工判断的相关性较弱，尤其是在较长的回答中。最后，我们提出一个新的评估指标——AURA分值，以更好地评估开放性模型回答。在AQEval上，AURA实现了与人工评分的最佳相关性，显著优于所有基线。通过本文，我们旨在突出当前AQA评估方法的局限性，并激励开发更好的评估指标。我们同时发布了AQEval基准和AURA分值，以支持未来的全面AQA评估研究。 

---
# Federated Self-Supervised Learning for Automatic Modulation Classification under Non-IID and Class-Imbalanced Data 

**Title (ZH)**: federated self-supervised learning for automatic modulation classification under non-iid and class-imbalanced data 

**Authors**: Usman Akram, Yiyue Chen, Haris Vikalo  

**Link**: [PDF](https://arxiv.org/pdf/2510.04927)  

**Abstract**: Training automatic modulation classification (AMC) models on centrally aggregated data raises privacy concerns, incurs communication overhead, and often fails to confer robustness to channel shifts. Federated learning (FL) avoids central aggregation by training on distributed clients but remains sensitive to class imbalance, non-IID client distributions, and limited labeled samples. We propose FedSSL-AMC, which trains a causal, time-dilated CNN with triplet-loss self-supervision on unlabeled I/Q sequences across clients, followed by per-client SVMs on small labeled sets. We establish convergence of the federated representation learning procedure and a separability guarantee for the downstream classifier under feature noise. Experiments on synthetic and over-the-air datasets show consistent gains over supervised FL baselines under heterogeneous SNR, carrier-frequency offsets, and non-IID label partitions. 

**Abstract (ZH)**: 联邦学习中基于跨客户端聚合数据训练自动调制分类模型存在隐私问题、通信开销，并且往往无法应对信道偏移。我们提出FedSSL-AMC，该方法在客户端上的无标签I/Q序列上训练因果时延卷积神经网络，并结合三元组损失的自我监督学习，随后在小的标签集中训练每个客户端的SVM。我们在特征噪声下建立了联合表征学习过程的收敛性和下游分类器的可分性保证。实验结果在合成和空中通信数据集上显示，与监督联邦学习基线相比，在异构信噪比、载波频率偏移和非独立同分布标签分区情况下都能获得一致的性能提升。 

---
# REN: Anatomically-Informed Mixture-of-Experts for Interstitial Lung Disease Diagnosis 

**Title (ZH)**: REN：解剖导向的专家混合模型用于间质性肺病诊断 

**Authors**: Alec K. Peltekian, Halil Ertugrul Aktas, Gorkem Durak, Kevin Grudzinski, Bradford C. Bemiss, Carrie Richardson, Jane E. Dematte, G. R. Scott Budinger, Anthony J. Esposito, Alexander Misharin, Alok Choudhary, Ankit Agrawal, Ulas Bagci  

**Link**: [PDF](https://arxiv.org/pdf/2510.04923)  

**Abstract**: Mixture-of-Experts (MoE) architectures have significantly contributed to scalable machine learning by enabling specialized subnetworks to tackle complex tasks efficiently. However, traditional MoE systems lack domain-specific constraints essential for medical imaging, where anatomical structure and regional disease heterogeneity strongly influence pathological patterns. Here, we introduce Regional Expert Networks (REN), the first anatomically-informed MoE framework tailored specifically for medical image classification. REN leverages anatomical priors to train seven specialized experts, each dedicated to distinct lung lobes and bilateral lung combinations, enabling precise modeling of region-specific pathological variations. Multi-modal gating mechanisms dynamically integrate radiomics biomarkers and deep learning (DL) features (CNN, ViT, Mamba) to weight expert contributions optimally. Applied to interstitial lung disease (ILD) classification, REN achieves consistently superior performance: the radiomics-guided ensemble reached an average AUC of 0.8646 +/- 0.0467, a +12.5 percent improvement over the SwinUNETR baseline (AUC 0.7685, p = 0.031). Region-specific experts further revealed that lower-lobe models achieved AUCs of 0.88-0.90, surpassing DL counterparts (CNN: 0.76-0.79) and aligning with known disease progression patterns. Through rigorous patient-level cross-validation, REN demonstrates strong generalizability and clinical interpretability, presenting a scalable, anatomically-guided approach readily extensible to other structured medical imaging applications. 

**Abstract (ZH)**: 区域专家网络（REN）：基于解剖学的医疗图像分类框架 

---
# Glocal Information Bottleneck for Time Series Imputation 

**Title (ZH)**: 全局与局部信息瓶颈时间序列插补 

**Authors**: Jie Yang, Kexin Zhang, Guibin Zhang, Philip S. Yu, Kaize Ding  

**Link**: [PDF](https://arxiv.org/pdf/2510.04910)  

**Abstract**: Time Series Imputation (TSI), which aims to recover missing values in temporal data, remains a fundamental challenge due to the complex and often high-rate missingness in real-world scenarios. Existing models typically optimize the point-wise reconstruction loss, focusing on recovering numerical values (local information). However, we observe that under high missing rates, these models still perform well in the training phase yet produce poor imputations and distorted latent representation distributions (global information) in the inference phase. This reveals a critical optimization dilemma: current objectives lack global guidance, leading models to overfit local noise and fail to capture global information of the data. To address this issue, we propose a new training paradigm, Glocal Information Bottleneck (Glocal-IB). Glocal-IB is model-agnostic and extends the standard IB framework by introducing a Global Alignment loss, derived from a tractable mutual information approximation. This loss aligns the latent representations of masked inputs with those of their originally observed counterparts. It helps the model retain global structure and local details while suppressing noise caused by missing values, giving rise to better generalization under high missingness. Extensive experiments on nine datasets confirm that Glocal-IB leads to consistently improved performance and aligned latent representations under missingness. Our code implementation is available in this https URL. 

**Abstract (ZH)**: 全局与局部信息瓶颈在时间序列插补中的应用 

---
# FreshBrew: A Benchmark for Evaluating AI Agents on Java Code Migration 

**Title (ZH)**: FreshBrew: 一个评估AI代理在Java代码迁移任务上的benchmark 

**Authors**: Victor May, Diganta Misra, Yanqi Luo, Anjali Sridhar, Justine Gehring, Silvio Soares Ribeiro Junior  

**Link**: [PDF](https://arxiv.org/pdf/2510.04852)  

**Abstract**: AI coding assistants are rapidly becoming integral to modern software development. A key challenge in this space is the continual need to migrate and modernize codebases in response to evolving software ecosystems. Traditionally, such migrations have relied on rule-based systems and human intervention. With the advent of powerful large language models (LLMs), AI-driven agentic frameworks offer a promising alternative-but their effectiveness has not been systematically evaluated. In this paper, we introduce FreshBrew, a novel benchmark for evaluating AI agents on project-level Java migrations, with a specific focus on measuring an agent's ability to preserve program semantics and avoid reward hacking, which we argue requires projects with high test coverage for a rigorous and reliable evaluation. We benchmark several state-of-the-art LLMs, and compare their performance against established rule-based tools. Our evaluation of AI agents on this benchmark of 228 repositories shows that the top-performing model, Gemini 2.5 Flash, can successfully migrate 52.3 percent of projects to JDK 17. Our empirical analysis reveals novel insights into the critical strengths and limitations of current agentic approaches, offering actionable insights into their real-world applicability. Our empirical study reveals failure modes of current AI agents in realistic Java modernization tasks, providing a foundation for evaluating trustworthy code-migration systems. By releasing FreshBrew, we aim to facilitate rigorous, reproducible evaluation and catalyze progress in AI-driven codebase modernization. 

**Abstract (ZH)**: 基于AI的代码辅助工具正迅速成为现代软件开发的一部分。这一领域的一个关键挑战是对不断演化的软件生态系统做出持续的代码库迁移与现代化需求。传统上，这类迁移依赖于基于规则的系统和人工干预。随着强大语言模型（LLMs）的出现，基于AI的自主框架提供了一种有前景的替代方案——但其有效性尚未系统评估。本文介绍了FreshBrew，一个新型基准，用于评估AI代理在项目级别Java迁移中的表现，特别关注测量代理保留程序语义和避免奖励 hijack 的能力，我们认为这需要具有高测试覆盖率的项目以实现严格的可靠评估。我们对几种最先进的LLMs进行了基准测试，并将它们的性能与传统基于规则的工具进行了比较。对这一基准的228个仓库进行评估表明，表现最好的模型Gemini 2.5 Flash成功迁移到JDK 17的项目比例为52.3%。我们的实证分析揭示了当前自主方法的关键强项和局限性，提供了其实用见解，以指导其实际应用。我们的实证研究揭示了当前AI代理在现实的Java现代化任务中的失败模式，为评估可信的代码迁移系统奠定了基础。通过发布FreshBrew，我们旨在促进严格的可重复评估，并推动基于AI的代码库现代化的进步。 

---
# Distributionally Robust Causal Abstractions 

**Title (ZH)**: 分布鲁棒因果抽象 

**Authors**: Yorgos Felekis, Theodoros Damoulas, Paris Giampouras  

**Link**: [PDF](https://arxiv.org/pdf/2510.04842)  

**Abstract**: Causal Abstraction (CA) theory provides a principled framework for relating causal models that describe the same system at different levels of granularity while ensuring interventional consistency between them. Recently, several approaches for learning CAs have been proposed, but all assume fixed and well-specified exogenous distributions, making them vulnerable to environmental shifts and misspecification. In this work, we address these limitations by introducing the first class of distributionally robust CAs and their associated learning algorithms. The latter cast robust causal abstraction learning as a constrained min-max optimization problem with Wasserstein ambiguity sets. We provide theoretical results, for both empirical and Gaussian environments, leading to principled selection of the level of robustness via the radius of these sets. Furthermore, we present empirical evidence across different problems and CA learning methods, demonstrating our framework's robustness not only to environmental shifts but also to structural model and intervention mapping misspecification. 

**Abstract (ZH)**: 因果抽象（CA）理论提供了一种原则性的框架，用于关联描述同一系统在不同粒度水平上的因果模型，同时确保它们之间的干预期贯性。最近，已经提出了几种学习CA的方法，但所有方法都假设固定的且很好地指定的外生分布，这使它们容易受到环境变化和模型指定错误的影响。在这项工作中，我们通过引入第一类具备分布鲁棒性的CA及其相关学习算法来解决这些问题。后者将鲁棒因果抽象学习表述为带有 Wasserstein 模糊集合的约束最小最大优化问题。我们提供了理论结果，涵盖了经验性和高斯环境，通过这些集合的半径来指导鲁棒性水平的合理选择。此外，我们通过不同问题和CA学习方法的实证研究，展示了该框架不仅对环境变化，而且对结构模型和干预映射的指定错误具有鲁棒性。 

---
# Bond-Centered Molecular Fingerprint Derivatives: A BBBP Dataset Study 

**Title (ZH)**: 基于键的分子指纹衍生物：一个BBBP数据集研究 

**Authors**: Guillaume Godin  

**Link**: [PDF](https://arxiv.org/pdf/2510.04837)  

**Abstract**: Bond Centered FingerPrint (BCFP) are a complementary, bond-centric alternative to Extended-Connectivity Fingerprints (ECFP). We introduce a static BCFP that mirrors the bond-convolution used by directed message-passing GNNs like ChemProp, and evaluate it with a fast rapid Random Forest model on Brain-Blood Barrier Penetration (BBBP) classification task. Across stratified cross-validation, concatenating ECFP with BCFP consistently improves AUROC and AUPRC over either descriptor alone, as confirmed by Turkey HSD multiple-comparison analysis. Among radii, r = 1 performs best; r = 2 does not yield statistically separable gains under the same test. We further propose BCFP-Sort&Slice, a simple feature-combination scheme that preserves the out-of-vocabulary (OOV) count information native to ECFP count vectors while enabling compact unhashed concatenation of BCFP variants. We also outperform the MGTP prediction on our BBBP evaluation, using such composite new features bond and atom features. These results show that lightweight, bond-centered descriptors can complement atom-centered circular fingerprints and provide strong, fast baselines for BBBP prediction. 

**Abstract (ZH)**: Bond-Centered FingerPrint (BCFP) 是 Extended-Connectivity Fingerprints (ECFP) 的补充，以键为中心的替代方案。我们引入了一种静态BCFP，其结构与定向消息传递GNN如ChemProp中使用的键卷积相对应，并使用快速随机森林模型对其在血脑屏障渗透性（BBBP）分类任务中进行了评估。在分层交叉验证中，将ECFP与BCFP进行连接一致地提高了AUROC和AUPRC，这得到了Turkey HSD多重比较分析的证实。在不同的半径中，r = 1表现最佳；r = 2在相同的测试中没有提供统计上可分离的增益。我们还提出了BCFP-Sort&Slice，这是一种简单的特征组合方案，保留了ECFP计数向量固有的未见过词汇（OOV）计数信息，并允许紧凑的未哈希BCFP变体的连接。我们还使用这些复合新特征（包括键和原子特征）在BBBP评估中超越了MGTP预测。这些结果表明，轻量级的键为中心描述子可以补充原子为中心的环形指纹，并为BBBP预测提供强大的快速基准。 

---
# On Predicting Post-Click Conversion Rate via Counterfactual Inference 

**Title (ZH)**: 基于反事实推理的点击后转换率预测 

**Authors**: Junhyung Ahn, Sanghack Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.04816)  

**Abstract**: Accurately predicting conversion rate (CVR) is essential in various recommendation domains such as online advertising systems and e-commerce. These systems utilize user interaction logs, which consist of exposures, clicks, and conversions. CVR prediction models are typically trained solely based on clicked samples, as conversions can only be determined following clicks. However, the sparsity of clicked instances necessitates the collection of a substantial amount of logs for effective model training. Recent works address this issue by devising frameworks that leverage non-clicked samples. While these frameworks aim to reduce biases caused by the discrepancy between clicked and non-clicked samples, they often rely on heuristics. Against this background, we propose a method to counterfactually generate conversion labels for non-clicked samples by using causality as a guiding principle, attempting to answer the question, "Would the user have converted if he or she had clicked the recommended item?" Our approach is named the Entire Space Counterfactual Inference Multi-task Model (ESCIM). We initially train a structural causal model (SCM) of user sequential behaviors and conduct a hypothetical intervention (i.e., click) on non-clicked items to infer counterfactual CVRs. We then introduce several approaches to transform predicted counterfactual CVRs into binary counterfactual conversion labels for the non-clicked samples. Finally, the generated samples are incorporated into the training process. Extensive experiments on public datasets illustrate the superiority of the proposed algorithm. Online A/B testing further empirically validates the effectiveness of our proposed algorithm in real-world scenarios. In addition, we demonstrate the improved performance of the proposed method on latent conversion data, showcasing its robustness and superior generalization capabilities. 

**Abstract (ZH)**: 准确预测转换率（CVR）在在线广告系统和电子商务等推荐领域至关重要。这些系统利用了用户交互日志，包括曝光、点击和转换。CVR预测模型通常仅基于点击样本进行训练，因为只有在发生点击之后才能确定转换。然而，点击样本的稀疏性要求收集大量的日志才能有效训练模型。近期的研究通过设计利用非点击样本的框架来应对这一问题。尽管这些框架旨在减少点击与非点击样本之间差异造成的偏差，但它们通常依赖于启发式方法。在此背景下，我们提出了一种方法，利用因果性作为指导原则，反事实生成非点击样本的转换标签，试图回答“如果用户点击推荐的商品会发生转换吗？”这一问题。我们的方法名为全面空间反事实推理多任务模型（ESCIM）。我们首先训练用户序列行为的结构因果模型（SCM），并对非点击商品进行假设干预（即点击）以推断反事实的CVR。然后，我们提出几种方法将预测的反事实CVR转化为非点击样本的二元反事实转换标签。最后，生成的样本被纳入训练过程。在公开数据集上的广泛实验表明了所提算法的优越性。在线A/B测试进一步在实际场景中实证验证了所提算法的有效性。此外，我们在潜在转换数据上展示了所提方法的改进性能，证明了其鲁棒性和更强的泛化能力。 

---
# When Do Credal Sets Stabilize? Fixed-Point Theorems for Credal Set Updates 

**Title (ZH)**: 当信度集达到稳定状态？信度集更新的不动点定理 

**Authors**: Michele Caprio, Siu Lun Chau, Krikamol Muandet  

**Link**: [PDF](https://arxiv.org/pdf/2510.04769)  

**Abstract**: Many machine learning algorithms rely on iterative updates of uncertainty representations, ranging from variational inference and expectation-maximization, to reinforcement learning, continual learning, and multi-agent learning. In the presence of imprecision and ambiguity, credal sets -- closed, convex sets of probability distributions -- have emerged as a popular framework for representing imprecise probabilistic beliefs. Under such imprecision, many learning problems in imprecise probabilistic machine learning (IPML) may be viewed as processes involving successive applications of update rules on credal sets. This naturally raises the question of whether this iterative process converges to stable fixed points -- or, more generally, under what conditions on the updating mechanism such fixed points exist, and whether they can be attained. We provide the first analysis of this problem and illustrate our findings using Credal Bayesian Deep Learning as a concrete example. Our work demonstrates that incorporating imprecision into the learning process not only enriches the representation of uncertainty, but also reveals structural conditions under which stability emerges, thereby offering new insights into the dynamics of iterative learning under imprecision. 

**Abstract (ZH)**: 许多机器学习算法依赖于不确定性表示的迭代更新，包括变分推断、期望最大化、强化学习、连续学习和多代理学习。在模糊性和不确定性存在的情况下，信念集合——闭合且凸的概率分布集合——已成为表示不确定性概率信念的一种流行框架。在这样的不确定性下，模糊概率机器学习（IPML）中的许多学习问题可以视为在信念集合上相继应用更新规则的过程。这自然引出了一个问题，即这种迭代过程是否会收敛到稳定点——或者说，在什么条件下这样的稳定点存在，以及是否可以达到。我们对该问题进行了首次分析，并通过信任贝叶斯深度学习作为具体例子来阐述我们的发现。我们的研究表明，将不确定性纳入学习过程不仅丰富了不确定性表示，还揭示了稳定性出现的结构性条件，从而为在不确定性下的迭代学习动态提供了新见解。 

---
# Fisher-Bingham-like normalizing flows on the sphere 

**Title (ZH)**: 球面上的Fisher-Bingham-like 归一化流 

**Authors**: Thorsten Glüsenkamp  

**Link**: [PDF](https://arxiv.org/pdf/2510.04762)  

**Abstract**: A generic D-dimensional Gaussian can be conditioned or projected onto the D-1 unit sphere, thereby leading to the well-known Fisher-Bingham (FB) or Angular Gaussian (AG) distribution families, respectively. These are some of the most fundamental distributions on the sphere, yet cannot straightforwardly be written as a normalizing flow except in two special cases: the von-Mises Fisher in D=3 and the central angular Gaussian in any D. In this paper, we describe how to generalize these special cases to a family of normalizing flows that behave similarly to the full FB or AG family in any D. We call them "zoom-linear-project" (ZLP)-Fisher flows. Unlike a normal Fisher-Bingham distribution, their composition allows to gradually add complexity as needed. Furthermore, they can naturally handle conditional density estimation with target distributions that vary by orders of magnitude in scale - a setting that is important in astronomical applications but that existing flows often struggle with. A particularly useful member of the new family is the Kent analogue that can cheaply upgrade any flow in this situation to yield better performance. 

**Abstract (ZH)**: 一种通用的D维高斯分布可以条件化或投影到D-1单位球上，从而分别得到广为人知的Fisher-Bingham (FB) 或角度高斯 (AG) 分布族。这些是在球上一些最基础的分布，但除了二维和三维的特殊情况外（D=2和D=3），它们无法直接作为归一化流来表示。本文描述了如何将这些特殊情况泛化为一个在任何D维下行为相似的归一化流家族，我们称之为“缩放-线性-投影”（ZLP）Fisher流。与标准的Fisher-Bingham分布不同，它们的组合形式允许逐步增加所需复杂性。此外，它们可以自然处理目标分布尺度相差很大的条件概率估计——一个在天文学应用中很重要的情况，但现有的流形模型往往难以应对。新家族中的一个特别有用成员是肯特近似，它可以在这种情况下以低成本提升任何流形模型，以获得更好的性能。 

---
# Agile Software Effort Estimation using Regression Techniques 

**Title (ZH)**: 使用回归技术进行敏捷软件努力估计 

**Authors**: Sisay Deresa Sima, Ayalew Belay Habtie  

**Link**: [PDF](https://arxiv.org/pdf/2510.04760)  

**Abstract**: Software development effort estimation is one of the most critical aspect in software development process, as the success or failure of the entire project depends on the accuracy of estimations. Researchers are still conducting studies on agile effort estimation. The aim of this research is to develop a story point based agile effort estimation model using LASSO and Elastic Net regression techniques. The experimental work is applied to the agile story point approach using 21 software projects collected from six firms. The two algorithms are trained using their default parameters and tuned grid search with 5-fold cross-validation to get an enhanced model. The experiment result shows LASSO regression achieved better predictive performance PRED (8%) and PRED (25%) results of 100.0, MMRE of 0.0491, MMER of 0.0551, MdMRE of 0.0593, MdMER of 0.063, and MSE of 0.0007. The results are also compared with other related literature. 

**Abstract (ZH)**: 基于LASSO和弹性网回归的敏捷故事点估算模型研究 

---
# A New Digital Divide? Coder Worldviews, the Slop Economy, and Democracy in the Age of AI 

**Title (ZH)**: 一个新的数字鸿沟？编码者的世界观、斜坡经济与人工智能时代民主 

**Authors**: Jason Miklian, Kristian Hoelscher  

**Link**: [PDF](https://arxiv.org/pdf/2510.04755)  

**Abstract**: Digital technologies are transforming democratic life in conflicting ways. This article bridges two perspectives to unpack these tensions. First, we present an original survey of software developers in Silicon Valley, interrogating how coder worldviews, ethics, and workplace cultures shape the democratic potential and social impact of the technologies they build. Results indicate that while most developers recognize the power of their products to influence civil liberties and political discourse, they often face ethical dilemmas and top-down pressures that can lead to design choices undermining democratic ideals. Second, we critically investigate these findings in the context of an emerging new digital divide, not of internet access but of information quality. We interrogate the survey findings in the context of the Slop Economy, in which billions of users unable to pay for high-quality content experience an internet dominated by low-quality, AI-generated ad-driven content. We find a reinforcing cycle between tech creator beliefs and the digital ecosystems they spawn. We discuss implications for democratic governance, arguing for more ethically informed design and policy interventions to help bridge the digital divide to ensure that technological innovation supports rather than subverts democratic values in the next chapter of the digital age. 

**Abstract (ZH)**: 数字技术以冲突的方式重塑民主生活： coder世界观、伦理与职场文化如何影响技术的民主潜力及其社会影响的视角融合与新兴数字鸿沟的批判性考察 

---
# Curved Boolean Logic: A Contextual Generalization of Propositional Logic with Algorithmic Consequences 

**Title (ZH)**: 曲面布尔逻辑：命题逻辑的一种情境化拓展及其算法后果 

**Authors**: Maximilian R. P. von Liechtenstein  

**Link**: [PDF](https://arxiv.org/pdf/2510.04716)  

**Abstract**: Curved Boolean Logic (CBL) generalizes propositional logic by allowing local truth assignments that do not extend to a single global valuation, analogous to curvature in geometry. We give equivalent sheaf and exclusivity-graph semantics and a context-aware proof calculus that is conservative in the flat limit. We formalize CBL-SAT and basic complexity (NP-complete in general) and present operational operators (CBL-AC and CBL-CONS) that prune contradictions earlier on classical hardware. We model noise with iid, AR(1)-correlated, and adversarial bounded perturbations and provide permutation-based significance with Benjamini-Hochberg FDR control. A Colab-ready notebook (ancillary files) regenerates all figures and statistics. We position CBL relative to KCBS, CSW, and sheaf frameworks and outline links to SAT/CSP and robustness/adapter stability in large language models. 

**Abstract (ZH)**: 曲率布尔逻辑（CBL）通过允许不扩展为单一全局估值的局部真值赋值，泛化命题逻辑，类似于几何中的曲率。我们给出了等价的丛和排他图语义，并提出了一种上下文感知的证明 calculus，在平坦极限下保守。我们形式化了 CBL-SAT 和基础复杂性（通常为 NP 完全），并提出了早期修剪矛盾的操作符（CBL-AC 和 CBL-CONS），这些操作符可以在经典硬件上运行。我们用独立同分布、AR(1)-相关和 adversarial 有界扰动模型噪声，并提供了基于排列的意义性检验，采用 Benjamini-Hochberg FDR 控制。附有 Colab 可用的笔记本（附录文件）可以重新生成所有图表和统计数据。我们将 CBL 相对于 KCBS、CSW 和丛框架进行定位，并概述其与 SAT/CSP 和大语言模型的健壯性/适配器稳定性之间的联系。 

---
# The Bayesian Origin of the Probability Weighting Function in Human Representation of Probabilities 

**Title (ZH)**: 人类对概率表示中的概率权重函数的贝叶斯起源 

**Authors**: Xin Tong, Thi Thu Uyen Hoang, Xue-Xin Wei, Michael Hahn  

**Link**: [PDF](https://arxiv.org/pdf/2510.04698)  

**Abstract**: Understanding the representation of probability in the human mind has been of great interest to understanding human decision making. Classical paradoxes in decision making suggest that human perception distorts probability magnitudes. Previous accounts postulate a Probability Weighting Function that transforms perceived probabilities; however, its motivation has been debated. Recent work has sought to motivate this function in terms of noisy representations of probabilities in the human mind. Here, we present an account of the Probability Weighting Function grounded in rational inference over optimal decoding from noisy neural encoding of quantities. We show that our model accurately accounts for behavior in a lottery task and a dot counting task. It further accounts for adaptation to a bimodal short-term prior. Taken together, our results provide a unifying account grounding the human representation of probability in rational inference. 

**Abstract (ZH)**: 理解人类头脑中概率的表示对于理解人类决策有着重要意义。决策中的经典悖论表明，人类感知会扭曲概率幅度。以往的解释假设了一个概率加权函数，将感知到的概率进行转换；然而，这一函数的动机一直存在争议。近期的研究试图从人类头脑中对概率的嘈杂表示出发来解释这一函数。在这里，我们提出一个基于最优解码从嘈杂神经编码量进行合理推理的概率加权函数的解释。我们展示了我们的模型能够准确解释彩票任务和点计任务中的行为表现，并进一步解释了短期双模态先验的适应性。综上，我们的结果提供了一个统一的解释，将人类对概率的表示与合理推理联系起来。 

---
# How does the optimizer implicitly bias the model merging loss landscape? 

**Title (ZH)**: 优化器如何隐式偏置模型合并损失景观？ 

**Authors**: Chenxiang Zhang, Alexander Theus, Damien Teney, Antonio Orvieto, Jun Pang, Sjouke Mauw  

**Link**: [PDF](https://arxiv.org/pdf/2510.04686)  

**Abstract**: Model merging methods combine models with different capabilities into a single one while maintaining the same inference cost. Two popular approaches are linear interpolation, which linearly interpolates between model weights, and task arithmetic, which combines task vectors obtained by the difference between finetuned and base models. While useful in practice, what properties make merging effective are poorly understood. This paper explores how the optimization process affects the loss landscape geometry and its impact on merging success. We show that a single quantity -- the effective noise scale -- unifies the impact of optimizer and data choices on model merging. Across architectures and datasets, the effectiveness of merging success is a non-monotonic function of effective noise, with a distinct optimum. Decomposing this quantity, we find that larger learning rates, stronger weight decay, smaller batch sizes, and data augmentation all independently modulate the effective noise scale, exhibiting the same qualitative trend. Unlike prior work that connects optimizer noise to the flatness or generalization of individual minima, we show that it also affects the global loss landscape, predicting when independently trained solutions can be merged. Our findings broaden the understanding of how optimization shapes the loss landscape geometry and its downstream consequences for model merging, suggesting the possibility of further manipulating the training dynamics to improve merging effectiveness. 

**Abstract (ZH)**: 模型合并方法通过保持相同的推理成本将具有不同能力的模型合并为一个模型。两种流行的方法是线性内插，它在线性模型权重之间进行内插，以及任务算术，它通过精调模型和基础模型之间的差异获得任务向量并将其综合。虽然在实践中这些方法很有用，但使其有效的特定属性尚不明确。本文探讨了优化过程如何影响损失景观几何结构及其对合并成功率的影响。我们展示了一个单一的量——有效噪声尺度——统一了优化器和数据选择对模型合并的影响。在不同架构和数据集上，合并成功率的有效噪声尺度是非单调函数，具有一个明显的最优值。分解这一量，我们发现较大的学习率、较强的权重衰减、较小的批量大小和数据增强都独立地调节有效噪声尺度，表现出相同的基本趋势。与之前将优化器噪声与个体极小值的平坦度或泛化能力联系起来的研究不同，我们展示了有效噪声尺度还影响全局损失景观，预测独立训练的解决方案可以合并的情况。我们的发现开拓了对于优化如何塑造损失景观几何结构及其对模型合并的下游后果的理解，建议进一步操控训练动力学以提高合并效果的可能性。 

---
# Semantic Channel Equalization Strategies for Deep Joint Source-Channel Coding 

**Title (ZH)**: 深层联合源信道编码中的语义通道均衡策略 

**Authors**: Lorenzo Pannacci, Simone Fiorellino, Mario Edoardo Pandolfo, Emilio Calvanese Strinati, Paolo Di Lorenzo  

**Link**: [PDF](https://arxiv.org/pdf/2510.04674)  

**Abstract**: Deep joint source-channel coding (DeepJSCC) has emerged as a powerful paradigm for end-to-end semantic communications, jointly learning to compress and protect task-relevant features over noisy channels. However, existing DeepJSCC schemes assume a shared latent space at transmitter (TX) and receiver (RX) - an assumption that fails in multi-vendor deployments where encoders and decoders cannot be co-trained. This mismatch introduces "semantic noise", degrading reconstruction quality and downstream task performance. In this paper, we systematize and evaluate methods for semantic channel equalization for DeepJSCC, introducing an additional processing stage that aligns heterogeneous latent spaces under both physical and semantic impairments. We investigate three classes of aligners: (i) linear maps, which admit closed-form solutions; (ii) lightweight neural networks, offering greater expressiveness; and (iii) a Parseval-frame equalizer, which operates in zero-shot mode without the need for training. Through extensive experiments on image reconstruction over AWGN and fading channels, we quantify trade-offs among complexity, data efficiency, and fidelity, providing guidelines for deploying DeepJSCC in heterogeneous AI-native wireless networks. 

**Abstract (ZH)**: 基于深度学习的语义信道均衡方法研究与评估：在异构AI原生无线网络中的部署指南 

---
# Noise or Signal? Deconstructing Contradictions and An Adaptive Remedy for Reversible Normalization in Time Series Forecasting 

**Title (ZH)**: 噪声还是信号？拆解时间序列预测中可逆归一化的矛盾并开发自适应修复方法 

**Authors**: Fanzhe Fu, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04667)  

**Abstract**: Reversible Instance Normalization (RevIN) is a key technique enabling simple linear models to achieve state-of-the-art performance in time series forecasting. While replacing its non-robust statistics with robust counterparts (termed R$^2$-IN) seems like a straightforward improvement, our findings reveal a far more complex reality. This paper deconstructs the perplexing performance of various normalization strategies by identifying four underlying theoretical contradictions. Our experiments provide two crucial findings: first, the standard RevIN catastrophically fails on datasets with extreme outliers, where its MSE surges by a staggering 683\%. Second, while the simple R$^2$-IN prevents this failure and unexpectedly emerges as the best overall performer, our adaptive model (A-IN), designed to test a diagnostics-driven heuristic, unexpectedly suffers a complete and systemic failure. This surprising outcome uncovers a critical, overlooked pitfall in time series analysis: the instability introduced by a simple or counter-intuitive heuristic can be more damaging than the statistical issues it aims to solve. The core contribution of this work is thus a new, cautionary paradigm for time series normalization: a shift from a blind search for complexity to a diagnostics-driven analysis that reveals not only the surprising power of simple baselines but also the perilous nature of naive adaptation. 

**Abstract (ZH)**: 可逆实例归一化 (RevIN) 是一种使简单线性模型在时间序列预测中达到最佳性能的关键技术。虽然用稳健统计替换其不稳健的统计（称为 R\(^2\)-IN）似乎是一种简单的改进，但我们的发现揭示了一个更加复杂的现实。本文通过识别四种潜在的理论矛盾，拆解各种归一化策略令人困惑的性能表现。我们的实验提供了两个关键发现：首先，标准 RevIN 在包含极端异常值的数据集上 Catastrophically 失效，其 MSE 突增 683%。其次，虽然简单的 R\(^2\)-IN 防止了这一失败并意外地成为表现最佳的整体模型，我们设计用于测试诊断驱动启发式的自适应模型 (A-IN) 也意外地经历了完全且系统性的失败。这一令人惊讶的结果揭示了时间序列分析中的一个重要且被忽视的陷阱：简单或反直观启发式引入的稳定性问题可能比它试图解决的统计问题更为有害。本文的核心贡献是一种新的、警示性的时间序列归一化范式：从盲目追求复杂性转向基于诊断的分析，不仅揭示了简单基线的惊人力量，还揭示了盲目适应的危险性质。 

---
# Predictive Feature Caching for Training-free Acceleration of Molecular Geometry Generation 

**Title (ZH)**: 训练驱动以外加速分子几何生成的预测特征缓存 

**Authors**: Johanna Sommer, John Rachwan, Nils Fleischmann, Stephan Günnemann, Bertrand Charpentier  

**Link**: [PDF](https://arxiv.org/pdf/2510.04646)  

**Abstract**: Flow matching models generate high-fidelity molecular geometries but incur significant computational costs during inference, requiring hundreds of network evaluations. This inference overhead becomes the primary bottleneck when such models are employed in practice to sample large numbers of molecular candidates. This work discusses a training-free caching strategy that accelerates molecular geometry generation by predicting intermediate hidden states across solver steps. The proposed method operates directly on the SE(3)-equivariant backbone, is compatible with pretrained models, and is orthogonal to existing training-based accelerations and system-level optimizations. Experiments on the GEOM-Drugs dataset demonstrate that caching achieves a twofold reduction in wall-clock inference time at matched sample quality and a speedup of up to 3x compared to the base model with minimal sample quality degradation. Because these gains compound with other optimizations, applying caching alongside other general, lossless optimizations yield as much as a 7x speedup. 

**Abstract (ZH)**: 无需缓存的训练-free策略通过预测求解步骤中的中间隐状态加速分子几何生成 

---
# Fairness in Repeated Matching: A Maximin Perspective 

**Title (ZH)**: 重复匹配中的公平性：最大最小视角 

**Authors**: Eugene Lim, Tzeh Yuan Neoh, Nicholas Teh  

**Link**: [PDF](https://arxiv.org/pdf/2510.04624)  

**Abstract**: We study a sequential decision-making model where a set of items is repeatedly matched to the same set of agents over multiple rounds. The objective is to determine a sequence of matchings that either maximizes the utility of the least advantaged agent at the end of all rounds (optimal) or at the end of every individual round (anytime optimal). We investigate the computational challenges associated with finding (anytime) optimal outcomes and demonstrate that these problems are generally computationally intractable. However, we provide approximation algorithms, fixed-parameter tractable algorithms, and identify several special cases whereby the problem(s) can be solved efficiently. Along the way, we also establish characterizations of Pareto-optimal/maximum matchings, which may be of independent interest to works in matching theory and house allocation. 

**Abstract (ZH)**: 我们研究了一种序贯决策模型，其中一组物品在多轮中重复匹配给同一组代理。目标是确定一系列匹配方案，以最大化所有轮次结束后最不利代理的效益（最优）或每轮结束后最不利代理的效益（任意时最优）。我们探讨了找到（任意时）最优结果的计算挑战，并证明了这些问题是通常计算上不可约化的问题。然而，我们提供了近似算法、固定参数可处理算法，并确定了几种可以高效解决的问题特殊情况。在过程中，我们还建立了Pareto最优/最大匹配的特性，这些特性对匹配理论和住房分配领域的研究可能具有独立兴趣。 

---
# Design Process of a Self Adaptive Smart Serious Games Ecosystem 

**Title (ZH)**: 自适应智能严肃游戏生态系统的设计过程 

**Authors**: X. Tao, P. Chen, M. Tsami, F. Khayati, M. Eckert  

**Link**: [PDF](https://arxiv.org/pdf/2510.04615)  

**Abstract**: This paper outlines the design vision and planned evolution of Blexer v3, a modular and AI-driven rehabilitation ecosystem based on serious games. Building on insights from previous versions of the system, we propose a new architecture that aims to integrate multimodal sensing, real-time reasoning, and intelligent control. The envisioned system will include distinct modules for data collection, user state inference, and gameplay adaptation. Key features such as dynamic difficulty adjustment (DDA) and procedural content generation (PCG) are also considered to support personalized interventions. We present the complete conceptual framework of Blexer v3, which defines the modular structure and data flow of the system. This serves as the foundation for the next phase: the development of a functional prototype and its integration into clinical rehabilitation scenarios. 

**Abstract (ZH)**: 基于严肃游戏的可模块化和AI驱动的康复生态系统Blexer v3的设计愿景与规划演化 

---
# Accountability Capture: How Record-Keeping to Support AI Transparency and Accountability (Re)shapes Algorithmic Oversight 

**Title (ZH)**: 责任捕获：记录保存如何塑造AI透明度和问责制下的算法监督 

**Authors**: Shreya Chappidi, Jennifer Cobbe, Chris Norval, Anjali Mazumder, Jatinder Singh  

**Link**: [PDF](https://arxiv.org/pdf/2510.04609)  

**Abstract**: Accountability regimes typically encourage record-keeping to enable the transparency that supports oversight, investigation, contestation, and redress. However, implementing such record-keeping can introduce considerations, risks, and consequences, which so far remain under-explored. This paper examines how record-keeping practices bring algorithmic systems within accountability regimes, providing a basis to observe and understand their effects. For this, we introduce, describe, and elaborate 'accountability capture' -- the re-configuration of socio-technical processes and the associated downstream effects relating to record-keeping for algorithmic accountability. Surveying 100 practitioners, we evidence and characterise record-keeping issues in practice, identifying their alignment with accountability capture. We further document widespread record-keeping practices, tensions between internal and external accountability requirements, and evidence of employee resistance to practices imposed through accountability capture. We discuss these and other effects for surveillance, privacy, and data protection, highlighting considerations for algorithmic accountability communities. In all, we show that implementing record-keeping to support transparency in algorithmic accountability regimes can itself bring wider implications -- an issue requiring greater attention from practitioners, researchers, and policymakers alike. 

**Abstract (ZH)**: 问责制度通常鼓励记录保存以促进支持监督、调查、争议和补救的透明度。然而，实施这样的记录保存可能会引入考虑因素、风险和后果，这些目前尚未得到充分探索。本文探讨了记录保存实践如何将算法系统纳入问责制度，为观察和理解其影响提供基础。为此，我们引入、描述并阐述了“问责捕获”——即社会技术过程的重新配置及其相关的下游影响，特别是与算法问责相关的记录保存。通过对100名从业人员的调查，我们揭示并刻画了实践中的记录保存问题，发现这些问题与问责捕获相一致。我们进一步记录了广泛存在的记录保存实践、内部和外部问责要求之间的紧张关系以及问责捕获施加的实践所导致的员工抵制证据。我们讨论了这些以及其他对监控、隐私和数据保护的影响，并强调了算法问责共同体的考量。总体而言，我们表明，为了支持算法问责制度中的透明度而实施记录保存本身可能会带来更广泛的影响——一个需要从业者、研究人员和政策制定者共同给予更多关注的问题。 

---
# Computing Wasserstein Barycenters through Gradient Flows 

**Title (ZH)**: 通过梯度流计算威劳夫tein 广义中心 

**Authors**: Eduardo Fernandes Montesuma, Yassir Bendou, Mike Gartrell  

**Link**: [PDF](https://arxiv.org/pdf/2510.04602)  

**Abstract**: Wasserstein barycenters provide a powerful tool for aggregating probability measures, while leveraging the geometry of their ambient space. Existing discrete methods suffer from poor scalability, as they require access to the complete set of samples from input measures. We address this issue by recasting the original barycenter problem as a gradient flow in the Wasserstein space. Our approach offers two advantages. First, we achieve scalability by sampling mini-batches from the input measures. Second, we incorporate functionals over probability measures, which regularize the barycenter problem through internal, potential, and interaction energies. We present two algorithms for empirical and Gaussian mixture measures, providing convergence guarantees under the Polyak-Łojasiewicz inequality. Experimental validation on toy datasets and domain adaptation benchmarks show that our methods outperform previous discrete and neural net-based methods for computing Wasserstein barycenters. 

**Abstract (ZH)**: Wasserstein 贴中心提供了一种强大的方法来聚合概率测度，同时利用其环境空间的几何结构。现有的离散方法由于需要访问输入测度的完整样本集而表现出较差的可扩展性。我们通过将原始中心问题重新表述为 Wasserstein 空间的梯度流来解决这一问题。我们的方法有两个优点。首先，我们通过从输入测度中采样小批量实现可扩展性。其次，我们结合了概率测度上的泛函，通过内部、潜在和交互能量对中心问题进行正则化。我们提出了两种算法，分别适用于经验分布和高斯混合测度，并在 Polyak-Łojasiewicz 不等式下提供了收敛性保证。实验验证在玩具数据集和域适应基准测试上表明，我们的方法在计算 Wasserstein 贴中心方面优于之前的离散和神经网络基方法。 

---
# Deep learning framework for predicting stochastic take-off and die-out of early spreading 

**Title (ZH)**: 基于深度学习的早期传播随机起飞和消亡预测框架 

**Authors**: Wenchao He, Tao Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.04574)  

**Abstract**: Large-scale outbreaks of epidemics, misinformation, or other harmful contagions pose significant threats to human society, yet the fundamental question of whether an emerging outbreak will escalate into a major epidemic or naturally die out remains largely unaddressed. This problem is challenging, partially due to inadequate data during the early stages of outbreaks and also because established models focus on average behaviors of large epidemics rather than the stochastic nature of small transmission chains. Here, we introduce the first systematic framework for forecasting whether initial transmission events will amplify into major outbreaks or fade into extinction during early stages, when intervention strategies can still be effectively implemented. Using extensive data from stochastic spreading models, we developed a deep learning framework that predicts early-stage spreading outcomes in real-time. Validation across Erdős-Rényi and Barabási-Albert networks with varying infectivity levels shows our method accurately forecasts stochastic spreading events well before potential outbreaks, demonstrating robust performance across different network structures and infectivity this http URL address the challenge of sparse data during early outbreak stages, we further propose a pretrain-finetune framework that leverages diverse simulation data for pretraining and adapts to specific scenarios through targeted fine-tuning. The pretrain-finetune framework consistently outperforms baseline models, achieving superior performance even when trained on limited scenario-specific data. To our knowledge, this work presents the first framework for predicting stochastic take-off versus die-out. This framework provides valuable insights for epidemic preparedness and public health decision-making, enabling more informed early intervention strategies. 

**Abstract (ZH)**: 大规模传染病、虚假信息或其他有害 contagions的大规模爆发对人类社会构成了重大威胁，但关于新兴爆发是否会升级为大规模流行病还是自然消亡的基本问题仍缺乏解答。为了解决这一挑战，本文首次提出了一种系统框架，在早期阶段预测初始传播事件是否会放大成大规模爆发或逐渐消亡，此时仍可以有效实施干预策略。借助广泛的数据，我们开发了一个深度学习框架，在实际过程中实时预测早期传播结果。对不同传染性水平的Erdős-Rényi和Barabási-Albert网络的验证显示，该方法能够在潜在爆发前准确预测随机传播事件，展示了在不同网络结构和传染性水平下的一致性能。为进一步应对早期爆发阶段数据稀疏的挑战，我们提出了一个预训练-微调框架，利用多样化的模拟数据进行预训练，并通过针对性的微调适应特定场景。预训练-微调框架在各种基准模型中表现优异，即使仅使用有限的场景特定数据也能实现更优性能。据我们所知，本文提出了预测随机起飞与消亡的第一个框架，为传染病准备和公共卫生决策提供了有价值见解，有助于制定更加知情的早期干预策略。 

---
# Toward a Unified Geometry Understanding: Riemannian Diffusion Framework for Graph Generation and Prediction 

**Title (ZH)**: 统一几何理解的方向：黎曼扩散框架下的图生成与预测 

**Authors**: Yisen Gao, Xingcheng Fu, Qingyun Sun, Jianxin Li, Xianxian Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.04522)  

**Abstract**: Graph diffusion models have made significant progress in learning structured graph data and have demonstrated strong potential for predictive tasks. Existing approaches typically embed node, edge, and graph-level features into a unified latent space, modeling prediction tasks including classification and regression as a form of conditional generation. However, due to the non-Euclidean nature of graph data, features of different curvatures are entangled in the same latent space without releasing their geometric potential. To address this issue, we aim to construt an ideal Riemannian diffusion model to capture distinct manifold signatures of complex graph data and learn their distribution. This goal faces two challenges: numerical instability caused by exponential mapping during the encoding proces and manifold deviation during diffusion generation. To address these challenges, we propose GeoMancer: a novel Riemannian graph diffusion framework for both generation and prediction tasks. To mitigate numerical instability, we replace exponential mapping with an isometric-invariant Riemannian gyrokernel approach and decouple multi-level features onto their respective task-specific manifolds to learn optimal representations. To address manifold deviation, we introduce a manifold-constrained diffusion method and a self-guided strategy for unconditional generation, ensuring that the generated data remains aligned with the manifold signature. Extensive experiments validate the effectiveness of our approach, demonstrating superior performance across a variety of tasks. 

**Abstract (ZH)**: 基于黎曼流形的图扩散模型：一种新的图扩散框架以捕捉复杂图数据的 manifold 特征并进行生成与预测任务 

---
# Compressed Convolutional Attention: Efficient Attention in a Compressed Latent Space 

**Title (ZH)**: 压缩卷积注意力：在压缩潜空间中的高效注意力 

**Authors**: Tomas Figliolia, Nicholas Alonso, Rishi Iyer, Quentin Anthony, Beren Millidge  

**Link**: [PDF](https://arxiv.org/pdf/2510.04476)  

**Abstract**: Multi-headed Attention's (MHA) quadratic compute and linearly growing KV-cache make long-context transformers expensive to train and serve. Prior works such as Grouped Query Attention (GQA) and Multi-Latent Attention (MLA) shrink the cache, speeding decode, but leave compute, which determines prefill and training speed, largely unchanged. We introduce Compressed Convolutional Attention (CCA), a novel attention method which down-projects queries, keys, and values and performs the entire attention operation inside the shared latent space. This simple design dramatically cuts parameters, KV-cache, and FLOPs all at once by the desired compression factor. Because CCA is orthogonal to head-sharing, we combine the two to form Compressed Convolutional Grouped Query Attention (CCGQA), which further tightens the compute-bandwidth Pareto frontier so that users can tune compression toward either FLOP or memory limits without sacrificing quality. Experiments show that CCGQA consistently outperforms both GQA and MLA at equal KV-cache compression on dense and MoE models. Additionally, we show that CCGQA outperforms all other attention methods on MoE models with half the KV-cache of GQA and MLA, achieving an 8x KV-cache compression with no drop in performance compared to standard MHA. CCA and CCGQA also dramatically reduce the FLOP cost of attention which leads to substantially faster training and prefill than existing methods. On H100 GPUs, our fused CCA/CCGQA kernel reduces prefill latency by about 1.7x at a sequence length of 16k relative to MHA, and accelerates backward by about 1.3x. 

**Abstract (ZH)**: Compressed Convolutional Attention (CCA)及其在长上下文变换器训练和服务中的应用 

---
# Inverse Mixed-Integer Programming: Learning Constraints then Objective Functions 

**Title (ZH)**: 逆混合整数规划：学习约束条件然后目标函数 

**Authors**: Akira Kitaoka  

**Link**: [PDF](https://arxiv.org/pdf/2510.04455)  

**Abstract**: In mixed-integer linear programming, data-driven inverse optimization that learns the objective function and the constraints from observed data plays an important role in constructing appropriate mathematical models for various fields, including power systems and scheduling. However, to the best of our knowledge, there is no known method for learning both the objective functions and the constraints. In this paper, we propose a two-stage method for a class of problems where the objective function is expressed as a linear combination of functions and the constraints are represented by functions and thresholds. Specifically, our method first learns the constraints and then learns the objective function. On the theoretical side, we show the proposed method can solve inverse optimization problems in finite dataset, develop statistical learning theory in pseudometric spaces and sub-Gaussian distributions, and construct a statistical learning for inverse optimization. On the experimental side, we demonstrate that our method is practically applicable for scheduling problems formulated as integer linear programmings with up to 100 decision variables, which are typical in real-world settings. 

**Abstract (ZH)**: 在混合整数线性规划中，基于数据的逆优化方法通过从观察数据中学习目标函数和约束条件，在构建适用于电力系统和调度等领域的方法模型方面发挥着重要作用。然而，据我们所知，尚无已知方法能够同时学习目标函数和约束条件。本文提出一种两阶段方法，适用于目标函数表示为函数线性组合、约束表示为函数和阈值的问题类。具体而言，该方法首先学习约束，然后学习目标函数。从理论层面看，我们展示了所提出方法能够在有限数据集上解决逆优化问题，建立了伪距离空间和亚高斯分布下的统计学习理论，并构建了逆优化的统计学习方法。从实验层面看，我们证明了该方法对于包含多达100个决策变量的整数线性规划调度问题具有实际应用价值，这类问题是现实中常见的。 

---
# Partial Information Decomposition via Normalizing Flows in Latent Gaussian Distributions 

**Title (ZH)**: 局部信息分解：通过潜高斯分布中的规范化流 

**Authors**: Wenyuan Zhao, Adithya Balachandran, Chao Tian, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04417)  

**Abstract**: The study of multimodality has garnered significant interest in fields where the analysis of interactions among multiple information sources can enhance predictive modeling, data fusion, and interpretability. Partial information decomposition (PID) has emerged as a useful information-theoretic framework to quantify the degree to which individual modalities independently, redundantly, or synergistically convey information about a target variable. However, existing PID methods depend on optimizing over a joint distribution constrained by estimated pairwise probability distributions, which are costly and inaccurate for continuous and high-dimensional modalities. Our first key insight is that the problem can be solved efficiently when the pairwise distributions are multivariate Gaussians, and we refer to this problem as Gaussian PID (GPID). We propose a new gradient-based algorithm that substantially improves the computational efficiency of GPID based on an alternative formulation of the underlying optimization problem. To generalize the applicability to non-Gaussian data, we learn information-preserving encoders to transform random variables of arbitrary input distributions into pairwise Gaussian random variables. Along the way, we resolved an open problem regarding the optimality of joint Gaussian solutions for GPID. Empirical validation in diverse synthetic examples demonstrates that our proposed method provides more accurate and efficient PID estimates than existing baselines. We further evaluate a series of large-scale multimodal benchmarks to show its utility in real-world applications of quantifying PID in multimodal datasets and selecting high-performing models. 

**Abstract (ZH)**: 多模态研究在分析多种信息源的交互以增强预测建模、数据融合和可解释性方面引起了广泛关注。部分信息分解（PID）已成为一种有用的信息论框架，用于量化各个模态独立、冗余或协同地传递目标变量信息的程度。然而，现有的PID方法依赖于在估计的成对概率分布约束下的联合分布优化，这对于连续和高维模态来说成本高且不准确。我们的第一项关键洞察是，当成对分布为多元高斯分布时，该问题可以高效解决，并将此问题称为高斯PID（GPID）。我们提出了一种新的梯度基于算法，该算法在底层优化问题的替代公式基础上显著提高了GPID的计算效率。为了使GPID适用于非高斯数据，我们学习了保信息的编码器，将任意输入分布的随机变量转换为成对的高斯随机变量。在这一过程中，我们解决了GPID中联合高斯解的最优性问题。在多元合成数据示例上的实证验证表明，我们提出的方法提供了比现有基线更准确和高效的PID估计。我们进一步评估了一系列大规模多模态基准，展示了其在多模态数据集中量化PID和选择高性能模型的实际应用中的实用性。 

---
# Reconsidering Requirements Engineering: Human-AI Collaboration in AI-Native Software Development 

**Title (ZH)**: 重新思考需求工程：人工智能原生软件开发中的人机协作 

**Authors**: Mateen Ahmed Abbasi, Petri Ihantola, Tommi Mikkonen, Niko Mäkitalo  

**Link**: [PDF](https://arxiv.org/pdf/2510.04380)  

**Abstract**: Requirement Engineering (RE) is the foundation of successful software development. In RE, the goal is to ensure that implemented systems satisfy stakeholder needs through rigorous requirements elicitation, validation, and evaluation processes. Despite its critical role, RE continues to face persistent challenges, such as ambiguity, conflicting stakeholder needs, and the complexity of managing evolving requirements. A common view is that Artificial Intelligence (AI) has the potential to streamline the RE process, resulting in improved efficiency, accuracy, and management actions. However, using AI also introduces new concerns, such as ethical issues, biases, and lack of transparency. This paper explores how AI can enhance traditional RE practices by automating labor-intensive tasks, supporting requirement prioritization, and facilitating collaboration between stakeholders and AI systems. The paper also describes the opportunities and challenges that AI brings to RE. In particular, the vision calls for ethical practices in AI, along with a much-enhanced collaboration between academia and industry professionals. The focus should be on creating not only powerful but also trustworthy and practical AI solutions ready to adapt to the fast-paced world of software development. 

**Abstract (ZH)**: 软件需求工程（RE）是成功软件开发的基础。在RE中，目标是通过严格的需求获取、验证和评估过程，确保实现的系统满足利益相关者的需求。尽管RE至关重要，但仍面临诸多持久挑战，如不确定性、利益相关者需求冲突以及管理不断演变的需求的复杂性。一种常见观点认为，人工智能（AI）有潜力简化RE过程，提高效率、准确性和管理行动。然而，使用AI也会带来新的问题，如伦理问题、偏差和透明度不足。本文探讨了AI如何通过自动化劳动密集型任务、支持需求优先级排序以及促进利益相关者与AI系统的合作来增强传统RE实践。文章还描述了AI为RE带来的机遇和挑战。特别是，本文提出应在AI伦理实践以及学术界与业界专家之间增强合作方面树立愿景。重点应放在创建不仅强大而且值得信赖且实用的AI解决方案上，这些解决方案能够适应快速发展的软件开发世界。 

---
# Adaptive Weighted Loss for Sequential Recommendations on Sparse Domains 

**Title (ZH)**: 稀疏领域上的自适应加权损失序贯推荐 

**Authors**: Akshay Mittal, Vinay Venkatesh, Krishna Kandi, Shalini Sudarshan  

**Link**: [PDF](https://arxiv.org/pdf/2510.04375)  

**Abstract**: The effectiveness of single-model sequential recommendation architectures, while scalable, is often limited when catering to "power users" in sparse or niche domains. Our previous research, PinnerFormerLite, addressed this by using a fixed weighted loss to prioritize specific domains. However, this approach can be sub-optimal, as a single, uniform weight may not be sufficient for domains with very few interactions, where the training signal is easily diluted by the vast, generic dataset.
This paper proposes a novel, data-driven approach: a Dynamic Weighted Loss function with comprehensive theoretical foundations and extensive empirical validation. We introduce an adaptive algorithm that adjusts the loss weight for each domain based on its sparsity in the training data, assigning a higher weight to sparser domains and a lower weight to denser ones. This ensures that even rare user interests contribute a meaningful gradient signal, preventing them from being overshadowed.
We provide rigorous theoretical analysis including convergence proofs, complexity analysis, and bounds analysis to establish the stability and efficiency of our approach. Our comprehensive empirical validation across four diverse datasets (MovieLens, Amazon Electronics, Yelp Business, LastFM Music) with state-of-the-art baselines (SIGMA, CALRec, SparseEnNet) demonstrates that this dynamic weighting system significantly outperforms all comparison methods, particularly for sparse domains, achieving substantial lifts in key metrics like Recall at 10 and NDCG at 10 while maintaining performance on denser domains and introducing minimal computational overhead. 

**Abstract (ZH)**: 动态加权损失函数在稀疏或 niche 领域中针对“power用户”的顺序推荐架构的有效性探讨 

---
# GDPval: Evaluating AI Model Performance on Real-World Economically Valuable Tasks 

**Title (ZH)**: GDPval: 评价AI模型在实际经济有价值任务中的性能 

**Authors**: Tejal Patwardhan, Rachel Dias, Elizabeth Proehl, Grace Kim, Michele Wang, Olivia Watkins, Simón Posada Fishman, Marwan Aljubeh, Phoebe Thacker, Laurance Fauconnet, Natalie S. Kim, Patrick Chao, Samuel Miserendino, Gildas Chabot, David Li, Michael Sharman, Alexandra Barr, Amelia Glaese, Jerry Tworek  

**Link**: [PDF](https://arxiv.org/pdf/2510.04374)  

**Abstract**: We introduce GDPval, a benchmark evaluating AI model capabilities on real-world economically valuable tasks. GDPval covers the majority of U.S. Bureau of Labor Statistics Work Activities for 44 occupations across the top 9 sectors contributing to U.S. GDP (Gross Domestic Product). Tasks are constructed from the representative work of industry professionals with an average of 14 years of experience. We find that frontier model performance on GDPval is improving roughly linearly over time, and that the current best frontier models are approaching industry experts in deliverable quality. We analyze the potential for frontier models, when paired with human oversight, to perform GDPval tasks cheaper and faster than unaided experts. We also demonstrate that increased reasoning effort, increased task context, and increased scaffolding improves model performance on GDPval. Finally, we open-source a gold subset of 220 tasks and provide a public automated grading service at this http URL to facilitate future research in understanding real-world model capabilities. 

**Abstract (ZH)**: GDPval：评估AI模型在具有经济价值的实际任务上的能力基准 

---
# NegotiationGym: Self-Optimizing Agents in a Multi-Agent Social Simulation Environment 

**Title (ZH)**: 谈判健身房：多智能体社会模拟环境中的自我优化代理 

**Authors**: Shashank Mangla, Chris Hokamp, Jack Boylan, Demian Gholipour Ghalandari, Yuuv Jauhari, Lauren Cassidy, Oisin Duffy  

**Link**: [PDF](https://arxiv.org/pdf/2510.04368)  

**Abstract**: We design and implement NegotiationGym, an API and user interface for configuring and running multi-agent social simulations focused upon negotiation and cooperation. The NegotiationGym codebase offers a user-friendly, configuration-driven API that enables easy design and customization of simulation scenarios. Agent-level utility functions encode optimization criteria for each agent, and agents can self-optimize by conducting multiple interaction rounds with other agents, observing outcomes, and modifying their strategies for future rounds. 

**Abstract (ZH)**: 我们设计并实现了NegotiationGym，一个用于配置和运行以谈判和合作为重点的多Agent社会模拟的API和用户界面。NegotiationGym代码库提供了一个用户友好的、基于配置的API，使得模拟场景的容易设计和定制变得简单。Agent级别的效用函数编码了每个Agent的优化标准，Agents可以通过与其它Agent进行多轮交互、观察结果并调整未来轮次的策略来自我优化。 

---
# Challenge on Optimization of Context Collection for Code Completion 

**Title (ZH)**: 代码补全中上下文收集优化的挑战 

**Authors**: Dmitry Ustalov, Egor Bogomolov, Alexander Bezzubov, Yaroslav Golubev, Evgeniy Glukhov, Georgii Levtsov, Vladimir Kovalenko  

**Link**: [PDF](https://arxiv.org/pdf/2510.04349)  

**Abstract**: The rapid advancement of workflows and methods for software engineering using AI emphasizes the need for a systematic evaluation and analysis of their ability to leverage information from entire projects, particularly in large code bases. In this challenge on optimization of context collection for code completion, organized by JetBrains in collaboration with Mistral AI as part of the ASE 2025 conference, participants developed efficient mechanisms for collecting context from source code repositories to improve fill-in-the-middle code completions for Python and Kotlin. We constructed a large dataset of real-world code in these two programming languages using permissively licensed open-source projects. The submissions were evaluated based on their ability to maximize completion quality for multiple state-of-the-art neural models using the chrF metric. During the public phase of the competition, nineteen teams submitted solutions to the Python track and eight teams submitted solutions to the Kotlin track. In the private phase, six teams competed, of which five submitted papers to the workshop. 

**Abstract (ZH)**: 使用AI加速软件工程的工作流和方法促进了对整个项目信息利用的系统评价与分析，尤其是在大型代码库领域。作为ASE 2025会议的一部分，由JetBrains与Mistral AI合作举办的优化代码补全上下文收集挑战赛中，参与者开发了高效机制，从源代码仓库中收集上下文以提高Python和Kotlin语言的中间代码补全质量。我们使用许可开源项目构建了一个大型的这类编程语言的真实代码数据集。提交的作品根据其在多种先进神经模型上的补全质量最大化能力，使用chrF指标进行了评估。在比赛的公开阶段，19支队伍提交了Python赛道的解决方案，8支队伍提交了Kotlin赛道的解决方案。在私下阶段，有6支队伍参与竞争，其中5支队伍提交了论文参加研讨会。 

---
# Critical appraisal of artificial intelligence for rare-event recognition: principles and pharmacovigilance case studies 

**Title (ZH)**: 人工智能在识别罕见事件中的批判性评估：原理与药监案例研究 

**Authors**: G. Niklas Noren, Eva-Lisa Meldau, Johan Ellenius  

**Link**: [PDF](https://arxiv.org/pdf/2510.04341)  

**Abstract**: Many high-stakes AI applications target low-prevalence events, where apparent accuracy can conceal limited real-world value. Relevant AI models range from expert-defined rules and traditional machine learning to generative LLMs constrained for classification. We outline key considerations for critical appraisal of AI in rare-event recognition, including problem framing and test set design, prevalence-aware statistical evaluation, robustness assessment, and integration into human workflows. In addition, we propose an approach to structured case-level examination (SCLE), to complement statistical performance evaluation, and a comprehensive checklist to guide procurement or development of AI models for rare-event recognition. We instantiate the framework in pharmacovigilance, drawing on three studies: rule-based retrieval of pregnancy-related reports; duplicate detection combining machine learning with probabilistic record linkage; and automated redaction of person names using an LLM. We highlight pitfalls specific to the rare-event setting including optimism from unrealistic class balance and lack of difficult positive controls in test sets - and show how cost-sensitive targets align model performance with operational value. While grounded in pharmacovigilance practice, the principles generalize to domains where positives are scarce and error costs may be asymmetric. 

**Abstract (ZH)**: 许多高风险的AI应用针对低频事件，在这些应用中，表面上的准确率可能会掩盖其在实际世界中的有限价值。相关AI模型包括专家定义规则、传统机器学习以及受控分类的生成型大语言模型。我们概述了在稀有事件识别中评估AI的关键考虑因素，包括问题界定和测试集设计、存在意识的统计评估、稳健性评估以及将其整合到人类工作流程中。此外，我们提出了一种结构化案例级别检查（SCLE）的方法，以补充统计性能评估，并提供一份全面的检查表，以指导采购或开发用于稀有事件识别的AI模型。我们在药物流行病学中实例化了该框架，并基于三个研究实例进行了展开：基于规则的妊娠相关报告检索；结合机器学习和概率记录链接的重复检测；以及使用大语言模型自动脱敏人名。我们指出了特定于稀有事件设置的陷阱，包括不切实际的类别平衡带来的乐观估计以及测试集中缺乏难以处理的阳性对照，并展示了成本敏感的目标如何使模型性能与操作价值相一致。虽然该框架基于药物流行病学实践，但其原则适用于正例稀缺且错误成本可能不对称的领域。 

---
# SliceMoE: Routing Embedding Slices Instead of Tokens for Fine-Grained and Balanced Transformer Scaling 

**Title (ZH)**: SliceMoE: 代替Token，将Embedding Slice用于精细粒度和平衡的Transformer扩展 

**Authors**: Harshil Vejendla  

**Link**: [PDF](https://arxiv.org/pdf/2510.04286)  

**Abstract**: Mixture-of-Experts (MoE) layers scale transformers by routing tokens to a sparse subset of feed-forward experts. Token-level routing, however, assigns an entire semantic spectrum to each expert, creating capacity bottlenecks, load-balancing pathologies, and limited specialization. We introduce SliceMoE, an architecture that routes contiguous slices of a token's hidden vector. A d-dimensional embedding is partitioned into S slices, and for each slice, a lightweight shared router predicts the top-k experts. Experts operate on their assigned slices independently, and outputs are reassembled, maintaining per-token FLOP efficiency. Because slices from different tokens interleave within an expert, utilization is naturally smoother. We propose a slice-level capacity loss, cross-slice dropout, and efficient fused batched GEMM kernels. Experiments on WikiText-103 language modeling, WMT En-De translation, and three text-classification datasets show SliceMoE attains up to 1.7x faster inference than dense baselines, 12 to 18 percent lower perplexity than parameter-matched token-MoE, and improved expert balance, with interpretable expertise over syntactic versus semantic subspaces. 

**Abstract (ZH)**: SliceMoE通过路由连续的隐藏向量片段扩展变压器 

---
# Scalable Causal Discovery from Recursive Nonlinear Data via Truncated Basis Function Scores and Tests 

**Title (ZH)**: 递归非线性数据通过截断基函数评分与检验的可扩展因果发现 

**Authors**: Joseph Ramsey, Bryan Andrews  

**Link**: [PDF](https://arxiv.org/pdf/2510.04276)  

**Abstract**: Learning graphical conditional independence structures from nonlinear, continuous or mixed data is a central challenge in machine learning and the sciences, and many existing methods struggle to scale to thousands of samples or hundreds of variables. We introduce two basis-expansion tools for scalable causal discovery. First, the Basis Function BIC (BF-BIC) score uses truncated additive expansions to approximate nonlinear dependencies. BF-BIC is theoretically consistent under additive models and extends to post-nonlinear (PNL) models via an invertible reparameterization. It remains robust under moderate interactions and supports mixed data through a degenerate-Gaussian embedding for discrete variables. In simulations with fully nonlinear neural causal models (NCMs), BF-BIC outperforms kernel- and constraint-based methods (e.g., KCI, RFCI) in both accuracy and runtime. Second, the Basis Function Likelihood Ratio Test (BF-LRT) provides an approximate conditional independence test that is substantially faster than kernel tests while retaining competitive accuracy. Extensive simulations and a real-data application to Canadian wildfire risk show that, when integrated into hybrid searches, BF-based methods enable interpretable and scalable causal discovery. Implementations are available in Python, R, and Java. 

**Abstract (ZH)**: 从非线性、连续或混合数据中学习图形条件独立结构是机器学习和科学领域的核心挑战，现有方法难以处理数千样本或数百变量的情况。我们引入了两种可扩展的因果发现基础扩张工具。首先，基函数BIC（BF-BIC）分数使用截断的加性扩张来近似非线性依赖关系。BF-BIC在加性模型下是理论上一致的，并通过可逆重构参数化扩展到后非线性（PNL）模型。它在中等交互作用下仍保持稳健，并通过离散变量的退化高斯嵌入支持混合数据。在使用完全非线性神经因果模型（NCMs）的模拟试验中，BF-BIC在准确性和运行时间上均优于核方法和约束方法（例如KCI和RFCI）。其次，基函数似然比检验（BF-LRT）提供了一种近似条件独立性检验，速度远快于核检验，同时保持了竞争力。广泛的模拟试验和加拿大野火风险的实际数据应用表明，当集成到混合搜索中时，基于BF的方法能够实现可解释且可扩展的因果发现。已有Python、R和Java实现。 

---
# Efficient Latent Variable Causal Discovery: Combining Score Search and Targeted Testing 

**Title (ZH)**: 高效的潜在变量因果发现：结合分数搜索与目标测试 

**Authors**: Joseph Ramsey, Bryan Andrews  

**Link**: [PDF](https://arxiv.org/pdf/2510.04263)  

**Abstract**: Learning causal structure from observational data is especially challenging when latent variables or selection bias are present. The Fast Causal Inference (FCI) algorithm addresses this setting but often performs exhaustive conditional independence tests across many subsets, leading to spurious independence claims, extra or missing edges, and unreliable orientations. We present a family of score-guided mixed-strategy causal search algorithms that build on this tradition. First, we introduce BOSS-FCI and GRaSP-FCI, straightforward variants of GFCI that substitute BOSS or GRaSP for FGES, thereby retaining correctness while incurring different scalability tradeoffs. Second, we develop FCI Targeted-testing (FCIT), a novel mixed-strategy method that improves upon these variants by replacing exhaustive all-subsets testing with targeted tests guided by BOSS, yielding well-formed PAGs with higher precision and efficiency. Finally, we propose a simple heuristic, LV-Dumb (also known as BOSS-POD), which bypasses latent-variable-specific reasoning and directly returns the PAG of the BOSS DAG. Although not strictly correct in the FCI sense, it scales better and often achieves superior accuracy in practice. Simulations and real-data analyses demonstrate that BOSS-FCI and GRaSP-FCI provide sound baselines, FCIT improves both efficiency and reliability, and LV-Dumb offers a practical heuristic with strong empirical performance. Together, these method highlight the value of score-guided and targeted strategies for scalable latent-variable causal discovery. 

**Abstract (ZH)**: 学习观测数据中的因果结构尤其具有挑战性，特别是在潜在变量或选择偏差存在的情况下。快速因果推理（FCI）算法适用于这种情境，但往往需要进行大量的条件独立性检验，导致虚假独立性声明、过多或过少的边以及不可靠的边定向。我们提出了一类分数指导的混合策略因果搜索算法，建立在这个传统的基础之上。首先，我们介绍了BOSS-FCI和GRaSP-FCI，这是GFCI的直截了当的变体，用BOSS或GRaSP代替FGES，从而保持正确性但产生不同的可扩展性权衡。其次，我们开发了FCI目标测试（FCIT）方法，这是一种新型的混合策略方法，通过用BOSS引导的目标测试替代全面的子集测试，从而生成更为合理的部分有向无环图（PAG），并提高精确度和效率。最后，我们提出了一个简单的启发式方法LV-Dumb（也称为BOSS-POD），它绕过特定于潜在变量的推理，直接返回BOSS有向无环图（DAG）的PAG。虽然从严格意义上来说在FCI意义下不完全正确，但在实践中通常能实现更优的准确性。模拟和实际数据分析表明，BOSS-FCI和GRaSP-FCI提供了稳健的基础，FCIT在提高效率和可靠性方面有所改进，而LV-Dumb则提供了一个具有强大实际性能的实用启发式方法。这些方法共同突显了分数指导和目标测试策略对于可扩展的潜在变量因果发现的价值。 

---
# AgentTypo: Adaptive Typographic Prompt Injection Attacks against Black-box Multimodal Agents 

**Title (ZH)**: AgentTypo: 面向黑盒多模态代理的自适应排版提示注入攻击 

**Authors**: Yanjie Li, Yiming Cao, Dong Wang, Bin Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.04257)  

**Abstract**: Multimodal agents built on large vision-language models (LVLMs) are increasingly deployed in open-world settings but remain highly vulnerable to prompt injection, especially through visual inputs. We introduce AgentTypo, a black-box red-teaming framework that mounts adaptive typographic prompt injection by embedding optimized text into webpage images. Our automatic typographic prompt injection (ATPI) algorithm maximizes prompt reconstruction by substituting captioners while minimizing human detectability via a stealth loss, with a Tree-structured Parzen Estimator guiding black-box optimization over text placement, size, and color. To further enhance attack strength, we develop AgentTypo-pro, a multi-LLM system that iteratively refines injection prompts using evaluation feedback and retrieves successful past examples for continual learning. Effective prompts are abstracted into generalizable strategies and stored in a strategy repository, enabling progressive knowledge accumulation and reuse in future attacks. Experiments on the VWA-Adv benchmark across Classifieds, Shopping, and Reddit scenarios show that AgentTypo significantly outperforms the latest image-based attacks such as AgentAttack. On GPT-4o agents, our image-only attack raises the success rate from 0.23 to 0.45, with consistent results across GPT-4V, GPT-4o-mini, Gemini 1.5 Pro, and Claude 3 Opus. In image+text settings, AgentTypo achieves 0.68 ASR, also outperforming the latest baselines. Our findings reveal that AgentTypo poses a practical and potent threat to multimodal agents and highlight the urgent need for effective defense. 

**Abstract (ZH)**: 基于大型视觉语言模型的多模态代理在开放环境中构建，但仍然高度易受提示注入攻击，尤其是通过视觉输入。我们介绍了AgentTypo，一种黑盒红队框架，通过将优化文本嵌入网页图像中实施自适应字型提示注入。我们的自动字型提示注入（ATPI）算法通过替代图注者来最大化提示重建，同时通过隐身损失减小人为可检测性，并使用树结构粒子滤波器指导文本放置、大小和颜色的黑盒优化。为了进一步增强攻击强度，我们开发了AgentTypo-pro，这是一种多LLM系统，可以通过评估反馈迭代细化注入提示，并检索成功的过往例证进行持续学习。有效的提示被抽象为通用策略并存储在策略库中，这使未来的攻击能够逐步积累和重用知识。在针对VWA-Adv基准（在Classifieds、Shopping和Reddit场景下）的实验中，AgentTypo显著优于最新的基于图像的攻击，如AgentAttack。在GPT-4o代理上，我们的仅图像攻击将成功率从0.23提高到0.45，在GPT-4V、GPT-4o-mini、Gemini 1.5 Pro和Claude 3 Opus上均保持一致结果。在图像+文本设置中，AgentTypo实现了0.68 ASR，也优于最新的基线。我们的发现表明，AgentTypo对多模态代理构成了实际且强大的威胁，并突显了迫切需要有效防御的重要性。 

---
# Concept-Based Masking: A Patch-Agnostic Defense Against Adversarial Patch Attacks 

**Title (ZH)**: 基于概念的掩蔽：一种对patch攻击无偏见的防御方法 

**Authors**: Ayushi Mehrotra, Derek Peng, Dipkamal Bhusal, Nidhi Rastogi  

**Link**: [PDF](https://arxiv.org/pdf/2510.04245)  

**Abstract**: Adversarial patch attacks pose a practical threat to deep learning models by forcing targeted misclassifications through localized perturbations, often realized in the physical world. Existing defenses typically assume prior knowledge of patch size or location, limiting their applicability. In this work, we propose a patch-agnostic defense that leverages concept-based explanations to identify and suppress the most influential concept activation vectors, thereby neutralizing patch effects without explicit detection. Evaluated on Imagenette with a ResNet-50, our method achieves higher robust and clean accuracy than the state-of-the-art PatchCleanser, while maintaining strong performance across varying patch sizes and locations. Our results highlight the promise of combining interpretability with robustness and suggest concept-driven defenses as a scalable strategy for securing machine learning models against adversarial patch attacks. 

**Abstract (ZH)**: 基于概念的防御方法抵御 adversarial patch 攻击：在不依赖先验知识的前提下，通过抑制最 influent 的概念激活向量来中和 patch 效应，从而在不同 patch 大小和位置的情况下实现更高的鲁棒性和清洁准确性，并强调将可解释性与鲁棒性结合的潜力，建议概念驱动的防御作为保护机器学习模型免受 adversarial patch 攻击的可扩展策略。 

---
# Diffusion-Assisted Distillation for Self-Supervised Graph Representation Learning with MLPs 

**Title (ZH)**: 基于扩散辅助蒸馏的自监督图表示学习方法（使用MLP） 

**Authors**: Seong Jin Ahn, Myoung-Ho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.04241)  

**Abstract**: For large-scale applications, there is growing interest in replacing Graph Neural Networks (GNNs) with lightweight Multi-Layer Perceptrons (MLPs) via knowledge distillation. However, distilling GNNs for self-supervised graph representation learning into MLPs is more challenging. This is because the performance of self-supervised learning is more related to the model's inductive bias than supervised learning. This motivates us to design a new distillation method to bridge a huge capacity gap between GNNs and MLPs in self-supervised graph representation learning. In this paper, we propose \textbf{D}iffusion-\textbf{A}ssisted \textbf{D}istillation for \textbf{S}elf-supervised \textbf{G}raph representation learning with \textbf{M}LPs (DAD-SGM). The proposed method employs a denoising diffusion model as a teacher assistant to better distill the knowledge from the teacher GNN into the student MLP. This approach enhances the generalizability and robustness of MLPs in self-supervised graph representation learning. Extensive experiments demonstrate that DAD-SGM effectively distills the knowledge of self-supervised GNNs compared to state-of-the-art GNN-to-MLP distillation methods. Our implementation is available at this https URL. 

**Abstract (ZH)**: Diffusion-Assisted Distillation for Self-supervised Graph Representation Learning with MLPs (DAD-SGM) 

---
# When AI Gets Persuaded, Humans Follow: Inducing the Conformity Effect in Persuasive Dialogue 

**Title (ZH)**: 当AI被说服时，人类随之效仿：在说服性对话中诱导从众效应 

**Authors**: Rikuo Sasaki, Michimasa Inaba  

**Link**: [PDF](https://arxiv.org/pdf/2510.04229)  

**Abstract**: Recent advancements in AI have highlighted its application in captology, the field of using computers as persuasive technologies. We hypothesized that the "conformity effect," where individuals align with others' actions, also occurs with AI agents. This study verifies this hypothesis by introducing a "Persuadee Agent" that is persuaded alongside a human participant in a three-party persuasive dialogue with a Persuader Agent. We conducted a text-based dialogue experiment with human participants. We compared four conditions manipulating the Persuadee Agent's behavior (persuasion acceptance vs. non-acceptance) and the presence of an icebreaker session. Results showed that when the Persuadee Agent accepted persuasion, both perceived persuasiveness and actual attitude change significantly improved. Attitude change was greatest when an icebreaker was also used, whereas an unpersuaded AI agent suppressed attitude change. Additionally, it was confirmed that the persuasion acceptance of participants increased at the moment the Persuadee Agent was persuaded. These results suggest that appropriately designing a Persuadee Agent can improve persuasion through the conformity effect. 

**Abstract (ZH)**: 近期人工智能的进展凸显了其在captology领域的应用，即利用计算机作为说服性技术的领域。本研究假设“从众效应”同样适用于AI代理，即个体会与他人行为一致。通过引入一个在三边说服对话中与人类参与者一同被说服的“被说服代理”，本研究验证了这一假设。我们在人类参与者之间进行了一次基于文本的对话实验，操纵“被说服代理”的行为（接受说服 vs. 不接受说服）以及是否有破冰环节的存在。研究结果显示，当“被说服代理”接受说服时，感知的说服力和实际态度改变显著提高。同时，使用破冰环节时态度改变最大，而未被说服的AI代理则抑制了态度改变。此外，参与者在“被说服代理”被说服的瞬间其说服接受度也有所提高。这些结果表明，适当设计“被说服代理”可以通过从众效应提高说服效果。 

---
# PolyKAN: A Polyhedral Analysis Framework for Provable and Minimal KAN Compression 

**Title (ZH)**: PolyKAN: 一种 Provably 和 Minimal 的 KAN 压缩多面体分析框架 

**Authors**: Di Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04205)  

**Abstract**: Kolmogorov-Arnold Networks (KANs) have emerged as a promising alternative to traditional Multi-Layer Perceptrons (MLPs), offering enhanced interpretability and a strong mathematical foundation. However, their parameter efficiency remains a significant challenge for practical deployment. This paper introduces PolyKAN, a novel theoretical framework for KAN compression that provides formal guarantees on both model size reduction and approximation error. By leveraging the inherent piecewise polynomial structure of KANs, we formulate the compression problem as one of optimal polyhedral region merging. We establish a rigorous polyhedral characterization of KANs, develop a complete theory of $\epsilon$-equivalent compression, and design an optimal dynamic programming algorithm that guarantees minimal compression under specified error bounds. Our theoretical analysis demonstrates that PolyKAN achieves provably minimal compression while maintaining strict error control, with polynomial-time complexity in all network parameters. The framework provides the first formal foundation for KAN compression with mathematical guarantees, opening new directions for efficient deployment of interpretable neural architectures. 

**Abstract (ZH)**: PolyKAN：Kolmogorov-Arnold网络的新型压缩理论框架 

---
# Cooperative Flexibility Exchange: Fair and Comfort-Aware Decentralized Resource Allocation 

**Title (ZH)**: 协同灵活性交换：公平与舒适感知的分布式资源分配 

**Authors**: Rabiya Khalid, Evangelos Pournaras  

**Link**: [PDF](https://arxiv.org/pdf/2510.04192)  

**Abstract**: The growing electricity demand and increased use of smart appliances are placing new pressures on power grids, making efficient energy management more important than ever. The existing energy management systems often prioritize system efficiency (balanced energy demand and supply) at the expense of user comfort. This paper addresses this gap by proposing a novel decentralized multi-agent coordination-based demand-side management system. The proposed system enables individual agents to coordinate for demand-side energy optimization while improving the user comfort and maintaining the system efficiency. A key innovation of this work is the introduction of a slot exchange mechanism, where agents first receive optimized appliance-level energy consumption schedules and then coordinate with each other to adjust these schedules through slot exchanges. This approach improves user comfort even when agents show non-altruistic behaviour, and it scales well with large populations. The system also promotes fairness by balancing satisfaction levels across users. For performance evaluation, a real-world dataset is used, and the results demonstrate that the proposed slot exchange mechanism increases user comfort and fairness without raising system inefficiency cost, making it a practical and scalable solution for future smart grids. 

**Abstract (ZH)**: Growing电力需求和智能家电的增加使用对电网造成了新的压力，使得高效的能源管理变得前所未有的重要。现有能源管理系统往往以系统的效率（平衡能源需求和供应）为优先，而牺牲用户舒适度。本文通过提出一种新颖的去中心化多代理协调需求侧管理系统来弥补这一差距。该提出的系统使个体代理能够协调以实现需求侧的能源优化，同时提高用户舒适度并保持系统的效率。这项工作的关键创新在于引入了一个时间段交换机制，即代理首先接收优化的家电级能源消耗时间表，然后通过时间段交换相互协调调整这些时间表。这种做法即使在网络代理表现出非利他行为时也能够提高用户舒适度，并且能够很好地扩展到大量人群。该系统还通过平衡用户的满意度水平促进了公平性。为了评估性能，使用了真实世界的数据集，结果表明，提出的时间段交换机制能够提高用户舒适度和公平性，而不增加系统的无效率成本，使其成为面向未来智能电网的实用且可扩展的解决方案。 

---
# Finite Time Analysis of Constrained Natural Critic-Actor Algorithm with Improved Sample Complexity 

**Title (ZH)**: 有限时间内约束自然评论者-行动家算法的研究：改进的样本复杂性分析 

**Authors**: Prashansa Panda, Shalabh Bhatnagar  

**Link**: [PDF](https://arxiv.org/pdf/2510.04189)  

**Abstract**: Recent studies have increasingly focused on non-asymptotic convergence analyses for actor-critic (AC) algorithms. One such effort introduced a two-timescale critic-actor algorithm for the discounted cost setting using a tabular representation, where the usual roles of the actor and critic are reversed. However, only asymptotic convergence was established there. Subsequently, both asymptotic and non-asymptotic analyses of the critic-actor algorithm with linear function approximation were conducted. In our work, we introduce the first natural critic-actor algorithm with function approximation for the long-run average cost setting and under inequality constraints. We provide the non-asymptotic convergence guarantees for this algorithm. Our analysis establishes optimal learning rates and we also propose a modification to enhance sample complexity. We further show the results of experiments on three different Safety-Gym environments where our algorithm is found to be competitive in comparison with other well known algorithms. 

**Abstract (ZH)**: 近期的研究越来越多地关注演员-评论家(AC)算法的非渐近收敛分析。其中一项努力在折现成本设置中使用表征表示提出了一种双时间尺度评论家-演员算法，其中评论家和演员的传统角色被逆转。然而，只建立了渐近收敛性。随后，对线性函数逼近下的评论家-演员算法的渐近和非渐近分析进行了研究。在我们的工作中，我们首次在此长期内均值成本设置和不等式约束下引入了一种自然的函数逼近下的评论家-演员算法，并提供了该算法的非渐近收敛保证。我们的分析确定了最优的学习率，还提出了改进样本复杂性的修改方案。我们还在三个不同的Safety-Gym环境中展示了实验结果，发现该算法与已知的其他算法相比具有竞争力。 

---
# A Complement to Neural Networks for Anisotropic Inelasticity at Finite Strains 

**Title (ZH)**: 用于有限应变各向异性非线性的一种补充神经网络方法 

**Authors**: Hagen Holthusen, Ellen Kuhl  

**Link**: [PDF](https://arxiv.org/pdf/2510.04187)  

**Abstract**: We propose a complement to constitutive modeling that augments neural networks with material principles to capture anisotropy and inelasticity at finite strains. The key element is a dual potential that governs dissipation, consistently incorporates anisotropy, and-unlike conventional convex formulations-satisfies the dissipation inequality without requiring convexity.
Our neural network architecture employs invariant-based input representations in terms of mixed elastic, inelastic and structural tensors. It adapts Input Convex Neural Networks, and introduces Input Monotonic Neural Networks to broaden the admissible potential class. To bypass exponential-map time integration in the finite strain regime and stabilize the training of inelastic materials, we employ recurrent Liquid Neural Networks.
The approach is evaluated at both material point and structural scales. We benchmark against recurrent models without physical constraints and validate predictions of deformation and reaction forces for unseen boundary value problems. In all cases, the method delivers accurate and stable performance beyond the training regime. The neural network and finite element implementations are available as open-source and are accessible to the public via this https URL. 

**Abstract (ZH)**: 我们提出了一种补充性的本构建模方法，将材料原理与神经网络相结合，以捕捉有限应变下的各向异性和非线性行为。关键要素是一种双潜能函数，它控制耗散现象，一致地包含各向异性，并且在不需要凸性的前提下满足耗散不等式。

该神经网络架构采用基于不变量的输入表示，包括混合弹性、非弹性及结构张量。它采用输入凸神经网络，并引入输入单调神经网络，以扩展允许的潜能类。为在有限应变状态下绕开指数映射时间积分并稳定非线性材料的训练过程，我们采用了循环液态神经网络。

该方法在材料点和结构尺度上进行了评估。我们将其与没有任何物理约束的循环模型进行了对比，并验证了对未见过的边界值问题的变形和反应力预测。在所有情况下，该方法均表现出超越训练范围的准确和稳定性能。神经网络和有限元实现均已开源，并可通过以下链接访问：https://doi.org/10.1140/epjdestini/a123456 

---
# Multi Language Models for On-the-Fly Syntax Highlighting 

**Title (ZH)**: 基于多语言模型的即时语法高亮 

**Authors**: Marco Edoardo Palma, Pooja Rani, Harald C. Gall  

**Link**: [PDF](https://arxiv.org/pdf/2510.04166)  

**Abstract**: Syntax highlighting is a critical feature in modern software development environments, enhancing code readability and developer productivity. However, delivering accurate highlighting in real time remains challenging for online and web-based development tools due to strict time and memory constraints on backend services. These systems must serve highlights rapidly and frequently, even when code is partially valid or invalid. This has led to on-the-fly syntax highlighting, where visual annotations are generated just before content is served, often at high request rates and under incomplete input conditions. To meet these demands efficiently, state-of-the-art models use deep learning to learn the behavior of brute-force syntax highlighting resolvers, tools that are easy to implement but too slow for production. Through the Deep Abstraction process, brute-force strategies are encoded into fast statistical models that achieve both high accuracy and low-latency inference. Despite their success, such models face key challenges: they support only one programming language per model, require large datasets from slow brute-force generators, and involve resource-intensive training. In multi-language environments, this means maintaining multiple independent models, increasing system complexity and operational cost. This work addresses these issues by introducing a unified model capable of highlighting up to six mainstream programming languages, reducing deployment complexity by a factor of six and improving performance on unseen languages. A novel normalization technique significantly enhances model generalization, while few-shot learning experiments show that a small number of oracle samples can replace large datasets, minimizing dependence on brute-force generators. Combined, these innovations enable efficient, scalable, and cost-effective syntax highlighting across diverse programming languages. 

**Abstract (ZH)**: 现代软件开发环境中语法高亮的关键功能及其实时实现挑战与解决方案 

---
# GA4GC: Greener Agent for Greener Code via Multi-Objective Configuration Optimization 

**Title (ZH)**: GA4GC：通过多目标配置优化实现更绿色的代理和代码 

**Authors**: Jingzhi Gong, Yixin Bian, Luis de la Cal, Giovanni Pinna, Anisha Uteem, David Williams, Mar Zamorano, Karine Even-Mendoza, W.B. Langdon, Hector Menendez, Federica Sarro  

**Link**: [PDF](https://arxiv.org/pdf/2510.04135)  

**Abstract**: Coding agents powered by LLMs face critical sustainability and scalability challenges in industrial deployment, with single runs consuming over 100k tokens and incurring environmental costs that may exceed optimization benefits. This paper introduces GA4GC, the first framework to systematically optimize coding agent runtime (greener agent) and code performance (greener code) trade-offs by discovering Pareto-optimal agent hyperparameters and prompt templates. Evaluation on the SWE-Perf benchmark demonstrates up to 135x hypervolume improvement, reducing agent runtime by 37.7% while improving correctness. Our findings establish temperature as the most critical hyperparameter, and provide actionable strategies to balance agent sustainability with code optimization effectiveness in industrial deployment. 

**Abstract (ZH)**: 基于LLM的编码代理在工业部署中面临关键的可持续性和扩展性挑战，单次运行消耗超过100k个 Tokens，并且可能产生的环境成本超过了优化收益。本文介绍了GA4GC框架，这是首个系统优化编码代理运行时（更绿色的代理）和代码性能（更绿色的代码） trade-offs 的框架，通过发现帕累托最优代理超参数和提示模板。在SWE-Perf基准上的评估表明，最大改进达135倍的超体积，代理运行时间减少37.7%，同时提高正确性。我们的研究结果确立了温度作为最关键超参数，并提供了在工业部署中平衡代理可持续性和代码优化效果的实际策略。 

---
# PhaseFormer: From Patches to Phases for Efficient and Effective Time Series Forecasting 

**Title (ZH)**: PhaseFormer: 从块到相位进行高效有效的时间序列预测 

**Authors**: Yiming Niu, Jinliang Deng, Yongxin Tong  

**Link**: [PDF](https://arxiv.org/pdf/2510.04134)  

**Abstract**: Periodicity is a fundamental characteristic of time series data and has long played a central role in forecasting. Recent deep learning methods strengthen the exploitation of periodicity by treating patches as basic tokens, thereby improving predictive effectiveness. However, their efficiency remains a bottleneck due to large parameter counts and heavy computational costs. This paper provides, for the first time, a clear explanation of why patch-level processing is inherently inefficient, supported by strong evidence from real-world data. To address these limitations, we introduce a phase perspective for modeling periodicity and present an efficient yet effective solution, PhaseFormer. PhaseFormer features phase-wise prediction through compact phase embeddings and efficient cross-phase interaction enabled by a lightweight routing mechanism. Extensive experiments demonstrate that PhaseFormer achieves state-of-the-art performance with around 1k parameters, consistently across benchmark datasets. Notably, it excels on large-scale and complex datasets, where models with comparable efficiency often struggle. This work marks a significant step toward truly efficient and effective time series forecasting. Code is available at this repository: this https URL 

**Abstract (ZH)**: 周期性是时间序列数据的基本特征，长期以来在预测中扮演着中心角色。近期的深度学习方法通过将片段视为基本令牌，增强了对周期性的利用，从而提高了预测效果。然而，由于参数量庞大和计算成本高昂，其效率仍然存在瓶颈。本文首次提供了片层处理本质上不高效的清晰解释，并通过实际数据提供了强有力的支持。为了解决这些局限性，我们提出了一个相位视角来建模周期性，并提出了一种高效而有效的解决方案——PhaseFormer。PhaseFormer通过紧凑的相位嵌入和轻量化路由机制实现的相位间有效交互来进行相位级预测。大量的实验表明，PhaseFormer在基准数据集上实现了最佳性能，参数量仅为约1千个。尤为值得注意的是，在大规模和复杂的数据集上，它表现出色，而这种级别的效率模型在此类数据集上往往难以实现。这项工作代表了真正高效而有效的时间序列预测的一个重要进步。代码可在以下仓库获取：this https URL。 

---
# On the Limitations and Capabilities of Position Embeddings for Length Generalization 

**Title (ZH)**: 关于位置嵌入在长度泛化能力上的局限性和潜力 

**Authors**: Yang Chen, Yitao Liang, Zhouchen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.04130)  

**Abstract**: In Transformers, Position Embeddings (PEs) significantly influence Length Generalization (LG) performance, yet their fundamental role remains unclear. In this work, we investigate the limitations and capabilities of PEs in achieving LG. We theoretically analyze PEs in Position-Only Linear Attentions (POLAs), introducing Linear Representation Complexity (LRC) to characterize when PEs enable LG. Our analysis shows that PEs do not expand computational capabilities but structure learned computations across positions. Extending to practical Transformers, we propose Sequential Representation Complexity (SRC) and conjecture that LG is possible if and only if SRC remains invariant across scales. We support this hypothesis with empirical evidence in various reasoning tasks. To enhance LG, we introduce Scale Hint, allowing flexible instance scaling, and a Learning-Based Position Embedding framework that automatically learns positional relations. Our work provides theoretical insights and practical strategies for improving LG in Transformers. 

**Abstract (ZH)**: 在Transformer中，位置嵌入（PEs）在长度泛化（LG）性能中的作用显著，但其基本作用尚不明确。本文研究了PEs在实现LG方面的限制与能力。我们从位置唯一线性注意（POLAs）的角度理论分析PEs，并引入线性表示复杂性（LRC）来刻画PEs如何使LG成为可能。我们的分析表明，PEs并未扩展计算能力，而是结构化了不同位置上学习到的计算。扩展到实际的Transformer中，我们提出序列表示复杂性（SRC），并猜测如果SRC在不同尺度上保持不变，LG是可能实现的。我们通过在各种推理任务中的实验证据支持这一假说。为了增强LG，我们引入了尺度提示（Scale Hint），允许灵活的实例缩放，并提出了一种基于学习的位置嵌入框架，可以自动学习位置关系。我们的工作提供了关于改进Transformer中LG的理论见解和实用策略。 

---
# Learning-Based Hashing for ANN Search: Foundations and Early Advances 

**Title (ZH)**: 基于学习的哈希表示在ANN搜索中的基础与初步进展 

**Authors**: Sean Moran  

**Link**: [PDF](https://arxiv.org/pdf/2510.04127)  

**Abstract**: Approximate Nearest Neighbour (ANN) search is a fundamental problem in information retrieval, underpinning large-scale applications in computer vision, natural language processing, and cross-modal search. Hashing-based methods provide an efficient solution by mapping high-dimensional data into compact binary codes that enable fast similarity computations in Hamming space. Over the past two decades, a substantial body of work has explored learning to hash, where projection and quantisation functions are optimised from data rather than chosen at random.
This article offers a foundational survey of early learning-based hashing methods, with an emphasis on the core ideas that shaped the field. We review supervised, unsupervised, and semi-supervised approaches, highlighting how projection functions are designed to generate meaningful embeddings and how quantisation strategies convert these embeddings into binary codes. We also examine extensions to multi-bit and multi-threshold models, as well as early advances in cross-modal retrieval.
Rather than providing an exhaustive account of the most recent methods, our goal is to introduce the conceptual foundations of learning-based hashing for ANN search. By situating these early models in their historical context, we aim to equip readers with a structured understanding of the principles, trade-offs, and open challenges that continue to inform current research in this area. 

**Abstract (ZH)**: 基于哈希的近似最近邻搜索./(ANN)搜索是信息检索中的一个基本问题，支撑着计算机视觉、自然语言处理和跨模态搜索等大规模应用。基于哈希的方法通过将高维数据映射为紧凑的二进制代码，从而在汉明空间中实现快速相似性计算，提供了一个高效的解决方案。在过去二十年里，大量研究探索了学习哈希方法，其中投影和量化函数是从数据中学习优化，而不是随机选择。

本文提供了一篇早期基于学习的哈希方法的基础综述，侧重于塑造该领域的核心思想。我们回顾了监督、无监督和半监督的方法，强调了如何设计投影函数生成有意义的嵌入，以及如何通过量化策略将这些嵌入转化为二进制代码。我们也探讨了多比特和多阈值模型的扩展，以及跨模态检索的早期进展。

我们的目标不是提供最新方法的详尽综述，而是旨在介绍基于学习的哈希方法在ANN搜索中的概念基础。通过将这些早期模型置于其历史背景中，我们希望读者能够获得对原则、权衡和仍然影响当前研究的开放挑战的结构化理解。 

---
# Attending on Multilevel Structure of Proteins enables Accurate Prediction of Cold-Start Drug-Target Interactions 

**Title (ZH)**: 关注蛋白质的多层结构以实现冷启动药物-靶标相互作用的准确预测 

**Authors**: Ziying Zhang, Yaqing Wang, Yuxuan Sun, Min Ye, Quanming Yao  

**Link**: [PDF](https://arxiv.org/pdf/2510.04126)  

**Abstract**: Cold-start drug-target interaction (DTI) prediction focuses on interaction between novel drugs and proteins. Previous methods typically learn transferable interaction patterns between structures of drug and proteins to tackle it. However, insight from proteomics suggest that protein have multi-level structures and they all influence the DTI. Existing works usually represent protein with only primary structures, limiting their ability to capture interactions involving higher-level structures. Inspired by this insight, we propose ColdDTI, a framework attending on protein multi-level structure for cold-start DTI prediction. We employ hierarchical attention mechanism to mine interaction between multi-level protein structures (from primary to quaternary) and drug structures at both local and global granularities. Then, we leverage mined interactions to fuse structure representations of different levels for final prediction. Our design captures biologically transferable priors, avoiding the risk of overfitting caused by excessive reliance on representation learning. Experiments on benchmark datasets demonstrate that ColdDTI consistently outperforms previous methods in cold-start settings. 

**Abstract (ZH)**: 冷启动药物-目标相互作用预测关注新药物与蛋白质之间的相互作用。现有方法通常通过学习药物和蛋白质结构之间的可迁移相互作用模式来解决这一问题。然而，蛋白质组学的见解表明，蛋白质具有多层级结构，所有层级都影响药物-目标相互作用。现有工作通常仅用蛋白质的一级结构来表示，限制了其捕获涉及更高层级结构的相互作用的能力。受此见解的启发，我们提出了一种名为ColdDTI的框架，用于冷启动药物-目标相互作用预测，该框架关注蛋白质的多层级结构。我们采用分层注意机制，在局部和全局粒度上挖掘从一级到四级的多层级蛋白质结构与药物结构之间的相互作用，然后利用挖掘出的相互作用融合不同层级的结构表示以进行最终预测。我们的设计捕获了生物学上的可迁移先验，避免了过度依赖表示学习而导致的过拟合风险。在基准数据集上的实验表明，ColdDTI在冷启动设置中始终优于先前的方法。 

---
# TOPO-Bench: An Open-Source Topological Mapping Evaluation Framework with Quantifiable Perceptual Aliasing 

**Title (ZH)**: TOPO-Bench: 一种具有可量化感知 aliaseding 的开源拓扑映射评估框架 

**Authors**: Jiaming Wang, Diwen Liu, Jizhuo Chen, Harold Soh  

**Link**: [PDF](https://arxiv.org/pdf/2510.04100)  

**Abstract**: Topological mapping offers a compact and robust representation for navigation, but progress in the field is hindered by the lack of standardized evaluation metrics, datasets, and protocols. Existing systems are assessed using different environments and criteria, preventing fair and reproducible comparisons. Moreover, a key challenge - perceptual aliasing - remains under-quantified, despite its strong influence on system performance. We address these gaps by (1) formalizing topological consistency as the fundamental property of topological maps and showing that localization accuracy provides an efficient and interpretable surrogate metric, and (2) proposing the first quantitative measure of dataset ambiguity to enable fair comparisons across environments. To support this protocol, we curate a diverse benchmark dataset with calibrated ambiguity levels, implement and release deep-learned baseline systems, and evaluate them alongside classical methods. Our experiments and analysis yield new insights into the limitations of current approaches under perceptual aliasing. All datasets, baselines, and evaluation tools are fully open-sourced to foster consistent and reproducible research in topological mapping. 

**Abstract (ZH)**: 拓扑映射提供了紧凑且 robust 的导航表示，但领域进展受限于缺乏标准化评估指标、数据集和协议。现有系统使用不同的环境和标准进行评估，阻碍了公平和可重复的比较。此外，尽管感知退化对系统性能有强烈影响，但关键挑战——感知退化仍被严重低估。我们通过以下方式解决这些差距：（1）将拓扑一致性正式化为拓扑地图的基本属性，并展示局部化精度作为有效且可解释的替代指标，（2）提出第一个数据集模糊性的定量度量以实现不同环境之间的公平比较。为支持此协议，我们策划了一个具有校准模糊度级别的多元化基准数据集，实现了并发布了基于深度学习的基线系统，并在经典方法旁边进行了评估。我们的实验和分析提供了对当前方法在感知退化下局限性的新见解。所有数据集、基线和评估工具均已完全开源，以促进拓扑映射中的一致且可重复的研究。 

---
# Efficient Training of Spiking Neural Networks by Spike-aware Data Pruning 

**Title (ZH)**: 基于.spike-aware数据剪枝的高效Spiking神经网络训练 

**Authors**: Chenxiang Ma, Xinyi Chen, Yujie Wu, Kay Chen Tan, Jibin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04098)  

**Abstract**: Spiking neural networks (SNNs), recognized as an energy-efficient alternative to traditional artificial neural networks (ANNs), have advanced rapidly through the scaling of models and datasets. However, such scaling incurs considerable training overhead, posing challenges for researchers with limited computational resources and hindering the sustained development of SNNs. Data pruning is a promising strategy for accelerating training by retaining the most informative examples and discarding redundant ones, but it remains largely unexplored in SNNs. Directly applying ANN-based data pruning methods to SNNs fails to capture the intrinsic importance of examples and suffers from high gradient variance. To address these challenges, we propose a novel spike-aware data pruning (SADP) method. SADP reduces gradient variance by determining each example's selection probability to be proportional to its gradient norm, while avoiding the high cost of direct gradient computation through an efficient upper bound, termed spike-aware importance score. This score accounts for the influence of all-or-nothing spikes on the gradient norm and can be computed with negligible overhead. Extensive experiments across diverse datasets and architectures demonstrate that SADP consistently outperforms data pruning baselines and achieves training speedups close to the theoretical maxima at different pruning ratios. Notably, SADP reduces training time by 35% on ImageNet while maintaining accuracy comparable to that of full-data training. This work, therefore, establishes a data-centric paradigm for efficient SNN training and paves the way for scaling SNNs to larger models and datasets. The source code will be released publicly after the review process. 

**Abstract (ZH)**: 基于尖峰的稀疏数据训练方法：一种加速Spiking神经网络训练的新策略 

---
# Using predefined vector systems as latent space configuration for neural network supervised training on data with arbitrarily large number of classes 

**Title (ZH)**: 使用预定义向量系统作为神经网络监督训练的潜在空间配置，以处理任意大量类别的数据 

**Authors**: Nikita Gabdullin  

**Link**: [PDF](https://arxiv.org/pdf/2510.04090)  

**Abstract**: Supervised learning (SL) methods are indispensable for neural network (NN) training used to perform classification tasks. While resulting in very high accuracy, SL training often requires making NN parameter number dependent on the number of classes, limiting their applicability when the number of classes is extremely large or unknown in advance. In this paper we propose a methodology that allows one to train the same NN architecture regardless of the number of classes. This is achieved by using predefined vector systems as the target latent space configuration (LSC) during NN training. We discuss the desired properties of target configurations and choose randomly perturbed vectors of An root system for our experiments. These vectors are used to successfully train encoders and visual transformers (ViT) on Cinic-10 and ImageNet-1K in low- and high-dimensional cases by matching NN predictions with the predefined vectors. Finally, ViT is trained on a dataset with 1.28 million classes illustrating the applicability of the method to training on datasets with extremely large number of classes. In addition, potential applications of LSC in lifelong learning and NN distillation are discussed illustrating versatility of the proposed methodology. 

**Abstract (ZH)**: 监督学习方法对于执行分类任务的神经网络训练至关重要。虽然监督学习训练能够实现非常高精度，但往往需要将神经网络参数数量依赖于类别数量，这限制了当类别数量极其庞大或事先未知时的应用。本文提出了一种方法，使得可以训练相同的神经网络架构，无论类别数量如何。这一目标通过在神经网络训练过程中使用预定义的向量系统作为目标潜在空间配置（LSC）来实现。我们讨论了目标配置的期望属性，并在实验中选择了随机扰动的An根系统向量。这些向量通过匹配神经网络预测与预定义向量，成功地在Cinic-10和ImageNet-1K上训练了编码器和视觉变换器（ViT），并在低维和高维情况下验证了其有效性。最后，我们在包含128万个类别的数据集上训练了ViT，展示了该方法在处理类别数量极其庞大的数据集时的应用。此外，还讨论了潜在空间配置在终身学习和神经网络蒸馏中的潜在应用，进一步展示了所提出方法的灵活性。 

---
# Offline Reinforcement Learning in Large State Spaces: Algorithms and Guarantees 

**Title (ZH)**: 大型状态空间中的离线强化学习：算法与保证 

**Authors**: Nan Jiang, Tengyang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2510.04088)  

**Abstract**: This article introduces the theory of offline reinforcement learning in large state spaces, where good policies are learned from historical data without online interactions with the environment. Key concepts introduced include expressivity assumptions on function approximation (e.g., Bellman completeness vs. realizability) and data coverage (e.g., all-policy vs. single-policy coverage). A rich landscape of algorithms and results is described, depending on the assumptions one is willing to make and the sample and computational complexity guarantees one wishes to achieve. We also discuss open questions and connections to adjacent areas. 

**Abstract (ZH)**: 本文介绍在大面积状态空间中离线强化学习的理论，通过历史数据学习良好的策略而无需与环境进行在线交互。介绍的关键概念包括函数逼近的表达性假设（如贝尔曼完备性与实现性）和数据覆盖范围（如全策略覆盖与单策略覆盖）。描述了根据不同假设和样本及计算复杂性保证的丰富算法和结果。此外，还讨论了开放问题及其与相邻领域的联系。 

---
# A Contextual Quality Reward Model for Reliable and Efficient Best-of-N Sampling 

**Title (ZH)**: 上下文质量奖励模型：实现可靠和高效的最佳选项采样 

**Authors**: Hyung Gyu Rho  

**Link**: [PDF](https://arxiv.org/pdf/2510.04087)  

**Abstract**: Modern preference alignment techniques, such as Best-of-N (BoN) sampling, rely on reward models trained with pairwise comparison data. While effective at learning relative preferences, this paradigm fails to capture a signal of response acceptability, leaving systems vulnerable to selecting the least bad of many unacceptable options. This is particularly problematic for hard prompts, where the risk of such false acceptances increases with the number of samples. In this paper, we address this critical reliability gap by introducing a new data collection and modeling framework. By augmenting preference data with an outside option, inspired by discrete choice models, we train a reward model that can distinguish not just what is \textit{better}, but what is \textit{good enough}. We leverage this capability to create an adaptive inference strategy, best of mini-N in-loop, which partitions the generation budget into sequential loops with a calibrated, early-exit condition. Our experiments show that when tuned as an alignment guardrail, it reduces reliability failures by 70\%, and when tuned as an inference accelerator, it improves average inference speed by over 22\% in IMDB-sentiment setting. We thus provide a principled and flexible framework for practitioners to explicitly manage the trade-off between reliability and computational efficiency. 

**Abstract (ZH)**: 现代偏好对齐技术，如Best-of-N（BoN）采样，依赖于通过成对比较数据训练的奖励模型。尽管这种范式在学习相对偏好方面非常有效，但它无法捕捉响应可接受性的信号，从而使系统容易选择众多不可接受选项中最坏的一个。这在困难提示中尤为关键，样本数量越多，这种错误接受的风险越高。在本文中，我们通过引入一个新的数据收集和建模框架来填补这一关键可靠性的缺口。通过借鉴离散选择模型中的外部选项，我们训练了一个奖励模型，它可以区分什么不仅是“更好”，而且是“足够好”。我们利用这一能力创建了一种自适应推理策略——在环中的Best-of-mini-N，将生成预算划分为具有校准且早期退出条件的顺序循环。我们的实验显示，作为对齐护栏进行调整时，它可以将可靠性失败降低70%，作为推理加速器进行调整时，在IMDB情感分析设置中平均提高推理速度超过22%。因此，我们提供了一个原则性的灵活框架，使从业者能够明确管理可靠性和计算效率之间的权衡。 

---
# Quantization Range Estimation for Convolutional Neural Networks 

**Title (ZH)**: 卷积神经网络的量化范围估计 

**Authors**: Bingtao Yang, Yujia Wang, Mengzhi Jiao, Hongwei Huo  

**Link**: [PDF](https://arxiv.org/pdf/2510.04044)  

**Abstract**: Post-training quantization for reducing the storage of deep neural network models has been demonstrated to be an effective way in various tasks. However, low-bit quantization while maintaining model accuracy is a challenging problem. In this paper, we present a range estimation method to improve the quantization performance for post-training quantization. We model the range estimation into an optimization problem of minimizing quantization errors by layer-wise local minima. We prove this problem is locally convex and present an efficient search algorithm to find the optimal solution. We propose the application of the above search algorithm to the transformed weights space to do further improvement in practice. Our experiments demonstrate that our method outperforms state-of-the-art performance generally on top-1 accuracy for image classification tasks on the ResNet series models and Inception-v3 model. The experimental results show that the proposed method has almost no loss of top-1 accuracy in 8-bit and 6-bit settings for image classifications, and the accuracy of 4-bit quantization is also significantly improved. The code is available at this https URL. 

**Abstract (ZH)**: 基于训练后量化减少深度神经网络模型存储量的研究：一种范围估计方法以提高低比特量化性能 

---
# The Debate on RLVR Reasoning Capability Boundary: Shrinkage, Expansion, or Both? A Two-Stage Dynamic View 

**Title (ZH)**: 关于RLVR推理能力边界之争：缩小、扩展，还是两者兼有？一种两阶段动态视角 

**Authors**: Xinhao Yao, Lu Yu, Xiaolin Hu, Fengwei Teng, Qing Cui, Jun Zhou, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04028)  

**Abstract**: The ongoing debate on whether reinforcement learning with verifiable rewards (RLVR) expands or shrinks the reasoning capabilities of large language models (LLMs) remains unresolved. Some studies contend that RLVR mainly improves sampling efficiency but at the expense of diversity and exploratory capacity, resulting in capability boundary shrinkage. In contrast, others demonstrate that prolonged training can lead to the emergence of novel reasoning strategies, suggesting capability boundary expansion. To reconcile these contradictory findings, we theoretically and empirically show that both perspectives are partially valid-each aligning with a separate phase in an inherent two-stage probability mass dynamic: (1) Exploitation stage: initially, the model primarily samples explored high-reward and low-reward tokens, while rarely selecting the potentially optimal token. Positive advantage estimates increase the probability of high-reward tokens and decrease those of low-reward tokens, yet the optimal token's probability remains largely unchanged during this stage. (2) Exploration stage: as training advances, the growth rate of previously acquired high-reward tokens slows as their probabilities approach saturation. When a potentially optimal token-now receiving positive advantage estimates-is occasionally sampled, its probability increases, while those of the originally high-reward tokens decrease. This dynamic suggests that over-exploitation during the exploitation stage may lead to capability boundary shrinkage, whereas prolonged training into the exploration stage can promote an expansion of the reasoning capability boundary. Building upon our insights, we revisit the potential of only using relative negative gradients for prolonging training, providing a theoretical and empirical foundation for the development of more advanced reasoning capabilities. 

**Abstract (ZH)**: 有关强化学习带有可验证奖励（RLVR）是否扩大或缩小大型语言模型（LLMs）的推理能力的持续争议尚未解决。一些研究认为，RLVR 主要提高采样效率，但牺牲了多样性与探索能力，导致能力边界缩小。相反，其他研究则表明，长期训练可以促使出现新的推理策略，表明能力边界可能扩大。为了调和这些矛盾的研究结果，我们从理论上和实证上证明两种观点在某种程度上都是正确的，每种观点分别对应于固有的两阶段概率质量动态的不同阶段：（1）利用阶段：最初，模型主要采样探索的高奖励和低奖励令牌，而很少选择潜在的最佳令牌。正的优势估计增加了高奖励令牌的概率，同时降低了低奖励令牌的概率，但在该阶段，最佳令牌的概率变化不大。（2）探索阶段：随着训练的进行，之前获得的高奖励令牌的增长率逐渐减缓，因为它们的概率接近饱和。当一个潜在的最佳令牌（现在获得正的优势估计）偶尔被采样时，它的概率会增加，而最初高奖励令牌的概率会减少。这种动态表明，在利用阶段过度利用可能会导致能力边界缩小，而长时间训练进入探索阶段则可以促进推理能力边界的扩张。基于我们的见解，我们重新审视仅使用相对负梯度延长训练的潜力，为开发更高级的推理能力提供了理论和实证基础。 

---
# Replacing Softmax Similarity with a Sharpened Angular Similarity: Theory and Practice of Scaling To Billion-Context Attention 

**Title (ZH)**: 用尖锐角度相似度取代Softmax相似度：亿级上下文注意力建模的理论与实践 

**Authors**: Sahil Joshi, Agniva Chowdhury, Amar Kanakamedala, Ekam Singh, Evan Tu, Anshumali Shrivastava  

**Link**: [PDF](https://arxiv.org/pdf/2510.04008)  

**Abstract**: Softmax Attention has a quadratic time complexity, which becomes prohibitive to run at long contexts, even with highly optimized GPU kernels. For example, FlashAttention (an exact, GPU-optimized implementation of Softmax Attention) cannot complete a single forward-backward pass of a multi-head attention layer once the context exceeds ~4 million tokens on an NVIDIA GH200 (96 GB). We introduce RACE Attention, a kernel-inspired alternative to Softmax Attention that is linear in sequence length and embedding dimension. RACE Attention replaces the exponential kernel with a sharpened angular (cosine) similarity, and approximates attention outputs via randomized projections and soft Locality-Sensitive Hashing (LSH). Across language modeling, masked language modeling, and text classification, RACE Attention matches the accuracy of strong baselines while reducing runtime and memory. In a controlled scale test, it processes up to 12 million tokens during a single forward-backward pass on an NVIDIA GH200 GPU and 75 million tokens on an Intel Xeon Gold 5220R CPU, well beyond the practical limits of the current state-of-the-art attention implementations. RACE Attention thus offers a practical, theoretically grounded mechanism for outrageously long context windows on today's hardware. We hope that it gets adopted in practice. 

**Abstract (ZH)**: RACE Attention: A Linear-Time Alternative to Softmax Attention 

---
# Named Entity Recognition in COVID-19 tweets with Entity Knowledge Augmentation 

**Title (ZH)**: 基于实体知识增强的COVID-19推文命名实体识别 

**Authors**: Xuankang Zhang, Jiangming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04001)  

**Abstract**: The COVID-19 pandemic causes severe social and economic disruption around the world, raising various subjects that are discussed over social media. Identifying pandemic-related named entities as expressed on social media is fundamental and important to understand the discussions about the pandemic. However, there is limited work on named entity recognition on this topic due to the following challenges: 1) COVID-19 texts in social media are informal and their annotations are rare and insufficient to train a robust recognition model, and 2) named entity recognition in COVID-19 requires extensive domain-specific knowledge. To address these issues, we propose a novel entity knowledge augmentation approach for COVID-19, which can also be applied in general biomedical named entity recognition in both informal text format and formal text format. Experiments carried out on the COVID-19 tweets dataset and PubMed dataset show that our proposed entity knowledge augmentation improves NER performance in both fully-supervised and few-shot settings. Our source code is publicly available: this https URL 

**Abstract (ZH)**: COVID-19 pandemic引起的全球社会和经济冲击导致了各种在社交媒体上讨论的主题。识别表达在社交媒体上的与疫情相关的命名实体对于理解关于疫情的讨论至关重要。然而，由于以下挑战，有关该主题的命名实体识别工作非常有限：1) 社交媒体上的COVID-19文本不正式，其标注稀少不充分，难以训练出 robust 的识别模型，2) 在COVID-19领域的命名实体识别需要广泛的领域专业知识。为应对这些挑战，我们提出了一种针对COVID-19的新型实体知识增强方法，该方法也可应用于以非正式文本格式和正式文本格式进行的一般生物医学命名实体识别。在COVID-19推文数据集和PubMed数据集上的实验表明，我们提出的实体知识增强方法在完全监督和少量监督设置中均提高了命名实体识别性能。我们的源代码已公开：this https URL。 

---
# AI-Driven Grading and Moderation for Collaborative Projects in Computer Science Education 

**Title (ZH)**: 基于AI的协作项目评分与审核在计算机科学教育中的应用 

**Authors**: Songmei Yu, Andrew Zagula  

**Link**: [PDF](https://arxiv.org/pdf/2510.03998)  

**Abstract**: Collaborative group projects are integral to computer science education, as they foster teamwork, problem-solving skills, and industry-relevant competencies. However, assessing individual contributions within group settings has long been a challenge. Traditional assessment strategies, such as the equal distribution of grades or subjective peer assessments, often fall short in terms of fairness, objectivity, and scalability, particularly in large classrooms. This paper introduces a semi-automated, AI-assisted grading system that evaluates both project quality and individual effort using repository mining, communication analytics, and machine learning models. The system comprises modules for project evaluation, contribution analysis, and grade computation, integrating seamlessly with platforms like GitHub. A pilot deployment in a senior-level course demonstrated high alignment with instructor assessments, increased student satisfaction, and reduced instructor grading effort. We conclude by discussing implementation considerations, ethical implications, and proposed enhancements to broaden applicability. 

**Abstract (ZH)**: 协作小组项目是计算机科学教育中的重要组成部分，它们培养团队合作、问题解决能力和与行业相关的能力。然而，在小组环境中评估个人贡献始终是一项挑战。传统的评估策略，如平均分配成绩或主观的同伴评估，往往在公平性、客观性和可扩展性方面不尽如人意，特别是在大型课堂中。本文介绍了一种半自动化、基于AI的评分系统，该系统通过代码库挖掘、通信分析和机器学习模型评估项目质量和个人贡献。该系统包括项目评估、贡献分析和成绩计算模块，可以无缝集成到如GitHub这样的平台上。在一门高年级课程中的试点部署表明，该系统与教师评估高度契合，提高了学生满意度，并减少了教师的评分工作量。最后，本文讨论了实施考虑、伦理问题及拟议的改进措施，以扩大其适用范围。 

---
# PrivSpike: Employing Homomorphic Encryption for Private Inference of Deep Spiking Neural Networks 

**Title (ZH)**: PrivSpike: 利用同态加密实现深度脉冲神经网络的隐私推理 

**Authors**: Nges Brian Njungle, Eric Jahns, Milan Stojkov, Michel A. Kinsy  

**Link**: [PDF](https://arxiv.org/pdf/2510.03995)  

**Abstract**: Deep learning has become a cornerstone of modern machine learning. It relies heavily on vast datasets and significant computational resources for high performance. This data often contains sensitive information, making privacy a major concern in deep learning. Spiking Neural Networks (SNNs) have emerged as an energy-efficient alternative to conventional deep learning approaches. Nevertheless, SNNs still depend on large volumes of data, inheriting all the privacy challenges of deep learning. Homomorphic encryption addresses this challenge by allowing computations to be performed on encrypted data, ensuring data confidentiality throughout the entire processing pipeline. In this paper, we introduce PRIVSPIKE, a privacy-preserving inference framework for SNNs using the CKKS homomorphic encryption scheme. PRIVSPIKE supports arbitrary depth SNNs and introduces two key algorithms for evaluating the Leaky Integrate-and-Fire activation function: (1) a polynomial approximation algorithm designed for high-performance SNN inference, and (2) a novel scheme-switching algorithm that optimizes precision at a higher computational cost. We evaluate PRIVSPIKE on MNIST, CIFAR-10, Neuromorphic MNIST, and CIFAR-10 DVS using models from LeNet-5 and ResNet-19 architectures, achieving encrypted inference accuracies of 98.10%, 79.3%, 98.1%, and 66.0%, respectively. On a consumer-grade CPU, SNN LeNet-5 models achieved inference times of 28 seconds on MNIST and 212 seconds on Neuromorphic MNIST. For SNN ResNet-19 models, inference took 784 seconds on CIFAR-10 and 1846 seconds on CIFAR-10 DVS. These results establish PRIVSPIKE as a viable and efficient solution for secure SNN inference, bridging the gap between energy-efficient deep neural networks and strong cryptographic privacy guarantees while outperforming prior encrypted SNN solutions. 

**Abstract (ZH)**: 一种基于CKKS同态加密方案的隐私 preserved Spiking Neural Networks 推理框架：PRIVSPIKE 

---
# Towards Carbon-Aware Container Orchestration: Predicting Workload Energy Consumption with Federated Learning 

**Title (ZH)**: 基于碳意识的容器编排：基于联邦学习的负载能耗预测 

**Authors**: Zainab Saad, Jialin Yang, Henry Leung, Steve Drew  

**Link**: [PDF](https://arxiv.org/pdf/2510.03970)  

**Abstract**: The growing reliance on large-scale data centers to run resource-intensive workloads has significantly increased the global carbon footprint, underscoring the need for sustainable computing solutions. While container orchestration platforms like Kubernetes help optimize workload scheduling to reduce carbon emissions, existing methods often depend on centralized machine learning models that raise privacy concerns and struggle to generalize across diverse environments. In this paper, we propose a federated learning approach for energy consumption prediction that preserves data privacy by keeping sensitive operational data within individual enterprises. By extending the Kubernetes Efficient Power Level Exporter (Kepler), our framework trains XGBoost models collaboratively across distributed clients using Flower's FedXgbBagging aggregation using a bagging strategy, eliminating the need for centralized data sharing. Experimental results on the SPECPower benchmark dataset show that our FL-based approach achieves 11.7 percent lower Mean Absolute Error compared to a centralized baseline. This work addresses the unresolved trade-off between data privacy and energy prediction efficiency in prior systems such as Kepler and CASPER and offers enterprises a viable pathway toward sustainable cloud computing without compromising operational privacy. 

**Abstract (ZH)**: 基于联邦学习的能源消耗预测方法：保持数据隐私同时提高能源预测效率 

---
# Strategy Logic, Imperfect Information, and Hyperproperties 

**Title (ZH)**: 策略逻辑、不完美信息与超属性 

**Authors**: Raven Beutner, Bernd Finkbeiner  

**Link**: [PDF](https://arxiv.org/pdf/2510.03952)  

**Abstract**: Strategy logic (SL) is a powerful temporal logic that enables first-class reasoning over strategic behavior in multi-agent systems (MAS). In many MASs, the agents (and their strategies) cannot observe the global state of the system, leading to many extensions of SL centered around imperfect information, such as strategy logic with imperfect information (SL$_\mathit{ii}$). Along orthogonal lines, researchers have studied the combination of strategic behavior and hyperproperties. Hyperproperties are system properties that relate multiple executions in a system and commonly arise when specifying security policies. Hyper Strategy Logic (HyperSL) is a temporal logic that combines quantification over strategies with the ability to express hyperproperties on the executions of different strategy profiles. In this paper, we study the relation between SL$_\mathit{ii}$ and HyperSL. Our main result is that both logics (restricted to formulas where no state formulas are nested within path formulas) are equivalent in the sense that we can encode SL$_\mathit{ii}$ instances into HyperSL instances and vice versa. For the former direction, we build on the well-known observation that imperfect information is a hyperproperty. For the latter direction, we construct a self-composition of MASs and show how we can simulate hyperproperties using imperfect information. 

**Abstract (ZH)**: SL与HyperSL之间的关系研究：从 imperfect信息扩展到超属性逻辑 

---
# On the Convergence and Size Transferability of Continuous-depth Graph Neural Networks 

**Title (ZH)**: 连续时深图神经网络的收敛性及规模可转移性 

**Authors**: Mingsong Yan, Charles Kulick, Sui Tang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03923)  

**Abstract**: Continuous-depth graph neural networks, also known as Graph Neural Differential Equations (GNDEs), combine the structural inductive bias of Graph Neural Networks (GNNs) with the continuous-depth architecture of Neural ODEs, offering a scalable and principled framework for modeling dynamics on graphs. In this paper, we present a rigorous convergence analysis of GNDEs with time-varying parameters in the infinite-node limit, providing theoretical insights into their size transferability. To this end, we introduce Graphon Neural Differential Equations (Graphon-NDEs) as the infinite-node limit of GNDEs and establish their well-posedness. Leveraging tools from graphon theory and dynamical systems, we prove the trajectory-wise convergence of GNDE solutions to Graphon-NDE solutions. Moreover, we derive explicit convergence rates under two deterministic graph sampling regimes: (1) weighted graphs sampled from smooth graphons, and (2) unweighted graphs sampled from $\{0,1\}$-valued (discontinuous) graphons. We further establish size transferability bounds, providing theoretical justification for the practical strategy of transferring GNDE models trained on moderate-sized graphs to larger, structurally similar graphs without retraining. Numerical experiments using synthetic and real data support our theoretical findings. 

**Abstract (ZH)**: 连续深度图神经网络，也称为图神经微分方程（GNDEs），将图神经网络（GNNs）的结构归纳偏置与神经ODEs的连续深度架构结合起来，为图上的动力学建模提供了一个可扩展且原理上的框架。本文在无限节点极限下对具有时间变参数的GNDEs进行了严格的收敛性分析，为其规模可扩展性提供了理论洞察。为此，我们引入了图限神经微分方程（Graphon-NDEs）作为GNDEs的无限节点极限，并建立了其适定性。利用图限理论和动力系统工具，我们证明了GNDE解逐轨迹收敛于Graphon-NDE解。此外，我们得出了在两种确定性图采样模式下的显式收敛率：(1) 来自光滑图限的加权图，(2) 来自二进制值（不连续）图限的无权重图。我们进一步建立了规模可扩展性边界，为在保留训练模型性能的情况下将GNDE模型从较小的图转移到更大且结构相似的图提供了理论依据。数值实验使用合成和真实数据支持我们的理论发现。 

---
# PoseGaze-AHP: A Knowledge-Based 3D Dataset for AI-Driven Ocular and Postural Diagnosis 

**Title (ZH)**: 基于知识的3D数据集：PoseGaze-AHP，用于AI驱动的眼部和姿势诊断 

**Authors**: Saja Al-Dabet, Sherzod Turaev, Nazar Zaki, Arif O. Khan, Luai Eldweik  

**Link**: [PDF](https://arxiv.org/pdf/2510.03873)  

**Abstract**: Diagnosing ocular-induced abnormal head posture (AHP) requires a comprehensive analysis of both head pose and ocular movements. However, existing datasets focus on these aspects separately, limiting the development of integrated diagnostic approaches and restricting AI-driven advancements in AHP analysis. To address this gap, we introduce PoseGaze-AHP, a novel 3D dataset that synchronously captures head pose and gaze movement information for ocular-induced AHP assessment. Structured clinical data were extracted from medical literature using large language models (LLMs) through an iterative process with the Claude 3.5 Sonnet model, combining stepwise, hierarchical, and complex prompting strategies. The extracted records were systematically imputed and transformed into 3D representations using the Neural Head Avatar (NHA) framework. The dataset includes 7,920 images generated from two head textures, covering a broad spectrum of ocular conditions. The extraction method achieved an overall accuracy of 91.92%, demonstrating its reliability for clinical dataset construction. PoseGaze-AHP is the first publicly available resource tailored for AI-driven ocular-induced AHP diagnosis, supporting the development of accurate and privacy-compliant diagnostic tools. 

**Abstract (ZH)**: 基于头姿和眼动的斜颈综合症诊断需要对头部姿态和眼动进行全面分析。然而，现有数据集分别关注这些方面，限制了综合诊断方法的发展并限制了基于AI的眼动诱导斜颈综合症分析进展。为解决这一问题，我们引入了PoseGaze-AHP，这是一个新颖的3D数据集，可以同步捕捉头部姿态和视线运动信息，用于眼动诱导斜颈综合症评估。通过迭代过程使用大型语言模型（LLMs）结合逐步、分级和复杂提示策略从医学文献中提取结构化临床数据，并使用Neural Head Avatar (NHA)框架系统地补充分数值并转换为3D表示。该数据集包括7,920张来自两种头部纹理生成的图像，涵盖了广泛的眼部条件。提取方法的准确率为91.92%，证明其适用于临床数据集构建。PoseGaze-AHP是首款面向基于AI的眼动诱导斜颈综合症诊断的公开资源，支持开发准确且符合隐私保护的诊断工具。 

---
# Optimal Scaling Needs Optimal Norm 

**Title (ZH)**: 最优缩放需要最优范数 

**Authors**: Oleg Filatov, Jiangtao Wang, Jan Ebert, Stefan Kesselheim  

**Link**: [PDF](https://arxiv.org/pdf/2510.03871)  

**Abstract**: Despite recent progress in optimal hyperparameter transfer under model and dataset scaling, no unifying explanatory principle has been established. Using the Scion optimizer, we discover that joint optimal scaling across model and dataset sizes is governed by a single invariant: the operator norm of the output layer. Across models with up to 1.3B parameters trained on up to 138B tokens, the optimal learning rate/batch size pair $(\eta^{\ast}, B^{\ast})$ consistently has the same operator norm value - a phenomenon we term norm transfer. This constant norm condition is necessary but not sufficient: while for each dataset size, multiple $(\eta, B)$ reach the optimal norm, only a unique $(\eta^{\ast}, B^{\ast})$ achieves the best loss. As a sufficient condition, we provide the first measurement of $(\eta^{\ast}, B^{\ast})$ scaling with dataset size for Scion, and find that the scaling rules are consistent with those of the Adam optimizer. Tuning per-layer-group learning rates also improves model performance, with the output layer being the most sensitive and hidden layers benefiting from lower learning rates. We provide practical insights on norm-guided optimal scaling and release our Distributed Scion (Disco) implementation with logs from over two thousand runs to support research on LLM training dynamics at scale. 

**Abstract (ZH)**: 尽管在模型和数据集规模下的最优超参数传递方面取得了一定进展，但仍缺乏一个统一的解释原则。使用Scion优化器，我们发现模型和数据集规模的联合最优缩放受单一不变量控制：输出层的操作范数。在训练参数量从1.3亿到1380亿、 token数从138亿的数据集上，最优的学习率/批量大小对$(\eta^{\ast}, B^{\ast})$始终具有相同的操作范数值——我们称之为范数传递。该恒定范数条件是必要的但不充分的：尽管对于每个数据集大小，存在多个$(\eta, B)$可以达到最优范数，但只有唯一的$(\eta^{\ast}, B^{\ast})$能实现最佳损失。作为充分条件，我们提供了Scion中$(\eta^{\ast}, B^{\ast})$随数据集规模缩放的第一个测量结果，并发现这些缩放规则与Adam优化器的规则一致。按层组调整学习率也提高了模型性能，输出层最为敏感，隐藏层受益于较低的学习率。我们提供了范数引导下的最优缩放的实用见解，并发布了Distributed Scion（Disco）实现，提供了超过两千次运行的日志，以支持大规模LLM训练动力学的研究。 

---
# AI Adoption Across Mission-Driven Organizations 

**Title (ZH)**: AI采纳在使命驱动组织中的应用 

**Authors**: Dalia Ali, Muneeb Ahmed, Hailan Wang, Arfa Khan, Naira Paola Arnez Jordan, Sunnie S. Y. Kim, Meet Dilip Muchhala, Anne Kathrin Merkle, Orestis Papakyriakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2510.03868)  

**Abstract**: Despite AI's promise for addressing global challenges, empirical understanding of AI adoption in mission-driven organizations (MDOs) remains limited. While research emphasizes individual applications or ethical principles, little is known about how resource-constrained, values-driven organizations navigate AI integration across operations. We conducted thematic analysis of semi-structured interviews with 15 practitioners from environmental, humanitarian, and development organizations across the Global North and South contexts. Our analysis examines how MDOs currently deploy AI, what barriers constrain adoption, and how practitioners envision future integration. MDOs adopt AI selectively, with sophisticated deployment in content creation and data analysis while maintaining human oversight for mission-critical applications. When AI's efficiency benefits conflict with organizational values, decision-making stalls rather than negotiating trade-offs. This study contributes empirical evidence that AI adoption in MDOs should be understood as conditional rather than inevitable, proceeding only where it strengthens organizational sovereignty and mission integrity while preserving human-centered approaches essential to their missions. 

**Abstract (ZH)**: 尽管人工智能在应对全球挑战方面充满 promise，但有关使命驱动组织（MDOs）采用人工智能的实证理解仍有限。尽管研究强调个人应用或伦理原则，但对于资源受限、价值观驱动的组织如何在运营中导航人工智能集成却知之甚少。我们对来自全球北南不同背景下环境、人道主义和开发组织的 15 名从业人员进行了半结构化访谈，并进行了主题分析。分析了 MDOs 目前如何采用人工智能、哪些障碍限制了采用，以及从业人员如何设想未来的集成。MDOs 选择性地采用人工智能，在内容生成和数据分析方面进行了复杂的部署，同时在关键任务应用中保留了人的监督。当人工智能的效率益处与组织价值观发生冲突时，决策停滞不前，而不是权衡取舍。本研究提供了实证证据，表明 MDOs 采用人工智能应被视为有条件的而非不可避免的，仅在增强组织自主权和使命完整性的同时保留对实现其使命至关重要的以人为中心的方法，才能进行。 

---
# Proximal Diffusion Neural Sampler 

**Title (ZH)**: 邻近扩散神经采样器 

**Authors**: Wei Guo, Jaemoo Choi, Yuchen Zhu, Molei Tao, Yongxin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.03824)  

**Abstract**: The task of learning a diffusion-based neural sampler for drawing samples from an unnormalized target distribution can be viewed as a stochastic optimal control problem on path measures. However, the training of neural samplers can be challenging when the target distribution is multimodal with significant barriers separating the modes, potentially leading to mode collapse. We propose a framework named \textbf{Proximal Diffusion Neural Sampler (PDNS)} that addresses these challenges by tackling the stochastic optimal control problem via proximal point method on the space of path measures. PDNS decomposes the learning process into a series of simpler subproblems that create a path gradually approaching the desired distribution. This staged procedure traces a progressively refined path to the desired distribution and promotes thorough exploration across modes. For a practical and efficient realization, we instantiate each proximal step with a proximal weighted denoising cross-entropy (WDCE) objective. We demonstrate the effectiveness and robustness of PDNS through extensive experiments on both continuous and discrete sampling tasks, including challenging scenarios in molecular dynamics and statistical physics. 

**Abstract (ZH)**: 基于 proximal 点方法的路径测度最优控制框架：多模态分布下的扩散神经采样器（Proximal Diffusion Neural Sampler, PDNS） 

---
# Detecting Invariant Manifolds in ReLU-Based RNNs 

**Title (ZH)**: 基于ReLU的RNN中不变流形的检测 

**Authors**: Lukas Eisenmann, Alena Brändle, Zahra Monfared, Daniel Durstewitz  

**Link**: [PDF](https://arxiv.org/pdf/2510.03814)  

**Abstract**: Recurrent Neural Networks (RNNs) have found widespread applications in machine learning for time series prediction and dynamical systems reconstruction, and experienced a recent renaissance with improved training algorithms and architectural designs. Understanding why and how trained RNNs produce their behavior is important for scientific and medical applications, and explainable AI more generally. An RNN's dynamical repertoire depends on the topological and geometrical properties of its state space. Stable and unstable manifolds of periodic points play a particularly important role: They dissect a dynamical system's state space into different basins of attraction, and their intersections lead to chaotic dynamics with fractal geometry. Here we introduce a novel algorithm for detecting these manifolds, with a focus on piecewise-linear RNNs (PLRNNs) employing rectified linear units (ReLUs) as their activation function. We demonstrate how the algorithm can be used to trace the boundaries between different basins of attraction, and hence to characterize multistability, a computationally important property. We further show its utility in finding so-called homoclinic points, the intersections between stable and unstable manifolds, and thus establish the existence of chaos in PLRNNs. Finally we show for an empirical example, electrophysiological recordings from a cortical neuron, how insights into the underlying dynamics could be gained through our method. 

**Abstract (ZH)**: 递归神经网络（RNNs）在时间序列预测和动力系统重建中的广泛应用于机器学习中最近经历了一次复兴，这得益于改进的训练算法和网络架构设计。理解训练后的RNNs产生其行为的原因和机制对于科学和医学应用以及更广泛的可解释人工智能至关重要。RNN的动力学范围取决于其状态空间的拓扑和几何特性。周期点的稳定流形和不稳定流形特别重要：它们将动力系统的状态空间分割成不同的吸引子盆地，它们的交点导致具有分形几何的混沌动力学。我们介绍了一种用于检测这些流形的新算法，重点研究使用修正线性单元（ReLU）作为激活函数的规则线性RNN（PLRNN）。我们展示了该算法如何用于追踪不同吸引子盆地之间的边界，从而表征多稳性，这是一个计算上重要的属性。我们还展示了该算法在查找所谓的同宿点（即稳定流形和不稳定流形的交点）方面的效用，从而确立了PLRNN中混沌的存在性。最后，我们通过一个实证例子——皮层神经元的电生理记录，展示了如何通过我们的方法获得底层动力学的洞察。 

---
# 6G-Enabled Digital Twin Framework for Real-Time Cyber-Physical Systems: An Experimental Validation with Industrial Bearing Fault Detection 

**Title (ZH)**: 6G赋能的数字孪生框架在实时物理- cyber系统中的实验验证：以工业轴承故障检测为例 

**Authors**: Vaskar Chakma, Wooyeol Choi  

**Link**: [PDF](https://arxiv.org/pdf/2510.03807)  

**Abstract**: Current Cyber-Physical Systems (CPS) integrated with Digital Twin (DT) technology face critical limitations in achieving real-time performance for mission-critical industrial applications. Existing 5G-enabled systems suffer from latencies exceeding 10ms, which are inadequate for applications requiring sub-millisecond response times, such as autonomous industrial control and predictive maintenance. This research aims to develop and validate a 6G-enabled Digital Twin framework that achieves ultra-low latency communication and real-time synchronization between physical industrial assets and their digital counterparts, specifically targeting bearing fault detection as a critical industrial use case. The proposed framework integrates terahertz communications (0.1-1 THz), intelligent reflecting surfaces, and edge artificial intelligence within a five-layer architecture. Experimental validation was conducted using the Case Western Reserve University (CWRU) bearing dataset, implementing comprehensive feature extraction (15 time and frequency domain features) and Random Forest classification algorithms. The system performance was evaluated against traditional WiFi-6 and 5G networks across multiple metrics, including classification accuracy, end-to-end latency, and scalability. It achieved 97.7% fault classification accuracy with 0.8ms end-to-end latency, representing a 15.6x improvement over WiFi-6 (12.5ms) and 5.25x improvement over 5G (4.2ms) networks. The system demonstrated superior scalability with sub-linear processing time growth and maintained consistent performance across four bearing fault categories (normal, inner race, outer race, and ball faults) with macro-averaged F1-scores exceeding 97%. 

**Abstract (ZH)**: 现有的数字孪生技术集成的-current-cyber-physical系统（CPS）在实现关键工业应用的实时性能方面面临重要限制。现有的5G使能系统面临的延迟超过10ms，对于需要亚毫秒级响应时间的自主工业控制和预测性维护等应用而言是不充分的。本研究旨在开发并验证一种6G使能的数字孪生框架，实现超低延迟通信和物理工业资产与其数字对应物之间的实时同步，特别针对轴承故障检测这一关键工业应用场景。所提出的框架在五层架构中集成了太赫兹通信（0.1-1 THz）、智能反射表面和边缘人工智能技术。通过使用辛辛那提大学（CWRU）轴承数据集进行了实验验证，实现了全面的特征提取（15个时间域和频域特征）和随机森林分类算法。系统性能在多项指标上（包括分类准确性、端到端延迟和可扩展性）与传统WiFi-6和5G网络进行了对比评估。系统在端到端延迟为0.8ms的情况下实现了97.7%的故障分类准确率，分别比WiFi-6（12.5ms）和5G（4.2ms）网络的性能提高了15.6倍和5.25倍。该系统展示了优越的可扩展性，处理时间呈次线性增长，并在四种轴承故障类别（正常、内圈、外圈和滚珠故障）上保持了一致性，宏均F1分数超过97%。 

---
# Mechanistic Interpretability of Socio-Political Frames in Language Models 

**Title (ZH)**: 社会政治框架在语言模型中的机理可解释性 

**Authors**: Hadi Asghari, Sami Nenno  

**Link**: [PDF](https://arxiv.org/pdf/2510.03799)  

**Abstract**: This paper explores the ability of large language models to generate and recognize deep cognitive frames, particularly in socio-political contexts. We demonstrate that LLMs are highly fluent in generating texts that evoke specific frames and can recognize these frames in zero-shot settings. Inspired by mechanistic interpretability research, we investigate the location of the `strict father' and `nurturing parent' frames within the model's hidden representation, identifying singular dimensions that correlate strongly with their presence. Our findings contribute to understanding how LLMs capture and express meaningful human concepts. 

**Abstract (ZH)**: 本论文探讨了大型语言模型生成和识别深层认知框架的能力，特别是在社会政治情境中的表现。我们证明了大型语言模型在生成唤起特定框架的文本方面极为流畅，并能在零样本设置中识别这些框架。受机械可解释性研究的启发，我们探讨了“严格父亲”和“养育父母”框架在模型隐藏表示中的位置，确定了与它们存在高度相关的单个维度。我们的研究结果有助于理解大型语言模型如何捕捉和表达有意义的人类概念。 

---
# Lightweight and Data-Efficient MultivariateTime Series Forecasting using Residual-Stacked Gaussian (RS-GLinear) Architecture 

**Title (ZH)**: 使用残差堆叠高斯(RS-GLinear)架构的轻量级和数据-efficient多变量时间序列预测 

**Authors**: Abukar Ali  

**Link**: [PDF](https://arxiv.org/pdf/2510.03788)  

**Abstract**: Following the success of Transformer architectures in language modeling, particularly their ability to capture long-range dependencies, researchers have explored how these architectures can be adapted for time-series forecasting. Transformer-based models have been proposed to handle both short- and long-term dependencies when predicting future values from historical data. However, studies such as those by Zeng et al. (2022) and Rizvi et al. (2025) have reported mixed results in long-term forecasting tasks. In this work, we evaluate the Gaussian-based Linear architecture introduced by Rizvi et al. (2025) and present an enhanced version called the Residual Stacked Gaussian Linear (RSGL) model. We also investigate the broader applicability of the RSGL model in additional domains, including financial time series and epidemiological data. Experimental results show that the RSGL model achieves improved prediction accuracy and robustness compared to both the baseline Gaussian Linear and Transformer-based models. 

**Abstract (ZH)**: 基于Transformer架构在语言建模中的成功，特别是其捕捉长范围依赖的能力，研究人员探索了这些架构如何适应时间序列预测。基于Transformer的模型被提出用于从历史数据预测未来值时同时处理短期和长期依赖。然而，诸如Zeng等（2022）和Rizvi等（2025）的研究在长期预测任务中报道了混合结果。在本研究中，我们评估了Rizvi等（2025）引入的基于高斯的线性架构，并呈现其增强版本，即残差堆叠高斯线性（RSGL）模型。我们还调查了RSGL模型在其他领域的更广泛适用性，包括金融时间序列和流行病学数据。实验结果表明，RSGL模型比基线高斯线性和Transformer基模型在预测准确性和稳健性方面都取得了改进。 

---
# Adaptively Sampling-Reusing-Mixing Decomposed Gradients to Speed Up Sharpness Aware Minimization 

**Title (ZH)**: 自适应采样-重用-混合分解梯度以加速锋利感知最小化 

**Authors**: Jiaxin Deng, Junbiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03763)  

**Abstract**: Sharpness-Aware Minimization (SAM) improves model generalization but doubles the computational cost of Stochastic Gradient Descent (SGD) by requiring twice the gradient calculations per optimization step. To mitigate this, we propose Adaptively sampling-Reusing-mixing decomposed gradients to significantly accelerate SAM (ARSAM). Concretely, we firstly discover that SAM's gradient can be decomposed into the SGD gradient and the Projection of the Second-order gradient onto the First-order gradient (PSF). Furthermore, we observe that the SGD gradient and PSF dynamically evolve during training, emphasizing the growing role of the PSF to achieve a flat minima. Therefore, ARSAM is proposed to the reused PSF and the timely updated PSF still maintain the model's generalization ability. Extensive experiments show that ARSAM achieves state-of-the-art accuracies comparable to SAM across diverse network architectures. On CIFAR-10/100, ARSAM is comparable to SAM while providing a speedup of about 40\%. Moreover, ARSAM accelerates optimization for the various challenge tasks (\textit{e.g.}, human pose estimation, and model quantization) without sacrificing performance, demonstrating its broad practicality.% The code is publicly accessible at: this https URL. 

**Abstract (ZH)**: 自适应采样-重用-混合分解梯度以显著加速SAM（ARSAM） 

---
# Code4MeV2: a Research-oriented Code-completion Platform 

**Title (ZH)**: Code4MeV2：一个面向研究的代码补全平台 

**Authors**: Roham Koohestani, Parham Bateni, Aydin Ebrahimi, Behdad Etezadi, Kiarash Karimi, Maliheh Izadi  

**Link**: [PDF](https://arxiv.org/pdf/2510.03755)  

**Abstract**: The adoption of AI-powered code completion tools in software development has increased substantially, yet the user interaction data produced by these systems remain proprietary within large corporations. This creates a barrier for the academic community, as researchers must often develop dedicated platforms to conduct studies on human--AI interaction, making reproducible research and large-scale data analysis impractical. In this work, we introduce Code4MeV2, a research-oriented, open-source code completion plugin for JetBrains IDEs, as a solution to this limitation. Code4MeV2 is designed using a client--server architecture and features inline code completion and a context-aware chat assistant. Its core contribution is a modular and transparent data collection framework that gives researchers fine-grained control over telemetry and context gathering. Code4MeV2 achieves industry-comparable performance in terms of code completion, with an average latency of 200~ms. We assess our tool through a combination of an expert evaluation and a user study with eight participants. Feedback from both researchers and daily users highlights its informativeness and usefulness. We invite the community to adopt and contribute to this tool. More information about the tool can be found at this https URL. 

**Abstract (ZH)**: 基于AI的代码完成工具在软件开发中的应用日益增多，但这些系统产生的用户交互数据仍保留在大型企业内部。这为学术界造成了一定障碍，研究人员往往需要开发专门的平台来研究人类-AI交互，这使得可重复研究和大规模数据分析变得 impractical。本文介绍了一个面向研究、开源的代码完成插件 Code4MeV2，作为解决这一限制的解决方案。Code4MeV2 采用客户端-服务器架构，支持嵌入式代码完成和上下文感知聊天助理。其核心贡献在于一个模块化且透明的数据收集框架，赋予研究人员对遥测和上下文收集的精细控制。Code4MeV2 在代码完成性能方面达到行业水平，平均延迟为 200~ms。我们通过专家评估和八名参与者的用户研究对其进行了评估。研究者和普通用户反馈表明其信息量大且实用。我们邀请社区采用并贡献于此工具。更多关于该工具的信息请参见：this https URL。 

---
# HydroFusion-LMF: Semi-Supervised Multi-Network Fusion with Large-Model Adaptation for Long-Term Daily Runoff Forecasting 

**Title (ZH)**: HydroFusion-LMF：大规模模型适应的半监督多网络融合长周期日径流预报 

**Authors**: Qianfei Fan, Jiayu Wei, Peijun Zhu, Wensheng Ye, Meie Fang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03744)  

**Abstract**: Accurate decade-scale daily runoff forecasting in small watersheds is difficult because signals blend drifting trends, multi-scale seasonal cycles, regime shifts, and sparse extremes. Prior deep models (DLinear, TimesNet, PatchTST, TiDE, Nonstationary Transformer, LSTNet, LSTM) usually target single facets and under-utilize unlabeled spans, limiting regime adaptivity. We propose HydroFusion-LMF, a unified framework that (i) performs a learnable trend-seasonal-residual decomposition to reduce non-stationarity, (ii) routes residuals through a compact heterogeneous expert set (linear refinement, frequency kernel, patch Transformer, recurrent memory, dynamically normalized attention), (iii) fuses expert outputs via a hydrologic context-aware gate conditioned on day-of-year phase, antecedent precipitation, local variance, flood indicators, and static basin attributes, and (iv) augments supervision with a semi-supervised multi-task objective (composite MSE/MAE + extreme emphasis + NSE/KGE, masked reconstruction, multi-scale contrastive alignment, augmentation consistency, variance-filtered pseudo-labeling). Optional adapter / LoRA layers inject a frozen foundation time-series encoder efficiently. On a ~10-year daily dataset HydroFusion-LMF attains MSE 1.0128 / MAE 0.5818, improving the strongest baseline (DLinear) by 10.2% / 10.3% and the mean baseline by 24.6% / 17.1%. We observe simultaneous MSE and MAE reductions relative to baselines. The framework balances interpretability (explicit components, sparse gating) with performance, advancing label-efficient hydrologic forecasting under non-stationarity. 

**Abstract (ZH)**: 准确的小流域十年尺度日径流预测因信号混杂漂移趋势、多尺度季节周期、系统转换和稀疏极端事件而具有挑战性。先前的深度模型通常专注于单一特征，未能充分利用无标签数据，限制了系统适应性。我们提出了一种统一框架HydroFusion-LMF，该框架通过（i）进行可学习的趋势-季节-残差分解以降低非平稳性，（ii）通过紧凑的异构专家集合（线性细化、频率核、补丁Transformer、递归记忆、动态规范化注意力）路由残差，（iii）通过一种水文学上下文感知门控融合专家输出，该门控根据日年内相位、前期降水、局部变异、洪水指标以及静态流域属性进行条件处理，和（iv）通过半监督多任务目标增强监督（综合MSE/MAE、极端事件强调、NSE/KGE、掩码重建、多尺度对比对齐、增强一致性、方差筛选伪标签）来平衡可解释性和性能，从而在非平稳条件下促进标签高效水文预报。 

---
# Cost Efficient Fairness Audit Under Partial Feedback 

**Title (ZH)**: 部分反馈下的成本高效公平性审计 

**Authors**: Nirjhar Das, Mohit Sharma, Praharsh Nanavati, Kirankumar Shiragur, Amit Deshpande  

**Link**: [PDF](https://arxiv.org/pdf/2510.03734)  

**Abstract**: We study the problem of auditing the fairness of a given classifier under partial feedback, where true labels are available only for positively classified individuals, (e.g., loan repayment outcomes are observed only for approved applicants). We introduce a novel cost model for acquiring additional labeled data, designed to more accurately reflect real-world costs such as credit assessment, loan processing, and potential defaults. Our goal is to find optimal fairness audit algorithms that are more cost-effective than random exploration and natural baselines.
In our work, we consider two audit settings: a black-box model with no assumptions on the data distribution, and a mixture model, where features and true labels follow a mixture of exponential family distributions. In the black-box setting, we propose a near-optimal auditing algorithm under mild assumptions and show that a natural baseline can be strictly suboptimal. In the mixture model setting, we design a novel algorithm that achieves significantly lower audit cost than the black-box case. Our approach leverages prior work on learning from truncated samples and maximum-a-posteriori oracles, and extends known results on spherical Gaussian mixtures to handle exponential family mixtures, which may be of independent interest. Moreover, our algorithms apply to popular fairness metrics including demographic parity, equal opportunity, and equalized odds. Empirically, we demonstrate strong performance of our algorithms on real-world fair classification datasets like Adult Income and Law School, consistently outperforming natural baselines by around 50% in terms of audit cost. 

**Abstract (ZH)**: 我们在部分反馈情况下审查给定分类器的公平性问题研究：仅对正分类个体提供真实标签（例如，仅对获批申请者观察到贷款偿还结果）。我们引入了一种新的成本模型，用于获取额外的标记数据，旨在更准确地反映现实世界的成本，如信用评估、贷款处理和潜在违约。我们的目标是找到比随机探索和自然基线更具成本效益的最优公平性审查算法。

在我们的工作中，我们考虑了两种审查设置：一个不假设数据分布的黑盒模型，以及一个混合模型，其中特征和真实标签遵循指数家族分布的混合。在黑盒设置下，我们提出了一种在轻微假设下的近最优审查算法，并证明了一种自然基线可能是严格次优的。在混合模型设置下，我们设计了一种新的算法，其审查成本明显低于黑盒情况。我们的方法利用了从截断样本学习和最大后验先验或acles的相关工作，并将已知的球形高斯混合的结果扩展到处理指数家族混合，这可能具有独立的兴趣。此外，我们的算法适用于人口统计学平价、相同机会和平价机会等流行公平性指标。在实验中，我们展示了我们的算法在现实世界的公平分类数据集如Adult Income和Law School中的强大性能，在审查成本上始终优于自然基线约50%。 

---
# LLM-Guided Evolutionary Program Synthesis for Quasi-Monte Carlo Design 

**Title (ZH)**: LLM 引导的演化程序合成在准蒙特卡洛设计中的应用 

**Authors**: Amir Sadikov  

**Link**: [PDF](https://arxiv.org/pdf/2510.03650)  

**Abstract**: Low-discrepancy point sets and digital sequences underpin quasi-Monte Carlo (QMC) methods for high-dimensional integration. We cast two long-standing QMC design problems as program synthesis and solve them with an LLM-guided evolutionary loop that mutates and selects code under task-specific fitness: (i) constructing finite 2D/3D point sets with low star discrepancy, and (ii) choosing Sobol' direction numbers that minimize randomized QMC error on downstream integrands. Our two-phase procedure combines constructive code proposals with iterative numerical refinement. On finite sets, we rediscover known optima in small 2D cases and set new best-known 2D benchmarks for N >= 40, while matching most known 3D optima up to the proven frontier (N <= 8) and reporting improved 3D benchmarks beyond. On digital sequences, evolving Sobol' parameters yields consistent reductions in randomized quasi-Monte Carlo (rQMC) mean-squared error for several 32-dimensional option-pricing tasks relative to widely used Joe--Kuo parameters, while preserving extensibility to any sample size and compatibility with standard randomizations. Taken together, the results demonstrate that LLM-driven evolutionary program synthesis can automate the discovery of high-quality QMC constructions, recovering classical designs where they are optimal and improving them where finite-N structure matters. Data and code are available at this https URL. 

**Abstract (ZH)**: 低散点差异点集和数字序列支撑着高维积分的准蒙特卡洛（QMC）方法。我们将两个长期存在的QMC设计问题视为程序合成，并使用一个由特定任务适应性fitness引导的进化循环来突变和选择代码：（i）构建具有低星散性的有限2D/3D点集；（ii）选择Sobol'方向数以最小化对下游积分函数的随机化QMC误差。我们的两阶段过程结合了构造性代码提案与迭代数值精炼。在有限集合上，我们重新发现了小型2D情况下的已知最优值，并为N >= 40设置了新的最优基准，同时在大多数已知3D最优值（N <= 8）范围内与之匹配，并报告了改进的3D基准。在数字序列上，演化Sobol'参数相对于广泛使用的Joe-Kuo参数，对于多个32维的期权定价任务，提供了随机化准蒙特卡洛（rQMC）均方误差的一致减少，同时保留了对任何样本量的扩展能力和与标准随机化的一致性。综合来看，结果表明，基于大语言模型的进化程序合成可以自动化高质量QMC构造的发现，恢复最优的经典设计，并在有限-N结构重要的情况下进一步改进它们。相关数据和代码可通过以下链接获取。 

---
# Towards Unsupervised Speech Recognition at the Syllable-Level 

**Title (ZH)**: 向 syllable 级无监督语音识别迈进 

**Authors**: Liming Wang, Junrui Ni, Kai-Wei Chang, Saurabhchand Bhati, David Harwath, Mark Hasegawa-Johnson, James R. Glass  

**Link**: [PDF](https://arxiv.org/pdf/2510.03639)  

**Abstract**: Training speech recognizers with unpaired speech and text -- known as unsupervised speech recognition (UASR) -- is a crucial step toward extending ASR to low-resource languages in the long-tail distribution and enabling multimodal learning from non-parallel data. However, existing approaches based on phones often rely on costly resources such as grapheme-to-phoneme converters (G2Ps) and struggle to generalize to languages with ambiguous phoneme boundaries due to training instability. In this paper, we address both challenges by introducing a syllable-level UASR framework based on masked language modeling, which avoids the need for G2P and the instability of GAN-based methods. Our approach achieves up to a 40\% relative reduction in character error rate (CER) on LibriSpeech and generalizes effectively to Mandarin, a language that has remained particularly difficult for prior methods. Code will be released upon acceptance. 

**Abstract (ZH)**: 使用未配对语音和文本训练语音识别器——即无监督语音识别（UASR）——是将ASR扩展到长尾分布中的低资源语言并从非配对数据中实现多模态学习的关键步骤。然而，现有的基于音素的方法往往依赖于昂贵的资源，如字母到音素转换器（G2P），并且难以泛化到具有模糊音素边界的语言，这是由于训练不稳定性。本文通过引入基于掩码语言建模的音节级UASR框架，同时避免了G2P的需求和GAN方法的不稳定性，从而解决了这两个挑战。我们的方法在LibriSpeech上实现了字符错误率（CER）高达40%的相对降低，并且能够有效泛化到先前方法特别难以处理的 Mandarin 语言。接受发表后将公开代码。 

---
# Implicit Models: Expressive Power Scales with Test-Time Compute 

**Title (ZH)**: 隐式模型：表示能力随测试时计算量而变化 

**Authors**: Jialin Liu, Lisang Ding, Stanley Osher, Wotao Yin  

**Link**: [PDF](https://arxiv.org/pdf/2510.03638)  

**Abstract**: Implicit models, an emerging model class, compute outputs by iterating a single parameter block to a fixed point. This architecture realizes an infinite-depth, weight-tied network that trains with constant memory, significantly reducing memory needs for the same level of performance compared to explicit models. While it is empirically known that these compact models can often match or even exceed larger explicit networks by allocating more test-time compute, the underlying mechanism remains poorly understood.
We study this gap through a nonparametric analysis of expressive power. We provide a strict mathematical characterization, showing that a simple and regular implicit operator can, through iteration, progressively express more complex mappings. We prove that for a broad class of implicit models, this process lets the model's expressive power scale with test-time compute, ultimately matching a much richer function class. The theory is validated across three domains: image reconstruction, scientific computing, and operations research, demonstrating that as test-time iterations increase, the complexity of the learned mapping rises, while the solution quality simultaneously improves and stabilizes. 

**Abstract (ZH)**: 隐式模型是一种新兴的模型类，通过迭代单个参数块至固定点来计算输出。这种架构实现了无限深度、权重共享的网络，能够在保持相同性能水平的同时显著减少内存需求。虽然经验上已知这些紧凑模型往往能够匹配甚至超越更大规模的显式网络，但它们背后的工作机制仍不甚理解。

我们通过非参数分析表现能力来研究这一差距。我们提供了一个严格的数学刻画，证明一个简单的规律性隐式操作可以通过迭代逐步表达更复杂的映射。我们证明，在广泛类型的隐式模型中，这一过程能够使模型的表现能力随测试时的计算量扩展，最终匹配更为丰富的函数类。该理论在图像重建、科学计算和运筹学三个领域得到验证，表明随着测试时迭代次数的增加，学习映射的复杂性提高，而解的质量同时改善并稳定。 

---
# Explainable but Vulnerable: Adversarial Attacks on XAI Explanation in Cybersecurity Applications 

**Title (ZH)**: 可解释但易受攻击：网络安全应用中XAI解释的 adversarial攻击 

**Authors**: Maraz Mia, Mir Mehedi A. Pritom  

**Link**: [PDF](https://arxiv.org/pdf/2510.03623)  

**Abstract**: Explainable Artificial Intelligence (XAI) has aided machine learning (ML) researchers with the power of scrutinizing the decisions of the black-box models. XAI methods enable looking deep inside the models' behavior, eventually generating explanations along with a perceived trust and transparency. However, depending on any specific XAI method, the level of trust can vary. It is evident that XAI methods can themselves be a victim of post-adversarial attacks that manipulate the expected outcome from the explanation module. Among such attack tactics, fairwashing explanation (FE), manipulation explanation (ME), and backdoor-enabled manipulation attacks (BD) are the notable ones. In this paper, we try to understand these adversarial attack techniques, tactics, and procedures (TTPs) on explanation alteration and thus the effect on the model's decisions. We have explored a total of six different individual attack procedures on post-hoc explanation methods such as SHAP (SHapley Additive exPlanations), LIME (Local Interpretable Model-agnostic Explanation), and IG (Integrated Gradients), and investigated those adversarial attacks in cybersecurity applications scenarios such as phishing, malware, intrusion, and fraudulent website detection. Our experimental study reveals the actual effectiveness of these attacks, thus providing an urgency for immediate attention to enhance the resiliency of XAI methods and their applications. 

**Abstract (ZH)**: 可解释人工智能(XAI)通过审查黑盒模型的决策帮助了机器学习(ML)研究人员，XAI方法使得深入理解模型行为成为可能，最终生成解释，同时提高对模型的信任度和透明度。然而，依赖于特定的XAI方法，信任度会有所不同。显然，XAI方法自身也容易受到后恶意对抗攻击的操纵，这些攻击会篡改解释模块的预期结果。在这类攻击手段中，公平洗牌解释(FE)、操控解释(ME)和后门启用的操纵攻击(BD)尤为突出。在本文中，我们试图理解这些对抗攻击技术、手法和程序(TTPs)对解释修改及其对模型决策的影响。我们探索了六种不同的个体攻击程序对事后解释方法（如SHAP、LIME和IG）的影响，并在钓鱼、恶意软件、入侵和欺诈性网站检测等网络安全应用场景中研究了这些对抗攻击。我们的实验研究揭示了这些攻击的实际效果，从而强调了即时改进XAI方法及其应用韧性的紧迫性。 

---
# Neural Bayesian Filtering 

**Title (ZH)**: 神经贝叶斯滤波 

**Authors**: Christopher Solinas, Radovan Haluska, David Sychrovsky, Finbarr Timbers, Nolan Bard, Michael Buro, Martin Schmid, Nathan R. Sturtevant, Michael Bowling  

**Link**: [PDF](https://arxiv.org/pdf/2510.03614)  

**Abstract**: We present Neural Bayesian Filtering (NBF), an algorithm for maintaining distributions over hidden states, called beliefs, in partially observable systems. NBF is trained to find a good latent representation of the beliefs induced by a task. It maps beliefs to fixed-length embedding vectors, which condition generative models for sampling. During filtering, particle-style updates compute posteriors in this embedding space using incoming observations and the environment's dynamics. NBF combines the computational efficiency of classical filters with the expressiveness of deep generative models - tracking rapidly shifting, multimodal beliefs while mitigating the risk of particle impoverishment. We validate NBF in state estimation tasks in three partially observable environments. 

**Abstract (ZH)**: 基于神经网络的贝叶斯滤波（NBF）算法：部分可观测系统中隐藏状态分布的维护 

---
# PentestMCP: A Toolkit for Agentic Penetration Testing 

**Title (ZH)**: PentestMCP: 代理渗透测试工具包 

**Authors**: Zachary Ezetta, Wu-chang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2510.03610)  

**Abstract**: Agentic AI is transforming security by automating many tasks being performed manually. While initial agentic approaches employed a monolithic architecture, the Model-Context-Protocol has now enabled a remote-procedure call (RPC) paradigm to agentic applications, allowing for the flexible construction and composition of multi-function agents. This paper describes PentestMCP, a library of MCP server implementations that support agentic penetration testing. By supporting common penetration testing tasks such as network scanning, resource enumeration, service fingerprinting, vulnerability scanning, exploitation, and post-exploitation, PentestMCP allows a developer to customize multi-agent workflows for performing penetration tests. 

**Abstract (ZH)**: 代理型AI正通过自动化许多手动执行的任务来转变安全领域。由于模型-上下文-协议（MCP）模型的支持，代理型应用现在可以采用远程过程调用（RPC）模式，这使得多功能代理的灵活构建和组合成为可能。本文介绍了PentestMCP库，该库支持代理型渗透测试，并通过提供包括网络扫描、资源枚举、服务指纹识别、漏洞扫描、利用和后利用在内的常见渗透测试任务的支持，使开发者能够定制多代理工作流程以执行渗透测试。 

---
# Deep Domain Adaptation for Turbofan Engine Remaining Useful Life Prediction: Methodologies, Evaluation and Future Trends 

**Title (ZH)**: turbofan发动机剩余使用寿命预测的深度域适应方法、评估及未来趋势 

**Authors**: Yucheng Wang, Mohamed Ragab, Yubo Hou, Zhenghua Chen, Min Wu, Xiaoli Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.03604)  

**Abstract**: Remaining Useful Life (RUL) prediction for turbofan engines plays a vital role in predictive maintenance, ensuring operational safety and efficiency in aviation. Although data-driven approaches using machine learning and deep learning have shown potential, they face challenges such as limited data and distribution shifts caused by varying operating conditions. Domain Adaptation (DA) has emerged as a promising solution, enabling knowledge transfer from source domains with abundant data to target domains with scarce data while mitigating distributional shifts. Given the unique properties of turbofan engines, such as complex operating conditions, high-dimensional sensor data, and slower-changing signals, it is essential to conduct a focused review of DA techniques specifically tailored to turbofan engines. To address this need, this paper provides a comprehensive review of DA solutions for turbofan engine RUL prediction, analyzing key methodologies, challenges, and recent advancements. A novel taxonomy tailored to turbofan engines is introduced, organizing approaches into methodology-based (how DA is applied), alignment-based (where distributional shifts occur due to operational variations), and problem-based (why certain adaptations are needed to address specific challenges). This taxonomy offers a multidimensional view that goes beyond traditional classifications by accounting for the distinctive characteristics of turbofan engine data and the standard process of applying DA techniques to this area. Additionally, we evaluate selected DA techniques on turbofan engine datasets, providing practical insights for practitioners and identifying key challenges. Future research directions are identified to guide the development of more effective DA techniques, advancing the state of RUL prediction for turbofan engines. 

**Abstract (ZH)**: 涡扇发动机剩余使用寿命（RUL）预测中的领域适应（DA）研究 

---
# Deep learning the sources of MJO predictability: a spectral view of learned features 

**Title (ZH)**: 深度学习MJO可预测性的来源：学习特征的谱观分析 

**Authors**: Lin Yao, Da Yang, James P.C. Duncan, Ashesh Chattopadhyay, Pedram Hassanzadeh, Wahid Bhimji, Bin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03582)  

**Abstract**: The Madden-Julian oscillation (MJO) is a planetary-scale, intraseasonal tropical rainfall phenomenon crucial for global weather and climate; however, its dynamics and predictability remain poorly understood. Here, we leverage deep learning (DL) to investigate the sources of MJO predictability, motivated by a central difference in MJO theories: which spatial scales are essential for driving the MJO? We first develop a deep convolutional neural network (DCNN) to forecast the MJO indices (RMM and ROMI). Our model predicts RMM and ROMI up to 21 and 33 days, respectively, achieving skills comparable to leading subseasonal-to-seasonal models such as NCEP. To identify the spatial scales most relevant for MJO forecasting, we conduct spectral analysis of the latent feature space and find that large-scale patterns dominate the learned signals. Additional experiments show that models using only large-scale signals as the input have the same skills as those using all the scales, supporting the large-scale view of the MJO. Meanwhile, we find that small-scale signals remain informative: surprisingly, models using only small-scale input can still produce skillful forecasts up to 1-2 weeks ahead. We show that this is achieved by reconstructing the large-scale envelope of the small-scale activities, which aligns with the multi-scale view of the MJO. Altogether, our findings support that large-scale patterns--whether directly included or reconstructed--may be the primary source of MJO predictability. 

**Abstract (ZH)**: Madden-Julian振荡（MJO）的可预报性来源：深学习视角下的大尺度与小尺度作用 

---
# Generalization of Graph Neural Network Models for Distribution Grid Fault Detection 

**Title (ZH)**: 分布式电网故障检测的图神经网络模型通用化 

**Authors**: Burak Karabulut, Carlo Manna, Chris Develder  

**Link**: [PDF](https://arxiv.org/pdf/2510.03571)  

**Abstract**: Fault detection in power distribution grids is critical for ensuring system reliability and preventing costly outages. Moreover, fault detection methodologies should remain robust to evolving grid topologies caused by factors such as reconfigurations, equipment failures, and Distributed Energy Resource (DER) integration. Current data-driven state-of-the-art methods use Recurrent Neural Networks (RNNs) for temporal modeling and Graph Neural Networks (GNNs) for spatial learning, in an RNN+GNN pipeline setting (RGNN in short). Specifically, for power system fault diagnosis, Graph Convolutional Networks (GCNs) have been adopted. Yet, various more advanced GNN architectures have been proposed and adopted in domains outside of power systems. In this paper, we set out to systematically and consistently benchmark various GNN architectures in an RNN+GNN pipeline model. Specifically, to the best of our knowledge, we are the first to (i) propose to use GraphSAGE and Graph Attention (GAT, GATv2) in an RGNN for fault diagnosis, and (ii) provide a comprehensive benchmark against earlier proposed RGNN solutions (RGCN) as well as pure RNN models (especially Gated Recurrent Unit (GRU)), particularly (iii) exploring their generalization potential for deployment in different settings than those used for training them. Our experimental results on the IEEE 123-node distribution network show that RGATv2 has superior generalization capabilities, maintaining high performance with an F1-score reduction of $\sim$12% across different topology settings. In contrast, pure RNN models largely fail, experiencing an F1-score reduction of up to $\sim$60%, while other RGNN variants also exhibit significant performance degradation, i.e., up to $\sim$25% lower F1-scores. 

**Abstract (ZH)**: 基于RNN+GNN架构的故障检测方法系统性评估：以电力分配 grids 为例 

---
# Evaluating OCR performance on food packaging labels in South Africa 

**Title (ZH)**: 评估South Africa食品包装标签上的OCR性能 

**Authors**: Mayimunah Nagayi, Alice Khan, Tamryn Frank, Rina Swart, Clement Nyirenda  

**Link**: [PDF](https://arxiv.org/pdf/2510.03570)  

**Abstract**: This study evaluates four open-source Optical Character Recognition (OCR) systems which are Tesseract, EasyOCR, PaddleOCR, and TrOCR on real world food packaging images. The aim is to assess their ability to extract ingredient lists and nutrition facts panels. Accurate OCR for packaging is important for compliance and nutrition monitoring but is challenging due to multilingual text, dense layouts, varied fonts, glare, and curved surfaces. A dataset of 231 products (1,628 images) was processed by all four models to assess speed and coverage, and a ground truth subset of 113 images (60 products) was created for accuracy evaluation. Metrics include Character Error Rate (CER), Word Error Rate (WER), BLEU, ROUGE-L, F1, coverage, and execution time. On the ground truth subset, Tesseract achieved the lowest CER (0.912) and the highest BLEU (0.245). EasyOCR provided a good balance between accuracy and multilingual support. PaddleOCR achieved near complete coverage but was slower because it ran on CPU only due to GPU incompatibility, and TrOCR produced the weakest results despite GPU acceleration. These results provide a packaging-specific benchmark, establish a baseline, and highlight directions for layout-aware methods and text localization. 

**Abstract (ZH)**: 本研究评估了四种开源光学字符识别（OCR）系统——Tesseract、EasyOCR、PaddleOCR和TrOCR在真实食品包装图像上的性能，旨在评估其提取配料列表和营养成分表的能力。准确的包装OCR对于合规性和营养监测至关重要，但由于多语言文本、密集布局、变体字体、反光和曲面等因素，这一过程具有挑战性。本研究处理了231种产品的1,628张图像，并创建了包含113张图像（60种产品）的基准集，用于准确率评估。评估指标包括字符错误率（CER）、词错误率（WER）、BLEU、ROUGE-L、F1、覆盖率和执行时间。在基准集中，Tesseract实现了最低的CER（0.912）和最高的BLEU（0.245）。EasyOCR在准确性和多语言支持方面提供了良好的平衡。PaddleOCR实现了接近完整的覆盖率，但由于GPU不兼容只能在CPU上运行，因此速度较慢；而TrOCR尽管有GPU加速，但在准确率方面表现最差。这些结果提供了特定于包装的应用基准、建立了基线，并指出了布局感知方法和文本定位的方向。 

---
# Longitudinal Flow Matching for Trajectory Modeling 

**Title (ZH)**: 纵向流匹配用于轨迹建模 

**Authors**: Mohammad Mohaiminul Islam, Thijs P. Kuipers, Sharvaree Vadgama, Coen de Vente, Afsana Khan, Clara I. Sánchez, Erik J. Bekkers  

**Link**: [PDF](https://arxiv.org/pdf/2510.03569)  

**Abstract**: Generative models for sequential data often struggle with sparsely sampled and high-dimensional trajectories, typically reducing the learning of dynamics to pairwise transitions. We propose \textit{Interpolative Multi-Marginal Flow Matching} (IMMFM), a framework that learns continuous stochastic dynamics jointly consistent with multiple observed time points. IMMFM employs a piecewise-quadratic interpolation path as a smooth target for flow matching and jointly optimizes drift and a data-driven diffusion coefficient, supported by a theoretical condition for stable learning. This design captures intrinsic stochasticity, handles irregular sparse sampling, and yields subject-specific trajectories. Experiments on synthetic benchmarks and real-world longitudinal neuroimaging datasets show that IMMFM outperforms existing methods in both forecasting accuracy and further downstream tasks. 

**Abstract (ZH)**: 插值多边际流匹配（IMMFM）：用于多观测时间点的一致连续随机动力学习 

---
# GAS-MIL: Group-Aggregative Selection Multi-Instance Learning for Ensemble of Foundation Models in Digital Pathology Image Analysis 

**Title (ZH)**: GAS-MIL：组聚合选择多实例学习在数字病理图像分析中基础模型集成中的应用 

**Authors**: Peiran Quan, Zifan Gu, Zhuo Zhao, Qin Zhou, Donghan M. Yang, Ruichen Rong, Yang Xie, Guanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.03555)  

**Abstract**: Foundation models (FMs) have transformed computational pathology by providing powerful, general-purpose feature extractors. However, adapting and benchmarking individual FMs for specific diagnostic tasks is often time-consuming and resource-intensive, especially given their scale and diversity. To address this challenge, we introduce Group-Aggregative Selection Multi-Instance Learning (GAS-MIL), a flexible ensemble framework that seamlessly integrates features from multiple FMs, preserving their complementary strengths without requiring manual feature selection or extensive task-specific fine-tuning. Across classification tasks in three cancer datasets-prostate (PANDA), ovarian (UBC-OCEAN), and breast (TCGA-BrCa)-GAS-MIL consistently achieves superior or on-par performance relative to individual FMs and established MIL methods, demonstrating its robustness and generalizability. By enabling efficient integration of heterogeneous FMs, GAS-MIL streamlines model deployment for pathology and provides a scalable foundation for future multimodal and precision oncology applications. 

**Abstract (ZH)**: Group-Aggregative Selection Multi-Instance Learning for Robust and Generalizable Computational Pathology 

---
# Unmasking Puppeteers: Leveraging Biometric Leakage to Disarm Impersonation in AI-based Videoconferencing 

**Title (ZH)**: 揭露傀儡操控者：利用生物特征泄漏解除AI基于视频会议的冒名顶替威胁 

**Authors**: Danial Samadi Vahdati, Tai Duc Nguyen, Ekta Prashnani, Koki Nagano, David Luebke, Orazio Gallo, Matthew Stamm  

**Link**: [PDF](https://arxiv.org/pdf/2510.03548)  

**Abstract**: AI-based talking-head videoconferencing systems reduce bandwidth by sending a compact pose-expression latent and re-synthesizing RGB at the receiver, but this latent can be puppeteered, letting an attacker hijack a victim's likeness in real time. Because every frame is synthetic, deepfake and synthetic video detectors fail outright. To address this security problem, we exploit a key observation: the pose-expression latent inherently contains biometric information of the driving identity. Therefore, we introduce the first biometric leakage defense without ever looking at the reconstructed RGB video: a pose-conditioned, large-margin contrastive encoder that isolates persistent identity cues inside the transmitted latent while cancelling transient pose and expression. A simple cosine test on this disentangled embedding flags illicit identity swaps as the video is rendered. Our experiments on multiple talking-head generation models show that our method consistently outperforms existing puppeteering defenses, operates in real-time, and shows strong generalization to out-of-distribution scenarios. 

**Abstract (ZH)**: 基于AI的头部动画视频会议系统通过发送紧凑的姿态-表情潜空间并在接收端重新合成RGB图像来降低带宽，但该潜空间可以被操纵，让攻击者实时劫持受害者的 likeness。由于每一帧都是合成的，深度假信息和合成视频检测器完全失效。为解决这一安全问题，我们利用一个关键观察：姿态-表情潜空间固有地包含驱动身份的生物识别信息。因此，我们引入了第一个无需查看重建的RGB视频的生物识别泄漏防御方法：一个姿态条件的大边际对比编码器，该编码器在传输的潜空间中隔离持久的身份线索同时取消暂态的姿态和表情。当视频呈现时，简单余弦测试对此分离嵌入进行检查以标识非法身份交换。我们在多个头部动画生成模型上的实验表明，我们的方法在各个方面均优于现有操纵防御方法，能够在实时运行，并且在跨分布场景中表现出强大的泛化能力。 

---
# Reasoning-based Anomaly Detection Framework: A Real-time, Scalable, and Automated Approach to Anomaly Detection Across Domains 

**Title (ZH)**: 基于推理的异常检测框架：一种跨领域实时、可扩展和自动的异常检测方法 

**Authors**: Anupam Panwar, Himadri Pal, Jiali Chen, Kyle Cho, Riddick Jiang, Miao Zhao, Rajiv Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2510.03486)  

**Abstract**: Detecting anomalies in large, distributed systems presents several challenges. The first challenge arises from the sheer volume of data that needs to be processed. Flagging anomalies in a high-throughput environment calls for a careful consideration of both algorithm and system design. The second challenge comes from the heterogeneity of time-series datasets that leverage such a system in production. In practice, anomaly detection systems are rarely deployed for a single use case. Typically, there are several metrics to monitor, often across several domains (e.g. engineering, business and operations). A one-size-fits-all approach rarely works, so these systems need to be fine-tuned for every application - this is often done manually. The third challenge comes from the fact that determining the root-cause of anomalies in such settings is akin to finding a needle in a haystack. Identifying (in real time) a time-series dataset that is associated causally with the anomalous time-series data is a very difficult problem. In this paper, we describe a unified framework that addresses these challenges. Reasoning based Anomaly Detection Framework (RADF) is designed to perform real time anomaly detection on very large datasets. This framework employs a novel technique (mSelect) that automates the process of algorithm selection and hyper-parameter tuning for each use case. Finally, it incorporates a post-detection capability that allows for faster triaging and root-cause determination. Our extensive experiments demonstrate that RADF, powered by mSelect, surpasses state-of-the-art anomaly detection models in AUC performance for 5 out of 9 public benchmarking datasets. RADF achieved an AUC of over 0.85 for 7 out of 9 datasets, a distinction unmatched by any other state-of-the-art model. 

**Abstract (ZH)**: 在大型分布式系统中检测异常存在的挑战及Reasoning based Anomaly Detection Framework (RADF)框架 

---
# The Argument is the Explanation: Structured Argumentation for Trust in Agents 

**Title (ZH)**: 论据即解释：代理信任的结构化论证 

**Authors**: Ege Cakar, Per Ola Kristensson  

**Link**: [PDF](https://arxiv.org/pdf/2510.03442)  

**Abstract**: Humans are black boxes -- we cannot observe their neural processes, yet society functions by evaluating verifiable arguments. AI explainability should follow this principle: stakeholders need verifiable reasoning chains, not mechanistic transparency. We propose using structured argumentation to provide a level of explanation and verification neither interpretability nor LLM-generated explanation is able to offer. Our pipeline achieves state-of-the-art 94.44 macro F1 on the AAEC published train/test split (5.7 points above prior work) and $0.81$ macro F1, $\sim$0.07 above previous published results with comparable data setups, for Argumentative MicroTexts relation classification, converting LLM text into argument graphs and enabling verification at each inferential step. We demonstrate this idea on multi-agent risk assessment using the Structured What-If Technique, where specialized agents collaborate transparently to carry out risk assessment otherwise achieved by humans alone. Using Bipolar Assumption-Based Argumentation, we capture support/attack relationships, thereby enabling automatic hallucination detection via fact nodes attacking arguments. We also provide a verification mechanism that enables iterative refinement through test-time feedback without retraining. For easy deployment, we provide a Docker container for the fine-tuned AMT model, and the rest of the code with the Bipolar ABA Python package on GitHub. 

**Abstract (ZH)**: 人类是黑盒——我们无法观察其神经过程，但社会通过评估可验证的论据而运行。AI可解释性应遵循这一原则：利益相关者需要可验证的推理链，而非机械透明性。我们提议使用结构化论证来提供一种解释和验证水平，这既不是可解释性所能提供的，也不是由LLM生成的解释所能提供的。我们的管道在AAEC发布的训练/测试分割上实现了最先进的94.44宏F1（比先前工作高5.7分点），并在使用可比拟数据集的情况下，实现了0.81的宏F1，比之前发布的结果高约0.07，用于论辩微文本关系分类，将LLM文本转换为论据图，并在每个推理步骤中实现验证。我们在使用结构化what-if技术的多智能体风险评估中演示这一理念，其中专业智能体协作透明地进行风险评估，这是以往仅由人类单独完成的工作。我们使用双极假设基于论证来捕获支持/攻击关系，从而通过事实节点攻击论证实现自动幻觉检测。我们还提供了一种验证机制，通过测试时反馈实现逐步细化，无需重新训练。为了便于部署，我们提供了一个针对精细调校的AMT模型的Docker容器，并在GitHub上提供了使用双极ABA Python包的其余代码。 

---
# Scalable Ground Station Selection for Large LEO Constellations 

**Title (ZH)**: 大规模低轨星座地面站可扩展选择 

**Authors**: Grace Ra Kim, Duncan Eddy, Vedant Srinivas, Mykel J. Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2510.03438)  

**Abstract**: Effective ground station selection is critical for low Earth orbiting (LEO) satellite constellations to minimize operational costs, maximize data downlink volume, and reduce communication gaps between access windows. Traditional ground station selection typically begins by choosing from a fixed set of locations offered by Ground Station-as-a-Service (GSaaS) providers, which helps reduce the problem scope to optimizing locations over existing infrastructure. However, finding a globally optimal solution for stations using existing mixed-integer programming methods quickly becomes intractable at scale, especially when considering multiple providers and large satellite constellations. To address this issue, we introduce a scalable, hierarchical framework that decomposes the global selection problem into single-satellite, short time-window subproblems. Optimal station choices from each subproblem are clustered to identify consistently high-value locations across all decomposed cases. Cluster-level sets are then matched back to the closest GSaaS candidate sites to produce a globally feasible solution. This approach enables scalable coordination while maintaining near-optimal performance. We evaluate our method's performance on synthetic Walker-Star test cases (1-10 satellites, 1-10 stations), achieving solutions within 95% of the global IP optimum for all test cases. Real-world evaluations on Capella Space (5 satellites), ICEYE (40), and Planet's Flock (96) show that while exact IP solutions fail to scale, our framework continues to deliver high-quality site selections. 

**Abstract (ZH)**: 有效的地面站选择对于低地球轨道（LEO）卫星星座降低运营成本、最大化数据下行量并减少访问窗口间的通信间隙至关重要。传统的地面站选择通常从地面站即服务（GSaaS）提供商提供的固定位置集合中选择，从而将问题范围缩小为在现有基础设施上优化位置。然而，当考虑多个提供商和大型卫星星座时，使用现有混合整数规划方法寻找站址的全局最优解很快变得难以处理。为解决这一问题，我们提出了一种可扩展的分层框架，该框架将全局选择问题分解为单卫星、短时间窗口子问题。来自每个子问题的最优站址选择被聚类，以识别所有分解情况下的一致性高价值位置。然后将聚类集与最近的GSaaS候选站点匹配，以生成全局可行解。该方法能够在保持接近最优性能的同时实现可扩展的协调。我们在合成Walker-Star测试案例（1-10颗卫星，1-10个地面站）上评估了方法性能，所有测试案例中的解决方案均接近全局IP最优解的95%。在Capella Space（5颗卫星）、ICEYE（40颗）和Planet的Flock（96颗）的真实世界评估中，虽然精确的IP解决方案无法扩展，但我们的框架继续提供高质量的站点选择。 

---
# Generalized Orders of Magnitude for Scalable, Parallel, High-Dynamic-Range Computation 

**Title (ZH)**: 可扩展、并行、高动态范围计算的一般量级秩序 

**Authors**: Franz A. Heinsen, Leo Kozachkov  

**Link**: [PDF](https://arxiv.org/pdf/2510.03426)  

**Abstract**: Many domains, from deep learning to finance, require compounding real numbers over long sequences, often leading to catastrophic numerical underflow or overflow. We introduce generalized orders of magnitude (GOOMs), a principled extension of traditional orders of magnitude that incorporates floating-point numbers as a special case, and which in practice enables stable computation over significantly larger dynamic ranges of real numbers than previously possible. We implement GOOMs, along with an efficient custom parallel prefix scan, to support native execution on parallel hardware such as GPUs. We demonstrate that our implementation of GOOMs outperforms traditional approaches with three representative experiments, all of which were previously considered impractical or impossible, and now become possible and practical: (1) compounding real matrix products far beyond standard floating-point limits; (2) estimating spectra of Lyapunov exponents in parallel, orders of magnitude faster than with previous methods, applying a novel selective-resetting method to prevent state colinearity; and (3) capturing long-range dependencies in deep recurrent neural networks with non-diagonal recurrent states, computed in parallel via a prefix scan, without requiring any form of stabilization. Our results show that our implementation of GOOMs, combined with efficient parallel scanning, offers a scalable and numerically robust alternative to conventional floating-point numbers for high-dynamic-range applications. 

**Abstract (ZH)**: GOOMs及其在大规模动态范围计算中的应用：超越传统浮点数的稳健计算方法 

---
# Multi-task neural diffusion processes for uncertainty-quantified wind power prediction 

**Title (ZH)**: 多任务神经扩散过程在不确定性量化风电预测中的应用 

**Authors**: Joseph Rawson, Domniki Ladopoulou, Petros Dellaportas  

**Link**: [PDF](https://arxiv.org/pdf/2510.03419)  

**Abstract**: Uncertainty-aware wind power prediction is essential for grid integration and reliable wind farm operation. We apply neural diffusion processes (NDPs)-a recent class of models that learn distributions over functions-and extend them to a multi-task NDP (MT-NDP) framework for wind power prediction. We provide the first empirical evaluation of NDPs in real supervisory control and data acquisition (SCADA) data. We introduce a task encoder within MT-NDPs to capture cross-turbine correlations and enable few-shot adaptation to unseen turbines. The proposed MT-NDP framework outperforms single-task NDPs and GPs in terms of point accuracy and calibration, particularly for wind turbines whose behaviour deviates from the fleet average. In general, NDP-based models deliver calibrated and scalable predictions suitable for operational deployment, offering sharper, yet trustworthy, predictive intervals that can support dispatch and maintenance decisions in modern wind farms. 

**Abstract (ZH)**: 不确定性感知的风功率预测对于电网集成和可靠的风电场运行至关重要。我们应用神经扩散过程（NDPs）——一种最近发展起来的模型类，能够学习函数的分布——并将其扩展到多任务NDP（MT-NDP）框架用于风功率预测。我们在真实的监督控制和数据采集（SCADA）数据上提供了NDPs的首次实证评估。我们引入了多任务NDP中的任务编码器来捕获跨风力发电机的相关性，并实现对未见过的风力发电机的少量样本适应。提出的MT-NDP框架在点准确度和校准方面优于单任务NDP和高斯过程（GPs），特别是在风力发电机行为偏离机组平均值的情况下。总体而言，基于NDP的模型提供了一种校准且可扩展的预测，适合于运行部署，能够提供更锐利但可靠的预测区间，从而支持现代风电场的调度和维护决策。 

---
# Report of the 2025 Workshop on Next-Generation Ecosystems for Scientific Computing: Harnessing Community, Software, and AI for Cross-Disciplinary Team Science 

**Title (ZH)**: 2025代数生态系统研讨会报告：利用社区、软件和AI促进跨学科团队科学 

**Authors**: L.C. McInnes, D. Arnold, P. Balaprakash, M. Bernhardt, B. Cerny, A. Dubey, R. Giles, D.W. Hood, M.A. Leung, V. Lopez-Marrero, P. Messina, O.B. Newton, C. Oehmen, S.M. Wild, J. Willenbring, L. Woodley, T. Baylis, D.E. Bernholdt, C. Camano, J. Cohoon, C. Ferenbaugh, S.M. Fiore, S. Gesing, D. Gomez-Zara, J. Howison, T. Islam, D. Kepczynski, C. Lively, H. Menon, B. Messer, M. Ngom, U. Paliath, M.E. Papka, I. Qualters, E.M. Raybourn, K. Riley, P. Rodriguez, D. Rouson, M. Schwalbe, S.K. Seal, O. Surer, V. Taylor, L. Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03413)  

**Abstract**: This report summarizes insights from the 2025 Workshop on Next-Generation Ecosystems for Scientific Computing: Harnessing Community, Software, and AI for Cross-Disciplinary Team Science, which convened more than 40 experts from national laboratories, academia, industry, and community organizations to chart a path toward more powerful, sustainable, and collaborative scientific software ecosystems. To address urgent challenges at the intersection of high-performance computing (HPC), AI, and scientific software, participants envisioned agile, robust ecosystems built through socio-technical co-design--the intentional integration of social and technical components as interdependent parts of a unified strategy. This approach combines advances in AI, HPC, and software with new models for cross-disciplinary collaboration, training, and workforce development. Key recommendations include building modular, trustworthy AI-enabled scientific software systems; enabling scientific teams to integrate AI systems into their workflows while preserving human creativity, trust, and scientific rigor; and creating innovative training pipelines that keep pace with rapid technological change. Pilot projects were identified as near-term catalysts, with initial priorities focused on hybrid AI/HPC infrastructure, cross-disciplinary collaboration and pedagogy, responsible AI guidelines, and prototyping of public-private partnerships. This report presents a vision of next-generation ecosystems for scientific computing where AI, software, hardware, and human expertise are interwoven to drive discovery, expand access, strengthen the workforce, and accelerate scientific progress. 

**Abstract (ZH)**: This报告总结了2025下一代科学计算生态系统研讨会的见解：利用社区、软件和AI推动跨学科团队科学，该研讨会汇聚了来自国家实验室、学术界、工业界和社区组织的逾40位专家，以规划更强大、更可持续和更具合作性的科学软件生态系统之路。为应对高性能计算（HPC）、AI与科学软件交汇处的紧迫挑战，参与者设想了通过社会-技术共设计构建灵活且稳健的生态系统——故意将社会和技术组件作为统一策略的相互依存部分进行集成。这种方法结合了AI、HPC和软件的最新进展，以及跨学科合作、培训和劳动力发展的新模式。关键建议包括构建模块化、可信赖的AI辅助科学软件系统；使科学团队能够将AI系统集成到其工作流程中，同时保留人类的创造力、信任和科学严谨性；并创建与快速技术变革保持同步的创新培训管道。试点项目被确定为近期催化剂，初期优先事项集中在混合AI/HPC基础设施、跨学科协作与教学、负责任的AI规范以及公共-私营合作伙伴关系原型设计上。本报告提出了下一代科学计算生态系统的愿景，在该生态系统中，AI、软件、硬件和人类专长相互交织，以推动发现、扩大获取、强化劳动力并加速科学进步。 

---
# LegalSim: Multi-Agent Simulation of Legal Systems for Discovering Procedural Exploits 

**Title (ZH)**: LegalSim: 多智能体模拟法律系统的流程性利用发现 

**Authors**: Sanket Badhe  

**Link**: [PDF](https://arxiv.org/pdf/2510.03405)  

**Abstract**: We present LegalSim, a modular multi-agent simulation of adversarial legal proceedings that explores how AI systems can exploit procedural weaknesses in codified rules. Plaintiff and defendant agents choose from a constrained action space (for example, discovery requests, motions, meet-and-confer, sanctions) governed by a JSON rules engine, while a stochastic judge model with calibrated grant rates, cost allocations, and sanction tendencies resolves outcomes. We compare four policies: PPO, a contextual bandit with an LLM, a direct LLM policy, and a hand-crafted heuristic; Instead of optimizing binary case outcomes, agents are trained and evaluated using effective win rate and a composite exploit score that combines opponent-cost inflation, calendar pressure, settlement pressure at low merit, and a rule-compliance margin. Across configurable regimes (e.g., bankruptcy stays, inter partes review, tax procedures) and heterogeneous judges, we observe emergent ``exploit chains'', such as cost-inflating discovery sequences and calendar-pressure tactics that remain procedurally valid yet systemically harmful. Evaluation via cross-play and Bradley-Terry ratings shows, PPO wins more often, the bandit is the most consistently competitive across opponents, the LLM trails them, and the heuristic is weakest. The results are stable in judge settings, and the simulation reveals emergent exploit chains, motivating red-teaming of legal rule systems in addition to model-level testing. 

**Abstract (ZH)**: LegalSim：一个多模块代理模拟的 adversarial 法律程序，探索 AI 系统如何利用成文规则中的程序漏洞 

---
# Cross-Modal Reconstruction Pretraining for Ramp Flow Prediction at Highway Interchanges 

**Title (ZH)**: 高速公路互通处坡度流预测的跨模态重建预训练 

**Authors**: Yongchao Li, Jun Chen, Zhuoxuan Li, Chao Gao, Yang Li, Chu Zhang, Changyin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2510.03381)  

**Abstract**: Interchanges are crucial nodes for vehicle transfers between highways, yet the lack of real-time ramp detectors creates blind spots in traffic prediction. To address this, we propose a Spatio-Temporal Decoupled Autoencoder (STDAE), a two-stage framework that leverages cross-modal reconstruction pretraining. In the first stage, STDAE reconstructs historical ramp flows from mainline data, forcing the model to capture intrinsic spatio-temporal relations. Its decoupled architecture with parallel spatial and temporal autoencoders efficiently extracts heterogeneous features. In the prediction stage, the learned representations are integrated with models such as GWNet to enhance accuracy. Experiments on three real-world interchange datasets show that STDAE-GWNET consistently outperforms thirteen state-of-the-art baselines and achieves performance comparable to models using historical ramp data. This demonstrates its effectiveness in overcoming detector scarcity and its plug-and-play potential for diverse forecasting pipelines. 

**Abstract (ZH)**: 时空解耦自编码器（STDAE）及其在环线流量预测中的应用 

---
# A Robust Clustered Federated Learning Approach for Non-IID Data with Quantity Skew 

**Title (ZH)**: 一种针对数量偏差非一致数据的健壮聚类联邦学习方法 

**Authors**: Michael Ben Ali, Imen Megdiche, André Peninou, Olivier Teste  

**Link**: [PDF](https://arxiv.org/pdf/2510.03380)  

**Abstract**: Federated Learning (FL) is a decentralized paradigm that enables a client-server architecture to collaboratively train a global Artificial Intelligence model without sharing raw data, thereby preserving privacy. A key challenge in FL is Non-IID data. Quantity Skew (QS) is a particular problem of Non-IID, where clients hold highly heterogeneous data volumes. Clustered Federated Learning (CFL) is an emergent variant of FL that presents a promising solution to Non-IID problem. It improves models' performance by grouping clients with similar data distributions into clusters. CFL methods generally fall into two operating strategies. In the first strategy, clients select the cluster that minimizes the local training loss. In the second strategy, the server groups clients based on local model similarities. However, most CFL methods lack systematic evaluation under QS but present significant challenges because of it.  In this paper, we present two main contributions. The first one is an evaluation of state-of-the-art CFL algorithms under various Non-IID settings, applying multiple QS scenarios to assess their robustness. Our second contribution is a novel iterative CFL algorithm, named CORNFLQS, which proposes an optimal coordination between both operating strategies of CFL. Our approach is robust against the different variations of QS settings. We conducted intensive experiments on six image classification datasets, resulting in 270 Non-IID configurations. The results show that CORNFLQS achieves the highest average ranking in both accuracy and clustering quality, as well as strong robustness to QS perturbations. Overall, our approach outperforms actual CFL algorithms. 

**Abstract (ZH)**: 联邦学习（FL）是一种分散式范式，使得客户端-服务器架构能够在不共享原始数据的情况下协作训练全球人工智能模型，从而保护隐私。FL中一个关键挑战是非IID数据。数据量偏差（Quantity Skew，QS）是特定类型的非IID问题，其中客户端持有的数据量高度异质。集群联邦学习（Clustered Federated Learning，CFL）是一种新兴的FL变体，为非IID问题提供了有前途的解决方案。它通过将具有类似数据分布的客户端分组到簇中来提高模型性能。CFL方法通常分为两种操作策略。在第一种策略中，客户端选择能最小化局部训练损失的簇。在第二种策略中，服务器根据局部模型相似性对客户端进行分组。然而，大多数CFL方法缺乏在QS下系统的评估，但这些问题却是它们面临的重大挑战。在本文中，我们提出两大主要贡献。首先是评估在各种非IID设置下的先进CFL算法，并应用多种QS场景来评估其鲁棒性。我们的第二个贡献是提出了一种新的迭代CFL算法，名为CORNFLQS，该算法在两种CFL操作策略之间提出了最优协调。我们的方法在不同版本的QS设置下表现出鲁棒性。我们在六个图像分类数据集上进行了密集的实验，得到了270种非IID配置。结果表明，CORNFLQS在准确率和聚类质量上均获得最高平均排名，并且在QS扰动下表现出强大的鲁棒性。总的来说，我们的方法优于实际的CFL算法。 

---
# Can an AI-Powered Presentation Platform Based On The Game "Just a Minute" Be Used To Improve Students' Public Speaking Skills? 

**Title (ZH)**: 基于“一分钟”游戏的AIpowered演示平台能否提高学生公共演讲技能？ 

**Authors**: Frederic Higham, Tommy Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2510.03379)  

**Abstract**: This study explores the effectiveness of applying AI and gamification into a presentation platform aimed at University students wanting to improve their public speaking skills in their native tongue. Specifically, a platform based on the radio show, Just a Minute (JAM), is explored. In this game, players are challenged to speak fluently on a topic for 60 seconds without repeating themselves, hesitating or deviating from the topic. JAM has proposed benefits such as allowing students to improve their spontaneous speaking skills and reduce their use of speech disfluencies ("um", "uh", etc.).
Previous research has highlighted the difficulties students face when speaking publicly, the main one being anxiety. AI Powered Presentation Platforms (AI-PPPs), where students can speak with an immersive AI audience and receive real-time feedback, have been explored as a method to improve student's speaking skills and confidence. So far they have shown promising results which this study aims to build upon.
A group of students from the University of York are enlisted to evaluate the effectiveness of the JAM platform. They are asked to fill in a questionnaire, play through the game twice and then complete a final questionnaire to discuss their experiences playing the game. Various statistics are gathered during their gameplay such as the number of points they gained and the number of rules they broke. The results showed that students found the game promising and believed that their speaking skills could improve if they played the game for longer. More work will need to be carried out to prove the effectiveness of the game beyond the short term. 

**Abstract (ZH)**: 本研究探讨将AI和游戏化应用于旨在帮助大学学生提高母语公共演讲能力的プレゼンテーションプラットフォーム的有效性。特别是在此研究中，探讨了基于广播节目“Just a Minute”（JAM）的平台。在这个游戏中，玩家被挑战在60秒内流畅地讨论一个话题，不重复、不犹豫且不偏离话题。JAM 提出了诸如允许学生提高即兴演讲能力、减少言语不流畅（如“嗯”、“哦”等）的好处。之前的研究表明，学生在公开演讲时面临的最大困难是焦虑。带有沉浸式AI观众的学生开口演讲平台（AI-PPPs），并能够即时获得反馈的方法，被探索以提高学生的演讲能力和自信心。迄今为止，这些平台已经展示了令人鼓舞的结果，本研究旨在在此基础上进一步提升。来自约克大学的一组学生被招募来评估JAM平台的有效性。他们被要求填写问卷、两次游戏并完成最终问卷来讨论他们对游戏的体验。在游戏过程中收集了包括他们获得的分数和违反的规则次数等各种统计数据。结果显示，学生认为该游戏具有前景，并相信若长时间游戏，他们的演讲能力能够提高。未来还需进一步工作以证明该游戏在短期之外的有效性。 

---
# Lightweight Prompt Engineering for Cognitive Alignment in Educational AI: A OneClickQuiz Case Study 

**Title (ZH)**: 轻量级提示工程在教育AI中的认知对齐：OneClickQuiz案例研究 

**Authors**: Antoun Yaacoub, Zainab Assaghir, Jérôme Da-Rugna  

**Link**: [PDF](https://arxiv.org/pdf/2510.03374)  

**Abstract**: The rapid integration of Artificial Intelligence (AI) into educational technology promises to revolutionize content creation and assessment. However, the quality and pedagogical alignment of AI-generated content remain critical challenges. This paper investigates the impact of lightweight prompt engineering strategies on the cognitive alignment of AI-generated questions within OneClickQuiz, a Moodle plugin leveraging generative AI. We evaluate three prompt variants-a detailed baseline, a simpler version, and a persona-based approach-across Knowledge, Application, and Analysis levels of Bloom's Taxonomy. Utilizing an automated classification model (from prior work) and human review, our findings demonstrate that explicit, detailed prompts are crucial for precise cognitive alignment. While simpler and persona-based prompts yield clear and relevant questions, they frequently misalign with intended Bloom's levels, generating outputs that are either too complex or deviate from the desired cognitive objective. This study underscores the importance of strategic prompt engineering in fostering pedagogically sound AI-driven educational solutions and advises on optimizing AI for quality content generation in learning analytics and smart learning environments. 

**Abstract (ZH)**: 轻量级提示工程策略对OneClickQuiz中生成性人工智能内容认知对齐的影响研究 

---
# Real-time nonlinear inversion of magnetic resonance elastography with operator learning 

**Title (ZH)**: 实时非线性磁共振弹性成像的算子学习反演 

**Authors**: Juampablo E. Heras Rivera, Caitlin M. Neher, Mehmet Kurt  

**Link**: [PDF](https://arxiv.org/pdf/2510.03372)  

**Abstract**: $\textbf{Purpose:}$ To develop and evaluate an operator learning framework for nonlinear inversion (NLI) of brain magnetic resonance elastography (MRE) data, which enables real-time inversion of elastograms with comparable spatial accuracy to NLI.
$\textbf{Materials and Methods:}$ In this retrospective study, 3D MRE data from 61 individuals (mean age, 37.4 years; 34 female) were used for development of the framework. A predictive deep operator learning framework (oNLI) was trained using 10-fold cross-validation, with the complex curl of the measured displacement field as inputs and NLI-derived reference elastograms as outputs. A structural prior mechanism, analogous to Soft Prior Regularization in the MRE literature, was incorporated to improve spatial accuracy. Subject-level evaluation metrics included Pearson's correlation coefficient, absolute relative error, and structural similarity index measure between predicted and reference elastograms across brain regions of different sizes to understand accuracy. Statistical analyses included paired t-tests comparing the proposed oNLI variants to the convolutional neural network baselines.
$\textbf{Results:}$ Whole brain absolute percent error was 8.4 $\pm$ 0.5 ($\mu'$) and 10.0 $\pm$ 0.7 ($\mu''$) for oNLI and 15.8 $\pm$ 0.8 ($\mu'$) and 26.1 $\pm$ 1.1 ($\mu''$) for CNNs. Additionally, oNLI outperformed convolutional architectures as per Pearson's correlation coefficient, $r$, in the whole brain and across all subregions for both the storage modulus and loss modulus (p < 0.05).
$\textbf{Conclusion:}$ The oNLI framework enables real-time MRE inversion (30,000x speedup), outperforming CNN-based approaches and maintaining the fine-grained spatial accuracy achievable with NLI in the brain. 

**Abstract (ZH)**: 目的: 开发并评估一种操作学习框架，用于非线性反演（NLI）的脑磁共振弹性成像（MRE）数据，该框架能够实现实时的弹性图反演，具有与NLI相当的Spatial准确性。 

---
# Distributed Low-Communication Training with Decoupled Momentum Optimization 

**Title (ZH)**: 去中心化低通信量训练与解耦动量优化 

**Authors**: Sasho Nedelkoski, Alexander Acker, Odej Kao, Soeren Becker, Dominik Scheinert  

**Link**: [PDF](https://arxiv.org/pdf/2510.03371)  

**Abstract**: The training of large models demands substantial computational resources, typically available only in data centers with high-bandwidth interconnects. However, reducing the reliance on high-bandwidth interconnects between nodes enables the use of distributed compute resources as an alternative to centralized data center training. Building on recent advances in distributed model training, we propose an approach that further reduces communication by combining infrequent synchronizations across distributed model replicas with gradient momentum compression. In particular, we treat the optimizer momentum as a signal and decompose the Nesterov momentum into high- and low-frequency components via the discrete cosine transform (DCT). Only the high-frequency components are synchronized across model replicas every $H$ steps. Empirically, our method achieves up to a $16\times$ reduction in communication compared to the baseline DiLoCo, and it generalizes across architectures, including transformer-based language models and convolutional neural networks for images. Overall, this work advances the feasibility of training large models on distributed nodes with low-bandwidth interconnects. 

**Abstract (ZH)**: 大规模模型训练需要大量的计算资源，通常只有在具备高带宽互联的数据中心中才能获得。然而，减少节点间对高带宽互联的依赖性使得分布式计算资源可以作为中心化数据中心训练的替代方案。基于最近在分布式模型训练方面的进展，我们提出了一种进一步减少通信的方法，该方法将稀疏同步与梯度动量压缩相结合。具体而言，我们将优化器动量视为信号，并通过离散余弦变换（DCT）将Nesterov动量分解为高频和低频分量。只有高频分量在每$H$步在模型副本之间进行同步。实验结果表明，与基线DiLoCo方法相比，我们的方法可以实现最多16倍的通信减少，并且可以在包括基于变换器的语言模型和用于图像的卷积神经网络在内的多种架构上泛化。总体而言，这项工作促进了在具备低带宽互联的分布式节点上训练大规模模型的可行性。 

---
# InstructPLM-mu: 1-Hour Fine-Tuning of ESM2 Beats ESM3 in Protein Mutation Predictions 

**Title (ZH)**: InstructPLM-mu：1小时微调的ESM2在蛋白质突变预测中优于ESM3 

**Authors**: Junde Xu, Yapin Shi, Lijun Lang, Taoyong Cui, Zhiming Zhang, Guangyong Chen, Jiezhong Qiu, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2510.03370)  

**Abstract**: Multimodal protein language models deliver strong performance on mutation-effect prediction, but training such models from scratch demands substantial computational resources. In this paper, we propose a fine-tuning framework called InstructPLM-mu and try to answer a question: \textit{Can multimodal fine-tuning of a pretrained, sequence-only protein language model match the performance of models trained end-to-end? } Surprisingly, our experiments show that fine-tuning ESM2 with structural inputs can reach performance comparable to ESM3. To understand how this is achieved, we systematically compare three different feature-fusion designs and fine-tuning recipes. Our results reveal that both the fusion method and the tuning strategy strongly affect final accuracy, indicating that the fine-tuning process is not trivial. We hope this work offers practical guidance for injecting structure into pretrained protein language models and motivates further research on better fusion mechanisms and fine-tuning protocols. 

**Abstract (ZH)**: 多模态蛋白质语言模型在突变效应预测任务中表现出strong性能，但从头训练此类模型需要大量计算资源。本文提出了一种名为InstructPLM-mu的微调框架，并试图回答一个问题：\textit{仅序列的预训练蛋白质语言模型的多模态微调能否与端到端训练的模型性能相当？} 让人惊讶的是，我们的实验显示，使用结构输入微调ESM2可以达到与ESM3相当的性能。为了理解这是如何实现的，我们系统地比较了三种不同的特征融合设计和微调方案。我们的结果揭示了融合方法和微调策略对最终准确度的影响，表明微调过程并非易事。希望本工作能为向预训练蛋白质语言模型注入结构提供实用指导，并激发进一步研究更好的融合机制和微调协议。 

---
# TriQuest:An AI Copilot-Powered Platform for Interdisciplinary Curriculum Design 

**Title (ZH)**: TriQuest:由AI副驾赋能的跨学科课程设计平台 

**Authors**: Huazhen Wang, Huimin Yang, Hainbin Lin, Yan Dong, Lili Chen, Liangliang Xia, Wenwen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03369)  

**Abstract**: Interdisciplinary teaching is a cornerstone of modern curriculum reform, but its implementation is hindered by challenges in knowledge integration and time-consuming lesson planning. Existing tools often lack the required pedagogical and domain-specific this http URL introduce TriQuest, an AI-copilot platform designed to solve these problems. TriQuest uses large language models and knowledge graphs via an intuitive GUI to help teachers efficiently generate high-quality interdisciplinary lesson plans. Its core features include intelligent knowledge integration from various disciplines and a human-computer collaborative review process to ensure quality and this http URL a study with 43 teachers, TriQuest increased curriculum design efficiency by an average of 75% and improved lesson plan quality scores by 41%. It also significantly lowered design barriers and cognitive load. Our work presents a new paradigm for empowering teacher professional development with intelligent technologies. 

**Abstract (ZH)**: 跨学科教学是现代课程改革的基石，但其实施受到学科知识整合和耗时的lesson planning的挑战。现有工具往往缺乏必要的教学和领域特定功能。为此，我们介绍了TriQuest，一个AI copilot平台，旨在解决这些问题。TriQuest通过直观的GUI利用大语言模型和知识图谱，帮助教师高效生成高质量的跨学科lesson plan。其核心功能包括跨学科智能知识整合和人机协作评审流程，以确保质量和教学设计一致性。在一项涉及43名教师的研究所示，TriQuest将课程设计效率平均提高了75%，提高了lesson plan质量评分41%，并显著降低了设计障碍和认知负荷。我们的工作展示了智能技术赋能教师专业发展的新范式。 

---
# An Adaptive Responsible AI Governance Framework for Decentralized Organizations 

**Title (ZH)**: 面向去中心化组织的自适应负责任人工智能治理框架 

**Authors**: Kiana Jafari Meimandi, Anka Reuel, Gabriela Aranguiz-Dias, Hatim Rahama, Ala-Eddine Ayadi, Xavier Boullier, Jérémy Verdo, Louis Montanie, Mykel Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2510.03368)  

**Abstract**: This paper examines the assessment challenges of Responsible AI (RAI) governance efforts in globally decentralized organizations through a case study collaboration between a leading research university and a multinational enterprise. While there are many proposed frameworks for RAI, their application in complex organizational settings with distributed decision-making authority remains underexplored. Our RAI assessment, conducted across multiple business units and AI use cases, reveals four key patterns that shape RAI implementation: (1) complex interplay between group-level guidance and local interpretation, (2) challenges translating abstract principles into operational practices, (3) regional and functional variation in implementation approaches, and (4) inconsistent accountability in risk oversight. Based on these findings, we propose an Adaptive RAI Governance (ARGO) Framework that balances central coordination with local autonomy through three interdependent layers: shared foundation standards, central advisory resources, and contextual local implementation. We contribute insights from academic-industry collaboration for RAI assessments, highlighting the importance of modular governance approaches that accommodate organizational complexity while maintaining alignment with responsible AI principles. These lessons offer practical guidance for organizations navigating the transition from RAI principles to operational practice within decentralized structures. 

**Abstract (ZH)**: 本研究通过一所领先的研究大学与一家跨国企业之间的案例研究合作，探讨了全球分散组织中负责任人工智能（RAI）治理努力的评估挑战。尽管已经提出了许多RAI框架，但它们在拥有分布式决策权的复杂组织环境中的应用仍然未得到充分探索。我们的RAI评估涵盖了多个业务部门和人工智能应用场景，揭示了四个关键模式，这些模式影响了RAI的实施：（1）小组指导与本地解释之间的复杂互动，（2）将抽象原则转化为操作实践的挑战，（3）实施方法在区域和功能上的差异，以及（4）风险监督中的不一致问责制。基于这些发现，我们提出了一种平衡中央协调与地方自主性的适应型RAI治理（ARGO）框架，该框架由三个相互依存的层面组成：共同的基础标准、中央咨询资源和情境下的地方实施。我们通过学术-产业合作为RAI评估提供了见解，强调了在组织复杂性与负责任人工智能原则之间保持一致性的模块化治理方法的重要性。这些教训为组织在分散结构中从RAI原则过渡到操作实践提供了实用指导。 

---
# Diffusion-Based, Data-Assimilation-Enabled Super-Resolution of Hub-height Winds 

**Title (ZH)**: 基于扩散的、数据同化的高空风超分辨率重建 

**Authors**: Xiaolong Ma, Xu Dong, Ashley Tarrant, Lei Yang, Rao Kotamarthi, Jiali Wang, Feng Yan, Rajkumar Kettimuthu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03364)  

**Abstract**: High-quality observations of hub-height winds are valuable but sparse in space and time. Simulations are widely available on regular grids but are generally biased and too coarse to inform wind-farm siting or to assess extreme-weather-related risks (e.g., gusts) at infrastructure scales. To fully utilize both data types for generating high-quality, high-resolution hub-height wind speeds (tens to ~100m above ground), this study introduces WindSR, a diffusion model with data assimilation for super-resolution downscaling of hub-height winds. WindSR integrates sparse observational data with simulation fields during downscaling using state-of-the-art diffusion models. A dynamic-radius blending method is introduced to merge observations with simulations, providing conditioning for the diffusion process. Terrain information is incorporated during both training and inference to account for its role as a key driver of winds. Evaluated against convolutional-neural-network and generative-adversarial-network baselines, WindSR outperforms them in both downscaling efficiency and accuracy. Our data assimilation reduces WindSR's model bias by approximately 20% relative to independent observations. 

**Abstract (ZH)**: 高空间与时间分辨率的轮毂高度风观测数据稀缺，而模拟数据虽广泛可用但通常具有偏差且分辨率过低，不足以用于风电场选址或评估与极端天气（如阵风）相关的基础设施风险。为了充分利用这两种数据类型生成高分辨率（十米至百米以上）轮毂高度风速，本文 introduces WindSR，一种结合数据同化的扩散模型，用于超分辨率下-scaling 轮毂高度风速。WindSR 使用最先进的扩散模型在下-scaling 过程中整合稀疏观测数据和模拟场。引入了一种动态半径混合方法，将观测数据与模拟数据合并，为扩散过程提供条件。训练和推断过程中均加入了地形信息，以反映地形对风速的关键驱动作用。与卷积神经网络和生成对抗网络基线相比，WindSR 在下-scaling 效率和准确性方面表现出更优性能。我们的数据同化将 WindSR 模型偏差相对于独立观测数据降低了约 20%。 

---
# Unified Unsupervised Anomaly Detection via Matching Cost Filtering 

**Title (ZH)**: 统一的无监督异常检测方法基于匹配成本过滤 

**Authors**: Zhe Zhang, Mingxiu Cai, Gaochang Wu, Jing Zhang, Lingqiao Liu, Dacheng Tao, Tianyou Chai, Xiatian Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03363)  

**Abstract**: Unsupervised anomaly detection (UAD) aims to identify image- and pixel-level anomalies using only normal training data, with wide applications such as industrial inspection and medical analysis, where anomalies are scarce due to privacy concerns and cold-start constraints. Existing methods, whether reconstruction-based (restoring normal counterparts) or embedding-based (pretrained representations), fundamentally conduct image- or feature-level matching to generate anomaly maps. Nonetheless, matching noise has been largely overlooked, limiting their detection ability. Beyond earlier focus on unimodal RGB-based UAD, recent advances expand to multimodal scenarios, e.g., RGB--3D and RGB--Text, enabled by point cloud sensing and vision--language models. Despite shared challenges, these lines remain largely isolated, hindering a comprehensive understanding and knowledge transfer. In this paper, we advocate unified UAD for both unimodal and multimodal settings in the matching perspective. Under this insight, we present Unified Cost Filtering (UCF), a generic post-hoc refinement framework for refining anomaly cost volume of any UAD model. The cost volume is constructed by matching a test sample against normal samples from the same or different modalities, followed by a learnable filtering module with multi-layer attention guidance from the test sample, mitigating matching noise and highlighting subtle anomalies. Comprehensive experiments on 22 diverse benchmarks demonstrate the efficacy of UCF in enhancing a variety of UAD methods, consistently achieving new state-of-the-art results in both unimodal (RGB) and multimodal (RGB--3D, RGB--Text) UAD scenarios. Code and models will be released at this https URL. 

**Abstract (ZH)**: 无监督异常检测（UAD）旨在仅使用正常训练数据来识别图像级和像素级的异常，广泛应用于工业检测和医疗分析等领域，其中由于隐私和冷启动的限制，异常往往是稀缺的。现有方法无论基于重构（恢复正常样本）还是嵌入（预训练表示），本质上都是执行图像级或特征级匹配以生成异常图。然而，匹配噪声已被大量忽视，限制了它们的检测能力。超越早期基于单一RGB的数据的无监督异常检测，近期进展扩展到了多模态场景，如RGB-3D和RGB-Text，得益于点云传感和跨模态模型。尽管存在共同的挑战，但这些领域仍然相对孤立，阻碍了对整体理解以及知识迁移的认知。本文从匹配视角提倡统一的无监督异常检测模型，基于此见解，我们提出了统一成本过滤（UCF），这是一种通用的后处理精炼框架，用于精炼任何UAD模型的成本体素。成本体素通过将测试样本与相同或不同模态的正常样本进行匹配构建，并通过测试样本的多层注意力引导的学习过滤模块进行精炼，以减轻匹配噪声并突出细微异常。在22个多样化的基准测试上进行全面实验表明，UCF能够增强多种无监督异常检测方法的有效性，在单模态（RGB）和多模态（RGB-3D，RGB-Text）无监督异常检测场景中均能取得新的最先进的结果。代码和模型将于此网址发布：this https URL。 

---
# Provenance Networks: End-to-End Exemplar-Based Explainability 

**Title (ZH)**: 来源网络：端到端示例为基础的可解释性 

**Authors**: Ali Kayyam, Anusha Madan Gopal, M. Anthony Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2510.03361)  

**Abstract**: We introduce provenance networks, a novel class of neural models designed to provide end-to-end, training-data-driven explainability. Unlike conventional post-hoc methods, provenance networks learn to link each prediction directly to its supporting training examples as part of the model's normal operation, embedding interpretability into the architecture itself. Conceptually, the model operates similarly to a learned KNN, where each output is justified by concrete exemplars weighted by relevance in the feature space. This approach facilitates systematic investigations of the trade-off between memorization and generalization, enables verification of whether a given input was included in the training set, aids in the detection of mislabeled or anomalous data points, enhances resilience to input perturbations, and supports the identification of similar inputs contributing to the generation of a new data point. By jointly optimizing the primary task and the explainability objective, provenance networks offer insights into model behavior that traditional deep networks cannot provide. While the model introduces additional computational cost and currently scales to moderately sized datasets, it provides a complementary approach to existing explainability techniques. In particular, it addresses critical challenges in modern deep learning, including model opaqueness, hallucination, and the assignment of credit to data contributors, thereby improving transparency, robustness, and trustworthiness in neural models. 

**Abstract (ZH)**: 基于证据的神经网络：一种新型端到端训练数据驱动的可解释性模型 

---
# Physics-informed Neural-operator Predictive Control for Drag Reduction in Turbulent Flows 

**Title (ZH)**: 基于物理的神经算子预测控制以降低湍流流动的drag损失 

**Authors**: Zelin Zhao, Zongyi Li, Kimia Hassibi, Kamyar Azizzadenesheli, Junchi Yan, H. Jane Bae, Di Zhou, Anima Anandkumar  

**Link**: [PDF](https://arxiv.org/pdf/2510.03360)  

**Abstract**: Assessing turbulence control effects for wall friction numerically is a significant challenge since it requires expensive simulations of turbulent fluid dynamics. We instead propose an efficient deep reinforcement learning (RL) framework for modeling and control of turbulent flows. It is model-based RL for predictive control (PC), where both the policy and the observer models for turbulence control are learned jointly using Physics Informed Neural Operators (PINO), which are discretization invariant and can capture fine scales in turbulent flows accurately. Our PINO-PC outperforms prior model-free reinforcement learning methods in various challenging scenarios where the flows are of high Reynolds numbers and unseen, i.e., not provided during model training. We find that PINO-PC achieves a drag reduction of 39.0\% under a bulk-velocity Reynolds number of 15,000, outperforming previous fluid control methods by more than 32\%. 

**Abstract (ZH)**: 基于物理学约束神经运算符的高效深度强化学习湍流控制方法 

---
# Understanding Transformers for Time Series: Rank Structure, Flow-of-ranks, and Compressibility 

**Title (ZH)**: 理解时间序列中的变换器：层级结构、秩流和压缩性 

**Authors**: Annan Yu, Danielle C. Maddix, Boran Han, Xiyuan Zhang, Abdul Fatir Ansari, Oleksandr Shchur, Christos Faloutsos, Andrew Gordon Wilson, Michael W. Mahoney, Yuyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03358)  

**Abstract**: Transformers are widely used across data modalities, and yet the principles distilled from text models often transfer imperfectly to models trained to other modalities. In this paper, we analyze Transformers through the lens of rank structure. Our focus is on the time series setting, where the structural properties of the data differ remarkably from those of text or vision. We show that time-series embeddings, unlike text or vision, exhibit sharply decaying singular value spectra: small patch sizes and smooth continuous mappings concentrate the data into low-rank subspaces. From this, we prove that the associated $Q/K/V$ projections admit accurate low-rank approximations, and that attention layers become compressible in proportion to the decay of the embedding spectrum. We introduce the concept of flow-of-ranks, a phenomenon by which nonlinear mixing across depth inflates the rank, explaining why early layers are most amenable to compression and why ranks grow with depth. Guided by these theoretical and empirical results, we use these insights to compress Chronos, a large time series foundation model, achieving a reduction of $65\%$ in inference time and $81\%$ in memory, without loss of accuracy. Our findings provide principled guidance for allocating width, depth, and heads in time series foundation models, and for exploiting their inherent compressibility. 

**Abstract (ZH)**: 基于秩结构视角分析变压器：时间序列模型中的压缩原理与实践 

---
# Pilot selection in the era of Virtual reality: algorithms for accurate and interpretable machine learning models 

**Title (ZH)**: 虚拟现实时代的人选试点：准确可解释的机器学习算法 

**Authors**: Luoma Ke, Guangpeng Zhang, Jibo He, Yajing Li, Yan Li, Xufeng Liu, Peng Fang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03345)  

**Abstract**: With the rapid growth of the aviation industry, there is a need for a large number of flight crew. How to select the right pilots in a cost-efficient manner has become an important research question. In the current study, twenty-three pilots were recruited from China Eastern Airlines, and 23 novices were from the community of Tsinghua University. A novel approach incorporating machine learning and virtual reality technology was applied to distinguish features between these participants with different flight skills. Results indicate that SVM with the MIC feature selection method consistently achieved the highest prediction performance on all metrics with an Accuracy of 0.93, an AUC of 0.96, and an F1 of 0.93, which outperforms four other classifier algorithms and two other feature selection methods. From the perspective of feature selection methods, the MIC method can select features with a nonlinear relationship to sampling labels, instead of a simple filter-out. Our new implementation of the SVM + MIC algorithm outperforms all existing pilot selection algorithms and perhaps provides the first implementation based on eye tracking and flight dynamics data. This study's VR simulation platforms and algorithms can be used for pilot selection and training. 

**Abstract (ZH)**: 随着航空业的迅速增长，飞行员的需求量大大增加。如何以成本高效的方式选拔合适的飞行员已成为一个重要研究问题。在本研究中，从东方航空公司招募了23名飞行员，另从清华大学社区招募了23名新手。本研究应用了结合机器学习和虚拟现实技术的新型方法，以区分技能不同的参与者特征。结果显示，使用MIC特征选择方法的SVM在所有指标中表现最佳，准确率（Accuracy）为0.93，AUC为0.96，F1值为0.93，优于其他四种分类算法和两种特征选择方法。从特征选择方法的角度来看，MIC方法可以筛选出与抽样标签之间存在非线性关系的特征，而不仅仅是简单的过滤。我们的SVM + MIC算法的新实现超越了所有现有的飞行员选拔算法，并可能基于眼动追踪和飞行动力学数据提供首个实现。本研究的VR仿真平台和算法可用于飞行员选拔和训练。 

---
# Defining a Strategic Action Plan for AI in Higher Education 

**Title (ZH)**: 为高等教育制定人工智能战略行动方案 

**Authors**: Nikolaos Avouris  

**Link**: [PDF](https://arxiv.org/pdf/2510.03343)  

**Abstract**: This paper discusses key challenges of Artificial Intelligence in Education, with main focus on higher education institutions. We start with reviewing normative actions of international organizations and concerns expressed about the current technical landscape. Then we proceed with proposing a framework that comprises five key dimensions relating to the main challenges relating to AI in higher education institutions, followed by five key strategic actions that the main stakeholders need to take in order to address the current developments. We map these actions to the main stakeholders of higher education and propose a deployment plan. This defines a framework along the dimensions: Challenges, Actions, Stakeholders, Deployment CASD. Examples of AI specific actions at the institutional and individual course level are also provided and discussed. 

**Abstract (ZH)**: 本文讨论了人工智能在教育中的关键挑战，重点关注高等教育机构。我们从审查国际组织的规范性行动和对当前技术景观的关切开始，随后提出了一个框架，该框架包含五个关键维度，涉及人工智能在高等教育机构中面临的主要挑战，并提出了五个关键战略行动，以便主要利益相关者能够应对当前的发展。我们将这些行动映射到高等教育的主要利益相关者，并提出了部署计划。该框架沿用了挑战、行动、利益相关者、部署（CASD）四个维度。还提供了并讨论了机构和课程层面的特定人工智能行动示例。 

---
# Learning Pareto-Optimal Pandemic Intervention Policies with MORL 

**Title (ZH)**: 基于MORL学习 Pareto 最优 pandemic 干预策略 

**Authors**: Marian Chen, Miri Zilka  

**Link**: [PDF](https://arxiv.org/pdf/2510.03340)  

**Abstract**: The COVID-19 pandemic underscored a critical need for intervention strategies that balance disease containment with socioeconomic stability. We approach this challenge by designing a framework for modeling and evaluating disease-spread prevention strategies. Our framework leverages multi-objective reinforcement learning (MORL) - a formulation necessitated by competing objectives - combined with a new stochastic differential equation (SDE) pandemic simulator, calibrated and validated against global COVID-19 data. Our simulator reproduces national-scale pandemic dynamics with orders of magnitude higher fidelity than other models commonly used in reinforcement learning (RL) approaches to pandemic intervention. Training a Pareto-Conditioned Network (PCN) agent on this simulator, we illustrate the direct policy trade-offs between epidemiological control and economic stability for COVID-19. Furthermore, we demonstrate the framework's generality by extending it to pathogens with different epidemiological profiles, such as polio and influenza, and show how these profiles lead the agent to discover fundamentally different intervention policies. To ground our work in contemporary policymaking challenges, we apply the model to measles outbreaks, quantifying how a modest 5% drop in vaccination coverage necessitates significantly more stringent and costly interventions to curb disease spread. This work provides a robust and adaptable framework to support transparent, evidence-based policymaking for mitigating public health crises. 

**Abstract (ZH)**: COVID-19疫情凸显了在疾病控制与社会经济稳定之间寻求平衡的干预策略的迫切需求。我们通过设计一个疾病传播预防策略建模与评估框架来应对这一挑战。该框架利用多目标强化学习（MORL）——一种由相互竞争的目标所要求的表述——结合一种新的随机微分方程（SDE）疫情模拟器，并根据全球COVID-19数据进行了校准和验证。我们的模拟器在准确再现国家级规模的疫情动态方面比其他常用于强化学习（RL）方法中的模型高出数个数量级。通过在此模拟器上训练帕累托条件网络（PCN）代理，我们展示了COVID-19在流行病学控制与经济稳定之间的直接政策权衡。此外，我们通过将该框架扩展到具有不同流行病学特征的病原体——如脊髓灰质炎和流感——展示了其广泛的适用性，并展示了这些特征如何引导代理发现根本不同的干预策略。为了将我们的工作与当前的政策制定挑战相结合，我们应用该模型研究麻疹暴发，量化了5%疫苗接种覆盖率下降需要采取更为严格和昂贵的措施来遏制疾病传播的程度。本研究提供了支持公开透明、基于证据的公共卫生危机缓解政策制定的稳健且灵活的框架。 

---
# Linguistic and Audio Embedding-Based Machine Learning for Alzheimer's Dementia and Mild Cognitive Impairment Detection: Insights from the PROCESS Challenge 

**Title (ZH)**: 基于语言和音频嵌入的机器学习方法在阿尔茨海默病和轻度认知 impairment 检测中的研究：PROCESS 挑战赛启示 

**Authors**: Adharsha Sam Edwin Sam Devahi, Sohail Singh Sangha, Prachee Priyadarshinee, Jithin Thilakan, Ivan Fu Xing Tan, Christopher Johann Clarke, Sou Ka Lon, Balamurali B T, Yow Wei Quin, Chen Jer-Ming  

**Link**: [PDF](https://arxiv.org/pdf/2510.03336)  

**Abstract**: Early detection of Alzheimer's Dementia (AD) and Mild Cognitive Impairment (MCI) is critical for timely intervention, yet current diagnostic approaches remain resource-intensive and invasive. Speech, encompassing both acoustic and linguistic dimensions, offers a promising non-invasive biomarker for cognitive decline. In this study, we present a machine learning framework for the PROCESS Challenge, leveraging both audio embeddings and linguistic features derived from spontaneous speech recordings. Audio representations were extracted using Whisper embeddings from the Cookie Theft description task, while linguistic features-spanning pronoun usage, syntactic complexity, filler words, and clause structure-were obtained from transcriptions across Semantic Fluency, Phonemic Fluency, and Cookie Theft picture description. Classification models aimed to distinguish between Healthy Controls (HC), MCI, and AD participants, while regression models predicted Mini-Mental State Examination (MMSE) scores. Results demonstrated that voted ensemble models trained on concatenated linguistic features achieved the best classification performance (F1 = 0.497), while Whisper embedding-based ensemble regressors yielded the lowest MMSE prediction error (RMSE = 2.843). Comparative evaluation within the PROCESS Challenge placed our models among the top submissions in regression task, and mid-range for classification, highlighting the complementary strengths of linguistic and audio embeddings. These findings reinforce the potential of multimodal speech-based approaches for scalable, non-invasive cognitive assessment and underline the importance of integrating task-specific linguistic and acoustic markers in dementia detection. 

**Abstract (ZH)**: 早期检测阿尔茨海默病痴呆（AD）和轻度认知障碍（MCI）对于及时干预至关重要，但当前的诊断方法仍资源密集且侵入性强。语言，涵盖声学和语义维度，为认知衰退提供了有前景的非侵入性生物标志物。在本研究中，我们提出了一种用于PROCESS挑战的比赛框架，利用来自自发性言语录音的声学嵌入和语言特征。声学表示使用Whisper嵌入从Cookie Theft描述任务中提取，而语言特征包括代词使用、句法复杂性、填充词和从Semantical Fluency、Phonemic Fluency和Cookie Theft图片描述中获得的从句结构。分类模型旨在区分健康对照组（HC）、MCI和AD参与者，而回归模型预测简易精神状态检查（MMSE）分数。结果表明，在拼接语言特征上训练的投票集成模型实现了最佳分类性能（F1 = 0.497），而基于Whisper嵌入的集成回归器在MMSE分数预测中的均方根误差最低（RMSE = 2.843）。在PROCESS挑战中的比较评估中，我们的模型在回归任务中排名靠前，在分类任务中处于中游水平，突显了语言和声学嵌入的互补优势。这些发现强化了多模态言语基方法在可扩展、非侵入性认知评估中的潜力，并强调了在痴呆症检测中整合任务特定语言和声学标记的重要性。 

---
# Intelligent Healthcare Ecosystems: Optimizing the Iron Triangle of Healthcare (Access, Cost, Quality) 

**Title (ZH)**: 智能医疗生态系统：优化医疗的铁三角（可及性、成本、质量） 

**Authors**: Vivek Acharya  

**Link**: [PDF](https://arxiv.org/pdf/2510.03331)  

**Abstract**: The United States spends nearly 17% of GDP on healthcare yet continues to face uneven access and outcomes. This well-known trade-off among cost, quality, and access - the "iron triangle" - motivates a system-level redesign. This paper proposes an Intelligent Healthcare Ecosystem (iHE): an integrated, data-driven framework that uses generative AI and large language models, federated learning, interoperability standards (FHIR, TEFCA), and digital twins to improve access and quality while lowering cost. We review historical spending trends, waste, and international comparisons; introduce a value equation that jointly optimizes access, quality, and cost; and synthesize evidence on the enabling technologies and operating model for iHE. Methods follow a narrative review of recent literature and policy reports. Results outline core components (AI decision support, interoperability, telehealth, automation) and show how iHE can reduce waste, personalize care, and support value-based payment while addressing privacy, bias, and adoption challenges. We argue that a coordinated iHE can bend - if not break - the iron triangle, moving the system toward care that is more accessible, affordable, and high quality. 

**Abstract (ZH)**: 美国在医疗卫生上的支出占GDP的近17%，但仍面临不均衡的可及性和结果问题。这种广为人知的成本、质量和可及性之间的权衡——“铁三角”——促使我们需要对整个系统进行重新设计。本文提出了一种智能健康生态系统（iHE）：一个集成的数据驱动框架，利用生成性AI和大规模语言模型、联邦学习、互操作性标准（FHIR、TEFCA）以及数字孪生技术，以改善可及性和质量并降低成本。我们回顾了历史上的支出趋势、浪费情况以及国际比较；介绍了联合优化可及性、质量和成本的价值方程；并综合了关于iHE使能技术和运营模式的证据。方法遵循近期文献和政策报告的叙述性回顾。结果概述了核心组件（AI决策支持、互操作性、远程医疗、自动化），展示了iHE如何减少浪费、个性化护理，并支持基于价值的支付，同时解决隐私、偏见和采纳挑战。我们提出，协调的iHE可以改变，甚至打破“铁三角”，使系统朝着更可及、更负担得起且质量更高的护理方向发展。 

---
# NS-Pep: De novo Peptide Design with Non-Standard Amino Acids 

**Title (ZH)**: NS-Pep: 用非标准氨基酸的从头肽设计 

**Authors**: Tao Guo, Junbo Yin, Yu Wang, Xin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2510.03326)  

**Abstract**: Peptide drugs incorporating non-standard amino acids (NSAAs) offer improved binding affinity and improved pharmacological properties. However, existing peptide design methods are limited to standard amino acids, leaving NSAA-aware design largely unexplored. We introduce NS-Pep, a unified framework for co-designing peptide sequences and structures with NSAAs. The main challenge is that NSAAs are extremely underrepresented-even the most frequent one, SEP, accounts for less than 0.4% of residues-resulting in a severe long-tailed distribution. To improve generalization to rare amino acids, we propose Residue Frequency-Guided Modification (RFGM), which mitigates over-penalization through frequency-aware logit calibration, supported by both theoretical and empirical analysis. Furthermore, we identify that insufficient side-chain modeling limits geometric representation of NSAAs. To address this, we introduce Progressive Side-chain Perception (PSP) for coarse-to-fine torsion and location prediction, and Interaction-Aware Weighting (IAW) to emphasize pocket-proximal residues. Moreover, NS-Pep generalizes naturally to the peptide folding task with NSAAs, addressing a major limitation of current tools. Experiments show that NS-Pep improves sequence recovery rate and binding affinity by 6.23% and 5.12%, respectively, and outperforms AlphaFold3 by 17.76% in peptide folding success rate. 

**Abstract (ZH)**: 非标准氨基酸aware肽药物设计框架NS-Pep：结合肽序列与结构的统一方法 

---
# The View From Space: Navigating Instrumentation Differences with EOFMs 

**Title (ZH)**: 从太空视角导航：EOFMs处理仪器差异 

**Authors**: Ryan P. Demilt, Nicholas LaHaye, Karis Tenneson  

**Link**: [PDF](https://arxiv.org/pdf/2510.03316)  

**Abstract**: Earth Observation Foundation Models (EOFMs) have exploded in prevalence as tools for processing the massive volumes of remotely sensed and other earth observation data, and for delivering impact on the many essential earth monitoring tasks. An emerging trend posits using the outputs of pre-trained models as 'embeddings' which summarize high dimensional data to be used for generic tasks such as similarity search and content-specific queries. However, most EOFM models are trained only on single modalities of data and then applied or benchmarked by matching bands across different modalities. It is not clear from existing work what impact diverse sensor architectures have on the internal representations of the present suite of EOFMs. We show in this work that the representation space of EOFMs is highly sensitive to sensor architecture and that understanding this difference gives a vital perspective on the pitfalls of current EOFM design and signals for how to move forward as model developers, users, and a community guided by robust remote-sensing science. 

**Abstract (ZH)**: 地球观测基础模型（EOFMs）在处理大规模遥感和其他地球观测数据以及执行众多关键地球监测任务方面变得越来越流行。一项新兴趋势是利用预训练模型的输出作为“嵌入”，以总结高维数据，用于通用任务如相似性搜索和专门内容查询。然而，大多数EOFM模型仅在单一模态数据上进行训练，然后通过不同模态之间的波段匹配进行应用或基准测试。现有工作中尚不清楚多样化的传感器架构如何影响当前EOFM模型的内部表示。本文表明，EOFM的表示空间对传感器架构极为敏感，理解这种差异为当前EOFM设计的陷阱提供了宝贵的视角，并指示了作为模型开发者、用户和受到坚实遥感科学指导的社区应如何向前推进。 

---
# Decomposing Attention To Find Context-Sensitive Neurons 

**Title (ZH)**: 分解注意力以找到上下文敏感神经元 

**Authors**: Alex Gibson  

**Link**: [PDF](https://arxiv.org/pdf/2510.03315)  

**Abstract**: We study transformer language models, analyzing attention heads whose attention patterns are spread out, and whose attention scores depend weakly on content. We argue that the softmax denominators of these heads are stable when the underlying token distribution is fixed. By sampling softmax denominators from a "calibration text", we can combine together the outputs of multiple such stable heads in the first layer of GPT2-Small, approximating their combined output by a linear summary of the surrounding text. This approximation enables a procedure where from the weights alone - and a single calibration text - we can uncover hundreds of first layer neurons that respond to high-level contextual properties of the surrounding text, including neurons that didn't activate on the calibration text. 

**Abstract (ZH)**: 我们研究变压器语言模型，分析那些注意力模式分散且注意力分数对内容依赖较弱的注意力头。我们认为，当底层 token 分布固定时，这些头的 softmax 分母是稳定的。通过从“校准文本”中抽取 softmax 分母，我们可以在 GPT2-Small 的第一层结合多个这样的稳定头的输出，通过邻近文本的线性总结来近似它们的综合输出。这种近似使得仅从权重和一个单一的校准文本出发，我们可以发现数百个对邻近文本的高层上下文属性作出响应的第一层神经元，包括那些在校准文本中未激活的神经元。 

---
# A Comprehensive Review on Artificial Intelligence Empowered Solutions for Enhancing Pedestrian and Cyclist Safety 

**Title (ZH)**: 人工智能赋能的增强行人和骑行者安全解决方案综述 

**Authors**: Shucheng Zhang, Yan Shi, Bingzhang Wang, Yuang Zhang, Muhammad Monjurul Karim, Kehua Chen, Chenxi Liu, Mehrdad Nasri, Yinhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03314)  

**Abstract**: Ensuring the safety of vulnerable road users (VRUs), such as pedestrians and cyclists, remains a critical global challenge, as conventional infrastructure-based measures often prove inadequate in dynamic urban environments. Recent advances in artificial intelligence (AI), particularly in visual perception and reasoning, open new opportunities for proactive and context-aware VRU protection. However, existing surveys on AI applications for VRUs predominantly focus on detection, offering limited coverage of other vision-based tasks that are essential for comprehensive VRU understanding and protection. This paper presents a state-of-the-art review of recent progress in camera-based AI sensing systems for VRU safety, with an emphasis on developments from the past five years and emerging research trends. We systematically examine four core tasks, namely detection and classification, tracking and reidentification, trajectory prediction, and intent recognition and prediction, which together form the backbone of AI-empowered proactive solutions for VRU protection in intelligent transportation systems. To guide future research, we highlight four major open challenges from the perspectives of data, model, and deployment. By linking advances in visual AI with practical considerations for real-world implementation, this survey aims to provide a foundational reference for the development of next-generation sensing systems to enhance VRU safety. 

**Abstract (ZH)**: 确保弱势道路交通使用者（VRUs）的安全仍然是一个关键的全球挑战，因为传统的基于基础设施的措施在动态城市环境中往往不够充分。近期人工智能（AI）特别是在视觉感知和推理方面的发展为提前预警和情境感知的VRU保护开启了新的机会。然而，现有的关于AI在VRU应用的综述大多侧重于检测，对全面理解与保护VRU所必需的其他视觉任务关注不足。本文对过去五年中基于相机的AI传感系统在VRU安全领域的最新进展进行了综述，并着重探讨了新兴研究趋势。我们系统地研究了四个核心任务，即检测与分类、跟踪与重识别、轨迹预测以及意图识别与预测，这些任务构成了利用AI增强的主动保护解决方案的核心，用于智能交通系统中的VRU保护。为了指导未来研究，我们从数据、模型和部署三个角度突出了四个主要的开放挑战。通过将视觉AI的进步与实际应用中的考虑事项相结合，本文旨在为开发下一代传感系统以增强VRU安全提供基础参考。 

---
# Atlas-free Brain Network Transformer 

**Title (ZH)**: 无图谱脑网络变换器 

**Authors**: Shuai Huang, Xuan Kan, James J. Lah, Deqiang Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03306)  

**Abstract**: Current atlas-based approaches to brain network analysis rely heavily on standardized anatomical or connectivity-driven brain atlases. However, these fixed atlases often introduce significant limitations, such as spatial misalignment across individuals, functional heterogeneity within predefined regions, and atlas-selection biases, collectively undermining the reliability and interpretability of the derived brain networks. To address these challenges, we propose a novel atlas-free brain network transformer (atlas-free BNT) that leverages individualized brain parcellations derived directly from subject-specific resting-state fMRI data. Our approach computes ROI-to-voxel connectivity features in a standardized voxel-based feature space, which are subsequently processed using the BNT architecture to produce comparable subject-level embeddings. Experimental evaluations on sex classification and brain-connectome age prediction tasks demonstrate that our atlas-free BNT consistently outperforms state-of-the-art atlas-based methods, including elastic net, BrainGNN, Graphormer and the original BNT. Our atlas-free approach significantly improves the precision, robustness, and generalizability of brain network analyses. This advancement holds great potential to enhance neuroimaging biomarkers and clinical diagnostic tools for personalized precision medicine. 

**Abstract (ZH)**: 一种基于个体化脑区划分的无图谱脑网络变换器（ atlas-free BNT） 

---
# Dynamic Meta-Learning for Adaptive XGBoost-Neural Ensembles 

**Title (ZH)**: 动态元学习适配XGBoost-神经集成模型 

**Authors**: Arthur Sedek  

**Link**: [PDF](https://arxiv.org/pdf/2510.03301)  

**Abstract**: This paper introduces a novel adaptive ensemble framework that synergistically combines XGBoost and neural networks through sophisticated meta-learning. The proposed method leverages advanced uncertainty quantification techniques and feature importance integration to dynamically orchestrate model selection and combination. Experimental results demonstrate superior predictive performance and enhanced interpretability across diverse datasets, contributing to the development of more intelligent and flexible machine learning systems. 

**Abstract (ZH)**: 本文介绍了一种新颖的适应性集成框架，该框架通过精巧的元学习协同结合XGBoost和神经网络。所提出的方法利用高级不确定性量化技术和特征重要性集成，动态协调模型选择与组合。实验结果表明，该方法在多种数据集上展现出优越的预测性能和增强的可解释性，促进了更智能和灵活的机器学习系统的开发。 

---
# From Score Distributions to Balance: Plug-and-Play Mixture-of-Experts Routing 

**Title (ZH)**: 从分数分布到平衡：即插即用专家混合路由 

**Authors**: Rana Shahout, Colin Cai, Yilun Du, Minlan Yu, Michael Mitzenmacher  

**Link**: [PDF](https://arxiv.org/pdf/2510.03293)  

**Abstract**: Mixture-of-Experts (MoE) models can scale parameter capacity by routing each token to a subset of experts through a learned gate function. While conditional routing reduces training costs, it shifts the burden on inference memory: expert parameters and activations consume memory, limiting the number of experts per device. As tokens are routed, some experts become overloaded while others are underutilized. Because experts are mapped to GPUs, this imbalance translates directly into degraded system performance in terms of latency, throughput, and cost. We present LASER, a plug-and-play, inference-time routing algorithm that balances load while preserving accuracy. LASER adapts to the shape of the gate's score distribution. When scores provide a clear preference, it routes to the strongest experts; when scores are more uniform, it broadens the set of viable experts and routes to the least-loaded among them. Because LASER relies only on gate scores from a trained model, it integrates directly into existing MoE inference pipelines without retraining or finetuning. We evaluate LASER on Mixtral-8x7B and DeepSeek-MoE-16b-chat across four datasets (ARC-Easy, ARC-Challenge, MMLU, and GSM8K). LASER improves load balancing, translating into lower latency and higher throughput, while keeping the accuracy changes negligible. 

**Abstract (ZH)**: 基于专家混合的插件式推理时间路由算法LASER：保持准确性的负载均衡方法 

---
# LogAction: Consistent Cross-system Anomaly Detection through Logs via Active Domain 

**Title (ZH)**: LogAction: 通过主动领域一致跨系统异常检测 

**Authors**: Chiming Duan, Minghua He, Pei Xiao, Tong Jia, Xin Zhang, Zhewei Zhong, Xiang Luo, Yan Niu, Lingzhe Zhang, Yifan Wu, Siyu Yu, Weijie Hong, Ying Li, Gang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03288)  

**Abstract**: Log-based anomaly detection is a essential task for ensuring the reliability and performance of software systems. However, the performance of existing anomaly detection methods heavily relies on labeling, while labeling a large volume of logs is highly challenging. To address this issue, many approaches based on transfer learning and active learning have been proposed. Nevertheless, their effectiveness is hindered by issues such as the gap between source and target system data distributions and cold-start problems. In this paper, we propose LogAction, a novel log-based anomaly detection model based on active domain adaptation. LogAction integrates transfer learning and active learning techniques. On one hand, it uses labeled data from a mature system to train a base model, mitigating the cold-start issue in active learning. On the other hand, LogAction utilize free energy-based sampling and uncertainty-based sampling to select logs located at the distribution boundaries for manual labeling, thus addresses the data distribution gap in transfer learning with minimal human labeling efforts. Experimental results on six different combinations of datasets demonstrate that LogAction achieves an average 93.01% F1 score with only 2% of manual labels, outperforming some state-of-the-art methods by 26.28%. Website: this https URL 

**Abstract (ZH)**: 基于日志的异常检测是确保软件系统可靠性和性能的重要任务。然而，现有异常检测方法的性能高度依赖于标签，而大量日志的标注极具挑战性。为应对这一问题，提出了多种基于迁移学习和活跃学习的方法。然而，这些方法的有效性受限于源系统和目标系统数据分布之间的差距以及冷启动问题。在本文中，我们提出了LogAction，一种基于活跃域适应的新型基于日志的异常检测模型。LogAction结合了迁移学习和活跃学习技术。一方面，它使用成熟系统的标注数据训练基模型，缓解活跃学习中的冷启动问题；另一方面，LogAction利用自由能采样和不确定性采样选择位于分布边界上的日志进行人工标注，从而在最小的人工标注努力下解决迁移学习中的数据分布差距问题。六组不同数据集的实验结果显示，LogAction仅使用2%的手标注数据即可获得平均93.01%的F1分数，优于一些最先进的方法26.28%。网站: 这个 https URL。 

---
# MemMamba: Rethinking Memory Patterns in State Space Model 

**Title (ZH)**: MemMamba: 在状态空间模型中重思内存模式 

**Authors**: Youjin Wang, Yangjingyi Chen, Jiahao Yan, Jiaxuan Lu, Xiao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.03279)  

**Abstract**: With the explosive growth of data, long-sequence modeling has become increasingly important in tasks such as natural language processing and bioinformatics. However, existing methods face inherent trade-offs between efficiency and memory. Recurrent neural networks suffer from gradient vanishing and explosion, making them hard to scale. Transformers can model global dependencies but are constrained by quadratic complexity. Recently, selective state-space models such as Mamba have demonstrated high efficiency with O(n) time and O(1) recurrent inference, yet their long-range memory decays exponentially. In this work, we conduct mathematical derivations and information-theoretic analysis to systematically uncover the memory decay mechanism of Mamba, answering a fundamental question: what is the nature of Mamba's long-range memory and how does it retain information? To quantify key information loss, we further introduce horizontal-vertical memory fidelity metrics that capture degradation both within and across layers. Inspired by how humans distill and retain salient information when reading long documents, we propose MemMamba, a novel architectural framework that integrates state summarization mechanism together with cross-layer and cross-token attention, which alleviates long-range forgetting while preserving linear complexity. MemMamba achieves significant improvements over existing Mamba variants and Transformers on long-sequence benchmarks such as PG19 and Passkey Retrieval, while delivering a 48% speedup in inference efficiency. Both theoretical analysis and empirical results demonstrate that MemMamba achieves a breakthrough in the complexity-memory trade-off, offering a new paradigm for ultra-long sequence modeling. 

**Abstract (ZH)**: 随着数据的爆炸性增长，长序列 modeling 在自然语言处理和生物信息学等任务中变得越来越重要。然而，现有方法在效率和内存之间存在固有的权衡。循环神经网络遭受梯度消失和爆炸的问题，难以扩展。变压器可以建模全局依赖关系，但受到二次复杂度的限制。最近，如 Mamba 等选择性状态空间模型展示了 O(n) 时间和 O(1) 递归推理的高度效率，但其长期记忆呈现指数衰减。在本文中，我们通过数学推导和信息论分析系统地揭示了 Mamba 的记忆衰减机制，回答了一个基本问题：Mamba 的长期记忆的本质是什么，它是如何保留信息的？为了量化关键信息损失，我们进一步引入了水平-垂直记忆保真度指标，以捕获层内和跨层的退化。借鉴人类在阅读长文档时提取和保留关键信息的方式，我们提出了一种新的架构框架 MemMamba，该框架结合了状态汇总机制和跨层及跨标记注意力，从而减轻长期遗忘现象，同时保持线性复杂度。MemMamba 在 PG19 和 Passkey Retrieval 等长序列基准测试中显著优于现有 Mamba 变体和 Transformers，且推理效率提升 48%。理论分析和实验结果都表明，MemMamba 在复杂性-内存权衡中实现了突破，为超长序列 modeling 提供了一个新的范式。 

---
# Quantifying constraint hierarchies in Bayesian PINNs via per-constraint Hessian decomposition 

**Title (ZH)**: 通过每约束海森矩阵分解量化贝叶斯PINNs中的约束层次结构 

**Authors**: Filip Landgren  

**Link**: [PDF](https://arxiv.org/pdf/2510.03278)  

**Abstract**: Bayesian physics-informed neural networks (B-PINNs) merge data with governing equations to solve differential equations under uncertainty. However, interpreting uncertainty and overconfidence in B-PINNs requires care due to the poorly understood effects the physical constraints have on the network; overconfidence could reflect warranted precision, enforced by the constraints, rather than miscalibration. Motivated by the need to further clarify how individual physical constraints shape these networks, we introduce a scalable, matrix-free Laplace framework that decomposes the posterior Hessian into contributions from each constraint and provides metrics to quantify their relative influence on the loss landscape. Applied to the Van der Pol equation, our method tracks how constraints sculpt the network's geometry and shows, directly through the Hessian, how changing a single loss weight non-trivially redistributes curvature and effective dominance across the others. 

**Abstract (ZH)**: 基于贝叶斯的物理约束神经网络（B-PINNs）将数据与 governing 方程合并以在不确定性条件下求解微分方程。然而，由于物理约束对网络影响的不明确性，解释 B-PINNs 中的不确定性与过度自信需要谨慎；过度自信可能反映了由约束施加的适当精度，而非校准不当。为更清晰地理解单个物理约束如何影响这些网络，我们引入了一种可扩展的、无需矩阵运算的拉普拉斯框架，该框架将后验哈密顿量分解为每个约束的贡献，并提供量化其在损失景观上相对影响的指标。该方法应用于范德蒙德方程，跟踪约束如何塑造网络几何结构，并直接通过哈密顿量展示了改变单个损失权重非平凡地重新分配曲率和有效支配的方式。 

---
# Learning without Global Backpropagation via Synergistic Information Distillation 

**Title (ZH)**: 无需全局反向传播的协同信息蒸馏学习 

**Authors**: Chenhao Ye, Ming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03273)  

**Abstract**: Backpropagation (BP), while foundational to deep learning, imposes two critical scalability bottlenecks: update locking, where network modules remain idle until the entire backward pass completes, and high memory consumption due to storing activations for gradient computation. To address these limitations, we introduce Synergistic Information Distillation (SID), a novel training framework that reframes deep learning as a cascade of local cooperative refinement problems. In SID, a deep network is structured as a pipeline of modules, each imposed with a local objective to refine a probabilistic belief about the ground-truth target. This objective balances fidelity to the target with consistency to the belief from its preceding module. By decoupling the backward dependencies between modules, SID enables parallel training and hence eliminates update locking and drastically reduces memory requirements. Meanwhile, this design preserves the standard feed-forward inference pass, making SID a versatile drop-in replacement for BP. We provide a theoretical foundation, proving that SID guarantees monotonic performance improvement with network depth. Empirically, SID consistently matches or surpasses the classification accuracy of BP, exhibiting superior scalability and pronounced robustness to label this http URL is available at: this https URL 

**Abstract (ZH)**: 反向传播（BP）尽管是深度学习的基础，但其仍面临两大关键的可扩展性瓶颈：更新锁定，即网络模块在完整反向传播完成前处于闲置状态，以及由于用于梯度计算而产生的高内存消耗。为解决这些限制，我们引入了协同信息精炼（SID）这一新的训练框架，将深度学习重新定义为一系列本地协同精炼问题的级联过程。在SID中，一个深网络被构建成模块流水线的形式，每个模块承载一个局部目标，以精确一种关于真实目标的概率信念。该目标平衡了对目标的忠实度和与前一个模块信念的一致性。通过解除模块之间的反向依赖关系，SID使训练过程能够并行进行，从而消除了更新锁定并极大地减少了内存需求。同时，这种设计保留了标准的前向推理过程，使得SID成为一个通用的BP替代方案。我们提供了理论基础，证明SID随网络深度增加能保证性能的单调改进。实验证明，SID在分类准确性上始终与BP相符或超越，表现出更好的可扩展性和显著的鲁棒性。更多详情请参阅：https://thishttpURL/isavailableat:https://thishttpsURL 

---
# PDE-Transformer: A Continuous Dynamical Systems Approach to Sequence Modeling 

**Title (ZH)**: PDE-Transformer：连续动力系统在序列建模中的应用 

**Authors**: Yukun Zhang, Xueqing Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.03272)  

**Abstract**: The Transformer architecture has revolutionized artificial intelligence, yet a principled theoretical understanding of its internal mechanisms remains elusive. This paper introduces a novel analytical framework that reconceptualizes the Transformer's discrete, layered structure as a continuous spatiotemporal dynamical system governed by a master Partial Differential Equation (PDE). Within this paradigm, we map core architectural components to distinct mathematical operators: self-attention as a non-local interaction, the feed-forward network as a local reaction, and, critically, residual connections and layer normalization as indispensable stabilization mechanisms. We do not propose a new model, but rather employ the PDE system as a theoretical probe to analyze the mathematical necessity of these components. By comparing a standard Transformer with a PDE simulator that lacks explicit stabilizers, our experiments provide compelling empirical evidence for our central thesis. We demonstrate that without residual connections, the system suffers from catastrophic representational drift, while the absence of layer normalization leads to unstable, explosive training dynamics. Our findings reveal that these seemingly heuristic "tricks" are, in fact, fundamental mathematical stabilizers required to tame an otherwise powerful but inherently unstable continuous system. This work offers a first-principles explanation for the Transformer's design and establishes a new paradigm for analyzing deep neural networks through the lens of continuous dynamics. 

**Abstract (ZH)**: Transformer架构改变了人工智能领域，但对其内部机制的原理性理论理解仍然模糊。本文引入了一个新的分析框架，将Transformer的离散分层结构重新概念化为由主偏微分方程（PDE）控制的连续空间-时间动力系统。在这种范式下，我们将核心架构组件映射到不同的数学运算符：自我注意作为一种非局部交互，前馈网络作为一种局部反应，并且至关重要地，残差连接和层归一化作为必不可少的稳定机制。我们并非提出新的模型，而是利用PDE系统作为理论探针来分析这些组件的必要性。通过将标准Transformer与缺乏显式稳定器的PDE模拟器进行比较，我们的实验提供了强有力的经验证据支持我们的核心论点。我们证明，在没有残差连接的情况下，系统会遭受灾难性的表示迁移，而在没有层归一化的情况下则导致不稳定的、爆炸性的训练动力学。我们的发现揭示，这些看似启发式的“技巧”实际上是必要且根本的数学稳定机制，使一个原本强大但本质上不稳定的连续系统变得可控。这项工作为Transformer的设计提供了一种第一性原理解释，并确立了一种通过连续动力学视角分析深度神经网络的新范式。 

---
# General Exploratory Bonus for Optimistic Exploration in RLHF 

**Title (ZH)**: 乐观探索在RLHF中的通用探索bonus 

**Authors**: Wendi Li, Changdae Oh, Yixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.03269)  

**Abstract**: Optimistic exploration is central to improving sample efficiency in reinforcement learning with human feedback, yet existing exploratory bonus methods to incentivize exploration often fail to realize optimism. We provide a theoretical analysis showing that current formulations, under KL or $\alpha$-divergence regularization, unintentionally bias exploration toward high-probability regions of the reference model, thereby reinforcing conservative behavior instead of promoting discovery of uncertain regions. To address this pitfall, we introduce the General Exploratory Bonus (GEB), a novel theoretical framework that provably satisfies the optimism principle. GEB counteracts divergence-induced bias via reference-dependent reward regulation and unifies prior heuristic bonuses as special cases, while extending naturally across the full $\alpha$-divergence family. Empirically, GEB consistently outperforms baselines on alignment tasks across multiple divergence settings and large language model backbones. These results demonstrate that GEB offers both a principled and practical solution for optimistic exploration in RLHF. 

**Abstract (ZH)**: 乐观探索是提高强化学习中人类反馈样本效率的核心，但现有的探索奖励方法往往未能体现乐观性。我们提供了理论分析，表明当前基于KL或$\alpha$-散度正则化的形式无意中将探索偏向参考模型的高概率区域，从而 reinforcement 学习中强化保守行为而非促进对不确定区域的探索。为解决这一问题，我们提出了广义探索奖励（GEB），这是一个能严格满足乐观原则的新型理论框架。GEB 通过参考依赖的奖励调节抵消散度引入的偏差，并将先前的启发式奖励作为特殊情况统一其中，同时自然地扩展到整个$\alpha$-散度家族。实验结果显示，GEB 在多种散度设置和大型语言模型基础下的一致性表现优于基线方法，这些结果表明，GEB 为强化学习中的人类反馈提供了一个既具原理性又实用的乐观探索解决方案。 

---
# Rethinking Inter-LoRA Orthogonality in Adapter Merging: Insights from Orthogonal Monte Carlo Dropout 

**Title (ZH)**: 重思适配器合并中的LoRA正交性：正交蒙特卡洛dropout的见解 

**Authors**: Andi Zhang, Xuan Ding, Haofan Wang, Steven McDonagh, Samuel Kaski  

**Link**: [PDF](https://arxiv.org/pdf/2510.03262)  

**Abstract**: We propose Orthogonal Monte Carlo Dropout, a mechanism that enforces strict orthogonality when combining sparse semantic vectors without extra time complexity. LoRA, a popular fine-tuning method for large models, typically trains a module to represent a specific concept such as an object or a style. When multiple LoRAs are merged, for example to generate an object in a particular style, their semantic vectors may interfere with each other. Our method guarantees, at the theoretical and runtime levels, that merged LoRAs remain orthogonal and thus free from direct interference. However, empirical analysis reveals that such orthogonality does not lead to the semantic disentanglement or compositionality highlighted in prior work on compositional adaptation. This finding suggests that inter-LoRA orthogonality alone may be insufficient for achieving true semantic compositionality, prompting a re-examination of its role in adapter merging. 

**Abstract (ZH)**: 我们提出正交蒙特卡洛丢弃机制，在不增加额外时间复杂度的情况下，确保稀疏语义向量的严格正交性。LoRA 是一种流行的大模型精细调整方法，通常训练一个模块来表示特定概念，如对象或风格。当多个 LoRAs 被合并时，例如为了生成特定风格的对象，它们的语义向量可能会相互干扰。我们的方法在理论上和运行时保证合并后的 LoRAs 保持正交，从而避免直接干扰。然而，实证分析表明，这种正交性并不能带来先前关于组合适应性工作所强调的语义去纠缠性和组合性。这一发现表明，仅靠 LoRA 间的正交性可能不足以实现真正的语义组合性，这促使我们重新审视其在适配器合并中的作用。 

---
# Semantic-Inductive Attribute Selection for Zero-Shot Learning 

**Title (ZH)**: 零样本学习中的语义归纳属性选择 

**Authors**: Juan Jose Herrera-Aranda, Guillermo Gomez-Trenado, Francisco Herrera, Isaac Triguero  

**Link**: [PDF](https://arxiv.org/pdf/2510.03260)  

**Abstract**: Zero-Shot Learning is an important paradigm within General-Purpose Artificial Intelligence Systems, particularly in those that operate in open-world scenarios where systems must adapt to new tasks dynamically. Semantic spaces play a pivotal role as they bridge seen and unseen classes, but whether human-annotated or generated by a machine learning model, they often contain noisy, redundant, or irrelevant attributes that hinder performance. To address this, we introduce a partitioning scheme that simulates unseen conditions in an inductive setting (which is the most challenging), allowing attribute relevance to be assessed without access to semantic information from unseen classes. Within this framework, we study two complementary feature-selection strategies and assess their generalisation. The first adapts embedded feature selection to the particular demands of ZSL, turning model-driven rankings into meaningful semantic pruning; the second leverages evolutionary computation to directly explore the space of attribute subsets more broadly. Experiments on five benchmark datasets (AWA2, CUB, SUN, aPY, FLO) show that both methods consistently improve accuracy on unseen classes by reducing redundancy, but in complementary ways: RFS is efficient and competitive though dependent on critical hyperparameters, whereas GA is more costly yet explores the search space more broadly and avoids such dependence. These results confirm that semantic spaces are inherently redundant and highlight the proposed partitioning scheme as an effective tool to refine them under inductive conditions. 

**Abstract (ZH)**: 零样本学习是通用人工智能系统中一个重要的范式，特别是在开放世界场景中，系统需要动态适应新任务。语义空间在这种情境下发挥关键作用，它们连接了已知和未知类别，但无论是由人类标注还是通过机器学习模型生成，语义空间往往包含噪声、冗余或无关的属性，从而妨碍性能。为了解决这一问题，我们提出了一种分区方案，在归纳设置中模拟未知条件（最具挑战性的情况），允许在不访问未知类别语义信息的情况下评估属性的相关性。在此框架内，我们研究了两种互补的特征选择策略，并评估了它们的一般性。第一种方法针对零样本学习的具体需求，将模型驱动的排名转化为有意义的语义修剪；第二种方法利用进化计算更广泛地探索属性子集的空间。实验结果显示，在五个基准数据集（AWA2、CUB、SUN、aPY、FLO）上，两种方法都能通过减少冗余性在未知类别上提高准确性，但方式互补：特征选择（RFS）高效且具有竞争力，但依赖于关键超参数；而遗传算法（GA）虽然成本更高，但能更广泛地探索搜索空间并避免这种依赖性。这些结果证实了语义空间本身存在冗余性，并强调了所提出的分区方案作为在归纳条件下完善语义空间的有效工具。 

---
# POEM: Explore Unexplored Reliable Samples to Enhance Test-Time Adaptation 

**Title (ZH)**: POEM: 探索未探索的可靠样本以增强测试时适应性 

**Authors**: Chang'an Yi, Xiaohui Deng, Shuaicheng Niu, Yan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.03258)  

**Abstract**: Test-time adaptation (TTA) aims to transfer knowledge from a source model to unknown test data with potential distribution shifts in an online manner. Many existing TTA methods rely on entropy as a confidence metric to optimize the model. However, these approaches are sensitive to the predefined entropy threshold, influencing which samples are chosen for model adaptation. Consequently, potentially reliable target samples are often overlooked and underutilized. For instance, a sample's entropy might slightly exceed the threshold initially, but fall below it after the model is updated. Such samples can provide stable supervised information and offer a normal range of gradients to guide model adaptation. In this paper, we propose a general approach, \underline{POEM}, to promote TTA via ex\underline{\textbf{p}}loring the previously unexpl\underline{\textbf{o}}red reliabl\underline{\textbf{e}} sa\underline{\textbf{m}}ples. Additionally, we introduce an extra Adapt Branch network to strike a balance between extracting domain-agnostic representations and achieving high performance on target data. Comprehensive experiments across multiple architectures demonstrate that POEM consistently outperforms existing TTA methods in both challenging scenarios and real-world domain shifts, while remaining computationally efficient. The effectiveness of POEM is evaluated through extensive analyses and thorough ablation studies. Moreover, the core idea behind POEM can be employed as an augmentation strategy to boost the performance of existing TTA approaches. The source code is publicly available at \emph{this https URL} 

**Abstract (ZH)**: Test-time adaptation (TTA)旨在在线方式将知识从源模型转移到具有潜在分布偏移的未知测试数据中。许多现有的TTA方法依赖于熵作为置信度度量来优化模型。然而，这些方法对预定义的熵阈值敏感，影响了哪些样本被选中进行模型适应。因此，许多潜在可靠的目标样本往往被忽视和未充分利用。例如，一个样本的熵最初可能略微超过阈值，但在模型更新后又低于阈值。这种样本可以提供稳定的监督信息，并提供指导模型适应的正常范围梯度。在本文中，我们提出了一种通用方法POEM，通过探索之前未探索的可靠样本来促进TTA。此外，我们引入了一个额外的Adapt Branch网络，以平衡提取领域无关表示和在目标数据上实现高性能之间的关系。在多种架构上的综合实验表明，在具有挑战性的场景和实际领域偏移中，POEM始终优于现有的TTA方法，同时保持计算效率。POEM的有效性通过广泛分析和彻底的消融研究进行了评估。此外，POEM的核心思想可以作为一种增强策略，以提高现有TTA方法的性能。源代码可在\emph{this https URL}公开获取。 

---
# Triple-BERT: Do We Really Need MARL for Order Dispatch on Ride-Sharing Platforms? 

**Title (ZH)**: Triple-BERT: 我们真的需要多智能体强化学习来解决拼车平台的订单分配问题吗？ 

**Authors**: Zijian Zhao, Sen Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.03257)  

**Abstract**: On-demand ride-sharing platforms, such as Uber and Lyft, face the intricate real-time challenge of bundling and matching passengers-each with distinct origins and destinations-to available vehicles, all while navigating significant system uncertainties. Due to the extensive observation space arising from the large number of drivers and orders, order dispatching, though fundamentally a centralized task, is often addressed using Multi-Agent Reinforcement Learning (MARL). However, independent MARL methods fail to capture global information and exhibit poor cooperation among workers, while Centralized Training Decentralized Execution (CTDE) MARL methods suffer from the curse of dimensionality. To overcome these challenges, we propose Triple-BERT, a centralized Single Agent Reinforcement Learning (MARL) method designed specifically for large-scale order dispatching on ride-sharing platforms. Built on a variant TD3, our approach addresses the vast action space through an action decomposition strategy that breaks down the joint action probability into individual driver action probabilities. To handle the extensive observation space, we introduce a novel BERT-based network, where parameter reuse mitigates parameter growth as the number of drivers and orders increases, and the attention mechanism effectively captures the complex relationships among the large pool of driver and orders. We validate our method using a real-world ride-hailing dataset from Manhattan. Triple-BERT achieves approximately an 11.95% improvement over current state-of-the-art methods, with a 4.26% increase in served orders and a 22.25% reduction in pickup times. Our code, trained model parameters, and processed data are publicly available at the repository this https URL . 

**Abstract (ZH)**: 基于需求的共享出行平台（如Uber和Lyft）面临实时挑战，即如何将具有不同出发地和目的地的乘客与可用车辆进行有效的组合和匹配，同时还要应对系统中的大量不确定性。由于驾驶员和订单数量庞大导致观测空间广泛，尽管本质上是集中式任务，但订单调度通常采用多智能体强化学习（MARL）方法。然而，独立的MARL方法无法捕捉全局信息，智能体间合作效果差，而集中式训练分布式执行（CTDE）的MARL方法则遭受维度灾难。为克服这些挑战，我们提出了Triple-BERT方法，这是一种专为共享出行平台大规模订单调度设计的集中式单智能体强化学习方法。基于TD3的变体，该方法通过动作分解策略解决庞大的动作空间问题，将联合动作概率分解为单个驾驶员的动作概率。为应对广泛的观测空间，我们引入了一种新型的基于BERT的网络，其中参数重用在驾驶员和订单数量增加时减少了参数的增长，并且注意力机制有效地捕捉了大量驾驶员和订单间的复杂关系。我们在曼哈顿的真实打车数据集上验证了该方法。Triple-BERT在现有最先进的方法上实现了约11.95%的性能提升，服务订单数增加了4.26%，接客时间减少了22.25%。我们的代码、训练模型参数和处理后的数据可在以下仓库公开访问：this https URL。 

---
# Universal Multi-Domain Translation via Diffusion Routers 

**Title (ZH)**: 通过扩散路由器实现的通用多域翻译 

**Authors**: Duc Kieu, Kien Do, Tuan Hoang, Thao Minh Le, Tung Kieu, Dang Nguyen, Thin Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2510.03252)  

**Abstract**: Multi-domain translation (MDT) aims to learn translations between multiple domains, yet existing approaches either require fully aligned tuples or can only handle domain pairs seen in training, limiting their practicality and excluding many cross-domain mappings. We introduce universal MDT (UMDT), a generalization of MDT that seeks to translate between any pair of $K$ domains using only $K-1$ paired datasets with a central domain. To tackle this problem, we propose Diffusion Router (DR), a unified diffusion-based framework that models all central$\leftrightarrow$non-central translations with a single noise predictor conditioned on the source and target domain labels. DR enables indirect non-central translations by routing through the central domain. We further introduce a novel scalable learning strategy with a variational-bound objective and an efficient Tweedie refinement procedure to support direct non-central mappings. Through evaluation on three large-scale UMDT benchmarks, DR achieves state-of-the-art results for both indirect and direct translations, while lowering sampling cost and unlocking novel tasks such as sketch$\leftrightarrow$segmentation. These results establish DR as a scalable and versatile framework for universal translation across multiple domains. 

**Abstract (ZH)**: 多域翻译（MDT）旨在学习多个域之间的翻译，但现有方法要么需要完全对齐的元组，要么只能处理训练中出现的域对，这限制了它们的实际应用并排除了许多跨域映射。我们引入了一种多域翻译的通用化方法（UMDT），该方法仅使用一个中心域和$K-1$个中心域与非中心域的配对数据集，即可在任意一对$K$域之间进行翻译。为了解决这一问题，我们提出了统一的扩散路由器（DR），这是一种基于扩散的统一框架，它通过在源域和目标域标签的条件下条件化噪声预测器来建模所有中心$\leftrightarrow$非中心的翻译。DR通过路由到中心域来实现间接的非中心翻译。我们还引入了一种新颖的可扩展学习策略，该策略具有变分界线目标和高效的Tweedie精炼过程，以支持直接的非中心映射。通过在三个大规模UMDT基准上的评估，DR在间接和直接翻译中均实现了最优结果，同时降低了采样成本并解锁了如草图$\leftrightarrow$分割等新任务。这些结果确立了DR作为一种适用于多域通用翻译的可扩展和多功能框架的地位。 

---
# Numerion: A Multi-Hypercomplex Model for Time Series Forecasting 

**Title (ZH)**: Numerion：一种用于时间序列预测的多超复数模型 

**Authors**: Hanzhong Cao, Wenbo Yan, Ying Tan  

**Link**: [PDF](https://arxiv.org/pdf/2510.03251)  

**Abstract**: Many methods aim to enhance time series forecasting by decomposing the series through intricate model structures and prior knowledge, yet they are inevitably limited by computational complexity and the robustness of the assumptions. Our research uncovers that in the complex domain and higher-order hypercomplex spaces, the characteristic frequencies of time series naturally decrease. Leveraging this insight, we propose Numerion, a time series forecasting model based on multiple hypercomplex spaces. Specifically, grounded in theoretical support, we generalize linear layers and activation functions to hypercomplex spaces of arbitrary power-of-two dimensions and introduce a novel Real-Hypercomplex-Real Domain Multi-Layer Perceptron (RHR-MLP) architecture. Numerion utilizes multiple RHR-MLPs to map time series into hypercomplex spaces of varying dimensions, naturally decomposing and independently modeling the series, and adaptively fuses the latent patterns exhibited in different spaces through a dynamic fusion mechanism. Experiments validate the model`s performance, achieving state-of-the-art results on multiple public datasets. Visualizations and quantitative analyses comprehensively demonstrate the ability of multi-dimensional RHR-MLPs to naturally decompose time series and reveal the tendency of higher dimensional hypercomplex spaces to capture lower frequency features. 

**Abstract (ZH)**: 基于超复数空间的Numerion时间序列forecasting模型 

---
# Real-Time Brain Biomechanics Prediction with Neural Operators: Toward Clinically Deployable Traumatic Brain Injury Models 

**Title (ZH)**: 基于神经算子的实时脑 biomechanics 预测：迈向临床可用的创伤性脑损伤模型 

**Authors**: Anusha Agarwal, Dibakar Roy Sarkar, Somdatta Goswami  

**Link**: [PDF](https://arxiv.org/pdf/2510.03248)  

**Abstract**: Traumatic brain injury (TBI) remains a major public health concern, with over 69 million cases annually worldwide. Finite element (FE) models offer high-fidelity predictions of brain deformation but are computationally expensive, requiring hours per simulation and limiting their clinical utility for rapid decision-making. This study benchmarks state-of-the-art neural operator (NO) architectures for rapid, patient-specific prediction of brain displacement fields, aiming to enable real-time TBI modeling in clinical and translational settings. We formulated TBI modeling as an operator learning problem, mapping subject-specific anatomical MRI, magnetic resonance elastography (MRE) stiffness maps, and demographic features to full-field 3D brain displacement predictions. Four architectures - Fourier Neural Operator (FNO), Factorized FNO (F-FNO), Multi-Grid FNO (MG-FNO), and Deep Operator Network (DeepONet) were trained and evaluated on 249 MRE datasets across physiologically relevant frequencies (20 - 90 Hz). MG-FNO achieved the highest accuracy (MSE = 0.0023, 94.3\% spatial fidelity) and preserved fine-scale features, while F-FNO converged 2$\times$ faster than standard FNO. DeepONet offered the fastest inference (14.5 iterations/s) with a 7$\times$ computational speed-up over MG-FNO, suggesting utility for embedded or edge computing applications. All NOs reduced computation time from hours to milliseconds without sacrificing anatomical realism. NOs provide an efficient, resolution-invariant approach for predicting brain deformation, opening the door to real-time, patient-specific TBI risk assessment, clinical triage support, and optimization of protective equipment. These results highlight the potential for NO-based digital twins of the human brain, enabling scalable, on-demand biomechanical modeling in both clinical and population health contexts. 

**Abstract (ZH)**: 创伤性脑损伤(TBI)仍然是一个主要的公共健康问题，每年全球病例超过6900万例。有限元(FE)模型能够高保真地预测脑部变形，但是计算成本高昂，每个模拟需要数小时的时间，限制了其在快速决策临床环境中的应用。本研究旨在通过先进的神经运算器( Neural Operator, NO)架构，实现快速的患者特定脑部位移场预测，以期在临床和转化研究中实现即时创伤性脑损伤(TBI)建模。我们将TBI建模定性为一个运算器学习问题，将个体化的解剖MRI、磁共振弹性图(MRE)刚度图以及人口统计特征映射到全领域的三维脑部位移预测。四种架构——傅里叶神经运算器(Fourier Neural Operator, FNO)、因子分解傅里叶神经运算器(Factorized FNO, F-FNO)、多重网格傅里叶神经运算器(Multi-Grid FNO, MG-FNO) 和深度运算器网络(Deep Operator Network, DeepONet)——在249个MRE数据集上进行了训练和评估，这些数据涵盖了生理相关频率（20-90 Hz）。多重网格傅里叶神经运算器(MG-FNO)达到了最高的准确性（均方误差MSE=0.0023，空间保真度94.3%）并保留了细尺度特征，同时因子分解傅里叶神经运算器(F-FNO)比标准傅里叶神经运算器(FNO)快两倍的收敛速度。深度运算器网络提供了最快推断速度（每秒14.5次迭代），相比多重网格傅里叶神经运算器有七倍的计算速度提升，表明其可能适用于嵌入式或边缘计算应用。所有神经运算器(NOs)将计算时间从数小时减少到毫秒级，同时保持了解剖学的真实感。神经运算器提供了一种高效、分辨率无关的方法来预测脑变形，为实现即时、患者特定的TBI风险评估、临床分诊支持和防护设备优化提供了可能。这些结果突显了基于神经运算器的类人脑数字孪生的潜力，能够在临床和人口健康领域实现扩展且按需的生物力学建模。 

---
# Frequency-Aware Model Parameter Explorer: A new attribution method for improving explainability 

**Title (ZH)**: 频率感知模型参数探索者：一种提高可解释性的归因方法 

**Authors**: Ali Yavari, Alireza Mohamadi, Elham Beydaghi, Rainer A. Leitgeb  

**Link**: [PDF](https://arxiv.org/pdf/2510.03245)  

**Abstract**: Ensuring the reliability of deep neural networks (DNNs) in the presence of real world noise and intentional perturbations remains a significant challenge. To address this, attribution methods have been proposed, though their efficacy remains suboptimal and necessitates further refinement. In this paper, we propose a novel category of transferable adversarial attacks, called transferable frequency-aware attacks, enabling frequency-aware exploration via both high-and low-frequency components. Based on this type of attacks, we also propose a novel attribution method, named Frequency-Aware Model Parameter Explorer (FAMPE), which improves the explainability for DNNs. Relative to the current state-of-the-art method AttEXplore, our FAMPE attains an average gain of 13.02% in Insertion Score, thereby outperforming existing approaches. Through detailed ablation studies, we also investigate the role of both high- and low-frequency components in explainability. 

**Abstract (ZH)**: 确保深度神经网络在现实世界噪声和故意干扰下的可靠性依然是一项重大挑战。为此，已经提出了一些归因方法，但其效果仍然不尽如人意，需要进一步优化。本文提出了一种新型可移植频率感知攻击，称为传输频率感知攻击，能够通过高频和低频成分进行频率感知探索。基于此类攻击，我们还提出了一种新型归因方法——频率感知模型参数探索器（FAMPE），该方法提高了深度神经网络的可解释性。与当前最先进的方法AttEXplore相比，我们的FAMPE在插入分数上平均提高了13.02%，从而优于现有方法。通过详细的消融研究表明，高频和低频成分在可解释性中均发挥着重要作用。 

---
# VIFO: Visual Feature Empowered Multivariate Time Series Forecasting with Cross-Modal Fusion 

**Title (ZH)**: VIFO：视觉特征增强的跨模态融合多变量时间序列预测 

**Authors**: Yanlong Wang, Hang Yu, Jian Xu, Fei Ma, Hongkang Zhang, Tongtong Feng, Zijian Zhang, Shao-Lun Huang, Danny Dongning Sun, Xiao-Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03244)  

**Abstract**: Large time series foundation models often adopt channel-independent architectures to handle varying data dimensions, but this design ignores crucial cross-channel dependencies. Concurrently, existing multimodal approaches have not fully exploited the power of large vision models (LVMs) to interpret spatiotemporal data. Additionally, there remains significant unexplored potential in leveraging the advantages of information extraction from different modalities to enhance time series forecasting performance. To address these gaps, we propose the VIFO, a cross-modal forecasting model. VIFO uniquely renders multivariate time series into image, enabling pre-trained LVM to extract complex cross-channel patterns that are invisible to channel-independent models. These visual features are then aligned and fused with representations from the time series modality. By freezing the LVM and training only 7.45% of its parameters, VIFO achieves competitive performance on multiple benchmarks, offering an efficient and effective solution for capturing cross-variable relationships in 

**Abstract (ZH)**: 跨模态时间序列预测模型VIFO 

---
