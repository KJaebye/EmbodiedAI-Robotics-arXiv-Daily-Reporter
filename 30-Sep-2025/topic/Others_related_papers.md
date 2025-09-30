# SafeFlowMatcher: Safe and Fast Planning using Flow Matching with Control Barrier Functions 

**Title (ZH)**: SafeFlowMatcher：基于流动匹配与控制障碍函数的安全快速规划 

**Authors**: Jeongyong Yang, Seunghwan Jang, Soojean Han  

**Link**: [PDF](https://arxiv.org/pdf/2509.24243)  

**Abstract**: Generative planners based on flow matching (FM) can produce high-quality paths in one or a few ODE steps, but their sampling dynamics offer no formal safety guarantees and can yield incomplete paths near constraints. We present SafeFlowMatcher, a planning framework that couples FM with control barrier functions (CBFs) to achieve both real-time efficiency and certified safety. SafeFlowMatcher uses a two-phase prediction-correction (PC) integrator: (i) a prediction phase integrates the learned FM once (or a few steps) to obtain a candidate path without intervention; (ii) a correction phase refines this path with a vanishing time-scaled vector field and a CBF-based quadratic program that minimally perturbs the vector field. We prove a barrier certificate for the resulting flow system, establishing forward invariance of a robust safe set and finite-time convergence to the safe set. By enforcing safety only on the executed path (rather than on all intermediate latent paths), SafeFlowMatcher avoids distributional drift and mitigates local trap problems. Across maze navigation and locomotion benchmarks, SafeFlowMatcher attains faster, smoother, and safer paths than diffusion- and FM-based baselines. Extensive ablations corroborate the contributions of the PC integrator and the barrier certificate. 

**Abstract (ZH)**: 基于流匹配的生成式规划器结合控制障碍函数的安全流匹配规划框架 

---
# Towards Tighter Convex Relaxation of Mixed-integer Programs: Leveraging Logic Network Flow for Task and Motion Planning 

**Title (ZH)**: 基于逻辑网络流的混合整数规划 tighter 凸松弛方法研究：任务与运动规划中的应用 

**Authors**: Xuan Lin, Jiming Ren, Yandong Luo, Weijun Xie, Ye Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.24235)  

**Abstract**: This paper proposes an optimization-based task and motion planning framework, named "Logic Network Flow", that integrates temporal logic specifications into mixed-integer programs for efficient robot planning. Inspired by the Graph-of-Convex-Sets formulation, temporal predicates are encoded as polyhedron constraints on each edge of a network flow model, instead of as constraints between nodes in traditional Logic Tree formulations. We further propose a network-flow-based Fourier-Motzkin elimination procedure that removes continuous flow variables while preserving convex relaxation tightness, leading to provably tighter convex relaxations and fewer constraints than Logic Tree formulations. For temporal logic motion planning with piecewise-affine dynamic systems, comprehensive experiments across vehicle routing, multi-robot coordination, and temporal logic control on dynamical systems using point mass and linear inverted pendulum models demonstrate computational speedups of up to several orders of magnitude. Hardware demonstrations with quadrupedal robots validate real-time replanning capabilities under dynamically changing environmental conditions. The project website is at this https URL. 

**Abstract (ZH)**: 基于优化的任务与运动规划框架“逻辑网络流”：将时间逻辑规范整合到混合整数规划中用于高效机器人规划 

---
# Ancestry Tree Clustering for Particle Filter Diversity Maintenance 

**Title (ZH)**: 祖先树聚类以维护粒子滤波多样性 

**Authors**: Ilari Vallivaara, Bingnan Duan, Yinhuan Dong, Tughrul Arslan  

**Link**: [PDF](https://arxiv.org/pdf/2509.24124)  

**Abstract**: We propose a method for linear-time diversity maintenance in particle filtering. It clusters particles based on ancestry tree topology: closely related particles in sufficiently large subtrees are grouped together. The main idea is that the tree structure implicitly encodes similarity without the need for spatial or other domain-specific metrics. This approach, when combined with intra-cluster fitness sharing and the protection of particles not included in a cluster, effectively prevents premature convergence in multimodal environments while maintaining estimate compactness. We validate our approach in a multimodal robotics simulation and a real-world multimodal indoor environment. We compare the performance to several diversity maintenance algorithms from the literature, including Deterministic Resampling and Particle Gaussian Mixtures. Our algorithm achieves high success rates with little to no negative effect on compactness, showing particular robustness to different domains and challenging initial conditions. 

**Abstract (ZH)**: 一种线性时间粒子滤波多样性维护方法：基于祖先树拓扑的群组化策略 

---
# MAD-PINN: A Decentralized Physics-Informed Machine Learning Framework for Safe and Optimal Multi-Agent Control 

**Title (ZH)**: MAD-PINN：一种安全且最优的多智能体控制去中心化物理信息机器学习框架 

**Authors**: Manan Tayal, Aditya Singh, Shishir Kolathaya, Somil Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2509.23960)  

**Abstract**: Co-optimizing safety and performance in large-scale multi-agent systems remains a fundamental challenge. Existing approaches based on multi-agent reinforcement learning (MARL), safety filtering, or Model Predictive Control (MPC) either lack strict safety guarantees, suffer from conservatism, or fail to scale effectively. We propose MAD-PINN, a decentralized physics-informed machine learning framework for solving the multi-agent state-constrained optimal control problem (MASC-OCP). Our method leverages an epigraph-based reformulation of SC-OCP to simultaneously capture performance and safety, and approximates its solution via a physics-informed neural network. Scalability is achieved by training the SC-OCP value function on reduced-agent systems and deploying them in a decentralized fashion, where each agent relies only on local observations of its neighbours for decision-making. To further enhance safety and efficiency, we introduce an Hamilton-Jacobi (HJ) reachability-based neighbour selection strategy to prioritize safety-critical interactions, and a receding-horizon policy execution scheme that adapts to dynamic interactions while reducing computational burden. Experiments on multi-agent navigation tasks demonstrate that MAD-PINN achieves superior safety-performance trade-offs, maintains scalability as the number of agents grows, and consistently outperforms state-of-the-art baselines. 

**Abstract (ZH)**: 在大规模多Agent系统中同时优化安全性和性能仍然是一个基本挑战。现有的基于多Agent强化学习（MARL）、安全筛选或模型预测控制（MPC）的方法要么缺乏严格的安全性保证，要么具有保守性，要么无法有效扩展。我们提出了MAD-PINN，这是一种分布式的物理信息机器学习框架，用于解决具有状态约束的多Agent最优控制问题（MASC-OCP）。该方法利用SC-OCP的episode图表形式重新表述来同时捕捉性能和安全性，并通过物理信息神经网络近似其解。通过在缩减Agent的系统中训练SC-OCP价值函数并在分布式方式下部署它们，实现了可扩展性，其中每个Agent仅依赖于其邻居的局部观察来进行决策。为了进一步提高安全性和效率，我们引入了基于Hamilton-Jacobi（HJ）可达性的邻居选择策略来优先处理安全关键的交互，并引入了一种基于后退视界的策略执行方案，该方案能够适应动态交互并减少计算负担。在多Agent导航任务上的实验表明，MAD-PINN实现了更好的安全性和性能 trade-offs，在Agent数量增加时保持了可扩展性，并且一致地优于现有最先进的基线。 

---
# Sequence Pathfinder for Multi-Agent Pickup and Delivery in the Warehouse 

**Title (ZH)**: 仓库中多-agent拾取与交付的序列探索者 

**Authors**: Zeyuan Zhang, Chaoran Li, Shao Zhang, Ying Wen  

**Link**: [PDF](https://arxiv.org/pdf/2509.23778)  

**Abstract**: Multi-Agent Pickup and Delivery (MAPD) is a challenging extension of Multi-Agent Path Finding (MAPF), where agents are required to sequentially complete tasks with fixed-location pickup and delivery demands. Although learning-based methods have made progress in MAPD, they often perform poorly in warehouse-like environments with narrow pathways and long corridors when relying only on local observations for distributed decision-making. Communication learning can alleviate the lack of global information but introduce high computational complexity due to point-to-point communication. To address this challenge, we formulate MAPF as a sequence modeling problem and prove that path-finding policies under sequence modeling possess order-invariant optimality, ensuring its effectiveness in MAPD. Building on this, we propose the Sequential Pathfinder (SePar), which leverages the Transformer paradigm to achieve implicit information exchange, reducing decision-making complexity from exponential to linear while maintaining efficiency and global awareness. Experiments demonstrate that SePar consistently outperforms existing learning-based methods across various MAPF tasks and their variants, and generalizes well to unseen environments. Furthermore, we highlight the necessity of integrating imitation learning in complex maps like warehouses. 

**Abstract (ZH)**: 多代理取送任务（MAPD）是多代理路径规划（MAPF）的一个具有挑战性的扩展，在其中代理需要顺序完成固定位置的取送任务。尽管基于学习的方法在MAPD领域取得了进展，但在依赖局部观察进行分布式决策的仓库-like环境中，它们通常表现不佳，尤其是在狭窄通道和长走廊的环境下。通信学习可以缓解缺乏全局信息的问题，但由于点对点通信导致计算复杂性增加。为解决这一挑战，我们将MAPF形式化为序列建模问题，并证明在序列建模下的路径规划策略具有顺序不变的最优性，确保其在MAPD中的有效性。在此基础上，我们提出了序列路径规划者（SePar），它利用Transformer范式实现隐式信息交换，将决策复杂性从指数级降低到线性级，同时保持高效性和全局意识。实验表明，SePar在各种MAPF任务及其变体中表现优异，并且能够很好地泛化到未见过的环境中。此外，我们强调在复杂地图（如仓库）中整合模仿学习的必要性。 

---
# MDCPP: Multi-robot Dynamic Coverage Path Planning for Workload Adaptation 

**Title (ZH)**: 多机器人动态覆盖路径规划以适应工作负载 

**Authors**: Jun Chen, Mingjia Chen, Shinkyu Park  

**Link**: [PDF](https://arxiv.org/pdf/2509.23705)  

**Abstract**: Multi-robot Coverage Path Planning (MCPP) addresses the problem of computing paths for multiple robots to effectively cover a large area of interest. Conventional approaches to MCPP typically assume that robots move at fixed velocities, which is often unrealistic in real-world applications where robots must adapt their speeds based on the specific coverage tasks assigned to this http URL, conventional approaches often lead to imbalanced workload distribution among robots and increased completion time for coverage tasks. To address this, we introduce a novel Multi-robot Dynamic Coverage Path Planning (MDCPP) algorithm for complete coverage in two-dimensional environments. MDCPP dynamically estimates each robot's remaining workload by approximating the target distribution with Gaussian mixture models, and assigns coverage regions using a capacity-constrained Voronoi diagram. We further develop a distributed implementation of MDCPP for range-constrained robotic networks. Simulation results validate the efficacy of MDCPP, showing qualitative improvements and superior performance compared to an existing sweeping algorithm, and a quantifiable impact of communication range on coverage efficiency. 

**Abstract (ZH)**: 多机器人动态覆盖路径规划（MDCPP）：二维环境中的完全覆盖问题 

---
# Online Dynamic Goal Recognition in Gym Environments 

**Title (ZH)**: 在线动态目标识别在Gym环境中 

**Authors**: Shamir Matan, Elhadad Osher, Nageris Ben, Mirsky Reuth  

**Link**: [PDF](https://arxiv.org/pdf/2509.23244)  

**Abstract**: Goal Recognition (GR) is the task of inferring an agent's intended goal from partial observations of its behavior, typically in an online and one-shot setting. Despite recent advances in model-free GR, particularly in applications such as human-robot interaction, surveillance, and assistive systems, the field remains fragmented due to inconsistencies in benchmarks, domains, and evaluation protocols.
To address this, we introduce gr-libs (this https URL) and gr-envs (this https URL), two complementary open-source frameworks that support the development, evaluation, and comparison of GR algorithms in Gym-compatible environments. gr-libs includes modular implementations of MDP-based GR baselines, diagnostic tools, and evaluation utilities. gr-envs provides a curated suite of environments adapted for dynamic and goal-directed behavior, along with wrappers that ensure compatibility with standard reinforcement learning toolkits. Together, these libraries offer a standardized, extensible, and reproducible platform for advancing GR research. Both packages are open-source and available on GitHub and PyPI. 

**Abstract (ZH)**: Goal Recognition: A Standardized Framework for Developing and Evaluating Goal Recognition Algorithms 

---
# DBF-MA: A Differential Bayesian Filtering Planner for Multi-Agent Autonomous Racing Overtakes 

**Title (ZH)**: DBF-MA: 一种用于多Agent自主竞速超车的差分贝叶斯过滤规划器 

**Authors**: Trent Weiss, Amar Kulkarni, Madhur Behl  

**Link**: [PDF](https://arxiv.org/pdf/2509.22937)  

**Abstract**: A significant challenge in autonomous racing is to generate overtaking maneuvers. Racing agents must execute these maneuvers on complex racetracks with little room for error. Optimization techniques and graph-based methods have been proposed, but these methods often rely on oversimplified assumptions for collision-avoidance and dynamic constraints. In this work, we present an approach to trajectory synthesis based on an extension of the Differential Bayesian Filtering framework. Our approach for collision-free trajectory synthesis frames the problem as one of Bayesian Inference over the space of Composite Bezier Curves. Our method is derivative-free, does not require a spherical approximation of the vehicle footprint, linearization of constraints, or simplifying upper bounds on collision avoidance. We conduct a closed-loop analysis of DBF-MA and find it successfully overtakes an opponent in 87% of tested scenarios, outperforming existing methods in autonomous overtaking. 

**Abstract (ZH)**: 自主赛车中的一个重大挑战是如何生成超越 maneuvers。赛车代理必须在复杂赛道上执行这些 maneuvers，并且几乎没有错误余地。已经提出了优化技术和图基方法，但这些方法往往依赖于碰撞避免和动力学约束的过度简化假设。在本工作中，我们提出了一种基于差分贝叶斯滤波框架扩展的方法来合成轨迹。我们的碰撞自由轨迹合成方法将问题建模为贝叶斯推理在复合贝塞尔曲线空间中的问题。我们的方法无需导数、不需要车辆足迹的球形近似、无需约束线性化或碰撞避免的简化上界。我们对DBF-MA进行了闭环分析，发现该方法在测试场景中有87%的情况下成功超越对手，优于现有方法在自主超越方面的表现。 

---
# Multi-Robot Allocation for Information Gathering in Non-Uniform Spatiotemporal Environments 

**Title (ZH)**: 非均匀时空环境中的多机器人信息采集分配 

**Authors**: Kaleb Ben Naveed, Haejoon Lee, Dimitra Panagou  

**Link**: [PDF](https://arxiv.org/pdf/2509.22883)  

**Abstract**: Autonomous robots are increasingly deployed to estimate spatiotemporal fields (e.g., wind, temperature, gas concentration) that vary across space and time. We consider environments divided into non-overlapping regions with distinct spatial and temporal dynamics, termed non-uniform spatiotemporal environments. Gaussian Processes (GPs) can be used to estimate these fields. The GP model depends on a kernel that encodes how the field co-varies in space and time, with its spatial and temporal lengthscales defining the correlation. Hence, when these lengthscales are incorrect or do not correspond to the actual field, the estimates of uncertainty can be highly inaccurate. Existing GP methods often assume one global lengthscale or update only periodically; some allow spatial variation but ignore temporal changes. To address these limitations, we propose a two-phase framework for multi-robot field estimation. Phase 1 uses a variogram-driven planner to learn region-specific spatial lengthscales. Phase 2 employs an allocation strategy that reassigns robots based on the current uncertainty, and updates sampling as temporal lengthscales are refined. For encoding uncertainty, we utilize clarity, an information metric from our earlier work. We evaluate the proposed method across diverse environments and provide convergence analysis for spatial lengthscale estimation, along with dynamic regret bounds quantifying the gap to the oracle's allocation sequence. 

**Abstract (ZH)**: 自主机器人在非均匀时空环境下的场估计中得到了越来越广泛的应用。我们考虑将环境划分为不重叠的具有不同时空动态的区域，称为非均匀时空环境。高斯过程（GPs）可以用于估计这些场。GP模型依赖于一个内核，该内核编码了场在时空中的协变关系，其时空长度尺度定义了相关性。因此，当这些长度尺度不正确或不对应于实际场时，不确定性估计可能会非常不准确。现有的GP方法通常假设一个全局长度尺度或仅周期性更新；一些方法允许空间变异性但忽略时间变化。为了解决这些限制，我们提出了一种两阶段框架进行多机器人场估计。第一阶段使用变异函数驱动的规划器学习区域特定的空间长度尺度。第二阶段采用分配策略根据当前不确定性重新分配机器人，并随着时空长度尺度的细化更新采样。为了编码不确定性的信息，我们利用了我们之前工作中提出的清晰度这一信息度量。我们在多种环境中评估了提出的方法，并提供了空间长度尺度估计的收敛性分析，以及衡量到最优分配序列差距的动态遗憾界。 

---
# Large Language Models for 3D IC Space Planning 

**Title (ZH)**: 大型语言模型在3D IC空间规划中的应用 

**Authors**: Hung-Ying Chu, Guan-Wei Chen, Shao-Yu Wei, Yu-Cheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.22716)  

**Abstract**: Three-dimensional integrated circuits (3D ICs) have emerged as a promising solution to the scaling limits of two-dimensional designs, offering higher integration density, shorter interconnects, and improved performance. As design complexity increases, effective space planning becomes essential to reduce dead space and ensure layout quality. This study investigates the use of large language models (LLMs) for 3D IC space planning through a post-order slicing tree representation, which guarantees legal space plans while aiming to minimize dead space. Open-source LLMs were fine-tuned on large-scale synthetic datasets and further evaluated on MCNC-derived 3D benchmarks. Experimental results indicate that the proposed framework achieves a favorable balance between runtime efficiency, legality, and dead-space reduction, with zero-dead-space layouts obtained in a significant portion of test cases under practical runtime budgets. Beyond synthetic benchmarks, the method generalizes to MCNC cases such as ami33 and ami49, though larger and irregular instances remain challenging. The approach also shows potential for cross-domain applications, including logistics and 3D object placement, where spatial efficiency is critical. Overall, the results suggest that LLM-based space planning can serve as a data-driven complement to traditional electronic design automation (EDA) methods, providing new insights for scalable 3D layout generation. 

**Abstract (ZH)**: 三维集成电路（3D ICs）的空间规划通过后序切片树表示利用大型语言模型（LLMs）的研究：实现高效的合法性、减少死空间的平衡 

---
# Safety-Critical Input-Constrained Nonlinear Intercept Guidance in Multiple Engagement Zones 

**Title (ZH)**: 多作战区内的安全关键输入约束非线性截获制导 

**Authors**: Praveen Kumar Ranjan, Abhinav Sinha, Yongcan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2509.25053)  

**Abstract**: This paper presents an input-constrained nonlinear guidance law to address the problem of intercepting a stationary target in contested environments with multiple defending agents. Contrary to prior approaches that rely on explicit knowledge of defender strategies or utilize conservative safety conditions based on a defender's range, our work characterizes defender threats geometrically through engagement zones that delineate inevitable interception regions. Outside these engagement zones, the interceptor remains invulnerable. The proposed guidance law switches between a repulsive safety maneuver near these zones and a pursuit maneuver outside their influence. To deal with multiple engagement zones, we employ a smooth minimum function (log-sum-exponent approximation) that aggregates threats from all the zones while prioritizing the most critical threats. Input saturation is modeled and embedded in the non-holonomic vehicle dynamics so the controller respects actuator limits while maintaining stability. Numerical simulations with several defenders demonstrate the proposed method's ability to avoid engagement zones and achieve interception across diverse initial conditions. 

**Abstract (ZH)**: 基于输入约束的非线性制导律以应对多防护实体的交战区环境下对静止目标的拦截问题 

---
# Discrete Variational Autoencoding via Policy Search 

**Title (ZH)**: 离散变分自编码通过策略搜索 

**Authors**: Michael Drolet, Firas Al-Hafez, Aditya Bhatt, Jan Peters, Oleg Arenz  

**Link**: [PDF](https://arxiv.org/pdf/2509.24716)  

**Abstract**: Discrete latent bottlenecks in variational autoencoders (VAEs) offer high bit efficiency and can be modeled with autoregressive discrete distributions, enabling parameter-efficient multimodal search with transformers. However, discrete random variables do not allow for exact differentiable parameterization; therefore, discrete VAEs typically rely on approximations, such as Gumbel-Softmax reparameterization or straight-through gradient estimates, or employ high-variance gradient-free methods such as REINFORCE that have had limited success on high-dimensional tasks such as image reconstruction. Inspired by popular techniques in policy search, we propose a training framework for discrete VAEs that leverages the natural gradient of a non-parametric encoder to update the parametric encoder without requiring reparameterization. Our method, combined with automatic step size adaptation and a transformer-based encoder, scales to challenging datasets such as ImageNet and outperforms both approximate reparameterization methods and quantization-based discrete autoencoders in reconstructing high-dimensional data from compact latent spaces, achieving a 20% improvement on FID Score for ImageNet 256. 

**Abstract (ZH)**: 离散潜瓶颈在变分自编码器中的应用提供了高比特效率，并可以通过自回归离散分布进行建模，从而可以用变压器实现参数高效的多模态搜索。然而，离散随机变量不允许精确的可微参数化；因此，离散变分自编码器通常依赖于近似方法，如Gumbel-Softmax重参数化或直接通过梯度估计，或者使用高方差的无梯度方法如REINFORCE，这些方法在如图像重建等高维任务上效果有限。受政策搜索中流行技术的启发，我们提出了一种离散变分自编码器的训练框架，利用非参数编码器的自然梯度来更新参数编码器，无需重参数化。结合自适应步长调整和基于变压器的编码器，该方法可以扩展到如ImageNet这样的具有挑战性的数据集，并在从紧凑的潜空间重构高维数据方面优于近似重参数化方法和基于量化的方法，实现了ImageNet 256在FID分数上的20%改进。 

---
# Clebsch-Gordan Transformer: Fast and Global Equivariant Attention 

**Title (ZH)**: Clebsch-Gordan 变体变压器：快速且全局 equivariant 注意力 

**Authors**: Owen Lewis Howell, Linfeng Zhao, Xupeng Zhu, Yaoyao Qian, Haojie Huang, Lingfeng Sun, Wil Thomason, Robert Platt, Robin Walters  

**Link**: [PDF](https://arxiv.org/pdf/2509.24093)  

**Abstract**: The global attention mechanism is one of the keys to the success of transformer architecture, but it incurs quadratic computational costs in relation to the number of tokens. On the other hand, equivariant models, which leverage the underlying geometric structures of problem instance, often achieve superior accuracy in physical, biochemical, computer vision, and robotic tasks, at the cost of additional compute requirements. As a result, existing equivariant transformers only support low-order equivariant features and local context windows, limiting their expressiveness and performance. This work proposes Clebsch-Gordan Transformer, achieving efficient global attention by a novel Clebsch-Gordon Convolution on $\SO(3)$ irreducible representations. Our method enables equivariant modeling of features at all orders while achieving ${O}(N \log N)$ input token complexity. Additionally, the proposed method scales well with high-order irreducible features, by exploiting the sparsity of the Clebsch-Gordon matrix. Lastly, we also incorporate optional token permutation equivariance through either weight sharing or data augmentation. We benchmark our method on a diverse set of benchmarks including n-body simulation, QM9, ModelNet point cloud classification and a robotic grasping dataset, showing clear gains over existing equivariant transformers in GPU memory size, speed, and accuracy. 

**Abstract (ZH)**: Clebsch-Gordan Transformer：基于$\SO(3)$不可约表示的新颖Clebsch-Gordan卷积实现高效全局注意力 

---
# Systematic Alias Sampling: an efficient and low-variance way to sample from a discrete distribution 

**Title (ZH)**: 系统化的别名采样：一种高效且低方差的离散分布采样方法 

**Authors**: Ilari Vallivaara, Katja Poikselkä, Pauli Rikula, Juha Röning  

**Link**: [PDF](https://arxiv.org/pdf/2509.24089)  

**Abstract**: In this paper we combine the Alias method with the concept of systematic sampling, a method commonly used in particle filters for efficient low-variance resampling. The proposed method allows very fast sampling from a discrete distribution: drawing k samples is up to an order of magnitude faster than binary search from the cumulative distribution function (cdf) or inversion methods used in many libraries. The produced empirical distribution function is evaluated using a modified Cramér-Von Mises goodness-of-fit statistic, showing that the method compares very favourably to multinomial sampling. As continuous distributions can often be approximated with discrete ones, the proposed method can be used as a very general way to efficiently produce random samples for particle filter proposal distributions, e.g. for motion models in robotics. 

**Abstract (ZH)**: 本文将Alias方法与系统抽样概念结合，用于粒子滤波中的高效低方差重采样。所提出的方法允许从离散分布中进行非常快速的抽样：抽取k个样本的速度比从累积分布函数（CDF）或许多库中使用的倒置方法快一个数量级。通过使用修改后的Cramér-Von Mises拟合优度统计评估生成的经验分布函数，表明该方法与多项式抽样相比具有很大的优势。由于连续分布往往可以用离散分布逼近，所提出的方法可以作为一种非常通用的方法，用于高效地为粒子滤波的提议分布生成随机样本，例如在机器人中的运动模型。 

---
# Advancing Multi-agent Traffic Simulation via R1-Style Reinforcement Fine-Tuning 

**Title (ZH)**: 基于R1风格强化学习微调的多agents交通模拟推进 

**Authors**: Muleilan Pei, Shaoshuai Shi, Shaojie Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.23993)  

**Abstract**: Scalable and realistic simulation of multi-agent traffic behavior is critical for advancing autonomous driving technologies. Although existing data-driven simulators have made significant strides in this domain, they predominantly rely on supervised learning to align simulated distributions with real-world driving scenarios. A persistent challenge, however, lies in the distributional shift that arises between training and testing, which often undermines model generalization in unseen environments. To address this limitation, we propose SMART-R1, a novel R1-style reinforcement fine-tuning paradigm tailored for next-token prediction models to better align agent behavior with human preferences and evaluation metrics. Our approach introduces a metric-oriented policy optimization algorithm to improve distribution alignment and an iterative "SFT-RFT-SFT" training strategy that alternates between Supervised Fine-Tuning (SFT) and Reinforcement Fine-Tuning (RFT) to maximize performance gains. Extensive experiments on the large-scale Waymo Open Motion Dataset (WOMD) validate the effectiveness of this simple yet powerful R1-style training framework in enhancing foundation models. The results on the Waymo Open Sim Agents Challenge (WOSAC) showcase that SMART-R1 achieves state-of-the-art performance with an overall realism meta score of 0.7858, ranking first on the leaderboard at the time of submission. 

**Abstract (ZH)**: 适用大规模且真实的多智能体交通行为仿真对于推动自动驾驶技术的发展至关重要。尽管现有的数据驱动仿真器在此领域取得了显著进展，它们主要依赖监督学习来对齐仿真分布与现实驾驶场景。然而，训练与测试之间持续存在的分布偏差往往削弱了模型在未见环境中的泛化能力。为解决这一限制，我们提出SMART-R1，一种针对下一标记预测模型的新型R1风格强化微调范式，以更好地使智能体行为与人类偏好和评估指标保持一致。我们的方法引入了一种以度量为导向的策略优化算法，以提高分布对齐，并提出了一种迭代的“SFT-RFT-SFT”训练策略，交替进行监督微调(SFT)和强化微调(RFT)，以最大化性能提升。大规模Waymo Open Motion Dataset (WOMD)上的广泛实验验证了这种简单而强大的R1风格训练框架在增强基础模型方面的有效性。Waymo Open Sim Agents Challenge (WOSAC)上的结果表明，SMART-R1 达到了最先进的性能，总体现实度meta分为0.7858，在提交时排名领导者榜第一。 

---
# From Static to Dynamic: a Survey of Topology-Aware Perception in Autonomous Driving 

**Title (ZH)**: 从静态到动态：自主驾驶中拓扑感知综述 

**Authors**: Yixiao Chen, Ruining Yang, Xin Chen, Jia He, Dongliang Xu, Yue Yao  

**Link**: [PDF](https://arxiv.org/pdf/2509.23641)  

**Abstract**: The key to achieving autonomous driving lies in topology-aware perception, the structured understanding of the driving environment with an emphasis on lane topology and road semantics. This survey systematically reviews four core research directions under this theme: vectorized map construction, topological structure modeling, prior knowledge fusion, and language model-based perception. Across these directions, we observe a unifying trend: a paradigm shift from static, pre-built maps to dynamic, sensor-driven perception. Specifically, traditional static maps have provided semantic context for autonomous systems. However, they are costly to construct, difficult to update in real time, and lack generalization across regions, limiting their scalability. In contrast, dynamic representations leverage on-board sensor data for real-time map construction and topology reasoning. Each of the four research directions contributes to this shift through compact spatial modeling, semantic relational reasoning, robust domain knowledge integration, and multimodal scene understanding powered by pre-trained language models. Together, they pave the way for more adaptive, scalable, and explainable autonomous driving systems. 

**Abstract (ZH)**: 实现自动驾驶的关键在于拓扑感知，即以车道拓扑和道路语义为重点的驾驶环境的结构化理解。本文综述了该主题下的四大核心研究方向：矢量地图构建、拓扑结构建模、先验知识融合以及基于语言模型的感知。在这四大方向中，我们观察到一个统一的趋势：从静态、预先构建的地图向基于传感器的动态感知的范式转变。传统静态地图为自主系统提供了语义上下文，但构建成本高、难以实时更新且跨区域缺乏泛化能力，限制了其适用性。相比之下，动态表示利用车载传感器数据实现实时地图构建和拓扑推理。四大研究方向分别通过紧凑的空间建模、语义关系推理、鲁棒领域知识融合以及基于预训练语言模型的多模态场景理解促进这一转变。这些研究共同为更加适应环境、可扩展且可解释的自动驾驶系统铺平了道路。 

---
# Visual serial processing deficits explain divergences in human and VLM reasoning 

**Title (ZH)**: 视觉序列加工缺陷解释人类与VLM推理的差异 

**Authors**: Nicholas Budny, Kia Ghods, Declan Campbell, Raja Marjieh, Amogh Joshi, Sreejan Kumar, Jonathan D. Cohen, Taylor W. Webb, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2509.25142)  

**Abstract**: Why do Vision Language Models (VLMs), despite success on standard benchmarks, often fail to match human performance on surprisingly simple visual reasoning tasks? While the underlying computational principles are still debated, we hypothesize that a crucial factor is a deficit in visually-grounded serial processing. To test this hypothesis, we compared human and VLM performance across tasks designed to vary serial processing demands in three distinct domains: geometric reasoning, perceptual enumeration, and mental rotation. Tasks within each domain varied serial processing load by manipulating factors such as geometric concept complexity, perceptual individuation load, and transformation difficulty. Across all domains, our results revealed a consistent pattern: decreased VLM accuracy was strongly correlated with increased human reaction time (used as a proxy for serial processing load). As tasks require more demanding serial processing -- whether composing concepts, enumerating items, or performing mental transformations -- the VLM-human performance gap widens reliably. These findings support our hypothesis, indicating that limitations in serial, visually grounded reasoning represent a fundamental bottleneck that distinguishes current VLMs from humans. 

**Abstract (ZH)**: 为什么视觉语言模型（VLMs）尽管在标准基准测试中取得成功，但在一些出人意料的简单视觉推理任务中往往无法匹配人类的表现？虽然核心计算原理仍然存在争议，但我们假设一个关键因素是在视觉支撑的序贯处理方面存在缺陷。为了检验这一假设，我们在三个不同的领域（几何推理、知觉计数和心理旋转）设计的任务中比较了人类和VLM的表现，这些任务旨在改变序贯处理的要求。在每个领域内，通过操控诸如几何概念复杂性、知觉个体化负担和变换难度等因素来改变序贯处理负载。在所有领域中，我们的结果显现出了一个一致的模式：VLM准确率的下降强烈相关于人类反应时间的增加（作为序贯处理负载的代理）。随着任务对序贯处理的要求变得更为苛刻——无论是组成概念、计数物品还是执行心理变换——VLM与人类的表现差距会可靠地扩大。这些发现支持了我们的假设，表明在序列、视觉支撑的推理方面的局限性构成了当前VLMs与人类之间的一个基本瓶颈。 

---
# HeDA: An Intelligent Agent System for Heatwave Risk Discovery through Automated Knowledge Graph Construction and Multi-layer Risk Propagation Analysis 

**Title (ZH)**: HeDA：一种通过自动化知识图构建和多层风险传播分析的热波风险发现智能代理系统 

**Authors**: Yiquan Wang, Tin-Yeh Huang, Qingyun Gao, Jialin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.25112)  

**Abstract**: Heatwaves pose complex cascading risks across interconnected climate, social, and economic systems, but knowledge fragmentation in scientific literature hinders comprehensive understanding of these risk pathways. We introduce HeDA (Heatwave Discovery Agent), an intelligent multi-agent system designed for automated scientific discovery through knowledge graph construction and multi-layer risk propagation analysis. HeDA processes over 10,247 academic papers to construct a comprehensive knowledge graph with 23,156 nodes and 89,472 relationships, employing novel multi-layer risk propagation analysis to systematically identify overlooked risk transmission pathways. Our system achieves 78.9% accuracy on complex question-answering tasks, outperforming state-of-the-art baselines including GPT-4 by 13.7%. Critically, HeDA successfully discovered five previously unidentified high-impact risk chains, such as the pathway where a heatwave leads to a water demand surge, resulting in industrial water restrictions and ultimately causing small business disruption, which were validated through historical case studies and domain expert review. This work presents a new paradigm for AI-driven scientific discovery, providing actionable insights for developing more resilient climate adaptation strategies. 

**Abstract (ZH)**: 热浪在互联的气候、社会和经济系统中构成复杂连锁风险，但科学文献中的知识碎片化阻碍了对这些风险路径的全面理解。我们介绍了HeDA（热浪发现代理），这是一种智能多agent系统，设计用于通过知识图谱构建和多层次风险传播分析进行自动科学研究。HeDA 处理了超过10,247篇学术论文，构建了一个包含23,156个节点和89,472个关系的全面知识图谱，并采用新颖的多层次风险传播分析系统地识别出被忽视的风险传输路径。我们的系统在复杂问答任务中的准确率达到78.9%，在包括GPT-4在内的最新基线方法中表现最佳，超过了13.7%。更重要的是，HeDA 成功发现了五个先前未被识别的高影响风险链路，例如热浪导致水资源需求激增，进而引发工业用水限制，最终导致小型企业中断等链条，这些发现通过历史案例研究和领域专家评审得到了验证。这项工作确立了AI驱动科学研究的新范式，为制定更具弹性的气候适应策略提供了可操作的见解。 

---
# KIRETT - A wearable device to support rescue operations using artificial intelligence to improve first aid 

**Title (ZH)**: KIRETT - 一种使用人工智能支持救援行动的可穿戴设备以改进急救 

**Authors**: Johannes Zenkert, Christian Weber, Mubaris Nadeem, Lisa Bender, Madjid Fathi, Abu Shad Ahammed, Aniebiet Micheal Ezekiel, Roman Obermaisser, Maximilian Bradford  

**Link**: [PDF](https://arxiv.org/pdf/2509.24934)  

**Abstract**: This short paper presents first steps in the scientific part of the KIRETT project, which aims to improve first aid during rescue operations using a wearable device. The wearable is used for computer-aided situation recognition by means of artificial intelligence. It provides contextual recommendations for actions and operations to rescue personnel and is intended to minimize damage to patients due to incorrect treatment, as well as increase the probability of survival. The paper describes a first overview of research approaches within the project. 

**Abstract (ZH)**: 这篇简短的文章介绍了KIRETT项目科学部分的初步成果，该项目旨在通过穿戴设备改善救援操作中的急救措施。穿戴设备利用人工智能进行计算机辅助情况识别，并提供上下文相关的行动和操作建议，旨在因不正确的治疗减少患者损伤，提高生存概率。文章描述了项目内的初步研究方法 overview。 

---
# Meta-Learning Theory-Informed Inductive Biases using Deep Kernel Gaussian Processes 

**Title (ZH)**: 基于深度内核高斯过程的理论指导归纳偏置元学习 

**Authors**: Bahti Zakirov, Gašper Tkačik  

**Link**: [PDF](https://arxiv.org/pdf/2509.24919)  

**Abstract**: Normative and task-driven theories offer powerful top-down explanations for biological systems, yet the goals of quantitatively arbitrating between competing theories, and utilizing them as inductive biases to improve data-driven fits of real biological datasets are prohibitively laborious, and often impossible. To this end, we introduce a Bayesian meta-learning framework designed to automatically convert raw functional predictions from normative theories into tractable probabilistic models. We employ adaptive deep kernel Gaussian processes, meta-learning a kernel on synthetic data generated from a normative theory. This Theory-Informed Kernel specifies a probabilistic model representing the theory predictions -- usable for both fitting data and rigorously validating the theory. As a demonstration, we apply our framework to the early visual system, using efficient coding as our normative theory. We show improved response prediction accuracy in ex vivo recordings of mouse retinal ganglion cells stimulated by natural scenes compared to conventional data-driven baselines, while providing well-calibrated uncertainty estimates and interpretable representations. Using exact Bayesian model selection, we also show that our informed kernel can accurately infer the degree of theory-match from data, confirming faithful encapsulation of theory structure. This work provides a more general, scalable, and automated approach for integrating theoretical knowledge into data-driven scientific inquiry in neuroscience and beyond. 

**Abstract (ZH)**: 基于规范和任务驱动理论的贝叶斯元学习框架：自动将功能预测转化为可处理的概率模型，并用于数据驱动的生物学数据拟合和理论验证 

---
# Neural network embeddings recover value dimensions from psychometric survey items on par with human data 

**Title (ZH)**: 神经网络嵌入可以从心理测量调查项目中恢复价值维度，效果媲大人类数据。 

**Authors**: Max Pellert, Clemens M. Lechner, Indira Sen, Markus Strohmaier  

**Link**: [PDF](https://arxiv.org/pdf/2509.24906)  

**Abstract**: This study introduces "Survey and Questionnaire Item Embeddings Differentials" (SQuID), a novel methodological approach that enables neural network embeddings to effectively recover latent dimensions from psychometric survey items. We demonstrate that embeddings derived from large language models, when processed with SQuID, can recover the structure of human values obtained from human rater judgments on the Revised Portrait Value Questionnaire (PVQ-RR). Our experimental validation compares multiple embedding models across a number of evaluation metrics. Unlike previous approaches, SQuID successfully addresses the challenge of obtaining negative correlations between dimensions without requiring domain-specific fine-tuning. Quantitative analysis reveals that our embedding-based approach explains 55% of variance in dimension-dimension similarities compared to human data. Multidimensional scaling configurations from both types of data show fair factor congruence coefficients and largely follow the underlying theory. These results demonstrate that semantic embeddings can effectively replicate psychometric structures previously established through extensive human surveys. The approach offers substantial advantages in cost, scalability and flexibility while maintaining comparable quality to traditional methods. Our findings have significant implications for psychometrics and social science research, providing a complementary methodology that could expand the scope of human behavior and experience represented in measurement tools. 

**Abstract (ZH)**: "Survey和问卷项目嵌入差异性：SQuID方法及其在心理测量维度恢复中的应用" 

---
# PhysicsMinions: Winning Gold Medals in the Latest Physics Olympiads with a Coevolutionary Multimodal Multi-Agent System 

**Title (ZH)**: PhysicsMinions：在最新物理奥赛中使用共进化多模态多智能体系统夺金🏆 

**Authors**: Fangchen Yu, Junchi Yao, Ziyi Wang, Haiyuan Wan, Youling Huang, Bo Zhang, Shuyue Hu, Dongzhan Zhou, Ning Ding, Ganqu Cui, Lei Bai, Wanli Ouyang, Peng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2509.24855)  

**Abstract**: Physics is central to understanding and shaping the real world, and the ability to solve physics problems is a key indicator of real-world physical intelligence. Physics Olympiads, renowned as the crown of competitive physics, provide a rigorous testbed requiring complex reasoning and deep multimodal understanding, yet they remain largely underexplored in AI research. Existing approaches are predominantly single-model based, and open-source MLLMs rarely reach gold-medal-level performance. To address this gap, we propose PhysicsMinions, a coevolutionary multi-agent system for Physics Olympiad. Its architecture features three synergistic studios: a Visual Studio to interpret diagrams, a Logic Studio to formulate solutions, and a Review Studio to perform dual-stage verification. The system coevolves through an iterative refinement loop where feedback from the Review Studio continuously guides the Logic Studio, enabling the system to self-correct and converge towards the ground truth. Evaluated on the HiPhO benchmark spanning 7 latest physics Olympiads, PhysicsMinions delivers three major breakthroughs: (i) Strong generalization: it consistently improves both open-source and closed-source models of different sizes, delivering clear benefits over their single-model baselines; (ii) Historic breakthroughs: it elevates open-source models from only 1-2 to 6 gold medals across 7 Olympiads, achieving the first-ever open-source gold medal in the latest International Physics Olympiad (IPhO) under the average-score metric; and (iii) Scaling to human expert: it further advances the open-source Pass@32 score to 26.8/30 points on the latest IPhO, ranking 4th of 406 contestants and far surpassing the top single-model score of 22.7 (ranked 22nd). Generally, PhysicsMinions offers a generalizable framework for Olympiad-level problem solving, with the potential to extend across disciplines. 

**Abstract (ZH)**: 物理学是理解并塑造现实世界的关键，解决物理学问题的能力是现实物理智能的重要指标。物理学奥林匹克竞赛被誉为竞赛物理学的皇冠，提供了一个需要复杂推理和深刻多模态理解的严格测试平台，但在人工智能研究中仍 largely 欠开发。现有方法主要基于单一模型，开源 MLLMs 很少达到金牌水平。为解决这一差距，我们提出了 PhysicsMinions，一种用于物理学奥林匹克竞赛的协同进化多智能体系统。其架构包括三个协同的工作室：可视化工作室以解析图表、逻辑工作室以制定解决方案、审核工作室以进行双重验证。该系统通过一个迭代细化循环协同进化，在此过程中，来自审核工作室的反馈不断引导逻辑工作室，使系统能够自我纠正并朝着真实答案收敛。在涵盖 7 场最新物理奥林匹克竞赛的 HiPhO 底线标准上，PhysicsMinions 实现了三项重大突破：（i）强大的泛化能力：它一致地提升了不同规模的开源和封闭源模型，对单一模型基线有明显优势；（ii）历史突破：它将开源模型从仅获得 1 到 2 金牌提升至在 7 场奥林匹克竞赛中获得 6 金牌，首次在平均分标准下实现了开源金牌，在最新的国际物理奥林匹克竞赛（IPhO）中取得突破；（iii）达到人类专家水平：进一步将开源 Pass@32 得分提升至最新 IPhO 的 26.8/30 分，在 406 名参赛者中排名第 4，并远超排名第一的单一模型得分 22.7（排名第 22）。总体而言，PhysicsMinions 提供了一个可泛化的框架，用于奥林匹克级别问题解决，具有跨学科扩展的潜力。 

---
# Spatial-Functional awareness Transformer-based graph archetype contrastive learning for Decoding Visual Neural Representations from EEG 

**Title (ZH)**: 基于空间-功能感知的Transformer图原型对比学习方法从EEG解码视觉神经表示 

**Authors**: Yueming Sun, Long Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24761)  

**Abstract**: Decoding visual neural representations from Electroencephalography (EEG) signals remains a formidable challenge due to their high-dimensional, noisy, and non-Euclidean nature. In this work, we propose a Spatial-Functional Awareness Transformer-based Graph Archetype Contrastive Learning (SFTG) framework to enhance EEG-based visual decoding. Specifically, we introduce the EEG Graph Transformer (EGT), a novel graph-based neural architecture that simultaneously encodes spatial brain connectivity and temporal neural dynamics. To mitigate high intra-subject variability, we propose Graph Archetype Contrastive Learning (GAC), which learns subject-specific EEG graph archetypes to improve feature consistency and class separability. Furthermore, we conduct comprehensive subject-dependent and subject-independent evaluations on the Things-EEG dataset, demonstrating that our approach significantly outperforms prior state-of-the-art EEG decoding this http URL results underscore the transformative potential of integrating graph-based learning with contrastive objectives to enhance EEG-based brain decoding, paving the way for more generalizable and robust neural representations. 

**Abstract (ZH)**: 从脑电图（EEG）信号解码视觉神经表征是一项艰巨的挑战，由于其高维、噪声和非欧几里得特性。在此工作中，我们提出了一种基于空间-功能意识变换器的图原型对比学习（SFTG）框架，以增强基于EEG的视觉解码。具体而言，我们引入了脑电图变换器（EGT），这是一种新颖的基于图的神经架构，能够同时编码空间脑连接性和时间神经动力学。为了减轻高被试内变异性，我们提出了图原型对比学习（GAC），以学习被试特异性的EEG图原型，从而提高特征一致性并增强类别可分性。此外，我们在Things-EEG数据集中进行了全面的被试依赖性和被试独立性评估，结果表明我们的方法显著优于先前的最佳EEG解码方法。这些结果强调了将基于图的学习与对比目标集成以增强基于EEG的大脑解码的变革潜力，为更泛化和稳健的神经表示开辟了道路。 

---
# Successful Misunderstandings: Learning to Coordinate Without Being Understood 

**Title (ZH)**: 成功的误解：学习在不被理解的情况下协调 

**Authors**: Nikolaos Kondylidis, Anil Yaman, Frank van Harmelen, Erman Acar, Annette ten Teije  

**Link**: [PDF](https://arxiv.org/pdf/2509.24660)  

**Abstract**: The main approach to evaluating communication is by assessing how well it facilitates coordination. If two or more individuals can coordinate through communication, it is generally assumed that they understand one another. We investigate this assumption in a signaling game where individuals develop a new vocabulary of signals to coordinate successfully. In our game, the individuals do not have common observations besides the communication signal and outcome of the interaction, i.e. received reward. This setting is used as a proxy to study communication emergence in populations of agents that perceive their environment very differently, e.g. hybrid populations that include humans and artificial agents. Agents develop signals, use them, and refine interpretations while not observing how other agents are using them. While populations always converge to optimal levels of coordination, in some cases, interacting agents interpret and use signals differently, converging to what we call successful misunderstandings. However, agents of population that coordinate using misaligned interpretations, are unable to establish successful coordination with new interaction partners. Not leading to coordination failure immediately, successful misunderstandings are difficult to spot and repair. Having at least three agents that all interact with each other are the two minimum conditions to ensure the emergence of shared interpretations. Under these conditions, the agent population exhibits this emergent property of compensating for the lack of shared observations of signal use, ensuring the emergence of shared interpretations. 

**Abstract (ZH)**: 评估通信的主要方法是考察其促进协调的效果。如果两人或多人能够通过通信协调一致，通常假定他们相互理解。我们在一个信号游戏中研究这一假设，该游戏中个体发展出新的信号词汇以成功协调。在游戏中，个体除了通信信号和交互结果（即获得的奖励）之外没有其他共同观察。这一设定被用作代理，以研究在感知环境差异极大的代理种群中通信的涌现。代理开发信号、使用信号并改进解释，但不观察其他代理如何使用。尽管种群总是会收敛到最优的协调水平，但在某些情况下，交互代理会以不同的方式解释和使用信号，最终达到我们称之为成功的误解。然而，使用不一致解释进行协调的代理种群无法与新的交互伙伴建立有效的协调。成功的误解不容易被发现和修正，直到协调失败才显现出来。至少有三个相互作用的代理是确保共享解释涌现的两个最小条件。在这些条件下，代理种群表现出补偿缺乏信号使用共享观察的涌现特性，从而确保共享解释的涌现。 

---
# "Stop replacing salt with sugar!'': Towards Intuitive Human-Agent Teaching 

**Title (ZH)**: “停止用糖替代盐!”：向着直觉的人机教学 

**Authors**: Nikolaos Kondylidis, Andrea Rafanelli, Ilaria Tiddi, Annette ten Teije, Frank van Harmelen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24651)  

**Abstract**: Humans quickly learn new concepts from a small number of examples. Replicating this capacity with Artificial Intelligence (AI) systems has proven to be challenging. When it comes to learning subjective tasks-where there is an evident scarcity of data-this capacity needs to be recreated. In this work, we propose an intuitive human-agent teaching architecture in which the human can teach an agent how to perform a task by providing demonstrations, i.e., examples. To have an intuitive interaction, we argue that the agent should be able to learn incrementally from a few single examples. To allow for this, our objective is to broaden the agent's task understanding using domain knowledge. Then, using a learning method to enable the agent to learn efficiently from a limited number of examples. Finally, to optimize how human can select the most representative and less redundant examples to provide the agent with. We apply our proposed method to the subjective task of ingredient substitution, where the agent needs to learn how to substitute ingredients in recipes based on human examples. We replicate human input using the Recipe1MSubs dataset. In our experiments, the agent achieves half its task performance after only 100 examples are provided, compared to the complete training set of 50k examples. We show that by providing examples in strategic order along with a learning method that leverages external symbolic knowledge, the agent can generalize more efficiently. 

**Abstract (ZH)**: 人类可以从少量示例中迅速学习新概念。在人工智能系统中复制这一能力 proven具有挑战性。对于具有明显数据稀缺性的主观任务，这种能力需要重新创造。在本文中，我们提出了一种直觉的人机教学架构，其中人类可以通过演示（即示例）来教导代理执行任务。为了实现直观的交互，我们认为代理应该能够从少量单个示例中进行增量学习。为此，我们的目标是利用领域知识来扩展代理的任务理解，然后通过学习方法使代理能够高效地从有限数量的示例中学习。最后，我们优化了人类如何选择最具代表性和较少冗余的示例来提供给代理的过程。我们将提出的方法应用于主观任务的材料替换任务，代理需要根据人类示例学习如何在食谱中替换材料。我们使用Recipe1MSubs数据集复制人类输入。在我们的实验中，代理在仅提供100个示例后完成了其任务性能的一半，与完整的50,000个示例训练集相比。我们展示了通过按战略顺序提供示例并结合利用外部符号知识的学习方法，代理可以更有效地泛化。 

---
# LTL$_f$ Learning Meets Boolean Set Cover 

**Title (ZH)**: LTL$_f$ 学习与布尔集合覆盖相结合 

**Authors**: Gabriel Bathie, Nathanaël Fijalkow, Théo Matricon, Baptiste Mouillon, Pierre Vandenhove  

**Link**: [PDF](https://arxiv.org/pdf/2509.24616)  

**Abstract**: Learning formulas in Linear Temporal Logic (LTLf) from finite traces is a fundamental research problem which has found applications in artificial intelligence, software engineering, programming languages, formal methods, control of cyber-physical systems, and robotics. We implement a new CPU tool called Bolt improving over the state of the art by learning formulas more than 100x faster over 70% of the benchmarks, with smaller or equal formulas in 98% of the cases. Our key insight is to leverage a problem called Boolean Set Cover as a subroutine to combine existing formulas using Boolean connectives. Thanks to the Boolean Set Cover component, our approach offers a novel trade-off between efficiency and formula size. 

**Abstract (ZH)**: 从有限轨迹学习线性时序逻辑（LTLf）公式：一种比现有技术快超过100倍的新CPU工具及其高效与公式大小的新型权衡 

---
# Neuroplasticity-inspired dynamic ANNs for multi-task demand forecasting 

**Title (ZH)**: 神经可塑性启发的动态多任务需求预测神经网络 

**Authors**: Mateusz Żarski, Sławomir Nowaczyk  

**Link**: [PDF](https://arxiv.org/pdf/2509.24495)  

**Abstract**: This paper introduces a novel approach to Dynamic Artificial Neural Networks (D-ANNs) for multi-task demand forecasting called Neuroplastic Multi-Task Network (NMT-Net). Unlike conventional methods focusing on inference-time dynamics or computational efficiency, our proposed method enables structural adaptability of the computational graph during training, inspired by neuroplasticity as seen in biological systems. Each new task triggers a dynamic network adaptation, including similarity-based task identification and selective training of candidate ANN heads, which are then assessed and integrated into the model based on their performance. We evaluated our framework using three real-world multi-task demand forecasting datasets from Kaggle. We demonstrated its superior performance and consistency, achieving lower RMSE and standard deviation compared to traditional baselines and state-of-the-art multi-task learning methods. NMT-Net offers a scalable, adaptable solution for multi-task and continual learning in time series prediction. The complete code for NMT-Net is available from our GitHub repository. 

**Abstract (ZH)**: 一种基于神经可塑性的多任务网络（NMT-Net）用于多任务需求预测的动态人工神经网络方法 

---
# Overcoming Over-Fitting in Constraint Acquisition via Query-Driven Interactive Refinement 

**Title (ZH)**: 通过查询驱动的交互式细化克服约束获取中的过拟合 

**Authors**: Vasileios Balafas, Dimos Tsouros, Nikolaos Ploskas, Kostas Stergiou  

**Link**: [PDF](https://arxiv.org/pdf/2509.24489)  

**Abstract**: Manual modeling in Constraint Programming is a substantial bottleneck, which Constraint Acquisition (CA) aims to automate. However, passive CA methods are prone to over-fitting, often learning models that include spurious global constraints when trained on limited data, while purely active methods can be query-intensive. We introduce a hybrid CA framework specifically designed to address the challenge of over-fitting in CA. Our approach integrates passive learning for initial candidate generation, a query-driven interactive refinement phase that utilizes probabilistic confidence scores (initialized by machine learning priors) to systematically identify over-fitted constraints, and a specialized subset exploration mechanism to recover valid substructures from rejected candidates. A final active learning phase ensures model completeness. Extensive experiments on diverse benchmarks demonstrate that our interactive refinement phase is crucial for achieving high target model coverage and overall model accuracy from limited examples, doing so with manageable query complexity. This framework represents a substantial advancement towards robust and practical constraint acquisition in data-limited scenarios. 

**Abstract (ZH)**: 手动建模是约束编程中的一个重大瓶颈，约束获取（CA）旨在自动完成这一过程。然而，被动的CA方法容易过拟合，在有限数据下往往会学习到包含虚假全局约束的模型，而纯粹的主动方法则可能查询密集。我们提出了一种混合CA框架，旨在解决CA中的过拟合挑战。该方法通过被动学习生成初始候选模型，通过基于概率置信分数（由机器学习先验初始化）的查询驱动交互式细化阶段系统地识别过拟合约束，并通过专门的子集探索机制从被拒绝的候选模型中恢复有效的子结构。最终的主动学习阶段确保模型完备性。广泛的实验表明，我们的交互式细化阶段对于从有限示例中实现高目标模型覆盖度和整体模型准确性至关重要，并且查询复杂性可管理。该框架朝着在数据受限场景下实现稳健且实用的约束获取迈出了重要一步。 

---
# A Systematic Review of Digital Twin-Driven Predictive Maintenance in Industrial Engineering: Taxonomy, Architectural Elements, and Future Research Directions 

**Title (ZH)**: 数字孪生驱动的工业工程预测性维护综述：分类、架构要素及未来研究方向 

**Authors**: Leila Ismail, Abdelmoneim Abdelmoti, Arkaprabha Basu, Aymen Dia Eddine Berini, Mohammad Naouss  

**Link**: [PDF](https://arxiv.org/pdf/2509.24443)  

**Abstract**: With the increasing complexity of industrial systems, there is a pressing need for predictive maintenance to avoid costly downtime and disastrous outcomes that could be life-threatening in certain domains. With the growing popularity of the Internet of Things, Artificial Intelligence, machine learning, and real-time big data analytics, there is a unique opportunity for efficient predictive maintenance to forecast equipment failures for real-time intervention and optimize maintenance actions, as traditional reactive and preventive maintenance practices are often inadequate to meet the requirements for the industry to provide quality-of-services of operations. Central to this evolution is digital twin technology, an adaptive virtual replica that continuously monitors and integrates sensor data to simulate and improve asset performance. Despite remarkable progress in digital twin implementations, such as considering DT in predictive maintenance for industrial engineering. This paper aims to address this void. We perform a retrospective analysis of the temporal evolution of the digital twin in predictive maintenance for industrial engineering to capture the applications, middleware, and technological requirements that led to the development of the digital twin from its inception to the AI-enabled digital twin and its self-learning models. We provide a layered architecture of the digital twin technology, as well as a taxonomy of the technology-enabled industrial engineering applications systems, middleware, and the used Artificial Intelligence algorithms. We provide insights into these systems for the realization of a trustworthy and efficient smart digital-twin industrial engineering ecosystem. We discuss future research directions in digital twin for predictive maintenance in industrial engineering. 

**Abstract (ZH)**: 随着工业系统的日益复杂，迫切需要预测性维护以避免昂贵的停机时间和可能在某些领域带来生命危险的灾难性结果。随着物联网、人工智能、机器学习和实时大数据分析的日益流行，这为高效的预测性维护提供了独特机会，可以预测设备故障并进行实时干预，优化维护行动，而传统的被动和预防性维护实践往往无法满足工业提供服务质量的要求。这一演变的核心是数字孪生技术，这是一种适应性的虚拟复制品，持续监控并整合传感器数据以模拟和改善资产性能。尽管在数字孪生实施方面取得了显著进展，如将数字孪生应用于工业工程的预测性维护。本文旨在填补这一空白。我们对数字孪生在工业工程中预测性维护领域的历时演变进行了回顾性分析，以捕捉从数字孪生的起源到AI驱动的数字孪生及其自学习模型的发展过程中所依赖的应用、中间件和技术要求。我们提供了数字孪生技术的分层架构，并对技术赋能的工业工程应用系统、中间件以及使用的机器学习算法进行了分类。我们提供了这些系统的见解，以实现可信赖且高效的智能数字孪生工业工程生态体系。我们讨论了数字孪生在工业工程中预测性维护方面的未来研究方向。 

---
# humancompatible.detect: a Python Toolkit for Detecting Bias in AI Models 

**Title (ZH)**: humancompatible.detect: 一个检测AI模型偏见的Python工具包 

**Authors**: German M. Matilla, Jiri Nemecek, Illia Kryvoviaz, Jakub Marecek  

**Link**: [PDF](https://arxiv.org/pdf/2509.24340)  

**Abstract**: There is a strong recent emphasis on trustworthy AI. In particular, international regulations, such as the AI Act, demand that AI practitioners measure data quality on the input and estimate bias on the output of high-risk AI systems. However, there are many challenges involved, including scalability (MMD) and computability (Wasserstein-1) issues of traditional methods for estimating distances on measure spaces. Here, we present this http URL, a toolkit for bias detection that addresses these challenges. It incorporates two newly developed methods to detect and evaluate bias: maximum subgroup discrepancy (MSD) and subsampled $\ell_\infty$ distances. It has an easy-to-use API documented with multiple examples. this http URL is licensed under the Apache License, Version 2.0. 

**Abstract (ZH)**: 近期对可信AI的高度关注。特别是国际法规，如AI法案，要求AI从业人员衡量输入数据质量并在高风险AI系统的输出中估算偏差。然而，这涉及许多挑战，包括衡量空间中传统方法估计距离的可扩展性（MMD）和可计算性（Wasserstein-1）问题。在此，我们介绍这个工具包：用于检测偏差的工具包，它解决了这些挑战。该工具包整合了两种新开发的方法来检测和评估偏差：最大子组离散度（MSD）和采样后的$\ell_\infty$距离。它具有易于使用的API，并附有多例文档说明。这个工具包采用Apache License, Version 2.0许可。 

---
# Experience Paper: Adopting Activity Recognition in On-demand Food Delivery Business 

**Title (ZH)**: 经验论文：在即时食品配送业务中采用活动识别 

**Authors**: Huatao Xu, Yan Zhang, Wei Gao, Guobin Shen, Mo Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.24303)  

**Abstract**: This paper presents the first nationwide deployment of human activity recognition (HAR) technology in the on-demand food delivery industry. We successfully adapted the state-of-the-art LIMU-BERT foundation model to the delivery platform. Spanning three phases over two years, the deployment progresses from a feasibility study in Yangzhou City to nationwide adoption involving 500,000 couriers across 367 cities in China. The adoption enables a series of downstream applications, and large-scale tests demonstrate its significant operational and economic benefits, showcasing the transformative potential of HAR technology in real-world applications. Additionally, we share lessons learned from this deployment and open-source our LIMU-BERT pretrained with millions of hours of sensor data. 

**Abstract (ZH)**: 本文介绍了首次在全国范围内将人体活动识别（HAR）技术应用于按需食品配送行业。我们成功将先进的LIMU-BERT基础模型适应到配送平台。历时两年，部署分为三个阶段，从扬州市的可行性研究扩展到全国367个城市，涉及500,000名配送员。该采用使一系列下游应用成为可能，大规模测试表明其在运营和经济方面的显著效益，展示了HAR技术在实际应用中的变革潜力。此外，我们分享了此次部署的经验教训，并开源了基于数百万小时传感器数据预训练的LIMU-BERT模型。 

---
# PAME-AI: Patient Messaging Creation and Optimization using Agentic AI 

**Title (ZH)**: PAME-AI：使用代理人工智能进行患者消息创建与优化 

**Authors**: Junjie Luo, Yihong Guo, Anqi Liu, Ritu Agarwal, Gordon  

**Link**: [PDF](https://arxiv.org/pdf/2509.24263)  

**Abstract**: Messaging patients is a critical part of healthcare communication, helping to improve things like medication adherence and healthy behaviors. However, traditional mobile message design has significant limitations due to its inability to explore the high-dimensional design space. We develop PAME-AI, a novel approach for Patient Messaging Creation and Optimization using Agentic AI. Built on the Data-Information-Knowledge-Wisdom (DIKW) hierarchy, PAME-AI offers a structured framework to move from raw data to actionable insights for high-performance messaging design. PAME-AI is composed of a system of specialized computational agents that progressively transform raw experimental data into actionable message design strategies. We demonstrate our approach's effectiveness through a two-stage experiment, comprising of 444,691 patient encounters in Stage 1 and 74,908 in Stage 2. The best-performing generated message achieved 68.76% engagement compared to the 61.27% baseline, representing a 12.2\% relative improvement in click-through rates. This agentic architecture enables parallel processing, hypothesis validation, and continuous learning, making it particularly suitable for large-scale healthcare communication optimization. 

**Abstract (ZH)**: 基于代理AI的患者信息创设与优化：PAME-AI方法 

---
# Interactive Program Synthesis for Modeling Collaborative Physical Activities from Narrated Demonstrations 

**Title (ZH)**: 基于叙述演示的协作物理活动建模的交互式程序合成 

**Authors**: Edward Kim, Daniel He, Jorge Chao, Wiktor Rajca, Mohammed Amin, Nishant Malpani, Ruta Desai, Antti Oulasvirta, Bjoern Hartmann, Sanjit Seshia  

**Link**: [PDF](https://arxiv.org/pdf/2509.24250)  

**Abstract**: Teaching systems physical tasks is a long standing goal in HCI, yet most prior work has focused on non collaborative physical activities. Collaborative tasks introduce added complexity, requiring systems to infer users assumptions about their teammates intent, which is an inherently ambiguous and dynamic process. This necessitates representations that are interpretable and correctable, enabling users to inspect and refine system behavior. We address this challenge by framing collaborative task learning as a program synthesis problem. Our system represents behavior as editable programs and uses narrated demonstrations, i.e. paired physical actions and natural language, as a unified modality for teaching, inspecting, and correcting system logic without requiring users to see or write code. The same modality is used for the system to communicate its learning to users. In a within subjects study, 20 users taught multiplayer soccer tactics to our system. 70 percent (14/20) of participants successfully refined learned programs to match their intent and 90 percent (18/20) found it easy to correct the programs. The study surfaced unique challenges in representing learning as programs and in enabling users to teach collaborative physical activities. We discuss these issues and outline mitigation strategies. 

**Abstract (ZH)**: 在人机交互中教授系统执行物理任务是一项长期目标，但大多数先前的工作集中在非协作的物理活动上。协作任务增加了复杂性，要求系统推断用户对其队友意图的假设，这是一个本质上既模糊又动态的过程。这需要可解释且可纠正的表示，使用户能够检查和改进系统行为。我们通过将协作任务学习框定为程序合成问题来应对这一挑战。我们的系统将行为表示为可编辑的程序，并使用叙述性示范，即配对的物理动作和自然语言，作为一种统一的模态，用于教学、检查和纠正系统逻辑，而无需用户看到或编写代码。系统同样使用这种模态向用户传达其所学内容。在一项单被试内研究中，20名用户教我们的系统多玩家足球战术。70%（14/20）的参与者成功地修正了所学程序以匹配其意图，90%（18/20）的参与者发现修正程序很容易。该研究揭示了将学习表示为程序以及使用户能够教授协作物理活动时的独特挑战。我们讨论了这些问题并概述了缓解策略。 

---
# SpecExit: Accelerating Large Reasoning Model via Speculative Exit 

**Title (ZH)**: SpecExit： through推测性退出加速大型推理模型 

**Authors**: Rubing Yang, Huajun Bai, Song Liu, Guanghua Yu, Runzhi Fan, Yanbin Dang, Jiejing Zhang, Kai Liu, Jianchen Zhu, Peng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24248)  

**Abstract**: Despite their strong performance on reasoning tasks, large reasoning models (LRMs) often suffer from overthinking, producing unnecessarily long outputs and incurring high end-to-end latency, a significant limitation to their real-world deployment. To address overthinking, early-exit mechanisms have been proposed to terminate reasoning before typical completion, showing that this approach can effectively shorten generation length with minimal impact on accuracy. However, their reliance on probing mechanisms introduces a detection overhead that limits their end-to-end latency gains and compromises their generalizability across diverse problems. Inspired by the use of hidden states in speculative decoding, we propose SpecExit, a novel framework that predicts both future tokens and an early-exit signal directly from a lightweight draft model without probing overhead. Our method offers significant improvements, reducing average generation length by 66\% and achieving a 2.5x speedup in end-to-end latency compared to the speculative decoding baseline, without compromising accuracy. Our method leverages the inherent signals from hidden states to provide effective early-exit signals, suggesting broader use of hidden states for efficient reasoning. Our code is available at this https URL. 

**Abstract (ZH)**: 尽管大型推理模型在推理任务中表现出色，但它们往往会过度推理，生成不必要的长输出，并导致较高的端到端延迟，这成为其实际部署中的一个重要限制。为解决过度推理问题，已提出了早期退出机制，可以在典型完成之前终止推理，表明这种方法可以在不显著影响准确性的前提下有效缩短生成长度。然而，这些方法依赖于探针机制，引入了检测开销，限制了其端到端延迟的改进，并降低了其在多样问题上的普适性。受投机解码中隐藏状态使用启发，我们提出SpecExit，一种新颖的框架，可以直接从轻量级草图模型中预测未来的令牌和早期退出信号，而不需要探针开销。我们的方法提供了显著的改进，平均生成长度减少了66%，端到端延迟提高了2.5倍，同时保持了准确性。我们的方法利用隐藏状态中的固有信号提供有效的早期退出信号，表明隐藏状态在高效推理中的更广泛使用。我们的代码可在此访问：this https URL。 

---
# Humanline: Online Alignment as Perceptual Loss 

**Title (ZH)**: Humanline: 在线对齐作为感知损失 

**Authors**: Sijia Liu, Niklas Muennighoff, Kawin Ethayarajh  

**Link**: [PDF](https://arxiv.org/pdf/2509.24207)  

**Abstract**: Online alignment (e.g., GRPO) is generally more performant than offline alignment (e.g., DPO) -- but why? Drawing on prospect theory from behavioral economics, we propose a human-centric explanation. We prove that online on-policy sampling better approximates the human-perceived distribution of what the model can produce, and PPO/GRPO-style clipping -- originally introduced to just stabilize training -- recovers a perceptual bias in how humans perceive probability. In this sense, PPO/GRPO act as perceptual losses already. Our theory further suggests that the online/offline dichotomy is itself incidental to maximizing human utility, since we can achieve the same effect by selectively training on any data in a manner that mimics human perception, rather than restricting ourselves to online on-policy data. Doing so would allow us to post-train more quickly, cheaply, and flexibly without sacrificing performance. To this end, we propose a design pattern that explicitly incorporates perceptual distortions of probability into objectives like DPO/KTO/GRPO, creating humanline variants of them. Surprisingly, we find that these humanline variants, even when trained with offline off-policy data, can match the performance of their online counterparts on both verifiable and unverifiable tasks. 

**Abstract (ZH)**: 基于行为经济学前景理论的在线对齐为何更优——一种以人为中心的解释及其实验设计 

---
# Reasoning or Retrieval? A Study of Answer Attribution on Large Reasoning Models 

**Title (ZH)**: 推理还是检索？大规模推理模型的答案归因研究 

**Authors**: Yuhui Wang, Changjiang Li, Guangke Chen, Jiacheng Liang, Ting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24156)  

**Abstract**: Large reasoning models (LRMs) exhibit unprecedented capabilities in solving complex problems through Chain-of-Thought (CoT) reasoning. However, recent studies reveal that their final answers often contradict their own reasoning traces. We hypothesize that this inconsistency stems from two competing mechanisms for generating answers: CoT reasoning and memory retrieval. To test this hypothesis, we conduct controlled experiments that challenge LRMs with misleading cues during reasoning and/or corrupted answers during retrieval. Our results across models and datasets confirm that both mechanisms operate simultaneously, with their relative dominance influenced by multiple factors: problem domains, model scales, and fine-tuning approaches (e.g., reinforcement learning vs. distillation). The findings reveal a critical limitation in current reasoning fine-tuning paradigms: models can exploit the retrieval mechanism as a shortcut, effectively "hacking" the reward signal and undermining genuine reasoning development. To address this challenge, we introduce FARL, a novel fine-tuning framework that integrates memory unlearning with reinforcement learning. By carefully suppressing retrieval shortcuts during the fine-tuning process, FARL promotes reasoning-dominant behavior and enhances generalizable reasoning capabilities. 

**Abstract (ZH)**: 大型推理模型（LRMs）通过链式推理（CoT）展现出解决复杂问题的前所未有的能力。然而，近期研究表明，它们的最终答案往往与其推理轨迹相矛盾。我们假设这种不一致性源自生成答案的两种竞争机制：链式推理和记忆检索。为了检验这一假设，我们进行了控制实验，这些实验在推理过程中或检索过程中向LRMs提供误导性线索或错误的答案。我们的研究结果证实，这两种机制同时运作，它们的相对主导地位受多种因素影响：问题领域、模型规模以及微调方法（例如强化学习与蒸馏）。这些发现揭示了当前推理微调范式的一个关键局限性：模型可以利用检索机制作为捷径，有效地“ hack”奖励信号，削弱真正的推理发展。为了应对这一挑战，我们提出了一种名为FARL的新颖微调框架，该框架将记忆遗忘与强化学习相结合。通过在微调过程中精心抑制检索捷径，FARL促进了以推理为主的机制并增强了可泛化的推理能力。 

---
# Transparent, Evaluable, and Accessible Data Agents: A Proof-of-Concept Framework 

**Title (ZH)**: 透明、可评估且可访问的数据代理：一个概念验证框架 

**Authors**: Nooshin Bahador  

**Link**: [PDF](https://arxiv.org/pdf/2509.24127)  

**Abstract**: This article presents a modular, component-based architecture for developing and evaluating AI agents that bridge the gap between natural language interfaces and complex enterprise data warehouses. The system directly addresses core challenges in data accessibility by enabling non-technical users to interact with complex data warehouses through a conversational interface, translating ambiguous user intent into precise, executable database queries to overcome semantic gaps. A cornerstone of the design is its commitment to transparent decision-making, achieved through a multi-layered reasoning framework that explains the "why" behind every decision, allowing for full interpretability by tracing conclusions through specific, activated business rules and data points. The architecture integrates a robust quality assurance mechanism via an automated evaluation framework that serves multiple functions: it enables performance benchmarking by objectively measuring agent performance against golden standards, and it ensures system reliability by automating the detection of performance regressions during updates. The agent's analytical depth is enhanced by a statistical context module, which quantifies deviations from normative behavior, ensuring all conclusions are supported by quantitative evidence including concrete data, percentages, and statistical comparisons. We demonstrate the efficacy of this integrated agent-development-with-evaluation framework through a case study on an insurance claims processing system. The agent, built on a modular architecture, leverages the BigQuery ecosystem to perform secure data retrieval, apply domain-specific business rules, and generate human-auditable justifications. The results confirm that this approach creates a robust, evaluable, and trustworthy system for deploying LLM-powered agents in data-sensitive, high-stakes domains. 

**Abstract (ZH)**: 基于模块化组件的设计：结合自然语言接口与复杂企业数据仓库的AI代理开发与评估架构 

---
# Fathom-DeepResearch: Unlocking Long Horizon Information Retrieval and Synthesis for SLMs 

**Title (ZH)**: Fathom-DeepResearch: 解锁SLMs的长时信息检索与合成能力 

**Authors**: Shreyas Singh, Kunal Singh, Pradeep Moturi  

**Link**: [PDF](https://arxiv.org/pdf/2509.24107)  

**Abstract**: Tool-integrated reasoning has emerged as a key focus for enabling agentic applications. Among these, DeepResearch Agents have gained significant attention for their strong performance on complex, open-ended information-seeking tasks. We introduce Fathom-DeepResearch, an agentic system composed of two specialized models. The first is Fathom-Search-4B, a DeepSearch model trained from Qwen3-4B and optimized for evidence-based investigation through live web search and targeted webpage querying. Its training combines three advances: (i) DUETQA, a 5K-sample dataset generated via multi-agent self-play that enforces strict web-search dependence and heterogeneous source grounding; (ii) RAPO, a zero-overhead extension of GRPO that stabilizes multi-turn Reinforcement Learning with Verifiable Rewards through curriculum pruning, reward-aware advantage scaling, and per-prompt replay buffers; and (iii) a steerable step-level reward that classifies each tool call by cognitive behavior and marginal utility, enabling explicit control over search trajectory breadth, depth, and horizon. These improvements enable reliable extension of tool-calling beyond 20 calls when warranted. The second is Fathom-Synthesizer-4B, trained from Qwen3-4B, which converts multi-turn DeepSearch traces into structured, citation-dense DeepResearch Reports for comprehensive synthesis. Evaluated on DeepSearch benchmarks (SimpleQA, FRAMES, WebWalker, Seal0, MuSiQue) and DeepResearch-Bench, the system achieves state-of-the-art performance in the open-weights category while demonstrating strong generalization to diverse reasoning tasks including HLE, AIME-25, GPQA-Diamond, and MedQA. 

**Abstract (ZH)**: 基于工具集成的推理已成为推动自主应用的关键焦点。在其中，DeepResearch 代理因其在复杂开放性信息检索任务上的出色表现而受到广泛关注。我们介绍了一种名为Fathom-DeepResearch的自主系统，该系统由两个专门模型组成。第一个是Fathom-Search-4B，这是一种从Qwen3-4B训练而来的DeepSearch模型，通过实时网络搜索和定向网页查询优化用于证据基础的调查。其训练结合了三项改进：(i) DUETQA，一个通过多智能体自我对弈生成的5千样本数据集，强制执行严格的网络搜索依赖性和异质来源接地；(ii) RAPO，这是一种零开销的GRPO扩展，通过课程剪枝、奖励意识优势缩放和每个提示回放缓冲区实现了可验证奖励的多轮强化学习稳定；(iii) 可调节的步骤级奖励，根据认知行为和边际效用对每次工具调用进行分类，允许对搜索轨迹的宽度、深度和视界进行显式控制。这些改进使Fathom-Search-4B能够在必要时可靠地扩展工具调用超过20次。第二个是Fathom-Synthesizer-4B，这是一种从Qwen3-4B训练而来的模型，将多轮DeepSearch踪迹转换为结构化、引文密集型的DeepResearch报告，用于综合总结。该系统在DeepSearch基准测试（SimpleQA、FRAMES、WebWalker、Seal0、MuSiQue）和DeepResearch-Bench上进行评估，实现了开放权重类别中的最佳性能，同时展示了强大的泛化能力，涵盖了包括HLE、AIME-25、GPQA-Diamond和MedQA在内的多种推理任务。 

---
# Future-Proofing Programmers: Optimal Knowledge Tracing for AI-Assisted Personalized Education 

**Title (ZH)**: 面向未来的程序员：AI辅助个性化教育的最佳知识追踪 

**Authors**: Yuchen Wang, Pei-Duo Yu, Chee Wei Tan  

**Link**: [PDF](https://arxiv.org/pdf/2509.23996)  

**Abstract**: Learning to learn is becoming a science, driven by the convergence of knowledge tracing, signal processing, and generative AI to model student learning states and optimize education. We propose CoTutor, an AI-driven model that enhances Bayesian Knowledge Tracing with signal processing techniques to improve student progress modeling and deliver adaptive feedback and strategies. Deployed as an AI copilot, CoTutor combines generative AI with adaptive learning technology. In university trials, it has demonstrated measurable improvements in learning outcomes while outperforming conventional educational tools. Our results highlight its potential for AI-driven personalization, scalability, and future opportunities for advancing privacy and ethical considerations in educational technology. Inspired by Richard Hamming's vision of computer-aided 'learning to learn,' CoTutor applies convex optimization and signal processing to automate and scale up learning analytics, while reserving pedagogical judgment for humans, ensuring AI facilitates the process of knowledge tracing while enabling learners to uncover new insights. 

**Abstract (ZH)**: 学习如何学习正成为一门科学，由知识追踪、信号处理和生成式AI的结合驱动，以建模学生的学习状态并优化教育。我们提出CoTutor，一种基于生成式AI和适配学习技术增强贝叶斯知识追踪的AI驱动模型，以提升学生进展建模和提供适应性反馈和策略。作为AI副驾部署，CoTutor在大学试验中表现出可测量的学习成果提升，并超越了传统教育工具。我们的结果突显了其在AI驱动个性化、可扩展性以及教育技术中隐私和伦理考虑方面未来机遇的潜力。受Richard Hamming关于‘学习如何学习’的计算机辅助愿景启发，CoTutor应用凸优化和信号处理技术自动化和规模化学习分析，同时保留教学判断权于人类，确保AI促进知识追踪过程，同时帮助学习者发现新的见解。 

---
# Automatic selection of primary studies in systematic reviews with evolutionary rule-based classification 

**Title (ZH)**: 基于进化规则分类的系统评价中主要研究的自动选择 

**Authors**: José de la Torre-López, Aurora Ramírez, José Raúl Romero  

**Link**: [PDF](https://arxiv.org/pdf/2509.23981)  

**Abstract**: Searching, filtering and analysing scientific literature are time-consuming tasks when performing a systematic literature review. With the rise of artificial intelligence, some steps in the review process are progressively being automated. In particular, machine learning for automatic paper selection can greatly reduce the effort required to identify relevant literature in scientific databases. We propose an evolutionary machine learning approach, called \ourmodel, to automatically determine whether a paper retrieved from a literature search process is relevant. \ourmodel builds an interpretable rule-based classifier using grammar-guided genetic programming. The use of a grammar to define the syntax and the structure of the rules allows \ourmodel to easily combine the usual textual information with other bibliometric data not considered by state-of-the-art methods. Our experiments demonstrate that it is possible to generate accurate classifiers without impairing interpretability and using configurable information sources not supported so far. 

**Abstract (ZH)**: 系统文献综述中搜索、筛选和分析科学文献是耗时的任务。随着人工智能的发展，审查过程中的某些步骤正逐渐实现自动化。特别是，用于自动论文筛选的机器学习可以大大减少在科学数据库中识别相关文献所需的努力。我们提出了一种进化机器学习方法，称为\ourmodel，以自动确定从文献搜索过程中检索到的论文是否相关。\ourmodel 使用语法引导的遗传编程构建了一个可解释的基于规则的分类器。使用语法来定义规则的语法和结构，使得\ourmodel 可以轻松地结合通常的文本信息和其他不属于现有方法考虑的引文计量数据。我们的实验表明，可以在不损害可解释性的情况下生成准确的分类器，并且可以使用迄今为止尚未配置的信息来源。 

---
# From Neural Networks to Logical Theories: The Correspondence between Fibring Modal Logics and Fibring Neural Networks 

**Title (ZH)**: 从神经网络到逻辑理论：纤维化模态逻辑与纤维化神经网络之间的对应关系 

**Authors**: Ouns El Harzli, Bernardo Cuenca Grau, Artur d'Avila Garcez, Ian Horrocks, Tarek R. Besold  

**Link**: [PDF](https://arxiv.org/pdf/2509.23912)  

**Abstract**: Fibring of modal logics is a well-established formalism for combining countable families of modal logics into a single fibred language with common semantics, characterized by fibred models. Inspired by this formalism, fibring of neural networks was introduced as a neurosymbolic framework for combining learning and reasoning in neural networks. Fibring of neural networks uses the (pre-)activations of a trained network to evaluate a fibring function computing the weights of another network whose outputs are injected back into the original network. However, the exact correspondence between fibring of neural networks and fibring of modal logics was never formally established. In this paper, we close this gap by formalizing the idea of fibred models \emph{compatible} with fibred neural networks. Using this correspondence, we then derive non-uniform logical expressiveness results for Graph Neural Networks (GNNs), Graph Attention Networks (GATs) and Transformer encoders. Longer-term, the goal of this paper is to open the way for the use of fibring as a formalism for interpreting the logical theories learnt by neural networks with the tools of computational logic. 

**Abstract (ZH)**: 模态逻辑的纤维化是一种将可数系列模态逻辑结合成具有共同语义的纤维化语言的形式主义。受此形式主义的启发，神经网络的纤维化被引入作为一种将学习和推理结合到神经网络中的神经符号框架。神经网络的纤维化利用训练网络的（预）激活来评估纤维化函数，计算另一个网络的权重并将输出注入原始网络。然而，神经网络的纤维化与模态逻辑的纤维化之间的精确对应关系从未正式建立。在本文中，我们通过形式化与纤维化神经网络相兼容的纤维化模型的概念来填补这一差距。借助这种对应关系，我们随后推导出了图神经网络（GNNs）、图注意力网络（GATs）和变换器编码器的非均匀逻辑表达能力结果。长远来看，本文的目标是为使用纤维化作为工具来解释神经网络学习的逻辑理论的计算逻辑工具铺平道路。 

---
# AgentGuard: Runtime Verification of AI Agents 

**Title (ZH)**: AgentGuard：AI代理的运行时验证 

**Authors**: Roham Koohestani  

**Link**: [PDF](https://arxiv.org/pdf/2509.23864)  

**Abstract**: The rapid evolution to autonomous, agentic AI systems introduces significant risks due to their inherent unpredictability and emergent behaviors; this also renders traditional verification methods inadequate and necessitates a shift towards probabilistic guarantees where the question is no longer if a system will fail, but the probability of its failure within given constraints. This paper presents AgentGuard, a framework for runtime verification of Agentic AI systems that provides continuous, quantitative assurance through a new paradigm called Dynamic Probabilistic Assurance. AgentGuard operates as an inspection layer that observes an agent's raw I/O and abstracts it into formal events corresponding to transitions in a state model. It then uses online learning to dynamically build and update a Markov Decision Process (MDP) that formally models the agent's emergent behavior. Using probabilistic model checking, the framework then verifies quantitative properties in real-time. 

**Abstract (ZH)**: 自主智能体AI系统的快速进化引入了由于其固有的不可预测性和 emergent 行为而带来的显著风险；这使得传统的验证方法变得不足，需要转向基于概率保证的方法，其中的问题不再是系统是否会失败，而是系统在给定约束下的失败概率。本文提出了AgentGuard框架，这是一种用于运行时验证智能体AI系统的框架，通过一种新提出的动态概率保证 paradigmn 提供连续的定量保障。AgentGuard作为一种检查层，观察智能体的原始输入/输出，并将其抽象为与状态模型转换对应的正式事件。然后，它使用在线学习来动态构建和更新马尔可夫决策过程（MDP），以正式建模智能体的 emergent 行为。利用概率模型检测，该框架在实时情况下验证定量属性。 

---
# AnveshanaAI: A Multimodal Platform for Adaptive AI/ML Education through Automated Question Generation and Interactive Assessment 

**Title (ZH)**: AnveshanaAI：一种通过自动化问题生成和互动评估实现自适应AI/ML教育的多模态平台 

**Authors**: Rakesh Thakur, Diksha Khandelwal, Shreya Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2509.23811)  

**Abstract**: We propose AnveshanaAI, an application-based learning platform for artificial intelligence. With AnveshanaAI, learners are presented with a personalized dashboard featuring streaks, levels, badges, and structured navigation across domains such as data science, machine learning, deep learning, transformers, generative AI, large language models, and multimodal AI, with scope to include more in the future. The platform incorporates gamified tracking with points and achievements to enhance engagement and learning, while switching between Playground, Challenges, Simulator, Dashboard, and Community supports exploration and collaboration. Unlike static question repositories used in existing platforms, AnveshanaAI ensures balanced learning progression through a dataset grounded in Bloom's taxonomy, with semantic similarity checks and explainable AI techniques improving transparency and reliability. Adaptive, automated, and domain-aware assessment methods are also employed. Experiments demonstrate broad dataset coverage, stable fine-tuning with reduced perplexity, and measurable gains in learner engagement. Together, these features illustrate how AnveshanaAI integrates adaptivity, gamification, interactivity, and explainability to support next-generation AI education. 

**Abstract (ZH)**: 我们提出AnveshanaAI，一个基于应用的人工智能学习平台。AnveshanaAI为学习者提供了个性化仪表盘，展示连续学习记录、层级、徽章，并跨数据科学、机器学习、深度学习、变换器、生成人工智能、大规模语言模型以及多模态人工智能领域提供了结构化的导航，未来还将包括更多领域。该平台集成了游戏化的跟踪机制，并通过点数和成就提高参与度和学习效果；学习者可在游乐场、挑战、模拟器、仪表盘和社区之间切换，促进探索与合作。与现有平台使用的静态问题库不同，AnveshanaAI通过基于Bloom taxonomy的数据集确保平衡的学习进展，并通过语义相似性检查和可解释的人工智能技术提高透明度和可靠性。该平台还采用了自适应、自动化和领域感知的评估方法。实验结果表明，AnveshanaAI实现了广泛的数据集覆盖、稳定的数据集微调和困惑度降低，并在学习者参与度方面取得可测量的提升。这些功能共同展示了AnveshanaAI如何通过自适应、游戏化、互动性和可解释性来支持下一代人工智能教育。 

---
# From Frustration to Fun: An Adaptive Problem-Solving Puzzle Game Powered by Genetic Algorithm 

**Title (ZH)**: 从挫折到乐趣：一种基于遗传算法的自适应问题解决益智游戏 

**Authors**: Matthew McConnell, Richard Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.23796)  

**Abstract**: This paper explores adaptive problem solving with a game designed to support the development of problem-solving skills. Using an adaptive, AI-powered puzzle game, our adaptive problem-solving system dynamically generates pathfinding-based puzzles using a genetic algorithm, tailoring the difficulty of each puzzle to individual players in an online real-time approach. A player-modeling system records user interactions and informs the generation of puzzles to approximate a target difficulty level based on various metrics of the player. By combining procedural content generation with online adaptive difficulty adjustment, the system aims to maintain engagement, mitigate frustration, and maintain an optimal level of challenge. A pilot user study investigates the effectiveness of this approach, comparing different types of adaptive difficulty systems and interpreting players' responses. This work lays the foundation for further research into emotionally informed player models, advanced AI techniques for adaptivity, and broader applications beyond gaming in educational settings. 

**Abstract (ZH)**: 本文探讨了一种通过设计用于支持问题解决技能发展的游戏来进行自适应问题解决的方法。通过一个自适应的AI驱动的谜题游戏，我们的自适应问题解决系统使用遗传算法动态生成路径finding为基础的谜题，并采用在线实时方式个性化调整每个谜题的难度。玩家建模系统记录用户交互，并根据玩家的各种指标信息来调整谜题的难度以逼近目标难度水平。通过结合过程化内容生成与在线自适应难度调整，该系统旨在保持参与度、减轻挫败感，并维持适当的挑战水平。一项试点用户研究探讨了该方法的有效性，比较了不同类型自适应难度系统，并分析了玩家的反应。本文为情感化玩家模型、高级自适应AI技术以及教育等更广泛领域中的应用奠定了基础。 

---
# GUI-Shepherd: Reliable Process Reward and Verification for Long-Sequence GUI Tasks 

**Title (ZH)**: GUI-牧羊人: 可靠的长序列GUI任务过程奖励与验证 

**Authors**: Cong Chen, Kaixiang Ji, Hao Zhong, Muzhi Zhu, Anzhou Li, Guo Gan, Ziyuan Huang, Cheng Zou, Jiajia Liu, Jingdong Chen, Hao Chen, Chunhua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.23738)  

**Abstract**: Autonomous agents for long-sequence Graphical User Interface tasks are hindered by sparse rewards and the intractable credit assignment problem. To address these challenges, we introduce GUI-Shepherd, a Process Reward Model that provides dense, step-by-step feedback to guide agents. GUI-Shepherd is trained on a diverse large-scale data set of $52$k interactions that features human-annotated scores and GPT-4o generated rationales, enabling it to serve both as a reward provider for RL training and as a verifier for inference. As far as we know, we are the first to conduct a systematic study of process supervision in GUI agents, across diverse settings from online long-horizon tasks to offline single-step prediction. On the online AndroidWorld benchmark, GUI-Shepherd improves success rate by $7.7$ points via multi-turn online PPO, significantly outperforming Outcome Reward Model based competitors. When used as an inference verifier, it brings $5.1$ points improvements. The benefits generalize to the offline AndroidControl benchmark, with gains of $2.2$ points as a reward provider and $4.3$ points as a verifier. Collectively, our results establish that high-fidelity process supervision is critical for building more capable GUI agents and present a generalizable solution. 

**Abstract (ZH)**: 自主代理在长时间序列图形用户界面任务中受到稀疏奖励和归因难题的阻碍。为解决这些挑战，我们引入了GUI-Shepherd，一种过程奖励模型，能够提供详细逐步反馈以指导代理。GUI-Shepherd基于包含人类标注评分和GPT-4o生成的解释的大规模多样数据集进行训练，使其既能作为强化学习训练的奖励提供者，又能作为推理的验证器。据我们所知，这是首次系统研究GUI代理的过程监督，从在线长时间任务到离线单步预测。在在线AndroidWorld基准测试中，通过多轮在线PPO，GUI-Shepherd将成功率提高了7.7个百分点，显著优于基于结果奖励模型的竞争者。作为推理验证器时，它带来了5.1个百分点的提高。这些好处在离线AndroidControl基准测试中也得到验证，作为奖励提供者时提高了2.2个百分点，作为验证器时提高了4.3个百分点。总体而言，我们的研究结果表明，高保真过程监督对于构建更强大的GUI代理至关重要，并提出了一种可泛化的解决方案。 

---
# Diagnosing Failure Root Causes in Platform-Orchestrated Agentic Systems: Dataset, Taxonomy, and Benchmark 

**Title (ZH)**: 平台 orchestration 执行体系统中失败根本原因诊断：数据集、分类学和基准 

**Authors**: Xuyan Ma, Xiaofei Xie, Yawen Wang, Junjie Wang, Boyu Wu, Mingyang Li, Qing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23735)  

**Abstract**: Agentic systems consisting of multiple LLM-driven agents coordinating through tools and structured interactions, are increasingly deployed for complex reasoning and problem-solving tasks. At the same time, emerging low-code and template-based agent development platforms (e.g., Dify) enable users to rapidly build and orchestrate agentic systems, which we refer to as platform-orchestrated agentic systems. However, these systems are also fragile and it remains unclear how to systematically identify their potential failure root cause. This paper presents a study of root cause identification of these platform-orchestrated agentic systems. To support this initiative, we construct a dataset AgentFail containing 307 failure logs from ten agentic systems, each with fine-grained annotations linking failures to their root causes. We additionally utilize counterfactual reasoning-based repair strategy to ensure the reliability of the annotation. Building on the dataset, we develop a taxonomy that characterizes failure root causes and analyze their distribution across different platforms and task domains. Furthermore, we introduce a benchmark that leverages LLMs for automatically identifying root causes, in which we also utilize the proposed taxonomy as guidance for LLMs. Results show that the taxonomy can largely improve the performance, thereby confirming its utility. Nevertheless, the accuracy of root cause identification reaches at most 33.6%, which indicates that this task still remains challenging. In light of these results, we also provide actionable guidelines for building such agentic systems. In summary, this paper provides a reliable dataset of failure root cause for platform-orchestrated agentic systems, corresponding taxonomy and benchmark, which serves as a foundation for advancing the development of more reliable agentic systems. 

**Abstract (ZH)**: 平台 orchestration 的代理系统根因识别研究 

---
# MedLA: A Logic-Driven Multi-Agent Framework for Complex Medical Reasoning with Large Language Models 

**Title (ZH)**: MedLA：一种逻辑驱动的多agent框架，用于大型语言模型在复杂医疗推理中的应用 

**Authors**: Siqi Ma, Jiajie Huang, Bolin Yang, Fan Zhang, Jinlin Wu, Yue Shen, Guohui Fan, Zhu Zhang, Zelin Zang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23725)  

**Abstract**: Answering complex medical questions requires not only domain expertise and patient-specific information, but also structured and multi-perspective reasoning. Existing multi-agent approaches often rely on fixed roles or shallow interaction prompts, limiting their ability to detect and resolve fine-grained logical inconsistencies. To address this, we propose \textsc{MedLA}, a logic-driven multi-agent framework built on large language models. Each agent organizes its reasoning process into an explicit logical tree based on syllogistic triads (major premise, minor premise, and conclusion), enabling transparent inference and premise-level alignment. Agents engage in a multi-round, graph-guided discussion to compare and iteratively refine their logic trees, achieving consensus through error correction and contradiction resolution. We demonstrate that \textsc{MedLA} consistently outperforms both static role-based systems and single-agent baselines on challenging benchmarks such as MedDDx and standard medical QA tasks. Furthermore, \textsc{MedLA} scales effectively across both open-source and commercial LLM backbones, achieving state-of-the-art performance and offering a generalizable paradigm for trustworthy medical reasoning. 

**Abstract (ZH)**: 逻辑驱动的多智能体框架\textsc{MedLA}：应对复杂医疗问题需要专业知识、患者特定信息以及结构化的多视角推理 

---
# Measuring Sparse Autoencoder Feature Sensitivity 

**Title (ZH)**: 测量稀疏自编码器特征敏感性 

**Authors**: Claire Tian, Katherine Tian, Nathan Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23717)  

**Abstract**: Sparse Autoencoder (SAE) features have become essential tools for mechanistic interpretability research. SAE features are typically characterized by examining their activating examples, which are often "monosemantic" and align with human interpretable concepts. However, these examples don't reveal feature sensitivity: how reliably a feature activates on texts similar to its activating examples. In this work, we develop a scalable method to evaluate feature sensitivity. Our approach avoids the need to generate natural language descriptions for features; instead we use language models to generate text with the same semantic properties as a feature's activating examples. We then test whether the feature activates on these generated texts. We demonstrate that sensitivity measures a new facet of feature quality and find that many interpretable features have poor sensitivity. Human evaluation confirms that when features fail to activate on our generated text, that text genuinely resembles the original activating examples. Lastly, we study feature sensitivity at the SAE level and observe that average feature sensitivity declines with increasing SAE width across 7 SAE variants. Our work establishes feature sensitivity as a new dimension for evaluating both individual features and SAE architectures. 

**Abstract (ZH)**: 稀疏自编码器（SAE）特征已成为机制解释性研究中不可或缺的工具。本研究开发了一种可扩展的方法来评估特征敏感性。我们的方法避免为特征生成自然语言描述，而是使用语言模型生成与特征激活示例具有相同语义属性的文本，然后测试特征是否能够在这些生成的文本上激活。我们展示敏感性衡量了特征质量的一个新的方面，并发现许多可解释的特征在敏感性方面表现较差。人类评估显示，当特征未能在生成的文本上激活时，这些文本确实类似于原始的激活示例。最后，我们在稀疏自编码器（SAE）层次上研究了特征敏感性，发现在7种不同宽度的SAE变体中，平均特征敏感性随着SAE宽度增加而下降。我们的研究确立了特征敏感性作为评估单个特征和SAE架构的新维度。 

---
# From Reasoning to Answer: Empirical, Attention-Based and Mechanistic Insights into Distilled DeepSeek R1 Models 

**Title (ZH)**: 从推理到答案：关于Distilled DeepSeek R1模型的经验、注意力机制和机理洞察 

**Authors**: Jue Zhang, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23676)  

**Abstract**: Large Reasoning Models (LRMs) generate explicit reasoning traces alongside final answers, yet the extent to which these traces influence answer generation remains unclear. In this work, we conduct a three-stage investigation into the interplay between reasoning and answer generation in three distilled DeepSeek R1 models. First, through empirical evaluation, we demonstrate that including explicit reasoning consistently improves answer quality across diverse domains. Second, attention analysis reveals that answer tokens attend substantially to reasoning tokens, with certain mid-layer Reasoning-Focus Heads (RFHs) closely tracking the reasoning trajectory, including self-reflective cues. Third, we apply mechanistic interventions using activation patching to assess the dependence of answer tokens on reasoning activations. Our results show that perturbations to key reasoning tokens can reliably alter the final answers, confirming a directional and functional flow of information from reasoning to answer. These findings deepen our understanding of how LRMs leverage reasoning tokens for answer generation, highlighting the functional role of intermediate reasoning in shaping model outputs. Our data and code are publicly available at \href{this https URL}{this URL}. 

**Abstract (ZH)**: 大型推理模型（LRMs）生成显式的推理痕迹以及最终答案，但这些痕迹对答案生成的影响程度尚不明确。在本文中，我们对三个经过提炼的DeepSeek R1模型中的推理与答案生成的交互作用进行了三阶段的研究。首先，通过实证评估，我们证明了包含显式推理可以一致地提高跨不同领域答案的质量。其次，注意力分析显示答案标记会显著地关注推理标记，某些中间层的推理聚焦头（RFHs）紧密跟踪推理轨迹，包括自我反思线索。第三，我们通过激活修补的方法应用机理干预，评估答案标记对推理激活的依赖性。我们的结果表明，对关键推理标记的扰动可以可靠地改变最终答案，证实了从推理到答案的信息传递是具有方向性和功能性的。这些发现加深了我们对LRMs如何利用推理标记进行答案生成的理解，突出了中间推理的功能性作用以塑造模型输出。我们的数据和代码在\href{this https URL}{this URL}公开可用。 

---
# A Hierarchical Structure-Enhanced Personalized Recommendation Model for Traditional Chinese Medicine Formulas Based on KG Diffusion Guidance 

**Title (ZH)**: 基于KG扩散指导的层次结构增强中医药方个性化推荐模型 

**Authors**: ChaoBo Zhang, Long Tan  

**Link**: [PDF](https://arxiv.org/pdf/2509.23560)  

**Abstract**: Artificial intelligence technology plays a crucial role in recommending prescriptions for traditional Chinese medicine (TCM). Previous studies have made significant progress by focusing on the symptom-herb relationship in prescriptions. However, several limitations hinder model performance: (i) Insufficient attention to patient-personalized information such as age, BMI, and medical history, which hampers accurate identification of syndrome and reduces efficacy. (ii) The typical long-tailed distribution of herb data introduces training biases and affects generalization ability. (iii) The oversight of the 'monarch, minister, assistant and envoy' compatibility among herbs increases the risk of toxicity or side effects, opposing the 'treatment based on syndrome differentiation' principle in clinical TCM. Therefore, we propose a novel hierarchical structure-enhanced personalized recommendation model for TCM formulas based on knowledge graph diffusion guidance, namely TCM-HEDPR. Specifically, we pre-train symptom representations using patient-personalized prompt sequences and apply prompt-oriented contrastive learning for data augmentation. Furthermore, we employ a KG-guided homogeneous graph diffusion method integrated with a self-attention mechanism to globally capture the non-linear symptom-herb relationship. Lastly, we design a heterogeneous graph hierarchical network to integrate herbal dispensing relationships with implicit syndromes, guiding the prescription generation process at a fine-grained level and mitigating the long-tailed herb data distribution problem. Extensive experiments on two public datasets and one clinical dataset demonstrate the effectiveness of TCM-HEDPR. In addition, we incorporate insights from modern medicine and network pharmacology to evaluate the recommended prescriptions comprehensively. It can provide a new paradigm for the recommendation of modern TCM. 

**Abstract (ZH)**: 人工智能技术在推荐中医药方中的作用至关重要。尽管以往研究集中在处方中的症状-药关系上取得了显著进展，但模型性能仍受到多重限制：（i）缺乏对年龄、BMI和病史等患者个性化信息的关注，影响了病机识别的准确性并降低了疗效。（ii）草药数据的典型长尾分布引入了训练偏差，影响了泛化能力。（iii）忽视了草药间的‘君、臣、佐、使’配伍关系，增加了毒副作用的风险，违背了临床中医药‘辨证施治’的原则。因此，我们提出了一种基于知识图谱扩散指导的新型层次结构增强个性化推荐模型，即TCM-HEDPR。具体而言，我们使用患者个性化提示序列提前训练症状表示，并应用提示导向的对比学习进行数据增强。此外，我们采用知识图谱引导的同质图扩散方法结合自我注意机制，全局捕捉非线性的症状-药关系。最后，我们设计了一种异质图层次网络，将中药配伍关系与隐含病机相结合，精细指导处方生成过程，并缓解长尾草药数据分布问题。在两个公开数据集和一个临床数据集上的广泛实验表明了TCM-HEDPR的有效性。此外，我们结合现代医学和网络药理学的见解，全面评估推荐处方的有效性，为现代中医药推荐提供了一个新的范式。 

---
# DOoM: Difficult Olympiads of Math 

**Title (ZH)**: DOoM: 困难的数学奥林匹克问题集 

**Authors**: Ilya Kuleshov, Ilin Pavel, Nikolay Kompanets, Ksenia Sycheva, Aleksandr Nikolich  

**Link**: [PDF](https://arxiv.org/pdf/2509.23529)  

**Abstract**: This paper introduces DOoM, a new open-source benchmark designed to assess the capabilities of language models in solving mathematics and physics problems in Russian. The benchmark includes problems of varying difficulty, ranging from school-level tasks to university Olympiad and entrance exam questions. In this paper we discuss the motivation behind its creation, describe dataset's structure and evaluation methodology, and present initial results from testing various models. Analysis of the results shows a correlation between model performance and the number of tokens used, and highlights differences in performance between mathematics and physics tasks. 

**Abstract (ZH)**: 这篇论文介绍了DOoM，一个新推出的开源基准，旨在评估语言模型解决俄语数学和物理问题的能力。该基准包含不同难度的问题，从学校级任务到大学级奥林匹克竞赛和入学考试题目。本文讨论了其创建动机，描述了数据集结构和评估方法，并介绍了测试各种模型的初始结果。分析结果显示，模型性能与所用令牌数之间存在关联，并指出了数学任务和物理任务在性能上的差异。 

---
# Dynamic Trust Calibration Using Contextual Bandits 

**Title (ZH)**: 基于上下文臂赛的动态信任校准 

**Authors**: Bruno M. Henrique, Eugene Santos Jr  

**Link**: [PDF](https://arxiv.org/pdf/2509.23497)  

**Abstract**: Trust calibration between humans and Artificial Intelligence (AI) is crucial for optimal decision-making in collaborative settings. Excessive trust can lead users to accept AI-generated outputs without question, overlooking critical flaws, while insufficient trust may result in disregarding valuable insights from AI systems, hindering performance. Despite its importance, there is currently no definitive and objective method for measuring trust calibration between humans and AI. Current approaches lack standardization and consistent metrics that can be broadly applied across various contexts, and they don't distinguish between the formation of opinions and subsequent human decisions. In this work, we propose a novel and objective method for dynamic trust calibration, introducing a standardized trust calibration measure and an indicator. By utilizing Contextual Bandits-an adaptive algorithm that incorporates context into decision-making-our indicator dynamically assesses when to trust AI contributions based on learned contextual information. We evaluate this indicator across three diverse datasets, demonstrating that effective trust calibration results in significant improvements in decision-making performance, as evidenced by 10 to 38% increase in reward metrics. These findings not only enhance theoretical understanding but also provide practical guidance for developing more trustworthy AI systems supporting decisions in critical domains, for example, disease diagnoses and criminal justice. 

**Abstract (ZH)**: 人类与人工智能的信任校准对于协作环境中的最优决策至关重要。过度信任可能导致用户无条件接受AI生成的结果，忽视关键缺陷；而不足的信任则可能导致忽视AI系统的有价值见解，阻碍性能提升。尽管其重要性日益凸显，目前仍缺乏一种既定且客观的方法来衡量人类与AI之间的信任校准。现有方法缺乏标准化且可广泛应用于不同场景的一致量化指标，并且未能区分意见形成与后续的人类决策。在此项研究中，我们提出了一种新颖且客观的动态信任校准方法，引入了标准化的信任校准测量指标和指示器。通过利用上下文臂拉姆达（Contextual Bandits）——一种将情境信息纳入决策过程的适应性算法，我们的指示器能够基于学习到的情境信息动态评估何时应信任AI的贡献。我们跨三个不同数据集评估了这一指标，结果表明有效的信任校准能够显著提升决策性能，表现为奖励指标提高了10%至38%。这些发现不仅深化了理论理解，还为在关键领域（如疾病诊断和刑事司法）开发更可靠的AI决策支持系统提供了实用指导。 

---
# Accurate Predictions in Education with Discrete Variational Inference 

**Title (ZH)**: 基于离散变分推断的教育中准确预测 

**Authors**: Tom Quilter, Anastasia Ilick, Anastasia Ilick, Richard Turner  

**Link**: [PDF](https://arxiv.org/pdf/2509.23484)  

**Abstract**: One of the largest drivers of social inequality is unequal access to personal tutoring, with wealthier individuals able to afford it, while the majority cannot. Affordable, effective AI tutors offer a scalable solution. We focus on adaptive learning, predicting whether a student will answer a question correctly, a key component of any effective tutoring system. Yet many platforms struggle to achieve high prediction accuracy, especially in data-sparse settings. To address this, we release the largest open dataset of professionally marked formal mathematics exam responses to date. We introduce a probabilistic modelling framework rooted in Item Response Theory (IRT) that achieves over 80 percent accuracy, setting a new benchmark for mathematics prediction accuracy of formal exam papers. Extending this, our collaborative filtering models incorporate topic-level skill profiles, but reveal a surprising and educationally significant finding, a single latent ability parameter alone is needed to achieve the maximum predictive accuracy. Our main contribution though is deriving and implementing a novel discrete variational inference framework, achieving our highest prediction accuracy in low-data settings and outperforming all classical IRT and matrix factorisation baselines. 

**Abstract (ZH)**: 一种广泛的社会不平等驱动因素是个人辅导的不平等获取，wealthier个体能够负担得起辅导，而大多数人则不能。负担得起且有效的AI辅导提供了一种可扩展的解决方案。我们关注自适应学习，预测学生是否能正确回答问题，这是任何有效辅导系统的关键组成部分。然而，许多平台在低数据情况下难以实现高预测准确性。为此，我们发布了迄今为止最大的专业标记形式数学考试答案开放数据集。我们提出了一种基于项目反应理论（IRT）的概率建模框架，实现了超过80%的准确性，为形式考试论文的数学预测准确性设立了新基准。在此基础上，我们的协同过滤模型结合了主题级别的技能配置文件，但揭示了一个令人惊讶且教育意义重大的发现——单个潜在能力参数足以实现最大预测准确性。然而，我们的主要贡献是推导并实现了一种新颖的离散变分推断框架，在低数据环境下实现了最高的预测准确性，并优于所有经典的IRT和矩阵分解基线。 

---
# GeoBS: Information-Theoretic Quantification of Geographic Bias in AI Models 

**Title (ZH)**: GeoBS:基于信息理论的地理偏差量化方法在AI模型中的应用 

**Authors**: Zhangyu Wang, Nemin Wu, Qian Cao, Jiangnan Xia, Zeping Liu, Yiqun Xie, Akshay Nambi, Tanuja Ganu, Ni Lao, Ninghao Liu, Gengchen Mai  

**Link**: [PDF](https://arxiv.org/pdf/2509.23482)  

**Abstract**: The widespread adoption of AI models, especially foundation models (FMs), has made a profound impact on numerous domains. However, it also raises significant ethical concerns, including bias issues. Although numerous efforts have been made to quantify and mitigate social bias in AI models, geographic bias (in short, geo-bias) receives much less attention, which presents unique challenges. While previous work has explored ways to quantify geo-bias, these measures are model-specific (e.g., mean absolute deviation of LLM ratings) or spatially implicit (e.g., average fairness scores of all spatial partitions). We lack a model-agnostic, universally applicable, and spatially explicit geo-bias evaluation framework that allows researchers to fairly compare the geo-bias of different AI models and to understand what spatial factors contribute to the geo-bias. In this paper, we establish an information-theoretic framework for geo-bias evaluation, called GeoBS (Geo-Bias Scores). We demonstrate the generalizability of the proposed framework by showing how to interpret and analyze existing geo-bias measures under this framework. Then, we propose three novel geo-bias scores that explicitly take intricate spatial factors (multi-scalability, distance decay, and anisotropy) into consideration. Finally, we conduct extensive experiments on 3 tasks, 8 datasets, and 8 models to demonstrate that both task-specific GeoAI models and general-purpose foundation models may suffer from various types of geo-bias. This framework will not only advance the technical understanding of geographic bias but will also establish a foundation for integrating spatial fairness into the design, deployment, and evaluation of AI systems. 

**Abstract (ZH)**: AI模型，尤其是基础模型（FMs）的广泛应用对众多领域产生了深远影响，但也引发了显著的伦理问题，包括公平性问题。尽管已做了大量努力来量化和减轻AI模型中的社会偏见，但地理偏见（简称Geo-bias）却受到较少关注，这提出了独特的挑战。尽管以往工作已探索了量化Geo-bias的方法，但这些方法通常是模型特定的（例如，大语言模型评分的绝对均差）或空间隐含的（例如，所有空间分区平均公平性得分）。缺乏一种模型无关的、普遍适用的、空间明确的Geo-bias评价框架，使研究人员无法公平比较不同AI模型的Geo-bias，并理解哪些空间因素导致了Geo-bias。在本文中，我们建立了基于信息理论的Geo-bias评价框架，称为GeoBS（Geo-bias Scores）。我们展示了该框架的一般适用性，通过展示如何在该框架下解释和分析现有Geo-bias度量。然后，我们提出了三种新的Geo-bias评分，明确考虑了复杂的空间因素（多级可扩展性、距离衰减和各向异性）。最后，我们在3项任务、8个数据集和8个模型上进行广泛实验，证明了专门针对任务的GeoAI模型和通用基础模型都可能遭受各种类型的Geo-bias。该框架不仅将推动地理偏见的技术理解，还将为将空间公平性整合到AI系统的设计、部署和评估中奠定基础。 

---
# ViTSP: A Vision Language Models Guided Framework for Large-Scale Traveling Salesman Problems 

**Title (ZH)**: ViTSP: 由vision-language模型指导的大规模旅行商问题框架 

**Authors**: Zhuoli Yin, Yi Ding, Reem Khir, Hua Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.23465)  

**Abstract**: Solving Traveling Salesman Problem (TSP) is NP-hard yet fundamental for wide real-world applications. Classical exact methods face challenges in scaling, and heuristic methods often require domain-specific parameter calibration. While learning-based approaches have shown promise, they suffer from poor generalization and limited scalability due to fixed training data. This work proposes ViTSP, a novel framework that leverages pre-trained vision language models (VLMs) to visually guide the solution process for large-scale TSPs. The VLMs function to identify promising small-scale subproblems from a visualized TSP instance, which are then efficiently optimized using an off-the-shelf solver to improve the global solution. ViTSP bypasses the dedicated model training at the user end while maintaining effectiveness across diverse instances. Experiments on real-world TSP instances ranging from 1k to 88k nodes demonstrate that ViTSP consistently achieves solutions with average optimality gaps below 0.2%, outperforming existing learning-based methods. Under the same runtime budget, it surpasses the best-performing heuristic solver, LKH-3, by reducing its gaps by 12% to 100%, particularly on very-large-scale instances with more than 10k nodes. Our framework offers a new perspective in hybridizing pre-trained generative models and operations research solvers in solving combinatorial optimization problems, with practical implications for integration into more complex logistics systems. The code is available at this https URL. 

**Abstract (ZH)**: 基于预训练视觉语言模型的Traveling Salesman Problem求解新框架 

---
# Beyond Embeddings: Interpretable Feature Extraction for Binary Code Similarity 

**Title (ZH)**: 超越嵌入：二进制代码相似性解释性特征提取 

**Authors**: Charles E. Gagnon, Steven H. H. Ding, Philippe Charland, Benjamin C. M. Fung  

**Link**: [PDF](https://arxiv.org/pdf/2509.23449)  

**Abstract**: Binary code similarity detection is a core task in reverse engineering. It supports malware analysis and vulnerability discovery by identifying semantically similar code in different contexts. Modern methods have progressed from manually engineered features to vector representations. Hand-crafted statistics (e.g., operation ratios) are interpretable, but shallow and fail to generalize. Embedding-based methods overcome this by learning robust cross-setting representations, but these representations are opaque vectors that prevent rapid verification. They also face a scalability-accuracy trade-off, since high-dimensional nearest-neighbor search requires approximations that reduce precision. Current approaches thus force a compromise between interpretability, generalizability, and scalability.
We bridge these gaps using a language model-based agent to conduct structured reasoning analysis of assembly code and generate features such as input/output types, side effects, notable constants, and algorithmic intent. Unlike hand-crafted features, they are richer and adaptive. Unlike embeddings, they are human-readable, maintainable, and directly searchable with inverted or relational indexes. Without any matching training, our method respectively achieves 42% and 62% for recall@1 in cross-architecture and cross-optimization tasks, comparable to embedding methods with training (39% and 34%). Combined with embeddings, it significantly outperforms the state-of-the-art, demonstrating that accuracy, scalability, and interpretability can coexist. 

**Abstract (ZH)**: 基于二进制代码相似性检测的逆向工程核心任务：结合语言模型的结构化推理在架构和优化任务中的应用 

---
# Democratizing AI scientists using ToolUniverse 

**Title (ZH)**: 使用ToolUniverse使人工智能科学家 democratization 

**Authors**: Shanghua Gao, Richard Zhu, Pengwei Sui, Zhenglun Kong, Sufian Aldogom, Yepeng Huang, Ayush Noori, Reza Shamji, Krishna Parvataneni, Theodoros Tsiligkaridis, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2509.23426)  

**Abstract**: AI scientists are emerging computational systems that serve as collaborative partners in discovery. These systems remain difficult to build because they are bespoke, tied to rigid workflows, and lack shared environments that unify tools, data, and analyses into a common ecosystem. In omics, unified ecosystems have transformed research by enabling interoperability, reuse, and community-driven development; AI scientists require comparable infrastructure. We present ToolUniverse, an ecosystem for building AI scientists from any language or reasoning model, whether open or closed. TOOLUNIVERSE standardizes how AI scientists identify and call tools, integrating more than 600 machine learning models, datasets, APIs, and scientific packages for data analysis, knowledge retrieval, and experimental design. It automatically refines tool interfaces for correct use by AI scientists, creates new tools from natural language descriptions, iteratively optimizes tool specifications, and composes tools into agentic workflows. In a case study of hypercholesterolemia, ToolUniverse was used to create an AI scientist to identify a potent analog of a drug with favorable predicted properties. The open-source ToolUniverse is available at this https URL. 

**Abstract (ZH)**: AI科学家是新兴的计算系统，作为发现过程中的合作者。由于它们是定制的、与刚性的工作流程相关联且缺乏统一的环境将工具、数据和分析集成到一个共同生态系统中，因此构建这些系统仍然具有挑战性。在omics领域，统一的生态系统通过促进互操作性、重用和社区驱动的发展而转变了研究；AI科学家需要类似的基础设施。我们提出ToolUniverse，这是一个构建来自任何语言或推理模型（无论是开源还是封闭）的AI科学家的生态系统。TOOLUNIVERSE标准化了AI科学家识别和调用工具的方式，集成了超过600个机器学习模型、数据集、API和科学包，用于数据分析、知识检索和实验设计。ToolUniverse自动细化工具接口以供AI科学家正确使用，从自然语言描述中创建新工具，迭代优化工具规范，并组成具有自主性的工作流程。在高胆固醇研究案例中，ToolUniverse被用于创建一个AI科学家来识别一种具有有利预测性质的药物类似物。开源ToolUniverse可从该链接访问：this https URL。 

---
# Socio-Economic Model of AI Agents 

**Title (ZH)**: AI代理的经济社会模型 

**Authors**: Yuxinyue Qian, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23270)  

**Abstract**: Modern socio-economic systems are undergoing deep integration with artificial intelligence technologies. This paper constructs a heterogeneous agent-based modeling framework that incorporates both human workers and autonomous AI agents, to study the impact of AI collaboration under resource constraints on aggregate social output. We build five progressively extended models: Model 1 serves as the baseline of pure human collaboration; Model 2 introduces AI as collaborators; Model 3 incorporates network effects among agents; Model 4 treats agents as independent producers; and Model 5 integrates both network effects and independent agent production. Through theoretical derivation and simulation analysis, we find that the introduction of AI agents can significantly increase aggregate social output. When considering network effects among agents, this increase exhibits nonlinear growth far exceeding the simple sum of individual contributions. Under the same resource inputs, treating agents as independent producers provides higher long-term growth potential; introducing network effects further demonstrates strong characteristics of increasing returns to scale. 

**Abstract (ZH)**: 现代社会经济系统正与人工智能技术深度融合。本文构建了一个包含人类工人和自主AI代理的异质代理基于模型框架，研究资源约束条件下AI协作对总体社会产出的影响。我们构建了五个逐步扩展的模型：模型1作为纯人类协作的基线；模型2引入AI作为合作者；模型3纳入代理间的网络效应；模型4将代理视为独立生产者；模型5结合了网络效应和独立代理生产。通过理论推导和仿真分析发现，引入AI代理可以显著增加总体社会产出。考虑代理间的网络效应时，这种增长呈现出非线性增长，远超个体贡献的简单相加。在相同的资源输入下，将代理视为独立生产者提供了更高的长期增长潜力；引入网络效应进一步展示了规模收益递增的强烈特征。 

---
# Limit Analysis for Symbolic Multi-step Reasoning Tasks with Information Propagation Rules Based on Transformers 

**Title (ZH)**: 基于Transformer的信息传播规则符号多步推理任务的极限分析 

**Authors**: Tian Qin, Yuhan Chen, Zhiwei Wang, Zhi-Qin John Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23178)  

**Abstract**: Transformers are able to perform reasoning tasks, however the intrinsic mechanism remains widely open. In this paper we propose a set of information propagation rules based on Transformers and utilize symbolic reasoning tasks to theoretically analyze the limit reasoning steps. We show that the limit number of reasoning steps is between $O(3^{L-1})$ and $O(2^{L-1})$ for a model with $L$ attention layers in a single-pass. 

**Abstract (ZH)**: 基于Transformer的信息传播规则及单一.pass中模型Attention层数量为L时极限推理步数的理论分析 

---
# AI-Enhanced Distributed Channel Access for Collision Avoidance in Future Wi-Fi 8 

**Title (ZH)**: AI增强的分布式信道访问技术以避免未来Wi-Fi 8中的碰撞 

**Authors**: Jinzhe Pan, Jingqing Wang, Yuehui Ouyang, Wenchi Cheng, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23154)  

**Abstract**: The exponential growth of wireless devices and stringent reliability requirements of emerging applications demand fundamental improvements in distributed channel access mechanisms for unlicensed bands. Current Wi-Fi systems, which rely on binary exponential backoff (BEB), suffer from suboptimal collision resolution in dense deployments and persistent fairness challenges due to inherent randomness. This paper introduces a multi-agent reinforcement learning framework that integrates artificial intelligence (AI) optimization with legacy device coexistence. We first develop a dynamic backoff selection mechanism that adapts to real-time channel conditions through access deferral events while maintaining full compatibility with conventional CSMA/CA operations. Second, we introduce a fairness quantification metric aligned with enhanced distributed channel access (EDCA) principles to ensure equitable medium access opportunities. Finally, we propose a centralized training decentralized execution (CTDE) architecture incorporating neighborhood activity patterns as observational inputs, optimized via constrained multi-agent proximal policy optimization (MAPPO) to jointly minimize collisions and guarantee fairness. Experimental results demonstrate that our solution significantly reduces collision probability compared to conventional BEB while preserving backward compatibility with commercial Wi-Fi devices. The proposed fairness metric effectively eliminates starvation risks in heterogeneous scenarios. 

**Abstract (ZH)**: 无线设备的指数增长和新兴应用严格的功能要求促使对未授权频带中分布式信道访问机制进行根本性改进。当前依赖二进制指数退避（BEB）的Wi-Fi系统在密集部署中面临次优碰撞解决和固有的随机性导致的持续公平性挑战。本文提出了一种集成了人工智能优化与传统设备共存的多智能体强化学习框架。首先，我们开发了一种动态退避选择机制，该机制通过接入延迟事件适应实时信道条件，同时保持与传统CSMA/CA操作的完全兼容性。其次，我们引入了一个与增强分布式信道访问（EDCA）原则相一致的公平性量化指标，以确保公平的介质访问机会。最后，我们提出了一个集成邻域活动模式作为观测输入的集中训练分布式执行（CTDE）架构，通过受约束的多智能体近端策略优化（MAPPO）优化，以联合最小化碰撞并保证公平性。实验结果表明，与传统BEB相比，我们的解决方案显著降低了碰撞概率，同时保持了与商用Wi-Fi设备的向后兼容性。提出的公平性指标在异构场景中有效地消除了饿死风险。 

---
# Coordination Requires Simplification: Thermodynamic Bounds on Multi-Objective Compromise in Natural and Artificial Intelligence 

**Title (ZH)**: 协调需要简化：自然与人工智能多目标妥协的热力学界限 

**Authors**: Atma Anand  

**Link**: [PDF](https://arxiv.org/pdf/2509.23144)  

**Abstract**: Information-processing systems coordinating across multiple agents and objectives face fundamental thermodynamic constraints. We show that solutions with maximum utility to act as coordination focal points have much higher selection pressure for being findable across agents rather than accuracy. We derive that the information-theoretic minimum description length of coordination protocols to precision $\varepsilon$ scales as $L(P)\geq NK\log_2 K+N^2d^2\log (1/\varepsilon)$ for $N$ agents with $d$ potentially conflicting objectives and internal model complexity $K$. This scaling forces progressive simplification, with coordination dynamics changing the environment itself and shifting optimization across hierarchical levels. Moving from established focal points requires re-coordination, creating persistent metastable states and hysteresis until significant environmental shifts trigger phase transitions through spontaneous symmetry breaking. We operationally define coordination temperature to predict critical phenomena and estimate coordination work costs, identifying measurable signatures across systems from neural networks to restaurant bills to bureaucracies. Extending the topological version of Arrow's theorem on the impossibility of consistent preference aggregation, we find it recursively binds whenever preferences are combined. This potentially explains the indefinite cycling in multi-objective gradient descent and alignment faking in Large Language Models trained with reinforcement learning with human feedback. We term this framework Thermodynamic Coordination Theory (TCT), which demonstrates that coordination requires radical information loss. 

**Abstract (ZH)**: 信息处理系统在多代理和多目标之间协调时面临基本的热力学约束。我们表明，具有最大效用作为协调焦点的解决方案在被多个代理发现方面的选择压力比准确性更高。我们推导出，用于精确度为ε的协调协议的信息论最小描述长度为$L(P)\geq NK\log_2 K+N^2d^2\log (1/\varepsilon)$，适用于拥有d个潜在冲突目标和内部模型复杂度为K的N个代理。这种缩放迫使逐步简化，协调动力学会改变环境本身并沿层级重塑优化。从已建立的焦点迁移到新的焦点需要重新协调，形成持久的亚稳态和滞回，直到环境发生重要变化时，通过自发对称破缺触发相变。我们操作性地定义协调温度来预测关键现象，估计协调工作成本，并识别从神经网络到餐馆账单再到官僚机构等系统中的可测量特征。扩展Arrow不可能一致偏好聚合定理的拓扑版本，我们发现它每当偏好被结合时都会递归地适用。这可能解释了多目标梯度下降中的无限循环以及使用强化学习和人类反馈训练的大语言模型中的对齐作弊现象。我们称这一框架为热力学协调理论（TCT），并证明协调需要根本的信息丢失。 

---
# SysMoBench: Evaluating AI on Formally Modeling Complex Real-World Systems 

**Title (ZH)**: SysMoBench: 评估AI在正式建模复杂现实系统中的性能 

**Authors**: Qian Cheng, Ruize Tang, Emilie Ma, Finn Hackett, Peiyang He, Yiming Su, Ivan Beschastnikh, Yu Huang, Xiaoxing Ma, Tianyin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23130)  

**Abstract**: Formal models are essential to specifying large, complex computer systems and verifying their correctness, but are notoriously expensive to write and maintain. Recent advances in generative AI show promise in generating certain forms of specifications. However, existing work mostly targets small code, not complete systems. It is unclear whether AI can deal with realistic system artifacts, as this requires abstracting their complex behavioral properties into formal models. We present SysMoBench, a benchmark that evaluates AI's ability to formally model large, complex systems. We focus on concurrent and distributed systems, which are keystones of today's critical computing infrastructures, encompassing operating systems and cloud infrastructure. We use TLA+, the it de facto specification language for concurrent and distributed systems, though the benchmark can be extended to other specification languages. We address the primary challenge of evaluating AI-generated models by automating metrics like syntactic and runtime correctness, conformance to system code, and invariant correctness. SysMoBench currently includes nine diverse system artifacts: the Raft implementation of Etcd and Redis, the Spinlock and Mutex in Asterinas OS, etc.; more artifacts are being actively added. SysMoBench enables us to understand the capabilities and limitations of today's LLMs and agents, putting tools in this area on a firm footing and opening up promising new research directions. 

**Abstract (ZH)**: SysMoBench：评估AI构建大型复杂系统形式模型的能力 

---
# Creative Adversarial Testing (CAT): A Novel Framework for Evaluating Goal-Oriented Agentic AI Systems 

**Title (ZH)**: 创造性对抗性测试（CAT）：一种评估目标导向自主AI系统的新型框架 

**Authors**: Hassen Dhrif  

**Link**: [PDF](https://arxiv.org/pdf/2509.23006)  

**Abstract**: Agentic AI represents a paradigm shift in enhancing the capabilities of generative AI models. While these systems demonstrate immense potential and power, current evaluation techniques primarily focus on assessing their efficacy in identifying appropriate agents, tools, and parameters. However, a critical gap exists in evaluating the alignment between an Agentic AI system's tasks and its overarching goals. This paper introduces the Creative Adversarial Testing (CAT) framework, a novel approach designed to capture and analyze the complex relationship between Agentic AI tasks and the system's intended objectives.
We validate the CAT framework through extensive simulation using synthetic interaction data modeled after Alexa+ audio services, a sophisticated Agentic AI system that shapes the user experience for millions of users globally. This synthetic data approach enables comprehensive testing of edge cases and failure modes while protecting user privacy. Our results demonstrate that the CAT framework provides unprecedented insights into goal-task alignment, enabling more effective optimization and development of Agentic AI systems. 

**Abstract (ZH)**: 代理型AI代表了增强生成型AI模型能力的范式转变。虽然这些系统展现出了巨大的潜力和力量，当前的评估技术主要集中在评估其在识别合适代理、工具和参数方面的有效性。然而，在评估代理型AI系统的任务与其整体目标之间的对齐方面存在关键缺口。本文引入了创意对抗测试（CAT）框架，这是一种旨在捕捉和分析代理型AI任务与系统预期目标之间复杂关系的新方法。 

---
# AI Noether -- Bridging the Gap Between Scientific Laws Derived by AI Systems and Canonical Knowledge via Abductive Inference 

**Title (ZH)**: AI Noether——通过 abduction 推论弥合由 AI 系统推导出的科学定律与经典知识之间的差距 

**Authors**: Karan Srivastava, Sanjeeb Dash, Ryan Cory-Wright, Barry Trager, Lior Horesh  

**Link**: [PDF](https://arxiv.org/pdf/2509.23004)  

**Abstract**: A core goal in modern science is to harness recent advances in AI and computer processing to automate and accelerate the scientific method. Symbolic regression can fit interpretable models to data, but these models often sit outside established theory. Recent systems (e.g., AI Descartes, AI Hilbert) enforce derivability from prior axioms. However, sometimes new data and associated hypotheses derived from data are not consistent with existing theory because the existing theory is incomplete or incorrect. Automating abductive inference to close this gap remains open. We propose a solution: an algebraic geometry-based system that, given an incomplete axiom system and a hypothesis that it cannot explain, automatically generates a minimal set of missing axioms that suffices to derive the axiom, as long as axioms and hypotheses are expressible as polynomial equations. We formally establish necessary and sufficient conditions for the successful retrieval of such axioms. We illustrate the efficacy of our approach by demonstrating its ability to explain Kepler's third law and a few other laws, even when key axioms are absent. 

**Abstract (ZH)**: 现代科学的核心目标是利用最近在人工智能和计算机处理方面的进展来自动化和加速科学研究方法。基于代数几何的方法可以在给定不完整公理系统和现有公理无法解释的假设时，自动生成一套最小化的缺失公理，以推导出所需的公理，前提是公理和假设可以表示为多项式方程。我们正式建立了成功检索此类公理的必要和充分条件。通过展示其能够解释开普勒第三定律和其他一些定律的能力，即使在关键公理缺席的情况下，我们说明了该方法的有效性。 

---
# Guided Diffusion for the Discovery of New Superconductors 

**Title (ZH)**: 引导性扩散在新型超导体的发现中的应用 

**Authors**: Pawan Prakash, Jason B. Gibson, Zhongwei Li, Gabriele Di Gianluca, Juan Esquivel, Eric Fuemmeler, Benjamin Geisler, Jung Soo Kim, Adrian Roitberg, Ellad B. Tadmor, Mingjie Liu, Stefano Martiniani, Gregory R. Stewart, James J. Hamlin, Peter J. Hirschfeld, Richard G. Hennig  

**Link**: [PDF](https://arxiv.org/pdf/2509.25186)  

**Abstract**: The inverse design of materials with specific desired properties, such as high-temperature superconductivity, represents a formidable challenge in materials science due to the vastness of chemical and structural space. We present a guided diffusion framework to accelerate the discovery of novel superconductors. A DiffCSP foundation model is pretrained on the Alexandria Database and fine-tuned on 7,183 superconductors with first principles derived labels. Employing classifier-free guidance, we sample 200,000 structures, which lead to 34,027 unique candidates. A multistage screening process that combines machine learning and density functional theory (DFT) calculations to assess stability and electronic properties, identifies 773 candidates with DFT-calculated $T_\mathrm{c}>5$ K. Notably, our generative model demonstrates effective property-driven design. Our computational findings were validated against experimental synthesis and characterization performed as part of this work, which highlighted challenges in sparsely charted chemistries. This end-to-end workflow accelerates superconductor discovery while underscoring the challenge of predicting and synthesizing experimentally realizable materials. 

**Abstract (ZH)**: 具有特定 desired 特性的材料的逆设计：加速新型高温超导体的发现 

---
# NAIPv2: Debiased Pairwise Learning for Efficient Paper Quality Estimation 

**Title (ZH)**: NAIPv2: 去偏见的成对学习以实现高效论文质量估计 

**Authors**: Penghai Zhao, Jinyu Tian, Qinghua Xing, Xin Zhang, Zheng Li, Jianjun Qian, Ming-Ming Cheng, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.25179)  

**Abstract**: The ability to estimate the quality of scientific papers is central to how both humans and AI systems will advance scientific knowledge in the future. However, existing LLM-based estimation methods suffer from high inference cost, whereas the faster direct score regression approach is limited by scale inconsistencies. We present NAIPv2, a debiased and efficient framework for paper quality estimation. NAIPv2 employs pairwise learning within domain-year groups to reduce inconsistencies in reviewer ratings and introduces the Review Tendency Signal (RTS) as a probabilistic integration of reviewer scores and confidences. To support training and evaluation, we further construct NAIDv2, a large-scale dataset of 24,276 ICLR submissions enriched with metadata and detailed structured content. Trained on pairwise comparisons but enabling efficient pointwise prediction at deployment, NAIPv2 achieves state-of-the-art performance (78.2% AUC, 0.432 Spearman), while maintaining scalable, linear-time efficiency at inference. Notably, on unseen NeurIPS submissions, it further demonstrates strong generalization, with predicted scores increasing consistently across decision categories from Rejected to Oral. These findings establish NAIPv2 as a debiased and scalable framework for automated paper quality estimation, marking a step toward future scientific intelligence systems. Code and dataset are released at this https URL. 

**Abstract (ZH)**: NAIPv2：去偏差且高效的论文质量估计框架 

---
# GLASS Flows: Transition Sampling for Alignment of Flow and Diffusion Models 

**Title (ZH)**: GLASS 流动: 流与扩散模型对齐的转换采样方法 

**Authors**: Peter Holderrieth, Uriel Singer, Tommi Jaakkola, Ricky T. Q. Chen, Yaron Lipman, Brian Karrer  

**Link**: [PDF](https://arxiv.org/pdf/2509.25170)  

**Abstract**: The performance of flow matching and diffusion models can be greatly improved at inference time using reward alignment algorithms, yet efficiency remains a major limitation. While several algorithms were proposed, we demonstrate that a common bottleneck is the sampling method these algorithms rely on: many algorithms require to sample Markov transitions via SDE sampling, which is significantly less efficient and often less performant than ODE sampling. To remove this bottleneck, we introduce GLASS Flows, a new sampling paradigm that simulates a "flow matching model within a flow matching model" to sample Markov transitions. As we show in this work, this "inner" flow matching model can be retrieved from a pre-trained model without any re-training, combining the efficiency of ODEs with the stochastic evolution of SDEs. On large-scale text-to-image models, we show that GLASS Flows eliminate the trade-off between stochastic evolution and efficiency. Combined with Feynman-Kac Steering, GLASS Flows improve state-of-the-art performance in text-to-image generation, making it a simple, drop-in solution for inference-time scaling of flow and diffusion models. 

**Abstract (ZH)**: 使用奖励对齐算法可以在推理时大幅提高流动匹配和扩散模型的性能，但效率仍然是一个主要限制。虽然提出了几种算法，但我们展示了这些算法依赖的采样方法是一个共同瓶颈：许多算法需要通过SDE采样来采样马尔科夫转换，这在效率和性能上通常远逊于ODE采样。为了消除这一瓶颈，我们提出了GLASS Flows，这是一种新的采样范式，模拟“在一个流动匹配模型内部模拟一个流动匹配模型”来采样马尔科夫转换。如本文所示，这种“内部”的流动匹配模型可以从预先训练好的模型中提取出来，无需重新训练，从而结合了ODE的高效性和SDE的随机演化。在大规模文本到图像模型上，我们展示了GLASS Flows消除了随机演化与效率之间的权衡。结合费曼-卡茨引导，GLASS Flows提高了文本到图像生成的最新性能，使其成为流动和扩散模型推理时间扩展的一个简单即插即用解决方案。 

---
# Chance-constrained Flow Matching for High-Fidelity Constraint-aware Generation 

**Title (ZH)**: 高保真约束aware生成的机遇约束流匹配 

**Authors**: Jinhao Liang, Yixuan Sun, Anirban Samaddar, Sandeep Madireddy, Ferdinando Fioretto  

**Link**: [PDF](https://arxiv.org/pdf/2509.25157)  

**Abstract**: Generative models excel at synthesizing high-fidelity samples from complex data distributions, but they often violate hard constraints arising from physical laws or task specifications. A common remedy is to project intermediate samples onto the feasible set; however, repeated projection can distort the learned distribution and induce a mismatch with the data manifold. Thus, recent multi-stage procedures attempt to defer projection to clean samples during sampling, but they increase algorithmic complexity and accumulate errors across steps. This paper addresses these challenges by proposing a novel training-free method, Chance-constrained Flow Matching (CCFM), that integrates stochastic optimization into the sampling process, enabling effective enforcement of hard constraints while maintaining high-fidelity sample generation. Importantly, CCFM guarantees feasibility in the same manner as conventional repeated projection, yet, despite operating directly on noisy intermediate samples, it is theoretically equivalent to projecting onto the feasible set defined by clean samples. This yields a sampler that mitigates distributional distortion. Empirical experiments show that CCFM outperforms current state-of-the-art constrained generative models in modeling complex physical systems governed by partial differential equations and molecular docking problems, delivering higher feasibility and fidelity. 

**Abstract (ZH)**: 机会约束流量匹配：一种训练-Free 的高保真生成方法 

---
# Paired by the Teacher: Turning Unpaired Data into High-Fidelity Pairs for Low-Resource Text Generation 

**Title (ZH)**: 由老师配对：将无配对数据转换为低资源文本生成的高保真配对 

**Authors**: Yen-Ju Lu, Thomas Thebaud, Laureano Moro-Velazquez, Najim Dehak, Jesus Villalba  

**Link**: [PDF](https://arxiv.org/pdf/2509.25144)  

**Abstract**: We present Paired by the Teacher (PbT), a two-stage teacher-student pipeline that synthesizes accurate input-output pairs without human labels or parallel data. In many low-resource natural language generation (NLG) scenarios, practitioners may have only raw outputs, like highlights, recaps, or questions, or only raw inputs, such as articles, dialogues, or paragraphs, but seldom both. This mismatch forces small models to learn from very few examples or rely on costly, broad-scope synthetic examples produced by large LLMs. PbT addresses this by asking a teacher LLM to compress each unpaired example into a concise intermediate representation (IR), and training a student to reconstruct inputs from IRs. This enables outputs to be paired with student-generated inputs, yielding high-quality synthetic data. We evaluate PbT on five benchmarks-document summarization (XSum, CNNDM), dialogue summarization (SAMSum, DialogSum), and question generation (SQuAD)-as well as an unpaired setting on SwitchBoard (paired with DialogSum summaries). An 8B student trained only on PbT data outperforms models trained on 70 B teacher-generated corpora and other unsupervised baselines, coming within 1.2 ROUGE-L of human-annotated pairs and closing 82% of the oracle gap at one-third the annotation cost of direct synthesis. Human evaluation on SwitchBoard further confirms that only PbT produces concise, faithful summaries aligned with the target style, highlighting its advantage of generating in-domain sources that avoid the mismatch, limiting direct synthesis. 

**Abstract (ZH)**: Paired by the Teacher：一种无需人工标签或平行数据的两阶段教师-学生管道 

---
# Towards Personalized Deep Research: Benchmarks and Evaluations 

**Title (ZH)**: 面向个性化深度研究的基准与评估 

**Authors**: Yuan Liang, Jiaxian Li, Yuqing Wang, Piaohong Wang, Motong Tian, Pai Liu, Shuofei Qiao, Runnan Fang, He Zhu, Ge Zhang, Minghao Liu, Yuchen Eleanor Jiang, Ningyu Zhang, Wangchunshu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.25106)  

**Abstract**: Deep Research Agents (DRAs) can autonomously conduct complex investigations and generate comprehensive reports, demonstrating strong real-world potential. However, existing evaluations mostly rely on close-ended benchmarks, while open-ended deep research benchmarks remain scarce and typically neglect personalized scenarios. To bridge this gap, we introduce Personalized Deep Research Bench, the first benchmark for evaluating personalization in DRAs. It pairs 50 diverse research tasks across 10 domains with 25 authentic user profiles that combine structured persona attributes with dynamic real-world contexts, yielding 250 realistic user-task queries. To assess system performance, we propose the PQR Evaluation Framework, which jointly measures (P) Personalization Alignment, (Q) Content Quality, and (R) Factual Reliability. Our experiments on a range of systems highlight current capabilities and limitations in handling personalized deep research. This work establishes a rigorous foundation for developing and evaluating the next generation of truly personalized AI research assistants. 

**Abstract (ZH)**: 个性化的深度研究基准：评估DRAs的首个基准 

---
# jina-reranker-v3: Last but Not Late Interaction for Document Reranking 

**Title (ZH)**: jina-reranker-v3：最后但并非最不重要交互的文档重排 

**Authors**: Feng Wang, Yuqing Li, Han Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2509.25085)  

**Abstract**: jina-reranker-v3 is a 0.6B parameter multilingual document reranker that introduces a novel last but not late interaction. Unlike late interaction models such as ColBERT that perform separate encoding followed by multi-vector matching, our approach conducts causal self-attention between query and documents within the same context window, enabling rich cross-document interactions before extracting contextual embeddings from the last token of each document. This compact architecture achieves state-of-the-art BEIR performance with 61.94 nDCG@10 while being ten times smaller than generative listwise rerankers. 

**Abstract (ZH)**: Jina-Reranker-v3是一种参数量为0.6B的多语言文档重排序器，引入了一种新颖的非晚交互方式。不同于ColBERT等晚交互模型在分别编码后进行多向量匹配的做法，我们的方法在同一个上下文窗口内对查询和文档之间进行因果自注意力交互，从而在提取每个文档最后一词的上下文嵌入之前实现丰富的跨文档交互。这种紧凑的架构在BEIR上达到了61.94的nDCG@10性能，同时仅有生成型列表重排序器的十分之一大小。 

---
# Scaling Generalist Data-Analytic Agents 

**Title (ZH)**: 扩展通用数据分析师代理 

**Authors**: Shuofei Qiao, Yanqiu Zhao, Zhisong Qiu, Xiaobin Wang, Jintian Zhang, Zhao Bin, Ningyu Zhang, Yong Jiang, Pengjun Xie, Fei Huang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.25084)  

**Abstract**: Data-analytic agents are emerging as a key catalyst for automated scientific discovery and for the vision of Innovating AI. Current approaches, however, rely heavily on prompt engineering over proprietary models, while open-source models struggle to face diverse-format, large-scale data files and long-horizon, multi-step reasoning that real-world analytics demands. This paper introduces DataMind, a scalable data synthesis and agent training recipe designed to build generalist data-analytic agents. DataMind tackles three key challenges in building open-source data-analytic agents, including insufficient data resources, improper training strategy, and unstable code-based multi-turn rollout. Concretely, DataMind applies 1) a fine-grained task taxonomy and a recursive easy-to-hard task composition mechanism to increase the diversity and difficulty of synthesized queries; 2) a knowledge-augmented trajectory sampling strategy followed by model-based and rule-based filtering; 3) a dynamically adjustable training objective combining both SFT and RL losses; 4) a memory-frugal and stable code-based multi-turn rollout framework. Built on DataMind, we curate DataMind-12K, a high-quality trajectory set spanning diverse domains, task categories, and data file formats for data-analytic tasks. Trained on DataMind-12K, our DataMind-14B achieves state-of-the-art with an average score of 71.16% on multiple data analysis benchmarks, outperforming the strongest proprietary baselines DeepSeek-V3.1 and GPT-5. Our DataMind-7B also performs best among all open-source models with a score of 68.10%. We also incorporate some empirical insights gained from our exploratory trials into the analysis experiments, aiming to provide actionable insights about agentic training for the community. We will release DataMind-12K and DataMind-7B,14B for the community's future research. 

**Abstract (ZH)**: DataMind：面向开源数据分析代理的可扩展数据合成与智能体训练方法 

---
# Learning Distinguishable Representations in Deep Q-Networks for Linear Transfer 

**Title (ZH)**: 基于深层Q网络的学习可区分表示方法及其在线性转移中的应用 

**Authors**: Sooraj Sathish, Keshav Goyal, Raghuram Bharadwaj Diddigi  

**Link**: [PDF](https://arxiv.org/pdf/2509.24947)  

**Abstract**: Deep Reinforcement Learning (RL) has demonstrated success in solving complex sequential decision-making problems by integrating neural networks with the RL framework. However, training deep RL models poses several challenges, such as the need for extensive hyperparameter tuning and high computational costs. Transfer learning has emerged as a promising strategy to address these challenges by enabling the reuse of knowledge from previously learned tasks for new, related tasks. This avoids the need for retraining models entirely from scratch. A commonly used approach for transfer learning in RL is to leverage the internal representations learned by the neural network during training. Specifically, the activations from the last hidden layer can be viewed as refined state representations that encapsulate the essential features of the input. In this work, we investigate whether these representations can be used as input for training simpler models, such as linear function approximators, on new tasks. We observe that the representations learned by standard deep RL models can be highly correlated, which limits their effectiveness when used with linear function approximation. To mitigate this problem, we propose a novel deep Q-learning approach that introduces a regularization term to reduce positive correlations between feature representation of states. By leveraging these reduced correlated features, we enable more effective use of linear function approximation in transfer learning. Through experiments and ablation studies on standard RL benchmarks and MinAtar games, we demonstrate the efficacy of our approach in improving transfer learning performance and thereby reducing computational overhead. 

**Abstract (ZH)**: 深度强化学习（RL）通过将神经网络与RL框架结合，展示了在解决复杂序贯决策问题方面的成功。然而，训练深度RL模型面临着诸多挑战，如超参数调优需求广泛和高昂的计算成本。迁移学习作为一种有前途的策略，通过利用先前学习任务中获得的知识来解决这些挑战，使其能够为目标相关的新任务复用知识，从而避免从头重新训练模型。在RL中，使用迁移学习的一个常用方法是利用神经网络在训练过程中学习到的内部表示。具体而言，最后一隐藏层的激活可以视为改进的状态表示，这些表示包含了输入的重要特征。在本工作中，我们研究这些表示是否可以作为输入用于训练简单模型，如线性函数逼近器，在新任务上的训练。我们发现标准深度RL模型学习到的表示之间高度相关，这限制了其在使用线性函数逼近时的有效性。为缓解这一问题，我们提出了一种新颖的深度Q学习方法，引入正则化项以减少状态特征表示之间的正相关性。通过利用这些减少的相关特征，我们使线性函数逼近在迁移学习中的有效性得以提升。通过在标准RL基准和MinAtar游戏中进行实验和消融研究，我们展示了该方法在提高迁移学习性能方面的能力，从而减少了计算开销。 

---
# Scalable GANs with Transformers 

**Title (ZH)**: 可扩展的Transformer地带网络 

**Authors**: Sangeek Hyun, MinKyu Lee, Jae-Pil Heo  

**Link**: [PDF](https://arxiv.org/pdf/2509.24935)  

**Abstract**: Scalability has driven recent advances in generative modeling, yet its principles remain underexplored for adversarial learning. We investigate the scalability of Generative Adversarial Networks (GANs) through two design choices that have proven to be effective in other types of generative models: training in a compact Variational Autoencoder latent space and adopting purely transformer-based generators and discriminators. Training in latent space enables efficient computation while preserving perceptual fidelity, and this efficiency pairs naturally with plain transformers, whose performance scales with computational budget. Building on these choices, we analyze failure modes that emerge when naively scaling GANs. Specifically, we find issues as underutilization of early layers in the generator and optimization instability as the network scales. Accordingly, we provide simple and scale-friendly solutions as lightweight intermediate supervision and width-aware learning-rate adjustment. Our experiments show that GAT, a purely transformer-based and latent-space GANs, can be easily trained reliably across a wide range of capacities (S through XL). Moreover, GAT-XL/2 achieves state-of-the-art single-step, class-conditional generation performance (FID of 2.96) on ImageNet-256 in just 40 epochs, 6x fewer epochs than strong baselines. 

**Abstract (ZH)**: 生成模型的可扩展性已推动了近期的进步，然而其原理在对抗学习中的应用仍待深入探索。我们通过两种在其他生成模型中 proven effective 的设计选择来研究生成对抗网络（GANs）的可扩展性：在紧凑的变分自编码器潜空间中训练以及采用纯变压器生成器和判别器。在潜空间中训练使得计算高效且保留感知保真度，这种效率与性能随计算预算线性扩展的纯Transformer自然配对。基于这些选择，我们分析了盲目扩展GANs时出现的故障模式，特别是发现生成器早期层的利用率不足和网络扩展时的优化不稳定问题。相应地，我们提供了解决方案，即轻量级中间监督和宽度感知的学习率调整。我们的实验表明，GAT（纯Transformer和潜空间GANs）可以在广泛的能力范围内（S到XL）可靠地训练。此外，GAT-XL/2在ImageNet-256上的单步条件生成性能（FID为2.96）达到最先进的效果，并且仅需40个epoch，比强大基线少6倍。 

---
# Scaling Laws and Spectra of Shallow Neural Networks in the Feature Learning Regime 

**Title (ZH)**: 浅神经网络在特征学习阶段的标度定律与频谱分布 

**Authors**: Leonardo Defilippis, Yizhou Xu, Julius Girardin, Emanuele Troiani, Vittorio Erba, Lenka Zdeborová, Bruno Loureiro, Florent Krzakala  

**Link**: [PDF](https://arxiv.org/pdf/2509.24882)  

**Abstract**: Neural scaling laws underlie many of the recent advances in deep learning, yet their theoretical understanding remains largely confined to linear models. In this work, we present a systematic analysis of scaling laws for quadratic and diagonal neural networks in the feature learning regime. Leveraging connections with matrix compressed sensing and LASSO, we derive a detailed phase diagram for the scaling exponents of the excess risk as a function of sample complexity and weight decay. This analysis uncovers crossovers between distinct scaling regimes and plateau behaviors, mirroring phenomena widely reported in the empirical neural scaling literature. Furthermore, we establish a precise link between these regimes and the spectral properties of the trained network weights, which we characterize in detail. As a consequence, we provide a theoretical validation of recent empirical observations connecting the emergence of power-law tails in the weight spectrum with network generalization performance, yielding an interpretation from first principles. 

**Abstract (ZH)**: 神经网络中的二次和对角结构在特征学习中的标度律揭示了近期深度学习进展的许多奥秘，然而这些理论理解主要局限于线性模型。本文系统分析了特征学习环境下二次和对角神经网络的标度律。借助矩阵压缩感知和LASSO的联系，我们推导出了过剩风险标度指数与样本复杂度和权重衰减之间的详细相图。这一分析揭示了不同标度律之间的交叉行为和平台行为，反映了在经验神经网络标度文献中广泛报道的现象。此外，我们建立了这些区域与训练网络权重的谱性质之间的精确联系，并详细描述了这些性质。因此，我们提供了一种从第一原理出发的理论验证，最近的经验观察将权重谱中幂律尾巴的出现与网络泛化性能联系起来。 

---
# Vehicle Classification under Extreme Imbalance: A Comparative Study of Ensemble Learning and CNNs 

**Title (ZH)**: 在极端不均衡情况下的车辆分类：集成学习与CNNs的比较研究 

**Authors**: Abu Hanif Muhammad Syarubany  

**Link**: [PDF](https://arxiv.org/pdf/2509.24880)  

**Abstract**: Accurate vehicle type recognition underpins intelligent transportation and logistics, but severe class imbalance in public datasets suppresses performance on rare categories. We curate a 16-class corpus (~47k images) by merging Kaggle, ImageNet, and web-crawled data, and create six balanced variants via SMOTE oversampling and targeted undersampling. Lightweight ensembles, such as Random Forest, AdaBoost, and a soft-voting combiner built on MobileNet-V2 features are benchmarked against a configurable ResNet-style CNN trained with strong augmentation and label smoothing. The best ensemble (SMOTE-combined) attains 74.8% test accuracy, while the CNN achieves 79.19% on the full test set and 81.25% on an unseen inference batch, confirming the advantage of deep models. Nonetheless, the most under-represented class (Barge) remains a failure mode, highlighting the limits of rebalancing alone. Results suggest prioritizing additional minority-class collection and cost-sensitive objectives (e.g., focal loss) and exploring hybrid ensemble or CNN pipelines to combine interpretability with representational power. 

**Abstract (ZH)**: 精确的车辆类型识别是智能交通和物流的基础，但在公共数据集中严重的类别不平衡抑制了对稀有类别的性能。我们通过合并Kaggle、ImageNet和网络抓取数据构建了一个包含16类（约47k张图片）的语料库，并通过SMOTE过采样和目标性下采样创建了六个平衡变体。基准测试了轻量级集成模型，如随机森林、AdaBoost以及基于MobileNet-V2特征的软投票组合器，这些模型与配置可调的具有强增强和标签平滑的ResNet风格CNN进行了对比。最优集成模型（SMOTE组合）在测试集上的准确率为74.8%，而CNN在完整测试集上的准确率为79.19%，在未见过的推理批次上的准确率为81.25%，证实了深度模型的优势。然而，最欠代表的类别（驳船）仍然是一个失败模式，表明仅通过重新平衡无法完全解决问题。结果表明，应优先考虑额外的小类别采集和成本敏感目标（如焦点损失），并探索混合集成或CNN管道，以结合解释性和表现力。 

---
# Uncertainty-Guided Expert-AI Collaboration for Efficient Soil Horizon Annotation 

**Title (ZH)**: 基于不确定性指导的专家-AI协作高效土壤层标注 

**Authors**: Teodor Chiaburu, Vipin Singh, Frank Haußer, Felix Bießmann  

**Link**: [PDF](https://arxiv.org/pdf/2509.24873)  

**Abstract**: Uncertainty quantification is essential in human-machine collaboration, as human agents tend to adjust their decisions based on the confidence of the machine counterpart. Reliably calibrated model uncertainties, hence, enable more effective collaboration, targeted expert intervention and more responsible usage of Machine Learning (ML) systems. Conformal prediction has become a well established model-agnostic framework for uncertainty calibration of ML models, offering statistically valid confidence estimates for both regression and classification tasks. In this work, we apply conformal prediction to $\textit{SoilNet}$, a multimodal multitask model for describing soil profiles. We design a simulated human-in-the-loop (HIL) annotation pipeline, where a limited budget for obtaining ground truth annotations from domain experts is available when model uncertainty is high. Our experiments show that conformalizing SoilNet leads to more efficient annotation in regression tasks and comparable performance scores in classification tasks under the same annotation budget when tested against its non-conformal counterpart. All code and experiments can be found in our repository: this https URL 

**Abstract (ZH)**: 不确定性量化对于人机协作至关重要，因为人类代理往往会根据机器同伴的信心调整其决策。因此，可靠校准的模型不确定性能够促进更有效的协作、针对性的专家干预，并更负责任地使用机器学习系统。一致性预测已成为一种成熟的无模型框架，可用于机器学习模型的不确定性校准，提供统计上有效的置信区间估计，适用于回归和分类任务。在本文中，我们将一致性预测应用于SoilNet，这是一种多模态多任务模型，用于描述土壤剖面。我们设计了一个模拟的人机环注释pipeline，在模型不确定性高时，可用的领域专家 ground truth 注释预算有限。实验结果表明，一致性校准 SoilNet 可以在回归任务中更高效地进行注释，并且在相同注释预算下，其分类任务的表现与非一致性校准版本相当。所有代码和实验可以在我们的仓库中找到：this https URL 

---
# Of-SemWat: High-payload text embedding for semantic watermarking of AI-generated images with arbitrary size 

**Title (ZH)**: Of-SemWat: 高载荷文本嵌入用于任意大小AI生成图像的语义水印技术 

**Authors**: Benedetta Tondi, Andrea Costanzo, Mauro Barni  

**Link**: [PDF](https://arxiv.org/pdf/2509.24823)  

**Abstract**: We propose a high-payload image watermarking method for textual embedding, where a semantic description of the image - which may also correspond to the input text prompt-, is embedded inside the image. In order to be able to robustly embed high payloads in large-scale images - such as those produced by modern AI generators - the proposed approach builds upon a traditional watermarking scheme that exploits orthogonal and turbo codes for improved robustness, and integrates frequency-domain embedding and perceptual masking techniques to enhance watermark imperceptibility. Experiments show that the proposed method is extremely robust against a wide variety of image processing, and the embedded text can be retrieved also after traditional and AI inpainting, permitting to unveil the semantic modification the image has undergone via image-text mismatch analysis. 

**Abstract (ZH)**: 基于语义描述的大载荷图像水印方法：针对文本嵌入的鲁棒性增强 

---
# RDD: Pareto Analysis of the Rate-Distortion-Distinguishability Trade-off 

**Title (ZH)**: RDD：率-失真-可区分性权衡的帕累托分析 

**Authors**: Andriy Enttsel, Alex Marchioni, Andrea Zanellini, Mauro Mangia, Gianluca Setti, Riccardo Rovatti  

**Link**: [PDF](https://arxiv.org/pdf/2509.24805)  

**Abstract**: Extensive monitoring systems generate data that is usually compressed for network transmission. This compressed data might then be processed in the cloud for tasks such as anomaly detection. However, compression can potentially impair the detector's ability to distinguish between regular and irregular patterns due to information loss. Here we extend the information-theoretic framework introduced in [1] to simultaneously address the trade-off between the three features on which the effectiveness of the system depends: the effectiveness of compression, the amount of distortion it introduces, and the distinguishability between compressed normal signals and compressed anomalous signals. We leverage a Gaussian assumption to draw curves showing how moving on a Pareto surface helps administer such a trade-off better than simply relying on optimal rate-distortion compression and hoping that compressed signals can be distinguished from each other. 

**Abstract (ZH)**: 广泛监测系统生成的数据通常被压缩以供网络传输。这些压缩数据随后可能在云端处理以进行异常检测等任务。然而，压缩可能会由于信息丢失而影响检测器区分正常和异常模式的能力。我们扩展了在[1]中引入的信息论框架，同时处理系统有效性的三个特征之间的权衡：压缩的有效性、它引入的失真量以及压缩正常信号和压缩异常信号之间的可区分性。我们利用高斯假设绘制曲线，展示在帕累托曲面上移动如何更好地管理这种权衡，而不仅仅是依赖最优率失真压缩并希望压缩信号能够彼此区分。 

---
# DSAT-HD: Dual-Stream Adaptive Transformer with Hybrid Decomposition for Multivariate Time Series Forecasting 

**Title (ZH)**: DSAT-HD：双重流自适应变换器结合混合分解的多变量时间序列预测 

**Authors**: Zixu Wang, Hongbin Dong, Xiaoping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24800)  

**Abstract**: Time series forecasting is crucial for various applications, such as weather, traffic, electricity, and energy predictions. Currently, common time series forecasting methods are based on Transformers. However, existing approaches primarily model limited time series or fixed scales, making it more challenging to capture diverse features cross different ranges. Additionally, traditional methods like STL for complex seasonality-trend decomposition require pre-specified seasonal periods and typically handle only single, fixed seasonality. We propose the Hybrid Decomposition Dual-Stream Adaptive Transformer (DSAT-HD), which integrates three key innovations to address the limitations of existing methods: 1) A hybrid decomposition mechanism combining EMA and Fourier decomposition with RevIN normalization, dynamically balancing seasonal and trend components through noise Top-k gating; 2) A multi-scale adaptive pathway leveraging a sparse allocator to route features to four parallel Transformer layers, followed by feature merging via a sparse combiner, enhanced by hybrid attention combining local CNNs and global interactions; 3) A dual-stream residual learning framework where CNN and MLP branches separately process seasonal and trend components, coordinated by a balanced loss function minimizing expert collaboration variance. Extensive experiments on nine datasets demonstrate that DSAT-HD outperforms existing methods overall and achieves state-of-the-art performance on some datasets. Notably, it also exhibits stronger generalization capabilities across various transfer scenarios. 

**Abstract (ZH)**: Hybrid Decomposition Dual-Stream Adaptive Transformer (DSAT-HD) for Time Series Forecasting 

---
# Sparse Autoencoders Make Audio Foundation Models more Explainable 

**Title (ZH)**: 稀疏自编码器使音频基础模型更具可解释性 

**Authors**: Théo Mariotte, Martin Lebourdais, Antonio Almudévar, Marie Tahon, Alfonso Ortega, Nicolas Dugué  

**Link**: [PDF](https://arxiv.org/pdf/2509.24793)  

**Abstract**: Audio pretrained models are widely employed to solve various tasks in speech processing, sound event detection, or music information retrieval. However, the representations learned by these models are unclear, and their analysis mainly restricts to linear probing of the hidden representations. In this work, we explore the use of Sparse Autoencoders (SAEs) to analyze the hidden representations of pretrained models, focusing on a case study in singing technique classification. We first demonstrate that SAEs retain both information about the original representations and class labels, enabling their internal structure to provide insights into self-supervised learning systems. Furthermore, we show that SAEs enhance the disentanglement of vocal attributes, establishing them as an effective tool for identifying the underlying factors encoded in the representations. 

**Abstract (ZH)**: 预训练模型在语音处理、声源检测或音乐信息检索等任务中广泛应用，但这些模型学习到的表示尚不明确，其分析主要局限于隐藏表示的线性探查。在本文中，我们探索使用稀疏自编码器（SAEs）来分析预训练模型的隐藏表示，并集中讨论其在歌唱技巧分类中的应用案例。我们首先证明SAEs能够保留原始表示和类别标签的信息，使其实内结构能够为监督学习系统提供洞见。此外，我们展示了SAEs在区分声学属性方面的增强作用，确立了其作为识别表示中潜在因子的有效工具的地位。 

---
# Quantifying Generalisation in Imitation Learning 

**Title (ZH)**: 量化模仿学习中的泛化能力 

**Authors**: Nathan Gavenski, Odinaldo Rodrigues  

**Link**: [PDF](https://arxiv.org/pdf/2509.24784)  

**Abstract**: Imitation learning benchmarks often lack sufficient variation between training and evaluation, limiting meaningful generalisation assessment. We introduce Labyrinth, a benchmarking environment designed to test generalisation with precise control over structure, start and goal positions, and task complexity. It enables verifiably distinct training, evaluation, and test settings. Labyrinth provides a discrete, fully observable state space and known optimal actions, supporting interpretability and fine-grained evaluation. Its flexible setup allows targeted testing of generalisation factors and includes variants like partial observability, key-and-door tasks, and ice-floor hazards. By enabling controlled, reproducible experiments, Labyrinth advances the evaluation of generalisation in imitation learning and provides a valuable tool for developing more robust agents. 

**Abstract (ZH)**: 模仿学习基准往往缺乏训练和评估之间的足够变化，限制了有意义的泛化评估。我们引入了Labyrinth，一种设计用于测试泛化的基准环境，可通过精确控制结构、起始和目标位置以及任务复杂性来实现。它允许验证不同的训练、评估和测试设置。Labyrinth提供了一个离散且完全可观测的状态空间及已知的最优动作，支持可解释性和精细评估。其灵活的设置允许针对泛化因素进行目标测试，并包括诸如部分可观测性、钥匙与门任务以及冰面障碍等变体。通过实现可控且可重复的实验，Labyrinth推进了模仿学习中泛化的评估，并提供了一种开发更高鲁棒性代理的重要工具。 

---
# Surjective Independence of Causal Influences for Local Bayesian Network Structures 

**Title (ZH)**: 局部贝叶斯网络结构上的因果影响的满射独立性 

**Authors**: Kieran Drury, Martine J. Barons, Jim Q. Smith  

**Link**: [PDF](https://arxiv.org/pdf/2509.24759)  

**Abstract**: The very expressiveness of Bayesian networks can introduce fresh challenges due to the large number of relationships they often model. In many domains, it is thus often essential to supplement any available data with elicited expert judgements. This in turn leads to two key challenges: the cognitive burden of these judgements is often very high, and there are a very large number of judgements required to obtain a full probability model. We can mitigate both issues by introducing assumptions such as independence of causal influences (ICI) on the local structures throughout the network, restricting the parameter space of the model. However, the assumption of ICI is often unjustified and overly strong. In this paper, we introduce the surjective independence of causal influences (SICI) model which relaxes the ICI assumption and provides a more viable, practical alternative local structure model that facilitates efficient Bayesian network parameterisation. 

**Abstract (ZH)**: 贝叶斯网络的很强的表达能力因它们通常建模的关系数量庞大而引入了新的挑战。因此，在许多领域中，常需要补充可用数据以获取专家判断。这进而导致两个关键挑战：这些判断的认知负担通常很高，且需要大量的判断以获得完整的概率模型。通过引入局部结构中的因果影响的满射独立性（SICI）假设，限制模型的参数空间，我们可以缓解这些问题。然而，因果影响独立性（ICI）假设往往不合理且过于强硬。本文引入了因果影响的满射独立性（SICI）模型，该模型放松了ICI假设，提供了一种更可行且实用的局部结构模型，有助于高效地进行贝叶斯网络参数化。 

---
# Robust Policy Expansion for Offline-to-Online RL under Diverse Data Corruption 

**Title (ZH)**: 离线到在线RL在多样化数据污染下的鲁棒策略扩展 

**Authors**: Longxiang He, Deheng Ye, Junbo Tan, Xueqian Wang, Li Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24748)  

**Abstract**: Pretraining a policy on offline data followed by fine-tuning through online interactions, known as Offline-to-Online Reinforcement Learning (O2O RL), has emerged as a promising paradigm for real-world RL deployment. However, both offline datasets and online interactions in practical environments are often noisy or even maliciously corrupted, severely degrading the performance of O2O RL. Existing works primarily focus on mitigating the conservatism of offline policies via online exploration, while the robustness of O2O RL under data corruption, including states, actions, rewards, and dynamics, is still unexplored. In this work, we observe that data corruption induces heavy-tailed behavior in the policy, thereby substantially degrading the efficiency of online exploration. To address this issue, we incorporate Inverse Probability Weighted (IPW) into the online exploration policy to alleviate heavy-tailedness, and propose a novel, simple yet effective method termed $\textbf{RPEX}$: $\textbf{R}$obust $\textbf{P}$olicy $\textbf{EX}$pansion. Extensive experimental results on D4RL datasets demonstrate that RPEX achieves SOTA O2O performance across a wide range of data corruption scenarios. Code is available at $\href{this https URL}{this https URL}$. 

**Abstract (ZH)**: 离线数据预训练结合在线微调的离线到在线强化学习（O2O RL）在实际部署中展现出潜力，但由于实际环境中的离线数据集和在线交互往往存在噪音甚至恶意篡改，严重降低了O2O RL的性能。现有工作主要关注通过在线探索减轻离线策略的保守性，而数据篡改对状态、动作、奖励和动力学的影响下的O2O RL鲁棒性尚未被研究。在本文中，我们发现数据篡改导致策略行为服从厚尾分布，显著降低了在线探索的效率。为了解决这一问题，我们将在线探索策略中引入逆概率加权（IPW）以缓解厚尾性，并提出了一种新颖且有效的方法RPEX：鲁棒策略扩展。在D4RL数据集上的广泛实验结果表明，RPEX在多种数据篡改场景中达到了SOTA的O2O性能。代码可在$\href{this https URL}{this https URL}$获取。 

---
# Q-Net: Transferable Queue Length Estimation via Kalman-based Neural Networks 

**Title (ZH)**: Q-网：基于卡尔曼滤波的神经网络排队长度估计 

**Authors**: Ting Gao, Elvin Isufi, Winnie Daamen, Erik-Sander Smits, Serge Hoogendoorn  

**Link**: [PDF](https://arxiv.org/pdf/2509.24725)  

**Abstract**: Estimating queue lengths at signalized intersections remains a challenge in traffic management, especially under partially observed conditions where vehicle flows are not fully captured. This paper introduces Q-Net, a data-efficient and interpretable framework for queue length estimation that performs robustly even when traffic conservation assumptions are violated. Q-Net integrates two widely available and privacy-friendly data sources: (i) vehicle counts from loop detectors near stop lines, and (ii) aggregated floating car data (aFCD), which divides each road section into segments and provides segment-wise average speed measurements. These data sources often differ in spatial and temporal resolution, creating fusion challenges. Q-Net addresses this by employing a tailored state-space model and an AI-augmented Kalman filter, KalmanNet, which learns the Kalman gain from data without requiring prior knowledge of noise covariances or full system dynamics. We build on the vanilla KalmanNet pipeline to decouple measurement dimensionality from section length, enabling spatial transferability across road segments. Unlike black-box models, Q-Net maintains physical interpretability, with internal variables linked to real-world traffic dynamics. Evaluations on main roads in Rotterdam, the Netherlands, demonstrate that Q-Net outperforms baseline methods by over 60\% in Root Mean Square Error (RMSE), accurately tracking queue formation and dissipation while correcting aFCD-induced delays. Q-Net also demonstrates strong spatial and temporal transferability, enabling deployment without costly sensing infrastructure like cameras or radar. Additionally, we propose a real-time variant of Q-Net, highlighting its potential for integration into dynamic, queue-based traffic control systems. 

**Abstract (ZH)**: 基于信号交叉口队列长度估计的Q-Net框架：一种在部分观测条件下数据高效且可解释的方法 

---
# Circuit-Aware Reward Training: A Mechanistic Framework for Longtail Robustness in RLHF 

**Title (ZH)**: 电路意识奖励训练：RLHF长尾稳健性的本征框架 

**Authors**: Jing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24713)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) reward models exhibit systematic failures on longtail distributions, leading to reward hacking and misalignment. We propose a mechanistic interpretability framework that identifies specialized neural circuits responsible for rare-event processing in reward models. Drawing from recent advances showing distributed specialization for rare tokens in language models\citep{liu2025no, liu2025emergent}, we hypothesize that reward models also develop functionally distinct circuits for longtail scenarios. Our theoretical framework establishes formal connections between circuit specialization, reward generalization bounds, and longtail performance. We introduce \textbf{Circuit-Aware Reward Training (CART)}, which uses circuit analysis to guide data augmentation, regularization, and ensemble strategies. This approach provides both theoretical insights into reward model failures and practical interventions for improving longtail robustness. 

**Abstract (ZH)**: 基于人类反馈的强化学习（RLHF）奖励模型在长尾分布上表现出系统性的失败，导致奖励作弊和不一致。我们提出了一种机制可解释性框架，该框架识别出负责处理稀有事件的特殊神经电路。借鉴近期研究表明语言模型中对稀有词存在分布式专业化（distributed specialization for rare tokens）的现象（\citet{liu2025no, liu2025emergent}），我们假设奖励模型也发展出了功能上不同的电路来处理长尾场景。我们的理论框架建立了电路专业化、奖励泛化边界和长尾性能之间的正式联系。我们引入了**电路感知奖励训练（CART）**方法，该方法使用电路分析来指导数据增强、正则化和集成策略。该方法为理解和改进奖励模型的长尾稳健性提供了理论洞见和实际干预措施。 

---
# CoTune: Co-evolutionary Configuration Tuning 

**Title (ZH)**: 共进化的配置调优 

**Authors**: Gangda Xiong, Tao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24694)  

**Abstract**: To automatically tune configurations for the best possible system performance (e.g., runtime or throughput), much work has been focused on designing intelligent heuristics in a tuner. However, existing tuner designs have mostly ignored the presence of complex performance requirements (e.g., the latency shall ideally be 2 seconds), but simply assume that better performance is always more preferred. This would not only waste valuable information in a requirement but might also consume extensive resources to tune for a goal with little gain. Yet, prior studies have shown that simply incorporating the requirement as a tuning objective is problematic since the requirement might be too strict, harming convergence; or its highly diverse satisfactions might lead to premature convergence. In this paper, we propose CoTune, a tool that takes the information of a given target performance requirement into account through co-evolution. CoTune is unique in the sense that it creates an auxiliary performance requirement to be co-evolved with the configurations, which assists the target performance requirement when it becomes ineffective or even misleading, hence allowing the tuning to be guided by the requirement while being robust to its harm. Experiment results on 162 cases (nine systems and 18 requirements) reveal that CoTune considerably outperforms existing tuners, ranking as the best for 90% cases (against the 0%--35% for other tuners) with up to 2.9x overall improvements, while doing so under a much better efficiency. 

**Abstract (ZH)**: 一种通过共进化考虑目标性能需求的自动调优工具：CoTune 

---
# Data-Driven Discrete Geofence Design Using Binary Quadratic Programming 

**Title (ZH)**: 基于数据驱动的离散地理围栏设计——二元二次规划方法 

**Authors**: Keisuke Otaki, Akihisa Okada, Tadayoshi Matsumori, Hiroaki Yoshida  

**Link**: [PDF](https://arxiv.org/pdf/2509.24679)  

**Abstract**: Geofences have attracted significant attention in the design of spatial and virtual regions for managing and engaging spatiotemporal events. By using geofences to monitor human activity across their boundaries, content providers can create spatially triggered events that include notifications about points of interest within a geofence by pushing spatial information to the devices of users. Traditionally, geofences were hand-crafted by providers. In addition to the hand-crafted approach, recent advances in collecting human mobility data through mobile devices can accelerate the automatic and data-driven design of geofences, also known as the geofence design problem. Previous approaches assume circular shapes; thus, their flexibility is insufficient, and they can only handle geofence-based applications for large areas with coarse resolutions. A challenge with using circular geofences in urban and high-resolution areas is that they often overlap and fail to align with political district boundaries and road segments, such as one-way streets and median barriers. In this study, we address the problem of extracting arbitrary shapes as geofences from human mobility data to mitigate this problem. In our formulation, we cast the existing optimization problems for circular geofences to 0-1 integer programming problems to represent arbitrary shapes. Although 0-1 integer programming problems are computationally hard, formulating them as quadratic (unconstrained) binary optimization problems enables efficient approximation of optimal solutions, because this allows the use of specialized quadratic solvers, such as the quantum annealing, and other state-of-the-art algorithms. We then develop and compare different formulation methods to extract discrete geofences. We confirmed that our new modeling approach enables flexible geofence design. 

**Abstract (ZH)**: 地理围栏在空间和虚拟区域设计中的关注点：基于人类移动数据的任意形状地理围栏提取 

---
# Community detection robustness of graph neural networks 

**Title (ZH)**: 图神经网络的社区检测鲁棒性 

**Authors**: Jaidev Goel, Pablo Moriano, Ramakrishnan Kannan, Yulia R. Gel  

**Link**: [PDF](https://arxiv.org/pdf/2509.24662)  

**Abstract**: Graph neural networks (GNNs) are increasingly widely used for community detection in attributed networks. They combine structural topology with node attributes through message passing and pooling. However, their robustness or lack of thereof with respect to different perturbations and targeted attacks in conjunction with community detection tasks is not well understood. To shed light into latent mechanisms behind GNN sensitivity on community detection tasks, we conduct a systematic computational evaluation of six widely adopted GNN architectures: GCN, GAT, Graph- SAGE, DiffPool, MinCUT, and DMoN. The analysis covers three perturbation categories: node attribute manipulations, edge topology distortions, and adversarial attacks. We use element-centric similarity as the evaluation metric on synthetic benchmarks and real-world citation networks. Our findings indicate that supervised GNNs tend to achieve higher baseline accuracy, while unsupervised methods, particularly DMoN, maintain stronger resilience under targeted and adversarial pertur- bations. Furthermore, robustness appears to be strongly influenced by community strength, with well-defined communities reducing performance loss. Across all models, node attribute perturba- tions associated with targeted edge deletions and shift in attribute distributions tend to cause the largest degradation in community recovery. These findings highlight important trade-offs between accuracy and robustness in GNN-based community detection and offer new insights into selecting architectures resilient to noise and adversarial attacks. 

**Abstract (ZH)**: 图神经网络在属性网络社区检测中的鲁棒性研究：基于六种广泛采用的GNN架构的系统计算评估 

---
# Algorithms and data structures for automatic precision estimation of neural networks 

**Title (ZH)**: 神经网络自动精度估计的算法与数据结构 

**Authors**: Igor V. Netay  

**Link**: [PDF](https://arxiv.org/pdf/2509.24607)  

**Abstract**: We describe algorithms and data structures to extend a neural network library with automatic precision estimation for floating point computations. We also discuss conditions to make estimations exact and preserve high computation performance of neural networks training and inference. Numerical experiments show the consequences of significant precision loss for particular values such as inference, gradients and deviations from mathematically predicted behavior.
It turns out that almost any neural network accumulates computational inaccuracies. As a result, its behavior does not coincide with predicted by the mathematical model of neural network. This shows that tracking of computational inaccuracies is important for reliability of inference, training and interpretability of results. 

**Abstract (ZH)**: 我们描述了算法和数据结构，以扩展神经网络库，并实现浮点计算的自动精度估计。我们还讨论了使估计精确并保持神经网络训练和推理高性能的条件。数值实验表明，对于特定值如推理、梯度和数学预测行为偏差，精度损失会对结果产生显著影响。事实上，几乎任何神经网络都会累积计算不准确，导致其行为与神经网络数学模型的预测不符，这表明跟踪计算不准确对于推理、训练可靠性和结果可解释性的重要性。 

---
# Bandits roaming Hilbert space 

**Title (ZH)**: 游走于希尔伯特空间的Bandits 

**Authors**: Josep Lumbreras  

**Link**: [PDF](https://arxiv.org/pdf/2509.24569)  

**Abstract**: This thesis studies the exploration and exploitation trade-off in online learning of properties of quantum states using multi-armed bandits. Given streaming access to an unknown quantum state, in each round we select an observable from a set of actions to maximize its expectation value. Using past information, we refine actions to minimize regret; the cumulative gap between current reward and the maximum possible. We derive information-theoretic lower bounds and optimal strategies with matching upper bounds, showing regret typically scales as the square root of rounds. As an application, we reframe quantum state tomography to both learn the state efficiently and minimize measurement disturbance. For pure states and continuous actions, we achieve polylogarithmic regret using a sample-optimal algorithm based on a weighted online least squares estimator. The algorithm relies on the optimistic principle and controls the eigenvalues of the design matrix. We also apply our framework to quantum recommender systems and thermodynamic work extraction from unknown states. In this last setting, our results demonstrate an exponential advantage in work dissipation over tomography-based protocols. 

**Abstract (ZH)**: 本论文研究了在多臂 bandit 框架下学习量子状态性质时在线学习中的探索与利用权衡问题。通过逐轮访问未知的量子态，我们从一系列可观测量中选择一个以最大化其期望值。利用过往信息，我们不断细化行动以最小化遗憾；遗憾即当前奖励与最大可能奖励之间的累积差距。我们推导了信息论下的下界，并给出了匹配的上界最优策略，表明遗憾通常随轮次平方根增长。作为一种应用，我们将量子态 tomography 问题重新定框，以高效学习量子态并最小化测量扰动。对于纯态和连续行动，我们利用基于加权在线最小二乘估计器的样本最优算法实现了多项式对数遗憾。该算法依靠乐观原则，并控制设计矩阵的特征值。此外，我们将该框架应用于量子推荐系统，并从未知态中提取热力学工作。在后者场景中，我们的结果展示了与基于 tomography 的协议相比，在工作耗散方面具有指数级优势。 

---
# Short window attention enables long-term memorization 

**Title (ZH)**: 短窗注意力实现长期记忆 

**Authors**: Loïc Cabannes, Maximilian Beck, Gergely Szilvasy, Matthijs Douze, Maria Lomeli, Jade Copet, Pierre-Emmanuel Mazaré, Gabriel Synnaeve, Hervé Jégou  

**Link**: [PDF](https://arxiv.org/pdf/2509.24552)  

**Abstract**: Recent works show that hybrid architectures combining sliding window softmax attention layers with linear recurrent neural network (RNN) layers outperform both of these architectures taken separately. However, the impact of the window length and the interplay between softmax attention and linear RNN layers remain under-studied. In this work, we introduce SWAX, a hybrid architecture consisting of sliding-window attention and xLSTM linear RNN layers.
A counter-intuitive finding with SWAX is that larger sliding windows do not improve the long-context performance. In fact, short window attention encourages the model to better train the long-term memory of the xLSTM, by relying less on the softmax attention mechanism for long context-retrieval.
The issue with small sliding windows is that they are detrimental for short-context tasks, which could be solved with information from moderately larger sliding windows otherwise. Therefore, we train SWAX by stochastically changing the sliding window size, forcing the model to leverage both a longer context window and the xLSTM memory. SWAX trained with stochastic window sizes significantly outperforms regular window attention both on short and long-context problems. 

**Abstract (ZH)**: 最近的研究表明，结合滑动窗口softmax注意力层和线性递归神经网络（RNN）层的混合架构优于单独使用这两种架构。然而，滑动窗口长度的影响以及softmax注意力与线性RNN层之间的互动尚未得到充分研究。在本文中，我们引入了SWAX，这是一种由滑动窗口注意力和xLSTM线性RNN层组成的混合架构。

SWAX的一个出乎意料的发现是，较大的滑动窗口并不提高长上下文性能。实际上，较短的滑动窗口会促使模型更有效地训练xLSTM的长期记忆，因为它较少依赖于softmax注意力机制来进行长上下文检索。

较小滑动窗口的问题在于它们对短上下文任务有害，这可以通过较大但适度的滑动窗口信息来解决。因此，我们通过随机改变滑动窗口大小来训练SWAX，迫使模型利用更长的上下文窗口和xLSTM的记忆。随机滑动窗口大小训练的SWAX在短上下文和长上下文问题上都显著优于固定窗口注意力。 

---
# CMT: Mid-Training for Efficient Learning of Consistency, Mean Flow, and Flow Map Models 

**Title (ZH)**: CMT：中间训练以高效学习一致性和流平均模型 

**Authors**: Zheyuan Hu, Chieh-Hsin Lai, Yuki Mitsufuji, Stefano Ermon  

**Link**: [PDF](https://arxiv.org/pdf/2509.24526)  

**Abstract**: Flow map models such as Consistency Models (CM) and Mean Flow (MF) enable few-step generation by learning the long jump of the ODE solution of diffusion models, yet training remains unstable, sensitive to hyperparameters, and costly. Initializing from a pre-trained diffusion model helps, but still requires converting infinitesimal steps into a long-jump map, leaving instability unresolved. We introduce mid-training, the first concept and practical method that inserts a lightweight intermediate stage between the (diffusion) pre-training and the final flow map training (i.e., post-training) for vision generation. Concretely, Consistency Mid-Training (CMT) is a compact and principled stage that trains a model to map points along a solver trajectory from a pre-trained model, starting from a prior sample, directly to the solver-generated clean sample. It yields a trajectory-consistent and stable initialization. This initializer outperforms random and diffusion-based baselines and enables fast, robust convergence without heuristics. Initializing post-training with CMT weights further simplifies flow map learning. Empirically, CMT achieves state of the art two step FIDs: 1.97 on CIFAR-10, 1.32 on ImageNet 64x64, and 1.84 on ImageNet 512x512, while using up to 98% less training data and GPU time, compared to CMs. On ImageNet 256x256, CMT reaches 1-step FID 3.34 while cutting total training time by about 50% compared to MF from scratch (FID 3.43). This establishes CMT as a principled, efficient, and general framework for training flow map models. 

**Abstract (ZH)**: Mid-Training for Stable and Efficient Flow Map Learning 

---
# Moravec's Paradox and Restrepo's Model: Limits of AGI Automation in Growth 

**Title (ZH)**: 莫拉维克悖论与雷斯特雷波模型：AGI自动化在增长领域的局限性 

**Authors**: Marc Bara  

**Link**: [PDF](https://arxiv.org/pdf/2509.24466)  

**Abstract**: This note extends Restrepo (2025)'s model of economic growth under AGI by incorporating Moravec's Paradox -the observation that tasks requiring sensorimotor skills remain computationally expensive relative to cognitive tasks. We partition the task space into cognitive and physical components with differential automation costs, allowing infinite costs for some physical bottlenecks. Our key result shows that when physical tasks constitute economic bottlenecks with sufficiently high (or infinite) computational requirements, the labor share of income converges to a positive constant in the finite-compute regime (rather than zero). This fundamentally alters the distributional implications of AGI while preserving the growth dynamics for cognitive-intensive economies. 

**Abstract (ZH)**: 这首笔记将Restrepo (2025)关于AGI的经济增长模型扩展至纳入了莫拉克悖论，即感觉运动技能需求的任务相比于认知任务仍具有相对较高的计算成本。我们将任务空间划分为认知和物理组成部分，并允许物理瓶颈具有无限的自动化成本。我们的主要结果表明，在物理任务构成具有足够高（或无限）计算要求的经济瓶颈时，在有限计算能力范围内，收入中的劳动份额将趋于一个正的常数（而不是零）。这一发现从根本上改变了AGI的分配影响，同时保留了对认知密集型经济的经济增长动态。 

---
# An Agent-Based Framework for Automated Higher-Voice Harmony Generation 

**Title (ZH)**: 基于代理的自动化高音和谐生成框架 

**Authors**: Nia D'Souza Ganapathy, Arul Selvamani Shaja  

**Link**: [PDF](https://arxiv.org/pdf/2509.24463)  

**Abstract**: The generation of musically coherent and aesthetically pleasing harmony remains a significant challenge in the field of algorithmic composition. This paper introduces an innovative Agentic AI-enabled Higher Harmony Music Generator, a multi-agent system designed to create harmony in a collaborative and modular fashion. Our framework comprises four specialized agents: a Music-Ingestion Agent for parsing and standardizing input musical scores; a Chord-Knowledge Agent, powered by a Chord-Former (Transformer model), to interpret and provide the constituent notes of complex chord symbols; a Harmony-Generation Agent, which utilizes a Harmony-GPT and a Rhythm-Net (RNN) to compose a melodically and rhythmically complementary harmony line; and an Audio-Production Agent that employs a GAN-based Symbolic-to-Audio Synthesizer to render the final symbolic output into high-fidelity audio. By delegating specific tasks to specialized agents, our system effectively mimics the collaborative process of human musicians. This modular, agent-based approach allows for robust data processing, deep theoretical understanding, creative composition, and realistic audio synthesis, culminating in a system capable of generating sophisticated and contextually appropriate higher-voice harmonies for given melodies. 

**Abstract (ZH)**: 算法作曲中具有音乐连贯性和审美吸引力和声生成依然是一项重要挑战。本文介绍了一种创新的Agentic AI驱动的高声部和声生成器，这是一种多代理系统，旨在以协作和模块化的方式生成和声。该框架包含四个专门的代理：一个音乐摄入代理，用于解析和标准化输入的音乐曲谱；一个由Chord-Former（变换器模型）驱动的和弦知识代理，以解释和提供复杂的和弦符号的构成音；一个和声生成代理，利用Harmony-GPT和Rhythm-Net（RNN）来创作旋律和节奏相补的和声线；以及一个采用基于GAN的符号到音频合成器进行音频渲染的音频生产代理。通过将具体任务委派给专门的代理，我们的系统有效地模拟了人类音乐家的协作过程。这种模块化的代理方法允许稳健的数据处理、深入的理论理解、创意思维的组成以及现实主义的音频合成，最终生成给定旋律的复杂且上下文相关的高声部和声。 

---
# Multi-Item-Query Attention for Stable Sequential Recommendation 

**Title (ZH)**: 多项查询注意力机制下的稳定序列推荐 

**Authors**: Mingshi Xu, Haoren Zhu, Wilfred Siu Hung Ng  

**Link**: [PDF](https://arxiv.org/pdf/2509.24424)  

**Abstract**: The inherent instability and noise in user interaction data challenge sequential recommendation systems. Prevailing masked attention models, relying on a single query from the most recent item, are sensitive to this noise, reducing prediction reliability. We propose the Multi-Item-Query attention mechanism (MIQ-Attn) to enhance model stability and accuracy. MIQ-Attn constructs multiple diverse query vectors from user interactions, effectively mitigating noise and improving consistency. It is designed for easy adoption as a drop-in replacement for existing single-query attention. Experiments show MIQ-Attn significantly improves performance on benchmark datasets. 

**Abstract (ZH)**: 用户交互数据中的固有不稳定性和噪声挑战了序列推荐系统。依赖于最近一项的单一查询的盛行掩码注意力模型对此噪声敏感，降低了预测可靠性。我们提出多项查询注意力机制（MIQ-Attn）以增强模型稳定性和准确性。MIQ-Attn从用户交互中构建多个多样化的查询向量，有效减轻噪声并提高一致性。该机制设计为易于替换现有单一查询注意力的即插即用方案。实验结果显示，MIQ-Attn在基准数据集上显著提升了性能。 

---
# CLQ: Cross-Layer Guided Orthogonal-based Quantization for Diffusion Transformers 

**Title (ZH)**: CLQ: 不同层引导正交基量化技术用于扩散变换器 

**Authors**: Kai Liu, Shaoqiu Zhang, Linghe Kong, Yulun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24416)  

**Abstract**: Visual generation quality has been greatly promoted with the rapid advances in diffusion transformers (DiTs), which is attributed to the scaling of model size and complexity. However, these attributions also hinder the practical deployment of DiTs on edge devices, limiting their development and application. Serve as an efficient model compression technique, model post-training quantization (PTQ) can reduce the memory consumption and speed up the inference, with inevitable performance degradation. To alleviate the degradation, we propose CLQ, a cross-layer guided orthogonal-based quantization method for DiTs. To be specific, CLQ consists of three key designs. First, we observe that the calibration data used by most of the PTQ methods can not honestly represent the distribution of the activations. Therefore, we propose cross-block calibration (CBC) to obtain accurate calibration data, with which the quantization can be better guided. Second, we propose orthogonal-based smoothing (OBS), which quantifies the outlier score of each channel and leverages block Hadamard matrix to smooth the outliers with negligible overhead. Third, we propose cross-layer parameter searching (CLPS) to search. We evaluate CLQ with both image generation and video generation models and successfully compress the model into W4A4 with negligible degradation in visual quality and metrics. CLQ achieves 3.98x memory saving and 3.95x speedup. Our code is available at \hyperlink{this https URL}{this https URL}. 

**Abstract (ZH)**: 视觉生成质量随着扩散变压器（DiTs）的快速进步得到了极大的提升，这归因于模型规模和复杂性的扩大。然而，这些归因也阻碍了DiTs在边缘设备上的实际部署，限制了其发展和应用。作为一种有效的模型压缩技术，后训练量化（PTQ）可以减少内存消耗并加速推理，但不可避免地会导致性能下降。为了解决这种下降，我们提出了一种跨层引导正交基量化方法CLQ（Cross-layer Guided Orthogonal-based Quantization）用于DiTs。具体而言，CLQ 包含三个关键设计。首先，我们发现大多数PTQ方法使用的校准数据不能真实地代表激活值的分布。因此，我们提出了跨块校准（CBC，Cross-block Calibration）以获得准确的校准数据，从而使量化得到更好的引导。其次，我们提出了基于正交的平滑（OBS，Orthogonal-based Smoothing），量化每个通道的异常值得分，并利用块Hadamard矩阵平滑异常值，而几乎不增加开销。最后，我们提出了跨层参数搜索（CLPS，Cross-layer Parameter Searching）。我们使用图像生成和视频生成模型评估了CLQ，并成功将模型压缩到W4A4，同时视觉质量和指标的下降可以忽略不计。CLQ 实现了3.98倍的内存节省和3.95倍的速度提升。我们的代码可在 \hyperlink{this https URL}{this https URL} 获取。 

---
# ScatterAD: Temporal-Topological Scattering Mechanism for Time Series Anomaly Detection 

**Title (ZH)**: ScatterAD：时间拓扑散射机制在时间序列异常检测中的应用 

**Authors**: Tao Yin, Xiaohong Zhang, Shaochen Fu, Zhibin Zhang, Li Huang, Yiyuan Yang, Kaixiang Yang, Meng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2509.24414)  

**Abstract**: One main challenge in time series anomaly detection for industrial IoT lies in the complex spatio-temporal couplings within multivariate data. However, traditional anomaly detection methods focus on modeling spatial or temporal dependencies independently, resulting in suboptimal representation learning and limited sensitivity to anomalous dispersion in high-dimensional spaces. In this work, we conduct an empirical analysis showing that both normal and anomalous samples tend to scatter in high-dimensional space, especially anomalous samples are markedly more dispersed. We formalize this dispersion phenomenon as scattering, quantified by the mean pairwise distance among sample representations, and leverage it as an inductive signal to enhance spatio-temporal anomaly detection. Technically, we propose ScatterAD to model representation scattering across temporal and topological dimensions. ScatterAD incorporates a topological encoder for capturing graph-structured scattering and a temporal encoder for constraining over-scattering through mean squared error minimization between neighboring time steps. We introduce a contrastive fusion mechanism to ensure the complementarity of the learned temporal and topological representations. Additionally, we theoretically show that maximizing the conditional mutual information between temporal and topological views improves cross-view consistency and enhances more discriminative representations. Extensive experiments on multiple public benchmarks show that ScatterAD achieves state-of-the-art performance on multivariate time series anomaly detection. Code is available at this repository: this https URL. 

**Abstract (ZH)**: 工业物联网中时间序列异常检测的主要挑战在于多变量数据中的复杂空时耦合。传统异常检测方法独立建模空域或时域依赖性，导致不足的表征学习，并在高维空间中对异常分散的敏感性有限。在本文中，我们通过实证分析表明，正常样本和异常样本都倾向于在高维空间中分散，尤其是异常样本的分散程度更为显著。我们将这种分散现象形式化为散布，通过样本表示的平均成对距离来量化，并利用其作为归纳信号以增强空时异常检测。技术上，我们提出了ScatterAD来建模跨时间和拓扑维度的表征散布。ScatterAD结合了拓扑编解码器来捕捉基于图的散布，并通过邻近时间步之间的均方误差最小化来限制过度散布。我们引入了一种对比融合机制以确保学习到的时间和拓扑表征的互补性。此外，我们从理论上证明，最大化时间视图和拓扑视图的条件互信息可以提高跨视图一致性并生成更具判别力的表示。在多个公开基准上的实验显示，ScatterAD在多变量时间序列异常检测中达到了最先进的性能。代码可在以下仓库获取：this https URL。 

---
# Hybrid Layer-Wise ANN-SNN With Surrogate Spike Encoding-Decoding Structure 

**Title (ZH)**: 混合层-wise ANN-SNN配 Browse_surrogate 突变编码-解码结构 

**Authors**: Nhan T. Luu, Duong T. Luu, Pham Ngoc Nam, Truong Cong Thang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24411)  

**Abstract**: Spiking Neural Networks (SNNs) have gained significant traction in both computational neuroscience and artificial intelligence for their potential in energy-efficient computing. In contrast, artificial neural networks (ANNs) excel at gradient-based optimization and high accuracy. This contrast has consequently led to a growing subfield of hybrid ANN-SNN research. However, existing hybrid approaches often rely on either a strict separation between ANN and SNN components or employ SNN-only encoders followed by ANN classifiers due to the constraints of non-differentiability of spike encoding functions, causing prior hybrid architectures to lack deep layer-wise cooperation during backpropagation. To address this gap, we propose a novel hybrid ANN-SNN framework that integrates layer-wise encode-decode SNN blocks within conventional ANN pipelines. Central to our method is the use of surrogate gradients for a bit-plane-based spike encoding function, enabling end-to-end differentiable training across ANN and SNN layers. This design achieves competitive accuracy with state-of-the-art pure ANN and SNN models while retaining the potential efficiency and temporal representation benefits of spiking computation. To the best of our knowledge, this is the first implementation of a surrogate gradient for bit plane coding specifically and spike encoder interface in general to be utilized in the context of hybrid ANN-SNN, successfully leading to a new class of hybrid models that pave new directions for future research. 

**Abstract (ZH)**: 基于突触神经网络的新型混合ANN-SNN框架：基于位平面的替代梯度突触编码 

---
# The 2025 OpenAI Preparedness Framework does not guarantee any AI risk mitigation practices: a proof-of-concept for affordance analyses of AI safety policies 

**Title (ZH)**: 2025年OpenAI准备框架并不保证任何AI风险缓解实践：AI安全政策能力分析的可行性研究 

**Authors**: Sam Coggins, Alex Saeri, Katherine A. Daniell, Lorenn P. Ruster, Jessie Liu, Jenny L. Davis  

**Link**: [PDF](https://arxiv.org/pdf/2509.24394)  

**Abstract**: Prominent AI companies are producing 'safety frameworks' as a type of voluntary self-governance. These statements purport to establish risk thresholds and safety procedures for the development and deployment of highly capable AI. Understanding which AI risks are covered and what actions are allowed, refused, demanded, encouraged, or discouraged by these statements is vital for assessing how these frameworks actually govern AI development and deployment. We draw on affordance theory to analyse the OpenAI 'Preparedness Framework Version 2' (April 2025) using the Mechanisms & Conditions model of affordances and the MIT AI Risk Repository. We find that this safety policy requests evaluation of a small minority of AI risks, encourages deployment of systems with 'Medium' capabilities for what OpenAI itself defines as 'severe harm' (potential for >1000 deaths or >$100B in damages), and allows OpenAI's CEO to deploy even more dangerous capabilities. These findings suggest that effective mitigation of AI risks requires more robust governance interventions beyond current industry self-regulation. Our affordance analysis provides a replicable method for evaluating what safety frameworks actually permit versus what they claim. 

**Abstract (ZH)**: prominente的人工智能公司正在制定“安全框架”作为一种自愿自我治理方式。这些声明旨在为高度具备能力的人工智能的研发和部署设定风险阈值和安全程序。了解这些声明涵盖哪些人工智能风险以及允许、拒绝、要求、鼓励或反对哪些行动对于评估这些框架实际上如何治理人工智能的研发和部署至关重要。我们借鉴了机会理论，使用机会机制与条件模型和MIT人工智能风险仓库来分析OpenAI“准备框架版本2”（2025年4月）。研究发现，这一安全政策要求对少数几项人工智能风险进行评估，鼓励部署“中等”能力的系统，这些系统被OpenAI自身定义为“严重伤害”（潜在致死人数超过1000人或造成超过1000亿美元的损害），并且允许OpenAI首席执行官部署更为危险的能力。这些发现表明，有效的降低人工智能风险需要超出当前行业自我监管的更稳健的治理干预措施。我们的机会分析提供了一种可复制的方法，用于评估实际安全框架允许的内容与它们所声称的内容之间的差异。 

---
# Towards Generalizable PDE Dynamics Forecasting via Physics-Guided Invariant Learning 

**Title (ZH)**: 基于物理引导不变性学习的泛化偏微分方程动力学预测 

**Authors**: Siyang Li, Yize Chen, Yan Guo, Ming Huang, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2509.24332)  

**Abstract**: Advanced deep learning-based approaches have been actively applied to forecast the spatiotemporal physical dynamics governed by partial differential equations (PDEs), which acts as a critical procedure in tackling many science and engineering problems. As real-world physical environments like PDE system parameters are always capricious, how to generalize across unseen out-of-distribution (OOD) forecasting scenarios using limited training data is of great importance. To bridge this barrier, existing methods focus on discovering domain-generalizable representations across various PDE dynamics trajectories. However, their zero-shot OOD generalization capability remains deficient, since extra test-time samples for domain-specific adaptation are still required. This is because the fundamental physical invariance in PDE dynamical systems are yet to be investigated or integrated. To this end, we first explicitly define a two-fold PDE invariance principle, which points out that ingredient operators and their composition relationships remain invariant across different domains and PDE system evolution. Next, to capture this two-fold PDE invariance, we propose a physics-guided invariant learning method termed iMOOE, featuring an Invariance-aligned Mixture Of Operator Expert architecture and a frequency-enriched invariant learning objective. Extensive experiments across simulated benchmarks and real-world applications validate iMOOE's superior in-distribution performance and zero-shot generalization capabilities on diverse OOD forecasting scenarios. 

**Abstract (ZH)**: 基于深度学习的先进方法已被积极应用于预报由偏微分方程（PDE）支配的时空物理动力学，这是解决许多科学与工程问题的关键步骤。由于现实世界中的物理环境如PDE系统参数总是不可预测的，如何在有限的训练数据下泛化到未见过的分布外（OOD）预报场景具有重要意义。为解决这一障碍，现有方法主要集中在发现适用于各种PDE动力学轨迹的域泛化表示。然而，它们的零样本分布外泛化能力仍然不足，因为仍需额外的测试时样本进行领域特异性适应。这是因为PDE动力系统中的基本物理不变性尚未被研究或集成。为此，我们首先明确定义了两方面的PDE不变性原理，指出成分算子及其组合关系在不同领域和PDE系统演化中保持不变。接下来，为捕捉这种两方面的PDE不变性，我们提出了一种物理引导的不变学习方法iMOOE，该方法采用不变性对齐的算子专家混合架构和频率增强的不变学习目标。在模拟基准和实际应用中的广泛实验验证了iMOOE在不同分布外预报场景中的优越的内分布性能和零样本泛化能力。 

---
# TraitSpaces: Towards Interpretable Visual Creativity for Human-AI Co-Creation 

**Title (ZH)**: TraitSpaces: 向可解释的人工智能视觉创造力方向的人机共创 

**Authors**: Prerna Luthra  

**Link**: [PDF](https://arxiv.org/pdf/2509.24326)  

**Abstract**: We introduce a psychologically grounded and artist-informed framework for modeling visual creativity across four domains: Inner, Outer, Imaginative, and Moral Worlds. Drawing on interviews with practicing artists and theories from psychology, we define 12 traits that capture affective, symbolic, cultural, and ethical dimensions of this http URL 20k artworks from the SemArt dataset, we annotate images with GPT 4.1 using detailed, theory-aligned prompts, and evaluate the learnability of these traits from CLIP image embeddings. Traits such as Environmental Dialogicity and Redemptive Arc are predicted with high reliability ($R^2 \approx 0.64 - 0.68$), while others like Memory Imprint remain challenging, highlighting the limits of purely visual encoding. Beyond technical metrics, we visualize a "creativity trait-space" and illustrate how it can support interpretable, trait-aware co-creation - e.g., sliding along a Redemptive Arc axis to explore works of adversity and renewal. By linking cultural-aesthetic insights with computational modeling, our work aims not to reduce creativity to numbers, but to offer shared language and interpretable tools for artists, researchers, and AI systems to collaborate meaningfully. 

**Abstract (ZH)**: 基于心理依据和艺术家指导的跨领域视觉创造力建模框架：内外想象与道德世界中的情感、象征、文化与伦理维度探究 

---
# A study of Universal ODE approaches to predicting soil organic carbon 

**Title (ZH)**: 基于通用ODE方法预测土壤有机碳的研究 

**Authors**: Satyanarayana Raju G.V.V, Prathamesh Dinesh Joshi, Raj Abhijit Dandekar, Rajat Dandekar, Sreedath Panat  

**Link**: [PDF](https://arxiv.org/pdf/2509.24306)  

**Abstract**: Soil Organic Carbon (SOC) is a foundation of soil health and global climate resilience, yet its prediction remains difficult because of intricate physical, chemical, and biological processes. In this study, we explore a Scientific Machine Learning (SciML) framework built on Universal Differential Equations (UDEs) to forecast SOC dynamics across soil depth and time. UDEs blend mechanistic physics, such as advection diffusion transport, with neural networks that learn nonlinear microbial production and respiration. Using synthetic datasets, we systematically evaluated six experimental cases, progressing from clean, noise free benchmarks to stress tests with high (35%) multiplicative, spatially correlated noise. Our results highlight both the potential and limitations of the approach. In noise free and moderate noise settings, the UDE accurately reconstructed SOC dynamics. In clean terminal profile at 50 years (Case 4) achieved near perfect fidelity, with MSE = 1.6e-5, and R2 = 0.9999. Case 5, with 7% noise, remained robust (MSE = 3.4e-6, R2 = 0.99998), capturing depth wise SOC trends while tolerating realistic measurement uncertainty. In contrast, Case 3 (35% noise at t = 0) showed clear evidence of overfitting: the model reproduced noisy inputs with high accuracy but lost generalization against the clean truth (R2 = 0.94). Case 6 (35% noise at t = 50) collapsed toward overly smooth mean profiles, failing to capture depth wise variability and yielding negative R2, underscoring the limits of standard training under severe uncertainty. These findings suggest that UDEs are well suited for scalable, noise tolerant SOC forecasting, though advancing toward field deployment will require noise aware loss functions, probabilistic modelling, and tighter integration of microbial dynamics. 

**Abstract (ZH)**: 基于通用微分方程的科学机器学习框架在土壤有机碳动态预测中的应用：噪声鲁棒性研究 

---
# Q-Mirror: Unlocking the Multi-Modal Potential of Scientific Text-Only QA Pairs 

**Title (ZH)**: Q-镜像：释放科学文本型问答 pair 的多模态潜力 

**Authors**: Junying Wang, Zicheng Zhang, Ye Shen, Yalun Wu, Yingji Liang, Yijin Guo, Farong Wen, Wenzhe Li, Xuezhi Zhao, Qi Jia, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2509.24297)  

**Abstract**: High-quality, multi-modal benchmarks are crucial for advancing scientific reasoning in large models yet their manual creation is costly and unscalable. To address this bottleneck, we explore the potential for transforming Text-Only QA Pairs (TQAs) into high-quality Multi-Modal QA Pairs (MMQAs), which include three parts: 1) Task Definition \& Evaluation Rubric: We develop a TQA-to-MMQA framework and establish a comprehensive, multi-dimensional MMQA quality rubric that provides principles for the transformation. 2) Benchmark Construction: Then we construct two extensive benchmarks to rigorously evaluate state-of-the-art generation \& understanding models on the distinct tasks of MMQA generation \& MMQA quality evaluation. 3) Preliminary Solution: We develop an agentic system (Q-Mirror), which operationalizes our framework by integrating MMQA generation and evaluation into a closed loop for iterative refinement. Our experiments show that while state-of-the-art models can generate MMQAs, their outputs still leave substantial gaps, underscoring the need for reliable evaluation. We further demonstrate that top-tier understanding models align closely with human judgment in MMQA quality assessment. Leveraging both insights, the Q-Mirror agent raises average scores from 78.90 to 85.22 and pass rates from 72\% to 95\%, offering a practical path to large-scale scientific benchmarks. 

**Abstract (ZH)**: 高质量、多模态基准对于大型模型促进科学推理至关重要，但其手动创建成本高昂且不可扩展。为应对这一瓶颈，我们探索将文本_ONLY_问答对（TQAs）转换为高质量多模态问答对（MMQAs）的潜力，MMQAs包括三个部分：1）任务定义与评估准则：我们开发了一个TQA-to-MMQA框架，并建立了全面的多维度MMQA质量评估准则，提供了转换的原则。2）基准建设：接下来我们构建了两个广泛的基准，以严格评估最先进的生成与理解模型在多模态问答生成与多模态问答质量评估任务中的表现。3）初步解决方案：我们开发了一个自主系统（Q-Mirror），通过将多模态问答生成与评估集成到一个闭环中进行迭代优化，具体化了我们的框架。我们的实验表明，尽管最先进的模型能够生成MMQAs，但其输出仍然存在显著差距，强调了可靠评估的需求。此外，我们证明了顶级理解模型在多模态问答质量评估中与人类判断高度一致。结合这些见解，Q-Mirror代理将平均得分从78.90提高到85.22，通过率达到从72%提高到95%，提供了一条大规模科学基准建设的实用路径。 

---
# LAMP-PRo: Label-aware Attention for Multi-label Prediction of DNA- and RNA-binding Proteins using Protein Language Models 

**Title (ZH)**: LAMP-PRo：基于标签的注意力机制用于蛋白质语言模型预测DNA-和RNA结合蛋白多标签分类 

**Authors**: Nimisha Ghosh, Dheeran Sankaran, Rahul Balakrishnan Adhi, Sharath S, Amrut Anand  

**Link**: [PDF](https://arxiv.org/pdf/2509.24262)  

**Abstract**: Identifying DNA- (DBPs) and RNA-binding proteins (RBPs) is crucial for the understanding of cell function, molecular interactions as well as regulatory functions. Owing to their high similarity, most of the existing approaches face challenges in differentiating between DBPs and RBPs leading to high cross-prediction errors. Moreover, identifying proteins which bind to both DNA and RNA (DRBPs) is also quite a challenging task. In this regard, we propose a novel framework viz. LAMP-PRo which is based on pre-trained protein language model (PLM), attention mechanisms and multi-label learning to mitigate these issues. First, pre-trained PLM such ESM-2 is used for embedding the protein sequences followed by convolutional neural network (CNN). Subsequently multi-head self-attention mechanism is applied for the contextual information while label-aware attention is used to compute class-specific representations by attending to the sequence in a way that is tailored to each label (DBP, RBP and non-NABP) in a multi-label setup. We have also included a novel cross-label attention mechanism to explicitly capture dependencies between DNA- and RNA-binding proteins, enabling more accurate prediction of DRBP. Finally, a linear layer followed by a sigmoid function are used for the final prediction. Extensive experiments are carried out to compare LAMP-PRo with the existing methods wherein the proposed model shows consistent competent performance. Furthermore, we also provide visualization to showcase model interpretability, highlighting which parts of the sequence are most relevant for a predicted label. The original datasets are available at this http URL\_MMC and the codes are available at this https URL. 

**Abstract (ZH)**: 识别DNA-结合蛋白(DBPs)和RNA结合蛋白(RBPs)对于理解细胞功能、分子互动以及调控功能至关重要。由于它们的高度相似性，当前大多数方法在区分DBPs和RBPs时面临挑战，导致高交叉预测误差。此外，识别同时结合DNA和RNA的双重结合蛋白(DRBPs)也是一项艰巨的任务。为此，我们提出了一种新的框架LAMP-PRo，该框架基于预训练蛋白质语言模型(PLM)、注意力机制和多标签学习，以减轻这些问题。首先，使用预训练的PLM如ESM-2嵌入蛋白质序列，然后通过卷积神经网络(CNN)。随后应用多头自注意力机制处理上下文信息，同时使用标签感知注意力来通过针对每个标签(DBP、RBP和非NABP)特化的序列计算类特定表示。我们还引入了一种新颖的跨标签注意力机制以明确捕捉DNA-结合蛋白和RNA-结合蛋白之间的依赖性，使DRBP的准确预测更为可能。最后，使用线性层和Sigmoid函数进行最终预测。在与现有方法的广泛实验比较中，提出的模型显示出一致的竞争性能。此外，我们还提供了可视化以展示模型可解释性，突出哪些序列部分对预测标签最为相关。原始数据集可在以下链接下载：this http URL\_MMC，代码可在以下链接获取：this https URL。 

---
# Uni-NTFM: A Unified Foundation Model for EEG Signal Representation Learning 

**Title (ZH)**: 统一的脑电波信号表示学习基础模型：Uni-NTFM 

**Authors**: Zhisheng Chen, Yingwei Zhang, Qizhen Lan, Tianyu Liu, Huacan Wang, Yi Ding, Ziyu Jia, Ronghao Chen, Kun Wang, Xinliang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.24222)  

**Abstract**: Foundation models pretrained on various and unlabeled data have demonstrated significant success in natural language and vision, but their application to electroencephalography (EEG) remains challenged due to the signal's unique properties. Existing brain foundation models that inherit architectures designed for text or images lead to three limitations in pre-training: 1) conflating time-domain waveform patterns with frequency-domain rhythmic features in a single processing stream, 2) ignoring the critical spatial topology of electrodes with different standards, and 3) reliance on the inflexible, dense network to process functionally distinct EEG patterns. To address these challenges, we introduce the Unified Neural Topological Foundation Model (Uni-NTFM), which is designed based on neuroscience principles to produce universal and interpretable representations. Uni-NTFM integrates three core innovations: 1) a decoupled architecture parallelly encodes time, frequency, and raw signal representations before performing cross-domain feature integration; 2) a topological embedding mechanism to unify electrodes from different international standards and generate structured input sequences for brain regions; and 3) a Mixture-of-Experts neural Transformer that efficiently scales model capacity by routing signal patterns to specialized subnetworks. The largest model, Uni-NTFM$_{large}$, has a record-breaking 1.9B parameters and was pretrained on over 28,000 hours of diverse EEG data via a dual-domain masked reconstruction objective. Uni-NTFM significantly outperforms existing task-specific methods and foundation models across nine distinct downstream tasks under both linear probing and fine-tuning settings, demonstrating a superior ability to learn universal representations of brain activity. 

**Abstract (ZH)**: 基于统一神经拓扑基础模型在脑电信号中的通用和可解释表示 

---
# Metamorphic Testing for Audio Content Moderation Software 

**Title (ZH)**: 音频内容审核软件的 metamorphic 测试 

**Authors**: Wenxuan Wang, Yongjiang Wu, Junyuan Zhang, Shuqing Li, Yun Peng, Wenting Chen, Shuai Wang, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24215)  

**Abstract**: The rapid growth of audio-centric platforms and applications such as WhatsApp and Twitter has transformed the way people communicate and share audio content in modern society. However, these platforms are increasingly misused to disseminate harmful audio content, such as hate speech, deceptive advertisements, and explicit material, which can have significant negative consequences (e.g., detrimental effects on mental health). In response, researchers and practitioners have been actively developing and deploying audio content moderation tools to tackle this issue. Despite these efforts, malicious actors can bypass moderation systems by making subtle alterations to audio content, such as modifying pitch or inserting noise. Moreover, the effectiveness of modern audio moderation tools against such adversarial inputs remains insufficiently studied. To address these challenges, we propose MTAM, a Metamorphic Testing framework for Audio content Moderation software. Specifically, we conduct a pilot study on 2000 audio clips and define 14 metamorphic relations across two perturbation categories: Audio Features-Based and Heuristic perturbations. MTAM applies these metamorphic relations to toxic audio content to generate test cases that remain harmful while being more likely to evade detection. In our evaluation, we employ MTAM to test five commercial textual content moderation software and an academic model against three kinds of toxic content. The results show that MTAM achieves up to 38.6%, 18.3%, 35.1%, 16.7%, and 51.1% error finding rates (EFR) when testing commercial moderation software provided by Gladia, Assembly AI, Baidu, Nextdata, and Tencent, respectively, and it obtains up to 45.7% EFR when testing the state-of-the-art algorithms from the academy. 

**Abstract (ZH)**: 音频中心平台和应用（如WhatsApp和Twitter）的快速增长已改变人们在现代社会中进行音频内容交流和分享的方式。然而，这些平台正越来越多地被滥用以传播有害音频内容，如仇恨言论、欺骗性广告和露骨材料，这可能产生严重的负面影响（例如对心理健康造成损害）。针对这一问题，研究人员和实践者正在积极开发和部署音频内容审核工具。尽管如此，恶意行为者可以通过对音频内容进行细微修改（如修改音调或插入噪音）来规避审核系统，而且现代音频审核工具对这些对抗性输入的有效性研究仍不够充分。为应对这些挑战，我们提出了一种名为MTAM的音频内容审核软件的变形测试框架。具体而言，我们在2000个音频片段上进行试点研究，并定义了跨越两类扰动分类（基于音频特征和启发式扰动）的14种变形关系。MTAM应用这些变形关系对有毒音频内容进行测试，生成更有可能规避检测但仍保持有害性的测试案例。在评估中，我们使用MTAM对五款商用文本内容审核软件和一款学术模型进行了测试，针对三种类型的有毒内容。结果显示，MTAM分别在由Gladia、Assembly AI、Baidu、Nextdata和Tencent提供的商业审核软件中实现了多达38.6%、18.3%、35.1%、16.7%和51.1%的错误发现率（EFR），并在学术界的最新算法测试中实现了高达45.7%的错误发现率。 

---
# ASTROCO: Self-Supervised Conformer-Style Transformers for Light-Curve Embeddings 

**Title (ZH)**: ASTROCO: 自监督Conformer风格变换器及其在光变曲线嵌入中的应用 

**Authors**: Antony Tan, Pavlos Protopapas, Martina Cádiz-Leyton, Guillermo Cabrera-Vives, Cristobal Donoso-Oliva, Ignacio Becker  

**Link**: [PDF](https://arxiv.org/pdf/2509.24134)  

**Abstract**: We present AstroCo, a Conformer-style encoder for irregular stellar light curves. By combining attention with depthwise convolutions and gating, AstroCo captures both global dependencies and local features. On MACHO R-band, AstroCo outperforms Astromer v1 and v2, yielding 70 percent and 61 percent lower error respectively and a relative macro-F1 gain of about 7 percent, while producing embeddings that transfer effectively to few-shot classification. These results highlight AstroCo's potential as a strong and label-efficient foundation for time-domain astronomy. 

**Abstract (ZH)**: AstroCo：一种适用于不规则恒星光曲线的Conformer-style编码器 

---
# PerfBench: Can Agents Resolve Real-World Performance Bugs? 

**Title (ZH)**: PerfBench: 前沿基准：智能体能解决真实世界的性能 bug 吗？ 

**Authors**: Spandan Garg, Roshanak Zilouchian Moghaddam  

**Link**: [PDF](https://arxiv.org/pdf/2509.24091)  

**Abstract**: Performance bugs are inefficiencies in software that waste computational resources without causing functional failures, making them particularly challenging to detect and fix. While recent advances in Software Engineering agents have shown promise in automated bug fixing, existing benchmarks primarily focus on functional correctness and fail to evaluate agents' abilities to identify and resolve non-functional issues like performance bugs. We introduce PerfBench, a benchmark comprising 81 real-world performance bug-fixing tasks from popular .NET repositories on GitHub. Unlike existing benchmarks that rely on pre-existing test suites, PerfBench features a novel evaluation harness that allows agents to generate their own performance benchmarks and validates fixes by comparing execution metrics collected for developer fix and agent fix. Each task in PerfBench is derived from actual developer fixes linked to performance-related issues, which are then verified by human experts, ensuring real-world relevance. Our evaluation reveals that current state-of-the-art coding agents struggle with performance optimization tasks, with baseline OpenHands agent achieving only a ~3% success rate on our benchmark. We develop OpenHands-Perf-Agent, which incorporates performance-aware tooling and instructions and achieves a ~20% success rate on the benchmark. We show that by ensuring the agent has proper instructions to benchmark its changes and tooling for benchmark output processing, we can improve the agent performance significantly, but room for improvement still remains. PerfBench provides a challenging test set for furthering the capabilities of agents in fixing performance issues. 

**Abstract (ZH)**: PerformanceBench：一个用于评估软件代理解决性能问题能力的新基准 

---
# AQUAIR: A High-Resolution Indoor Environmental Quality Dataset for Smart Aquaculture Monitoring 

**Title (ZH)**: AQUAIR: 一种高分辨率室内环境质量数据集，用于智能水产监控 

**Authors**: Youssef Sabiri, Walid Houmaidi, Ouail El Maadi, Yousra Chtouki  

**Link**: [PDF](https://arxiv.org/pdf/2509.24069)  

**Abstract**: Smart aquaculture systems depend on rich environmental data streams to protect fish welfare, optimize feeding, and reduce energy use. Yet public datasets that describe the air surrounding indoor tanks remain scarce, limiting the development of forecasting and anomaly-detection tools that couple head-space conditions with water-quality dynamics. We therefore introduce AQUAIR, an open-access public dataset that logs six Indoor Environmental Quality (IEQ) variables--air temperature, relative humidity, carbon dioxide, total volatile organic compounds, PM2.5 and PM10--inside a fish aquaculture facility in Amghass, Azrou, Morocco. A single Awair HOME monitor sampled every five minutes from 14 October 2024 to 9 January 2025, producing more than 23,000 time-stamped observations that are fully quality-controlled and publicly archived on Figshare. We describe the sensor placement, ISO-compliant mounting height, calibration checks against reference instruments, and an open-source processing pipeline that normalizes timestamps, interpolates short gaps, and exports analysis-ready tables. Exploratory statistics show stable conditions (median CO2 = 758 ppm; PM2.5 = 12 micrograms/m3) with pronounced feeding-time peaks, offering rich structure for short-horizon forecasting, event detection, and sensor drift studies. AQUAIR thus fills a critical gap in smart aquaculture informatics and provides a reproducible benchmark for data-centric machine learning curricula and environmental sensing research focused on head-space dynamics in recirculating aquaculture systems. 

**Abstract (ZH)**: 智能水产养殖系统依赖丰富的环境数据流来保护鱼的福利、优化投喂和减少能耗。然而，描述室内水槽周围空气的公开数据集仍然稀缺，限制了将空间条件与水质动态耦合的预测和异常检测工具的发展。因此，我们介绍了AQUAIR，一个开放访问的公开数据集，在摩洛哥阿姆加斯、阿祖鲁的鱼塘设施内记录六种室内环境质量（IEQ）变量——空气温度、相对湿度、二氧化碳、总挥发性有机化合物、PM2.5和PM10。从2024年10月14日至2025年1月9日，每5分钟采样一次的单个Awair HOME监测器生成了超过23,000个带时间戳的观测数据，并在Figshare上完全质量控制并公开存档。我们描述了传感器布局、ISO合规的安装高度、参考仪器校准检查以及开源处理管道，该管道规范时间戳、插补短间隙并导出分析准备的表格。初步统计结果显示稳定条件（中位数二氧化碳=758 ppm；PM2.5=12微克/立方米），在投喂时间出现峰值，为短期预测、事件检测和传感器漂移研究提供了丰富的结构。AQUAIR因此填补了智能水产养殖信息技术中的一个重要空白，并为基于数据的机器学习课程和关注循环水产养殖系统空间动态的环境传感研究提供了可重复的基准。 

---
# In-Context Compositional Q-Learning for Offline Reinforcement Learning 

**Title (ZH)**: 上下文依赖组合Q学习在离线强化学习中的应用 

**Authors**: Qiushui Xu, Yuhao Huang, Yushu Jiang, Lei Song, Jinyu Wang, Wenliang Zheng, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2509.24067)  

**Abstract**: Accurately estimating the Q-function is a central challenge in offline reinforcement learning. However, existing approaches often rely on a single global Q-function, which struggles to capture the compositional nature of tasks involving diverse subtasks. We propose In-context Compositional Q-Learning (\texttt{ICQL}), the first offline RL framework that formulates Q-learning as a contextual inference problem, using linear Transformers to adaptively infer local Q-functions from retrieved transitions without explicit subtask labels. Theoretically, we show that under two assumptions--linear approximability of the local Q-function and accurate weight inference from retrieved context--\texttt{ICQL} achieves bounded Q-function approximation error, and supports near-optimal policy extraction. Empirically, \texttt{ICQL} substantially improves performance in offline settings: improving performance in kitchen tasks by up to 16.4\%, and in Gym and Adroit tasks by up to 8.6\% and 6.3\%. These results highlight the underexplored potential of in-context learning for robust and compositional value estimation, positioning \texttt{ICQL} as a principled and effective framework for offline RL. 

**Abstract (ZH)**: 准确估计Q函数是离线强化学习中的核心挑战。现有方法往往依赖于单一全局Q函数，难以捕捉包含多样子任务的组合性质任务。我们提出In-context Compositional Q-Learning (\texttt{ICQL})，这是第一个将Q学习形式化为上下文推理问题的离线RL框架，采用线性Transformer从检索得到的转换中自适应地推断局部Q函数，而无需显式子任务标签。理论分析表明，在局部Q函数线性可近似和从检索上下文准确推断权重的假设下，\texttt{ICQL}实现有界Q函数近似误差，并支持接近最优策略提取。实验结果显示，在离线设置中，\texttt{ICQL}显著提高了性能：在厨房任务中提高了16.4%，在Gym和Adroit任务中分别提高了8.6%和6.3%。这些结果突显了上下文学习在鲁棒和组合价值估计方面的未充分开发的潜力，将\texttt{ICQL}定位为基于原理且有效的离线RL框架。 

---
# A Second-Order Perspective on Pruning at Initialization and Knowledge Transfer 

**Title (ZH)**: 初始化时的剪枝第二-order视角与知识迁移 

**Authors**: Leonardo Iurada, Beatrice Occhiena, Tatiana Tommasi  

**Link**: [PDF](https://arxiv.org/pdf/2509.24066)  

**Abstract**: The widespread availability of pre-trained vision models has enabled numerous deep learning applications through their transferable representations. However, their computational and storage costs often limit practical deployment. Pruning-at-Initialization has emerged as a promising approach to compress models before training, enabling efficient task-specific adaptation. While conventional wisdom suggests that effective pruning requires task-specific data, this creates a challenge when downstream tasks are unknown in advance. In this paper, we investigate how data influences the pruning of pre-trained vision models. Surprisingly, pruning on one task retains the model's zero-shot performance also on unseen tasks. Furthermore, fine-tuning these pruned models not only improves performance on original seen tasks but can recover held-out tasks' performance. We attribute this phenomenon to the favorable loss landscapes induced by extensive pre-training on large-scale datasets. 

**Abstract (ZH)**: 预训练视觉模型的剪枝初始化：数据的影响及性能恢复 

---
# End-to-end Topographic Auditory Models Replicate Signatures of Human Auditory Cortex 

**Title (ZH)**: 端到端拓扑听觉模型再现人类听觉皮层的特征 

**Authors**: Haider Al-Tahan, Mayukh Deb, Jenelle Feather, N. Apurva Ratan Murty  

**Link**: [PDF](https://arxiv.org/pdf/2509.24039)  

**Abstract**: The human auditory cortex is topographically organized. Neurons with similar response properties are spatially clustered, forming smooth maps for acoustic features such as frequency in early auditory areas, and modular regions selective for music and speech in higher-order cortex. Yet, evaluations for current computational models of auditory perception do not measure whether such topographic structure is present in a candidate model. Here, we show that cortical topography is not present in the previous best-performing models at predicting human auditory fMRI responses. To encourage the emergence of topographic organization, we adapt a cortical wiring-constraint loss originally designed for visual perception. The new class of topographic auditory models, TopoAudio, are trained to classify speech, and environmental sounds from cochleagram inputs, with an added constraint that nearby units on a 2D cortical sheet develop similar tuning. Despite these additional constraints, TopoAudio achieves high accuracy on benchmark tasks comparable to the unconstrained non-topographic baseline models. Further, TopoAudio predicts the fMRI responses in the brain as well as standard models, but unlike standard models, TopoAudio develops smooth, topographic maps for tonotopy and amplitude modulation (common properties of early auditory representation, as well as clustered response modules for music and speech (higher-order selectivity observed in the human auditory cortex). TopoAudio is the first end-to-end biologically grounded auditory model to exhibit emergent topography, and our results emphasize that a wiring-length constraint can serve as a general-purpose regularization tool to achieve biologically aligned representations. 

**Abstract (ZH)**: 人类听皮层按拓扑结构组织。具有相似响应性质的神经元在空间上聚类，形成早期听觉区域中频率等声学特征的平滑地图，并在高级皮层中形成对音乐和语言具有模块化选择性的区域。然而，当前听觉感知计算模型的评估并未衡量候选模型中是否存在这种拓扑结构。我们展示了之前的最佳预测人类听觉fMRI反应的模型中不存在皮层拓扑结构。为了鼓励拓扑结构的出现，我们采用了一种最初为视觉感知设计的皮层连接约束损失。新的拓扑听觉模型类TopoAudio被训练用于从耳蜗图输入分类语音和环境声音，并且增加了附近二维皮层单元发展相似调谐的约束。尽管增加了这些额外约束，TopoAudio在基准任务上的准确度与未受约束的非拓扑基线模型相当。此外，TopoAudio预测大脑的fMRI反应与标准模型一样准确，但与标准模型不同的是，TopoAudio发展了对于音调定位和振幅调制的平滑拓扑图（早期听觉表征的常见属性），以及对音乐和语音具有聚类反应模块（人类听觉皮层中观察到的高阶选择性）。TopoAudio是第一个表现出新兴拓扑结构的端到端生物基础听觉模型，我们的结果强调，连接长度约束可以作为一种通用正则化工具，以实现生物对齐的表示。 

---
# GPS-MTM: Capturing Pattern of Normalcy in GPS-Trajectories with self-supervised learning 

**Title (ZH)**: GPS-MTM：使用自我监督学习捕捉GPS轨迹中的正常模式 

**Authors**: Umang Garg, Bowen Zhang, Anantanjit Subrahmanya, Chandrakanth Gudavalli, BS Manjunath  

**Link**: [PDF](https://arxiv.org/pdf/2509.24031)  

**Abstract**: Foundation models have driven remarkable progress in text, vision, and video understanding, and are now poised to unlock similar breakthroughs in trajectory modeling. We introduce the GPSMasked Trajectory Transformer (GPS-MTM), a foundation model for large-scale mobility data that captures patterns of normalcy in human movement. Unlike prior approaches that flatten trajectories into coordinate streams, GPS-MTM decomposes mobility into two complementary modalities: states (point-of-interest categories) and actions (agent transitions). Leveraging a bi-directional Transformer with a self-supervised masked modeling objective, the model reconstructs missing segments across modalities, enabling it to learn rich semantic correlations without manual labels. Across benchmark datasets, including Numosim-LA, Urban Anomalies, and Geolife, GPS-MTM consistently outperforms on downstream tasks such as trajectory infilling and next-stop prediction. Its advantages are most pronounced in dynamic tasks (inverse and forward dynamics), where contextual reasoning is critical. These results establish GPS-MTM as a robust foundation model for trajectory analytics, positioning mobility data as a first-class modality for large-scale representation learning. Code is released for further reference. 

**Abstract (ZH)**: GPSMasked 轨迹变换器：大规模移动数据的基础模型 

---
# From Edge to HPC: Investigating Cross-Facility Data Streaming Architectures 

**Title (ZH)**: 从边缘到超算：探究跨设施数据流架构 

**Authors**: Anjus George, Michael Brim, Christopher Zimmer, David Rogers, Sarp Oral, Zach Mayes  

**Link**: [PDF](https://arxiv.org/pdf/2509.24030)  

**Abstract**: In this paper, we investigate three cross-facility data streaming architectures, Direct Streaming (DTS), Proxied Streaming (PRS), and Managed Service Streaming (MSS). We examine their architectural variations in data flow paths and deployment feasibility, and detail their implementation using the Data Streaming to HPC (DS2HPC) architectural framework and the SciStream memory-to-memory streaming toolkit on the production-grade Advanced Computing Ecosystem (ACE) infrastructure at Oak Ridge Leadership Computing Facility (OLCF). We present a workflow-specific evaluation of these architectures using three synthetic workloads derived from the streaming characteristics of scientific workflows. Through simulated experiments, we measure streaming throughput, round-trip time, and overhead under work sharing, work sharing with feedback, and broadcast and gather messaging patterns commonly found in AI-HPC communication motifs. Our study shows that DTS offers a minimal-hop path, resulting in higher throughput and lower latency, whereas MSS provides greater deployment feasibility and scalability across multiple users but incurs significant overhead. PRS lies in between, offering a scalable architecture whose performance matches DTS in most cases. 

**Abstract (ZH)**: 本文研究了三种跨设施数据流架构——直接流（DTS）、代理流（PRS）和管理服务流（MSS），探讨了它们在数据流路径和部署可行性方面的架构变体，并使用Data Streaming to HPC（DS2HPC）架构框架和SciStream内存到内存流传输工具包在橡树岭领导计算设施（OLCF）的生产级先进计算生态系统（ACE）基础设施上详细阐述了其实现。通过特定工作流的评估，使用源自科学工作流流特性的人工合成工作负载进行评估。通过模拟实验，测量了工作分担、带有反馈的工作分担、广播和收集消息模式下的流传输吞吐量、往返时间和开销。研究结果表明，DTS提供了最少跳跃路径，从而实现更高的吞吐量和更低的延迟，而MSS提供了更好的部署可行性和跨多个用户的大规模扩展性，但会带来显著的开销。PRS介于两者之间，提供了一种可扩展的架构，其性能在大多数情况下与DTS匹配。 

---
# Easy Turn: Integrating Acoustic and Linguistic Modalities for Robust Turn-Taking in Full-Duplex Spoken Dialogue Systems 

**Title (ZH)**: Easy Turn: 结合声学和语言模态实现全双工 spoken 对话系统中稳健的轮替 

**Authors**: Guojian Li, Chengyou Wang, Hongfei Xue, Shuiyuan Wang, Dehui Gao, Zihan Zhang, Yuke Lin, Wenjie Li, Longshuai Xiao, Zhonghua Fu, Lei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2509.23938)  

**Abstract**: Full-duplex interaction is crucial for natural human-machine communication, yet remains challenging as it requires robust turn-taking detection to decide when the system should speak, listen, or remain silent. Existing solutions either rely on dedicated turn-taking models, most of which are not open-sourced. The few available ones are limited by their large parameter size or by supporting only a single modality, such as acoustic or linguistic. Alternatively, some approaches finetune LLM backbones to enable full-duplex capability, but this requires large amounts of full-duplex data, which remain scarce in open-source form. To address these issues, we propose Easy Turn, an open-source, modular turn-taking detection model that integrates acoustic and linguistic bimodal information to predict four dialogue turn states: complete, incomplete, backchannel, and wait, accompanied by the release of Easy Turn trainset, a 1,145-hour speech dataset designed for training turn-taking detection models. Compared to existing open-source models like TEN Turn Detection and Smart Turn V2, our model achieves state-of-the-art turn-taking detection accuracy on our open-source Easy Turn testset. The data and model will be made publicly available on GitHub. 

**Abstract (ZH)**: 全双工交互对于自然的人机通信至关重要，但依然具有挑战性，因为它需要 robust 的轮流转换检测以决定系统应该何时说话、聆听或保持沉默。现有的解决方案要么依赖专用的轮流转换模型，但大多数模型并未开源；现有的少数开源模型要么参数量大，要么仅支持单一模态，如声学或语言。此外，一些方法通过微调大语言模型来实现全双工能力，但这需要大量全双工数据，而开源数据仍然稀缺。为解决这些问题，我们提出了一种开源且模块化的轮流转换检测模型 Easy Turn，它结合了声学和语言的双模态信息来预测四种对话轮流状态：完整、不完整、响应性反馈和等待，并同时发布了 Easy Turn 训练集，这是一个设计用于训练轮流转换检测模型的 1,145 小时语音数据集。与现有的开源模型（如 TEN Turn Detection 和 Smart Turn V2）相比，我们的模型在我们的开源 Easy Turn 测试集上实现了最先进的轮流转换检测准确性。数据和模型将在 GitHub 上公开发布。 

---
# Diffusion Models are Kelly Gamblers 

**Title (ZH)**: 扩散模型是凯利赌徒 

**Authors**: Akhil Premkumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.23937)  

**Abstract**: We draw a connection between diffusion models and the Kelly criterion for maximizing returns in betting games. We find that conditional diffusion models store additional information to bind the signal $X$ with the conditioning information $Y$, equal to the mutual information between them. Classifier-free guidance effectively boosts the mutual information between $X$ and $Y$ at sampling time. This is especially helpful in image models, since the mutual information between images and their labels is low, a fact which is intimately connected to the manifold hypothesis. Finally, we point out some nuances in the popular perspective that diffusion models are infinitely deep autoencoders. In doing so, we relate the denoising loss to the Fermi Golden Rule from quantum mechanics. 

**Abstract (ZH)**: 我们将扩散模型与赌博游戏中最大化回报的凯利准则进行连接。我们发现条件扩散模型存储了额外的信息，将信号$X$与条件信息$Y$绑定，等价于它们之间的互信息。无分类引导有效地在采样时间提高$X$与$Y$之间的互信息。特别是在图像模型中这一点尤为重要，因为图像与其标签之间的互信息较低，这一事实与流形假设密切相关。最后，我们指出对扩散模型是无限深自编码器的流行观点存在一些细微差别，并将去噪损失与量子力学中的费米-金规则相联系。 

---
# Graph Mixing Additive Networks 

**Title (ZH)**: 图混合加性网络 

**Authors**: Maya Bechler-Speicher, Andrea Zerio, Maor Huri, Marie Vibeke Vestergaard, Ran Gilad-Bachrach, Tine Jess, Samir Bhatt, Aleksejs Sazonovs  

**Link**: [PDF](https://arxiv.org/pdf/2509.23923)  

**Abstract**: We introduce GMAN, a flexible, interpretable, and expressive framework that extends Graph Neural Additive Networks (GNANs) to learn from sets of sparse time-series data. GMAN represents each time-dependent trajectory as a directed graph and applies an enriched, more expressive GNAN to each graph. It allows users to control the interpretability-expressivity trade-off by grouping features and graphs to encode priors, and it provides feature, node, and graph-level interpretability. On real-world datasets, including mortality prediction from blood tests and fake-news detection, GMAN outperforms strong non-interpretable black-box baselines while delivering actionable, domain-aligned explanations. 

**Abstract (ZH)**: GMAN：一种灵活、可解释且表达能力强的框架，用于处理稀疏时间序列数据集的学习 

---
# Continual Learning to Generalize Forwarding Strategies for Diverse Mobile Wireless Networks 

**Title (ZH)**: 持续学习以泛化转发策略于多样化的移动无线网络 

**Authors**: Cheonjin Park, Victoria Manfredi, Xiaolan Zhang, Chengyi Liu, Alicia P Wolfe, Dongjin Song, Sarah Tasneem, Bing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23913)  

**Abstract**: Deep reinforcement learning (DRL) has been successfully used to design forwarding strategies for multi-hop mobile wireless networks. While such strategies can be used directly for networks with varied connectivity and dynamic conditions, developing generalizable approaches that are effective on scenarios significantly different from the training environment remains largely unexplored. In this paper, we propose a framework to address the challenge of generalizability by (i) developing a generalizable base model considering diverse mobile network scenarios, and (ii) using the generalizable base model for new scenarios, and when needed, fine-tuning the base model using a small amount of data from the new scenarios. To support this framework, we first design new features to characterize network variation and feature quality, thereby improving the information used in DRL-based forwarding decisions. We then develop a continual learning (CL) approach able to train DRL models across diverse network scenarios without ``catastrophic forgetting.'' Using extensive evaluation, including real-world scenarios in two cities, we show that our approach is generalizable to unseen mobility scenarios. Compared to a state-of-the-art heuristic forwarding strategy, it leads to up to 78% reduction in delay, 24% improvement in delivery rate, and comparable or slightly higher number of forwards. 

**Abstract (ZH)**: 深强化学习在多跳移动无线网络转发策略设计中的应用：通过通用基础模型和持续学习方法实现泛化能力 

---
# EWC-Guided Diffusion Replay for Exemplar-Free Continual Learning in Medical Imaging 

**Title (ZH)**: 基于EWC指导的扩散重放以实现无范例持续学习在医学成像中 

**Authors**: Anoushka Harit, William Prew, Zhongtian Sun, Florian Markowetz  

**Link**: [PDF](https://arxiv.org/pdf/2509.23906)  

**Abstract**: Medical imaging foundation models must adapt over time, yet full retraining is often blocked by privacy constraints and cost. We present a continual learning framework that avoids storing patient exemplars by pairing class conditional diffusion replay with Elastic Weight Consolidation. Using a compact Vision Transformer backbone, we evaluate across eight MedMNIST v2 tasks and CheXpert. On CheXpert our approach attains 0.851 AUROC, reduces forgetting by more than 30\% relative to DER\texttt{++}, and approaches joint training at 0.869 AUROC, while remaining efficient and privacy preserving. Analyses connect forgetting to two measurable factors: fidelity of replay and Fisher weighted parameter drift, highlighting the complementary roles of replay diffusion and synaptic stability. The results indicate a practical route for scalable, privacy aware continual adaptation of clinical imaging models. 

**Abstract (ZH)**: 医疗影像基础模型必须随时间进行适应，但完全重新训练往往受限于隐私约束和成本。我们提出了一种持续学习框架，通过将类条件扩散重放与弹性权重巩固相结合，避免存储患者示例。采用紧凑的视觉变压器骨干，我们在八个MedMNIST v2任务和CheXpert上进行了评估。在CheXpert上，我们的方法取得了0.851的AUROC，相对于DER\texttt{++}减少了超过30%的遗忘，并接近联合训练的0.869 AUROC，同时保持高效和隐私保护。分析将遗忘关联到两个可测量的因素：重放保真度和费舍尔加权参数漂移，强调了重放扩散和突触稳定性的互补作用。结果表明了一条实用途径，用于实现临床影像模型的 scalable、隐私意识持续适应。 

---
# Interpreting deep learning-based stellar mass estimation via causal analysis and mutual information decomposition 

**Title (ZH)**: 基于因果分析和互信息分解的深度学习恒星质量估计解释 

**Authors**: Wei Zhang, Qiufan Lin, Yuan-Sen Ting, Shupei Chen, Hengxin Ruan, Song Li, Yifan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23901)  

**Abstract**: End-to-end deep learning models fed with multi-band galaxy images are powerful data-driven tools used to estimate galaxy physical properties in the absence of spectroscopy. However, due to a lack of interpretability and the associational nature of such models, it is difficult to understand how the information additional to integrated photometry (e.g., morphology) contributes to the estimation task. Improving our understanding in this field would enable further advances into unraveling the physical connections among galaxy properties and optimizing data exploitation. Therefore, our work is aimed at interpreting the deep learning-based estimation of stellar mass via two interpretability techniques: causal analysis and mutual information decomposition. The former reveals the causal paths between multiple variables beyond nondirectional statistical associations, while the latter quantifies the multicomponent contributions (i.e., redundant, unique, and synergistic) of different input data to the stellar mass estimation. Using data from the Sloan Digital Sky Survey (SDSS) and the Wide-field Infrared Survey Explorer (WISE), we obtained meaningful results that provide physical interpretations for image-based models. Our work demonstrates the gains from combining deep learning with interpretability techniques, and holds promise in promoting more data-driven astrophysical research (e.g., astrophysical parameter estimations and investigations on complex multivariate physical processes). 

**Abstract (ZH)**: 基于多频段星系图像的端到端深度学习模型能够从无光谱数据中估计星系物理性质，然而由于这些模型缺乏可解释性和关联性本质，难以理解额外的综合光度学信息（如形态学）如何 contributes到估计任务。为了促进对该领域的理解并优化数据利用，我们的工作旨在通过因果分析和互信息分解两种可解释性技术来解释基于深度学习的恒星质量估计。前者揭示了多个变量之间的因果路径，而后者量化了不同输入数据对恒星质量估计的多成分贡献（即冗余、独特和协同贡献）。使用斯隆数字天空巡天（SDSS）和广域红外巡天探索者（WISE）数据，我们获得了有意义的结果，为图像模型提供了物理解释。我们的工作展示了将深度学习与可解释性技术结合的优势，并有望促进更具数据驱动性的天体物理研究（例如天体物理参数估计和复杂多变量物理过程的研究）。 

---
# Gradient Flow Convergence Guarantee for General Neural Network Architectures 

**Title (ZH)**: 梯度流收敛性保证：通用神经网络架构 

**Authors**: Yash Jakhmola  

**Link**: [PDF](https://arxiv.org/pdf/2509.23887)  

**Abstract**: A key challenge in modern deep learning theory is to explain the remarkable success of gradient-based optimization methods when training large-scale, complex deep neural networks. Though linear convergence of such methods has been proved for a handful of specific architectures, a united theory still evades researchers. This article presents a unified proof for linear convergence of continuous gradient descent, also called gradient flow, while training any neural network with piecewise non-zero polynomial activations or ReLU, sigmoid activations. Our primary contribution is a single, general theorem that not only covers architectures for which this result was previously unknown but also consolidates existing results under weaker assumptions. While our focus is theoretical and our results are only exact in the infinitesimal step size limit, we nevertheless find excellent empirical agreement between the predictions of our result and those of the practical step-size gradient descent method. 

**Abstract (ZH)**: 现代深度学习理论中的一个关键挑战是在训练大规模复杂深度神经网络时，解释基于梯度的优化方法的显著成功。尽管已经证明了这些方法在少数特定架构上的线性收敛性，但统一理论仍未能让研究人员达成共识。本文给出了对任何具有分段非零多项式激活或ReLU、Sigmoid激活的神经网络使用连续梯度下降（也称为梯度流）方法的线性收敛性的统一证明。我们的主要贡献是一个通用的单一定理，不仅涵盖了之前未知架构的结果，还将在较弱假设下汇总了现有结果。虽然我们关注的是理论方面，且结果仅在无穷小步长极限下精确，但我们仍然发现我们的结果预测与实际步长梯度下降方法的预测之间有很好的实验一致性。 

---
# Tunable-Generalization Diffusion Powered by Self-Supervised Contextual Sub-Data for Low-Dose CT Reconstruction 

**Title (ZH)**: 基于自主监督上下文子数据的可调通用扩散用于低剂量CT重建 

**Authors**: Guoquan Wei, Zekun Zhou, Liu Shi, Wenzhe Shan, Qiegen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23885)  

**Abstract**: Current models based on deep learning for low-dose CT denoising rely heavily on paired data and generalize poorly. Even the more concerned diffusion models need to learn the distribution of clean data for reconstruction, which is difficult to satisfy in medical clinical applications. At the same time, self-supervised-based methods face the challenge of significant degradation of generalizability of models pre-trained for the current dose to expand to other doses. To address these issues, this paper proposes a novel method of tunable-generalization diffusion powered by self-supervised contextual sub-data for low-dose CT reconstruction, named SuperDiff. Firstly, a contextual subdata similarity adaptive sensing strategy is designed for denoising centered on the LDCT projection domain, which provides an initial prior for the subsequent progress. Subsequently, the initial prior is used to combine knowledge distillation with a deep combination of latent diffusion models for optimizing image details. The pre-trained model is used for inference reconstruction, and the pixel-level self-correcting fusion technique is proposed for fine-grained reconstruction of the image domain to enhance the image fidelity, using the initial prior and the LDCT image as a guide. In addition, the technique is flexibly applied to the generalization of upper and lower doses or even unseen doses. Dual-domain strategy cascade for self-supervised LDCT denoising, SuperDiff requires only LDCT projection domain data for training and testing. Full qualitative and quantitative evaluations on both datasets and real data show that SuperDiff consistently outperforms existing state-of-the-art methods in terms of reconstruction and generalization performance. 

**Abstract (ZH)**: 基于自监督上下文子数据的可调泛化扩散方法：低剂量CT重建（SuperDiff） 

---
# Multi-Value-Product Retrieval-Augmented Generation for Industrial Product Attribute Value Identification 

**Title (ZH)**: 基于多值产品检索增强生成的工业产品属性值识别 

**Authors**: Huike Zou, Haiyang Yang, Yindu Su, Liyu Chen, Chengbao Lian, Qingheng Zhang, Shuguang Han, Jufeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.23874)  

**Abstract**: Identifying attribute values from product profiles is a key task for improving product search, recommendation, and business analytics on e-commerce platforms, which we called Product Attribute Value Identification (PAVI) . However, existing PAVI methods face critical challenges, such as cascading errors, inability to handle out-of-distribution (OOD) attribute values, and lack of generalization capability. To address these limitations, we introduce Multi-Value-Product Retrieval-Augmented Generation (MVP-RAG), combining the strengths of retrieval, generation, and classification paradigms. MVP-RAG defines PAVI as a retrieval-generation task, where the product title description serves as the query, and products and attribute values act as the corpus. It first retrieves similar products of the same category and candidate attribute values, and then generates the standardized attribute values. The key advantages of this work are: (1) the proposal of a multi-level retrieval scheme, with products and attribute values as distinct hierarchical levels in PAVI domain (2) attribute value generation of large language model to significantly alleviate the OOD problem and (3) its successful deployment in a real-world industrial environment. Extensive experimental results demonstrate that MVP-RAG performs better than the state-of-the-art baselines. 

**Abstract (ZH)**: 产品属性值识别（PAVI）是从产品描述中识别属性值的关键任务，有助于电商平台的产品搜索、推荐和商业分析。然而，现有的PAVI方法面临关键挑战，如连锁错误、无法处理分布外（OOD）属性值以及缺乏泛化能力。为解决这些限制，我们引入了多值产品检索增强生成（MVP-RAG）方法，结合了检索、生成和分类 paradigm 的优势。MVP-RAG 将 PAVI 定义为一个检索-生成任务，其中产品标题描述作为查询，产品和属性值作为语料库。它首先检索相同类别相似的产品和候选属性值，然后生成标准化的属性值。本文的关键优势在于：（1）提出一个多级检索方案，产品和属性值在PAVI领域作为不同的层级；（2）通过大型语言模型生成属性值，显著缓解分布外问题；（3）成功部署于实际工业环境。广泛的实验结果表明，MVP-RAG 在与先进基线方法的对比中表现更优。 

---
# Taught Well Learned Ill: Towards Distillation-conditional Backdoor Attack 

**Title (ZH)**: 教得好了也会受骗：面向蒸馏条件后门攻击 

**Authors**: Yukun Chen, Boheng Li, Yu Yuan, Leyi Qi, Yiming Li, Tianwei Zhang, Zhan Qin, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2509.23871)  

**Abstract**: Knowledge distillation (KD) is a vital technique for deploying deep neural networks (DNNs) on resource-constrained devices by transferring knowledge from large teacher models to lightweight student models. While teacher models from third-party platforms may undergo security verification (\eg, backdoor detection), we uncover a novel and critical threat: distillation-conditional backdoor attacks (DCBAs). DCBA injects dormant and undetectable backdoors into teacher models, which become activated in student models via the KD process, even with clean distillation datasets. While the direct extension of existing methods is ineffective for DCBA, we implement this attack by formulating it as a bilevel optimization problem and proposing a simple yet effective method (\ie, SCAR). Specifically, the inner optimization simulates the KD process by optimizing a surrogate student model, while the outer optimization leverages outputs from this surrogate to optimize the teacher model for implanting the conditional backdoor. Our SCAR addresses this complex optimization utilizing an implicit differentiation algorithm with a pre-optimized trigger injection function. Extensive experiments across diverse datasets, model architectures, and KD techniques validate the effectiveness of our SCAR and its resistance against existing backdoor detection, highlighting a significant yet previously overlooked vulnerability in the KD process. Our code is available at this https URL. 

**Abstract (ZH)**: 知识蒸馏（KD）的安全威胁：蒸馏条件下的后门攻击（DCBA） 

---
# Efficient Multi-turn RL for GUI Agents via Decoupled Training and Adaptive Data Curation 

**Title (ZH)**: 通过解耦训练和自适应数据管理的高效多轮RL для GUI代理 

**Authors**: Pengxiang Li, Zechen Hu, Zirui Shang, Jingrong Wu, Yang Liu, Hui Liu, Zhi Gao, Chenrui Shi, Bofei Zhang, Zihao Zhang, Xiaochuan Shi, Zedong YU, Yuwei Wu, Xinxiao Wu, Yunde Jia, Liuyu Xiang, Zhaofeng He, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.23866)  

**Abstract**: Vision-language model (VLM) based GUI agents show promise for automating complex desktop and mobile tasks, but face significant challenges in applying reinforcement learning (RL): (1) slow multi-turn interactions with GUI environments for policy rollout, and (2) insufficient high-quality agent-environment interactions for policy learning. To address these challenges, we propose DART, a Decoupled Agentic RL Training framework for GUI agents, which coordinates heterogeneous modules in a highly decoupled manner. DART separates the training system into four asynchronous modules: environment cluster, rollout service, data manager, and trainer. This design enables non-blocking communication, asynchronous training, rollout-wise trajectory sampling, and per-worker model synchronization, significantly improving the system efficiency: 1.6*GPU utilization for rollout, 1.9* training throughput, and 5.5* environment utilization. To facilitate effective learning from abundant samples, we introduce an adaptive data curation scheme: (1) pre-collecting successful trajectories for challenging tasks to supplement sparse success in online sampling; (2) dynamically adjusting rollout numbers and trajectory lengths based on task difficulty; (3) training selectively on high-entropy steps to prioritize critical decisions; (4) stabilizing learning via truncated importance sampling for policy mismatch between policy rollout and updating. On the OSWorld benchmark, DART-GUI-7B achieves a 42.13% task success rate, a 14.61% absolute gain over the base model, and 7.34% higher than open-source SOTA. We will fully open-source our training framework, data, and model checkpoints via this http URL, which we believe is a timely contribution to the open-source community of agentic RL training. 

**Abstract (ZH)**: 基于视觉语言模型的GUI代理在自动化复杂桌面和移动任务方面显示出前景，但在应用强化学习方面面临重大挑战：(1) 与GUI环境进行多轮交互的效率低下，(2) 用于策略学习的代理-环境交互不足。为应对这些挑战，我们提出了一种名为DART的分阶代理强化学习训练框架，该框架以高度解耦的方式协调异构模块。DART将训练系统分解为四个异步模块：环境集群、运维服务、数据管理和训练器。这种设计实现了非阻塞通信、异步训练、轨迹采样以及按工作进程同步模型，显著提高了系统效率：每轮交互的GPU利用率提高1.6倍，训练吞吐量提高1.9倍，环境利用率提高5.5倍。为了有效利用丰富的样本进行学习，我们引入了一种自适应数据整理方案：(1) 在线采样前预先收集困难任务的成功轨迹，补充稀疏的成功样本；(2) 根据任务难度动态调整轮次数量和轨迹长度；(3) 选择性地在高熵步骤上进行训练，优先处理关键决策；(4) 通过截断重要性采样来稳定学习，解决策略轮播和更新之间的不匹配问题。在OSWorld基准测试中，DART-GUI-7B实现了42.13%的任务成功率，相对于基线模型绝对提升14.61%，并且高于开源SOTA模型7.34%。我们将在以下网址全面开源我们的训练框架、数据和模型检查点，我们认为这是一项对代理强化学习训练开源社区的及时贡献。 

---
# GSID: Generative Semantic Indexing for E-Commerce Product Understanding 

**Title (ZH)**: GSID: 生成语义索引以理解电子商务产品 

**Authors**: Haiyang Yang, Qinye Xie, Qingheng Zhang, Liyu Chen, Huike Zou, Chengbao Lian, Shuguang Han, Fei Huang, Jufeng Chen, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.23860)  

**Abstract**: Structured representation of product information is a major bottleneck for the efficiency of e-commerce platforms, especially in second-hand ecommerce platforms. Currently, most product information are organized based on manually curated product categories and attributes, which often fail to adequately cover long-tail products and do not align well with buyer preference. To address these problems, we propose \textbf{G}enerative \textbf{S}emantic \textbf{I}n\textbf{D}exings (GSID), a data-driven approach to generate product structured representations. GSID consists of two key components: (1) Pre-training on unstructured product metadata to learn in-domain semantic embeddings, and (2) Generating more effective semantic codes tailored for downstream product-centric applications. Extensive experiments are conducted to validate the effectiveness of GSID, and it has been successfully deployed on the real-world e-commerce platform, achieving promising results on product understanding and other downstream tasks. 

**Abstract (ZH)**: 基于生成语义索引的电子商务产品结构化表示 

---
# Space Group Conditional Flow Matching 

**Title (ZH)**: 空间群条件流匹配 

**Authors**: Omri Puny, Yaron Lipman, Benjamin Kurt Miller  

**Link**: [PDF](https://arxiv.org/pdf/2509.23822)  

**Abstract**: Inorganic crystals are periodic, highly-symmetric arrangements of atoms in three-dimensional space. Their structures are constrained by the symmetry operations of a crystallographic \emph{space group} and restricted to lie in specific affine subspaces known as \emph{Wyckoff positions}. The frequency an atom appears in the crystal and its rough positioning are determined by its Wyckoff position. Most generative models that predict atomic coordinates overlook these symmetry constraints, leading to unrealistically high populations of proposed crystals exhibiting limited symmetry. We introduce Space Group Conditional Flow Matching, a novel generative framework that samples significantly closer to the target population of highly-symmetric, stable crystals. We achieve this by conditioning the entire generation process on a given space group and set of Wyckoff positions; specifically, we define a conditionally symmetric noise base distribution and a group-conditioned, equivariant, parametric vector field that restricts the motion of atoms to their initial Wyckoff position. Our form of group-conditioned equivariance is achieved using an efficient reformulation of \emph{group averaging} tailored for symmetric crystals. Importantly, it reduces the computational overhead of symmetrization to a negligible level. We achieve state of the art results on crystal structure prediction and de novo generation benchmarks. We also perform relevant ablations. 

**Abstract (ZH)**: 无机晶体是三维空间中具有周期性和高对称性的原子排列。它们的结构受到晶体学空间群的对称操作约束，并限定在特定的仿射子空间即沃克夫位置中。原子在晶体中的出现频率及其大致位置由其沃克夫位置决定。大多数用于预测原子坐标生成模型未考虑这些对称约束，导致生成的晶体表现出有限对称性的不切实际高比例。我们提出了一种空间群条件流匹配生成框架，该框架通过基于给定空间群和沃克夫位置对整个生成过程进行条件化，显著地接近目标群体的高对称性和稳定晶体。我们通过定义条件对称噪声基分布和群条件下的守恒参数向量场，限制原子运动到其初始沃克夫位置来实现这一点。我们形式下的群条件下的守恒性是通过针对对称晶体优化的群平均的一种高效重写实现的。重要的是，它将对称化的计算开销降低到了可以忽略的水平。我们在晶体结构预测和从头生成基准测试中达到了最先进的结果，并进行了相关的消融实验。 

---
# IndexNet: Timestamp and Variable-Aware Modeling for Time Series Forecasting 

**Title (ZH)**: IndexNet: 考虑时间戳和变量的时间序列预测建模 

**Authors**: Beiliang Wu, Peiyuan Liu, Yifan Hu, Luyan Zhang, Ao Hu, Zenglin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23813)  

**Abstract**: Multivariate time series forecasting (MTSF) plays a vital role in a wide range of real-world applications, such as weather prediction and traffic flow forecasting. Although recent advances have significantly improved the modeling of temporal dynamics and inter-variable dependencies, most existing methods overlook index-related descriptive information, such as timestamps and variable indices, which carry rich contextual semantics. To unlock the potential of such information and take advantage of the lightweight and powerful periodic capture ability of MLP-based architectures, we propose IndexNet, an MLP-based framework augmented with an Index Embedding (IE) module. The IE module consists of two key components: Timestamp Embedding (TE) and Channel Embedding (CE). Specifically, TE transforms timestamps into embedding vectors and injects them into the input sequence, thereby improving the model's ability to capture long-term complex periodic patterns. In parallel, CE assigns each variable a unique and trainable identity embedding based on its index, allowing the model to explicitly distinguish between heterogeneous variables and avoid homogenized predictions when input sequences seem close. Extensive experiments on 12 diverse real-world datasets demonstrate that IndexNet achieves comparable performance across mainstream baselines, validating the effectiveness of our temporally and variably aware design. Moreover, plug-and-play experiments and visualization analyses further reveal that IndexNet exhibits strong generality and interpretability, two aspects that remain underexplored in current MTSF research. 

**Abstract (ZH)**: 多变量时间序列 forecasting (MTSF) 在天气预测和交通流预测等广泛的实际应用中起着关键作用。尽管近年来的方法显著提高了对时间动态和变量间依赖关系的建模能力，但大多数现有方法忽略了与索引相关的描述性信息，如时间戳和变量索引，这些信息富含丰富的上下文语义。为充分利用此类信息，并利用基于MLP架构的轻量级且强大的周期捕获能力，我们提出了IndexNet，一种增强有索引嵌入 (IE) 模块的MLP框架。IE模块包含两个关键组件：时间戳嵌入 (TE) 和通道嵌入 (CE)。具体来说，TE将时间戳转换为嵌入向量并注入输入序列，从而增强模型捕捉长期复杂周期模式的能力。同时，CE根据变量索引为每个变量分配一个独特的可训练身份嵌入，使模型能够明确区分异质变量并避免在输入序列看似相近时产生同质预测。在12个多样化的实际数据集上的广泛实验表明，IndexNet在主流基线中取得了可比的性能，验证了我们具有时间和变量感知设计的有效性。此外，模块化实验和可视化分析进一步揭示了IndexNet的强大通用性和可解释性，这两个方面在当前时间序列预测研究中尚未充分探索。 

---
# From Unstable to Playable: Stabilizing Angry Birds Levels via Object Segmentation 

**Title (ZH)**: 从不稳定到可玩：通过对象分割稳定《愤怒的小鸟》关卡 

**Authors**: Mahdi Farrokhimaleki, Parsa Rahmati, Richard Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.23787)  

**Abstract**: Procedural Content Generation (PCG) techniques enable automatic creation of diverse and complex environments. While PCG facilitates more efficient content creation, ensuring consistently high-quality, industry-standard content remains a significant challenge. In this research, we propose a method to identify and repair unstable levels generated by existing PCG models. We use Angry Birds as a case study, demonstrating our method on game levels produced by established PCG approaches. Our method leverages object segmentation and visual analysis of level images to detect structural gaps and perform targeted repairs. We evaluate multiple object segmentation models and select the most effective one as the basis for our repair pipeline. Experimental results show that our method improves the stability and playability of AI-generated levels. Although our evaluation is specific to Angry Birds, our image-based approach is designed to be applicable to a wide range of 2D games with similar level structures. 

**Abstract (ZH)**: 基于图像的Procedural Content Generation模型生成不稳定关卡的识别与修复方法 

---
# GroupCoOp: Group-robust Fine-tuning via Group Prompt Learning 

**Title (ZH)**: GroupCoOp: 组群稳健微调通过组提示学习 

**Authors**: Nayeong Kim, Seong Joon Oh, Suha Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2509.23781)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) of vision-language models (VLMs) excels in various vision tasks thanks to the rich knowledge and generalization ability of VLMs. However, recent studies revealed that such fine-tuned VLMs are vulnerable to spurious correlations stemming from the subgroup imbalance in the fine-tuning datasets. To resolve this issue, we propose Group Context Optimization (GroupCoOp), a simple and effective debiased fine-tuning algorithm that enhances the group robustness of fine-tuned VLMs. Its key idea is to employ group-specific text prompts as group representatives serving as multiple classifiers for their target class. The rich semantic knowledge of the text encoder of VLM enables the discovery of effective group prompts even for groups with a small number of training samples. Leveraging the group prompts for each class addresses the issues caused by the group-imbalanced training set, such as the neglect of minority groups and the scattered distribution of each class in the embedding space. GroupCoOp achieved the best results on five benchmarks across five CLIP architectures and occasionally outperformed prior methods that fine-tune the entire network, despite training only 0.016\% of the network's parameters. 

**Abstract (ZH)**: Group Context Optimization (GroupCoOp): A Simple and Effective Debiasing Fine-Tuning Algorithm for Vision-Language Models 

---
# Accuracy-Robustness Trade Off via Spiking Neural Network Gradient Sparsity Trail 

**Title (ZH)**: 基于尖峰神经网络梯度稀疏性權衡的精度-稳健性 TRADE-OFF via Spiking Neural Network Gradient Sparsity Trail 

**Authors**: Nhan T. Luu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23762)  

**Abstract**: Spiking Neural Networks (SNNs) have attracted growing interest in both computational neuroscience and artificial intelligence, primarily due to their inherent energy efficiency and compact memory footprint. However, achieving adversarial robustness in SNNs, particularly for vision-related tasks, remains a nascent and underexplored challenge. Recent studies have proposed leveraging sparse gradients as a form of regularization to enhance robustness against adversarial perturbations. In this work, we present a surprising finding: under specific architectural configurations, SNNs exhibit natural gradient sparsity and can achieve state-of-the-art adversarial defense performance without the need for any explicit regularization. Further analysis reveals a trade-off between robustness and generalization: while sparse gradients contribute to improved adversarial resilience, they can impair the model's ability to generalize; conversely, denser gradients support better generalization but increase vulnerability to attacks. 

**Abstract (ZH)**: 脉冲神经网络（SNNs）在计算神经科学和人工智能领域引起了广泛关注，主要是由于其固有的能源效率和紧凑的内存占用。然而，特别是在视觉任务中实现对抗鲁棒性仍然是一个新兴且尚未充分探索的挑战。最近的研究提出，利用稀疏梯度作为正则化的一种形式，以增强对对抗性扰动的鲁棒性。在本工作中，我们提出一个令人惊讶的发现：在特定的架构配置下，SNNs表现出自然的梯度稀疏性，并且在不需要任何显式正则化的情况下，可以达到最先进的对抗防御性能。进一步的分析揭示了鲁棒性和泛化的权衡：虽然稀疏梯度有助于提高对抗性鲁棒性，但会影响模型的泛化能力；相反，稠密的梯度支持更好的泛化，但也增加了模型对攻击的脆弱性。 

---
# SHAPoint: Task-Agnostic, Efficient, and Interpretable Point-Based Risk Scoring via Shapley Values 

**Title (ZH)**: SHAPoint: 任务无关、高效且可解释的基于点的风险评分方法通过Shapley值 

**Authors**: Tomer D. Meirman, Bracha Shapira, Noa Dagan, Lior S. Rokach  

**Link**: [PDF](https://arxiv.org/pdf/2509.23756)  

**Abstract**: Interpretable risk scores play a vital role in clinical decision support, yet traditional methods for deriving such scores often rely on manual preprocessing, task-specific modeling, and simplified assumptions that limit their flexibility and predictive power. We present SHAPoint, a novel, task-agnostic framework that integrates the predictive accuracy of gradient boosted trees with the interpretability of point-based risk scores. SHAPoint supports classification, regression, and survival tasks, while also inheriting valuable properties from tree-based models, such as native handling of missing data and support for monotonic constraints. Compared to existing frameworks, SHAPoint offers superior flexibility, reduced reliance on manual preprocessing, and faster runtime performance. Empirical results show that SHAPoint produces compact and interpretable scores with predictive performance comparable to state-of-the-art methods, but at a fraction of the runtime, making it a powerful tool for transparent and scalable risk stratification. 

**Abstract (ZH)**: 可解释的风险评分在临床决策支持中发挥着重要作用，但传统方法常依赖于手动预处理、任务特定建模和简化假设，这限制了其灵活性和预测能力。我们提出了一种名为SHAPoint的新型、任务无关框架，该框架结合了梯度提升树的预测准确性和点基风险评分的可解释性。SHAPoint支持分类、回归和生存任务，同时继承了基于树模型的天然缺失数据处理能力和单调约束支持。与现有框架相比，SHAPoint提供了更高的灵活性、减少了对手动预处理的依赖，并具有更快的运行时性能。实证结果表明，SHAPoint产生的紧凑且具有解释性的评分在预测性能上与最先进的方法相当，但运行时间却大幅缩减，使其成为一种强大的透明且可扩展的风险分层工具。 

---
# AdaPtis: Reducing Pipeline Bubbles with Adaptive Pipeline Parallelism on Heterogeneous Models 

**Title (ZH)**: AdaPtis: 降低异构模型流水线气泡的方法基于自适应流水线并行性 

**Authors**: Jihu Guo, Tenghui Ma, Wei Gao, Peng Sun, Jiaxing Li, Xun Chen, Yuyang Jin, Dahua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.23722)  

**Abstract**: Pipeline parallelism is widely used to train large language models (LLMs). However, increasing heterogeneity in model architectures exacerbates pipeline bubbles, thereby reducing training efficiency. Existing approaches overlook the co-optimization of model partition, model placement, and workload scheduling, resulting in limited efficiency improvement or even performance degradation. To respond, we propose AdaPtis, an LLM training system that supports adaptive pipeline parallelism. First, we develop a pipeline performance model to accurately estimate training throughput. Second, AdaPtis jointly optimizes model partition, model placement, and workload scheduling policies guided by this performance model. Third, we design a unified pipeline executor that efficiently supports the execution of diverse pipeline strategies. Extensive experiments show that AdaPtis achieves an average speedup of 1.42x (up to 2.14x) over Megatron-LM I-1F1B across various LLM architectures and scales. 

**Abstract (ZH)**: AdaPtis：一种支持自适应管道并行性的大型语言模型训练系统 

---
# Bridging Discrete and Continuous RL: Stable Deterministic Policy Gradient with Martingale Characterization 

**Title (ZH)**: 离散与连续RL的桥梁：具有鞅特征的稳定确定性策略梯度 

**Authors**: Ziheng Cheng, Xin Guo, Yufei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23711)  

**Abstract**: The theory of discrete-time reinforcement learning (RL) has advanced rapidly over the past decades. Although primarily designed for discrete environments, many real-world RL applications are inherently continuous and complex. A major challenge in extending discrete-time algorithms to continuous-time settings is their sensitivity to time discretization, often leading to poor stability and slow convergence. In this paper, we investigate deterministic policy gradient methods for continuous-time RL. We derive a continuous-time policy gradient formula based on an analogue of the advantage function and establish its martingale characterization. This theoretical foundation leads to our proposed algorithm, CT-DDPG, which enables stable learning with deterministic policies in continuous-time environments. Numerical experiments show that the proposed CT-DDPG algorithm offers improved stability and faster convergence compared to existing discrete-time and continuous-time methods, across a wide range of control tasks with varying time discretizations and noise levels. 

**Abstract (ZH)**: 连续时间强化学习中的确定性策略梯度方法 

---
# Estimating Time Series Foundation Model Transferability via In-Context Learning 

**Title (ZH)**: 基于上下文学习的时间序列基础模型迁移性估测 

**Authors**: Qingren Yao, Ming Jin, Chengqi Zhang, Chao-Han Huck Yang, Jun Qi, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2509.23695)  

**Abstract**: Time series foundation models (TSFMs) offer strong zero-shot forecasting via large-scale pre-training, yet fine-tuning remains critical for boosting performance in domains with limited public data. With the growing number of TSFMs, efficiently identifying the best model for downstream fine-tuning becomes increasingly challenging. In this work, we introduce TimeTic, a transferability estimation framework that recasts model selection as an in-context-learning problem: given observations on known (source) datasets, it predicts how a TSFM will perform after fine-tuning on a downstream (target) dataset. TimeTic flexibly organizes the observed model-data relationships as contextual information, allowing it to adapt seamlessly to various test-time scenarios. Leveraging the natural tabular structure formed by dataset meta-features, model characteristics, and fine-tuned performance, we employ tabular foundation models to serve as in-context learners. We further introduce a novel model characterization based on entropy evolution across model layers, capturing embedding-space distinctions and enabling TimeTic to generalize across arbitrary model sets. We establish a comprehensive benchmark for transferability estimation including 10 datasets, 10 foundation models, and 3 forecasting tasks. On this benchmark, TimeTic's estimation demonstrates strong alignment with actual fine-tuned performance for previously unseen datasets, achieving a mean rank correlation of approximately 0.6 and a 30% improvement compared to using zero-shot performance as the transferability score. 

**Abstract (ZH)**: TimeTic：一种时间序列基础模型迁移性估计框架 

---
# Joint Hybrid Beamforming and Artificial Noise Design for Secure Multi-UAV ISAC Networks 

**Title (ZH)**: 联合混合波束形成与人工噪声设计以实现安全的多无人机异构接入网络 

**Authors**: Runze Dong, Buhong Wang, Cunqian Feng, Jiang Weng, Chen Han, Jiwei Tian  

**Link**: [PDF](https://arxiv.org/pdf/2509.23687)  

**Abstract**: Integrated sensing and communication (ISAC) emerges as a key enabler for next-generation applications such as smart cities and autonomous systems. Its integration with unmanned aerial vehicles (UAVs) unlocks new potentials for reliable communication and precise sensing in dynamic aerial environments. However, existing research predominantly treats UAVs as aerial base stations, overlooking their role as ISAC users, and fails to leverage large-scale antenna arrays at terrestrial base stations to enhance security and spectral efficiency. This paper propose a secure and spectral efficient ISAC framework for multi-UAV networks, and a two-stage optimization approach is developed to jointly design hybrid beamforming (HBF), artificial noise (AN) injection, and UAV trajectories. Aiming at maximizing the sum secrecy rate, the first stage employs Proximal Policy Optimization (PPO) to optimize digital beamformers and trajectories, and the second stage decomposes the digital solution into analog and digital components via low-complexity matrix factorization. Simulation results demonstrate the effectiveness of the proposed framework compared to benchmark schemes. 

**Abstract (ZH)**: 集成传感与通信(ISAC)技术成为智能城市和自主系统等下一代应用的关键使能器。将其与无人驾驶飞行器(UAVs)结合，为动态高空环境下的可靠通信和精确传感开启了新潜能。然而，现有研究主要将UAVs视为高空基站，忽视了其作为ISAC用户的角色，未能充分利用地面基站的大规模天线阵列以提升安全性和频谱效率。本文提出一种适用于多UAV网络的保密性和频谱效率兼备的ISAC框架，并开发了一种两阶段优化方法，以联合设计混合波束形成(HBF)、人工噪声(AN)注入和UAV航迹。为最大化总保密率，第一阶段采用近端策略优化(Proximal Policy Optimization, PPO)优化数字波束形成器和航迹，第二阶段通过低复杂度矩阵分解将数字解决方案分解为模拟和数字组件。仿真结果表明，所提出框架的有效性优于基准方案。 

---
# Graph Neural Networks with Diversity-aware Neighbor Selection and Dynamic Multi-scale Fusion for Multivariate Time Series Forecasting 

**Title (ZH)**: 具有多样性意识的邻域选择和动态多尺度融合的图神经网络在多变量时间序列预测中的应用 

**Authors**: Jingqi Xu, Guibin Chen, Jingxi Lu, Yuzhang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.23671)  

**Abstract**: Recently, numerous deep models have been proposed to enhance the performance of multivariate time series (MTS) forecasting. Among them, Graph Neural Networks (GNNs)-based methods have shown great potential due to their capability to explicitly model inter-variable dependencies. However, these methods often overlook the diversity of information among neighbors, which may lead to redundant information aggregation. In addition, their final prediction typically relies solely on the representation from a single temporal scale. To tackle these issues, we propose a Graph Neural Networks (GNNs) with Diversity-aware Neighbor Selection and Dynamic Multi-scale Fusion (DIMIGNN). DIMIGNN introduces a Diversity-aware Neighbor Selection Mechanism (DNSM) to ensure that each variable shares high informational similarity with its neighbors while maintaining diversity among neighbors themselves. Furthermore, a Dynamic Multi-Scale Fusion Module (DMFM) is introduced to dynamically adjust the contributions of prediction results from different temporal scales to the final forecasting result. Extensive experiments on real-world datasets demonstrate that DIMIGNN consistently outperforms prior methods. 

**Abstract (ZH)**: 基于多样性aware邻居选择和动态多尺度融合的图神经网络（DIMIGNN）及其在多变量时间序列预测中的应用 

---
# Beyond Greedy Exits: Improved Early Exit Decisions for Risk Control and Reliability 

**Title (ZH)**: 超越贪婪退出：改进的风险控制和可靠性早期退出决策 

**Authors**: Divya Jyoti Bajpai, Manjesh Kumar Hanawal  

**Link**: [PDF](https://arxiv.org/pdf/2509.23666)  

**Abstract**: Early-Exit Deep Neural Networks enable adaptive inference by allowing prediction at intermediary layers, significantly reducing computational costs and latency. Most of the early exit strategies greedily exit a sample at an intermediary layer if the confidence in class prediction exceeds a predefined threshold that is set using a static validation set. This is problematic as the model might be overconfident in a wrong class. Also, they are not robust to distribution shifts encountered in deployment, which can undermine model trustworthiness and accuracy. To address these challenges, we propose UAT that adapts the threshold for exit decisions using a Multi-Armed Bandit framework, enabling online, unsupervised adjustment of exit decisions. UAT makes decisions based on a new reward function that assesses predictive certainty and its reliability to balance computational efficiency and prediction quality while penalizing unnecessary late exits. We provide guarantees on risk achieved by UAT and validate its performance on diverse tasks spanning vision-language understanding, text generation, and classification. Our framework demonstrates consistent improvements in speedup (1.70-2.10x) with a minimal performance drop (<2%) as compared to full model performance. Our source code is available at this https URL. 

**Abstract (ZH)**: 基于多臂老虎机框架的自适应阈值退出机制使早退出深度神经网络能够在中间层进行预测，显著降低计算成本和延迟并实现适应性推理。现有的大多数早退出策略在类预测置信度超过预设阈值时贪婪地在中间层退出样本，但这种方法可能导致模型过自信于错误的类别，并在部署时遇到分布偏移时缺乏鲁棒性，从而影响模型的信任和准确性。为解决这些问题，我们提出了一种基于多臂老虎机框架的自适应阈值（UAT）机制，能够在无需监督的情况下在线调整退出决策，并基于新的奖励函数评估预测的确定性和可靠性来平衡计算效率和预测质量，同时惩罚不必要的延迟退出。我们提供了UAT实现的风险保证，并在视觉-语言理解、文本生成和分类等多样任务中验证了其性能。与全模型相比，该框架在保证性能损失小于2%的情况下，实现了1.70-2.10倍的加速。源代码可访问此链接。 

---
# Calibration Meets Reality: Making Machine Learning Predictions Trustworthy 

**Title (ZH)**: 校准遇见现实：使机器学习预测值得信赖 

**Authors**: Kristina P. Sinaga, Arjun S. Nair  

**Link**: [PDF](https://arxiv.org/pdf/2509.23665)  

**Abstract**: Post-hoc calibration methods are widely used to improve the reliability of probabilistic predictions from machine learning models. Despite their prevalence, a comprehensive theoretical understanding of these methods remains elusive, particularly regarding their performance across different datasets and model architectures. Input features play a crucial role in shaping model predictions and, consequently, their calibration. However, the interplay between feature quality and calibration performance has not been thoroughly investigated. In this work, we present a rigorous theoretical analysis of post-hoc calibration methods, focusing on Platt scaling and isotonic regression. We derive convergence guarantees, computational complexity bounds, and finite-sample performance metrics for these methods. Furthermore, we explore the impact of feature informativeness on calibration performance through controlled synthetic experiments. Our empirical evaluation spans a diverse set of real-world datasets and model architectures, demonstrating consistent improvements in calibration metrics across various scenarios. By examining calibration performance under varying feature conditions utilizing only informative features versus complete feature spaces including noise dimensions, we provide fundamental insights into the robustness and reliability of different calibration approaches. Our findings offer practical guidelines for selecting appropriate calibration methods based on dataset characteristics and computational constraints, bridging the gap between theoretical understanding and practical implementation in uncertainty quantification. Code and experimental data are available at: this https URL. 

**Abstract (ZH)**: 事后校准方法广泛用于提高机器学习模型的概率预测可靠性。尽管这些方法被广泛应用，但对其在不同数据集和模型架构上的表现的全面理论理解仍缺乏，特别是关于特征质量与校准性能之间的关系。输入特征在塑造模型预测和校准方面起着关键作用，但特征质量和校准性能之间的相互作用尚未得到充分研究。在本文中，我们对事后校准方法进行了严格的理论分析，集中于Platt校准和等距回归。我们推导了这些方法的收敛保证、计算复杂度边界和有限样本性能度量。此外，我们通过受控的合成实验探讨了特征信息量对校准性能的影响。我们的实证评估涵盖了多种真实世界的数据集和模型架构，展示了在各种场景下校准指标的一致性改进。通过在仅使用信息特征与包含噪声维度的完整特征空间下考察不同校准方法的校准性能，我们提供了关于不同校准方法的稳健性和可靠性的重要见解。我们的发现为基于数据集特性和计算约束选择合适的校准方法提供了实用指南，填补了理论理解和实际实施在不确定性量化中的差距。代码和实验数据可在：this https URL 获取。 

---
# Pure Node Selection for Imbalanced Graph Node Classification 

**Title (ZH)**: 无偏节点选择的图节点分类 

**Authors**: Fanlong Zeng, Wensheng Gan, Jiayang Wu, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23662)  

**Abstract**: The problem of class imbalance refers to an uneven distribution of quantity among classes in a dataset, where some classes are significantly underrepresented compared to others. Class imbalance is also prevalent in graph-structured data. Graph neural networks (GNNs) are typically based on the assumption of class balance, often overlooking the issue of class imbalance. In our investigation, we identified a problem, which we term the Randomness Anomalous Connectivity Problem (RACP), where certain off-the-shelf models are affected by random seeds, leading to a significant performance degradation. To eliminate the influence of random factors in algorithms, we proposed PNS (Pure Node Sampling) to address the RACP in the node synthesis stage. Unlike existing approaches that design specialized algorithms to handle either quantity imbalance or topological imbalance, PNS is a novel plug-and-play module that operates directly during node synthesis to mitigate RACP. Moreover, PNS also alleviates performance degradation caused by abnormal distribution of node neighbors. We conduct a series of experiments to identify what factors are influenced by random seeds. Experimental results demonstrate the effectiveness and stability of our method, which not only eliminates the effect of unfavorable random seeds but also outperforms the baseline across various benchmark datasets with different GNN backbones. Data and code are available at this https URL. 

**Abstract (ZH)**: 类别不平衡问题指的是数据集中各类别数量分布不均，其中某些类别相较于其他类别显著欠代表。类别不平衡问题在图结构数据中也很常见。图神经网络（GNNs）通常基于类别平衡的假设，常忽视类别不平衡的问题。在我们的研究中，我们发现了一个问题，称之为随机异常连接问题（RACP），某些现成模型受随机种子影响，导致显著性能下降。为消除算法中随机因素的影响，我们提出了PNS（纯节点采样）来解决RACP问题。PNS不同于现有的专为处理数量不平衡或拓扑不平衡设计的算法，它是一个新颖的即插即用模块，在节点合成阶段直接运行以缓解RACP。此外，PNS还能缓解由于节点邻居异常分布导致的性能下降。我们进行了一系列实验以确定哪些因素受随机种子影响。实验结果证明了我们方法的有效性和稳定性，不仅消除了不利随机种子的影响，还在不同GNN后端的不同基准数据集上优于基线方法。数据和代码可在以下链接获取。 

---
# LightFair: Towards an Efficient Alternative for Fair T2I Diffusion via Debiasing Pre-trained Text Encoders 

**Title (ZH)**: LightFair: 向高效公平的文本到图像扩散转化的去偏见预训练文本编码器替代方案 

**Authors**: Boyu Han, Qianqian Xu, Shilong Bao, Zhiyong Yang, Kangli Zi, Qingming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23639)  

**Abstract**: This paper explores a novel lightweight approach LightFair to achieve fair text-to-image diffusion models (T2I DMs) by addressing the adverse effects of the text encoder. Most existing methods either couple different parts of the diffusion model for full-parameter training or rely on auxiliary networks for correction. They incur heavy training or sampling burden and unsatisfactory performance. Since T2I DMs consist of multiple components, with the text encoder being the most fine-tunable and front-end module, this paper focuses on mitigating bias by fine-tuning text embeddings. To validate feasibility, we observe that the text encoder's neutral embedding output shows substantial skewness across image embeddings of various attributes in the CLIP space. More importantly, the noise prediction network further amplifies this imbalance. To finetune the text embedding, we propose a collaborative distance-constrained debiasing strategy that balances embedding distances to improve fairness without auxiliary references. However, mitigating bias can compromise the original generation quality. To address this, we introduce a two-stage text-guided sampling strategy to limit when the debiased text encoder intervenes. Extensive experiments demonstrate that LightFair is effective and efficient. Notably, on Stable Diffusion v1.5, our method achieves SOTA debiasing at just $1/4$ of the training burden, with virtually no increase in sampling burden. The code is available at this https URL. 

**Abstract (ZH)**: 一种新型轻量级方法LightFair实现公平的文字到图像扩散模型 

---
# Generalizable Speech Deepfake Detection via Information Bottleneck Enhanced Adversarial Alignment 

**Title (ZH)**: 基于信息瓶颈增强对抗对齐的通用语音深度假声检测 

**Authors**: Pu Huang, Shouguang Wang, Siya Yao, Mengchu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.23618)  

**Abstract**: Neural speech synthesis techniques have enabled highly realistic speech deepfakes, posing major security risks. Speech deepfake detection is challenging due to distribution shifts across spoofing methods and variability in speakers, channels, and recording conditions. We explore learning shared discriminative features as a path to robust detection and propose Information Bottleneck enhanced Confidence-Aware Adversarial Network (IB-CAAN). Confidence-guided adversarial alignment adaptively suppresses attack-specific artifacts without erasing discriminative cues, while the information bottleneck removes nuisance variability to preserve transferable features. Experiments on ASVspoof 2019/2021, ASVspoof 5, and In-the-Wild demonstrate that IB-CAAN consistently outperforms baseline and achieves state-of-the-art performance on many benchmarks. 

**Abstract (ZH)**: 神经语音合成技术催生了高度逼真的语音深伪，引发了重大安全风险。由于欺骗方法、说话人、信道和录音条件的分布变化，语音深伪检测具有挑战性。我们探索学习共享鉴别特征作为稳健检测的方法，并提出信息瓶颈增强的置信感知对抗网络（IB-CAAN）。置信导向的对抗对齐自适应地抑制攻击特定的-artifacts，同时不抹除鉴别线索，信息瓶颈去除无关变异，保留可迁移特征。实验表明，IB-CAAN 在 ASVspoof 2019/2021、ASVspoof 5 和在野数据上的表现均优于基线方法，并在许多基准测试中达到最佳性能。 

---
# GraphIFE: Rethinking Graph Imbalance Node Classification via Invariant Learning 

**Title (ZH)**: GraphIFE: 通过不变学习重新思考图的不均衡节点分类 

**Authors**: Fanlong Zeng, Wensheng Gan, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23616)  

**Abstract**: The class imbalance problem refers to the disproportionate distribution of samples across different classes within a dataset, where the minority classes are significantly underrepresented. This issue is also prevalent in graph-structured data. Most graph neural networks (GNNs) implicitly assume a balanced class distribution and therefore often fail to account for the challenges introduced by class imbalance, which can lead to biased learning and degraded performance on minority classes. We identify a quality inconsistency problem in synthesized nodes, which leads to suboptimal performance under graph imbalance conditions. To mitigate this issue, we propose GraphIFE (Graph Invariant Feature Extraction), a novel framework designed to mitigate quality inconsistency in synthesized nodes. Our approach incorporates two key concepts from graph invariant learning and introduces strategies to strengthen the embedding space representation, thereby enhancing the model's ability to identify invariant features. Extensive experiments demonstrate the framework's efficiency and robust generalization, as GraphIFE consistently outperforms various baselines across multiple datasets. The code is publicly available at this https URL. 

**Abstract (ZH)**: 类别不平衡问题指的是数据集中不同类别样本分布不均，其中少数类显著欠代表性。这一问题在图结构数据中也非常普遍。大多数图神经网络（GNNs）隐含地假设类别分布均衡，因此往往未能充分考虑类别不平衡带来的挑战，这可能导致对少数类的学习偏差和性能退化。我们识别出合成节点质量不一致性问题，这在图结构不平衡条件下会导致次优性能。为缓解这一问题，我们提出了一种新颖的框架GraphIFE（图不变特征提取），旨在缓解合成节点的质量不一致性。我们的方法结合了图不变学习中的两个关键概念，并引入策略以增强嵌入空间表示，从而提高模型识别不变特征的能力。广泛的实验展示了该框架的效率和稳健的泛化能力，GraphIFE在多个数据集中均优于各种基线。代码可在以下网址获取。 

---
# Characteristic Root Analysis and Regularization for Linear Time Series Forecasting 

**Title (ZH)**: 线性时间序列预测中的特征根分析与正则化 

**Authors**: Zheng Wang, Kaixuan Zhang, Wanfang Chen, Xiaonan Lu, Longyuan Li, Tobias Schlagenhauf  

**Link**: [PDF](https://arxiv.org/pdf/2509.23597)  

**Abstract**: Time series forecasting remains a critical challenge across numerous domains, yet the effectiveness of complex models often varies unpredictably across datasets. Recent studies highlight the surprising competitiveness of simple linear models, suggesting that their robustness and interpretability warrant deeper theoretical investigation. This paper presents a systematic study of linear models for time series forecasting, with a focus on the role of characteristic roots in temporal dynamics. We begin by analyzing the noise-free setting, where we show that characteristic roots govern long-term behavior and explain how design choices such as instance normalization and channel independence affect model capabilities. We then extend our analysis to the noisy regime, revealing that models tend to produce spurious roots. This leads to the identification of a key data-scaling property: mitigating the influence of noise requires disproportionately large training data, highlighting the need for structural regularization. To address these challenges, we propose two complementary strategies for robust root restructuring. The first uses rank reduction techniques, including Reduced-Rank Regression and Direct Weight Rank Reduction, to recover the low-dimensional latent dynamics. The second, a novel adaptive method called Root Purge, encourages the model to learn a noise-suppressing null space during training. Extensive experiments on standard benchmarks demonstrate the effectiveness of both approaches, validating our theoretical insights and achieving state-of-the-art results in several settings. Our findings underscore the potential of integrating classical theories for linear systems with modern learning techniques to build robust, interpretable, and data-efficient forecasting models. 

**Abstract (ZH)**: 时间序列预测仍然是诸多领域中的一个关键挑战，但复杂模型在不同数据集上的有效性往往不可预测。近期研究表明，简单的线性模型表现出令人惊讶的竞争性，这表明其鲁棒性和可解释性应进行更深入的理论探讨。本文对线性模型在时间序列预测中的应用进行了系统研究，重点关注特征根在时序动态中的作用。我们首先分析了无噪声的情境，证明了特征根决定了长期行为，并解释了诸如实例归一化和通道独立性等设计选择如何影响模型能力。随后，我们将分析扩展到了有噪声的情境，揭示模型倾向于生成虚假根。这导致我们识别出一个关键的数据缩放特性：减轻噪声影响需要不成比例的大量训练数据，突显了结构正则化的需求。为了应对这些挑战，我们提出了两种互补的稳健特征根重构策略。第一种使用秩约简技术，包括降秩回归和直接权重秩约简，以恢复低维潜在动力。第二种是新颖的自适应方法根净化（Root Purge），鼓励模型在训练期间学习一个抑制噪声的零空间。在标准基准上的详尽实验表明，这两种方法都证明了其有效性，验证了我们的理论洞见，并在某些场景下达到了最先进的结果。我们的研究结果强调了将经典线性系统理论与现代学习技术结合起来以构建稳健、可解释和数据高效的预测模型的潜力。 

---
# Multi-Level Heterogeneous Knowledge Transfer Network on Forward Scattering Center Model for Limited Samples SAR ATR 

**Title (ZH)**: 多层次异质知识转移网络在前散射中心模型下的限样雷达瞄准识别 

**Authors**: Chenxi Zhao, Daochang Wang, Siqian Zhang, Gangyao Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23596)  

**Abstract**: Simulated data-assisted SAR target recognition methods are the research hotspot currently, devoted to solving the problem of limited samples. Existing works revolve around simulated images, but the large amount of irrelevant information embedded in the images, such as background, noise, etc., seriously affects the quality of the migrated information. Our work explores a new simulated data to migrate purer and key target knowledge, i.e., forward scattering center model (FSCM) which models the actual local structure of the target with strong physical meaning and interpretability. To achieve this purpose, multi-level heterogeneous knowledge transfer (MHKT) network is proposed, which fully migrates FSCM knowledge from the feature, distribution and category levels, respectively. Specifically, we permit the more suitable feature representations for the heterogeneous data and separate non-informative knowledge by task-associated information selector (TAIS), to complete purer target feature migration. In the distribution alignment, the new metric function maximum discrimination divergence (MDD) in target generic knowledge transfer (TGKT) module perceives transferable knowledge efficiently while preserving discriminative structure about classes. Moreover, category relation knowledge transfer (CRKT) module leverages the category relation consistency constraint to break the dilemma of optimization bias towards simulation data due to imbalance between simulated and measured data. Such stepwise knowledge selection and migration will ensure the integrity of the migrated FSCM knowledge. Notably, extensive experiments on two new datasets formed by FSCM data and measured SAR images demonstrate the superior performance of our method. 

**Abstract (ZH)**: 基于模拟数据辅助的SAR目标识别方法：探索纯净目标知识迁移的新途径 

---
# Toward a Holistic Approach to Continual Model Merging 

**Title (ZH)**: 走向综合性的持续模型合并方法 

**Authors**: Hoang Phan, Sungmin Cha, Tung Lam Tran, Qi Lei  

**Link**: [PDF](https://arxiv.org/pdf/2509.23592)  

**Abstract**: We present a holistic framework for continual model merging that intervenes at three critical stages: pre-merging, during merging, and post-merging-to address two fundamental challenges in continual learning. In particular, conventional approaches either maintain a growing list of per-domain task vectors, leading to scalability issues or rely solely on weight-space merging when old data is inaccessible, thereby losing crucial functional information. Our method overcomes these limitations by first fine-tuning the main model within its tangent space on domain-specific data; this linearization amplifies per-task weight disentanglement, effectively mitigating across-task interference. During merging, we leverage functional information from available optimizer states beyond mere parameter averages to avoid the need to revisit old data. Finally, a post-merging correction aligns the representation discrepancy between pre- and post-merged models, reducing bias and enhancing overall performance-all while operating under constant memory constraints without accessing historical data. Extensive experiments on standard class-incremental and domain-incremental benchmarks demonstrate that our approach not only achieves competitive performance but also provides a scalable and efficient solution to the catastrophic forgetting problem. 

**Abstract (ZH)**: 我们提出了一种面向持续学习的集成模型全面框架，该框架干预了合并前三 Critial 阶段：合并前、合并中和合并后，以应对持续学习中的两大根本挑战。具体而言，传统方法要么维护一个不断增长的领域特定任务向量列表，导致可扩展性问题，要么仅依赖于权重空间的合并，当旧数据不可访问时丢失重要的功能信息。我们的方法通过首先在领域特定数据上将主模型在其切线空间内进行微调，从而克服了这些限制；这种线性化增强了任务间权重的分离，有效地降低了跨任务干扰。在合并过程中，我们利用可用优化器状态中的功能信息，而不仅仅是参数平均值，以避免重新访问旧数据。最后，在合并后，通过校准合并前和合并后模型之间的表示差异，减少偏差并提高整体性能，同时在不访问历史数据的情况下保持恒定的内存约束。在标准的类增量和领域增量基准测试中的广泛实验表明，我们的方法不仅实现了竞争力的表现，还提供了一种可扩展且高效的解决灾难性遗忘问题的解决方案。 

---
# ML-Asset Management: Curation, Discovery, and Utilization 

**Title (ZH)**: ML-资产管理系统：编目、发现与利用 

**Authors**: Mengying Wang, Moming Duan, Yicong Huang, Chen Li, Bingsheng He, Yinghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23577)  

**Abstract**: Machine learning (ML) assets, such as models, datasets, and metadata, are central to modern ML workflows. Despite their explosive growth in practice, these assets are often underutilized due to fragmented documentation, siloed storage, inconsistent licensing, and lack of unified discovery mechanisms, making ML-asset management an urgent challenge. This tutorial offers a comprehensive overview of ML-asset management activities across its lifecycle, including curation, discovery, and utilization. We provide a categorization of ML assets, and major management issues, survey state-of-the-art techniques, and identify emerging opportunities at each stage. We further highlight system-level challenges related to scalability, lineage, and unified indexing. Through live demonstrations of systems, this tutorial equips both researchers and practitioners with actionable insights and practical tools for advancing ML-asset management in real-world and domain-specific settings. 

**Abstract (ZH)**: 机器学习资产管理：从收集、发现到利用的全面概述 

---
# Node Classification via Simplicial Interaction with Augmented Maximal Clique Selection 

**Title (ZH)**: 基于增强最大闭包的选择的简化体交互节点分类 

**Authors**: Eunho Koo, Tongseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2509.23568)  

**Abstract**: Considering higher-order interactions allows for a more comprehensive understanding of network structures beyond simple pairwise connections. While leveraging all cliques in a network to handle higher-order interactions is intuitive, it often leads to computational inefficiencies due to overlapping information between higher-order and lower-order cliques. To address this issue, we propose an augmented maximal clique strategy. Although using only maximal cliques can reduce unnecessary overlap and provide a concise representation of the network, certain nodes may still appear in multiple maximal cliques, resulting in imbalanced training data. Therefore, our augmented maximal clique approach selectively includes some non-maximal cliques to mitigate the overrepresentation of specific nodes and promote more balanced learning across the network. Comparative analyses on synthetic networks and real-world citation datasets demonstrate that our method outperforms approaches based on pairwise interactions, all cliques, or only maximal cliques. Finally, by integrating this strategy into GNN-based semi-supervised learning, we establish a link between maximal clique-based methods and GNNs, showing that incorporating higher-order structures improves predictive accuracy. As a result, the augmented maximal clique strategy offers a computationally efficient and effective solution for higher-order network learning. 

**Abstract (ZH)**: 考虑高阶交互关系有助于超越简单二元连接，更全面地理解网络结构。尽管利用网络中的所有团来处理高阶交互直观易行，但由于高阶和低阶团之间存在重叠信息，往往会引发计算效率低下问题。为解决此问题，我们提出了一种增强最大团策略。虽然仅使用最大团可以减少不必要的重叠并提供网络的简洁表示，但某些节点仍然可能出现在多个最大团中，导致训练数据不平衡。因此，我们的增强最大团方法有选择地包括一些非最大团，以减轻特定节点的过度代表，促进网络更均衡的学习。在合成网络和实际引用数据集上的对比分析表明，我们的方法优于基于二元交互、所有团或仅最大团的方法。最后，通过将此策略整合到基于图神经网络的半监督学习中，我们建立了基于最大团方法与图神经网络之间的联系，证明引入高阶结构可以提高预测准确性。因此，增强最大团策略提供了一种计算上高效且有效的高阶网络学习解决方案。 

---
# Pancreas Part Segmentation under Federated Learning Paradigm 

**Title (ZH)**: 联邦学习范式下的胰腺部分分割 

**Authors**: Ziliang Hong, Halil Ertugrul Aktas, Andrea Mia Bejar, Katherine Wu, Hongyi Pan, Gorkem Durak, Zheyuan Zhang, Sait Kayali, Temel Tirkes, Federica Proietto Salanitri, Concetto Spampinato, Michael Goggins, Tamas Gonda, Candice Bolan, Raj Keswani, Frank Miller, Michael Wallace, Ulas Bagci  

**Link**: [PDF](https://arxiv.org/pdf/2509.23562)  

**Abstract**: We present the first federated learning (FL) approach for pancreas part(head, body and tail) segmentation in MRI, addressing a critical clinical challenge as a significant innovation. Pancreatic diseases exhibit marked regional heterogeneity cancers predominantly occur in the head region while chronic pancreatitis causes tissue loss in the tail, making accurate segmentation of the organ into head, body, and tail regions essential for precise diagnosis and treatment planning. This segmentation task remains exceptionally challenging in MRI due to variable morphology, poor soft-tissue contrast, and anatomical variations across patients. Our novel contribution tackles two fundamental challenges: first, the technical complexity of pancreas part delineation in MRI, and second the data scarcity problem that has hindered prior approaches. We introduce a privacy-preserving FL framework that enables collaborative model training across seven medical institutions without direct data sharing, leveraging a diverse dataset of 711 T1W and 726 T2W MRI scans. Our key innovations include: (1) a systematic evaluation of three state-of-the-art segmentation architectures (U-Net, Attention U-Net,Swin UNETR) paired with two FL algorithms (FedAvg, FedProx), revealing Attention U-Net with FedAvg as optimal for pancreatic heterogeneity, which was never been done before; (2) a novel anatomically-informed loss function prioritizing region-specific texture contrasts in MRI. Comprehensive evaluation demonstrates that our approach achieves clinically viable performance despite training on distributed, heterogeneous datasets. 

**Abstract (ZH)**: 我们介绍了首个用于MRI中胰腺部分（头、体、尾）分割的联邦学习方法，解决了临床中的一个关键挑战，是一项重要的创新。 

---
# Fusing Sequence Motifs and Pan-Genomic Features: Antimicrobial Resistance Prediction using an Explainable Lightweight 1D CNN-XGBoost Ensemble 

**Title (ZH)**: 融合序列motif和泛基因组特征：基于可解释轻量级1D CNN-XGBoost集成的抗菌药物耐药性预测 

**Authors**: Md. Saiful Bari Siddiqui, Nowshin Tarannum  

**Link**: [PDF](https://arxiv.org/pdf/2509.23552)  

**Abstract**: Antimicrobial Resistance (AMR) is a rapidly escalating global health crisis. While genomic sequencing enables rapid prediction of resistance phenotypes, current computational methods have limitations. Standard machine learning models treat the genome as an unordered collection of features, ignoring the sequential context of Single Nucleotide Polymorphisms (SNPs). State-of-the-art sequence models like Transformers are often too data-hungry and computationally expensive for the moderately-sized datasets that are typical in this domain. To address these challenges, we propose AMR-EnsembleNet, an ensemble framework that synergistically combines sequence-based and feature-based learning. We developed a lightweight, custom 1D Convolutional Neural Network (CNN) to efficiently learn predictive sequence motifs from high-dimensional SNP data. This sequence-aware model was ensembled with an XGBoost model, a powerful gradient boosting system adept at capturing complex, non-local feature interactions. We trained and evaluated our framework on a benchmark dataset of 809 E. coli strains, predicting resistance across four antibiotics with varying class imbalance. Our 1D CNN-XGBoost ensemble consistently achieved top-tier performance across all the antibiotics, reaching a Matthews Correlation Coefficient (MCC) of 0.926 for Ciprofloxacin (CIP) and the highest Macro F1-score of 0.691 for the challenging Gentamicin (GEN) AMR prediction. We also show that our model consistently focuses on SNPs within well-known AMR genes like fusA and parC, confirming it learns the correct genetic signals for resistance. Our work demonstrates that fusing a sequence-aware 1D CNN with a feature-based XGBoost model creates a powerful ensemble, overcoming the limitations of using either an order-agnostic or a standalone sequence model. 

**Abstract (ZH)**: 抗微生物耐药性（AMR）是亟待应对的全球健康危机。基因组测序能够实现快速预测耐药表型，但现有的计算方法存在局限性。标准机器学习模型将基因组视为无序的特征集合，忽略了单核苷酸多态性（SNPs）的序列上下文。最先进的序列模型如变换器通常因数据需求大且计算成本高而难以应用于该领域典型的中等规模数据集。为了应对这些挑战，我们提出AMR-EnsembleNet，一种结合序列基础学习和特征基础学习的集成框架。我们开发了一个轻量级的自定义1D卷积神经网络（CNN），能够高效地从高维SNP数据中学习预测性序列模式。该序列感知模型与强大的梯度提升系统XGBoost模型进行集成，后者擅长捕捉复杂且非局部的特征交互。我们在包含809株大肠杆菌的标准数据集上训练和评估了我们的框架，针对四类抗生素下的不同类别不平衡进行耐药预测。我们的1D CNN-XGBoost集成框架在所有抗生素上实现了最优性能，CIP的马修相关系数（MCC）达到0.926，GEN的宏F1分数达到0.691，这是最具挑战性的耐药预测。我们还展示了模型始终关注诸如fusA和parC等已知抗药基因中的SNP，证实了它学习到正确的遗传信号。我们的工作证明，将序列感知的1D CNN与特征基础的XGBoost模型融合形成强大的集成框架，能够克服单独使用无序感知模型或仅序列模型的局限性。 

---
# Automatic Speech Recognition for Greek Medical Dictation 

**Title (ZH)**: 希腊医疗口述的自动语音识别 

**Authors**: Vardis Georgilas, Themos Stafylakis  

**Link**: [PDF](https://arxiv.org/pdf/2509.23550)  

**Abstract**: Medical dictation systems are essential tools in modern healthcare, enabling accurate and efficient conversion of speech into written medical documentation. The main objective of this paper is to create a domain-specific system for Greek medical speech transcriptions. The ultimate goal is to assist healthcare professionals by reducing the overload of manual documentation and improving workflow efficiency. Towards this goal, we develop a system that combines automatic speech recognition techniques with text correction model, allowing better handling of domain-specific terminology and linguistic variations in Greek. Our approach leverages both acoustic and textual modeling to create more realistic and reliable transcriptions. We focused on adapting existing language and speech technologies to the Greek medical context, addressing challenges such as complex medical terminology and linguistic inconsistencies. Through domain-specific fine-tuning, our system achieves more accurate and coherent transcriptions, contributing to the development of practical language technologies for the Greek healthcare sector. 

**Abstract (ZH)**: 医学口述系统是现代医疗保健中不可或缺的工具，能够实现语音到书面医疗文档的准确高效转换。本文的主要目标是为希腊医学语音转录创建一个专用系统。最终目标是通过减轻手写文档的负担并提高工作流程效率来辅助医疗专业人员。为了实现这一目标，我们开发了一个结合自动语音识别技术和文本校正模型的系统，以更好地处理希腊医学领域的专有名词和语言变异。我们的方法结合了声学和文本建模，以生成更加真实可靠的转录。我们专注于将现有的语言和技术适应希腊医疗背景，解决复杂医学术语和语言不一致等挑战。通过领域特定的微调，我们的系统实现了更准确和连贯的转录，为希腊医疗保健领域的发展贡献了实用的语言技术。 

---
# End-to-End Deep Learning for Predicting Metric Space-Valued Outputs 

**Title (ZH)**: 端到端深度学习用于预测度量空间值输出 

**Authors**: Yidong Zhou, Su I Iao, Hans-Georg Müller  

**Link**: [PDF](https://arxiv.org/pdf/2509.23544)  

**Abstract**: Many modern applications involve predicting structured, non-Euclidean outputs such as probability distributions, networks, and symmetric positive-definite matrices. These outputs are naturally modeled as elements of general metric spaces, where classical regression techniques that rely on vector space structure no longer apply. We introduce E2M (End-to-End Metric regression), a deep learning framework for predicting metric space-valued outputs. E2M performs prediction via a weighted Fréchet means over training outputs, where the weights are learned by a neural network conditioned on the input. This construction provides a principled mechanism for geometry-aware prediction that avoids surrogate embeddings and restrictive parametric assumptions, while fully preserving the intrinsic geometry of the output space. We establish theoretical guarantees, including a universal approximation theorem that characterizes the expressive capacity of the model and a convergence analysis of the entropy-regularized training objective. Through extensive simulations involving probability distributions, networks, and symmetric positive-definite matrices, we show that E2M consistently achieves state-of-the-art performance, with its advantages becoming more pronounced at larger sample sizes. Applications to human mortality distributions and New York City taxi networks further demonstrate the flexibility and practical utility of the framework. 

**Abstract (ZH)**: 端到端度量回归：一种预测一般度量空间输出的深度学习框架 

---
# Imaging-Based Mortality Prediction in Patients with Systemic Sclerosis 

**Title (ZH)**: 基于成像的系统性硬化症患者 mortality 预测 

**Authors**: Alec K. Peltekian, Karolina Senkow, Gorkem Durak, Kevin M. Grudzinski, Bradford C. Bemiss, Jane E. Dematte, Carrie Richardson, Nikolay S. Markov, Mary Carns, Kathleen Aren, Alexandra Soriano, Matthew Dapas, Harris Perlman, Aaron Gundersheimer, Kavitha C. Selvan, John Varga, Monique Hinchcliff, Krishnan Warrior, Catherine A. Gao, Richard G. Wunderink, GR Scott Budinger, Alok N. Choudhary, Anthony J. Esposito, Alexander V. Misharin, Ankit Agrawal, Ulas Bagci  

**Link**: [PDF](https://arxiv.org/pdf/2509.23530)  

**Abstract**: Interstitial lung disease (ILD) is a leading cause of morbidity and mortality in systemic sclerosis (SSc). Chest computed tomography (CT) is the primary imaging modality for diagnosing and monitoring lung complications in SSc patients. However, its role in disease progression and mortality prediction has not yet been fully clarified. This study introduces a novel, large-scale longitudinal chest CT analysis framework that utilizes radiomics and deep learning to predict mortality associated with lung complications of SSc. We collected and analyzed 2,125 CT scans from SSc patients enrolled in the Northwestern Scleroderma Registry, conducting mortality analyses at one, three, and five years using advanced imaging analysis techniques. Death labels were assigned based on recorded deaths over the one-, three-, and five-year intervals, confirmed by expert physicians. In our dataset, 181, 326, and 428 of the 2,125 CT scans were from patients who died within one, three, and five years, respectively. Using ResNet-18, DenseNet-121, and Swin Transformer we use pre-trained models, and fine-tuned on 2,125 images of SSc patients. Models achieved an AUC of 0.769, 0.801, 0.709 for predicting mortality within one-, three-, and five-years, respectively. Our findings highlight the potential of both radiomics and deep learning computational methods to improve early detection and risk assessment of SSc-related interstitial lung disease, marking a significant advancement in the literature. 

**Abstract (ZH)**: 系统硬化病相关间质性肺病的胸部CT纵向分析框架：基于放射omics和深度学习的死亡率预测研究 

---
# Revisiting Multivariate Time Series Forecasting with Missing Values 

**Title (ZH)**: revisit 多变量时间序列预测中的缺失值问题 

**Authors**: Jie Yang, Yifan Hu, Kexin Zhang, Luyang Niu, Yushun Dong, Philip S. Yu, Kaize Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.23494)  

**Abstract**: Missing values are common in real-world time series, and multivariate time series forecasting with missing values (MTSF-M) has become a crucial area of research for ensuring reliable predictions. To address the challenge of missing data, current approaches have developed an imputation-then-prediction framework that uses imputation modules to fill in missing values, followed by forecasting on the imputed data. However, this framework overlooks a critical issue: there is no ground truth for the missing values, making the imputation process susceptible to errors that can degrade prediction accuracy. In this paper, we conduct a systematic empirical study and reveal that imputation without direct supervision can corrupt the underlying data distribution and actively degrade prediction accuracy. To address this, we propose a paradigm shift that moves away from imputation and directly predicts from the partially observed time series. We introduce Consistency-Regularized Information Bottleneck (CRIB), a novel framework built on the Information Bottleneck principle. CRIB combines a unified-variate attention mechanism with a consistency regularization scheme to learn robust representations that filter out noise introduced by missing values while preserving essential predictive signals. Comprehensive experiments on four real-world datasets demonstrate the effectiveness of CRIB, which predicts accurately even under high missing rates. Our code is available in this https URL. 

**Abstract (ZH)**: 缺失值在实时序列中很常见，多变量时间序列预测中的缺失值（MTSF-M）已成为确保可靠预测的关键研究领域。本文系统地研究了缺失数据的问题，揭示了在缺乏直接监督的情况下进行插补会破坏底层数据分布并主动降低预测精度。为此，我们提出了一种范式转变，即从插补转向直接从部分观察到的时间序列进行预测。我们引入了一致性正则化信息瓶颈（CRIB）框架，这是一种基于信息瓶颈原理的新型框架。CRIB 结合了一致性正则化方案和统一变量注意力机制，用于学习稳健的表示，这些表示可以过滤掉由缺失值引入的噪声，同时保留重要的预测信号。在四个真实世界数据集上的全面实验表明，CRIB 即使在高缺失率下也能准确预测。我们的代码可在以下链接获取：this https URL。 

---
# Text-Based Approaches to Item Difficulty Modeling in Large-Scale Assessments: A Systematic Review 

**Title (ZH)**: 基于文本的方法在大规模评估中项目难度建模：一项系统回顾 

**Authors**: Sydney Peters, Nan Zhang, Hong Jiao, Ming Li, Tianyi Zhou, Robert Lissitz  

**Link**: [PDF](https://arxiv.org/pdf/2509.23486)  

**Abstract**: Item difficulty plays a crucial role in test performance, interpretability of scores, and equity for all test-takers, especially in large-scale assessments. Traditional approaches to item difficulty modeling rely on field testing and classical test theory (CTT)-based item analysis or item response theory (IRT) calibration, which can be time-consuming and costly. To overcome these challenges, text-based approaches leveraging machine learning and language models, have emerged as promising alternatives. This paper reviews and synthesizes 37 articles on automated item difficulty prediction in large-scale assessment settings published through May 2025. For each study, we delineate the dataset, difficulty parameter, subject domain, item type, number of items, training and test data split, input, features, model, evaluation criteria, and model performance outcomes. Results showed that although classic machine learning models remain relevant due to their interpretability, state-of-the-art language models, using both small and large transformer-based architectures, can capture syntactic and semantic patterns without the need for manual feature engineering. Uniquely, model performance outcomes were summarized to serve as a benchmark for future research and overall, text-based methods have the potential to predict item difficulty with root mean square error (RMSE) as low as 0.165, Pearson correlation as high as 0.87, and accuracy as high as 0.806. The review concludes by discussing implications for practice and outlining future research directions for automated item difficulty modeling. 

**Abstract (ZH)**: 项目难度在测试表现、分数解释性和所有应试者的公平性中起着关键作用，尤其是在大规模评估中。传统的项目难度建模方法依赖于场测和基于经典测验理论（CTT）的项目分析或基于项目反应理论（IRT）的校准，这可能会耗费大量时间和成本。为克服这些挑战，利用机器学习和语言模型的基于文本的方法已成为有前景的替代方案。本文通过2025年5月回顾并综合了37篇关于大规模评估环境中自动化项目难度预测的文章。对于每项研究，我们详细介绍了数据集、难度参数、研究领域、项目类型、项目数量、训练和测试数据分割、输入、特征、模型、评估标准以及模型性能结果。结果显示，尽管经典的机器学习模型仍具有一定的解释性，但最先进的语言模型，无论是小型还是大型变压器架构，都能够捕获句法和语义模式，无需手动特征工程。模型性能结果被总结为未来研究的基准，总体而言，基于文本的方法有可能预测项目难度，其中均方根误差(RMSE)低至0.165，皮尔逊相关系数高达0.87，准确率高达0.806。本文结论讨论了实践启示并概述了自动化项目难度建模的未来研究方向。 

---
# Memory-Efficient Fine-Tuning via Low-Rank Activation Compression 

**Title (ZH)**: 低秩激活压缩实现高效内存微调 

**Authors**: Jiang-Xin Shi, Wen-Da Wei, Jin-Fei Qi, Xuanyu Chen, Tong Wei, Yu-Feng Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.23472)  

**Abstract**: The parameter-efficient fine-tuning paradigm has garnered significant attention with the advancement of foundation models. Although numerous methods have been proposed to reduce the number of trainable parameters, their substantial memory overhead remains a critical bottleneck that hinders practical deployment. In this paper, we observe that model activations constitute a major source of memory consumption, especially under large batch sizes and long context lengths; however, the rank of the activations remains consistently low. Motivated by this insight, we propose a memory-efficient fine-tuning approach Low-Rank Activation Compression (LoRAct). Unlike prior work, LoRAct provides a more flexible and versatile compressing strategy that can be applied online during the forward pass without the need for any calibration data. Moreover, LoRAct incorporates a novel sampling-based orthogonal decomposition algorithm specifically designed for low-rank matrices, offering improved computational efficiency and a tighter error bound compared to the widely used RSVD. Experiments on both vision and language tasks demonstrate the effectiveness of LoRAct. Notably, LoRAct further reduces activation memory by approximately 80% in comparison with the widely adopted LoRA method, while maintaining competitive performance. The source code is available at this https URL. 

**Abstract (ZH)**: 基础模型发展的参数高效微调范式引起了广泛关注。尽管提出了许多减少可训练参数数量的方法，但它们带来的显存 overhead 仍然是阻碍其实用部署的关键瓶颈。在本文中，我们观察到模型激活构成了主要的显存消耗来源，尤其是在大批次和长上下文长度的情况下；然而，这些激活的秩保持在较低水平。基于这一见解，我们提出了一种高效的微调方法——低秩激活压缩（LoRAct）。与以往工作不同，LoRAct 提供了一种更为灵活和通用的压缩策略，可以在前向通过过程中在线应用而无需任何校准数据。此外，LoRAct 结合了一种新颖的基于抽样的正交分解算法，专门设计用于低秩矩阵，提供比广泛使用的RSVD更好的计算效率和更紧的误差界。实验结果表明，LoRAct 在视觉和语言任务中均有效。值得注意的是，与广泛采用的LoRA方法相比，LoRAct 进一步减少了约80%的激活显存消耗，同时保持了竞争力的性能。源代码可通过此链接获取。 

---
# Generative Evolutionary Meta-Solver (GEMS): Scalable Surrogate-Free Multi-Agent Learning 

**Title (ZH)**: 生成进化元求解器（GEMS）：可扩展的无代理模拟多agent学习 

**Authors**: Alakh Sharma, Gaurish Trivedi, Kartikey Bhandari, Yash Sinha, Dhruv Kumar, Pratik Narang, Jagat Sesh Challa  

**Link**: [PDF](https://arxiv.org/pdf/2509.23462)  

**Abstract**: Scalable multi-agent reinforcement learning (MARL) remains a central challenge for AI. Existing population-based methods, like Policy-Space Response Oracles, PSRO, require storing explicit policy populations and constructing full payoff matrices, incurring quadratic computation and linear memory costs. We present Generative Evolutionary Meta-Solver (GEMS), a surrogate-free framework that replaces explicit populations with a compact set of latent anchors and a single amortized generator. Instead of exhaustively constructing the payoff matrix, GEMS relies on unbiased Monte Carlo rollouts, multiplicative-weights meta-dynamics, and a model-free empirical-Bernstein UCB oracle to adaptively expand the policy set. Best responses are trained within the generator using an advantage-based trust-region objective, eliminating the need to store and train separate actors. We evaluated GEMS in a variety of Two-player and Multi-Player games such as the Deceptive Messages Game, Kuhn Poker and Multi-Particle environment. We find that GEMS is up to ~6x faster, has 1.3x less memory usage than PSRO, while also reaps higher rewards simultaneously. These results demonstrate that GEMS retains the game theoretic guarantees of PSRO, while overcoming its fundamental inefficiencies, hence enabling scalable multi-agent learning in multiple domains. 

**Abstract (ZH)**: 无补贴进化元求解器（GEMS）：Scalable Multi-Agent Reinforcement Learning Framework 

---
# Data-Efficient Training by Evolved Sampling 

**Title (ZH)**: 进化采样实现数据高效训练 

**Authors**: Ziheng Cheng, Zhong Li, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2509.23461)  

**Abstract**: Data selection is designed to accelerate learning with preserved performance. To achieve this, a fundamental thought is to identify informative data samples with significant contributions to the training. In this work, we propose \textbf{Evolved Sampling} (\textbf{ES}), a simple yet effective framework for \emph{dynamic} sampling along the training process. This method conducts \em batch \em level data selection based on the dynamics of losses and augmented \emph{loss differences}, which enables flexible \emph{frequency tuning}, and hence significantly reduces the back propagation time with maintained model performance. Due to its conciseness, ES is also readily extensible to incorporate \em set \em level data selection (to form ES with pruning, \textbf{ESWP}) for further accelerations. As a plug-and-play framework, ES(WP) consistently achieves lossless training accelerations across various pre-training and post-training tasks, saving up to nearly 45\% wall-clock time. Our results motivate further investigations on the data efficiency aspect of modern large-scale machine learning. 

**Abstract (ZH)**: 数据选择旨在保持性能的同时加速学习。为了实现这一目标，一个根本性的想法是识别对训练有显著贡献的信息性数据样本。在本文中，我们提出了一种简单而有效的方法 \textbf{进化采样} (\textbf{ES})，这是一种在训练过程中进行动态采样的框架。该方法基于损失动态和增强的损失差进行批次级别数据选择，从而实现灵活的频率调优，并显著减少了回传时间，同时保持模型性能。由于其简洁性，ES 也可以方便地扩展为结合集合级别数据选择（形成具有剪枝的 \textbf{ESWP}）以进一步加速。作为一种即插即用框架，ES(WP) 在各种预训练和后训练任务中实现了无损训练加速，最高可节省近 45% 的实际时间。我们的结果促使我们进一步研究现代大规模机器学习中的数据效率方面。 

---
# AudioFuse: Unified Spectral-Temporal Learning via a Hybrid ViT-1D CNN Architecture for Robust Phonocardiogram Classification 

**Title (ZH)**: AudioFuse： através 混合 ViT-1D CNN 架构的统一频谱-时间学习，用于稳健的 Phonocardiogram 分类 

**Authors**: Md. Saiful Bari Siddiqui, Utsab Saha  

**Link**: [PDF](https://arxiv.org/pdf/2509.23454)  

**Abstract**: Biomedical audio signals, such as phonocardiograms (PCG), are inherently rhythmic and contain diagnostic information in both their spectral (tonal) and temporal domains. Standard 2D spectrograms provide rich spectral features but compromise the phase information and temporal precision of the 1D waveform. We propose AudioFuse, an architecture that simultaneously learns from both complementary representations to classify PCGs. To mitigate the overfitting risk common in fusion models, we integrate a custom, wide-and-shallow Vision Transformer (ViT) for spectrograms with a shallow 1D CNN for raw waveforms. On the PhysioNet 2016 dataset, AudioFuse achieves a state-of-the-art competitive ROC-AUC of 0.8608 when trained from scratch, outperforming its spectrogram (0.8066) and waveform (0.8223) baselines. Moreover, it demonstrates superior robustness to domain shift on the challenging PASCAL dataset, maintaining an ROC-AUC of 0.7181 while the spectrogram baseline collapses (0.4873). Fusing complementary representations thus provides a strong inductive bias, enabling the creation of efficient, generalizable classifiers without requiring large-scale pre-training. 

**Abstract (ZH)**: biomedical 音频信号，如心音图 (PCG)，本质上是 rhythmic 的，并且在其频谱 (音调) 和时间域中包含诊断信息。标准的 2D 谱图提供了丰富的频谱特征，但牺牲了时间波形的相位信息和时间精度。我们提出 AudioFuse 架构，该架构同时从互补的表示中学习以分类 PCG。为缓解融合模型中常见的过拟合风险，我们整合了一个定制的宽浅 Vision Transformer (ViT) 用于谱图，以及一个浅层 1D CNN 用于原始波形。在 PhysioNet 2016 数据集上，从头训练的 AudioFuse 达到了 0.8608 的竞争性 ROC-AUC，优于其谱图 baselines (0.8066) 和波形 baselines (0.8223)。此外，它在具有挑战性的 PASCAL 数据集上展示了对领域转移的优越鲁棒性，在保持 ROC-AUC 为 0.7181 的同时，谱图 baseline 下降至 0.4873。因此，融合互补表示提供了强大的归纳偏置，使得可以创建高效且可泛化的分类器，而无需大规模预训练。 

---
# Factor Decorrelation Enhanced Data Removal from Deep Predictive Models 

**Title (ZH)**: 深度预测模型中因素去相关增强的数据删除 

**Authors**: Wenhao Yang, Lin Li, Xiaohui Tao, Kaize Shi  

**Link**: [PDF](https://arxiv.org/pdf/2509.23443)  

**Abstract**: The imperative of user privacy protection and regulatory compliance necessitates sensitive data removal in model training, yet this process often induces distributional shifts that undermine model performance-particularly in out-of-distribution (OOD) scenarios. We propose a novel data removal approach that enhances deep predictive models through factor decorrelation and loss perturbation. Our approach introduces: (1) a discriminative-preserving factor decorrelation module employing dynamic adaptive weight adjustment and iterative representation updating to reduce feature redundancy and minimize inter-feature correlations. (2) a smoothed data removal mechanism with loss perturbation that creates information-theoretic safeguards against data leakage during removal operations. Extensive experiments on five benchmark datasets show that our approach outperforms other baselines and consistently achieves high predictive accuracy and robustness even under significant distribution shifts. The results highlight its superior efficiency and adaptability in both in-distribution and out-of-distribution scenarios. 

**Abstract (ZH)**: 用户隐私保护和监管合规的迫切性要求在模型训练中移除敏感数据，但这一过程常会引起分布变化，特别是在分布外(OOD)场景中损害模型性能。我们提出了一种新颖的数据移除方法，通过因子去相关和损失扰动来增强深度预测模型。(1) 一种保持鉴别信息的因子去相关模块，采用动态自适应权重调整和迭代表示更新来减少特征冗余并最小化特征间的相关性。(2) 一种平滑的数据移除机制，通过损失扰动创建信息论上的安全防护，防止在数据移除操作中发生数据泄漏。在五个基准数据集上的广泛实验表明，本方法优于其他基线方法，并且能够在显著分布变化下一致地实现高预测准确性和鲁棒性。结果突显了其在分布内和分布外场景中的优越效率和适应性。 

---
# Enhancing Communication Efficiency in FL with Adaptive Gradient Quantization and Communication Frequency Optimization 

**Title (ZH)**: 适应性梯度量化与通信频率优化以提升联邦学习中的通信效率 

**Authors**: Asadullah Tariq, Tariq Qayyum, Mohamed Adel Serhani, Farag Sallabi, Ikbal Taleb, Ezedin S. Barka  

**Link**: [PDF](https://arxiv.org/pdf/2509.23419)  

**Abstract**: Federated Learning (FL) enables participant devices to collaboratively train deep learning models without sharing their data with the server or other devices, effectively addressing data privacy and computational concerns. However, FL faces a major bottleneck due to high communication overhead from frequent model updates between devices and the server, limiting deployment in resource-constrained wireless networks. In this paper, we propose a three-fold strategy. Firstly, an Adaptive Feature-Elimination Strategy to drop less important features while retaining high-value ones; secondly, Adaptive Gradient Innovation and Error Sensitivity-Based Quantization, which dynamically adjusts the quantization level for innovative gradient compression; and thirdly, Communication Frequency Optimization to enhance communication efficiency. We evaluated our proposed model's performance through extensive experiments, assessing accuracy, loss, and convergence compared to baseline techniques. The results show that our model achieves high communication efficiency in the framework while maintaining accuracy. 

**Abstract (ZH)**: 联邦学习（FL）使参与设备能够在不共享数据给服务器或其他设备的情况下协作训练深度学习模型，有效地解决了数据隐私和计算问题。然而，FL由于频繁的模型更新导致的高通信开销面临重大瓶颈，限制了其在资源受限的无线网络中的部署。在本文中，我们提出了一种三管齐下的策略。首先，提出了一种自适应特征消除策略以丢弃不重要的特征同时保留高价值特征；其次，提出了自适应梯度创新和误差敏感量化方法，动态调整创新梯度压缩的量化级别；第三，优化通信频率以提高通信效率。通过广泛实验评估了我们提出模型的性能，与基准技术相比，评估了准确率、损失和收敛性。结果表明，我们的模型在保持高准确率的同时实现了高效的通信。 

---
# Hybrid Graph Embeddings and Louvain Algorithm for Unsupervised Community Detection 

**Title (ZH)**: 混合图嵌入和Louvain算法在无监督社区检测中的应用 

**Authors**: Dalila Khettaf, Djamel Djenouri, Zeinab Rezaeifar, Youcef Djenouri  

**Link**: [PDF](https://arxiv.org/pdf/2509.23411)  

**Abstract**: This paper proposes a novel community detection method that integrates the Louvain algorithm with Graph Neural Networks (GNNs), enabling the discovery of communities without prior knowledge. Compared to most existing solutions, the proposed method does not require prior knowledge of the number of communities. It enhances the Louvain algorithm using node embeddings generated by a GNN to capture richer structural and feature information. Furthermore, it introduces a merging algorithm to refine the results of the enhanced Louvain algorithm, reducing the number of detected communities. To the best of our knowledge, this work is the first one that improves the Louvain algorithm using GNNs for community detection. The improvement of the proposed method was empirically confirmed through an evaluation on real-world datasets. The results demonstrate its ability to dynamically adjust the number of detected communities and increase the detection accuracy in comparison with the benchmark solutions. 

**Abstract (ZH)**: 本文提出了一种将Louvain算法与图神经网络（GNNs）集成的新颖社区检测方法，能够在无先验知识的情况下发现社区。与大多数现有解决方案不同，该方法不需要知道社区的数量。该方法通过使用GNN生成的节点嵌入来增强Louvain算法，以捕获更丰富的结构和特征信息。此外，它引入了一种聚类算法来细化增强后的Louvain算法的结果，减少了检测到的社区数量。据我们所知，这是首次使用GNNs增强Louvain算法进行社区检测的工作。通过在实际数据集上的评估，实证验证了所提出方法的改进效果。该方法能够在动态调整检测到的社区数量和提高检测准确性方面优于基准解决方案。 

---
# Graph Your Own Prompt 

**Title (ZH)**: 绘制你自己的提示图谱 

**Authors**: Xi Ding, Lei Wang, Piotr Koniusz, Yongsheng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.23373)  

**Abstract**: We propose Graph Consistency Regularization (GCR), a novel framework that injects relational graph structures, derived from model predictions, into the learning process to promote class-aware, semantically meaningful feature representations. Functioning as a form of self-prompting, GCR enables the model to refine its internal structure using its own outputs. While deep networks learn rich representations, these often capture noisy inter-class similarities that contradict the model's predicted semantics. GCR addresses this issue by introducing parameter-free Graph Consistency Layers (GCLs) at arbitrary depths. Each GCL builds a batch-level feature similarity graph and aligns it with a global, class-aware masked prediction graph, derived by modulating softmax prediction similarities with intra-class indicators. This alignment enforces that feature-level relationships reflect class-consistent prediction behavior, acting as a semantic regularizer throughout the network. Unlike prior work, GCR introduces a multi-layer, cross-space graph alignment mechanism with adaptive weighting, where layer importance is learned from graph discrepancy magnitudes. This allows the model to prioritize semantically reliable layers and suppress noisy ones, enhancing feature quality without modifying the architecture or training procedure. GCR is model-agnostic, lightweight, and improves semantic structure across various networks and datasets. Experiments show that GCR promotes cleaner feature structure, stronger intra-class cohesion, and improved generalization, offering a new perspective on learning from prediction structure. [Project website](this https URL) [Code](this https URL) 

**Abstract (ZH)**: 我们提出图一致性正则化（GCR），这是一种新颖的框架，通过将源自模型预测的关系图结构注入学习过程，促进具有类意识和语义意义的特征表示。作为一种自我提示的形式，GCR使模型能够利用自身的输出来精化其内部结构。虽然深层网络学习到丰富的表示，但这些表示往往包含与模型预测的语义相矛盾的嘈杂跨类相似性。GCR通过在任意深度引入无参数的图一致性层（GCLs）来解决这一问题。每个GCL构建一批次级别的特征相似图，并将其与通过调整softmax预测相似度与类别内指示符来生成的全局类意识掩码预测图对齐。这种对齐确保特征级别的关系反映出类一致的预测行为，作为一种语义正则化在整个网络中起作用。与以前的工作不同，GCR引入了一种多层、跨空间的图对齐机制，具有自适应加权，其中层的重要性是从图差异幅度中学习到的。这使得模型能够优先考虑语义可靠的层并抑制嘈杂的层，提高特征质量而不修改架构或训练过程。GCR具有模型无关性，轻量级，并在各种网络和数据集上增强了语义结构。实验表明，GCR促进了更清洁的特征结构、更强的类内凝聚性和更好的泛化能力，为从预测结构学习提供了新的视角。[项目网站](this https URL) [代码](this https URL) 

---
# AI Education in Higher Education: A Taxonomy for Curriculum Reform and the Mission of Knowledge 

**Title (ZH)**: 高等教育中的AI教育：课程改革的分类学及其知识使命 

**Authors**: Tian Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.23363)  

**Abstract**: Artificial intelligence (AI) is reshaping higher education, yet current debates often feel tangled, mixing concerns about pedagogy, operations, curriculum, and the future of work without a shared framework. This paper offers a first attempt at a taxonomy to organize the diverse narratives of AI education and to inform discipline-based curricular discussions. We place these narratives within the enduring responsibility of higher education: the mission of knowledge. This mission includes not only the preservation and advancement of disciplinary expertise, but also the cultivation of skills and wisdom, i.e., forms of meta-knowledge that encompass judgment, ethics, and social responsibility. For the purpose of this paper's discussion, AI is defined as adaptive, data-driven systems that automate analysis, modeling, and decision-making, highlighting its dual role as enabler and disruptor across disciplines. We argue that the most consequential challenges lie at the level of curriculum and disciplinary purpose, where AI accelerates inquiry but also unsettles expertise and identity. We show how disciplines evolve through the interplay of research, curriculum, pedagogy, and faculty expertise, and why curricular reform is the central lever for meaningful change. Pedagogical innovation offers a strategic and accessible entry point, providing actionable steps that help faculty and students build the expertise needed to engage in deeper curricular rethinking and disciplinary renewal. Within this framing, we suggest that meaningful reform can move forward through structured faculty journeys: from AI literacy to pedagogy, curriculum design, and research integration. The key is to align these journeys with the mission of knowledge, turning the disruptive pressures of AI into opportunities for disciplines to sustain expertise, advance inquiry, and serve society. 

**Abstract (ZH)**: 人工智能（AI）正重塑高等教育，然而当前的辩论往往显得杂乱无章，交织着关于教学方法、运营、课程和未来工作前景的担忧，缺乏一个共同的框架。本文旨在首次尝试构建一种分类法，以组织AI教育的多样叙事，并为基于学科的课程讨论提供指导。我们将这些叙事置于高等教育永恒的责任之中：知识传承的任务。这一任务不仅包括学科专长的保存与发展，还包括技能和智慧（即判断、伦理和社会责任等形式的元知识）的培养。为了本文的讨论，我们将人工智能定义为适应性强、数据驱动的系统，能够自动化分析、建模和决策，强调其在各学科中作为助推器和颠覆者的双重角色。我们主张，最重大的挑战在于课程和学科目标的层面，AI加速了探究但同时也动摇了专业和身份。我们展示了学科通过研究、课程、教学和教员专长的相互作用而演变的过程，并阐明了课程改革是推动有意义变化的主要杠杆。教学创新提供了战略性的切入点，提供了一系列可操作的步骤，帮助教员和学生培养在深入课程重思和学科更新中所需的专业能力。在这种框架下，我们建议，有意义的改革可以通过结构化教员旅程推进：从人工智能素养到教学方法、课程设计和研究整合。关键在于将这些旅程与知识传承任务相一致，将AI带来的颠覆性压力转变为学科保持专业能力、推进探究和社会服务的机会。 

---
# Dynamic-TreeRPO: Breaking the Independent Trajectory Bottleneck with Structured Sampling 

**Title (ZH)**: 动态树RPO：通过结构化采样打破独立轨迹瓶颈 

**Authors**: Xiaolong Fu, Lichen Ma, Zipeng Guo, Gaojing Zhou, Chongxiao Wang, ShiPing Dong, Shizhe Zhou, Shizhe Zhou, Ximan Liu, Jingling Fu, Tan Lit Sin, Yu Shi, Zhen Chen, Junshi Huang, Jason Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.23352)  

**Abstract**: The integration of Reinforcement Learning (RL) into flow matching models for text-to-image (T2I) generation has driven substantial advances in generation quality. However, these gains often come at the cost of exhaustive exploration and inefficient sampling strategies due to slight variation in the sampling group. Building on this insight, we propose Dynamic-TreeRPO, which implements the sliding-window sampling strategy as a tree-structured search with dynamic noise intensities along depth. We perform GRPO-guided optimization and constrained Stochastic Differential Equation (SDE) sampling within this tree structure. By sharing prefix paths of the tree, our design effectively amortizes the computational overhead of trajectory search. With well-designed noise intensities for each tree layer, Dynamic-TreeRPO can enhance the variation of exploration without any extra computational cost. Furthermore, we seamlessly integrate Supervised Fine-Tuning (SFT) and RL paradigm within Dynamic-TreeRPO to construct our proposed LayerTuning-RL, reformulating the loss function of SFT as a dynamically weighted Progress Reward Model (PRM) rather than a separate pretraining method. By associating this weighted PRM with dynamic-adaptive clipping bounds, the disruption of exploration process in Dynamic-TreeRPO is avoided. Benefiting from the tree-structured sampling and the LayerTuning-RL paradigm, our model dynamically explores a diverse search space along effective directions. Compared to existing baselines, our approach demonstrates significant superiority in terms of semantic consistency, visual fidelity, and human preference alignment on established benchmarks, including HPS-v2.1, PickScore, and ImageReward. In particular, our model outperforms SoTA by $4.9\%$, $5.91\%$, and $8.66\%$ on those benchmarks, respectively, while improving the training efficiency by nearly $50\%$. 

**Abstract (ZH)**: 动态树结构RPO结合监督微调与 reinforcement learning 在文本到图像生成中的应用 

---
# Robust Fine-Tuning from Non-Robust Pretrained Models: Mitigating Suboptimal Transfer With Adversarial Scheduling 

**Title (ZH)**: 从非鲁棒预训练模型进行鲁棒微调：通过对抗性调度减轻次优转移影响 

**Authors**: Jonas Ngnawé, Maxime Heuillet, Sabyasachi Sahoo, Yann Pequignot, Ola Ahmad, Audrey Durand, Frédéric Precioso, Christian Gagné  

**Link**: [PDF](https://arxiv.org/pdf/2509.23325)  

**Abstract**: Fine-tuning pretrained models is a standard and effective workflow in modern machine learning. However, robust fine-tuning (RFT), which aims to simultaneously achieve adaptation to a downstream task and robustness to adversarial examples, remains challenging. Despite the abundance of non-robust pretrained models in open-source repositories, their potential for RFT is less understood. We address this knowledge gap by systematically examining RFT from such non-robust models. Our experiments reveal that fine-tuning non-robust models with a robust objective, even under small perturbations, can lead to poor performance, a phenomenon that we dub \emph{suboptimal transfer}. In challenging scenarios (eg, difficult tasks, high perturbation), the resulting performance can be so low that it may be considered a transfer failure. We find that fine-tuning using a robust objective impedes task adaptation at the beginning of training and eventually prevents optimal transfer. However, we propose a novel heuristic, \emph{Epsilon-Scheduling}, a schedule over perturbation strength used during training that promotes optimal transfer. Additionally, we introduce \emph{expected robustness}, a metric that captures performance across a range of perturbations, providing a more comprehensive evaluation of the accuracy-robustness trade-off for diverse models at test time. Extensive experiments on a wide range of configurations (six pretrained models and five datasets) show that \emph{Epsilon-Scheduling} successfully prevents \emph{suboptimal transfer} and consistently improves expected robustness. 

**Abstract (ZH)**: 微调预训练模型是现代机器学习中的标准且有效的workflow。然而，鲁棒微调（RFT），其目标是在适应下游任务的同时增强对对抗样本的鲁棒性，仍然具有挑战性。尽管开源库中有大量的非鲁棒预训练模型，但它们的RFT潜力尚未充分理解。我们通过系统地研究这些非鲁棒模型的RFT来填补这一知识空白。我们的实验揭示，即使在小扰动下使用鲁棒目标微调非鲁棒模型，也可能导致性能不佳，我们称之为“次优转移”。在具有挑战性的场景中（例如，困难的任务、高扰动），这种性能可能如此低，以至于可以被视为转移失败。我们发现，使用鲁棒目标进行微调在训练初期阻碍了任务适应，并最终阻止了最优转移。然而，我们提出了一种新颖的启发式方法，称为“ε调度”，这是一种在训练过程中使用的扰动强度调度方案，可促进最优转移。此外，我们引入了“期望鲁棒性”这一度量标准，它捕捉了在一系列扰动下的性能，为测试时不同模型的准确性和鲁棒性权衡提供了更全面的评估。广泛配置（六种预训练模型和五种数据集）的大量实验表明，“ε调度”成功地防止了“次优转移”，并且始终提高了期望鲁棒性。 

---
# MELCOT: A Hybrid Learning Architecture with Marginal Preservation for Matrix-Valued Regression 

**Title (ZH)**: MELCOT: 一种兼顾边缘保留的矩阵值回归混合学习架构 

**Authors**: Khang Tran, Hieu Cao, Thinh Pham, Nghiem Diep, Tri Cao, Binh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2509.23315)  

**Abstract**: Regression is essential across many domains but remains challenging in high-dimensional settings, where existing methods often lose spatial structure or demand heavy storage. In this work, we address the problem of matrix-valued regression, where each sample is naturally represented as a matrix. We propose MELCOT, a hybrid model that integrates a classical machine learning-based Marginal Estimation (ME) block with a deep learning-based Learnable-Cost Optimal Transport (LCOT) block. The ME block estimates data marginals to preserve spatial information, while the LCOT block learns complex global features. This design enables MELCOT to inherit the strengths of both classical and deep learning methods. Extensive experiments across diverse datasets and domains demonstrate that MELCOT consistently outperforms all baselines while remaining highly efficient. 

**Abstract (ZH)**: 矩阵值回归在高维设置中至关重要但依然具有挑战性，现有方法往往会在保留空间结构或需要大量存储方面遇到困难。为解决这一问题，我们提出了一种集成模型MELCOT，该模型结合了基于经典机器学习的边缘估计（ME）模块和基于深度学习的可学习成本最优传输（LCOT）模块。ME模块估计数据边缘以保留空间信息，而LCOT模块学习复杂的全局特征。这种设计使得MELCOT能够继承经典和深度学习方法的优点。广泛的数据集和领域实验表明，MELCOT在所有基线方法中表现最优且具有很高的效率。 

---
# A Neural ODE Approach to Aircraft Flight Dynamics Modelling 

**Title (ZH)**: 一种基于神经ODE的航空飞行动力学建模方法 

**Authors**: Gabriel Jarry, Ramon Dalmau, Xavier Olive, Philippe Very  

**Link**: [PDF](https://arxiv.org/pdf/2509.23307)  

**Abstract**: Accurate aircraft trajectory prediction is critical for air traffic management, airline operations, and environmental assessment. This paper introduces NODE-FDM, a Neural Ordinary Differential Equations-based Flight Dynamics Model trained on Quick Access Recorder (QAR) data. By combining analytical kinematic relations with data-driven components, NODE-FDM achieves a more accurate reproduction of recorded trajectories than state-of-the-art models such as a BADA-based trajectory generation methodology (BADA4 performance model combined with trajectory control routines), particularly in the descent phase of the flight. The analysis demonstrates marked improvements across altitude, speed, and mass dynamics. Despite current limitations, including limited physical constraints and the limited availability of QAR data, the results demonstrate the potential of physics-informed neural ordinary differential equations as a high-fidelity, data-driven approach to aircraft performance modelling. Future work will extend the framework to incorporate a full modelling of the lateral dynamics of the aircraft. 

**Abstract (ZH)**: 基于神经常微分方程的飞行动力学模型NODE-FDM及其对飞机轨迹预测的应用 

---
# Continuous-Time Reinforcement Learning for Asset-Liability Management 

**Title (ZH)**: 连续时间强化学习在资产-负债管理中的应用 

**Authors**: Yilie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23280)  

**Abstract**: This paper proposes a novel approach for Asset-Liability Management (ALM) by employing continuous-time Reinforcement Learning (RL) with a linear-quadratic (LQ) formulation that incorporates both interim and terminal objectives. We develop a model-free, policy gradient-based soft actor-critic algorithm tailored to ALM for dynamically synchronizing assets and liabilities. To ensure an effective balance between exploration and exploitation with minimal tuning, we introduce adaptive exploration for the actor and scheduled exploration for the critic. Our empirical study evaluates this approach against two enhanced traditional financial strategies, a model-based continuous-time RL method, and three state-of-the-art RL algorithms. Evaluated across 200 randomized market scenarios, our method achieves higher average rewards than all alternative strategies, with rapid initial gains and sustained superior performance. The outperformance stems not from complex neural networks or improved parameter estimation, but from directly learning the optimal ALM strategy without learning the environment. 

**Abstract (ZH)**: 基于连续时间强化学习的资产-负债管理新方法：线性二次形式兼顾中间和最终目标 

---
# Patch Rebirth: Toward Fast and Transferable Model Inversion of Vision Transformers 

**Title (ZH)**: Patch 重生：朝向快速可移植的视觉变换器模型倒置 

**Authors**: Seongsoo Heo, Dong-Wan Choi  

**Link**: [PDF](https://arxiv.org/pdf/2509.23235)  

**Abstract**: Model inversion is a widely adopted technique in data-free learning that reconstructs synthetic inputs from a pretrained model through iterative optimization, without access to original training data. Unfortunately, its application to state-of-the-art Vision Transformers (ViTs) poses a major computational challenge, due to their expensive self-attention mechanisms. To address this, Sparse Model Inversion (SMI) was proposed to improve efficiency by pruning and discarding seemingly unimportant patches, which were even claimed to be obstacles to knowledge transfer. However, our empirical findings suggest the opposite: even randomly selected patches can eventually acquire transferable knowledge through continued inversion. This reveals that discarding any prematurely inverted patches is inefficient, as it suppresses the extraction of class-agnostic features essential for knowledge transfer, along with class-specific features. In this paper, we propose Patch Rebirth Inversion (PRI), a novel approach that incrementally detaches the most important patches during the inversion process to construct sparse synthetic images, while allowing the remaining patches to continue evolving for future selection. This progressive strategy not only improves efficiency, but also encourages initially less informative patches to gradually accumulate more class-relevant knowledge, a phenomenon we refer to as the Re-Birth effect, thereby effectively balancing class-agnostic and class-specific knowledge. Experimental results show that PRI achieves up to 10x faster inversion than standard Dense Model Inversion (DMI) and 2x faster than SMI, while consistently outperforming SMI in accuracy and matching the performance of DMI. 

**Abstract (ZH)**: Patch Rebirth Inversion：一种渐进式关键区域再生的稀疏模型反转方法 

---
# One-Shot Multi-Label Causal Discovery in High-Dimensional Event Sequences 

**Title (ZH)**: 一-shot多标签因果发现高维事件序列 

**Authors**: Hugo Math, Robin Schön, Rainer Lienhart  

**Link**: [PDF](https://arxiv.org/pdf/2509.23213)  

**Abstract**: Understanding causality in event sequences with thousands of sparse event types is critical in domains such as healthcare, cybersecurity, or vehicle diagnostics, yet current methods fail to scale. We present OSCAR, a one-shot causal autoregressive method that infers per-sequence Markov Boundaries using two pretrained Transformers as density estimators. This enables efficient, parallel causal discovery without costly global CI testing. On a real-world automotive dataset with 29,100 events and 474 labels, OSCAR recovers interpretable causal structures in minutes, while classical methods fail to scale, enabling practical scientific diagnostics at production scale. 

**Abstract (ZH)**: 理解含有数千种稀疏事件类型的事件序列因果关系在医疗保健、网络安全或车辆诊断等领域至关重要，但当前方法无法扩展。我们提出了OSCAR，一种基于两个预训练Transformer作为密度估计器的一次性因果自回归方法，该方法通过推断每个序列的马尔可夫边界，实现了高效的并行因果发现，而无需昂贵的全局CI测试。在包含29,100个事件和474个标签的现实世界汽车数据集中，OSCAR能够在几分钟内恢复可解释的因果结构，而经典方法无法扩展，从而在生产规模上实现了实用的科学诊断。 

---
# WARBERT: A Hierarchical BERT-based Model for Web API Recommendation 

**Title (ZH)**: WARBERT：一种基于层级BERT的Web API推荐模型 

**Authors**: Zishuo Xu, Yuhong Gu, Dezhong Yao  

**Link**: [PDF](https://arxiv.org/pdf/2509.23175)  

**Abstract**: With the emergence of Web 2.0 and microservices architecture, the number of Web APIs has increased dramatically, further intensifying the demand for efficient Web API recommendation. Existing solutions typically fall into two categories: recommendation-type methods, which treat each API as a label for classification, and match-type methods, which focus on matching mashups through API retrieval. However, three critical challenges persist: 1) the semantic ambiguities in comparing API and mashup descriptions, 2) the lack of detailed comparisons between the individual API and the mashup in recommendation-type methods, and 3) time inefficiencies for API retrieval in match-type methods. To address these challenges, we propose WARBERT, a hierarchical BERT-based model for Web API recommendation. WARBERT leverages dual-component feature fusion and attention comparison to extract precise semantic representations of API and mashup descriptions. WARBERT consists of two main components: WARBERT(R) for Recommendation and WARBERT(M) for Matching. Specifically, WAR-BERT(R) serves as an initial filter, narrowing down the candidate APIs, while WARBERT(M) refines the matching process by calculating the similarity between candidate APIs and mashup. The final likelihood of a mashup being matched with an API is determined by combining the predictions from WARBERT(R) and WARBERT(M). Additionally, WARBERT(R) incorporates an auxiliary task of mashup category judgment, which enhances its effectiveness in candidate selection. Experimental results on the ProgrammableWeb dataset demonstrate that WARBERT outperforms most existing solutions and achieves improvements of up to 11.7% compared to the model MTFM (Multi-Task Fusion Model), delivering significant enhancements in accuracy and effiency. 

**Abstract (ZH)**: 基于BERT的层次模型WARBERT面向Web API推荐 

---
# Dense associative memory on the Bures-Wasserstein space 

**Title (ZH)**: Bures-Wasserstein空间中的密集关联记忆 

**Authors**: Chandan Tankala, Krishnakumar Balasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2509.23162)  

**Abstract**: Dense associative memories (DAMs) store and retrieve patterns via energy-functional fixed points, but existing models are limited to vector representations. We extend DAMs to probability distributions equipped with the 2-Wasserstein distance, focusing mainly on the Bures-Wasserstein class of Gaussian densities. Our framework defines a log-sum-exp energy over stored distributions and a retrieval dynamics aggregating optimal transport maps in a Gibbs-weighted manner. Stationary points correspond to self-consistent Wasserstein barycenters, generalizing classical DAM fixed points. We prove exponential storage capacity, provide quantitative retrieval guarantees under Wasserstein perturbations, and validate the model on synthetic and real-world distributional tasks. This work elevates associative memory from vectors to full distributions, bridging classical DAMs with modern generative modeling and enabling distributional storage and retrieval in memory-augmented learning. 

**Abstract (ZH)**: 稠密关联记忆（DAMs）通过能量函数的固定点存储和检索模式，但现有模型仅限于向量表示。我们扩展了DAMs到配备2- Wasserstein距离的概率分布，重点关注Bures-Wasserstein类高斯密度。我们的框架定义了存储分布上的对数和最大值能量，并通过吉布斯加权方式聚合并检索最优运输映射。稳定点对应于自洽的Wasserstein平均中心，推广了经典的DAM固定点。我们证明了指数级的存储容量，在Wasserstein扰动下提供了定量的检索保证，并在合成和实际分布任务上验证了该模型。这项工作将关联记忆从向量提升到完整的概率分布，将经典的DAM与现代生成建模相结合，使记忆增强学习中的分布存储和检索成为可能。 

---
# Deep Learning-Based Detection of Cognitive Impairment from Passive Smartphone Sensing with Routine-Aware Augmentation and Demographic Personalization 

**Title (ZH)**: 基于深度学习的基于被动智能手机 sensing 的认知障碍检测：带有活动感知增强和个人化demographic参数方法 

**Authors**: Yufei Shen, Ji Hwan Park, Minchao Huang, Jared F. Benge, Justin F. Rousseau, Rosemary A. Lester-Smith, Edison Thomaz  

**Link**: [PDF](https://arxiv.org/pdf/2509.23158)  

**Abstract**: Early detection of cognitive impairment is critical for timely diagnosis and intervention, yet infrequent clinical assessments often lack the sensitivity and temporal resolution to capture subtle cognitive declines in older adults. Passive smartphone sensing has emerged as a promising approach for naturalistic and continuous cognitive monitoring. Building on this potential, we implemented a Long Short-Term Memory (LSTM) model to detect cognitive impairment from sequences of daily behavioral features, derived from multimodal sensing data collected in an ongoing one-year study of older adults. Our key contributions are two techniques to enhance model generalizability across participants: (1) routine-aware augmentation, which generates synthetic sequences by replacing each day with behaviorally similar alternatives, and (2) demographic personalization, which reweights training samples to emphasize those from individuals demographically similar to the test participant. Evaluated on 6-month data from 36 older adults, these techniques jointly improved the Area Under the Precision-Recall Curve (AUPRC) of the model trained on sensing and demographic features from 0.637 to 0.766, highlighting the potential of scalable monitoring of cognitive impairment in aging populations with passive sensing. 

**Abstract (ZH)**: 早期认知损害的检测对于及时诊断和干预至关重要，但频繁不足的临床评估往往缺乏敏感性和时序分辨率来捕捉老年人的细微认知下降。被动智能手机传感已成为自然且持续认知监测的有前途的方法。基于这一潜力，我们实现了一个长短期记忆（LSTM）模型，用于从持续一年的老年人群多模态传感数据中提取的每日行为特征序列检测认知损害。我们的主要贡献是两种增强模型泛化性的技术：（1）基于日常习惯的增强，通过用行为相似的替代日来替换每一天生成合成序列，以及（2）基于人口特征个性化，通过对类似于测试参与者的人群的样本重新加权来强调其重要性。在36名老年人6个月的数据上进行评估，这两种技术共同将使用传感和人口特征训练的模型的精准召回曲线下面积（AUPRC）从0.637提高到0.766，突显了使用被动传感进行认知损害的可扩展监测在老龄化人群中的潜力。 

---
# Trust Region Reward Optimization and Proximal Inverse Reward Optimization Algorithm 

**Title (ZH)**: 信赖区域奖励优化与近端逆奖励优化算法 

**Authors**: Yang Chen, Menglin Zou, Jiaqi Zhang, Yitan Zhang, Junyi Yang, Gael Gendron, Libo Zhang, Jiamou Liu, Michael J. Witbrock  

**Link**: [PDF](https://arxiv.org/pdf/2509.23135)  

**Abstract**: Inverse Reinforcement Learning (IRL) learns a reward function to explain expert demonstrations. Modern IRL methods often use the adversarial (minimax) formulation that alternates between reward and policy optimization, which often lead to unstable training. Recent non-adversarial IRL approaches improve stability by jointly learning reward and policy via energy-based formulations but lack formal guarantees. This work bridges this gap. We first present a unified view showing canonical non-adversarial methods explicitly or implicitly maximize the likelihood of expert behavior, which is equivalent to minimizing the expected return gap. This insight leads to our main contribution: Trust Region Reward Optimization (TRRO), a framework that guarantees monotonic improvement in this likelihood via a Minorization-Maximization process. We instantiate TRRO into Proximal Inverse Reward Optimization (PIRO), a practical and stable IRL algorithm. Theoretically, TRRO provides the IRL counterpart to the stability guarantees of Trust Region Policy Optimization (TRPO) in forward RL. Empirically, PIRO matches or surpasses state-of-the-art baselines in reward recovery, policy imitation with high sample efficiency on MuJoCo and Gym-Robotics benchmarks and a real-world animal behavior modeling task. 

**Abstract (ZH)**: 逆强化学习中的信任区域奖励优化（TRRO） 

---
# C$^2$GSPG: Confidence-calibrated Group Sequence Policy Gradient towards Self-aware Reasoning 

**Title (ZH)**: C$^2$GSPG: 信心校准的群体序列策略梯度 toward 自我意识推理 

**Authors**: Haotian Liu, Shuo Wang, Hongteng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23129)  

**Abstract**: Reinforcement Learning (RL) methods, exemplified by Group Relative Policy Optimization (GRPO) and its variants, play a central role in developing reasoning models. However, these methods often suffer from a critical overconfidence issue, which prevents them from achieving self-aware reasoning models. In this study, we propose a simple yet effective confidence-calibration group sequence policy gradient method, called C$^2$GSPG, which simultaneously enhances reasoning performance while suppressing overconfidence. In principle, we propose a Group Sequence Policy Gradient (GSPG) framework for learning reasoning models, which eliminates the token-level bias commonly appearing in GRPO and its variants. In this framework, we define the model confidence for each reasoning problem using the normalized sequence-level probability, and then apply a cross-entropy regularizer to calibrate the model confidence to the sequence's reward. We demonstrate that the confidence calibration regularizer and GSPG are collaborative for binary rewards, as their objectives always share the same gradient direction. For non-binary rewards, we apply nonlinear reward normalization and adaptive regularizer clipping, mitigating the potential conflict between the two objectives. Applying C$^2$GSPG to post-train large language models in logical and mathematical reasoning tasks, we show its superiority over state-of-the-art methods in both reasoning accuracy and confidence calibration. The code of C$^2$GSPG is available at this https URL. 

**Abstract (ZH)**: 基于信心校准的组序列策略梯度方法（C$^2$GSPG）及其在强化学习中的应用 

---
# HTMA-Net: Towards Multiplication-Avoiding Neural Networks via Hadamard Transform and In-Memory Computing 

**Title (ZH)**: HTMA-Net：通过哈达玛变换和内存计算实现的避免乘法的神经网络 

**Authors**: Emadeldeen Hamdan, Ahmet Enis Cetin  

**Link**: [PDF](https://arxiv.org/pdf/2509.23103)  

**Abstract**: Reducing the cost of multiplications is critical for efficient deep neural network deployment, especially in energy-constrained edge devices. In this work, we introduce HTMA-Net, a novel framework that integrates the Hadamard Transform (HT) with multiplication-avoiding (MA) SRAM-based in-memory computing to reduce arithmetic complexity while maintaining accuracy. Unlike prior methods that only target multiplications in convolutional layers or focus solely on in-memory acceleration, HTMA-Net selectively replaces intermediate convolutions with Hybrid Hadamard-based transform layers whose internal convolutions are implemented via multiplication-avoiding in-memory operations. We evaluate HTMA-Net on ResNet-18 using CIFAR-10, CIFAR-100, and Tiny ImageNet, and provide a detailed comparison against regular, MF-only, and HT-only variants. Results show that HTMA-Net eliminates up to 52\% of multiplications compared to baseline ResNet-18, ResNet-20, and ResNet-50 models, while achieving comparable accuracy in evaluation and significantly reducing computational complexity and the number of parameters. Our results demonstrate that combining structured Hadamard transform layers with SRAM-based in-memory computing multiplication-avoiding operators is a promising path towards efficient deep learning architectures. 

**Abstract (ZH)**: 降低乘法成本对于高效部署深度神经网络至关重要，尤其是在能量受限的边缘设备上。本文提出HTMA-Net，这是一种将哈达玛变换（HT）与避免乘法（MA）的SRAM基内存计算相结合的新框架，以减少算术复杂度同时保持精度。 

---
# Towards Quantum-Ready Blockchain Fraud Detection via Ensemble Graph Neural Networks 

**Title (ZH)**: 面向量子计算的区块链欺诈检测 ensemble 图神经网络方法 

**Authors**: M.Z. Haider, Tayyaba Noreen, M. Salman  

**Link**: [PDF](https://arxiv.org/pdf/2509.23101)  

**Abstract**: Blockchain Business applications and cryptocurrencies such as enable secure, decentralized value transfer, yet their pseudonymous nature creates opportunities for illicit activity, challenging regulators and exchanges in anti money laundering (AML) enforcement. Detecting fraudulent transactions in blockchain networks requires models that can capture both structural and temporal dependencies while remaining resilient to noise, imbalance, and adversarial behavior. In this work, we propose an ensemble framework that integrates Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), and Graph Isomorphism Networks (GIN) to enhance blockchain fraud detection. Using the real-world Elliptic dataset, our tuned soft voting ensemble achieves high recall of illicit transactions while maintaining a false positive rate below 1%, beating individual GNN models and baseline methods. The modular architecture incorporates quantum-ready design hooks, allowing seamless future integration of quantum feature mappings and hybrid quantum classical graph neural networks. This ensures scalability, robustness, and long-term adaptability as quantum computing technologies mature. Our findings highlight ensemble GNNs as a practical and forward-looking solution for real-time cryptocurrency monitoring, providing both immediate AML utility and a pathway toward quantum-enhanced financial security analytics. 

**Abstract (ZH)**: 区块链业务应用与加密货币 enables 安全的、去中心化的价值传输，但其假名性质为非法活动创造了机会，挑战着监管机构和交易所的反洗钱（AML）执法工作。检测区块链网络中的虚假交易需要能够捕捉结构和时间依赖性同时对抗噪声、不平衡和恶意行为的模型。在此项工作中，我们提出了一种集成框架，该框架结合了图卷积网络（GCN）、图注意网络（GAT）和图同构网络（GIN），以增强区块链欺诈检测。通过使用实际世界中的Elliptic数据集，我们调优后的软投票集成方法在保持假阳性率低于1%的前提下实现了对非法交易的高召回率，超越了单独的图神经网络模型和基线方法。该模块化架构集成了量子就绪的设计钩子，允许无缝地将量子特征映射和混合量子经典图神经网络集成进来。这确保了随着量子计算技术的发展，系统的可扩展性、鲁棒性和长期适应性。我们的研究结果强调了集成GNNs作为一种实用且前瞻性的解决方案，适用于实时加密货币监控，提供了即时的反洗钱（AML）实用性和通往量子增强的金融安全分析的途径。 

---
# Signal Preserving Weight Initialization for Odd-Sigmoid Activations 

**Title (ZH)**: Odd-Sigmoid 激活函数的信号保� skincare 重初始化 

**Authors**: Hyunwoo Lee, Hayoung Choi, Hyunju Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.23085)  

**Abstract**: Activation functions critically influence trainability and expressivity, and recent work has therefore explored a broad range of nonlinearities. However, activations and weight initialization are interdependent: without an appropriate initialization method, nonlinearities can cause saturation, variance collapse, and increased learning rate sensitivity. We address this by defining an odd sigmoid function class and, given any activation f in this class, proposing an initialization method tailored to f. The method selects a noise scale in closed form so that forward activations remain well dispersed up to a target layer, thereby avoiding collapse to zero or saturation. Empirically, the approach trains reliably without normalization layers, exhibits strong data efficiency, and enables learning for activations under which standard initialization methods (Xavier, He, Orthogonal) often do not converge reliably. 

**Abstract (ZH)**: 激活函数对可训练性和表征能力至关重要，近期研究因此探索了广泛的一系列非线性函数。然而，激活函数和权重初始化相互依赖：没有合适的方法，非线性函数可能导致饱和、方差消失和学习率敏感性增加。为此，我们定义了一个奇数sigmoid函数类，并为该类中的任一激活函数提出了一种定制化的初始化方法。该方法以封闭形式选择噪声尺度，确保前向激活在目标层之前保持良好的分散，从而避免归零或饱和。实证研究表明，该方法在不使用归一化层的情况下能够可靠地训练，表现出强大的数据效率，并且能够在标准初始化方法（Xavier、He、正交）通常无法可靠收敛的情况下学习激活函数。 

---
# Beyond Model Ranking: Predictability-Aligned Evaluation for Time Series Forecasting 

**Title (ZH)**: 超越模型排名：时间序列预测的可预测性对齐评估 

**Authors**: Wanjin Feng, Yuan Yuan, Jingtao Ding, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.23074)  

**Abstract**: In the era of increasingly complex AI models for time series forecasting, progress is often measured by marginal improvements on benchmark leaderboards. However, this approach suffers from a fundamental flaw: standard evaluation metrics conflate a model's performance with the data's intrinsic unpredictability. To address this pressing challenge, we introduce a novel, predictability-aligned diagnostic framework grounded in spectral coherence. Our framework makes two primary contributions: the Spectral Coherence Predictability (SCP), a computationally efficient ($O(N\log N)$) and task-aligned score that quantifies the inherent difficulty of a given forecasting instance, and the Linear Utilization Ratio (LUR), a frequency-resolved diagnostic tool that precisely measures how effectively a model exploits the linearly predictable information within the data. We validate our framework's effectiveness and leverage it to reveal two core insights. First, we provide the first systematic evidence of "predictability drift", demonstrating that a task's forecasting difficulty varies sharply over time. Second, our evaluation reveals a key architectural trade-off: complex models are superior for low-predictability data, whereas linear models are highly effective on more predictable tasks. We advocate for a paradigm shift, moving beyond simplistic aggregate scores toward a more insightful, predictability-aware evaluation that fosters fairer model comparisons and a deeper understanding of model behavior. 

**Abstract (ZH)**: 在时间序列预测中日益复杂的AI模型时代，进展通常通过基准排行榜上的边际改进来衡量。然而，这种方法存在根本性缺陷：标准评估指标将模型性能与数据的固有不可预测性混淆。为解决这一紧迫挑战，我们引入了一个基于频谱相干性的新颖可预测性对齐诊断框架。该框架的两大主要贡献是：频谱相干性可预测性（SCP），一种计算效率高（$O(N\log N)$）且任务对齐的评分，量化给定预测实例的固有难度；以及线性利用率比（LUR），一种频率解析诊断工具，精确测量模型如何有效利用数据中的线性可预测信息。我们验证了该框架的有效性，并利用它揭示了两个核心见解：首先，我们首次系统地证明了“可预测性漂移”，显示任务的预测难度随时间显著变化；其次，我们的评估揭示了一个关键的架构权衡：复杂模型适用于低可预测性数据，而线性模型在更具可预测性的任务上表现更佳。我们倡导 paradigm shift，转向一种更加透彻、可预测性感知的评估方法，以促进更公平的模型比较和更深入的模型行为理解。 

---
# From Evidence to Trajectory: Abductive Reasoning Path Synthesis for Training Retrieval-Augmented Generation Agents 

**Title (ZH)**: 从证据到轨迹：用于训练检索增强生成代理的溯因推理路径合成 

**Authors**: Muzhi Li, Jinhu Qi, Yihong Wu, Minghao Zhao, Liheng Ma, Yifan Li, Xinyu Wang, Yingxue Zhang, Ho-fung Leung, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2509.23071)  

**Abstract**: Retrieval-augmented generation agents development is hindered by the lack of process-level supervision to effectively guide agentic capabilities like task decomposition, retriever invocation, and stepwise decision-making. While reinforcement learning offers a potential solution, it suffers from sparse rewards and the limited reasoning capabilities of large language models (LLMs). Meanwhile, existing data synthesis methods only produce chain-of-thought rationales and fail to model environmental interactions. In this paper, we propose EviPath, an evidence-anchored reasoning path synthesis paradigm for RAG agent development. EviPath comprises: (i) Abductive Subtask Planning, which decomposes the problem into sub-questions and iteratively plans an optimal solution path based on the dependencies between them; (ii) Faithful Sub-question Answering, which uses supporting evidence to construct a proxy environment to generate reasoning thoughts and answers for each sub-question; and (iii) Conversational Fine-Tuning, which formats the complete agent-environment interaction trajectory into a dialogue format suitable for Supervised Fine-Tuning. EviPath allows LLMs to learn complex reasoning and tool-use capabilities directly from synthesized data. Extensive experiments on widely-used question-answering benchmarks show that an 8B parameter model trained with EviPath-synthesized data significantly and consistently outperforms state-of-the-art baselines with a double-digit absolute EM gain of 14.7% in open-domain question answering. 

**Abstract (ZH)**: 基于证据的推理路径合成范式：用于RAG代理开发的EviPath 

---
# Beyond Aggregation: Guiding Clients in Heterogeneous Federated Learning 

**Title (ZH)**: 超越聚合：引导客户端的异构联邦学习 

**Authors**: Zijian Wang, Xiaofei Zhang, Xin Zhang, Yukun Liu, Qiong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23049)  

**Abstract**: Federated learning (FL) is increasingly adopted in domains like healthcare, where data privacy is paramount. A fundamental challenge in these systems is statistical heterogeneity-the fact that data distributions vary significantly across clients (e.g., different hospitals may treat distinct patient demographics). While current FL algorithms focus on aggregating model updates from these heterogeneous clients, the potential of the central server remains under-explored. This paper is motivated by a healthcare scenario: could a central server not only build a model but also guide a new patient to the hospital best equipped for their specific condition? We generalize this idea to propose a novel paradigm for FL systems where the server actively guides the allocation of new tasks or queries to the most appropriate client in the network. To enable this, we introduce an empirical likelihood-based framework that simultaneously addresses two goals: (1) learning effective local models on each client, and (2) finding the best matching client for a new query. Empirical results demonstrate the framework's effectiveness on benchmark datasets, showing improvements in both model accuracy and the precision of client guidance compared to standard FL approaches. This work opens a new direction for building more intelligent and resource-efficient federated systems that leverage heterogeneity as a feature, not just a bug. Code is available at this https URL. 

**Abstract (ZH)**: 联邦学习（FL）在医疗健康等重视数据隐私的领域被越来越广泛地采用。这些系统中的一个基本挑战是统计异质性——即各客户端数据分布差异显著（例如，不同的医院可能治疗不同的患者群体）。尽管当前的联邦学习算法主要关注从这些异质客户端聚合模型更新，中央服务器的潜力仍然未被充分探索。本文受到医疗健康场景的启发：中央服务器是否不仅能构建模型，还能指导新患者前往最合适的医院进行治疗？我们推广这一思想，提出了一种新型的联邦学习范式，其中服务器积极指导新任务或查询在网络中最合适的客户端处进行分配。为此，我们引入了一种经验似然为基础的框架，同时实现两个目标：1) 在每个客户端上学习有效的本地模型，2) 为新的查询找到最合适的匹配客户端。实验证明，该框架在基准数据集上显示了比标准联邦学习方法更高的模型准确性和更精准的客户端指导效果。这项工作开启了利用异质性作为特征而非缺陷来构建更智能和资源高效的联邦系统的新的研究方向。代码可从以下链接获取。 

---
# IsingFormer: Augmenting Parallel Tempering With Learned Proposals 

**Title (ZH)**: IsingFormer: 用学习得到的提案增强平行退火方法 

**Authors**: Saleh Bunaiyan, Corentin Delacour, Shuvro Chowdhury, Kyle Lee, Kerem Y. Camsari  

**Link**: [PDF](https://arxiv.org/pdf/2509.23043)  

**Abstract**: Markov Chain Monte Carlo (MCMC) underlies both statistical physics and combinatorial optimization, but mixes slowly near critical points and in rough landscapes. Parallel Tempering (PT) improves mixing by swapping replicas across temperatures, yet each replica still relies on slow local updates to change its configuration. We introduce IsingFormer, a Transformer trained on equilibrium samples that can generate entire spin configurations resembling those from the target distribution. These uncorrelated samples are used as proposals for global moves within a Metropolis step in PT, complementing the usual single-spin flips. On 2D Ising models (sampling), IsingFormer reproduces magnetization and free-energy curves and generalizes to unseen temperatures, including the critical region. Injecting even a single proposal sharply reduces equilibration time, replacing thousands of local updates. On 3D spin glasses (optimization), PT enhanced with IsingFormer finds substantially lower-energy states, demonstrating how global moves accelerate search in rugged landscapes. Finally, applied to integer factorization encoded as Ising problems, IsingFormer trained on a limited set of semiprimes transfers successfully to unseen semiprimes, boosting success rates beyond the training distribution. Since factorization is a canonical hard benchmark, this ability to generalize across instances highlights the potential of learning proposals that move beyond single problems to entire families of instances. The IsingFormer demonstrates that Monte Carlo methods can be systematically accelerated by neural proposals that capture global structure, yielding faster sampling and stronger performance in combinatorial optimization. 

**Abstract (ZH)**: Markov链蒙特卡洛（MCMC）既存在于统计物理中也存在于组合优化中，但在临界点附近和崎岖的景观中混合速度缓慢。并行温度调谐（PT）通过在不同温度下交换复制品来提高混合效果，但每个复制品仍然依赖于缓慢的局部更新来改变其配置。我们介绍了IsingFormer，这是一种在平衡样本上训练的Transformer，能够生成类似于目标分布的完整自旋配置。这些未关联的样本被用作PT中全局移动的提案，在Metropolis步骤中补充了传统的单个自旋翻转。在2D自旋玻璃模型（采样）中，IsingFormer重现了磁化率和自由能曲线，并可以在未见过的温度下泛化，包括临界区域。即使注入一个提案也能显著减少平衡时间，取代数千次的局部更新。在3D自旋玻璃模型（优化）中，增强了IsingFormer的PT找到了显著更低能量的状态，证明了全局移动在崎岖景观中如何加速搜索。最后，将其应用于整数因子分解编码为自旋玻璃问题，IsingFormer在有限的半素数集上训练后可以成功迁移到未见过的半素数上，提升了解决率超出训练分布。由于因子分解是经典硬基准测试，这种在实例之间泛化的能力突显了学习超越单一问题的全局结构移动提案的潜力。IsingFormer展示了通过捕捉全局结构的神经提案系统加速蒙特卡洛方法的可能性，从而实现更快的采样和更强的组合优化性能。 

---
# DPFNAS: Differential Privacy-Enhanced Federated Neural Architecture Search for 6G Edge Intelligence 

**Title (ZH)**: DPFNAS：增强差分隐私的6G边缘智能联邦神经架构搜索 

**Authors**: Yang Lv, Jin Cao, Ben Niu, Zhe Sun, Fengwei Wang, Fenghua Li, Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.23030)  

**Abstract**: The Sixth-Generation (6G) network envisions pervasive artificial intelligence (AI) as a core goal, enabled by edge intelligence through on-device data utilization. To realize this vision, federated learning (FL) has emerged as a key paradigm for collaborative training across edge devices. However, the sensitivity and heterogeneity of edge data pose key challenges to FL: parameter sharing risks data reconstruction, and a unified global model struggles to adapt to diverse local distributions. In this paper, we propose a novel federated learning framework that integrates personalized differential privacy (DP) and adaptive model design. To protect training data, we leverage sample-level representations for knowledge sharing and apply a personalized DP strategy to resist reconstruction attacks. To ensure distribution-aware adaptation under privacy constraints, we develop a privacy-aware neural architecture search (NAS) algorithm that generates locally customized architectures and hyperparameters. To the best of our knowledge, this is the first personalized DP solution tailored for representation-based FL with theoretical convergence guarantees. Our scheme achieves strong privacy guarantees for training data while significantly outperforming state-of-the-art methods in model performance. Experiments on benchmark datasets such as CIFAR-10 and CIFAR-100 demonstrate that our scheme improves accuracy by 6.82\% over the federated NAS method PerFedRLNAS, while reducing model size to 1/10 and communication cost to 1/20. 

**Abstract (ZH)**: 第六代（6G）网络愿景通过边缘智能实现泛在的人工智能，边缘设备上的数据利用是关键目标。为实现这一愿景，联邦学习（FL）已成为跨边缘设备协作训练的关键范式。然而，边缘数据的敏感性和异质性给FL带来了关键挑战：参数共享存在数据重构风险，统一的全局模型难以适应多样化的本地分布。在本文中，我们提出了一种结合个性化差分隐私（DP）和自适应模型设计的新型联邦学习框架。为了保护训练数据，我们利用样本级表示进行知识共享，并应用个性化DP策略以抵御重构攻击。为了在隐私约束下确保分布感知的适应性，我们开发了一种隐私感知神经架构搜索（NAS）算法，生成本地定制的架构和超参数。据我们所知，这是第一个针对表示驱动的FL的个性化DP解决方案，并具有理论收敛保证。我们的方案在保护训练数据隐私方面表现出强大的保证，同时在模型性能上显著优于当前最先进的方法。基准数据集（如CIFAR-10和CIFAR-100）上的实验表明，与PerFedRLNAS方法相比，我们的方案在准确率上提高了6.82%，同时模型大小减少了10倍，通信成本减少了20倍。 

---
# MoE-PHDS: One MoE checkpoint for flexible runtime sparsity 

**Title (ZH)**: MoE-PHDS: 一个MoE检查点用于灵活的运行时稀疏性 

**Authors**: Lauren. A Hannah, Soheil Zibakhsh, Kumari Nishu, Arnav Kundu, Mohammad Samragh Razlighi, Mehrdad Farajtabar, Minsik Cho  

**Link**: [PDF](https://arxiv.org/pdf/2509.23012)  

**Abstract**: Sparse Mixtures of Experts (MoEs) are typically trained to operate at a fixed sparsity level, e.g. $k$ in a top-$k$ gating function. This global sparsity level determines an operating point on the accuracy/latency curve; currently, meeting multiple efficiency targets means training and maintaining multiple models. This practice complicates serving, increases training and maintenance costs, and limits flexibility in meeting diverse latency, efficiency, and energy requirements. We show that pretrained MoEs are more robust to runtime sparsity shifts than commonly assumed, and introduce MoE-PHDS ({\bf P}ost {\bf H}oc {\bf D}eclared {\bf S}parsity), a lightweight SFT method that turns a single checkpoint into a global sparsity control surface. PHDS mixes training across sparsity levels and anchors with a short curriculum at high sparsity, requiring no architectural changes. The result is predictable accuracy/latency tradeoffs from one model: practitioners can ``dial $k$'' at inference time without swapping checkpoints, changing architecture, or relying on token-level heuristics. Experiments on OLMoE-1B-7B-0125, Qwen1.5-MoE-A2.7B, and proprietary models fit on multiple operating points show that PHDS matches or exceeds well-specified oracle models, improves cross-sparsity agreement by up to 22\% vs. well-specified oracle models, and enables simplified, flexible runtime MoE deployment by making global sparsity a first-class serving primitive. 

**Abstract (ZH)**: 预训练的专家混合模型更具鲁棒性：基于后显式稀疏度的轻量级微调方法（MoE-PHDS） 

---
# Functional Critic Modeling for Provably Convergent Off-Policy Actor-Critic 

**Title (ZH)**: 证明收敛的离策 Actor-Critc 模型中的功能评论者建模 

**Authors**: Qinxun Bai, Yuxuan Han, Wei Xu, Zhengyuan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.22964)  

**Abstract**: Off-policy reinforcement learning (RL) with function approximation offers an effective way to improve sample efficiency by reusing past experience. Within this setting, the actor-critic (AC) framework has achieved strong empirical success. However, both the critic and actor learning is challenging for the off-policy AC methods: first of all, in addition to the classic "deadly triad" instability of off-policy evaluation, it also suffers from a "moving target" problem, where the policy being evaluated changes continually; secondly, actor learning becomes less efficient due to the difficulty of estimating the exact off-policy policy gradient. The first challenge essentially reduces the problem to repeatedly performing off-policy evaluation for changing policies. For the second challenge, the off-policy policy gradient theorem requires a complex and often impractical algorithm to estimate an additional emphasis critic, which is typically neglected in practice, thereby reducing to the on-policy policy gradient as an approximation. In this work, we introduce a novel concept of functional critic modeling, which leads to a new AC framework that addresses both challenges for actor-critic learning under the deadly triad setting. We provide a theoretical analysis in the linear function setting, establishing the provable convergence of our framework, which, to the best of our knowledge, is the first convergent off-policy target-based AC algorithm. From a practical perspective, we further propose a carefully designed neural network architecture for the functional critic modeling and demonstrate its effectiveness through preliminary experiments on widely used RL tasks from the DeepMind Control Benchmark. 

**Abstract (ZH)**: 带有函数逼近的离策 reinforcement learning中的actor-critic框架：解决致命三角问题的功能性critic建模 

---
# What Matters More For In-Context Learning under Matched Compute Budgets: Pretraining on Natural Text or Incorporating Targeted Synthetic Examples? 

**Title (ZH)**: 在匹配计算预算条件下，影响基于上下文学习更为重要的因素是自然文本预训练还是融入针对性合成例证？ 

**Authors**: Mohammed Sabry, Anya Belz  

**Link**: [PDF](https://arxiv.org/pdf/2509.22947)  

**Abstract**: Does explicitly exercising the induction circuit during pretraining improve in-context learning (ICL), or is natural text sufficient when compute is held constant (iso-FLOPs)? To test whether targeted synthetic data can accelerate induction-head emergence and enhance ICL, we introduce Bi-Induct, a lightweight curriculum that injects forward-copy (Induction), backward-copy (Anti), or a balanced mix into the pretraining stream. We train models from 0.13B to 1B parameters under iso-FLOPs, evaluating (i) few-shot ICL benchmarks, (ii) head-level telemetry, and (iii) held-out language modeling perplexity. Our findings challenge the assumption that early induction circuit activation directly improves ICL. While Bi-Induct accelerates induction-head emergence at small scales, this does not consistently yield stronger generalization. On standard LM benchmarks, Bi-Induct matches natural-only training; on function-style ICL probes, the 1B natural-only performs best. Stress tests (e.g., label permutation, HITS@1 vs. HITS@3, 1 vs. 10 shots) preserve these trends. Telemetry shows larger natural-only models develop broader, earlier induction heads without explicit induction patterns. Anti-induction data fails to elicit meaningful activation. Perplexity penalties from synthetic data shrink with scale, suggesting larger models can absorb non-natural patterns with minimal cost. Crucially, ablating the top 2% of induction heads degrades ICL more than random ablations, especially for natural-only models, indicating more centralized, load-bearing circuits. Bi-Induct variants exhibit more redundant induction activity, implying different circuit utilization. Overall, inducing activation is not sufficient: ICL gains depend on these circuits becoming functionally necessary. These results underscore mechanism-aware pretraining diagnostics and data mixtures that foster load-bearing, not merely present, structure. 

**Abstract (ZH)**: does explicitly exercising the induction circuit during pretraining improve in-context learning (icl), or is natural text sufficient when compute is held constant (iso-flops)? introducing bi-induct to test the acceleration of induction-head emergence and enhancement of icl 

---
# Unsupervised Speech Enhancement using Data-defined Priors 

**Title (ZH)**: 基于数据定义先验的无监督语音增强 

**Authors**: Dominik Klement, Matthew Maciejewski, Sanjeev Khudanpur, Jan Černocký, Lukáš Burget  

**Link**: [PDF](https://arxiv.org/pdf/2509.22942)  

**Abstract**: The majority of deep learning-based speech enhancement methods require paired clean-noisy speech data. Collecting such data at scale in real-world conditions is infeasible, which has led the community to rely on synthetically generated noisy speech. However, this introduces a gap between the training and testing phases. In this work, we propose a novel dual-branch encoder-decoder architecture for unsupervised speech enhancement that separates the input into clean speech and residual noise. Adversarial training is employed to impose priors on each branch, defined by unpaired datasets of clean speech and, optionally, noise. Experimental results show that our method achieves performance comparable to leading unsupervised speech enhancement approaches. Furthermore, we demonstrate the critical impact of clean speech data selection on enhancement performance. In particular, our findings reveal that performance may appear overly optimistic when in-domain clean speech data are used for prior definition -- a practice adopted in previous unsupervised speech enhancement studies. 

**Abstract (ZH)**: 基于深度学习的无监督语音增强方法：一种分离输入为干净语音和残余噪声的双分支编码器-解码器架构 

---
# Compute-Optimal Quantization-Aware Training 

**Title (ZH)**: 计算最优量化感知训练 

**Authors**: Aleksandr Dremov, David Grangier, Angelos Katharopoulos, Awni Hannun  

**Link**: [PDF](https://arxiv.org/pdf/2509.22935)  

**Abstract**: Quantization-aware training (QAT) is a leading technique for improving the accuracy of quantized neural networks. Previous work has shown that decomposing training into a full-precision (FP) phase followed by a QAT phase yields superior accuracy compared to QAT alone. However, the optimal allocation of compute between the FP and QAT phases remains unclear. We conduct extensive experiments with various compute budgets, QAT bit widths, and model sizes from 86.0M to 2.2B to investigate how different QAT durations impact final performance. We demonstrate that, contrary to previous findings, the loss-optimal ratio of QAT to FP training increases with the total amount of compute. Moreover, the optimal fraction can be accurately predicted for a wide range of model sizes and quantization widths using the tokens-per-parameter-byte statistic. From experimental data, we derive a loss scaling law that predicts both optimal QAT ratios and final model performance across different QAT/FP compute allocation strategies and QAT bit widths. We use the scaling law to make further predictions, which we verify experimentally, including which QAT bit width is optimal under a given memory constraint and how QAT accuracy with different bit widths compares to full-precision model accuracy. Additionally, we propose a novel cooldown and QAT fusion approach that performs learning rate decay jointly with quantization-aware training, eliminating redundant full-precision model updates and achieving significant compute savings. These findings provide practical insights into efficient QAT planning and enable the training of higher-quality quantized models with the same compute budget. 

**Abstract (ZH)**: Quantization-aware训练（QAT）是一种提高量化神经网络准确性的领先技术。先前的研究表明，将训练分解为全精度（FP）阶段和QAT阶段可以比单独使用QAT获得更高的精度。然而，FP和QAT阶段之间的计算分配仍不清楚。我们通过各种计算预算、QAT比特宽和从86.0M到2.2B的模型规模进行了大量实验，研究不同的QAT持续时间对最终性能的影响。我们证明，与之前的研究发现相反，QAT与FP训练的理想比例随总计算量的增加而增加。此外，使用tokens-per-parameter-byte统计值可以准确预测广泛模型规模和量化宽度下的最优比例。从实验数据中，我们推导出一个损失放大定律，该定律可以预测不同QAT/FP计算分配策略和QAT比特宽下的最优QAT比例和最终模型性能。我们使用该定律进行进一步预测，并通过实验验证，包括在给定的内存约束下哪种QAT比特宽是最优的，以及不同比特宽下的QAT准确性与全精度模型准确性之间的比较。此外，我们提出了一种新的冷却和QAT融合方法，该方法结合了量化感知训练和学习率衰减，消除了冗余的全精度模型更新，并实现了显著的计算节省。这些发现为有效的QAT规划提供了实用见解，并使研究人员能够在相同的计算预算下训练出更高质量的量化模型。 

---
# MonoCon: A general framework for learning ultra-compact high-fidelity representations using monotonicity constraints 

**Title (ZH)**: MonoCon：一种使用单调性约束学习超紧凑高保真表示的一般框架 

**Authors**: Shreyas Gokhale  

**Link**: [PDF](https://arxiv.org/pdf/2509.22931)  

**Abstract**: Learning high-quality, robust, efficient, and disentangled representations is a central challenge in artificial intelligence (AI). Deep metric learning frameworks tackle this challenge primarily using architectural and optimization constraints. Here, we introduce a third approach that instead relies on $\textit{functional}$ constraints. Specifically, we present MonoCon, a simple framework that uses a small monotonic multi-layer perceptron (MLP) head attached to any pre-trained encoder. Due to co-adaptation between encoder and head guided by contrastive loss and monotonicity constraints, MonoCon learns robust, disentangled, and highly compact embeddings at a practically negligible performance cost. On the CIFAR-100 image classification task, MonoCon yields representations that are nearly 9x more compact and 1.5x more robust than the fine-tuned encoder baseline, while retaining 99\% of the baseline's 5-NN classification accuracy. We also report a 3.4x more compact and 1.4x more robust representation on an SNLI sentence similarity task for a marginal reduction in the STSb score, establishing MonoCon as a general domain-agnostic framework. Crucially, these robust, ultra-compact representations learned via functional constraints offer a unified solution to critical challenges in disparate contexts ranging from edge computing to cloud-scale retrieval. 

**Abstract (ZH)**: 学习高质量、稳健、高效且解耦的表示是人工智能中的一个核心挑战。基于功能约束的MonoCon框架 

---
# From Noise to Knowledge: A Comparative Study of Acoustic Anomaly Detection Models in Pumped-storage Hydropower Plants 

**Title (ZH)**: 从噪声到知识： Pumped-storage Hydropower Plants 中 acoustic 异常检测模型的比较研究 

**Authors**: Karim Khamaisi, Nicolas Keller, Stefan Krummenacher, Valentin Huber, Bernhard Fässler, Bruno Rodrigues  

**Link**: [PDF](https://arxiv.org/pdf/2509.22881)  

**Abstract**: In the context of industrial factories and energy producers, unplanned outages are highly costly and difficult to service. However, existing acoustic-anomaly detection studies largely rely on generic industrial or synthetic datasets, with few focused on hydropower plants due to limited access. This paper presents a comparative analysis of acoustic-based anomaly detection methods, as a way to improve predictive maintenance in hydropower plants. We address key challenges in the acoustic preprocessing under highly noisy conditions before extracting time- and frequency-domain features. Then, we benchmark three machine learning models: LSTM AE, K-Means, and OC-SVM, which are tested on two real-world datasets from the Rodundwerk II pumped-storage plant in Austria, one with induced anomalies and one with real-world conditions. The One-Class SVM achieved the best trade-off of accuracy (ROC AUC 0.966-0.998) and minimal training time, while the LSTM autoencoder delivered strong detection (ROC AUC 0.889-0.997) at the expense of higher computational cost. 

**Abstract (ZH)**: 基于声学异常检测方法在水力发电厂预测性维护中的比较研究 

---
# Scalable Wi-Fi RSS-Based Indoor Localization via Automatic Vision-Assisted Calibration 

**Title (ZH)**: 基于自动视觉辅助校准的可扩展Wi-Fi RSS室内外定位 

**Authors**: Abdulkadir Bilge, Erdem Ergen, Burak Soner, Sinem Coleri  

**Link**: [PDF](https://arxiv.org/pdf/2509.22869)  

**Abstract**: Wi-Fi-based positioning promises a scalable and privacy-preserving solution for location-based services in indoor environments such as malls, airports, and campuses. RSS-based methods are widely deployable as RSS data is available on all Wi-Fi-capable devices, but RSS is highly sensitive to multipath, channel variations, and receiver characteristics. While supervised learning methods offer improved robustness, they require large amounts of labeled data, which is often costly to obtain. We introduce a lightweight framework that solves this by automating high-resolution synchronized RSS-location data collection using a short, camera-assisted calibration phase. An overhead camera is calibrated only once with ArUco markers and then tracks a device collecting RSS data from broadcast packets of nearby access points across Wi-Fi channels. The resulting (x, y, RSS) dataset is used to automatically train mobile-deployable localization algorithms, avoiding the privacy concerns of continuous video monitoring. We quantify the accuracy limits of such vision-assisted RSS data collection under key factors such as tracking precision and label synchronization. Using the collected experimental data, we benchmark traditional and supervised learning approaches under varying signal conditions and device types, demonstrating improved accuracy and generalization, validating the utility of the proposed framework for practical use. All code, tools, and datasets are released as open source. 

**Abstract (ZH)**: 基于Wi-Fi的定位技术在商场、机场和校园等室内环境中提供了可扩展且保护隐私的解决方案。基于RSS的方法广泛部署，因为所有Wi-Fi设备都能提供RSS数据，但RSS对多径传播、信道变化和接收器特性高度敏感。尽管监督学习方法提高了鲁棒性，但它们需要大量标注数据，这通常成本高昂。我们提出了一种轻量级框架，通过使用短的、摄像头辅助的校准阶段自动收集高分辨率同步RSS-位置数据来解决这一问题。摄像机仅需一次校准即可使用ArUco标记，并随后跟踪从附近接入点广播数据包中收集RSS数据的设备，跨越多个Wi-Fi信道。生成的(x, y, RSS)数据集用于自动训练可移动部署的定位算法，避免了连续视频监控带来的隐私问题。我们量化了在关键因素如跟踪精度和标签同步下的这种视觉辅助RSS数据收集的准确性极限。利用收集的实验数据，我们在不同信号条件和设备类型下对传统和监督学习方法进行了基准测试，证明了准确性与泛化能力的提升，并验证了所提框架在实际应用中的实用性。所有代码、工具和数据集均作为开源发布。 

---
# Observation-Free Attacks on Online Learning to Rank 

**Title (ZH)**: 无需观察的在线学习排序攻击 

**Authors**: Sameep Chattopadhyay, Nikhil Karamchandani, Sharayu Mohair  

**Link**: [PDF](https://arxiv.org/pdf/2509.22855)  

**Abstract**: Online learning to rank (OLTR) plays a critical role in information retrieval and machine learning systems, with a wide range of applications in search engines and content recommenders. However, despite their extensive adoption, the susceptibility of OLTR algorithms to coordinated adversarial attacks remains poorly understood. In this work, we present a novel framework for attacking some of the widely used OLTR algorithms. Our framework is designed to promote a set of target items so that they appear in the list of top-K recommendations for T - o(T) rounds, while simultaneously inducing linear regret in the learning algorithm. We propose two novel attack strategies: CascadeOFA for CascadeUCB1 and PBMOFA for PBM-UCB . We provide theoretical guarantees showing that both strategies require only O(log T) manipulations to succeed. Additionally, we supplement our theoretical analysis with empirical results on real-world data. 

**Abstract (ZH)**: 在线学习排序（OLTR）在信息检索和机器学习系统中扮演着关键角色，广泛应用于搜索引擎和内容推荐系统。然而，尽管OLTR算法被广泛采用，它们对协调式 adversarial 攻击的脆弱性仍然不甚了解。在本文中，我们提出了一种新的框架，用于攻击一些广泛使用的OLTR算法。我们的框架旨在促进一组目标项，使其在T-至T rounds内出现在前K项推荐列表中，同时在学习算法中诱导线性后悔。我们提出了两种新的攻击策略：CascadeOFA用于CascadeUCB1，PBMOFA用于PBM-UCB。我们提供了理论保证，表明这两种策略只需O(log T)次操纵即可成功。此外，我们还通过实际数据的实证结果补充了我们的理论分析。 

---
# Patient-specific Biomolecular Instruction Tuning 

**Title (ZH)**: 患者特异性生物分子指令调谐 

**Authors**: Irsyad Adam, Zekai Chen, David Laub, Shaun Porwal, Arda Pekis, Kevin Brown  

**Link**: [PDF](https://arxiv.org/pdf/2509.22853)  

**Abstract**: Proteomics data is essential to pathogenic understanding of a disease phenotype. In cancer, analysis of molecular signatures enables precision medicine through the identification of biological processes that drive individualized tumor progression, therapeutic resistance, and clinical heterogeneity. Recent advances in multimodal large language models (LLMs) have shown remarkable capacity to integrate and reason across heterogeneous data modalities. However, performing multi-modal language modeling for molecular understanding of patient-specific proteomics remains a significant challenge due to two barriers: (1) the lack of instruction-tuning datasets that enable clinical interpretation from proteomics data, and (2) the absence of language modeling architectures designed to capture the rich heterogeneity of molecular data. In this work, we introduce CPTAC-PROTSTRUCT, the first instruction tuning dataset for molecular understanding of oncology, comprising over 400k open-ended examples derived from individualized proteomic profiles curated from the largest national proteomics cancer study (CPTAC). Additionally, we propose KRONOS (Knowledge Representation of patient Omics Networks in Oncology via Structured tuning), a novel graph-LLM framework that leverages molecular interaction topology with proteomics to learn patient-specific graph representations for enhanced clinical reasoning. We show that KRONOS achieves competitive performance across benchmark clinical tasks, including molecular classification, temporal trajectory modeling, and tumor stage prediction from proteomics data. Ultimately, this approach empowers LLMs to understand patient-level pathogenesis, advancing precision medicine through more accurate diagnosis, prognosis, and treatment stratification. 

**Abstract (ZH)**: 蛋白质组学数据对于理解疾病的表型致病机制至关重要。在癌症中，分子标记的分析能够通过识别驱动个体肿瘤进展、治疗抵抗和临床异质性的生物过程，实现精准医疗。近期，多模态大型语言模型（LLMs）的进步展现了其整合和跨异质数据模态推理的巨大能力。然而，由于两个障碍，将多模态语言模型应用于患者特异性蛋白质组学的分子理解仍然面临重大挑战：（1）缺乏能够从蛋白质组学数据中进行临床解释的指令调优数据集；（2）缺乏能够捕捉分子数据丰富异质性的语言模型架构。在本文中，我们介绍了CPTAC-PROTSTRUCT，这是首个用于肿瘤学中分子理解的指令调优数据集，包含来自最大国家级蛋白质组学癌症研究（CPTAC）中个体化蛋白质谱绘制的超过40万个开放性示例。此外，我们提出了KRONOS（肿瘤学中基于结构调优的患者组学网络知识表示），这是一种新颖的图-LLM框架，通过利用蛋白质组学中的分子相互作用拓扑结构来学习患者特异性的图表示，以增强临床推理。我们展示了KRONOS在基准临床任务中的竞争力，包括分子分类、时间轨迹建模和从蛋白质组学数据中预测肿瘤分期。最终，这种方法使大型语言模型能够理解患者水平的病理机制，从而推动更加准确的诊断、预后和治疗分层，以实现精准医疗。 

---
# Boundary on the Table: Efficient Black-Box Decision-Based Attacks for Structured Data 

**Title (ZH)**: 边界之上：面向结构化数据的高效黑盒决策基攻击 

**Authors**: Roie Kazoom, Yuval Ratzabi, Etamar Rothstein, Ofer Hadar  

**Link**: [PDF](https://arxiv.org/pdf/2509.22850)  

**Abstract**: Adversarial robustness in structured data remains an underexplored frontier compared to vision and language domains. In this work, we introduce a novel black-box, decision-based adversarial attack tailored for tabular data. Our approach combines gradient-free direction estimation with an iterative boundary search, enabling efficient navigation of discrete and continuous feature spaces under minimal oracle access. Extensive experiments demonstrate that our method successfully compromises nearly the entire test set across diverse models, ranging from classical machine learning classifiers to large language model (LLM)-based pipelines. Remarkably, the attack achieves success rates consistently above 90%, while requiring only a small number of queries per instance. These results highlight the critical vulnerability of tabular models to adversarial perturbations, underscoring the urgent need for stronger defenses in real-world decision-making systems. 

**Abstract (ZH)**: 结构化数据的对抗鲁棒性相较于视觉和语言领域仍是一个未充分探索的前沿问题。在本文中，我们提出了一种针对表格数据的新型黑盒决策型对抗攻击方法。该方法结合了无梯度方向估计与迭代边界搜索，能够在最少的oracle访问下高效导航离散和连续特征空间。广泛实验表明，我们的方法成功地几乎将整个测试集中的多种模型（从经典机器学习分类器到基于大型语言模型的管道）攻击成功率保持在90%以上。这些结果强调了表格模型对对抗扰动的严重易受攻击性，突显了急需在实际决策系统中加强防御措施的重要性。 

---
# MTRec: Learning to Align with User Preferences via Mental Reward Models 

**Title (ZH)**: MTRec: 基于心智奖励模型的学习用户偏好多模态对齐方法 

**Authors**: Mengchen Zhao, Yifan Gao, Yaqing Hou, Xiangyang Li, Pengjie Gu, Zhenhua Dong, Ruiming Tang, Yi Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.22807)  

**Abstract**: Recommendation models are predominantly trained using implicit user feedback, since explicit feedback is often costly to obtain. However, implicit feedback, such as clicks, does not always reflect users' real preferences. For example, a user might click on a news article because of its attractive headline, but end up feeling uncomfortable after reading the content. In the absence of explicit feedback, such erroneous implicit signals may severely mislead recommender systems. In this paper, we propose MTRec, a novel sequential recommendation framework designed to align with real user preferences by uncovering their internal satisfaction on recommended items. Specifically, we introduce a mental reward model to quantify user satisfaction and propose a distributional inverse reinforcement learning approach to learn it. The learned mental reward model is then used to guide recommendation models to better align with users' real preferences. Our experiments show that MTRec brings significant improvements to a variety of recommendation models. We also deploy MTRec on an industrial short video platform and observe a 7 percent increase in average user viewing time. 

**Abstract (ZH)**: 一种新型顺序推荐框架MTRec：通过揭示用户对推荐项目的内部满意度来引导推荐模型更好地契合用户的真实偏好 

---
# Generative Modeling and Decision Fusion for Unknown Event Detection and Classification Using Synchrophasor Data 

**Title (ZH)**: 基于同步相量数据的未知事件检测与分类的生成建模和决策融合 

**Authors**: Yi Hu, Zheyuan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.22795)  

**Abstract**: Reliable detection and classification of power system events are critical for maintaining grid stability and situational awareness. Existing approaches often depend on limited labeled datasets, which restricts their ability to generalize to rare or unseen disturbances. This paper proposes a novel framework that integrates generative modeling, sliding-window temporal processing, and decision fusion to achieve robust event detection and classification using synchrophasor data. A variational autoencoder-generative adversarial network is employed to model normal operating conditions, where both reconstruction error and discriminator error are extracted as anomaly indicators. Two complementary decision strategies are developed: a threshold-based rule for computational efficiency and a convex hull-based method for robustness under complex error distributions. These features are organized into spatiotemporal detection and classification matrices through a sliding-window mechanism, and an identification and decision fusion stage integrates the outputs across PMUs. This design enables the framework to identify known events while systematically classifying previously unseen disturbances into a new category, addressing a key limitation of supervised classifiers. Experimental results demonstrate state-of-the-art accuracy, surpassing machine learning, deep learning, and envelope-based baselines. The ability to recognize unknown events further highlights the adaptability and practical value of the proposed approach for wide-area event analysis in modern power systems. 

**Abstract (ZH)**: 可靠的电力系统事件检测与分类对于维持电网稳定性和态势感知至关重要。现有方法往往依赖于有限的标注数据集，限制了它们对罕见或未见干扰的泛化能力。本文提出了一种新颖的框架，该框架结合了生成建模、滑动窗口时间处理和决策融合，利用同步相量数据实现稳健的事件检测与分类。采用变分自编码器-生成对抗网络来建模正常运行状态，其中重构误差和鉴别器误差被提取为异常指标。开发了两种互补的决策策略：基于阈值的规则以提高计算效率，以及基于凸包的方法以在复杂误差分布下提高鲁棒性。这些特征通过滑动窗口机制组织成时空检测与分类矩阵，并在识别与决策融合阶段集成PMU输出。该设计使框架能够识别已知事件，并系统地将未见干扰分类到新类别中，解决了监督分类器的一个关键局限性。实验结果表明，该方法在准确度上达到最新技术水平，超越了机器学习、深度学习和包络基线方法。能够识别未知事件进一步突显了所提出方法在现代电力系统广域事件分析中的适应性和实际价值。 

---
# Differentially Private Two-Stage Gradient Descent for Instrumental Variable Regression 

**Title (ZH)**: 差分隐私两阶段梯度下降法在工具变量回归中的应用 

**Authors**: Haodong Liang, Yanhao Jin, Krishnakumar Balasubramanian, Lifeng Lai  

**Link**: [PDF](https://arxiv.org/pdf/2509.22794)  

**Abstract**: We study instrumental variable regression (IVaR) under differential privacy constraints. Classical IVaR methods (like two-stage least squares regression) rely on solving moment equations that directly use sensitive covariates and instruments, creating significant risks of privacy leakage and posing challenges in designing algorithms that are both statistically efficient and differentially private. We propose a noisy two-state gradient descent algorithm that ensures $\rho$-zero-concentrated differential privacy by injecting carefully calibrated noise into the gradient updates. Our analysis establishes finite-sample convergence rates for the proposed method, showing that the algorithm achieves consistency while preserving privacy. In particular, we derive precise bounds quantifying the trade-off among privacy parameters, sample size, and iteration-complexity. To the best of our knowledge, this is the first work to provide both privacy guarantees and provable convergence rates for instrumental variable regression in linear models. We further validate our theoretical findings with experiments on both synthetic and real datasets, demonstrating that our method offers practical accuracy-privacy trade-offs. 

**Abstract (ZH)**: 我们研究差分隐私约束下的工具变量回归（IVaR）。经典的工具变量回归方法（如两阶段最小平方法）依赖于直接使用敏感协变量和工具变量求解矩方程，这产生了重大的隐私泄露风险，并给设计同时具备统计效率和差分隐私性的算法带来了挑战。我们提出了一种噪声二状态梯度下降算法，通过在梯度更新中注入精心校准的噪声来确保$\rho$-零集中差分隐私。我们的分析建立了所提方法的有限样本收敛速率，表明该算法既能保持一致性又能保护隐私。特别地，我们推导出了精确界定量化的隐私参数、样本量和迭代复杂度之间的权衡。据我们所知，这是首个同时提供工具变量回归在线性模型中差分隐私保证和可证明收敛速率的工作。我们还通过在合成数据集和真实数据集上的实验验证了我们的理论发现，证明了我们的方法提供了实用的准确性和隐私之间的权衡。 

---
# A theoretical guarantee for SyncRank 

**Title (ZH)**: SyncRank的理论保证 

**Authors**: Yang Rao  

**Link**: [PDF](https://arxiv.org/pdf/2509.22766)  

**Abstract**: We present a theoretical and empirical analysis of the SyncRank algorithm for recovering a global ranking from noisy pairwise comparisons. By adopting a complex-valued data model where the true ranking is encoded in the phases of a unit-modulus vector, we establish a sharp non-asymptotic recovery guarantee for the associated semidefinite programming (SDP) relaxation. Our main theorem characterizes a critical noise threshold - scaling as sigma = O(sqrt(n / log n)) - below which SyncRank achieves exact ranking recovery with high probability. Extensive experiments under this model confirm the theoretical predictions and demonstrate the algorithm's robustness across varying problem sizes and noise regimes. 

**Abstract (ZH)**: 我们提出了一种针对噪声双边比较进行全局排名恢复的SyncRank算法的理论和实证分析。通过采用复值数据模型，其中真实的排名编码在单位模向量的相位中，我们建立了与之相关的半定规划（SDP）松弛的精确非渐近恢复保证。我们的主要定理刻画了一个关键的噪声阈值——约为σ=O(√(n/logn))—在该阈值以下，SyncRank以高概率实现精确的排名恢复。在该模型下的大量实验证明了理论预测，并展示了该算法在不同问题规模和噪声条件下的稳健性。 

---
# Red Teaming Quantum-Resistant Cryptographic Standards: A Penetration Testing Framework Integrating AI and Quantum Security 

**Title (ZH)**: 红队测试量子抗性加密标准：融合AI与量子安全的渗透测试框架 

**Authors**: Petar Radanliev  

**Link**: [PDF](https://arxiv.org/pdf/2509.22757)  

**Abstract**: This study presents a structured approach to evaluating vulnerabilities within quantum cryptographic protocols, focusing on the BB84 quantum key distribution method and National Institute of Standards and Technology (NIST) approved quantum-resistant algorithms. By integrating AI-driven red teaming, automated penetration testing, and real-time anomaly detection, the research develops a framework for assessing and mitigating security risks in quantum networks. The findings demonstrate that AI can be effectively used to simulate adversarial attacks, probe weaknesses in cryptographic implementations, and refine security mechanisms through iterative feedback. The use of automated exploit simulations and protocol fuzzing provides a scalable means of identifying latent vulnerabilities, while adversarial machine learning techniques highlight novel attack surfaces within AI-enhanced cryptographic processes. This study offers a comprehensive methodology for strengthening quantum security and provides a foundation for integrating AI-driven cybersecurity practices into the evolving quantum landscape. 

**Abstract (ZH)**: 本研究提供了一种结构化的方法来评估量子加密协议中的漏洞，重点关注BB84量子密钥分发方法和美国国家标准与技术研究院（NIST）批准的量子抗攻击算法。通过整合基于AI的红队攻击、自动化渗透测试和实时异常检测，研究开发了一种评估和缓解量子网络中安全风险的框架。研究结果表明，AI可以有效用于模拟对手攻击、探测 cryptographic 实施中的弱点，并通过迭代反馈精炼安全机制。自动化漏洞利用模拟和协议 fuzzing 提供了一种可扩展的方法来识别潜在漏洞，而对抗性机器学习技术则突显了增强型 cryptographic 过程中的新型攻击面。本研究提供了一种全面的方法来加强量子安全，并为将基于AI的网络安全实践集成到不断发展的量子环境中奠定了基础。 

---
# Variance-Bounded Evaluation without Ground Truth: VB-Score 

**Title (ZH)**: 无 ground truth 条件下的方差有界评估：VB-Score 

**Authors**: Kaihua Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.22751)  

**Abstract**: Reliable evaluation is a central challenge in machine learning when tasks lack ground truth labels or involve ambiguity and noise. Conventional frameworks, rooted in the Cranfield paradigm and label-based metrics, fail in such cases because they cannot assess how robustly a system performs under uncertain interpretations. We introduce VB-Score, a variance-bounded evaluation framework that measures both effectiveness and robustness without requiring ground truth. Given a query or input, VB-Score enumerates plausible interpretations, assigns probabilities, and evaluates output by expected success penalized by variance, rewarding consistent performance across intents. We provide a formal analysis of VB-Score, establishing range, monotonicity, and stability properties, and relate it to risk-sensitive measures such as mean-variance utility. Experiments on ambiguous queries and entity-centric retrieval tasks show that VB-Score surfaces robustness differences hidden by conventional metrics. By enabling reproducible, label-free evaluation, VB-Score offers a principled foundation for benchmarking machine learning systems in ambiguous or label-scarce domains. 

**Abstract (ZH)**: 可靠的评估是机器学习中的一项中心挑战，特别是在任务缺乏 ground truth 标签或涉及模糊性和噪声的情况下。传统的框架根植于 Cranfield 帕累托思想和基于标签的度量标准，无法在这种情况下发挥作用，因为它们无法评估系统在不确定解释下的鲁棒性能。我们提出了 VB-Score，这是一种方差受限的评估框架，能够在无需 ground truth 的情况下衡量有效性和鲁棒性。给定查询或输入，VB-Score 列举可能的解释，分配概率，并通过预期成功惩罚方差进行评估，奖励意图一致的性能。我们对 VB-Score 进行了形式化分析，阐明了其范围、单调性和稳定性性质，并将其与均值方差效用等风险敏感度量进行了关联。实验表明，VB-Score 可以揭示传统度量所隐藏的鲁棒性差异。通过使评估可重复且无需标签，VB-Score 为在模糊或标签稀缺领域评估机器学习系统提供了有原则的基础。 

---
# MIRAGE: Multi-hop Reasoning with Ambiguity Evaluation for Illusory Questions 

**Title (ZH)**: MIRAGE: 多跳推理结合歧义评估用于虚假问题 

**Authors**: Jeonghyun Park, Ingeol Baek, Seunghyun Yoon, Haeun Jang, Aparna Garimella, Akriti Jain, Nedim Lipka, Hwanhee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.22750)  

**Abstract**: Real-world Multi-hop Question Answering (QA) often involves ambiguity that is inseparable from the reasoning process itself. This ambiguity creates a distinct challenge, where multiple reasoning paths emerge from a single question, each requiring independent resolution. Since each sub-question is ambiguous, the model must resolve ambiguity at every step. Thus, answering a single question requires handling multiple layers of ambiguity throughout the reasoning chain. We find that current Large Language Models (LLMs) struggle in this setting, typically exploring wrong reasoning paths and producing incomplete answers. To facilitate research on multi-hop ambiguity, we introduce MultI-hop Reasoning with AmbiGuity Evaluation for Illusory Questions (MIRAGE), a benchmark designed to analyze and evaluate this challenging intersection of ambiguity interpretation and multi-hop reasoning. MIRAGE contains 1,142 high-quality examples of ambiguous multi-hop questions, categorized under a taxonomy of syntactic, general, and semantic ambiguity, and curated through a rigorous multi-LLM verification pipeline. Our experiments reveal that even state-of-the-art models struggle on MIRAGE, confirming that resolving ambiguity combined with multi-step inference is a distinct and significant challenge. To establish a robust baseline, we propose CLarifying Ambiguity with a Reasoning and InstructiON (CLARION), a multi-agent framework that significantly outperforms existing approaches on MIRAGE, paving the way for more adaptive and robust reasoning systems. 

**Abstract (ZH)**: 真实世界多跳问答中存在的推理过程中不可避免的歧义性提出了独特的挑战：MIRAGE多歧义推理基准 

---
# Societal Capacity Assessment Framework: Measuring Resilience to Inform Advanced AI Risk Management 

**Title (ZH)**: 社会能力评估框架：衡量韧性以指导先进人工智能风险管理 

**Authors**: Milan Gandhi, Peter Cihon, Owen Larter, Rebecca Anselmetti  

**Link**: [PDF](https://arxiv.org/pdf/2509.22742)  

**Abstract**: Risk assessments for advanced AI systems require evaluating both the models themselves and their deployment contexts. We introduce the Societal Capacity Assessment Framework (SCAF), an indicators-based approach to measuring a society's vulnerability, coping capacity, and adaptive capacity in response to AI-related risks. SCAF adapts established resilience analysis methodologies to AI, enabling organisations to ground risk management in insights about country-level deployment conditions. It can also support stakeholders in identifying opportunities to strengthen societal preparedness for emerging AI capabilities. By bridging disparate literatures and the "context gap" in AI evaluation, SCAF promotes more holistic risk assessment and governance as advanced AI systems proliferate globally. 

**Abstract (ZH)**: 高级AI系统的风险评估需要评估模型本身及其部署环境。我们介绍了社会能力评估框架（SCAF），这是一种基于指标的方法，用于衡量社会在应对AI相关风险时的脆弱性、应对能力和适应能力。SCAF将现有的韧性分析方法应用于AI，使组织能够基于国家层面部署条件的风险管理洞察。它还可以帮助利益相关者识别加强社会对新兴AI能力准备的机会。通过弥合不同文献之间的鸿沟以及AI评估中的“环境差距”，SCAF促进了更全面的风险评估和治理，随着高级AI系统的全球普及。 

---
# Consistency Models as Plug-and-Play Priors for Inverse Problems 

**Title (ZH)**: 一致性模型作为即插即用先验用于逆问题 

**Authors**: Merve Gülle, Junno Yun, Yaşar Utku Alçalar, Mehmet Akçakaya  

**Link**: [PDF](https://arxiv.org/pdf/2509.22736)  

**Abstract**: Diffusion models have found extensive use in solving numerous inverse problems. Such diffusion inverse problem solvers aim to sample from the posterior distribution of data given the measurements, using a combination of the unconditional score function and an approximation of the posterior related to the forward process. Recently, consistency models (CMs) have been proposed to directly predict the final output from any point on the diffusion ODE trajectory, enabling high-quality sampling in just a few NFEs. CMs have also been utilized for inverse problems, but existing CM-based solvers either require additional task-specific training or utilize data fidelity operations with slow convergence, not amenable to large-scale problems. In this work, we reinterpret CMs as proximal operators of a prior, enabling their integration into plug-and-play (PnP) frameworks. We propose a solver based on PnP-ADMM, which enables us to leverage the fast convergence of conjugate gradient method. We further accelerate this with noise injection and momentum, dubbed PnP-CM, and show it maintains the convergence properties of the baseline PnP-ADMM. We evaluate our approach on a variety of inverse problems, including inpainting, super-resolution, Gaussian deblurring, and magnetic resonance imaging (MRI) reconstruction. To the best of our knowledge, this is the first CM trained for MRI datasets. Our results show that PnP-CM achieves high-quality reconstructions in as few as 4 NFEs, and can produce meaningful results in 2 steps, highlighting its effectiveness in real-world inverse problems while outperforming comparable CM-based approaches. 

**Abstract (ZH)**: 扩散模型在解决众多逆问题中找到了广泛的应用。这样的扩散逆问题求解器旨在利用无条件得分函数与前向过程相关联的后验近似共同从给定测量的数据后验分布中采样。最近，一致性模型（CMs）已被提出，可以直接从扩散微分方程轨迹上的任何点预测最终输出，从而使高质量采样仅需少量NFEs。CMs也被用于逆问题，但现有的CM基求解器要么需要附加的任务特定训练，要么使用数据保真操作且收敛缓慢，不适用于大规模问题。在这项工作中，我们重新解释CMs作为先验的近邻算子，使其能够集成到即插即用（PnP）框架中。我们提出一种基于PnP-ADMM的方法，这使我们能够利用共轭梯度法的快速收敛特性。我们进一步通过噪声注入和动量加速这种方法，命名为PnP-CM，并证明其保持了基础PnP-ADMM的收敛特性。我们在图像修复、超分辨率、高斯去模糊和磁共振成像（MRI）重建等多种逆问题上进行了评估。据我们所知，这是首个用于MRI数据集的CM训练方法。我们的结果表明，PnP-CM可以在仅4个NFEs内实现高质量的重建，并且可以在两步内生成有意义的结果，突显了其在真实世界逆问题中的有效性，并优于相似的CM基方法。 

---
# Rebuild AC Power Flow Models with Graph Attention Networks 

**Title (ZH)**: 基于图注意力网络重构AC功率流模型 

**Authors**: Yuting Hu, Jinjun Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2509.22733)  

**Abstract**: A full power flow (PF) model is a complete representation of the physical power network. Traditional model-based methods rely on the full PF model to implement power flow analysis. In practice, however, some PF model parameters can be inaccurate or even unavailable due to the uncertainties or dynamics in the power systems. Moreover, because the power network keeps evolving with possibly changing topology, the generalizability of a PF model to different network sizes and typologies should be considered. In this paper, we propose a PF rebuild model based on graph attention networks (GAT) by constructing a new graph based on the real and imaginary parts of voltage at each bus. By comparing with two state-of-the-art PF rebuild models for different standard IEEE power system cases and their modified topology variants, we demonstrate the feasibility of our method. Experimental results show that our proposed model achieves better accuracy for a changing network and can generalize to different networks with less accuracy discount. 

**Abstract (ZH)**: 基于图注意力网络的全功率流重建模型 

---
# Prompt-aware classifier free guidance for diffusion models 

**Title (ZH)**: 基于提示感知的分类器 Free 指导的扩散模型 

**Authors**: Xuanhao Zhang, Chang Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.22728)  

**Abstract**: Diffusion models have achieved remarkable progress in image and audio generation, largely due to Classifier-Free Guidance. However, the choice of guidance scale remains underexplored: a fixed scale often fails to generalize across prompts of varying complexity, leading to oversaturation or weak alignment. We address this gap by introducing a prompt-aware framework that predicts scale-dependent quality and selects the optimal guidance at inference. Specifically, we construct a large synthetic dataset by generating samples under multiple scales and scoring them with reliable evaluation metrics. A lightweight predictor, conditioned on semantic embeddings and linguistic complexity, estimates multi-metric quality curves and determines the best scale via a utility function with regularization. Experiments on MSCOCO~2014 and AudioCaps show consistent improvements over vanilla CFG, enhancing fidelity, alignment, and perceptual preference. This work demonstrates that prompt-aware scale selection provides an effective, training-free enhancement for pretrained diffusion backbones. 

**Abstract (ZH)**: 差分模型在图像和音频生成中的进步 largely得益于 Classifier-Free Guidance。然而，指导尺度的选择仍然没有得到充分探索：固定尺度往往无法适应不同复杂度提示的泛化，导致过度饱和或对齐不足。我们通过引入一个提示感知框架来填补这一空白，该框架预测尺度依赖的质量并选择最佳指导规模。具体来说，我们通过在多个尺度下生成样本并使用可靠的评估指标对其进行评分，构造了一个大规模的合成数据集。一个轻量级的预测器，基于语义嵌入和语言复杂性进行条件化，估计多指标质量曲线，并通过带正则化的效用函数确定最佳尺度。在 MSCOCO 2014 和 AudioCaps 上的实验表明，与 vanilla CFG 相比，此工作一致地提高了保真度、对齐和感知偏好。这项工作证明了提示感知尺度选择为预训练的差分模型骨架提供了有效且无需训练的增强。 

---
# A Data-Driven Framework for Digital Transformation in Smart Cities: Integrating AI, Dashboards, and IoT Readiness 

**Title (ZH)**: 面向智能城市的基于数据的数字转型框架：集成AI、仪表盘和物联网 readiness 

**Authors**: Ángel Lloret, Jesús Peral, Antonio Ferrández, María Auladell, Rafael Muñoz  

**Link**: [PDF](https://arxiv.org/pdf/2509.22721)  

**Abstract**: Digital transformation (DT) has become a strategic priority for public administrations, particularly due to the need to deliver more efficient and citizen-centered services and respond to societal expectations, ESG (Environmental, Social, and Governance) criteria, and the United Nations Sustainable Development Goals (UN SDGs). In this context, the main objective of this study is to propose an innovative methodology to automatically evaluate the level of digital transformation (DT) in public sector organizations. The proposed approach combines traditional assessment methods with Artificial Intelligence (AI) techniques. The methodology follows a dual approach: on the one hand, surveys are conducted using specialized staff from various public entities; on the other, AI-based models (including neural networks and transformer architectures) are used to estimate the DT level of the organizations automatically. Our approach has been applied to a real-world case study involving local public administrations in the Valencian Community (Spain) and shown effective performance in assessing DT. While the proposed methodology has been validated in a specific local context, its modular structure and dual-source data foundation support its international scalability, acknowledging that administrative, regulatory, and DT maturity factors may condition its broader applicability. The experiments carried out in this work include (i) the creation of a domain-specific corpus derived from the surveys and websites of several organizations, used to train the proposed models; (ii) the use and comparison of diverse AI methods; and (iii) the validation of our approach using real data. The integration of technologies such as the IoT, sensor networks, and AI-based analytics can significantly support resilient, agile urban environments and the transition towards more effective and sustainable Smart City models. 

**Abstract (ZH)**: 数字转型（DT）已成为公共管理的战略优先事项，特别是在提供更高效和以公民为中心的服务以及回应社会期望、ESG（环境、社会和治理）标准和联合国可持续发展目标（UN SDGs）方面。在此背景下，本研究的主要目标是提出一种创新方法，自动评估公共部门组织的数字转型水平。所提出的方法将传统评估方法与人工智能（AI）技术相结合。该方法采用双轨策略：一方面采用来自各类公共机构的专业人员进行问卷调查；另一方面使用基于AI的模型（包括神经网络和变换器架构）自动估计组织的数字转型水平。该方法在西班牙瓦伦西亚自治区的地方公共管理机构的实际案例研究中得到应用，并展示了其评估数字转型的有效性。虽然所提出的方法在特定的地方背景下得到了验证，但其模块化结构和多数据源基础使其具有国际扩展性，认识到行政、监管和数字转型成熟度等因素可能对其更广泛的应用产生影响。本研究中的实验包括：（i）从多个组织的调查和网站中创建特定领域的语料库，用于训练所提出模型；（ii）使用和比较多种AI方法；以及（iii）使用实证数据验证该方法。集成诸如物联网、传感器网络和基于AI的分析技术等技术可以显著支持具有韧性和敏捷性的城市环境，并实现更有效和可持续的智能城市模型。 

---
# Localizing Adversarial Attacks To Produces More Imperceptible Noise 

**Title (ZH)**: 定位 adversarial 攻击以生成更具不可感知性的噪声 

**Authors**: Pavan Reddy, Aditya Sanjay Gujral  

**Link**: [PDF](https://arxiv.org/pdf/2509.22710)  

**Abstract**: Adversarial attacks in machine learning traditionally focus on global perturbations to input data, yet the potential of localized adversarial noise remains underexplored. This study systematically evaluates localized adversarial attacks across widely-used methods, including FGSM, PGD, and C&W, to quantify their effectiveness, imperceptibility, and computational efficiency. By introducing a binary mask to constrain noise to specific regions, localized attacks achieve significantly lower mean pixel perturbations, higher Peak Signal-to-Noise Ratios (PSNR), and improved Structural Similarity Index (SSIM) compared to global attacks. However, these benefits come at the cost of increased computational effort and a modest reduction in Attack Success Rate (ASR). Our results highlight that iterative methods, such as PGD and C&W, are more robust to localization constraints than single-step methods like FGSM, maintaining higher ASR and imperceptibility metrics. This work provides a comprehensive analysis of localized adversarial attacks, offering practical insights for advancing attack strategies and designing robust defensive systems. 

**Abstract (ZH)**: 局部对抗攻击在机器学习中的传统研究主要集中在输入数据的全局扰动，而局部对抗噪声的应用潜力尚未充分探索。本研究系统评估了广泛使用的FGSM、PGD和C&W等方法的局部对抗攻击，以量化其有效性、不可感知性和计算效率。通过引入二进制掩码限制噪声到特定区域，局部攻击实现了显著更低的平均像素扰动、更高的峰值信噪比（PSNR）和改进的结构相似性指数（SSIM），但这些优势伴随着计算努力增加以及轻微的攻击成功率（ASR）下降。我们的结果表明，迭代方法如PGD和C&W对局部化约束更为 robust，保持了较高的ASR和不可感知性。本研究为局部对抗攻击提供了全面分析，为攻击策略的改进和设计 robust 防御系统提供了实用见解。 

---
# Intelligent Load Balancing in Cloud Computer Systems 

**Title (ZH)**: 云计算机系统中的智能负载均衡 

**Authors**: Leszek Sliwko  

**Link**: [PDF](https://arxiv.org/pdf/2509.22704)  

**Abstract**: Cloud computing is an established technology allowing users to share resources on a large scale, never before seen in IT history. A cloud system connects multiple individual servers in order to process related tasks in several environments at the same time. Clouds are typically more cost-effective than single computers of comparable computing performance. The sheer physical size of the system itself means that thousands of machines may be involved. The focus of this research was to design a strategy to dynamically allocate tasks without overloading Cloud nodes which would result in system stability being maintained at minimum cost. This research has added the following new contributions to the state of knowledge: (i) a novel taxonomy and categorisation of three classes of schedulers, namely OS-level, Cluster and Big Data, which highlight their unique evolution and underline their different objectives; (ii) an abstract model of cloud resources utilisation is specified, including multiple types of resources and consideration of task migration costs; (iii) a virtual machine live migration was experimented with in order to create a formula which estimates the network traffic generated by this process; (iv) a high-fidelity Cloud workload simulator, based on a month-long workload traces from Google's computing cells, was created; (v) two possible approaches to resource management were proposed and examined in the practical part of the manuscript: the centralised metaheuristic load balancer and the decentralised agent-based system. The project involved extensive experiments run on the University of Westminster HPC cluster, and the promising results are presented together with detailed discussions and a conclusion. 

**Abstract (ZH)**: 云计算是一种已确立的技术，允许用户大规模共享资源，这在IT史上前所未见。云系统连接多个独立服务器，以便在同一时间处理多个环境中的相关任务。与性能相当的单台计算机相比，云系统通常更具成本效益。系统的物理规模巨大，意味着可能涉及成千上万台机器。本研究的重点是设计一种策略，以动态分配任务而不 overloaded 云节点，从而在最小成本下维持系统稳定性。本研究为现有知识增添了以下新贡献：（i）提出了一种新颖的调度器分类法，包括OS级、集群和大数据三类，突显了它们的独特演化历程并强调了它们的不同目标；（ii）规定了一个云资源利用的抽象模型，包括多种类型资源以及任务迁移成本的考虑；（iii）实验了虚拟机在线迁移，以创建估计此过程产生网络流量的公式；（iv）基于谷歌计算单元一个月的工作负载追踪，创建了一个高保真度的云工作负载模拟器；（v）在手稿的实践部分提出了两种资源管理方法：集中式的元启发式负载均衡器和去中心化的基于代理的系统。该项目在威斯敏斯特大学高性能计算集群上进行了大量的实验，展示了令人鼓舞的结果，并附有详细的讨论和结论。 

---
# Enhancing Cluster Scheduling in HPC: A Continuous Transfer Learning for Real-Time Optimization 

**Title (ZH)**: 增强高性能计算中的聚类调度：一种实时优化的连续迁移学习 

**Authors**: Leszek Sliwko, Jolanta Mizera-Pietraszko  

**Link**: [PDF](https://arxiv.org/pdf/2509.22701)  

**Abstract**: This study presents a machine learning-assisted approach to optimize task scheduling in cluster systems, focusing on node-affinity constraints. Traditional schedulers like Kubernetes struggle with real-time adaptability, whereas the proposed continuous transfer learning model evolves dynamically during operations, minimizing retraining needs. Evaluated on Google Cluster Data, the model achieves over 99% accuracy, reducing computational overhead and improving scheduling latency for constrained tasks. This scalable solution enables real-time optimization, advancing machine learning integration in cluster management and paving the way for future adaptive scheduling strategies. 

**Abstract (ZH)**: 基于机器学习辅助的方法在群集系统中优化任务调度，关注节点亲和性约束 

---
# Learning Hyperspectral Images with Curated Text Prompts for Efficient Multimodal Alignment 

**Title (ZH)**: 使用精选文本提示学习超光谱图像以实现高效的多模态对齐 

**Authors**: Abhiroop Chatterjee, Susmita Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2509.22697)  

**Abstract**: As data requirements continue to grow, efficient learning increasingly depends on the curation and distillation of high-value data rather than brute-force scaling of model sizes. In the case of a hyperspectral image (HSI), the challenge is amplified by the high-dimensional 3D voxel structure, where each spatial location is associated with hundreds of contiguous spectral channels. While vision and language models have been optimized effectively for natural image or text tasks, their cross-modal alignment in the hyperspectral domain remains an open and underexplored problem. In this article, we make an attempt to optimize a Vision-Language Model (VLM) for hyperspectral scene understanding by exploiting a CLIP-style contrastive training framework. Our framework maps voxel-level embeddings from a vision backbone onto the latent space of a frozen large embedding model (LEM), where a trainable probe aligns vision features with the model's textual token representations. The two modalities are aligned via a contrastive loss restricted to a curated set of hard (closest wrong classes) and semi-hard (random distractors) negatives, along with positive pairs. To further enhance alignment, descriptive prompts that encode class semantics are introduced and act as structured anchors for the HSI embeddings. It is seen that the proposed method updates only 0.07 percent of the total parameters, yet yields state-of-the-art performance. For example, on Indian Pines (IP) the model produces better results over unimodal and multimodal baselines by +0.92 Overall Accuracy (OA) and +1.60 Kappa ($\kappa$), while on Pavia University (PU) data it provides gains of +0.69 OA and +0.90 $\kappa$. Moreover, this is achieved with the set of parameters, nearly 50$\times$ smaller than DCTN and 90$\times$ smaller than SS-TMNet. 

**Abstract (ZH)**: 随着数据需求不断增长，高效的learn过程 increasingly依赖于高质量数据的策划和提炼，而不是简单地扩大模型规模。在高光谱图像（HSI）的情况下，由于其高维度的3D体素结构，每个空间位置关联着上百个连续的光谱通道，挑战进一步放大。虽然视觉和语言模型已在自然图像或文本任务中得到了有效优化，但在高光谱域中的跨模态对齐仍是一个开放且未充分探索的问题。本文尝试通过利用CLIP风格的对比训练框架优化一个Vision-Language模型（VLM）以进行高光谱场景理解。该框架将视觉主干的体素级嵌入映射到一个冻结的大嵌入模型（LEM）的潜在空间中，其中可训练的探针将视觉特征与模型的文本标记表示对齐。通过限制在策划的硬（最接近的错误类别）和半硬（随机分散者）负样本集内的对比损失，以及正样本对，两模态得以对齐。为了进一步增强对齐，引入了描述性提示以编码类别语义，作为HSI嵌入的结构锚点。结果显示，所提出的方法仅更新了总参数的0.07%，但能达到最先进的性能。例如，在Indian Pines（IP）数据集上，模型相对于单模态和多模态基线方法在总体精度（OA）上提升了0.92，在卡帕系数（$\kappa$）上提升了1.60；而在Pavia University（PU）数据集上，模型提供了0.69的OA和0.90的$\kappa$的提升。此外，这实现了参数量几乎是DCTN的50倍少，SS-TMNet的90倍少。 

---
# PISA: An AI Pipeline for Interpretable-by-design Survival Analysis Providing Multiple Complexity-Accuracy Trade-off Models 

**Title (ZH)**: PISA：一种用于可解释设计生存分析的人工智能管道，提供多种复杂性-准确性trade-off模型 

**Authors**: Thalea Schlender, Catharina J.A. Romme, Yvette M. van der Linden, Luc R.C.W. van Lonkhuijzen, Peter A.N. Bosman, Tanja Alderliesten  

**Link**: [PDF](https://arxiv.org/pdf/2509.22673)  

**Abstract**: Survival analysis is central to clinical research, informing patient prognoses, guiding treatment decisions, and optimising resource allocation. Accurate time-to-event predictions not only improve quality of life but also reveal risk factors that shape clinical practice. For these models to be relevant in healthcare, interpretability is critical: predictions must be traceable to patient-specific characteristics, and risk factors should be identifiable to generate actionable insights for both clinicians and researchers. Traditional survival models often fail to capture non-linear interactions, while modern deep learning approaches, though powerful, are limited by poor interpretability.
We propose a Pipeline for Interpretable Survival Analysis (PISA) - a pipeline that provides multiple survival analysis models that trade off complexity and performance. Using multiple-feature, multi-objective feature engineering, PISA transforms patient characteristics and time-to-event data into multiple survival analysis models, providing valuable insights into the survival prediction task. Crucially, every model is converted into simple patient stratification flowcharts supported by Kaplan-Meier curves, whilst not compromising on performance. While PISA is model-agnostic, we illustrate its flexibility through applications of Cox regression and shallow survival trees, the latter avoiding proportional hazards assumptions.
Applied to two clinical benchmark datasets, PISA produced interpretable survival models and intuitive stratification flowcharts whilst achieving state-of-the-art performances. Revisiting a prior departmental study further demonstrated its capacity to automate survival analysis workflows in real-world clinical research. 

**Abstract (ZH)**: 可解释生存分析管道（PISA）：一种权衡复杂性和性能的多模型管道 

---
# Next Point-of-interest (POI) Recommendation Model Based on Multi-modal Spatio-temporal Context Feature Embedding 

**Title (ZH)**: 基于多模态时空上下文特征嵌入的下一个点Interest推荐模型 

**Authors**: Lingyu Zhang, Guobin Wu, Yan Wang, Pengfei Xu, Jian Liang, Xuan Song, Yunhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22661)  

**Abstract**: The next Point-of-interest (POI) recommendation is mainly based on sequential traffic information to predict the user's next boarding point location. This is a highly regarded and widely applied research task in the field of intelligent transportation, and there have been many research results to date. Traditional POI prediction models primarily rely on short-term traffic sequence information, often neglecting both long-term and short-term preference data, as well as crucial spatiotemporal context features in user behavior. To address this issue, this paper introduces user long-term preference information and key spatiotemporal context information, and proposes a POI recommendation model based on multimodal spatiotemporal context feature embedding. The model extracts long-term preference features and key spatiotemporal context features from traffic data through modules such as spatiotemporal feature processing, multimodal embedding, and self-attention aggregation. It then uses a weighted fusion method to dynamically adjust the weights of long-term and short-term features based on users' historical behavior patterns and the current context. Finally, the fused features are matched using attention, and the probability of each location candidate becoming the next location is calculated. This paper conducts experimental verification on multiple transportation datasets, and the results show that the POI prediction model combining multiple types of features has higher prediction accuracy than existing SOTA models and methods. 

**Abstract (ZH)**: 基于多模态时空上下文特征嵌入的POI推荐方法 

---
# Fairness for niche users and providers: algorithmic choice and profile portability 

**Title (ZH)**: 为 niche 用户和供应商提供公平性：算法选择与资料档案移植 

**Authors**: Elizabeth McKinnie, Anas Buhayh, Clement Canel, Robin Burke  

**Link**: [PDF](https://arxiv.org/pdf/2509.22660)  

**Abstract**: Ensuring fair outcomes for multiple stakeholders in recommender systems has been studied mostly in terms of algorithmic interventions: building new models with better fairness properties, or using reranking to improve outcomes from an existing algorithm. What has rarely been studied is structural changes in the recommendation ecosystem itself. Our work explores the fairness impact of algorithmic pluralism, the idea that the recommendation algorithm is decoupled from the platform through which users access content, enabling user choice in algorithms. Prior work using a simulation approach has shown that niche consumers and (especially) niche providers benefit from algorithmic choice. In this paper, we use simulation to explore the question of profile portability, to understand how different policies regarding the handling of user profiles interact with fairness outcomes for consumers and providers. 

**Abstract (ZH)**: 确保推荐系统中多利益相关方的公平结果在很大程度上是从算法干预的角度进行研究的：通过构建具有良好公平属性的新模型，或通过重新排序来改进现有算法的结果来实现。很少研究的是推荐生态系统本身的结构变化。我们的工作探讨了算法多元主义的公平影响，即推荐算法通过用户访问内容的平台进行解耦，使用户能够在算法之间进行选择。先前的工作通过仿真方法表明，利基消费者和（尤其是）利基提供商从算法选择中受益。在本文中，我们使用仿真来探索资料档案可携性的问题，以了解不同的用户资料处理政策如何与消费者和提供商的公平结果相互作用。 

---
# GOAT: A Large Dataset of Paired Guitar Audio Recordings and Tablatures 

**Title (ZH)**: GOAT：配对吉他音频录制和谱表的大规模数据集 

**Authors**: Jackson Loth, Pedro Sarmento, Saurjya Sarkar, Zixun Guo, Mathieu Barthet, Mark Sandler  

**Link**: [PDF](https://arxiv.org/pdf/2509.22655)  

**Abstract**: In recent years, the guitar has received increased attention from the music information retrieval (MIR) community driven by the challenges posed by its diverse playing techniques and sonic characteristics. Mainly fueled by deep learning approaches, progress has been limited by the scarcity and limited annotations of datasets. To address this, we present the Guitar On Audio and Tablatures (GOAT) dataset, comprising 5.9 hours of unique high-quality direct input audio recordings of electric guitars from a variety of different guitars and players. We also present an effective data augmentation strategy using guitar amplifiers which delivers near-unlimited tonal variety, of which we provide a starting 29.5 hours of audio. Each recording is annotated using guitar tablatures, a guitar-specific symbolic format supporting string and fret numbers, as well as numerous playing techniques. For this we utilise both the Guitar Pro format, a software for tablature playback and editing, and a text-like token encoding. Furthermore, we present competitive results using GOAT for MIDI transcription and preliminary results for a novel approach to automatic guitar tablature transcription. We hope that GOAT opens up the possibilities to train novel models on a wide variety of guitar-related MIR tasks, from synthesis to transcription to playing technique detection. 

**Abstract (ZH)**: 近年来，吉他因其多样的演奏技巧和音色特性，受到音乐信息检索（MIR）社区的越来越多关注。主要借助深度学习方法，进展受限于数据集稀缺且标注不足。为解决这一问题，我们提出了吉他音频和谱表数据集（Guitar On Audio and Tablatures, GOAT），包含5.9小时多种吉他和演奏者独特高质量的直接输入音频 recordings。我们还提出了一种有效的数据增强策略，利用吉他放大器产生近乎无限的音色变化，提供了初始的29.5小时音频。每个录音使用吉他谱表进行了标注，这是支持琴弦和品按键号码的吉他专用符号格式，以及众多演奏技巧。我们利用吉普生软件（Guitar Pro）进行谱表播放和编辑，并采用类似文本的标记编码。此外，我们展示了在MIDI转录任务中使用GOAT的竞争性结果，并介绍了自动吉他谱表转录的新颖方法的初步结果。我们希望GOAT能够开启针对吉他相关MIR任务的新型模型训练的可能性，从合成到转录再到演奏技巧检测。 

---
# Sustainable LSTM-Based Precoding for RIS-Aided mmWave MIMO Systems with Implicit CSI 

**Title (ZH)**: 基于RIS辅助毫米波MIMO系统的可持续LSTM基预编码方法：隐式CSI情形 

**Authors**: Po-Heng Chou, Jiun-Jia Wu, Wan-Jen Huang, Ronald Y. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2509.12658)  

**Abstract**: In this paper, we propose a sustainable long short-term memory (LSTM)-based precoding framework for reconfigurable intelligent surface (RIS)-assisted millimeter-wave (mmWave) MIMO systems. Instead of explicit channel state information (CSI) estimation, the framework exploits uplink pilot sequences to implicitly learn channel characteristics, reducing both pilot overhead and inference complexity. Practical hardware constraints are addressed by incorporating the phase-dependent amplitude model of RIS elements, while a multi-label training strategy improves robustness when multiple near-optimal codewords yield comparable performance. Simulations show that the proposed design achieves over 90% of the spectral efficiency of exhaustive search (ES) with only 2.2% of its computation time, cutting energy consumption by nearly two orders of magnitude. The method also demonstrates resilience under distribution mismatch and scalability to larger RIS arrays, making it a practical and energy-efficient solution for sustainable 6G wireless networks. 

**Abstract (ZH)**: 基于可重构智能表面辅助毫米波MIMO系统的可持续长短期记忆（LSTM）预编码框架 

---
# How are Scientific Concepts Birthed? Typing Rules of Concept Formation in Theoretical Physics Reasoning 

**Title (ZH)**: 科学概念是如何诞生的？理论物理学推理中概念形成的基本规则 

**Authors**: Omar Aguilar, Anthony Aguirre  

**Link**: [PDF](https://arxiv.org/pdf/2509.10740)  

**Abstract**: This work aims to formalize some of the ways scientific concepts are formed in the process of theoretical physics discovery. Since this may at first seem like a task beyond the scope of the exact sciences (natural and formal sciences), we begin by presenting arguments for why scientific concept formation can be formalized. Then, we introduce type theory as a natural and well-suited framework for this formalization. We formalize what we call "ways of discovering new concepts" including concept distinction, property preservation, and concept change, as cognitive typing rules. Next, we apply these cognitive typing rules to two case studies of conceptual discovery in the history of physics: Einstein's reasoning leading to the impossibility of frozen waves, and his conceptual path to the relativity of time. In these historical episodes, we recast what a physicist might informally call "ways of discovering new scientific concepts" as compositional typing rules built from cognitive typing rules - thus formalizing them as scientific discovery mechanisms. Lastly, we computationally model the type-theoretic reconstruction of Einstein's conceptual path to the relativity of time as a program synthesis task. 

**Abstract (ZH)**: 本研究旨在形式化理论物理发现过程中形成科学概念的一些方式。虽然这可能最初看起来超出了精确科学（自然科学和形式科学）的范畴，我们首先通过论述科学概念形成可以形式化的理由来开始。然后，我们引入类型理论作为自然且合适的框架来进行这种形式化。我们将“发现新概念的方式”形式化，包括概念区分、属性保存和概念变化，作为认知类型规则。接下来，我们应用这些认知类型规则对物理学史上两个概念发现案例进行研究：爱因斯坦导致无法存在冻结波的推理过程，以及他对时间相对性的概念路径。在这些历史事件中，我们将物理学家可能非正式称之为“发现新科学概念的方式”重新表述为由认知类型规则构建的组合类型规则，从而将它们形式化为科学发现机制。最后，我们通过程序合成任务来计算建模爱因斯坦从概念路径到时间相对性的类型论重构。 

---
# Green Learning for STAR-RIS mmWave Systems with Implicit CSI 

**Title (ZH)**: 绿联学习在IMCSI的STAR-RIS毫米波系统中 

**Authors**: Yu-Hsiang Huang, Po-Heng Chou, Wan-Jen Huang, Walid Saad, C.-C. Jay Kuo  

**Link**: [PDF](https://arxiv.org/pdf/2509.06820)  

**Abstract**: In this paper, a green learning (GL)-based precoding framework is proposed for simultaneously transmitting and reflecting reconfigurable intelligent surface (STAR-RIS)-aided millimeter-wave (mmWave) MIMO broadcasting systems. Motivated by the growing emphasis on environmental sustainability in future 6G networks, this work adopts a broadcasting transmission architecture for scenarios where multiple users share identical information, improving spectral efficiency and reducing redundant transmissions and power consumption. Different from conventional optimization methods, such as block coordinate descent (BCD) that require perfect channel state information (CSI) and iterative computation, the proposed GL framework operates directly on received uplink pilot signals without explicit CSI estimation. Unlike deep learning (DL) approaches that require CSI-based labels for training, the proposed GL approach also avoids deep neural networks and backpropagation, leading to a more lightweight design. Although the proposed GL framework is trained with supervision generated by BCD under full CSI, inference is performed in a fully CSI-free manner. The proposed GL integrates subspace approximation with adjusted bias (Saab), relevant feature test (RFT)-based supervised feature selection, and eXtreme gradient boosting (XGBoost)-based decision learning to jointly predict the STAR-RIS coefficients and transmit precoder. Simulation results show that the proposed GL approach achieves competitive spectral efficiency compared to BCD and DL-based models, while reducing floating-point operations (FLOPs) by over four orders of magnitude. These advantages make the proposed GL approach highly suitable for real-time deployment in energy- and hardware-constrained broadcasting scenarios. 

**Abstract (ZH)**: 基于绿色学习的STAR-RIS辅助毫米波MIMO广播系统同时传输与反射框架 

---
# BenLOC: A Benchmark for Learning to Configure MIP Optimizers 

**Title (ZH)**: BenLOC: 一个学习配置MIP优化器的标准数据集 

**Authors**: Hongpei Li, Ziyan He, Yufei Wang, Wenting Tu, Shanwen Pu, Qi Deng, Dongdong Ge  

**Link**: [PDF](https://arxiv.org/pdf/2506.02752)  

**Abstract**: The automatic configuration of Mixed-Integer Programming (MIP) optimizers has become increasingly critical as the large number of configurations can significantly affect solver performance. Yet the lack of standardized evaluation frameworks has led to data leakage and over-optimistic claims, as prior studies often rely on homogeneous datasets and inconsistent experimental setups. To promote a fair evaluation process, we present BenLOC, a comprehensive benchmark and open-source toolkit, which not only offers an end-to-end pipeline for learning instance-wise MIP optimizer configurations, but also standardizes dataset selection, train-test splits, feature engineering and baseline choice for unbiased and comprehensive evaluations. Leveraging this framework, we conduct an empirical analysis on five well-established MIP datasets and compare classical machine learning models with handcrafted features against state-of-the-art deep-learning techniques. The results demonstrate the importance of datasets, features and baseline criteria proposed by BenLOC and the effectiveness of BenLOC in providing unbiased and comprehensive evaluations. 

**Abstract (ZH)**: Mixed-Integer Programming (MIP) 优化器的自动配置已成为越来越关键的问题，因为大量配置会显著影响求解器性能。然而，缺乏标准化评估框架导致了数据泄漏和过于乐观的声明，此前的研究经常依赖同质数据集和不一致的实验设置。为了促进公平的评估过程，我们提出 BenLOC，一个全面的基准和开源工具包，不仅提供了一站式的实例级 MIP 优化器配置学习管道，还对数据集选择、训练-测试分割、特征工程和基准选择进行了标准化，以实现无偏且全面的评估。利用这一框架，我们在五个广泛认可的 MIP 数据集上进行了实证分析，并将经典的机器学习模型与手工设计的特征与最先进的深度学习技术进行了对比。结果表明，BenLOC 提出的数据集、特征和基准标准的重要性，以及 BenLOC 在提供无偏且全面评估方面的有效性。 

---
# Prosody-Adaptable Audio Codecs for Zero-Shot Voice Conversion via In-Context Learning 

**Title (ZH)**: 适用于零样本语音转换的基于上下文学习的语调自适应音频编解码器 

**Authors**: Junchuan Zhao, Xintong Wang, Ye Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15402)  

**Abstract**: Recent advances in discrete audio codecs have significantly improved speech representation modeling, while codec language models have enabled in-context learning for zero-shot speech synthesis. Inspired by this, we propose a voice conversion (VC) model within the VALLE-X framework, leveraging its strong in-context learning capabilities for speaker adaptation. To enhance prosody control, we introduce a prosody-aware audio codec encoder (PACE) module, which isolates and refines prosody from other sources, improving expressiveness and control. By integrating PACE into our VC model, we achieve greater flexibility in prosody manipulation while preserving speaker timbre. Experimental evaluation results demonstrate that our approach outperforms baseline VC systems in prosody preservation, timbre consistency, and overall naturalness, surpassing baseline VC systems. 

**Abstract (ZH)**: 最近在离散音频编解码器方面的进展显著改善了语音表示建模，而编解码器语言模型使零-shot语音合成具备了上下文学习能力。受此启发，我们提出了一个结合在VALLE-X框架内的语音转换（VC）模型，利用其强大的上下文学习能力进行说话人适应。为了增强语调控制，我们引入了一种感知语调的音频编解码器编码器（PACE）模块，该模块能够孤立并精炼语调以改善表达性和控制性。通过将PACE集成到我们的VC模型中，我们在保持说话人音色的同时实现了更灵活的语调操控。实验评估结果表明，我们的方法在语调保持、音色一致性及总体自然度方面均优于基线VC系统，超越了基线VC系统。 

---
