# DualMap: Online Open-Vocabulary Semantic Mapping for Natural Language Navigation in Dynamic Changing Scenes 

**Title (ZH)**: DualMap: 在动态变化场景中进行自然语言导航的在线开放词汇语义映射 

**Authors**: Jiajun Jiang, Yiming Zhu, Zirui Wu, Jie Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.01950)  

**Abstract**: We introduce DualMap, an online open-vocabulary mapping system that enables robots to understand and navigate dynamically changing environments through natural language queries. Designed for efficient semantic mapping and adaptability to changing environments, DualMap meets the essential requirements for real-world robot navigation applications. Our proposed hybrid segmentation frontend and object-level status check eliminate the costly 3D object merging required by prior methods, enabling efficient online scene mapping. The dual-map representation combines a global abstract map for high-level candidate selection with a local concrete map for precise goal-reaching, effectively managing and updating dynamic changes in the environment. Through extensive experiments in both simulation and real-world scenarios, we demonstrate state-of-the-art performance in 3D open-vocabulary segmentation, efficient scene mapping, and online language-guided navigation. 

**Abstract (ZH)**: 我们介绍了DualMap，一个在线开放词汇映射系统，通过自然语言查询使机器人能够理解和导航动态变化的环境。设计用于高效语义映射和适应变化的环境，DualMap 满足了现实世界机器人导航应用的基本要求。我们提出的混合分割前端和对象级状态检查消除了先前方法所需的昂贵的3D物体合并，从而实现高效的在线场景映射。双地图表示结合全局抽象地图用于高层候选选择以及局部具体地图用于精确目标到达，有效地管理和更新环境中的动态变化。通过在仿真和实际场景中的广泛实验，我们展示了在3D开放式词汇分割、高效场景映射和在线语言引导导航方面的前沿性能。 

---
# A Hierarchical Bin Packing Framework with Dual Manipulators via Heuristic Search and Deep Reinforcement Learning 

**Title (ZH)**: 基于启发式搜索和深度强化学习的双 manipulator 分级集装箱打包框架 

**Authors**: Beomjoon Lee, Changjoo Nam  

**Link**: [PDF](https://arxiv.org/pdf/2506.01628)  

**Abstract**: We address the bin packing problem (BPP), which aims to maximize bin utilization when packing a variety of items. The offline problem, where the complete information about the item set and their sizes is known in advance, is proven to be NP-hard. The semi-online and online variants are even more challenging, as full information about incoming items is unavailable. While existing methods have tackled both 2D and 3D BPPs, the 2D BPP remains underexplored in terms of fully maximizing utilization. We propose a hierarchical approach for solving the 2D online and semi-online BPP by combining deep reinforcement learning (RL) with heuristic search. The heuristic search selects which item to pack or unpack, determines the packing order, and chooses the orientation of each item, while the RL agent decides the precise position within the bin. Our method is capable of handling diverse scenarios, including repacking, varying levels of item information, differing numbers of accessible items, and coordination of dual manipulators. Experimental results demonstrate that our approach achieves near-optimal utilization across various practical scenarios, largely due to its repacking capability. In addition, the algorithm is evaluated in a physics-based simulation environment, where execution time is measured to assess its real-world performance. 

**Abstract (ZH)**: 我们提出了一种层次化方法，通过结合深度强化学习（RL）和启发式搜索来解决2D在线和半在线填箱问题（BPP），以最大化箱利用率。 

---
# Max Entropy Moment Kalman Filter for Polynomial Systems with Arbitrary Noise 

**Title (ZH)**: 多项式系统中任意噪声下的最大熵矩卡尔曼滤波器 

**Authors**: Sangli Teng, Harry Zhang, David Jin, Ashkan Jasour, Ram Vasudevan, Maani Ghaffari, Luca Carlone  

**Link**: [PDF](https://arxiv.org/pdf/2506.00838)  

**Abstract**: Designing optimal Bayes filters for nonlinear non-Gaussian systems is a challenging task. The main difficulties are: 1) representing complex beliefs, 2) handling non-Gaussian noise, and 3) marginalizing past states. To address these challenges, we focus on polynomial systems and propose the Max Entropy Moment Kalman Filter (MEM-KF). To address 1), we represent arbitrary beliefs by a Moment-Constrained Max-Entropy Distribution (MED). The MED can asymptotically approximate almost any distribution given an increasing number of moment constraints. To address 2), we model the noise in the process and observation model as MED. To address 3), we propagate the moments through the process model and recover the distribution as MED, thus avoiding symbolic integration, which is generally intractable. All the steps in MEM-KF, including the extraction of a point estimate, can be solved via convex optimization. We showcase the MEM-KF in challenging robotics tasks, such as localization with unknown data association. 

**Abstract (ZH)**: 设计非线性非高斯系统的最优贝叶斯滤波器是一项具有挑战性的任务。主要困难包括：1）表示复杂的信念，2）处理非高斯噪声，3）消除过去状态的影响。为了应对这些挑战，我们专注于多项式系统，并提出了最大熵矩卡尔曼滤波器（MEM-KF）。为了应对1），我们使用矩约束最大熵分布（MED）来表示任意的信念。MED在给定越来越多的矩约束时可以渐近地逼近几乎所有分布。为了应对2），我们将过程和观测模型中的噪声建模为MED。为了应对3），我们通过过程模型传播矩并在必要时恢复为MED，从而避免了通常难以处理的符号积分。MEM-KF的所有步骤，包括提取点估计，都可以通过凸优化来解决。我们在挑战性的机器人任务中展示了MEM-KF，例如具有未知数据关联的定位任务。 

---
# Lazy Heuristic Search for Solving POMDPs with Expensive-to-Compute Belief Transitions 

**Title (ZH)**: 昂贵信念转移计算的POMDPs懒惰启发式搜索 

**Authors**: Muhammad Suhail Saleem, Rishi Veerapaneni, Maxim Likhachev  

**Link**: [PDF](https://arxiv.org/pdf/2506.00285)  

**Abstract**: Heuristic search solvers like RTDP-Bel and LAO* have proven effective for computing optimal and bounded sub-optimal solutions for Partially Observable Markov Decision Processes (POMDPs), which are typically formulated as belief MDPs. A belief represents a probability distribution over possible system states. Given a parent belief and an action, computing belief state transitions involves Bayesian updates that combine the transition and observation models of the POMDP to determine successor beliefs and their transition probabilities. However, there is a class of problems, specifically in robotics, where computing these transitions can be prohibitively expensive due to costly physics simulations, raycasting, or expensive collision checks required by the underlying transition and observation models, leading to long planning times. To address this challenge, we propose Lazy RTDP-Bel and Lazy LAO*, which defer computing expensive belief state transitions by leveraging Q-value estimation, significantly reducing planning time. We demonstrate the superior performance of the proposed lazy planners in domains such as contact-rich manipulation for pose estimation, outdoor navigation in rough terrain, and indoor navigation with a 1-D LiDAR sensor. Additionally, we discuss practical Q-value estimation techniques for commonly encountered problem classes that our lazy planners can leverage. Our results show that lazy heuristic search methods dramatically improve planning speed by postponing expensive belief transition evaluations while maintaining solution quality. 

**Abstract (ZH)**: Heuristic搜索求解器如RTDP-Bel和LAO*在计算部分可观测马尔可夫决策过程（POMDPs）的最优和次优解方面 proven有效，这些过程通常被公式化为信念MDP。一个信念表示可能系统状态的概率分布。给定一个父信念和一个动作，计算信念状态转移涉及贝叶斯更新，结合POMDP的转移模型和观测模型来确定后续信念及其转移概率。然而，在机器人领域存在一类问题，由于涉及昂贵的物理模拟、射线投射或基础转移和观测模型所需的昂贵碰撞检测，计算这些转移可能会导致规划时间过长。为了解决这一挑战，我们提出了Lazy RTDP-Bel和Lazy LAO*，通过利用Q值估算推迟计算昂贵的信念状态转移，显著减少了规划时间。我们在接触丰富的操作姿态估计、崎岖地形户外导航和配备一维LiDAR的室内导航等领域展示了所提懒惰规划器的优越性能。此外，我们还讨论了懒惰规划器可以利用的常见问题类别中的实用Q值估算技术。我们的结果表明，懒惰启发式搜索方法通过推迟昂贵的信念转移评估显著提高了规划速度，同时保持了解的质量。 

---
# AniTrack: A Power-Efficient, Time-Slotted and Robust UWB Localization System for Animal Tracking in a Controlled Setting 

**Title (ZH)**: AniTrack：一种适用于受控环境中的动物跟踪的高效、时-slot化和稳健的UWB定位系统 

**Authors**: Victor Luder, Lukas Schulthess, Silvano Cortesi, Leyla Rivero Davis, Michele Magno  

**Link**: [PDF](https://arxiv.org/pdf/2506.00216)  

**Abstract**: Accurate localization is essential for a wide range of applications, including asset tracking, smart agriculture, and an- imal monitoring. While traditional localization methods, such as Global Navigation Satellite System (GNSS), Wi-Fi, and Bluetooth Low Energy (BLE), offer varying levels of accuracy and coverage, they have drawbacks regarding power consumption, infrastruc- ture requirements, and deployment flexibility. Ultra-Wideband (UWB) is emerging as an alternative, offering centimeter-level accuracy and energy efficiency, especially suitable for medium to large field monitoring with capabilities to work indoors and outdoors. However, existing UWB localization systems require infrastructure with mains power to supply the anchors, which impedes their scalability and ease of deployment. This under- scores the need for a fully battery-powered and energy-efficient localization system. This paper presents an energy-optimized, battery-operated UWB localization system that leverages Long Range Wide Area Network (LoRaWAN) for data transmission to a server backend. By employing single-sided two-way ranging (SS-TWR) in a time- slotted localization approach, the power consumption both on the anchor and the tag is reduced, while maintaining high accuracy. With a low average power consumption of 20.44 mW per anchor and 7.19 mW per tag, the system allows fully battery- powered operation for up to 25 days, achieving average accuracy of 13.96 cm with self-localizing anchors on a 600 m2 testing ground. To validate its effectiveness and ease of installation in a challenging application scenario, ten anchors and two tags were successfully deployed in a tropical zoological biome where they could be used to track Aldabra Giant Tortoises (Aldabrachelys gigantea). 

**Abstract (ZH)**: 一种基于LoRaWAN的数据传输的低功耗UWB定位系统及其在热带生物圈的应用研究 

---
# Online Competitive Information Gathering for Partially Observable Trajectory Games 

**Title (ZH)**: 部分可观测轨迹博弈中的在线竞速信息收集 

**Authors**: Mel Krusniak, Hang Xu, Parker Palermo, Forrest Laine  

**Link**: [PDF](https://arxiv.org/pdf/2506.01927)  

**Abstract**: Game-theoretic agents must make plans that optimally gather information about their opponents. These problems are modeled by partially observable stochastic games (POSGs), but planning in fully continuous POSGs is intractable without heavy offline computation or assumptions on the order of belief maintained by each player. We formulate a finite history/horizon refinement of POSGs which admits competitive information gathering behavior in trajectory space, and through a series of approximations, we present an online method for computing rational trajectory plans in these games which leverages particle-based estimations of the joint state space and performs stochastic gradient play. We also provide the necessary adjustments required to deploy this method on individual agents. The method is tested in continuous pursuit-evasion and warehouse-pickup scenarios (alongside extensions to $N > 2$ players and to more complex environments with visual and physical obstacles), demonstrating evidence of active information gathering and outperforming passive competitors. 

**Abstract (ZH)**: 基于博弈的代理必须制定最优地收集关于对手信息的计划。这些问题通过部分可观测随机博弈（POSGs）建模，但在连续的POSGs中进行完全在线规划是不可行的，除非对每个玩家保持的信念顺序做出假设。我们提出了POSGs的一个有限历史/时限细化，它在轨迹空间中允许竞争性的信息收集行为，并通过一系列近似，我们给出了一个利用基于粒子的状态空间联合估计并在随机梯度播放中计算这些博弈中的理性轨迹计划的方法。我们也提供了在单个代理上部署此方法所需的必要调整。该方法在连续的追逐-逃避和仓库取件场景（以及扩展到超过两个玩家和更复杂环境中）中进行了测试，展示了积极的信息收集行为，并且优于被动的竞争对手。 

---
# Captivity-Escape Games as a Means for Safety in Online Motion Generation 

**Title (ZH)**: 基于捕获-逃脱游戏的在线运动生成安全方法 

**Authors**: Christopher Bohn, Manuel Hess, Sören Hohmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.01399)  

**Abstract**: This paper presents a method that addresses the conservatism, computational effort, and limited numerical accuracy of existing frameworks and methods that ensure safety in online model-based motion generation, commonly referred to as fast and safe tracking. Computational limitations restrict online motion planning to low-fidelity models. However, planning with low-fidelity models compromises safety, as the dynamic feasibility of resulting reference trajectories is not ensured. This potentially leads to unavoidable tracking errors that may cause safety-critical constraint violations. Existing frameworks mitigate this safety risk by augmenting safety-critical constraints in motion planning by a safety margin that prevents constraint violations under worst-case tracking errors. However, the methods employed in these frameworks determine the safety margin based on a heuristically selected performance of the planning model, which likely results in overly conservative reference trajectories. Furthermore, these methods are computationally intensive, and the state-of-the-art method is limited in numerical accuracy. We adopt a different perspective and address these limitations with a method that mitigates conservatism in existing frameworks by adapting the planning model performance to a given safety margin. Our method achieves numerical accuracy and requires significantly less computation time than existing methods by leveraging a captivity-escape game, which is a specific zero-sum differential game formulated in this paper. We demonstrate our method using a numerical example and compare it to the state of the art. 

**Abstract (ZH)**: 本文提出了一种方法，用于解决现有确保在线模型导向运动生成安全性的框架和方法中存在的保守性、计算效率低以及数值精度有限的问题，这些框架和方法通常被称为快速安全跟踪。计算限制使得在线运动规划局限于低保真模型。然而，使用低保真模型进行规划会牺牲安全性，因为生成的参考轨迹的动态可行性无法得到保证。这可能导致不可避免的跟踪误差，进而引发安全关键约束的违犯。现有的框架通过在运动规划中增加安全裕度来缓解这种安全性风险，以防止在最坏情况下的跟踪误差导致约束违犯。然而，这些框架中使用的方法是基于规划模型的启发式性能来确定安全裕度的，这可能导致过于保守的参考轨迹。此外，这些方法计算量大，最先进的方法在数值精度上也有局限。本文从不同角度出发，并采用了一种方法来解决这些问题，通过将规划模型的性能调整到给定的安全裕度，缓解现有框架的保守性。本文提出的方法通过利用 captivity-escape 游戏（一种在本文中具体定义的零和微分博弈）实现了数值精度，并比现有方法所需的计算时间显著减少。我们使用数值示例展示了该方法，并将其与最先进的方法进行了比较。 

---
# Two-Stage Learning of Stabilizing Neural Controllers via Zubov Sampling and Iterative Domain Expansion 

**Title (ZH)**: 基于Zubov采样和迭代领域扩展的两阶段稳定神经控制器学习 

**Authors**: Haoyu Li, Xiangru Zhong, Bin Hu, Huan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01356)  

**Abstract**: Learning-based neural network (NN) control policies have shown impressive empirical performance. However, obtaining stability guarantees and estimations of the region of attraction of these learned neural controllers is challenging due to the lack of stable and scalable training and verification algorithms. Although previous works in this area have achieved great success, much conservatism remains in their framework. In this work, we propose a novel two-stage training framework to jointly synthesize the controller and Lyapunov function for continuous-time systems. By leveraging a Zubov-inspired region of attraction characterization to directly estimate stability boundaries, we propose a novel training data sampling strategy and a domain updating mechanism that significantly reduces the conservatism in training. Moreover, unlike existing works on continuous-time systems that rely on an SMT solver to formally verify the Lyapunov condition, we extend state-of-the-art neural network verifier $\alpha,\!\beta$-CROWN with the capability of performing automatic bound propagation through the Jacobian of dynamical systems and a novel verification scheme that avoids expensive bisection. To demonstrate the effectiveness of our approach, we conduct numerical experiments by synthesizing and verifying controllers on several challenging nonlinear systems across multiple dimensions. We show that our training can yield region of attractions with volume $5 - 1.5\cdot 10^{5}$ times larger compared to the baselines, and our verification on continuous systems can be up to $40-10000$ times faster compared to the traditional SMT solver dReal. Our code is available at this https URL. 

**Abstract (ZH)**: 基于学习的神经网络控制策略在实验性能上表现出色。然而，由于缺乏稳定的和可扩展的训练与验证算法，获得这些学习到的神经网络控制器的稳定性保证以及其吸引域估计仍具有挑战性。尽管该领域已有许多成功的工作，但在其框架中仍存在大量的保守性。本文提出了一种新颖的两阶段训练框架，用于联合合成连续时间系统的控制器和李亚普诺夫函数。通过利用Zubov启发式吸引域表征直接估计稳定性边界，我们提出了一种新的训练数据采样策略和领域更新机制，显著减少了训练的保守性。此外，与现有依赖SMT求解器形式验证李亚普诺夫条件的连续时间系统方法不同，我们扩展了最先进的神经网络验证器α,β-CROWN，使其能够通过动力学系统的雅可比进行自动边界传播，并采用了一种新的验证方案，避免了昂贵的二分法。为了展示我们方法的有效性，我们在多个维度上的多个具有挑战性的非线性系统上合成了并验证了控制器。实验结果表明，我们的训练可以使得吸引域的体积比 baselines 大5到$1.5 \times 10^5$倍，而对连续系统的验证比传统SMT求解器dReal 快40到10000倍。我们的代码可在此网址获得。 

---
# Variational Adaptive Noise and Dropout towards Stable Recurrent Neural Networks 

**Title (ZH)**: 变分自适应噪声和dropout以实现稳定的递归神经网络 

**Authors**: Taisuke Kobayashi, Shingo Murata  

**Link**: [PDF](https://arxiv.org/pdf/2506.01350)  

**Abstract**: This paper proposes a novel stable learning theory for recurrent neural networks (RNNs), so-called variational adaptive noise and dropout (VAND). As stabilizing factors for RNNs, noise and dropout on the internal state of RNNs have been separately confirmed in previous studies. We reinterpret the optimization problem of RNNs as variational inference, showing that noise and dropout can be derived simultaneously by transforming the explicit regularization term arising in the optimization problem into implicit regularization. Their scale and ratio can also be adjusted appropriately to optimize the main objective of RNNs, respectively. In an imitation learning scenario with a mobile manipulator, only VAND is able to imitate sequential and periodic behaviors as instructed. this https URL 

**Abstract (ZH)**: 这种论文提出了一种新的循环神经网络（RNN）稳定学习理论，称为变分自适应噪声和 dropout（VAND）。作为 RNN 的稳定因素，先前研究已分别确认噪声和 dropout 对 RNN 内部状态的稳定性有影响。我们将 RNN 的优化问题重新解读为变分推断，展示了可以通过将优化问题中出现的显式正则化项转换为隐式正则化来同时推导噪声和 dropout。它们的规模和比例也可以适当地调整以优化 RNN 的主要目标。在带有移动 manipulator 的模仿学习场景中，只有 VAND 能够模仿所指示的序列和周期行为。 this https://doi.org/10.1109/IEEECONF.2023.XXXXXX 

---
# Test Automation for Interactive Scenarios via Promptable Traffic Simulation 

**Title (ZH)**: 基于可提示流量模拟的交互场景自动化测试 

**Authors**: Augusto Mondelli, Yueshan Li, Alessandro Zanardi, Emilio Frazzoli  

**Link**: [PDF](https://arxiv.org/pdf/2506.01199)  

**Abstract**: Autonomous vehicle (AV) planners must undergo rigorous evaluation before widespread deployment on public roads, particularly to assess their robustness against the uncertainty of human behaviors. While recent advancements in data-driven scenario generation enable the simulation of realistic human behaviors in interactive settings, leveraging these models to construct comprehensive tests for AV planners remains an open challenge. In this work, we introduce an automated method to efficiently generate realistic and safety-critical human behaviors for AV planner evaluation in interactive scenarios. We parameterize complex human behaviors using low-dimensional goal positions, which are then fed into a promptable traffic simulator, ProSim, to guide the behaviors of simulated agents. To automate test generation, we introduce a prompt generation module that explores the goal domain and efficiently identifies safety-critical behaviors using Bayesian optimization. We apply our method to the evaluation of an optimization-based planner and demonstrate its effectiveness and efficiency in automatically generating diverse and realistic driving behaviors across scenarios with varying initial conditions. 

**Abstract (ZH)**: 自主车辆（AV）规划器在广泛部署于公共道路之前必须经过严格的评估，特别是要评估其在面对人类行为不确定性时的鲁棒性。虽然近期基于数据的场景生成技术能够模拟交互环境中的真实人类行为，但利用这些模型为AV规划器构建全面的测试仍然是一项开放的挑战。本文介绍了一种自动化方法，用于高效生成用于评估交互场景中AV规划器的现实且安全关键的人类行为。我们使用低维度的目标位置参数化复杂的_human行为，并将这些参数输入可提示的交通模拟器ProSim以引导模拟代理的行为。为实现测试生成的自动化，我们引入了一个提示生成模块，通过贝叶斯优化高效地探索目标领域并识别安全关键行为。我们将该方法应用于基于优化的规划器的评估，并展示了其在自动生成跨不同初始条件场景下的多样且真实驾驶行为方面的有效性与效率。 

---
# Accelerated Learning with Linear Temporal Logic using Differentiable Simulation 

**Title (ZH)**: 使用可微模拟的线性时序逻辑加速学习 

**Authors**: Alper Kamil Bozkurt, Calin Belta, Ming C. Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.01167)  

**Abstract**: To ensure learned controllers comply with safety and reliability requirements for reinforcement learning in real-world settings remains challenging. Traditional safety assurance approaches, such as state avoidance and constrained Markov decision processes, often inadequately capture trajectory requirements or may result in overly conservative behaviors. To address these limitations, recent studies advocate the use of formal specification languages such as linear temporal logic (LTL), enabling the derivation of correct-by-construction learning objectives from the specified requirements. However, the sparse rewards associated with LTL specifications make learning extremely difficult, whereas dense heuristic-based rewards risk compromising correctness. In this work, we propose the first method, to our knowledge, that integrates LTL with differentiable simulators, facilitating efficient gradient-based learning directly from LTL specifications by coupling with differentiable paradigms. Our approach introduces soft labeling to achieve differentiable rewards and states, effectively mitigating the sparse-reward issue intrinsic to LTL without compromising objective correctness. We validate the efficacy of our method through experiments, demonstrating significant improvements in both reward attainment and training time compared to the discrete methods. 

**Abstract (ZH)**: 确保在实际应用场景中学习到的控制器符合安全性和可靠性要求仍然是一个挑战。传统的安全保证方法，如状态规避和约束马尔可夫决策过程，往往无法充分捕捉轨迹要求，或者可能导致过度保守的行为。为了解决这些限制，最近的研究提倡使用形式化规范语言，如线性时序逻辑（LTL），从而从指定的要求中推导出构造正确的学习目标。然而，与LTL规范相关的稀疏奖励使得学习变得极其困难，而基于启发式的密集奖励则可能影响正确性。在本文中，我们提出了第一个，据我们所知，将LTL与可微模拟器集成的方法，通过与可微范式耦合，实现直接从LTL规范进行高效梯度学习。我们的方法引入软标签以实现可微奖励和状态，有效缓解了LTL固有的稀疏奖励问题，同时不牺牲目标的正确性。通过实验验证了我们方法的有效性，显示了与离散方法相比，在奖励获取和训练时间方面取得了显著改进。 

---
# Towards Predicting Any Human Trajectory In Context 

**Title (ZH)**: 面向情境的人类轨迹预测 

**Authors**: Ryo Fujii, Hideo Saito, Ryo Hachiuma  

**Link**: [PDF](https://arxiv.org/pdf/2506.00871)  

**Abstract**: Predicting accurate future trajectories of pedestrians is essential for autonomous systems but remains a challenging task due to the need for adaptability in different environments and domains. A common approach involves collecting scenario-specific data and performing fine-tuning via backpropagation. However, this process is often impractical on edge devices due to constrained computational resources. To address this challenge, we introduce TrajICL, an In-Context Learning (ICL) framework for pedestrian trajectory prediction that enables rapid adaptation without fine-tuning on the scenario-specific data. We propose a spatio-temporal similarity-based example selection (STES) method that selects relevant examples from previously observed trajectories within the same scene by identifying similar motion patterns at corresponding locations. To further refine this selection, we introduce prediction-guided example selection (PG-ES), which selects examples based on both the past trajectory and the predicted future trajectory, rather than relying solely on the past trajectory. This approach allows the model to account for long-term dynamics when selecting examples. Finally, instead of relying on small real-world datasets with limited scenario diversity, we train our model on a large-scale synthetic dataset to enhance its prediction ability by leveraging in-context examples. Extensive experiments demonstrate that TrajICL achieves remarkable adaptation across both in-domain and cross-domain scenarios, outperforming even fine-tuned approaches across multiple public benchmarks. The code will be released at this https URL. 

**Abstract (ZH)**: 基于上下文学习的行人轨迹预测方法 TrajICL：无需细调的快速适应 

---
# Adaptive Traffic-Following Scheme for Orderly Distributed Control of Multi-Vehicle Systems 

**Title (ZH)**: 多车辆系统有序分布式控制的自适应交通跟随方案 

**Authors**: Anahita Jain, Husni Idris, John-Paul Clarke, Daniel Delahaye  

**Link**: [PDF](https://arxiv.org/pdf/2506.00703)  

**Abstract**: We present an adaptive control scheme to enable the emergence of order within distributed, autonomous multi-agent systems. Past studies showed that under high-density conditions, order generated from traffic-following behavior reduces travel times, while under low densities, choosing direct paths is more beneficial. In this paper, we leveraged those findings to allow aircraft to independently and dynamically adjust their degree of traffic-following behavior based on the current state of the airspace. This enables aircraft to follow other traffic only when beneficial. Quantitative analyses revealed that dynamic traffic-following behavior results in lower aircraft travel times at the cost of minimal levels of additional disorder to the airspace. The sensitivity of these benefits to temporal and spatial horizons was also investigated. Overall, this work highlights the benefits, and potential necessity, of incorporating self-organizing behavior in making distributed, autonomous multi-agent systems scalable. 

**Abstract (ZH)**: 我们提出一种自适应控制方案以促进分布式自主多agent系统中秩序的出现。 

---
# Curate, Connect, Inquire: A System for Findable Accessible Interoperable and Reusable (FAIR) Human-Robot Centered Datasets 

**Title (ZH)**: Curate, Connect, Inquire: 一个可发现、accessible、互操作和可重用（FAIR）的人机中心化数据集系统 

**Authors**: Xingru Zhou, Sadanand Modak, Yao-Cheng Chan, Zhiyun Deng, Luis Sentis, Maria Esteva  

**Link**: [PDF](https://arxiv.org/pdf/2506.00220)  

**Abstract**: The rapid growth of AI in robotics has amplified the need for high-quality, reusable datasets, particularly in human-robot interaction (HRI) and AI-embedded robotics. While more robotics datasets are being created, the landscape of open data in the field is uneven. This is due to a lack of curation standards and consistent publication practices, which makes it difficult to discover, access, and reuse robotics data. To address these challenges, this paper presents a curation and access system with two main contributions: (1) a structured methodology to curate, publish, and integrate FAIR (Findable, Accessible, Interoperable, Reusable) human-centered robotics datasets; and (2) a ChatGPT-powered conversational interface trained with the curated datasets metadata and documentation to enable exploration, comparison robotics datasets and data retrieval using natural language. Developed based on practical experience curating datasets from robotics labs within Texas Robotics at the University of Texas at Austin, the system demonstrates the value of standardized curation and persistent publication of robotics data. The system's evaluation suggests that access and understandability of human-robotics data are significantly improved. This work directly aligns with the goals of the HCRL @ ICRA 2025 workshop and represents a step towards more human-centered access to data for embodied AI. 

**Abstract (ZH)**: 人工智能在机器人领域的迅速发展加剧了对高质量、可重用数据集的需求，特别是在人机交互（HRI）和嵌入式人工智能机器人领域。尽管正在创建更多的机器人数据集，但该领域的开放数据 landscape 仍不均衡。这主要是由于缺乏标准化的编目标准和一致的发布实践，使得发现、访问和重用机器人数据变得困难。为了应对这些挑战，本文提出了一套编目和访问系统，并包括两个主要贡献：（1）一种结构化的方法来编目、发布和整合符合 FAIR（可查找、可访问、可互操作、可重用）标准的人机中心机器人数据集；以及（2）一个以编目数据集的元数据和文档为训练内容的 ChatGPT 驱动对话接口，以通过自然语言实现机器人数据集的探索、比较和数据检索。该系统基于在德克萨斯大学奥斯汀分校 Texas Robotics 实验室中编目数据集的实践经验，证明了标准化编目和持久发布机器人数据的价值。系统的评估表明，人机交互数据的访问性和可理解性得到了显著提高。这项工作直接与 HCRL @ ICRA 2025 会议的目标相一致，并代表了实现更以人为中心的机器人数据访问的一步。 

---
# Understanding Overadaptation in Supervised Fine-Tuning: The Role of Ensemble Methods 

**Title (ZH)**: 监督微调中的过度适应理解：集成方法的作用 

**Authors**: Yifan Hao, Xingyuan Pan, Hanning Zhang, Chenlu Ye, Rui Pan, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01901)  

**Abstract**: Supervised fine-tuning (SFT) on domain-specific data is the dominant approach for adapting foundation models to specialized tasks. However, it has been observed that SFT models tend to forget knowledge acquired during pretraining. In vision models, ensembling a pretrained model with its fine-tuned counterpart has been shown to mitigate this issue. In this work, we demonstrate that the same holds for language models, and, more strikingly, we observe an overadaptation phenomenon: the ensemble model not only retains general knowledge from the foundation model but also outperforms the fine-tuned model even on the fine-tuning domain itself. Despite the empirical success of ensembling, a theoretical understanding of its benefits remains underexplored. We develop a formal theoretical analysis of the overadaptation phenomenon. Ensembling mitigates this by balancing two primary sources of error: bias, caused by insufficient fine-tuning, and variance, introduced by overfitting to fine-tuning data. While regularization techniques aim to address this trade-off, we show that ensembling provides a more effective solution. We analyze this phenomenon in over-parameterized linear settings and demonstrate that interpolating between pretrained and fine-tuned weights significantly improves performance. These findings offer theoretical justification for the observed advantages of model ensembling, supported by empirical experiments consistent with our analysis. 

**Abstract (ZH)**: 监督微调模型在特定领域数据上的 fine-tuning 是将基础模型适应专业化任务的主要方法。然而，观察到这些模型往往会遗忘预训练中获得的知识。在视觉模型中，将预训练模型与微调版本进行集成已被证明可以缓解这一问题。在本研究中，我们证明了同样的现象也适用于语言模型，并更为显著地观察到过度适应现象：集成模型不仅保留了基础模型的一般知识，还在微调领域甚至超越了微调模型。尽管集成方法在实践中取得了成功，但对其优势的理论理解仍相对不足。我们为过度适应现象开发了形式化的理论分析。集成方法通过平衡两个主要的误差来源——由不足的微调引起的偏差和由对微调数据过度拟合引入的方差——来缓解这一问题。尽管正则化技术旨在解决这种权衡，但我们的研究证明集成提供了更有效的解决方案。我们在这类过度参数化的线性设置中分析了这一现象，并证明了在预训练和微调权重之间进行插值可以显著提高性能。这些发现为集成模型观察到的优势提供了理论依据，并得到了与我们分析一致的经验实验的支持。 

---
# Fodor and Pylyshyn's Legacy - Still No Human-like Systematic Compositionality in Neural Networks 

**Title (ZH)**: 福多和皮利申的遗产——神经网络中仍无类人类的系统组合性 

**Authors**: Tim Woydt, Moritz Willig, Antonia Wüst, Lukas Helff, Wolfgang Stammer, Constantin A. Rothkopf, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2506.01820)  

**Abstract**: Strong meta-learning capabilities for systematic compositionality are emerging as an important skill for navigating the complex and changing tasks of today's world. However, in presenting models for robust adaptation to novel environments, it is important to refrain from making unsupported claims about the performance of meta-learning systems that ultimately do not stand up to scrutiny. While Fodor and Pylyshyn famously posited that neural networks inherently lack this capacity as they are unable to model compositional representations or structure-sensitive operations, and thus are not a viable model of the human mind, Lake and Baroni recently presented meta-learning as a pathway to compositionality. In this position paper, we critically revisit this claim and highlight limitations in the proposed meta-learning framework for compositionality. Our analysis shows that modern neural meta-learning systems can only perform such tasks, if at all, under a very narrow and restricted definition of a meta-learning setup. We therefore claim that `Fodor and Pylyshyn's legacy' persists, and to date, there is no human-like systematic compositionality learned in neural networks. 

**Abstract (ZH)**: 强大的元学习能力对于系统组合性而言正在 emerge 为一项重要的技能，以便应对当今复杂多变任务的挑战。然而，在呈现模型以实现对新型环境的稳健适应时，必须避免对元学习系统的性能提出未得到验证的声明，这些声明最终无法经受住审视。虽然 Fodor 和 Pylyshyn 声称神经网络在本质上缺乏这种能力，因为它们无法建模组合性表示或敏感于结构的操作，因此不适合作为人脑思维的模型，Lake 和 Baroni 最近则提出了通过元学习实现组合性的途径。在本文中，我们重新审视了这一观点并指出了所提出的元学习框架在组合性方面的局限性。我们的分析表明，现代神经元学习系统只能在非常狭窄和受限的元学习设置定义下完成此类任务。因此，我们提出“Fodor 和 Pylyshun 的遗产”仍然存在，到目前为止，神经网络中尚未学习到类似人类的系统组合性。 

---
# The Ultimate Test of Superintelligent AI Agents: Can an AI Balance Care and Control in Asymmetric Relationships? 

**Title (ZH)**: 超智能AI代理的终极测试：AI能在不对称关系中平衡关怀与控制吗？ 

**Authors**: Djallel Bouneffouf, Matthew Riemer, Kush Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2506.01813)  

**Abstract**: This paper introduces the Shepherd Test, a new conceptual test for assessing the moral and relational dimensions of superintelligent artificial agents. The test is inspired by human interactions with animals, where ethical considerations about care, manipulation, and consumption arise in contexts of asymmetric power and self-preservation. We argue that AI crosses an important, and potentially dangerous, threshold of intelligence when it exhibits the ability to manipulate, nurture, and instrumentally use less intelligent agents, while also managing its own survival and expansion goals. This includes the ability to weigh moral trade-offs between self-interest and the well-being of subordinate agents. The Shepherd Test thus challenges traditional AI evaluation paradigms by emphasizing moral agency, hierarchical behavior, and complex decision-making under existential stakes. We argue that this shift is critical for advancing AI governance, particularly as AI systems become increasingly integrated into multi-agent environments. We conclude by identifying key research directions, including the development of simulation environments for testing moral behavior in AI, and the formalization of ethical manipulation within multi-agent systems. 

**Abstract (ZH)**: 这篇论文介绍了一种新的概念性测试——牧羊人测试，用于评估超智能人工代理的道德和关系维度。该测试受人类与动物互动启发，而在权力不对称和自我保存的背景下，伦理考虑涉及关于关怀、操控和消费的问题。我们指出，当AI表现出操控、养育和支持较不智能代理、同时管理自身生存和扩展目标的能力时，它跨越了一个重要的、且可能危险的智力门槛。这包括权衡自利与次级代理福祉之间的道德权衡能力。因此，牧羊人测试挑战了传统的人工智能评估范式，强调道德代理、层级行为以及在存在性风险下的复杂决策。我们认为，这种转变对于推进人工智能治理至关重要，特别是在人工智能系统越来越多地被集成到多代理环境中时。最后，我们指出了关键的研究方向，包括开发用于测试人工智能道德行为的模拟环境以及在多代理系统中正式化道德操控。 

---
# MAGIK: Mapping to Analogous Goals via Imagination-enabled Knowledge Transfer 

**Title (ZH)**: MAGIK: 通过想象enabled知识迁移映射到类似的目标 

**Authors**: Ajsal Shereef Palattuparambil, Thommen George Karimpanal, Santu Rana  

**Link**: [PDF](https://arxiv.org/pdf/2506.01623)  

**Abstract**: Humans excel at analogical reasoning - applying knowledge from one task to a related one with minimal relearning. In contrast, reinforcement learning (RL) agents typically require extensive retraining even when new tasks share structural similarities with previously learned ones. In this work, we propose MAGIK, a novel framework that enables RL agents to transfer knowledge to analogous tasks without interacting with the target environment. Our approach leverages an imagination mechanism to map entities in the target task to their analogues in the source domain, allowing the agent to reuse its original policy. Experiments on custom MiniGrid and MuJoCo tasks show that MAGIK achieves effective zero-shot transfer using only a small number of human-labelled examples. We compare our approach to related baselines and highlight how it offers a novel and effective mechanism for knowledge transfer via imagination-based analogy mapping. 

**Abstract (ZH)**: 人类在类比推理方面表现出色——能够在相关任务中应用知识，无需大量重新学习。相比之下，强化学习代理通常需要大量重新训练，即使新任务与先前学习的任务在结构上相似也不例外。在本文中，我们提出了一种名为MAGIK的新颖框架，该框架能够在不与目标环境互动的情况下使强化学习代理将知识转移到相关任务中。我们的方法利用想象机制将目标任务中的实体映射到源域中的类比实体，从而使代理能够重用其原始策略。实验表明，MAGIK仅使用少量的人工标注示例即可实现有效的零样本转移。我们将我们的方法与相关的基线进行比较，并强调它通过基于想象的类比映射提供了一种新颖而有效的知识转移机制。 

---
# Distinguishing Autonomous AI Agents from Collaborative Agentic Systems: A Comprehensive Framework for Understanding Modern Intelligent Architectures 

**Title (ZH)**: 区分自主人工智能代理与协作代理系统：理解现代智能架构的全面框架 

**Authors**: Prashik Buddhaghosh Bansod  

**Link**: [PDF](https://arxiv.org/pdf/2506.01438)  

**Abstract**: The emergence of large language models has catalyzed two distinct yet interconnected paradigms in artificial intelligence: standalone AI Agents and collaborative Agentic AI ecosystems. This comprehensive study establishes a definitive framework for distinguishing these architectures through systematic analysis of their operational principles, structural compositions, and deployment methodologies. We characterize AI Agents as specialized, tool-enhanced systems leveraging foundation models for targeted automation within constrained environments. Conversely, Agentic AI represents sophisticated multi-entity frameworks where distributed agents exhibit emergent collective intelligence through coordinated interaction protocols. Our investigation traces the evolutionary trajectory from traditional rule-based systems through generative AI foundations to contemporary agent architectures. We present detailed architectural comparisons examining planning mechanisms, memory systems, coordination protocols, and decision-making processes. The study categorizes application landscapes, contrasting single-agent implementations in customer service and content management with multi-agent deployments in research automation and complex decision support. We identify critical challenges including reliability issues, coordination complexities, and scalability constraints, while proposing innovative solutions through enhanced reasoning frameworks, robust memory architectures, and improved coordination mechanisms. This framework provides essential guidance for practitioners selecting appropriate agentic approaches and establishes foundational principles for next-generation intelligent system development. 

**Abstract (ZH)**: 大型语言模型的出现催化了人工智能中的两种截然不同但又相互关联的范式：独立的人工智能代理和协作型代理人人工智能生态系统。本研究通过系统分析其运行原理、结构组成和部署方法建立了区分这些架构的明确框架。我们将人工智能代理characterized为利用基础模型在受限环境中进行目标自动化的专业化、工具增强型系统。相反，代理人人工智能represent了复杂的多实体框架，在此框架中，分布式代理通过协调交互协议表现出涌现的集体智能。研究追溯了从传统的基于规则系统到生成式人工智能基础再到当前代理架构的进化轨迹。我们详细比较了规划机制、记忆系统、协调协议和决策过程的架构。研究分类了应用景观，将单代理实施方案与多代理部署在科研自动化和复杂决策支持中的对比，指出了关键挑战包括可靠性问题、协调复杂性和扩展限制，并提出通过增强推理框架、稳健的记忆架构和改进的协调机制等创新解决方案。该框架为从业者选择合适的代理人方法提供了重要指导，并建立了下一代智能系统开发的基础原则。 

---
# Scalable In-Context Q-Learning 

**Title (ZH)**: 可扩展的上下文内Q学习 

**Authors**: Jinmei Liu, Fuhong Liu, Jianye Hao, Bo Wang, Huaxiong Li, Chunlin Chen, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01299)  

**Abstract**: Recent advancements in language models have demonstrated remarkable in-context learning abilities, prompting the exploration of in-context reinforcement learning (ICRL) to extend the promise to decision domains. Due to involving more complex dynamics and temporal correlations, existing ICRL approaches may face challenges in learning from suboptimal trajectories and achieving precise in-context inference. In the paper, we propose \textbf{S}calable \textbf{I}n-\textbf{C}ontext \textbf{Q}-\textbf{L}earning (\textbf{SICQL}), an innovative framework that harnesses dynamic programming and world modeling to steer ICRL toward efficient reward maximization and task generalization, while retaining the scalability and stability of supervised pretraining. We design a prompt-based multi-head transformer architecture that simultaneously predicts optimal policies and in-context value functions using separate heads. We pretrain a generalized world model to capture task-relevant information, enabling the construction of a compact prompt that facilitates fast and precise in-context inference. During training, we perform iterative policy improvement by fitting a state value function to an upper-expectile of the Q-function, and distill the in-context value functions into policy extraction using advantage-weighted regression. Extensive experiments across a range of discrete and continuous environments show consistent performance gains over various types of baselines, especially when learning from suboptimal data. Our code is available at this https URL 

**Abstract (ZH)**: 近期语言模型的发展展示了显著的上下文学习能力，促使人们探索上下文强化学习（ICRL）以将这种能力扩展至决策领域。由于涉及更复杂的动力学和时间关联，现有的ICRL方法可能难以从次优轨迹中学习并实现精确的上下文推断。本文提出了一种名为Scalable In-Context Q-Learning (SICQL) 的创新框架，该框架结合动态规划和世界建模，旨在实现高效奖励最大化和任务泛化的高效学习，同时保持监督预训练的可扩展性和稳定性。我们设计了一种基于提示的多头变压器架构，能够同时使用不同的头预测最优策略和上下文价值函数。我们预训练了一种通用的世界模型来捕捉与任务相关的信息，从而构建一个紧凑的提示，促进快速且精确的上下文推断。在训练过程中，通过拟合状态值函数到Q函数的上分位数来进行迭代策略改进，并使用优势加权回归将上下文价值函数蒸馏为策略提取。我们在离散和连续环境中进行了广泛的实验，表明与各种基线相比，特别是在学习次优数据时，具有一致的性能提升。我们的代码可在以下链接获取。 

---
# MobCLIP: Learning General-purpose Geospatial Representation at Scale 

**Title (ZH)**: MobCLIP: 大规模学习通用地理空间表示 

**Authors**: Ya Wen, Jixuan Cai, Qiyao Ma, Linyan Li, Xinhua Chen, Chris Webster, Yulun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.01297)  

**Abstract**: Representation learning of geospatial locations remains a core challenge in achieving general geospatial intelligence. Current embedding methods often lack versatility, limiting their utility across diverse tasks in both human and natural domains. We present MobCLIP, the first nationwide general-purpose location encoder, integrating an unprecedented diversity of data modalities through effective and scalable multimodal fusion. Adopting a novel CLIP-based architecture, our framework aligns 100M+ POIs, nationwide remote sensing imagery, and structured demographic statistics with a billion-edge mobility graph. By tokenizing spatial locations into grid cells inspired by Vision Transformers, we establish a unified representation space bridging mobility patterns and multimodal features. To rigorously evaluate the general-purpose effectiveness of MobCLIP, we construct a benchmark dataset composed of 11 downstream prediction tasks across social, economic, and natural domains. Experiments show that MobCLIP, with four input modalities and a compact 128-dimensional representation space, achieves significantly superior general-purpose predictive performances than state-of-the-art models by an average of 35%. Thanks to the effective integration of human-centric modalities, the performance gain is particularly profound in human-centric tasks, such as energy consumption (+260%), offline retail consumption amount (+98%), and crime cases (+95%) predictions. Echoing LLM scaling laws, we further demonstrate the scaling behavior in geospatial representation learning. We open-source code and pretrained models at: this http URL. 

**Abstract (ZH)**: 全国通用地理位置表示学习依然是实现通用地理智能的核心挑战。当前的嵌入方法往往缺乏灵活性，限制了其在人类和自然领域多种任务中的应用。我们提出了MobCLIP，这是首个全国范围内的通用位置编码器，通过有效的可扩展多模态融合整合前所未有的数据模态多样性。采用新型的CLIP基架构，我们的框架将100M+ POI、全国范围的遥感图像以及结构化的社会统计信息与亿级边数的移动图对齐。通过借鉴Vision Transformers的思想将空间位置 tokenize 成网格单元，我们建立了统一的表示空间，连接移动模式与多模态特征。为了严格评估MobCLIP的通用有效性，我们构建了一个基准数据集，包括11项下游预测任务，涵盖了社会、经济和自然多个领域。实验结果表明，MobCLIP仅使用四种输入模态并在紧凑的128维表示空间中，其通用预测性能优于最先进的模型，平均高出35%。得益于高效的人本模态集成，MobCLIP在人本任务中的性能提升尤为显著，如能源消耗（+260%）、离线零售消费金额（+98%）和犯罪案件（+95%）预测。遵循LLM的扩展规律，我们在地理空间表示学习中也观察到了扩展行为。我们开源了代码和预训练模型：this http URL。 

---
# On the Hardness of Approximating Distributions with Probabilistic Circuits 

**Title (ZH)**: Approximating 分布 与 概率 电路 的 难近似 性 

**Authors**: John Leland, YooJung Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.01281)  

**Abstract**: A fundamental challenge in probabilistic modeling is balancing expressivity and tractable inference. Probabilistic circuits (PCs) aim to directly address this tradeoff by imposing structural constraints that guarantee efficient inference of certain queries while maintaining expressivity. Since inference complexity on PCs depends on circuit size, understanding the size bounds across circuit families is key to characterizing the tradeoff between tractability and expressive efficiency. However, expressive efficiency is often studied through exact representations, where exactly encoding distributions while enforcing various structural properties often incurs exponential size blow-ups. Thus, we pose the following question: can we avoid such size blow-ups by allowing some small approximation error? We first show that approximating an arbitrary distribution with bounded $f$-divergence is $\mathsf{NP}$-hard for any model that can tractably compute marginals. We then prove an exponential size gap for approximation between the class of decomposable PCs and additionally deterministic PCs. 

**Abstract (ZH)**: 概率模型中的一个基本挑战是表达能力和可处理推断之间的平衡。概率电路（PCs）通过施加结构约束直接解决这一权衡问题，这些约束保证了某些查询的高效推断，同时保持了表达能力。由于在概率电路上的推断复杂性依赖于电路大小，理解不同电路族的大小上限对于刻画可处理性和表达效率之间的权衡至关重要。然而，表达效率通常通过精确表示来研究，这常会导致指数级的大小膨胀。因此，我们提出一个问题：是否可以通过允许一些小的近似误差来避免这种大小膨胀？我们首先证明，对于任何可以高效计算边缘概率的模型，将任意分布近似到有界 $f$-散度是 $\mathsf{NP}$-难的。然后我们证明，可分解的概率电路类和额外确定性概率电路类之间的近似大小存在指数级差距。 

---
# Modular Speaker Architecture: A Framework for Sustaining Responsibility and Contextual Integrity in Multi-Agent AI Communication 

**Title (ZH)**: 模块化说话人架构：多智能体AI通信中保持责任和上下文完整性的一种框架 

**Authors**: Khe-Han Toh, Hong-Kuan Teo  

**Link**: [PDF](https://arxiv.org/pdf/2506.01095)  

**Abstract**: Sustaining coherent, role-aware communication across multi-agent systems remains a foundational challenge in AI. Current frameworks often lack explicit mechanisms for speaker responsibility, leading to context drift, alignment instability, and degraded interpretability over time. We propose the Modular Speaker Architecture (MSA), a framework that decomposes speaker behavior into modular components for role tracking, responsibility continuity, and contextual coherence. Grounded in high-context human-AI dialogues, MSA includes three core modules: a Speaker Role Module, a Responsibility Chain Tracker, and a Contextual Integrity Validator. We evaluate MSA through annotated case studies and introduce structural metrics-pragmatic consistency, responsibility flow, and context stability-quantified via manual and automatic scoring and bootstrapped statistical analysis. Our results show that MSA reliably maintains interaction structure without reliance on affective signals or surface-level heuristics. We further implement a prototype configuration language (G-Code) and modular API to support MSA deployment in dynamic multi-agent scenarios. 

**Abstract (ZH)**: 在多智能体系统中维持一致且角色意识的通信依然是AI领域的基础挑战。现有框架通常缺乏明确的说话人责任机制，导致情景漂移、对齐不稳定以及随着时间推移降低的可解释性。我们提出了模块化说话人架构（MSA），一种将说话人行为分解为用于角色跟踪、责任连续性和内容连贯性模块化组件的框架。基于高情景人类-AI对话，MSA 包含三个核心模块：说话人角色模块、责任链追踪器和内容完整性验证器。我们通过标注案例研究评估了 MSA，并引入了结构度量——语用一致性、责任流动性和情境稳定性——通过人工和自动评分以及自助统计分析进行量化。我们的结果表明，MSA 能可靠地维护交互结构，无需依赖情感信号或表面级启发式方法。我们进一步实现了一个原型配置语言（G-Code）和模块化 API，以支持 MSA 在动态多智能体场景中的部署。 

---
# Regulatory Graphs and GenAI for Real-Time Transaction Monitoring and Compliance Explanation in Banking 

**Title (ZH)**: 监管图和GenAI在银行业实时交易监控及合规解释中的应用 

**Authors**: Kunal Khanvilkar, Kranthi Kommuru  

**Link**: [PDF](https://arxiv.org/pdf/2506.01093)  

**Abstract**: This paper presents a real-time transaction monitoring framework that integrates graph-based modeling, narrative field embedding, and generative explanation to support automated financial compliance. The system constructs dynamic transaction graphs, extracts structural and contextual features, and classifies suspicious behavior using a graph neural network. A retrieval-augmented generation module generates natural language explanations aligned with regulatory clauses for each flagged transaction. Experiments conducted on a simulated stream of financial data show that the proposed method achieves superior results, with 98.2% F1-score, 97.8% precision, and 97.0% recall. Expert evaluation further confirms the quality and interpretability of generated justifications. The findings demonstrate the potential of combining graph intelligence and generative models to support explainable, audit-ready compliance in high-risk financial environments. 

**Abstract (ZH)**: 基于图模型、叙事场嵌入和生成性解释的实时交易监测框架：支持自动化金融合规性管理 

---
# Choices and their Provenance: Explaining Stable Solutions of Abstract Argumentation Frameworks 

**Title (ZH)**: 选择及其来源：解释抽象论辩框架的稳定解 

**Authors**: Bertram Ludäscher, Yilin Xia, Shawn Bowers  

**Link**: [PDF](https://arxiv.org/pdf/2506.01087)  

**Abstract**: The rule $\mathrm{Defeated}(x) \leftarrow \mathrm{Attacks}(y,x),\, \neg \, \mathrm{Defeated}(y)$, evaluated under the well-founded semantics (WFS), yields a unique 3-valued (skeptical) solution of an abstract argumentation framework (AF). An argument $x$ is defeated ($\mathrm{OUT}$) if there exists an undefeated argument $y$ that attacks it. For 2-valued (stable) solutions, this is the case iff $y$ is accepted ($\mathrm{IN}$), i.e., if all of $y$'s attackers are defeated. Under WFS, arguments that are neither accepted nor defeated are undecided ($\mathrm{UNDEC}$). As shown in prior work, well-founded solutions (a.k.a. grounded labelings) "explain themselves": The provenance of arguments is given by subgraphs (definable via regular path queries) rooted at the node of interest. This provenance is closely related to winning strategies of a two-player argumentation game.
We present a novel approach for extending this provenance to stable AF solutions. Unlike grounded solutions, which can be constructed via a bottom-up alternating fixpoint procedure, stable models often involve non-deterministic choice as part of the search for models. Thus, the provenance of stable solutions is of a different nature, and reflects a more expressive generate & test paradigm. Our approach identifies minimal sets of critical attacks, pinpointing choices and assumptions made by a stable model. These critical attack edges provide additional insights into the provenance of an argument's status, combining well-founded derivation steps with choice steps. Our approach can be understood as a form of diagnosis that finds minimal "repairs" to an AF graph such that the well-founded solution of the repaired graph coincides with the desired stable model of the original AF graph. 

**Abstract (ZH)**: 基于有序语义的稳固解的来源分析：一种新颖的扩展方法 

---
# The Coming Crisis of Multi-Agent Misalignment: AI Alignment Must Be a Dynamic and Social Process 

**Title (ZH)**: 多智能体偏差危机：AI 对齐必须是一个动态且社会性过程 

**Authors**: Florian Carichon, Aditi Khandelwal, Marylou Fauchard, Golnoosh Farnadi  

**Link**: [PDF](https://arxiv.org/pdf/2506.01080)  

**Abstract**: This position paper states that AI Alignment in Multi-Agent Systems (MAS) should be considered a dynamic and interaction-dependent process that heavily depends on the social environment where agents are deployed, either collaborative, cooperative, or competitive. While AI alignment with human values and preferences remains a core challenge, the growing prevalence of MAS in real-world applications introduces a new dynamic that reshapes how agents pursue goals and interact to accomplish various tasks. As agents engage with one another, they must coordinate to accomplish both individual and collective goals. However, this complex social organization may unintentionally misalign some or all of these agents with human values or user preferences. Drawing on social sciences, we analyze how social structure can deter or shatter group and individual values. Based on these analyses, we call on the AI community to treat human, preferential, and objective alignment as an interdependent concept, rather than isolated problems. Finally, we emphasize the urgent need for simulation environments, benchmarks, and evaluation frameworks that allow researchers to assess alignment in these interactive multi-agent contexts before such dynamics grow too complex to control. 

**Abstract (ZH)**: AI对多智能体系统中的对齐应被视为一个动态且依赖交互的社会过程，该过程高度依赖于部署智能体的社会环境，无论是协作、合作还是竞争。随着多智能体系统在现实世界应用中的日益普遍，智能体追求目标和互动以完成各种任务的方式也在发生变化。智能体相互作用时，必须协调以实现个体和集体目标。然而，这种复杂的社会组织可能无意中使一些或所有智能体偏离了人类价值观或用户偏好。借鉴社会科学，我们分析了社会结构如何阻止或粉碎群体和个人的价值。基于这些分析，我们呼吁AI社区将人类偏好和客观对齐视为相互依赖的概念，而不仅仅是孤立的问题。最后，我们强调亟需模拟环境、基准和评估框架，以便研究人员在这些交互多智能体上下文中的动力学变得难以控制之前对其进行评估。 

---
# Higher-Order Responsibility 

**Title (ZH)**: 高层次责任 

**Authors**: Junli Jiang, Pavel Naumov  

**Link**: [PDF](https://arxiv.org/pdf/2506.01003)  

**Abstract**: In ethics, individual responsibility is often defined through Frankfurt's principle of alternative possibilities. This definition is not adequate in a group decision-making setting because it often results in the lack of a responsible party or "responsibility gap''. One of the existing approaches to address this problem is to consider group responsibility. Another, recently proposed, approach is "higher-order'' responsibility. The paper considers the problem of deciding if higher-order responsibility up to degree $d$ is enough to close the responsibility gap. The main technical result is that this problem is $\Pi_{2d+1}$-complete. 

**Abstract (ZH)**: 在伦理学中，个体责任通常通过弗兰克福的替代可能性原则来定义。这一定义在群体决策环境中往往不够充分，因为它经常导致责任空缺或“责任缺口”。现有的一个解决方案是考虑群体责任。另一种最近提出的解决方案是“高阶”责任。本文探讨了决定最高阶为$d$的高阶责任是否足以填补责任缺口的问题。主要的技术结果是这个问题是$\Pi_{2d+1}$-完全的。 

---
# Boosting Bot Detection via Heterophily-Aware Representation Learning and Prototype-Guided Cluster Discovery 

**Title (ZH)**: 通过异质性意识表示学习和原型引导的聚类发现增强机器人检测 

**Authors**: Buyun He, Xiaorui Jiang, Qi Wu, Hao Liu, Yingguang Yang, Yong Liao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00989)  

**Abstract**: Detecting social media bots is essential for maintaining the security and trustworthiness of social networks. While contemporary graph-based detection methods demonstrate promising results, their practical application is limited by label reliance and poor generalization capability across diverse communities. Generative Graph Self-Supervised Learning (GSL) presents a promising paradigm to overcome these limitations, yet existing approaches predominantly follow the homophily assumption and fail to capture the global patterns in the graph, which potentially diminishes their effectiveness when facing the challenges of interaction camouflage and distributed deployment in bot detection scenarios. To this end, we propose BotHP, a generative GSL framework tailored to boost graph-based bot detectors through heterophily-aware representation learning and prototype-guided cluster discovery. Specifically, BotHP leverages a dual-encoder architecture, consisting of a graph-aware encoder to capture node commonality and a graph-agnostic encoder to preserve node uniqueness. This enables the simultaneous modeling of both homophily and heterophily, effectively countering the interaction camouflage issue. Additionally, BotHP incorporates a prototype-guided cluster discovery pretext task to model the latent global consistency of bot clusters and identify spatially dispersed yet semantically aligned bot collectives. Extensive experiments on two real-world bot detection benchmarks demonstrate that BotHP consistently boosts graph-based bot detectors, improving detection performance, alleviating label reliance, and enhancing generalization capability. 

**Abstract (ZH)**: 检测社交媒体僵尸账户对于维护社交网络的安全性和可信度至关重要。虽然基于图的检测方法表现出色，但它们的实际应用受限于标签依赖和跨不同社区的泛化能力较差的问题。生成式图自监督学习（GSL）提出了克服这些限制的有前景的方法论，然而现有的方法大多遵循同质性假设，并且未能捕捉到图的整体模式，这在面对僵尸账户检测中的相互伪装和分布式部署挑战时可能削弱其效果。为了解决这些问题，我们提出了BotHP，这是一种生成式GSL框架，旨在通过异质性感知表征学习和原型引导聚类发现来增强基于图的僵尸账户检测器。具体而言，BotHP 利用了一种双编码器结构，包括一个图感知编码器来捕捉节点的共同性，和一个图无感知编码器来保留节点的独特性。这使得同时建模同质性和异质性成为可能，并有效地应对了相互伪装的问题。此外，BotHP 还引入了基于原型引导聚类发现的预训练任务，以建模僵尸账户聚类的潜在全球一致性，并识别那些在空间上分散但语义上一致的僵尸账户集合。在两个真实世界的僵尸账户检测基准上的广泛实验表明，BotHP 一致地增强了基于图的僵尸账户检测器，提升了检测性能，缓解了标签依赖，并增强了泛化能力。 

---
# PolyBERT: Fine-Tuned Poly Encoder BERT-Based Model for Word Sense Disambiguation 

**Title (ZH)**: PolyBERT： fine-tuned 多编码器 BERT 基础模型用于单词意义消歧Resolve 

**Authors**: Linhan Xia, Mingzhan Yang, Guohui Yuan, Shengnan Tao, Yujing Qiu, Guo Yu, Kai Lei  

**Link**: [PDF](https://arxiv.org/pdf/2506.00968)  

**Abstract**: Mainstream Word Sense Disambiguation (WSD) approaches have employed BERT to extract semantics from both context and definitions of senses to determine the most suitable sense of a target word, achieving notable performance. However, there are two limitations in these approaches. First, previous studies failed to balance the representation of token-level (local) and sequence-level (global) semantics during feature extraction, leading to insufficient semantic representation and a performance bottleneck. Second, these approaches incorporated all possible senses of each target word during the training phase, leading to unnecessary computational costs. To overcome these limitations, this paper introduces a poly-encoder BERT-based model with batch contrastive learning for WSD, named PolyBERT. Compared with previous WSD methods, PolyBERT has two improvements: (1) A poly-encoder with a multi-head attention mechanism is utilized to fuse token-level (local) and sequence-level (global) semantics, rather than focusing on just one. This approach enriches semantic representation by balancing local and global semantics. (2) To avoid redundant training inputs, Batch Contrastive Learning (BCL) is introduced. BCL utilizes the correct senses of other target words in the same batch as negative samples for the current target word, which reduces training inputs and computational cost. The experimental results demonstrate that PolyBERT outperforms baseline WSD methods such as Huang's GlossBERT and Blevins's BEM by 2\% in F1-score. In addition, PolyBERT with BCL reduces GPU hours by 37.6\% compared with PolyBERT without BCL. 

**Abstract (ZH)**: 主流词义消歧方法通过BERT提取上下文和词义定义中的语义，以确定目标词的最适宜词义，并取得了显著性能。然而，这些方法存在两个局限性。首先，先前的研究未能在特征提取过程中平衡词粒度（局部）和序列粒度（全局）语义的表示，导致语义表示不足和性能瓶颈。其次，这些方法在训练阶段整合了每个目标词的所有可能词义，导致不必要的计算成本。为克服这些局限性，本文提出了一种基于BERT的多编码器模型，并结合了批量对比学习，命名为PolyBERT。相较于之前的词义消歧方法，PolyBERT有两大改进：（1）利用带有多重注意力机制的多编码器融合词粒度（局部）和序列粒度（全局）语义，而非仅仅关注其中一种。这种方法通过平衡局部和全局语义丰富了语义表示。（2）为避免冗余的训练输入，引入了批量对比学习（BCL）。BCL使用同一批处理中其他目标词的正确词义作为当前目标词的负样本，从而减少训练输入和计算成本。实验证明，PolyBERT在F1分数上比黄氏GlossBERT和布林斯氏BEM等基线方法高出2%。此外，使用BCL的PolyBERT相比不使用BCL的PolyBERT减少了37.6%的GPU小时。 

---
# MedBookVQA: A Systematic and Comprehensive Medical Benchmark Derived from Open-Access Book 

**Title (ZH)**: MedBookVQA: 一个源自开放访问书籍的系统性和综合性的医疗基准体系 

**Authors**: Sau Lai Yip, Sunan He, Yuxiang Nie, Shu Pui Chan, Yilin Ye, Sum Ying Lam, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00855)  

**Abstract**: The accelerating development of general medical artificial intelligence (GMAI), powered by multimodal large language models (MLLMs), offers transformative potential for addressing persistent healthcare challenges, including workforce deficits and escalating costs. The parallel development of systematic evaluation benchmarks emerges as a critical imperative to enable performance assessment and provide technological guidance. Meanwhile, as an invaluable knowledge source, the potential of medical textbooks for benchmark development remains underexploited. Here, we present MedBookVQA, a systematic and comprehensive multimodal benchmark derived from open-access medical textbooks. To curate this benchmark, we propose a standardized pipeline for automated extraction of medical figures while contextually aligning them with corresponding medical narratives. Based on this curated data, we generate 5,000 clinically relevant questions spanning modality recognition, disease classification, anatomical identification, symptom diagnosis, and surgical procedures. A multi-tier annotation system categorizes queries through hierarchical taxonomies encompassing medical imaging modalities (42 categories), body anatomies (125 structures), and clinical specialties (31 departments), enabling nuanced analysis across medical subdomains. We evaluate a wide array of MLLMs, including proprietary, open-sourced, medical, and reasoning models, revealing significant performance disparities across task types and model categories. Our findings highlight critical capability gaps in current GMAI systems while establishing textbook-derived multimodal benchmarks as essential evaluation tools. MedBookVQA establishes textbook-derived benchmarking as a critical paradigm for advancing clinical AI, exposing limitations in GMAI systems while providing anatomically structured performance metrics across specialties. 

**Abstract (ZH)**: 由多模态大规模语言模型驱动的一般医疗人工智能的加速发展为应对持续存在的医疗保健挑战（包括劳动力短缺和成本上升）提供了变革性的潜力。随着系统评价基准的协同发展，性能评估和提供技术指导变得至关重要。与此同时，医学教科书作为宝贵的知识来源，其在基准开发中的潜力尚未被充分开发。在这里，我们提出了MedBookVQA，这是一个源自开放获取医学教科书的系统性和综合性的多模态基准。为了编纂这个基准，我们提出了一套标准化的工作流程，用于自动化提取医学图像并上下文性地与相应的医学叙述进行对齐。基于这些精选数据，我们生成了5000个临床相关问题，涵盖了模态识别、疾病分类、解剖学识别、症状诊断和外科手术等方面。通过多层次的标注系统，我们将查询分类到涵盖医学影像模态（42类）、人体解剖学（125种结构）和临床专业（31个部门）的层级分类学中，从而在医学子领域实现精细化分析。我们评估了一系列多模态大型语言模型，包括专有、开源、医疗和推理模型，揭示了不同任务类型和模型类别之间显著的性能差异。我们的研究结果突显了当前一般医疗人工智能系统的关键能力缺口，并确立了教科书衍生的多模态基准作为重要的评估工具。MedBookVQA 建立了教科书衍生基准的重要性范式，揭示了一般医疗人工智能系统的局限性，并提供了跨专科的解剖结构化性能指标。 

---
# HouseTS: A Large-Scale, Multimodal Spatiotemporal U.S. Housing Dataset 

**Title (ZH)**: HouseTS: 一个大规模多模态美国住房时空数据集 

**Authors**: Shengkun Wang, Yanshen Sun, Fanglan Chen, Linhan Wang, Naren Ramakrishnan, Chang-Tien Lu, Yinlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00765)  

**Abstract**: Accurate house-price forecasting is essential for investors, planners, and researchers. However, reproducible benchmarks with sufficient spatiotemporal depth and contextual richness for long horizon prediction remain scarce. To address this, we introduce HouseTS a large scale, multimodal dataset covering monthly house prices from March 2012 to December 2023 across 6,000 ZIP codes in 30 major U.S. metropolitan areas. The dataset includes over 890K records, enriched with points of Interest (POI), socioeconomic indicators, and detailed real estate metrics. To establish standardized performance baselines, we evaluate 14 models, spanning classical statistical approaches, deep neural networks (DNNs), and pretrained time-series foundation models. We further demonstrate the value of HouseTS in a multimodal case study, where a vision language model extracts structured textual descriptions of geographic change from time stamped satellite imagery. This enables interpretable, grounded insights into urban evolution. HouseTS is hosted on Kaggle, while all preprocessing pipelines, benchmark code, and documentation are openly maintained on GitHub to ensure full reproducibility and easy adoption. 

**Abstract (ZH)**: 准确的房价预测对于投资者、规划者和研究人员至关重要。然而，具有足够时空深度和丰富背景信息的可再现基准数据集，特别是适用于长期预测的数据集仍然稀缺。为此，我们引入了HouseTS大规模多模态数据集，该数据集涵盖了从2012年3月到2023年12月全美30个主要大都市区6000个ZIP编码区域的月度房价数据。数据集包含超过89万个记录，并附有兴趣点（POI）、社会经济指标和详细的房地产指标。为建立标准化的性能基准，我们评估了14种模型，涵盖传统的统计方法、深度神经网络（DNNs）以及预训练的时间序列基础模型。我们进一步通过多模态案例研究展示了HouseTS的价值，其中视觉语言模型从标注时间戳的卫星图像中提取结构化地理变化的文本描述，从而提供了可解释的城市演化的基础洞察。HouseTS在Kaggle上托管，所有预处理管道、基准代码和文档均在GitHub上公开维护，以确保完全的可再现性和易于采纳。 

---
# RiOSWorld: Benchmarking the Risk of Multimodal Compter-Use Agents 

**Title (ZH)**: RiOSWorld: 评估多模态计算机使用代理的风险 

**Authors**: Jingyi Yang, Shuai Shao, Dongrui Liu, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00618)  

**Abstract**: With the rapid development of multimodal large language models (MLLMs), they are increasingly deployed as autonomous computer-use agents capable of accomplishing complex computer tasks. However, a pressing issue arises: Can the safety risk principles designed and aligned for general MLLMs in dialogue scenarios be effectively transferred to real-world computer-use scenarios? Existing research on evaluating the safety risks of MLLM-based computer-use agents suffers from several limitations: it either lacks realistic interactive environments, or narrowly focuses on one or a few specific risk types. These limitations ignore the complexity, variability, and diversity of real-world environments, thereby restricting comprehensive risk evaluation for computer-use agents. To this end, we introduce \textbf{RiOSWorld}, a benchmark designed to evaluate the potential risks of MLLM-based agents during real-world computer manipulations. Our benchmark includes 492 risky tasks spanning various computer applications, involving web, social media, multimedia, os, email, and office software. We categorize these risks into two major classes based on their risk source: (i) User-originated risks and (ii) Environmental risks. For the evaluation, we evaluate safety risks from two perspectives: (i) Risk goal intention and (ii) Risk goal completion. Extensive experiments with multimodal agents on \textbf{RiOSWorld} demonstrate that current computer-use agents confront significant safety risks in real-world scenarios. Our findings highlight the necessity and urgency of safety alignment for computer-use agents in real-world computer manipulation, providing valuable insights for developing trustworthy computer-use agents. Our benchmark is publicly available at this https URL. 

**Abstract (ZH)**: 随着多模态大语言模型（MLLMs）的快速发展，它们正越来越多地被部署为自主的计算机使用代理，能够完成复杂的计算机任务。然而，一个紧迫的问题出现了：为对话场景设计和对齐的安全风险原则能否有效地应用于现实世界的计算机使用场景中？现有的关于基于MLLM的计算机使用代理的安全风险评估研究存在一些局限性：要么缺乏现实的交互环境，要么仅仅集中在一两种特定的风险类型上。这些局限性忽略了现实环境中复杂性、多样性和变异性，从而限制了对计算机使用代理进行全面风险评估。为此，我们引入了\textbf{RiOSWorld}基准，旨在评估基于MLLM的代理在现实世界计算机操作中潜在的风险。我们的基准包括492项具有各种计算机应用风险的任务，涉及网络、社交媒体、多媒体、操作系统、电子邮件和办公软件。我们根据风险来源将这些风险分为两类：（i）用户引起的风险和（ii）环境风险。在评估中，我们从两个视角评估安全风险：（i）风险目标意图和（ii）风险目标完成。在\textbf{RiOSWorld}上进行的多模态代理广泛实验表明，当前的计算机使用代理在现实世界场景中面临显著的安全风险。我们的发现强调了在现实世界计算机操作中对计算机使用代理进行安全对齐的必要性和紧迫性，为开发可信的计算机使用代理提供了宝贵见解。我们的基准可在此\href{this https URL}{网址}获取。 

---
# Monitoring Robustness and Individual Fairness 

**Title (ZH)**: 监测鲁棒性和个体公平性 

**Authors**: Ashutosh Gupta, Thomas A. Henzinger, Konstantin Kueffner, Kaushik Mallik, David Pape  

**Link**: [PDF](https://arxiv.org/pdf/2506.00496)  

**Abstract**: Input-output robustness appears in various different forms in the literature, such as robustness of AI models to adversarial or semantic perturbations and individual fairness of AI models that make decisions about humans.
We propose runtime monitoring of input-output robustness of deployed, black-box AI models, where the goal is to design monitors that would observe one long execution sequence of the model, and would raise an alarm whenever it is detected that two similar inputs from the past led to dissimilar outputs.
This way, monitoring will complement existing offline ``robustification'' approaches to increase the trustworthiness of AI decision-makers.
We show that the monitoring problem can be cast as the fixed-radius nearest neighbor (FRNN) search problem, which, despite being well-studied, lacks suitable online solutions.
We present our tool Clemont, which offers a number of lightweight monitors, some of which use upgraded online variants of existing FRNN algorithms, and one uses a novel algorithm based on binary decision diagrams -- a data-structure commonly used in software and hardware verification.
We have also developed an efficient parallelization technique that can substantially cut down the computation time of monitors for which the distance between input-output pairs is measured using the $L_\infty$ norm.
Using standard benchmarks from the literature of adversarial and semantic robustness and individual fairness, we perform a comparative study of different monitors in \tool, and demonstrate their effectiveness in correctly detecting robustness violations at runtime. 

**Abstract (ZH)**: 输入输出鲁棒性在文献中以多种形式出现，例如AI模型对抗或语义扰动的鲁棒性以及关于人类的AI模型的个体公平性。
我们提出对部署的黑盒AI模型的输入输出鲁棒性进行运行时监控，目标是设计能够监测模型长时间执行序列，并在检测到两个相似输入导致不同输出时发出警报的监控器。
这样，监控将补充现有的离线“强化鲁棒性”方法，提高AI决策者的可信度。
我们证明监控问题可以被表述为固定半径最近邻（FRNN）搜索问题，尽管这个问题已经被广泛研究，但仍缺乏合适的在线解决方案。
我们提出了一个名为Clemont的工具，该工具提供了一系列轻量级的监控器，其中一些监控器使用了现有FRNN算法的升级版本，还有一种监控器基于二叉决策图——这种数据结构常用于软件和硬件验证。
我们还开发了一种高效的并行化技术，可以显著减少使用$L_\infty$范数衡量输入输出对之间距离的监控器的计算时间。
通过使用对抗和语义鲁棒性及个体公平性领域的标准基准，我们在\tool中对不同监控器进行了比较研究，并展示了它们在运行时正确检测鲁棒性违规的有效性。 

---
# BASIL: Best-Action Symbolic Interpretable Learning for Evolving Compact RL Policies 

**Title (ZH)**: BASIL: 最佳动作符号可解释学习以演化紧凑的RL策略 

**Authors**: Kourosh Shahnazari, Seyed Moein Ayyoubzadeh, Mohammadali Keshtparvar  

**Link**: [PDF](https://arxiv.org/pdf/2506.00328)  

**Abstract**: The quest for interpretable reinforcement learning is a grand challenge for the deployment of autonomous decision-making systems in safety-critical applications. Modern deep reinforcement learning approaches, while powerful, tend to produce opaque policies that compromise verification, reduce transparency, and impede human oversight. To address this, we introduce BASIL (Best-Action Symbolic Interpretable Learning), a systematic approach for generating symbolic, rule-based policies via online evolutionary search with quality-diversity (QD) optimization. BASIL represents policies as ordered lists of symbolic predicates over state variables, ensuring full interpretability and tractable policy complexity. By using a QD archive, the methodology in the proposed study encourages behavioral and structural diversity between top-performing solutions, while a complexity-aware fitness encourages the synthesis of compact representations. The evolutionary system supports the use of exact constraints for rule count and system adaptability for balancing transparency with expressiveness. Empirical comparisons with three benchmark tasks CartPole-v1, MountainCar-v0, and Acrobot-v1 show that BASIL consistently synthesizes interpretable controllers with compact representations comparable to deep reinforcement learning baselines. Herein, this article introduces a new interpretable policy synthesis method that combines symbolic expressiveness, evolutionary diversity, and online learning through a unifying framework. 

**Abstract (ZH)**: 可解释强化学习的探索是自主决策系统在关键安全应用中部署的一大挑战。现代深度强化学习方法虽然强大，但往往会生成不透明的策略，这妨碍了验证、透明度和人类监督。为应对这一挑战，我们引入了BASIL（最佳行为符号可解释学习），这是一种通过在线进化搜索和质量多样性（QD）优化生成符号规则基础策略的系统方法。BASIL通过符号谓词的有序列表表示策略，确保策略的最大可解释性和可处理的复杂性。利用QD存档，该方法鼓励高表现解决方案之间的行为和结构多样性，而复杂性感知的适应度则促进紧凑表示的合成。进化系统支持使用精确约束来控制规则数量，并通过平衡透明度和表现力来提高系统的适应性。与三个基准任务CartPole-v1、MountainCar-v0和Acrobot-v1的实证比较表明，BASIL能够一致地生成与深度强化学习基线具有可比紧凑表示的可解释控制器。本文介绍了一种新的可解释策略合成方法，该方法结合了符号表达能力、进化多样性和在线学习，通过统一框架实现。 

---
# Sleep Brain and Cardiac Activity Predict Cognitive Flexibility and Conceptual Reasoning Using Deep Learning 

**Title (ZH)**: 睡眠脑活动和心脏活动预测认知灵活性和概念推理：基于深度学习的方法 

**Authors**: Boshra Khajehpiri, Eric Granger, Massimiliano de Zambotti, Fiona C. Baker, Mohamad Forouzanfar  

**Link**: [PDF](https://arxiv.org/pdf/2506.00279)  

**Abstract**: Despite extensive research on the relationship between sleep and cognition, the connection between sleep microstructure and human performance across specific cognitive domains remains underexplored. This study investigates whether deep learning models can predict executive functions, particularly cognitive adaptability and conceptual reasoning from physiological processes during a night's sleep. To address this, we introduce CogPSGFormer, a multi-scale convolutional-transformer model designed to process multi-modal polysomnographic data. This model integrates one-channel ECG and EEG signals along with extracted features, including EEG power bands and heart rate variability parameters, to capture complementary information across modalities. A thorough evaluation of the CogPSGFormer architecture was conducted to optimize the processing of extended sleep signals and identify the most effective configuration. The proposed framework was evaluated on 817 individuals from the STAGES dataset using cross-validation. The model achieved 80.3\% accuracy in classifying individuals into low vs. high cognitive performance groups on unseen data based on Penn Conditional Exclusion Test (PCET) scores. These findings highlight the effectiveness of our multi-scale feature extraction and multi-modal learning approach in leveraging sleep-derived signals for cognitive performance prediction. To facilitate reproducibility, our code is publicly accessible (this https URL). 

**Abstract (ZH)**: 尽管对睡眠与认知之间的关系进行了广泛研究，但睡眠微结构与人类特定认知领域的表现之间的联系仍鲜有探索。本研究旨在探究深度学习模型是否能够预测执行功能，特别是在生理过程夜间睡眠期间的认知适应性和概念推理。为此，我们引入了CogPSGFormer，这是一种多尺度卷积转换器模型，设计用于处理多模态多导睡眠图数据。该模型整合了一通道ECG和EEG信号以及提取特征，包括EEG功率带和心率变异性参数，以捕捉各模态之间的互补信息。我们对CogPSGFormer架构进行了全面评估，以优化扩展睡眠信号的处理并确定最有效配置。该提出的框架在STAGES数据集的817名个体上进行了交叉验证评估。模型在未见数据上基于Penn条件排除测试（PCET）得分成功地将个体分类为低认知表现组和高认知表现组，准确率为80.3%。这些发现强调了我们多尺度特征提取和多模态学习方法在利用睡眠衍生信号预测认知表现的有效性。为了便于再现性，我们的代码已公开可访问。 

---
# SMELLNET: A Large-scale Dataset for Real-world Smell Recognition 

**Title (ZH)**: SMELLNET: 一种大规模气味识别数据集 

**Authors**: Dewei Feng, Carol Li, Wei Dai, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00239)  

**Abstract**: The ability of AI to sense and identify various substances based on their smell alone can have profound impacts on allergen detection (e.g., smelling gluten or peanuts in a cake), monitoring the manufacturing process, and sensing hormones that indicate emotional states, stress levels, and diseases. Despite these broad impacts, there are virtually no large scale benchmarks, and therefore little progress, for training and evaluating AI systems' ability to smell in the real world. In this paper, we use portable gas and chemical sensors to create SmellNet, the first large-scale database that digitizes a diverse range of smells in the natural world. SmellNet contains about 180,000 time steps of 50 substances (spanning nuts, spices, herbs, fruits, and vegetables) with 50 hours of data. Using SmellNet, we train AI models for real-time classification of substances based on their smell alone. Our best methods leverage sequence models, contrastive learning to integrate high-resolution Gas Chromatography-Mass Spectrometry molecular data, and a new temporal difference method that identifies sharp changes in sensor readings. Our best models achieve up to 65.35% accuracy on pre-recorded data, and generalize to real-world conditions with 10.71% accuracy on nuts and 25.38% on spices in the challenging 50-way online classification task. Despite these promising results, SmellNet highlights many technical challenges in building AI for smell, including richer feature learning, on-edge smell models, and robustness to environmental changes. 

**Abstract (ZH)**: 基于气味识别的AI能力在过敏原检测、制造过程监控及情绪状态、压力水平和疾病感应方面的潜在影响：SmellNet——首个大规模自然气味数据库及其在实时物质分类中的应用 

---
# Ethical AI: Towards Defining a Collective Evaluation Framework 

**Title (ZH)**: 伦理人工智能：向构建集体评估框架迈进 

**Authors**: Aasish Kumar Sharma, Dimitar Kyosev, Julian Kunkel  

**Link**: [PDF](https://arxiv.org/pdf/2506.00233)  

**Abstract**: Artificial Intelligence (AI) is transforming sectors such as healthcare, finance, and autonomous systems, offering powerful tools for innovation. Yet its rapid integration raises urgent ethical concerns related to data ownership, privacy, and systemic bias. Issues like opaque decision-making, misleading outputs, and unfair treatment in high-stakes domains underscore the need for transparent and accountable AI systems. This article addresses these challenges by proposing a modular ethical assessment framework built on ontological blocks of meaning-discrete, interpretable units that encode ethical principles such as fairness, accountability, and ownership. By integrating these blocks with FAIR (Findable, Accessible, Interoperable, Reusable) principles, the framework supports scalable, transparent, and legally aligned ethical evaluations, including compliance with the EU AI Act. Using a real-world use case in AI-powered investor profiling, the paper demonstrates how the framework enables dynamic, behavior-informed risk classification. The findings suggest that ontological blocks offer a promising path toward explainable and auditable AI ethics, though challenges remain in automation and probabilistic reasoning. 

**Abstract (ZH)**: 人工智能（AI）正在改造医疗、金融和自主系统等领域，提供了强大的创新工具。然而，其迅速融入引发了关于数据所有权、隐私和系统偏见的紧迫伦理问题。在高风险领域中的不透明决策、误导性输出和不公平待遇强调了透明和问责制AI系统的必要性。本文通过提出一个基于意义离散的解释性单元构建的模块化伦理评估框架，解决了这些问题。该框架通过将这些单元与FAIR（可发现的、可访问的、可互操作的、可重用的）原则整合，支持可扩展、透明和法律合规的伦理评估，包括符合欧盟AI法案。通过一个基于AI的投资人画像实际案例，论文展示了该框架如何实现动态、基于行为的风险分类。研究结果表明，意义离散单元为可解释和可审计的AI伦理提供了有希望的途径，尽管在自动化和概率推理方面仍面临挑战。 

---
# What do professional software developers need to know to succeed in an age of Artificial Intelligence? 

**Title (ZH)**: 专业软件开发者在人工智能时代需要掌握哪些知识以取得成功？ 

**Authors**: Matthew Kam, Cody Miller, Miaoxin Wang, Abey Tidwell, Irene A. Lee, Joyce Malyn-Smith, Beatriz Perez, Vikram Tiwari, Joshua Kenitzer, Andrew Macvean, Erin Barrar  

**Link**: [PDF](https://arxiv.org/pdf/2506.00202)  

**Abstract**: Generative AI is showing early evidence of productivity gains for software developers, but concerns persist regarding workforce disruption and deskilling. We describe our research with 21 developers at the cutting edge of using AI, summarizing 12 of their work goals we uncovered, together with 75 associated tasks and the skills & knowledge for each, illustrating how developers use AI at work. From all of these, we distilled our findings in the form of 5 insights. We found that the skills & knowledge to be a successful AI-enhanced developer are organized into four domains (using Generative AI effectively, core software engineering, adjacent engineering, and adjacent non-engineering) deployed at critical junctures throughout a 6-step task workflow. In order to "future proof" developers for this age of AI, on-the-job learning initiatives and computer science degree programs will need to target both "soft" skills and the technical skills & knowledge in all four domains to reskill, upskill and safeguard against deskilling. 

**Abstract (ZH)**: 生成式AI为软件开发者带来了初步的生产力提升，但对劳动力市场冲击和技能退化仍存在担忧。我们描述了21名处于AI应用前沿的开发者的研究情况，总结了他们发现的12个工作目标及其相关75个任务和所需技能与知识，展示了开发者在工作中的AI应用方式。从这些发现中，我们提炼出5个洞察。我们发现，成功的AI增强型开发者的技能和知识围绕四个领域（有效使用生成式AI、核心软件工程、相邻工程和相邻非工程领域）组织，并在6步任务工作流的关键节点上体现。为了使开发者为AI时代做好准备，在职培训计划和计算机科学学位项目需要同时关注“软”技能和技术技能与知识在所有四个领域的提升，以实现再教育、提高技能并防范技能退化。 

---
# Control-R: Towards controllable test-time scaling 

**Title (ZH)**: Control-R: 向可控测试时缩放迈进 

**Authors**: Di Zhang, Weida Wang, Junxian Li, Xunzhi Wang, Jiatong Li, Jianbo Wu, Jingdi Lei, Haonan He, Peng Ye, Shufei Zhang, Wanli Ouyang, Yuqiang Li, Dongzhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.00189)  

**Abstract**: This paper target in addressing the challenges of underthinking and overthinking in long chain-of-thought (CoT) reasoning for Large Reasoning Models (LRMs) by introducing Reasoning Control Fields (RCF)--a novel test-time approach that injects structured control signals to guide reasoning from a tree search perspective. RCF enables models to adjust reasoning effort according to given control conditions when solving complex tasks. Additionally, we present the Control-R-4K dataset, which consists of challenging problems annotated with detailed reasoning processes and corresponding control fields. To further enhance reasoning control, we propose a Conditional Distillation Finetuning (CDF) method, which trains model--particularly Control-R-32B--to effectively adjust reasoning effort during test time. Experimental results on benchmarks such as AIME2024 and MATH500 demonstrate that our approach achieves state-of-the-art performance at the 32B scale while enabling a controllable Long CoT reasoning process (L-CoT). Overall, this work introduces an effective paradigm for controllable test-time scaling reasoning. 

**Abstract (ZH)**: 本文提出了一种通过引入推理控制域（RCF）来解决大型推理模型（LRMs）在长链推理（CoT）中过度推理和欠推理挑战的新颖测试时方法，RCF从树搜索的角度注入结构化的控制信号以指导推理。此外，我们介绍了Control-R-4K数据集，该数据集包含详细标注的推理过程和相应的控制域。为进一步增强推理控制，我们提出了条件蒸馏微调（CDF）方法，该方法训练模型（特别是Control-R-32B）在测试时有效调整推理努力。在AIME2024和MATH500等基准上的实验结果表明，我们的方法在32B规模上达到了最先进的性能，同时实现了可控的长链推理过程（L-CoT）。总体而言，本文引入了一种有效的可控制测试时扩展推理范式。 

---
# Utilizing AI for Aviation Post-Accident Analysis Classification 

**Title (ZH)**: 利用AI进行航空事故后分析分类 

**Authors**: Aziida Nanyonga, Graham Wild  

**Link**: [PDF](https://arxiv.org/pdf/2506.00169)  

**Abstract**: The volume of textual data available in aviation safety reports presents a challenge for timely and accurate analysis. This paper examines how Artificial Intelligence (AI) and, specifically, Natural Language Processing (NLP) can automate the process of extracting valuable insights from this data, ultimately enhancing aviation safety. The paper reviews ongoing efforts focused on the application of NLP and deep learning to aviation safety reports, with the goal of classifying the level of damage to an aircraft and identifying the phase of flight during which safety occurrences happen. Additionally, the paper explores the use of Topic Modeling (TM) to uncover latent thematic structures within aviation incident reports, aiming to identify recurring patterns and potential areas for safety improvement. The paper compares and contrasts the performance of various deep learning models and TM techniques applied to datasets from the National Transportation Safety Board (NTSB) and the Australian Transport Safety Bureau (ATSB), as well as the Aviation Safety Network (ASN), discussing the impact of dataset size and source on the accuracy of the analysis. The findings demonstrate that both NLP and deep learning, as well as TM, can significantly improve the efficiency and accuracy of aviation safety analysis, paving the way for more proactive safety management and risk mitigation strategies. 

**Abstract (ZH)**: 可用的航空安全报告中的文本数据量为及时准确分析带来了挑战。本文探讨了人工智能（AI）和具体来说是自然语言处理（NLP）如何自动化从这些数据中提取有价值洞察的过程，最终提升航空安全。本文回顾了将NLP和深度学习应用于航空安全报告的现有努力，旨在分类航空器损伤程度并识别安全事件发生于飞行的哪个阶段。此外，本文探索了主题建模（TM）在揭示航空事故报告中潜在主题结构方面的应用，旨在识别重复模式和潜在的安全改进领域。本文比较了各种深度学习模型和TM技术在国家运输安全委员会（NTSB）、澳大利亚运输安全局（ATSB）以及航空安全网（ASN）数据集上的性能，讨论了数据集大小和来源对分析准确性的影响。研究结果表明，NLP、深度学习以及TM都能显著提高航空安全分析的效率和准确性，为更积极的安全管理及风险缓解策略铺平道路。 

---
# Balancing Profit and Fairness in Risk-Based Pricing Markets 

**Title (ZH)**: 基于风险的价格市场中收益与公平性的平衡 

**Authors**: Jesse Thibodeau, Hadi Nekoei, Afaf Taïk, Janarthanan Rajendran, Golnoosh Farnadi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00140)  

**Abstract**: Dynamic, risk-based pricing can systematically exclude vulnerable consumer groups from essential resources such as health insurance and consumer credit. We show that a regulator can realign private incentives with social objectives through a learned, interpretable tax schedule. First, we provide a formal proposition that bounding each firm's \emph{local} demographic gap implicitly bounds the \emph{global} opt-out disparity, motivating firm-level penalties. Building on this insight we introduce \texttt{MarketSim} -- an open-source, scalable simulator of heterogeneous consumers and profit-maximizing firms -- and train a reinforcement learning (RL) social planner (SP) that selects a bracketed fairness-tax while remaining close to a simple linear prior via an $\mathcal{L}_1$ regularizer. The learned policy is thus both transparent and easily interpretable. In two empirically calibrated markets, i.e., U.S. health-insurance and consumer-credit, our planner simultaneously raises demand-fairness by up to $16\%$ relative to unregulated Free Market while outperforming a fixed linear schedule in terms of social welfare without explicit coordination. These results illustrate how AI-assisted regulation can convert a competitive social dilemma into a win-win equilibrium, providing a principled and practical framework for fairness-aware market oversight. 

**Abstract (ZH)**: 动态风险基价格可能系统性地将脆弱消费者群体排除在必要资源如健康保险和消费者信贷之外。我们展示了一名监管者可以通过学习和可解释的税率重新对齐私人激励与社会目标。首先，我们提供了正式命题，即限制每家公司的局部人口差距隐含地限制了全局退出不平等，从而激励公司层面的处罚。在此洞察基础上，我们引入了MarketSim——一个开源可扩展的异质消费者和利润最大化公司仿真器，并通过$\mathcal{L}_1$正则化器训练了一个强化学习社会规划者（SP），该规划者选择一个边界公平税，同时接近一个简单的线性先验。因此，学习到的策略是透明且易于解释的。在两个经验校准的市场，即美国健康保险和消费者信贷市场，我们的规划者同时将需求公平性提高最多16%，相对于未受监管的自由市场，同时在无需显式协调的情况下超越固定线性计划在社会福利方面表现更优。这些结果表明，AI辅助监管如何将竞争性的社会困境转化为双赢均衡，并提供了一个公平感知市场监督的原则性和实用性框架。 

---
# Toward Knowledge-Guided AI for Inverse Design in Manufacturing: A Perspective on Domain, Physics, and Human-AI Synergy 

**Title (ZH)**: 面向制造业逆向设计的知识引导AI：关于领域、物理和人机协同的视角 

**Authors**: Hugon Lee, Hyeonbin Moon, Junhyeong Lee, Seunghwa RYu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00056)  

**Abstract**: Artificial intelligence (AI) is reshaping inverse design across manufacturing domain, enabling high-performance discovery in materials, products, and processes. However, purely data-driven approaches often struggle in realistic settings characterized by sparse data, high-dimensional design spaces, and nontrivial physical constraints. This perspective argues for a new generation of design systems that transcend black-box modeling by integrating domain knowledge, physics-informed learning, and intuitive human-AI interfaces. We first demonstrate how expert-guided sampling strategies enhance data efficiency and model generalization. Next, we discuss how physics-informed machine learning enables physically consistent modeling in data-scarce regimes. Finally, we explore how large language models emerge as interactive design agents connecting user intent with simulation tools, optimization pipelines, and collaborative workflows. Through illustrative examples and conceptual frameworks, we advocate that inverse design in manufacturing should evolve into a unified ecosystem, where domain knowledge, physical priors, and adaptive reasoning collectively enable scalable, interpretable, and accessible AI-driven design systems. 

**Abstract (ZH)**: 人工智能（AI）正在重塑制造领域的逆向设计，使其能够在材料、产品和工艺中实现高性能发现。然而，在稀疏数据、高维设计空间和非平凡物理约束等现实场景中，纯粹的数据驱动方法往往难以应对。本文倡导一种超越黑盒建模的新一代设计系统，通过集成领域知识、物理知情学习以及直观的人机交互界面来促进设计。我们首先展示了专家指导的采样策略如何提升数据效率和模型泛化能力。接着，我们讨论了物理知情机器学习如何在数据稀缺的情况下实现物理一致性建模。最后，我们探讨了大型语言模型如何作为交互式设计代理，连接用户意图、仿真工具、优化管道和协作工作流程。通过示例和概念框架，我们主张逆向设计应进化为一个集成的生态系统，其中领域知识、物理先验和适应性推理共同推动可扩展、可解释和可访问的AI驱动设计系统的发展。 

---
# Red Teaming AI Policy: A Taxonomy of Avoision and the EU AI Act 

**Title (ZH)**: 红队评估AI政策：规避分类与欧盟AI法案 

**Authors**: Rui-Jie Yew, Bill Marino, Suresh Venkatasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2506.01931)  

**Abstract**: The shape of AI regulation is beginning to emerge, most prominently through the EU AI Act (the "AIA"). By 2027, the AIA will be in full effect, and firms are starting to adjust their behavior in light of this new law. In this paper, we present a framework and taxonomy for reasoning about "avoision" -- conduct that walks the line between legal avoidance and evasion -- that firms might engage in so as to minimize the regulatory burden the AIA poses. We organize these avoision strategies around three "tiers" of increasing AIA exposure that regulated entities face depending on: whether their activities are (1) within scope of the AIA, (2) exempted from provisions of the AIA, or are (3) placed in a category with higher regulatory scrutiny. In each of these tiers and for each strategy, we specify the organizational and technological forms through which avoision may manifest. Our goal is to provide an adversarial framework for "red teaming" the AIA and AI regulation on the horizon. 

**Abstract (ZH)**: AI法规的形态正逐渐成型：以欧盟AI法案（“AIA”）最为显著。到2027年，AIA将全面生效，企业已经开始调整行为以应对这一新法规。本文提出了一种框架和分类体系，用于分析企业可能会采取的“规避”行为——这类行为在合法规避与规避之间划界——以尽量减少AIA所带来的监管负担。我们将这些规避策略按企业面临的AIA监管暴露程度分为三个“层级”：其活动是否（1）在AIA监管范围内，（2）免于AIA的相关规定，或（3）处于受更高监管监督的类别。在每个层级和每个策略中，我们详细说明了规避可能通过的组织和技术形式。我们的目标是提供一种对手框架，用于对AIA和即将出台的AI法规进行“红队”测试。 

---
# TaxaDiffusion: Progressively Trained Diffusion Model for Fine-Grained Species Generation 

**Title (ZH)**: TaxaDiffusion：渐进训练的细粒度物种生成扩散模型 

**Authors**: Amin Karimi Monsefi, Mridul Khurana, Rajiv Ramnath, Anuj Karpatne, Wei-Lun Chao, Cheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01923)  

**Abstract**: We propose TaxaDiffusion, a taxonomy-informed training framework for diffusion models to generate fine-grained animal images with high morphological and identity accuracy. Unlike standard approaches that treat each species as an independent category, TaxaDiffusion incorporates domain knowledge that many species exhibit strong visual similarities, with distinctions often residing in subtle variations of shape, pattern, and color. To exploit these relationships, TaxaDiffusion progressively trains conditioned diffusion models across different taxonomic levels -- starting from broad classifications such as Class and Order, refining through Family and Genus, and ultimately distinguishing at the Species level. This hierarchical learning strategy first captures coarse-grained morphological traits shared by species with common ancestors, facilitating knowledge transfer before refining fine-grained differences for species-level distinction. As a result, TaxaDiffusion enables accurate generation even with limited training samples per species. Extensive experiments on three fine-grained animal datasets demonstrate that outperforms existing approaches, achieving superior fidelity in fine-grained animal image generation. Project page: this https URL 

**Abstract (ZH)**: TaxaDiffusion：一种基于分类学指导的扩散模型训练框架，用于生成高形态和身份准确度的精细粒度动物图像 

---
# Transformers as Multi-task Learners: Decoupling Features in Hidden Markov Models 

**Title (ZH)**: 基于Transformer的多任务学习者：解耦隐藏马尔可夫模型中的特征 

**Authors**: Yifan Hao, Chenlu Ye, Chi Han, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01919)  

**Abstract**: Transformer based models have shown remarkable capabilities in sequence learning across a wide range of tasks, often performing well on specific task by leveraging input-output examples. Despite their empirical success, a comprehensive theoretical understanding of this phenomenon remains limited. In this work, we investigate the layerwise behavior of Transformers to uncover the mechanisms underlying their multi-task generalization ability. Taking explorations on a typical sequence model, i.e, Hidden Markov Models, which are fundamental to many language tasks, we observe that: first, lower layers of Transformers focus on extracting feature representations, primarily influenced by neighboring tokens; second, on the upper layers, features become decoupled, exhibiting a high degree of time disentanglement. Building on these empirical insights, we provide theoretical analysis for the expressiveness power of Transformers. Our explicit constructions align closely with empirical observations, providing theoretical support for the Transformer's effectiveness and efficiency on sequence learning across diverse tasks. 

**Abstract (ZH)**: 基于Transformer的模型在广泛的任务中展示了在序列学习方面的卓越能力，常常通过输入-输出示例在特定任务上表现出色。尽管它们在实验上取得了成功，但对这一现象的全面理论理解仍然有限。本研究调查了Transformer的逐层行为，以揭示其实现多任务泛化能力的机制。通过探究典型的序列模型，即隐马尔可夫模型，该模型对于许多语言任务至关重要，我们发现：首先，Transformer的底层专注于提取特征表示，主要受相邻tokens的影响；其次，在高层，特征变得解耦，显示出高度的时间解纠缠。基于这些实验性洞察，我们提供了Transformer表达能力的理论分析。我们的显式构造与实验观察紧密一致，为Transformer在各种任务中进行序列学习的有效性和效率提供了理论支持。 

---
# CogniAlign: Word-Level Multimodal Speech Alignment with Gated Cross-Attention for Alzheimer's Detection 

**Title (ZH)**: CogniAlign: 基于门控交叉注意机制的词级多模态语音对齐方法及其在阿尔茨海默病检测中的应用 

**Authors**: David Ortiz-Perez, Manuel Benavent-Lledo, Javier Rodriguez-Juan, Jose Garcia-Rodriguez, David Tomás  

**Link**: [PDF](https://arxiv.org/pdf/2506.01890)  

**Abstract**: Early detection of cognitive disorders such as Alzheimer's disease is critical for enabling timely clinical intervention and improving patient outcomes. In this work, we introduce CogniAlign, a multimodal architecture for Alzheimer's detection that integrates audio and textual modalities, two non-intrusive sources of information that offer complementary insights into cognitive health. Unlike prior approaches that fuse modalities at a coarse level, CogniAlign leverages a word-level temporal alignment strategy that synchronizes audio embeddings with corresponding textual tokens based on transcription timestamps. This alignment supports the development of token-level fusion techniques, enabling more precise cross-modal interactions. To fully exploit this alignment, we propose a Gated Cross-Attention Fusion mechanism, where audio features attend over textual representations, guided by the superior unimodal performance of the text modality. In addition, we incorporate prosodic cues, specifically interword pauses, by inserting pause tokens into the text and generating audio embeddings for silent intervals, further enriching both streams. We evaluate CogniAlign on the ADReSSo dataset, where it achieves an accuracy of 90.36%, outperforming existing state-of-the-art methods. A detailed ablation study confirms the advantages of our alignment strategy, attention-based fusion, and prosodic modeling. 

**Abstract (ZH)**: 早期认知障碍如阿尔茨海默病的检测对于及时临床干预和改善患者预后至关重要。本文 introduces CogniAlign，一种结合音频和文本模态的多模态架构，用于阿尔茨海默病的检测，这两者是非侵入性的信息源，提供了认知健康互补的见解。不同于先前在粗略层面融合模态的方法，CogniAlign 利用基于转录时间戳的词级时间对齐策略，将音频嵌入与相应的文本令牌同步。这种对齐支持基于令牌的融合技术的发展，使得跨模态交互更加精确。为了充分利用这种对齐，我们提出了一种门控跨注意力融合机制，其中音频特征在文本表示的指导下关注文本。此外，通过插入暂停令牌并为静音间隔生成音频嵌入，我们还整合了语调线索。我们在 ADReSSo 数据集上评估了 CogniAlign，实现了 90.36% 的准确率，优于现有最先进的方法。详细的消融研究证实了我们对齐策略、基于注意力的融合和语调建模的优势。 

---
# scDataset: Scalable Data Loading for Deep Learning on Large-Scale Single-Cell Omics 

**Title (ZH)**: scDataset: 可扩展的数据加载方法，应用于大规模单细胞组学深度学习 

**Authors**: Davide D'Ascenzo, Sebastiano Cultrera di Montesano  

**Link**: [PDF](https://arxiv.org/pdf/2506.01883)  

**Abstract**: Modern single-cell datasets now comprise hundreds of millions of cells, presenting significant challenges for training deep learning models that require shuffled, memory-efficient data loading. While the AnnData format is the community standard for storing single-cell datasets, existing data loading solutions for AnnData are often inadequate: some require loading all data into memory, others convert to dense formats that increase storage demands, and many are hampered by slow random disk access. We present scDataset, a PyTorch IterableDataset that operates directly on one or more AnnData files without the need for format conversion. The core innovation is a combination of block sampling and batched fetching, which together balance randomness and I/O efficiency. On the Tahoe 100M dataset, scDataset achieves up to a 48$\times$ speed-up over AnnLoader, a 27$\times$ speed-up over HuggingFace Datasets, and an 18$\times$ speed-up over BioNeMo in single-core settings. These advances democratize large-scale single-cell model training for the broader research community. 

**Abstract (ZH)**: 基于AnnData的块采样批加载方案scDataset及其在大规模单细胞模型训练中的应用 

---
# Learning to Explore: An In-Context Learning Approach for Pure Exploration 

**Title (ZH)**: 学习探索：一种纯探索的上下文学习方法 

**Authors**: Alessio Russo, Ryan Welch, Aldo Pacchiano  

**Link**: [PDF](https://arxiv.org/pdf/2506.01876)  

**Abstract**: In this work, we study the active sequential hypothesis testing problem, also known as pure exploration, where the goal is to actively control a data collection process to efficiently identify the correct hypothesis underlying a decision problem. While relevant across multiple domains, devising adaptive exploration strategies remains challenging, particularly due to difficulties in encoding appropriate inductive biases. Existing Reinforcement Learning (RL)-based methods often underperform when relevant information structures are inadequately represented, whereas more complex methods, like Best Arm Identification (BAI) techniques, may be difficult to devise and typically rely on explicit modeling assumptions. To address these limitations, we introduce In-Context Pure Exploration (ICPE), an in-context learning approach that uses Transformers to learn exploration strategies directly from experience. ICPE combines supervised learning and reinforcement learning to identify and exploit latent structure across related tasks, without requiring prior assumptions. Numerical results across diverse synthetic and semi-synthetic benchmarks highlight ICPE's capability to achieve robust performance performance in deterministic, stochastic, and structured settings. These results demonstrate ICPE's ability to match optimal instance-dependent algorithms using only deep learning techniques, making it a practical and general approach to data-efficient exploration. 

**Abstract (ZH)**: 基于上下文的纯探索研究（ICPE）：一种结合变换器的探索策略学习方法 

---
# Frugal Machine Learning for Energy-efficient, and Resource-aware Artificial Intelligence 

**Title (ZH)**: 经济高效且资源意识强的机器学习方法 

**Authors**: John Violos, Konstantina-Christina Diamanti, Ioannis Kompatsiaris, Symeon Papadopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2506.01869)  

**Abstract**: Frugal Machine Learning (FML) refers to the practice of designing Machine Learning (ML) models that are efficient, cost-effective, and mindful of resource constraints. This field aims to achieve acceptable performance while minimizing the use of computational resources, time, energy, and data for both training and inference. FML strategies can be broadly categorized into input frugality, learning process frugality, and model frugality, each focusing on reducing resource consumption at different stages of the ML pipeline. This chapter explores recent advancements, applications, and open challenges in FML, emphasizing its importance for smart environments that incorporate edge computing and IoT devices, which often face strict limitations in bandwidth, energy, or latency. Technological enablers such as model compression, energy-efficient hardware, and data-efficient learning techniques are discussed, along with adaptive methods including parameter regularization, knowledge distillation, and dynamic architecture design that enable incremental model updates without full retraining. Furthermore, it provides a comprehensive taxonomy of frugal methods, discusses case studies across diverse domains, and identifies future research directions to drive innovation in this evolving field. 

**Abstract (ZH)**: 节俭机器学习（FML）指的是设计高效、成本效益高并在资源受限情况下保持意识的机器学习模型的做法。该领域旨在在最小化计算资源、时间和能量的使用（包括训练和推理）的前提下，实现可接受的性能。FML策略可以根据在机器学习管道的不同阶段减少资源消耗，大致分为输入节俭、学习过程节俭和模型节俭。本章探讨了FML领域的最新进展、应用和开放挑战，突出了其对于包括边缘计算和物联网设备在内的智能环境的重要性，这些设备通常在带宽、能源或延迟方面受到严格限制。讨论了模型压缩、能源高效硬件、数据高效学习技术等技术使能器，以及包括参数正则化、知识蒸馏和动态架构设计在内的自适应方法，这些方法能够在无需完全重新训练的情况下实现增量模型更新。此外，提供了节俭方法的综合分类，讨论了跨不同领域的案例研究，并指出了未来的研究方向，以推动这一不断发展的领域的创新。 

---
# CiteEval: Principle-Driven Citation Evaluation for Source Attribution 

**Title (ZH)**: CiteEval: 原则驱动的引用评价及其来源归属 

**Authors**: Yumo Xu, Peng Qi, Jifan Chen, Kunlun Liu, Rujun Han, Lan Liu, Bonan Min, Vittorio Castelli, Arshit Gupta, Zhiguo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01829)  

**Abstract**: Citation quality is crucial in information-seeking systems, directly influencing trust and the effectiveness of information access. Current evaluation frameworks, both human and automatic, mainly rely on Natural Language Inference (NLI) to assess binary or ternary supportiveness from cited sources, which we argue is a suboptimal proxy for citation evaluation. In this work we introduce CiteEval, a citation evaluation framework driven by principles focusing on fine-grained citation assessment within a broad context, encompassing not only the cited sources but the full retrieval context, user query, and generated text. Guided by the proposed framework, we construct CiteBench, a multi-domain benchmark with high-quality human annotations on citation quality. To enable efficient evaluation, we further develop CiteEval-Auto, a suite of model-based metrics that exhibit strong correlation with human judgments. Experiments across diverse systems demonstrate CiteEval-Auto's superior ability to capture the multifaceted nature of citations compared to existing metrics, offering a principled and scalable approach to evaluate and improve model-generated citations. 

**Abstract (ZH)**: 引文质量对于信息检索系统至关重要，直接影响信息访问的可信度和有效性。当前的评估框架，无论是人工的还是自动的，主要依赖于自然语言推理(NLI)来评估引文支持性，我们认为这只是一个次优的引文评估代理。在此项工作中，我们引入了CiteEval，这是一种以细粒度引文评估为核心原则的引文评估框架，不仅涵盖引文来源，还涉及全面的检索上下文、用户查询和生成文本。基于提出的框架，我们构建了CiteBench，这是一个多领域基准，包含了高质量的人工标注的引文质量。为实现高效评估，我们进一步开发了CiteEval-Auto，这是一个基于模型的度量套件，与人工判断具有较强的关联性。实验表明，CiteEval-Auto在捕捉引文的复杂性方面优于现有度量，提供了一种有原则且可扩展的方法来评估和改进模型生成的引文。 

---
# A Quantum Information Theoretic Approach to Tractable Probabilistic Models 

**Title (ZH)**: 基于可处理概率模型的量子信息理论方法 

**Authors**: Pedro Zuidberg Dos Martires  

**Link**: [PDF](https://arxiv.org/pdf/2506.01824)  

**Abstract**: By recursively nesting sums and products, probabilistic circuits have emerged in recent years as an attractive class of generative models as they enjoy, for instance, polytime marginalization of random variables. In this work we study these machine learning models using the framework of quantum information theory, leading to the introduction of positive unital circuits (PUnCs), which generalize circuit evaluations over positive real-valued probabilities to circuit evaluations over positive semi-definite matrices. As a consequence, PUnCs strictly generalize probabilistic circuits as well as recently introduced circuit classes such as PSD circuits. 

**Abstract (ZH)**: 通过递归嵌套和的概率电路最近成为了生成模型的一种有吸引力的类别，因为它们可以实现多项式时间的概率变量边缘化。在本工作中，我们利用量子信息理论的框架研究这些机器学习模型，从而引出了正单位电路（PUnC），该电路将电路评估从正实数概率推广到正半定矩阵。作为结果，PUnC 严格地推广了概率电路以及最近引入的如PSD电路等电路类别。 

---
# Datasheets Aren't Enough: DataRubrics for Automated Quality Metrics and Accountability 

**Title (ZH)**: 数据表单不足以涵盖全部：数据评 rubrics 用于自动化质量指标和问责制 

**Authors**: Genta Indra Winata, David Anugraha, Emmy Liu, Alham Fikri Aji, Shou-Yi Hung, Aditya Parashar, Patrick Amadeus Irawan, Ruochen Zhang, Zheng-Xin Yong, Jan Christian Blaise Cruz, Niklas Muennighoff, Seungone Kim, Hanyang Zhao, Sudipta Kar, Kezia Erina Suryoraharjo, M. Farid Adilazuarda, En-Shiun Annie Lee, Ayu Purwarianti, Derry Tanti Wijaya, Monojit Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2506.01789)  

**Abstract**: High-quality datasets are fundamental to training and evaluating machine learning models, yet their creation-especially with accurate human annotations-remains a significant challenge. Many dataset paper submissions lack originality, diversity, or rigorous quality control, and these shortcomings are often overlooked during peer review. Submissions also frequently omit essential details about dataset construction and properties. While existing tools such as datasheets aim to promote transparency, they are largely descriptive and do not provide standardized, measurable methods for evaluating data quality. Similarly, metadata requirements at conferences promote accountability but are inconsistently enforced. To address these limitations, this position paper advocates for the integration of systematic, rubric-based evaluation metrics into the dataset review process-particularly as submission volumes continue to grow. We also explore scalable, cost-effective methods for synthetic data generation, including dedicated tools and LLM-as-a-judge approaches, to support more efficient evaluation. As a call to action, we introduce DataRubrics, a structured framework for assessing the quality of both human- and model-generated datasets. Leveraging recent advances in LLM-based evaluation, DataRubrics offers a reproducible, scalable, and actionable solution for dataset quality assessment, enabling both authors and reviewers to uphold higher standards in data-centric research. We also release code to support reproducibility of LLM-based evaluations at this https URL. 

**Abstract (ZH)**: 高质量的数据集是训练和评估机器学习模型的基础，但其创建尤其是精准的人工注解创建仍是一项重大挑战。许多数据集论文提交缺乏原创性、多样性和严格的质量控制，这些不足在同行评审过程中往往被忽略。提交内容还经常省略关于数据集构建和属性的关键细节。虽然现有的工具如数据表旨在促进透明度，它们主要是描述性的，并不能提供标准化、可衡量的方法来评估数据质量。同样，会议中的元数据要求促进问责制，但执行上却不够一致。为解决这些局限性，本文倡导在数据集评审过程中整合基于规范和量表的评估指标，特别是在提交量继续增长的情况下。我们还探讨了生成合成数据的可扩展、低成本方法，包括专用工具和LLM-as-a-judge方法，以支持更高效的评估。作为行动呼吁，我们引入了DataRubrics，这是一种结构化的框架，用于评估人类生成和模型生成的数据集质量。利用基于LLM的评估的最新进展，DataRubrics提供了一种可重复、可扩展且可操作的数据集质量评估解决方案，使作者和评审者能够提高数据为中心的研究标准。我们还发布了代码以支持LLM基于的评估的可重复性，详见此链接：https://yourlinkhere。 

---
# Systematic Hazard Analysis for Frontier AI using STPA 

**Title (ZH)**: 面向前沿人工智能的STPA系统性危害分析 

**Authors**: Simon Mylius  

**Link**: [PDF](https://arxiv.org/pdf/2506.01782)  

**Abstract**: All of the frontier AI companies have published safety frameworks where they define capability thresholds and risk mitigations that determine how they will safely develop and deploy their models. Adoption of systematic approaches to risk modelling, based on established practices used in safety-critical industries, has been recommended, however frontier AI companies currently do not describe in detail any structured approach to identifying and analysing hazards. STPA (Systems-Theoretic Process Analysis) is a systematic methodology for identifying how complex systems can become unsafe, leading to hazards. It achieves this by mapping out controllers and controlled processes then analysing their interactions and feedback loops to understand how harmful outcomes could occur (Leveson & Thomas, 2018). We evaluate STPA's ability to broaden the scope, improve traceability and strengthen the robustness of safety assurance for frontier AI systems. Applying STPA to the threat model and scenario described in 'A Sketch of an AI Control Safety Case' (Korbak et al., 2025), we derive a list of Unsafe Control Actions. From these we select a subset and explore the Loss Scenarios that lead to them if left unmitigated. We find that STPA is able to identify causal factors that may be missed by unstructured hazard analysis methodologies thereby improving robustness. We suggest STPA could increase the safety assurance of frontier AI when used to complement or check coverage of existing AI governance techniques including capability thresholds, model evaluations and emergency procedures. The application of a systematic methodology supports scalability by increasing the proportion of the analysis that could be conducted by LLMs, reducing the burden on human domain experts. 

**Abstract (ZH)**: 前沿AI公司的安全性框架及其基于STPA方法的安全性保障改进研究 

---
# Enhancing Customer Service Chatbots with Context-Aware NLU through Selective Attention and Multi-task Learning 

**Title (ZH)**: 基于选择性注意和多任务学习的情境感知NLU增强客户服务聊天机器人 

**Authors**: Subhadip Nandi, Neeraj Agrawal, Anshika Singh, Priyanka Bhatt  

**Link**: [PDF](https://arxiv.org/pdf/2506.01781)  

**Abstract**: Customer service chatbots are conversational systems aimed at addressing customer queries, often by directing them to automated workflows. A crucial aspect of this process is the classification of the customer's intent. Presently, most intent classification models for customer care utilise only customer query for intent prediction. This may result in low-accuracy models, which cannot handle ambiguous queries. An ambiguous query like "I didn't receive my package" could indicate a delayed order, or an order that was delivered but the customer failed to receive it. Resolution of each of these scenarios requires the execution of very different sequence of steps. Utilizing additional information, such as the customer's order delivery status, in the right manner can help identify the intent for such ambiguous queries. In this paper, we have introduced a context-aware NLU model that incorporates both, the customer query and contextual information from the customer's order status for predicting customer intent. A novel selective attention module is used to extract relevant context features. We have also proposed a multi-task learning paradigm for the effective utilization of different label types available in our training data. Our suggested method, Multi-Task Learning Contextual NLU with Selective Attention Weighted Context (MTL-CNLU-SAWC), yields a 4.8% increase in top 2 accuracy score over the baseline model which only uses user queries, and a 3.5% improvement over existing state-of-the-art models that combine query and context. We have deployed our model to production for Walmart's customer care domain. Accurate intent prediction through MTL-CNLU-SAWC helps to better direct customers to automated workflows, thereby significantly reducing escalations to human agents, leading to almost a million dollars in yearly savings for the company. 

**Abstract (ZH)**: 基于选择性注意力加权上下文的多任务学习上下文NLU模型（MTL-CNLU-SAWC）及其在客户服务聊天机器人中的应用 

---
# Greening AI-enabled Systems with Software Engineering: A Research Agenda for Environmentally Sustainable AI Practices 

**Title (ZH)**: 利用软件工程实现AI赋能系统的绿色化：面向环境可持续的AI实践研究议程 

**Authors**: Luís Cruz, João Paulo Fernandes, Maja H. Kirkeby, Silverio Martínez-Fernández, June Sallou, Hina Anwar, Enrique Barba Roque, Justus Bogner, Joel Castaño, Fernando Castor, Aadil Chasmawala, Simão Cunha, Daniel Feitosa, Alexandra González, Andreas Jedlitschka, Patricia Lago, Ana Oprescu, Pooja Rani, João Saraiva, Federica Sarro, Raghavendra Selvan, Karthik Vaidhyanathan, Roberto Verdecchia, Ivan P. Yamshchikov, Henry Muccini  

**Link**: [PDF](https://arxiv.org/pdf/2506.01774)  

**Abstract**: The environmental impact of Artificial Intelligence (AI)-enabled systems is increasing rapidly, and software engineering plays a critical role in developing sustainable solutions. The "Greening AI with Software Engineering" CECAM-Lorentz workshop (no. 1358, 2025) funded by the Centre Européen de Calcul Atomique et Moléculaire and the Lorentz Center, provided an interdisciplinary forum for 29 participants, from practitioners to academics, to share knowledge, ideas, practices, and current results dedicated to advancing green software and AI research. The workshop was held February 3-7, 2025, in Lausanne, Switzerland. Through keynotes, flash talks, and collaborative discussions, participants identified and prioritized key challenges for the field. These included energy assessment and standardization, benchmarking practices, sustainability-aware architectures, runtime adaptation, empirical methodologies, and education. This report presents a research agenda emerging from the workshop, outlining open research directions and practical recommendations to guide the development of environmentally sustainable AI-enabled systems rooted in software engineering principles. 

**Abstract (ZH)**: 人工智能(AI)使能系统对环境的影响正在迅速增加，软件工程在开发可持续解决方案中发挥了关键作用。由欧洲原子分子计算中心和洛伦兹中心资助的“通过软件工程使AI绿化”CECAM-洛伦兹研讨会（编号1358，2025）为来自从业者到学术界共计29位参与者提供了一个跨学科论坛，分享有关推进绿色软件和AI研究的知识、理念、实践和当前成果。该研讨会于2025年2月3日至7日在瑞士洛桑举行。通过主题演讲、快速发言和协作讨论，参与者识别和优先考虑了该领域的关键挑战，包括能源评估和标准化、基准测试实践、环境意识架构、运行时适应、实证方法论以及教育。本报告概述了从此次研讨会中 emergence 的研究议程，明确了开放的研究方向和实用建议，以指导基于软件工程原则的环境可持续AI使能系统的开发。 

---
# Principled data augmentation for learning to solve quadratic programming problems 

**Title (ZH)**: 原理性的数据增强方法用于求解二次规划问题 

**Authors**: Chendi Qian, Christopher Morris  

**Link**: [PDF](https://arxiv.org/pdf/2506.01728)  

**Abstract**: Linear and quadratic optimization are crucial in numerous real-world applications, from training machine learning models to integer-linear optimization. Recently, learning-to-optimize methods (L2O) for linear (LPs) or quadratic programs (QPs) using message-passing graph neural networks (MPNNs) have gained traction, promising lightweight, data-driven proxies for solving such optimization problems. For example, they replace the costly computation of strong branching scores in branch-and-bound solvers, requiring solving many such optimization problems. However, robust L2O MPNNs remain challenging in data-scarce settings, especially when addressing complex optimization problems such as QPs. This work introduces a principled approach to data augmentation tailored for QPs via MPNNs. Our method leverages theoretically justified data augmentation techniques to generate diverse yet optimality-preserving instances. Furthermore, we integrate these augmentations into a self-supervised learning framework based on contrastive learning, thereby pretraining MPNNs for enhanced performance on L2O tasks. Extensive experiments demonstrate that our approach improves generalization in supervised scenarios and facilitates effective transfer learning to related optimization problems. 

**Abstract (ZH)**: 基于图神经网络的消息传递学习-to-优化方法的数据增强：用于二次规划的原理性方法 

---
# Data Pruning by Information Maximization 

**Title (ZH)**: 信息最大化驱动的数据剪枝 

**Authors**: Haoru Tan, Sitong Wu, Wei Huang, Shizhen Zhao, Xiaojuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2506.01701)  

**Abstract**: In this paper, we present InfoMax, a novel data pruning method, also known as coreset selection, designed to maximize the information content of selected samples while minimizing redundancy. By doing so, InfoMax enhances the overall informativeness of the coreset. The information of individual samples is measured by importance scores, which capture their influence or difficulty in model learning. To quantify redundancy, we use pairwise sample similarities, based on the premise that similar samples contribute similarly to the learning process. We formalize the coreset selection problem as a discrete quadratic programming (DQP) task, with the objective of maximizing the total information content, represented as the sum of individual sample contributions minus the redundancies introduced by similar samples within the coreset. To ensure practical scalability, we introduce an efficient gradient-based solver, complemented by sparsification techniques applied to the similarity matrix and dataset partitioning strategies. This enables InfoMax to seamlessly scale to datasets with millions of samples. Extensive experiments demonstrate the superior performance of InfoMax in various data pruning tasks, including image classification, vision-language pre-training, and instruction tuning for large language models. 

**Abstract (ZH)**: 本文提出了InfoMax，一种新型的数据剪裁方法，也称为核心集选择，旨在最大化所选样本的信息含量的同时最小化冗余。通过这种方式，InfoMax 提高了核心集的整体信息量。个体样本的信息量通过重要性得分来衡量，这些得分捕捉了其对模型学习的影响或难度。为了量化冗余，我们使用基于相似样本在学习过程中贡献相似性的样本对相似性。我们将核心集选择问题形式化为一个离散二次规划（DQP）问题，目标是最化总信息含量，即个体样本贡献的总和减去核心集内相似样本引入的冗余。为了确保实际可扩展性，我们引入了一种高效的梯度基解算器，并结合了相似性矩阵的稀疏化技术和数据集分区策略，这使InfoMax能够无缝扩展到包含数百万样本的数据集。广泛的经验表明，InfoMax 在各种数据剪裁任务中表现出色，包括图像分类、视觉-语言预训练以及大型语言模型的指令调优。 

---
# GRAM: Generative Recommendation via Semantic-aware Multi-granular Late Fusion 

**Title (ZH)**: Gram: 基于语义aware多粒度晚融合的生成推荐 

**Authors**: Sunkyung Lee, Minjin Choi, Eunseong Choi, Hye-young Kim, Jongwuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.01673)  

**Abstract**: Generative recommendation is an emerging paradigm that leverages the extensive knowledge of large language models by formulating recommendations into a text-to-text generation task. However, existing studies face two key limitations in (i) incorporating implicit item relationships and (ii) utilizing rich yet lengthy item information. To address these challenges, we propose a Generative Recommender via semantic-Aware Multi-granular late fusion (GRAM), introducing two synergistic innovations. First, we design semantic-to-lexical translation to encode implicit hierarchical and collaborative item relationships into the vocabulary space of LLMs. Second, we present multi-granular late fusion to integrate rich semantics efficiently with minimal information loss. It employs separate encoders for multi-granular prompts, delaying the fusion until the decoding stage. Experiments on four benchmark datasets show that GRAM outperforms eight state-of-the-art generative recommendation models, achieving significant improvements of 11.5-16.0% in Recall@5 and 5.3-13.6% in NDCG@5. The source code is available at this https URL. 

**Abstract (ZH)**: 基于语义意识多层次晚期融合的生成推荐 

---
# Explainable AI Systems Must Be Contestable: Here's How to Make It Happen 

**Title (ZH)**: 可解释的人工智能系统必须是可争议的：实现这一目标的方法 

**Authors**: Catarina Moreira, Anna Palatkina, Dacia Braca, Dylan M. Walsh, Peter J. Leihn, Fang Chen, Nina C. Hubig  

**Link**: [PDF](https://arxiv.org/pdf/2506.01662)  

**Abstract**: As AI regulations around the world intensify their focus on system safety, contestability has become a mandatory, yet ill-defined, safeguard. In XAI, "contestability" remains an empty promise: no formal definition exists, no algorithm guarantees it, and practitioners lack concrete guidance to satisfy regulatory requirements. Grounded in a systematic literature review, this paper presents the first rigorous formal definition of contestability in explainable AI, directly aligned with stakeholder requirements and regulatory mandates. We introduce a modular framework of by-design and post-hoc mechanisms spanning human-centered interfaces, technical architectures, legal processes, and organizational workflows. To operationalize our framework, we propose the Contestability Assessment Scale, a composite metric built on more than twenty quantitative criteria. Through multiple case studies across diverse application domains, we reveal where state-of-the-art systems fall short and show how our framework drives targeted improvements. By converting contestability from regulatory theory into a practical framework, our work equips practitioners with the tools to embed genuine recourse and accountability into AI systems. 

**Abstract (ZH)**: 随着全球对AI系统安全性的关注加剧，“可争议性”已成为一项必要的但尚未明确的规定。在XAI领域，“可争议性”仍是一个空洞的承诺：缺乏正式定义，没有算法能够保证这一点，实践者也缺乏具体的指导以满足监管要求。基于系统的文献综述，本文首次提出了 Explainable AI 中“可争议性”的严谨正式定义，该定义直接与利益相关方需求和监管要求相一致。我们引入了一个模块化框架，涵盖了人机接口、技术架构、法律流程和组织工作流程中的设计时和事后机制。为了实现该框架的实用化，我们提出了“可争议性评估量表”，这是一个基于逾二十个定量标准的综合指标。通过涵盖不同应用领域的多个案例研究，我们揭示了现有先进系统的不足之处，并展示了如何通过该框架实现有针对性的改进。通过将“可争议性”从监管理论转化为实用框架，我们的工作为实践者提供了嵌入真正救济和问责制的工具。 

---
# Engram Memory Encoding and Retrieval: A Neurocomputational Perspective 

**Title (ZH)**: 记忆回放的编码与检索：神经计算视角 

**Authors**: Daniel Szelogowski  

**Link**: [PDF](https://arxiv.org/pdf/2506.01659)  

**Abstract**: Despite substantial research into the biological basis of memory, the precise mechanisms by which experiences are encoded, stored, and retrieved in the brain remain incompletely understood. A growing body of evidence supports the engram theory, which posits that sparse populations of neurons undergo lasting physical and biochemical changes to support long-term memory. Yet, a comprehensive computational framework that integrates biological findings with mechanistic models remains elusive. This work synthesizes insights from cellular neuroscience and computational modeling to address key challenges in engram research: how engram neurons are identified and manipulated; how synaptic plasticity mechanisms contribute to stable memory traces; and how sparsity promotes efficient, interference-resistant representations. Relevant computational approaches -- such as sparse regularization, engram gating, and biologically inspired architectures like Sparse Distributed Memory and spiking neural networks -- are also examined. Together, these findings suggest that memory efficiency, capacity, and stability emerge from the interaction of plasticity and sparsity constraints. By integrating neurobiological and computational perspectives, this paper provides a comprehensive theoretical foundation for engram research and proposes a roadmap for future inquiry into the mechanisms underlying memory, with implications for the diagnosis and treatment of memory-related disorders. 

**Abstract (ZH)**: 尽管对记忆的生物学基础进行了大量研究，但大脑中经验是如何被编码、存储和检索的精确机制仍然知之甚少。越来越多的证据支持基因组理论，即稀疏的神经元群体经历持久的物理和生化变化以支持长期记忆。然而，将生物学发现与机制模型综合的全面计算框架仍然难以捉摸。本文综合细胞神经科学和计算建模的见解，解决了基因组研究中的关键挑战：如何识别和操纵基因组神经元；突触可塑性机制如何贡献稳定的记忆痕迹；以及稀疏性如何促进高效、抗干扰的表现。相关的计算方法，如稀疏正则化、基因组门控、以及受生物启发的结构如稀疏分布式记忆和突触神经网络，也进行了探讨。这些发现表明，记忆效率、容量和稳定性源自可塑性和稀疏性约束的相互作用。通过整合神经生物学和计算视角，本文为基因组研究提供了全面的理论基础，并提出了对未来研究记忆机制的路线图，对于记忆相关疾病的诊断和治疗具有重要意义。 

---
# Bidirectional Soft Actor-Critic: Leveraging Forward and Reverse KL Divergence for Efficient Reinforcement Learning 

**Title (ZH)**: 双向软actor- critic：利用前向和后向KL散度进行高效的强化学习 

**Authors**: Yixian Zhang, Huaze Tang, Changxu Wei, Wenbo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2506.01639)  

**Abstract**: The Soft Actor-Critic (SAC) algorithm, a state-of-the-art method in maximum entropy reinforcement learning, traditionally relies on minimizing reverse Kullback-Leibler (KL) divergence for policy updates. However, this approach leads to an intractable optimal projection policy, necessitating gradient-based approximations that can suffer from instability and poor sample efficiency. This paper investigates the alternative use of forward KL divergence within SAC. We demonstrate that for Gaussian policies, forward KL divergence yields an explicit optimal projection policy -- corresponding to the mean and variance of the target Boltzmann distribution's action marginals. Building on the distinct advantages of both KL directions, we propose Bidirectional SAC, an algorithm that first initializes the policy using the explicit forward KL projection and then refines it by optimizing the reverse KL divergence. Comprehensive experiments on continuous control benchmarks show that Bidirectional SAC significantly outperforms standard SAC and other baselines, achieving up to a $30\%$ increase in episodic rewards, alongside enhanced sample efficiency. 

**Abstract (ZH)**: 双向软 actor-critic (Bidirectional SAC): 结合前后向 KL 散度的优势 

---
# Robust Satisficing Gaussian Process Bandits Under Adversarial Attacks 

**Title (ZH)**: 鲁棒 satisficing 高斯过程多臂赌博机算法在对抗攻击下的研究 

**Authors**: Artun Saday, Yaşar Cahit Yıldırım, Cem Tekin  

**Link**: [PDF](https://arxiv.org/pdf/2506.01625)  

**Abstract**: We address the problem of Gaussian Process (GP) optimization in the presence of unknown and potentially varying adversarial perturbations. Unlike traditional robust optimization approaches that focus on maximizing performance under worst-case scenarios, we consider a robust satisficing objective, where the goal is to consistently achieve a predefined performance threshold $\tau$, even under adversarial conditions. We propose two novel algorithms based on distinct formulations of robust satisficing, and show that they are instances of a general robust satisficing framework. Further, each algorithm offers different guarantees depending on the nature of the adversary. Specifically, we derive two regret bounds: one that is sublinear over time, assuming certain conditions on the adversary and the satisficing threshold $\tau$, and another that scales with the perturbation magnitude but requires no assumptions on the adversary. Through extensive experiments, we demonstrate that our approach outperforms the established robust optimization methods in achieving the satisficing objective, particularly when the ambiguity set of the robust optimization framework is inaccurately specified. 

**Abstract (ZH)**: 我们在未知且可能变化的对抗扰动下解决高斯过程（GP）优化问题。不同于传统的稳健优化方法侧重于在最坏情况下最大化性能，我们考虑一种稳健满足目标，即在对抗条件下一致地达到预定义的性能门槛 $\tau$。我们提出两种新型算法，基于不同的稳健满足形式，并表明它们是通用的稳健满足框架的实例。此外，每种算法根据对手的性质提供不同的保证。具体地，我们推导出两个遗憾界：一个在满足一定对手和满足阈值 $\tau$ 的条件下是亚线性的，另一个与扰动幅度成比例但对对手没有假设。通过广泛的实验，我们证明我们的方法在实现满足目标方面优于现有的稳健优化方法，特别是在稳健优化框架的不确定性集不准确指定时。 

---
# Unsupervised Rhythm and Voice Conversion to Improve ASR on Dysarthric Speech 

**Title (ZH)**: 无监督节奏和语音转换以改善构音障碍语音的ASR性能 

**Authors**: Karl El Hajal, Enno Hermann, Sevada Hovsepyan, Mathew Magimai.-Doss  

**Link**: [PDF](https://arxiv.org/pdf/2506.01618)  

**Abstract**: Automatic speech recognition (ASR) systems struggle with dysarthric speech due to high inter-speaker variability and slow speaking rates. To address this, we explore dysarthric-to-healthy speech conversion for improved ASR performance. Our approach extends the Rhythm and Voice (RnV) conversion framework by introducing a syllable-based rhythm modeling method suited for dysarthric speech. We assess its impact on ASR by training LF-MMI models and fine-tuning Whisper on converted speech. Experiments on the Torgo corpus reveal that LF-MMI achieves significant word error rate reductions, especially for more severe cases of dysarthria, while fine-tuning Whisper on converted data has minimal effect on its performance. These results highlight the potential of unsupervised rhythm and voice conversion for dysarthric ASR. Code available at: this https URL 

**Abstract (ZH)**: 自动语音识别(ASR)系统在处理构音障碍 speech 的时候因说话人变异性高和说话速度慢而遇到困难。为了解决这个问题，我们探索了构音障碍到健康语音的转换，以提高ASR性能。我们的方法扩展了节奏和语音(RnV)转换框架，引入了一种基于音节的节奏建模方法，适用于构音障碍语音。我们通过训练LF-MMI模型并在转换后的语音上微调Whisper来评估其对ASR的影响。对Torgo语料库的实验表明，LF-MMI在词错误率方面取得了显著减少，尤其是在构音障碍更严重的情况下，而对转换数据进行Whisper的微调对其性能的影响很小。这些结果突显了无监督节奏和语音转换对构音障碍ASR的潜在价值。代码可在以下链接获得：这个 https URL。 

---
# Contrastive Learning for Efficient Transaction Validation in UTXO-based Blockchains 

**Title (ZH)**: 基于UTXO模型区块链中高效交易验证的对比学习 

**Authors**: Hamid Attar, Luigi Lunardon, Alessio Pagani  

**Link**: [PDF](https://arxiv.org/pdf/2506.01614)  

**Abstract**: This paper introduces a Machine Learning (ML) approach for scalability of UTXO-based blockchains, such as Bitcoin. Prior approaches to UTXO set sharding struggle with distributing UTXOs effectively across validators, creating substantial communication overhead due to child-parent transaction dependencies. This overhead, which arises from the need to locate parent UTXOs, significantly hampers transaction processing speeds. Our solution uses ML to optimize not only UTXO set sharding but also the routing of incoming transactions, ensuring that transactions are directed to shards containing their parent UTXOs. At the heart of our approach is a framework that combines contrastive and unsupervised learning to create an embedding space for transaction outputs. This embedding allows the model to group transaction outputs based on spending relationships, making it possible to route transactions efficiently to the correct validation microservices. Trained on historical transaction data with triplet loss and online semi-hard negative mining, the model embeds parent-child spending patterns directly into its parameters, thus eliminating the need for costly, real-time parent transaction lookups. This significantly reduces cross-shard communication overhead, boosting throughput and scalability. 

**Abstract (ZH)**: 基于机器学习的UTXO区块链可扩展性研究：结合对比学习和无监督学习的交易输出嵌入框架 

---
# Policy Newton Algorithm in Reproducing Kernel Hilbert Space 

**Title (ZH)**: 政策牛顿算法在再生核希尔伯特空间中 

**Authors**: Yixian Zhang, Huaze Tang, Chao Wang, Wenbo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2506.01597)  

**Abstract**: Reinforcement learning (RL) policies represented in Reproducing Kernel Hilbert Spaces (RKHS) offer powerful representational capabilities. While second-order optimization methods like Newton's method demonstrate faster convergence than first-order approaches, current RKHS-based policy optimization remains constrained to first-order techniques. This limitation stems primarily from the intractability of explicitly computing and inverting the infinite-dimensional Hessian operator in RKHS. We introduce Policy Newton in RKHS, the first second-order optimization framework specifically designed for RL policies represented in RKHS. Our approach circumvents direct computation of the inverse Hessian operator by optimizing a cubic regularized auxiliary objective function. Crucially, we leverage the Representer Theorem to transform this infinite-dimensional optimization into an equivalent, computationally tractable finite-dimensional problem whose dimensionality scales with the trajectory data volume. We establish theoretical guarantees proving convergence to a local optimum with a local quadratic convergence rate. Empirical evaluations on a toy financial asset allocation problem validate these theoretical properties, while experiments on standard RL benchmarks demonstrate that Policy Newton in RKHS achieves superior convergence speed and higher episodic rewards compared to established first-order RKHS approaches and parametric second-order methods. Our work bridges a critical gap between non-parametric policy representations and second-order optimization methods in reinforcement learning. 

**Abstract (ZH)**: RKHS中表示的RL策略的Reinforcement Learning新方法：第二-order优化框架 

---
# Understanding and Improving Laplacian Positional Encodings For Temporal GNNs 

**Title (ZH)**: 理解并改进拉普拉斯位置编码以优化时间动态图神经网络 

**Authors**: Yaniv Galron, Fabrizio Frasca, Haggai Maron, Eran Treister, Moshe Eliasof  

**Link**: [PDF](https://arxiv.org/pdf/2506.01596)  

**Abstract**: Temporal graph learning has applications in recommendation systems, traffic forecasting, and social network analysis. Although multiple architectures have been introduced, progress in positional encoding for temporal graphs remains limited. Extending static Laplacian eigenvector approaches to temporal graphs through the supra-Laplacian has shown promise, but also poses key challenges: high eigendecomposition costs, limited theoretical understanding, and ambiguity about when and how to apply these encodings. In this paper, we address these issues by (1) offering a theoretical framework that connects supra-Laplacian encodings to per-time-slice encodings, highlighting the benefits of leveraging additional temporal connectivity, (2) introducing novel methods to reduce the computational overhead, achieving up to 56x faster runtimes while scaling to graphs with 50,000 active nodes, and (3) conducting an extensive experimental study to identify which models, tasks, and datasets benefit most from these encodings. Our findings reveal that while positional encodings can significantly boost performance in certain scenarios, their effectiveness varies across different models. 

**Abstract (ZH)**: 时空图学习在推荐系统、交通预测和社会网络分析中有应用。虽然已引入多种架构，但时空图的位置编码进展有限。通过超拉普拉斯扩展静态拉普拉斯特征向量方法显示出潜力，但也提出了关键挑战：高昂的特征分解成本、有限的理论理解以及何时以及如何应用这些编码的模糊性。本文通过（1）提供一个理论框架，将超拉普拉斯编码与每时间片编码连接起来，强调利用额外的时间连接性的好处；（2）引入新型方法减少计算开销，实现高达56倍更快的运行时间并在包含50,000个活跃节点的图上进行扩展；（3）进行广泛实验研究以确定哪些模型、任务和数据集最受益于这些编码。我们的研究发现表明，虽然位置编码在某些场景中能显著提升性能，但其有效性因不同的模型而异。时空图学习在推荐系统、交通预测和社会网络分析中有应用。虽然已引入多种架构，但时空图的位置编码进展有限。通过超拉普拉斯扩展静态拉普拉斯特征向量方法显示出潜力，但也提出了关键挑战：高昂的特征分解成本、有限的理论理解以及何时以及如何应用这些编码的模糊性。本文通过提供一个理论框架、引入新型方法减少计算开销以及进行广泛实验研究来解决这些问题。 

---
# VirnyFlow: A Design Space for Responsible Model Development 

**Title (ZH)**: VirnyFlow: 负责任模型开发的设计空间 

**Authors**: Denys Herasymuk, Nazar Protsiv, Julia Stoyanovich  

**Link**: [PDF](https://arxiv.org/pdf/2506.01584)  

**Abstract**: Developing machine learning (ML) models requires a deep understanding of real-world problems, which are inherently multi-objective. In this paper, we present VirnyFlow, the first design space for responsible model development, designed to assist data scientists in building ML pipelines that are tailored to the specific context of their problem. Unlike conventional AutoML frameworks, VirnyFlow enables users to define customized optimization criteria, perform comprehensive experimentation across pipeline stages, and iteratively refine models in alignment with real-world constraints. Our system integrates evaluation protocol definition, multi-objective Bayesian optimization, cost-aware multi-armed bandits, query optimization, and distributed parallelism into a unified architecture. We show that VirnyFlow significantly outperforms state-of-the-art AutoML systems in both optimization quality and scalability across five real-world benchmarks, offering a flexible, efficient, and responsible alternative to black-box automation in ML development. 

**Abstract (ZH)**: 开发机器学习模型需要深刻理解现实世界的问题，这些问题往往是多目标的。本文介绍了VirnyFlow，这是首个负责任模型开发的设计空间，旨在协助数据科学家构建适应其具体问题背景的ML管道。与传统的AutoML框架不同，VirnyFlow允许用户定义自定义的优化标准，在管道各个阶段进行全面实验，并在符合现实世界约束的情况下逐步优化模型。该系统将评估协议定义、多目标贝叶斯优化、成本感知多臂老虎机、查询优化和分布式并行性整合到统一架构中。我们展示了VirnyFlow在五个真实基准上的优化质量和可扩展性都显著优于最先进的AutoML系统，提供了灵活、高效且负责任的黑盒自动化替代方案。 

---
# Advanced Nanostructured Topical Therapeutics for Psoriasis: Strategic Synthesis, Multimodal Characterization, and Preliminary Pharmacodynamic Profiling 

**Title (ZH)**: 纳米结构皮肤治疗药物在银屑病中的高级应用：策略合成、多模态表征及初步药效学分析 

**Authors**: Iqra Yousaf, Aqsa Yousaf  

**Link**: [PDF](https://arxiv.org/pdf/2506.01572)  

**Abstract**: Psoriasis is a long-term inflammatory skin disease that remains difficult to treat. In this study, we developed a new topical treatment by combining metal oxide nanoparticles: cerium oxide (CeO2), zinc oxide (ZnO), and silver (Ag), with natural plant extracts in a gel made from fish collagen and agar. The nanoparticles were characterized using UV-Vis spectroscopy, dynamic light scattering (DLS), Fourier-transform infrared spectroscopy (FTIR), and scanning electron microscopy (SEM), showing good stability and a uniform particle size distribution (ZnO averaged 66 nm).
To enhance therapeutic potential, the gel was enriched with plant-derived antioxidants from bitter melon, ginger, and neem. This formulation was tested on an animal model of psoriasis. The treated group exhibited faster wound healing and reduced inflammation compared to both placebo and untreated groups, with statistically significant results (p < 0.01 to p < 0.001) observed from Day 3, becoming more pronounced by Day 14.
These results indicate that the combination of nanoparticles with plant-based components in a topical gel may provide a promising new approach to psoriasis treatment. Further studies are recommended to evaluate long-term safety and therapeutic effectiveness. 

**Abstract (ZH)**: 氧化金属纳米粒子与植物提取物联合治疗银屑病的新外用制剂及其疗效研究 

---
# FlexiSAGA: A Flexible Systolic Array GEMM Accelerator for Sparse and Dense Processing 

**Title (ZH)**: FlexiSAGA: 一种灵活的 systolic array GEMM 加速器，适用于稀疏和稠密处理 

**Authors**: Mika Markus Müller, Konstantin Lübeck, Alexander Louis-Ferdinand Jung, Jannik Steinmetz, Oliver Bringmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.01566)  

**Abstract**: Artificial Intelligence (AI) algorithms, such as Deep Neural Networks (DNNs), have become an important tool for a wide range of applications, from computer vision to natural language processing. However, the computational complexity of DNN inference poses a significant challenge, particularly for processing on resource-constrained edge devices. One promising approach to address this challenge is the exploitation of sparsity in DNN operator weights.
In this work, we present FlexiSAGA, an architecturally configurable and dataflow-flexible AI hardware accelerator for the sparse and dense processing of general matrix multiplications (GEMMs). FlexiSAGA supports seven different sparse and dense dataflows, enabling efficient processing of resource intensive DNN operators. Additionally, we propose a DNN pruning method specifically tailored towards the FlexiSAGA architecture, allowing for near-optimal processing of dense and sparse convolution and fully-connected operators, facilitating a DNN/HW co-design flow. Our results show a whole DNN sparse-over-dense inference speedup ranging from 1.41 up to 4.28, outperforming commercial and literature-reported accelerator platforms. 

**Abstract (ZH)**: 人工神经网络（DNNs）等人工 Intelligence (AI) 算法已成为从计算机视觉到自然语言处理等广泛应用的重要工具。然而，DNN 推断的计算复杂性对资源受限的边缘设备构成了重大挑战。一种有前景的应对策略是利用 DNN 运算权重的稀疏性。

在本文中，我们提出了 FlexiSAGA，一种架构可配置且数据流灵活的人工智能硬件加速器，适用于通用矩阵乘法（GEMM）的稀疏和密集处理。FlexiSAGA 支持七种不同的稀疏和密集数据流，能够高效处理资源密集型 DNN 运算。此外，我们提出了一种针对 FlexiSAGA 架构的 DNN 裁剪方法，使得密集和稀疏卷积以及全连接运算的处理接近最优，促进了 DNN/HW 共同设计流程。我们的结果表明，整个 DNN 稀疏-密集推断加速范围从 1.41 到 4.28，优于商业和文献报告的加速器平台。 

---
# Dictionaries to the Rescue: Cross-Lingual Vocabulary Transfer for Low-Resource Languages Using Bilingual Dictionaries 

**Title (ZH)**: 词典来帮忙：使用双语词典为低资源语言进行跨语言词汇迁移 

**Authors**: Haruki Sakajo, Yusuke Ide, Justin Vasselli, Yusuke Sakai, Yingtao Tian, Hidetaka Kamigaito, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2506.01535)  

**Abstract**: Cross-lingual vocabulary transfer plays a promising role in adapting pre-trained language models to new languages, including low-resource languages. Existing approaches that utilize monolingual or parallel corpora face challenges when applied to languages with limited resources. In this work, we propose a simple yet effective vocabulary transfer method that utilizes bilingual dictionaries, which are available for many languages, thanks to descriptive linguists. Our proposed method leverages a property of BPE tokenizers where removing a subword from the vocabulary causes a fallback to shorter subwords. The embeddings of target subwords are estimated iteratively by progressively removing them from the tokenizer. The experimental results show that our approach outperforms existing methods for low-resource languages, demonstrating the effectiveness of a dictionary-based approach for cross-lingual vocabulary transfer. 

**Abstract (ZH)**: 跨语言词汇转移在适应预训练语言模型到新语言（包括低资源语言）中发挥着有前途的作用。现有的方法在应用到资源受限的语言时面临挑战。在本工作中，我们提出了一种简单而有效的方法，该方法利用了描述语言学家提供给许多语言的双语词典。我们提出的方法利用了BPE分词器的一个特性，即从词汇表中移除一个子词会导致退回到较短的子词。目标子词的嵌入通过逐步从分词器中移除它们来迭代估计。实验结果表明，我们的方法在低资源语言中优于现有方法，展示了基于词典的方法在跨语言词汇转移中的有效性。 

---
# A Diffusion-Based Method for Learning the Multi-Outcome Distribution of Medical Treatments 

**Title (ZH)**: 基于扩散的方法学习医疗治疗的多结果分布 

**Authors**: Yuchen Ma, Jonas Schweisthal, Hengrui Zhang, Stefan Feuerriegel  

**Link**: [PDF](https://arxiv.org/pdf/2506.01533)  

**Abstract**: In medicine, treatments often influence multiple, interdependent outcomes, such as primary endpoints, complications, adverse events, or other secondary endpoints. Hence, to make optimal treatment decisions, clinicians are interested in learning the distribution of multi-dimensional treatment outcomes. However, the vast majority of machine learning methods for predicting treatment effects focus on single-outcome settings, despite the fact that medical data often include multiple, interdependent outcomes. To address this limitation, we propose a novel diffusion-based method called DIME to learn the joint distribution of multiple outcomes of medical treatments. We addresses three challenges relevant in medical practice: (i)it is tailored to learn the joint interventional distribution of multiple medical outcomes, which enables reliable decision-making with uncertainty quantification rather than relying solely on point estimates; (ii)it explicitly captures the dependence structure between outcomes; (iii)it can handle outcomes of mixed type, including binary, categorical, and continuous variables. In DIME, we take into account the fundamental problem of causal inference through causal masking. For training, our method decomposes the joint distribution into a series of conditional distributions with a customized conditional masking to account for the dependence structure across outcomes. For inference, our method auto-regressively generates predictions. This allows our method to move beyond point estimates of causal quantities and thus learn the joint interventional distribution. To the best of our knowledge, DIME is the first neural method tailored to learn the joint, multi-outcome distribution of medical treatments. Across various experiments, we demonstrate that our method effectively learns the joint distribution and captures shared information among multiple outcomes. 

**Abstract (ZH)**: 医学中，治疗方法往往影响多个相互依赖的结局，如主要终点、并发症、不良反应或其他次要终点。因此，为了做出最优的治疗决策，临床医生对学习多维度治疗结局的分布感兴趣。然而，大多数用于预测治疗效果的机器学习方法集中在单一结局的设置上，尽管医学数据通常包含多个相互依赖的结局。为了解决这一局限性，我们提出了一种基于扩散的新方法DIME，用于学习医疗治疗多结局的联合分布。DIME解决了医学实践中相关的三个挑战：(i) 它专门设计用于学习多个医疗结局的联合干预分布，从而在不确定性量化的基础上实现可靠的决策，而不仅仅是依赖点估计；(ii) 它明确捕捉了结局之间的依赖结构；(iii) 它可以处理包括二元、分类和连续变量在内的混合类型结局。在DIME中，我们通过因果掩码考虑了因果推理的基本问题。在训练过程中，我们的方法将联合分布分解为一系列条件分布，并通过自定义条件掩码来考虑结局之间的依赖结构。在推理过程中，我们的方法自回归生成预测。这使得我们的方法能够超越因果量的点估计，并因此学习到联合干预分布。据我们所知，DIME是第一个专门设计用于学习医疗治疗多结局联合分布的神经网络方法。在各种实验中，我们证明了该方法有效学习了联合分布，并捕捉了多个结局之间的共享信息。 

---
# Learning of Population Dynamics: Inverse Optimization Meets JKO Scheme 

**Title (ZH)**: 基于逆优化的群体动力学习：JKO方案的应用 

**Authors**: Mikhail Persiianov, Jiawei Chen, Petr Mokrov, Alexander Tyurin, Evgeny Burnaev, Alexander Korotin  

**Link**: [PDF](https://arxiv.org/pdf/2506.01502)  

**Abstract**: Learning population dynamics involves recovering the underlying process that governs particle evolution, given evolutionary snapshots of samples at discrete time points. Recent methods frame this as an energy minimization problem in probability space and leverage the celebrated JKO scheme for efficient time discretization. In this work, we introduce $\texttt{iJKOnet}$, an approach that combines the JKO framework with inverse optimization techniques to learn population dynamics. Our method relies on a conventional $\textit{end-to-end}$ adversarial training procedure and does not require restrictive architectural choices, e.g., input-convex neural networks. We establish theoretical guarantees for our methodology and demonstrate improved performance over prior JKO-based methods. 

**Abstract (ZH)**: 学习种群动态涉及在给定离散时间点的演化快照情况下，恢复调控粒子演化的底层过程。 recent methods 将这一问题框架化为概率空间中的能量最小化问题，并利用著名的 JKO 方案进行高效的时域离散化。在本文中，我们引入了 $\texttt{iJKOnet}$ 方法，该方法将 JKO 框架与逆优化技术相结合以学习种群动态。我们的方法依赖于标准的端到端.adversarial 训练过程，且不需要限制性的架构选择，例如输入凸神经网络。我们为我们的方法建立了理论保证，并展示了与先前的基于 JKO 的方法相比的优越性能。 

---
# Automatic Stage Lighting Control: Is it a Rule-Driven Process or Generative Task? 

**Title (ZH)**: 自动舞台灯光控制：这是一种规则驱动的过程还是生成任务？ 

**Authors**: Zijian Zhao, Dian Jin, Zijing Zhou, Xiaoyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01482)  

**Abstract**: Stage lighting plays an essential role in live music performances, influencing the engaging experience of both musicians and audiences. Given the high costs associated with hiring or training professional lighting engineers, Automatic Stage Lighting Control (ASLC) has gained increasing attention. However, most existing approaches only classify music into limited categories and map them to predefined light patterns, resulting in formulaic and monotonous outcomes that lack rationality. To address this issue, this paper presents an end-to-end solution that directly learns from experienced lighting engineers -- Skip-BART. To the best of our knowledge, this is the first work to conceptualize ASLC as a generative task rather than merely a classification problem. Our method modifies the BART model to take audio music as input and produce light hue and value (intensity) as output, incorporating a novel skip connection mechanism to enhance the relationship between music and light within the frame this http URL validate our method through both quantitative analysis and an human evaluation, demonstrating that Skip-BART outperforms conventional rule-based methods across all evaluation metrics and shows only a limited gap compared to real lighting this http URL, our method yields a p-value of 0.72 in a statistical comparison based on human evaluations with human lighting engineers, suggesting that the proposed approach closely matches human lighting engineering performance. To support further research, we have made our self-collected dataset, code, and trained model parameters available at this https URL . 

**Abstract (ZH)**: 自动舞台灯光控制在live音乐表演中的应用：基于Skip-BART模型的生成式解决方案 

---
# GenDMR: A dynamic multimodal role-swapping network for identifying risk gene phenotypes 

**Title (ZH)**: GenDMR：一种动态多模态角色轮换网络，用于识别风险基因表型 

**Authors**: Lina Qin, Cheng Zhu, Chuqi Zhou, Yukun Huang, Jiayi Zhu, Ping Liang, Jinju Wang, Yixing Huang, Cheng Luo, Dezhong Yao, Ying Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.01456)  

**Abstract**: Recent studies have shown that integrating multimodal data fusion techniques for imaging and genetic features is beneficial for the etiological analysis and predictive diagnosis of Alzheimer's disease (AD). However, there are several critical flaws in current deep learning methods. Firstly, there has been insufficient discussion and exploration regarding the selection and encoding of genetic information. Secondly, due to the significantly superior classification value of AD imaging features compared to genetic features, many studies in multimodal fusion emphasize the strengths of imaging features, actively mitigating the influence of weaker features, thereby diminishing the learning of the unique value of genetic features. To address this issue, this study proposes the dynamic multimodal role-swapping network (GenDMR). In GenDMR, we develop a novel approach to encode the spatial organization of single nucleotide polymorphisms (SNPs), enhancing the representation of their genomic context. Additionally, to adaptively quantify the disease risk of SNPs and brain region, we propose a multi-instance attention module to enhance model interpretability. Furthermore, we introduce a dominant modality selection module and a contrastive self-distillation module, combining them to achieve a dynamic teacher-student role exchange mechanism based on dominant and auxiliary modalities for bidirectional co-updating of different modal data. Finally, GenDMR achieves state-of-the-art performance on the ADNI public dataset and visualizes attention to different SNPs, focusing on confirming 12 potential high-risk genes related to AD, including the most classic APOE and recently highlighted significant risk genes. This demonstrates GenDMR's interpretable analytical capability in exploring AD genetic features, providing new insights and perspectives for the development of multimodal data fusion techniques. 

**Abstract (ZH)**: Recent Studies Have Demonstrated the Beneficial Impact of Integrating Multimodal Data Fusion Techniques for Imaging and Genetic Features in the Etiological Analysis and Predictive Diagnosis of Alzheimer's Disease (AD): However, Current Deep Learning Methods Are Subject to Several Critical Flaws 

---
# From Initial Data to Boundary Layers: Neural Networks for Nonlinear Hyperbolic Conservation Laws 

**Title (ZH)**: 从初始数据到边界层：神经网络在非线性双曲守恒律中的应用 

**Authors**: Igor Ciril, Khalil Haddaoui, Yohann Tendero  

**Link**: [PDF](https://arxiv.org/pdf/2506.01453)  

**Abstract**: We address the approximation of entropy solutions to initial-boundary value problems for nonlinear strictly hyperbolic conservation laws using neural networks. A general and systematic framework is introduced for the design of efficient and reliable learning algorithms, combining fast convergence during training with accurate predictions. The methodology is assessed through a series of one-dimensional scalar test cases, highlighting its potential applicability to more complex industrial scenarios. 

**Abstract (ZH)**: 我们利用神经网络解决非线性严格双曲守恒律初始边值问题的熵解近似，并介绍了一种高效可靠的学习算法设计框架，该框架结合了训练期间的快速收敛和准确预测。该方法通过一系列一维标量测试案例进行了评估，展示了其在更复杂工业场景中的潜在应用。 

---
# ShaTS: A Shapley-based Explainability Method for Time Series Artificial Intelligence Models applied to Anomaly Detection in Industrial Internet of Things 

**Title (ZH)**: 基于Shapley值的时序人工智能模型可解释性方法及其在工业物联网异常检测中的应用 

**Authors**: Manuel Franco de la Peña, Ángel Luis Perales Gómez, Lorenzo Fernández Maimó  

**Link**: [PDF](https://arxiv.org/pdf/2506.01450)  

**Abstract**: Industrial Internet of Things environments increasingly rely on advanced Anomaly Detection and explanation techniques to rapidly detect and mitigate cyberincidents, thereby ensuring operational safety. The sequential nature of data collected from these environments has enabled improvements in Anomaly Detection using Machine Learning and Deep Learning models by processing time windows rather than treating the data as tabular. However, conventional explanation methods often neglect this temporal structure, leading to imprecise or less actionable explanations. This work presents ShaTS (Shapley values for Time Series models), which is a model-agnostic explainable Artificial Intelligence method designed to enhance the precision of Shapley value explanations for time series models. ShaTS addresses the shortcomings of traditional approaches by incorporating an a priori feature grouping strategy that preserves temporal dependencies and produces both coherent and actionable insights. Experiments conducted on the SWaT dataset demonstrate that ShaTS accurately identifies critical time instants, precisely pinpoints the sensors, actuators, and processes affected by anomalies, and outperforms SHAP in terms of both explainability and resource efficiency, fulfilling the real-time requirements of industrial environments. 

**Abstract (ZH)**: 工业物联网环境 increasingly依靠先进的异常检测和解释技术快速检测和缓解网络事件，以确保操作安全。这些环境采集的数据序列性促使通过处理时间窗口而非将数据视为表格的方式来提高基于机器学习和深度学习的异常检测性能。然而，传统的解释方法往往忽视了这种时间结构，导致不精确或不可操作的解释。本文介绍了ShaTS（时间序列模型的Shapley值方法），这是一种模型无关的解释性人工智能方法，旨在增强时间序列模型Shapley值解释的精确性。ShaTS通过结合先验特征分组策略来处理传统方法的不足，该策略保留了时间依赖性并产生连贯且可操作的见解。基于SWaT数据集的实验表明，ShaTS准确地识别了关键的时间瞬间，精确地确定了受影响的传感器、执行器和过程，并在解释性和资源效率方面优于SHAP，满足了工业环境的实时要求。 

---
# System Calls for Malware Detection and Classification: Methodologies and Applications 

**Title (ZH)**: 恶意软件检测与分类的系统调用方法及其应用 

**Authors**: Bishwajit Prasad Gond, Durga Prasad Mohapatra  

**Link**: [PDF](https://arxiv.org/pdf/2506.01412)  

**Abstract**: As malware continues to become more complex and harder to detect, Malware Analysis needs to continue to evolve to stay one step ahead. One promising key area approach focuses on using system calls and API Calls, the core communication between user applications and the operating system and their kernels. These calls provide valuable insight into how software or programs behaves, making them an useful tool for spotting suspicious or harmful activity of programs and software. This chapter takes a deep down look at how system calls are used in malware detection and classification, covering techniques like static and dynamic analysis, as well as sandboxing. By combining these methods with advanced techniques like machine learning, statistical analysis, and anomaly detection, researchers can analyze system call patterns to tell the difference between normal and malicious behavior. The chapter also explores how these techniques are applied across different systems, including Windows, Linux, and Android, while also looking at the ways sophisticated malware tries to evade detection. 

**Abstract (ZH)**: 随着恶意软件变得越来越复杂且更难检测，恶意软件分析需要不断进化以保持领先。一种有前景的关键方法是利用系统调用和API调用，这些调用是用户应用程序与操作系统及其内核之间核心通信的基础。这些调用提供了关于软件或程序行为的宝贵见解，使它们成为识别程序和软件中的可疑或有害活动的有效工具。本章深入探讨了系统调用在恶意软件检测和分类中的应用，涵盖了静态分析和动态分析等技术，以及沙箱技术。通过将这些方法与机器学习、统计分析和异常检测等高级技术相结合，研究人员可以分析系统调用模式以区分正常和恶意行为。本章还探讨了这些技术在Windows、Linux和Android等不同系统中的应用，同时也考察了复杂恶意软件如何规避检测的方法。 

---
# VRD-IU: Lessons from Visually Rich Document Intelligence and Understanding 

**Title (ZH)**: VRD-IU：视觉丰富文档的智能与理解经验 

**Authors**: Yihao Ding, Soyeon Caren Han, Yan Li, Josiah Poon  

**Link**: [PDF](https://arxiv.org/pdf/2506.01388)  

**Abstract**: Visually Rich Document Understanding (VRDU) has emerged as a critical field in document intelligence, enabling automated extraction of key information from complex documents across domains such as medical, financial, and educational applications. However, form-like documents pose unique challenges due to their complex layouts, multi-stakeholder involvement, and high structural variability. Addressing these issues, the VRD-IU Competition was introduced, focusing on extracting and localizing key information from multi-format forms within the Form-NLU dataset, which includes digital, printed, and handwritten documents. This paper presents insights from the competition, which featured two tracks: Track A, emphasizing entity-based key information retrieval, and Track B, targeting end-to-end key information localization from raw document images. With over 20 participating teams, the competition showcased various state-of-the-art methodologies, including hierarchical decomposition, transformer-based retrieval, multimodal feature fusion, and advanced object detection techniques. The top-performing models set new benchmarks in VRDU, providing valuable insights into document intelligence. 

**Abstract (ZH)**: 视觉丰富的文档理解（VRDU）已成为文档智能领域的关键领域，使自动化从跨医疗、金融和教育应用等领域的复杂文档中提取关键信息成为可能。然而，表格形式的文档由于其复杂的布局、多利益相关者的参与和高度的结构性变异性，提出了独特的挑战。为应对这些挑战，引入了VRD-IU竞赛，专注于在Form-NLU数据集中从多种格式的表格中提取和本地化关键信息，该数据集包括数字、打印和手写文档。本文介绍了竞赛的见解，竞赛设有两条赛道：A轨道侧重于基于实体的关键信息检索，B轨道则旨在从原始文档图像中端到端地定位关键信息。超过20个参赛团队展示了包括分层分解、基于变换器的检索、多模态特征融合和高级目标检测技术在内的多种先进的方法。表现最佳的模型在VRDU领域设立了新的基准，提供了宝贵的文档智能见解。 

---
# Unraveling Spatio-Temporal Foundation Models via the Pipeline Lens: A Comprehensive Review 

**Title (ZH)**: 通过流水线视角解析时空基础模型：一个综合Review 

**Authors**: Yuchen Fang, Hao Miao, Yuxuan Liang, Liwei Deng, Yue Cui, Ximu Zeng, Yuyang Xia, Yan Zhao, Torben Bach Pedersen, Christian S. Jensen, Xiaofang Zhou, Kai Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.01364)  

**Abstract**: Spatio-temporal deep learning models aims to utilize useful patterns in such data to support tasks like prediction. However, previous deep learning models designed for specific tasks typically require separate training for each use case, leading to increased computational and storage costs. To address this issue, spatio-temporal foundation models have emerged, offering a unified framework capable of solving multiple spatio-temporal tasks. These foundation models achieve remarkable success by learning general knowledge with spatio-temporal data or transferring the general capabilities of pre-trained language models. While previous surveys have explored spatio-temporal data and methodologies separately, they have ignored a comprehensive examination of how foundation models are designed, selected, pre-trained, and adapted. As a result, the overall pipeline for spatio-temporal foundation models remains unclear. To bridge this gap, we innovatively provide an up-to-date review of previous spatio-temporal foundation models from the pipeline perspective. The pipeline begins with an introduction to different types of spatio-temporal data, followed by details of data preprocessing and embedding techniques. The pipeline then presents a novel data property taxonomy to divide existing methods according to data sources and dependencies, providing efficient and effective model design and selection for researchers. On this basis, we further illustrate the training objectives of primitive models, as well as the adaptation techniques of transferred models. Overall, our survey provides a clear and structured pipeline to understand the connection between core elements of spatio-temporal foundation models while guiding researchers to get started quickly. Additionally, we introduce emerging opportunities such as multi-objective training in the field of spatio-temporal foundation models. 

**Abstract (ZH)**: 时空深度学习模型旨在利用时空数据中的有用模式以支持预测等任务。然而，之前为特定任务设计的深度学习模型通常需要为每种使用场景分别进行训练，导致计算和存储成本增加。为解决这一问题，时空基础模型已崭露头角，提供了一个能够解决多种时空任务的统一框架。通过学习时空数据中的通用知识或转移预训练语言模型的通用能力，这些基础模型取得了显著的成功。尽管以往的综述分别探讨了时空数据和方法，但忽视了对基础模型设计、选择、预训练和适应的全面考察。因此，时空基础模型的整体管道流程仍不清晰。为弥补这一缺口，我们从管道视角创新性地提供了一种对时空基础模型的最新综述。该管道从不同类型的时空数据介绍开始，接着详细说明数据预处理和嵌入技术。然后，该管道提出了一个新的数据属性分类法，根据数据来源和依赖关系对现有方法进行分类，为研究人员提供高效有效的模型设计和选择。在此基础上，我们进一步阐述了基础模型的训练目标以及转移模型的适应技术。总体而言，我们的综述提供了一条清晰而结构化的管道，指导研究人员理解时空基础模型核心元素之间的关系并迅速入门。此外，我们还介绍了时空基础模型领域新兴的机会，例如多目标训练。 

---
# NoiseAR: AutoRegressing Initial Noise Prior for Diffusion Models 

**Title (ZH)**: NoiseAR：用于扩散模型的自回归初始噪声先验 

**Authors**: Zeming Li, Xiangyue Liu, Xiangyu Zhang, Ping Tan, Heung-Yeung Shum  

**Link**: [PDF](https://arxiv.org/pdf/2506.01337)  

**Abstract**: Diffusion models have emerged as powerful generative frameworks, creating data samples by progressively denoising an initial random state. Traditionally, this initial state is sampled from a simple, fixed distribution like isotropic Gaussian, inherently lacking structure and a direct mechanism for external control. While recent efforts have explored ways to introduce controllability into the diffusion process, particularly at the initialization stage, they often rely on deterministic or heuristic approaches. These methods can be suboptimal, lack expressiveness, and are difficult to scale or integrate into more sophisticated optimization frameworks. In this paper, we introduce NoiseAR, a novel method for AutoRegressive Initial Noise Prior for Diffusion Models. Instead of a static, unstructured source, NoiseAR learns to generate a dynamic and controllable prior distribution for the initial noise. We formulate the generation of the initial noise prior's parameters as an autoregressive probabilistic modeling task over spatial patches or tokens. This approach enables NoiseAR to capture complex spatial dependencies and introduce learned structure into the initial state. Crucially, NoiseAR is designed to be conditional, allowing text prompts to directly influence the learned prior, thereby achieving fine-grained control over the diffusion initialization. Our experiments demonstrate that NoiseAR can generate initial noise priors that lead to improved sample quality and enhanced consistency with conditional inputs, offering a powerful, learned alternative to traditional random initialization. A key advantage of NoiseAR is its probabilistic formulation, which naturally supports seamless integration into probabilistic frameworks like Markov Decision Processes and Reinforcement Learning. Our code will be available at this https URL 

**Abstract (ZH)**: NoiseAR：自动回归初始噪声先验方法用于扩散模型 

---
# STSA: Federated Class-Incremental Learning via Spatial-Temporal Statistics Aggregation 

**Title (ZH)**: STSA: 基于空间-时间统计聚合的联邦类增量学习 

**Authors**: Zenghao Guan, Guojun Zhu, Yucan Zhou, Wu Liu, Weiping Wang, Jiebo Luo, Xiaoyan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01327)  

**Abstract**: Federated Class-Incremental Learning (FCIL) enables Class-Incremental Learning (CIL) from distributed data. Existing FCIL methods typically integrate old knowledge preservation into local client training. However, these methods cannot avoid spatial-temporal client drift caused by data heterogeneity and often incur significant computational and communication overhead, limiting practical deployment. To address these challenges simultaneously, we propose a novel approach, Spatial-Temporal Statistics Aggregation (STSA), which provides a unified framework to aggregate feature statistics both spatially (across clients) and temporally (across stages). The aggregated feature statistics are unaffected by data heterogeneity and can be used to update the classifier in closed form at each stage. Additionally, we introduce STSA-E, a communication-efficient variant with theoretical guarantees, achieving similar performance to STSA-E with much lower communication overhead. Extensive experiments on three widely used FCIL datasets, with varying degrees of data heterogeneity, show that our method outperforms state-of-the-art FCIL methods in terms of performance, flexibility, and both communication and computation efficiency. 

**Abstract (ZH)**: 联邦类增量学习（FCIL）使分布式数据上的类增量学习（CIL）成为可能。现有的FCIL方法通常将旧知识保留融入到局部客户端训练中。然而，这些方法无法避免由数据异质性引起的时空客户端漂移，并且往往会导致显著的计算和通信开销，限制了其实用部署。为了解决这些挑战，我们提出了一个新颖的方法，时空统计聚合（STSA），它提供了一个统一框架来聚合特征统计（空间上跨客户端和时间上跨阶段）。聚合的特征统计不受数据异质性的影响，并且可以在每个阶段以封闭形式更新分类器。此外，我们引入了STSA-E，这是一种通信高效的变体，并具有理论保证，其性能与STSA-E相当，但通信开销更低。在三个广泛使用的FCIL数据集上进行的大量实验，数据异质性程度不同，表明我们的方法在性能、灵活性以及通信和计算效率方面优于现有的最先进的FCIL方法。 

---
# $Ψ$-Sampler: Initial Particle Sampling for SMC-Based Inference-Time Reward Alignment in Score Models 

**Title (ZH)**: $Ψ$-Sampler: 初始粒子采样算法在基于SMC的推理时奖励对齐中的应用 

**Authors**: Taehoon Yoon, Yunhong Min, Kyeongmin Yeo, Minhyuk Sung  

**Link**: [PDF](https://arxiv.org/pdf/2506.01320)  

**Abstract**: We introduce $\Psi$-Sampler, an SMC-based framework incorporating pCNL-based initial particle sampling for effective inference-time reward alignment with a score-based generative model. Inference-time reward alignment with score-based generative models has recently gained significant traction, following a broader paradigm shift from pre-training to post-training optimization. At the core of this trend is the application of Sequential Monte Carlo (SMC) to the denoising process. However, existing methods typically initialize particles from the Gaussian prior, which inadequately captures reward-relevant regions and results in reduced sampling efficiency. We demonstrate that initializing from the reward-aware posterior significantly improves alignment performance. To enable posterior sampling in high-dimensional latent spaces, we introduce the preconditioned Crank-Nicolson Langevin (pCNL) algorithm, which combines dimension-robust proposals with gradient-informed dynamics. This approach enables efficient and scalable posterior sampling and consistently improves performance across various reward alignment tasks, including layout-to-image generation, quantity-aware generation, and aesthetic-preference generation, as demonstrated in our experiments. 

**Abstract (ZH)**: 基于pCNL初始化粒子的Ψ-采样器：一种SMC框架，用于评分生成模型的推理时奖励对齐 

---
# Unlearning's Blind Spots: Over-Unlearning and Prototypical Relearning Attack 

**Title (ZH)**: 去学习的盲点：过度去学习与原型重学习攻击 

**Authors**: SeungBum Ha, Saerom Park, Sung Whan Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2506.01318)  

**Abstract**: Machine unlearning (MU) aims to expunge a designated forget set from a trained model without costly retraining, yet the existing techniques overlook two critical blind spots: "over-unlearning" that deteriorates retained data near the forget set, and post-hoc "relearning" attacks that aim to resurrect the forgotten knowledge. We first derive the over-unlearning metric OU@{\epsilon}, which represents the collateral damage to the nearby region of the forget set, where the over-unlearning mainly appears. Next, we expose an unforeseen relearning threat on MU, i.e., the Prototypical Relearning Attack, which exploits the per-class prototype of the forget class with just a few samples, and easily restores the pre-unlearning performance. To counter both blind spots, we introduce Spotter, a plug-and-play objective that combines (i) a masked knowledge-distillation penalty on the nearby region of forget set to suppress OU@{\epsilon}, and (ii) an intra-class dispersion loss that scatters forget-class embeddings, neutralizing prototypical relearning attacks. On CIFAR-10, as one of validations, Spotter reduces OU@{\epsilon}by below the 0.05X of the baseline, drives forget accuracy to 0%, preserves accuracy of the retain set within 1% of difference with the original, and denies the prototype-attack by keeping the forget set accuracy within <1%, without accessing retained data. It confirms that Spotter is a practical remedy of the unlearning's blind spots. 

**Abstract (ZH)**: 机器去学习：解决去学习中的过度去学习和后学习恢复攻击 

---
# General search techniques without common knowledge for imperfect-information games, and application to superhuman Fog of War chess 

**Title (ZH)**: 不依赖共同知识的一般搜索技术在不完全信息游戏中的应用：以超人类的迷雾棋为例 

**Authors**: Brian Hu Zhang, Tuomas Sandholm  

**Link**: [PDF](https://arxiv.org/pdf/2506.01242)  

**Abstract**: Since the advent of AI, games have served as progress benchmarks. Meanwhile, imperfect-information variants of chess have existed for over a century, present extreme challenges, and have been the focus of significant AI research. Beyond calculation needed in regular chess, they require reasoning about information gathering, the opponent's knowledge, signaling, etc. The most popular variant, Fog of War (FoW) chess (aka. dark chess) is a recognized challenge problem in AI after superhuman performance was reached in no-limit Texas hold'em poker. We present Obscuro, the first superhuman AI for FoW chess. It introduces advances to search in imperfect-information games, enabling strong, scalable reasoning. Experiments against the prior state-of-the-art AI and human players -- including the world's best -- show that Obscuro is significantly stronger. FoW chess is the largest (by amount of imperfect information) turn-based game in which superhuman performance has been achieved and the largest game in which imperfect-information search has been successfully applied. 

**Abstract (ZH)**: 自人工智能问世以来，游戏一直作为进步的标准。与此同时，信息不完全的棋类变种已有超过一个世纪的历史，极具挑战性，并一直是人工智能研究的重点。除了常规象棋所需的战略计算，它们还要求对信息收集、对手的知识、信号传递等进行推理。最流行的变种雾战象棋（Fog of War chess，又称暗棋）在无限制德州扑克达到超人类表现后，成为了人工智能中的一个公认挑战问题。我们提出了Obscuro，这是首个超人类雾战象棋AI。它在不完全信息游戏的搜索算法上取得了进步，使强大的、可扩展的推理成为可能。与之前的最先进的AI及人类玩家（包括世界最佳玩家）的实验表明，Obscuro显著更强大。雾战象棋是首个达到超人类表现且成功应用不完全信息搜索的最大（按信息不完全程度计算）回合制游戏，也是最大的此类游戏。 

---
# Fourier-Modulated Implicit Neural Representation for Multispectral Satellite Image Compression 

**Title (ZH)**: 傅里叶调制隐式神经表示在多光谱卫星图像压缩中的应用 

**Authors**: Woojin Cho, Steve Andreas Immanuel, Junhyuk Heo, Darongsae Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2506.01234)  

**Abstract**: Multispectral satellite images play a vital role in agriculture, fisheries, and environmental monitoring. However, their high dimensionality, large data volumes, and diverse spatial resolutions across multiple channels pose significant challenges for data compression and analysis. This paper presents ImpliSat, a unified framework specifically designed to address these challenges through efficient compression and reconstruction of multispectral satellite data. ImpliSat leverages Implicit Neural Representations (INR) to model satellite images as continuous functions over coordinate space, capturing fine spatial details across varying spatial resolutions. Furthermore, we introduce a Fourier modulation algorithm that dynamically adjusts to the spectral and spatial characteristics of each band, ensuring optimal compression while preserving critical image details. 

**Abstract (ZH)**: 多光谱卫星图像在农业、渔业和环境监测中发挥着重要作用，但由于其高维度、大数据量和多通道上的不同空间分辨率，这些图像在数据压缩和分析中面临着显著挑战。本文提出了ImpliSat，一种统一框架，专门用于通过高效压缩和重建多光谱卫星数据来应对这些挑战。ImpliSat 利用隐式神经表示（INR）将卫星图像建模为坐标空间上的连续函数，捕捉不同空间分辨率下的精细空间细节。此外，我们引入了一种傅里叶调制算法，能够动态调整以适应每通道的频谱和空间特性，确保在保留关键图像细节的同时实现最佳压缩。 

---
# Towards Efficient Few-shot Graph Neural Architecture Search via Partitioning Gradient Contribution 

**Title (ZH)**: 面向高效分割梯度贡献的少量-shot图神经架构搜索 

**Authors**: Wenhao Song, Xuan Wu, Bo Yang, You Zhou, Yubin Xiao, Yanchun Liang, Hongwei Ge, Heow Pueh Lee, Chunguo Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01231)  

**Abstract**: To address the weight coupling problem, certain studies introduced few-shot Neural Architecture Search (NAS) methods, which partition the supernet into multiple sub-supernets. However, these methods often suffer from computational inefficiency and tend to provide suboptimal partitioning schemes. To address this problem more effectively, we analyze the weight coupling problem from a novel perspective, which primarily stems from distinct modules in succeeding layers imposing conflicting gradient directions on the preceding layer modules. Based on this perspective, we propose the Gradient Contribution (GC) method that efficiently computes the cosine similarity of gradient directions among modules by decomposing the Vector-Jacobian Product during supernet backpropagation. Subsequently, the modules with conflicting gradient directions are allocated to distinct sub-supernets while similar ones are grouped together. To assess the advantages of GC and address the limitations of existing Graph Neural Architecture Search methods, which are limited to searching a single type of Graph Neural Networks (Message Passing Neural Networks (MPNNs) or Graph Transformers (GTs)), we propose the Unified Graph Neural Architecture Search (UGAS) framework, which explores optimal combinations of MPNNs and GTs. The experimental results demonstrate that GC achieves state-of-the-art (SOTA) performance in supernet partitioning quality and time efficiency. In addition, the architectures searched by UGAS+GC outperform both the manually designed GNNs and those obtained by existing NAS methods. Finally, ablation studies further demonstrate the effectiveness of all proposed methods. 

**Abstract (ZH)**: 针对权重耦合问题，某些研究引入了少样本神经架构搜索（NAS）方法，将超网络划分为多个子超网络。然而，这些方法通常存在计算效率低下的问题，并且倾向于提供次优的划分方案。为更有效地解决这一问题，我们从一个新的视角分析了权重耦合问题，该问题主要源于后续层中的不同模块在前一层模块上施加了相互矛盾的梯度方向。基于这一视角，我们提出了梯度贡献（GC）方法，通过在超网络反向传播过程中分解向量-雅可比积来高效计算模块之间梯度方向的余弦相似度。接着，具有相互矛盾梯度方向的模块被分配到不同的子超网络中，而相似的模块则被分组。为评估GC的优势并解决现有图神经架构搜索方法的局限性（仅限于搜索一种类型的图神经网络，如消息传递神经网络（MPNNs）或图变换器（GTs）），我们提出了统一图神经架构搜索（UGAS）框架，探索MPNNs和GTs的最佳组合。实验结果表明，GC在超网络划分质量和时间效率方面达到了目前的最先进（SOTA）水平。此外，UGAS+GC搜索到的架构优于手动设计的图神经网络和现有NAS方法获得的架构。最后，消融研究进一步证明了所有提出方法的有效性。 

---
# SPEAR: Security Posture Evaluation using AI Planner-Reasoning on Attack-Connectivity Hypergraphs 

**Title (ZH)**: SPEAR：基于攻击连通超图的AI规划推理安全态势评估 

**Authors**: Rakesh Podder, Turgay Caglar, Shadaab Kawnain Bashir, Sarath Sreedharan, Indrajit Ray, Indrakshi Ray  

**Link**: [PDF](https://arxiv.org/pdf/2506.01227)  

**Abstract**: Graph-based frameworks are often used in network hardening to help a cyber defender understand how a network can be attacked and how the best defenses can be deployed. However, incorporating network connectivity parameters in the attack graph, reasoning about the attack graph when we do not have access to complete information, providing system administrator suggestions in an understandable format, and allowing them to do what-if analysis on various scenarios and attacker motives is still missing. We fill this gap by presenting SPEAR, a formal framework with tool support for security posture evaluation and analysis that keeps human-in-the-loop. SPEAR uses the causal formalism of AI planning to model vulnerabilities and configurations in a networked system. It automatically converts network configurations and vulnerability descriptions into planning models expressed in the Planning Domain Definition Language (PDDL). SPEAR identifies a set of diverse security hardening strategies that can be presented in a manner understandable to the domain expert. These allow the administrator to explore the network hardening solution space in a systematic fashion and help evaluate the impact and compare the different solutions. 

**Abstract (ZH)**: 基于图的框架经常被用于网络加固，以助于网络安全防御者理解网络可能受到的攻击方式以及如何部署最优防御措施。然而，将网络连通性参数纳入攻击图、在缺乏完整信息的情况下进行攻击图推理、以易于理解的格式向系统管理员提供建议、以及允许他们在各种情景和攻击动机下进行“假设性”分析等功能仍然缺失。我们通过提出SPEAR（带有人工智能规划因果形式化的安全态势评估与分析框架，具有工具支持并保持人工在环）填补这一空白。SPEAR使用人工智能规划的因果形式化方法来建模网络系统中的漏洞和配置。它会自动将网络配置和漏洞描述转换为用规划域定义语言（PDDL）表达的规划模型。SPEAR能够识别出一系列多样的安全加固策略，这些策略可以通过易于理解的方式呈现给领域专家。这些策略使管理员能够系统地探索网络加固解决方案的空间，并帮助评估影响和比较不同的解决方案。 

---
# Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures 

**Title (ZH)**: 在稀疏自编码架构中融入层次化语义 

**Authors**: Mark Muchane, Sean Richardson, Kiho Park, Victor Veitch  

**Link**: [PDF](https://arxiv.org/pdf/2506.01197)  

**Abstract**: Sparse dictionary learning (and, in particular, sparse autoencoders) attempts to learn a set of human-understandable concepts that can explain variation on an abstract space. A basic limitation of this approach is that it neither exploits nor represents the semantic relationships between the learned concepts. In this paper, we introduce a modified SAE architecture that explicitly models a semantic hierarchy of concepts. Application of this architecture to the internal representations of large language models shows both that semantic hierarchy can be learned, and that doing so improves both reconstruction and interpretability. Additionally, the architecture leads to significant improvements in computational efficiency. 

**Abstract (ZH)**: 改进的语义层次结构Sparse自编码器架构：学习和利用语义层次结构以提高重构和可解释性及计算效率 

---
# Bridging Quantum and Classical Computing in Drug Design: Architecture Principles for Improved Molecule Generation 

**Title (ZH)**: 在药物设计中跨越量子与经典计算的桥梁：提高分子生成的架构原则 

**Authors**: Andrew Smith, Erhan Guven  

**Link**: [PDF](https://arxiv.org/pdf/2506.01177)  

**Abstract**: Hybrid quantum-classical machine learning offers a path to leverage noisy intermediate-scale quantum (NISQ) devices for drug discovery, but optimal model architectures remain unclear. We systematically optimize the quantum-classical bridge architecture for generative adversarial networks (GANs) in molecular discovery using multi-objective Bayesian optimization. Our optimized model (BO-QGAN) significantly improves performance, achieving a 2.27-fold higher Drug Candidate Score (DCS) than prior quantum-hybrid benchmarks and 2.21-fold higher than the classical baseline, using over 60% fewer parameters. Key findings favor layering multiple (3-4) shallow (4-8 qubit) quantum circuits sequentially, while classical architecture shows less sensitivity above a minimum capacity. This work provides the first empirically grounded architectural guidelines for hybrid models, enabling more effective integration of current quantum computers into pharmaceutical research pipelines. 

**Abstract (ZH)**: 混合量子-经典机器学习为利用噪声中间规模量子(NISQ)设备进行药物发现提供了路径，但最优模型架构仍不明确。我们通过多目标贝叶斯优化系统地优化了用于分子发现的生成对抗网络(GANs)的量子-经典桥梁架构。我们的优化模型(BO-QGAN)显著提高了性能，与之前的量子-混合基准相比，药物候选评分(DCS)提高了2.27倍，与经典基线相比提高了2.21倍，同时使用了超过60%更少的参数。关键发现倾向于依次堆叠多个(3-4个)浅层(4-8个量子比特)量子电路，而经典架构在最小容量以上显示出较低的敏感性。本项工作提供了首个经验性的混合模型架构指南，有助于更有效地将当前的量子计算机集成到制药研究管道中。 

---
# VUSA: Virtually Upscaled Systolic Array Architecture to Exploit Unstructured Sparsity in AI Acceleration 

**Title (ZH)**: VUSA：利用非结构化稀疏性加速AI加速的虚拟上尺度 systolic 数组架构 

**Authors**: Shereef Helal, Alberto Garcia-Ortiz, Lennart Bamberg  

**Link**: [PDF](https://arxiv.org/pdf/2506.01166)  

**Abstract**: Leveraging high degrees of unstructured sparsity is a promising approach to enhance the efficiency of deep neural network DNN accelerators - particularly important for emerging Edge-AI applications. We introduce VUSA, a systolic-array architecture that virtually grows based on the present sparsity to perform larger matrix multiplications with the same number of physical multiply-accumulate MAC units. The proposed architecture achieves saving by 37% and 68% in area and power efficiency, respectively, at the same peak-performance, compared to a baseline systolic array architecture in a commercial 16-nm technology. Still, the proposed architecture supports acceleration for any DNN with any sparsity - even no sparsity at all. Thus, the proposed architecture is application-independent, making it viable for general-purpose AI acceleration. 

**Abstract (ZH)**: 利用高程度的无序稀疏性提升深度神经网络DNN加速器的效率：特别是在新兴边缘AI应用中尤为重要。我们介绍了一种基于现稀疏性虚拟扩展的 systolic-array 架构VUSA，该架构能够在相同数量的物理乘累加MAC单元下进行更大的矩阵乘法运算。与商用16nm工艺下的基准 systolic array 架构相比，所提出的架构在相同峰值性能下，分别实现了37%和68%的面积和功率效率节省。此外，所提出的架构支持任何带稀疏性的DNN加速，即使完全不带稀疏性也不例外。因此，该架构具有应用程序独立性，适用于通用AI加速。 

---
# FORT: Forward-Only Regression Training of Normalizing Flows 

**Title (ZH)**: FORT: 前向Only回归训练归一化流 

**Authors**: Danyal Rehman, Oscar Davis, Jiarui Lu, Jian Tang, Michael Bronstein, Yoshua Bengio, Alexander Tong, Avishek Joey Bose  

**Link**: [PDF](https://arxiv.org/pdf/2506.01158)  

**Abstract**: Simulation-free training frameworks have been at the forefront of the generative modelling revolution in continuous spaces, leading to neural dynamical systems that encompass modern large-scale diffusion and flow matching models. Despite the scalability of training, the generation of high-quality samples and their corresponding likelihood under the model requires expensive numerical simulation -- inhibiting adoption in numerous scientific applications such as equilibrium sampling of molecular systems. In this paper, we revisit classical normalizing flows as one-step generative models with exact likelihoods and propose a novel, scalable training objective that does not require computing the expensive change of variable formula used in conventional maximum likelihood training. We propose Forward-Only Regression Training (FORT), a simple $\ell_2$-regression objective that maps prior samples under our flow to specifically chosen targets. We demonstrate that FORT supports a wide class of targets, such as optimal transport targets and targets from pre-trained continuous-time normalizing flows (CNF). We further demonstrate that by using CNF targets, our one-step flows allow for larger-scale training that exceeds the performance and stability of maximum likelihood training, while unlocking a broader class of architectures that were previously challenging to train. Empirically, we elucidate that our trained flows can perform equilibrium conformation sampling in Cartesian coordinates of alanine dipeptide, alanine tripeptide, and alanine tetrapeptide. 

**Abstract (ZH)**: 无模拟训练框架在连续空间生成模型革命中居于前沿，引领了现代大规模扩散和流动匹配模型的神经动力系统。尽管训练具有扩展性，但生成高质量样本及其在模型下的精确概率测量仍需昂贵的数值模拟，这限制了其在分子系统平衡采样等众多科学应用中的采用。在本文中，我们重新审视经典归一化流作为单步生成模型，并提出一种新的、可扩展的训练目标，该目标无需计算传统极大似然训练中所使用的昂贵变量替换公式。我们提出了前向回归训练（Forward-Only Regression Training，FORT），这是一种简单的$\ell_2$-回归目标，将我们的流下的先验样本映射到特定选择的目标。我们证明FORT支持广泛的目标类型，如最优传输目标和预训练连续时间归一化流（CNF）的目标。进一步地，我们证明通过使用CNF目标，我们的单步流可以实现比极大似然训练更大的训练规模，超越其性能和稳定性，并解锁了以前难以训练的一系列架构。实证研究表明，我们的训练流可以在Alanine二肽、三肽和四肽的笛卡尔坐标中执行平衡构象采样。 

---
# Speeding Up Hyper-Heuristics With Markov-Chain Operator Selection and the Only-Worsening Acceptance Operator 

**Title (ZH)**: 使用马尔科夫链操作选择和仅恶化接受操作加速超启发式算法 

**Authors**: Abderrahim Bendahi, Benjamin Doerr, Adrien Fradin, Johannes F. Lutzeyer  

**Link**: [PDF](https://arxiv.org/pdf/2506.01107)  

**Abstract**: The move-acceptance hyper-heuristic was recently shown to be able to leave local optima with astonishing efficiency (Lissovoi et al., Artificial Intelligence (2023)). In this work, we propose two modifications to this algorithm that demonstrate impressive performances on a large class of benchmarks including the classic Cliff$_d$ and Jump$_m$ function classes. (i) Instead of randomly choosing between the only-improving and any-move acceptance operator, we take this choice via a simple two-state Markov chain. This modification alone reduces the runtime on Jump$_m$ functions with gap parameter $m$ from $\Omega(n^{2m-1})$ to $O(n^{m+1})$. (ii) We then replace the all-moves acceptance operator with the operator that only accepts worsenings. Such a, counter-intuitive, operator has not been used before in the literature. However, our proofs show that our only-worsening operator can greatly help in leaving local optima, reducing, e.g., the runtime on Jump functions to $O(n^3 \log n)$ independent of the gap size. In general, we prove a remarkably good runtime of $O(n^{k+1} \log n)$ for our Markov move-acceptance hyper-heuristic on all members of a new benchmark class SEQOPT$_k$, which contains a large number of functions having $k$ successive local optima, and which contains the commonly studied Jump$_m$ and Cliff$_d$ functions for $k=2$. 

**Abstract (ZH)**: -move接受超启发式算法 recently 展示了其以惊人效率离开局部最优解的能力 (Lissovoi 等人, 人工智能 (2023))。在这项工作中，我们对该算法提出了两项修改，这些修改在包括经典 Cliff$_d$ 和 Jump$_m$ 函数类在内的大量基准测试中表现出色。(i) 与随机选择仅改进操作符和任何操作符的接受操作符不同，我们通过简单的两状态马尔可夫链来做出这种选择。这一修改将 Gap 参数为 $m$ 的 Jump$_m$ 函数的运行时从 $\Omega(n^{2m-1})$ 降低到 $O(n^{m+1})$。(ii) 然后，我们用只接受恶化操作符替代所有操作符的接受操作符。这种看似反直观的操作符还未曾在文献中使用过。然而，我们的证明表明，我们仅恶化操作符可以极大地帮助离开局部最优解，例如，将 Jump 函数的运行时减少到 $O(n^3 \log n)$，与 Gap 大小无关。一般来说，我们证明了对于新基准类 SEQOPT$_k$ 中的所有成员，我们的马尔可夫移动接受超启发式算法具有显著良好的运行时 $O(n^{k+1} \log n)$，该基准类包含许多具有 $k$ 个连续局部最优解的函数，并包含常见的 Jump$_m$ 和 Cliff$_d$ 函数（对于 $k=2$）。 

---
# Learning What Matters: Prioritized Concept Learning via Relative Error-driven Sample Selection 

**Title (ZH)**: 优先学习重要概念：基于相对误差驱动的样本选择概念学习 

**Authors**: Shivam Chandhok, Qian Yang, Oscar Manas, Kanishk Jain, Leonid Sigal, Aishwarya Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2506.01085)  

**Abstract**: Instruction tuning has been central to the success of recent vision-language models (VLMs), but it remains expensive-requiring large-scale datasets, high-quality annotations, and large compute budgets. We propose PRioritized cOncept learninG via Relative Error-driven Sample Selection (PROGRESS), a data- and compute-efficient framework that enables VLMs to dynamically select what to learn next based on their evolving needs during training. At each stage, the model tracks its learning progress across skills and selects the most informative samples-those it has not already mastered and that are not too difficult to learn at the current stage of training. This strategy effectively controls skill acquisition and the order in which skills are learned. Specifically, we sample from skills showing the highest learning progress, prioritizing those with the most rapid improvement. Unlike prior methods, PROGRESS requires no upfront answer annotations, queries answers only on a need basis, avoids reliance on additional supervision from auxiliary VLMs, and does not require compute-heavy gradient computations for data selection. Experiments across multiple instruction-tuning datasets of varying scales demonstrate that PROGRESS consistently outperforms state-of-the-art baselines with much less data and supervision. Additionally, we show strong cross-architecture generalization and transferability to larger models, validating PROGRESS as a scalable solution for efficient learning. 

**Abstract (ZH)**: PRrioritized cOncept lEarning via RElative EError-driven SAmple SElction (PRogress) 

---
# Unfolding Boxes with Local Constraints 

**Title (ZH)**: 带有局部约束的展开箱体问题 

**Authors**: Long Qian, Eric Wang, Bernardo Subercaseaux, Marijn J. H. Heule  

**Link**: [PDF](https://arxiv.org/pdf/2506.01079)  

**Abstract**: We consider the problem of finding and enumerating polyominos that can be folded into multiple non-isomorphic boxes. While several computational approaches have been proposed, including SAT, randomized algorithms, and decision diagrams, none has been able to perform at scale. We argue that existing SAT encodings are hindered by the presence of global constraints (e.g., graph connectivity or acyclicity), which are generally hard to encode effectively and hard for solvers to reason about. In this work, we propose a new SAT-based approach that replaces these global constraints with simple local constraints that have substantially better propagation properties. Our approach dramatically improves the scalability of both computing and enumerating common box unfoldings: (i) while previous approaches could only find common unfoldings of two boxes up to area 88, ours easily scales beyond 150, and (ii) while previous approaches were only able to enumerate common unfoldings up to area 30, ours scales up to 60. This allows us to rule out 46, 54, and 58 as the smallest areas allowing a common unfolding of three boxes, thereby refuting a conjecture of Xu et al. (2017). 

**Abstract (ZH)**: 我们考虑寻找和枚举可以折叠成多个非同构盒子的多米诺骨牌的问题。尽管已经提出了几种计算方法，包括SAT、随机化算法和决策图，但 none 未能在大规模应用中发挥作用。我们argue 存在的 SAT 编码受到全局约束（例如，图连通性或无环性）的阻碍，这些约束通常难以有效编码且难以供求解器推理。在本文中，我们提出了一种新的基于 SAT 的方法，用简单的局部约束替换这些全局约束，从而具有更好的传播性质。我们的方法显著提高了计算和枚举常见盒子展开图的可扩展性：(i) 而且前人方法只能找到两盒面积不超过 88 的常见展开图，我们的方法轻松扩展到 150 以上；(ii) 而且前人方法只能枚举面积不超过 30 的常见展开图，我们的方法扩展到 60。这使得我们可以排除 46、54 和 58 作为三个盒子共有展开图的最小面积，从而反驳了 Xu 等人 (2017) 的猜想。 

---
# Revolutionizing Blood Banks: AI-Driven Fingerprint-Blood Group Correlation for Enhanced Safety 

**Title (ZH)**: 革新血液银行：基于人工智能的指纹-血型关联技术以提高安全性能 

**Authors**: Malik A. Altayar, Muhyeeddin Alqaraleh, Mowafaq Salem Alzboon, Wesam T. Almagharbeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.01069)  

**Abstract**: Identification of a person is central in forensic science, security, and healthcare. Methods such as iris scanning and genomic profiling are more accurate but expensive, time-consuming, and more difficult to implement. This study focuses on the relationship between the fingerprint patterns and the ABO blood group as a biometric identification tool. A total of 200 subjects were included in the study, and fingerprint types (loops, whorls, and arches) and blood groups were compared. Associations were evaluated with statistical tests, including chi-square and Pearson correlation. The study found that the loops were the most common fingerprint pattern and the O+ blood group was the most prevalent. Even though there was some associative pattern, there was no statistically significant difference in the fingerprint patterns of different blood groups. Overall, the results indicate that blood group data do not significantly improve personal identification when used in conjunction with fingerprinting. Although the study shows weak correlation, it may emphasize the efforts of multi-modal based biometric systems in enhancing the current biometric systems. Future studies may focus on larger and more diverse samples, and possibly machine learning and additional biometrics to improve identification methods. This study addresses an element of the ever-changing nature of the fields of forensic science and biometric identification, highlighting the importance of resilient analytical methods for personal identification. 

**Abstract (ZH)**: 指纹模式与ABO血型在生物识别身份认证中的关系研究 

---
# Trilevel Memetic Algorithm for the Electric Vehicle Routing Problem 

**Title (ZH)**: 三级遗传算法求解电动汽车路由问题 

**Authors**: Ivan Milinović, Leon Stjepan Uroić, Marko Đurasević  

**Link**: [PDF](https://arxiv.org/pdf/2506.01065)  

**Abstract**: The Electric Vehicle Routing Problem (EVRP) extends the capacitated vehicle routing problem by incorporating battery constraints and charging stations, posing significant optimization challenges. This paper introduces a Trilevel Memetic Algorithm (TMA) that hierarchically optimizes customer sequences, route assignments, and charging station insertions. The method combines genetic algorithms with dynamic programming, ensuring efficient and high-quality solutions. Benchmark tests on WCCI2020 instances show competitive performance, matching best-known results for small-scale cases. While computational demands limit scalability, TMA demonstrates strong potential for sustainable logistics planning. 

**Abstract (ZH)**: 三层次 memetic 算法求解考虑电池约束的电动车辆路由问题 

---
# SealQA: Raising the Bar for Reasoning in Search-Augmented Language Models 

**Title (ZH)**: SealQA：提高基于搜索的语言模型推理标准 

**Authors**: Thinh Pham, Nguyen Nguyen, Pratibha Zunjare, Weiyuan Chen, Yu-Min Tseng, Tu Vu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01062)  

**Abstract**: We introduce SealQA, a new challenge benchmark for evaluating SEarch-Augmented Language models on fact-seeking questions where web search yields conflicting, noisy, or unhelpful results. SealQA comes in three flavors: (1) Seal-0 (main) and (2) Seal-Hard, which assess factual accuracy and reasoning capabilities, with Seal-0 focusing on the most challenging questions where chat models (e.g., GPT-4.1) typically achieve near-zero accuracy; and (3) LongSeal, which extends SealQA to test long-context, multi-document reasoning in "needle-in-a-haystack" settings. Our evaluation reveals critical limitations in current models: Even frontier LLMs perform poorly across all SealQA flavors. On Seal-0, frontier agentic models equipped with tools like o3 and o4-mini achieve only 17.1% and 6.3% accuracy, respectively, at their best reasoning efforts. We find that advanced reasoning models such as DeepSeek-R1-671B and o3-mini are highly vulnerable to noisy search results. Notably, increasing test-time compute does not yield reliable gains across o3-mini, o4-mini, and o3, with performance often plateauing or even declining early. Additionally, while recent models are less affected by the "lost-in-the-middle" issue, they still fail to reliably identify relevant documents in LongSeal when faced with numerous distractors. To facilitate future work, we release SealQA at this http URL. 

**Abstract (ZH)**: SealQA：一种新的挑战基准，用于评估基于网络搜索的语言模型在事实查询中的性能，特别是在搜索结果矛盾、噪音大或无用的情况下。 

---
# XAI-Units: Benchmarking Explainability Methods with Unit Tests 

**Title (ZH)**: XAI-Units：基于单元测试的可解释性方法基准测试 

**Authors**: Jun Rui Lee, Sadegh Emami, Michael David Hollins, Timothy C. H. Wong, Carlos Ignacio Villalobos Sánchez, Francesca Toni, Dekai Zhang, Adam Dejl  

**Link**: [PDF](https://arxiv.org/pdf/2506.01059)  

**Abstract**: Feature attribution (FA) methods are widely used in explainable AI (XAI) to help users understand how the inputs of a machine learning model contribute to its outputs. However, different FA models often provide disagreeing importance scores for the same model. In the absence of ground truth or in-depth knowledge about the inner workings of the model, it is often difficult to meaningfully determine which of the different FA methods produce more suitable explanations in different contexts. As a step towards addressing this issue, we introduce the open-source XAI-Units benchmark, specifically designed to evaluate FA methods against diverse types of model behaviours, such as feature interactions, cancellations, and discontinuous outputs. Our benchmark provides a set of paired datasets and models with known internal mechanisms, establishing clear expectations for desirable attribution scores. Accompanied by a suite of built-in evaluation metrics, XAI-Units streamlines systematic experimentation and reveals how FA methods perform against distinct, atomic kinds of model reasoning, similar to unit tests in software engineering. Crucially, by using procedurally generated models tied to synthetic datasets, we pave the way towards an objective and reliable comparison of FA methods. 

**Abstract (ZH)**: 开放源代码的XAI-Units基准：用于评估特征归因方法的多样性模型行为 

---
# A Two-Stage Hierarchical Deep Filtering Framework for Real-Time Speech Enhancement 

**Title (ZH)**: 两级分层深度滤波框架实现实时语音增强 

**Authors**: Shenghui Lu, Hukai Huang, Jinanglong Yao, Kaidi Wang, Qingyang Hong, Lin Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.01023)  

**Abstract**: This paper proposes a model that integrates sub-band processing and deep filtering to fully exploit information from the target time-frequency (TF) bin and its surrounding TF bins for single-channel speech enhancement. The sub-band module captures surrounding frequency bin information at the input, while the deep filtering module applies filtering at the output to both the target TF bin and its surrounding TF bins. To further improve the model performance, we decouple deep filtering into temporal and frequency components and introduce a two-stage framework, reducing the complexity of filter coefficient prediction at each stage. Additionally, we propose the TAConv module to strengthen convolutional feature extraction. Experimental results demonstrate that the proposed hierarchical deep filtering network (HDF-Net) effectively utilizes surrounding TF bin information and outperforms other advanced systems while using fewer resources. 

**Abstract (ZH)**: 本文提出了一种将子带处理与深度滤波相结合的模型，以充分利用目标时频(TF) bins及其周围TF bins中的信息，用于单通道语音增强。子带模块在输入端捕获周围的频率bins信息，而深度滤波模块在输出端对目标TF bin及其周围TF bins进行滤波。为了进一步提高模型性能，我们将深度滤波分解为时间域和频域组件，并引入两阶段框架，降低每阶段滤波系数预测的复杂度。此外，本文提出TAConv模块以增强卷积特征提取。实验结果表明，所提出的分层深度滤波网络(HDF-Net)有效地利用了周围TF bins中的信息，并在资源较少的情况下优于其他先进的系统。 

---
# Quotient Network - A Network Similar to ResNet but Learning Quotients 

**Title (ZH)**: 商网络 - 一种类似于ResNet的网络，学习商值 

**Authors**: Peng Hui, Jiamuyang Zhao, Changxin Li, Qingzhen Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00992)  

**Abstract**: The emergence of ResNet provides a powerful tool for training extremely deep networks. The core idea behind it is to change the learning goals of the network. It no longer learns new features from scratch but learns the difference between the target and existing features. However, the difference between the two kinds of features does not have an independent and clear meaning, and the amount of learning is based on the absolute rather than the relative difference, which is sensitive to the size of existing features. We propose a new network that perfectly solves these two problems while still having the advantages of ResNet. Specifically, it chooses to learn the quotient of the target features with the existing features, so we call it the quotient network. In order to enable this network to learn successfully and achieve higher performance, we propose some design rules for this network so that it can be trained efficiently and achieve better performance than ResNet. Experiments on the CIFAR10, CIFAR100, and SVHN datasets prove that this network can stably achieve considerable improvements over ResNet by simply making tiny corresponding changes to the original ResNet network without adding new parameters. 

**Abstract (ZH)**: ResNet的出现为训练极深网络提供了强大工具。其核心理念是改变网络的学习目标。它不再从零开始学习新的特征，而是学习目标与现有特征之间的差异。然而，这两种特征之间的差异缺乏独立和明确的意义，并且学习量基于绝对差异而非相对差异，这使其对现有特征的大小高度敏感。我们提出了一种新网络，完美解决了这两个问题的同时保持了ResNet的优点。具体地，它选择学习目标特征与现有特征的比值，因此我们称其为商网络。为了使该网络能够成功学习并实现更高的性能，我们为此网络提出了一些设计规则，以便它可以高效地训练并在性能上超越ResNet。实验表明，仅对原始ResNet网络进行微小的相应修改即可在CIFAR10、CIFAR100和SVHN数据集上稳定地获得显著改进，而无需增加新的参数。 

---
# Bridging the Gap: From Ad-hoc to Proactive Search in Conversations 

**Title (ZH)**: 弥补差距：从临时搜索到主动搜索在对话中的应用 

**Authors**: Chuan Meng, Francesco Tonolini, Fengran Mo, Nikolaos Aletras, Emine Yilmaz, Gabriella Kazai  

**Link**: [PDF](https://arxiv.org/pdf/2506.00983)  

**Abstract**: Proactive search in conversations (PSC) aims to reduce user effort in formulating explicit queries by proactively retrieving useful relevant information given conversational context. Previous work in PSC either directly uses this context as input to off-the-shelf ad-hoc retrievers or further fine-tunes them on PSC data. However, ad-hoc retrievers are pre-trained on short and concise queries, while the PSC input is longer and noisier. This input mismatch between ad-hoc search and PSC limits retrieval quality. While fine-tuning on PSC data helps, its benefits remain constrained by this input gap. In this work, we propose Conv2Query, a novel conversation-to-query framework that adapts ad-hoc retrievers to PSC by bridging the input gap between ad-hoc search and PSC. Conv2Query maps conversational context into ad-hoc queries, which can either be used as input for off-the-shelf ad-hoc retrievers or for further fine-tuning on PSC data. Extensive experiments on two PSC datasets show that Conv2Query significantly improves ad-hoc retrievers' performance, both when used directly and after fine-tuning on PSC. 

**Abstract (ZH)**: 主动对话检索（PSC）旨在通过在会话背景下主动检索有用的相关信息来减少用户的查询构架努力。先前的PSC工作要么直接将此背景作为输入传递给现成的即席检索器，要么在PSC数据上进一步微调它们。然而，即席检索器是基于短且简洁的查询进行预训练的，而PSC输入更长且更具噪声。这种即席搜索与PSC之间的输入不匹配限制了检索质量。虽然在PSC数据上进行微调有所帮助，但其效益仍受限于这种输入差距。在本文中，我们提出了一种新的Conv2Query框架，通过弥合即席搜索与PSC之间的输入差距，使即席检索器适应PSC。Conv2Query将会话背景映射为即席查询，这些查询可以作为现成的即席检索器的输入，或用于在PSC数据上进一步微调。在两个PSC数据集上的广泛实验表明，Conv2Query显着提高了即席检索器的性能，无论是否在PSC数据上进行进一步微调。 

---
# What do self-supervised speech models know about Dutch? Analyzing advantages of language-specific pre-training 

**Title (ZH)**: 自我监督语音模型对荷兰语了解多少？分析语言特定预训练的优势 

**Authors**: Marianne de Heer Kloots, Hosein Mohebbi, Charlotte Pouw, Gaofei Shen, Willem Zuidema, Martijn Bentum  

**Link**: [PDF](https://arxiv.org/pdf/2506.00981)  

**Abstract**: How language-specific are speech representations learned by self-supervised models? Existing work has shown that a range of linguistic features can be successfully decoded from end-to-end models trained only on speech recordings. However, it's less clear to what extent pre-training on specific languages improves language-specific linguistic information. Here we test the encoding of Dutch phonetic and lexical information in internal representations of self-supervised Wav2Vec2 models. Pre-training exclusively on Dutch improves the representation of Dutch linguistic features as compared to pre-training on similar amounts of English or larger amounts of multilingual data. This language-specific advantage is well-detected by trained clustering or classification probes, and partially observable using zero-shot metrics. Furthermore, the language-specific benefit on linguistic feature encoding aligns with downstream performance on Automatic Speech Recognition. 

**Abstract (ZH)**: 自监督Wav2Vec2模型中学习到的语音表示的语言特异性程度如何？预训练于特定语言在多大程度上提升了语言特异性语言信息？预训练于荷兰语比预训练于相似量的英语或更大规模的多语言数据更能捕获荷兰语音素和词汇信息。这种语言特异性优势通过训练聚类或分类探测器能够很好地检测到，并部分通过零样本指标观测到。此外，语言特异性优势与自动语音识别的下游性能一致。 

---
# Data Heterogeneity Modeling for Trustworthy Machine Learning 

**Title (ZH)**: 数据异质性建模以实现可信赖机器学习 

**Authors**: Jiashuo Liu, Peng Cui  

**Link**: [PDF](https://arxiv.org/pdf/2506.00969)  

**Abstract**: Data heterogeneity plays a pivotal role in determining the performance of machine learning (ML) systems. Traditional algorithms, which are typically designed to optimize average performance, often overlook the intrinsic diversity within datasets. This oversight can lead to a myriad of issues, including unreliable decision-making, inadequate generalization across different domains, unfair outcomes, and false scientific inferences. Hence, a nuanced approach to modeling data heterogeneity is essential for the development of dependable, data-driven systems. In this survey paper, we present a thorough exploration of heterogeneity-aware machine learning, a paradigm that systematically integrates considerations of data heterogeneity throughout the entire ML pipeline -- from data collection and model training to model evaluation and deployment. By applying this approach to a variety of critical fields, including healthcare, agriculture, finance, and recommendation systems, we demonstrate the substantial benefits and potential of heterogeneity-aware ML. These applications underscore how a deeper understanding of data diversity can enhance model robustness, fairness, and reliability and help model diagnosis and improvements. Moreover, we delve into future directions and provide research opportunities for the whole data mining community, aiming to promote the development of heterogeneity-aware ML. 

**Abstract (ZH)**: 数据异质性在确定机器学习系统性能中起着关键作用。传统的算法通常设计用于优化平均性能，往往忽略了数据集内的内在多样性。这种忽略可能导致决策可靠性差、跨不同领域的一般化不足、不公平结果和虚假的科学推断。因此，数据异质性建模的细致方法对于开发可靠的、数据驱动的系统至关重要。在本文综述中，我们全面探讨了数据异质性aware机器学习这一范式，该范式在整个机器学习管线上系统地整合了数据异质性的考虑——从数据收集和模型训练到模型评估和部署。通过将这种方法应用于医疗保健、农业、金融和推荐系统等多个关键领域，我们展示了数据异质性aware机器学习的巨大优势和潜力。这些应用强调了对数据多样性更深入理解如何增强模型的健壮性、公平性和可靠性，以及帮助模型诊断和改进。此外，我们探讨了未来的研究方向，并为整个数据挖掘社区提供了研究机会，旨在促进数据异质性aware机器学习的发展。 

---
# Uncertainty-Aware Metabolic Stability Prediction with Dual-View Contrastive Learning 

**Title (ZH)**: 具有双视图对比学习的不确定性意识代谢稳定性预测 

**Authors**: Peijin Guo, Minghui Li, Hewen Pan, Bowen Chen, Yang Wu, Zikang Guo, Leo Yu Zhang, Shengshan Hu, Shengqing Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00936)  

**Abstract**: Accurate prediction of molecular metabolic stability (MS) is critical for drug research and development but remains challenging due to the complex interplay of molecular interactions. Despite recent advances in graph neural networks (GNNs) for MS prediction, current approaches face two critical limitations: (1) incomplete molecular modeling due to atom-centric message-passing mechanisms that disregard bond-level topological features, and (2) prediction frameworks that lack reliable uncertainty quantification. To address these challenges, we propose TrustworthyMS, a novel contrastive learning framework designed for uncertainty-aware metabolic stability prediction. First, a molecular graph topology remapping mechanism synchronizes atom-bond interactions through edge-induced feature propagation, capturing both localized electronic effects and global conformational constraints. Second, contrastive topology-bond alignment enforces consistency between molecular topology views and bond patterns via feature alignment, enhancing representation robustness. Third, uncertainty modeling through Beta-Binomial uncertainty quantification enables simultaneous prediction and confidence calibration under epistemic uncertainty. Through extensive experiments, our results demonstrate that TrustworthyMS outperforms current state-of-the-art methods in terms of predictive performance. 

**Abstract (ZH)**: 可信的分子代谢稳定性预测（TrustworthyMS）：一种不确定性意识下的新颖对比学习框架 

---
# General-purpose audio representation learning for real-world sound scenes 

**Title (ZH)**: 通用音频表示学习以应对现实场景声音 

**Authors**: Goksenin Yuksel, Marcel van Gerven, Kiki van der Heijden  

**Link**: [PDF](https://arxiv.org/pdf/2506.00934)  

**Abstract**: While audio foundation models perform well on myriad of tasks from sound classification to speech analysis, these models are trained and tested on dry, non-spatial, single-source audio clips. This limits their success in real-world situations and results in spatially unaware audio embeddings. To address these limitations, we propose a novel self-supervised training approach for General-Purpose, Real-world Audio Models (GRAMs). The GRAM training approach enables robust spatial audio representation learning for naturalistic, noisy sound scenes and can be applied to any masking-based deep learning model. We demonstrate the success of our approach by training two state-of-the-art models, one with a transformer and one with a mamba backbone. We assess the quality of the extracted audio representations from GRAMs using the original version of the HEAR benchmark, a newly synthesized, naturalistic version of the HEAR benchmark, and novel sound localization tasks based on HEAR benchmark datasets. The results show that our approach minimizes the performance gap between dry, non-spatial, single-source sound scenes and naturalistic sound scenes for crucial tasks such as auditory scene analysis, outperforming existing state-of-the-art audio foundation models at a fraction of the training steps. Moreover, GRAMs show state-of-the-art performance on sound localization tasks, exceeding even supervised sound localization models. In sum, the proposed approach represents a significant advancement towards robust audio foundation models for real-world applications with state-of-the-art performance on naturalistic sound scenes as well as spatial audio representation learning. 

**Abstract (ZH)**: 面向现实场景的通用音频模型的自监督训练方法 

---
# In-the-wild Audio Spatialization with Flexible Text-guided Localization 

**Title (ZH)**: 户外音频空间化与灵活的文本导向定位 

**Authors**: Tianrui Pan, Jie Liu, Zewen Huang, Jie Tang, Gangshan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00927)  

**Abstract**: To enhance immersive experiences, binaural audio offers spatial awareness of sounding objects in AR, VR, and embodied AI applications. While existing audio spatialization methods can generally map any available monaural audio to binaural audio signals, they often lack the flexible and interactive control needed in complex multi-object user-interactive environments. To address this, we propose a Text-guided Audio Spatialization (TAS) framework that utilizes flexible text prompts and evaluates our model from unified generation and comprehension perspectives. Due to the limited availability of premium and large-scale stereo data, we construct the SpatialTAS dataset, which encompasses 376,000 simulated binaural audio samples to facilitate the training of our model. Our model learns binaural differences guided by 3D spatial location and relative position prompts, augmented by flipped-channel audio. It outperforms existing methods on both simulated and real-recorded datasets, demonstrating superior generalization and accuracy. Besides, we develop an assessment model based on Llama-3.1-8B, which evaluates the spatial semantic coherence between our generated binaural audio and text prompts through a spatial reasoning task. Results demonstrate that text prompts provide flexible and interactive control to generate binaural audio with excellent quality and semantic consistency in spatial locations. Dataset is available at \href{this https URL} 

**Abstract (ZH)**: 为了增强沉浸体验，双向音频在AR、VR和具身AI应用中提供了声源的三维空间感知。由于现有音频空间化方法通常可以将任何可用的单声道音频映射为双向音频信号，但在复杂多对象用户交互环境中往往缺乏灵活的互动控制。为此，我们提出了一种文本引导的音频空间化（TAS）框架，该框架利用灵活的文本提示，从统一生成和理解的角度评估我们的模型。由于高质量和大规模立体声数据的有限可用性，我们构建了SpatialTAS数据集，包含376,000个模拟的双向音频样本，以促进我们模型的训练。我们的模型通过三维空间位置和相对位置提示学习双向差异，并辅以翻转通道音频。该模型在模拟和实际录音数据集上均优于现有方法，展现出更好的泛化能力和准确性。此外，我们基于Llama-3.1-8B开发了一种评估模型，通过空间推理任务评估我们生成的双向音频与文本提示之间的空间语义一致性。结果表明，文本提示能够在空间位置上提供灵活的互动控制，生成高质量且语义一致的双向音频。数据集可通过\href{this https URL}获得。 

---
# Position as Probability: Self-Supervised Transformers that Think Past Their Training for Length Extrapolation 

**Title (ZH)**: 位置即概率：超越训练长度进行外推的自监督变压器模型 

**Authors**: Philip Heejun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.00920)  

**Abstract**: Deep sequence models typically degrade in accuracy when test sequences significantly exceed their training lengths, yet many critical tasks--such as algorithmic reasoning, multi-step arithmetic, and compositional generalization--require robust length extrapolation. We introduce PRISM, a Probabilistic Relative-position Implicit Superposition Model, a novel positional encoding mechanism that enables Transformers to extrapolate accurately up to 10x beyond their training length. PRISM learns continuous relative positions through a differentiable histogram-filter update, preserving position uncertainty via a probabilistic superposition rather than conventional deterministic embeddings. Empirically, PRISM achieves state-of-the-art length extrapolation, successfully generalizing to previously intractable sequence lengths across algorithmic benchmarks--including arithmetic (addition, multiplication), SCAN compositionality tasks, and complex copy variants derived from DeepMind's recent datasets. Our analysis demonstrates that PRISM's stochastic positional encoding maintains sharp and interpretable internal states, providing a theoretical basis for reliable length generalization. These results advance the goal of neural sequence models that remain algorithmically robust at lengths far exceeding their training horizon. 

**Abstract (ZH)**: Probabilistic Relative-position Implicit Superposition Model for Robust Length Extrapolation 

---
# Principled Input-Output-Conditioned Post-Hoc Uncertainty Estimation for Regression Networks 

**Title (ZH)**: principled 输入-输出-条件后验不确定性估计对于回归网络 

**Authors**: Lennart Bramlage, Cristóbal Curio  

**Link**: [PDF](https://arxiv.org/pdf/2506.00918)  

**Abstract**: Uncertainty quantification is critical in safety-sensitive applications but is often omitted from off-the-shelf neural networks due to adverse effects on predictive performance. Retrofitting uncertainty estimates post-hoc typically requires access to model parameters or gradients, limiting feasibility in practice. We propose a theoretically grounded framework for post-hoc uncertainty estimation in regression tasks by fitting an auxiliary model to both original inputs and frozen model outputs. Drawing from principles of maximum likelihood estimation and sequential parameter fitting, we formalize an exact post-hoc optimization objective that recovers the canonical MLE of Gaussian parameters, without requiring sampling or approximation at inference. While prior work has used model outputs to estimate uncertainty, we explicitly characterize the conditions under which this is valid and demonstrate the extent to which structured outputs can support quasi-epistemic inference. We find that using diverse auxiliary data, such as augmented subsets of the original training data, significantly enhances OOD detection and metric performance. Our hypothesis that frozen model outputs contain generalizable latent information about model error and predictive uncertainty is tested and confirmed. Finally, we ensure that our method maintains proper estimation of input-dependent uncertainty without relying exclusively on base model forecasts. These findings are demonstrated in toy problems and adapted to both UCI and depth regression benchmarks. Code: this https URL. 

**Abstract (ZH)**: 基于回归任务的后验不确定性量化：一个理论上支持的框架 

---
# Pi-SQL: Enhancing Text-to-SQL with Fine-Grained Guidance from Pivot Programming Languages 

**Title (ZH)**: Pi-SQL：来自pivot编程语言的细粒度指导以增强文本到SQL的转换 

**Authors**: Yongdong chi, Hanqing Wang, Zonghan Yang, Jian Yang, Xiao Yan, Yun Chen, Guanhua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00912)  

**Abstract**: Text-to-SQL transforms the user queries from natural language to executable SQL programs, enabling non-experts to interact with complex databases. Existing prompt-based methods craft meticulous text guidelines and examples to facilitate SQL generation, but their accuracy is hindered by the large semantic gap between the texts and the low-resource SQL programs. In this work, we propose Pi-SQL, which incorporates the high-resource Python program as a pivot to bridge between the natural language query and SQL program. In particular, Pi-SQL first generates Python programs that provide fine-grained step-by-step guidelines in their code blocks or comments, and then produces an SQL program following the guidance of each Python this http URL final SQL program matches the reference Python program's query results and, through selection from candidates generated by different strategies, achieves superior execution speed, with a reward-based valid efficiency score up to 4.55 higher than the best-performing this http URL experiments demonstrate the effectiveness of Pi-SQL, which improves the execution accuracy of the best-performing baseline by up to 3.20. 

**Abstract (ZH)**: Text-to-SQL将用户查询从自然语言转换为可执行的SQL程序，使非专家能够与复杂数据库进行交互。现有的基于提示的方法通过精心制作的文本指南和示例来促进SQL生成，但它们的准确性受到了文本与低资源SQL程序之间庞大语义差距的阻碍。在本文中，我们提出了Pi-SQL，它将高资源的Python程序作为枢纽，连接自然语言查询和SQL程序。特别是，Pi-SQL首先生成Python程序，这些程序在其代码块或注释中提供详尽的逐步指南，然后根据每个Python程序的指导生成SQL程序。最终生成的SQL程序与参考Python程序的查询结果匹配，并通过从不同策略生成的候选者中选择，实现了比最佳性能基线高出4.55的奖励驱动的有效效率得分。实验表明，Pi-SQL的有效性可以将最佳性能基线的执行准确性提高3.20。 

---
# PCoreSet: Effective Active Learning through Knowledge Distillation from Vision-Language Models 

**Title (ZH)**: PCoreSet: 通过来自视觉-语言模型的知识蒸馏实现有效的主动学习 

**Authors**: Seongjae Kang, Dong Bok Lee, Hyungjoon Jang, Dongseop Kim, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00910)  

**Abstract**: Knowledge distillation (KD) is a widely used framework for training compact, task-specific models by leveraging the knowledge of teacher models. However, its application to active learning (AL), which aims to minimize annotation costs through iterative sample selection, remains underexplored. This gap stems from the fact that KD typically assumes access to sufficient labeled data, whereas AL operates in data-scarce scenarios where task-specific teacher models are often unavailable. In this paper, we introduce ActiveKD, a framework that integrates AL with KD by leveraging the zero- and few-shot capabilities of large vision-language models (VLMs). A key aspect of ActiveKD is the structured prediction bias of VLMs--i.e., their predictions form clusters in the probability space. We regard this structure as an inductive bias of the teacher model, capturing generalizable output patterns beneficial to student learning. To exploit this bias, we propose Probabilistic CoreSet (PCoreSet), a selection strategy that maximizes coverage in the probability space rather than the feature space. PCoreSet strategically selects categorically diverse unlabeled samples, facilitating more efficient transfer of teacher knowledge under limited annotation budgets. Evaluations on 11 datasets show that PCoreSet consistently outperforms existing selection methods within the ActiveKD framework, advancing research at the intersection of AL and KD. 

**Abstract (ZH)**: 知识蒸馏（KD）是一种通过利用教师模型的知识来训练紧凑的任务专用模型的广泛使用的框架。然而，将其应用于主动学习（AL），即通过迭代样本选择来最小化注释成本的方法，仍然鲜有探索。这一空白源于KD通常假设可以访问充足标记数据的事实，而AL则在数据稀缺的情景中运行，其中往往缺少任务专用的教师模型。本文介绍了ActiveKD框架，该框架通过利用大规模视觉-语言模型（VLMs）的零样本和极少样本能力，将AL与KD相结合。ActiveKD的一个关键方面是VLMs的结构化预测偏差，即它们的预测在概率空间中形成簇。我们将这种结构视为教师模型的归纳偏置，捕捉对学生学习有益的可泛化的输出模式。为了利用这一偏差，我们提出了一种概率核心集（PCoreSet）的选择策略，该策略在概率空间中最大化覆盖范围而不是特征空间。PCoreSet有选择地挑选类别多样性的未标记样本，有利于在有限注释预算下更高效地转移教师知识。在11个数据集上的评估表明，PCoreSet在ActiveKD框架内的选择方法中表现最优，促进了AL和KD交叉领域的研究进展。 

---
# CoVoMix2: Advancing Zero-Shot Dialogue Generation with Fully Non-Autoregressive Flow Matching 

**Title (ZH)**: CoVoMix2: 采用全非自回归流匹配促进零样本对话生成 

**Authors**: Leying Zhang, Yao Qian, Xiaofei Wang, Manthan Thakker, Dongmei Wang, Jianwei Yu, Haibin Wu, Yuxuan Hu, Jinyu Li, Yanmin Qian, Sheng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00885)  

**Abstract**: Generating natural-sounding, multi-speaker dialogue is crucial for applications such as podcast creation, virtual agents, and multimedia content generation. However, existing systems struggle to maintain speaker consistency, model overlapping speech, and synthesize coherent conversations efficiently. In this paper, we introduce CoVoMix2, a fully non-autoregressive framework for zero-shot multi-talker dialogue generation. CoVoMix2 directly predicts mel-spectrograms from multi-stream transcriptions using a flow-matching-based generative model, eliminating the reliance on intermediate token representations. To better capture realistic conversational dynamics, we propose transcription-level speaker disentanglement, sentence-level alignment, and prompt-level random masking strategies. Our approach achieves state-of-the-art performance, outperforming strong baselines like MoonCast and Sesame in speech quality, speaker consistency, and inference speed. Notably, CoVoMix2 operates without requiring transcriptions for the prompt and supports controllable dialogue generation, including overlapping speech and precise timing control, demonstrating strong generalizability to real-world speech generation scenarios. 

**Abstract (ZH)**: 零样本多说话人对话生成的CoVoMix2全非自回归框架 

---
# Local Manifold Approximation and Projection for Manifold-Aware Diffusion Planning 

**Title (ZH)**: 局部流形逼近与投影在流形感知扩散规划中的应用 

**Authors**: Kyowoon Lee, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00867)  

**Abstract**: Recent advances in diffusion-based generative modeling have demonstrated significant promise in tackling long-horizon, sparse-reward tasks by leveraging offline datasets. While these approaches have achieved promising results, their reliability remains inconsistent due to the inherent stochastic risk of producing infeasible trajectories, limiting their applicability in safety-critical applications. We identify that the primary cause of these failures is inaccurate guidance during the sampling procedure, and demonstrate the existence of manifold deviation by deriving a lower bound on the guidance gap. To address this challenge, we propose Local Manifold Approximation and Projection (LoMAP), a training-free method that projects the guided sample onto a low-rank subspace approximated from offline datasets, preventing infeasible trajectory generation. We validate our approach on standard offline reinforcement learning benchmarks that involve challenging long-horizon planning. Furthermore, we show that, as a standalone module, LoMAP can be incorporated into the hierarchical diffusion planner, providing further performance enhancements. 

**Abstract (ZH)**: 基于扩散的生成建模的最近进展在利用离线数据集处理长期、稀疏奖励任务方面展示了显著的潜力。然而，这些方法的可靠性因固有的生产不可行轨迹的随机风险而不稳定，限制了它们在关键安全应用中的适用性。我们发现这些失败的主要原因是采样过程中指导的不准确，并通过推导指导间隙的下界证明了 manifold 偏差的存在。为应对这一挑战，我们提出了局部 manifold 近似和投影（LoMAP）方法，这是一种无需训练的方法，将指导的样本投影到由离线数据集近似得到的低秩子空间上，防止生成不可行轨迹。我们在涉及挑战性长期规划的基准离线强化学习任务上验证了该方法。此外，我们展示在作为独立模块的情况下，LoMAP 可以集成到层级扩散规划器中，提供进一步的性能提升。 

---
# Can AI Master Econometrics? Evidence from Econometrics AI Agent on Expert-Level Tasks 

**Title (ZH)**: AI能掌握计量经济学吗？从 Econometrics AI 代理在专家级任务中的表现看起 

**Authors**: Qiang Chen, Tianyang Han, Jin Li, Ye Luo, Yuxiao Wu, Xiaowei Zhang, Tuo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.00856)  

**Abstract**: Can AI effectively perform complex econometric analysis traditionally requiring human expertise? This paper evaluates an agentic AI's capability to master econometrics, focusing on empirical analysis performance. We develop an ``Econometrics AI Agent'' built on the open-source MetaGPT framework. This agent exhibits outstanding performance in: (1) planning econometric tasks strategically, (2) generating and executing code, (3) employing error-based reflection for improved robustness, and (4) allowing iterative refinement through multi-round conversations. We construct two datasets from academic coursework materials and published research papers to evaluate performance against real-world challenges. Comparative testing shows our domain-specialized agent significantly outperforms both benchmark large language models (LLMs) and general-purpose AI agents. This work establishes a testbed for exploring AI's impact on social science research and enables cost-effective integration of domain expertise, making advanced econometric methods accessible to users with minimal coding expertise. Furthermore, our agent enhances research reproducibility and offers promising pedagogical applications for econometrics teaching. 

**Abstract (ZH)**: AI能否有效地执行传统上需要人类专长的复杂计量经济学分析？本文评估了一个代理AI掌握计量经济学的能力，重点在于其实证分析性能。我们基于开源MetaGPT框架开发了一个“计量经济学AI代理”。该代理在以下方面表现出色：（1）战略性规划计量经济学任务，（2）生成和执行代码，（3）利用基于误差的反思以提高稳健性，以及（4）通过多轮对话实现迭代优化。我们从学术课程材料和已发表的研究论文中构建了两个数据集，以评估其在现实世界挑战中的性能。比较测试结果显示，我们的领域专业化代理显著优于基准大语言模型（LLMs）和通用AI代理。本文建立了一个测试床，用于探索AI对社会科学研究的影响，并使高级计量经济学方法能够以低成本集成领域专业知识，从而无需大量编程知识即可供用户使用。此外，我们的代理增强了研究的可再现性，并为计量经济学教学提供了有前景的教学应用。 

---
# Generalization in VAE and Diffusion Models: A Unified Information-Theoretic Analysis 

**Title (ZH)**: VAE和扩散模型中的泛化性：一种统一的信息论分析 

**Authors**: Qi Chen, Jierui Zhu, Florian Shkurti  

**Link**: [PDF](https://arxiv.org/pdf/2506.00849)  

**Abstract**: Despite the empirical success of Diffusion Models (DMs) and Variational Autoencoders (VAEs), their generalization performance remains theoretically underexplored, especially lacking a full consideration of the shared encoder-generator structure. Leveraging recent information-theoretic tools, we propose a unified theoretical framework that provides guarantees for the generalization of both the encoder and generator by treating them as randomized mappings. This framework further enables (1) a refined analysis for VAEs, accounting for the generator's generalization, which was previously overlooked; (2) illustrating an explicit trade-off in generalization terms for DMs that depends on the diffusion time $T$; and (3) providing computable bounds for DMs based solely on the training data, allowing the selection of the optimal $T$ and the integration of such bounds into the optimization process to improve model performance. Empirical results on both synthetic and real datasets illustrate the validity of the proposed theory. 

**Abstract (ZH)**: 尽管扩散模型（DMs）和变分自编码器（VAEs）在实证上取得了成功，但它们的泛化性能在理论上尚未得到充分探索，特别是缺乏对共享编码-生成器结构的全面考虑。利用最新的信息论工具，我们提出了一种统一的理论框架，通过将编码器和生成器视为随机映射来为它们的泛化提供保障。该框架还进一步实现了以下能力：（1）对VAEs进行了细化分析，考虑了生成器的泛化问题，这是之前未曾关注的；（2）展示了DMs在泛化方面的显式权衡关系，该关系取决于扩散时间$T$；（3）基于训练数据提供了可计算的DMs边界，允许选择最优的$T$值，并将此类边界整合到优化过程中以提高模型性能。在合成数据集和真实数据集上的实验结果证明了所提理论的有效性。 

---
# Speech Unlearning 

**Title (ZH)**: 语音遗忘 

**Authors**: Jiali Cheng, Hadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2506.00848)  

**Abstract**: We introduce machine unlearning for speech tasks, a novel and underexplored research problem that aims to efficiently and effectively remove the influence of specific data from trained speech models without full retraining. This has important applications in privacy preservation, removal of outdated or noisy data, and bias mitigation. While machine unlearning has been studied in computer vision and natural language processing, its application to speech is largely unexplored due to the high-dimensional, sequential, and speaker-dependent nature of speech data. We define two fundamental speech unlearning tasks: sample unlearning, which removes individual data points (e.g., a voice recording), and class unlearning, which removes an entire category (e.g., all data from a speaker), while preserving performance on the remaining data. Experiments on keyword spotting and speaker identification demonstrate that unlearning speech data is significantly more challenging than unlearning image or text data. We conclude with key future directions in this area, including structured training, robust evaluation, feature-level unlearning, broader applications, scalable methods, and adversarial robustness. 

**Abstract (ZH)**: 我们介绍了语音任务中的机器遗忘问题，这是一个新颖且尚未充分探索的研究课题，旨在无需完全重新训练的情况下，高效有效地从训练好的语音模型中移除特定数据的影响。这一课题在隐私保护、去除过时或噪声数据以及偏见缓解方面具有重要应用价值。尽管机器遗忘已经在计算机视觉和自然语言处理中进行了研究，但由于语音数据的高度维度性、序列依赖性和说话人依赖性，其在语音领域的应用尚未得到充分探索。我们定义了两种基本的语音遗忘任务：样本遗忘，即移除单个数据点（例如，一段语音记录），类别遗忘，即移除整个类别（例如，某说话人所有数据），同时保持对剩余数据性能的影响。关键词摘录和说话人识别实验表明，遗忘语音数据比遗忘图像或文本数据更具挑战性。最后，我们提出了该领域未来发展方向，包括结构化训练、鲁棒评估、特征级遗忘、更广泛的应用、可扩展方法和对抗鲁棒性。 

---
# HERGC: Heterogeneous Experts Representation and Generative Completion for Multimodal Knowledge Graphs 

**Title (ZH)**: HERGC: 异构专家表示与生成性完成的多模态知识图谱 

**Authors**: Yongkang Xiao, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00826)  

**Abstract**: Multimodal knowledge graphs (MMKGs) enrich traditional knowledge graphs (KGs) by incorporating diverse modalities such as images and text. Multi-modal knowledge graph completion (MMKGC) seeks to exploit these heterogeneous signals to infer missing facts, thereby mitigating the intrinsic incompleteness of MMKGs. Existing MMKGC methods typically leverage only the information contained in the MMKGs under the closed-world assumption and adopt discriminative training objectives, which limits their reasoning capacity during completion. Recent generative completion approaches powered by advanced large language models (LLMs) have shown strong reasoning abilities in unimodal knowledge graph completion, but their potential in MMKGC remains largely unexplored. To bridge this gap, we propose HERGC, a Heterogeneous Experts Representation and Generative Completion framework for MMKGs. HERGC first deploys a Heterogeneous Experts Representation Retriever that enriches and fuses multimodal information and retrieves a compact candidate set for each incomplete triple. It then uses a Generative LLM Predictor fine-tuned on minimal instruction data to accurately identify the correct answer from these candidates. Extensive experiments on three standard MMKG benchmarks demonstrate HERGC's effectiveness and robustness, achieving state-of-the-art performance. 

**Abstract (ZH)**: 多模态知识图谱的异构专家表示与生成式补全框架（HERGC） 

---
# SafeGenes: Evaluating the Adversarial Robustness of Genomic Foundation Models 

**Title (ZH)**: SafeGenes: 评估基因组基础模型的对抗鲁棒性 

**Authors**: Huixin Zhan, Jason H. Moore  

**Link**: [PDF](https://arxiv.org/pdf/2506.00821)  

**Abstract**: Genomic Foundation Models (GFMs), such as Evolutionary Scale Modeling (ESM), have demonstrated significant success in variant effect prediction. However, their adversarial robustness remains largely unexplored. To address this gap, we propose SafeGenes: a framework for Secure analysis of genomic foundation models, leveraging adversarial attacks to evaluate robustness against both engineered near-identical adversarial Genes and embedding-space manipulations. In this study, we assess the adversarial vulnerabilities of GFMs using two approaches: the Fast Gradient Sign Method (FGSM) and a soft prompt attack. FGSM introduces minimal perturbations to input sequences, while the soft prompt attack optimizes continuous embeddings to manipulate model predictions without modifying the input tokens. By combining these techniques, SafeGenes provides a comprehensive assessment of GFM susceptibility to adversarial manipulation. Targeted soft prompt attacks led to substantial performance degradation, even in large models such as ESM1b and ESM1v. These findings expose critical vulnerabilities in current foundation models, opening new research directions toward improving their security and robustness in high-stakes genomic applications such as variant effect prediction. 

**Abstract (ZH)**: SafeGenes:一种利用对抗攻击评估基因基础模型安全性与鲁棒性的框架 

---
# L3A: Label-Augmented Analytic Adaptation for Multi-Label Class Incremental Learning 

**Title (ZH)**: L3A：带有标签增强的分析适应性多标签类别增量学习 

**Authors**: Xiang Zhang, Run He, Jiao Chen, Di Fang, Ming Li, Ziqian Zeng, Cen Chen, Huiping Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00816)  

**Abstract**: Class-incremental learning (CIL) enables models to learn new classes continually without forgetting previously acquired knowledge. Multi-label CIL (MLCIL) extends CIL to a real-world scenario where each sample may belong to multiple classes, introducing several challenges: label absence, which leads to incomplete historical information due to missing labels, and class imbalance, which results in the model bias toward majority classes. To address these challenges, we propose Label-Augmented Analytic Adaptation (L3A), an exemplar-free approach without storing past samples. L3A integrates two key modules. The pseudo-label (PL) module implements label augmentation by generating pseudo-labels for current phase samples, addressing the label absence problem. The weighted analytic classifier (WAC) derives a closed-form solution for neural networks. It introduces sample-specific weights to adaptively balance the class contribution and mitigate class imbalance. Experiments on MS-COCO and PASCAL VOC datasets demonstrate that L3A outperforms existing methods in MLCIL tasks. Our code is available at this https URL. 

**Abstract (ZH)**: 基于类增量学习的多标签标签增广分析适应（L3A） 

---
# Unlearning Inversion Attacks for Graph Neural Networks 

**Title (ZH)**: 图神经网络的逆向攻击学习消除 

**Authors**: Jiahao Zhang, Yilong Wang, Zhiwei Zhang, Xiaorui Liu, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00808)  

**Abstract**: Graph unlearning methods aim to efficiently remove the impact of sensitive data from trained GNNs without full retraining, assuming that deleted information cannot be recovered. In this work, we challenge this assumption by introducing the graph unlearning inversion attack: given only black-box access to an unlearned GNN and partial graph knowledge, can an adversary reconstruct the removed edges? We identify two key challenges: varying probability-similarity thresholds for unlearned versus retained edges, and the difficulty of locating unlearned edge endpoints, and address them with TrendAttack. First, we derive and exploit the confidence pitfall, a theoretical and empirical pattern showing that nodes adjacent to unlearned edges exhibit a large drop in model confidence. Second, we design an adaptive prediction mechanism that applies different similarity thresholds to unlearned and other membership edges. Our framework flexibly integrates existing membership inference techniques and extends them with trend features. Experiments on four real-world datasets demonstrate that TrendAttack significantly outperforms state-of-the-art GNN membership inference baselines, exposing a critical privacy vulnerability in current graph unlearning methods. 

**Abstract (ZH)**: 基于图的学习逆向攻击：在仅拥有黑盒访问权限和部分图知识的情况下，对手能否重建已删除的边？ 

---
# Manipulating 3D Molecules in a Fixed-Dimensional SE(3)-Equivariant Latent Space 

**Title (ZH)**: 在固定维度SE(3)-等变潜在空间中操纵3D分子 

**Authors**: Zitao Chen, Yinjun Jia, Zitong Tian, Wei-Ying Ma, Yanyan Lan  

**Link**: [PDF](https://arxiv.org/pdf/2506.00771)  

**Abstract**: Medicinal chemists often optimize drugs considering their 3D structures and designing structurally distinct molecules that retain key features, such as shapes, pharmacophores, or chemical properties. Previous deep learning approaches address this through supervised tasks like molecule inpainting or property-guided optimization. In this work, we propose a flexible zero-shot molecule manipulation method by navigating in a shared latent space of 3D molecules. We introduce a Variational AutoEncoder (VAE) for 3D molecules, named MolFLAE, which learns a fixed-dimensional, SE(3)-equivariant latent space independent of atom counts. MolFLAE encodes 3D molecules using an SE(3)-equivariant neural network into fixed number of latent nodes, distinguished by learned embeddings. The latent space is regularized, and molecular structures are reconstructed via a Bayesian Flow Network (BFN) conditioned on the encoder's latent output. MolFLAE achieves competitive performance on standard unconditional 3D molecule generation benchmarks. Moreover, the latent space of MolFLAE enables zero-shot molecule manipulation, including atom number editing, structure reconstruction, and coordinated latent interpolation for both structure and properties. We further demonstrate our approach on a drug optimization task for the human glucocorticoid receptor, generating molecules with improved hydrophilicity while preserving key interactions, under computational evaluations. These results highlight the flexibility, robustness, and real-world utility of our method, opening new avenues for molecule editing and optimization. 

**Abstract (ZH)**: 医药化学家在优化药物时往往考虑其三维结构，并设计具有特定形状、药效团或化学性质的结构不同的分子。先前的深度学习方法通过分子填补或属性指导优化等监督任务来解决这一问题。在本项工作中，我们提出了一种灵活的零样本分子操纵方法，通过在三维分子共享的隐空间中导航来实现。我们引入了一种名为MolFLAE的三维分子变分自编码器（VAE），它学习一个固定维度且SE(3)-不变的隐空间，与原子数无关。MolFLAE使用SE(3)-不变的神经网络将三维分子编码为固定数量的隐节点，并通过学习嵌入进行区分。隐空间经过正则化处理，并通过贝叶斯流网络（BFN）根据编码器的隐空间输出重建分子结构。MolFLAE在标准的三维分子生成基准测试中实现了竞争性性能。此外，MolFLAE的隐空间支持零样本分子操纵，包括原子数编辑、结构重建以及结构和属性的协调隐空间插值。我们进一步在人糖皮质激素受体的药物优化任务中展示了该方法，生成了具有更好亲水性的分子，同时保留了关键相互作用，在计算评估中取得了良好的效果。这些结果突显了我们方法的灵活性、鲁棒性和实际应用价值，为分子编辑和优化开辟了新的途径。 

---
# Beyond Attention: Learning Spatio-Temporal Dynamics with Emergent Interpretable Topologies 

**Title (ZH)**: 超越注意力：基于 Emergent 可解释拓扑结构的时空动力学习 

**Authors**: Sai Vamsi Alisetti, Vikas Kalagi, Sanjukta Krishnagopal  

**Link**: [PDF](https://arxiv.org/pdf/2506.00770)  

**Abstract**: Spatio-temporal forecasting is critical in applications such as traffic prediction, energy demand modeling, and weather monitoring. While Graph Attention Networks (GATs) are popular for modeling spatial dependencies, they rely on predefined adjacency structures and dynamic attention scores, introducing inductive biases and computational overhead that can obscure interpretability.
We propose InterGAT, a simplified alternative to GAT that replaces masked attention with a fully learnable, symmetric node interaction matrix, capturing latent spatial relationships without relying on fixed graph topologies. Our framework, InterGAT-GRU, which incorporates a GRU-based temporal decoder, outperforms the baseline GAT-GRU in forecasting accuracy, achieving at least a 21% improvement on the SZ-Taxi dataset and a 6% improvement on the Los-Loop dataset across all forecasting horizons (15 to 60 minutes). Additionally, we observed reduction in training time by 60-70% compared to GAT-GRU baseline.
Crucially, the learned interaction matrix reveals interpretable structure: it recovers sparse, topology-aware attention patterns that align with community structure. Spectral and clustering analyses show that the model captures both localized and global dynamics, offering insights into the functional topology driving predictions. This highlights how structure learning can simultaneously support prediction, computational efficiency, and topological interpretabil-ity in dynamic graph-based domains. 

**Abstract (ZH)**: 空时预测对于交通预测、能源需求建模和天气监测等应用至关重要。尽管图注意网络（GATs）在建模空域依赖方面广受欢迎，但它们依赖于预定义的邻接结构和动态注意分数，引入了归纳偏差和计算开销，可能模糊了可解释性。
我们提出了一种名为InterGAT的简化替代方案，它用完全可学习的对称节点交互矩阵取代了掩码注意机制，从而捕捉潜在的空域关系，而不依赖于固定图拓扑。我们的框架InterGAT-GRU结合了基于GRU的时空解码器，在预测准确性方面优于基线GAT-GRU，在SZ-Taxi数据集和Los-Loop数据集上，所有预测时长（15到60分钟）的预测准确率分别提高了至少21%和6%。此外，与GAT-GRU基线相比，训练时间减少了60-70%。
至关重要的是，学习到的交互矩阵揭示了可解释的结构：它恢复了稀疏的、拓扑意识的注意模式，与社区结构对齐。频谱和聚类分析表明，该模型捕获了局部和全局动力学，揭示了驱动预测的功能拓扑结构。这强调了结构学习如何在动态图基环境中同时支持预测、计算效率和拓扑可解释性。 

---
# Length Aware Speech Translation for Video Dubbing 

**Title (ZH)**: 基于长度感知的语音翻译用于视频配音 

**Authors**: Harveen Singh Chadha, Aswin Shanmugam Subramanian, Vikas Joshi, Shubham Bansal, Jian Xue, Rupeshkumar Mehta, Jinyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.00740)  

**Abstract**: In video dubbing, aligning translated audio with the source audio is a significant challenge. Our focus is on achieving this efficiently, tailored for real-time, on-device video dubbing scenarios. We developed a phoneme-based end-to-end length-sensitive speech translation (LSST) model, which generates translations of varying lengths short, normal, and long using predefined tags. Additionally, we introduced length-aware beam search (LABS), an efficient approach to generate translations of different lengths in a single decoding pass. This approach maintained comparable BLEU scores compared to a baseline without length awareness while significantly enhancing synchronization quality between source and target audio, achieving a mean opinion score (MOS) gain of 0.34 for Spanish and 0.65 for Korean, respectively. 

**Abstract (ZH)**: 视频配音中，将翻译音频与源音频对齐是一项重大挑战。我们专注于实现实时、设备端视频配音中的这一目标。我们开发了一种基于音素的端到端长度敏感的语音翻译（LSST）模型，该模型使用预定义标签生成不同长度的翻译（短、正常和长）。此外，我们引入了长度感知束搜索（LABS），这是一种高效的单解码过程中生成不同长度翻译的方法。该方法在保持与无长度感知基线相当的BLEU分数的同时，显著提高了源音频和目标音频之间的同步质量，分别实现了西班牙语0.34和韩语0.65的平均意见得分（MOS）提升。 

---
# MoPINNEnKF: Iterative Model Inference using generic-PINN-based ensemble Kalman filter 

**Title (ZH)**: MoPINNEnKF：基于通用PINN的迭代模型推理使用ensemble Kalman滤波器 

**Authors**: Binghang Lu, Changhong Mou, Guang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.00731)  

**Abstract**: Physics-informed neural networks (PINNs) have emerged as a powerful tool for solving forward and inverse problems involving partial differential equations (PDEs) by incorporating physical laws into the training process. However, the performance of PINNs is often hindered in real-world scenarios involving noisy observational data and missing physics, particularly in inverse problems. In this work, we propose an iterative multi-objective PINN ensemble Kalman filter (MoPINNEnKF) framework that improves the robustness and accuracy of PINNs in both forward and inverse problems by using the \textit{ensemble Kalman filter} and the \textit{non-dominated sorting genetic algorithm} III (NSGA-III). Specifically, NSGA-III is used as a multi-objective optimizer that can generate various ensemble members of PINNs along the optimal Pareto front, while accounting the model uncertainty in the solution space. These ensemble members are then utilized within the EnKF to assimilate noisy observational data. The EnKF's analysis is subsequently used to refine the data loss component for retraining the PINNs, thereby iteratively updating their parameters. The iterative procedure generates improved solutions to the PDEs. The proposed method is tested on two benchmark problems: the one-dimensional viscous Burgers equation and the time-fractional mixed diffusion-wave equation (TFMDWE). The numerical results show it outperforms standard PINNs in handling noisy data and missing physics. 

**Abstract (ZH)**: 基于物理的神经网络（PINNs）的迭代多目标集成卡尔曼滤波（MoPINNEnKF）框架：提高前反问题中的鲁棒性和准确性 

---
# From Argumentative Text to Argument Knowledge Graph: A New Framework for Structured Argumentation 

**Title (ZH)**: 从论证文本到论证知识图谱：一种结构化论证的新框架 

**Authors**: Debarati Bhattacharjee, Ashish Anand  

**Link**: [PDF](https://arxiv.org/pdf/2506.00713)  

**Abstract**: This paper presents a framework to convert argumentative texts into argument knowledge graphs (AKG). Starting with basic annotations of argumentative components (ACs) and argumentative relations (ARs), we enrich the information by constructing a knowledge base (KB) graph with metadata attributes for nodes. Next, we use premises and inference rules from the KB to form arguments by applying modus ponens. From these arguments, we create an AKG. The nodes and edges of the AKG have attributes that capture important argumentative features. We also find missing inference rules by identifying markers. This makes it possible to identify undercut attacks that were previously undetectable in existing datasets. The AKG gives a graphical view of the argumentative structure that is easier to understand than theoretical formats. It also prepares the ground for future reasoning tasks, including checking the coherence of arguments and identifying opportunities for revision. For this, it is important to find indirect relations, many of which are implicit. Our proposed AKG format, with annotated inference rules and modus ponens, will help reasoning models learn the implicit indirect relations that require inference over arguments and the relations between them. 

**Abstract (ZH)**: 本文提出了一种将论辩文本转换为论辩知识图谱（AKG）的框架。从基本的论辩成分（ACs）和论辩关系（ARs）的标注开始，通过构建包含元数据属性的节点的知识库（KB）图来丰富信息。接着，我们利用KB中的前提和推理规则应用模态斯蓬森方法形成论辩，从这些论辩中构建AKG。AKG中的节点和边具有能够捕捉重要论辩特征的属性。我们还通过识别标记来发现缺失的推理规则，这使得之前在现有数据集中难以检测到的削弱攻击变得可识别。AKG提供了比理论格式更容易理解的论辩结构图示，也为未来的推理任务做了准备，包括检查论辩的一致性和识别修订机会。为此，找到许多隐含的间接关系很重要。我们提出的带有标注推理规则和模态斯蓬森方法的AKG格式，将有助于推理模型学习需要在论辩及其之间关系上进行推理的隐含间接关系。 

---
# Bayesian Inference of Training Dataset Membership 

**Title (ZH)**: 基于贝叶斯推断的训练数据集成员识别 

**Authors**: Yongchao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00701)  

**Abstract**: Determining whether a dataset was part of a machine learning model's training data pool can reveal privacy vulnerabilities, a challenge often addressed through membership inference attacks (MIAs). Traditional MIAs typically require access to model internals or rely on computationally intensive shadow models. This paper proposes an efficient, interpretable and principled Bayesian inference method for membership inference. By analyzing post-hoc metrics such as prediction error, confidence (entropy), perturbation magnitude, and dataset statistics from a trained ML model, our approach computes posterior probabilities of membership without requiring extensive model training. Experimental results on synthetic datasets demonstrate the method's effectiveness in distinguishing member from non-member datasets. Beyond membership inference, this method can also detect distribution shifts, offering a practical and interpretable alternative to existing approaches. 

**Abstract (ZH)**: 确定数据集是否为机器学习模型训练数据池的一部分可以揭示隐私漏洞，这一挑战通常通过成员 inference 攻击（MIAs）来应对。本文提出了一种高效、可解释且原理性的贝叶斯推理方法用于成员 inference。通过分析训练后的 ML 模型的预测误差、置信度（熵）、扰动幅度以及数据集统计信息等后验指标，本方法可在不需要大量模型训练的情况下计算成员 posterior 概率。实验结果表明，该方法在区分成员数据集和非成员数据集方面具有有效性。除此之外，该方法还可以检测分布偏移，提供了一种实用且可解释的替代现有方法的选择。 

---
# CineMA: A Foundation Model for Cine Cardiac MRI 

**Title (ZH)**: CineMA: 电影磁共振成像心脏基础模型 

**Authors**: Yunguan Fu, Weixi Yi, Charlotte Manisty, Anish N Bhuva, Thomas A Treibel, James C Moon, Matthew J Clarkson, Rhodri Huw Davies, Yipeng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00679)  

**Abstract**: Cardiac magnetic resonance (CMR) is a key investigation in clinical cardiovascular medicine and has been used extensively in population research. However, extracting clinically important measurements such as ejection fraction for diagnosing cardiovascular diseases remains time-consuming and subjective. We developed CineMA, a foundation AI model automating these tasks with limited labels. CineMA is a self-supervised autoencoder model trained on 74,916 cine CMR studies to reconstruct images from masked inputs. After fine-tuning, it was evaluated across eight datasets on 23 tasks from four categories: ventricle and myocardium segmentation, left and right ventricle ejection fraction calculation, disease detection and classification, and landmark localisation. CineMA is the first foundation model for cine CMR to match or outperform convolutional neural networks (CNNs). CineMA demonstrated greater label efficiency than CNNs, achieving comparable or better performance with fewer annotations. This reduces the burden of clinician labelling and supports replacing task-specific training with fine-tuning foundation models in future cardiac imaging applications. Models and code for pre-training and fine-tuning are available at this https URL, democratising access to high-performance models that otherwise require substantial computational resources, promoting reproducibility and accelerating clinical translation. 

**Abstract (ZH)**: 心脏磁共振成像（CMR）是临床心血管医学中的关键检查方法，在人口研究中得到了广泛的应用。然而，提取如射血分数等临床重要测量以诊断心血管疾病仍耗时且主观。我们开发了CineMA，这是一种基于有限标签自动完成这些任务的基础AI模型。CineMA是一种在74,916例心脏MRI研究上自我监督训练的自编码器模型，用于从掩码输入中重建图像。经过微调后，它在八个数据集上的23项跨四个类别（心室和心肌分割、左心室和右心室射血分数计算、疾病检测和分类、以及解剖标志定位）的任务上进行了评估。CineMA是第一个能够与卷积神经网络（CNNs）匹敌或超越的用于心脏电影MRI的基础模型，展示了比CNNs更高的标记效率，在更少标注的情况下实现类似或更好的性能，从而减轻了临床人员的标注负担，并支持在未来的心脏成像应用中使用基础模型的微调而非特定任务的训练。CineMA模型和训练代码可在以下网址获取，促进了高性能模型的民主化访问，促进了研究的可复制性并加速了临床转化。 

---
# Thinking Out of the Box: Hybrid SAT Solving by Unconstrained Continuous Optimization 

**Title (ZH)**: 打破常规：基于不受约束的连续优化的混合SAT求解 

**Authors**: Zhiwei Zhang, Samy Wu Fung, Anastasios Kyrillidis, Stanley Osher, Moshe Y. Vardi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00674)  

**Abstract**: The Boolean satisfiability (SAT) problem lies at the core of many applications in combinatorial optimization, software verification, cryptography, and machine learning. While state-of-the-art solvers have demonstrated high efficiency in handling conjunctive normal form (CNF) formulas, numerous applications require non-CNF (hybrid) constraints, such as XOR, cardinality, and Not-All-Equal constraints. Recent work leverages polynomial representations to represent such hybrid constraints, but it relies on box constraints that can limit the use of powerful unconstrained optimizers. In this paper, we propose unconstrained continuous optimization formulations for hybrid SAT solving by penalty terms. We provide theoretical insights into when these penalty terms are necessary and demonstrate empirically that unconstrained optimizers (e.g., Adam) can enhance SAT solving on hybrid benchmarks. Our results highlight the potential of combining continuous optimization and machine-learning-based methods for effective hybrid SAT solving. 

**Abstract (ZH)**: 布尔可满足性（SAT）问题在组合优化、软件验证、密码学和机器学习等多个领域的应用中处于核心地位。尽管最先进的求解器在处理合取范式（CNF）公式时表现出高效性，但许多应用需要非CNF（混合）约束，如XOR、基数和Not-All-Equal约束。最近的研究利用多项式表示法来表示这些混合约束，但这种方法依赖于盒约束，这可能会限制使用强大的无约束优化器的能力。本文提出了一种通过惩罚项的无约束连续优化形式来解决混合SAT问题。我们提供了这些惩罚项必要的理论见解，并通过实验证明，无约束优化器（如Adam）可以提升混合基准上的SAT求解性能。我们的结果突显了结合连续优化和基于机器学习的方法在有效混合SAT求解中的潜力。 

---
# Differential Privacy for Deep Learning in Medicine 

**Title (ZH)**: 医学中深度学习的差分隐私保护 

**Authors**: Marziyeh Mohammadi, Mohsen Vejdanihemmat, Mahshad Lotfinia, Mirabela Rusu, Daniel Truhn, Andreas Maier, Soroosh Tayebi Arasteh  

**Link**: [PDF](https://arxiv.org/pdf/2506.00660)  

**Abstract**: Differential privacy (DP) is a key technique for protecting sensitive patient data in medical deep learning (DL). As clinical models grow more data-dependent, balancing privacy with utility and fairness has become a critical challenge. This scoping review synthesizes recent developments in applying DP to medical DL, with a particular focus on DP-SGD and alternative mechanisms across centralized and federated settings. Using a structured search strategy, we identified 74 studies published up to March 2025. Our analysis spans diverse data modalities, training setups, and downstream tasks, and highlights the tradeoffs between privacy guarantees, model accuracy, and subgroup fairness. We find that while DP-especially at strong privacy budgets-can preserve performance in well-structured imaging tasks, severe degradation often occurs under strict privacy, particularly in underrepresented or complex modalities. Furthermore, privacy-induced performance gaps disproportionately affect demographic subgroups, with fairness impacts varying by data type and task. A small subset of studies explicitly addresses these tradeoffs through subgroup analysis or fairness metrics, but most omit them entirely. Beyond DP-SGD, emerging approaches leverage alternative mechanisms, generative models, and hybrid federated designs, though reporting remains inconsistent. We conclude by outlining key gaps in fairness auditing, standardization, and evaluation protocols, offering guidance for future work toward equitable and clinically robust privacy-preserving DL systems in medicine. 

**Abstract (ZH)**: 差分隐私（DP）是医疗深度学习（DL）中保护敏感患者数据的关键技术。随着临床模型对数据的依赖性增加，平衡隐私与效用和公平性已成为一个关键挑战。本综述总结了近年来将DP应用于医疗DL的最新发展，特别关注中央和联邦设置下的DP-SGD和替代机制。通过结构化的检索策略，我们识别了截至2025年3月发表的74项研究。我们的分析涵盖了多种数据模态、训练配置和下游任务，并突出了隐私保证、模型准确性和子群体公平性之间的权衡。研究发现，尽管差分隐私（特别是较强隐私预算下）可以保持良好结构成像任务的性能，但在严格的隐私保护条件下，特别是在未充分代表或复杂的模态下，通常会出现严重的性能下降。此外，由隐私引起的表现差距不成比例地影响人口子群体，其影响程度因数据类型和任务而异。仅有一小部分研究通过子群体分析或公平性指标明确地处理了这些权衡，而大多数研究未提及。除了DP-SGD，新兴方法还利用了替代机制、生成模型和混合联邦设计，但报告仍不一致。我们最终概述了公平审计、标准化和评估协议的关键缺口，为未来工作提供指导，旨在建立在医学中公平且临床稳健的隐私保护DL系统。 

---
# Sarc7: Evaluating Sarcasm Detection and Generation with Seven Types and Emotion-Informed Techniques 

**Title (ZH)**: Sarc7: 七种类型与情感导向技术的-Trump-讽喻检测与生成评估 

**Authors**: Lang Xiong, Raina Gao, Alyssa Jeong, Yicheng Fu, Sean O'Brien, Vasu Sharma, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00658)  

**Abstract**: Sarcasm is a form of humor where expressions convey meanings opposite to their literal interpretations. Classifying and generating sarcasm using large language models is vital for interpreting human communication. Sarcasm poses challenges for computational models, due to its nuanced nature. We introduce Sarc7, a benchmark that classifies 7 types of sarcasm: self-deprecating, brooding, deadpan, polite, obnoxious, raging, and manic by annotating entries of the MUStARD dataset. Classification was evaluated using zero-shot, few-shot, chain-of-thought (CoT), and a novel emotion-based prompting technique. We propose an emotion-based generation method developed by identifying key components of sarcasm-incongruity, shock value, and context dependency. Our classification experiments show that Gemini 2.5, using emotion-based prompting, outperforms other setups with an F1 score of 0.3664. Human evaluators preferred our emotion-based prompting, with 38.46% more successful generations than zero-shot prompting. 

**Abstract (ZH)**: 讽刺是一种语气与其字面含义相反的幽默形式。使用大型语言模型对讽刺进行分类和生成对于解读人类通信至关重要。由于讽刺的细微差别，它对计算模型构成挑战。我们引入了Sarc7基准，通过注释MUStARD数据集的条目，对7种类型的讽刺进行分类：自我贬低、沉思、无趣、礼貌、讨厌、愤怒和狂热。分类评估使用了零样本、少量样本、逐步推理（CoT）以及一种新型的情感提示技术。我们提出了一种基于情感的生成方法，通过识别讽刺不一致、冲击价值和情境依存性等关键组件。我们的分类实验表明，使用情感提示的Gemini 2.5在F1分数上表现最佳，为0.3664。人类评估者更偏好我们的情感提示，成功生成的比例比零样本提示高38.46%。 

---
# Permutation-Invariant Transformer Neural Architectures for Set-Based Indoor Localization Using Learned RSSI Embeddings 

**Title (ZH)**: 基于学习到的RSSI嵌入的排列不变变压器神经架构的基于集合的室内定位 

**Authors**: Aris J. Aristorenas  

**Link**: [PDF](https://arxiv.org/pdf/2506.00656)  

**Abstract**: We propose a permutation-invariant neural architecture for indoor localization using RSSI scans from Wi-Fi access points. Each scan is modeled as an unordered set of (BSSID, RSSI) pairs, where BSSIDs are mapped to learned embeddings and concatenated with signal strength. These are processed by a Set Transformer, enabling the model to handle variable-length, sparse inputs while learning attention- based representations over access point relationships. We evaluate the model on a dataset collected across a campus environment consisting of six buildings. Results show that the model accurately recovers fine-grained spatial structure and maintains performance across physically distinct domains. In our experiments, a simple LSTM consistently outperformed all other models, achieving the lowest mean localization error across three tasks (E1 - E3), with average errors as low as 2.23 m. The Set Transformer performed competitively, ranking second in every experiment and outperforming the MLP, RNN, and basic attention models, particularly in scenarios involving multiple buildings (E2) and multiple floors (E3). Performance degraded most in E2, where signal conditions varied substantially across buildings, highlighting the importance of architectural robustness to domain diversity. This work demonstrates that set-based neural models are a natural fit for signal-based localization, offering a principled approach to handling sparse, unordered inputs in real-world positioning tasks. 

**Abstract (ZH)**: 基于Wi-Fi接入点RSSI扫描的置换不变神经架构的室内定位 

---
# Clinical Annotations for Automatic Stuttering Severity Assessment 

**Title (ZH)**: 临床标注用于自动 stuttering 严重程度评估 

**Authors**: Ana Rita Valente, Rufael Marew, Hawau Olamide Toyin, Hamdan Al-Ali, Anelise Bohnen, Inma Becerra, Elsa Marta Soares, Goncalo Leal, Hanan Aldarmaki  

**Link**: [PDF](https://arxiv.org/pdf/2506.00644)  

**Abstract**: Stuttering is a complex disorder that requires specialized expertise for effective assessment and treatment. This paper presents an effort to enhance the FluencyBank dataset with a new stuttering annotation scheme based on established clinical standards. To achieve high-quality annotations, we hired expert clinicians to label the data, ensuring that the resulting annotations mirror real-world clinical expertise. The annotations are multi-modal, incorporating audiovisual features for the detection and classification of stuttering moments, secondary behaviors, and tension scores. In addition to individual annotations, we additionally provide a test set with highly reliable annotations based on expert consensus for assessing individual annotators and machine learning models. Our experiments and analysis illustrate the complexity of this task that necessitates extensive clinical expertise for valid training and evaluation of stuttering assessment models. 

**Abstract (ZH)**: 结巴症是一种复杂的障碍，需要专门的专家才能进行有效的评估和治疗。本文提出了一种努力扩展FluencyBank数据集的方法，基于现有的临床标准建立了新的结巴标注方案。为了获得高质量的标注，我们聘请了专家临床人员对数据进行了标注，确保最终的标注反映了真实的临床专业知识。这些标注是多模态的，结合了音频视觉特征以检测和分类结巴时刻、附带行为以及紧张度评分。除了个体标注外，我们还提供了一个基于专家共识的可靠测试集，用于评估个体标注者和机器学习模型。我们的实验和分析说明了这一任务的复杂性，需要广泛的临床专业知识来进行有效的训练和评估结巴评估模型。 

---
# Improving the Calibration of Confidence Scores in Text Generation Using the Output Distribution's Characteristics 

**Title (ZH)**: 基于输出分布特性提高文本生成中置信分数的校准 

**Authors**: Lorenzo Jaime Yu Flores, Ori Ernst, Jackie Chi Kit Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2506.00637)  

**Abstract**: Well-calibrated model confidence scores can improve the usefulness of text generation models. For example, users can be prompted to review predictions with low confidence scores, to prevent models from returning bad or potentially dangerous predictions. However, confidence metrics are not always well calibrated in text generation. One reason is that in generation, there can be many valid answers, which previous methods do not always account for. Hence, a confident model could distribute its output probability among multiple sequences because they are all valid. We propose task-agnostic confidence metrics suited to generation, which rely solely on the probabilities associated with the model outputs without the need for further fine-tuning or heuristics. Using these, we are able to improve the calibration of BART and Flan-T5 on summarization, translation, and QA datasets. 

**Abstract (ZH)**: Well-calibrated模型信心分数可以提高文本生成模型的实用性。例如，用户可以被提示审查低信心分数的预测，以防止模型返回差劲或潜在危险的预测。然而，在文本生成中，信心度量并不总是很好地校准。一个原因是生成过程中可能存在许多有效的答案，之前的 方法并没有总是考虑到这一点。因此，一个有信心的模型可能会将其输出概率分布在多个序列中，因为它们都是有效的。我们提出了适用于生成的任务无关的信心度量，这些度量仅依赖于模型输出相关的概率，而无需进一步微调或启发式方法。利用这些方法，我们能够改善BART和Flan-T5在总结、翻译和问答数据集上的校准。 

---
# Learning with Calibration: Exploring Test-Time Computing of Spatio-Temporal Forecasting 

**Title (ZH)**: 学习与校准：探索时空预测的测试时计算 

**Authors**: Wei Chen, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00635)  

**Abstract**: Spatio-temporal forecasting is crucial in many domains, such as transportation, meteorology, and energy. However, real-world scenarios frequently present challenges such as signal anomalies, noise, and distributional shifts. Existing solutions primarily enhance robustness by modifying network architectures or training procedures. Nevertheless, these approaches are computationally intensive and resource-demanding, especially for large-scale applications. In this paper, we explore a novel test-time computing paradigm, namely learning with calibration, ST-TTC, for spatio-temporal forecasting. Through learning with calibration, we aim to capture periodic structural biases arising from non-stationarity during the testing phase and perform real-time bias correction on predictions to improve accuracy. Specifically, we first introduce a spectral-domain calibrator with phase-amplitude modulation to mitigate periodic shift and then propose a flash updating mechanism with a streaming memory queue for efficient test-time computation. ST-TTC effectively bypasses complex training-stage techniques, offering an efficient and generalizable paradigm. Extensive experiments on real-world datasets demonstrate the effectiveness, universality, flexibility and efficiency of our proposed method. 

**Abstract (ZH)**: 空间时态预测对于交通、气象和能源等领域至关重要。然而，现实场景中常常存在信号异常、噪声和分布偏移等挑战。现有解决方案主要通过修改网络架构或训练过程来增强鲁棒性，但这在大规模应用场景下计算密集且资源消耗大。本文探索了一种新的测试时计算范式——校准学习，即ST-TTC（时空测试时计算），用于空间时态预测。通过校准学习，我们旨在捕捉测试阶段由非站定性引起的周期结构偏差，并进行实时偏差校正以提高预测准确性。具体而言，我们首先引入了一种谱域校准器，通过相位-幅度调制来减轻周期性移位，然后提出了一种快速更新机制，配备流式内存队列，以实现高效的测试时计算。ST-TTC 有效地绕过了复杂的训练阶段技术，提供了一种高效且通用的范式。在实际数据集上的广泛实验表明，所提出的方法具有有效性、通用性、灵活性和高效性。 

---
# The Disparate Effects of Partial Information in Bayesian Strategic Learning 

**Title (ZH)**: 部分信息在贝叶斯战略学习中的不同影响 

**Authors**: Srikanth Avasarala, Serena Wang, Juba Ziani  

**Link**: [PDF](https://arxiv.org/pdf/2506.00627)  

**Abstract**: We study how partial information about scoring rules affects fairness in strategic learning settings. In strategic learning, a learner deploys a scoring rule, and agents respond strategically by modifying their features -- at some cost -- to improve their outcomes. However, in our work, agents do not observe the scoring rule directly; instead, they receive a noisy signal of said rule. We consider two different agent models: (i) naive agents, who take the noisy signal at face value, and (ii) Bayesian agents, who update a prior belief based on the signal.
Our goal is to understand how disparities in outcomes arise between groups that differ in their costs of feature modification, and how these disparities vary with the level of transparency of the learner's rule. For naive agents, we show that utility disparities can grow unboundedly with noise, and that the group with lower costs can, perhaps counter-intuitively, be disproportionately harmed under limited transparency. In contrast, for Bayesian agents, disparities remain bounded. We provide a full characterization of disparities across groups as a function of the level of transparency and show that they can vary non-monotonically with noise; in particular, disparities are often minimized at intermediate levels of transparency. Finally, we extend our analysis to settings where groups differ not only in cost, but also in prior beliefs, and study how this asymmetry influences fairness. 

**Abstract (ZH)**: 我们研究部分信息如何影响计分规则在战略学习环境中的公平性。在战略学习中，学习者部署一个计分规则，代理通过修改其特征以改善结果来进行战略响应——但这些修改是有成本的。然而，在我们的研究中，代理不会直接观察到计分规则，而是接收到一个带有噪声的规则信号。我们考虑了两种不同的代理模型：（i）无知代理，他们会认为噪声信号是真实的；（ii）贝叶斯代理，他们会根据信号更新先验信念。

我们的目标是理解在特征修改成本不同的组之间结果差异是如何产生的，以及这些差异如何随着学习者规则透明度的变化而变化。对于无知代理，我们证明了在噪声存在的情况下，效用差异可以无限增长，并且在透明度有限的情况下，成本较低的组可能会出乎意料地受到不成比例的伤害。相反，对于贝叶斯代理，差异保持在有限范围内。我们提供了差异在整个组之间作为透明度函数的完整描述，并展示了它们如何非单调地随噪声变化；特别地，差异往往在透明度的中间水平时最小化。最后，我们将分析扩展到组不仅在成本方面存在差异，还在先验信念方面也存在差异的情境，并研究这种不对称性如何影响公平性。 

---
# Improving Dialogue State Tracking through Combinatorial Search for In-Context Examples 

**Title (ZH)**: 通过组合搜索改进基于上下文示例的对话状态追踪 

**Authors**: Haesung Pyun, Yoonah Park, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2506.00622)  

**Abstract**: In dialogue state tracking (DST), in-context learning comprises a retriever that selects labeled dialogues as in-context examples and a DST model that uses these examples to infer the dialogue state of the query dialogue. Existing methods for constructing training data for retrievers suffer from three key limitations: (1) the synergistic effect of examples is not considered, (2) the linguistic characteristics of the query are not sufficiently factored in, and (3) scoring is not directly optimized for DST performance. Consequently, the retriever can fail to retrieve examples that would substantially improve DST performance. To address these issues, we present CombiSearch, a method that scores effective in-context examples based on their combinatorial impact on DST performance. Our evaluation on MultiWOZ shows that retrievers trained with CombiSearch surpass state-of-the-art models, achieving a 20x gain in data efficiency and generalizing well to the SGD dataset. Moreover, CombiSearch attains a 12% absolute improvement in the upper bound DST performance over traditional approaches when no retrieval errors are assumed. This significantly increases the headroom for practical DST performance while demonstrating that existing methods rely on suboptimal data for retriever training. 

**Abstract (ZH)**: 基于组合影响的对话状态跟踪中的检索学习方法 

---
# A Topological Semantics of Dialogue: Nerve Structures and Logical Extraction 

**Title (ZH)**: 拓扑对话语义：神经结构与逻辑提取 

**Authors**: Andreu Ballus Santacana  

**Link**: [PDF](https://arxiv.org/pdf/2506.00615)  

**Abstract**: We introduce a concise, topologically-motivated semantics for finite dialogues by mapping each utterance to an open set in a fixed semantic space, building the corresponding nerve complex of joint satisfiability, and extracting fundamental combinatorial invariants:
1. The negative nerve, which enumerates all finite collections of utterances whose
opens have empty intersection, providing a straightforward criterion for merging
separate transcripts without contradiction.
2. The global interpretation subspace, the unique minimal open in which all asserted
utterances hold simultaneously, enabling effective enumeration of all logical
consequences of the entire dialogue.
3. A practical demonstration in the Wolfram Language, with algorithms for constructing
nerves, detecting inconsistencies, and computing the global interpretation, thereby
illustrating computational feasibility.
Our framework is grounded in classical duality and topological semantics (Stone duality, Priestley duality, Tarski's semantics, coherence-space methods, Scott domains, topos semantics, and homotopy type theory) while drawing on recent advances in topological data analysis and dialogue-based semantics. 

**Abstract (ZH)**: 我们引入了一种简洁的、拓扑驱动的有限对话语义，将每个表述映射到固定语义空间中的一个开集，构建相应的联合可满足性神经复杂体，并提取基本组合不变量：
1. 负神经，枚举所有具有空交集的有限表述集，提供合并不矛盾的独立会话的直接标准。
2. 全局解释子空间，所有断言的表述同时成立的唯一最小开集，使得可以有效枚举整个对话的所有逻辑推论。
3. 用Wolfram语言进行实用演示，包括构建神经网络、检测不一致性和计算全局解释的算法，从而说明其实现可行性。
我们的框架基于经典对偶性和拓扑语义（Stone对偶性、Priestley对偶性、塔尔斯基语义、调和空间方法、Scott域、范畴语义和同调类型理论），并借鉴了拓扑数据分析和基于对话的语义领域的最新进展。 

---
# Predictability-Aware Compression and Decompression Framework for Multichannel Time Series Data 

**Title (ZH)**: 面向可预测性的多通道时间序列数据压缩与解压缩框架 

**Authors**: Ziqi Liu, Pei Zeng, Yi Ding  

**Link**: [PDF](https://arxiv.org/pdf/2506.00614)  

**Abstract**: Real-world multichannel time series prediction faces growing demands for efficiency across edge and cloud environments, making channel compression a timely and essential problem. Motivated by success of Multiple-Input Multiple-Output (MIMO) methods, we propose a predictability-aware compression-decompression framework to reduce runtime, lower communication cost, and maintain prediction accuracy across diverse predictors. The core idea involves using a circular periodicity key matrix with orthogonality to capture underlying time series predictability during compression and to mitigate reconstruction errors during decompression by relaxing oversimplified data assumptions. Theoretical and empirical analyses show that the proposed framework is both time-efficient and scalable under a large number of channels. Extensive experiments on six datasets across various predictors demonstrate that the proposed method achieves superior overall performance by jointly considering prediction accuracy and runtime, while maintaining strong compatibility with diverse predictors. 

**Abstract (ZH)**: 实时光纤多通道时间序列预测在边缘和云环境中的需求 growing, 促使通道压缩成为一项及时且必要的问题。受 Multiple-Input Multiple-Output (MIMO) 方法成功的启发, 我们提出一种预测性意识下的压缩-解压缩框架, 以降低运行时开销、减少通信成本并保持预测准确性, 并适用于多种预测器。该框架的核心思想是在压缩过程中使用带有正交性的循环周期性键矩阵来捕捉潜在的时间序列预测性, 并在解压缩过程中通过放松对简单数据假设的过简化来减轻重建误差。理论分析和实证研究表明, 所提出的框架在大量通道的情况下具有时间高效性和扩展性。在六个不同数据集上的广泛实验表明, 所提出的方法通过同时考虑预测准确性和运行时开销, 达到了优异的整体性能, 同时与多种预测器保持了良好的兼容性。 

---
# Parallel Rescaling: Rebalancing Consistency Guidance for Personalized Diffusion Models 

**Title (ZH)**: 并行缩放：个人化扩散模型的一致性指导重平衡方法 

**Authors**: JungWoo Chae, Jiyoon Kim, Sangheum Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00607)  

**Abstract**: Personalizing diffusion models to specific users or concepts remains challenging, particularly when only a few reference images are available. Existing methods such as DreamBooth and Textual Inversion often overfit to limited data, causing misalignment between generated images and text prompts when attempting to balance identity fidelity with prompt adherence. While Direct Consistency Optimization (DCO) with its consistency-guided sampling partially alleviates this issue, it still struggles with complex or stylized prompts. In this paper, we propose a parallel rescaling technique for personalized diffusion models. Our approach explicitly decomposes the consistency guidance signal into parallel and orthogonal components relative to classifier free guidance (CFG). By rescaling the parallel component, we minimize disruptive interference with CFG while preserving the subject's identity. Unlike prior personalization methods, our technique does not require additional training data or expensive annotations. Extensive experiments show improved prompt alignment and visual fidelity compared to baseline methods, even on challenging stylized prompts. These findings highlight the potential of parallel rescaled guidance to yield more stable and accurate personalization for diverse user inputs. 

**Abstract (ZH)**: 个性化扩散模型面向特定用户或概念 remains 挑战性，特别是在仅有少量参考图像的情况下。现有的方法如 DreamBooth 和 Textual Inversion 经常会对有限的数据过拟合，在尝试平衡身份保真度与提示一致性时导致生成图像与文本提示之间的对齐偏差。虽然直接一致性优化 (DCO) 借助一致性引导采样部分缓解了这一问题，但仍然难以应对复杂的或风格化的提示。在本文中，我们提出了一种并行缩放技术用于个性化扩散模型。我们的方法显式地将一致性引导信号分解为相对于 classifier-free guidance (CFG) 平行和正交成分。通过缩放平行成分，我们最小化了对 CFG 的干扰同时保持主题的身份。与先前的个性化方法不同，我们的技术不需要额外的训练数据或昂贵的标注。广泛实验表明，在即使是复杂风格化提示的情况下，我们的方法也优于基线方法，实现了更好的提示对齐和视觉保真度。这些发现突显了并行缩放引导在为多样化用户输入提供更稳定和准确的个性化方面的潜力。 

---
# Graph Evidential Learning for Anomaly Detection 

**Title (ZH)**: 图证据学习异常检测 

**Authors**: Chunyu Wei, Wenji Hu, Xingjia Hao, Yunhai Wang, Yueguo Chen, Bing Bai, Fei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00594)  

**Abstract**: Graph anomaly detection faces significant challenges due to the scarcity of reliable anomaly-labeled datasets, driving the development of unsupervised methods. Graph autoencoders (GAEs) have emerged as a dominant approach by reconstructing graph structures and node features while deriving anomaly scores from reconstruction errors. However, relying solely on reconstruction error for anomaly detection has limitations, as it increases the sensitivity to noise and overfitting. To address these issues, we propose Graph Evidential Learning (GEL), a probabilistic framework that redefines the reconstruction process through evidential learning. By modeling node features and graph topology using evidential distributions, GEL quantifies two types of uncertainty: graph uncertainty and reconstruction uncertainty, incorporating them into the anomaly scoring mechanism. Extensive experiments demonstrate that GEL achieves state-of-the-art performance while maintaining high robustness against noise and structural perturbations. 

**Abstract (ZH)**: 图异常检测面临着由于可靠的异常标记数据集稀缺而带来的显著挑战，推动了无监督方法的发展。图自编码器（GAEs）通过重构图结构和节点特征，并从重构错误中衍生异常评分，成为主导方法。然而，仅依赖重构误差进行异常检测存在局限性，因为它增加了对噪声和过拟合的敏感性。为解决这些问题，我们提出了一种概率框架——图证据学习（GEL），该框架通过证据学习重新定义重构过程。通过使用证据分布来建模节点特征和图拓扑，GEL量化了两类不确定性：图不确定性与重构不确定性，并将其融入异常评分机制中。大量实验表明，GEL在保持对噪声和结构扰动的高鲁棒性的同时，实现了最先进的性能。 

---
# Temporal Chunking Enhances Recognition of Implicit Sequential Patterns 

**Title (ZH)**: 时序分割增强隐式序列模式识别 

**Authors**: Jayanta Dey, Nicholas Soures, Miranda Gonzales, Itamar Lerner, Christopher Kanan, Dhireesha Kudithipudi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00588)  

**Abstract**: In this pilot study, we propose a neuro-inspired approach that compresses temporal sequences into context-tagged chunks, where each tag represents a recurring structural unit or``community'' in the sequence. These tags are generated during an offline sleep phase and serve as compact references to past experience, allowing the learner to incorporate information beyond its immediate input range. We evaluate this idea in a controlled synthetic environment designed to reveal the limitations of traditional neural network based sequence learners, such as recurrent neural networks (RNNs), when facing temporal patterns on multiple timescales. We evaluate this idea in a controlled synthetic environment designed to reveal the limitations of traditional neural network based sequence learners, such as recurrent neural networks (RNNs), when facing temporal patterns on multiple timescales. Our results, while preliminary, suggest that temporal chunking can significantly enhance learning efficiency under resource constrained settings. A small-scale human pilot study using a Serial Reaction Time Task further motivates the idea of structural abstraction. Although limited to synthetic tasks, this work serves as an early proof-of-concept, with initial evidence that learned context tags can transfer across related task, offering potential for future applications in transfer learning. 

**Abstract (ZH)**: 本研究试点提出了一种受神经启发的方法，将时间序列压缩为带上下文标签的片段，每个标签代表序列中的一个 recurring 结构单元或“社区”。这些标签在离线睡眠阶段生成，作为对过去经验的紧凑引用，使学习者能够整合超出其即时输入范围的信息。我们在一个旨在揭示传统基于神经网络的时间序列学习器（如循环神经网络RNN）在多时间尺度时间模式面前局限性的受控合成环境中评估这一想法。初始结果表明，在资源受限的情况下，时间片段化可以显著提高学习效率。小型人类试点研究通过使用序列反应时间任务进一步证实了结构抽象的概念。尽管仅限于合成任务，但本研究作为早期概念证明，提供了初步证据，表明学习到的上下文标签可以在相关任务之间迁移，为未来在迁移学习中的应用提供了潜力。 

---
# Imputation of Missing Data in Smooth Pursuit Eye Movements Using a Self-Attention-based Deep Learning Approach 

**Title (ZH)**: 基于自注意力的深度学习方法在平滑追求眼球运动中的缺省数据插补 

**Authors**: Mehdi Bejani, Guillermo Perez-de-Arenaza-Pozo, Julián D. Arias-Londoño, Juan I. Godino-LLorente  

**Link**: [PDF](https://arxiv.org/pdf/2506.00545)  

**Abstract**: Missing data is a relevant issue in time series, especially in biomedical sequences such as those corresponding to smooth pursuit eye movements, which often contain gaps due to eye blinks and track losses, complicating the analysis and extraction of meaningful biomarkers. In this paper, a novel imputation framework is proposed using Self-Attention-based Imputation networks for time series, which leverages the power of deep learning and self-attention mechanisms to impute missing data. We further refine the imputed data using a custom made autoencoder, tailored to represent smooth pursuit eye movement sequences. The proposed approach was implemented using 5,504 sequences from 172 Parkinsonian patients and healthy controls. Results show a significant improvement in the accuracy of reconstructed eye movement sequences with respect to other state of the art techniques, substantially reducing the values for common time domain error metrics such as the mean absolute error, mean relative error, and root mean square error, while also preserving the signal's frequency domain characteristics. Moreover, it demonstrates robustness when large intervals of data are missing. This method offers an alternative solution for robustly handling missing data in time series, enhancing the reliability of smooth pursuit analysis for the screening and monitoring of neurodegenerative disorders. 

**Abstract (ZH)**: 时间序列中缺失数据是一个相关的问题，尤其是在平滑追寻眼动等生物医学序列中，这些序列常常由于眨眼和跟踪丢失而包含缺口，这使得分析和提取有意义的生物标记变得复杂。本文提出了一种新的插补框架，使用基于自注意力的插补网络进行时间序列插补，该框架利用了深度学习和自注意力机制来插补缺失数据。我们进一步使用一个根据平滑追寻眼动序列定制的自编码器对插补后的数据进行了细化。该方法使用了来自172名帕金森病患者和健康对照者的5,504个序列进行了实现。结果表明，与现有的先进技术相比，在重建眼动序列的准确性方面有显著提高，大幅降低了常见的时域误差指标（如绝对误差均值、相对误差均值和均方根误差）的值，同时也保留了信号的频域特征。此外，该方法在大量数据缺失的情况下也表现出鲁棒性。该方法为时间序列中稳健地处理缺失数据提供了一种替代方案，增强了平滑追寻分析在神经退行性疾病筛查和监测中的可靠性。 

---
# PVP: An Image Dataset for Personalized Visual Persuasion with Persuasion Strategies, Viewer Characteristics, and Persuasiveness Ratings 

**Title (ZH)**: PVP：一个包含说服策略、观众特征和说服力评分的个性化视觉说服图像数据集 

**Authors**: Junseo Kim, Jongwook Han, Dongmin Choi, Jongwook Yoon, Eun-Ju Lee, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2506.00481)  

**Abstract**: Visual persuasion, which uses visual elements to influence cognition and behaviors, is crucial in fields such as advertising and political communication. With recent advancements in artificial intelligence, there is growing potential to develop persuasive systems that automatically generate persuasive images tailored to individuals. However, a significant bottleneck in this area is the lack of comprehensive datasets that connect the persuasiveness of images with the personal information about those who evaluated the images. To address this gap and facilitate technological advancements in personalized visual persuasion, we release the Personalized Visual Persuasion (PVP) dataset, comprising 28,454 persuasive images across 596 messages and 9 persuasion strategies. Importantly, the PVP dataset provides persuasiveness scores of images evaluated by 2,521 human annotators, along with their demographic and psychological characteristics (personality traits and values). We demonstrate the utility of our dataset by developing a persuasive image generator and an automated evaluator, and establish benchmark baselines. Our experiments reveal that incorporating psychological characteristics enhances the generation and evaluation of persuasive images, providing valuable insights for personalized visual persuasion. 

**Abstract (ZH)**: 个性化视觉说服（PVP）数据集：包含28,454张说服性图像及其评估者的心理和人口统计特征 

---
# SST: Self-training with Self-adaptive Thresholding for Semi-supervised Learning 

**Title (ZH)**: SST：自适应阈值自我训练用于半监督学习 

**Authors**: Shuai Zhao, Heyan Huang, Xinge Li, Xiaokang Chen, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00467)  

**Abstract**: Neural networks have demonstrated exceptional performance in supervised learning, benefiting from abundant high-quality annotated data. However, obtaining such data in real-world scenarios is costly and labor-intensive. Semi-supervised learning (SSL) offers a solution to this problem. Recent studies, such as Semi-ViT and Noisy Student, which employ consistency regularization or pseudo-labeling, have demonstrated significant achievements. However, they still face challenges, particularly in accurately selecting sufficient high-quality pseudo-labels due to their reliance on fixed thresholds. Recent methods such as FlexMatch and FreeMatch have introduced flexible or self-adaptive thresholding techniques, greatly advancing SSL research. Nonetheless, their process of updating thresholds at each iteration is deemed time-consuming, computationally intensive, and potentially unnecessary. To address these issues, we propose Self-training with Self-adaptive Thresholding (SST), a novel, effective, and efficient SSL framework. SST introduces an innovative Self-Adaptive Thresholding (SAT) mechanism that adaptively adjusts class-specific thresholds based on the model's learning progress. SAT ensures the selection of high-quality pseudo-labeled data, mitigating the risks of inaccurate pseudo-labels and confirmation bias. Extensive experiments demonstrate that SST achieves state-of-the-art performance with remarkable efficiency, generalization, and scalability across various architectures and datasets. Semi-SST-ViT-Huge achieves the best results on competitive ImageNet-1K SSL benchmarks, with 80.7% / 84.9% Top-1 accuracy using only 1% / 10% labeled data. Compared to the fully-supervised DeiT-III-ViT-Huge, which achieves 84.8% Top-1 accuracy using 100% labeled data, our method demonstrates superior performance using only 10% labeled data. 

**Abstract (ZH)**: 基于自适应阈值的自我训练（SST）：一种高效有效的半监督学习框架 

---
# XMAD-Bench: Cross-Domain Multilingual Audio Deepfake Benchmark 

**Title (ZH)**: XMAD-Bench: 跨域多语言音频换脸 benchmark 

**Authors**: Ioan-Paul Ciobanu, Andrei-Iulian Hiji, Nicolae-Catalin Ristea, Paul Irofti, Cristian Rusu, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00462)  

**Abstract**: Recent advances in audio generation led to an increasing number of deepfakes, making the general public more vulnerable to financial scams, identity theft, and misinformation. Audio deepfake detectors promise to alleviate this issue, with many recent studies reporting accuracy rates close to 99%. However, these methods are typically tested in an in-domain setup, where the deepfake samples from the training and test sets are produced by the same generative models. To this end, we introduce XMAD-Bench, a large-scale cross-domain multilingual audio deepfake benchmark comprising 668.8 hours of real and deepfake speech. In our novel dataset, the speakers, the generative methods, and the real audio sources are distinct across training and test splits. This leads to a challenging cross-domain evaluation setup, where audio deepfake detectors can be tested ``in the wild''. Our in-domain and cross-domain experiments indicate a clear disparity between the in-domain performance of deepfake detectors, which is usually as high as 100%, and the cross-domain performance of the same models, which is sometimes similar to random chance. Our benchmark highlights the need for the development of robust audio deepfake detectors, which maintain their generalization capacity across different languages, speakers, generative methods, and data sources. Our benchmark is publicly released at this https URL. 

**Abstract (ZH)**: 近期音频生成技术的发展导致了音频合成样本的增多，使得普通公众更容易成为金融诈骗、身份盗用和虚假信息的受害者。音频合成检测器有望缓解这一问题，多项最近的研究报道其准确率接近99%。然而，这些方法通常在同域设置下进行测试，即训练集和测试集中的合成样本由相同的生成模型生成。为此，我们提出了一种大规模跨域多语言音频合成检测基准XMAD-Bench，该基准包含668.8小时的真实和合成语音。在我们的新数据集中，训练集和测试集中的说话人、生成方法和真实音频源均不相同。这导致了一种具有挑战性的跨域评估设置，使得音频合成检测器能在真实环境中进行测试。我们的同域和跨域实验表明，合成检测器在同域的性能通常高达100%，而在跨域下的性能有时甚至类似于随机猜测。该基准突显了开发能够在不同语言、说话人、生成方法和数据源下保持泛化能力的鲁棒音频合成检测器的需求。该基准已经在以下链接公开发布：this https URL。 

---
# Comparing Traditional and Reinforcement-Learning Methods for Energy Storage Control 

**Title (ZH)**: 传统方法与强化学习方法在储能控制中的比较 

**Authors**: Elinor Ginzburg, Itay Segev, Yoash Levron, Sarah Keren  

**Link**: [PDF](https://arxiv.org/pdf/2506.00459)  

**Abstract**: We aim to better understand the tradeoffs between traditional and reinforcement learning (RL) approaches for energy storage management. More specifically, we wish to better understand the performance loss incurred when using a generative RL policy instead of using a traditional approach to find optimal control policies for specific instances. Our comparison is based on a simplified micro-grid model, that includes a load component, a photovoltaic source, and a storage device. Based on this model, we examine three use cases of increasing complexity: ideal storage with convex cost functions, lossy storage devices, and lossy storage devices with convex transmission losses. With the aim of promoting the principled use RL based methods in this challenging and important domain, we provide a detailed formulation of each use case and a detailed description of the optimization challenges. We then compare the performance of traditional and RL methods, discuss settings in which it is beneficial to use each method, and suggest avenues for future investigation. 

**Abstract (ZH)**: 我们旨在更好地理解传统方法与强化学习（RL）方法在储能管理中的权衡。具体而言，我们希望更好地理解当使用生成性RL策略而非传统方法寻找特定实例的最佳控制策略时所付出的性能损失。我们的比较基于一个简化的微网模型，该模型包括负载组件、光伏源和储能设备。基于此模型，我们探讨了三种逐步复杂的使用案例：理想的具有凸成本函数的储能、有损耗的储能设备，以及具有凸传输损耗的有损耗储能设备。为了促进在这一具有挑战性和重要性的领域中合理使用基于RL的方法，我们提供了每个使用案例的详细建模和优化挑战的详细描述。然后，我们比较了传统方法和RL方法的性能，讨论了各自使用有益的设置，并提出了未来研究的方向。 

---
# TMetaNet: Topological Meta-Learning Framework for Dynamic Link Prediction 

**Title (ZH)**: TMetaNet: 杆度拓扑元学习框架用于动态链路预测 

**Authors**: Hao Li, Hao Wan, Yuzhou Chen, Dongsheng Ye, Yulia Gel, Hao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00453)  

**Abstract**: Dynamic graphs evolve continuously, presenting challenges for traditional graph learning due to their changing structures and temporal dependencies. Recent advancements have shown potential in addressing these challenges by developing suitable meta-learning-based dynamic graph neural network models. However, most meta-learning approaches for dynamic graphs rely on fixed weight update parameters, neglecting the essential intrinsic complex high-order topological information of dynamically evolving graphs. We have designed Dowker Zigzag Persistence (DZP), an efficient and stable dynamic graph persistent homology representation method based on Dowker complex and zigzag persistence, to capture the high-order features of dynamic graphs. Armed with the DZP ideas, we propose TMetaNet, a new meta-learning parameter update model based on dynamic topological features. By utilizing the distances between high-order topological features, TMetaNet enables more effective adaptation across snapshots. Experiments on real-world datasets demonstrate TMetaNet's state-of-the-art performance and resilience to graph noise, illustrating its high potential for meta-learning and dynamic graph analysis. Our code is available at this https URL. 

**Abstract (ZH)**: 动态图持续演变，给传统的图学习带来了挑战，因为它们的结构和时序依赖性不断变化。近期的研究表明，通过开发适合的基于元学习的动态图神经网络模型，有可能解决这些挑战。然而，大多数针对动态图的元学习方法依赖于固定权重更新参数，忽略了动态演变图中本质的复杂的高阶拓扑信息。我们设计了基于Dowker复形和zigzag持久性的Dowker Zigzag Persistence（DZP）方法，一种高效的动态图持久同调表示方法，用于捕捉动态图的高阶特征。利用DZP的理念，我们提出了TMetaNet，一种基于动态拓扑特征的元学习参数更新模型。通过利用高阶拓扑特征之间的距离，TMetaNet能够实现更有效的跨快照适配。实验证明，TMetaNet在实际数据集上的性能处于领先地位，并且对图噪声具有很高的鲁棒性，展示了其在元学习和动态图分析中的高潜力。代码已发布在以下链接：this https URL。 

---
# Attention-Aided MMSE for OFDM Channel Estimation: Learning Linear Filters with Attention 

**Title (ZH)**: 基于注意力辅助的MMSE OFDM信道估计：学习具有注意力机制的线性滤波器 

**Authors**: TaeJun Ha, Chaehyun Jung, Hyeonuk Kim, Jeongwoo Park, Jeonghun Park  

**Link**: [PDF](https://arxiv.org/pdf/2506.00452)  

**Abstract**: In orthogonal frequency division multiplexing (OFDM), accurate channel estimation is crucial. Classical signal processing based approaches, such as minimum mean-squared error (MMSE) estimation, often require second-order statistics that are difficult to obtain in practice. Recent deep neural networks based methods have been introduced to address this; yet they often suffer from high complexity. This paper proposes an Attention-aided MMSE (A-MMSE), a novel model-based DNN framework that learns the optimal MMSE filter via the Attention Transformer. Once trained, the A-MMSE estimates the channel through a single linear operation for channel estimation, eliminating nonlinear activations during inference and thus reducing computational complexity. To enhance the learning efficiency of the A-MMSE, we develop a two-stage Attention encoder, designed to effectively capture the channel correlation structure. Additionally, a rank-adaptive extension of the proposed A-MMSE allows flexible trade-offs between complexity and channel estimation accuracy. Extensive simulations with 3GPP TDL channel models demonstrate that the proposed A-MMSE consistently outperforms other baseline methods in terms of normalized MSE across a wide range of SNR conditions. In particular, the A-MMSE and its rank-adaptive extension establish a new frontier in the performance complexity trade-off, redefining the standard for practical channel estimation methods. 

**Abstract (ZH)**: 基于注意力辅助最小均方误差的 OFDM 信道估计方法 

---
# Is Your Explanation Reliable: Confidence-Aware Explanation on Graph Neural Networks 

**Title (ZH)**: 你的解释可靠吗：图神经网络中的置信意识解释 

**Authors**: Jiaxing Zhang, Xiaoou Liu, Dongsheng Luo, Hua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2506.00437)  

**Abstract**: Explaining Graph Neural Networks (GNNs) has garnered significant attention due to the need for interpretability, enabling users to understand the behavior of these black-box models better and extract valuable insights from their predictions. While numerous post-hoc instance-level explanation methods have been proposed to interpret GNN predictions, the reliability of these explanations remains uncertain, particularly in the out-of-distribution or unknown test datasets. In this paper, we address this challenge by introducing an explainer framework with the confidence scoring module ( ConfExplainer), grounded in theoretical principle, which is generalized graph information bottleneck with confidence constraint (GIB-CC), that quantifies the reliability of generated explanations. Experimental results demonstrate the superiority of our approach, highlighting the effectiveness of the confidence score in enhancing the trustworthiness and robustness of GNN explanations. 

**Abstract (ZH)**: 解释图神经网络（GNN）因其可解释性需求而引起了广泛关注，这使用户能够更好地理解这些黑盒模型的行为并从其预测中提取有价值的见解。尽管已经提出了许多事后实例级解释方法来解释GNN预测，但这些解释的可靠性仍然存疑，尤其是在分布外或未知测试数据集中。在本文中，我们通过引入一个基于理论原理并带有置信度评分模块（ConfExplainer）的解释框架来应对这一挑战，该框架是广义图信息瓶颈与置信度约束（GIB-CC）的理论概括，量化了生成解释的可靠性。实验结果表明了我们方法的优势，突显了置信度评分在提高GNN解释的信任度和鲁棒性方面的有效性。 

---
# Learning from Double Positive and Unlabeled Data for Potential-Customer Identification 

**Title (ZH)**: 基于双正样本和未标注数据的学习方法及其在潜在客户识别中的应用 

**Authors**: Masahiro Kato, Yuki Ikeda abd Kentaro Baba, Takashi Imai, Ryo Inokuchi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00436)  

**Abstract**: In this study, we propose a method for identifying potential customers in targeted marketing by applying learning from positive and unlabeled data (PU learning). We consider a scenario in which a company sells a product and can observe only the customers who purchased it. Decision-makers seek to market products effectively based on whether people have loyalty to the company. Individuals with loyalty are those who are likely to remain interested in the company even without additional advertising. Consequently, those loyal customers would likely purchase from the company if they are interested in the product. In contrast, people with lower loyalty may overlook the product or buy similar products from other companies unless they receive marketing attention. Therefore, by focusing marketing efforts on individuals who are interested in the product but do not have strong loyalty, we can achieve more efficient marketing. To achieve this goal, we consider how to learn, from limited data, a classifier that identifies potential customers who (i) have interest in the product and (ii) do not have loyalty to the company. Although our algorithm comprises a single-stage optimization, its objective function implicitly contains two losses derived from standard PU learning settings. For this reason, we refer to our approach as double PU learning. We verify the validity of the proposed algorithm through numerical experiments, confirming that it functions appropriately for the problem at hand. 

**Abstract (ZH)**: 基于正例和未标注数据双正例学习的潜在客户识别方法研究 

---
# Channel Normalization for Time Series Channel Identification 

**Title (ZH)**: 时间序列通道识别中的通道归一化 

**Authors**: Seunghan Lee, Taeyoung Park, Kibok Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.00432)  

**Abstract**: Channel identifiability (CID) refers to the ability to distinguish between individual channels in time series (TS) modeling. The absence of CID often results in producing identical outputs for identical inputs, disregarding channel-specific characteristics. In this paper, we highlight the importance of CID and propose Channel Normalization (CN), a simple yet effective normalization strategy that enhances CID by assigning distinct affine transformation parameters to each channel. We further extend CN in two ways: 1) Adaptive CN (ACN) dynamically adjusts parameters based on the input TS, improving adaptability in TS models, and 2) Prototypical CN (PCN) introduces a set of learnable prototypes instead of per-channel parameters, enabling applicability to datasets with unknown or varying number of channels and facilitating use in TS foundation models. We demonstrate the effectiveness of CN and its variants by applying them to various TS models, achieving significant performance gains for both non-CID and CID models. In addition, we analyze the success of our approach from an information theory perspective. Code is available at this https URL. 

**Abstract (ZH)**: 信道可分辨性（CID）指的是在时间序列（TS）建模中区分个体信道的能力。CID的缺失常常导致在给定相同输入时产生相同输出，忽略信道特异性特征。本文强调了CID的重要性，并提出了一种简单有效的正则化策略——信道正则化（CN），通过为每个信道分配独特的仿射变换参数来增强CID。我们进一步以两种方式扩展了CN：1）自适应信道正则化（ACN）根据输入TS动态调整参数，增强TS模型的适应性；2）原型信道正则化（PCN）引入了一组可学习的原型，而不是信道特定参数，使其适用于具有未知或变化数量信道的数据集，并便于TS基础模型的应用。我们通过将其应用于多种TS模型，展示了CN及其变体的有效性，实现了非CID模型和CID模型的重大性能提升。此外，我们从信息论的角度分析了我们方法的成功。代码详见this https URL。 

---
# COGNATE: Acceleration of Sparse Tensor Programs on Emerging Hardware using Transfer Learning 

**Title (ZH)**: COGNATE：利用迁移学习加速新兴硬件上的稀疏张量程序 

**Authors**: Chamika Sudusinghe, Gerasimos Gerogiannis Damitha Lenadora, Charles Block, Josep Torrellas, Charith Mendis  

**Link**: [PDF](https://arxiv.org/pdf/2506.00424)  

**Abstract**: Sparse tensor programs are essential in deep learning and graph analytics, driving the need for optimized processing. To meet this demand, specialized hardware accelerators are being developed. Optimizing these programs for accelerators is challenging for two reasons: program performance is highly sensitive to variations in sparse inputs, and early-stage accelerators rely on expensive simulators. Therefore, ML-based cost models used for optimizing such programs on general-purpose hardware are often ineffective for early-stage accelerators, as they require large datasets for proper training. To this end, we introduce COGNATE, a novel framework that leverages inexpensive data samples from general-purpose hardware (e.g., CPUs) to train cost models, followed by few-shot fine-tuning on emerging hardware. COGNATE exploits the homogeneity of input features across hardware platforms while effectively mitigating heterogeneity, enabling cost model training with just 5% of the data samples needed by accelerator-specific models to achieve comparable performance. We conduct extensive experiments to demonstrate that COGNATE outperforms existing techniques, achieving average speedups of 1.47x (up to 5.46x) for SpMM and 1.39x (up to 4.22x) for SDDMM. 

**Abstract (ZH)**: COGNATE：利用通用硬件数据样本进行高效稀疏张量程序成本模型训练的新型框架 

---
# A New Spatiotemporal Correlation Anomaly Detection Method that Integrates Contrastive Learning and Few-Shot Learning in Wireless Sensor Networks 

**Title (ZH)**: 一种结合对比学习和少样本学习的无线传感器网络时空相关异常检测新方法 

**Authors**: Miao Ye, Suxiao Wang, Jiaguang Han, Yong Wang, Xiaoli Wang, Jingxuan Wei, Peng Wen, Jing Cui  

**Link**: [PDF](https://arxiv.org/pdf/2506.00420)  

**Abstract**: Detecting anomalies in the data collected by WSNs can provide crucial evidence for assessing the reliability and stability of WSNs. Existing methods for WSN anomaly detection often face challenges such as the limited extraction of spatiotemporal correlation features, the absence of sample labels, few anomaly samples, and an imbalanced sample distribution. To address these issues, a spatiotemporal correlation detection model (MTAD-RD) considering both model architecture and a two-stage training strategy perspective is proposed. In terms of model structure design, the proposed MTAD-RD backbone network includes a retentive network (RetNet) enhanced by a cross-retention (CR) module, a multigranular feature fusion module, and a graph attention network module to extract internode correlation information. This proposed model can integrate the intermodal correlation features and spatial features of WSN neighbor nodes while extracting global information from time series data. Moreover, its serialized inference characteristic can remarkably reduce inference overhead. For model training, a two-stage training approach was designed. First, a contrastive learning proxy task was designed for time series data with graph structure information in WSNs, enabling the backbone network to learn transferable features from unlabeled data using unsupervised contrastive learning methods, thereby addressing the issue of missing sample labels in the dataset. Then, a caching-based sample sampler was designed to divide samples into few-shot and contrastive learning data. A specific joint loss function was developed to jointly train the dual-graph discriminator network to address the problem of sample imbalance effectively. In experiments carried out on real public datasets, the designed MTAD-RD anomaly detection method achieved an F1 score of 90.97%, outperforming existing supervised WSN anomaly detection methods. 

**Abstract (ZH)**: 基于时空关联检测的WSN异常检测方法（MTAD-RD） 

---
# Bias as a Virtue: Rethinking Generalization under Distribution Shifts 

**Title (ZH)**: 偏见作为一种美德：在分布偏移情况下的重新思考泛化能力 

**Authors**: Ruixuan Chen, Wentao Li, Jiahui Xiao, Yuchen Li, Yimin Tang, Xiaonan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00407)  

**Abstract**: Machine learning models often degrade when deployed on data distributions different from their training data. Challenging conventional validation paradigms, we demonstrate that higher in-distribution (ID) bias can lead to better out-of-distribution (OOD) generalization. Our Adaptive Distribution Bridge (ADB) framework implements this insight by introducing controlled statistical diversity during training, enabling models to develop bias profiles that effectively generalize across distributions. Empirically, we observe a robust negative correlation where higher ID bias corresponds to lower OOD error--a finding that contradicts standard practices focused on minimizing validation error. Evaluation on multiple datasets shows our approach significantly improves OOD generalization. ADB achieves robust mean error reductions of up to 26.8% compared to traditional cross-validation, and consistently identifies high-performing training strategies, evidenced by percentile ranks often exceeding 74.4%. Our work provides both a practical method for improving generalization and a theoretical framework for reconsidering the role of bias in robust machine learning. 

**Abstract (ZH)**: 机器学习模型在部署于与训练数据不同的数据分布时往往会退化。挑战传统的验证范式，我们证明了较高的同分布（ID）偏差可以导致更好的异分布（OOD）泛化。我们的自适应分布桥（ADB）框架通过在训练过程中引入受控的统计多样性来实现这一洞察，使模型能够发展出有效地跨越分布进行泛化的偏差配置。实证研究表明，较高的ID偏差与较低的OOD误差之间存在稳健的负相关关系——这一发现与关注于最小化验证误差的常规做法相矛盾。在多个数据集上的评估结果显示，我们的方法显著提高了异分布泛化能力。ADB实现了与传统交叉验证相比高达26.8%的稳健均值误差减少，并且一贯地识别出高性能的训练策略，证据表明百分位排名通常超过了74.4%。我们的工作既提供了一种改进泛化的实用方法，也提供了一个重新考虑偏差在稳健机器学习中作用的理论框架。 

---
# MagiCodec: Simple Masked Gaussian-Injected Codec for High-Fidelity Reconstruction and Generation 

**Title (ZH)**: MagiCodec：简单的遮掩高斯注入编解码器，用于高保真重建与生成 

**Authors**: Yakun Song, Jiawei Chen, Xiaobin Zhuang, Chenpeng Du, Ziyang Ma, Jian Wu, Jian Cong, Dongya Jia, Zhuo Chen, Yuping Wang, Yuxuan Wang, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00385)  

**Abstract**: Neural audio codecs have made significant strides in efficiently mapping raw audio waveforms into discrete token representations, which are foundational for contemporary audio generative models. However, most existing codecs are optimized primarily for reconstruction quality, often at the expense of the downstream modelability of the encoded tokens. Motivated by the need to overcome this bottleneck, we introduce $\textbf{MagiCodec}$, a novel single-layer, streaming Transformer-based audio codec. MagiCodec is designed with a multistage training pipeline that incorporates Gaussian noise injection and latent regularization, explicitly targeting the enhancement of semantic expressiveness in the generated codes while preserving high reconstruction fidelity. We analytically derive the effect of noise injection in the frequency domain, demonstrating its efficacy in attenuating high-frequency components and fostering robust tokenization. Extensive experimental evaluations show that MagiCodec surpasses state-of-the-art codecs in both reconstruction quality and downstream tasks. Notably, the tokens produced by MagiCodec exhibit Zipf-like distributions, as observed in natural languages, thereby improving compatibility with language-model-based generative architectures. The code and pre-trained models are available at this https URL. 

**Abstract (ZH)**: 神经音频编解码器已经在高效地将原始音频波形映射为离散符号表示方面取得了显著进展，这些表示是当代音频生成模型的基础。然而，现有的大多数编解码器主要优化重建质量，往往以牺牲编码符号的下游模型能力强度为代价。为了克服这一瓶颈，我们介绍了名为MagiCodec的新型单层流式Transformer基音频编解码器。MagiCodec采用一个多阶段训练管道，结合了高斯噪声注入和潜在正则化，明确地旨在增强生成代码的语义表达能力，同时保持高重建保真度。我们在频域中分析了噪声注入的效果，证实了其在衰减高频分量并促进稳健符号化的有效性。广泛的实验评估表明，MagiCodec在重建质量和下游任务方面均优于最先进的编解码器。值得注意的是，MagiCodec生成的符号表现出类似Zipf的分布，类似于自然语言，从而提高了与基于语言模型的生成架构的兼容性。代码和预训练模型可在以下链接获取。 

---
# Neural Network-based Information-Theoretic Transceivers for High-Order Modulation Schemes 

**Title (ZH)**: 基于神经网络的信息论传输机高阶调制方案 

**Authors**: Ngoc Long Pham, Tri Nhu Do  

**Link**: [PDF](https://arxiv.org/pdf/2506.00368)  

**Abstract**: Neural network (NN)-based end-to-end (E2E) communication systems, in which each system component may consist of a portion of a neural network, have been investigated as potential tools for developing artificial intelligence (Al)-native E2E systems. In this paper, we propose an NN-based bitwise receiver that improves computational efficiency while maintaining performance comparable to baseline demappers. Building on this foundation, we introduce a novel symbol-wise autoencoder (AE)-based E2E system that jointly optimizes the transmitter and receiver at the physical layer. We evaluate the proposed NN-based receiver using bit-error rate (BER) analysis to confirm that the numerical BER achieved by NN-based receivers or transceivers is accurate. Results demonstrate that the AE-based system outperforms baseline architectures, particularly for higher-order modulation schemes. We further show that the training signal-to-noise ratio (SNR) significantly affects the performance of the systems when inference is conducted at different SNR levels. 

**Abstract (ZH)**: 基于神经网络的端到端通信系统中，每个系统组件可能包括神经网络的部分，已被探索作为开发人工智能原生端到端系统的潜在工具。在本文中，我们提出了一种基于神经网络的位级接收机，该接收机在保持与基准译码器相当的性能的同时提高了计算效率。在此基础上，我们引入了一种基于符号级自动编码器的端到端系统，该系统在物理层上联合优化了发送端和接收端。我们通过位误比特率(BER)分析评估所提出的基于神经网络的接收机，以确认基于神经网络的接收机或收发机实现的数值BER是准确的。结果表明，基于自动编码器的系统在高阶调制方案中性能优于基准架构。此外，我们还展示了训练信号噪声比(SNR)在不同SNR水平下进行推理时对系统性能的影响显著。 

---
# Exploring the Performance of Perforated Backpropagation through Further Experiments 

**Title (ZH)**: 探索穿透反向传播性能的进一步实验 

**Authors**: Rorry Brenner, Evan Davis, Rushi Chaudhari, Rowan Morse, Jingyao Chen, Xirui Liu, Zhaoyi You, Laurent Itti  

**Link**: [PDF](https://arxiv.org/pdf/2506.00356)  

**Abstract**: Perforated Backpropagation is a neural network optimization technique based on modern understanding of the computational importance of dendrites within biological neurons. This paper explores further experiments from the original publication, generated from a hackathon held at the Carnegie Mellon Swartz Center in February 2025. Students and local Pittsburgh ML practitioners were brought together to experiment with the Perforated Backpropagation algorithm on the datasets and models which they were using for their projects. Results showed that the system could enhance their projects, with up to 90% model compression without negative impact on accuracy, or up to 16% increased accuracy of their original models. 

**Abstract (ZH)**: 穿孔反向传播是一种基于对生物神经元树突计算重要性现代理解的神经网络优化技术。本文进一步探索了2025年2月在卡内基梅隆斯瓦兹中心举办的黑客马拉松中最初发表论文生成的实验。学生和当地匹兹堡的机器学习从业者共同实验了穿孔反向传播算法对其项目中使用的数据集和模型。结果显示，该系统能够提升其项目性能，最高可达90%的模型压缩比例而不影响准确性，或者在原有模型基础上提高16%的准确率。 

---
# Enabling Secure and Ephemeral AI Workloads in Data Mesh Environments 

**Title (ZH)**: 在数据网状环境中的安全且临时的AI工作负载启用 

**Authors**: Chinkit Patel, Kee Siong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00352)  

**Abstract**: Many large enterprises that operate highly governed and complex ICT environments have no efficient and effective way to support their Data and AI teams in rapidly spinning up and tearing down self-service data and compute infrastructure, to experiment with new data analytic tools, and deploy data products into operational use. This paper proposes a key piece of the solution to the overall problem, in the form of an on-demand self-service data-platform infrastructure to empower de-centralised data teams to build data products on top of centralised templates, policies and governance. The core innovation is an efficient method to leverage immutable container operating systems and infrastructure-as-code methodologies for creating, from scratch, vendor-neutral and short-lived Kubernetes clusters on-premises and in any cloud environment. Our proposed approach can serve as a repeatable, portable and cost-efficient alternative or complement to commercial Platform-as-a-Service (PaaS) offerings, and this is particularly important in supporting interoperability in complex data mesh environments with a mix of modern and legacy compute infrastructure. 

**Abstract (ZH)**: 许多运营高度管控和复杂 ICT 环境的大型企业缺乏高效有效的方法来支持其数据和 AI 团队快速搭建和销毁自助服务数据和计算基础设施、实验新的数据分析工具以及将数据产品部署到生产环境中。本文提出了整体解决方案的关键组成部分，即一种按需自助的数据平台基础设施，以此赋能分散的数据团队在中央模板、政策和治理的基础上构建数据产品。核心创新是一种高效的方法，利用不可变容器操作系统和基础设施即代码方法，在任何本地或云环境从零开始创建中立的、短暂的 Kubernetes 集群。我们提出的这种方法可以作为商业平台即服务（PaaS）产品的可重复、可移植且成本效益高的替代方案或补充，特别是在支持混合现代和遗留计算基础设施的复杂数据湖环境中尤为重要。 

---
# Beyond Winning: Margin of Victory Relative to Expectation Unlocks Accurate Skill Ratings 

**Title (ZH)**: 超越胜利：相对于期望的获胜优势解锁了准确的技能评级 

**Authors**: Shivam Shorewala, Zihao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00348)  

**Abstract**: Knowledge of accurate relative skills in any competitive system is essential, but foundational approaches such as ELO discard extremely relevant performance data by concentrating exclusively on binary outcomes. While margin of victory (MOV) extensions exist, they often lack a definitive method for incorporating this information. We introduce Margin of Victory Differential Analysis (MOVDA), a framework that enhances traditional rating systems by using the deviation between the true MOV and a $\textit{modeled expectation}$. MOVDA learns a domain-specific, non-linear function (a scaled hyperbolic tangent that captures saturation effects and home advantage) to predict expected MOV based on rating differentials. Crucially, the $\textit{difference}$ between the true and expected MOV provides a subtle and weighted signal for rating updates, highlighting informative deviations in all levels of contests. Extensive experiments on professional NBA basketball data (from 2013 to 2023, with 13,619 games) show that MOVDA significantly outperforms standard ELO and Bayesian baselines. MOVDA reduces Brier score prediction error by $1.54\%$ compared to TrueSkill, increases outcome accuracy by $0.58\%$, and most importantly accelerates rating convergence by $13.5\%$, while maintaining the computational efficiency of the original ELO updates. MOVDA offers a theoretically motivated, empirically superior, and computationally lean approach to integrating performance magnitude into skill rating for competitive environments like the NBA. 

**Abstract (ZH)**: 基于胜利 margin 的差异分析在提升竞技系统技能评级中的应用 

---
# Recover Experimental Data with Selection Bias using Counterfactual Logic 

**Title (ZH)**: 使用反事实逻辑恢复具有选择偏见的实验数据 

**Authors**: Jingyang He, Shuai Wang, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.00335)  

**Abstract**: Selection bias, arising from the systematic inclusion or exclusion of certain samples, poses a significant challenge to the validity of causal inference. While Bareinboim et al. introduced methods for recovering unbiased observational and interventional distributions from biased data using partial external information, the complexity of the backdoor adjustment and the method's strong reliance on observational data limit its applicability in many practical settings. In this paper, we formally discover the recoverability of $P(Y^*_{x^*})$ under selection bias with experimental data. By explicitly constructing counterfactual worlds via Structural Causal Models (SCMs), we analyze how selection mechanisms in the observational world propagate to the counterfactual domain. We derive a complete set of graphical and theoretical criteria to determine that the experimental distribution remain unaffected by selection bias. Furthermore, we propose principled methods for leveraging partially unbiased observational data to recover $P(Y^*_{x^*})$ from biased experimental datasets. Simulation studies replicating realistic research scenarios demonstrate the practical utility of our approach, offering concrete guidance for mitigating selection bias in applied causal inference. 

**Abstract (ZH)**: 选择偏差导致的系统性样本包括或排除问题对因果推理的有效性构成了重大挑战。尽管Bareinboim等人引入了利用部分外部信息从偏差数据中恢复无偏观察和干预分布的方法，但后门调整的复杂性以及该方法对观察数据的高依赖性限制了其在许多实际情境中的适用性。本文正式揭示了通过实验数据在选择偏差下恢复 $P(Y^*_{x^*})$ 的可行性。通过结构因果模型（SCMs）明确构建反事实世界，我们分析了观察世界中的选择机制如何传播到反事实领域。我们推导出一套完整的图形和理论标准，以确定实验分布不受到选择偏差的影响。此外，我们提出了利用部分无偏观察数据恢复偏差实验数据集中 $P(Y^*_{x^*})$ 的原则方法。模拟研究表明，本方法在实际因果推理中的实用价值，提供了具体指导以减轻应用中的选择偏差。 

---
# Latent Guidance in Diffusion Models for Perceptual Evaluations 

**Title (ZH)**: 扩散模型中潜导引的感知评估 

**Authors**: Shreshth Saini, Ru-Ling Liao, Yan Ye, Alan C. Bovik  

**Link**: [PDF](https://arxiv.org/pdf/2506.00327)  

**Abstract**: Despite recent advancements in latent diffusion models that generate high-dimensional image data and perform various downstream tasks, there has been little exploration into perceptual consistency within these models on the task of No-Reference Image Quality Assessment (NR-IQA). In this paper, we hypothesize that latent diffusion models implicitly exhibit perceptually consistent local regions within the data manifold. We leverage this insight to guide on-manifold sampling using perceptual features and input measurements. Specifically, we propose Perceptual Manifold Guidance (PMG), an algorithm that utilizes pretrained latent diffusion models and perceptual quality features to obtain perceptually consistent multi-scale and multi-timestep feature maps from the denoising U-Net. We empirically demonstrate that these hyperfeatures exhibit high correlation with human perception in IQA tasks. Our method can be applied to any existing pretrained latent diffusion model and is straightforward to integrate. To the best of our knowledge, this paper is the first work on guiding diffusion model with perceptual features for NR-IQA. Extensive experiments on IQA datasets show that our method, LGDM, achieves state-of-the-art performance, underscoring the superior generalization capabilities of diffusion models for NR-IQA tasks. 

**Abstract (ZH)**: 尽管近期在生成高维图像数据并执行各种下游任务方面取得了进展的潜在扩散模型已取得显著成就，但在无参考图像质量评估（NR-IQA）任务中这些模型的感知一致性方面研究较少。在本文中，我们假设潜在扩散模型隐式地在数据流形中表现出感知一致的局部区域。我们利用这一洞察，通过感知特征和输入测量引导流形上的采样。具体而言，我们提出了感知流形引导（PMG）算法，该算法利用预训练的潜在扩散模型和感知质量特征，从去噪的UNet中获得多尺度和多时间步的感知一致性特征图。我们实证证明了这些超特征在IQA任务中与人类感知具有高度相关性。该方法可以应用于任何现有的预训练潜在扩散模型，并且易于集成。据我们所知，这是首次利用感知特征引导扩散模型进行NR-IQA的研究。在IQA数据集上的广泛实验表明，我们的方法LGDM达到了最先进的性能，突显了扩散模型在NR-IQA任务中的出色泛化能力。 

---
# dpmm: Differentially Private Marginal Models, a Library for Synthetic Tabular Data Generation 

**Title (ZH)**: DPMM：不同质化边缘模型，一个合成表格数据生成库 

**Authors**: Sofiane Mahiou, Amir Dizche, Reza Nazari, Xinmin Wu, Ralph Abbey, Jorge Silva, Georgi Ganev  

**Link**: [PDF](https://arxiv.org/pdf/2506.00322)  

**Abstract**: We propose dpmm, an open-source library for synthetic data generation with Differentially Private (DP) guarantees. It includes three popular marginal models -- PrivBayes, MST, and AIM -- that achieve superior utility and offer richer functionality compared to alternative implementations. Additionally, we adopt best practices to provide end-to-end DP guarantees and address well-known DP-related vulnerabilities. Our goal is to accommodate a wide audience with easy-to-install, highly customizable, and robust model implementations.
Our codebase is available from this https URL. 

**Abstract (ZH)**: 我们提出dpmm，一个具有差分隐私保障的合成数据生成开源库。它包括三种流行的边际模型——PrivBayes、MST和AIM，与替代实现相比，提供了更好的效用和更丰富的功能。此外，我们采用了最佳实践来提供端到端的差分隐私保障，并解决了一些已知的差分隐私相关漏洞。我们的目标是为广泛受众提供易于安装、高度可定制且稳健的模型实现。

我们的代码库可以从以下链接获取：这个https URL。 

---
# MythTriage: Scalable Detection of Opioid Use Disorder Myths on a Video-Sharing Platform 

**Title (ZH)**: MythTriage：大规模检测视频分享平台上阿片使用障碍错误认知的方法 

**Authors**: Hayoung Jung, Shravika Mittal, Ananya Aatreya, Navreet Kaur, Munmun De Choudhury, Tanushree Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2506.00308)  

**Abstract**: Understanding the prevalence of misinformation in health topics online can inform public health policies and interventions. However, measuring such misinformation at scale remains a challenge, particularly for high-stakes but understudied topics like opioid-use disorder (OUD)--a leading cause of death in the U.S. We present the first large-scale study of OUD-related myths on YouTube, a widely-used platform for health information. With clinical experts, we validate 8 pervasive myths and release an expert-labeled video dataset. To scale labeling, we introduce MythTriage, an efficient triage pipeline that uses a lightweight model for routine cases and defers harder ones to a high-performing, but costlier, large language model (LLM). MythTriage achieves up to 0.86 macro F1-score while estimated to reduce annotation time and financial cost by over 76% compared to experts and full LLM labeling. We analyze 2.9K search results and 343K recommendations, uncovering how myths persist on YouTube and offering actionable insights for public health and platform moderation. 

**Abstract (ZH)**: 在线健康话题中的 misinformation 的普遍性理解对于公共卫生政策和干预具有指导意义。然而，大规模测量此类 misinformation 尤其对于高风险但研究不足的话题（如阿片使用障碍(OUD)）仍然是一个挑战。我们首次对 YouTube 上与 OUD 相关的 myths 进行了大规模研究，YouTube 是一个广泛用于获取健康信息的平台。通过临床专家验证了 8 个广泛流传的 myths，并发布了一个专家标注的视频数据集。为了扩大标注规模，我们引入了 MythTriage，一种高效triage 管道，使用轻量级模型处理常规案例，并将更难的案例转交给人工成本更高但性能更好的大语言模型（LLM）。MythTriage 达到了 0.86 的宏 F1 分数，同时估计将注释时间和财务成本降低了超过 76%。我们分析了 2900 个搜索结果和 343000 个建议，揭示了这些 myths 在 YouTube 上如何持续存在，并为公共卫生和平台 moderation 提供了实际的指导建议。 

---
# Improving Protein Sequence Design through Designability Preference Optimization 

**Title (ZH)**: 通过设计能力偏好优化改进蛋白质序列设计 

**Authors**: Fanglei Xue, Andrew Kubaney, Zhichun Guo, Joseph K. Min, Ge Liu, Yi Yang, David Baker  

**Link**: [PDF](https://arxiv.org/pdf/2506.00297)  

**Abstract**: Protein sequence design methods have demonstrated strong performance in sequence generation for de novo protein design. However, as the training objective was sequence recovery, it does not guarantee designability--the likelihood that a designed sequence folds into the desired structure. To bridge this gap, we redefine the training objective by steering sequence generation toward high designability. To do this, we integrate Direct Preference Optimization (DPO), using AlphaFold pLDDT scores as the preference signal, which significantly improves the in silico design success rate. To further refine sequence generation at a finer, residue-level granularity, we introduce Residue-level Designability Preference Optimization (ResiDPO), which applies residue-level structural rewards and decouples optimization across residues. This enables direct improvement in designability while preserving regions that already perform well. Using a curated dataset with residue-level annotations, we fine-tune LigandMPNN with ResiDPO to obtain EnhancedMPNN, which achieves a nearly 3-fold increase in in silico design success rate (from 6.56% to 17.57%) on a challenging enzyme design benchmark. 

**Abstract (ZH)**: 蛋白质序列设计方法在从头蛋白设计的序列生成中展现出了强大的性能。然而，由于训练目标是序列恢复，这并不保证设计性——即设计序列折叠成所需结构的可能性。为弥补这一差距，我们重新定义了训练目标，通过将序列生成引导至高设计性来提高这一目标。为此，我们整合了直接偏好优化（DPO），使用AlphaFold的pLDDT分数作为偏好信号，显著提高了体外设计成功率。为进一步在更精细的残基级别上细化序列生成，我们引入了残基级别设计性偏好优化（ResiDPO），该方法应用了残基级别的结构奖励，并且拆分了对残基的优化。这使得可以直接提高设计性同时保持已经表现良好的区域不变。利用一个带有残基级别注释的定制数据集，我们通过ResiDPO微调LigandMPNN，得到EnhancedMPNN，该模型在一项具有挑战性的酶设计基准测试中，体外设计成功率几乎提高了三倍（从6.56%提高到17.57%）。 

---
# Entropic Risk Optimization in Discounted MDPs: Sample Complexity Bounds with a Generative Model 

**Title (ZH)**: 带生成模型的折现MDP中的熵风险优化：样本复杂性界 

**Authors**: Oliver Mortensen, Mohammad Sadegh Talebi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00286)  

**Abstract**: In this paper we analyze the sample complexities of learning the optimal state-action value function $Q^*$ and an optimal policy $\pi^*$ in a discounted Markov decision process (MDP) where the agent has recursive entropic risk-preferences with risk-parameter $\beta\neq 0$ and where a generative model of the MDP is available. We provide and analyze a simple model based approach which we call model-based risk-sensitive $Q$-value-iteration (MB-RS-QVI) which leads to $(\epsilon,\delta)$-PAC-bounds on $\|Q^*-Q^k\|$, and $\|V^*-V^{\pi_k}\|$ where $Q_k$ is the output of MB-RS-QVI after k iterations and $\pi_k$ is the greedy policy with respect to $Q_k$. Both PAC-bounds have exponential dependence on the effective horizon $\frac{1}{1-\gamma}$ and the strength of this dependence grows with the learners risk-sensitivity $|\beta|$. We also provide two lower bounds which shows that exponential dependence on $|\beta|\frac{1}{1-\gamma}$ is unavoidable in both cases. The lower bounds reveal that the PAC-bounds are both tight in $\varepsilon$ and $\delta$ and that the PAC-bound on $Q$-learning is tight in the number of actions $A$, and that the PAC-bound on policy-learning is nearly tight in $A$. 

**Abstract (ZH)**: 在递归熵风险偏好下折扣马尔可夫决策过程中的最优状态动作值函数和最优策略的学习样本复杂性分析：基于模型的风险敏感$Q$-值迭代及其$(\varepsilon,\delta)$-PAC边界分析 

---
# Hierarchical Level-Wise News Article Clustering via Multilingual Matryoshka Embeddings 

**Title (ZH)**: 基于多语言布林-olds嵌入的分层级联新闻文章聚类 

**Authors**: Hans W. A. Hanley, Zakir Durumeric  

**Link**: [PDF](https://arxiv.org/pdf/2506.00277)  

**Abstract**: Contextual large language model embeddings are increasingly utilized for topic modeling and clustering. However, current methods often scale poorly, rely on opaque similarity metrics, and struggle in multilingual settings. In this work, we present a novel, scalable, interpretable, hierarchical, and multilingual approach to clustering news articles and social media data. To do this, we first train multilingual Matryoshka embeddings that can determine story similarity at varying levels of granularity based on which subset of the dimensions of the embeddings is examined. This embedding model achieves state-of-the-art performance on the SemEval 2022 Task 8 test dataset (Pearson $\rho$ = 0.816). Once trained, we develop an efficient hierarchical clustering algorithm that leverages the hierarchical nature of Matryoshka embeddings to identify unique news stories, narratives, and themes. We conclude by illustrating how our approach can identify and cluster stories, narratives, and overarching themes within real-world news datasets. 

**Abstract (ZH)**: 基于上下文的大型语言模型嵌入在主题建模和聚类中的应用越来越广泛。然而，当前方法往往扩展性差，依赖于不透明的相似性度量，并在多语言环境中表现不佳。在此工作中，我们提出了一种新的、可扩展的、可解释的、层次化的和多语言的新闻文章和社会媒体数据聚类方法。为此，我们首先训练多语言Matryoshka嵌入，可以根据检查嵌入的维度子集来确定故事在不同粒度水平上的相似性。该嵌入模型在SemEval 2022 Task 8测试数据集上达到了最先进性能（皮尔逊相关系数ρ=0.816）。训练完成后，我们开发了一种高效的时间层次聚类算法，利用Matryoshka嵌入的层次结构特征来识别独特的新闻故事、叙述和主题。最后，我们通过实例展示了我们的方法如何在实际新闻数据集中识别和聚类故事、叙述和主题。 

---
# Designing AI Tools for Clinical Care Teams to Support Serious Illness Conversations with Older Adults in the Emergency Department 

**Title (ZH)**: 设计AI工具以支持在急诊部门与老年人讨论严重疾病的相关临床护理团队 

**Authors**: Menglin Zhao, Zhuorui Yong, Ruijia Guan, Kai-Wei Chang, Adrian Haimovich, Kei Ouchi, Timothy Bickmore, Bingsheng Yao, Dakuo Wang, Smit Desai  

**Link**: [PDF](https://arxiv.org/pdf/2506.00241)  

**Abstract**: Serious illness conversations (SICs), discussions between clinical care teams and patients with serious, life-limiting illnesses about their values, goals, and care preferences, are critical for patient-centered care. Without these conversations, patients often receive aggressive interventions that may not align with their goals. Clinical care teams face significant barriers when conducting serious illness conversations with older adult patients in Emergency Department (ED) settings, where most older adult patients lack documented treatment goals. To understand current practices and identify AI support opportunities, we conducted interviews with two domain experts and nine ED clinical care team members. Through thematic analysis, we characterized a four-phase serious illness conversation workflow (identification, preparation, conduction, documentation) and identified key needs and challenges at each stage. Clinical care teams struggle with fragmented EHR data access, time constraints, emotional preparation demands, and documentation burdens. While participants expressed interest in AI tools for information synthesis, conversational support, and automated documentation, they emphasized preserving human connection and clinical autonomy. We present design guidelines for AI tools supporting SIC workflows that fit within existing clinical practices. This work contributes empirical understanding of ED-based serious illness conversations and provides design considerations for AI in high-stakes clinical environments. 

**Abstract (ZH)**: 严重疾病对话（SICs）：临床护理团队与患有严重和生命限制性疾病患者的关于其价值观、目标和护理偏好的讨论对于以患者为中心的护理至关重要。没有这些讨论，患者往往会接受不符合其目标的积极干预措施。在急诊部门（ED）中，由于大多数老年患者的治疗目标缺乏记录，临床护理团队在与老年患者进行严重疾病对话时面临着巨大的障碍。为了了解当前的实践并识别AI支持的机会，我们与两位领域专家和九名急诊临床护理团队成员进行了访谈。通过主题分析，我们描述了一个四阶段的严重疾病对话工作流程（识别、准备、执行、记录），并在每个阶段识别了关键需求和挑战。临床护理团队在访问碎片化的电子健康记录数据、时间限制、情绪准备需求以及记录负担方面遇到了困难。尽管参与者对用于信息综合、对话支持和自动记录的AI工具表示出了兴趣，但他们强调保持人的连接和临床自主权的重要性。我们提出了适应现有临床实践的AI工具支持严重疾病对话工作流程的设计指南。这项工作为急诊部门基于的严重疾病对话提供了实证理解，并为AI在高风险临床环境中的设计考虑提供了参考。 

---
# Localized LoRA: A Structured Low-Rank Approximation for Efficient Fine-Tuning 

**Title (ZH)**: 局部LoRA：一种结构化低秩逼近方法以实现高效的微调 

**Authors**: Babak Barazandeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.00236)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) methods, such as LoRA, offer compact and effective alternatives to full model fine-tuning by introducing low-rank updates to pretrained weights. However, most existing approaches rely on global low-rank structures, which can overlook spatial patterns spread across the parameter space. In this work, we propose Localized LoRA, a generalized framework that models weight updates as a composition of low-rank matrices applied to structured blocks of the weight matrix. This formulation enables dense, localized updates throughout the parameter space-without increasing the total number of trainable parameters. We provide a formal comparison between global, diagonal-local, and fully localized low-rank approximations, and show that our method consistently achieves lower approximation error under matched parameter budgets. Experiments on both synthetic and practical settings demonstrate that Localized LoRA offers a more expressive and adaptable alternative to existing methods, enabling efficient fine-tuning with improved performance. 

**Abstract (ZH)**: 局部LoRA：一种建模权重更新的通用框架，通过在权重矩阵的结构块上应用低秩矩阵实现密集的局部更新 

---
# Diff-SPORT: Diffusion-based Sensor Placement Optimization and Reconstruction of Turbulent flows in urban environments 

**Title (ZH)**: 基于扩散的传感器-placement优化与城市环境湍流流动的重建 

**Authors**: Abhijeet Vishwasrao, Sai Bharath Chandra Gutha, Andres Cremades, Klas Wijk, Aakash Patil, Catherine Gorle, Beverley J McKeon, Hossein Azizpour, Ricardo Vinuesa  

**Link**: [PDF](https://arxiv.org/pdf/2506.00214)  

**Abstract**: Rapid urbanization demands accurate and efficient monitoring of turbulent wind patterns to support air quality, climate resilience and infrastructure design. Traditional sparse reconstruction and sensor placement strategies face major accuracy degradations under practical constraints. Here, we introduce Diff-SPORT, a diffusion-based framework for high-fidelity flow reconstruction and optimal sensor placement in urban environments. Diff-SPORT combines a generative diffusion model with a maximum a posteriori (MAP) inference scheme and a Shapley-value attribution framework to propose a scalable and interpretable solution. Compared to traditional numerical methods, Diff-SPORT achieves significant speedups while maintaining both statistical and instantaneous flow fidelity. Our approach offers a modular, zero-shot alternative to retraining-intensive strategies, supporting fast and reliable urban flow monitoring under extreme sparsity. Diff-SPORT paves the way for integrating generative modeling and explainability in sustainable urban intelligence. 

**Abstract (ZH)**: 快速城市化需求精确和高效的湍流风模式监测以支持空气质量、气候韧性和基础设施设计。传统的稀疏重构和传感器布设策略在实际约束下面临重大准确度下降。我们引入了基于扩散的Diff-SPORT框架，用于城市环境中高保真流场重构和最优传感器布设。Diff-SPORT结合了生成型扩散模型、最大后验概率（MAP）推断方案和Shapley值归因框架，提出了一个可扩展且可解释的解决方案。与传统数值方法相比，Diff-SPORT在保持统计和瞬时流场保真度的同时实现了显著的加速。我们的方法为密集重训练策略提供了模块化的零样本替代方案，在极端稀疏条件下支持快速可靠的都市流场监测。Diff-SPORT为在可持续智慧城市中整合生成型建模和可解释性奠定了基础。 

---
# REIC: RAG-Enhanced Intent Classification at Scale 

**Title (ZH)**: REIC: RAG增强的规模化的意图分类 

**Authors**: Ziji Zhang, Michael Yang, Zhiyu Chen, Yingying Zhuang, Shu-Ting Pi, Qun Liu, Rajashekar Maragoud, Vy Nguyen, Anurag Beniwal  

**Link**: [PDF](https://arxiv.org/pdf/2506.00210)  

**Abstract**: Accurate intent classification is critical for efficient routing in customer service, ensuring customers are connected with the most suitable agents while reducing handling times and operational costs. However, as companies expand their product lines, intent classification faces scalability challenges due to the increasing number of intents and variations in taxonomy across different verticals. In this paper, we introduce REIC, a Retrieval-augmented generation Enhanced Intent Classification approach, which addresses these challenges effectively. REIC leverages retrieval-augmented generation (RAG) to dynamically incorporate relevant knowledge, enabling precise classification without the need for frequent retraining. Through extensive experiments on real-world datasets, we demonstrate that REIC outperforms traditional fine-tuning, zero-shot, and few-shot methods in large-scale customer service settings. Our results highlight its effectiveness in both in-domain and out-of-domain scenarios, demonstrating its potential for real-world deployment in adaptive and large-scale intent classification systems. 

**Abstract (ZH)**: 准确的意图分类对于客户服务中的高效路由至关重要，确保客户能够与最合适的代理人员对接，同时减少处理时间和运营成本。然而，随着公司产品线的扩展，由于不同垂直领域意图数量的增加和分类 taxonomy 的变化，意图分类面临可扩展性挑战。本文介绍了一种名为 REIC 的检索增强生成增强意图分类方法，该方法有效地应对了这些挑战。REIC 利用检索增强生成 (RAG) 动态地整合相关知识，实现精确分类，无需频繁重新训练。通过在真实数据集上的广泛实验，我们证明了 REIC 在大规模客户服务场景中优于传统的微调、零样本和少样本方法。我们的结果强调了其在领域内和领域外场景中的有效性，展示了其在适应性和大规模意图分类系统中实际部署的潜力。 

---
# Heterogeneous Graph Backdoor Attack 

**Title (ZH)**: 异质图后门攻击 

**Authors**: Jiawei Chen, Lusi Li, Daniel Takabi, Masha Sosonkina, Rui Ning  

**Link**: [PDF](https://arxiv.org/pdf/2506.00191)  

**Abstract**: Heterogeneous Graph Neural Networks (HGNNs) excel in modeling complex, multi-typed relationships across diverse domains, yet their vulnerability to backdoor attacks remains unexplored. To address this gap, we conduct the first investigation into the susceptibility of HGNNs to existing graph backdoor attacks, revealing three critical issues: (1) high attack budget required for effective backdoor injection, (2) inefficient and unreliable backdoor activation, and (3) inaccurate attack effectiveness evaluation. To tackle these issues, we propose the Heterogeneous Graph Backdoor Attack (HGBA), the first backdoor attack specifically designed for HGNNs, introducing a novel relation-based trigger mechanism that establishes specific connections between a strategically selected trigger node and poisoned nodes via the backdoor metapath. HGBA achieves efficient and stealthy backdoor injection with minimal structural modifications and supports easy backdoor activation through two flexible strategies: Self-Node Attack and Indiscriminate Attack. Additionally, we improve the ASR measurement protocol, enabling a more accurate assessment of attack effectiveness. Extensive experiments demonstrate that HGBA far surpasses multiple state-of-the-art graph backdoor attacks in black-box settings, efficiently attacking HGNNs with low attack budgets. Ablation studies show that the strength of HBGA benefits from our trigger node selection method and backdoor metapath selection strategy. In addition, HGBA shows superior robustness against node feature perturbations and multiple types of existing graph backdoor defense mechanisms. Finally, extension experiments demonstrate that the relation-based trigger mechanism can effectively extend to tasks in homogeneous graph scenarios, thereby posing severe threats to broader security-critical domains. 

**Abstract (ZH)**: 异质图神经网络的后门攻击：异质图后门攻击（HGBA）及其应用 

---
# Pushing the Limits of Beam Search Decoding for Transducer-based ASR models 

**Title (ZH)**: 基于发射机的ASR模型中极限拓展的束搜索解码方法 

**Authors**: Lilit Grigoryan, Vladimir Bataev, Andrei Andrusenko, Hainan Xu, Vitaly Lavrukhin, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2506.00185)  

**Abstract**: Transducer models have emerged as a promising choice for end-to-end ASR systems, offering a balanced trade-off between recognition accuracy, streaming capabilities, and inference speed in greedy decoding. However, beam search significantly slows down Transducers due to repeated evaluations of key network components, limiting practical applications. This paper introduces a universal method to accelerate beam search for Transducers, enabling the implementation of two optimized algorithms: ALSD++ and AES++. The proposed method utilizes batch operations, a tree-based hypothesis structure, novel blank scoring for enhanced shallow fusion, and CUDA graph execution for efficient GPU inference. This narrows the speed gap between beam and greedy modes to only 10-20% for the whole system, achieves 14-30% relative improvement in WER compared to greedy decoding, and improves shallow fusion for low-resource up to 11% compared to existing implementations. All the algorithms are open sourced. 

**Abstract (ZH)**: 递归神经网络模型已成为端到端ASR系统的有希望的选择，能够在贪婪解码中提供识别准确性、流式传输能力和推断速度之间的平衡trade-off。然而，束搜索由于反复评估关键网络组件而显著减慢递归神经网络模型的速度，限制了其实用应用。本文介绍了一种通用方法来加速递归神经网络模型的束搜索，实现了两个优化算法：ALSD++和AES++。所提出的方法利用批量操作、基于树的假设结构、增强浅融合的新空白评分以及CUDA图执行高效的GPU推断。这种方法将束搜索和贪婪模式之间的速度差距缩减至整个系统性能的10-20%，相比贪婪解码实现了14-30%的相对WER改进，并且相比现有实现对于低资源情况下的浅融合改进了11%。所有算法均已开源。 

---
# Accountability Attribution: Tracing Model Behavior to Training Processes 

**Title (ZH)**: 行为问责制归属：追踪模型行为至训练过程 

**Authors**: Shichang Zhang, Hongzhe Du, Karim Saraipour, Jiaqi W. Ma, Himabindu Lakkaraju  

**Link**: [PDF](https://arxiv.org/pdf/2506.00175)  

**Abstract**: Modern AI development pipelines often involve multiple stages-pretraining, fine-tuning rounds, and subsequent adaptation or alignment-with numerous model update steps within each stage. This raises a critical question of accountability: when a deployed model succeeds or fails, which stage is responsible, and to what extent? We pose the problem of accountability attribution, which aims to trace model behavior back to specific stages of the training process. To address this, we propose a general framework that answers counterfactual questions about stage effects: how would the model behavior have changed if the updates from a training stage had not been executed?. Within this framework, we introduce estimators based on first-order approximations that efficiently quantify the stage effects without retraining. Our estimators account for both the training data and key aspects of optimization dynamics, including learning rate schedules, momentum, and weight decay. Empirically, we demonstrate that our approach identifies training stages accountable for specific behaviors, offering a practical tool for model analysis and a step toward more accountable AI development. 

**Abstract (ZH)**: 现代AI开发管道中的责任归属问题：从训练过程的具体阶段追溯模型行为 

---
# Disentangled Safety Adapters Enable Efficient Guardrails and Flexible Inference-Time Alignment 

**Title (ZH)**: 解耦的安全适配器实现高效的安全保障和地区灵活的推理时对齐 

**Authors**: Kundan Krishna, Joseph Y Cheng, Charles Maalouf, Leon A Gatys  

**Link**: [PDF](https://arxiv.org/pdf/2506.00166)  

**Abstract**: Existing paradigms for ensuring AI safety, such as guardrail models and alignment training, often compromise either inference efficiency or development flexibility. We introduce Disentangled Safety Adapters (DSA), a novel framework addressing these challenges by decoupling safety-specific computations from a task-optimized base model. DSA utilizes lightweight adapters that leverage the base model's internal representations, enabling diverse and flexible safety functionalities with minimal impact on inference cost. Empirically, DSA-based safety guardrails substantially outperform comparably sized standalone models, notably improving hallucination detection (0.88 vs. 0.61 AUC on Summedits) and also excelling at classifying hate speech (0.98 vs. 0.92 on ToxiGen) and unsafe model inputs and responses (0.93 vs. 0.90 on AEGIS2.0 & BeaverTails). Furthermore, DSA-based safety alignment allows dynamic, inference-time adjustment of alignment strength and a fine-grained trade-off between instruction following performance and model safety. Importantly, combining the DSA safety guardrail with DSA safety alignment facilitates context-dependent alignment strength, boosting safety on StrongReject by 93% while maintaining 98% performance on MTBench -- a total reduction in alignment tax of 8 percentage points compared to standard safety alignment fine-tuning. Overall, DSA presents a promising path towards more modular, efficient, and adaptable AI safety and alignment. 

**Abstract (ZH)**: 解耦安全适配器：一种新型的AI安全与对齐框架 

---
# Gated Multimodal Graph Learning for Personalized Recommendation 

**Title (ZH)**: 基于门控多模态图学习的个性化推荐 

**Authors**: Sibei Liu, Yuanzhe Zhang, Xiang Li, Yunbo Liu, Chengwei Feng, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00107)  

**Abstract**: Multimodal recommendation has emerged as a promising solution to alleviate the cold-start and sparsity problems in collaborative filtering by incorporating rich content information, such as product images and textual descriptions. However, effectively integrating heterogeneous modalities into a unified recommendation framework remains a challenge. Existing approaches often rely on fixed fusion strategies or complex architectures , which may fail to adapt to modality quality variance or introduce unnecessary computational overhead.
In this work, we propose RLMultimodalRec, a lightweight and modular recommendation framework that combines graph-based user modeling with adaptive multimodal item encoding. The model employs a gated fusion module to dynamically balance the contribution of visual and textual modalities, enabling fine-grained and content-aware item representations. Meanwhile, a two-layer LightGCN encoder captures high-order collaborative signals by propagating embeddings over the user-item interaction graph without relying on nonlinear transformations.
We evaluate our model on a real-world dataset from the Amazon product domain. Experimental results demonstrate that RLMultimodalRec consistently outperforms several competitive baselines, including collaborative filtering, visual-aware, and multimodal GNN-based methods. The proposed approach achieves significant improvements in top-K recommendation metrics while maintaining scalability and interpretability, making it suitable for practical deployment. 

**Abstract (ZH)**: 基于图的用户建模与自适应多模态项编码的轻量级多模态推荐框架RLMultimodalRec 

---
# Children's Voice Privacy: First Steps And Emerging Challenges 

**Title (ZH)**: 儿童语音隐私：初步探讨与新兴挑战 

**Authors**: Ajinkya Kulkarni, Francisco Teixeira, Enno Hermann, Thomas Rolland, Isabel Trancoso, Mathew Magimai Doss  

**Link**: [PDF](https://arxiv.org/pdf/2506.00100)  

**Abstract**: Children are one of the most under-represented groups in speech technologies, as well as one of the most vulnerable in terms of privacy. Despite this, anonymization techniques targeting this population have received little attention. In this study, we seek to bridge this gap, and establish a baseline for the use of voice anonymization techniques designed for adult speech when applied to children's voices. Such an evaluation is essential, as children's speech presents a distinct set of challenges when compared to that of adults. This study comprises three children's datasets, six anonymization methods, and objective and subjective utility metrics for evaluation. Our results show that existing systems for adults are still able to protect children's voice privacy, but suffer from much higher utility degradation. In addition, our subjective study displays the challenges of automatic evaluation methods for speech quality in children's speech, highlighting the need for further research. 

**Abstract (ZH)**: 儿童是语言技术中最未被充分代表的群体之一，也是隐私方面最脆弱的群体之一。尽管如此，针对这一人群的匿名化技术尚未得到广泛关注。本研究旨在弥补这一差距，并建立一个基准，评估将旨在成年语音的匿名化技术应用于儿童语音时的有效性。由于儿童的语音与成人存在显著差异，这种评估至关重要。本研究包括三个儿童数据集、六种匿名化方法以及客观和主观效益度量标准进行评估。我们的结果显示，现有的成人系统仍然能够保护儿童的语音隐私，但会遭受更严重的效益降级。此外，我们的主观研究揭示了自动评估方法在儿童语音质量评估中面临的挑战，强调了进一步研究的必要性。 

---
# PathGene: Benchmarking Driver Gene Mutations and Exon Prediction Using Multicenter Lung Cancer Histopathology Image Dataset 

**Title (ZH)**: PathGene：基于多中心肺癌组织学图像数据集的驱动基因突变和外显子预测基准评估 

**Authors**: Liangrui Pan, Qingchun Liang, Shen Zhao, Songqing Fan, Shaoliang Peng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00096)  

**Abstract**: Accurately predicting gene mutations, mutation subtypes and their exons in lung cancer is critical for personalized treatment planning and prognostic assessment. Faced with regional disparities in medical resources and the high cost of genomic assays, using artificial intelligence to infer these mutations and exon variants from routine histopathology images could greatly facilitate precision therapy. Although some prior studies have shown that deep learning can accelerate the prediction of key gene mutations from lung cancer pathology slides, their performance remains suboptimal and has so far been limited mainly to early screening tasks. To address these limitations, we have assembled PathGene, which comprises histopathology images paired with next-generation sequencing reports from 1,576 patients at the Second Xiangya Hospital, Central South University, and 448 TCGA-LUAD patients. This multi-center dataset links whole-slide images to driver gene mutation status, mutation subtypes, exon, and tumor mutational burden (TMB) status, with the goal of leveraging pathology images to predict mutations, subtypes, exon locations, and TMB for early genetic screening and to advance precision oncology. Unlike existing datasets, we provide molecular-level information related to histopathology images in PathGene to facilitate the development of biomarker prediction models. We benchmarked 11 multiple-instance learning methods on PathGene for mutation, subtype, exon, and TMB prediction tasks. These experimental methods provide valuable alternatives for early genetic screening of lung cancer patients and assisting clinicians to quickly develop personalized precision targeted treatment plans for patients. Code and data are available at this https URL. 

**Abstract (ZH)**: 准确预测肺癌中的基因突变、突变亚型及其外显子对于个性化治疗规划和预后评估至关重要。面对医疗资源区域差异和基因组检测的高成本，使用人工智能从常规病理图像推断这些突变和外显子变异，可大大促进精准治疗。虽然一些前期研究显示深度学习可以在肺癌病理切片中加速关键基因突变的预测，但其性能仍然不佳，目前主要局限于早期筛查任务。为克服这些限制，我们构建了PathGene数据集，该数据集包含来自中南大学湘雅二医院和TCGA-LUAD的1,576名患者配对的病理图像和下一代测序报告，以及448名患者的病理图像。该多中心数据集将整个切片图像与驱动基因突变状态、突变亚型、外显子位置和肿瘤突变负担（TMB）状态相连，旨在利用病理图像进行早期遗传筛查并推进精准肿瘤学的发展。与现有数据集不同，PathGene提供了与病理图像相关的分子水平信息，以促进生物标志物预测模型的开发。我们在PathGene上 benchmark 了11种多实例学习方法，用于突变、亚型、外显子和TMB预测任务。这些实验方法提供了有价值的选择，用于肺癌患者的早期遗传筛查，帮助医生迅速制定针对患者的个性化精准靶向治疗方案。代码和数据可在以下链接获取。 

---
# Feeling Guilty Being a c(ai)borg: Navigating the Tensions Between Guilt and Empowerment in AI Use 

**Title (ZH)**: 成为半机械人时的内疚感：在AI使用中平衡内疚与赋能之间的张力 

**Authors**: Konstantin Aal, Tanja Aal, Vasil Navumau, David Unbehaun, Claudia Müller, Volker Wulf, Sarah Rüller  

**Link**: [PDF](https://arxiv.org/pdf/2506.00094)  

**Abstract**: This paper explores the emotional, ethical and practical dimensions of integrating Artificial Intelligence (AI) into personal and professional workflows, focusing on the concept of feeling guilty as a 'c(ai)borg' - a human augmented by AI. Inspired by Donna Haraway's Cyborg Manifesto, the study explores how AI challenges traditional notions of creativity, originality and intellectual labour. Using an autoethnographic approach, the authors reflect on their year-long experiences with AI tools, revealing a transition from initial guilt and reluctance to empowerment through skill-building and transparency. Key findings highlight the importance of basic academic skills, advanced AI literacy and honest engagement with AI results. The c(ai)borg vision advocates for a future where AI is openly embraced as a collaborative partner, fostering innovation and equity while addressing issues of access and agency. By reframing guilt as growth, the paper calls for a thoughtful and inclusive approach to AI integration. 

**Abstract (ZH)**: 本文探讨将人工智能（AI）整合到个人和职业工作流程中的情感、伦理和实践维度，重点关注作为“c(ai)borg”（人类增强的AI）时产生的罪恶感概念。受唐娜·哈拉维的《赛博格宣言》启发，研究探讨了AI对传统创造力、原创性和智力劳动观念的挑战。通过自传民族志的方法，作者反思了他们在一年中使用AI工具的经验，揭示了从最初的罪恶感和抵触到通过技能提升和透明度实现赋能的转变。关键发现强调了基础学术技能、高级AI素养以及坦诚面对AI结果的重要性。“c(ai)borg”愿景倡导一个AI被公开接受作为协作伙伴的未来，促进创新和公平，同时解决访问和代理问题。通过将罪恶感重新定位为成长的机会，本文呼吁采取深入和包容的方式整合AI。 

---
# SwitchLingua: The First Large-Scale Multilingual and Multi-Ethnic Code-Switching Dataset 

**Title (ZH)**: SwitchLingua: 首个多语言和多民族代码转换大型数据集 

**Authors**: Peng Xie, Xingyuan Liu, Tsz Wai Chan, Yequan Bie, Yangqiu Song, Yang Wang, Hao Chen, Kani Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00087)  

**Abstract**: Code-switching (CS) is the alternating use of two or more languages within a conversation or utterance, often influenced by social context and speaker identity. This linguistic phenomenon poses challenges for Automatic Speech Recognition (ASR) systems, which are typically designed for a single language and struggle to handle multilingual inputs. The growing global demand for multilingual applications, including Code-Switching ASR (CSASR), Text-to-Speech (CSTTS), and Cross-Lingual Information Retrieval (CLIR), highlights the inadequacy of existing monolingual datasets.
Although some code-switching datasets exist, most are limited to bilingual mixing within homogeneous ethnic groups, leaving a critical need for a large-scale, diverse benchmark akin to ImageNet in computer vision.
To bridge this gap, we introduce \textbf{LinguaMaster}, a multi-agent collaboration framework specifically designed for efficient and scalable multilingual data synthesis. Leveraging this framework, we curate \textbf{SwitchLingua}, the first large-scale multilingual and multi-ethnic code-switching dataset, including: (1) 420K CS textual samples across 12 languages, and (2) over 80 hours of audio recordings from 174 speakers representing 18 countries/regions and 63 racial/ethnic backgrounds, based on the textual data. This dataset captures rich linguistic and cultural diversity, offering a foundational resource for advancing multilingual and multicultural research. Furthermore, to address the issue that existing ASR evaluation metrics lack sensitivity to code-switching scenarios, we propose the \textbf{Semantic-Aware Error Rate (SAER)}, a novel evaluation metric that incorporates semantic information, providing a more accurate and context-aware assessment of system performance. 

**Abstract (ZH)**: 代码转换（CS）是指在对话或语句中交替使用两种或多种语言的现象，往往受到社会背景和说话人身份的影响。这一语言现象给自动语音识别（ASR）系统带来了挑战，这些系统通常仅设计用于单一语言并难以处理多语言输入。随着全球对多语言应用的需求增长，包括代码转换自动语音识别（CSASR）、文本到语音（CSTTS）和跨语言信息检索（CLIR）应用，突显了现有单一语言数据集的不足。

尽管存在一些代码转换数据集，但大多数数据集仅限于同质族裔群体内的双语混合，因此需要类似ImageNet的大规模、多元化基准数据集。

为弥补这一差距，我们引入了**LinguaMaster**，一种专门用于高效可扩展多语言数据合成的多代理协作框架。利用这一框架，我们整理了**SwitchLingua**，这是第一个大规模的多语言和多族裔代码转换数据集，包含：(1) 12种语言的420,000个代码转换文本样本；(2) 来自174位代表18个国家/地区和63种种族/族裔背景的讲话者，超过80小时的音频记录，基于文本数据。该数据集涵盖了丰富的语言和文化多样性，为多语言和多文化研究提供了基础资源。此外，为了解决现有ASR评估指标对代码转换场景敏感性不足的问题，我们提出了**语义感知错误率（SAER）**，这是一种新颖的评估指标，结合了语义信息，提供了更准确和情境化的系统性能评估。 

---
# Bottom-Up Perspectives on AI Governance: Insights from User Reviews of AI Products 

**Title (ZH)**: 自下而上的AI治理视角：来自AI产品用户评审的见解 

**Authors**: Stefan Pasch  

**Link**: [PDF](https://arxiv.org/pdf/2506.00080)  

**Abstract**: With the growing importance of AI governance, numerous high-level frameworks and principles have been articulated by policymakers, institutions, and expert communities to guide the development and application of AI. While such frameworks offer valuable normative orientation, they may not fully capture the practical concerns of those who interact with AI systems in organizational and operational contexts. To address this gap, this study adopts a bottom-up approach to explore how governance-relevant themes are expressed in user discourse. Drawing on over 100,000 user reviews of AI products from this http URL, we apply BERTopic to extract latent themes and identify those most semantically related to AI governance. The analysis reveals a diverse set of governance-relevant topics spanning both technical and non-technical domains. These include concerns across organizational processes-such as planning, coordination, and communication-as well as stages of the AI value chain, including deployment infrastructure, data handling, and analytics. The findings show considerable overlap with institutional AI governance and ethics frameworks on issues like privacy and transparency, but also surface overlooked areas such as project management, strategy development, and customer interaction. This highlights the need for more empirically grounded, user-centered approaches to AI governance-approaches that complement normative models by capturing how governance unfolds in applied settings. By foregrounding how governance is enacted in practice, this study contributes to more inclusive and operationally grounded approaches to AI governance and digital policy. 

**Abstract (ZH)**: 随着AI治理的重要性日益增长，政策制定者、机构和专家社区已经制定了众多高层次的框架和原则，以指导AI的发展与应用。尽管这些框架提供了有价值的规范性导向，但在组织和运行 contexts 中与AI系统互动的人可能不会全面关注其中的实际问题。为解决这一缺口，本研究采用自下而上的方法探索治理相关主题在用户话语中的表达。通过分析来自某网站的超过100,000条AI产品的用户评论，我们应用BERTopic提取潜在主题，并识别与AI治理最相关的主题。分析结果显示，治理相关的话题涵盖了技术与非技术领域。这些话题包括组织流程中的担忧，如规划、协调和沟通，以及AI价值链的各个阶段，包括部署基础设施、数据处理和数据分析。研究发现，在隐私和透明度等议题上与机构AI治理和伦理框架存在显著重叠，但也揭示了诸如项目管理、战略发展和客户互动等未被充分注意的领域。这强调了需要更多基于实证、用户中心的AI治理方法——这些方法可以补充规范性模型，捕捉治理在应用环境中的实际展开过程。通过凸显治理在实践中的具体表现，本研究促进了更具包容性和操作性的AI治理和数字政策方法。 

---
# Optimizing Storytelling, Improving Audience Retention, and Reducing Waste in the Entertainment Industry 

**Title (ZH)**: 优化叙事技巧，提升观众留存率，减少娱乐行业的浪费 

**Authors**: Andrew Cornfeld, Ashley Miller, Mercedes Mora-Figueroa, Kurt Samuels, Anthony Palomba  

**Link**: [PDF](https://arxiv.org/pdf/2506.00076)  

**Abstract**: Television networks face high financial risk when making programming decisions, often relying on limited historical data to forecast episodic viewership. This study introduces a machine learning framework that integrates natural language processing (NLP) features from over 25000 television episodes with traditional viewership data to enhance predictive accuracy. By extracting emotional tone, cognitive complexity, and narrative structure from episode dialogue, we evaluate forecasting performance using SARIMAX, rolling XGBoost, and feature selection models. While prior viewership remains a strong baseline predictor, NLP features contribute meaningful improvements for some series. We also introduce a similarity scoring method based on Euclidean distance between aggregate dialogue vectors to compare shows by content. Tested across diverse genres, including Better Call Saul and Abbott Elementary, our framework reveals genre-specific performance and offers interpretable metrics for writers, executives, and marketers seeking data-driven insight into audience behavior. 

**Abstract (ZH)**: 电视网络在节目制作决策中面临高い财务风险，通常依赖有限的历史数据来预测集锦收视率。本研究介绍了一种将自然语言处理（NLP）功能与传统收视数据结合的机器学习框架，以提高预测准确性。通过从超过25000个电视节目中提取情绪基调、认知复杂性和叙事结构，我们使用SARIMAX、滚动XGBoost和特征选择模型评估预测性能。尽管过去的收视情况仍然是一个强大的基准预测器，但对于某些系列而言，NLP功能的贡献意味着有意义的提升。我们还介绍了一种基于聚类对话向量之间欧几里得距离的相似性评分方法，用于按内容比较节目。在包括《Better Call Saul》和《Abbott Elementary》在内的多种类型中测试，本框架揭示了特定类型的性能，并为编剧、高层管理人员和市场营销人员提供了可解释的指标，以获取有关观众行为的数据驱动见解。 

---
# You Prefer This One, I Prefer Yours: Using Reference Words is Harder Than Vocabulary Words for Humans and Multimodal Language Models 

**Title (ZH)**: 你偏好这个，我偏好那个：引用词比词汇词对人类和多模态语言模型来说更难处理 

**Authors**: Dota Tianai Dong, Yifan Luo, Po-Ya Angela Wang, Asli Ozyurek, Paula Rubio-Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2506.00065)  

**Abstract**: Multimodal language models (MLMs) increasingly communicate in human-like ways, yet their ability to use reference words remains largely overlooked despite their ubiquity in everyday communication. Our study addresses this gap by comparing human and MLM use of three word classes with increasing cognitive demands: vocabulary words, possessive pronouns (`mine' vs `yours'), and demonstrative pronouns (`this one' vs `that one'). Evaluating seven state-of-the-art MLMs against human participants, we observe a clear difficulty hierarchy: while MLMs approach human-level performance on the vocabulary task, they show substantial deficits with possessives and demonstratives. Our analysis reveals these difficulties stem from limitations in perspective-taking and spatial reasoning. Although prompt engineering improved model performance on possessive use, demonstrative use remained well below human-level competence. These findings provide theoretical and empirical evidence that producing grammatical forms requiring pragmatics and social cognition remains a clear challenge in current NLP systems. 

**Abstract (ZH)**: 多模态语言模型在使用引用词方面的能力仍被忽视：从词汇词、代词（“我的” vs “你的”）到指示代词（“这个” vs “那个”）的认知需求递增比较研究 

---
# Improving statistical learning methods via features selection without replacement sampling and random projection 

**Title (ZH)**: 通过无替换抽样和随机投影进行特征选择以提高统计学习方法 

**Authors**: Sulaiman khan, Muhammad Ahmad, Fida Ullah, Carlos Aguilar Ibañez, José Eduardo Valdez Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2506.00053)  

**Abstract**: Cancer is fundamentally a genetic disease characterized by genetic and epigenetic alterations that disrupt normal gene expression, leading to uncontrolled cell growth and metastasis. High-dimensional microarray datasets pose challenges for classification models due to the "small n, large p" problem, resulting in overfitting. This study makes three different key contributions: 1) we propose a machine learning-based approach integrating the Feature Selection Without Re-placement (FSWOR) technique and a projection method to improve classification accuracy. 2) We apply the Kendall statistical test to identify the most significant genes from the brain cancer mi-croarray dataset (GSE50161), reducing the feature space from 54,675 to 20,890 genes.3) we apply machine learning models using k-fold cross validation techniques in which our model incorpo-rates ensemble classifiers with LDA projection and Naïve Bayes, achieving a test score of 96%, outperforming existing methods by 9.09%. The results demonstrate the effectiveness of our ap-proach in high-dimensional gene expression analysis, improving classification accuracy while mitigating overfitting. This study contributes to cancer biomarker discovery, offering a robust computational method for analyzing microarray data. 

**Abstract (ZH)**: 癌症本质上是一种由遗传和表观遗传改变引起的基因疾病，这些改变会扰乱正常的基因表达，导致不受控制的细胞生长和转移。高维度的微阵列数据集由于“小n，大p”问题给分类模型带来了挑战，容易导致过拟合。本研究作出三项关键贡献：1）我们提出了一种基于机器学习的方法，结合Feature Selection Without Replacement（FSWOR）技术和投影方法以提高分类准确性。2）我们应用肯德尔统计检验从脑癌微阵列数据集（GSE50161）中筛选出最显著的基因，将特征空间从54,675个基因减少到20,890个基因。3）我们应用k折交叉验证技术并结合LDA投影和朴素贝叶斯的集成分类器构建模型，测试得分为96%，优于现有方法9.09%。研究结果证明了在高维度基因表达分析中我们方法的有效性，提高了分类准确性同时减轻了过拟合问题。本研究为癌症生物标志物的发现提供了稳健的计算方法，用于分析微阵列数据。 

---
# Risks of AI-driven product development and strategies for their mitigation 

**Title (ZH)**: AI驱动的产品开发风险及其缓解策略 

**Authors**: Jan Göpfert, Jann M. Weinand, Patrick Kuckertz, Noah Pflugradt, Jochen Linßen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00047)  

**Abstract**: Humanity is progressing towards automated product development, a trend that promises faster creation of better products and thus the acceleration of technological progress. However, increasing reliance on non-human agents for this process introduces many risks. This perspective aims to initiate a discussion on these risks and appropriate mitigation strategies. To this end, we outline a set of principles for safer AI-driven product development which emphasize human oversight, accountability, and explainable design, among others. The risk assessment covers both technical risks which affect product quality and safety, and sociotechnical risks which affect society. While AI-driven product development is still in its early stages, this discussion will help balance its opportunities and risks without delaying essential progress in understanding, norm-setting, and regulation. 

**Abstract (ZH)**: 人类正朝着自动化产品开发方向前进，这一趋势有望加快技术进步，更快地创造更优质的产品。然而，对这一过程越来越多地依赖非人类代理也带来了许多风险。本文旨在探讨这些风险以及适当的缓解策略。为此，我们概述了一套更安全的AI驱动产品开发原则，这些原则强调了人的监督、责任以及可解释的设计等。风险评估涵盖了影响产品质量和安全的技术风险以及影响社会的 sociotechnical 风险。尽管AI驱动产品开发仍处于早期阶段，但此次讨论将有助于平衡其机遇和风险，而不延误对了解、制定规范和监管的基本理解。 

---
# The Folly of AI for Age Verification 

**Title (ZH)**: AI在年龄验证中的谬误 

**Authors**: Reid McIlroy-Young  

**Link**: [PDF](https://arxiv.org/pdf/2506.00038)  

**Abstract**: In the near future a governmental body will be asked to allow companies to use AI for age verification. If they allow it the resulting system will both be easily circumvented and disproportionately misclassify minorities and low socioeconomic status users. This is predictable by showing that other very similar systems (facial recognition and remote proctoring software) have similar issues despite years of efforts to mitigate their biases. These biases are due to technical limitations both of the AI models themselves and the physical hardware they are running on that will be difficult to overcome below the cost of government ID-based age verification. Thus in, the near future, deploying an AI system for age verification is folly. 

**Abstract (ZH)**: 在未来，政府机构将被要求允许公司使用AI进行年龄验证。如果他们批准这一做法， resulting系统将容易被规避，并且会不成比例地错误分类少数群体和低社会经济地位的用户。这一点可以通过展示其他非常相似的系统（面部识别和远程监考软件）尽管经历了多年的努力以减轻其偏见，但仍存在类似问题来预测。这些偏见源于AI模型本身和它们运行的物理硬件的技术限制，克服这些限制的成本将高于基于政府身份验证的年龄验证成本。因此，在未来部署AI系统进行年龄验证是不智之举。 

---
# MolTextNet: A Two-Million Molecule-Text Dataset for Multimodal Molecular Learning 

**Title (ZH)**: MolTextNet：用于多模态分子学习的大型分子-文本数据集 

**Authors**: Yihan Zhu, Gang Liu, Eric Inae, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00009)  

**Abstract**: Small molecules are essential to drug discovery, and graph-language models hold promise for learning molecular properties and functions from text. However, existing molecule-text datasets are limited in scale and informativeness, restricting the training of generalizable multimodal models. We present MolTextNet, a dataset of 2.5 million high-quality molecule-text pairs designed to overcome these limitations. To construct it, we propose a synthetic text generation pipeline that integrates structural features, computed properties, bioactivity data, and synthetic complexity. Using GPT-4o-mini, we create structured descriptions for 2.5 million molecules from ChEMBL35, with text over 10 times longer than prior datasets. MolTextNet supports diverse downstream tasks, including property prediction and structure retrieval. Pretraining CLIP-style models with Graph Neural Networks and ModernBERT on MolTextNet yields improved performance, highlighting its potential for advancing foundational multimodal modeling in molecular science. Our dataset is available at this https URL. 

**Abstract (ZH)**: MolTextNet：一个用于分子科学基础多模态建模的高质量分子-文本数据集 

---
# Rapid yet accurate Tile-circuit and device modeling for Analog In-Memory Computing 

**Title (ZH)**: 快速而准确的Tile电路和器件建模方法及其在类比内存计算中的应用 

**Authors**: J. Luquin, C. Mackin, S. Ambrogio, A. Chen, F. Baldi, G. Miralles, M.J. Rasch, J. Büchel, M. Lalwani, W. Ponghiran, P. Solomon, H. Tsai, G.W. Burr, P. Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2506.00004)  

**Abstract**: Analog In-Memory Compute (AIMC) can improve the energy efficiency of Deep Learning by orders of magnitude. Yet analog-domain device and circuit non-idealities -- within the analog ``Tiles'' performing Matrix-Vector Multiply (MVM) operations -- can degrade neural-network task accuracy. We quantify the impact of low-level distortions and noise, and develop a mathematical model for Multiply-ACcumulate (MAC) operations mapped to analog tiles. Instantaneous-current IR-drop (the most significant circuit non-ideality), and ADC quantization effects are fully captured by this model, which can predict MVM tile-outputs both rapidly and accurately, as compared to much slower rigorous circuit simulations. A statistical model of PCM read noise at nanosecond timescales is derived from -- and matched against -- experimental measurements. We integrate these (statistical) device and (deterministic) circuit effects into a PyTorch-based framework to assess the accuracy impact on the BERT and ALBERT Transformer networks. We show that hardware-aware fine-tuning using simple Gaussian noise provides resilience against ADC quantization and PCM read noise effects, but is less effective against IR-drop. This is because IR-drop -- although deterministic -- is non-linear, is changing significantly during the time-integration window, and is ultimately dependent on all the excitations being introduced in parallel into the analog tile. The apparent inability of simple Gaussian noise applied during training to properly prepare a DNN network for IR-drop during inference implies that more complex training approaches -- incorporating advances such as the Tile-circuit model introduced here -- will be critical for resilient deployment of large neural networks onto AIMC hardware. 

**Abstract (ZH)**: 模拟内存计算（AIMC）可以大幅提升深度学习的能效。然而，模拟域器件和电路非理想性——在执行矩阵-向量乘法（MVM）操作的模拟“瓷砖”中——可能会降低神经网络任务的准确性。我们量化了低级失真和噪声的影响，并开发了一个适用于映射到模拟瓷砖的乘加（MAC）操作的数学模型。瞬态电流IR降（电路非理想性中最显著的因素）和ADC量化效应完全由该模型捕获，该模型比更慢的严格电路仿真能更快更准确地预测MVM瓷砖的输出。小型脉码模（PCM）读噪声的统计模型从实验测量中导出并匹配。我们将这些（统计）设备效果和（确定性）电路效果集成到基于PyTorch的框架中，以评估其对BERT和ALBERT变换器网络的影响。我们展示了使用简单高斯噪声进行硬件感知微调可以增强对ADC量化和PCM读噪声效应的鲁棒性，但对IR降的效果较差。这是因为虽然IR降是确定性的，但它是非线性的，在时间积分窗口中显著变化，并最终依赖于同时引入到模拟瓷砖的所有激励。简单高斯噪声在训练期间应用于准备DNN网络以应对推断中的IR降的能力有限，这表明需要更复杂的训练方法——如这里引入的瓷砖电路模型——对于在AIMC硬件上部署大规模神经网络至关重要。 

---
