# Infinite-Horizon Value Function Approximation for Model Predictive Control 

**Title (ZH)**: 无限 horizons 价值函数逼近用于模型预测控制 

**Authors**: Armand Jordana, Sébastien Kleff, Arthur Haffemayer, Joaquim Ortiz-Haro, Justin Carpentier, Nicolas Mansard, Ludovic Righetti  

**Link**: [PDF](https://arxiv.org/pdf/2502.06760)  

**Abstract**: Model Predictive Control has emerged as a popular tool for robots to generate complex motions. However, the real-time requirement has limited the use of hard constraints and large preview horizons, which are necessary to ensure safety and stability. In practice, practitioners have to carefully design cost functions that can imitate an infinite horizon formulation, which is tedious and often results in local minima. In this work, we study how to approximate the infinite horizon value function of constrained optimal control problems with neural networks using value iteration and trajectory optimization. Furthermore, we demonstrate how using this value function approximation as a terminal cost provides global stability to the model predictive controller. The approach is validated on two toy problems and a real-world scenario with online obstacle avoidance on an industrial manipulator where the value function is conditioned to the goal and obstacle. 

**Abstract (ZH)**: 我们研究了如何使用神经网络通过值迭代和轨迹优化来逼近约束最优控制问题的无界 horizons 价值函数，并进一步展示了将此价值函数逼近作为终端代价如何为模型预测控制器提供全局稳定性。该方法在两个玩具问题和一个具有在线障碍避免的实际工业 manipulator 场景中得到了验证，其中价值函数根据目标和障碍进行条件化。 

---
# SIGMA: Sheaf-Informed Geometric Multi-Agent Pathfinding 

**Title (ZH)**: SIGMA: 基于层化几何的多方路径寻找 

**Authors**: Shuhao Liao, Weihang Xia, Yuhong Cao, Weiheng Dai, Chengyang He, Wenjun Wu, Guillaume Sartoretti  

**Link**: [PDF](https://arxiv.org/pdf/2502.06440)  

**Abstract**: The Multi-Agent Path Finding (MAPF) problem aims to determine the shortest and collision-free paths for multiple agents in a known, potentially obstacle-ridden environment. It is the core challenge for robotic deployments in large-scale logistics and transportation. Decentralized learning-based approaches have shown great potential for addressing the MAPF problems, offering more reactive and scalable solutions. However, existing learning-based MAPF methods usually rely on agents making decisions based on a limited field of view (FOV), resulting in short-sighted policies and inefficient cooperation in complex scenarios. There, a critical challenge is to achieve consensus on potential movements between agents based on limited observations and communications. To tackle this challenge, we introduce a new framework that applies sheaf theory to decentralized deep reinforcement learning, enabling agents to learn geometric cross-dependencies between each other through local consensus and utilize them for tightly cooperative decision-making. In particular, sheaf theory provides a mathematical proof of conditions for achieving global consensus through local observation. Inspired by this, we incorporate a neural network to approximately model the consensus in latent space based on sheaf theory and train it through self-supervised learning. During the task, in addition to normal features for MAPF as in previous works, each agent distributedly reasons about a learned consensus feature, leading to efficient cooperation on pathfinding and collision avoidance. As a result, our proposed method demonstrates significant improvements over state-of-the-art learning-based MAPF planners, especially in relatively large and complex scenarios, demonstrating its superiority over baselines in various simulations and real-world robot experiments. 

**Abstract (ZH)**: 基于束理论的分布式深度强化学习在多Agent路径规划中的应用 

---
# Improved Extrinsic Calibration of Acoustic Cameras via Batch Optimization 

**Title (ZH)**: 基于批量优化的声摄像机外部标定改进方法 

**Authors**: Zhi Li, Jiang Wang, Xiaoyang Li, He Kong  

**Link**: [PDF](https://arxiv.org/pdf/2502.06196)  

**Abstract**: Acoustic cameras have found many applications in practice. Accurate and reliable extrinsic calibration of the microphone array and visual sensors within acoustic cameras is crucial for fusing visual and auditory measurements. Existing calibration methods either require prior knowledge of the microphone array geometry or rely on grid search which suffers from slow iteration speed or poor convergence. To overcome these limitations, in this paper, we propose an automatic calibration technique using a calibration board with both visual and acoustic markers to identify each microphone position in the camera frame. We formulate the extrinsic calibration problem (between microphones and the visual sensor) as a nonlinear least squares problem and employ a batch optimization strategy to solve the associated problem. Extensive numerical simulations and realworld experiments show that the proposed method improves both the accuracy and robustness of extrinsic parameter calibration for acoustic cameras, in comparison to existing methods. To benefit the community, we open-source all the codes and data at this https URL. 

**Abstract (ZH)**: 声学相机在实践中找到了许多应用。声学相机中麦克风阵列和视觉传感器的精确可靠的外部校准对于融合视觉和听觉测量至关重要。现有的校准方法要么需要麦克风阵列几何结构的先验知识，要么依赖于网格搜索，这会导致迭代速度慢或收敛效果差。为克服这些限制，本文提出了一种使用带有视觉和声学标记的校准板的自动校准技术，以在相机框架中识别每个麦克风的位置。我们将外部校准问题（麦克风与视觉传感器之间）表述为非线性最小二乘问题，并采用批量优化策略来解决相关问题。广泛的数值模拟和实地实验表明，所提出的方法在声学相机的外部参数校准的精确性和鲁棒性方面均优于现有方法。为了惠及社区，我们在此处开放了所有代码和数据。 

---
# Mixed Reality Outperforms Virtual Reality for Remote Error Resolution in Pick-and-Place Tasks 

**Title (ZH)**: 混合现实优于虚拟现实的远程拾取放置任务错误解决性能 

**Authors**: Advay Kumar, Stephanie Simangunsong, Pamela Carreno-Medrano, Akansel Cosgun  

**Link**: [PDF](https://arxiv.org/pdf/2502.06141)  

**Abstract**: This study evaluates the performance and usability of Mixed Reality (MR), Virtual Reality (VR), and camera stream interfaces for remote error resolution tasks, such as correcting warehouse packaging errors. Specifically, we consider a scenario where a robotic arm halts after detecting an error, requiring a remote operator to intervene and resolve it via pick-and-place actions. Twenty-one participants performed simulated pick-and-place tasks using each interface. A linear mixed model (LMM) analysis of task resolution time, usability scores (SUS), and mental workload scores (NASA-TLX) showed that the MR interface outperformed both VR and camera interfaces. MR enabled significantly faster task completion, was rated higher in usability, and was perceived to be less cognitively demanding. Notably, the MR interface, which projected a virtual robot onto a physical table, provided superior spatial understanding and physical reference cues. Post-study surveys further confirmed participants' preference for MR over other interfaces. 

**Abstract (ZH)**: 本研究评估了混合现实（MR）、虚拟现实（VR）和摄像头流接口在远程错误解决任务中的性能和易用性，例如纠正仓库包装错误。具体而言，我们考虑了机器人手臂在检测到错误后停止工作，需要远程操作员通过抓取和放置操作介入并解决问题的场景。二十一名参与者使用每种接口完成了模拟的抓取和放置任务。线性混合模型（LMM）分析任务解决时间、易用性评分（SUS）和心理负荷评分（NASA-TLX）显示，MR接口在性能和可用性方面均优于VR和摄像头接口。MR使任务完成显著加快，易用性评分较高，并被认为认知负担较小。值得注意的是，能够将虚拟机器人投射到物理桌面上的MR接口提供了更好的空间理解和物理参考提示。研究后的调查进一步证实了参与者更偏好MR接口。 

---
# Real-Time LiDAR Point Cloud Compression and Transmission for Resource-constrained Robots 

**Title (ZH)**: 基于资源受限机器人实时LiDAR点云压缩与传输 

**Authors**: Yuhao Cao, Yu Wang, Haoyao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.06123)  

**Abstract**: LiDARs are widely used in autonomous robots due to their ability to provide accurate environment structural information. However, the large size of point clouds poses challenges in terms of data storage and transmission. In this paper, we propose a novel point cloud compression and transmission framework for resource-constrained robotic applications, called RCPCC. We iteratively fit the surface of point clouds with a similar range value and eliminate redundancy through their spatial relationships. Then, we use Shape-adaptive DCT (SA-DCT) to transform the unfit points and reduce the data volume by quantizing the transformed coefficients. We design an adaptive bitrate control strategy based on QoE as the optimization goal to control the quality of the transmitted point cloud. Experiments show that our framework achieves compression rates of 40$\times$ to 80$\times$ while maintaining high accuracy for downstream applications. our method significantly outperforms other baselines in terms of accuracy when the compression rate exceeds 70$\times$. Furthermore, in situations of reduced communication bandwidth, our adaptive bitrate control strategy demonstrates significant QoE improvements. The code will be available at this https URL. 

**Abstract (ZH)**: 基于资源约束的机器人应用的新型点云压缩与传输框架RCPCC 

---
# Surprise Potential as a Measure of Interactivity in Driving Scenarios 

**Title (ZH)**: Surprise潜力作为驾驶场景中互动性的度量 

**Authors**: Wenhao Ding, Sushant Veer, Karen Leung, Yulong Cao, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2502.05677)  

**Abstract**: Validating the safety and performance of an autonomous vehicle (AV) requires benchmarking on real-world driving logs. However, typical driving logs contain mostly uneventful scenarios with minimal interactions between road users. Identifying interactive scenarios in real-world driving logs enables the curation of datasets that amplify critical signals and provide a more accurate assessment of an AV's performance. In this paper, we present a novel metric that identifies interactive scenarios by measuring an AV's surprise potential on others. First, we identify three dimensions of the design space to describe a family of surprise potential measures. Second, we exhaustively evaluate and compare different instantiations of the surprise potential measure within this design space on the nuScenes dataset. To determine how well a surprise potential measure correctly identifies an interactive scenario, we use a reward model learned from human preferences to assess alignment with human intuition. Our proposed surprise potential, arising from this exhaustive comparative study, achieves a correlation of more than 0.82 with the human-aligned reward function, outperforming existing approaches. Lastly, we validate motion planners on curated interactive scenarios to demonstrate downstream applications. 

**Abstract (ZH)**: 验证自动驾驶汽车（AV）的安全性和性能需要在实际驾驶日志中进行基准测试。然而，典型的驾驶日志主要包含无事件场景，路用户之间的互动很少。在实际驾驶日志中识别互动场景能够创建更能放大关键信号的数据集，从而提供对AV性能更准确的评估。本文提出了一种新的度量标准，通过衡量AV对其他方的惊喜潜力来识别互动场景。首先，我们定义了设计空间的三个维度以描述惊喜潜力度量的一组度量。其次，我们在nuScenes数据集上全面评估和比较设计空间内不同实例化的惊喜潜力度量。为了确定惊喜潜力度量在多大程度上正确识别互动场景，我们使用从人类偏好中学习到的奖励模型来评估其与人类直觉的契合度。我们提出的一种全面比较研究中产生的惊喜潜力，与人类对齐的奖励函数的相关性超过0.82，优于现有方法。最后，我们在精心策划的互动场景中验证运动规划器，以展示下游应用。 

---
# Model Validity in Observers: When to Increase the Complexity of Your Model? 

**Title (ZH)**: 观察者中模型有效性的问题：何时增加模型的复杂度？ 

**Authors**: Agapius Bou Ghosn, Philip Polack, Arnaud de La Fortelle  

**Link**: [PDF](https://arxiv.org/pdf/2502.05479)  

**Abstract**: Model validity is key to the accurate and safe behavior of autonomous vehicles. Using invalid vehicle models in the different plan and control vehicle frameworks puts the stability of the vehicle, and thus its safety at stake. In this work, we analyze the validity of several popular vehicle models used in the literature with respect to a real vehicle and we prove that serious accuracy issues are encountered beyond a specific lateral acceleration point. We set a clear lateral acceleration domain in which the used models are an accurate representation of the behavior of the vehicle. We then target the necessity of using learned methods to model the vehicle's behavior. The effects of model validity on state observers are investigated. The performance of model-based observers is compared to learning-based ones. Overall, the presented work emphasizes the validity of vehicle models and presents clear operational domains in which models could be used safely. 

**Abstract (ZH)**: 模型的有效性是自动驾驶车辆准确和安全行为的关键。使用无效的车辆模型会影响不同的规划和控制框架的稳定性，从而影响其安全性。在本工作中，我们分析了几种在文献中使用的流行车辆模型与实际车辆的一致性，并证明在特定侧向加速度点之后会出现严重的准确度问题。我们设定了一个明确的侧向加速度域，在此域内所使用的模型能准确地反映车辆的行为。然后我们强调了使用学习方法来建模车辆行为的必要性。研究了模型有效性对状态观测器的影响。比较了基于模型的观测器与基于学习的观测器的性能。总体而言，本工作强调了车辆模型的有效性，并提出了模型可以安全使用的明确操作域。 

---
# Rough Stochastic Pontryagin Maximum Principle and an Indirect Shooting Method 

**Title (ZH)**: 不连续随机庞特里亚金最大原理及间接射击方法 

**Authors**: Thomas Lew  

**Link**: [PDF](https://arxiv.org/pdf/2502.06726)  

**Abstract**: We derive first-order Pontryagin optimality conditions for stochastic optimal control with deterministic controls for systems modeled by rough differential equations (RDE) driven by Gaussian rough paths. This Pontryagin Maximum Principle (PMP) applies to systems following stochastic differential equations (SDE) driven by Brownian motion, yet it does not rely on forward-backward SDEs and involves the same Hamiltonian as the deterministic PMP. The proof consists of first deriving various integrable error bounds for solutions to nonlinear and linear RDEs by leveraging recent results on Gaussian rough paths. The PMP then follows using standard techniques based on needle-like variations. As an application, we propose the first indirect shooting method for nonlinear stochastic optimal control and show that it converges 10x faster than a direct method on a stabilization task. 

**Abstract (ZH)**: 我们推导了基于高斯粗糙道路驱动的粗糙微分方程（RDE）系统的随机最优控制的一阶庞特里亚金最优性条件。该庞特里亚金最大原理（PMP）适用于由布朗运动驱动的随机微分方程（SDE）系统，且不依赖于前向后向SDE，并且涉及与确定性PMP相同的哈密尔顿量。证明过程首先通过利用近期关于高斯粗糙道路的结果，推导出非线性和线性RDE解的各种可积误差界。随后使用基于细针变化的标准技术得出PMP。作为应用，我们提出了首个非线性随机最优控制的间接射击方法，并展示了其在稳定化任务上比直接方法快10倍的收敛速度。 

---
# An Automated Machine Learning Framework for Surgical Suturing Action Detection under Class Imbalance 

**Title (ZH)**: 面向类别不平衡的外科缝合动作检测的自动化机器学习框架 

**Authors**: Baobing Zhang, Paul Sullivan, Benjie Tang, Ghulam Nabi, Mustafa Suphi Erden  

**Link**: [PDF](https://arxiv.org/pdf/2502.06407)  

**Abstract**: In laparoscopy surgical training and evaluation, real-time detection of surgical actions with interpretable outputs is crucial for automated and real-time instructional feedback and skill development. Such capability would enable development of machine guided training systems. This paper presents a rapid deployment approach utilizing automated machine learning methods, based on surgical action data collected from both experienced and trainee surgeons. The proposed approach effectively tackles the challenge of highly imbalanced class distributions, ensuring robust predictions across varying skill levels of surgeons. Additionally, our method partially incorporates model transparency, addressing the reliability requirements in medical applications. Compared to deep learning approaches, traditional machine learning models not only facilitate efficient rapid deployment but also offer significant advantages in interpretability. Through experiments, this study demonstrates the potential of this approach to provide quick, reliable and effective real-time detection in surgical training environments 

**Abstract (ZH)**: 在腹腔镜手术培训与评估中，实时检测手术动作并产生可解释的输出对于自动化和实时教学反馈及技能发展至关重要。这种能力将使指导性训练系统的开发成为可能。本文提出了一种基于自动机器学习方法的快速部署方法，利用从经验丰富的和实习外科医生处收集的手术动作数据。所提出的方法有效解决了类分布高度不平衡的挑战，确保了在不同水平外科医生的预测具有鲁棒性。此外，该方法部分实现了模型透明性，以应对医疗应用中的可靠性要求。与深度学习方法相比，传统机器学习模型不仅促进了高效快速部署，还提供了显著的可解释性优势。通过实验，本文证明了该方法在手术训练环境中提供快速、可靠和有效的实时检测潜力。 

---
# Calibration of Multiple Asynchronous Microphone Arrays using Hybrid TDOA 

**Title (ZH)**: 异步麦克风阵列基于混合TDOA的校准 

**Authors**: Chengjie Zhang, Wenda Pan, Xinyang Han, He Kong  

**Link**: [PDF](https://arxiv.org/pdf/2502.06195)  

**Abstract**: Accurate calibration of acoustic sensing systems made of multiple asynchronous microphone arrays is essential for satisfactory performance in sound source localization and tracking. State-of-the-art calibration methods for this type of system rely on the time difference of arrival and direction of arrival measurements among the microphone arrays (denoted as TDOA-M and DOA, respectively). In this paper, to enhance calibration accuracy, we propose to incorporate the time difference of arrival measurements between adjacent sound events (TDOAS) with respect to the microphone arrays. More specifically, we propose a two-stage calibration approach, including an initial value estimation (IVE) procedure and the final joint optimization step. The IVE stage first initializes all parameters except for microphone array orientations, using hybrid TDOA (i.e., TDOAM and TDOA-S), odometer data from a moving robot carrying a speaker, and DOA. Subsequently, microphone orientations are estimated through the iterative closest point method. The final joint optimization step estimates multiple microphone array locations, orientations, time offsets, clock drift rates, and sound source locations simultaneously. Both simulation and experiment results show that for scenarios with low or moderate TDOA noise levels, our approach outperforms existing methods in terms of accuracy. All code and data are available at this https URL. 

**Abstract (ZH)**: 多异步麦克风阵列的声学传感系统精确校准对于声音源定位和跟踪的满意性能至关重要。本文为提高校准精度，提出了一种结合相邻声事件到达时间差测量（TDOAS）的两阶段校准方法，包括初步值估计（IVE）程序和最终联合优化步骤。 

---
# Kalman Filter-Based Distributed Gaussian Process for Unknown Scalar Field Estimation in Wireless Sensor Networks 

**Title (ZH)**: 基于卡尔曼滤波的分布式高斯过程未知标量场估计在无线传感器网络中 

**Authors**: Jaemin Seo, Geunsik Bae, Hyondong Oh  

**Link**: [PDF](https://arxiv.org/pdf/2502.05802)  

**Abstract**: In this letter, we propose an online scalar field estimation algorithm of unknown environments using a distributed Gaussian process (DGP) framework in wireless sensor networks (WSNs). While the kernel-based Gaussian process (GP) has been widely employed for estimating unknown scalar fields, its centralized nature is not well-suited for handling a large amount of data from WSNs. To overcome the limitations of the kernel-based GP, recent advancements in GP research focus on approximating kernel functions as products of E-dimensional nonlinear basis functions, which can handle large WSNs more efficiently in a distributed manner. However, this approach requires a large number of basis functions for accurate approximation, leading to increased computational and communication complexities. To address these complexity issues, the paper proposes a distributed GP framework by incorporating a Kalman filter scheme (termed as K-DGP), which scales linearly with the number of nonlinear basis functions. Moreover, we propose a new consensus protocol designed to handle the unique data transmission requirement residing in the proposed K-DGP framework. This protocol preserves the inherent elements in the form of a certain column in the nonlinear function matrix of the communicated message; it enables wireless sensors to cooperatively estimate the environment and reach the global consensus through distributed learning with faster convergence than the widely-used average consensus protocol. Simulation results demonstrate rapid consensus convergence and outstanding estimation accuracy achieved by the proposed K-DGP algorithm. The scalability and efficiency of the proposed approach are further demonstrated by online dynamic environment estimation using WSNs. 

**Abstract (ZH)**: 基于分布式高斯过程的无线传感器网络中未知环境在线标量场估计算法 

---
# Application of Artificial Intelligence (AI) in Civil Engineering 

**Title (ZH)**: 人工智能在土木工程中的应用 

**Authors**: Temitope Funmilayo Awolusi, Bernard Chukwuemeka Finbarrs-Ezema, Isaac Munachimdinamma Chukwudulue, Marc Azab  

**Link**: [PDF](https://arxiv.org/pdf/2502.06727)  

**Abstract**: Hard computing generally deals with precise data, which provides ideal solutions to problems. However, in the civil engineering field, amongst other disciplines, that is not always the case as real-world systems are continuously changing. Here lies the need to explore soft computing methods and artificial intelligence to solve civil engineering shortcomings. The integration of advanced computational models, including Artificial Neural Networks (ANNs), Fuzzy Logic, Genetic Algorithms (GAs), and Probabilistic Reasoning, has revolutionized the domain of civil engineering. These models have significantly advanced diverse sub-fields by offering innovative solutions and improved analysis capabilities. Sub-fields such as: slope stability analysis, bearing capacity, water quality and treatment, transportation systems, air quality, structural materials, etc. ANNs predict non-linearities and provide accurate estimates. Fuzzy logic uses an efficient decision-making process to provide a more precise assessment of systems. Lastly, while GAs optimizes models (based on evolutionary processes) for better outcomes, probabilistic reasoning lowers their statistical uncertainties. 

**Abstract (ZH)**: 软计算方法和人工智能在解决土木工程不足中的应用 

---
# A Frontier AI Risk Management Framework: Bridging the Gap Between Current AI Practices and Established Risk Management 

**Title (ZH)**: 前沿AI风险管理框架：弥合当前AI实践与成熟风险管理之间的差距 

**Authors**: Simeon Campos, Henry Papadatos, Fabien Roger, Chloé Touzet, Malcolm Murray, Otter Quarks  

**Link**: [PDF](https://arxiv.org/pdf/2502.06656)  

**Abstract**: The recent development of powerful AI systems has highlighted the need for robust risk management frameworks in the AI industry. Although companies have begun to implement safety frameworks, current approaches often lack the systematic rigor found in other high-risk industries. This paper presents a comprehensive risk management framework for the development of frontier AI that bridges this gap by integrating established risk management principles with emerging AI-specific practices. The framework consists of four key components: (1) risk identification (through literature review, open-ended red-teaming, and risk modeling), (2) risk analysis and evaluation using quantitative metrics and clearly defined thresholds, (3) risk treatment through mitigation measures such as containment, deployment controls, and assurance processes, and (4) risk governance establishing clear organizational structures and accountability. Drawing from best practices in mature industries such as aviation or nuclear power, while accounting for AI's unique challenges, this framework provides AI developers with actionable guidelines for implementing robust risk management. The paper details how each component should be implemented throughout the life-cycle of the AI system - from planning through deployment - and emphasizes the importance and feasibility of conducting risk management work prior to the final training run to minimize the burden associated with it. 

**Abstract (ZH)**: 最近强大人工智能系统的开发强调了AI行业需要 robust 风险管理框架。尽管公司已经开始实施安全框架，但当前的方法往往缺乏其他高风险行业所具有的系统严谨性。本文提出了一种综合的风险管理框架，以弥补这一差距，该框架将已有的风险管理原则与新兴的特定于AI的做法相结合。该框架包括四个关键组成部分：（1）风险识别（通过文献综述、开放性红队测试和风险建模），（2）使用定量指标和明确的阈值进行风险分析和评估，（3）通过缓解措施（如控制、部署控制和保证过程）进行风险管理，（4）通过明确的组织结构和责任建立风险治理。借鉴成熟行业如航空或核能的最佳实践，同时考虑到AI的独特挑战，该框架为AI开发者提供了实施 robust 风险管理的可操作指南。论文详细说明了在AI系统整个生命周期中（从规划到部署）应如何实施每个组成部分，并强调在最终训练运行之前开展风险管理工作的重要性，以减少其负担。 

---
# On the Impact of the Utility in Semivalue-based Data Valuation 

**Title (ZH)**: 基于半值的數據估值中效用的影响 

**Authors**: Mélissa Tamine, Benjamin Heymann, Patrick Loiseau, Maxime Vono  

**Link**: [PDF](https://arxiv.org/pdf/2502.06574)  

**Abstract**: Semivalue-based data valuation in machine learning (ML) quantifies the contribution of individual data points to a downstream ML task by leveraging principles from cooperative game theory and the notion of utility. While this framework has been used in practice for assessing data quality, our experiments reveal inconsistent valuation outcomes across different utilities, albeit all related to ML performance. Beyond raising concerns about the reliability of data valuation, this inconsistency is challenging to interpret, as it stems from the complex interaction of the utility with data points and semivalue weights, which has barely been studied in prior work. In this paper, we take a first step toward clarifying the utility impact on semivalue-based data valuation. Specifically, we provide geometric interpretations of this impact for a broad family of classification utilities, which includes the accuracy and the arithmetic mean. We introduce the notion of spatial signatures: given a semivalue, data points can be embedded into a two-dimensional space, and utility functions map to the dual of this space. This geometric perspective separates the influence of the dataset and semivalue from that of the utility, providing a theoretical explanation for the experimentally observed sensitivity of valuation outcomes to the utility choice. 

**Abstract (ZH)**: 基于半值函数的数据估值在机器学习中的研究：从合作博弈论原理和效用概念量化个体数据点对下游机器学习任务的贡献 

---
# Can We Trust AI Benchmarks? An Interdisciplinary Review of Current Issues in AI Evaluation 

**Title (ZH)**: AI基准可信吗？当前AI评估问题的跨学科评审 

**Authors**: Maria Eriksson, Erasmo Purificato, Arman Noroozian, Joao Vinagre, Guillaume Chaslot, Emilia Gomez, David Fernandez-Llorca  

**Link**: [PDF](https://arxiv.org/pdf/2502.06559)  

**Abstract**: Quantitative Artificial Intelligence (AI) Benchmarks have emerged as fundamental tools for evaluating the performance, capability, and safety of AI models and systems. Currently, they shape the direction of AI development and are playing an increasingly prominent role in regulatory frameworks. As their influence grows, however, so too does concerns about how and with what effects they evaluate highly sensitive topics such as capabilities, including high-impact capabilities, safety and systemic risks. This paper presents an interdisciplinary meta-review of about 100 studies that discuss shortcomings in quantitative benchmarking practices, published in the last 10 years. It brings together many fine-grained issues in the design and application of benchmarks (such as biases in dataset creation, inadequate documentation, data contamination, and failures to distinguish signal from noise) with broader sociotechnical issues (such as an over-focus on evaluating text-based AI models according to one-time testing logic that fails to account for how AI models are increasingly multimodal and interact with humans and other technical systems). Our review also highlights a series of systemic flaws in current benchmarking practices, such as misaligned incentives, construct validity issues, unknown unknowns, and problems with the gaming of benchmark results. Furthermore, it underscores how benchmark practices are fundamentally shaped by cultural, commercial and competitive dynamics that often prioritise state-of-the-art performance at the expense of broader societal concerns. By providing an overview of risks associated with existing benchmarking procedures, we problematise disproportionate trust placed in benchmarks and contribute to ongoing efforts to improve the accountability and relevance of quantitative AI benchmarks within the complexities of real-world scenarios. 

**Abstract (ZH)**: 定量人工智能基准：评估、挑战与改进 

---
# Tighter Value-Function Approximations for POMDPs 

**Title (ZH)**: 更紧致的价值函数近似方法 for POMDPs 

**Authors**: Merlijn Krale, Wietze Koops, Sebastian Junges, Thiago D. Simão, Nils Jansen  

**Link**: [PDF](https://arxiv.org/pdf/2502.06523)  

**Abstract**: Solving partially observable Markov decision processes (POMDPs) typically requires reasoning about the values of exponentially many state beliefs. Towards practical performance, state-of-the-art solvers use value bounds to guide this reasoning. However, sound upper value bounds are often computationally expensive to compute, and there is a tradeoff between the tightness of such bounds and their computational cost. This paper introduces new and provably tighter upper value bounds than the commonly used fast informed bound. Our empirical evaluation shows that, despite their additional computational overhead, the new upper bounds accelerate state-of-the-art POMDP solvers on a wide range of benchmarks. 

**Abstract (ZH)**: 解决部分可观测马尔可夫决策过程（POMDPs）通常需要推断指数级状态信念的价值。为了实用性能，最先进求解器使用价值界来引导这种推断。然而，精确的上价值界通常计算成本高昂，并且界的质量与计算成本之间存在权衡。本文提出了新的、可证明更紧的上价值界，这些界比常用快速知情界更紧。实验证明，尽管这些新界具有额外的计算开销，但它们可以加速广泛基准上的最先进POMDP求解器。 

---
# Conditioning and AGM-like belief change in the Desirability-Indifference framework 

**Title (ZH)**: 在偏奋试度-无差别框架中的条件化与AGM类信念变更 

**Authors**: Kathelijne Coussement, Gert de Cooman, Keano De Vos  

**Link**: [PDF](https://arxiv.org/pdf/2502.06235)  

**Abstract**: We show how the AGM framework for belief change (expansion, revision, contraction) can be extended to deal with conditioning in the so-called Desirability-Indifference framework, based on abstract notions of accepting and rejecting options, as well as on abstract notions of events. This level of abstraction allows us to deal simultaneously with classical and quantum probability theory. 

**Abstract (ZH)**: 我们展示了如何将信念变更（扩展、修订、收缩）的AGM框架扩展到所称为欲望-无差异框架中处理条件问题，该框架基于接受和拒绝选项的抽象概念以及事件的抽象概念。这种抽象层次使得我们能够同时处理经典和量子概率理论。 

---
# The Value of Information in Human-AI Decision-making 

**Title (ZH)**: 人类与人工智能决策中的信息价值 

**Authors**: Ziyang Guo, Yifan Wu, Jason Hartline, Jessica Hullman  

**Link**: [PDF](https://arxiv.org/pdf/2502.06152)  

**Abstract**: Humans and AIs are often paired on decision tasks with the expectation of achieving complementary performance, where the combination of human and AI outperforms either one alone. However, how to improve performance of a human-AI team is often not clear without knowing more about what particular information and strategies each agent employs. We provide a decision-theoretic framework for characterizing the value of information -- and consequently, opportunities for agents to better exploit available information--in AI-assisted decision workflow. We demonstrate the use of the framework for model selection, empirical evaluation of human-AI performance, and explanation design. We propose a novel information-based instance-level explanation technique that adapts a conventional saliency-based explanation to explain information value in decision making. 

**Abstract (ZH)**: 人类和AI在决策任务中的配对增强：信息价值的决策理论框架及其应用 

---
# Managing Geological Uncertainty in Critical Mineral Supply Chains: A POMDP Approach with Application to U.S. Lithium Resources 

**Title (ZH)**: 管理关键矿产供应链中的地质不确定性：基于POMDP的方法及其在美国锂资源中的应用 

**Authors**: Mansur Arief, Yasmine Alonso, CJ Oshiro, William Xu, Anthony Corso, David Zhen Yin, Jef K. Caers, Mykel J. Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2502.05690)  

**Abstract**: The world is entering an unprecedented period of critical mineral demand, driven by the global transition to renewable energy technologies and electric vehicles. This transition presents unique challenges in mineral resource development, particularly due to geological uncertainty-a key characteristic that traditional supply chain optimization approaches do not adequately address. To tackle this challenge, we propose a novel application of Partially Observable Markov Decision Processes (POMDPs) that optimizes critical mineral sourcing decisions while explicitly accounting for the dynamic nature of geological uncertainty. Through a case study of the U.S. lithium supply chain, we demonstrate that POMDP-based policies achieve superior outcomes compared to traditional approaches, especially when initial reserve estimates are imperfect. Our framework provides quantitative insights for balancing domestic resource development with international supply diversification, offering policymakers a systematic approach to strategic decision-making in critical mineral supply chains. 

**Abstract (ZH)**: 全球正进入前所未有的关键矿产需求时期，推动这一趋势的是全球向可再生能源技术和电动汽车的转型。这一转型为矿产资源整合带来了独特挑战，特别是在地质不确定性方面，这是传统供应链优化方法未能充分应对的关键特征。为应对这一挑战，我们提出了一种新颖的应用部分可观测马尔可夫决策过程（POMDP）的方法，该方法在明确考虑地质不确定性动态性的同时优化关键矿产的采购决策。通过美国锂供应� geçen示例，我们证明基于POMDP的策略在初始资源储量估计不完善的条件下，比传统方法能取得更优的结果。我们的框架提供了定量分析国内资源开发与国际供应多元化之间平衡的见解，为政策制定者提供了一种系统化的关键矿产供应链战略决策方法。 

---
# Amorphous Fortress Online: Collaboratively Designing Open-Ended Multi-Agent AI and Game Environments 

**Title (ZH)**: 无序要塞在线：协作设计开放性多智能体AI和游戏环境 

**Authors**: M Charity, Mayu Wilson, Steven Lee, Dipika Rajesh, Sam Earle, Julian Togelius  

**Link**: [PDF](https://arxiv.org/pdf/2502.05632)  

**Abstract**: This work introduces Amorphous Fortress Online -- a web-based platform where users can design petri-dish-like environments and games consisting of multi-agent AI characters. Users can play, create, and share artificial life and game environments made up of microscopic but transparent finite-state machine agents that interact with each other. The website features multiple interactive editors and accessible settings to view the multi-agent interactions directly from the browser. This system serves to provide a database of thematically diverse AI and game environments that use the emergent behaviors of simple AI agents. 

**Abstract (ZH)**: This work introduces Amorphous Fortress Online——一个基于网页的平台，用户可以在其中设计类似培养皿的环境和游戏，包含多智能体AI角色。用户可以游玩、创造并分享由微观但透明的有限状态机代理组成的人工生命和游戏环境，这些代理能够相互交互。该网站配备了多个交互式编辑器和易于访问的设置，使用户可以直接在浏览器中查看多智能体的交互。该系统旨在提供一个包含各种主题的AI和游戏环境数据库，这些环境利用了简单AI代理的涌现行为。 

---
# Closing the Responsibility Gap in AI-based Network Management: An Intelligent Audit System Approach 

**Title (ZH)**: 基于智能审计系统的责任差距闭合在AI驱动的网络管理中的实现 

**Authors**: Emanuel Figetakis, Ahmed Refaey Hussein  

**Link**: [PDF](https://arxiv.org/pdf/2502.05608)  

**Abstract**: Existing network paradigms have achieved lower downtime as well as a higher Quality of Experience (QoE) through the use of Artificial Intelligence (AI)-based network management tools. These AI management systems, allow for automatic responses to changes in network conditions, lowering operation costs for operators, and improving overall performance. While adopting AI-based management tools enhance the overall network performance, it also introduce challenges such as removing human supervision, privacy violations, algorithmic bias, and model inaccuracies. Furthermore, AI-based agents that fail to address these challenges should be culpable themselves rather than the network as a whole. To address this accountability gap, a framework consisting of a Deep Reinforcement Learning (DRL) model and a Machine Learning (ML) model is proposed to identify and assign numerical values of responsibility to the AI-based management agents involved in any decision-making regarding the network conditions, which eventually affects the end-user. A simulation environment was created for the framework to be trained using simulated network operation parameters. The DRL model had a 96% accuracy during testing for identifying the AI-based management agents, while the ML model using gradient descent learned the network conditions at an 83% accuracy during testing. 

**Abstract (ZH)**: 现有的网络范式通过使用基于人工智能（AI）的网络管理工具，实现了更低的停机时间和更高质量的用户体验（QoE）。这些基于AI的管理系统能够自动应对网络条件的变化，降低运营成本，提高整体性能。虽然采用基于AI的管理工具可以提升整体网络性能，但也带来了移除人工监督、隐私侵犯、算法偏见和模型不准确等挑战。进一步地，未能解决这些挑战的基于AI的代理自身应该承担责任，而不是整个网络。为此，提出了一种框架，该框架由深度强化学习（DRL）模型和机器学习（ML）模型组成，以识别并为涉及任何网络条件决策的基于AI的管理代理分配责任的数值。该框架通过使用模拟的网络操作参数进行训练。在测试中，DRL模型在识别基于AI的管理代理方面达到了96%的准确性，而使用梯度下降的ML模型在测试中识别网络条件的准确性为83%。 

---
# ITBench: Evaluating AI Agents across Diverse Real-World IT Automation Tasks 

**Title (ZH)**: ITBench: 评估AI代理在多样化的实际IT自动化任务中的性能 

**Authors**: Saurabh Jha, Rohan Arora, Yuji Watanabe, Takumi Yanagawa, Yinfang Chen, Jackson Clark, Bhavya Bhavya, Mudit Verma, Harshit Kumar, Hirokuni Kitahara, Noah Zheutlin, Saki Takano, Divya Pathak, Felix George, Xinbo Wu, Bekir O. Turkkan, Gerard Vanloo, Michael Nidd, Ting Dai, Oishik Chatterjee, Pranjal Gupta, Suranjana Samanta, Pooja Aggarwal, Rong Lee, Pavankumar Murali, Jae-wook Ahn, Debanjana Kar, Ameet Rahane, Carlos Fonseca, Amit Paradkar, Yu Deng, Pratibha Moogi, Prateeti Mohapatra, Naoki Abe, Chandrasekhar Narayanaswami, Tianyin Xu, Lav R. Varshney, Ruchi Mahindru, Anca Sailer, Laura Shwartz, Daby Sow, Nicholas C. M. Fuller, Ruchir Puri  

**Link**: [PDF](https://arxiv.org/pdf/2502.05352)  

**Abstract**: Realizing the vision of using AI agents to automate critical IT tasks depends on the ability to measure and understand effectiveness of proposed solutions. We introduce ITBench, a framework that offers a systematic methodology for benchmarking AI agents to address real-world IT automation tasks. Our initial release targets three key areas: Site Reliability Engineering (SRE), Compliance and Security Operations (CISO), and Financial Operations (FinOps). The design enables AI researchers to understand the challenges and opportunities of AI agents for IT automation with push-button workflows and interpretable metrics. ITBench includes an initial set of 94 real-world scenarios, which can be easily extended by community contributions. Our results show that agents powered by state-of-the-art models resolve only 13.8% of SRE scenarios, 25.2% of CISO scenarios, and 0% of FinOps scenarios. We expect ITBench to be a key enabler of AI-driven IT automation that is correct, safe, and fast. 

**Abstract (ZH)**: 基于AI代理自动化的愿景实现取决于对其有效性进行测量和理解的能力。我们引入了ITBench框架，提供了一种系统的方法来基准测试AI代理以应对实际的IT自动化任务。我们的初始发布针对三个关键领域：系统可靠性工程（SRE）、合规与安全运营（CISO）和财务运营（FinOps）。该设计使AI研究人员能够通过一键式工作流和可解释的指标了解AI代理在IT自动化中的挑战和机遇。ITBench包括94个初始现实场景，可以通过社区贡献轻松扩展。我们的结果表明，基于最先进的模型的代理仅解决了13.8%的SRE场景、25.2%的CISO场景和0%的FinOps场景。我们预期ITBench将成为推动正确、安全和快速的AI驱动IT自动化的关键使能器。 

---
# RelGNN: Composite Message Passing for Relational Deep Learning 

**Title (ZH)**: RelGNN: 关系复合消息传递深度学习 

**Authors**: Tianlang Chen, Charilaos Kanatsoulis, Jure Leskovec  

**Link**: [PDF](https://arxiv.org/pdf/2502.06784)  

**Abstract**: Predictive tasks on relational databases are critical in real-world applications spanning e-commerce, healthcare, and social media. To address these tasks effectively, Relational Deep Learning (RDL) encodes relational data as graphs, enabling Graph Neural Networks (GNNs) to exploit relational structures for improved predictions. However, existing heterogeneous GNNs often overlook the intrinsic structural properties of relational databases, leading to modeling inefficiencies. Here we introduce RelGNN, a novel GNN framework specifically designed to capture the unique characteristics of relational databases. At the core of our approach is the introduction of atomic routes, which are sequences of nodes forming high-order tripartite structures. Building upon these atomic routes, RelGNN designs new composite message passing mechanisms between heterogeneous nodes, allowing direct single-hop interactions between them. This approach avoids redundant aggregations and mitigates information entanglement, ultimately leading to more efficient and accurate predictive modeling. RelGNN is evaluated on 30 diverse real-world tasks from RelBench (Fey et al., 2024), and consistently achieves state-of-the-art accuracy with up to 25% improvement. 

**Abstract (ZH)**: 关系数据库上的预测任务在电子商务、医疗保健和社交媒体等领域具有关键性。为有效地应对这些任务，关系深度学习（RDL）将关系数据编码为图形，使图神经网络（GNNs）能够利用关系结构以提高预测性能。然而，现有的异构GNNs往往忽视了关系数据库的固有结构属性，导致建模效率低下。为此，我们提出了一种新的GNN框架——RelGNN，专门用于捕捉关系数据库的独特特性。在该方法的核心是引入原子路由，即形成高阶三部结构的节点序列。基于这些原子路由，RelGNN设计了新颖的异构节点复合消息传递机制，允许它们之间进行直接单跳交互。这种方法避免了冗余聚合，减轻了信息纠缠，最终实现了更加高效和准确的预测建模。RelGNN在RelBench（Fey等，2024）的30个多样化真实世界任务上进行了评估，并在所有任务上均实现了最先进的准确性，最高提高了25%。 

---
# Rationalization Models for Text-to-SQL 

**Title (ZH)**: Text-to-SQL的合理性模型 

**Authors**: Gaetano Rossiello, Nhan Pham, Michael Glass, Junkyu Lee, Shankar Subramanian  

**Link**: [PDF](https://arxiv.org/pdf/2502.06759)  

**Abstract**: We introduce a framework for generating Chain-of-Thought (CoT) rationales to enhance text-to-SQL model fine-tuning. These rationales consist of intermediate SQL statements and explanations, serving as incremental steps toward constructing the final SQL query. The process begins with manually annotating a small set of examples, which are then used to prompt a large language model in an iterative, dynamic few-shot knowledge distillation procedure from a teacher model. A rationalization model is subsequently trained on the validated decomposed queries, enabling extensive synthetic CoT annotations for text-to-SQL datasets. To evaluate the approach, we fine-tune small language models with and without these rationales on the BIRD dataset. Results indicate that step-by-step query generation improves execution accuracy, especially for moderately and highly complex queries, while also enhancing explainability. 

**Abstract (ZH)**: 我们引入了一种生成链式思考（CoT）推理框架以增强文本到SQL模型微调。这些推理包括中间SQL语句及其解释，作为构建最终SQL查询的逐步步骤。过程始于手动标注一小部分示例，随后使用这些示例在迭代的动态少量样本知识精炼程序中提示一个大型语言模型，该程序源自一个教师模型。随后，通过在验证分解查询上训练一个合理化模型，可以为文本到SQL数据集生成大量的合成CoT注解。为了评估该方法，我们在BIRD数据集上使用带有和不带这些推理的小型语言模型进行微调。结果表明，逐步查询生成可以提高执行准确性，特别是在中等复杂度和高度复杂度的查询方面，同时也有助于提高可解释性。 

---
# What makes a good feedforward computational graph? 

**Title (ZH)**: 什么是好的前馈计算图？ 

**Authors**: Alex Vitvitskyi, João G. M. Araújo, Marc Lackenby, Petar Veličković  

**Link**: [PDF](https://arxiv.org/pdf/2502.06751)  

**Abstract**: As implied by the plethora of literature on graph rewiring, the choice of computational graph employed by a neural network can make a significant impact on its downstream performance. Certain effects related to the computational graph, such as under-reaching and over-squashing, may even render the model incapable of learning certain functions. Most of these effects have only been thoroughly studied in the domain of undirected graphs; however, recent years have seen a significant rise in interest in feedforward computational graphs: directed graphs without any back edges. In this paper, we study the desirable properties of a feedforward computational graph, discovering two important complementary measures: fidelity and mixing time, and evaluating a few popular choices of graphs through the lens of these measures. Our study is backed by both theoretical analyses of the metrics' asymptotic behaviour for various graphs, as well as correlating these metrics to the performance of trained neural network models using the corresponding graphs. 

**Abstract (ZH)**: 图重配文献中所隐含的意义在于，神经网络所采用的计算图的选择对其下游性能会产生显著影响。与计算图相关的一些效应，如未达到和过度挤压，甚至可能使模型无法学习某些函数。这些效应主要在无向图领域得到了充分研究；然而，在过去几年里，对前向计算图（无反向边的有向图）的兴趣显著增加。在这篇论文中，我们研究了前向计算图的 desirable 属性，发现两种重要的互补衡量标准：忠实度和混合时间，并通过这些衡量标准评估了几种流行的图的选择。我们的研究得到了这些度量的渐近行为的理论分析支持，并将这些度量与使用相应图训练的神经网络模型的性能相关联。 

---
# Low-power Spike-based Wearable Analytics on RRAM Crossbars 

**Title (ZH)**: 基于RRAM交叉bars的低功耗尖峰神经网络可穿戴分析 

**Authors**: Abhiroop Bhattacharjee, Jinquan Shi, Wei-Chen Chen, Xinxin Wang, Priyadarshini Panda  

**Link**: [PDF](https://arxiv.org/pdf/2502.06736)  

**Abstract**: This work introduces a spike-based wearable analytics system utilizing Spiking Neural Networks (SNNs) deployed on an In-memory Computing engine based on RRAM crossbars, which are known for their compactness and energy-efficiency. Given the hardware constraints and noise characteristics of the underlying RRAM crossbars, we propose online adaptation of pre-trained SNNs in real-time using Direct Feedback Alignment (DFA) against traditional backpropagation (BP). Direct Feedback Alignment (DFA) learning, that allows layer-parallel gradient computations, acts as a fast, energy & area-efficient method for online adaptation of SNNs on RRAM crossbars, unleashing better algorithmic performance against those adapted using BP. Through extensive simulations using our in-house hardware evaluation engine called DFA_Sim, we find that DFA achieves upto 64.1% lower energy consumption, 10.1% lower area overhead, and a 2.1x reduction in latency compared to BP, while delivering upto 7.55% higher inference accuracy on human activity recognition (HAR) tasks. 

**Abstract (ZH)**: 基于RRAM交叉开关的忆阻计算平台的.spike-触发可穿戴分析系统及其在线自适应学习方法 

---
# FlexDeMo: Decoupled Momentum Optimization for Fully and Hybrid Sharded Training 

**Title (ZH)**: FlexDeMo: 解耦动量优化用于全程和混合分片训练 

**Authors**: Mogens Henrik From, Jacob Nielsen, Lukas Galke, Peter Schneider-Kamp  

**Link**: [PDF](https://arxiv.org/pdf/2502.06728)  

**Abstract**: Training large neural network models requires extensive computational resources, often distributed across several nodes and accelerators. Recent findings suggest that it may be sufficient to only exchange the fast moving components of the gradients, while accumulating momentum locally (Decoupled Momentum, or DeMo). However, when considering larger models that do not fit on a single accelerate, the exchange of gradient information and the integration of DeMo needs to be reconsidered. Here, we propose employing a hybrid strategy, FlexDeMo, whereby nodes fully synchronize locally between different GPUs and inter-node communication is improved through only using the fast-moving components. This effectively combines previous hybrid sharding strategies with the advantages of decoupled momentum. Our experimental results show that FlexDeMo is on par with AdamW in terms of validation loss, demonstrating its viability. 

**Abstract (ZH)**: 使用FlexDeMo策略训练大型神经网络模型时，通过局部完全同步不同GPU并在节点间通信中仅使用快速移动的梯度组件来提升性能，这一策略结合了混合分割策略的优势并采用解藕动量，实验结果显示其验证损失与AdamW相当，证明了其可行性。 

---
# Recent Advances, Applications and Open Challenges in Machine Learning for Health: Reflections from Research Roundtables at ML4H 2024 Symposium 

**Title (ZH)**: 机器学习在健康领域的发展、应用和开放挑战：来自2024 ML4H研讨会圆桌论坛的反思 

**Authors**: Amin Adibi, Xu Cao, Zongliang Ji, Jivat Neet Kaur, Winston Chen, Elizabeth Healey, Brighton Nuwagira, Wenqian Ye, Geoffrey Woollard, Maxwell A Xu, Hejie Cui, Johnny Xi, Trenton Chang, Vasiliki Bikia, Nicole Zhang, Ayush Noori, Yuan Xia, Md. Belal Hossain, Hanna A. Frank, Alina Peluso, Yuan Pu, Shannon Zejiang Shen, John Wu, Adibvafa Fallahpour, Sazan Mahbub, Ross Duncan, Yuwei Zhang, Yurui Cao, Zuheng Xu, Michael Craig, Rahul G. Krishnan, Rahmatollah Beheshti, James M. Rehg, Mohammad Ehsanul Karim, Megan Coffee, Leo Anthony Celi, Jason Alan Fries, Mohsen Sadatsafavi, Dennis Shung, Shannon McWeeney, Jessica Dafflon, Sarah Jabbour  

**Link**: [PDF](https://arxiv.org/pdf/2502.06693)  

**Abstract**: The fourth Machine Learning for Health (ML4H) symposium was held in person on December 15th and 16th, 2024, in the traditional, ancestral, and unceded territories of the Musqueam, Squamish, and Tsleil-Waututh Nations in Vancouver, British Columbia, Canada. The symposium included research roundtable sessions to foster discussions between participants and senior researchers on timely and relevant topics for the ML4H community. The organization of the research roundtables at the conference involved 13 senior and 27 junior chairs across 13 tables. Each roundtable session included an invited senior chair (with substantial experience in the field), junior chairs (responsible for facilitating the discussion), and attendees from diverse backgrounds with an interest in the session's topic. 

**Abstract (ZH)**: 第四届医学机器学习（ML4H）研讨会于2024年12月15日至16日在加拿大不列颠哥伦比亚省温哥华的传统、祖先和未割让领土——穆斯夸姆、斯quamish和特利-奥图特民族的领地上举行。研讨会包括了研究圆桌会议环节，旨在促进参与者与资深研究人员就与ML4H社区相关的及时和相关话题进行讨论。会议期间的圆桌会议组织工作涉及13位资深主席和27位初级主席，共13个桌子。每个圆桌会议环节包括一位特邀资深主席（具有该领域丰富经验）、几位初级主席（负责促进讨论）以及来自不同背景并对会议主题感兴趣的参会者。 

---
# Multi-label Scandinavian Language Identification (SLIDE) 

**Title (ZH)**: 多标签斯堪的纳维亚语言识别（SLIDE） 

**Authors**: Mariia Fedorova, Jonas Sebulon Frydenberg, Victoria Handford, Victoria Ovedie Chruickshank Langø, Solveig Helene Willoch, Marthe Løken Midtgaard, Yves Scherrer, Petter Mæhlum, David Samuel  

**Link**: [PDF](https://arxiv.org/pdf/2502.06692)  

**Abstract**: Identifying closely related languages at sentence level is difficult, in particular because it is often impossible to assign a sentence to a single language. In this paper, we focus on multi-label sentence-level Scandinavian language identification (LID) for Danish, Norwegian Bokmål, Norwegian Nynorsk, and Swedish. We present the Scandinavian Language Identification and Evaluation, SLIDE, a manually curated multi-label evaluation dataset and a suite of LID models with varying speed-accuracy tradeoffs. We demonstrate that the ability to identify multiple languages simultaneously is necessary for any accurate LID method, and present a novel approach to training such multi-label LID models. 

**Abstract (ZH)**: 在句子层面识别紧密相关的语言具有挑战性，特别是因为往往无法将一个句子归属于单一语言。本文聚焦于丹麦语、挪威语（诺尔威genes克标准语和诺尔维genes新挪威语）和瑞典语的多标签句子级斯堪的纳维亚语言识别（LID）。我们介绍了斯堪的纳维亚语言识别与评估（SLIDE）数据集，这是一个手动策划的多标签评估数据集，以及具有不同速度-准确度权衡的LID模型系列。我们证明了同时识别多种语言的能力对于任何准确的LID方法都是必要的，并提出了一种训练此类多标签LID模型的新方法。 

---
# EquiTabPFN: A Target-Permutation Equivariant Prior Fitted Networks 

**Title (ZH)**: EquiTabPFN：目标置换等变先验拟合网络 

**Authors**: Michael Arbel, David Salinas, Frank Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2502.06684)  

**Abstract**: Recent foundational models for tabular data, such as TabPFN, have demonstrated remarkable effectiveness in adapting to new tasks through in-context learning. However, these models overlook a crucial equivariance property: the arbitrary ordering of target dimensions should not influence model predictions. In this study, we identify this oversight as a source of incompressible error, termed the equivariance gap, which introduces instability in predictions. To mitigate these issues, we propose a novel model designed to preserve equivariance across output dimensions. Our experimental results indicate that our proposed model not only addresses these pitfalls effectively but also achieves competitive benchmark performance. 

**Abstract (ZH)**: 近期用于表结构数据的基础模型，如TabPFN，通过上下文学习展示了在新任务上的显著适应性。然而，这些模型忽视了一个重要的不变性属性：目标维度的任意排列不应影响模型预测。本研究将这一疏忽识别为不可压缩错误的来源，称为不变性差距，它引入了预测的不稳定性。为缓解这些问题，我们提出了一种新模型，旨在在输出维度上保持不变性。我们的实验结果表明，我们提出的新模型不仅有效解决了这些问题，而且还实现了竞争力的基准性能。 

---
# Evaluation of Deep Audio Representations for Hearables 

**Title (ZH)**: 可穿戴设备中深度音频表示的评估 

**Authors**: Fabian Gröger, Pascal Baumann, Ludovic Amruthalingam, Laurent Simon, Ruksana Giurda, Simone Lionetti  

**Link**: [PDF](https://arxiv.org/pdf/2502.06664)  

**Abstract**: Effectively steering hearable devices requires understanding the acoustic environment around the user. In the computational analysis of sound scenes, foundation models have emerged as the state of the art to produce high-performance, robust, multi-purpose audio representations. We introduce and release Deep Evaluation of Audio Representations (DEAR), the first dataset and benchmark to evaluate the efficacy of foundation models in capturing essential acoustic properties for hearables. The dataset includes 1,158 audio tracks, each 30 seconds long, created by spatially mixing proprietary monologues with commercial, high-quality recordings of everyday acoustic scenes. Our benchmark encompasses eight tasks that assess the general context, speech sources, and technical acoustic properties of the audio scenes. Through our evaluation of four general-purpose audio representation models, we demonstrate that the BEATs model significantly surpasses its counterparts. This superiority underscores the advantage of models trained on diverse audio collections, confirming their applicability to a wide array of auditory tasks, including encoding the environment properties necessary for hearable steering. The DEAR dataset and associated code are available at this https URL. 

**Abstract (ZH)**: 有效引导可听设备要求理解用户周围的声学环境。在声场景的计算分析中，基础模型已发展成为最先进的方法，以生成高性能、鲁棒性和多用途的音频表示。我们引入并发布了Deep Evaluation of Audio Representations（DEAR），这是首个评估基础模型捕获可听设备关键声学特性的数据集和基准。该数据集包含1,158条音频轨道，每条30秒，由专有独白与商业高标准的日常生活声场景录音空间混合而成。我们的基准涵盖了八个任务，评估声场景的一般背景、对话源以及技术声学特性。通过四种通用音频表示模型的评估，我们证明了BEATs模型显著超越了其他模型。这一优势突显了在多样化的音频集合上训练的模型的优势，证实了它们在各种听觉任务中的适用性，包括编码对可听设备控制所必需的环境特性。DEAR数据集及关联代码可在以下链接访问：this https URL。 

---
# The 2021 Tokyo Olympics Multilingual News Article Dataset 

**Title (ZH)**: 2021东京东京奥运会多语言新闻文章数据集 

**Authors**: Erik Novak, Erik Calcina, Dunja Mladenić, Marko Grobelnik  

**Link**: [PDF](https://arxiv.org/pdf/2502.06648)  

**Abstract**: In this paper, we introduce a dataset of multilingual news articles covering the 2021 Tokyo Olympics. A total of 10,940 news articles were gathered from 1,918 different publishers, covering 1,350 sub-events of the 2021 Olympics, and published between July 1, 2021, and August 14, 2021. These articles are written in nine languages from different language families and in different scripts. To create the dataset, the raw news articles were first retrieved via a service that collects and analyzes news articles. Then, the articles were grouped using an online clustering algorithm, with each group containing articles reporting on the same sub-event. Finally, the groups were manually annotated and evaluated. The development of this dataset aims to provide a resource for evaluating the performance of multilingual news clustering algorithms, for which limited datasets are available. It can also be used to analyze the dynamics and events of the 2021 Tokyo Olympics from different perspectives. The dataset is available in CSV format and can be accessed from the this http URL repository. 

**Abstract (ZH)**: 本文介绍了涵盖2021东京奥运会的多语言新闻文章数据集。共收集了10940篇来自1918家不同出版商的新闻文章，覆盖了2021年奥运会的1350个子项目，发布时间为2021年7月1日至8月14日。这些文章使用了九种来自不同语系和不同文字脚本的语言。为创建数据集，原始新闻文章首先通过一个收集和分析新闻文章的服务获取。然后，使用在线聚类算法将文章分组，每组包含报道同一子项目的文章。最后，对手动注释和评估。该数据集的开发旨在提供一种评估多语言新闻聚类算法性能的资源，目前此类数据集相对匮乏。此外，该数据集也可用于从不同角度分析2021东京奥运会的动力和事件。数据集以CSV格式提供，并可通过以下链接访问：this http URL存储库。 

---
# Automatic Annotation Augmentation Boosts Translation between Molecules and Natural Language 

**Title (ZH)**: 自动注释增强提升分子与自然语言之间的翻译 

**Authors**: Zhiqiang Zhong, Simon Sataa-Yu Larsen, Haoyu Guo, Tao Tang, Kuangyu Zhou, Davide Mottin  

**Link**: [PDF](https://arxiv.org/pdf/2502.06634)  

**Abstract**: Recent advancements in AI for biological research focus on integrating molecular data with natural language to accelerate drug discovery. However, the scarcity of high-quality annotations limits progress in this area. This paper introduces LA$^3$, a Language-based Automatic Annotation Augmentation framework that leverages large language models to augment existing datasets, thereby improving AI training. We demonstrate the effectiveness of LA$^3$ by creating an enhanced dataset, LaChEBI-20, where we systematically rewrite the annotations of molecules from an established dataset. These rewritten annotations preserve essential molecular information while providing more varied sentence structures and vocabulary. Using LaChEBI-20, we train LaMolT5 based on a benchmark architecture to learn the mapping between molecular representations and augmented annotations.
Experimental results on text-based *de novo* molecule generation and molecule captioning demonstrate that LaMolT5 outperforms state-of-the-art models. Notably, incorporating LA$^3$ leads to improvements of up to 301% over the benchmark architecture. Furthermore, we validate the effectiveness of LA$^3$ notable applications in *image*, *text* and *graph* tasks, affirming its versatility and utility. 

**Abstract (ZH)**: Recent advancements in AI for biological research focus on integrating molecular data with natural language to accelerate drug discovery. However, the scarcity of high-quality annotations limits progress in this area. This paper introduces LA$^3$, a Language-based Automatic Annotation Augmentation framework that leverages large language models to augment existing datasets, thereby improving AI training. 

---
# Illegal Waste Detection in Remote Sensing Images: A Case Study 

**Title (ZH)**: 遥感图像中非法废物检测：一个案例研究 

**Authors**: Federico Gibellini, Piero Fraternali, Giacomo Boracchi, Luca Morandini, Andrea Diecidue, Simona Malegori  

**Link**: [PDF](https://arxiv.org/pdf/2502.06607)  

**Abstract**: Environmental crime currently represents the third largest criminal activity worldwide while threatening ecosystems as well as human health. Among the crimes related to this activity, improper waste management can nowadays be countered more easily thanks to the increasing availability and decreasing cost of Very-High-Resolution Remote Sensing images, which enable semi-automatic territory scanning in search of illegal landfills. This paper proposes a pipeline, developed in collaboration with professionals from a local environmental agency, for detecting candidate illegal dumping sites leveraging a classifier of Remote Sensing images. To identify the best configuration for such classifier, an extensive set of experiments was conducted and the impact of diverse image characteristics and training settings was thoroughly analyzed. The local environmental agency was then involved in an experimental exercise where outputs from the developed classifier were integrated in the experts' everyday work, resulting in time savings with respect to manual photo-interpretation. The classifier was eventually run with valuable results on a location outside of the training area, highlighting potential for cross-border applicability of the proposed pipeline. 

**Abstract (ZH)**: 环境犯罪目前是全球第三大犯罪活动，对生态系统及人类健康构成威胁。针对这种犯罪活动，与不当废物管理相关的犯罪如今可以通过不断增加且成本降低的高分辨率遥感图像更易于对抗，这些图像能够用于半自动地扫描领土，以寻找非法填埋场。本文提出了一种由当地环境机构的专业人员共同开发的流程，利用遥感图像分类器检测候选非法倾倒场地。为了确定此类分类器的最佳配置，进行了大量的实验，并且详细分析了各种图像特性和训练设置的影响。随后，当地环境机构参与了一次实验演习，将开发的分类器输出整合到专家的日常工作中，实现了与手动图像解释相比的时间节约。最终，分类器在训练区域外的位置产生了有价值的结果，突显了所提流程在跨境应用中的潜在可能性。 

---
# Amortized In-Context Bayesian Posterior Estimation 

**Title (ZH)**: amortized 在上下文中的贝叶斯后验估计 

**Authors**: Sarthak Mittal, Niels Leif Bracher, Guillaume Lajoie, Priyank Jaini, Marcus Brubaker  

**Link**: [PDF](https://arxiv.org/pdf/2502.06601)  

**Abstract**: Bayesian inference provides a natural way of incorporating prior beliefs and assigning a probability measure to the space of hypotheses. Current solutions rely on iterative routines like Markov Chain Monte Carlo (MCMC) sampling and Variational Inference (VI), which need to be re-run whenever new observations are available. Amortization, through conditional estimation, is a viable strategy to alleviate such difficulties and has been the guiding principle behind simulation-based inference, neural processes and in-context methods using pre-trained models. In this work, we conduct a thorough comparative analysis of amortized in-context Bayesian posterior estimation methods from the lens of different optimization objectives and architectural choices. Such methods train an amortized estimator to perform posterior parameter inference by conditioning on a set of data examples passed as context to a sequence model such as a transformer. In contrast to language models, we leverage permutation invariant architectures as the true posterior is invariant to the ordering of context examples. Our empirical study includes generalization to out-of-distribution tasks, cases where the assumed underlying model is misspecified, and transfer from simulated to real problems. Subsequently, it highlights the superiority of the reverse KL estimator for predictive problems, especially when combined with the transformer architecture and normalizing flows. 

**Abstract (ZH)**: 贝叶斯推断提供了一种自然地 Incorporating 先验信念并将概率测度赋予假设空间的方法。当前的方法依赖于马尔可夫链蒙特卡罗（MCMC）采样和变分推断（VI）等迭代过程，这些方法在新观测数据可用时需要重新运行。通过条件估计进行的学习化策略是缓解此类困难的有效方法，并且是基于模拟的推断、神经过程以及使用预训练模型的上下文方法背后的指导原则。在本文中，我们从不同的优化目标和网络架构视角对学习化的上下文贝叶斯后验估计方法进行了全面的比较分析。此类方法通过条件估计训练一个学习器，在序列模型（如变换器）输入的一组数据示例上下文中执行后验参数推断。与语言模型不同，我们利用不变排序架构，因为真实的后验在示例上下文的排序上是不变的。我们的实证研究包括对异常分布任务的一般化、假设底层模型指定错误的情况，以及从模拟问题转移到真实问题。随后的研究突显了逆KL估计器在预测问题中的优越性，尤其是在与变换器架构和归一化流结合使用时。 

---
# Evaluation of Multilingual Image Captioning: How far can we get with CLIP models? 

**Title (ZH)**: 多语言图像字幕评价：CLIP模型能带我们走多远？ 

**Authors**: Gonçalo Gomes, Chrysoula Zerva, Bruno Martins  

**Link**: [PDF](https://arxiv.org/pdf/2502.06600)  

**Abstract**: The evaluation of image captions, looking at both linguistic fluency and semantic correspondence to visual contents, has witnessed a significant effort. Still, despite advancements such as the CLIPScore metric, multilingual captioning evaluation has remained relatively unexplored. This work presents several strategies, and extensive experiments, related to evaluating CLIPScore variants in multilingual settings. To address the lack of multilingual test data, we consider two different strategies: (1) using quality aware machine-translated datasets with human judgements, and (2) re-purposing multilingual datasets that target semantic inference and reasoning. Our results highlight the potential of finetuned multilingual models to generalize across languages and to handle complex linguistic challenges. Tests with machine-translated data show that multilingual CLIPScore models can maintain a high correlation with human judgements across different languages, and additional tests with natively multilingual and multicultural data further attest to the high-quality assessments. 

**Abstract (ZH)**: 多语言图像说明评估：细说CLIPScore变体在多语言环境中的评价策略与实验 

---
# The Minimal Search Space for Conditional Causal Bandits 

**Title (ZH)**: 条件因果多臂问题的最小搜索空间 

**Authors**: Francisco N. F. Q. Simoes, Itai Feigenbaum, Mehdi Dastani, Thijs van Ommen  

**Link**: [PDF](https://arxiv.org/pdf/2502.06577)  

**Abstract**: Causal knowledge can be used to support decision-making problems. This has been recognized in the causal bandits literature, where a causal (multi-armed) bandit is characterized by a causal graphical model and a target variable. The arms are then interventions on the causal model, and rewards are samples of the target variable. Causal bandits were originally studied with a focus on hard interventions. We focus instead on cases where the arms are conditional interventions, which more accurately model many real-world decision-making problems by allowing the value of the intervened variable to be chosen based on the observed values of other variables. This paper presents a graphical characterization of the minimal set of nodes guaranteed to contain the optimal conditional intervention, which maximizes the expected reward. We then propose an efficient algorithm with a time complexity of $O(|V| + |E|)$ to identify this minimal set of nodes. We prove that the graphical characterization and the proposed algorithm are correct. Finally, we empirically demonstrate that our algorithm significantly prunes the search space and substantially accelerates convergence rates when integrated into standard multi-armed bandit algorithms. 

**Abstract (ZH)**: 因果知识可以用于支持决策问题。在这方面，因果 bandits 的文献已经有所认识，其中因果 bandits 通过因果图形模型和目标变量来表征。手臂则为对因果模型的干预，奖励为目标变量的样本。因果 bandits 原始的研究主要侧重于硬干预。我们相反地关注手臂为条件干预的情况，这更准确地 modeling 许多现实世界的决策问题，允许被干预变量的值基于其他观测变量的值来选择。本文提出了一个图化的表征，描述了包含最优条件干预（能够最大化期望奖励）的最小节点集合。随后，我们提出一个时间复杂度为 $O(|V| + |E|)$ 的高效算法来识别这个最小节点集合。我们证明了这个图化的表征和提出的算法是正确的。最后，我们通过实证研究证明，在集成到标准 bandits 算法中时，我们的算法能够显著减少搜索空间，并显著提高收敛速度。 

---
# Boost-and-Skip: A Simple Guidance-Free Diffusion for Minority Generation 

**Title (ZH)**: Boost-and-Skip: 一种简单的无引导扩散 minority 生成方法 

**Authors**: Soobin Um, Beomsu Kim, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2502.06516)  

**Abstract**: Minority samples are underrepresented instances located in low-density regions of a data manifold, and are valuable in many generative AI applications, such as data augmentation, creative content generation, etc. Unfortunately, existing diffusion-based minority generators often rely on computationally expensive guidance dedicated for minority generation. To address this, here we present a simple yet powerful guidance-free approach called Boost-and-Skip for generating minority samples using diffusion models. The key advantage of our framework requires only two minimal changes to standard generative processes: (i) variance-boosted initialization and (ii) timestep skipping. We highlight that these seemingly-trivial modifications are supported by solid theoretical and empirical evidence, thereby effectively promoting emergence of underrepresented minority features. Our comprehensive experiments demonstrate that Boost-and-Skip greatly enhances the capability of generating minority samples, even rivaling guidance-based state-of-the-art approaches while requiring significantly fewer computations. 

**Abstract (ZH)**: 少数样本是位于数据流形低密度区域的未代表实例，在生成式AI应用中如数据增强、创意内容生成等方面具有价值。现有基于扩散的少数样本生成器往往依赖于专门用于少数样本生成的计算昂贵的指导。为解决这一问题，我们提出了一种简单而强大的无指导方法，称为Boost-and-Skip，用于使用扩散模型生成少数样本。该框架的关键优势仅需要对标准生成过程进行两项最小更改：（i）方差放大初始化和（ii）时间步跳过。我们强调，这些看似简单的修改由坚实的理论和实证证据支持，从而有效地促进了未代表少数样本特征的出现。全面的实验表明，Boost-and-Skip显著增强了生成少数样本的能力，甚至在计算量显著减少的情况下，能够匹业界现有的基于指导的最先进方法。 

---
# Model-Based Offline Reinforcement Learning with Reliability-Guaranteed Sequence Modeling 

**Title (ZH)**: 基于模型的离线强化学习及可靠性保障序列建模 

**Authors**: Shenghong He  

**Link**: [PDF](https://arxiv.org/pdf/2502.06491)  

**Abstract**: Model-based offline reinforcement learning (MORL) aims to learn a policy by exploiting a dynamics model derived from an existing dataset. Applying conservative quantification to the dynamics model, most existing works on MORL generate trajectories that approximate the real data distribution to facilitate policy learning by using current information (e.g., the state and action at time step $t$). However, these works neglect the impact of historical information on environmental dynamics, leading to the generation of unreliable trajectories that may not align with the real data distribution. In this paper, we propose a new MORL algorithm \textbf{R}eliability-guaranteed \textbf{T}ransformer (RT), which can eliminate unreliable trajectories by calculating the cumulative reliability of the generated trajectory (i.e., using a weighted variational distance away from the real data). Moreover, by sampling candidate actions with high rewards, RT can efficiently generate high-return trajectories from the existing offline data. We theoretically prove the performance guarantees of RT in policy learning, and empirically demonstrate its effectiveness against state-of-the-art model-based methods on several benchmark tasks. 

**Abstract (ZH)**: 基于模型的离线强化学习（MORL）旨在通过利用从现有数据集派生的动力学模型来学习策略。通过保守量化动力学模型，现有大多数MORL工作生成轨迹以近似现实数据分布，以便利用当前信息（例如时间步$t$的状态和动作）来辅助策略学习。然而，这些工作忽略了历史信息对环境动力学的影响，导致生成的轨迹可能无法与现实数据分布一致。本文提出了一种新的MORL算法——可靠性保证变换器（RT），该算法可以通过计算生成轨迹的累计可靠性（即使用加权变异距离远离真实数据）来消除不可靠的轨迹。此外，通过采样高回报的动作，RT可以从现有的离线数据高效地生成高回报轨迹。我们从理论上证明了RT在策略学习中的性能保证，并在多个基准任务上实验证明了其相对于最先进的基于模型的方法的有效性。 

---
# WyckoffDiff - A Generative Diffusion Model for Crystal Symmetry 

**Title (ZH)**: WyckoffDiff - 一种晶体对称性的生成扩散模型 

**Authors**: Filip Ekström Kelvinius, Oskar B. Andersson, Abhijith S. Parackal, Dong Qian, Rickard Armiento, Fredrik Lindsten  

**Link**: [PDF](https://arxiv.org/pdf/2502.06485)  

**Abstract**: Crystalline materials often exhibit a high level of symmetry. However, most generative models do not account for symmetry, but rather model each atom without any constraints on its position or element. We propose a generative model, Wyckoff Diffusion (WyckoffDiff), which generates symmetry-based descriptions of crystals. This is enabled by considering a crystal structure representation that encodes all symmetry, and we design a novel neural network architecture which enables using this representation inside a discrete generative model framework. In addition to respecting symmetry by construction, the discrete nature of our model enables fast generation. We additionally present a new metric, Fréchet Wrenformer Distance, which captures the symmetry aspects of the materials generated, and we benchmark WyckoffDiff against recently proposed generative models for crystal generation. 

**Abstract (ZH)**: 晶体材料通常表现出高度的对称性。然而，大多数生成模型并未考虑对称性，而是对每个原子的位置或元素不做任何约束地进行建模。我们提出了一种生成模型Wyckoff Diff (WyckoffDiff)，它可以生成基于对称性的晶体描述。这通过考虑一个包含所有对称性的晶体结构表示来实现，并且我们设计了一种新的神经网络架构，使其能够在离散生成模型框架内部使用这种表示。除了通过设计本身尊重对称性外，我们模型的离散性质还使得生成过程快速。我们还提出了一个新的度量标准Fréchet Wrenformer Distance，它可以捕捉所生成材料的对称性方面，并将WyckoffDiff与最近提出的晶体生成生成模型进行了基准测试。 

---
# Testing software for non-discrimination: an updated and extended audit in the Italian car insurance domain 

**Title (ZH)**: 测试软件是否存在歧视：意大利汽车保险领域更新和扩展的审计 

**Authors**: Marco Rondina, Antonio Vetrò, Riccardo Coppola, Oumaima Regragrui, Alessandro Fabris, Gianmaria Silvello, Gian Antonio Susto, Juan Carlos De Martin  

**Link**: [PDF](https://arxiv.org/pdf/2502.06439)  

**Abstract**: Context. As software systems become more integrated into society's infrastructure, the responsibility of software professionals to ensure compliance with various non-functional requirements increases. These requirements include security, safety, privacy, and, increasingly, non-discrimination.
Motivation. Fairness in pricing algorithms grants equitable access to basic services without discriminating on the basis of protected attributes.
Method. We replicate a previous empirical study that used black box testing to audit pricing algorithms used by Italian car insurance companies, accessible through a popular online system. With respect to the previous study, we enlarged the number of tests and the number of demographic variables under analysis.
Results. Our work confirms and extends previous findings, highlighting the problematic permanence of discrimination across time: demographic variables significantly impact pricing to this day, with birthplace remaining the main discriminatory factor against individuals not born in Italian cities. We also found that driver profiles can determine the number of quotes available to the user, denying equal opportunities to all.
Conclusion. The study underscores the importance of testing for non-discrimination in software systems that affect people's everyday lives. Performing algorithmic audits over time makes it possible to evaluate the evolution of such algorithms. It also demonstrates the role that empirical software engineering can play in making software systems more accountable. 

**Abstract (ZH)**: 上下文。随着软件系统越来越多地融入社会基础设施，软件专业人士确保满足各种非功能性要求的责任不断增加。这些要求包括安全、安全、隐私，以及越来越重要的非歧视。

动机。价格算法中的公平性确保基本服务的平等访问，而不过度依赖受保护属性进行歧视。

方法。我们复制了之前的一项实地研究，该研究使用黑盒测试审计意大利汽车保险公司通过一个流行的在线系统使用的定价算法。与之前的研究所相比，我们扩大了测试的范围和分析的种族变量的数量。

结果。我们的研究确认并扩展了先前的研究结果，突出了歧视在时间上的持续性问题：到目前为止，种族变量仍然显著影响价格，出生地仍然是非意大利城市出生的人士的主要歧视因素。我们还发现，驾驶员档案可以决定用户可以获得的报价数量，从而剥夺了一切平等的机会。

结论。该研究强调了在影响人们日常生活软件系统中测试非歧视的重要性。随着时间的推移进行算法审计，可以评估此类算法的发展。它还表明经验软件工程如何使软件系统更具问责性。 

---
# Prompt-SID: Learning Structural Representation Prompt via Latent Diffusion for Single-Image Denoising 

**Title (ZH)**: Prompt-SID：基于潜在扩散学习结构表示的单图像去噪 

**Authors**: Huaqiu Li, Wang Zhang, Xiaowan Hu, Tao Jiang, Zikang Chen, Haoqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.06432)  

**Abstract**: Many studies have concentrated on constructing supervised models utilizing paired datasets for image denoising, which proves to be expensive and time-consuming. Current self-supervised and unsupervised approaches typically rely on blind-spot networks or sub-image pairs sampling, resulting in pixel information loss and destruction of detailed structural information, thereby significantly constraining the efficacy of such methods. In this paper, we introduce Prompt-SID, a prompt-learning-based single image denoising framework that emphasizes preserving of structural details. This approach is trained in a self-supervised manner using downsampled image pairs. It captures original-scale image information through structural encoding and integrates this prompt into the denoiser. To achieve this, we propose a structural representation generation model based on the latent diffusion process and design a structural attention module within the transformer-based denoiser architecture to decode the prompt. Additionally, we introduce a scale replay training mechanism, which effectively mitigates the scale gap from images of different resolutions. We conduct comprehensive experiments on synthetic, real-world, and fluorescence imaging datasets, showcasing the remarkable effectiveness of Prompt-SID. 

**Abstract (ZH)**: 基于提示学习的单图像去噪框架：Prompt-SID 

---
# CS-SHAP: Extending SHAP to Cyclic-Spectral Domain for Better Interpretability of Intelligent Fault Diagnosis 

**Title (ZH)**: CS-SHAP: 将 SHAP 扩展到循环谱域以提高智能故障诊断的可解释性 

**Authors**: Qian Chen, Xingjian Dong, Kui Hu, Kangkang Chen, Zhike Peng, Guang Meng  

**Link**: [PDF](https://arxiv.org/pdf/2502.06424)  

**Abstract**: Neural networks (NNs), with their powerful nonlinear mapping and end-to-end capabilities, are widely applied in mechanical intelligent fault diagnosis (IFD). However, as typical black-box models, they pose challenges in understanding their decision basis and logic, limiting their deployment in high-reliability scenarios. Hence, various methods have been proposed to enhance the interpretability of IFD. Among these, post-hoc approaches can provide explanations without changing model architecture, preserving its flexibility and scalability. However, existing post-hoc methods often suffer from limitations in explanation forms. They either require preprocessing that disrupts the end-to-end nature or overlook fault mechanisms, leading to suboptimal explanations. To address these issues, we derived the cyclic-spectral (CS) transform and proposed the CS-SHAP by extending Shapley additive explanations (SHAP) to the CS domain. CS-SHAP can evaluate contributions from both carrier and modulation frequencies, aligning more closely with fault mechanisms and delivering clearer and more accurate explanations. Three datasets are utilized to validate the superior interpretability of CS-SHAP, ensuring its correctness, reproducibility, and practical performance. With open-source code and outstanding interpretability, CS-SHAP has the potential to be widely adopted and become the post-hoc interpretability benchmark in IFD, even in other classification tasks. The code is available on this https URL. 

**Abstract (ZH)**: 神经网络在机械智能故障诊断中的可解释性增强：基于周期谱变换的CS-SHAP方法 

---
# Solving Linear-Gaussian Bayesian Inverse Problems with Decoupled Diffusion Sequential Monte Carlo 

**Title (ZH)**: 解线性高斯贝叶斯逆问题的解耦扩散顺序蒙特卡洛方法 

**Authors**: Filip Ekström Kelvinius, Zheng Zhao, Fredrik Lindsten  

**Link**: [PDF](https://arxiv.org/pdf/2502.06379)  

**Abstract**: A recent line of research has exploited pre-trained generative diffusion models as priors for solving Bayesian inverse problems. We contribute to this research direction by designing a sequential Monte Carlo method for linear-Gaussian inverse problems which builds on ``decoupled diffusion", where the generative process is designed such that larger updates to the sample are possible. The method is asymptotically exact and we demonstrate the effectiveness of our Decoupled Diffusion Sequential Monte Carlo (DDSMC) algorithm on both synthetic data and image reconstruction tasks. Further, we demonstrate how the approach can be extended to discrete data. 

**Abstract (ZH)**: 最近的研究方向利用预训练生成扩散模型作为先验来解决贝叶斯逆问题。我们在此研究方向上做出了贡献，设计了一种基于“解耦扩散”的序列蒙特卡洛方法，用于线性高斯逆问题，该方法允许对样本进行更大规模的更新。该方法在数值上是精确的，并通过合成数据和图像重构任务证明了Decoupled Diffusion Sequential Monte Carlo (DDSMC) 算法的有效性。此外，我们展示了该方法如何扩展到离散数据。 

---
# Hyperparameters in Score-Based Membership Inference Attacks 

**Title (ZH)**: 基于分数的成员推断攻击中的超参数研究 

**Authors**: Gauri Pradhan, Joonas Jälkö, Marlon Tobaben, Antti Honkela  

**Link**: [PDF](https://arxiv.org/pdf/2502.06374)  

**Abstract**: Membership Inference Attacks (MIAs) have emerged as a valuable framework for evaluating privacy leakage by machine learning models. Score-based MIAs are distinguished, in particular, by their ability to exploit the confidence scores that the model generates for particular inputs. Existing score-based MIAs implicitly assume that the adversary has access to the target model's hyperparameters, which can be used to train the shadow models for the attack. In this work, we demonstrate that the knowledge of target hyperparameters is not a prerequisite for MIA in the transfer learning setting. Based on this, we propose a novel approach to select the hyperparameters for training the shadow models for MIA when the attacker has no prior knowledge about them by matching the output distributions of target and shadow models. We demonstrate that using the new approach yields hyperparameters that lead to an attack near indistinguishable in performance from an attack that uses target hyperparameters to train the shadow models. Furthermore, we study the empirical privacy risk of unaccounted use of training data for hyperparameter optimization (HPO) in differentially private (DP) transfer learning. We find no statistically significant evidence that performing HPO using training data would increase vulnerability to MIA. 

**Abstract (ZH)**: 基于迁移学习的成员推理攻击：无需目标模型超参数的知识提出一种新颖的阴影模型超参数选择方法及其隐私风险研究 

---
# Facial Analysis Systems and Down Syndrome 

**Title (ZH)**: 面部分析系统与唐氏综合症 

**Authors**: Marco Rondina, Fabiana Vinci, Antonio Vetrò, Juan Carlos De Martin  

**Link**: [PDF](https://arxiv.org/pdf/2502.06341)  

**Abstract**: The ethical, social and legal issues surrounding facial analysis technologies have been widely debated in recent years. Key critics have argued that these technologies can perpetuate bias and discrimination, particularly against marginalized groups. We contribute to this field of research by reporting on the limitations of facial analysis systems with the faces of people with Down syndrome: this particularly vulnerable group has received very little attention in the literature so far. This study involved the creation of a specific dataset of face images. An experimental group with faces of people with Down syndrome, and a control group with faces of people who are not affected by the syndrome. Two commercial tools were tested on the dataset, along three tasks: gender recognition, age prediction and face labelling. The results show an overall lower accuracy of prediction in the experimental group, and other specific patterns of performance differences: i) high error rates in gender recognition in the category of males with Down syndrome; ii) adults with Down syndrome were more often incorrectly labelled as children; iii) social stereotypes are propagated in both the control and experimental groups, with labels related to aesthetics more often associated with women, and labels related to education level and skills more often associated with men. These results, although limited in scope, shed new light on the biases that alter face classification when applied to faces of people with Down syndrome. They confirm the structural limitation of the technology, which is inherently dependent on the datasets used to train the models. 

**Abstract (ZH)**: 面部分析技术 Surrounding 的伦理、社会和法律问题已在近年广泛争论。本研究通过报告杜威综合征患者面部图像的局限性，为该领域研究做出了贡献：这一特别脆弱的群体在文献中受到的关注很少。本研究涉及创建特定的面部图像数据集，实验组包括杜威综合征患者的面部图像，对照组包括未受综合征影响的人的面部图像。测试了两种商业工具，进行了三项任务：性别识别、年龄预测和面部标记。结果表明，实验组的整体预测准确性较低，并且存在其他特定的性能差异：I）杜威综合征男性在性别识别中的高错误率；II）杜威综合征成人更常被错误地标注为儿童；III）社会刻板印象在控制组和实验组中普遍存在，与美学相关的标签更常与女性关联，与教育水平和技能相关的标签更常与男性关联。这些结果虽然范围有限，但仍为面部分类应用于杜威综合征患者面部时存在的偏见提供了新的见解，并确认了该技术的结构性限制，该限制在很大程度上依赖于用于训练模型的数据集。 

---
# Prompt-Driven Continual Graph Learning 

**Title (ZH)**: 提示驱动的持续图学习 

**Authors**: Qi Wang, Tianfei Zhou, Ye Yuan, Rui Mao  

**Link**: [PDF](https://arxiv.org/pdf/2502.06327)  

**Abstract**: Continual Graph Learning (CGL), which aims to accommodate new tasks over evolving graph data without forgetting prior knowledge, is garnering significant research interest. Mainstream solutions adopt the memory replay-based idea, ie, caching representative data from earlier tasks for retraining the graph model. However, this strategy struggles with scalability issues for constantly evolving graphs and raises concerns regarding data privacy. Inspired by recent advancements in the prompt-based learning paradigm, this paper introduces a novel prompt-driven continual graph learning (PROMPTCGL) framework, which learns a separate prompt for each incoming task and maintains the underlying graph neural network model fixed. In this way, PROMPTCGL naturally avoids catastrophic forgetting of knowledge from previous tasks. More specifically, we propose hierarchical prompting to instruct the model from both feature- and topology-level to fully address the variability of task graphs in dynamic continual learning. Additionally, we develop a personalized prompt generator to generate tailored prompts for each graph node while minimizing the number of prompts needed, leading to constant memory consumption regardless of the graph scale. Extensive experiments on four benchmarks show that PROMPTCGL achieves superior performance against existing CGL approaches while significantly reducing memory consumption. Our code is available at this https URL. 

**Abstract (ZH)**: 持续图学习（CGL）：基于提示驱动的持续图学习（PROMPTCGL）框架 

---
# Is an Ultra Large Natural Image-Based Foundation Model Superior to a Retina-Specific Model for Detecting Ocular and Systemic Diseases? 

**Title (ZH)**: 基于超大规模自然图像的基础模型是否优于专门针对视网膜的模型，用于检测眼内和全身疾病？ 

**Authors**: Qingshan Hou, Yukun Zhou, Jocelyn Hui Lin Goh, Ke Zou, Samantha Min Er Yew, Sahana Srinivasan, Meng Wang, Thaddaeus Lo, Xiaofeng Lei, Siegfried K. Wagner, Mark A. Chia, Dawei Yang, Hongyang Jiang, AnRan Ran, Rui Santos, Gabor Mark Somfai, Juan Helen Zhou, Haoyu Chen, Qingyu Chen, Carol Yim-Lui Cheung, Pearse A. Keane, Yih Chung Tham  

**Link**: [PDF](https://arxiv.org/pdf/2502.06289)  

**Abstract**: The advent of foundation models (FMs) is transforming medical domain. In ophthalmology, RETFound, a retina-specific FM pre-trained sequentially on 1.4 million natural images and 1.6 million retinal images, has demonstrated high adaptability across clinical applications. Conversely, DINOv2, a general-purpose vision FM pre-trained on 142 million natural images, has shown promise in non-medical domains. However, its applicability to clinical tasks remains underexplored. To address this, we conducted head-to-head evaluations by fine-tuning RETFound and three DINOv2 models (large, base, small) for ocular disease detection and systemic disease prediction tasks, across eight standardized open-source ocular datasets, as well as the Moorfields AlzEye and the UK Biobank datasets. DINOv2-large model outperformed RETFound in detecting diabetic retinopathy (AUROC=0.850-0.952 vs 0.823-0.944, across three datasets, all P<=0.007) and multi-class eye diseases (AUROC=0.892 vs. 0.846, P<0.001). In glaucoma, DINOv2-base model outperformed RETFound (AUROC=0.958 vs 0.940, P<0.001). Conversely, RETFound achieved superior performance over all DINOv2 models in predicting heart failure, myocardial infarction, and ischaemic stroke (AUROC=0.732-0.796 vs 0.663-0.771, all P<0.001). These trends persisted even with 10% of the fine-tuning data. These findings showcase the distinct scenarios where general-purpose and domain-specific FMs excel, highlighting the importance of aligning FM selection with task-specific requirements to optimise clinical performance. 

**Abstract (ZH)**: 基础模型的兴起正在变革医疗领域。在眼科领域，RETFound是一种针对黄斑特定的基础模型，依次在140万自然图像和160万视网膜图像上预训练，已经在临床应用中展示了高度的适应性。相反，DINOv2是一种通用视觉基础模型，在1.42亿自然图像上预训练，已经在非医疗领域展现了潜力，但其在临床任务中的应用仍待进一步探索。为解决这一问题，我们通过在八个标准化开源眼科数据集以及Moorfields AlzEye和UK Biobank数据集上对RETFound和三种DINOv2模型（大型、基线、小型）进行微调，进行了头对头的评估。在检测眼疾病和预测全身疾病任务中，DINOv2大型模型在检测糖尿病视网膜病变(AUROC=0.850-0.952 vs 0.823-0.944，所有P≤0.007)和多类别眼疾病(AUROC=0.892 vs 0.846，P<0.001)中表现优于RETFound。在青光眼中，DINOv2基线模型在检测方面优于RETFound(AUROC=0.958 vs 0.940，P<0.001)。相反，在预测心力衰竭、心肌梗死和缺血性中风方面，RETFound优于所有DINOv2模型(AUROC=0.732-0.796 vs 0.663-0.771，所有P<0.001)，即使使用10%的微调数据也是如此。这些发现展示了通用和领域特定基础模型各自的优势场景，强调了根据任务特定要求选择基础模型的重要性，以优化临床表现。 

---
# HODDI: A Dataset of High-Order Drug-Drug Interactions for Computational Pharmacovigilance 

**Title (ZH)**: HODDI：用于计算药 Vigilance 的高阶药物-药物相互作用数据集 

**Authors**: Zhaoying Wang, Yingdan Shi, Xiang Liu, Can Chen, Jun Wen, Ren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.06274)  

**Abstract**: Drug-side effect research is vital for understanding adverse reactions arising in complex multi-drug therapies. However, the scarcity of higher-order datasets that capture the combinatorial effects of multiple drugs severely limits progress in this field. Existing resources such as TWOSIDES primarily focus on pairwise interactions. To fill this critical gap, we introduce HODDI, the first Higher-Order Drug-Drug Interaction Dataset, constructed from U.S. Food and Drug Administration (FDA) Adverse Event Reporting System (FAERS) records spanning the past decade, to advance computational pharmacovigilance. HODDI contains 109,744 records involving 2,506 unique drugs and 4,569 unique side effects, specifically curated to capture multi-drug interactions and their collective impact on adverse effects. Comprehensive statistical analyses demonstrate HODDI's extensive coverage and robust analytical metrics, making it a valuable resource for studying higher-order drug relationships. Evaluating HODDI with multiple models, we found that simple Multi-Layer Perceptron (MLP) can outperform graph models, while hypergraph models demonstrate superior performance in capturing complex multi-drug interactions, further validating HODDI's effectiveness. Our findings highlight the inherent value of higher-order information in drug-side effect prediction and position HODDI as a benchmark dataset for advancing research in pharmacovigilance, drug safety, and personalized medicine. The dataset and codes are available at this https URL. 

**Abstract (ZH)**: 高阶药物相互作用数据集对于理解复杂多药治疗引起的不良反应至关重要。然而，缺乏能够捕捉多种药物组合效果的高阶数据集严重限制了该领域的进展。现有资源如TWOSIDES主要关注双药交互。为填补这一关键空白，我们引入了HODDI，这是首个高阶药物-药物交互数据集，从过去十年美国食品药品监督管理局不良事件报告系统（FAERS）记录中构建，以推进计算性药监科学。HODDI包含涉及2,506种独特药物和4,569种独特不良反应的109,744条记录，特别筛选以捕捉多药交互及其对不良反应的综合影响。全面的统计分析表明HODDI的广泛覆盖范围和稳健的分析指标，使其成为研究高阶药物关系的宝贵资源。评估HODDI时，我们发现简单的多层感知机（MLP）模型可以优于图模型，而超图模型在捕捉复杂多药交互方面表现出更优性能，进一步验证了HODDI的有效性。我们的研究结果强调了高阶信息在药物-不良反应预测中的固有价值，将HODDI定位为药监科学、药物安全和个人化医疗研究的基准数据集。数据集和代码可在以下链接获取。 

---
# Conditioning through indifference in quantum mechanics 

**Title (ZH)**: 在量子力学中通过对称性进行条件化 

**Authors**: Keano De Vos, Gert de Cooman  

**Link**: [PDF](https://arxiv.org/pdf/2502.06249)  

**Abstract**: We can learn (more) about the state a quantum system is in through measurements. We look at how to describe the uncertainty about a quantum system's state conditional on executing such measurements. We show that by exploiting the interplay between desirability, coherence and indifference, a general rule for conditioning can be derived. We then apply this rule to conditioning on measurement outcomes, and show how it generalises to conditioning on a set of measurement outcomes. 

**Abstract (ZH)**: 我们可以通过测量更多地了解量子系统所处的状态。我们研究如何在执行此类测量的前提下描述对量子系统状态的不确定性。我们展示了通过利用可欲性、相干性和无差别性之间的相互作用，可以推导出一般的条件化规则。然后将此规则应用于测量结果的条件化，并展示了它如何推广到一组测量结果的条件化。 

---
# Examining False Positives under Inference Scaling for Mathematical Reasoning 

**Title (ZH)**: 考察推理缩放对数学推理中假阳性的影响 

**Authors**: Yu Wang, Nan Yang, Liang Wang, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2502.06217)  

**Abstract**: Recent advancements in language models have led to significant improvements in mathematical reasoning across various benchmarks. However, most of these benchmarks rely on automatic evaluation methods that only compare final answers using heuristics, without verifying the underlying reasoning steps. This limitation results in false positive solutions, where models may produce correct final answers but with flawed deduction paths. In this paper, we systematically examine the prevalence of false positive solutions in mathematical problem solving for language models. We analyze the characteristics and extent of this issue across different open-source models, datasets of varying difficulty levels, and decoding strategies. Specifically, we explore how false positives influence the inference time scaling behavior of language models. Our experimental results reveal that: (1) false positive solutions persist across different models, datasets, and decoding methods, (2) sampling-based inference time scaling methods do not alleviate the problem, and (3) the pass@N evaluation metric is more susceptible to false positives, suggesting a significantly lower scaling ceiling than what automatic evaluations indicate. Additionally, we analyze specific instances of false positives and discuss potential limitations in self-improvement techniques and synthetic data generation under such conditions. 

**Abstract (ZH)**: Recent advancements in语言模型在各类基准测试中显著提升了数学推理能力。然而，这些基准测试大多依赖于仅通过启发式方法比较最终答案的自动评估方法，而不验证后续的推理步骤。这一局限性导致了虚假正面解决方案的产生，即模型可能产生正确的最终答案，但推理路径却有误。本文系统地探讨了语言模型在数学问题解决中虚假正面解决方案的普遍性。我们分析了这一问题在不同开源模型、不同难度级别的数据集以及不同解码策略下的特征和影响范围。具体而言，我们探讨了虚假正面解决方案如何影响语言模型的推理时间缩放行为。实验结果表明：（1）虚假正面解决方案在不同的模型、数据集和解码方法中普遍存在；（2）基于采样的推理时间缩放方法未能缓解这一问题；（3）pass@N评估指标更容易受到虚假正面解决方案的影响，表明其缩放上限显著低于自动评估所示。此外，我们分析了虚假正面解决方案的具体实例，并讨论了在这种条件下自我改进技术和合成数据生成可能存在的局限性。 

---
# Right Time to Learn:Promoting Generalization via Bio-inspired Spacing Effect in Knowledge Distillation 

**Title (ZH)**: 最佳学习时机：通过生物启发的间隔效应在知识蒸馏中促进泛化 

**Authors**: Guanglong Sun, Hongwei Yan, Liyuan Wang, Qian Li, Bo Lei, Yi Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2502.06192)  

**Abstract**: Knowledge distillation (KD) is a powerful strategy for training deep neural networks (DNNs). Although it was originally proposed to train a more compact ``student'' model from a large ``teacher'' model, many recent efforts have focused on adapting it to promote generalization of the model itself, such as online KD and self KD. % as an effective way Here, we propose an accessible and compatible strategy named Spaced KD to improve the effectiveness of both online KD and self KD, in which the student model distills knowledge from a teacher model trained with a space interval ahead. This strategy is inspired by a prominent theory named \emph{spacing effect} in biological learning and memory, positing that appropriate intervals between learning trials can significantly enhance learning performance. With both theoretical and empirical analyses, we demonstrate that the benefits of the proposed Spaced KD stem from convergence to a flatter loss landscape during stochastic gradient descent (SGD). We perform extensive experiments to validate the effectiveness of Spaced KD in improving the learning performance of DNNs (e.g., the performance gain is up to 2.31\% and 3.34\% on Tiny-ImageNet over online KD and self KD, respectively). 

**Abstract (ZH)**: 空间间隔知识蒸馏（Spaced KD）：提高在线知识蒸馏和自我知识蒸馏效果的策略 

---
# Discourse-Driven Evaluation: Unveiling Factual Inconsistency in Long Document Summarization 

**Title (ZH)**: 基于话语驱动的评估：揭示长文档摘要中的事实不一致性 

**Authors**: Yang Zhong, Diane Litman  

**Link**: [PDF](https://arxiv.org/pdf/2502.06185)  

**Abstract**: Detecting factual inconsistency for long document summarization remains challenging, given the complex structure of the source article and long summary length. In this work, we study factual inconsistency errors and connect them with a line of discourse analysis. We find that errors are more common in complex sentences and are associated with several discourse features. We propose a framework that decomposes long texts into discourse-inspired chunks and utilizes discourse information to better aggregate sentence-level scores predicted by natural language inference models. Our approach shows improved performance on top of different model baselines over several evaluation benchmarks, covering rich domains of texts, focusing on long document summarization. This underscores the significance of incorporating discourse features in developing models for scoring summaries for long document factual inconsistency. 

**Abstract (ZH)**: 长文档摘要中检测事实不一致仍然具有挑战性，鉴于源文章的复杂结构和长摘要长度。在此工作中，我们研究了事实不一致错误，并将其与论述分析线联系起来。我们发现，错误在复杂句子中更为常见，并与几种论述特征相关。我们提出了一种框架，该框架将长文本分解为受论述启发的块，并利用论述信息来更好地聚合自然语言推理模型预测的句子级别得分。我们的方法在多个评价基准上优于不同的模型基线，涵盖多种文本领域，重点关注长文档摘要。这强调了在开发评分摘要模型以处理长文档事实不一致时整合论述特征的重要性。 

---
# RideKE: Leveraging Low-Resource, User-Generated Twitter Content for Sentiment and Emotion Detection in Kenyan Code-Switched Dataset 

**Title (ZH)**: RideKE：利用低资源用户生成的Twitter内容进行肯尼亚双语转换数据集的情感与情绪检测 

**Authors**: Naome A. Etori, Maria L. Gini  

**Link**: [PDF](https://arxiv.org/pdf/2502.06180)  

**Abstract**: Social media has become a crucial open-access platform for individuals to express opinions and share experiences. However, leveraging low-resource language data from Twitter is challenging due to scarce, poor-quality content and the major variations in language use, such as slang and code-switching. Identifying tweets in these languages can be difficult as Twitter primarily supports high-resource languages. We analyze Kenyan code-switched data and evaluate four state-of-the-art (SOTA) transformer-based pretrained models for sentiment and emotion classification, using supervised and semi-supervised methods. We detail the methodology behind data collection and annotation, and the challenges encountered during the data curation phase. Our results show that XLM-R outperforms other models; for sentiment analysis, XLM-R supervised model achieves the highest accuracy (69.2\%) and F1 score (66.1\%), XLM-R semi-supervised (67.2\% accuracy, 64.1\% F1 score). In emotion analysis, DistilBERT supervised leads in accuracy (59.8\%) and F1 score (31\%), mBERT semi-supervised (accuracy (59\% and F1 score 26.5\%). AfriBERTa models show the lowest accuracy and F1 scores. All models tend to predict neutral sentiment, with Afri-BERT showing the highest bias and unique sensitivity to empathy emotion. this https URL 

**Abstract (ZH)**: 社交媒体已成为个人表达意见和分享经验的重要开放访问平台。然而，由于推特上的低资源语言数据稀缺且质量较差，且语言使用存在极大差异，如俚语和语言转换，因此利用这些语言数据具有挑战性。识别这些语言的推文在推特上主要是支持高资源语言的情况下尤为困难。我们分析了肯尼亚的语言转换数据，并使用监督和半监督方法评估了四种最先进的变压器预训练模型在情感和情绪分类任务中的表现。我们详细介绍了数据收集和标注的方法，以及数据整理阶段遇到的挑战。结果显示，XLM-R表现出色；在情感分析中，XLM-R监督模型的准确率最高（69.2%）和F1分数最高（66.1%），XLM-R半监督模型的准确率为67.2%，F1分为64.1%。在情绪分析中，DistilBERT监督模型在准确率（59.8%）和F1分数（31%）上领先，mBERT半监督模型的准确率为59%，F1分为26.5%，AfriBERTa模型的准确率和F1分数最低。所有模型倾向于预测中性情感，AfriBERTa模型显示出最高的偏见和独特的情感共鸣敏感性。 

---
# An Interpretable Implicit-Based Approach for Modeling Local Spatial Effects: A Case Study of Global Gross Primary Productivity 

**Title (ZH)**: 基于隐式的方法对局部空间效应的可解释建模：全球初级生产力案例研究 

**Authors**: Siqi Du, Hongsheng Huang, Kaixin Shen, Ziqi Liu, Shengjun Tang  

**Link**: [PDF](https://arxiv.org/pdf/2502.06170)  

**Abstract**: In Earth sciences, unobserved factors exhibit non-stationary spatial distributions, causing the relationships between features and targets to display spatial heterogeneity. In geographic machine learning tasks, conventional statistical learning methods often struggle to capture spatial heterogeneity, leading to unsatisfactory prediction accuracy and unreliable interpretability. While approaches like Geographically Weighted Regression (GWR) capture local variations, they fall short of uncovering global patterns and tracking the continuous evolution of spatial heterogeneity. Motivated by this limitation, we propose a novel perspective - that is, simultaneously modeling common features across different locations alongside spatial differences using deep neural networks. The proposed method is a dual-branch neural network with an encoder-decoder structure. In the encoding stage, the method aggregates node information in a spatiotemporal conditional graph using GCN and LSTM, encoding location-specific spatiotemporal heterogeneity as an implicit conditional vector. Additionally, a self-attention-based encoder is used to extract location-invariant common features from the data. In the decoding stage, the approach employs a conditional generation strategy that predicts response variables and interpretative weights based on data features under spatiotemporal conditions. The approach is validated by predicting vegetation gross primary productivity (GPP) using global climate and land cover data from 2001 to 2020. Trained on 50 million samples and tested on 2.8 million, the proposed model achieves an RMSE of 0.836, outperforming LightGBM (1.063) and TabNet (0.944). Visualization analyses indicate that our method can reveal the distribution differences of the dominant factors of GPP across various times and locations. 

**Abstract (ZH)**: 在地球科学中，未观察到的因素表现出非平稳的空间分布，导致特征与目标之间的关系显示出空间异质性。在地理机器学习任务中，传统的统计学习方法往往难以捕捉空间异质性，导致预测精度不佳且解释性不可靠。虽然像地理加权回归（GWR）这样的方法能够捕捉局部变化，但它们在揭示全局模式和追踪空间异质性的连续演变方面存在不足。为克服这一局限，我们提出了一个新的视角——利用深度神经网络同时建模不同地点的共同特征和空间差异。所提出的方法是一种具有编码器-解码器结构的双分支神经网络。在编码阶段，该方法使用GCN和LSTM在网络时空条件图中聚合节点信息，将位置特定的时空异质性编码为隐式的条件向量。此外，还使用基于自注意力的编码器从数据中提取位置不变的共同特征。在解码阶段，该方法采用条件生成策略，在时空条件下预测响应变量和解释权重。该方法通过对2001年至2020年全球气候和土地覆盖数据进行植被净初级生产力（GPP）预测得到验证。该模型基于5000万样本训练，并在280万样本上进行测试，取得了RMSE为0.836的结果，优于LightGBM（1.063）和TabNet（0.944）。可视化分析表明，我们的方法能够揭示GPP主导因素在不同时空间分布的差异。 

---
# Low Tensor-Rank Adaptation of Kolmogorov--Arnold Networks 

**Title (ZH)**: Kolmogorov-Arnold网络的低张量秩适应 

**Authors**: Yihang Gao, Michael K. Ng, Vincent Y.F. Tan  

**Link**: [PDF](https://arxiv.org/pdf/2502.06153)  

**Abstract**: Kolmogorov--Arnold networks (KANs) have demonstrated their potential as an alternative to multi-layer perceptions (MLPs) in various domains, especially for science-related tasks. However, transfer learning of KANs remains a relatively unexplored area. In this paper, inspired by Tucker decomposition of tensors and evidence on the low tensor-rank structure in KAN parameter updates, we develop low tensor-rank adaptation (LoTRA) for fine-tuning KANs. We study the expressiveness of LoTRA based on Tucker decomposition approximations. Furthermore, we provide a theoretical analysis to select the learning rates for each LoTRA component to enable efficient training. Our analysis also shows that using identical learning rates across all components leads to inefficient training, highlighting the need for an adaptive learning rate strategy. Beyond theoretical insights, we explore the application of LoTRA for efficiently solving various partial differential equations (PDEs) by fine-tuning KANs. Additionally, we propose Slim KANs that incorporate the inherent low-tensor-rank properties of KAN parameter tensors to reduce model size while maintaining superior performance. Experimental results validate the efficacy of the proposed learning rate selection strategy and demonstrate the effectiveness of LoTRA for transfer learning of KANs in solving PDEs. Further evaluations on Slim KANs for function representation and image classification tasks highlight the expressiveness of LoTRA and the potential for parameter reduction through low tensor-rank decomposition. 

**Abstract (ZH)**: Kolmogorov--Arnold网络（KANs）在各种领域中被证明是多层感知机（MLPs）的潜在替代方案，尤其是在科学任务方面。然而，KANs的迁移学习仍然是一块未充分探索的领域。本文受张量的Tucker分解及其在KAN参数更新中低张量秩结构证据的启发，我们开发了低张量秩适应（LoTRA）方法以对KANs进行微调。我们基于Tucker分解近似研究了LoTRA的表达能力，并提供了理论分析以选择每个LoTRA组件的适学习率以实现高效训练。我们的分析表明，所有组件使用相同的适学习率会导致训练效率低下，突显了适应性学习率策略的必要性。除了理论见解，我们还研究了LoTRA在通过微调KANs高效求解各种偏微分方程（PDEs）中的应用。此外，我们提出了Slim KANs，结合了KAN参数张量的固有低张量秩特性，以减小模型大小同时保持优异性能。实验结果验证了所提出的适学习率选择策略的有效性，并展示了LoTRA在KANs迁移学习中解决PDEs的有效性。进一步对Slim KANs用于函数表示和图像分类任务的评估强调了LoTRA的表达能力及其通过低张量秩分解减少参数量的潜力。 

---
# Powerformer: A Transformer with Weighted Causal Attention for Time-series Forecasting 

**Title (ZH)**: Powerformer：一种用于时间序列预测的加权因果注意力变换器 

**Authors**: Kareem Hegazy, Michael W. Mahoney, N. Benjamin Erichson  

**Link**: [PDF](https://arxiv.org/pdf/2502.06151)  

**Abstract**: Transformers have recently shown strong performance in time-series forecasting, but their all-to-all attention mechanism overlooks the (temporal) causal and often (temporally) local nature of data. We introduce Powerformer, a novel Transformer variant that replaces noncausal attention weights with causal weights that are reweighted according to a smooth heavy-tailed decay. This simple yet effective modification endows the model with an inductive bias favoring temporally local dependencies, while still allowing sufficient flexibility to learn the unique correlation structure of each dataset. Our empirical results demonstrate that Powerformer not only achieves state-of-the-art accuracy on public time-series benchmarks, but also that it offers improved interpretability of attention patterns. Our analyses show that the model's locality bias is amplified during training, demonstrating an interplay between time-series data and power-law-based attention. These findings highlight the importance of domain-specific modifications to the Transformer architecture for time-series forecasting, and they establish Powerformer as a strong, efficient, and principled baseline for future research and real-world applications. 

**Abstract (ZH)**: Powerformer：一种基于功率衰减的因果注意力机制的时间序列预测Transformer变体 

---
# Guided Exploration for Efficient Relational Model Learning 

**Title (ZH)**: 引导探索以实现高效关系模型学习 

**Authors**: Annie Feng, Nishanth Kumar, Tomas Lozano-Perez, Leslie Pack-Kaelbling  

**Link**: [PDF](https://arxiv.org/pdf/2502.06146)  

**Abstract**: Efficient exploration is critical for learning relational models in large-scale environments with complex, long-horizon tasks. Random exploration methods often collect redundant or irrelevant data, limiting their ability to learn accurate relational models of the environment. Goal-literal babbling (GLIB) improves upon random exploration by setting and planning to novel goals, but its reliance on random actions and random novel goal selection limits its scalability to larger domains. In this work, we identify the principles underlying efficient exploration in relational domains: (1) operator initialization with demonstrations that cover the distinct lifted effects necessary for planning and (2) refining preconditions to collect maximally informative transitions by selecting informative goal-action pairs and executing plans to them. To demonstrate these principles, we introduce Baking-Large, a challenging domain with extensive state-action spaces and long-horizon tasks. We evaluate methods using oracle-driven demonstrations for operator initialization and precondition-targeting guidance to efficiently gather critical transitions. Experiments show that both the oracle demonstrations and precondition-targeting oracle guidance significantly improve sample efficiency and generalization, paving the way for future methods to use these principles to efficiently learn accurate relational models in complex domains. 

**Abstract (ZH)**: 高效探索对于大规模环境中复杂、长期任务的学习关系模型至关重要。随机探索方法通常收集冗余或无关的数据，限制了它们学习环境准确关系模型的能力。目标实义 babbling (GLIB) 通过设定并将计划应用于新颖目标来改进随机探索，但其依赖于随机动作和随机新颖目标的选择限制了其在更大领域的规模化应用。在本文中，我们确定了关系领域高效探索的基本原则：(1) 通过涵盖规划所需的不同提升效果的示范进行操作初始化；(2) 通过选择具有信息性的目标-动作对并执行计划来收集最具有信息性的转换来细化先决条件。为了证明这些原则，我们引入了 Baking-Large，这是一个具有广泛状态-动作空间和长期任务的具有挑战性的领域。我们使用启发式驱动的示范进行操作初始化，并使用先决条件目标朝向指导来高效地收集关键转换。实验结果显示，启发式示范和先决条件目标朝向启发式指导显著提高了样本效率和泛化能力，为未来方法利用这些原则在复杂领域高效地学习准确的关系模型铺平了道路。 

---
# Graph Neural Networks at a Fraction 

**Title (ZH)**: 图神经网络 fractions 代价 

**Authors**: Rucha Bhalchandra Joshi, Sagar Prakash Barad, Nidhi Tiwari, Subhankar Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2502.06136)  

**Abstract**: Graph Neural Networks (GNNs) have emerged as powerful tools for learning representations of graph-structured data. In addition to real-valued GNNs, quaternion GNNs also perform well on tasks on graph-structured data. With the aim of reducing the energy footprint, we reduce the model size while maintaining accuracy comparable to that of the original-sized GNNs. This paper introduces Quaternion Message Passing Neural Networks (QMPNNs), a framework that leverages quaternion space to compute node representations. Our approach offers a generalizable method for incorporating quaternion representations into GNN architectures at one-fourth of the original parameter count. Furthermore, we present a novel perspective on Graph Lottery Tickets, redefining their applicability within the context of GNNs and QMPNNs. We specifically aim to find the initialization lottery from the subnetwork of the GNNs that can achieve comparable performance to the original GNN upon training. Thereby reducing the trainable model parameters even further. To validate the effectiveness of our proposed QMPNN framework and LTH for both GNNs and QMPNNs, we evaluate their performance on real-world datasets across three fundamental graph-based tasks: node classification, link prediction, and graph classification. 

**Abstract (ZH)**: Quaternion消息传递神经网络（QMPNNs）：一种四元数空间中的节点表示框架及GNN剪枝的新视角 

---
# Integrating Sequence and Image Modeling in Irregular Medical Time Series Through Self-Supervised Learning 

**Title (ZH)**: 通过自我监督学习将序列模型与图像模型集成到不规则医疗时间序列中 

**Authors**: Liuqing Chen, Shuhong Xiao, Shixian Ding, Shanhai Hu, Lingyun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.06134)  

**Abstract**: Medical time series are often irregular and face significant missingness, posing challenges for data analysis and clinical decision-making. Existing methods typically adopt a single modeling perspective, either treating series data as sequences or transforming them into image representations for further classification. In this paper, we propose a joint learning framework that incorporates both sequence and image representations. We also design three self-supervised learning strategies to facilitate the fusion of sequence and image representations, capturing a more generalizable joint representation. The results indicate that our approach outperforms seven other state-of-the-art models in three representative real-world clinical datasets. We further validate our approach by simulating two major types of real-world missingness through leave-sensors-out and leave-samples-out techniques. The results demonstrate that our approach is more robust and significantly surpasses other baselines in terms of classification performance. 

**Abstract (ZH)**: 医学时间序列数据往往不规则且存在严重的数据缺失，这对数据分析和临床决策构成了挑战。现有方法通常从单一建模视角出发，要么将序列数据视为序列，要么将其转换为图像表示以进行进一步的分类。本文提出了一种结合序列和图像表示的联合学习框架，并设计了三种自监督学习策略以促进序列和图像表示的融合，捕获更具泛化能力的联合表示。实验结果表明，本文方法在三个代表性的临床数据集中优于七个最先进的模型。我们还通过留传感器法和留样本法模拟了两种主要的真实世界缺失性，进一步验证了本文方法。结果表明，本文方法在分类性能上更具鲁棒性，并显著优于其他基线方法。 

---
# Foundation Model of Electronic Medical Records for Adaptive Risk Estimation 

**Title (ZH)**: 电子医疗记录的适应性风险估计基础模型 

**Authors**: Pawel Renc, Michal K. Grzeszczyk, Nassim Oufattole, Deirdre Goode, Yugang Jia, Szymon Bieganski, Matthew B. A. McDermott, Jaroslaw Was, Anthony E. Samir, Jonathan W. Cunningham, David W. Bates, Arkadiusz Sitek  

**Link**: [PDF](https://arxiv.org/pdf/2502.06124)  

**Abstract**: We developed the Enhanced Transformer for Health Outcome Simulation (ETHOS), an AI model that tokenizes patient health timelines (PHTs) from EHRs. ETHOS predicts future PHTs using transformer-based architectures. The Adaptive Risk Estimation System (ARES) employs ETHOS to compute dynamic and personalized risk probabilities for clinician-defined critical events. ARES incorporates a personalized explainability module that identifies key clinical factors influencing risk estimates for individual patients. ARES was evaluated on the MIMIC-IV v2.2 dataset in emergency department (ED) settings, benchmarking its performance against traditional early warning systems and machine learning models. We processed 299,721 unique patients from MIMIC-IV into 285,622 PHTs, with 60% including hospital admissions. The dataset contained over 357 million tokens. ETHOS outperformed benchmark models in predicting hospital admissions, ICU admissions, and prolonged hospital stays, achieving superior AUC scores. ETHOS-based risk estimates demonstrated robustness across demographic subgroups with strong model reliability, confirmed via calibration curves. The personalized explainability module provides insights into patient-specific factors contributing to risk. ARES, powered by ETHOS, advances predictive healthcare AI by providing dynamic, real-time, and personalized risk estimation with patient-specific explainability to enhance clinician trust. Its adaptability and superior accuracy position it as a transformative tool for clinical decision-making, potentially improving patient outcomes and resource allocation in emergency and inpatient settings. We release the full code at this http URL to facilitate future research. 

**Abstract (ZH)**: 我们开发了健康结局模拟增强变压器模型（ETHOS），这是一种AI模型，用于将电子健康记录（EHRs）中的患者健康时间线（PHTs）进行标记化。ETHOS使用基于变换器的架构来预测未来的PHTs。自适应风险估计系统（ARES）利用ETHOS计算由临床医生定义的关键事件的动态和个性化风险概率。ARES结合了一个个性化可解释性模块，用于识别影响个别患者风险估计的关键临床因素。ARES在包含紧急部门（ED）设置的MIMIC-IV v2.2数据集中进行了评估，将其性能与传统的早期预警系统和机器学习模型进行了对比。我们从MIMIC-IV中处理了299,721名独特患者，生成了285,622个PHTs，其中60%包括医院入院记录，数据集包含超过3.57亿个标记。ETHOS在预测医院入院、ICU入院和住院时间方面优于基准模型，取得了更好的AUC分数。基于ETHOS的风险估计在不同人口统计亚组中表现出色，具有强大的模型可靠性，这一点通过校准曲线得到了证实。个性化可解释性模块提供了有关患者特定因素对风险影响的见解。ARES通过提供动态、实时和个性化风险估计以及患者特定的可解释性，成为增强临床信任的先进预测医疗AI工具。其高度可适应性和卓越的准确度使其成为临床决策的变革性工具，有望改善紧急和住院设置中的患者结果和资源分配。我们在此网址发布完整代码以促进未来研究。 

---
# Revisiting Dynamic Graph Clustering via Matrix Factorization 

**Title (ZH)**: 基于矩阵分解 revisit 动态图聚类 

**Authors**: Dongyuan Li, Satoshi Kosugi, Ying Zhang, Manabu Okumura, Feng Xia, Renhe Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2502.06117)  

**Abstract**: Dynamic graph clustering aims to detect and track time-varying clusters in dynamic graphs, revealing the evolutionary mechanisms of complex real-world dynamic systems. Matrix factorization-based methods are promising approaches for this task; however, these methods often struggle with scalability and can be time-consuming when applied to large-scale dynamic graphs. Moreover, they tend to lack robustness and are vulnerable to real-world noisy data. To address these issues, we make three key contributions. First, to improve scalability, we propose temporal separated matrix factorization, where a single matrix is divided into multiple smaller matrices for independent factorization, resulting in faster computation. Second, to improve robustness, we introduce bi-clustering regularization, which jointly optimizes graph embedding and clustering, thereby filtering out noisy features from the graph embeddings. Third, to further enhance effectiveness and efficiency, we propose selective embedding updating, where we update only the embeddings of dynamic nodes while the embeddings of static nodes are fixed among different timestamps. Experimental results on six synthetic and five real-world benchmarks demonstrate the scalability, robustness and effectiveness of our proposed method. Source code is available at this https URL. 

**Abstract (ZH)**: 动态图聚类旨在检测和追踪动态图中的时间变化聚类，揭示复杂现实动态系统的演化机制。基于矩阵分解的方法是这一任务有前途的途径；然而，这些方法通常在处理大规模动态图时面临扩展性问题，并且计算耗时。此外，它们往往缺乏 robustness，容易受到现实世界噪声数据的影响。为了解决这些问题，我们做出了三项关键贡献。首先，为了提高扩展性，我们提出了时间分离的矩阵分解方法，即将一个矩阵分解为多个较小的矩阵进行独立分解，从而加快计算速度。其次，为了提高 robustness，我们引入了双聚类正则化，该方法通过同时优化图嵌入和聚类，从而从图嵌入中过滤出噪声特征。最后，为了进一步提高有效性和效率，我们提出了选择性嵌入更新方法，在不同时间戳上仅更新动态节点的嵌入，而静态节点的嵌入保持固定。在六个合成数据集和五个真实世界数据集上的实验结果证明了我们提出方法的扩展性、robustness和有效性。相关源代码可在以下链接获取。 

---
# Circuit-tuning: A Mechanistic Approach for Identifying Parameter Redundancy and Fine-tuning Neural Networks 

**Title (ZH)**: 电路调谐：一种机理方法用于识别参数冗余和精细调整神经网络 

**Authors**: Yueyan Li, Caixia Yuan, Xiaojie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.06106)  

**Abstract**: The study of mechanistic interpretability aims to reverse-engineer a model to explain its behaviors. While recent studies have focused on the static mechanism of a certain behavior, the training dynamics inside a model remain to be explored. In this work, we develop an interpretable method for fine-tuning and reveal the mechanism behind learning. We first propose the concept of node redundancy as an extension of intrinsic dimension and explain the idea behind circuit discovery from a fresh view. Based on the theory, we propose circuit-tuning, a two-stage algorithm that iteratively performs circuit discovery to mask out irrelevant edges and updates the remaining parameters responsible for a specific task. Experiments show that our method not only improves performance on a wide range of tasks but is also scalable while preserving general capabilities. We visualize and analyze the circuits before, during, and after fine-tuning, providing new insights into the self-organization mechanism of a neural network in the learning process. 

**Abstract (ZH)**: 机制可解释性研究旨在逆向工程模型以解释其行为。虽然近期的研究主要关注特定行为的静态机制，但模型内的训练动力学仍有待探索。在这项工作中，我们开发了一种可解释的方法进行微调，并揭示了学习背后的机制。我们首先提出了节点冗余的概念，将其作为固有维度的拓展，并从新视角解释了电路发现的想法。基于这一理论，我们提出了电路微调，这是一种两阶段算法，通过迭代进行电路发现以屏蔽无关边，并更新负责特定任务的剩余参数。实验表明，我们的方法不仅在多种任务上提高了性能，而且在保持通用能力的同时具有可扩展性。我们在微调前后可视化并分析电路，提供了对神经网络在学习过程中自组织机制的新见解。 

---
# Comprehensive Framework for Evaluating Conversational AI Chatbots 

**Title (ZH)**: 综合评价对话式AI聊天机器人框架 

**Authors**: Shailja Gupta, Rajesh Ranjan, Surya Narayan Singh  

**Link**: [PDF](https://arxiv.org/pdf/2502.06105)  

**Abstract**: Conversational AI chatbots are transforming industries by streamlining customer service, automating transactions, and enhancing user engagement. However, evaluating these systems remains a challenge, particularly in financial services, where compliance, user trust, and operational efficiency are critical. This paper introduces a novel evaluation framework that systematically assesses chatbots across four dimensions: cognitive and conversational intelligence, user experience, operational efficiency, and ethical and regulatory compliance. By integrating advanced AI methodologies with financial regulations, the framework bridges theoretical foundations and real-world deployment challenges. Additionally, we outline future research directions, emphasizing improvements in conversational coherence, real-time adaptability, and fairness. 

**Abstract (ZH)**: 基于对话的AI聊天机器人正通过简化客户服务、自动化交易和增强用户参与度来改造各行各业。然而，在金融服务业中，合规性、用户信任和运营效率至关重要，这使得评估这些系统成为一个挑战。本文提出了一种新的评估框架，系统地从四大维度评估聊天机器人：认知和对话智能、用户体验、运营效率以及伦理和法规合规性。通过整合先进的AI方法与金融监管，该框架弥合了理论基础与实际部署挑战之间的差距。此外，我们还概述了未来的研究方向，强调提高对话连贯性、实时适应性和公平性。 

---
# NLGR: Utilizing Neighbor Lists for Generative Rerank in Personalized Recommendation Systems 

**Title (ZH)**: NLGR：利用邻居列表进行个性化推荐系统中的生成式重排序 

**Authors**: Shuli Wang, Xue Wei, Senjie Kou, Chi Wang, Wenshuai Chen, Qi Tang, Yinhua Zhu, Xiong Xiao, Xingxing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.06097)  

**Abstract**: Reranking plays a crucial role in modern multi-stage recommender systems by rearranging the initial ranking list. Due to the inherent challenges of combinatorial search spaces, some current research adopts an evaluator-generator paradigm, with a generator generating feasible sequences and an evaluator selecting the best sequence based on the estimated list utility. However, these methods still face two issues. Firstly, due to the goal inconsistency problem between the evaluator and generator, the generator tends to fit the local optimal solution of exposure distribution rather than combinatorial space optimization. Secondly, the strategy of generating target items one by one is difficult to achieve optimality because it ignores the information of subsequent items.
To address these issues, we propose a utilizing Neighbor Lists model for Generative Reranking (NLGR), which aims to improve the performance of the generator in the combinatorial space. NLGR follows the evaluator-generator paradigm and improves the generator's training and generating methods. Specifically, we use neighbor lists in combination space to enhance the training process, making the generator perceive the relative scores and find the optimization direction. Furthermore, we propose a novel sampling-based non-autoregressive generation method, which allows the generator to jump flexibly from the current list to any neighbor list. Extensive experiments on public and industrial datasets validate NLGR's effectiveness and we have successfully deployed NLGR on the Meituan food delivery platform. 

**Abstract (ZH)**: 利用邻近列表的生成重排序模型（NLGR） 

---
# Post-detection inference for sequential changepoint localization 

**Title (ZH)**: 检测后序贯变化点定位的后推 inference 

**Authors**: Aytijhya Saha, Aaditya Ramdas  

**Link**: [PDF](https://arxiv.org/pdf/2502.06096)  

**Abstract**: This paper addresses a fundamental but largely unexplored challenge in sequential changepoint analysis: conducting inference following a detected change. We study the problem of localizing the changepoint using only the data observed up to a data-dependent stopping time at which a sequential detection algorithm $\mathcal A$ declares a change. We first construct confidence sets for the unknown changepoint when pre- and post-change distributions are assumed to be known. We then extend our framework to composite pre- and post-change scenarios. We impose no conditions on the observation space or on $\mathcal A$ -- we only need to be able to run $\mathcal A$ on simulated data sequences. In summary, this work offers both theoretically sound and practically effective tools for sequential changepoint localization. 

**Abstract (ZH)**: 这篇论文解决了序列变化点分析中一个基本但尚未充分探索的挑战：在检测到变化之后进行推断。我们研究了仅使用在数据依赖性停止时间之前观察到的数据来定位变化点的问题，此时序列检测算法$\mathcal A$宣布检测到变化。首先，我们假设变化前后的分布已知，构建未知变化点的置信集。然后，我们将框架扩展到复合变化前后的场景。我们对观测空间和算法$\mathcal A$没有任何假设——只需能够在模拟数据序列上运行$\mathcal A$即可。总之，这项工作提供了既符合理论又实用有效的工具，用于序列变化点定位。 

---
# Rateless Joint Source-Channel Coding, and a Blueprint for 6G Semantic Communications System Design 

**Title (ZH)**: 无逸联合源信道编码，以及六 generation语义通信系统设计蓝图 

**Authors**: Saeed R. Khosravirad  

**Link**: [PDF](https://arxiv.org/pdf/2502.06095)  

**Abstract**: This paper introduces rateless joint source-channel coding (rateless JSCC). The code is rateless in that it is designed and optimized for a continuum of coding rates such that it achieves a desired distortion for any rate in that continuum. We further introduce rate-adaptive and stable communication link operation to accommodate rateless JSCCs. The link operation resembles a ``bit pipe'' that is identified by its rate in bits per frame, and, by the rate of bits that are flipped in each frame. Thus, the link operation is rate-adaptive such that it punctures the rateless JSCC codeword to adapt its length (and coding rate) to the underlying channel capacity, and is stable in maintaining the bit flipping ratio across time frames.
Next, a new family of autoencoder rateless JSCC codes are introduced. The code family is dubbed RLACS code (read as relax code, standing for ratelss and lossy autoencoder channel and source code). The code is tested for reconstruction loss of image signals and demonstrates powerful performance that is resilient to variation of channel quality. RLACS code is readily applicable to the case of semantic distortion suited to variety of semantic and effectiveness communications use cases.
In the second part of the paper, we dive into the practical concerns around semantic communication and provide a blueprint for semantic networking system design relying on updating the existing network systems with some essential modifications. We further outline a comprehensive list of open research problems and development challenges towards a practical 6G communications system design that enables semantic networking. 

**Abstract (ZH)**: 本文介绍了一种无率联源信道编码（无率联合源信道编码，rateless JSCC）。该编码设计并优化为适用于一系列连续的编码速率，使其能够在该系列中的任何速率下达到所需的失真度。我们进一步介绍了适应速率和稳定的通信链路操作，以适应无率JSCC。链路操作类似于“位管道”，其传输速率由每帧传输的位数和每帧翻转的位数来定义。因此，链路操作是适应速率的，它通过对无率JSCC码字进行刺穿来调整其长度（和编码速率），并保持跨时间帧的位翻转比例的稳定性。
接下来，我们引入了一种新的自编码器无率联源信道编码系列。该编码系列被称为RLACS码（读作relax码，代表无率和失真的自编码器信道和源码）。该编码被测试用于图像信号的重构损失，并展现了对信道质量变化具有强大鲁棒性的性能。RLACS码适用于适应性失真场景，适用于多种语义和有效性通信使用场景。
在论文的第二部分，我们探讨了语义通信的实际关切，并提供了依赖于对现有网络系统进行一些必要修改的语义网络系统设计的蓝图。我们进一步列出了针对实际6G通信系统设计以实现语义网络的开放研究问题和开发挑战。 

---
# Physics-Guided Foundation Model for Scientific Discovery: An Application to Aquatic Science 

**Title (ZH)**: 基于物理的foundation model在科学研究中的应用：以水文科学为例 

**Authors**: Runlong Yu, Chonghao Qiu, Robert Ladwig, Paul Hanson, Yiqun Xie, Xiaowei Jia  

**Link**: [PDF](https://arxiv.org/pdf/2502.06084)  

**Abstract**: Physics-guided machine learning (PGML) has become a prevalent approach in studying scientific systems due to its ability to integrate scientific theories for enhancing machine learning (ML) models. However, most PGML approaches are tailored to isolated and relatively simple tasks, which limits their applicability to complex systems involving multiple interacting processes and numerous influencing features. In this paper, we propose a \textit{\textbf{P}hysics-\textbf{G}uided \textbf{F}oundation \textbf{M}odel (\textbf{PGFM})} that combines pre-trained ML models and physics-based models and leverages their complementary strengths to improve the modeling of multiple coupled processes. To effectively conduct pre-training, we construct a simulated environmental system that encompasses a wide range of influencing features and various simulated variables generated by physics-based models. The model is pre-trained in this system to adaptively select important feature interactions guided by multi-task objectives. We then fine-tune the model for each specific task using true observations, while maintaining consistency with established physical theories, such as the principles of mass and energy conservation. We demonstrate the effectiveness of this methodology in modeling water temperature and dissolved oxygen dynamics in real-world lakes. The proposed PGFM is also broadly applicable to a range of scientific fields where physics-based models are being used. 

**Abstract (ZH)**: 基于物理的预训练模型（Physics-Guided Foundation Model, PGFM）：一种结合预训练机器学习模型和物理模型的方法 

---
# Online Reward-Weighted Fine-Tuning of Flow Matching with Wasserstein Regularization 

**Title (ZH)**: 基于 Wasserstein 正则化的在线奖励加权微调流匹配算法 

**Authors**: Jiajun Fan, Shuaike Shen, Chaoran Cheng, Yuxin Chen, Chumeng Liang, Ge Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.06061)  

**Abstract**: Recent advancements in reinforcement learning (RL) have achieved great success in fine-tuning diffusion-based generative models. However, fine-tuning continuous flow-based generative models to align with arbitrary user-defined reward functions remains challenging, particularly due to issues such as policy collapse from overoptimization and the prohibitively high computational cost of likelihoods in continuous-time flows. In this paper, we propose an easy-to-use and theoretically sound RL fine-tuning method, which we term Online Reward-Weighted Conditional Flow Matching with Wasserstein-2 Regularization (ORW-CFM-W2). Our method integrates RL into the flow matching framework to fine-tune generative models with arbitrary reward functions, without relying on gradients of rewards or filtered datasets. By introducing an online reward-weighting mechanism, our approach guides the model to prioritize high-reward regions in the data manifold. To prevent policy collapse and maintain diversity, we incorporate Wasserstein-2 (W2) distance regularization into our method and derive a tractable upper bound for it in flow matching, effectively balancing exploration and exploitation of policy optimization. We provide theoretical analyses to demonstrate the convergence properties and induced data distributions of our method, establishing connections with traditional RL algorithms featuring Kullback-Leibler (KL) regularization and offering a more comprehensive understanding of the underlying mechanisms and learning behavior of our approach. Extensive experiments on tasks including target image generation, image compression, and text-image alignment demonstrate the effectiveness of our method, where our method achieves optimal policy convergence while allowing controllable trade-offs between reward maximization and diversity preservation. 

**Abstract (ZH)**: Recent advancements in reinforcement learning (RL) have achieved great success in fine-tuning diffusion-based generative models. However, fine-tuning continuous flow-based generative models to align with arbitrary user-defined reward functions remains challenging, particularly due to issues such as policy collapse from overoptimization and the prohibitively high computational cost of likelihoods in continuous-time flows.

在线奖励加权条件流匹配与Wasserstein-2正则化（ORW-CFM-W2）是一种理论上可靠且易于使用的RL微调方法。该方法将RL整合到流匹配框架中，以任意奖励函数微调生成模型，无需依赖奖励的梯度或滤波数据集。通过引入在线奖励加权机制，我们的方法引导模型优先处理数据流形中的高奖励区域。为防止策略塌缩并保持多样性，我们将在方法中引入Wasserstein-2（W2）距离正则化，并在流匹配中推导出其可解上界，有效平衡策略优化的探索与利用。我们提供了理论分析来证明该方法的收敛性质和诱发数据分布，并将其与传统配备Kullback-Leibler（KL）正则化的RL算法建立联系，从而更全面地理解我们方法的内在机制和学习行为。在目标图像生成、图像压缩和图文对齐等任务上的广泛实验展示了该方法的有效性，其中该方法在实现最优策略收敛的同时，允许在奖励最大化和多样性保持之间进行可控的权衡。 

---
# Nearly Optimal Sample Complexity of Offline KL-Regularized Contextual Bandits under Single-Policy Concentrability 

**Title (ZH)**: 近最优样本复杂性：基于单策略集中性的离线KL正则化上下文多臂老虎机 

**Authors**: Qingyue Zhao, Kaixuan Ji, Heyang Zhao, Tong Zhang, Quanquan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2502.06051)  

**Abstract**: KL-regularized policy optimization has become a workhorse in learning-based decision making, while its theoretical understanding is still very limited. Although recent progress has been made towards settling the sample complexity of KL-regularized contextual bandits, existing sample complexity bounds are either $\tilde{O}(\epsilon^{-2})$ under single-policy concentrability or $\tilde{O}(\epsilon^{-1})$ under all-policy concentrability. In this paper, we propose the \emph{first} algorithm with $\tilde{O}(\epsilon^{-1})$ sample complexity under single-policy concentrability for offline contextual bandits. Our algorithm is designed for general function approximation and based on the principle of \emph{pessimism in the face of uncertainty}. The core of our proof leverages the strong convexity of the KL regularization, and the conditional non-negativity of the gap between the true reward and its pessimistic estimator to refine a mean-value-type risk upper bound to its extreme. This in turn leads to a novel covariance-based analysis, effectively bypassing the need for uniform control over the discrepancy between any two functions in the function class. The near-optimality of our algorithm is demonstrated by an $\tilde{\Omega}(\epsilon^{-1})$ lower bound. Furthermore, we extend our algorithm to contextual dueling bandits and achieve a similar nearly optimal sample complexity. 

**Abstract (ZH)**: KL-正则化策略优化已成为基于学习的决策制定的核心方法，但其理论理解依然非常有限。尽管在解决KL-正则化上下文多臂老虎机的样本复杂性方面取得了一些进展，现有的样本复杂性上界在单一策略可集中情况下为$\tilde{O}(\epsilon^{-2})$，在所有策略可集中情况下为$\tilde{O}(\epsilon^{-1})$。在本文中，我们提出了第一个在单一策略可集中情况下样本复杂性为$\tilde{O}(\epsilon^{-1})$的算法，应用于离线上下文多臂老虎机。该算法适用于通用函数近似，并基于“在不确定性面前悲观”的原则。我们证明的核心利用了KL正则化的强凸性和真奖励与其悲观估计之间的条件非负差值来细化一种均值类型的风险上界，从而达到一种新型协方差分析，有效地绕过了在函数类中任何两个函数之间差异的统一控制需求。我们算法的接近最优性由$\tilde{\Omega}(\epsilon^{-1})$的下界得到证明。此外，我们将该算法扩展到上下文对决多臂老虎机，实现了相似的接近最优样本复杂性。 

---
# Provably Overwhelming Transformer Models with Designed Inputs 

**Title (ZH)**: 证明性压倒性Transformer模型通过设计输入实现 

**Authors**: Lev Stambler, Seyed Sajjad Nezhadi, Matthew Coudron  

**Link**: [PDF](https://arxiv.org/pdf/2502.06038)  

**Abstract**: We develop an algorithm which, given a trained transformer model $\mathcal{M}$ as input, as well as a string of tokens $s$ of length $n_{fix}$ and an integer $n_{free}$, can generate a mathematical proof that $\mathcal{M}$ is ``overwhelmed'' by $s$, in time and space $\widetilde{O}(n_{fix}^2 + n_{free}^3)$. We say that $\mathcal{M}$ is ``overwhelmed'' by $s$ when the output of the model evaluated on this string plus any additional string $t$, $\mathcal{M}(s + t)$, is completely insensitive to the value of the string $t$ whenever length($t$) $\leq n_{free}$. Along the way, we prove a particularly strong worst-case form of ``over-squashing'', which we use to bound the model's behavior. Our technique uses computer-aided proofs to establish this type of operationally relevant guarantee about transformer models. We empirically test our algorithm on a single layer transformer complete with an attention head, layer-norm, MLP/ReLU layers, and RoPE positional encoding. We believe that this work is a stepping stone towards the difficult task of obtaining useful guarantees for trained transformer models. 

**Abstract (ZH)**: 我们开发了一个算法，在给定一个训练好的变换器模型$\mathcal{M}$、一个长度为$n_{fix}$的令牌字符串$s$以及一个整数$n_{free}$作为输入的情况下，可以在近$\widetilde{O}(n_{fix}^2 + n_{free}^3)$的时间和空间内生成证明，表明模型$\mathcal{M}$对于字符串$s$是“被压制”的。当我们说模型$\mathcal{M}$对于字符串$s$是“被压制”的时，意味着在输入字符串$s$加上任意附加字符串$t$后的模型输出$\mathcal{M}(s + t)$，当$t$的长度$\leq n_{free}$时，完全不受字符串$t$值的影响。在证明过程中，我们还证明了一种特别强的最坏情况形式的“过度压制”，并利用其来限制模型的行为。我们的方法通过计算机辅助证明，提供了关于变换器模型的一种操作上相关的保证。我们通过实证测试了该算法在包含一个注意力头、层规范化、MLP/ReLU层以及RoPE位置编码的一层变换器上。我们认为这项工作是朝着为训练好的变换器模型获得有用保证的目标迈出的一步。 

---
# Kolmogorov-Arnold Fourier Networks 

**Title (ZH)**: 柯莫戈罗夫-阿诺尔德傅里叶网络 

**Authors**: Jusheng Zhang, Yijia Fan, Kaitong Cai, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.06018)  

**Abstract**: Although Kolmogorov-Arnold based interpretable networks (KAN) have strong theoretical expressiveness, they face significant parameter explosion and high-frequency feature capture challenges in high-dimensional tasks. To address this issue, we propose the Kolmogorov-Arnold-Fourier Network (KAF), which effectively integrates trainable Random Fourier Features (RFF) and a novel hybrid GELU-Fourier activation mechanism to balance parameter efficiency and spectral representation capabilities. Our key technical contributions include: (1) merging KAN's dual-matrix structure through matrix association properties to substantially reduce parameters; (2) introducing learnable RFF initialization strategies to eliminate spectral distortion in high-dimensional approximation tasks; (3) implementing an adaptive hybrid activation function that progressively enhances frequency representation during the training process. Comprehensive experiments demonstrate the superiority of our KAF across various domains including vision, NLP, audio processing, and differential equation-solving tasks, effectively combining theoretical interpretability with practical utility and computational efficiency. 

**Abstract (ZH)**: Kolmogorov-Arnold-Fourier 网络（KAF）：有效融合可训练的随机傅里叶特征及新型混合 GELU-傅里叶激活机制以实现高效参数表示与频谱表示能力的平衡 

---
# Pencils to Pixels: A Systematic Study of Creative Drawings across Children, Adults and AI 

**Title (ZH)**: 从铅笔到像素：关于儿童、成人和AI创造性绘画的系统研究 

**Authors**: Surabhi S Nath, Guiomar del Cuvillo y Schröder, Claire E. Stevenson  

**Link**: [PDF](https://arxiv.org/pdf/2502.05999)  

**Abstract**: Can we derive computational metrics to quantify visual creativity in drawings across intelligent agents, while accounting for inherent differences in technical skill and style? To answer this, we curate a novel dataset consisting of 1338 drawings by children, adults and AI on a creative drawing task. We characterize two aspects of the drawings -- (1) style and (2) content. For style, we define measures of ink density, ink distribution and number of elements. For content, we use expert-annotated categories to study conceptual diversity, and image and text embeddings to compute distance measures. We compare the style, content and creativity of children, adults and AI drawings and build simple models to predict expert and automated creativity scores. We find significant differences in style and content in the groups -- children's drawings had more components, AI drawings had greater ink density, and adult drawings revealed maximum conceptual diversity. Notably, we highlight a misalignment between creativity judgments obtained through expert and automated ratings and discuss its implications. Through these efforts, our work provides, to the best of our knowledge, the first framework for studying human and artificial creativity beyond the textual modality, and attempts to arrive at the domain-agnostic principles underlying creativity. Our data and scripts are available on GitHub. 

**Abstract (ZH)**: 我们能否推导出计算指标来衡量绘画中智能代理的视觉创造力，同时考虑到技术水平和风格的固有差异？为此，我们编纂了一个由1338幅儿童、成人和AI完成的创意绘画组成的新数据集。我们对绘画进行了两个方面的特性描述：（1）风格和（2）内容。在风格方面，我们定义了墨迹密度、墨迹分布和元素数量的指标。在内容方面，我们使用专家标注的类别来研究概念多样性，并使用图像和文本嵌入来计算距离度量。我们比较了儿童、成人和AI绘画的风格、内容和创造力，并构建了简单的模型来预测专家和自动评级的创造力评分。我们发现各组在风格和内容上存在显著差异——儿童绘画的组成部分更多，AI绘画的墨迹密度更大，而成人绘画展示了最大的概念多样性。值得注意的是，我们指出了基于专家评级和自动化评级所得创造力判断之间的不一致，并讨论了其影响。通过这些努力，本工作据我们所知，提供了首个超越文本模态的人工与人工创造力研究框架，并尝试提炼出适用于各个领域的创造力基本原则。我们的数据和脚本可在GitHub上获取。 

---
# Speech to Speech Translation with Translatotron: A State of the Art Review 

**Title (ZH)**: 基于Translatotron的端到端语音转语音翻译：一项前沿综述 

**Authors**: Jules R. Kala, Emmanuel Adetiba, Abdultaofeek Abayom, Oluwatobi E. Dare, Ayodele H. Ifijeh  

**Link**: [PDF](https://arxiv.org/pdf/2502.05980)  

**Abstract**: A cascade-based speech-to-speech translation has been considered a benchmark for a very long time, but it is plagued by many issues, like the time taken to translate a speech from one language to another and compound errors. These issues are because a cascade-based method uses a combination of methods such as speech recognition, speech-to-text translation, and finally, text-to-speech translation. Translatotron, a sequence-to-sequence direct speech-to-speech translation model was designed by Google to address the issues of compound errors associated with cascade model. Today there are 3 versions of the Translatotron model: Translatotron 1, Translatotron 2, and Translatotron3. The first version was designed as a proof of concept to show that a direct speech-to-speech translation was possible, it was found to be less effective than the cascade model but was producing promising results. Translatotron2 was an improved version of Translatotron 1 with results similar to the cascade model. Translatotron 3 the latest version of the model is better than the cascade model at some points. In this paper, a complete review of speech-to-speech translation will be presented, with a particular focus on all the versions of Translatotron models. We will also show that Translatotron is the best model to bridge the language gap between African Languages and other well-formalized languages. 

**Abstract (ZH)**: 基于级联的语音到语音翻译一直被视为一个基准，但存在许多问题，如从一种语言翻译到另一种语言所需时间较长以及复合错误。这些问题是因为级联方法结合了语音识别、语音到文本翻译和最终的文本到语音翻译等多种方法。谷歌设计的Translatotron是一种序列到序列的直接语音到语音翻译模型，旨在解决级联模型关联的复合错误问题。目前，Translatotron模型有三个版本：Translatotron 1、Translatotron 2 和 Translatotron 3。第一版旨在证明直接语音到语音翻译的可能性，发现其效果不如级联模型，但表现出潜在结果。Translatotron 2 是Translatotron 1的改进版，结果与级联模型相当。Translatotron 3 是该模型的最新版本，在某些方面优于级联模型。本文将对语音到语音翻译进行全面回顾，特别关注Translatotron的所有版本。同时，我们将展示Translatotron是最适合弥合非洲语言与其他形式化语言之间语言差距的模型。 

---
# Survival Concept-Based Learning Models 

**Title (ZH)**: 基于生存概念的学习模型 

**Authors**: Stanislav R. Kirpichenko, Lev V. Utkin, Andrei V. Konstantinov, Natalya M. Verbova  

**Link**: [PDF](https://arxiv.org/pdf/2502.05950)  

**Abstract**: Concept-based learning enhances prediction accuracy and interpretability by leveraging high-level, human-understandable concepts. However, existing CBL frameworks do not address survival analysis tasks, which involve predicting event times in the presence of censored data -- a common scenario in fields like medicine and reliability analysis. To bridge this gap, we propose two novel models: SurvCBM (Survival Concept-based Bottleneck Model) and SurvRCM (Survival Regularized Concept-based Model), which integrate concept-based learning with survival analysis to handle censored event time data. The models employ the Cox proportional hazards model and the Beran estimator. SurvCBM is based on the architecture of the well-known concept bottleneck model, offering interpretable predictions through concept-based explanations. SurvRCM uses concepts as regularization to enhance accuracy. Both models are trained end-to-end and provide interpretable predictions in terms of concepts. Two interpretability approaches are proposed: one leveraging the linear relationship in the Cox model and another using an instance-based explanation framework with the Beran estimator. Numerical experiments demonstrate that SurvCBM outperforms SurvRCM and traditional survival models, underscoring the importance and advantages of incorporating concept information. The code for the proposed algorithms is publicly available. 

**Abstract (ZH)**: 基于概念的学习增强预测准确性和可解释性：通过利用高层次的人类可理解的概念。然而，现有的基于概念的学习（CBL）框架未解决涉及截尾数据的事件时间预测任务——这是医学和可靠性分析等领域中的常见场景。为弥合这一缺口，我们提出了两个新型模型：生存概念瓶颈模型（SurvCBM）和生存正则化概念模型（SurvRCM），以结合概念学习与生存分析处理截尾事件时间数据。该模型采用Cox比例风险模型和Beran估计量。SurvCBM基于著名概念瓶颈模型的架构，通过概念解释提供可解释的预测。SurvRCM使用概念作为正则化以提高准确性。两种模型都端到端训练，并以概念形式提供可解释的预测。提出了两种可解释性方法：一种利用Cox模型中的线性关系，另一种使用以Beran估计量为基础的实例解释框架。数值实验表明，SurvCBM优于SurvRCM和传统生存模型，突显了结合概念信息的重要性与优势。提出的算法的代码已公开。 

---
# Verifying Proportionality in Temporal Voting 

**Title (ZH)**: 验证时间投票中的比例性 

**Authors**: Edith Elkind, Svetlana Obraztsova, Jannik Peters, Nicholas Teh  

**Link**: [PDF](https://arxiv.org/pdf/2502.05949)  

**Abstract**: We study a model of temporal voting where there is a fixed time horizon, and at each round the voters report their preferences over the available candidates and a single candidate is selected. Prior work has adapted popular notions of justified representation as well as voting rules that provide strong representation guarantees from the multiwinner election setting to this model. In our work, we focus on the complexity of verifying whether a given outcome offers proportional representation. We show that in the temporal setting verification is strictly harder than in multiwinner voting, but identify natural special cases that enable efficient algorithms. 

**Abstract (ZH)**: 我们研究了一个固定时间 horizons 的投票模型，在每一轮中选民报告他们对可用候选人的偏好，并选择一名候选人。先前的工作将多席位选举中流行的正当代表概念及其提供强大代表保证的投票规则适应到这个模型中。在我们的工作中，我们关注验证给定结果是否提供比例代表的复杂性。我们展示了在时间序列设置中验证比多席位投票更难，但识别了一些自然的特殊情况以使算法有效。 

---
# Skill Expansion and Composition in Parameter Space 

**Title (ZH)**: 参数空间中的技能扩展与组成 

**Authors**: Tenglong Liu, Jianxiong Li, Yinan Zheng, Haoyi Niu, Yixing Lan, Xin Xu, Xianyuan Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2502.05932)  

**Abstract**: Humans excel at reusing prior knowledge to address new challenges and developing skills while solving problems. This paradigm becomes increasingly popular in the development of autonomous agents, as it develops systems that can self-evolve in response to new challenges like human beings. However, previous methods suffer from limited training efficiency when expanding new skills and fail to fully leverage prior knowledge to facilitate new task learning. In this paper, we propose Parametric Skill Expansion and Composition (PSEC), a new framework designed to iteratively evolve the agents' capabilities and efficiently address new challenges by maintaining a manageable skill library. This library can progressively integrate skill primitives as plug-and-play Low-Rank Adaptation (LoRA) modules in parameter-efficient finetuning, facilitating efficient and flexible skill expansion. This structure also enables the direct skill compositions in parameter space by merging LoRA modules that encode different skills, leveraging shared information across skills to effectively program new skills. Based on this, we propose a context-aware module to dynamically activate different skills to collaboratively handle new tasks. Empowering diverse applications including multi-objective composition, dynamics shift, and continual policy shift, the results on D4RL, DSRL benchmarks, and the DeepMind Control Suite show that PSEC exhibits superior capacity to leverage prior knowledge to efficiently tackle new challenges, as well as expand its skill libraries to evolve the capabilities. Project website: this https URL. 

**Abstract (ZH)**: 人类擅长利用先验知识应对新挑战并在解决问题过程中发展技能。这一范式在自主代理系统的发展中变得越来越流行，因为它能够使系统根据新的挑战自我进化，类似于人类的行为。然而，先前的方法在扩展新技能时训练效率有限，并且未能充分利用先验知识来促进新任务的学习。在本文中，我们提出了一种新的框架——参数化技能扩展与组合（PSEC），旨在通过维护一个可管理的技能库，逐步进化代理的能力，并有效解决新挑战。该库可以通过参数高效的微调逐步整合技能原语作为即插即用的低秩适应（LoRA）模块，从而促进高效的技能扩展。这种结构还能够在参数空间直接组合技能，通过合并表示不同技能的LoRA模块来利用技能之间的共享信息，有效编程新技能。在此基础上，我们提出了一种上下文感知模块，以动态激活不同的技能来合作处理新任务。PSEC在D4RL、DSRL基准以及DeepMind Control Suite上的结果表明，它能够在有效利用先验知识应对新挑战的同时，扩展其技能库以进化能力。项目网站: 这个 https URL。 

---
# Protecting Intellectual Property of EEG-based Neural Networks with Watermarking 

**Title (ZH)**: 基于EEG的神经网络知识产权保护方法中的水印技术 

**Authors**: Ahmed Abdelaziz, Ahmed Fathi, Ahmed Fares  

**Link**: [PDF](https://arxiv.org/pdf/2502.05931)  

**Abstract**: EEG-based neural networks, pivotal in medical diagnosis and brain-computer interfaces, face significant intellectual property (IP) risks due to their reliance on sensitive neurophysiological data and resource-intensive development. Current watermarking methods, particularly those using abstract trigger sets, lack robust authentication and fail to address the unique challenges of EEG models. This paper introduces a cryptographic wonder filter-based watermarking framework tailored for EEG-based neural networks. Leveraging collision-resistant hashing and public-key encryption, the wonder filter embeds the watermark during training, ensuring minimal distortion ($\leq 5\%$ drop in EEG task accuracy) and high reliability (100\% watermark detection). The framework is rigorously evaluated against adversarial attacks, including fine-tuning, transfer learning, and neuron pruning. Results demonstrate persistent watermark retention, with classification accuracy for watermarked states remaining above 90\% even after aggressive pruning, while primary task performance degrades faster, deterring removal attempts. Piracy resistance is validated by the inability to embed secondary watermarks without severe accuracy loss ( $>10\%$ in EEGNet and CCNN models). Cryptographic hashing ensures authentication, reducing brute-force attack success probabilities. Evaluated on the DEAP dataset across models (CCNN, EEGNet, TSception), the method achieves $>99.4\%$ null-embedding accuracy, effectively eliminating false positives. By integrating wonder filters with EEG-specific adaptations, this work bridges a critical gap in IP protection for neurophysiological models, offering a secure, tamper-proof solution for healthcare and biometric applications. The framework's robustness against adversarial modifications underscores its potential to safeguard sensitive EEG models while maintaining diagnostic utility. 

**Abstract (ZH)**: 基于EEG的神经网络的加密奇偶校验滤波器嵌入水印框架：应对知识产权风险 

---
# Sign-Symmetry Learning Rules are Robust Fine-Tuners 

**Title (ZH)**: 签名对称学习规则是稳健的微调器 

**Authors**: Aymene Berriche, Mehdi Zakaria Adjal, Riyadh Baghdadi  

**Link**: [PDF](https://arxiv.org/pdf/2502.05925)  

**Abstract**: Backpropagation (BP) has long been the predominant method for training neural networks due to its effectiveness. However, numerous alternative approaches, broadly categorized under feedback alignment, have been proposed, many of which are motivated by the search for biologically plausible learning mechanisms. Despite their theoretical appeal, these methods have consistently underperformed compared to BP, leading to a decline in research interest. In this work, we revisit the role of such methods and explore how they can be integrated into standard neural network training pipelines. Specifically, we propose fine-tuning BP-pre-trained models using Sign-Symmetry learning rules and demonstrate that this approach not only maintains performance parity with BP but also enhances robustness. Through extensive experiments across multiple tasks and benchmarks, we establish the validity of our approach. Our findings introduce a novel perspective on neural network training and open new research directions for leveraging biologically inspired learning rules in deep learning. 

**Abstract (ZH)**: 反馈调整方法在神经网络训练中的角色重探及其与BP的集成研究 

---
# NeuralPrefix: A Zero-shot Sensory Data Imputation Plugin 

**Title (ZH)**: NeuralPrefix: 一种零样本感官数据插值插件 

**Authors**: Abdelwahed Khamis, Sara Khalifa  

**Link**: [PDF](https://arxiv.org/pdf/2502.05883)  

**Abstract**: Real-world sensing challenges such as sensor failures, communication issues, and power constraints lead to data intermittency. An issue that is known to undermine the traditional classification task that assumes a continuous data stream. Previous works addressed this issue by designing bespoke solutions (i.e. task-specific and/or modality-specific imputation). These approaches, while effective for their intended purposes, had limitations in their applicability across different tasks and sensor modalities. This raises an important question: Can we build a task-agnostic imputation pipeline that is transferable to new sensors without requiring additional training? In this work, we formalise the concept of zero-shot imputation and propose a novel approach that enables the adaptation of pre-trained models to handle data intermittency. This framework, named NeuralPrefix, is a generative neural component that precedes a task model during inference, filling in gaps caused by data intermittency. NeuralPrefix is built as a continuous dynamical system, where its internal state can be estimated at any point in time by solving an Ordinary Differential Equation (ODE). This approach allows for a more versatile and adaptable imputation method, overcoming the limitations of task-specific and modality-specific solutions. We conduct a comprehensive evaluation of NeuralPrefix on multiple sensory datasets, demonstrating its effectiveness across various domains. When tested on intermittent data with a high 50% missing data rate, NeuralPreifx accurately recovers all the missing samples, achieving SSIM score between 0.93-0.96. Zero-shot evaluations show that NeuralPrefix generalises well to unseen datasets, even when the measurements come from a different modality. 

**Abstract (ZH)**: 无监督填补挑战：NeuralPrefix框架在新传感器上的可迁移性研究 

---
# Uni-Retrieval: A Multi-Style Retrieval Framework for STEM's Education 

**Title (ZH)**: Uni-Retrieval：面向STEM教育的多风格检索框架 

**Authors**: Yanhao Jia, Xinyi Wu, Hao Li, Qinglin Zhang, Yuxiao Hu, Shuai Zhao, Wenqi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2502.05863)  

**Abstract**: In AI-facilitated teaching, leveraging various query styles to interpret abstract text descriptions is crucial for ensuring high-quality teaching. However, current retrieval models primarily focus on natural text-image retrieval, making them insufficiently tailored to educational scenarios due to the ambiguities in the retrieval process. In this paper, we propose a diverse expression retrieval task tailored to educational scenarios, supporting retrieval based on multiple query styles and expressions. We introduce the STEM Education Retrieval Dataset (SER), which contains over 24,000 query pairs of different styles, and the Uni-Retrieval, an efficient and style-diversified retrieval vision-language model based on prompt tuning. Uni-Retrieval extracts query style features as prototypes and builds a continuously updated Prompt Bank containing prompt tokens for diverse queries. This bank can updated during test time to represent domain-specific knowledge for different subject retrieval scenarios. Our framework demonstrates scalability and robustness by dynamically retrieving prompt tokens based on prototype similarity, effectively facilitating learning for unknown queries. Experimental results indicate that Uni-Retrieval outperforms existing retrieval models in most retrieval tasks. This advancement provides a scalable and precise solution for diverse educational needs. 

**Abstract (ZH)**: 在AI辅助教学中，利用多种查询风格解释抽象的文字描述对于确保高质量的教学至关重要。然而，当前的检索模型主要集中在自然文本-图像检索上，因检索过程中的歧义性，使其未能充分适应教育场景的需求。本文提出了一种针对教育场景的多样化表达检索任务，支持基于多种查询风格和表达的检索。我们引入了STEM教育检索数据集（SER），包含超过24,000个不同风格的查询对，并提出了一种基于提示调优的高效且风格多样化检索视觉语言模型Uni-Retrieval。Uni-Retrieval提取查询风格特征作为原型，并构建一个不断更新的提示库，包含用于各种查询的提示标记。该库可在测试时更新，以代表不同学科检索场景的领域特定知识。我们的框架通过根据原型相似性动态检索提示标记，展示了可扩展性和鲁棒性，有效促进了未知查询的学习。实验结果表明，Uni-Retrieval在大多数检索任务中优于现有检索模型。这一进展为满足多样化的教育需求提供了可扩展且精确的解决方案。 

---
# Contrastive Representation Distillation via Multi-Scale Feature Decoupling 

**Title (ZH)**: 多尺度特征解耦的对比表示蒸馏 

**Authors**: Cuipeng Wang, Tieyuan Chen, Haipeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.05835)  

**Abstract**: Knowledge distillation is a technique aimed at enhancing the performance of a smaller student network without increasing its parameter size by transferring knowledge from a larger, pre-trained teacher network. Previous approaches have predominantly focused on distilling global feature information while overlooking the importance of disentangling the diverse types of information embedded within different regions of the feature. In this work, we introduce multi-scale decoupling in the feature transfer process for the first time, where the decoupled local features are individually processed and integrated with contrastive learning. Moreover, compared to previous contrastive learning-based distillation methods, our approach not only reduces computational costs but also enhances efficiency, enabling performance improvements for the student network using only single-batch samples. Extensive evaluations on CIFAR-100 and ImageNet demonstrate our method's superiority, with some student networks distilled using our method even surpassing the performance of their pre-trained teacher networks. These results underscore the effectiveness of our approach in enabling student networks to thoroughly absorb knowledge from teacher networks. 

**Abstract (ZH)**: 多尺度解耦特征转移与对比学习的知识蒸馏 

---
# Compressing Model with Few Class-Imbalance Samples: An Out-of-Distribution Expedition 

**Title (ZH)**: 基于少量不平衡类样本的模型压缩：一种异常分布探索 

**Authors**: Tian-Shuang Wu, Shen-Huan Lyu, Ning Chen, Zhihao Qu, Baoliu Ye  

**Link**: [PDF](https://arxiv.org/pdf/2502.05832)  

**Abstract**: In recent years, as a compromise between privacy and performance, few-sample model compression has been widely adopted to deal with limited data resulting from privacy and security concerns. However, when the number of available samples is extremely limited, class imbalance becomes a common and tricky problem. Achieving an equal number of samples across all classes is often costly and impractical in real-world applications, and previous studies on few-sample model compression have mostly ignored this significant issue. Our experiments comprehensively demonstrate that class imbalance negatively affects the overall performance of few-sample model compression methods. To address this problem, we propose a novel and adaptive framework named OOD-Enhanced Few-Sample Model Compression (OE-FSMC). This framework integrates easily accessible out-of-distribution (OOD) data into both the compression and fine-tuning processes, effectively rebalancing the training distribution. We also incorporate a joint distillation loss and a regularization term to reduce the risk of the model overfitting to the OOD data. Extensive experiments on multiple benchmark datasets show that our framework can be seamlessly incorporated into existing few-sample model compression methods, effectively mitigating the accuracy degradation caused by class imbalance. 

**Abstract (ZH)**: 面向异常数据增强的小样本模型压缩（OOD-增强的小样本模型压缩） 

---
# HyGEN: Regularizing Negative Hyperedge Generation for Accurate Hyperedge Prediction 

**Title (ZH)**: HyGEN: 正则化负超边生成以实现准确的超边预测 

**Authors**: Song Kyung Yu, Da Eun Lee, Yunyong Ko, Sang-Wook Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.05827)  

**Abstract**: Hyperedge prediction is a fundamental task to predict future high-order relations based on the observed network structure. Existing hyperedge prediction methods, however, suffer from the data sparsity problem. To alleviate this problem, negative sampling methods can be used, which leverage non-existing hyperedges as contrastive information for model training. However, the following important challenges have been rarely studied: (C1) lack of guidance for generating negatives and (C2) possibility of producing false negatives. To address them, we propose a novel hyperedge prediction method, HyGEN, that employs (1) a negative hyperedge generator that employs positive hyperedges as a guidance to generate more realistic ones and (2) a regularization term that prevents the generated hyperedges from being false negatives. Extensive experiments on six real-world hypergraphs reveal that HyGEN consistently outperforms four state-of-the-art hyperedge prediction methods. 

**Abstract (ZH)**: 高阶边预测是基于观测网络结构预测未来高阶关系的基本任务。现有的高阶边预测方法面临数据稀疏问题。为缓解这一问题，可以使用负采样方法，利用不存在的高阶边作为模型训练的对比信息。然而，生成负样本的指导不足以及产生虚假负样本的可能性等问题鲜有研究。为解决这些问题，我们提出了一种新颖的高阶边预测方法HyGEN，该方法采用（1）一种高阶边生成器，利用正高阶边作为指导生成更现实的高阶边；（2）一种正则化项，防止生成的高阶边成为虚假负样本。在六个真实世界的超图上的 extensive 实验表明，HyGEN 一贯优于四种最新的高阶边预测方法。 

---
# MindCraft: Revolutionizing Education through AI-Powered Personalized Learning and Mentorship for Rural India 

**Title (ZH)**: MindCraft：通过AI驱动的个性化学习和导师制 revolutionizing 教育以惠及印度农村地区 

**Authors**: Arihant Bardia, Aayush Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2502.05826)  

**Abstract**: MindCraft is a modern platform designed to revolutionize education in rural India by leveraging Artificial Intelligence (AI) to create personalized learning experiences, provide mentorship, and foster resource-sharing. In a country where access to quality education is deeply influenced by geography and socio economic status, rural students often face significant barriers in their educational journeys. MindCraft aims to bridge this gap by utilizing AI to create tailored learning paths, connect students with mentors, and enable a collaborative network of educational resources that transcends both physical and digital divides. This paper explores the challenges faced by rural students, the transformative potential of AI, and how MindCraft offers a scalable, sustainable solution for equitable education system. By focusing on inclusivity, personalized learning, and mentorship, MindCraft seeks to empower rural students, equipping them with the skills, knowledge, and opportunities needed to thrive in an increasingly digital world. Ultimately, MindCraft envisions a future in which technology not only bridges educational gaps but also becomes the driving force for a more inclusive and empowered society. 

**Abstract (ZH)**: MindCraft是依托人工智能（AI）创造个性化学习体验、提供导师指导并促进资源分享的现代平台，旨在重塑印度农村的教育。在教育质量受地理和社会经济状况深深影响的国家，农村学生在教育旅程中常常面临巨大障碍。MindCraft希望通过利用AI创造个性化学习路径、连接学生与导师，并建立跨越物理与数字鸿沟的协作性教育资源网络来弥合这一差距。本文探讨了农村学生面临的挑战、AI的变革潜力，以及MindCraft提供的可扩展、可持续的公平教育解决方案。通过注重包容性、个性化学习和导师指导，MindCraft旨在赋能农村学生，为其提供在日益数字化的世界中所需的知识、技能和机会。最终，MindCraft构想了一个未来，在这个未来中，技术不仅弥合了教育鸿沟，还成为推动更包容和赋权社会的力量。 

---
# WatchGuardian: Enabling User-Defined Personalized Just-in-Time Intervention on Smartwatch 

**Title (ZH)**: WatchGuardian: 允许用户定义个性化即时干预的智能手环系统 

**Authors**: Ying Lei, Yancheng Cao, Will Wang, Yuanzhe Dong, Changchang Yin, Weidan Cao, Ping Zhang, Jingzhen Yang, Bingsheng Yao, Yifan Peng, Chunhua Weng, Randy Auerbach, Lena Mamykina, Dakuo Wang, Yuntao Wang, Xuhai Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.05783)  

**Abstract**: While just-in-time interventions (JITIs) have effectively targeted common health behaviors, individuals often have unique needs to intervene in personal undesirable actions that can negatively affect physical, mental, and social well-being. We present WatchGuardian, a smartwatch-based JITI system that empowers users to define custom interventions for these personal actions with a small number of samples. For the model to detect new actions based on limited new data samples, we developed a few-shot learning pipeline that finetuned a pre-trained inertial measurement unit (IMU) model on public hand-gesture datasets. We then designed a data augmentation and synthesis process to train additional classification layers for customization. Our offline evaluation with 26 participants showed that with three, five, and ten examples, our approach achieved an average accuracy of 76.8%, 84.7%, and 87.7%, and an F1 score of 74.8%, 84.2%, and 87.2% We then conducted a four-hour intervention study to compare WatchGuardian against a rule-based intervention. Our results demonstrated that our system led to a significant reduction by 64.0 +- 22.6% in undesirable actions, substantially outperforming the baseline by 29.0%. Our findings underscore the effectiveness of a customizable, AI-driven JITI system for individuals in need of behavioral intervention in personal undesirable actions. We envision that our work can inspire broader applications of user-defined personalized intervention with advanced AI solutions. 

**Abstract (ZH)**: 基于智能手环的个性化即时干预系统：WatchGuardian及其应用研究 

---
# Predictive Crash Analytics for Traffic Safety using Deep Learning 

**Title (ZH)**: 基于深度学习的交通安全性事故预测分析 

**Authors**: Karthik Sivakoti  

**Link**: [PDF](https://arxiv.org/pdf/2502.05777)  

**Abstract**: Traditional automated crash analysis systems heavily rely on static statistical models and historical data, requiring significant manual interpretation and lacking real-time predictive capabilities. This research presents an innovative approach to traffic safety analysis through the integration of ensemble learning methods and multi-modal data fusion for real-time crash risk assessment and prediction. Our primary contribution lies in developing a hierarchical severity classification system that combines spatial-temporal crash patterns with environmental conditions, achieving significant improvements over traditional statistical approaches. The system demonstrates a Mean Average Precision (mAP) of 0.893, representing a 15% improvement over current state-of-the-art methods (baseline mAP: 0.776). We introduce a novel feature engineering technique that integrates crash location data with incident reports and weather conditions, achieving 92.4% accuracy in risk prediction and 89.7% precision in hotspot identification. Through extensive validation using 500,000 initial crash records filtered to 59,496 high-quality samples, our solution shows marked improvements in both prediction accuracy and computational efficiency. Key innovations include a robust data cleaning pipeline, adaptive feature generation, and a scalable real-time prediction system capable of handling peak loads of 1,000 concurrent requests while maintaining sub-100ms response times. 

**Abstract (ZH)**: 基于集成学习方法和多模态数据融合的实时 crash 风险评估与预测研究 

---
# Rethinking Link Prediction for Directed Graphs 

**Title (ZH)**: 重思有向图中的链接预测 

**Authors**: Mingguo He, Yuhe Guo, Yanping Zheng, Zhewei Wei, Stephan Günnemann, Xiaokui Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2502.05724)  

**Abstract**: Link prediction for directed graphs is a crucial task with diverse real-world applications. Recent advances in embedding methods and Graph Neural Networks (GNNs) have shown promising improvements. However, these methods often lack a thorough analysis of embedding expressiveness and suffer from ineffective benchmarks for a fair evaluation. In this paper, we propose a unified framework to assess the expressiveness of existing methods, highlighting the impact of dual embeddings and decoder design on performance. To address limitations in current experimental setups, we introduce DirLinkBench, a robust new benchmark with comprehensive coverage and standardized evaluation. The results show that current methods struggle to achieve strong performance on the new benchmark, while DiGAE outperforms others overall. We further revisit DiGAE theoretically, showing its graph convolution aligns with GCN on an undirected bipartite graph. Inspired by these insights, we propose a novel spectral directed graph auto-encoder SDGAE that achieves SOTA results on DirLinkBench. Finally, we analyze key factors influencing directed link prediction and highlight open challenges. 

**Abstract (ZH)**: 有向图链接预测是一个具有多种实际应用的关键任务。最近的嵌入方法和图神经网络（GNNs）的进步展现了有希望的改进。然而，这些方法常常缺乏对嵌入表征能力的全面分析，并且缺乏有效的基准测试来实现公平的评估。本文提出了一种统一框架来评估现有方法的表征能力，并强调了双嵌入和解码器设计对性能的影响。为了应对当前实验设置的局限性，我们引入了DirLinkBench这一新的稳健基准，该基准具有全面覆盖和标准化评估。结果表明，当前方法在新基准上难以实现 strong 表现，而 DiGAE 整体上表现出色。我们进一步从理论上重新审视了 DiGAE，证明其图卷积在无向二分图上与 GCN 一致。受这些见解的启发，我们提出了一种新颖的谱有向图自编码器 SDGAE，该模型在 DirLinkBench 上取得了 SOTA 结果。最后，我们分析了影响有向链接预测的关键因素，并指出了开放挑战。 

---
# Pareto-Optimality, Smoothness, and Stochasticity in Learning-Augmented One-Max-Search 

**Title (ZH)**: Pareto-最优性、平滑性与随机性在学习增强的一最大化搜索中的应用 

**Authors**: Ziyad Benomar, Lorenzo Croissant, Vianney Perchet, Spyros Angelopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2502.05720)  

**Abstract**: One-max search is a classic problem in online decision-making, in which a trader acts on a sequence of revealed prices and accepts one of them irrevocably to maximise its profit. The problem has been studied both in probabilistic and in worst-case settings, notably through competitive analysis, and more recently in learning-augmented settings in which the trader has access to a prediction on the sequence. However, existing approaches either lack smoothness, or do not achieve optimal worst-case guarantees: they do not attain the best possible trade-off between the consistency and the robustness of the algorithm. We close this gap by presenting the first algorithm that simultaneously achieves both of these important objectives. Furthermore, we show how to leverage the obtained smoothness to provide an analysis of one-max search in stochastic learning-augmented settings which capture randomness in both the observed prices and the prediction. 

**Abstract (ZH)**: One-max 搜索是在线决策中的一个经典问题，在该问题中，交易者对一系列公布的价格做出反应，并不可撤销地接受其中一个以最大化其利润。该问题在概率性和最坏情况设置下均有研究，尤其是通过竞 competitiveness 分析进行研究，并且最近在预测增强的学习环境下的研究中也有涉及，其中交易者可以访问价格序列的预测。然而，现有方法要么缺乏平滑性，要么未能实现最优的最坏情况保证：它们未能在一致性与鲁棒性之间达到最佳权衡。我们通过提出第一个同时实现这两个重要目标的算法来填补这一空白。此外，我们展示了如何利用获得的平滑性来分析随机增强学习环境下的 one-max 搜索，该环境既包含了观测价格中的随机性，也包含了预测中的随机性。 

---
# Extended Histogram-based Outlier Score (EHBOS) 

**Title (ZH)**: 基于扩展直方图的离群点分数(EHBOS) 

**Authors**: Tanvir Islam  

**Link**: [PDF](https://arxiv.org/pdf/2502.05719)  

**Abstract**: Histogram-Based Outlier Score (HBOS) is a widely used outlier or anomaly detection method known for its computational efficiency and simplicity. However, its assumption of feature independence limits its ability to detect anomalies in datasets where interactions between features are critical. In this paper, we propose the Extended Histogram-Based Outlier Score (EHBOS), which enhances HBOS by incorporating two-dimensional histograms to capture dependencies between feature pairs. This extension allows EHBOS to identify contextual and dependency-driven anomalies that HBOS fails to detect. We evaluate EHBOS on 17 benchmark datasets, demonstrating its effectiveness and robustness across diverse anomaly detection scenarios. EHBOS outperforms HBOS on several datasets, particularly those where feature interactions are critical in defining the anomaly structure, achieving notable improvements in ROC AUC. These results highlight that EHBOS can be a valuable extension to HBOS, with the ability to model complex feature dependencies. EHBOS offers a powerful new tool for anomaly detection, particularly in datasets where contextual or relational anomalies play a significant role. 

**Abstract (ZH)**: 基于直方图的扩展异常评分（EHBOS）：考虑特征间依赖性的异常检测方法 

---
# Proving the Coding Interview: A Benchmark for Formally Verified Code Generation 

**Title (ZH)**: 证明的编码面试：正式验证代码生成的基准 

**Authors**: Quinn Dougherty, Ronak Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2502.05714)  

**Abstract**: We introduce the Formally Verified Automated Programming Progress Standards, or FVAPPS, a benchmark of 4715 samples for writing programs and proving their correctness, the largest formal verification benchmark, including 1083 curated and quality controlled samples. Previously, APPS provided a benchmark and dataset for programming puzzles to be completed in Python and checked against unit tests, of the kind seen in technical assessments in the software engineering industry. Building upon recent approaches for benchmarks in interactive theorem proving, we generalize the unit tests to Lean 4 theorems given without proof (i.e., using Lean's "sorry" keyword). On the 406 theorems of 100 randomly selected samples, Sonnet correctly proves 30% and Gemini correctly proves 18%. We challenge the machine learning and program synthesis communities to solve both each general purpose programming problem and its associated correctness specifications. The benchmark is available at this https URL. 

**Abstract (ZH)**: 正式验证自动编程进展标准：FVAPPS及其基准测试 

---
# Rethinking Word Similarity: Semantic Similarity through Classification Confusion 

**Title (ZH)**: 重思词相似性：通过分类混淆实现语义相似性 

**Authors**: Kaitlyn Zhou, Haishan Gao, Sarah Chen, Dan Edelstein, Dan Jurafsky, Chen Shani  

**Link**: [PDF](https://arxiv.org/pdf/2502.05704)  

**Abstract**: Word similarity has many applications to social science and cultural analytics tasks like measuring meaning change over time and making sense of contested terms. Yet traditional similarity methods based on cosine similarity between word embeddings cannot capture the context-dependent, asymmetrical, polysemous nature of semantic similarity. We propose a new measure of similarity, Word Confusion, that reframes semantic similarity in terms of feature-based classification confusion. Word Confusion is inspired by Tversky's suggestion that similarity features be chosen dynamically. Here we train a classifier to map contextual embeddings to word identities and use the classifier confusion (the probability of choosing a confounding word c instead of the correct target word t) as a measure of the similarity of c and t. The set of potential confounding words acts as the chosen features. Our method is comparable to cosine similarity in matching human similarity judgments across several datasets (MEN, WirdSim353, and SimLex), and can measure similarity using predetermined features of interest. We demonstrate our model's ability to make use of dynamic features by applying it to test a hypothesis about changes in the 18th C. meaning of the French word "revolution" from popular to state action during the French Revolution. We hope this reimagining of semantic similarity will inspire the development of new tools that better capture the multi-faceted and dynamic nature of language, advancing the fields of computational social science and cultural analytics and beyond. 

**Abstract (ZH)**: 基于词混淆的语义相似度测量在社会科学研究和文化分析任务中的应用 

---
# Mobile Application Threats and Security 

**Title (ZH)**: 移动应用威胁与安全 

**Authors**: Timur Mirzoev, Mark Miller, Shamimara Lasker, Michael Brannon  

**Link**: [PDF](https://arxiv.org/pdf/2502.05685)  

**Abstract**: The movement to mobile computing solutions provides flexibility to different users whether it is a business user, a student, or even providing entertainment to children and adults of all ages. Due to these emerging technologies mobile users are unable to safeguard private information in a very effective way and cybercrimes are increasing day by day. This manuscript will focus on security vulnerabilities in the mobile computing industry, especially focusing on tablets and smart phones. This study will dive into current security threats for the Android & Apple iOS market, exposing security risks and threats that the novice or average user may not be aware of. The purpose of this study is to analyze current security risks and threats, and provide solutions that may be deployed to protect against such threats. 

**Abstract (ZH)**: 移动计算解决方案的推广为不同的用户提供了灵活性，无论是商业用户、学生，还是为各年龄段的儿童和成人提供娱乐。由于这些新兴技术，移动用户无法有效保护私人信息，网络犯罪日益增多。本文将关注移动计算行业的安全漏洞，特别是针对平板电脑和智能手机。本研究将深入探讨Android和Apple iOS市场的当前安全威胁，揭露新手或普通用户可能 unaware 的安全风险和威胁。本文的研究目的是分析当前的安全风险和威胁，并提供可能部署的解决方案以防止这些威胁。 

---
# Machine Unlearning via Information Theoretic Regularization 

**Title (ZH)**: 基于信息论正则化的机器卸载 

**Authors**: Shizhou Xu, Thomas Strohmer  

**Link**: [PDF](https://arxiv.org/pdf/2502.05684)  

**Abstract**: How can we effectively remove or "unlearn" undesirable information, such as specific features or individual data points, from a learning outcome while minimizing utility loss and ensuring rigorous guarantees? We introduce a mathematical framework based on information-theoretic regularization to address both feature and data point unlearning. For feature unlearning, we derive a unified solution that simultaneously optimizes diverse learning objectives, including entropy, conditional entropy, KL-divergence, and the energy of conditional probability. For data point unlearning, we first propose a novel definition that serves as a practical condition for unlearning via retraining, is easy to verify, and aligns with the principles of differential privacy from an inference perspective. Then, we provide provable guarantees for our framework on data point unlearning. By combining flexibility in learning objectives with simplicity in regularization design, our approach is highly adaptable and practical for a wide range of machine learning and AI applications. 

**Abstract (ZH)**: 如何在最小化实用性损失的同时，确保严格保证地从学习成果中有效地移除或“忘掉”不良信息（如特定特征或个体数据点），并确保数据点忘掉的严谨性？我们提出了一种基于信息论正则化的方法，以解决特征和数据点忘掉的问题。对于特征忘掉，我们推导出一个统一的解决方案，同时优化包括熵、条件熵、KL散度和条件概率的能量在内的多种学习目标。对于数据点忘掉，我们首先提出了一种新的定义，作为通过重新训练实现忘掉的实用条件，易于验证，并从推理视角与差分隐私的原则一致。然后，我们为我们的框架提供了数据点忘掉的可证明保证。通过在学习目标上的灵活性与正则化设计的简单性相结合，我们的方法具有高度的适应性和实用性，适用于广泛的人工智能和机器学习应用。 

---
# On the Convergence and Stability of Upside-Down Reinforcement Learning, Goal-Conditioned Supervised Learning, and Online Decision Transformers 

**Title (ZH)**: upside-down 强化学习、目标导向监督学习和在线决策变换器的收敛性和稳定性分析 

**Authors**: Miroslav Štrupl, Oleg Szehr, Francesco Faccio, Dylan R. Ashley, Rupesh Kumar Srivastava, Jürgen Schmidhuber  

**Link**: [PDF](https://arxiv.org/pdf/2502.05672)  

**Abstract**: This article provides a rigorous analysis of convergence and stability of Episodic Upside-Down Reinforcement Learning, Goal-Conditioned Supervised Learning and Online Decision Transformers. These algorithms performed competitively across various benchmarks, from games to robotic tasks, but their theoretical understanding is limited to specific environmental conditions. This work initiates a theoretical foundation for algorithms that build on the broad paradigm of approaching reinforcement learning through supervised learning or sequence modeling. At the core of this investigation lies the analysis of conditions on the underlying environment, under which the algorithms can identify optimal solutions. We also assess whether emerging solutions remain stable in situations where the environment is subject to tiny levels of noise. Specifically, we study the continuity and asymptotic convergence of command-conditioned policies, values and the goal-reaching objective depending on the transition kernel of the underlying Markov Decision Process. We demonstrate that near-optimal behavior is achieved if the transition kernel is located in a sufficiently small neighborhood of a deterministic kernel. The mentioned quantities are continuous (with respect to a specific topology) at deterministic kernels, both asymptotically and after a finite number of learning cycles. The developed methods allow us to present the first explicit estimates on the convergence and stability of policies and values in terms of the underlying transition kernels. On the theoretical side we introduce a number of new concepts to reinforcement learning, like working in segment spaces, studying continuity in quotient topologies and the application of the fixed-point theory of dynamical systems. The theoretical study is accompanied by a detailed investigation of example environments and numerical experiments. 

**Abstract (ZH)**: 本文提供了对Episodic Upside-Down强化学习、目标条件监督学习和在线决策变换器等算法收敛性和稳定性的严格分析。这些算法在从游戏到机器人任务的各种基准测试中表现出色，但其理论理解仅限于特定的环境条件。本文为基于监督学习或序列建模框架的强化学习算法构建了一个理论基础。本文的核心在于分析算法能够在何种环境条件下识别出最优解，并评估在环境存在微小噪声的情况下，这些解决方案是否保持稳定。具体而言，我们研究了命令条件策略、价值和目标从属目标随基本马尔可夫决策过程转移核变化的连续性和渐近收敛性。研究表明，如果转移核位于确定性核的一个足够小的邻域内，则可以实现接近最优的行为。这些量在确定性核处都是连续的（以特定拓扑为准），无论是渐近意义上还是在有限的学习周期后。本文所发展的方法使我们能够首次明确给出策略和价值收敛及稳定性的估计，以基本转移核为准。在理论方面，我们引入了几个新的概念，例如在区间空间中工作、研究商拓扑下的连续性和动力系统不动点理论的应用。理论研究伴随着对示例环境的详细研究和数值实验。 

---
# Adversarial Machine Learning: Attacks, Defenses, and Open Challenges 

**Title (ZH)**: 对抗机器学习：攻击、防御及开放挑战 

**Authors**: Pranav K Jha  

**Link**: [PDF](https://arxiv.org/pdf/2502.05637)  

**Abstract**: Adversarial Machine Learning (AML) addresses vulnerabilities in AI systems where adversaries manipulate inputs or training data to degrade performance. This article provides a comprehensive analysis of evasion and poisoning attacks, formalizes defense mechanisms with mathematical rigor, and discusses the challenges of implementing robust solutions in adaptive threat models. Additionally, it highlights open challenges in certified robustness, scalability, and real-world deployment. 

**Abstract (ZH)**: adversarial machine learning (AML) 在对手操控输入或训练数据以削弱人工智能系统性能的漏洞中进行应对。本文提供了对规避攻击和投毒攻击的全面分析，以数学严格性形式化了防御机制，并讨论了在适应性威胁模型中实施稳健解决方案的挑战。此外，本文还强调了在认证稳健性、扩展性和实际部署方面存在的开放性挑战。 

---
# ATLAS: Autoformalizing Theorems through Lifting, Augmentation, and Synthesis of Data 

**Title (ZH)**: ATLAS: 通过提升、扩增和数据合成自动形式化定理 

**Authors**: Xiaoyang Liu, Kangjie Bao, Jiashuo Zhang, Yunqi Liu, Yu Chen, Yuntian Liu, Yang Jiao, Tao Luo  

**Link**: [PDF](https://arxiv.org/pdf/2502.05567)  

**Abstract**: Autoformalization, the process of automatically translating natural language mathematics into machine-verifiable formal language, has demonstrated advancements with the progress of large language models (LLMs). However, a key obstacle to further advancements is the scarcity of paired datasets that align natural language with formal language. To address this challenge, we introduce ATLAS (Autoformalizing Theorems through Lifting, Augmentation, and Synthesis of Data), an iterative data generation framework designed to produce large-scale, high-quality parallel theorem statements. With the proposed ATLAS running for 10 iterations, we construct an undergraduate-level dataset comprising 300k theorem statements and develop the ATLAS translator, achieving accuracies of 80.59% (pass@8) and 92.99% (pass@128) on ProofNet, significantly outperforming the base model (23.99% and 47.17%) and InternLM2-Math-Plus-7B (50.94% and 80.32%). Furthermore, the ATLAS translator also achieves state-of-the-art performance on both the high-school-level miniF2F dataset and the graduate-level MathQual dataset introduced in this work. The datasets, model, and code will be released to the public soon. 

**Abstract (ZH)**: 自动形式化：通过提升、扩充和合成数据自动生成形式化定理陈述的方法及其实现 

---
# TabICL: A Tabular Foundation Model for In-Context Learning on Large Data 

**Title (ZH)**: TabICL：大规模数据下基于上下文的表格预训练模型 

**Authors**: Jingang Qu, David Holzmüller, Gaël Varoquaux, Marine Le Morvan  

**Link**: [PDF](https://arxiv.org/pdf/2502.05564)  

**Abstract**: The long-standing dominance of gradient-boosted decision trees on tabular data is currently challenged by tabular foundation models using In-Context Learning (ICL): setting the training data as context for the test data and predicting in a single forward pass without parameter updates. While the very recent TabPFNv2 foundation model (2025) excels on tables with up to 10K samples, its alternating column- and row-wise attentions make handling large training sets computationally prohibitive. So, can ICL be effectively scaled and deliver a benefit for larger tables? We introduce TabICL, a tabular foundation model for classification, pretrained on synthetic datasets with up to 60K samples and capable of handling 500K samples on affordable resources. This is enabled by a novel two-stage architecture: a column-then-row attention mechanism to build fixed-dimensional embeddings of rows, followed by a transformer for efficient ICL. Across 200 classification datasets from the TALENT benchmark, TabICL is on par with TabPFNv2 while being systematically faster (up to 10 times), and significantly outperforms all other approaches. On 56 datasets with over 10K samples, TabICL surpasses both TabPFNv2 and CatBoost, demonstrating the potential of ICL for large data. 

**Abstract (ZH)**: 表格数据上基于上下文学习的表格基础模型TabICL：一种新型两阶段架构的分类表格基础模型 

---
# Dual Defense: Enhancing Privacy and Mitigating Poisoning Attacks in Federated Learning 

**Title (ZH)**: 双层防御：增强联邦学习中的隐私保护并减轻 poisoning 攻击 

**Authors**: Runhua Xu, Shiqi Gao, Chao Li, James Joshi, Jianxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.05547)  

**Abstract**: Federated learning (FL) is inherently susceptible to privacy breaches and poisoning attacks. To tackle these challenges, researchers have separately devised secure aggregation mechanisms to protect data privacy and robust aggregation methods that withstand poisoning attacks. However, simultaneously addressing both concerns is challenging; secure aggregation facilitates poisoning attacks as most anomaly detection techniques require access to unencrypted local model updates, which are obscured by secure aggregation. Few recent efforts to simultaneously tackle both challenges offen depend on impractical assumption of non-colluding two-server setups that disrupt FL's topology, or three-party computation which introduces scalability issues, complicating deployment and application. To overcome this dilemma, this paper introduce a Dual Defense Federated learning (DDFed) framework. DDFed simultaneously boosts privacy protection and mitigates poisoning attacks, without introducing new participant roles or disrupting the existing FL topology. DDFed initially leverages cutting-edge fully homomorphic encryption (FHE) to securely aggregate model updates, without the impractical requirement for non-colluding two-server setups and ensures strong privacy protection. Additionally, we proposes a unique two-phase anomaly detection mechanism for encrypted model updates, featuring secure similarity computation and feedback-driven collaborative selection, with additional measures to prevent potential privacy breaches from Byzantine clients incorporated into the detection process. We conducted extensive experiments on various model poisoning attacks and FL scenarios, including both cross-device and cross-silo FL. Experiments on publicly available datasets demonstrate that DDFed successfully protects model privacy and effectively defends against model poisoning threats. 

**Abstract (ZH)**: 联邦学习（FL）本质上容易遭受隐私泄露和污染攻击。为应对这些挑战，研究人员分别设计了安全聚合机制来保护数据隐私和抗污染攻击的方法。然而，同时应对这两种挑战颇具挑战性；安全聚合会促进污染攻击，因为大多数异常检测技术需要访问未加密的本地模型更新，而这些更新被安全聚合所遮蔽。近期少数同时应对这两种挑战的努力往往依赖于无法实现的非串通双服务器设置假设，这干扰了FL的拓扑结构，或者依赖于三方计算，这引入了可扩展性问题，复杂了部署和应用。为克服这一困境，本文提出了一种双防御联邦学习（DDFed）框架。DDFed同步增强了隐私保护并缓解了污染攻击，无需引入新的参与者角色或扰乱现有的FL拓扑结构。DDFed初始利用最先进的全同态加密（FHE）安全聚合模型更新，无需无法实现的非串通双服务器设置假设，并确保强大的隐私保护。此外，我们提出了一种独特的两阶段异常检测机制，用于加密模型更新，该机制包括安全相似度计算和反馈驱动的合作选择，并通过检测过程中的额外措施防止潜在的拜占庭客户端导致的隐私泄露。我们在各种模型污染攻击和FL场景中进行了广泛的实验，包括设备间联邦学习和库间联邦学习。公开数据集上的实验结果表明，DDFed成功保护了模型隐私并有效防御了模型污染威胁。 

---
# Riemannian Manifold Learning for Stackelberg Games with Neural Flow Representations 

**Title (ZH)**: 基于神经流表示的 Stackelberg 游戏的黎曼流形学习 

**Authors**: Larkin Liu, Kashif Rasul, Yutong Chao, Jalal Etesami  

**Link**: [PDF](https://arxiv.org/pdf/2502.05498)  

**Abstract**: We present a novel framework for online learning in Stackelberg general-sum games, where two agents, the leader and follower, engage in sequential turn-based interactions. At the core of this approach is a learned diffeomorphism that maps the joint action space to a smooth Riemannian manifold, referred to as the Stackelberg manifold. This mapping, facilitated by neural normalizing flows, ensures the formation of tractable isoplanar subspaces, enabling efficient techniques for online learning. By assuming linearity between the agents' reward functions on the Stackelberg manifold, our construct allows the application of standard bandit algorithms. We then provide a rigorous theoretical basis for regret minimization on convex manifolds and establish finite-time bounds on simple regret for learning Stackelberg equilibria. This integration of manifold learning into game theory uncovers a previously unrecognized potential for neural normalizing flows as an effective tool for multi-agent learning. We present empirical results demonstrating the effectiveness of our approach compared to standard baselines, with applications spanning domains such as cybersecurity and economic supply chain optimization. 

**Abstract (ZH)**: 我们提出了一种在Stackelberg广义博弈中进行在线学习的新型框架，其中两个代理，领导者和追随者，进行顺序的轮流互动。该方法的核心是一个学习到的 diffeomorphism 映射，将联合动作空间映射到一个光滑的黎曼流形，称为Stackelberg流形。这种映射通过神经归一化流实现，确保形成可处理的同面子空间，从而使得在线学习的高效技术成为可能。通过在Stackelberg流形上假设代理的奖励函数之间存在线性关系，我们的构建允许使用标准的多臂 bandit 算法。然后，我们为凸流形上的后悔最小化提供了严格的理论基础，并建立了学习Stackelberg均衡的有限时间简单后悔的上界。将流形学习整合到博弈论中揭示了神经归一化流作为多代理学习有效工具的先前未被认识到的潜力。我们展示了与标准基线相比，该方法的有效性实证结果，涉及的领域包括网络安全和经济供应链优化。 

---
# Multi-scale Masked Autoencoder for Electrocardiogram Anomaly Detection 

**Title (ZH)**: 多尺度遮蔽自动编码器用于心电图异常检测 

**Authors**: Ya Zhou, Yujie Yang, Jianhuang Gan, Xiangjie Li, Jing Yuan, Wei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.05494)  

**Abstract**: Electrocardiogram (ECG) analysis is a fundamental tool for diagnosing cardiovascular conditions, yet anomaly detection in ECG signals remains challenging due to their inherent complexity and variability. We propose Multi-scale Masked Autoencoder for ECG anomaly detection (MMAE-ECG), a novel end-to-end framework that effectively captures both global and local dependencies in ECG data. Unlike state-of-the-art methods that rely on heartbeat segmentation or R-peak detection, MMAE-ECG eliminates the need for such pre-processing steps, enhancing its suitability for clinical deployment. MMAE-ECG partitions ECG signals into non-overlapping segments, with each segment assigned learnable positional embeddings. A novel multi-scale masking strategy and multi-scale attention mechanism, along with distinct positional embeddings, enable a lightweight Transformer encoder to effectively capture both local and global dependencies. The masked segments are then reconstructed using a single-layer Transformer block, with an aggregation strategy employed during inference to refine the outputs. Experimental results demonstrate that our method achieves performance comparable to state-of-the-art approaches while significantly reducing computational complexity-approximately 1/78 of the floating-point operations (FLOPs) required for inference. Ablation studies further validate the effectiveness of each component, highlighting the potential of multi-scale masked autoencoders for anomaly detection. 

**Abstract (ZH)**: 多尺度掩蔽自动编码器在ECG异常检测中的应用（MMAE-ECG） 

---
# Unbiased Sliced Wasserstein Kernels for High-Quality Audio Captioning 

**Title (ZH)**: 无偏分层 Wasserstein 核用于高质量音频字幕生成 

**Authors**: Manh Luong, Khai Nguyen, Dinh Phung, Gholamreza Haffari, Lizhen Qu  

**Link**: [PDF](https://arxiv.org/pdf/2502.05435)  

**Abstract**: Teacher-forcing training for audio captioning usually leads to exposure bias due to training and inference mismatch. Prior works propose the contrastive method to deal with caption degeneration. However, the contrastive method ignores the temporal information when measuring similarity across acoustic and linguistic modalities, leading to inferior performance. In this work, we develop the temporal-similarity score by introducing the unbiased sliced Wasserstein RBF (USW-RBF) kernel equipped with rotary positional embedding to account for temporal information across modalities. In contrast to the conventional sliced Wasserstein RBF kernel, we can form an unbiased estimation of USW-RBF kernel via Monte Carlo estimation. Therefore, it is well-suited to stochastic gradient optimization algorithms, and its approximation error decreases at a parametric rate of $\mathcal{O}(L^{-1/2})$ with $L$ Monte Carlo samples. Additionally, we introduce an audio captioning framework based on the unbiased sliced Wasserstein kernel, incorporating stochastic decoding methods to mitigate caption degeneration during the generation process. We conduct extensive quantitative and qualitative experiments on two datasets, AudioCaps and Clotho, to illustrate the capability of generating high-quality audio captions. Experimental results show that our framework is able to increase caption length, lexical diversity, and text-to-audio self-retrieval accuracy. 

**Abstract (ZH)**: 基于无偏时空拟合的音频字幕生成教师强制训练方法 

---
# APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding 

**Title (ZH)**: APE：通过自适应并行编码实现更快、更长上下文增强生成 

**Authors**: Xinyu Yang, Tianqi Chen, Beidi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.05431)  

**Abstract**: Context-augmented generation (CAG) techniques, including RAG and ICL, require the efficient combination of multiple contexts to generate responses to user queries. Directly inputting these contexts as a sequence introduces a considerable computational burden by re-encoding the combined selection of contexts for every request. To address this, we explore the promising potential of parallel encoding to independently pre-compute and cache each context's KV states. This approach enables the direct loading of cached states during inference while accommodating more contexts through position reuse across contexts. However, due to misalignments in attention distribution, directly applying parallel encoding results in a significant performance drop. To enable effective and efficient CAG, we propose Adaptive Parallel Encoding ($\textbf{APE}$), which brings shared prefix, attention temperature, and scaling factor to align the distribution of parallel encoding with sequential encoding. Results on RAG and ICL tasks demonstrate that APE can preserve 98% and 93% sequential encoding performance using the same inputs while outperforming parallel encoding by 3.6% and 7.9%, respectively. It also scales to many-shot CAG, effectively encoding hundreds of contexts in parallel. Efficiency evaluation shows that APE can achieve an end-to-end 4.5$\times$ speedup by reducing 28$\times$ prefilling time for a 128K-length context. 

**Abstract (ZH)**: 基于上下文增强生成（CAG）技术，包括RAG和ICL，需要高效地结合多个上下文以生成用户查询的响应。直接将这些上下文作为序列输入会因每次请求都需要重新编译组合后的上下文而导致显著的计算负担。为解决这一问题，我们探索了并行编码的潜力，独立预计算和缓存每个上下文的KV状态。这种方法允许在推理过程中直接加载缓存状态，并通过上下文间的共享位置实现更多上下文的处理。然而，由于注意力分布的不匹配，直接应用并行编码会导致显著的性能下降。为了实现有效的和高效的CAG，我们提出了自适应并行编码（APE），它引入了共享前缀、注意力温度和缩放因子，以使并行编码的注意力分布与序列编码的注意力分布相匹配。在RAG和ICL任务上的结果表明，APE能够在使用相同输入的情况下保持98%和93%的序列编码性能，并且分别比并行编码高出3.6%和7.9%。此外，APE可以扩展到多-shot CAG，有效地并行编码数百个上下文。效率评估显示，APE可以通过减少28倍的预填充时间（对于128K长度的上下文），实现端到端4.5倍的加速。 

---
# Is attention all you need to solve the correlated electron problem? 

**Title (ZH)**: 解决相关电子问题是否只需要注意力？ 

**Authors**: Max Geier, Khachatur Nazaryan, Timothy Zaklama, Liang Fu  

**Link**: [PDF](https://arxiv.org/pdf/2502.05383)  

**Abstract**: The attention mechanism has transformed artificial intelligence research by its ability to learn relations between objects. In this work, we explore how a many-body wavefunction ansatz constructed from a large-parameter self-attention neural network can be used to solve the interacting electron problem in solids. By a systematic neural-network variational Monte Carlo study on a moiré quantum material, we demonstrate that the self-attention ansatz provides an accurate, efficient, and unbiased solution. Moreover, our numerical study finds that the required number of variational parameters scales roughly as $N^2$ with the number of electrons, which opens a path towards efficient large-scale simulations. 

**Abstract (ZH)**: self-attention机制构建的多体波函数Ansatz在固态交互电子问题中的应用研究 

---
# Estimating Voltage Drop: Models, Features and Data Representation Towards a Neural Surrogate 

**Title (ZH)**: 电压降估算：基于神经近似的模型、特征和数据表示 

**Authors**: Yifei Jin, Dimitrios Koutlis, Hector Bandala, Marios Daoutis  

**Link**: [PDF](https://arxiv.org/pdf/2502.05345)  

**Abstract**: Accurate estimation of voltage drop (IR drop) in modern Application-Specific Integrated Circuits (ASICs) is highly time and resource demanding, due to the growing complexity and the transistor density in recent technology nodes. To mitigate this challenge, we investigate how Machine Learning (ML) techniques, including Extreme Gradient Boosting (XGBoost), Convolutional Neural Network (CNN), and Graph Neural Network (GNN) can aid in reducing the computational effort and implicitly the time required to estimate the IR drop in Integrated Circuits (ICs). Traditional methods, including commercial tools, require considerable time to produce accurate approximations, especially for complicated designs with numerous transistors. ML algorithms, on the other hand, are explored as an alternative solution to offer quick and precise IR drop estimation, but in considerably less time. Our approach leverages ASICs' electrical, timing, and physical to train ML models, ensuring adaptability across diverse designs with minimal adjustments. Experimental results underscore the superiority of ML models over commercial tools, greatly enhancing prediction speed. Particularly, GNNs exhibit promising performance with minimal prediction errors in voltage drop estimation. The incorporation of GNNs marks a groundbreaking advancement in accurate IR drop prediction. This study illustrates the effectiveness of ML algorithms in precisely estimating IR drop and optimizing ASIC sign-off. Utilizing ML models leads to expedited predictions, reducing calculation time and improving energy efficiency, thereby reducing environmental impact through optimized power circuits. 

**Abstract (ZH)**: 现代Application-Specific Integrated Circuits (ASICs)中准确估计电压降（IR drop）的方法：基于机器学习技术的高效解决方案 

---
# Towards the Development of Balanced Synthetic Data for Correcting Grammatical Errors in Arabic: An Approach Based on Error Tagging Model and Synthetic Data Generating Model 

**Title (ZH)**: 基于错误标注模型和合成数据生成模型的平衡合成数据开发方法：用于纠正阿拉伯语语法错误的研究 

**Authors**: Ahlam Alrehili, Areej Alhothali  

**Link**: [PDF](https://arxiv.org/pdf/2502.05312)  

**Abstract**: Synthetic data generation is widely recognized as a way to enhance the quality of neural grammatical error correction (GEC) systems. However, current approaches often lack diversity or are too simplistic to generate the wide range of grammatical errors made by humans, especially for low-resource languages such as Arabic. In this paper, we will develop the error tagging model and the synthetic data generation model to create a large synthetic dataset in Arabic for grammatical error correction. In the error tagging model, the correct sentence is categorized into multiple error types by using the DeBERTav3 model. Arabic Error Type Annotation tool (ARETA) is used to guide multi-label classification tasks in an error tagging model in which each sentence is classified into 26 error tags. The synthetic data generation model is a back-translation-based model that generates incorrect sentences by appending error tags before the correct sentence that was generated from the error tagging model using the ARAT5 model. In the QALB-14 and QALB-15 Test sets, the error tagging model achieved 94.42% F1, which is state-of-the-art in identifying error tags in clean sentences. As a result of our syntactic data training in grammatical error correction, we achieved a new state-of-the-art result of F1-Score: 79.36% in the QALB-14 Test set. We generate 30,219,310 synthetic sentence pairs by using a synthetic data generation model. 

**Abstract (ZH)**: 合成数据生成被广泛认为是一种提升神经语法错误修正系统质量的方法。然而，当前的方法往往缺乏多样性和简化度，难以生成人类所犯的各种语法错误，尤其是对于阿拉伯语等低资源语言。在本文中，我们将开发错误标记模型和合成数据生成模型，为阿拉伯语语法错误修正创建大规模合成数据集。在错误标记模型中，使用DeBERTav3模型将正确句子分类为多个错误类型。阿拉伯错误类型注释工具（ARETA）用于指导错误标记模型中的多标签分类任务，其中每句句子被分类为26个错误标签。合成数据生成模型是一种基于反向翻译的模型，通过在由ARAT5模型生成的正确句子之前附加错误标签来生成错误句子。在QALB-14和QALB-15测试集中，错误标记模型实现了94.42%的F1值，这是在干净句子中识别错误标签的最新技术水平。通过我们的句法数据训练，我们在QALB-14测试集中实现了新的最佳F1-Score：79.36%。我们使用合成数据生成模型生成了30,219,310对合成句子对。 

---
# Parameter Symmetry Breaking and Restoration Determines the Hierarchical Learning in AI Systems 

**Title (ZH)**: 参数对称性破缺与恢复决定AI系统的分层学习 

**Authors**: Liu Ziyin, Yizhou Xu, Tomaso Poggio, Isaac Chuang  

**Link**: [PDF](https://arxiv.org/pdf/2502.05300)  

**Abstract**: The dynamics of learning in modern large AI systems is hierarchical, often characterized by abrupt, qualitative shifts akin to phase transitions observed in physical systems. While these phenomena hold promise for uncovering the mechanisms behind neural networks and language models, existing theories remain fragmented, addressing specific cases. In this paper, we posit that parameter symmetry breaking and restoration serve as a unifying mechanism underlying these behaviors. We synthesize prior observations and show how this mechanism explains three distinct hierarchies in neural networks: learning dynamics, model complexity, and representation formation. By connecting these hierarchies, we highlight symmetry -- a cornerstone of theoretical physics -- as a potential fundamental principle in modern AI. 

**Abstract (ZH)**: 现代大型AI系统的学习动力学是分层的，通常表现为类似于物理系统相变的 abrupt、qualitative 转变。尽管这些现象为揭示神经网络和语言模型的机制提供了希望，现有理论仍碎片化，仅针对特定案例。本文认为，参数对称性破缺与恢复是一种统一这些行为的机制。我们综合了先前的观察，并展示了这一机制如何解释神经网络中的三种不同层次：学习动力学、模型复杂性和表示形成。通过连接这些层次，我们将对称性——理论物理的基石——突出为现代AI中潜在的基本原则。 

---
# Quantum automated learning with provable and explainable trainability 

**Title (ZH)**: 量子自动学习具有可证明和可解释的可训练性 

**Authors**: Qi Ye, Shuangyue Geng, Zizhao Han, Weikang Li, L.-M. Duan, Dong-Ling Deng  

**Link**: [PDF](https://arxiv.org/pdf/2502.05264)  

**Abstract**: Machine learning is widely believed to be one of the most promising practical applications of quantum computing. Existing quantum machine learning schemes typically employ a quantum-classical hybrid approach that relies crucially on gradients of model parameters. Such an approach lacks provable convergence to global minima and will become infeasible as quantum learning models scale up. Here, we introduce quantum automated learning, where no variational parameter is involved and the training process is converted to quantum state preparation. In particular, we encode training data into unitary operations and iteratively evolve a random initial state under these unitaries and their inverses, with a target-oriented perturbation towards higher prediction accuracy sandwiched in between. Under reasonable assumptions, we rigorously prove that the evolution converges exponentially to the desired state corresponding to the global minimum of the loss function. We show that such a training process can be understood from the perspective of preparing quantum states by imaginary time evolution, where the data-encoded unitaries together with target-oriented perturbations would train the quantum learning model in an automated fashion. We further prove that the quantum automated learning paradigm features good generalization ability with the generalization error upper bounded by the ratio between a logarithmic function of the Hilbert space dimension and the number of training samples. In addition, we carry out extensive numerical simulations on real-life images and quantum data to demonstrate the effectiveness of our approach and validate the assumptions. Our results establish an unconventional quantum learning strategy that is gradient-free with provable and explainable trainability, which would be crucial for large-scale practical applications of quantum computing in machine learning scenarios. 

**Abstract (ZH)**: 无梯度的量子自动化学习：具有可证明和可解释的训练能力的量子机器学习新策略 

---
# PSM-SQL: Progressive Schema Learning with Multi-granularity Semantics for Text-to-SQL 

**Title (ZH)**: PSM-SQL：基于多粒度语义的 progressive 架构学习文本到SQL转换 

**Authors**: Zhuopan Yang, Yuanzhen Xie, Ruichao Zhong, Yunzhi Tan, Enjie Liu, Zhenguo Yang, Mochi Gao, Bo Hu, Zang Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.05237)  

**Abstract**: It is challenging to convert natural language (NL) questions into executable structured query language (SQL) queries for text-to-SQL tasks due to the vast number of database schemas with redundancy, which interferes with semantic learning, and the domain shift between NL and SQL. Existing works for schema linking focus on the table level and perform it once, ignoring the multi-granularity semantics and chainable cyclicity of schemas. In this paper, we propose a progressive schema linking with multi-granularity semantics (PSM-SQL) framework to reduce the redundant database schemas for text-to-SQL. Using the multi-granularity schema linking (MSL) module, PSM-SQL learns the schema semantics at the column, table, and database levels. More specifically, a triplet loss is used at the column level to learn embeddings, while fine-tuning LLMs is employed at the database level for schema reasoning. MSL employs classifier and similarity scores to model schema interactions for schema linking at the table level. In particular, PSM-SQL adopts a chain loop strategy to reduce the task difficulty of schema linking by continuously reducing the number of redundant schemas. Experiments conducted on text-to-SQL datasets show that the proposed PSM-SQL is 1-3 percentage points higher than the existing methods. 

**Abstract (ZH)**: 一种多粒度语义渐进模式链接的文本到SQL框架（PSM-SQL） 

---
# Aligner-Encoders: Self-Attention Transformers Can Be Self-Transducers 

**Title (ZH)**: 对齐编码器：自我注意变换器可以是自转换器 

**Authors**: Adam Stooke, Rohit Prabhavalkar, Khe Chai Sim, Pedro Moreno Mengibar  

**Link**: [PDF](https://arxiv.org/pdf/2502.05232)  

**Abstract**: Modern systems for automatic speech recognition, including the RNN-Transducer and Attention-based Encoder-Decoder (AED), are designed so that the encoder is not required to alter the time-position of information from the audio sequence into the embedding; alignment to the final text output is processed during decoding. We discover that the transformer-based encoder adopted in recent years is actually capable of performing the alignment internally during the forward pass, prior to decoding. This new phenomenon enables a simpler and more efficient model, the "Aligner-Encoder". To train it, we discard the dynamic programming of RNN-T in favor of the frame-wise cross-entropy loss of AED, while the decoder employs the lighter text-only recurrence of RNN-T without learned cross-attention -- it simply scans embedding frames in order from the beginning, producing one token each until predicting the end-of-message. We conduct experiments demonstrating performance remarkably close to the state of the art, including a special inference configuration enabling long-form recognition. In a representative comparison, we measure the total inference time for our model to be 2x faster than RNN-T and 16x faster than AED. Lastly, we find that the audio-text alignment is clearly visible in the self-attention weights of a certain layer, which could be said to perform "self-transduction". 

**Abstract (ZH)**: 基于变压器的“对齐-编码器”模型：一种简单高效的声音到文本的自动转换方法 

---
# DiffNMR2: NMR Guided Sampling Acquisition Through Diffusion Model Uncertainty 

**Title (ZH)**: DiffNMR2：基于扩散模型不确定性指导的核磁共振采样 Acquisition通过核磁共振不确定性指导的扩散模型采样 

**Authors**: Etienne Goffinet, Sen Yan, Fabrizio Gabellieri, Laurence Jennings, Lydia Gkoura, Filippo Castiglione, Ryan Young, Idir Malki, Ankita Singh, Thomas Launey  

**Link**: [PDF](https://arxiv.org/pdf/2502.05230)  

**Abstract**: Nuclear Magnetic Resonance (NMR) spectrometry uses electro-frequency pulses to probe the resonance of a compound's nucleus, which is then analyzed to determine its structure. The acquisition time of high-resolution NMR spectra remains a significant bottleneck, especially for complex biological samples such as proteins. In this study, we propose a novel and efficient sub-sampling strategy based on a diffusion model trained on protein NMR data. Our method iteratively reconstructs under-sampled spectra while using model uncertainty to guide subsequent sampling, significantly reducing acquisition time. Compared to state-of-the-art strategies, our approach improves reconstruction accuracy by 52.9\%, reduces hallucinated peaks by 55.6%, and requires 60% less time in complex NMR experiments. This advancement holds promise for many applications, from drug discovery to materials science, where rapid and high-resolution spectral analysis is critical. 

**Abstract (ZH)**: 核磁共振（NMR）光谱学使用射频脉冲探测化合物核的共振状态，然后通过分析确定其结构。高分辨率NMR光谱的采集时间仍然是一个显著的瓶颈，尤其是在蛋白质等复杂生物样品中。在本研究中，我们提出了一种基于蛋白质NMR数据训练的扩散模型的新型高效子抽样策略。该方法通过使用模型不确定性指导后续抽样，逐次重建欠采样光谱，显著减少了采集时间。相较于最先进的策略，我们的方法在重构准确性上提高了52.9%，减少了55.6%的幻峰，并在复杂NMR实验中所需时间减少了60%。这一进展在药物发现、材料科学等领域具有重要应用前景，特别是在需要快速高分辨率光谱分析的情况下。 

---
# Multi-Objective Mobile Damped Wave Algorithm (MOMDWA): A Novel Approach For Quantum System Control 

**Title (ZH)**: 多目标移动阻尼波算法（MOMDWA）：一种量子系统控制的新方法 

**Authors**: Juntao Yu, Jiaquan Yu, Dedai Wei, Xinye Sha, Shengwei Fu, Miuyu Qiu, Yurun Jin, Kaichen Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2502.05228)  

**Abstract**: In this paper, we introduce a novel multi-objective optimization algorithm, the Multi-Objective Mobile Damped Wave Algorithm (MOMDWA), specifically designed to address complex quantum control problems. Our approach extends the capabilities of the original Mobile Damped Wave Algorithm (MDWA) by incorporating multiple objectives, enabling a more comprehensive optimization process. We applied MOMDWA to three quantum control scenarios, focusing on optimizing the balance between control fidelity, energy consumption, and control smoothness. The results demonstrate that MOMDWA significantly enhances quantum control efficiency and robustness, achieving high fidelity while minimizing energy use and ensuring smooth control pulses. This advancement offers a valuable tool for quantum computing and other domains requiring precise, multi-objective control. 

**Abstract (ZH)**: 基于移动阻尼波算法的多目标优化量子控制方法（MOMDWA） 

---
# BitAbuse: A Dataset of Visually Perturbed Texts for Defending Phishing Attacks 

**Title (ZH)**: BitAbuse: 一种视觉扰动文本数据集，用于防骗攻击 

**Authors**: Hanyong Lee, Chaelyn Lee, Yongjae Lee, Jaesung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.05225)  

**Abstract**: Phishing often targets victims through visually perturbed texts to bypass security systems. The noise contained in these texts functions as an adversarial attack, designed to deceive language models and hinder their ability to accurately interpret the content. However, since it is difficult to obtain sufficient phishing cases, previous studies have used synthetic datasets that do not contain real-world cases. In this study, we propose the BitAbuse dataset, which includes real-world phishing cases, to address the limitations of previous research. Our dataset comprises a total of 325,580 visually perturbed texts. The dataset inputs are drawn from the raw corpus, consisting of visually perturbed sentences and sentences generated through an artificial perturbation process. Each input sentence is labeled with its corresponding ground truth, representing the restored, non-perturbed version. Language models trained on our proposed dataset demonstrated significantly better performance compared to previous methods, achieving an accuracy of approximately 96%. Our analysis revealed a significant gap between real-world and synthetic examples, underscoring the value of our dataset for building reliable pre-trained models for restoration tasks. We release the BitAbuse dataset, which includes real-world phishing cases annotated with visual perturbations, to support future research in adversarial attack defense. 

**Abstract (ZH)**: Phishing often targets victims through visually perturbed texts to bypass security systems. The noise contained in these texts functions as an adversarial attack, designed to deceive language models and hinder their ability to accurately interpret the content. However, since it is difficult to obtain sufficient phishing cases, previous studies have used synthetic datasets that do not contain real-world cases. In this study, we propose the BitAbuse dataset, which includes real-world phishing cases, to address the limitations of previous research. Our dataset comprises a total of 325,580 visually perturbed texts. The dataset inputs are drawn from the raw corpus, consisting of visually perturbed sentences and sentences generated through an artificial perturbation process. Each input sentence is labeled with its corresponding ground truth, representing the restored, non-perturbed version. Language models trained on our proposed dataset demonstrated significantly better performance compared to previous methods, achieving an accuracy of approximately 96%. Our analysis revealed a significant gap between real-world and synthetic examples, underscoring the value of our dataset for building reliable pre-trained models for restoration tasks. We release the BitAbuse dataset, which includes real-world phishing cases annotated with visual perturbations, to support future research in adversarial attack defense. 

---
# Blackout DIFUSCO 

**Title (ZH)**: blackout DIFUSCO 

**Authors**: Jun Pyo Seo  

**Link**: [PDF](https://arxiv.org/pdf/2502.05221)  

**Abstract**: This study explores the integration of Blackout Diffusion into the DIFUSCO framework for combinatorial optimization, specifically targeting the Traveling Salesman Problem (TSP). Inspired by the success of discrete-time diffusion models (D3PM) in maintaining structural integrity, we extend the paradigm to a continuous-time framework, leveraging the unique properties of Blackout Diffusion. Continuous-time modeling introduces smoother transitions and refined control, hypothesizing enhanced solution quality over traditional discrete methods. We propose three key improvements to enhance the diffusion process. First, we transition from a discrete-time-based model to a continuous-time framework, providing a more refined and flexible formulation. Second, we refine the observation time scheduling to ensure a smooth and linear transformation throughout the diffusion process, allowing for a more natural progression of states. Finally, building upon the second improvement, we further enhance the reverse process by introducing finer time slices in regions that are particularly challenging for the model, thereby improving accuracy and stability in the reconstruction phase. Although the experimental results did not exceed the baseline performance, they demonstrate the effectiveness of these methods in balancing simplicity and complexity, offering new insights into diffusion-based combinatorial optimization. This work represents the first application of Blackout Diffusion to combinatorial optimization, providing a foundation for further advancements in this domain. * The code is available for review at this https URL. 

**Abstract (ZH)**: 本研究探讨将Blackout Diffusion整合到DIFUSCO框架中以解决组合优化问题，具体针对旅行商问题（TSP）。受离散时间扩散模型（D3PM）在保持结构完整性方面的成功启发，我们将这一范式扩展到连续时间框架，利用Blackout Diffusion的独特属性。连续时间建模引入了更平滑的过渡和更精细的控制，假设与传统离散方法相比能提高解的质量。我们提出了三种关键改进以增强扩散过程。首先，我们从基于离散时间的模型过渡到连续时间框架，提供了更精细和灵活的表述。其次，我们优化了观测时间的调度，确保扩散过程中有平滑和线性的转换，允许状态更自然地演变。最后，在第二个改进的基础上，我们通过在特别具有挑战性的区域引入更细的时间片来进一步增强逆过程，从而在重建阶段提高准确性和稳定性。尽管实验结果未超过基线性能，但它们展示了这些方法在平衡简单性和复杂性方面的有效性，并为基于扩散的组合优化提供了新的见解。本工作是将Blackout Diffusion应用于组合优化的第一个尝试，为其在此领域的进一步发展提供了基础。* 代码可在以下链接进行查看：![](https://your-code-link.com)。 

---
# Enabling External Scrutiny of AI Systems with Privacy-Enhancing Technologies 

**Title (ZH)**: 借助隐私增强技术实现AI系统的外部审视 

**Authors**: Kendrea Beers, Helen Toner  

**Link**: [PDF](https://arxiv.org/pdf/2502.05219)  

**Abstract**: This article describes how technical infrastructure developed by the nonprofit OpenMined enables external scrutiny of AI systems without compromising sensitive information.
Independent external scrutiny of AI systems provides crucial transparency into AI development, so it should be an integral component of any approach to AI governance. In practice, external researchers have struggled to gain access to AI systems because of AI companies' legitimate concerns about security, privacy, and intellectual property.
But now, privacy-enhancing technologies (PETs) have reached a new level of maturity: end-to-end technical infrastructure developed by OpenMined combines several PETs into various setups that enable privacy-preserving audits of AI systems. We showcase two case studies where this infrastructure has been deployed in real-world governance scenarios: "Understanding Social Media Recommendation Algorithms with the Christchurch Call" and "Evaluating Frontier Models with the UK AI Safety Institute." We describe types of scrutiny of AI systems that could be facilitated by current setups and OpenMined's proposed future setups.
We conclude that these innovative approaches deserve further exploration and support from the AI governance community. Interested policymakers can focus on empowering researchers on a legal level. 

**Abstract (ZH)**: 开源组织OpenMined开发的技术基础设施如何实现AI系统的外部审查同时保护敏感信息 

---
# FactorGCL: A Hypergraph-Based Factor Model with Temporal Residual Contrastive Learning for Stock Returns Prediction 

**Title (ZH)**: 基于超图的因子模型与时间残差对比学习相结合的股票收益预测 

**Authors**: Yitong Duan, Weiran Wang, Jian Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.05218)  

**Abstract**: As a fundamental method in economics and finance, the factor model has been extensively utilized in quantitative investment. In recent years, there has been a paradigm shift from traditional linear models with expert-designed factors to more flexible nonlinear machine learning-based models with data-driven factors, aiming to enhance the effectiveness of these factor models. However, due to the low signal-to-noise ratio in market data, mining effective factors in data-driven models remains challenging. In this work, we propose a hypergraph-based factor model with temporal residual contrastive learning (FactorGCL) that employs a hypergraph structure to better capture high-order nonlinear relationships among stock returns and factors. To mine hidden factors that supplement human-designed prior factors for predicting stock returns, we design a cascading residual hypergraph architecture, in which the hidden factors are extracted from the residual information after removing the influence of prior factors. Additionally, we propose a temporal residual contrastive learning method to guide the extraction of effective and comprehensive hidden factors by contrasting stock-specific residual information over different time periods. Our extensive experiments on real stock market data demonstrate that FactorGCL not only outperforms existing state-of-the-art methods but also mines effective hidden factors for predicting stock returns. 

**Abstract (ZH)**: 基于超图的时空残差对比学习因子模型（FactorGCL） 

---
# Watermarking across Modalities for Content Tracing and Generative AI 

**Title (ZH)**: 跨模态水印技术在内容追踪与生成型AI中的应用 

**Authors**: Pierre Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2502.05215)  

**Abstract**: Watermarking embeds information into digital content like images, audio, or text, imperceptible to humans but robustly detectable by specific algorithms. This technology has important applications in many challenges of the industry such as content moderation, tracing AI-generated content, and monitoring the usage of AI models. The contributions of this thesis include the development of new watermarking techniques for images, audio, and text. We first introduce methods for active moderation of images on social platforms. We then develop specific techniques for AI-generated content. We specifically demonstrate methods to adapt latent generative models to embed watermarks in all generated content, identify watermarked sections in speech, and improve watermarking in large language models with tests that ensure low false positive rates. Furthermore, we explore the use of digital watermarking to detect model misuse, including the detection of watermarks in language models fine-tuned on watermarked text, and introduce training-free watermarks for the weights of large transformers. Through these contributions, the thesis provides effective solutions for the challenges posed by the increasing use of generative AI models and the need for model monitoring and content moderation. It finally examines the challenges and limitations of watermarking techniques and discuss potential future directions for research in this area. 

**Abstract (ZH)**: 水印技术嵌入数字内容如图像、音频或文本，对人类不可感知但可通过特定算法可靠检测。该项技术在内容审核、追踪AI生成内容以及监控AI模型使用等方面有着重要的应用价值。本论文的贡献在于开发了适用于图像、音频和文本的新水印技术。我们首先介绍了在社交平台上的主动图像审核方法。然后，我们开发了特定技术以处理AI生成的内容。我们具体展示了如何调整潜在生成模型以在所有生成内容中嵌入水印、如何在语音中识别水marked部分、以及通过确保低误报率来改进在大规模语言模型中的水印技术。此外，我们探讨了使用数字水印检测模型滥用的方法，包括在基于水marked文本 fine-tuned 的语言模型中检测水印，并引入了无训练水印以应用于大规模变换器的权重。通过这些贡献，论文提供了应对生成AI模型使用增加带来的挑战以及模型监控和内容审核需要的有效解决方案。最后，论文探讨了水印技术的挑战和限制，并讨论了该领域的未来研究方向。 

---
# CoRPA: Adversarial Image Generation for Chest X-rays Using Concept Vector Perturbations and Generative Models 

**Title (ZH)**: CoRPA：使用概念向量扰动和生成模型的胸部X光 adversarial 图像生成 

**Authors**: Amy Rafferty, Rishi Ramaesh, Ajitha Rajan  

**Link**: [PDF](https://arxiv.org/pdf/2502.05214)  

**Abstract**: Deep learning models for medical image classification tasks are becoming widely implemented in AI-assisted diagnostic tools, aiming to enhance diagnostic accuracy, reduce clinician workloads, and improve patient outcomes. However, their vulnerability to adversarial attacks poses significant risks to patient safety. Current attack methodologies use general techniques such as model querying or pixel value perturbations to generate adversarial examples designed to fool a model. These approaches may not adequately address the unique characteristics of clinical errors stemming from missed or incorrectly identified clinical features. We propose the Concept-based Report Perturbation Attack (CoRPA), a clinically-focused black-box adversarial attack framework tailored to the medical imaging domain. CoRPA leverages clinical concepts to generate adversarial radiological reports and images that closely mirror realistic clinical misdiagnosis scenarios. We demonstrate the utility of CoRPA using the MIMIC-CXR-JPG dataset of chest X-rays and radiological reports. Our evaluation reveals that deep learning models exhibiting strong resilience to conventional adversarial attacks are significantly less robust when subjected to CoRPA's clinically-focused perturbations. This underscores the importance of addressing domain-specific vulnerabilities in medical AI systems. By introducing a specialized adversarial attack framework, this study provides a foundation for developing robust, real-world-ready AI models in healthcare, ensuring their safe and reliable deployment in high-stakes clinical environments. 

**Abstract (ZH)**: 基于概念的报告扰动攻击：面向医疗成像领域的临床聚焦黑盒对抗攻击框架 

---
# Decoding FL Defenses: Systemization, Pitfalls, and Remedies 

**Title (ZH)**: 解码FL防御：系统化、陷阱与对策 

**Authors**: Momin Ahmad Khan, Virat Shejwalkar, Yasra Chandio, Amir Houmansadr, Fatima Muhammad Anwar  

**Link**: [PDF](https://arxiv.org/pdf/2502.05211)  

**Abstract**: While the community has designed various defenses to counter the threat of poisoning attacks in Federated Learning (FL), there are no guidelines for evaluating these defenses. These defenses are prone to subtle pitfalls in their experimental setups that lead to a false sense of security, rendering them unsuitable for practical deployment. In this paper, we systematically understand, identify, and provide a better approach to address these challenges. First, we design a comprehensive systemization of FL defenses along three dimensions: i) how client updates are processed, ii) what the server knows, and iii) at what stage the defense is applied. Next, we thoroughly survey 50 top-tier defense papers and identify the commonly used components in their evaluation setups. Based on this survey, we uncover six distinct pitfalls and study their prevalence. For example, we discover that around 30% of these works solely use the intrinsically robust MNIST dataset, and 40% employ simplistic attacks, which may inadvertently portray their defense as robust. Using three representative defenses as case studies, we perform a critical reevaluation to study the impact of the identified pitfalls and show how they lead to incorrect conclusions about robustness. We provide actionable recommendations to help researchers overcome each pitfall. 

**Abstract (ZH)**: 在联邦学习中抵御投毒攻击的各种防御措施虽已设计，但缺乏评估指南。实验设置中的细微陷阱可能导致虚假的安全感，使其不适合实际部署。本文系统地理解、识别这些问题，并提供更好的解决方案。首先，我们从三个维度设计了联邦学习防御系统的全面体系结构：i) 客户端更新的处理方式，ii) 服务器掌握的信息，iii) 防御措施的应用阶段。接着，我们彻底调查了50篇顶级防御论文，并识别出其评估设置中常用的部分。基于此调查，我们发现了六种不同的陷阱，并研究了它们的普遍性。例如，我们发现约30%的工作仅使用内在 robust 的MNIST数据集，而40%的工作采用简单的攻击方法，这可能会无意中将其防御措施表现为 robust。通过三篇代表性防御措施作为案例研究，我们进行关键性的重新评估，研究识别出的陷阱的影响，并展示它们如何导致关于 robust 性的错误结论。我们提供了可操作的建议，帮助研究人员克服每个陷阱。 

---
# Multimodal Stock Price Prediction 

**Title (ZH)**: 多模态股票价格预测 

**Authors**: Furkan Karadaş, Bahaeddin Eravcı, Ahmet Murat Özbayoğlu  

**Link**: [PDF](https://arxiv.org/pdf/2502.05186)  

**Abstract**: In an era where financial markets are heavily influenced by many static and dynamic factors, it has become increasingly critical to carefully integrate diverse data sources with machine learning for accurate stock price prediction. This paper explores a multimodal machine learning approach for stock price prediction by combining data from diverse sources, including traditional financial metrics, tweets, and news articles. We capture real-time market dynamics and investor mood through sentiment analysis on these textual data using both ChatGPT-4o and FinBERT models. We look at how these integrated data streams augment predictions made with a standard Long Short-Term Memory (LSTM model) to illustrate the extent of performance gains. Our study's results indicate that incorporating the mentioned data sources considerably increases the forecast effectiveness of the reference model by up to 5%. We also provide insights into the individual and combined predictive capacities of these modalities, highlighting the substantial impact of incorporating sentiment analysis from tweets and news articles. This research offers a systematic and effective framework for applying multimodal data analytics techniques in financial time series forecasting that provides a new view for investors to leverage data for decision-making. 

**Abstract (ZH)**: 在金融市场受到众多静态和动态因素强烈影响的时代，准确整合多种数据源并结合机器学习进行股票价格预测变得日益关键。本文探讨了一种基于多模态机器学习的股票价格预测方法，综合了传统财务指标、推特和新闻文章等多种数据源。通过使用ChatGPT-4o和FinBERT模型对这些文本数据进行情感分析，我们捕捉实时的市场动态和投资者情绪。我们研究这些集成数据流如何增强标准长短期记忆（LSTM）模型的预测能力，并展示了性能提升的程度。研究结果表明，整合提及的数据源可将参考模型的预测效果提高多达5%。我们还分析了这些模态的单独及综合预测能力，强调了从推特和新闻文章中进行情感分析的重要性。本文提供了一种系统且有效的方法，用于在金融时间序列预测中应用多模态数据分析技术，为投资者利用数据进行决策提供了新的视角。 

---
