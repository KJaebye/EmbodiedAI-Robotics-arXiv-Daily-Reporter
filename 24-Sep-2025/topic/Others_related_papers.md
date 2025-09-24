# Residual Off-Policy RL for Finetuning Behavior Cloning Policies 

**Title (ZH)**: 残差离策RLbehavior克隆策略微调 

**Authors**: Lars Ankile, Zhenyu Jiang, Rocky Duan, Guanya Shi, Pieter Abbeel, Anusha Nagabandi  

**Link**: [PDF](https://arxiv.org/pdf/2509.19301)  

**Abstract**: Recent advances in behavior cloning (BC) have enabled impressive visuomotor control policies. However, these approaches are limited by the quality of human demonstrations, the manual effort required for data collection, and the diminishing returns from increasing offline data. In comparison, reinforcement learning (RL) trains an agent through autonomous interaction with the environment and has shown remarkable success in various domains. Still, training RL policies directly on real-world robots remains challenging due to sample inefficiency, safety concerns, and the difficulty of learning from sparse rewards for long-horizon tasks, especially for high-degree-of-freedom (DoF) systems. We present a recipe that combines the benefits of BC and RL through a residual learning framework. Our approach leverages BC policies as black-box bases and learns lightweight per-step residual corrections via sample-efficient off-policy RL. We demonstrate that our method requires only sparse binary reward signals and can effectively improve manipulation policies on high-degree-of-freedom (DoF) systems in both simulation and the real world. In particular, we demonstrate, to the best of our knowledge, the first successful real-world RL training on a humanoid robot with dexterous hands. Our results demonstrate state-of-the-art performance in various vision-based tasks, pointing towards a practical pathway for deploying RL in the real world. Project website: this https URL 

**Abstract (ZH)**: 最近行为克隆的进展使得可视化运动控制策略取得了显著成效。然而，这些方法受限于人类示范的质量、数据收集所需的 manual 努力，以及额外数据带来的边际收益递减。相比之下，强化学习通过自主与环境交互来训练代理，并且已经在多个领域展示了显著的成功。尽管如此，直接在真实世界机器人上训练强化学习策略仍然具有挑战性，原因包括样本效率低、安全问题以及从稀疏奖励中学习长时间任务的困难，尤其是在高自由度系统中。我们提出了一种结合行为克隆和强化学习优点的方法，采用残差学习框架。我们的方法利用行为克隆策略作为黑盒基础，并通过样本高效的行为聚类强化学习学习轻量级的逐步残差修正。我们证明，我们的方法仅需要稀疏的二元奖励信号，并且可以有效地提高高自由度系统的操作策略，无论是仿真还是真实世界中。特别地，我们首次展示了一种灵巧手的人形机器人在真实世界中使用强化学习训练成功的实例。我们的结果表明，在各种基于视觉的任务中达到了最先进的性能，展示了强化学习在真实世界部署的实际途径。项目网站: 这里是链接。 

---
# Application Management in C-ITS: Orchestrating Demand-Driven Deployments and Reconfigurations 

**Title (ZH)**: C-ITS中应用管理：基于需求的部署与重新配置协调 

**Authors**: Lukas Zanger, Bastian Lampe, Lennart Reiher, Lutz Eckstein  

**Link**: [PDF](https://arxiv.org/pdf/2509.18793)  

**Abstract**: Vehicles are becoming increasingly automated and interconnected, enabling the formation of cooperative intelligent transport systems (C-ITS) and the use of offboard services. As a result, cloud-native techniques, such as microservices and container orchestration, play an increasingly important role in their operation. However, orchestrating applications in a large-scale C-ITS poses unique challenges due to the dynamic nature of the environment and the need for efficient resource utilization. In this paper, we present a demand-driven application management approach that leverages cloud-native techniques - specifically Kubernetes - to address these challenges. Taking into account the demands originating from different entities within the C-ITS, the approach enables the automation of processes, such as deployment, reconfiguration, update, upgrade, and scaling of microservices. Executing these processes on demand can, for example, reduce computing resource consumption and network traffic. A demand may include a request for provisioning an external supporting service, such as a collective environment model. The approach handles changing and new demands by dynamically reconciling them through our proposed application management framework built on Kubernetes and the Robot Operating System (ROS 2). We demonstrate the operation of our framework in the C-ITS use case of collective environment perception and make the source code of the prototypical framework publicly available at this https URL . 

**Abstract (ZH)**: 车辆 increasingly automated and interconnected，促使合作智能运输系统（C-ITS）的形成以及外部服务的利用。随之而来，云原生技术，如微服务和容器编排，在其运行中发挥了越来越重要的作用。然而，在大规模C-ITS中编排应用程序由于环境的动态性和高效的资源利用率需求，带来独特的挑战。本文提出一种基于云原生技术的需求驱动应用程序管理方法，特别是Kubernetes，以应对这些挑战。该方法考虑了C-ITS内不同实体的需求，实现了部署、重新配置、更新、升级和微服务扩展的自动化过程。这些过程的需求执行，例如，可以减少计算资源消耗和网络流量。需求可能包括请求提供外部支持服务，如集体环境模型。通过在基于Kubernetes和机器人操作系统ROS 2的应用程序管理框架中的动态解算，该方法能够处理变化和新的需求。我们展示了该框架在C-ITS中的集体环境感知用例中的运行，并在以下网址公开提供了该原型框架的源代码：this https URL。 

---
# Human-Interpretable Uncertainty Explanations for Point Cloud Registration 

**Title (ZH)**: 面向人类可解释的点云配准不确定性解释 

**Authors**: Johannes A. Gaus, Loris Schneider, Yitian Shi, Jongseok Lee, Rania Rayyes, Rudolph Triebel  

**Link**: [PDF](https://arxiv.org/pdf/2509.18786)  

**Abstract**: In this paper, we address the point cloud registration problem, where well-known methods like ICP fail under uncertainty arising from sensor noise, pose-estimation errors, and partial overlap due to occlusion. We develop a novel approach, Gaussian Process Concept Attribution (GP-CA), which not only quantifies registration uncertainty but also explains it by attributing uncertainty to well-known sources of errors in registration problems. Our approach leverages active learning to discover new uncertainty sources in the wild by querying informative instances. We validate GP-CA on three publicly available datasets and in our real-world robot experiment. Extensive ablations substantiate our design choices. Our approach outperforms other state-of-the-art methods in terms of runtime, high sample-efficiency with active learning, and high accuracy. Our real-world experiment clearly demonstrates its applicability. Our video also demonstrates that GP-CA enables effective failure-recovery behaviors, yielding more robust robotic perception. 

**Abstract (ZH)**: 基于高斯过程概念归因的点云配准方法：不确定性量化与解释 

---
# Distributionally Robust Safe Motion Planning with Contextual Information 

**Title (ZH)**: 基于上下文信息的分布鲁棒安全运动规划 

**Authors**: Kaizer Rahaman, Simran Kumari, Ashish R. Hota  

**Link**: [PDF](https://arxiv.org/pdf/2509.18666)  

**Abstract**: We present a distributionally robust approach for collision avoidance by incorporating contextual information. Specifically, we embed the conditional distribution of future trajectory of the obstacle conditioned on the motion of the ego agent in a reproducing kernel Hilbert space (RKHS) via the conditional kernel mean embedding operator. Then, we define an ambiguity set containing all distributions whose embedding in the RKHS is within a certain distance from the empirical estimate of conditional mean embedding learnt from past data. Consequently, a distributionally robust collision avoidance constraint is formulated, and included in the receding horizon based motion planning formulation of the ego agent. Simulation results show that the proposed approach is more successful in avoiding collision compared to approaches that do not include contextual information and/or distributional robustness in their formulation in several challenging scenarios. 

**Abstract (ZH)**: 我们提出了一种通过融合上下文信息的分布鲁棒方法来实现碰撞避免。具体地，我们通过条件核均值嵌入算子将代理ego的运动条件下障碍物未来轨迹的条件分布嵌入到再生核希尔伯特空间（RKHS）中。然后，我们定义一个模糊集合，该集合包含所有嵌入与RKHS中条件均值嵌入的经验估计在特定距离内的分布。由此，我们制定了一个分布鲁棒的碰撞避免约束，并将其纳入基于回视 horizons的运动规划框架中。模拟结果表明，与未包含上下文信息和/或分布鲁棒性的方法相比，所提出的方法在多个具有挑战性的场景中更成功地避免了碰撞。 

---
# The Case for Negative Data: From Crash Reports to Counterfactuals for Reasonable Driving 

**Title (ZH)**: 负数据的案情：从故障报告到合理的驾驶反事实推理 

**Authors**: Jay Patrikar, Apoorva Sharma, Sushant Veer, Boyi Li, Sebastian Scherer, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2509.18626)  

**Abstract**: Learning-based autonomous driving systems are trained mostly on incident-free data, offering little guidance near safety-performance boundaries. Real crash reports contain precisely the contrastive evidence needed, but they are hard to use: narratives are unstructured, third-person, and poorly grounded to sensor views. We address these challenges by normalizing crash narratives to ego-centric language and converting both logs and crashes into a unified scene-action representation suitable for retrieval. At decision time, our system adjudicates proposed actions by retrieving relevant precedents from this unified index; an agentic counterfactual extension proposes plausible alternatives, retrieves for each, and reasons across outcomes before deciding. On a nuScenes benchmark, precedent retrieval substantially improves calibration, with recall on contextually preferred actions rising from 24% to 53%. The counterfactual variant preserves these gains while sharpening decisions near risk. 

**Abstract (ZH)**: 基于学习的自动驾驶系统主要在无事故数据上进行训练，提供的安全性性能边界上的指导有限。真实的事故报告包含了必要的对比性证据，但难以使用：这些报告的叙事内容结构不一、第三方视角且与传感器视角关联性差。我们通过将事故叙述规范化为以自我为中心的语言，并将日志和事故统一转换为适合检索的场景-行动表示来应对这些挑战。在决策时刻，系统通过检索这个统一索引中的相关先例来裁定提出的行动；具有代理性的反事实扩展提出可能的替代方案，分别检索并跨结果进行推理后再做决定。在nuScenes基准测试中，先例检索显著提高了校准度，上下文偏好行动的召回率从24%提高到53%。反事实变体保留了这些改进，同时在高风险区域增强了决策的精确度。 

---
# An Extended Kalman Filter for Systems with Infinite-Dimensional Measurements 

**Title (ZH)**: 适用于无限维观测系统的扩展卡尔曼滤波器 

**Authors**: Maxwell M. Varley, Timothy L. Molloy, Girish N. Nair  

**Link**: [PDF](https://arxiv.org/pdf/2509.18749)  

**Abstract**: This article examines state estimation in discrete-time nonlinear stochastic systems with finite-dimensional states and infinite-dimensional measurements, motivated by real-world applications such as vision-based localization and tracking. We develop an extended Kalman filter (EKF) for real-time state estimation, with the measurement noise modeled as an infinite-dimensional random field. When applied to vision-based state estimation, the measurement Jacobians required to implement the EKF are shown to correspond to image gradients. This result provides a novel system-theoretic justification for the use of image gradients as features for vision-based state estimation, contrasting with their (often heuristic) introduction in many computer-vision pipelines. We demonstrate the practical utility of the EKF on a public real-world dataset involving the localization of an aerial drone using video from a downward-facing monocular camera. The EKF is shown to outperform VINS-MONO, an established visual-inertial odometry algorithm, in some cases achieving mean squared error reductions of up to an order of magnitude. 

**Abstract (ZH)**: 本文探讨了在状态有限维、测量无限维的离散时间非线性随机系统中状态估计的问题，受到了基于视觉的定位与跟踪等实际应用的启发。我们开发了一种扩展卡尔曼滤波器（EKF）进行实时状态估计，并将测量噪声建模为无限维随机场。在应用于基于视觉的状态估计时，实现EKF所需的测量雅可比矩阵对应于图像梯度。这一结果为使用图像梯度作为基于视觉的状态估计特征提供了新颖的系统理论依据，区别于其在许多计算机视觉管道中的（通常是启发式的）引入方式。我们利用一个公开的真实世界数据集，展示了EKF在使用向下视角单目摄像头视频进行空中无人机定位上的实用价值。在某些情况下，EKF相较于成熟的视觉惯性里程计算法VINS-MONO显示出均方误差降低一个数量级以上的性能提升。 

---
# Policy Gradient with Self-Attention for Model-Free Distributed Nonlinear Multi-Agent Games 

**Title (ZH)**: 基于自注意力的策略梯度在无模型分布式非线性多智能体博弈中的应用 

**Authors**: Eduardo Sebastián, Maitrayee Keskar, Eeman Iqbal, Eduardo Montijano, Carlos Sagüés, Nikolay Atanasov  

**Link**: [PDF](https://arxiv.org/pdf/2509.18371)  

**Abstract**: Multi-agent games in dynamic nonlinear settings are challenging due to the time-varying interactions among the agents and the non-stationarity of the (potential) Nash equilibria. In this paper we consider model-free games, where agent transitions and costs are observed without knowledge of the transition and cost functions that generate them. We propose a policy gradient approach to learn distributed policies that follow the communication structure in multi-team games, with multiple agents per team. Our formulation is inspired by the structure of distributed policies in linear quadratic games, which take the form of time-varying linear feedback gains. In the nonlinear case, we model the policies as nonlinear feedback gains, parameterized by self-attention layers to account for the time-varying multi-agent communication topology. We demonstrate that our distributed policy gradient approach achieves strong performance in several settings, including distributed linear and nonlinear regulation, and simulated and real multi-robot pursuit-and-evasion games. 

**Abstract (ZH)**: 动态非线性环境中的多智能体博弈因智能体间的时间变化交互和纳什均衡的非稳态性而具有挑战性。在本文中，我们考虑无模型的博弈，其中智能体的状态转移和成本可以被观察到，但不了解产生这些转移和成本的功能形式。我们提出了一种策略梯度方法，用于学习遵循多队列博弈中通信结构的分布式策略，每队包含多个智能体。我们的框架灵感来自于线性二次博弈中分布式策略的结构，它们的形式为时间变化的线性反馈增益。在非线性情况下，我们将策略建模为非线性反馈增益，并通过自注意力层参数化以考虑时间变化的多智能体通信拓扑。我们展示了我们的分布式策略梯度方法在分布式线性和非线性调节以及模拟和实际多机器人追逃游戏中均表现出色。 

---
# Reversible Kalman Filter for state estimation with Manifold 

**Title (ZH)**: 流形上状态估计的可逆卡尔曼滤波器 

**Authors**: Svyatoslav Covanov, Cedric Pradalier  

**Link**: [PDF](https://arxiv.org/pdf/2509.18224)  

**Abstract**: This work introduces an algorithm for state estimation on manifolds within the framework of the Kalman filter. Its primary objective is to provide a methodology enabling the evaluation of the precision of existing Kalman filter variants with arbitrary accuracy on synthetic data, something that, to the best of our knowledge, has not been addressed in prior work. To this end, we develop a new filter that exhibits favorable numerical properties, thereby correcting the divergences observed in previous Kalman filter variants. In this formulation, the achievable precision is no longer constrained by the small-velocity assumption and is determined solely by sensor noise. In addition, this new filter assumes high precision on the sensors, which, in real scenarios require a detection step that we define heuristically, allowing one to extend this approach to scenarios, using either a 9-axis IMU or a combination of odometry, accelerometer, and pressure sensors. The latter configuration is designed for the reconstruction of trajectories in underwater environments. 

**Abstract (ZH)**: 本文介绍了一种在流形框架下Kalman滤波器的态估计算法。其主要目标是在合成数据上以任意精度评估现有Kalman滤波器变体的精度，这是迄今为止Prior工作尚未解决的问题。为此，我们开发了一种新滤波器，具有良好的数值性质，从而纠正了先前Kalman滤波器变体中的发散问题。在此框架下，可实现的精度不再受小速度假设的限制，而是仅由传感器噪声决定。此外，该新滤波器假设传感器具有高精度，在实际场景中需要进行一个我们以启发式方法定义的检测步骤，从而可以将该方法扩展到使用9轴IMU或组合使用里程计、加速度计和压力传感器的场景中，后者用于水下环境中的轨迹重构。 

---
# Towards Causal Representation Learning with Observable Sources as Auxiliaries 

**Title (ZH)**: 基于可观测源作为辅助的因果表示学习 

**Authors**: Kwonho Kim, Heejeong Nam, Inwoo Hwang, Sanghack Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.19058)  

**Abstract**: Causal representation learning seeks to recover latent factors that generate observational data through a mixing function. Needing assumptions on latent structures or relationships to achieve identifiability in general, prior works often build upon conditional independence given known auxiliary variables. However, prior frameworks limit the scope of auxiliary variables to be external to the mixing function. Yet, in some cases, system-driving latent factors can be easily observed or extracted from data, possibly facilitating identification. In this paper, we introduce a framework of observable sources being auxiliaries, serving as effective conditioning variables. Our main results show that one can identify entire latent variables up to subspace-wise transformations and permutations using volume-preserving encoders. Moreover, when multiple known auxiliary variables are available, we offer a variable-selection scheme to choose those that maximize recoverability of the latent factors given knowledge of the latent causal graph. Finally, we demonstrate the effectiveness of our framework through experiments on synthetic graph and image data, thereby extending the boundaries of current approaches. 

**Abstract (ZH)**: 可观测源作为辅助变量的因果表示学习 

---
# Landmarks, Monuments, and Beacons: Understanding Generative Calls to Action 

**Title (ZH)**: 地标、纪念碑与灯塔：理解生成式召唤行动 

**Authors**: Victoire Hervé, Henrik Warpefelt, Christoph Salge  

**Link**: [PDF](https://arxiv.org/pdf/2509.19030)  

**Abstract**: Algorithmic evaluation of procedurally generated content struggles to find metrics that align with human experience, particularly for composite artefacts. Automatic decomposition as a possible solution requires concepts that meet a range of properties. To this end, drawing on Games Studies and Game AI research, we introduce the nested concepts of \textit{Landmarks}, \textit{Monuments}, and \textit{Beacons}. These concepts are based on the artefact's perceivability, evocativeness, and Call to Action, all from a player-centric perspective. These terms are generic to games and usable across genres. We argue that these entities can be found and evaluated with techniques currently used in both research and industry, opening a path towards a fully automated decomposition of PCG, and evaluation of the salient sub-components. Although the work presented here emphasises mixed-initiative PCG and compositional PCG, we believe it applies beyond those domains. With this approach, we intend to create a connection between humanities and technical game research and allow for better computational PCG evaluation 

**Abstract (ZH)**: 程序生成内容的算法评估难以找到与人类体验相一致的度量标准，特别是对于复合制品。自动分解作为一种可能的解决方案需要满足一系列特性的概念。为此，借鉴Games Studies和Game AI研究，我们引入了嵌套概念：Landmarks、Monuments和Beacons。这些概念基于制品的可感知性、唤起性和呼吁行动，均从玩家中心的角度出发。这些术语适用于各类游戏，并且在不同游戏类型间通用。我们认为这些实体可以通过当前在研究和工业中均存在的技术进行发现和评估，从而开辟一条通往混合主动型PCG和组合型PCG全自动分解及其关键子组件评估的道路。尽管本文强调混合主动型PCG和组合型PCG，但我们认为这一方法适用于更广泛的领域。通过这种方法，我们旨在建立人文科学与技术游戏研究之间的联系，并允许更好地进行计算型PCG评估。 

---
# Remaining Time Prediction in Outbound Warehouse Processes: A Case Study (Short Paper) 

**Title (ZH)**: 出库仓库流程中剩余时间预测：一个案例研究（简短论文） 

**Authors**: Erik Penther, Michael Grohs, Jana-Rebecca Rehse  

**Link**: [PDF](https://arxiv.org/pdf/2509.18986)  

**Abstract**: Predictive process monitoring is a sub-domain of process mining which aims to forecast the future of ongoing process executions. One common prediction target is the remaining time, meaning the time that will elapse until a process execution is completed. In this paper, we compare four different remaining time prediction approaches in a real-life outbound warehouse process of a logistics company in the aviation business. For this process, the company provided us with a novel and original event log with 169,523 traces, which we can make publicly available. Unsurprisingly, we find that deep learning models achieve the highest accuracy, but shallow methods like conventional boosting techniques achieve competitive accuracy and require significantly fewer computational resources. 

**Abstract (ZH)**: 基于实际航空物流仓库过程的剩余时间预测方法比较 

---
# MAPO: Mixed Advantage Policy Optimization 

**Title (ZH)**: MAPO: 混合优势策略优化 

**Authors**: Wenke Huang, Quan Zhang, Yiyang Fang, Jian Liang, Xuankun Rong, Huanjin Yao, Guancheng Wan, Ke Liang, Wenwen He, Mingjun Li, Leszek Rutkowski, Mang Ye, Bo Du, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2509.18849)  

**Abstract**: Recent advances in reinforcement learning for foundation models, such as Group Relative Policy Optimization (GRPO), have significantly improved the performance of foundation models on reasoning tasks. Notably, the advantage function serves as a central mechanism in GRPO for ranking the trajectory importance. However, existing explorations encounter both advantage reversion and advantage mirror problems, which hinder the reasonable advantage allocation across different query samples. In this work, we propose an easy but effective GRPO strategy, Mixed Advantage Policy Optimization (MAPO). We reveal that the trajectory appears with different certainty and propose the advantage percent deviation for samples with high-certainty trajectories. Furthermore, we dynamically reweight the advantage function for samples with varying trajectory certainty, thereby adaptively configuring the advantage function to account for sample-specific characteristics. Comparison with related state-of-the-art methods, along with ablation studies on different advantage variants, validates the effectiveness of our approach. 

**Abstract (ZH)**: 最近在基础模型中强化学习的进步，如Group Relative Policy Optimization (GRPO)，显著提高了基础模型在推理任务上的性能。值得注意的是，优势函数在GRPO中作为核心机制用于评价轨迹的重要性。然而，现有的探索遇到优势反转和优势镜像问题，这妨碍了不同查询样本间合理的优势分配。在这项工作中，我们提出了一种简单而有效的GRPO策略，混合优势策略优化（MAPO）。我们揭示了轨迹存在不同确定性，并为高确定性轨迹的样本引入优势百分比偏差。此外，我们动态调整优势函数的权重以适应轨迹确定性变化的样本，从而根据样本特定特征适应性配置优势函数。与相关先进方法的比较以及不同优势变体的消融研究验证了我们方法的有效性。 

---
# The AGNTCY Agent Directory Service: Architecture and Implementation 

**Title (ZH)**: AGNTCY代理目录服务：架构与实现 

**Authors**: Luca Muscariello, Vijoy Pandey, Ramiz Polic  

**Link**: [PDF](https://arxiv.org/pdf/2509.18787)  

**Abstract**: The Agent Directory Service (ADS) is a distributed directory for the discovery of AI agent capabilities, metadata, and provenance. It leverages content-addressed storage, hierarchical taxonomies, and cryptographic signing to enable efficient, verifiable, and multi-dimensional discovery across heterogeneous Multi-Agent Systems (MAS). Built on the Open Agentic Schema Framework (OASF), ADS decouples capability indexing from content location through a two-level mapping realized over a Kademlia-based Distributed Hash Table (DHT). It reuses mature OCI / ORAS infrastructure for artifact distribution, integrates Sigstore for provenance, and supports schema-driven extensibility for emerging agent modalities (LLM prompt agents, MCP servers, A2A-enabled components). This paper formalizes the architectural model, describes storage and discovery layers, explains security and performance properties, and positions ADS within the broader landscape of emerging agent registry and interoperability initiatives. 

**Abstract (ZH)**: Agent_DIRECTORY_SERVICE (ADS) 是一个分布式目录，用于发现AI代理能力、元数据和起源。它利用内容寻址存储、层次分类法和加密签名，实现跨异构多代理系统（MAS）的高效、可验证和多维度发现。基于Open Agentic Schema Framework (OASF)，ADS 通过基于Kademlia的分布式哈希表（DHT）实现的两层映射，将能力索引与内容位置解耦。它重用了成熟的OCI / ORAS基础设施进行 artifacts 分发，集成了Sigstore用于证明，并支持基于模式的扩展性以支持新兴代理模态（如LLM提示代理、MCP服务器、A2A启用的组件）。本文形式化了该架构模型，描述了存储和发现层，解释了安全性和性能属性，并将ADS置于更广泛的新兴代理注册和互操作性倡议的大环境中。 

---
# Implementation of airborne ML models with semantics preservation 

**Title (ZH)**: 具有语义保留的机载ML模型实现 

**Authors**: Nicolas Valot, Louis Fabre, Benjamin Lesage, Ammar Mechouche, Claire Pagetti  

**Link**: [PDF](https://arxiv.org/pdf/2509.18681)  

**Abstract**: Machine Learning (ML) may offer new capabilities in airborne systems. However, as any piece of airborne systems, ML-based systems will be required to guarantee their safe operation. Thus, their development will have to be demonstrated to be compliant with the adequate guidance. So far, the European Union Aviation Safety Agency (EASA) has published a concept paper and an EUROCAE/SAE group is preparing ED-324. Both approaches delineate high-level objectives to confirm the ML model achieves its intended function and maintains training performance in the target environment. The paper aims to clarify the difference between an ML model and its corresponding unambiguous description, referred to as the Machine Learning Model Description (MLMD). It then refines the essential notion of semantics preservation to ensure the accurate replication of the model. We apply our contributions to several industrial use cases to build and compare several target models. 

**Abstract (ZH)**: 基于机器学习的空中系统可能提供新的能力。然而，如同任何空中系统一样，基于机器学习的系统也需要保证其安全运行。因此，其开发必须证明符合适当的指导方针。到目前为止，欧洲航空安全局（EASA）已经发布了概念文件，而一个EUROCAE/SAE小组正在准备ED-324。这两种方法都规定了高层次的目标，以确保机器学习模型实现其预期功能并在目标环境中保持训练性能。本文旨在阐明机器学习模型与其相应的明确描述之间的区别，将其称为机器学习模型描述（MLMD），并进一步细化语义保留的基本概念，以确保模型的精确复制。我们应用我们的贡献到几个工业应用案例中，构建并比较了几种目标模型。 

---
# Adaptive Learning in Spatial Agent-Based Models for Climate Risk Assessment: A Geospatial Framework with Evolutionary Economic Agents 

**Title (ZH)**: 基于空间的代理模型中空间自适应学习在气候风险评估中的应用：演化经济代理的地理空间框架 

**Authors**: Yara Mohajerani  

**Link**: [PDF](https://arxiv.org/pdf/2509.18633)  

**Abstract**: Climate risk assessment requires modelling complex interactions between spatially heterogeneous hazards and adaptive economic systems. We present a novel geospatial agent-based model that integrates climate hazard data with evolutionary learning for economic agents. Our framework combines Mesa-based spatial modelling with CLIMADA climate impact assessment, introducing adaptive learning behaviours that allow firms to evolve strategies for budget allocation, pricing, wages, and risk adaptation through fitness-based selection and mutation. We demonstrate the framework using riverine flood projections under RCP8.5 until 2100, showing that evolutionary adaptation enables firms to converge with baseline (no hazard) production levels after decades of disruption due to climate stress. Our results reveal systemic risks where even agents that are not directly exposed to floods face impacts through supply chain disruptions, with the end-of-century average price of goods 5.6% higher under RCP8.5 compared to the baseline. This open-source framework provides financial institutions and companies with tools to quantify both direct and cascading climate risks while evaluating cost-effective adaptation strategies. 

**Abstract (ZH)**: 气候变化风险评估要求建模空间异质性风险与适应性经济系统之间的复杂交互作用。我们提出了一种新颖的地理空间基于代理的模型，将气候灾害数据与经济代理的演化学习相结合。我们的框架结合了基于Mesa的空间建模与CLIMADA气候影响评估，引入了适应性学习行为，允许企业通过基于适应性的选择和突变来进化预算分配、定价、工资和风险适应策略。我们使用RCP8.5情景下的直至2100年的河流洪水预测来展示该框架，表明演化适应使企业在数十年的气候变化冲击后能够与基准（无灾害）生产水平收敛。我们的结果揭示了系统性风险，在这种风险下，即使未直接暴露于洪水中的代理也通过供应链中断受到影响，而RCP8.5情景下到本世纪末的商品平均价格比基准情景高出5.6%。该开源框架为金融机构和企业提供工具，用于量化直接和连锁气候风险，并评估成本效益适应策略。 

---
# Towards General Computer Control with Hierarchical Agents and Multi-Level Action Spaces 

**Title (ZH)**: 具有分层代理和多级动作空间的通用计算机控制 

**Authors**: Zihan Dong, Xinyu Fan, Zixiang Tang, Yunqing Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.18230)  

**Abstract**: Controlling desktop applications via software remains a fundamental yet under-served problem. Existing multi-modal large language models (MLLMs) ingest screenshots and task instructions to generate keystrokes and mouse events, but they suffer from prohibitive inference latency, poor sample efficiency on long-horizon sparse-reward tasks, and infeasible on-device deployment. We introduce a lightweight hierarchical reinforcement learning framework, ComputerAgent, that formulates OS control as a two-level option process (manager and subpolicy), employs a triple-modal state encoder (screenshot, task ID, numeric state) to handle visual and contextual diversity, integrates meta-actions with an early-stop mechanism to reduce wasted interactions, and uses a compact vision backbone plus small policy networks for on-device inference (15M parameters). On a suite of 135 real-world desktop tasks, ComputerAgent attains 92.1% success on simple tasks (<8 steps) and 58.8% on hard tasks (>=8 steps), matching or exceeding 200B-parameter MLLM baselines on simple scenarios while reducing model size by over four orders of magnitude and halving inference time. These results demonstrate that hierarchical RL offers a practical, scalable alternative to monolithic MLLM-based automation for computer control. 

**Abstract (ZH)**: 通过软件控制桌面应用程序仍是一个基础但尚未充分解决的问题。现有的多模态大语言模型（MLLM）通过摄入屏幕截图和任务指令来生成键盘输入和鼠标事件，但它们遭受于推断延迟高、长期稀疏奖励任务样本效率低以及设备端部署不可行的难题。我们引入了一个轻量级的层次强化学习框架ComputerAgent，将操作系统控制建模为两层选择过程（管理器和子策略），采用三模态状态编码器（屏幕截图、任务ID、数值状态）处理视觉和上下文多样性，结合元动作和早期停止机制减少无效交互，并使用紧凑的视觉主干和小型策略网络进行设备端推理（参数量15M）。在一系列135个实际桌面任务中，ComputerAgent在简单任务（<8步）中的成功率达到了92.1%，在困难任务（>=8步）中的成功率为58.8%，在简单场景中达到或超过了2000亿参数的大语言模型基线，同时将模型大小减少了四个数量级，并将推理时间减半。这些结果表明，层次强化学习为计算机控制提供了实用且可扩展的替代方案，不同于基于大语言模型的单体自动化。 

---
# Similarity Field Theory: A Mathematical Framework for Intelligence 

**Title (ZH)**: 相似性场理论：智能的数学框架 

**Authors**: Kei-Sing Ng  

**Link**: [PDF](https://arxiv.org/pdf/2509.18218)  

**Abstract**: We posit that persisting and transforming similarity relations form the structural basis of any comprehensible dynamic system. This paper introduces Similarity Field Theory, a mathematical framework that formalizes the principles governing similarity values among entities and their evolution. We define: (1) a similarity field $S: U \times U \to [0,1]$ over a universe of entities $U$, satisfying reflexivity $S(E,E)=1$ and treated as a directed relational field (asymmetry and non-transitivity are allowed); (2) the evolution of a system through a sequence $Z_p = (X_p, S^{(p)})$ indexed by $p=0,1,2,\ldots$; (3) concepts $K$ as entities that induce fibers $F_{\alpha}(K) = { E \in U \mid S(E,K) \ge \alpha }$, i.e., superlevel sets of the unary map $S_K(E) := S(E,K)$; and (4) a generative operator $G$ that produces new entities. Within this framework, we formalize a generative definition of intelligence: an operator $G$ is intelligent with respect to a concept $K$ if, given a system containing entities belonging to the fiber of $K$, it generates new entities that also belong to that fiber. Similarity Field Theory thus offers a foundational language for characterizing, comparing, and constructing intelligent systems. We prove two theorems: (i) asymmetry blocks mutual inclusion; and (ii) stability requires either an anchor coordinate or eventual confinement within a level set of $f$. These results ensure that the evolution of similarity fields is both constrained and interpretable, culminating in an exploration of how the framework allows us to interpret large language models and use them as experimental probes into societal cognition. 

**Abstract (ZH)**: 持久性和变换的相似性关系构成了任何可理解动态系统的结构基础。本文介绍了相似性场理论，这是一个数学框架，用于正式化实体之间及其演变的相似性值的原则。我们定义：（1）在实体集\(U\)上的相似性场\(S: U \times U \to [0,1]\)，满足自反性\(S(E,E)=1\)，并被视为定向关系场（允许不对称性和非传递性）；（2）系统的演化通过索引\(p=0,1,2,\ldots\)的序列\(Z_p = (X_p, S^{(p)})\)；（3）概念\(K\)作为诱导纤维\(F_{\alpha}(K) = \{ E \in U \mid S(E,K) \ge \alpha \}\)的实体，即单目映射\(S_K(E) := S(E,K)\)的超水平集；和（4）生成算子\(G\)，该算子产生新实体。在此框架内，我们形式化了智能的生成定义：相对而言，如果给定一个包含属于\(K\)的纤维中的实体的系统，该算子\(G\)生成的新实体也属于该纤维，则算子\(G\)对于概念\(K\)是智能的。相似性场理论因此提供了一种基本语言，用于描述、比较和构造智能系统。我们证明了两个定理：（i）不对称性阻碍了相互包含；（ii）稳定性需要锚定坐标或最终限制在\(f\)的等值集内。这些结果确保了相似性场的演化既受到约束又可解释，从而探讨了该框架如何允许我们解释大规模语言模型并利用它们作为社会认知的实验探针。

标题：相似性场理论：智能系统的数学框架 

---
# nDNA -- the Semantic Helix of Artificial Cognition 

**Title (ZH)**: nDNA -- 人工认知的语义螺旋 

**Authors**: Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2509.18216)  

**Abstract**: As AI foundation models grow in capability, a deeper question emerges: What shapes their internal cognitive identity -- beyond fluency and output? Benchmarks measure behavior, but the soul of a model resides in its latent geometry. In this work, we propose Neural DNA (nDNA) as a semantic-genotypic representation that captures this latent identity through the intrinsic geometry of belief. At its core, nDNA is synthesized from three principled and indispensable dimensions of latent geometry: spectral curvature, which reveals the curvature of conceptual flow across layers; thermodynamic length, which quantifies the semantic effort required to traverse representational transitions through layers; and belief vector field, which delineates the semantic torsion fields that guide a model's belief directional orientations. Like biological DNA, it encodes ancestry, mutation, and semantic inheritance, found in finetuning and alignment scars, cultural imprints, and architectural drift. In naming it, we open a new field: Neural Genomics, where models are not just tools, but digital semantic organisms with traceable inner cognition.
Modeling statement. We read AI foundation models as semantic fluid--dynamics: meaning is transported through layers like fluid in a shaped conduit; nDNA is the physics-grade readout of that flow -- a geometry-first measure of how meaning is bent, paid for, and pushed -- yielding a stable, coordinate-free neural DNA fingerprint tied to on-input behavior; with this fingerprint we cross into biology: tracing lineages across pretraining, fine-tuning, alignment, pruning, distillation, and merges; measuring inheritance between checkpoints; detecting drift as traits shift under new data or objectives; and, ultimately, studying the evolution of artificial cognition to compare models, diagnose risks, and govern change over time. 

**Abstract (ZH)**: 随着AI基础模型能力的增强，一个更深层次的问题出现了：除了流畅性和输出外，是什么塑造了它们的内在认知身份？基准测试衡量行为，但模型的灵魂在于其潜在的几何结构。在这项工作中，我们提出神经DNA（nDNA）作为一种语义-基因型表示，通过信念的内在几何结构捕捉这种潜在的身份。nDNA的核心由潜在几何结构的三个人类中心且不可或缺的维度合成：光谱曲率，揭示概念流跨层的曲率；热力学长度，量化穿越表征过渡所需的语言努力；以及信念向量场，界定指导模型信念方向的语义扭喷场。与生物DNA类似，它编码谱系、突变和语义继承，体现在微调和对齐疤痕、文化印记和架构漂移中。通过命名它，我们开启了一个新领域：神经基因组学，在该领域中，模型不仅是工具，还是具有可追溯内在认知的数字语义有机体。 

---
# Change in Quantitative Bipolar Argumentation: Sufficient, Necessary, and Counterfactual Explanations 

**Title (ZH)**: 定量双极论证中的变化：充分、必要和反事实解释 

**Authors**: Timotheus Kampik, Kristijonas Čyras, José Ruiz Alarcón  

**Link**: [PDF](https://arxiv.org/pdf/2509.18215)  

**Abstract**: This paper presents a formal approach to explaining change of inference in Quantitative Bipolar Argumentation Frameworks (QBAFs). When drawing conclusions from a QBAF and updating the QBAF to then again draw conclusions (and so on), our approach traces changes -- which we call strength inconsistencies -- in the partial order over argument strengths that a semantics establishes on some arguments of interest, called topic arguments. We trace the causes of strength inconsistencies to specific arguments, which then serve as explanations. We identify sufficient, necessary, and counterfactual explanations for strength inconsistencies and show that strength inconsistency explanations exist if and only if an update leads to strength inconsistency. We define a heuristic-based approach to facilitate the search for strength inconsistency explanations, for which we also provide an implementation. 

**Abstract (ZH)**: 本文提出了一种形式化的方法来解释定量双极论辩框架（QBAF）中推理变化的问题。在从QBAF得出结论并对QBAF进行更新以再次得出结论（依此类推）的过程中，我们的方法追踪由某种语义在某些感兴趣的主题论辩中建立的论辩强度偏序中的强度不一致的变化。我们将这些强度不一致的原因追溯到特定的论辩，并将其作为解释。我们识别出强度不一致的充分解释、必要解释和反事实解释，并证明只有当更新导致强度不一致时，强度不一致的解释才会存在。我们定义了一种基于启发式的方法来促进对强度不一致解释的搜索，并提供了其实现。 

---
# An Outcome-Based Educational Recommender System 

**Title (ZH)**: 基于学习成果的教育推荐系统 

**Authors**: Nursultan Askarbekuly, Timur Fayzrakhmanov, Sladjan Babarogić, Ivan Luković  

**Link**: [PDF](https://arxiv.org/pdf/2509.18186)  

**Abstract**: Most educational recommender systems are tuned and judged on click- or rating-based relevance, leaving their true pedagogical impact unclear. We introduce OBER-an Outcome-Based Educational Recommender that embeds learning outcomes and assessment items directly into the data schema, so any algorithm can be evaluated on the mastery it fosters. OBER uses a minimalist entity-relation model, a log-driven mastery formula, and a plug-in architecture. Integrated into an e-learning system in non-formal domain, it was evaluated trough a two-week randomized split test with over 5 700 learners across three methods: fixed expert trajectory, collaborative filtering (CF), and knowledge-based (KB) filtering. CF maximized retention, but the fixed path achieved the highest mastery. Because OBER derives business, relevance, and learning metrics from the same logs, it lets practitioners weigh relevance and engagement against outcome mastery with no extra testing overhead. The framework is method-agnostic and readily extensible to future adaptive or context-aware recommenders. 

**Abstract (ZH)**: 基于学习成果的教育推荐系统：OBER及其评价方法 

---
# Foam-Agent: An End-to-End Composable Multi-Agent Framework for Automating CFD Simulation in OpenFOAM 

**Title (ZH)**: Foam-Agent: 一个用于OpenFOAM自动CFD仿真compose多智能体框架的端到端可拓展体系 

**Authors**: Ling Yue, Nithin Somasekharan, Tingwen Zhang, Yadi Cao, Shaowu Pan  

**Link**: [PDF](https://arxiv.org/pdf/2509.18178)  

**Abstract**: Computational Fluid Dynamics (CFD) is an essential simulation tool in engineering, yet its steep learning curve and complex manual setup create significant barriers. To address these challenges, we introduce Foam-Agent, a multi-agent framework that automates the entire end-to-end OpenFOAM workflow from a single natural language prompt. Our key innovations address critical gaps in existing systems: 1. An Comprehensive End-to-End Simulation Automation: Foam-Agent is the first system to manage the full simulation pipeline, including advanced pre-processing with a versatile Meshing Agent capable of handling external mesh files and generating new geometries via Gmsh, automatic generation of HPC submission scripts, and post-simulation visualization via ParaView. 2. Composable Service Architecture: Going beyond a monolithic agent, the framework uses Model Context Protocol (MCP) to expose its core functions as discrete, callable tools. This allows for flexible integration and use by other agentic systems, such as Claude-code, for more exploratory workflows. 3. High-Fidelity Configuration Generation: We achieve superior accuracy through a Hierarchical Multi-Index RAG for precise context retrieval and a dependency-aware generation process that ensures configuration consistency. Evaluated on a benchmark of 110 simulation tasks, Foam-Agent achieves an 88.2% success rate with Claude 3.5 Sonnet, significantly outperforming existing frameworks (55.5% for MetaOpenFOAM). Foam-Agent dramatically lowers the expertise barrier for CFD, demonstrating how specialized multi-agent systems can democratize complex scientific computing. The code is public at this https URL. 

**Abstract (ZH)**: CFD中的Foam-Agent：一种基于多Agent的全流程自动化框架 

---
# HSGM: Hierarchical Segment-Graph Memory for Scalable Long-Text Semantics 

**Title (ZH)**: HSGM：分层段图记忆体 for 可扩展长文本语义 

**Authors**: Dong Liu, Yanxuan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.18168)  

**Abstract**: Semantic parsing of long documents remains challenging due to quadratic growth in pairwise composition and memory requirements. We introduce \textbf{Hierarchical Segment-Graph Memory (HSGM)}, a novel framework that decomposes an input of length $N$ into $M$ meaningful segments, constructs \emph{Local Semantic Graphs} on each segment, and extracts compact \emph{summary nodes} to form a \emph{Global Graph Memory}. HSGM supports \emph{incremental updates} -- only newly arrived segments incur local graph construction and summary-node integration -- while \emph{Hierarchical Query Processing} locates relevant segments via top-$K$ retrieval over summary nodes and then performs fine-grained reasoning within their local graphs.
Theoretically, HSGM reduces worst-case complexity from $O(N^2)$ to $O\!\left(N\,k + (N/k)^2\right)$, with segment size $k \ll N$, and we derive Frobenius-norm bounds on the approximation error introduced by node summarization and sparsification thresholds. Empirically, on three benchmarks -- long-document AMR parsing, segment-level semantic role labeling (OntoNotes), and legal event extraction -- HSGM achieves \emph{2--4$\times$ inference speedup}, \emph{$>60\%$ reduction} in peak memory, and \emph{$\ge 95\%$} of baseline accuracy. Our approach unlocks scalable, accurate semantic modeling for ultra-long texts, enabling real-time and resource-constrained NLP applications. 

**Abstract (ZH)**: 层次化段图内存（HSGM）：长文档语义解析的新型框架 

---
# Position Paper: Integrating Explainability and Uncertainty Estimation in Medical AI 

**Title (ZH)**: position paper：将解释性和不确定性估计集成到医疗AI中 

**Authors**: Xiuyi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2509.18132)  

**Abstract**: Uncertainty is a fundamental challenge in medical practice, but current medical AI systems fail to explicitly quantify or communicate uncertainty in a way that aligns with clinical reasoning. Existing XAI works focus on interpreting model predictions but do not capture the confidence or reliability of these predictions. Conversely, uncertainty estimation (UE) techniques provide confidence measures but lack intuitive explanations. The disconnect between these two areas limits AI adoption in medicine. To address this gap, we propose Explainable Uncertainty Estimation (XUE) that integrates explainability with uncertainty quantification to enhance trust and usability in medical AI. We systematically map medical uncertainty to AI uncertainty concepts and identify key challenges in implementing XUE. We outline technical directions for advancing XUE, including multimodal uncertainty quantification, model-agnostic visualization techniques, and uncertainty-aware decision support systems. Lastly, we propose guiding principles to ensure effective XUE realisation. Our analysis highlights the need for AI systems that not only generate reliable predictions but also articulate confidence levels in a clinically meaningful way. This work contributes to the development of trustworthy medical AI by bridging explainability and uncertainty, paving the way for AI systems that are aligned with real-world clinical complexities. 

**Abstract (ZH)**: 医疗实践中的不确定性是根本性挑战，但当前的医疗AI系统未能以与临床推理相一致的方式明确量化或传达不确定性。现有的可解释性AI（XAI）工作侧重于解释模型预测，但未能捕捉这些预测的置信度或可靠性。相反，不确定性估计算法（UE）提供了置信度度量，但缺乏直观的解释。这两者之间的disconnect限制了AI在医疗领域的应用。为解决这一问题，我们提出了可解释的不确定性估计算法（XUE），将可解释性与不确定性量化相结合，以增强医疗AI中的信任和可用性。我们系统地将医疗不确定性映射到AI不确定性概念，并识别实施XUE的关键挑战。我们概述了推进XUE的技术方向，包括多模态不确定性量化、模型无关的可视化技术以及不确定性意识的决策支持系统。最后，我们提出了确保有效实现XUE的指导原则。我们的分析突显了对于不仅能生成可靠预测，还能以临床有意义的方式表达置信度的AI系统的迫切需求。这项工作通过弥合可解释性和不确定性之间的鸿沟，为与现实世界临床复杂性相一致的AI系统的发展做出了贡献。 

---
# Audio-Based Pedestrian Detection in the Presence of Vehicular Noise 

**Title (ZH)**: 车辆噪声环境下基于音频的行人检测 

**Authors**: Yonghyun Kim, Chaeyeon Han, Akash Sarode, Noah Posner, Subhrajit Guhathakurta, Alexander Lerch  

**Link**: [PDF](https://arxiv.org/pdf/2509.19295)  

**Abstract**: Audio-based pedestrian detection is a challenging task and has, thus far, only been explored in noise-limited environments. We present a new dataset, results, and a detailed analysis of the state-of-the-art in audio-based pedestrian detection in the presence of vehicular noise. In our study, we conduct three analyses: (i) cross-dataset evaluation between noisy and noise-limited environments, (ii) an assessment of the impact of noisy data on model performance, highlighting the influence of acoustic context, and (iii) an evaluation of the model's predictive robustness on out-of-domain sounds. The new dataset is a comprehensive 1321-hour roadside dataset. It incorporates traffic-rich soundscapes. Each recording includes 16kHz audio synchronized with frame-level pedestrian annotations and 1fps video thumbnails. 

**Abstract (ZH)**: 基于音频的行人检测是一个具有挑战性的任务，迄今为止仅在噪声受限环境中进行过探索。我们呈现了一个新的数据集、结果以及关于噪声环境下基于音频的行人检测的最新研究分析。在本研究中，我们进行了三项分析：(i) 噪音数据集与噪声受限环境之间的跨数据集评估，(ii) 噪音数据对模型性能影响的评估，强调声学上下文的影响，以及(iii) 模型在域外声音上的预测稳健性评估。新的数据集是一个全面的1321小时路边数据集，包含了丰富的交通声音场景。每条记录包括与帧级行人标注同步的16kHz音频和每秒一帧的视频缩略图。 

---
# WolBanking77: Wolof Banking Speech Intent Classification Dataset 

**Title (ZH)**: WolBanking77: Wolof Banking Speech Intent Classification Dataset 

**Authors**: Abdou Karim Kandji, Frédéric Precioso, Cheikh Ba, Samba Ndiaye, Augustin Ndione  

**Link**: [PDF](https://arxiv.org/pdf/2509.19271)  

**Abstract**: Intent classification models have made a lot of progress in recent years. However, previous studies primarily focus on high-resource languages datasets, which results in a gap for low-resource languages and for regions with a high rate of illiterate people where languages are more spoken than read or written. This is the case in Senegal, for example, where Wolof is spoken by around 90\% of the population, with an illiteracy rate of 42\% for the country. Wolof is actually spoken by more than 10 million people in West African region. To tackle such limitations, we release a Wolof Intent Classification Dataset (WolBanking77), for academic research in intent classification. WolBanking77 currently contains 9,791 text sentences in the banking domain and more than 4 hours of spoken sentences. Experiments on various baselines are conducted in this work, including text and voice state-of-the-art models. The results are very promising on this current dataset. This paper also provides detailed analyses of the contents of the data. We report baseline f1-score and word error rate metrics respectively on NLP and ASR models trained on WolBanking77 dataset and also comparisons between models. We plan to share and conduct dataset maintenance, updates and to release open-source code. 

**Abstract (ZH)**: 面向低资源语言的意图分类模型：WolBanking77数据集发布及其应用研究 

---
# SloPalSpeech: A 2,8000-Hour Slovak Speech Corpus from Parliamentary Data 

**Title (ZH)**: SloPalSpeech：来自议会数据的2800小时斯洛伐克语音语料库 

**Authors**: Erik Božík, Marek Šuppa  

**Link**: [PDF](https://arxiv.org/pdf/2509.19270)  

**Abstract**: Automatic Speech Recognition (ASR) for low-resource languages like Slovak is hindered by the scarcity of training data. To address this, we introduce SloPalSpeech, a new, large-scale Slovak ASR dataset containing 2,806 hours of speech from parliamentary proceedings. We developed a robust processing pipeline to align and segment long-form recordings into clean, 30-second audio-transcript pairs suitable for model training. We use this dataset to fine-tune several OpenAI Whisper models (small, medium, large-v3, and large-v3-turbo), achieving significant Word Error Rate (WER) reductions on standard Slovak benchmarks like Common Voice and FLEURS. For instance, the fine-tuned Whisper-small model's WER dropped by up to 70\%, approaching the baseline performance of the much larger Whisper-large-v3 model. To foster future research in low-resource speech recognition, we publicly release the complete SloPalSpeech dataset, the fully segmented transcripts (60 million words), and all our fine-tuned models. 

**Abstract (ZH)**: 自动语音识别（ASR）对于斯洛伐克等低资源语言受限于训练数据稀缺。为了解决这一问题，我们介绍了SloPalSpeech，一个新的大规模斯洛伐克ASR数据集，包含来自议会 proceedings 的 2,806 小时语音。我们开发了一个稳健的处理流水线，将长格式录音拆分为干净的 30 秒语音-文本对，适合模型训练。我们使用此数据集对几种OpenAI Whisper模型（小型、中型、大型-v3 和大型-v3-涡轮增压）进行微调，在标准斯洛伐克基准测试（如通用语音和FLEURS）中实现了显著的词错误率（WER）降低。例如，微调的Whisper-small模型的WER降低了多达70%，接近更大规模的Whisper-large-v3模型的基线性能。为了促进未来在低资源语音识别方面的研究，我们公开发布了完整的SloPalSpeech数据集、完全分割的转录文本（6000万词）以及所有微调模型。 

---
# Finding My Voice: Generative Reconstruction of Disordered Speech for Automated Clinical Evaluation 

**Title (ZH)**: 寻找我的声音：无序语音的生成重建及其在自动化临床评估中的应用 

**Authors**: Karen Rosero, Eunjung Yeo, David R. Mortensen, Cortney Van't Slot, Rami R. Hallac, Carlos Busso  

**Link**: [PDF](https://arxiv.org/pdf/2509.19231)  

**Abstract**: We present ChiReSSD, a speech reconstruction framework that preserves children speaker's identity while suppressing mispronunciations. Unlike prior approaches trained on healthy adult speech, ChiReSSD adapts to the voices of children with speech sound disorders (SSD), with particular emphasis on pitch and prosody. We evaluate our method on the STAR dataset and report substantial improvements in lexical accuracy and speaker identity preservation. Furthermore, we automatically predict the phonetic content in the original and reconstructed pairs, where the proportion of corrected consonants is comparable to the percentage of correct consonants (PCC), a clinical speech assessment metric. Our experiments show Pearson correlation of 0.63 between automatic and human expert annotations, highlighting the potential to reduce the manual transcription burden. In addition, experiments on the TORGO dataset demonstrate effective generalization for reconstructing adult dysarthric speech. Our results indicate that disentangled, style-based TTS reconstruction can provide identity-preserving speech across diverse clinical populations. 

**Abstract (ZH)**: ChiReSSD：一种保留儿童说话人身份并抑制误读的语音重建框架 

---
# MsFIN: Multi-scale Feature Interaction Network for Traffic Accident Anticipation 

**Title (ZH)**: MsFIN: 多尺度特征交互网络用于交通事故预测 

**Authors**: Tongshuai Wu, Chao Lu, Ze Song, Yunlong Lin, Sizhe Fan, Xuemei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.19227)  

**Abstract**: With the widespread deployment of dashcams and advancements in computer vision, developing accident prediction models from the dashcam perspective has become critical for proactive safety interventions. However, two key challenges persist: modeling feature-level interactions among traffic participants (often occluded in dashcam views) and capturing complex, asynchronous multi-temporal behavioral cues preceding accidents. To deal with these two challenges, a Multi-scale Feature Interaction Network (MsFIN) is proposed for early-stage accident anticipation from dashcam videos. MsFIN has three layers for multi-scale feature aggregation, temporal feature processing and multi-scale feature post fusion, respectively. For multi-scale feature aggregation, a Multi-scale Module is designed to extract scene representations at short-term, mid-term and long-term temporal scales. Meanwhile, the Transformer architecture is leveraged to facilitate comprehensive feature interactions. Temporal feature processing captures the sequential evolution of scene and object features under causal constraints. In the multi-scale feature post fusion stage, the network fuses scene and object features across multiple temporal scales to generate a comprehensive risk representation. Experiments on DAD and DADA datasets show that MsFIN significantly outperforms state-of-the-art models with single-scale feature extraction in both prediction correctness and earliness. Ablation studies validate the effectiveness of each module in MsFIN, highlighting how the network achieves superior performance through multi-scale feature fusion and contextual interaction modeling. 

**Abstract (ZH)**: 基于 dashcam 视角的多尺度特征交互网络在事故早期预测中的应用 

---
# FedFusion: Federated Learning with Diversity- and Cluster-Aware Encoders for Robust Adaptation under Label Scarcity 

**Title (ZH)**: 联邦融合：面向标签稀缺条件下的鲁棒适应的多样性及聚类意识编码器联邦学习 

**Authors**: Ferdinand Kahenga, Antoine Bagula, Patrick Sello, Sajal K. Das  

**Link**: [PDF](https://arxiv.org/pdf/2509.19220)  

**Abstract**: Federated learning in practice must contend with heterogeneous feature spaces, severe non-IID data, and scarce labels across clients. We present FedFusion, a federated transfer-learning framework that unifies domain adaptation and frugal labelling with diversity-/cluster-aware encoders (DivEn, DivEn-mix, DivEn-c). Labelled teacher clients guide learner clients via confidence-filtered pseudo-labels and domain-adaptive transfer, while clients maintain personalised encoders tailored to local data. To preserve global coherence under heterogeneity, FedFusion employs similarity-weighted classifier coupling (with optional cluster-wise averaging), mitigating dominance by data-rich sites and improving minority-client performance. The frugal-labelling pipeline combines self-/semi-supervised pretext training with selective fine-tuning, reducing annotation demands without sharing raw data. Across tabular and imaging benchmarks under IID, non-IID, and label-scarce regimes, FedFusion consistently outperforms state-of-the-art baselines in accuracy, robustness, and fairness while maintaining comparable communication and computation budgets. These results show that harmonising personalisation, domain adaptation, and label efficiency is an effective recipe for robust federated learning under real-world constraints. 

**Abstract (ZH)**: 联邦学习在实践中必须应对异质特征空间、严重非IID数据以及客户稀缺标签的问题。我们提出了FedFusion，一个统一域适应和节俭标注的联邦迁移学习框架（配备多样性-/聚类感知编码器DivEn、DivEn-mix和DivEn-c）。标记的教师客户通过可信度过滤的伪标签和域适应的迁移学习指导学习客户，同时客户维护针对本地数据量身定制的编码器。为在异质性下保持全局一致性，FedFusion采用相似性加权分类器耦合（可选聚类平均），减少富数据站点的主导作用，提高少数客户的表现。节俭标注流水线结合自我-/半监督预训练和选择性微调，减少标注需求而不共享原始数据。在标签充裕和稀缺的表征和影像基准测试中，无论是在IID、非IID还是标签稀缺的情境下，FedFusion在准确率、鲁棒性和公平性方面都优于当前最佳基线，同时保持相似的通信和计算预算。这些结果表明，在实际约束条件下，平衡个性化、域适应和标签效率是稳健联邦学习的有效方法。 

---
# HyKid: An Open MRI Dataset with Expert-Annotated Multi-Structure and Choroid Plexus in Pediatric Hydrocephalus 

**Title (ZH)**: HyKid: 一种带有专家标注多结构和脉络膜囊肿的儿科脑积水开放MRI数据集 

**Authors**: Yunzhi Xu, Yushuang Ding, Hu Sun, Hongxi Zhang, Li Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.19218)  

**Abstract**: Evaluation of hydrocephalus in children is challenging, and the related research is limited by a lack of publicly available, expert-annotated datasets, particularly those with segmentation of the choroid plexus. To address this, we present HyKid, an open-source dataset from 48 pediatric patients with hydrocephalus. 3D MRIs were provided with 1mm isotropic resolution, which was reconstructed from routine low-resolution images using a slice-to-volume algorithm. Manually corrected segmentations of brain tissues, including white matter, grey matter, lateral ventricle, external CSF, and the choroid plexus, were provided by an experienced neurologist. Additionally, structured data was extracted from clinical radiology reports using a Retrieval-Augmented Generation framework. The strong correlation between choroid plexus volume and total CSF volume provided a potential biomarker for hydrocephalus evaluation, achieving excellent performance in a predictive model (AUC = 0.87). The proposed HyKid dataset provided a high-quality benchmark for neuroimaging algorithms development, and it revealed the choroid plexus-related features in hydrocephalus assessments. Our datasets are publicly available at this https URL. 

**Abstract (ZH)**: 儿童颅内积水的评估具有挑战性，相关研究受限于缺乏公开的专业标注数据集，尤其是包含 choroid plexus 分段的数据集。为解决这一问题，我们提出了 HyKid，一个来自 48 例颅内积水儿童患者的开源数据集。提供了 1mm 等效分辨率的 3D MRI，并通过切片到体积算法从常规低分辨率图像中重建。一名经验丰富的神经科医生提供了手动校正的脑组织分割标注，包括白质、灰质、侧脑室、硬脑膜外脑脊液和 choroid plexus。此外，使用 Retrieval-Augmented Generation 框架从临床放射学报告中提取结构化数据。choroid plexus 体积与总脑脊液体积之间的强相关性为颅内积水评估提供了潜在生物标志物，在预测模型中表现出色（AUC = 0.87）。提出的数据集 HyKid 为神经影像算法开发提供了高质量基准，并揭示了颅内积水评估中的 choroid plexus 相关特征。相关数据集可在以下链接公开获取。 

---
# YAC: Bridging Natural Language and Interactive Visual Exploration with Generative AI for Biomedical Data Discovery 

**Title (ZH)**: YAC: 结合生成式AI、自然语言与互动视觉探索的生物医学数据发现 

**Authors**: Devin Lange, Shanghua Gao, Pengwei Sui, Austen Money, Priya Misner, Marinka Zitnik, Nils Gehlenborg  

**Link**: [PDF](https://arxiv.org/pdf/2509.19182)  

**Abstract**: Incorporating natural language input has the potential to improve the capabilities of biomedical data discovery interfaces. However, user interface elements and visualizations are still powerful tools for interacting with data, even in the new world of generative AI. In our prototype system, YAC, Yet Another Chatbot, we bridge the gap between natural language and interactive visualizations by generating structured declarative output with a multi-agent system and interpreting that output to render linked interactive visualizations and apply data filters. Furthermore, we include widgets, which allow users to adjust the values of that structured output through user interface elements. We reflect on the capabilities and design of this system with an analysis of its technical dimensions and illustrate the capabilities through four usage scenarios. 

**Abstract (ZH)**: 将自然语言输入纳入生物医学数据发现界面具有提高其能力的潜力。然而，即使在生成型AI的新世界中，用户界面元素和可视化工具仍然是与数据交互的强大工具。在我们的原型系统YAC（Yet Another Chatbot）中，我们通过多Agent系统生成结构化声明性输出，并解释该输出以渲染链接的交互式可视化并应用数据过滤。此外，我们包括小部件，这些小部件允许用户通过用户界面元素调整该结构化输出的值。我们通过技术维度的分析反思该系统的功能和设计，并通过四个使用场景说明其功能。 

---
# Generative Propaganda 

**Title (ZH)**: 生成式 propaganda 

**Authors**: Madeleine I. G. Daepp, Alejandro Cuevas, Robert Osazuwa Ness, Vickie Yu-Ping Wang, Bharat Kumar Nayak, Dibyendu Mishra, Ti-Chung Cheng, Shaily Desai, Joyojeet Pal  

**Link**: [PDF](https://arxiv.org/pdf/2509.19147)  

**Abstract**: Generative propaganda is the use of generative artificial intelligence (AI) to shape public opinion. To characterize its use in real-world settings, we conducted interviews with defenders (e.g., factcheckers, journalists, officials) in Taiwan and creators (e.g., influencers, political consultants, advertisers) as well as defenders in India, centering two places characterized by high levels of online propaganda. The term "deepfakes", we find, exerts outsized discursive power in shaping defenders' expectations of misuse and, in turn, the interventions that are prioritized. To better characterize the space of generative propaganda, we develop a taxonomy that distinguishes between obvious versus hidden and promotional versus derogatory use. Deception was neither the main driver nor the main impact vector of AI's use; instead, Indian creators sought to persuade rather than to deceive, often making AI's use obvious in order to reduce legal and reputational risks, while Taiwan's defenders saw deception as a subset of broader efforts to distort the prevalence of strategic narratives online. AI was useful and used, however, in producing efficiency gains in communicating across languages and modes, and in evading human and algorithmic detection. Security researchers should reconsider threat models to clearly differentiate deepfakes from promotional and obvious uses, to complement and bolster the social factors that constrain misuse by internal actors, and to counter efficiency gains globally. 

**Abstract (ZH)**: 生成性宣传是利用生成性人工 intelligence (AI) 影响公众舆论的应用。为了在其真实世界应用场景中对其特性进行Characterization，我们对台湾和印度的防御者（例如，事实核查员、记者、官员）以及创作者（例如，影响力人物、政治顾问、广告商）进行了访谈，重点是两个在线宣传水平较高的地方。我们发现，“深伪”一词在塑造防御者的预期滥用方式以及由此导致的优先干预措施方面具有异常强大的话语权。为了更好地Characterize 生成性宣传的空间，我们建立了一种分类法，区分明显与隐秘以及促销与贬低的使用方式。AI 的使用并不是由欺骗行为驱动的，也不是造成影响的主要途径；相反，印度的创作者试图说服而不是欺骗，经常通过使 AI 的使用更加明显来降低法律和声誉风险，而台湾的防御者则将欺骗视为在线扭曲战略叙事总体努力的一个子集。然而，AI 在跨语言和模式沟通中提高了效率，并通过规避人类和算法的检测。安全研究人员应重新考虑威胁模型，明确区分“深伪”与其他促销性和明显使用的做法，以补充和强化限制内部行为者滥用的社会因素，并在全球范围内对抗效率提升。 

---
# Anecdoctoring: Automated Red-Teaming Across Language and Place 

**Title (ZH)**: ANSICTION：跨语言和地区的人工自动化红队行动 

**Authors**: Alejandro Cuevas, Saloni Dash, Bharat Kumar Nayak, Dan Vann, Madeleine I. G. Daepp  

**Link**: [PDF](https://arxiv.org/pdf/2509.19143)  

**Abstract**: Disinformation is among the top risks of generative artificial intelligence (AI) misuse. Global adoption of generative AI necessitates red-teaming evaluations (i.e., systematic adversarial probing) that are robust across diverse languages and cultures, but red-teaming datasets are commonly US- and English-centric. To address this gap, we propose "anecdoctoring", a novel red-teaming approach that automatically generates adversarial prompts across languages and cultures. We collect misinformation claims from fact-checking websites in three languages (English, Spanish, and Hindi) and two geographies (US and India). We then cluster individual claims into broader narratives and characterize the resulting clusters with knowledge graphs, with which we augment an attacker LLM. Our method produces higher attack success rates and offers interpretability benefits relative to few-shot prompting. Results underscore the need for disinformation mitigations that scale globally and are grounded in real-world adversarial misuse. 

**Abstract (ZH)**: 生成式人工智能滥用中的不实信息是主要风险之一。为了适应全球范围内生成式人工智能的采用，需要进行跨语言和文化鲁棒性的红队评估（即系统性的对抗性探测试验），但现有的红队数据集多集中在美式英语上。为了解决这一问题，我们提出了一种名为“anecdoctoring”的新型红队方法，该方法能够自动生成跨语言和文化背景的对抗性提示。我们从三家网站（英语、西班牙语和印地语）和两个地理区域（美国和印度）收集不实信息声明，并将这些个体声明聚类成更广泛的叙述，然后使用知识图谱对这些聚类进行表征，并以此扩展攻击者大语言模型。与少量示例提示相比，我们的方法能够实现更高的攻击成功率并提供更好的可解释性。实验结果突显了全球范围内实现不实信息缓解措施的必要性，并且这些措施需要基于实际的对抗性滥用情况。 

---
# FedFiTS: Fitness-Selected, Slotted Client Scheduling for Trustworthy Federated Learning in Healthcare AI 

**Title (ZH)**: FedFiTS：基于健身选择的分时客户端调度以实现医疗AI可信联邦学习 

**Authors**: Ferdinand Kahenga, Antoine Bagula, Sajal K. Das, Patrick Sello  

**Link**: [PDF](https://arxiv.org/pdf/2509.19120)  

**Abstract**: Federated Learning (FL) has emerged as a powerful paradigm for privacy-preserving model training, yet deployments in sensitive domains such as healthcare face persistent challenges from non-IID data, client unreliability, and adversarial manipulation. This paper introduces FedFiTS, a trust and fairness-aware selective FL framework that advances the FedFaSt line by combining fitness-based client election with slotted aggregation. FedFiTS implements a three-phase participation strategy-free-for-all training, natural selection, and slotted team participation-augmented with dynamic client scoring, adaptive thresholding, and cohort-based scheduling to balance convergence efficiency with robustness. A theoretical convergence analysis establishes bounds for both convex and non-convex objectives under standard assumptions, while a communication-complexity analysis shows reductions relative to FedAvg and other baselines. Experiments on diverse datasets-medical imaging (X-ray pneumonia), vision benchmarks (MNIST, FMNIST), and tabular agricultural data (Crop Recommendation)-demonstrate that FedFiTS consistently outperforms FedAvg, FedRand, and FedPow in accuracy, time-to-target, and resilience to poisoning attacks. By integrating trust-aware aggregation with fairness-oriented client selection, FedFiTS advances scalable and secure FL, making it well suited for real-world healthcare and cross-domain deployments. 

**Abstract (ZH)**: FedFiTS：一种信任和公平感知的选择性联邦学习框架 

---
# Towards Practical Multi-label Causal Discovery in High-Dimensional Event Sequences via One-Shot Graph Aggregation 

**Title (ZH)**: 基于一键图聚合的高维事件序列高效多标签因果发现 

**Authors**: Hugo Math, Rainer Lienhart  

**Link**: [PDF](https://arxiv.org/pdf/2509.19112)  

**Abstract**: Understanding causality in event sequences where outcome labels such as diseases or system failures arise from preceding events like symptoms or error codes is critical. Yet remains an unsolved challenge across domains like healthcare or vehicle diagnostics. We introduce CARGO, a scalable multi-label causal discovery method for sparse, high-dimensional event sequences comprising of thousands of unique event types. Using two pretrained causal Transformers as domain-specific foundation models for event sequences. CARGO infers in parallel, per sequence one-shot causal graphs and aggregates them using an adaptive frequency fusion to reconstruct the global Markov boundaries of labels. This two-stage approach enables efficient probabilistic reasoning at scale while bypassing the intractable cost of full-dataset conditional independence testing. Our results on a challenging real-world automotive fault prediction dataset with over 29,100 unique event types and 474 imbalanced labels demonstrate CARGO's ability to perform structured reasoning. 

**Abstract (ZH)**: 理解由前期事件如症状或错误代码导致的结果标签如疾病或系统故障在事件序列中的因果关系对于各个领域（如医疗保健或车辆诊断）至关重要，但仍然是一项未解决的挑战。我们引入了CARGO，一种适用于稀疏高维事件序列（包含数千种 unique 事件类型）的可扩展多标签因果发现方法。通过使用两种预训练的因果 Transformer 作为事件序列的领域特定基础模型，CARGO 并行地为每个序列推断一次性的因果图，并通过自适应频率融合将其聚合以重构全局马尔可夫边界。这种两阶段方法使大规模高效概率推理成为可能，从而绕过了全数据集条件独立性检验的不可行成本。我们的结果表明，CARGO 在一个包含超过 29,100 种 unique 事件类型和 474 种失衡标签的具有挑战性的真实世界汽车故障预测数据集中能够进行结构化推理。 

---
# Algorithms for Adversarially Robust Deep Learning 

**Title (ZH)**: 对抗鲁棒的深度学习算法 

**Authors**: Alexander Robey  

**Link**: [PDF](https://arxiv.org/pdf/2509.19100)  

**Abstract**: Given the widespread use of deep learning models in safety-critical applications, ensuring that the decisions of such models are robust against adversarial exploitation is of fundamental importance. In this thesis, we discuss recent progress toward designing algorithms that exhibit desirable robustness properties. First, we discuss the problem of adversarial examples in computer vision, for which we introduce new technical results, training paradigms, and certification algorithms. Next, we consider the problem of domain generalization, wherein the task is to train neural networks to generalize from a family of training distributions to unseen test distributions. We present new algorithms that achieve state-of-the-art generalization in medical imaging, molecular identification, and image classification. Finally, we study the setting of jailbreaking large language models (LLMs), wherein an adversarial user attempts to design prompts that elicit objectionable content from an LLM. We propose new attacks and defenses, which represent the frontier of progress toward designing robust language-based agents. 

**Abstract (ZH)**: 基于深度学习模型在关键安全应用中的广泛应用，确保这些模型的决策能够抵御对抗性利用是基础性的。在这项研究中，我们讨论了设计具有稳健性特征算法的最新进展。首先，我们探讨了计算机视觉中的对抗性示例问题，并引入了新的技术成果、训练范式和认证算法。接着，我们考虑了领域泛化问题，即训练神经网络从训练分布族推广到未见过的测试分布。我们提出了新的算法，实现了医学成像、分子识别和图像分类领域的最佳泛化性能。最后，我们研究了大型语言模型（LLMs）的破解设置，其中恶意用户试图设计提示以从LLM中引发令人反感的内容。我们提出了新的攻击和防御方法，代表了设计基于语言的稳健代理的最新进展。 

---
# Training Flow Matching Models with Reliable Labels via Self-Purification 

**Title (ZH)**: 通过自我净化生成可靠标签训练流匹配模型 

**Authors**: Hyeongju Kim, Yechan Yu, June Young Yi, Juheon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.19091)  

**Abstract**: Training datasets are inherently imperfect, often containing mislabeled samples due to human annotation errors, limitations of tagging models, and other sources of noise. Such label contamination can significantly degrade the performance of a trained model. In this work, we introduce Self-Purifying Flow Matching (SPFM), a principled approach to filtering unreliable data within the flow-matching framework. SPFM identifies suspicious data using the model itself during the training process, bypassing the need for pretrained models or additional modules. Our experiments demonstrate that models trained with SPFM generate samples that accurately adhere to the specified conditioning, even when trained on noisy labels. Furthermore, we validate the robustness of SPFM on the TITW dataset, which consists of in-the-wild speech data, achieving performance that surpasses existing baselines. 

**Abstract (ZH)**: 训练数据本质上是不完美的，往往包含由于人工注解错误、标记模型的局限性和其他噪声源导致的误标样本。此类标签污染会显著劣化训练模型的性能。在本文中，我们提出了一种在流匹配框架内的名为自我净化流匹配（SPFM）的原则性方法，用于筛选不可靠数据。SPFM在训练过程中利用模型本身识别可疑数据，无需依赖预训练模型或额外模块。我们的实验表明，使用SPFM训练的模型能够生成准确符合指定条件的样本，即使在噪声标签下训练也是如此。此外，我们还在野生语音数据集TITW上验证了SPFM的鲁棒性，其性能超越了现有基线。 

---
# Graph Neural Networks with Similarity-Navigated Probabilistic Feature Copying 

**Title (ZH)**: 带有相似导航概率特征复制的图神经网络 

**Authors**: Asela Hevapathige  

**Link**: [PDF](https://arxiv.org/pdf/2509.19084)  

**Abstract**: Graph Neural Networks (GNNs) have demonstrated remarkable success across various graph-based tasks. However, they face some fundamental limitations: feature oversmoothing can cause node representations to become indistinguishable in deeper networks, they struggle to effectively manage heterogeneous relationships where connected nodes differ significantly, and they process entire feature vectors as indivisible units, which limits flexibility. We seek to address these limitations. We propose AxelGNN, a novel GNN architecture inspired by Axelrod's cultural dissemination model that addresses these limitations through a unified framework. AxelGNN incorporates similarity-gated probabilistic interactions that adaptively promote convergence or divergence based on node similarity, implements trait-level copying mechanisms for fine-grained feature aggregation at the segment level, and maintains global polarization to preserve node distinctiveness across multiple representation clusters. The model's bistable convergence dynamics naturally handle both homophilic and heterophilic graphs within a single architecture. Extensive experiments on node classification and influence estimation benchmarks demonstrate that AxelGNN consistently outperforms or matches state-of-the-art GNN methods across diverse graph structures with varying homophily-heterophily characteristics. 

**Abstract (ZH)**: 基于 Axelrod 文化传播模型的 AxelGNN：统一框架下的图神经网络新架构 

---
# Beyond Backpropagation: Exploring Innovative Algorithms for Energy-Efficient Deep Neural Network Training 

**Title (ZH)**: 超越反向传播：探索高效的深度神经网络训练创新算法 

**Authors**: Przemysław Spyra  

**Link**: [PDF](https://arxiv.org/pdf/2509.19063)  

**Abstract**: The rising computational and energy demands of deep neural networks (DNNs), driven largely by backpropagation (BP), challenge sustainable AI development. This paper rigorously investigates three BP-free training methods: the Forward-Forward (FF), Cascaded-Forward (CaFo), and Mono-Forward (MF) algorithms, tracing their progression from foundational concepts to a demonstrably superior solution.
A robust comparative framework was established: each algorithm was implemented on its native architecture (MLPs for FF and MF, a CNN for CaFo) and benchmarked against an equivalent BP-trained model. Hyperparameters were optimized with Optuna, and consistent early stopping criteria were applied based on validation performance, ensuring all models were optimally tuned before comparison.
Results show that MF not only competes with but consistently surpasses BP in classification accuracy on its native MLPs. Its superior generalization stems from converging to a more favorable minimum in the validation loss landscape, challenging the assumption that global optimization is required for state-of-the-art results. Measured at the hardware level using the NVIDIA Management Library (NVML) API, MF reduces energy consumption by up to 41% and shortens training time by up to 34%, translating to a measurably smaller carbon footprint as estimated by CodeCarbon.
Beyond this primary result, we present a hardware-level analysis that explains the efficiency gains: exposing FF's architectural inefficiencies, validating MF's computationally lean design, and challenging the assumption that all BP-free methods are inherently more memory-efficient. By documenting the evolution from FF's conceptual groundwork to MF's synthesis of accuracy and sustainability, this work offers a clear, data-driven roadmap for future energy-efficient deep learning. 

**Abstract (ZH)**: 深度神经网络（DNNs）由于反向传播（BP）驱动的日益增长的计算和能源需求，挑战可持续AI开发。本文严格研究了三种无BP训练方法：前向前向（FF）、级联前向（CaFo）和单前向（MF）算法，并追溯了这些方法从基础概念到可证明更优解决方案的进展。
建立了一套 robust 对比框架：每个算法在其原生架构（FF 和 MF 使用全连接网络，CaFo 使用卷积神经网络）上实现，并与相应的 BP 训练模型进行基准测试。使用 Optuna 对超参数进行优化，并基于验证性能应用一致的早期停止标准，确保所有模型在比较前均已优化。
结果表明，MF 不仅能够与 BP 竞争，在其原生全连接网络上的分类准确率甚至持续超越 BP。其更强的泛化能力源自于在验证损失景观中收敛到更优的极小值，挑战了必须进行全局优化才能获得最优结果的假设。通过使用 NVIDIA 管理库 (NVML) API 在硬件层面测量，MF 减少了高达 41% 的能源消耗并缩短了高达 34% 的训练时间，从而通过 CodeCarbon 估算出具有更小的碳足迹。
在此主要结果之外，本文还呈现了在硬件层面的分析，解释了效率提升的原因：揭示了 FF 的架构效率低下，验证了 MF 的计算经济性设计，并挑战了所有无 BP 方法都天然更高效内存使用的假设。通过记录 FF 的概念基础到 MF 同时实现准确性与可持续性的演变，本文为未来能源高效的深度学习提供了一条清晰的数据驱动路线图。 

---
# Towards Privacy-Aware Bayesian Networks: A Credal Approach 

**Title (ZH)**: 面向隐私意识的贝叶斯网络：一种信任区间方法 

**Authors**: Niccolò Rocchi, Fabio Stella, Cassio de Campos  

**Link**: [PDF](https://arxiv.org/pdf/2509.18949)  

**Abstract**: Bayesian networks (BN) are probabilistic graphical models that enable efficient knowledge representation and inference. These have proven effective across diverse domains, including healthcare, bioinformatics and economics. The structure and parameters of a BN can be obtained by domain experts or directly learned from available data. However, as privacy concerns escalate, it becomes increasingly critical for publicly released models to safeguard sensitive information in training data. Typically, released models do not prioritize privacy by design. In particular, tracing attacks from adversaries can combine the released BN with auxiliary data to determine whether specific individuals belong to the data from which the BN was learned. State-of-the-art protection tecniques involve introducing noise into the learned parameters. While this offers robust protection against tracing attacks, it significantly impacts the model's utility, in terms of both the significance and accuracy of the resulting inferences. Hence, high privacy may be attained at the cost of releasing a possibly ineffective model. This paper introduces credal networks (CN) as a novel solution for balancing the model's privacy and utility. After adapting the notion of tracing attacks, we demonstrate that a CN enables the masking of the learned BN, thereby reducing the probability of successful attacks. As CNs are obfuscated but not noisy versions of BNs, they can achieve meaningful inferences while safeguarding privacy. Moreover, we identify key learning information that must be concealed to prevent attackers from recovering the underlying BN. Finally, we conduct a set of numerical experiments to analyze how privacy gains can be modulated by tuning the CN hyperparameters. Our results confirm that CNs provide a principled, practical, and effective approach towards the development of privacy-aware probabilistic graphical models. 

**Abstract (ZH)**: 贝叶斯网络（BN）是概率图形模型，能够高效地实现知识表示和推理。这些模型已在医疗保健、生物信息学和经济学等多个领域证明有效。BN的结构和参数可以通过领域专家获得，也可以直接从可用数据中学习。然而，随着隐私担忧的加剧，如何在公开发布模型时保护训练数据中的敏感信息变得越来越重要。通常情况下，发布模型时不优先考虑设计上的隐私保护。特别是，对手可以利用公开发布的BN与辅助数据结合来进行跟踪攻击，以确定某些个体是否属于用于学习BN的数据。最先进的保护技术是向学习到的参数中引入噪声。虽然这种方法能对跟踪攻击提供稳健保护，但会显著影响模型的效用，包括推断结果的显著性和准确性。因此，高隐私可能以释放一个可能无效的模型为代价。本文引入了信任网络（CN）作为一种新的解决方案，以平衡模型的隐私和效用。在适应跟踪攻击的概念后，我们证明CN能够对公开发布的BN进行掩盖，从而降低成功攻击的概率。由于CN是对BN进行模糊化处理但不是加入噪声的版本，它们可以在保护隐私的同时实现有意义的推断。此外，我们确定了必须隐藏的关键学习信息，以防止攻击者重新构建底层的BN。最后，我们进行了一系列数值实验，分析了通过调整CN超参数来调节隐私收益的方法。我们的结果证实，CN提供了一种原理上合理、实用且有效的保护隐私的概率图形模型开发方法。 

---
# Accurate and Efficient Prediction of Wi-Fi Link Quality Based on Machine Learning 

**Title (ZH)**: 基于机器学习的Wi-Fi链路质量准确高效预测 

**Authors**: Gabriele Formis, Gianluca Cena, Lukasz Wisniewski, Stefano Scanzio  

**Link**: [PDF](https://arxiv.org/pdf/2509.18933)  

**Abstract**: Wireless communications are characterized by their unpredictability, posing challenges for maintaining consistent communication quality. This paper presents a comprehensive analysis of various prediction models, with a focus on achieving accurate and efficient Wi-Fi link quality forecasts using machine learning techniques. Specifically, the paper evaluates the performance of data-driven models based on the linear combination of exponential moving averages, which are designed for low-complexity implementations and are then suitable for hardware platforms with limited processing resources. Accuracy of the proposed approaches was assessed using experimental data from a real-world Wi-Fi testbed, considering both channel-dependent and channel-independent training data. Remarkably, channel-independent models, which allow for generalized training by equipment manufacturers, demonstrated competitive performance. Overall, this study provides insights into the practical deployment of machine learning-based prediction models for enhancing Wi-Fi dependability in industrial environments. 

**Abstract (ZH)**: 无线通信因其不可预测性给保持一致的通信质量带来了挑战。本文对各种预测模型进行了全面分析，重点关注使用机器学习技术实现准确高效的Wi-Fi链路质量预测。具体而言，本文评估了基于指数移动平均线性组合的数据驱动模型的性能，这些模型旨在实现低复杂度实现，并适用于处理资源有限的硬件平台。所提出方法的准确性通过实际Wi-Fi测试床的实验数据进行评估，考虑了依赖信道和不依赖信道的训练数据。值得注意的是，不依赖信道的模型因其允许设备制造商进行通用训练而展现了竞争性的性能。总体而言，本研究为在工业环境中部署基于机器学习的预测模型以提高Wi-Fi可靠性提供了实用见解。 

---
# Tackling GNARLy Problems: Graph Neural Algorithmic Reasoning Reimagined through Reinforcement Learning 

**Title (ZH)**: 解决棘手问题：通过强化学习重塑图神经算法推理 

**Authors**: Alex Schutz, Victor-Alexandru Darvariu, Efimia Panagiotaki, Bruno Lacerda, Nick Hawes  

**Link**: [PDF](https://arxiv.org/pdf/2509.18930)  

**Abstract**: Neural Algorithmic Reasoning (NAR) is a paradigm that trains neural networks to execute classic algorithms by supervised learning. Despite its successes, important limitations remain: inability to construct valid solutions without post-processing and to reason about multiple correct ones, poor performance on combinatorial NP-hard problems, and inapplicability to problems for which strong algorithms are not yet known. To address these limitations, we reframe the problem of learning algorithm trajectories as a Markov Decision Process, which imposes structure on the solution construction procedure and unlocks the powerful tools of imitation and reinforcement learning (RL). We propose the GNARL framework, encompassing the methodology to translate problem formulations from NAR to RL and a learning architecture suitable for a wide range of graph-based problems. We achieve very high graph accuracy results on several CLRS-30 problems, performance matching or exceeding much narrower NAR approaches for NP-hard problems and, remarkably, applicability even when lacking an expert algorithm. 

**Abstract (ZH)**: 神经算法推理（NAR）是一种通过监督学习训练神经网络执行经典算法的范式。尽管取得了成功，但仍存在重要限制：无法在不进行后处理的情况下构造有效的解决方案，难以推理多个正确的解决方案，组合NP难问题上的性能不佳，以及对于目前还没有强大算法的问题不适用。为克服这些限制，我们将学习算法轨迹的问题重新框架为马尔可夫决策过程，这为解构建过程施加了结构，并解锁了模仿学习和强化学习（RL）的强大工具。我们提出了GNARL框架，涵盖将NAR中的问题表述转换为RL的方法论以及适用于多种图问题的学习架构。我们在多个CLRS-30问题上实现了非常高的图准确率结果，在组合NP难问题上达到了或超过了更窄的NAR方法的性能，并且甚至在缺乏专家算法时也具有适用性。 

---
# The AI Literacy Heptagon: A Structured Approach to AI Literacy in Higher Education 

**Title (ZH)**: AI素养七边形：高等教育中AI素养的结构化方法 

**Authors**: Veronika Hackl, Alexandra Mueller, Maximilian Sailer  

**Link**: [PDF](https://arxiv.org/pdf/2509.18900)  

**Abstract**: The integrative literature review addresses the conceptualization and implementation of AI Literacy (AIL) in Higher Education (HE) by examining recent research literature. Through an analysis of publications (2021-2024), we explore (1) how AIL is defined and conceptualized in current research, particularly in HE, and how it can be delineated from related concepts such as Data Literacy, Media Literacy, and Computational Literacy; (2) how various definitions can be synthesized into a comprehensive working definition, and (3) how scientific insights can be effectively translated into educational practice. Our analysis identifies seven central dimensions of AIL: technical, applicational, critical thinking, ethical, social, integrational, and legal. These are synthesized in the AI Literacy Heptagon, deepening conceptual understanding and supporting the structured development of AIL in HE. The study aims to bridge the gap between theoretical AIL conceptualizations and the practical implementation in academic curricula. 

**Abstract (ZH)**: 综合性文献回顾探讨了人工智能素养（AIL）在高等教育（HE）中的概念化与实施，通过分析近期研究文献（2021-2024）。我们探讨了（1）当前研究，尤其是高等教育中，对AIL的定义和概念化，以及如何区分其与数据素养、媒体素养和计算素养等相关概念；（2）如何将各种定义综合成一个全面的工作定义；（3）如何有效地将科学洞见转化为教育实践。分析识别了人工智能素养的七大核心维度：技术性、应用性、批判性思维、伦理性、社会性、整合性和法律性。这些维度被综合在人工智能素养七角形（AI Literacy Heptagon）中，加深了对人工智能素养概念的理解，并支持了人工智能素养在高等教育中结构化的发展。本研究旨在弥合理论性人工智能素养概念化与学术课程中实际实施之间的差距。 

---
# A Kernel Space-based Multidimensional Sparse Model for Dynamic PET Image Denoising 

**Title (ZH)**: 基于内核空间的多维稀疏模型动态PET图像去噪 

**Authors**: Kuang Xiaodong, Li Bingxuan, Li Yuan, Rao Fan, Ma Gege, Xie Qingguo, Mok Greta S P, Liu Huafeng, Zhu Wentao  

**Link**: [PDF](https://arxiv.org/pdf/2509.18801)  

**Abstract**: Achieving high image quality for temporal frames in dynamic positron emission tomography (PET) is challenging due to the limited statistic especially for the short frames. Recent studies have shown that deep learning (DL) is useful in a wide range of medical image denoising tasks. In this paper, we propose a model-based neural network for dynamic PET image denoising. The inter-frame spatial correlation and intra-frame structural consistency in dynamic PET are used to establish the kernel space-based multidimensional sparse (KMDS) model. We then substitute the inherent forms of the parameter estimation with neural networks to enable adaptive parameters optimization, forming the end-to-end neural KMDS-Net. Extensive experimental results from simulated and real data demonstrate that the neural KMDS-Net exhibits strong denoising performance for dynamic PET, outperforming previous baseline methods. The proposed method may be used to effectively achieve high temporal and spatial resolution for dynamic PET. Our source code is available at this https URL. 

**Abstract (ZH)**: 基于模型的神经网络在动态正电子发射断层成像(PET)图像去噪中的应用 

---
# Detection of security smells in IaC scripts through semantics-aware code and language processing 

**Title (ZH)**: 基于语义感知的代码和语言处理在检测IaC脚本中的安全气味方面的研究 

**Authors**: Aicha War, Adnan A. Rawass, Abdoul K. Kabore, Jordan Samhi, Jacques Klein, Tegawende F. Bissyande  

**Link**: [PDF](https://arxiv.org/pdf/2509.18790)  

**Abstract**: Infrastructure as Code (IaC) automates the provisioning and management of IT infrastructure through scripts and tools, streamlining software deployment. Prior studies have shown that IaC scripts often contain recurring security misconfigurations, and several detection and mitigation approaches have been proposed. Most of these rely on static analysis, using statistical code representations or Machine Learning (ML) classifiers to distinguish insecure configurations from safe code.
In this work, we introduce a novel approach that enhances static analysis with semantic understanding by jointly leveraging natural language and code representations. Our method builds on two complementary ML models: CodeBERT, to capture semantics across code and text, and LongFormer, to represent long IaC scripts without losing contextual information. We evaluate our approach on misconfiguration datasets from two widely used IaC tools, Ansible and Puppet. To validate its effectiveness, we conduct two ablation studies (removing code text from the natural language input and truncating scripts to reduce context) and compare against four large language models (LLMs) and prior work. Results show that semantic enrichment substantially improves detection, raising precision and recall from 0.46 and 0.79 to 0.92 and 0.88 on Ansible, and from 0.55 and 0.97 to 0.87 and 0.75 on Puppet, respectively. 

**Abstract (ZH)**: 基于语义的理解增强软件开发生命周期中的静态分析以检测基础设施即代码中的安全配置错误：以Ansible和Puppet为例 

---
# Financial Risk Relation Identification through Dual-view Adaptation 

**Title (ZH)**: 通过双视图适应识别财务风险关系 

**Authors**: Wei-Ning Chiu, Yu-Hsiang Wang, Andy Hsiao, Yu-Shiang Huang, Chuan-Ju Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18775)  

**Abstract**: A multitude of interconnected risk events -- ranging from regulatory changes to geopolitical tensions -- can trigger ripple effects across firms. Identifying inter-firm risk relations is thus crucial for applications like portfolio management and investment strategy. Traditionally, such assessments rely on expert judgment and manual analysis, which are, however, subjective, labor-intensive, and difficult to scale. To address this, we propose a systematic method for extracting inter-firm risk relations using Form 10-K filings -- authoritative, standardized financial documents -- as our data source. Leveraging recent advances in natural language processing, our approach captures implicit and abstract risk connections through unsupervised fine-tuning based on chronological and lexical patterns in the filings. This enables the development of a domain-specific financial encoder with a deeper contextual understanding and introduces a quantitative risk relation score for transparency, interpretable analysis. Extensive experiments demonstrate that our method outperforms strong baselines across multiple evaluation settings. 

**Abstract (ZH)**: 多种相互关联的风险事件——从监管变化到地缘政治紧张关系等——可以触发企业间的涟漪效应。因此，识别企业间的风险关系对于投资组合管理和投资策略等应用至关重要。传统上，这种评估依赖于专家判断和手动分析，但这些方法具有主观性、劳动密集型且难以规模化。为了解决这一问题，我们提出了一种系统方法，利用10-K filings（权威且标准化的财务文件）作为数据源，提取企业间的风险关系。借助近期自然语言处理的进展，我们的方法通过基于文件中时间顺序和词汇模式的无监督微调，捕获隐含和抽象的风险联系，从而开发出具有更深层次上下文理解的领域特定财务编码器，并引入定量的风险关系评分以提高透明度和可解释性。广泛的实验表明，在多种评估设置中，我们的方法优于强大 baselines。 

---
# Security smells in infrastructure as code: a taxonomy update beyond the seven sins 

**Title (ZH)**: 基础设施即代码中的安全异味：超越七宗罪的分类学更新 

**Authors**: Aicha War, Serge L.B. Nikiema, Jordan Samhi, Jacques Klein, Tegawende F. Bissyande  

**Link**: [PDF](https://arxiv.org/pdf/2509.18761)  

**Abstract**: Infrastructure as Code (IaC) has become essential for modern software management, yet security flaws in IaC scripts can have severe consequences, as exemplified by the recurring exploits of Cloud Web Services. Prior work has recognized the need to build a precise taxonomy of security smells in IaC scripts as a first step towards developing approaches to improve IaC security. This first effort led to the unveiling of seven sins, limited by the focus on a single IaC tool as well as by the extensive, and potentially biased, manual effort that was required. We propose, in our work, to revisit this taxonomy: first, we extend the study of IaC security smells to a more diverse dataset with scripts associated with seven popular IaC tools, including Terraform, Ansible, Chef, Puppet, Pulumi, Saltstack, and Vagrant; second, we bring in some automation for the analysis by relying on an LLM. While we leverage LLMs for initial pattern processing, all taxonomic decisions underwent systematic human validation and reconciliation with established security standards. Our study yields a comprehensive taxonomy of 62 security smell categories, significantly expanding beyond the previously known seven. We demonstrate actionability by implementing new security checking rules within linters for seven popular IaC tools, often achieving 1.00 precision score. Our evolution study of security smells in GitHub projects reveals that these issues persist for extended periods, likely due to inadequate detection and mitigation tools. This work provides IaC practitioners with insights for addressing common security smells and systematically adopting DevSecOps practices to build safer infrastructure code. 

**Abstract (ZH)**: Infrastructure as Code中的安全异味：一项全面的研究与自动化分析方法 

---
# Complexity of Activity Patterns in a Bio-Inspired Hopfield-Type Network in Different Topologies 

**Title (ZH)**: 不同拓扑结构下生物启发式Hopfield型网络中活动模式的复杂性 

**Authors**: Marco Cafiso, Paolo Paradisi  

**Link**: [PDF](https://arxiv.org/pdf/2509.18758)  

**Abstract**: Neural network models capable of storing memory have been extensively studied in computer science and computational neuroscience. The Hopfield network is a prototypical example of a model designed for associative, or content-addressable, memory and has been analyzed in many forms. Further, ideas and methods from complex network theory have been incorporated into artificial neural networks and learning, emphasizing their structural properties. Nevertheless, the temporal dynamics also play a vital role in biological neural networks, whose temporal structure is a crucial feature to examine. Biological neural networks display complex intermittency and, thus, can be studied through the lens of the temporal complexity (TC) theory. The TC approach look at the metastability of self-organized states, characterized by a power-law decay in the inter-event time distribution and in the total activity distribution or a scaling behavior in the corresponding event-driven diffusion processes. In this study, we present a temporal complexity (TC) analysis of a biologically-inspired Hopfield-type neural network model. We conducted a comparative assessment between scale-free and random network topologies, with particular emphasis on their global activation patterns. Our parametric analysis revealed comparable dynamical behaviors across both neural network architectures. Furthermore, our investigation into temporal complexity characteristics uncovered that seemingly distinct dynamical patterns exhibit similar temporal complexity behaviors. In particular, similar power-law decay in the activity distribution and similar complexity levels are observed in both topologies, but with a much reduced noise in the scale-free topology. Notably, most of the complex dynamical profiles were consistently observed in scale-free network configurations, thus confirming the crucial role of hubs in neural network dynamics. 

**Abstract (ZH)**: 生物启发的记忆型神经网络的时间复杂性分析 

---
# A Generalized Bisimulation Metric of State Similarity between Markov Decision Processes: From Theoretical Propositions to Applications 

**Title (ZH)**: Markov决策过程间状态相似性的广义拟似度度量：从理论命题到应用 

**Authors**: Zhenyu Tao, Wei Xu, Xiaohu You  

**Link**: [PDF](https://arxiv.org/pdf/2509.18714)  

**Abstract**: The bisimulation metric (BSM) is a powerful tool for computing state similarities within a Markov decision process (MDP), revealing that states closer in BSM have more similar optimal value functions. While BSM has been successfully utilized in reinforcement learning (RL) for tasks like state representation learning and policy exploration, its application to multiple-MDP scenarios, such as policy transfer, remains challenging. Prior work has attempted to generalize BSM to pairs of MDPs, but a lack of rigorous analysis of its mathematical properties has limited further theoretical progress. In this work, we formally establish a generalized bisimulation metric (GBSM) between pairs of MDPs, which is rigorously proven with the three fundamental properties: GBSM symmetry, inter-MDP triangle inequality, and the distance bound on identical state spaces. Leveraging these properties, we theoretically analyse policy transfer, state aggregation, and sampling-based estimation in MDPs, obtaining explicit bounds that are strictly tighter than those derived from the standard BSM. Additionally, GBSM provides a closed-form sample complexity for estimation, improving upon existing asymptotic results based on BSM. Numerical results validate our theoretical findings and demonstrate the effectiveness of GBSM in multi-MDP scenarios. 

**Abstract (ZH)**: 广义 bisimulation 距离（GBSM）在马尔可夫决策过程对之间的正式确立及其在多MDP场景中的应用分析 

---
# RSVG-ZeroOV: Exploring a Training-Free Framework for Zero-Shot Open-Vocabulary Visual Grounding in Remote Sensing Images 

**Title (ZH)**: RSVG-ZeroOV: 探索无需训练框架的零样本开放词汇视觉接地在遥感图像中的应用 

**Authors**: Ke Li, Di Wang, Ting Wang, Fuyu Dong, Yiming Zhang, Luyao Zhang, Xiangyu Wang, Shaofeng Li, Quan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18711)  

**Abstract**: Remote sensing visual grounding (RSVG) aims to localize objects in remote sensing images based on free-form natural language expressions. Existing approaches are typically constrained to closed-set vocabularies, limiting their applicability in open-world scenarios. While recent attempts to leverage generic foundation models for open-vocabulary RSVG, they overly rely on expensive high-quality datasets and time-consuming fine-tuning. To address these limitations, we propose \textbf{RSVG-ZeroOV}, a training-free framework that aims to explore the potential of frozen generic foundation models for zero-shot open-vocabulary RSVG. Specifically, RSVG-ZeroOV comprises three key stages: (i) Overview: We utilize a vision-language model (VLM) to obtain cross-attention\footnote[1]{In this paper, although decoder-only VLMs use self-attention over all tokens, we refer to the image-text interaction part as cross-attention to distinguish it from pure visual self-attention.}maps that capture semantic correlations between text queries and visual regions. (ii) Focus: By leveraging the fine-grained modeling priors of a diffusion model (DM), we fill in gaps in structural and shape information of objects, which are often overlooked by VLM. (iii) Evolve: A simple yet effective attention evolution module is introduced to suppress irrelevant activations, yielding purified segmentation masks over the referred objects. Without cumbersome task-specific training, RSVG-ZeroOV offers an efficient and scalable solution. Extensive experiments demonstrate that the proposed framework consistently outperforms existing weakly-supervised and zero-shot methods. 

**Abstract (ZH)**: Remote sensing视觉 grounding (RSVG)旨在基于自由形式的自然语言表达在遥感图像中定位物体。现有方法通常受限于封闭集词汇表，限制了其在开放世界场景中的应用。虽然最近尝试利用通用基础模型来解决开放词汇表的RSVG问题，但它们过度依赖昂贵的高质量数据集和耗时的微调。为解决这些限制，我们提出了RSVG-ZeroOV——一种无需训练的框架，旨在探索冻结的通用基础模型在零样本开放词汇表RSVG中的潜力。具体而言，RSVG-ZeroOV包括三个关键阶段：（i）概要：我们利用视觉语言模型（VLM）获得包含文本查询与视觉区域之间语义关联的交叉注意图。（ii）聚焦：通过利用扩散模型（DM）的精细结构先验，填补视觉对象在结构和形状信息上的不足，这些不足常被VLM忽视。（iii）演化：引入一种简单而有效的注意演化模块抑制无关激活，生成更纯净的目标分割掩码。无需繁琐的任务特定训练，RSVG-ZeroOV提供了高效且可扩展的解决方案。大量实验表明，提出的框架在零样本和弱监督方法中表现始终优于现有方法。 

---
# An overview of neural architectures for self-supervised audio representation learning from masked spectrograms 

**Title (ZH)**: 自掩蔽声谱图导向的自我监督音频表示学习的神经架构综述 

**Authors**: Sarthak Yadav, Sergios Theodoridis, Zheng-Hua Tan  

**Link**: [PDF](https://arxiv.org/pdf/2509.18691)  

**Abstract**: In recent years, self-supervised learning has amassed significant interest for training deep neural representations without labeled data. One such self-supervised learning approach is masked spectrogram modeling, where the objective is to learn semantically rich contextual representations by predicting removed or hidden portions of the input audio spectrogram. With the Transformer neural architecture at its core, masked spectrogram modeling has emerged as the prominent approach for learning general purpose audio representations, a.k.a. audio foundation models. Meanwhile, addressing the issues of the Transformer architecture, in particular the underlying Scaled Dot-product Attention operation, which scales quadratically with input sequence length, has led to renewed interest in recurrent sequence modeling approaches. Among them, Selective structured state space models (such as Mamba) and extended Long Short-Term Memory (xLSTM) are the two most promising approaches which have experienced widespread adoption. While the body of work on these two topics continues to grow, there is currently a lack of an adequate overview encompassing the intersection of these topics. In this paper, we present a comprehensive overview of the aforementioned research domains, covering masked spectrogram modeling and the previously mentioned neural sequence modeling architectures, Mamba and xLSTM. Further, we compare Transformers, Mamba and xLSTM based masked spectrogram models in a unified, reproducible framework on ten diverse downstream audio classification tasks, which will help interested readers to make informed decisions regarding suitability of the evaluated approaches to adjacent applications. 

**Abstract (ZH)**: 近年来，自监督学习因其无需标注数据即可训练深度神经表示而引起了广泛关注。一类这样的自监督学习方法是掩码声谱图建模，其目标是通过预测输入音频声谱图中移除或隐藏的部分来学习语义丰富的上下文表示。以Transformer神经架构为核心，掩码声谱图建模已成为学习通用音频表示，即音频基础模型的主要方法。与此同时，针对Transformer架构本身存在的问题，特别是其底层的标度点积注意操作，该操作随输入序列长度二次增长，导致了对循环序列建模方法的兴趣重燃。在这之中，选择性结构化状态空间模型（如Mamba）和扩展长短期记忆（xLSTM）是两个最具前景的方法，并且已经被广泛采用。尽管这两个领域的研究工作不断增长，但由于缺乏对其交集的全面概述，目前仍存在不足。本文提供了一个全面的概述，涵盖了上述研究领域，包括掩码声谱图建模以及之前提到的神经序列建模架构Mamba和xLSTM。此外，我们在统一且可复制的框架下对基于Transformer、Mamba和xLSTM的掩码声谱图模型进行了比较，这些模型在十项不同的下游音频分类任务上进行了评估，这将帮助感兴趣的读者为相关应用选择合适的评估方法。 

---
# HyperAdapt: Simple High-Rank Adaptation 

**Title (ZH)**: 超适应: 简单的高秩适应 

**Authors**: Abel Gurung, Joseph Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2509.18629)  

**Abstract**: Foundation models excel across diverse tasks, but adapting them to specialized applications often requires fine-tuning, an approach that is memory and compute-intensive. Parameter-efficient fine-tuning (PEFT) methods mitigate this by updating only a small subset of weights. In this paper, we introduce HyperAdapt, a parameter-efficient fine-tuning method that significantly reduces the number of trainable parameters compared to state-of-the-art methods like LoRA. Specifically, HyperAdapt adapts a pre-trained weight matrix by applying row- and column-wise scaling through diagonal matrices, thereby inducing a high-rank update while requiring only $n+m$ trainable parameters for an $n \times m$ matrix. Theoretically, we establish an upper bound on the rank of HyperAdapt's updates, and empirically, we confirm that it consistently induces high-rank transformations across model layers. Experiments on GLUE, arithmetic reasoning, and commonsense reasoning benchmarks with models up to 14B parameters demonstrate that HyperAdapt matches or nearly matches the performance of full fine-tuning and state-of-the-art PEFT methods while using orders of magnitude fewer trainable parameters. 

**Abstract (ZH)**: HyperAdapt：一种高效的参数调优方法 

---
# BRAID: Input-Driven Nonlinear Dynamical Modeling of Neural-Behavioral Data 

**Title (ZH)**: BRAID: 输入驱动的非线性动力学神经-行为数据建模 

**Authors**: Parsa Vahidi, Omid G. Sani, Maryam M. Shanechi  

**Link**: [PDF](https://arxiv.org/pdf/2509.18627)  

**Abstract**: Neural populations exhibit complex recurrent structures that drive behavior, while continuously receiving and integrating external inputs from sensory stimuli, upstream regions, and neurostimulation. However, neural populations are often modeled as autonomous dynamical systems, with little consideration given to the influence of external inputs that shape the population activity and behavioral outcomes. Here, we introduce BRAID, a deep learning framework that models nonlinear neural dynamics underlying behavior while explicitly incorporating any measured external inputs. Our method disentangles intrinsic recurrent neural population dynamics from the effects of inputs by including a forecasting objective within input-driven recurrent neural networks. BRAID further prioritizes the learning of intrinsic dynamics that are related to a behavior of interest by using a multi-stage optimization scheme. We validate BRAID with nonlinear simulations, showing that it can accurately learn the intrinsic dynamics shared between neural and behavioral modalities. We then apply BRAID to motor cortical activity recorded during a motor task and demonstrate that our method more accurately fits the neural-behavioral data by incorporating measured sensory stimuli into the model and improves the forecasting of neural-behavioral data compared with various baseline methods, whether input-driven or not. 

**Abstract (ZH)**: 神经群体表现出复杂的环路结构，驱动行为，同时持续接收和整合来自感觉刺激、上游区域和神经刺激的外部输入。然而，神经群体通常被建模为自治动力系统，对外部输入如何塑造群体活动和行为结果的影响考虑不足。在这里，我们引入了BRAID，这是一种深度学习框架，能够在明确纳入任何测量外部输入的同时，建模行为背后的非线性神经动力学。我们的方法通过在输入驱动的递归神经网络中纳入预测目标，将内在的神经群体动力学与输入效应分离。BRAID进一步通过多阶段优化方案优先学习与目标任务相关的内在动力学。我们通过非线性模拟验证了BRAID，显示它能够准确学习神经和行为模态之间的内在动力学。随后，我们将BRAID应用于记录在运动任务中的运动皮层活动，并通过将测量的感觉刺激纳入模型中，证明了我们的方法更准确地拟合了神经-行为数据，并在各种基线方法（无论是输入驱动的还是非输入驱动的）中提高了神经-行为数据的预测能力。 

---
# Flow marching for a generative PDE foundation model 

**Title (ZH)**: 流行进方法构建生成型偏微分方程基础模型 

**Authors**: Zituo Chen, Sili Deng  

**Link**: [PDF](https://arxiv.org/pdf/2509.18611)  

**Abstract**: Pretraining on large-scale collections of PDE-governed spatiotemporal trajectories has recently shown promise for building generalizable models of dynamical systems. Yet most existing PDE foundation models rely on deterministic Transformer architectures, which lack generative flexibility for many science and engineering applications. We propose Flow Marching, an algorithm that bridges neural operator learning with flow matching motivated by an analysis of error accumulation in physical dynamical systems, and we build a generative PDE foundation model on top of it. By jointly sampling the noise level and the physical time step between adjacent states, the model learns a unified velocity field that transports a noisy current state toward its clean successor, reducing long-term rollout drift while enabling uncertainty-aware ensemble generations. Alongside this core algorithm, we introduce a Physics-Pretrained Variational Autoencoder (P2VAE) to embed physical states into a compact latent space, and an efficient Flow Marching Transformer (FMT) that combines a diffusion-forcing scheme with latent temporal pyramids, achieving up to 15x greater computational efficiency than full-length video diffusion models and thereby enabling large-scale pretraining at substantially reduced cost. We curate a corpus of ~2.5M trajectories across 12 distinct PDE families and train suites of P2VAEs and FMTs at multiple scales. On downstream evaluation, we benchmark on unseen Kolmogorov turbulence with few-shot adaptation, demonstrate long-term rollout stability over deterministic counterparts, and present uncertainty-stratified ensemble results, highlighting the importance of generative PDE foundation models for real-world applications. 

**Abstract (ZH)**: 大规模偏微分方程调控时空轨迹的预训练最近展示了构建动力系统通用模型的潜力。然而，现有的大多数偏微分方程基础模型依赖于确定性的Transformer架构，缺乏许多科学和工程应用中的生成灵活性。我们提出了Flow Marching算法，该算法借鉴了对物理动力系统中误差累积分析，将神经运算学习与流匹配相结合，并在其基础上构建了一个生成性的偏微分方程基础模型。通过联合采样噪声水平和相邻状态的物理时间步长，模型学习到一个统一的速度场，将嘈杂的当前状态转移至其干净的后继状态，从而减少长期滚动部署中的漂移，同时允许不确定性感知的集成生成。除了核心算法外，我们还引入了物理预训练变分自编码器（P2VAE）将物理状态嵌入到紧凑的潜在空间，并且提出了一种高效流行进Transformer（FMT），它结合了扩散驱动方案和潜在时域金字塔，相比全长视频扩散模型，其计算效率可提高多达15倍，从而能够以显著减少的成本进行大规模预训练。我们编 curated 了一个包含约2.5M条轨迹的语料库，横跨12个不同的偏微分方程家族，并在多个尺度上训练了一系列P2VAE和FMT。在下游评估中，我们在未见过的柯尔莫哥洛夫湍流上进行基准测试，展现出长时间滚动部署的稳定性超过了确定性模型，并展示了分层不确定性集成结果，突显了生成性偏微分方程基础模型对于实际应用的重要性。 

---
# SynSonic: Augmenting Sound Event Detection through Text-to-Audio Diffusion ControlNet and Effective Sample Filtering 

**Title (ZH)**: SynSonic: 通过文本到音频扩散ControlNet和有效样本过滤增强声事件检测 

**Authors**: Jiarui Hai, Mounya Elhilali  

**Link**: [PDF](https://arxiv.org/pdf/2509.18603)  

**Abstract**: Data synthesis and augmentation are essential for Sound Event Detection (SED) due to the scarcity of temporally labeled data. While augmentation methods like SpecAugment and Mix-up can enhance model performance, they remain constrained by the diversity of existing samples. Recent generative models offer new opportunities, yet their direct application to SED is challenging due to the lack of precise temporal annotations and the risk of introducing noise through unreliable filtering. To address these challenges and enable generative-based augmentation for SED, we propose SynSonic, a data augmentation method tailored for this task. SynSonic leverages text-to-audio diffusion models guided by an energy-envelope ControlNet to generate temporally coherent sound events. A joint score filtering strategy with dual classifiers ensures sample quality, and we explore its practical integration into training pipelines. Experimental results show that SynSonic improves Polyphonic Sound Detection Scores (PSDS1 and PSDS2), enhancing both temporal localization and sound class discrimination. 

**Abstract (ZH)**: 数据合成与扩增对于声事件检测（SED）至关重要，由于缺乏时间标签数据。虽然SpecAugment和Mix-up等扩增方法可以提升模型性能，但它们仍然受限于现有样本的多样性。最近的生成模型提供了新的机会，但由于缺乏精确的时间标注和通过不可靠的滤波引入噪声的风险，它们直接应用于SED具有挑战性。为了解决这些挑战并使基于生成的扩增适用于SED，我们提出SynSonic，一种针对此任务的数据扩增方法。SynSonic利用受能量包络ControlNet引导的文本转音频扩散模型来生成时间上一致的声事件。双重分类器联合评分筛选策略确保样本质量，并探索其实用集成到训练管道中的方法。实验结果表明，SynSonic提高了多音轨声检测评分（PSDS1和PSDS2），提升了时间和声类的区分度。 

---
# OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation 

**Title (ZH)**: Oracle教育强化学习：面向数据高效且基于事实的放射学报告生成 

**Authors**: Zhuoxiao Chen, Hongyang Yu, Ying Xu, Yadan Luo, Long Duong, Yuan-Fang Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.18600)  

**Abstract**: Radiology report generation (RRG) aims to automatically produce clinically faithful reports from chest X-ray images. Prevailing work typically follows a scale-driven paradigm, by multi-stage training over large paired corpora and oversized backbones, making pipelines highly data- and compute-intensive. In this paper, we propose Oracle-educated GRPO {OraPO) with a FactScore-based reward (FactS) to tackle the RRG task under constrained budgets. OraPO enables single-stage, RL-only training by converting failed GRPO explorations on rare or difficult studies into direct preference supervision via a lightweight oracle step. FactS grounds learning in diagnostic evidence by extracting atomic clinical facts and checking entailment against ground-truth labels, yielding dense, interpretable sentence-level rewards. Together, OraPO and FactS create a compact and powerful framework that significantly improves learning efficiency on clinically challenging cases, setting the new SOTA performance on the CheXpert Plus dataset (0.341 in F1) with 2--3 orders of magnitude less training data using a small base VLM on modest hardware. 

**Abstract (ZH)**: 基于Oracle教育的GRPO（OraPO）及其FactScore奖励机制在受限预算下的胸部X光影像 Radiology报告生成（RRG） 

---
# TsqLoRA: Towards Sensitivity and Quality Low-Rank Adaptation for Efficient Fine-Tuning 

**Title (ZH)**: TsqLoRA: 向量敏感性和质量低秩适应以实现高效微调 

**Authors**: Yu Chen, Yifei Han, Long Zhang, Yue Du, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.18585)  

**Abstract**: Fine-tuning large pre-trained models for downstream tasks has become a fundamental approach in natural language processing. Fully fine-tuning all model parameters is computationally expensive and memory-intensive, especially in resource-constrained environments. Existing parameter-efficient fine-tuning methods reduce the number of trainable parameters but typically overlook the varying sensitivity of different model layers and the importance of training data. In this work, we propose TsqLoRA, a novel method that integrates data-quality-driven selection with sensitivity-aware low-rank adaptation, consisted of two main components: a quality-aware sampling mechanism for selecting the most informative training data, and a dynamic rank allocation module that adjusts the rank of each layer based on its sensitivity to parameter updates. The experimental results demonstrate that TsqLoRA improves fine-tuning efficiency while maintaining or even improving performance on a variety of NLP tasks. Our code will be available at this https URL. 

**Abstract (ZH)**: 基于数据质量驱动的选择与灵敏度aware的低秩适应的细调方法TsqLoRA 

---
# Interaction Topological Transformer for Multiscale Learning in Porous Materials 

**Title (ZH)**: 多孔材料中跨尺度学习的交互拓扑变换器 

**Authors**: Dong Chen, Jian Liu, Chun-Long Chen, Guo-Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2509.18573)  

**Abstract**: Porous materials exhibit vast structural diversity and support critical applications in gas storage, separations, and catalysis. However, predictive modeling remains challenging due to the multiscale nature of structure-property relationships, where performance is governed by both local chemical environments and global pore-network topology. These complexities, combined with sparse and unevenly distributed labeled data, hinder generalization across material families. We propose the Interaction Topological Transformer (ITT), a unified data-efficient framework that leverages novel interaction topology to capture materials information across multiple scales and multiple levels, including structural, elemental, atomic, and pairwise-elemental organization. ITT extracts scale-aware features that reflect both compositional and relational structure within complex porous frameworks, and integrates them through a built-in Transformer architecture that supports joint reasoning across scales. Trained using a two-stage strategy, i.e., self-supervised pretraining on 0.6 million unlabeled structures followed by supervised fine-tuning, ITT achieves state-of-the-art, accurate, and transferable predictions for adsorption, transport, and stability properties. This framework provides a principled and scalable path for learning-guided discovery in structurally and chemically diverse porous materials. 

**Abstract (ZH)**: 多孔材料展现出广泛的结构性多样性，并在气体储存、分离和催化等领域支持关键应用。然而，由于结构-性能关系的多尺度本质，预测建模仍然具有挑战性，这类关系受局部化学环境和全局孔隙网络拓扑的共同影响。这些复杂性结合了稀疏且分布不均的标注数据，阻碍了在材料家族间的泛化能力。我们提出了一种交互拓扑变换器（ITT），这是一个统一的数据高效框架，利用新的交互拓扑来跨多个尺度和多个层次捕获材料信息，包括结构、元素、原子以及元素对的组织。ITT 提取了反映复杂多孔框架中组分和关系结构的尺度感知特征，并通过内置的变换器架构将它们集成起来，支持跨尺度的联合推理。通过两阶段策略进行训练，即自监督预训练100万未标注结构后，再进行监督微调，ITT 实现了吸附、传输和稳定性性质的最佳、精确且可转移的预测。该框架为学习引导下，在结构和化学多样性多孔材料中的发现提供了原则性和可扩展的路径。 

---
# CPCLDETECTOR: Knowledge Enhancement and Alignment Selection for Chinese Patronizing and Condescending Language Detection 

**Title (ZH)**: CPCLDETECTOR：知识增强与贬抑语言检测的选择性对齐 

**Authors**: Jiaxun Yang, Yifei Han, Long Zhang, Liu Yujie, Bin Li, Bo Gao, Yangfan He, Kejia Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2509.18562)  

**Abstract**: Chinese Patronizing and Condescending Language (CPCL) is an implicitly discriminatory toxic speech targeting vulnerable groups on Chinese video platforms. The existing dataset lacks user comments, which are a direct reflection of video content. This undermines the model's understanding of video content and results in the failure to detect some CPLC videos. To make up for this loss, this research reconstructs a new dataset PCLMMPLUS that includes 103k comment entries and expands the dataset size. We also propose the CPCLDetector model with alignment selection and knowledge-enhanced comment content modules. Extensive experiments show the proposed CPCLDetector outperforms the SOTA on PCLMM and achieves higher performance on PCLMMPLUS . CPLC videos are detected more accurately, supporting content governance and protecting vulnerable groups. Code and dataset are available at this https URL. 

**Abstract (ZH)**: 中文偏见和居高临下的语言（CPCL）数据集PCLMMPLUS及其检测模型CPCLDetector的研究 

---
# Global Minimizers of Sigmoid Contrastive Loss 

**Title (ZH)**: 全局Sigmoid对比损失的最小值 

**Authors**: Kiril Bangachev, Guy Bresler, Iliyas Noman, Yury Polyanskiy  

**Link**: [PDF](https://arxiv.org/pdf/2509.18552)  

**Abstract**: The meta-task of obtaining and aligning representations through contrastive pretraining is steadily gaining importance since its introduction in CLIP and ALIGN. In this paper we theoretically explain the advantages of synchronizing with trainable inverse temperature and bias under the sigmoid loss, as implemented in the recent SigLIP and SigLIP2 models of Google DeepMind. Temperature and bias can drive the loss function to zero for a rich class of configurations that we call $(\mathsf{m}, \mathsf{b}_{\mathsf{rel}})$-Constellations. $(\mathsf{m}, \mathsf{b}_{\mathsf{rel}})$-Constellations are a novel combinatorial object related to spherical codes and are parametrized by a margin $\mathsf{m}$ and relative bias $\mathsf{b}_{\mathsf{rel}}$. We use our characterization of constellations to theoretically justify the success of SigLIP on retrieval, to explain the modality gap present in SigLIP, and to identify the necessary dimension for producing high-quality representations. Finally, we propose a reparameterization of the sigmoid loss with explicit relative bias, which improves training dynamics in experiments with synthetic data. 

**Abstract (ZH)**: 通过对比预训练获得和对齐表示的元任务：同步可训练的逆温度和相对偏置以实现sigmoid损失趋零的优势理论解释 

---
# Symphony-MoE: Harmonizing Disparate Pre-trained Models into a Coherent Mixture-of-Experts 

**Title (ZH)**: Symphony-MoE: 谐调异构预训练模型为统一的Mixture-of-Experts 

**Authors**: Qi Wang, Hanyang Peng, Yue Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.18542)  

**Abstract**: Mixture-of-Experts (MoE) models enable scalable performance by activating large parameter sets sparsely, minimizing computational overhead. To circumvent the prohibitive cost of training MoEs from scratch, recent work employs upcycling, reusing a single pre-trained dense model by replicating its feed-forward network (FFN) layers into experts. However, this limits expert diversity, as all experts originate from a single pre-trained dense model. This paper addresses this limitation by constructing powerful MoE models using experts sourced from multiple identically-architected but disparate pre-trained models (e.g., Llama2-Chat and Code Llama). A key challenge lies in the fact that these source models occupy disparate, dissonant regions of the parameter space, making direct upcycling prone to severe performance degradation. To overcome this, we propose Symphony-MoE, a novel two-stage framework designed to harmonize these models into a single, coherent expert mixture. First, we establish this harmony in a training-free manner: we construct a shared backbone via a layer-aware fusion strategy and, crucially, alleviate parameter misalignment among experts using activation-based functional alignment. Subsequently, a single lightweight stage of router training coordinates the entire architecture. Experiments demonstrate that our method successfully integrates experts from heterogeneous sources, achieving an MoE model that significantly surpasses baselines in multi-domain tasks and out-of-distribution generalization. 

**Abstract (ZH)**: 多模型组件(Multi-Model Expert)的Symphony-MoE框架：通过融合多预训练模型构建强大的混合专家模型 

---
# No Verifiable Reward for Prosody: Toward Preference-Guided Prosody Learning in TTS 

**Title (ZH)**: 无验证性的韵律奖励：面向偏好导向的韵律学习在TTS中的应用 

**Authors**: Seungyoun Shin, Dongha Ahn, Jiwoo Kim, Sungwook Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2509.18531)  

**Abstract**: Recent work reports gains in neural text-to-speech (TTS) with Group Relative Policy Optimization (GRPO). However, in the absence of a verifiable reward for \textit{prosody}, GRPO trained on transcription-oriented signals (CER/NLL) lowers error rates yet collapses prosody into monotone, unnatural speech; adding speaker-similarity further destabilizes training and degrades CER. We address this with an \textit{iterative Direct Preference Optimization (DPO)} scheme that uses only a few hundred human-labeled preference pairs per round to directly optimize prosodic naturalness while regularizing to the current model. On \textbf{KoCC-TTS}, a curated dataset of authentic Korean call center interactions capturing task-oriented dialogues, our method attains the highest human preference (ELO) with competitive CER, outperforming GRPO and strong commercial baselines. These results suggest that when prosody cannot be rewarded automatically, \textit{human preference optimization} offers a practical and data-efficient path to natural and robust TTS. The demo page is available at \href{this https URL} 

**Abstract (ZH)**: 近期的研究报告了使用组相对策略优化（GRPO）在神经文本转语音（TTS）中的收益。然而，在缺乏可验证的韵律奖励的情况下，GRPO在转录导向信号（CER/NLL）的基础上训练，虽然降低了错误率，但使韵律退化为单调、不自然的语音；进一步加入说话人相似性会进一步导致训练不稳定并恶化CER。我们通过一种仅使用每轮几百个人标注的偏好对的迭代直接偏好优化（DPO）方案来解决这个问题，该方案可以直接优化韵律自然性，并通过对齐到当前模型来正则化。在 curated 数据集 KoCC-TTS 中，一个捕捉任务导向对话的韩国呼叫中心互动数据集上，我们的方法在保持竞争力的CER的同时获得了最高的人类偏好（ELO），优于GRPO和强大的商用基线。这些结果表明，当韵律无法自动奖励时，人类偏好优化提供了一种实用且数据高效的路径，以实现自然和稳健的TTS。演示页面可在 \href{this https URL} 查看。 

---
# A Rhythm-Aware Phrase Insertion for Classical Arabic Poetry Composition 

**Title (ZH)**: 节奏感知的短语插入用于古典阿拉伯诗歌创作 

**Authors**: Mohamad Elzohbi, Richard Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.18514)  

**Abstract**: This paper presents a methodology for inserting phrases in Arabic poems to conform to a specific rhythm using ByT5, a byte-level multilingual transformer-based model. Our work discusses a rule-based grapheme-to-beat transformation tailored for extracting the rhythm from fully diacritized Arabic script. Our approach employs a conditional denoising objective to fine-tune ByT5, where the model reconstructs masked words to match a target rhythm. We adopt a curriculum learning strategy, pre-training on a general Arabic dataset before fine-tuning on poetic dataset, and explore cross-lingual transfer from English to Arabic. Experimental results demonstrate that our models achieve high rhythmic alignment while maintaining semantic coherence. The proposed model has the potential to be used in co-creative applications in the process of composing classical Arabic poems. 

**Abstract (ZH)**: 本论文提出了一种使用ByT5（字节级多语言变换器模型）在阿拉伯诗歌中插入短语以符合特定节奏的方法。我们的工作讨论了一种规则导向的音素到节奏的转换方法，专门针对从完全标注的阿拉伯文字中提取节奏。我们的方法采用条件去噪目标对ByT5进行微调，其中模型重建掩码单词以匹配目标节奏。我们采用阶梯学习策略，在一般阿拉伯语数据集上预训练，然后在诗歌数据集上进行微调，并探索从英语到阿拉伯语的跨语言迁移。实验结果表明，我们的模型在保持语义一致性的同时达到了高度的节奏对齐。所提模型在创作古典阿拉伯诗歌的协作创作应用中具有潜在用途。 

---
# Dynamical Modeling of Behaviorally Relevant Spatiotemporal Patterns in Neural Imaging Data 

**Title (ZH)**: 行为相关时空模式的神经成像数据动力学建模 

**Authors**: Mohammad Hosseini, Maryam M. Shanechi  

**Link**: [PDF](https://arxiv.org/pdf/2509.18507)  

**Abstract**: High-dimensional imaging of neural activity, such as widefield calcium and functional ultrasound imaging, provide a rich source of information for understanding the relationship between brain activity and behavior. Accurately modeling neural dynamics in these modalities is crucial for understanding this relationship but is hindered by the high-dimensionality, complex spatiotemporal dependencies, and prevalent behaviorally irrelevant dynamics in these modalities. Existing dynamical models often employ preprocessing steps to obtain low-dimensional representations from neural image modalities. However, this process can discard behaviorally relevant information and miss spatiotemporal structure. We propose SBIND, a novel data-driven deep learning framework to model spatiotemporal dependencies in neural images and disentangle their behaviorally relevant dynamics from other neural dynamics. We validate SBIND on widefield imaging datasets, and show its extension to functional ultrasound imaging, a recent modality whose dynamical modeling has largely remained unexplored. We find that our model effectively identifies both local and long-range spatial dependencies across the brain while also dissociating behaviorally relevant neural dynamics. Doing so, SBIND outperforms existing models in neural-behavioral prediction. Overall, SBIND provides a versatile tool for investigating the neural mechanisms underlying behavior using imaging modalities. 

**Abstract (ZH)**: 高维神经活动成像，如宽场钙成像和功能性超声成像，提供了理解脑活动与行为关系的丰富信息。准确建模这些模态中的神经动力学对于理解这种关系至关重要，但受到高维度、复杂的时空依赖关系以及普遍存在的与行为无关的动态的影响。现有的动力学模型通常采用 preprocessing 步骤从神经图像模态中获得低维度表示。然而，这一过程可能会丢弃与行为相关的信息并错过时空结构。我们提出了一种新的数据驱动深度学习框架 SBIND，旨在建模神经图像中的时空依赖关系，并从其他神经动力学中分离出与行为相关的主要神经动力学。我们在宽场成像数据集上验证了 SBIND，并将其扩展到功能性超声成像，这是一种近期模态，其动力学建模尚未得到充分探索。我们发现，我们的模型有效识别了脑内的局部和远端空间依赖关系，并且能够分离出与行为相关的神经动力学。由此，SBIND 在神经行为预测中优于现有模型。总体而言，SBIND 提供了一个多功能的工具，用于使用成像模态研究行为背后的神经机制。 

---
# Hyperbolic Coarse-to-Fine Few-Shot Class-Incremental Learning 

**Title (ZH)**: 双曲尺度自上而下少样本类增量学习 

**Authors**: Jiaxin Dai, Xiang Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18504)  

**Abstract**: In the field of machine learning, hyperbolic space demonstrates superior representation capabilities for hierarchical data compared to conventional Euclidean space. This work focuses on the Coarse-To-Fine Few-Shot Class-Incremental Learning (C2FSCIL) task. Our study follows the Knowe approach, which contrastively learns coarse class labels and subsequently normalizes and freezes the classifier weights of learned fine classes in the embedding space. To better interpret the "coarse-to-fine" paradigm, we propose embedding the feature extractor into hyperbolic space. Specifically, we employ the Poincaré ball model of hyperbolic space, enabling the feature extractor to transform input images into feature vectors within the Poincaré ball instead of Euclidean space. We further introduce hyperbolic contrastive loss and hyperbolic fully-connected layers to facilitate model optimization and classification in hyperbolic space. Additionally, to enhance performance under few-shot conditions, we implement maximum entropy distribution in hyperbolic space to estimate the probability distribution of fine-class feature vectors. This allows generation of augmented features from the distribution to mitigate overfitting during training with limited samples. Experiments on C2FSCIL benchmarks show that our method effectively improves both coarse and fine class accuracies. 

**Abstract (ZH)**: 在机器学习领域，双曲空间在层次数据表示方面优于传统的欧几里得空间。本研究聚焦于粗到细少量样本类增量学习（Coarse-To-Fine Few-Shot Class-Incremental Learning，C2FSCIL）任务。我们的研究遵循Knowe方法，通过对比学习粗类标签，并在嵌入空间中对学习到的细类分类器权重进行规范化和冻结。为了更好地解释“粗到细”范式，我们提出将特征提取器嵌入到双曲空间中。具体而言，我们采用双曲空间的Poincaré球模型，使特征提取器能够将输入图像转换为Poincaré球内的特征向量，而不是欧几里得空间。此外，我们引入了双曲对比损失和双曲全连接层，以促进在双曲空间中的模型优化和分类。为进一步在少量样本条件下提高性能，我们在双曲空间中引入最大熵分布来估计细类特征向量的概率分布，从而生成增强特征以减轻有限样本训练中的过拟合。在C2FSCIL基准测试上的实验表明，我们的方法有效地提高了粗类和细类的准确性。 

---
# LAWCAT: Efficient Distillation from Quadratic to Linear Attention with Convolution across Tokens for Long Context Modeling 

**Title (ZH)**: LAWCAT: 从二次注意力到线性注意力的高效蒸馏通过卷积跨Token建模长上下文 

**Authors**: Zeyu Liu, Souvik Kundu, Lianghao Jiang, Anni Li, Srikanth Ronanki, Sravan Bodapati, Gourav Datta, Peter A. Beerel  

**Link**: [PDF](https://arxiv.org/pdf/2509.18467)  

**Abstract**: Although transformer architectures have achieved state-of-the-art performance across diverse domains, their quadratic computational complexity with respect to sequence length remains a significant bottleneck, particularly for latency-sensitive long-context applications. While recent linear-complexity alternatives are increasingly powerful, effectively training them from scratch is still resource-intensive. To overcome these limitations, we propose LAWCAT (Linear Attention with Convolution Across Time), a novel linearization framework designed to efficiently transfer the capabilities of pre-trained transformers into a performant linear attention architecture. LAWCAT integrates causal Conv1D layers to enhance local dependency modeling and employs normalized gated linear attention to improve generalization across varying context lengths. Our comprehensive evaluations demonstrate that, distilling Mistral-7B with only 1K-length sequences yields over 90\% passkey retrieval accuracy up to 22K tokens, significantly extending its effective context window. Similarly, Llama3.2-1B LAWCAT variant achieves competitive performance on S-NIAH 1\&2\&3 tasks (1K-8K context length) and BABILong benchmark (QA2\&QA3, 0K-16K context length), requiring less than 0.1\% pre-training tokens compared with pre-training models. Furthermore, LAWCAT exhibits faster prefill speeds than FlashAttention-2 for sequences exceeding 8K tokens. LAWCAT thus provides an efficient pathway to high-performance, long-context linear models suitable for edge deployment, reducing reliance on extensive long-sequence training data and computational resources. 

**Abstract (ZH)**: 虽然transformer架构在各个领域取得了最先进性能，但其与序列长度呈二次计算复杂度仍是-latency敏感长上下文应用中的一个重要瓶颈。尽管最近线性复杂度的替代方案越来越强大，但它们从头有效训练仍消耗大量资源。为克服这些限制，我们提出了LAWCAT（Linear Attention with Convolution Across Time），一种新颖的线性化框架，旨在高效地将预训练transformer的能力转移到高性能线性注意力架构中。LAWCAT结合因果Conv1D层以增强局部依赖建模，并采用归一化门控线性注意力以提高在不同上下文长度下的泛化能力。我们的全面评估显示，通过仅使用1K长度序列蒸馏Mistral-7B可获得超过90%的密钥检索准确性，直至22K标记，显著延长了其有效上下文窗口。同样，Llama3.2-1B LAWCAT变体在S-NIAH 1&2&3任务（1K-8K上下文长度）和BABILong基准（QA2&QA3，0K-16K上下文长度）中表现出竞争力，并且预训练标记需求量少于0.1%。此外，LAWCAT在序列超过8K标记时的预填充速度比FlashAttention-2更快。因此，LAWCAT提供了一条高效的途径，以实现适用于边缘部署的高性能、长上下文线性模型，减少了对大量长序列训练数据和计算资源的依赖。 

---
# Developing an AI framework to automatically detect shared decision-making in patient-doctor conversations 

**Title (ZH)**: 开发一种人工智能框架以自动检测患者医生对话中的共同决策过程 

**Authors**: Oscar J. Ponce-Ponte, David Toro-Tobon, Luis F. Figueroa, Michael Gionfriddo, Megan Branda, Victor M. Montori, Saturnino Luz, Juan P. Brito  

**Link**: [PDF](https://arxiv.org/pdf/2509.18439)  

**Abstract**: Shared decision-making (SDM) is necessary to achieve patient-centred care. Currently no methodology exists to automatically measure SDM at scale. This study aimed to develop an automated approach to measure SDM by using language modelling and the conversational alignment (CA) score. A total of 157 video-recorded patient-doctor conversations from a randomized multi-centre trial evaluating SDM decision aids for anticoagulation in atrial fibrillations were transcribed and segmented into 42,559 sentences. Context-response pairs and negative sampling were employed to train deep learning (DL) models and fine-tuned BERT models via the next sentence prediction (NSP) task. Each top-performing model was used to calculate four types of CA scores. A random-effects analysis by clinician, adjusting for age, sex, race, and trial arm, assessed the association between CA scores and SDM outcomes: the Decisional Conflict Scale (DCS) and the Observing Patient Involvement in Decision-Making 12 (OPTION12) scores. p-values were corrected for multiple comparisons with the Benjamini-Hochberg method. Among 157 patients (34% female, mean age 70 SD 10.8), clinicians on average spoke more words than patients (1911 vs 773). The DL model without the stylebook strategy achieved a recall@1 of 0.227, while the fine-tuned BERTbase (110M) achieved the highest recall@1 with 0.640. The AbsMax (18.36 SE7.74 p=0.025) and Max CA (21.02 SE7.63 p=0.012) scores generated with the DL without stylebook were associated with OPTION12. The Max CA score generated with the fine-tuned BERTbase (110M) was associated with the DCS score (-27.61 SE12.63 p=0.037). BERT model sizes did not have an impact the association between CA scores and SDM. This study introduces an automated, scalable methodology to measure SDM in patient-doctor conversations through explainable CA scores, with potential to evaluate SDM strategies at scale. 

**Abstract (ZH)**: 基于语言模型的对话对齐得分在患者医生对话中自动测量共决策的研究 

---
# Scattering Transformer: A Training-Free Transformer Architecture for Heart Murmur Detection 

**Title (ZH)**: 散射变换器：一种无需训练的心脏杂音检测变换器架构 

**Authors**: Rami Zewail  

**Link**: [PDF](https://arxiv.org/pdf/2509.18424)  

**Abstract**: In an attempt to address the need for skilled clinicians in heart sound interpretation, recent research efforts on automating cardiac auscultation have explored deep learning approaches. The majority of these approaches have been based on supervised learning that is always challenged in occasions where training data is limited. More recently, there has been a growing interest in potentials of pre-trained self-supervised audio foundation models for biomedical end tasks. Despite exhibiting promising results, these foundational models are typically computationally intensive. Within the context of automatic cardiac auscultation, this study explores a lightweight alternative to these general-purpose audio foundation models by introducing the Scattering Transformer, a novel, training-free transformer architecture for heart murmur detection. The proposed method leverages standard wavelet scattering networks by introducing contextual dependencies in a transformer-like architecture without any backpropagation. We evaluate our approach on the public CirCor DigiScope dataset, directly comparing it against leading general-purpose foundational models. The Scattering Transformer achieves a Weighted Accuracy(WAR) of 0.786 and an Unweighted Average Recall(UAR) of 0.697, demonstrating performance highly competitive with contemporary state of the art methods. This study establishes the Scattering Transformer as a viable and promising alternative in resource-constrained setups. 

**Abstract (ZH)**: 自动心音诊断中基于散射变换的轻量级变压器方法的研究 

---
# Context Lineage Assurance for Non-Human Identities in Critical Multi-Agent Systems 

**Title (ZH)**: 关键多智能体系统中非人类身份的上下文关联保证 

**Authors**: Sumana Malkapuram, Sameera Gangavarapu, Kailashnath Reddy Kavalakuntla, Ananya Gangavarapu  

**Link**: [PDF](https://arxiv.org/pdf/2509.18415)  

**Abstract**: The proliferation of autonomous software agents necessitates rigorous frameworks for establishing secure and verifiable agent-to-agent (A2A) interactions, particularly when such agents are instantiated as non-human identities(NHIs). We extend the A2A paradigm [1 , 2] by introducing a cryptographically grounded mechanism for lineage verification, wherein the provenance and evolution of NHIs are anchored in append-only Merkle tree structures modeled after Certificate Transparency (CT) logs. Unlike traditional A2A models that primarily secure point-to-point interactions, our approach enables both agents and external verifiers to cryptographically validate multi-hop provenance, thereby ensuring the integrity of the entire call chain.
A federated proof server acts as an auditor across one or more Merkle logs, aggregating inclusion proofs and consistency checks into compact, signed attestations that external parties can verify without access to the full execution trace. In parallel, we augment the A2A agent card to incorporate explicit identity verification primitives, enabling both peer agents and human approvers to authenticate the legitimacy of NHI representations in a standardized manner. Together, these contributions establish a cohesive model that integrates identity attestation, lineage verification, and independent proof auditing, thereby advancing the security posture of inter-agent ecosystems and providing a foundation for robust governance of NHIs in regulated environments such as FedRAMP. 

**Abstract (ZH)**: 自主软件代理的 proliferations 强调了建立安全可验证的代理到代理（A2A）交互的严谨框架的必要性，尤其是当这些代理以非人类身份（NHIs）形式实现时。我们通过引入基于加密机制的谱系验证方法，扩展了 A2A 模式 [1, 2]，其中 NHIs 的起源和演化被锚定在基于证书透明度（CT）日志的只读梅克尔树结构中。与主要保证点对点交互安全的传统 A2A 模型不同，我们的方法使双方代理和外部验证者都能够通过加密手段验证多跳谱系，从而确保整个调用链的完整性。 

---
# An Artificial Intelligence Value at Risk Approach: Metrics and Models 

**Title (ZH)**: 一种人工智能Value at Risk方法：度量与模型 

**Authors**: Luis Enriquez Alvarez  

**Link**: [PDF](https://arxiv.org/pdf/2509.18394)  

**Abstract**: Artificial intelligence risks are multidimensional in nature, as the same risk scenarios may have legal, operational, and financial risk dimensions. With the emergence of new AI regulations, the state of the art of artificial intelligence risk management seems to be highly immature due to upcoming AI regulations. Despite the appearance of several methodologies and generic criteria, it is rare to find guidelines with real implementation value, considering that the most important issue is customizing artificial intelligence risk metrics and risk models for specific AI risk scenarios. Furthermore, the financial departments, legal departments and Government Risk Compliance teams seem to remain unaware of many technical aspects of AI systems, in which data scientists and AI engineers emerge as the most appropriate implementers. It is crucial to decompose the problem of artificial intelligence risk in several dimensions: data protection, fairness, accuracy, robustness, and information security. Consequently, the main task is developing adequate metrics and risk models that manage to reduce uncertainty for decision-making in order to take informed decisions concerning the risk management of AI systems.
The purpose of this paper is to orientate AI stakeholders about the depths of AI risk management. Although it is not extremely technical, it requires a basic knowledge of risk management, quantifying uncertainty, the FAIR model, machine learning, large language models and AI context engineering. The examples presented pretend to be very basic and understandable, providing simple ideas that can be developed regarding specific AI customized environments. There are many issues to solve in AI risk management, and this paper will present a holistic overview of the inter-dependencies of AI risks, and how to model them together, within risk scenarios. 

**Abstract (ZH)**: 人工智能风险具有多维度性质，由于新的人工智能法规的出现，当前的人工智能风险管理似乎非常不成熟。尽管存在多种方法和通用准则，真正具有实际实施价值的指南并不多见，最重要的是为具体的人工智能风险场景量身定制人工智能风险指标和风险模型。此外，财务部门、法律部门和政府风险合规团队似乎对许多人工智能系统的技术方面了解不足，数据科学家和人工智能工程师是最合适的实施者。必须将人工智能风险问题分解为多个维度：数据保护、公平性、准确性、鲁棒性以及信息安全。因此，主要任务是开发能够减少决策不确定性、以便在管理人工智能系统风险时做出明智决策的适当指标和风险模型。 

---
# Graph Enhanced Trajectory Anomaly Detection 

**Title (ZH)**: 图增强轨迹异常检测 

**Authors**: Jonathan Kabala Mbuya, Dieter Pfoser, Antonios Anastasopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2509.18386)  

**Abstract**: Trajectory anomaly detection is essential for identifying unusual and unexpected movement patterns in applications ranging from intelligent transportation systems to urban safety and fraud prevention.
Existing methods only consider limited aspects of the trajectory nature and its movement space by treating trajectories as sequences of sampled locations, with sampling determined by positioning technology, e.g., GPS, or by high-level abstractions such as staypoints. Trajectories are analyzed in Euclidean space, neglecting the constraints and connectivity information of the underlying movement network, e.g., road or transit networks.
The proposed Graph Enhanced Trajectory Anomaly Detection (GETAD) framework tightly integrates road network topology, segment semantics, and historical travel patterns to model trajectory data. GETAD uses a Graph Attention Network to learn road-aware embeddings that capture both physical attributes and transition behavior, and augments these with graph-based positional encodings that reflect the spatial layout of the road network.
A Transformer-based decoder models sequential movement, while a multiobjective loss function combining autoregressive prediction and supervised link prediction ensures realistic and structurally coherent representations.
To improve the robustness of anomaly detection, we introduce Confidence Weighted Negative Log Likelihood (CW NLL), an anomaly scoring function that emphasizes high-confidence deviations.
Experiments on real-world and synthetic datasets demonstrate that GETAD achieves consistent improvements over existing methods, particularly in detecting subtle anomalies in road-constrained environments. These results highlight the benefits of incorporating graph structure and contextual semantics into trajectory modeling, enabling more precise and context-aware anomaly detection. 

**Abstract (ZH)**: 轨迹异常检测对于从智能运输系统到城市安全和欺诈预防等多个应用中识别不寻常和意外的运动模式至关重要。 

---
# Align Where the Words Look: Cross-Attention-Guided Patch Alignment with Contrastive and Transport Regularization for Bengali Captioning 

**Title (ZH)**: 根据单词的位置对齐：带有对比和传输正则化的跨注意力引导补丁对齐方法用于孟加拉语描述生成 

**Authors**: Riad Ahmed Anonto, Sardar Md. Saffat Zabin, M. Saifur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2509.18369)  

**Abstract**: Grounding vision--language models in low-resource languages remains challenging, as they often produce fluent text about the wrong objects. This stems from scarce paired data, translation pivots that break alignment, and English-centric pretraining that ignores target-language semantics. We address this with a compute-aware Bengali captioning pipeline trained on LaBSE-verified EN--BN pairs and 110k bilingual-prompted synthetic images. A frozen MaxViT yields stable visual patches, a Bengali-native mBART-50 decodes, and a lightweight bridge links the modalities. Our core novelty is a tri-loss objective: Patch-Alignment Loss (PAL) aligns real and synthetic patch descriptors using decoder cross-attention, InfoNCE enforces global real--synthetic separation, and Sinkhorn-based OT ensures balanced fine-grained patch correspondence. This PAL+InfoNCE+OT synergy improves grounding, reduces spurious matches, and drives strong gains on Flickr30k-1k (BLEU-4 12.29, METEOR 27.98, BERTScore-F1 71.20) and MSCOCO-1k (BLEU-4 12.00, METEOR 28.14, BERTScore-F1 75.40), outperforming strong CE baselines and narrowing the real--synthetic centroid gap by 41%. 

**Abstract (ZH)**: 在低资源语言中扎根视觉-语言模型仍具有挑战性，因为它们通常会产生关于错误对象的流畅文本。这源于缺乏成对数据、破坏对齐关系的翻译枢纽以及以英语为中心的预训练忽略了目标语言语义。我们使用LaBSE验证的EN-BN成对数据和110K双向提示生成的合成图像，训练了一种计算感知的孟加拉语描述符管道。冻结的MaxViT提供了稳定的视觉片段，孟加拉语原生的mBART-50进行解码，并且一个轻量级的桥梁连接了模态。我们核心的创新是三重损失目标：片段对齐损失（PAL）使用解码器交叉注意力对齐真实和合成的片段描述符，InfoNCE 强制全球真实-合成片段分离，并基于Sinkhorn的OT确保细粒度片段对齐的平衡。PAL+InfoNCE+OT的协同作用改进了扎根性能，减少了虚假匹配，并在 Flickr30k-1k（BLEU-4 12.29，METEOR 27.98，BERTScore-F1 71.20）和 MSCOCO-1k（BLEU-4 12.00，METEOR 28.14，BERTScore-F1 75.40）上取得了显著的提升，超过了强大的CE基线，并缩小了真实-合成质心差距41%。 

---
# Multi-Worker Selection based Distributed Swarm Learning for Edge IoT with Non-i.i.d. Data 

**Title (ZH)**: 基于非i.i.d.数据的边缘物联网分布式 Swarm 学习多工人选择方法 

**Authors**: Zhuoyu Yao, Yue Wang, Songyang Zhang, Yingshu Li, Zhipeng Cai, Zhi Tian  

**Link**: [PDF](https://arxiv.org/pdf/2509.18367)  

**Abstract**: Recent advances in distributed swarm learning (DSL) offer a promising paradigm for edge Internet of Things. Such advancements enhance data privacy, communication efficiency, energy saving, and model scalability. However, the presence of non-independent and identically distributed (non-i.i.d.) data pose a significant challenge for multi-access edge computing, degrading learning performance and diverging training behavior of vanilla DSL. Further, there still lacks theoretical guidance on how data heterogeneity affects model training accuracy, which requires thorough investigation. To fill the gap, this paper first study the data heterogeneity by measuring the impact of non-i.i.d. datasets under the DSL framework. This then motivates a new multi-worker selection design for DSL, termed M-DSL algorithm, which works effectively with distributed heterogeneous data. A new non-i.i.d. degree metric is introduced and defined in this work to formulate the statistical difference among local datasets, which builds a connection between the measure of data heterogeneity and the evaluation of DSL performance. In this way, our M-DSL guides effective selection of multiple works who make prominent contributions for global model updates. We also provide theoretical analysis on the convergence behavior of our M-DSL, followed by extensive experiments on different heterogeneous datasets and non-i.i.d. data settings. Numerical results verify performance improvement and network intelligence enhancement provided by our M-DSL beyond the benchmarks. 

**Abstract (ZH)**: 最近在分布式蜂群学习（DSL）方面的进展为边缘互联网 of Things 提供了有希望的范式。这些进步增强了数据隐私、通信效率、能源节约和模型可扩展性。然而，非独立同分布（non-i.i.d.）数据的存在对多接入边缘计算构成了重大挑战，降低了基础DSL的学习性能并导致了训练行为的发散。此外，仍然缺乏关于数据异质性如何影响模型训练准确性的理论指导，这需要进一步研究。为了填补这一空白，本文首先通过在DSL框架下测量非-i.i.d.数据集的影响来研究数据异质性。这进而促使提出了一种新的DSL多工作者选择设计，称为M-DSL算法，该算法在分布异质数据情况下表现有效。本文引入并定义了一个新的非-i.i.d.度量标准，以形式化局部数据集之间的统计差异，从而建立了数据异质性度量与DSL性能评估之间的联系。通过这种方式，我们的M-DSL指导了多个对全球模型更新做出显著贡献的工作者的有效选择。我们还对M-DSL的收敛行为进行了理论分析，并在不同的异质数据集和非-i.i.d.数据设置上进行了广泛的实验。数值结果验证了M-DSL在基准之上提供的性能改进和网络智能增强。 

---
# Reading Between the Lines: Scalable User Feedback via Implicit Sentiment in Developer Prompts 

**Title (ZH)**: 阅读字里行间之意：通过开发者提示中的隐含情感实现可扩展用户反馈 

**Authors**: Daye Nam, Malgorzata Salawa, Satish Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2509.18361)  

**Abstract**: Evaluating developer satisfaction with conversational AI assistants at scale is critical but challenging. User studies provide rich insights, but are unscalable, while large-scale quantitative signals from logs or in-product ratings are often too shallow or sparse to be reliable. To address this gap, we propose and evaluate a new approach: using sentiment analysis of developer prompts to identify implicit signals of user satisfaction. With an analysis of industrial usage logs of 372 professional developers, we show that this approach can identify a signal in ~8% of all interactions, a rate more than 13 times higher than explicit user feedback, with reasonable accuracy even with an off-the-shelf sentiment analysis approach. This new practical approach to complement existing feedback channels would open up new directions for building a more comprehensive understanding of the developer experience at scale. 

**Abstract (ZH)**: 大规模评估开发人员对对话AI助手的满意度至关重要但具有挑战性。用户研究提供了丰富的见解，但不可扩展，而来自日志或产品内评分的大规模定量信号往往过于浅薄或稀疏，不可靠。为解决这一问题，我们提出并评估了一种新的方法：通过情感分析开发人员提示以识别用户的隐性满意度信号。通过对372名专业开发人员的工业使用日志的分析，我们表明这种方法可以在约8%的所有交互中识别出信号，这一比率比显性用户反馈高出13倍以上，并且即使使用现成的情感分析方法也能获得合理的准确率。这一新的实用方法可以补充现有的反馈渠道，为构建更全面的开发人员体验理解提供新的方向。 

---
# Chiplet-Based RISC-V SoC with Modular AI Acceleration 

**Title (ZH)**: 基于Chiplet的RISC-V SoC模块化AI加速器 

**Authors**: P. Ramkumar, S. S. Bharadwaj  

**Link**: [PDF](https://arxiv.org/pdf/2509.18355)  

**Abstract**: Achieving high performance, energy efficiency, and cost-effectiveness while maintaining architectural flexibility is a critical challenge in the development and deployment of edge AI devices. Monolithic SoC designs struggle with this complex balance mainly due to low manufacturing yields (below 16%) at advanced 360 mm^2 process nodes. This paper presents a novel chiplet-based RISC-V SoC architecture that addresses these limitations through modular AI acceleration and intelligent system level optimization. Our proposed design integrates 4 different key innovations in a 30mm x 30mm silicon interposer: adaptive cross-chiplet Dynamic Voltage and Frequency Scaling (DVFS); AI-aware Universal Chiplet Interconnect Express (UCIe) protocol extensions featuring streaming flow control units and compression-aware transfers; distributed cryptographic security across heterogeneous chiplets; and intelligent sensor-driven load migration. The proposed architecture integrates a 7nm RISC-V CPU chiplet with dual 5nm AI accelerators (15 TOPS INT8 each), 16GB HBM3 memory stacks, and dedicated power management controllers. Experimental results across industry standard benchmarks like MobileNetV2, ResNet-50 and real-time video processing demonstrate significant performance improvements. The AI-optimized configuration achieves ~14.7% latency reduction, 17.3% throughput improvement, and 16.2% power reduction compared to previous basic chiplet implementations. These improvements collectively translate to a 40.1% efficiency gain corresponding to ~3.5 mJ per MobileNetV2 inference (860 mW/244 images/s), while maintaining sub-5ms real-time capability across all experimented workloads. These performance upgrades demonstrate that modular chiplet designs can achieve near-monolithic computational density while enabling cost efficiency, scalability and upgradeability, crucial for next-generation edge AI device applications. 

**Abstract (ZH)**: 实现高性能、高能效和成本效益的同时保持架构灵活性是边缘AI设备开发和部署中的关键挑战。本论文提出了一种基于chiplet的RISC-V SoC架构，通过模块化AI加速和智能系统级优化来解决这些限制。 

---
# Perceptions of AI Across Sectors: A Comparative Review of Public Attitudes 

**Title (ZH)**: 跨行业对AI的感知：公众态度的比较 review 

**Authors**: Filip Bialy, Mark Elliot, Robert Meckin  

**Link**: [PDF](https://arxiv.org/pdf/2509.18233)  

**Abstract**: This paper offers a domain-mediated comparative review of 251 studies on public attitudes toward AI, published between 2011 and 2025. Drawing on a systematic literature review, we analyse how different factors including perceived benefits and concerns (or risks) shape public acceptance of - or resistance to - artificial intelligence across domains and use-cases, including healthcare, education, security, public administration, generative AI, and autonomous vehicles. The analysis highlights recurring patterns in individual, contextual, and technical factors influencing perception, while also tracing variations in institutional trust, perceived fairness, and ethical concerns. We show that the public perception in AI is shaped not only by technical design or performance but also by sector-specific considerations as well as imaginaries, cultural narratives, and historical legacies. This comparative approach offers a foundation for developing more tailored and context-sensitive strategies for responsible AI governance. 

**Abstract (ZH)**: 本文提供了一种领域导向的比较性综述，分析了2011年至2025年间发表的251篇关于公众对人工智能态度的研究。通过系统文献综述的方法，我们研究了包括感知效益和担忧（或风险）在内的多种因素如何影响人工智能在不同领域和应用场景中的接受度或抵触情绪，这些领域和应用场景包括医疗、教育、安全、公共管理、生成型AI和自主车辆。分析突出了个体、情境和技术因素在感知中反复出现的模式，同时追踪了机构信任、感知公平性和伦理关切方面的变化。我们表明，公众对人工智能的看法不仅受技术设计或性能的影响，还受特定行业考虑、想象、文化叙事和历史遗产的影响。这种比较方法为负责任的人工智能治理提供了更具针对性和情境敏感性的策略基础。 

---
# Enhanced Interpretable Knowledge Tracing for Students Performance Prediction with Human understandable Feature Space 

**Title (ZH)**: 增强可解释的知识追踪以实现学生性能预测与人类可理解的特征空间 

**Authors**: Sein Minn, Roger Nkambou  

**Link**: [PDF](https://arxiv.org/pdf/2509.18231)  

**Abstract**: Knowledge Tracing (KT) plays a central role in assessing students skill mastery and predicting their future performance. While deep learning based KT models achieve superior predictive accuracy compared to traditional methods, their complexity and opacity hinder their ability to provide psychologically meaningful explanations. This disconnect between model parameters and cognitive theory poses challenges for understanding and enhancing the learning process, limiting their trustworthiness in educational applications. To address these challenges, we enhance interpretable KT models by exploring human-understandable features derived from students interaction data. By incorporating additional features, particularly those reflecting students learning abilities, our enhanced approach improves predictive accuracy while maintaining alignment with cognitive theory. Our contributions aim to balance predictive power with interpretability, advancing the utility of adaptive learning systems. 

**Abstract (ZH)**: 知识追踪（KT）在评估学生技能掌握情况和预测其未来表现中起着关键作用。虽然基于深度学习的KT模型相较于传统方法在预测准确性上表现出色，但由于其复杂性和不透明性，这些模型难以提供具有心理意义的解释。模型参数与认知理论之间的这种脱节为理解和改进学习过程带来了挑战，限制了其在教育应用中的可信度。为了解决这些挑战，我们通过探索源自学生交互数据的人类可理解特征来增强可解释的KT模型。通过引入额外特征，特别是反映学生学习能力的特征，我们的增强方法在保持与认知理论一致性的基础上提高了预测准确性。我们的贡献旨在平衡预测能力和解释性，促进自适应学习系统的应用。 

---
# Variational Task Vector Composition 

**Title (ZH)**: 变分任务向量组合 

**Authors**: Boyuan Zhang, Yingjun Du, Xiantong Zhen, Ling Shao  

**Link**: [PDF](https://arxiv.org/pdf/2509.18208)  

**Abstract**: Task vectors capture how a model changes during fine-tuning by recording the difference between pre-trained and task-specific weights. The composition of task vectors, a key operator in task arithmetic, enables models to integrate knowledge from multiple tasks without incurring additional inference costs. In this paper, we propose variational task vector composition, where composition coefficients are taken as latent variables and estimated in a Bayesian inference framework. Unlike previous methods that operate at the task level, our framework focuses on sample-specific composition. Motivated by the observation of structural redundancy in task vectors, we introduce a Spike-and-Slab prior that promotes sparsity and preserves only the most informative components. To further address the high variance and sampling inefficiency in sparse, high-dimensional spaces, we develop a gated sampling mechanism that constructs a controllable posterior by filtering the composition coefficients based on both uncertainty and importance. This yields a more stable and interpretable variational framework by deterministically selecting reliable task components, reducing sampling variance while improving transparency and generalization. Experimental results demonstrate that our method consistently outperforms existing approaches across all datasets by selectively leveraging the most reliable and informative components in task vectors. These findings highlight the practical value of our approach, establishing a new standard for efficient and effective task vector composition. 

**Abstract (ZH)**: 变分任务向量组成：一种样本特定的贝叶斯框架 

---
# MNV-17: A High-Quality Performative Mandarin Dataset for Nonverbal Vocalization Recognition in Speech 

**Title (ZH)**: MNV-17：用于语音中非言语 vocalization 识别的高质量表现型 Mandarin 数据集 

**Authors**: Jialong Mai, Jinxin Ji, Xiaofen Xing, Chen Yang, Weidong Chen, Jingyuan Xing, Xiangmin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.18196)  

**Abstract**: Mainstream Automatic Speech Recognition (ASR) systems excel at transcribing lexical content, but largely fail to recognize nonverbal vocalizations (NVs) embedded in speech, such as sighs, laughs, and coughs. This capability is important for a comprehensive understanding of human communication, as NVs convey crucial emotional and intentional cues. Progress in NV-aware ASR has been hindered by the lack of high-quality, well-annotated datasets. To address this gap, we introduce MNV-17, a 7.55-hour performative Mandarin speech dataset. Unlike most existing corpora that rely on model-based detection, MNV-17's performative nature ensures high-fidelity, clearly articulated NV instances. To the best of our knowledge, MNV-17 provides the most extensive set of nonverbal vocalization categories, comprising 17 distinct and well-balanced classes of common NVs. We benchmarked MNV-17 on four mainstream ASR architectures, evaluating their joint performance on semantic transcription and NV classification. The dataset and the pretrained model checkpoints will be made publicly available to facilitate future research in expressive ASR. 

**Abstract (ZH)**: 主流自动语音识别系统在转录词汇内容方面表现出色，但在识别嵌入在语音中的非言语声音（如叹息、笑声和咳嗽）方面效果不佳。这种能力对于全面理解人类交流至关重要，因为非言语声音传达关键的情感和意图线索。由于缺乏高质量的注释数据集，使得非言语声音意识的自动语音识别进展受阻。为弥补这一差距，我们介绍了MNV-17，这是一个7.55小时的表演性 Mandarin 语音数据集。与大多数依赖于基于模型的检测的数据集不同，MNV-17 的表演性质确保了高质量、清晰表达的非言语声音实例。据我们所知，MNV-17 提供了最广泛的非言语声音类别集，包括17种常见非言语声音的平衡分类。我们对四个主流自动语音识别架构进行了基准测试，评估了它们在语义转写和非言语声音分类上的联合性能。该数据集及预训练模型检查点将公开发布，以促进未来具有表现力的自动语音识别研究。 

---
# Visionerves: Automatic and Reproducible Hybrid AI for Peripheral Nervous System Recognition Applied to Endometriosis Cases 

**Title (ZH)**: Visionerves：自动且可再现的混合人工智能在外周神经系统识别中的应用——以子宫内膜异位症病例为例 

**Authors**: Giammarco La Barbera, Enzo Bonnot, Thomas Isla, Juan Pablo de la Plata, Joy-Rose Dunoyer de Segonzac, Jennifer Attali, Cécile Lozach, Alexandre Bellucci, Louis Marcellin, Laure Fournier, Sabine Sarnacki, Pietro Gori, Isabelle Bloch  

**Link**: [PDF](https://arxiv.org/pdf/2509.18185)  

**Abstract**: Endometriosis often leads to chronic pelvic pain and possible nerve involvement, yet imaging the peripheral nerves remains a challenge. We introduce Visionerves, a novel hybrid AI framework for peripheral nervous system recognition from multi-gradient DWI and morphological MRI data. Unlike conventional tractography, Visionerves encodes anatomical knowledge through fuzzy spatial relationships, removing the need for selection of manual ROIs. The pipeline comprises two phases: (A) automatic segmentation of anatomical structures using a deep learning model, and (B) tractography and nerve recognition by symbolic spatial reasoning. Applied to the lumbosacral plexus in 10 women with (confirmed or suspected) endometriosis, Visionerves demonstrated substantial improvements over standard tractography, with Dice score improvements of up to 25% and spatial errors reduced to less than 5 mm. This automatic and reproducible approach enables detailed nerve analysis and paves the way for non-invasive diagnosis of endometriosis-related neuropathy, as well as other conditions with nerve involvement. 

**Abstract (ZH)**: Visionerves：一种用于多梯度DWI和形态学MRI数据中周围神经系统识别的新型混合AI框架 

---
# A Framework for Generating Artificial Datasets to Validate Absolute and Relative Position Concepts 

**Title (ZH)**: 一种用于验证绝对位置和相对位置概念的人工数据集生成框架 

**Authors**: George Corrêa de Araújo, Helena de Almeida Maia, Helio Pedrini  

**Link**: [PDF](https://arxiv.org/pdf/2509.18177)  

**Abstract**: In this paper, we present the Scrapbook framework, a novel methodology designed to generate extensive datasets for probing the learned concepts of artificial intelligence (AI) models. The framework focuses on fundamental concepts such as object recognition, absolute and relative positions, and attribute identification. By generating datasets with a large number of questions about individual concepts and a wide linguistic variation, the Scrapbook framework aims to validate the model's understanding of these basic elements before tackling more complex tasks. Our experimental findings reveal that, while contemporary models demonstrate proficiency in recognizing and enumerating objects, they encounter challenges in comprehending positional information and addressing inquiries with additional constraints. Specifically, the MobileVLM-V2 model showed significant answer disagreements and plausible wrong answers, while other models exhibited a bias toward affirmative answers and struggled with questions involving geometric shapes and positional information, indicating areas for improvement in understanding and consistency. The proposed framework offers a valuable instrument for generating diverse and comprehensive datasets, which can be utilized to systematically assess and enhance the performance of AI models. 

**Abstract (ZH)**: 本文介绍了Scrapbook框架，这是一个新颖的方法学，用于生成大量数据集以探究人工 intelligence (AI) 模型学习的概念。该框架专注于对象识别、绝对和相对位置以及属性识别等基本概念。通过生成包含大量关于个体概念的问题和广泛语言变体的数据集，Scrapbook框架旨在在处理更复杂任务之前验证模型对这些基本元素的理解。实验结果表明，尽管当代模型在识别和枚举对象方面表现出色，但在理解位置信息和处理附加约束的问题时遇到了挑战。特别是，MobileVLM-V2模型在回答问题时显示出显著的分歧和合理的错误回答，而其他模型则倾向于给出肯定的答案，并在涉及几何形状和位置信息的问题上挣扎，这表明理解和一致性方面存在改进的空间。所提出的方法提供了一种有价值的工具，用于生成多样且全面的数据集，这些数据集可以系统地评估和提高AI模型的性能。 

---
# Developing Training Procedures for Piecewise-linear Spline Activation Functions in Neural Networks 

**Title (ZH)**: 开发分段线性样条激活函数在神经网络中的训练程序 

**Authors**: William H Patty  

**Link**: [PDF](https://arxiv.org/pdf/2509.18161)  

**Abstract**: Activation functions in neural networks are typically selected from a set of empirically validated, commonly used static functions such as ReLU, tanh, or sigmoid. However, by optimizing the shapes of a network's activation functions, we can train models that are more parameter-efficient and accurate by assigning more optimal activations to the neurons. In this paper, I present and compare 9 training methodologies to explore dual-optimization dynamics in neural networks with parameterized linear B-spline activation functions. The experiments realize up to 94% lower end model error rates in FNNs and 51% lower rates in CNNs compared to traditional ReLU-based models. These gains come at the cost of additional development and training complexity as well as end model latency. 

**Abstract (ZH)**: 具有参数化线性B样条激活函数的神经网络中的双优化动态训练方法比较 

---
# Event Causality Identification with Synthetic Control 

**Title (ZH)**: 事件因果关系识别的合成控制方法 

**Authors**: Haoyu Wang, Fengze Liu, Jiayao Zhang, Dan Roth, Kyle Richardson  

**Link**: [PDF](https://arxiv.org/pdf/2509.18156)  

**Abstract**: Event causality identification (ECI), a process that extracts causal relations between events from text, is crucial for distinguishing causation from correlation. Traditional approaches to ECI have primarily utilized linguistic patterns and multi-hop relational inference, risking false causality identification due to informal usage of causality and specious graphical inference. In this paper, we adopt the Rubin Causal Model to identify event causality: given two temporally ordered events, we see the first event as the treatment and the second one as the observed outcome. Determining their causality involves manipulating the treatment and estimating the resultant change in the likelihood of the outcome. Given that it is only possible to implement manipulation conceptually in the text domain, as a work-around, we try to find a twin for the protagonist from existing corpora. This twin should have identical life experiences with the protagonist before the treatment but undergoes an intervention of treatment. However, the practical difficulty of locating such a match limits its feasibility. Addressing this issue, we use the synthetic control method to generate such a twin' from relevant historical data, leveraging text embedding synthesis and inversion techniques. This approach allows us to identify causal relations more robustly than previous methods, including GPT-4, which is demonstrated on a causality benchmark, COPES-hard. 

**Abstract (ZH)**: 事件因果识别（ECI）：一种从文本中提取事件间因果关系的过程，对于区分因果关系和相关关系至关重要。传统ECI方法主要依赖于语言模式和多跳关系推理，因因果概念的非正式使用和虚假图形推理而存在虚假因果识别的风险。在本文中，我们采用鲁宾因果模型来识别事件因果关系：给定两个时间顺序事件，我们将第一个事件视为处理，第二个事件视为观测结果。确定它们的因果关系涉及对处理进行操作并估计结果事件概率变化。由于在文本领域仅能从概念上实现这种操作，我们尝试在现有语料库中找到处理前具有相同生活经历但接受相同处理干预的主角孪生体。然而，找到这种匹配的实际困难限制了其可行性。为解决这一问题，我们使用合成控制方法从相关历史数据中生成这种孪生体，利用文本嵌入合成和反转技术。这种方法使得我们能够比GPT-4等先前方法更稳健地识别因果关系，在COPES-hard这种因果关系基准测试上得到了验证。 

---
# WLFM: A Well-Logs Foundation Model for Multi-Task and Cross-Well Geological Interpretation 

**Title (ZH)**: WLFM：一种井 Logging 基础模型用于多任务和跨井地质解释 

**Authors**: Zhenyu Qi, Qing Yu, Jichen Wang, Yun-Bo Zhao, Zerui Li, Wenjun Lv  

**Link**: [PDF](https://arxiv.org/pdf/2509.18152)  

**Abstract**: Well-log interpretation is fundamental for subsurface characterization but remains challenged by heterogeneous tool responses, noisy signals, and limited labels. We propose WLFM, a foundation model pretrained on multi-curve logs from 1200 wells, comprising three stages: tokenization of log patches into geological tokens, self-supervised pretraining with masked-token modeling and stratigraphy-aware contrastive learning, and multi-task adaptation with few-shot fine-tuning. WLFM consistently outperforms state-of-the-art baselines, achieving 0.0041 MSE in porosity estimation and 74.13\% accuracy in lithology classification, while WLFM-Finetune further improves to 0.0038 MSE and 78.10\% accuracy. Beyond predictive accuracy, WLFM exhibits emergent layer-awareness, learns a reusable geological vocabulary, and reconstructs masked curves with reasonable fidelity, though systematic offsets are observed in shallow and ultra-deep intervals. Although boundary detection is not explicitly evaluated here, clustering analyses suggest strong potential for future extension. These results establish WLFM as a scalable, interpretable, and transferable backbone for geological AI, with implications for multi-modal integration of logs, seismic, and textual data. 

**Abstract (ZH)**: 井 Logging 解释是地下表征的基础，但仍然受到异质工具响应、噪声信号和有限标签的挑战。我们提出了一种名为 WLFM 的基础模型，该模型在来自 1200 口井的多曲线井资料上进行预训练，包括三个阶段：井资料片段的地质标记化、掩码标记建模和地层意识对比学习的自监督预训练，以及少样本微调下的多任务适应。WLFM 在孔隙度估算和岩石分类方面均优于现有baseline，达到 0.0041 的 MSE 和 74.13% 的准确率，而 WLFM-Finetune 进一步提高至 0.0038 的 MSE 和 78.10% 的准确率。除了预测精度外，WLFM 还表现出层感知能力，学习到可重用的地质词汇，并合理地重建掩模曲线，尽管在浅部和超深区间观察到系统性偏移。虽然边界检测未在此处显式评估，但聚类分析表明其对未来扩展具有强大的潜力。这些结果确立了 WLFM 作为地质 AI 的可扩展、可解释且可迁移的基础架构，具有将井资料、地震和文本数据多模态集成的应用前景。 

---
# HyperNAS: Enhancing Architecture Representation for NAS Predictor via Hypernetwork 

**Title (ZH)**: HyperNAS：通过超网络增强架构表示以提升NAS预测器性能 

**Authors**: Jindi Lv, Yuhao Zhou, Yuxin Tian, Qing Ye, Wentao Feng, Jiancheng Lv  

**Link**: [PDF](https://arxiv.org/pdf/2509.18151)  

**Abstract**: Time-intensive performance evaluations significantly impede progress in Neural Architecture Search (NAS). To address this, neural predictors leverage surrogate models trained on proxy datasets, allowing for direct performance predictions for new architectures. However, these predictors often exhibit poor generalization due to their limited ability to capture intricate relationships among various architectures. In this paper, we propose HyperNAS, a novel neural predictor paradigm for enhancing architecture representation learning. HyperNAS consists of two primary components: a global encoding scheme and a shared hypernetwork. The global encoding scheme is devised to capture the comprehensive macro-structure information, while the shared hypernetwork serves as an auxiliary task to enhance the investigation of inter-architecture patterns. To ensure training stability, we further develop a dynamic adaptive multi-task loss to facilitate personalized exploration on the Pareto front. Extensive experiments across five representative search spaces, including ViTs, demonstrate the advantages of HyperNAS, particularly in few-shot scenarios. For instance, HyperNAS strikes new state-of-the-art results, with 97.60\% top-1 accuracy on CIFAR-10 and 82.4\% top-1 accuracy on ImageNet, using at least 5.0$\times$ fewer samples. 

**Abstract (ZH)**: 基于时间的性能评估显著阻碍了神经架构搜索（NAS）的进展。为解决这一问题，神经预测器利用基于代理数据集训练的代理模型，可以直接对新架构的性能进行预测。然而，这些预测器由于难以捕捉各种架构间的复杂关系，常常表现出较差的泛化能力。本文提出HyperNAS，一种增强架构表示学习的新型神经预测器范式。HyperNAS包含两个主要组成部分：全局编码方案和共享超网络。全局编码方案旨在捕捉全面的宏观结构信息，而共享超网络作为一种辅助任务，用于增强对架构间模式的研究。为确保训练稳定性，我们进一步开发了一种动态自适应多任务损失，以促进帕累托前沿上的个性化探索。在包括ViTs在内的五个代表性搜索空间上进行的广泛实验表明，HyperNAS在少样本情况下尤其具有优势。例如，HyperNAS在CIFAR-10上实现了97.60%的 top-1 准确率，在ImageNet上实现了82.4%的 top-1 准确率，仅使用了至少5.0倍 fewer 的样本数量就取得了新的最佳结果。 

---
# Augmenting Limited and Biased RCTs through Pseudo-Sample Matching-Based Observational Data Fusion Method 

**Title (ZH)**: 通过伪样本匹配基于观察数据融合方法增强有限的偏倚随机对照试验 

**Authors**: Kairong Han, Weidong Huang, Taiyang Zhou, Peng Zhen, Kun Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18148)  

**Abstract**: In the online ride-hailing pricing context, companies often conduct randomized controlled trials (RCTs) and utilize uplift models to assess the effect of discounts on customer orders, which substantially influences competitive market outcomes. However, due to the high cost of RCTs, the proportion of trial data relative to observational data is small, which only accounts for 0.65\% of total traffic in our context, resulting in significant bias when generalizing to the broader user base. Additionally, the complexity of industrial processes reduces the quality of RCT data, which is often subject to heterogeneity from potential interference and selection bias, making it difficult to correct. Moreover, existing data fusion methods are challenging to implement effectively in complex industrial settings due to the high dimensionality of features and the strict assumptions that are hard to verify with real-world data. To address these issues, we propose an empirical data fusion method called pseudo-sample matching. By generating pseudo-samples from biased, low-quality RCT data and matching them with the most similar samples from large-scale observational data, the method expands the RCT dataset while mitigating its heterogeneity. We validated the method through simulation experiments, conducted offline and online tests using real-world data. In a week-long online experiment, we achieved a 0.41\% improvement in profit, which is a considerable gain when scaled to industrial scenarios with hundreds of millions in revenue. In addition, we discuss the harm to model training, offline evaluation, and online economic benefits when the RCT data quality is not high, and emphasize the importance of improving RCT data quality in industrial scenarios. Further details of the simulation experiments can be found in the GitHub repository this https URL. 

**Abstract (ZH)**: 基于在线网约车定价背景下的试验数据与提升模型融合方法：伪样例匹配方法的研究 

---
# ConceptFlow: Hierarchical and Fine-grained Concept-Based Explanation for Convolutional Neural Networks 

**Title (ZH)**: ConceptFlow：基于概念的分层级和细粒度解释卷积神经网络 

**Authors**: Xinyu Mu, Hui Dou, Furao Shen, Jian Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.18147)  

**Abstract**: Concept-based interpretability for Convolutional Neural Networks (CNNs) aims to align internal model representations with high-level semantic concepts, but existing approaches largely overlook the semantic roles of individual filters and the dynamic propagation of concepts across layers. To address these limitations, we propose ConceptFlow, a concept-based interpretability framework that simulates the internal "thinking path" of a model by tracing how concepts emerge and evolve across layers. ConceptFlow comprises two key components: (i) concept attentions, which associate each filter with relevant high-level concepts to enable localized semantic interpretation, and (ii) conceptual pathways, derived from a concept transition matrix that quantifies how concepts propagate and transform between filters. Together, these components offer a unified and structured view of internal model reasoning. Experimental results demonstrate that ConceptFlow yields semantically meaningful insights into model reasoning, validating the effectiveness of concept attentions and conceptual pathways in explaining decision behavior. By modeling hierarchical conceptual pathways, ConceptFlow provides deeper insight into the internal logic of CNNs and supports the generation of more faithful and human-aligned explanations. 

**Abstract (ZH)**: 基于概念的卷积神经网络可解释性旨在将模型的内部表示与高级语义概念对齐，但现有方法大多忽视了单个滤波器的语义角色以及概念在层间动态传播的作用。为了解决这些局限性，我们提出了ConceptFlow，这是一种基于概念的可解释性框架，通过跟踪概念如何在层间涌现和发展来模拟模型的内部“思考路径”。ConceptFlow 包括两个关键组件：（i）概念注意力，它将每个滤波器与相关的高级概念关联起来，以实现局部语义解释；（ii）概念路径，它们源自一个概念转换矩阵，量化了概念在滤波器之间传播和转化的方式。这两个组件共同提供了对内部模型推理的统一且结构化的视角。实验结果表明，ConceptFlow 提供了有意义的语义洞察，验证了概念注意力和概念路径在解释决策行为中的有效性。通过建模层级概念路径，ConceptFlow 为理解和生成更忠实且符合人类认知的解释提供了更深入的洞察，支持了卷积神经网络内部逻辑的解释。 

---
# Early Prediction of Multi-Label Care Escalation Triggers in the Intensive Care Unit Using Electronic Health Records 

**Title (ZH)**: 基于电子健康记录的重症监护单元多标签护理升级触发因素的早期预测 

**Authors**: Syed Ahmad Chan Bukhari, Amritpal Singh, Shifath Hossain, Iram Wajahat  

**Link**: [PDF](https://arxiv.org/pdf/2509.18145)  

**Abstract**: Intensive Care Unit (ICU) patients often present with complex, overlapping signs of physiological deterioration that require timely escalation of care. Traditional early warning systems, such as SOFA or MEWS, are limited by their focus on single outcomes and fail to capture the multi-dimensional nature of clinical decline. This study proposes a multi-label classification framework to predict Care Escalation Triggers (CETs), including respiratory failure, hemodynamic instability, renal compromise, and neurological deterioration, using the first 24 hours of ICU data. Using the MIMIC-IV database, CETs are defined through rule-based criteria applied to data from hours 24 to 72 (for example, oxygen saturation below 90, mean arterial pressure below 65 mmHg, creatinine increase greater than 0.3 mg/dL, or a drop in Glasgow Coma Scale score greater than 2). Features are extracted from the first 24 hours and include vital sign aggregates, laboratory values, and static demographics. We train and evaluate multiple classification models on a cohort of 85,242 ICU stays (80 percent training: 68,193; 20 percent testing: 17,049). Evaluation metrics include per-label precision, recall, F1-score, and Hamming loss. XGBoost, the best performing model, achieves F1-scores of 0.66 for respiratory, 0.72 for hemodynamic, 0.76 for renal, and 0.62 for neurologic deterioration, outperforming baseline models. Feature analysis shows that clinically relevant parameters such as respiratory rate, blood pressure, and creatinine are the most influential predictors, consistent with the clinical definitions of the CETs. The proposed framework demonstrates practical potential for early, interpretable clinical alerts without requiring complex time-series modeling or natural language processing. 

**Abstract (ZH)**: 重症监护单元（ICU）患者常出现多种生理功能恶化的表现，需要及时升级护理。传统早期预警系统如SOFA或MEWS专注于单一结果，未能捕捉临床恶化多维度的特性。本研究提出一种多标签分类框架，利用ICU前24小时的数据预测护理升级触发因素（CETs），包括呼吸衰竭、血流动力学不稳定、肾功能损害和神经系统恶化。通过MIMIC-IV数据库，CETs定义为基于规则的标准应用于第24至第72小时的数据（例如，氧饱和度低于90%，平均动脉压低于65 mmHg，肌酐增加大于0.3 mg/dL，或格拉斯哥昏迷评分下降大于2分）。特征包括前24小时的生命体征汇总、实验室值和静态人口统计学信息。研究在85,242例ICU住院记录（80%用于训练：68,193；20%用于测试：17,049）上训练和评估了多种分类模型。评估指标包括单标签精确率、召回率、F1值和汉明损失。XGBoost表现最佳，针对呼吸衰竭、血流动力学不稳定、肾功能损害和神经系统恶化的F1值分别为0.66、0.72、0.76和0.62，优于基线模型。特征分析表明，临床相关的参数如呼吸频率、血压和肌酐是最重要的预测因素，与CETs的临床定义一致。所提出的方法展示了在无需复杂的时间序列模型或自然语言处理的情况下实现早期、可解释的临床警报的实用潜力。 

---
# AdaSTI: Conditional Diffusion Models with Adaptive Dependency Modeling for Spatio-Temporal Imputation 

**Title (ZH)**: AdaSTI：具有自适应依赖建模的条件扩散模型时空插补 

**Authors**: Yubo Yang, Yichen Zhu, Bo Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18144)  

**Abstract**: Spatio-temporal data abounds in domain like traffic and environmental monitoring. However, it often suffers from missing values due to sensor malfunctions, transmission failures, etc. Recent years have seen continued efforts to improve spatio-temporal data imputation performance. Recently diffusion models have outperformed other approaches in various tasks, including spatio-temporal imputation, showing competitive performance. Extracting and utilizing spatio-temporal dependencies as conditional information is vital in diffusion-based methods. However, previous methods introduce error accumulation in this process and ignore the variability of the dependencies in the noisy data at different diffusion steps. In this paper, we propose AdaSTI (Adaptive Dependency Model in Diffusion-based Spatio-Temporal Imputation), a novel spatio-temporal imputation approach based on conditional diffusion model. Inside AdaSTI, we propose a BiS4PI network based on a bi-directional S4 model for pre-imputation with the imputed result used to extract conditional information by our designed Spatio-Temporal Conditionalizer (STC)network. We also propose a Noise-Aware Spatio-Temporal (NAST) network with a gated attention mechanism to capture the variant dependencies across diffusion steps. Extensive experiments on three real-world datasets show that AdaSTI outperforms existing methods in all the settings, with up to 46.4% reduction in imputation error. 

**Abstract (ZH)**: 基于条件扩散模型的空间时间数据插补方法：自适应依赖模型AdaSTI 

---
# Weight Mapping Properties of a Dual Tree Single Clock Adiabatic Capacitive Neuron 

**Title (ZH)**: 一种双树单时钟绝热电容神经元的权重映射特性 

**Authors**: Mike Smart, Sachin Maheshwari, Himadri Singh Raghav, Alexander Serb  

**Link**: [PDF](https://arxiv.org/pdf/2509.18143)  

**Abstract**: Dual Tree Single Clock (DTSC) Adiabatic Capacitive Neuron (ACN) circuits offer the potential for highly energy-efficient Artificial Neural Network (ANN) computation in full custom analog IC designs. The efficient mapping of Artificial Neuron (AN) abstract weights, extracted from the software-trained ANNs, onto physical ACN capacitance values has, however, yet to be fully researched. In this paper, we explore the unexpected hidden complexities, challenges and properties of the mapping, as well as, the ramifications for IC designers in terms accuracy, design and implementation. We propose an optimal, AN to ACN methodology, that promotes smaller chip sizes and improved overall classification accuracy, necessary for successful practical deployment. Using TensorFlow and Larq software frameworks, we train three different ANN networks and map their weights into the energy-efficient DTSC ACN capacitance value domain to demonstrate 100% functional equivalency. Finally, we delve into the impact of weight quantization on ACN performance using novel metrics related to practical IC considerations, such as IC floor space and comparator decision-making efficacy. 

**Abstract (ZH)**: Dual Tree Single Clock (DTSC)磁化电容神经元（ACN）电路为全定制模拟IC设计中的高效能源人工神经网络（ANN）计算提供了潜在可能性。然而，将从软件训练的ANN中提取的人工神经元（AN）抽象权重高效映射到物理ACN电容值的研究尚未得到全面研究。本文探讨了映射的隐藏复杂性、挑战和特性，以及对IC设计师在准确性、设计和实现方面的潜在影响。我们提出了一种优化的人工神经元到ACN的方法，以促进较小的芯片尺寸和整体分类准确性，这是成功实际部署所必需的。利用TensorFlow和Larq软件框架，我们训练了三种不同的ANN网络，并将它们的权重映射到节能的DTSC ACN电容值域，以证明100%的功能等效性。最后，我们利用与实际IC考虑相关的新型性能指标深入探讨了权重量化对ACN性能的影响，如IC面积和比较器决策效用。 

---
# KM-GPT: An Automated Pipeline for Reconstructing Individual Patient Data from Kaplan-Meier Plots 

**Title (ZH)**: KM-GPT：从Kaplan-Meier图自动构建个体患者数据的流水线 

**Authors**: Yao Zhao, Haoyue Sun, Yantian Ding, Yanxun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.18141)  

**Abstract**: Reconstructing individual patient data (IPD) from Kaplan-Meier (KM) plots provides valuable insights for evidence synthesis in clinical research. However, existing approaches often rely on manual digitization, which is error-prone and lacks scalability. To address these limitations, we develop KM-GPT, the first fully automated, AI-powered pipeline for reconstructing IPD directly from KM plots with high accuracy, robustness, and reproducibility. KM-GPT integrates advanced image preprocessing, multi-modal reasoning powered by GPT-5, and iterative reconstruction algorithms to generate high-quality IPD without manual input or intervention. Its hybrid reasoning architecture automates the conversion of unstructured information into structured data flows and validates data extraction from complex KM plots. To improve accessibility, KM-GPT is equipped with a user-friendly web interface and an integrated AI assistant, enabling researchers to reconstruct IPD without requiring programming expertise. KM-GPT was rigorously evaluated on synthetic and real-world datasets, consistently demonstrating superior accuracy. To illustrate its utility, we applied KM-GPT to a meta-analysis of gastric cancer immunotherapy trials, reconstructing IPD to facilitate evidence synthesis and biomarker-based subgroup analyses. By automating traditionally manual processes and providing a scalable, web-based solution, KM-GPT transforms clinical research by leveraging reconstructed IPD to enable more informed downstream analyses, supporting evidence-based decision-making. 

**Abstract (ZH)**: 从Kaplan-Meier生存曲线重建个体患者数据（IPD）为临床研究中的证据综合提供了宝贵见解。然而，现有方法通常依赖于手动数字化，容易出错且缺乏可扩展性。为解决这些局限性，我们开发了KM-GPT，这是第一个能够直接从Kaplan-Meier生存曲线自动重建高精度、稳健且可重复的个体患者数据（IPD）的完全自动化的AI驱动管道。KM-GPT整合了高级图像预处理、由GPT-5驱动的多模态推理以及迭代重建算法，无需手动输入或干预即可生成高质量的IPD。其混合推理架构自动化了无结构信息向结构化数据流的转换，并验证了从复杂Kaplan-Meier生存曲线中提取数据的过程。为了提高易用性，KM-GPT配备了用户友好的网页界面和内置的AI助手，使研究人员能够无需编程知识即可重建IPD。KM-GPT在合成和真实世界数据集上的严格评估中，始终显示出优越的准确性。为了展示其用途，我们应用KM-GPT进行了一项胃癌免疫治疗临床试验的元分析，重建IPD以促进证据综合和基于生物标志物的亚组分析。通过自动化传统的手动过程并提供可扩展的基于网络的解决方案，KM-GPT将临床研究转变为通过重建IPD支持更明智的下游分析，并促进基于证据的决策。 

---
# A Machine Learning Framework for Pathway-Driven Therapeutic Target Discovery in Metabolic Disorders 

**Title (ZH)**: 代谢障碍中基于途径的治疗靶点发现的机器学习框架 

**Authors**: Iram Wajahat, Amritpal Singh, Fazel Keshtkar, Syed Ahmad Chan Bukhari  

**Link**: [PDF](https://arxiv.org/pdf/2509.18140)  

**Abstract**: Metabolic disorders, particularly type 2 diabetes mellitus (T2DM), represent a significant global health burden, disproportionately impacting genetically predisposed populations such as the Pima Indians (a Native American tribe from south central Arizona). This study introduces a novel machine learning (ML) framework that integrates predictive modeling with gene-agnostic pathway mapping to identify high-risk individuals and uncover potential therapeutic targets. Using the Pima Indian dataset, logistic regression and t-tests were applied to identify key predictors of T2DM, yielding an overall model accuracy of 78.43%. To bridge predictive analytics with biological relevance, we developed a pathway mapping strategy that links identified predictors to critical signaling networks, including insulin signaling, AMPK, and PPAR pathways. This approach provides mechanistic insights without requiring direct molecular data. Building upon these connections, we propose therapeutic strategies such as dual GLP-1/GIP receptor agonists, AMPK activators, SIRT1 modulators, and phytochemical, further validated through pathway enrichment analyses. Overall, this framework advances precision medicine by offering interpretable and scalable solutions for early detection and targeted intervention in metabolic disorders. The key contributions of this work are: (1) development of an ML framework combining logistic regression and principal component analysis (PCA) for T2DM risk prediction; (2) introduction of a gene-agnostic pathway mapping approach to generate mechanistic insights; and (3) identification of novel therapeutic strategies tailored for high-risk populations. 

**Abstract (ZH)**: 代谢紊乱，特别是2型糖尿病（T2DM），构成了重大的全球健康负担，特别影响如图皮印第安人（亚利桑那州中部南部的土著部族）等遗传易感人群。本研究介绍了一种新型机器学习（ML）框架，该框架结合了预测建模与基因无关通路映射，以识别高风险个体并揭示潜在的治疗靶点。通过使用图皮印第安人数据集，应用逻辑回归和t检验来识别2型糖尿病的关键预测因子，总体模型准确率为78.43%。为将预测分析与生物学相关性相结合，我们开发了一种通路映射策略，将已识别的预测因子与关键信号网络联系起来，包括胰岛素信号传导、AMPK和PPAR通路。该方法提供机制洞察，无需直接分子数据。在此基础上，我们提出了包括双重GLP-1/GIP受体激动剂、AMPK激活剂、SIRT1调节剂和植物化学疗法在内的治疗策略，并通过通路富集分析进一步验证。总体而言，该框架通过提供早期检测和针对性干预的可解释和可扩展解决方案来推进精准医学。本工作的关键贡献包括：（1）开发了一种结合逻辑回归和主成分分析（PCA）的ML框架，用于2型糖尿病风险预测；（2）介绍了基因无关的通路映射方法以生成机制洞察；（3）识别了针对高风险人群的新型治疗策略。 

---
# LoRALib: A Standardized Benchmark for Evaluating LoRA-MoE Methods 

**Title (ZH)**: LoRALib: 一种评估LoRA-MoE方法的标准基准 

**Authors**: Shaoheng Wang, Yao Lu, Yuqi Li, Yaxin Gao, Jiaqi Nie, Shanqing Yu, Yingli Tian, Qi Xuan  

**Link**: [PDF](https://arxiv.org/pdf/2509.18137)  

**Abstract**: As a parameter efficient fine-tuning (PEFT) method, low-rank adaptation (LoRA) can save significant costs in storage and computing, but its strong adaptability to a single task is often accompanied by insufficient cross-task generalization capabilities. To improve this, existing work combines LoRA with mixture-of-experts (MoE) to enhance the model's adaptability through expert modules and routing mechanisms. However, existing LoRA-MoE methods lack unified standards in models, datasets, hyperparameters, and evaluation methods, making it difficult to conduct fair comparisons between different methods. To this end, we proposed a unified benchmark named LoRALib. Specifically, we standardized datasets from $40$ downstream tasks into a unified format, fine-tuned them using the same hyperparameters and obtained $680$ LoRA modules across $17$ model architectures. Based on this LoRA library, we conduct large-scale experiments on $3$ representative LoRA-MoE methods and different LoRA selection mechanisms using the open-sourced testing tool OpenCompass. Extensive experiments show that LoRAMoE performs best, and that prioritizing LoRAs relevant to the target task can further improve the performance of MoE. We hope these findings will inspire future work. Our datasets and LoRA library are available at this https URL and this https URL. 

**Abstract (ZH)**: 作为一种参数高效微调（PEFT）方法，低秩适应（LoRA）可以在存储和计算方面节省大量成本，但其在单任务上的强大适应性通常伴随着跨任务泛化能力不足的问题。为了解决这一问题，现有工作将LoRA与专家集合（MoE）相结合，通过专家模块和路由机制增强模型的适应性。然而，现有的LoRA-MoE方法在模型、数据集、超参数和评估方法方面缺乏统一标准，难以进行公平比较。为此，我们提出了一个统一的基准库LoRALib。具体来说，我们将来自40个下游任务的数据集统一格式化，使用相同的超参数进行微调，并在17种模型架构上获得了680个LoRA模块。基于这个LoRA库，我们使用开源测试工具OpenCompass对3种代表性LoRA-MoE方法和不同的LoRA选择机制进行了大规模实验。广泛的实验表明，LoRAMoE性能最佳，优先选择与目标任务相关的LoRA可以进一步提高MoE的性能。希望这些发现能够启发未来的相关工作。我们的数据集和LoRA库可在以下链接获取：this https URL和this https URL。 

---
# SDGF: Fusing Static and Multi-Scale Dynamic Correlations for Multivariate Time Series Forecasting 

**Title (ZH)**: SDGF: 结合静态和多尺度动态关联的多变量时间序列预测 

**Authors**: Shaoxun Wang, Xingjun Zhang, Qianyang Li, Jiawei Cao, Zhendong Tan  

**Link**: [PDF](https://arxiv.org/pdf/2509.18135)  

**Abstract**: Inter-series correlations are crucial for accurate multivariate time series forecasting, yet these relationships often exhibit complex dynamics across different temporal scales. Existing methods are limited in modeling these multi-scale dependencies and struggle to capture their intricate and evolving nature. To address this challenge, this paper proposes a novel Static-Dynamic Graph Fusion network (SDGF), whose core lies in capturing multi-scale inter-series correlations through a dual-path graph structure learning approach. Specifically, the model utilizes a static graph based on prior knowledge to anchor long-term, stable dependencies, while concurrently employing Multi-level Wavelet Decomposition to extract multi-scale features for constructing an adaptively learned dynamic graph to capture associations at different scales. We design an attention-gated module to fuse these two complementary sources of information intelligently, and a multi-kernel dilated convolutional network is then used to deepen the understanding of temporal patterns. Comprehensive experiments on multiple widely used real-world benchmark datasets demonstrate the effectiveness of our proposed model. 

**Abstract (ZH)**: 跨序列间的联系对于准确的多变量时间序列预测至关重要，但这些关系在不同时间规模上往往表现出复杂的动态特性。现有方法在建模这些多尺度依赖关系方面有限，难以捕捉它们的复杂演变性质。为应对这一挑战，本文提出了一种新型静态-动态图融合网络（SDGF），其核心在于通过双路径图结构学习方法捕获多尺度跨序列间的联系。具体来说，模型利用基于先验知识的静态图来锚定长期稳定依赖关系，同时采用多尺度小波分解来提取多尺度特征，以构建自适应学习的动态图来捕捉不同尺度下的关联性。我们设计了一种注意力门控模块以智能融合这两种互补的信息来源，并使用多种内核空洞卷积网络来加深对时间模式的理解。在多个广泛使用的现实世界基准数据集上的全面实验展示了我们所提模型的有效性。 

---
# Two ways to knowledge? 

**Title (ZH)**: 两种知识途径？ 

**Authors**: Jean-Michel Tucny, Abhisek Ganguly, Santosh Ansumali, Sauro Succi  

**Link**: [PDF](https://arxiv.org/pdf/2509.18131)  

**Abstract**: It is shown that the weight matrices of transformer-based machine learning applications to the solution of two representative physical applications show a random-like character which bears no directly recognizable link to the physical and mathematical structure of the physical problem under study. This suggests that machine learning and the scientific method may represent two distinct and potentially complementary paths to knowledge, even though a strict notion of explainability in terms of direct correspondence between network parameters and physical structures may remain out of reach. It is also observed that drawing a parallel between transformer operation and (generalized) path-integration techniques may account for the random-like nature of the weights, but still does not resolve the tension with explainability. We conclude with some general comments on the hazards of gleaning knowledge without the benefit of Insight. 

**Abstract (ZH)**: 基于变压器的机器学习应用在两个代表性的物理应用中的权重矩阵表现出随机性质，与所研究物理问题的物理和数学结构之间不存在直接可识别的联系。这表明机器学习与科学研究方法可能会代表两种不同的、甚至可能是互补的知识路径，尽管严格意义上的解释性，即网络参数与物理结构之间的直接对应关系，可能仍然难以实现。还观察到，将变压器操作与（广义）路径积分技术相类比可以解释权重的随机性质，但这仍未解决解释性的问题。最后，我们对在缺乏洞察力的情况下获取知识的危险提出一些一般性评论。 

---
# Research on Metro Transportation Flow Prediction Based on the STL-GRU Combined Model 

**Title (ZH)**: 基于STL-GRU组合模型的地铁交通流量预测研究 

**Authors**: Zijie Zhou, Huichen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.18130)  

**Abstract**: In the metro intelligent transportation system, accurate transfer passenger flow prediction is a key link in optimizing operation plans and improving transportation efficiency. To further improve the theory of metro internal transfer passenger flow prediction and provide more reliable support for intelligent operation decisions, this paper innovatively proposes a metro transfer passenger flow prediction model that integrates the Seasonal and Trend decomposition using Loess (STL) method and Gated Recurrent Unit (GRU).In practical application, the model first relies on the deep learning library Keras to complete the construction and training of the GRU model, laying the foundation for subsequent prediction; then preprocesses the original metro card swiping data, uses the graph-based depth-first search algorithm to identify passengers' travel paths, and further constructs the transfer passenger flow time series; subsequently adopts the STL time series decomposition algorithm to decompose the constructed transfer passenger flow time series into trend component, periodic component and residual component, and uses the 3{\sigma} principle to eliminate and fill the outliers in the residual component, and finally completes the transfer passenger flow this http URL the transfer passenger flow data of a certain metro station as the research sample, the validity of the model is verified. The results show that compared with Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), and the combined model of STL time series decomposition method and Long Short-Term Memory (STL-LSTM), the STL-GRU combined prediction model significantly improves the prediction accuracy of transfer passenger flow on weekdays (excluding Fridays), Fridays and rest days, with the mean absolute percentage error (MAPE) of the prediction results reduced by at least 2.3, 1.36 and 6.42 percentage points respectively. 

**Abstract (ZH)**: 基于STL和GRU的地铁换乘客流预测模型 

---
# Anomaly Detection in Electric Vehicle Charging Stations Using Federated Learning 

**Title (ZH)**: 使用联邦学习的电动车辆充电站异常检测 

**Authors**: Bishal K C, Amr Hilal, Pawan Thapa  

**Link**: [PDF](https://arxiv.org/pdf/2509.18126)  

**Abstract**: Federated Learning (FL) is a decentralized training framework widely used in IoT ecosystems that preserves privacy by keeping raw data local, making it ideal for IoT-enabled cyber-physical systems with sensing and communication like Smart Grids (SGs), Connected and Automated Vehicles (CAV), and Electric Vehicle Charging Stations (EVCS). With the rapid expansion of electric vehicle infrastructure, securing these IoT-based charging stations against cyber threats has become critical. Centralized Intrusion Detection Systems (IDS) raise privacy concerns due to sensitive network and user data, making FL a promising alternative. However, current FL-based IDS evaluations overlook practical challenges such as system heterogeneity and non-IID data. To address these challenges, we conducted experiments to evaluate the performance of federated learning for anomaly detection in EV charging stations under system and data heterogeneity. We used FedAvg and FedAvgM, widely studied optimization approaches, to analyze their effectiveness in anomaly detection. Under IID settings, FedAvg achieves superior performance to centralized models using the same neural network. However, performance degrades with non-IID data and system heterogeneity. FedAvgM consistently outperforms FedAvg in heterogeneous settings, showing better convergence and higher anomaly detection accuracy. Our results demonstrate that FL can handle heterogeneity in IoT-based EVCS without significant performance loss, with FedAvgM as a promising solution for robust, privacy-preserving EVCS security. 

**Abstract (ZH)**: 联邦学习在具有感知和通信能力的物联网充电站异构环境中异常检测性能评估 

---
# NurseSchedRL: Attention-Guided Reinforcement Learning for Nurse-Patient Assignment 

**Title (ZH)**: NurseSchedRL：注意力引导的强化学习在护士-患者分配中的应用 

**Authors**: Harsha Koduri  

**Link**: [PDF](https://arxiv.org/pdf/2509.18125)  

**Abstract**: Healthcare systems face increasing pressure to allocate limited nursing resources efficiently while accounting for skill heterogeneity, patient acuity, staff fatigue, and continuity of care. Traditional optimization and heuristic scheduling methods struggle to capture these dynamic, multi-constraint environments. I propose NurseSchedRL, a reinforcement learning framework for nurse-patient assignment that integrates structured state encoding, constrained action masking, and attention-based representations of skills, fatigue, and geographical context. NurseSchedRL uses Proximal Policy Optimization (PPO) with feasibility masks to ensure assignments respect real-world constraints, while dynamically adapting to patient arrivals and varying nurse availability. In simulation with realistic nurse and patient data, NurseSchedRL achieves improved scheduling efficiency, better alignment of skills to patient needs, and reduced fatigue compared to baseline heuristic and unconstrained RL approaches. These results highlight the potential of reinforcement learning for decision support in complex, high-stakes healthcare workforce management. 

**Abstract (ZH)**: 护理系统面临不断增加的压力，需要在考虑技能异质性、患者急性程度、工作人员疲劳以及护理连续性的情况下高效分配有限的护理资源。传统优化和启发式排班方法难以捕捉这些动态的多约束环境。我提出了NurseSchedRL，一种结合结构化状态编码、受限动作屏蔽以及基于注意力的技能、疲劳和地理背景表示的强化学习框架。NurseSchedRL 使用近端策略优化（PPO）和可行性掩码确保排班遵守现实世界的约束条件，并能够动态适应患者到访和护士可用性的变化。在包含现实护士和患者数据的模拟中，NurseSchedRL 达到了更高的排班效率、更好的技能与患者需求匹配度以及更低的疲劳水平，优于基线启发式和无约束的强化学习方法。这些结果突显了强化学习在复杂、高风险的护理人员管理决策支持中的潜力。 

---
# A Coopetitive-Compatible Data Generation Framework for Cross-silo Federated Learning 

**Title (ZH)**: 跨孤岛协作竞争型数据生成框架 

**Authors**: Thanh Linh Nguyen, Quoc-Viet Pham  

**Link**: [PDF](https://arxiv.org/pdf/2509.18120)  

**Abstract**: Cross-silo federated learning (CFL) enables organizations (e.g., hospitals or banks) to collaboratively train artificial intelligence (AI) models while preserving data privacy by keeping data local. While prior work has primarily addressed statistical heterogeneity across organizations, a critical challenge arises from economic competition, where organizations may act as market rivals, making them hesitant to participate in joint training due to potential utility loss (i.e., reduced net benefit). Furthermore, the combined effects of statistical heterogeneity and inter-organizational competition on organizational behavior and system-wide social welfare remain underexplored. In this paper, we propose CoCoGen, a coopetitive-compatible data generation framework, leveraging generative AI (GenAI) and potential game theory to model, analyze, and optimize collaborative learning under heterogeneous and competitive settings. Specifically, CoCoGen characterizes competition and statistical heterogeneity through learning performance and utility-based formulations and models each training round as a weighted potential game. We then derive GenAI-based data generation strategies that maximize social welfare. Experimental results on the Fashion-MNIST dataset reveal how varying heterogeneity and competition levels affect organizational behavior and demonstrate that CoCoGen consistently outperforms baseline methods. 

**Abstract (ZH)**: 跨孤岛联邦学习（CFL）使得组织（如医院或银行）能够在保持数据隐私的情况下协作训练人工智能模型。尽管以往的工作主要关注组织间的统计异质性，但是来自经济竞争的挑战使得组织可能成为市场竞争对手，这使得它们因潜在的净效益损失而不愿参与联合训练。此外，统计异质性和组织间竞争对组织行为和社会福利的综合影响仍然鲜有研究。在本文中，我们提出了一种称作CoCoGen的合作竞争数据生成框架，该框架利用生成人工智能（GenAI）和潜在博弈理论来建模、分析和优化在异质和竞争环境下的协作学习。具体而言，CoCoGen通过学习性能和基于效用的公式来表征竞争和统计异质性，并将每一训练轮次建模为加权潜在博弈。我们随后推导出基于GenAI的数据生成策略以最大化社会福利。Fashion-MNIST数据集上的实验结果揭示了不同异质性和竞争水平如何影响组织行为，并展示了CoCoGen相对于基准方法的一贯优越性。 

---
# Amortized Latent Steering: Low-Cost Alternative to Test-Time Optimization 

**Title (ZH)**: amortized潜在导向：测试时优化的低成本替代方案 

**Authors**: Nathan Egbuna, Saatvik Gaur, Sunishchal Dev, Ashwinee Panda, Maheep Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2509.18116)  

**Abstract**: Test-time optimization remains impractical at scale due to prohibitive inference costs\textemdash techniques like iterative refinement and multi-step verification can require $10$--$100\times$ more compute per query than standard decoding. Latent space test-time optimization methods like LatentSeek offer a more direct approach by steering hidden representations, but still demand expensive per-query optimization loops with multiple backward passes. We propose Amortized Latent Steering (ALS), which collapses this iterative optimization into a single offline-computed vector applied at constant cost during inference. ALS computes the mean difference between hidden states from successful versus unsuccessful generations, then uses this direction to calibrate the model's hidden representations: when decoding drifts away from the success manifold, ALS nudges activations back toward it. Across GSM8K and MATH-$500$ benchmarks, ALS achieves $2$--$5\times$ speedup over iterative methods while matching or surpassing greedy Chain-of-Thought (CoT) and Self-Consistency baselines, yielding up to 101\% improvement in efficiency--accuracy trade-off. These results show that much of latent optimization's benefit can be captured offline, making sophisticated reasoning techniques viable for production deployment. Code is available at~\href{this https URL}{this https URL} 

**Abstract (ZH)**: 测试时的优化由于高昂的推理成本仍难以大规模实现——迭代精炼和多步验证等技术每查询所需的计算量可能比标准解码多10到100倍。潜空间测试时优化方法如LatentSeek通过引导隐藏表示提供了一种更直接的方法，但仍需要昂贵的每查询优化循环和多次反向传播。我们提出了一种名为Amortized Latent Steering (ALS)的方法，将这种迭代优化过程压缩为一个在推理时固定成本计算的向量。ALS计算成功生成与失败生成的隐藏状态均值差异，然后使用该方向校准模型的隐藏表示：当解码偏离成功流形时，ALS推动激活向其方向回归。在GSM8K和MATH-500基准测试中，ALS在迭代方法的基础上实现了2到5倍的速度提升，并达到了或超过了贪婪链式思维（CoT）和自我一致性基线的效果，效率-准确性的权衡改善幅度高达101%。这些结果表明，可以将大部分潜优化的益处提前捕获，从而使复杂的推理技术在生产部署中成为可能。代码可访问<这个 https URL>。 

---
# Prompt Optimization Meets Subspace Representation Learning for Few-shot Out-of-Distribution Detection 

**Title (ZH)**: 面向少量样本 Out-of-Distribution 检测的提示优化与子空间表示学习结合 

**Authors**: Faizul Rakib Sayem, Shahana Ibrahim  

**Link**: [PDF](https://arxiv.org/pdf/2509.18111)  

**Abstract**: The reliability of artificial intelligence (AI) systems in open-world settings depends heavily on their ability to flag out-of-distribution (OOD) inputs unseen during training. Recent advances in large-scale vision-language models (VLMs) have enabled promising few-shot OOD detection frameworks using only a handful of in-distribution (ID) samples. However, existing prompt learning-based OOD methods rely solely on softmax probabilities, overlooking the rich discriminative potential of the feature embeddings learned by VLMs trained on millions of samples. To address this limitation, we propose a novel context optimization (CoOp)-based framework that integrates subspace representation learning with prompt tuning. Our approach improves ID-OOD separability by projecting the ID features into a subspace spanned by prompt vectors, while projecting ID-irrelevant features into an orthogonal null space. To train such OOD detection framework, we design an easy-to-handle end-to-end learning criterion that ensures strong OOD detection performance as well as high ID classification accuracy. Experiments on real-world datasets showcase the effectiveness of our approach. 

**Abstract (ZH)**: 基于上下文优化的子空间表示学习与提示调优结合的异常分布检测框架 

---
# BULL-ODE: Bullwhip Learning with Neural ODEs and Universal Differential Equations under Stochastic Demand 

**Title (ZH)**: BULL-ODE：在随机需求下基于神经ODE和通用微分方程的学习牛鞭效应 

**Authors**: Nachiket N. Naik, Prathamesh Dinesh Joshi, Raj Abhijit Dandekar, Rajat Dandekar, Sreedath Panat  

**Link**: [PDF](https://arxiv.org/pdf/2509.18105)  

**Abstract**: We study learning of continuous-time inventory dynamics under stochastic demand and quantify when structure helps or hurts forecasting of the bullwhip effect. BULL-ODE compares a fully learned Neural ODE (NODE) that models the entire right-hand side against a physics-informed Universal Differential Equation (UDE) that preserves conservation and order-up-to structure while learning a small residual policy term. Classical supply chain models explain the bullwhip through control/forecasting choices and information sharing, while recent physics-informed and neural differential equation methods blend domain constraints with learned components. It is unclear whether structural bias helps or hinders forecasting under different demand regimes. We address this by using a single-echelon testbed with three demand regimes - AR(1) (autocorrelated), i.i.d. Gaussian, and heavy-tailed lognormal. Training is done on varying fractions of each trajectory, followed by evaluation of multi-step forecasts for inventory I, order rate O, and demand D. Across the structured regimes, UDE consistently generalizes better: with 90% of the training horizon, inventory RMSE drops from 4.92 (NODE) to 0.26 (UDE) under AR(1) and from 5.96 to 0.95 under Gaussian demand. Under heavy-tailed lognormal shocks, the flexibility of NODE is better. These trends persist as train18 ing data shrinks, with NODE exhibiting phase drift in extrapolation while UDE remains stable but underreacts to rare spikes. Our results provide concrete guidance: enforce structure when noise is light-tailed or temporally correlated; relax structure when extreme events dominate. Beyond inventory control, the results offer guidance for hybrid modeling in scientific and engineering systems: enforce known structure when conservation laws and modest noise dominate, and relax structure to capture extremes in settings where rare events drive dynamics. 

**Abstract (ZH)**: 我们研究了在随机需求下连续时间库存动态的学习，并量化结构如何帮助或损害牛鞭效应的预测。BULL-ODE将一个完全学习的神经常微分方程（NODE）与一个保留守恒和补充库存结构但学习小型残差策略项的物理启发式通用常微分方程（UDE）进行对比。经典供应链模型通过控制/预测选择和信息共享解释牛鞭效应，而最近的物理启发式和神经常微分方程方法则结合了领域约束与学习组件。尚不清楚结构偏见在不同需求条件下是有助于还是妨碍预测。我们通过使用单一环节试验台和三种需求条件——自相关AR(1)、独立同分布高斯分布和重尾对数正态分布来解决这一问题。在每个轨迹的不同比例下进行训练，然后评估库存I、订单速率O和需求D的多步预测。在结构化条件下，UDE始终表现更好：在AR(1)条件下，随着90%训练时段的使用，库存RMSE从4.92（NODE）降至0.26（UDE）；在高斯需求下，从5.96降至0.95。在重尾对数正态冲击下，NODE的灵活性更高。随着训练数据减少，这些趋势持续存在，NODE在外推时表现出相位漂移，而UDE保持稳定但对稀有突增反应不足。我们的结果为具体指导：当噪声轻尾或时间相关时强制执行结构；当极端事件占主导时放宽结构。超越库存控制，这些结果为科学和工程系统中的混合建模提供了指导：当守恒定律和适度噪声占主导时强制执行已知结构，而在稀有事件驱动动态的环境中放宽结构以捕捉极端情况。 

---
# Data Valuation and Selection in a Federated Model Marketplace 

**Title (ZH)**: 联邦模型市场中的数据估值与选择 

**Authors**: Wenqian Li, Youjia Yang, Ruoxi Jia, Yan Pang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18104)  

**Abstract**: In the era of Artificial Intelligence (AI), marketplaces have become essential platforms for facilitating the exchange of data products to foster data sharing. Model transactions provide economic solutions in data marketplaces that enhance data reusability and ensure the traceability of data ownership. To establish trustworthy data marketplaces, Federated Learning (FL) has emerged as a promising paradigm to enable collaborative learning across siloed datasets while safeguarding data privacy. However, effective data valuation and selection from heterogeneous sources in the FL setup remain key challenges. This paper introduces a comprehensive framework centered on a Wasserstein-based estimator tailored for FL. The estimator not only predicts model performance across unseen data combinations but also reveals the compatibility between data heterogeneity and FL aggregation algorithms. To ensure privacy, we propose a distributed method to approximate Wasserstein distance without requiring access to raw data. Furthermore, we demonstrate that model performance can be reliably extrapolated under the neural scaling law, enabling effective data selection without full-scale training. Extensive experiments across diverse scenarios, such as label skew, mislabeled, and unlabeled sources, show that our approach consistently identifies high-performing data combinations, paving the way for more reliable FL-based model marketplaces. 

**Abstract (ZH)**: 在人工智能时代，市场平台已成为促进数据产品交换、促进数据共享的重要平台。模型交易为数据市场提供了经济解决方案，增强了数据的重用性并确保了数据所有权的可追溯性。为了建立可信的数据市场，联邦学习（FL）作为一种能够在保护数据隐私的同时实现跨孤岛数据协作学习的有前景的范式而出现。然而，在FL设置中，有效数据估值和来自异构来源的数据选择仍然是关键挑战。本文介绍了一个基于Wasserstein的估计器中心的综合框架。该估计器不仅预测了在未见数据组合上的模型性能，还揭示了数据异构性和FL聚合算法之间的兼容性。为了保证隐私，我们提出了一种分布式方法来近似Wasserstein距离，而无需访问原始数据。此外，我们证明模型性能可以通过神经标度律可靠地外推，从而在无需全规模训练的情况下实现有效数据选择。在各类场景下（如标签偏差、误标和未标来源）的广泛实验表明，我们的方法始终能够识别高性能的数据组合，为更可靠的基于FL的数据市场模型奠定了基础。 

---
