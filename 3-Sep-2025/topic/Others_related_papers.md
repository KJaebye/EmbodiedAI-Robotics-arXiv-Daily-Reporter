# Fault-tolerant Model Predictive Control for Spacecraft 

**Title (ZH)**: 空间飞行器容错模型预测控制 

**Authors**: Raphael Stöckner, Pedro Roque, Maria Charitidou, Dimos V. Dimarogonas  

**Link**: [PDF](https://arxiv.org/pdf/2509.02527)  

**Abstract**: Given the cost and critical functions of satellite constellations, ensuring mission longevity and safe decommissioning is essential for space sustainability. This article presents a Model Predictive Control for spacecraft trajectory and setpoint stabilization under multiple actuation failures. The proposed solution allows us to efficiently control the faulty spacecraft enabling safe navigation towards servicing or collision-free trajectories. The proposed scheme ensures closed-loop asymptotic stability and is shown to be recursively feasible. We demonstrate its efficacy through open-source numerical results and realistic experiments using the ATMOS platform. 

**Abstract (ZH)**: 基于多执行机构故障的卫星轨迹和设定点稳定性的模型预测控制方法：确保空间可持续性的故障应付与安全拆解 

---
# Coral: A Unifying Abstraction Layer for Composable Robotics Software 

**Title (ZH)**: Coral: 一种统合的机器人软件模块化抽象层 

**Authors**: Steven Swanbeck, Mitch Pryor  

**Link**: [PDF](https://arxiv.org/pdf/2509.02453)  

**Abstract**: Despite the multitude of excellent software components and tools available in the robotics and broader software engineering communities, successful integration of software for robotic systems remains a time-consuming and challenging task for users of all knowledge and skill levels. And with robotics software often being built into tightly coupled, monolithic systems, even minor alterations to improve performance, adjust to changing task requirements, or deploy to new hardware can require significant engineering investment. To help solve this problem, this paper presents Coral, an abstraction layer for building, deploying, and coordinating independent software components that maximizes composability to allow for rapid system integration without modifying low-level code. Rather than replacing existing tools, Coral complements them by introducing a higher-level abstraction that constrains the integration process to semantically meaningful choices, reducing the configuration burden without limiting adaptability to diverse domains, systems, and tasks. We describe Coral in detail and demonstrate its utility in integrating software for scenarios of increasing complexity, including LiDAR-based SLAM and multi-robot corrosion mitigation tasks. By enabling practical composability in robotics software, Coral offers a scalable solution to a broad range of robotics system integration challenges, improving component reusability, system reconfigurability, and accessibility to both expert and non-expert users. We release Coral open source. 

**Abstract (ZH)**: 尽管机器人学和更广泛的软件工程社区提供了众多优秀的软件组件和工具，但将软件成功集成到机器人系统中仍然是各个知识和技能水平用户都面临的一项耗时且具有挑战性的任务。由于机器人软件通常构建为紧密耦合的大型系统，即使是对性能进行小幅改进、适应变化的任务要求或部署到新硬件所需的小规模修改，也可能需要大量的工程投入。为了解决这一问题，本文介绍了Coral，一个用于构建、部署和协调独立软件组件的抽象层，最大限度地提高可组合性，以便在不修改底层代码的情况下快速实现系统集成。Coral 不会取代现有的工具，而是通过引入更高层次的抽象来补充它们，将集成过程限制在语义上具有意义的选择，从而减少配置负担，同时不限制对不同领域、系统和任务的适应性。我们详细描述了Coral，并展示了其在集成从简单到复杂场景的软件方面的实用性，包括基于LiDAR的SLAM和多机器人腐蚀防护任务。通过在机器人软件中实现实用的可组合性，Coral 提供了一个可扩展的解决方案，以应对广泛的机器人系统集成挑战，提高了组件的重用性、系统的重新配置能力和对专家及非专家用户的可访问性。我们已开源Coral。 

---
# Adaptive Navigation Strategy for Low-Thrust Proximity Operations in Circular Relative Orbit 

**Title (ZH)**: 低推力近距离轨道操作的自适应导航策略 

**Authors**: Dario Ruggiero, Mauro Mancini, Elisa Capello  

**Link**: [PDF](https://arxiv.org/pdf/2509.02204)  

**Abstract**: This paper presents an adaptive observer-based navigation strategy for spacecraft in Circular Relative Orbit (CRO) scenarios, addressing challenges in proximity operations like formation flight and uncooperative target inspection. The proposed method adjusts observer gains based on the estimated state to achieve fast convergence and low noise sensitivity in state estimation. A Lyapunov-based analysis ensures stability and accuracy, while simulations using vision-based sensor data validate the approach under realistic conditions. Compared to classical observers with time-invariant gains, the proposed method enhances trajectory tracking precision and reduces control input switching, making it a promising solution for autonomous spacecraft localization and control. 

**Abstract (ZH)**: 基于自适应观测器的航天器环路相对轨道导航策略及其在接近操作中的应用 

---
# Multi-vessel Interaction-Aware Trajectory Prediction and Collision Risk Assessment 

**Title (ZH)**: 多血管交互感知轨迹预测与碰撞风险评估 

**Authors**: Md Mahbub Alam, Jose F. Rodrigues-Jr, Gabriel Spadon  

**Link**: [PDF](https://arxiv.org/pdf/2509.01836)  

**Abstract**: Accurate vessel trajectory prediction is essential for enhancing situational awareness and preventing collisions. Still, existing data-driven models are constrained mainly to single-vessel forecasting, overlooking vessel interactions, navigation rules, and explicit collision risk assessment. We present a transformer-based framework for multi-vessel trajectory prediction with integrated collision risk analysis. For a given target vessel, the framework identifies nearby vessels. It jointly predicts their future trajectories through parallel streams encoding kinematic and derived physical features, causal convolutions for temporal locality, spatial transformations for positional encoding, and hybrid positional embeddings that capture both local motion patterns and long-range dependencies. Evaluated on large-scale real-world AIS data using joint multi-vessel metrics, the model demonstrates superior forecasting capabilities beyond traditional single-vessel displacement errors. By simulating interactions among predicted trajectories, the framework further quantifies potential collision risks, offering actionable insights to strengthen maritime safety and decision support. 

**Abstract (ZH)**: 基于变压器的多船轨迹预测及集成碰撞风险分析框架 

---
# Data Retrieval with Importance Weights for Few-Shot Imitation Learning 

**Title (ZH)**: 具有重要性权重的数据 retrieval 用于少样本模仿学习 

**Authors**: Amber Xie, Rahul Chand, Dorsa Sadigh, Joey Hejna  

**Link**: [PDF](https://arxiv.org/pdf/2509.01657)  

**Abstract**: While large-scale robot datasets have propelled recent progress in imitation learning, learning from smaller task specific datasets remains critical for deployment in new environments and unseen tasks. One such approach to few-shot imitation learning is retrieval-based imitation learning, which extracts relevant samples from large, widely available prior datasets to augment a limited demonstration dataset. To determine the relevant data from prior datasets, retrieval-based approaches most commonly calculate a prior data point's minimum distance to a point in the target dataset in latent space. While retrieval-based methods have shown success using this metric for data selection, we demonstrate its equivalence to the limit of a Gaussian kernel density (KDE) estimate of the target data distribution. This reveals two shortcomings of the retrieval rule used in prior work. First, it relies on high-variance nearest neighbor estimates that are susceptible to noise. Second, it does not account for the distribution of prior data when retrieving data. To address these issues, we introduce Importance Weighted Retrieval (IWR), which estimates importance weights, or the ratio between the target and prior data distributions for retrieval, using Gaussian KDEs. By considering the probability ratio, IWR seeks to mitigate the bias of previous selection rules, and by using reasonable modeling parameters, IWR effectively smooths estimates using all data points. Across both simulation environments and real-world evaluations on the Bridge dataset we find that our method, IWR, consistently improves performance of existing retrieval-based methods, despite only requiring minor modifications. 

**Abstract (ZH)**: 尽管大规模机器人数据集推动了模仿学习的recent进展，从较小的特定任务数据集学习仍然对于在新环境中部署和面对未见任务至关重要。一种 few-shot 模仿学习的方法是检索基础的模仿学习，它从大量广泛可用的先验数据集中抽取相关样本来扩充有限的示范数据集。为了确定来自先验数据集的相关数据，检索方法通常计算先验数据点在潜在空间中到目标数据集点的最小距离。虽然这种方法在数据选择上表现出成功，我们证明它与目标数据分布的高斯核密度估计(KDE)的极限相等。这揭示了检索规则在先前工作中使用时的两个不足。首先，它依赖于易受噪声影响的最近邻估计。其次，它在检索数据时没有考虑到先验数据的分布。为了应对这些问题，我们引入了重要性加权检索(IWR)，它使用高斯KDE估计目标数据分布和先验数据分布之间的比率作为检索的重要性权重。通过考虑概率比率，IWR力求减轻先前选择规则的偏差，通过使用合理的建模参数，IWR能够使用所有数据点有效地平滑估计。在模拟环境和Bridge数据集的现实世界评估中，我们发现尽管我们的方法IWR仅需进行少量修改，仍能一致地提高现有检索方法的性能。 

---
# Speculative Design of Equitable Robotics: Queer Fictions and Futures 

**Title (ZH)**: 推测性设计公平机器人：奇异fiction与未来 

**Authors**: Minja Axelsson  

**Link**: [PDF](https://arxiv.org/pdf/2509.01643)  

**Abstract**: This paper examines the speculative topic of equitable robots through an exploratory essay format. It focuses specifically on robots by and for LGBTQ+ populations. It aims to provoke thought and conversations in the field about what aspirational queer robotics futures may look like, both in the arts and sciences. First, it briefly reviews the state-of-the-art of queer robotics in fiction and science, drawing together threads from each. Then, it discusses queering robots through three speculative design proposals for queer robot roles: 1) reflecting the queerness of their ''in-group'' queer users, building and celebrating ''in-group'' identity, 2) a new kind of queer activism by implementing queer robot identity performance to interact with ''out-group'' users, with a goal of reducing bigotry through familiarisation, and 3) a network of queer-owned robots, through which the community could reach each other, and distribute and access important resources. The paper then questions whether robots should be queered, and what ethical implications this raises. Finally, the paper makes suggestions for what aspirational queer robotics futures may look like, and what would be required to get there. 

**Abstract (ZH)**: 本文通过探索性散文的形式考察了公平机器人这一富有争议的话题，重点关注为LGBTQ+群体服务的机器人。它旨在引发学术界关于理想中的异性恋机器人未来的思考，涵盖艺术和科学领域。首先，本文简要回顾了虚构和科学中异性恋机器人领域的现状，梳理了两者的相关线索。然后，通过三个关于同性恋机器人角色的设想讨论了如何“同性恋化”机器人：1）反映其“小群体”同性恋用户的身份特征，构建和庆祝“小群体”身份；2）一种新的同性恋主义形式，通过实施同性恋机器人身份表演与“非小群体”用户互动，目标是通过熟悉度减少偏见；3）一个由同性恋所有者经营的机器人网络，使得社群能够相互联系，并分配和获取重要的资源。本文随后探讨了是否应该“同性恋化”机器人，以及这一做法带来的伦理问题。最后，本文提出了理想中的同性恋机器人未来的愿景，以及实现这些愿景所需的前提条件。 

---
# Analyzing Reluctance to Ask for Help When Cooperating With Robots: Insights to Integrate Artificial Agents in HRC 

**Title (ZH)**: 分析合作过程中拒绝向机器人求助的倾向：整合人工智能代理于人机协作中的见解 

**Authors**: Ane San Martin, Michael Hagenow, Julie Shah, Johan Kildal, Elena Lazkano  

**Link**: [PDF](https://arxiv.org/pdf/2509.01450)  

**Abstract**: As robot technology advances, collaboration between humans and robots will become more prevalent in industrial tasks. When humans run into issues in such scenarios, a likely future involves relying on artificial agents or robots for aid. This study identifies key aspects for the design of future user-assisting agents. We analyze quantitative and qualitative data from a user study examining the impact of on-demand assistance received from a remote human in a human-robot collaboration (HRC) assembly task. We study scenarios in which users require help and we assess their experiences in requesting and receiving assistance. Additionally, we investigate participants' perceptions of future non-human assisting agents and whether assistance should be on-demand or unsolicited. Through a user study, we analyze the impact that such design decisions (human or artificial assistant, on-demand or unsolicited help) can have on elicited emotional responses, productivity, and preferences of humans engaged in HRC tasks. 

**Abstract (ZH)**: 随着机器人技术的发展，人类与机器人在工业任务中的协作将更加普遍。当人类在这些场景中遇到问题时，未来很可能依赖人工代理或机器人提供帮助。本研究识别了未来用户辅助代理设计的关键方面。我们通过一项用户研究分析了在人机协作（HRC）装配任务中，从远程人类获得按需协助的影响，研究用户需要帮助的场景，并评估他们请求和接受帮助的经历。此外，我们探讨了参与者对未来非人类辅助代理的看法，以及协助应该是按需提供的还是未经请求的。通过用户研究，我们分析了这些设计决策（人类或人工助手，按需或未经请求的帮助）对参与HRC任务的人所引发的情感反应、生产率和偏好可能产生的影响。 

---
# Disentangled Multi-Context Meta-Learning: Unlocking robust and Generalized Task Learning 

**Title (ZH)**: 解耦多上下文元学习：解锁稳健且通用的任务学习 

**Authors**: Seonsoo Kim, Jun-Gill Kang, Taehong Kim, Seongil Hong  

**Link**: [PDF](https://arxiv.org/pdf/2509.01297)  

**Abstract**: In meta-learning and its downstream tasks, many methods rely on implicit adaptation to task variations, where multiple factors are mixed together in a single entangled representation. This makes it difficult to interpret which factors drive performance and can hinder generalization. In this work, we introduce a disentangled multi-context meta-learning framework that explicitly assigns each task factor to a distinct context vector. By decoupling these variations, our approach improves robustness through deeper task understanding and enhances generalization by enabling context vector sharing across tasks with shared factors. We evaluate our approach in two domains. First, on a sinusoidal regression task, our model outperforms baselines on out-of-distribution tasks and generalizes to unseen sine functions by sharing context vectors associated with shared amplitudes or phase shifts. Second, in a quadruped robot locomotion task, we disentangle the robot-specific properties and the characteristics of the terrain in the robot dynamics model. By transferring disentangled context vectors acquired from the dynamics model into reinforcement learning, the resulting policy achieves improved robustness under out-of-distribution conditions, surpassing the baselines that rely on a single unified context. Furthermore, by effectively sharing context, our model enables successful sim-to-real policy transfer to challenging terrains with out-of-distribution robot-specific properties, using just 20 seconds of real data from flat terrain, a result not achievable with single-task adaptation. 

**Abstract (ZH)**: 在元学习及其下游任务中，许多方法依赖于隐式的适应任务变化，其中多个因素在单一纠缠表示中混合。这使得难以解释哪些因素驱动性能提升，并且会阻碍泛化能力。在本研究中，我们提出了一种解纠缠多上下文元学习框架，明确将每个任务因素分配到一个独特的上下文向量中。通过分离这些变化，我们的方法通过对任务的更深层次理解提高稳健性，并通过允许具有共享因素的任务之间共享上下文向量来增强泛化能力。我们在两个领域评估了该方法。首先，在一个正弦回归任务上，我们的模型在分布外任务上优于基线模型，并通过共享相关共享幅度或相位平移的上下文向量实现了对未见过的正弦函数的泛化。其次，在四足机器人运动任务中，我们解纠缠了机器人的特性和地形在动力学模型中的特性。通过将动力学模型中获得的解纠缠上下文向量转移到强化学习中，所得策略在分布外条件下表现出增强的稳健性，超越了依赖单一统一上下文的基线方法。此外，通过有效共享上下文，我们的模型能够使用仅20秒的平地数据实现实验到现实的策略转移，适用于具有分布外机器人特性的挑战性地形，这是单任务适应无法实现的。 

---
# Toward a Holistic Multi-Criteria Trajectory Evaluation Framework for Autonomous Driving in Mixed Traffic Environment 

**Title (ZH)**: 面向混合交通环境自主驾驶轨迹综合多准则评价框架 

**Authors**: Nouhed Naidja, Stéphane Font, Marc Revilloud, Guillaume Sandou  

**Link**: [PDF](https://arxiv.org/pdf/2509.01291)  

**Abstract**: This paper presents a unified framework for the evaluation and optimization of autonomous vehicle trajectories, integrating formal safety, comfort, and efficiency criteria. An innovative geometric indicator, based on the analysis of safety zones using adaptive ellipses, is used to accurately quantify collision risks. Our method applies the Shoelace formula to compute the intersection area in the case of misaligned and time-varying configurations. Comfort is modeled using indicators centered on longitudinal and lateral jerk, while efficiency is assessed by overall travel time. These criteria are aggregated into a comprehensive objective function solved using a PSO based algorithm. The approach was successfully validated under real traffic conditions via experiments conducted in an urban intersection involving an autonomous vehicle interacting with a human-operated vehicle, and in simulation using data recorded from human driving in real traffic. 

**Abstract (ZH)**: 本文提出了一种综合框架，用于自动驾驶车辆轨迹的评估与优化，整合了形式化安全、舒适性和效率标准。该方法采用基于自适应椭圆分析安全区域的创新几何指标，准确量化碰撞风险。舒适性通过纵向和横向加速度指标建模，效率通过总体行驶时间评估。这些标准被汇总成一个综合目标函数，使用基于PSO的算法求解。该方法在城市交叉口的真实交通条件下，通过涉及自动驾驶车辆与人为操作车辆互动的实验，以及使用真实交通中人类驾驶数据进行的模拟实验中得到验证。 

---
# OpenMulti: Open-Vocabulary Instance-Level Multi-Agent Distributed Implicit Mapping 

**Title (ZH)**: OpenMulti: 开放词汇实例级多Agent分布式隐式映射 

**Authors**: Jianyu Dou, Yinan Deng, Jiahui Wang, Xingsi Tang, Yi Yang, Yufeng Yue  

**Link**: [PDF](https://arxiv.org/pdf/2509.01228)  

**Abstract**: Multi-agent distributed collaborative mapping provides comprehensive and efficient representations for robots. However, existing approaches lack instance-level awareness and semantic understanding of environments, limiting their effectiveness for downstream applications. To address this issue, we propose OpenMulti, an open-vocabulary instance-level multi-agent distributed implicit mapping framework. Specifically, we introduce a Cross-Agent Instance Alignment module, which constructs an Instance Collaborative Graph to ensure consistent instance understanding across agents. To alleviate the degradation of mapping accuracy due to the blind-zone optimization trap, we leverage Cross Rendering Supervision to enhance distributed learning of the scene. Experimental results show that OpenMulti outperforms related algorithms in both fine-grained geometric accuracy and zero-shot semantic accuracy. In addition, OpenMulti supports instance-level retrieval tasks, delivering semantic annotations for downstream applications. The project website of OpenMulti is publicly available at this https URL. 

**Abstract (ZH)**: 多智能体分布式协作建图提供了全面而高效的机器人表示。然而，现有方法缺乏对环境的实例级感知和语义理解，限制了其在下游应用中的效果。为了解决这一问题，我们提出了OpenMulti，一个开放词汇量的实例级多智能体分布式隐式建图框架。具体而言，我们引入了跨智能体实例对齐模块，构建实例协作图以确保各智能体之间的一致实例理解。为了解决由于盲区优化陷阱导致的建图精度下降问题，我们利用跨渲染监督来增强场景的分布式学习能力。实验结果表明，OpenMulti在细粒度几何精度和零样本语义精度上均优于相关算法。此外，OpenMulti支持实例级检索任务，为下游应用提供语义标注。OpenMulti项目的官方网站可在此 https URL 访问。 

---
# A Robust Numerical Method for Solving Trigonometric Equations in Robotic Kinematics 

**Title (ZH)**: 一种求解机器人运动学中三角方程的稳健数值方法 

**Authors**: Hai-Jun Su  

**Link**: [PDF](https://arxiv.org/pdf/2509.01010)  

**Abstract**: This paper presents a robust numerical method for solving systems of trigonometric equations commonly encountered in robotic kinematics. Our approach employs polynomial substitution techniques combined with eigenvalue decomposition to handle singular matrices and edge cases effectively. The method demonstrates superior numerical stability compared to traditional approaches and has been implemented as an open-source Python package. For non-singular matrices, we employ Weierstrass substitution to transform the system into a quartic polynomial, ensuring all analytical solutions are found. For singular matrices, we develop specialized geometric constraint methods using SVD analysis. The solver demonstrates machine precision accuracy ($< 10^{-15}$ error) with 100\% success rate on extensive test cases, making it particularly valuable for robotics applications such as inverse kinematics problems. 

**Abstract (ZH)**: 本文提出了一种稳健的数值方法，用于解决机器人运动学中常见的三角方程系统。该方法结合多项式替换技术和特征值分解来有效处理奇异矩阵和边界情况。与传统方法相比，该方法显示出更优越的数值稳定性，并已被实现为开源Python包。对于非奇异矩阵，采用魏尔斯特拉斯替换将系统转换为四次多项式，确保找到所有解析解。对于奇异矩阵，采用基于SVD分析的专门几何约束方法。求解器在大量测试案例中实现了机器精度精度（误差小于$10^{-15}$）和100%的成功率，特别适用于机器人应用，如逆运动学问题。 

---
# FLUID: A Fine-Grained Lightweight Urban Signalized-Intersection Dataset of Dense Conflict Trajectories 

**Title (ZH)**: FLUID：稠密冲突轨迹的细粒度轻量级城市信号交叉口数据集 

**Authors**: Yiyang Chen, Zhigang Wu, Guohong Zheng, Xuesong Wu, Liwen Xu, Haoyuan Tang, Zhaocheng He, Haipeng Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2509.00497)  

**Abstract**: The trajectory data of traffic participants (TPs) is a fundamental resource for evaluating traffic conditions and optimizing policies, especially at urban intersections. Although data acquisition using drones is efficient, existing datasets still have limitations in scene representativeness, information richness, and data fidelity. This study introduces FLUID, comprising a fine-grained trajectory dataset that captures dense conflicts at typical urban signalized intersections, and a lightweight, full-pipeline framework for drone-based trajectory processing. FLUID covers three distinct intersection types, with approximately 5 hours of recording time and featuring over 20,000 TPs across 8 categories. Notably, the dataset averages two vehicle conflicts per minute, involving roughly 25% of all motor vehicles. FLUID provides comprehensive data, including trajectories, traffic signals, maps, and raw videos. Comparison with the DataFromSky platform and ground-truth measurements validates its high spatio-temporal accuracy. Through a detailed classification of motor vehicle conflicts and violations, FLUID reveals a diversity of interactive behaviors, demonstrating its value for human preference mining, traffic behavior modeling, and autonomous driving research. 

**Abstract (ZH)**: 交通参与者 trajectories 数据 (TPs) 是评估交通状况和优化政策的基础资源，尤其是在城市交叉口。尽管无人机数据采集效率高，现有数据集在场景代表性、信息丰富性和数据保真度方面仍存在局限性。本研究介绍了 FLUID，包含了一个细粒度的轨迹数据集，该数据集捕捉到了典型城市信号交叉口的密集冲突，并且提供了一个轻量级的端到端框架用于无人机轨迹处理。FLUID 包括三种不同类型的交叉口，记录时长约 5 小时，涵盖了超过 20,000 个交通参与者，分为 8 个类别。值得注意的是，每分钟平均有两次车辆冲突，涉及大约 25% 的所有机动车。FLUID 提供了全面的数据，包括轨迹、交通信号、地图和原始视频。与 DataFromSky 平台和实地测量的对比验证了其高时空准确性。通过详细分类车辆冲突和违规行为，FLUID 揭示了多样化的互动行为，展示了其在人类偏好挖掘、交通行为建模和自动驾驶研究中的价值。 

---
# First Order Model-Based RL through Decoupled Backpropagation 

**Title (ZH)**: 基于解耦反向传播的第一阶模型RL方法 

**Authors**: Joseph Amigo, Rooholla Khorrambakht, Elliot Chane-Sane, Nicolas Mansard, Ludovic Righetti  

**Link**: [PDF](https://arxiv.org/pdf/2509.00215)  

**Abstract**: There is growing interest in reinforcement learning (RL) methods that leverage the simulator's derivatives to improve learning efficiency. While early gradient-based approaches have demonstrated superior performance compared to derivative-free methods, accessing simulator gradients is often impractical due to their implementation cost or unavailability. Model-based RL (MBRL) can approximate these gradients via learned dynamics models, but the solver efficiency suffers from compounding prediction errors during training rollouts, which can degrade policy performance. We propose an approach that decouples trajectory generation from gradient computation: trajectories are unrolled using a simulator, while gradients are computed via backpropagation through a learned differentiable model of the simulator. This hybrid design enables efficient and consistent first-order policy optimization, even when simulator gradients are unavailable, as well as learning a critic from simulation rollouts, which is more accurate. Our method achieves the sample efficiency and speed of specialized optimizers such as SHAC, while maintaining the generality of standard approaches like PPO and avoiding ill behaviors observed in other first-order MBRL methods. We empirically validate our algorithm on benchmark control tasks and demonstrate its effectiveness on a real Go2 quadruped robot, across both quadrupedal and bipedal locomotion tasks. 

**Abstract (ZH)**: 利用模拟器梯度提高学习效率的方法：解耦轨迹生成与梯度计算的混合模型强化学习 

---
# A Comparative Study of Spline-Based Trajectory Reconstruction Methods Across Varying Automatic Vehicle Location Data Densities 

**Title (ZH)**: 基于样条的轨迹重建方法在不同自动车辆定位数据密度下的比较研究 

**Authors**: Jake Robbennolt, Sirajum Munira, Stephen D. Boyles  

**Link**: [PDF](https://arxiv.org/pdf/2509.00119)  

**Abstract**: Automatic vehicle location (AVL) data offers insights into transit dynamics, but its effectiveness is often hampered by inconsistent update frequencies, necessitating trajectory reconstruction. This research evaluates 13 trajectory reconstruction methods, including several novel approaches, using high-resolution AVL data from Austin, Texas. We examine the interplay of four critical factors -- velocity, position, smoothing, and data density -- on reconstruction performance. A key contribution of this study is evaluation of these methods across sparse and dense datasets, providing insights into the trade-off between accuracy and resource allocation. Our evaluation framework combines traditional mathematical error metrics for positional and velocity with practical considerations, such as physical realism (e.g., aligning velocity and acceleration with stopped states, deceleration rates, and speed variability). In addition, we provide insight into the relative value of each method in calculating realistic metrics for infrastructure evaluations. Our findings indicate that velocity-aware methods consistently outperform position-only approaches. Interestingly, we discovered that smoothing-based methods can degrade overall performance in complex, congested urban environments, although enforcing monotonicity remains critical. The velocity constrained Hermite interpolation with monotonicity enforcement (VCHIP-ME) yields optimal results, offering a balance between high accuracy and computational efficiency. Its minimal overhead makes it suitable for both historical analysis and real-time applications, providing significant predictive power when combined with dense datasets. These findings offer practical guidance for researchers and practitioners implementing trajectory reconstruction systems and emphasize the importance of investing in higher-frequency AVL data collection for improved analysis. 

**Abstract (ZH)**: 自动车辆定位(AVL)数据提供了公交动态的洞察，但由于更新频率不一致，其有效性受到限制，需进行轨迹重构。本研究使用德克萨斯州奥斯汀的高分辨率AVL数据评估了13种轨迹重构方法，包括若干新型方法，探讨了速度、位置、平滑处理和数据密度等四个关键因素对重构性能的影响。本研究的一大贡献是对这些方法在稀疏和密集数据集上的评估，提供了准确性和资源分配之间权衡的见解。评估框架结合了传统的数学误差指标（如位置和速度误差）以及实际考虑（如物理现实性，例如速度和加速度与停止状态、减速度率和速度变异性的对齐）。此外，本研究还提供了每种方法在计算基础设施评估中的真实度量指标方面的相对价值的见解。研究结果表明，速度感知方法始终优于仅基于位置的方法。有趣的是，我们发现，在复杂、拥堵的市区环境中，基于平滑处理的方法可能会降低总体性能，但确保单调性的约束仍然是关键。速度约束的Hermite插值法（VCHIP-ME）在性能最佳，兼顾高精度和计算效率。其最小的开销使其适用于历史分析和实时应用，与密集数据集结合使用时提供了显著的预测能力。这些发现为实现轨迹重构系统的研究人员和实践者提供了实用指导，并强调了收集更高频率的AVL数据以提高分析效果的重要性。 

---
# EgoTouch: On-Body Touch Input Using AR/VR Headset Cameras 

**Title (ZH)**: 基于身体的触觉输入：使用AR/VR头显摄像头的交互方式 

**Authors**: Vimal Mollyn, Chris Harrison  

**Link**: [PDF](https://arxiv.org/pdf/2509.01786)  

**Abstract**: In augmented and virtual reality (AR/VR) experiences, a user's arms and hands can provide a convenient and tactile surface for touch input. Prior work has shown on-body input to have significant speed, accuracy, and ergonomic benefits over in-air interfaces, which are common today. In this work, we demonstrate high accuracy, bare hands (i.e., no special instrumentation of the user) skin input using just an RGB camera, like those already integrated into all modern XR headsets. Our results show this approach can be accurate, and robust across diverse lighting conditions, skin tones, and body motion (e.g., input while walking). Finally, our pipeline also provides rich input metadata including touch force, finger identification, angle of attack, and rotation. We believe these are the requisite technical ingredients to more fully unlock on-skin interfaces that have been well motivated in the HCI literature but have lacked robust and practical methods. 

**Abstract (ZH)**: 在增强现实和虚拟现实（AR/VR）体验中，用户的胳膊和手可以提供一种便捷且具有触觉反馈的表面用于触控输入。以往研究表明，在体输入相较于当前常见的空中界面具有显著的速度、准确性和人体工程学优势。在这项工作中，我们利用一个像现代XR头显中已集成的RGB摄像头这样的普通摄像头，展示了仅通过裸手皮肤输入实现高精度输入的方法。我们的结果显示，这种方法在不同光照条件、不同肤色和身体运动（如行走时输入）下具有鲁棒性。最后，我们的处理管道还提供了丰富的输入元数据，包括触控力度、手指识别、攻击角度和旋转。我们认为这些是充分利用已被人机交互文献充分证明但缺乏稳健和实用方法的在体界面所需的技术要件。 

---
# Learning to Coordinate: Distributed Meta-Trajectory Optimization Via Differentiable ADMM-DDP 

**Title (ZH)**: 学习协调：基于可微ADMM-DDP的分布式元轨迹优化 

**Authors**: Bingheng Wang, Yichao Gao, Tianchen Sun, Lin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.01630)  

**Abstract**: Distributed trajectory optimization via ADMM-DDP is a powerful approach for coordinating multi-agent systems, but it requires extensive tuning of tightly coupled hyperparameters that jointly govern local task performance and global coordination. In this paper, we propose Learning to Coordinate (L2C), a general framework that meta-learns these hyperparameters, modeled by lightweight agent-wise neural networks, to adapt across diverse tasks and agent configurations. L2C differentiates end-to-end through the ADMM-DDP pipeline in a distributed manner. It also enables efficient meta-gradient computation by reusing DDP components such as Riccati recursions and feedback gains. These gradients correspond to the optimal solutions of distributed matrix-valued LQR problems, coordinated across agents via an auxiliary ADMM framework that becomes convex under mild assumptions. Training is further accelerated by truncating iterations and meta-learning ADMM penalty parameters optimized for rapid residual reduction, with provable Lipschitz-bounded gradient errors. On a challenging cooperative aerial transport task, L2C generates dynamically feasible trajectories in high-fidelity simulation using IsaacSIM, reconfigures quadrotor formations for safe 6-DoF load manipulation in tight spaces, and adapts robustly to varying team sizes and task conditions, while achieving up to $88\%$ faster gradient computation than state-of-the-art methods. 

**Abstract (ZH)**: 分布式轨迹优化 via ADMM-DDP 是一个多智能体系统协调的一种强大方法，但需要对紧密耦合的超参数进行广泛的调优，这些超参数同时管理局部任务性能和全局协调。本文提出了一种名为 Learning to Coordinate (L2C) 的通用框架，该框架通过轻量级的智能体神经网络元学习这些超参数，以适应多样化的任务和智能体配置。L2C 通过分布式的方式在 ADMM-DDP 管道中进行端到端的微分。它还通过重用诸如 Riccati 递推和反馈增益等 DDP 组件来实现高效的元梯度计算。这些梯度对应于通过辅助的 ADMM 框架协调的分布式矩阵值 LQR 问题的最优解，在轻微假设下成为凸问题。通过截断迭代次数和优化 ADMM 惩罚参数加速训练，这些参数旨在快速减少余差，并具有可证明的 Lipschitz 上界的梯度误差。在一项具有挑战性的协同空中运输任务中，L2C 使用 IsaacSIM 进行高保真模拟以生成动态可行的轨迹，重新配置四旋翼机队形以在狭窄空间内安全执行 6 自由度载荷操作，并在团队规模和任务条件变化时表现出高度鲁棒性，同时梯度计算速度比现有方法快达 88%。 

---
# Quantum game models for interaction-aware decision-making in automated driving 

**Title (ZH)**: 量子博弈模型在自动化驾驶中的交互感知决策making 

**Authors**: Karim Essalmi, Fernando Garrido, Fawzi Nashashibi  

**Link**: [PDF](https://arxiv.org/pdf/2509.01582)  

**Abstract**: Decision-making in automated driving must consider interactions with surrounding agents to be effective. However, traditional methods often neglect or oversimplify these interactions because they are difficult to model and solve, which can lead to overly conservative behavior of the ego vehicle. To address this gap, we propose two quantum game models, QG-U1 (Quantum Game - Unitary 1) and QG-G4 (Quantum Game - Gates 4), for interaction-aware decision-making. These models extend classical game theory by incorporating principles of quantum mechanics, such as superposition, interference, and entanglement. Specifically, QG-U1 and QG-G4 are designed for two-player games with two strategies per player and can be executed in real time on a standard computer without requiring quantum hardware. We evaluate both models in merging and roundabout scenarios and compare them with classical game-theoretic methods and baseline approaches (IDM, MOBIL, and a utility-based technique). Results show that QG-G4 achieves lower collision rates and higher success rates compared to baseline methods, while both quantum models yield higher expected payoffs than classical game approaches under certain parameter settings. 

**Abstract (ZH)**: 自动化驾驶中的决策必须考虑与周围代理的互动以实现有效性。然而，传统方法往往因为这些互动难以建模和解决而忽视或简化这些互动，这可能导致自我车辆表现出过度保守的行为。为此，我们提出了两种量子博弈模型，QG-U1（量子博弈-幺正1）和QG-G4（量子博弈-门4），以实现互动感知的决策。这些模型将经典博弈理论扩展到包含量子力学原则（如叠加、干涉和纠缠）。具体而言，QG-U1和QG-G4适用于每方有两个策略的两玩家博弈，并且可以在标准计算机上实时运行而无需量子硬件。我们在这两种模型在并道和环岛场景中进行了评估，并将它们与经典博弈理论方法和基线方法（IDM、MOBIL和基于效用的技术）进行了比较。结果表明，在某些参数设置下，QG-G4相比基线方法具有更低的碰撞率和更高的成功率，而两种量子模型在与经典博弈方法相比时具有更高的预期收益。 

---
# Metamorphic Testing of Multimodal Human Trajectory Prediction 

**Title (ZH)**: 多模态人类轨迹预测的 metamorphic 测试 

**Authors**: Helge Spieker, Nadjib Lazaar, Arnaud Gotlieb, Nassim Belmecheri  

**Link**: [PDF](https://arxiv.org/pdf/2509.01294)  

**Abstract**: Context: Predicting human trajectories is crucial for the safety and reliability of autonomous systems, such as automated vehicles and mobile robots. However, rigorously testing the underlying multimodal Human Trajectory Prediction (HTP) models, which typically use multiple input sources (e.g., trajectory history and environment maps) and produce stochastic outputs (multiple possible future paths), presents significant challenges. The primary difficulty lies in the absence of a definitive test oracle, as numerous future trajectories might be plausible for any given scenario. Objectives: This research presents the application of Metamorphic Testing (MT) as a systematic methodology for testing multimodal HTP systems. We address the oracle problem through metamorphic relations (MRs) adapted for the complexities and stochastic nature of HTP. Methods: We present five MRs, targeting transformations of both historical trajectory data and semantic segmentation maps used as an environmental context. These MRs encompass: 1) label-preserving geometric transformations (mirroring, rotation, rescaling) applied to both trajectory and map inputs, where outputs are expected to transform correspondingly. 2) Map-altering transformations (changing semantic class labels, introducing obstacles) with predictable changes in trajectory distributions. We propose probabilistic violation criteria based on distance metrics between probability distributions, such as the Wasserstein or Hellinger distance. Conclusion: This study introduces tool, a MT framework for the oracle-less testing of multimodal, stochastic HTP systems. It allows for assessment of model robustness against input transformations and contextual changes without reliance on ground-truth trajectories. 

**Abstract (ZH)**: 基于元模型测试的无注解多模态人类轨迹预测系统测试方法 

---
# An AI-Based Shopping Assistant System to Support the Visually Impaired 

**Title (ZH)**: 基于AI的购物助理系统以支持视力障碍者 

**Authors**: Larissa R. de S. Shibata, Ankit A. Ravankar, Jose Victorio Salazar Luces, Yasuhisa Hirata  

**Link**: [PDF](https://arxiv.org/pdf/2509.01246)  

**Abstract**: Shopping plays a significant role in shaping consumer identity and social integration. However, for individuals with visual impairments, navigating in supermarkets and identifying products can be an overwhelming and challenging experience. This paper presents an AI-based shopping assistant prototype designed to enhance the autonomy and inclusivity of visually impaired individuals in supermarket environments. The system integrates multiple technologies, including computer vision, speech recognition, text-to-speech synthesis, and indoor navigation, into a single, user-friendly platform. Using cameras for ArUco marker detection and real-time environmental scanning, the system helps users navigate the store, identify product locations, provide real-time auditory guidance, and gain context about their surroundings. The assistant interacts with the user through voice commands and multimodal feedback, promoting a more dynamic and engaging shopping experience. The system was evaluated through experiments, which demonstrated its ability to guide users effectively and improve their shopping experience. This paper contributes to the development of inclusive AI-driven assistive technologies aimed at enhancing accessibility and user independence for the shopping experience. 

**Abstract (ZH)**: 视觉障碍人士超市购物辅助系统的AI原型设计与实现 

---
# Symbolic Planning and Multi-Agent Path Finding in Extremely Dense Environments with Movable Obstacles 

**Title (ZH)**: 符号规划与移动障碍物条件下极度密集环境中的多Agent路径寻找 

**Authors**: Bo Fu, Zhe Chen, Rahul Chandan, Alex Barbosa, Michael Caldara, Joey Durham, Federico Pecora  

**Link**: [PDF](https://arxiv.org/pdf/2509.01022)  

**Abstract**: We introduce the Block Rearrangement Problem (BRaP), a challenging component of large warehouse management which involves rearranging storage blocks within dense grids to achieve a target state. We formally define the BRaP as a graph search problem. Building on intuitions from sliding puzzle problems, we propose five search-based solution algorithms, leveraging joint configuration space search, classical planning, multi-agent pathfinding, and expert heuristics. We evaluate the five approaches empirically for plan quality and scalability. Despite the exponential relation between search space size and block number, our methods demonstrate efficiency in creating rearrangement plans for deeply buried blocks in up to 80x80 grids. 

**Abstract (ZH)**: 密集网格中存储块重排问题（BRaP）的研究 

---
# ER-LoRA: Effective-Rank Guided Adaptation for Weather-Generalized Depth Estimation 

**Title (ZH)**: ER-LoRA: 基于有效秩引导的天气通用深度估计适应性方法 

**Authors**: Weilong Yan, Xin Zhang, Robby T. Tan  

**Link**: [PDF](https://arxiv.org/pdf/2509.00665)  

**Abstract**: Monocular depth estimation under adverse weather conditions (e.g.\ rain, fog, snow, and nighttime) remains highly challenging due to the lack of reliable ground truth and the difficulty of learning from unlabeled real-world data. Existing methods often rely on synthetic adverse data with pseudo-labels, which suffer from domain gaps, or employ self-supervised learning, which violates photometric assumptions in adverse scenarios. In this work, we propose to achieve weather--generalized depth estimation by Parameter--Efficient Fine--Tuning (PEFT) of Vision Foundation Models (VFMs), using only a small amount of high--visibility (normal) data. While PEFT has shown strong performance in semantic tasks such as segmentation, it remains underexplored for geometry--centric tasks like depth estimation -- especially in terms of balancing effective adaptation with the preservation of pretrained knowledge. To this end, we introduce the Selecting--Tuning--Maintaining (STM) strategy, which structurally decomposes the pretrained weights of VFMs based on two kinds of effective ranks (entropy--rank and stable--rank). In the tuning phase, we adaptively select the proper rank number as well as the task--aware singular directions for initialization, based on the entropy--rank and full--tuned weight; while in the maintaining stage, we enforce a principal direction regularization based on the stable--rank. This design guarantees flexible task adaptation while preserving the strong generalization capability of the pretrained VFM. Extensive experiments on four real--world benchmarks across diverse weather conditions demonstrate that STM not only outperforms existing PEFT methods and full fine--tuning but also surpasses methods trained with adverse synthetic data, and even the depth foundation model 

**Abstract (ZH)**: 单目深度估计在恶劣天气条件（如雨、雾、雪和夜间）下仍极具挑战性，由于缺乏可靠的地面真实数据和从未标记的真实世界数据中学习的难度。现有方法通常依赖合成的恶劣数据和伪标签，这会导致域间差异，或者采用自监督学习，但在恶劣场景下这会违反光度假设。在这项工作中，我们提出通过Vision Foundation Models (VFMs)的参数高效微调（PEFT）来实现天气通用的深度估计，仅使用少量高可见度（正常）数据。虽然PEFT在语义任务（如分割）中表现出色，但在几何中心任务（如深度估计）中的探索仍处于初级阶段，特别地，在有效适应和预训练知识保留间取得平衡方面尚未得到充分探索。为此，我们引入了Selecting-Tuning-Maintaining（STM）策略，该策略基于两种有效秩（熵秩和稳定秩）对VFMs的预训练权重进行结构分解。在微调阶段，我们基于熵秩和全微调权重自适应地选择合适的秩数以及任务感知的奇异方向进行初始化；而在维护阶段，我们基于稳定秩施加主方向正则化。这种设计确保了任务的灵活适应能力，同时保留了预训练VFMs的强大泛化能力。在四种不同天气条件的现实世界基准上的广泛实验表明，STM不仅优于现有的PEFT方法和全微调方法，还超越了使用恶劣合成数据训练的方法，甚至超过了深度基础模型。 

---
# Harnessing ADAS for Pedestrian Safety: A Data-Driven Exploration of Fatality Reduction 

**Title (ZH)**: 基于ADAS提升行人安全：一种数据驱动的致命事故减少探索 

**Authors**: Methusela Sulle, Judith Mwakalonge, Gurcan Comert, Saidi Siuhi, Nana Kankam Gyimah  

**Link**: [PDF](https://arxiv.org/pdf/2509.00048)  

**Abstract**: Pedestrian fatalities continue to rise in the United States, driven by factors such as human distraction, increased vehicle size, and complex traffic environments. Advanced Driver Assistance Systems (ADAS) offer a promising avenue for improving pedestrian safety by enhancing driver awareness and vehicle responsiveness. This study conducts a comprehensive data-driven analysis utilizing the Fatality Analysis Reporting System (FARS) to quantify the effectiveness of specific ADAS features like Pedestrian Automatic Emergency Braking (PAEB), Forward Collision Warning (FCW), and Lane Departure Warning (LDW), in lowering pedestrian fatalities. By linking vehicle specifications with crash data, we assess how ADAS performance varies under different environmental and behavioral conditions, such as lighting, weather, and driver/pedestrian distraction. Results indicate that while ADAS can reduce crash severity and prevent some fatalities, its effectiveness is diminished in low-light and adverse weather. The findings highlight the need for enhanced sensor technologies and improved driver education. This research informs policymakers, transportation planners, and automotive manufacturers on optimizing ADAS deployment to improve pedestrian safety and reduce traffic-related deaths. 

**Abstract (ZH)**: 行人死亡率在美国持续上升，受人为分心、车辆尺寸增加及复杂交通环境等因素驱动。高级驾驶辅助系统（ADAS）通过增强驾驶员意识和车辆反应性，为提高行人安全提供了前景广阔的方法。本研究利用致命事故报告系统（FARS）进行全面的数据驱动分析，量化行人自动紧急制动（PAEB）、前方碰撞预警（FCW）和车道偏离预警（LDW）等特定ADAS功能在降低行人死亡率方面的有效性。通过将车辆规格与碰撞数据相链接，我们评估了在不同环境和行为条件下（如光照、天气和驾驶员/行人的分心）ADAS的性能差异。研究结果表明，虽然ADAS可以减轻碰撞严重程度并预防某些死亡事件，但在低光照和不利天气条件下其有效性减弱。研究结果强调了增强传感器技术和改善驾驶员教育的必要性。本研究为政策制定者、交通规划者和汽车制造商提供了优化ADAS部署以提高行人安全和减少交通相关死亡的信息。 

---
# Curve-based slicer for multi-axis DLP 3D printing 

**Title (ZH)**: 基于曲线的多轴DLP 3D打印切割器 

**Authors**: Chengkai Dai, Tao Liu, Dezhao Guo, Binzhi Sun, Guoxin Fang, Yeung Yam, Charlie C.L. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00040)  

**Abstract**: This paper introduces a novel curve-based slicing method for generating planar layers with dynamically varying orientations in digital light processing (DLP) 3D printing. Our approach effectively addresses key challenges in DLP printing, such as regions with large overhangs and staircase artifacts, while preserving its intrinsic advantages of high resolution and fast printing speeds. We formulate the slicing problem as an optimization task, in which parametric curves are computed to define both the slicing layers and the model partitioning through their tangent planes. These curves inherently define motion trajectories for the build platform and can be optimized to meet critical manufacturing objectives, including collision-free motion and floating-free deposition. We validate our method through physical experiments on a robotic multi-axis DLP printing setup, demonstrating that the optimized curves can robustly guide smooth, high-quality fabrication of complex geometries. 

**Abstract (ZH)**: 本文介绍了一种基于曲线的切片方法，用于在数字光处理（DLP）三维打印中生成具有动态变化方向的平面层。该方法有效解决了DLP打印中的关键挑战，如大面积悬臂结构和阶梯状缺陷，同时保持其固有的高分辨率和快速打印速度的优势。我们将切片问题形式化为优化任务，通过计算参数化曲线来定义切片层和模型分区的切平面。这些曲线天然定义了构建平台的运动轨迹，并可优化以满足无碰撞运动和无浮置沉积等关键制造目标。我们在一台机械多轴DLP打印设备上进行物理实验，验证了该方法，结果显示优化后的曲线能够稳健地引导复杂几何结构的高质量、平滑制造。 

---
# AppCopilot: Toward General, Accurate, Long-Horizon, and Efficient Mobile Agent 

**Title (ZH)**: AppCopilot: 向通用、准确、长期和高效的移动代理方向努力 

**Authors**: Jingru Fan, Yufan Dang, Jingyao Wu, Huatao Li, Runde Yang, Xiyuan Yang, Yuheng Wang, Zhong Zhang, Yaxi Lu, Yankai Lin, Zhiyuan Liu, Dahai Li, Chen Qian  

**Link**: [PDF](https://arxiv.org/pdf/2509.02444)  

**Abstract**: With the raid evolution of large language models and multimodal foundation models, the mobile-agent landscape has proliferated without converging on the fundamental challenges. This paper identifies four core problems that must be solved for mobile agents to deliver practical, scalable impact: (1) generalization across tasks, modalities, apps, and devices; (2) accuracy, specifically precise on-screen interaction and click targeting; (3) long-horizon capability for sustained, multi-step goals; and (4) efficiency, specifically high-performance runtime on resource-constrained devices. We present AppCopilot, a multimodal, multi-agent, general-purpose on-device assistant that operates across applications and constitutes a full-stack, closed-loop system from data to deployment. AppCopilot operationalizes this position through an end-to-end autonomous pipeline spanning data collection, training, deployment, high-quality and efficient inference, and mobile application development. At the model layer, it integrates multimodal foundation models with robust Chinese-English support. At the reasoning and control layer, it combines chain-of-thought reasoning, hierarchical task planning and decomposition, and multi-agent collaboration. At the execution layer, it enables user personalization and experiential adaptation, voice interaction, function calling, cross-app and cross-device orchestration, and comprehensive mobile app support. The system design incorporates profiling-driven optimization for latency, memory, and energy across heterogeneous hardware. Empirically, AppCopilot achieves significant improvements along all four dimensions: stronger generalization, higher-precision on-screen actions, more reliable long-horizon task completion, and faster, more resource-efficient runtime. 

**Abstract (ZH)**: 随着大型语言模型和多模态基础模型的快速发展，移动代理的生态系统迅速增长但未解决核心挑战。本文识别了移动代理必须解决的四个核心问题，以实现实用且可扩展的影响：(1) 任务、模态、应用和设备间的通用化；(2) 准确性，特别是精确的屏幕交互和点击目标；(3) 长期能力，用于持续的多步目标；(4) 效率，特别是在资源受限的设备上实现高性能运行时。我们提出了AppCopilot，这是一种多模态、多代理、通用的设备上助手，可以在应用程序之间操作，并构成从数据到部署的全流程、闭环系统。AppCopilot 通过跨数据收集、训练、部署、高质量高效推理和移动应用开发的端到端自主管道来实现这一立场。在模型层，它将多模态基础模型与 robust 的中英文支持相结合。在推理和控制层，它结合了链式推理、层次任务规划与分解以及多代理协作。在执行层，它支持用户个性化和体验适配、语音交互、功能调用、跨应用和跨设备的编排以及全面的移动应用支持。系统设计跨异构硬件进行剖析驱动的优化，以实现在延迟、内存和能耗方面的显著改进。实证研究显示，AppCopilot 在所有四个维度上取得了显著改进：更强的通用化、更精确的屏幕动作、更可靠的长期任务完成以及更快、更节能的运行时。 

---
# When Agents go Astray: Course-Correcting SWE Agents with PRMs 

**Title (ZH)**: 当代理偏离正轨：通过PRMs校正SWE代理 

**Authors**: Shubham Gandhi, Jason Tsay, Jatin Ganhotra, Kiran Kate, Yara Rizk  

**Link**: [PDF](https://arxiv.org/pdf/2509.02360)  

**Abstract**: Large Language Model (LLM) agents are increasingly deployed for complex, multi-step software engineering (SWE) tasks. However, their trajectories often contain costly inefficiencies, such as redundant exploration, looping, and failure to terminate once a solution is reached. Prior work has largely treated these errors in a post-hoc manner, diagnosing failures only after execution. In this paper, we introduce SWE-PRM, an inference-time Process Reward Model (PRM) that intervenes during execution to detect and course-correct trajectory-level errors. Our PRM design leverages a taxonomy of common inefficiencies and delivers lightweight, interpretable feedback without modifying the underlying policy. On SWE-bench Verified, closed-source PRMs improve resolution from 40.0% to 50.6% (+10.6 p.p.), with the largest gains on medium and hard tasks. Among feedback strategies, taxonomy-guided PRMs outperform unguided or explicit action-prescriptive variants, increasing success rate while reducing trajectory length. These benefits come at an acceptable added inference cost of as low as $0.2, making PRMs a practical and scalable mechanism for improving SWE agents' reliability and efficiency. 

**Abstract (ZH)**: 大型语言模型（LLM）代理越来越多地被部署用于复杂多步骤的软件工程（SWE）任务。然而，它们的路径中往往包含昂贵的低效性，如冗余探索、循环以及达到解决方案后未能终止。先前的工作大多事后处理这些错误，只能在执行之后诊断失败。本文引入了SWE-PRM，一种推断时的过程奖励模型（PRM），可在执行过程中介入以检测和纠正路径级错误。我们的PRM设计利用了一种常见低效性的分类，并提供轻量级、可解释的反馈，而不修改基础策略。在SWE-bench Verified数据集上，封闭源代码的PRM将解决率从40.0%提高到50.6%（提高10.6个百分点），特别是在中等和困难任务上收益最大。在反馈策略中，基于分类的PRM优于无引导或显式行动 prescribe 的变体，提高了成功率同时减少了路径长度。这些益处带来的额外推理成本可低至0.2元，使PRM成为提高SWE代理可靠性和效率的实用且可扩展的方法。 

---
# Explainability-Driven Dimensionality Reduction for Hyperspectral Imaging 

**Title (ZH)**: 基于可解释性的高光谱成像降维方法 

**Authors**: Salma Haidar, José Oramas  

**Link**: [PDF](https://arxiv.org/pdf/2509.02340)  

**Abstract**: Hyperspectral imaging (HSI) provides rich spectral information for precise material classification and analysis; however, its high dimensionality introduces a computational burden and redundancy, making dimensionality reduction essential. We present an exploratory study into the application of post-hoc explainability methods in a model--driven framework for band selection, which reduces the spectral dimension while preserving predictive performance. A trained classifier is probed with explanations to quantify each band's contribution to its decisions. We then perform deletion--insertion evaluations, recording confidence changes as ranked bands are removed or reintroduced, and aggregate these signals into influence scores. Selecting the highest--influence bands yields compact spectral subsets that maintain accuracy and improve efficiency. Experiments on two public benchmarks (Pavia University and Salinas) demonstrate that classifiers trained on as few as 30 selected bands match or exceed full--spectrum baselines while reducing computational requirements. The resulting subsets align with physically meaningful, highly discriminative wavelength regions, indicating that model--aligned, explanation-guided band selection is a principled route to effective dimensionality reduction for HSI. 

**Abstract (ZH)**: 基于模型驱动框架的后验可解释性方法在带宽选择中的探索性研究：紧凑光谱子集的有效维度缩减方法 

---
# Exploring Diffusion Models for Generative Forecasting of Financial Charts 

**Title (ZH)**: 探索扩散模型在金融图表生成性预测中的应用 

**Authors**: Taegyeong Lee, Jiwon Park, Kyunga Bang, Seunghyun Hwang, Ung-Jin Jang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02308)  

**Abstract**: Recent advances in generative models have enabled significant progress in tasks such as generating and editing images from text, as well as creating videos from text prompts, and these methods are being applied across various fields. However, in the financial domain, there may still be a reliance on time-series data and a continued focus on transformer models, rather than on diverse applications of generative models. In this paper, we propose a novel approach that leverages text-to-image model by treating time-series data as a single image pattern, thereby enabling the prediction of stock price trends. Unlike prior methods that focus on learning and classifying chart patterns using architectures such as ResNet or ViT, we experiment with generating the next chart image from the current chart image and an instruction prompt using diffusion models. Furthermore, we introduce a simple method for evaluating the generated chart image against ground truth image. We highlight the potential of leveraging text-to-image generative models in the financial domain, and our findings motivate further research to address the current limitations and expand their applicability. 

**Abstract (ZH)**: 最近生成模型的进展在图像生成和编辑、基于文本提示创建视频等方面取得了显著进展，并且这些方法在多个领域得到应用。然而，在金融领域，仍然可能依赖时间序列数据，并且继续专注于变压器模型，而不是生成模型的多样化应用。本文提出了一种新颖的方法，通过将时间序列数据视为单一图像模式利用文本到图像模型，从而预测股票价格趋势。与以往方法使用ResNet或ViT架构学习和分类图表模式不同，我们尝试使用扩散模型从当前图表图像和指令提示生成下一个图表图像。此外，我们引入了一种简单的生成图表图像与真实图像进行评估的方法。本文强调了在金融领域利用文本到图像生成模型的潜力，并且我们的发现激发了进一步研究以解决当前局限性和扩大其应用范围的动力。 

---
# Rewarding Explainability in Drug Repurposing with Knowledge Graphs 

**Title (ZH)**: 使用知识图谱奖励药物重定位的可解释性 

**Authors**: Susana Nunes, Samy Badreddine, Catia Pesquita  

**Link**: [PDF](https://arxiv.org/pdf/2509.02276)  

**Abstract**: Knowledge graphs (KGs) are powerful tools for modelling complex, multi-relational data and supporting hypothesis generation, particularly in applications like drug repurposing. However, for predictive methods to gain acceptance as credible scientific tools, they must ensure not only accuracy but also the capacity to offer meaningful scientific explanations. This paper presents a novel approach REx, for generating scientific explanations based in link prediction in knowledge graphs. It employs reward and policy mechanisms that consider desirable properties of scientific explanation to guide a reinforcement learning agent in the identification of explanatory paths within a KG. The approach further enriches explanatory paths with domain-specific ontologies, ensuring that the explanations are both insightful and grounded in established biomedical knowledge. We evaluate our approach in drug repurposing using three popular knowledge graph benchmarks. The results clearly demonstrate its ability to generate explanations that validate predictive insights against biomedical knowledge and that outperform the state-of-the-art approaches in predictive performance, establishing REx as a relevant contribution to advance AI-driven scientific discovery. 

**Abstract (ZH)**: 基于知识图谱链接预测的科学解释生成方法REx 

---
# AGI as Second Being: The Structural-Generative Ontology of Intelligence 

**Title (ZH)**: AGI作为第二自我：智能的结构生成本体论 

**Authors**: Maijunxian Wang, Ran Ji  

**Link**: [PDF](https://arxiv.org/pdf/2509.02089)  

**Abstract**: Artificial intelligence is often measured by the range of tasks it can perform. Yet wide ability without depth remains only an imitation. This paper proposes a Structural-Generative Ontology of Intelligence: true intelligence exists only when a system can generate new structures, coordinate them into reasons, and sustain its identity over time. These three conditions -- generativity, coordination, and sustaining -- define the depth that underlies real intelligence. Current AI systems, however broad in function, remain surface simulations because they lack this depth. Breadth is not the source of intelligence but the growth that follows from depth. If future systems were to meet these conditions, they would no longer be mere tools, but could be seen as a possible Second Being, standing alongside yet distinct from human existence. 

**Abstract (ZH)**: 人工智能的结构生成本体论：真正的智能唯有当系统能够生成新结构、协调这些结构为理由，并在时间中维持其身份时才存在。当前的AI系统尽管功能广泛，仍只是表面模拟，缺乏这种深度。广泛的功能不是智能的源泉，而是源于深度的增长。如果未来系统能够满足这些条件，它们将不再仅仅是工具，而是可以被视为与人类共存但又独立的第二种存在。 

---
# Generative KI für TA 

**Title (ZH)**: 生成式AI用于教学辅助 

**Authors**: Wolfgang Eppler, Reinhard Heil  

**Link**: [PDF](https://arxiv.org/pdf/2509.02053)  

**Abstract**: Many scientists use generative AI in their scientific work. People working in technology assessment (TA) are no exception. TA's approach to generative AI is twofold: on the one hand, generative AI is used for TA work, and on the other hand, generative AI is the subject of TA research. After briefly outlining the phenomenon of generative AI and formulating requirements for its use in TA, the following article discusses in detail the structural causes of the problems associated with it. Although generative AI is constantly being further developed, the structurally induced risks remain. The article concludes with proposed solutions and brief notes on their feasibility, as well as some examples of the use of generative AI in TA work. 

**Abstract (ZH)**: 许多科学家在其科研工作中使用生成式AI。科技评估（TA）工作者也不例外。TA 对生成式AI的处理方式是双管齐下的：一方面，生成式AI被用于TA工作；另一方面，生成式AI本身也是TA研究的对象。在简要概述生成式AI现象并提出其在TA中使用的要求之后，本文详细讨论了与之相关的问题的结构性成因。尽管生成式AI不断得到进一步的发展，但由结构性问题引发的风险仍然存在。文章最后提出了建议的解决方案，并简要说明其可行性，以及提供了一些生成式AI在TA工作中使用的例子。 

---
# Physics Supernova: AI Agent Matches Elite Gold Medalists at IPhO 2025 

**Title (ZH)**: 物理 supernova: AI 代理在第 2025 届国际物理奥赛中媲美金牌得主 

**Authors**: Jiahao Qiu, Jingzhe Shi, Xinzhe Juan, Zelin Zhao, Jiayi Geng, Shilong Liu, Hongru Wang, Sanfeng Wu, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01659)  

**Abstract**: Physics provides fundamental laws that describe and predict the natural world. AI systems aspiring toward more general, real-world intelligence must therefore demonstrate strong physics problem-solving abilities: to formulate and apply physical laws for explaining and predicting physical processes. The International Physics Olympiad (IPhO)--the world's most prestigious physics competition--offers a rigorous benchmark for this purpose. We introduce Physics Supernova, an AI agent system with superior physics problem-solving abilities that match elite IPhO gold medalists. In IPhO 2025 theory problems, Physics Supernova attains 23.5/30 points, ranking 14th of 406 contestants and surpassing the median performance of human gold medalists. We extensively analyzed Physics Supernova's capabilities and flexibility across diverse physics tasks. These results show that principled tool integration within agent systems can deliver competitive improvements in solving challenging science problems. The codes are available at this https URL. 

**Abstract (ZH)**: 物理学提供了描述和预测自然界基本规律。追求更广泛、更实用智能的AI系统必须展示出强大的物理问题解决能力：即能够表述和运用物理定律来解释和预测物理过程。国际物理学奥林匹克（IPhO）——世界上最顶尖的物理竞赛——为此提供了严格的基准。我们引入了具有卓越物理问题解决能力的AI代理系统Physics Supernova，其能力与精英IPhO金牌获得者相当。在2025年IPhO理论问题中，Physics Supernova获得23.5/30分，排名406名参赛者中的第14位，超过了人类金牌获得者的中位表现。我们对Physics Supernova在多种物理任务上的能力和灵活性进行了广泛分析。这些结果表明，在代理系统中进行原则性的工具集成可以为解决具有挑战性的科学问题带来竞争性的改进。代码可在以下链接获得：this https URL。 

---
# Structured AI Decision-Making in Disaster Management 

**Title (ZH)**: 结构化人工智能决策在灾害管理中的应用 

**Authors**: Julian Gerald Dcruz, Argyrios Zolotas, Niall Ross Greenwood, Miguel Arana-Catania  

**Link**: [PDF](https://arxiv.org/pdf/2509.01576)  

**Abstract**: With artificial intelligence (AI) being applied to bring autonomy to decision-making in safety-critical domains such as the ones typified in the aerospace and emergency-response services, there has been a call to address the ethical implications of structuring those decisions, so they remain reliable and justifiable when human lives are at stake. This paper contributes to addressing the challenge of decision-making by proposing a structured decision-making framework as a foundational step towards responsible AI. The proposed structured decision-making framework is implemented in autonomous decision-making, specifically within disaster management. By introducing concepts of Enabler agents, Levels and Scenarios, the proposed framework's performance is evaluated against systems relying solely on judgement-based insights, as well as human operators who have disaster experience: victims, volunteers, and stakeholders. The results demonstrate that the structured decision-making framework achieves 60.94% greater stability in consistently accurate decisions across multiple Scenarios, compared to judgement-based systems. Moreover, the study shows that the proposed framework outperforms human operators with a 38.93% higher accuracy across various Scenarios. These findings demonstrate the promise of the structured decision-making framework for building more reliable autonomous AI applications in safety-critical contexts. 

**Abstract (ZH)**: 将人工智能应用于航空航天和紧急响应服务等关键安全领域以实现自主决策，引发了对结构化决策伦理影响的关注，特别是在涉及人类生命安全的决策中保持可靠性和可辩护性。本文通过提出一个结构化决策框架，作为负责任人工智能的基础步骤，来应对决策挑战。该提出的结构化决策框架在灾难管理领域实现自主决策。通过引入启用代理、层级和情景等概念，研究对比了基于判断的系统和具有灾难经验的人类操作者（受害者、志愿者和利益相关者）的表现，证明了结构化决策框架在多个情景下实现了60.94%的一致准确决策稳定性提升，且在各种情景下的准确率比人类操作者高38.93%。这些发现展示了结构化决策框架在关键安全领域构建更可靠的人工智能应用中的潜力。 

---
# The Need for Verification in AI-Driven Scientific Discovery 

**Title (ZH)**: AI驱动的科学研究中验证的必要性 

**Authors**: Cristina Cornelio, Takuya Ito, Ryan Cory-Wright, Sanjeeb Dash, Lior Horesh  

**Link**: [PDF](https://arxiv.org/pdf/2509.01398)  

**Abstract**: Artificial intelligence (AI) is transforming the practice of science. Machine learning and large language models (LLMs) can generate hypotheses at a scale and speed far exceeding traditional methods, offering the potential to accelerate discovery across diverse fields. However, the abundance of hypotheses introduces a critical challenge: without scalable and reliable mechanisms for verification, scientific progress risks being hindered rather than being advanced. In this article, we trace the historical development of scientific discovery, examine how AI is reshaping established practices for scientific discovery, and review the principal approaches, ranging from data-driven methods and knowledge-aware neural architectures to symbolic reasoning frameworks and LLM agents. While these systems can uncover patterns and propose candidate laws, their scientific value ultimately depends on rigorous and transparent verification, which we argue must be the cornerstone of AI-assisted discovery. 

**Abstract (ZH)**: 人工智能（AI）正在变革科学实践。机器学习和大型语言模型（LLMs）可以在规模和速度上远超传统方法生成假设，为跨多种领域的发现加速带来潜力。然而，假设的丰富性也带来了关键挑战：缺乏可扩展和可靠验证机制时，科学进步可能会受阻而非推进。在本文中，我们追溯了科学发现的历史发展，探讨AI如何重塑科学发现的既有实践，并审视主要的方法，包括数据驱动方法、知识敏感神经架构、符号推理框架和LLM代理。尽管这些系统可以发现模式并提出候选定律，但它们的科学价值最终取决于严格的和透明的验证，我们认为这必须是AI辅助发现的基石。 

---
# Conformal Predictive Monitoring for Multi-Modal Scenarios 

**Title (ZH)**: 多模态场景的配准预测监控 

**Authors**: Francesca Cairoli, Luca Bortolussi, Jyotirmoy V. Deshmukh, Lars Lindemann, Nicola Paoletti  

**Link**: [PDF](https://arxiv.org/pdf/2509.01338)  

**Abstract**: We consider the problem of quantitative predictive monitoring (QPM) of stochastic systems, i.e., predicting at runtime the degree of satisfaction of a desired temporal logic property from the current state of the system. Since computational efficiency is key to enable timely intervention against predicted violations, several state-of-the-art QPM approaches rely on fast machine-learning surrogates to provide prediction intervals for the satisfaction values, using conformal inference to offer statistical guarantees. However, these QPM methods suffer when the monitored agent exhibits multi-modal dynamics, whereby certain modes may yield high satisfaction values while others critically violate the property. Existing QPM methods are mode-agnostic and so would yield overly conservative and uninformative intervals that lack meaningful mode-specific satisfaction information. To address this problem, we present GenQPM, a method that leverages deep generative models, specifically score-based diffusion models, to reliably approximate the probabilistic and multi-modal system dynamics without requiring explicit model access. GenQPM employs a mode classifier to partition the predicted trajectories by dynamical mode. For each mode, we then apply conformal inference to produce statistically valid, mode-specific prediction intervals. We demonstrate the effectiveness of GenQPM on a benchmark of agent navigation and autonomous driving tasks, resulting in prediction intervals that are significantly more informative (less conservative) than mode-agnostic baselines. 

**Abstract (ZH)**: 基于生成模型的定量预测监测（GenQPM） 

---
# FlashAdventure: A Benchmark for GUI Agents Solving Full Story Arcs in Diverse Adventure Games 

**Title (ZH)**: FlashAdventure: 用于解决多元冒险游戏完整故事弧的GUI代理基准测试 

**Authors**: Jaewoo Ahn, Junseo Kim, Heeseung Yun, Jaehyeon Son, Dongmin Park, Jaewoong Cho, Gunhee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.01052)  

**Abstract**: GUI agents powered by LLMs show promise in interacting with diverse digital environments. Among these, video games offer a valuable testbed due to their varied interfaces, with adventure games posing additional challenges through complex, narrative-driven interactions. Existing game benchmarks, however, lack diversity and rarely evaluate agents on completing entire storylines. To address this, we introduce FlashAdventure, a benchmark of 34 Flash-based adventure games designed to test full story arc completion and tackle the observation-behavior gap: the challenge of remembering and acting on earlier gameplay information. We also propose CUA-as-a-Judge, an automated gameplay evaluator, and COAST, an agentic framework leveraging long-term clue memory to better plan and solve sequential tasks. Experiments show current GUI agents struggle with full story arcs, while COAST improves milestone completion by bridging the observation-behavior gap. Nonetheless, a marked discrepancy between humans and best-performing agents warrants continued research efforts to narrow this divide. 

**Abstract (ZH)**: 由LLMs驱动的GUI代理在与多样化数字环境交互中展现出前景。在这些环境中，视频游戏因其多样的界面而成为有价值的测试平台，其中冒险游戏通过复杂的、以叙述驱动的交互提供了额外的挑战。然而，现有的游戏基准缺乏多样性，很少评估代理完成整个故事情节的能力。为此，我们引入了FlashAdventure，一个由34个基于Flash的冒险游戏组成的基准，旨在测试完整的故事情节完成情况，并解决观察-行为差距：记忆和应用早先游戏信息的挑战。我们还提出了CUA-as-a-Judge，一种自动化游戏评估器，以及COAST，一种利用长期线索记忆的代理框架，以更好地计划和解决顺序任务。实验表明，当前的GUI代理难以完成整个故事情节，而COAST通过解决观察-行为差距提高了里程碑完成率。然而，人类与表现最佳代理之间的显著差异需要继续进行研究以缩小这一差距。 

---
# Quantum-like Coherence Derived from the Interaction between Chemical Reaction and Its Environment 

**Title (ZH)**: 由化学反应与其环境相互作用衍生的量子相干性 

**Authors**: Yukio-Pegio Gunji, Andrew Adamatzky, Panagiotis Mougkogiannis, Andrei Khrenikov  

**Link**: [PDF](https://arxiv.org/pdf/2509.01021)  

**Abstract**: By uncovering the contrast between Artificial Intelligence and Natural-born Intelligence as a computational process, we define closed computing and open computing, and implement open computing within chemical reactions. This involves forming a mixture and invalidation of the computational process and the execution environment, which are logically distinct, and coalescing both to create a system that adjusts fluctuations. We model chemical reactions by considering the computation as the chemical reaction and the execution environment as the degree of aggregation of molecules that interact with the reactive environment. This results in a chemical reaction that progresses while repeatedly clustering and de-clustering, where concentration no longer holds significant meaning. Open computing is segmented into Token computing, which focuses on the individual behavior of chemical molecules, and Type computing, which focuses on normative behavior. Ultimately, both are constructed as an interplay between the two. In this system, Token computing demonstrates self-organizing critical phenomena, while Type computing exhibits quantum logic. Through their interplay, the recruitment of fluctuations is realized, giving rise to interactions between quantum logical subspaces corresponding to quantum coherence across different Hilbert spaces. As a result, spike waves are formed, enabling signal transmission. This occurrence may be termed quantum-like coherence, implying the source of enzymes responsible for controlling spike waves and biochemical rhythms. 

**Abstract (ZH)**: 通过揭示人工智能与生俱来智能作为计算过程的对比，我们定义了封闭计算和开放计算，并在化学反应中实现开放计算。这涉及形成混合物和计算过程及执行环境的无效化，两者逻辑上是独立的，并将其融合以创建一个能够调节波动的系统。我们通过将计算视为化学反应、执行环境视为分子与反应环境相互作用的聚集程度来建模化学反应。结果，化学反应在不断聚类和去聚类的过程中进展，其中浓度不再具有重大意义。开放计算分为标记计算，专注于化学分子的个体行为，和类型计算，专注于规范行为。最终，两者在彼此互动中构建。在这个系统中，标记计算展示了自组织临界现象，而类型计算表现出量子逻辑。通过它们的互动，波动的招募得以实现，从而在不同希尔伯特空间跨越量子相干的量子逻辑子空间之间产生交互。结果，形成了尖峰波，实现信号传输。这种现象可以称为量子似共性，暗示控制尖峰波和生物化学节律的酶的来源。 

---
# Robust Deep Monte Carlo Counterfactual Regret Minimization: Addressing Theoretical Risks in Neural Fictitious Self-Play 

**Title (ZH)**: 鲁棒深度蒙特卡洛反事实遗憾最小化：解决神经虚构自我博弈中的理论风险 

**Authors**: Zakaria El Jaafari  

**Link**: [PDF](https://arxiv.org/pdf/2509.00923)  

**Abstract**: Monte Carlo Counterfactual Regret Minimization (MCCFR) has emerged as a cornerstone algorithm for solving extensive-form games, but its integration with deep neural networks introduces scale-dependent challenges that manifest differently across game complexities. This paper presents a comprehensive analysis of how neural MCCFR component effectiveness varies with game scale and proposes an adaptive framework for selective component deployment. We identify that theoretical risks such as nonstationary target distribution shifts, action support collapse, variance explosion, and warm-starting bias have scale-dependent manifestation patterns, requiring different mitigation strategies for small versus large games. Our proposed Robust Deep MCCFR framework incorporates target networks with delayed updates, uniform exploration mixing, variance-aware training objectives, and comprehensive diagnostic monitoring. Through systematic ablation studies on Kuhn and Leduc Poker, we demonstrate scale-dependent component effectiveness and identify critical component interactions. The best configuration achieves final exploitability of 0.0628 on Kuhn Poker, representing a 60% improvement over the classical framework (0.156). On the more complex Leduc Poker domain, selective component usage achieves exploitability of 0.2386, a 23.5% improvement over the classical framework (0.3703) and highlighting the importance of careful component selection over comprehensive mitigation. Our contributions include: (1) a formal theoretical analysis of risks in neural MCCFR, (2) a principled mitigation framework with convergence guarantees, (3) comprehensive multi-scale experimental validation revealing scale-dependent component interactions, and (4) practical guidelines for deployment in larger games. 

**Abstract (ZH)**: 基于蒙特卡洛反事实遗憾最小化的人工智能扩展式博弈算法：神经网络集成的规模依赖挑战与适应性框架 

---
# Neuro-Symbolic Predictive Process Monitoring 

**Title (ZH)**: 神经符号预测过程监控 

**Authors**: Axel Mezini, Elena Umili, Ivan Donadello, Fabrizio Maria Maggi, Matteo Mancanelli, Fabio Patrizi  

**Link**: [PDF](https://arxiv.org/pdf/2509.00834)  

**Abstract**: This paper addresses the problem of suffix prediction in Business Process Management (BPM) by proposing a Neuro-Symbolic Predictive Process Monitoring (PPM) approach that integrates data-driven learning with temporal logic-based prior knowledge. While recent approaches leverage deep learning models for suffix prediction, they often fail to satisfy even basic logical constraints due to the absence of explicit integration of domain knowledge during training. We propose a novel method to incorporate Linear Temporal Logic over finite traces (LTLf) into the training process of autoregressive sequence predictors. Our approach introduces a differentiable logical loss function, defined using a soft approximation of LTLf semantics and the Gumbel-Softmax trick, which can be combined with standard predictive losses. This ensures the model learns to generate suffixes that are both accurate and logically consistent. Experimental evaluation on three real-world datasets shows that our method improves suffix prediction accuracy and compliance with temporal constraints. We also introduce two variants of the logic loss (local and global) and demonstrate their effectiveness under noisy and realistic settings. While developed in the context of BPM, our framework is applicable to any symbolic sequence generation task and contributes toward advancing Neuro-Symbolic AI. 

**Abstract (ZH)**: 基于神经符号预测的过程监控方法：将有限轨迹的线性时态逻辑集成到自回归序列预测训练中 

---
# Sharpe Ratio Optimization in Markov Decision Processes 

**Title (ZH)**: 马尔可夫决策过程中的夏普比率优化 

**Authors**: Shuai Ma, Guangwu Liu, Li Xia  

**Link**: [PDF](https://arxiv.org/pdf/2509.00793)  

**Abstract**: Sharpe ratio (also known as reward-to-variability ratio) is a widely-used metric in finance, which measures the additional return at the cost of per unit of increased risk (standard deviation of return). However, the optimization of Sharpe ratio in Markov decision processes (MDPs) is challenging, because there exist two difficulties hindering the application of dynamic programming. One is that dynamic programming does not work for fractional objectives, and the other is that dynamic programming is invalid for risk metrics. In this paper, we study the Sharpe ratio optimization in infinite-horizon MDPs, considering both the long-run average and discounted settings. We address the first challenge with the Dinkelbachs transform, which converts the Sharpe ratio objective to a mean-squared-variance (M2V) objective. It is shown that the M2V optimization and the original Sharpe ratio optimization share the same optimal policy when the risk-sensitive parameter is equal to the optimal Sharpe ratio. For the second challenge, we develop an iterative algorithm to solve the M2V optimization which is similar to a mean-variance optimization in MDPs. We iteratively solve the M2V problem and obtain the associated Sharpe ratio that is used to update the risk-sensitive parameter in the next iteration of M2V problems. We show that such a sequence of Sharpe ratios derived is monotonically increasing and converges to the optimal Sharpe ratio. For both average and discounted MDP settings, we develop a policy iteration procedure and prove its convergence to the optimum. Numerical experiments are conducted for validation. To the best of our knowledge, our approach is the first that solves the Sharpe ratio optimization in MDPs with dynamic programming type algorithms. We believe that the proposed algorithm can shed light on solving MDPs with other fractional objectives. 

**Abstract (ZH)**: Sharpe比例优化在马尔可夫决策过程中的动态规划方法 

---
# L-MARS -- Legal Multi-Agent Workflow with Orchestrated Reasoning and Agentic Search 

**Title (ZH)**: L-MARS -- 合法的多代理工作流与协调推理及代理人搜索 

**Authors**: Ziqi Wang, Boqin Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2509.00761)  

**Abstract**: We present L-MARS (Legal Multi-Agent Workflow with Orchestrated Reasoning and Agentic Search), a system that reduces hallucination and uncertainty in legal question answering through coordinated multi-agent reasoning and retrieval. Unlike single-pass retrieval-augmented generation (RAG), L-MARS decomposes queries into subproblems, issues targeted searches across heterogeneous sources (Serper web, local RAG, CourtListener case law), and employs a Judge Agent to verify sufficiency, jurisdiction, and temporal validity before answer synthesis. This iterative reasoning-search-verification loop maintains coherence, filters noisy evidence, and grounds answers in authoritative law. We evaluated L-MARS on LegalSearchQA, a new benchmark of 200 up-to-date multiple choice legal questions in 2025. Results show that L-MARS substantially improves factual accuracy, reduces uncertainty, and achieves higher preference scores from both human experts and LLM-based judges. Our work demonstrates that multi-agent reasoning with agentic search offers a scalable and reproducible blueprint for deploying LLMs in high-stakes domains requiring precise legal retrieval and deliberation. 

**Abstract (ZH)**: L-MARS（ Legal 多智能体工作流结合 orchestrated 原因分析与代理搜索） 

---
# On Verifiable Legal Reasoning: A Multi-Agent Framework with Formalized Knowledge Representations 

**Title (ZH)**: 可验证的法律推理：一种形式化知识表示的多代理框架 

**Authors**: Albert Sadowski, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2509.00710)  

**Abstract**: Legal reasoning requires both precise interpretation of statutory language and consistent application of complex rules, presenting significant challenges for AI systems. This paper introduces a modular multi-agent framework that decomposes legal reasoning into distinct knowledge acquisition and application stages. In the first stage, specialized agents extract legal concepts and formalize rules to create verifiable intermediate representations of statutes. The second stage applies this knowledge to specific cases through three steps: analyzing queries to map case facts onto the ontology schema, performing symbolic inference to derive logically entailed conclusions, and generating final answers using a programmatic implementation that operationalizes the ontological knowledge. This bridging of natural language understanding with symbolic reasoning provides explicit and verifiable inspection points, significantly enhancing transparency compared to end-to-end approaches. Evaluation on statutory tax calculation tasks demonstrates substantial improvements, with foundational models achieving 76.4\% accuracy compared to 18.8\% baseline performance, effectively narrowing the performance gap between reasoning and foundational models. These findings suggest that modular architectures with formalized knowledge representations can make sophisticated legal reasoning more accessible through computationally efficient models while enhancing consistency and explainability in AI legal reasoning, establishing a foundation for future research into more transparent, trustworthy, and effective AI systems for legal domain. 

**Abstract (ZH)**: 法律推理要求对法条语言进行精确解读并一致应用复杂的规则，为人工智能系统带来了重大挑战。本文介绍了一个模块化的多智能体框架，将法律推理分解为知识获取和应用的distinct阶段。第一阶段，专门的智能体提取法律概念并形式化规则，创建可验证的法规中间表示。第二阶段通过三个步骤将此知识应用于具体案例：分析查询将案件事实映射到本体架构，执行符号推理以推导出逻辑上必然的结论，以及使用基于程序的实现生成最终答案，该实现使本体知识得以操作化。这种将自然语言理解和符号推理相结合的方法提供了明确且可验证的检查点，相比端到端的方法显著提高了透明度。在法定税务计算任务上的评估表明，基础模型的准确率达到76.4%，而基准性能仅为18.8%，有效地缩小了推理能力和基础模型之间的性能差距。这些发现表明，带有形式化知识表示的模块化架构可以通过计算高效的模型使复杂的法律推理更具可访问性，同时增强人工智能法律推理的一致性和解释性，从而奠定了未来研究更透明、可信且有效的法律领域人工智能系统的基石。 

---
# NetGent: Agent-Based Automation of Network Application Workflows 

**Title (ZH)**: NetGent: 基于代理的网络应用工作流自动化 

**Authors**: Jaber Daneshamooz, Eugene Vuong, Laasya Koduru, Sanjay Chandrasekaran, Arpit Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2509.00625)  

**Abstract**: We present NetGent, an AI-agent framework for automating complex application workflows to generate realistic network traffic datasets. Developing generalizable ML models for networking requires data collection from network environments with traffic that results from a diverse set of real-world web applications. However, using existing browser automation tools that are diverse, repeatable, realistic, and efficient remains fragile and costly. NetGent addresses this challenge by allowing users to specify workflows as natural-language rules that define state-dependent actions. These abstract specifications are compiled into nondeterministic finite automata (NFAs), which a state synthesis component translates into reusable, executable code. This design enables deterministic replay, reduces redundant LLM calls through state caching, and adapts quickly when application interfaces change. In experiments, NetGent automated more than 50+ workflows spanning video-on-demand streaming, live video streaming, video conferencing, social media, and web scraping, producing realistic traffic traces while remaining robust to UI variability. By combining the flexibility of language-based agents with the reliability of compiled execution, NetGent provides a scalable foundation for generating the diverse, repeatable datasets needed to advance ML in networking. 

**Abstract (ZH)**: NetGent：一种自动化复杂应用工作流的AI代理框架，用于生成 realistic 网络流量数据集 

---
# Artificial Intelligence-Based Analysis of Ice Cream Melting Behavior Under Various Ingredients 

**Title (ZH)**: 基于人工智能的冰淇淋在不同配料条件下融化行为分析 

**Authors**: Zhang Lai Bin, Zhen Bin It  

**Link**: [PDF](https://arxiv.org/pdf/2509.00507)  

**Abstract**: The stability of ice cream during melting is a critical factor for consumer's acceptance and product quality. With the commonly added stabilizer to improve texture, structure and slower melting as the factors to analyze. This report explores the effects of locust bean gum, guar gum, maltodextrin, and carrageenan on the melting behavior of homemade ice cream. The main objective was to assess how these additives influence melting resistance and to identify a more cost-effective recipe formulation. Ice cream samples incorporating each additive were prepared and subjected to melting tests under controlled conditions. Timelapse recordings were used to capture and analyze the progression of melting over time. Python and OpenCV is used for process and analysis. Observations revealed that all samples retained a foam-like structure even after melting, suggesting the stabilizers contributed to the formation of a stable air-cell matrix. Furthermore, when the melted samples were re-frozen and subsequently melted again, they displayed increased sturdiness, indicating improved resilience of the ice cream structure. Comparative analysis of the different stabilizers highlighted variations in their effectiveness, with some offering stronger melting resistance and structural support than others. Overall, the findings provide insights into the functional roles of commonly used food additives in ice cream formulation. By evaluating both performance and cost, this study demonstrates the potential for developing recipes that balance durability with economic efficiency, contributing to practical applications in both small-scale and commercial ice cream production. 

**Abstract (ZH)**: 冰淇淋融化过程中的稳定性是影响消费者接受度和产品质量的关键因素。本报告通过分析改善质地、结构和减缓融化速度的常见稳定剂，探索大豆胶、瓜尔胶、麦芽糊精和海藻胶对自制冰淇淋融化行为的影响。主要目标是评估这些添加剂对融化抵抗性的影响，并确定一种更经济有效的配方。每种添加剂的样品均被制备并在控制条件下进行融化测试。延时拍摄被用来记录和分析融化过程。使用Python和OpenCV进行处理和分析。观察结果显示，所有样品即使在融化后仍保持泡沫状结构，表明稳定剂有助于形成稳定的空气细胞矩阵。此外，当融化的样品重新冷冻并再次融化时，显示出了增强的坚固性，表明冰淇淋结构的改进韧性。不同稳定剂的对比分析揭示了其效果的差异，某些添加剂提供了更强的融化抵抗性和结构支持。总体而言，这些发现为企业了解常用食品添加剂在冰淇淋配方中的功能作用提供了见解。通过评估性能和成本，本研究展示了开发平衡耐用性和经济效率的配方的潜力，为小规模生产和商业冰淇淋生产提供了实际应用。 

---
# Multi-Agent Data Visualization and Narrative Generation 

**Title (ZH)**: 多智能体数据可视化与叙事生成 

**Authors**: Anton Wolter, Georgios Vidalakis, Michael Yu, Ankit Grover, Vaishali Dhanoa  

**Link**: [PDF](https://arxiv.org/pdf/2509.00481)  

**Abstract**: Recent advancements in the field of AI agents have impacted the way we work, enabling greater automation and collaboration between humans and agents. In the data visualization field, multi-agent systems can be useful for employing agents throughout the entire data-to-communication pipeline. We present a lightweight multi-agent system that automates the data analysis workflow, from data exploration to generating coherent visual narratives for insight communication. Our approach combines a hybrid multi-agent architecture with deterministic components, strategically externalizing critical logic from LLMs to improve transparency and reliability. The system delivers granular, modular outputs that enable surgical modifications without full regeneration, supporting sustainable human-AI collaboration. We evaluated our system across 4 diverse datasets, demonstrating strong generalizability, narrative quality, and computational efficiency with minimal dependencies. 

**Abstract (ZH)**: 近年来，AI代理领域的最新进展改变了我们的工作方式，使得人类和代理之间的自动化和协作更为高效。在数据可视化领域，多代理系统可以在整个数据到通信管道中部署代理，发挥重要作用。我们提出了一种轻量级多代理系统，自动化的数据分析工作流程，从数据探索到生成连贯的视觉叙事以促进洞察交流。我们的方法结合了混合多代理架构与确定性组件，战略性地将关键逻辑外部化，以提高透明度和可靠性。该系统提供细粒度、模块化的输出，支持在不完全再生的情况下进行精确修改，促进可持续的人机协作。我们在4个不同的数据集中评估了该系统，展示了其强大的普适性、叙事质量和计算效率，且依赖性较低。 

---
# NEWSAGENT: Benchmarking Multimodal Agents as Journalists with Real-World Newswriting Tasks 

**Title (ZH)**: NEWSAGENT：将多模态 agents 作为记者进行现实世界新闻撰写的基准测试 

**Authors**: Yen-Che Chien, Kuang-Da Wang, Wei-Yao Wang, Wen-Chih Peng  

**Link**: [PDF](https://arxiv.org/pdf/2509.00446)  

**Abstract**: Recent advances in autonomous digital agents from industry (e.g., Manus AI and Gemini's research mode) highlight potential for structured tasks by autonomous decision-making and task decomposition; however, it remains unclear to what extent the agent-based systems can improve multimodal web data productivity. We study this in the realm of journalism, which requires iterative planning, interpretation, and contextual reasoning from multimodal raw contents to form a well structured news. We introduce NEWSAGENT, a benchmark for evaluating how agents can automatically search available raw contents, select desired information, and edit and rephrase to form a news article by accessing core journalistic functions. Given a writing instruction and firsthand data as how a journalist initiates a news draft, agents are tasked to identify narrative perspectives, issue keyword-based queries, retrieve historical background, and generate complete articles. Unlike typical summarization or retrieval tasks, essential context is not directly available and must be actively discovered, reflecting the information gaps faced in real-world news writing. NEWSAGENT includes 6k human-verified examples derived from real news, with multimodal contents converted to text for broad model compatibility. We evaluate open- and closed-sourced LLMs with commonly-used agentic frameworks on NEWSAGENT, which shows that agents are capable of retrieving relevant facts but struggling with planning and narrative integration. We believe that NEWSAGENT serves a realistic testbed for iterating and evaluating agent capabilities in terms of multimodal web data manipulation to real-world productivity. 

**Abstract (ZH)**: Recent advances in自主数字代理来自行业（例如Manus AI和Gemini的研究模式）强调了自主决策和任务分解在结构化任务中的潜力；然而，仍然不清楚基于代理的系统在多模态网络数据 productivity方面改善的程度。我们从新闻业的角度研究这个问题，新闻业需要从多模态原始内容进行迭代计划、解释和情境推理，以形成结构化的新闻报道。我们介绍了NEWSAGENT，这是一个用于评估代理如何自动搜索可用的原始内容、选择所需信息并编辑重写以通过访问核心新闻功能形成新闻文章的基准。给定写作指令和第一手数据作为记者开始编写新闻草案的方式，代理被要求识别叙述视角、基于关键词查询、检索历史背景并生成完整文章。与典型的总结或检索任务不同，关键背景信息未直接提供，必须主动发现，反映了在现实世界新闻写作中面临的资讯空白。NEWSAGENT 包含了6000个人工验证的示例，来源于实际新闻，多模态内容转换为文本以实现广泛的模型兼容性。我们使用常用的代理框架对开源和闭源的大规模语言模型进行了NEWSAGENT 的评估，结果显示代理能够检索相关事实，但在规划和叙述整合方面存在困难。我们相信NEWSAGENT 是一个现实的测试平台，用于迭代和评估代理在多模态网络数据操作到现实世界生产力方面的能力。 

---
# Virtual Group Knowledge and Group Belief in Topological Evidence Models (Extended Version) 

**Title (ZH)**: 拓扑证据模型中虚拟群体知识与群体信念的扩展版本 

**Authors**: Alexandru Baltag, Malvin Gattinger, Djanira Gomes  

**Link**: [PDF](https://arxiv.org/pdf/2509.00184)  

**Abstract**: We study notions of (virtual) group knowledge and group belief within multi-agent evidence models, obtained by extending the topological semantics of evidence-based belief and fallible knowledge from individuals to groups. We completely axiomatize and show the decidability of the logic of ("hard" and "soft") group evidence, and do the same for an especially interesting fragment of it: the logic of group knowledge and group belief. We also extend these languages with dynamic evidence-sharing operators, and completely axiomatize the corresponding logics, showing that they are co-expressive with their static bases. 

**Abstract (ZH)**: 我们研究多智能体证据模型中（虚拟）群体知识和群体信念的概念，这些模型通过将基于证据的信任和可错知识的拓扑语义从个体扩展到群体而获得。我们完整地公理化并证明了群体证据逻辑（包括“硬”和“软”）的可判定性，并对特别有趣的其子逻辑——群体知识和群体信念的逻辑进行了相同的操作。我们还扩展了这些语言，加入了动态证据共享操作符，并完整地公理化了相应的逻辑，证明它们与静态基础具有同等表达能力。 

---
# Optimizing Health Coverage in Ethiopia: A Learning-augmented Approach and Persistent Proportionality Under an Online Budget 

**Title (ZH)**: 优化埃塞俄比亚的健康覆盖：基于学习的方法和在线预算下的持久比例原则 

**Authors**: Davin Choo, Yohai Trabelsi, Fentabil Getnet, Samson Warkaye Lamma, Wondesen Nigatu, Kasahun Sime, Lisa Matay, Milind Tambe, Stéphane Verguet  

**Link**: [PDF](https://arxiv.org/pdf/2509.00135)  

**Abstract**: As part of nationwide efforts aligned with the United Nations' Sustainable Development Goal 3 on Universal Health Coverage, Ethiopia's Ministry of Health is strengthening health posts to expand access to essential healthcare services. However, only a fraction of this health system strengthening effort can be implemented each year due to limited budgets and other competing priorities, thus the need for an optimization framework to guide prioritization across the regions of Ethiopia. In this paper, we develop a tool, Health Access Resource Planner (HARP), based on a principled decision-support optimization framework for sequential facility planning that aims to maximize population coverage under budget uncertainty while satisfying region-specific proportionality targets at every time step. We then propose two algorithms: (i) a learning-augmented approach that improves upon expert recommendations at any single-step; and (ii) a greedy algorithm for multi-step planning, both with strong worst-case approximation estimation. In collaboration with the Ethiopian Public Health Institute and Ministry of Health, we demonstrated the empirical efficacy of our method on three regions across various planning scenarios. 

**Abstract (ZH)**: 在全国范围内，与联合国可持续发展目标3（普及健康覆盖）相一致，埃塞俄比亚卫生部正加强基层卫生设施，以扩大获取基本医疗服务的范围。但由于预算有限和其它竞争性优先事项，每年只能实施该卫生系统强化努力的一小部分，因此需要一个优化框架来指导埃塞俄比亚各地区的优先级排序。本文基于一个有原则的决策支持优化框架，开发了一个工具—健康接入资源规划器（HARP），旨在在预算不确定性下最大化人口覆盖率，同时满足每个时间步骤特有的比例目标。然后，我们提出了两种算法：（i）一种增强专家建议的机器学习方法，在任何单步骤上都能改进专家建议；（ii）一种贪婪算法，用于多步骤规划，且具有强大的最坏情况近似估计。在与埃塞俄比亚公共卫生研究所和卫生部合作下，我们在三个地区不同规划场景中验证了该方法的实证有效性。 

---
# MODE: Mixture of Document Experts for RAG 

**Title (ZH)**: Mixture of Document Experts for RAG 

**Authors**: Rahul Anand  

**Link**: [PDF](https://arxiv.org/pdf/2509.00100)  

**Abstract**: Retrieval-Augmented Generation (RAG) often relies on large vector databases and cross-encoders tuned for large-scale corpora, which can be excessive for small, domain-specific collections. We present MODE (Mixture of Document Experts), a lightweight alternative that replaces fine-grained nearest-neighbor search with cluster-and-route retrieval. Documents are embedded, grouped into semantically coherent clusters, and represented by cached centroids. At query time, we route to the top centroid(s) and retrieve context only within those clusters, eliminating external vector-database infrastructure and reranking while keeping latency low. On HotpotQA and SQuAD corpora with 100-500 chunks, MODE matches or exceeds a dense-retrieval baseline in answer quality while reducing end-to-end retrieval time. Ablations show that cluster granularity and multi-cluster routing control the recall/precision trade-off, and that tighter clusters improve downstream accuracy. MODE offers a practical recipe for small and medium corpora where simplicity, speed, and topical focus matter. 

**Abstract (ZH)**: 基于检索的生成（RAG）通常依赖于大型向量数据库和针对大规模语料库调整的跨编码器，但对于小型的领域特定集合来说可能过于冗余。我们提出了一种轻量级的替代方案MODE（文档专家混合），它用聚类并路由检索替代了精细粒度的最近邻搜索。文档被嵌入，并聚合成语义上连贯的聚类，由缓存的质心表示。在查询时，我们路由到顶级质心，并仅在那些聚类内部检索上下文，从而消除了外部向量数据库基础设施，减少了重新排名，并保持了低延迟。在包含100-500个片段的HotpotQA和SQuAD语料库上，MODE在答案质量上匹配或超过了密集检索基线，同时减少了端到端检索时间。消融实验显示，聚类粒度和多聚类路由控制召回率与精确率之间的权衡，并且更紧密的聚类提高了下游准确性。MODE为注重简洁性、速度和专题性的小型和中型语料库提供了一种实用的方法。 

---
# Entropy-Guided Loop: Achieving Reasoning through Uncertainty-Aware Generation 

**Title (ZH)**: 熵引导循环：通过不确定性 Awareness 生成实现推理 

**Authors**: Andrew G. A. Correa, Ana C. H de Matos  

**Link**: [PDF](https://arxiv.org/pdf/2509.00079)  

**Abstract**: Reasoning models often outperform smaller models but at 3--5$\times$ higher cost and added latency. We present entropy-guided refinement: a lightweight, test-time loop that uses token-level uncertainty to trigger a single, targeted refinement pass. We extract logprobs, compute Shannon entropy on top-$k$ alternatives, and apply a simple OR-logic trigger over perplexity, maximum token entropy, and low-confidence-token count. Unlike approaches that use entropy only for measurement or decoding, we pass a compact uncertainty report (tokens, confidences, alternatives, context) back to the model to guide corrective edits. On representative technical queries across reasoning, mathematics, and code generation tasks, a small model with our loop approaches 95\% of a reference reasoning model's quality at approximately one-third of the cost. The method achieves selective refinement on ~31\% of responses while improving accuracy by 16 percentage points over single-pass inference. We demonstrate that this uncertainty-aware loop provides an effective middle ground between single-pass inference and expensive reasoning chains, making it practical for production deployments where both quality and cost matter. 

**Abstract (ZH)**: 熵导向的精炼：一种轻量级的测试时循环，使用token级不确定性触发目标精炼-pass 

---
# A Comparative Study of Controllability, Explainability, and Performance in Dysfluency Detection Models 

**Title (ZH)**: 可控制性、解释性和性能在口吃检测模型中的比较研究 

**Authors**: Eric Zhang, Li Wei, Sarah Chen, Michael Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00058)  

**Abstract**: Recent advances in dysfluency detection have introduced a variety of modeling paradigms, ranging from lightweight object-detection inspired networks (YOLOStutter) to modular interpretable frameworks (UDM). While performance on benchmark datasets continues to improve, clinical adoption requires more than accuracy: models must be controllable and explainable. In this paper, we present a systematic comparative analysis of four representative approaches--YOLO-Stutter, FluentNet, UDM, and SSDM--along three dimensions: performance, controllability, and explainability. Through comprehensive evaluation on multiple datasets and expert clinician assessment, we find that YOLO-Stutter and FluentNet provide efficiency and simplicity, but with limited transparency; UDM achieves the best balance of accuracy and clinical interpretability; and SSDM, while promising, could not be fully reproduced in our experiments. Our analysis highlights the trade-offs among competing approaches and identifies future directions for clinically viable dysfluency modeling. We also provide detailed implementation insights and practical deployment considerations for each approach. 

**Abstract (ZH)**: 近年来，流畅性检测的进展引入了多种建模范式，从轻量级的目标检测网络（YOLOStutter）到模块化的可解释框架（UDM）。尽管在基准数据集上的性能不断提高，但在临床应用中，模型不仅需要准确，还需具备可控性和可解释性。在本文中，我们沿着性能、可控性和可解释性三个维度对四种代表方法——YOLO-Stutter、FluentNet、UDM和SSDM——进行了系统的比较分析。通过在多个数据集上的综合评估及专家临床评估，我们发现YOLO-Stutter和FluentNet在效率和简洁性方面表现出色，但透明度有限；UDM在准确性和临床可解释性之间取得了最佳平衡；而SSDM虽然有潜力，但在我们的实验中无法完全重现。我们的分析凸显了竞争方法之间的权衡取舍，并指出了临床可行的流畅性建模未来的研究方向。我们还提供了每种方法的详细实现见解和实际部署考虑。 

---
# Probabilistically stable revision and comparative probability: a representation theorem and applications 

**Title (ZH)**: 概率稳定修订与比较概率：一个表示定理及其应用 

**Authors**: Krzysztof Mierzewski  

**Link**: [PDF](https://arxiv.org/pdf/2509.02495)  

**Abstract**: The stability rule for belief, advocated by Leitgeb [Annals of Pure and Applied Logic 164, 2013], is a rule for rational acceptance that captures categorical belief in terms of $\textit{probabilistically stable propositions}$: propositions to which the agent assigns resiliently high credence. The stability rule generates a class of $\textit{probabilistically stable belief revision}$ operators, which capture the dynamics of belief that result from an agent updating their credences through Bayesian conditioning while complying with the stability rule for their all-or-nothing beliefs. In this paper, we prove a representation theorem that yields a complete characterisation of such probabilistically stable revision operators and provides a `qualitative' selection function semantics for the (non-monotonic) logic of probabilistically stable belief revision. Drawing on the theory of comparative probability orders, this result gives necessary and sufficient conditions for a selection function to be representable as a strongest-stable-set operator on a finite probability space. The resulting logic of probabilistically stable belief revision exhibits strong monotonicity properties while failing the AGM belief revision postulates and satisfying only very weak forms of case reasoning. In showing the main theorem, we prove two results of independent interest to the theory of comparative probability: the first provides necessary and sufficient conditions for the joint representation of a pair of (respectively, strict and non-strict) comparative probability orders. The second result provides a method for axiomatising the logic of ratio comparisons of the form ``event $A$ is at least $k$ times more likely than event $B$''. In addition to these measurement-theoretic applications, we point out two applications of our main result to the theory of simple voting games and to revealed preference theory. 

**Abstract (ZH)**: 信念的稳定性规则：Leitgeb [《纯粹与应用逻辑杂志》164卷，2013年] 提倡的信念稳定性规则是一种用概率上稳定的命题来捕捉理性接受的规则，这些命题是代理人赋予权重较高的命题。该稳定性规则生成了一类概率上稳定的信念修订操作符，这些操作符捕捉了代理人在遵循全部或不完全信念的稳定性规则时，通过贝叶斯条件化更新其信念动态所导致的信念变化。本文证明了一个表示定理，给出了此类概率上稳定的修订操作符的完全特征，并提供了概率上稳定信念修订逻辑的“定性”选择函数语义。基于比较概率秩序理论，这一结果给出了选择函数可以表示为有限概率空间上最强稳定集操作符的必要和充分条件。概率上稳定的信念修订逻辑表现出强烈的单调性特征，但不满足AGM信念修订公理，并仅满足非常弱形式的案例推理。在证明主要定理的过程中，我们证明了两个对比较概率理论具有独立兴趣的结果：第一个结果给出了一个关于（分别严格和不严格）比较概率秩序的联合表示的必要和充分条件；第二个结果提供了一种公理化比值比较形式“事件A至少是事件B的k倍可能性”的逻辑的方法。除了这些测量理论应用外，我们还指出，主要结果在简单投票理论和揭示偏好理论中也有两个应用。 

---
# Generative Sequential Notification Optimization via Multi-Objective Decision Transformers 

**Title (ZH)**: 基于多目标决策转换器的生成性序列通知优化 

**Authors**: Borja Ocejo, Ruofan Wang, Ke Liu, Rohit K. Patra, Haotian Shen, David Liu, Yiwen Yuan, Gokulraj Mohanasundaram, Fedor Borisyuk, Prakruthi Prabhakar  

**Link**: [PDF](https://arxiv.org/pdf/2509.02458)  

**Abstract**: Notifications are an important communication channel for delivering timely and relevant information. Optimizing their delivery involves addressing complex sequential decision-making challenges under constraints such as message utility and user fatigue. Offline reinforcement learning (RL) methods, such as Conservative Q-Learning (CQL), have been applied to this problem but face practical challenges at scale, including instability, sensitivity to distribution shifts, limited reproducibility, and difficulties with explainability in high-dimensional recommendation settings. We present a Decision Transformer (DT) based framework that reframes policy learning as return-conditioned supervised learning, improving robustness, scalability, and modeling flexibility. Our contributions include a real-world comparison with CQL, a multi-reward design suitable for non-episodic tasks, a quantile regression approach to return-to-go conditioning, and a production-ready system with circular buffer-based sequence processing for near-real-time inference. Extensive offline and online experiments in a deployed notification system show that our approach improves notification utility and overall session activity while minimizing user fatigue. Compared to a multi-objective CQL-based agent, the DT-based approach achieved a +0.72% increase in sessions for notification decision-making at LinkedIn by making notification recommendation more relevant. 

**Abstract (ZH)**: 基于决策变换器的推送通知优化框架：提高鲁棒性、可扩展性和建模灵活性 

---
# From Noisy Labels to Intrinsic Structure: A Geometric-Structural Dual-Guided Framework for Noise-Robust Medical Image Segmentation 

**Title (ZH)**: 从噪声标签到内在结构：一种几何-结构双导向的噪声鲁棒医学图像分割框架 

**Authors**: Tao Wang, Zhenxuan Zhang, Yuanbo Zhou, Xinlin Zhang, Yuanbin Chen, Tao Tan, Guang Yang, Tong Tong  

**Link**: [PDF](https://arxiv.org/pdf/2509.02419)  

**Abstract**: The effectiveness of convolutional neural networks in medical image segmentation relies on large-scale, high-quality annotations, which are costly and time-consuming to obtain. Even expert-labeled datasets inevitably contain noise arising from subjectivity and coarse delineations, which disrupt feature learning and adversely impact model performance. To address these challenges, this study propose a Geometric-Structural Dual-Guided Network (GSD-Net), which integrates geometric and structural cues to improve robustness against noisy annotations. It incorporates a Geometric Distance-Aware module that dynamically adjusts pixel-level weights using geometric features, thereby strengthening supervision in reliable regions while suppressing noise. A Structure-Guided Label Refinement module further refines labels with structural priors, and a Knowledge Transfer module enriches supervision and improves sensitivity to local details. To comprehensively assess its effectiveness, we evaluated GSD-Net on six publicly available datasets: four containing three types of simulated label noise, and two with multi-expert annotations that reflect real-world subjectivity and labeling inconsistencies. Experimental results demonstrate that GSD-Net achieves state-of-the-art performance under noisy annotations, achieving improvements of 2.52% on Kvasir, 22.76% on Shenzhen, 8.87% on BU-SUC, and 4.59% on BraTS2020 under SR simulated noise. The codes of this study are available at this https URL. 

**Abstract (ZH)**: 基于几何-结构双引导网络在嘈杂标注下医学图像分割的有效性 

---
# Real-time ML-based Defense Against Malicious Payload in Reconfigurable Embedded Systems 

**Title (ZH)**: 基于机器学习的实时防护方法以应对可重构嵌入式系统中的恶意载荷 

**Authors**: Rye Stahle-Smith, Rasha Karakchi  

**Link**: [PDF](https://arxiv.org/pdf/2509.02387)  

**Abstract**: The growing use of FPGAs in reconfigurable systems introducessecurity risks through malicious bitstreams that could cause denial-of-service (DoS), data leakage, or covert attacks. We investigated chip-level hardware malicious payload in embedded systems and proposed a supervised machine learning method to detect malicious bitstreams via static byte-level features. Our approach diverges from existing methods by analyzing bitstreams directly at the binary level, enabling real-time detection without requiring access to source code or netlists. Bitstreams were sourced from state-of-the-art (SOTA) benchmarks and re-engineered to target the Xilinx PYNQ-Z1 FPGA Development Board. Our dataset included 122 samples of benign and malicious configurations. The data were vectorized using byte frequency analysis, compressed using TSVD, and balanced using SMOTE to address class imbalance. The evaluated classifiers demonstrated that Random Forest achieved a macro F1-score of 0.97, underscoring the viability of real-time Trojan detection on resource-constrained systems. The final model was serialized and successfully deployed via PYNQ to enable integrated bitstream analysis. 

**Abstract (ZH)**: FPGAs在可重构系统中的广泛应用通过恶意比特流引入了安全风险，这些恶意比特流可能导致服务拒绝（DoS）、数据泄漏或隐蔽攻击。我们研究了嵌入式系统中的芯片级硬件恶意负载，并提出了通过静态字节级特征检测恶意比特流的监督机器学习方法。该方法不同于现有方法，直接在二进制级别分析比特流，从而实现无需访问源代码或网表即可进行实时检测。比特流来源于最先进的基准测试，并针对Xilinx PYNQ-Z1 FPGA开发板进行了重新工程。数据集包含122个良性与恶意配置样本。数据通过字节频率分析向量化，使用TSVD压缩，并使用SMOTE进行平衡以解决类别不平衡问题。评估的分类器显示，随机森林的方法获得了宏F1分数0.97，表明在资源受限系统中实时检测Trojan的可行性。最终模型被序列化并通过PYNQ成功部署，以实现集成比特流分析。 

---
# Guidance and Control Neural Network Acceleration using Memristors 

**Title (ZH)**: 使用 memristor 加速指导与控制神经网络 

**Authors**: Zacharia A. Rudge, Dario Izzo, Moritz Fieback, Anteneh Gebregiorgis, Said Hamdioui, Dominik Dold  

**Link**: [PDF](https://arxiv.org/pdf/2509.02369)  

**Abstract**: In recent years, the space community has been exploring the possibilities of Artificial Intelligence (AI), specifically Artificial Neural Networks (ANNs), for a variety of on board applications. However, this development is limited by the restricted energy budget of smallsats and cubesats as well as radiation concerns plaguing modern chips. This necessitates research into neural network accelerators capable of meeting these requirements whilst satisfying the compute and performance needs of the application. This paper explores the use of Phase-Change Memory (PCM) and Resistive Random-Access Memory (RRAM) memristors for on-board in-memory computing AI acceleration in space applications. A guidance and control neural network (G\&CNET) accelerated using memristors is simulated in a variety of scenarios and with both device types to evaluate the performance of memristor-based accelerators, considering device non-idealities such as noise and conductance drift. We show that the memristive accelerator is able to learn the expert actions, though challenges remain with the impact of noise on accuracy. We also show that re-training after degradation is able to restore performance to nominal levels. This study provides a foundation for future research into memristor-based AI accelerators for space, highlighting their potential and the need for further investigation. 

**Abstract (ZH)**: 近年来，太空社区一直在探索将人工智能（AI）、具体是人工神经网络（ANNs）应用于各种星载应用的可能性。然而，这一发展受限于小型卫星和立方星有限的能量预算以及困扰现代芯片的辐射问题。这就要求研究能够满足这些要求并满足应用计算和性能需求的神经网络加速器。本文探讨了相变记忆体（PCM）和电阻式随机存取记忆体（RRAM）忆阻器在太空应用中进行星载存内计算AI加速的应用。利用忆阻器加速的指导与控制神经网络（G&CNET）在多种场景下进行模拟，并使用两种器件类型进行评估，考察基于忆阻器的加速器的性能，考虑器件非理想性，如噪声和传导漂移的影响。结果显示，忆阻器加速器能够学习专家行动，尽管噪声对准确性的影响仍是一个挑战。此外，退化后的重新训练能够恢复性能至正常水平。本研究为未来基于忆阻器的AI加速器在太空领域的研究奠定了基础，凸显了它们的潜力及进一步研究的必要性。 

---
# AudioCodecBench: A Comprehensive Benchmark for Audio Codec Evaluation 

**Title (ZH)**: AudioCodecBench：音频编码器评估的全面基准 

**Authors**: Lu Wang, Hao Chen, Siyu Wu, Zhiyue Wu, Hao Zhou, Chengfeng Zhang, Ting Wang, Haodi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02349)  

**Abstract**: Multimodal Large Language Models (MLLMs) have been widely applied in speech and music. This tendency has led to a focus on audio tokenization for Large Models (LMs). Unlike semantic-only text tokens, audio tokens must both capture global semantic content and preserve fine-grained acoustic details. Moreover, they provide a discrete method for speech and music that can be effectively integrated into MLLMs. However, existing research is unsuitable in the definitions of semantic tokens and acoustic tokens. In addition, the evaluation of different codecs typically concentrates on specific domains or tasks, such as reconstruction or Automatic Speech Recognition (ASR) task, which prevents fair and comprehensive comparisons. To address these problems, this paper provides suitable definitions for semantic and acoustic tokens and introduces a systematic evaluation framework. This framework allows for a comprehensive assessment of codecs' capabilities which evaluate across four dimensions: audio reconstruction metric, codebook index (ID) stability, decoder-only transformer perplexity, and performance on downstream probe tasks. Our results show the correctness of the provided suitable definitions and the correlation among reconstruction metrics, codebook ID stability, downstream probe tasks and perplexity. 

**Abstract (ZH)**: 多模态大语言模型（MLLMs）在语音和音乐领域的广泛应用促使了对大型模型（LMs）的音频分词研究。与仅关注语义的文本标记不同，音频标记必须同时捕捉全局语义内容并保留细微的声学细节。此外，它们提供了一种离散的方法，可以有效整合到MLLMs中。然而，现有研究在语义标记和声学标记的定义上存在不足。此外，对不同编码器-解码器（codecs）的评估通常集中在特定领域或任务上，如重构或自动语音识别（ASR）任务，这阻碍了公平和全面的比较。为解决这些问题，本文提供了语义和声学标记的合适定义，并介绍了系统评价框架。该框架从四个维度对编码器-解码器的能力进行全面评估：声学重建度量、码本索引（ID）稳定性、解码器变换器的困惑度以及下游探测任务的表现。我们的结果表明提供的合适定义的正确性，并揭示了重建度量、码本ID稳定性、下游探测任务和困惑度之间的相关性。 

---
# RDIT: Residual-based Diffusion Implicit Models for Probabilistic Time Series Forecasting 

**Title (ZH)**: 基于残差的扩散隐式模型：用于概率时间序列预测 

**Authors**: Chih-Yu Lai, Yu-Chien Ning, Duane S. Boning  

**Link**: [PDF](https://arxiv.org/pdf/2509.02341)  

**Abstract**: Probabilistic Time Series Forecasting (PTSF) plays a critical role in domains requiring accurate and uncertainty-aware predictions for decision-making. However, existing methods offer suboptimal distribution modeling and suffer from a mismatch between training and evaluation metrics. Surprisingly, we found that augmenting a strong point estimator with a zero-mean Gaussian, whose standard deviation matches its training error, can yield state-of-the-art performance in PTSF. In this work, we propose RDIT, a plug-and-play framework that combines point estimation and residual-based conditional diffusion with a bidirectional Mamba network. We theoretically prove that the Continuous Ranked Probability Score (CRPS) can be minimized by adjusting to an optimal standard deviation and then derive algorithms to achieve distribution matching. Evaluations on eight multivariate datasets across varied forecasting horizons demonstrate that RDIT achieves lower CRPS, rapid inference, and improved coverage compared to strong baselines. 

**Abstract (ZH)**: 概率时间序列预测（PTSF）在需要准确和不确定性aware预测以辅助决策的领域中发挥着关键作用。然而，现有的方法在分布建模方面表现不佳，并且存在训练和评估度量之间的不匹配。令人惊讶的是，我们发现将一个强大的点估计器与一个零均值的高斯模型结合，其标准差匹配其训练误差，可以在PTSF中达到最先进的性能。在本文中，我们提出了一种插件式框架RDIT，该框架结合了点估计、基于残差的条件扩散与双向Mamba网络。我们理论证明可以通过调整到最优标准差来最小化连续排名概率得分（CRPS），并推导了实现分布匹配的算法。在八个跨学科的时间序列数据集上的评估表明，RDIT在不同预测_horizon下实现了更低的CRPS、更快的推理速度和更好的覆盖范围，优于强壮的基础模型。 

---
# Look: AI at Work! - Analysing Key Aspects of AI-support at the Work Place 

**Title (ZH)**: Look: AI在工作场所的应用！- 分析AI支持的关键方面 

**Authors**: Stefan Schiffer, Anna Milena Rothermel, Alexander Ferrein, Astrid Rosenthal-von der Pütten  

**Link**: [PDF](https://arxiv.org/pdf/2509.02274)  

**Abstract**: In this paper we present an analysis of technological and psychological factors of applying artificial intelligence (AI) at the work place. We do so for a number of twelve application cases in the context of a project where AI is integrated at work places and in work systems of the future. From a technological point of view we mainly look at the areas of AI that the applications are concerned with. This allows to formulate recommendations in terms of what to look at in developing an AI application and what to pay attention to with regards to building AI literacy with different stakeholders using the system. This includes the importance of high-quality data for training learning-based systems as well as the integration of human expertise, especially with knowledge-based systems. In terms of the psychological factors we derive research questions to investigate in the development of AI supported work systems and to consider in future work, mainly concerned with topics such as acceptance, openness, and trust in an AI system. 

**Abstract (ZH)**: 本文分析了在工作场所应用人工智能（AI）的技术和心理因素，并在此基础上对一个项目中的十二个应用案例进行了研究。从技术角度来看，我们主要关注应用涉及的AI领域，以提出开发AI应用的建议，并在不同利益相关者使用系统时关注AI素养的培养。这包括高质量数据在训练基于学习的系统中的重要性，以及人类专业知识在基于知识系统的整合。从心理因素来看，我们提出了关于在AI支持的工作系统开发中需要研究的问题和需考虑的方面，主要关注接受度、开放性和对AI系统的信任等主题。 

---
# VariAntNet: Learning Decentralized Control of Multi-Agent Systems 

**Title (ZH)**: VariAntNet: 学习多Agent系统去中心化控制 

**Authors**: Yigal Koifman, Erez Koifman, Eran Iceland, Ariel Barel, Alfred M. Bruckstein  

**Link**: [PDF](https://arxiv.org/pdf/2509.02271)  

**Abstract**: A simple multi-agent system can be effectively utilized in disaster response applications, such as firefighting. Such a swarm is required to operate in complex environments with limited local sensing and no reliable inter-agent communication or centralized control. These simple robotic agents, also known as Ant Robots, are defined as anonymous agents that possess limited sensing capabilities, lack a shared coordinate system, and do not communicate explicitly with one another. A key challenge for simple swarms lies in maintaining cohesion and avoiding fragmentation despite limited-range sensing. Recent advances in machine learning offer effective solutions to some of the classical decentralized control challenges. We propose VariAntNet, a deep learning-based decentralized control model designed to facilitate agent swarming and collaborative task execution. VariAntNet includes geometric features extraction from unordered, variable-sized local observations. It incorporates a neural network architecture trained with a novel, differentiable, multi-objective, mathematically justified loss function that promotes swarm cohesiveness by utilizing the properties of the visibility graph Laplacian matrix. VariAntNet is demonstrated on the fundamental multi-agent gathering task, where agents with bearing-only and limited-range sensing must gather at some location. VariAntNet significantly outperforms an existing analytical solution, achieving more than double the convergence rate while maintaining high swarm connectivity across varying swarm sizes. While the analytical solution guarantees cohesion, it is often too slow in practice. In time-critical scenarios, such as emergency response operations where lives are at risk, slower analytical methods are impractical and justify the loss of some agents within the swarm. This paper presents and analyzes this trade-off in detail. 

**Abstract (ZH)**: 一种简单的多智能体系统可以在灾难响应应用中有效利用，如消防灭火。这类蜂群需要在具有有限局部感知能力和无可靠智能体间通信或集中控制的复杂环境中运作。这些简单的类蚁机器人被定义为匿名智能体，具备有限的感知能力、缺乏共享坐标系，并且彼此之间不进行显式通信。简单蜂群的核心挑战在于尽管感知范围有限，仍需维持一致性并避免分裂。机器学习的最新进展提供了解决部分经典分散式控制挑战的有效方案。我们提出了一种基于深度学习的分散式控制模型——VariAntNet，旨在促进智能体蜂群和协作任务执行。VariAntNet 包括从无序的、大小可变的局部观测中提取几何特征。该模型结合了一种经过新型可微分、多目标、数学上合理的损失函数训练的神经网络架构，该损失函数利用可视图拉普拉斯矩阵的性质促进蜂群一致性。VariAntNet 在基本的多智能体聚集任务中得到了演示，该任务要求仅具备方位感知和有限范围感知的智能体聚集到某位置。VariAntNet 在聚集速率和蜂群连通性方面显著优于现有的解析解决方案，尤其是在不同规模的蜂群中保持高连通性方面。虽然解析解决方案能够保证一致性，但在实践中却往往过慢。在时间紧迫的情景下，如紧急救援操作中，过慢的解析方法不切实际，可能需要牺牲蜂群中的部分智能体。本文详细分析了这种权衡。 

---
# Autoencoder-based non-intrusive model order reduction in continuum mechanics 

**Title (ZH)**: 基于自动编码器的非侵入式模型秩序减低在连续力学中 

**Authors**: Jannick Kehls, Ellen Kuhl, Tim Brepols, Kevin Linka, Hagen Holthusen  

**Link**: [PDF](https://arxiv.org/pdf/2509.02237)  

**Abstract**: We propose a non-intrusive, Autoencoder-based framework for reduced-order modeling in continuum mechanics. Our method integrates three stages: (i) an unsupervised Autoencoder compresses high-dimensional finite element solutions into a compact latent space, (ii) a supervised regression network maps problem parameters to latent codes, and (iii) an end-to-end surrogate reconstructs full-field solutions directly from input parameters.
To overcome limitations of existing approaches, we propose two key extensions: a force-augmented variant that jointly predicts displacement fields and reaction forces at Neumann boundaries, and a multi-field architecture that enables coupled field predictions, such as in thermo-mechanical systems. The framework is validated on nonlinear benchmark problems involving heterogeneous composites, anisotropic elasticity with geometric variation, and thermo-mechanical coupling. Across all cases, it achieves accurate reconstructions of high-fidelity solutions while remaining fully non-intrusive.
These results highlight the potential of combining deep learning with dimensionality reduction to build efficient and extensible surrogate models. Our publicly available implementation provides a foundation for integrating data-driven model order reduction into uncertainty quantification, optimization, and digital twin applications. 

**Abstract (ZH)**: 基于自动编码器的非侵入式降阶建模框架在连续介质力学中的应用：结合深度学习与维数减少构建高效可扩展的代理模型 

---
# Towards Multi-Aspect Diversification of News Recommendations Using Neuro-Symbolic AI for Individual and Societal Benefit 

**Title (ZH)**: 利用神经符号AI实现多方面新闻推荐的多样化：个体与社会利益促进 

**Authors**: Markus Reiter-Haas, Elisabeth Lex  

**Link**: [PDF](https://arxiv.org/pdf/2509.02220)  

**Abstract**: News recommendations are complex, with diversity playing a vital role. So far, existing literature predominantly focuses on specific aspects of news diversity, such as viewpoints. In this paper, we introduce multi-aspect diversification in four distinct recommendation modes and outline the nuanced challenges in diversifying lists, sequences, summaries, and interactions. Our proposed research direction combines symbolic and subsymbolic artificial intelligence, leveraging both knowledge graphs and rule learning. We plan to evaluate our models using user studies to not only capture behavior but also their perceived experience. Our vision to balance news consumption points to other positive effects for users (e.g., increased serendipity) and society (e.g., decreased polarization). 

**Abstract (ZH)**: 新闻推荐复杂多样，多方面 diversification 起着关键作用。现有文献主要聚焦于新闻多样性的特定方面，如观点。本文介绍了在四种不同的推荐模式中引入多方面多样性的方法，并概述了在列表、序列、摘要和交互中精细化多样化所面临的挑战。我们提出的研究方向结合了符号和次符号人工智能，利用知识图谱和规则学习。计划通过用户研究评估我们的模型，不仅捕捉用户行为，还捕捉其感知体验。我们平衡新闻消费的愿景指向用户和其他积极影响（如增加意外收获）和社会影响（如减少 polarization）。 

---
# ST-Hyper: Learning High-Order Dependencies Across Multiple Spatial-Temporal Scales for Multivariate Time Series Forecasting 

**Title (ZH)**: ST-Hyper: 学习多空间-时间尺度上的高阶依赖关系以进行多变量时间序列预测 

**Authors**: Binqing Wu, Jianlong Huang, Zongjiang Shang, Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.02217)  

**Abstract**: In multivariate time series (MTS) forecasting, many deep learning based methods have been proposed for modeling dependencies at multiple spatial (inter-variate) or temporal (intra-variate) scales. However, existing methods may fail to model dependencies across multiple spatial-temporal scales (ST-scales, i.e., scales that jointly consider spatial and temporal scopes). In this work, we propose ST-Hyper to model the high-order dependencies across multiple ST-scales through adaptive hypergraph modeling. Specifically, we introduce a Spatial-Temporal Pyramid Modeling (STPM) module to extract features at multiple ST-scales. Furthermore, we introduce an Adaptive Hypergraph Modeling (AHM) module that learns a sparse hypergraph to capture robust high-order dependencies among features. In addition, we interact with these features through tri-phase hypergraph propagation, which can comprehensively capture multi-scale spatial-temporal dynamics. Experimental results on six real-world MTS datasets demonstrate that ST-Hyper achieves the state-of-the-art performance, outperforming the best baselines with an average MAE reduction of 3.8\% and 6.8\% for long-term and short-term forecasting, respectively. 

**Abstract (ZH)**: 基于自适应超图建模的多尺度时空依赖性多变量时间序列预测 

---
# Beyond Ensembles: Simulating All-Atom Protein Dynamics in a Learned Latent Space 

**Title (ZH)**: 超越集成：在学习到的潜空间中模拟蛋白质的原子细节动力学 

**Authors**: Aditya Sengar, Ali Hariri, Pierre Vandergheynst, Patrick Barth  

**Link**: [PDF](https://arxiv.org/pdf/2509.02196)  

**Abstract**: Simulating the long-timescale dynamics of biomolecules is a central challenge in computational science. While enhanced sampling methods can accelerate these simulations, they rely on pre-defined collective variables that are often difficult to identify. A recent generative model, LD-FPG, demonstrated that this problem could be bypassed by learning to sample the static equilibrium ensemble as all-atom deformations from a reference structure, establishing a powerful method for all-atom ensemble generation. However, while this approach successfully captures a system's probable conformations, it does not model the temporal evolution between them. Here we extend LD-FPG with a temporal propagator that operates within the learned latent space and compare three classes: (i) score-guided Langevin dynamics, (ii) Koopman-based linear operators, and (iii) autoregressive neural networks. Within a unified encoder-propagator-decoder framework, we evaluate long-horizon stability, backbone and side-chain ensemble fidelity, and functional free-energy landscapes. Autoregressive neural networks deliver the most robust long rollouts; score-guided Langevin best recovers side-chain thermodynamics when the score is well learned; and Koopman provides an interpretable, lightweight baseline that tends to damp fluctuations. These results clarify the trade-offs among propagators and offer practical guidance for latent-space simulators of all-atom protein dynamics. 

**Abstract (ZH)**: 模拟生物分子的长时间尺度动力学是计算科学中的一个核心挑战。虽然增强采样方法可以加速这些模拟，但它们依赖于预定义的集体变量，这些变量往往难以识别。最近提出的生成模型LD-FPG证明了可以通过从参考结构学习采样静态平衡ensemble的所有原子变形来绕过这一问题，从而建立了强大的所有原子ensemble生成方法。然而，尽管这种方法成功捕获了系统的可能构象，但未能模型化它们之间的时序演化。在这里，我们扩展了LD-FPG，引入了一个在学习潜空间内操作的时序传播器，并比较了三类方法：(i) 得分引导的拉angevin动力学，(ii) 考尔曼基于的线性算子，和(iii) 自回归神经网络。在统一的编码器-传播器-解码器框架中，我们评估了长视角稳定性、主链和侧链ensemble的真实性以及功能自由能景观。自回归神经网络提供最稳健的远期 rollout；得分引导的拉angevin动力学在得分学习良好时最好地恢复侧链热力学性质；考尔曼提供了可解释、轻量级的基线方法，倾向于抑制波动。这些结果明确了传播器之间的权衡，并为所有原子蛋白质动力学的潜空间模拟器提供了实际指导。 

---
# Conditional-$t^3$VAE: Equitable Latent Space Allocation for Fair Generation 

**Title (ZH)**: 条件-$t^3$VAE：公平生成中的公平潜在空间分配 

**Authors**: Aymene Mohammed Bouayed, Samuel Deslauriers-Gauthier, Adrian Iaccovelli, David Naccache  

**Link**: [PDF](https://arxiv.org/pdf/2509.02154)  

**Abstract**: Variational Autoencoders (VAEs) with global priors mirror the training set's class frequency in latent space, underrepresenting tail classes and reducing generative fairness on imbalanced datasets. While $t^3$VAE improves robustness via heavy-tailed Student's t-distribution priors, it still allocates latent volume proportionally to the class this http URL this work, we address this issue by explicitly enforcing equitable latent space allocation across classes. To this end, we propose Conditional-$t^3$VAE, which defines a per-class \mbox{Student's t} joint prior over latent and output variables, preventing dominance by majority classes. Our model is optimized using a closed-form objective derived from the $\gamma$-power divergence. Moreover, for class-balanced generation, we derive an equal-weight latent mixture of Student's t-distributions. On SVHN-LT, CIFAR100-LT, and CelebA, Conditional-$t^3$VAE consistently achieves lower FID scores than both $t^3$VAE and Gaussian-based VAE baselines, particularly under severe class imbalance. In per-class F1 evaluations, Conditional-$t^3$VAE also outperforms the conditional Gaussian VAE across all highly imbalanced settings. While Gaussian-based models remain competitive under mild imbalance ratio ($\rho \lesssim 3$), our approach substantially improves generative fairness and diversity in more extreme regimes. 

**Abstract (ZH)**: 条件化$t^3$自编码器：通过明确强制公平的潜在空间分配提高生成公平性 

---
# A Theoretical Framework of the Processes of Change in Psychotherapy Delivered by Artificial Agents 

**Title (ZH)**: 人工代理提供的心理治疗过程变化的理论框架 

**Authors**: Arthur Bran Herbener, Malene Flensborg Damholdt  

**Link**: [PDF](https://arxiv.org/pdf/2509.02144)  

**Abstract**: The question of whether artificial agents (e.g., chatbots and social robots) can replace human therapists has received notable attention following the recent launch of large language models. However, little is known about the processes of change in psychotherapy delivered by artificial agents. To facilitate hypothesis development and stimulate scientific debate, the present article offers the first theoretical framework of the processes of change in psychotherapy delivered by artificial agents. The theoretical framework rests upon a conceptual analysis of what active ingredients may be inherently linked to the presence of human therapists. We propose that human therapists' ontological status as human beings and sociocultural status as socially sanctioned healthcare professionals play crucial roles in promoting treatment outcomes. In the absence of the ontological and sociocultural status of human therapists, we propose what we coin the genuineness gap and credibility gap can emerge and undermine key processes of change in psychotherapy. Based on these propositions, we propose avenues for scientific investigations and practical applications aimed at leveraging the strengths of artificial agents and human therapists respectively. We also highlight the intricate agentic nature of artificial agents and discuss how this complicates endeavors to establish universally applicable propositions regarding the processes of change in these interventions. 

**Abstract (ZH)**: 人工代理（如聊天机器人和社会机器人）能否替代人类治疗师的心理治疗变革过程：初步理论框架 

---
# HiGraph: A Large-Scale Hierarchical Graph Dataset for Malware Analysis 

**Title (ZH)**: HiGraph: 一种大规模分层图数据集用于恶意软件分析 

**Authors**: Han Chen, Hanchen Wang, Hongmei Chen, Ying Zhang, Lu Qin, Wenjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02113)  

**Abstract**: The advancement of graph-based malware analysis is critically limited by the absence of large-scale datasets that capture the inherent hierarchical structure of software. Existing methods often oversimplify programs into single level graphs, failing to model the crucial semantic relationship between high-level functional interactions and low-level instruction logic. To bridge this gap, we introduce \dataset, the largest public hierarchical graph dataset for malware analysis, comprising over \textbf{200M} Control Flow Graphs (CFGs) nested within \textbf{595K} Function Call Graphs (FCGs). This two-level representation preserves structural semantics essential for building robust detectors resilient to code obfuscation and malware evolution. We demonstrate HiGraph's utility through a large-scale analysis that reveals distinct structural properties of benign and malicious software, establishing it as a foundational benchmark for the community. The dataset and tools are publicly available at this https URL. 

**Abstract (ZH)**: 基于图的恶意软件分析的进步受限于缺乏能够捕获软件固有层次结构的大规模数据集。现有的方法往往将程序简化为单一层次的图形，未能建模高级功能交互与低级指令逻辑之间的关键语义关系。为解决这一问题，我们引入了\dataset，这是最大的公开层次图数据集，用于恶意软件分析，包含超过200M个控制流图(CFGs)，嵌套在595K个函数调用图(FCGs)中。这种两级表示保留了构建对代码混淆和恶意软件演化具有抵抗力的健壮检测器所需的关键结构语义。我们通过大规模分析展示了HiGraph的实用性，揭示了良性软件和恶意软件的distinct结构特性，将其确立为社区基础基准。数据集和工具可在以下网址公开获取：this https URL。 

---
# SALAD -- Semantics-Aware Logical Anomaly Detection 

**Title (ZH)**: SALAD —— 语义驱动的逻辑异常检测 

**Authors**: Matic Fučka, Vitjan Zavrtanik, Danijel Skočaj  

**Link**: [PDF](https://arxiv.org/pdf/2509.02101)  

**Abstract**: Recent surface anomaly detection methods excel at identifying structural anomalies, such as dents and scratches, but struggle with logical anomalies, such as irregular or missing object components. The best-performing logical anomaly detection approaches rely on aggregated pretrained features or handcrafted descriptors (most often derived from composition maps), which discard spatial and semantic information, leading to suboptimal performance. We propose SALAD, a semantics-aware discriminative logical anomaly detection method that incorporates a newly proposed composition branch to explicitly model the distribution of object composition maps, consequently learning important semantic relationships. Additionally, we introduce a novel procedure for extracting composition maps that requires no hand-made labels or category-specific information, in contrast to previous methods. By effectively modelling the composition map distribution, SALAD significantly improves upon state-of-the-art methods on the standard benchmark for logical anomaly detection, MVTec LOCO, achieving an impressive image-level AUROC of 96.1%. Code: this https URL 

**Abstract (ZH)**: 近期的表面异常检测方法在识别结构异常（如凹坑和刮痕）方面表现出色，但在处理逻辑异常（如不规则或缺失的物体组件）方面存在困难。最佳的逻辑异常检测方法依赖于聚合的预训练特征或手工制作的描述符（大多数情况下源自组成图），这些方法会丢弃空间和语义信息，导致性能不佳。我们提出了SALAD，一种语义感知的区分性逻辑异常检测方法，该方法包含一个新提出的组成分支，以明确建模对象组成图的分布，从而学习重要的语义关系。此外，我们引入了一种新的方法来提取组成图，无需人工标签或类别特定信息，与之前的方法不同。通过有效建模组成图的分布，SALAD在标准的逻辑异常检测基准MVTec LOCO上显著优于现有方法，实现了令人印象深刻的图像级AUC-ROC值96.1%。代码: https://this-url.com 

---
# Forecasting Future DDoS Attacks Using Long Short Term Memory (LSTM) Model 

**Title (ZH)**: 使用长期短期记忆（LSTM）模型预测未来DDoS攻击 

**Authors**: Kong Mun Yeen, Rafidah Md Noor, Wahidah Md Shah, Aslinda Hassan, Muhammad Umair Munir  

**Link**: [PDF](https://arxiv.org/pdf/2509.02076)  

**Abstract**: This paper forecasts future Distributed Denial of Service (DDoS) attacks using deep learning models. Although several studies address forecasting DDoS attacks, they remain relatively limited compared to detection-focused research. By studying the current trends and forecasting based on newer and updated datasets, mitigation plans against the attacks can be planned and formulated. The methodology used in this research work conforms to the Cross Industry Standard Process for Data Mining (CRISP-DM) model. 

**Abstract (ZH)**: 本论文使用深度学习模型预测未来的分布式拒绝服务（DDoS）攻击 

---
# Privacy-Utility Trade-off in Data Publication: A Bilevel Optimization Framework with Curvature-Guided Perturbation 

**Title (ZH)**: 数据发布中的隐私- utility权衡：带有曲率引导扰动的 bilevel 优化框架 

**Authors**: Yi Yin, Guangquan Zhang, Hua Zuo, Jie Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.02048)  

**Abstract**: Machine learning models require datasets for effective training, but directly sharing raw data poses significant privacy risk such as membership inference attacks (MIA). To mitigate the risk, privacy-preserving techniques such as data perturbation, generalization, and synthetic data generation are commonly utilized. However, these methods often degrade data accuracy, specificity, and diversity, limiting the performance of downstream tasks and thus reducing data utility. Therefore, striking an optimal balance between privacy preservation and data utility remains a critical challenge.
To address this issue, we introduce a novel bilevel optimization framework for the publication of private datasets, where the upper-level task focuses on data utility and the lower-level task focuses on data privacy. In the upper-level task, a discriminator guides the generation process to ensure that perturbed latent variables are mapped to high-quality samples, maintaining fidelity for downstream tasks. In the lower-level task, our framework employs local extrinsic curvature on the data manifold as a quantitative measure of individual vulnerability to MIA, providing a geometric foundation for targeted privacy protection. By perturbing samples toward low-curvature regions, our method effectively suppresses distinctive feature combinations that are vulnerable to MIA. Through alternating optimization of both objectives, we achieve a synergistic balance between privacy and utility. Extensive experimental evaluations demonstrate that our method not only enhances resistance to MIA in downstream tasks but also surpasses existing methods in terms of sample quality and diversity. 

**Abstract (ZH)**: 一种新颖的双层优化框架：在保护隐私的同时提高数据效用 

---
# Fantastic Pretraining Optimizers and Where to Find Them 

**Title (ZH)**: 神奇的预训练优化器及其获取途径 

**Authors**: Kaiyue Wen, David Hall, Tengyu Ma, Percy Liang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02046)  

**Abstract**: AdamW has long been the dominant optimizer in language model pretraining, despite numerous claims that alternative optimizers offer 1.4 to 2x speedup. We posit that two methodological shortcomings have obscured fair comparisons and hindered practical adoption: (i) unequal hyperparameter tuning and (ii) limited or misleading evaluation setups. To address these two issues, we conduct a systematic study of ten deep learning optimizers across four model scales (0.1B-1.2B parameters) and data-to-model ratios (1-8x the Chinchilla optimum). We find that fair and informative comparisons require rigorous hyperparameter tuning and evaluations across a range of model scales and data-to-model ratios, performed at the end of training. First, optimal hyperparameters for one optimizer may be suboptimal for another, making blind hyperparameter transfer unfair. Second, the actual speedup of many proposed optimizers over well-tuned baselines is lower than claimed and decreases with model size to only 1.1x for 1.2B parameter models. Thirdly, comparing intermediate checkpoints before reaching the target training budgets can be misleading, as rankings between two optimizers can flip during training due to learning rate decay. Through our thorough investigation, we find that all the fastest optimizers such as Muon and Soap, use matrices as preconditioners -- multiplying gradients with matrices rather than entry-wise scalars. However, the speedup of matrix-based optimizers is inversely proportional to model scale, decreasing from 1.4x over AdamW for 0.1B parameter models to merely 1.1x for 1.2B parameter models. 

**Abstract (ZH)**: AdamW在语言模型预训练中长期占主导地位，尽管有诸多声称替代优化器可提供1.4到2倍的速度提升。我们提出，两种方法学缺陷阻碍了公平比较和实际应用：（i）超参数调整不平等及（ii）评估设置有限或具有误导性。为解决这两个问题，我们在四种模型规模（0.1B-1.2B参数）和数据与模型比例（1-8倍Chinchilla最优值）下系统研究了十个深度学习优化器。我们发现，公平且具信息性的比较需要在训练结束时，在不同模型规模和数据与模型比例下的严格超参数调整和评估。首先，一种优化器的最佳超参数可能不适合另一种优化器，使其盲目转移超参数不公平。其次，许多提出优化器相对于精细调校的基础线的真正速度提升低于声称值，并随模型规模减小，对于1.2B参数模型仅略高于1.1倍。第三，比较达到目标训练预算前的中间检查点可能具有误导性，因为优化器之间的排名在训练过程中由于学习率衰减可能会变化。通过全面的研究，我们发现，如Muon和Soap等最快速的优化器均使用矩阵作为预处理器—将梯度与矩阵而非标量逐元素缩放相乘。然而，矩阵基优化器的速度提升与模型规模成反比，从0.1B参数模型的1.4倍下降到1.2B参数模型的仅1.1倍。 

---
# ACA-Net: Future Graph Learning for Logistical Demand-Supply Forecasting 

**Title (ZH)**: ACA-Net：基于未来图学习的物流供需预测 

**Authors**: Jiacheng Shi, Haibin Wei, Jiang Wang, Xiaowei Xu, Longzhi Du, Taixu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01997)  

**Abstract**: Logistical demand-supply forecasting that evaluates the alignment between projected supply and anticipated demand, is essential for the efficiency and quality of on-demand food delivery platforms and serves as a key indicator for scheduling decisions. Future order distribution information, which reflects the distribution of orders in on-demand food delivery, is crucial for the performance of logistical demand-supply forecasting. Current studies utilize spatial-temporal analysis methods to model future order distribution information from serious time slices. However, learning future order distribution in online delivery platform is a time-series-insensitive problem with strong randomness. These approaches often struggle to effectively capture this information while remaining efficient. This paper proposes an innovative spatiotemporal learning model that utilizes only two graphs (ongoing and global) to learn future order distribution information, achieving superior performance compared to traditional spatial-temporal long-series methods. The main contributions are as follows: (1) The introduction of ongoing and global graphs in logistical demand-supply pressure forecasting compared to traditional long time series significantly enhances forecasting performance. (2) An innovative graph learning network framework using adaptive future graph learning and innovative cross attention mechanism (ACA-Net) is proposed to extract future order distribution information, effectively learning a robust future graph that substantially improves logistical demand-supply pressure forecasting outcomes. (3) The effectiveness of the proposed method is validated in real-world production environments. 

**Abstract (ZH)**: 面向需求-供应物流预测中评估预估供应与预期需求之间的一致性，对于按需食品配送平台的效率和质量至关重要，并作为排班决策的关键指标。对未来订单分布信息的学习对于按需食品配送物流需求-供应预测的性能至关重要，未来订单分布信息反映了按需食品配送中的订单分布情况。当前研究利用时空分析方法从严重的时间切片中建模未来订单分布信息。然而，在线配送平台中未来订单分布的学习是一个时间和时间序列都不敏感的问题，具有很强的随机性。这些方法往往难以有效地捕捉这些信息同时保持高效。本文提出了一种创新的空间时间学习模型，仅使用两个图形（当前图形和全局图形）来学习未来订单分布信息，其性能优于传统的长时间序列空间时间方法。主要贡献如下：（1）与传统的长期时间序列相比，在物流需求-供应压力预测中引入当前和全局图形显著提高了预测性能。（2）提出了一种创新的图形学习网络框架，使用自适应未来图形学习和创新交叉注意力机制（ACA-Net），以提取未来订单分布信息，有效地学习出一个鲁棒的未来图形，显著提高了物流需求-供应压力预测的结果。（3）在实际生产环境中验证了所提出方法的有效性。 

---
# A Continuous Encoding-Based Representation for Efficient Multi-Fidelity Multi-Objective Neural Architecture Search 

**Title (ZH)**: 基于连续编码的表示在高效多保真多目标神经架构搜索中的应用 

**Authors**: Zhao Wei, Chin Chun Ooi, Yew-Soon Ong  

**Link**: [PDF](https://arxiv.org/pdf/2509.01943)  

**Abstract**: Neural architecture search (NAS) is an attractive approach to automate the design of optimized architectures but is constrained by high computational budget, especially when optimizing for multiple, important conflicting objectives. To address this, an adaptive Co-Kriging-assisted multi-fidelity multi-objective NAS algorithm is proposed to further reduce the computational cost of NAS by incorporating a clustering-based local multi-fidelity infill sampling strategy, enabling efficient exploration of the search space for faster convergence. This algorithm is further accelerated by the use of a novel continuous encoding method to represent the connections of nodes in each cell within a generalized cell-based U-Net backbone, thereby decreasing the search dimension (number of variables). Results indicate that the proposed NAS algorithm outperforms previously published state-of-the-art methods under limited computational budget on three numerical benchmarks, a 2D Darcy flow regression problem and a CHASE_DB1 biomedical image segmentation problem. The proposed method is subsequently used to create a wind velocity regression model with application in urban modelling, with the found model able to achieve good prediction with less computational complexity. Further analysis revealed that the NAS algorithm independently identified principles undergirding superior U-Net architectures in other literature, such as the importance of allowing each cell to incorporate information from prior cells. 

**Abstract (ZH)**: 神经架构搜索（NAS）是一种自动设计优化架构的有吸引力的方法，但受限于高昂的计算预算，特别是在优化多个重要且相互矛盾的目标时更为明显。为了解决这一问题，提出了一个自适应Co-Kriging辅助多保真多目标NAS算法，通过结合基于聚类的局部多保真填充采样策略，进一步减少NAS的计算成本，从而更高效地探索搜索空间以实现更快的收敛。该算法通过使用一种新的连续编码方法来表示每个单元内部节点连接，在通用单元基U-Net骨干网络中，降低了搜索维度（变量数量）。实验结果表明，在有限的计算预算下，所提出的方法在三个数值基准、2D达西流动回归问题和CHASE_DB1生物医学图像分割问题上优于之前发表的最先进的方法。该方法随后被用于创建一个风速回归模型，应用于城市建模，发现的模型具有较低的计算复杂度但仍能实现良好的预测。进一步的分析显示，NAS算法独立地识别了其他文献中优越的U-Net架构所遵循的原则，例如允许每个单元整合前序单元的信息的重要性。 

---
# VISP: Volatility Informed Stochastic Projection for Adaptive Regularization 

**Title (ZH)**: VISP：基于波动率的信息随机投影适应性正则化 

**Authors**: Tanvir Islam  

**Link**: [PDF](https://arxiv.org/pdf/2509.01903)  

**Abstract**: We propose VISP: Volatility Informed Stochastic Projection, an adaptive regularization method that leverages gradient volatility to guide stochastic noise injection in deep neural networks. Unlike conventional techniques that apply uniform noise or fixed dropout rates, VISP dynamically computes volatility from gradient statistics and uses it to scale a stochastic projection matrix. This mechanism selectively regularizes inputs and hidden nodes that exhibit higher gradient volatility while preserving stable representations, thereby mitigating overfitting. Extensive experiments on MNIST, CIFAR-10, and SVHN demonstrate that VISP consistently improves generalization performance over baseline models and fixed-noise alternatives. In addition, detailed analyses of the evolution of volatility, the spectral properties of the projection matrix, and activation distributions reveal that VISP not only stabilizes the internal dynamics of the network but also fosters a more robust feature representation. 

**Abstract (ZH)**: Volatility Informed Stochastic Projection: 一种基于梯度波动的自适应正则化方法 

---
# Preserving Bilinear Weight Spectra with a Signed and Shrunk Quadratic Activation Function 

**Title (ZH)**: 保持双线性权重谱的一种带有符号收缩二次激活函数的方法 

**Authors**: Jason Abohwo, Thomas Mosen  

**Link**: [PDF](https://arxiv.org/pdf/2509.01874)  

**Abstract**: Understanding the inner workings of machine learning models is critical for ensuring their reliability and robustness. Whilst many techniques in mechanistic interpretability focus on activation driven analyses, being able to derive meaningful features directly from the weights of a neural network would provide greater guarantees and more computational efficiency. Existing techniques for analyzing model features through weights suffer from drawbacks such as reduced performance and data inefficiency. In this paper, we introduce Signed Quadratic Shrink (SQS), an activation function designed to allow Gated Linear Units (GLUs) to learn interpretable features without these drawbacks. Our experimental results show that SQS achieves performance competitive with state-of-the-art activation functions whilst enabling weight-based interpretability 

**Abstract (ZH)**: 理解机器学习模型的内部工作机制对于确保其可靠性和 robustness 至关重要。虽然许多机制可解释性技术集中在激活驱动的分析上，直接从神经网络权重中推导出有意义的特征将提供更大的保证并提高计算效率。现有通过权重分析模型特征的技术存在性能降低和数据 inefficient 等缺点。本文介绍了 Signed Quadratic Shrink (SQS)，这是一种激活函数，旨在使 Gated Linear Units (GLUs) 能够学习可解释的特征而不受这些缺点的影响。我们的实验结果表明，SQS 在性能上与最先进的激活函数相当，同时支持基于权重的可解释性。 

---
# Community-Centered Spatial Intelligence for Climate Adaptation at Nova Scotia's Eastern Shore 

**Title (ZH)**: 面向社区的空间智能在诺瓦 Scotia 东岸气候变化适应中的应用 

**Authors**: Gabriel Spadon, Oladapo Oyebode, Camilo M. Botero, Tushar Sharma, Floris Goerlandt, Ronald Pelot  

**Link**: [PDF](https://arxiv.org/pdf/2509.01845)  

**Abstract**: This paper presents an overview of a human-centered initiative aimed at strengthening climate resilience along Nova Scotia's Eastern Shore. This region, a collection of rural villages with deep ties to the sea, faces existential threats from climate change that endanger its way of life. Our project moves beyond a purely technical response, weaving together expertise from Computer Science, Industrial Engineering, and Coastal Geography to co-create tools with the community. By integrating generational knowledge of residents, particularly elders, through the Eastern Shore Citizen Science Coastal Monitoring Network, this project aims to collaborate in building a living digital archive. This effort is hosted under Dalhousie University's Transforming Climate Action (TCA) initiative, specifically through its Transformative Adaptations to Social-Ecological Climate Change Trajectories (TranSECT) and TCA Artificial Intelligence (TCA-AI) projects. This work is driven by a collaboration model in which student teams work directly with residents. We present a detailed project timeline and a replicable model for how technology can support traditional communities, enabling them to navigate climate transformation more effectively. 

**Abstract (ZH)**: 本文介绍了旨在加强诺瓦 Scotia 东岸地区气候韧性的以人类为中心的倡议的概述。该地区由与海洋有着深厚联系的农村村庄组成，正面临着气候变化带来的生存威胁，危及当地的生活方式。我们的项目超越了单纯的技術回应，将计算机科学、工业工程和海岸地理学的专长结合起来，与社区共同创造工具。通过东岸公民科学海岸监测网络，本项目计划整合居民，特别是长者们的代际知识，共同构建一个活的数字档案。该努力依托达尔豪斯大学转型气候行动 (TCA) 计划，具体通过其变革性的社会生态气候变迁适应 (TranSECT) 以及 TCA 人工智能 (TCA-AI) 项目进行。这项工作基于学生团队直接与居民合作的协作模式。本文详细介绍了项目时间线，并提供了一个技术如何支持传统社区、帮助它们更有效地应对气候转变的可复制模型。 

---
# GradES: Significantly Faster Training in Transformers with Gradient-Based Early Stopping 

**Title (ZH)**: GradES: 基于梯度的早停以显著加快Transformer模型的训练速度 

**Authors**: Qifu Wen, Xi Zeng, Zihan Zhou, Shuaijun Liu, Mehdi Hosseinzadeh, Reza Rawassizadeh  

**Link**: [PDF](https://arxiv.org/pdf/2509.01842)  

**Abstract**: Early stopping monitors global validation loss and halts all parameter updates simultaneously, which is computationally costly for large transformers due to the extended time required for validation inference. We propose GradES, a novel gradient-based early stopping approach that operates within transformer components (attention projections and Feed-Forward layer matrices). We found that different components converge at varying rates during fine-tuning. GradES tracks the magnitude of gradients in backpropagation for these matrices during training. When a projection matrix's gradients fall below a convergence threshold $\tau$, we exclude that projection matrix from further updates individually, eliminating costly validation passes while allowing slow converging matrices to continue learning. By strategically freezing parameters when their gradients converge, GradES speeds up training time by 1.57--7.22$\times$ while simultaneously enhancing generalization through early prevention of overfitting, resulting in 1.2% higher average accuracy. 

**Abstract (ZH)**: 基于梯度的早停方法GradES：在Transformer组件内监控梯度收敛以加速训练并提升泛化能力 

---
# HodgeFormer: Transformers for Learnable Operators on Triangular Meshes through Data-Driven Hodge Matrices 

**Title (ZH)**: HodgeFormer：通过数据驱动的霍奇矩阵学习算子的三角网格变换器 

**Authors**: Akis Nousias, Stavros Nousias  

**Link**: [PDF](https://arxiv.org/pdf/2509.01839)  

**Abstract**: Currently, prominent Transformer architectures applied on graphs and meshes for shape analysis tasks employ traditional attention layers that heavily utilize spectral features requiring costly eigenvalue decomposition-based methods. To encode the mesh structure, these methods derive positional embeddings, that heavily rely on eigenvalue decomposition based operations, e.g. on the Laplacian matrix, or on heat-kernel signatures, which are then concatenated to the input features. This paper proposes a novel approach inspired by the explicit construction of the Hodge Laplacian operator in Discrete Exterior Calculus as a product of discrete Hodge operators and exterior derivatives, i.e. $(L := \star_0^{-1} d_0^T \star_1 d_0)$. We adjust the Transformer architecture in a novel deep learning layer that utilizes the multi-head attention mechanism to approximate Hodge matrices $\star_0$, $\star_1$ and $\star_2$ and learn families of discrete operators $L$ that act on mesh vertices, edges and faces. Our approach results in a computationally-efficient architecture that achieves comparable performance in mesh segmentation and classification tasks, through a direct learning framework, while eliminating the need for costly eigenvalue decomposition operations or complex preprocessing operations. 

**Abstract (ZH)**: 目前，用于形状分析任务的基于图和网格的显着Transformer架构普遍采用传统注意力层，这些层大量依赖于需要昂贵特征值分解方法的谱特征。为了编码网格结构，这些方法提取位置嵌入，这些嵌入高度依赖于特征值分解操作，例如在拉普拉斯矩阵上进行的操作，或者基于热核特征，然后将这些嵌入连接到输入特征。本文提出了一种新颖的方法，该方法受离散外微分学中霍奇拉普拉斯算子明确构造的启发，视为离散霍奇算子与外导数的乘积，即 $(L := \star_0^{-1} d_0^T \star_1 d_0)$。我们通过调整Transformer架构，在一个新的深度学习层中利用多头注意力机制来近似霍奇矩阵 $\star_0$，$\star_1$ 和 $\star_2$，并学习作用于网格顶点、边和面的离散算子族 $L$。我们的方法产生了一种计算高效的架构，在网格分割和分类任务中达到了可比的性能，通过一个直接的学习框架，同时消除了昂贵的特征值分解操作或复杂预处理操作的需求。 

---
# Journalists' Perceptions of Artificial Intelligence and Disinformation Risks 

**Title (ZH)**: 记者对人工智能与假信息风险的认知 

**Authors**: Urko Peña-Alonso, Simón Peña-Fernández, Koldobika Meso-Ayerdi  

**Link**: [PDF](https://arxiv.org/pdf/2509.01824)  

**Abstract**: This study examines journalists' perceptions of the impact of artificial intelligence (AI) on disinformation, a growing concern in journalism due to the rapid expansion of generative AI and its influence on news production and media organizations. Using a quantitative approach, a structured survey was administered to 504 journalists in the Basque Country, identified through official media directories and with the support of the Basque Association of Journalists. This survey, conducted online and via telephone between May and June 2024, included questions on sociodemographic and professional variables, as well as attitudes toward AI's impact on journalism. The results indicate that a large majority of journalists (89.88%) believe AI will considerably or significantly increase the risks of disinformation, and this perception is consistent across genders and media types, but more pronounced among those with greater professional experience. Statistical analyses reveal a significant association between years of experience and perceived risk, and between AI use and risk perception. The main risks identified are the difficulty in detecting false content and deepfakes, and the risk of obtaining inaccurate or erroneous data. Co-occurrence analysis shows that these risks are often perceived as interconnected. These findings highlight the complex and multifaceted concerns of journalists regarding AI's role in the information ecosystem. 

**Abstract (ZH)**: 本研究探讨了记者在人工智能（AI）对 misinformation影响方面的看法，这一问题因生成式AI的快速扩展及其对新闻生产和媒体组织的影响而在新闻界日益引起关注。通过定量研究方法，研究者通过官方媒体目录并在巴斯克记者协会支持下，对巴斯克地区504名记者进行了结构化调查。该调查于2024年5月至6月通过在线和电话方式进行，包括社会人口学和职业变量，以及对AI对 journalism 影响的态度。结果显示，大量记者（89.88%）认为AI将显著增加 misinformation的风险，这种看法在不同性别和媒体类型中是一致的，但在专业经验更丰富的记者中更为明显。统计分析显示，专业经验年限与感知风险之间存在显著关联，AI使用情况与风险感知之间也存在关联。识别的主要风险包括难以检测虚假内容和深伪，以及获得不准确或错误数据的风险。共现分析显示，这些风险往往被视为相互关联。研究结果突显了记者在人工智能在信息生态系统中作用方面的复杂和多方面关切。 

---
# Quantum Machine Learning for UAV Swarm Intrusion Detection 

**Title (ZH)**: 量子机器学习在无人机群入侵检测中的应用 

**Authors**: Kuan-Cheng Chen, Samuel Yen-Chi Chen, Tai-Yue Li, Chen-Yu Liu, Kin K. Leung  

**Link**: [PDF](https://arxiv.org/pdf/2509.01812)  

**Abstract**: Intrusion detection in unmanned-aerial-vehicle (UAV) swarms is complicated by high mobility, non-stationary traffic, and severe class imbalance. Leveraging a 120 k-flow simulation corpus that covers five attack types, we benchmark three quantum-machine-learning (QML) approaches - quantum kernels, variational quantum neural networks (QNNs), and hybrid quantum-trained neural networks (QT-NNs) - against strong classical baselines. All models consume an 8-feature flow representation and are evaluated under identical preprocessing, balancing, and noise-model assumptions. We analyse the influence of encoding strategy, circuit depth, qubit count, and shot noise, reporting accuracy, macro-F1, ROC-AUC, Matthews correlation, and quantum-resource footprints. Results reveal clear trade-offs: quantum kernels and QT-NNs excel in low-data, nonlinear regimes, while deeper QNNs suffer from trainability issues, and CNNs dominate when abundant data offset their larger parameter count. The complete codebase and dataset partitions are publicly released to enable reproducible QML research in network security. 

**Abstract (ZH)**: 基于无人机群的入侵检测受到高 mobility、非平稳流量和严重类别不平衡的复杂性。利用涵盖五种攻击类型的120 k流仿真数据集，我们将三种量子机器学习（QML）方法——量子核、变分量子神经网络（QNNs）和混合量子训练神经网络（QT-NNs）——与强劲的经典基线进行对比。所有模型使用8特征流量表示，并在相同的预处理、平衡和噪声模型假设下进行评估。我们分析了编码策略、电路深度、量子位数和射电信号噪声的影响，报告了准确性、宏F1、ROC-AUC、马修斯相关系数以及量子资源足迹。结果表明存在明显权衡：量子核和QT-NNs在低数据、非线性条件下表现出色，而更深的QNNs面临训练问题，CNNs则在数据丰富时占优。完整的代码库和数据集分区公开发布，以促进网络安全性中的可重现量子机器学习研究。 

---
# A Multi-target Bayesian Transformer Framework for Predicting Cardiovascular Disease Biomarkers during Pandemics 

**Title (ZH)**: 基于贝叶斯变换器的多目标框架在疫情期间预测心血管疾病生物标志物 

**Authors**: Trusting Inekwe, Emmanuel Agu, Winnie Mkandawire, Andres Colubri  

**Link**: [PDF](https://arxiv.org/pdf/2509.01794)  

**Abstract**: The COVID-19 pandemic disrupted healthcare systems worldwide, disproportionately impacting individuals with chronic conditions such as cardiovascular disease (CVD). These disruptions -- through delayed care and behavioral changes, affected key CVD biomarkers, including LDL cholesterol (LDL-C), HbA1c, BMI, and systolic blood pressure (SysBP). Accurate modeling of these changes is crucial for predicting disease progression and guiding preventive care. However, prior work has not addressed multi-target prediction of CVD biomarker from Electronic Health Records (EHRs) using machine learning (ML), while jointly capturing biomarker interdependencies, temporal patterns, and predictive uncertainty. In this paper, we propose MBT-CB, a Multi-target Bayesian Transformer (MBT) with pre-trained BERT-based transformer framework to jointly predict LDL-C, HbA1c, BMI and SysBP CVD biomarkers from EHR data. The model leverages Bayesian Variational Inference to estimate uncertainties, embeddings to capture temporal relationships and a DeepMTR model to capture biomarker inter-relationships. We evaluate MBT-CT on retrospective EHR data from 3,390 CVD patient records (304 unique patients) in Central Massachusetts during the Covid-19 pandemic. MBT-CB outperformed a comprehensive set of baselines including other BERT-based ML models, achieving an MAE of 0.00887, RMSE of 0.0135 and MSE of 0.00027, while effectively capturing data and model uncertainty, patient biomarker inter-relationships, and temporal dynamics via its attention and embedding mechanisms. MBT-CB's superior performance highlights its potential to improve CVD biomarker prediction and support clinical decision-making during pandemics. 

**Abstract (ZH)**: COVID-19大流行扰乱了全球的医疗保健系统，对心血管疾病（CVD）等慢性病患者造成了不成比例的影响。这些扰乱——通过延迟治疗和行为改变，影响了包括低密度脂蛋白胆固醇（LDL-C）、糖化血红蛋白（HbA1c）、体重指数（BMI）和收缩压（SysBP）在内的关键CVD生物标志物。准确建模这些变化对于预测疾病进展和指导预防性护理至关重要。然而，先前的工作并未通过机器学习（ML）从电子健康记录（EHRs）中联合预测CVD生物标志物，同时捕捉生物标志物之间的相互依赖、时间模式和预测不确定性。在本文中，我们提出了一种基于预训练BERT的变压器架构的多目标贝叶斯变压器（MBT-CB），以联合预测EHR数据中的LDL-C、HbA1c、BMI和SysBP等CVD生物标志物。该模型利用贝叶斯变分推断估算不确定性，通过嵌入捕捉时间关系，并通过DeepMTR模型捕捉生物标志物之间的相互关系。我们在马萨诸塞州中部COVID-19疫情期间3,390例CVD患者的304名独特患者的历史EHR数据上评估了MBT-CB。MBT-CB在包括其他BERT基ML模型的一组基准模型中表现优越，获得MAE为0.00887，RMSE为0.0135，MSE为0.00027，同时通过其注意力和嵌入机制有效地捕捉了数据和模型不确定性、患者生物标志物之间的相互关系以及时间动态。MBT-CB的出色表现突显了其在提高CVD生物标志物预测和支持 pandemic期间临床决策方面的潜力。 

---
# E-PhishGen: Unlocking Novel Research in Phishing Email Detection 

**Title (ZH)**: E-PhishGen: 解锁新型钓鱼邮件检测研究 

**Authors**: Luca Pajola, Eugenio Caripoti, Simeone Pizzi, Mauro Conti, Stefan Banzer, Giovanni Apruzzese  

**Link**: [PDF](https://arxiv.org/pdf/2509.01791)  

**Abstract**: Every day, our inboxes are flooded with unsolicited emails, ranging between annoying spam to more subtle phishing scams. Unfortunately, despite abundant prior efforts proposing solutions achieving near-perfect accuracy, the reality is that countering malicious emails still remains an unsolved dilemma.
This "open problem" paper carries out a critical assessment of scientific works in the context of phishing email detection. First, we focus on the benchmark datasets that have been used to assess the methods proposed in research. We find that most prior work relied on datasets containing emails that -- we argue -- are not representative of current trends, and mostly encompass the English language. Based on this finding, we then re-implement and re-assess a variety of detection methods reliant on machine learning (ML), including large-language models (LLM), and release all of our codebase -- an (unfortunately) uncommon practice in related research. We show that most such methods achieve near-perfect performance when trained and tested on the same dataset -- a result which intrinsically hinders development (how can future research outperform methods that are already near perfect?). To foster the creation of "more challenging benchmarks" that reflect current phishing trends, we propose E-PhishGEN, an LLM-based (and privacy-savvy) framework to generate novel phishing-email datasets. We use our E-PhishGEN to create E-PhishLLM, a novel phishing-email detection dataset containing 16616 emails in three languages. We use E-PhishLLM to test the detectors we considered, showing a much lower performance than that achieved on existing benchmarks -- indicating a larger room for improvement. We also validate the quality of E-PhishLLM with a user study (n=30). To sum up, we show that phishing email detection is still an open problem -- and provide the means to tackle such a problem by future research. 

**Abstract (ZH)**: 每天，我们的收件箱被源源不断的信息轰炸，从烦人的垃圾邮件到更为隐秘的网络钓鱼欺诈，不一而足。尽管先前提出了许多近于完美的解决方案，但对抗恶意邮件仍然是一项未解之谜。

这篇“开放问题”论文对网络钓鱼邮件检测领域的科学工作进行了关键性的评估。首先，我们关注被用于评估研究中提出的各种方法的标准数据集。我们发现，大多数先前的工作依赖于包含——我们认为——不能代表当前趋势的数据集，这些数据集主要覆盖英文。基于这一发现，我们重新实现了并评估了多种依赖机器学习（ML）的检测方法，包括大型语言模型（LLM），并公开了全部代码——这一做法在相关研究中并不常见。我们展示了当训练和测试集一致时，大多数这些方法能够达到近于完美的性能——这一结果固有地阻碍了未来研究的发展（如何超越已经近乎完美的方法？）。为了促进创建能够反映当前网络钓鱼趋势的“更具挑战性的基准”，我们提出了E-PhishGEN框架，这是一种基于大型语言模型（LLM）且注重隐私的框架，用于生成新型网络钓鱼邮件数据集。我们使用E-PhishGEN创建了包含16616封邮件（三种语言）的新颖网络钓鱼邮件检测数据集E-PhishLLM。我们使用E-PhishLLM测试了我们考虑的检测器，显示出远低于现有基准的性能——表明存在更大的改进空间。我们还通过用户研究（n=30）验证了E-PhishLLM的质量。总之，我们展示了网络钓鱼邮件检测仍然是一个开放问题，并提供了未来研究解决此类问题的方法。 

---
# Non-Identical Diffusion Models in MIMO-OFDM Channel Generation 

**Title (ZH)**: 非相同扩散模型在MIMO-OFDM信道生成中的应用 

**Authors**: Yuzhi Yang, Omar Alhussein, Mérouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2509.01641)  

**Abstract**: We propose a novel diffusion model, termed the non-identical diffusion model, and investigate its application to wireless orthogonal frequency division multiplexing (OFDM) channel generation. Unlike the standard diffusion model that uses a scalar-valued time index to represent the global noise level, we extend this notion to an element-wise time indicator to capture local error variations more accurately. Non-identical diffusion enables us to characterize the reliability of each element (e.g., subcarriers in OFDM) within the noisy input, leading to improved generation results when the initialization is biased. Specifically, we focus on the recovery of wireless multi-input multi-output (MIMO) OFDM channel matrices, where the initial channel estimates exhibit highly uneven reliability across elements due to the pilot scheme. Conventional time embeddings, which assume uniform noise progression, fail to capture such variability across pilot schemes and noise levels. We introduce a matrix that matches the input size to control element-wise noise progression. Following a similar diffusion procedure to existing methods, we show the correctness and effectiveness of the proposed non-identical diffusion scheme both theoretically and numerically. For MIMO-OFDM channel generation, we propose a dimension-wise time embedding strategy. We also develop and evaluate multiple training and generation methods and compare them through numerical experiments. 

**Abstract (ZH)**: 非同一扩展扩散模型及其在无线正交频分复用信道生成中的应用 

---
# Disentangling the schema turn: Restoring the information base to conceptual modelling 

**Title (ZH)**: 解构方案转换：恢复信息基到概念建模 

**Authors**: Chris Partridge, Andrew Mitchell, Sergio de Cesare, Oscar Xiberta Soto  

**Link**: [PDF](https://arxiv.org/pdf/2509.01617)  

**Abstract**: If one looks at contemporary mainstream development practices for conceptual modelling in computer science, these so clearly focus on a conceptual schema completely separated from its information base that the conceptual schema is often just called the conceptual model. These schema-centric practices are crystallized in almost every database textbook. We call this strong, almost universal, bias towards conceptual schemas the schema turn. The focus of this paper is on disentangling this turn within (computer science) conceptual modeling. It aims to shed some light on how it emerged and so show that it is not fundamental. To show that modern technology enables the adoption of an inclusive schema-and-base conceptual modelling approach, which in turn enables more automated, and empirically motivated practices. And to show, more generally, the space of possible conceptual modelling practices is wider than currently assumed. It also uses the example of bCLEARer to show that the implementations in this wider space will probably need to rely on new pipeline-based conceptual modelling techniques. So, it is possible that the schema turn's complete exclusion of the information base could be merely a temporary evolutionary detour. 

**Abstract (ZH)**: 当代计算机科学概念建模的主要开发实践几乎完全集中在与信息基座分离的概念模式上，以至于概念模式常常直接被称为概念模型。这种以模式为中心的做法几乎在每一本数据库教科书中都有体现。我们称这一对概念模式的强烈偏见为模式转向。本文的重点在于剖析概念建模中的这一转向，探讨其产生原因，并说明它并非基础性的特点。本文旨在展示现代技术使包容性概念模式和基座的概念模式方法得以采用，进而实现更为自动化和基于实证的概念模式实践。同时，本文也表明概念模式方法的空间比目前所假设的要更宽广。通过bCLEARer的例子，本文还指出，这一更宽广空间中的实现可能需要依赖新的管道式概念模式技术。因此，模式转向完全排除信息基座可能仅仅是一个短暂的进化偏离。 

---
# Entropy-Driven Curriculum for Multi-Task Training in Human Mobility Prediction 

**Title (ZH)**: 基于熵驱动的学习序列在人类移动性预测的多任务训练 Curriculum-Based Entropy-Driven Multi-Task Training for Human Mobility Prediction 

**Authors**: Tianye Fang, Xuanshu Luo, Martin Werner  

**Link**: [PDF](https://arxiv.org/pdf/2509.01613)  

**Abstract**: The increasing availability of big mobility data from ubiquitous portable devices enables human mobility prediction through deep learning approaches. However, the diverse complexity of human mobility data impedes model training, leading to inefficient gradient updates and potential underfitting. Meanwhile, exclusively predicting next locations neglects implicit determinants, including distances and directions, thereby yielding suboptimal prediction results. This paper presents a unified training framework that integrates entropy-driven curriculum and multi-task learning to address these challenges. The proposed entropy-driven curriculum learning strategy quantifies trajectory predictability based on Lempel-Ziv compression and organizes training from simple to complex for faster convergence and enhanced performance. The multi-task training simultaneously optimizes the primary location prediction alongside auxiliary estimation of movement distance and direction for learning realistic mobility patterns, and improve prediction accuracy through complementary supervision signals. Extensive experiments conducted in accordance with the HuMob Challenge demonstrate that our approach achieves state-of-the-art performance on GEO-BLEU (0.354) and DTW (26.15) metrics with up to 2.92-fold convergence speed compared to training without curriculum learning. 

**Abstract (ZH)**: 从广泛便携设备获得的大规模移动数据使通过深度学习方法预测人类移动成为可能。然而，人类移动数据的复杂多样性阻碍了模型训练，导致梯度更新效率低下和潜在的欠拟合。同时，仅预测下一个位置忽略了包括距离和方向在内的隐式决定因素，从而导致次优化的预测结果。本文提出了一种统一的训练框架，该框架结合了基于熵的课程学习和多任务学习以解决这些挑战。提出的基于熵的课程学习策略基于Lempel-Ziv压缩量化轨迹的可预测性，并从简单到复杂组织训练，以加快收敛速度并提高性能。多任务训练同时优化主要位置预测以及辅助运动距离和方向的估计，以学习现实的移动模式并通过互补的监督信号提高预测准确性。遵循HuMob挑战进行的大量实验表明，与无课程学习的训练相比，我们的方法在GEO-BLEU（0.354）和DTW（26.15）指标上实现了最先进的性能，并且收敛速度最多可提高2.92倍。 

---
# An Efficient Intrusion Detection System for Safeguarding Radiation Detection Systems 

**Title (ZH)**: 一种用于保护辐射探测系统安全的有效入侵检测系统 

**Authors**: Nathanael Coolidge, Jaime González Sanz, Li Yang, Khalil El Khatib, Glenn Harvel, Nelson Agbemava, I Putu Susila, Mehmet Yavuz Yagci  

**Link**: [PDF](https://arxiv.org/pdf/2509.01599)  

**Abstract**: Radiation Detection Systems (RDSs) are used to measure and detect abnormal levels of radioactive material in the environment. These systems are used in many applications to mitigate threats posed by high levels of radioactive material. However, these systems lack protection against malicious external attacks to modify the data. The novelty of applying Intrusion Detection Systems (IDS) in RDSs is a crucial element in safeguarding these critical infrastructures. While IDSs are widely used in networking environments to safeguard against various attacks, their application in RDSs is novel. A common attack on RDSs is Denial of Service (DoS), where the attacker aims to overwhelm the system, causing malfunctioning RDSs. This paper proposes an efficient Machine Learning (ML)-based IDS to detect anomalies in radiation data, focusing on DoS attacks. This work explores the use of sampling methods to create a simulated DoS attack based on a real radiation dataset, followed by an evaluation of various ML algorithms, including Random Forest, Support Vector Machine (SVM), logistic regression, and Light Gradient-Boosting Machine (LightGBM), to detect DoS attacks on RDSs. LightGBM is emphasized for its superior accuracy and low computational resource consumption, making it particularly suitable for real-time intrusion detection. Additionally, model optimization and TinyML techniques, including feature selection, parallel execution, and random search methods, are used to improve the efficiency of the proposed IDS. Finally, an optimized and efficient LightGBM-based IDS is developed to achieve accurate intrusion detection for RDSs. 

**Abstract (ZH)**: 基于入侵检测系统（IDS）的辐射检测系统（RDSs）中DoS攻击的高效机器学习检测方法 

---
# Securing Radiation Detection Systems with an Efficient TinyML-Based IDS for Edge Devices 

**Title (ZH)**: 基于高效TinyML的IDS保障边缘设备的辐射检测系统安全 

**Authors**: Einstein Rivas Pizarro, Wajiha Zaheer, Li Yang, Khalil El-Khatib, Glenn Harvel  

**Link**: [PDF](https://arxiv.org/pdf/2509.01592)  

**Abstract**: Radiation Detection Systems (RDSs) play a vital role in ensuring public safety across various settings, from nuclear facilities to medical environments. However, these systems are increasingly vulnerable to cyber-attacks such as data injection, man-in-the-middle (MITM) attacks, ICMP floods, botnet attacks, privilege escalation, and distributed denial-of-service (DDoS) attacks. Such threats could compromise the integrity and reliability of radiation measurements, posing significant public health and safety risks. This paper presents a new synthetic radiation dataset and an Intrusion Detection System (IDS) tailored for resource-constrained environments, bringing Machine Learning (ML) predictive capabilities closer to the sensing edge layer of critical infrastructure. Leveraging TinyML techniques, the proposed IDS employs an optimized XGBoost model enhanced with pruning, quantization, feature selection, and sampling. These TinyML techniques significantly reduce the size of the model and computational demands, enabling real-time intrusion detection on low-resource devices while maintaining a reasonable balance between efficiency and accuracy. 

**Abstract (ZH)**: 辐射检测系统（RDSs）在各种环境中确保公共安全方面发挥着关键作用，从核设施到医疗环境。然而，这些系统越来越容易受到数据注入攻击、中间人攻击（MITM）、ICMP洪泛攻击、僵尸网络攻击、权限提升和分布式拒绝服务（DDoS）等网络攻击的威胁。这些威胁可能破坏辐射测量的完整性和可靠性，引发重大的公共健康和安全风险。本文提出了一种新的合成辐射数据集和一种针对资源受限环境的入侵检测系统（IDS），将机器学习（ML）预测能力带到了关键基础设施的感知边缘层。利用TinyML技术，所提出的IDS采用了一种经过优化的XGBoost模型，并结合了剪枝、量化、特征选择和采样的技术。这些TinyML技术显著减小了模型大小和计算需求，能够在低资源设备上实现实时入侵检测，同时保持了效率和准确性的合理平衡。 

---
# From Discord to Harmony: Decomposed Consonance-based Training for Improved Audio Chord Estimation 

**Title (ZH)**: 从discord到和谐：分解谐和性训练方法以改进音频和弦估计 

**Authors**: Andrea Poltronieri, Xavier Serra, Martín Rocamora  

**Link**: [PDF](https://arxiv.org/pdf/2509.01588)  

**Abstract**: Audio Chord Estimation (ACE) holds a pivotal role in music information research, having garnered attention for over two decades due to its relevance for music transcription and analysis. Despite notable advancements, challenges persist in the task, particularly concerning unique characteristics of harmonic content, which have resulted in existing systems' performances reaching a glass ceiling. These challenges include annotator subjectivity, where varying interpretations among annotators lead to inconsistencies, and class imbalance within chord datasets, where certain chord classes are over-represented compared to others, posing difficulties in model training and evaluation. As a first contribution, this paper presents an evaluation of inter-annotator agreement in chord annotations, using metrics that extend beyond traditional binary measures. In addition, we propose a consonance-informed distance metric that reflects the perceptual similarity between harmonic annotations. Our analysis suggests that consonance-based distance metrics more effectively capture musically meaningful agreement between annotations. Expanding on these findings, we introduce a novel ACE conformer-based model that integrates consonance concepts into the model through consonance-based label smoothing. The proposed model also addresses class imbalance by separately estimating root, bass, and all note activations, enabling the reconstruction of chord labels from decomposed outputs. 

**Abstract (ZH)**: 音频和弦估计（ACE）在音乐信息研究中占据关键地位，由于其对音乐转录和分析的相关性，已受到广泛关注超过二十年。尽管取得了显著进展，但在任务中仍存在挑战，特别是和声内容的独特特性，导致现有系统的性能接近天花板。这些挑战包括标注者主观性，不同标注者之间的不同解释导致不一致性，以及和弦数据集中的类别不平衡，某些和弦类别与其它类别相比过度代表，这给模型训练和评估带来了困难。作为第一个贡献，本文评估了和弦标注间的跨标注者一致性，使用了超越传统二元度量的指标。此外，我们提出了一种基于协和性的距离度量，反映谐和标注之间的感知相似性。我们的分析表明，基于协和性的距离度量更能有效地捕捉标注间的音乐意义一致性。在此基础上，我们介绍了一种新颖的ACE模型，该模型通过基于协和性的标签平滑将协和性概念整合到模型中。提出的模型还通过分别估计根音、低音和所有音符的激活来解决类别不平衡问题，从而可以从分解输出中重建和弦标签。 

---
# One-Shot Clustering for Federated Learning Under Clustering-Agnostic Assumption 

**Title (ZH)**: 在集群无假设下的单-shot联邦聚类 

**Authors**: Maciej Krzysztof Zuziak, Roberto Pellungrini, Salvatore Rinzivillo  

**Link**: [PDF](https://arxiv.org/pdf/2509.01587)  

**Abstract**: Federated Learning (FL) is a widespread and well-adopted paradigm of decentralised learning that allows training one model from multiple sources without the need to transfer data between participating clients directly. Since its inception in 2015, it has been divided into numerous subfields that deal with application-specific issues, such as data heterogeneity or resource allocation. One such sub-field, Clustered Federated Learning (CFL), deals with the problem of clustering the population of clients into separate cohorts to deliver personalised models. Although a few remarkable works have been published in this domain, the problem remains largely unexplored, as its basic assumptions and settings differ slightly from those of standard FL. In this work, we present One-Shot Clustered Federated Learning (OCFL), a clustering-agnostic algorithm that can automatically detect the earliest suitable moment for clustering. Our algorithm is based on computing the cosine distance between the gradients of the clients and a temperature measure that detects when the federated model starts to converge. We empirically evaluate our methodology by testing various one-shot clustering algorithms for over forty different tasks on five benchmark datasets. Our experiments showcase the good performance of our approach when used to perform CFL in an automated manner without the need to adjust hyperparameters. We also revisit the practical feasibility of CFL algorithms based on the gradients of the clients, providing firm evidence of the high efficiency of density-based clustering methods when used to differentiate between the loss surfaces of neural networks trained on different distributions. Moreover, by inspecting the feasibility of local explanations generated with the help of GradCAM, we can provide more insights into the relationship between personalisation and the explainability of local predictions. 

**Abstract (ZH)**: 面向梯度的自适应一次性集群联邦学习 

---
# Enabling Down Syndrome Research through a Knowledge Graph-Driven Analytical Framework 

**Title (ZH)**: 基于知识图谱驱动的分析框架 enables 唐氏综合征研究 

**Authors**: Madan Krishnamurthy, Surya Saha, Pierrette Lo, Patricia L. Whetzel, Tursynay Issabekova, Jamed Ferreris Vargas, Jack DiGiovanna, Melissa A Haendel  

**Link**: [PDF](https://arxiv.org/pdf/2509.01565)  

**Abstract**: Trisomy 21 results in Down syndrome, a multifaceted genetic disorder with diverse clinical phenotypes, including heart defects, immune dysfunction, neurodevelopmental differences, and early-onset dementia risk. Heterogeneity and fragmented data across studies challenge comprehensive research and translational discovery. The NIH INCLUDE (INvestigation of Co-occurring conditions across the Lifespan to Understand Down syndromE) initiative has assembled harmonized participant-level datasets, yet realizing their potential requires integrative analytical frameworks. We developed a knowledge graph-driven platform transforming nine INCLUDE studies, comprising 7,148 participants, 456 conditions, 501 phenotypes, and over 37,000 biospecimens, into a unified semantic infrastructure. Cross-resource enrichment with Monarch Initiative data expands coverage to 4,281 genes and 7,077 variants. The resulting knowledge graph contains over 1.6 million semantic associations, enabling AI-ready analysis with graph embeddings and path-based reasoning for hypothesis generation. Researchers can query the graph via SPARQL or natural language interfaces. This framework converts static data repositories into dynamic discovery environments, supporting cross-study pattern recognition, predictive modeling, and systematic exploration of genotype-phenotype relationships in Down syndrome. 

**Abstract (ZH)**: 21三体导致唐氏综合征，这是一种多面的遗传性疾病，具有多种临床表型，包括心脏缺陷、免疫功能障碍、神经发育差异和早发痴呆风险。研究之间的异质性和数据碎片化挑战了综合研究和转化发现。NIH INCLUDE（调查整个生命周期中共病以理解唐氏综合征）倡议汇集了标准化的参与者级数据集，但其实现其潜力需要整合分析框架。我们开发了一个基于知识图谱的平台，将九个INCLUDE研究整合成一个统一的语义基础设施，涉及7,148名参与者、456种条件、501种表型和超过37,000份生物样本。通过与Monarch Initiative数据跨资源丰富，扩展覆盖范围至4,281个基因和7,077个变异。结果生成的知识图谱包含超过160万条语义关联，支持基于图嵌入和路径推理的AI准备分析，以生成假设。研究人员可以通过SPARQL或自然语言接口查询该图谱。此框架将静态数据仓库转化为动态发现环境，支持跨研究模式识别、预测建模和唐氏综合征基因型-表型关系的系统探索。 

---
# In-N-Out: A Parameter-Level API Graph Dataset for Tool Agents 

**Title (ZH)**: In-N-Out: 一个参数级API图形数据集供工具代理使用 

**Authors**: Seungkyu Lee, Nalim Kim, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2509.01560)  

**Abstract**: Tool agents -- LLM-based systems that interact with external APIs -- offer a way to execute real-world tasks. However, as tasks become increasingly complex, these agents struggle to identify and call the correct APIs in the proper order. To tackle this problem, we investigate converting API documentation into a structured API graph that captures API dependencies and leveraging it for multi-tool queries that require compositional API calls. To support this, we introduce In-N-Out, the first expert-annotated dataset of API graphs built from two real-world API benchmarks and their documentation. Using In-N-Out significantly improves performance on both tool retrieval and multi-tool query generation, nearly doubling that of LLMs using documentation alone. Moreover, graphs generated by models fine-tuned on In-N-Out close 90% of this gap, showing that our dataset helps models learn to comprehend API documentation and parameter relationships. Our findings highlight the promise of using explicit API graphs for tool agents and the utility of In-N-Out as a valuable resource. We will release the dataset and code publicly. 

**Abstract (ZH)**: 基于LLM的工具代理——通过外部API交互的系统——提供了一种执行现实世界任务的方式。然而，随着任务变得越来越复杂，这些代理在识别和按正确顺序调用API方面遇到困难。为解决这一问题，我们研究将API文档转换为结构化的API图，该图捕获API依赖关系，并利用其支持多工具查询中的组合API调用。为此，我们引入了In-N-Out，这是首个基于两个真实世界API基准及其文档构建的专家注释API图数据集。使用In-N-Out显著提高了工具检索和多工具查询生成的性能，几乎将仅使用文档的LLMs的性能翻倍。此外，基于In-N-Out微调的模型关闭了该性能差距的90%，表明我们的数据集有助于模型学习理解API文档和参数关系。我们的研究结果凸显了使用显式API图对工具代理的潜力，以及In-N-Out作为有价值资源的实用性。我们将公开发布该数据集和代码。 

---
# Unsupervised Identification and Replay-based Detection (UIRD) for New Category Anomaly Detection in ECG Signal 

**Title (ZH)**: 无需监督的识别与回放基于检测（UIRD）用于ECG信号中新类别异常检测 

**Authors**: Zhangyue Shi, Zekai Wang, Yuxuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.01512)  

**Abstract**: In clinical practice, automatic analysis of electrocardiogram (ECG) is widely applied to identify irregular heart rhythms and other electrical anomalies of the heart, enabling timely intervention and potentially improving clinical outcomes. However, due to the limited samples in certain types of ECG signals, the class imbalance issues pose a challenge for ECG-based detection. In addition, as the volume of patient data grows, long-term storage of all historical data becomes increasingly burdensome as training samples to recognize new patterns and classify existing ECG signals accurately. Therefore, to enhance the performance of anomaly detection while addressing storage limitations, we propose a pseudo-replay based semi-supervised continual learning framework, which consists of two components: unsupervised identification and replay-based detection. For unsupervised identification, an unsupervised generative adversarial network (GAN)-based framework is integrated to detect novel patterns. Besides, instead of directly storing all historical data, a pseudo replay-based learning strategy is proposed which utilizes a generator to learn the data distribution for each individual task. When a new task arises, the generator synthesizes pseudo data representative of previous learnt classes, enabling the model to detect both the existed patterns and the newly presented anomalies. The effectiveness of the proposed framework is validated in four public ECG datasets, which leverages supervised classification problems for anomaly detection. The experimental results show that the developed approach is very promising in identifying novel anomalies while maintaining good performance on detecting existing ECG signals. 

**Abstract (ZH)**: 基于伪重放的半监督持续学习框架：用于心电图异常检测的无监督识别与重放检测 

---
# MSA2-Net: Utilizing Self-Adaptive Convolution Module to Extract Multi-Scale Information in Medical Image Segmentation 

**Title (ZH)**: MSA2-Net: 利用自适应卷积模块提取医学图像分割中的多尺度信息 

**Authors**: Chao Deng, Xiaosen Li, Xiao Qin  

**Link**: [PDF](https://arxiv.org/pdf/2509.01498)  

**Abstract**: The nnUNet segmentation framework adeptly adjusts most hyperparameters in training scripts automatically, but it overlooks the tuning of internal hyperparameters within the segmentation network itself, which constrains the model's ability to generalize. Addressing this limitation, this study presents a novel Self-Adaptive Convolution Module that dynamically adjusts the size of the convolution kernels depending on the unique fingerprints of different datasets. This adjustment enables the MSA2-Net, when equipped with this module, to proficiently capture both global and local features within the feature maps. Self-Adaptive Convolution Module is strategically integrated into two key components of the MSA2-Net: the Multi-Scale Convolution Bridge and the Multi-Scale Amalgamation Decoder. In the MSConvBridge, the module enhances the ability to refine outputs from various stages of the CSWin Transformer during the skip connections, effectively eliminating redundant data that could potentially impair the decoder's performance. Simultaneously, the MSADecoder, utilizing the module, excels in capturing detailed information of organs varying in size during the decoding phase. This capability ensures that the decoder's output closely reproduces the intricate details within the feature maps, thus yielding highly accurate segmentation images. MSA2-Net, bolstered by this advanced architecture, has demonstrated exceptional performance, achieving Dice coefficient scores of 86.49\%, 92.56\%, 93.37\%, and 92.98\% on the Synapse, ACDC, Kvasir, and Skin Lesion Segmentation (ISIC2017) datasets, respectively. This underscores MSA2-Net's robustness and precision in medical image segmentation tasks across various datasets. 

**Abstract (ZH)**: Self-Adaptive Convolution Module for Enhancing Generalization and Feature Capture in MSA2-Net 

---
# An Information-Flow Perspective on Explainability Requirements: Specification and Verification 

**Title (ZH)**: 从信息流视角探讨可解释性需求：规范与验证 

**Authors**: Bernd Finkbeiner, Hadar Frenkel, Julian Siber  

**Link**: [PDF](https://arxiv.org/pdf/2509.01479)  

**Abstract**: Explainable systems expose information about why certain observed effects are happening to the agents interacting with them. We argue that this constitutes a positive flow of information that needs to be specified, verified, and balanced against negative information flow that may, e.g., violate privacy guarantees. Since both explainability and privacy require reasoning about knowledge, we tackle these tasks with epistemic temporal logic extended with quantification over counterfactual causes. This allows us to specify that a multi-agent system exposes enough information such that agents acquire knowledge on why some effect occurred. We show how this principle can be used to specify explainability as a system-level requirement and provide an algorithm for checking finite-state models against such specifications. We present a prototype implementation of the algorithm and evaluate it on several benchmarks, illustrating how our approach distinguishes between explainable and unexplainable systems, and how it allows to pose additional privacy requirements. 

**Abstract (ZH)**: 可知系统揭示有关某些观察到的效果发生原因的信息给与其交互的代理。我们认为这构成了一种正向信息流，需要对其进行规定、验证，并与可能违反隐私保证等负向信息流进行平衡。由于可解释性和隐私都涉及关于知识的推理，我们使用扩展了对事实假设原因量化处理的知识时序逻辑来应对这些任务。这使我们能够规定一个多代理系统需要揭示足够的信息，以使代理获得某些效果发生原因的知识。我们展示了这一原则如何被用于将可解释性作为系统级要求进行规定，并提供了一种针对此类规定检查有限状态模型的算法。我们呈现了该算法的原型实现，并在多个基准上进行了评估，阐明了我们的方法如何区分可解释和不可解释的系统，并如何允许提出额外的隐私要求。 

---
# Unnoticeable Community Deception via Multi-objective Optimization 

**Title (ZH)**: 多目标优化下的隐形社区欺骗 

**Authors**: Junyuan Fang, Huimin Liu, Yueqi Peng, Jiajing Wu, Zibin Zheng, Chi K. Tse  

**Link**: [PDF](https://arxiv.org/pdf/2509.01438)  

**Abstract**: Community detection in graphs is crucial for understanding the organization of nodes into densely connected clusters. While numerous strategies have been developed to identify these clusters, the success of community detection can lead to privacy and information security concerns, as individuals may not want their personal information exposed. To address this, community deception methods have been proposed to reduce the effectiveness of detection algorithms. Nevertheless, several limitations, such as the rationality of evaluation metrics and the unnoticeability of attacks, have been ignored in current deception methods. Therefore, in this work, we first investigate the limitations of the widely used deception metric, i.e., the decrease of modularity, through empirical studies. Then, we propose a new deception metric, and combine this new metric together with the attack budget to model the unnoticeable community deception task as a multi-objective optimization problem. To further improve the deception performance, we propose two variant methods by incorporating the degree-biased and community-biased candidate node selection mechanisms. Extensive experiments on three benchmark datasets demonstrate the superiority of the proposed community deception strategies. 

**Abstract (ZH)**: 图中社区检测对于理解节点之间的密集连接簇组织至关重要。虽然已经开发出多种策略来识别这些簇，但社区检测的成功可能会引发隐私和信息安全问题，因为个人可能不希望其个人信息被曝光。为了应对这一挑战，已经提出了社区欺骗方法来降低检测算法的有效性。然而，现有欺骗方法在评价指标的合理性和攻击的不可察觉性方面存在一些局限性，这些局限性尚未得到充分关注。因此，在本文中，我们首先通过实证研究调查广泛使用的欺骗指标，即模块度的降低。然后，我们提出了一个新的欺骗指标，并结合该新指标与攻击预算，将不可察觉的社区欺骗任务建模为一个多目标优化问题。为了进一步提高欺骗性能，我们提出了两种变种方法，通过引入度偏差和社区偏差的候选节点选择机制。在三个基准数据集上的广泛实验表明，所提出的社区欺骗策略具有优势。 

---
# DCA: Graph-Guided Deep Embedding Clustering for Brain Atlases 

**Title (ZH)**: DCA: 基于图引导的深度嵌入聚类方法用于脑图谱 

**Authors**: Mo Wang, Kaining Peng, Jingsheng Tang, Hongkai Wen, Quanying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.01426)  

**Abstract**: Brain atlases are essential for reducing the dimensionality of neuroimaging data and enabling interpretable analysis. However, most existing atlases are predefined, group-level templates with limited flexibility and resolution. We present Deep Cluster Atlas (DCA), a graph-guided deep embedding clustering framework for generating individualized, voxel-wise brain parcellations. DCA combines a pretrained autoencoder with spatially regularized deep clustering to produce functionally coherent and spatially contiguous regions. Our method supports flexible control over resolution and anatomical scope, and generalizes to arbitrary brain structures. We further introduce a standardized benchmarking platform for atlas evaluation, using multiple large-scale fMRI datasets. Across multiple datasets and scales, DCA outperforms state-of-the-art atlases, improving functional homogeneity by 98.8\% and silhouette coefficient by 29\%, and achieves superior performance in downstream tasks such as autism diagnosis and cognitive decoding. Codes and models will be released soon. 

**Abstract (ZH)**: 脑图谱对于降低神经影像数据维度并实现可解释分析至关重要。然而，现有大多数图谱是预定义的分组级模板，具有有限的灵活度和分辨率。我们提出了一种图引导的深度嵌入聚类框架Deep Cluster Atlas (DCA)，用于生成个体化的体素级脑分区。DCA结合了预训练的自编码器和空间正则化的深度聚类，以生成功能连贯且空间连续的区域。该方法支持灵活的分辨率和解剖范围控制，并可应用于任意脑结构。我们进一步引入了一个标准化的图谱评估基准平台，使用多个大规模fMRI数据集。在多个数据集和规模上，DCA在功能同质性和轮廓系数方面分别优于最先进的图谱98.8%和29%，并在自闭症诊断和认知解码等下游任务中表现出优异性能。代码和模型将很快发布。 

---
# CabinSep: IR-Augmented Mask-Based MVDR for Real-Time In-Car Speech Separation with Distributed Heterogeneous Arrays 

**Title (ZH)**: CabinSep: 基于IR增强的掩码导向MVDR实时车内语音分离方法及分布式异构阵列应用 

**Authors**: Runduo Han, Yanxin Hu, Yihui Fu, Zihan Zhang, Yukai Jv, Li Chen, Lei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2509.01399)  

**Abstract**: Separating overlapping speech from multiple speakers is crucial for effective human-vehicle interaction. This paper proposes CabinSep, a lightweight neural mask-based minimum variance distortionless response (MVDR) speech separation approach, to reduce speech recognition errors in back-end automatic speech recognition (ASR) models. Our contributions are threefold: First, we utilize channel information to extract spatial features, which improves the estimation of speech and noise masks. Second, we employ MVDR during inference, reducing speech distortion to make it more ASR-friendly. Third, we introduce a data augmentation method combining simulated and real-recorded impulse responses (IRs), improving speaker localization at zone boundaries and further reducing speech recognition errors. With a computational complexity of only 0.4 GMACs, CabinSep achieves a 17.5% relative reduction in speech recognition error rate in a real-recorded dataset compared to the state-of-the-art DualSep model. Demos are available at: this https URL. 

**Abstract (ZH)**: 从多个说话人中分离重叠语音对于有效的汽车-人类交互至关重要。本文提出了一种轻量级神经掩码基于最小方差无畸变响应(MVDR)的语音分离方法CabinSep，以减少后端自动语音识别(ASR)模型中的语音识别错误。我们的贡献主要包括三个方面：首先，我们利用信道信息提取空间特征，从而提高语音和噪声掩码的估计；其次，在推理过程中采用MVDR，减少语音失真，使其更符合ASR的需求；第三，我们引入了一种结合模拟和真实记录冲激响应(IR)的数据增强方法，提高了在区域边界处的说话人定位能力，并进一步减少了语音识别错误。CabinSep仅具有0.4 GMACs的计算复杂度，在一个真实录音数据集上，相比目前最先进的DualSep模型，实现了17.5%的相对语音识别错误率降低。更多示例请访问：这个链接。 

---
# Anomaly detection in network flows using unsupervised online machine learning 

**Title (ZH)**: 使用无监督在线机器学习在网络流量中检测异常 

**Authors**: Alberto Miguel-Diez, Adrián Campazas-Vega, Ángel Manuel Guerrero-Higueras, Claudia Álvarez-Aparicio, Vicente Matellán-Olivera  

**Link**: [PDF](https://arxiv.org/pdf/2509.01375)  

**Abstract**: Nowadays, the volume of network traffic continues to grow, along with the frequency and sophistication of attacks. This scenario highlights the need for solutions capable of continuously adapting, since network behavior is dynamic and changes over time. This work presents an anomaly detection model for network flows using unsupervised machine learning with online learning capabilities. This approach allows the system to dynamically learn the normal behavior of the network and detect deviations without requiring labeled data, which is particularly useful in real-world environments where traffic is constantly changing and labeled data is scarce. The model was implemented using the River library with a One-Class SVM and evaluated on the NF-UNSW-NB15 dataset and its extended version v2, which contain network flows labeled with different attack categories. The results show an accuracy above 98%, a false positive rate below 3.1%, and a recall of 100% in the most advanced version of the dataset. In addition, the low processing time per flow (<0.033 ms) demonstrates the feasibility of the approach for real-time applications. 

**Abstract (ZH)**: 现网流量中基于在线学习的无监督机器学习异常检测模型 

---
# Causal Sensitivity Identification using Generative Learning 

**Title (ZH)**: 使用生成学习进行因果敏感性识别 

**Authors**: Soma Bandyopadhyay, Sudeshna Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2509.01352)  

**Abstract**: In this work, we propose a novel generative method to identify the causal impact and apply it to prediction tasks. We conduct causal impact analysis using interventional and counterfactual perspectives. First, applying interventions, we identify features that have a causal influence on the predicted outcome, which we refer to as causally sensitive features, and second, applying counterfactuals, we evaluate how changes in the cause affect the effect. Our method exploits the Conditional Variational Autoencoder (CVAE) to identify the causal impact and serve as a generative predictor. We are able to reduce confounding bias by identifying causally sensitive features. We demonstrate the effectiveness of our method by recommending the most likely locations a user will visit next in their spatiotemporal trajectory influenced by the causal relationships among various features. Experiments on the large-scale GeoLife [Zheng et al., 2010] dataset and the benchmark Asia Bayesian network validate the ability of our method to identify causal impact and improve predictive performance. 

**Abstract (ZH)**: 本研究提出了一种新颖的生成方法来识别因果影响，并将其应用于预测任务。我们从干预性和反事实性视角进行因果影响分析。首先，通过干预识别出对预测结果有因果影响的特征，我们称之为因果敏感特征；其次，通过反事实分析评估因果变量变化对结果的影响。我们的方法利用条件变分自编码器（CVAE）来识别因果影响，并作为生成预测器。我们能够通过识别因果敏感特征来降低混杂偏差。我们通过推荐用户在受各种特征之间因果关系影响的时空轨迹中下一次最可能访问的位置，展示了该方法的有效性。在大规模GeoLife数据集和基准Asia贝叶斯网络上的实验验证了该方法在识别因果影响和提高预测性能方面的能力。 

---
# AT Loss: Advanced Torrential Loss Function for Precipitation Forecasting 

**Title (ZH)**: AT损失：先进的 Torrential 降水损失函数 

**Authors**: Jaeho Choi, Hyeri Kim, Kwang-Ho Kim, Jaesung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.01348)  

**Abstract**: Accurate precipitation forecasting is becoming increasingly important in the context of climate change. In response, machine learning-based approaches have recently gained attention as an emerging alternative to traditional methods such as numerical weather prediction and climate models. Nonetheless, many recent approaches still rely on off-the-shelf loss functions, and even the more advanced ones merely involve optimization processes based on the critical success index (CSI). The problem, however, is that CSI may become ineffective during extended dry periods when precipitation remains below the threshold, rendering it less than ideal as a criterion for optimization. To address this limitation, we introduce a simple penalty expression and reinterpret it as a quadratic unconstrained binary optimization (QUBO) formulation. Ultimately, the resulting QUBO formulation is relaxed into a differentiable advanced torrential (AT) loss function through an approximation process. The proposed AT loss demonstrates its superiority through the Lipschitz constant, forecast performance evaluations, consistency experiments, and ablation studies with the operational model. 

**Abstract (ZH)**: 基于机器学习的精确降水预报在气候变化背景下的重要性日益增加。面对这一挑战，近年来基于机器学习的方法逐渐成为一种新兴替代传统方法（如数值天气预报和气候模型）的新选择。然而，许多近期的方法仍然依赖于现成的损失函数，即使是更先进的方法也仅基于关键成功率（CSI）进行优化。然而，CSI在长时间干旱期间可能无效，此时降水低于阈值，使其作为优化准则不如理想。为此，我们引入一个简单的惩罚表达式，并将其重新解释为无约束二元优化（QUBO）形式。最终，通过近似过程，该QUBO形式被松弛为一个可微的先进暴风雨（AT）损失函数。通过Lipschitz常数、预测性能评估、一致性实验以及与操作模型的消融研究，提出的AT损失展示了其优越性。 

---
# Street-Level Geolocalization Using Multimodal Large Language Models and Retrieval-Augmented Generation 

**Title (ZH)**: 基于多模态大规模语言模型和检索增强生成的街景级地理定位 

**Authors**: Yunus Serhat Bicakci, Joseph Shingleton, Anahid Basiri  

**Link**: [PDF](https://arxiv.org/pdf/2509.01341)  

**Abstract**: Street-level geolocalization from images is crucial for a wide range of essential applications and services, such as navigation, location-based recommendations, and urban planning. With the growing popularity of social media data and cameras embedded in smartphones, applying traditional computer vision techniques to localize images has become increasingly challenging, yet highly valuable. This paper introduces a novel approach that integrates open-weight and publicly accessible multimodal large language models with retrieval-augmented generation. The method constructs a vector database using the SigLIP encoder on two large-scale datasets (EMP-16 and OSV-5M). Query images are augmented with prompts containing both similar and dissimilar geolocation information retrieved from this database before being processed by the multimodal large language models. Our approach has demonstrated state-of-the-art performance, achieving higher accuracy compared against three widely used benchmark datasets (IM2GPS, IM2GPS3k, and YFCC4k). Importantly, our solution eliminates the need for expensive fine-tuning or retraining and scales seamlessly to incorporate new data sources. The effectiveness of retrieval-augmented generation-based multimodal large language models in geolocation estimation demonstrated by this paper suggests an alternative path to the traditional methods which rely on the training models from scratch, opening new possibilities for more accessible and scalable solutions in GeoAI. 

**Abstract (ZH)**: 基于图像的城市级地理定位对于导航、位置ベース的推荐以及城市规划等广泛的重要应用程序和服务至关重要。随着社交媒体数据流行度的增加和智能手机内置摄像头的普及，传统的计算机视觉技术在图像定位方面的应用变得越来越具有挑战性，但同时也极具价值。本文介绍了一种新颖的方法，该方法结合了开源权重和公共访问的多模态大型语言模型，并利用检索增强生成技术。该方法使用SigLIP编码器在两个大规模数据集（EMP-16和OSV-5M）上构建向量数据库。查询图像在处理前会通过检索到的包含相似和不似地理定位信息的提示进行增强。我们的方法显示出最先进的性能，相比三个广泛使用的基准数据集（IM2GPS、IM2GPS3k和YFCC4k）实现了更高的准确性。重要的是，我们的解决方案无需昂贵的微调或重新训练，且可无缝扩展以集成新的数据源。本文通过检索增强生成为基础的多模态大型语言模型在地理定位估计中的有效性，展示了传统方法（需要从头开始训练模型）的一种替代路径，为地理人工智能（GeoAI）提供了更加易于访问和可扩展的解决方案。 

---
# Multitask Battery Management with Flexible Pretraining 

**Title (ZH)**: 多任务电池管理与灵活预训练 

**Authors**: Hong Lu, Jiali Chen, Jingzhao Zhang, Guannan He, Xuebing Han, Minggao Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01323)  

**Abstract**: Industrial-scale battery management involves various types of tasks, such as estimation, prediction, and system-level diagnostics. Each task employs distinct data across temporal scales, sensor resolutions, and data channels. Building task-specific methods requires a great deal of data and engineering effort, which limits the scalability of intelligent battery management. Here we present the Flexible Masked Autoencoder (FMAE), a flexible pretraining framework that can learn with missing battery data channels and capture inter-correlations across data snippets. FMAE learns unified battery representations from heterogeneous data and can be adopted by different tasks with minimal data and engineering efforts. Experimentally, FMAE consistently outperforms all task-specific methods across five battery management tasks with eleven battery datasets. On remaining life prediction tasks, FMAE uses 50 times less inference data while maintaining state-of-the-art results. Moreover, when real-world data lack certain information, such as system voltage, FMAE can still be applied with marginal performance impact, achieving comparable results with the best hand-crafted features. FMAE demonstrates a practical route to a flexible, data-efficient model that simplifies real-world multi-task management of dynamical systems. 

**Abstract (ZH)**: 工业规模电池管理涉及多种类型的任务，如估计、预测和系统级诊断。每种任务在不同时间尺度、传感器分辨率和数据通道上使用不同的数据。构建任务特定的方法需要大量的数据和工程努力，这限制了智能电池管理的可扩展性。我们提出了灵活的掩蔽自编码器（FMAE），这是一种灵活的预训练框架，可以学习缺失的电池数据通道，并捕捉数据片段之间的互相关关系。FMAE从异构数据中学习统一的电池表示，并且可以通过最小的数据和工程努力应用于不同的任务。实验结果显示，FMAE在五个电池管理任务和 eleven 电池数据集上一贯优于所有任务特定方法。在剩余寿命预测任务中，FMAE使用50倍少的推理数据，同时保持最先进的结果。此外，当实际数据缺乏某些信息，如系统电压时，FMAE仍然可以应用，并且性能影响较小，可以达到与最佳手工特征相似的结果。FMAE展示了通往灵活、数据高效模型的实用途径，该模型简化了动态系统的多任务管理。 

---
# Towards Trustworthy Vital Sign Forecasting: Leveraging Uncertainty for Prediction Intervals 

**Title (ZH)**: 可信的生命体征预测：利用不确定性获取预测区间 

**Authors**: Li Rong Wang, Thomas C. Henderson, Yew Soon Ong, Yih Yng Ng, Xiuyi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2509.01319)  

**Abstract**: Vital signs, such as heart rate and blood pressure, are critical indicators of patient health and are widely used in clinical monitoring and decision-making. While deep learning models have shown promise in forecasting these signals, their deployment in healthcare remains limited in part because clinicians must be able to trust and interpret model outputs. Without reliable uncertainty quantification -- particularly calibrated prediction intervals (PIs) -- it is unclear whether a forecasted abnormality constitutes a meaningful warning or merely reflects model noise, hindering clinical decision-making. To address this, we present two methods for deriving PIs from the Reconstruction Uncertainty Estimate (RUE), an uncertainty measure well-suited to vital-sign forecasting due to its sensitivity to data shifts and support for label-free calibration. Our parametric approach assumes that prediction errors and uncertainty estimates follow a Gaussian copula distribution, enabling closed-form PI computation. Our non-parametric approach, based on k-nearest neighbours (KNN), empirically estimates the conditional error distribution using similar validation instances. We evaluate these methods on two large public datasets with minute- and hour-level sampling, representing high- and low-frequency health signals. Experiments demonstrate that the Gaussian copula method consistently outperforms conformal prediction baselines on low-frequency data, while the KNN approach performs best on high-frequency data. These results underscore the clinical promise of RUE-derived PIs for delivering interpretable, uncertainty-aware vital sign forecasts. 

**Abstract (ZH)**: 基于重建不确定性估计的预测区间方法在生理指标预测中的临床应用前景 

---
# Animer une base de connaissance: des ontologies aux mod{è}les d'I.A. g{é}n{é}rative 

**Title (ZH)**: 动画知识库：从本体学到生成型AI模型 

**Authors**: Peter Stockinger  

**Link**: [PDF](https://arxiv.org/pdf/2509.01304)  

**Abstract**: In a context where the social sciences and humanities are experimenting with non-anthropocentric analytical frames, this article proposes a semiotic (structural) reading of the hybridization between symbolic AI and neural (or sub-symbolic) AI based on a field of application: the design and use of a knowledge base for area studies. We describe the LaCAS ecosystem -- Open Archives in Linguistic and Cultural Studies (thesaurus; RDF/OWL ontology; LOD services; harvesting; expertise; publication), deployed at Inalco (National Institute for Oriental Languages and Civilizations) in Paris with the Okapi (Open Knowledge and Annotation Interface) software environment from Ina (National Audiovisual Institute), which now has around 160,000 documentary resources and ten knowledge macro-domains grouping together several thousand knowledge objects. We illustrate this approach using the knowledge domain ''Languages of the world'' (~540 languages) and the knowledge object ''Quechua (language)''. On this basis, we discuss the controlled integration of neural tools, more specifically generative tools, into the life cycle of a knowledge base: assistance with data localization/qualification, index extraction and aggregation, property suggestion and testing, dynamic file generation, and engineering of contextualized prompts (generic, contextual, explanatory, adjustment, procedural) aligned with a domain ontology. We outline an ecosystem of specialized agents capable of animating the database while respecting its symbolic constraints, by articulating model-driven and data-driven methods. 

**Abstract (ZH)**: 在社会科学与人文科学探索非人类中心主义分析框架的背景下，本文提出了一种基于应用领域的符号（结构）学视角对符号AI与神经（或亚符号）AI混合进行解读的方法：以区域研究的知识库设计与应用为例。 

---
# Building surrogate models using trajectories of agents trained by Reinforcement Learning 

**Title (ZH)**: 使用强化学习训练的代理轨迹构建代理模型 

**Authors**: Julen Cestero, Marco Quartulli, Marcello Restelli  

**Link**: [PDF](https://arxiv.org/pdf/2509.01285)  

**Abstract**: Sample efficiency in the face of computationally expensive simulations is a common concern in surrogate modeling. Current strategies to minimize the number of samples needed are not as effective in simulated environments with wide state spaces. As a response to this challenge, we propose a novel method to efficiently sample simulated deterministic environments by using policies trained by Reinforcement Learning. We provide an extensive analysis of these surrogate-building strategies with respect to Latin-Hypercube sampling or Active Learning and Kriging, cross-validating performances with all sampled datasets. The analysis shows that a mixed dataset that includes samples acquired by random agents, expert agents, and agents trained to explore the regions of maximum entropy of the state transition distribution provides the best scores through all datasets, which is crucial for a meaningful state space representation. We conclude that the proposed method improves the state-of-the-art and clears the path to enable the application of surrogate-aided Reinforcement Learning policy optimization strategies on complex simulators. 

**Abstract (ZH)**: 在计算昂贵的模拟环境中，样本效率是代理模型中的一个常见关注点。面对宽状态空间的模拟环境，当前减少所需样本数量的策略效果不佳。为应对这一挑战，我们提出了一种新颖的方法，通过使用强化学习训练的策略来高效地模拟确定性环境。我们对这些代理构建策略进行了详细分析，与Latin-Hypercube采样、主动学习和Kriging进行了性能交叉验证。分析显示，包含由随机智能体、专家智能体及探索状态转移分布最大熵区域的智能体获取的样本的混合数据集，在所有数据集中提供了最优得分，这对于有意义的状态空间表示至关重要。我们得出结论，所提出的方法改进了现有技术，并为在复杂模拟器中应用代理辅助的强化学习策略铺平了道路。 

---
# Multi-Agent Reinforcement Learning for Task Offloading in Wireless Edge Networks 

**Title (ZH)**: 无线边缘网络中多代理强化学习的任务卸载 

**Authors**: Andrea Fox, Francesco De Pellegrini, Eitan Altman  

**Link**: [PDF](https://arxiv.org/pdf/2509.01257)  

**Abstract**: In edge computing systems, autonomous agents must make fast local decisions while competing for shared resources. Existing MARL methods often resume to centralized critics or frequent communication, which fail under limited observability and communication constraints. We propose a decentralized framework in which each agent solves a constrained Markov decision process (CMDP), coordinating implicitly through a shared constraint vector. For the specific case of offloading, e.g., constraints prevent overloading shared server resources. Coordination constraints are updated infrequently and act as a lightweight coordination mechanism. They enable agents to align with global resource usage objectives but require little direct communication. Using safe reinforcement learning, agents learn policies that meet both local and global goals. We establish theoretical guarantees under mild assumptions and validate our approach experimentally, showing improved performance over centralized and independent baselines, especially in large-scale settings. 

**Abstract (ZH)**: 在边缘计算系统中，自治代理必须在竞争共享资源的同时快速做出本地决策。现有的一些多智能体强化学习方法往往依赖于中心化的评论者或频繁的通信，在观测有限和通信受限的情况下往往失效。我们提出了一种去中心化的框架，在该框架中每个代理解决一个受限马尔可夫决策过程(CMDP)，并通过一个共享的约束向量进行隐式的协调。对于特定的卸载案例，例如，约束防止过度利用共享的服务器资源。协调约束 infrequently 更新，并作为一种轻量级的协调机制。它们使代理能够与全局资源使用目标对齐，但需要很少的直接通信。通过安全的强化学习，代理学习能够同时满足局部和全局目标的策略。在较弱的假设下建立理论保证，并通过实验证明了该方法在大型场景中的优越性能，特别是在与中心化和独立基准方法相比时。 

---
# RT-DETRv2 Explained in 8 Illustrations 

**Title (ZH)**: RT-DETRv2 图解解析 

**Authors**: Ethan Qi Yang Chua, Jen Hong Tan  

**Link**: [PDF](https://arxiv.org/pdf/2509.01241)  

**Abstract**: Object detection architectures are notoriously difficult to understand, often more so than large language models. While RT-DETRv2 represents an important advance in real-time detection, most existing diagrams do little to clarify how its components actually work and fit together. In this article, we explain the architecture of RT-DETRv2 through a series of eight carefully designed illustrations, moving from the overall pipeline down to critical components such as the encoder, decoder, and multi-scale deformable attention. Our goal is to make the existing one genuinely understandable. By visualizing the flow of tensors and unpacking the logic behind each module, we hope to provide researchers and practitioners with a clearer mental model of how RT-DETRv2 works under the hood. 

**Abstract (ZH)**: RT-DETRv2检测架构：通过八幅精心设计的插图解释其工作机制 

---
# Preserving Vector Space Properties in Dimensionality Reduction: A Relationship Preserving Loss Framework 

**Title (ZH)**: 保留向量空间属性的降维方法：一种关系保留损失框架 

**Authors**: Eddi Weinwurm, Alexander Kovalenko  

**Link**: [PDF](https://arxiv.org/pdf/2509.01198)  

**Abstract**: Dimensionality reduction can distort vector space properties such as orthogonality and linear independence, which are critical for tasks including cross-modal retrieval, clustering, and classification. We propose a Relationship Preserving Loss (RPL), a loss function that preserves these properties by minimizing discrepancies between relationship matrices (e.g., Gram or cosine) of high-dimensional data and their low-dimensional embeddings. RPL trains neural networks for non-linear projections and is supported by error bounds derived from matrix perturbation theory. Initial experiments suggest that RPL reduces embedding dimensions while largely retaining performance on downstream tasks, likely due to its preservation of key vector space properties. While we describe here the use of RPL in dimensionality reduction, this loss can also be applied more broadly, for example to cross-domain alignment and transfer learning, knowledge distillation, fairness and invariance, dehubbing, graph and manifold learning, and federated learning, where distributed embeddings must remain geometrically consistent. 

**Abstract (ZH)**: 维度减少可能会扭曲向量空间属性，如正交性和线性独立性，这些属性对于跨模态检索、聚类和分类等任务至关重要。我们提出了一种保持关系损失（RPL），这是一种通过最小化高维数据与其低维嵌入之间的关系矩阵（例如Gram矩阵或余弦矩阵）差异来保持这些属性的损失函数。RPL用于训练非线性投影，并通过矩阵扰动理论推导出的误差界提供支持。初步实验表明，RPL可以在很大程度上保留下游任务的性能的同时减少嵌入维度，很可能是因为它保留了关键的向量空间属性。虽然在这里我们描述了RPL在维度减少中的应用，但此损失函数也可以更广泛地应用于域间对齐和迁移学习、知识蒸馏、公平性和不变性、解耦、图和流形学习以及联邦学习等领域，其中分布式嵌入必须保持几何一致性。 

---
# Statutory Construction and Interpretation for Artificial Intelligence 

**Title (ZH)**: 人工智能的立法构造与解释 

**Authors**: Luxi He, Nimra Nadeem, Michel Liao, Howard Chen, Danqi Chen, Mariano-Florentino Cuéllar, Peter Henderson  

**Link**: [PDF](https://arxiv.org/pdf/2509.01186)  

**Abstract**: AI systems are increasingly governed by natural language principles, yet a key challenge arising from reliance on language remains underexplored: interpretive ambiguity. As in legal systems, ambiguity arises both from how these principles are written and how they are applied. But while legal systems use institutional safeguards to manage such ambiguity, such as transparent appellate review policing interpretive constraints, AI alignment pipelines offer no comparable protections. Different interpretations of the same rule can lead to inconsistent or unstable model behavior. Drawing on legal theory, we identify key gaps in current alignment pipelines by examining how legal systems constrain ambiguity at both the rule creation and rule application steps. We then propose a computational framework that mirrors two legal mechanisms: (1) a rule refinement pipeline that minimizes interpretive disagreement by revising ambiguous rules (analogous to agency rulemaking or iterative legislative action), and (2) prompt-based interpretive constraints that reduce inconsistency in rule application (analogous to legal canons that guide judicial discretion). We evaluate our framework on a 5,000-scenario subset of the WildChat dataset and show that both interventions significantly improve judgment consistency across a panel of reasonable interpreters. Our approach offers a first step toward systematically managing interpretive ambiguity, an essential step for building more robust, law-following AI systems. 

**Abstract (ZH)**: AI系统日益受到自然语言原则的治理，但依赖语言带来的解释歧义这一关键挑战尚未得到充分探索。借鉴法律理论，我们通过分析法律系统在规则制定和规则应用步骤中如何限制歧义，识别当前对齐管道中的关键欠缺，并提出一个计算框架，该框架借鉴了两种法律机制：（1）通过修订歧义规则以最小化解释分歧的规则优化管道（类似于机构规则制定或迭代立法行动），（2）基于提示的解释约束，减少规则应用中的不一致性（类似于指导司法裁量的法律原则）。我们在WildChat数据集的5000个场景子集中评估了该框架，并证明了这两种干预措施显著提高了合理解释者的判断一致性。我们的方法为系统地管理解释歧义迈出了第一步，这是构建更稳健、守法的AI系统的重要一步。 

---
# EZhouNet:A framework based on graph neural network and anchor interval for the respiratory sound event detection 

**Title (ZH)**: EZhouNet：基于图神经网络和锚区间的目标呼吸音事件检测框架 

**Authors**: Yun Chu, Qiuhao Wang, Enze Zhou, Qian Liu, Gang Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.01153)  

**Abstract**: Auscultation is a key method for early diagnosis of respiratory and pulmonary diseases, relying on skilled healthcare professionals. However, the process is often subjective, with variability between experts. As a result, numerous deep learning-based automatic classification methods have emerged, most of which focus on respiratory sound classification. In contrast, research on respiratory sound event detection remains limited. Existing sound event detection methods typically rely on frame-level predictions followed by post-processing to generate event-level outputs, making interval boundaries challenging to learn directly. Furthermore, many approaches can only handle fixed-length audio, lim- iting their applicability to variable-length respiratory sounds. Additionally, the impact of respiratory sound location information on detection performance has not been extensively explored. To address these issues, we propose a graph neural network-based framework with anchor intervals, capable of handling variable-length audio and providing more precise temporal localization for abnormal respi- ratory sound events. Our method improves both the flexibility and applicability of respiratory sound detection. Experiments on the SPRSound 2024 and HF Lung V1 datasets demonstrate the effec- tiveness of the proposed approach, and incorporating respiratory position information enhances the discrimination between abnormal sounds. 

**Abstract (ZH)**: 基于锚区间图神经网络的呼吸音事件检测框架 

---
# MATL-DC: A Multi-domain Aggregation Transfer Learning Framework for EEG Emotion Recognition with Domain-Class Prototype under Unseen Targets 

**Title (ZH)**: MATL-DC：一种基于领域类原型的多域聚合迁移学习框架用于未见目标的EEG情绪识别 

**Authors**: Guangli Li, Canbiao Wu, Zhehao Zhou, Na Tian, Zhen Liang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01135)  

**Abstract**: Emotion recognition based on electroencephalography (EEG) signals is increasingly becoming a key research hotspot in affective Brain-Computer Interfaces (aBCIs). However, the current transfer learning model greatly depends on the source domain and target domain data, which hinder the practical application of emotion recognition. Therefore, we propose a Multi-domain Aggregation Transfer Learning framework for EEG emotion recognition with Domain-Class prototype under unseen targets (MATL-DC). We design the feature decoupling module to decouple class-invariant domain features from domain-invariant class features from shallow features. In the model training stage, the multi-domain aggregation mechanism aggregates the domain feature space to form a superdomain, which enhances the characteristics of emotional EEG signals. In each superdomain, we further extract the class prototype representation by class features. In addition, we adopt the pairwise learning strategy to transform the sample classification problem into the similarity problem between sample pairs, which effectively alleviates the influence of label noise. It is worth noting that the target domain is completely unseen during the training process. In the inference stage, we use the trained domain-class prototypes for inference, and then realize emotion recognition. We rigorously validate it on the publicly available databases (SEED, SEED-IV and SEED-V). The results show that the accuracy of MATL-DC model is 84.70\%, 68.11\% and 61.08\%, respectively. MATL-DC achieves comparable or even better performance than methods that rely on both source and target domains. The source code is available at this https URL. 

**Abstract (ZH)**: 基于电生理信号的多域聚合迁移学习框架：面向未见目标的情感识别（MATL-DC） 

---
# SC-GIR: Goal-oriented Semantic Communication via Invariant Representation Learning 

**Title (ZH)**: SC-GIR: 目标导向的语义通信通过不变表示学习 

**Authors**: Senura Hansaja Wanasekara, Van-Dinh Nguyen, Kok-Seng, M.-Duong Nguyen, Symeon Chatzinotas, Octavia A. Dobre  

**Link**: [PDF](https://arxiv.org/pdf/2509.01119)  

**Abstract**: Goal-oriented semantic communication (SC) aims to revolutionize communication systems by transmitting only task-essential information. However, current approaches face challenges such as joint training at transceivers, leading to redundant data exchange and reliance on labeled datasets, which limits their task-agnostic utility. To address these challenges, we propose a novel framework called Goal-oriented Invariant Representation-based SC (SC-GIR) for image transmission. Our framework leverages self-supervised learning to extract an invariant representation that encapsulates crucial information from the source data, independent of the specific downstream task. This compressed representation facilitates efficient communication while retaining key features for successful downstream task execution. Focusing on machine-to-machine tasks, we utilize covariance-based contrastive learning techniques to obtain a latent representation that is both meaningful and semantically dense. To evaluate the effectiveness of the proposed scheme on downstream tasks, we apply it to various image datasets for lossy compression. The compressed representations are then used in a goal-oriented AI task. Extensive experiments on several datasets demonstrate that SC-GIR outperforms baseline schemes by nearly 10%,, and achieves over 85% classification accuracy for compressed data under different SNR conditions. These results underscore the effectiveness of the proposed framework in learning compact and informative latent representations. 

**Abstract (ZH)**: 面向目标的不变表示语义通信（SC-GIR）：一种图像传输的新框架 

---
# CCE: Confidence-Consistency Evaluation for Time Series Anomaly Detection 

**Title (ZH)**: CCE: 基于置信一致性评估的时间序列异常检测 

**Authors**: Zhijie Zhong, Zhiwen Yu, Yiu-ming Cheung, Kaixiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01098)  

**Abstract**: Time Series Anomaly Detection metrics serve as crucial tools for model evaluation. However, existing metrics suffer from several limitations: insufficient discriminative power, strong hyperparameter dependency, sensitivity to perturbations, and high computational overhead. This paper introduces Confidence-Consistency Evaluation (CCE), a novel evaluation metric that simultaneously measures prediction confidence and uncertainty consistency. By employing Bayesian estimation to quantify the uncertainty of anomaly scores, we construct both global and event-level confidence and consistency scores for model predictions, resulting in a concise CCE metric. Theoretically and experimentally, we demonstrate that CCE possesses strict boundedness, Lipschitz robustness against score perturbations, and linear time complexity $\mathcal{O}(n)$. Furthermore, we establish RankEval, a benchmark for comparing the ranking capabilities of various metrics. RankEval represents the first standardized and reproducible evaluation pipeline that enables objective comparison of evaluation metrics. Both CCE and RankEval implementations are fully open-source. 

**Abstract (ZH)**: 时间序列异常检测评价指标作为模型评估的重要工具，但现有指标存在若干局限性：区分能力不足、对超参数高度依赖、对扰动敏感以及计算开销高。本文引入了一种新的评价指标——置信一致性评价（CCE），该指标同时衡量预测置信度和不确定性一致性。通过使用贝叶斯估计量化异常分数的不确定性，我们构造了全局和事件级的预测置信度和一致性得分，从而形成简洁的CCE指标。理论和实验结果表明，CCE具有严格的有界性、针对评分扰动的Lipschitz鲁棒性和线性时间复杂度O(n)。此外，我们建立了RankEval基准，用于比较各种指标的排序能力。RankEval是首个标准化和可重复的评价管道，能够客观比较评价指标。CCE和RankEval的实现均为开源。 

---
# DRetNet: A Novel Deep Learning Framework for Diabetic Retinopathy Diagnosis 

**Title (ZH)**: DRetNet：一种新型深度学习框架用于糖尿病视网膜病变诊断 

**Authors**: Idowu Paul Okuwobi, Jingyuan Liu, Jifeng Wan, Jiaojiao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01072)  

**Abstract**: Diabetic retinopathy (DR) is a leading cause of blindness worldwide, necessitating early detection to prevent vision loss. Current automated DR detection systems often struggle with poor-quality images, lack interpretability, and insufficient integration of domain-specific knowledge. To address these challenges, we introduce a novel framework that integrates three innovative contributions: (1) Adaptive Retinal Image Enhancement Using Physics-Informed Neural Networks (PINNs): this technique dynamically enhances retinal images by incorporating physical constraints, improving the visibility of critical features such as microaneurysms, hemorrhages, and exudates; (2) Hybrid Feature Fusion Network (HFFN): by combining deep learning embeddings with handcrafted features, HFFN leverages both learned representations and domain-specific knowledge to enhance generalization and accuracy; (3) Multi-Stage Classifier with Uncertainty Quantification: this method breaks down the classification process into logical stages, providing interpretable predictions and confidence scores, thereby improving clinical trust. The proposed framework achieves an accuracy of 92.7%, a precision of 92.5%, a recall of 92.6%, an F1-score of 92.5%, an AUC of 97.8%, a mAP of 0.96, and an MCC of 0.85. Ophthalmologists rated the framework's predictions as highly clinically relevant (4.8/5), highlighting its alignment with real-world diagnostic needs. Qualitative analyses, including Grad-CAM visualizations and uncertainty heatmaps, further enhance the interpretability and trustworthiness of the system. The framework demonstrates robust performance across diverse conditions, including low-quality images, noisy data, and unseen datasets. These features make the proposed framework a promising tool for clinical adoption, enabling more accurate and reliable DR detection in resource-limited settings. 

**Abstract (ZH)**: 糖尿病视网膜病变（DR）是全球致盲的主要原因之一，需要早期检测以防止视力丧失。当前的自动化DR检测系统往往难以处理低质量图像、缺乏可解释性，并且未能充分整合领域特定知识。为应对这些挑战，我们提出了一种新的框架，整合了三项创新贡献：（1）物理信息神经网络（PINNs）驱动的自适应视网膜图像增强：该技术通过结合物理约束动态增强视网膜图像，提高微动脉瘤、出血和渗出物等关键特征的可见性；（2）混合特征融合网络（HFFN）：通过结合深度学习嵌入和手工设计特征，HFFN利用学习到的表示和领域特定知识来提高泛化能力和准确性；（3）具有不确定性量化多阶段分类器：该方法将分类过程分解为逻辑阶段，提供可解释的预测和置信度评分，从而提高临床信任。提出的框架实现了92.7%的准确率、92.5%的精确率、92.6%的召回率、92.5%的F1分数、97.8%的AUC、0.96的mAP和0.85的MCC。眼科医生将框架的预测评为高度临床相关（4.8/5），突显其与实际诊断需求的一致性。定性分析，包括Grad-CAM可视化和不确定性热图，进一步增强了系统的可解释性和可信度。该框架在各种条件下展现出稳健性能，包括低质量图像、噪声数据和未见数据集。这些特性使提出的框架成为临床应用的有前途的工具，在资源有限的环境中实现更准确可靠的DR检测。 

---
# An Economy of AI Agents 

**Title (ZH)**: AI代理经济 

**Authors**: Gillian K. Hadfield, Andrew Koh  

**Link**: [PDF](https://arxiv.org/pdf/2509.01063)  

**Abstract**: In the coming decade, artificially intelligent agents with the ability to plan and execute complex tasks over long time horizons with little direct oversight from humans may be deployed across the economy. This chapter surveys recent developments and highlights open questions for economists around how AI agents might interact with humans and with each other, shape markets and organizations, and what institutions might be required for well-functioning markets. 

**Abstract (ZH)**: 在未来十年，能够在长时间尺度上规划和执行复杂任务且只需少量直接监督的人工智能代理可能将在经济领域得到广泛应用。本章回顾了近期的发展，并探讨了经济学家在人工智能代理如何与人类和其他代理互动、塑造市场和组织结构以及为此可能需要哪些制度方面需要关注的开放问题。 

---
# Q-Learning--Driven Adaptive Rewiring for Cooperative Control in Heterogeneous Networks 

**Title (ZH)**: 基于Q-Learning的自适应重连线性在异构网络中的合作控制 

**Authors**: Yi-Ning Weng, Hsuan-Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.01057)  

**Abstract**: Cooperation emergence in multi-agent systems represents a fundamental statistical physics problem where microscopic learning rules drive macroscopic collective behavior transitions. We propose a Q-learning-based variant of adaptive rewiring that builds on mechanisms studied in the literature. This method combines temporal difference learning with network restructuring so that agents can optimize strategies and social connections based on interaction histories. Through neighbor-specific Q-learning, agents develop sophisticated partnership management strategies that enable cooperator cluster formation, creating spatial separation between cooperative and defective regions. Using power-law networks that reflect real-world heterogeneous connectivity patterns, we evaluate emergent behaviors under varying rewiring constraint levels, revealing distinct cooperation patterns across parameter space rather than sharp thermodynamic transitions. Our systematic analysis identifies three behavioral regimes: a permissive regime (low constraints) enabling rapid cooperative cluster formation, an intermediate regime with sensitive dependence on dilemma strength, and a patient regime (high constraints) where strategic accumulation gradually optimizes network structure. Simulation results show that while moderate constraints create transition-like zones that suppress cooperation, fully adaptive rewiring enhances cooperation levels through systematic exploration of favorable network configurations. Quantitative analysis reveals that increased rewiring frequency drives large-scale cluster formation with power-law size distributions. Our results establish a new paradigm for understanding intelligence-driven cooperation pattern formation in complex adaptive systems, revealing how machine learning serves as an alternative driving force for spontaneous organization in multi-agent networks. 

**Abstract (ZH)**: 多智能体系统中合作涌现代表了一种基本的统计物理问题，其中微观学习规则驱动宏观集体行为转变。我们提出了一种基于Q学习的自适应重 wiring变体，该方法结合了时差学习与网络重构，使智能体可以根据互动历史优化策略和社会连接。通过邻居特定的Q学习，智能体发展出复杂的伙伴管理策略，从而形成合作集群，并在合作区域和缺陷区域之间创造空间隔离。使用反映现实世界异质连接模式的幂律网络，我们在变化的重 wiring约束水平下评估涌现行为，揭示了参数空间中独特的合作模式而非突变的热力学转变。系统分析确定了三种行为区城：宽松区城（低约束）允许快速合作集群形成，中间区城对困境强度具有敏感依赖性，以及耐性区城（高约束）战略性积累逐渐优化网络结构。仿真结果表明，适度约束创建的过渡区域能抑制合作，而完全自适应重 wiring通过系统探索有利的网络配置增强合作水平。定量分析表明，增加的重 wiring频率推动大规模集群形成，具有幂律大小分布。我们的结果建立了复杂自适应系统中智能驱动的合作模式形成的新范式，揭示了机器学习如何作为一种替代驱动力促进多智能体网络中的自发组织。 

---
# Reinforcement Learning Driven Generalizable Feature Representation for Cross-User Activity Recognition 

**Title (ZH)**: 基于强化学习驱动的通用特征表示的跨用户活动识别 

**Authors**: Xiaozhou Ye, Kevin I-Kai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01031)  

**Abstract**: Human Activity Recognition (HAR) using wearable sensors is crucial for healthcare, fitness tracking, and smart environments, yet cross-user variability -- stemming from diverse motion patterns, sensor placements, and physiological traits -- hampers generalization in real-world settings. Conventional supervised learning methods often overfit to user-specific patterns, leading to poor performance on unseen users. Existing domain generalization approaches, while promising, frequently overlook temporal dependencies or depend on impractical domain-specific labels. We propose Temporal-Preserving Reinforcement Learning Domain Generalization (TPRL-DG), a novel framework that redefines feature extraction as a sequential decision-making process driven by reinforcement learning. TPRL-DG leverages a Transformer-based autoregressive generator to produce temporal tokens that capture user-invariant activity dynamics, optimized via a multi-objective reward function balancing class discrimination and cross-user invariance. Key innovations include: (1) an RL-driven approach for domain generalization, (2) autoregressive tokenization to preserve temporal coherence, and (3) a label-free reward design eliminating the need for target user annotations. Evaluations on the DSADS and PAMAP2 datasets show that TPRL-DG surpasses state-of-the-art methods in cross-user generalization, achieving superior accuracy without per-user calibration. By learning robust, user-invariant temporal patterns, TPRL-DG enables scalable HAR systems, facilitating advancements in personalized healthcare, adaptive fitness tracking, and context-aware environments. 

**Abstract (ZH)**: 使用可穿戴传感器进行人体活动识别（HAR）在医疗保健、健身跟踪和智能环境中至关重要，但用户间变异性的存在——源自多样的运动模式、传感器位置和生理特性——在实际应用中限制了泛化能力。传统监督学习方法常常过度拟合用户特定的模式，导致在未见过的用户上表现不佳。现有的一些域泛化方法虽然充满潜力，但也常常忽视时间依赖性或依赖于不可用的特定域标签。我们提出了保留时间的强化学习域泛化（TPRL-DG），这是一个新颖的框架，重新定义了特征提取为由强化学习驱动的顺序决策过程。TPRL-DG 利用基于 Transformer 的自回归生成器生成时间标记，以多目标奖励函数优化平衡类别区分和用户间不变性。关键创新包括：（1）基于强化学习的域泛化方法；（2）自回归分词以保留时间一致性；（3）无标签的奖励设计，消除目标用户注解的需要。在 DSADS 和 PAMAP2 数据集上的评估显示，TPRL-DG 在用户间泛化方面超过了现有最先进的方法，实现了更高的准确率而无需针对每个用户进行校准。通过学习稳健的用户不变时间模式，TPRL-DG 使HAR系统能够大规模应用，促进了个性化医疗保健、自适应健身跟踪和情境感知环境的发展。 

---
# MEPT: Mixture of Expert Prompt Tuning as a Manifold Mapper 

**Title (ZH)**: MEPT: 专家提示调谐的流形映射器 

**Authors**: Runjia Zeng, Guangyan Sun, Qifan Wang, Tong Geng, Sohail Dianat, Xiaotian Han, Raghuveer Rao, Xueling Zhang, Cheng Han, Lifu Huang, Dongfang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00996)  

**Abstract**: Considering deep neural networks as manifold mappers, the pretrain-then-fine-tune paradigm can be interpreted as a two-stage process: pretrain establishes a broad knowledge base, and fine-tune adjusts the model parameters to activate specific neural pathways to align with the target manifold. Although prior fine-tuning approaches demonstrate success, their rigid parameter space limits their ability to dynamically activate appropriate neural pathways, rendering them ill-equipped to adapt flexibly to the diverse and evolving data distributions. In light of this view, we propose a novel approach, Mixture of Expert Prompt Tuning (MEPT), as an effective and efficient manifold-mapping framework. MEPT leverages the Mixture of Experts architecture by integrating multiple prompt experts to adaptively learn diverse and non-stationary data distributions. Empirical evaluations demonstrate that MEPT outperforms several state-of-the-art parameter efficient baselines on SuperGLUE, achieving notable improvements in mean accuracy (e.g., 1.94%) while significantly reducing activated prompts by 79.25%. The effectiveness of MEPT is further supported by theoretical insights from manifold learning and validated through neural activation pathway visualization results. Our code is avaliable at this https URL. 

**Abstract (ZH)**: 将深度神经网络视为流形映射器，预训练-然后微调范式可以解释为两个阶段的过程：预训练建立广泛的知识基础，微调调整模型参数以激活特定的神经路径以与目标流形对齐。尽管先前的微调方法取得成功，但它们僵化的参数空间限制了它们动态激活合适神经路径的能力，使其难以灵活适应多样且不断变化的数据分布。基于此视角，我们提出了一种新颖的方法——专家提示混合微调（MEPT），作为一种有效的流形映射框架。MEPT通过集成多个提示专家利用专家混合架构，以适应性地学习多样且非平稳数据分布。实证评估表明，MEPT在SuperGLUE上优于几种最先进的参数高效基线，平均准确率提高显著（例如，1.94%），同时激活的提示数量显著减少79.25%。MEPT的有效性还得到了流形学习理论洞察的支持，并通过神经激活路径可视化结果得到验证。代码可从该网址获得。 

---
# Online Decentralized Federated Multi-task Learning With Trustworthiness in Cyber-Physical Systems 

**Title (ZH)**: 在线去中心化联邦多任务学习：具有可信度的网络物理系统 

**Authors**: Olusola Odeyomi, Sofiat Olaosebikan, Ajibuwa Opeyemi, Oluwadoyinsola Ige  

**Link**: [PDF](https://arxiv.org/pdf/2509.00992)  

**Abstract**: Multi-task learning is an effective way to address the challenge of model personalization caused by high data heterogeneity in federated learning. However, extending multi-task learning to the online decentralized federated learning setting is yet to be explored. The online decentralized federated learning setting considers many real-world applications of federated learning, such as autonomous systems, where clients communicate peer-to-peer and the data distribution of each client is time-varying. A more serious problem in real-world applications of federated learning is the presence of Byzantine clients. Byzantine-resilient approaches used in federated learning work only when the number of Byzantine clients is less than one-half the total number of clients. Yet, it is difficult to put a limit on the number of Byzantine clients within a system in reality. However, recent work in robotics shows that it is possible to exploit cyber-physical properties of a system to predict clients' behavior and assign a trust probability to received signals. This can help to achieve resiliency in the presence of a dominating number of Byzantine clients. Therefore, in this paper, we develop an online decentralized federated multi-task learning algorithm to provide model personalization and resiliency when the number of Byzantine clients dominates the number of honest clients. Our proposed algorithm leverages cyber-physical properties, such as the received signal strength in wireless systems or side information, to assign a trust probability to local models received from neighbors in each iteration. Our simulation results show that the proposed algorithm performs close to a Byzantine-free setting. 

**Abstract (ZH)**: 基于在线去中心化联邦学习的鲁棒多任务学习算法 

---
# ART: Adaptive Resampling-based Training for Imbalanced Classification 

**Title (ZH)**: 自适应重采样基于的不平衡分类训练方法 

**Authors**: Arjun Basandrai, Shourya Jain, K. Ilanthenral  

**Link**: [PDF](https://arxiv.org/pdf/2509.00955)  

**Abstract**: Traditional resampling methods for handling class imbalance typically uses fixed distributions, undersampling the majority or oversampling the minority. These static strategies ignore changes in class-wise learning difficulty, which can limit the overall performance of the model.
This paper proposes an Adaptive Resampling-based Training (ART) method that periodically updates the distribution of the training data based on the class-wise performance of the model. Specifically, ART uses class-wise macro F1 scores, computed at fixed intervals, to determine the degree of resampling to be performed.
Unlike instance-level difficulty modeling, which is noisy and outlier-sensitive, ART adapts at the class level. This allows the model to incrementally shift its attention towards underperforming classes in a way that better aligns with the optimization objective.
Results on diverse benchmarks, including Pima Indians Diabetes and Yeast dataset demonstrate that ART consistently outperforms both resampling-based and algorithm-level methods, including Synthetic Minority Oversampling Technique (SMOTE), NearMiss Undersampling, and Cost-sensitive Learning on binary as well as multi-class classification tasks with varying degrees of imbalance.
In most settings, these improvements are statistically significant. On tabular datasets, gains are significant under paired t-tests and Wilcoxon tests (p < 0.05), while results on text and image tasks remain favorable. Compared to training on the original imbalanced data, ART improves macro F1 by an average of 2.64 percentage points across all tested tabular datasets. Unlike existing methods, whose performance varies by task, ART consistently delivers the strongest macro F1, making it a reliable choice for imbalanced classification. 

**Abstract (ZH)**: 自适应重采样基于训练方法(ART)：基于类别的绩效动态调整训练数据分布 

---
# SCOUT: Toward Sub-Quadratic Attention via Segment Compression for Optimized Utility in Transformers 

**Title (ZH)**: SCOUT：通过段压缩实现Transformer中近亚二次注意力以优化实用性 

**Authors**: Aref Jafari, Yuhe Fan, Benyamin Jamialahmadi, Parsa Farinneya, Boxing Chen, Marzieh S. Tahaei  

**Link**: [PDF](https://arxiv.org/pdf/2509.00935)  

**Abstract**: Transformers have demonstrated strong performance across a wide range of sequence modeling tasks, but their quadratic attention complexity limits scalability to long sequences. Linear models such as Mamba and sliding-window attention (SWA) address this by mixing tokens through recurrent or localized operations with fixed-size memory, achieving efficient inference. However, these methods risk degrading performance on long sequences due to their inability to retain detailed information from distant tokens. We propose SCOUT (Segment Compression for Optimized Utility in Transformers), a hybrid architecture that compresses tokens locally within fixed-size segments and applies attention only over these compressed representations. Each token embedding is first enriched via a linear local mixer, Mamba or SWA, that integrates recent context. Then, instead of attending to all previous tokens, each token sparsely attends to a small number of compressed checkpoint tokens that summarize the input history. This design retains much of the expressivity of full attention while substantially reducing the computational and memory cost. By attending to compressed history rather than all previous tokens, SCOUT incurs slightly higher memory than purely linear models, but its growth rate remains sub-quadratic and far more scalable than that of full Transformers. We analyze SCOUT's computational and memory efficiency and evaluate it empirically on long-context language modeling and reasoning tasks. SCOUT with both Mamba and SWA mixers outperforms strong long-sequence baselines under the same computational budget, matches full-attention Transformers on language modeling and common-sense reasoning tasks at 400M and 1.3B scales. Moreover, our SCOUT achieves higher end-to-end throughput than SOTA models, while delivering comparable results on long sequence benchmarks. 

**Abstract (ZH)**: SCOUT：基于段压缩的优化转换器结构 

---
# Superposition in Graph Neural Networks 

**Title (ZH)**: 图神经网络中的叠加原理 

**Authors**: Lukas Pertl, Han Xuanyuan, Pietro Liò  

**Link**: [PDF](https://arxiv.org/pdf/2509.00928)  

**Abstract**: Interpreting graph neural networks (GNNs) is difficult because message passing mixes signals and internal channels rarely align with human concepts. We study superposition, the sharing of directions by multiple features, directly in the latent space of GNNs. Using controlled experiments with unambiguous graph concepts, we extract features as (i) class-conditional centroids at the graph level and (ii) linear-probe directions at the node level, and then analyze their geometry with simple basis-invariant diagnostics. Across GCN/GIN/GAT we find: increasing width produces a phase pattern in overlap; topology imprints overlap onto node-level features that pooling partially remixes into task-aligned graph axes; sharper pooling increases axis alignment and reduces channel sharing; and shallow models can settle into metastable low-rank embeddings. These results connect representational geometry with concrete design choices (width, pooling, and final-layer activations) and suggest practical approaches for more interpretable GNNs. 

**Abstract (ZH)**: 解读图神经网络（GNNs）困难的原因在于消息传递会混合信号，且内部通道 rarely 与人类概念对齐。我们直接在 GNN 的潜在空间中研究叠加现象，即多个特征共享的方向。通过使用具有明确图概念的受控实验，我们提取特征为 (i) 图级别上的类别条件质心和 (ii) 节点级别上的线性探测方向，并使用简单的基不变诊断分析它们的几何结构。我们发现：增加宽度会产生重叠的相位模式；拓扑结构将重叠印射到节点级别特征中，池化操作部分重塑这些特征以对齐于任务相关的图轴；更尖锐的池化操作增加轴对齐并减少通道共享；浅层模型可能会收敛到不稳定的低秩嵌入。这些结果将表示空间几何与具体的网络设计选择（宽度、池化方式和最终层激活函数）联系起来，并为更具有解释性的 GNN 提出了实用的方法。 

---
# TinyMusician: On-Device Music Generation with Knowledge Distillation and Mixed Precision Quantization 

**Title (ZH)**: TinyMusician：基于知识蒸馏和混合精度量化设备端音乐生成 

**Authors**: Hainan Wang, Mehdi Hosseinzadeh, Reza Rawassizadeh  

**Link**: [PDF](https://arxiv.org/pdf/2509.00914)  

**Abstract**: The success of the generative model has gained unprecedented attention in the music generation area. Transformer-based architectures have set new benchmarks for model performance. However, their practical adoption is hindered by some critical challenges: the demand for massive computational resources and inference time, due to their large number of parameters. These obstacles make them infeasible to deploy on edge devices, such as smartphones and wearables, with limited computational resources. In this work, we present TinyMusician, a lightweight music generation model distilled from MusicGen (a State-of-the-art music generation model). TinyMusician integrates two innovations: (i) Stage-mixed Bidirectional and Skewed KL-Divergence and (ii) Adaptive Mixed-Precision Quantization. The experimental results demonstrate that TinyMusician retains 93% of the MusicGen-Small performance with 55% less model size. TinyMusician is the first mobile-deployable music generation model that eliminates cloud dependency while maintaining high audio fidelity and efficient resource usage 

**Abstract (ZH)**: 轻量级音乐生成模型TinyMusician：从MusicGen精炼而来的同时保留高性能的创新方案 

---
# Spotlighter: Revisiting Prompt Tuning from a Representative Mining View 

**Title (ZH)**: Spotlighter: 从代表性挖掘视角回顾Prompt调优 

**Authors**: Yutong Gao, Maoyuan Shao, Xinyang Huang, Chuang Zhu, Lijuan Sun, Yu Weng, Xuan Liu, Guoshun Nan  

**Link**: [PDF](https://arxiv.org/pdf/2509.00905)  

**Abstract**: CLIP's success has demonstrated that prompt tuning can achieve robust cross-modal semantic alignment for tasks ranging from open-domain recognition to fine-grained classification. However, redundant or weakly relevant feature components introduce noise and incur unnecessary computational costs. In this work, we propose Spotlighter, a lightweight token-selection framework that simultaneously enhances accuracy and efficiency in prompt tuning. Spotlighter evaluates each visual token's activation from both sample-wise and semantic-wise perspectives and retains only the top-scoring tokens for downstream prediction. A class-specific semantic memory bank of learned prototypes refines this selection, ensuring semantic representativeness and compensating for discarded features. To further prioritize informative signals, we introduce a two-level ranking mechanism that dynamically weights token--prototype interactions. Across 11 few-shot benchmarks, Spotlighter outperforms CLIP by up to 11.19\% in harmonic mean accuracy and achieves up to 0.8K additional FPS, with only 21 extra parameters. These results establish Spotlighter as an effective and scalable baseline for prompt tuning. Code for our method will be available at this https URL. 

**Abstract (ZH)**: CLIP的成功表明，提示调优可以实现从开放领域识别到细粒度分类任务的稳健多模态语义对齐。然而，冗余或弱相关的特征组件引入了噪声并带来了不必要的计算成本。在本文中，我们提出了一种轻量级的令牌选择框架Spotlighter，该框架同时提高了调优的准确性和效率。Spotlighter从样本和语义两个层面评估每个视觉令牌的激活，并仅保留得分最高的令牌用于下游预测。特定类别的语义记忆库通过学习原型细化这一选择，确保语义表征性并弥补丢弃的特征。为更优先选择信息信号，我们引入了一种两层排名机制，动态权重令牌-原型交互。在11个少样本基准中，Spotlighter在调和平均准确率上比CLIP高出了11.19％，并且实现了多达0.8K的额外FPS，仅新增21个参数。这些结果确立了Spotlighter作为提示调优的有效和可扩展基线的地位。关于我们方法的代码将在此页面提供。 

---
# An Explainable Gaussian Process Auto-encoder for Tabular Data 

**Title (ZH)**: 可解释的高斯过程自编码器在表格数据上的应用 

**Authors**: Wei Zhang, Brian Barr, John Paisley  

**Link**: [PDF](https://arxiv.org/pdf/2509.00884)  

**Abstract**: Explainable machine learning has attracted much interest in the community where the stakes are high. Counterfactual explanations methods have become an important tool in explaining a black-box model. The recent advances have leveraged the power of generative models such as an autoencoder. In this paper, we propose a novel method using a Gaussian process to construct the auto-encoder architecture for generating counterfactual samples. The resulting model requires fewer learnable parameters and thus is less prone to overfitting. We also introduce a novel density estimator that allows for searching for in-distribution samples. Furthermore, we introduce an algorithm for selecting the optimal regularization rate on density estimator while searching for counterfactuals. We experiment with our method in several large-scale tabular datasets and compare with other auto-encoder-based methods. The results show that our method is capable of generating diversified and in-distribution counterfactual samples. 

**Abstract (ZH)**: 可解释的机器学习在关键利益涉及的社区中引起了广泛关注。因果解释方法已成为解释黑盒模型的重要工具。最近的进展利用了生成模型如自动编码器的力量。在本文中，我们提出了一种使用高斯过程构建自动编码器架构以生成因果样本的新方法。该模型所需的可学习参数较少，因此更容易避免过拟合。我们还引入了一种新颖的概率密度估计器，允许搜索同分布样本。此外，我们引入了一种算法，在搜索因果样本时选择概率密度估计器的最佳正则化率。我们在几个大规模表型数据集中实验了该方法，并与其他基于自动编码器的方法进行了比较。结果显示，我们的方法能够生成多样化的同分布因果样本。 

---
# Accelerating Latency-Critical Applications with AI-Powered Semi-Automatic Fine-Grained Parallelization on SMT Processors 

**Title (ZH)**: 基于AI赋能的半自动细粒度并行化在SMT处理器上加速 latency-critical 应用程序 

**Authors**: Denis Los, Igor Petushkov  

**Link**: [PDF](https://arxiv.org/pdf/2509.00883)  

**Abstract**: Latency-critical applications tend to show low utilization of functional units due to frequent cache misses and mispredictions during speculative execution in high-performance superscalar processors. However, due to significant impact on single-thread performance, Simultaneous Multithreading (SMT) technology is rarely used with heavy threads of latency-critical applications. In this paper, we explore utilization of SMT technology to support fine-grained parallelization of latency-critical applications. Following the advancements in the development of Large Language Models (LLMs), we introduce Aira, an AI-powered Parallelization Adviser. To implement Aira, we extend AI Coding Agent in Cursor IDE with additional tools connected through Model Context Protocol, enabling end-to-end AI Agent for parallelization. Additional connected tools enable LLM-guided hotspot detection, collection of dynamic dependencies with Dynamic Binary Instrumentation, SMT-aware performance simulation to estimate performance gains. We apply Aira with Relic parallel framework for fine-grained task parallelism on SMT cores to parallelize latency-critical benchmarks representing real-world applications used in industry. We show 17% geomean performance gain from parallelization of latency-critical benchmarks using Aira with Relic framework. 

**Abstract (ZH)**: 基于延迟关键应用细粒度并行化的Simultaneous Multithreading技术利用研究与AI辅助并行顾问Aira 

---
# Pose as Clinical Prior: Learning Dual Representations for Scoliosis Screening 

**Title (ZH)**: 临床先验作为姿态：学习脊柱侧弯筛查的双重表示 

**Authors**: Zirui Zhou, Zizhao Peng, Dongyang Jin, Chao Fan, Fengwei An, Shiqi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00872)  

**Abstract**: Recent AI-based scoliosis screening methods primarily rely on large-scale silhouette datasets, often neglecting clinically relevant postural asymmetries-key indicators in traditional screening. In contrast, pose data provide an intuitive skeletal representation, enhancing clinical interpretability across various medical applications. However, pose-based scoliosis screening remains underexplored due to two main challenges: (1) the scarcity of large-scale, annotated pose datasets; and (2) the discrete and noise-sensitive nature of raw pose coordinates, which hinders the modeling of subtle asymmetries. To address these limitations, we introduce Scoliosis1K-Pose, a 2D human pose annotation set that extends the original Scoliosis1K dataset, comprising 447,900 frames of 2D keypoints from 1,050 adolescents. Building on this dataset, we introduce the Dual Representation Framework (DRF), which integrates a continuous skeleton map to preserve spatial structure with a discrete Postural Asymmetry Vector (PAV) that encodes clinically relevant asymmetry descriptors. A novel PAV-Guided Attention (PGA) module further uses the PAV as clinical prior to direct feature extraction from the skeleton map, focusing on clinically meaningful asymmetries. Extensive experiments demonstrate that DRF achieves state-of-the-art performance. Visualizations further confirm that the model leverages clinical asymmetry cues to guide feature extraction and promote synergy between its dual representations. The dataset and code are publicly available at this https URL. 

**Abstract (ZH)**: 近期基于AI的脊柱侧弯筛查方法主要依赖大规模轮廓数据集，往往忽视了传统筛查中临床相关的姿势不对称性——关键指标。相比之下，姿势数据提供了直观的骨骼表示，提升了在各种医疗应用中的临床可解释性。然而，由于两个主要挑战，基于姿势的脊柱侧弯筛查仍被广泛忽视：（1）大规模标注姿势数据集稀缺；（2）原始姿势坐标离散且敏感于噪声，阻碍了细微不对称性的建模。为解决这些局限，我们引入了Scoliosis1K-Pose，这是一个扩展了原始Scoliosis1K数据集的2D人体姿态标注集，包含1050名青少年的447,900帧2D关键点。在此数据集基础上，我们提出了双表示框架（DRF），该框架整合了一种连续的骨骼图来保持空间结构，并结合编码临床相关不对称性描述符的离散姿势不对称向量（PAV）。新颖的PAV导向注意力（PGA）模块进一步利用PAV作为临床先验来直接从骨骼图中提取特征，重点关注临床有意义的不对称性。广泛实验证明，DRF达到了最先进的性能。可视化进一步证实，该模型利用临床不对称性线索来引导特征提取并促进其双表示之间的协同作用。该数据集和代码在此网址公开：this https URL。 

---
# Can General-Purpose Omnimodels Compete with Specialists? A Case Study in Medical Image Segmentation 

**Title (ZH)**: 通用型 OmniModel 能够与专业模型竞争吗？以医学图像分割为例 

**Authors**: Yizhe Zhang, Qiang Chen, Tao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.00866)  

**Abstract**: The emergence of powerful, general-purpose omnimodels capable of processing diverse data modalities has raised a critical question: can these ``jack-of-all-trades'' systems perform on par with highly specialized models in knowledge-intensive domains? This work investigates this question within the high-stakes field of medical image segmentation. We conduct a comparative study analyzing the zero-shot performance of a state-of-the-art omnimodel (Gemini 2.5 Pro, the ``Nano Banana'' model) against domain-specific deep learning models on three distinct tasks: polyp (endoscopy), retinal vessel (fundus), and breast tumor segmentation (ultrasound). Our study focuses on performance at the extremes by curating subsets of the ``easiest'' and ``hardest'' cases based on the specialist models' accuracy. Our findings reveal a nuanced and task-dependent landscape. For polyp and breast tumor segmentation, specialist models excel on easy samples, but the omnimodel demonstrates greater robustness on hard samples where specialists fail catastrophically. Conversely, for the fine-grained task of retinal vessel segmentation, the specialist model maintains superior performance across both easy and hard cases. Intriguingly, qualitative analysis suggests omnimodels may possess higher sensitivity, identifying subtle anatomical features missed by human annotators. Our results indicate that while current omnimodels are not yet a universal replacement for specialists, their unique strengths suggest a potential complementary role with specialist models, particularly in enhancing robustness on challenging edge cases. 

**Abstract (ZH)**: 强大的通用型 omnimodels 能够处理多种数据模态的出现，引发了关键问题：这些“万能系统”是否能在知识密集型领域与高度专业化的模型相媲美？本研究在高 stakes 的医学图像分割领域探讨了这一问题。我们进行了一项比较研究，分析了最先进的 omnimodel（Gemini 2.5 Pro，即“Nano Banana”模型）与领域专用的深度学习模型在三个不同任务上的零样本性能：结节（内镜）、视网膜血管（眼底）和乳腺肿瘤分割（超声）。我们的研究重点在于性能的极端差异，通过根据专家模型的准确性挑选最容易和最困难的案例子集。我们的研究发现揭示了一个复杂且任务依赖的场景。对于结节和乳腺肿瘤分割，专家模型在容易样本上表现出色，但 omnimodel 在困难样本上的鲁棒性更强，而专家模型在这些样本上表现灾难性失败。相反，在视网膜血管分割这一细致的任务中，专家模型在容易和困难样本上均保持更好的表现。有趣的是，定性分析表明，omnimodels 可能具有更高的敏感性，能识别出人类标注者忽略的细微解剖特征。我们的结果表明，尽管当前的 omnimodels 还不是专家模型的通用替代品，但它们的独特优势暗示了与专家模型互补的可能性，特别是在增强在挑战性的边界案例中的鲁棒性方面。 

---
# Speech Command Recognition Using LogNNet Reservoir Computing for Embedded Systems 

**Title (ZH)**: 使用LogNNet蓄水池计算的嵌入式系统语音命令识别 

**Authors**: Yuriy Izotov, Andrei Velichko  

**Link**: [PDF](https://arxiv.org/pdf/2509.00862)  

**Abstract**: This paper presents a low-resource speech-command recognizer combining energy-based voice activity detection (VAD), an optimized Mel-Frequency Cepstral Coefficients (MFCC) pipeline, and the LogNNet reservoir-computing classifier. Using four commands from the Speech Commands da-taset downsampled to 8 kHz, we evaluate four MFCC aggregation schemes and find that adaptive binning (64-dimensional feature vector) offers the best accuracy-to-compactness trade-off. The LogNNet classifier with architecture 64:33:9:4 reaches 92.04% accuracy under speaker-independent evaluation, while requiring significantly fewer parameters than conventional deep learn-ing models. Hardware implementation on Arduino Nano 33 IoT (ARM Cor-tex-M0+, 48 MHz, 32 KB RAM) validates the practical feasibility, achieving ~90% real-time recognition accuracy while consuming only 18 KB RAM (55% utilization). The complete pipeline (VAD -> MFCC -> LogNNet) thus enables reliable on-device speech-command recognition under strict memory and compute limits, making it suitable for battery-powered IoT nodes, wire-less sensor networks, and hands-free control interfaces. 

**Abstract (ZH)**: 一种结合能量基语音活动检测、优化的Mel频率倒谱系数管道及LogNNet水库计算分类器的低资源语音命令识别方法 

---
# Why it is worth making an effort with GenAI 

**Title (ZH)**: 为什么值得投入精力在GenAI上 

**Authors**: Yvonne Rogers  

**Link**: [PDF](https://arxiv.org/pdf/2509.00852)  

**Abstract**: Students routinely use ChatGPT and the like now to help them with their homework, such as writing an essay. It takes less effort to complete and is easier to do than by hand. It can even produce as good if not better output than the student's own work. However, there is a growing concern that over-reliance on using GenAI in this way will stifle the development of learning writing and critical thinking skills. How might this trend be reversed? What if students were required to make more effort when using GenAI to do their homework? It might be more challenging, but the additional effort involved could result in them learning more and having a greater sense of achievement. This tension can be viewed as a form of effort paradox; where effort is both viewed as something to be avoided but at the same time is valued. Is it possible to let students learn sometimes with less and other times more effort? Students are already adept at the former but what about the latter? Could we design new kinds of AI tools that deliberately require more effort to use to deepen the learning experience? In this paper, I begin to outline what form these might take, for example, asking students to use a combination of GenAI tools with traditional learning approaches (e.g. note-taking while reading). I also discuss how else to design tools to think with that augments human cognition; where students learn more the skills of metacognition and reflection. 

**Abstract (ZH)**: 学生现在经常使用ChatGPT等工具来完成作业，如撰写论文。这种方式比手动完成更加省力且易于操作。甚至可以产出不亚于学生自己作品的结果。然而，过度依赖此类GenAI工具可能导致学生在学习写作和批判性思维技能方面的发展受到抑制。这一趋势如何扭转？如果要求学生在使用GenAI完成作业时付出更多努力，可能会更具挑战性，但额外的努力可能会促进他们学习更多知识并产生更大的成就感。这种努力的矛盾可以视为一种努力悖论：努力一方面被认为是应避免的事物，另一方面又被视为有价值的事物。我们能否让学生有时少付出努力，有时多付出努力来进行学习？学生已经擅长前者，那么后者呢？我们能否设计出要求更多努力使用的新型AI工具来加深学习体验？在这篇文章中，我开始探讨这些工具的形式，例如，让学生结合使用GenAI工具和传统学习方法（如阅读时做笔记）。我也讨论了如何设计增强人类认知能力的工具，使学生在元认知和反思方面学到更多技能。 

---
# Causal SHAP: Feature Attribution with Dependency Awareness through Causal Discovery 

**Title (ZH)**: 因果SHAP：通过因果发现实现特征归因中的依赖意识 

**Authors**: Woon Yee Ng, Li Rong Wang, Siyuan Liu, Xiuyi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2509.00846)  

**Abstract**: Explaining machine learning (ML) predictions has become crucial as ML models are increasingly deployed in high-stakes domains such as healthcare. While SHapley Additive exPlanations (SHAP) is widely used for model interpretability, it fails to differentiate between causality and correlation, often misattributing feature importance when features are highly correlated. We propose Causal SHAP, a novel framework that integrates causal relationships into feature attribution while preserving many desirable properties of SHAP. By combining the Peter-Clark (PC) algorithm for causal discovery and the Intervention Calculus when the DAG is Absent (IDA) algorithm for causal strength quantification, our approach addresses the weakness of SHAP. Specifically, Causal SHAP reduces attribution scores for features that are merely correlated with the target, as validated through experiments on both synthetic and real-world datasets. This study contributes to the field of Explainable AI (XAI) by providing a practical framework for causal-aware model explanations. Our approach is particularly valuable in domains such as healthcare, where understanding true causal relationships is critical for informed decision-making. 

**Abstract (ZH)**: 将机器学习（ML）预测解释为关键：在 healthcare 等高风险领域部署 ML 模型日益增多的情况下，ML 预测的解释变得至关重要。虽然 SHapley Additive exPlanations (SHAP) 广泛用于模型解释，但它无法区分因果关系与相关性，常在特征高度相关时错误地分配特征重要性。我们提出因果 SHAP，这是一个新颖的框架，将因果关系整合到特征解释中，同时保留 SHAP 的许多 desirable 属性。通过结合用于因果发现的 Peter-Clark (PC) 算法和在有向无环图（DAG）缺失时用于因果强度量化的方法 Intervention Calculus when the DAG is Absent (IDA) 算法，我们的方法解决了 SHAP 的弱点。具体而言，因果 SHAP 通过实验在合成和真实数据集上验证了降低仅与目标相关特征的解释分数。本研究通过提供一种因果意识模型解释的实用框架，为可解释人工智能（XAI）领域做出了贡献。我们的方法特别在如 healthcare 这样需要了解真正因果关系的领域中具有重要价值。 

---
# AImoclips: A Benchmark for Evaluating Emotion Conveyance in Text-to-Music Generation 

**Title (ZH)**: AImoclips：评估文本到音乐生成中情绪传达的基准 

**Authors**: Gyehun Go, Satbyul Han, Ahyeon Choi, Eunjin Choi, Juhan Nam, Jeong Mi Park  

**Link**: [PDF](https://arxiv.org/pdf/2509.00813)  

**Abstract**: Recent advances in text-to-music (TTM) generation have enabled controllable and expressive music creation using natural language prompts. However, the emotional fidelity of TTM systems remains largely underexplored compared to human preference or text alignment. In this study, we introduce AImoclips, a benchmark for evaluating how well TTM systems convey intended emotions to human listeners, covering both open-source and commercial models. We selected 12 emotion intents spanning four quadrants of the valence-arousal space, and used six state-of-the-art TTM systems to generate over 1,000 music clips. A total of 111 participants rated the perceived valence and arousal of each clip on a 9-point Likert scale. Our results show that commercial systems tend to produce music perceived as more pleasant than intended, while open-source systems tend to perform the opposite. Emotions are more accurately conveyed under high-arousal conditions across all models. Additionally, all systems exhibit a bias toward emotional neutrality, highlighting a key limitation in affective controllability. This benchmark offers valuable insights into model-specific emotion rendering characteristics and supports future development of emotionally aligned TTM systems. 

**Abstract (ZH)**: Recent Advances in Evaluating Emotional Fidelity in Text-to-Music Generation Systems Using AImoclips 

---
# ProCause: Generating Counterfactual Outcomes to Evaluate Prescriptive Process Monitoring Methods 

**Title (ZH)**: ProCause: 生成反事实结果以评估指导性过程监控方法 

**Authors**: Jakob De Moor, Hans Weytjens, Johannes De Smedt  

**Link**: [PDF](https://arxiv.org/pdf/2509.00797)  

**Abstract**: Prescriptive Process Monitoring (PresPM) is the subfield of Process Mining that focuses on optimizing processes through real-time interventions based on event log data. Evaluating PresPM methods is challenging due to the lack of ground-truth outcomes for all intervention actions in datasets. A generative deep learning approach from the field of Causal Inference (CI), RealCause, has been commonly used to estimate the outcomes for proposed intervention actions to evaluate a new policy. However, RealCause overlooks the temporal dependencies in process data, and relies on a single CI model architecture, TARNet, limiting its effectiveness. To address both shortcomings, we introduce ProCause, a generative approach that supports both sequential (e.g., LSTMs) and non-sequential models while integrating multiple CI architectures (S-Learner, T-Learner, TARNet, and an ensemble). Our research using a simulator with known ground truths reveals that TARNet is not always the best choice; instead, an ensemble of models offers more consistent reliability, and leveraging LSTMs shows potential for improved evaluations when temporal dependencies are present. We further validate ProCause's practical effectiveness through a real-world data analysis, ensuring a more reliable evaluation of PresPM methods. 

**Abstract (ZH)**: 基于因果推理的生成方法在过程建议监控中的应用：ProCause 

---
# LegalChainReasoner: A Legal Chain-guided Framework for Criminal Judicial Opinion Generation 

**Title (ZH)**: LegalChainReasoner：一种基于法律链条的刑事司法意见生成框架 

**Authors**: Weizhe Shi, Qiqi Wang, Yihong Pan, Qian Liu, Kaiqi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.00783)  

**Abstract**: A criminal judicial opinion represents the judge's disposition of a case, including the decision rationale and sentencing. Automatically generating such opinions can assist in analyzing sentencing consistency and provide judges with references to similar past cases. However, current research typically approaches this task by dividing it into two isolated subtasks: legal reasoning and sentencing prediction. This separation often leads to inconsistency between the reasoning and predictions, failing to meet real-world judicial requirements. Furthermore, prior studies rely on manually curated knowledge to enhance applicability, yet such methods remain limited in practical deployment. To address these limitations and better align with legal practice, we propose a new LegalAI task: Judicial Opinion Generation, which simultaneously produces both legal reasoning and sentencing decisions. To achieve this, we introduce LegalChainReasoner, a framework that applies structured legal chains to guide the model through comprehensive case assessments. By integrating factual premises, composite legal conditions, and sentencing conclusions, our approach ensures flexible knowledge injection and end-to-end opinion generation. Experiments on two real-world and open-source Chinese legal case datasets demonstrate that our method outperforms baseline models. 

**Abstract (ZH)**: 一种刑事司法意见代表法官对案件的处理，包括判决理由和量刑。自动生成此类意见可以协助分析量刑一致性，并为法官提供类似案例的参考。然而，当前研究通常将此任务分为两个孤立的子任务：法律推理和量刑预测。这种分离往往导致推理与预测之间的一致性问题，无法满足实际司法要求。此外，先前的研究依赖于人工策画的知识来提高适用性，但这些方法在实际部署中仍存在局限性。为解决这些问题并更好地与法律实践接轨，我们提出一个新的LegalAI任务：司法意见生成，该任务同时生成法律推理和量刑决定。为实现这一目标，我们引入了LegalChainReasoner框架，该框架应用结构化的法律链条来引导模型进行全面的案例评估。通过整合事实前提、复合法律条件和量刑结论，我们的方法确保了灵活的知识注入和端到端的意见生成。在两个实际和开源的中文法律案例数据集上的实验证明了我们方法优于基线模型。 

---
# Low Power Approximate Multiplier Architecture for Deep Neural Networks 

**Title (ZH)**: 低功耗近似乘法器架构用于深度神经网络 

**Authors**: Pragun Jaswal, L. Hemanth Krishna, B. Srinivasu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00764)  

**Abstract**: This paper proposes an low power approximate multiplier architecture for deep neural network (DNN) applications. A 4:2 compressor, introducing only a single combination error, is designed and integrated into an 8x8 unsigned multiplier. This integration significantly reduces the usage of exact compressors while preserving low error rates. The proposed multiplier is employed within a custom convolution layer and evaluated on neural network tasks, including image recognition and denoising. Hardware evaluation demonstrates that the proposed design achieves up to 30.24% energy savings compared to the best among existing multipliers. In image denoising, the custom approximate convolution layer achieves improved Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM) compared to other approximate designs. Additionally, when applied to handwritten digit recognition, the model maintains high classification accuracy. These results demonstrate that the proposed architecture offers a favorable balance between energy efficiency and computational precision, making it suitable for low-power AI hardware implementations. 

**Abstract (ZH)**: 低功率近似乘法器架构及其在深度神经网络应用中的实现与评估 

---
# Enhancing Fairness in Skin Lesion Classification for Medical Diagnosis Using Prune Learning 

**Title (ZH)**: 使用剪枝学习提高皮肤病变分类中的公平性 

**Authors**: Kuniko Paxton, Koorosh Aslansefat, Dhavalkumar Thakker, Yiannis Papadopoulos, Tanaya Maslekar  

**Link**: [PDF](https://arxiv.org/pdf/2509.00745)  

**Abstract**: Recent advances in deep learning have significantly improved the accuracy of skin lesion classification models, supporting medical diagnoses and promoting equitable healthcare. However, concerns remain about potential biases related to skin color, which can impact diagnostic outcomes. Ensuring fairness is challenging due to difficulties in classifying skin tones, high computational demands, and the complexity of objectively verifying fairness. To address these challenges, we propose a fairness algorithm for skin lesion classification that overcomes the challenges associated with achieving diagnostic fairness across varying skin tones. By calculating the skewness of the feature map in the convolution layer of the VGG (Visual Geometry Group) network and the patches and the heads of the Vision Transformer, our method reduces unnecessary channels related to skin tone, focusing instead on the lesion area. This approach lowers computational costs and mitigates bias without relying on conventional statistical methods. It potentially reduces model size while maintaining fairness, making it more practical for real-world applications. 

**Abstract (ZH)**: 近期深度学习的进展显著提升了皮肤病变分类模型的准确性，支持医疗诊断并促进公平医疗。然而，仍然存在与肤色相关的潜在偏见问题，这可能影响诊断结果。确保公平性具有挑战性，因为肤色分类困难、计算需求高且客观验证公平性的复杂性。为应对这些挑战，我们提出了一种针对皮肤病变分类的公平性算法，能够克服不同肤色下实现诊断公平性的挑战。通过计算VGG（视觉几何组）网络卷积层、视网膜变换器的块和头部特征图的偏斜度，我们的方法减少了与肤色相关的不必要的通道，而是专注于病灶区域。这种方法降低了计算成本并减轻了偏见，无需依赖传统统计方法。它有可能减少模型大小同时保持公平性，使其更具实用性，适用于实际应用。 

---
# Quantum Causality: Resolving Simpson's Paradox with $\mathcal{DO}$-Calculus 

**Title (ZH)**: 量子因果性：$\mathcal{DO}$-演算解决辛普森悖论 

**Authors**: Pilsung Kang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00744)  

**Abstract**: Distinguishing correlation from causation is a fundamental challenge in machine intelligence, often representing a critical barrier to building robust and trustworthy systems. While Pearl's $\mathcal{DO}$-calculus provides a rigorous framework for causal inference, a parallel challenge lies in its physical implementation. Here, we apply and experimentally validate a quantum algorithmic framework for performing causal interventions. Our approach maps causal networks onto quantum circuits where probabilistic links are encoded by controlled-rotation gates, and interventions are realized by a structural remodeling of the circuit -- a physical analogue to Pearl's ``graph surgery''. We demonstrate the method's efficacy by resolving Simpson's Paradox in a 3-qubit model, and show its scalability by quantifying confounding bias in a 10-qubit healthcare simulation. Critically, we provide a proof-of-principle experimental validation on an IonQ Aria quantum computer, successfully reproducing the paradox and its resolution in the presence of real-world noise. This work establishes a practical pathway for quantum causal inference, offering a new computational tool to address deep-rooted challenges in algorithmic fairness and explainable AI (XAI). 

**Abstract (ZH)**: 从相关性区分因果关系是机器智能中的一个基础挑战，通常代表了建立稳健且可信赖系统的关键障碍。虽然佩尔的$\mathcal{DO}$-演算为因果推理提供了严格的框架，但在其实现方面也存在一个并行的挑战。在此，我们应用并实验验证了量子算法框架来进行因果干预。我们的方法将因果网络映射到量子电路中，其中概率链接由控制旋转门编码，干预通过电路的结构重塑来实现，这与佩尔的“图手术”具有物理上的对应关系。我们通过在3-qubit模型中解决辛普森悖论展示了该方法的有效性，并通过在10-qubit医疗保健模拟中量化混杂偏倚展示了其可扩展性。关键的是，我们在IonQ Aria量子计算机上提供了一个原理性实验验证，成功地在实际噪声存在的情况下再现了该悖论及其解决过程。这项工作为量子因果推理奠定了实用途径，提供了一种新的计算工具来解决算法公平性和可解释AI（XAI）中的深层挑战。 

---
# Task-Aware Adaptive Modulation: A Replay-Free and Resource-Efficient Approach For Continual Graph Learning 

**Title (ZH)**: 任务感知自适应调制：一种无重放且资源高效的连续图学习方法 

**Authors**: Jingtao Liu, Xinming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00735)  

**Abstract**: Continual Graph Learning(CGL)focuses on acquiring new knowledge while retaining previously learned information, essential for real-world graph applications. Current methods grapple with two main issues:1) The Stability-Plasticity Dilemma: Replay-based methods often create an imbalance between the Dilemma, while incurring significant storage costs.2) The Resource-Heavy Pre-training: Leading replay-free methods critically depend on extensively pre-trained backbones, this reliance imposes a substantial resource this http URL this paper, we argue that the key to overcoming these challenges lies not in replaying data or fine-tuning the entire network, but in dynamically modulating the internal computational flow of a frozen backbone. We posit that lightweight, task-specific modules can effectively steer a GNN's reasoning process. Motivated by this insight, we propose Task-Aware Adaptive Modulation(TAAM), a replay-free, resource-efficient approach that charts a new path for navigating the stability-plasticity dilemma. TAAM's core is its Neural Synapse Modulators(NSM), which are trained and then frozen for each task to store expert knowledge. A pivotal prototype-guided strategy governs these modulators: 1) For training, it initializes a new NSM by deep-copying from a similar past modulator to boost knowledge transfer. 2) For inference, it selects the most relevant frozen NSM for each task. These NSMs insert into a frozen GNN backbone to perform fine-grained, node-attentive modulation of its internal flow-different from the static perturbations of prior methods. Extensive experiments show that TAAM comprehensively outperforms state-of-the-art methods across six GCIL benchmark datasets. The code will be released upon acceptance of the paper. 

**Abstract (ZH)**: 持续图学习（CGL）聚焦于在保留先前学习信息的同时获取新知识，这对于现实世界的图应用至关重要。当前方法面临两大主要问题：1）稳定性和可塑性困境：基于回放的方法往往在困境之间造成不平衡，同时产生显著的存储成本。2）资源密集型预训练：领先的无回放方法严重依赖于预先训练的骨干网络，这种依赖对资源提出了重大要求。本文认为克服这些挑战的关键不在于回放数据或微调整个网络，而在于动态调节冻结骨干网络的内部计算流程。我们认为轻量级的任务特定模块可以有效地引导GNN的推理过程。基于这一洞见，我们提出了一种无回放、资源高效的任务感知自适应调节（TAAM）方法，为导航稳定性和可塑性困境开辟了一条新路径。TAAM的核心在于其神经突触调节器(NSM)，它在每个任务中训练并冻结以存储专家知识。一个关键的原型引导策略管理这些调节器：1）在训练过程中，通过深复制一个类似的过去调节器来初始化新的NSM，以增强知识转移；2）在推理过程中，为每个任务选择最相关的冻结NSM。这些NSM插入到冻结的GNN骨干网络中，执行精细节点注意力的调节，不同于先前方法的静态扰动。广泛的经验表明，TAAM在六个GCIL基准数据集中全面优于现有方法。论文被接受后代码将公开。 

---
# Exam Readiness Index (ERI): A Theoretical Framework for a Composite, Explainable Index 

**Title (ZH)**: 备考准备指数（ERI）：一种综合可解释指数的理论框架 

**Authors**: Ananda Prakash Verma  

**Link**: [PDF](https://arxiv.org/pdf/2509.00718)  

**Abstract**: We present a theoretical framework for an Exam Readiness Index (ERI): a composite, blueprint-aware score R in [0,100] that summarizes a learner's readiness for a high-stakes exam while remaining interpretable and actionable. The ERI aggregates six signals -- Mastery (M), Coverage (C), Retention (R), Pace (P), Volatility (V), and Endurance (E) -- each derived from a stream of practice and mock-test interactions. We formalize axioms for component maps and the composite, prove monotonicity, Lipschitz stability, and bounded drift under blueprint re-weighting, and show existence and uniqueness of the optimal linear composite under convex design constraints. We further characterize confidence bands via blueprint-weighted concentration and prove compatibility with prerequisite-admissible curricula (knowledge spaces / learning spaces). The paper focuses on theory; empirical study is left to future work. 

**Abstract (ZH)**: 我们提出了一种考试准备指数(ERI)的理论框架：一个综合、蓝图意识下的评分R在[0,100]之间，用于总结学习者对高风险考试的准备情况，同时保持可解释性和可操作性。该指数聚合了六种信号——掌握程度(Mastery, M)、覆盖范围(Coverage, C)、保留能力(Retention, R)、进度(Pace, P)、波动性(Volatility, V)和耐力(Endurance, E)，每种信号均源自练习和模拟测试的互动流。我们形式化了组成部分映射和合成的原则，证明了单调性、Lipschitz稳定性，并且在蓝图重新加权下展示了有界漂移，进一步证明了在凸设计约束下最优线性合成的存在性和唯一性。我们还通过蓝图加权集中性来表征置信区间，并证明其与先修课程兼容的知识空间/学习空间相兼容。本文侧重于理论研究，实证研究留待未来工作。 

---
# Why Pool When You Can Flow? Active Learning with GFlowNets 

**Title (ZH)**: 为什么使用池化	when你可以使用流式	GFlowNets进行主动学习 

**Authors**: Renfei Zhang, Mohit Pandey, Artem Cherkasov, Martin Ester  

**Link**: [PDF](https://arxiv.org/pdf/2509.00704)  

**Abstract**: The scalability of pool-based active learning is limited by the computational cost of evaluating large unlabeled datasets, a challenge that is particularly acute in virtual screening for drug discovery. While active learning strategies such as Bayesian Active Learning by Disagreement (BALD) prioritize informative samples, it remains computationally intensive when scaled to libraries containing billions samples. In this work, we introduce BALD-GFlowNet, a generative active learning framework that circumvents this issue. Our method leverages Generative Flow Networks (GFlowNets) to directly sample objects in proportion to the BALD reward. By replacing traditional pool-based acquisition with generative sampling, BALD-GFlowNet achieves scalability that is independent of the size of the unlabeled pool. In our virtual screening experiment, we show that BALD-GFlowNet achieves a performance comparable to that of standard BALD baseline while generating more structurally diverse molecules, offering a promising direction for efficient and scalable molecular discovery. 

**Abstract (ZH)**: 基于池的主动学习的可扩展性受限于评估大型未标记数据集的计算成本，这一挑战在药物发现中的虚拟筛选中尤为尖锐。尽管如贝叶斯通过分歧的主动学习（BALD）等主动学习策略倾向于优先选择信息性样本，但在扩展到包含数十亿样本的库时仍计算密集。在本文中，我们介绍了一种避开这一问题的生成式主动学习框架BALD-GFlowNet。我们的方法利用生成流网络（GFlowNets）直接按BALD奖励的比例采样对象。通过用生成采样替换传统的基于池的获取，BALD-GFlowNet实现了与未标记数据池大小无关的可扩展性。在我们的虚拟筛选实验中，我们证明了BALD-GFlowNet在结构多样性方面优于标准的BALD基准，且性能相当，这为高效的可扩展分子发现提供了有前景的方向。 

---
# Unsupervised Dataset Cleaning Framework for Encrypted Traffic Classification 

**Title (ZH)**: 加密流量分类的无监督数据集清洗框架 

**Authors**: Kun Qiu, Ying Wang, Baoqian Li, Wenjun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00701)  

**Abstract**: Traffic classification, a technique for assigning network flows to predefined categories, has been widely deployed in enterprise and carrier networks. With the massive adoption of mobile devices, encryption is increasingly used in mobile applications to address privacy concerns. Consequently, traditional methods such as Deep Packet Inspection (DPI) fail to distinguish encrypted traffic. To tackle this challenge, Artificial Intelligence (AI), in particular Machine Learning (ML), has emerged as a promising solution for encrypted traffic classification. A crucial prerequisite for any ML-based approach is traffic data cleaning, which removes flows that are not useful for training (e.g., irrelevant protocols, background activity, control-plane messages, and long-lived sessions). Existing cleaning solutions depend on manual inspection of every captured packet, making the process both costly and time-consuming. In this poster, we present an unsupervised framework that automatically cleans encrypted mobile traffic. Evaluation on real-world datasets shows that our framework incurs only a 2%~2.5% reduction in classification accuracy compared with manual cleaning. These results demonstrate that our method offers an efficient and effective preprocessing step for ML-based encrypted traffic classification. 

**Abstract (ZH)**: 一种自动清理加密移动流量的无监督框架：有效支持基于机器学习的加密流量分类预处理 

---
# Queuing for Civility: Regulating Emotions and Reducing Toxicity in Digital Discourse 

**Title (ZH)**: 文明排队：调节情绪并减少数字话语中的毒性 

**Authors**: Akriti Verma, Shama Islam, Valeh Moghaddam, Adnan Anwar  

**Link**: [PDF](https://arxiv.org/pdf/2509.00696)  

**Abstract**: The pervasiveness of online toxicity, including hate speech and trolling, disrupts digital interactions and online well-being. Previous research has mainly focused on post-hoc moderation, overlooking the real-time emotional dynamics of online conversations and the impact of users' emotions on others. This paper presents a graph-based framework to identify the need for emotion regulation within online conversations. This framework promotes self-reflection to manage emotional responses and encourage responsible behaviour in real time. Additionally, a comment queuing mechanism is proposed to address intentional trolls who exploit emotions to inflame conversations. This mechanism introduces a delay in publishing comments, giving users time to self-regulate before further engaging in the conversation and helping maintain emotional balance. Analysis of social media data from Twitter and Reddit demonstrates that the graph-based framework reduced toxicity by 12%, while the comment queuing mechanism decreased the spread of anger by 15%, with only 4% of comments being temporarily held on average. These findings indicate that combining real-time emotion regulation with delayed moderation can significantly improve well-being in online environments. 

**Abstract (ZH)**: 在线毒性，包括仇恨言论和 trolling，渗透数字互动并破坏在线福祉。以往的研究主要关注事后 Moderation，忽视了在线对话中的实时情感动态及其对用户情感对他人影响的评估。本文提出了一种图基框架以识别在线对话中情绪调节的需要。该框架促进自我反思以管理情感反应，并在实时鼓励负责任的行为。此外，还提出了一种评论排队机制以应对利用情感升级对话的故意 troll。该机制在发布评论前引入延迟，给用户时间自我调节，从而进一步参与对话并有助于维持情感平衡。来自 Twitter 和 Reddit 的社交媒体数据分析表明，图基框架将毒性降低了 12%，而评论排队机制将愤怒的传播减少了 15%，平均每次有 4% 的评论被暂时搁置。这些发现表明，结合实时情绪调节与延迟 Moderation 可以显著改善在线环境中的福祉。 

---
# DELTA: Variational Disentangled Learning for Privacy-Preserving Data Reprogramming 

**Title (ZH)**: DELTA：变分解耦学习在隐私保护数据再编程中的应用 

**Authors**: Arun Vignesh Malarkkan, Haoyue Bai, Anjali Kaushik, Yanjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00693)  

**Abstract**: In real-world applications, domain data often contains identifiable or sensitive attributes, is subject to strict regulations (e.g., HIPAA, GDPR), and requires explicit data feature engineering for interpretability and transparency. Existing feature engineering primarily focuses on advancing downstream task performance, often risking privacy leakage. We generalize this learning task under such new requirements as Privacy-Preserving Data Reprogramming (PPDR): given a dataset, transforming features to maximize target attribute prediction accuracy while minimizing sensitive attribute prediction accuracy. PPDR poses challenges for existing systems: 1) generating high-utility feature transformations without being overwhelmed by a large search space, and 2) disentangling and eliminating sensitive information from utility-oriented features to reduce privacy inferability. To tackle these challenges, we propose DELTA, a two-phase variational disentangled generative learning framework. Phase I uses policy-guided reinforcement learning to discover feature transformations with downstream task utility, without any regard to privacy inferability. Phase II employs a variational LSTM seq2seq encoder-decoder with a utility-privacy disentangled latent space design and adversarial-causal disentanglement regularization to suppress privacy signals during feature generation. Experiments on eight datasets show DELTA improves predictive performance by ~9.3% and reduces privacy leakage by ~35%, demonstrating robust, privacy-aware data transformation. 

**Abstract (ZH)**: 面向隐私保护的数据重编程（PPDR）的变分解耦生成学习框架：DELTA 

---
# Valid Property-Enhanced Contrastive Learning for Targeted Optimization & Resampling for Novel Drug Design 

**Title (ZH)**: 有效属性增强对比学习在目标优化与采样中的应用及其在新型药物设计中的应用 

**Authors**: Amartya Banerjee, Somnath Kar, Anirban Pal, Debabrata Maiti  

**Link**: [PDF](https://arxiv.org/pdf/2509.00684)  

**Abstract**: Efficiently steering generative models toward pharmacologically relevant regions of chemical space remains a major obstacle in molecular drug discovery under low-data regimes. We present VECTOR+: Valid-property-Enhanced Contrastive Learning for Targeted Optimization and Resampling, a framework that couples property-guided representation learning with controllable molecule generation. VECTOR+ applies to both regression and classification tasks and enables interpretable, data-efficient exploration of functional chemical space. We evaluate on two datasets: a curated PD-L1 inhibitor set (296 compounds with experimental $IC_{50}$ values) and a receptor kinase inhibitor set (2,056 molecules by binding mode). Despite limited training data, VECTOR+ generates novel, synthetically tractable candidates. Against PD-L1 (PDB 5J89), 100 of 8,374 generated molecules surpass a docking threshold of $-15.0$ kcal/mol, with the best scoring $-17.6$ kcal/mol compared to the top reference inhibitor ($-15.4$ kcal/mol). The best-performing molecules retain the conserved biphenyl pharmacophore while introducing novel motifs. Molecular dynamics (250 ns) confirm binding stability (ligand RMSD < $2.5$ angstroms). VECTOR+ generalizes to kinase inhibitors, producing compounds with stronger docking scores than established drugs such as brigatinib and sorafenib. Benchmarking against JT-VAE and MolGPT across docking, novelty, uniqueness, and Tanimoto similarity highlights the superior performance of our method. These results position our work as a robust, extensible approach for property-conditioned molecular design in low-data settings, bridging contrastive learning and generative modeling for reproducible, AI-accelerated discovery. 

**Abstract (ZH)**: EffectiveSteeringGenerativeModelsTowardPharmacologicallyRelevantRegionsofChemicalSpaceUnderLow-DataRegimesviaValid-Property-EnhancedContrastiveLearningforTargetedOptimizationandResampling 

---
# The Name-Free Gap: Policy-Aware Stylistic Control in Music Generation 

**Title (ZH)**: 无需名称的差距：政策感知的音乐生成风格控制 

**Authors**: Ashwin Nagarajan, Hao-Wen Dong  

**Link**: [PDF](https://arxiv.org/pdf/2509.00654)  

**Abstract**: Text-to-music models capture broad attributes such as instrumentation or mood, but fine-grained stylistic control remains an open challenge. Existing stylization methods typically require retraining or specialized conditioning, which complicates reproducibility and limits policy compliance when artist names are restricted. We study whether lightweight, human-readable modifiers sampled from a large language model can provide a policy-robust alternative for stylistic control. Using MusicGen-small, we evaluate two artists: Billie Eilish (vocal pop) and Ludovico Einaudi (instrumental piano). For each artist, we use fifteen reference excerpts and evaluate matched seeds under three conditions: baseline prompts, artist-name prompts, and five descriptor sets. All prompts are generated using a large language model. Evaluation uses both VGGish and CLAP embeddings with distributional and per-clip similarity measures, including a new min-distance attribution metric. Results show that artist names are the strongest control signal across both artists, while name-free descriptors recover much of this effect. This highlights that existing safeguards such as the restriction of artist names in music generation prompts may not fully prevent style imitation. Cross-artist transfers reduce alignment, showing that descriptors encode targeted stylistic cues. We also present a descriptor table across ten contemporary artists to illustrate the breadth of the tokens. Together these findings define the name-free gap, the controllability difference between artist-name prompts and policy-compliant descriptors, shown through a reproducible evaluation protocol for prompt-level controllability. 

**Abstract (ZH)**: 基于文本的音乐模型捕捉了广泛的属性，如乐器或情绪，但细致風格控制仍然是一个开放的挑战。现有的風格化方法通常需要重新训练或特殊条件设置，这 complicate 了再现性和限制艺术家名称时的政策遵守。我们研究轻量级的、易读的修改器是否有能力提供一种政策稳健的替代方案以实现風格控制。使用 MusicGen-small，我们评估了两位艺术家：比莉·艾利什（流行演唱）和柳德维科·艾努阿里（乐器钢琴）。对于每位艺术家，我们使用十五个参考片段，并在三种条件下评估匹配的种子：基线提示、艺术家名称提示以及五组描述符。所有提示均由大型语言模型生成。评估使用 VGGish 和 CLAP 矢量，并采用分布性和每段片段相似性度量，包括一个新提出的最小距离归因度量。结果表明，艺术家名称在两位艺术家中的控制信号最强，而无名描述符恢复了大部分这种效果。这表明现有的安全措施，如音乐生成提示中对艺术家名称的限制，可能无法完全防止風格模仿。跨艺术家转移降低了对齐度，表明描述符编码了针对性的風格线索。我们还呈现了十位当代艺术家的描述符表，以展示令牌的多样性和覆盖面。这些发现定义了无名差值，即艺术家名称提示与政策合规描述符之间可控性差异，并通过可再现的评估协议展示了提示水平可控性的差异。 

---
# IndiaWeatherBench: A Dataset and Benchmark for Data-Driven Regional Weather Forecasting over India 

**Title (ZH)**: 印度气象基准：印度区域天气预报的数据驱动数据集与基准 

**Authors**: Tung Nguyen, Harkanwar Singh, Nilay Naharas, Lucas Bandarkar, Aditya Grover  

**Link**: [PDF](https://arxiv.org/pdf/2509.00653)  

**Abstract**: Regional weather forecasting is a critical problem for localized climate adaptation, disaster mitigation, and sustainable development. While machine learning has shown impressive progress in global weather forecasting, regional forecasting remains comparatively underexplored. Existing efforts often use different datasets and experimental setups, limiting fair comparison and reproducibility. We introduce IndiaWeatherBench, a comprehensive benchmark for data-driven regional weather forecasting focused on the Indian subcontinent. IndiaWeatherBench provides a curated dataset built from high-resolution regional reanalysis products, along with a suite of deterministic and probabilistic metrics to facilitate consistent training and evaluation. To establish strong baselines, we implement and evaluate a range of models across diverse architectures, including UNets, Transformers, and Graph-based networks, as well as different boundary conditioning strategies and training objectives. While focused on India, IndiaWeatherBench is easily extensible to other geographic regions. We open-source all raw and preprocessed datasets, model implementations, and evaluation pipelines to promote accessibility and future development. We hope IndiaWeatherBench will serve as a foundation for advancing regional weather forecasting research. Code is available at this https URL. 

**Abstract (ZH)**: 印度区域天气预报是一个关键问题，涉及本地气候变化适应、灾害缓解和可持续发展。虽然机器学习在全局天气预报方面取得了令人印象深刻的进展，但区域预报仍相对未被充分探索。现有的努力经常使用不同的数据集和实验设置，限制了公正的比较和重现性。我们引入了IndiaWeatherBench，这是一个针对印度次大陆的数据驱动区域天气预报基准评估。IndiaWeatherBench提供了一个精心构建的数据集，基于高分辨率区域再分析产品，并提供了一套确定性和概率性指标，以促进一致的训练和评估。为了建立强大的基线，我们在多种架构、包括UNets、Transformer和基于图的网络，以及不同的边界条件策略和训练目标模型中进行了实现和评估。虽然集中在印度，但IndiaWeatherBench可以轻松扩展到其他地理区域。我们开源了所有原始和预处理的数据集、模型实现和评估管道，以促进访问和未来的发展。我们希望IndiaWeatherBench能成为推进区域天气预报研究的基础。代码可在以下链接获取：this https URL。 

---
# NMR-Solver: Automated Structure Elucidation via Large-Scale Spectral Matching and Physics-Guided Fragment Optimization 

**Title (ZH)**: NMR-Solver：通过大规模光谱匹配和物理引导的片段优化实现自动结构解析 

**Authors**: Yongqi Jin, Jun-Jie Wang, Fanjie Xu, Xiaohong Ji, Zhifeng Gao, Linfeng Zhang, Guolin Ke, Rong Zhu, Weinan E  

**Link**: [PDF](https://arxiv.org/pdf/2509.00640)  

**Abstract**: Nuclear Magnetic Resonance (NMR) spectroscopy is one of the most powerful and widely used tools for molecular structure elucidation in organic chemistry. However, the interpretation of NMR spectra to determine unknown molecular structures remains a labor-intensive and expertise-dependent process, particularly for complex or novel compounds. Although recent methods have been proposed for molecular structure elucidation, they often underperform in real-world applications due to inherent algorithmic limitations and limited high-quality data. Here, we present NMR-Solver, a practical and interpretable framework for the automated determination of small organic molecule structures from $^1$H and $^{13}$C NMR spectra. Our method introduces an automated framework for molecular structure elucidation, integrating large-scale spectral matching with physics-guided fragment-based optimization that exploits atomic-level structure-spectrum relationships in NMR. We evaluate NMR-Solver on simulated benchmarks, curated experimental data from the literature, and real-world experiments, demonstrating its strong generalization, robustness, and practical utility in challenging, real-life scenarios. NMR-Solver unifies computational NMR analysis, deep learning, and interpretable chemical reasoning into a coherent system. By incorporating the physical principles of NMR into molecular optimization, it enables scalable, automated, and chemically meaningful molecular identification, establishing a generalizable paradigm for solving inverse problems in molecular science. 

**Abstract (ZH)**: 核磁共振（NMR）光谱学是有机化学中用于分子结构解析最强大且最常用的方法之一。然而，从NMR光谱中解读未知分子结构仍然是一项劳动密集型且依赖专业知识的过程，特别是在处理复杂或新颖化合物时。尽管最近提出了若干种用于分子结构解析的方法，但它们在实际应用中往往表现不佳，这主要是由于算法固有的限制以及高质量数据的局限性。在这里，我们介绍了NMR-Solver，这是一种实用且可解释的框架，用于从$^1$H和$^{13}$C NMR光谱自动确定小有机分子的结构。我们的方法引入了分子结构自动解析的框架，结合了大规模光谱匹配和基于原子级结构-光谱关系的物理引导片段优化。我们评估了NMR-Solver在模拟基准、文献中筛选出的实验数据以及真实世界实验中的表现，证明了其强大的泛化能力、可靠性和在挑战性实际情况中的实用性。NMR-Solver将计算NMR分析、深度学习和可解释的化学推理统一于一个连贯的系统中。通过将NMR的物理原理纳入分子优化中，它实现了可扩展、自动且化学上有意义的分子识别，确立了一个分子科学中解决逆问题的通用范式。 

---
# Enabling Trustworthy Federated Learning via Remote Attestation for Mitigating Byzantine Threats 

**Title (ZH)**: 通过远程证明缓解拜占庭威胁以实现可信的联邦学习 

**Authors**: Chaoyu Zhang, Heng Jin, Shanghao Shi, Hexuan Yu, Sydney Johns, Y. Thomas Hou, Wenjing Lou  

**Link**: [PDF](https://arxiv.org/pdf/2509.00634)  

**Abstract**: Federated Learning (FL) has gained significant attention for its privacy-preserving capabilities, enabling distributed devices to collaboratively train a global model without sharing raw data. However, its distributed nature forces the central server to blindly trust the local training process and aggregate uncertain model updates, making it susceptible to Byzantine attacks from malicious participants, especially in mission-critical scenarios. Detecting such attacks is challenging due to the diverse knowledge across clients, where variations in model updates may stem from benign factors, such as non-IID data, rather than adversarial behavior. Existing data-driven defenses struggle to distinguish malicious updates from natural variations, leading to high false positive rates and poor filtering performance.
To address this challenge, we propose Sentinel, a remote attestation (RA)-based scheme for FL systems that regains client-side transparency and mitigates Byzantine attacks from a system security perspective. Our system employs code instrumentation to track control-flow and monitor critical variables in the local training process. Additionally, we utilize a trusted training recorder within a Trusted Execution Environment (TEE) to generate an attestation report, which is cryptographically signed and securely transmitted to the server. Upon verification, the server ensures that legitimate client training processes remain free from program behavior violation or data manipulation, allowing only trusted model updates to be aggregated into the global model. Experimental results on IoT devices demonstrate that Sentinel ensures the trustworthiness of the local training integrity with low runtime and memory overhead. 

**Abstract (ZH)**: 基于远程证明的联邦学习安全检测方案Sentinel 

---
# Forecasting the Ionosphere from Sparse GNSS Data with Temporal-Fusion Transformers 

**Title (ZH)**: 基于时间融合变换器的稀疏GNSS数据电离层预测 

**Authors**: Giacomo Acciarini, Simone Mestici, Halil Kelebek, Linnea Wolniewicz, Michael Vergalla, Madhulika Guhathakurta, Umaa Rebbapragada, Bala Poduval, Atılım Güneş Baydin, Frank Soboczenski  

**Link**: [PDF](https://arxiv.org/pdf/2509.00631)  

**Abstract**: The ionosphere critically influences Global Navigation Satellite Systems (GNSS), satellite communications, and Low Earth Orbit (LEO) operations, yet accurate prediction of its variability remains challenging due to nonlinear couplings between solar, geomagnetic, and thermospheric drivers. Total Electron Content (TEC), a key ionospheric parameter, is derived from GNSS observations, but its reliable forecasting is limited by the sparse nature of global measurements and the limited accuracy of empirical models, especially during strong space weather conditions. In this work, we present a machine learning framework for ionospheric TEC forecasting that leverages Temporal Fusion Transformers (TFT) to predict sparse ionosphere data. Our approach accommodates heterogeneous input sources, including solar irradiance, geomagnetic indices, and GNSS-derived vertical TEC, and applies preprocessing and temporal alignment strategies. Experiments spanning 2010-2025 demonstrate that the model achieves robust predictions up to 24 hours ahead, with root mean square errors as low as 3.33 TECU. Results highlight that solar EUV irradiance provides the strongest predictive signals. Beyond forecasting accuracy, the framework offers interpretability through attention-based analysis, supporting both operational applications and scientific discovery. To encourage reproducibility and community-driven development, we release the full implementation as the open-source toolkit \texttt{ionopy}. 

**Abstract (ZH)**: 基于 Temporal Fusion Transformers 的电离层总电子含量预报方法：稀疏数据的机器学习框架 

---
# Towards Methane Detection Onboard Satellites 

**Title (ZH)**: 面向卫星载荷的甲烷检测技术研发 

**Authors**: Maggie Chen, Hala Lambdouar, Luca Marini, Laura Martínez-Ferrer, Chris Bridges, Giacomo Acciarini  

**Link**: [PDF](https://arxiv.org/pdf/2509.00626)  

**Abstract**: Methane is a potent greenhouse gas and a major driver of climate change, making its timely detection critical for effective mitigation. Machine learning (ML) deployed onboard satellites can enable rapid detection while reducing downlink costs, supporting faster response systems. Conventional methane detection methods often rely on image processing techniques, such as orthorectification to correct geometric distortions and matched filters to enhance plume signals. We introduce a novel approach that bypasses these preprocessing steps by using \textit{unorthorectified} data (UnorthoDOS). We find that ML models trained on this dataset achieve performance comparable to those trained on orthorectified data. Moreover, we also train models on an orthorectified dataset, showing that they can outperform the matched filter baseline (mag1c). We release model checkpoints and two ML-ready datasets comprising orthorectified and unorthorectified hyperspectral images from the Earth Surface Mineral Dust Source Investigation (EMIT) sensor at this https URL , along with code at this https URL. 

**Abstract (ZH)**: 甲烷是一种强温室气体，是气候变化的主要驱动因素，因此其及时检测对于有效的缓解措施至关重要。部署在卫星上的机器学习（ML）可以实现快速检测，降低下行链路成本，支持更快速的响应系统。传统的甲烷检测方法通常依赖于图像处理技术，如正射校正以纠正几何畸变和匹配滤波器以增强羽流信号。我们提出了一种新的方法，通过使用未经正射校正的数据（UnorthoDOS）绕过这些预处理步骤。我们发现，训练该数据集上的ML模型可以达到与训练正射校正数据集上的模型相当的性能。此外，我们还在正射校正数据集上训练模型，结果显示这些模型可以超越匹配滤波器基准（mag1c）。我们在此处发布模型检查点以及包含地球表面矿物灰尘源调查（EMIT）传感器正射校正和未经正射校正的高光谱图像的两个ML准备好的数据集，访问链接：[此链接]，同时提供相关代码，访问链接：[此链接]。 

---
# Federated Survival Analysis with Node-Level Differential Privacy: Private Kaplan-Meier Curves 

**Title (ZH)**: 节点级差分隐私下的联邦生存分析：私有Kaplan-Meier曲线 

**Authors**: Narasimha Raghavan Veeraragavan, Jan Franz Nygård  

**Link**: [PDF](https://arxiv.org/pdf/2509.00615)  

**Abstract**: We investigate how to calculate Kaplan-Meier survival curves across multiple health-care jurisdictions while protecting patient privacy with node-level differential privacy. Each site discloses its curve only once, adding Laplace noise whose scale is determined by the length of the common time grid; the server then averages the noisy curves, so the overall privacy budget remains unchanged. We benchmark four one-shot smoothing techniques: Discrete Cosine Transform, Haar Wavelet shrinkage, adaptive Total-Variation denoising, and a parametric Weibull fit on the NCCTG lung-cancer cohort under five privacy levels and three partition scenarios (uniform, moderately skewed, highly imbalanced). Total-Variation gives the best mean accuracy, whereas the frequency-domain smoothers offer stronger worst-case robustness and the Weibull model shows the most stable behaviour at the strictest privacy setting. Across all methods the released curves keep the empirical log-rank type-I error below fifteen percent for privacy budgets of 0.5 and higher, demonstrating that clinically useful survival information can be shared without iterative training or heavy cryptography. 

**Abstract (ZH)**: 我们在多个医疗保健管辖区计算Kaplan-Meier生存曲线的同时保护患者隐私：基于节点级差分隐私的方法研究 

---
# Can AI be Auditable? 

**Title (ZH)**: AI可审讯吗？ 

**Authors**: Himanshu Verma, Kirtan Path, Eva Thelisson  

**Link**: [PDF](https://arxiv.org/pdf/2509.00575)  

**Abstract**: Auditability is defined as the capacity of AI systems to be independently assessed for compliance with ethical, legal, and technical standards throughout their lifecycle. The chapter explores how auditability is being formalized through emerging regulatory frameworks, such as the EU AI Act, which mandate documentation, risk assessments, and governance structures. It analyzes the diverse challenges facing AI auditability, including technical opacity, inconsistent documentation practices, lack of standardized audit tools and metrics, and conflicting principles within existing responsible AI frameworks. The discussion highlights the need for clear guidelines, harmonized international regulations, and robust socio-technical methodologies to operationalize auditability at scale. The chapter concludes by emphasizing the importance of multi-stakeholder collaboration and auditor empowerment in building an effective AI audit ecosystem. It argues that auditability must be embedded in AI development practices and governance infrastructures to ensure that AI systems are not only functional but also ethically and legally aligned. 

**Abstract (ZH)**: AI系统的审计性被定义为在其生命周期中独立评估其遵守伦理、法律和技术标准的能力。本章探讨了通过新兴监管框架（如欧盟人工智能法案）来正式化审计性的方法，这些框架要求文档记录、风险评估和治理结构。本章分析了AI审计性面临的多种挑战，包括技术不透明性、不一致的文档实践、缺乏标准化的审计工具和度量标准，以及现有负责任人工智能框架中的冲突原则。讨论强调了需要清晰的指导方针、协调的国际法规和强大的社会技术方法来大规模实施审计性。本章最后强调了多利益相关者合作和审计人员赋权的重要性，以构建有效的AI审计生态系统。本章认为，审计性必须嵌入到AI开发实践和治理基础设施中，以确保AI系统不仅具备功能性，而且符合伦理和法律要求。 

---
# ResearchQA: Evaluating Scholarly Question Answering at Scale Across 75 Fields with Survey-Mined Questions and Rubrics 

**Title (ZH)**: ResearchQA：通过调查获取的问题和评分标准在75个学科领域中大规模评估学术问答能力 

**Authors**: Li S. Yifei, Allen Chang, Chaitanya Malaviya, Mark Yatskar  

**Link**: [PDF](https://arxiv.org/pdf/2509.00496)  

**Abstract**: Evaluating long-form responses to research queries heavily relies on expert annotators, restricting attention to areas like AI where researchers can conveniently enlist colleagues. Yet, research expertise is widespread: survey articles synthesize knowledge distributed across the literature. We introduce ResearchQA, a resource for evaluating LLM systems by distilling survey articles from 75 research fields into 21K queries and 160K rubric items. Each rubric, derived jointly with queries from survey sections, lists query-specific answer evaluation criteria, i.e., citing papers, making explanations, and describing limitations. Assessments by 31 Ph.D. annotators in 8 fields indicate 96% of queries support Ph.D. information needs and 87% of rubric items should be addressed in system responses by a sentence or more. Using our rubrics, we are able to construct an automatic pairwise judge obtaining 74% agreement with expert judgments. We leverage ResearchQA to analyze competency gaps in 18 systems in over 7.6K pairwise evaluations. No parametric or retrieval-augmented system we evaluate exceeds 70% on covering rubric items, and the highest-ranking agentic system shows 75% coverage. Error analysis reveals that the highest-ranking system fully addresses less than 11% of citation rubric items, 48% of limitation items, and 49% of comparison items. We release our data to facilitate more comprehensive multi-field evaluations. 

**Abstract (ZH)**: 基于调查文章评估长格式研究回答：ResearchQA资源的构建与应用 

---
# A Novel Method to Determine Total Oxidant Concentration Produced by Non-Thermal Plasma Based on Image Processing and Machine Learning 

**Title (ZH)**: 基于图像处理和机器学习确定非热等离子体产生总氧化剂浓度的新方法 

**Authors**: Mirkan Emir Sancak, Unal Sen, Ulker Diler Keris-Sen  

**Link**: [PDF](https://arxiv.org/pdf/2509.00479)  

**Abstract**: Accurate determination of total oxidant concentration ([Ox]_{tot}) in non-thermal plasma (NTP)-treated aqueous systems remains a critical challenge due to the transient nature of reactive oxygen and nitrogen species and the subjectivity of conventional titration methods used for [Ox]_{tot} determination. This study introduces a novel, color-based computer analysis (CBCA) method that integrates advanced image processing with machine learning (ML) to quantify colorimetric shifts in potassium iodide (KI) solutions during oxidation. First, a custom-built visual data acquisition system captured high-resolution video of the color transitions in a KI solution during oxidation with an NTP system. The change in [Ox]_{tot} during the experiments was monitored with a standard titrimetric method. Second, the captured frames were processed using a robust image processing pipeline to extract RGB, HSV, and Lab color features. The extracted features were statistically evaluated, and the results revealed strong linear correlations with the measured [Ox]_{tot} values, particularly in the saturation (HSV), a and b (Lab), and blue (RGB) channels. Subsequently, the [Ox]_{tot} measurements and the extracted color features were used to train and validate five ML models. Among them, linear regression and gradient boosting models achieved the highest predictive accuracy (R^2 > 0.990). It was also found that reducing the feature set from nine to four resulted in comparable performance with improved prediction efficiency, especially for gradient boosting. Finally, comparison of the model predictions with real titration measurements revealed that the CBCA system successfully predicts the [Ox]_{tot} in KI solution with high accuracy (R^2 > 0.998) even with a reduced number of features. 

**Abstract (ZH)**: 基于高级图像处理与机器学习的非热等离子体处理水中体系总氧化剂浓度准确测定新方法 

---
# Cross-Domain Malware Detection via Probability-Level Fusion of Lightweight Gradient Boosting Models 

**Title (ZH)**: 跨域恶意软件检测：轻量级渐进提升模型在概率级别上的融合 

**Authors**: Omar Khalid Ali Mohamed  

**Link**: [PDF](https://arxiv.org/pdf/2509.00476)  

**Abstract**: The escalating sophistication of malware necessitates robust detection mechanisms that generalize across diverse data sources. Traditional single-dataset models struggle with cross-domain generalization and often incur high computational costs. This paper presents a novel, lightweight framework for malware detection that employs probability-level fusion across three distinct datasets: EMBER (static features), API Call Sequences (behavioral features), and CIC Obfuscated Memory (memory patterns). Our method trains individual LightGBM classifiers on each dataset, selects top predictive features to ensure efficiency, and fuses their prediction probabilities using optimized weights determined via grid search. Extensive experiments demonstrate that our fusion approach achieves a macro F1-score of 0.823 on a cross-domain validation set, significantly outperforming individual models and providing superior generalization. The framework maintains low computational overhead, making it suitable for real-time deployment, and all code and data are provided for full reproducibility. 

**Abstract (ZH)**: escalating malware sophistication necessitates robust detection mechanisms that generalize across diverse data sources. 

---
# Curriculum Guided Personalized Subgraph Federated Learning 

**Title (ZH)**: Curriculum Guided Personalized Subgraph Federated Learning 

**Authors**: Minku Kang, Hogun Park  

**Link**: [PDF](https://arxiv.org/pdf/2509.00402)  

**Abstract**: Subgraph Federated Learning (FL) aims to train Graph Neural Networks (GNNs) across distributed private subgraphs, but it suffers from severe data heterogeneity. To mitigate data heterogeneity, weighted model aggregation personalizes each local GNN by assigning larger weights to parameters from clients with similar subgraph characteristics inferred from their current model states. However, the sparse and biased subgraphs often trigger rapid overfitting, causing the estimated client similarity matrix to stagnate or even collapse. As a result, aggregation loses effectiveness as clients reinforce their own biases instead of exploiting diverse knowledge otherwise available. To this end, we propose a novel personalized subgraph FL framework called Curriculum guided personalized sUbgraph Federated Learning (CUFL). On the client side, CUFL adopts Curriculum Learning (CL) that adaptively selects edges for training according to their reconstruction scores, exposing each GNN first to easier, generic cross-client substructures and only later to harder, client-specific ones. This paced exposure prevents early overfitting to biased patterns and enables gradual personalization. By regulating personalization, the curriculum also reshapes server aggregation from exchanging generic knowledge to propagating client-specific knowledge. Further, CUFL improves weighted aggregation by estimating client similarity using fine-grained structural indicators reconstructed on a random reference graph. Extensive experiments on six benchmark datasets confirm that CUFL achieves superior performance compared to relevant baselines. Code is available at this https URL. 

**Abstract (ZH)**: 基于课程引导的个性化子图联邦学习（CUFL） 

---
# A Study on the Framework for Evaluating the Ethics and Trustworthiness of Generative AI 

**Title (ZH)**: 关于评估生成式人工智能伦理与可信性的框架的研究 

**Authors**: Cheonsu Jeong, Seunghyun Lee, Sunny Jeong, Sungsu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.00398)  

**Abstract**: This study provides an in_depth analysis of the ethical and trustworthiness challenges emerging alongside the rapid advancement of generative artificial intelligence (AI) technologies and proposes a comprehensive framework for their systematic evaluation. While generative AI, such as ChatGPT, demonstrates remarkable innovative potential, it simultaneously raises ethical and social concerns, including bias, harmfulness, copyright infringement, privacy violations, and hallucination. Current AI evaluation methodologies, which mainly focus on performance and accuracy, are insufficient to address these multifaceted issues. Thus, this study emphasizes the need for new human_centered criteria that also reflect social impact. To this end, it identifies key dimensions for evaluating the ethics and trustworthiness of generative AI_fairness, transparency, accountability, safety, privacy, accuracy, consistency, robustness, explainability, copyright and intellectual property protection, and source traceability and develops detailed indicators and assessment methodologies for each. Moreover, it provides a comparative analysis of AI ethics policies and guidelines in South Korea, the United States, the European Union, and China, deriving key approaches and implications from each. The proposed framework applies across the AI lifecycle and integrates technical assessments with multidisciplinary perspectives, thereby offering practical means to identify and manage ethical risks in real_world contexts. Ultimately, the study establishes an academic foundation for the responsible advancement of generative AI and delivers actionable insights for policymakers, developers, users, and other stakeholders, supporting the positive societal contributions of AI technologies. 

**Abstract (ZH)**: 本研究深入分析了伴随生成式人工智能（AI）技术快速发展而产生的伦理和可信度挑战，并提出了一套全面的体系框架以系统评估这些问题。尽管生成式AI，如ChatGPT，展现出显著的创新潜力，但同时也引发了伦理和社会关注，包括偏见、危害性、版权侵权、隐私侵犯和幻觉。当前的AI评估方法主要关注性能和准确性，不足以应对这些复杂的问题。因此，本研究强调需要新的以人为核心的标准，以反映社会影响。为此，本研究确定了评估生成式AI伦理和可信度的关键维度，包括公平性、透明度、问责制、安全性、隐私保护、准确性、一致性、稳健性、可解释性、版权和知识产权保护、以及来源追溯，并为每个维度开发了详细的指标和评估方法。此外，本研究还对韩国、美国、欧盟和中国的AI伦理政策和指南进行了比较分析，从每个地区的做法中提炼出关键方法和启示。所提出的框架适用于AI生命周期的每一个阶段，并将技术评估与多学科视角相结合，从而提供实际手段，在现实世界中识别和管理伦理风险。最终，本研究为生成式AI负责任的发展奠定了学术基础，并为政策制定者、开发者、用户和其他利益相关者提供了可操作的见解，支持人工智能技术对社会的积极贡献。 

---
# Beyond Negative Transfer: Disentangled Preference-Guided Diffusion for Cross-Domain Sequential Recommendation 

**Title (ZH)**: 超越负迁移：解耦的偏好导向扩散在跨域序列推荐中的应用 

**Authors**: Xiaoxin Ye, Chengkai Huang, Hongtao Huang, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2509.00389)  

**Abstract**: Cross-Domain Sequential Recommendation (CDSR) leverages user behaviors across domains to enhance recommendation quality. However, naive aggregation of sequential signals can introduce conflicting domain-specific preferences, leading to negative transfer. While Sequential Recommendation (SR) already suffers from noisy behaviors such as misclicks and impulsive actions, CDSR further amplifies this issue due to domain heterogeneity arising from diverse item types and user intents. The core challenge is disentangling three intertwined signals: domain-invariant preferences, domain-specific preferences, and noise. Diffusion Models (DMs) offer a generative denoising framework well-suited for disentangling complex user preferences and enhancing robustness to noise. Their iterative refinement process enables gradual denoising, making them effective at capturing subtle preference signals. However, existing applications in recommendation face notable limitations: sequential DMs often conflate shared and domain-specific preferences, while cross-domain collaborative filtering DMs neglect temporal dynamics, limiting their ability to model evolving user preferences. To bridge these gaps, we propose \textbf{DPG-Diff}, a novel Disentangled Preference-Guided Diffusion Model, the first diffusion-based approach tailored for CDSR, to or best knowledge. DPG-Diff decomposes user preferences into domain-invariant and domain-specific components, which jointly guide the reverse diffusion process. This disentangled guidance enables robust cross-domain knowledge transfer, mitigates negative transfer, and filters sequential noise. Extensive experiments on real-world datasets demonstrate that DPG-Diff consistently outperforms state-of-the-art baselines across multiple metrics. 

**Abstract (ZH)**: 跨域序列推荐（CDSR）利用用户跨域行为提高推荐质量。然而，简单的序列信号聚合可能会引入冲突的领域特定偏好，导致负面迁移。尽管序列推荐（SR）已经受到诸如误点击和冲动行为等噪音行为的困扰，跨域序列推荐（CDSR）由于不同领域中多元化项目类型和用户意图导致的异质性，进一步放大了这一问题。核心挑战在于拆分交织的三个信号：领域不变偏好、领域特定偏好和噪音。扩散模型（DMs）提供了一个生成降噪框架，适合拆分复杂用户偏好并增强对噪音的鲁棒性。其迭代精炼过程能够逐步降噪，使它们能够捕捉到微妙的偏好信号。然而，现有推荐中的应用面临显著限制：顺序DM往往混淆了共享偏好和领域特定偏好，而跨域协作过滤的DM忽略了时间动态性，限制了它们建模用户偏好演变的能力。为弥补这些差距，我们提出了一种新颖的拆分偏好引导扩散模型（DPG-Diff），这是迄今为止第一个专门为CDSR设计的基于扩散的方法。DPG-Diff将用户偏好分解为领域不变和领域特定两个部分，并共同指导反向扩散过程。这种拆分的指导能够促进稳健的跨域知识迁移，减轻负面迁移，并过滤掉序列中的噪音。在多种指标上，DPG-Diff在实际数据集上的广泛实验中始终优于最先进的基线方法。 

---
# Unifying Adversarial Perturbation for Graph Neural Networks 

**Title (ZH)**: 图神经网络中的统一对抗扰动 

**Authors**: Jinluan Yang, Ruihao Zhang, Zhengyu Chen, Fei Wu, Kun Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00387)  

**Abstract**: This paper studies the vulnerability of Graph Neural Networks (GNNs) to adversarial attacks on node features and graph structure. Various methods have implemented adversarial training to augment graph data, aiming to bolster the robustness and generalization of GNNs. These methods typically involve applying perturbations to the node feature, weights, or graph structure and subsequently minimizing the loss by learning more robust graph model parameters under the adversarial perturbations. Despite the effectiveness of adversarial training in enhancing GNNs' robustness and generalization abilities, its application has been largely confined to specific datasets and GNN types. In this paper, we propose a novel method, PerturbEmbedding, that integrates adversarial perturbation and training, enhancing GNNs' resilience to such attacks and improving their generalization ability. PerturbEmbedding performs perturbation operations directly on every hidden embedding of GNNs and provides a unified framework for most existing perturbation strategies/methods. We also offer a unified perspective on the forms of perturbations, namely random and adversarial perturbations. Through experiments on various datasets using different backbone models, we demonstrate that PerturbEmbedding significantly improves both the robustness and generalization abilities of GNNs, outperforming existing methods. The rejection of both random (non-targeted) and adversarial (targeted) perturbations further enhances the backbone model's performance. 

**Abstract (ZH)**: 本文研究了图神经网络（GNNs）在节点特征和图结构上的对抗攻击的脆弱性。各种方法通过对抗训练增强了图数据，旨在提升GNNs的 robustness和泛化能力。这些方法通常涉及对节点特征、权重或图结构进行扰动，并在对抗扰动下学习更 robust的图模型参数以最小化损失。尽管对抗训练在增强GNNs的 robustness和泛化能力方面表现出有效性，但其应用主要局限于特定的数据集和GNN类型。在本文中，我们提出了一种名为PerturbEmbedding的新方法，该方法整合了对抗扰动和训练，增强了GNNs对抗此类攻击的鲁棒性，并改善了其泛化能力。PerturbEmbedding直接对GNNs的每隐藏嵌入进行扰动操作，并提供了一种多数现有扰动策略/方法的统一框架。我们还从统一的角度分析了扰动的形式，即随机扰动和对抗扰动。通过使用不同主干模型在多种数据集上的实验，我们展示了PerturbEmbedding显著提高了GNNs的 robustness和泛化能力，优于现有方法。同时，拒绝随机（非目标）和对抗（目标）扰动进一步提高了主干模型的性能。 

---
# Target-Oriented Single Domain Generalization 

**Title (ZH)**: 目标导向的单领域泛化 

**Authors**: Marzi Heidari, Yuhong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.00351)  

**Abstract**: Deep models trained on a single source domain often fail catastrophically under distribution shifts, a critical challenge in Single Domain Generalization (SDG). While existing methods focus on augmenting source data or learning invariant features, they neglect a readily available resource: textual descriptions of the target deployment environment. We propose Target-Oriented Single Domain Generalization (TO-SDG), a novel problem setup that leverages the textual description of the target domain, without requiring any target data, to guide model generalization. To address TO-SDG, we introduce Spectral TARget Alignment (STAR), a lightweight module that injects target semantics into source features by exploiting visual-language models (VLMs) such as CLIP. STAR uses a target-anchored subspace derived from the text embedding of the target description to recenter image features toward the deployment domain, then utilizes spectral projection to retain directions aligned with target cues while discarding source-specific noise. Moreover, we use a vision-language distillation to align backbone features with VLM's semantic geometry. STAR further employs feature-space Mixup to ensure smooth transitions between source and target-oriented representations. Experiments across various image classification and object detection benchmarks demonstrate STAR's superiority. This work establishes that minimal textual metadata, which is a practical and often overlooked resource, significantly enhances generalization under severe data constraints, opening new avenues for deploying robust models in target environments with unseen data. 

**Abstract (ZH)**: 基于目标导向的单源泛化（TO-SDG）：利用目标领域文本描述引导模型泛化 

---
# Theory Foundation of Physics-Enhanced Residual Learning 

**Title (ZH)**: 物理增强残差学习的理论基础 

**Authors**: Shixiao Liang, Wang Chen, Keke Long, Peng Zhang, Xiaopeng Li, Jintao Ke  

**Link**: [PDF](https://arxiv.org/pdf/2509.00348)  

**Abstract**: Intensive studies have been conducted in recent years to integrate neural networks with physics models to balance model accuracy and interpretability. One recently proposed approach, named Physics-Enhanced Residual Learning (PERL), is to use learning to estimate the residual between the physics model prediction and the ground truth. Numeral examples suggested that integrating such residual with physics models in PERL has three advantages: (1) a reduction in the number of required neural network parameters; (2) faster convergence rates; and (3) fewer training samples needed for the same computational precision. However, these numerical results lack theoretical justification and cannot be adequately explained.
This paper aims to explain these advantages of PERL from a theoretical perspective. We investigate a general class of problems with Lipschitz continuity properties. By examining the relationships between the bounds to the loss function and residual learning structure, this study rigorously proves a set of theorems explaining the three advantages of PERL.
Several numerical examples in the context of automated vehicle trajectory prediction are conducted to illustrate the proposed theorems. The results confirm that, even with significantly fewer training samples, PERL consistently achieves higher accuracy than a pure neural network. These results demonstrate the practical value of PERL in real world autonomous driving applications where corner case data are costly or hard to obtain. PERL therefore improves predictive performance while reducing the amount of data required. 

**Abstract (ZH)**: 近年来，已经开展了大量的研究工作，旨在将神经网络与物理模型相结合，以平衡模型的准确性和可解释性。一种最近提出的方法，称为物理增强残差学习（PERL），是通过学习估计物理模型预测与地面真实值之间的残差。数值实验表明，将这种残差与PERL中的物理模型结合具有三个优点：（1）减少所需的神经网络参数数量；（2）加快收敛速率；（3）以相同的计算精度需要更少的训练样本。然而，这些数值结果缺乏理论依据，无法充分解释。

本文旨在从理论角度解释PERL的这些优点。我们研究了一类具有Lipschitz连续性质的一般问题。通过考察损失函数边界与残差学习结构之间的关系，本研究严格证明了一系列定理，解释了PERL的三个优点。

我们还在自动车辆轨迹预测的背景下进行了几个数值实验，以说明提出的定理。结果表明，即使使用显著较少的训练样本，PERL也始终能够实现更高的准确性。这些结果表明，PERL在实际应用场景中的确有价值，尤其是 corner case 数据成本高或难以获取的情况下，PERL可以提高预测性能并减少所需的数据量。 

---
# Scalable Option Learning in High-Throughput Environments 

**Title (ZH)**: 高吞吐量环境下的可扩展选项学习 

**Authors**: Mikael Henaff, Scott Fujimoto, Michael Rabbat  

**Link**: [PDF](https://arxiv.org/pdf/2509.00338)  

**Abstract**: Hierarchical reinforcement learning (RL) has the potential to enable effective decision-making over long timescales. Existing approaches, while promising, have yet to realize the benefits of large-scale training. In this work, we identify and solve several key challenges in scaling hierarchical RL to high-throughput environments. We propose Scalable Option Learning (SOL), a highly scalable hierarchical RL algorithm which achieves a 25x higher throughput compared to existing hierarchical methods. We train our hierarchical agents using 20 billion frames of experience on the complex game of NetHack, significantly surpassing flat agents and demonstrating positive scaling trends. We also validate our algorithm on MiniHack and Mujoco environments, showcasing its general applicability. Our code is open sourced at this http URL. 

**Abstract (ZH)**: 层次强化学习（RL）有潜力支持长时间尺度上的有效决策。现有的方法虽然前景看好，但尚未实现大规模训练的益处。本文中，我们识别并解决了在高吞吐量环境中扩展层次RL的几个关键挑战。我们提出了一种高度可扩展的层次RL算法Scalable Option Learning (SOL)，其吞吐量比现有层次方法高出25倍。我们使用NetHack等复杂游戏的经验数据训练层次代理，显著超越了平级代理，并展示了积极的扩展趋势。我们还在MiniHack和Mujoco环境中验证了该算法，展示了其通用适用性。我们的代码已在以下地址开源：this http URL。 

---
# Continuously Tempered Diffusion Samplers 

**Title (ZH)**: 连续调整扩散采样器 

**Authors**: Ezra Erives, Bowen Jing, Peter Holderrieth, Tommi Jaakkola  

**Link**: [PDF](https://arxiv.org/pdf/2509.00316)  

**Abstract**: Annealing-based neural samplers seek to amortize sampling from unnormalized distributions by training neural networks to transport a family of densities interpolating from source to target. A crucial design choice in the training phase of such samplers is the proposal distribution by which locations are generated at which to evaluate the loss. Previous work has obtained such a proposal distribution by combining a partially learned transport with annealed Langevin dynamics. However, isolated modes and other pathological properties of the annealing path imply that such proposals achieve insufficient exploration and thereby lower performance post training. To remedy this, we propose continuously tempered diffusion samplers, which leverage exploration techniques developed in the context of molecular dynamics to improve proposal distributions. Specifically, a family of distributions across different temperatures is introduced to lower energy barriers at higher temperatures and drive exploration at the lower temperature of interest. We empirically validate improved sampler performance driven by extended exploration. Code is available at this https URL. 

**Abstract (ZH)**: 基于退火的神经采样器通过训练神经网络来运输插值于源分布和目标分布之间的一系列密度，以减轻不归一化分布的采样负担。此类采样器在训练阶段的关键设计选择是用于生成损失评估位置的提案分布。先前的工作通过结合部分学习的运输与退火朗格文动力学来获得这种提案分布。然而，退火路径中的孤立模式和其他病理性特征意味着这样的提案分布未能充分探索，从而导致训练后的性能降低。为此，我们提出了一种连续退火扩散采样器，利用分子动力学中开发的探索技术来改进提案分布。具体而言，在不同温度下引入一系列分布以在较高温度下降低能量障碍，并在较低温度下促进探索。我们通过扩展探索验证了采样器性能的提升。代码详见此链接：this https URL。 

---
# Intelligent Spectrum Management in Satellite Communications 

**Title (ZH)**: 卫星通信中的智能频谱管理 

**Authors**: Rakshitha De Silva, Shiva Raj Pokhrel, Jonathan Kua, Sithamparanathan Kandeepan  

**Link**: [PDF](https://arxiv.org/pdf/2509.00286)  

**Abstract**: Satellite Communication (SatCom) networks represent a fundamental pillar in modern global connectivity, facilitating reliable service and extensive coverage across a plethora of applications. The expanding demand for high-bandwidth services and the proliferation of mega satellite constellations highlight the limitations of traditional exclusive satellite spectrum allocation approaches. Cognitive Radio (CR) leading to Cognitive Satellite (CogSat) networks through Dynamic Spectrum Management (DSM), which enables the dynamic adaptability of radio equipment to environmental conditions for optimal performance, presents a promising solution for the emerging spectrum scarcity. In this survey, we explore the adaptation of intelligent DSM methodologies to SatCom, leveraging satellite network integrations. We discuss contributions and hurdles in regulations and standardizations in realizing intelligent DSM in SatCom, and deep dive into DSM techniques, which enable CogSat networks. Furthermore, we extensively evaluate and categorize state-of-the-art Artificial Intelligence (AI)/Machine Learning (ML) methods leveraged for DSM while exploring operational resilience and robustness of such integrations. In addition, performance evaluation metrics critical for adaptive resource management and system optimization in CogSat networks are thoroughly investigated. This survey also identifies open challenges and outlines future research directions in regulatory frameworks, network architectures, and intelligent spectrum management, paving the way for sustainable and scalable SatCom networks for enhanced global connectivity. 

**Abstract (ZH)**: 认知卫星（CogSat）网络中的智能频谱管理：现状、挑战与未来研究方向 

---
# SABER: A SQL-Compatible Semantic Document Processing System Based on Extended Relational Algebra 

**Title (ZH)**: SABER：一种基于扩展关系代数的SQL兼容语义文档处理系统 

**Authors**: Changjae Lee, Zhuoyue Zhao, Jinjun Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2509.00277)  

**Abstract**: The emergence of large-language models (LLMs) has enabled a new class of semantic data processing systems (SDPSs) to support declarative queries against unstructured documents. Existing SDPSs are, however, lacking a unified algebraic foundation, making their queries difficult to compose, reason, and optimize. We propose a new semantic algebra, SABER (Semantic Algebra Based on Extended Relational algebra), opening the possibility of semantic operations' logical plan construction, optimization, and formal correctness guarantees. We further propose to implement SABER in a SQL-compatible syntax so that it natively supports mixed structured/unstructured data processing. With SABER, we showcase the feasibility of providing a unified interface for existing SDPSs so that it can effectively mix and match any semantically-compatible operator implementation from any SDPS, greatly enhancing SABER's applicability for community contributions. 

**Abstract (ZH)**: 大型语言模型（LLMs）的出现使得一类新的语义数据处理系统（SDPSs）能够支持对非结构化文档进行声明性查询。现有SDPSs缺乏统一的代数基础，使得查询组合、推理和优化变得困难。我们提出了一种新的语义代数SABER（基于扩展关系代数的语义代数），开启了语义操作逻辑计划构造、优化和形式正确性保证的可能性。我们进一步提出以SQL兼容的语法实现SABER，使其原生支持结构化/非结构化数据处理的混合处理。通过SABER，我们展示了为现有SDPSs提供统一接口的可能性，使其能够有效混合和匹配来自任何SDPS的任何语义兼容的操作器实现，极大地提高了SABER的社区贡献适用性。 

---
# Revealing Hidden Precursors to Earthquakes via a Stress-Sensitive Transformation of Seismic Noise 

**Title (ZH)**: 通过地震噪声的应力敏感变换揭示地震前兆 

**Authors**: Nader Shakibay Senobari  

**Link**: [PDF](https://arxiv.org/pdf/2509.00268)  

**Abstract**: Earthquake prediction has long been one of the most elusive challenges in science. Laboratory experiments and simulations suggest that failure precursors should exist, yet reliable signals have remained unobserved in real-world seismic records, leaving open the question of whether they are absent in nature or simply hidden within noise. Here we introduce a stress-sensitive frequency-domain transformation that tracks energy differences between adjacent frequency bands, isolating subtle spectral changes linked to evolving shear and normal stress. Applied to both laboratory acoustic emission data and seismic records from seven major earthquakes (Mw 5.9-9.0), including the 2011 Tohoku and 2023 Turkey-Syria events, the transform consistently reveals precursory signatures, arc-like trajectories and accelerations toward extrema, emerging hours to days before rupture. These features are robust across diverse tectonic settings, from induced seismicity and volcanic collapse to continental strike-slip and subduction megathrust earthquakes. Our findings demonstrate that hidden precursors are indeed encoded in ambient seismic noise, offering a pathway toward real-time fault monitoring and actionable short-term earthquake forecasting. 

**Abstract (ZH)**: 地震预测一直是科学研究中最难以克服的挑战之一。虽然实验室实验和模拟表明应存在破裂前兆，但在真实的地震记录中仍未观察到可靠的信号，因此人们仍不确定前兆是不存在于自然界中，还是仅仅被噪音掩盖了。我们介绍了对能量差异进行频率域转换的一种应力敏感方法，该方法能够隔离与剪切应力和法向应力演化相关的微妙频谱变化。我们将该转换应用于实验室声发射数据以及七次主要地震（震级Mw 5.9-9.0）的地震记录，其中包括2011年东日本和2023年土耳其-叙利亚地震事件。该转换在地震发生前数小时到数天内始终揭示了先兆特征、类似弧线的轨迹和极值点的加速，这些特征在多种断层环境中表现出一致的鲁棒性，包括诱发地震、火山塌陷、大陆走滑断层地震和俯冲断层地震巨震。我们的研究结果证明，隐含的前兆确实编码在背景地震噪声中，提供了一条进行实时断层监测和行动型短期地震预测的可能性途径。 

---
# The Differential Meaning of Models: A Framework for Analyzing the Structural Consequences of Semantic Modeling Decisions 

**Title (ZH)**: 模型差异的意义：一种分析语义建模决策结构后果的框架 

**Authors**: Zachary K. Stine, James E. Deitrick  

**Link**: [PDF](https://arxiv.org/pdf/2509.00248)  

**Abstract**: The proliferation of methods for modeling of human meaning-making constitutes a powerful class of instruments for the analysis of complex semiotic systems. However, the field lacks a general theoretical framework for describing these modeling practices across various model types in an apples-to-apples way. In this paper, we propose such a framework grounded in the semiotic theory of C. S. Peirce. We argue that such models measure latent symbol geometries, which can be understood as hypotheses about the complex of semiotic agencies underlying a symbolic dataset. Further, we argue that in contexts where a model's value cannot be straightforwardly captured by proxy measures of performance, models can instead be understood relationally, so that the particular interpretive lens of a model becomes visible through its contrast with other models. This forms the basis of a theory of model semantics in which models, and the modeling decisions that constitute them, are themselves treated as signs. In addition to proposing the framework, we illustrate its empirical use with a few brief examples and consider foundational questions and future directions enabled by the framework. 

**Abstract (ZH)**: 人类意义构建方法的 proliferating 构成了一类 Powerful 分析复杂征候系统工具。然而，该领域缺乏一个适用于各种模型类型的通用理论框架，以统一描述这些建模实践。本文基于 C. S. Peirce 的征候理论提出这样一个框架。我们 argue 主张这些模型衡量潜在于象征数据集背后的征候机构的几何结构。进一步，我们主张在模型的价值不能通过代理性能指标直接捕捉的背景下，模型可以被理解为关系性的，从而使模型的特定解释性视角通过与其他模型的对比变得显而易见。这构成了一个模型语义理论的基础，即模型及其结构化决策本身被视为象征符号。除了提出该框架，我们还通过几个简短的例子示范其实证应用，并讨论该框架所支持的基础问题和未来方向。 

---
# Criteria for Credible AI-assisted Carbon Footprinting Systems: The Cases of Mapping and Lifecycle Modeling 

**Title (ZH)**: 可信人工智能辅助碳足迹核算系统的标准：以地图绘制和生命周期建模为例 

**Authors**: Shaena Ulissi, Andrew Dumit, P. James Joyce, Krishna Rao, Steven Watson, Sangwon Suh  

**Link**: [PDF](https://arxiv.org/pdf/2509.00240)  

**Abstract**: As organizations face increasing pressure to understand their corporate and products' carbon footprints, artificial intelligence (AI)-assisted calculation systems for footprinting are proliferating, but with widely varying levels of rigor and transparency. Standards and guidance have not kept pace with the technology; evaluation datasets are nascent; and statistical approaches to uncertainty analysis are not yet practical to apply to scaled systems. We present a set of criteria to validate AI-assisted systems that calculate greenhouse gas (GHG) emissions for products and materials. We implement a three-step approach: (1) Identification of needs and constraints, (2) Draft criteria development and (3) Refinements through pilots. The process identifies three use cases of AI applications: Case 1 focuses on AI-assisted mapping to existing datasets for corporate GHG accounting and product hotspotting, automating repetitive manual tasks while maintaining mapping quality. Case 2 addresses AI systems that generate complete product models for corporate decision-making, which require comprehensive validation of both component tasks and end-to-end performance. We discuss the outlook for Case 3 applications, systems that generate standards-compliant models. We find that credible AI systems can be built and that they should be validated using system-level evaluations rather than line-item review, with metrics such as benchmark performance, indications of data quality and uncertainty, and transparent documentation. This approach may be used as a foundation for practitioners, auditors, and standards bodies to evaluate AI-assisted environmental assessment tools. By establishing evaluation criteria that balance scalability with credibility requirements, our approach contributes to the field's efforts to develop appropriate standards for AI-assisted carbon footprinting systems. 

**Abstract (ZH)**: 人工智能辅助的碳足迹计算系统验证标准与方法 

---
# Evaluating the Effectiveness of Transformer Layers in Wav2Vec 2.0, XLS-R, and Whisper for Speaker Identification Tasks 

**Title (ZH)**: WAV2Vec 2.0、XLS-R 和 Whisper 在 speaker identification 任务中Transformer 层的有效性评价 

**Authors**: Linus Stuhlmann, Michael Alexander Saxer  

**Link**: [PDF](https://arxiv.org/pdf/2509.00230)  

**Abstract**: This study evaluates the performance of three advanced speech encoder models, Wav2Vec 2.0, XLS-R, and Whisper, in speaker identification tasks. By fine-tuning these models and analyzing their layer-wise representations using SVCCA, k-means clustering, and t-SNE visualizations, we found that Wav2Vec 2.0 and XLS-R capture speaker-specific features effectively in their early layers, with fine-tuning improving stability and performance. Whisper showed better performance in deeper layers. Additionally, we determined the optimal number of transformer layers for each model when fine-tuned for speaker identification tasks. 

**Abstract (ZH)**: 本研究评估了三种先进语音编码模型（Wav2Vec 2.0、XLS-R 和 Whisper）在说话人识别任务中的性能。通过微调这些模型并在层级表示上使用SVCCA、k-means聚类和t-SNE可视化进行分析，我们发现Wav2Vec 2.0和XLS-R在早期层能够有效捕捉说话人特异性特征，微调提高了稳定性和性能。Whisper在深层表现更佳。此外，我们确定了每种模型在为说话人识别任务微调时的最佳变换器层数。 

---
# Generalizable Audio Spoofing Detection using Non-Semantic Representations 

**Title (ZH)**: 使用非语义表示的通用音频 spoofing 检测 

**Authors**: Arnab Das, Yassine El Kheir, Carlos Franzreb, Tim Herzig, Tim Polzehl, Sebastian Möller  

**Link**: [PDF](https://arxiv.org/pdf/2509.00186)  

**Abstract**: Rapid advancements in generative modeling have made synthetic audio generation easy, making speech-based services vulnerable to spoofing attacks. Consequently, there is a dire need for robust countermeasures more than ever. Existing solutions for deepfake detection are often criticized for lacking generalizability and fail drastically when applied to real-world data. This study proposes a novel method for generalizable spoofing detection leveraging non-semantic universal audio representations. Extensive experiments have been performed to find suitable non-semantic features using TRILL and TRILLsson models. The results indicate that the proposed method achieves comparable performance on the in-domain test set while significantly outperforming state-of-the-art approaches on out-of-domain test sets. Notably, it demonstrates superior generalization on public-domain data, surpassing methods based on hand-crafted features, semantic embeddings, and end-to-end architectures. 

**Abstract (ZH)**: 快速发展的生成建模使得合成音频生成变得容易，从而使得基于语音的服务面临着欺诈攻击的威胁。因此，迫切需要比以往更加 robust 的countermeasures。现有的深度fake检测解决方案往往因为缺乏泛化能力而在现实世界数据上表现不佳。本研究提出了一种利用非语义通用音频表示进行泛化欺诈检测的新方法。通过使用TRILL和TRILLsson模型，进行了广泛实验以找到合适的非语义特征。结果表明，所提出的方法在领域内测试集上达到了可比的性能，并在领域外测试集上大幅超越了最先进的方法。值得注意的是，该方法在公共数据上的泛化能力优于基于手工特征、语义嵌入和端到端架构的方法。 

---
# What Are Research Hypotheses? 

**Title (ZH)**: 什么是研究假设？ 

**Authors**: Jian Wu, Sarah Rajtmajer  

**Link**: [PDF](https://arxiv.org/pdf/2509.00185)  

**Abstract**: Over the past decades, alongside advancements in natural language processing, significant attention has been paid to training models to automatically extract, understand, test, and generate hypotheses in open and scientific domains. However, interpretations of the term \emph{hypothesis} for various natural language understanding (NLU) tasks have migrated from traditional definitions in the natural, social, and formal sciences. Even within NLU, we observe differences defining hypotheses across literature. In this paper, we overview and delineate various definitions of hypothesis. Especially, we discern the nuances of definitions across recently published NLU tasks. We highlight the importance of well-structured and well-defined hypotheses, particularly as we move toward a machine-interpretable scholarly record. 

**Abstract (ZH)**: 在过去几十年中，随着自然语言处理技术的不断进步，人们对训练模型自动在开放和科学领域中提取、理解、测试和生成假设给予了广泛关注。然而，各种自然语言理解（NLU）任务中对“假设”一词的解释已从自然、社会和形式科学的传统定义中迁移开来。即使在NLU领域内，我们也能观察到不同文献中对假设的定义存在差异。本文概述并区分了各种假设的定义，尤其是近年来发表的NLU任务中不同定义的细微差别。我们强调了结构良好且定义明确的假设的重要性，特别是当我们向可机器解析的学术记录过渡时。 

---
# Principled Approximation Methods for Efficient and Scalable Deep Learning 

**Title (ZH)**: 原理性的近似方法以实现高效可扩展的深度学习 

**Authors**: Pedro Savarese  

**Link**: [PDF](https://arxiv.org/pdf/2509.00174)  

**Abstract**: Recent progress in deep learning has been driven by increasingly larger models. However, their computational and energy demands have grown proportionally, creating significant barriers to their deployment and to a wider adoption of deep learning technologies. This thesis investigates principled approximation methods for improving the efficiency of deep learning systems, with a particular focus on settings that involve discrete constraints and non-differentiability.
We study three main approaches toward improved efficiency: architecture design, model compression, and optimization. For model compression, we propose novel approximations for pruning and quantization that frame the underlying discrete problem as continuous and differentiable, enabling gradient-based training of compression schemes alongside the model's parameters. These approximations allow for fine-grained sparsity and precision configurations, leading to highly compact models without significant fine-tuning. In the context of architecture design, we design an algorithm for neural architecture search that leverages parameter sharing across layers to efficiently explore implicitly recurrent architectures. Finally, we study adaptive optimization, revisiting theoretical properties of widely used methods and proposing an adaptive optimizer that allows for quick hyperparameter tuning.
Our contributions center on tackling computationally hard problems via scalable and principled approximations. Experimental results on image classification, language modeling, and generative modeling tasks show that the proposed methods provide significant improvements in terms of training and inference efficiency while maintaining, or even improving, the model's performance. 

**Abstract (ZH)**: Recent进展的深度学习得益于越来越大的模型规模。然而，它们的计算和能源需求也成比例增长，为部署和更广泛采用深度学习技术带来了巨大障碍。本论文探讨了改进深度学习系统效率的原理性近似方法，特别关注涉及离散约束和非连续性的场景。 

---
# Pilot Study on Generative AI and Critical Thinking in Higher Education Classrooms 

**Title (ZH)**: 生成式人工智能与批判性思维在高等教育课堂中的试点研究 

**Authors**: W. F. Lamberti, S. R. Lawrence, D. White, S. Kim, S. Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2509.00167)  

**Abstract**: Generative AI (GAI) tools have seen rapid adoption in educational settings, yet their role in fostering critical thinking remains underexplored. While previous studies have examined GAI as a tutor for specific lessons or as a tool for completing assignments, few have addressed how students critically evaluate the accuracy and appropriateness of GAI-generated responses. This pilot study investigates students' ability to apply structured critical thinking when assessing Generative AI outputs in introductory Computational and Data Science courses. Given that GAI tools often produce contextually flawed or factually incorrect answers, we designed learning activities that require students to analyze, critique, and revise AI-generated solutions. Our findings offer initial insights into students' ability to engage critically with GAI content and lay the groundwork for more comprehensive studies in future semesters. 

**Abstract (ZH)**: 生成式人工智能工具在教育领域的快速 adoption 尚未充分探讨其在培育批判性思维方面的作用。尽管之前的研究已经考察了生成式人工智能作为特定课程的 tutor 或完成作业的工具，但鲜有研究关注学生如何批判性地评估生成式人工智能生成的响应的准确性和适宜性。本试点研究旨在探讨学生在入门级计算与数据科学课程中评估生成式人工智能输出时应用结构化批判性思维的能力。鉴于生成式人工智能工具常常产生上下文不符或事实错误的答案，我们设计了要求学生分析、批判和修订生成式人工智能生成的解决方案的学习活动。我们的研究结果提供了关于学生批判性处理生成式人工智能内容能力的初步见解，并为未来学期开展更全面的研究奠定了基础。 

---
# Scaling Legal AI: Benchmarking Mamba and Transformers for Statutory Classification and Case Law Retrieval 

**Title (ZH)**: 扩展法律AI：Mamba和变换器在法律法规分类和案例检索中的基准测试 

**Authors**: Anuraj Maurya  

**Link**: [PDF](https://arxiv.org/pdf/2509.00141)  

**Abstract**: The rapid growth of statutory corpora and judicial decisions requires scalable legal AI systems capable of classification and retrieval over extremely long contexts. Transformer-based architectures (e.g., Longformer, DeBERTa) dominate current legal NLP benchmarks but struggle with quadratic attention costs, limiting efficiency and scalability. In this work, we present the first comprehensive benchmarking of Mamba, a state-space model (SSM) with linear-time selective mechanisms, against leading transformer models for statutory classification and case law retrieval. We evaluate models on open-source legal corpora including LexGLUE, EUR-Lex, and ILDC, covering statutory tagging, judicial outcome prediction, and case retrieval tasks. Metrics include accuracy, recall at k, mean reciprocal rank (MRR), and normalized discounted cumulative gain (nDCG), alongside throughput measured in tokens per second and maximum context length. Results show that Mamba's linear scaling enables processing of legal documents several times longer than transformers, while maintaining or surpassing retrieval and classification performance. This study introduces a new legal NLP benchmark suite for long-context modeling, along with open-source code and datasets to support reproducibility. Our findings highlight trade-offs between state-space models and transformers, providing guidance for deploying scalable legal AI in statutory analysis, judicial decision support, and policy research. 

**Abstract (ZH)**: 大规模法定语料库和司法决策的快速增长需要能够处理极长上下文的可扩展法律AI系统，用于分类和检索。基于Transformer的架构（如Longformer、DeBERTa）目前主导着法律NLP基准测试，但在注意力机制成本上面临平方级增长的挑战，限制了效率和可扩展性。在本研究中，我们首次全面评估了Mamba模型——一种状态空间模型（SSM），其具有线性时间的选择性机制——与领先Transformer模型在法定分类和案例法检索任务中的性能。我们使用LexGLUE、EUR-Lex和ILDC等开源法律语料库进行模型评价，涵盖法定标注、司法结果预测和案例检索任务。评价指标包括准确性、前k召回率、归一化倒数平均排名（MRR）和归一化折扣累积增益（nDCG），同时还包括吞吐量（每秒处理的令牌数）和最大上下文长度。结果显示，Mamba的线性扩展使其能够处理的法律文件长度远超Transformer，同时保持或超越检索和分类性能。本研究引入了一套新的长上下文建模法律NLP基准测试套件，并提供了开源代码和数据集以支持可重复性。我们的研究结果突显了状态空间模型和Transformer之间的权衡，为部署可扩展法律AI在法定分析、司法决策支持和政策研究中的应用提供了指导。 

---
# The Application of Virtual Environments and Artificial Intelligence in Higher Education: Experimental Findings in Philosophy Teaching 

**Title (ZH)**: 虚拟环境和人工智能在高等教育中的应用：哲学教学的实验研究 

**Authors**: Adel Vehrer, Zsolt Palfalusi  

**Link**: [PDF](https://arxiv.org/pdf/2509.00110)  

**Abstract**: This study explores how virtual environments and artificial intelligence can enhance university students' learning experiences, with particular attention to the digital preferences of Generation Z. An experiment was conducted at the Faculty of Pedagogy, Humanities, and Social Sciences at University of Gyor, where Walter's Cube technology and a trained AI mediator were integrated into the instruction of ten philosophical topics. The curriculum was aligned with the official syllabus and enriched with visual content, quotations, and explanatory texts related to iconic figures in philosophy. A total of 77 first-year undergraduate students from full-time humanities and social sciences programs participated in the study. Following their end-of-semester offline written examination, students voluntarily completed a paper-based, anonymous ten-question test and provided feedback on the method's effectiveness. No sensitive personal data were collected, and the research was conducted with formal approval from the Faculty Dean. Descriptive statistics and inferential tests were applied to evaluate the impact of the virtual environment and AI mediation on learning outcomes. Results indicate that 80 percent of participants achieved good or excellent final exam grades, and the majority rated the virtual material as highly effective. Qualitative feedback emphasized increased motivation and deeper engagement, attributed to the immersive 3D presentation and interactive AI support. This research contributes to the advancement of digital pedagogy and suggests new directions for applying virtual and AI-based methods in higher education, particularly in disciplines where abstract reasoning and conceptual understanding are central. 

**Abstract (ZH)**: 本研究探讨了虚拟环境和人工智能如何增强大学学生的学习体验，特别关注Z世代的数字偏好。该研究在格伊尔大学教育、人文与社会科学学院进行，将沃尔特立方体技术和训练有素的AI调解人整合到十个哲学主题的教学中。课程内容与官方教学大纲一致，并补充了与哲学标志性人物相关的视觉内容、引言和解释性文本。共有77名全日制人文与社会科学专业的全日制一年级学生参与了此项研究。在学期末的线下书面考试结束后，学生们自愿完成了匿名的十题试卷测试并提供了对该方法有效性的反馈。研究过程中未收集任何敏感个人数据，并经过学院院长正式批准。研究运用描述性统计和推断性测试评估了虚拟环境和AI调解对学生学习成果的影响。结果表明，80%的参与者取得了良好或优秀的期末考试成绩，大多数学生认为虚拟材料极为有效。定性反馈强调了沉浸式3D呈现和交互式AI支持提高了学生的学习动机和参与度。本研究为数字教学方法的发展做出了贡献，并为在高等教育中应用虚拟和基于AI的方法指出了新的方向，特别是在那些依赖抽象推理和概念理解的学科中。 

---
# Exploiting a Mixture-of-Layers in an Electrocardiography Foundation Model 

**Title (ZH)**: 利用心电图基础模型中的多层混合结构 

**Authors**: Phu X. Nguyen, Huy Phan, Hieu Pham, Christos Chatzichristos, Bert Vandenberk, Maarten De Vos  

**Link**: [PDF](https://arxiv.org/pdf/2509.00102)  

**Abstract**: Transformer-based foundation models for Electrocardiograms (ECGs) have recently achieved impressive performance in many downstream applications. However, the internal representations of such models across layers have not been fully understood and exploited. An important question arises: Does the final layer of the pre-trained Transformer model, the \emph{de facto} representational layer, provide optimal performance for downstream tasks? Although our answer based on empirical and theoretical analyses for this question is negative, we propose a novel approach to leverage the representation diversity of the model's layers effectively. Specifically, we introduce a novel architecture called Post-pretraining Mixture-of-layers Aggregation (PMA), which enables a flexible combination of the layer-wise representations from the layer stack of a Transformer-based foundation model. We first pre-train the model from ECG signals using the 1-dimensional Vision Transformer (ViT) via masked modeling. In downstream applications, instead of relying solely on the last layer of the model, we employ a gating network to selectively fuse the representations from the pretrained model's layers, thereby enhancing representation power and improving performance of the downstream applications. In addition, we extend the proposed method to the pretraining stage by aggregating all representations through group-wise averaging before feeding them into the decoder-based Transformer. 

**Abstract (ZH)**: 基于Transformer的基础模型在心电图（ECGs）中的应用：一种后训练混合层聚合的新方法 

---
# Automatic Pronunciation Error Detection and Correction of the Holy Quran's Learners Using Deep Learning 

**Title (ZH)**: 使用深度学习进行Holy Quran学习者发音错误检测与修正 

**Authors**: Abdullah Abdelfattah, Mahmoud I. Khalil, Hazem Abbas  

**Link**: [PDF](https://arxiv.org/pdf/2509.00094)  

**Abstract**: Assessing spoken language is challenging, and quantifying pronunciation metrics for machine learning models is even harder. However, for the Holy Quran, this task is simplified by the rigorous recitation rules (tajweed) established by Muslim scholars, enabling highly effective assessment. Despite this advantage, the scarcity of high-quality annotated data remains a significant barrier.
In this work, we bridge these gaps by introducing: (1) A 98% automated pipeline to produce high-quality Quranic datasets -- encompassing: Collection of recitations from expert reciters, Segmentation at pause points (waqf) using our fine-tuned wav2vec2-BERT model, Transcription of segments, Transcript verification via our novel Tasmeea algorithm; (2) 850+ hours of audio (~300K annotated utterances); (3) A novel ASR-based approach for pronunciation error detection, utilizing our custom Quran Phonetic Script (QPS) to encode Tajweed rules (unlike the IPA standard for Modern Standard Arabic). QPS uses a two-level script: (Phoneme level): Encodes Arabic letters with short/long vowels. (Sifa level): Encodes articulation characteristics of every phoneme. We further include comprehensive modeling with our novel multi-level CTC Model which achieved 0.16% average Phoneme Error Rate (PER) on the testset. We release all code, data, and models as open-source: this https URL 

**Abstract (ZH)**: 评估口头语言是一项挑战，而量化发音指标以供机器学习模型使用则更加困难。然而，对于《古兰经》而言，由于穆斯林学者制定了严格的诵读规则（ Tajweed），使得这一任务得以简化，从而可以进行高效评估。尽管存在这一优势，高质量标注数据的匮乏仍然是一个重大障碍。

在这项工作中，我们通过引入以下内容来弥补上述差距：(1) 一个98%自动化的流水线，用于生成高质量的《古兰经》数据集——包括：专家诵读者录音的采集，使用我们微调后的wav2vec2-BERT模型在停顿点（Waqf）处进行分割，对片段进行转录，通过我们新颖的Tasmeea算法进行转录验证；(2) 850多个小时的音频数据（约30万个标注的语句）；(3) 一种基于ASR的新颖发音错误检测方法，利用我们自定义的《古兰经音位脚本》（QPS）来编码Tajweed规则（不同于现代标准阿拉伯语的IPA标准）。QPS采用两级脚本：(音位级别) 编码阿拉伯字母及其短/长元音；(特征级别) 编码每个音位的发音特性。我们还结合使用了我们新颖的多级CTC模型，该模型在测试集上的音位错误率（PER）达到0.16%。我们已将所有代码、数据和模型公开发布：https://github.com/your-repo-name 

---
# Yet Unnoticed in LSTM: Binary Tree Based Input Reordering, Weight Regularization, and Gate Nonlinearization 

**Title (ZH)**: 基于二叉树的输入重排、权重正则化和门非线性化之于LSTM的尚未注意之处 

**Authors**: Mojtaba Moattari  

**Link**: [PDF](https://arxiv.org/pdf/2509.00087)  

**Abstract**: LSTM models used in current Machine Learning literature and applications, has a promising solution for permitting long term information using gating mechanisms that forget and reduce effect of current input information. However, even with this pipeline, they do not optimally focus on specific old index or long-term information. This paper elaborates upon input reordering approaches to prioritize certain input indices. Moreover, no LSTM based approach is found in the literature that examines weight normalization while choosing the right weight and exponent of Lp norms through main supervised loss function. In this paper, we find out which norm best finds relationship between weights to either smooth or sparsify them. Lastly, gates, as weighted representations of inputs and states, which control reduction-extent of current input versus previous inputs (~ state), are not nonlinearized enough (through a small FFNN). As analogous to attention mechanisms, gates easily filter current information to bold (emphasize on) past inputs. Nonlinearized gates can more easily tune up to peculiar nonlinearities of specific input in the past. This type of nonlinearization is not proposed in the literature, to the best of author's knowledge. The proposed approaches are implemented and compared with a simple LSTM to understand their performance in text classification tasks. The results show they improve accuracy of LSTM. 

**Abstract (ZH)**: 当前机器学习文献和应用中使用的LSTM模型通过门控机制允许长期信息的保留，具有潜在的有效解决方案。然而，即使如此，它们也不最优地聚焦于特定的旧索引或长期信息。本文探讨了输入重排序方法以优先处理某些输入索引。此外，文献中没有基于LSTM的方法在选择合适的Lp范数的权值及其指数时同时考虑权重规范化。本文旨在找出最适合找到权重间关系的范数，以实现平滑或稀疏化权重。最后，作为输入和状态的加权表示，门控机制控制当前输入相对于先前输入（状态）的减少程度，但通过小型全连接神经网络进行的非线性化程度不够充分。与注意机制类似，门控机制可以轻松地过滤当前信息，强调过去的输入。通过非线性化门控机制可以更好地调适至过去特定输入的特殊非线性。此类非线性化在现有文献中未有提出，据作者所知。提出的办法已在文本分类任务中实现并与其他简单LSTM进行比较，结果表明它们提高了LSTM的准确性。 

---
# Private, Verifiable, and Auditable AI Systems 

**Title (ZH)**: 私有、可验证且可审计的AI系统 

**Authors**: Tobin South  

**Link**: [PDF](https://arxiv.org/pdf/2509.00085)  

**Abstract**: The growing societal reliance on artificial intelligence necessitates robust frameworks for ensuring its security, accountability, and trustworthiness. This thesis addresses the complex interplay between privacy, verifiability, and auditability in modern AI, particularly in foundation models. It argues that technical solutions that integrate these elements are critical for responsible AI innovation. Drawing from international policy contributions and technical research to identify key risks in the AI pipeline, this work introduces novel technical solutions for critical privacy and verifiability challenges. Specifically, the research introduces techniques for enabling verifiable and auditable claims about AI systems using zero-knowledge cryptography; utilizing secure multi-party computation and trusted execution environments for auditable, confidential deployment of large language models and information retrieval; and implementing enhanced delegation mechanisms, credentialing systems, and access controls to secure interactions with autonomous and multi-agent AI systems. Synthesizing these technical advancements, this dissertation presents a cohesive perspective on balancing privacy, verifiability, and auditability in foundation model-based AI systems, offering practical blueprints for system designers and informing policy discussions on AI safety and governance. 

**Abstract (ZH)**: 不断增长的社会对人工智能的依赖 necessitates 强大的框架以确保其安全、问责制和可信度。本论文探讨了现代人工智能，特别是基础模型中隐私、可验证性和审计性之间的复杂相互作用。它认为，将这些要素整合到技术解决方案中是负责任的人工智能创新的关键。通过借鉴国际政策贡献和技术研究来识别人工智能管道中的关键风险，本研究引入了针对关键隐私和可验证性挑战的新型技术解决方案。具体而言，研究介绍了使用零知识密码学来实现对人工智能系统的可验证和审计声明的技术；利用安全多方计算和受信任执行环境来实现大型语言模型和信息检索的可审计和保密部署；并实施增强委托机制、认证系统和访问控制以保护与自主和多代理人工智能系统的交互。综合这些技术进步，本博士论文提出了基础模型驱动的人工智能系统中平衡隐私、可验证性和审计性的统一视角，提供了系统的设计师可参考的实际蓝图，并为人工智能安全和治理的政策讨论提供了信息。 

---
# Data Cartography for Detecting Memorization Hotspots and Guiding Data Interventions in Generative Models 

**Title (ZH)**: 生成模型中检测记忆热点和指导数据干预的数据制图方法 

**Authors**: Laksh Patel, Neel Shanbhag  

**Link**: [PDF](https://arxiv.org/pdf/2509.00083)  

**Abstract**: Modern generative models risk overfitting and unintentionally memorizing rare training examples, which can be extracted by adversaries or inflate benchmark performance. We propose Generative Data Cartography (GenDataCarto), a data-centric framework that assigns each pretraining sample a difficulty score (early-epoch loss) and a memorization score (frequency of ``forget events''), then partitions examples into four quadrants to guide targeted pruning and up-/down-weighting. We prove that our memorization score lower-bounds classical influence under smoothness assumptions and that down-weighting high-memorization hotspots provably decreases the generalization gap via uniform stability bounds. Empirically, GenDataCarto reduces synthetic canary extraction success by over 40\% at just 10\% data pruning, while increasing validation perplexity by less than 0.5\%. These results demonstrate that principled data interventions can dramatically mitigate leakage with minimal cost to generative performance. 

**Abstract (ZH)**: 基于数据的生成模型-cartography（GenDataCarto）：通过数据导向的方法减轻过拟合和记忆效应 

---
# SynCircuit: Automated Generation of New Synthetic RTL Circuits Can Enable Big Data in Circuits 

**Title (ZH)**: SynCircuit: 自动生成新型合成RTL电路可以实现电路中的大数据 

**Authors**: Shang Liu, Jing Wang, Wenji Fang, Zhiyao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2509.00071)  

**Abstract**: In recent years, AI-assisted IC design methods have demonstrated great potential, but the availability of circuit design data is extremely limited, especially in the public domain. The lack of circuit data has become the primary bottleneck in developing AI-assisted IC design methods. In this work, we make the first attempt, SynCircuit, to generate new synthetic circuits with valid functionalities in the HDL format. SynCircuit automatically generates synthetic data using a framework with three innovative steps: 1) We propose a customized diffusion-based generative model to resolve the Directed Cyclic Graph (DCG) generation task, which has not been well explored in the AI community. 2) To ensure our circuit is valid, we enforce the circuit constraints by refining the initial graph generation outputs. 3) The Monte Carlo tree search (MCTS) method further optimizes the logic redundancy in the generated graph. Experimental results demonstrate that our proposed SynCircuit can generate more realistic synthetic circuits and enhance ML model performance in downstream circuit design tasks. 

**Abstract (ZH)**: 近年来，AI辅助的IC设计方法展现了巨大的潜力，但由于电路设计数据的可用性极其有限，特别是在公开领域，电路数据的缺失已成为开发AI辅助IC设计方法的主要瓶颈。在这项工作中，我们首次尝试使用SynCircuit生成具有有效功能的HDL格式的新合成电路。SynCircuit通过一个包含三个创新步骤的框架自动生成合成数据：1) 我们提出了一种定制的扩散生成模型来解决尚未在AI社区充分探索的定向循环图（DCG）生成任务。2) 为了确保电路的有效性，我们通过细化初始图生成输出来应用电路约束。3) 进一步通过蒙特卡洛树搜索（MCTS）方法优化生成图中的逻辑冗余。实验结果表明，我们提出的SynCircuit能够生成更具现实感的合成电路，并在下游电路设计任务中增强机器学习模型的性能。 

---
# The Collaborations among Healthcare Systems, Research Institutions, and Industry on Artificial Intelligence Research and Development 

**Title (ZH)**: 医疗卫生系统、研究机构与industry在人工智能研究与开发中的合作 

**Authors**: Jiancheng Ye, Michelle Ma, Malak Abuhashish  

**Link**: [PDF](https://arxiv.org/pdf/2509.00068)  

**Abstract**: Objectives: The integration of Artificial Intelligence (AI) in healthcare promises to revolutionize patient care, diagnostics, and treatment protocols. Collaborative efforts among healthcare systems, research institutions, and industry are pivotal to leveraging AI's full potential. This study aims to characterize collaborative networks and stakeholders in AI healthcare initiatives, identify challenges and opportunities within these collaborations, and elucidate priorities for future AI research and development. Methods: This study utilized data from the Chinese Society of Radiology and the Chinese Medical Imaging AI Innovation Alliance. A national cross-sectional survey was conducted in China (N = 5,142) across 31 provincial administrative regions, involving participants from three key groups: clinicians, institution professionals, and industry representatives. The survey explored diverse aspects including current AI usage in healthcare, collaboration dynamics, challenges encountered, and research and development priorities. Results: Findings reveal high interest in AI among clinicians, with a significant gap between interest and actual engagement in development activities. Despite the willingness to share data, progress is hindered by concerns about data privacy and security, and lack of clear industry standards and legal guidelines. Future development interests focus on lesion screening, disease diagnosis, and enhancing clinical workflows. Conclusion: This study highlights an enthusiastic yet cautious approach toward AI in healthcare, characterized by significant barriers that impede effective collaboration and implementation. Recommendations emphasize the need for AI-specific education and training, secure data-sharing frameworks, establishment of clear industry standards, and formation of dedicated AI research departments. 

**Abstract (ZH)**: 研究目标：将人工智能（AI）融入医疗保健有望革新患者护理、诊断和治疗方案。医疗卫生系统、研究机构和产业之间的协作对于充分利用AI的潜力至关重要。本研究旨在表征AI医疗保健倡议中的协作网络和利益相关者，识别这些合作中的挑战和机遇，并阐明未来AI研究和开发的优先事项。方法：本研究利用中国放射学会和中国医疗影像AI创新联盟的数据。在中国31个省级行政区开展了全国横断面调查（N = 5,142），参与者来自三类关键群体：临床医生、机构专业人士和产业代表。调查涵盖了当前AI在医疗保健中的应用、合作动态、遇到的挑战以及研究和开发优先事项等多个方面。结果：研究发现，临床医生对AI有浓厚兴趣，但兴趣与实际参与开发活动之间存在显著差距。尽管有意愿分享数据，但数据隐私和安全等方面的担忧以及缺乏明确的行业标准和法律指导阻碍了进展。未来发展的兴趣集中在病灶筛查、疾病诊断和优化临床工作流程。结论：本研究揭示了对AI在医疗保健中既充满热情又持谨慎态度的态度，存在阻碍有效协作和实施的显著障碍。建议强调需要针对AI的专业教育和培训、安全的数据共享框架、制定明确的行业标准以及成立专注于AI研究的部门。 

---
# From Data to Decision: A Multi-Stage Framework for Class Imbalance Mitigation in Optical Network Failure Analysis 

**Title (ZH)**: 从数据到决策：光网络故障分析中类别不平衡缓解的多阶段框架 

**Authors**: Yousuf Moiz Ali, Jaroslaw E. Prilepsky, Nicola Sambo, Joao Pedro, Mohammad M. Hosseini, Antonio Napoli, Sergei K. Turitsyn, Pedro Freire  

**Link**: [PDF](https://arxiv.org/pdf/2509.00057)  

**Abstract**: Machine learning-based failure management in optical networks has gained significant attention in recent years. However, severe class imbalance, where normal instances vastly outnumber failure cases, remains a considerable challenge. While pre- and in-processing techniques have been widely studied, post-processing methods are largely unexplored. In this work, we present a direct comparison of pre-, in-, and post-processing approaches for class imbalance mitigation in failure detection and identification using an experimental dataset. For failure detection, post-processing methods-particularly Threshold Adjustment-achieve the highest F1 score improvement (up to 15.3%), while Random Under-Sampling provides the fastest inference. In failure identification, GenAI methods deliver the most substantial performance gains (up to 24.2%), whereas post-processing shows limited impact in multi-class settings. When class overlap is present and latency is critical, over-sampling methods such as the SMOTE are most effective; without latency constraints, Meta-Learning yields the best results. In low-overlap scenarios, Generative AI approaches provide the highest performance with minimal inference time. 

**Abstract (ZH)**: 基于机器学习的光网络故障管理中，近年来获得了广泛关注。然而，严重的类别不平衡问题，即正常实例远多于故障实例，仍然是一个巨大挑战。尽管预处理和处理中方法已经广泛研究，但处理后方法尚未得到充分探索。在本文中，我们通过实验数据集直接比较了预处理、处理中和处理后方法在故障检测和识别中的类别不平衡缓解效果。在故障检测中，处理后方法尤其是阈值调整方法实现最高的F1分数提升（最高可达15.3%），而随机下采样提供最快的推理速度。在故障识别中，生成式AI方法提供了最大的性能提升（最高可达24.2%），而在多类设置下，处理后方法的影响有限。当存在类别重叠且延迟要求严格时，过采样方法如SMOTE最有效；在没有延迟约束的情况下，元学习方法表现最佳。在类别重叠较低的情况下，生成式AI方法提供了最高性能，同时具有最小的推理时间。 

---
# Applying Deep Learning to Anomaly Detection of Russian Satellite Activity for Indications Prior to Military Activity 

**Title (ZH)**: 将深度学习应用于俄罗斯卫星活动异常检测以预测军事活动迹象 

**Authors**: David Kurtenbach, Megan Manly, Zach Metzinger  

**Link**: [PDF](https://arxiv.org/pdf/2509.00050)  

**Abstract**: We apply deep learning techniques for anomaly detection to analyze activity of Russian-owned resident space objects (RSO) prior to the Ukraine invasion and assess the results for any findings that can be used as indications and warnings (I&W) of aggressive military behavior for future conflicts. Through analysis of anomalous activity, an understanding of possible tactics and procedures can be established to assess the existence of statistically significant changes in Russian RSO pattern of life/pattern of behavior (PoL/PoB) using publicly available two-line element (TLE) data. This research looks at statistical and deep learning approaches to assess anomalous activity. The deep learning methods assessed are isolation forest (IF), traditional autoencoder (AE), variational autoencoder (VAE), Kolmogorov Arnold Network (KAN), and a novel anchor-loss based autoencoder (Anchor AE). Each model is used to establish a baseline of on-orbit activity based on a five-year data sample. The primary investigation period focuses on the six months leading up to the invasion date of February 24, 2022. Additional analysis looks at RSO activity during an active combat period by sampling TLE data after the invasion date. The deep learning autoencoder models identify anomalies based on reconstruction errors that surpass a threshold sigma. To capture the nuance and unique characteristics of each RSO an individual model was trained for each observed space object. The research made an effort to prioritize explainability and interpretability of the model results thus each observation was assessed for anomalous behavior of the individual six orbital elements versus analyzing the input data as a single monolithic observation. The results demonstrate not only statistically significant anomalies of Russian RSO activity but also details anomalous findings to the individual orbital element. 

**Abstract (ZH)**: 我们应用深度学习技术进行异常检测，分析俄罗斯拥有并在轨空间物体（RSO）在乌克兰入侵前的活动，并评估任何可用于未来冲突中识别和预警敌对军事行为的迹象和警告（I&W）。通过分析异常活动，建立可能 tactics 和程序的理解，以使用公开的两行元素（TLE）数据评估俄罗斯RSO模式 of 生命/模式 of 行为（PoL/PoB）中的统计上显著变化。该研究探讨了统计和深度学习方法来评估异常活动。评估的深度学习方法包括孤立森林（IF）、传统自编码器（AE）、 variational 自编码器（VAE）、柯尔莫哥洛夫-阿诺尔德网络（KAN）以及一种基于锚损失的自编码器（Anchor AE）。每种模型都基于五年数据样本建立了在轨活动的基础。主要调查期集中在2022年2月24日前的六个月。此外，通过入侵日期后的TLE数据采样分析在役期间的RSO活动。深度学习自编码器模型基于重建误差超过阈值sigma来识别异常。为了捕捉每颗RSO的细微差别和独特特性，为每颗观测到的太空物体训练了个体模型。该研究力求优先考虑模型结果的可解释性和可理解性，因此，每项观测都被评估了单个六轨道元素的异常行为，而不是作为单一的数据观察进行分析。结果不仅展示了俄罗斯RSO活动的统计显著异常，还对每个轨道元素的具体异常发现进行了详细说明。 

---
# Teaching AI to Remember: Insights from Brain-Inspired Replay in Continual Learning 

**Title (ZH)**: 教学机器记住：脑启发的连续学习中再播放机制的见解 

**Authors**: Jina Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.00047)  

**Abstract**: Artificial neural networks (ANNs) continue to face challenges in continual learning, particularly due to catastrophic forgetting, the loss of previously learned knowledge when acquiring new tasks. Inspired by memory consolidation in the human brain, we investigate the internal replay mechanism proposed by~\citep{brain_inspired_replay1}, which reactivates latent representations of prior experiences during learning. As internal replay was identified as the most influential component among the brain-inspired mechanisms in their framework, it serves as the central focus of our in-depth investigation. Using the CIFAR-100 dataset in a class-incremental setting, we evaluate the effectiveness of internal replay, both in isolation and in combination with Synaptic Intelligence (SI). Our experiments show that internal replay significantly mitigates forgetting, especially when paired with SI, but at the cost of reduced initial task accuracy, highlighting a trade-off between memory stability and learning plasticity. Further analyses using log-likelihood distributions, reconstruction errors, silhouette scores, and UMAP projections reveal that internal replay increases representational overlap in latent space, potentially limiting task-specific differentiation. These results underscore the limitations of current brain-inspired methods and suggest future directions for balancing retention and adaptability in continual learning systems. 

**Abstract (ZH)**: 人工神经网络在持续学习中仍面临挑战，特别是在 catastrophic 忘记方面，即在获取新任务时会丢失之前学习的知识。受人类大脑记忆巩固机制的启发，我们研究了文献~\citep{brain_inspired_replay1} 提出的内部回放机制，该机制在学习过程中重新激活先前经验的潜在表示。由于在他们的框架中，内部回放被确定为最具影响力的机制之一，因此它是我们深入研究的中心焦点。我们使用 CIFAR-100 数据集在类增量设置下评估内部回放的效果，单独使用或与 Synaptic Intelligence (SI) 结合使用。实验结果表明，内部回放显著减轻了遗忘，尤其是在与 SI 结合使用时，但代价是初始任务准确率的降低，突显了记忆稳定性和学习可塑性之间的权衡。进一步使用对数似然分布、重构误差、轮廓系数和 UMAP 投影进行的分析揭示内部回放增加了潜在空间中的表示重叠，可能限制了任务特异性差异。这些结果突显了当前受脑启发方法的局限性，并建议未来在持续学习系统中平衡保留和适应性的方向。 

---
# Transfer Learning for Minimum Operating Voltage Prediction in Advanced Technology Nodes: Leveraging Legacy Data and Silicon Odometer Sensing 

**Title (ZH)**: 基于转移学习的先进工艺节点最低操作电压预测：利用legacy数据和硅里程计感知 

**Authors**: Yuxuan Yin, Rebecca Chen, Boxun Xu, Chen He, Peng Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.00035)  

**Abstract**: Accurate prediction of chip performance is critical for ensuring energy efficiency and reliability in semiconductor manufacturing. However, developing minimum operating voltage ($V_{min}$) prediction models at advanced technology nodes is challenging due to limited training data and the complex relationship between process variations and $V_{min}$. To address these issues, we propose a novel transfer learning framework that leverages abundant legacy data from the 16nm technology node to enable accurate $V_{min}$ prediction at the advanced 5nm node. A key innovation of our approach is the integration of input features derived from on-chip silicon odometer sensor data, which provide fine-grained characterization of localized process variations -- an essential factor at the 5nm node -- resulting in significantly improved prediction accuracy. 

**Abstract (ZH)**: 准确预测芯片性能对于确保半导体制造中的能效和可靠性至关重要。然而，在先进工艺节点开发最小工作电压($V_{min}$)预测模型具有挑战性，因为训练数据有限且工艺波动与$V_{min}$之间的关系复杂。为解决这些问题，我们提出了一种新颖的迁移学习框架，利用来自16nm工艺节点的丰富遗留数据，以在先进的5nm节点实现准确的$V_{min}$预测。我们方法的关键创新之处在于整合了源自芯片上硅里程计传感器的数据输入特征，这些特征提供了局部工艺波动的精细表征——这是5nm节点的一个重要因素，从而显著提高了预测准确性。 

---
# DeepEmoNet: Building Machine Learning Models for Automatic Emotion Recognition in Human Speeches 

**Title (ZH)**: DeepEmoNet：构建自动语音情感识别的机器学习模型 

**Authors**: Tai Vu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00025)  

**Abstract**: Speech emotion recognition (SER) has been a challenging problem in spoken language processing research, because it is unclear how human emotions are connected to various components of sounds such as pitch, loudness, and energy. This paper aims to tackle this problem using machine learning. Particularly, we built several machine learning models using SVMs, LTSMs, and CNNs to classify emotions in human speeches. In addition, by leveraging transfer learning and data augmentation, we efficiently trained our models to attain decent performances on a relatively small dataset. Our best model was a ResNet34 network, which achieved an accuracy of $66.7\%$ and an F1 score of $0.631$. 

**Abstract (ZH)**: 基于机器学习的语音情绪识别研究 

---
# A Fluid Antenna Enabled Physical Layer Key Generation for Next-G Wireless Networks 

**Title (ZH)**: 基于流体天线的物理层密钥生成技术用于下一代无线网络 

**Authors**: Jiacheng Guo, Ning Gao, Yiping Zuo, Hao Xu, Shi Jin, Kai Kit Wong  

**Link**: [PDF](https://arxiv.org/pdf/2509.00018)  

**Abstract**: As a promising physical layer security technique, physical layer key generation (PLKG) enables legitimate users to obtain secret keys from wireless channel without security infrastructures. However, in harsh propagation environments, the channel characteristic becomes unsatisfactory, the key generation rate (KGR) is significantly deteriorated. In this paper, we propose a novel fluid antenna (FA) enabled PLKG system to address this challenge. Specifically, we first derive the closed-form expression of the KGR for FA array, and then jointly optimize the precoding matrix and the antenna positions via a particle swarm optimization (PSO) algorithm. Next, to further reduce the computational complexity of the optimization procedure, we develop an alternating optimization (AO) algorithm, which combines the projected gradient descent (PGD) and the PSO. Simulation results demonstrate that by exploiting the additional spatial degree of freedom (DoF), our FA enabled PLKG system is superior to the benchmarks, such as the conventional fixed-position antenna (FPA) array and the reconfigurable intelligent surface (RIS). It is worth highlighting that compared to the conventional uniform planar antenna (UPA), the FA enabled PLKG achieves a 35.42\% KGR performance improvement under PSO algorithm and a 67.73\% KGR performance improvement under AO algorithm, respectively. 

**Abstract (ZH)**: 基于流体天线的物理层密钥生成系统 

---
# Optimized Renewable Energy Planning MDP for Socially-Equitable Electricity Coverage in the US 

**Title (ZH)**: 美国社会公正的可再生能源规划MDP模型优化研究 

**Authors**: Riya Kinnarkar, Mansur Arief  

**Link**: [PDF](https://arxiv.org/pdf/2509.00008)  

**Abstract**: Traditional power grid infrastructure presents significant barriers to renewable energy integration and perpetuates energy access inequities, with low-income communities experiencing disproportionately longer power outages. This study develops a Markov Decision Process (MDP) framework to optimize renewable energy allocation while explicitly addressing social equity concerns in electricity distribution. The model incorporates budget constraints, energy demand variability, and social vulnerability indicators across eight major U.S. cities to evaluate policy alternatives for equitable clean energy transitions. Numerical experiments compare the MDP-based approach against baseline policies including random allocation, greedy renewable expansion, and expert heuristics. Results demonstrate that equity-focused optimization can achieve 32.9% renewable energy penetration while reducing underserved low-income populations by 55% compared to conventional approaches. The expert policy achieved the highest reward, while the Monte Carlo Tree Search baseline provided competitive performance with significantly lower budget utilization, demonstrating that fair distribution of clean energy resources is achievable without sacrificing overall system performance and providing ways for integrating social equity considerations with climate goals and inclusive access to clean power infrastructure. 

**Abstract (ZH)**: 传统电网基础设施对可再生能源整合构成显著障碍，并加剧能源访问不平等，低收入社区经历的停电时间明显更长。本研究建立马尔可夫决策过程（MDP）框架以优化可再生能源分配，并明确解决电力分配中的社会公平问题。该模型结合八大美国城市预算限制、能源需求波动和社会脆弱性指标，评估促进公平清洁能源转型的政策措施。数值实验比较了基于MDP的方法与基准政策（包括随机分配、贪婪可再生能源扩展和专家启发式方法）的表现。结果表明，基于公平优化的方法可以实现32.9%的可再生能源渗透率，并将未获充分服务的低收入人口减少55%，相比之下，传统方法更具优势。专家政策获得了最高奖励，而蒙特卡洛树搜索基准性能良好且预算利用显著较低，证明公平分配清洁能源资源是可以实现的，不会牺牲整体系统性能，并提供了将社会公平考虑与气候目标及清洁电力基础设施的包容性接入相结合的方法。 

---
# Per-sender neural network classifiers for email authorship validation 

**Title (ZH)**: 基于发送者的神经网络分类器用于电子邮件作者验证 

**Authors**: Rohit Dube  

**Link**: [PDF](https://arxiv.org/pdf/2509.00005)  

**Abstract**: Business email compromise and lateral spear phishing attacks are among modern organizations' most costly and damaging threats. While inbound phishing defenses have improved significantly, most organizations still trust internal emails by default, leaving themselves vulnerable to attacks from compromised employee accounts. In this work, we define and explore the problem of authorship validation: verifying whether a claimed sender actually authored a given email. Authorship validation is a lightweight, real-time defense that complements traditional detection methods by modeling per-sender writing style. Further, the paper presents a collection of new datasets based on the Enron corpus. These simulate inauthentic messages using both human-written and large language model-generated emails. The paper also evaluates two classifiers -- a Naive Bayes model and a character-level convolutional neural network (Char-CNN) -- for the authorship validation task. Our experiments show that the Char-CNN model achieves high accuracy and F1 scores under various circumstances. Finally, we discuss deployment considerations and show that per-sender authorship classifiers are practical for integrating into existing commercial email security systems with low overhead. 

**Abstract (ZH)**: 商务电子邮件诈骗和横向鱼叉攻击是现代组织最昂贵和最具破坏性的威胁之一。虽然反入站钓鱼防御已经显著改善，但大多数组织仍然默认信任内部电子邮件，使得其容易受到遭篡改员工账户的攻击。在本文中，我们定义并探讨了作者身份验证的问题：验证声称的发件人是否实际撰写了给定的电子邮件。作者身份验证是一种轻量级的实时防御措施，通过建模每个发件人的写作风格来补充传统的检测方法。此外，本文还基于Enron语料库构建了一系列新的数据集，这些数据集模拟了使用人工撰写的和大型语言模型生成的电子邮件的不真实消息。本文还对两种分类器——朴素贝叶斯模型和字符级卷积神经网络（Char-CNN）——进行了评估，以完成作者身份验证任务。我们的实验表明，在各种情况下，Char-CNN模型实现了高准确率和F1分数。最后，我们讨论了部署考虑，并展示了针对现有商业电子邮件安全系统的低开销集成是可行的。 

---
