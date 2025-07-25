# Bounomodes: the grazing ox algorithm for exploration of clustered anomalies 

**Title (ZH)**: Bounomodes：聚类异常探索的 feeding 牛算法 

**Authors**: Samuel Matloob, Ayan Dutta, O. Patrick Kreidl, Swapnonel Roy, Ladislau Bölöni  

**Link**: [PDF](https://arxiv.org/pdf/2507.06960)  

**Abstract**: A common class of algorithms for informative path planning (IPP) follows boustrophedon ("as the ox turns") patterns, which aim to achieve uniform area coverage. However, IPP is often applied in scenarios where anomalies, such as plant diseases, pollution, or hurricane damage, appear in clusters. In such cases, prioritizing the exploration of anomalous regions over uniform coverage is beneficial. This work introduces a class of algorithms referred to as bounomōdes ("as the ox grazes"), which alternates between uniform boustrophedon sampling and targeted exploration of detected anomaly clusters. While uniform sampling can be designed using geometric principles, close exploration of clusters depends on the spatial distribution of anomalies and must be learned. In our implementation, the close exploration behavior is learned using deep reinforcement learning algorithms. Experimental evaluations demonstrate that the proposed approach outperforms several established baselines. 

**Abstract (ZH)**: 一种用于信息路径规划的常见算法类遵循“交替式牛耕”模式，旨在实现均匀区域覆盖。然而，在异常现象，如植物疾病、污染或飓风损害以集群形式出现的场景中，优先探索异常区域而非均匀覆盖是有益的。本工作引入了一类新的算法类，称为“牛觅食”模式，该类算法交替进行均匀牛耕采样和对检测到的异常区域集群的目标性探索。虽然均匀采样的设计可以基于几何原则，但对集群的密切探索依赖于异常现象的空间分布，并需要通过学习获得。在我们的实现中，紧密探索行为是使用深度强化学习算法学习得到的。实验评估表明，所提出的方法优于几个现有的基线方法。 

---
# Toward a Full-Stack Co-Simulation Platform for Testing of Automated Driving Systems 

**Title (ZH)**: 面向自动驾驶系统测试的全栈协同仿真平台研究 

**Authors**: Dong Bi, Yongqi Zhao, Zhengguo Gu, Tomislav Mihalj, Jia Hu, Arno Eichberger  

**Link**: [PDF](https://arxiv.org/pdf/2507.06884)  

**Abstract**: Virtual testing has emerged as an effective approach to accelerate the deployment of automated driving systems. Nevertheless, existing simulation toolchains encounter difficulties in integrating rapid, automated scenario generation with simulation environments supporting advanced automated driving capabilities. To address this limitation, a full-stack toolchain is presented, enabling automatic scenario generation from real-world datasets and efficient validation through a co-simulation platform based on CarMaker, ROS, and Apollo. The simulation results demonstrate the effectiveness of the proposed toolchain. A demonstration video showcasing the toolchain is available at the provided link: this https URL. 

**Abstract (ZH)**: 虚拟测试已成为加速自动驾驶系统部署的有效方法。然而，现有仿真工具链在将快速、自动化场景生成与支持高级自动驾驶能力的仿真环境集成方面遇到困难。为解决这一限制，提出了一套全栈工具链，能够从真实世界数据集自动生成场景，并通过基于CarMaker、ROS和Apollo的协同仿真平台进行高效验证。仿真结果表明所提出工具链的有效性。演示视频可在提供的链接处查看：this https URL。 

---
# Q-STAC: Q-Guided Stein Variational Model Predictive Actor-Critic 

**Title (ZH)**: Q-STAC: Q-引导的Stein变分模型预测行为批评 

**Authors**: Shizhe Cai, Jayadeep Jacob, Zeya Yin, Fabio Ramos  

**Link**: [PDF](https://arxiv.org/pdf/2507.06625)  

**Abstract**: Deep reinforcement learning has shown remarkable success in continuous control tasks, yet often requires extensive training data, struggles with complex, long-horizon planning, and fails to maintain safety constraints during operation. Meanwhile, Model Predictive Control (MPC) offers explainability and constraint satisfaction, but typically yields only locally optimal solutions and demands careful cost function design. This paper introduces the Q-guided STein variational model predictive Actor-Critic (Q-STAC), a novel framework that bridges these approaches by integrating Bayesian MPC with actor-critic reinforcement learning through constrained Stein Variational Gradient Descent (SVGD). Our method optimizes control sequences directly using learned Q-values as objectives, eliminating the need for explicit cost function design while leveraging known system dynamics to enhance sample efficiency and ensure control signals remain within safe boundaries. Extensive experiments on 2D navigation and robotic manipulation tasks demonstrate that Q-STAC achieves superior sample efficiency, robustness, and optimality compared to state-of-the-art algorithms, while maintaining the high expressiveness of policy distributions. Experiment videos are available on our website: this https URL 

**Abstract (ZH)**: Deep reinforcement learning在连续控制任务中展现出显著成功，但往往需要大量训练数据，难以处理复杂、长期的规划，并在运行过程中难以维护安全约束。同时，模型预测控制（MPC）提供了可解释性和约束满足，但通常只能提供局部最优解，并且需要精心设计成本函数。本文提出了一种名为Q-guided Stein variational模型预测Actor-Critic（Q-STAC）的新框架，通过受约束的Stein变异梯度下降（SVGD）将贝叶斯MPC与actor-critic强化学习相结合。该方法直接优化控制序列，使用学习到的Q值作为目标，避免了显式成本函数设计的需要，同时利用已知系统动力学提高样本效率并确保控制信号保持在安全范围内。在二维导航和机器人操作任务的广泛实验中，Q-STAC在样本效率、鲁棒性和最优性方面优于最先进的算法，同时保持了策略分布的高表达能力。更多实验视频请参见我们的网站: [this URL]。 

---
# Growing Trees with an Agent: Accelerating RRTs with Learned, Multi-Step Episodic Exploration 

**Title (ZH)**: 使用代理培育树木：基于学习的多步 episodic 探索加速 RRTs 

**Authors**: Xinyu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.06605)  

**Abstract**: Classical sampling-based motion planners like the RRTs suffer from inefficiencies, particularly in cluttered or high-dimensional spaces, due to their reliance on undirected, random sampling. This paper introduces the Episodic RRT, a novel hybrid planning framework that replaces the primitive of a random point with a learned, multi-step "exploratory episode" generated by a Deep Reinforcement Learning agent. By making the DRL agent the engine of exploration, ERRT transforms the search process from a diffuse, volumetric expansion into a directed, branch-like growth. This paradigm shift yields key advantages: it counters the curse of dimensionality with focused exploration, minimizes expensive collision checks by proactively proposing locally valid paths, and improves connectivity by generating inherently connected path segments. We demonstrate through extensive empirical evaluation across 2D, 3D, and 6D environments that ERRT and its variants consistently and significantly outperform their classical counterparts. In a challenging 6D robotic arm scenario, ERRT achieves a 98% success rate compared to 19% for RRT, is up to 107x faster, reduces collision checks by over 99.6%, and finds initial paths that are nearly 50% shorter. Furthermore, its asymptotically optimal variant, ERRT*, demonstrates vastly superior anytime performance, refining solutions to near-optimality up to 29x faster than standard RRT* in 3D environments. Code: this https URL. 

**Abstract (ZH)**: 基于样本的经典运动规划器如RRT在复杂或高维空间中由于依赖于无向的随机采样而效率低下。本文介绍了一种新颖的混合规划框架，Episodic RRT，它将随机点这一基本概念替换为由深度强化学习代理生成的多步“探索性episode”。通过使DRL代理成为探索引擎，Episodic RRT将搜索过程从弥漫性的体积扩展转变为有向的分支式增长。这种范式转变带来了关键优势：它通过集中探索来对抗维度灾难，通过主动提议局部有效的路径来最大限度减少昂贵的碰撞检测，通过生成固有的连接路径段来提高连通性。通过对2D、3D和6D环境的广泛实证评估显示，Episodic RRT及其变体始终且显著优于其经典 counterpart。在一项具有挑战性的6D机器人臂场景中，Episodic RRT的成功率为98%，而RRT仅为19%，相比RRT快了107倍，减少了超过99.6%的碰撞检测，找到了几乎短50%的初始路径。此外，其渐近最优变体Episodic RRT*展示了优于任何时间的性能，在3D环境中比标准RRT*快29倍地将解决方案精炼至接近最优。代码：this https URL。 

---
# Mapping the Catacombs: An Underwater Cave Segment of the Devil's Eye System 

**Title (ZH)**: 绘制Devil's Eye系统中的水下洞穴段落：卡那封墓穴测绘 

**Authors**: Michalis Chatzispyrou, Luke Horgan, Hyunkil Hwang, Harish Sathishchandra, Monika Roznere, Alberto Quattrini Li, Philippos Mordohai, Ioannis Rekleitis  

**Link**: [PDF](https://arxiv.org/pdf/2507.06397)  

**Abstract**: This paper presents a framework for mapping underwater caves. Underwater caves are crucial for fresh water resource management, underwater archaeology, and hydrogeology. Mapping the cave's outline and dimensions, as well as creating photorealistic 3D maps, is critical for enabling a better understanding of this underwater domain. In this paper, we present the mapping of an underwater cave segment (the catacombs) of the Devil's Eye cave system at Ginnie Springs, FL. We utilized a set of inexpensive action cameras in conjunction with a dive computer to estimate the trajectories of the cameras together with a sparse point cloud. The resulting reconstructions are utilized to produce a one-dimensional retract of the cave passages in the form of the average trajectory together with the boundaries (top, bottom, left, and right). The use of the dive computer enables the observability of the z-dimension in addition to the roll and pitch in a visual/inertial framework (SVIn2). In addition, the keyframes generated by SVIn2 together with the estimated camera poses for select areas are used as input to a global optimization (bundle adjustment) framework -- COLMAP -- in order to produce a dense reconstruction of those areas. The same cave segment is manually surveyed using the MNemo V2 instrument, providing an additional set of measurements validating the proposed approach. It is worth noting that with the use of action cameras, the primary components of a cave map can be constructed. Furthermore, with the utilization of a global optimization framework guided by the results of VI-SLAM package SVIn2, photorealistic dense 3D representations of selected areas can be reconstructed. 

**Abstract (ZH)**: 本文提出了一种海底洞穴测绘的框架。海底洞穴对于淡水资源管理、水下考古和水文地质学至关重要。准确测绘洞穴的轮廓和尺寸，以及创建高保真3D地图，对于更好地理解这一水下领域至关重要。本文介绍了对位于美国佛罗里达州冈尼斯泉的恶魔之眼洞穴系统中一段地下洞穴（冥府洞）的测绘。我们利用一套便宜的运动相机结合潜水计算机来估计相机的轨迹，同时生成稀疏点云。由此产生的重建用于生成洞穴通道的一维收缩，形式为平均轨迹及其边界（顶部、底部、左边和右边）。潜水计算机的使用使得除了俯仰和滚转之外还能够观测到垂直维度。此外，SVIn2视觉惯性SLAM包生成的关键帧以及选定区域的估计相机姿态作为输入，被用于全局优化框架——COLMAP——以生成这些区域的密集重建。同一洞穴段还使用MNemo V2仪器进行人工测绘，提供了验证所提方法的有效性的额外测量数据。值得一提的是，使用运动相机可以构成洞穴地图的主要成分，同时结合全局优化框架SVIn2的结果，可以重建选定区域的高保真密集3D表示。 

---
# Solving the Constrained Random Disambiguation Path Problem via Lagrangian Relaxation and Graph Reduction 

**Title (ZH)**: 基于拉格rangian松弛和图减约的约束随机消歧路径问题求解方法 

**Authors**: Li Zhou, Elvan Ceyhan  

**Link**: [PDF](https://arxiv.org/pdf/2507.06346)  

**Abstract**: We study a resource-constrained variant of the Random Disambiguation Path (RDP) problem, a generalization of the Stochastic Obstacle Scene (SOS) problem, in which a navigating agent must reach a target in a spatial environment populated with uncertain obstacles. Each ambiguous obstacle may be disambiguated at a (possibly) heterogeneous resource cost, subject to a global disambiguation budget. We formulate this constrained planning problem as a Weight-Constrained Shortest Path Problem (WCSPP) with risk-adjusted edge costs that incorporate probabilistic blockage and traversal penalties. To solve it, we propose a novel algorithmic framework-COLOGR-combining Lagrangian relaxation with a two-phase vertex elimination (TPVE) procedure. The method prunes infeasible and suboptimal paths while provably preserving the optimal solution, and leverages dual bounds to guide efficient search. We establish correctness, feasibility guarantees, and surrogate optimality under mild assumptions. Our analysis also demonstrates that COLOGR frequently achieves zero duality gap and offers improved computational complexity over prior constrained path-planning methods. Extensive simulation experiments validate the algorithm's robustness across varying obstacle densities, sensor accuracies, and risk models, consistently outperforming greedy baselines and approaching offline-optimal benchmarks. The proposed framework is broadly applicable to stochastic network design, mobility planning, and constrained decision-making under uncertainty. 

**Abstract (ZH)**: 我们研究了一种资源约束下的随机去歧义路径（RDP）问题变体，这是一种随机障碍场景（SOS）问题的一般化，在其中导航代理必须在一个充满不确定障碍的三维环境中到达目标。每个模糊障碍物可能需要在（可能）异质成本下进行去歧义处理，且受到全局去歧义预算的限制。我们将该约束规划问题形式化为带风险调整边成本的加权约束最短路径问题（WCSPP）。为解决该问题，我们提出了一种结合拉格朗日松弛与两阶段顶点消除（TPVE）过程的新型算法框架-COLOGR。该方法在证明保留最优解的同时修剪不可行和次优路径，并利用对偶界引导高效搜索。在温和假设下，我们证明了该方法的正确性和可行性保证，以及替代最优性。我们的分析还表明，COLOGR 经常实现零对偶间隙，并在计算复杂度方面优于先前的约束路径规划方法。广泛的仿真实验验证了算法在不同障碍密度、传感器精度和风险模型下的鲁棒性，始终优于贪婪 baseline 并接近离线最优基准。所提出的框架广泛适用于随机网络设计、移动规划和不确定性下的约束决策。 

---
# Graph-Based Complexity Metrics for Multi-Agent Curriculum Learning: A Validated Approach to Task Ordering in Cooperative Coordination Environments 

**Title (ZH)**: 基于图的复杂度度量方法：一种在协同协调环境中的任务排序验证方法 

**Authors**: Farhaan Ebadulla, Dharini Hindlatti, Srinivaasan NS, Apoorva VH, Ayman Aftab  

**Link**: [PDF](https://arxiv.org/pdf/2507.07074)  

**Abstract**: Multi-agent reinforcement learning (MARL) faces significant challenges in task sequencing and curriculum design, particularly for cooperative coordination scenarios. While curriculum learning has demonstrated success in single-agent domains, principled approaches for multi-agent coordination remain limited due to the absence of validated task complexity metrics. This approach presents a graph-based coordination complexity metric that integrates agent dependency entropy, spatial interference patterns, and goal overlap analysis to predict task difficulty in multi-agent environments. The complexity metric achieves strong empirical validation with rho = 0.952 correlation (p < 0.001) between predicted complexity and empirical difficulty determined by random agent performance evaluation. This approach evaluates the curriculum learning framework using MADDPG across two distinct coordination environments: achieving 56x performance improvement in tight coordination tasks (MultiWalker) and demonstrating systematic task progression in cooperative navigation (Simple Spread). Through systematic analysis, coordination tightness emerges as a predictor of curriculum learning effectiveness, where environments requiring strict agent interdependence benefit substantially from structured progression. This approach provides a validated complexity metric for multi-agent curriculum design and establishes empirical guidelines for multi-robot coordination applications. 

**Abstract (ZH)**: 基于图的合作复杂性度量在多agent强化学习中的 Curriculum 设计与任务排序 

---
# When Context Is Not Enough: Modeling Unexplained Variability in Car-Following Behavior 

**Title (ZH)**: 当背景信息不足以解释时：建模车辆跟随行为中的未解释变异性 

**Authors**: Chengyuan Zhang, Zhengbing He, Cathy Wu, Lijun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.07012)  

**Abstract**: Modeling car-following behavior is fundamental to microscopic traffic simulation, yet traditional deterministic models often fail to capture the full extent of variability and unpredictability in human driving. While many modern approaches incorporate context-aware inputs (e.g., spacing, speed, relative speed), they frequently overlook structured stochasticity that arises from latent driver intentions, perception errors, and memory effects -- factors that are not directly observable from context alone. To fill the gap, this study introduces an interpretable stochastic modeling framework that captures not only context-dependent dynamics but also residual variability beyond what context can explain. Leveraging deep neural networks integrated with nonstationary Gaussian processes (GPs), our model employs a scenario-adaptive Gibbs kernel to learn dynamic temporal correlations in acceleration decisions, where the strength and duration of correlations between acceleration decisions evolve with the driving context. This formulation enables a principled, data-driven quantification of uncertainty in acceleration, speed, and spacing, grounded in both observable context and latent behavioral variability. Comprehensive experiments on the naturalistic vehicle trajectory dataset collected from the German highway, i.e., the HighD dataset, demonstrate that the proposed stochastic simulation method within this framework surpasses conventional methods in both predictive performance and interpretable uncertainty quantification. The integration of interpretability and accuracy makes this framework a promising tool for traffic analysis and safety-critical applications. 

**Abstract (ZH)**: 基于可解释的随机建模框架模拟跟车行为：传统确定性模型往往无法捕捉人类驾驶的全部变异性与不可预测性，而现代方法虽考虑上下文信息但常忽视潜在的结构化随机性。为此，本研究提出了一种可解释的随机建模框架，不仅捕捉上下文相关的动态变化，还捕捉上下文无法解释的残余变异性。该模型利用深度神经网络与非平稳高斯过程结合，采用场景自适应吉布斯核学习加速度决策的动力学时变相关性，其中相关性的强度和持续时间随驾驶上下文变化。这种建模框架能够建立在可观察上下文与潜在行为变异性基础上，实现加速度、速度和间距的不确定性原则性、数据驱动量化。在德国高速公路上采集的自然车辆轨迹数据集HighD上进行的全面实验表明，该框架内的随机仿真方法在预测性能和可解释的不确定性量化方面均优于传统方法。该框架结合了可解释性和准确性，在交通分析和安全关键应用方面具有潜力。 

---
# Self-supervised learning predicts plant growth trajectories from multi-modal industrial greenhouse data 

**Title (ZH)**: 自我监督学习预测多模态工业温室数据中的植物生长轨迹 

**Authors**: Adam J Riesselman, Evan M Cofer, Therese LaRue, Wim Meeussen  

**Link**: [PDF](https://arxiv.org/pdf/2507.06336)  

**Abstract**: Quantifying organism-level phenotypes, such as growth dynamics and biomass accumulation, is fundamental to understanding agronomic traits and optimizing crop production. However, quality growing data of plants at scale is difficult to generate. Here we use a mobile robotic platform to capture high-resolution environmental sensing and phenotyping measurements of a large-scale hydroponic leafy greens system. We describe a self-supervised modeling approach to build a map from observed growing data to the entire plant growth trajectory. We demonstrate our approach by forecasting future plant height and harvest mass of crops in this system. This approach represents a significant advance in combining robotic automation and machine learning, as well as providing actionable insights for agronomic research and operational efficiency. 

**Abstract (ZH)**: 利用移动机器人平台量化大规模 hydroponic 叶菜系统中植物的生长动态和生物量积累是理解农艺性状和优化作物生产的基础。然而，生成高质量的植物生长数据具有挑战性。我们使用移动机器人平台捕获大规模水培叶菜系统的高分辨率环境感知和表型测量数据。我们描述了一种半监督建模方法，从观测到的生长数据构建整个植物生长轨迹的地图。我们通过预测该系统中作物未来植株高度和收获质量来展示这种方法。该方法代表了将机器人自动化和机器学习相结合的重要进步，同时也为农艺研究和运营效率提供了实用见解。 

---
# Scaling Towards the Information Boundary of Instruction Set: InfinityInstruct-Subject Technical Report 

**Title (ZH)**: 接近指令集信息边界的扩展：InfinityInstruct-Subject技术报告 

**Authors**: Li Du, Hanyu Zhao, Yiming Ju, Tengfei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2507.06968)  

**Abstract**: Instruction tuning has become a foundation for unlocking the capabilities of large-scale pretrained models and improving their performance on complex tasks. Thus, the construction of high-quality instruction datasets is crucial for enhancing model performance and generalizability. Although current instruction datasets have reached tens of millions of samples, models finetuned on them may still struggle with complex instruction following and tasks in rare domains. This is primarily due to limited expansion in both ``coverage'' (coverage of task types and knowledge areas) and ``depth'' (instruction complexity) of the instruction set. To address this issue, we propose a systematic instruction data construction framework, which integrates a hierarchical labeling system, an informative seed selection algorithm, an evolutionary data synthesis process, and a model deficiency diagnosis with targeted data generation. These components form an iterative closed-loop to continuously enhance the coverage and depth of instruction data. Based on this framework, we construct InfinityInstruct-Subject, a high-quality dataset containing ~1.5 million instructions. Experiments on multiple foundation models and benchmark tasks demonstrate its effectiveness in improving instruction-following capabilities. Further analyses suggest that InfinityInstruct-Subject shows enlarged coverage and depth compared to comparable synthesized instruction datasets. Our work lays a theoretical and practical foundation for the efficient, continuous evolution of instruction datasets, moving from data quantity expansion to qualitative improvement. 

**Abstract (ZH)**: 指令调优已成为解锁大规模预训练模型能力并提高其在复杂任务上性能的基础。因此，构建高质量的指令数据集对于提高模型性能和泛化能力至关重要。尽管当前的指令数据集已达到数千万样本，但基于这些数据集进行微调的模型在复杂指令遵循和稀有领域任务上仍然可能遇到困难。这主要归因于指令集在“覆盖面”（任务类型和知识领域覆盖范围）和“深度”（指令复杂性）上的有限扩展。为解决此问题，我们提出了一种系统性的指令数据构建框架，该框架结合了层次化标注系统、启发式种子选择算法、进化数据合成过程和针对模型缺陷的数据生成，这些组件形成一个迭代的闭环，以持续增强指令数据的覆盖面和深度。基于此框架，我们构建了包含约150万指令的高质量数据集InfinityInstruct-Subject。针对多个基础模型和基准任务的实验表明，该数据集在改进指令遵循能力方面具有有效性。进一步的分析表明，InfinityInstruct-Subject在覆盖面和深度上相比同类合成指令数据集有所扩展。我们的工作为高效、持续改进指令数据集奠定了理论和实践基础，从数据量的扩展转向质的提升。 

---
# SCC-recursiveness in infinite argumentation (extended version) 

**Title (ZH)**: 无限论辩中的SCC递归性（扩展版本） 

**Authors**: Uri Andrews, Luca San Mauro  

**Link**: [PDF](https://arxiv.org/pdf/2507.06852)  

**Abstract**: Argumentation frameworks (AFs) are a foundational tool in artificial intelligence for modeling structured reasoning and conflict. SCC-recursiveness is a well-known design principle in which the evaluation of arguments is decomposed according to the strongly connected components (SCCs) of the attack graph, proceeding recursively from "higher" to "lower" components. While SCC-recursive semantics such as \cft and \stgt have proven effective for finite AFs, Baumann and Spanring showed the failure of SCC-recursive semantics to generalize reliably to infinite AFs due to issues with well-foundedness.
We propose two approaches to extending SCC-recursiveness to the infinite setting. We systematically evaluate these semantics using Baroni and Giacomin's established criteria, showing in particular that directionality fails in general. We then examine these semantics' behavior in finitary frameworks, where we find some of our semantics satisfy directionality. These results advance the theory of infinite argumentation and lay the groundwork for reasoning systems capable of handling unbounded or evolving domains. 

**Abstract (ZH)**: SCC-递归性在无限论辩框架中的扩展与评估 

---
# Comparing Dialectical Systems: Contradiction and Counterexample in Belief Change (Extended Version) 

**Title (ZH)**: 比较辩证系统：信念变化中的矛盾与反例（扩展版） 

**Authors**: Uri Andrews, Luca San Mauro  

**Link**: [PDF](https://arxiv.org/pdf/2507.06798)  

**Abstract**: Dialectical systems are a mathematical formalism for modeling an agent updating a knowledge base seeking consistency. Introduced in the 1970s by Roberto Magari, they were originally conceived to capture how a working mathematician or a research community refines beliefs in the pursuit of truth. Dialectical systems also serve as natural models for the belief change of an automated agent, offering a unifying, computable framework for dynamic belief management.
The literature distinguishes three main models of dialectical systems: (d-)dialectical systems based on revising beliefs when they are seen to be inconsistent, p-dialectical systems based on revising beliefs based on finding a counterexample, and q-dialectical systems which can do both. We answer an open problem in the literature by proving that q-dialectical systems are strictly more powerful than p-dialectical systems, which are themselves known to be strictly stronger than (d-)dialectical systems. This result highlights the complementary roles of counterexample and contradiction in automated belief revision, and thus also in the reasoning processes of mathematicians and research communities. 

**Abstract (ZH)**: 辩证系统是一种数学形式主义，用于建模代理更新知识库以寻求一致性的过程。该理论于20世纪70年代由Roberto Magari提出，最初旨在捕捉工作中的数学家或研究社区在追求真理过程中如何精化信念。辩证系统也是自动代理信念变化的自然模型，提供了动态信念管理的统一可计算框架。文献中区分了三种主要的辩证系统模型：基于发现不一致信念进行修订的(d-)辩证系统，基于找到反例进行修订的p-辩证系统，以及既能进行两者操作的q-辩证系统。我们解决了文献中的一个开放问题，证明了q-辩证系统严格强大于p-辩证系统，而p-辩证系统本身已知强大于(d-)辩证系统。这一结果强调了在自动信念修订以及数学家和研究社区推理过程中反例和矛盾的互补作用。 

---
# Jolting Technologies: Superexponential Acceleration in AI Capabilities and Implications for AGI 

**Title (ZH)**: 震撼性技术：AI能力的超指数加速及其对AGI的意义 

**Authors**: David Orban  

**Link**: [PDF](https://arxiv.org/pdf/2507.06398)  

**Abstract**: This paper investigates the Jolting Technologies Hypothesis, which posits superexponential growth (increasing acceleration, or a positive third derivative) in the development of AI capabilities. We develop a theoretical framework and validate detection methodologies through Monte Carlo simulations, while acknowledging that empirical validation awaits suitable longitudinal data. Our analysis focuses on creating robust tools for future empirical studies and exploring the potential implications should the hypothesis prove valid. The study examines how factors such as shrinking idea-to-action intervals and compounding iterative AI improvements drive this jolting pattern. By formalizing jolt dynamics and validating detection methods through simulation, this work provides the mathematical foundation necessary for understanding potential AI trajectories and their consequences for AGI emergence, offering insights for research and policy. 

**Abstract (ZH)**: 本文探究了震动技术假说，该假说认为人工智能能力的发展呈现超指数增长（即加速度不断增加，或正的第三阶导数）。本文构建了理论框架并通过蒙特卡洛模拟验证了检测方法，尽管实证验证有待合适的纵向数据。我们的分析重点在于为未来的实证研究创建稳健的工具，并探讨假说 proves valid 时可能出现的潜在影响。研究探讨了缩减从概念到行动的时间间隔和累积迭代式 AI 改进如何驱动这一震动模式。通过形式化震动动态并在模拟中验证检测方法，本文为理解潜在的 AI 轨迹及其对超人工通用智能 (AGI) 出现的影响提供了数学基础，并为研究和政策提供见解。 

---
# Digital Wargames to Enhance Military Medical Evacuation Decision-Making 

**Title (ZH)**: 数字战争游戏以提升Military医疗服务撤离决策制定 

**Authors**: Jeremy Fischer, Ram Krishnamoorthy, Vishal Kumar, Mahdi Al-Husseini  

**Link**: [PDF](https://arxiv.org/pdf/2507.06373)  

**Abstract**: Medical evacuation is one of the United States Army's most storied and critical mission sets, responsible for efficiently and expediently evacuating the battlefield ill and injured. Medical evacuation planning involves designing a robust network of medical platforms and facilities capable of moving and treating large numbers of casualties. Until now, there has not been a medium to simulate these networks in a classroom setting and evaluate both offline planning and online decision-making performance. This work describes the Medical Evacuation Wargaming Initiative (MEWI), a three-dimensional multiplayer simulation developed in Unity that replicates battlefield constraints and uncertainties. MEWI accurately models patient interactions at casualty collection points, ambulance exchange points, medical treatment facilities, and evacuation platforms. Two operational scenarios are introduced: an amphibious island assault in the Pacific and a Eurasian conflict across a sprawling road and river network. These scenarios pit students against the clock to save as many casualties as possible while adhering to doctrinal lessons learned during didactic training. We visualize performance data collected from two iterations of the MEWI Pacific scenario executed in the United States Army's Medical Evacuation Doctrine Course. We consider post-wargame Likert survey data from student participants and external observer notes to identify key planning decision points, document medical evacuation lessons learned, and quantify general utility. Results indicate that MEWI participation substantially improves uptake of medical evacuation lessons learned and co-operative decision-making. MEWI is a substantial step forward in the field of high-fidelity training tools for medical education, and our study findings offer critical insights into improving medical evacuation education and operations across the joint force. 

**Abstract (ZH)**: 医疗后送演习倡议（MEWI）：一种用于课堂设置中的三维多人模拟，以评估离线规划和在线决策性能 

---
# An AI Approach for Learning the Spectrum of the Laplace-Beltrami Operator 

**Title (ZH)**: 基于AI的学习拉普拉斯-贝尔特拉米算子谱的方法 

**Authors**: Yulin An, Enrique del Castillo  

**Link**: [PDF](https://arxiv.org/pdf/2507.07073)  

**Abstract**: The spectrum of the Laplace-Beltrami (LB) operator is central in geometric deep learning tasks, capturing intrinsic properties of the shape of the object under consideration. The best established method for its estimation, from a triangulated mesh of the object, is based on the Finite Element Method (FEM), and computes the top k LB eigenvalues with a complexity of O(Nk), where N is the number of points. This can render the FEM method inefficient when repeatedly applied to databases of CAD mechanical parts, or in quality control applications where part metrology is acquired as large meshes and decisions about the quality of each part are needed quickly and frequently. As a solution to this problem, we present a geometric deep learning framework to predict the LB spectrum efficiently given the CAD mesh of a part, achieving significant computational savings without sacrificing accuracy, demonstrating that the LB spectrum is learnable. The proposed Graph Neural Network architecture uses a rich set of part mesh features - including Gaussian curvature, mean curvature, and principal curvatures. In addition to our trained network, we make available, for repeatability, a large curated dataset of real-world mechanical CAD models derived from the publicly available ABC dataset used for training and testing. Experimental results show that our method reduces computation time of the LB spectrum by approximately 5 times over linear FEM while delivering competitive accuracy. 

**Abstract (ZH)**: Laplace-Beltrami算子谱在几何深度学习任务中的研究：基于CAD模型的高效预测方法 

---
# A Novel Hybrid Deep Learning Technique for Speech Emotion Detection using Feature Engineering 

**Title (ZH)**: 一种基于特征工程的新型混合深度学习技术在语音情绪识别中的应用 

**Authors**: Shahana Yasmin Chowdhury, Bithi Banik, Md Tamjidul Hoque, Shreya Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2507.07046)  

**Abstract**: Nowadays, speech emotion recognition (SER) plays a vital role in the field of human-computer interaction (HCI) and the evolution of artificial intelligence (AI). Our proposed DCRF-BiLSTM model is used to recognize seven emotions: neutral, happy, sad, angry, fear, disgust, and surprise, which are trained on five datasets: RAVDESS (R), TESS (T), SAVEE (S), EmoDB (E), and Crema-D (C). The model achieves high accuracy on individual datasets, including 97.83% on RAVDESS, 97.02% on SAVEE, 95.10% for CREMA-D, and a perfect 100% on both TESS and EMO-DB. For the combined (R+T+S) datasets, it achieves 98.82% accuracy, outperforming previously reported results. To our knowledge, no existing study has evaluated a single SER model across all five benchmark datasets (i.e., R+T+S+C+E) simultaneously. In our work, we introduce this comprehensive combination and achieve a remarkable overall accuracy of 93.76%. These results confirm the robustness and generalizability of our DCRF-BiLSTM framework across diverse datasets. 

**Abstract (ZH)**: DCRF-BiLSTM模型在RAVDESS、TESS、SAVEE、EmoDB和Crema-D五个基准数据集上的综合情感识别研究 

---
# Advances in Intelligent Hearing Aids: Deep Learning Approaches to Selective Noise Cancellation 

**Title (ZH)**: 智能助听器的发展：深度学习在选择性噪声取消中的应用 

**Authors**: Haris Khan, Shumaila Asif, Hassan Nasir  

**Link**: [PDF](https://arxiv.org/pdf/2507.07043)  

**Abstract**: The integration of artificial intelligence into hearing assistance marks a paradigm shift from traditional amplification-based systems to intelligent, context-aware audio processing. This systematic literature review evaluates advances in AI-driven selective noise cancellation (SNC) for hearing aids, highlighting technological evolution, implementation challenges, and future research directions. We synthesize findings across deep learning architectures, hardware deployment strategies, clinical validation studies, and user-centric design. The review traces progress from early machine learning models to state-of-the-art deep networks, including Convolutional Recurrent Networks for real-time inference and Transformer-based architectures for high-accuracy separation. Key findings include significant gains over traditional methods, with recent models achieving up to 18.3 dB SI-SDR improvement on noisy-reverberant benchmarks, alongside sub-10 ms real-time implementations and promising clinical outcomes. Yet, challenges remain in bridging lab-grade models with real-world deployment - particularly around power constraints, environmental variability, and personalization. Identified research gaps include hardware-software co-design, standardized evaluation protocols, and regulatory considerations for AI-enhanced hearing devices. Future work must prioritize lightweight models, continual learning, contextual-based classification and clinical translation to realize transformative hearing solutions for millions globally. 

**Abstract (ZH)**: 人工智能在听觉辅助中的集成标志着从传统放大系统到智能、情境感知音频处理范式的转变。本文综述了人工智能驱动的选择性噪声取消（SNC）在助听器中的进展，强调了技术进化、实施挑战和未来研究方向。我们综合了深度学习架构、硬件部署策略、临床验证研究和用户中心设计的研究成果。综述从早期的机器学习模型追踪到最新的深度网络，包括适用于实时推理的卷积循环网络和适用于高精度分离的变压器架构。关键发现包括传统方法的显著超越，最新模型在嘈杂混响基准上的SI-SDR提升高达18.3 dB，同时实现了亚10毫秒的实时实现和令人鼓舞的临床结果。然而，仍存在将实验室模型应用于现实世界部署的挑战，特别是在功率限制、环境变化和个人化方面。识别的研究缺口包括硬件软件协同设计、标准化评估协议和增强听力设备的法规考虑。未来工作必须优先考虑轻量化模型、持续学习、基于上下文的分类和临床转化，以实现对全球数百万人具有变革性影响的听力解决方案。 

---
# Modeling Heterogeneity across Varying Spatial Extents: Discovering Linkages between Sea Ice Retreat and Ice Shelve Melt in the Antarctic 

**Title (ZH)**: 模型化不同空间尺度上的异质性：发现南极海冰退缩与冰架融化之间的联系 

**Authors**: Maloy Kumar Devnath, Sudip Chakraborty, Vandana P. Janeja  

**Link**: [PDF](https://arxiv.org/pdf/2507.07036)  

**Abstract**: Spatial phenomena often exhibit heterogeneity across spatial extents and in proximity, making them complex to model-especially in dynamic regions like ice shelves and sea ice. In this study, we address this challenge by exploring the linkages between sea ice retreat and Antarctic ice shelf (AIS) melt. Although atmospheric forcing and basal melting have been widely studied, the direct impact of sea ice retreat on AIS mass loss remains underexplored. Traditional models treat sea ice and AIS as separate systems. It limits their ability to capture localized linkages and cascading feedback. To overcome this, we propose Spatial-Link, a novel graph-based framework that quantifies spatial heterogeneity to capture linkages between sea ice retreat and AIS melt. Our method constructs a spatial graph using Delaunay triangulation of satellite-derived ice change matrices, where nodes represent regions of significant change and edges encode proximity and directional consistency. We extract and statistically validate linkage paths using breadth-first search and Monte Carlo simulations. Results reveal non-local, spatially heterogeneous coupling patterns, suggesting sea ice loss can initiate or amplify downstream AIS melt. Our analysis shows how sea ice retreat evolves over an oceanic grid and progresses toward ice shelves-establishing a direct linkage. To our knowledge, this is the first proposed methodology linking sea ice retreat to AIS melt. Spatial-Link offers a scalable, data-driven tool to improve sea-level rise projections and inform climate adaptation strategies. 

**Abstract (ZH)**: 空间现象在不同空间尺度和临近区域表现出异质性，使其在建模时尤为复杂，尤其是在像冰shelf和海冰这样的动态区域。本研究通过探讨海冰退缩与南极冰shelf (AIS) 融化之间的关联来应对这一挑战。尽管大气强迫和基底融化已被广泛研究，但海冰退缩直接对AIS质量损失的影响仍待进一步探索。传统模型将海冰和AIS视为独立系统，限制了它们捕捉局部关联和级联反馈的能力。为此，我们提出了一种称为Spatial-Link的新型图为基础的框架，该框架通过度量空间异质性来捕捉海冰退缩与AIS融化之间的关联。该方法使用卫星 derived 冰变化矩阵的 Delaunay 三角剖分构建空间图，节点代表显著变化的区域，边编码临近性和方向一致性。我们通过广度优先搜索和蒙特卡洛模拟提取并统计验证关联路径。结果揭示了非局部的空间异质性耦合模式，表明海冰损失可以引发或放大下游的AIS融化。我们的分析展示了海冰退缩在海洋网格中的演变过程，并逐步向冰shelf推进，建立了直接的关联。据我们所知，这是首次提出的将海冰退缩与AIS融化联系起来的方法。Spatial-Link提供了可扩展的数据驱动工具，以提高海平面上升预测，并为气候适应策略提供指导。 

---
# Surrogate Model for Heat Transfer Prediction in Impinging Jet Arrays using Dynamic Inlet/Outlet and Flow Rate Control 

**Title (ZH)**: 基于动态进口/出口控制和流量率调节的喷射平板阵列热传递预测代理模型 

**Authors**: Mikael Vaillant, Victor Oliveira Ferreira, Wiebke Mainville, Jean-Michel Lamarre, Vincent Raymond, Moncef Chioua, Bruno Blais  

**Link**: [PDF](https://arxiv.org/pdf/2507.07034)  

**Abstract**: This study presents a surrogate model designed to predict the Nusselt number distribution in an enclosed impinging jet arrays, where each jet function independently and where jets can be transformed from inlets to outlets, leading to a vast number of possible flow arrangements. While computational fluid dynamics (CFD) simulations can model heat transfer with high fidelity, their cost prohibits real-time application such as model-based temperature control. To address this, we generate a CNN-based surrogate model that can predict the Nusselt distribution in real time. We train it with data from implicit large eddy computational fluid dynamics simulations (Re < 2,000). We train two distinct models, one for a five by one array of jets (83 simulations) and one for a three by three array of jets (100 simulations). We introduce a method to extrapolate predictions to higher Reynolds numbers (Re < 10,000) using a correlation-based scaling. The surrogate models achieve high accuracy, with a normalized mean average error below 2% on validation data for the five by one surrogate model and 0.6% for the three by three surrogate model. Experimental validation confirms the model's predictive capabilities. This work provides a foundation for model-based control strategies in advanced thermal management applications. 

**Abstract (ZH)**: 本研究提出了一种代理模型，用于预测封闭喷射阵列中努塞尔数分布，其中每个喷射独立作用，喷射可以从进气口转变为出气口，导致大量可能的流场布置。虽然计算流体动力学（CFD）模拟可以高保真地建模热传递，但其成本限制了实时应用，如基于模型的温度控制。为此，我们生成了一种基于CNN的代理模型，可以实现喷射阵列努塞尔数的实时预测。我们使用Re < 2,000的隐式大涡模拟数据进行训练，分别训练了五喷射一排（83次仿真）和三喷射三排（100次仿真）的两类模型。我们提出了一种基于相关性的扩展方法，用于将预测扩展到更高的雷诺数（Re < 10,000）。代理模型在验证数据上的归一化均方误差低于2%（五喷射一排模型）和0.6%（三喷射三排模型）。实验验证证实了模型的预测能力。本研究为先进热管理应用程序中的基于模型的控制策略奠定了基础。 

---
# PLAME: Leveraging Pretrained Language Models to Generate Enhanced Protein Multiple Sequence Alignments 

**Title (ZH)**: PLAME: 利用预训练语言模型生成增强的蛋白质多序列比对 

**Authors**: Hanqun Cao, Xinyi Zhou, Zijun Gao, Chenyu Wang, Xin Gao, Zhi Zhang, Chunbin Gu, Ge Liu, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2507.07032)  

**Abstract**: Protein structure prediction is essential for drug discovery and understanding biological functions. While recent advancements like AlphaFold have achieved remarkable accuracy, most folding models rely heavily on multiple sequence alignments (MSAs) to boost prediction performance. This dependency limits their effectiveness on low-homology proteins and orphan proteins, where MSA information is sparse or unavailable. To address this limitation, we propose PLAME, a novel MSA design model that leverages evolutionary embeddings from pretrained protein language models. Unlike existing methods, PLAME introduces pretrained representations to enhance evolutionary information and employs a conservation-diversity loss to enhance generation quality. Additionally, we propose a novel MSA selection method to effectively screen high-quality MSAs and improve folding performance. We also propose a sequence quality assessment metric that provides an orthogonal perspective to evaluate MSA quality. On the AlphaFold2 benchmark of low-homology and orphan proteins, PLAME achieves state-of-the-art performance in folding enhancement and sequence quality assessment, with consistent improvements demonstrated on AlphaFold3. Ablation studies validate the effectiveness of the MSA selection method, while extensive case studies on various protein types provide insights into the relationship between AlphaFold's prediction quality and MSA characteristics. Furthermore, we demonstrate that PLAME can serve as an adapter achieving AlphaFold2-level accuracy with the ESMFold's inference speed. 

**Abstract (ZH)**: 蛋白质结构预测对于药物发现和理解生物功能至关重要。尽管AlphaFold等最近的进展取得了显著的准确性，大多数折叠模型依然高度依赖多序列比对（MSAs）以提高预测性能。这种依赖限制了它们在低同源性和孤儿蛋白上的效果，因为在这个领域MSA信息稀缺或不可用。为了解决这一局限性，我们提出PLAME，一种新颖的MSA设计模型，利用预训练蛋白语言模型的进化嵌入。与现有方法不同，PLAME引入了预训练表示以增强进化信息，并采用了保真度-多样性损失来提高生成质量。此外，我们还提出了一种新的MSA选择方法，以有效筛选高质量的MSA并提高折叠性能。我们还提出了一种序列质量评估指标，从另一种视角评估MSA质量。在AlphaFold2基准测试中，PLAME在低同源性和孤儿蛋白的折叠增强和序列质量评估中达到了最先进的性能，并在AlphaFold3上也表现出一致的改进。消融研究表明MSA选择方法的有效性，而广泛的蛋白质类型案例研究为AlphaFold预测质量和MSA特性之间的关系提供了见解。此外，我们证明PLAME可以作为适配器，以ESMFold的推理速度达到AlphaFold2级别的准确性。 

---
# Design and Implementation of an OCR-Powered Pipeline for Table Extraction from Invoices 

**Title (ZH)**: 基于OCR的技术发票表格提取管道的设计与实现 

**Authors**: Parshva Dhilankumar Patel  

**Link**: [PDF](https://arxiv.org/pdf/2507.07029)  

**Abstract**: This paper presents the design and development of an OCR-powered pipeline for efficient table extraction from invoices. The system leverages Tesseract OCR for text recognition and custom post-processing logic to detect, align, and extract structured tabular data from scanned invoice documents. Our approach includes dynamic preprocessing, table boundary detection, and row-column mapping, optimized for noisy and non-standard invoice formats. The resulting pipeline significantly improves data extraction accuracy and consistency, supporting real-world use cases such as automated financial workflows and digital archiving. 

**Abstract (ZH)**: 基于OCR的发票表格高效提取管道设计与开发 

---
# Generating Multi-Table Time Series EHR from Latent Space with Minimal Preprocessing 

**Title (ZH)**: 从潜在空间生成具有最小前置处理的多表时间序列EHR 

**Authors**: Eunbyeol Cho, Jiyoun Kim, Minjae Lee, Sungjin Park, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2507.06996)  

**Abstract**: Electronic Health Records (EHR) are time-series relational databases that record patient interactions and medical events over time, serving as a critical resource for healthcare research and applications. However, privacy concerns and regulatory restrictions limit the sharing and utilization of such sensitive data, necessitating the generation of synthetic EHR datasets. Unlike previous EHR synthesis methods, which typically generate medical records consisting of expert-chosen features (e.g. a few vital signs or structured codes only), we introduce RawMed, the first framework to synthesize multi-table, time-series EHR data that closely resembles raw EHRs. Using text-based representation and compression techniques, RawMed captures complex structures and temporal dynamics with minimal preprocessing. We also propose a new evaluation framework for multi-table time-series synthetic EHRs, assessing distributional similarity, inter-table relationships, temporal dynamics, and privacy. Validated on two open-source EHR datasets, RawMed outperforms baseline models in fidelity and utility. The code is available at this https URL. 

**Abstract (ZH)**: 电子健康记录(EHR)是时间序列关系数据库，记录了患者的交互和随时间变化的医疗事件，是 Healthcare 研究和应用中的关键资源。然而，隐私担忧和监管限制限制了这类敏感数据的共享和利用，因此需要生成合成的 EHR 数据集。与以往通常生成由专家选择特征（如少数生命体征或结构化代码）构成的医疗记录的方法不同，我们介绍了 RawMed，这是第一个用于生成多表时间序列 EHR 数据的框架，这些数据与原始 EHR 数据高度相似。利用基于文本的表示和压缩技术，RawMed 在最少预处理的情况下捕捉到了复杂的结构和时序动态。我们还提出了一种新的多表时间序列合成 EHR 的评估框架，评估分布相似性、表间关系、时序动态和隐私特性。在两个开源 EHR 数据集上验证，RawMed 在保真度和实用性上均优于基准模型。代码可在以下网址获取。 

---
# Cross-Modality Masked Learning for Survival Prediction in ICI Treated NSCLC Patients 

**Title (ZH)**: ICI治疗NSCLC患者跨模态掩蔽学习生存预测 

**Authors**: Qilong Xing, Zikai Song, Bingxin Gong, Lian Yang, Junqing Yu, Wei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06994)  

**Abstract**: Accurate prognosis of non-small cell lung cancer (NSCLC) patients undergoing immunotherapy is essential for personalized treatment planning, enabling informed patient decisions, and improving both treatment outcomes and quality of life. However, the lack of large, relevant datasets and effective multi-modal feature fusion strategies pose significant challenges in this domain. To address these challenges, we present a large-scale dataset and introduce a novel framework for multi-modal feature fusion aimed at enhancing the accuracy of survival prediction. The dataset comprises 3D CT images and corresponding clinical records from NSCLC patients treated with immune checkpoint inhibitors (ICI), along with progression-free survival (PFS) and overall survival (OS) data. We further propose a cross-modality masked learning approach for medical feature fusion, consisting of two distinct branches, each tailored to its respective modality: a Slice-Depth Transformer for extracting 3D features from CT images and a graph-based Transformer for learning node features and relationships among clinical variables in tabular data. The fusion process is guided by a masked modality learning strategy, wherein the model utilizes the intact modality to reconstruct missing components. This mechanism improves the integration of modality-specific features, fostering more effective inter-modality relationships and feature interactions. Our approach demonstrates superior performance in multi-modal integration for NSCLC survival prediction, surpassing existing methods and setting a new benchmark for prognostic models in this context. 

**Abstract (ZH)**: 非小细胞肺癌（NSCLC）患者在接受免疫疗法后的准确预后对于个性化治疗规划、支持患者的知情决策以及提高治疗效果和生活质量至关重要。然而，缺乏相关的大型数据集和有效的多模态特征融合策略在这一领域构成了重大挑战。为进一步应对这些挑战，我们提出了一项大规模数据集，并引入了一种新的多模态特征融合框架，旨在提高生存预测的准确性。该数据集包含接受免疫检查点抑制剂（ICI）治疗的NSCLC患者的3D CT图像及其临床记录，并提供了无进展生存期（PFS）和总生存期（OS）数据。此外，我们还提出了一种跨模态遮蔽学习方法，用于医学特征融合，该方法由两个专门针对各自模态的分支组成：卷积深度变压器（Slice-Depth Transformer）用于从CT图像中提取3D特征，以及基于图的变压器用于学习表格数据中临床变量的节点特征及其关系。融合过程由遮蔽模态学习策略指导，模型利用完整模态来重建缺失的部分，从而改善了模态特异性特征的整合，促进更有效的跨模态关系和特征交互。我们的方法在NSCLC生存预测的多模态集成方面表现出色，超越了现有方法，并在这一领域为预后模型设立了新的基准。 

---
# Unifying Re-Identification, Attribute Inference, and Data Reconstruction Risks in Differential Privacy 

**Title (ZH)**: 统一辨识、属性推理和数据重构风险在差分隐私中的研究 

**Authors**: Bogdan Kulynych, Juan Felipe Gomez, Georgios Kaissis, Jamie Hayes, Borja Balle, Flavio du Pin Calmon, Jean Louis Raisaro  

**Link**: [PDF](https://arxiv.org/pdf/2507.06969)  

**Abstract**: Differentially private (DP) mechanisms are difficult to interpret and calibrate because existing methods for mapping standard privacy parameters to concrete privacy risks -- re-identification, attribute inference, and data reconstruction -- are both overly pessimistic and inconsistent. In this work, we use the hypothesis-testing interpretation of DP ($f$-DP), and determine that bounds on attack success can take the same unified form across re-identification, attribute inference, and data reconstruction risks. Our unified bounds are (1) consistent across a multitude of attack settings, and (2) tunable, enabling practitioners to evaluate risk with respect to arbitrary (including worst-case) levels of baseline risk. Empirically, our results are tighter than prior methods using $\varepsilon$-DP, Rényi DP, and concentrated DP. As a result, calibrating noise using our bounds can reduce the required noise by 20% at the same risk level, which yields, e.g., more than 15pp accuracy increase in a text classification task. Overall, this unifying perspective provides a principled framework for interpreting and calibrating the degree of protection in DP against specific levels of re-identification, attribute inference, or data reconstruction risk. 

**Abstract (ZH)**: 不同的私密性（Differentially Private, DP）机制难以解释和校准，因为现有将标准隐私参数映射到具体隐私风险（重识别、属性推理和数据重构）的方法既过于悲观又不一致。在本工作中，我们采用DP的假设检验解释（$f$-DP），并确定这些攻击成功的边界可以以统一的形式应用于重识别、属性推理和数据重构风险。我们的统一边界是（1）在多种攻击设置下保持一致，（2）可调节，使实践者能够根据任意（包括最坏情况）基线风险水平评估风险。实验结果表明，我们的方法比使用$\varepsilon$-DP、Rényi DP和集中DP的先前方法更为严格。因此，使用我们的边界校准噪声可以使在相同风险水平下所需的噪声减少20%，例如，在文本分类任务中这可以带来超过15个百分点的准确率提高。总体而言，这种统一的观点提供了一个原理性的框架，用于解释和校准DP在特定重识别、属性推理或数据重构风险水平下的保护程度。 

---
# Noisy PDE Training Requires Bigger PINNs 

**Title (ZH)**: 噪声偏微分方程训练需要更大的PINNs 

**Authors**: Sebastien Andre-Sloan, Anirbit Mukherjee, Matthew Colbrook  

**Link**: [PDF](https://arxiv.org/pdf/2507.06967)  

**Abstract**: Physics-Informed Neural Networks (PINNs) are increasingly used to approximate solutions of partial differential equations (PDEs), especially in high dimensions. In real-world applications, data samples are noisy, so it is important to know when a predictor can still achieve low empirical risk. However, little is known about the conditions under which a PINN can do so effectively. We prove a lower bound on the size of neural networks required for the supervised PINN empirical risk to fall below the variance of noisy supervision labels. Specifically, if a predictor achieves an empirical risk $O(\eta)$ below $\sigma^2$ (variance of supervision data), then necessarily $d_N\log d_N\gtrsim N_s \eta^2$, where $N_s$ is the number of samples and $d_N$ is the number of trainable parameters of the PINN. A similar constraint applies to the fully unsupervised PINN setting when boundary labels are sampled noisily. Consequently, increasing the number of noisy supervision labels alone does not provide a ``free lunch'' in reducing empirical risk. We also show empirically that PINNs can indeed achieve empirical risks below $\sigma^2$ under such conditions. As a case study, we investigate PINNs applied to the Hamilton--Jacobi--Bellman (HJB) PDE. Our findings lay the groundwork for quantitatively understanding the parameter requirements for training PINNs in the presence of noise. 

**Abstract (ZH)**: 物理知情神经网络（PINNs）在高维偏微分方程（PDEs）解的逼近中日益受到重视。在实际应用中，数据样本存在噪声，因此了解预测器何时仍能实现低经验风险至关重要。然而，关于PINN何时能有效实现这一点的具体条件知之甚少。我们证明了在监督PINN情形下，使其经验风险低于噪声监督标签方差所需的神经网络规模下界。具体而言，如果预测器的经验风险低于$\sigma^2$（监督数据方差）的$O(\eta)$，则必定有$d_N\log d_N\gtrsim N_s \eta^2$，其中$N_s$为样本数，$d_N$为PINN的可训练参数数。在边界标签噪声采样情况下，无监督PINN的类似约束同样适用。因此，单独增加噪声监督标签的数量并不能免费地降低经验风险。我们还通过实验表明，在这种条件下，PINNs确实能够实现低于$\sigma^2$的经验风险。作为案例研究，我们探讨了PINNs在哈密尔顿-雅可比-贝尔曼（HJB）偏微分方程中的应用。我们的研究为在噪声存在情况下训练PINNs的参数需求提供了定量理解的基础。 

---
# Beyond Connectivity: An Open Architecture for AI-RAN Convergence in 6G 

**Title (ZH)**: 超越连接性：面向6G的AI-RAN融合的开放架构 

**Authors**: Michele Polese, Niloofar Mohamadi, Salvatore D'Oro, Tommaso Melodia  

**Link**: [PDF](https://arxiv.org/pdf/2507.06911)  

**Abstract**: The proliferation of data-intensive Artificial Intelligence (AI) applications at the network edge demands a fundamental shift in RAN design, from merely consuming AI for network optimization, to actively enabling distributed AI workloads. This paradigm shift presents a significant opportunity for network operators to monetize AI at the edge while leveraging existing infrastructure investments. To realize this vision, this article presents a novel converged O-RAN and AI-RAN architecture that unifies orchestration and management of both telecommunications and AI workloads on shared infrastructure. The proposed architecture extends the Open RAN principles of modularity, disaggregation, and cloud-nativeness to support heterogeneous AI deployments. We introduce two key architectural innovations: (i) the AI-RAN Orchestrator, which extends the O-RAN Service Management and Orchestration (SMO) to enable integrated resource and allocation across RAN and AI workloads; and (ii) AI-RAN sites that provide distributed edge AI platforms with real-time processing capabilities. The proposed system supports flexible deployment options, allowing AI workloads to be orchestrated with specific timing requirements (real-time or batch processing) and geographic targeting. The proposed architecture addresses the orchestration requirements for managing heterogeneous workloads at different time scales while maintaining open, standardized interfaces and multi-vendor interoperability. 

**Abstract (ZH)**: 网络边缘数据密集型人工智能应用的普及要求在RAN设计上实现根本性的转变，从 merely 消费AI以优化网络，转变为积极地使能分布式AI工作负载。这种范式的转变为网络运营商利用现有基础设施投资在边缘 monetize AI 提供了重大机会。为了实现这一愿景，本文提出了一种新型的融合O-RAN和AI-RAN架构，该架构在共享基础设施上统一管理和协调电信和AI工作负载。所提出的架构将O-RAN的模块化、分解和云原生原则扩展到支持异构AI部署。我们介绍了两项关键架构创新：（i）AI-RAN协调器，该协调器扩展了O-RAN服务管理与协调（SMO）以在RAN和AI工作负载之间实现集成的资源管理和分配；（ii）AI-RAN站点，提供分布式边缘AI平台并具备实时处理能力。所提出的系统支持灵活的部署选项，允许AI工作负载根据具体的时间要求（实时或批处理）和地理目标进行协调。所提出的架构解决了在不同时间尺度上管理异构工作负载的协调需求，同时保持开放的标准接口和多供应商互操作性。 

---
# MIND: A Multi-agent Framework for Zero-shot Harmful Meme Detection 

**Title (ZH)**: MIND：零样本有害 meme 检测的多Agent框架 

**Authors**: Ziyan Liu, Chunxiao Fan, Haoran Lou, Yuexin Wu, Kaiwei Deng  

**Link**: [PDF](https://arxiv.org/pdf/2507.06908)  

**Abstract**: The rapid expansion of memes on social media has highlighted the urgent need for effective approaches to detect harmful content. However, traditional data-driven approaches struggle to detect new memes due to their evolving nature and the lack of up-to-date annotated data. To address this issue, we propose MIND, a multi-agent framework for zero-shot harmful meme detection that does not rely on annotated data. MIND implements three key strategies: 1) We retrieve similar memes from an unannotated reference set to provide contextual information. 2) We propose a bi-directional insight derivation mechanism to extract a comprehensive understanding of similar memes. 3) We then employ a multi-agent debate mechanism to ensure robust decision-making through reasoned arbitration. Extensive experiments on three meme datasets demonstrate that our proposed framework not only outperforms existing zero-shot approaches but also shows strong generalization across different model architectures and parameter scales, providing a scalable solution for harmful meme detection. The code is available at this https URL. 

**Abstract (ZH)**: 社交媒体上 meme 的快速扩张凸显了有效检测有害内容的迫切需求。然而，传统基于数据的方法由于 meme 的演变性质和缺乏最新的标注数据，在检测新 meme 方面遇到困难。为了解决这一问题，我们提出了 MIND，一种无需标注数据的多agent框架，用于零样本有害 meme 检测。MIND 实现了三种关键策略：1）从未标注参考集中检索相似的 meme 以提供上下文信息。2）提出了一种双向洞察提取机制，以提取相似 meme 的全面理解。3）然后通过有根据的仲裁使用多agent辩论机制以确保稳健的决策。在三个 meme 数据集上的 extensive 实验表明，我们提出的框架不仅优于现有的零样本方法，而且在不同的模型架构和参数规模下具有强大的泛化能力，提供了有害 meme 检测的可扩展解决方案。代码详见此链接。 

---
# SCoRE: Streamlined Corpus-based Relation Extraction using Multi-Label Contrastive Learning and Bayesian kNN 

**Title (ZH)**: SCoRE: 基于语料库的关系提取流水线方法：多标签对比学习与贝叶斯kNN 

**Authors**: Luca Mariotti, Veronica Guidetti, Federica Mandreoli  

**Link**: [PDF](https://arxiv.org/pdf/2507.06895)  

**Abstract**: The growing demand for efficient knowledge graph (KG) enrichment leveraging external corpora has intensified interest in relation extraction (RE), particularly under low-supervision settings. To address the need for adaptable and noise-resilient RE solutions that integrate seamlessly with pre-trained large language models (PLMs), we introduce SCoRE, a modular and cost-effective sentence-level RE system. SCoRE enables easy PLM switching, requires no finetuning, and adapts smoothly to diverse corpora and KGs. By combining supervised contrastive learning with a Bayesian k-Nearest Neighbors (kNN) classifier for multi-label classification, it delivers robust performance despite the noisy annotations of distantly supervised corpora. To improve RE evaluation, we propose two novel metrics: Correlation Structure Distance (CSD), measuring the alignment between learned relational patterns and KG structures, and Precision at R (P@R), assessing utility as a recommender system. We also release Wiki20d, a benchmark dataset replicating real-world RE conditions where only KG-derived annotations are available. Experiments on five benchmarks show that SCoRE matches or surpasses state-of-the-art methods while significantly reducing energy consumption. Further analyses reveal that increasing model complexity, as seen in prior work, degrades performance, highlighting the advantages of SCoRE's minimal design. Combining efficiency, modularity, and scalability, SCoRE stands as an optimal choice for real-world RE applications. 

**Abstract (ZH)**: 基于外部语料库的知识图谱（KG）增强日益增长的需求促进了在低监督环境下关系提取（RE）研究的兴趣。为了应对能够与预训练大规模语言模型（PLMs）无缝集成的可适应且抗噪声的RE解决方案的需求，我们提出了SCoRE，一种模块化且成本效益高的句级RE系统。SCoRE支持PLM的轻松切换，无需微调，并能平滑适应多种语料库和知识图谱。通过结合监督对比学习和贝叶斯k近邻（kNN）分类器进行多标签分类，尽管远程监督语料库带有噪声标注，仍能实现稳健的性能。为了改进RE评估，我们提出两种新的评估指标：关联结构距离（CSD），衡量学习到的关系模式与KG结构之间的对齐程度；以及R召回率（P@R），评估其作为推荐系统的实用性。我们还发布了Wiki20d，这是一个基准数据集，模仿了只有KG衍生的标注的真实世界RE场景。在五个基准上的实验结果显示，SCoRE在显著降低能耗的同时匹配或超越了最先进的方法。进一步的分析表明，如先前工作所见的增加模型复杂度会降低性能，突显了SCoRE简约设计的优势。结合效率、模块化和可扩展性，SCoRE是实际应用中理想的RE解决方案。 

---
# Developing and Maintaining an Open-Source Repository of AI Evaluations: Challenges and Insights 

**Title (ZH)**: 开发和维护一个开源AI评估仓库：挑战与见解 

**Authors**: Alexandra Abbas, Celia Waggoner, Justin Olive  

**Link**: [PDF](https://arxiv.org/pdf/2507.06893)  

**Abstract**: AI evaluations have become critical tools for assessing large language model capabilities and safety. This paper presents practical insights from eight months of maintaining $inspect\_evals$, an open-source repository of 70+ community-contributed AI evaluations. We identify key challenges in implementing and maintaining AI evaluations and develop solutions including: (1) a structured cohort management framework for scaling community contributions, (2) statistical methodologies for optimal resampling and cross-model comparison with uncertainty quantification, and (3) systematic quality control processes for reproducibility. Our analysis reveals that AI evaluation requires specialized infrastructure, statistical rigor, and community coordination beyond traditional software development practices. 

**Abstract (ZH)**: AI评估已成为评估大型语言模型能力和安全性的关键工具。本文基于维护八个月的开源AI评估库$inspect\_evals$（包含70多个社区贡献的AI评估），提供了实用见解，并识别出实施和维护AI评估的关键挑战，提出了包括：（1）面向社区贡献的结构化群体管理框架，（2）最优重采样和跨模型比较的统计方法，包含不确定性量化，以及（3）可重复性的系统质量控制流程。我们的分析表明，AI评估需要专门的基础设施、统计严谨性和社区协调，超越传统的软件开发实践。 

---
# A Single-Point Measurement Framework for Robust Cyber-Attack Diagnosis in Smart Microgrids Using Dual Fractional-Order Feature Analysis 

**Title (ZH)**: 基于双分数阶特征分析的智能微电网稳健网络攻击诊断单一测量点框架 

**Authors**: Yifan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06890)  

**Abstract**: Cyber-attacks jeopardize the safe operation of smart microgrids. At the same time, existing diagnostic methods either depend on expensive multi-point instrumentation or stringent modelling assumptions that are untenable under single-sensor constraints. This paper proposes a Fractional-Order Memory-Enhanced Attack-Diagnosis Scheme (FO-MADS) that achieves low-latency fault localisation and cyber-attack detection using only one VPQ (Voltage-Power-Reactive-power) sensor. FO-MADS first constructs a dual fractional-order feature library by jointly applying Caputo and Grünwald-Letnikov derivatives, thereby amplifying micro-perturbations and slow drifts in the VPQ signal. A two-stage hierarchical classifier then pinpoints the affected inverter and isolates the faulty IGBT switch, effectively alleviating class imbalance. Robustness is further strengthened through Progressive Memory-Replay Adversarial Training (PMR-AT), whose attack-aware loss is dynamically re-weighted via Online Hard Example Mining (OHEM) to prioritise the most challenging samples. Experiments on a four-inverter microgrid testbed comprising 1 normal and 24 fault classes under four attack scenarios demonstrate diagnostic accuracies of 96.6 % (bias), 94.0 % (noise), 92.8 % (data replacement), and 95.7 % (replay), while sustaining 96.7 % under attack-free conditions. These results establish FO-MADS as a cost-effective and readily deployable solution that markedly enhances the cyber-physical resilience of smart microgrids. 

**Abstract (ZH)**: 基于分数阶记忆增强的低延迟故障与网络攻击诊断方案（FO-MADS） 

---
# Winning and losing with Artificial Intelligence: What public discourse about ChatGPT tells us about how societies make sense of technological change 

**Title (ZH)**: 人工智能的胜败之道：关于ChatGPT的公共 discourse 告诉我们社会如何理解技术变革 

**Authors**: Adrian Rauchfleisch, Joshua Philip Suarez, Nikka Marie Sales, Andreas Jungherr  

**Link**: [PDF](https://arxiv.org/pdf/2507.06876)  

**Abstract**: Public product launches in Artificial Intelligence can serve as focusing events for collective attention, surfacing how societies react to technological change. Social media provide a window into the sensemaking around these events, surfacing hopes and fears and showing who chooses to engage in the discourse and when. We demonstrate that public sensemaking about AI is shaped by economic interests and cultural values of those involved. We analyze 3.8 million tweets posted by 1.6 million users across 117 countries in response to the public launch of ChatGPT in 2022. Our analysis shows how economic self-interest, proxied by occupational skill types in writing, programming, and mathematics, and national cultural orientations, as measured by Hofstede's individualism, uncertainty avoidance, and power distance dimensions, shape who speaks, when they speak, and their stance towards ChatGPT. Roles requiring more technical skills, such as programming and mathematics, tend to engage earlier and express more positive stances, whereas writing-centric occupations join later with greater skepticism. At the cultural level, individualism predicts both earlier engagement and a more negative stance, and uncertainty avoidance reduces the prevalence of positive stances but does not delay when users first engage with ChatGPT. Aggregate sentiment trends mask the dynamics observed in our study. The shift toward a more critical stance towards ChatGPT over time stems primarily from the entry of more skeptical voices rather than a change of heart among early adopters. Our findings underscore the importance of both the occupational background and cultural context in understanding public reactions to AI. 

**Abstract (ZH)**: 人工智能领域的公共产品发布可以作为群体注意力集中的事件，揭示社会对技术变革的反应。社交媒体提供了这些事件感知含义的窗口，揭示了人们的希望与恐惧，并展示了选择参与讨论的人及其时间。我们证明了人们对AI的公共感知受参与者经济利益和文化价值观的影响。我们分析了2022年ChatGPT公布后，来自117个国家的160万用户发布的380万条推文。分析显示，写作、编程和数学等职业技能类型以及以霍夫斯泰德个体主义、不确定性规避和权力距离维度衡量的国家文化取向，影响了谁发言、何时发言及其对ChatGPT的态度。需要更多技术技能的职位更早参与并表达更积极的态度，而以写作为中心的职业则后来参与并持更大疑虑的态度。在文化层面，个体主义预测了更早参与和更负面的态度，不确定性规避减少了积极态度的频率，但不影响用户首次参与ChatGPT的时间。总体情绪趋势掩盖了我们在研究中观察到的动力学。对ChatGPT的态度从更批判性转变为较悲观主要由更多怀疑论声音的进入而非早期采用者态度的改变所致。我们的研究强调了理解和解释公众对AI的反应时，职业背景和文化环境的重要性。 

---
# DiffSpectra: Molecular Structure Elucidation from Spectra using Diffusion Models 

**Title (ZH)**: DiffSpectra：使用扩散模型从光谱推断分子结构 

**Authors**: Liang Wang, Yu Rong, Tingyang Xu, Zhenyi Zhong, Zhiyuan Liu, Pengju Wang, Deli Zhao, Qiang Liu, Shu Wu, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06853)  

**Abstract**: Molecular structure elucidation from spectra is a foundational problem in chemistry, with profound implications for compound identification, synthesis, and drug development. Traditional methods rely heavily on expert interpretation and lack scalability. Pioneering machine learning methods have introduced retrieval-based strategies, but their reliance on finite libraries limits generalization to novel molecules. Generative models offer a promising alternative, yet most adopt autoregressive SMILES-based architectures that overlook 3D geometry and struggle to integrate diverse spectral modalities. In this work, we present DiffSpectra, a generative framework that directly infers both 2D and 3D molecular structures from multi-modal spectral data using diffusion models. DiffSpectra formulates structure elucidation as a conditional generation process. Its denoising network is parameterized by Diffusion Molecule Transformer, an SE(3)-equivariant architecture that integrates topological and geometric information. Conditioning is provided by SpecFormer, a transformer-based spectral encoder that captures intra- and inter-spectral dependencies from multi-modal spectra. Extensive experiments demonstrate that DiffSpectra achieves high accuracy in structure elucidation, recovering exact structures with 16.01% top-1 accuracy and 96.86% top-20 accuracy through sampling. The model benefits significantly from 3D geometric modeling, SpecFormer pre-training, and multi-modal conditioning. These results highlight the effectiveness of spectrum-conditioned diffusion modeling in addressing the challenge of molecular structure elucidation. To our knowledge, DiffSpectra is the first framework to unify multi-modal spectral reasoning and joint 2D/3D generative modeling for de novo molecular structure elucidation. 

**Abstract (ZH)**: 从光谱推断分子结构是化学中的一个基础问题，对于化合物识别、合成和药物开发具有深远影响。传统方法严重依赖专家解读，并缺乏可扩展性。创新型机器学习方法引入了检索策略，但依赖有限的数据库限制了其对新型分子的泛化能力。生成模型提供了一种有前景的替代方案，但大多数采用基于自回归SMILES的架构，忽视了三维几何结构，并难以整合多种光谱模态。在本工作中，我们提出了一种生成框架DiffSpectra，该框架利用扩散模型直接从多模态光谱数据推断出2D和3D分子结构。DiffSpectra将结构推断公式化为一个条件生成过程。其去噪网络由SE(3)-对称的扩散分子变压器参数化，整合了拓扑和几何信息。条件信息由基于变压器的光谱编码器SpecFormer提供，该编码器可以从多模态光谱中捕获体内和体间光谱依赖性。广泛的经验表明，DiffSpectra在结构推断中实现了高精度，通过采样获得精确结构的准确率为16.01%，前20位的准确率为96.86%。该模型显著受益于三维几何建模、SpecFormer预训练和多模态条件。这些结果突显了光谱条件下的扩散建模在分子结构推断挑战中的有效性。据我们所知，DiffSpectra是首个统一多模态光谱推理和联合2D/3D生成建模的框架，用于从头推断分子结构。 

---
# OpenDPDv2: A Unified Learning and Optimization Framework for Neural Network Digital Predistortion 

**Title (ZH)**: OpenDPDv2: 统一的神经网络数字预失真学习与优化框架 

**Authors**: Yizhuo Wu, Ang Li, Chang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2507.06849)  

**Abstract**: Neural network (NN)-based Digital Predistortion (DPD) stands out in improving signal quality in wideband radio frequency (RF) power amplifiers (PAs) employing complex modulation. However, NN DPDs usually rely on a large number of parameters for effective linearization and can significantly contribute to the energy consumption of the digital back-end in RF systems. This paper presents OpenDPDv2, a unified framework for PA modeling, DPD learning, and model optimization to reduce power consumption while maintaining high linearization performance. The optimization techniques feature a novel DPD algorithm, TRes-DeltaGRU, alongside two energy-efficient methods. The top-performing 32-bit floating-point (FP32) TRes-DeltaGRU-DPD model achieves an Adjacent Channel Power Ratio (ACPR) of -59.4 dBc and Error Vector Magnitude (EVM) of -42.1 dBc. By exploiting fixed-point quantization and dynamic temporal sparsity of input signals and hidden neurons, the inference energy of our model can be reduced by 4.5X while still maintaining -50.3 dBc ACPR and -35.2 dB EVM with 56% temporal sparsity. This was evaluated using a TM3.1a 200 MHz bandwidth 256-QAM OFDM signal applied to a 3.5 GHz GaN Doherty RF PA. OpenDPDv2 code, datasets, and documentation are publicly accessible at: this https URL. 

**Abstract (ZH)**: 基于神经网络的数字预失真（NN-DPD）在宽频带射频功率放大器（PA）采用复杂调制时显著提高了信号质量。然而，NN DPD通常需要大量的参数以实现有效的线性化，这会显著增加射频系统中数字后端的能耗。本文提出了一种统一框架OpenDPDv2，用于PA建模、DPD学习和模型优化，以降低能耗同时保持高线性化性能。优化技术包括一种新型DPD算法TRes-DeltaGRU以及两种能效方法。性能最佳的32位浮点数（FP32）TRes-DeltaGRU-DPD模型实现了-59.4 dBc的相邻通道功率比（ACPR）和-42.1 dBc的误差矢量幅度（EVM）。通过利用定点量化和输入信号及隐藏神经元的动态时域稀疏性，模型的推理能耗可降低4.5倍，同时保持-50.3 dBc的ACPR和-35.2 dB的EVM，其中时域稀疏性为56%。该研究使用带宽为200 MHz的TM3.1a 256-QAM OFDM信号应用于3.5 GHz GaN Doherty射频PA进行评估。OpenDPDv2代码、数据集和文档已公开访问：this https URL。 

---
# Comprehensive Evaluation of Prototype Neural Networks 

**Title (ZH)**: 原型神经网络综合评估 

**Authors**: Philipp Schlinge, Steffen Meinert, Martin Atzmueller  

**Link**: [PDF](https://arxiv.org/pdf/2507.06819)  

**Abstract**: Prototype models are an important method for explainable artificial intelligence (XAI) and interpretable machine learning. In this paper, we perform an in-depth analysis of a set of prominent prototype models including ProtoPNet, ProtoPool and PIPNet. For their assessment, we apply a comprehensive set of metrics. In addition to applying standard metrics from literature, we propose several new metrics to further complement the analysis of model interpretability. In our experimentation, we apply the set of prototype models on a diverse set of datasets including fine-grained classification, Non-IID settings and multi-label classification to further contrast the performance. Furthermore, we also provide our code as an open-source library, which facilitates simple application of the metrics itself, as well as extensibility - providing the option for easily adding new metrics and models. this https URL 

**Abstract (ZH)**: 原型模型是可解释人工智能(XAI)和可解释机器学习的重要方法。本文对ProtoPNet、ProtoPool和PIPNet等一组突出的原型模型进行了深入分析。为了评估这些模型，我们应用了一整套综合的评估指标。除了应用文献中的标准指标，我们还提出了几种新的指标，以进一步补充模型可解释性的分析。在我们的实验中，我们将原型模型应用于多种数据集，包括细粒度分类、非伊id设置和多标签分类，以进一步对比性能。此外，我们还提供了作为开源库的代码，该库不仅方便直接应用这些指标，还支持扩展，提供轻松添加新指标和模型的选项。 

---
# Intrinsic Training Signals for Federated Learning Aggregation 

**Title (ZH)**: 联邦学习聚合中的固有训练信号 

**Authors**: Cosimo Fiorini, Matteo Mosconi, Pietro Buzzega, Riccardo Salami, Simone Calderara  

**Link**: [PDF](https://arxiv.org/pdf/2507.06813)  

**Abstract**: Federated Learning (FL) enables collaborative model training across distributed clients while preserving data privacy. While existing approaches for aggregating client-specific classification heads and adapted backbone parameters require architectural modifications or loss function changes, our method uniquely leverages intrinsic training signals already available during standard optimization. We present LIVAR (Layer Importance and VARiance-based merging), which introduces: i) a variance-weighted classifier aggregation scheme using naturally emergent feature statistics, and ii) an explainability-driven LoRA merging technique based on SHAP analysis of existing update parameter patterns. Without any architectural overhead, LIVAR achieves state-of-the-art performance on multiple benchmarks while maintaining seamless integration with existing FL methods. This work demonstrates that effective model merging can be achieved solely through existing training signals, establishing a new paradigm for efficient federated model aggregation. The code will be made publicly available upon acceptance. 

**Abstract (ZH)**: 联邦学习 (FL) 允许在分布式客户端之间协作训练模型的同时保护数据隐私。我们的方法独特地利用了标准优化过程中固有可用的训练信号。我们提出了一种基于特征统计自然涌现的方差加权分类器聚合方案，并结合基于 SHAP 分析的可解释性驱动的 LoRA 聚合技术。不增加任何架构开销，LIVAR 在多个基准测试上实现了最先进的性能，同时保持与现有联邦学习方法的无缝集成。这项工作表明，有效的模型聚合仅通过现有训练信号即可实现，建立了联邦模型聚合的新范式。接受后代码将公开发布。 

---
# Towards Solving More Challenging IMO Problems via Decoupled Reasoning and Proving 

**Title (ZH)**: 通过解耦推理与证明走向解决更具有挑战性的国际数学奥林匹克问题 

**Authors**: Zhenwen Liang, Linfeng Song, Yang Li, Tao Yang, Feng Zhang, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.06804)  

**Abstract**: Automated Theorem Proving (ATP) in formal languages is a foundational challenge for AI. While Large Language Models (LLMs) have driven remarkable progress, a significant gap remains between their powerful informal reasoning capabilities and their weak formal proving performance. Recent studies show that the informal accuracy exceeds 80% while formal success remains below 8% on benchmarks like PutnamBench. We argue this gap persists because current state-of-the-art provers, by tightly coupling reasoning and proving, are trained with paradigms that inadvertently punish deep reasoning in favor of shallow, tactic-based strategies. To bridge this fundamental gap, we propose a novel framework that decouples high-level reasoning from low-level proof generation. Our approach utilizes two distinct, specialized models: a powerful, general-purpose Reasoner to generate diverse, strategic subgoal lemmas, and an efficient Prover to rigorously verify them. This modular design liberates the model's full reasoning potential and bypasses the pitfalls of end-to-end training. We evaluate our method on a challenging set of post-2000 IMO problems, a problem set on which no prior open-source prover has reported success. Our decoupled framework successfully solves 5 of these problems, demonstrating a significant step towards automated reasoning on exceptionally difficult mathematical challenges. To foster future research, we release our full dataset of generated and verified lemmas for a wide range of IMO problems, available at this https URL . 

**Abstract (ZH)**: 形式语言中的自动定理证明（ATP）是AI的基础挑战。尽管大型语言模型（LLMs）已经推动了显著的进步，但在其强大的非正式推理能力和薄弱的形式证明性能之间仍存在显著差距。最近的研究显示，在PutnamBench等基准测试中，非正式准确性超过80%，而形式成功率低于8%。我们argue这一差距持续存在，因为当前最先进的证明工具通过紧密耦合推理与证明，以无意中惩罚深层推理而偏好浅层、策略性的方法进行训练。为了弥合这一根本差距，我们提出了一种新的框架，将高层次推理与低层次证明生成解耦。我们的方法利用两种专门化的模型：一个强大的通用推理器（Reasoner）生成多样化的、战略性的子目标引理，以及一个高效的验证器（Prover）严格验证它们。这种模块化设计解放了模型的全部推理潜力，并绕过了端到端训练的陷阱。我们将在2000年后的IMO难题上评估我们的方法，这是迄今为止没有任何开源证明工具报告成功的难题集。我们的解耦框架成功解决了其中5个问题，展示了在处理极其复杂的数学挑战方面迈出的重要一步。为了促进未来研究，我们提供了涵盖广泛IMO问题的生成和验证引理的完整数据集，可通过以下链接访问：这个https URL。 

---
# Temporal Information Retrieval via Time-Specifier Model Merging 

**Title (ZH)**: 时间信息检索通过时间限定词模型合并 

**Authors**: SeungYoon Han, Taeho Hwang, Sukmin Cho, Soyeong Jeong, Hoyun Song, Huije Lee, Jong C. Park  

**Link**: [PDF](https://arxiv.org/pdf/2507.06782)  

**Abstract**: The rapid expansion of digital information and knowledge across structured and unstructured sources has heightened the importance of Information Retrieval (IR). While dense retrieval methods have substantially improved semantic matching for general queries, they consistently underperform on queries with explicit temporal constraints--often those containing numerical expressions and time specifiers such as ``in 2015.'' Existing approaches to Temporal Information Retrieval (TIR) improve temporal reasoning but often suffer from catastrophic forgetting, leading to reduced performance on non-temporal queries. To address this, we propose Time-Specifier Model Merging (TSM), a novel method that enhances temporal retrieval while preserving accuracy on non-temporal queries. TSM trains specialized retrievers for individual time specifiers and merges them in to a unified model, enabling precise handling of temporal constraints without compromising non-temporal retrieval. Extensive experiments on both temporal and non-temporal datasets demonstrate that TSM significantly improves performance on temporally constrained queries while maintaining strong results on non-temporal queries, consistently outperforming other baseline methods. Our code is available at this https URL . 

**Abstract (ZH)**: 数字信息和知识在结构化和非结构化来源中的迅速扩展突显了信息检索（IR）的重要性。虽然密集检索方法在普通查询的语义匹配上已有显著改进，但在具有明确时间约束的查询上表现不佳——这些查询往往包含数值表达和时间限定词，如“2015年”。现有的时间信息检索（TIR）方法虽然能提高时间推理能力，但往往会遭受灾难性遗忘的困扰，导致非时间查询的性能下降。为解决这一问题，我们提出了时间限定词模型融合（TSM），这是一种新颖的方法，能够在增强时间检索的同时保持非时间查询的准确性。TSM为个别时间限定词训练专门的检索器，并将它们合并到一个统一模型中，从而能够精确处理时间约束，而不损害非时间检索的性能。通过对时间和非时间数据集进行广泛实验，结果表明TSM在时间约束查询上显著提高了性能，并在非时间查询上保持了强劲的结果，始终优于其他基准方法。我们的代码可在该网址获取。 

---
# FOLC-Net: A Federated-Optimized Lightweight Architecture for Enhanced MRI Disease Diagnosis across Axial, Coronal, and Sagittal Views 

**Title (ZH)**: FOLC-Net：跨轴位、冠状位和矢状位增强MRI疾病诊断的优化轻量级联邦架构 

**Authors**: Saif Ur Rehman Khan, Muhammad Nabeel Asim, Sebastian Vollmer, Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2507.06763)  

**Abstract**: The framework is designed to improve performance in the analysis of combined as well as single anatomical perspectives for MRI disease diagnosis. It specifically addresses the performance degradation observed in state-of-the-art (SOTA) models, particularly when processing axial, coronal, and sagittal anatomical planes. The paper introduces the FOLC-Net framework, which incorporates a novel federated-optimized lightweight architecture with approximately 1.217 million parameters and a storage requirement of only 0.9 MB. FOLC-Net integrates Manta-ray foraging optimization (MRFO) mechanisms for efficient model structure generation, global model cloning for scalable training, and ConvNeXt for enhanced client adaptability. The model was evaluated on combined multi-view data as well as individual views, such as axial, coronal, and sagittal, to assess its robustness in various medical imaging scenarios. Moreover, FOLC-Net tests a ShallowFed model on different data to evaluate its ability to generalize beyond the training dataset. The results show that FOLC-Net outperforms existing models, particularly in the challenging sagittal view. For instance, FOLC-Net achieved an accuracy of 92.44% on the sagittal view, significantly higher than the 88.37% accuracy of study method (DL + Residual Learning) and 88.95% of DL models. Additionally, FOLC-Net demonstrated improved accuracy across all individual views, providing a more reliable and robust solution for medical image analysis in decentralized environments. FOLC-Net addresses the limitations of existing SOTA models by providing a framework that ensures better adaptability to individual views while maintaining strong performance in multi-view settings. The incorporation of MRFO, global model cloning, and ConvNeXt ensures that FOLC-Net performs better in real-world medical applications. 

**Abstract (ZH)**: 一种用于改善联合及单一解剖视角下MRI疾病诊断性能的框架：FOLC-Net及其应用 

---
# KAConvText: Novel Approach to Burmese Sentence Classification using Kolmogorov-Arnold Convolution 

**Title (ZH)**: KAConvText：基于柯尔莫戈洛夫-阿诺尔德卷积的缅甸语句分类新方法 

**Authors**: Ye Kyaw Thu, Thura Aung, Thazin Myint Oo, Thepchai Supnithi  

**Link**: [PDF](https://arxiv.org/pdf/2507.06753)  

**Abstract**: This paper presents the first application of Kolmogorov-Arnold Convolution for Text (KAConvText) in sentence classification, addressing three tasks: imbalanced binary hate speech detection, balanced multiclass news classification, and imbalanced multiclass ethnic language identification. We investigate various embedding configurations, comparing random to fastText embeddings in both static and fine-tuned settings, with embedding dimensions of 100 and 300 using CBOW and Skip-gram models. Baselines include standard CNNs and CNNs augmented with a Kolmogorov-Arnold Network (CNN-KAN). In addition, we investigated KAConvText with different classification heads - MLP and KAN, where using KAN head supports enhanced interpretability. Results show that KAConvText-MLP with fine-tuned fastText embeddings achieves the best performance of 91.23% accuracy (F1-score = 0.9109) for hate speech detection, 92.66% accuracy (F1-score = 0.9267) for news classification, and 99.82% accuracy (F1-score = 0.9982) for language identification. 

**Abstract (ZH)**: 本文首次将Kolmogorov-Arnold Convolution for Text (KAConvText) 应用到句子分类中，解决了三项任务：不平衡二分类仇恨言论检测、平衡多分类新闻分类以及不平衡多分类族裔语言识别。我们探讨了不同的嵌入配置，比较了随机嵌入和 fastText 嵌入在静态和微调设置下的表现，使用 CBOW 和 Skip-gram 模型，嵌入维度分别为 100 和 300。基线模型包括标准 CNN 和结合 Kolmogorov-Arnold 网络的 CNN (CNN-KAN)。此外，我们还研究了使用不同分类头的 KAConvText — MLP 和 KAN，其中使用 KAN 头支持增强的可解释性。结果表明，使用微调的 fastText 嵌入的 KAConvText-MLP 在仇恨言论检测中达到了 91.23% 的准确率（F1 分数 = 0.9109），在新闻分类中达到了 92.66% 的准确率（F1 分数 = 0.9267），在语言识别中达到了 99.82% 的准确率（F1 分数 = 0.9982）。 

---
# Deep Disentangled Representation Network for Treatment Effect Estimation 

**Title (ZH)**: 深度解耦表示网络用于治疗效果估计 

**Authors**: Hui Meng, Keping Yang, Xuyu Peng, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.06650)  

**Abstract**: Estimating individual-level treatment effect from observational data is a fundamental problem in causal inference and has attracted increasing attention in the fields of education, healthcare, and public this http URL this work, we concentrate on the study of disentangled representation methods that have shown promising outcomes by decomposing observed covariates into instrumental, confounding, and adjustment factors. However, most of the previous work has primarily revolved around generative models or hard decomposition methods for covariates, which often struggle to guarantee the attainment of precisely disentangled factors. In order to effectively model different causal relationships, we propose a novel treatment effect estimation algorithm that incorporates a mixture of experts with multi-head attention and a linear orthogonal regularizer to softly decompose the pre-treatment variables, and simultaneously eliminates selection bias via importance sampling re-weighting techniques. We conduct extensive experiments on both public semi-synthetic and real-world production datasets. The experimental results clearly demonstrate that our algorithm outperforms the state-of-the-art methods focused on individual treatment effects. 

**Abstract (ZH)**: 从observational数据中估计个体水平的治疗效果是一个因果推断中的基本问题，近年来在教育、医疗和公共政策等领域引起了越来越多的关注。在本文中，我们专注于研究通过分解观测协变量为工具变量、干扰变量和调整变量的去耦表示方法。然而，大多数先前的工作主要集中在生成模型或硬分解方法上，这些方法往往难以保证精确地获得去耦变量。为了有效建模不同的因果关系，我们提出了一种新颖的治疗效果估计算法，该算法结合了专家混合与多头注意力机制和线性正交正则化器，以软性分解预处理变量，并通过重要性采样加权技术同时消除选择偏见。我们在公共半合成和真实世界生产数据集上进行了广泛实验。实验结果清楚地表明，我们的算法在专注于个体治疗效果的方法中表现更优。 

---
# Denoising Multi-Beta VAE: Representation Learning for Disentanglement and Generation 

**Title (ZH)**: 去噪多贝塔VAE：解缠表示学习与生成 

**Authors**: Anshuk Uppal, Yuhta Takida, Chieh-Hsin Lai, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2507.06613)  

**Abstract**: Disentangled and interpretable latent representations in generative models typically come at the cost of generation quality. The $\beta$-VAE framework introduces a hyperparameter $\beta$ to balance disentanglement and reconstruction quality, where setting $\beta > 1$ introduces an information bottleneck that favors disentanglement over sharp, accurate reconstructions. To address this trade-off, we propose a novel generative modeling framework that leverages a range of $\beta$ values to learn multiple corresponding latent representations. First, we obtain a slew of representations by training a single variational autoencoder (VAE), with a new loss function that controls the information retained in each latent representation such that the higher $\beta$ value prioritize disentanglement over reconstruction fidelity. We then, introduce a non-linear diffusion model that smoothly transitions latent representations corresponding to different $\beta$ values. This model denoises towards less disentangled and more informative representations, ultimately leading to (almost) lossless representations, enabling sharp reconstructions. Furthermore, our model supports sample generation without input images, functioning as a standalone generative model. We evaluate our framework in terms of both disentanglement and generation quality. Additionally, we observe smooth transitions in the latent spaces with respect to changes in $\beta$, facilitating consistent manipulation of generated outputs. 

**Abstract (ZH)**: 解缠和可解释的潜在表示通常会牺牲生成质量。$\beta$-VAE框架通过引入超参数$\beta$来平衡解缠和重建质量，其中设置$\beta > 1$会导致信息瓶颈，倾向于解缠而牺牲锐度和准确的重建。为了解决这种权衡，我们提出了一种新型生成建模框架，该框架利用一系列$\beta$值学习多个相应的潜在表示。首先，通过训练单一的变分自编码器（VAE），并使用一种新损失函数来控制每个潜在表示保留的信息量，其中较高的$\beta$值优先考虑解缠而不是重建保真度。我们随后引入了一个非线性扩散模型，该模型平滑地过渡到不同$\beta$值对应的潜在表示。该模型向更不解缠但更具信息量的表示去噪，最终导致（几乎）无损的表示，从而实现锐利的重建。此外，我们的模型支持无需输入图像的样本生成，作为一种独立的生成模型运行。我们从解缠和生成质量两个方面评估了该框架。另外，我们观察到在$\beta$值变化时潜在空间中的平滑过渡，有助于生成输出的一致操作。 

---
# Learning controllable dynamics through informative exploration 

**Title (ZH)**: 通过信息性探索学习可控动力学 

**Authors**: Peter N. Loxley, Friedrich T. Sommer  

**Link**: [PDF](https://arxiv.org/pdf/2507.06582)  

**Abstract**: Environments with controllable dynamics are usually understood in terms of explicit models. However, such models are not always available, but may sometimes be learned by exploring an environment. In this work, we investigate using an information measure called "predicted information gain" to determine the most informative regions of an environment to explore next. Applying methods from reinforcement learning allows good suboptimal exploring policies to be found, and leads to reliable estimates of the underlying controllable dynamics. This approach is demonstrated by comparing with several myopic exploration approaches. 

**Abstract (ZH)**: 可控动力学环境通常通过显式模型来理解。然而，并非总是可以获取这样的模型，有时可以通过探索环境来学习这些模型。本文研究使用一种称为“预测信息增益”的信息量度量来确定下一个探索的最具信息性的环境区域。应用强化学习方法可以找到良好的次优探索策略，并能可靠地估计潜在的可控动力学。通过与几种短视探索方法的比较展示了这一方法。 

---
# Graph-based Fake Account Detection: A Survey 

**Title (ZH)**: 基于图的虚假账号检测：一个综述 

**Authors**: Ali Safarpoor Dehkordi, Ahad N. Zehmakan  

**Link**: [PDF](https://arxiv.org/pdf/2507.06541)  

**Abstract**: In recent years, there has been a growing effort to develop effective and efficient algorithms for fake account detection in online social networks. This survey comprehensively reviews existing methods, with a focus on graph-based techniques that utilise topological features of social graphs (in addition to account information, such as their shared contents and profile data) to distinguish between fake and real accounts. We provide several categorisations of these methods (for example, based on techniques used, input data, and detection time), discuss their strengths and limitations, and explain how these methods connect in the broader context. We also investigate the available datasets, including both real-world data and synthesised models. We conclude the paper by proposing several potential avenues for future research. 

**Abstract (ZH)**: 近年来，不断发展有效的在线社交网络虚假账号检测算法。本文全面回顾了现有方法，重点关注利用社交图的拓扑特征（除了账号信息，还包括其共享内容和资料数据）来区分虚假账号和真实账号的图基于技术。我们提供了这些方法的几种类别（例如，基于所使用的技术、输入数据和检测时间），讨论了它们的优势和局限性，并解释了这些方法在更广泛的背景下的联系。我们还研究了可用的数据集，包括真实世界数据和合成模型。本文最后提出了若干潜在的研究方向。 

---
# MoFE-Time: Mixture of Frequency Domain Experts for Time-Series Forecasting Models 

**Title (ZH)**: MoFE-Time: 频域专家混合的时间序列预测模型 

**Authors**: Yiwen Liu, Chenyu Zhang, Junjie Song, Siqi Chen, Sun Yin, Zihan Wang, Lingming Zeng, Yuji Cao, Junming Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2507.06502)  

**Abstract**: As a prominent data modality task, time series forecasting plays a pivotal role in diverse applications. With the remarkable advancements in Large Language Models (LLMs), the adoption of LLMs as the foundational architecture for time series modeling has gained significant attention. Although existing models achieve some success, they rarely both model time and frequency characteristics in a pretraining-finetuning paradigm leading to suboptimal performance in predictions of complex time series, which requires both modeling periodicity and prior pattern knowledge of signals. We propose MoFE-Time, an innovative time series forecasting model that integrates time and frequency domain features within a Mixture of Experts (MoE) network. Moreover, we use the pretraining-finetuning paradigm as our training framework to effectively transfer prior pattern knowledge across pretraining and finetuning datasets with different periodicity distributions. Our method introduces both frequency and time cells as experts after attention modules and leverages the MoE routing mechanism to construct multidimensional sparse representations of input signals. In experiments on six public benchmarks, MoFE-Time has achieved new state-of-the-art performance, reducing MSE and MAE by 6.95% and 6.02% compared to the representative methods Time-MoE. Beyond the existing evaluation benchmarks, we have developed a proprietary dataset, NEV-sales, derived from real-world business scenarios. Our method achieves outstanding results on this dataset, underscoring the effectiveness of the MoFE-Time model in practical commercial applications. 

**Abstract (ZH)**: MoFE-Time：一种结合时间与频率域特征的Mixture of Experts时间序列预测模型 

---
# Generative Lagrangian data assimilation for ocean dynamics under extreme sparsity 

**Title (ZH)**: 生成拉格朗日数据同化方法在极端稀疏情况下的海洋动力学应用 

**Authors**: Niloofar Asefi, Leonard Lupin-Jimenez, Tianning Wu, Ruoying He, Ashesh Chattopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2507.06479)  

**Abstract**: Reconstructing ocean dynamics from observational data is fundamentally limited by the sparse, irregular, and Lagrangian nature of spatial sampling, particularly in subsurface and remote regions. This sparsity poses significant challenges for forecasting key phenomena such as eddy shedding and rogue waves. Traditional data assimilation methods and deep learning models often struggle to recover mesoscale turbulence under such constraints. We leverage a deep learning framework that combines neural operators with denoising diffusion probabilistic models (DDPMs) to reconstruct high-resolution ocean states from extremely sparse Lagrangian observations. By conditioning the generative model on neural operator outputs, the framework accurately captures small-scale, high-wavenumber dynamics even at $99\%$ sparsity (for synthetic data) and $99.9\%$ sparsity (for real satellite observations). We validate our method on benchmark systems, synthetic float observations, and real satellite data, demonstrating robust performance under severe spatial sampling limitations as compared to other deep learning baselines. 

**Abstract (ZH)**: 从观测数据重构海洋动力学受到采样稀疏、不规则及拉格朗日性质的 fundamental 限制，尤其是在次表层和偏远区域。这种稀疏性对预测涡旋脱离和 rogue 波等关键现象构成了重大挑战。传统数据同化方法和深度学习模型在这些限制下通常难以恢复中尺度湍流。我们利用结合神经运算子与去噪扩散概率模型（DDPMs）的深度学习框架，从极其稀疏的拉格朗日观测数据中重构高分辨率的海洋状态。通过条件生成模型于神经运算子输出，该框架在合成数据的 99% 稀疏度和真实卫星观测的 99.9% 稀疏度下，准确捕捉小尺度、高波数动力学。我们在基准系统、合成浮标观测和真实卫星数据上验证了该方法，与其它深度学习基线相比，表现出较强的鲁棒性。 

---
# Foundation Model Self-Play: Open-Ended Strategy Innovation via Foundation Models 

**Title (ZH)**: 基础模型自对弈：基础模型驱动的开放式策略创新 

**Authors**: Aaron Dharna, Cong Lu, Jeff Clune  

**Link**: [PDF](https://arxiv.org/pdf/2507.06466)  

**Abstract**: Multi-agent interactions have long fueled innovation, from natural predator-prey dynamics to the space race. Self-play (SP) algorithms try to harness these dynamics by pitting agents against ever-improving opponents, thereby creating an implicit curriculum toward learning high-quality solutions. However, SP often fails to produce diverse solutions and can get stuck in locally optimal behaviors. We introduce Foundation-Model Self-Play (FMSP), a new direction that leverages the code-generation capabilities and vast knowledge of foundation models (FMs) to overcome these challenges by leaping across local optima in policy space. We propose a family of approaches: (1) \textbf{Vanilla Foundation-Model Self-Play (vFMSP)} continually refines agent policies via competitive self-play; (2) \textbf{Novelty-Search Self-Play (NSSP)} builds a diverse population of strategies, ignoring performance; and (3) the most promising variant, \textbf{Quality-Diveristy Self-Play (QDSP)}, creates a diverse set of high-quality policies by combining the diversity of NSSP and refinement of vFMSP. We evaluate FMSPs in Car Tag, a continuous-control pursuer-evader setting, and in Gandalf, a simple AI safety simulation in which an attacker tries to jailbreak an LLM's defenses. In Car Tag, FMSPs explore a wide variety of reinforcement learning, tree search, and heuristic-based methods, to name just a few. In terms of discovered policy quality, \ouralgo and vFMSP surpass strong human-designed strategies. In Gandalf, FMSPs can successfully automatically red-team an LLM, breaking through and jailbreaking six different, progressively stronger levels of defense. Furthermore, FMSPs can automatically proceed to patch the discovered vulnerabilities. Overall, FMSPs represent a promising new research frontier of improving self-play with foundation models, opening fresh paths toward more creative and open-ended strategy discovery 

**Abstract (ZH)**: 基于基础模型的自游戏（Foundation-Model Self-Play）：克服局部最优的多智能体创新方法 

---
# SoftSignSGD(S3): An Enhanced Optimizer for Practical DNN Training and Loss Spikes Minimization Beyond Adam 

**Title (ZH)**: SoftSignSGD(S3): 一种超越Adam的增强型优化器，用于实际DNN训练和损失峰值最小化 

**Authors**: Hanyang Peng, Shuang Qin, Yue Yu, Fangqing Jiang, Hui Wang, Wen Gao  

**Link**: [PDF](https://arxiv.org/pdf/2507.06464)  

**Abstract**: Adam has proven remarkable successful in training deep neural networks, but the mechanisms underlying its empirical successes and limitations remain underexplored. In this study, we demonstrate that the effectiveness of Adam stems largely from its similarity to SignSGD in robustly handling large gradient fluctuations, yet it is also vulnerable to destabilizing loss spikes due to its uncontrolled update scaling. To enhance the advantage of Adam and mitigate its limitation, we propose SignSoftSGD (S3), a novel optimizer with three key innovations. \emph{First}, S3 generalizes the sign-like update by employing a flexible $p$-th order momentum ($p \geq 1$) in the denominator, departing from the conventional second-order momentum (variance) preconditioning. This design enables enhanced performance while achieving stable training even with aggressive learning rates. \emph{Second}, S3 minimizes the occurrences of loss spikes through unified exponential moving average coefficients for numerator and denominator momenta, which inherently bound updates to $[-1, 1]$ and simplify hyperparameter tuning. \emph{Third}, S3 incorporates an equivalent Nesterov's accelerated gradient(NAG) module, accelerating convergence without memory overhead. Theoretically, we prove that S3 achieves the optimal convergence rate of $O\left(\frac{1}{T^{\sfrac{1}{4}}}\right)$ for general nonconvex stochastic optimization under weak assumptions. Extensive experiments across a range of vision and language tasks show that \textsf{\small S3} not only converges more rapidly and improves performance but also rarely experiences loss spikes, even with a \textbf{$\bm{10 \times}$} larger learning rate. In fact, S3 delivers performance comparable to or better than AdamW with \textbf{$2 \times$} the training steps, establishing its efficacy in both efficiency and final task performance. 

**Abstract (ZH)**: Adam在训练深度神经网络方面已证明极为成功，但其的经验成功和局限性的机制仍待深入探究。本研究展示了Adam的有效性主要源于它在稳健处理大梯度波动方面与SignSGD的相似性，但也因其不受控的更新缩放而容易受到损失峰值的破坏。为了增强Adam的优势并减轻其局限性，我们提出了SignSoftSGD (S3)，一种具有三大创新的新优化器。首先，S3通过在分母中采用灵活的p-阶动量($p \geq 1$)，扩展了类似符号的更新，脱离了传统的二阶动量（方差）预条件化。这一设计使得即使在使用激进的学习率时也能实现高性能和稳定的训练。其次，S3通过统一的指数移动平均系数来最小化损失峰值的发生次数，这内在地将更新限制在$[-1, 1]$范围内，并简化了超参数调整。第三，S3结合了等效的Nesterov加速梯度(NAG)模块，在不增加内存开销的情况下加速收敛。理论上，我们在较弱假设下证明了S3在一般非凸随机最优化下的最优收敛率为$O\left(\frac{1}{T^{\sfrac{1}{4}}}\right)$。广泛实验表明，\textsf{S3}不仅收敛更快且改善了性能，还很少经历损失峰值，即便使用$\bm{10 \times}$更大的学习率。事实上，S3在训练步数仅为AdamW的$\bm{2 \times}$的情况下，实现了与之相当或更好的性能，证明了其在效率和最终任务性能方面的有效性。 

---
# FedPhD: Federated Pruning with Hierarchical Learning of Diffusion Models 

**Title (ZH)**: 联邦普渡：基于分层学习的扩散模型剪枝 

**Authors**: Qianyu Long, Qiyuan Wang, Christos Anagnostopoulos, Daning Bi  

**Link**: [PDF](https://arxiv.org/pdf/2507.06449)  

**Abstract**: Federated Learning (FL), as a distributed learning paradigm, trains models over distributed clients' data. FL is particularly beneficial for distributed training of Diffusion Models (DMs), which are high-quality image generators that require diverse data. However, challenges such as high communication costs and data heterogeneity persist in training DMs similar to training Transformers and Convolutional Neural Networks. Limited research has addressed these issues in FL environments. To address this gap and challenges, we introduce a novel approach, FedPhD, designed to efficiently train DMs in FL environments. FedPhD leverages Hierarchical FL with homogeneity-aware model aggregation and selection policy to tackle data heterogeneity while reducing communication costs. The distributed structured pruning of FedPhD enhances computational efficiency and reduces model storage requirements in clients. Our experiments across multiple datasets demonstrate that FedPhD achieves high model performance regarding Fréchet Inception Distance (FID) scores while reducing communication costs by up to $88\%$. FedPhD outperforms baseline methods achieving at least a $34\%$ improvement in FID, while utilizing only $56\%$ of the total computation and communication resources. 

**Abstract (ZH)**: 联邦学习（FL）作为一种分布式学习范式，在分布式客户端数据上训练模型。FL特别适用于分布式训练扩散模型（DMs），这些高质量的图像生成器需要多种类的数据。然而，类似于训练Transformers和卷积神经网络，训练DMs时仍然存在高通信成本和数据异质性等挑战。现有研究在FL环境中尚未充分解决这些问题。为解决这一差距和挑战，我们提出了一种名为FedPhD的新型方法，旨在高效地在FL环境中训练DMs。FedPhD利用分层联邦学习与同质性感知模型聚合和选择策略来应对数据异质性并降低通信成本。FedPhD的分布式结构化剪枝增强了计算效率并减少了客户端的模型存储需求。我们的跨多个数据集的实验表明，FedPhD在提高弗雷谢特 inception 距离（FID）分数的同时，通信成本最多可降低88%。与基线方法相比，FedPhD在FID上至少提升了34%，而仅使用了总计算和通信资源的56%。 

---
# Can Interpretation Predict Behavior on Unseen Data? 

**Title (ZH)**: 解读能否预测未见数据上的行为？ 

**Authors**: Victoria R. Li, Jenny Kaufmann, Martin Wattenberg, David Alvarez-Melis, Naomi Saphra  

**Link**: [PDF](https://arxiv.org/pdf/2507.06445)  

**Abstract**: Interpretability research often aims to predict how a model will respond to targeted interventions on specific mechanisms. However, it rarely predicts how a model will respond to unseen input data. This paper explores the promises and challenges of interpretability as a tool for predicting out-of-distribution (OOD) model behavior. Specifically, we investigate the correspondence between attention patterns and OOD generalization in hundreds of Transformer models independently trained on a synthetic classification task. These models exhibit several distinct systematic generalization rules OOD, forming a diverse population for correlational analysis. In this setting, we find that simple observational tools from interpretability can predict OOD performance. In particular, when in-distribution attention exhibits hierarchical patterns, the model is likely to generalize hierarchically on OOD data -- even when the rule's implementation does not rely on these hierarchical patterns, according to ablation tests. Our findings offer a proof-of-concept to motivate further interpretability work on predicting unseen model behavior. 

**Abstract (ZH)**: 可解释性研究通常旨在预测模型在针对特定机制进行目标干预时的响应方式，但很少预测模型对未见过的输入数据的响应方式。本文探讨了可解释性作为一种工具，用于预测模型超出分布（OOD）行为的潜力和挑战。具体地，我们研究了几百个独立训练于合成分类任务上的Transformer模型的注意力模式与OOD泛化的对应关系。这些模型表现出若干种不同的系统性泛化规则，构成了一种多样的人群，便于进行相关性分析。在这种情况下，我们发现简单的可解释性观察工具可以预测模型的OOD性能。特别是，当分布内注意力表现出分层模式时，模型在处理OOD数据时很可能表现出分层的泛化能力——即使在消融测试中，规则的实现并不依赖于这些分层模式。我们的发现为推动进一步的可解释性研究，以预测未见过的模型行为提供了概念性验证。 

---
# Assessing the Prevalence of AI-assisted Cheating in Programming Courses: A Pilot Study 

**Title (ZH)**: 评估编程课程中AI辅助作弊的盛行情况：一项试点研究 

**Authors**: Kaléu Delphino  

**Link**: [PDF](https://arxiv.org/pdf/2507.06438)  

**Abstract**: Tools that can generate computer code in response to inputs written in natural language, such as ChatGPT, pose an existential threat to Computer Science education in its current form, since students can now use these tools to solve assignments without much effort. While that risk has already been recognized by scholars, the proportion of the student body that is incurring in this new kind of plagiarism is still an open problem. We conducted a pilot study in a large CS class (n=120) to assess the feasibility of estimating AI plagiarism through anonymous surveys and interviews. More than 25% of the survey respondents admitted to committing AI plagiarism. Conversely, only one student accepted to be interviewed. Given the high levels of misconduct acknowledgment, we conclude that surveys are an effective method for studies on the matter, while interviews should be avoided or designed in a way that can entice participation. 

**Abstract (ZH)**: 使用自然语言生成计算机代码的工具（如ChatGPT）对当前形式的计算机科学教育构成了存在性的威胁，因为学生现在可以轻易地使用这些工具完成作业。尽管这一风险已经被学者们认识到，但采用这种新形式抄袭的学生比例仍然是一个开放性问题。我们在一个大型CS班级（n=120）中进行了一项试点研究，以评估通过匿名调查和访谈估算AI抄袭的可行性。超过25%的调查 respondents 承认了AI抄袭行为。相反，只有一名学生同意接受访谈。鉴于不良行为承认率较高，我们得出结论，调查是研究该问题的有效方法，而访谈应避免或通过设计来鼓励参与。 

---
# Deprecating Benchmarks: Criteria and Framework 

**Title (ZH)**: 废止基准：标准与框架 

**Authors**: Ayrton San Joaquin, Rokas Gipiškis, Leon Staufer, Ariel Gil  

**Link**: [PDF](https://arxiv.org/pdf/2507.06434)  

**Abstract**: As frontier artificial intelligence (AI) models rapidly advance, benchmarks are integral to comparing different models and measuring their progress in different task-specific domains. However, there is a lack of guidance on when and how benchmarks should be deprecated once they cease to effectively perform their purpose. This risks benchmark scores over-valuing model capabilities, or worse, obscuring capabilities and safety-washing. Based on a review of benchmarking practices, we propose criteria to decide when to fully or partially deprecate benchmarks, and a framework for deprecating benchmarks. Our work aims to advance the state of benchmarking towards rigorous and quality evaluations, especially for frontier models, and our recommendations are aimed to benefit benchmark developers, benchmark users, AI governance actors (across governments, academia, and industry panels), and policy makers. 

**Abstract (ZH)**: 随着前沿人工智能模型的迅速发展，基准测试对于比较不同模型并在特定任务领域衡量其进步至关重要。然而，一旦基准测试不再有效执行其功能，关于何时以及如何废弃这些基准测试的指导仍然不足。这可能导致基准测试分数过度夸大模型能力，甚至更糟糕的是，掩盖这些能力并进行安全漂洗。基于基准测试实践的审查，我们提出了废弃完整或部分废弃基准测试的准则，并构建了废弃基准测试的框架。我们的工作旨在推动基准测试向严谨和高质量评估的发展，尤其是对于前沿模型，并且我们的建议旨在惠及基准测试开发者、基准测试用户、人工智能治理参与者（包括政府、学术界和产业委员会）以及决策者。 

---
# Bridging Data Gaps of Rare Conditions in ICU: A Multi-Disease Adaptation Approach for Clinical Prediction 

**Title (ZH)**: ICU中罕见疾病数据缺口的桥梁构建：多疾病适应性方法在临床预测中的应用 

**Authors**: Mingcheng Zhu, Yu Liu, Zhiyao Luo, Tingting Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.06432)  

**Abstract**: Artificial Intelligence has revolutionised critical care for common conditions. Yet, rare conditions in the intensive care unit (ICU), including recognised rare diseases and low-prevalence conditions in the ICU, remain underserved due to data scarcity and intra-condition heterogeneity. To bridge such gaps, we developed KnowRare, a domain adaptation-based deep learning framework for predicting clinical outcomes for rare conditions in the ICU. KnowRare mitigates data scarcity by initially learning condition-agnostic representations from diverse electronic health records through self-supervised pre-training. It addresses intra-condition heterogeneity by selectively adapting knowledge from clinically similar conditions with a developed condition knowledge graph. Evaluated on two ICU datasets across five clinical prediction tasks (90-day mortality, 30-day readmission, ICU mortality, remaining length of stay, and phenotyping), KnowRare consistently outperformed existing state-of-the-art models. Additionally, KnowRare demonstrated superior predictive performance compared to established ICU scoring systems, including APACHE IV and IV-a. Case studies further demonstrated KnowRare's flexibility in adapting its parameters to accommodate dataset-specific and task-specific characteristics, its generalisation to common conditions under limited data scenarios, and its rationality in selecting source conditions. These findings highlight KnowRare's potential as a robust and practical solution for supporting clinical decision-making and improving care for rare conditions in the ICU. 

**Abstract (ZH)**: 人工智能已革新常见重症疾病的护理，但在重症监护病房（ICU）中，由于数据稀疏和条件内异质性，罕见疾病的护理仍得不到充分服务。为弥补这些差距，我们开发了KnowRare，这是一种基于领域适应的深度学习框架，用于预测ICU中罕见疾病患者的临床结果。KnowRare通过自我监督预训练从多样化的电子健康记录中学习无条件的表示来缓解数据稀疏问题。它通过使用开发的条件知识图来选择性地适应临床相似条件的知识，来解决条件内的异质性。在五个临床预测任务（90天内死亡率、30天内再入院、ICU死亡率、剩余住院天数和分类诊断）上的两个ICU数据集上进行评估，KnowRare在所有任务中都优于现有的最先进的模型。此外，KnowRare在与APACHE IV和IV-a等现有ICU评分系统相比时，显示出了更出色的预测性能。案例研究进一步证明了KnowRare在适应特定数据集和任务特征时的灵活性、在数据稀少数量有限条件下对常见疾病的泛化能力以及其合理选择源条件的合理性。这些发现突显了KnowRare作为支持临床决策并改善ICU中罕见疾病护理的稳健且实用解决方案的潜力。 

---
# SImpHAR: Advancing impedance-based human activity recognition using 3D simulation and text-to-motion models 

**Title (ZH)**: SIMP HAR: 基于阻抗的人体活动识别方法的进展——使用3D模拟和文本到动作模型 

**Authors**: Lala Shakti Swarup Ray, Mengxi Liu, Deepika Gurung, Bo Zhou, Sungho Suh, Paul Lukowicz  

**Link**: [PDF](https://arxiv.org/pdf/2507.06405)  

**Abstract**: Human Activity Recognition (HAR) with wearable sensors is essential for applications in healthcare, fitness, and human-computer interaction. Bio-impedance sensing offers unique advantages for fine-grained motion capture but remains underutilized due to the scarcity of labeled data. We introduce SImpHAR, a novel framework addressing this limitation through two core contributions. First, we propose a simulation pipeline that generates realistic bio-impedance signals from 3D human meshes using shortest-path estimation, soft-body physics, and text-to-motion generation serving as a digital twin for data augmentation. Second, we design a two-stage training strategy with decoupled approach that enables broader activity coverage without requiring label-aligned synthetic data. We evaluate SImpHAR on our collected ImpAct dataset and two public benchmarks, showing consistent improvements over state-of-the-art methods, with gains of up to 22.3% and 21.8%, in terms of accuracy and macro F1 score, respectively. Our results highlight the promise of simulation-driven augmentation and modular training for impedance-based HAR. 

**Abstract (ZH)**: 穿戴传感器的人类活动识别（HAR）在医疗保健、健身和人机交互应用中至关重要。生物阻抗传感提供了精细动作捕捉的独特优势，但由于标记数据稀缺，其应用仍然不足。我们提出了SImpHAR，一种通过两个核心贡献解决这一问题的新型框架。首先，我们提出了一种仿真管道，通过最短路径估计、软体物理和文本到动作生成从三维人体网格生成真实生物阻抗信号，作为数据增强的数字双胞胎。其次，我们设计了一种解耦的两阶段训练策略，无需要求标签对齐的合成数据即可实现更广泛的活动覆盖。我们在自己的收集的ImpAct数据集以及两个公开基准上评估了SImpHAR，结果显示与最先进的方法相比，在准确性上提升高达22.3%，在宏F1分数上提升高达21.8%。我们的结果突显了基于仿真增强和模块化训练的阻抗基础HAR的潜力。 

---
# KPFlow: An Operator Perspective on Dynamic Collapse Under Gradient Descent Training of Recurrent Networks 

**Title (ZH)**: KPFlow: 反向传播训练循环网络过程中动态坍缩的一种算子视角 

**Authors**: James Hazelden, Laura Driscoll, Eli Shlizerman, Eric Shea-Brown  

**Link**: [PDF](https://arxiv.org/pdf/2507.06381)  

**Abstract**: Gradient Descent (GD) and its variants are the primary tool for enabling efficient training of recurrent dynamical systems such as Recurrent Neural Networks (RNNs), Neural ODEs and Gated Recurrent units (GRUs). The dynamics that are formed in these models exhibit features such as neural collapse and emergence of latent representations that may support the remarkable generalization properties of networks. In neuroscience, qualitative features of these representations are used to compare learning in biological and artificial systems. Despite recent progress, there remains a need for theoretical tools to rigorously understand the mechanisms shaping learned representations, especially in finite, non-linear models. Here, we show that the gradient flow, which describes how the model's dynamics evolve over GD, can be decomposed into a product that involves two operators: a Parameter Operator, K, and a Linearized Flow Propagator, P. K mirrors the Neural Tangent Kernel in feed-forward neural networks, while P appears in Lyapunov stability and optimal control theory. We demonstrate two applications of our decomposition. First, we show how their interplay gives rise to low-dimensional latent dynamics under GD, and, specifically, how the collapse is a result of the network structure, over and above the nature of the underlying task. Second, for multi-task training, we show that the operators can be used to measure how objectives relevant to individual sub-tasks align. We experimentally and theoretically validate these findings, providing an efficient Pytorch package, \emph{KPFlow}, implementing robust analysis tools for general recurrent architectures. Taken together, our work moves towards building a next stage of understanding of GD learning in non-linear recurrent models. 

**Abstract (ZH)**: 梯度下降（GD）及其变体是实现递归神经网络（RNNs）、神经ODE和门控递归单元（GRUs）等递归动态系统高效训练的主要工具。这些模型中的动力学特征包括神经崩溃和潜在表示的涌现，可能支持网络的出色泛化能力。在神经科学中，这些表示的定性特征被用于比较生物系统和人工系统的学习。尽管取得了近期的进步，但对于有限的非线性模型，仍需理论工具来严格理解塑造学习表示的机制。在这里，我们展示了梯度流如何可以被分解为涉及两个算子的乘积：参数算子K和线性化流传播算子P。K类似于前馈神经网络中的神经 tangent 核函数，而P出现在李亚普诺夫稳定性与最优控制理论中。我们展示了该分解的两个应用。首先，我们展示了它们的相互作用如何在梯度下降下导致低维度的潜在动态，并具体说明了网络结构如何导致崩溃，而不仅仅是任务的本质。其次，对于多任务训练，我们展示了这些算子可以用来衡量与各个子任务目标的对齐情况。我们通过实验和理论验证了这些发现，并提供了一个高效的Pytorch包KPFlow，实现了对通用递归架构的稳健分析工具。总体而言，我们的工作朝着理解非线性递归模型下梯度下降学习的下一阶段迈进。 

---
# Secure and Storage-Efficient Deep Learning Models for Edge AI Using Automatic Weight Generation 

**Title (ZH)**: 基于自动权重生成的边缘AI安全高效深度学习模型及存储优化 

**Authors**: Habibur Rahaman, Atri Chatterjee, Swarup Bhunia  

**Link**: [PDF](https://arxiv.org/pdf/2507.06380)  

**Abstract**: Complex neural networks require substantial memory to store a large number of synaptic weights. This work introduces WINGs (Automatic Weight Generator for Secure and Storage-Efficient Deep Learning Models), a novel framework that dynamically generates layer weights in a fully connected neural network (FC) and compresses the weights in convolutional neural networks (CNNs) during inference, significantly reducing memory requirements without sacrificing accuracy. WINGs framework uses principal component analysis (PCA) for dimensionality reduction and lightweight support vector regression (SVR) models to predict layer weights in the FC networks, removing the need for storing full-weight matrices and achieving substantial memory savings. It also preferentially compresses the weights in low-sensitivity layers of CNNs using PCA and SVR with sensitivity analysis. The sensitivity-aware design also offers an added level of security, as any bit-flip attack with weights in compressed layers has an amplified and readily detectable effect on accuracy. WINGs achieves 53x compression for the FC layers and 28x for AlexNet with MNIST dataset, and 18x for Alexnet with CIFAR-10 dataset with 1-2% accuracy loss. This significant reduction in memory results in higher throughput and lower energy for DNN inference, making it attractive for resource-constrained edge applications. 

**Abstract (ZH)**: WINGs（自动权重生成器，用于安全高效的深度学习模型压缩） 

---
# SymFlux: deep symbolic regression of Hamiltonian vector fields 

**Title (ZH)**: SymFlux：哈密顿向量场的深度符号回归 

**Authors**: M.A. Evangelista-Alvarado, P. Suárez-Serrato  

**Link**: [PDF](https://arxiv.org/pdf/2507.06342)  

**Abstract**: We present SymFlux, a novel deep learning framework that performs symbolic regression to identify Hamiltonian functions from their corresponding vector fields on the standard symplectic plane. SymFlux models utilize hybrid CNN-LSTM architectures to learn and output the symbolic mathematical expression of the underlying Hamiltonian. Training and validation are conducted on newly developed datasets of Hamiltonian vector fields, a key contribution of this work. Our results demonstrate the model's effectiveness in accurately recovering these symbolic expressions, advancing automated discovery in Hamiltonian mechanics. 

**Abstract (ZH)**: SymFlux：一种用于从标准辛平面的伴随向量场识别广义坐标函数的新型深度学习框架 

---
# MixAssist: An Audio-Language Dataset for Co-Creative AI Assistance in Music Mixing 

**Title (ZH)**: MixAssist：一个用于音乐混音协作式AI辅助的音频-语言数据集 

**Authors**: Michael Clemens, Ana Marasović  

**Link**: [PDF](https://arxiv.org/pdf/2507.06329)  

**Abstract**: While AI presents significant potential for enhancing music mixing and mastering workflows, current research predominantly emphasizes end-to-end automation or generation, often overlooking the collaborative and instructional dimensions vital for co-creative processes. This gap leaves artists, particularly amateurs seeking to develop expertise, underserved. To bridge this, we introduce MixAssist, a novel audio-language dataset capturing the situated, multi-turn dialogue between expert and amateur music producers during collaborative mixing sessions. Comprising 431 audio-grounded conversational turns derived from 7 in-depth sessions involving 12 producers, MixAssist provides a unique resource for training and evaluating audio-language models that can comprehend and respond to the complexities of real-world music production dialogues. Our evaluations, including automated LLM-as-a-judge assessments and human expert comparisons, demonstrate that fine-tuning models such as Qwen-Audio on MixAssist can yield promising results, with Qwen significantly outperforming other tested models in generating helpful, contextually relevant mixing advice. By focusing on co-creative instruction grounded in audio context, MixAssist enables the development of intelligent AI assistants designed to support and augment the creative process in music mixing. 

**Abstract (ZH)**: AI在音乐混音和母带处理工作流中的潜在价值虽显著，但现有研究主要侧重于端到端自动化或生成，往往忽视了协同创作过程中不可或缺的协作和指导维度。为填补这一空白，我们引入了MixAssist，一个新音频-语言数据集，捕捉了音乐制作专家与业余制作者在协同混音会话中进行的多轮对话。MixAssist包含源自7场深入会话（涉及12名制作者）的431个音频接地对话回合，为训练和评估能够理解并回应真实音乐制作对话复杂性的音频-语言模型提供了独特资源。我们的评估显示，使用MixAssist微调如Qwen-Audio等模型可以取得有前景的结果，其中Qwen在生成有助于混音并具有上下文相关性的建议方面明显优于其他测试模型。通过聚焦于基于音频上下文的协同创作指导，MixAssist促进了设计用于支持和增强音乐混音创作过程的智能AI助手的发展。 

---
# Sample-Efficient Reinforcement Learning Controller for Deep Brain Stimulation in Parkinson's Disease 

**Title (ZH)**: 帕金森病中基于深度脑刺激的样本效率强化学习控制器 

**Authors**: Harsh Ravivarapu, Gaurav Bagwe, Xiaoyong Yuan, Chunxiu Yu, Lan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06326)  

**Abstract**: Deep brain stimulation (DBS) is an established intervention for Parkinson's disease (PD), but conventional open-loop systems lack adaptability, are energy-inefficient due to continuous stimulation, and provide limited personalization to individual neural dynamics. Adaptive DBS (aDBS) offers a closed-loop alternative, using biomarkers such as beta-band oscillations to dynamically modulate stimulation. While reinforcement learning (RL) holds promise for personalized aDBS control, existing methods suffer from high sample complexity, unstable exploration in binary action spaces, and limited deployability on resource-constrained hardware.
We propose SEA-DBS, a sample-efficient actor-critic framework that addresses the core challenges of RL-based adaptive neurostimulation. SEA-DBS integrates a predictive reward model to reduce reliance on real-time feedback and employs Gumbel Softmax-based exploration for stable, differentiable policy updates in binary action spaces. Together, these components improve sample efficiency, exploration robustness, and compatibility with resource-constrained neuromodulatory hardware. We evaluate SEA-DBS on a biologically realistic simulation of Parkinsonian basal ganglia activity, demonstrating faster convergence, stronger suppression of pathological beta-band power, and resilience to post-training FP16 quantization. Our results show that SEA-DBS offers a practical and effective RL-based aDBS framework for real-time, resource-constrained neuromodulation. 

**Abstract (ZH)**: SEA-DBS: 一种高效的学习者-策略框架用于基于强化学习的自适应脑刺激 

---
# A Survey of Multi Agent Reinforcement Learning: Federated Learning and Cooperative and Noncooperative Decentralized Regimes 

**Title (ZH)**: 多智能体强化学习综述：联邦学习与合作与非合作去中心化机制 

**Authors**: Kemboi Cheruiyot, Nickson Kiprotich, Vyacheslav Kungurtsev, Kennedy Mugo, Vivian Mwirigi, Marvin Ngesa  

**Link**: [PDF](https://arxiv.org/pdf/2507.06278)  

**Abstract**: The increasing interest in research and innovation towards the development of autonomous agents presents a number of complex yet important scenarios of multiple AI Agents interacting with each other in an environment. The particular setting can be understood as exhibiting three possibly topologies of interaction - centrally coordinated cooperation, ad-hoc interaction and cooperation, and settings with noncooperative incentive structures. This article presents a comprehensive survey of all three domains, defined under the formalism of Federal Reinforcement Learning (RL), Decentralized RL, and Noncooperative RL, respectively. Highlighting the structural similarities and distinctions, we review the state of the art in these subjects, primarily explored and developed only recently in the literature. We include the formulations as well as known theoretical guarantees and highlights and limitations of numerical performance. 

**Abstract (ZH)**: 面向自主代理发展的研究与创新日益增加，提出了多个复杂但重要的场景，涉及多个AI代理在环境中的相互作用。该特定设置可以理解为表现出三种可能的交互拓扑结构——中心协调合作、即兴合作以及非合作激励结构。本文分别在联邦强化学习（RL）、去中心化RL和非合作RL的形式化定义下，对这三个领域进行了全面综述。强调其结构相似性和区别，回顾了这些主题的最新进展，主要是在文献中最近才被探索和发展的内容。包括相应的形式化表述、已知的理论保证以及数值性能的优缺点。 

---
# The Prompt War: How AI Decides on a Military Intervention 

**Title (ZH)**: AI 决策军事干预的prompt战争 

**Authors**: Maxim Chupilkin  

**Link**: [PDF](https://arxiv.org/pdf/2507.06277)  

**Abstract**: Which factors determine AI propensity for military intervention? While the use of AI in war games and military planning is growing exponentially, the simple analysis of key drivers embedded in the models has not yet been done. This paper does a simple conjoint experiment proposing a model to decide on military intervention in 640 vignettes where each was run for 100 times allowing to explore AI decision on military intervention systematically. The analysis finds that largest predictors of AI decision to intervene are high domestic support and high probability of success. Costs such as international condemnation, military deaths, civilian deaths, and negative economic effect are statistically significant, but their effect is around half of domestic support and probability of victory. Closing window of opportunity only reaches statistical significance in interaction with other factors. The results are remarkably consistent across scenarios and across different models (OpenAI GPT, Anthropic Claude, Google Gemini) suggesting a pattern in AI decision-making. 

**Abstract (ZH)**: 哪些因素决定了人工智能在军事干预中的倾向性？本文通过对640个情境进行100次运行的简要联合实验，提出了一个模型来决定在高国内支持和高成功概率的情况下进行军事干预。成本因素（如国际谴责、军事人员伤亡、平民伤亡和负面经济效益）在统计上是显著的，但其影响大约仅为国内支持和胜利概率的一半。机会窗口仅在与其他因素的交互作用中达到统计显著性。结果在不同场景和不同模型（OpenAI GPT、Anthropic Claude、Google Gemini）中表现出惊人的一致性，表明人工智能决策中存在模式。 

---
# Advancing Offline Handwritten Text Recognition: A Systematic Review of Data Augmentation and Generation Techniques 

**Title (ZH)**: 基于数据增强和生成技术的手写文本离线识别进展：一项系统性综述 

**Authors**: Yassin Hussein Rassul, Aram M. Ahmed, Polla Fattah, Bryar A. Hassan, Arwaa W. Abdulkareem, Tarik A. Rashid, Joan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2507.06275)  

**Abstract**: Offline Handwritten Text Recognition (HTR) systems play a crucial role in applications such as historical document digitization, automatic form processing, and biometric authentication. However, their performance is often hindered by the limited availability of annotated training data, particularly for low-resource languages and complex scripts. This paper presents a comprehensive survey of offline handwritten data augmentation and generation techniques designed to improve the accuracy and robustness of HTR systems. We systematically examine traditional augmentation methods alongside recent advances in deep learning, including Generative Adversarial Networks (GANs), diffusion models, and transformer-based approaches. Furthermore, we explore the challenges associated with generating diverse and realistic handwriting samples, particularly in preserving script authenticity and addressing data scarcity. This survey follows the PRISMA methodology, ensuring a structured and rigorous selection process. Our analysis began with 1,302 primary studies, which were filtered down to 848 after removing duplicates, drawing from key academic sources such as IEEE Digital Library, Springer Link, Science Direct, and ACM Digital Library. By evaluating existing datasets, assessment metrics, and state-of-the-art methodologies, this survey identifies key research gaps and proposes future directions to advance the field of handwritten text generation across diverse linguistic and stylistic landscapes. 

**Abstract (ZH)**: 线下手写文本识别（HTR）系统在历史文档数字化、自动表单处理和生物特征认证等应用中发挥着关键作用。然而，它们的性能常常受到标注训练数据有限的限制，特别是在低资源语言和复杂书写字体方面。本文综述了线下手写数据增强和生成技术，旨在提高HTR系统的准确性和鲁棒性。我们系统地考察了传统的增强方法以及近年来深度学习的最新进展，包括生成对抗网络（GANs）、扩散模型和基于变换器的方法。此外，我们探讨了生成多样且真实的书写样本所面临的挑战，特别是在保留书写字体真实性以及应对数据稀缺性方面的问题。本文遵循PRISMA方法论，确保了选择过程的结构化和严谨性。通过评估现有数据集、评估指标和最先进的方法，本文识别了关键的研究缺口，并提议了未来的研究方向，以推进跨多样化语言和风格景观的手写文本生成领域。 

---
# Magneto-radiative modelling and artificial neural network optimization of biofluid flow in a stenosed arterial domain 

**Title (ZH)**: 磁辐射建模与人工神经网络优化在狭窄动脉域内生物流体流动的研究 

**Authors**: S P Shivakumar, Gunisetty Ramasekhar, P Nimmy, Sujesh Areekara, L Thanuja, T V Smitha, S Devanathan, Ganesh R Naik, K V Nagaraja  

**Link**: [PDF](https://arxiv.org/pdf/2507.06273)  

**Abstract**: The increasing complexity of cardiovascular diseases and limitations in traditional healing methods mandate the invention of new drug delivery systems that assure targeted, effective, and regulated treatments, contributing directly to UN SDGs 3 and 9, thereby encouraging the utilization of sustainable medical technologies in healthcare. This study investigates the flow of a Casson-Maxwell nanofluid through a stenosed arterial domain. The quantities, such as skin friction and heat transfer rate, are analysed in detail. The Casson-Maxwell fluid shows a lower velocity profile than the Casson fluids, which indicates the improved residence time for efficient drug delivery. The heat transfer rate shows an increase with higher volume fractions of copper and aluminium oxide nanoparticles and a decrease with higher volume fractions of silver nanoparticles. The skin friction coefficient decreases by 219% with a unit increase in the Maxwell parameter, whereas it increases by 66.1% with a unit rise in the Casson parameter. This work supports SDGs 4 and 17 by fostering interdisciplinary learning and collaboration in fluid dynamics and healthcare innovation. Additionally, the rate of heat flow was forecasted (with an overall R-value of 0.99457) using the Levenberg-Marquardt backpropagation training scheme under the influence of magneto-radiative, linear heat source and Casson-Maxwell parameters along with the tri-metallic nanoparticle volume fractions. It is also observed that the drag coefficient is most sensitive to the changes in the Maxwell parameter. 

**Abstract (ZH)**: 心血管疾病复杂性的增加和传统治疗手段的局限性 necessitates 新型药物递送系统的发展，以实现精准、有效和受控的治疗，直接促进联合国可持续发展目标3和9，从而鼓励使用可持续医疗技术。本研究探讨了Casson-Maxwell纳米流体在狭窄动脉域中的流动，详细分析了皮肤摩擦和热量传输率等量度。Casson-Maxwell流体的流速剖面低于Casson流体，表明改善了药物递送的保留时间。热量传输率随着铜和铝氧化物纳米颗粒体积分数的增加而增加，随着银纳米颗粒体积分数的增加而减少。单位增加Maxwell参数导致皮肤摩擦系数降低219%，而单位增加Casson参数导致皮肤摩擦系数增加66.1%。本工作通过促进流体动力学和医疗创新的跨学科学习与合作来支持可持续发展目标4和17。此外，通过Levenberg-Marquardt反向传播训练方案预测了受磁辐射效应、线性热源和Casson-Maxwell参数及三元金属纳米颗粒体积分数影响下的热流速率（整体R值为0.99457）。同时观察到，阻力系数对Maxwell参数的变化最为敏感。 

---
# A Collectivist, Economic Perspective on AI 

**Title (ZH)**: 集体主义视角下的经济AI 

**Authors**: Michael I. Jordan  

**Link**: [PDF](https://arxiv.org/pdf/2507.06268)  

**Abstract**: Information technology is in the midst of a revolution in which omnipresent data collection and machine learning are impacting the human world as never before. The word "intelligence" is being used as a North Star for the development of this technology, with human cognition viewed as a baseline. This view neglects the fact that humans are social animals, and that much of our intelligence is social and cultural in origin. A related issue is that the current view treats the social consequences of technology as an afterthought. The path forward is not merely more data and compute, and not merely more attention paid to cognitive or symbolic representations, but a thorough blending of economic and social concepts with computational and inferential concepts, in the service of system-level designs in which social welfare is a first-class citizen, and with the aspiration that a new human-centric engineering field will emerge. 

**Abstract (ZH)**: 信息技术正经历一场革命，其中无所不在的数据收集和机器学习以前所未有的方式影响着人类世界。“智能”被用作这一技术发展的北极星，而人类认知被视为基准。这一观点忽视了人类是社会性动物的事实，以及我们许多智能源于社会和文化。相关的问题是，当前的观点把技术的社会后果当作次要考虑。前进的道路不仅需要更多的数据和计算，也不仅仅是更多地关注认知或符号表示，而是需要将经济和社会概念与计算和推理概念进行彻底融合，以服务于将社会福利作为头等大事的系统级设计，并希望一个以人类为中心的工程领域将会出现。 

---
# Machine Learning based Enterprise Financial Audit Framework and High Risk Identification 

**Title (ZH)**: 基于机器学习的企业财务审计框架与高风险识别 

**Authors**: Tingyu Yuan, Xi Zhang, Xuanjing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.06266)  

**Abstract**: In the face of global economic uncertainty, financial auditing has become essential for regulatory compliance and risk mitigation. Traditional manual auditing methods are increasingly limited by large data volumes, complex business structures, and evolving fraud tactics. This study proposes an AI-driven framework for enterprise financial audits and high-risk identification, leveraging machine learning to improve efficiency and accuracy. Using a dataset from the Big Four accounting firms (EY, PwC, Deloitte, KPMG) from 2020 to 2025, the research examines trends in risk assessment, compliance violations, and fraud detection. The dataset includes key indicators such as audit project counts, high-risk cases, fraud instances, compliance breaches, employee workload, and client satisfaction, capturing both audit behaviors and AI's impact on operations. To build a robust risk prediction model, three algorithms - Support Vector Machine (SVM), Random Forest (RF), and K-Nearest Neighbors (KNN) - are evaluated. SVM uses hyperplane optimization for complex classification, RF combines decision trees to manage high-dimensional, nonlinear data with resistance to overfitting, and KNN applies distance-based learning for flexible performance. Through hierarchical K-fold cross-validation and evaluation using F1-score, accuracy, and recall, Random Forest achieves the best performance, with an F1-score of 0.9012, excelling in identifying fraud and compliance anomalies. Feature importance analysis reveals audit frequency, past violations, employee workload, and client ratings as key predictors. The study recommends adopting Random Forest as a core model, enhancing features via engineering, and implementing real-time risk monitoring. This research contributes valuable insights into using machine learning for intelligent auditing and risk management in modern enterprises. 

**Abstract (ZH)**: 面向全球经济不确定性，财务审计已成为合规性和风险减轻的必备手段。传统的手工审计方法 increasingly受到大数据量、复杂的企业结构和不断变化的欺诈手段的限制。本研究提出了一种基于人工智能的企业财务审计和高风险识别框架，利用机器学习提高效率和准确性。通过使用从四大会计师事务所（安永、普华永道、德勤、毕马威）2020年至2025年的数据集，研究探讨了风险评估、合规违规和欺诈检测的趋势。数据集包括审计项目数量、高风险案例、欺诈事件、合规违规、员工工作量和客户满意度等关键指标，捕捉审计行为和AI对运营的影响。为构建 robust的风险预测模型，评估了三种算法——支持向量机（SVM）、随机森林（RF）和K最近邻（KNN）。SVM通过超平面优化进行复杂分类，RF结合决策树管理高维非线性数据并具有抗过拟合能力，KNN通过基于距离的学习实现灵活性能。通过层次K折交叉验证和使用F1分数、准确率和召回率评估，随机森林显示出最佳性能，F1分数为0.9012，特别擅长识别欺诈和合规异常。特征重要性分析表明，审计频率、过去违规、员工工作量和客户评分是关键预测因子。研究建议采用随机森林作为核心模型，通过特征工程增强，并实施实时风险监控。本研究为利用机器学习进行现代化企业的智能审计和风险管理提供了宝贵见解。 

---
# SPARC: Concept-Aligned Sparse Autoencoders for Cross-Model and Cross-Modal Interpretability 

**Title (ZH)**: SPARC: 概念对齐的稀疏自编码器以实现跨模型和跨模态可解释性 

**Authors**: Ali Nasiri-Sarvi, Hassan Rivaz, Mahdi S. Hosseini  

**Link**: [PDF](https://arxiv.org/pdf/2507.06265)  

**Abstract**: Understanding how different AI models encode the same high-level concepts, such as objects or attributes, remains challenging because each model typically produces its own isolated representation. Existing interpretability methods like Sparse Autoencoders (SAEs) produce latent concepts individually for each model, resulting in incompatible concept spaces and limiting cross-model interpretability. To address this, we introduce SPARC (Sparse Autoencoders for Aligned Representation of Concepts), a new framework that learns a single, unified latent space shared across diverse architectures and modalities (e.g., vision models like DINO, and multimodal models like CLIP). SPARC's alignment is enforced through two key innovations: (1) a Global TopK sparsity mechanism, ensuring all input streams activate identical latent dimensions for a given concept; and (2) a Cross-Reconstruction Loss, which explicitly encourages semantic consistency between models. On Open Images, SPARC dramatically improves concept alignment, achieving a Jaccard similarity of 0.80, more than tripling the alignment compared to previous methods. SPARC creates a shared sparse latent space where individual dimensions often correspond to similar high-level concepts across models and modalities, enabling direct comparison of how different architectures represent identical concepts without requiring manual alignment or model-specific analysis. As a consequence of this aligned representation, SPARC also enables practical applications such as text-guided spatial localization in vision-only models and cross-model/cross-modal retrieval. Code and models are available at this https URL. 

**Abstract (ZH)**: 理解不同AI模型如何编码相同的高层次概念（如对象或属性）仍然具有挑战性，因为每个模型通常会生成自己独立的表示。现有的可解释性方法，如稀疏自动编码器（SAEs），为每个模型单独生成潜在概念，导致不兼容的概念空间并限制跨模型的可解释性。为此，我们引入了SPARC（Sparse Autoencoders for Aligned Representation of Concepts）这一新框架，它可以在多种架构和模态（例如，视觉模型DINO和多模态模型CLIP）之间学习一个统一的潜在空间。SPARC的对齐通过两个关键创新来实现：（1）全局TopK稀疏性机制，确保所有输入流在给定概念下激活相同的潜在维度；（2）交叉重建损失，明确鼓励模型之间的语义一致性。在Open Images数据集上，SPARC显著提高了概念对齐度，实现0.80的杰卡德相似度，相比之前的方法提高了近三倍。SPARC创建了一个共享的稀疏潜在空间，在这个空间中，各个维度经常在不同模型和模态之间对应类似的高层次概念，从而可以在不需要手动对齐或模型特定分析的情况下直接比较不同架构如何表示相同的概念。由于这种对齐的表示，SPARC也使得一些实际应用成为可能，例如在纯视觉模型中实现受文本指导的空间定位以及跨模型/跨模态检索。代码和模型可从以下链接获取。 

---
# X-ray transferable polyrepresentation learning 

**Title (ZH)**: X射线转移可变表示学习 

**Authors**: Weronika Hryniewska-Guzik, Przemyslaw Biecek  

**Link**: [PDF](https://arxiv.org/pdf/2507.06264)  

**Abstract**: The success of machine learning algorithms is inherently related to the extraction of meaningful features, as they play a pivotal role in the performance of these algorithms. Central to this challenge is the quality of data representation. However, the ability to generalize and extract these features effectively from unseen datasets is also crucial. In light of this, we introduce a novel concept: the polyrepresentation. Polyrepresentation integrates multiple representations of the same modality extracted from distinct sources, for example, vector embeddings from the Siamese Network, self-supervised models, and interpretable radiomic features. This approach yields better performance metrics compared to relying on a single representation. Additionally, in the context of X-ray images, we demonstrate the transferability of the created polyrepresentation to a smaller dataset, underscoring its potential as a pragmatic and resource-efficient approach in various image-related solutions. It is worth noting that the concept of polyprepresentation on the example of medical data can also be applied to other domains, showcasing its versatility and broad potential impact. 

**Abstract (ZH)**: 机器学习算法的成功内在地依赖于有意义特征的提取，因为这些特征对于算法性能至关重要。这一挑战的核心在于数据表示的质量。然而，有效地从未见数据集中泛化和提取这些特征的能力也同样重要。基于此，我们提出了一种新颖的概念：多表示。多表示将同一模态从不同来源提取的多种表示整合在一起，例如来自Siamese网络的向量嵌入、自监督模型和可解释的影像omics特征。这种方法在性能指标上优于依赖单一表示。此外，在X射线图像的背景下，我们展示了所创建的多表示在较小数据集上的可迁移性，证明了其作为多种图像相关解决方案的实用且资源高效的途径的潜力。值得注意的是，多表示的概念不仅适用于医疗数据，还可以应用于其他领域，展示了其多样性和广泛的应用潜力。 

---
# The Emotional Alignment Design Policy 

**Title (ZH)**: 情感一致性设计政策 

**Authors**: Eric Schwitzgebel, Jeff Sebo  

**Link**: [PDF](https://arxiv.org/pdf/2507.06263)  

**Abstract**: According to what we call the Emotional Alignment Design Policy, artificial entities should be designed to elicit emotional reactions from users that appropriately reflect the entities' capacities and moral status, or lack thereof. This principle can be violated in two ways: by designing an artificial system that elicits stronger or weaker emotional reactions than its capacities and moral status warrant (overshooting or undershooting), or by designing a system that elicits the wrong type of emotional reaction (hitting the wrong target). Although presumably attractive, practical implementation faces several challenges including: How can we respect user autonomy while promoting appropriate responses? How should we navigate expert and public disagreement and uncertainty about facts and values? What if emotional alignment seems to require creating or destroying entities with moral status? To what extent should designs conform to versus attempt to alter user assumptions and attitudes? 

**Abstract (ZH)**: 根据我们称为情感对齐设计政策的原则，人工实体应当被设计成引发适当反映其能力及其道德地位或缺乏的情感反应。这一原则可以通过以下两种方式被违背：通过设计一个引发比其能力和道德地位应引起的情感反应更强或更弱的人工系统（过度反应或反应不足），或者通过设计一个引发错误类型的情感反应的人工系统（打击错误的目标）。尽管可能具有吸引力，但其实用实施面临着若干挑战，包括：如何在促进适当反应的同时尊重用户自主权？如何在专家和公众关于事实与价值观的分歧和不确定性面前进行权衡？如果情感对齐似乎需要创造或销毁具有道德地位的实体，该怎么办？设计应多大程度上遵循用户假设和态度，以及尝试改变它们？ 

---
# Q-Detection: A Quantum-Classical Hybrid Poisoning Attack Detection Method 

**Title (ZH)**: Q-Detection：一种量子-经典混合恶意攻击检测方法 

**Authors**: Haoqi He, Xiaokai Lin, Jiancai Chen, Yan Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2507.06262)  

**Abstract**: Data poisoning attacks pose significant threats to machine learning models by introducing malicious data into the training process, thereby degrading model performance or manipulating predictions. Detecting and sifting out poisoned data is an important method to prevent data poisoning attacks. Limited by classical computation frameworks, upcoming larger-scale and more complex datasets may pose difficulties for detection. We introduce the unique speedup of quantum computing for the first time in the task of detecting data poisoning. We present Q-Detection, a quantum-classical hybrid defense method for detecting poisoning attacks. Q-Detection also introduces the Q-WAN, which is optimized using quantum computing devices. Experimental results using multiple quantum simulation libraries show that Q-Detection effectively defends against label manipulation and backdoor attacks. The metrics demonstrate that Q-Detection consistently outperforms the baseline methods and is comparable to the state-of-the-art. Theoretical analysis shows that Q-Detection is expected to achieve more than a 20% speedup using quantum computing power. 

**Abstract (ZH)**: 量子计算在检测数据污染攻击中的独特加速作用：Q-Detection方法的研究 

---
# Phantom Subgroup Poisoning: Stealth Attacks on Federated Recommender Systems 

**Title (ZH)**: phantom 子组污染：联邦推荐系统中的隐身攻击 

**Authors**: Bo Yan, Yurong Hao, Dingqi Liu, Huabin Sun, Pengpeng Qiao, Wei Yang Bryan Lim, Yang Cao, Chuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2507.06258)  

**Abstract**: Federated recommender systems (FedRec) have emerged as a promising solution for delivering personalized recommendations while safeguarding user privacy. However, recent studies have demonstrated their vulnerability to poisoning attacks. Existing attacks typically target the entire user group, which compromises stealth and increases the risk of detection. In contrast, real-world adversaries may prefer to prompt target items to specific user subgroups, such as recommending health supplements to elderly users. Motivated by this gap, we introduce Spattack, the first targeted poisoning attack designed to manipulate recommendations for specific user subgroups in the federated setting. Specifically, Spattack adopts a two-stage approximation-and-promotion strategy, which first simulates user embeddings of target/non-target subgroups and then prompts target items to the target subgroups. To enhance the approximation stage, we push the inter-group embeddings away based on contrastive learning and augment the target group's relevant item set based on clustering. To enhance the promotion stage, we further propose to adaptively tune the optimization weights between target and non-target subgroups. Besides, an embedding alignment strategy is proposed to align the embeddings between the target items and the relevant items. We conduct comprehensive experiments on three real-world datasets, comparing Spattack against seven state-of-the-art poisoning attacks and seven representative defense mechanisms. Experimental results demonstrate that Spattack consistently achieves strong manipulation performance on the specific user subgroup, while incurring minimal impact on non-target users, even when only 0.1\% of users are malicious. Moreover, Spattack maintains competitive overall recommendation performance and exhibits strong resilience against existing mainstream defenses. 

**Abstract (ZH)**: 联邦推荐系统中的Spattack目标中毒攻击：针对特定用户子群的推荐操纵 

---
# We Urgently Need Privilege Management in MCP: A Measurement of API Usage in MCP Ecosystems 

**Title (ZH)**: 我们迫切需要在MCP中实现权限管理：MCP生态系统中API使用情况的测量 

**Authors**: Zhihao Li, Kun Li, Boyang Ma, Minghui Xu, Yue Zhang, Xiuzhen Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.06250)  

**Abstract**: The Model Context Protocol (MCP) has emerged as a widely adopted mechanism for connecting large language models to external tools and resources. While MCP promises seamless extensibility and rich integrations, it also introduces a substantially expanded attack surface: any plugin can inherit broad system privileges with minimal isolation or oversight. In this work, we conduct the first large-scale empirical analysis of MCP security risks. We develop an automated static analysis framework and systematically examine 2,562 real-world MCP applications spanning 23 functional categories. Our measurements reveal that network and system resource APIs dominate usage patterns, affecting 1,438 and 1,237 servers respectively, while file and memory resources are less frequent but still significant. We find that Developer Tools and API Development plugins are the most API-intensive, and that less popular plugins often contain disproportionately high-risk operations. Through concrete case studies, we demonstrate how insufficient privilege separation enables privilege escalation, misinformation propagation, and data tampering. Based on these findings, we propose a detailed taxonomy of MCP resource access, quantify security-relevant API usage, and identify open challenges for building safer MCP ecosystems, including dynamic permission models and automated trust assessment. 

**Abstract (ZH)**: MCP安全风险的大规模实证分析：从开发者工具到API开发插件的权限分离不足及其风险 

---
# Pronunciation-Lexicon Free Training for Phoneme-based Crosslingual ASR via Joint Stochastic Approximation 

**Title (ZH)**: 基于phoneme的跨语言ASR无 pronunciation-lexicon 训练：共轭随机逼近方法 

**Authors**: Saierdaer Yusuyin, Te Ma, Hao Huang, Zhijian Ou  

**Link**: [PDF](https://arxiv.org/pdf/2507.06249)  

**Abstract**: Recently, pre-trained models with phonetic supervision have demonstrated their advantages for crosslingual speech recognition in data efficiency and information sharing across languages. However, a limitation is that a pronunciation lexicon is needed for such phoneme-based crosslingual speech recognition. In this study, we aim to eliminate the need for pronunciation lexicons and propose a latent variable model based method, with phonemes being treated as discrete latent variables. The new method consists of a speech-to-phoneme (S2P) model and a phoneme-to-grapheme (P2G) model, and a grapheme-to-phoneme (G2P) model is introduced as an auxiliary inference model. To jointly train the three models, we utilize the joint stochastic approximation (JSA) algorithm, which is a stochastic extension of the EM (expectation-maximization) algorithm and has demonstrated superior performance particularly in estimating discrete latent variable models. Based on the Whistle multilingual pre-trained S2P model, crosslingual experiments are conducted in Polish (130 h) and Indonesian (20 h). With only 10 minutes of phoneme supervision, the new method, JSA-SPG, achieves 5\% error rate reductions compared to the best crosslingual fine-tuning approach using subword or full phoneme supervision. Furthermore, it is found that in language domain adaptation (i.e., utilizing cross-domain text-only data), JSA-SPG outperforms the standard practice of language model fusion via the auxiliary support of the G2P model by 9% error rate reductions. To facilitate reproducibility and encourage further exploration in this field, we open-source the JSA-SPG training code and complete pipeline. 

**Abstract (ZH)**: Recent预训练模型在音素监督下的跨语言语音识别中的数据效率和信息共享优势已经得到证实，然而仍需发音词典。本文旨在消除发音词典的需要，并提出一种基于潜在变量的方法，将音素视为离散的潜在变量。该新方法由语音到音素（S2P）模型和音素到字母（P2G）模型组成，引入了字母到音素（G2P）模型作为辅助推断模型。为了联合训练这三种模型，我们采用了联合随机逼近（JSA）算法，这是一种随机化的EM（期望最大化）算法扩展，并且在估计离散的潜在变量模型方面表现优异。基于Whistle多语言预训练S2P模型，在波兰语（130小时）和印度尼西亚语（20小时）上进行了跨语言实验。仅使用10分钟的音素监督，新方法JSA-SPG相比使用子词或完整音素监督的最佳跨语言微调方法，错误率降低了5%。此外，研究发现，在语言领域适应（即利用跨域文本数据）中，JSA-SPG比标准的语言模型融合方法通过G2P模型的辅助支持，错误率降低了9%。为了促进可再现性和鼓励对该领域的进一步探索，我们开源了JSA-SPG训练代码和完整管道。 

---
# Super Kawaii Vocalics: Amplifying the "Cute" Factor in Computer Voice 

**Title (ZH)**: 超级可爱语音：增强计算机语音中的“可爱”因素 

**Authors**: Yuto Mandai, Katie Seaborn, Tomoyasu Nakano, Xin Sun, Yijia Wang, Jun Kato  

**Link**: [PDF](https://arxiv.org/pdf/2507.06235)  

**Abstract**: "Kawaii" is the Japanese concept of cute, which carries sociocultural connotations related to social identities and emotional responses. Yet, virtually all work to date has focused on the visual side of kawaii, including in studies of computer agents and social robots. In pursuit of formalizing the new science of kawaii vocalics, we explored what elements of voice relate to kawaii and how they might be manipulated, manually and automatically. We conducted a four-phase study (grand N = 512) with two varieties of computer voices: text-to-speech (TTS) and game character voices. We found kawaii "sweet spots" through manipulation of fundamental and formant frequencies, but only for certain voices and to a certain extent. Findings also suggest a ceiling effect for the kawaii vocalics of certain voices. We offer empirical validation of the preliminary kawaii vocalics model and an elementary method for manipulating kawaii perceptions of computer voice. 

**Abstract (ZH)**: “可爱”是日本对可爱的概念，蕴含与社会身份和情感反应相关的社会文化含义。然而，迄今为止的绝大多数研究都集中在可爱的视觉方面，包括对计算机代理和社交机器人的研究。为了构建新的可爱声学学科学体系，我们探索了哪些声音元素与可爱相关，以及如何手动和自动地操纵这些元素。我们开展了四阶段的研究（总量N=512），使用了两种类型的计算机声音：文本到语音（TTS）和游戏角色声音。我们发现通过操控基频和形音频率可以找到可爱的声音“黄金点”，但仅限于某些声音，并且有一定限度。研究结果还表明，某些声音的可爱声学具有天花板效应。我们提供了初步可爱声学模型的实证验证，并提出了一种基本方法来操纵计算机声音的可爱感知。 

---
