# SMapper: A Multi-Modal Data Acquisition Platform for SLAM Benchmarking 

**Title (ZH)**: SMapper：用于SLAM基准测试的多模态数据采集平台 

**Authors**: Pedro Miguel Bastos Soares, Ali Tourani, Miguel Fernandez-Cortizas, Asier Bikandi Noya, Jose Luis Sanchez-Lopez, Holger Voos  

**Link**: [PDF](https://arxiv.org/pdf/2509.09509)  

**Abstract**: Advancing research in fields like Simultaneous Localization and Mapping (SLAM) and autonomous navigation critically depends on reliable and reproducible multimodal datasets. While several influential datasets have driven progress in these domains, they often suffer from limitations in sensing modalities, environmental diversity, and the reproducibility of the underlying hardware setups. To address these challenges, this paper introduces SMapper, a novel open-hardware, multi-sensor platform designed explicitly for, though not limited to, SLAM research. The device integrates synchronized LiDAR, multi-camera, and inertial sensing, supported by a robust calibration and synchronization pipeline that ensures precise spatio-temporal alignment across modalities. Its open and replicable design allows researchers to extend its capabilities and reproduce experiments across both handheld and robot-mounted scenarios. To demonstrate its practicality, we additionally release SMapper-light, a publicly available SLAM dataset containing representative indoor and outdoor sequences. The dataset includes tightly synchronized multimodal data and ground-truth trajectories derived from offline LiDAR-based SLAM with sub-centimeter accuracy, alongside dense 3D reconstructions. Furthermore, the paper contains benchmarking results on state-of-the-art LiDAR and visual SLAM frameworks using the SMapper-light dataset. By combining open-hardware design, reproducible data collection, and comprehensive benchmarking, SMapper establishes a robust foundation for advancing SLAM algorithm development, evaluation, and reproducibility. 

**Abstract (ZH)**: 先进同时定位与地图构建（SLAM）和自主导航领域研究的进步关键依赖于可靠且可再现的多模态数据集。为了应对这些挑战，本文介绍了一种新颖的开源硬件多传感器平台SMapper，该平台专门用于SLAM研究，虽然不局限于SLAM研究领域。设备集成了同步 Lidar、多摄像头和惯性传感器，并通过一个稳健的校准和同步管线确保各模态之间的精确时空对齐。其开源和可再现的设计使研究人员能够扩展其功能并在手持和机器人搭载的场景中再现实验。为了证明其实用性，我们还发布了SMapper-light，这是一个包含代表性室内和室外序列的公开SLAM数据集。该数据集包括精确到亚厘米级别的离线LiDAR SLAM的参考轨迹和密集三维重建。此外，论文使用SMapper-light数据集对先进的LiDAR和视觉SLAM框架进行了基准测试。通过结合开源硬件设计、可再现的数据收集和全面的基准测试，SMapper为SLAM算法的发展、评估和再现性奠定了坚实的基础。 

---
# KoopMotion: Learning Almost Divergence Free Koopman Flow Fields for Motion Planning 

**Title (ZH)**: KoopMotion: 学习几乎无散度的Koopman流场以进行运动规划 

**Authors**: Alice Kate Li, Thales C Silva, Victoria Edwards, Vijay Kumar, M. Ani Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2509.09074)  

**Abstract**: In this work, we propose a novel flow field-based motion planning method that drives a robot from any initial state to a desired reference trajectory such that it converges to the trajectory's end point. Despite demonstrated efficacy in using Koopman operator theory for modeling dynamical systems, Koopman does not inherently enforce convergence to desired trajectories nor to specified goals -- a requirement when learning from demonstrations (LfD). We present KoopMotion which represents motion flow fields as dynamical systems, parameterized by Koopman Operators to mimic desired trajectories, and leverages the divergence properties of the learnt flow fields to obtain smooth motion fields that converge to a desired reference trajectory when a robot is placed away from the desired trajectory, and tracks the trajectory until the end point. To demonstrate the effectiveness of our approach, we show evaluations of KoopMotion on the LASA human handwriting dataset and a 3D manipulator end-effector trajectory dataset, including spectral analysis. We also perform experiments on a physical robot, verifying KoopMotion on a miniature autonomous surface vehicle operating in a non-static fluid flow environment. Our approach is highly sample efficient in both space and time, requiring only 3\% of the LASA dataset to generate dense motion plans. Additionally, KoopMotion provides a significant improvement over baselines when comparing metrics that measure spatial and temporal dynamics modeling efficacy. 

**Abstract (ZH)**: 基于流场的运动规划方法：KoopMotion 

---
# Global Optimization of Stochastic Black-Box Functions with Arbitrary Noise Distributions using Wilson Score Kernel Density Estimation 

**Title (ZH)**: 使用威尔逊分数核密度估计全局优化任意噪声分布的随机黑盒函数 

**Authors**: Thorbjørn Mosekjær Iversen, Lars Carøe Sørensen, Simon Faarvang Mathiesen, Henrik Gordon Petersen  

**Link**: [PDF](https://arxiv.org/pdf/2509.09238)  

**Abstract**: Many optimization problems in robotics involve the optimization of time-expensive black-box functions, such as those involving complex simulations or evaluation of real-world experiments. Furthermore, these functions are often stochastic as repeated experiments are subject to unmeasurable disturbances. Bayesian optimization can be used to optimize such methods in an efficient manner by deploying a probabilistic function estimator to estimate with a given confidence so that regions of the search space can be pruned away. Consequently, the success of the Bayesian optimization depends on the function estimator's ability to provide informative confidence bounds. Existing function estimators require many function evaluations to infer the underlying confidence or depend on modeling of the disturbances. In this paper, it is shown that the confidence bounds provided by the Wilson Score Kernel Density Estimator (WS-KDE) are applicable as excellent bounds to any stochastic function with an output confined to the closed interval [0;1] regardless of the distribution of the output. This finding opens up the use of WS-KDE for stable global optimization on a wider range of cost functions. The properties of WS-KDE in the context of Bayesian optimization are demonstrated in simulation and applied to the problem of automated trap design for vibrational part feeders. 

**Abstract (ZH)**: 机器人领域中涉及的时间昂贵的黑盒函数优化问题及其Bayesian优化方法的研究：Wilson Score内核密度估计在[0;1]区间输出的随机函数中的应用 

---
# ProgD: Progressive Multi-scale Decoding with Dynamic Graphs for Joint Multi-agent Motion Forecasting 

**Title (ZH)**: ProgD：基于动态图的分阶段多尺度解码联合多智能体运动预测 

**Authors**: Xing Gao, Zherui Huang, Weiyao Lin, Xiao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.09210)  

**Abstract**: Accurate motion prediction of surrounding agents is crucial for the safe planning of autonomous vehicles. Recent advancements have extended prediction techniques from individual agents to joint predictions of multiple interacting agents, with various strategies to address complex interactions within future motions of agents. However, these methods overlook the evolving nature of these interactions. To address this limitation, we propose a novel progressive multi-scale decoding strategy, termed ProgD, with the help of dynamic heterogeneous graph-based scenario modeling. In particular, to explicitly and comprehensively capture the evolving social interactions in future scenarios, given their inherent uncertainty, we design a progressive modeling of scenarios with dynamic heterogeneous graphs. With the unfolding of such dynamic heterogeneous graphs, a factorized architecture is designed to process the spatio-temporal dependencies within future scenarios and progressively eliminate uncertainty in future motions of multiple agents. Furthermore, a multi-scale decoding procedure is incorporated to improve on the future scenario modeling and consistent prediction of agents' future motion. The proposed ProgD achieves state-of-the-art performance on the INTERACTION multi-agent prediction benchmark, ranking $1^{st}$, and the Argoverse 2 multi-world forecasting benchmark. 

**Abstract (ZH)**: 准确预测周围代理的运动对于自主车辆的安全规划至关重要。为了解决复杂交互带来的挑战，近年来的研究将预测技术从单个代理扩展到多个互动代理的联合预测，并提出了各种策略来处理未来代理运动中的交互。然而，这些方法忽视了这些交互的动态性。为了解决这一局限，我们提出了一种新颖的渐进多尺度解码策略，称为ProgD，并借助动态异构图基场景建模。特别是，为了明确且全面地捕捉未来场景中的动态社会交互（鉴于其本质上的不确定性），我们设计了一种基于动态异构图的渐进场景建模方法。随着动态异构图的展开，我们设计了一种因式化解构架构来处理未来场景中的时空依赖性，并逐步消除多个代理未来运动中的不确定性。此外，我们引入了一种多尺度解码过程来提高未来场景建模的一致性和代理未来运动的预测准确性。提出的ProgD在INTERACTION多代理预测基准和Argoverse 2多世界预测基准上取得了最先进的性能，分别排名第一。 

---
# Compositional Concept Generalization with Variational Quantum Circuits 

**Title (ZH)**: 基于变分量子电路的合成概念泛化 

**Authors**: Hala Hawashin, Mina Abbaszadeh, Nicholas Joseph, Beth Pearson, Martha Lewis, Mehrnoosh sadrzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2509.09541)  

**Abstract**: Compositional generalization is a key facet of human cognition, but lacking in current AI tools such as vision-language models. Previous work examined whether a compositional tensor-based sentence semantics can overcome the challenge, but led to negative results. We conjecture that the increased training efficiency of quantum models will improve performance in these tasks. We interpret the representations of compositional tensor-based models in Hilbert spaces and train Variational Quantum Circuits to learn these representations on an image captioning task requiring compositional generalization. We used two image encoding techniques: a multi-hot encoding (MHE) on binary image vectors and an angle/amplitude encoding on image vectors taken from the vision-language model CLIP. We achieve good proof-of-concept results using noisy MHE encodings. Performance on CLIP image vectors was more mixed, but still outperformed classical compositional models. 

**Abstract (ZH)**: 组分泛化是人类认知的一个关键方面，但目前的AI工具如视觉-语言模型中缺乏这一功能。我们推测，量子模型训练效率的提升将在这些任务中改善性能。我们将组分张量基句子语义的表示解释为希尔伯特空间中的表示，并训练变量子电路在需要组分泛化的图像描述任务中学习这些表示。我们使用了两种图像编码技术：二元图像向量的多热编码（MHE）和图像向量（来自视觉-语言模型CLIP）的角度/振幅编码。使用嘈杂的MHE编码取得了良好的概念验证结果。在CLIP图像向量上的性能更为参差不齐，但仍优于经典的组分模型。 

---
# SEDM: Scalable Self-Evolving Distributed Memory for Agents 

**Title (ZH)**: SEDM：可扩展自进化的分布式内存架构 

**Authors**: Haoran Xu, Jiacong Hu, Ke Zhang, Lei Yu, Yuxin Tang, Xinyuan Song, Yiqun Duan, Lynn Ai, Bill Shi  

**Link**: [PDF](https://arxiv.org/pdf/2509.09498)  

**Abstract**: Long-term multi-agent systems inevitably generate vast amounts of trajectories and historical interactions, which makes efficient memory management essential for both performance and scalability. Existing methods typically depend on vector retrieval and hierarchical storage, yet they are prone to noise accumulation, uncontrolled memory expansion, and limited generalization across domains. To address these challenges, we present SEDM, Self-Evolving Distributed Memory, a verifiable and adaptive framework that transforms memory from a passive repository into an active, self-optimizing component. SEDM integrates verifiable write admission based on reproducible replay, a self-scheduling memory controller that dynamically ranks and consolidates entries according to empirical utility, and cross-domain knowledge diffusion that abstracts reusable insights to support transfer across heterogeneous tasks. Evaluations on benchmark datasets demonstrate that SEDM improves reasoning accuracy while reducing token overhead compared with strong memory baselines, and further enables knowledge distilled from fact verification to enhance multi-hop reasoning. The results highlight SEDM as a scalable and sustainable memory mechanism for open-ended multi-agent collaboration. The code will be released in the later stage of this project. 

**Abstract (ZH)**: 自适应演化分布式记忆：可验证的长期多智能体系统高效内存管理框架 

---
# Measuring Implicit Spatial Coordination in Teams: Effects on Collective Intelligence and Performance 

**Title (ZH)**: 测量团队中的隐性空间协调：对其集体智能和表现的影响 

**Authors**: Thuy Ngoc Nguyen, Anita Williams Woolley, Cleotilde Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2509.09314)  

**Abstract**: Coordinated teamwork is essential in fast-paced decision-making environments that require dynamic adaptation, often without an opportunity for explicit communication. Although implicit coordination has been extensively considered in the existing literature, the majority of work has focused on co-located, synchronous teamwork (such as sports teams) or, in distributed teams, primarily on coordination of knowledge work. However, many teams (firefighters, military, law enforcement, emergency response) must coordinate their movements in physical space without the benefit of visual cues or extensive explicit communication. This paper investigates how three dimensions of spatial coordination, namely exploration diversity, movement specialization, and adaptive spatial proximity, influence team performance in a collaborative online search and rescue task where explicit communication is restricted and team members rely on movement patterns to infer others' intentions and coordinate actions. Our metrics capture the relational aspects of teamwork by measuring spatial proximity, distribution patterns, and alignment of movements within shared environments. We analyze data from 34 four-person teams (136 participants) assigned to specialized roles in a search and rescue task. Results show that spatial specialization positively predicts performance, while adaptive spatial proximity exhibits a marginal inverted U-shaped relationship, suggesting moderate levels of adaptation are optimal. Furthermore, the temporal dynamics of these metrics differentiate high- from low-performing teams over time. These findings provide insights into implicit spatial coordination in role-based teamwork and highlight the importance of balanced adaptive strategies, with implications for training and AI-assisted team support systems. 

**Abstract (ZH)**: 快速决策环境中，在动态适应需求下协调团队合作至关重要，往往缺乏明确的沟通机会。尽管现有文献中已经广泛考虑了隐性协调，但大多数工作主要聚焦于共处且同步工作的团队（如运动队）或分布式团队中的知识工作协调。然而，许多团队（如消防员、军事人员、执法机构、紧急救援人员）必须在缺乏视觉提示和大量明确沟通的情况下，在物理空间中协调行动。本文探讨了探索多样性、运动专业化和适应性空间接近这三种空间协调维度如何影响受限明确沟通条件下的协作在线搜索与营救任务中的团队表现。我们通过测量共享环境中空间接近性、分布模式和运动方向的一致性来捕捉团队合作中的关系层面。我们分析了34个四人团队（136名参与者）在搜索与救援任务中分配专门角色的数据。结果显示，空间专业化正向预测团队表现，而适应性空间接近呈现轻微的倒U型关系，表明中等程度的适应性最优化。此外，这些指标的时间动态在时间上区分了高绩效和低绩效团队。这些发现揭示了基于角色团队中隐性空间协调的洞见，并强调了平衡适应策略的重要性，这对训练和AI辅助团队支持系统有启示意义。 

---
# Explaining Tournament Solutions with Minimal Supports 

**Title (ZH)**: 用最小支持集解释锦标赛解 

**Authors**: Clément Contet, Umberto Grandi, Jérôme Mengin  

**Link**: [PDF](https://arxiv.org/pdf/2509.09312)  

**Abstract**: Tournaments are widely used models to represent pairwise dominance between candidates, alternatives, or teams. We study the problem of providing certified explanations for why a candidate appears among the winners under various tournament rules. To this end, we identify minimal supports, minimal sub-tournaments in which the candidate is guaranteed to win regardless of how the rest of the tournament is completed (that is, the candidate is a necessary winner of the sub-tournament). This notion corresponds to an abductive explanation for the question,"Why does the winner win the tournament", a central concept in formal explainable AI. We focus on common tournament solutions: the top cycle, the uncovered set, the Copeland rule, the Borda rule, the maximin rule, and the weighted uncovered set. For each rule we determine the size of the smallest minimal supports, and we present polynomial-time algorithms to compute them for all but the weighted uncovered set, for which the problem is NP-complete. Finally, we show how minimal supports can serve to produce compact, certified, and intuitive explanations. 

**Abstract (ZH)**: 竞赛广泛用于表示候选人、替代品或团队之间的两两支配关系。我们研究在各种竞赛规则下提供认证解释的问题，解释为什么某个候选人会出现在获胜者之中。为此，我们识别最小支撑，即在其中候选人可以确保获胜的最小子竞赛（即，候选人是子竞赛的必要获胜者），而不考虑其余竞赛如何完成。这一概念对应于关于“为什么获胜者赢得竞赛”的归因解释，是形式可解释人工智能中的一个核心概念。我们关注常用的竞赛解决方案：顶周期、未覆盖集、Copeland法则、Borda法则、最大化最小值法则和加权未覆盖集。对于每种法则，我们确定最小支撑的大小，并提供了一类法则计算最小支撑的多项式时间算法，对于加权未覆盖集，该问题为NP完全问题。最后，我们展示了最小支撑如何用于生成紧凑、认证且直观的解释。 

---
# Tree-OPO: Off-policy Monte Carlo Tree-Guided Advantage Optimization for Multistep Reasoning 

**Title (ZH)**: 树-OPO：基于蒙特卡洛树引导的优势多步优化算法 

**Authors**: Bingning Huang, Tu Nguyen, Matthieu Zimmer  

**Link**: [PDF](https://arxiv.org/pdf/2509.09284)  

**Abstract**: Recent advances in reasoning with large language models (LLMs) have shown the effectiveness of Monte Carlo Tree Search (MCTS) for generating high-quality intermediate trajectories, particularly in math and symbolic domains. Inspired by this, we explore how MCTS-derived trajectories, traditionally used for training value or reward models, can be repurposed to improve policy optimization in preference-based reinforcement learning (RL). Specifically, we focus on Group Relative Policy Optimization (GRPO), a recent algorithm that enables preference-consistent policy learning without value networks. We propose a staged GRPO training paradigm where completions are derived from partially revealed MCTS rollouts, introducing a novel tree-structured setting for advantage estimation. This leads to a rich class of prefix-conditioned reward signals, which we analyze theoretically and empirically. Our initial results indicate that while structured advantage estimation can stabilize updates and better reflect compositional reasoning quality, challenges such as advantage saturation and reward signal collapse remain. We propose heuristic and statistical solutions to mitigate these issues and discuss open challenges for learning under staged or tree-like reward structures. 

**Abstract (ZH)**: Recent advances in reasoning with large language models (LLMs) have shown the effectiveness of Monte Carlo Tree Search (MCTS) for generating high-quality intermediate trajectories, particularly in math and symbolic domains. Inspired by this, we explore how MCTS-derived trajectories, traditionally used for training value or reward models, can be repurposed to improve policy optimization in preference-based reinforcement learning (RL). Specifically, we focus on Group Relative Policy Optimization (GRPO), a recent algorithm that enables preference-consistent policy learning without value networks. We propose a staged GRPO training paradigm where completions are derived from partially revealed MCTS rollouts, introducing a novel tree-structured setting for advantage estimation. This leads to a rich class of prefix-conditioned reward signals, which we analyze theoretically and empirically. Our initial results indicate that while structured advantage estimation can stabilize updates and better reflect compositional reasoning quality, challenges such as advantage saturation and reward signal collapse remain. We propose heuristic and statistical solutions to mitigate these issues and discuss open challenges for learning under staged or tree-like reward structures. 

---
# Anti-Money Laundering Machine Learning Pipelines; A Technical Analysis on Identifying High-risk Bank Clients with Supervised Learning 

**Title (ZH)**: 反洗钱机器学习管道：一种基于监督学习识别高风险银行客户的技術分析 

**Authors**: Khashayar Namdar, Pin-Chien Wang, Tushar Raju, Steven Zheng, Fiona Li, Safwat Tahmin Khan  

**Link**: [PDF](https://arxiv.org/pdf/2509.09127)  

**Abstract**: Anti-money laundering (AML) actions and measurements are among the priorities of financial institutions, for which machine learning (ML) has shown to have a high potential. In this paper, we propose a comprehensive and systematic approach for developing ML pipelines to identify high-risk bank clients in a dataset curated for Task 1 of the University of Toronto 2023-2024 Institute for Management and Innovation (IMI) Big Data and Artificial Intelligence Competition. The dataset included 195,789 customer IDs, and we employed a 16-step design and statistical analysis to ensure the final pipeline was robust. We also framed the data in a SQLite database, developed SQL-based feature engineering algorithms, connected our pre-trained model to the database, and made it inference-ready, and provided explainable artificial intelligence (XAI) modules to derive feature importance. Our pipeline achieved a mean area under the receiver operating characteristic curve (AUROC) of 0.961 with a standard deviation (SD) of 0.005. The proposed pipeline achieved second place in the competition. 

**Abstract (ZH)**: 面向金融风险管理的机器学习管道构建：以多伦多大学2023-2024年管理与创新学院大数据与人工智能竞赛Task 1数据集为例 

---
# Understanding Economic Tradeoffs Between Human and AI Agents in Bargaining Games 

**Title (ZH)**: 理解人类与AI代理在谈判游戏中的人工智能与人类经济权衡 

**Authors**: Crystal Qian, Kehang Zhu, John Horton, Benjamin S. Manning, Vivian Tsai, James Wexler, Nithum Thain  

**Link**: [PDF](https://arxiv.org/pdf/2509.09071)  

**Abstract**: Coordination tasks traditionally performed by humans are increasingly being delegated to autonomous agents. As this pattern progresses, it becomes critical to evaluate not only these agents' performance but also the processes through which they negotiate in dynamic, multi-agent environments. Furthermore, different agents exhibit distinct advantages: traditional statistical agents, such as Bayesian models, may excel under well-specified conditions, whereas large language models (LLMs) can generalize across contexts. In this work, we compare humans (N = 216), LLMs (GPT-4o, Gemini 1.5 Pro), and Bayesian agents in a dynamic negotiation setting that enables direct, identical-condition comparisons across populations, capturing both outcomes and behavioral dynamics. Bayesian agents extract the highest surplus through aggressive optimization, at the cost of frequent trade rejections. Humans and LLMs can achieve similar overall surplus, but through distinct behaviors: LLMs favor conservative, concessionary trades with few rejections, while humans employ more strategic, risk-taking, and fairness-oriented behaviors. Thus, we find that performance parity -- a common benchmark in agent evaluation -- can conceal fundamental differences in process and alignment, which are critical for practical deployment in real-world coordination tasks. 

**Abstract (ZH)**: 传统上由人类执行的合作任务越来越多地被自主代理接管。随着这一模式的发展，不仅需要评估这些代理的表现，还需要评估它们在动态多代理环境中谈判的过程。此外，不同的代理展现出不同的优势：传统统计代理，如贝叶斯模型，在特定条件下可能表现出色，而大型语言模型（LLMs）则能够在不同场景中泛化。在本研究中，我们将人类（N=216）、LLMs（GPT-4o、Gemini 1.5 Pro）和贝叶斯代理在一种动态谈判环境中进行对比，该环境能够跨群体提供直接且条件一致的比较，捕捉到结果和行为动力学。贝叶斯代理通过激进优化获得最高的剩余价值，但代价是频繁的交易拒绝。人类和LLMs可以实现类似的总体剩余价值，但通过不同的行为：LLMs偏好保守、让步的交易且拒绝较少，而人类则采取更具战略性和冒险性的公平导向行为。因此，我们发现代理性能一致——代理评估中常用的标准——可能掩盖了过程和对齐的根本差异，这对于实际部署在实际协调任务中的代理至关重要。 

---
# Uncertainty Awareness and Trust in Explainable AI- On Trust Calibration using Local and Global Explanations 

**Title (ZH)**: 不确定性意识与可解释人工智能中的信任校准——基于局部和全局解释的信任 calibration 研究 

**Authors**: Carina Newen, Daniel Bodemer, Sonja Glantz, Emmanuel Müller, Magdalena Wischnewski, Lenka Schnaubert  

**Link**: [PDF](https://arxiv.org/pdf/2509.08989)  

**Abstract**: Explainable AI has become a common term in the literature, scrutinized by computer scientists and statisticians and highlighted by psychological or philosophical researchers. One major effort many researchers tackle is constructing general guidelines for XAI schemes, which we derived from our study. While some areas of XAI are well studied, we focus on uncertainty explanations and consider global explanations, which are often left out. We chose an algorithm that covers various concepts simultaneously, such as uncertainty, robustness, and global XAI, and tested its ability to calibrate trust. We then checked whether an algorithm that aims to provide more of an intuitive visual understanding, despite being complicated to understand, can provide higher user satisfaction and human interpretability. 

**Abstract (ZH)**: 可解释AI已成为文献中的一个常见术语，受到计算机科学家和统计学家的审查，并引起了心理学家或哲学家的高度重视。许多研究者的一项主要努力是构建通用的可解释AI方案指南，我们从我们的研究中得出了这些指南。尽管某些可解释AI领域已研究得较为充分，但我们专注于不确定性解释，并考虑了通常被忽略的全局解释。我们选择了能够同时涵盖各种概念（如不确定性、鲁棒性和全局可解释性）的算法，并测试了其校准信任的能力。然后我们检查了一种旨在提供更直观的视觉理解的算法，尽管其理解起来比较复杂，是否能够提供更高的用户满意度和人类可解释性。 

---
# An Interval Type-2 Version of Bayes Theorem Derived from Interval Probability Range Estimates Provided by Subject Matter Experts 

**Title (ZH)**: 基于领域专家提供的区间概率范围估计推导出的区间类型2贝叶斯定理 

**Authors**: John T. Rickard, William A. Dembski, James Rickards  

**Link**: [PDF](https://arxiv.org/pdf/2509.08834)  

**Abstract**: Bayesian inference is widely used in many different fields to test hypotheses against observations. In most such applications, an assumption is made of precise input values to produce a precise output value. However, this is unrealistic for real-world applications. Often the best available information from subject matter experts (SMEs) in a given field is interval range estimates of the input probabilities involved in Bayes Theorem. This paper provides two key contributions to extend Bayes Theorem to an interval type-2 (IT2) version. First, we develop an IT2 version of Bayes Theorem that uses a novel and conservative method to avoid potential inconsistencies in the input IT2 MFs that otherwise might produce invalid output results. We then describe a novel and flexible algorithm for encoding SME-provided intervals into IT2 fuzzy membership functions (MFs), which we can use to specify the input probabilities in Bayes Theorem. Our algorithm generalizes and extends previous work on this problem that primarily addressed the encoding of intervals into word MFs for Computing with Words applications. 

**Abstract (ZH)**: 贝叶斯推断广泛应用于多个领域以假设测试与观测数据的对比。然而，在大多数此类应用中，人们假设输入值精确以产生精确的输出值。但在实际应用中，这是不现实的。通常，在贝叶斯定理中涉及的输入概率的最佳区间估计是由给定领域内的专家提供的区间范围估计。本文为将贝叶斯定理扩展到区间类型-2 (IT2) 版本做出了两个关键贡献。首先，我们开发了一个采用新颖且保守方法的IT2版本的贝叶斯定理，以避免输入IT2隶属函数（MFs）可能产生的潜在不一致性，从而产生无效的输出结果。其次，我们描述了一个新颖且灵活的算法，用于将专家提供的区间编码为IT2模糊隶属函数（MFs），以此来指定贝叶斯定理中的输入概率。该算法扩展了以往主要针对将区间编码为词汇隶属函数的研究，以应用于计算与词语的应用。 

---
# Feasibility-Guided Fair Adaptive Offline Reinforcement Learning for Medicaid Care Management 

**Title (ZH)**: 基于可行性的公正自适应离线强化学习在 Medicaid 照顾管理中的可行性研究 

**Authors**: Sanjay Basu, Sadiq Y. Patel, Parth Sheth, Bhairavi Muralidharan, Namrata Elamaran, Aakriti Kinra, Rajaie Batniji  

**Link**: [PDF](https://arxiv.org/pdf/2509.09655)  

**Abstract**: We introduce Feasibility-Guided Fair Adaptive Reinforcement Learning (FG-FARL), an offline RL procedure that calibrates per-group safety thresholds to reduce harm while equalizing a chosen fairness target (coverage or harm) across protected subgroups. Using de-identified longitudinal trajectories from a Medicaid population health management program, we evaluate FG-FARL against behavior cloning (BC) and HACO (Hybrid Adaptive Conformal Offline RL; a global conformal safety baseline). We report off-policy value estimates with bootstrap 95% confidence intervals and subgroup disparity analyses with p-values. FG-FARL achieves comparable value to baselines while improving fairness metrics, demonstrating a practical path to safer and more equitable decision support. 

**Abstract (ZH)**: 基于可行性的公平自适应强化学习（FG-FARL）：一种减少危害并平等化公平目标的离线强化学习方法 

---
# Retrieval-Augmented Generation for Reliable Interpretation of Radio Regulations 

**Title (ZH)**: 基于检索增强生成的无线电规则可靠解释方法 

**Authors**: Zakaria El Kassimi, Fares Fourati, Mohamed-Slim Alouini  

**Link**: [PDF](https://arxiv.org/pdf/2509.09651)  

**Abstract**: We study question answering in the domain of radio regulations, a legally sensitive and high-stakes area. We propose a telecom-specific Retrieval-Augmented Generation (RAG) pipeline and introduce, to our knowledge, the first multiple-choice evaluation set for this domain, constructed from authoritative sources using automated filtering and human validation. To assess retrieval quality, we define a domain-specific retrieval metric, under which our retriever achieves approximately 97% accuracy. Beyond retrieval, our approach consistently improves generation accuracy across all tested models. In particular, while naively inserting documents without structured retrieval yields only marginal gains for GPT-4o (less than 1%), applying our pipeline results in nearly a 12% relative improvement. These findings demonstrate that carefully targeted grounding provides a simple yet strong baseline and an effective domain-specific solution for regulatory question answering. All code and evaluation scripts, along with our derived question-answer dataset, are available at this https URL. 

**Abstract (ZH)**: 我们在无线电管理法规领域的问答研究，这是一个法律敏感且高风险的领域。我们提出了一种针对电信的检索增强生成（RAG）管道，并且，据我们所知，首次构建了一个来自权威来源的多选评价数据集，使用自动过滤和人工验证。为了评估检索质量，我们定义了一个领域特定的检索指标，在此指标下，我们的检索器准确率达到约97%。除了检索之外，我们的方法在所有测试的模型中都一致地提高了生成准确性。特别是，与未经结构化检索直接插入文档的情况相比，我们的管道使得GPT-4o的相对改进达到近12%。这些发现表明，精确的目标定位提供了简单而有效的基线和领域特定解决方案，以解决监管问答问题。所有代码和评价脚本，以及我们衍生的问答数据集均可在以下链接获取。 

---
# Explaining Concept Drift through the Evolution of Group Counterfactuals 

**Title (ZH)**: 通过群体反事实的演变解释概念漂移 

**Authors**: Ignacy Stępka, Jerzy Stefanowski  

**Link**: [PDF](https://arxiv.org/pdf/2509.09616)  

**Abstract**: Machine learning models in dynamic environments often suffer from concept drift, where changes in the data distribution degrade performance. While detecting this drift is a well-studied topic, explaining how and why the model's decision-making logic changes still remains a significant challenge. In this paper, we introduce a novel methodology to explain concept drift by analyzing the temporal evolution of group-based counterfactual explanations (GCEs). Our approach tracks shifts in the GCEs' cluster centroids and their associated counterfactual action vectors before and after a drift. These evolving GCEs act as an interpretable proxy, revealing structural changes in the model's decision boundary and its underlying rationale. We operationalize this analysis within a three-layer framework that synergistically combines insights from the data layer (distributional shifts), the model layer (prediction disagreement), and our proposed explanation layer. We show that such holistic view allows for a more comprehensive diagnosis of drift, making it possible to distinguish between different root causes, such as a spatial data shift versus a re-labeling of concepts. 

**Abstract (ZH)**: 机器学习模型在动态环境中的概念漂移解释方法探究：通过基于组的反事实解释的时间演化分析 

---
# Mechanistic Learning with Guided Diffusion Models to Predict Spatio-Temporal Brain Tumor Growth 

**Title (ZH)**: 基于引导扩散模型的机制性学习以预测空间-时间脑肿瘤生长 

**Authors**: Daria Laslo, Efthymios Georgiou, Marius George Linguraru, Andreas Rauschecker, Sabine Muller, Catherine R. Jutzeler, Sarah Bruningk  

**Link**: [PDF](https://arxiv.org/pdf/2509.09610)  

**Abstract**: Predicting the spatio-temporal progression of brain tumors is essential for guiding clinical decisions in neuro-oncology. We propose a hybrid mechanistic learning framework that combines a mathematical tumor growth model with a guided denoising diffusion implicit model (DDIM) to synthesize anatomically feasible future MRIs from preceding scans. The mechanistic model, formulated as a system of ordinary differential equations, captures temporal tumor dynamics including radiotherapy effects and estimates future tumor burden. These estimates condition a gradient-guided DDIM, enabling image synthesis that aligns with both predicted growth and patient anatomy. We train our model on the BraTS adult and pediatric glioma datasets and evaluate on 60 axial slices of in-house longitudinal pediatric diffuse midline glioma (DMG) cases. Our framework generates realistic follow-up scans based on spatial similarity metrics. It also introduces tumor growth probability maps, which capture both clinically relevant extent and directionality of tumor growth as shown by 95th percentile Hausdorff Distance. The method enables biologically informed image generation in data-limited scenarios, offering generative-space-time predictions that account for mechanistic priors. 

**Abstract (ZH)**: 基于空间-时间进展的脑肿瘤预测对于神经 Oncology 的临床决策至关重要。我们提出了一种结合数学肿瘤生长模型和引导去噪扩散隐式模型（DDIM）的混合机理学习框架，以从先前的扫描中合成符合解剖学的未来 MRI 图像。机理模型以常微分方程系统的形式表述，捕捉包括放疗效果在内的时间肿瘤动态，并估计未来肿瘤负荷。这些估计值条件引导梯度下的 DDIM，使得生成的图像与预测的生长和患者解剖结构相一致。我们在 BraTS 成人和儿童胶质瘤数据集上训练模型，并在内部儿童弥漫中线胶质瘤 (DMG) 横截面数据集的 60 个层面进行评估。该框架基于空间相似度指标生成现实的随访扫描。它还引入了肿瘤生长概率图，这些图捕捉由 95 个百分点 Hausdorff 距离所示的临床相关范围和肿瘤生长的方向性。该方法在数据有限的情况下实现生物学指导的图像生成，提供了考虑机理先验的空间-时间生成预测。 

---
# Graph Alignment via Dual-Pass Spectral Encoding and Latent Space Communication 

**Title (ZH)**: 图对齐 via 双通道谱编码和潜在空间通信 

**Authors**: Maysam Behmanesh, Erkan Turan, Maks Ovsjanikov  

**Link**: [PDF](https://arxiv.org/pdf/2509.09597)  

**Abstract**: Graph alignment-the problem of identifying corresponding nodes across multiple graphs-is fundamental to numerous applications. Most existing unsupervised methods embed node features into latent representations to enable cross-graph comparison without ground-truth correspondences. However, these methods suffer from two critical limitations: the degradation of node distinctiveness due to oversmoothing in GNN-based embeddings, and the misalignment of latent spaces across graphs caused by structural noise, feature heterogeneity, and training instability, ultimately leading to unreliable node correspondences. We propose a novel graph alignment framework that simultaneously enhances node distinctiveness and enforces geometric consistency across latent spaces. Our approach introduces a dual-pass encoder that combines low-pass and high-pass spectral filters to generate embeddings that are both structure-aware and highly discriminative. To address latent space misalignment, we incorporate a geometry-aware functional map module that learns bijective and isometric transformations between graph embeddings, ensuring consistent geometric relationships across different representations. Extensive experiments on graph benchmarks demonstrate that our method consistently outperforms existing unsupervised alignment baselines, exhibiting superior robustness to structural inconsistencies and challenging alignment scenarios. Additionally, comprehensive evaluation on vision-language benchmarks using diverse pretrained models shows that our framework effectively generalizes beyond graph domains, enabling unsupervised alignment of vision and language representations. 

**Abstract (ZH)**: 图对齐——即在多个图中识别对应节点的问题——是众多应用的基础。现有的大多数无监督方法通过将节点特征嵌入到潜在表示中，以在没有地面truth对应关系的情况下进行跨图比较。然而，这些方法面临着两个关键限制：基于GNN的嵌入中节点区分度下降的泛化过度平滑现象，以及由于结构噪声、特征异质性和训练不稳定性导致的跨图潜在空间对齐不良，最终导致节点对应关系不可靠。我们提出了一种新颖的图对齐框架，该框架同时增强了节点的区分度并确保跨潜在空间的一致几何一致性。我们的方法引入了一种双通道编码器，结合低通和高通谱滤波器生成结构意识强且高度区分的嵌入。为了解决潜在空间对齐不良的问题，我们采用了一种几何意识的功能映射模块，学习图嵌入之间的双射和等距变换，从而在不同的表示之间保持一致的几何关系。在广泛的图基准实验中，我们的方法在所有无监督对齐基线方法中表现优异，展现出对结构不一致和挑战性对齐场景的优越鲁棒性。此外，通过对多种预训练模型在视觉-语言基准上的全面评估，证明了我们框架的有效泛化能力，使其能够超越图域，实现视觉和语言表示的无监督对齐。 

---
# Invisible Attributes, Visible Biases: Exploring Demographic Shortcuts in MRI-based Alzheimer's Disease Classification 

**Title (ZH)**: 隐形属性，显性偏见：基于MRI的阿尔茨海默病分类中的人口统计捷径探索 

**Authors**: Akshit Achara, Esther Puyol Anton, Alexander Hammers, Andrew P. King  

**Link**: [PDF](https://arxiv.org/pdf/2509.09558)  

**Abstract**: Magnetic resonance imaging (MRI) is the gold standard for brain imaging. Deep learning (DL) algorithms have been proposed to aid in the diagnosis of diseases such as Alzheimer's disease (AD) from MRI scans. However, DL algorithms can suffer from shortcut learning, in which spurious features, not directly related to the output label, are used for prediction. When these features are related to protected attributes, they can lead to performance bias against underrepresented protected groups, such as those defined by race and sex. In this work, we explore the potential for shortcut learning and demographic bias in DL based AD diagnosis from MRI. We first investigate if DL algorithms can identify race or sex from 3D brain MRI scans to establish the presence or otherwise of race and sex based distributional shifts. Next, we investigate whether training set imbalance by race or sex can cause a drop in model performance, indicating shortcut learning and bias. Finally, we conduct a quantitative and qualitative analysis of feature attributions in different brain regions for both the protected attribute and AD classification tasks. Through these experiments, and using multiple datasets and DL models (ResNet and SwinTransformer), we demonstrate the existence of both race and sex based shortcut learning and bias in DL based AD classification. Our work lays the foundation for fairer DL diagnostic tools in brain MRI. The code is provided at this https URL 

**Abstract (ZH)**: 磁共振成像（MRI）是脑部成像的金标准。深度学习（DL）算法已被提出用于从MRI扫描中诊断阿尔茨海默病（AD）等疾病。然而，DL算法可能会遭受捷径学习的问题，即使用与输出标签无直接关系的虚假特征进行预测。当这些特征与保护性特征相关时，它们可能导致模型性能对少数代表性不足的保护性群体产生偏差，例如按照种族和性别定义的群体。在本文中，我们探讨了基于DL的AD MRI诊断中捷径学习和人口统计偏差的潜在性。我们首先调查DL算法是否可以从3D脑部MRI扫描中识别种族或性别，以确定是否存在基于种族或性别的分布变化。接下来，我们研究训练集中的种族或性别不平衡是否会导致模型性能下降，这表明存在捷径学习和偏差。最后，我们对不同脑区的保护性特征和AD分类任务中的特征归因进行了定量和定性分析。通过这些实验，使用多个数据集和DL模型（ResNet和SwinTransformer），我们证明了基于DL的AD分类中存在基于种族和性别的捷径学习和偏差。我们的工作为构建更公平的DL诊断工具奠定了基础。代码可在以下网址获得：this https URL。 

---
# An improved educational competition optimizer with multi-covariance learning operators for global optimization problems 

**Title (ZH)**: 基于多协方差学习操作者的改进教育竞赛优化算法用于全局优化问题 

**Authors**: Baoqi Zhao, Xiong Yang, Hoileong Lee, Bowen Dong  

**Link**: [PDF](https://arxiv.org/pdf/2509.09552)  

**Abstract**: The educational competition optimizer is a recently introduced metaheuristic algorithm inspired by human behavior, originating from the dynamics of educational competition within society. Nonetheless, ECO faces constraints due to an imbalance between exploitation and exploration, rendering it susceptible to local optima and demonstrating restricted effectiveness in addressing complex optimization problems. To address these limitations, this study presents an enhanced educational competition optimizer (IECO-MCO) utilizing multi-covariance learning operators. In IECO, three distinct covariance learning operators are introduced to improve the performance of ECO. Each operator effectively balances exploitation and exploration while preventing premature convergence of the population. The effectiveness of IECO is assessed through benchmark functions derived from the CEC 2017 and CEC 2022 test suites, and its performance is compared with various basic and improved algorithms across different categories. The results demonstrate that IECO-MCO surpasses the basic ECO and other competing algorithms in convergence speed, stability, and the capability to avoid local optima. Furthermore, statistical analyses, including the Friedman test, Kruskal-Wallis test, and Wilcoxon rank-sum test, are conducted to validate the superiority of IECO-MCO over the compared algorithms. Compared with the basic algorithm (improved algorithm), IECO-MCO achieved an average ranking of 2.213 (2.488) on the CE2017 and CEC2022 test suites. Additionally, the practical applicability of the proposed IECO-MCO algorithm is verified by solving constrained optimization problems. The experimental outcomes demonstrate the superior performance of IECO-MCO in tackling intricate optimization problems, underscoring its robustness and practical effectiveness in real-world scenarios. 

**Abstract (ZH)**: 基于多协方差学习操作的增强教育竞赛优化器（IECO-MCO） 

---
# A modified RIME algorithm with covariance learning and diversity enhancement for numerical optimization 

**Title (ZH)**: 具有协方差学习和多样性增强的修改RIME算法用于数值优化 

**Authors**: Shangqing Shi, Luoxiao Zhang, Yuchen Yin, Xiong Yang, Hoileong Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.09529)  

**Abstract**: Metaheuristics are widely applied for their ability to provide more efficient solutions. The RIME algorithm is a recently proposed physical-based metaheuristic algorithm with certain advantages. However, it suffers from rapid loss of population diversity during optimization and is prone to fall into local optima, leading to unbalanced exploitation and exploration. To address the shortcomings of RIME, this paper proposes a modified RIME with covariance learning and diversity enhancement (MRIME-CD). The algorithm applies three strategies to improve the optimization capability. First, a covariance learning strategy is introduced in the soft-rime search stage to increase the population diversity and balance the over-exploitation ability of RIME through the bootstrapping effect of dominant populations. Second, in order to moderate the tendency of RIME population to approach the optimal individual in the early search stage, an average bootstrapping strategy is introduced into the hard-rime puncture mechanism, which guides the population search through the weighted position of the dominant populations, thus enhancing the global search ability of RIME in the early stage. Finally, a new stagnation indicator is proposed, and a stochastic covariance learning strategy is used to update the stagnant individuals in the population when the algorithm gets stagnant, thus enhancing the ability to jump out of the local optimal solution. The proposed MRIME-CD algorithm is subjected to a series of validations on the CEC2017 test set, the CEC2022 test set, and the experimental results are analyzed using the Friedman test, the Wilcoxon rank sum test, and the Kruskal Wallis test. The results show that MRIME-CD can effectively improve the performance of basic RIME and has obvious superiorities in terms of solution accuracy, convergence speed and stability. 

**Abstract (ZH)**: 基于协方差学习和多样性增强的改进RIME算法（MRIME-CD） 

---
# Towards Explainable Job Title Matching: Leveraging Semantic Textual Relatedness and Knowledge Graphs 

**Title (ZH)**: 面向可解释的职位标题匹配：利用语义文本相关性和知识图谱 

**Authors**: Vadim Zadykian, Bruno Andrade, Haithem Afli  

**Link**: [PDF](https://arxiv.org/pdf/2509.09522)  

**Abstract**: Semantic Textual Relatedness (STR) captures nuanced relationships between texts that extend beyond superficial lexical similarity. In this study, we investigate STR in the context of job title matching - a key challenge in resume recommendation systems, where overlapping terms are often limited or misleading. We introduce a self-supervised hybrid architecture that combines dense sentence embeddings with domain-specific Knowledge Graphs (KGs) to improve both semantic alignment and explainability. Unlike previous work that evaluated models on aggregate performance, our approach emphasizes data stratification by partitioning the STR score continuum into distinct regions: low, medium, and high semantic relatedness. This stratified evaluation enables a fine-grained analysis of model performance across semantically meaningful subspaces. We evaluate several embedding models, both with and without KG integration via graph neural networks. The results show that fine-tuned SBERT models augmented with KGs produce consistent improvements in the high-STR region, where the RMSE is reduced by 25% over strong baselines. Our findings highlight not only the benefits of combining KGs with text embeddings, but also the importance of regional performance analysis in understanding model behavior. This granular approach reveals strengths and weaknesses hidden by global metrics, and supports more targeted model selection for use in Human Resources (HR) systems and applications where fairness, explainability, and contextual matching are essential. 

**Abstract (ZH)**: 语义文本相关性（STR）捕捉了文本之间的微妙关系，超出了表层词形相似性的范畴。本研究在职业衔标题匹配的背景下探讨STR，这是简历推荐系统中的一个关键挑战，其中重叠的术语往往有限或误导性。我们提出了一种自监督混合架构，结合密集句子嵌入与领域特定的知识图谱（KGs），以提高语义对齐和可解释性。与之前仅评估模型综合性能的工作不同，我们的方法强调数据分层，通过将STR分数连续性划分为不同的区域：低、中、高语义相关性。这种分层评估方法使我们能够对模型在语义上有意义的子空间中进行细粒度的性能分析。我们评估了几种嵌入模型，包括通过图神经网络整合KG的模型。结果显示，与强基线相比，细调的SBERT模型结合KG在高STR区域中产生了一致的改进，RMSE减少了25%。我们的研究不仅突出了结合KG与文本嵌入的优势，还强调了区域性能分析在理解模型行为中的重要性。这种细粒度的方法揭示了全球指标掩盖的优势和不足，并支持更具针对性的模型选择在人力资源（HR）系统和需要公平性、可解释性和上下文匹配的应用中使用。 

---
# Explainable AI for Accelerated Microstructure Imaging: A SHAP-Guided Protocol on the Connectome 2.0 scanner 

**Title (ZH)**: 可解释的人工智能加速微结构成像：基于SHAP的Connectome 2.0扫描仪协议 

**Authors**: Quentin Uhl, Tommaso Pavan, Julianna Gerold, Kwok-Shing Chan, Yohan Jun, Shohei Fujita, Aneri Bhatt, Yixin Ma, Qiaochu Wang, Hong-Hsi Lee, Susie Y. Huang, Berkin Bilgic, Ileana Jelescu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09513)  

**Abstract**: The diffusion MRI Neurite Exchange Imaging model offers a promising framework for probing gray matter microstructure by estimating parameters such as compartment sizes, diffusivities, and inter-compartmental water exchange time. However, existing protocols require long scan times. This study proposes a reduced acquisition scheme for the Connectome 2.0 scanner that preserves model accuracy while substantially shortening scan duration. We developed a data-driven framework using explainable artificial intelligence with a guided recursive feature elimination strategy to identify an optimal 8-feature subset from a 15-feature protocol. The performance of this optimized protocol was validated in vivo and benchmarked against the full acquisition and alternative reduction strategies. Parameter accuracy, preservation of anatomical contrast, and test-retest reproducibility were assessed. The reduced protocol yielded parameter estimates and cortical maps comparable to the full protocol, with low estimation errors in synthetic data and minimal impact on test-retest variability. Compared to theory-driven and heuristic reduction schemes, the optimized protocol demonstrated superior robustness, reducing the deviation in water exchange time estimates by over two-fold. In conclusion, this hybrid optimization framework enables viable imaging of neurite exchange in 14 minutes without loss of parameter fidelity. This approach supports the broader application of exchange-sensitive diffusion magnetic resonance imaging in neuroscience and clinical research, and offers a generalizable method for designing efficient acquisition protocols in biophysical parameter mapping. 

**Abstract (ZH)**: 基于扩散MRI神经突丛交换成像模型提供了一种有前景的框架，用于通过估计隔室大小、扩散系数和隔室间水交换时间等参数来探查灰质微结构。然而，现有的方案需要较长的扫描时间。本研究提出了一种适用于Connectome 2.0扫描器的简化采集方案，该方案在保留模型准确性的同时大幅缩短了扫描时间。我们利用可解释的人工智能和指导递归特征消除策略开发了一个数据驱动的框架，从15个特征协议中选择了最优的8个特征子集。对该优化协议的性能进行了体内验证，并与完整采集方案和替代的简化策略进行了基准测试。评估了参数准确性、解剖对比度的保留和测试-再测试的重复性。简化协议在合成数据中的参数估计误差较低，且对测试-再测试变异性的影响 minimal。与理论驱动和启发式简化方案相比，优化协议显示出了更强的稳健性，水交换时间估计的偏差减少了约两倍。总之，这种混合优化框架能够在14分钟内实现神经突丛交换的有效成像而不损失参数保真度。该方法支持交换敏感的扩散磁共振成像在神经科学和临床研究中的广泛应用，并提供了一种有效采集协议设计的一般化方法，用于生物物理参数映射。 

---
# Incorporating AI Incident Reporting into Telecommunications Law and Policy: Insights from India 

**Title (ZH)**: 将AI事故报告纳入电信法律与政策：来自印度的启示 

**Authors**: Avinash Agarwal, Manisha J. Nene  

**Link**: [PDF](https://arxiv.org/pdf/2509.09508)  

**Abstract**: The integration of artificial intelligence (AI) into telecommunications infrastructure introduces novel risks, such as algorithmic bias and unpredictable system behavior, that fall outside the scope of traditional cybersecurity and data protection frameworks. This paper introduces a precise definition and a detailed typology of telecommunications AI incidents, establishing them as a distinct category of risk that extends beyond conventional cybersecurity and data protection breaches. It argues for their recognition as a distinct regulatory concern. Using India as a case study for jurisdictions that lack a horizontal AI law, the paper analyzes the country's key digital regulations. The analysis reveals that India's existing legal instruments, including the Telecommunications Act, 2023, the CERT-In Rules, and the Digital Personal Data Protection Act, 2023, focus on cybersecurity and data breaches, creating a significant regulatory gap for AI-specific operational incidents, such as performance degradation and algorithmic bias. The paper also examines structural barriers to disclosure and the limitations of existing AI incident repositories. Based on these findings, the paper proposes targeted policy recommendations centered on integrating AI incident reporting into India's existing telecom governance. Key proposals include mandating reporting for high-risk AI failures, designating an existing government body as a nodal agency to manage incident data, and developing standardized reporting frameworks. These recommendations aim to enhance regulatory clarity and strengthen long-term resilience, offering a pragmatic and replicable blueprint for other nations seeking to govern AI risks within their existing sectoral frameworks. 

**Abstract (ZH)**: 人工智能在电信基础设施中的集成引入了新型风险，如算法偏见和不可预测的系统行为，这些风险超出了传统网络安全和数据保护框架的范畴。本文提出了电信人工智能事件的精确定义和详细分类，将其确立为超越传统网络安全和数据保护泄露的独立风险类别，并主张将其作为独立的监管关注点进行考虑。以印度为例，缺乏横向人工智能法律的司法管辖区，分析了该国的关键数字法规。分析显示，印度现有的法律工具，包括2023年电信法、CERT-In规则和2023年数字个人数据保护法，主要关注网络安全和数据泄露，形成了针对特定于人工智能的操作事件，如性能退化和算法偏见的显著监管缺口。本文还探讨了披露结构障碍和现有人工智能事件仓库的限制。基于这些发现，本文提出了针对印度现有电信治理的针对性政策建议，重点是将人工智能事件报告纳入其中。关键建议包括要求报告高风险的人工智能故障、指定一个现有政府机构作为节点机构来管理事件数据，并开发标准化报告框架。这些建议旨在提高监管明晰度并增强长期韧性，为其他国家在现有部门框架内治理人工智能风险提供实用且可复制的蓝图。 

---
# OpenFake: An Open Dataset and Platform Toward Large-Scale Deepfake Detection 

**Title (ZH)**: OpenFake：面向大规模深度假信息检测的开放数据集与平台 

**Authors**: Victor Livernoche, Akshatha Arodi, Andreea Musulan, Zachary Yang, Adam Salvail, Gaétan Marceau Caron, Jean-François Godbout, Reihaneh Rabbany  

**Link**: [PDF](https://arxiv.org/pdf/2509.09495)  

**Abstract**: Deepfakes, synthetic media created using advanced AI techniques, have intensified the spread of misinformation, particularly in politically sensitive contexts. Existing deepfake detection datasets are often limited, relying on outdated generation methods, low realism, or single-face imagery, restricting the effectiveness for general synthetic image detection. By analyzing social media posts, we identify multiple modalities through which deepfakes propagate misinformation. Furthermore, our human perception study demonstrates that recently developed proprietary models produce synthetic images increasingly indistinguishable from real ones, complicating accurate identification by the general public. Consequently, we present a comprehensive, politically-focused dataset specifically crafted for benchmarking detection against modern generative models. This dataset contains three million real images paired with descriptive captions, which are used for generating 963k corresponding high-quality synthetic images from a mix of proprietary and open-source models. Recognizing the continual evolution of generative techniques, we introduce an innovative crowdsourced adversarial platform, where participants are incentivized to generate and submit challenging synthetic images. This ongoing community-driven initiative ensures that deepfake detection methods remain robust and adaptive, proactively safeguarding public discourse from sophisticated misinformation threats. 

**Abstract (ZH)**: 深度伪造：一种利用先进AI技术生成的合成媒体，在政治敏感背景下加剧了假信息的传播。现有的深度伪造检测数据集往往存在局限性，依赖于过时的生成方法、低保真度或单人图像，限制了其在一般合成图像检测中的有效性。通过对社交媒体帖子的分析，我们发现多种传播假信息的模式。此外，我们的感知研究表明，最近开发的专有模型生成的合成图像越来越难以与真实图像区分开来，使公众难以准确识别。因此，我们提出一个针对政治主题的综合性数据集，用于基准测试针对现代生成模型的检测方法。该数据集包含300万张真实图像配以描述性说明，用于生成源自专有和开源模型混合的963,000张高质量合成图像。鉴于生成技术的持续演变，我们推出了一个创新的众包对抗平台，鼓励参与者生成和提交具有挑战性的合成图像。这一社区驱动的持续性举措确保检测方法保持稳健和适应性，从而积极地保护公共讨论免受复杂虚假信息的威胁。 

---
# Prompt Pirates Need a Map: Stealing Seeds helps Stealing Prompts 

**Title (ZH)**: 海盗需要一张地图：偷取提示词有助于偷取生成种子 

**Authors**: Felix Mächtle, Ashwath Shetty, Jonas Sander, Nils Loose, Sören Pirk, Thomas Eisenbarth  

**Link**: [PDF](https://arxiv.org/pdf/2509.09488)  

**Abstract**: Diffusion models have significantly advanced text-to-image generation, enabling the creation of highly realistic images conditioned on textual prompts and seeds. Given the considerable intellectual and economic value embedded in such prompts, prompt theft poses a critical security and privacy concern. In this paper, we investigate prompt-stealing attacks targeting diffusion models. We reveal that numerical optimization-based prompt recovery methods are fundamentally limited as they do not account for the initial random noise used during image generation. We identify and exploit a noise-generation vulnerability (CWE-339), prevalent in major image-generation frameworks, originating from PyTorch's restriction of seed values to a range of $2^{32}$ when generating the initial random noise on CPUs. Through a large-scale empirical analysis conducted on images shared via the popular platform CivitAI, we demonstrate that approximately 95% of these images' seed values can be effectively brute-forced in 140 minutes per seed using our seed-recovery tool, SeedSnitch. Leveraging the recovered seed, we propose PromptPirate, a genetic algorithm-based optimization method explicitly designed for prompt stealing. PromptPirate surpasses state-of-the-art methods, i.e., PromptStealer, P2HP, and CLIP-Interrogator, achieving an 8-11% improvement in LPIPS similarity. Furthermore, we introduce straightforward and effective countermeasures that render seed stealing, and thus optimization-based prompt stealing, ineffective. We have disclosed our findings responsibly and initiated coordinated mitigation efforts with the developers to address this critical vulnerability. 

**Abstract (ZH)**: 基于扩散模型的提示窃取攻击研究 

---
# We're Still Doing It (All) Wrong: Recommender Systems, Fifteen Years Later 

**Title (ZH)**: 我们仍然做错了（所有事情）：十五年后，推荐系统仍存在问题 

**Authors**: Alan Said, Maria Soledad Pera, Michael D. Ekstrand  

**Link**: [PDF](https://arxiv.org/pdf/2509.09414)  

**Abstract**: In 2011, Xavier Amatriain sounded the alarm: recommender systems research was "doing it all wrong" [1]. His critique, rooted in statistical misinterpretation and methodological shortcuts, remains as relevant today as it was then. But rather than correcting course, we added new layers of sophistication on top of the same broken foundations. This paper revisits Amatriain's diagnosis and argues that many of the conceptual, epistemological, and infrastructural failures he identified still persist, in more subtle or systemic forms. Drawing on recent work in reproducibility, evaluation methodology, environmental impact, and participatory design, we showcase how the field's accelerating complexity has outpaced its introspection. We highlight ongoing community-led initiatives that attempt to shift the paradigm, including workshops, evaluation frameworks, and calls for value-sensitive and participatory research. At the same time, we contend that meaningful change will require not only new metrics or better tooling, but a fundamental reframing of what recommender systems research is for, who it serves, and how knowledge is produced and validated. Our call is not just for technical reform, but for a recommender systems research agenda grounded in epistemic humility, human impact, and sustainable practice. 

**Abstract (ZH)**: 2011年，Xavier Amatriain 发出警告：推荐系统研究“走错了方向”[1]。虽然他的批评基于统计误读和方法论捷径，至今依然 relevant，但研究领域并未纠正航线，反而在旧有缺陷之上增加了新的复杂层。本文重新审视 Amatriain 的诊断，认为他识别出的概念性、知识论性和基础设施性失败仍然以更微妙或系统化的方式存在。借助最近在可再现性、评估方法、环境影响及参与设计等方面的工作，我们展示了研究领域的加速复杂化已经超越了其自我反省。我们强调了社区主导的倡议，这些倡议试图改变范式，包括研讨会、评估框架及呼吁价值敏感和参与式研究。同时，我们认为有意义的变革不仅需要新的指标或更好的工具，还需要根本性地重塑推荐系统研究的目的、服务对象及知识的生成与验证方式。我们呼吁的不仅仅是技术改革，而是基于知识谦逊、人类影响和可持续实践的推荐系统研究议程。 

---
# Robust Non-Linear Correlations via Polynomial Regression 

**Title (ZH)**: 稳健的非线性相关性通过多项式回归 

**Authors**: Luca Giuliani, Michele Lombardi  

**Link**: [PDF](https://arxiv.org/pdf/2509.09380)  

**Abstract**: The Hirschfeld-Gebelein-Rényi (HGR) correlation coefficient is an extension of Pearson's correlation that is not limited to linear correlations, with potential applications in algorithmic fairness, scientific analysis, and causal discovery. Recently, novel algorithms to estimate HGR in a differentiable manner have been proposed to facilitate its use as a loss regularizer in constrained machine learning applications. However, the inherent uncomputability of HGR requires a bias-variance trade-off, which can possibly compromise the robustness of the proposed methods, hence raising technical concerns if applied in real-world scenarios. We introduce a novel computational approach for HGR that relies on user-configurable polynomial kernels, offering greater robustness compared to previous methods and featuring a faster yet almost equally effective restriction. Our approach provides significant advantages in terms of robustness and determinism, making it a more reliable option for real-world applications. Moreover, we present a brief experimental analysis to validate the applicability of our approach within a constrained machine learning framework, showing that its computation yields an insightful subgradient that can serve as a loss regularizer. 

**Abstract (ZH)**: Hirschfeld-Gebelein-Rényi (HGR) 相关系数是一种扩展的皮尔逊相关系数，适用于非线性相关分析，具有在算法公平性、科学研究和因果发现等方面的应用潜力。最近，提出了以可微方式估计 HGR 的新型算法，以促进其作为约束机器学习应用中损失正则化项的使用。然而，HGR 内在的不可计算性要求在偏倚与方差之间进行权衡，这可能会损害提出方法的鲁棒性，从而在实际应用场景中引发技术关切。我们引入了一种基于可配置多项式核的新计算方法，与以往方法相比提供了更高的鲁棒性，并具备更快且几乎同样有效的限制效果。我们的方法在鲁棒性和确定性方面具有显著优势，使其成为更可靠的现实应用选择。此外，我们呈现了简要的实验分析，验证了在约束机器学习框架中应用该方法的有效性，表明其计算可以产生具有洞察力的次梯度，可用作损失正则化项。 

---
# MoSE: Unveiling Structural Patterns in Graphs via Mixture of Subgraph Experts 

**Title (ZH)**: MoSE: 通过子图专家混合体揭示图的结构模式 

**Authors**: Junda Ye, Zhongbao Zhang, Li Sun, Siqiang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2509.09337)  

**Abstract**: While graph neural networks (GNNs) have achieved great success in learning from graph-structured data, their reliance on local, pairwise message passing restricts their ability to capture complex, high-order subgraph patterns. leading to insufficient structural expressiveness. Recent efforts have attempted to enhance structural expressiveness by integrating random walk kernels into GNNs. However, these methods are inherently designed for graph-level tasks, which limits their applicability to other downstream tasks such as node classification. Moreover, their fixed kernel configurations hinder the model's flexibility in capturing diverse subgraph structures. To address these limitations, this paper proposes a novel Mixture of Subgraph Experts (MoSE) framework for flexible and expressive subgraph-based representation learning across diverse graph tasks. Specifically, MoSE extracts informative subgraphs via anonymous walks and dynamically routes them to specialized experts based on structural semantics, enabling the model to capture diverse subgraph patterns with improved flexibility and interpretability. We further provide a theoretical analysis of MoSE's expressivity within the Subgraph Weisfeiler-Lehman (SWL) Test, proving that it is more powerful than SWL. Extensive experiments, together with visualizations of learned subgraph experts, demonstrate that MoSE not only outperforms competitive baselines but also provides interpretable insights into structural patterns learned by the model. 

**Abstract (ZH)**: 具有灵活表达性的子图专家混合框架：跨多种图任务的子图表示学习 

---
# Adaptive Knowledge Distillation using a Device-Aware Teacher for Low-Complexity Acoustic Scene Classification 

**Title (ZH)**: 基于设备感知教师的自适应知识蒸馏在低复杂度声 scene 分类中的应用 

**Authors**: Seung Gyu Jeong, Seong Eun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.09262)  

**Abstract**: In this technical report, we describe our submission for Task 1, Low-Complexity Device-Robust Acoustic Scene Classification, of the DCASE 2025 Challenge. Our work tackles the dual challenges of strict complexity constraints and robust generalization to both seen and unseen devices, while also leveraging the new rule allowing the use of device labels at test time. Our proposed system is based on a knowledge distillation framework where an efficient CP-MobileNet student learns from a compact, specialized two-teacher ensemble. This ensemble combines a baseline PaSST teacher, trained with standard cross-entropy, and a 'generalization expert' teacher. This expert is trained using our novel Device-Aware Feature Alignment (DAFA) loss, adapted from prior work, which explicitly structures the feature space for device robustness. To capitalize on the availability of test-time device labels, the distilled student model then undergoes a final device-specific fine-tuning stage. Our proposed system achieves a final accuracy of 57.93\% on the development set, demonstrating a significant improvement over the official baseline, particularly on unseen devices. 

**Abstract (ZH)**: 本技术报告描述了我们参加2025 DCASE挑战任务1——低复杂度设备鲁棒声场景分类的提交内容。我们的工作解决了严格复杂度约束和在已见和未见设备上鲁棒泛化的双重挑战，并利用了新规则，允许在测试时使用设备标签。我们提出了一种基于知识蒸馏框架的系统，其中高效的CP-MobileNet学生从一个紧凑的、专门的双师集成中学习。该集成结合了一个用标准交叉熵训练的基本PaSST教师和一个“泛化专家”教师。该专家通过我们提出的设备感知特征对齐（DAFA）损失进行训练，这是一种从先前工作改编而来的损失函数，可以明确地结构化特征空间以实现设备鲁棒性。为了利用测试时设备标签的可用性，蒸馏后的学生模型随后进行最终的设备特定微调。我们提出的系统在开发集上取得了57.93%的最终准确率，相比官方基线，特别是在未见设备上，显示出显著的改进。 

---
# Incentivizing Safer Actions in Policy Optimization for Constrained Reinforcement Learning 

**Title (ZH)**: 受约束强化学习中促进更安全行为的策略优化激励方法 

**Authors**: Somnath Hazra, Pallab Dasgupta, Soumyajit Dey  

**Link**: [PDF](https://arxiv.org/pdf/2509.09208)  

**Abstract**: Constrained Reinforcement Learning (RL) aims to maximize the return while adhering to predefined constraint limits, which represent domain-specific safety requirements. In continuous control settings, where learning agents govern system actions, balancing the trade-off between reward maximization and constraint satisfaction remains a significant challenge. Policy optimization methods often exhibit instability near constraint boundaries, resulting in suboptimal training performance. To address this issue, we introduce a novel approach that integrates an adaptive incentive mechanism in addition to the reward structure to stay within the constraint bound before approaching the constraint boundary. Building on this insight, we propose Incrementally Penalized Proximal Policy Optimization (IP3O), a practical algorithm that enforces a progressively increasing penalty to stabilize training dynamics. Through empirical evaluation on benchmark environments, we demonstrate the efficacy of IP3O compared to the performance of state-of-the-art Safe RL algorithms. Furthermore, we provide theoretical guarantees by deriving a bound on the worst-case error of the optimality achieved by our algorithm. 

**Abstract (ZH)**: 受约束的 reinforcement learning (RL) 目的是在遵守预定义约束限制的同时最大化回报，这些约束代表了特定领域的安全要求。在连续控制设置中，当学习智能体管理系统行动时，要在回报最大化和约束满足之间取得平衡仍然是一个重大挑战。策略优化方法往往在接近约束边界时表现出不稳定性，导致训练性能不佳。为解决这一问题，我们提出了一种新的方法，该方法除了优化奖励结构外，还集成了一个自适应激励机制，以在接近约束边界之前保持在约束范围内。基于这一洞察，我们提出了增量惩罚近端策略优化（IP3O），这是一种能够通过逐步增加惩罚来稳定训练动力学的实用算法。通过在基准环境中进行实证评估，我们展示了与最先进的安全 RL 算法相比，IP3O 的有效性。此外，我们通过推导出我们的算法所实现最优性最坏情况误差的上界，提供了理论保证。 

---
# Improving Synthetic Data Training for Contextual Biasing Models with a Keyword-Aware Cost Function 

**Title (ZH)**: 基于关键字感知成本函数提升合成数据训练的上下文偏见模型性能 

**Authors**: Chin Yuen Kwok, Jia Qi Yip, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2509.09197)  

**Abstract**: Rare word recognition can be improved by adapting ASR models to synthetic data that includes these words. Further improvements can be achieved through contextual biasing, which trains and adds a biasing module into the model architecture to prioritize rare words. While training the module on synthetic rare word data is more effective than using non-rare-word data, it can lead to overfitting due to artifacts in the synthetic audio. To address this, we enhance the TCPGen-based contextual biasing approach and propose a keyword-aware loss function that additionally focuses on biased words when training biasing modules. This loss includes a masked cross-entropy term for biased word prediction and a binary classification term for detecting biased word positions. These two terms complementarily support the decoding of biased words during inference. By adapting Whisper to 10 hours of synthetic data, our method reduced the word error rate on the NSC Part 2 test set from 29.71% to 11.81%. 

**Abstract (ZH)**: 稀有词识别可以通过适应包含这些词汇的合成数据来提高。进一步的改进可以通过上下文偏差实现，即在模型架构中训练并添加一个偏差模块以优先考虑稀有词。虽然在合成稀有词数据上训练模块比使用非稀有词数据更有效，但可能会由于合成音频中的伪影导致过拟合。为解决这一问题，我们增强了基于TCPGen的上下文偏差方法，并提出了一种关键词感知的损失函数，该函数在训练偏差模块时还重点关注偏差词汇。该损失函数包括一个用于预测偏差词汇的掩码交叉熵项和一个用于检测偏差词汇位置的二元分类项。这两个项互补地支持推断过程中偏差词汇的解码。通过将Whisper适应10小时的合成数据，我们的方法将NSC Part 2测试集上的词错误率从29.71%降低到11.81%。 

---
# Efficient Trie-based Biasing using K-step Prediction for Rare Word Recognition 

**Title (ZH)**: 基于 Trie 结构的 K 步预测高效偏置算法及其在罕见词识别中的应用 

**Authors**: Chin Yuen Kwok, Jia Qi yip  

**Link**: [PDF](https://arxiv.org/pdf/2509.09196)  

**Abstract**: Contextual biasing improves rare word recognition of ASR models by prioritizing the output of rare words during decoding. A common approach is Trie-based biasing, which gives "bonus scores" to partial hypothesis (e.g. "Bon") that may lead to the generation of the rare word (e.g. "Bonham"). If the full word ("Bonham") isn't ultimately recognized, the system revokes those earlier bonuses. This revocation is limited to beam search and is computationally expensive, particularly for models with large decoders. To overcome these limitations, we propose adapting ASR models to look ahead and predict multiple steps at once. This avoids the revocation step entirely by better estimating whether a partial hypothesis will lead to the generation of the full rare word. By fine-tuning Whisper with only 10 hours of synthetic data, our method reduces the word error rate on the NSC Part 2 test set from 30.86% to 12.19%. 

**Abstract (ZH)**: 上下文偏差增强ASR模型对稀有词的识别能力通过在解码过程中优先处理稀有词的输出。一种常见方法是基于Trie的偏差方法，该方法会给可能导致生成稀有词（例如“Bonham”）的部分假设（例如“Bon”）加分。如果最终未能识别出完整的词（例如“Bonham”），系统会撤销这些早期加分。这种撤销仅限于束搜索，对于具有大型解码器的模型来说计算成本高昂。为克服这些局限，我们提出调整ASR模型以向前预测多个步骤，从而完全避免撤销步骤，更好地估计部分假设是否会导致生成完整的稀有词。仅通过微调Whisper使用10小时合成数据，我们的方法将NSC Part 2测试集上的单词错误率从30.86%降低到12.19%。 

---
# HISPASpoof: A New Dataset For Spanish Speech Forensics 

**Title (ZH)**: HISPASpoof: 一个新的西班牙语语音鉴真数据集 

**Authors**: Maria Risques, Kratika Bhagtani, Amit Kumar Singh Yadav, Edward J. Delp  

**Link**: [PDF](https://arxiv.org/pdf/2509.09155)  

**Abstract**: Zero-shot Voice Cloning (VC) and Text-to-Speech (TTS) methods have advanced rapidly, enabling the generation of highly realistic synthetic speech and raising serious concerns about their misuse. While numerous detectors have been developed for English and Chinese, Spanish-spoken by over 600 million people worldwide-remains underrepresented in speech forensics. To address this gap, we introduce HISPASpoof, the first large-scale Spanish dataset designed for synthetic speech detection and attribution. It includes real speech from public corpora across six accents and synthetic speech generated with six zero-shot TTS systems. We evaluate five representative methods, showing that detectors trained on English fail to generalize to Spanish, while training on HISPASpoof substantially improves detection. We also evaluate synthetic speech attribution performance on HISPASpoof, i.e., identifying the generation method of synthetic speech. HISPASpoof thus provides a critical benchmark for advancing reliable and inclusive speech forensics in Spanish. 

**Abstract (ZH)**: 零样本语音克隆(VC)和文本到语音(TTS)方法取得了 rapid进展，使得生成高度逼真的合成语音成为可能，并引发了对其误用的严重担忧。虽然针对英汉的检测器已有广泛开发，但 spoken于全球超过6亿人口中的西班牙语，在语音取证方面仍然严重代表性不足。为填补这一空白，我们引入了 HISPASpoof，这是首个专门为合成语音检测和归因设计的大型西班牙语数据集。该数据集包含来自六个口音的公共语料库的真实语音以及使用六个零样本TTS系统生成的合成语音。我们评估了五种代表性方法，结果表明，针对英语训练的检测器无法泛化到西班牙语，而使用HISPASpoof训练显著提高了检测效果。我们还评估了HISPASpoof上的合成语音归因性能，即识别合成语音的生成方法。因此，HISPASpoof为推进可靠且包容的西班牙语语音取证提供了关键基准。 

---
# ViRanker: A BGE-M3 & Blockwise Parallel Transformer Cross-Encoder for Vietnamese Reranking 

**Title (ZH)**: ViRanker: 基于BGE-M3及块级并行变压器交叉编码的越南语重排模型 

**Authors**: Phuong-Nam Dang, Kieu-Linh Nguyen, Thanh-Hieu Pham  

**Link**: [PDF](https://arxiv.org/pdf/2509.09131)  

**Abstract**: This paper presents ViRanker, a cross-encoder reranking model tailored to the Vietnamese language. Built on the BGE-M3 encoder and enhanced with the Blockwise Parallel Transformer, ViRanker addresses the lack of competitive rerankers for Vietnamese, a low-resource language with complex syntax and diacritics. The model was trained on an 8 GB curated corpus and fine-tuned with hybrid hard-negative sampling to strengthen robustness. Evaluated on the MMARCO-VI benchmark, ViRanker achieves strong early-rank accuracy, surpassing multilingual baselines and competing closely with PhoRanker. By releasing the model openly on Hugging Face, we aim to support reproducibility and encourage wider adoption in real-world retrieval systems. Beyond Vietnamese, this study illustrates how careful architectural adaptation and data curation can advance reranking in other underrepresented languages. 

**Abstract (ZH)**: ViRanker：一种针对越南语的跨编码重排模型 

---
# Automated Classification of Tutors' Dialogue Acts Using Generative AI: A Case Study Using the CIMA Corpus 

**Title (ZH)**: 使用生成式AI对导师对话行为进行自动分类：基于CIMA语料库的案例研究 

**Authors**: Liqun He, Jiaqi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09125)  

**Abstract**: This study explores the use of generative AI for automating the classification of tutors' Dialogue Acts (DAs), aiming to reduce the time and effort required by traditional manual coding. This case study uses the open-source CIMA corpus, in which tutors' responses are pre-annotated into four DA categories. Both GPT-3.5-turbo and GPT-4 models were tested using tailored prompts. Results show that GPT-4 achieved 80% accuracy, a weighted F1-score of 0.81, and a Cohen's Kappa of 0.74, surpassing baseline performance and indicating substantial agreement with human annotations. These findings suggest that generative AI has strong potential to provide an efficient and accessible approach to DA classification, with meaningful implications for educational dialogue analysis. The study also highlights the importance of task-specific label definitions and contextual information in enhancing the quality of automated annotation. Finally, it underscores the ethical considerations associated with the use of generative AI and the need for responsible and transparent research practices. The script of this research is publicly available at this https URL. 

**Abstract (ZH)**: 本研究探讨了生成式AI在自动分类辅导对话行为（DAs）中的应用，旨在减少传统手动编码所需的时间和 effort。该案例研究使用开源的CIMA语料库，在该语料库中，辅导者的响应已被标注为四个DA类别。测试了GPT-3.5-turbo和GPT-4模型，并使用了定制的提示。结果显示，GPT-4实现了80%的准确率、加权F1分数为0.81、科恩κ系数为0.74，超过了基线性能，并表明与人类标注存在显著一致性。这些发现表明，生成式AI在提供高效且易 accessibility 的DA分类方法方面具有巨大潜力，对于教育对话分析具有重要含义。此外，研究还强调了特定任务标签定义和上下文信息在提高自动标注质量方面的的重要性。最后，研究突显了生成式AI使用的伦理考虑及负责任、透明的研究实践的重要性。该研究的代码在此httpsURL公开可用。 

---
# STRIDE: Scalable and Interpretable XAI via Subset-Free Functional Decomposition 

**Title (ZH)**: STRIDE: 面向子集无损的功能分解以实现可解释性的人工智能 scalability 和可解释性 

**Authors**: Chaeyun Ko  

**Link**: [PDF](https://arxiv.org/pdf/2509.09070)  

**Abstract**: Most explainable AI (XAI) frameworks face two practical limitations: the exponential cost of reasoning over feature subsets and the reduced expressiveness of summarizing effects as single scalar values. We present STRIDE, a scalable framework that aims to mitigate both issues by framing explanation as a subset-enumeration-free, orthogonal functional decomposition in a Reproducing Kernel Hilbert Space (RKHS). Rather than focusing only on scalar attributions, STRIDE computes functional components f_S(x_S) via an analytical projection scheme based on a recursive kernel-centering procedure, avoiding explicit subset enumeration. In the tabular setups we study, the approach is model-agnostic, provides both local and global views, and is supported by theoretical results on orthogonality and L^2 convergence under stated assumptions. On public tabular benchmarks in our environment, we observed speedups ranging from 0.6 times (slower than TreeSHAP on a small dataset) to 9.7 times (California), with a median approximate 3.0 times across 10 datasets, while maintaining high fidelity (R^2 between 0.81 and 0.999) and substantial rank agreement on most datasets. Overall, STRIDE complements scalar attribution methods by offering a structured functional perspective, enabling novel diagnostics like 'component surgery' to quantitatively measure the impact of specific interactions within our experimental scope. 

**Abstract (ZH)**: STRIDE：一种缓解特征子集枚举和效果单一标量值总结问题的可解释AI框架 

---
# A Scoping Review of Machine Learning Applications in Power System Protection and Disturbance Management 

**Title (ZH)**: 机器学习在电力系统保护与扰动管理中的应用综述 

**Authors**: Julian Oelhaf, Georg Kordowich, Mehran Pashaei, Christian Bergler, Andreas Maier, Johann Jäger, Siming Bayer  

**Link**: [PDF](https://arxiv.org/pdf/2509.09053)  

**Abstract**: The integration of renewable and distributed energy resources reshapes modern power systems, challenging conventional protection schemes. This scoping review synthesizes recent literature on machine learning (ML) applications in power system protection and disturbance management, following the PRISMA for Scoping Reviews framework. Based on over 100 publications, three key objectives are addressed: (i) assessing the scope of ML research in protection tasks; (ii) evaluating ML performance across diverse operational scenarios; and (iii) identifying methods suitable for evolving grid conditions. ML models often demonstrate high accuracy on simulated datasets; however, their performance under real-world conditions remains insufficiently validated. The existing literature is fragmented, with inconsistencies in methodological rigor, dataset quality, and evaluation metrics. This lack of standardization hampers the comparability of results and limits the generalizability of findings. To address these challenges, this review introduces a ML-oriented taxonomy for protection tasks, resolves key terminological inconsistencies, and advocates for standardized reporting practices. It further provides guidelines for comprehensive dataset documentation, methodological transparency, and consistent evaluation protocols, aiming to improve reproducibility and enhance the practical relevance of research outcomes. Critical gaps remain, including the scarcity of real-world validation, insufficient robustness testing, and limited consideration of deployment feasibility. Future research should prioritize public benchmark datasets, realistic validation methods, and advanced ML architectures. These steps are essential to move ML-based protection from theoretical promise to practical deployment in increasingly dynamic and decentralized power systems. 

**Abstract (ZH)**: 可再生和分布式能源资源的集成重塑了现代电力系统，挑战了传统的保护方案。本综述性研究遵循PRISMA for Scoping Reviews框架，综合分析了机器学习在电力系统保护与扰动管理中的应用文献，基于超过100篇出版物，重点探讨了三大目标：（i）评估机器学习在保护任务中的研究范围；（ii）评估机器学习在不同运行场景中的性能；（iii）识别适用于不断变化的电网条件的方法。机器学习模型在模拟数据集上通常表现出较高的准确性，但在现实世界条件下的性能仍缺乏充分验证。现有文献碎片化，方法学严谨性、数据集质量及评估指标存在不一致。缺乏标准化限制了结果的可比性和发现的一般适用性。为应对这些挑战，本综述提出了面向机器学习的保护任务分类体系，解决了关键术语的一致性问题，并倡导标准化报告实践。此外，提供了全面的数据集文档、方法学透明及一致评估协议的指导原则，旨在提高可重复性并增强研究成果的实际相关性。现有的研究仍存在现实验证不足、鲁棒性测试不够以及部署可行性考虑有限的关键空白。未来研究应优先采用公共基准数据集、现实验证方法及先进机器学习架构。这样做对于将基于机器学习的保护从理论潜力转化为在日益动态和分布式的电力系统中的实际部署至关重要。 

---
# MoWE : A Mixture of Weather Experts 

**Title (ZH)**: MoWE : 一种混合天气专家系统 

**Authors**: Dibyajyoti Chakraborty, Romit Maulik, Peter Harrington, Dallas Foster, Mohammad Amin Nabian, Sanjay Choudhry  

**Link**: [PDF](https://arxiv.org/pdf/2509.09052)  

**Abstract**: Data-driven weather models have recently achieved state-of-the-art performance, yet progress has plateaued in recent years. This paper introduces a Mixture of Experts (MoWE) approach as a novel paradigm to overcome these limitations, not by creating a new forecaster, but by optimally combining the outputs of existing models. The MoWE model is trained with significantly lower computational resources than the individual experts. Our model employs a Vision Transformer-based gating network that dynamically learns to weight the contributions of multiple "expert" models at each grid point, conditioned on forecast lead time. This approach creates a synthesized deterministic forecast that is more accurate than any individual component in terms of Root Mean Squared Error (RMSE). Our results demonstrate the effectiveness of this method, achieving up to a 10% lower RMSE than the best-performing AI weather model on a 2-day forecast horizon, significantly outperforming individual experts as well as a simple average across experts. This work presents a computationally efficient and scalable strategy to push the state of the art in data-driven weather prediction by making the most out of leading high-quality forecast models. 

**Abstract (ZH)**: 基于专家混合的天气预测新范式：超越现有极限以更低计算资源实现更优性能 

---
# Envy-Free but Still Unfair: Envy-Freeness Up To One Item (EF-1) in Personalized Recommendation 

**Title (ZH)**: 嫉妒心免费但仍可能不公平：单一物品嫉妒心免费（EF-1）在个性化推荐中的应用 

**Authors**: Amanda Aird, Ben Armstrong, Nicholas Mattei, Robin Burke  

**Link**: [PDF](https://arxiv.org/pdf/2509.09037)  

**Abstract**: Envy-freeness and the relaxation to Envy-freeness up to one item (EF-1) have been used as fairness concepts in the economics, game theory, and social choice literatures since the 1960s, and have recently gained popularity within the recommendation systems communities. In this short position paper we will give an overview of envy-freeness and its use in economics and recommendation systems; and illustrate why envy is not appropriate to measure fairness for use in settings where personalization plays a role. 

**Abstract (ZH)**: envy-freeness 和 Envy-freeness up to one item (EF-1) 自20世纪60年代以来在经济学、博弈论和社会选择文献中作为公平性的概念被使用，并且最近在推荐系统社区中也变得流行。在本文中，我们将概述 envy-freeness 及其在经济学和推荐系统中的应用；并说明在个性化起作用的环境中，envy 并不适合用来衡量公平性。 

---
# Personalized Sleep Prediction via Deep Adaptive Spatiotemporal Modeling and Sparse Data 

**Title (ZH)**: 基于深度自适应时空建模的个性化睡眠预测 

**Authors**: Xueyi Wang, C. J. C., Lamoth, Elisabeth Wilhelm  

**Link**: [PDF](https://arxiv.org/pdf/2509.09018)  

**Abstract**: A sleep forecast allows individuals and healthcare providers to anticipate and proactively address factors influencing restful rest, ultimately improving mental and physical well-being. This work presents an adaptive spatial and temporal model (AdaST-Sleep) for predicting sleep scores. Our proposed model combines convolutional layers to capture spatial feature interactions between multiple features and recurrent neural network layers to handle longer-term temporal health-related data. A domain classifier is further integrated to generalize across different subjects. We conducted several experiments using five input window sizes (3, 5, 7, 9, 11 days) and five predicting window sizes (1, 3, 5, 7, 9 days). Our approach consistently outperformed four baseline models, achieving its lowest RMSE (0.282) with a seven-day input window and a one-day predicting window. Moreover, the method maintained strong performance even when forecasting multiple days into the future, demonstrating its versatility for real-world applications. Visual comparisons reveal that the model accurately tracks both the overall sleep score level and daily fluctuations. These findings prove that the proposed framework provides a robust and adaptable solution for personalized sleep forecasting using sparse data from commercial wearable devices and domain adaptation techniques. 

**Abstract (ZH)**: 一种睡眠预测方法使个人和医疗保健提供者能够提前预见并主动应对影响良好睡眠的因素，从而改善身心福祉。本工作提出了一种自适应空间和时间模型（AdaST-Sleep）用于预测睡眠评分。我们提出的模型结合了卷积层以捕捉多个特征之间的空间特征交互，并结合递归神经网络层以处理长期的健康相关时间序列数据。还集成了一个领域分类器以实现跨不同受试者的泛化能力。我们使用五种输入窗口大小（3, 5, 7, 9, 11天）和五种预测窗口大小（1, 3, 5, 7, 9天）进行了多项实验。我们的方法在所有基线模型中表现最佳，使用7天输入窗口和1天预测窗口时RMSE最低（0.282）。此外，该方法在多天预测时仍保持强大的性能，证明了其在实际应用中的 versatility。视觉比较表明，该模型能够准确跟踪整体睡眠评分水平和每日波动。这些发现证明了所提出的框架能够利用商业可穿戴设备的稀疏数据和领域适应技术提供一种稳健且灵活的个性化睡眠预测解决方案。 

---
# Implicit Neural Representations of Intramyocardial Motion and Strain 

**Title (ZH)**: 隐式神经表示中的内心肌运动和应变 

**Authors**: Andrew Bell, Yan Kit Choi, Steffen Peterson, Andrew King, Muhummad Sohaib Nazir, Alistair Young  

**Link**: [PDF](https://arxiv.org/pdf/2509.09004)  

**Abstract**: Automatic quantification of intramyocardial motion and strain from tagging MRI remains an important but challenging task. We propose a method using implicit neural representations (INRs), conditioned on learned latent codes, to predict continuous left ventricular (LV) displacement -- without requiring inference-time optimisation. Evaluated on 452 UK Biobank test cases, our method achieved the best tracking accuracy (2.14 mm RMSE) and the lowest combined error in global circumferential (2.86%) and radial (6.42%) strain compared to three deep learning baselines. In addition, our method is $\sim$380$\times$ faster than the most accurate baseline. These results highlight the suitability of INR-based models for accurate and scalable analysis of myocardial strain in large CMR datasets. 

**Abstract (ZH)**: 从标记MRI自动量化心肌运动和应变仍是一项重要但具有挑战性的工作。我们提出了一种方法，使用条件化隐式神经表示（INRs）预测左心室（LV）连续位移——无需在推断时进行优化。在452个UK Biobank测试案例上的评估表明，我们的方法在综合环向应变（2.86%）和径向应变（6.42%）的总误差上低于三种深度学习基线，并实现了最佳的跟踪精度（2.14 mm RMSE）。此外，我们的方法比最准确的基线快约380倍。这些结果突显了基于INR的模型在大型CMR数据集中准确且可扩展地分析心肌应变的适用性。 

---
# Similarity-based Outlier Detection for Noisy Object Re-Identification Using Beta Mixtures 

**Title (ZH)**: 基于Beta混合模型的噪声对象重识别相似性异常检测 

**Authors**: Waqar Ahmad, Evan Murphy, Vladimir A. Krylov  

**Link**: [PDF](https://arxiv.org/pdf/2509.08926)  

**Abstract**: Object re-identification (Re-ID) methods are highly sensitive to label noise, which typically leads to significant performance degradation. We address this challenge by reframing Re-ID as a supervised image similarity task and adopting a Siamese network architecture trained to capture discriminative pairwise relationships. Central to our approach is a novel statistical outlier detection (OD) framework, termed Beta-SOD (Beta mixture Similarity-based Outlier Detection), which models the distribution of cosine similarities between embedding pairs using a two-component Beta distribution mixture model. We establish a novel identifiability result for mixtures of two Beta distributions, ensuring that our learning task is this http URL proposed OD step complements the Re-ID architecture combining binary cross-entropy, contrastive, and cosine embedding losses that jointly optimize feature-level similarity this http URL demonstrate the effectiveness of Beta-SOD in de-noising and Re-ID tasks for person Re-ID, on CUHK03 and Market-1501 datasets, and vehicle Re-ID, on VeRi-776 dataset. Our method shows superior performance compared to the state-of-the-art methods across various noise levels (10-30\%), demonstrating both robustness and broad applicability in noisy Re-ID scenarios. The implementation of Beta-SOD is available at: this https URL 

**Abstract (ZH)**: 基于Beta混合相似度的统计离群点检测在重识别中的应用 

---
# Instance-Optimal Matrix Multiplicative Weight Update and Its Quantum Applications 

**Title (ZH)**: 实例最优矩阵乘法权重更新及其量子应用 

**Authors**: Weiyuan Gong, Tongyang Li, Xinzhao Wang, Zhiyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.08911)  

**Abstract**: The Matrix Multiplicative Weight Update (MMWU) is a seminal online learning algorithm with numerous applications. Applied to the matrix version of the Learning from Expert Advice (LEA) problem on the $d$-dimensional spectraplex, it is well known that MMWU achieves the minimax-optimal regret bound of $O(\sqrt{T\log d})$, where $T$ is the time horizon. In this paper, we present an improved algorithm achieving the instance-optimal regret bound of $O(\sqrt{T\cdot S(X||d^{-1}I_d)})$, where $X$ is the comparator in the regret, $I_d$ is the identity matrix, and $S(\cdot||\cdot)$ denotes the quantum relative entropy. Furthermore, our algorithm has the same computational complexity as MMWU, indicating that the improvement in the regret bound is ``free''.
Technically, we first develop a general potential-based framework for matrix LEA, with MMWU being its special case induced by the standard exponential potential. Then, the crux of our analysis is a new ``one-sided'' Jensen's trace inequality built on a Laplace transform technique, which allows the application of general potential functions beyond exponential to matrix LEA. Our algorithm is finally induced by an optimal potential function from the vector LEA problem, based on the imaginary error function.
Complementing the above, we provide a memory lower bound for matrix LEA, and explore the applications of our algorithm in quantum learning theory. We show that it outperforms the state of the art for learning quantum states corrupted by depolarization noise, random quantum states, and Gibbs states. In addition, applying our algorithm to linearized convex losses enables predicting nonlinear quantum properties, such as purity, quantum virtual cooling, and Rényi-$2$ correlation. 

**Abstract (ZH)**: 矩阵乘法权重更新算法（MMWU）：从专家建议中学习的先驱在线学习算法及其在谱谱线上的应用，其最小最大化遗憾边界为$O(\sqrt{T\cdot S(X||d^{-1}I_d)})$，并通过量子相对熵表示。 

---
# A vibe coding learning design to enhance EFL students' talking to, through, and about AI 

**Title (ZH)**: 一种情绪编码学习设计以增强英语作为外语学生与人工智能交谈、关于人工智能交谈以及通过人工智能交谈的能力 

**Authors**: David James Woo, Kai Guo, Yangyang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.08854)  

**Abstract**: This innovative practice article reports on the piloting of vibe coding (using natural language to create software applications with AI) for English as a Foreign Language (EFL) education. We developed a human-AI meta-languaging framework with three dimensions: talking to AI (prompt engineering), talking through AI (negotiating authorship), and talking about AI (mental models of AI). Using backward design principles, we created a four-hour workshop where two students designed applications addressing authentic EFL writing challenges. We adopted a case study methodology, collecting data from worksheets and video recordings, think-aloud protocols, screen recordings, and AI-generated images. Contrasting cases showed one student successfully vibe coding a functional application cohering to her intended design, while another encountered technical difficulties with major gaps between intended design and actual functionality. Analysis reveals differences in students' prompt engineering approaches, suggesting different AI mental models and tensions in attributing authorship. We argue that AI functions as a beneficial languaging machine, and that differences in how students talk to, through, and about AI explain vibe coding outcome variations. Findings indicate that effective vibe coding instruction requires explicit meta-languaging scaffolding, teaching structured prompt engineering, facilitating critical authorship discussions, and developing vocabulary for articulating AI mental models. 

**Abstract (ZH)**: 本创新实践文章报道了在外国语言教学（EFL）中使用vibe coding（利用自然语言创建具有AI的应用程序）的试点研究。我们开发了一个包含三个维度的人工智能元语言框架：与AI对话（提示工程）、通过AI对话（谈判作者身份）以及关于AI的对话（AI的心理模型）。基于逆向设计原则，我们创建了一个四小时的工作坊，让学生设计解决真实EFL写作挑战的应用程序。我们采用了案例研究方法，收集了工作表、视频录音、思考 aloud 协议、屏幕录制和AI生成的图像的数据。对比案例显示，一名学生成功地使用vibe coding创建了一个功能齐全且与她预期设计一致的应用程序，而另一名学生则遇到了技术困难，设计意图与实际功能之间存在重大差距。分析表明，学生在提示工程方面的不同方法，显示出不同的AI心理模型以及在确定作者身份方面存在的张力。我们认为，AI作为一个有益的言语工具发挥作用，学生与、通过以及关于AI的对话方式差异解释了vibe coding结果的差异。研究结果表明，有效的vibe coding教学需要明确的元语言支架，教授结构化的提示工程、促进关键的作者身份讨论，并发展表达AI心理模型的词汇。 

---
# Safe and Certifiable AI Systems: Concepts, Challenges, and Lessons Learned 

**Title (ZH)**: 安全可认证的AI系统：概念、挑战及经验教训 

**Authors**: Kajetan Schweighofer, Barbara Brune, Lukas Gruber, Simon Schmid, Alexander Aufreiter, Andreas Gruber, Thomas Doms, Sebastian Eder, Florian Mayer, Xaver-Paul Stadlbauer, Christoph Schwald, Werner Zellinger, Bernhard Nessler, Sepp Hochreiter  

**Link**: [PDF](https://arxiv.org/pdf/2509.08852)  

**Abstract**: There is an increasing adoption of artificial intelligence in safety-critical applications, yet practical schemes for certifying that AI systems are safe, lawful and socially acceptable remain scarce. This white paper presents the TÜV AUSTRIA Trusted AI framework an end-to-end audit catalog and methodology for assessing and certifying machine learning systems. The audit catalog has been in continuous development since 2019 in an ongoing collaboration with scientific partners. Building on three pillars - Secure Software Development, Functional Requirements, and Ethics & Data Privacy - the catalog translates the high-level obligations of the EU AI Act into specific, testable criteria. Its core concept of functional trustworthiness couples a statistically defined application domain with risk-based minimum performance requirements and statistical testing on independently sampled data, providing transparent and reproducible evidence of model quality in real-world settings. We provide an overview of the functional requirements that we assess, which are oriented on the lifecycle of an AI system. In addition, we share some lessons learned from the practical application of the audit catalog, highlighting common pitfalls we encountered, such as data leakage scenarios, inadequate domain definitions, neglect of biases, or a lack of distribution drift controls. We further discuss key aspects of certifying AI systems, such as robustness, algorithmic fairness, or post-certification requirements, outlining both our current conclusions and a roadmap for future research. In general, by aligning technical best practices with emerging European standards, the approach offers regulators, providers, and users a practical roadmap for legally compliant, functionally trustworthy, and certifiable AI systems. 

**Abstract (ZH)**: 人工智能在关键应用中的采用日益增多，但实际的方案以确保AI系统安全、合法且社会可接受仍显匮乏。本白皮书介绍了奥地利技术监督协会（TÜV AUSTRIA）可信人工智能框架，提供了一个端到端的审计目录和评估及认证机器学习系统的 methodology。该审计目录自2019年起不断开发，在与科学伙伴的持续合作中不断完善。该目录基于三大支柱——安全软件开发、功能需求、伦理与数据隐私，将欧盟AI法案中的高层义务转化为具体的、可测试的标准。其核心概念功能可信性将统计定义的应用领域与基于风险的最低性能要求以及独立采样的统计测试相结合，提供透明且可重复的证据，证明模型在实际环境中的质量。我们概述了我们评估的功能需求，这些需求基于AI系统的生命周期。此外，我们还分享了在实际应用审计目录时的一些经验教训，指出了常见的陷阱，如数据泄露场景、不充分的领域定义、忽视偏差或缺乏分布漂移控制。我们进一步讨论了认证AI系统的关键方面，如鲁棒性、算法公平性或认证后要求，概述了我们当前的结论和未来研究的路线图。总而言之，通过将技术最佳实践与新兴的欧洲标准对齐，该方法为监管者、供应商和用户提供了一条实用的道路，以实现合法合规、功能可信且可认证的AI系统。 

---
# Uncertainty Estimation using Variance-Gated Distributions 

**Title (ZH)**: 基于方差门控分布的不确定性估计 

**Authors**: H. Martin Gillis, Isaac Xu, Thomas Trappenberg  

**Link**: [PDF](https://arxiv.org/pdf/2509.08846)  

**Abstract**: Evaluation of per-sample uncertainty quantification from neural networks is essential for decision-making involving high-risk applications. A common approach is to use the predictive distribution from Bayesian or approximation models and decompose the corresponding predictive uncertainty into epistemic (model-related) and aleatoric (data-related) components. However, additive decomposition has recently been questioned. In this work, we propose an intuitive framework for uncertainty estimation and decomposition based on the signal-to-noise ratio of class probability distributions across different model predictions. We introduce a variance-gated measure that scales predictions by a confidence factor derived from ensembles. We use this measure to discuss the existence of a collapse in the diversity of committee machines. 

**Abstract (ZH)**: 基于类概率分布信噪比的不确定性估计与分解对于高风险应用决策至关重要。我们提出了一种直观的框架，基于不同模型预测中的类概率分布信噪比进行不确定性估计与分解。引入了一种方差门控度量，通过集成获得的置信因子对预测进行缩放。我们使用该度量探讨委员会机器多样性坍缩的存在性。 

---
# Deep opacity and AI: A threat to XAI and to privacy protection mechanisms 

**Title (ZH)**: 深度不透明性与AI：对可解释AI和隐私保护机制的威胁 

**Authors**: Vincent C. Müller  

**Link**: [PDF](https://arxiv.org/pdf/2509.08835)  

**Abstract**: It is known that big data analytics and AI pose a threat to privacy, and that some of this is due to some kind of "black box problem" in AI. I explain how this becomes a problem in the context of justification for judgments and actions. Furthermore, I suggest distinguishing three kinds of opacity: 1) the subjects do not know what the system does ("shallow opacity"), 2) the analysts do not know what the system does ("standard black box opacity"), or 3) the analysts cannot possibly know what the system might do ("deep opacity"). If the agents, data subjects as well as analytics experts, operate under opacity, then these agents cannot provide justifications for judgments that are necessary to protect privacy, e.g., they cannot give "informed consent", or guarantee "anonymity". It follows from these points that agents in big data analytics and AI often cannot make the judgments needed to protect privacy. So I conclude that big data analytics makes the privacy problems worse and the remedies less effective. As a positive note, I provide a brief outlook on technical ways to handle this situation. 

**Abstract (ZH)**: 大数据分析和AI对隐私构成威胁，其中部分原因在于某种形式的“黑盒问题”。本文解释了这一问题如何影响判断和行动的正当性，并建议区分三种类型的不透明性：1）主体不知道系统做了什么（浅层不透明性）；2）分析师不知道系统做了什么（标准黑盒不透明性）；3）分析师无法知道系统可能做了什么（深层不透明性）。如果代理人在不透明的情况下运作，即数据主体和分析专家不清楚系统的行为，那么这些代理将无法提供保护隐私所需的正当性判断，例如无法提供“知情同意”或保证“匿名性”。由此得出结论，在大数据分析和AI领域，代理人往往无法做出保护隐私所需的判断，因此大数据分析加剧了隐私问题并使解决方法的效果降低。作为积极的一面，本文简要展望了技术手段应对这一局面。 

---
