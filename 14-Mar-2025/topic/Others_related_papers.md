# Stratified Topological Autonomy for Long-Range Coordination (STALC) 

**Title (ZH)**: 分层拓扑自主性长距离协调（STALC） 

**Authors**: Cora A. Dimmig, Adam Goertz, Adam Polevoy, Mark Gonzales, Kevin C. Wolfe, Bradley Woosley, John Rogers, Joseph Moore  

**Link**: [PDF](https://arxiv.org/pdf/2503.10475)  

**Abstract**: Achieving unified multi-robot coordination and motion planning in complex environments is a challenging problem. In this paper, we present a hierarchical approach to long-range coordination, which we call Stratified Topological Autonomy for Long-Range Coordination (STALC). In particular, we look at the problem of minimizing visibility to observers and maximizing safety with a multi-robot team navigating through a hazardous environment. At its core, our approach relies on the notion of a dynamic topological graph, where the edge weights vary dynamically based on the locations of the robots in the graph. To create this dynamic topological graph, we evaluate the visibility of the robot team from a discrete set of observer locations (both adversarial and friendly), and construct a topological graph whose edge weights depend on both adversary position and robot team configuration. We then impose temporal constraints on the evolution of those edge weights based on robot team state and use Mixed-Integer Programming (MIP) to generate optimal multirobot plans through the graph. The visibility information also informs the lower layers of the autonomy stack to plan minimal visibility paths through the environment for the team of robots. Our approach presents methods to reduce the computational complexity for a team of robots that interact and coordinate across the team to accomplish a common goal. We demonstrate our approach in simulated and hardware experiments in forested and urban environments. 

**Abstract (ZH)**: 基于层次化的长距离多机器人分布式协调与路径规划方法（STALC） 

---
# HALO: Fault-Tolerant Safety Architecture For High-Speed Autonomous Racing 

**Title (ZH)**: HALO：高速自动驾驶赛车的容错安全架构 

**Authors**: Aron Harder, Amar Kulkarni, Madhur Behl  

**Link**: [PDF](https://arxiv.org/pdf/2503.10341)  

**Abstract**: The field of high-speed autonomous racing has seen significant advances in recent years, with the rise of competitions such as RoboRace and the Indy Autonomous Challenge providing a platform for researchers to develop software stacks for autonomous race vehicles capable of reaching speeds in excess of 170 mph. Ensuring the safety of these vehicles requires the software to continuously monitor for different faults and erroneous operating conditions during high-speed operation, with the goal of mitigating any unreasonable risks posed by malfunctions in sub-systems and components. This paper presents a comprehensive overview of the HALO safety architecture, which has been implemented on a full-scale autonomous racing vehicle as part of the Indy Autonomous Challenge. The paper begins with a failure mode and criticality analysis of the perception, planning, control, and communication modules of the software stack. Specifically, we examine three different types of faults - node health, data health, and behavioral-safety faults. To mitigate these faults, the paper then outlines HALO safety archetypes and runtime monitoring methods. Finally, the paper demonstrates the effectiveness of the HALO safety architecture for each of the faults, through real-world data gathered from autonomous racing vehicle trials during multi-agent scenarios. 

**Abstract (ZH)**: 高速自主赛车领域的安全架构HALO： Indy Autonomous Challenge中的综合分析与实时监控 

---
# APECS: Adaptive Personalized Control System Architecture 

**Title (ZH)**: APECS：自适应个性化控制系统架构 

**Authors**: Marius F. R. Juston, Alex Gisi, William R. Norris, Dustin Nottage, Ahmet Soylemezoglu  

**Link**: [PDF](https://arxiv.org/pdf/2503.09624)  

**Abstract**: This paper presents the Adaptive Personalized Control System (APECS) architecture, a novel framework for human-in-the-loop control. An architecture is developed which defines appropriate constraints for the system objectives. A method for enacting Lipschitz and sector bounds on the resulting controller is derived to ensure desirable control properties. An analysis of worst-case loss functions and the optimal loss function weighting is made to implement an effective training scheme. Finally, simulations are carried out to demonstrate the effectiveness of the proposed architecture. This architecture resulted in a 4.5% performance increase compared to the human operator and 9% to an unconstrained feedforward neural network trained in the same way. 

**Abstract (ZH)**: 适人交互的自适应个性化控制系统（APECS）架构：基于人工环路的新型框架 

---
# Adaptive Deadlock Avoidance for Decentralized Multi-agent Systems via CBF-inspired Risk Measurement 

**Title (ZH)**: 基于CBF启发的风险评估的分布式多agent系统自适应死锁避免方法 

**Authors**: Yanze Zhang, Yiwei Lyu, Siwon Jo, Yupeng Yang, Wenhao Luo  

**Link**: [PDF](https://arxiv.org/pdf/2503.09621)  

**Abstract**: Decentralized safe control plays an important role in multi-agent systems given the scalability and robustness without reliance on a central authority. However, without an explicit global coordinator, the decentralized control methods are often prone to deadlock -- a state where the system reaches equilibrium, causing the robots to stall. In this paper, we propose a generalized decentralized framework that unifies the Control Lyapunov Function (CLF) and Control Barrier Function (CBF) to facilitate efficient task execution and ensure deadlock-free trajectories for the multi-agent systems. As the agents approach the deadlock-related undesirable equilibrium, the framework can detect the equilibrium and drive agents away before that happens. This is achieved by a secondary deadlock resolution design with an auxiliary CBF to prevent the multi-agent systems from converging to the undesirable equilibrium. To avoid dominating effects due to the deadlock resolution over the original task-related controllers, a deadlock indicator function using CBF-inspired risk measurement is proposed and encoded in the unified framework for the agents to adaptively determine when to activate the deadlock resolution. This allows the agents to follow their original control tasks and seamlessly unlock or deactivate deadlock resolution as necessary, effectively improving task efficiency. We demonstrate the effectiveness of the proposed method through theoretical analysis, numerical simulations, and real-world experiments. 

**Abstract (ZH)**: 去中心化安全控制在多agent系统中发挥着重要作用，能够在无需依赖中央权威的情况下实现可扩展性和鲁棒性。然而，缺乏明确的全局协调器时，去中心化控制方法往往容易导致死锁——一种系统达到平衡状态，导致机器人停滞的情况。本文提出了一种通用的去中心化框架，统一了控制李雅普诺夫函数（CLF）和控制屏障函数（CBF），以促进高效的任务执行并确保多agent系统的死锁自由轨迹。当agent接近与死锁相关的不良平衡状态时，框架可以检测到这一平衡，并在发生之前引导agent远离。这通过一个辅助的死锁解决设计实现，该设计包含一个基于CBF的风险度量的死锁指示函数，以防止多agent系统向不良平衡状态收敛。为了防止死锁解决机制对原始任务相关控制器产生主导作用，提出了一种基于CBF的风险度量的死锁指示函数，并将其编码到统一框架中，使agent能够自主判断何时激活死锁解决机制。这使得agent能够遵循其原始控制任务，并在必要时无缝地解锁或停用死锁解决机制，从而有效提高任务效率。我们通过理论分析、数值仿真和实地实验证明了所提方法的有效性。 

---
# DeclareAligner: A Leap Towards Efficient Optimal Alignments for Declarative Process Model Conformance Checking 

**Title (ZH)**: DeclareAligner: 向声明性过程模型符合性检查的高效最优对齐迈出一步 

**Authors**: Jacobo Casas-Ramos, Manuel Lama, Manuel Mucientes  

**Link**: [PDF](https://arxiv.org/pdf/2503.10479)  

**Abstract**: In many engineering applications, processes must be followed precisely, making conformance checking between event logs and declarative process models crucial for ensuring adherence to desired behaviors. This is a critical area where Artificial Intelligence (AI) plays a pivotal role in driving effective process improvement. However, computing optimal alignments poses significant computational challenges due to the vast search space inherent in these models. Consequently, existing approaches often struggle with scalability and efficiency, limiting their applicability in real-world settings. This paper introduces DeclareAligner, a novel algorithm that uses the A* search algorithm, an established AI pathfinding technique, to tackle the problem from a fresh perspective leveraging the flexibility of declarative models. Key features of DeclareAligner include only performing actions that actively contribute to fixing constraint violations, utilizing a tailored heuristic to navigate towards optimal solutions, and employing early pruning to eliminate unproductive branches, while also streamlining the process through preprocessing and consolidating multiple fixes into unified actions. The proposed method is evaluated using 8,054 synthetic and real-life alignment problems, demonstrating its ability to efficiently compute optimal alignments by significantly outperforming the current state of the art. By enabling process analysts to more effectively identify and understand conformance issues, DeclareAligner has the potential to drive meaningful process improvement and management. 

**Abstract (ZH)**: 在许多工程应用中，必须严格遵循工艺流程，因此事件日志与声明性流程模型之间的符合性检查对于确保遵循预期行为至关重要。这是一个关键领域，人工智能（AI）在推动有效流程改进中扮演着关键角色。然而，计算最优对齐由于这些模型固有的庞大搜索空间而面临重大计算挑战。因此，现有方法往往在可扩展性和效率方面存在局限，限制了其在实际场景中的应用。本文提出了一种名为DeclareAligner的新算法，该算法利用A*搜索算法——一种成熟的AI路径查找技术——从一个新的角度应对这一问题，充分利用声明性模型的灵活性。DeclareAligner的关键特征包括仅执行对修复约束冲突有积极贡献的动作，利用定制的启发式方法导航至最优解，并通过早期剪枝消除无生产力分支，同时通过预处理和技术合并多个修正为统一操作来简化流程。所提出的方法通过对8,054个合成和实际对齐问题的评估，展示了其能够显著优于当前最佳方法高效计算最优对齐的能力。通过使过程分析师更有效地识别和理解符合性问题，DeclareAligner有望推动实际流程的改进和管理。 

---
# Adaptive Preference Aggregation 

**Title (ZH)**: 自适应偏好聚合 

**Authors**: Benjamin Heymann  

**Link**: [PDF](https://arxiv.org/pdf/2503.10215)  

**Abstract**: AI alignment, the challenge of ensuring AI systems act in accordance with human values, has emerged as a critical problem in the development of systems such as foundation models and recommender systems. Still, the current dominant approach, reinforcement learning with human feedback (RLHF) faces known theoretical limitations in aggregating diverse human preferences. Social choice theory provides a framework to aggregate preferences, but was not developed for the multidimensional applications typical of AI. Leveraging insights from a recently published urn process, this work introduces a preference aggregation strategy that adapts to the user's context and that inherits the good properties of the maximal lottery, a Condorcet-consistent solution concept. 

**Abstract (ZH)**: AI对齐：确保AI系统按照人类价值观行动的挑战已成为基础模型和推荐系统等系统开发中的关键问题。尽管当前主流方法（强化学习结合人类反馈）存在聚合多元人类偏好已知的理论限制，社会选择理论提供了一种聚合偏好框架，但该理论尚未为常见的多维AI应用进行开发。借鉴最近发表的 urn 过程的启示，本研究提出了一种适应用户情境的偏好聚合策略，该策略继承了最大Lottery这一Condorcet一致解概念的良好属性。 

---
# Semantic Synergy: Unlocking Policy Insights and Learning Pathways Through Advanced Skill Mapping 

**Title (ZH)**: 语义协同：通过高级技能映射解锁政策洞察和学习路径 

**Authors**: Phoebe Koundouri, Conrad Landis, Georgios Feretzakis  

**Link**: [PDF](https://arxiv.org/pdf/2503.10094)  

**Abstract**: This research introduces a comprehensive system based on state-of-the-art natural language processing, semantic embedding, and efficient search techniques for retrieving similarities and thus generating actionable insights from raw textual information. The system automatically extracts and aggregates normalized competencies from multiple documents (such as policy files and curricula vitae) and creates strong relationships between recognized competencies, occupation profiles, and related learning courses. To validate its performance, we conducted a multi-tier evaluation that included both explicit and implicit skill references in synthetic and real-world documents. The results showed near-human-level accuracy, with F1 scores exceeding 0.95 for explicit skill detection and above 0.93 for implicit mentions. The system thereby establishes a sound foundation for supporting in-depth collaboration across the AE4RIA network. The methodology involves a multi-stage pipeline based on extensive preprocessing and data cleaning, semantic embedding and segmentation via SentenceTransformer, and skill extraction using a FAISS-based search method. The extracted skills are associated with occupation frameworks (as formulated in the ESCO ontology) and with learning paths offered through the Sustainable Development Goals Academy. Moreover, interactive visualization software, implemented with Dash and Plotly, presents graphs and tables for real-time exploration and informed decision-making by those involved in policymaking, training and learning supply, career transitions, and recruitment. Overall, this system, backed by rigorous validation, offers promising prospects for improved policymaking, human resource development, and lifelong learning by providing structured and actionable insights from raw, complex textual information. 

**Abstract (ZH)**: 基于最新自然语言处理、语义嵌入和高效检索技术的综合系统，用于从原始文本信息中检索相似性并生成可操作洞察 

---
# Parallelizing Multi-objective A* Search 

**Title (ZH)**: 并行化多目标A*搜索 

**Authors**: Saman Ahmadi, Nathan R. Sturtevant, Andrea Raith, Daniel Harabor, Mahdi Jalili  

**Link**: [PDF](https://arxiv.org/pdf/2503.10075)  

**Abstract**: The Multi-objective Shortest Path (MOSP) problem is a classic network optimization problem that aims to find all Pareto-optimal paths between two points in a graph with multiple edge costs. Recent studies on multi-objective search with A* (MOA*) have demonstrated superior performance in solving difficult MOSP instances. This paper presents a novel search framework that allows efficient parallelization of MOA* with different objective orders. The framework incorporates a unique upper bounding strategy that helps the search reduce the problem's dimensionality to one in certain cases. Experimental results demonstrate that the proposed framework can enhance the performance of recent A*-based solutions, with the speed-up proportional to the problem dimension. 

**Abstract (ZH)**: 多目标最短路径（MOSP）问题是一种经典的网络优化问题，旨在在一个具有多种边成本的图中找到两个点之间的所有帕累托最优路径。最近关于使用A*的多目标搜索（MOA*）的研究证明了其在解决复杂的MOSP实例方面的优越性能。本文提出了一种新的搜索框架，该框架允许在不同的目标顺序下高效地并行化MOA*。该框架包含一种独特的上界策略，有助于在某些情况下减少问题的维度。实验结果表明，所提出的框架能够提升基于A*的解决方案的性能，加速程度与问题维度成正比。 

---
# A New Benchmark for Few-Shot Class-Incremental Learning: Redefining the Upper Bound 

**Title (ZH)**: 一个新的 Few-Shot 类增量学习基准：重定义上界 

**Authors**: Shiwon Kim, Dongjun Hwang, Sungwon Woo, Rita Singh  

**Link**: [PDF](https://arxiv.org/pdf/2503.10003)  

**Abstract**: Class-incremental learning (CIL) aims to continuously adapt to emerging classes while retaining knowledge of previously learned ones. Few-shot class-incremental learning (FSCIL) presents an even greater challenge which requires the model to learn incremental classes with only a limited number of samples. In conventional CIL, joint training is widely considered the upper bound, serving as both a benchmark and a methodological guide. However, we find that joint training fails to be a meaningful upper bound in FSCIL due to the inherent difficulty of inter-task class separation (ICS) caused by severe class imbalance. In this work, we introduce a new joint training benchmark tailored for FSCIL by integrating imbalance-aware techniques, effectively bridging the performance gap between base and incremental classes. Furthermore, we point out inconsistencies in the experimental setup and evaluation of existing FSCIL methods. To ensure fair comparisons between different FSCIL approaches and joint training, we standardize training conditions and propose a unified evaluation protocol that simultaneously considers the validation set and computational complexity. By establishing a reliable upper bound and a standardized evaluation framework for FSCIL, our work provides a clear benchmark and a practical foundation for future research. 

**Abstract (ZH)**: 面向少量样本的类增量学习（FSCIL）的目标是在保留以前学习的类的知识的同时，连续适应新兴的类。类增量学习（CIL）旨在连续适应新兴类目并保留之前学习的知识。少数样本类增量学习（FSCIL）提出了更大的挑战，要求模型仅使用有限数量的样本学习增量类目。在传统的CIL中，联合训练通常被视为上限，既是基准又是方法论指导。然而，我们发现由于严重的类不平衡引起的任务间类分离（ICS）固有难度，联合训练在FSCIL中不能成为有意义的上限。在本文中，我们通过结合不平衡感知技术引入了一种新的联合训练基准，有效地弥合了基础类和增量类之间的性能差距。此外，我们指出了现有FSCIL方法中实验设置和评估中的不一致性。为了确保不同FSCIL方法与联合训练之间的公平比较，我们标准化了训练条件并提出了一个统一的评估协议，该协议同时考虑了验证集和计算复杂度。通过为FSCIL建立可靠的上限和标准化的评估框架，我们的工作为未来的研究提供了清晰的基准和实用的基础。 

---
# Local Look-Ahead Guidance via Verifier-in-the-Loop for Automated Theorem Proving 

**Title (ZH)**: 循环验证器辅以局部前瞻引导的自动化定理证明 

**Authors**: Sara Rajaee, Kumar Pratik, Gabriele Cesa, Arash Behboodi  

**Link**: [PDF](https://arxiv.org/pdf/2503.09730)  

**Abstract**: The most promising recent methods for AI reasoning require applying variants of reinforcement learning (RL) either on rolled out trajectories from the model, even for the step-wise rewards, or large quantities of human annotated trajectory data. The reliance on the rolled-out trajectory renders the compute cost and time prohibitively high. In particular, the correctness of a reasoning trajectory can typically only be judged at its completion, leading to sparse rewards in RL or requiring expensive synthetic data generation in expert iteration-like methods. In this work, we focus on the Automatic Theorem Proving (ATP) task and propose a novel verifier-in-the-loop design, which unlike existing approaches that leverage feedback on the entire reasoning trajectory, employs an automated verifier to give intermediate feedback at each step of the reasoning process. Using Lean as the verifier, we empirically show that the step-by-step local verification produces a global improvement in the model's reasoning accuracy and efficiency. 

**Abstract (ZH)**: 最近最有前景的AI推理方法要求应用强化学习（RL）的变种，无论是针对模型展开的轨迹，即使是逐步奖励，还是大量的人工标注轨迹数据。对展开轨迹的依赖使得计算成本和时间变得难以承受。特别是，推理轨迹的正确性通常只能在其完成时才能判断，这在RL中会导致稀疏奖励，或在专家迭代方法中需要昂贵的合成数据生成。在本工作中，我们专注于自动定理证明（ATP）任务，并提出了一种新的验证者在环设计，该设计与现有利用整个推理轨迹反馈的方法不同，而是使用自动化验证器在推理过程的每一步提供中间反馈。通过使用Lean作为验证器，我们实验证明，逐步局部验证在全局上提高了模型的推理准确性和效率。 

---
# Transformers without Normalization 

**Title (ZH)**: 无需归一化的 Transformers 

**Authors**: Jiachen Zhu, Xinlei Chen, Kaiming He, Yann LeCun, Zhuang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.10622)  

**Abstract**: Normalization layers are ubiquitous in modern neural networks and have long been considered essential. This work demonstrates that Transformers without normalization can achieve the same or better performance using a remarkably simple technique. We introduce Dynamic Tanh (DyT), an element-wise operation $DyT($x$) = \tanh(\alpha $x$)$, as a drop-in replacement for normalization layers in Transformers. DyT is inspired by the observation that layer normalization in Transformers often produces tanh-like, $S$-shaped input-output mappings. By incorporating DyT, Transformers without normalization can match or exceed the performance of their normalized counterparts, mostly without hyperparameter tuning. We validate the effectiveness of Transformers with DyT across diverse settings, ranging from recognition to generation, supervised to self-supervised learning, and computer vision to language models. These findings challenge the conventional understanding that normalization layers are indispensable in modern neural networks, and offer new insights into their role in deep networks. 

**Abstract (ZH)**: 无归一化变换器：Dynamic Tanh 的引入及其效果 

---
# The Spectral Bias of Shallow Neural Network Learning is Shaped by the Choice of Non-linearity 

**Title (ZH)**: 浅层神经网络学习的光谱偏置由非线性函数的选择塑造 

**Authors**: Justin Sahs, Ryan Pyle, Fabio Anselmi, Ankit Patel  

**Link**: [PDF](https://arxiv.org/pdf/2503.10587)  

**Abstract**: Despite classical statistical theory predicting severe overfitting, modern massively overparameterized neural networks still generalize well. This unexpected property is attributed to the network's so-called implicit bias, which describes its propensity to converge to solutions that generalize effectively, among the many possible that correctly label the training data. The aim of our research is to explore this bias from a new perspective, focusing on how non-linear activation functions contribute to shaping it. First, we introduce a reparameterization which removes a continuous weight rescaling symmetry. Second, in the kernel regime, we leverage this reparameterization to generalize recent findings that relate shallow Neural Networks to the Radon transform, deriving an explicit formula for the implicit bias induced by a broad class of activation functions. Specifically, by utilizing the connection between the Radon transform and the Fourier transform, we interpret the kernel regime's inductive bias as minimizing a spectral seminorm that penalizes high-frequency components, in a manner dependent on the activation function. Finally, in the adaptive regime, we demonstrate the existence of local dynamical attractors that facilitate the formation of clusters of hyperplanes where the input to a neuron's activation function is zero, yielding alignment between many neurons' response functions. We confirm these theoretical results with simulations. All together, our work provides a deeper understanding of the mechanisms underlying the generalization capabilities of overparameterized neural networks and its relation with the implicit bias, offering potential pathways for designing more efficient and robust models. 

**Abstract (ZH)**: 尽管经典统计理论预测严重的过拟合现象，但现代高度过参数化的神经网络仍然能够泛化良好。这一意想不到的特性归因于网络所谓的隐式偏置，即其倾向于收敛到能够有效泛化的解决方案，而不是仅仅正确标记训练数据的众多可能解。我们研究的目的是从一个新的角度探索这种偏置，重点关注非线性激活函数如何塑造这种偏置。首先，我们引入了一种重新参数化方法，以消除连续权重缩放对称性。其次，在核区，我们利用这种重新参数化来推广最近与浅层神经网络相关的拉东变换的研究成果，推导出由广泛类别激活函数引发的隐式偏置的显式公式。具体来说，通过拉东变换与傅里叶变换之间的联系，我们将核区的归纳偏置解释为通过激活函数依赖的方式最小化频谱半范数，惩罚高频率分量。最后，在自适应区，我们证明存在局部动力学吸引子，有助于形成神经元激活函数输入为零的超平面簇，从而实现大量神经元响应函数的对齐。我们通过模拟证实了这些理论结果。综上所述，我们的工作为理解过度参数化神经网络泛化能力背后的机制及其与隐式偏置的关系提供了更深入的理解，并为设计更高效和鲁棒的模型提供了潜在途径。 

---
# GBSVR: Granular Ball Support Vector Regression 

**Title (ZH)**: 粒球支持向量回归：GBSVR 

**Authors**: Reshma Rastogi, Ankush Bisht, Sanjay Kumar, Suresh Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2503.10539)  

**Abstract**: Support Vector Regression (SVR) and its variants are widely used to handle regression tasks, however, since their solution involves solving an expensive quadratic programming problem, it limits its application, especially when dealing with large datasets. Additionally, SVR uses an epsilon-insensitive loss function which is sensitive to outliers and therefore can adversely affect its performance. We propose Granular Ball Support Vector Regression (GBSVR) to tackle problem of regression by using granular ball concept. These balls are useful in simplifying complex data spaces for machine learning tasks, however, to the best of our knowledge, they have not been sufficiently explored for regression problems. Granular balls group the data points into balls based on their proximity and reduce the computational cost in SVR by replacing the large number of data points with far fewer granular balls. This work also suggests a discretization method for continuous-valued attributes to facilitate the construction of granular balls. The effectiveness of the proposed approach is evaluated on several benchmark datasets and it outperforms existing state-of-the-art approaches 

**Abstract (ZH)**: 粒度球支持向量回归（GBSVR）：基于粒度球概念的回归问题处理 

---
# The Impact of Item-Writing Flaws on Difficulty and Discrimination in Item Response Theory 

**Title (ZH)**: 项目编写缺陷对项目反应理论中难度和区分度的影响 

**Authors**: Robin Schmucker, Steven Moore  

**Link**: [PDF](https://arxiv.org/pdf/2503.10533)  

**Abstract**: High-quality test items are essential for educational assessments, particularly within Item Response Theory (IRT). Traditional validation methods rely on resource-intensive pilot testing to estimate item difficulty and discrimination. More recently, Item-Writing Flaw (IWF) rubrics emerged as a domain-general approach for evaluating test items based on textual features. However, their relationship to IRT parameters remains underexplored. To address this gap, we conducted a study involving over 7,000 multiple-choice questions across various STEM subjects (e.g., math and biology). Using an automated approach, we annotated each question with a 19-criteria IWF rubric and studied relationships to data-driven IRT parameters. Our analysis revealed statistically significant links between the number of IWFs and IRT difficulty and discrimination parameters, particularly in life and physical science domains. We further observed how specific IWF criteria can impact item quality more and less severely (e.g., negative wording vs. implausible distractors). Overall, while IWFs are useful for predicting IRT parameters--particularly for screening low-difficulty MCQs--they cannot replace traditional data-driven validation methods. Our findings highlight the need for further research on domain-general evaluation rubrics and algorithms that understand domain-specific content for robust item validation. 

**Abstract (ZH)**: 高质量的测试项目对于教育评估至关重要，特别是在项目反应理论（IRT）领域。传统的验证方法依赖于耗时的试点测试以估算项目的难度和区分度。近年来，项目写作缺陷（IWF）量表作为基于文本特征的通用评估方法得到了发展。然而，它们与IRT参数之间的关系尚待进一步探索。为了填补这一空白，我们对跨多个STEM科目（如数学和生物学）的超过7,000个选择题进行了研究。通过自动化方法，我们为每个问题标注了一个包含19项标准的IWF量表，并研究了其与数据驱动的IRT参数之间的关系。我们的分析揭示了项目写作缺陷的数量与IRT难度和区分度参数之间存在统计显著的相关性，尤其是在生命科学和物理科学领域。我们进一步观察了特定的IWF标准如何不同程度地影响项目质量（例如，负面措辞与不合理干扰项）。总之，虽然IWFs对于预测IRT参数——尤其是筛查难度较低的选择题——非常有用，但它们不能替代传统的数据驱动验证方法。我们的研究结果强调了进一步研究通用评估量表和理解特定领域内容的算法的重要性，以进行稳健的项目验证。 

---
# CountPath: Automating Fragment Counting in Digital Pathology 

**Title (ZH)**: CountPath: 数字病理学中的片段计数自动化 

**Authors**: Ana Beatriz Vieira, Maria Valente, Diana Montezuma, Tomé Albuquerque, Liliana Ribeiro, Domingos Oliveira, João Monteiro, Sofia Gonçalves, Isabel M. Pinto, Jaime S. Cardoso, Arlindo L. Oliveira  

**Link**: [PDF](https://arxiv.org/pdf/2503.10520)  

**Abstract**: Quality control of medical images is a critical component of digital pathology, ensuring that diagnostic images meet required standards. A pre-analytical task within this process is the verification of the number of specimen fragments, a process that ensures that the number of fragments on a slide matches the number documented in the macroscopic report. This step is important to ensure that the slides contain the appropriate diagnostic material from the grossing process, thereby guaranteeing the accuracy of subsequent microscopic examination and diagnosis. Traditionally, this assessment is performed manually, requiring significant time and effort while being subject to significant variability due to its subjective nature. To address these challenges, this study explores an automated approach to fragment counting using the YOLOv9 and Vision Transformer models. Our results demonstrate that the automated system achieves a level of performance comparable to expert assessments, offering a reliable and efficient alternative to manual counting. Additionally, we present findings on interobserver variability, showing that the automated approach achieves an accuracy of 86%, which falls within the range of variation observed among experts (82-88%), further supporting its potential for integration into routine pathology workflows. 

**Abstract (ZH)**: 医疗图像质量控制是数字病理学的关键组成部分，确保诊断图像达到所需标准。这一过程中的一项预分析任务是标本片段数量的验证，该过程确保载玻片上的片段数量与宏报告中记录的数量相符。这一步骤对于确保载玻片包含来自大体处理的适当诊断材料，从而保证后续显微检查和诊断的准确至关重要。传统上，这种评估是通过人工完成的，耗时且主观性高，容易产生显著的变异。为解决这些挑战，本研究探索了使用YOLOv9和Vision Transformer模型的自动化片段计数方法。研究结果表明，自动化系统在性能上与专家评估相当，提供了一种可靠且高效的替代手工计数的方法。此外，我们还展示了不同观察者间变异性的研究结果，表明自动化方法的准确率为86%，这一数值在专家间观察到的变异范围（82%-88%）之内，进一步支持了其在常规病理工作流程中整合的潜力。 

---
# Why the Brain Cannot Be a Digital Computer: History-Dependence and the Computational Limits of Consciousness 

**Title (ZH)**: 为什么大脑不能是数字计算机：基于历史的依赖性和意识的计算限制 

**Authors**: Andrew Knight  

**Link**: [PDF](https://arxiv.org/pdf/2503.10518)  

**Abstract**: This paper presents a novel information-theoretic proof demonstrating that the human brain as currently understood cannot function as a classical digital computer. Through systematic quantification of distinguishable conscious states and their historical dependencies, we establish that the minimum information required to specify a conscious state exceeds the physical information capacity of the human brain by a significant factor. Our analysis calculates the bit-length requirements for representing consciously distinguishable sensory "stimulus frames" and demonstrates that consciousness exhibits mandatory temporal-historical dependencies that multiply these requirements beyond the brain's storage capabilities. This mathematical approach offers new insights into the fundamental limitations of computational models of consciousness and suggests that non-classical information processing mechanisms may be necessary to account for conscious experience. 

**Abstract (ZH)**: 本研究提出了一种新的信息论证明，表明根据当前的理解，人类大脑无法作为经典数字计算机 functioning，通过系统量化可区分的意识状态及其历史依赖性，我们确立了指定一个意识状态所需的最小信息量超出了人类大脑的物理信息容量一个显著的因子。我们的分析计算了表示可区分的感官“刺激帧”的位长要求，并证明了意识表现出强制性的时序历史依赖性，这些依赖性将这些要求乘倍于大脑的存储能力。这种数学方法为计算模型的意识基本限制提供了新的见解，并建议可能需要非经典信息处理机制来解释意识经验。 

---
# Conformal Prediction Sets for Deep Generative Models via Reduction to Conformal Regression 

**Title (ZH)**: 深生成模型中的 conformal 预测集通过转化为 conformal 回归。 

**Authors**: Hooman Shahrokhi, Devjeet Raj Roy, Yan Yan, Venera Arnaoudova, Janaradhan Rao Doppa  

**Link**: [PDF](https://arxiv.org/pdf/2503.10512)  

**Abstract**: We consider the problem of generating valid and small prediction sets by sampling outputs (e.g., software code and natural language text) from a black-box deep generative model for a given input (e.g., textual prompt). The validity of a prediction set is determined by a user-defined binary admissibility function depending on the target application. For example, requiring at least one program in the set to pass all test cases in code generation application. To address this problem, we develop a simple and effective conformal inference algorithm referred to as Generative Prediction Sets (GPS). Given a set of calibration examples and black-box access to a deep generative model, GPS can generate prediction sets with provable guarantees. The key insight behind GPS is to exploit the inherent structure within the distribution over the minimum number of samples needed to obtain an admissible output to develop a simple conformal regression approach over the minimum number of samples. Experiments on multiple datasets for code and math word problems using different large language models demonstrate the efficacy of GPS over state-of-the-art methods. 

**Abstract (ZH)**: 我们考虑通过从给定输入（例如文本提示）的黑盒深度生成模型中采样输出（例如软件代码和自然语言文本）来生成有效且小型的预测集的问题。预测集的有效性由用户定义的二元可接受性函数确定，该函数依赖于目标应用。例如，在代码生成应用中，要求集合中至少有一个程序能够通过所有测试案例。为了解决这一问题，我们开发了一个简单而有效的符合性推断算法，称之为生成预测集（GPS）。给定校准示例集和对深度生成模型的黑盒访问，GPS可以生成具有证明保证的预测集。GPS的核心洞察是利用分布中固有的结构，该结构描述了生成可接受输出所需的最小样本数量，从而开发一种简单的基于最小样本数量的符合性回归方法。使用不同大型语言模型在代码和数学词问题多个数据集上的实验表明，GPS方法优于最新方法的效能。 

---
# Explainable Bayesian deep learning through input-skip Latent Binary Bayesian Neural Networks 

**Title (ZH)**: 通过输入跳过潜在二元贝叶斯神经网络的可解释贝叶斯深度学习 

**Authors**: Eirik Høyheim, Lars Skaaret-Lund, Solve Sæbø, Aliaksandr Hubin  

**Link**: [PDF](https://arxiv.org/pdf/2503.10496)  

**Abstract**: Modeling natural phenomena with artificial neural networks (ANNs) often provides highly accurate predictions. However, ANNs often suffer from over-parameterization, complicating interpretation and raising uncertainty issues. Bayesian neural networks (BNNs) address the latter by representing weights as probability distributions, allowing for predictive uncertainty evaluation. Latent binary Bayesian neural networks (LBBNNs) further handle structural uncertainty and sparsify models by removing redundant weights. This article advances LBBNNs by enabling covariates to skip to any succeeding layer or be excluded, simplifying networks and clarifying input impacts on predictions. Ultimately, a linear model or even a constant can be found to be optimal for a specific problem at hand. Furthermore, the input-skip LBBNN approach reduces network density significantly compared to standard LBBNNs, achieving over 99% reduction for small networks and over 99.9% for larger ones, while still maintaining high predictive accuracy and uncertainty measurement. For example, on MNIST, we reached 97% accuracy and great calibration with just 935 weights, reaching state-of-the-art for compression of neural networks. Furthermore, the proposed method accurately identifies the true covariates and adjusts for system non-linearity. The main contribution is the introduction of active paths, enhancing directly designed global and local explanations within the LBBNN framework, that have theoretical guarantees and do not require post hoc external tools for explanations. 

**Abstract (ZH)**: 用人工神经网络（ANNs）建模自然现象通常能提供高度准确的预测。然而，ANNs往往受到过度参数化的困扰，这 complicating 了解释并引发了不确定性问题。贝叶斯神经网络（BNNs）通过将权重表示为概率分布来解决后者的问题，从而允许对预测不确定性进行评估。潜在二进制贝叶斯神经网络（LBBNNs）进一步处理结构不确定性，并通过移除冗余权重来稀疏化模型。本文通过使协变量能够跳过任何后续层或被排除，推动了LBBNNs的发展，简化了网络并阐明了输入对预测的影响。最终，可以发现对于特定问题来说，线性模型甚至常数可能是最优的。此外，输入跳过的LBBNN方法与标准LBBNN相比，显著减少了网络密度，对于小型网络减少了超过99%，对于大型网络减少了超过99.9%，同时仍保持了高预测准确性和不确定性测量。例如，在MNIST数据集上，我们仅使用935个权重就达到了97%的准确率和良好的校准，这在神经网络压缩技术中达到了最先进的水平。此外，所提出的方法准确地识别了真实的协变量，并调整了系统非线性。主要贡献是引入了活动路径，在LBBNN框架内直接设计全局和局部解释，具有理论保证，并且不需要事后外部工具来进行解释。 

---
# Siamese Foundation Models for Crystal Structure Prediction 

**Title (ZH)**: 晶体结构预测中的孪生基础模型 

**Authors**: Liming Wu, Wenbing Huang, Rui Jiao, Jianxing Huang, Liwei Liu, Yipeng Zhou, Hao Sun, Yang Liu, Fuchun Sun, Yuxiang Ren, Jirong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2503.10471)  

**Abstract**: Crystal Structure Prediction (CSP), which aims to generate stable crystal structures from compositions, represents a critical pathway for discovering novel materials. While structure prediction tasks in other domains, such as proteins, have seen remarkable progress, CSP remains a relatively underexplored area due to the more complex geometries inherent in crystal structures. In this paper, we propose Siamese foundation models specifically designed to address CSP. Our pretrain-finetune framework, named DAO, comprises two complementary foundation models: DAO-G for structure generation and DAO-P for energy prediction. Experiments on CSP benchmarks (MP-20 and MPTS-52) demonstrate that our DAO-G significantly surpasses state-of-the-art (SOTA) methods across all metrics. Extensive ablation studies further confirm that DAO-G excels in generating diverse polymorphic structures, and the dataset relaxation and energy guidance provided by DAO-P are essential for enhancing DAO-G's performance. When applied to three real-world superconductors ($\text{CsV}_3\text{Sb}_5$, $ \text{Zr}_{16}\text{Rh}_8\text{O}_4$ and $\text{Zr}_{16}\text{Pd}_8\text{O}_4$) that are known to be challenging to analyze, our foundation models achieve accurate critical temperature predictions and structure generations. For instance, on $\text{CsV}_3\text{Sb}_5$, DAO-G generates a structure close to the experimental one with an RMSE of 0.0085; DAO-P predicts the $T_c$ value with high accuracy (2.26 K vs. the ground-truth value of 2.30 K). In contrast, conventional DFT calculators like Quantum Espresso only successfully derive the structure of the first superconductor within an acceptable time, while the RMSE is nearly 8 times larger, and the computation speed is more than 1000 times slower. These compelling results collectively highlight the potential of our approach for advancing materials science research and development. 

**Abstract (ZH)**: 晶体结构预测（CSP）：从组成生成稳定晶体结构的关键路径及其在新型材料发现中的应用 

---
# Whisper Speaker Identification: Leveraging Pre-Trained Multilingual Transformers for Robust Speaker Embeddings 

**Title (ZH)**: Whisper说话人识别：利用预训练多语言变换器获得稳健的说话人嵌入 

**Authors**: Jakaria Islam Emon, Md Abu Salek, Kazi Tamanna Alam  

**Link**: [PDF](https://arxiv.org/pdf/2503.10446)  

**Abstract**: Speaker identification in multilingual settings presents unique challenges, particularly when conventional models are predominantly trained on English data. In this paper, we propose WSI (Whisper Speaker Identification), a framework that repurposes the encoder of the Whisper automatic speech recognition model pre trained on extensive multilingual data to generate robust speaker embeddings via a joint loss optimization strategy that leverages online hard triplet mining and self supervised Normalized Temperature-scaled Cross Entropy loss. By capitalizing on Whisper language-agnostic acoustic representations, our approach effectively distinguishes speakers across diverse languages and recording conditions. Extensive evaluations on multiple corpora, including VoxTube (multilingual), JVS (Japanese), CallHome (German, Spanish, Chinese, and Japanese), and Voxconverse (English), demonstrate that WSI consistently outperforms state-of-the-art baselines, namely Pyannote Embedding, ECAPA TDNN, and Xvector, in terms of lower equal error rates and higher AUC scores. These results validate our hypothesis that a multilingual pre-trained ASR encoder, combined with joint loss optimization, substantially improves speaker identification performance in non-English languages. 

**Abstract (ZH)**: 多语言环境下的说话人识别面临独特挑战，特别是在传统模型主要基于英语数据训练的情况下。本文提出了一种WSI（Whisper说话人识别）框架，该框架利用广泛多语言数据预训练的Whisper自动语音识别模型的编码器生成鲁棒的说话人嵌入，并通过结合在线硬三元组挖掘和自监督Normalized Temperature-scaled Cross Entropy损失的联合损失优化策略实现。借助Whisper语言无关的声学表示，我们的方法能够有效区分不同语言和录音条件下的说话人。在包括VoxTube（多语言）、JVS（日语）、CallHome（德语、西班牙语、汉语和日语）和Voxconverse（英语）等多个语料库的广泛评估中，WSI在等错误率和AUC分数方面均优于当前最先进的基线方法（如Pyannote Embedding、ECAPA TDNN和Xvector），验证了我们关于多语言预训练ASR编码器与联合损失优化相结合在非英语语言中显著提高说话人识别性能的假设。 

---
# dFLMoE: Decentralized Federated Learning via Mixture of Experts for Medical Data Analysis 

**Title (ZH)**: dFLMoE: 基于专家混合的去中心化联邦学习在医疗数据分析中的应用 

**Authors**: Luyuan Xie, Tianyu Luan, Wenyuan Cai, Guochen Yan, Zhaoyu Chen, Nan Xi, Yuejian Fang, Qingni Shen, Zhonghai Wu, Junsong Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2503.10412)  

**Abstract**: Federated learning has wide applications in the medical field. It enables knowledge sharing among different healthcare institutes while protecting patients' privacy. However, existing federated learning systems are typically centralized, requiring clients to upload client-specific knowledge to a central server for aggregation. This centralized approach would integrate the knowledge from each client into a centralized server, and the knowledge would be already undermined during the centralized integration before it reaches back to each client. Besides, the centralized approach also creates a dependency on the central server, which may affect training stability if the server malfunctions or connections are unstable. To address these issues, we propose a decentralized federated learning framework named dFLMoE. In our framework, clients directly exchange lightweight head models with each other. After exchanging, each client treats both local and received head models as individual experts, and utilizes a client-specific Mixture of Experts (MoE) approach to make collective decisions. This design not only reduces the knowledge damage with client-specific aggregations but also removes the dependency on the central server to enhance the robustness of the framework. We validate our framework on multiple medical tasks, demonstrating that our method evidently outperforms state-of-the-art approaches under both model homogeneity and heterogeneity settings. 

**Abstract (ZH)**: 联邦学习在医疗领域有广泛的应用。它使得不同的医疗机构能够在保护患者隐私的前提下进行知识共享。然而，现有的联邦学习系统通常采用中心化的方式，要求客户端将客户特定的知识上传到中央服务器进行聚合。这种中心化的方法会在中央服务器整合知识时就已经损害了知识的完整性，在返回给每个客户端之前进一步削弱了知识。此外，中心化方法还会对中央服务器产生依赖，如果服务器出现故障或连接不稳定，会影响训练的稳定性。为了解决这些问题，我们提出了一种去中心化的联邦学习框架，名为dFLMoE。在该框架中，客户端直接互相交换轻量级头部模型。之后，每个客户端将本地和接收到的头部模型视为独立专家，并利用客户端特定的专家混合（MoE）方法进行集体决策。这种设计不仅通过客户端特定的聚合减少了知识损害，还消除了对中央服务器的依赖，增强了框架的鲁棒性。我们在多个医疗任务上验证了该框架，结果表明，在模型同质性和异质性设置下，我们的方法明显优于现有最先进的方法。 

---
# Enhance Exploration in Safe Reinforcement Learning with Contrastive Representation Learning 

**Title (ZH)**: 通过对比表示学习增强安全强化学习中的探索 

**Authors**: Duc Kien Doan, Bang Giang Le, Viet Cuong Ta  

**Link**: [PDF](https://arxiv.org/pdf/2503.10318)  

**Abstract**: In safe reinforcement learning, agent needs to balance between exploration actions and safety constraints. Following this paradigm, domain transfer approaches learn a prior Q-function from the related environments to prevent unsafe actions. However, because of the large number of false positives, some safe actions are never executed, leading to inadequate exploration in sparse-reward environments. In this work, we aim to learn an efficient state representation to balance the exploration and safety-prefer action in a sparse-reward environment. Firstly, the image input is mapped to latent representation by an auto-encoder. A further contrastive learning objective is employed to distinguish safe and unsafe states. In the learning phase, the latent distance is used to construct an additional safety check, which allows the agent to bias the exploration if it visits an unsafe state. To verify the effectiveness of our method, the experiment is carried out in three navigation-based MiniGrid environments. The result highlights that our method can explore the environment better while maintaining a good balance between safety and efficiency. 

**Abstract (ZH)**: 在安全强化学习中，代理需要在探索动作和安全性约束之间寻求平衡。遵循这一范式，领域迁移方法从相关环境中学习先验Q函数，以防止不安全动作。然而，由于大量误报警，一些安全动作从未被执行，导致在稀疏奖励环境中探索不足。在此工作中，我们旨在学习一种高效的状态表示，以在稀疏奖励环境中平衡探索和安全优先动作。首先，图像输入通过自动编码器映射到潜在表示。进一步采用对比学习目标以区分安全和不安全状态。在学习阶段，使用潜在距离构建额外的安全检查，允许代理在访问不安全状态时偏向探索。为验证方法的有效性，实验在三個基于导航的MiniGrid环境中进行。结果显示，我们的方法可以在保持安全和效率良好平衡的同时更好地探索环境。 

---
# Nash Equilibrium Constrained Auto-bidding With Bi-level Reinforcement Learning 

**Title (ZH)**: Nash均衡约束下的双层强化学习自出价策略 

**Authors**: Zhiyu Mou, Miao Xu, Rongquan Bai, Zhuoran Yang, Chuan Yu, Jian Xu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.10304)  

**Abstract**: Many online advertising platforms provide advertisers with auto-bidding services to enhance their advertising performance. However, most existing auto-bidding algorithms fail to accurately capture the auto-bidding problem formulation that the platform truly faces, let alone solve it. Actually, we argue that the platform should try to help optimize each advertiser's performance to the greatest extent -- which makes $\epsilon$-Nash Equilibrium ($\epsilon$-NE) a necessary solution concept -- while maximizing the social welfare of all the advertisers for the platform's long-term value. Based on this, we introduce the \emph{Nash-Equilibrium Constrained Bidding} (NCB), a new formulation of the auto-bidding problem from the platform's perspective. Specifically, it aims to maximize the social welfare of all advertisers under the $\epsilon$-NE constraint. However, the NCB problem presents significant challenges due to its constrained bi-level structure and the typically large number of advertisers involved. To address these challenges, we propose a \emph{Bi-level Policy Gradient} (BPG) framework with theoretical guarantees. Notably, its computational complexity is independent of the number of advertisers, and the associated gradients are straightforward to compute. Extensive simulated and real-world experiments validate the effectiveness of the BPG framework. 

**Abstract (ZH)**: Many Online广告平台提供的自动出价服务旨在提升广告效果。然而，现有的大多数自动出价算法未能准确捕捉到平台实际面临的问题形式，更不用说解决它了。实际上，我们认为平台应该尽可能帮助优化每个广告商的表现——这使得ε纳什均衡（ε-NE）成为必要的解决方案概念——同时最大化所有广告商的社会福利，以提升平台的长期价值。基于此，我们介绍了一种从平台视角出发的新形式的自动出价问题——纳什均衡约束出价（Nash-Equilibrium Constrained Bidding, NCB），其目标是在ε-NE约束下最大化所有广告商的社会福利。然而，由于NCB问题具有约束的双层结构，并且涉及的广告商数量通常较多，因此提出了一个具有理论保证的双层策略梯度（Bi-level Policy Gradient, BPG）框架。值得注意的是，其计算复杂度与广告商数量无关，相关的梯度也易于计算。大量的仿真和实际实验验证了BPG框架的有效性。 

---
# Bilingual Dual-Head Deep Model for Parkinson's Disease Detection from Speech 

**Title (ZH)**: 针对语音的双头双语深度模型在帕金森病检测中的应用 

**Authors**: Moreno La Quatra, Juan Rafael Orozco-Arroyave, Marco Sabato Siniscalchi  

**Link**: [PDF](https://arxiv.org/pdf/2503.10301)  

**Abstract**: This work aims to tackle the Parkinson's disease (PD) detection problem from the speech signal in a bilingual setting by proposing an ad-hoc dual-head deep neural architecture for type-based binary classification. One head is specialized for diadochokinetic patterns. The other head looks for natural speech patterns present in continuous spoken utterances. Only one of the two heads is operative accordingly to the nature of the input. Speech representations are extracted from self-supervised learning (SSL) models and wavelet transforms. Adaptive layers, convolutional bottlenecks, and contrastive learning are exploited to reduce variations across languages. Our solution is assessed against two distinct datasets, EWA-DB, and PC-GITA, which cover Slovak and Spanish languages, respectively. Results indicate that conventional models trained on a single language dataset struggle with cross-linguistic generalization, and naive combinations of datasets are suboptimal. In contrast, our model improves generalization on both languages, simultaneously. 

**Abstract (ZH)**: 本研究旨在通过提出一种特定于类型的小双头深度神经架构，解决双语环境中从语音信号检测帕金森病（PD）的问题。一个分支专门识别迭言运动模式，另一个分支寻找连续口头表达中存在的自然语言模式。根据输入的性质，仅有一个分支处于激活状态。语音表示通过半监督学习模型和小波变换提取。利用自适应层、卷积瓶颈和对比学习减少语言间的变异。我们的解决方案在EWA-DB（斯洛伐克语）和PC-GITA（西班牙语）两个不同数据集上进行评估。结果表明，单语言数据集训练的模型在跨语言泛化上存在困难，简单地组合数据集也效果不佳。相比之下，我们的模型能够同时在两种语言上提高泛化能力。 

---
# PyGDA: A Python Library for Graph Domain Adaptation 

**Title (ZH)**: PyGDA：一种用于图域适应的Python库 

**Authors**: Zhen Zhang, Meihan Liu, Bingsheng He  

**Link**: [PDF](https://arxiv.org/pdf/2503.10284)  

**Abstract**: Graph domain adaptation has emerged as a promising approach to facilitate knowledge transfer across different domains. Recently, numerous models have been proposed to enhance their generalization capabilities in this field. However, there is still no unified library that brings together existing techniques and simplifies their implementation. To fill this gap, we introduce PyGDA, an open-source Python library tailored for graph domain adaptation. As the first comprehensive library in this area, PyGDA covers more than 20 widely used graph domain adaptation methods together with different types of graph datasets. Specifically, PyGDA offers modular components, enabling users to seamlessly build custom models with a variety of commonly used utility functions. To handle large-scale graphs, PyGDA includes support for features such as sampling and mini-batch processing, ensuring efficient computation. In addition, PyGDA also includes comprehensive performance benchmarks and well-documented user-friendly API for both researchers and practitioners. To foster convenient accessibility, PyGDA is released under the MIT license at this https URL, and the API documentation is this https URL. 

**Abstract (ZH)**: Graph域适应已经成为一种促进不同领域知识迁移的有前途的方法。最近，提出了许多模型以增强其在该领域的泛化能力。然而，目前还没有一个统一的库将现有技术整合在一起并简化其实现。为填补这一空白，我们介绍了PyGDA，一个专为Graph域适应开发的开源Python库。作为该领域的首个综合库，PyGDA涵盖了超过20种广泛使用的方法以及不同类型的Graph数据集。具体而言，PyGDA提供模块化组件，使用户能够无缝构建自定义模型，并使用各种常用工具函数。为了处理大规模Graph，PyGDA支持采样和小批量处理等功能，确保高效计算。此外，PyGDA还包含全面的性能基准测试和文档详尽的用户友好型API，适用于研究人员和实践者。为便于访问，PyGDA在<https://>下以MIT许可证发布，并提供了API文档<https://>。 

---
# PIMRL: Physics-Informed Multi-Scale Recurrent Learning for Spatiotemporal Prediction 

**Title (ZH)**: PIMRL：物理启发的多尺度递归学习在时空预测中的应用 

**Authors**: Han Wan, Qi Wang, Hao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.10253)  

**Abstract**: Simulation of spatiotemporal systems governed by partial differential equations is widely applied in fields such as biology, chemistry, aerospace dynamics, and meteorology. Traditional numerical methods incur high computational costs due to the requirement of small time steps for accurate predictions. While machine learning has reduced these costs, long-term predictions remain challenged by error accumulation, particularly in scenarios with insufficient data or varying time scales, where stability and accuracy are compromised. Existing methods often neglect the effective utilization of multi-scale data, leading to suboptimal robustness in predictions. To address these issues, we propose a novel multi-scale learning framework, namely, the Physics-Informed Multi-Scale Recurrent Learning (PIMRL), to effectively leverage multi-scale data for spatiotemporal dynamics prediction. The PIMRL framework comprises two modules: the micro-scale module embeds physical knowledge into neural networks via pretraining, and the macro-scale module adopts a data-driven approach to learn the temporal evolution of physics in the latent space. Experimental results demonstrate that the PIMRL framework consistently achieves state-of-the-art performance across five benchmark datasets ranging from one to three dimensions, showing average improvements of over 9\% in both RMSE and MAE evaluation metrics, with maximum enhancements reaching up to 80%. 

**Abstract (ZH)**: 基于偏微分方程支配的空间时间系统的仿真广泛应用于生物学、化学、航空航天动力学和气象学等领域。传统的数值方法由于需要较小的时间步长以获得准确预测而产生高额的计算成本。尽管机器学习降低了这些成本，但在数据不足或时间尺度变化的情况下，长期预测仍然面临误差累积的挑战，导致稳定性和准确性受损。现有方法往往忽视了多尺度数据的有效利用，从而导致预测的鲁棒性不足。为了解决这些问题，我们提出了一种新的多尺度学习框架，即物理知情多尺度递归学习（PIMRL），以有效利用多尺度数据进行空间时间动力学预测。PIMRL框架包括两个模块：微观尺度模块通过预训练将物理知识嵌入神经网络中，宏观尺度模块采用数据驱动的方法在潜在空间学习物理的时空演变。实验结果表明，PIMRL框架在从一维到三维的五个基准数据集中均取得了最先进的性能，展示了平均9%以上的RMSE和MAE评估指标改进，最大提升达到80%。 

---
# Deep Learning for Time Series Forecasting: A Survey 

**Title (ZH)**: 深度学习在时间序列预测中的应用：一种综述 

**Authors**: Xiangjie Kong, Zhenghao Chen, Weiyao Liu, Kaili Ning, Lechao Zhang, Syauqie Muhammad Marier, Yichen Liu, Yuhao Chen, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2503.10198)  

**Abstract**: Time series forecasting (TSF) has long been a crucial task in both industry and daily life. Most classical statistical models may have certain limitations when applied to practical scenarios in fields such as energy, healthcare, traffic, meteorology, and economics, especially when high accuracy is required. With the continuous development of deep learning, numerous new models have emerged in the field of time series forecasting in recent years. However, existing surveys have not provided a unified summary of the wide range of model architectures in this field, nor have they given detailed summaries of works in feature extraction and datasets. To address this gap, in this review, we comprehensively study the previous works and summarize the general paradigms of Deep Time Series Forecasting (DTSF) in terms of model architectures. Besides, we take an innovative approach by focusing on the composition of time series and systematically explain important feature extraction methods. Additionally, we provide an overall compilation of datasets from various domains in existing works. Finally, we systematically emphasize the significant challenges faced and future research directions in this field. 

**Abstract (ZH)**: 时间序列预测（TSF）一直是工业和日常生活中的一项关键任务。大多数经典统计模型在能源、医疗、交通、气象和经济学等领域实际应用场景中可能存在一定的局限性，特别是在高精度要求的情况下。随着深度学习的不断进步，近年来在时间序列预测领域涌现出了众多新的模型。然而，现有的综述尚未对这一领域的广泛模型架构提供统一的总结，也没有对特征提取和数据集进行详细的综述。为弥补这一空白，在本文综述中，我们全面研究了先前的研究工作，从模型架构的角度总结了深度时间序列预测（DTSF）的一般范式。此外，我们以创新的方式关注时间序列的组成，并系统地解释了重要的特征提取方法。同时，我们提供了现有研究中不同领域数据集的总体编目。最后，我们系统地强调了该领域面临的重大挑战和未来研究方向。 

---
# Predicting Chemical Reaction Outcomes Based on Electron Movements Using Machine Learning 

**Title (ZH)**: 基于电子移动预测化学反应结果的机器学习方法 

**Authors**: Shuan Chen, Kye Sung Park, Taewan Kim, Sunkyu Han, Yousung Jung  

**Link**: [PDF](https://arxiv.org/pdf/2503.10197)  

**Abstract**: Accurately predicting chemical reaction outcomes and potential byproducts is a fundamental task of modern chemistry, enabling the efficient design of synthetic pathways and driving progress in chemical science. Reaction mechanism, which tracks electron movements during chemical reactions, is critical for understanding reaction kinetics and identifying unexpected products. Here, we present Reactron, the first electron-based machine learning model for general reaction prediction. Reactron integrates electron movement into its predictions, generating detailed arrow-pushing diagrams that elucidate each mechanistic step leading to product formation. We demonstrate the high predictive performance of Reactron over existing product-only models by a large-scale reaction outcome prediction benchmark, and the adaptability of the model to learn new reactivity upon providing a few examples. Furthermore, it explores combinatorial reaction spaces, uncovering novel reactivities beyond its training data. With robust performance in both in- and out-of-distribution predictions, Reactron embodies human-like reasoning in chemistry and opens new frontiers in reaction discovery and synthesis design. 

**Abstract (ZH)**: 准确预测化学反应结果和潜在副产品的任务是现代化学的基础，有助于合成路径的高效设计并推动化学科学的进步。反应机理追踪化学反应中的电子移动，对于理解反应动力学和识别意外产物至关重要。在这里，我们介绍Reactron，这是首个基于电子的机器学习模型，用于通用反应预测。Reactron通过整合电子移动来生成详细的箭头推导图，阐明每一步机制直至产物形成。通过大规模反应结果预测基准测试，我们展示了Reactron相对于现有仅基于产物的模型的高预测性能。同时，该模型在提供少量示例后能够学习新的反应性，并探索组合反应空间，揭示超出训练数据的新反应性。凭借稳健的内在和外推性能，Reactron体现了化学中的人类级推理，并为反应发现和合成设计开辟了新的前沿。 

---
# Multi-Agent Q-Learning Dynamics in Random Networks: Convergence due to Exploration and Sparsity 

**Title (ZH)**: 随机网络中多代理Q学习的动力学：探索与稀疏性导致的收敛性 

**Authors**: Aamal Hussain, Dan Leonte, Francesco Belardinelli, Raphael Huser, Dario Paccagnan  

**Link**: [PDF](https://arxiv.org/pdf/2503.10186)  

**Abstract**: Beyond specific settings, many multi-agent learning algorithms fail to converge to an equilibrium solution, and instead display complex, non-stationary behaviours such as recurrent or chaotic orbits. In fact, recent literature suggests that such complex behaviours are likely to occur when the number of agents increases. In this paper, we study Q-learning dynamics in network polymatrix games where the network structure is drawn from classical random graph models. In particular, we focus on the Erdos-Renyi model, a well-studied model for social networks, and the Stochastic Block model, which generalizes the above by accounting for community structures within the network. In each setting, we establish sufficient conditions under which the agents' joint strategies converge to a unique equilibrium. We investigate how this condition depends on the exploration rates, payoff matrices and, crucially, the sparsity of the network. Finally, we validate our theoretical findings through numerical simulations and demonstrate that convergence can be reliably achieved in many-agent systems, provided network sparsity is controlled. 

**Abstract (ZH)**: 超越特定设置，许多多智能体学习算法未能收敛到均衡解，而是表现出复杂的、非平稳的行为，如反复出现或混沌轨道。实际上，近期文献表明，随着智能体数量的增加，此类复杂行为更可能发生。在本文中，我们研究网络聚合矩阵博弈中Q学习动态，其中网络结构来自经典的随机图模型。特别地，我们关注Erdos-Renyi模型，这是一种广泛研究的社会网络模型，以及考虑网络中社区结构的随机块模型。在每种情况下，我们建立了充分条件，以确保智能体的联合策略收敛到唯一的均衡。我们研究了这一条件如何依赖于探索率、支付矩阵以及关键的网络稀疏性。最后，通过数值模拟验证了我们的理论发现，并展示了在控制网络稀疏性的情况下，多智能体系统中的收敛可以可靠地实现。 

---
# Multiplicative Learning 

**Title (ZH)**: 乘法学习 

**Authors**: Han Kim, Hyungjoon Soh, Vipul Periwal, Junghyo Jo  

**Link**: [PDF](https://arxiv.org/pdf/2503.10144)  

**Abstract**: Efficient training of artificial neural networks remains a key challenge in deep learning. Backpropagation (BP), the standard learning algorithm, relies on gradient descent and typically requires numerous iterations for convergence. In this study, we introduce Expectation Reflection (ER), a novel learning approach that updates weights multiplicatively based on the ratio of observed to predicted outputs. Unlike traditional methods, ER maintains consistency without requiring ad hoc loss functions or learning rate hyperparameters. We extend ER to multilayer networks and demonstrate its effectiveness in performing image classification tasks. Notably, ER achieves optimal weight updates in a single iteration. Additionally, we reinterpret ER as a modified form of gradient descent incorporating the inverse mapping of target propagation. These findings suggest that ER provides an efficient and scalable alternative for training neural networks. 

**Abstract (ZH)**: 高效的 artificial neural networks 训练仍然是深度学习中的 key 挑战。Backpropagation (BP) 作为标准的学习算法，依赖于梯度下降，并通常需要多次迭代才能收敛。本研究引入了 Expectation Reflection (ER) 这一新颖的学习方法，该方法基于观测输出与预测输出的比例更新权重。与传统方法不同，ER 保持一致性无需使用随意设定的损失函数或学习率超参数。我们将 ER 扩展到多层网络，并展示了其在执行图像分类任务方面的有效性。值得注意的是，ER 在单次迭代中实现了最优权重更新。此外，我们将 ER 重新解释为一种改进的梯度下降形式，结合了目标传播的逆映射。这些发现表明，ER 为训练神经网络提供了高效且可扩展的替代方案。 

---
# Deep Learning Approaches for Anti-Money Laundering on Mobile Transactions: Review, Framework, and Directions 

**Title (ZH)**: 移动交易反洗钱的深度学习方法：综述、框架与方向 

**Authors**: Jiani Fan, Lwin Khin Shar, Ruichen Zhang, Ziyao Liu, Wenzhuo Yang, Dusit Niyato, Bomin Mao, Kwok-Yan Lam  

**Link**: [PDF](https://arxiv.org/pdf/2503.10058)  

**Abstract**: Money laundering is a financial crime that obscures the origin of illicit funds, necessitating the development and enforcement of anti-money laundering (AML) policies by governments and organizations. The proliferation of mobile payment platforms and smart IoT devices has significantly complicated AML investigations. As payment networks become more interconnected, there is an increasing need for efficient real-time detection to process large volumes of transaction data on heterogeneous payment systems by different operators such as digital currencies, cryptocurrencies and account-based payments. Most of these mobile payment networks are supported by connected devices, many of which are considered loT devices in the FinTech space that constantly generate data. Furthermore, the growing complexity and unpredictability of transaction patterns across these networks contribute to a higher incidence of false positives. While machine learning solutions have the potential to enhance detection efficiency, their application in AML faces unique challenges, such as addressing privacy concerns tied to sensitive financial data and managing the real-world constraint of limited data availability due to data regulations. Existing surveys in the AML literature broadly review machine learning approaches for money laundering detection, but they often lack an in-depth exploration of advanced deep learning techniques - an emerging field with significant potential. To address this gap, this paper conducts a comprehensive review of deep learning solutions and the challenges associated with their use in AML. Additionally, we propose a novel framework that applies the least-privilege principle by integrating machine learning techniques, codifying AML red flags, and employing account profiling to provide context for predictions and enable effective fraud detection under limited data availability.... 

**Abstract (ZH)**: 移动支付平台和智能物联网设备 proliferation 对反洗钱调查的复杂化及其深度学习解决方案的研究：最小权限原则下的机器学习技术、AML红旗标志及账户 profiling 应用 

---
# DTA: Dual Temporal-channel-wise Attention for Spiking Neural Networks 

**Title (ZH)**: DTA: 双时序通道注意力机制在脉冲神经网络中的应用 

**Authors**: Minje Kim, Minjun Kim, Xu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10052)  

**Abstract**: Spiking Neural Networks (SNNs) present a more energy-efficient alternative to Artificial Neural Networks (ANNs) by harnessing spatio-temporal dynamics and event-driven spikes. Effective utilization of temporal information is crucial for SNNs, leading to the exploration of attention mechanisms to enhance this capability. Conventional attention operations either apply identical operation or employ non-identical operations across target dimensions. We identify that these approaches provide distinct perspectives on temporal information. To leverage the strengths of both operations, we propose a novel Dual Temporal-channel-wise Attention (DTA) mechanism that integrates both identical/non-identical attention strategies. To the best of our knowledge, this is the first attempt to concentrate on both the correlation and dependency of temporal-channel using both identical and non-identical attention operations. Experimental results demonstrate that the DTA mechanism achieves state-of-the-art performance on both static datasets (CIFAR10, CIFAR100, ImageNet-1k) and dynamic dataset (CIFAR10-DVS), elevating spike representation and capturing complex temporal-channel relationship. We open-source our code: this https URL. 

**Abstract (ZH)**: 基于尖劈神经元的双时序通道注意力机制：同时利用相同和不同注意力操作增强时间通道相关性和依赖性 

---
# Rapid analysis of point-contact Andreev reflection spectra via machine learning with adaptive data augmentation 

**Title (ZH)**: 基于自适应数据扩增的机器学习快速分析点接触安德陇反射谱 

**Authors**: Dongik Lee, Valentin Stanev, Xiaohang Zhang, Mijeong Kang, Ichiro Takeuchi, Seunghun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.10040)  

**Abstract**: Delineating the superconducting order parameters is a pivotal task in investigating superconductivity for probing pairing mechanisms, as well as their symmetry and topology. Point-contact Andreev reflection (PCAR) measurement is a simple yet powerful tool for identifying the order parameters. The PCAR spectra exhibit significant variations depending on the type of the order parameter in a superconductor, including its magnitude ($\mathit{\Delta}$), as well as temperature, interfacial quality, Fermi velocity mismatch, and other factors. The information on the order parameter can be obtained by finding the combination of these parameters, generating a theoretical spectrum that fits a measured experimental spectrum. However, due to the complexity of the spectra and the high dimensionality of parameters, extracting the fitting parameters is often time-consuming and labor-intensive. In this study, we employ a convolutional neural network (CNN) algorithm to create models for rapid and automated analysis of PCAR spectra of various superconductors with different pairing symmetries (conventional $s$-wave, chiral $p_x+ip_y$-wave, and $d_{x^2-y^2}$-wave). The training datasets are generated based on the Blonder-Tinkham-Klapwijk (BTK) theory and further modified and augmented by selectively incorporating noise and peaks according to the bias voltages. This approach not only replicates the experimental spectra but also brings the model's attention to important features within the spectra. The optimized models provide fitting parameters for experimentally measured spectra in less than 100 ms per spectrum. Our approaches and findings pave the way for rapid and automated spectral analysis which will help accelerate research on superconductors with complex order parameters. 

**Abstract (ZH)**: 划分超导有序参数是探索超导电性、探查配对机制及其对称性和拓扑性的关键任务。点接触安德烈夫反射（PCAR）测量是一种简单而强大的工具，用于识别有序参数。PCAR光谱会根据超导体中有序参数的类型以及其幅度（$\mathit{\Delta}$）、温度、界面质量、费米速度不匹配及其他因素表现出显著差异。通过找到这些参数的组合，生成与测量实验光谱匹配的理论光谱，可以获得有序参数的信息。然而，由于光谱的复杂性和参数的高维度性，提取拟合参数往往耗时且劳动密集。在这项研究中，我们采用卷积神经网络（CNN）算法，为具有不同配对对称性（常规$s$波、手征$p_x+ip_y$波和$d_{x^2-y^2}$波）的各种超导体的PCAR光谱创建快速和自动分析模型。训练数据集基于Blonder-Tinkham-Klapwijk（BTK）理论生成，并通过根据偏置电压选择性地引入噪声和峰值进行了进一步修改和扩充。这种方法不仅复现了实验光谱，还使模型将注意力集中在光谱中的重要特征上。优化后的模型可在每个光谱少于100毫秒的时间内提供实验测量光谱的拟合参数。我们的方法和发现为快速和自动光谱分析铺平了道路，这将有助于加速对具有复杂有序参数的超导体的研究。 

---
# Label Unbalance in High-frequency Trading 

**Title (ZH)**: 高频交易中的标签不平衡问题 

**Authors**: Zijian Zhao, Xuming Chen, Jiayu Wen, Mingwen Liu, Xiaoteng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.09988)  

**Abstract**: In financial trading, return prediction is one of the foundation for a successful trading system. By the fast development of the deep learning in various areas such as graphical processing, natural language, it has also demonstrate significant edge in handling with financial data. While the success of the deep learning relies on huge amount of labeled sample, labeling each time/event as profitable or unprofitable, under the transaction cost, especially in the high-frequency trading world, suffers from serious label imbalance this http URL this paper, we adopts rigurious end-to-end deep learning framework with comprehensive label imbalance adjustment methods and succeed in predicting in high-frequency return in the Chinese future market. The code for our method is publicly available at this https URL . 

**Abstract (ZH)**: 在金融交易中，回报预测是成功交易系统的基础。随着深度学习在图形处理、自然语言处理等领域的发展，它在处理金融数据方面也展现出了显著的优势。尽管深度学习的成功依赖于大量的标注样本，但在交易成本较高的情况下，特别是在高频交易领域，每笔交易事件标记为盈利或亏损时存在着严重的标签不平衡问题。本文采用严格的端到端深度学习框架，并结合全面的标签不平衡调整方法，成功预测了中国期货市场的高频回报。我们的方法代码已公开，可访问此链接： эта ссылка 。 

---
# Uncertainty-aware Long-tailed Weights Model the Utility of Pseudo-labels for Semi-supervised Learning 

**Title (ZH)**: 不确定性意识长尾权重模型：伪标签在半监督学习中的效用 

**Authors**: Jiaqi Wu, Junbiao Pang, Qingming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.09974)  

**Abstract**: Current Semi-supervised Learning (SSL) adopts the pseudo-labeling strategy and further filters pseudo-labels based on confidence thresholds. However, this mechanism has notable drawbacks: 1) setting the reasonable threshold is an open problem which significantly influences the selection of the high-quality pseudo-labels; and 2) deep models often exhibit the over-confidence phenomenon which makes the confidence value an unreliable indicator for assessing the quality of pseudo-labels due to the scarcity of labeled data. In this paper, we propose an Uncertainty-aware Ensemble Structure (UES) to assess the utility of pseudo-labels for unlabeled samples. We further model the utility of pseudo-labels as long-tailed weights to avoid the open problem of setting the threshold. Concretely, the advantage of the long-tailed weights ensures that even unreliable pseudo-labels still contribute to enhancing the model's robustness. Besides, UES is lightweight and architecture-agnostic, easily extending to various computer vision tasks, including classification and regression. Experimental results demonstrate that combining the proposed method with DualPose leads to a 3.47% improvement in Percentage of Correct Keypoints (PCK) on the Sniffing dataset with 100 data points (30 labeled), a 7.29\% improvement in PCK on the FLIC dataset with 100 data points (50 labeled), and a 3.91% improvement in PCK on the LSP dataset with 200 data points (100 labeled). Furthermore, when combined with FixMatch, the proposed method achieves a 0.2% accuracy improvement on the CIFAR-10 dataset with 40 labeled data points and a 0.26% accuracy improvement on the CIFAR-100 dataset with 400 labeled data points. 

**Abstract (ZH)**: 基于不确定性感知的伪标签评估结构(Uncertainty-aware Ensemble Structure for Assessing Pseudo-label Utility in Semi-supervised Learning) 

---
# Detecting Dataset Bias in Medical AI: A Generalized and Modality-Agnostic Auditing Framework 

**Title (ZH)**: 在医疗人工智能中检测数据集偏差：一种通用且模态无关的审计框架 

**Authors**: Nathan Drenkow, Mitchell Pavlak, Keith Harrigian, Ayah Zirikly, Adarsh Subbaswamy, Mathias Unberath  

**Link**: [PDF](https://arxiv.org/pdf/2503.09969)  

**Abstract**: Data-driven AI is establishing itself at the center of evidence-based medicine. However, reports of shortcomings and unexpected behavior are growing due to AI's reliance on association-based learning. A major reason for this behavior: latent bias in machine learning datasets can be amplified during training and/or hidden during testing. We present a data modality-agnostic auditing framework for generating targeted hypotheses about sources of bias which we refer to as Generalized Attribute Utility and Detectability-Induced bias Testing (G-AUDIT) for datasets. Our method examines the relationship between task-level annotations and data properties including protected attributes (e.g., race, age, sex) and environment and acquisition characteristics (e.g., clinical site, imaging protocols). G-AUDIT automatically quantifies the extent to which the observed data attributes may enable shortcut learning, or in the case of testing data, hide predictions made based on spurious associations. We demonstrate the broad applicability and value of our method by analyzing large-scale medical datasets for three distinct modalities and learning tasks: skin lesion classification in images, stigmatizing language classification in Electronic Health Records (EHR), and mortality prediction for ICU tabular data. In each setting, G-AUDIT successfully identifies subtle biases commonly overlooked by traditional qualitative methods that focus primarily on social and ethical objectives, underscoring its practical value in exposing dataset-level risks and supporting the downstream development of reliable AI systems. Our method paves the way for achieving deeper understanding of machine learning datasets throughout the AI development life-cycle from initial prototyping all the way to regulation, and creates opportunities to reduce model bias, enabling safer and more trustworthy AI systems. 

**Abstract (ZH)**: 数据驱动的人工智能正成为基于证据的医学的核心。然而，由于AI依赖于基于关联的学习，其不足之处和意外行为的报告正在增加。我们提出了一种数据模态无关的审计框架，用于生成关于偏倚来源的靶向假设，我们称之为广义属性效用和可检测性诱导偏倚测试（G-AUDIT）。该方法检查任务级注释与数据属性之间的关系，包括保护属性（如种族、年龄、性别）和环境与获取特征（如临床站点、成像协议）。G-AUDIT自动量化观察到的数据属性可能使快速学习变得可行的程度，或者在测试数据的情况下，隐藏基于虚假关联的预测。我们通过分析针对三种不同模态和学习任务的大规模医学数据集（皮肤病变分类图像、电子健康记录中的污名化语言分类、重症监护病房表数据的死亡率预测）来展示该方法的广泛适用性和价值。在每个场景中，G-AUDIT成功识别了传统定性方法通常忽视的细微偏倚，这些方法主要关注社会和伦理目标，突显了其在揭示数据集层面风险和支持下游开发可靠AI系统方面的实用价值。该方法为从初步原型设计到监管的整个AI开发生命周期中实现对机器学习数据集的更深入理解铺平了道路，并创造了减少模型偏倚的机会，从而实现更安全、更值得信赖的AI系统。 

---
# Optimizing Fire Safety: Reducing False Alarms Using Advanced Machine Learning Techniques 

**Title (ZH)**: 优化消防安全性：使用高级机器学习技术减少误报 

**Authors**: Muhammad Hassan Jamal, Abdulwahab Alazeb, Shahid Allah Bakhsh, Wadii Boulila, Syed Aziz Shah, Aizaz Ahmad Khattak, Muhammad Shahbaz Khan  

**Link**: [PDF](https://arxiv.org/pdf/2503.09960)  

**Abstract**: Fire safety practices are important to reduce the extent of destruction caused by fire. While smoke alarms help save lives, firefighters struggle with the increasing number of false alarms. This paper presents a precise and efficient Weighted ensemble model for decreasing false alarms. It estimates the density, computes weights according to the high and low-density regions, forwards the high region weights to KNN and low region weights to XGBoost and combines the predictions. The proposed model is effective at reducing response time, increasing fire safety, and minimizing the damage that fires cause. A specifically designed dataset for smoke detection is utilized to test the proposed model. In addition, a variety of ML models, such as Logistic Regression (LR), Decision Tree (DT), Random Forest (RF), Nai:ve Bayes (NB), K-Nearest Neighbour (KNN), Support Vector Machine (SVM), Extreme Gradient Boosting (XGBoost), Adaptive Boosting (ADAB), have also been utilized. To maximize the use of the smoke detection dataset, all the algorithms utilize the SMOTE re-sampling technique. After evaluating the assessment criteria, this paper presents a concise summary of the comprehensive findings obtained by comparing the outcomes of all models. 

**Abstract (ZH)**: 防火安全实践对于减少火灾造成的破坏至关重要。虽然烟雾报警器有助于挽救生命，但消防员却面临着日益增多的误报警情况。本文提出了一种精确且高效的加权集成模型，用于降低误报警率。该模型通过估计密度、根据高密度和低密度区域计算权重、将高区域权重传递给KNN并将低区域权重传递给XGBoost，从而结合预测结果。所提出的模型能够有效缩短响应时间、提高防火安全性和最小化火灾造成的损害。本文利用专门设计的烟雾检测数据集测试所提出的模型。此外，还使用了多种机器学习模型，如逻辑回归（LR）、决策树（DT）、随机森林（RF）、朴素贝叶斯（NB）、K最近邻（KNN）、支持向量机（SVM）、极端梯度提升（XGBoost）、自适应提升（AdaB）。为了充分利用烟雾检测数据集，所有算法均采用SMOTE重采样技术。在评估评估标准后，本文简洁地总结了所有模型比较结果所得出的综合发现。 

---
# Identifying Trustworthiness Challenges in Deep Learning Models for Continental-Scale Water Quality Prediction 

**Title (ZH)**: 识别大陆尺度水质预测深度学习模型的信任度挑战 

**Authors**: Xiaobo Xia, Xiaofeng Liu, Jiale Liu, Kuai Fang, Lu Lu, Samet Oymak, William S. Currie, Tongliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.09947)  

**Abstract**: Water quality is foundational to environmental sustainability, ecosystem resilience, and public health. Deep learning models, particularly Long Short-Term Memory (LSTM) networks, offer transformative potential for large-scale water quality prediction and scientific insights generation. However, their widespread adoption in high-stakes decision-making, such as pollution mitigation and equitable resource allocation, is prevented by unresolved trustworthiness challenges including fairness, uncertainty, interpretability, robustness, generalizability, and reproducibility. In this work, we present the first comprehensive evaluation of trustworthiness in a continental-scale multi-task LSTM model predicting 20 water quality variables (encompassing physical/chemical processes, geochemical weathering, and nutrient cycling) across 482 U.S. basins. Our investigation uncovers systematic patterns of model performance disparities linked to basin characteristics, the inherent complexity of biogeochemical processes, and variable predictability, emphasizing critical performance fairness concerns. We further propose methodological frameworks for quantitatively evaluating critical aspects of trustworthiness, including uncertainty, interpretability, and robustness, identifying key limitations that could challenge reliable real-world deployment. This work serves as a timely call to action for advancing trustworthy data-driven methods for water resources management and provides a pathway to offering critical insights for researchers, decision-makers, and practitioners seeking to leverage artificial intelligence (AI) responsibly in environmental management. 

**Abstract (ZH)**: 大范围多任务LSTM模型在20个水质量变量（涵盖物理/化学过程、地球化学风化和养分循环）预测中的可信性综合评估：对美国482个流域的系统性研究及其在水资源管理中的应用呼吁 

---
# Developing and Evaluating an AI-Assisted Prediction Model for Unplanned Intensive Care Admissions following Elective Neurosurgery using Natural Language Processing within an Electronic Healthcare Record System 

**Title (ZH)**: 基于电子健康记录系统中的自然语言处理开发并评估人工智能辅助预测模型，用于择期神经外科手术后突发重症监护入院预测 

**Authors**: Julia Ive, Olatomiwa Olukoya, Jonathan P. Funnell, James Booker, Sze H M Lam, Ugan Reddy, Kawsar Noor, Richard JB Dobson, Astri M.V. Luoma, Hani J Marcus  

**Link**: [PDF](https://arxiv.org/pdf/2503.09927)  

**Abstract**: Introduction: Timely care in a specialised neuro-intensive therapy unit (ITU) reduces mortality and hospital stays, with planned admissions being safer than unplanned ones. However, post-operative care decisions remain subjective. This study used artificial intelligence (AI), specifically natural language processing (NLP) to analyse electronic health records (EHRs) and predict ITU admissions for elective surgery patients. Methods: This study analysed the EHRs of elective neurosurgery patients from University College London Hospital (UCLH) using NLP. Patients were categorised into planned high dependency unit (HDU) or ITU admission; unplanned HDU or ITU admission; or ward / overnight recovery (ONR). The Medical Concept Annotation Tool (MedCAT) was used to identify SNOMED-CT concepts within the clinical notes. We then explored the utility of these identified concepts for a range of AI algorithms trained to predict ITU admission. Results: The CogStack-MedCAT NLP model, initially trained on hospital-wide EHRs, underwent two refinements: first with data from patients with Normal Pressure Hydrocephalus (NPH) and then with data from Vestibular Schwannoma (VS) patients, achieving a concept detection F1-score of 0.93. This refined model was then used to extract concepts from EHR notes of 2,268 eligible neurosurgical patients. We integrated the extracted concepts into AI models, including a decision tree model and a neural time-series model. Using the simpler decision tree model, we achieved a recall of 0.87 (CI 0.82 - 0.91) for ITU admissions, reducing the proportion of unplanned ITU cases missed by human experts from 36% to 4%. Conclusion: The NLP model, refined for accuracy, has proven its efficiency in extracting relevant concepts, providing a reliable basis for predictive AI models to use in clinically valid applications. 

**Abstract (ZH)**: 介绍：专门的神经重症治疗单元（ITU）中的及时护理可降低死亡率和缩短住院时间，计划入院比非计划入院更安全。然而，手术后护理决策仍具有主观性。本研究利用人工智能（AI），具体为自然语言处理（NLP）分析电子健康记录（EHRs），预测择期手术患者进入ITU的入院情况。方法：本研究使用NLP分析伦敦大学学院医院（UCLH）择期神经外科患者的EHRs。患者被分为计划高依赖单元（HDU）或ITU住院；非计划HDU或ITU住院；或病房/过夜恢复（ONR）。使用医疗概念标注工具（MedCAT）识别临床笔记中的SNOMED-CT概念。然后探讨了这些识别的概念对于训练预测ITU住院的多种AI算法的实用性。结果：CogStack-MedCAT NLP模型首先在全院EHRs上进行训练，随后分别用正常压力脑积水（NPH）和前庭神经鞘瘤（VS）患者的资料进行了两次优化，概念检测F1分数达到0.93。优化后的模型用于从2,268例合格神经外科患者的EHR笔记中提取概念。将提取的概念集成到包括决策树模型和神经时序模型在内的AI模型中。使用较为简单的决策树模型，ITU住院的召回率为0.87（95% CI 0.82 - 0.91），降低了人类专家遗漏非计划ITU病例的比例，从36%降至4%。结论：经过精确优化的NLP模型证明了其在提取相关概念方面的效率，为基础的预测AI模型在临床有效应用提供了可靠的基础。 

---
# eXpLogic: Explaining Logic Types and Patterns in DiffLogic Networks 

**Title (ZH)**: eXpLogic: 解释DiffLogic网络中的逻辑类型和模式 

**Authors**: Stephen Wormald, David Koblah, Matheus Kunzler Maldaner, Domenic Forte, Damon L. Woodard  

**Link**: [PDF](https://arxiv.org/pdf/2503.09910)  

**Abstract**: Constraining deep neural networks (DNNs) to learn individual logic types per node, as performed using the DiffLogic network architecture, opens the door to model-specific explanation techniques that quell the complexity inherent to DNNs. Inspired by principles of circuit analysis from computer engineering, this work presents an algorithm (eXpLogic) for producing saliency maps which explain input patterns that activate certain functions. The eXpLogic explanations: (1) show the exact set of inputs responsible for a decision, which helps interpret false negative and false positive predictions, (2) highlight common input patterns that activate certain outputs, and (3) help reduce the network size to improve class-specific inference. To evaluate the eXpLogic saliency map, we introduce a metric that quantifies how much an input changes before switching a model's class prediction (the SwitchDist) and use this metric to compare eXpLogic against the Vanilla Gradients (VG) and Integrated Gradient (IG) methods. Generally, we show that eXpLogic saliency maps are better at predicting which inputs will change the class score. These maps help reduce the network size and inference times by 87\% and 8\%, respectively, while having a limited impact (-3.8\%) on class-specific predictions. The broader value of this work to machine learning is in demonstrating how certain DNN architectures promote explainability, which is relevant to healthcare, defense, and law. 

**Abstract (ZH)**: 约束深度神经网络（DNNs）使每个节点学习单一的逻辑类型，如DiffLogic网络架构所做，为特定模型的解释技术打开了大门，这些技术可以缓解DNNs固有的复杂性。受到计算机工程中电路分析原则的启发，本文提出了一种算法（eXpLogic）来生成解释输入模式的显著性图，这些模式激活了某些功能。eXpLogic解释包括：（1）展示了导致决策的精确输入集合，有助于解释假阴性和假阳性预测，（2）突显了激活某些输出的常见输入模式，以及（3）有助于减小网络规模以改进类特定推理。为了评估eXpLogic显著性图，我们引入了一个度量标准，该标准量化了在改变模型类预测前所需输入的改变量（SwitchDist），并使用此度量标准将eXpLogic与其他VG和IG方法进行比较。总体而言，我们表明，eXpLogic显著性图在预测哪些输入会改变类评分方面更为准确。这些图有助于分别减少网络规模和推理时间87%和8%，同时对类特定预测的影响有限（-3.8%）。本文对机器学习的更广泛价值在于，证明了某些DNN架构促进了可解释性，这在医疗保健、国防和法律领域具有相关性。 

---
# AI Rivalry as a Craft: How Resisting and Embracing Generative AI Reshape Writing Professions 

**Title (ZH)**: AI rivalry as a craft:如何抵制与拥抱生成式AI重塑写作职业 

**Authors**: Rama Adithya Varanasi, Batia Mishan Wiesenfeld, Oded Nov  

**Link**: [PDF](https://arxiv.org/pdf/2503.09901)  

**Abstract**: Generative AI (GAI) technologies are disrupting professional writing, challenging traditional practices. Recent studies explore GAI adoption experiences of creative practitioners, but we know little about how these experiences evolve into established practices and how GAI resistance alters these practices. To address this gap, we conducted 25 semi-structured interviews with writing professionals who adopted and/or resisted GAI. Using the theoretical lens of Job Crafting, we identify four strategies professionals employ to reshape their roles. Writing professionals employed GAI resisting strategies to maximize human potential, reinforce professional identity, carve out a professional niche, and preserve credibility within their networks. In contrast, GAI-enabled strategies allowed writers who embraced GAI to enhance desirable workflows, minimize mundane tasks, and engage in new AI-managerial labor. These strategies amplified their collaborations with GAI while reducing their reliance on other people. We conclude by discussing implications of GAI practices on writers' identity and practices as well as crafting theory. 

**Abstract (ZH)**: 生成性人工智能（GAI）技术正在颠覆专业写作，挑战传统实践。当前的研究探索了创造性从业者采用GAI的经历，但我们对这些经历如何演变成既定实践以及GAI抵抗如何改变这些实践知之甚少。为了弥补这一空白，我们对25位采用和/或抵抗GAI的写作专业人士进行了半结构化访谈，运用工作重塑的理论视角，我们识别出专业人士采用的四种策略以重塑其角色。写作专业人士采用GAI抵抗策略以最大化人类潜力、强化专业身份、开拓专业细分领域，并在其网络中维护信誉。相比之下，GAI赋能策略使拥抱GAI的写作者能够优化高效的工作流程、减少乏味任务，并参与新的AI管理劳动。这些策略不仅放大了他们与GAI的合作，还减少了他们对其他人的依赖。我们最后讨论了GAI实践对写作者身份和实践的影响以及工作重塑理论的意义。 

---
# A Rule Based Solution to Co-reference Resolution in Clinical Text 

**Title (ZH)**: 基于规则的方法在临床文本中的同指消解 

**Authors**: Ping Chen, David Hinote, Guoqing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.09896)  

**Abstract**: Objective: The aim of this study was to build an effective co-reference resolution system tailored for the biomedical domain. Materials and Methods: Experiment materials used in this study is provided by the 2011 i2b2 Natural Language Processing Challenge. The 2011 i2b2 challenge involves coreference resolution in medical documents. Concept mentions have been annotated in clinical texts, and the mentions that co-refer in each document are to be linked by coreference chains. Normally, there are two ways of constructing a system to automatically discover co-referent links. One is to manually build rules for co-reference resolution, and the other category of approaches is to use machine learning systems to learn automatically from training datasets and then perform the resolution task on testing datasets. Results: Experiments show the existing co-reference resolution systems are able to find some of the co-referent links, and our rule based system performs well finding the majority of the co-referent links. Our system achieved 89.6% overall performance on multiple medical datasets. Conclusion: The experiment results show that manually crafted rules based on observation of training data is a valid way to accomplish high performance in this coreference resolution task for the critical biomedical domain. 

**Abstract (ZH)**: 研究目的：本文旨在构建一个针对生物医学领域的有效共指消解系统。材料与方法：本研究使用的实验材料来自2011年i2b2自然语言处理挑战。2011年i2b2挑战包括医疗文档中的共指消解任务。概念提及已在临床文本中进行了标注，并需通过共指链将每份文档中相互共指的提及连接起来。通常，自动发现共指链接的系统有两种构建方式：一种是人工构建共指消解规则，另一种是使用机器学习系统从训练数据集自动学习，然后在测试数据集上执行消解任务。结果：实验结果表明现有的共指消解系统能够找到一些共指链接，而我们的基于规则的系统在发现大多数共指链接上表现良好。我们的系统在多个生物医学数据集上的整体性能达到了89.6%。结论：实验结果表明，在关键的生物医学领域完成共指消解任务时，根据训练数据观察人工构建规则是一种有效的高 performance 方法。 

---
# Who Are You Behind the Screen? Implicit MBTI and Gender Detection Using Artificial Intelligence 

**Title (ZH)**: 屏幕背后的真实身份：基于人工智能的隐含MBTI和性别识别 

**Authors**: Kourosh Shahnazari, Seyed Moein Ayyoubzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2503.09853)  

**Abstract**: In personalized technology and psychological research, precisely detecting demographic features and personality traits from digital interactions becomes ever more important. This work investigates implicit categorization, inferring personality and gender variables directly from linguistic patterns in Telegram conversation data, while conventional personality prediction techniques mostly depend on explicitly self-reported labels. We refine a Transformer-based language model (RoBERTa) to capture complex linguistic cues indicative of personality traits and gender differences using a dataset comprising 138,866 messages from 1,602 users annotated with MBTI types and 195,016 messages from 2,598 users annotated with gender. Confidence levels help to greatly raise model accuracy to 86.16\%, hence proving RoBERTa's capacity to consistently identify implicit personality types from conversational text data. Our results highlight the usefulness of Transformer topologies for implicit personality and gender classification, hence stressing their efficiency and stressing important trade-offs between accuracy and coverage in realistic conversational environments. With regard to gender classification, the model obtained an accuracy of 74.4\%, therefore capturing gender-specific language patterns. Personality dimension analysis showed that people with introverted and intuitive preferences are especially more active in text-based interactions. This study emphasizes practical issues in balancing accuracy and data coverage as Transformer-based models show their efficiency in implicit personality and gender prediction tasks from conversational texts. 

**Abstract (ZH)**: 个性化技术与心理研究中，精准检测数字互动中的 demographic 特征和个人特质变得越来越重要。本研究通过分析 Telegram 对话数据中的语言模式，直接推断个性和性别变量，而传统的人格预测技术主要依赖于显式自我报告标签。我们对基于 Transformer 的语言模型（RoBERTa）进行精炼，使用包含 138,866 条消息（1,602 用户，标记有 MBTI 类型）和 195,016 条消息（2,598 用户，标记有性别）的数据集，以捕捉预测个性特质和性别差异的语言线索。置信度水平有助于将模型准确性提高到 86.16%，从而证明 RoBERTa 有能力从对话文本数据中一致性地识别隐含的人格类型。本研究强调了 Transformers 架构在隐含人格和性别分类中的有效性，这突出了在实际对话环境中准确性和覆盖面之间的贸易-offs。在性别分类方面，模型的准确率为 74.4%，因此捕捉到了性别特异性语言模式。人格维度分析表明，具有内倾和直觉倾向的人在文本互动中更为活跃。本研究强调了在平衡准确性与数据覆盖面时的实际问题，Transformer 基础模型在隐含人格和性别预测任务中的效率得到了体现。 

---
# On the Limitations of Vision-Language Models in Understanding Image Transforms 

**Title (ZH)**: 视觉语言模型在理解图像变换方面的局限性 

**Authors**: Ahmad Mustafa Anis, Hasnain Ali, Saquib Sarfraz  

**Link**: [PDF](https://arxiv.org/pdf/2503.09837)  

**Abstract**: Vision Language Models (VLMs) have demonstrated significant potential in various downstream tasks, including Image/Video Generation, Visual Question Answering, Multimodal Chatbots, and Video Understanding. However, these models often struggle with basic image transformations. This paper investigates the image-level understanding of VLMs, specifically CLIP by OpenAI and SigLIP by Google. Our findings reveal that these models lack comprehension of multiple image-level augmentations. To facilitate this study, we created an augmented version of the Flickr8k dataset, pairing each image with a detailed description of the applied transformation. We further explore how this deficiency impacts downstream tasks, particularly in image editing, and evaluate the performance of state-of-the-art Image2Image models on simple transformations. 

**Abstract (ZH)**: 视觉语言模型（VLMs）在图像/视频生成、视觉问答、多模态聊天机器人和视频理解等下游任务中展现了显著潜力。然而，这些模型常在基本的图像变换方面表现不佳。本文研究了VLMs的图像级理解能力，特别是由OpenAI开发的CLIP和由Google开发的SigLIP。我们的研究发现，这些模型对多种图像级增强缺乏理解。为进行这项研究，我们创建了 Flickr8k 数据集的增强版本，为每张图像提供了详细的转换描述。我们进一步探讨了这种不足对下游任务的影响，特别是在图像编辑任务方面，并评估了最先进的Image2Image模型在简单变换上的性能。 

---
# Un-Straightening Generative AI: How Queer Artists Surface and Challenge the Normativity of Generative AI Models 

**Title (ZH)**: 非规范化的生成AI： queer艺术家揭示并挑战生成AI模型的规范性 

**Authors**: Jordan Taylor, Joel Mire, Franchesca Spektor, Alicia DeVrio, Maarten Sap, Haiyi Zhu, Sarah Fox  

**Link**: [PDF](https://arxiv.org/pdf/2503.09805)  

**Abstract**: Queer people are often discussed as targets of bias, harm, or discrimination in research on generative AI. However, the specific ways that queer people engage with generative AI, and thus possible uses that support queer people, have yet to be explored. We conducted a workshop study with 13 queer artists, during which we gave participants access to GPT-4 and DALL-E 3 and facilitated group sensemaking activities. We found our participants struggled to use these models due to various normative values embedded in their designs, such as hyper-positivity and anti-sexuality. We describe various strategies our participants developed to overcome these models' limitations and how, nevertheless, our participants found value in these highly-normative technologies. Drawing on queer feminist theory, we discuss implications for the conceptualization of "state-of-the-art" models and consider how FAccT researchers might support queer alternatives. 

**Abstract (ZH)**: 同性恋人群在生成式AI研究中往往被视为偏见、伤害或歧视的目标。然而，同性恋人群与生成式AI的具体互动方式及其可能支持同性恋人群的应用尚未被探索。我们通过一项包含13名同性恋艺术家的工作坊研究，让参与者接触GPT-4和DALL-E 3，并促进集体意义建构活动。我们发现参与者由于这些模型设计中嵌入的各种规范性价值观（如过度积极和反性倾向）而难以使用这些模型。我们描述了参与者为克服这些模型的局限性而开发的各种策略，并探讨了尽管存在这些局限性，参与者仍然认为这些高度规范化的技术具有价值的原因。基于女权主义的同性恋理论，我们讨论了对“最先进”模型概念化的启示，并考虑FAccT研究人员如何支持同性恋替代方案。 

---
# Solving Bayesian inverse problems with diffusion priors and off-policy RL 

**Title (ZH)**: 使用扩散先验和离策强化学习求解贝叶斯逆问题 

**Authors**: Luca Scimeca, Siddarth Venkatraman, Moksh Jain, Minsu Kim, Marcin Sendera, Mohsin Hasan, Luke Rowe, Sarthak Mittal, Pablo Lemos, Emmanuel Bengio, Alexandre Adam, Jarrid Rector-Brooks, Yashar Hezaveh, Laurence Perreault-Levasseur, Yoshua Bengio, Glen Berseth, Nikolay Malkin  

**Link**: [PDF](https://arxiv.org/pdf/2503.09746)  

**Abstract**: This paper presents a practical application of Relative Trajectory Balance (RTB), a recently introduced off-policy reinforcement learning (RL) objective that can asymptotically solve Bayesian inverse problems optimally. We extend the original work by using RTB to train conditional diffusion model posteriors from pretrained unconditional priors for challenging linear and non-linear inverse problems in vision, and science. We use the objective alongside techniques such as off-policy backtracking exploration to improve training. Importantly, our results show that existing training-free diffusion posterior methods struggle to perform effective posterior inference in latent space due to inherent biases. 

**Abstract (ZH)**: 本文介绍了相对轨迹平衡(RTB)的实用性应用，RTB是最近引入的一种_off-policy_强化学习(RL)目标，能够渐近地最优解决贝叶斯逆问题。我们通过使用RTB从预训练的无条件先验中训练条件扩散模型后验，解决了视觉和科学领域中的挑战性线性和非线性逆问题。我们使用该目标与其他技术如_off-policy_回溯探索相结合，以改进训练。重要的是，我们的结果表明，现有的无训练扩散后验方法由于固有的偏差，在潜在空间中难以进行有效的后验推理。 

---
# Unveiling Hidden Pivotal Players with GoalNet: A GNN-Based Soccer Player Evaluation System 

**Title (ZH)**: 使用GoalNet揭示隐藏的关键球员：基于GNN的足球球员评估系统 

**Authors**: Jacky Hao Jiang, Jerry Cai, Anastasios Kyrillidis  

**Link**: [PDF](https://arxiv.org/pdf/2503.09737)  

**Abstract**: Soccer analysis tools emphasize metrics such as expected goals, leading to an overrepresentation of attacking players' contributions and overlooking players who facilitate ball control and link attacks. Examples include Rodri from Manchester City and Palhinha who just transferred to Bayern Munich. To address this bias, we aim to identify players with pivotal roles in a soccer team, incorporating both spatial and temporal features.
In this work, we introduce a GNN-based framework that assigns individual credit for changes in expected threat (xT), thus capturing overlooked yet vital contributions in soccer. Our pipeline encodes both spatial and temporal features in event-centric graphs, enabling fair attribution of non-scoring actions such as defensive or transitional plays. We incorporate centrality measures into the learned player embeddings, ensuring that ball-retaining defenders and defensive midfielders receive due recognition for their overall impact. Furthermore, we explore diverse GNN variants-including Graph Attention Networks and Transformer-based models-to handle long-range dependencies and evolving match contexts, discussing their relative performance and computational complexity. Experiments on real match data confirm the robustness of our approach in highlighting pivotal roles that traditional attacking metrics typically miss, underscoring the model's utility for more comprehensive soccer analytics. 

**Abstract (ZH)**: 基于GNN的足球分析工具框架：识别关键球员并公平归因非得分行为 

---
# Finding the Muses: Identifying Coresets through Loss Trajectories 

**Title (ZH)**: 寻找灵感：通过损失轨迹识别核样本 

**Authors**: Manish Nagaraj, Deepak Ravikumar, Efstathia Soufleri, Kaushik Roy  

**Link**: [PDF](https://arxiv.org/pdf/2503.09721)  

**Abstract**: Deep learning models achieve state-of-the-art performance across domains but face scalability challenges in real-time or resource-constrained scenarios. To address this, we propose Loss Trajectory Correlation (LTC), a novel metric for coreset selection that identifies critical training samples driving generalization. $LTC$ quantifies the alignment between training sample loss trajectories and validation set loss trajectories, enabling the construction of compact, representative subsets. Unlike traditional methods with computational and storage overheads that are infeasible to scale to large datasets, $LTC$ achieves superior efficiency as it can be computed as a byproduct of training. Our results on CIFAR-100 and ImageNet-1k show that $LTC$ consistently achieves accuracy on par with or surpassing state-of-the-art coreset selection methods, with any differences remaining under 1%. LTC also effectively transfers across various architectures, including ResNet, VGG, DenseNet, and Swin Transformer, with minimal performance degradation (<2%). Additionally, LTC offers insights into training dynamics, such as identifying aligned and conflicting sample behaviors, at a fraction of the computational cost of traditional methods. This framework paves the way for scalable coreset selection and efficient dataset optimization. 

**Abstract (ZH)**: 深度学习模型在各个领域 achieves 现有技术水平，但在实时或资源受限场景下面临可扩展性挑战。为解决这一问题，我们提出了损失轨迹相关性（LTC）这一新颖的聚芯选择度量标准，以识别驱动泛化的关键训练样本。LTC 通过量化训练样本损失轨迹与验证集损失轨迹之间的对齐程度，能够构建紧凑且具代表性的子集。与传统方法相比，LTC 不会产生额外的计算和存储开销，因此在处理大规模数据集时更具优势。我们的实验结果表明，LTC 在 Cifar-100 和 ImageNet-1k 上的准确率与现有最先进的聚芯选择方法相当或更优，差异不超过 1%。此外，LTC 在包括 ResNet、VGG、DenseNet 和 Swin Transformer 等各种架构上表现出色，性能退化低于 2%。LTC 还在较低的计算成本下提供了训练动态的见解，如识别对齐和冲突样本的行为。这一框架为可扩展的聚芯选择和高效的数据集优化铺平了道路。 

---
# Revisiting Backdoor Attacks on Time Series Classification in the Frequency Domain 

**Title (ZH)**: frequency域中时间序列分类的后门攻击再探 

**Authors**: Yuanmin Huang, Mi Zhang, Zhaoxiang Wang, Wenxuan Li, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.09712)  

**Abstract**: Time series classification (TSC) is a cornerstone of modern web applications, powering tasks such as financial data analysis, network traffic monitoring, and user behavior analysis. In recent years, deep neural networks (DNNs) have greatly enhanced the performance of TSC models in these critical domains. However, DNNs are vulnerable to backdoor attacks, where attackers can covertly implant triggers into models to induce malicious outcomes. Existing backdoor attacks targeting DNN-based TSC models remain elementary. In particular, early methods borrow trigger designs from computer vision, which are ineffective for time series data. More recent approaches utilize generative models for trigger generation, but at the cost of significant computational complexity. In this work, we analyze the limitations of existing attacks and introduce an enhanced method, FreqBack. Drawing inspiration from the fact that DNN models inherently capture frequency domain features in time series data, we identify that improper perturbations in the frequency domain are the root cause of ineffective attacks. To address this, we propose to generate triggers both effectively and efficiently, guided by frequency analysis. FreqBack exhibits substantial performance across five models and eight datasets, achieving an impressive attack success rate of over 90%, while maintaining less than a 3% drop in model accuracy on clean data. 

**Abstract (ZH)**: 基于时间序列分类的频域后门攻击：FreqBack方法 

---
# Revisiting semi-supervised learning in the era of foundation models 

**Title (ZH)**: revisiting 半监督学习于大模型时代 

**Authors**: Ping Zhang, Zheda Mai, Quang-Huy Nguyen, Wei-Lun Chao  

**Link**: [PDF](https://arxiv.org/pdf/2503.09707)  

**Abstract**: Semi-supervised learning (SSL) leverages abundant unlabeled data alongside limited labeled data to enhance learning. As vision foundation models (VFMs) increasingly serve as the backbone of vision applications, it remains unclear how SSL interacts with these pre-trained models. To address this gap, we develop new SSL benchmark datasets where frozen VFMs underperform and systematically evaluate representative SSL methods. We make a surprising observation: parameter-efficient fine-tuning (PEFT) using only labeled data often matches SSL performance, even without leveraging unlabeled data. This motivates us to revisit self-training, a conceptually simple SSL baseline, where we use the supervised PEFT model to pseudo-label unlabeled data for further training. To overcome the notorious issue of noisy pseudo-labels, we propose ensembling multiple PEFT approaches and VFM backbones to produce more robust pseudo-labels. Empirical results validate the effectiveness of this simple yet powerful approach, providing actionable insights into SSL with VFMs and paving the way for more scalable and practical semi-supervised learning in the era of foundation models. 

**Abstract (ZH)**: 半监督学习（SSL）利用丰富的未标签数据和有限的标签数据来增强学习。随着视觉基础模型（VFMs）在视觉应用中越来越担任骨干角色，尚不清楚SSL如何与这些预训练模型互动。为填补这一空白，我们开发了新的SSL基准数据集，在这些数据集上冻结的VFMs表现不佳，并系统地评估了代表性的SSL方法。我们意外地发现：仅使用标签数据的参数高效微调（PEFT）往往能匹配SSL性能，甚至无需利用未标签数据。这一发现促使我们重新审视自训练这一概念上简单的SSL基线方法，通过使用监督PEFT模型为未标签数据生成伪标签来进行进一步训练。为克服伪标签噪声这一著名问题，我们提出将多个PEFT方法和VFMs的骨干网络进行集成，以生成更稳健的伪标签。实验结果验证了这一简单而有效的方法的有效性，为使用VFMs的SSL提供了 actionable 洞察，并为基于基础模型时代的可扩展且实用的半监督学习铺平了道路。 

---
# Towards Robust Model Evolution with Algorithmic Recourse 

**Title (ZH)**: 面向具有算法干预的鲁棒模型演化 

**Authors**: Hao-Tsung Yang, Jie Gao, Bo-Yi Liu, Zhi-Xuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.09658)  

**Abstract**: Algorithmic Recourse is a way for users to modify their attributes to align with a model's expectations, thereby improving their outcomes after receiving unfavorable decisions. In real-world scenarios, users often need to strategically adjust their attributes to compete for limited resources. However, such strategic behavior induces users to "game" algorithms, causing model collapse due to distribution shifts. These shifts arise from user competition, resource constraints, and adaptive user responses. While prior research on Algorithmic Recourse has explored its effects on both systems and users, the impact of resource constraints and competition over time remains underexplored. In this work, we develop a general framework to model user strategic behaviors and their interactions with decision-making systems under resource constraints and competitive dynamics. Through theoretical analysis and empirical evaluation, we identify three key phenomena that arise consistently in both synthetic and real-world datasets: escalating decision boundaries, non-robust model predictions, and inequitable recourse actions. Finally, we discuss the broader social implications of these findings and present two algorithmic strategies aimed at mitigating these challenges. 

**Abstract (ZH)**: 算法可溯性是用户修改其属性以使其与模型期望相一致的一种方式，从而在收到不利决定后改善其结果。在现实场景中，用户往往需要战略性地调整其属性以竞争有限资源。然而，这种战略性行为促使用户“游戏”算法，导致由于分布移位而使模型崩溃。这些移位源于用户竞争、资源限制和用户适应性反应。尽管先前关于算法可溯性的研究探讨了其对系统和用户的影响，但资源限制和时间竞争效应的影响尚未得到充分探索。在本文中，我们开发了一种通用框架来建模在资源限制和竞争动态下用户的战略性行为及其与决策系统之间的互动。通过理论分析和实证评估，我们发现了在合成数据集和真实数据集上均一致出现的三种关键现象：决策边界升级、模型预测不稳健以及不公的可溯性行动。最后，我们讨论了这些发现的更广泛社会影响，并提出了两种算法策略以缓解这些挑战。 

---
# Empowering the Future Workforce: Prioritizing Education for the AI-Accelerated Job Market 

**Title (ZH)**: 赋能未来 workforce: 优先推动面向AI加速job市场的发展教育 

**Authors**: Lisa Amini, Henry F. Korth, Nita Patel, Evan Peck, Ben Zorn  

**Link**: [PDF](https://arxiv.org/pdf/2503.09613)  

**Abstract**: AI's rapid integration into the workplace demands new approaches to workforce education and training and broader AI literacy across disciplines. Coordinated action from government, industry, and educational institutions is necessary to ensure workers can adapt to accelerating technological change. 

**Abstract (ZH)**: AI在职场的迅速融入要求跨学科的新培训方法和更广泛的AI素养，并需政府、行业和教育机构的协调行动以确保工人能够适应加速的技术变革。 

---
