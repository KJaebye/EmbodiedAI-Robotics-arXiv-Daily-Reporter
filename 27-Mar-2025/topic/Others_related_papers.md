# Safety integrity framework for automated driving 

**Title (ZH)**: 自动-driving的安全性完整性框架 

**Authors**: Moritz Werling, Rainer Faller, Wolfgang Betz, Daniel Straub  

**Link**: [PDF](https://arxiv.org/pdf/2503.20544)  

**Abstract**: This paper describes the comprehensive safety framework that underpinned the development, release process, and regulatory approval of BMW's first SAE Level 3 Automated Driving System. The framework combines established qualitative and quantitative methods from the fields of Systems Engineering, Engineering Risk Analysis, Bayesian Data Analysis, Design of Experiments, and Statistical Learning in a novel manner. The approach systematically minimizes the risks associated with hardware and software faults, performance limitations, and insufficient specifications to an acceptable level that achieves a Positive Risk Balance. At the core of the framework is the systematic identification and quantification of uncertainties associated with hazard scenarios and the redundantly designed system based on designed experiments, field data, and expert knowledge. The residual risk of the system is then estimated through Stochastic Simulation and evaluated by Sensitivity Analysis. By integrating these advanced analytical techniques into the V-Model, the framework fulfills, unifies, and complements existing automotive safety standards. It therefore provides a comprehensive, rigorous, and transparent safety assurance process for the development and deployment of Automated Driving Systems. 

**Abstract (ZH)**: 本文描述了支撑宝马首款SAE Level 3自动驾驶系统开发、发布过程和监管审批的综合安全框架。该框架以系统工程、工程风险管理、贝叶斯数据分析、试验设计和统计学习等领域的成熟定量和定性方法为基础，以新颖的方式加以整合。该方法系统地将与硬件和软件故障、性能限制以及功能不足相关的风险降至可接受水平，实现积极的风险平衡。该框架的核心是对潜在风险场景中的不确定性进行系统识别和量化，并基于试验设计、现场数据和专家知识构建冗余系统。通过蒙特卡洛模拟估计系统的残余风险，并通过灵敏度分析进行评估。通过将这些高级分析技术整合到V模型中，该框架实现了、统一和补充了现有的汽车安全标准，因此为自动驾驶系统的开发和部署提供了全面、严格和透明的安全保障过程。 

---
# Combining Machine Learning and Sampling-Based Search for Multi-Goal Motion Planning with Dynamics 

**Title (ZH)**: 基于采样搜索的机器学习与动力学约束多目标运动规划 

**Authors**: Yuanjie Lu, Erion Plaku  

**Link**: [PDF](https://arxiv.org/pdf/2503.20530)  

**Abstract**: This paper considers multi-goal motion planning in unstructured, obstacle-rich environments where a robot is required to reach multiple regions while avoiding collisions. The planned motions must also satisfy the differential constraints imposed by the robot dynamics. To find solutions efficiently, this paper leverages machine learning, Traveling Salesman Problem (TSP), and sampling-based motion planning. The approach expands a motion tree by adding collision-free and dynamically-feasible trajectories as branches. A TSP solver is used to compute a tour for each node to determine the order in which to reach the remaining goals by utilizing a cost matrix. An important aspect of the approach is that it leverages machine learning to construct the cost matrix by combining runtime and distance predictions to single-goal motion-planning problems. During the motion-tree expansion, priority is given to nodes associated with low-cost tours. Experiments with a vehicle model operating in obstacle-rich environments demonstrate the computational efficiency and scalability of the approach. 

**Abstract (ZH)**: 本文考虑在充满障碍的未结构化环境中开展多目标运动规划，要求机器人到达多个区域并避免碰撞。所规划的运动还需满足由机器人动力学施加的微分约束。为高效找到解，本文利用机器学习、旅行商问题（TSP）和基于采样的运动规划方法。该方法通过添加无碰撞且动力学可行的轨迹作为分支来扩展运动树。使用TSP求解器计算每个节点的成本矩阵，以确定到达剩余目标的顺序。该方法的一个重要方面在于利用机器学习结合单目标运动规划的运行时和距离预测来构造成本矩阵。在运动树扩展过程中，优先处理与低成本路线相关的节点。实验结果显示，该方法在障碍丰富的环境中表现出良好的计算效率和可扩展性。 

---
# CTS-CBS: A New Approach for Multi-Agent Collaborative Task Sequencing and Path Finding 

**Title (ZH)**: CTS-CBS：一种新的多agents协作任务序列化和路径寻找方法 

**Authors**: Junkai Jiang, Ruochen Li, Yibin Yang, Yihe Chen, Yuning Wang, Shaobing Xu, Jianqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20324)  

**Abstract**: This paper addresses a generalization problem of Multi-Agent Pathfinding (MAPF), called Collaborative Task Sequencing - Multi-Agent Pathfinding (CTS-MAPF), where agents must plan collision-free paths and visit a series of intermediate task locations in a specific order before reaching their final destinations. To address this problem, we propose a new approach, Collaborative Task Sequencing - Conflict-Based Search (CTS-CBS), which conducts a two-level search. In the high level, it generates a search forest, where each tree corresponds to a joint task sequence derived from the jTSP solution. In the low level, CTS-CBS performs constrained single-agent path planning to generate paths for each agent while adhering to high-level constraints. We also provide heoretical guarantees of its completeness and optimality (or sub-optimality with a bounded parameter). To evaluate the performance of CTS-CBS, we create two datasets, CTS-MAPF and MG-MAPF, and conduct comprehensive experiments. The results show that CTS-CBS adaptations for MG-MAPF outperform baseline algorithms in terms of success rate (up to 20 times larger) and runtime (up to 100 times faster), with less than a 10% sacrifice in solution quality. Furthermore, CTS-CBS offers flexibility by allowing users to adjust the sub-optimality bound omega to balance between solution quality and efficiency. Finally, practical robot tests demonstrate the algorithm's applicability in real-world scenarios. 

**Abstract (ZH)**: CTS-CBS：协同任务序列化-冲突基于搜索的多智能体路径规划 

---
# Bandwidth Allocation for Cloud-Augmented Autonomous Driving 

**Title (ZH)**: 基于云增强的自动驾驶带宽分配 

**Authors**: Peter Schafhalter, Alexander Krentsel, Joseph E. Gonzalez, Sylvia Ratnasamy, Scott Shenker, Ion Stoica  

**Link**: [PDF](https://arxiv.org/pdf/2503.20127)  

**Abstract**: Autonomous vehicle (AV) control systems increasingly rely on ML models for tasks such as perception and planning. Current practice is to run these models on the car's local hardware due to real-time latency constraints and reliability concerns, which limits model size and thus accuracy. Prior work has observed that we could augment current systems by running larger models in the cloud, relying on faster cloud runtimes to offset the cellular network latency. However, prior work does not account for an important practical constraint: limited cellular bandwidth. We show that, for typical bandwidth levels, proposed techniques for cloud-augmented AV models take too long to transfer data, thus mostly falling back to the on-car models and resulting in no accuracy improvement.
In this work, we show that realizing cloud-augmented AV models requires intelligent use of this scarce bandwidth, i.e. carefully allocating bandwidth across tasks and providing multiple data compression and model options. We formulate this as a resource allocation problem to maximize car utility, and present our system \sysname which achieves an increase in average model accuracy by up to 15 percentage points on driving scenarios from the Waymo Open Dataset. 

**Abstract (ZH)**: 自主驾驶车辆（AV）控制系统越来越多地依赖于机器学习模型完成感知和规划等任务。由于实时延迟和可靠性方面的考虑，当前的做法是将这些模型运行在汽车本地硬件上，这限制了模型的大小，从而降低了准确性。前期工作观察到，可以通过在云端运行更大规模的模型来增强现有系统，利用更快的云计算运行时间来抵消蜂窝网络延迟。然而，前期工作并未考虑到一个重要的实践约束：有限的蜂窝带宽。我们发现，在典型带宽水平下，提议的增强型AV模型技术在数据传输方面耗时过长，从而主要依赖于本地模型，导致未能提高准确性。

在本文中，我们展示了实现增强型AV模型需要智能利用稀缺带宽，即跨任务精细分配带宽，并提供多种数据压缩和模型选项。我们将这定义为一种资源分配问题，旨在最大化车辆利用率，并提出了我们的系统\sysname，在Waymo开放数据集的驾驶场景中，实现了平均模型准确性最多提高15个百分点。 

---
# Immersive and Wearable Thermal Rendering for Augmented Reality 

**Title (ZH)**: 沉浸式可穿戴热渲染技术在增强现实中的应用 

**Authors**: Alexandra Watkins, Ritam Ghosh, Evan Chow, Nilanjan Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2503.20646)  

**Abstract**: In augmented reality (AR), where digital content is overlaid onto the real world, realistic thermal feedback has been shown to enhance immersion. Yet current thermal feedback devices, heavily influenced by the needs of virtual reality, often hinder physical interactions and are ineffective for immersion in AR. To bridge this gap, we have identified three design considerations relevant for AR thermal feedback: indirect feedback to maintain dexterity, thermal passthrough to preserve real-world temperature perception, and spatiotemporal rendering for dynamic sensations. We then created a unique and innovative thermal feedback device that satisfies these criteria. Human subject experiments assessing perceptual sensitivity, object temperature matching, spatial pattern recognition, and moving thermal stimuli demonstrated the impact of our design, enabling realistic temperature discrimination, virtual object perception, and enhanced immersion. These findings demonstrate that carefully designed thermal feedback systems can bridge the sensory gap between physical and virtual interactions, enhancing AR realism and usability. 

**Abstract (ZH)**: 在增强现实（AR）中，将数字内容叠加到现实世界，现实的热反馈已被证明可以增强沉浸感。然而，当前的热反馈设备受到虚拟现实需求的影响，往往妨碍物理互动，并在AR中无效。为了弥合这一差距，我们确定了三个适用于AR热反馈的设计考量：间接反馈以保持灵巧性、热传递以保持真实世界的温度感知以及时空渲染以产生动态感觉。然后，我们创造了一种独特且创新的热反馈设备，符合这些标准。人体实验评估感知敏感性、物体温度匹配、空间模式识别以及移动热刺激的效果，证明了我们设计的影响，实现了现实的温度区分、虚拟对象感知和增强的沉浸感。这些发现表明，精心设计的热反馈系统可以弥合物理互动和虚拟互动之间的感知差距，增强AR的真实性和可用性。 

---
# Representation Improvement in Latent Space for Search-Based Testing of Autonomous Robotic Systems 

**Title (ZH)**: 基于搜索的自主机器人系统测试中Latent空间表示改进 

**Authors**: Dmytro Humeniuk, Foutse Khomh  

**Link**: [PDF](https://arxiv.org/pdf/2503.20642)  

**Abstract**: Testing autonomous robotic systems, such as self-driving cars and unmanned aerial vehicles, is challenging due to their interaction with highly unpredictable environments. A common practice is to first conduct simulation-based testing, which, despite reducing real-world risks, remains time-consuming and resource-intensive due to the vast space of possible test scenarios. A number of search-based approaches were proposed to generate test scenarios more efficiently. A key aspect of any search-based test generation approach is the choice of representation used during the search process. However, existing methods for improving test scenario representation remain limited. We propose RILaST (Representation Improvement in Latent Space for Search-Based Testing) approach, which enhances test representation by mapping it to the latent space of a variational autoencoder. We evaluate RILaST on two use cases, including autonomous drone and autonomous lane-keeping assist system. The obtained results show that RILaST allows finding between 3 to 4.6 times more failures than baseline approaches, achieving a high level of test diversity. 

**Abstract (ZH)**: 基于潜在空间的测试表示改进在自主系统测试中的应用 

---
# Reasoning and Learning a Perceptual Metric for Self-Training of Reflective Objects in Bin-Picking with a Low-cost Camera 

**Title (ZH)**: 基于低成本摄像头的反射物体_bin-拾取_自训练感知度量推理与学习 

**Authors**: Peiyuan Ni, Chee Meng Chew, Marcelo H. Ang Jr., Gregory S. Chirikjian  

**Link**: [PDF](https://arxiv.org/pdf/2503.20207)  

**Abstract**: Bin-picking of metal objects using low-cost RGB-D cameras often suffers from sparse depth information and reflective surface textures, leading to errors and the need for manual labeling. To reduce human intervention, we propose a two-stage framework consisting of a metric learning stage and a self-training stage. Specifically, to automatically process data captured by a low-cost camera (LC), we introduce a Multi-object Pose Reasoning (MoPR) algorithm that optimizes pose hypotheses under depth, collision, and boundary constraints. To further refine pose candidates, we adopt a Symmetry-aware Lie-group based Bayesian Gaussian Mixture Model (SaL-BGMM), integrated with the Expectation-Maximization (EM) algorithm, for symmetry-aware filtering. Additionally, we propose a Weighted Ranking Information Noise Contrastive Estimation (WR-InfoNCE) loss to enable the LC to learn a perceptual metric from reconstructed data, supporting self-training on untrained or even unseen objects. Experimental results show that our approach outperforms several state-of-the-art methods on both the ROBI dataset and our newly introduced Self-ROBI dataset. 

**Abstract (ZH)**: 使用低成本RGB-D相机拾取金属物体的方法通常受到稀疏深度信息和反射表面纹理的影响，导致错误和需要手动标注。为降低人工干预，我们提出了一种两阶段框架，包括度量学习阶段和自我训练阶段。具体而言，为了自动处理低成本相机（LC）捕获的数据，我们引入了一种多目标姿态推理（MoPR）算法，该算法在深度、碰撞和边界约束下优化姿态假设。为了进一步细化姿态候选人，我们采用了一种基于Lie群的贝叶斯高斯混合模型（SaL-BGMM），并结合EM算法进行对称性意识过滤。此外，我们提出了一种加权排名信息噪声对比估计（WR-InfoNCE）损失，使LC能够从重构数据中学习感知度量，支持未训练或甚至未见过的物体的自我训练。实验结果表明，我们的方法在ROBI数据集和我们新引入的Self-ROBI数据集上均优于几种最先进的方法。 

---
# Graph-Enhanced Model-Free Reinforcement Learning Agents for Efficient Power Grid Topological Control 

**Title (ZH)**: 基于图增强的模型自由强化学习电网拓扑控制代理 

**Authors**: Eloy Anguiano Batanero, Ángela Fernández, Álvaro Barbero  

**Link**: [PDF](https://arxiv.org/pdf/2503.20688)  

**Abstract**: The increasing complexity of power grid management, driven by the emergence of prosumers and the demand for cleaner energy solutions, has needed innovative approaches to ensure stability and efficiency. This paper presents a novel approach within the model-free framework of reinforcement learning, aimed at optimizing power network operations without prior expert knowledge. We introduce a masked topological action space, enabling agents to explore diverse strategies for cost reduction while maintaining reliable service using the state logic as a guide for choosing proper actions. Through extensive experimentation across 20 different scenarios in a simulated 5-substation environment, we demonstrate that our approach achieves a consistent reduction in power losses, while ensuring grid stability against potential blackouts. The results underscore the effectiveness of combining dynamic observation formalization with opponent-based training, showing a viable way for autonomous management solutions in modern energy systems or even for building a foundational model for this field. 

**Abstract (ZH)**: 随着生产者消费者和对清洁能源解决方案需求的出现，电网管理日益复杂，需要创新方法以确保稳定性和效率。本文提出了一种基于 reinforcement learning 的无模型框架下的新颖方法，旨在在无需先验专家知识的情况下优化电力网络运行。我们引入了屏蔽拓扑动作空间，使代理能够在保证可靠服务的同时，利用状态逻辑作为选择适当行动的指南，探索降低成本的多种策略。通过在模拟的5个变电站环境下的20种不同场景中进行大量实验，我们展示了我们的方法能够持续降低电力损耗，同时确保在潜在停电情况下的电网稳定性。结果强调了将动态观察形式化与对手训练相结合的有效性，为现代能源系统提供了自主管理解决方案的可能性，甚至可以为基础模型的建立提供范式。 

---
# Inductive Link Prediction on N-ary Relational Facts via Semantic Hypergraph Reasoning 

**Title (ZH)**: 基于语义超图推理的N元关系事实归纳链接预测 

**Authors**: Gongzhu Yin, Hongli Zhang, Yuchen Yang, Yi Luo  

**Link**: [PDF](https://arxiv.org/pdf/2503.20676)  

**Abstract**: N-ary relational facts represent semantic correlations among more than two entities. While recent studies have developed link prediction (LP) methods to infer missing relations for knowledge graphs (KGs) containing n-ary relational facts, they are generally limited to transductive settings. Fully inductive settings, where predictions are made on previously unseen entities, remain a significant challenge. As existing methods are mainly entity embedding-based, they struggle to capture entity-independent logical rules. To fill in this gap, we propose an n-ary subgraph reasoning framework for fully inductive link prediction (ILP) on n-ary relational facts. This framework reasons over local subgraphs and has a strong inductive inference ability to capture n-ary patterns. Specifically, we introduce a novel graph structure, the n-ary semantic hypergraph, to facilitate subgraph extraction. Moreover, we develop a subgraph aggregating network, NS-HART, to effectively mine complex semantic correlations within subgraphs. Theoretically, we provide a thorough analysis from the score function optimization perspective to shed light on NS-HART's effectiveness for n-ary ILP tasks. Empirically, we conduct extensive experiments on a series of inductive benchmarks, including transfer reasoning (with and without entity features) and pairwise subgraph reasoning. The results highlight the superiority of the n-ary subgraph reasoning framework and the exceptional inductive ability of NS-HART. The source code of this paper has been made publicly available at this https URL. 

**Abstract (ZH)**: N-元关系事实表示多个实体之间的语义关联。虽然近年的研究开发了链接预测（LP）方法来推断知识图谱（KGs）中包含N-元关系事实的缺失关系，但这些方法通常局限于归纳性设置。在未见过的新实体上进行预测的完全归纳性设置仍然是一项重大挑战。由于现有方法主要是基于实体嵌入的，它们难以捕捉到与实体无关的逻辑规则。为填补这一空白，我们提出了一种针对N-元关系事实的完全归纳性链接预测（ILP）的N-元子图推理框架。该框架在局部子图上进行推理，并具备较强的归纳推断能力来捕捉N-元模式。具体而言，我们引入了一种新的图结构，N-元语义超图，以方便子图提取。此外，我们开发了一种子图聚合网络NS-HART，以有效地挖掘子图内的复杂语义关联。从理论上讲，我们从评分函数优化的角度进行了全面分析，以阐明NS-HART在N-元ILP任务中的有效性。从实验上讲，我们在一系列归纳基准测试上进行了广泛实验，包括转移推理（有和没有实体特征）以及成对子图推理。实验结果突显了N-元子图推理框架和NS-HART的卓越归纳能力。本文的源代码已公开，可访问以下链接：此 https URL。 

---
# Procedural Knowledge Ontology (PKO) 

**Title (ZH)**: 过程性知识本体（PKO） 

**Authors**: Valentina Anita Carriero, Mario Scrocca, Ilaria Baroni, Antonia Azzini, Irene Celino  

**Link**: [PDF](https://arxiv.org/pdf/2503.20634)  

**Abstract**: Processes, workflows and guidelines are core to ensure the correct functioning of industrial companies: for the successful operations of factory lines, machinery or services, often industry operators rely on their past experience and know-how. The effect is that this Procedural Knowledge (PK) remains tacit and, as such, difficult to exploit efficiently and effectively. This paper presents PKO, the Procedural Knowledge Ontology, which enables the explicit modeling of procedures and their executions, by reusing and extending existing ontologies. PKO is built on requirements collected from three heterogeneous industrial use cases and can be exploited by any AI and data-driven tools that rely on a shared and interoperable representation to support the governance of PK throughout its life cycle. We describe its structure and design methodology, and outline its relevance, quality, and impact by discussing applications leveraging PKO for PK elicitation and exploitation. 

**Abstract (ZH)**: 基于过程的知识本体：确保工业公司正确运作的核心要素 

---
# Unsupervised Learning for Quadratic Assignment 

**Title (ZH)**: 无监督学习在二次指派问题中的应用 

**Authors**: Yimeng Min, Carla P. Gomes  

**Link**: [PDF](https://arxiv.org/pdf/2503.20001)  

**Abstract**: We introduce PLUME search, a data-driven framework that enhances search efficiency in combinatorial optimization through unsupervised learning. Unlike supervised or reinforcement learning, PLUME search learns directly from problem instances using a permutation-based loss with a non-autoregressive approach. We evaluate its performance on the quadratic assignment problem, a fundamental NP-hard problem that encompasses various combinatorial optimization problems. Experimental results demonstrate that PLUME search consistently improves solution quality. Furthermore, we study the generalization behavior and show that the learned model generalizes across different densities and sizes. 

**Abstract (ZH)**: PLUME搜索：一种通过无监督学习增强组合优化搜索效率的数据驱动框架 

---
