# Energy-aware Joint Orchestration of 5G and Robots: Experimental Testbed and Field Validation 

**Title (ZH)**: 5G与机器人能量感知联合协同：实验测试床与现场验证 

**Authors**: Milan Groshev, Lanfranco Zanzi, Carmen Delgado, Xi Li, Antonio de la Oliva, Xavier Costa-Perez  

**Link**: [PDF](https://arxiv.org/pdf/2503.19613)  

**Abstract**: 5G mobile networks introduce a new dimension for connecting and operating mobile robots in outdoor environments, leveraging cloud-native and offloading features of 5G networks to enable fully flexible and collaborative cloud robot operations. However, the limited battery life of robots remains a significant obstacle to their effective adoption in real-world exploration scenarios. This paper explores, via field experiments, the potential energy-saving gains of OROS, a joint orchestration of 5G and Robot Operating System (ROS) that coordinates multiple 5G-connected robots both in terms of navigation and sensing, as well as optimizes their cloud-native service resource utilization while minimizing total resource and energy consumption on the robots based on real-time feedback. We designed, implemented and evaluated our proposed OROS in an experimental testbed composed of commercial off-the-shelf robots and a local 5G infrastructure deployed on a campus. The experimental results demonstrated that OROS significantly outperforms state-of-the-art approaches in terms of energy savings by offloading demanding computational tasks to the 5G edge infrastructure and dynamic energy management of on-board sensors (e.g., switching them off when they are not needed). This strategy achieves approximately 15% energy savings on the robots, thereby extending battery life, which in turn allows for longer operating times and better resource utilization. 

**Abstract (ZH)**: 5G移动网络为户外环境中的移动机器人连接和操作引入了新的维度，通过利用5G网络的云原生和卸载特性，实现全方位灵活协作的云机器人操作。然而，机器人的有限电池寿命仍然是其在实际探索场景中有效应用的一个重要障碍。本文通过现场实验探讨了OROS（5G和机器人操作系统联合 orchestration）的节能潜力，该方法协调多台5G连接的机器人在导航和感知方面，并优化其云原生服务资源利用，同时基于实时反馈最小化总资源和能量消耗。我们设计、实施并评估了提出的OROS，该系统由商用现成的机器人和部署在校园内的本地5G基础设施构成。实验结果表明，OROS通过将计算密集型任务卸载到5G边缘基础设施以及动态管理机载传感器的能量（例如，在不需要时关闭它们）在节能方面显著优于现有方法，实现了约15%的节能效果，从而延长了电池寿命，进而允许更长的运行时间和更好的资源利用。 

---
# Towards Uncertainty Unification: A Case Study for Preference Learning 

**Title (ZH)**: 向不确定性统一迈进：一种偏好学习的案例研究 

**Authors**: Shaoting Peng, Haonan Chen, Katherine Driggs-Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2503.19317)  

**Abstract**: Learning human preferences is essential for human-robot interaction, as it enables robots to adapt their behaviors to align with human expectations and goals. However, the inherent uncertainties in both human behavior and robotic systems make preference learning a challenging task. While probabilistic robotics algorithms offer uncertainty quantification, the integration of human preference uncertainty remains underexplored. To bridge this gap, we introduce uncertainty unification and propose a novel framework, uncertainty-unified preference learning (UUPL), which enhances Gaussian Process (GP)-based preference learning by unifying human and robot uncertainties. Specifically, UUPL includes a human preference uncertainty model that improves GP posterior mean estimation, and an uncertainty-weighted Gaussian Mixture Model (GMM) that enhances GP predictive variance accuracy. Additionally, we design a user-specific calibration process to align uncertainty representations across users, ensuring consistency and reliability in the model performance. Comprehensive experiments and user studies demonstrate that UUPL achieves state-of-the-art performance in both prediction accuracy and user rating. An ablation study further validates the effectiveness of human uncertainty model and uncertainty-weighted GMM of UUPL. 

**Abstract (ZH)**: 学习人类偏好对于人机交互至关重要，因为它使机器人能够根据人类的期望和目标调整其行为。然而，人类行为和机器人系统的固有不确定性使得偏好学习成为一个具有挑战性的任务。虽然概率机器人算法可以提供不确定性量化，但人类偏好不确定性与机器人系统的集成仍处于探索阶段。为了弥补这一差距，我们提出了不确定性统一的方法，并提出了一种新颖的框架——不确定性统一的偏好学习（UUPL），该框架通过统一人类和机器人的不确定性来增强基于高斯过程（GP）的偏好学习。具体而言，UUPL 包括一个改进 GP 后验均值估计的人类偏好不确定性模型，以及一个增强 GP 预测方差准确性的加权高斯混合模型（GMM）。此外，我们设计了一个用户特定的校准过程，以确保不同用户之间的不确定性表示一致性，从而保证模型性能的可靠性和一致性。全面的实验和用户研究证明，UUPL 在预测准确性和用户评分方面均达到了目前的最佳性能。进一步的消融研究还验证了人类不确定性模型和 UUPL 的加权 GMM 的有效性。 

---
# SuperFlow++: Enhanced Spatiotemporal Consistency for Cross-Modal Data Pretraining 

**Title (ZH)**: SuperFlow++: 提升跨模态数据预训练的时空一致性 

**Authors**: Xiang Xu, Lingdong Kong, Hui Shuai, Wenwei Zhang, Liang Pan, Kai Chen, Ziwei Liu, Qingshan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.19912)  

**Abstract**: LiDAR representation learning has emerged as a promising approach to reducing reliance on costly and labor-intensive human annotations. While existing methods primarily focus on spatial alignment between LiDAR and camera sensors, they often overlook the temporal dynamics critical for capturing motion and scene continuity in driving scenarios. To address this limitation, we propose SuperFlow++, a novel framework that integrates spatiotemporal cues in both pretraining and downstream tasks using consecutive LiDAR-camera pairs. SuperFlow++ introduces four key components: (1) a view consistency alignment module to unify semantic information across camera views, (2) a dense-to-sparse consistency regularization mechanism to enhance feature robustness across varying point cloud densities, (3) a flow-based contrastive learning approach that models temporal relationships for improved scene understanding, and (4) a temporal voting strategy that propagates semantic information across LiDAR scans to improve prediction consistency. Extensive evaluations on 11 heterogeneous LiDAR datasets demonstrate that SuperFlow++ outperforms state-of-the-art methods across diverse tasks and driving conditions. Furthermore, by scaling both 2D and 3D backbones during pretraining, we uncover emergent properties that provide deeper insights into developing scalable 3D foundation models. With strong generalizability and computational efficiency, SuperFlow++ establishes a new benchmark for data-efficient LiDAR-based perception in autonomous driving. The code is publicly available at this https URL 

**Abstract (ZH)**: 基于LiDAR的时空特征学习：SuperFlow++在自动驾驶中的应用 

---
# A Multi-Agent Framework Integrating Large Language Models and Generative AI for Accelerated Metamaterial Design 

**Title (ZH)**: 一种集成大型语言模型和生成式AI的多Agent框架加速 metamaterial 设计 

**Authors**: Jie Tian, Martin Taylor Sobczak, Dhanush Patil, Jixin Hou, Lin Pang, Arunachalam Ramanathan, Libin Yang, Xianyan Chen, Yuval Golan, Hongyue Sun, Kenan Song, Xianqiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.19889)  

**Abstract**: Metamaterials, renowned for their exceptional mechanical, electromagnetic, and thermal properties, hold transformative potential across diverse applications, yet their design remains constrained by labor-intensive trial-and-error methods and limited data interoperability. Here, we introduce CrossMatAgent--a novel multi-agent framework that synergistically integrates large language models with state-of-the-art generative AI to revolutionize metamaterial design. By orchestrating a hierarchical team of agents--each specializing in tasks such as pattern analysis, architectural synthesis, prompt engineering, and supervisory feedback--our system leverages the multimodal reasoning of GPT-4o alongside the generative precision of DALL-E 3 and a fine-tuned Stable Diffusion XL model. This integrated approach automates data augmentation, enhances design fidelity, and produces simulation- and 3D printing-ready metamaterial patterns. Comprehensive evaluations, including CLIP-based alignment, SHAP interpretability analyses, and mechanical simulations under varied load conditions, demonstrate the framework's ability to generate diverse, reproducible, and application-ready designs. CrossMatAgent thus establishes a scalable, AI-driven paradigm that bridges the gap between conceptual innovation and practical realization, paving the way for accelerated metamaterial development. 

**Abstract (ZH)**: 超材料，凭借其卓越的机械、电磁和热学性能，在多个应用领域具有变革潜力，但其设计仍受限于劳动密集型的试错方法和数据互联互通的限制。为此，我们引入了CrossMatAgent——一种新颖的多智能体框架，将大规模语言模型与最先进的生成AI协同整合，以革命性地变革超材料设计。通过协调一个分层团队的智能体——每个智能体专注于如模式分析、建筑合成、提示工程和监督反馈等任务，我们的系统利用GPT-4o的多模态推理能力，结合DALL-E 3的生成精确性和微调后的Stable Diffusion XL模型的精准生成。这种集成方法实现了数据增强的自动化、提升了设计保真度，并生成了适用于模拟和3D打印的超材料模式。全面的评估，包括基于CLIP的对齐、SHAP可解释性分析和在不同载荷条件下的力学模拟，表明该框架能够生成多样、可复现且适用于实际应用的设计。CrossMatAgent因此建立了一种可扩展、AI驱动的范式，填补了概念创新与实际实现之间的差距，为加速超材料开发铺平了道路。 

---
# OpenLex3D: A New Evaluation Benchmark for Open-Vocabulary 3D Scene Representations 

**Title (ZH)**: OpenLex3D：一种新的开放词汇3D场景表示评价基准 

**Authors**: Christina Kassab, Sacha Morin, Martin Büchner, Matías Mattamala, Kumaraditya Gupta, Abhinav Valada, Liam Paull, Maurice Fallon  

**Link**: [PDF](https://arxiv.org/pdf/2503.19764)  

**Abstract**: 3D scene understanding has been transformed by open-vocabulary language models that enable interaction via natural language. However, the evaluation of these representations is limited to closed-set semantics that do not capture the richness of language. This work presents OpenLex3D, a dedicated benchmark to evaluate 3D open-vocabulary scene representations. OpenLex3D provides entirely new label annotations for 23 scenes from Replica, ScanNet++, and HM3D, which capture real-world linguistic variability by introducing synonymical object categories and additional nuanced descriptions. By introducing an open-set 3D semantic segmentation task and an object retrieval task, we provide insights on feature precision, segmentation, and downstream capabilities. We evaluate various existing 3D open-vocabulary methods on OpenLex3D, showcasing failure cases, and avenues for improvement. The benchmark is publicly available at: this https URL. 

**Abstract (ZH)**: 3D场景理解通过开放词汇语言模型得到革新，这些模型能够通过自然语言进行交互。然而，这些表示的评估局限于封闭集语义，不能充分捕捉语言的丰富性。本工作提出OpenLex3D，一个专门用于评估3D开放词汇场景表示的基准。OpenLex3D为来自Replica、ScanNet++和HM3D的23个场景提供了全新的标签注释，通过引入同义对象类别和额外的细微描述，捕捉现实世界中的语言变异性。通过引入开放集3D语义分割任务和对象检索任务，我们提供了关于特征精度、分割及下游能力的见解。我们对OpenLex3D上各种现有的3D开放词汇方法进行了评估，展示了失败案例及改进方向。基准已公开发布于：this https URL。 

---
# Observation Adaptation via Annealed Importance Resampling for Partially Observable Markov Decision Processes 

**Title (ZH)**: 基于退火重要性重采样的部分可观测马尔可夫决策过程的观测自适应方法 

**Authors**: Yunuo Zhang, Baiting Luo, Ayan Mukhopadhyay, Abhishek Dubey  

**Link**: [PDF](https://arxiv.org/pdf/2503.19302)  

**Abstract**: Partially observable Markov decision processes (POMDPs) are a general mathematical model for sequential decision-making in stochastic environments under state uncertainty. POMDPs are often solved \textit{online}, which enables the algorithm to adapt to new information in real time. Online solvers typically use bootstrap particle filters based on importance resampling for updating the belief distribution. Since directly sampling from the ideal state distribution given the latest observation and previous state is infeasible, particle filters approximate the posterior belief distribution by propagating states and adjusting weights through prediction and resampling steps. However, in practice, the importance resampling technique often leads to particle degeneracy and sample impoverishment when the state transition model poorly aligns with the posterior belief distribution, especially when the received observation is highly informative. We propose an approach that constructs a sequence of bridge distributions between the state-transition and optimal distributions through iterative Monte Carlo steps, better accommodating noisy observations in online POMDP solvers. Our algorithm demonstrates significantly superior performance compared to state-of-the-art methods when evaluated across multiple challenging POMDP domains. 

**Abstract (ZH)**: 部分可观测马尔可夫决策过程（POMDPs）是用于在具有状态不确定性的情况下随机环境中顺序决策的一个通用数学模型。POMDPs通常在线求解，使算法能够实时适应新信息。在线求解器通常使用基于重要性重采样的粒子滤波器来更新信念分布。由于直接从最新的观测和先前状态的理想状态分布中采样是不现实的，粒子滤波器通过预测和重采样步骤传播状态并调整权重来近似后验信念分布。然而，在实践中，当状态转移模型与后验信念分布不匹配时，重要性重采样技术往往会导致粒子衰减和样本贫乏，尤其是在接收到的观测信息非常丰富的情况下。我们提出了一种方法，通过迭代蒙特卡洛步骤构建从状态转移分布到最优分布的一系列桥梁分布，以更好地适应在线POMDP求解器中的噪声观测。我们的算法在多个具有挑战性的POMDP领域中评估时，表现出显著的优越性能。 

---
# Optimal Modified Feedback Strategies in LQ Games under Control Imperfections 

**Title (ZH)**: 在控制不完美情况下的LQ博弈的最优修正反馈策略 

**Authors**: Mahdis Rabbani, Navid Mojahed, Shima Nazari  

**Link**: [PDF](https://arxiv.org/pdf/2503.19200)  

**Abstract**: Game-theoretic approaches and Nash equilibrium have been widely applied across various engineering domains. However, practical challenges such as disturbances, delays, and actuator limitations can hinder the precise execution of Nash equilibrium strategies. This work explores the impact of such implementation imperfections on game trajectories and players' costs within the context of a two-player linear quadratic (LQ) nonzero-sum game. Specifically, we analyze how small deviations by one player affect the state and cost function of the other player. To address these deviations, we propose an adjusted control policy that not only mitigates adverse effects optimally but can also exploit the deviations to enhance performance. Rigorous mathematical analysis and proofs are presented, demonstrating through a representative example that the proposed policy modification achieves up to $61\%$ improvement compared to the unadjusted feedback policy and up to $0.59\%$ compared to the feedback Nash strategy. 

**Abstract (ZH)**: 博弈论方法和纳什均衡在各类工程领域中广泛应用。然而，诸如扰动、延迟和执行器限制等实际挑战可能阻碍纳什均衡策略的精确执行。本文探讨了这些实施缺陷对两人线性二次（LQ）非零和博弈的游戏轨迹及其参与者的成本的影响。具体而言，我们分析了一方的小幅偏差如何影响另一方的状态和成本函数。为应对这些偏差，我们提出了一种调整控制策略，不仅能够最优地减轻不利影响，还能利用这些偏差提升性能。通过严格的数学分析和证明，本文示例表明，所提策略改进比未调整的反馈策略高达61%，比反馈纳什策略高0.59%。 

---
# Evolutionary Policy Optimization 

**Title (ZH)**: 进化策略优化 

**Authors**: Jianren Wang, Yifan Su, Abhinav Gupta, Deepak Pathak  

**Link**: [PDF](https://arxiv.org/pdf/2503.19037)  

**Abstract**: Despite its extreme sample inefficiency, on-policy reinforcement learning has become a fundamental tool in real-world applications. With recent advances in GPU-driven simulation, the ability to collect vast amounts of data for RL training has scaled exponentially. However, studies show that current on-policy methods, such as PPO, fail to fully leverage the benefits of parallelized environments, leading to performance saturation beyond a certain scale. In contrast, Evolutionary Algorithms (EAs) excel at increasing diversity through randomization, making them a natural complement to RL. However, existing EvoRL methods have struggled to gain widespread adoption due to their extreme sample inefficiency. To address these challenges, we introduce Evolutionary Policy Optimization (EPO), a novel policy gradient algorithm that combines the strengths of EA and policy gradients. We show that EPO significantly improves performance across diverse and challenging environments, demonstrating superior scalability with parallelized simulations. 

**Abstract (ZH)**: 尽管强化学习的样本效率极低，但在线策略强化学习已成为现实生活应用中的基本工具。随着基于GPU的模拟技术的recent进展，用于RL训练的数据收集能力已呈指数级增长。然而，研究表明，当前的在线策略方法，如PPO，在达到一定规模后无法充分利用并行环境的优势，导致性能饱和。相比之下，进化算法通过随机化增加多样性，使其成为RL的天然补充。然而，现有的EvoRL方法因极低的样本效率而难以广泛采用。为了解决这些挑战，我们引入了进化策略优化（EPO），这是一种结合了进化算法和策略梯度优势的新型策略梯度算法。我们证明EPO在多种复杂环境中显著提高性能，并展示了其在并行模拟中的优越可扩展性。 

---
# SG-Tailor: Inter-Object Commonsense Relationship Reasoning for Scene Graph Manipulation 

**Title (ZH)**: SG-Tailor: 不同对象常识关系推理以操纵场景图 

**Authors**: Haoliang Shang, Hanyu Wu, Guangyao Zhai, Boyang Sun, Fangjinhua Wang, Federico Tombari, Marc Pollefeys  

**Link**: [PDF](https://arxiv.org/pdf/2503.18988)  

**Abstract**: Scene graphs capture complex relationships among objects, serving as strong priors for content generation and manipulation. Yet, reasonably manipulating scene graphs -- whether by adding nodes or modifying edges -- remains a challenging and untouched task. Tasks such as adding a node to the graph or reasoning about a node's relationships with all others are computationally intractable, as even a single edge modification can trigger conflicts due to the intricate interdependencies within the graph. To address these challenges, we introduce SG-Tailor, an autoregressive model that predicts the conflict-free relationship between any two nodes. SG-Tailor not only infers inter-object relationships, including generating commonsense edges for newly added nodes but also resolves conflicts arising from edge modifications to produce coherent, manipulated graphs for downstream tasks. For node addition, the model queries the target node and other nodes from the graph to predict the appropriate relationships. For edge modification, SG-Tailor employs a Cut-And-Stitch strategy to solve the conflicts and globally adjust the graph. Extensive experiments demonstrate that SG-Tailor outperforms competing methods by a large margin and can be seamlessly integrated as a plug-in module for scene generation and robotic manipulation tasks. 

**Abstract (ZH)**: 场景图捕获对象之间复杂的相互关系，作为内容生成和操控的强大先验。然而，合理地操控场景图——无论是添加节点还是修改边——仍然是一个具有挑战性和未被充分探讨的任务。诸如向图中添加节点或推断节点与所有其他节点的关系之类的任务在计算上是不可行的，因为即使是单个边的修改也可能由于图内部复杂的相互依赖而引发冲突。为了解决这些挑战，我们引入了SG-Tailor，这是一种自回归模型，用于预测图中任意两个节点之间的无冲突关系。SG-Tailor 不仅推断物体间的相互关系，包括为新添加的节点生成常识边，还能解决边修改引发的冲突，生成连贯的、被操控的图以供下游任务使用。对于节点添加，模型从图中查询目标节点和其他节点来预测适当的相互关系。对于边修改，SG-Tailor 使用剪切和缝合策略来解决冲突并全局调整图。广泛实验表明，SG-Tailor 在性能上大幅优于竞争方法，并可无缝集成到场景生成和机器人操控任务的插件模块中。 

---
# Guidelines For The Choice Of The Baseline in XAI Attribution Methods 

**Title (ZH)**: XAI归因方法中基线选择指南 

**Authors**: Cristian Morasso, Giorgio Dolci, Ilaria Boscolo Galazzo, Sergey M. Plis, Gloria Menegaz  

**Link**: [PDF](https://arxiv.org/pdf/2503.19813)  

**Abstract**: Given the broad adoption of artificial intelligence, it is essential to provide evidence that AI models are reliable, trustable, and fair. To this end, the emerging field of eXplainable AI develops techniques to probe such requirements, counterbalancing the hype pushing the pervasiveness of this technology. Among the many facets of this issue, this paper focuses on baseline attribution methods, aiming at deriving a feature attribution map at the network input relying on a "neutral" stimulus usually called "baseline". The choice of the baseline is crucial as it determines the explanation of the network behavior. In this framework, this paper has the twofold goal of shedding light on the implications of the choice of the baseline and providing a simple yet effective method for identifying the best baseline for the task. To achieve this, we propose a decision boundary sampling method, since the baseline, by definition, lies on the decision boundary, which naturally becomes the search domain. Experiments are performed on synthetic examples and validated relying on state-of-the-art methods. Despite being limited to the experimental scope, this contribution is relevant as it offers clear guidelines and a simple proxy for baseline selection, reducing ambiguity and enhancing deep models' reliability and trust. 

**Abstract (ZH)**: 随着人工智能的广泛 adoption，提供证据证明 AI 模型的可靠性、可信赖性和公平性显得尤为必要。为此，可解释 AI 这一新兴领域正在发展出技术，以满足这些要求，从而抵消对该技术普遍性的过度宣传。本文重点关注基线归因方法，旨在通过“中立”刺激（通常称为“基线”）在网络输入上推导出特征归因图。基线的选择至关重要，因为它决定了对网络行为的解释。在此框架下，本文的双重目标是揭示基线选择的影响，并提供一种简单有效的基线选择方法。为实现这一目标，我们提出了一种决策边界采样方法，因为基线被定义为决策边界的一部分，自然成为搜索域。实验在合成样本上进行，并通过最新方法进行验证。尽管仅限于实验范围，但本文的贡献依然重要，因为它提供了清晰的指导方针和简单的基线选择代理，减少了模糊性并增强了深度模型的可靠性和可信赖性。 

---
# Simulating Tracking Data to Advance Sports Analytics Research 

**Title (ZH)**: 模拟追踪数据以推动体育分析研究 

**Authors**: David Radke, Kyle Tilbury  

**Link**: [PDF](https://arxiv.org/pdf/2503.19809)  

**Abstract**: Advanced analytics have transformed how sports teams operate, particularly in episodic sports like baseball. Their impact on continuous invasion sports, such as soccer and ice hockey, has been limited due to increased game complexity and restricted access to high-resolution game tracking data. In this demo, we present a method to collect and utilize simulated soccer tracking data from the Google Research Football environment to support the development of models designed for continuous tracking data. The data is stored in a schema that is representative of real tracking data and we provide processes that extract high-level features and events. We include examples of established tracking data models to showcase the efficacy of the simulated data. We address the scarcity of publicly available tracking data, providing support for research at the intersection of artificial intelligence and sports analytics. 

**Abstract (ZH)**: 高级数据分析已 transforming 运动队的运营模式，特别是在棒球等周期性运动中。这类技术对足球和冰球等持续侵入性运动的影响有限，原因在于比赛复杂度的增加以及获取高分辨率比赛跟踪数据的限制。在本演示中，我们介绍了一种方法，用于从 Google Research Football 环境中收集和利用模拟足球跟踪数据，以支持用于持续跟踪数据的模型开发。数据存储在代表实际跟踪数据模式的结构中，并提供了提取高级特征和事件的过程。我们展示了现有的跟踪数据模型示例，以展示模拟数据的有效性，并解决了公开可用跟踪数据稀缺的问题，从而支持人工智能和运动分析交叉领域的研究。 

---
# Splitting Answer Set Programs with respect to Intensionality Statements (Extended Version) 

**Title (ZH)**: 基于意向性语句划分答案集程序（扩展版本） 

**Authors**: Jorge Fandinno, Yuliya Lierler  

**Link**: [PDF](https://arxiv.org/pdf/2503.19762)  

**Abstract**: Splitting a logic program allows us to reduce the task of computing its stable models to similar tasks for its subprograms. This can be used to increase solving performance and prove program correctness. We generalize the conditions under which this technique is applicable, by considering not only dependencies between predicates but also their arguments and context. This allows splitting programs commonly used in practice to which previous results were not applicable. 

**Abstract (ZH)**: 将逻辑程序分割以其实现子程序稳定模型的计算减少为目标，可以提高求解性能并证明程序的正确性。我们通过考虑谓词及其参数和上下文之间的依赖关系，推广了该技术可应用的条件，使得更多实践中常用的程序可以被分割处理。 

---
# Multi-agent Application System in Office Collaboration Scenarios 

**Title (ZH)**: 办公协作场景下的多Agent应用系统 

**Authors**: Songtao Sun, Jingyi Li, Yuanfei Dong, Haoguang Liu, Chenxin Xu, Fuyang Li, Qiang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.19584)  

**Abstract**: This paper introduces a multi-agent application system designed to enhance office collaboration efficiency and work quality. The system integrates artificial intelligence, machine learning, and natural language processing technologies, achieving functionalities such as task allocation, progress monitoring, and information sharing. The agents within the system are capable of providing personalized collaboration support based on team members' needs and incorporate data analysis tools to improve decision-making quality. The paper also proposes an intelligent agent architecture that separates Plan and Solver, and through techniques such as multi-turn query rewriting and business tool retrieval, it enhances the agent's multi-intent and multi-turn dialogue capabilities. Furthermore, the paper details the design of tools and multi-turn dialogue in the context of office collaboration scenarios, and validates the system's effectiveness through experiments and evaluations. Ultimately, the system has demonstrated outstanding performance in real business applications, particularly in query understanding, task planning, and tool calling. Looking forward, the system is expected to play a more significant role in addressing complex interaction issues within dynamic environments and large-scale multi-agent systems. 

**Abstract (ZH)**: 一种用于提升办公室协作效率和工作质量的多智能体应用系统及其智能代理架构研究 

---
# Browsing Lost Unformed Recollections: A Benchmark for Tip-of-the-Tongue Search and Reasoning 

**Title (ZH)**: 找回失落的未形成记忆：舌尖上的搜索与推理基准 

**Authors**: Sky CH-Wang, Darshan Deshpande, Smaranda Muresan, Anand Kannappan, Rebecca Qian  

**Link**: [PDF](https://arxiv.org/pdf/2503.19193)  

**Abstract**: We introduce Browsing Lost Unformed Recollections, a tip-of-the-tongue known-item search and reasoning benchmark for general AI assistants. BLUR introduces a set of 573 real-world validated questions that demand searching and reasoning across multi-modal and multilingual inputs, as well as proficient tool use, in order to excel on. Humans easily ace these questions (scoring on average 98%), while the best-performing system scores around 56%. To facilitate progress toward addressing this challenging and aspirational use case for general AI assistants, we release 350 questions through a public leaderboard, retain the answers to 250 of them, and have the rest as a private test set. 

**Abstract (ZH)**: 浏览迷失未形记：面向通用人工智能助手的舌尖上的回忆已知项搜索与推理基准 

---
# AssertionForge: Enhancing Formal Verification Assertion Generation with Structured Representation of Specifications and RTL 

**Title (ZH)**: AssertionForge：通过结构化规范和RTL表示增强形式验证断言生成 

**Authors**: Yunsheng Bai, Ghaith Bany Hamad, Syed Suhaib, Haoxing Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.19174)  

**Abstract**: Generating SystemVerilog Assertions (SVAs) from natural language specifications remains a major challenge in formal verification (FV) due to the inherent ambiguity and incompleteness of specifications. Existing LLM-based approaches, such as AssertLLM, focus on extracting information solely from specification documents, often failing to capture essential internal signal interactions and design details present in the RTL code, leading to incomplete or incorrect assertions. We propose a novel approach that constructs a Knowledge Graph (KG) from both specifications and RTL, using a hardware-specific schema with domain-specific entity and relation types. We create an initial KG from the specification and then systematically fuse it with information extracted from the RTL code, resulting in a unified, comprehensive KG. This combined representation enables a more thorough understanding of the design and allows for a multi-resolution context synthesis process which is designed to extract diverse verification contexts from the KG. Experiments on four designs demonstrate that our method significantly enhances SVA quality over prior methods. This structured representation not only improves FV but also paves the way for future research in tasks like code generation and design understanding. 

**Abstract (ZH)**: 从自然语言规范生成SystemVerilog断言（SVAs）在形式验证（FV）中依然面临着因规范固有的模糊性和不完整性带来的重大挑战。现有的基于LLM的方法，如AssertLLM，专注于从规范文档中提取信息，常常未能捕捉RTL代码中存在的关键内部信号交互和设计细节，导致生成的断言不完整或不正确。我们提出了一种新的方法，从规范和RTL代码中构建知识图谱（KG），使用硬件特定的模式和领域特定的实体及关系类型。我们从规范中构建初始KG，然后系统地将其与从RTL代码中提取的信息融合，形成一个统一、全面的KG。这种联合表示使对设计的理解更加深入，并允许进行一个多分辨率上下文合成过程，旨在从KG中提取多样化的验证上下文。实验结果表明，我们的方法在生成SVAs方面相比先前方法有显著提升。这种结构化表示不仅改善了形式验证，还为未来的代码生成和设计理解研究铺平了道路。 

---
# Information-Seeking Decision Strategies Mitigate Risk in Dynamic, Uncertain Environments 

**Title (ZH)**: 信息搜寻决策策略在动态不确定环境中降低风险 

**Authors**: Nicholas W. Barendregt, Joshua I. Gold, Krešimir Josić, Zachary P. Kilpatrick  

**Link**: [PDF](https://arxiv.org/pdf/2503.19107)  

**Abstract**: To survive in dynamic and uncertain environments, individuals must develop effective decision strategies that balance information gathering and decision commitment. Models of such strategies often prioritize either optimizing tangible payoffs, like reward rate, or gathering information to support a diversity of (possibly unknown) objectives. However, our understanding of the relative merits of these two approaches remains incomplete, in part because direct comparisons have been limited to idealized, static environments that lack the dynamic complexity of the real world. Here we compared the performance of normative reward- and information-seeking strategies in a dynamic foraging task. Both strategies show similar transitions between exploratory and exploitative behaviors as environmental uncertainty changes. However, we find subtle disparities in the actions they take, resulting in meaningful performance differences: whereas reward-seeking strategies generate slightly more reward on average, information-seeking strategies provide more consistent and predictable outcomes. Our findings support the adaptive value of information-seeking behaviors that can mitigate risk with minimal reward loss. 

**Abstract (ZH)**: 在动态和不确定性环境中生存，个体必须发展有效的决策策略，平衡信息收集与决策承诺。在这样的策略模型中，通常更侧重于优化具体的回报，如奖励率，或者收集信息以支持多样性的（可能未知的）目标。然而，这些两种方法的优势之间的相对 merits 我们的理解仍然不完整，部分原因是直接比较仅限于理想化的静态环境，缺乏现实世界的动态复杂性。我们在一个动态觅食任务中比较了规范的回报寻求和信息寻求策略的表现。这两种策略在环境不确定性变化时都表现出类似的探索性和利用性行为的转变。然而，我们发现它们采取的行动存在微妙的差异，导致有意义的表现差异：尽管回报寻求策略平均获得更多的回报，但信息寻求策略提供更为一致和可预测的结果。我们的研究结果支持了信息寻求行为的适应价值，可以在最小化回报损失的情况下降低风险。 

---
# The Misinterpretable Evidence Conveyed by Arbitrary Codes 

**Title (ZH)**: 任意编码传达的可误解释证据 

**Authors**: Guido Fioretti  

**Link**: [PDF](https://arxiv.org/pdf/2503.18984)  

**Abstract**: Evidence Theory is a mathematical framework for handling imprecise reasoning in the context of a judge evaluating testimonies or a detective evaluating cues, rather than a gambler playing games of chance. In comparison to Probability Theory, it is better equipped to deal with ambiguous information and novel possibilities. Furthermore, arrival and evaluation of testimonies implies a communication channel.
This paper explores the possibility of employing Evidence Theory to represent arbitrary communication codes between and within living organisms. In this paper, different schemes are explored for living organisms incapable of anticipation, animals sufficiently sophisticated to be capable of extrapolation, and humans capable of reading one other's minds. 

**Abstract (ZH)**: 证据理论是一种在法官评估证词或侦探评估线索的背景下处理模棱两可推理的数学框架，而不是赌博者参与机会游戏。与概率理论相比，它更能应对模糊信息和新型可能性。此外，证词的接收和评估意味着存在通信渠道。
本文探讨将证据理论用于表示生物之间及生物内部任意通信码的可能性。本文研究了适用于缺乏预见能力的生物、能够进行外推的足够复杂动物以及能够读取彼此想法的人类的不同方案。 

---
# Advancing Deep Learning through Probability Engineering: A Pragmatic Paradigm for Modern AI 

**Title (ZH)**: 通过概率工程推动深度学习：一种面向现代AI的实际范式 

**Authors**: Jianyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.18958)  

**Abstract**: Recent years have witnessed the rapid progression of deep learning, pushing us closer to the realization of AGI (Artificial General Intelligence). Probabilistic modeling is critical to many of these advancements, which provides a foundational framework for capturing data distributions. However, as the scale and complexity of AI applications grow, traditional probabilistic modeling faces escalating challenges, such as high-dimensional parameter spaces, heterogeneous data sources, and evolving real-world requirements often render classical approaches insufficiently flexible.
This paper proposes a novel concept, Probability Engineering, which treats the already-learned probability distributions within deep learning as engineering artifacts. Rather than merely fitting or inferring distributions, we actively modify and reinforce them to better address the diverse and evolving demands of modern AI. Specifically, Probability Engineering introduces novel techniques and constraints to refine existing probability distributions, improving their robustness, efficiency, adaptability, or trustworthiness.
We showcase this paradigm through a series of applications spanning Bayesian deep learning, Edge AI (including federated learning and knowledge distillation), and Generative AI (such as text-to-image generation with diffusion models and high-quality text generation with large language models). These case studies demonstrate how probability distributions once treated as static objects can be engineered to meet the diverse and evolving requirements of large-scale, data-intensive, and trustworthy AI systems. By systematically expanding and strengthening the role of probabilistic modeling, Probability Engineering paves the way for more robust, adaptive, efficient, and trustworthy deep learning solutions in today's fast-growing AI era. 

**Abstract (ZH)**: 近年来，深度学习取得了 rapid progression，推动我们向通用人工智能（AGI）的实现更近一步。概率模型对这些进展至关重要，为捕捉数据分布提供了基础框架。然而，随着AI应用规模和复杂性的增加，传统概率模型面临越来越大的挑战，如高维度参数空间、异质数据源以及不断变化的现实世界需求， often render 其经典方法不够灵活。

本文提出了一种新的概念——概率工程，将深度学习中已学习的概率分布视为工程制品。我们不仅限于拟合或推断分布，而是积极修改和强化它们，以更好地应对现代AI的多样性和不断变化的需求。具体来说，概率工程引入了新颖的技术和约束，以改进现有概率分布的稳健性、效率、适应性和可信度。

通过一系列应用，包括贝叶斯深度学习、边缘AI（包括联邦学习和知识蒸馏）以及生成AI（如使用扩散模型的文字转图像生成和大语言模型的高质量文字生成），本文展示了如何将原本视为静态对象的概率分布工程化，以满足大规模、数据密集和可信AI系统的需求。通过系统地扩大和加强概率模型的作用，概率工程为当今快速发展的人工智能时代提供了更稳健、更具适应性、更高效和更可信的深度学习解决方案。 

---
# Is there a future for AI without representation? 

**Title (ZH)**: 没有代表性的AI，未来在哪里？ 

**Authors**: Vincent C. Müller  

**Link**: [PDF](https://arxiv.org/pdf/2503.18955)  

**Abstract**: This paper investigates the prospects of AI without representation in general, and the proposals of Rodney Brooks in particular. What turns out to be characteristic of Brooks' proposal is the rejection of central control in intelligent agents; his systems has as much or as little representation as traditional AI. The traditional view that representation is necessary for intelligence presupposes that intelligence requires central control. However, much of recent cognitive science suggests that we should dispose of the image of intelligent agents as central representation processors. If this paradigm shift is achieved, Brooks' proposal for non-centralized cognition without representation appears promising for full-blown intelligent agents - though not for conscious agents and thus not for human-like AI. 

**Abstract (ZH)**: 本文探讨了无表征的AI的前景，尤其是Rodney Brooks的提议。Brooks提议的特征在于拒绝智能代理中的中心控制；他的系统可能具有与传统AI同样量级的表征，或者根本不具备表征。传统观点认为表征是智能的必要条件，这暗示智能需要中心控制。然而，近期认知科学的大量研究表明，我们可能应该摒弃智能代理为中心表征处理器的形象。如果这种范式转变得以实现，Brooks提出的无中心化的无表征认知方案可能对完全智能化代理是有前景的——尽管不是对有意识的代理，因此不是对类人AI。 

---
# A proposal for an incident regime that tracks and counters threats to national security posed by AI systems 

**Title (ZH)**: 一种追踪并抵御由人工智能系统威胁国家安全的事件机制提案 

**Authors**: Alejandro Ortega  

**Link**: [PDF](https://arxiv.org/pdf/2503.19887)  

**Abstract**: Recent progress in AI capabilities has heightened concerns that AI systems could pose a threat to national security, for example, by making it easier for malicious actors to perform cyberattacks on critical national infrastructure, or through loss of control of autonomous AI systems. In parallel, federal legislators in the US have proposed nascent 'AI incident regimes' to identify and counter similar threats. In this paper, we consolidate these two trends and present a proposal for a legally mandated post-deployment AI incident regie that aims to counter potential national security threats from AI systems. We start the paper by introducing the concept of 'security-critical' to describe doctors that pose extreme risks to national security, before arguing that 'security-critical' describes civilian nuclear power, aviation, life science dual-use research of concern, and frontier AI development. We then present in detail our AI incident regime proposal,, justifying each component of the proposal by demonstrating its similarity to US domestic incident regimes in other 'security-critical' sectors. Finally, we sketch a hypothetical scenario where our proposed AI incident regime deals with an AI cyber incident. Our proposed AI incident regime is split into three phases. The first phase revolves around a novel operationalization of what counts as an 'AI incident' and we suggest that AI providers must create a 'national security case' before deploying a frontier AI system. The second and third phases spell out that AI providers should notify a government agency about incidents, and that the government agency should be involved in amending AI providers' security and safety procedures, in order to counter future threats to national security. Our proposal is timely, given ongoing policy interest in the potential national security threats posed by AI systems. 

**Abstract (ZH)**: 近期人工智能能力的进展加剧了对人工智能系统可能对国家安全构成威胁的担忧，例如通过使恶意行为者更容易对关键基础设施进行网络攻击，或通过失去对自主人工智能系统的控制。与此同时，美国联邦立法者提出了初步的“人工智能事件制度”来识别和应对类似威胁。本文整合了这两种趋势，并提出了一项旨在应对人工智能系统潜在国家安全威胁的法律强制执行型部署后人工智能事件制度的提案。本文首先引入“安全关键型”的概念来描述对国家安全构成极端风险的医生，然后 argument 说明“安全关键型”也适用于民用核能、航空、生物双用途研究以及前沿人工智能开发。然后，我们详细阐述了我们的人工智能事件制度提案，并通过证明其与美国其他“安全关键型”领域内事件制度相似之处来为每个提案组件提供正当性。最后，我们描绘了一个假设场景，说明我们提议的人工智能事件制度如何处理人工智能网络攻击事件。我们的提议被分成三个阶段。第一阶段围绕着对“人工智能事件”的新颖定义展开，我们建议人工智能提供商在部署前沿人工智能系统前必须创建一个“国家安全案例”。第二和第三阶段规定人工智能提供商必须向政府部门报告事件，并要求政府部门参与修改人工智能提供商的安全和安全程序，以应对未来的国家安全威胁。鉴于对人工智能系统潜在国家安全威胁的持续政策兴趣，我们的提议非常及时。 

---
# Dynamics of Structured Complex-Valued Hopfield Neural Networks 

**Title (ZH)**: 结构化复值霍普菲尔德神经网络的动力学研究 

**Authors**: Rama Murthy Garimella, Marcos Eduardo Valle, Guilherme Vieira, Anil Rayala, Dileep Munugoti  

**Link**: [PDF](https://arxiv.org/pdf/2503.19885)  

**Abstract**: In this paper, we explore the dynamics of structured complex-valued Hopfield neural networks (CvHNNs), which arise when the synaptic weight matrix possesses specific structural properties. We begin by analyzing CvHNNs with a Hermitian synaptic weight matrix and establish the existence of four-cycle dynamics in CvHNNs with skew-Hermitian weight matrices operating synchronously. Furthermore, we introduce two new classes of complex-valued matrices: braided Hermitian and braided skew-Hermitian matrices. We demonstrate that CvHNNs utilizing these matrix types exhibit cycles of length eight when operating in full parallel update mode. Finally, we conduct extensive computational experiments on synchronous CvHNNs, exploring other synaptic weight matrix structures. The findings provide a comprehensive overview of the dynamics of structured CvHNNs, offering insights that may contribute to developing improved associative memory models when integrated with suitable learning rules. 

**Abstract (ZH)**: 本文探讨了结构化复值霍普菲尔德神经网络（CvHNNs）的动力学，这些网络在突触权重矩阵具有特定结构属性时产生。我们首先分析了具有 Hermitian 突触权重矩阵的 CvHNNs，并建立了具有 skew-Hermitian 权重矩阵且同步操作的 CvHNNs 中四循环动力学的存在性。此外，我们引入了两种新的复值矩阵类别：编braided Hermitian 和编braided skew-Hermitian 矩阵。我们证明了这些矩阵类型在全并行更新模式下操作的 CvHNNs 展现出长度为八的循环。最后，我们对同步 CvHNNs 进行了大量计算实验，探索其他突触权重矩阵结构。研究发现提供了结构化 CvHNNs 动力学的全面概述，所提供的见解可能有助于结合适当的联想记忆模型时的发展改进。 

---
# Geometric Meta-Learning via Coupled Ricci Flow: Unifying Knowledge Representation and Quantum Entanglement 

**Title (ZH)**: 几何元学习通过耦合里奇流：知识表示与量子纠缠的统一 

**Authors**: Ming Lei, Christophe Baehr  

**Link**: [PDF](https://arxiv.org/pdf/2503.19867)  

**Abstract**: This paper establishes a unified framework integrating geometric flows with deep learning through three fundamental innovations. First, we propose a thermodynamically coupled Ricci flow that dynamically adapts parameter space geometry to loss landscape topology, formally proved to preserve isometric knowledge embedding (Theorem~\ref{thm:isometric}). Second, we derive explicit phase transition thresholds and critical learning rates (Theorem~\ref{thm:critical}) through curvature blowup analysis, enabling automated singularity resolution via geometric surgery (Lemma~\ref{lem:surgery}). Third, we establish an AdS/CFT-type holographic duality (Theorem~\ref{thm:ads}) between neural networks and conformal field theories, providing entanglement entropy bounds for regularization design. Experiments demonstrate 2.1$\times$ convergence acceleration and 63\% topological simplification while maintaining $\mathcal{O}(N\log N)$ complexity, outperforming Riemannian baselines by 15.2\% in few-shot accuracy. Theoretically, we prove exponential stability (Theorem~\ref{thm:converge}) through a new Lyapunov function combining Perelman entropy with Wasserstein gradient flows, fundamentally advancing geometric deep learning. 

**Abstract (ZH)**: 本文通过三项基本原则创新，建立了将几何流与深度学习统一起来的框架：首先，提出了一种热力学耦合 Ricci 流，动态适应参数空间几何以匹配损失景观拓扑，并形式上证明了保持等距知识嵌入（定理~\ref{thm:isometric}）。其次，通过曲率爆炸分析推导出明确的相转变阈值和关键学习率（定理~\ref{thm:critical}），使几何外科手术能够自动解决奇异性问题。第三，建立了神经网络与共形场理论之间的 AdS/CFT 类型全息对偶性（定理~\ref{thm:ads}），为正则化设计提供了纠缠熵界。实验结果表明，该方法在保持 $\mathcal{O}(N\log N)$ 复杂度的情况下实现了 2.1 倍的收敛加速和 63% 的拓扑简化，且准确率超越黎曼基线 15.2%。理论上，我们通过将 Perelman 能量与 Wasserstein 梯度流结合的新 Lyapunov 函数证明了指数稳定（定理~\ref{thm:converge}），从根本上推进了几何深度学习。 

---
# Guarding against artificial intelligence--hallucinated citations: the case for full-text reference deposit 

**Title (ZH)**: 防范人工智能虚构引文：全文参考文献存档的必要性 

**Authors**: Alex Glynn  

**Link**: [PDF](https://arxiv.org/pdf/2503.19848)  

**Abstract**: The tendency of generative artificial intelligence (AI) systems to "hallucinate" false information is well-known; AI-generated citations to non-existent sources have made their way into the reference lists of peer-reviewed publications. Here, I propose a solution to this problem, taking inspiration from the Transparency and Openness Promotion (TOP) data sharing guidelines, the clash of generative AI with the American judiciary, and the precedent set by submissions of prior art to the United States Patent and Trademark Office. Journals should require authors to submit the full text of each cited source along with their manuscripts, thereby preventing authors from citing any material whose full text they cannot produce. This solution requires limited additional work on the part of authors or editors while effectively immunizing journals against hallucinated references. 

**Abstract (ZH)**: 生成式人工智能系统生成虚假信息的倾向是众所周知的；人工智能生成的引用非-existent来源的文献已进入经过同行评审的出版物的参考文献列表。在此，我提出一种解决这一问题的方法，借鉴了透明度和开放性促进（TOP）数据共享指导原则、生成式人工智能与美国司法体系的冲突以及向美国专利和商标局提交现有技术的先例。期刊应要求作者在提交稿件时一并提交每个引用来源的全文，从而防止作者引用他们无法提供全文的材料。该解决方案仅要求作者或编辑进行少量额外工作，同时有效保护期刊免受虚假参考文献的影响。 

---
# GyralNet Subnetwork Partitioning via Differentiable Spectral Modularity Optimization 

**Title (ZH)**: GyralNet 带有可微谱模ularity优化的皮层网络子网络划分 

**Authors**: Yan Zhuang, Minheng Chen, Chao Cao, Tong Chen, Jing Zhang, Xiaowei Yu, Yanjun Lyu, Lu Zhang, Tianming Liu, Dajiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.19823)  

**Abstract**: Understanding the structural and functional organization of the human brain requires a detailed examination of cortical folding patterns, among which the three-hinge gyrus (3HG) has been identified as a key structural landmark. GyralNet, a network representation of cortical folding, models 3HGs as nodes and gyral crests as edges, highlighting their role as critical hubs in cortico-cortical connectivity. However, existing methods for analyzing 3HGs face significant challenges, including the sub-voxel scale of 3HGs at typical neuroimaging resolutions, the computational complexity of establishing cross-subject correspondences, and the oversimplification of treating 3HGs as independent nodes without considering their community-level relationships. To address these limitations, we propose a fully differentiable subnetwork partitioning framework that employs a spectral modularity maximization optimization strategy to modularize the organization of 3HGs within GyralNet. By incorporating topological structural similarity and DTI-derived connectivity patterns as attribute features, our approach provides a biologically meaningful representation of cortical organization. Extensive experiments on the Human Connectome Project (HCP) dataset demonstrate that our method effectively partitions GyralNet at the individual level while preserving the community-level consistency of 3HGs across subjects, offering a robust foundation for understanding brain connectivity. 

**Abstract (ZH)**: 理解人类大脑的结构和功能组织需要对皮层折叠模式进行详细的检查，其中三铰褶回（3HG）已被识别为关键的结构标志。GyralNet 是一种皮层折叠的网络表示方法，将 3HGs 表示为节点，将皮层褶皱脊表示为边，并突显其在皮层-皮层连接中的关键枢纽作用。然而，现有的 3HGs 分析方法面临着显著挑战，包括在典型神经影像学分辨率下的亚体素尺度、跨个体对应关系的计算复杂性以及将 3HGs 简单视为独立节点而不考虑其社区层级关系的过度简化。为了解决这些限制，我们提出了一种完全可微的子网络分割框架，该框架采用谱模块最大化优化策略来模块化 GyralNet 中 3HGs 的组织。通过结合拓扑结构相似性和来自扩散张量成像的连接模式作为属性特征，我们的方法为皮层组织提供了生物学意义的表示。在人类连接组计划（HCP）数据集上的广泛实验表明，我们的方法有效地在个体水平上分割了 GyralNet，同时在不同个体之间保持了 3HGs 的社区层级一致性，为理解大脑连接提供了一个稳健的基础。 

---
# Bitstream Collisions in Neural Image Compression via Adversarial Perturbations 

**Title (ZH)**: 基于 adversarial 永整的神经图像压缩中的比特流碰撞 

**Authors**: Jordan Madden, Lhamo Dorje, Xiaohua Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.19817)  

**Abstract**: Neural image compression (NIC) has emerged as a promising alternative to classical compression techniques, offering improved compression ratios. Despite its progress towards standardization and practical deployment, there has been minimal exploration into it's robustness and security. This study reveals an unexpected vulnerability in NIC - bitstream collisions - where semantically different images produce identical compressed bitstreams. Utilizing a novel whitebox adversarial attack algorithm, this paper demonstrates that adding carefully crafted perturbations to semantically different images can cause their compressed bitstreams to collide exactly. The collision vulnerability poses a threat to the practical usability of NIC, particularly in security-critical applications. The cause of the collision is analyzed, and a simple yet effective mitigation method is presented. 

**Abstract (ZH)**: 神经图像压缩的比特流碰撞脆弱性：一种新型白盒 adversarial 攻击揭示的意外漏洞 

---
# LENVIZ: A High-Resolution Low-Exposure Night Vision Benchmark Dataset 

**Title (ZH)**: LENVIZ: 一种高分辨率低曝光夜视基准数据集 

**Authors**: Manjushree Aithal, Rosaura G. VidalMata, Manikandtan Kartha, Gong Chen, Eashan Adhikarla, Lucas N. Kirsten, Zhicheng Fu, Nikhil A. Madhusudhana, Joe Nasti  

**Link**: [PDF](https://arxiv.org/pdf/2503.19804)  

**Abstract**: Low-light image enhancement is crucial for a myriad of applications, from night vision and surveillance, to autonomous driving. However, due to the inherent limitations that come in hand with capturing images in low-illumination environments, the task of enhancing such scenes still presents a formidable challenge. To advance research in this field, we introduce our Low Exposure Night Vision (LENVIZ) Dataset, a comprehensive multi-exposure benchmark dataset for low-light image enhancement comprising of over 230K frames showcasing 24K real-world indoor and outdoor, with-and without human, scenes. Captured using 3 different camera sensors, LENVIZ offers a wide range of lighting conditions, noise levels, and scene complexities, making it the largest publicly available up-to 4K resolution benchmark in the field. LENVIZ includes high quality human-generated ground truth, for which each multi-exposure low-light scene has been meticulously curated and edited by expert photographers to ensure optimal image quality. Furthermore, we also conduct a comprehensive analysis of current state-of-the-art low-light image enhancement techniques on our dataset and highlight potential areas of improvement. 

**Abstract (ZH)**: 低光照图像增强对于夜间视觉和监视等多种应用至关重要，从夜视和监控到自动驾驶。然而，由于在低光照环境中捕捉图像固有的局限性，增强这类场景依然面临巨大挑战。为推动该领域研究，我们介绍了Low Exposure Night Vision (LENVIZ) 数据集，这是一个包含超过230,000帧、涵盖24,000个真实室内和室外场景（有人或无人）的全面多曝光基准数据集。LENVIZ采用3种不同的相机传感器捕捉，涵盖了广泛的光照条件、噪声水平和场景复杂性，使之成为迄今为止最大的公共可用4K分辨率基准数据集。LENVIZ包括高质量的人工生成的真实 ground truth，每个多曝光低光照场景均由专家摄影师精心策划和编辑，以确保最佳图像质量。此外，我们在该数据集上对当前最先进的低光照图像增强技术进行了全面分析，并指出了改进的潜力。 

---
# On What Depends the Robustness of Multi-source Models to Missing Data in Earth Observation? 

**Title (ZH)**: 多源模型在地球观测中对缺失数据的鲁棒性依赖于什么？ 

**Authors**: Francisco Mena, Diego Arenas, Miro Miranda, Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2503.19719)  

**Abstract**: In recent years, the development of robust multi-source models has emerged in the Earth Observation (EO) field. These are models that leverage data from diverse sources to improve predictive accuracy when there is missing data. Despite these advancements, the factors influencing the varying effectiveness of such models remain poorly understood. In this study, we evaluate the predictive performance of six state-of-the-art multi-source models in predicting scenarios where either a single data source is missing or only a single source is available. Our analysis reveals that the efficacy of these models is intricately tied to the nature of the task, the complementarity among data sources, and the model design. Surprisingly, we observe instances where the removal of certain data sources leads to improved predictive performance, challenging the assumption that incorporating all available data is always beneficial. These findings prompt critical reflections on model complexity and the necessity of all collected data sources, potentially shaping the way for more streamlined approaches in EO applications. 

**Abstract (ZH)**: 近年来，地球观测（EO）领域涌现出了一类鲁棒的多源模型。这些模型利用多来源数据来提高在数据缺失情况下的预测准确性。尽管取得了这些进展，但影响这类模型不同有效性的因素依然知之甚少。在本研究中，我们评估了六种最先进的多源模型在单个数据源缺失或仅有一个数据源可用条件下的预测性能。我们的分析表明，这些模型的有效性紧密依赖于任务特性、数据源间的互补性和模型设计。令人惊讶的是，我们发现去除某些数据源会提高预测性能，这一现象挑战了将所有可用数据纳入模型总是有益的看法。这些发现促使我们对模型复杂性和所有收集数据源的必要性进行深入反思，可能引导地球观测（EO）应用领域的更简洁方法。 

---
# Invertible Koopman neural operator for data-driven modeling of partial differential equations 

**Title (ZH)**: 基于数据驱动的偏微分方程建模的可逆考夫曼神经算子 

**Authors**: Yuhong Jin, Andong Cong, Lei Hou, Qiang Gao, Xiangdong Ge, Chonglong Zhu, Yongzhi Feng, Jun Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.19717)  

**Abstract**: Koopman operator theory is a popular candidate for data-driven modeling because it provides a global linearization representation for nonlinear dynamical systems. However, existing Koopman operator-based methods suffer from shortcomings in constructing the well-behaved observable function and its inverse and are inefficient enough when dealing with partial differential equations (PDEs). To address these issues, this paper proposes the Invertible Koopman Neural Operator (IKNO), a novel data-driven modeling approach inspired by the Koopman operator theory and neural operator. IKNO leverages an Invertible Neural Network to parameterize observable function and its inverse simultaneously under the same learnable parameters, explicitly guaranteeing the reconstruction relation, thus eliminating the dependency on the reconstruction loss, which is an essential improvement over the original Koopman Neural Operator (KNO). The structured linear matrix inspired by the Koopman operator theory is parameterized to learn the evolution of observables' low-frequency modes in the frequency space rather than directly in the observable space, sustaining IKNO is resolution-invariant like other neural operators. Moreover, with preprocessing such as interpolation and dimension expansion, IKNO can be extended to operator learning tasks defined on non-Cartesian domains. We fully support the above claims based on rich numerical and real-world examples and demonstrate the effectiveness of IKNO and superiority over other neural operators. 

**Abstract (ZH)**: 可逆Koopman神经算子（IKNO）：一种基于Koopman算子理论和神经算子的新型数据驱动建模方法 

---
# Deep Learning for Speech Emotion Recognition: A CNN Approach Utilizing Mel Spectrograms 

**Title (ZH)**: 基于CNN利用梅尔谱图的语音情感识别深度学习方法 

**Authors**: Niketa Penumajji  

**Link**: [PDF](https://arxiv.org/pdf/2503.19677)  

**Abstract**: This paper explores the application of Convolutional Neural Networks CNNs for classifying emotions in speech through Mel Spectrogram representations of audio files. Traditional methods such as Gaussian Mixture Models and Hidden Markov Models have proven insufficient for practical deployment, prompting a shift towards deep learning techniques. By transforming audio data into a visual format, the CNN model autonomously learns to identify intricate patterns, enhancing classification accuracy. The developed model is integrated into a user-friendly graphical interface, facilitating realtime predictions and potential applications in educational environments. The study aims to advance the understanding of deep learning in speech emotion recognition, assess the models feasibility, and contribute to the integration of technology in learning contexts 

**Abstract (ZH)**: 本文探讨了通过梅尔频谱图表示的音频文件应用卷积神经网络CNN进行语音情感分类的应用。传统的高斯混合模型和隐马尔可夫模型已被证明不适合实际部署，推动了向深度学习技术的转变。通过将音频数据转换为可视化格式，CNN模型自主学习识别复杂的模式，从而提高分类准确性。开发的模型被集成到一个用户友好的图形界面中，便于实时预测，并有可能在教育环境中应用。本研究旨在推进对深度学习在语音情感识别中的理解，评估模型的可行性，并促进技术在学习环境中的集成。 

---
# BiblioPage: A Dataset of Scanned Title Pages for Bibliographic Metadata Extraction 

**Title (ZH)**: BiblioPage: 一本涵盖扫描标题页的语料库，用于提取文献元数据 

**Authors**: Jan Kohút, Martin Dočekal, Michal Hradiš, Marek Vaško  

**Link**: [PDF](https://arxiv.org/pdf/2503.19658)  

**Abstract**: Manual digitization of bibliographic metadata is time consuming and labor intensive, especially for historical and real-world archives with highly variable formatting across documents. Despite advances in machine learning, the absence of dedicated datasets for metadata extraction hinders automation. To address this gap, we introduce BiblioPage, a dataset of scanned title pages annotated with structured bibliographic metadata. The dataset consists of approximately 2,000 monograph title pages collected from 14 Czech libraries, spanning a wide range of publication periods, typographic styles, and layout structures. Each title page is annotated with 16 bibliographic attributes, including title, contributors, and publication metadata, along with precise positional information in the form of bounding boxes. To extract structured information from this dataset, we valuated object detection models such as YOLO and DETR combined with transformer-based OCR, achieving a maximum mAP of 52 and an F1 score of 59. Additionally, we assess the performance of various visual large language models, including LlamA 3.2-Vision and GPT-4o, with the best model reaching an F1 score of 67. BiblioPage serves as a real-world benchmark for bibliographic metadata extraction, contributing to document understanding, document question answering, and document information extraction. Dataset and evaluation scripts are availible at: this https URL 

**Abstract (ZH)**: 手动数字化 bibliographic 元数据耗时且劳动密集，尤其是在具有高度变体格式的历史和现实档案中。尽管机器学习取得了进展，但由于缺乏专门的元数据提取数据集，自动化受到了阻碍。为解决这一问题，我们引入了 BiblioPage，一个包含扫描标题页并注释有结构化 bibliographic 元数据的数据集。该数据集包含来自 14 个捷克图书馆的约 2,000 个单行标题页，涵盖了广泛的不同出版时期、版式风格和布局结构。每个标题页都注释有 16 个 bibliographic 属性，包括书名、作者和其他出版元数据，以及以边界框形式的精确位置信息。为了从这个数据集中提取结构化信息，我们评估了YOLO和DETR对象检测模型与基于变换器的OCR相结合的方法，实现了最高mAP 52和F1分数 59。此外，我们评估了各种视觉大型语言模型，包括LlamA 3.2-Vision和GPT-4o，最佳模型的F1分数达到67。BiblioPage作为 bibliographic 元数据提取的实际 benchmarks，有助于文档理解、文档问答和文档信息提取。数据集和评估脚本可在以下链接获取：this https URL。 

---
# Towards Reliable Time Series Forecasting under Future Uncertainty: Ambiguity and Novelty Rejection Mechanisms 

**Title (ZH)**: 面向未来不确定性的时间序列预测可靠性：含模糊性和新颖性拒绝机制 

**Authors**: Ninghui Feng, Songning Lai, Xin Zhou, Jiayu Yang, Kunlong Feng, Zhenxiao Yin, Fobao Zhou, Zhangyi Hu, Yutao Yue, Yuxuan Liang, Boyu Wang, Hang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.19656)  

**Abstract**: In real-world time series forecasting, uncertainty and lack of reliable evaluation pose significant challenges. Notably, forecasting errors often arise from underfitting in-distribution data and failing to handle out-of-distribution inputs. To enhance model reliability, we introduce a dual rejection mechanism combining ambiguity and novelty rejection. Ambiguity rejection, using prediction error variance, allows the model to abstain under low confidence, assessed through historical error variance analysis without future ground truth. Novelty rejection, employing Variational Autoencoders and Mahalanobis distance, detects deviations from training data. This dual approach improves forecasting reliability in dynamic environments by reducing errors and adapting to data changes, advancing reliability in complex scenarios. 

**Abstract (ZH)**: 在实际时间序列预测中，不确定性与可靠评价的缺乏提出重大挑战。值得注意的是，预测误差通常源自于分布内数据的拟合不足以及无法处理分布外输入。为提升模型可靠性，我们引入一种结合模糊性和新颖性拒绝的双重拒绝机制。模糊性拒绝利用预测误差方差，使模型在低置信度下避免做出预测，通过历史误差方差分析而不依赖未来真实值。新颖性拒绝利用变分自编码器和马氏距离检测训练数据之外的偏差。这种双重方法通过减少错误和适应数据变化，在动态环境中提高预测可靠性，推进复杂场景下的可靠性提升。 

---
# Recover from Horcrux: A Spectrogram Augmentation Method for Cardiac Feature Monitoring from Radar Signal Components 

**Title (ZH)**: 从 horcrux 恢复：一种基于雷达信号成分的声谱图增强方法用于心脏特征监测 

**Authors**: Yuanyuan Zhang, Sijie Xiong, Rui Yang, EngGee Lim, Yutao Yue  

**Link**: [PDF](https://arxiv.org/pdf/2503.19649)  

**Abstract**: Radar-based wellness monitoring is becoming an effective measurement to provide accurate vital signs in a contactless manner, but data scarcity retards the related research on deep-learning-based methods. Data augmentation is commonly used to enrich the dataset by modifying the existing data, but most augmentation techniques can only couple with classification tasks. To enable the augmentation for regression tasks, this research proposes a spectrogram augmentation method, Horcrux, for radar-based cardiac feature monitoring (e.g., heartbeat detection, electrocardiogram reconstruction) with both classification and regression tasks involved. The proposed method is designed to increase the diversity of input samples while the augmented spectrogram is still faithful to the original ground truth vital sign. In addition, Horcrux proposes to inject zero values in specific areas to enhance the awareness of the deep learning model on subtle cardiac features, improving the performance for the limited dataset. Experimental result shows that Horcrux achieves an overall improvement of 16.20% in cardiac monitoring and has the potential to be extended to other spectrogram-based tasks. The code will be released upon publication. 

**Abstract (ZH)**: 基于雷达的健康监测正成为一种有效的非接触式方法，用于提供准确的生命体征测量，但数据稀缺性限制了相关深度学习方法的研究。数据增强常用来通过修改现有数据来丰富数据集，但大多数增强技术只能与分类任务结合使用。为了使增强技术能够应用于回归任务，本研究提出了一种名为Horcrux的谱图增强方法，用于涉及分类和回归任务的雷达基心脏特征监测（如心搏检测、心电图重建）。所提出的方法旨在增加输入样本的多样性，同时增强的谱图仍忠于原始的真实生命体征。此外，Horcrux方法提出在特定区域注入零值，以增强深度学习模型对细微心脏特征的意识，从而改善有限数据集上的性能。实验结果表明，Horcrux在心脏监测上总体提高了16.20%，并有可能扩展到其他谱图基任务中。代码将在发布时公开。 

---
# Show or Tell? Effectively prompting Vision-Language Models for semantic segmentation 

**Title (ZH)**: 说还是做？有效提示视觉语言模型进行语义分割 

**Authors**: Niccolo Avogaro, Thomas Frick, Mattia Rigotti, Andrea Bartezzaghi, Filip Janicki, Cristiano Malossi, Konrad Schindler, Roy Assaf  

**Link**: [PDF](https://arxiv.org/pdf/2503.19647)  

**Abstract**: Large Vision-Language Models (VLMs) are increasingly being regarded as foundation models that can be instructed to solve diverse tasks by prompting, without task-specific training. We examine the seemingly obvious question: how to effectively prompt VLMs for semantic segmentation. To that end, we systematically evaluate the segmentation performance of several recent models guided by either text or visual prompts on the out-of-distribution MESS dataset collection. We introduce a scalable prompting scheme, few-shot prompted semantic segmentation, inspired by open-vocabulary segmentation and few-shot learning. It turns out that VLMs lag far behind specialist models trained for a specific segmentation task, by about 30% on average on the Intersection-over-Union metric. Moreover, we find that text prompts and visual prompts are complementary: each one of the two modes fails on many examples that the other one can solve. Our analysis suggests that being able to anticipate the most effective prompt modality can lead to a 11% improvement in performance. Motivated by our findings, we propose PromptMatcher, a remarkably simple training-free baseline that combines both text and visual prompts, achieving state-of-the-art results outperforming the best text-prompted VLM by 2.5%, and the top visual-prompted VLM by 3.5% on few-shot prompted semantic segmentation. 

**Abstract (ZH)**: 大规模视觉-语言模型（VLMs） increasingly被视为可以通过提示进行指令以解决多样任务的基础模型，而不需要针对特定任务进行训练。我们系统地评估了几个 recent 模型在 out-of-distribution MESS 数据集集合中的分割性能，这些模型要么由文本提示引导，要么由视觉提示引导。我们提出了一种可扩展的提示方案，少量 Shot 提示语义分割，受到开源词汇分割和少量 Shot 学习的启发。结果表明，VLMs 在交并比（Intersection-over-Union，IoU）度量标准上的表现比为特定分割任务训练的专业模型落后约 30%。此外，我们发现文本提示和视觉提示是互补的：两种模式中的每一种都会在其他模式可以解决的许多例子上失败。我们的分析表明，能够预见最有效的提示模式可以提高 11% 的性能。受我们发现的启发，我们提出了 PromptMatcher，这一极其简单的无训练基础模型结合了文本和视觉提示，在少量 Shot 提示语义分割中达到了最新成果，分别优于最佳文本提示的 VLM 和顶级视觉提示的 VLM 2.5% 和 3.5%。 

---
# Enabling Rapid Shared Human-AI Mental Model Alignment via the After-Action Review 

**Title (ZH)**: 通过行动回顾实现快速共享人机思维模型对齐 

**Authors**: Edward Gu, Ho Chit Siu, Melanie Platt, Isabelle Hurley, Jaime Peña, Rohan Paleja  

**Link**: [PDF](https://arxiv.org/pdf/2503.19607)  

**Abstract**: In this work, we present two novel contributions toward improving research in human-machine teaming (HMT): 1) a Minecraft testbed to accelerate testing and deployment of collaborative AI agents and 2) a tool to allow users to revisit and analyze behaviors within an HMT episode to facilitate shared mental model development. Our browser-based Minecraft testbed allows for rapid testing of collaborative agents in a continuous-space, real-time, partially-observable environment with real humans without cumbersome setup typical to human-AI interaction user studies. As Minecraft has an extensive player base and a rich ecosystem of pre-built AI agents, we hope this contribution can help to facilitate research quickly in the design of new collaborative agents and in understanding different human factors within HMT. Our mental model alignment tool facilitates user-led post-mission analysis by including video displays of first-person perspectives of the team members (i.e., the human and AI) that can be replayed, and a chat interface that leverages GPT-4 to provide answers to various queries regarding the AI's experiences and model details. 

**Abstract (ZH)**: 在本工作中，我们提出了两项关于提升人机团队合作（HMT）研究的新贡献：1) 一个基于浏览器的Minecraft测试床，用于加速协作AI代理的测试和部署；2) 一个工具，允许用户回顾和分析HMT情景中的行为，以促进共享心智模型的发展。我们的浏览器-based Minecraft测试床允许在真实的实时、部分可观测环境中快速测试协作代理，同时避免了传统的人机交互用户研究中繁琐的设置。由于Minecraft拥有广泛的玩家基础和丰富的预构建AI代理生态系统，我们希望此贡献能够加速新协作代理的设计研究，并理解HMT中的不同人因因素。我们的认知模型对齐工具通过提供团队成员（即人类和AI）的第一人称视角视频回放以及利用GPT-4的聊天界面来回答关于AI经历和模型细节的各种查询，从而促进用户主导的后任务分析。 

---
# A Contradiction-Centered Model for the Emergence of Swarm Intelligence 

**Title (ZH)**: 基于矛盾中心的群体智能涌现模型 

**Authors**: Wenpin Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2503.19585)  

**Abstract**: The phenomenon of emergence of swarm intelligence exists widely in nature and human society. People have been exploring the root cause of emergence of swarm intelligence and trying to establish general theories and models for emergence of swarm intelligence. However, the existing theories or models do not grasp the essence of swarm intelligence, so they lack generality and are difficult to explain various phenomena of emergence of swarm intelligence. In this paper, a contradiction-centered model for the emergence of swarm intelligence is proposed, in which the internal contradictions of individuals determine their behavior and properties, individuals are related and interact within the swarm because of competing and occupying environmental resources, interactions and swarm potential affect the internal contradictions of individuals and their distribution in the swarm, and the swarm intelligence is manifested as the specific distribution of individual contradictions. This model completely explains the conditions, dynamics, pathways, formations and processes of the emergence of swarm intelligence. In order to verify the validity of this model, several swarm intelligence systems are implemented and analyzed in this paper. The experimental results show that the model has good generality and can be used to describe the emergence of various swarm intelligence. 

**Abstract (ZH)**: 群体智能涌现现象中心矛盾模型 

---
# VectorFit : Adaptive Singular & Bias Vector Fine-Tuning of Pre-trained Foundation Models 

**Title (ZH)**: VectorFit：自适应奇异矢量与偏差矢量微调的预训练基础模型调优 

**Authors**: Suhas G Hegde, Shilpy Kaur, Aruna Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2503.19530)  

**Abstract**: Popular PEFT methods achieve parameter efficiency by assuming that incremental weight updates are inherently low-rank, which often leads to a performance gap compared to full fine-tuning. While recent methods have attempted to address this limitation, they typically lack sufficient parameter and memory efficiency. We propose VectorFit, an effective and easily deployable approach that adaptively trains the singular vectors and biases of pre-trained weight matrices. We demonstrate that the utilization of structural and transformational characteristics of pre-trained weights enables high-rank updates comparable to those of full fine-tuning. As a result, VectorFit achieves superior performance with 9X less trainable parameters compared to state-of-the-art PEFT methods. Through extensive experiments over 17 datasets spanning diverse language and vision tasks such as natural language understanding and generation, question answering, image classification, and image generation, we exhibit that VectorFit consistently outperforms baselines, even in extremely low-budget scenarios. 

**Abstract (ZH)**: VectorFit：一种有效的可适应训练预训练权重奇异向量和偏置的方法 

---
# Towards Long-Range ENSO Prediction with an Explainable Deep Learning Model 

**Title (ZH)**: 基于可解释深度学习模型的长-range ENSO预测研究 

**Authors**: Qi Chen, Yinghao Cui, Guobin Hong, Karumuri Ashok, Yuchun Pu, Xiaogu Zheng, Xuanze Zhang, Wei Zhong, Peng Zhan, Zhonglei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.19502)  

**Abstract**: El Niño-Southern Oscillation (ENSO) is a prominent mode of interannual climate variability with far-reaching global impacts. Its evolution is governed by intricate air-sea interactions, posing significant challenges for long-term prediction. In this study, we introduce CTEFNet, a multivariate deep learning model that synergizes convolutional neural networks and transformers to enhance ENSO forecasting. By integrating multiple oceanic and atmospheric predictors, CTEFNet extends the effective forecast lead time to 20 months while mitigating the impact of the spring predictability barrier, outperforming both dynamical models and state-of-the-art deep learning approaches. Furthermore, CTEFNet offers physically meaningful and statistically significant insights through gradient-based sensitivity analysis, revealing the key precursor signals that govern ENSO dynamics, which align with well-established theories and reveal new insights about inter-basin interactions among the Pacific, Atlantic, and Indian Oceans. The CTEFNet's superior predictive skill and interpretable sensitivity assessments underscore its potential for advancing climate prediction. Our findings highlight the importance of multivariate coupling in ENSO evolution and demonstrate the promise of deep learning in capturing complex climate dynamics with enhanced interpretability. 

**Abstract (ZH)**: El Niño-Southern Oscillation (ENSO)是一种影响深远的年际气候变异性模式，其演变受复杂的空气-海洋相互作用控制，给长期预测带来重大挑战。在本研究中，我们引入了CTEFNet，这是一种结合卷积神经网络和-transformers的多变量深度学习模型，以增强ENSO预测。通过整合多种海洋和大气预测因子，CTEFNet延长了有效的预报提前期至20个月，减轻了春季可预报性障碍的影响，优于动力模型和现有的深度学习方法。此外，CTEFNet通过基于梯度的敏感性分析提供物理上合理和统计上显著的洞见，揭示了控制ENSO动力学的关键先行信号，这些信号与已建立的理论一致，并揭示了太平洋、大西洋和印度洋之间交互作用的新见解。CTEFNet卓越的预测能力和可解释的敏感性评估证明了其在推进气候预测方面的潜力。我们的研究强调了ENSO演变中多变量耦合的重要性，并展示了深度学习在具有增强解释性的复杂气候动力学建模方面的前景。 

---
# Pose-Based Fall Detection System: Efficient Monitoring on Standard CPUs 

**Title (ZH)**: 基于姿态的跌倒检测系统：高效的标准CPU监测 

**Authors**: Vinayak Mali, Saurabh Jaiswal  

**Link**: [PDF](https://arxiv.org/pdf/2503.19501)  

**Abstract**: Falls among elderly residents in assisted living homes pose significant health risks, often leading to injuries and a decreased quality of life. Current fall detection solutions typically rely on sensor-based systems that require dedicated hardware, or on video-based models that demand high computational resources and GPUs for real-time processing. In contrast, this paper presents a robust fall detection system that does not require any additional sensors or high-powered hardware. The system uses pose estimation techniques, combined with threshold-based analysis and a voting mechanism, to effectively distinguish between fall and non-fall activities. For pose detection, we leverage MediaPipe, a lightweight and efficient framework that enables real-time processing on standard CPUs with minimal computational overhead. By analyzing motion, body position, and key pose points, the system processes pose features with a 20-frame buffer, minimizing false positives and maintaining high accuracy even in real-world settings. This unobtrusive, resource-efficient approach provides a practical solution for enhancing resident safety in old age homes, without the need for expensive sensors or high-end computational resources. 

**Abstract (ZH)**: 入住辅助生活设施的老年人跌倒对其健康构成了显著风险，常导致受伤并降低生活质量。当前的跌倒检测解决方案通常依赖于需要专用硬件的传感器系统，或依赖于需要大量计算资源和GPU进行实时处理的视频模型。与此相反，本文提出了一种无需额外传感器或高性能硬件的稳健跌倒检测系统。该系统结合了姿态估计技术、阈值分析和投票机制，有效地区分跌倒和非跌倒活动。在姿态检测方面，我们利用MediaPipe这一轻量级且高效的框架，能够在标准CPU上进行实时处理，并且具有最小的计算开销。通过分析运动、身体位置和关键姿态点，系统使用20帧缓冲区处理姿态特征，减少了误报，即使在实际环境中也能保持高准确性。这一不显眼、资源高效的方案为提高养老设施居民的安全性提供了实用解决方法，无需昂贵的传感器或高性能计算资源。 

---
# SMT-EX: An Explainable Surrogate Modeling Toolbox for Mixed-Variables Design Exploration 

**Title (ZH)**: SMT-EX：一种用于混合变量设计探索的可解释代理模型工具箱 

**Authors**: Mohammad Daffa Robani, Paul Saves, Pramudita Satria Palar, Lavi Rizki Zuhal, oseph Morlier  

**Link**: [PDF](https://arxiv.org/pdf/2503.19496)  

**Abstract**: Surrogate models are of high interest for many engineering applications, serving as cheap-to-evaluate time-efficient approximations of black-box functions to help engineers and practitioners make decisions and understand complex systems. As such, the need for explainability methods is rising and many studies have been performed to facilitate knowledge discovery from surrogate models. To respond to these enquiries, this paper introduces SMT-EX, an enhancement of the open-source Python Surrogate Modeling Toolbox (SMT) that integrates explainability techniques into a state-of-the-art surrogate modelling framework. More precisely, SMT-EX includes three key explainability methods: Shapley Additive Explanations, Partial Dependence Plot, and Individual Conditional Expectations. A peculiar explainability dependency of SMT has been developed for such purpose that can be easily activated once the surrogate model is built, offering a user-friendly and efficient tool for swift insight extraction. The effectiveness of SMT-EX is showcased through two test cases. The first case is a 10-variable wing weight problem with purely continuous variables and the second one is a 3-variable mixed-categorical cantilever beam bending problem. Relying on SMT-EX analyses for these problems, we demonstrate its versatility in addressing a diverse range of problem characteristics. SMT-Explainability is freely available on Github: this https URL . 

**Abstract (ZH)**: 代理模型在许多工程应用中备受关注，作为黑箱函数的经济高效的近似模型，有助于工程师和实践者做出决策并理解复杂系统。因此，解释方法的需求日益增加，许多研究已经开展以促进从代理模型中发现知识。为应对这些需求，本文介绍了SMT-EX，这是一种增强的开源Python代理建模工具箱（SMT），其集成了一流的代理建模框架中的解释技术。具体而言，SMT-EX 包括三种关键的解释方法：Shapley 加权解释、部分依赖图和个体条件期望。为此目的而开发了一种特有的解释依赖性，可以在构建代理模型后轻松激活，提供了一个用户友好且高效的工具，用于快速获取洞察。通过两个案例展示了SMT-EX的有效性。第一个案例是具有纯连续变量的10变量机翼重量问题，第二个案例是具有混合分类变量的3变量悬臂梁弯折问题。借助SMT-EX 对这些问题的分析，我们展示了其解决各种问题特征的灵活性。SMT-EX 解释模块可在Github上免费获取：this https URL。 

---
# Quantifying Symptom Causality in Clinical Decision Making: An Exploration Using CausaLM 

**Title (ZH)**: 在临床决策中量化症状因果关系：基于CausaLM的探索 

**Authors**: Mehul Shetty, Connor Jordan  

**Link**: [PDF](https://arxiv.org/pdf/2503.19394)  

**Abstract**: Current machine learning approaches to medical diagnosis often rely on correlational patterns between symptoms and diseases, risking misdiagnoses when symptoms are ambiguous or common across multiple conditions. In this work, we move beyond correlation to investigate the causal influence of key symptoms-specifically "chest pain" on diagnostic predictions. Leveraging the CausaLM framework, we generate counterfactual text representations in which target concepts are effectively "forgotten" enabling a principled estimation of the causal effect of that concept on a model's predicted disease distribution. By employing Textual Representation-based Average Treatment Effect (TReATE), we quantify how the presence or absence of a symptom shapes the model's diagnostic outcomes, and contrast these findings against correlation-based baselines such as CONEXP. Our results offer deeper insight into the decision-making behavior of clinical NLP models and have the potential to inform more trustworthy, interpretable, and causally-grounded decision support tools in medical practice. 

**Abstract (ZH)**: 当前医学诊断中的机器学习方法通常依赖于症状与疾病的关联模式，在症状模糊或跨多种条件普遍时存在误诊风险。本研究超越关联性，探讨关键症状（特别是“胸痛”）对诊断预测的因果影响。利用CausaLM框架生成包含目标概念被有效“遗忘”的反事实文本表示，从而对模型预测疾病分布中的因果效应进行原则性估计。通过使用基于文本表示的平均治疗效果（TReATE）方法，我们量化症状的存在或缺失是如何塑造模型诊断结果的，并将这些发现与基于关联性的基准方法（如CONEXP）进行对比。我们的结果为进一步理解临床NLP模型的决策行为提供了深刻的见解，并有可能为医学实践中更可靠、可解释和基于因果关系的决策支持工具提供指导。 

---
# Causal invariant geographic network representations with feature and structural distribution shifts 

**Title (ZH)**: 因果不变地理网络表示：特征和结构分布转移 

**Authors**: Yuhan Wang, Silu He, Qinyao Luo, Hongyuan Yuan, Ling Zhao, Jiawei Zhu, Haifeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.19382)  

**Abstract**: The existing methods learn geographic network representations through deep graph neural networks (GNNs) based on the i.i.d. assumption. However, the spatial heterogeneity and temporal dynamics of geographic data make the out-of-distribution (OOD) generalisation problem particularly salient. The latter are particularly sensitive to distribution shifts (feature and structural shifts) between testing and training data and are the main causes of the OOD generalisation problem. Spurious correlations are present between invariant and background representations due to selection biases and environmental effects, resulting in the model extremes being more likely to learn background representations. The existing approaches focus on background representation changes that are determined by shifts in the feature distributions of nodes in the training and test data while ignoring changes in the proportional distributions of heterogeneous and homogeneous neighbour nodes, which we refer to as structural distribution shifts. We propose a feature-structure mixed invariant representation learning (FSM-IRL) model that accounts for both feature distribution shifts and structural distribution shifts. To address structural distribution shifts, we introduce a sampling method based on causal attention, encouraging the model to identify nodes possessing strong causal relationships with labels or nodes that are more similar to the target node. Inspired by the Hilbert-Schmidt independence criterion, we implement a reweighting strategy to maximise the orthogonality of the node representations, thereby mitigating the spurious correlations among the node representations and suppressing the learning of background representations. Our experiments demonstrate that FSM-IRL exhibits strong learning capabilities on both geographic and social network datasets in OOD scenarios. 

**Abstract (ZH)**: 基于特征-结构混合不变表示学习的地理空间网络异分布泛化方法 

---
# Efficient IoT Intrusion Detection with an Improved Attention-Based CNN-BiLSTM Architecture 

**Title (ZH)**: 基于改进的注意力机制CNN-BiLSTM架构的高效物联网入侵检测 

**Authors**: Amna Naeem, Muazzam A. Khan, Nada Alasbali, Jawad Ahmad, Aizaz Ahmad Khattak, Muhammad Shahbaz Khan  

**Link**: [PDF](https://arxiv.org/pdf/2503.19339)  

**Abstract**: The ever-increasing security vulnerabilities in the Internet-of-Things (IoT) systems require improved threat detection approaches. This paper presents a compact and efficient approach to detect botnet attacks by employing an integrated approach that consists of traffic pattern analysis, temporal support learning, and focused feature extraction. The proposed attention-based model benefits from a hybrid CNN-BiLSTM architecture and achieves 99% classification accuracy in detecting botnet attacks utilizing the N-BaIoT dataset, while maintaining high precision and recall across various scenarios. The proposed model's performance is further validated by key parameters, such as Mathews Correlation Coefficient and Cohen's kappa Correlation Coefficient. The close-to-ideal results for these parameters demonstrate the proposed model's ability to detect botnet attacks accurately and efficiently in practical settings and on unseen data. The proposed model proved to be a powerful defense mechanism for IoT networks to face emerging security challenges. 

**Abstract (ZH)**: 不断增加的物联网（IoT）系统中的安全漏洞需要改进的威胁检测方法。本文提出了一种紧凑且高效的方法，通过结合流量模式分析、时间支持学习和聚焦特征提取的集成方法来检测botnet攻击。所提出的基于注意力的模型利用了混合CNN-BiLSTM架构，并在使用N-BaIoT数据集检测botnet攻击时实现了99%的分类准确性，同时在各种场景中保持了高精确度和召回率。所提出的模型的性能通过关键参数，如马修斯相关系数和科恩κ相关系数进一步得到验证。这些参数的接近理想的结果展示了所提出模型在实际设置和未见数据中准确且高效地检测botnet攻击的能力。所提出的模型证明是一种强大的防御机制，用于应对物联网网络面临的新兴安全挑战。 

---
# Substance over Style: Evaluating Proactive Conversational Coaching Agents 

**Title (ZH)**: 重实质轻形式：评估主动对话教练代理 

**Authors**: Vidya Srinivas, Xuhai Xu, Xin Liu, Kumar Ayush, Isaac Galatzer-Levy, Shwetak Patel, Daniel McDuff, Tim Althoff  

**Link**: [PDF](https://arxiv.org/pdf/2503.19328)  

**Abstract**: While NLP research has made strides in conversational tasks, many approaches focus on single-turn responses with well-defined objectives or evaluation criteria. In contrast, coaching presents unique challenges with initially undefined goals that evolve through multi-turn interactions, subjective evaluation criteria, mixed-initiative dialogue. In this work, we describe and implement five multi-turn coaching agents that exhibit distinct conversational styles, and evaluate them through a user study, collecting first-person feedback on 155 conversations. We find that users highly value core functionality, and that stylistic components in absence of core components are viewed negatively. By comparing user feedback with third-person evaluations from health experts and an LM, we reveal significant misalignment across evaluation approaches. Our findings provide insights into design and evaluation of conversational coaching agents and contribute toward improving human-centered NLP applications. 

**Abstract (ZH)**: 尽管自然语言处理研究在对话任务上取得了进展，许多方法侧重于具有明确目标或评估标准的单轮响应。相比之下，辅导任务则面临初始目标不明确，通过多轮交互演变，具有主观评估标准和混合主动对话的独特挑战。在本工作中，我们描述并实现五种具有不同对话风格的多轮辅导代理，并通过用户研究对其进行了评估，收集了关于155次对话的first-person反馈。我们发现用户高度认可核心功能，而缺乏核心功能的样式组件则被视为负面。通过将用户反馈与来自健康专家和语言模型的第三人称评估进行比较，我们揭示了评估方法之间的显著不一致。我们的发现为对话辅导代理的设计和评估提供了见解，并有助于改善以人类为中心的自然语言处理应用。 

---
# LRSCLIP: A Vision-Language Foundation Model for Aligning Remote Sensing Image with Longer Text 

**Title (ZH)**: LRSCLIP: 一种用于对齐遥感图像与长文本的视觉-语言基础模型 

**Authors**: Weizhi Chen, Jingbo Chen, Yupeng Deng, Jiansheng Chen, Yuman Feng, Zhihao Xi, Diyou Liu, Kai Li, Yu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2503.19311)  

**Abstract**: This study addresses the technical bottlenecks in handling long text and the "hallucination" issue caused by insufficient short text information in remote sensing vision-language foundation models (VLFM). We propose a novel vision-language foundation model, LRSCLIP, and a multimodal dataset, LRS2M. The main contributions are as follows: (1) By integrating multi-source remote sensing data and adopting a large language model labeling strategy, we construct the LRS2M dataset, which contains 2 million image-text pairs, providing both short and long texts for the first time, thus solving the problem of semantic granularity limitations in existing datasets; (2) The design of the LRSCLIP architecture based on Long-CLIP's KPS module, which extends CLIP's text processing capacity and achieves fine-grained cross-modal feature alignment through a dual-text loss weighting mechanism. Experimental results show that LRSCLIP improves retrieval accuracy by 10\%-20\% over the Long-CLIP baseline in the zero-shot long-text cross-modal retrieval task. For the zero-shot short-text cross-modal retrieval task, LRSCLIP achieves improvements over the current best model, GeoRSCLIP, with increases of 0.17\%, 0.67\%, and 0.92\% in Text to Image R@1, Image to Text R@1, and mR on RSITMD, respectively, and 0.04\%, 2.93\%, and 1.28\% on RSICD. In the zero-shot image classification task (average accuracy=75.75\%) and semantic localization task (Rmi=0.7653), LRSCLIP achieves state-of-the-art performance. These results validate the dual advantages of fine-grained semantic understanding and global feature matching in LRSCLIP. This work provides a new benchmark model and data support for remote sensing multimodal learning. The related code has been open source and is available at this https URL. 

**Abstract (ZH)**: 本研究解决远程 sensing 视觉-语言基础模型（VLFM）在处理长文本和技术瓶颈以及由短文本信息不足引起的“幻觉”问题。我们提出了一种新型的视觉-语言基础模型LRSCLIP，以及一个多模态数据集LRS2M。主要贡献包括：（1）通过集成多源遥感数据并采用大规模语言模型标签策略，构建了包含200万图像-文本对的LRS2M数据集，首次提供短文本和长文本，解决了现有数据集在语义粒度限制方面的问题；（2）基于Long-CLIP的KPS模块设计LRSCLIP架构，扩展了CLIP的文字处理能力，并通过双文本损失加权机制实现精细粒度的跨模态特征对齐。实验结果表明，在零样本长文本跨模态检索任务中，LRSCLIP比Long-CLIP基线提高了10%-20%的检索精度。在零样本短文本跨模态检索任务中，LRSCLIP分别在RSITMD和RSICD上比当前最佳模型GeoRSCLIP在Text to Image R@1、Image to Text R@1和mR上提高了0.17%、0.67%、0.92%和0.04%、2.93%、1.28%。在零样本图像分类任务（平均准确率=75.75%）和语义定位任务（Rmi=0.7653）中，LRSCLIP取得了最先进水平。这些结果验证了LRSCLIP在细粒度语义理解和全球特征匹配方面的双重优势。本研究为遥感多模态学习提供了一个新的基准模型和数据支持。相关代码已开源，可在以下网址获取。 

---
# Adaptive Wavelet Filters as Practical Texture Feature Amplifiers for Parkinson's Disease Screening in OCT 

**Title (ZH)**: 自适应小波滤波器作为实用的纹理特征放大器用于OCT帕金森病筛查 

**Authors**: Xiaoqing Zhang, Hanfeng Shi, Xiangyu Li, Haili Ye, Tao Xu, Na Li, Yan Hu, Fan Lv, Jiangfan Chen, Jiang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.19292)  

**Abstract**: Parkinson's disease (PD) is a prevalent neurodegenerative disorder globally. The eye's retina is an extension of the brain and has great potential in PD screening. Recent studies have suggested that texture features extracted from retinal layers can be adopted as biomarkers for PD diagnosis under optical coherence tomography (OCT) images. Frequency domain learning techniques can enhance the feature representations of deep neural networks (DNNs) by decomposing frequency components involving rich texture features. Additionally, previous works have not exploited texture features for automated PD screening in OCT. Motivated by the above analysis, we propose a novel Adaptive Wavelet Filter (AWF) that serves as the Practical Texture Feature Amplifier to fully leverage the merits of texture features to boost the PD screening performance of DNNs with the aid of frequency domain learning. Specifically, AWF first enhances texture feature representation diversities via channel mixer, then emphasizes informative texture feature representations with the well-designed adaptive wavelet filtering token mixer. By combining the AWFs with the DNN stem, AWFNet is constructed for automated PD screening. Additionally, we introduce a novel Balanced Confidence (BC) Loss by mining the potential of sample-wise predicted probabilities of all classes and class frequency prior, to further boost the PD screening performance and trustworthiness of AWFNet. The extensive experiments manifest the superiority of our AWFNet and BC over state-of-the-art methods in terms of PD screening performance and trustworthiness. 

**Abstract (ZH)**: 帕金森病（PD）是一种全球性的神经退行性疾病。眼睛的视网膜是脑部的延伸，具有在PD筛查中巨大潜力。近期研究表明，可以从光学相干断层扫描（OCT）图像中的视网膜层提取的纹理特征可作为PD诊断的生物标志物。频率域学习技术可以通过分解包含丰富纹理特征的频率成分来增强深度神经网络（DNNs）的特征表示能力。此外，以往的工作尚未利用纹理特征进行OCT自动PD筛查。在上述分析的动机下，我们提出了一种新颖的自适应小波滤波器（AWF），作为实际纹理特征放大器，以充分利用纹理特征的优点，通过频率域学习提升DNNs的PD筛查性能。具体而言，AWF首先通过通道混合器增强纹理特征表示的多样性，然后通过精心设计的自适应小波过滤器令牌混合器强调关键的纹理特征表示。通过将AWFs与DNN主干结合，构建了AWFNet用于自动PD筛查。此外，我们引入了一种新颖的平衡信心（BC）损失，通过挖掘每个类别样本预测概率和类别频率先验的潜力，进一步提升了AWFNet的PD筛查性能和可信度。广泛的实验表明，与最先进的方法相比，我们的AWFNet和BC在PD筛查性能和可信度方面具有优势。 

---
# No Black Box Anymore: Demystifying Clinical Predictive Modeling with Temporal-Feature Cross Attention Mechanism 

**Title (ZH)**: 不再黑箱：时空特征跨注意力机制解析临床预测建模 

**Authors**: Yubo Li, Xinyu Yao, Rema Padman  

**Link**: [PDF](https://arxiv.org/pdf/2503.19285)  

**Abstract**: Despite the outstanding performance of deep learning models in clinical prediction tasks, explainability remains a significant challenge. Inspired by transformer architectures, we introduce the Temporal-Feature Cross Attention Mechanism (TFCAM), a novel deep learning framework designed to capture dynamic interactions among clinical features across time, enhancing both predictive accuracy and interpretability. In an experiment with 1,422 patients with Chronic Kidney Disease, predicting progression to End-Stage Renal Disease, TFCAM outperformed LSTM and RETAIN baselines, achieving an AUROC of 0.95 and an F1-score of 0.69. Beyond performance gains, TFCAM provides multi-level explainability by identifying critical temporal periods, ranking feature importance, and quantifying how features influence each other across time before affecting predictions. Our approach addresses the "black box" limitations of deep learning in healthcare, offering clinicians transparent insights into disease progression mechanisms while maintaining state-of-the-art predictive performance. 

**Abstract (ZH)**: 尽管深度学习模型在临床预测任务中表现出色，但解释性仍然是一个重大挑战。受变压器架构启发，我们引入了时间-特征交叉注意力机制（TFCAM），这是一种新型的深度学习框架，旨在捕捉临床特征随时间动态交互，同时提高预测准确性和解释性。在涉及1,422例慢性肾病患者的试验中，预测进展至终末期肾病，TFCAM优于LSTM和RETAIN基准模型，AUROC达到0.95，F1-score为0.69。除了性能提升，TFCAM还通过识别关键时间周期、排名特征重要性以及量化特征如何随时间相互影响从而影响预测，提供了多级解释性。我们的方法解决了深度学习在医疗保健中的“黑盒”限制，为临床医生提供透明的疾病进展机制洞察，同时保持了最先进的预测性能。 

---
# LogicLearner: A Tool for the Guided Practice of Propositional Logic Proofs 

**Title (ZH)**: 逻辑学习器：命题逻辑证明引导练习的工具 

**Authors**: Amogh Inamdar, Uzay Macar, Michel Vazirani, Michael Tarnow, Zarina Mustapha, Natalia Dittren, Sam Sadeh, Nakul Verma, Ansaf Salleb-Aouissi  

**Link**: [PDF](https://arxiv.org/pdf/2503.19280)  

**Abstract**: The study of propositional logic -- fundamental to the theory of computing -- is a cornerstone of the undergraduate computer science curriculum. Learning to solve logical proofs requires repeated guided practice, but undergraduate students often lack access to on-demand tutoring in a judgment-free environment. In this work, we highlight the need for guided practice tools in undergraduate mathematics education and outline the desiderata of an effective practice tool. We accordingly develop LogicLearner, a web application for guided logic proof practice. LogicLearner consists of an interface to attempt logic proofs step-by-step and an automated proof solver to generate solutions on the fly, allowing users to request guidance as needed. We pilot LogicLearner as a practice tool in two semesters of an undergraduate discrete mathematics course and receive strongly positive feedback for usability and pedagogical value in student surveys. To the best of our knowledge, LogicLearner is the only learning tool that provides an end-to-end practice environment for logic proofs with immediate, judgment-free feedback. 

**Abstract (ZH)**: 命题逻辑的研究——这是计算理论的基础，在本科计算机科学课程中是基石。学习解决逻辑证明需要反复的指导性练习，但本科生通常缺乏一个无评判环境下的即时辅导资源。在此项工作中，我们强调了指导性练习工具在本科数学教育中的重要性，并概述了有效练习工具的理想特征。我们据此开发了LogicLearner，一个在线逻辑证明指导练习的网络应用。LogicLearner包含一个逐步尝试逻辑证明的界面和一个自动生成解决方案的自动化证明解决工具，允许用户在需要时请求指导。我们在一个本科离散数学课程的两个学期中试点LogicLearner作为练习工具，并在学生调查中收到了关于易用性和教学价值的强烈正面反馈。据我们所知，LogicLearner是唯一一个提供即时无评判逻辑证明练习环境的学习工具。 

---
# Continual Reinforcement Learning for HVAC Systems Control: Integrating Hypernetworks and Transfer Learning 

**Title (ZH)**: 持续强化学习在 HVAC 系统控制中的应用：集成超网络和迁移学习 

**Authors**: Gautham Udayakumar Bekal, Ahmed Ghareeb, Ashish Pujari  

**Link**: [PDF](https://arxiv.org/pdf/2503.19212)  

**Abstract**: Buildings with Heating, Ventilation, and Air Conditioning (HVAC) systems play a crucial role in ensuring indoor comfort and efficiency. While traditionally governed by physics-based models, the emergence of big data has enabled data-driven methods like Deep Reinforcement Learning (DRL). However, Reinforcement Learning (RL)-based techniques often suffer from sample inefficiency and limited generalization, especially across varying HVAC systems. We introduce a model-based reinforcement learning framework that uses a Hypernetwork to continuously learn environment dynamics across tasks with different action spaces. This enables efficient synthetic rollout generation and improved sample usage. Our approach demonstrates strong backward transfer in a continual learning setting after training on a second task, minimal fine-tuning on the first task allows rapid convergence within just 5 episodes and thus outperforming Model Free Reinforcement Learning (MFRL) and effectively mitigating catastrophic forgetting. These findings have significant implications for reducing energy consumption and operational costs in building management, thus supporting global sustainability goals.
Keywords: Deep Reinforcement Learning, HVAC Systems Control, Hypernetworks, Transfer and Continual Learning, Catastrophic Forgetting 

**Abstract (ZH)**: 具有供暖、通风和空调系统的建筑物在确保室内舒适性和效率方面扮演着关键角色。传统上，这类系统受基于物理模型的控制，而大数据的出现使得基于数据驱动的方法，如深度强化学习（DRL）成为可能。然而，基于强化学习（RL）的技术往往存在样本效率低和泛化能力有限的问题，尤其是在不同类型的HVAC系统之间。我们提出了一种基于模型的强化学习框架，利用Hyper网络在具有不同动作空间的任务中持续学习环境动态，这使得合成轨迹生成更加高效并且样本使用更有效。我们的方法在训练于第二个任务后，在连续学习设置中表现出强大的反向迁移能力，仅仅在第一个任务上进行少量微调即可在短短5个episode内实现快速收敛，从而优于基于价值的强化学习（MFRL），并有效防止灾难性遗忘。这些发现对降低建筑管理中的能耗和运营成本具有重要意义，从而支持全球可持续发展目标。

关键词：深度强化学习，HVAC系统控制，Hyper网络，迁移和连续学习，灾难性遗忘。 

---
# Mining-Gym: A Configurable RL Benchmarking Environment for Truck Dispatch Scheduling 

**Title (ZH)**: Mining-Gym：一种用于卡车调度协同的可配置RL基准环境 

**Authors**: Chayan Banerjee, Kien Nguyen, Clinton Fookes  

**Link**: [PDF](https://arxiv.org/pdf/2503.19195)  

**Abstract**: Mining process optimization particularly truck dispatch scheduling is a critical factor in enhancing the efficiency of open pit mining operations However the dynamic and stochastic nature of mining environments characterized by uncertainties such as equipment failures truck maintenance and variable haul cycle times poses significant challenges for traditional optimization methods While Reinforcement Learning RL has shown promise in adaptive decision making for mining logistics its practical deployment requires rigorous evaluation in realistic and customizable simulation environments The lack of standardized benchmarking environments limits fair algorithm comparisons reproducibility and the real world applicability of RL based approaches in open pit mining settings To address this challenge we introduce Mining Gym a configurable open source benchmarking environment designed for training testing and comparing RL algorithms in mining process optimization Built on Discrete Event Simulation DES and seamlessly integrated with the OpenAI Gym interface Mining Gym provides a structured testbed that enables the direct application of advanced RL algorithms from Stable Baselines The framework models key mining specific uncertainties such as equipment failures queue congestion and the stochasticity of mining processes ensuring a realistic and adaptive learning environment Additionally Mining Gym features a graphical user interface GUI for intuitive mine site configuration a comprehensive data logging system a built in KPI dashboard and real time visual representation of the mine site These capabilities facilitate standardized reproducible evaluations across multiple RL strategies and baseline heuristics 

**Abstract (ZH)**: 采矿过程优化特别是卡车调度调度是提高露天矿作业效率的关键因素。然而，采矿环境的动态性和随机性，如设备故障、卡车维护和变动的运载周期时间等特点所带来的不确定性，对传统优化方法构成了重大挑战。尽管强化学习（RL）在采矿物流的自适应决策制定方面显示出 promise，但在实际部署中需要在现实的且可定制的仿真环境中进行严格的评估。缺乏标准化的基准测试环境限制了公平的算法比较、可重复性和基于 RL 的方法在露天矿山中的实际应用。为解决这一挑战，我们介绍了一种可配置的开源基准测试环境 Mining Gym，该环境旨在用于采矿过程优化中的 RL 算法的培训、测试和比较。Mining Gym 以离散事件仿真（DES）为基础，并无缝集成了 OpenAI Gym 接口。该框架模拟了关键的采矿特定不确定性，如设备故障、队列拥堵和采矿过程的随机性，确保了一个现实且适应性的学习环境。此外，Mining Gym 还配备了图形用户界面（GUI）用于直观的矿山配置、全面的数据日志系统、内置的 KPI 仪表盘以及实时可视化采矿现场的功能。这些功能使不同 RL 策略和基线启发式方法的标准、可重复评估变得容易。 

---
# SoK: How Robust is Audio Watermarking in Generative AI models? 

**Title (ZH)**: SoK: 生成式AI模型中的音频数字水印robustness如何？ 

**Authors**: Yizhu Wen, Ashwin Innuganti, Aaron Bien Ramos, Hanqing Guo, Qiben Yan  

**Link**: [PDF](https://arxiv.org/pdf/2503.19176)  

**Abstract**: Audio watermarking is increasingly used to verify the provenance of AI-generated content, enabling applications such as detecting AI-generated speech, protecting music IP, and defending against voice cloning. To be effective, audio watermarks must resist removal attacks that distort signals to evade detection. While many schemes claim robustness, these claims are typically tested in isolation and against a limited set of attacks. A systematic evaluation against diverse removal attacks is lacking, hindering practical deployment. In this paper, we investigate whether recent watermarking schemes that claim robustness can withstand a broad range of removal attacks. First, we introduce a taxonomy covering 22 audio watermarking schemes. Next, we summarize their underlying technologies and potential vulnerabilities. We then present a large-scale empirical study to assess their robustness. To support this, we build an evaluation framework encompassing 22 types of removal attacks (109 configurations) including signal-level, physical-level, and AI-induced distortions. We reproduce 9 watermarking schemes using open-source code, identify 8 new highly effective attacks, and highlight 11 key findings that expose the fundamental limitations of these methods across 3 public datasets. Our results reveal that none of the surveyed schemes can withstand all tested distortions. This evaluation offers a comprehensive view of how current watermarking methods perform under real-world threats. Our demo and code are available at this https URL. 

**Abstract (ZH)**: 音频水印在验证AI生成内容的来源中 increasingly used, 使检测AI生成语音、保护音乐IP和防御语音克隆等应用成为可能。为了有效，音频水印必须抵御那些用于规避检测的信号篡改攻击。尽管许多方案声称具有鲁棒性，但这些声明通常是在孤立的情况下并且仅针对有限的攻击进行测试。缺乏针对多样化去除攻击的系统性评估，阻碍了其实用部署。本文研究最近声称具有鲁棒性的水印方案是否能够抵御广泛范围的去除攻击。首先，我们介绍了涵盖22种音频水印方案的分类系统。接着，我们总结了它们的基础技术和潜在脆弱性。然后，我们进行了一项大规模的实证研究来评估它们的鲁棒性。为此，我们构建了一个包含22种去除攻击类型（109种配置）的评估框架，包括信号级、物理级和AI诱导的失真。我们使用开源代码复现了9种水印方案，发现了8种新的高效攻击，并突出了11个关键发现，这些发现揭示了这些方法在3个公开数据集上面临的根本局限性。我们的结果表明，调查的方案都不能抵御所有测试的失真。这项评估为当前水印方法在真实世界威胁下的表现提供了全面视角。相关演示和代码可在以下链接获取。 

---
# The Case for "Thick Evaluations" of Cultural Representation in AI 

**Title (ZH)**: " Thick 评价" 在人工智能文化表征中的重要性 

**Authors**: Rida Qadri, Mark Diaz, Ding Wang, Michael Madaio  

**Link**: [PDF](https://arxiv.org/pdf/2503.19075)  

**Abstract**: Generative AI image models have been increasingly evaluated for their (in)ability to represent non-Western cultures. We argue that these evaluations operate through reductive ideals of representation, abstracted from how people define their own representation and neglecting the inherently interpretive and contextual nature of cultural representation. In contrast to these 'thin' evaluations, we introduce the idea of 'thick evaluations': a more granular, situated, and discursive measurement framework for evaluating representations of social worlds in AI images, steeped in communities' own understandings of representation. We develop this evaluation framework through workshops in South Asia, by studying the 'thick' ways in which people interpret and assign meaning to images of their own cultures. We introduce practices for thicker evaluations of representation that expand the understanding of representation underpinning AI evaluations and by co-constructing metrics with communities, bringing measurement in line with the experiences of communities on the ground. 

**Abstract (ZH)**: 生成式AI图像模型越来越被评估其（不）能够代表非西方文化的能力。我们argue提出这些评估依赖于简化了的表现形式理想，这些理想从人们如何定义自己的表现中抽象出来，并忽视了文化表现本质上是解释性和情境性的特征。与这些“薄”评估相比，我们引入了“厚”评估的理念：一种更为精细、情境化和论述性的评价框架，用于评估AI图像中社会世界的表征，植根于社区自身对表征的理解。我们通过在南亚的工作坊开发这一评价框架，研究人们如何“厚”地解释和赋予文化图像意义。我们提出了扩展AI评估中表征理解的“厚”评价实践，并通过与社区共同构建指标，使评价与地面上社区的经验相一致。 

---
# Graph-Level Label-Only Membership Inference Attack against Graph Neural Networks 

**Title (ZH)**: 基于图神经网络的图级别标签唯一性成员推理攻击 

**Authors**: Jiazhu Dai, Yubing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.19070)  

**Abstract**: Graph neural networks (GNNs) are widely used for graph-structured data but are vulnerable to membership inference attacks (MIAs) in graph classification tasks, which determine if a graph was part of the training dataset, potentially causing data leakage. Existing MIAs rely on prediction probability vectors, but they become ineffective when only prediction labels are available. We propose a Graph-level Label-Only Membership Inference Attack (GLO-MIA), which is based on the intuition that the target model's predictions on training data are more stable than those on testing data. GLO-MIA generates a set of perturbed graphs for target graph by adding perturbations to its effective features and queries the target model with the perturbed graphs to get their prediction labels, which are then used to calculate robustness score of the target graph. Finally, by comparing the robustness score with a predefined threshold, the membership of the target graph can be inferred correctly with high probability. Our evaluation on three datasets and four GNN models shows that GLO-MIA achieves an attack accuracy of up to 0.825, outperforming baseline work by 8.5% and closely matching the performance of probability-based MIAs, even with only prediction labels. 

**Abstract (ZH)**: 基于图级别标签唯一性推理攻击（GLO-MIA）：图神经网络中的会员推理攻击 

---
# Minimum Volume Conformal Sets for Multivariate Regression 

**Title (ZH)**: 多元回归中最小区间套合集 

**Authors**: Sacha Braun, Liviu Aolaritei, Michael I. Jordan, Francis Bach  

**Link**: [PDF](https://arxiv.org/pdf/2503.19068)  

**Abstract**: Conformal prediction provides a principled framework for constructing predictive sets with finite-sample validity. While much of the focus has been on univariate response variables, existing multivariate methods either impose rigid geometric assumptions or rely on flexible but computationally expensive approaches that do not explicitly optimize prediction set volume. We propose an optimization-driven framework based on a novel loss function that directly learns minimum-volume covering sets while ensuring valid coverage. This formulation naturally induces a new nonconformity score for conformal prediction, which adapts to the residual distribution and covariates. Our approach optimizes over prediction sets defined by arbitrary norm balls, including single and multi-norm formulations. Additionally, by jointly optimizing both the predictive model and predictive uncertainty, we obtain prediction sets that are tight, informative, and computationally efficient, as demonstrated in our experiments on real-world datasets. 

**Abstract (ZH)**: 构形预测提供了一种原理性的框架，用于构建具有有限样本有效性的预测集。尽管大部分关注集中在一元响应变量上，现有的多元方法要么施加严格的几何假设，要么依赖于灵活但计算成本高昂的方法，这些方法并没有明确优化预测集的体积。我们提出了一种基于新型损失函数的优化驱动框架，该框架可以直接学习最小体积覆盖集，同时确保有效覆盖。该表述自然诱导了一种新的非一致性得分，该得分能够适应残差分布和协变量。我们的方法通过任意范数球定义的预测集进行优化，包括单范数和多范数形式。此外，通过同时优化预测模型和预测不确定性，我们获得了紧致、信息丰富且计算高效的预测集，这一特点已在我们的实证研究中得到验证。 

---
# Forecasting Labor Demand: Predicting JOLT Job Openings using Deep Learning Model 

**Title (ZH)**: 劳动力需求预测：基于深度学习模型预测JOLT岗位空缺 

**Authors**: Kyungsu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.19048)  

**Abstract**: This thesis studies the effectiveness of Long Short Term Memory model in forecasting future Job Openings and Labor Turnover Survey data in the United States. Drawing on multiple economic indicators from various sources, the data are fed directly into LSTM model to predict JOLT job openings in subsequent periods. The performance of the LSTM model is compared with conventional autoregressive approaches, including ARIMA, SARIMA, and Holt-Winters. Findings suggest that the LSTM model outperforms these traditional models in predicting JOLT job openings, as it not only captures the dependent variables trends but also harmonized with key economic factors. These results highlight the potential of deep learning techniques in capturing complex temporal dependencies in economic data, offering valuable insights for policymakers and stakeholders in developing data-driven labor market strategies 

**Abstract (ZH)**: 本论文研究长短期记忆模型在预测美国就业空缺和劳动力流动调查数据方面的有效性。利用多种经济指标数据，直接输入LSTM模型以预测后续时期的JOLT就业空缺数据。LSTM模型的性能与传统的自回归方法（包括ARIMA、SARIMA和Holt-Winters）进行比较。研究发现，LSTM模型在预测JOLT就业空缺方面优于传统模型，不仅捕捉了因变量的趋势，还与关键经济因素相协调。这些结果突显了深度学习技术在捕捉经济数据中复杂时间依赖关系的潜力，为政策制定者和利益相关者制定基于数据的劳动力市场策略提供了宝贵见解。 

---
# FACE: Few-shot Adapter with Cross-view Fusion for Cross-subject EEG Emotion Recognition 

**Title (ZH)**: FACE: 少量样本适配器与跨视图融合在跨被试EEG情绪识别中的应用 

**Authors**: Haiqi Liu, C. L. Philip Chen, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.18998)  

**Abstract**: Cross-subject EEG emotion recognition is challenged by significant inter-subject variability and intricately entangled intra-subject variability. Existing works have primarily addressed these challenges through domain adaptation or generalization strategies. However, they typically require extensive target subject data or demonstrate limited generalization performance to unseen subjects. Recent few-shot learning paradigms attempt to address these limitations but often encounter catastrophic overfitting during subject-specific adaptation with limited samples. This article introduces the few-shot adapter with a cross-view fusion method called FACE for cross-subject EEG emotion recognition, which leverages dynamic multi-view fusion and effective subject-specific adaptation. Specifically, FACE incorporates a cross-view fusion module that dynamically integrates global brain connectivity with localized patterns via subject-specific fusion weights to provide complementary emotional information. Moreover, the few-shot adapter module is proposed to enable rapid adaptation for unseen subjects while reducing overfitting by enhancing adapter structures with meta-learning. Experimental results on three public EEG emotion recognition benchmarks demonstrate FACE's superior generalization performance over state-of-the-art methods. FACE provides a practical solution for cross-subject scenarios with limited labeled data. 

**Abstract (ZH)**: 跨被试的EEG情绪识别受到显著的被试间变异性及复杂交织的被试内变异性挑战。现有的工作主要通过领域适应或泛化策略来应对这些挑战，但通常需要大量的目标被试数据或在未见被试上的泛化性能有限。最近的少样本学习范式试图解决这些局限性，但在有限样本下的被试特异性适应中常常遇到灾难性的过拟合。本文介绍了利用跨视角融合方法FACE进行跨被试EEG情绪识别的少样本适配器，该方法利用动态多视角融合和有效的被试特异性适应。具体来说，FACE整合了一个跨视角融合模块，该模块通过被试特异性融合权重动态地将全局脑连接与局部模式结合起来，提供互补的情绪信息。此外，提出了少样本适配器模块，能够通过增强适配器结构以元学习的方式减少过拟合来实现对未见被试的快速适应。在三个公开的EEG情绪识别基准上的实验结果表明，FACE在泛化性能上优于现有方法。FACE为有限标注数据的跨被试场景提供了一个实用的解决方案。 

---
# Enhanced prediction of spine surgery outcomes using advanced machine learning techniques and oversampling methods 

**Title (ZH)**: 使用高级机器学习技术和过采样方法增强脊柱手术结果预测 

**Authors**: José Alberto Benítez-Andrades, Camino Prada-García, Nicolás Ordás-Reyes, Marta Esteban Blanco, Alicia Merayo, Antonio Serrano-García  

**Link**: [PDF](https://arxiv.org/pdf/2503.18996)  

**Abstract**: The study proposes an advanced machine learning approach to predict spine surgery outcomes by incorporating oversampling techniques and grid search optimization. A variety of models including GaussianNB, ComplementNB, KNN, Decision Tree, and optimized versions with RandomOverSampler and SMOTE were tested on a dataset of 244 patients, which included pre-surgical, psychometric, socioeconomic, and analytical variables. The enhanced KNN models achieved up to 76% accuracy and a 67% F1-score, while grid-search optimization further improved performance. The findings underscore the potential of these advanced techniques to aid healthcare professionals in decision-making, with future research needed to refine these models on larger and more diverse datasets. 

**Abstract (ZH)**: 该研究提出一种先进的机器学习方法，通过结合过_sampling技术与网格搜索优化以预测脊柱手术结果。该方法在包含预手术、心理测量、社会经济和分析变量的244名患者数据集上测试了多种模型，包括GaussianNB、ComplementNB、KNN、决策树以及使用RandomOverSampler和SMOTE优化的版本。增强的KNN模型达到了最高76%的准确率和67%的F1分数，而网格搜索优化进一步提升了性能。研究结果强调了这些高级技术在辅助医疗专业人员决策方面的潜力，未来的研究需要在更大和更具多样性的数据集上细化这些模型。 

---
# HH4AI: A methodological Framework for AI Human Rights impact assessment under the EUAI ACT 

**Title (ZH)**: HH4AI: EUAI ACT背景下的人工智能人权影响评估方法框架 

**Authors**: Paolo Ceravolo, Ernesto Damiani, Maria Elisa D'Amico, Bianca de Teffe Erb, Simone Favaro, Nannerel Fiano, Paolo Gambatesa, Simone La Porta, Samira Maghool, Lara Mauri, Niccolo Panigada, Lorenzo Maria Ratto Vaquer, Marta A. Tamborini  

**Link**: [PDF](https://arxiv.org/pdf/2503.18994)  

**Abstract**: This paper introduces the HH4AI Methodology, a structured approach to assessing the impact of AI systems on human rights, focusing on compliance with the EU AI Act and addressing technical, ethical, and regulatory challenges. The paper highlights AIs transformative nature, driven by autonomy, data, and goal-oriented design, and how the EU AI Act promotes transparency, accountability, and safety. A key challenge is defining and assessing "high-risk" AI systems across industries, complicated by the lack of universally accepted standards and AIs rapid evolution.
To address these challenges, the paper explores the relevance of ISO/IEC and IEEE standards, focusing on risk management, data quality, bias mitigation, and governance. It proposes a Fundamental Rights Impact Assessment (FRIA) methodology, a gate-based framework designed to isolate and assess risks through phases including an AI system overview, a human rights checklist, an impact assessment, and a final output phase. A filtering mechanism tailors the assessment to the system's characteristics, targeting areas like accountability, AI literacy, data governance, and transparency.
The paper illustrates the FRIA methodology through a fictional case study of an automated healthcare triage service. The structured approach enables systematic filtering, comprehensive risk assessment, and mitigation planning, effectively prioritizing critical risks and providing clear remediation strategies. This promotes better alignment with human rights principles and enhances regulatory compliance. 

**Abstract (ZH)**: 本文介绍了HH4AI方法论，这是一种结构化的评估人工智能系统对人权影响的方法，重点关注欧盟AI法案的合规性，并解决技术、伦理和监管挑战。本文强调了人工智能的变革性质，受自主性、数据和以目标为导向的设计驱动，并指出欧盟AI法案如何促进透明度、问责制和安全性。一个关键挑战是定义和评估跨行业的“高风险”人工智能系统，这由于缺乏普遍接受的标准以及人工智能的快速发展而变得复杂。
为应对这些挑战，本文探讨了ISO/IEC和IEEE标准的相关性，重点关注风险管理、数据质量、偏见缓解和治理。本文提出了一种基本权利影响评估（FRIA）方法论，这是一种基于门控的框架，通过包括人工智能系统概览、人权检查清单、影响评估和最终输出阶段的阶段来隔离和评估风险。过滤机制根据系统的特性进行了定制，以针对问责制、人工智能素养、数据治理和透明度等领域。
本文通过一个虚构的自动化医疗服务分诊案例研究说明了FRIA方法论。这一结构化方法使系统性的筛选、全面的风险评估和缓解规划成为可能，有效优先考虑关键风险并提供清晰的补救策略，从而更好地与人权原则保持一致，并增强合规性。 

---
# Balanced Direction from Multifarious Choices: Arithmetic Meta-Learning for Domain Generalization 

**Title (ZH)**: 多方面选择中的平衡方向：领域泛化的算术元学习 

**Authors**: Xiran Wang, Jian Zhang, Lei Qi, Yinghuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2503.18987)  

**Abstract**: Domain generalization is proposed to address distribution shift, arising from statistical disparities between training source and unseen target domains. The widely used first-order meta-learning algorithms demonstrate strong performance for domain generalization by leveraging the gradient matching theory, which aims to establish balanced parameters across source domains to reduce overfitting to any particular domain. However, our analysis reveals that there are actually numerous directions to achieve gradient matching, with current methods representing just one possible path. These methods actually overlook another critical factor that the balanced parameters should be close to the centroid of optimal parameters of each source domain. To address this, we propose a simple yet effective arithmetic meta-learning with arithmetic-weighted gradients. This approach, while adhering to the principles of gradient matching, promotes a more precise balance by estimating the centroid between domain-specific optimal parameters. Experimental results validate the effectiveness of our strategy. 

**Abstract (ZH)**: 域泛化用于解决由于训练源域和未见过的目标域之间统计差异导致的分布偏移问题。广泛使用的基于一阶元学习算法通过梯度匹配理论在域泛化任务中展现了强大的性能，该理论旨在通过建立跨源域的平衡参数来减少对任何特定域的过拟合。然而，我们的分析表明，实际上存在许多实现梯度匹配的方向，当前方法仅代表其中一种路径。这些方法实际上忽略了另一个关键因素，即平衡参数应接近每个源域最优参数的质心。为了解决这一问题，我们提出了一个简单有效的算术元学习方法，该方法以算术加权梯度为特征，遵循梯度匹配的原则，通过估计不同源域最优参数的质心来促进更精确的平衡。实验结果验证了该策略的有效性。 

---
# LoRA Subtraction for Drift-Resistant Space in Exemplar-Free Continual Learning 

**Title (ZH)**: LoRA子空间减法用于无示例持续学习中的漂移抵抗空间 

**Authors**: Xuan Liu, Xiaobin Chang  

**Link**: [PDF](https://arxiv.org/pdf/2503.18985)  

**Abstract**: In continual learning (CL), catastrophic forgetting often arises due to feature drift. This challenge is particularly prominent in the exemplar-free continual learning (EFCL) setting, where samples from previous tasks cannot be retained, making it difficult to preserve prior knowledge. To address this issue, some EFCL methods aim to identify feature spaces that minimize the impact on previous tasks while accommodating new ones. However, they rely on static features or outdated statistics stored from old tasks, which prevents them from capturing the dynamic evolution of the feature space in CL, leading to performance degradation over time. In this paper, we introduce the Drift-Resistant Space (DRS), which effectively handles feature drifts without requiring explicit feature modeling or the storage of previous tasks. A novel parameter-efficient fine-tuning approach called Low-Rank Adaptation Subtraction (LoRA-) is proposed to develop the DRS. This method subtracts the LoRA weights of old tasks from the initial pre-trained weight before processing new task data to establish the DRS for model training. Therefore, LoRA- enhances stability, improves efficiency, and simplifies implementation. Furthermore, stabilizing feature drifts allows for better plasticity by learning with a triplet loss. Our method consistently achieves state-of-the-art results, especially for long task sequences, across multiple datasets. 

**Abstract (ZH)**: 在持续学习中，无例集的持续学习（EFCL）设置下特征漂移常导致灾难性遗忘。为此，本文提出了一种称为Drift-Resistant Space（DRS）的方法，无需显式建模特征或存储先前任务的数据，有效处理特征漂移。本文还提出了一种新的参数高效的微调方法——低秩适应减法（LoRA-），通过从初始化的预训练权重中减去老任务的LoRA权重来构建DRS，以适应新任务数据进行模型训练。LoRA-方法提高了稳定性、提高了效率并简化了实现。此外，稳定特征漂移有助于通过三元损失学习提高模型的可塑性。实验结果表明，该方法在多个数据集上，特别是在长任务序列上，达到了最先进的性能。 

---
# Confronting Catastrophic Risk: The International Obligation to Regulate Artificial Intelligence 

**Title (ZH)**: 应对 catastrophic 风险：国际 regulating 人工智能的义务 

**Authors**: Bryan Druzin, Anatole Boute, Michael Ramsden  

**Link**: [PDF](https://arxiv.org/pdf/2503.18983)  

**Abstract**: While artificial intelligence (AI) holds enormous promise, many experts in the field are warning that there is a non-trivial chance that the development of AI poses an existential threat to humanity. Existing regulatory initiative do not address this threat but merely instead focus on discrete AI-related risks such as consumer safety, cybersecurity, data protection, and privacy. In the absence of regulatory action to address the possible risk of human extinction by AI, the question arises: What legal obligations, if any, does public international law impose on states to regulate its development. Grounded in the precautionary principle, we argue that there exists an international obligation to mitigate the threat of human extinction by AI. Often invoked in relation to environmental regulation and the regulation of potentially harmful technologies, the principle holds that in situations where there is the potential for significant harm, even in the absence of full scientific certainty, preventive measures should not be postponed if delayed action may result in irreversible consequences. We argue that the precautionary principle is a general principle of international law and, therefore, that there is a positive obligation on states under the right to life within international human rights law to proactively take regulatory action to mitigate the potential existential risk of AI. This is significant because, if an international obligation to regulate the development of AI can be established under international law, then the basic legal framework would be in place to address this evolving threat. 

**Abstract (ZH)**: 人工智能的发展存在本质性威胁，国际法是否有规制义务 

---
# Generative Data Imputation for Sparse Learner Performance Data Using Generative Adversarial Imputation Networks 

**Title (ZH)**: 使用生成对抗插补网络的生成型数据插补法处理稀疏学习者绩效数据 

**Authors**: Liang Zhang, Jionghao Lin, John Sabatini, Diego Zapata-Rivera, Carol Forsyth, Yang Jiang, John Hollander, Xiangen Hu, Arthur C. Graesser  

**Link**: [PDF](https://arxiv.org/pdf/2503.18982)  

**Abstract**: Learner performance data collected by Intelligent Tutoring Systems (ITSs), such as responses to questions, is essential for modeling and predicting learners' knowledge states. However, missing responses due to skips or incomplete attempts create data sparsity, challenging accurate assessment and personalized instruction. To address this, we propose a generative imputation approach using Generative Adversarial Imputation Networks (GAIN). Our method features a three-dimensional (3D) framework (learners, questions, and attempts), flexibly accommodating various sparsity levels. Enhanced by convolutional neural networks and optimized with a least squares loss function, the GAIN-based method aligns input and output dimensions to question-attempt matrices along the learners' dimension. Extensive experiments using datasets from AutoTutor Adult Reading Comprehension (ARC), ASSISTments, and MATHia demonstrate that our approach significantly outperforms tensor factorization and alternative GAN methods in imputation accuracy across different attempt scenarios. Bayesian Knowledge Tracing (BKT) further validates the effectiveness of the imputed data by estimating learning parameters: initial knowledge (P(L0)), learning rate (P(T)), guess rate (P(G)), and slip rate (P(S)). Results indicate the imputed data enhances model fit and closely mirrors original distributions, capturing underlying learning behaviors reliably. Kullback-Leibler (KL) divergence assessments confirm minimal divergence, showing the imputed data preserves essential learning characteristics effectively. These findings underscore GAIN's capability as a robust imputation tool in ITSs, alleviating data sparsity and supporting adaptive, individualized instruction, ultimately leading to more precise and responsive learner assessments and improved educational outcomes. 

**Abstract (ZH)**: 基于生成对抗填充网络的智能辅导系统缺失响应数据填充方法 

---
# FedSKD: Aggregation-free Model-heterogeneous Federated Learning using Multi-dimensional Similarity Knowledge Distillation 

**Title (ZH)**: FedSKD：基于多维度相似性知识蒸馏的无聚合模型异构联邦学习 

**Authors**: Ziqiao Weng, Weidong Cai, Bo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.18981)  

**Abstract**: Federated learning (FL) enables privacy-preserving collaborative model training without direct data sharing. Model-heterogeneous FL (MHFL) extends this paradigm by allowing clients to train personalized models with heterogeneous architectures tailored to their computational resources and application-specific needs. However, existing MHFL methods predominantly rely on centralized aggregation, which introduces scalability and efficiency bottlenecks, or impose restrictions requiring partially identical model architectures across clients. While peer-to-peer (P2P) FL removes server dependence, it suffers from model drift and knowledge dilution, limiting its effectiveness in heterogeneous settings. To address these challenges, we propose FedSKD, a novel MHFL framework that facilitates direct knowledge exchange through round-robin model circulation, eliminating the need for centralized aggregation while allowing fully heterogeneous model architectures across clients. FedSKD's key innovation lies in multi-dimensional similarity knowledge distillation, which enables bidirectional cross-client knowledge transfer at batch, pixel/voxel, and region levels for heterogeneous models in FL. This approach mitigates catastrophic forgetting and model drift through progressive reinforcement and distribution alignment while preserving model heterogeneity. Extensive evaluations on fMRI-based autism spectrum disorder diagnosis and skin lesion classification demonstrate that FedSKD outperforms state-of-the-art heterogeneous and homogeneous FL baselines, achieving superior personalization (client-specific accuracy) and generalization (cross-institutional adaptability). These findings underscore FedSKD's potential as a scalable and robust solution for real-world medical federated learning applications. 

**Abstract (ZH)**: 联邦学习（FL）在不直接共享数据的情况下实现隐私保护的协作模型训练。异构模型联邦学习（MHFL）通过允许客户端使用针对其计算资源和应用特定需求定制的异构架构训练个性化模型，扩展了这一范式。然而，现有的MHFL方法主要依赖中心化聚合，这引入了可扩展性和效率瓶颈，或者需要客户端之间部分相同的模型架构。虽然对等（P2P）FL去除了对服务器的依赖，但它遭受模型漂移和知识稀释的问题，限制了其在异构环境中的有效性。为了解决这些挑战，我们提出了一种名为FedSKD的新型MHFL框架，该框架通过轮询模型循环直接促进知识交流，无需中心化聚合，同时允许客户端之间具有完全异构的模型架构。FedSKD的核心创新在于多维度相似性知识蒸馏，这使得在批处理、像素/体素和区域级别，异构模型在FL中能够进行双向跨客户端知识传输。这种方法通过渐进强化和分布对齐来缓解灾难性遗忘和模型漂移，同时保持模型异质性。在基于fMRI的自闭症谱系障碍诊断和皮肤病变分类的广泛评估中，FedSKD表现出色，优于最先进的异构和同构FL基线，实现了更好的个性化（客户端特定准确度）和泛化能力（跨机构适应性）。这些发现强调了FedSKD作为适用于实际医疗联邦学习应用的可扩展和稳健解决方案的潜力。 

---
# Threshold Crossings as Tail Events for Catastrophic AI Risk 

**Title (ZH)**: 阈值穿越作为灾难性AI风险的尾事件 

**Authors**: Elija Perrier  

**Link**: [PDF](https://arxiv.org/pdf/2503.18979)  

**Abstract**: We analyse circumstances in which bifurcation-driven jumps in AI systems with their emergent heavy-tailed outcome distributions. By analysing how a control parameter's random fluctuations near a catastrophic threshold generate extreme outcomes, we demonstrate in what circumstances the probability of a sudden, large-scale, transition aligns closely with the tail probability of the resulting damage distribution. Our results contribute to research in monitoring, mitigation and control of AI systems when seeking to manage potentially catastrophic AI risk. 

**Abstract (ZH)**: 我们分析由分叉驱动的AI系统中 Emergent 稀尾结果分布的跃变情况。通过分析控制参数在灾难性阈值附近随机波动如何产生极端结果，我们展示了在何种情况下突然的大规模转变的概率与结果损害分布的尾部概率高度一致。我们的研究结果有助于在管理潜在灾难性AI风险时对AI系统的监控、缓解和控制方面的研究。 

---
# Machine Learning - Driven Materials Discovery: Unlocking Next-Generation Functional Materials - A minireview 

**Title (ZH)**: 基于机器学习的材料发现：解锁下一代功能材料——一篇简评 

**Authors**: Dilshod Nematov, Mirabbos Hojamberdiev  

**Link**: [PDF](https://arxiv.org/pdf/2503.18975)  

**Abstract**: The rapid advancement of machine learning and artificial intelligence (AI)-driven techniques is revolutionizing materials discovery, property prediction, and material design by minimizing human intervention and accelerating scientific progress. This review provides a comprehensive overview of smart, machine learning (ML)-driven approaches, emphasizing their role in predicting material properties, discovering novel compounds, and optimizing material structures. Key methodologies ranging from deep learning, graph neural networks, and Bayesian optimization to automated generative models, such as generative adversarial networks (GANs) and variational autoencoders (VAEs) enable the autonomous design of materials with tailored functionalities. By leveraging AutoML frameworks (e.g., AutoGluon, TPOT, and this http URL), researchers can automate the model selection, hyperparameter tuning, and feature engineering, significantly improving the efficiency of materials informatics. Furthermore, the integration of AI-driven robotic laboratories and high-throughput computing has established a fully automated pipeline for rapid synthesis and experimental validation, drastically reducing the time and cost of material discovery. This review highlights real-world applications of automated ML-driven approaches in predicting mechanical, thermal, electrical, and optical properties of materials, demonstrating successful cases in superconductors, catalysts, photovoltaics, and energy storage systems. We also address key challenges, such as data quality, interpretability, and the integration of AutoML with quantum computing, which are essential for future advancements. Ultimately, the synergy between AI, automated experimentation, and computational modeling transforms the way the materials are discovered, optimized, and designed, paving the way for next-generation innovations in energy, electronics, and nanotechnology. 

**Abstract (ZH)**: 机器学习和人工智能驱动方法的快速进步正在变革材料发现、性质预测和材料设计，通过减少人为干预并加速科学研究进程。本文综述了智能、机器学习（ML）驱动的方法，强调了其在预测材料性质、发现新型化合物和优化材料结构方面的作用。从深度学习、图神经网络、贝叶斯优化到自动化生成模型（如生成对抗网络GANs和变分自编码器VAEs），这些方法使能够自主设计具有定制功能的材料。通过利用AutoML框架（如AutoGluon、TPOT等），研究人员可以自动进行模型选择、超参数调优和特征工程，显著提高材料信息化的效率。此外，AI驱动的机器人实验室和高通量计算的整合建立了从快速合成到实验验证的全自动管道，极大地减少了材料发现的时间和成本。本文还强调了自动化ML驱动方法在预测材料的机械、热、电和光学性质方面的实际应用，展示了在超导体、催化剂、光伏和能量存储系统中的成功案例。本文也探讨了数据质量、解释性以及AutoML与量子计算的集成等关键挑战，这些是未来发展的必要条件。最终，AI、自动化实验和计算建模的协同作用改变了材料的发现、优化和设计方式，为能源、电子和纳米技术领域的新一代创新铺平了道路。 

---
# International Agreements on AI Safety: Review and Recommendations for a Conditional AI Safety Treaty 

**Title (ZH)**: 国际人工智能安全协议：审核与建议——条件性人工智能安全条约 

**Authors**: Rebecca Scholefield, Samuel Martin, Otto Barten  

**Link**: [PDF](https://arxiv.org/pdf/2503.18956)  

**Abstract**: The malicious use or malfunction of advanced general-purpose AI (GPAI) poses risks that, according to leading experts, could lead to the 'marginalisation or extinction of humanity.' To address these risks, there are an increasing number of proposals for international agreements on AI safety. In this paper, we review recent (2023-) proposals, identifying areas of consensus and disagreement, and drawing on related literature to assess their feasibility. We focus our discussion on risk thresholds, regulations, types of international agreement and five related processes: building scientific consensus, standardisation, auditing, verification and incentivisation.
Based on this review, we propose a treaty establishing a compute threshold above which development requires rigorous oversight. This treaty would mandate complementary audits of models, information security and governance practices, overseen by an international network of AI Safety Institutes (AISIs) with authority to pause development if risks are unacceptable. Our approach combines immediately implementable measures with a flexible structure that can adapt to ongoing research. 

**Abstract (ZH)**: 高级通用人工智能的恶意使用或故障可能给人类带来‘边缘化或灭绝’的风险，根据领先专家的观点。为应对这些风险，国际上关于人工智能安全的协议正逐渐增多。本文回顾了近期（2023年及以后）的提议，识别共识与分歧，并结合相关文献评估其可行性。我们重点关注风险管理阈值、监管措施、国际协议类型以及五项相关过程：建立科学共识、标准化、审计、验证和激励机制。基于这一回顾，我们提出一项条约，设立一个计算阈值，在此阈值之上，开发需接受严格的监管。该条约要求由国际人工智能安全性研究所网络进行互补审计，包括模型、信息安全和治理实践的审计，必要时有权暂停开发。我们的方法结合了可立即实施的措施，并具备根据持续研究进行调整的灵活性。 

---
# On the Hopf-Cole Transform for Control-affine Schrödinger Bridge 

**Title (ZH)**: Hopf-Cole 变换在控制拟线性薛定谔桥中的应用 

**Authors**: Alexis Teter, Abhishek Halder  

**Link**: [PDF](https://arxiv.org/pdf/2503.17640)  

**Abstract**: The purpose of this note is to clarify the importance of the relation $\boldsymbol{gg}^{\top}\propto \boldsymbol{\sigma\sigma}^{\top}$ in solving control-affine Schrödinger bridge problems via the Hopf-Cole transform, where $\boldsymbol{g},\boldsymbol{\sigma}$ are the control and noise coefficients, respectively. We show that the Hopf-Cole transform applied to the conditions of optimality for generic control-affine Schrödinger bridge problems, i.e., without the assumption $\boldsymbol{gg}^{\top}\propto\boldsymbol{\sigma\sigma}^{\top}$, gives a pair of forward-backward PDEs that are neither linear nor equation-level decoupled. We explain how the resulting PDEs can be interpreted as nonlinear forward-backward advection-diffusion-reaction equations, where the nonlinearity stem from additional drift and reaction terms involving the gradient of the log-likelihood a.k.a. the score. These additional drift and reaction vanish when $\boldsymbol{gg}^{\top}\propto\boldsymbol{\sigma\sigma}^{\top}$, and the resulting boundary-coupled system of linear PDEs can then be solved by dynamic Sinkhorn recursions. A key takeaway of our work is that the numerical solution of the generic control-affine Schrödinger bridge requires further algorithmic development, possibly generalizing the dynamic Sinkhorn recursion or otherwise. 

**Abstract (ZH)**: 探讨控制 affine 施勞丁格桥梁问题通过霍普夫-科尔变换求解的重要性：$\boldsymbol{gg}^{\top}\propto \boldsymbol{\sigma\sigma}^{\top}$ 关系的作用 

---
