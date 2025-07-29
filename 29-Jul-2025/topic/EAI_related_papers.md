# PixelNav: Towards Model-based Vision-Only Navigation with Topological Graphs 

**Title (ZH)**: PixelNav: 向基于拓扑图的纯视觉导航模型方向发展 

**Authors**: Sergey Bakulin, Timur Akhtyamov, Denis Fatykhov, German Devchich, Gonzalo Ferrer  

**Link**: [PDF](https://arxiv.org/pdf/2507.20892)  

**Abstract**: This work proposes a novel hybrid approach for vision-only navigation of mobile robots, which combines advances of both deep learning approaches and classical model-based planning algorithms. Today, purely data-driven end-to-end models are dominant solutions to this problem. Despite advantages such as flexibility and adaptability, the requirement of a large amount of training data and limited interpretability are the main bottlenecks for their practical applications. To address these limitations, we propose a hierarchical system that utilizes recent advances in model predictive control, traversability estimation, visual place recognition, and pose estimation, employing topological graphs as a representation of the target environment. Using such a combination, we provide a scalable system with a higher level of interpretability compared to end-to-end approaches. Extensive real-world experiments show the efficiency of the proposed method. 

**Abstract (ZH)**: 本研究提出了一种新颖的混合方法，用于仅依靠视觉的移动机器人导航，该方法结合了深度学习方法和经典模型导向规划算法的最新进展。 

---
# A Human-in-the-loop Approach to Robot Action Replanning through LLM Common-Sense Reasoning 

**Title (ZH)**: 基于LLM常识推理的人在回环中的机器人动作重规划方法 

**Authors**: Elena Merlo, Marta Lagomarsino, Arash Ajoudani  

**Link**: [PDF](https://arxiv.org/pdf/2507.20870)  

**Abstract**: To facilitate the wider adoption of robotics, accessible programming tools are required for non-experts. Observational learning enables intuitive human skills transfer through hands-on demonstrations, but relying solely on visual input can be inefficient in terms of scalability and failure mitigation, especially when based on a single demonstration. This paper presents a human-in-the-loop method for enhancing the robot execution plan, automatically generated based on a single RGB video, with natural language input to a Large Language Model (LLM). By including user-specified goals or critical task aspects and exploiting the LLM common-sense reasoning, the system adjusts the vision-based plan to prevent potential failures and adapts it based on the received instructions. Experiments demonstrated the framework intuitiveness and effectiveness in correcting vision-derived errors and adapting plans without requiring additional demonstrations. Moreover, interactive plan refinement and hallucination corrections promoted system robustness. 

**Abstract (ZH)**: 面向非专家的易用编程工具是促进机器人更广泛采用的关键。观察学习可以通过亲手示范实现直观的人类技能转移，但仅依赖视觉输入在可扩展性和故障缓解方面可能效率低下，尤其是在基于单次示范时。本文提出一种在环人类辅助方法，通过自然语言输入到大型语言模型（LLM），增强基于单个RGB视频自动生成的机器人执行计划。通过纳入用户指定的目标或关键任务方面，并利用LLM的常识推理，系统调整基于视觉的计划以防止潜在失败，并根据接收到的指令进行适应。实验表明，该框架在纠正基于视觉的错误和不需额外示范的情况下调整计划方面具有直观性和有效性。此外，交互式计划细化和幻觉修正提升了系统的鲁棒性。 

---
# Uncertainty-aware Planning with Inaccurate Models for Robotized Liquid Handling 

**Title (ZH)**: 基于不准确模型的aware不确定性规划与机器人液体处理 

**Authors**: Marco Faroni, Carlo Odesco, Andrea Zanchettin, Paolo Rocco  

**Link**: [PDF](https://arxiv.org/pdf/2507.20861)  

**Abstract**: Physics-based simulations and learning-based models are vital for complex robotics tasks like deformable object manipulation and liquid handling. However, these models often struggle with accuracy due to epistemic uncertainty or the sim-to-real gap. For instance, accurately pouring liquid from one container to another poses challenges, particularly when models are trained on limited demonstrations and may perform poorly in novel situations. This paper proposes an uncertainty-aware Monte Carlo Tree Search (MCTS) algorithm designed to mitigate these inaccuracies. By incorporating estimates of model uncertainty, the proposed MCTS strategy biases the search towards actions with lower predicted uncertainty. This approach enhances the reliability of planning under uncertain conditions. Applied to a liquid pouring task, our method demonstrates improved success rates even with models trained on minimal data, outperforming traditional methods and showcasing its potential for robust decision-making in robotics. 

**Abstract (ZH)**: 基于物理的模拟和基于学习的模型对于复杂机器人任务如可变形物体操作和液体处理至关重要。然而，这些模型往往由于认识不确定性或仿真到真实世界的差距而准确性不足。例如，精确地将液体从一个容器倒入另一个容器时会面临挑战，特别是在模型基于有限演示训练的情况下，在新颖情况下表现较差。本文提出了一种意识到不确定性的蒙特卡洛树搜索（MCTS）算法，旨在减轻这些不准确性。通过融入模型不确定性估计，所提出的MCTS策略倾向于那些预测不确定性较低的动作。这种方法在不确定条件下增强了规划的可靠性。应用于液体倾倒任务时，我们的方法即使在使用少量数据训练的模型上也表现出更高的成功率，优于传统方法，并展示了其在机器人中进行稳健决策的潜力。 

---
# Free Energy-Inspired Cognitive Risk Integration for AV Navigation in Pedestrian-Rich Environments 

**Title (ZH)**: 受自由能启发的认知风险集成在行人密集环境中的自动驾驶导航 

**Authors**: Meiting Dang, Yanping Wu, Yafei Wang, Dezong Zhao, David Flynn, Chongfeng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2507.20850)  

**Abstract**: Recent advances in autonomous vehicle (AV) behavior planning have shown impressive social interaction capabilities when interacting with other road users. However, achieving human-like prediction and decision-making in interactions with vulnerable road users remains a key challenge in complex multi-agent interactive environments. Existing research focuses primarily on crowd navigation for small mobile robots, which cannot be directly applied to AVs due to inherent differences in their decision-making strategies and dynamic boundaries. Moreover, pedestrians in these multi-agent simulations follow fixed behavior patterns that cannot dynamically respond to AV actions. To overcome these limitations, this paper proposes a novel framework for modeling interactions between the AV and multiple pedestrians. In this framework, a cognitive process modeling approach inspired by the Free Energy Principle is integrated into both the AV and pedestrian models to simulate more realistic interaction dynamics. Specifically, the proposed pedestrian Cognitive-Risk Social Force Model adjusts goal-directed and repulsive forces using a fused measure of cognitive uncertainty and physical risk to produce human-like trajectories. Meanwhile, the AV leverages this fused risk to construct a dynamic, risk-aware adjacency matrix for a Graph Convolutional Network within a Soft Actor-Critic architecture, allowing it to make more reasonable and informed decisions. Simulation results indicate that our proposed framework effectively improves safety, efficiency, and smoothness of AV navigation compared to the state-of-the-art method. 

**Abstract (ZH)**: Recent Advances in Autonomous Vehicle Behavior Planning for Realistic Interaction with Vulnerable Road Users Through a Novel Cognitive-Risk Framework 

---
# Hanging Around: Cognitive Inspired Reasoning for Reactive Robotics 

**Title (ZH)**: 悬挂于此：认知启发式的反应式机器人推理 

**Authors**: Mihai Pomarlan, Stefano De Giorgis, Rachel Ringe, Maria M. Hedblom, Nikolaos Tsiogkas  

**Link**: [PDF](https://arxiv.org/pdf/2507.20832)  

**Abstract**: Situationally-aware artificial agents operating with competence in natural environments face several challenges: spatial awareness, object affordance detection, dynamic changes and unpredictability. A critical challenge is the agent's ability to identify and monitor environmental elements pertinent to its objectives. Our research introduces a neurosymbolic modular architecture for reactive robotics. Our system combines a neural component performing object recognition over the environment and image processing techniques such as optical flow, with symbolic representation and reasoning. The reasoning system is grounded in the embodied cognition paradigm, via integrating image schematic knowledge in an ontological structure. The ontology is operatively used to create queries for the perception system, decide on actions, and infer entities' capabilities derived from perceptual data. The combination of reasoning and image processing allows the agent to focus its perception for normal operation as well as discover new concepts for parts of objects involved in particular interactions. The discovered concepts allow the robot to autonomously acquire training data and adjust its subsymbolic perception to recognize the parts, as well as making planning for more complex tasks feasible by focusing search on those relevant object parts. We demonstrate our approach in a simulated world, in which an agent learns to recognize parts of objects involved in support relations. While the agent has no concept of handle initially, by observing examples of supported objects hanging from a hook it learns to recognize the parts involved in establishing support and becomes able to plan the establishment/destruction of the support relation. This underscores the agent's capability to expand its knowledge through observation in a systematic way, and illustrates the potential of combining deep reasoning [...]. 

**Abstract (ZH)**: 情景感知的人工智能代理在自然环境中高效运作面临的挑战：空间意识、物体操作检测、动态变化和不确定性。一个关键挑战是代理识别和监控与其目标相关环境元素的能力。我们的研究提出了一种神经符号模块化架构以应对反应式机器人面临的挑战。该系统结合了神经网络执行的环境物体识别和光学流等图像处理技术，同时使用符号表示和推理。推理系统通过将图像示意性知识整合到本体结构中，基于体表认知范式运行。本体用于为感知系统创建查询、决定行动以及从感知数据中推断实体的能力。推理与图像处理的结合使代理能够在正常操作时集中感知，并且能够发现与特定交互相关的对象部分的新概念。发现的概念使机器人能够自主获取训练数据，调整其次符号感知以识别这些部分，并通过聚焦于相关对象部分的搜索使复杂任务的规划成为可能。我们在一个模拟世界中展示了该方法，一个代理学会了识别涉及支持关系的对象部分。在最初没有把手的概念的情况下，通过观察悬挂于挂钩上的支撑对象示例，代理学会了识别建立支持的部分，并能够规划支持关系的建立或破坏。这强调了代理通过系统观察扩展知识的能力，并展示了结合深度推理 […] 的潜力。 

---
# FMimic: Foundation Models are Fine-grained Action Learners from Human Videos 

**Title (ZH)**: FMimic: 基础模型是从人类视频中细粒度学习动作的模型 

**Authors**: Guangyan Chen, Meiling Wang, Te Cui, Yao Mu, Haoyang Lu, Zicai Peng, Mengxiao Hu, Tianxing Zhou, Mengyin Fu, Yi Yang, Yufeng Yue  

**Link**: [PDF](https://arxiv.org/pdf/2507.20622)  

**Abstract**: Visual imitation learning (VIL) provides an efficient and intuitive strategy for robotic systems to acquire novel skills. Recent advancements in foundation models, particularly Vision Language Models (VLMs), have demonstrated remarkable capabilities in visual and linguistic reasoning for VIL tasks. Despite this progress, existing approaches primarily utilize these models for learning high-level plans from human demonstrations, relying on pre-defined motion primitives for executing physical interactions, which remains a major bottleneck for robotic systems. In this work, we present FMimic, a novel paradigm that harnesses foundation models to directly learn generalizable skills at even fine-grained action levels, using only a limited number of human videos. Extensive experiments demonstrate that our FMimic delivers strong performance with a single human video, and significantly outperforms all other methods with five videos. Furthermore, our method exhibits significant improvements of over 39% and 29% in RLBench multi-task experiments and real-world manipulation tasks, respectively, and exceeds baselines by more than 34% in high-precision tasks and 47% in long-horizon tasks. 

**Abstract (ZH)**: 视觉模仿学习（VIL）为机器人系统获取新型技能提供了一种高效直观的策略。基于基础模型的最新进展，尤其是视觉语言模型（VLMs），展示了在VIL任务中视觉和语言推理的非凡能力。尽管取得了这些进展，现有的方法主要利用这些模型从人类示范中学习高级规划，并依赖预定义的运动基元来执行物理交互，这是机器人系统的一个主要瓶颈。在此项工作中，我们提出了FMimic，这是一种全新的范式，利用基础模型直接在极细粒度的动作层面学习可泛化的技能，仅使用少量的人类视频。 extensive实验表明，我们的FMimic仅使用单个人类视频就能达到出色的性能，并且在使用五个视频时显著优于所有其他方法。此外，在RLBench多任务实验和实际操作任务中，我们的方法分别表现出超过39%和29%的改进，并在高精度任务和长时序任务中分别超过基线方法34%和47%。 

---
# Uni-Mapper: Unified Mapping Framework for Multi-modal LiDARs in Complex and Dynamic Environments 

**Title (ZH)**: Uni-Mapper: 统一多模态LiDAR在复杂动态环境中的mapping框架 

**Authors**: Gilhwan Kang, Hogyun Kim, Byunghee Choi, Seokhwan Jeong, Young-Sik Shin, Younggun Cho  

**Link**: [PDF](https://arxiv.org/pdf/2507.20538)  

**Abstract**: The unification of disparate maps is crucial for enabling scalable robot operation across multiple sessions and collaborative multi-robot scenarios. However, achieving a unified map robust to sensor modalities and dynamic environments remains a challenging problem. Variations in LiDAR types and dynamic elements lead to differences in point cloud distribution and scene consistency, hindering reliable descriptor generation and loop closure detection essential for accurate map alignment. To address these challenges, this paper presents Uni-Mapper, a dynamic-aware 3D point cloud map merging framework for multi-modal LiDAR systems. It comprises dynamic object removal, dynamic-aware loop closure, and multi-modal LiDAR map merging modules. A voxel-wise free space hash map is built in a coarse-to-fine manner to identify and reject dynamic objects via temporal occupancy inconsistencies. The removal module is integrated with a LiDAR global descriptor, which encodes preserved static local features to ensure robust place recognition in dynamic environments. In the final stage, multiple pose graph optimizations are conducted for both intra-session and inter-map loop closures. We adopt a centralized anchor-node strategy to mitigate intra-session drift errors during map merging. In the final stage, centralized anchor-node-based pose graph optimization is performed to address intra- and inter-map loop closures for globally consistent map merging. Our framework is evaluated on diverse real-world datasets with dynamic objects and heterogeneous LiDARs, showing superior performance in loop detection across sensor modalities, robust mapping in dynamic environments, and accurate multi-map alignment over existing methods. Project Page: this https URL. 

**Abstract (ZH)**: 多模态激光雷达系统中动态感知的3D点云地图融合框架 

---
# Learning Physical Interaction Skills from Human Demonstrations 

**Title (ZH)**: 从人类示范中学习物理交互技能 

**Authors**: Tianyu Li, Hengbo Ma, Sehoon Ha, Kwonjoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.20445)  

**Abstract**: Learning physical interaction skills, such as dancing, handshaking, or sparring, remains a fundamental challenge for agents operating in human environments, particularly when the agent's morphology differs significantly from that of the demonstrator. Existing approaches often rely on handcrafted objectives or morphological similarity, limiting their capacity for generalization. Here, we introduce a framework that enables agents with diverse embodiments to learn wholebbody interaction behaviors directly from human demonstrations. The framework extracts a compact, transferable representation of interaction dynamics, called the Embedded Interaction Graph (EIG), which captures key spatiotemporal relationships between the interacting agents. This graph is then used as an imitation objective to train control policies in physics-based simulations, allowing the agent to generate motions that are both semantically meaningful and physically feasible. We demonstrate BuddyImitation on multiple agents, such as humans, quadrupedal robots with manipulators, or mobile manipulators and various interaction scenarios, including sparring, handshaking, rock-paper-scissors, or dancing. Our results demonstrate a promising path toward coordinated behaviors across morphologically distinct characters via cross embodiment interaction learning. 

**Abstract (ZH)**: 一种框架使不同形态的代理可以直接从人类示范中学习全身交互行为，以解决在人类环境中操作时形态差异显著的代理学习物理互动技能的问题。该框架提取了一个紧凑且可转移的交互动力学表示——嵌入式交互图（EIG），以捕捉交互代理间的时空关系。随后，该图被用作模仿目标，用于基于物理的模拟中训练控制策略，从而使代理能够生成既具有语义意义又符合物理可能性的运动。我们展示了BuddyImitation在多种代理上，如人类、具有 manipulator 的四足机器人或移动 manipulator，以及多种交互场景，包括对打、握手、石头剪刀布或跳舞中的应用。我们的结果展示了一种通过跨形态互动学习实现不同形态角色协调行为的有希望的途径。 

---
# Bipedalism for Quadrupedal Robots: Versatile Loco-Manipulation through Risk-Adaptive Reinforcement Learning 

**Title (ZH)**: 四足机器人的人行 :";
user
请纠正这个标题：Bipedalism for Quadrupedal Robots: Versatile Loco-Manipulation through Risk-Adaptive Reinforcement Learning

四足机器人的人行走态：基于风险自适应强化学习的多功能运动操控 

**Authors**: Yuyou Zhang, Radu Corcodel, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.20382)  

**Abstract**: Loco-manipulation of quadrupedal robots has broadened robotic applications, but using legs as manipulators often compromises locomotion, while mounting arms complicates the system. To mitigate this issue, we introduce bipedalism for quadrupedal robots, thus freeing the front legs for versatile interactions with the environment. We propose a risk-adaptive distributional Reinforcement Learning (RL) framework designed for quadrupedal robots walking on their hind legs, balancing worst-case conservativeness with optimal performance in this inherently unstable task. During training, the adaptive risk preference is dynamically adjusted based on the uncertainty of the return, measured by the coefficient of variation of the estimated return distribution. Extensive experiments in simulation show our method's superior performance over baselines. Real-world deployment on a Unitree Go2 robot further demonstrates the versatility of our policy, enabling tasks like cart pushing, obstacle probing, and payload transport, while showcasing robustness against challenging dynamics and external disturbances. 

**Abstract (ZH)**: 四足机器人 bipedalism 的引入拓宽了机器人的应用范围，通过使前腿自由，实现了与环境的多样互动。我们提出一种基于分布的风险适应性强化学习（RL）框架，该框架适用于四足机器人后腿行走的任务，能够在这一固有的不稳定任务中实现最坏情况下的保守性和最优性能的最佳平衡。在训练过程中，根据估计回报分布的标准差系数动态调整适应的风险偏好。仿真中的广泛实验显示，我们的方法在基准方法之上表现出更优的性能。在 Unitree Go2 网格机器人的实际部署中，进一步展示了该策略的灵活性，能够执行小车推拉、障碍探查和负载运输等任务，并展示了对复杂动态和外部干扰的鲁棒性。 

---
# Advancing Shared and Multi-Agent Autonomy in Underwater Missions: Integrating Knowledge Graphs and Retrieval-Augmented Generation 

**Title (ZH)**: 推进海底任务中共享与多智能体自主性：集成知识图谱与检索增强生成 

**Authors**: Michele Grimaldi, Carlo Cernicchiaro, Sebastian Realpe Rua, Alaaeddine El-Masri-El-Chaarani, Markus Buchholz, Loizos Michael, Pere Ridao Rodriguez, Ignacio Carlucho, Yvan R. Petillot  

**Link**: [PDF](https://arxiv.org/pdf/2507.20370)  

**Abstract**: Robotic platforms have become essential for marine operations by providing regular and continuous access to offshore assets, such as underwater infrastructure inspection, environmental monitoring, and resource exploration. However, the complex and dynamic nature of underwater environments, characterized by limited visibility, unpredictable currents, and communication constraints, presents significant challenges that demand advanced autonomy while ensuring operator trust and oversight. Central to addressing these challenges are knowledge representation and reasoning techniques, particularly knowledge graphs and retrieval-augmented generation (RAG) systems, that enable robots to efficiently structure, retrieve, and interpret complex environmental data. These capabilities empower robotic agents to reason, adapt, and respond effectively to changing conditions. The primary goal of this work is to demonstrate both multi-agent autonomy and shared autonomy, where multiple robotic agents operate independently while remaining connected to a human supervisor. We show how a RAG-powered large language model, augmented with knowledge graph data and domain taxonomy, enables autonomous multi-agent decision-making and facilitates seamless human-robot interaction, resulting in 100\% mission validation and behavior completeness. Finally, ablation studies reveal that without structured knowledge from the graph and/or taxonomy, the LLM is prone to hallucinations, which can compromise decision quality. 

**Abstract (ZH)**: 机器人平台已成为海上操作不可或缺的部分，通过提供定期且连续的离岸资产访问，如水下基础设施 inspection、环境监测和资源勘探。然而，海底环境的复杂和动态特性，包括有限的能见度、不可预测的洋流和通信限制，提出了重大挑战，要求先进的自主性以确保操作员的信任和监督。解决这些挑战的关键在于知识表示和推理技术，尤其是知识图谱和检索增强生成（RAG）系统，这些技术使机器人能够高效地结构化、检索和解释复杂的环境数据。这些能力使机器人代理能够进行推理、适应并有效应对变化的条件。本文的主要目标是展示多代理自主性和共享自主性，其中多个机器人代理独立操作同时保持与人类监督员的连接。我们展示了如何通过增强的大语言模型和知识图谱数据及领域分类，实现自主多代理决策并促进无缝的人机交互，结果验证率为100%，行为完备性为100%。最后，消融研究显示，没有图表结构化知识和/或分类，大语言模型容易产生幻觉，这可能会影响决策质量。 

---
# Decentralized Uncertainty-Aware Multi-Agent Collision Avoidance With Model Predictive Path Integral 

**Title (ZH)**: 去中心化感知不确定性多智能体碰撞 avoidance ewith 模型预测路径积分 

**Authors**: Stepan Dergachev, Konstantin Yakovlev  

**Link**: [PDF](https://arxiv.org/pdf/2507.20293)  

**Abstract**: Decentralized multi-agent navigation under uncertainty is a complex task that arises in numerous robotic applications. It requires collision avoidance strategies that account for both kinematic constraints, sensing and action execution noise. In this paper, we propose a novel approach that integrates the Model Predictive Path Integral (MPPI) with a probabilistic adaptation of Optimal Reciprocal Collision Avoidance. Our method ensures safe and efficient multi-agent navigation by incorporating probabilistic safety constraints directly into the MPPI sampling process via a Second-Order Cone Programming formulation. This approach enables agents to operate independently using local noisy observations while maintaining safety guarantees. We validate our algorithm through extensive simulations with differential-drive robots and benchmark it against state-of-the-art methods, including ORCA-DD and B-UAVC. Results demonstrate that our approach outperforms them while achieving high success rates, even in densely populated environments. Additionally, validation in the Gazebo simulator confirms its practical applicability to robotic platforms. 

**Abstract (ZH)**: 在不确定性下的去中心化多代理导航是一个复杂的任务，广泛应用于各种机器人应用中。它需要考虑到运动约束、感测和动作执行噪声的碰撞避免策略。在本文中，我们提出了一种新型方法，将模型预测路径积分（MPPI）与最优相互碰撞避免的概率适应方法相结合。通过使用圆锥二次规划形式，我们的方法直接将概率安全约束集成到MPPI采样过程中，从而确保代理在利用局部噪声观测独立操作的同时保持安全保证。我们通过广泛的仿真验证了该算法，并将其与当今最先进的方法（包括ORCA-DD和B-UAVC）进行了基准测试。结果表明，我们的方法在不同环境中实现了更高的成功率，并且在Gazebo仿真器中的验证也证明了其在机器人平台上的实际适用性。 

---
# Humanoid Occupancy: Enabling A Generalized Multimodal Occupancy Perception System on Humanoid Robots 

**Title (ZH)**: 类人形占用感知：为类人机器人实现通用多模态占用感知系统 

**Authors**: Wei Cui, Haoyu Wang, Wenkang Qin, Yijie Guo, Gang Han, Wen Zhao, Jiahang Cao, Zhang Zhang, Jiaru Zhong, Jingkai Sun, Pihai Sun, Shuai Shi, Botuo Jiang, Jiahao Ma, Jiaxu Wang, Hao Cheng, Zhichao Liu, Yang Wang, Zheng Zhu, Guan Huang, Jian Tang, Qiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.20217)  

**Abstract**: Humanoid robot technology is advancing rapidly, with manufacturers introducing diverse heterogeneous visual perception modules tailored to specific scenarios. Among various perception paradigms, occupancy-based representation has become widely recognized as particularly suitable for humanoid robots, as it provides both rich semantic and 3D geometric information essential for comprehensive environmental understanding. In this work, we present Humanoid Occupancy, a generalized multimodal occupancy perception system that integrates hardware and software components, data acquisition devices, and a dedicated annotation pipeline. Our framework employs advanced multi-modal fusion techniques to generate grid-based occupancy outputs encoding both occupancy status and semantic labels, thereby enabling holistic environmental understanding for downstream tasks such as task planning and navigation. To address the unique challenges of humanoid robots, we overcome issues such as kinematic interference and occlusion, and establish an effective sensor layout strategy. Furthermore, we have developed the first panoramic occupancy dataset specifically for humanoid robots, offering a valuable benchmark and resource for future research and development in this domain. The network architecture incorporates multi-modal feature fusion and temporal information integration to ensure robust perception. Overall, Humanoid Occupancy delivers effective environmental perception for humanoid robots and establishes a technical foundation for standardizing universal visual modules, paving the way for the widespread deployment of humanoid robots in complex real-world scenarios. 

**Abstract (ZH)**: humano形机器人占用感知技术：一种通用多模态占用感知系统及其应用 

---
# CLASP: General-Purpose Clothes Manipulation with Semantic Keypoints 

**Title (ZH)**: CLASP: 基于语义关键点的一般服装操作方法 

**Authors**: Yuhong Deng, Chao Tang, Cunjun Yu, Linfeng Li, David Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2507.19983)  

**Abstract**: Clothes manipulation, such as folding or hanging, is a critical capability for home service robots. Despite recent advances, most existing methods remain limited to specific tasks and clothes types, due to the complex, high-dimensional geometry of clothes. This paper presents CLothes mAnipulation with Semantic keyPoints (CLASP), which aims at general-purpose clothes manipulation over different clothes types, T-shirts, shorts, skirts, long dresses, ... , as well as different tasks, folding, flattening, hanging, ... . The core idea of CLASP is semantic keypoints -- e.g., ''left sleeve'', ''right shoulder'', etc. -- a sparse spatial-semantic representation that is salient for both perception and action. Semantic keypoints of clothes can be reliably extracted from RGB-D images and provide an effective intermediate representation of clothes manipulation policies. CLASP uses semantic keypoints to bridge high-level task planning and low-level action execution. At the high level, it exploits vision language models (VLMs) to predict task plans over the semantic keypoints. At the low level, it executes the plans with the help of a simple pre-built manipulation skill library. Extensive simulation experiments show that CLASP outperforms state-of-the-art baseline methods on multiple tasks across diverse clothes types, demonstrating strong performance and generalization. Further experiments with a Franka dual-arm system on four distinct tasks -- folding, flattening, hanging, and placing -- confirm CLASP's performance on a real robot. 

**Abstract (ZH)**: Clothes 操作：基于语义关键点的通用家用服务机器人衣物操作方法 

---
# A roadmap for AI in robotics 

**Title (ZH)**: AI在机器人领域的应用 roadmap 

**Authors**: Aude Billard, Alin Albu-Schaeffer, Michael Beetz, Wolfram Burgard, Peter Corke, Matei Ciocarlie, Ravinder Dahiya, Danica Kragic, Ken Goldberg, Yukie Nagai, Davide Scaramuzza  

**Link**: [PDF](https://arxiv.org/pdf/2507.19975)  

**Abstract**: AI technologies, including deep learning, large-language models have gone from one breakthrough to the other. As a result, we are witnessing growing excitement in robotics at the prospect of leveraging the potential of AI to tackle some of the outstanding barriers to the full deployment of robots in our daily lives. However, action and sensing in the physical world pose greater and different challenges than analysing data in isolation. As the development and application of AI in robotic products advances, it is important to reflect on which technologies, among the vast array of network architectures and learning models now available in the AI field, are most likely to be successfully applied to robots; how they can be adapted to specific robot designs, tasks, environments; which challenges must be overcome. This article offers an assessment of what AI for robotics has achieved since the 1990s and proposes a short- and medium-term research roadmap listing challenges and promises. These range from keeping up-to-date large datasets, representatives of a diversity of tasks robots may have to perform, and of environments they may encounter, to designing AI algorithms tailored specifically to robotics problems but generic enough to apply to a wide range of applications and transfer easily to a variety of robotic platforms. For robots to collaborate effectively with humans, they must predict human behavior without relying on bias-based profiling. Explainability and transparency in AI-driven robot control are not optional but essential for building trust, preventing misuse, and attributing responsibility in accidents. We close on what we view as the primary long-term challenges, that is, to design robots capable of lifelong learning, while guaranteeing safe deployment and usage, and sustainable computational costs. 

**Abstract (ZH)**: AI技术，包括深度学习和大规模语言模型，已从一个突破接踵而至的另一个突破。随着人工智能潜力在解决机器人全面部署中遇到的一些障碍方面的展望，我们在机器人领域正见证着日益增长的热情。然而，物理世界的操作和感知比单独分析数据带来了更大的、不同的挑战。随着人工智能在机器人产品中的发展和应用，反思哪些网络架构和技术，在当前可供选择的广泛人工智能学习模型中，最有可能成功应用于机器人；如何适应特定的机器人设计、任务和环境；哪些挑战必须被克服，这很重要。本文评估了自20世纪90年代以来人工智能在机器人领域的成就，并提出了一份短期和中期的研究路线图，列出了一系列挑战和前景。这些从保持与机器人可能执行的各种任务和可能遇到的各种环境相代表的大规模数据集更新，到为机器人问题量身定制的AI算法，这些算法具有广泛的适用性和灵活的机器人平台过渡能力。为了使机器人有效协作，它们必须预测人类行为，而无需依赖基于偏见的画像。基于人工智能的机器人控制中的解释性与透明性不仅是可选的，而是构建信任、防止滥用和在事故中承担责任的关键。最后，我们着重讨论我们所认为的主要长期挑战，即设计能够终身学习的机器人，同时确保安全部署和使用，以及可持续的计算成本。 

---
# Spatial Language Likelihood Grounding Network for Bayesian Fusion of Human-Robot Observations 

**Title (ZH)**: 基于空间语言likelihood grounding的贝叶斯融合人类-机器人观测网络 

**Authors**: Supawich Sitdhipol, Waritwong Sukprasongdee, Ekapol Chuangsuwanich, Rina Tse  

**Link**: [PDF](https://arxiv.org/pdf/2507.19947)  

**Abstract**: Fusing information from human observations can help robots overcome sensing limitations in collaborative tasks. However, an uncertainty-aware fusion framework requires a grounded likelihood representing the uncertainty of human inputs. This paper presents a Feature Pyramid Likelihood Grounding Network (FP-LGN) that grounds spatial language by learning relevant map image features and their relationships with spatial relation semantics. The model is trained as a probability estimator to capture aleatoric uncertainty in human language using three-stage curriculum learning. Results showed that FP-LGN matched expert-designed rules in mean Negative Log-Likelihood (NLL) and demonstrated greater robustness with lower standard deviation. Collaborative sensing results demonstrated that the grounded likelihood successfully enabled uncertainty-aware fusion of heterogeneous human language observations and robot sensor measurements, achieving significant improvements in human-robot collaborative task performance. 

**Abstract (ZH)**: 融合人类观察信息可以幫助机器人在协作任务中克服传感限制。然而，一个具备不确定性的融合框架需要一个基于地面的likelihood来表示人类输入的不确定性。本文提出了一种特征金字塔可能性接地网络（FP-LGN），该网络通过学习与空间关系语义相关的地图图像特征及其关系来接地空间语言。模型利用三阶段 Curriculum 学习作为概率估计器，捕捉人类语言中的偶然不确定性。结果显示，FP-LGN 在平均负对数似然（NLL）上与专家设计的规则相匹配，并且表现出更低的标准差，从而提高了鲁棒性。协作感知结果表明，接地的似然性成功地使异质人类语言观察和机器人传感器测量的不确定性感知融合成为可能，显著提高了人机协作任务的性能。 

---
# Think, Act, Learn: A Framework for Autonomous Robotic Agents using Closed-Loop Large Language Models 

**Title (ZH)**: 思考、行动、学习：一种基于闭环大型语言模型的自主机器人框架 

**Authors**: Anjali R. Menon, Rohit K. Sharma, Priya Singh, Chengyu Wang, Aurora M. Ferreira, Mateja Novak  

**Link**: [PDF](https://arxiv.org/pdf/2507.19854)  

**Abstract**: The integration of Large Language Models (LLMs) into robotics has unlocked unprecedented capabilities in high-level task planning. However, most current systems operate in an open-loop fashion, where LLMs act as one-shot planners, rendering them brittle and unable to adapt to unforeseen circumstances in dynamic physical environments. To overcome this limitation, this paper introduces the "Think, Act, Learn" (T-A-L) framework, a novel architecture that enables an embodied agent to autonomously learn and refine its policies through continuous interaction. Our framework establishes a closed-loop cycle where an LLM first "thinks" by decomposing high-level commands into actionable plans. The robot then "acts" by executing these plans while gathering rich, multimodal sensory feedback. Critically, the "learn" module processes this feedback to facilitate LLM-driven self-reflection, allowing the agent to perform causal analysis on its failures and generate corrective strategies. These insights are stored in an experiential memory to guide future planning cycles. We demonstrate through extensive experiments in both simulation and the real world that our T-A-L agent significantly outperforms baseline methods, including open-loop LLMs, Behavioral Cloning, and traditional Reinforcement Learning. Our framework achieves over a 97% success rate on complex, long-horizon tasks, converges to a stable policy in an average of just 9 trials, and exhibits remarkable generalization to unseen tasks. This work presents a significant step towards developing more robust, adaptive, and truly autonomous robotic agents. 

**Abstract (ZH)**: Large Language Models Integrated into Robotics: The "Think, Act, Learn" Framework for Autonomous Policy Refinement 

---
# PlaneHEC: Efficient Hand-Eye Calibration for Multi-view Robotic Arm via Any Point Cloud Plane Detection 

**Title (ZH)**: PlaneHEC: 基于任意点云平面检测的多视点机器人手臂高效手眼标定 

**Authors**: Ye Wang, Haodong Jing, Yang Liao, Yongqiang Ma, Nanning Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.19851)  

**Abstract**: Hand-eye calibration is an important task in vision-guided robotic systems and is crucial for determining the transformation matrix between the camera coordinate system and the robot end-effector. Existing methods, for multi-view robotic systems, usually rely on accurate geometric models or manual assistance, generalize poorly, and can be very complicated and inefficient. Therefore, in this study, we propose PlaneHEC, a generalized hand-eye calibration method that does not require complex models and can be accomplished using only depth cameras, which achieves the optimal and fastest calibration results using arbitrary planar surfaces like walls and tables. PlaneHEC introduces hand-eye calibration equations based on planar constraints, which makes it strongly interpretable and generalizable. PlaneHEC also uses a comprehensive solution that starts with a closed-form solution and improves it withiterative optimization, which greatly improves accuracy. We comprehensively evaluated the performance of PlaneHEC in both simulated and real-world environments and compared the results with other point-cloud-based calibration methods, proving its superiority. Our approach achieves universal and fast calibration with an innovative design of computational models, providing a strong contribution to the development of multi-agent systems and embodied intelligence. 

**Abstract (ZH)**: 平面Hand-eye标定方法：PlaneHEC 

---
# Ag2x2: Robust Agent-Agnostic Visual Representations for Zero-Shot Bimanual Manipulation 

**Title (ZH)**: Ag2x2: 基于稳健的、代理无关的视觉表示的零样本双臂操作 

**Authors**: Ziyin Xiong, Yinghan Chen, Puhao Li, Yixin Zhu, Tengyu Liu, Siyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.19817)  

**Abstract**: Bimanual manipulation, fundamental to human daily activities, remains a challenging task due to its inherent complexity of coordinated control. Recent advances have enabled zero-shot learning of single-arm manipulation skills through agent-agnostic visual representations derived from human videos; however, these methods overlook crucial agent-specific information necessary for bimanual coordination, such as end-effector positions. We propose Ag2x2, a computational framework for bimanual manipulation through coordination-aware visual representations that jointly encode object states and hand motion patterns while maintaining agent-agnosticism. Extensive experiments demonstrate that Ag2x2 achieves a 73.5% success rate across 13 diverse bimanual tasks from Bi-DexHands and PerAct2, including challenging scenarios with deformable objects like ropes. This performance outperforms baseline methods and even surpasses the success rate of policies trained with expert-engineered rewards. Furthermore, we show that representations learned through Ag2x2 can be effectively leveraged for imitation learning, establishing a scalable pipeline for skill acquisition without expert supervision. By maintaining robust performance across diverse tasks without human demonstrations or engineered rewards, Ag2x2 represents a step toward scalable learning of complex bimanual robotic skills. 

**Abstract (ZH)**: 双臂操作，对于人类日常活动至关重要，但由于其固有的协调控制复杂性，仍然是一项具有挑战性的任务。近期进展通过从人类视频中提取的代理无关视觉表示使单臂操作技能实现了零样本学习；然而，这些方法忽略了对于双臂协调至关重要的代理特定信息，如末端执行器位置。我们提出了一种名为Ag2x2的计算框架，该框架通过感知代理意识的视觉表示同时编码物体状态和手部运动模式，同时保持代理无关性。大量的实验表明，Ag2x2在Bi-DexHands和PerAct2的13种不同双臂任务中取得了73.5%的成功率，包括涉及可变形物体（如绳索）的具有挑战性的场景。该性能超越了基线方法，并甚至超过了使用专家设计奖励训练的策略的成功率。此外，我们展示了通过Ag2x2学习的表示可以有效应用于模仿学习，建立了一种在无专家监督的情况下可扩展的技能获取管道。通过在不同任务中保持稳健的性能，无需人类示范或工程化奖励，Ag2x2代表了复杂双臂机器人技能可扩展学习的一个重要进展。 

---
# DOA: A Degeneracy Optimization Agent with Adaptive Pose Compensation Capability based on Deep Reinforcement Learning 

**Title (ZH)**: DOA：一种基于深度强化学习的自适应姿态补偿退化优化代理 

**Authors**: Yanbin Li, Canran Xiao, Hongyang He, Shenghai Yuan, Zong Ke, Jiajie Yu, Zixiong Qin, Zhiguo Zhang, Wenzheng Chi, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.19742)  

**Abstract**: Particle filter-based 2D-SLAM is widely used in indoor localization tasks due to its efficiency. However, indoor environments such as long straight corridors can cause severe degeneracy problems in SLAM. In this paper, we use Proximal Policy Optimization (PPO) to train an adaptive degeneracy optimization agent (DOA) to address degeneracy problem. We propose a systematic methodology to address three critical challenges in traditional supervised learning frameworks: (1) data acquisition bottlenecks in degenerate dataset, (2) inherent quality deterioration of training samples, and (3) ambiguity in annotation protocol design. We design a specialized reward function to guide the agent in developing perception capabilities for degenerate environments. Using the output degeneracy factor as a reference weight, the agent can dynamically adjust the contribution of different sensors to pose optimization. Specifically, the observation distribution is shifted towards the motion model distribution, with the step size determined by a linear interpolation formula related to the degeneracy factor. In addition, we employ a transfer learning module to endow the agent with generalization capabilities across different environments and address the inefficiency of training in degenerate environments. Finally, we conduct ablation studies to demonstrate the rationality of our model design and the role of transfer learning. We also compare the proposed DOA with SOTA methods to prove its superior degeneracy detection and optimization capabilities across various environments. 

**Abstract (ZH)**: 基于粒子滤波的2D-SLAM在室内定位任务中广泛应用，但由于其高效性。然而，如长直走廊等室内环境会导致SLAM严重退化问题。本文使用临近策略优化(PPO)训练自适应退化优化代理(DOA)来解决退化问题。我们提出了一种系统方法来应对传统监督学习框架中的三个关键挑战：(1)退化数据集的数据获取瓶颈，(2)训练样本固有的质量退化，以及(3)标注协议设计的模糊性。我们设计了专门的奖励函数来引导代理在退化环境中的感知能力发展。利用输出退化因子作为参考权重，代理可以动态调整不同传感器在姿态优化中的贡献。具体而言，观测分布朝向运动模型分布移动，步长由与退化因子相关的线性插值公式确定。此外，我们采用迁移学习模块赋予代理跨不同环境的泛化能力，并解决退化环境中的训练效率低下问题。最后，我们进行消融研究以证明我们模型设计的合理性及其迁移学习的作用。我们还将所提出的DOA与当前最先进的方法进行比较，以证明其在各种环境中的优越的退化检测和优化能力。 

---
# RAKOMO: Reachability-Aware K-Order Markov Path Optimization for Quadrupedal Loco-Manipulation 

**Title (ZH)**: RAKOMO：可达性感知的四元 Markov 路径优化方法用于四足Manipulation 

**Authors**: Mattia Risiglione, Abdelrahman Abdalla, Victor Barasuol, Kim Tien Ly, Ioannis Havoutis, Claudio Semini  

**Link**: [PDF](https://arxiv.org/pdf/2507.19652)  

**Abstract**: Legged manipulators, such as quadrupeds equipped with robotic arms, require motion planning techniques that account for their complex kinematic constraints in order to perform manipulation tasks both safely and effectively. However, trajectory optimization methods often face challenges due to the hybrid dynamics introduced by contact discontinuities, and tend to neglect leg limitations during planning for computational reasons. In this work, we propose RAKOMO, a path optimization technique that integrates the strengths of K-Order Markov Optimization (KOMO) with a kinematically-aware criterion based on the reachable region defined as reachability margin. We leverage a neural-network to predict the margin and optimize it by incorporating it in the standard KOMO formulation. This approach enables rapid convergence of gradient-based motion planning -- commonly tailored for continuous systems -- while adapting it effectively to legged manipulators, successfully executing loco-manipulation tasks. We benchmark RAKOMO against a baseline KOMO approach through a set of simulations for pick-and-place tasks with the HyQReal quadruped robot equipped with a Kinova Gen3 robotic arm. 

**Abstract (ZH)**: 装有机械臂的腿足 manipulator，如四足机器人，为了安全高效地执行操作任务，需要考虑其复杂的运动约束的动力学规划技术。然而，由于接触断点引入的混合动力学，轨迹优化方法往往会遇到挑战，并且在计算原因上倾向于在规划过程中忽视腿的限制。在这种背景下，我们提出了RAKOMO，一种结合了K-Order Markov Optimization (KOMO) 强点并与可达区域定义为基础的动力学感知标准相结合的方法。我们利用神经网络预测可达区域并将其纳入标准的KOMO公式中进行优化。这种方法能够使基于梯度的运动规划快速收敛，同时有效地适应腿足 manipulator，成功执行行进操作任务。我们通过一系列针对HyQReal四足机器人配以Kinova Gen3机械臂的放置任务仿真实验，将RAKOMO与基准KOMO方法进行对比评估。 

---
# Reward-Augmented Reinforcement Learning for Continuous Control in Precision Autonomous Parking via Policy Optimization Methods 

**Title (ZH)**: 基于策略优化方法的精确自主停车连续控制的奖励增强强化学习 

**Authors**: Ahmad Suleman, Misha Urooj Khan, Zeeshan Kaleem, Ali H. Alenezi, Iqra Shabbir Sinem Coleri, Chau Yuen  

**Link**: [PDF](https://arxiv.org/pdf/2507.19642)  

**Abstract**: Autonomous parking (AP) represents a critical yet complex subset of intelligent vehicle automation, characterized by tight spatial constraints, frequent close-range obstacle interactions, and stringent safety margins. However, conventional rule-based and model-predictive methods often lack the adaptability and generalization needed to handle the nonlinear and environment-dependent complexities of AP. To address these limitations, we propose a reward-augmented learning framework for AP (RARLAP), that mitigates the inherent complexities of continuous-domain control by leveraging structured reward design to induce smooth and adaptable policy behavior, trained entirely within a high-fidelity Unity-based custom 3D simulation environment. We systematically design and assess three structured reward strategies: goal-only reward (GOR), dense proximity reward (DPR), and milestone-augmented reward (MAR), each integrated with both on-policy and off-policy optimization paradigms. Empirical evaluations demonstrate that the on-policy MAR achieves a 91\% success rate, yielding smoother trajectories and more robust behavior, while GOR and DPR fail to guide effective learning. Convergence and trajectory analyses demonstrate that the proposed framework enhances policy adaptability, accelerates training, and improves safety in continuous control. Overall, RARLAP establishes that reward augmentation effectively addresses complex autonomous parking challenges, enabling scalable and efficient policy optimization with both on- and off-policy methods. To support reproducibility, the code accompanying this paper is publicly available. 

**Abstract (ZH)**: 自主泊车(Autonomous Parking, AP)代表了一种关键但复杂的智能车辆自动化子集，其特征包括严格的 spatial 约束、频繁的近距离障碍交互以及严格的安全裕度。然而，传统的基于规则的方法和模型预测方法往往缺乏处理 AP 非线性和环境依赖性复杂性的适应性和泛化能力。为了解决这些限制，我们提出了一种增强奖励的学习框架（Reward-Augmented Learning Framework for Autonomous Parking, RARLAP），该框架通过利用结构化奖励设计来减轻连续域控制的内在复杂性，并在高保真度的 Unity 基础自定义 3D 仿真环境中完全训练，诱导平稳且适应性强的策略行为。我们系统地设计并评估了三种结构化奖励策略：仅目标奖励（Goal-Only Reward, GOR）、密集距离奖励（Dense Proximity Reward, DPR）以及里程碑增强奖励（Milestone-Augmented Reward, MAR），每种策略都与在线策略优化和离线策略优化范式相结合。实证评估表明，仅目标增强奖励（on-policy MAR）实现了 91% 的成功率，产生更平稳的轨迹和更稳健的行为，而 GOR 和 DPR 未能引导有效的学习。收敛性和轨迹分析表明，所提出的框架增强了策略适应性，加速了训练，并提高了连续控制中的安全性。总体而言，RARLAP 证实了奖励增强有效地解决了复杂的自主泊车挑战，使得使用在线和离线方法进行可扩展且高效的策略优化成为可能。为了支持可重复性，与本文配套的代码已公开。 

---
# VLMPlanner: Integrating Visual Language Models with Motion Planning 

**Title (ZH)**: VLMPlanner：将视觉语言模型集成到运动规划中 

**Authors**: Zhipeng Tang, Sha Zhang, Jiajun Deng, Chenjie Wang, Guoliang You, Yuting Huang, Xinrui Lin, Yanyong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.20342)  

**Abstract**: Integrating large language models (LLMs) into autonomous driving motion planning has recently emerged as a promising direction, offering enhanced interpretability, better controllability, and improved generalization in rare and long-tail scenarios. However, existing methods often rely on abstracted perception or map-based inputs, missing crucial visual context, such as fine-grained road cues, accident aftermath, or unexpected obstacles, which are essential for robust decision-making in complex driving environments. To bridge this gap, we propose VLMPlanner, a hybrid framework that combines a learning-based real-time planner with a vision-language model (VLM) capable of reasoning over raw images. The VLM processes multi-view images to capture rich, detailed visual information and leverages its common-sense reasoning capabilities to guide the real-time planner in generating robust and safe trajectories. Furthermore, we develop the Context-Adaptive Inference Gate (CAI-Gate) mechanism that enables the VLM to mimic human driving behavior by dynamically adjusting its inference frequency based on scene complexity, thereby achieving an optimal balance between planning performance and computational efficiency. We evaluate our approach on the large-scale, challenging nuPlan benchmark, with comprehensive experimental results demonstrating superior planning performance in scenarios with intricate road conditions and dynamic elements. Code will be available. 

**Abstract (ZH)**: 将大型语言模型（LLMs）集成到自主驾驶运动规划中， recently emerged as a promising direction，提供增强的可解释性、更好的可控性和在稀有和长尾场景中更好的泛化能力。然而，现有方法通常依赖于抽象的感知或基于地图的输入，缺乏关键的视觉上下文，如精细的道路提示、事故后果或意想不到的障碍物，这些都是在复杂驾驶环境中实现稳健决策所必需的。为了解决这一问题，我们提出了一种名为VLMPlanner的混合框架，该框架结合了一个基于学习的实时规划器和能够对原始图像进行推理的视觉语言模型（VLM）。VLM处理多视角图像以捕捉丰富的详细视觉信息，并利用其常识推理能力指导实时规划器生成稳健和安全的轨迹。此外，我们开发了上下文自适应推断门控（CAI-Gate）机制，该机制使VLM能够根据场景复杂度动态调整其推理频率，从而在规划性能与计算效率之间实现最优平衡。我们在大规模、具有挑战性的nuPlan基准上评估了我们的方法，全面的实验结果表明，该方法在复杂道路条件和动态元素场景中的规划性能更优。代码将开源。 

---
# MMGraphRAG: Bridging Vision and Language with Interpretable Multimodal Knowledge Graphs 

**Title (ZH)**: MMGraphRAG: 通过可解释的多模态知识图谱连接视觉与语言 

**Authors**: Xueyao Wan, Hang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.20804)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances language model generation by retrieving relevant information from external knowledge bases. However, conventional RAG methods face the issue of missing multimodal information. Multimodal RAG methods address this by fusing images and text through mapping them into a shared embedding space, but they fail to capture the structure of knowledge and logical chains between modalities. Moreover, they also require large-scale training for specific tasks, resulting in limited generalizing ability. To address these limitations, we propose MMGraphRAG, which refines visual content through scene graphs and constructs a multimodal knowledge graph (MMKG) in conjunction with text-based KG. It employs spectral clustering to achieve cross-modal entity linking and retrieves context along reasoning paths to guide the generative process. Experimental results show that MMGraphRAG achieves state-of-the-art performance on the DocBench and MMLongBench datasets, demonstrating strong domain adaptability and clear reasoning paths. 

**Abstract (ZH)**: 基于场景图的多模态图增强生成（MMGraphRAG）通过 refine 视觉内容并结合基于文本的知识图构建多模态知识图，以解决常规 Retrieval-Augmented Generation (RAG) 方法中存在的多模态信息缺失问题。 

---
# Concept Learning for Cooperative Multi-Agent Reinforcement Learning 

**Title (ZH)**: 合作多智能体强化学习中的概念学习 

**Authors**: Zhonghan Ge, Yuanyang Zhu, Chunlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.20143)  

**Abstract**: Despite substantial progress in applying neural networks (NN) to multi-agent reinforcement learning (MARL) areas, they still largely suffer from a lack of transparency and interoperability. However, its implicit cooperative mechanism is not yet fully understood due to black-box networks. In this work, we study an interpretable value decomposition framework via concept bottleneck models, which promote trustworthiness by conditioning credit assignment on an intermediate level of human-like cooperation concepts. To address this problem, we propose a novel value-based method, named Concepts learning for Multi-agent Q-learning (CMQ), that goes beyond the current performance-vs-interpretability trade-off by learning interpretable cooperation concepts. CMQ represents each cooperation concept as a supervised vector, as opposed to existing models where the information flowing through their end-to-end mechanism is concept-agnostic. Intuitively, using individual action value conditioning on global state embeddings to represent each concept allows for extra cooperation representation capacity. Empirical evaluations on the StarCraft II micromanagement challenge and level-based foraging (LBF) show that CMQ achieves superior performance compared with the state-of-the-art counterparts. The results also demonstrate that CMQ provides more cooperation concept representation capturing meaningful cooperation modes, and supports test-time concept interventions for detecting potential biases of cooperation mode and identifying spurious artifacts that impact cooperation. 

**Abstract (ZH)**: 基于概念瓶颈模型的可解释价值分解框架：促进多agent Q学习中的透明度和互操作性 

---
# Minding Motivation: The Effect of Intrinsic Motivation on Agent Behaviors 

**Title (ZH)**: 关注动机：内在动机对面剂行为的影响 

**Authors**: Leonardo Villalobos-Arias, Grant Forbes, Jianxun Wang, David L Roberts, Arnav Jhala  

**Link**: [PDF](https://arxiv.org/pdf/2507.19725)  

**Abstract**: Games are challenging for Reinforcement Learning~(RL) agents due to their reward-sparsity, as rewards are only obtainable after long sequences of deliberate actions. Intrinsic Motivation~(IM) methods -- which introduce exploration rewards -- are an effective solution to reward-sparsity. However, IM also causes an issue known as `reward hacking' where the agent optimizes for the new reward at the expense of properly playing the game. The larger problem is that reward hacking itself is largely unknown; there is no answer to whether, and to what extent, IM rewards change the behavior of RL agents. This study takes a first step by empirically evaluating the impact on behavior of three IM techniques on the MiniGrid game-like environment. We compare these IM models with Generalized Reward Matching~(GRM), a method that can be used with any intrinsic reward function to guarantee optimality. Our results suggest that IM causes noticeable change by increasing the initial rewards, but also altering the way the agent plays; and that GRM mitigated reward hacking in some scenarios. 

**Abstract (ZH)**: 游戏由于奖励稀疏性对强化学习（RL）代理构成挑战，奖励仅在经过一系列故意动作后方可获得。内在动机（IM）方法——通过引入探索奖励——是解决奖励稀疏性的一种有效方案。然而，IM也会引起一种被称为“奖励劫持”的问题，代理会优化新奖励而代价是未能正确地玩游戏。更大的问题是，奖励劫持本身 largely unknown；目前尚无答案来确定IM奖励是否以及在多大程度上改变了RL代理的行为。本研究通过实证评估三种IM技术对MiniGrid游戏环境的影响，首次对此进行探索。我们将这些IM模型与通用奖励匹配（GRM）方法进行了对比，GRM是一种可以与任何内在奖励函数结合使用以确保最优性的方法。我们的结果显示，IM通过增加初始奖励引起了显著变化，同时也改变了代理的玩法；而在某些场景下，GRM减轻了奖励劫持的问题。 

---
# Multi-Masked Querying Network for Robust Emotion Recognition from Incomplete Multi-Modal Physiological Signals 

**Title (ZH)**: 基于不完备多模态生理信号的鲁棒情绪识别的多掩码查询网络 

**Authors**: Geng-Xin Xu, Xiang Zuo, Ye Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.20737)  

**Abstract**: Emotion recognition from physiological data is crucial for mental health assessment, yet it faces two significant challenges: incomplete multi-modal signals and interference from body movements and artifacts. This paper presents a novel Multi-Masked Querying Network (MMQ-Net) to address these issues by integrating multiple querying mechanisms into a unified framework. Specifically, it uses modality queries to reconstruct missing data from incomplete signals, category queries to focus on emotional state features, and interference queries to separate relevant information from noise. Extensive experiment results demonstrate the superior emotion recognition performance of MMQ-Net compared to existing approaches, particularly under high levels of data incompleteness. 

**Abstract (ZH)**: 从生理数据中识别情绪对于心理健康评估至关重要，但面临着两大挑战：Incomplete Multi-modal Signals和身体运动及噪声干扰。本文提出了一种新型的多掩码查询网络（MMQ-Net）以解决这些问题，通过将多种查询机制整合到统一框架中。具体而言，它使用模态查询来从不完整信号中重构缺失数据，类别查询来聚焦于情绪状态特征，以及干扰查询来分离相关信息与噪声。广泛实验结果表明，MMQ-Net在数据不完整性较高时的情绪识别性能显著优于现有方法。 

---
# DmC: Nearest Neighbor Guidance Diffusion Model for Offline Cross-domain Reinforcement Learning 

**Title (ZH)**: DmC: 基于最近邻指导的离线跨域强化学习扩散模型 

**Authors**: Linh Le Pham Van, Minh Hoang Nguyen, Duc Kieu, Hung Le, Hung The Tran, Sunil Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2507.20499)  

**Abstract**: Cross-domain offline reinforcement learning (RL) seeks to enhance sample efficiency in offline RL by utilizing additional offline source datasets. A key challenge is to identify and utilize source samples that are most relevant to the target domain. Existing approaches address this challenge by measuring domain gaps through domain classifiers, target transition dynamics modeling, or mutual information estimation using contrastive loss. However, these methods often require large target datasets, which is impractical in many real-world scenarios. In this work, we address cross-domain offline RL under a limited target data setting, identifying two primary challenges: (1) Dataset imbalance, which is caused by large source and small target datasets and leads to overfitting in neural network-based domain gap estimators, resulting in uninformative measurements; and (2) Partial domain overlap, where only a subset of the source data is closely aligned with the target domain. To overcome these issues, we propose DmC, a novel framework for cross-domain offline RL with limited target samples. Specifically, DmC utilizes $k$-nearest neighbor ($k$-NN) based estimation to measure domain proximity without neural network training, effectively mitigating overfitting. Then, by utilizing this domain proximity, we introduce a nearest-neighbor-guided diffusion model to generate additional source samples that are better aligned with the target domain, thus enhancing policy learning with more effective source samples. Through theoretical analysis and extensive experiments in diverse MuJoCo environments, we demonstrate that DmC significantly outperforms state-of-the-art cross-domain offline RL methods, achieving substantial performance gains. 

**Abstract (ZH)**: 跨域离线强化学习（DmC）：在有限目标样本下的方法 

---
# Cultivating Helpful, Personalized, and Creative AI Tutors: A Framework for Pedagogical Alignment using Reinforcement Learning 

**Title (ZH)**: 培养有帮助、个性化和创造性的AI导师：一种基于强化学习的教学对齐框架 

**Authors**: Siyu Song, Wentao Liu, Ye Lu, Ruohua Zhang, Tao Liu, Jinze Lv, Xinyun Wang, Aimin Zhou, Fei Tan, Bo Jiang, Hao Hao  

**Link**: [PDF](https://arxiv.org/pdf/2507.20335)  

**Abstract**: The integration of large language models (LLMs) into education presents unprecedented opportunities for scalable personalized learning. However, standard LLMs often function as generic information providers, lacking alignment with fundamental pedagogical principles such as helpfulness, student-centered personalization, and creativity cultivation. To bridge this gap, we propose EduAlign, a novel framework designed to guide LLMs toward becoming more effective and responsible educational assistants. EduAlign consists of two main stages. In the first stage, we curate a dataset of 8k educational interactions and annotate them-both manually and automatically-along three key educational dimensions: Helpfulness, Personalization, and Creativity (HPC). These annotations are used to train HPC-RM, a multi-dimensional reward model capable of accurately scoring LLM outputs according to these educational principles. We further evaluate the consistency and reliability of this reward model. In the second stage, we leverage HPC-RM as a reward signal to fine-tune a pre-trained LLM using Group Relative Policy Optimization (GRPO) on a set of 2k diverse prompts. We then assess the pre- and post-finetuning models on both educational and general-domain benchmarks across the three HPC dimensions. Experimental results demonstrate that the fine-tuned model exhibits significantly improved alignment with pedagogical helpfulness, personalization, and creativity stimulation. This study presents a scalable and effective approach to aligning LLMs with nuanced and desirable educational traits, paving the way for the development of more engaging, pedagogically aligned AI tutors. 

**Abstract (ZH)**: 大型语言模型（LLMs）融入教育呈现了前所未有的个性化学习 scalability 机会，然而标准的LLMs通常作为通用信息提供者运作，缺乏与教学基本原则如有益性、学生中心化个性化和创造力培养的契合。为弥合这一差距，我们提出EduAlign，这是一种新型框架，旨在引导LLMs成为更有效且负责任的教育助手。EduAlign包括两个主要阶段。在第一阶段，我们编纂了一个包含8000个教育交互的数据集，并手动和自动地对其进行三个关键教育维度的帮助性、个性化和创造力（HPC）的标注。这些标注用于训练HPC-RM，这是一种多维度奖励模型，能够精确评分LLMs输出，依据这些教育原则。我们进一步评估了该奖励模型的一致性和可靠性。在第二阶段，我们利用HPC-RM作为奖励信号，通过组相对策略优化（GRPO）对预训练的LLMs进行微调，使用2000个多样性的提示。然后，我们在三个HPC维度上的教育和通用领域基准上评估了微调前后的模型。实验结果表明，微调后的模型在教学上有明显改进的契合度，个性化和创造力的激发也得到提升。本研究提出了一种可扩展且有效的方法，将LLMs与精细化且可取的教育特质对齐，为开发更具吸引力且教学目标对齐的人工智能导师铺平了道路。 

---
# LRR-Bench: Left, Right or Rotate? Vision-Language models Still Struggle With Spatial Understanding Tasks 

**Title (ZH)**: LRR-Bench: 左、右还是旋转？视觉-语言模型在空间理解任务上仍然挣扎 

**Authors**: Fei Kong, Jinhao Duan, Kaidi Xu, Zhenhua Guo, Xiaofeng Zhu, Xiaoshuang Shi  

**Link**: [PDF](https://arxiv.org/pdf/2507.20174)  

**Abstract**: Real-world applications, such as autonomous driving and humanoid robot manipulation, require precise spatial perception. However, it remains underexplored how Vision-Language Models (VLMs) recognize spatial relationships and perceive spatial movement. In this work, we introduce a spatial evaluation pipeline and construct a corresponding benchmark. Specifically, we categorize spatial understanding into two main types: absolute spatial understanding, which involves querying the absolute spatial position (e.g., left, right) of an object within an image, and 3D spatial understanding, which includes movement and rotation. Notably, our dataset is entirely synthetic, enabling the generation of test samples at a low cost while also preventing dataset contamination. We conduct experiments on multiple state-of-the-art VLMs and observe that there is significant room for improvement in their spatial understanding abilities. Explicitly, in our experiments, humans achieve near-perfect performance on all tasks, whereas current VLMs attain human-level performance only on the two simplest tasks. For the remaining tasks, the performance of VLMs is distinctly lower than that of humans. In fact, the best-performing Vision-Language Models even achieve near-zero scores on multiple tasks. The dataset and code are available on this https URL. 

**Abstract (ZH)**: 现实世界的应用，如自主驾驶和类人机器人操作，需要精确的空间感知。然而，VLMs如何识别空间关系和感知空间运动仍然未被充分探索。本文引入一个空间评估管道并构建相应的基准。具体而言，我们将空间理解分为两大类：绝对空间理解，涉及查询图像中物体的绝对空间位置（如左、右），以及三维空间理解，包括运动和旋转。值得注意的是，我们的数据集完全是合成的，这不仅降低了测试样本的生成成本，还防止了数据集的污染。我们在多个最先进的VLMs上进行了实验，发现它们的空间理解能力有很大的提升空间。具体而言，在我们的实验中，人类在所有任务上均接近完美表现，而当前的VLMs仅在两个最简单的任务上达到了人类水平的表现。对于剩余的任务，VLMs的表现明显低于人类。事实上，表现最佳的Vision-Language模型甚至在多个任务上获得了接近零的得分。数据集和代码可在此处访问：https://this-url.com。 

---
# Anomaly Detection in Human Language via Meta-Learning: A Few-Shot Approach 

**Title (ZH)**: 基于元学习的少样本人类语言异常检测 

**Authors**: Saurav Singla, Aarav Singla, Advik Gupta, Parnika Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2507.20019)  

**Abstract**: We propose a meta learning framework for detecting anomalies in human language across diverse domains with limited labeled data. Anomalies in language ranging from spam and fake news to hate speech pose a major challenge due to their sparsity and variability. We treat anomaly detection as a few shot binary classification problem and leverage meta-learning to train models that generalize across tasks. Using datasets from domains such as SMS spam, COVID-19 fake news, and hate speech, we evaluate model generalization on unseen tasks with minimal labeled anomalies. Our method combines episodic training with prototypical networks and domain resampling to adapt quickly to new anomaly detection tasks. Empirical results show that our method outperforms strong baselines in F1 and AUC scores. We also release the code and benchmarks to facilitate further research in few-shot text anomaly detection. 

**Abstract (ZH)**: 一种基于元学习的跨领域有限标注数据异常检测框架及应用 

---
# Salsa as a Nonverbal Embodied Language -- The CoMPAS3D Dataset and Benchmarks 

**Title (ZH)**: 萨萨拉作为非言语 embodied 语言——CoMPAS3D 数据集与基准 

**Authors**: Bermet Burkanova, Payam Jome Yazdian, Chuxuan Zhang, Trinity Evans, Paige Tuttösí, Angelica Lim  

**Link**: [PDF](https://arxiv.org/pdf/2507.19684)  

**Abstract**: Imagine a humanoid that can safely and creatively dance with a human, adapting to its partner's proficiency, using haptic signaling as a primary form of communication. While today's AI systems excel at text or voice-based interaction with large language models, human communication extends far beyond text-it includes embodied movement, timing, and physical coordination. Modeling coupled interaction between two agents poses a formidable challenge: it is continuous, bidirectionally reactive, and shaped by individual variation. We present CoMPAS3D, the largest and most diverse motion capture dataset of improvised salsa dancing, designed as a challenging testbed for interactive, expressive humanoid AI. The dataset includes 3 hours of leader-follower salsa dances performed by 18 dancers spanning beginner, intermediate, and professional skill levels. For the first time, we provide fine-grained salsa expert annotations, covering over 2,800 move segments, including move types, combinations, execution errors and stylistic elements. We draw analogies between partner dance communication and natural language, evaluating CoMPAS3D on two benchmark tasks for synthetic humans that parallel key problems in spoken language and dialogue processing: leader or follower generation with proficiency levels (speaker or listener synthesis), and duet (conversation) generation. Towards a long-term goal of partner dance with humans, we release the dataset, annotations, and code, along with a multitask SalsaAgent model capable of performing all benchmark tasks, alongside additional baselines to encourage research in socially interactive embodied AI and creative, expressive humanoid motion generation. 

**Abstract (ZH)**: 一种用于人类与类人机器人安全创造性舞蹈交互的触觉信号传输方法及CoMPAS3D即兴萨尔萨舞动捕获数据集 

---
# Efficient and Scalable Agentic AI with Heterogeneous Systems 

**Title (ZH)**: 高效的可扩展代理人工智能系统 

**Authors**: Zain Asgar, Michelle Nguyen, Sachin Katti  

**Link**: [PDF](https://arxiv.org/pdf/2507.19635)  

**Abstract**: AI agents are emerging as a dominant workload in a wide range of applications, promising to be the vehicle that delivers the promised benefits of AI to enterprises and consumers. Unlike conventional software or static inference, agentic workloads are dynamic and structurally complex. Often these agents are directed graphs of compute and IO operations that span multi-modal data input and conversion), data processing and context gathering (e.g vector DB lookups), multiple LLM inferences, tool calls, etc. To scale AI agent usage, we need efficient and scalable deployment and agent-serving infrastructure.
To tackle this challenge, in this paper, we present a system design for dynamic orchestration of AI agent workloads on heterogeneous compute infrastructure spanning CPUs and accelerators, both from different vendors and across different performance tiers within a single vendor. The system delivers several building blocks: a framework for planning and optimizing agentic AI execution graphs using cost models that account for compute, memory, and bandwidth constraints of different HW; a MLIR based representation and compilation system that can decompose AI agent execution graphs into granular operators and generate code for different HW options; and a dynamic orchestration system that can place the granular components across a heterogeneous compute infrastructure and stitch them together while meeting an end-to-end SLA. Our design performs a systems level TCO optimization and preliminary results show that leveraging a heterogeneous infrastructure can deliver significant TCO benefits. A preliminary surprising finding is that for some workloads a heterogeneous combination of older generation GPUs with newer accelerators can deliver similar TCO as the latest generation homogenous GPU infrastructure design, potentially extending the life of deployed infrastructure. 

**Abstract (ZH)**: AI代理正在成为广泛应用场景中的主导工作负载，有望为企业和消费者带来人工智能承诺的好处。与传统的软件或静态推理不同，代理工作负载是动态且结构复杂的。这些代理通常是由计算和IO操作组成的有向图，跨多模态数据输入和转换、数据处理和上下文收集（例如向量数据库查找），以及多轮语言模型推理、工具调用等。为了扩展AI代理的使用，我们需要高效的可扩展部署和代理服务基础设施。

为应对这一挑战，本文提出了一种系统设计，用于在异构计算基础设施上动态编排跨不同供应商的CPU和加速器（包括单个供应商内的不同性能级别）的AI代理工作负载。该系统提供了一系列构建块：用于使用成本模型规划和优化代理AI执行图的框架，该模型考虑了不同硬件的计算、内存和带宽约束；基于MLIR的表示和编译系统，能够将AI代理执行图分解为粒度操作，并生成适合不同硬件选项的代码；以及一个动态编排系统，在异构计算基础设施上放置粒度组件，并在满足端到端SLA的同时将它们连接起来。我们的设计进行了一体化TCO优化，初步结果显示，采用异构基础设施可以带来显著的TCO优势。初步令人惊讶的发现是，对于某些工作负载，使用较老一代GPU与新一代加速器的混合组合可以提供与最新同质GPU基础设施设计相似的TCO，这可能延长了部署基础设施的使用寿命。 

---
# Simulating Human Behavior with the Psychological-mechanism Agent: Integrating Feeling, Thought, and Action 

**Title (ZH)**: 基于心理机制代理模拟人类行为：融合感觉、思考与行动 

**Authors**: Qing Dong, Pengyuan Liu, Dong Yu, Chen Kang  

**Link**: [PDF](https://arxiv.org/pdf/2507.19495)  

**Abstract**: Generative agents have made significant progress in simulating human behavior, but existing frameworks often simplify emotional modeling and focus primarily on specific tasks, limiting the authenticity of the simulation. Our work proposes the Psychological-mechanism Agent (PSYA) framework, based on the Cognitive Triangle (Feeling-Thought-Action), designed to more accurately simulate human behavior. The PSYA consists of three core modules: the Feeling module (using a layer model of affect to simulate changes in short-term, medium-term, and long-term emotions), the Thought module (based on the Triple Network Model to support goal-directed and spontaneous thinking), and the Action module (optimizing agent behavior through the integration of emotions, needs and plans). To evaluate the framework's effectiveness, we conducted daily life simulations and extended the evaluation metrics to self-influence, one-influence, and group-influence, selection five classic psychological experiments for simulation. The results show that the PSYA framework generates more natural, consistent, diverse, and credible behaviors, successfully replicating human experimental outcomes. Our work provides a richer and more accurate emotional and cognitive modeling approach for generative agents and offers an alternative to human participants in psychological experiments. 

**Abstract (ZH)**: 心理机制代理（PSYA）框架：基于认知三角形的生成代理情感与认知建模 

---
