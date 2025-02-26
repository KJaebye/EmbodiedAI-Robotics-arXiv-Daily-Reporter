# Enhancing Reusability of Learned Skills for Robot Manipulation via Gaze and Bottleneck 

**Title (ZH)**: 通过凝视和瓶颈增强机器人操作中学习技能的可重用性 

**Authors**: Ryo Takizawa, Izumi Karino, Koki Nakagawa, Yoshiyuki Ohmura, Yasuo Kuniyoshi  

**Link**: [PDF](https://arxiv.org/pdf/2502.18121)  

**Abstract**: Autonomous agents capable of diverse object manipulations should be able to acquire a wide range of manipulation skills with high reusability. Although advances in deep learning have made it increasingly feasible to replicate the dexterity of human teleoperation in robots, generalizing these acquired skills to previously unseen scenarios remains a significant challenge. In this study, we propose a novel algorithm, Gaze-based Bottleneck-aware Robot Manipulation (GazeBot), which enables high reusability of the learned motions even when the object positions and end-effector poses differ from those in the provided demonstrations. By leveraging gaze information and motion bottlenecks, both crucial features for object manipulation, GazeBot achieves high generalization performance compared with state-of-the-art imitation learning methods, without sacrificing its dexterity and reactivity. Furthermore, the training process of GazeBot is entirely data-driven once a demonstration dataset with gaze data is provided. Videos and code are available at this https URL. 

**Abstract (ZH)**: 能够执行多样化物体操作的自主代理应该能够获得广泛且高度可重用的操纵技能。尽管深度学习的进步使得在机器人上复制人类远程操作的灵巧性变得越来越可行，但将这些学到的技能推广到以前未见过的场景中仍然是一项重大挑战。在本研究中，我们提出了一种新型算法——基于视线的瓶颈感知机器人操纵（GazeBot），该算法能够在物体位置和末端执行器姿态与提供的示范不同的情况下，仍然实现高可重用性。通过利用视线信息和操纵瓶颈，GazeBot 在与最先进的模拟学习方法相比时，其泛化性能更高，同时不牺牲其灵巧性和反应性。此外，一旦提供了包含视线数据的示范数据集，GazeBot 的训练过程完全是数据驱动的。更多信息和代码请访问这个网址。 

---
# MRBTP: Efficient Multi-Robot Behavior Tree Planning and Collaboration 

**Title (ZH)**: MRBTP：高效多机器人行为树规划与协作 

**Authors**: Yishuai Cai, Xinglin Chen, Zhongxuan Cai, Yunxin Mao, Minglong Li, Wenjing Yang, Ji Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18072)  

**Abstract**: Multi-robot task planning and collaboration are critical challenges in robotics. While Behavior Trees (BTs) have been established as a popular control architecture and are plannable for a single robot, the development of effective multi-robot BT planning algorithms remains challenging due to the complexity of coordinating diverse action spaces. We propose the Multi-Robot Behavior Tree Planning (MRBTP) algorithm, with theoretical guarantees of both soundness and completeness. MRBTP features cross-tree expansion to coordinate heterogeneous actions across different BTs to achieve the team's goal. For homogeneous actions, we retain backup structures among BTs to ensure robustness and prevent redundant execution through intention sharing. While MRBTP is capable of generating BTs for both homogeneous and heterogeneous robot teams, its efficiency can be further improved. We then propose an optional plugin for MRBTP when Large Language Models (LLMs) are available to reason goal-related actions for each robot. These relevant actions can be pre-planned to form long-horizon subtrees, significantly enhancing the planning speed and collaboration efficiency of MRBTP. We evaluate our algorithm in warehouse management and everyday service scenarios. Results demonstrate MRBTP's robustness and execution efficiency under varying settings, as well as the ability of the pre-trained LLM to generate effective task-specific subtrees for MRBTP. 

**Abstract (ZH)**: 多机器人行为树规划（MRBTP）及其在仓库管理和日常服务场景中的应用 

---
# Multimodal Interaction and Intention Communication for Industrial Robots 

**Title (ZH)**: 工业机器人多模态交互与意图通信 

**Authors**: Tim Schreiter, Andrey Rudenko, Jens V. Rüppel, Martin Magnusson, Achim J. Lilienthal  

**Link**: [PDF](https://arxiv.org/pdf/2502.17971)  

**Abstract**: Successful adoption of industrial robots will strongly depend on their ability to safely and efficiently operate in human environments, engage in natural communication, understand their users, and express intentions intuitively while avoiding unnecessary distractions. To achieve this advanced level of Human-Robot Interaction (HRI), robots need to acquire and incorporate knowledge of their users' tasks and environment and adopt multimodal communication approaches with expressive cues that combine speech, movement, gazes, and other modalities. This paper presents several methods to design, enhance, and evaluate expressive HRI systems for non-humanoid industrial robots. We present the concept of a small anthropomorphic robot communicating as a proxy for its non-humanoid host, such as a forklift. We developed a multimodal and LLM-enhanced communication framework for this robot and evaluated it in several lab experiments, using gaze tracking and motion capture to quantify how users perceive the robot and measure the task progress. 

**Abstract (ZH)**: 工业机器人在人类环境中的成功应用将强烈依赖于其安全高效地操作、自然沟通、理解用户以及以直观方式表达意图的能力，同时避免不必要的干扰。为了实现这一高级水平的人机交互（HRI），机器人需要获取并整合用户任务和环境的知识，并采用结合语音、动作、目光等多种模态的表达性交互方法。本文提出了几种设计、增强和评估表达性HRI系统的办法。我们介绍了作为非类人工业机器人代理的小型拟人化机器人概念，并开发了一种多模态和大语言模型增强的通信框架，通过实验评估了该框架，使用眼动追踪和运动捕捉量化用户对机器人的感知并测量任务进度。 

---
# InVDriver: Intra-Instance Aware Vectorized Query-Based Autonomous Driving Transformer 

**Title (ZH)**: InVDriver: Awareness of 内存实例向量查询驱动自主驾驶变换器 

**Authors**: Bo Zhang, Heye Huang, Chunyang Liu, Yaqin Zhang, Zhenhua Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17949)  

**Abstract**: End-to-end autonomous driving with its holistic optimization capabilities, has gained increasing traction in academia and industry. Vectorized representations, which preserve instance-level topological information while reducing computational overhead, have emerged as a promising paradigm. While existing vectorized query-based frameworks often overlook the inherent spatial correlations among intra-instance points, resulting in geometrically inconsistent outputs (e.g., fragmented HD map elements or oscillatory trajectories). To address these limitations, we propose InVDriver, a novel vectorized query-based system that systematically models intra-instance spatial dependencies through masked self-attention layers, thereby enhancing planning accuracy and trajectory smoothness. Across all core modules, i.e., perception, prediction, and planning, InVDriver incorporates masked self-attention mechanisms that restrict attention to intra-instance point interactions, enabling coordinated refinement of structural elements while suppressing irrelevant inter-instance noise. Experimental results on the nuScenes benchmark demonstrate that InVDriver achieves state-of-the-art performance, surpassing prior methods in both accuracy and safety, while maintaining high computational efficiency. Our work validates that explicit modeling of intra-instance geometric coherence is critical for advancing vectorized autonomous driving systems, bridging the gap between theoretical advantages of end-to-end frameworks and practical deployment requirements. 

**Abstract (ZH)**: 端到端自主驾驶系统通过其整体优化能力，在学术界和工业界获得了越来越多的关注。矢量化表示在保留实例级拓扑信息的同时减少计算开销，已成为一个有前途的范式。尽管现有的基于矢量化查询的框架往往忽视了实例内部点之间的固有空间相关性，导致几何不一致的输出（例如，断裂的高精度地图元素或振荡轨迹）。为了解决这些限制，我们提出了一种新颖的基于矢量化查询的系统InVDriver，该系统通过掩码自注意力层系统地建模实例内部的空间依赖性，从而提高规划准确性和轨迹平滑度。在感知、预测和规划的所有核心模块中，InVDriver 集成了掩码自注意力机制，限制注意力仅关注实例内部点之间的交互，从而在抑制无关实例间噪声的同时实现结构元素的协调精化。在nuScenes基准测试上的实验结果表明，InVDriver 达到了最先进的性能，在准确性和安全性方面均超过了先前的方法，同时保持了高效的计算效率。我们的工作验证了明确建模实例内部几何一致性对于推动矢量化自主驾驶系统的发展至关重要，弥合了端到端框架理论优势与实际部署需求之间的差距。 

---
# FetchBot: Object Fetching in Cluttered Shelves via Zero-Shot Sim2Real 

**Title (ZH)**: FetchBot: 在杂乱货架上进行零样本Sim2Real对象抓取 

**Authors**: Weiheng Liu, Yuxuan Wan, Jilong Wang, Yuxuan Kuang, Xuesong Shi, Haoran Li, Dongbin Zhao, Zhizheng Zhang, He Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17894)  

**Abstract**: Object fetching from cluttered shelves is an important capability for robots to assist humans in real-world scenarios. Achieving this task demands robotic behaviors that prioritize safety by minimizing disturbances to surrounding objects, an essential but highly challenging requirement due to restricted motion space, limited fields of view, and complex object dynamics. In this paper, we introduce FetchBot, a sim-to-real framework designed to enable zero-shot generalizable and safety-aware object fetching from cluttered shelves in real-world settings. To address data scarcity, we propose an efficient voxel-based method for generating diverse simulated cluttered shelf scenes at scale and train a dynamics-aware reinforcement learning (RL) policy to generate object fetching trajectories within these scenes. This RL policy, which leverages oracle information, is subsequently distilled into a vision-based policy for real-world deployment. Considering that sim-to-real discrepancies stem from texture variations mostly while from geometric dimensions rarely, we propose to adopt depth information estimated by full-fledged depth foundation models as the input for the vision-based policy to mitigate sim-to-real gap. To tackle the challenge of limited views, we design a novel architecture for learning multi-view representations, allowing for comprehensive encoding of cluttered shelf scenes. This enables FetchBot to effectively minimize collisions while fetching objects from varying positions and depths, ensuring robust and safety-aware operation. Both simulation and real-robot experiments demonstrate FetchBot's superior generalization ability, particularly in handling a broad range of real-world scenarios, includ 

**Abstract (ZH)**: 从杂乱货架上抓取物体是机器人在实际场景中协助人类的重要能力。实现这一任务需要机器人行为优先确保安全，通过最小化对周围物体的干扰，这一要求由于受限的运动空间、有限的视野和复杂的物体动力学而变得至关重要。本文介绍了FetchBot，一个用于实现实用场景中杂乱货架上零样本泛化和安全意识物体抓取的从仿真到现实的框架。为了解决数据稀缺问题，我们提出了一种高效的方法，用于生成大规模多样的模拟杂乱货架场景，并训练一种动力学感知的强化学习（RL）策略来在这些场景中生成物体抓取轨迹。该RL策略利用先验信息，进而被提取为基于视觉的策略以部署到现实世界。考虑到仿真实验与现实之间的差异主要源自纹理变化而非几何尺寸变化，我们提出采用由完整深度基础模型估计的深度信息作为基于视觉策略的输入，以减轻仿真实验与现实的差距。为应对有限视野的挑战，我们设计了一种新的架构以学习多视角表示，使得可以全面编码杂乱货架场景。这使FetchBot能够在不同位置和深度抓取物体时有效减少碰撞，确保稳健和安全的操作。仿真实验和真实机器人实验均证明了FetchBot在处理各种实用场景中的优越泛化能力。 

---
# CAML: Collaborative Auxiliary Modality Learning for Multi-Agent Systems 

**Title (ZH)**: CAML：多agent系统中的协作辅助模态学习 

**Authors**: Rui Liu, Yu Shen, Peng Gao, Pratap Tokekar, Ming Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.17821)  

**Abstract**: Multi-modality learning has become a crucial technique for improving the performance of machine learning applications across domains such as autonomous driving, robotics, and perception systems. While existing frameworks such as Auxiliary Modality Learning (AML) effectively utilize multiple data sources during training and enable inference with reduced modalities, they primarily operate in a single-agent context. This limitation is particularly critical in dynamic environments, such as connected autonomous vehicles (CAV), where incomplete data coverage can lead to decision-making blind spots. To address these challenges, we propose Collaborative Auxiliary Modality Learning ($\textbf{CAML}$), a novel multi-agent multi-modality framework that enables agents to collaborate and share multimodal data during training while allowing inference with reduced modalities per agent during testing. We systematically analyze the effectiveness of $\textbf{CAML}$ from the perspective of uncertainty reduction and data coverage, providing theoretical insights into its advantages over AML. Experimental results in collaborative decision-making for CAV in accident-prone scenarios demonstrate that \ours~achieves up to a ${\bf 58.13}\%$ improvement in accident detection. Additionally, we validate $\textbf{CAML}$ on real-world aerial-ground robot data for collaborative semantic segmentation, achieving up to a ${\bf 10.61}\%$ improvement in mIoU. 

**Abstract (ZH)**: 协作辅助模态学习（CAML）：一种多agent多模态框架 

---
# Safe Multi-Agent Navigation guided by Goal-Conditioned Safe Reinforcement Learning 

**Title (ZH)**: 基于目标导向的安全强化学习引导的多智能体导航 

**Authors**: Meng Feng, Viraj Parimi, Brian Williams  

**Link**: [PDF](https://arxiv.org/pdf/2502.17813)  

**Abstract**: Safe navigation is essential for autonomous systems operating in hazardous environments. Traditional planning methods excel at long-horizon tasks but rely on a predefined graph with fixed distance metrics. In contrast, safe Reinforcement Learning (RL) can learn complex behaviors without relying on manual heuristics but fails to solve long-horizon tasks, particularly in goal-conditioned and multi-agent scenarios.
In this paper, we introduce a novel method that integrates the strengths of both planning and safe RL. Our method leverages goal-conditioned RL and safe RL to learn a goal-conditioned policy for navigation while concurrently estimating cumulative distance and safety levels using learned value functions via an automated self-training algorithm. By constructing a graph with states from the replay buffer, our method prunes unsafe edges and generates a waypoint-based plan that the agent follows until reaching its goal, effectively balancing faster and safer routes over extended distances.
Utilizing this unified high-level graph and a shared low-level goal-conditioned safe RL policy, we extend this approach to address the multi-agent safe navigation problem. In particular, we leverage Conflict-Based Search (CBS) to create waypoint-based plans for multiple agents allowing for their safe navigation over extended horizons. This integration enhances the scalability of goal-conditioned safe RL in multi-agent scenarios, enabling efficient coordination among agents.
Extensive benchmarking against state-of-the-art baselines demonstrates the effectiveness of our method in achieving distance goals safely for multiple agents in complex and hazardous environments. Our code will be released to support future research. 

**Abstract (ZH)**: 安全导航对于在危险环境中操作的自主系统至关重要。传统的规划方法在长期任务方面表现出色，但依赖于预先定义的具有固定距离度量的图。相比之下，安全的强化学习（RL）可以在不依赖手动启发式的情况下学习复杂的行为，但在解决长期任务方面尤其失败，特别是在目标条件和多agent场景中。
在本文中，我们提出了一种结合规划和安全RL优点的新方法。该方法利用目标条件RL和安全RL学习导航的目标条件策略，同时通过自训练算法利用学习的价值函数估计累积距离和安全水平。通过从回放缓冲区构建图，该方法删除不安全的边，生成agent遵循直至到达目标的基于路径点的计划，从而在较远的距离上实现更快更安全的路径。
利用这一统一的高层图和共享的目标条件安全RL策略，我们将此方法扩展以解决多agent安全导航问题。具体而言，我们利用冲突基搜索（CBS）为多个agent创建基于路径点的计划，使它们能够跨越长时间安全导航。这种集成增强了目标条件安全RL在多agent场景中的可扩展性，使agent之间能够实现高效的协调。
与最先进的基线方法的广泛基准测试显示，我们的方法在复杂和危险环境中能够有效地为多agent安全地实现距离目标。我们的代码将向未来的研究开放。 

---
# Toward 6-DOF Autonomous Underwater Vehicle Energy-Aware Position Control based on Deep Reinforcement Learning: Preliminary Results 

**Title (ZH)**: 基于深度强化学习的6自由度自主水下车辆能量感知位置控制：初步结果 

**Authors**: Gustavo Boré, Vicente Sufán, Sebastián Rodríguez-Martínez, Giancarlo Troni  

**Link**: [PDF](https://arxiv.org/pdf/2502.17742)  

**Abstract**: The use of autonomous underwater vehicles (AUVs) for surveying, mapping, and inspecting unexplored underwater areas plays a crucial role, where maneuverability and power efficiency are key factors for extending the use of these platforms, making six degrees of freedom (6-DOF) holonomic platforms essential tools. Although Proportional-Integral-Derivative (PID) and Model Predictive Control controllers are widely used in these applications, they often require accurate system knowledge, struggle with repeatability when facing payload or configuration changes, and can be time-consuming to fine-tune. While more advanced methods based on Deep Reinforcement Learning (DRL) have been proposed, they are typically limited to operating in fewer degrees of freedom. This paper proposes a novel DRL-based approach for controlling holonomic 6-DOF AUVs using the Truncated Quantile Critics (TQC) algorithm, which does not require manual tuning and directly feeds commands to the thrusters without prior knowledge of their configuration. Furthermore, it incorporates power consumption directly into the reward function. Simulation results show that the TQC High-Performance method achieves better performance to a fine-tuned PID controller when reaching a goal point, while the TQC Energy-Aware method demonstrates slightly lower performance but consumes 30% less power on average. 

**Abstract (ZH)**: 基于Truncated Quantile Critics算法的深度强化学习控制六自由度水下自主航行器方法 

---
# SET-PAiREd: Designing for Parental Involvement in Learning with an AI-Assisted Educational Robot 

**Title (ZH)**: SET-PAiREd: 设计一种基于AI辅助教育机器人的家长参与学习方法 

**Authors**: Hui-Ru Ho, Nitigya Kargeti, Ziqi Liu, Bilge Mutlu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17623)  

**Abstract**: AI-assisted learning companion robots are increasingly used in early education. Many parents express concerns about content appropriateness, while they also value how AI and robots could supplement their limited skill, time, and energy to support their children's learning. We designed a card-based kit, SET, to systematically capture scenarios that have different extents of parental involvement. We developed a prototype interface, PAiREd, with a learning companion robot to deliver LLM-generated educational content that can be reviewed and revised by parents. Parents can flexibly adjust their involvement in the activity by determining what they want the robot to help with. We conducted an in-home field study involving 20 families with children aged 3-5. Our work contributes to an empirical understanding of the level of support parents with different expectations may need from AI and robots and a prototype that demonstrates an innovative interaction paradigm for flexibly including parents in supporting their children. 

**Abstract (ZH)**: AI辅助的学习伴侣机器人在幼儿教育中的应用：父母参与程度的系统化捕捉及创新交互范式的原型设计 

---
# Learning Decentralized Swarms Using Rotation Equivariant Graph Neural Networks 

**Title (ZH)**: 使用旋转等变图神经网络学习去中心化 swarm 

**Authors**: Taos Transue, Bao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17612)  

**Abstract**: The orchestration of agents to optimize a collective objective without centralized control is challenging yet crucial for applications such as controlling autonomous fleets, and surveillance and reconnaissance using sensor networks. Decentralized controller design has been inspired by self-organization found in nature, with a prominent source of inspiration being flocking; however, decentralized controllers struggle to maintain flock cohesion. The graph neural network (GNN) architecture has emerged as an indispensable machine learning tool for developing decentralized controllers capable of maintaining flock cohesion, but they fail to exploit the symmetries present in flocking dynamics, hindering their generalizability. We enforce rotation equivariance and translation invariance symmetries in decentralized flocking GNN controllers and achieve comparable flocking control with 70% less training data and 75% fewer trainable weights than existing GNN controllers without these symmetries enforced. We also show that our symmetry-aware controller generalizes better than existing GNN controllers. Code and animations are available at this http URL. 

**Abstract (ZH)**: 无需协调中心控制的代理 orchestrating 以优化集体目标在自主车队控制和传感器网络监视与侦察等应用中具有挑战性但至关重要。缺乏中心控制的控制器设计受到自然界中自我组织现象的启发，其中飞行动物群集是最显著的灵感来源之一；然而，缺乏对群集凝聚力的维持。图神经网络架构已成为开发能够维持群集凝聚力的分散控制器不可或缺的机器学习工具，但它们未能利用群集动态中存在的对称性，阻碍了其泛化能力。我们强制执行旋转等变性和平移不变性对称性，在分散群集GNN控制器中实现类似性能的群集控制，所需训练数据减少70%，可训练参数减少75%，优于未强制执行这些对称性的现有GNN控制器。我们还展示了我们的对称性感知控制器比现有GNN控制器具有更好的泛化能力。代码和动画请访问此网址。 

---
# Self-Supervised Data Generation for Precision Agriculture: Blending Simulated Environments with Real Imagery 

**Title (ZH)**: 自监督数据生成在精准农业中的应用：模拟环境与实际影像的融合 

**Authors**: Leonardo Saraceni, Ionut Marian Motoi, Daniele Nardi, Thomas Alessandro Ciarfuglia  

**Link**: [PDF](https://arxiv.org/pdf/2502.18320)  

**Abstract**: In precision agriculture, the scarcity of labeled data and significant covariate shifts pose unique challenges for training machine learning models. This scarcity is particularly problematic due to the dynamic nature of the environment and the evolving appearance of agricultural subjects as living things. We propose a novel system for generating realistic synthetic data to address these challenges. Utilizing a vineyard simulator based on the Unity engine, our system employs a cut-and-paste technique with geometrical consistency considerations to produce accurate photo-realistic images and labels from synthetic environments to train detection algorithms. This approach generates diverse data samples across various viewpoints and lighting conditions. We demonstrate considerable performance improvements in training a state-of-the-art detector by applying our method to table grapes cultivation. The combination of techniques can be easily automated, an increasingly important consideration for adoption in agricultural practice. 

**Abstract (ZH)**: 在精准农业中，标注数据的稀缺性和显著的协变量转移为训练机器学习模型带来了独特的挑战。由于环境的动态性质和农业对象作为生物体的不断演变，这种稀缺性尤为成问题。我们提出了一种新颖的系统，用于生成现实主义的合成数据以应对这些挑战。基于Unity引擎的葡萄园模拟器，我们的系统采用带有几何一致性考虑的拼接技术，从合成环境中生成准确的逼真图像和标签，用于训练检测算法。这种方法能够在多种视角和光照条件下生成多样化的数据样本。我们将该方法应用于葡萄栽培，展示了对最先进的检测器训练性能的显著改进。该方法的结合可以轻松实现自动化，对于在农业实践中采用来说越来越重要。 

---
# OpenFly: A Versatile Toolchain and Large-scale Benchmark for Aerial Vision-Language Navigation 

**Title (ZH)**: OpenFly：一种多功能工具链及大规模benchmark用于航空视觉语言导航 

**Authors**: Yunpeng Gao, Chenhui Li, Zhongrui You, Junli Liu, Zhen Li, Pengan Chen, Qizhi Chen, Zhonghan Tang, Liansheng Wang, Penghui Yang, Yiwen Tang, Yuhang Tang, Shuai Liang, Songyi Zhu, Ziqin Xiong, Yifei Su, Xinyi Ye, Jianan Li, Yan Ding, Dong Wang, Zhigang Wang, Bin Zhao, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.18041)  

**Abstract**: Vision-Language Navigation (VLN) aims to guide agents through an environment by leveraging both language instructions and visual cues, playing a pivotal role in embodied AI. Indoor VLN has been extensively studied, whereas outdoor aerial VLN remains underexplored. The potential reason is that outdoor aerial view encompasses vast areas, making data collection more challenging, which results in a lack of benchmarks. To address this problem, we propose OpenFly, a platform comprising a versatile toolchain and large-scale benchmark for aerial VLN. Firstly, we develop a highly automated toolchain for data collection, enabling automatic point cloud acquisition, scene semantic segmentation, flight trajectory creation, and instruction generation. Secondly, based on the toolchain, we construct a large-scale aerial VLN dataset with 100k trajectories, covering diverse heights and lengths across 18 scenes. The corresponding visual data are generated using various rendering engines and advanced techniques, including Unreal Engine, GTA V, Google Earth, and 3D Gaussian Splatting (3D GS). All data exhibit high visual quality. Particularly, 3D GS supports real-to-sim rendering, further enhancing the realism of the dataset. Thirdly, we propose OpenFly-Agent, a keyframe-aware VLN model, which takes language instructions, current observations, and historical keyframes as input, and outputs flight actions directly. Extensive analyses and experiments are conducted, showcasing the superiority of our OpenFly platform and OpenFly-Agent. The toolchain, dataset, and codes will be open-sourced. 

**Abstract (ZH)**: 开放飞行：面向高空导航的多功能工具链和大规模基准平台 

---
# Intention Recognition in Real-Time Interactive Navigation Maps 

**Title (ZH)**: 实时交互导航地图中的意图识别 

**Authors**: Peijie Zhao, Zunayed Arefin, Felipe Meneguzzi, Ramon Fraga Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2502.17581)  

**Abstract**: In this demonstration, we develop IntentRec4Maps, a system to recognise users' intentions in interactive maps for real-world navigation. IntentRec4Maps uses the Google Maps Platform as the real-world interactive map, and a very effective approach for recognising users' intentions in real-time. We showcase the recognition process of IntentRec4Maps using two different Path-Planners and a Large Language Model (LLM).
GitHub: this https URL 

**Abstract (ZH)**: 在本次演示中，我们开发了IntentRec4Maps系统，用于识别用户在实境导航互动地图中的意图。IntentRec4Maps使用Google Maps Platform作为实境互动地图，并采用一种非常有效的实时识别用户意图的方法。我们使用两种不同的路径规划器和一个大型语言模型（LLM）展示了IntentRec4Maps的识别过程。GitHub: this https URL。 

---
# AgentRM: Enhancing Agent Generalization with Reward Modeling 

**Title (ZH)**: AgentRM：通过奖励建模提升智能体泛化能力 

**Authors**: Yu Xia, Jingru Fan, Weize Chen, Siyu Yan, Xin Cong, Zhong Zhang, Yaxi Lu, Yankai Lin, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.18407)  

**Abstract**: Existing LLM-based agents have achieved strong performance on held-in tasks, but their generalizability to unseen tasks remains poor. Hence, some recent work focus on fine-tuning the policy model with more diverse tasks to improve the generalizability. In this work, we find that finetuning a reward model to guide the policy model is more robust than directly finetuning the policy model. Based on this finding, we propose AgentRM, a generalizable reward model, to guide the policy model for effective test-time search. We comprehensively investigate three approaches to construct the reward model, including explicit reward modeling, implicit reward modeling and LLM-as-a-judge. We then use AgentRM to guide the answer generation with Best-of-N sampling and step-level beam search. On four types of nine agent tasks, AgentRM enhances the base policy model by $8.8$ points on average, surpassing the top general agent by $4.0$. Moreover, it demonstrates weak-to-strong generalization, yielding greater improvement of $12.6$ on LLaMA-3-70B policy model. As for the specializability, AgentRM can also boost a finetuned policy model and outperform the top specialized agent by $11.4$ on three held-in tasks. Further analysis verifies its effectiveness in test-time scaling. Codes will be released to facilitate the research in this area. 

**Abstract (ZH)**: 基于LLM的奖励模型AgentRM在未见任务上的泛化和专用化能力研究 

---
# Hierarchical Imitation Learning of Team Behavior from Heterogeneous Demonstrations 

**Title (ZH)**: 异质示范指导下分层模仿学习团队行为 

**Authors**: Sangwon Seo, Vaibhav Unhelkar  

**Link**: [PDF](https://arxiv.org/pdf/2502.17618)  

**Abstract**: Successful collaboration requires team members to stay aligned, especially in complex sequential tasks. Team members must dynamically coordinate which subtasks to perform and in what order. However, real-world constraints like partial observability and limited communication bandwidth often lead to suboptimal collaboration. Even among expert teams, the same task can be executed in multiple ways. To develop multi-agent systems and human-AI teams for such tasks, we are interested in data-driven learning of multimodal team behaviors. Multi-Agent Imitation Learning (MAIL) provides a promising framework for data-driven learning of team behavior from demonstrations, but existing methods struggle with heterogeneous demonstrations, as they assume that all demonstrations originate from a single team policy. Hence, in this work, we introduce DTIL: a hierarchical MAIL algorithm designed to learn multimodal team behaviors in complex sequential tasks. DTIL represents each team member with a hierarchical policy and learns these policies from heterogeneous team demonstrations in a factored manner. By employing a distribution-matching approach, DTIL mitigates compounding errors and scales effectively to long horizons and continuous state representations. Experimental results show that DTIL outperforms MAIL baselines and accurately models team behavior across a variety of collaborative scenarios. 

**Abstract (ZH)**: 成功的协作需要团队成员保持一致，特别是在复杂的序列任务中。团队成员必须动态协调执行哪些亚任务及其顺序。然而，现实世界中的部分可观测性和有限的通信带宽常导致协作效果不佳。即使是专家团队，相同的任务也可以采用多种方式执行。为了开发此类任务的多智能体系统和人机团队，我们对数据驱动的多模态团队行为学习感兴趣。多智能体模仿学习（MAIL）为从示范中学习团队行为的数据驱动学习提供了有前途的框架，但现有方法在处理异构示范时遇到了困难，因为它们假定所有示范都源自一个单一的团队策略。因此，在这项工作中，我们引入了DTIL：一种用于复杂序列任务中学习多模态团队行为的层次MAIL算法。DTIL通过层次政策表示每个团队成员，并以分解的方式从异构团队示范中学习这些政策。通过采用分布匹配方法，DTIL减轻了累积误差，并有效地扩展到长期限和连续状态表示中。实验结果显示，DTIL优于MAIL基线，并且能够准确地建模多种协作场景下的团队行为。 

---
# Training a Generally Curious Agent 

**Title (ZH)**: 训练一个普遍好奇的智能体 

**Authors**: Fahim Tajwar, Yiding Jiang, Abitha Thankaraj, Sumaita Sadia Rahman, J Zico Kolter, Jeff Schneider, Ruslan Salakhutdinov  

**Link**: [PDF](https://arxiv.org/pdf/2502.17543)  

**Abstract**: Efficient exploration is essential for intelligent systems interacting with their environment, but existing language models often fall short in scenarios that require strategic information gathering. In this paper, we present PAPRIKA, a fine-tuning approach that enables language models to develop general decision-making capabilities that are not confined to particular environments. By training on synthetic interaction data from different tasks that require diverse strategies, PAPRIKA teaches models to explore and adapt their behavior on a new task based on environment feedback in-context without more gradient updates. Experimental results show that models fine-tuned with PAPRIKA can effectively transfer their learned decision-making capabilities to entirely unseen tasks without additional training. Unlike traditional training, our approach's primary bottleneck lies in sampling useful interaction data instead of model updates. To improve sample efficiency, we propose a curriculum learning strategy that prioritizes sampling trajectories from tasks with high learning potential. These results suggest a promising path towards AI systems that can autonomously solve novel sequential decision-making problems that require interactions with the external world. 

**Abstract (ZH)**: 有效探索对于与环境互动的智能系统至关重要，但现有语言模型在需要策略性信息收集的场景中往往表现不佳。本文介绍了PAPRIKA，一种微调方法，使语言模型能够发展出不受特定环境限制的一般决策能力。通过在需要不同策略的多种任务中合成交互数据进行训练，PAPRIKA使模型能够根据环境反馈在新任务中探索并调整其行为，而无需更多梯度更新。实验结果表明，使用PAPRIKA微调的语言模型可以在无需额外训练的情况下有效地将学到的决策能力转移到全新的任务中。与传统训练相比，我们方法的主要瓶颈在于采样有用的交互数据而非模型更新。为了提高样本效率，我们提出了一种课程学习策略，优先从具有高学习潜力的任务中采样轨迹。这些结果指出了一个有前景的方向，即自主解决需要与外部世界互动的新型序列决策问题的AI系统。 

---
# A Survey on Mechanistic Interpretability for Multi-Modal Foundation Models 

**Title (ZH)**: 多模态基础模型的机理可解释性综述 

**Authors**: Zihao Lin, Samyadeep Basu, Mohammad Beigi, Varun Manjunatha, Ryan A. Rossi, Zichao Wang, Yufan Zhou, Sriram Balasubramanian, Arman Zarei, Keivan Rezaei, Ying Shen, Barry Menglong Yao, Zhiyang Xu, Qin Liu, Yuxiang Zhang, Yan Sun, Shilong Liu, Li Shen, Hongxuan Li, Soheil Feizi, Lifu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17516)  

**Abstract**: The rise of foundation models has transformed machine learning research, prompting efforts to uncover their inner workings and develop more efficient and reliable applications for better control. While significant progress has been made in interpreting Large Language Models (LLMs), multimodal foundation models (MMFMs) - such as contrastive vision-language models, generative vision-language models, and text-to-image models - pose unique interpretability challenges beyond unimodal frameworks. Despite initial studies, a substantial gap remains between the interpretability of LLMs and MMFMs. This survey explores two key aspects: (1) the adaptation of LLM interpretability methods to multimodal models and (2) understanding the mechanistic differences between unimodal language models and crossmodal systems. By systematically reviewing current MMFM analysis techniques, we propose a structured taxonomy of interpretability methods, compare insights across unimodal and multimodal architectures, and highlight critical research gaps. 

**Abstract (ZH)**: 基础模型的兴起已改变了机器学习研究，促使人们努力揭示其内在工作机制，并开发更高效和可靠的多模态应用以实现更好控制。尽管在解释大规模语言模型方面取得了显著进展，但对比视觉语言模型、生成式视觉语言模型和文本到图像模型等多模态基础模型（MMFMs）提出了超越单模态框架的独特可解释性挑战。尽管初步研究已经开展，大规模语言模型和多模态基础模型之间的可解释性差距仍然较大。本综述探讨了两个关键方面：（1）将大规模语言模型的解释方法适应多模态模型；（2）理解单模态语言模型与跨模态系统之间的机制差异。通过系统性地回顾当前的多模态基础模型分析技术，我们提出了一种结构化的解释方法分类体系，比较了单模态和多模态架构之间的洞见，并突出了关键的研究空白。 

---
# SpikeRL: A Scalable and Energy-efficient Framework for Deep Spiking Reinforcement Learning 

**Title (ZH)**: SpikeRL：一种可扩展和能效高的深度尖峰强化学习框架 

**Authors**: Tokey Tahmid, Mark Gates, Piotr Luszczek, Catherine D. Schuman  

**Link**: [PDF](https://arxiv.org/pdf/2502.17496)  

**Abstract**: In this era of AI revolution, massive investments in large-scale data-driven AI systems demand high-performance computing, consuming tremendous energy and resources. This trend raises new challenges in optimizing sustainability without sacrificing scalability or performance. Among the energy-efficient alternatives of the traditional Von Neumann architecture, neuromorphic computing and its Spiking Neural Networks (SNNs) are a promising choice due to their inherent energy efficiency. However, in some real-world application scenarios such as complex continuous control tasks, SNNs often lack the performance optimizations that traditional artificial neural networks have. Researchers have addressed this by combining SNNs with Deep Reinforcement Learning (DeepRL), yet scalability remains unexplored. In this paper, we extend our previous work on SpikeRL, which is a scalable and energy efficient framework for DeepRL-based SNNs for continuous control. In our initial implementation of SpikeRL framework, we depended on the population encoding from the Population-coded Spiking Actor Network (PopSAN) method for our SNN model and implemented distributed training with Message Passing Interface (MPI) through mpi4py. Also, further optimizing our model training by using mixed-precision for parameter updates. In our new SpikeRL framework, we have implemented our own DeepRL-SNN component with population encoding, and distributed training with PyTorch Distributed package with NCCL backend while still optimizing with mixed precision training. Our new SpikeRL implementation is 4.26X faster and 2.25X more energy efficient than state-of-the-art DeepRL-SNN methods. Our proposed SpikeRL framework demonstrates a truly scalable and sustainable solution for complex continuous control tasks in real-world applications. 

**Abstract (ZH)**: 在人工智能革命时代，大规模数据驱动的人工智能系统投资需要高性能计算，消耗大量能源和资源。这一趋势提出了在不牺牲可扩展性或性能的情况下优化可持续性的新挑战。作为传统冯·诺依曼架构的节能替代方案，神经形态计算及其脉冲神经网络（SNNs）因其固有的能效而颇具前景。然而，在复杂的连续控制任务等实际应用场景中，SNNs往往缺乏传统人工神经网络的性能优化。研究人员通过将SNNs与深度强化学习（DeepRL）相结合来解决这一问题，但可扩展性尚未被探索。本文扩展了我们关于SpikeRL的先前工作，这是一个基于DeepRL的SNNs连续控制的可扩展和节能框架。在我们最初实现的SpikeRL框架中，我们基于Population-coded Spiking Actor Network（PopSAN）方法的群体编码构建了SNN模型，并使用mpi4py通过消息传递接口（MPI）实现了分布式训练，同时通过混合精度训练优化了模型训练。在我们最新的SpikeRL框架中，我们实现了自己的具有群体编码的DeepRL-SNN组件，并使用NCCL后端的PyTorch Distribute包实现了分布式训练，同时仍然通过混合精度训练优化了模型训练。我们新的SpikeRL实现比最先进的DeepRL-SNN方法快4.26倍，节能2.25倍。我们提出的SpikeRL框架展示了在实际应用中为复杂连续控制任务提供真正可扩展和可持续的解决方案。 

---
