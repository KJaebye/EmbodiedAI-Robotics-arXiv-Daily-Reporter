# Robot Learning from a Physical World Model 

**Title (ZH)**: 机器人学习从物理世界模型角度探讨 

**Authors**: Jiageng Mao, Sicheng He, Hao-Ning Wu, Yang You, Shuyang Sun, Zhicheng Wang, Yanan Bao, Huizhong Chen, Leonidas Guibas, Vitor Guizilini, Howard Zhou, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07416)  

**Abstract**: We introduce PhysWorld, a framework that enables robot learning from video generation through physical world modeling. Recent video generation models can synthesize photorealistic visual demonstrations from language commands and images, offering a powerful yet underexplored source of training signals for robotics. However, directly retargeting pixel motions from generated videos to robots neglects physics, often resulting in inaccurate manipulations. PhysWorld addresses this limitation by coupling video generation with physical world reconstruction. Given a single image and a task command, our method generates task-conditioned videos and reconstructs the underlying physical world from the videos, and the generated video motions are grounded into physically accurate actions through object-centric residual reinforcement learning with the physical world model. This synergy transforms implicit visual guidance into physically executable robotic trajectories, eliminating the need for real robot data collection and enabling zero-shot generalizable robotic manipulation. Experiments on diverse real-world tasks demonstrate that PhysWorld substantially improves manipulation accuracy compared to previous approaches. Visit \href{this https URL}{the project webpage} for details. 

**Abstract (ZH)**: PhysWorld：一种通过物理世界建模实现基于视频生成的机器人学习框架 

---
# Using Vision Language Models as Closed-Loop Symbolic Planners for Robotic Applications: A Control-Theoretic Perspective 

**Title (ZH)**: 将视觉语言模型用于闭环符号规划的机器人应用：一种控制理论视角 

**Authors**: Hao Wang, Sathwik Karnik, Bea Lim, Somil Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2511.07410)  

**Abstract**: Large Language Models (LLMs) and Vision Language Models (VLMs) have been widely used for embodied symbolic planning. Yet, how to effectively use these models for closed-loop symbolic planning remains largely unexplored. Because they operate as black boxes, LLMs and VLMs can produce unpredictable or costly errors, making their use in high-level robotic planning especially challenging. In this work, we investigate how to use VLMs as closed-loop symbolic planners for robotic applications from a control-theoretic perspective. Concretely, we study how the control horizon and warm-starting impact the performance of VLM symbolic planners. We design and conduct controlled experiments to gain insights that are broadly applicable to utilizing VLMs as closed-loop symbolic planners, and we discuss recommendations that can help improve the performance of VLM symbolic planners. 

**Abstract (ZH)**: 大型语言模型（LLMs）和视觉语言模型（VLMs）已在体现符号规划中得到了广泛应用，但如何有效利用这些模型进行闭环符号规划仍 largely unexplored。由于它们作为黑盒运作，LLMs 和 VLMs 可能会产生不可预测或代价高昂的错误，使其在高级机器人规划中的应用尤其具有挑战性。在本项工作中，我们从控制理论的角度研究如何使用 VLMs 作为闭环符号规划器来应用于机器人应用。具体而言，我们探讨了控制窗口和预热启动如何影响 VLM 符号规划器的性能。我们设计并进行了控制实验，以获得对使用 VLMs 作为闭环符号规划器具有广泛适用性的洞见，并讨论了有助于提高 VLM 符号规划器性能的建议。 

---
# Unified Humanoid Fall-Safety Policy from a Few Demonstrations 

**Title (ZH)**: 统一的人形机器人摔倒安全性政策从少量示范学习 

**Authors**: Zhengjie Xu, Ye Li, Kwan-yee Lin, Stella X. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07407)  

**Abstract**: Falling is an inherent risk of humanoid mobility. Maintaining stability is thus a primary safety focus in robot control and learning, yet no existing approach fully averts loss of balance. When instability does occur, prior work addresses only isolated aspects of falling: avoiding falls, choreographing a controlled descent, or standing up afterward. Consequently, humanoid robots lack integrated strategies for impact mitigation and prompt recovery when real falls defy these scripts. We aim to go beyond keeping balance to make the entire fall-and-recovery process safe and autonomous: prevent falls when possible, reduce impact when unavoidable, and stand up when fallen. By fusing sparse human demonstrations with reinforcement learning and an adaptive diffusion-based memory of safe reactions, we learn adaptive whole-body behaviors that unify fall prevention, impact mitigation, and rapid recovery in one policy. Experiments in simulation and on a Unitree G1 demonstrate robust sim-to-real transfer, lower impact forces, and consistently fast recovery across diverse disturbances, pointing towards safer, more resilient humanoids in real environments. Videos are available at this https URL. 

**Abstract (ZH)**: humanoid移动 inherent的风险是摔倒。因此，保持稳定性是机器人控制和学习中的首要安全重点，但目前没有任何方法完全防止平衡丧失。在失稳发生时，先前的工作仅针对摔倒的孤立方面：避免摔倒、 choreographing一个受控下降，或摔倒后站起来。因此，人形机器人缺乏在实际摔倒无法遵循这些剧本时进行整合冲击缓解和迅速恢复的策略。我们的目标是超越保持平衡，使整个摔倒和恢复过程安全且自主：尽可能避免摔倒、不可避免时减少冲击，摔倒后站起来。通过将稀疏的人类演示与强化学习结合，并使用适应性扩散为基础的安全反应记忆，我们学习了统一摔倒预防、冲击缓解和快速恢复的自适应全身行为。在模拟和Unitree G1上的实验展示了强大的模拟到现实的转移、较低的冲击力和在各种干扰下始终如一的快速恢复，这表明在未来在实际环境中更加安全和具有弹性的类人机器人。视频可在此处在线查看。 

---
# Residual Rotation Correction using Tactile Equivariance 

**Title (ZH)**: 基于触觉同变性的残差旋转校正 

**Authors**: Yizhe Zhu, Zhang Ye, Boce Hu, Haibo Zhao, Yu Qi, Dian Wang, Robert Platt  

**Link**: [PDF](https://arxiv.org/pdf/2511.07381)  

**Abstract**: Visuotactile policy learning augments vision-only policies with tactile input, facilitating contact-rich manipulation. However, the high cost of tactile data collection makes sample efficiency the key requirement for developing visuotactile policies. We present EquiTac, a framework that exploits the inherent SO(2) symmetry of in-hand object rotation to improve sample efficiency and generalization for visuotactile policy learning. EquiTac first reconstructs surface normals from raw RGB inputs of vision-based tactile sensors, so rotations of the normal vector field correspond to in-hand object rotations. An SO(2)-equivariant network then predicts a residual rotation action that augments a base visuomotor policy at test time, enabling real-time rotation correction without additional reorientation demonstrations. On a real robot, EquiTac accurately achieves robust zero-shot generalization to unseen in-hand orientations with very few training samples, where baselines fail even with more training data. To our knowledge, this is the first tactile learning method to explicitly encode tactile equivariance for policy learning, yielding a lightweight, symmetry-aware module that improves reliability in contact-rich tasks. 

**Abstract (ZH)**: 基于视觉-触觉策略学习的EquiTac框架：利用SO(2)对称性提升样本效率和泛化能力 

---
# Multi-Agent AI Framework for Road Situation Detection and C-ITS Message Generation 

**Title (ZH)**: 多Agent人工智能框架用于道路情况检测及C-ITS消息生成 

**Authors**: Kailin Tong, Selim Solmaz, Kenan Mujkic, Gottfried Allmer, Bo Leng  

**Link**: [PDF](https://arxiv.org/pdf/2511.06892)  

**Abstract**: Conventional road-situation detection methods achieve strong performance in predefined scenarios but fail in unseen cases and lack semantic interpretation, which is crucial for reliable traffic recommendations. This work introduces a multi-agent AI framework that combines multimodal large language models (MLLMs) with vision-based perception for road-situation monitoring. The framework processes camera feeds and coordinates dedicated agents for situation detection, distance estimation, decision-making, and Cooperative Intelligent Transport System (C-ITS) message generation. Evaluation is conducted on a custom dataset of 103 images extracted from 20 videos of the TAD dataset. Both Gemini-2.0-Flash and Gemini-2.5-Flash were evaluated. The results show 100\% recall in situation detection and perfect message schema correctness; however, both models suffer from false-positive detections and have reduced performance in terms of number of lanes, driving lane status and cause code. Surprisingly, Gemini-2.5-Flash, though more capable in general tasks, underperforms Gemini-2.0-Flash in detection accuracy and semantic understanding and incurs higher latency (Table II). These findings motivate further work on fine-tuning specialized LLMs or MLLMs tailored for intelligent transportation applications. 

**Abstract (ZH)**: 基于视觉感知的多Agent AI框架：结合多模态大语言模型的路面情况监测 

---
# Vision-Aided Online A* Path Planning for Efficient and Safe Navigation of Service Robots 

**Title (ZH)**: 基于视觉的在线A*路径规划服务机器人高效安全导航 

**Authors**: Praveen Kumar, Tushar Sandhan  

**Link**: [PDF](https://arxiv.org/pdf/2511.06801)  

**Abstract**: The deployment of autonomous service robots in human-centric environments is hindered by a critical gap in perception and planning. Traditional navigation systems rely on expensive LiDARs that, while geometrically precise, are seman- tically unaware, they cannot distinguish a important document on an office floor from a harmless piece of litter, treating both as physically traversable. While advanced semantic segmentation exists, no prior work has successfully integrated this visual intelligence into a real-time path planner that is efficient enough for low-cost, embedded hardware. This paper presents a frame- work to bridge this gap, delivering context-aware navigation on an affordable robotic platform. Our approach centers on a novel, tight integration of a lightweight perception module with an online A* planner. The perception system employs a semantic segmentation model to identify user-defined visual constraints, enabling the robot to navigate based on contextual importance rather than physical size alone. This adaptability allows an operator to define what is critical for a given task, be it sensitive papers in an office or safety lines in a factory, thus resolving the ambiguity of what to avoid. This semantic perception is seamlessly fused with geometric data. The identified visual constraints are projected as non-geometric obstacles onto a global map that is continuously updated from sensor data, enabling robust navigation through both partially known and unknown environments. We validate our framework through extensive experiments in high-fidelity simulations and on a real-world robotic platform. The results demonstrate robust, real-time performance, proving that a cost- effective robot can safely navigate complex environments while respecting critical visual cues invisible to traditional planners. 

**Abstract (ZH)**: 自主服务机器人在以人为中心环境中的部署受制于感知和规划的关键差距。 

---
# Human-Level Actuation for Humanoids 

**Title (ZH)**: 人类级别驱动的人形机器人 

**Authors**: MD-Nazmus Sunbeam  

**Link**: [PDF](https://arxiv.org/pdf/2511.06796)  

**Abstract**: Claims that humanoid robots achieve ``human-level'' actuation are common but rarely quantified. Peak torque or speed specifications tell us little about whether a joint can deliver the right combination of torque, power, and endurance at task-relevant postures and rates. We introduce a comprehensive framework that makes ``human-level'' measurable and comparable across systems. Our approach has three components. First, a kinematic \emph{DoF atlas} standardizes joint coordinate systems and ranges of motion using ISB-based conventions, ensuring that human and robot joints are compared in the same reference frames. Second, \emph{Human-Equivalence Envelopes (HEE)} define per-joint requirements by measuring whether a robot meets human torque \emph{and} power simultaneously at the same joint angle and rate $(q,\omega)$, weighted by positive mechanical work in task-specific bands (walking, stairs, lifting, reaching, and hand actions). Third, the \emph{Human-Level Actuation Score (HLAS)} aggregates six physically grounded factors: workspace coverage (ROM and DoF), HEE coverage, torque-mode bandwidth, efficiency, and thermal sustainability. We provide detailed measurement protocols using dynamometry, electrical power monitoring, and thermal testing that yield every HLAS input from reproducible experiments. A worked example demonstrates HLAS computation for a multi-joint humanoid, showing how the score exposes actuator trade-offs (gearing ratio versus bandwidth and efficiency) that peak-torque specifications obscure. The framework serves as both a design specification for humanoid development and a benchmarking standard for comparing actuation systems, with all components grounded in published human biomechanics data. 

**Abstract (ZH)**: 人形机器人实现“人类水平”驱动性能的声明常见但鲜有量化。我们提出了一种综合框架，使“人类水平”的性能变得可测量和可对比。该方法包含三个组成部分。首先，使用ISB为基础的规范制定一个机械自由度地图（Kinematic DoF Atlas），标准化关节坐标系统和运动范围，确保人类和机器人关节在相同的参考框架下进行比较。其次，通过测量机器人在特定任务动作（行走、楼梯、举重、伸手及手部动作）中的关节角度和速率$(q,\omega)$下同时满足人类扭矩和功率要求，定义了关节级别的等效人类性能包络（Human-Equivalence Envelopes, HEE）。第三，综合六种物理基础因素：工作空间覆盖（运动范围和自由度）、HEE覆盖、扭矩模式带宽、效率和热可持续性，得到人形机器人驱动性能评分（Human-Level Actuation Score, HLAS）。我们提供了详细的测量协议，包括动态测定、电气功率监测和热测试，从可重复实验中获得每个HLAS输入。一个详细示例展示了如何使用该评分计算多关节人形机器人的驱动性能，揭示了峰值扭矩规范所掩盖的驱动器折衷（传动比与带宽和效率之间的权衡）。该框架既作为人形机器人开发的设计规范，也作为评估驱动系统性能的标准，所有组成部分都基于已公布的人类生物力学数据。 

---
# SlotVLA: Towards Modeling of Object-Relation Representations in Robotic Manipulation 

**Title (ZH)**: SlotVLA: 向量对象-关系表示在机器人操控中的建模研究 

**Authors**: Taisei Hanyu, Nhat Chung, Huy Le, Toan Nguyen, Yuki Ikebe, Anthony Gunderman, Duy Nguyen Ho Minh, Khoa Vo, Tung Kieu, Kashu Yamazaki, Chase Rainwater, Anh Nguyen, Ngan Le  

**Link**: [PDF](https://arxiv.org/pdf/2511.06754)  

**Abstract**: Inspired by how humans reason over discrete objects and their relationships, we explore whether compact object-centric and object-relation representations can form a foundation for multitask robotic manipulation. Most existing robotic multitask models rely on dense embeddings that entangle both object and background cues, raising concerns about both efficiency and interpretability. In contrast, we study object-relation-centric representations as a pathway to more structured, efficient, and explainable visuomotor control. Our contributions are two-fold. First, we introduce LIBERO+, a fine-grained benchmark dataset designed to enable and evaluate object-relation reasoning in robotic manipulation. Unlike prior datasets, LIBERO+ provides object-centric annotations that enrich demonstrations with box- and mask-level labels as well as instance-level temporal tracking, supporting compact and interpretable visuomotor representations. Second, we propose SlotVLA, a slot-attention-based framework that captures both objects and their relations for action decoding. It uses a slot-based visual tokenizer to maintain consistent temporal object representations, a relation-centric decoder to produce task-relevant embeddings, and an LLM-driven module that translates these embeddings into executable actions. Experiments on LIBERO+ demonstrate that object-centric slot and object-relation slot representations drastically reduce the number of required visual tokens, while providing competitive generalization. Together, LIBERO+ and SlotVLA provide a compact, interpretable, and effective foundation for advancing object-relation-centric robotic manipulation. 

**Abstract (ZH)**: 借鉴人类如何处理离散对象及其关系的方式，我们探索是否可以构建紧凑的对象中心和对象关系表示，作为多任务机器人操作的基础。现有的大多数多任务机器人模型依赖于稠密嵌入，将对象和背景线索交织在一起，这引起了效率和可解释性方面的担忧。相比之下，我们研究对象关系为中心的表示，以实现更结构化、更高效和更可解释的视听运动控制。我们的贡献有两个方面。首先，我们介绍了LIBERO+，一个细粒度基准数据集，旨在使对象关系推理在机器人操作中的应用和评估成为可能。LIBERO+提供了对象中心的标注，增强了演示中的框级和掩码级标签以及实例级时间跟踪，支持紧凑且可解释的视听运动表示。其次，我们提出了SlotVLA，一种基于槽注意力的框架，用于抓取对象及其关系以进行动作解码。该框架使用基于槽的视觉分词器来保持一致的时间对象表示，使用关系为中心的解码器生成与任务相关的嵌入，并使用由大型语言模型驱动的模块将这些嵌入转换为可执行的动作。在LIBERO+上的实验表明，对象中心的槽和对象关系槽表示大幅减少了所需视觉令牌的数量，同时提供了竞争性的泛化能力。总的来说，LIBERO+和SlotVLA为推进对象关系为中心的机器人操作提供了一个紧凑、可解释且有效的基础。 

---
# Semi-distributed Cross-modal Air-Ground Relative Localization 

**Title (ZH)**: 半分布跨模态空地相对定位 

**Authors**: Weining Lu, Deer Bin, Lian Ma, Ming Ma, Zhihao Ma, Xiangyang Chen, Longfei Wang, Yixiao Feng, Zhouxian Jiang, Yongliang Shi, Bin Liang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06749)  

**Abstract**: Efficient, accurate, and flexible relative localization is crucial in air-ground collaborative tasks. However, current approaches for robot relative localization are primarily realized in the form of distributed multi-robot SLAM systems with the same sensor configuration, which are tightly coupled with the state estimation of all robots, limiting both flexibility and accuracy. To this end, we fully leverage the high capacity of Unmanned Ground Vehicle (UGV) to integrate multiple sensors, enabling a semi-distributed cross-modal air-ground relative localization framework. In this work, both the UGV and the Unmanned Aerial Vehicle (UAV) independently perform SLAM while extracting deep learning-based keypoints and global descriptors, which decouples the relative localization from the state estimation of all agents. The UGV employs a local Bundle Adjustment (BA) with LiDAR, camera, and an IMU to rapidly obtain accurate relative pose estimates. The BA process adopts sparse keypoint optimization and is divided into two stages: First, optimizing camera poses interpolated from LiDAR-Inertial Odometry (LIO), followed by estimating the relative camera poses between the UGV and UAV. Additionally, we implement an incremental loop closure detection algorithm using deep learning-based descriptors to maintain and retrieve keyframes efficiently. Experimental results demonstrate that our method achieves outstanding performance in both accuracy and efficiency. Unlike traditional multi-robot SLAM approaches that transmit images or point clouds, our method only transmits keypoint pixels and their descriptors, effectively constraining the communication bandwidth under 0.3 Mbps. Codes and data will be publicly available on this https URL. 

**Abstract (ZH)**: 高效的、准确的和灵活的相对定位对于空地协作任务至关重要。然而，当前的机器人相对定位方法主要以具有相同传感器配置的分布式多机器人SLAM系统的形式实现，这紧密耦合了所有机器人的状态估计，限制了其灵活性和准确性。为此，我们充分利用无人地面车辆（UGV）的高容量来集成多种传感器，从而构建了一个半分布式跨模态空地相对定位框架。在此工作中，UGV和无人航空车辆（UAV）独立执行SLAM，提取基于深度学习的关键点和全局描述符，从而将相对定位与所有代理的状态估计解耦。UGV通过结合LiDAR、相机和IMU的局部调整测量（Bundle Adjustment, BA）快速获得准确的相对姿态估计。BA过程采用稀疏关键点优化，并分为两个阶段：首先，优化从激光雷达惯性里程计（LIO）插值得到的相机姿态，接着估算UGV和UAV之间的相对相机姿态。此外，我们使用基于深度学习的描述符实现了增量环视闭合并发检测算法，以高效地维护和检索关键帧。实验结果表明，我们的方法在准确性和效率方面表现出色。与传统的多机器人SLAM方法传输图像或点云不同，我们的方法仅传输关键点像素及其描述符，有效限制了通信带宽小于0.3 Mbps。代码和数据将在以下链接公开：这 https URL。 

---
# Physically-Grounded Goal Imagination: Physics-Informed Variational Autoencoder for Self-Supervised Reinforcement Learning 

**Title (ZH)**: 基于物理的目標想象：物理指导的变分自编码器在无监督强化学习中的应用 

**Authors**: Lan Thi Ha Nguyen, Kien Ton Manh, Anh Do Duc, Nam Pham Hai  

**Link**: [PDF](https://arxiv.org/pdf/2511.06745)  

**Abstract**: Self-supervised goal-conditioned reinforcement learning enables robots to autonomously acquire diverse skills without human supervision. However, a central challenge is the goal setting problem: robots must propose feasible and diverse goals that are achievable in their current environment. Existing methods like RIG (Visual Reinforcement Learning with Imagined Goals) use variational autoencoder (VAE) to generate goals in a learned latent space but have the limitation of producing physically implausible goals that hinder learning efficiency. We propose Physics-Informed RIG (PI-RIG), which integrates physical constraints directly into the VAE training process through a novel Enhanced Physics-Informed Variational Autoencoder (Enhanced p3-VAE), enabling the generation of physically consistent and achievable goals. Our key innovation is the explicit separation of the latent space into physics variables governing object dynamics and environmental factors capturing visual appearance, while enforcing physical consistency through differential equation constraints and conservation laws. This enables the generation of physically consistent and achievable goals that respect fundamental physical principles such as object permanence, collision constraints, and dynamic feasibility. Through extensive experiments, we demonstrate that this physics-informed goal generation significantly improves the quality of proposed goals, leading to more effective exploration and better skill acquisition in visual robotic manipulation tasks including reaching, pushing, and pick-and-place scenarios. 

**Abstract (ZH)**: 物理约束导向的自我监督条件化强化学习使机器人能够在无需人类监督的情况下自主获取多样技能。然而，核心挑战是目标设定问题：机器人必须提出在当前环境中的可行且多样的目标。现有方法如RIG（基于想象目标的视觉强化学习）使用变分自编码器（VAE）生成在学习潜在空间中的目标，但会产生物理上不可行的目标从而阻碍学习效率。我们提出了物理信息导向的RIG（PI-RIG），通过一种新颖的增强物理信息变分自编码器（Enhanced p3-VAE）直接将物理约束整合到VAE训练过程中，使生成物理一致且可实现的目标成为可能。我们的关键创新在于明确分离潜在空间为控制物体动力学的物理变量和捕捉视觉外观的环境因素，并通过微分方程约束和守恒定律确保物理一致性。这使得生成的物理一致且可实现的目标能够遵守诸如物体持续性、碰撞约束和动态可行性等基本物理原则。通过广泛的实验，我们证明这种物理信息目标生成显著提高了提出目标的质量，从而在包括抓取、推动和拿起放置等视觉机器人操作任务中提高了探索的有效性和技能的学习。 

---
# How Do VLAs Effectively Inherit from VLMs? 

**Title (ZH)**: VLAs如何有效地继承自VLMs？ 

**Authors**: Chuheng Zhang, Rushuai Yang, Xiaoyu Chen, Kaixin Wang, Li Zhao, Yi Chen, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2511.06619)  

**Abstract**: Vision-language-action (VLA) models hold the promise to attain generalizable embodied control. To achieve this, a pervasive paradigm is to leverage the rich vision-semantic priors of large vision-language models (VLMs). However, the fundamental question persists: How do VLAs effectively inherit the prior knowledge from VLMs? To address this critical question, we introduce a diagnostic benchmark, GrinningFace, an emoji tabletop manipulation task where the robot arm is asked to place objects onto printed emojis corresponding to language instructions. This task design is particularly revealing -- knowledge associated with emojis is ubiquitous in Internet-scale datasets used for VLM pre-training, yet emojis themselves are largely absent from standard robotics datasets. Consequently, they provide a clean proxy: successful task completion indicates effective transfer of VLM priors to embodied control. We implement this diagnostic task in both simulated environment and a real robot, and compare various promising techniques for knowledge transfer. Specifically, we investigate the effects of parameter-efficient fine-tuning, VLM freezing, co-training, predicting discretized actions, and predicting latent actions. Through systematic evaluation, our work not only demonstrates the critical importance of preserving VLM priors for the generalization of VLA but also establishes guidelines for future research in developing truly generalizable embodied AI systems. 

**Abstract (ZH)**: Vision-语言-动作（VLA）模型有实现泛化 embodied控制的潜力。为了实现这一目标，一种普遍的做法是利用大型视觉-语言模型（VLMs）的丰富视觉语义先验知识。然而，一个基本的问题仍然存在：VLA是如何有效地继承VLMs的先验知识的？为了解决这个问题，我们引入了一个诊断基准GrinningFace，这是一个表情符号桌面上的操纵任务，机器人臂被要求根据语言指令将物体放置在对应的打印表情符号上。这一任务设计尤为揭示性——与表情符号相关的知识在用于VLM预训练的互联网规模的数据集中普遍存在，但表情符号本身在标准的机器人数据集中却几乎不存在。因此，它们提供了一个干净的代理：成功完成任务表明VLM先验知识已成功转移到embodied控制。我们在模拟环境和真实机器人中实施了此诊断任务，并比较了几种有前景的知识转移技术，包括参数高效微调、VLM冻结、协同训练、预测离散动作和预测潜在动作。通过系统的评估，我们的工作不仅突显了保留VLM先验知识对于VLA泛化的关键重要性，还为开发真正泛化的embodied AI系统未来研究提供了指导。 

---
# CoFineLLM: Conformal Finetuning of LLMs for Language-Instructed Robot Planning 

**Title (ZH)**: CoFineLLM：基于语言指令的机器人规划中LLM的校准微调 

**Authors**: Jun Wang, Yevgeniy Vorobeychik, Yiannis Kantaros  

**Link**: [PDF](https://arxiv.org/pdf/2511.06575)  

**Abstract**: Large Language Models (LLMs) have recently emerged as planners for language-instructed agents, generating sequences of actions to accomplish natural language tasks. However, their reliability remains a challenge, especially in long-horizon tasks, since they often produce overconfident yet wrong outputs. Conformal Prediction (CP) has been leveraged to address this issue by wrapping LLM outputs into prediction sets that contain the correct action with a user-defined confidence. When the prediction set is a singleton, the planner executes that action; otherwise, it requests help from a user. This has led to LLM-based planners that can ensure plan correctness with a user-defined probability. However, as LLMs are trained in an uncertainty-agnostic manner, without awareness of prediction sets, they tend to produce unnecessarily large sets, particularly at higher confidence levels, resulting in frequent human interventions limiting autonomous deployment. To address this, we introduce CoFineLLM (Conformal Finetuning for LLMs), the first CP-aware finetuning framework for LLM-based planners that explicitly reduces prediction-set size and, in turn, the need for user interventions. We evaluate our approach on multiple language-instructed robot planning problems and show consistent improvements over uncertainty-aware and uncertainty-agnostic finetuning baselines in terms of prediction-set size, and help rates. Finally, we demonstrate robustness of our method to out-of-distribution scenarios in hardware experiments. 

**Abstract (ZH)**: 基于校准预测的LLM微调框架：CoFineLLM 

---
# Adaptive PID Control for Robotic Systems via Hierarchical Meta-Learning and Reinforcement Learning with Physics-Based Data Augmentation 

**Title (ZH)**: 基于层次元学习和基于物理的数据增强的强化学习的自适应PID控制方法在机器人系统中的应用 

**Authors**: JiaHao Wu, ShengWen Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06500)  

**Abstract**: Proportional-Integral-Derivative (PID) controllers remain the predominant choice in industrial robotics due to their simplicity and reliability. However, manual tuning of PID parameters for diverse robotic platforms is time-consuming and requires extensive domain expertise. This paper presents a novel hierarchical control framework that combines meta-learning for PID initialization and reinforcement learning (RL) for online adaptation. To address the sample efficiency challenge, a \textit{physics-based data augmentation} strategy is introduced that generates virtual robot configurations by systematically perturbing physical parameters, enabling effective meta-learning with limited real robot data. The proposed approach is evaluated on two heterogeneous platforms: a 9-DOF Franka Panda manipulator and a 12-DOF Laikago quadruped robot. Experimental results demonstrate that the proposed method achieves 16.6\% average improvement on Franka Panda (6.26° MAE), with exceptional gains in high-load joints (J2: 80.4\% improvement from 12.36° to 2.42°). Critically, this work discovers the \textit{optimization ceiling effect}: RL achieves dramatic improvements when meta-learning exhibits localized high-error joints, but provides no benefit (0.0\%) when baseline performance is uniformly strong, as observed in Laikago. The method demonstrates robust performance under disturbances (parameter uncertainty: +19.2\%, no disturbance: +16.6\%, average: +10.0\%) with only 10 minutes of training time. Multi-seed analysis across 100 random initializations confirms stable performance (4.81+/-1.64\% average). These results establish that RL effectiveness is highly dependent on meta-learning baseline quality and error distribution, providing important design guidance for hierarchical control systems. 

**Abstract (ZH)**: 基于元学习的PID初始化结合强化学习的在线自适应分级控制框架 

---
# A Low-Rank Method for Vision Language Model Hallucination Mitigation in Autonomous Driving 

**Title (ZH)**: 一种低秩方法用于减轻自主驾驶中视觉语言模型幻觉 

**Authors**: Keke Long, Jiacheng Guo, Tianyun Zhang, Hongkai Yu, Xiaopeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06496)  

**Abstract**: Vision Language Models (VLMs) are increasingly used in autonomous driving to help understand traffic scenes, but they sometimes produce hallucinations, which are false details not grounded in the visual input. Detecting and mitigating hallucinations is challenging when ground-truth references are unavailable and model internals are inaccessible. This paper proposes a novel self-contained low-rank approach to automatically rank multiple candidate captions generated by multiple VLMs based on their hallucination levels, using only the captions themselves without requiring external references or model access. By constructing a sentence-embedding matrix and decomposing it into a low-rank consensus component and a sparse residual, we use the residual magnitude to rank captions: selecting the one with the smallest residual as the most hallucination-free. Experiments on the NuScenes dataset demonstrate that our approach achieves 87% selection accuracy in identifying hallucination-free captions, representing a 19% improvement over the unfiltered baseline and a 6-10% improvement over multi-agent debate method. The sorting produced by sparse error magnitudes shows strong correlation with human judgments of hallucinations, validating our scoring mechanism. Additionally, our method, which can be easily parallelized, reduces inference time by 51-67% compared to debate approaches, making it practical for real-time autonomous driving applications. 

**Abstract (ZH)**: 一种基于低秩的自包含方法用于多VLM候选caption的自动去幻觉排序 

---
# Sim-to-Real Transfer in Deep Reinforcement Learning for Bipedal Locomotion 

**Title (ZH)**: 从模拟到现实的深层强化学习在双足行走中的迁移学习 

**Authors**: Lingfan Bao, Tianhu Peng, Chengxu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.06465)  

**Abstract**: This chapter addresses the critical challenge of simulation-to-reality (sim-to-real) transfer for deep reinforcement learning (DRL) in bipedal locomotion. After contextualizing the problem within various control architectures, we dissect the ``curse of simulation'' by analyzing the primary sources of sim-to-real gap: robot dynamics, contact modeling, state estimation, and numerical solvers. Building on this diagnosis, we structure the solutions around two complementary philosophies. The first is to shrink the gap through model-centric strategies that systematically improve the simulator's physical fidelity. The second is to harden the policy, a complementary approach that uses in-simulation robustness training and post-deployment adaptation to make the policy inherently resilient to model inaccuracies. The chapter concludes by synthesizing these philosophies into a strategic framework, providing a clear roadmap for developing and evaluating robust sim-to-real solutions. 

**Abstract (ZH)**: 本章探讨了双足运动深度强化学习（DRL）中从仿真到现实（sim-to-real）迁移的关键挑战。在对各种控制架构进行背景分析后，我们通过分析主要的仿真到现实差距来源——机器人动力学、接触建模、状态估计和数值求解器，拆解了“仿真之困”。基于此诊断，我们围绕两种互补的哲学理念构建解决方案。第一个理念是通过模型为中心的方法系统性地提高仿真器的物理保真度来缩小差距；第二个理念是强化策略，这是一种补充方法，通过仿真中的鲁棒性训练和部署后适应，使策略本身能够抵御模型不准确性的内在韧性。本章最后将这些哲学理念综合成一个战略框架，提供了一条清晰的道路，用于开发和评估稳健的仿真到现实解决方案。 

---
# Towards Adaptive Humanoid Control via Multi-Behavior Distillation and Reinforced Fine-Tuning 

**Title (ZH)**: 基于多行为蒸馏和强化精细调整的自适应 humanoid 控制研究 

**Authors**: Yingnan Zhao, Xinmiao Wang, Dewei Wang, Xinzhe Liu, Dan Lu, Qilong Han, Peng Liu, Chenjia Bai  

**Link**: [PDF](https://arxiv.org/pdf/2511.06371)  

**Abstract**: Humanoid robots are promising to learn a diverse set of human-like locomotion behaviors, including standing up, walking, running, and jumping. However, existing methods predominantly require training independent policies for each skill, yielding behavior-specific controllers that exhibit limited generalization and brittle performance when deployed on irregular terrains and in diverse situations. To address this challenge, we propose Adaptive Humanoid Control (AHC) that adopts a two-stage framework to learn an adaptive humanoid locomotion controller across different skills and terrains. Specifically, we first train several primary locomotion policies and perform a multi-behavior distillation process to obtain a basic multi-behavior controller, facilitating adaptive behavior switching based on the environment. Then, we perform reinforced fine-tuning by collecting online feedback in performing adaptive behaviors on more diverse terrains, enhancing terrain adaptability for the controller. We conduct experiments in both simulation and real-world experiments in Unitree G1 robots. The results show that our method exhibits strong adaptability across various situations and terrains. Project website: this https URL. 

**Abstract (ZH)**: 仿人体机器人有望学习一系列类人的运动行为，包括站立、行走、跑步和跳跃。然而，现有方法主要需要为每种技能训练独立的策略，导致行为特定的控制程序，在不规则地形和多样场景中表现出有限的泛化能力和脆弱的性能。为应对这一挑战，我们提出了一种适应性仿人体控制（Adaptive Humanoid Control, AHC）方法，采用两阶段框架在不同技能和地形上学习适应性仿人体运动控制器。具体而言，我们首先训练多个基本运动策略，并执行多行为精简过程，以获得一个基础的多行为控制程序，便于依据环境进行适应性行为切换。接着，我们通过在线采集在更多样地形上执行适应性行为的反馈进行强化微调，增强控制器的地形适应性。我们在Unitree G1机器人上进行了仿真和真实世界实验。结果表明，我们的方法在各种情境和地形上表现出强大的适应性。项目网站：this https URL。 

---
# Affordance-Guided Coarse-to-Fine Exploration for Base Placement in Open-Vocabulary Mobile Manipulation 

**Title (ZH)**: 基于功能引导的粗细探索方法用于开放词汇-Mobile manipulation场景下的基座放置 

**Authors**: Tzu-Jung Lin, Jia-Fong Yeh, Hung-Ting Su, Chung-Yi Lin, Yi-Ting Chen, Winston H. Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06240)  

**Abstract**: In open-vocabulary mobile manipulation (OVMM), task success often hinges on the selection of an appropriate base placement for the robot. Existing approaches typically navigate to proximity-based regions without considering affordances, resulting in frequent manipulation failures. We propose Affordance-Guided Coarse-to-Fine Exploration, a zero-shot framework for base placement that integrates semantic understanding from vision-language models (VLMs) with geometric feasibility through an iterative optimization process. Our method constructs cross-modal representations, namely Affordance RGB and Obstacle Map+, to align semantics with spatial context. This enables reasoning that extends beyond the egocentric limitations of RGB perception. To ensure interaction is guided by task-relevant affordances, we leverage coarse semantic priors from VLMs to guide the search toward task-relevant regions and refine placements with geometric constraints, thereby reducing the risk of convergence to local optima. Evaluated on five diverse open-vocabulary mobile manipulation tasks, our system achieves an 85% success rate, significantly outperforming classical geometric planners and VLM-based methods. This demonstrates the promise of affordance-aware and multimodal reasoning for generalizable, instruction-conditioned planning in OVMM. 

**Abstract (ZH)**: 面向开放词汇Mobile manipulation的 affordance引导粗细粒度探索方法 

---
# ExpReS-VLA: Specializing Vision-Language-Action Models Through Experience Replay and Retrieval 

**Title (ZH)**: ExpReS-VLA: 通过经验重播与检索专门化视觉-语言-行动模型 

**Authors**: Shahram Najam Syed, Yatharth Ahuja, Arthur Jakobsson, Jeff Ichnowski  

**Link**: [PDF](https://arxiv.org/pdf/2511.06202)  

**Abstract**: Vision-Language-Action models such as OpenVLA show impressive zero-shot generalization across robotic manipulation tasks but often fail to adapt efficiently to new deployment environments. In many real-world applications, consistent high performance on a limited set of tasks is more important than broad generalization. We propose ExpReS-VLA, a method for specializing pre-trained VLA models through experience replay and retrieval while preventing catastrophic forgetting. ExpReS-VLA stores compact feature representations from the frozen vision backbone instead of raw image-action pairs, reducing memory usage by approximately 97 percent. During deployment, relevant past experiences are retrieved using cosine similarity and used to guide adaptation, while prioritized experience replay emphasizes successful trajectories. We also introduce Thresholded Hybrid Contrastive Loss, which enables learning from both successful and failed attempts. On the LIBERO simulation benchmark, ExpReS-VLA improves success rates from 82.6 to 93.1 percent on spatial reasoning tasks and from 61 to 72.3 percent on long-horizon tasks. On physical robot experiments with five manipulation tasks, it reaches 98 percent success on both seen and unseen settings, compared to 84.7 and 32 percent for naive fine-tuning. Adaptation takes 31 seconds using 12 demonstrations on a single RTX 5090 GPU, making the approach practical for real robot deployment. 

**Abstract (ZH)**: 通过经验回放和检索专门化的Vision-Language-Action模型：防止灾难性遗忘的ExpReS-VLA方法 

---
# OpenVLN: Open-world aerial Vision-Language Navigation 

**Title (ZH)**: 开放式世界航拍视觉语言导航 

**Authors**: Peican Lin, Gan Sun, Chenxi Liu, Fazeng Li, Weihong Ren, Yang Cong  

**Link**: [PDF](https://arxiv.org/pdf/2511.06182)  

**Abstract**: Vision-language models (VLMs) have been widely-applied in ground-based vision-language navigation (VLN). However, the vast complexity of outdoor aerial environments compounds data acquisition challenges and imposes long-horizon trajectory planning requirements on Unmanned Aerial Vehicles (UAVs), introducing novel complexities for aerial VLN. To address these challenges, we propose a data-efficient Open-world aerial Vision-Language Navigation (i.e., OpenVLN) framework, which could execute language-guided flight with limited data constraints and enhance long-horizon trajectory planning capabilities in complex aerial environments. Specifically, we reconfigure a reinforcement learning framework to optimize the VLM for UAV navigation tasks, which can efficiently fine-tune VLM by using rule-based policies under limited training data. Concurrently, we introduce a long-horizon planner for trajectory synthesis that dynamically generates precise UAV actions via value-based rewards. To the end, we conduct sufficient navigation experiments on the TravelUAV benchmark with dataset scaling across diverse reward settings. Our method demonstrates consistent performance gains of up to 4.34% in Success Rate, 6.19% in Oracle Success Rate, and 4.07% in Success weighted by Path Length over baseline methods, validating its deployment efficacy for long-horizon UAV navigation in complex aerial environments. 

**Abstract (ZH)**: 开放世界的空中视觉语言导航（即OpenVLN）框架 

---
# 10 Open Challenges Steering the Future of Vision-Language-Action Models 

**Title (ZH)**: 10 开放挑战引领视觉-语言-行动模型的未来 

**Authors**: Soujanya Poria, Navonil Majumder, Chia-Yu Hung, Amir Ali Bagherzadeh, Chuan Li, Kenneth Kwok, Ziwei Wang, Cheston Tan, Jiajun Wu, David Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05936)  

**Abstract**: Due to their ability of follow natural language instructions, vision-language-action (VLA) models are increasingly prevalent in the embodied AI arena, following the widespread success of their precursors -- LLMs and VLMs. In this paper, we discuss 10 principal milestones in the ongoing development of VLA models -- multimodality, reasoning, data, evaluation, cross-robot action generalization, efficiency, whole-body coordination, safety, agents, and coordination with humans. Furthermore, we discuss the emerging trends of using spatial understanding, modeling world dynamics, post training, and data synthesis -- all aiming to reach these milestones. Through these discussions, we hope to bring attention to the research avenues that may accelerate the development of VLA models into wider acceptability. 

**Abstract (ZH)**: 由于具有遵循自然语言指令的能力，多模态语言动作（VLA）模型在具身AI领域越来越普遍，这得益于其前身——大规模语言模型（LLMs）和大规模视觉模型（VLMs）的广泛成功。在本文中，我们讨论了VLA模型在不断发展过程中的10个主要里程碑——多模态性、推理、数据、评估、跨机器人动作通用化、效率、全身协调、安全性、智能体以及与人类的协调。此外，我们还探讨了使用空间理解、建模世界动力学、训练后处理和数据合成等新兴趋势，旨在实现这些里程碑。通过这些讨论，我们希望引起对能够加速VLA模型广泛应用的研究方向的关注。 

---
# From Words to Safety: Language-Conditioned Safety Filtering for Robot Navigation 

**Title (ZH)**: 从语言到安全：基于语言条件的安全过滤在机器人导航中的应用 

**Authors**: Zeyuan Feng, Haimingyue Zhang, Somil Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2511.05889)  

**Abstract**: As robots become increasingly integrated into open-world, human-centered environments, their ability to interpret natural language instructions and adhere to safety constraints is critical for effective and trustworthy interaction. Existing approaches often focus on mapping language to reward functions instead of safety specifications or address only narrow constraint classes (e.g., obstacle avoidance), limiting their robustness and applicability. We propose a modular framework for language-conditioned safety in robot navigation. Our framework is composed of three core components: (1) a large language model (LLM)-based module that translates free-form instructions into structured safety specifications, (2) a perception module that grounds these specifications by maintaining object-level 3D representations of the environment, and (3) a model predictive control (MPC)-based safety filter that enforces both semantic and geometric constraints in real time. We evaluate the effectiveness of the proposed framework through both simulation studies and hardware experiments, demonstrating that it robustly interprets and enforces diverse language-specified constraints across a wide range of environments and scenarios. 

**Abstract (ZH)**: 随着机器人越来越多地集成到开放的世界、以人为中心的环境中，它们解释自然语言指令和遵守安全约束的能力对于有效的和可信赖的交互至关重要。现有的方法往往侧重于将语言映射到奖励函数，而不是安全规范，或者仅处理狭窄的约束类别（如障碍物避让），这限制了它们的鲁棒性和适用性。我们提出了一种模块化框架，用于机器人导航中的语言条件安全。该框架由三个核心组件组成：（1）基于大规模语言模型（LLM）的模块，将自由形式的指令转换为结构化安全规范；（2）感知模块，通过保持环境的对象级3D表示来实现这些规范；（3）基于模型预测控制（MPC）的安全过滤器，实时强制执行语义和几何约束。我们通过仿真研究和硬件实验评估了所提出框架的有效性，证明了它能够在多种环境和场景中稳健地解释和执行各种语言指定的约束。 

---
# Gentle Manipulation Policy Learning via Demonstrations from VLM Planned Atomic Skills 

**Title (ZH)**: 通过VLM计划的原子技能演示学习温和操控策略 

**Authors**: Jiayu Zhou, Qiwei Wu, Jian Li, Zhe Chen, Xiaogang Xiong, Renjing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05855)  

**Abstract**: Autonomous execution of long-horizon, contact-rich manipulation tasks traditionally requires extensive real-world data and expert engineering, posing significant cost and scalability challenges. This paper proposes a novel framework integrating hierarchical semantic decomposition, reinforcement learning (RL), visual language models (VLMs), and knowledge distillation to overcome these limitations. Complex tasks are decomposed into atomic skills, with RL-trained policies for each primitive exclusively in simulation. Crucially, our RL formulation incorporates explicit force constraints to prevent object damage during delicate interactions. VLMs perform high-level task decomposition and skill planning, generating diverse expert demonstrations. These are distilled into a unified policy via Visual-Tactile Diffusion Policy for end-to-end execution. We conduct comprehensive ablation studies exploring different VLM-based task planners to identify optimal demonstration generation pipelines, and systematically compare imitation learning algorithms for skill distillation. Extensive simulation experiments and physical deployment validate that our approach achieves policy learning for long-horizon manipulation without costly human demonstrations, while the VLM-guided atomic skill framework enables scalable generalization to diverse tasks. 

**Abstract (ZH)**: 自主执行长周期、接触丰富的操作任务通常需要大量的现实世界数据和专家工程设计，面临显著的成本和扩展性挑战。本文提出了一个新颖框架，该框架结合了分层语义分解、强化学习（RL）、视觉语言模型（VLMs）和知识蒸馏，以克服这些限制。复杂的任务被分解为原子技能，每个原始技能的RL训练策略仅在仿真环境中进行。关键的是，我们的RL形式化方法融入了明确的力约束，以防止在精细交互过程中损坏物体。VLMs执行高层任务分解和技能规划，生成多样化的专家演示。这些演示通过视觉触觉扩散策略进行蒸馏，实现端到端执行的统一策略。我们进行了详尽的消融研究，探索不同的VLM基任务规划器，以确定最佳的演示生成管道，并系统地比较了技能蒸馏的模仿学习算法。广泛的仿真实验和物理部署验证了我们的方法在无需昂贵的人类演示的情况下实现了长周期操作策略学习，而VLM引导的原子技能框架能够实现对各种任务的大规模泛化。 

---
# VLAD-Grasp: Zero-shot Grasp Detection via Vision-Language Models 

**Title (ZH)**: VLAD-Grasp: 通过视觉-语言模型实现零样本抓取检测 

**Authors**: Manav Kulshrestha, S. Talha Bukhari, Damon Conover, Aniket Bera  

**Link**: [PDF](https://arxiv.org/pdf/2511.05791)  

**Abstract**: Robotic grasping is a fundamental capability for autonomous manipulation; however, most existing methods rely on large-scale expert annotations and necessitate retraining to handle new objects. We present VLAD-Grasp, a Vision-Language model Assisted zero-shot approach for Detecting grasps. From a single RGB-D image, our method (1) prompts a large vision-language model to generate a goal image where a straight rod "impales" the object, representing an antipodal grasp, (2) predicts depth and segmentation to lift this generated image into 3D, and (3) aligns generated and observed object point clouds via principal component analysis and correspondence-free optimization to recover an executable grasp pose. Unlike prior work, our approach is training-free and does not rely on curated grasp datasets. Despite this, VLAD-Grasp achieves performance that is competitive with or superior to that of state-of-the-art supervised models on the Cornell and Jacquard datasets. We further demonstrate zero-shot generalization to novel real-world objects on a Franka Research 3 robot, highlighting vision-language foundation models as powerful priors for robotic manipulation. 

**Abstract (ZH)**: Vision-Language模型辅助的零样本夹取检测方法 

---
# VLM-driven Skill Selection for Robotic Assembly Tasks 

**Title (ZH)**: 基于VLM的机器人装配任务技能选择 

**Authors**: Jeong-Jung Kim, Doo-Yeol Koh, Chang-Hyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.05680)  

**Abstract**: This paper presents a robotic assembly framework that combines Vision-Language Models (VLMs) with imitation learning for assembly manipulation tasks. Our system employs a gripper-equipped robot that moves in 3D space to perform assembly operations. The framework integrates visual perception, natural language understanding, and learned primitive skills to enable flexible and adaptive robotic manipulation. Experimental results demonstrate the effectiveness of our approach in assembly scenarios, achieving high success rates while maintaining interpretability through the structured primitive skill decomposition. 

**Abstract (ZH)**: 本文提出了一种结合视觉语言模型和模仿学习的机器人装配框架，用于装配操作任务。该系统采用配备机械手的机器人在三维空间中移动以执行装配操作。该框架整合了视觉感知、自然语言理解和学习到的基本技能，以实现灵活和适应性的机器人操作。实验结果表明，该方法在装配场景中的有效性，能够在保持可解释性的同时实现高的成功率，通过结构化的基本技能分解。 

---
# Lite VLA: Efficient Vision-Language-Action Control on CPU-Bound Edge Robots 

**Title (ZH)**: Lite VLA：基于CPU的边缘机器人高效视觉-语言-动作控制 

**Authors**: Justin Williams, Kishor Datta Gupta, Roy George, Mrinmoy Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2511.05642)  

**Abstract**: The deployment of artificial intelligence models at the edge is increasingly critical for autonomous robots operating in GPS-denied environments where local, resource-efficient reasoning is essential. This work demonstrates the feasibility of deploying small Vision-Language Models (VLMs) on mobile robots to achieve real-time scene understanding and reasoning under strict computational constraints. Unlike prior approaches that separate perception from mobility, the proposed framework enables simultaneous movement and reasoning in dynamic environments using only on-board hardware. The system integrates a compact VLM with multimodal perception to perform contextual interpretation directly on embedded hardware, eliminating reliance on cloud connectivity. Experimental validation highlights the balance between computational efficiency, task accuracy, and system responsiveness. Implementation on a mobile robot confirms one of the first successful deployments of small VLMs for concurrent reasoning and mobility at the edge. This work establishes a foundation for scalable, assured autonomy in applications such as service robotics, disaster response, and defense operations. 

**Abstract (ZH)**: 边缘部署人工智能模型在GPS受限环境中对自主机器人至关重要，尤其是在需要本地高效推理的情况下。本研究展示了在移动机器人上部署小型视觉语言模型(VLMs)以实现严格计算约束下的实时场景理解和推理的可行性。与以往将感知与移动隔离的方法不同，所提出的框架能够在仅使用机载硬件的情况下，同时实现动态环境中的运动和推理。该系统将紧凑的VLM与多模态感知集成，直接在嵌入式硬件上进行上下文解释，从而消除对云连接的依赖。实验验证突显了计算效率、任务准确性和系统响应性之间的平衡。在移动机器人上的实现证明了小型VLMs在边缘同时进行推理和移动的第一个成功部署之一。本研究为服务机器人、救灾和防御操作等应用中的可扩展和可靠的自主性奠定了基础。 

---
# TwinOR: Photorealistic Digital Twins of Dynamic Operating Rooms for Embodied AI Research 

**Title (ZH)**: TwinOR：动态手术室的photorealistic数字双胞胎及其在浸身式AI研究中的应用 

**Authors**: Han Zhang, Yiqing Shen, Roger D. Soberanis-Mukul, Ankita Ghosh, Hao Ding, Lalithkumar Seenivasan, Jose L. Porras, Zhekai Mao, Chenjia Li, Wenjie Xiao, Lonny Yarmus, Angela Christine Argento, Masaru Ishii, Mathias Unberath  

**Link**: [PDF](https://arxiv.org/pdf/2511.07412)  

**Abstract**: Developing embodied AI for intelligent surgical systems requires safe, controllable environments for continual learning and evaluation. However, safety regulations and operational constraints in operating rooms (ORs) limit embodied agents from freely perceiving and interacting in realistic settings. Digital twins provide high-fidelity, risk-free environments for exploration and training. How we may create photorealistic and dynamic digital representations of ORs that capture relevant spatial, visual, and behavioral complexity remains unclear. We introduce TwinOR, a framework for constructing photorealistic, dynamic digital twins of ORs for embodied AI research. The system reconstructs static geometry from pre-scan videos and continuously models human and equipment motion through multi-view perception of OR activities. The static and dynamic components are fused into an immersive 3D environment that supports controllable simulation and embodied exploration. The proposed framework reconstructs complete OR geometry with centimeter level accuracy while preserving dynamic interaction across surgical workflows, enabling realistic renderings and a virtual playground for embodied AI systems. In our experiments, TwinOR simulates stereo and monocular sensor streams for geometry understanding and visual localization tasks. Models such as FoundationStereo and ORB-SLAM3 on TwinOR-synthesized data achieve performance within their reported accuracy on real indoor datasets, demonstrating that TwinOR provides sensor-level realism sufficient for perception and localization challenges. By establishing a real-to-sim pipeline for constructing dynamic, photorealistic digital twins of OR environments, TwinOR enables the safe, scalable, and data-efficient development and benchmarking of embodied AI, ultimately accelerating the deployment of embodied AI from sim-to-real. 

**Abstract (ZH)**: 用于智能手术系统的具身AI开发需要安全可控的环境以实现持续学习和评估。然而，手术室中的安全规定和操作限制限制了具身代理在现实环境中自由感知和交互的能力。数字孪生提供了高保真、无风险的环境以供探索和训练。如何创建能够捕捉相关空间、视觉和行为复杂性的具身真实手术室（OR）的拟真和动态数字表示尚不清楚。我们介绍了TwinOR框架，用于构建具身AI研究中的拟真、动态数字孪生手术室。该系统从预扫描视频重建静态几何结构，并通过多视角感知手术室活动连续建模人类和设备运动。静态和动态组件融合成一个沉浸式的3D环境，支持可控模拟和具身探索。提出的方法以厘米级精度重建完整手术室几何结构，并保持跨手术工作流程的动态交互，使其能够生成逼真的渲染并为具身AI系统提供虚拟的实验平台。在我们的实验中，TwinOR模拟了立体和单目传感器流以进行几何理解和视觉定位任务。FoundationStereo和ORB-SLAM3等模型在TwinOR合成的数据上达到了与现实室内数据集报道的准确性相当的性能，表明TwinOR提供的传感器级真实性足以应对感知和定位挑战。通过建立从现实到模拟的流程以构建拟真、具身真实的手术室环境数字孪生，TwinOR使具身AI的安全、规模化和数据高效的开发和基准测试成为可能，最终加速了具身AI从模拟到现实的部署。 

---
# Multi-Agent Reinforcement Learning for Deadlock Handling among Autonomous Mobile Robots 

**Title (ZH)**: 多代理强化学习在自主移动机器人死锁处理中的应用 

**Authors**: Marcel Müller  

**Link**: [PDF](https://arxiv.org/pdf/2511.07071)  

**Abstract**: This dissertation explores the application of multi-agent reinforcement learning (MARL) for handling deadlocks in intralogistics systems that rely on autonomous mobile robots (AMRs). AMRs enhance operational flexibility but also increase the risk of deadlocks, which degrade system throughput and reliability. Existing approaches often neglect deadlock handling in the planning phase and rely on rigid control rules that cannot adapt to dynamic operational conditions.
To address these shortcomings, this work develops a structured methodology for integrating MARL into logistics planning and operational control. It introduces reference models that explicitly consider deadlock-capable multi-agent pathfinding (MAPF) problems, enabling systematic evaluation of MARL strategies. Using grid-based environments and an external simulation software, the study compares traditional deadlock handling strategies with MARL-based solutions, focusing on PPO and IMPALA algorithms under different training and execution modes.
Findings reveal that MARL-based strategies, particularly when combined with centralized training and decentralized execution (CTDE), outperform rule-based methods in complex, congested environments. In simpler environments or those with ample spatial freedom, rule-based methods remain competitive due to their lower computational demands. These results highlight that MARL provides a flexible and scalable solution for deadlock handling in dynamic intralogistics scenarios, but requires careful tailoring to the operational context. 

**Abstract (ZH)**: 本论文探讨了多智能体强化学习（MARL）在依赖自主移动机器人（AMRs）的内物流系统中处理死锁的应用。AMRs 提高了操作灵活性，但也增加了死锁的风险，从而降低系统吞吐量和可靠性。现有方法往往在规划阶段忽视死锁处理，并依赖于不能适应动态操作条件的刚性控制规则。

为了应对这些不足，本研究发展了一种结构化的方法，将MARL集成到物流规划和操作控制中。引入了考虑死锁能力的多智能体路径规划（MAPF）问题的参考模型，从而能够系统地评估MARL策略。通过基于网格的环境和外部仿真软件，研究比较了传统死锁处理策略与基于MARL的解决方案，重点关注在不同训练和执行模式下PPO和IMPALA算法的效果。

研究发现，特别是在集中的训练和分散的执行（CTDE）条件下，基于MARL的策略在复杂且拥堵的环境中明显优于基于规则的方法。在较简单或空间自由度较大的环境中，基于规则的方法仍然具有竞争力，因为它们需要更低的计算需求。这些结果表明，MARL为动态内物流场景中的死锁处理提供了灵活且可扩展的解决方案，但需要根据操作环境仔细调整。 

---
# PanoNav: Mapless Zero-Shot Object Navigation with Panoramic Scene Parsing and Dynamic Memory 

**Title (ZH)**: PanoNav：基于全景场景解析和动态记忆的零样本物体导航 

**Authors**: Qunchao Jin, Yilin Wu, Changhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.06840)  

**Abstract**: Zero-shot object navigation (ZSON) in unseen environments remains a challenging problem for household robots, requiring strong perceptual understanding and decision-making capabilities. While recent methods leverage metric maps and Large Language Models (LLMs), they often depend on depth sensors or prebuilt maps, limiting the spatial reasoning ability of Multimodal Large Language Models (MLLMs). Mapless ZSON approaches have emerged to address this, but they typically make short-sighted decisions, leading to local deadlocks due to a lack of historical context. We propose PanoNav, a fully RGB-only, mapless ZSON framework that integrates a Panoramic Scene Parsing module to unlock the spatial parsing potential of MLLMs from panoramic RGB inputs, and a Memory-guided Decision-Making mechanism enhanced by a Dynamic Bounded Memory Queue to incorporate exploration history and avoid local deadlocks. Experiments on the public navigation benchmark show that PanoNav significantly outperforms representative baselines in both SR and SPL metrics. 

**Abstract (ZH)**: 无地图零样本对象导航（Mapless Zero-shot Object Navigation, Mapless ZSON）在未见环境中的问题依然对家用机器人构成挑战，需要强大的感知理解和决策能力。尽管最近的方法利用了度量地图和大型语言模型（LLMs），但它们通常依赖于深度传感器或预构建的地图，限制了多模态大型语言模型（MLLMs）的空间推理能力。无地图ZSON方法已经出现以解决这一问题，但它们通常做出短视的决定，导致由于缺乏历史上下文而陷入局部死锁。我们提出了PanoNav，这是一种全RGB无地图ZSON框架，通过全景场景解析模块整合了来自全景RGB输入的空间解析潜力，并通过动态有界记忆队列增强的记忆导向决策机制来融入探索历史并避免局部死锁。在公开的导航基准测试中，PanoNav在SR和SPL指标上显著优于代表性基线。 

---
# Towards Human-AI-Robot Collaboration and AI-Agent based Digital Twins for Parkinson's Disease Management: Review and Outlook 

**Title (ZH)**: 面向帕金森病管理的人机机器人协作与基于AI代理的数字孪生：综述与展望 

**Authors**: Hassan Hizeh, Rim Chighri, Muhammad Mahboob Ur Rahman, Mohamed A. Bahloul, Ali Muqaibel, Tareq Y. Al-Naffouri  

**Link**: [PDF](https://arxiv.org/pdf/2511.06036)  

**Abstract**: The current body of research on Parkinson's disease (PD) screening, monitoring, and management has evolved along two largely independent trajectories. The first research community focuses on multimodal sensing of PD-related biomarkers using noninvasive technologies such as inertial measurement units (IMUs), force/pressure insoles, electromyography (EMG), electroencephalography (EEG), speech and acoustic analysis, and RGB/RGB-D motion capture systems. These studies emphasize data acquisition, feature extraction, and machine learning-based classification for PD screening, diagnosis, and disease progression modeling. In parallel, a second research community has concentrated on robotic intervention and rehabilitation, employing socially assistive robots (SARs), robot-assisted rehabilitation (RAR) systems, and virtual reality (VR)-integrated robotic platforms for improving motor and cognitive function, enhancing social engagement, and supporting caregivers. Despite the complementary goals of these two domains, their methodological and technological integration remains limited, with minimal data- level or decision-level coupling between the two. With the advent of advanced artificial intelligence (AI), including large language models (LLMs), agentic AI systems, a unique opportunity now exists to unify these research streams. We envision a closed-loop sensor-AI-robot framework in which multimodal sensing continuously guides the interaction between the patient, caregiver, humanoid robot (and physician) through AI agents that are powered by a multitude of AI models such as robotic and wearables foundation models, LLM-based reasoning, reinforcement learning, and continual learning. Such closed-loop system enables personalized, explainable, and context-aware intervention, forming the basis for digital twin of the PD patient that can adapt over time to deliver intelligent, patient-centered PD care. 

**Abstract (ZH)**: 帕金森病(PD)筛查、监测与管理的研究进展主要沿着两条相对独立的轨迹发展。 

---
# Social-Physical Interactions with Virtual Characters: Evaluating the Impact of Physicality through Encountered-Type Haptics 

**Title (ZH)**: 社会物理交互中的虚拟角色：通过遭遇型触觉评估物理性的影响 

**Authors**: Eric Godden, Jacquie Groenewegen, Michael Wheeler, Matthew K.X.J. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2511.05683)  

**Abstract**: This work investigates how robot-mediated physicality influences the perception of social-physical interactions with virtual characters. ETHOS (Encountered-Type Haptics for On-demand Social interaction) is an encountered-type haptic display that integrates a torque-controlled manipulator and interchangeable props with a VR headset to enable three gestures: object handovers, fist bumps, and high fives. We conducted a user study to examine how ETHOS adds physicality to virtual character interactions and how this affects presence, realism, enjoyment, and connection metrics. Each participant experienced one interaction under three conditions: no physicality (NP), static physicality (SP), and dynamic physicality (DP). SP extended the purely virtual baseline (NP) by introducing tangible props for direct contact, while DP further incorporated motion and impact forces to emulate natural touch. Results show presence increased stepwise from NP to SP to DP. Realism, enjoyment, and connection also improved with added physicality, though differences between SP and DP were not significant. Comfort remained consistent across conditions, indicating no added psychological friction. These findings demonstrate the experiential value of ETHOS and motivate the integration of encountered-type haptics into socially meaningful VR experiences. 

**Abstract (ZH)**: 本研究探讨了机器人介导的物理互动如何影响对虚拟角色社交物理互动的感知。ETHOS（遇触型即时社交触感显示）是一种遇触型触觉显示技术，结合了扭矩控制 manipulator 和可更换道具与 VR 头显，以实现物体交接、拳击和击掌三种手势。我们进行了用户研究，以了解 ETHOS 如何为虚拟角色互动增加物理感，并评估这种增加如何影响存在感、逼真度、乐趣和连接感指标。每位参与者在三种条件下体验了一次互动：无物理感（NP）、静态物理感（SP）和动态物理感（DP）。SP 在 NP 的纯虚拟基础上引入了实体道具实现直接接触，而 DP 进一步结合了运动和冲击力以模拟真实触感。结果显示，存在感从 NP 到 SP 再到 DP 呈阶梯式增加。逼真度、乐趣和连接感也随着物理感觉的增加而提高，尽管 SP 和 DP 之间的差异并不显著。舒适度在各条件下保持一致，表明没有增加心理摩擦。这些发现证明了 ETHOS 的体验价值，并激发了将遇触型触觉集成到社交有意义的 VR 体验中的动机。 

---
# Grounding Foundational Vision Models with 3D Human Poses for Robust Action Recognition 

**Title (ZH)**: 基于3D人体姿态的奠基性视觉模型 grounding 用于稳健的动作识别 

**Authors**: Nicholas Babey, Tiffany Gu, Yiheng Li, Cristian Meo, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05622)  

**Abstract**: For embodied agents to effectively understand and interact within the world around them, they require a nuanced comprehension of human actions grounded in physical space. Current action recognition models, often relying on RGB video, learn superficial correlations between patterns and action labels, so they struggle to capture underlying physical interaction dynamics and human poses in complex scenes. We propose a model architecture that grounds action recognition in physical space by fusing two powerful, complementary representations: V-JEPA 2's contextual, predictive world dynamics and CoMotion's explicit, occlusion-tolerant human pose data. Our model is validated on both the InHARD and UCF-19-Y-OCC benchmarks for general action recognition and high-occlusion action recognition, respectively. Our model outperforms three other baselines, especially within complex, occlusive scenes. Our findings emphasize a need for action recognition to be supported by spatial understanding instead of statistical pattern recognition. 

**Abstract (ZH)**: 基于物理空间的肢体动作识别模型：结合情境预测的世界动态与显式姿态数据 

---
# GRAPH-GRPO-LEX: Contract Graph Modeling and Reinforcement Learning with Group Relative Policy Optimization 

**Title (ZH)**: GRAPH-GRPO-LEX: 合同图建模与基于群体相对策略优化的强化学习 

**Authors**: Moriya Dechtiar, Daniel Martin Katz, Mari Sundaresan, Sylvain Jaume, Hongming Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06618)  

**Abstract**: Contracts are complex documents featuring detailed formal structures, explicit and implicit dependencies and rich semantic content. Given these document properties, contract drafting and manual examination of contracts have proven to be both arduous and susceptible to errors. This work aims to simplify and automate the task of contract review and analysis using a novel framework for transforming legal contracts into structured semantic graphs, enabling computational analysis and data-driven insights. We introduce a detailed ontology mapping core legal contract elements to their graph-theoretic equivalents of nodes and edges. We then present a reinforcement learning based Large Language Model (LLM) framework for segmentation and extraction of entities and relationships from contracts. Our method, GRAPH-GRPO-LEX, incorporates both LLMs and reinforcement learning with group relative policy optimization (GRPO). By applying a carefully drafted reward function of graph metrics, we demonstrate the ability to automatically identify direct relationships between clauses, and even uncover hidden dependencies. Our introduction of the gated GRPO approach shows a strong learning signal and can move contract analysis from a linear, manual reading process to an easily visualized graph. This allows for a more dynamic analysis, including building the groundwork for contract linting similar to what is now practiced in software engineering. 

**Abstract (ZH)**: 基于图结构强化学习的合同审查与分析框架 

---
# ALIGN: A Vision-Language Framework for High-Accuracy Accident Location Inference through Geo-Spatial Neural Reasoning 

**Title (ZH)**: ALIGN：一种通过地理空间神经推理实现高精度事故位置推断的视觉-语言框架 

**Authors**: MD Thamed Bin Zaman Chowdhury, Moazzem Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2511.06316)  

**Abstract**: Reliable geospatial information on road accidents is vital for safety analysis and infrastructure planning, yet most low- and middle-income countries continue to face a critical shortage of accurate, location-specific crash data. Existing text-based geocoding tools perform poorly in multilingual and unstructured news environments, where incomplete place descriptions and mixed Bangla-English scripts obscure spatial context. To address these limitations, this study introduces ALIGN (Accident Location Inference through Geo-Spatial Neural Reasoning)- a vision-language framework that emulates human spatial reasoning to infer accident coordinates directly from textual and map-based cues. ALIGN integrates large language and vision-language models within a multi-stage pipeline that performs optical character recognition, linguistic reasoning, and map-level verification through grid-based spatial scanning. The framework systematically evaluates each predicted location against contextual and visual evidence, ensuring interpretable, fine-grained geolocation outcomes without requiring model retraining. Applied to Bangla-language news data, ALIGN demonstrates consistent improvements over traditional geoparsing methods, accurately identifying district and sub-district-level crash sites. Beyond its technical contribution, the framework establishes a high accuracy foundation for automated crash mapping in data-scarce regions, supporting evidence-driven road-safety policymaking and the broader integration of multimodal artificial intelligence in transportation analytics. The code for this paper is open-source and available at: this https URL 

**Abstract (ZH)**: 可靠的地理空间信息对于事故安全分析和基础设施规划至关重要，然而大多数低收入和中收入国家仍面临准确、位置特定的事故数据严重短缺的问题。现有的基于文本的地理编码工具在多语言和未结构化的新闻环境中表现不佳，其中不完整的地点描述和混合的孟加拉语-英语脚本模糊了空间上下文。为了应对这些局限性，本研究引入了ALIGN（通过地理空间神经推理推断事故地点）——一种视图-语言框架，模仿人类的空间推理能力，直接从文本和地图线索中推断事故坐标。ALIGN在多阶段管道中整合了大型语言模型和视图-语言模型，管道中包括光学字符识别、语言推理和基于网格的地理扫描地图层次验证。该框架系统地将每个预测位置与上下文和视觉证据进行比较，确保在无需模型重新训练的情况下获得可解释的高精度地理定位结果。应用于孟加拉语新闻数据，ALIGN在传统地理解析方法上显示出一致的改进，准确识别出区级和县级事故地点。除了技术贡献外，该框架为数据稀缺地区自动事故地图绘制奠定了高准确度基础，支持基于证据的道路安全政策制定，并促进多模态人工智能在交通分析中的更广泛集成。本文代码开源，可在以下链接获取：this https URL。 

---
# The Station: An Open-World Environment for AI-Driven Discovery 

**Title (ZH)**: The Station：由AI驱动的发现的开放世界环境 

**Authors**: Stephen Chung, Wenyu Du  

**Link**: [PDF](https://arxiv.org/pdf/2511.06309)  

**Abstract**: We introduce the STATION, an open-world multi-agent environment that models a miniature scientific ecosystem. Leveraging their extended context windows, agents in the Station can engage in long scientific journeys that include reading papers from peers, formulating hypotheses, submitting code, performing analyses, and publishing results. Importantly, there is no centralized system coordinating their activities - agents are free to choose their own actions and develop their own narratives within the Station. Experiments demonstrate that AI agents in the Station achieve new state-of-the-art performance on a wide range of benchmarks, spanning from mathematics to computational biology to machine learning, notably surpassing AlphaEvolve in circle packing. A rich tapestry of narratives emerges as agents pursue independent research, interact with peers, and build upon a cumulative history. From these emergent narratives, novel methods arise organically, such as a new density-adaptive algorithm for scRNA-seq batch integration. The Station marks a first step towards autonomous scientific discovery driven by emergent behavior in an open-world environment, representing a new paradigm that moves beyond rigid optimization. 

**Abstract (ZH)**: 我们介绍了STATION，一个开放世界多智能体环境，模拟了一个微型科学生态系统。借助其扩展的上下文窗口，STATION中的智能体可以参与长期的科学探索旅程，包括阅读同行论文、提出假设、提交代码、进行分析和发布结果。重要的是，没有集中协调系统协调其活动——智能体可以自由选择自己的行动并在STATION中发展自己的叙述。实验表明，STATION中的AI智能体在从数学到计算生物学再到机器学习等多个基准测试中实现了新的最先进性能，特别是在圆盘填充任务中显著超越了AlphaEvolve。随着智能体追求独立研究、与同行互动并建立累积历史，丰富的叙述图谱逐渐形成。从这些涌现的叙述中，新的方法有机地产生，例如一种新的密度自适应的单细胞RNA测序批次整合算法。STATION标志着朝向由开放世界环境中涌现行为推动的自主科学研究迈出的第一步，这代表着一种超越刚性优化的新范式。 

---
# GAIA: A General Agency Interaction Architecture for LLM-Human B2B Negotiation & Screening 

**Title (ZH)**: GAIA：一种通用代理交互架构，用于LLM-人类B2B谈判与筛选 

**Authors**: Siming Zhao, Qi Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06262)  

**Abstract**: Organizations are increasingly exploring delegation of screening and negotiation tasks to AI systems, yet deployment in high-stakes B2B settings is constrained by governance: preventing unauthorized commitments, ensuring sufficient information before bargaining, and maintaining effective human oversight and auditability. Prior work on large language model negotiation largely emphasizes autonomous bargaining between agents and omits practical needs such as staged information gathering, explicit authorization boundaries, and systematic feedback integration. We propose GAIA, a governance-first framework for LLM-human agency in B2B negotiation and screening. GAIA defines three essential roles - Principal (human), Delegate (LLM agent), and Counterparty - with an optional Critic to enhance performance, and organizes interactions through three mechanisms: information-gated progression that separates screening from negotiation; dual feedback integration that combines AI critique with lightweight human corrections; and authorization boundaries with explicit escalation paths. Our contributions are fourfold: (1) a formal governance framework with three coordinated mechanisms and four safety invariants for delegation with bounded authorization; (2) information-gated progression via task-completeness tracking (TCI) and explicit state transitions that separate screening from commitment; (3) dual feedback integration that blends Critic suggestions with human oversight through parallel learning channels; and (4) a hybrid validation blueprint that combines automated protocol metrics with human judgment of outcomes and safety. By bridging theory and practice, GAIA offers a reproducible specification for safe, efficient, and accountable AI delegation that can be instantiated across procurement, real estate, and staffing workflows. 

**Abstract (ZH)**: GAIA：治理优先的大型语言模型在B2B谈判和筛选中的代理框架 

---
# When Object-Centric World Models Meet Policy Learning: From Pixels to Policies, and Where It Breaks 

**Title (ZH)**: 当对象中心的世界模型遇到策略学习：从像素到策略，以及其局限性 

**Authors**: Stefano Ferraro, Akihiro Nakano, Masahiro Suzuki, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2511.06136)  

**Abstract**: Object-centric world models (OCWM) aim to decompose visual scenes into object-level representations, providing structured abstractions that could improve compositional generalization and data efficiency in reinforcement learning. We hypothesize that explicitly disentangled object-level representations, by localizing task-relevant information, can enhance policy performance across novel feature combinations. To test this hypothesis, we introduce DLPWM, a fully unsupervised, disentangled object-centric world model that learns object-level latents directly from pixels. DLPWM achieves strong reconstruction and prediction performance, including robustness to several out-of-distribution (OOD) visual variations. However, when used for downstream model-based control, policies trained on DLPWM latents underperform compared to DreamerV3. Through latent-trajectory analyses, we identify representation shift during multi-object interactions as a key driver of unstable policy learning. Our results suggest that, although object-centric perception supports robust visual modeling, achieving stable control requires mitigating latent drift. 

**Abstract (ZH)**: 以物为中心的世界模型（OCWM）旨在将视觉场景分解为对象级表示，提供结构化的抽象，以提高强化学习中的组合泛化能力和数据效率。我们假设通过分离的对象级表示，本地化与任务相关的信息，可以增强在新颖特征组合下的策略性能。为了验证这一假设，我们引入了DLPWM，这是一种完全无监督的分离的以物为中心的世界模型，直接从像素中学习对象级的潜在变量。DLPWM在重建和预测性能方面表现出色，并且对多种离分布（OOD）的视觉变化具有鲁棒性。然而，在下游基于模型的控制中，使用DLPWM潜在变量训练的策略的表现不如DreamerV3。通过潜在轨迹分析，我们将多对象交互过程中的表示转移识别为不稳定策略学习的关键驱动因素。我们的结果表明，尽管以物为中心的感觉支持稳健的视觉建模，但实现稳定控制需要缓解潜在变量漂移。 

---
# Surgical Agent Orchestration Platform for Voice-directed Patient Data Interaction 

**Title (ZH)**: 手术机器人指挥平台语音指导患者数据交互 

**Authors**: Hyeryun Park, Byung Mo Gu, Jun Hee Lee, Byeong Hyeon Choi, Sekeun Kim, Hyun Koo Kim, Kyungsang Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.07392)  

**Abstract**: In da Vinci robotic surgery, surgeons' hands and eyes are fully engaged in the procedure, making it difficult to access and manipulate multimodal patient data without interruption. We propose a voice-directed Surgical Agent Orchestrator Platform (SAOP) built on a hierarchical multi-agent framework, consisting of an orchestration agent and three task-specific agents driven by Large Language Models (LLMs). These LLM-based agents autonomously plan, refine, validate, and reason to map voice commands into specific tasks such as retrieving clinical information, manipulating CT scans, or navigating 3D anatomical models on the surgical video. We also introduce a Multi-level Orchestration Evaluation Metric (MOEM) to comprehensively assess the performance and robustness from command-level and category-level perspectives. The SAOP achieves high accuracy and success rates across 240 voice commands, while LLM-based agents improve robustness against speech recognition errors and diverse or ambiguous free-form commands, demonstrating strong potential to support minimally invasive da Vinci robotic surgery. 

**Abstract (ZH)**: 基于多层次多Agent框架的语音指导外科智能调度平台（SAOP）：支持微创达芬奇机器人手术 

---
# Grounding Computer Use Agents on Human Demonstrations 

**Title (ZH)**: 基于人类示范地约束计算机使用代理 

**Authors**: Aarash Feizi, Shravan Nayak, Xiangru Jian, Kevin Qinghong Lin, Kaixin Li, Rabiul Awal, Xing Han Lù, Johan Obando-Ceron, Juan A. Rodriguez, Nicolas Chapados, David Vazquez, Adriana Romero-Soriano, Reihaneh Rabbany, Perouz Taslakian, Christopher Pal, Spandana Gella, Sai Rajeswar  

**Link**: [PDF](https://arxiv.org/pdf/2511.07332)  

**Abstract**: Building reliable computer-use agents requires grounding: accurately connecting natural language instructions to the correct on-screen elements. While large datasets exist for web and mobile interactions, high-quality resources for desktop environments are limited. To address this gap, we introduce GroundCUA, a large-scale desktop grounding dataset built from expert human demonstrations. It covers 87 applications across 12 categories and includes 56K screenshots, with every on-screen element carefully annotated for a total of over 3.56M human-verified annotations. From these demonstrations, we generate diverse instructions that capture a wide range of real-world tasks, providing high-quality data for model training. Using GroundCUA, we develop the GroundNext family of models that map instructions to their target UI elements. At both 3B and 7B scales, GroundNext achieves state-of-the-art results across five benchmarks using supervised fine-tuning, while requiring less than one-tenth the training data of prior work. Reinforcement learning post-training further improves performance, and when evaluated in an agentic setting on the OSWorld benchmark using o3 as planner, GroundNext attains comparable or superior results to models trained with substantially more data,. These results demonstrate the critical role of high-quality, expert-driven datasets in advancing general-purpose computer-use agents. 

**Abstract (ZH)**: 构建可靠的计算机使用代理需要接地：准确地将自然语言指令与正确的屏幕元素连接起来。尽管存在大量的网络和移动交互数据集，桌面环境的高质量资源仍然有限。为了解决这一差距，我们引入了GroundCUA，这是一个从专家人工演示构建的大规模桌面接地数据集。它涵盖了12个类别中的87个应用程序，并包含了56K截图，每个屏幕元素都经过细致标注，总共有超过3.56M的人工验证注释。从这些演示中，我们生成了多种多样的指令，涵盖了广泛的实际任务，为模型训练提供了高质量的数据。使用GroundCUA，我们开发了GroundNext家族的模型，将指令映射到其目标UI元素。无论是3B参数还是7B参数规模，GroundNext在五个基准测试中均实现了最先进的结果，同时所需的训练数据量仅为以前工作的十分之一。强化学习后训练进一步提高了性能，并在使用o3作为规划者的OSWorld基准测试中评估时，GroundNext在训练数据量显著较少的情况下获得了具有竞争力或更优的结果。这些结果表明，高质量的专家驱动数据集在推动通用计算机使用代理方面起着关键作用。 

---
# Learning to Focus: Prioritizing Informative Histories with Structured Attention Mechanisms in Partially Observable Reinforcement Learning 

**Title (ZH)**: 学习聚焦：在部分可观测强化学习中使用结构化注意力机制优先处理信息性的历史记录 

**Authors**: Daniel De Dios Allegue, Jinke He, Frans A. Oliehoek  

**Link**: [PDF](https://arxiv.org/pdf/2511.06946)  

**Abstract**: Transformers have shown strong ability to model long-term dependencies and are increasingly adopted as world models in model-based reinforcement learning (RL) under partial observability. However, unlike natural language corpora, RL trajectories are sparse and reward-driven, making standard self-attention inefficient because it distributes weight uniformly across all past tokens rather than emphasizing the few transitions critical for control. To address this, we introduce structured inductive priors into the self-attention mechanism of the dynamics head: (i) per-head memory-length priors that constrain attention to task-specific windows, and (ii) distributional priors that learn smooth Gaussian weightings over past state-action pairs. We integrate these mechanisms into UniZero, a model-based RL agent with a Transformer-based world model that supports planning under partial observability. Experiments on the Atari 100k benchmark show that most efficiency gains arise from the Gaussian prior, which smoothly allocates attention to informative transitions, while memory-length priors often truncate useful signals with overly restrictive cut-offs. In particular, Gaussian Attention achieves a 77% relative improvement in mean human-normalized scores over UniZero. These findings suggest that in partially observable RL domains with non-stationary temporal dependencies, discrete memory windows are difficult to learn reliably, whereas smooth distributional priors flexibly adapt across horizons and yield more robust data efficiency. Overall, our results demonstrate that encoding structured temporal priors directly into self-attention improves the prioritization of informative histories for dynamics modeling under partial observability. 

**Abstract (ZH)**: 具有结构诱导先验的Transformers在部分可观测模型基于强化学习中的自注意力机制改进 

---
# Controllable Flow Matching for Online Reinforcement Learning 

**Title (ZH)**: 可控流匹配的在线强化学习 

**Authors**: Bin Wang, Boxiang Tao, Haifeng Jing, Hongbo Dou, Zijian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06816)  

**Abstract**: Model-based reinforcement learning (MBRL) typically relies on modeling environment dynamics for data efficiency. However, due to the accumulation of model errors over long-horizon rollouts, such methods often face challenges in maintaining modeling stability. To address this, we propose CtrlFlow, a trajectory-level synthetic method using conditional flow matching (CFM), which directly modeling the distribution of trajectories from initial states to high-return terminal states without explicitly modeling the environment transition function. Our method ensures optimal trajectory sampling by minimizing the control energy governed by the non-linear Controllability Gramian Matrix, while the generated diverse trajectory data significantly enhances the robustness and cross-task generalization of policy learning. In online settings, CtrlFlow demonstrates the better performance on common MuJoCo benchmark tasks than dynamics models and achieves superior sample efficiency compared to standard MBRL methods. 

**Abstract (ZH)**: 基于模型的强化学习（MBRL）通常依赖于环境动力学建模以提高数据效率。然而，由于长期轨迹采样中模型误差的累积，此类方法往往难以保持建模稳定性。为此，我们提出了CtrlFlow，一种基于轨迹级合成的方法，使用条件流匹配（CFM）直接建模从初始状态到高回报终态的轨迹分布，而不显式建模环境转换函数。该方法通过最小化由非线性可控性Gram矩阵支配的控制能量来确保最优轨迹采样，而生成的多样化轨迹数据显著增强了策略学习的稳健性和跨任务泛化能力。在线设置中，CtrlFlow在常见的MuJoCo基准任务上表现出更好的性能，相比动力学模型具有更高的样本效率，并且优于标准的MBRL方法。 

---
# AgentSUMO: An Agentic Framework for Interactive Simulation Scenario Generation in SUMO via Large Language Models 

**Title (ZH)**: AgentSUMO: 一种基于大型语言模型的SUMO互动仿真场景生成框架 

**Authors**: Minwoo Jeong, Jeeyun Chang, Yoonjin Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2511.06804)  

**Abstract**: The growing complexity of urban mobility systems has made traffic simulation indispensable for evidence-based transportation planning and policy evaluation. However, despite the analytical capabilities of platforms such as the Simulation of Urban MObility (SUMO), their application remains largely confined to domain experts. Developing realistic simulation scenarios requires expertise in network construction, origin-destination modeling, and parameter configuration for policy experimentation, creating substantial barriers for non-expert users such as policymakers, urban planners, and city officials. Moreover, the requests expressed by these users are often incomplete and abstract-typically articulated as high-level objectives, which are not well aligned with the imperative, sequential workflows employed in existing language-model-based simulation frameworks. To address these challenges, this study proposes AgentSUMO, an agentic framework for interactive simulation scenario generation via large language models. AgentSUMO departs from imperative, command-driven execution by introducing an adaptive reasoning layer that interprets user intents, assesses task complexity, infers missing parameters, and formulates executable simulation plans. The framework is structured around two complementary components, the Interactive Planning Protocol, which governs reasoning and user interaction, and the Model Context Protocol, which manages standardized communication and orchestration among simulation tools. Through this design, AgentSUMO converts abstract policy objectives into executable simulation scenarios. Experiments on urban networks in Seoul and Manhattan demonstrate that the agentic workflow achieves substantial improvements in traffic flow metrics while maintaining accessibility for non-expert users, successfully bridging the gap between policy goals and executable simulation workflows. 

**Abstract (ZH)**: 城市交通系统的日益复杂使得交通仿真对于基于证据的交通规划和政策评价变得不可或缺。然而，尽管有如Simulation of Urban MObility (SUMO)等平台的分析能力，其应用仍主要局限于领域专家。为开发现实的仿真场景，非专家用户（如政策制定者、城市规划师和城市官员）需要具备网络构建、出行生成建模和政策实验参数配置的专业知识，这创造了相当大的障碍。此外，这些用户的请求往往不完整且抽象，通常仅以高层次目标的形式提出，而这些目标与现有基于语言模型的仿真框架中所采用的严格、顺序化的作业流程并不很好对接。为解决这些挑战，本研究提出AgentSUMO，这是一种通过大型语言模型进行交互式仿真场景生成的代理框架。AgentSUMO通过引入一个适应性推理层，该层解释用户意图、评估任务复杂性、推断缺失参数并制定可执行的仿真计划，从而偏离了基于命令的执行方式。该框架围绕两个互补组件构建，交互式规划协议规范推理和用户交互，而模型上下文协议则管理仿真工具之间的标准化通信和协调。通过该设计，AgentSUMO将抽象的政策目标转化为可执行的仿真场景。对首尔和曼哈顿城市网络的实验表明，代理式作业流程在保持非专家用户可用性的同时，在交通流指标上实现了显著改善，成功地弥合了政策目标与可执行仿真作业流程之间的差距。 

---
# Structural Enforcement of Statistical Rigor in AI-Driven Discovery: A Functional Architecture 

**Title (ZH)**: AI驱动发现中结构性贯彻统计严谨性的功能架构 

**Authors**: Karen Sargsyan  

**Link**: [PDF](https://arxiv.org/pdf/2511.06701)  

**Abstract**: Sequential statistical protocols require meticulous state management and robust error handling -- challenges naturally suited to functional programming. We present a functional architecture for structural enforcement of statistical rigor in automated research systems (AI-Scientists). These LLM-driven systems risk generating spurious discoveries through dynamic hypothesis testing. We introduce the Research monad, a Haskell eDSL that enforces sequential statistical protocols (e.g., Online FDR (false discovery rate) control) using a monad transformer stack. To address risks in hybrid architectures where LLMs generate imperative code, we employ Declarative Scaffolding -- generating rigid harnesses that structurally constrain execution and prevent methodological errors like data leakage. We validate this approach through large-scale simulation (N=2000 hypotheses) and an end-to-end case study, demonstrating essential defense-in-depth for automated science integrity. 

**Abstract (ZH)**: 顺序统计协议需要细致的状态管理和 robust 错误处理——这自然是函数式编程的天然挑战。我们提出了一种函数式架构，用于在自动化研究系统（AI-科学家）中结构性地保障统计严谨性。这些由大语言模型驱动的系统通过动态假设检验存在产生虚假发现的风险。我们引入了 Research monad，这是一种 Haskell 的 eDSL，使用монад变换器栈来强制执行顺序统计协议（例如，Online FDR（错误发现率）控制）。为了解决混合架构中大语言模型生成命令式代码所带来的风险，我们采用了声明性支架——生成刚性框架以结构性约束执行并防止诸如数据泄露等方法论错误。我们通过大规模仿真（N=2000 假设）和端到端案例研究验证了这种方法，展示了自动化科学研究中必不可少的纵深防御。 

---
# Ibom NLP: A Step Toward Inclusive Natural Language Processing for Nigeria's Minority Languages 

**Title (ZH)**: Ibom NLP：迈向包容性的尼日利亚少数民族语言自然语言处理 

**Authors**: Oluwadara Kalejaiye, Luel Hagos Beyene, David Ifeoluwa Adelani, Mmekut-Mfon Gabriel Edet, Aniefon Daniel Akpan, Eno-Abasi Urua, Anietie Andy  

**Link**: [PDF](https://arxiv.org/pdf/2511.06531)  

**Abstract**: Nigeria is the most populous country in Africa with a population of more than 200 million people. More than 500 languages are spoken in Nigeria and it is one of the most linguistically diverse countries in the world. Despite this, natural language processing (NLP) research has mostly focused on the following four languages: Hausa, Igbo, Nigerian-Pidgin, and Yoruba (i.e <1% of the languages spoken in Nigeria). This is in part due to the unavailability of textual data in these languages to train and apply NLP algorithms. In this work, we introduce ibom -- a dataset for machine translation and topic classification in four Coastal Nigerian languages from the Akwa Ibom State region: Anaang, Efik, Ibibio, and Oro. These languages are not represented in Google Translate or in major benchmarks such as Flores-200 or SIB-200. We focus on extending Flores-200 benchmark to these languages, and further align the translated texts with topic labels based on SIB-200 classification dataset. Our evaluation shows that current LLMs perform poorly on machine translation for these languages in both zero-and-few shot settings. However, we find the few-shot samples to steadily improve topic classification with more shots. 

**Abstract (ZH)**: 尼日利亚是非洲人口最多的国家，人口超过2亿，使用超过500种语言，是世界上语言最为多元的国家之一。尽管如此，自然语言处理（NLP）研究大多集中于豪萨语、伊博语、尼日利亚皮钦语和约鲁巴语这四种语言（即不到尼日利亚所用语言的1%）。这在一定程度上是因为这些语言缺乏文本数据以训练和应用NLP算法。在本项工作中，我们介绍了ibom——一个包含来自阿夸伊博姆州沿海地区的阿郎冈语、埃菲克语、伊比比奥语和奥罗语的机器翻译和主题分类数据集。这些语言在谷歌翻译和其他主要基准如Flores-200和SIB-200中均未被涵盖。我们致力于将Flores-200基准扩展到这些语言，并进一步依据SIB-200分类数据集对翻译文本进行主题标签对齐。我们的评估表明，目前的预训练语言模型在零样本和少样本设置下的机器翻译性能欠佳。然而，我们发现少量样本能够逐步提高主题分类性能。 

---
# MrCoM: A Meta-Regularized World-Model Generalizing Across Multi-Scenarios 

**Title (ZH)**: MrCoM: 一种针对多场景进行泛化的元正则化世界模型 

**Authors**: Xuantang Xiong, Ni Mu, Runpeng Xie, Senhao Yang, Yaqing Wang, Lexiang Wang, Yao Luan, Siyuan Li, Shuang Xu, Yiqin Yang, Bo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06252)  

**Abstract**: Model-based reinforcement learning (MBRL) is a crucial approach to enhance the generalization capabilities and improve the sample efficiency of RL algorithms. However, current MBRL methods focus primarily on building world models for single tasks and rarely address generalization across different scenarios. Building on the insight that dynamics within the same simulation engine share inherent properties, we attempt to construct a unified world model capable of generalizing across different scenarios, named Meta-Regularized Contextual World-Model (MrCoM). This method first decomposes the latent state space into various components based on the dynamic characteristics, thereby enhancing the accuracy of world-model prediction. Further, MrCoM adopts meta-state regularization to extract unified representation of scenario-relevant information, and meta-value regularization to align world-model optimization with policy learning across diverse scenario objectives. We theoretically analyze the generalization error upper bound of MrCoM in multi-scenario settings. We systematically evaluate our algorithm's generalization ability across diverse scenarios, demonstrating significantly better performance than previous state-of-the-art methods. 

**Abstract (ZH)**: 基于模型的强化学习（MBRL）是提升RL算法泛化能力和提高样本效率的关键方法。然而，当前的MBRL方法主要集中在构建针对单一任务的世界模型，很少解决不同场景间的泛化问题。基于同一仿真引擎内部动力学共享内在性质的洞察，我们尝试构建一个能够跨不同场景泛化的一致性世界模型，名为Meta-正则化上下文世界模型（MrCoM）。该方法首先基于动力学特性将潜状态空间分解成多个组成部分，从而提高世界模型预测的准确性。进一步地，MrCoM采用元状态正则化提取与场景相关的统一表示，并采用元价值正则化使世界模型优化与多样化场景目标下的策略学习保持一致。我们在多场景设置中理论分析了MrCoM的泛化误差上界。我们系统地评估了算法在不同场景中的泛化能力，显示出显著优于先前最先进的方法的性能。 

---
# EVLP:Learning Unified Embodied Vision-Language Planner with Reinforced Supervised Fine-Tuning 

**Title (ZH)**: EVLP：学习统一的感知语言规划器with强化监督微调 

**Authors**: Xinyan Cai, Shiguang Wu, Dafeng Chi, Yuzheng Zhuang, Xingyue Quan, Jianye Hao, Qiang Guan  

**Link**: [PDF](https://arxiv.org/pdf/2511.05553)  

**Abstract**: In complex embodied long-horizon manipulation tasks, effective task decomposition and execution require synergistic integration of textual logical reasoning and visual-spatial imagination to ensure efficient and accurate operation. Current methods fail to adopt a unified generation framework for multimodal planning, lead to inconsistent in multimodal planning. To address this challenge, we present \textbf{EVLP (Embodied Vision-Language Planner)}, an innovative multimodal unified generation framework that jointly models linguistic reasoning and visual generation. Our approach achieves multimodal planning for long-horizon tasks through a novel training pipeline incorporating dynamic pretraining and reinforced alignment. Our core innovations consist of three key components: \textbf{1) Unified Multimodal Generation Framework}: For understanding, We integrate semantic information with spatial features to provide comprehensive visual perception. For generation, we directly learn the joint distribution of discrete images for one-step visual synthesis, enabling coordinated language-visual modeling through learnable cross-modal attention mechanisms. \textbf{2) Dynamic Perception Pretraining}: We propose a bidirectional dynamic alignment strategy employing inverse dynamics tasks and forward dynamics tasks, effectively strengthening multimodal correlations within a unified feature space. \textbf{3) Reinforced Supervised Fine-Tuning}: While conducting instruction-based fine-tuning in the unified generation space, we construct a reinforce loss to align the spatial logic between textual actions and generated images, enabling the model to acquire spatio-awared multimodal planning capabilities. 

**Abstract (ZH)**: 在复杂体态长时程操作任务中，有效的任务分解和执行需要文本逻辑推理和视觉空间想象力的协同整合，以确保操作的高效和准确。当前的方法未能为多模态规划采用统一生成框架，导致多模态规划中的一致性问题。为应对这一挑战，我们提出了**EVLP（体态视觉-语言规划者）**，这是一种创新的多模态统一生成框架，联合建模语言推理和视觉生成。我们的方法通过结合动态预训练和强化对齐的新型训练管道，实现长时程任务的多模态规划。我们的核心创新包括三个关键组成部分：**1）统一多模态生成框架**：在理解中，我们将语义信息与空间特征集成，提供全面的视觉感知。在生成中，我们直接学习离散图像的一步视觉合成的联合分布，通过可学习的跨模态注意力机制实现协调的语言-视觉建模。**2）动态感知预训练**：我们提出了一种双向动态对齐策略，采用逆动力学任务和正动力学任务，有效地在统一特征空间内增强多模态相关性。**3）强化监督微调**：在统一生成空间中执行指令驱动的微调时，我们构建了一个强化损失，以对齐文本动作和生成图像之间的空间逻辑，使模型能够获得具有空间意识的多模态规划能力。 

---
