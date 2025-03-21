# RoboFactory: Exploring Embodied Agent Collaboration with Compositional Constraints 

**Title (ZH)**: RoboFactory: 探索组合约束下的实体代理协作 

**Authors**: Yiran Qin, Li Kang, Xiufeng Song, Zhenfei Yin, Xiaohong Liu, Xihui Liu, Ruimao Zhang, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2503.16408)  

**Abstract**: Designing effective embodied multi-agent systems is critical for solving complex real-world tasks across domains. Due to the complexity of multi-agent embodied systems, existing methods fail to automatically generate safe and efficient training data for such systems. To this end, we propose the concept of compositional constraints for embodied multi-agent systems, addressing the challenges arising from collaboration among embodied agents. We design various interfaces tailored to different types of constraints, enabling seamless interaction with the physical world. Leveraging compositional constraints and specifically designed interfaces, we develop an automated data collection framework for embodied multi-agent systems and introduce the first benchmark for embodied multi-agent manipulation, RoboFactory. Based on RoboFactory benchmark, we adapt and evaluate the method of imitation learning and analyzed its performance in different difficulty agent tasks. Furthermore, we explore the architectures and training strategies for multi-agent imitation learning, aiming to build safe and efficient embodied multi-agent systems. 

**Abstract (ZH)**: 设计有效的具身多代理系统对于解决跨域复杂实际任务至关重要。由于具身多代理系统的复杂性，现有方法无法自动生成此类系统的安全和高效训练数据。为此，我们提出了具身多代理系统的组合约束概念，以解决具身代理间协作带来的挑战。我们为不同类型的约束设计了各种接口，使与物理世界的交互更加无缝。借助组合约束和专门设计的接口，我们开发了具身多代理系统的自动化数据收集框架，并引入了首个具身多代理操作基准RoboFactory。基于RoboFactory基准，我们适应并评估了模仿学习的方法，并分析了其在不同难度代理任务中的性能。此外，我们探讨了多代理模仿学习的架构和训练策略，旨在构建安全和高效的具身多代理系统。 

---
# Can Real-to-Sim Approaches Capture Dynamic Fabric Behavior for Robotic Fabric Manipulation? 

**Title (ZH)**: 真实场景到模拟的转化方法能否捕捉到织物动态行为以实现机器人织物操控？ 

**Authors**: Yingdong Ru, Lipeng Zhuang, Zhuo He, Florent P. Audonnet, Gerardo Aragon-Caramasa  

**Link**: [PDF](https://arxiv.org/pdf/2503.16310)  

**Abstract**: This paper presents a rigorous evaluation of Real-to-Sim parameter estimation approaches for fabric manipulation in robotics. The study systematically assesses three state-of-the-art approaches, namely two differential pipelines and a data-driven approach. We also devise a novel physics-informed neural network approach for physics parameter estimation. These approaches are interfaced with two simulations across multiple Real-to-Sim scenarios (lifting, wind blowing, and stretching) for five different fabric types and evaluated on three unseen scenarios (folding, fling, and shaking). We found that the simulation engines and the choice of Real-to-Sim approaches significantly impact fabric manipulation performance in our evaluation scenarios. Moreover, PINN observes superior performance in quasi-static tasks but shows limitations in dynamic scenarios. 

**Abstract (ZH)**: 本文对机器人织物 manipulation 中 Real-to-Sim 参数 estimation 方法进行了严格的评估。研究系统评估了三种最先进的方法，即两种微分管道和一个数据驱动的方法。同时，我们还设计了一种新颖的基于物理的神经网络方法用于物理参数 estimation。这些方法在多个 Real-to-Sim 场景（提升、风动和拉伸）下与两种模拟进行接口对接，并在五种不同类型的织物上进行了评估，还测试了三种未见过的场景（折叠、抛掷和摇动）。我们发现，模拟引擎和 Real-to-Sim 方法的选择显著影响我们在评估场景中的织物 manipulation 性能。此外，PINN 在准静态任务中表现出色，但在动态场景中显示出局限性。 

---
# Loop Closure from Two Views: Revisiting PGO for Scalable Trajectory Estimation through Monocular Priors 

**Title (ZH)**: 两视图循环闭合：通过单目先验重访张量估计中的PGO以实现可扩展性 

**Authors**: Tian Yi Lim, Boyang Sun, Marc Pollefeys, Hermann Blum  

**Link**: [PDF](https://arxiv.org/pdf/2503.16275)  

**Abstract**: (Visual) Simultaneous Localization and Mapping (SLAM) remains a fundamental challenge in enabling autonomous systems to navigate and understand large-scale environments. Traditional SLAM approaches struggle to balance efficiency and accuracy, particularly in large-scale settings where extensive computational resources are required for scene reconstruction and Bundle Adjustment (BA). However, this scene reconstruction, in the form of sparse pointclouds of visual landmarks, is often only used within the SLAM system because navigation and planning methods require different map representations. In this work, we therefore investigate a more scalable Visual SLAM (VSLAM) approach without reconstruction, mainly based on approaches for two-view loop closures. By restricting the map to a sparse keyframed pose graph without dense geometry representations, our '2GO' system achieves efficient optimization with competitive absolute trajectory accuracy. In particular, we find that recent advancements in image matching and monocular depth priors enable very accurate trajectory optimization from two-view edges. We conduct extensive experiments on diverse datasets, including large-scale scenarios, and provide a detailed analysis of the trade-offs between runtime, accuracy, and map size. Our results demonstrate that this streamlined approach supports real-time performance, scales well in map size and trajectory duration, and effectively broadens the capabilities of VSLAM for long-duration deployments to large environments. 

**Abstract (ZH)**: 视觉同时定位与建图（SLAM）仍是在使自主系统导航和理解大规模环境方面的一项基本挑战。传统的SLAM方法难以在效率和准确性之间找到平衡，特别是在需要大量计算资源进行场景重建和捆集调整（BA）的大规模环境中。然而，这种场景重建，以稀疏的视觉地标点云的形式，通常仅在SLAM系统内部使用，因为导航和规划方法需要不同的地图表示。因此，在本文中，我们研究了一种不进行重建的更可扩展的视觉SLAM（VSLAM）方法，主要基于两视图回环闭合的方法。通过将地图限制为稀疏的关键帧位姿图，而不包含密集的几何表示，我们的'2GO'系统实现了高效优化，并且具有竞争力的绝对轨迹准确度。特别是在此过程中，我们发现最近在图像匹配和单目深度先验方面的进步使得仅从两视图边就可实现非常准确的轨迹优化。我们在多种数据集上进行了广泛的实验，包括大规模场景，并详细分析了运行时间、准确性和地图大小之间的权衡。实验结果表明，这种精简的方法支持实时性能，在地图大小和轨迹持续时间上可扩展，并有效扩大了VSLAM在长时间部署到大型环境中的能力。 

---
# Explosive Jumping with Rigid and Articulated Soft Quadrupeds via Example Guided Reinforcement Learning 

**Title (ZH)**: 基于示例引导强化学习的刚性与柔体关节四足跳跃爆炸性起跳 

**Authors**: Georgios Apostolides, Wei Pan, Jens Kober, Cosimo Della Santina, Jiatao Ding  

**Link**: [PDF](https://arxiv.org/pdf/2503.16197)  

**Abstract**: Achieving controlled jumping behaviour for a quadruped robot is a challenging task, especially when introducing passive compliance in mechanical design. This study addresses this challenge via imitation-based deep reinforcement learning with a progressive training process. To start, we learn the jumping skill by mimicking a coarse jumping example generated by model-based trajectory optimization. Subsequently, we generalize the learned policy to broader situations, including various distances in both forward and lateral directions, and then pursue robust jumping in unknown ground unevenness. In addition, without tuning the reward much, we learn the jumping policy for a quadruped with parallel elasticity. Results show that using the proposed method, i) the robot learns versatile jumps by learning only from a single demonstration, ii) the robot with parallel compliance reduces the landing error by 11.1%, saves energy cost by 15.2% and reduces the peak torque by 15.8%, compared to the rigid robot without parallel elasticity, iii) the robot can perform jumps of variable distances with robustness against ground unevenness (maximal 4cm height perturbations) using only proprioceptive perception. 

**Abstract (ZH)**: 实现四足机器人的可控跳跃行为是一项具有挑战的任务，尤其是在机械设计中引入被动顺应性时。本研究通过基于模仿的深度强化学习及渐进式训练过程解决了这一挑战。首先，通过模仿基于模型轨迹优化生成的粗略跳跃示例来学习跳跃技能；随后，将学到的策略推广到不同的情况，包括前后和侧向方向的各种距离，进而追求在未知不平地上具有鲁棒性的跳跃。此外，在不调整奖励函数的情况下，学习具有平行弹性的四足机器人的跳跃策略。结果表明，使用所提出的方法，i) 机器人仅从一个示范中学习，即可掌握多种跳跃技巧；ii) 具有平行顺应性的机器人着陆误差减少11.1%，能耗降低15.2%，峰值扭矩减少15.8%，相比于无平行弹性的刚性机器人；iii) 机器人仅依靠本体感受信息即可在最大4cm高度的不平地上表现出鲁棒性的跳跃，适用于不同距离的跳跃。 

---
# Asymptotically Optimal Path Planning With an Approximation of the Omniscient Set 

**Title (ZH)**: 自 Optim 化路径规划与全知集近似 

**Authors**: Jonáš Kříž, Vojtěch Vonásek  

**Link**: [PDF](https://arxiv.org/pdf/2503.16164)  

**Abstract**: The asymptotically optimal version of Rapidly-exploring Random Tree (RRT*) is often used to find optimal paths in a high-dimensional configuration space. The well-known issue of RRT* is its slow convergence towards the optimal solution. A possible solution is to draw random samples only from a subset of the configuration space that is known to contain configurations that can improve the cost of the path (omniscient set). A fast convergence rate may be achieved by approximating the omniscient with a low-volume set. In this letter, we propose new methods to approximate the omniscient set and methods for their effective sampling. First, we propose to approximate the omniscient set using several (small) hyperellipsoids defined by sections of the current best solution. The second approach approximates the omniscient set by a convex hull computed from the current solution. Both approaches ensure asymptotical optimality and work in a general n-dimensional configuration space. The experiments have shown superior performance of our approaches in multiple scenarios in 3D and 6D configuration spaces. 

**Abstract (ZH)**: 急速扩展随机树（RRT*）的渐近最优版本常用于在高维配置空间中寻找最优路径。本文提出了一种新的方法来近似全知集合，并有效地对其进行采样。首先，我们提出使用若干由当前最佳解部分定义的超椭球体来近似全知集合。第二种方法通过从当前解计算凸包来近似全知集合。两种方法在一般n维配置空间中保证渐近最优性。实验表明，在3D和6D配置空间的多个场景中，我们的方法表现出优越的性能。 

---
# The Morphology-Control Trade-Off: Insights into Soft Robotic Efficiency 

**Title (ZH)**: 形态控制权衡：软体机器人效率的洞察 

**Authors**: Yue Xie, Kai-feng Chu, Xing Wang, Fumiya Iida  

**Link**: [PDF](https://arxiv.org/pdf/2503.16127)  

**Abstract**: Soft robotics holds transformative potential for enabling adaptive and adaptable systems in dynamic environments. However, the interplay between morphological and control complexities and their collective impact on task performance remains poorly understood. Therefore, in this study, we investigate these trade-offs across tasks of differing difficulty levels using four well-used morphological complexity metrics and control complexity measured by FLOPs. We investigate how these factors jointly influence task performance by utilizing the evolutionary robot experiments. Results show that optimal performance depends on the alignment between morphology and control: simpler morphologies and lightweight controllers suffice for easier tasks, while harder tasks demand higher complexities in both dimensions. In addition, a clear trade-off between morphological and control complexities that achieve the same task performance can be observed. Moreover, we also propose a sensitivity analysis to expose the task-specific contributions of individual morphological metrics. Our study establishes a framework for investigating the relationships between morphology, control, and task performance, advancing the development of task-specific robotic designs that balance computational efficiency with adaptability. This study contributes to the practical application of soft robotics in real-world scenarios by providing actionable insights. 

**Abstract (ZH)**: 软体机器人为动态环境中的适应性和可适应系统提供了变革性的潜力。然而，形态学复杂性和控制复杂性之间的互动及其对任务性能的共同影响尚未完全理解。因此，在这项研究中，我们使用四种常用的形态学复杂性度量标准和使用FLOPs衡量的控制复杂性，在不同难度级别的任务中探讨这些权衡。通过运用演化机器人实验，我们研究这些因素如何共同影响任务性能。结果表明，最优性能依赖于形态和控制之间的匹配：简单的形态和轻量级的控制器足以应对较简单的任务，而较难的任务则需要两个维度的更高复杂性。此外，实现相同任务性能的形态学和控制复杂性之间存在明显的权衡。此外，我们还提出了一种敏感性分析，以揭示各个形态学度量对特定任务的贡献。本研究建立了一个研究形态学、控制与任务性能之间关系的框架，推动了在计算效率与可适应性之间平衡的特定任务机器人设计的发展。本研究通过提供实用的见解，促进了软体机器人在现实场景中的应用。 

---
# Rejecting Outliers in 2D-3D Point Correspondences from 2D Forward-Looking Sonar Observations 

**Title (ZH)**: 从2D向前声纳观测中 Rejecting 2D-3D 点对应中的离群值 

**Authors**: Jiayi Su, Shaofeng Zou, Jingyu Qian, Yan Wei, Fengzhong Qu, Liuqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16066)  

**Abstract**: Rejecting outliers before applying classical robust methods is a common approach to increase the success rate of estimation, particularly when the outlier ratio is extremely high (e.g. 90%). However, this method often relies on sensor- or task-specific characteristics, which may not be easily transferable across different scenarios. In this paper, we focus on the problem of rejecting 2D-3D point correspondence outliers from 2D forward-looking sonar (2D FLS) observations, which is one of the most popular perception device in the underwater field but has a significantly different imaging mechanism compared to widely used perspective cameras and LiDAR. We fully leverage the narrow field of view in the elevation of 2D FLS and develop two compatibility tests for different 3D point configurations: (1) In general cases, we design a pairwise length in-range test to filter out overly long or short edges formed from point sets; (2) In coplanar cases, we design a coplanarity test to check if any four correspondences are compatible under a coplanar setting. Both tests are integrated into outlier rejection pipelines, where they are followed by maximum clique searching to identify the largest consistent measurement set as inliers. Extensive simulations demonstrate that the proposed methods for general and coplanar cases perform effectively under outlier ratios of 80% and 90%, respectively. 

**Abstract (ZH)**: 在高比例离群值（如90%）情况下，拒绝离群值后再应用经典鲁棒方法以提高估计成功率是一种常见做法，但往往依赖于特定传感器或任务的特性，可能难以跨不同场景移植。本文专注于从2D前方声纳（2D FLS）观测中剔除2D-3D点对应离群值的问题，这是水下领域中最常用的一种感知设备，但其成像机制与广泛使用的透视相机和LiDAR存在显著差异。我们充分利用2D FLS在垂直方向上的窄视野，并为此开发了两种兼容性测试，以处理不同3D点配置：（1）在一般情况下，设计一对一点集形成的边缘长度范围测试，以剔除过长或过短的边缘；（2）在共面情况下，设计共面性测试，检查任何四组对应是否在共面设置下兼容。这两种测试整合进了离群值剔除管道中，之后通过最大_clique_搜索确定最大的一致测量集作为内点。广泛仿真实验表明，在80%和90%的离群值比例下，所提出的方法分别在一般情况和共面情况下表现有效。 

---
# GraspCoT: Integrating Physical Property Reasoning for 6-DoF Grasping under Flexible Language Instructions 

**Title (ZH)**: GraspCoT：在灵活语言指令下的6-自由度抓取物理属性推理集成 

**Authors**: Xiaomeng Chu, Jiajun Deng, Guoliang You, Wei Liu, Xingchen Li, Jianmin Ji, Yanyong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16013)  

**Abstract**: Flexible instruction-guided 6-DoF grasping is a significant yet challenging task for real-world robotic systems. Existing methods utilize the contextual understanding capabilities of the large language models (LLMs) to establish mappings between expressions and targets, allowing robots to comprehend users' intentions in the instructions. However, the LLM's knowledge about objects' physical properties remains underexplored despite its tight relevance to grasping. In this work, we propose GraspCoT, a 6-DoF grasp detection framework that integrates a Chain-of-Thought (CoT) reasoning mechanism oriented to physical properties, guided by auxiliary question-answering (QA) tasks. Particularly, we design a set of QA templates to enable hierarchical reasoning that includes three stages: target parsing, physical property analysis, and grasp action selection. Moreover, GraspCoT presents a unified multimodal LLM architecture, which encodes multi-view observations of 3D scenes into 3D-aware visual tokens, and then jointly embeds these visual tokens with CoT-derived textual tokens within LLMs to generate grasp pose predictions. Furthermore, we present IntentGrasp, a large-scale benchmark that fills the gap in public datasets for multi-object grasp detection under diverse and indirect verbal commands. Extensive experiments on IntentGrasp demonstrate the superiority of our method, with additional validation in real-world robotic applications confirming its practicality. Codes and data will be released. 

**Abstract (ZH)**: 六自由度指令引导抓取是一种显著且具有挑战性的实际机器人系统任务。现有的方法利用大规模语言模型（LLMs）的上下文理解能力建立表达与目标之间的映射，使机器人能够理解用户的意图。然而，LLM对物体物理属性的知识尚未得到充分利用，尽管这与抓取密切相关。在本文中，我们提出了GraspCoT，这是一种结合了面向物理属性的Chain-of-Thought（CoT）推理机制的六自由度抓取检测框架，该机制由辅助问答（QA）任务引导。特别是，我们设计了一组问答模板以实现分层推理，包括三个阶段：目标解析、物理属性分析和抓取动作选择。此外，GraspCoT展示了一个统一的多模态LLM架构，该架构将多视图3D场景观察编码为3D感知的视觉令牌，并在LLM中联合嵌入由CoT导出的文本令牌以生成抓取姿态预测。此外，我们提出了IntentGrasp，这是一个大规模基准，填补了公共数据集中关于多种间接口头指令下的多物体抓取检测的空白。在IntentGrasp上的广泛实验显示了我们方法的优势，并在现实世界的机器人应用中的进一步验证证实了其实用性。代码和数据将被公开。 

---
# Wearable Haptics for a Marionette-inspired Teleoperation of Highly Redundant Robotic Systems 

**Title (ZH)**: 基于木偶启发的高冗余度机器人系统远程操作的可穿戴触觉技术 

**Authors**: Davide Torielli, Leonardo Franco, Maria Pozzi, Luca Muratore, Monica Malvezzi, Nikos Tsagarakis, Domenico Prattichizzo  

**Link**: [PDF](https://arxiv.org/pdf/2503.15998)  

**Abstract**: The teleoperation of complex, kinematically redundant robots with loco-manipulation capabilities represents a challenge for human operators, who have to learn how to operate the many degrees of freedom of the robot to accomplish a desired task. In this context, developing an easy-to-learn and easy-to-use human-robot interface is paramount. Recent works introduced a novel teleoperation concept, which relies on a virtual physical interaction interface between the human operator and the remote robot equivalent to a "Marionette" control, but whose feedback was limited to only visual feedback on the human side. In this paper, we propose extending the "Marionette" interface by adding a wearable haptic interface to cope with the limitations given by the previous works. Leveraging the additional haptic feedback modality, the human operator gains full sensorimotor control over the robot, and the awareness about the robot's response and interactions with the environment is greatly improved. We evaluated the proposed interface and the related teleoperation framework with naive users, assessing the teleoperation performance and the user experience with and without haptic feedback. The conducted experiments consisted in a loco-manipulation mission with the CENTAURO robot, a hybrid leg-wheel quadruped with a humanoid dual-arm upper body. 

**Abstract (ZH)**: 具有远程操作和移动操控能力的复杂自由度机器人远程操作对人类操作者构成挑战，操作者需要学习如何操控机器人的多个自由度以完成目标任务。在这种背景下，开发易于学习和使用的机器人-人类界面至关重要。 recent works引入了一种新颖的远程操作概念，该概念依赖于人类操作者与远程机器人之间的虚拟物理交互界面，类似于“木偶”控制，但之前的工作仅提供了视觉反馈。在这篇论文中，我们提出扩展“木偶”界面，通过添加可穿戴力反馈接口来应对之前工作的局限性。利用额外的力反馈模态，人类操作者能够完全控制机器人，并大大提高了对机器人响应及其与环境互动的意识。我们通过非专业用户评估了所提出的界面及其相关的远程操作框架，评估了有无力反馈条件下的远程操作性能和用户体验。实验内容包括使用CENTAURO机器人进行的移动操控任务，这是一种具有类人双臂上半身的混合腿轮四足机器人。 

---
# A Laser-guided Interaction Interface for Providing Effective Robot Assistance to People with Upper Limbs Impairments 

**Title (ZH)**: 一种用于提供有效上肢功能障碍人士机器人辅助的激光引导交互界面 

**Authors**: Davide Torielli, Liana Bertoni, Luca Muratore, Nikos Tsagarakis  

**Link**: [PDF](https://arxiv.org/pdf/2503.15987)  

**Abstract**: Robotics has shown significant potential in assisting people with disabilities to enhance their independence and involvement in daily activities. Indeed, a societal long-term impact is expected in home-care assistance with the deployment of intelligent robotic interfaces. This work presents a human-robot interface developed to help people with upper limbs impairments, such as those affected by stroke injuries, in activities of everyday life. The proposed interface leverages on a visual servoing guidance component, which utilizes an inexpensive but effective laser emitter device. By projecting the laser on a surface within the workspace of the robot, the user is able to guide the robotic manipulator to desired locations, to reach, grasp and manipulate objects. Considering the targeted users, the laser emitter is worn on the head, enabling to intuitively control the robot motions with head movements that point the laser in the environment, which projection is detected with a neural network based perception module. The interface implements two control modalities: the first allows the user to select specific locations directly, commanding the robot to reach those points; the second employs a paper keyboard with buttons that can be virtually pressed by pointing the laser at them. These buttons enable a more direct control of the Cartesian velocity of the end-effector and provides additional functionalities such as commanding the action of the gripper. The proposed interface is evaluated in a series of manipulation tasks involving a 6DOF assistive robot manipulator equipped with 1DOF beak-like gripper. The two interface modalities are combined to successfully accomplish tasks requiring bimanual capacity that is usually affected in people with upper limbs impairments. 

**Abstract (ZH)**: 机器人技术在辅助肢体障碍人士增强日常生活独立性和参与度方面显示出显著潜力。在智能机器人界面部署后，预计将在家庭护理辅助方面产生社会长期影响。本研究介绍了一种旨在帮助上肢损伤者（如中风损伤患者）在日常活动中使用的交互界面。该交互界面利用了基于视觉伺服引导的组件，该组件采用了一种便宜但有效的激光发射器设备。通过在机器人工作空间内的表面投影激光，用户可以引导机器人 manipulator 至所需位置，完成抓取和操作物体的动作。考虑到目标用户，激光发射器佩戴在头部，使用户能够通过指向环境中的激光进行直观地控制机器人动作，并通过神经网络基于的感知模块检测激光的投影。交互界面实现了两种控制模式：第一种模式允许用户直接选择特定位置，并命令机器人前往这些点；第二种模式使用虚拟按键纸键盘，用户可以通过指向激光来虚拟按下一个按钮。这些按钮使用户能够更直接地控制末端执行器的笛卡尔速度，并提供附加功能，如命令抓取器的动作。本研究评估了该交互界面在操作一个配备单自由度喙式抓取器的六自由度辅助机器人执行一系列操作任务时的表现。两种交互模式的结合成功地完成了通常由于上肢损伤而受影响的双侧操作能力所需的任务。 

---
# Development of a Magnetorheological Hand Exoskeleton Featuring High Force-to-power Ratio for Enhancing Grip Endurance 

**Title (ZH)**: 具有高功率密度的磁流变手部外骨骼开发，以增强握持耐力 

**Authors**: Wenbo Li, Xianlong Mai, Ying Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.15915)  

**Abstract**: Hand exoskeletons have significant potential in labor-intensive fields by mitigating hand grip fatigue, enhancing hand strength, and preventing this http URL, most traditional hand exoskeletons are driven by motors whose output force is limited under constrained installation conditions. In addition, they also come with the disadvantages of high power consumption, complex and bulky assistive systems, and high this http URL this work, we develop a novel hand exoskeleton integrated with magnetorheological (MR) clutches that offers a high force-to-power ratio to improve grip endurance. The clutch features an enhanced structure design, a micro roller enhancing structure, which can significantly boost output forces. The experimental data demonstrate that the clutch can deliver a peak holding force of 380 N with a consumption of 1.48 W, yielding a force-to-power ratio of 256.75N/W, which is 2.35 times higher than the best reported actuator used for hand exoskeletons. The designed MR hand exoskeleton is highly integrated and comprises an exoskeleton frame, MR clutches, a control unit, and a battery. Evaluations through static grip endurance tests and dynamic carrying and lifting tests confirm that the MR hand exoskeleton can effectively reduce muscle fatigue, extend grip endurance, and minimize injuries. These findings highlight its strong potential for practical applications in repetitive tasks such as carrying and lifting in industrial settings. 

**Abstract (ZH)**: 磁流变（MR）合辍离驱动的手部外骨骼在劳动密集型领域的潜在应用 

---
# CONTHER: Human-Like Contextual Robot Learning via Hindsight Experience Replay and Transformers without Expert Demonstrations 

**Title (ZH)**: CONTHER: 基于 hindsight experience replay 和 transformers 的类人类背景机器人学习方法，无需专家示范 

**Authors**: Maria Makarova, Qian Liu, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2503.15895)  

**Abstract**: This paper presents CONTHER, a novel reinforcement learning algorithm designed to efficiently and rapidly train robotic agents for goal-oriented manipulation tasks and obstacle avoidance. The algorithm uses a modified replay buffer inspired by the Hindsight Experience Replay (HER) approach to artificially populate experience with successful trajectories, effectively addressing the problem of sparse reward scenarios and eliminating the need to manually collect expert demonstrations.
The developed algorithm proposes a Transformer-based architecture to incorporate the context of previous states, allowing the agent to perform a deeper analysis and make decisions in a manner more akin to human learning. The effectiveness of the built-in replay buffer, which acts as an "internal demonstrator", is twofold: it accelerates learning and allows the algorithm to adapt to different tasks. Empirical data confirm the superiority of the algorithm by an average of 38.46% over other considered methods, and the most successful baseline by 28.21%, showing higher success rates and faster convergence in the point-reaching task. Since the control is performed through the robot's joints, the algorithm facilitates potential adaptation to a real robot system and construction of an obstacle avoidance task. Therefore, the algorithm has also been tested on tasks requiring following a complex dynamic trajectory and obstacle avoidance. The design of the algorithm ensures its applicability to a wide range of goal-oriented tasks, making it an easily integrated solution for real-world robotics applications. 

**Abstract (ZH)**: 这篇论文介绍了CONTHER，一种新型强化学习算法，旨在高效快速地训练具有目标导向操作任务和障碍避障能力的机器人代理。该算法使用了受Hindsight Experience Replay (HER) 方法启发的修改过的重播缓冲区，人工填充成功轨迹的经验，有效地解决了稀疏奖励场景的问题，并消除了手动收集专家示范的需求。开发的算法提出了基于Transformer的架构，以整合先前状态的上下文，使代理能够进行更深入的分析并以更接近人类学习的方式做出决策。内置重播缓冲区作为“内部示范者”的有效性体现在两个方面：加速学习并使算法能够适应不同的任务。实验证据表明，与所考虑的其他方法相比，该算法平均提高38.46%，相对于最成功的基线提高28.21%，在点接触任务中显示出更高的成功率和更快的收敛速度。由于控制是通过机器人关节实现的，该算法便于潜在适应真实的机器人系统，并构建障碍避障任务。因此，该算法还被测试了需要跟随复杂动态轨迹和障碍避障的任务。算法的设计使其适用于广泛的具有目标导向的任务，使其成为一个易于集成的真实世界机器人应用解决方案。 

---
# APEX-MR: Multi-Robot Asynchronous Planning and Execution for Cooperative Assembly 

**Title (ZH)**: APEX-MR：多机器人异步规划与协同装配执行 

**Authors**: Philip Huang, Ruixuan Liu, Changliu Liu, Jiaoyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.15836)  

**Abstract**: Compared to a single-robot workstation, a multi-robot system offers several advantages: 1) it expands the system's workspace, 2) improves task efficiency, and more importantly, 3) enables robots to achieve significantly more complex and dexterous tasks, such as cooperative assembly. However, coordinating the tasks and motions of multiple robots is challenging due to issues, e.g. system uncertainty, task efficiency, algorithm scalability, and safety concerns. To address these challenges, this paper studies multi-robot coordination and proposes APEX-MR, an asynchronous planning and execution framework designed to safely and efficiently coordinate multiple robots to achieve cooperative assembly, e.g. LEGO assembly. In particular, APEX-MR provides a systematic approach to post-process multi-robot tasks and motion plans to enable robust asynchronous execution under uncertainty. Experimental results demonstrate that APEX-MR can significantly speed up the execution time of many long-horizon LEGO assembly tasks by 48% compared to sequential planning and 36% compared to synchronous planning on average. To further demonstrate the performance, we deploy APEX-MR to a dual-arm system to perform physical LEGO assembly. To our knowledge, this is the first robotic system capable of performing customized LEGO assembly using commercial LEGO bricks. The experiment results demonstrate that the dual-arm system, with APEX-MR, can safely coordinate robot motions, efficiently collaborate, and construct complex LEGO structures. Our project website is available at this https URL 

**Abstract (ZH)**: 多机器人系统相较于单机器人工作站具有多个优势：1) 扩展了系统的作业空间，2) 提高了任务效率，更重要的是，3) 使机器人能够实现更为复杂和灵巧的任务，例如协作装配。然而，协调多台机器人的任务和动作存在诸多挑战，包括系统不确定性、任务效率、算法可扩展性和安全性问题。为应对这些挑战，本文研究了多机器人协调，并提出了一种异步规划与执行框架APEX-MR，该框架旨在安全高效地协调多台机器人以实现协作装配，例如乐高装配。特别是，APEX-MR 为在不确定性条件下执行多次机器人任务和运动计划提供了一种系统性方法。实验结果表明，与顺序规划相比，APEX-MR 可以显著加速许多长期视角下的乐高装配任务的执行时间，平均速度快48%，同步规划平均速度快36%。为了进一步证明其性能，我们将APEX-MR 部署到双臂系统中执行物理乐高装配。据我们所知，这是首个能够使用商用乐高积木进行定制化乐高装配的机器人系统。实验结果表明，配备APEX-MR 的双臂系统能够安全协调机器人动作，高效协作并构建复杂的乐高结构。我们的项目网址可在此访问：[该网址] 

---
# Control Pneumatic Soft Bending Actuator with Online Learning Pneumatic Physical Reservoir Computing 

**Title (ZH)**: 基于在线学习气动物理蓄水池计算的控制 pneumatically 软弯曲执行器 

**Authors**: Junyi Shen, Tetsuro Miyazaki, Kenji Kawashima  

**Link**: [PDF](https://arxiv.org/pdf/2503.15819)  

**Abstract**: The intrinsic nonlinearities of soft robots present significant control but simultaneously provide them with rich computational potential. Reservoir computing (RC) has shown effectiveness in online learning systems for controlling nonlinear systems such as soft actuators. Conventional RC can be extended into physical reservoir computing (PRC) by leveraging the nonlinear dynamics of soft actuators for computation. This paper introduces a PRC-based online learning framework to control the motion of a pneumatic soft bending actuator, utilizing another pneumatic soft actuator as the PRC model. Unlike conventional designs requiring two RC models, the proposed control system employs a more compact architecture with a single RC model. Additionally, the framework enables zero-shot online learning, addressing limitations of previous PRC-based control systems reliant on offline training. Simulations and experiments validated the performance of the proposed system. Experimental results indicate that the PRC model achieved superior control performance compared to a linear model, reducing the root-mean-square error (RMSE) by an average of over 37% in bending motion control tasks. The proposed PRC-based online learning control framework provides a novel approach for harnessing physical systems' inherent nonlinearities to enhance the control of soft actuators. 

**Abstract (ZH)**: 软体机器人内在的非线性特性为其控制带来了挑战，但同时也赋予了丰富的计算潜力。复现计算（RC）在控制诸如软执行器这类非线性系统时显示出有效性。传统的RC可以通过利用软执行器的非线性动态扩展为物理复现计算（PRC）。本文介绍了一种基于PRC的在线学习框架，用于控制气动软弯曲执行器的运动，同时使用另一个气动软执行器作为PRC模型。与传统设计需要两个RC模型相比，提出的控制系统采用更紧凑的架构，仅使用一个RC模型。此外，该框架实现了零样本在线学习，解决了依赖离线训练的先前PRC控制系统的局限性。仿真和实验验证了所提系统的性能。实验结果表明，PRC模型在弯曲运动控制任务中实现了优于线性模型的控制性能，弯曲运动控制任务中的均方根误差（RMSE）平均降低超过37%。所提出的PRC基于的在线学习控制框架提供了一种利用物理系统固有的非线性特性以增强软执行器控制的新方法。 

---
# UAS Visual Navigation in Large and Unseen Environments via a Meta Agent 

**Title (ZH)**: 基于元代理的UAS在大型未见环境中的视觉导航 

**Authors**: Yuci Han, Charles Toth, Alper Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2503.15781)  

**Abstract**: The aim of this work is to develop an approach that enables Unmanned Aerial System (UAS) to efficiently learn to navigate in large-scale urban environments and transfer their acquired expertise to novel environments. To achieve this, we propose a meta-curriculum training scheme. First, meta-training allows the agent to learn a master policy to generalize across tasks. The resulting model is then fine-tuned on the downstream tasks. We organize the training curriculum in a hierarchical manner such that the agent is guided from coarse to fine towards the target task. In addition, we introduce Incremental Self-Adaptive Reinforcement learning (ISAR), an algorithm that combines the ideas of incremental learning and meta-reinforcement learning (MRL). In contrast to traditional reinforcement learning (RL), which focuses on acquiring a policy for a specific task, MRL aims to learn a policy with fast transfer ability to novel tasks. However, the MRL training process is time consuming, whereas our proposed ISAR algorithm achieves faster convergence than the conventional MRL algorithm. We evaluate the proposed methodologies in simulated environments and demonstrate that using this training philosophy in conjunction with the ISAR algorithm significantly improves the convergence speed for navigation in large-scale cities and the adaptation proficiency in novel environments. 

**Abstract (ZH)**: 本研究的目的是开发一种方法，使无人驾驶航空系统（UAS）能够高效地学习在大规模城市环境中导航，并将其获得的专业知识转移到新的环境中。为此，我们提出了一种元课程训练方案。首先，通过元训练，使代理学习一个主策略以在任务间泛化。然后，对该模型进行下游任务的微调。我们将训练课程分层组织，引导代理从粗略到精细地向目标任务过渡。此外，我们引入了增量自适应强化学习（ISAR），这是一种结合增量学习和元强化学习（MRL）思想的算法。与传统的强化学习（RL）专注于特定任务的策略学习不同，MRL旨在学习具有快速迁移能力的策略。然而，MRL的训练过程耗时较长，而我们提出的ISAR算法比传统的MRL算法更快收敛。我们在模拟环境中评估了所提出的方法，并证明这种训练理念与ISAR算法相结合，显著提高了在大规模城市中导航的收敛速度和在新环境中的适应能力。 

---
# Reward Training Wheels: Adaptive Auxiliary Rewards for Robotics Reinforcement Learning 

**Title (ZH)**: 奖励训练辅助轮：机器人强化学习的自适应辅助奖励 

**Authors**: Linji Wang, Tong Xu, Yuanjie Lu, Xuesu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2503.15724)  

**Abstract**: Robotics Reinforcement Learning (RL) often relies on carefully engineered auxiliary rewards to supplement sparse primary learning objectives to compensate for the lack of large-scale, real-world, trial-and-error data. While these auxiliary rewards accelerate learning, they require significant engineering effort, may introduce human biases, and cannot adapt to the robot's evolving capabilities during training. In this paper, we introduce Reward Training Wheels (RTW), a teacher-student framework that automates auxiliary reward adaptation for robotics RL. To be specific, the RTW teacher dynamically adjusts auxiliary reward weights based on the student's evolving capabilities to determine which auxiliary reward aspects require more or less emphasis to improve the primary objective. We demonstrate RTW on two challenging robot tasks: navigation in highly constrained spaces and off-road vehicle mobility on vertically challenging terrain. In simulation, RTW outperforms expert-designed rewards by 2.35% in navigation success rate and improves off-road mobility performance by 122.62%, while achieving 35% and 3X faster training efficiency, respectively. Physical robot experiments further validate RTW's effectiveness, achieving a perfect success rate (5/5 trials vs. 2/5 for expert-designed rewards) and improving vehicle stability with up to 47.4% reduction in orientation angles. 

**Abstract (ZH)**: 机器人强化学习中的奖励训练轮（RTW）：自动化辅助奖励适应框架 

---
# Experience-based Optimal Motion Planning Algorithm for Solving Difficult Planning Problems Using a Limited Dataset 

**Title (ZH)**: 基于经验的最优运动规划算法：利用有限数据集解决复杂规划问题 

**Authors**: Ryota Takamido, Jun Ota  

**Link**: [PDF](https://arxiv.org/pdf/2503.15715)  

**Abstract**: This study aims to address the key challenge of obtaining a high-quality solution path within a short calculation time by generalizing a limited dataset. In the informed experience-driven random trees connect star (IERTC*) process, the algorithm flexibly explores the search trees by morphing the micro paths generated from a single experience while reducing the path cost by introducing a re-wiring process and an informed sampling process. The core idea of this algorithm is to apply different strategies depending on the complexity of the local environment; for example, it adopts a more complex curved trajectory if obstacles are densely arranged near the search tree, and it adopts a simpler straight line if the local environment is sparse. The results of experiments using a general motion benchmark test revealed that IERTC* significantly improved the planning success rate in difficult problems in the cluttered environment (an average improvement of 49.3% compared to the state-of-the-art algorithm) while also significantly reducing the solution cost (a reduction of 56.3%) when using one hundred experiences. Furthermore, the results demonstrated outstanding planning performance even when only one experience was available (a 43.8% improvement in success rate and a 57.8% reduction in solution cost). 

**Abstract (ZH)**: 本研究旨在通过泛化有限的数据集，在较短的计算时间内获得高质量的解路径。在知情经验驱动随机树连接星（IERTC*）过程中，算法通过形态变化生成的微路径并引入重连线过程和知情采样过程灵活探索搜索树，同时降低路径成本。该算法的核心思想是根据局部环境的复杂性采用不同的策略；例如，如果搜索树附近障碍物密集，则采用复杂的曲线轨迹，而如果局部环境稀疏则采用简单的直线。实验使用通用运动基准测试表明，IERTC*在拥挤环境中难问题的规划成功率显著提高（与最先进的算法相比平均提高49.3%），同时使用一百次经验时显著降低了解路径成本（减少56.3%）。此外，结果显示即使只有一次经验可用，其规划性能也表现出色（成功率提高43.8%，解路径成本减少57.8%）。 

---
# Safety Aware Task Planning via Large Language Models in Robotics 

**Title (ZH)**: 基于大型语言模型的机器人安全意识任务规划 

**Authors**: Azal Ahmad Khan, Michael Andrev, Muhammad Ali Murtaza, Sergio Aguilera, Rui Zhang, Jie Ding, Seth Hutchinson, Ali Anwar  

**Link**: [PDF](https://arxiv.org/pdf/2503.15707)  

**Abstract**: The integration of large language models (LLMs) into robotic task planning has unlocked better reasoning capabilities for complex, long-horizon workflows. However, ensuring safety in LLM-driven plans remains a critical challenge, as these models often prioritize task completion over risk mitigation. This paper introduces SAFER (Safety-Aware Framework for Execution in Robotics), a multi-LLM framework designed to embed safety awareness into robotic task planning. SAFER employs a Safety Agent that operates alongside the primary task planner, providing safety feedback. Additionally, we introduce LLM-as-a-Judge, a novel metric leveraging LLMs as evaluators to quantify safety violations within generated task plans. Our framework integrates safety feedback at multiple stages of execution, enabling real-time risk assessment, proactive error correction, and transparent safety evaluation. We also integrate a control framework using Control Barrier Functions (CBFs) to ensure safety guarantees within SAFER's task planning. We evaluated SAFER against state-of-the-art LLM planners on complex long-horizon tasks involving heterogeneous robotic agents, demonstrating its effectiveness in reducing safety violations while maintaining task efficiency. We also verify the task planner and safety planner through actual hardware experiments involving multiple robots and a human. 

**Abstract (ZH)**: 大语言模型在机器人任务规划中的整合增强了复杂长时间流程的推理能力，但确保这些模型驱动计划的安全性仍然是一个关键挑战。本文介绍了SAFER（Safety-Aware Framework for Execution in Robotics），一种多大语言模型框架，旨在将安全意识嵌入到机器人任务规划中。SAFER采用了安全代理与主要任务规划器并行工作，提供安全反馈。此外，我们还引入了LLM-as-a-Judge，这是一种创新的度量标准，利用大语言模型作为评估器来量化生成任务计划中的安全违规行为。我们的框架在执行的多个阶段整合了安全反馈，实现了实时风险评估、主动错误修正和透明的安全评估。我们还使用控制屏障函数（CBFs）控制框架来确保SAFER任务规划中的安全保证。我们在涉及异构机器人代理的复杂长时间任务上对SAFER与最先进的大语言模型规划器进行了评估，证明了其在减少安全违规行为的同时保持任务效率的有效性。我们还通过涉及多个机器人和人类的实际硬件实验验证了任务规划器和安全规划器。 

---
# Capturing a Moving Target by Two Robots in the F2F Model 

**Title (ZH)**: 两位机器人在F2F模型中捕捉移动目标 

**Authors**: Khaled Jawhar, Evangelos Kranakis  

**Link**: [PDF](https://arxiv.org/pdf/2503.15688)  

**Abstract**: We study a search problem on capturing a moving target on an infinite real line. Two autonomous mobile robots (which can move with a maximum speed of 1) are initially placed at the origin, while an oblivious moving target is initially placed at a distance $d$ away from the origin. The robots can move along the line in any direction, but the target is oblivious, cannot change direction, and moves either away from or toward the origin at a constant speed $v$. Our aim is to design efficient algorithms for the two robots to capture the target. The target is captured only when both robots are co-located with it. The robots communicate with each other only face-to-face (F2F), meaning they can exchange information only when co-located, while the target remains oblivious and has no communication capabilities.
We design algorithms under various knowledge scenarios, which take into account the prior knowledge the robots have about the starting distance $d$, the direction of movement (either toward or away from the origin), and the speed $v$ of the target. As a measure of the efficiency of the algorithms, we use the competitive ratio, which is the ratio of the capture time of an algorithm with limited knowledge to the capture time in the full-knowledge model. In our analysis, we are mindful of the cost of changing direction of movement, and show how to accomplish the capture of the target with at most three direction changes (turns). 

**Abstract (ZH)**: 我们在无限实线上研究捕获移动目标的搜索问题。两个自主移动机器人（最大移动速度为1）初始位于原点，而一个不知情的移动目标初始位于距离原点d的位置。机器人可以在直线上任意方向移动，但目标是不知情的，无法改变方向，以恒定速度v朝向或远离原点移动。我们的目标是设计高效的算法让两机器人捕获目标。目标仅在两机器人同时位于其位置时被捕获。机器人仅面对面（F2F）通信，意味着它们只能在同时位于同一位置时交换信息，而目标保持不知情且无法进行通信。我们在不同的知识场景下设计算法，考虑机器人关于起始距离d、移动方向（朝向或远离原点）和目标速度v的先验知识。我们以算法的竞争比为效率度量标准，这是算法在有限知识下的捕获时间与完全知识模型下捕获时间的比率。在分析中，我们考虑了改变移动方向的成本，并展示了如何最多通过三次方向改变（转弯）来完成目标的捕获。 

---
# Robotic Paper Wrapping by Learning Force Control 

**Title (ZH)**: 基于学习力控制的机器人纸张包裹技术 

**Authors**: Hiroki Hanai, Takuya Kiyokawa, Weiwei Wan, Kensuke Harada  

**Link**: [PDF](https://arxiv.org/pdf/2503.15685)  

**Abstract**: Robotic packaging using wrapping paper poses significant challenges due to the material's complex deformation properties. The packaging process itself involves multiple steps, primarily categorized as folding the paper or creating creases. Small deviations in the robot's arm trajectory or force vector can lead to tearing or wrinkling of the paper, exacerbated by the variability in material properties.
This study introduces a novel framework that combines imitation learning and reinforcement learning to enable a robot to perform each step of the packaging process efficiently. The framework allows the robot to follow approximate trajectories of the tool-center point (TCP) based on human demonstrations while optimizing force control parameters to prevent tearing or wrinkling, even with variable wrapping paper materials.
The proposed method was validated through ablation studies, which demonstrated successful task completion with a significant reduction in tear and wrinkle rates. Furthermore, the force control strategy proved to be adaptable across different wrapping paper materials and robust against variations in the size of the target object. 

**Abstract (ZH)**: 使用包装纸进行机器人包装面临显著挑战，由于材料的复杂变形特性。包装过程涉及多个步骤，主要分为折叠纸张或形成褶皱。机器人手臂轨迹或力矢量的小偏差可能会导致纸张撕裂或起皱，而材料性质的变异性会加剧这一问题。

本研究引入了一种结合模仿学习和强化学习的新框架，以使机器人能够高效地完成包装过程的每个步骤。该框架使机器人能够在人类示范的基础上跟随工具中心点（TCP）的大致轨迹，同时优化力控制参数以防止撕裂或起皱，即使使用不同类型的包装纸材料也是如此。 

---
# Neural Lyapunov Function Approximation with Self-Supervised Reinforcement Learning 

**Title (ZH)**: 自监督强化学习辅助的神经李雅普诺夫函数逼近 

**Authors**: Luc McCutcheon, Bahman Gharesifard, Saber Fallah  

**Link**: [PDF](https://arxiv.org/pdf/2503.15629)  

**Abstract**: Control Lyapunov functions are traditionally used to design a controller which ensures convergence to a desired state, yet deriving these functions for nonlinear systems remains a complex challenge. This paper presents a novel, sample-efficient method for neural approximation of nonlinear Lyapunov functions, leveraging self-supervised Reinforcement Learning (RL) to enhance training data generation, particularly for inaccurately represented regions of the state space. The proposed approach employs a data-driven World Model to train Lyapunov functions from off-policy trajectories. The method is validated on both standard and goal-conditioned robotic tasks, demonstrating faster convergence and higher approximation accuracy compared to the state-of-the-art neural Lyapunov approximation baseline. The code is available at: this https URL 

**Abstract (ZH)**: 非线性Lyapunov函数的神经近似：基于自我监督强化学习的样本高效方法 

---
# M3: 3D-Spatial MultiModal Memory 

**Title (ZH)**: M3: 3D-空间多模态记忆 

**Authors**: Xueyan Zou, Yuchen Song, Ri-Zhao Qiu, Xuanbin Peng, Jianglong Ye, Sifei Liu, Xiaolong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16413)  

**Abstract**: We present 3D Spatial MultiModal Memory (M3), a multimodal memory system designed to retain information about medium-sized static scenes through video sources for visual perception. By integrating 3D Gaussian Splatting techniques with foundation models, M3 builds a multimodal memory capable of rendering feature representations across granularities, encompassing a wide range of knowledge. In our exploration, we identify two key challenges in previous works on feature splatting: (1) computational constraints in storing high-dimensional features for each Gaussian primitive, and (2) misalignment or information loss between distilled features and foundation model features. To address these challenges, we propose M3 with key components of principal scene components and Gaussian memory attention, enabling efficient training and inference. To validate M3, we conduct comprehensive quantitative evaluations of feature similarity and downstream tasks, as well as qualitative visualizations to highlight the pixel trace of Gaussian memory attention. Our approach encompasses a diverse range of foundation models, including vision-language models (VLMs), perception models, and large multimodal and language models (LMMs/LLMs). Furthermore, to demonstrate real-world applicability, we deploy M3's feature field in indoor scenes on a quadruped robot. Notably, we claim that M3 is the first work to address the core compression challenges in 3D feature distillation. 

**Abstract (ZH)**: 我们呈现了3D空间多模态记忆（M3），这是一个设计用于通过视频源保留中等规模静态场景信息的多模态记忆系统，适用于视觉感知。通过将3D高斯点绘制技术与基础模型相结合，M3构建了一个多模态记忆，能够渲染不同粒度下的特征表示，涵盖广泛的知识。在我们的研究中，我们识别了特征点绘制以前工作中存在的两个主要挑战：（1）存储每个高维特征的高计算成本，以及（2）提炼特征与基础模型特征之间的对齐或信息丢失。为了解决这些挑战，我们提出了M3，并包含主场景成分和高斯记忆注意力的关键组件，从而实现高效的训练和推理。为了验证M3的有效性，我们进行了全面的特征相似度定量评估和下游任务评估，并通过定性的可视化突出显示高斯记忆注意力的像素轨迹。我们的方法涵盖了多种基础模型，包括视觉-语言模型、感知模型以及大型多模态和语言模型。此外，为了展示其实用性，我们在四足机器人内部场景中部署了M3的特征场。值得注意的是，我们声称M3是首个解决3D特征提炼核心压缩挑战的工作。 

---
# Do Visual Imaginations Improve Vision-and-Language Navigation Agents? 

**Title (ZH)**: 视觉想象能提高视觉语言导航代理的能力吗？ 

**Authors**: Akhil Perincherry, Jacob Krantz, Stefan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.16394)  

**Abstract**: Vision-and-Language Navigation (VLN) agents are tasked with navigating an unseen environment using natural language instructions. In this work, we study if visual representations of sub-goals implied by the instructions can serve as navigational cues and lead to increased navigation performance. To synthesize these visual representations or imaginations, we leverage a text-to-image diffusion model on landmark references contained in segmented instructions. These imaginations are provided to VLN agents as an added modality to act as landmark cues and an auxiliary loss is added to explicitly encourage relating these with their corresponding referring expressions. Our findings reveal an increase in success rate (SR) of around 1 point and up to 0.5 points in success scaled by inverse path length (SPL) across agents. These results suggest that the proposed approach reinforces visual understanding compared to relying on language instructions alone. Code and data for our work can be found at this https URL. 

**Abstract (ZH)**: 基于视觉-语言导航（VLN）代理在使用自然语言指令导航未知环境时，我们研究了指令中隐含的子目标视觉表示能否作为导航线索，从而提高导航性能。为了合成这些视觉表示或想象，我们利用文本到图像扩散模型，该模型针对分段指令中包含的地标参考进行操作。这些想象作为地标线索提供给VLN代理，并添加了一个辅助损失，以明确鼓励将这些想象与其相应的引用表达式相关联。我们的发现表明，代理的成功率（SR）提高了约1个百分点，成功尺度下的成功率（SPL）提高了多达0.5个百分点。这些结果表明，与仅依赖语言指令相比，所提出的方法增强了视觉理解。我们的代码和数据可以在此处找到：this https URL。 

---
# Nonlinear action prediction models reveal multi-timescale locomotor control 

**Title (ZH)**: 非线性动作预测模型揭示多时标运动控制 

**Authors**: Wei-Chen Wang, Antoine De Comite, Monica Daley, Alexandra Voloshina, Nidhi Seethapathi  

**Link**: [PDF](https://arxiv.org/pdf/2503.16340)  

**Abstract**: Modeling movement in real-world tasks is a fundamental scientific goal. However, it is unclear whether existing models and their assumptions, overwhelmingly tested in laboratory-constrained settings, generalize to the real world. For example, data-driven models of foot placement control -- a crucial action for stable locomotion -- assume linear and single timescale mappings. We develop nonlinear foot placement prediction models, finding that neural network architectures with flexible input history-dependence like GRU and Transformer perform best across multiple contexts (walking and running, treadmill and overground, varying terrains) and input modalities (multiple body states, gaze), outperforming traditional models. These models reveal context- and modality-dependent timescales: there is more reliance on fast-timescale predictions in complex terrain, gaze predictions precede body state predictions, and full-body state predictions precede center-of-mass-relevant predictions. Thus, nonlinear action prediction models provide quantifiable insights into real-world motor control and can be extended to other actions, contexts, and populations. 

**Abstract (ZH)**: 真实世界任务中运动建模是ー个基本的科学目标。然而，现有模型及其假设在实验室受限环境中被广泛测试，其在真实世界中的泛化能力尚不明确。例如，足部放置控制的数据驱动模型——对于稳定运动至关重要的一项操作——假定线性和单一时间尺度映射。我们开发了非线性足部放置预测模型，发现具有灵活输入历史依赖性的神经网络结构如GRU和Transformer在多个上下文（步行和跑步，台上跑动和户外跑动，不同地形）和输入模态（多种身体状态，注视）中表现最优，超越了传统模型。这些模型揭示了上下文和模态依赖的时间尺度：在复杂地形中更多依赖快速时间尺度的预测，注视预测先于身体状态预测，全身状态预测先于质心相关预测。因此，非线性动作预测模型为真实世界运动控制提供了可量化的洞见，并可以扩展到其他动作、上下文和人群。 

---
# From Monocular Vision to Autonomous Action: Guiding Tumor Resection via 3D Reconstruction 

**Title (ZH)**: 从单目视觉到自主行动：通过三维重建引导肿瘤切除 

**Authors**: Ayberk Acar, Mariana Smith, Lidia Al-Zogbi, Tanner Watts, Fangjie Li, Hao Li, Nural Yilmaz, Paul Maria Scheikl, Jesse F. d'Almeida, Susheela Sharma, Lauren Branscombe, Tayfun Efe Ertop, Robert J. Webster III, Ipek Oguz, Alan Kuntz, Axel Krieger, Jie Ying Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16263)  

**Abstract**: Surgical automation requires precise guidance and understanding of the scene. Current methods in the literature rely on bulky depth cameras to create maps of the anatomy, however this does not translate well to space-limited clinical applications. Monocular cameras are small and allow minimally invasive surgeries in tight spaces but additional processing is required to generate 3D scene understanding. We propose a 3D mapping pipeline that uses only RGB images to create segmented point clouds of the target anatomy. To ensure the most precise reconstruction, we compare different structure from motion algorithms' performance on mapping the central airway obstructions, and test the pipeline on a downstream task of tumor resection. In several metrics, including post-procedure tissue model evaluation, our pipeline performs comparably to RGB-D cameras and, in some cases, even surpasses their performance. These promising results demonstrate that automation guidance can be achieved in minimally invasive procedures with monocular cameras. This study is a step toward the complete autonomy of surgical robots. 

**Abstract (ZH)**: 手术自动化需要精确的指导和对场景的深刻理解。现有文献中的方法依赖于体积较大的深度摄像头来生成 Anatomy 的地图，但在空间受限的临床应用中并不适用。单目摄像头小巧，并允许在狭小空间内进行微创手术，但需要额外处理以生成三维场景理解。我们提出了一种仅使用 RGB 图像的三维地图生成管道，以创建目标 Anatomy 的分割点云。为了确保最精确的重建，我们在中央气道阻塞映射任务中比较了不同结构从运动算法的性能，并在肿瘤切除的下游任务上测试了该管道。在包括术后组织模型评估的多个指标中，我们的管道与 RGB-D 摄像头的表现相当，甚至在某些情况下超过了其性能。这些有前景的结果表明，使用单目摄像头可以在微创手术中实现自动化指导。本研究是实现手术机器人完全自主性的一步。 

---
# Dispersion is (Almost) Optimal under (A)synchrony 

**Title (ZH)**: 异步环境下分散式算法几乎最优 

**Authors**: Ajay D. Kshemkalyani, Manish Kumar, Anisur Rahaman Molla, Gokarna Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2503.16216)  

**Abstract**: The dispersion problem has received much attention recently in the distributed computing literature. In this problem, $k\leq n$ agents placed initially arbitrarily on the nodes of an $n$-node, $m$-edge anonymous graph of maximum degree $\Delta$ have to reposition autonomously to reach a configuration in which each agent is on a distinct node of the graph. Dispersion is interesting as well as important due to its connections to many fundamental coordination problems by mobile agents on graphs, such as exploration, scattering, load balancing, relocation of self-driven electric cars (robots) to recharge stations (nodes), etc. The objective has been to provide a solution that optimizes simultaneously time and memory complexities. There exist graphs for which the lower bound on time complexity is $\Omega(k)$. Memory complexity is $\Omega(\log k)$ per agent independent of graph topology. The state-of-the-art algorithms have (i) time complexity $O(k\log^2k)$ and memory complexity $O(\log(k+\Delta))$ under the synchronous setting [DISC'24] and (ii) time complexity $O(\min\{m,k\Delta\})$ and memory complexity $O(\log(k+\Delta))$ under the asynchronous setting [OPODIS'21]. In this paper, we improve substantially on this state-of-the-art. Under the synchronous setting as in [DISC'24], we present the first optimal $O(k)$ time algorithm keeping memory complexity $O(\log (k+\Delta))$. Under the asynchronous setting as in [OPODIS'21], we present the first algorithm with time complexity $O(k\log k)$ keeping memory complexity $O(\log (k+\Delta))$, which is time-optimal within an $O(\log k)$ factor despite asynchrony. Both results were obtained through novel techniques to quickly find empty nodes to settle agents, which may be of independent interest. 

**Abstract (ZH)**: 分布式计算文献中最近对分散问题给予了广泛关注。在该问题中，初始时有$k \leq n$个代理随机放置在具有$n$个节点、$m$条边且最大度为$\Delta$的匿名图的节点上，它们需要自主重新定位，以达到每个代理占据图的一个不同节点的配置。分散问题因其与基于图上移动代理的许多基本协调问题（如探索、散播、负载均衡、自主电动汽车（机器人）移至充电站（节点）的再定位等）的联系而有趣且重要。目标是提供一个同时优化时间复杂度和空间复杂度的解决方案。已知存在图形使得时间复杂度的下界为$\Omega(k)$。空间复杂度对每个代理来说是独立于图形拓扑的$\Omega(\log k)$。最新的算法在同步设置下（如DISC'24）时间复杂度为$O(k\log^2 k)$，空间复杂度为$O(\log(k+\Delta))$，在异步设置下（如OPODIS'21）时间复杂度为$O(\min\{m,k\Delta\})$，空间复杂度为$O(\log(k+\Delta))$。本文在同步设置下显著改进了上述最新成果，提出了第一个保持空间复杂度$O(\log (k+\Delta))$的最优$O(k)$时间算法。在异步设置下，我们提出了第一个时间复杂度为$O(k\log k)$且空间复杂度保持在$O(\log (k+\Delta))$的时间最优算法（尽管异步环境下，最优时间复杂度理论上有$O(\log k)$的因子）。这两个结果通过新颖的技术快速找到空闲节点安放代理而获得，这些技术自身可能独立具有重要性。 

---
# AutoDrive-QA- Automated Generation of Multiple-Choice Questions for Autonomous Driving Datasets Using Large Vision-Language Models 

**Title (ZH)**: AutoDrive-QA：使用大规模视觉-语言模型自动生成自动驾驶数据集的多项选择题 

**Authors**: Boshra Khalili, Andrew W.Smyth  

**Link**: [PDF](https://arxiv.org/pdf/2503.15778)  

**Abstract**: In autonomous driving, open-ended question answering often suffers from unreliable evaluations because freeform responses require either complex metrics or subjective human judgment. To address this challenge, we introduce AutoDrive-QA, an automatic pipeline that converts existing driving QA datasets (including DriveLM, NuScenes-QA, and LingoQA) into a structured multiple-choice question (MCQ) format. This benchmark systematically assesses perception, prediction, and planning tasks, providing a standardized and objective evaluation framework. AutoDrive-QA employs an automated pipeline that leverages large language models (LLMs) to generate high-quality, contextually relevant distractors based on domain-specific error patterns commonly found in autonomous driving scenarios. To evaluate both general capabilities and generalization performance, we test the benchmark on three public datasets and conduct zero-shot experiments on an unseen dataset. The zero-shot evaluations reveal that GPT-4V leads with 69.57% accuracy -- achieving 74.94% in Perception, 65.33% in Prediction, and 68.45% in Planning -- demonstrating that while all models excel in Perception, they struggle in Prediction. Consequently, AutoDrive-QA establishes a rigorous, unbiased standard for integrating and evaluating different vision-language models across various autonomous driving datasets, thereby improving generalization in this field. We release all the codes in the AutoDrive-QA GitHub Repository. 

**Abstract (ZH)**: 自动驾駛中的开放式问题回答常常因為自由形式的 답변requires either complex metrics or subjective human judgment而难以获得可靠的评估。为解决这一挑战，我们引入了AutoDrive-QA，这是一种自动管道，将现有的驾驶QA数据集（包括DriveLM、NuScenes-QA和LingoQA）转换为结构化的多项选择题（MCQ）格式，并系统地评估感知、预测和规划任务，提供了一个标准化和客观的评估框架。AutoDrive-QA采用了一种自动化管道，利用大型语言模型（LLMs）根据自动驾驶场景中常见的领域特定错误模式生成高质量、上下文相关的选择项。为了评估模型的一般能力和泛化性能，我们在三个公开数据集上测试了基准测试，并在未见过的数据集上进行了零样本实验。零样本评估表明，GPT-4V以69.57%的准确率领先——感知任务中为74.94%，预测任务中为65.33%，规划任务中为68.45%，表明虽然所有模型在感知任务中表现出色，但在预测任务中却存在问题。因此，AutoDrive-QA为跨各种自动驾驶数据集整合和评估不同视觉-语言模型建立了严格的、无偏的标准，从而促进了该领域的泛化。我们在AutoDrive-QA GitHub Repository中发布了所有代码。 

---
# GASP: Unifying Geometric and Semantic Self-Supervised Pre-training for Autonomous Driving 

**Title (ZH)**: GASP：统一几何和语义自主预训练的自动驾驶技术 

**Authors**: William Ljungbergh, Adam Lilja, Adam Tonderski. Arvid Laveno Ling, Carl Lindström, Willem Verbeke, Junsheng Fu, Christoffer Petersson, Lars Hammarstrand, Michael Felsberg  

**Link**: [PDF](https://arxiv.org/pdf/2503.15672)  

**Abstract**: Self-supervised pre-training based on next-token prediction has enabled large language models to capture the underlying structure of text, and has led to unprecedented performance on a large array of tasks when applied at scale. Similarly, autonomous driving generates vast amounts of spatiotemporal data, alluding to the possibility of harnessing scale to learn the underlying geometric and semantic structure of the environment and its evolution over time. In this direction, we propose a geometric and semantic self-supervised pre-training method, GASP, that learns a unified representation by predicting, at any queried future point in spacetime, (1) general occupancy, capturing the evolving structure of the 3D scene; (2) ego occupancy, modeling the ego vehicle path through the environment; and (3) distilled high-level features from a vision foundation model. By modeling geometric and semantic 4D occupancy fields instead of raw sensor measurements, the model learns a structured, generalizable representation of the environment and its evolution through time. We validate GASP on multiple autonomous driving benchmarks, demonstrating significant improvements in semantic occupancy forecasting, online mapping, and ego trajectory prediction. Our results demonstrate that continuous 4D geometric and semantic occupancy prediction provides a scalable and effective pre-training paradigm for autonomous driving. For code and additional visualizations, see \href{this https URL. 

**Abstract (ZH)**: 基于下一个词预测的自监督预训练使大规模语言模型能够捕获文本的潜在结构，并在应用时取得了前所未有的多任务性能。类似地，自动驾驶产生了大量的时空数据，暗示可以通过规模学习环境及其随时间的变化的几何和语义结构。在此方向上，我们提出了一种几何和语义自监督预训练方法GASP，该方法通过在时空中的任意查询点预测（1）通用占用率，捕捉3D场景的演变结构；（2）本体占用率，建模车辆在环境中的路径；以及（3）来自视觉基础模型的提炼高层特征。通过建模几何和语义4D占用率场而不是原始传感器测量值，该模型学会了一种结构化且可泛化的环境及其随时间变化的表示。我们在多个自动驾驶基准上验证了GASP，展示了在语义占用率预测、在线建图和本体路径预测方面的显着改进。我们的结果表明，连续的4D几何和语义占用率预测为自动驾驶提供了一种可扩展且有效的预训练范式。更多信息和代码，请参见\href{this https URL。 

---
# PEnGUiN: Partially Equivariant Graph NeUral Networks for Sample Efficient MARL 

**Title (ZH)**: PEnG.UiN: 部分等变图神经网络在样本高效多智能体 reinforcement 学习中的应用 

**Authors**: Joshua McClellan, Greyson Brothers, Furong Huang, Pratap Tokekar  

**Link**: [PDF](https://arxiv.org/pdf/2503.15615)  

**Abstract**: Equivariant Graph Neural Networks (EGNNs) have emerged as a promising approach in Multi-Agent Reinforcement Learning (MARL), leveraging symmetry guarantees to greatly improve sample efficiency and generalization. However, real-world environments often exhibit inherent asymmetries arising from factors such as external forces, measurement inaccuracies, or intrinsic system biases. This paper introduces \textit{Partially Equivariant Graph NeUral Networks (PEnGUiN)}, a novel architecture specifically designed to address these challenges. We formally identify and categorize various types of partial equivariance relevant to MARL, including subgroup equivariance, feature-wise equivariance, regional equivariance, and approximate equivariance. We theoretically demonstrate that PEnGUiN is capable of learning both fully equivariant (EGNN) and non-equivariant (GNN) representations within a unified framework. Through extensive experiments on a range of MARL problems incorporating various asymmetries, we empirically validate the efficacy of PEnGUiN. Our results consistently demonstrate that PEnGUiN outperforms both EGNNs and standard GNNs in asymmetric environments, highlighting their potential to improve the robustness and applicability of graph-based MARL algorithms in real-world scenarios. 

**Abstract (ZH)**: 部分等变图神经网络（PEnGUiN）在多智能体强化学习（MARL）中的应用 

---
# Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning 

**Title (ZH)**: Cosmos-Reason1：从物理常识到具身推理 

**Authors**: NVIDIA, Alisson Azzolini, Hannah Brandon, Prithvijit Chattopadhyay, Huayu Chen, Jinju Chu, Yin Cui, Jenna Diamond, Yifan Ding, Francesco Ferroni, Rama Govindaraju, Jinwei Gu, Siddharth Gururani, Imad El Hanafi, Zekun Hao, Jacob Huffman, Jingyi Jin, Brendan Johnson, Rizwan Khan, George Kurian, Elena Lantz, Nayeon Lee, Zhaoshuo Li, Xuan Li, Tsung-Yi Lin, Yen-Chen Lin, Ming-Yu Liu, Andrew Mathau, Yun Ni, Lindsey Pavao, Wei Ping, David W. Romero, Misha Smelyanskiy, Shuran Song, Lyne Tchapmi, Andrew Z. Wang, Boxin Wang, Haoxiang Wang, Fangyin Wei, Jiashu Xu, Yao Xu, Xiaodong Yang, Zhuolin Yang, Xiaohui Zeng, Zhe Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.15558)  

**Abstract**: Physical AI systems need to perceive, understand, and perform complex actions in the physical world. In this paper, we present the Cosmos-Reason1 models that can understand the physical world and generate appropriate embodied decisions (e.g., next step action) in natural language through long chain-of-thought reasoning processes. We begin by defining key capabilities for Physical AI reasoning, with a focus on physical common sense and embodied reasoning. To represent physical common sense, we use a hierarchical ontology that captures fundamental knowledge about space, time, and physics. For embodied reasoning, we rely on a two-dimensional ontology that generalizes across different physical embodiments. Building on these capabilities, we develop two multimodal large language models, Cosmos-Reason1-8B and Cosmos-Reason1-56B. We curate data and train our models in four stages: vision pre-training, general supervised fine-tuning (SFT), Physical AI SFT, and Physical AI reinforcement learning (RL) as the post-training. To evaluate our models, we build comprehensive benchmarks for physical common sense and embodied reasoning according to our ontologies. Evaluation results show that Physical AI SFT and reinforcement learning bring significant improvements. To facilitate the development of Physical AI, we will make our code and pre-trained models available under the NVIDIA Open Model License at this https URL. 

**Abstract (ZH)**: 物理AI系统需要在物理世界中感知、理解和执行复杂的动作。本文介绍了Cosmos-Reason1模型，该模型可以通过长链条推理过程理解物理世界并生成适当的具身决策（例如，下一步行动）的自然语言描述。我们首先定义了物理AI推理的关键能力，重点关注物理常识和具身推理。为了表示物理常识，我们使用了一个层次化的本体，捕捉了关于空间、时间和物理的基本知识。对于具身推理，我们依赖于一个二维的本体，可以跨不同物理具身进行推广。基于这些能力，我们开发了两个多模态大型语言模型：Cosmos-Reason1-8B和Cosmos-Reason1-56B。我们分四个阶段收集数据并训练模型：视觉预训练、通用监督微调（SFT）、物理AI SFT和物理AI强化学习（RL）作为后训练。为了评估我们的模型，我们根据各自的本体构建了全面的基准测试平台，用于物理常识和具身推理的评估。评估结果显示，物理AI SFT和强化学习带来了显著的改进。为了促进物理AI的发展，我们将按照NVIDIA开源模型许可协议，在这个网址提供我们的代码和预训练模型：[这个 https URL]。 

---
# Motion Synthesis with Sparse and Flexible Keyjoint Control 

**Title (ZH)**: 稀疏且灵活的关键关节控制Motion合成 

**Authors**: Inwoo Hwang, Jinseok Bae, Donggeun Lim, Young Min Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.15557)  

**Abstract**: Creating expressive character animations is labor-intensive, requiring intricate manual adjustment of animators across space and time. Previous works on controllable motion generation often rely on a predefined set of dense spatio-temporal specifications (e.g., dense pelvis trajectories with exact per-frame timing), limiting practicality for animators. To process high-level intent and intuitive control in diverse scenarios, we propose a practical controllable motions synthesis framework that respects sparse and flexible keyjoint signals. Our approach employs a decomposed diffusion-based motion synthesis framework that first synthesizes keyjoint movements from sparse input control signals and then synthesizes full-body motion based on the completed keyjoint trajectories. The low-dimensional keyjoint movements can easily adapt to various control signal types, such as end-effector position for diverse goal-driven motion synthesis, or incorporate functional constraints on a subset of keyjoints. Additionally, we introduce a time-agnostic control formulation, eliminating the need for frame-specific timing annotations and enhancing control flexibility. Then, the shared second stage can synthesize a natural whole-body motion that precisely satisfies the task requirement from dense keyjoint movements. We demonstrate the effectiveness of sparse and flexible keyjoint control through comprehensive experiments on diverse datasets and scenarios. 

**Abstract (ZH)**: 基于稀疏灵活关键关节信号的可控运动合成框架 

---
# Joint Decision-Making in Robot Teleoperation: When are Two Heads Better Than One? 

**Title (ZH)**: 协作决策在机器人遥操作中：两人胜过一人吗？ 

**Authors**: Duc-An Nguyen, Raunak Bhattacharyya, Clara Colombatto, Steve Fleming, Ingmar Posner, Nick Hawes  

**Link**: [PDF](https://arxiv.org/pdf/2503.15510)  

**Abstract**: Operators working with robots in safety-critical domains have to make decisions under uncertainty, which remains a challenging problem for a single human operator. An open question is whether two human operators can make better decisions jointly, as compared to a single operator alone. While prior work has shown that two heads are better than one, such studies have been mostly limited to static and passive tasks. We investigate joint decision-making in a dynamic task involving humans teleoperating robots. We conduct a human-subject experiment with $N=100$ participants where each participant performed a navigation task with two mobiles robots in simulation. We find that joint decision-making through confidence sharing improves dyad performance beyond the better-performing individual (p<0.0001). Further, we find that the extent of this benefit is regulated both by the skill level of each individual, as well as how well-calibrated their confidence estimates are. Finally, we present findings on characterising the human-human dyad's confidence calibration based on the individuals constituting the dyad. Our findings demonstrate for the first time that two heads are better than one, even on a spatiotemporal task which includes active operator control of robots. 

**Abstract (ZH)**: 基于任务不确定性的机器人操作者在安全关键领域需要做出决策，这一直是单个人类操作者的挑战性问题。一个开放的问题是两位人类操作者是否可以共同做出优于单个操作者的决策。尽管先前的研究表明两个人比一个人好，但这些研究主要集中在静态和被动任务上。我们研究了人类远程操作机器人在动态任务中的联合决策。我们在包含100名参与者的实验中让每位参与者在模拟环境中使用两个移动机器人完成导航任务。我们发现，通过信心共享进行的联合决策超越了表现较好的个体（p<0.0001）。此外，我们发现这种益处的程度受到每个个体技术水平以及其信心估计准确度的影响。最后，我们基于构成双人的个体介绍了人类-人类双人组信心校准的特征分析。我们的发现首次证明，即使是在包括机器人主动操作控制的时空任务中，两个人也比一个人好。 

---
# GRETA: Modular Platform to Create Adaptive Socially Interactive Agents 

**Title (ZH)**: GRETA：模块化平台，用于创建适应性强的社会互动代理 

**Authors**: Michele Grimaldi, Jieyeon Woo, Fabien Boucaud, Lucie Galland, Nezih Younsi, Liu Yang, Mireille Fares, Sean Graux, Philippe Gauthier, Catherine Pelachaud  

**Link**: [PDF](https://arxiv.org/pdf/2503.15504)  

**Abstract**: The interaction between humans is very complex to describe since it is composed of different elements from different modalities such as speech, gaze, and gestures influenced by social attitudes and emotions. Furthermore, the interaction can be affected by some features which refer to the interlocutor's state. Actual Socially Interactive Agents SIAs aim to adapt themselves to the state of the interaction partner. In this paper, we discuss this adaptation by describing the architecture of the GRETA platform which considers external features while interacting with humans and/or another ECA and process the dialogue incrementally. We illustrate the new architecture of GRETA which deals with the external features, the adaptation, and the incremental approach for the dialogue processing. 

**Abstract (ZH)**: 人类交互非常复杂，因为它由不同模态的元素如语言、注视和手势构成，这些元素受社会态度和情绪的影响。此外，交互还可能受到说话者状态的一些特征影响。实际社会交互代理（SIAs）旨在适应交互伙伴的状态。本文通过描述GRETA平台的架构，讨论这种适应性，该架构在与人类和/或另一个社会型代理互动时考虑外部特征，并采用增量处理对话的方法。我们展示了GRETA的新架构，它处理外部特征、适应性和对话的增量处理方法。 

---
# Impact of Extended Reality on Robot-Assisted Surgery Training 

**Title (ZH)**: 扩展现实对机器人辅助手术培训的影响 

**Authors**: Michael Bickford, Fayez Alruwaili, Sara Ragab, Hanna Rothenberg, Mohammad Abedin-Nasab  

**Link**: [PDF](https://arxiv.org/pdf/2503.15503)  

**Abstract**: Robot Assisted Surgeries (RAS) have one of the steepest learning curves of any type of surgery. Because of this, methods to practice RAS outside the operating room have been developed to improve the surgeons skills. These strategies include the incorporation of extended reality simulators into surgical training programs. In this Systematic review, we seek to determine if extended reality simulators can improve the performance of novice surgeons and how their performance compares to the conventional training of surgeons on Surgical robots. Using the PRISMA 2020 guidelines, a systematic review and meta-analysis was performed searching PubMed, Embase, Web of Science, and Cochrane library for studies that compared the performance of novice surgeons that received no additional training, trained with extended reality, or trained with inanimate physical simulators (conventional additional training). We included articles that gauged performance using either GEARS or Time to complete measurements and used SPSS to perform a meta-analysis to compare the performance outcomes of the surgeons after training. Surgeons trained using extended reality completed their surgical tasks statistically significantly faster than those who did not receive training (Cohen's d=-0.95, p=0.02), and moderately slower than those conventionally trained (Cohen's d=0.65, p=0.14). However, this difference was not statistically significant. Surgeons trained on extended reality demonstrated a statistically significant improvement in GEARS scores over those who did not train (Cohen's d=0.964, p<0.001). While surgeons trained in extended reality had comparable GEARS scores to surgeons trained conventionally (Cohen's d=0.65, p=0.14). This meta-analysis demonstrates that extended reality simulators translated complex skills to surgeons in a low cost and low risk environment. 

**Abstract (ZH)**: 虚拟现实模拟器能否提高外科新手的手术表现及其与传统手术机器人培训的比较：一项系统评价和meta分析 

---
# ImageInThat: Manipulating Images to Convey User Instructions to Robots 

**Title (ZH)**: ImageInThat: 向机器人传达用户指令的图像 manipuation 

**Authors**: Karthik Mahadevan, Blaine Lewis, Jiannan Li, Bilge Mutlu, Anthony Tang, Tovi Grossman  

**Link**: [PDF](https://arxiv.org/pdf/2503.15500)  

**Abstract**: Foundation models are rapidly improving the capability of robots in performing everyday tasks autonomously such as meal preparation, yet robots will still need to be instructed by humans due to model performance, the difficulty of capturing user preferences, and the need for user agency. Robots can be instructed using various methods-natural language conveys immediate instructions but can be abstract or ambiguous, whereas end-user programming supports longer horizon tasks but interfaces face difficulties in capturing user intent. In this work, we propose using direct manipulation of images as an alternative paradigm to instruct robots, and introduce a specific instantiation called ImageInThat which allows users to perform direct manipulation on images in a timeline-style interface to generate robot instructions. Through a user study, we demonstrate the efficacy of ImageInThat to instruct robots in kitchen manipulation tasks, comparing it to a text-based natural language instruction method. The results show that participants were faster with ImageInThat and preferred to use it over the text-based method. Supplementary material including code can be found at: this https URL. 

**Abstract (ZH)**: 基础模型正迅速提升机器人独立完成日常任务的能力，如烹饪，但机器人仍将需要人类的指令，原因包括模型性能限制、捕捉用户偏好难度大以及需要用户自主权。可以通过多种方法对机器人进行指令，自然语言传达即时指令但可能存在抽象或模糊性，而最终用户编程支持长期任务，但在界面方面难以捕捉用户意图。在本工作中，我们提出使用直接对图像进行操作作为一种替代指令范式，并介绍了一种特定实例ImageInThat，允许用户在时间线式界面中直接对图像进行操作以生成机器人指令。通过用户研究，我们展示了ImageInThat在厨房操作任务中指导机器人有效性，将其与基于文本的自然语言指令方法进行对比。结果显示，参与者使用ImageInThat更快速，更偏好使用该方法。补充材料包括代码，可在以下链接找到：this https URL。 

---
# Fast Multi-Party Open-Ended Conversation with a Social Robot 

**Title (ZH)**: 快速多方开放式对话的社会机器人 

**Authors**: Giulio Antonio Abbo, Maria Jose Pinto-Bernal, Martijn Catrycke, Tony Belpaeme  

**Link**: [PDF](https://arxiv.org/pdf/2503.15496)  

**Abstract**: This paper presents the implementation and evaluation of a conversational agent designed for multi-party open-ended interactions. Leveraging state-of-the-art technologies such as voice direction of arrival, voice recognition, face tracking, and large language models, the system aims to facilitate natural and intuitive human-robot conversations. Deployed on the Furhat robot, the system was tested with 30 participants engaging in open-ended group conversations and then in two overlapping discussions. Quantitative metrics, such as latencies and recognition accuracy, along with qualitative measures from user questionnaires, were collected to assess performance. The results highlight the system's effectiveness in managing multi-party interactions, though improvements are needed in response relevance and latency. This study contributes valuable insights for advancing human-robot interaction, particularly in enhancing the naturalness and engagement in group conversations. 

**Abstract (ZH)**: 本文介绍了为多人群体开放式交互设计的对话代理的实现与评估。利用语音到达方向、语音识别、面部跟踪以及大规模语言模型等最先进技术，系统旨在促进自然直观的人机对话。该系统部署在Furhat机器人上，并通过30名参与者进行开放式群体对话和两次重叠讨论测试。通过收集量化指标（如延迟和识别准确性）以及用户问卷的定性评估，对系统性能进行了评估。结果显示，该系统在管理多人群体交互方面具有有效性，但需要改进响应相关性和延迟问题。本研究为促进人机交互提供了宝贵的见解，特别是在增强群体对话的自然性和参与度方面。 

---
# Agreeing to Interact in Human-Robot Interaction using Large Language Models and Vision Language Models 

**Title (ZH)**: 使用大型语言模型和视觉语言模型实现人类与机器人交互的共识机制 

**Authors**: Kazuhiro Sasabuchi, Naoki Wake, Atsushi Kanehira, Jun Takamatsu, Katsushi Ikeuchi  

**Link**: [PDF](https://arxiv.org/pdf/2503.15491)  

**Abstract**: In human-robot interaction (HRI), the beginning of an interaction is often complex. Whether the robot should communicate with the human is dependent on several situational factors (e.g., the current human's activity, urgency of the interaction, etc.). We test whether large language models (LLM) and vision language models (VLM) can provide solutions to this problem. We compare four different system-design patterns using LLMs and VLMs, and test on a test set containing 84 human-robot situations. The test set mixes several publicly available datasets and also includes situations where the appropriate action to take is open-ended. Our results using the GPT-4o and Phi-3 Vision model indicate that LLMs and VLMs are capable of handling interaction beginnings when the desired actions are clear, however, challenge remains in the open-ended situations where the model must balance between the human and robot situation. 

**Abstract (ZH)**: 在人机交互（HRI）中，互动的开始往往较为复杂。机器人是否与人类交流取决于多种情景因素（例如，人类当前的活动，互动的紧迫性等）。我们测试大型语言模型（LLM）和视觉语言模型（VLM）是否能解决这一问题。我们比较了四种不同的系统设计模式，使用LLM和VLM进行了测试，测试集包含84个人机情况。测试集混用了一些公开可用的数据集，并包括一些需要开放性决策的情境。使用GPT-4o和Phi-3 Vision模型的结果表明，当期望的动作明确时，LLM和VLM能够处理互动开始的问题，但在需要在人类和机器人情境之间权衡的开放性情境中仍面临挑战。 

---
