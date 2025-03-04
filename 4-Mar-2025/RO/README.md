# Discrete-Time Hybrid Automata Learning: Legged Locomotion Meets Skateboarding 

**Title (ZH)**: 离散时间混合自动机学习：腿足运动与滑板运动的结合 

**Authors**: Hang Liu, Sangli Teng, Ben Liu, Wei Zhang, Maani Ghaffari  

**Link**: [PDF](https://arxiv.org/pdf/2503.01842)  

**Abstract**: This paper introduces Discrete-time Hybrid Automata Learning (DHAL), a framework using on-policy Reinforcement Learning to identify and execute mode-switching without trajectory segmentation or event function learning. Hybrid dynamical systems, which include continuous flow and discrete mode switching, can model robotics tasks like legged robot locomotion. Model-based methods usually depend on predefined gaits, while model-free approaches lack explicit mode-switching knowledge. Current methods identify discrete modes via segmentation before regressing continuous flow, but learning high-dimensional complex rigid body dynamics without trajectory labels or segmentation is a challenging open problem. Our approach incorporates a beta policy distribution and a multi-critic architecture to model contact-guided motions, exemplified by a challenging quadrupedal robot skateboard task. We validate our method through simulations and real-world tests, demonstrating robust performance in hybrid dynamical systems. 

**Abstract (ZH)**: 基于策略的离散时间混合自动机学习（DHAL）：无需轨迹分割或事件函数学习的模式切换识别与执行 

---
# TacCap: A Wearable FBG-Based Tactile Sensor for Seamless Human-to-Robot Skill Transfer 

**Title (ZH)**: TacCap: 一种基于FBG的可穿戴触觉传感器，实现无缝人到机器人技能转移 

**Authors**: Chengyi Xing, Hao Li, Yi-Lin Wei, Tian-Ao Ren, Tianyu Tu, Yuhao Lin, Elizabeth Schumann, Wei-Shi Zheng, Mark R. Cutkosky  

**Link**: [PDF](https://arxiv.org/pdf/2503.01789)  

**Abstract**: Tactile sensing is essential for dexterous manipulation, yet large-scale human demonstration datasets lack tactile feedback, limiting their effectiveness in skill transfer to robots. To address this, we introduce TacCap, a wearable Fiber Bragg Grating (FBG)-based tactile sensor designed for seamless human-to-robot transfer. TacCap is lightweight, durable, and immune to electromagnetic interference, making it ideal for real-world data collection. We detail its design and fabrication, evaluate its sensitivity, repeatability, and cross-sensor consistency, and assess its effectiveness through grasp stability prediction and ablation studies. Our results demonstrate that TacCap enables transferable tactile data collection, bridging the gap between human demonstrations and robotic execution. To support further research and development, we open-source our hardware design and software. 

**Abstract (ZH)**: 触觉感知对于灵巧操作至关重要，但大规模的人类示范数据集缺乏触觉反馈，限制了其在技能转移给机器人中的有效性。为解决这一问题，我们引入了TacCap，一种基于光纤布拉格光栅（FBG）的可穿戴触觉传感器，旨在实现无缝的人机触觉数据转移。TacCap轻便、耐用且不受电磁干扰影响，使其非常适合用于真实世界的数据采集。我们详细介绍了其设计和制造过程，评估了其灵敏度、重复性和跨传感器一致性，并通过抓取稳定性预测和消融研究评估了其有效性。我们的结果表明，TacCap能够实现可转移的触觉数据采集，弥补了人类示范与机器人执行之间的差距。为支持进一步的研究和开发，我们开放了其硬件设计和软件。 

---
# vS-Graphs: Integrating Visual SLAM and Situational Graphs through Multi-level Scene Understanding 

**Title (ZH)**: vS-图：通过多层场景理解结合视觉SLAM和情境图谱 

**Authors**: Ali Tourani, Saad Ejaz, Hriday Bavle, David Morilla-Cabello, Jose Luis Sanchez-Lopez, Holger Voos  

**Link**: [PDF](https://arxiv.org/pdf/2503.01783)  

**Abstract**: Current Visual Simultaneous Localization and Mapping (VSLAM) systems often struggle to create maps that are both semantically rich and easily interpretable. While incorporating semantic scene knowledge aids in building richer maps with contextual associations among mapped objects, representing them in structured formats like scene graphs has not been widely addressed, encountering complex map comprehension and limited scalability. This paper introduces visual S-Graphs (vS-Graphs), a novel real-time VSLAM framework that integrates vision-based scene understanding with map reconstruction and comprehensible graph-based representation. The framework infers structural elements (i.e., rooms and corridors) from detected building components (i.e., walls and ground surfaces) and incorporates them into optimizable 3D scene graphs. This solution enhances the reconstructed map's semantic richness, comprehensibility, and localization accuracy. Extensive experiments on standard benchmarks and real-world datasets demonstrate that vS-Graphs outperforms state-of-the-art VSLAM methods, reducing trajectory error by an average of 3.38% and up to 9.58% on real-world data. Furthermore, the proposed framework achieves environment-driven semantic entity detection accuracy comparable to precise LiDAR-based frameworks using only visual features. A web page containing more media and evaluation outcomes is available on this https URL. 

**Abstract (ZH)**: 当前的视觉同时定位与建图(VSLAM)系统往往难以创建既富有语义信息又易于解释的地图。虽然结合语义场景知识有助于构建具有上下文关联的更丰富的地图，但将这些信息表示为场景图等结构化格式尚未广泛解决，面临地图理解复杂和扩展性差的问题。本文介绍了一种名为视觉S-图(vS-Graphs)的新颖实时VSLAM框架，该框架将基于视觉的场景理解与地图重建和易于理解的图表示结合在一起。该框架从检测到的建筑组件（如墙壁和地面表面）推断出结构元素（如房间和走廊），并将其纳入可优化的3D场景图中。此解决方案增强了重建地图的语义丰富性、可解释性和定位准确性。在标准基准数据集和真实世界数据集上的大量实验表明，vS-Graphs在平均轨迹误差减少3.38%至9.58%方面优于最先进的VSLAM方法。此外，所提出的框架使用仅视觉特征即可实现与精确LiDAR基框架相当的环境驱动语义实体检测精度。更多信息和评估结果可在以下网址找到。 

---
# No Plan but Everything Under Control: Robustly Solving Sequential Tasks with Dynamically Composed Gradient Descent 

**Title (ZH)**: 有备无患，一切尽在掌控：使用动态组合梯度下降稳健解决序列任务 

**Authors**: Vito Mengers, Oliver Brock  

**Link**: [PDF](https://arxiv.org/pdf/2503.01732)  

**Abstract**: We introduce a novel gradient-based approach for solving sequential tasks by dynamically adjusting the underlying myopic potential field in response to feedback and the world's regularities. This adjustment implicitly considers subgoals encoded in these regularities, enabling the solution of long sequential tasks, as demonstrated by solving the traditional planning domain of Blocks World - without any planning. Unlike conventional planning methods, our feedback-driven approach adapts to uncertain and dynamic environments, as demonstrated by one hundred real-world trials involving drawer manipulation. These experiments highlight the robustness of our method compared to planning and show how interactive perception and error recovery naturally emerge from gradient descent without explicitly implementing them. This offers a computationally efficient alternative to planning for a variety of sequential tasks, while aligning with observations on biological problem-solving strategies. 

**Abstract (ZH)**: 我们提出了一种基于梯度的新颖方法，通过动态调整底层短视势场以响应反馈和世界的规律来解决序列任务。这种方法隐含地考虑了这些规律中编码的子目标，从而能够解决长期序列任务，如通过免规划解决传统的规划领域Block World。与传统规划方法不同，我们的基于反馈的方法能够适应不确定和动态环境，如在涉及抽屉操作的一百次真实世界实验中所证明的那样。这些实验突显了我们方法在与规划相比的鲁棒性，并展示了如何从梯度下降自然地涌现出交互式感知和错误恢复，而无需显式实现它们。这种方法为各种序列任务提供了一种计算效率更高的替代规划方案，同时与生物问题解决策略的观察相一致。 

---
# FLAME: A Federated Learning Benchmark for Robotic Manipulation 

**Title (ZH)**: FLAME：用于机器人 manipulation 的联邦学习基准 

**Authors**: Santiago Bou Betran, Alberta Longhini, Miguel Vasco, Yuchong Zhang, Danica Kragic  

**Link**: [PDF](https://arxiv.org/pdf/2503.01729)  

**Abstract**: Recent progress in robotic manipulation has been fueled by large-scale datasets collected across diverse environments. Training robotic manipulation policies on these datasets is traditionally performed in a centralized manner, raising concerns regarding scalability, adaptability, and data privacy. While federated learning enables decentralized, privacy-preserving training, its application to robotic manipulation remains largely unexplored. We introduce FLAME (Federated Learning Across Manipulation Environments), the first benchmark designed for federated learning in robotic manipulation. FLAME consists of: (i) a set of large-scale datasets of over 160,000 expert demonstrations of multiple manipulation tasks, collected across a wide range of simulated environments; (ii) a training and evaluation framework for robotic policy learning in a federated setting. We evaluate standard federated learning algorithms in FLAME, showing their potential for distributed policy learning and highlighting key challenges. Our benchmark establishes a foundation for scalable, adaptive, and privacy-aware robotic learning. 

**Abstract (ZH)**: 近期，机器人操作方面的进展得益于跨多种环境收集的大规模数据集。在这些数据集上训练机器人操作策略通常采用集中式方式，这引起了可扩展性、适应性和数据隐私方面的担忧。尽管联邦学习可以实现分布式、隐私保护的训练，但其在机器人操作中的应用尚未得到充分探索。我们介绍了FLAME（跨操作环境的联邦学习），这是首个为机器人操作中的联邦学习设计的基准测试。FLAME包含：(i) 一个包含超过160,000个专家演示的数据集，涵盖了多种操作任务和广泛模拟环境；(ii) 一种用于联邦设置中机器人策略学习的训练和评估框架。我们在FLAME中评估了标准联邦学习算法，展示了其在分布式策略学习中的潜力，并指出了关键挑战。该基准测试为可扩展、适应性和隐私意识的机器人学习奠定了基础。 

---
# Code-as-Symbolic-Planner: Foundation Model-Based Robot Planning via Symbolic Code Generation 

**Title (ZH)**: 代码作为符号规划者：基于符号代码生成的foundation模型机器人规划 

**Authors**: Yongchao Chen, Yilun Hao, Yang Zhang, Chuchu Fan  

**Link**: [PDF](https://arxiv.org/pdf/2503.01700)  

**Abstract**: Recent works have shown great potentials of Large Language Models (LLMs) in robot task and motion planning (TAMP). Current LLM approaches generate text- or code-based reasoning chains with sub-goals and action plans. However, they do not fully leverage LLMs' symbolic computing and code generation capabilities. Many robot TAMP tasks involve complex optimization under multiple constraints, where pure textual reasoning is insufficient. While augmenting LLMs with predefined solvers and planners improves performance, it lacks generalization across tasks. Given LLMs' growing coding proficiency, we enhance their TAMP capabilities by steering them to generate code as symbolic planners for optimization and constraint verification. Unlike prior work that uses code to interface with robot action modules, we steer LLMs to generate code as solvers, planners, and checkers for TAMP tasks requiring symbolic computing, while still leveraging textual reasoning to incorporate common sense. With a multi-round guidance and answer evolution framework, the proposed Code-as-Symbolic-Planner improves success rates by average 24.1\% over best baseline methods across seven typical TAMP tasks and three popular LLMs. Code-as-Symbolic-Planner shows strong effectiveness and generalizability across discrete and continuous environments, 2D/3D simulations and real-world settings, as well as single- and multi-robot tasks with diverse requirements. See our project website this https URL for prompts, videos, and code. 

**Abstract (ZH)**: Recent Works Show Great Potentials of Large Language Models in Robot Task and Motion Planning: Enhancing Capabilities through Code-as-Symbolic-Planner 

---
# Perceptual Motor Learning with Active Inference Framework for Robust Lateral Control 

**Title (ZH)**: 基于主动推断框架的知觉运动学习在稳健横向控制中的应用 

**Authors**: Elahe Delavari, John Moore, Junho Hong, Jaerock Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2503.01676)  

**Abstract**: This paper presents a novel Perceptual Motor Learning (PML) framework integrated with Active Inference (AIF) to enhance lateral control in Highly Automated Vehicles (HAVs). PML, inspired by human motor learning, emphasizes the seamless integration of perception and action, enabling efficient decision-making in dynamic environments. Traditional autonomous driving approaches--including modular pipelines, imitation learning, and reinforcement learning--struggle with adaptability, generalization, and computational efficiency. In contrast, PML with AIF leverages a generative model to minimize prediction error ("surprise") and actively shape vehicle control based on learned perceptual-motor representations. Our approach unifies deep learning with active inference principles, allowing HAVs to perform lane-keeping maneuvers with minimal data and without extensive retraining across different environments. Extensive experiments in the CARLA simulator demonstrate that PML with AIF enhances adaptability without increasing computational overhead while achieving performance comparable to conventional methods. These findings highlight the potential of PML-driven active inference as a robust alternative for real-world autonomous driving applications. 

**Abstract (ZH)**: 基于主动推断的感知运动学习框架在高度自动化车辆中增强侧向控制 

---
# RoboDexVLM: Visual Language Model-Enabled Task Planning and Motion Control for Dexterous Robot Manipulation 

**Title (ZH)**: RoboDexVLM：视觉语言模型驱动的灵巧机器人操作的任务规划与运动控制 

**Authors**: Haichao Liu, Sikai Guo, Pengfei Mai, Jiahang Cao, Haoang Li, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.01616)  

**Abstract**: This paper introduces RoboDexVLM, an innovative framework for robot task planning and grasp detection tailored for a collaborative manipulator equipped with a dexterous hand. Previous methods focus on simplified and limited manipulation tasks, which often neglect the complexities associated with grasping a diverse array of objects in a long-horizon manner. In contrast, our proposed framework utilizes a dexterous hand capable of grasping objects of varying shapes and sizes while executing tasks based on natural language commands. The proposed approach has the following core components: First, a robust task planner with a task-level recovery mechanism that leverages vision-language models (VLMs) is designed, which enables the system to interpret and execute open-vocabulary commands for long sequence tasks. Second, a language-guided dexterous grasp perception algorithm is presented based on robot kinematics and formal methods, tailored for zero-shot dexterous manipulation with diverse objects and commands. Comprehensive experimental results validate the effectiveness, adaptability, and robustness of RoboDexVLM in handling long-horizon scenarios and performing dexterous grasping. These results highlight the framework's ability to operate in complex environments, showcasing its potential for open-vocabulary dexterous manipulation. Our open-source project page can be found at this https URL. 

**Abstract (ZH)**: RoboDexVLM：一种面向协作 manipulator 的灵巧手任务规划与抓取检测创新框架 

---
# Soft Everting Prosthetic Hand and Comparison with Existing Body-Powered Terminal Devices 

**Title (ZH)**: 软 everting 截肢假手及其与现有体控终端装置的比较 

**Authors**: Gayoung Park, Katalin Schäffer, Margaret M. Coad  

**Link**: [PDF](https://arxiv.org/pdf/2503.01585)  

**Abstract**: In this paper, we explore the use of a soft gripper, specifically a soft inverting-everting toroidal hydrostat, as a prosthetic hand. We present a design of the gripper integrated into a body-powered elbow-driven system and evaluate its performance compared to similar body-powered terminal devices: the Kwawu 3D-printed hand and the Hosmer hook. Our experiments highlight advantages of the Everting hand, such as low required cable tension for operation (1.6 N for Everting, 30.0 N for Kwawu, 28.1 N for Hosmer), limited restriction on the elbow angle range, and secure grasping capability (peak pulling force required to remove an object: 15.8 N for Everting, 6.9 N for Kwawu, 4.0 N for Hosmer). In our pilot user study, six able-bodied participants performed standardized hand dexterity tests. With the Everting hand compared to the Kwawu hand, users transferred more blocks in one minute and completed three tasks (moving small common objects, simulated feeding with a spoon, and moving large empty cans) faster (p~$\leq$~0.05). With the Everting hand compared to the Hosmer hook, users moved large empty cans faster (p~$\leq$~0.05) and achieved similar performance on all other tasks. Overall, user preference leaned toward the Everting hand for its adaptable grip and ease of use, although its abilities could be improved in tasks requiring high precision such as writing with a pen, and in handling heavier objects such as large heavy cans. 

**Abstract (ZH)**: 软倒置扩张环形液压软体夹手在假手中的应用研究：与Kwawu 3D打印手和Hosmer钩的性能比较 

---
# MLINE-VINS: Robust Monocular Visual-Inertial SLAM With Flow Manhattan and Line Features 

**Title (ZH)**: MLINE-VINS：具有流动曼哈顿和平面线特征的鲁棒单目视觉-惯性SLAM 

**Authors**: Chao Ye, Haoyuan Li, Weiyang Lin, Xianqiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01571)  

**Abstract**: In this paper we introduce MLINE-VINS, a novel monocular visual-inertial odometry (VIO) system that leverages line features and Manhattan Word assumption. Specifically, for line matching process, we propose a novel geometric line optical flow algorithm that efficiently tracks line features with varying lengths, whitch is do not require detections and descriptors in every frame. To address the instability of Manhattan estimation from line features, we propose a tracking-by-detection module that consistently tracks and optimizes Manhattan framse in consecutive images. By aligning the Manhattan World with the VIO world frame, the tracking could restart using the latest pose from back-end, simplifying the coordinate transformations within the system. Furthermore, we implement a mechanism to validate Manhattan frames and a novel global structural constraints back-end optimization. Extensive experiments results on vairous datasets, including benchmark and self-collected datasets, show that the proposed approach outperforms existing methods in terms of accuracy and long-range robustness. The source code of our method is available at: this https URL. 

**Abstract (ZH)**: 本文介绍了一种新的单目视觉惯性里程计（VIO）系统MLINE-VINS，该系统利用直线特征和曼哈顿词假设。对于直线匹配过程，我们提出了一种新颖的几何直线光学流算法，该算法可以高效地跟踪各种长度的直线特征，且不需要每帧都进行检测和描述符匹配。针对从直线特征估计曼哈顿结构的不稳定性，我们提出了一种跟踪-检测模块，该模块可以一致地跟踪和优化连续图像中的曼哈顿帧。通过将曼哈顿世界与VIO世界坐标帧对齐，跟踪可以在后端最新姿态的指导下重启，简化系统的坐标转换。此外，我们实现了一种验证曼哈顿帧的机制和一种新的全局结构约束后端优化方法。在各种数据集（包括基准数据集和自采集数据集）上进行的广泛实验结果显示，所提出的方法在准确性和长距离鲁棒性方面优于现有方法。我们的方法源代码可在以下链接获取：这个 https URL。 

---
# VF-Plan: Bridging the Art Gallery Problem and Static LiDAR Scanning with Visibility Field Optimization 

**Title (ZH)**: VF-Plan: 将视野场优化与艺术画廊问题和静态LiDAR扫描相结合 

**Authors**: Biao Xionga, Longjun Zhanga, Ruiqi Huanga, Junwei Zhoua, Bojian Wub, Fashuai Lic  

**Link**: [PDF](https://arxiv.org/pdf/2503.01562)  

**Abstract**: Viewpoint planning is crucial for 3D data collection and autonomous navigation, yet existing methods often miss key optimization objectives for static LiDAR, resulting in suboptimal network designs. The Viewpoint Planning Problem (VPP), which builds upon the Art Gallery Problem (AGP), requires not only full coverage but also robust registrability and connectivity under limited sensor views. We introduce a greedy optimization algorithm that tackles these VPP and AGP challenges through a novel Visibility Field (VF) approach. The VF captures visibility characteristics unique to static LiDAR, enabling a reduction from 2D to 1D by focusing on medial axis and joints. This leads to a minimal, fully connected viewpoint network with comprehensive coverage and minimal redundancy. Experiments across diverse environments show that our method achieves high efficiency and scalability, matching or surpassing expert designs. Compared to state-of-the-art methods, our approach achieves comparable viewpoint counts (VC) while reducing Weighted Average Path Length (WAPL) by approximately 95\%, indicating a much more compact and connected network. Dataset and source code will be released upon acceptance. 

**Abstract (ZH)**: 视点规划对于3D数据采集和自主导航至关重要，现有方法往往忽视了静态LiDAR的关键优化目标，导致网络设计欠佳。视点规划问题（VPP），基于艺术画廊问题（AGP），除了要求全面覆盖外，还需要在有限传感器视角下具备鲁棒的可注册性和连通性。我们提出了一种贪婪优化算法，通过一种新颖的可见性场（VF）方法来解决这些VPP和AGP挑战。VF捕捉静态LiDAR特有的可见性特征，使得问题从二维减少到一维，通过集中于中轴线和节点。这导致了一个最小的、完全连通的视点网络，具备全面覆盖且冗余最少。实验结果表明，我们的方法在多种环境下表现出高效性和可扩展性，能够达到或超越专家设计。与现有的最先进的方法相比，我们的方法在视点计数（VC）相当的同时，平均路径长度加权平均值（WAPL）减少了约95%，表明一个更为紧凑和连通的网络。数据集和源代码将在接受后发布。 

---
# MapExRL: Human-Inspired Indoor Exploration with Predicted Environment Context and Reinforcement Learning 

**Title (ZH)**: MapExRL: 基于预测环境上下文和强化学习的人类启发式室内探索 

**Authors**: Narek Harutyunyan, Brady Moon, Seungchan Kim, Cherie Ho, Adam Hung, Sebastian Scherer  

**Link**: [PDF](https://arxiv.org/pdf/2503.01548)  

**Abstract**: Path planning for robotic exploration is challenging, requiring reasoning over unknown spaces and anticipating future observations. Efficient exploration requires selecting budget-constrained paths that maximize information gain. Despite advances in autonomous exploration, existing algorithms still fall short of human performance, particularly in structured environments where predictive cues exist but are underutilized. Guided by insights from our user study, we introduce MapExRL, which improves robot exploration efficiency in structured indoor environments by enabling longer-horizon planning through reinforcement learning (RL) and global map predictions. Unlike many RL-based exploration methods that use motion primitives as the action space, our approach leverages frontiers for more efficient model learning and longer horizon reasoning. Our framework generates global map predictions from the observed map, which our policy utilizes, along with the prediction uncertainty, estimated sensor coverage, frontier distance, and remaining distance budget, to assess the strategic long-term value of frontiers. By leveraging multiple frontier scoring methods and additional context, our policy makes more informed decisions at each stage of the exploration. We evaluate our framework on a real-world indoor map dataset, achieving up to an 18.8% improvement over the strongest state-of-the-art baseline, with even greater gains compared to conventional frontier-based algorithms. 

**Abstract (ZH)**: 基于强化学习的结构化室内环境中的地图预测辅助路径规划 

---
# Exo-ViHa: A Cross-Platform Exoskeleton System with Visual and Haptic Feedback for Efficient Dexterous Skill Learning 

**Title (ZH)**: Exo-ViHa: 一种具备视觉与触觉反馈的跨平台外骨骼系统，用于高效的学习灵巧技能 

**Authors**: Xintao Chao, Shilong Mu, Yushan Liu, Shoujie Li, Chuqiao Lyu, Xiao-Ping Zhang, Wenbo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2503.01543)  

**Abstract**: Imitation learning has emerged as a powerful paradigm for robot skills learning. However, traditional data collection systems for dexterous manipulation face challenges, including a lack of balance between acquisition efficiency, consistency, and accuracy. To address these issues, we introduce Exo-ViHa, an innovative 3D-printed exoskeleton system that enables users to collect data from a first-person perspective while providing real-time haptic feedback. This system combines a 3D-printed modular structure with a slam camera, a motion capture glove, and a wrist-mounted camera. Various dexterous hands can be installed at the end, enabling it to simultaneously collect the posture of the end effector, hand movements, and visual data. By leveraging the first-person perspective and direct interaction, the exoskeleton enhances the task realism and haptic feedback, improving the consistency between demonstrations and actual robot deployments. In addition, it has cross-platform compatibility with various robotic arms and dexterous hands. Experiments show that the system can significantly improve the success rate and efficiency of data collection for dexterous manipulation tasks. 

**Abstract (ZH)**: 基于仿生学习的Exo-ViHa外骨骼系统：一种用于灵巧操作的数据收集方法 

---
# Origami-Inspired Soft Gripper with Tunable Constant Force Output 

**Title (ZH)**: 受 Origami 启发的可调恒定力输出软夹持器 

**Authors**: Zhenwei Ni, Chang Xu, Zhihang Qin, Ceng Zhang, Zhiqiang Tang, Peiyi Wang, Cecilia Laschi  

**Link**: [PDF](https://arxiv.org/pdf/2503.01481)  

**Abstract**: Soft robotic grippers gently and safely manipulate delicate objects due to their inherent adaptability and softness. Limited by insufficient stiffness and imprecise force control, conventional soft grippers are not suitable for applications that require stable grasping force. In this work, we propose a soft gripper that utilizes an origami-inspired structure to achieve tunable constant force output over a wide strain range. The geometry of each taper panel is established to provide necessary parameters such as protrusion distance, taper angle, and crease thickness required for 3D modeling and FEA analysis. Simulations and experiments show that by optimizing these parameters, our design can achieve a tunable constant force output. Moreover, the origami-inspired soft gripper dynamically adapts to different shapes while preventing excessive forces, with potential applications in logistics, manufacturing, and other industrial settings that require stable and adaptive operations 

**Abstract (ZH)**: 基于 Origami 灵感结构的可调恒定力柔顺夹持器柔顺地且安全地Manipulate Delicate Objects Due to Their Inherent Adaptability and Softness 

---
# Interactive Navigation for Legged Manipulators with Learned Arm-Pushing Controller 

**Title (ZH)**: 基于学习的臂推控制器的腿足 manipulator 交互导航 

**Authors**: Zhihai Bi, Kai Chen, Chunxin Zheng, Yulin Li, Haoang Li, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.01474)  

**Abstract**: Interactive navigation is crucial in scenarios where proactively interacting with objects can yield shorter paths, thus significantly improving traversal efficiency. Existing methods primarily focus on using the robot body to relocate large obstacles (which could be comparable to the size of a robot). However, they prove ineffective in narrow or constrained spaces where the robot's dimensions restrict its manipulation capabilities. This paper introduces a novel interactive navigation framework for legged manipulators, featuring an active arm-pushing mechanism that enables the robot to reposition movable obstacles in space-constrained environments. To this end, we develop a reinforcement learning-based arm-pushing controller with a two-stage reward strategy for large-object manipulation. Specifically, this strategy first directs the manipulator to a designated pushing zone to achieve a kinematically feasible contact configuration. Then, the end effector is guided to maintain its position at appropriate contact points for stable object displacement while preventing toppling. The simulations validate the robustness of the arm-pushing controller, showing that the two-stage reward strategy improves policy convergence and long-term performance. Real-world experiments further demonstrate the effectiveness of the proposed navigation framework, which achieves shorter paths and reduced traversal time. The open-source project can be found at this https URL. 

**Abstract (ZH)**: 基于腿部 manipulator 的主动臂推送交互导航框架 

---
# Aerial Gym Simulator: A Framework for Highly Parallelized Simulation of Aerial Robots 

**Title (ZH)**: 基于空中的健身房模拟器：一种高度并行化的空中机器人仿真框架 

**Authors**: Mihir Kulkarni, Welf Rehberg, Kostas Alexis  

**Link**: [PDF](https://arxiv.org/pdf/2503.01471)  

**Abstract**: This paper contributes the Aerial Gym Simulator, a highly parallelized, modular framework for simulation and rendering of arbitrary multirotor platforms based on NVIDIA Isaac Gym. Aerial Gym supports the simulation of under-, fully- and over-actuated multirotors offering parallelized geometric controllers, alongside a custom GPU-accelerated rendering framework for ray-casting capable of capturing depth, segmentation and vertex-level annotations from the environment. Multiple examples for key tasks, such as depth-based navigation through reinforcement learning are provided. The comprehensive set of tools developed within the framework makes it a powerful resource for research on learning for control, planning, and navigation using state information as well as exteroceptive sensor observations. Extensive simulation studies are conducted and successful sim2real transfer of trained policies is demonstrated. The Aerial Gym Simulator is open-sourced at: this https URL. 

**Abstract (ZH)**: 本文贡献了Aerial Gym模拟器，这是一个基于NVIDIA Isaac Gym的高度并行化、模块化的多旋翼平台模拟与渲染框架。Aerial Gym支持欠驱动、全驱动和过驱动多旋翼的模拟，配备了并行化的几何控制器，以及一个用于光线投射的自定义GPU加速渲染框架，能够捕捉环境的深度、分割和顶点级标注。提供了关键任务（如基于深度的学习导航）的多个示例。框架内开发的一系列工具使其成为利用状态信息及外部传感器观察进行控制学习、规划和导航研究的强大资源。进行了广泛的模拟研究，并展示了训练策略从模拟到现实世界的成功转移。Aerial Gym模拟器已开源：this https URL。 

---
# AVR: Active Vision-Driven Robotic Precision Manipulation with Viewpoint and Focal Length Optimization 

**Title (ZH)**: AVR：基于视角和焦距优化的主动视觉驱动精密操作机器人 

**Authors**: Yushan Liu, Shilong Mu, Xintao Chao, Zizhen Li, Yao Mu, Tianxing Chen, Shoujie Li, Chuqiao Lyu, Xiao-ping Zhang, Wenbo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2503.01439)  

**Abstract**: Robotic manipulation within dynamic environments presents challenges to precise control and adaptability. Traditional fixed-view camera systems face challenges adapting to change viewpoints and scale variations, limiting perception and manipulation precision. To tackle these issues, we propose the Active Vision-driven Robotic (AVR) framework, a teleoperation hardware solution that supports dynamic viewpoint and dynamic focal length adjustments to continuously center targets and maintain optimal scale, accompanied by a corresponding algorithm that effectively enhances the success rates of various operational tasks. Using the RoboTwin platform with a real-time image processing plugin, AVR framework improves task success rates by 5%-16% on five manipulation tasks. Physical deployment on a dual-arm system demonstrates in collaborative tasks and 36% precision in screwdriver insertion, outperforming baselines by over 25%. Experimental results confirm that AVR framework enhances environmental perception, manipulation repeatability (40% $\le $1 cm error), and robustness in complex scenarios, paving the way for future robotic precision manipulation methods in the pursuit of human-level robot dexterity and precision. 

**Abstract (ZH)**: 动态环境下的机器人 manipulation 面临精确控制和适应性挑战。传统的固定视角摄像机系统在适应视角变化和尺度变化方面存在局限，限制了感知和操作精度。为应对这些问题，我们提出了主动视觉驱动机器人（AVR）框架，这是一种支持动态视角和动态焦距调整的远程操作硬件解决方案，能够连续对准目标并保持最佳尺度，伴随相应的算法有效提升了各种操作任务的成功率。利用带实时图像处理插件的 RoboTwin 平台，AVR 框架在五个操作任务中将任务成功率提升了 5%-16%。物理部署在双臂系统中展示了在协作任务中的优势，并在螺丝刀插入操作中达到了 36% 的精度，优于基线方法 25% 以上。实验结果表明，AVR 框架增强了环境感知、操作重复性（小于 1 cm 的误差率为 40%）以及在复杂场景中的鲁棒性，为您未来的机器人精确操作方法提供了实现人类级别机器人灵巧性和精度的道路。 

---
# CAO-RONet: A Robust 4D Radar Odometry with Exploring More Information from Low-Quality Points 

**Title (ZH)**: CAO-RONet：一种从低质量点中探索更多信息的 robust 4D 雷达里程计 

**Authors**: Zhiheng Li, Yubo Cui, Ningyuan Huang, Chenglin Pang, Zheng Fang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01438)  

**Abstract**: Recently, 4D millimetre-wave radar exhibits more stable perception ability than LiDAR and camera under adverse conditions (e.g. rain and fog). However, low-quality radar points hinder its application, especially the odometry task that requires a dense and accurate matching. To fully explore the potential of 4D radar, we introduce a learning-based odometry framework, enabling robust ego-motion estimation from finite and uncertain geometry information. First, for sparse radar points, we propose a local completion to supplement missing structures and provide denser guideline for aligning two frames. Then, a context-aware association with a hierarchical structure flexibly matches points of different scales aided by feature similarity, and improves local matching consistency through correlation balancing. Finally, we present a window-based optimizer that uses historical priors to establish a coupling state estimation and correct errors of inter-frame matching. The superiority of our algorithm is confirmed on View-of-Delft dataset, achieving around a 50% performance improvement over previous approaches and delivering accuracy on par with LiDAR odometry. Our code will be available. 

**Abstract (ZH)**: 近期，4D毫米波雷达在恶劣条件（如雨雾）下的感知能力比LiDAR和相机更稳定。然而，低质量的雷达点云妨碍了其应用，尤其是在需要密集和精确匹配的建图任务中。为了充分利用4D雷达的潜力，我们提出了一种基于学习的里程计框架，能够从有限和不确定的几何信息中估计稳健的 ego 运动。首先，对于稀疏的雷达点云，我们提出了一种局部补全方法来补充缺失的结构，并提供更密集的对齐指导方针。然后，利用上下文感知的多层次结构进行特征相似匹配，并通过相关性平衡提高局部匹配一致性。最后，我们提出了一种基于窗口的优化器，利用历史先验建立耦合状态估计，并纠正帧间匹配的误差。在Delft视角数据集上的实验表明，我们的算法性能提高了约50%，精度与LiDAR里程计相当。我们的代码将开源。 

---
# RUSSO: Robust Underwater SLAM with Sonar Optimization against Visual Degradation 

**Title (ZH)**: RUSSO:  robust underwater SLAM with sonar optimization against visual degradation 

**Authors**: Shu Pan, Ziyang Hong, Zhangrui Hu, Xiandong Xu, Wenjie Lu, Liang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01434)  

**Abstract**: Visual degradation in underwater environments poses unique and significant challenges, which distinguishes underwater SLAM from popular vision-based SLAM on the ground. In this paper, we propose RUSSO, a robust underwater SLAM system which fuses stereo camera, inertial measurement unit (IMU), and imaging sonar to achieve robust and accurate localization in challenging underwater environments for 6 degrees of freedom (DoF) estimation. During visual degradation, the system is reduced to a sonar-inertial system estimating 3-DoF poses. The sonar pose estimation serves as a strong prior for IMU propagation, thereby enhancing the reliability of pose estimation with IMU propagation. Additionally, we propose a SLAM initialization method that leverages the imaging sonar to counteract the lack of visual features during the initialization stage of SLAM. We extensively validate RUSSO through experiments in simulator, pool, and sea scenarios. The results demonstrate that RUSSO achieves better robustness and localization accuracy compared to the state-of-the-art visual-inertial SLAM systems, especially in visually challenging scenarios. To the best of our knowledge, this is the first time fusing stereo camera, IMU, and imaging sonar to realize robust underwater SLAM against visual degradation. 

**Abstract (ZH)**: 水下视觉退化给水下SLAM带来了独特的和重要的挑战，这也使得水下SLAM不同于陆地上的视觉SLAM。本文提出了一种鲁棒的水下SLAM系统RUSSO，该系统融合了立体相机、惯性测量单元（IMU）和成像声纳，以在挑战性的水下环境中实现六自由度（6DoF）的鲁棒和准确定位。在视觉退化时，系统被简化为一个声纳-IMU系统，用于估计三维姿态。声纳姿态估计为IMU姿态传播提供了强大的先验信息，从而增强了姿态估计的可靠性。此外，我们提出了一种利用成像声纳进行SLAM初始化的方法，以弥补视觉特征不足的问题。我们通过在仿真实验、泳池和海场景况下进行广泛验证，结果表明，RUSSO在鲁棒性和定位精度方面优于最先进的视觉-惯性SLAM系统，特别是在视觉挑战性场景中表现尤为显著。据我们所知，这是首次将立体相机、IMU和成像声纳融合以实现针对视觉退化的鲁棒水下SLAM。 

---
# CognitiveDrone: A VLA Model and Evaluation Benchmark for Real-Time Cognitive Task Solving and Reasoning in UAVs 

**Title (ZH)**: 认知无人机：一种适用于 UAV 实时认知任务解决与推理的长距离模型及评估基准 

**Authors**: Artem Lykov, Valerii Serpiva, Muhammad Haris Khan, Oleg Sautenkov, Artyom Myshlyaev, Grik Tadevosyan, Yasheerah Yaqoot, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2503.01378)  

**Abstract**: This paper introduces CognitiveDrone, a novel Vision-Language-Action (VLA) model tailored for complex Unmanned Aerial Vehicles (UAVs) tasks that demand advanced cognitive abilities. Trained on a dataset comprising over 8,000 simulated flight trajectories across three key categories-Human Recognition, Symbol Understanding, and Reasoning-the model generates real-time 4D action commands based on first-person visual inputs and textual instructions. To further enhance performance in intricate scenarios, we propose CognitiveDrone-R1, which integrates an additional Vision-Language Model (VLM) reasoning module to simplify task directives prior to high-frequency control. Experimental evaluations using our open-source benchmark, CognitiveDroneBench, reveal that while a racing-oriented model (RaceVLA) achieves an overall success rate of 31.3%, the base CognitiveDrone model reaches 59.6%, and CognitiveDrone-R1 attains a success rate of 77.2%. These results demonstrate improvements of up to 30% in critical cognitive tasks, underscoring the effectiveness of incorporating advanced reasoning capabilities into UAV control systems. Our contributions include the development of a state-of-the-art VLA model for UAV control and the introduction of the first dedicated benchmark for assessing cognitive tasks in drone operations. The complete repository is available at this http URL 

**Abstract (ZH)**: 认知无人机：一种针对复杂无人机任务的新型视觉-语言-行动模型 

---
# FABG : End-to-end Imitation Learning for Embodied Affective Human-Robot Interaction 

**Title (ZH)**: FABG：端到端模仿学习在具身情感人机交互中的应用 

**Authors**: Yanghai Zhang, Changyi Liu, Keting Fu, Wenbin Zhou, Qingdu Li, Jianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01363)  

**Abstract**: This paper proposes FABG (Facial Affective Behavior Generation), an end-to-end imitation learning system for human-robot interaction, designed to generate natural and fluid facial affective behaviors. In interaction, effectively obtaining high-quality demonstrations remains a challenge. In this work, we develop an immersive virtual reality (VR) demonstration system that allows operators to perceive stereoscopic environments. This system ensures "the operator's visual perception matches the robot's sensory input" and "the operator's actions directly determine the robot's behaviors" - as if the operator replaces the robot in human interaction engagements. We propose a prediction-driven latency compensation strategy to reduce robotic reaction delays and enhance interaction fluency. FABG naturally acquires human interactive behaviors and subconscious motions driven by intuition, eliminating manual behavior scripting. We deploy FABG on a real-world 25-degree-of-freedom (DoF) humanoid robot, validating its effectiveness through four fundamental interaction tasks: expression response, dynamic gaze, foveated attention, and gesture recognition, supported by data collection and policy training. Project website: this https URL 

**Abstract (ZH)**: 本论文提出FABG（面部情感行为生成）系统，这是一个端到端的模仿学习系统，旨在用于人机交互，以生成自然流畅的面部情感行为。在交互过程中，有效获取高品质示范仍然是一项挑战。在此项工作中，我们开发了一种沉浸式虚拟现实（VR）示范系统，允许操作员感知立体环境。该系统确保“操作员的视觉感知与机器人的感输入相匹配”且“操作员的动作直接决定机器人的行为”——仿佛操作员替换了机器人在人机交互中的角色。我们提出了一种预测驱动的延迟补偿策略，以减少机器人反应延迟并提升交互流畅性。FABG自然地获取由直觉驱动的人际互动行为和潜意识动作，消除了手动行为脚本化的需求。我们在一个实际应用的25自由度（DoF）的人形机器人上部署FABG，并通过四种基本交互任务的数据收集和策略训练验证其有效性：表情响应、动态凝视、中心视觉关注和手势识别。项目网站：this https URL。 

---
# Flexible Exoskeleton Control Based on Binding Alignment Strategy and Full-arm Coordination Mechanism 

**Title (ZH)**: 基于绑定对齐策略与全臂协调机制的柔性外骨骼控制 

**Authors**: Chuang Cheng, Xinglong Zhang, Xieyuanli Chen, Wei Dai, Longwen Chen, Daoxun Zhang, Hui Zhang, Jie Jiang, Huimin Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01338)  

**Abstract**: In rehabilitation, powered, and teleoperation exoskeletons, connecting the human body to the exoskeleton through binding attachments is a common configuration. However, the uncertainty of the tightness and the donning deviation of the binding attachments will affect the flexibility and comfort of the exoskeletons, especially during high-speed movement. To address this challenge, this paper presents a flexible exoskeleton control approach with binding alignment and full-arm coordination. Firstly, the sources of the force interaction caused by donning offsets are analyzed, based on which the interactive force data is classified into the major, assistant, coordination, and redundant component categories. Then, a binding alignment strategy (BAS) is proposed to reduce the donning disturbances by combining different force data. Furthermore, we propose a full-arm coordination mechanism (FCM) that focuses on two modes of arm movement intent, joint-oriented and target-oriented, to improve the flexible performance of the whole exoskeleton control during high-speed motion. In this method, we propose an algorithm to distinguish the two intentions to resolve the conflict issue of the force component. Finally, a series of experiments covering various aspects of exoskeleton performance (flexibility, adaptability, accuracy, speed, and fatigue) were conducted to demonstrate the benefits of our control framework in our full-arm exoskeleton. 

**Abstract (ZH)**: 康复、助力和远程操作外骨骼中，通过绑定附件将人体连接到外骨骼是一种常见的配置。然而，绑定附件紧固程度的不确定性和佩戴偏差将影响外骨骼的灵活性和舒适性，尤其是在高速运动时。为应对这一挑战，本文提出了一种具备绑定对齐和全臂协调的灵活外骨骼控制方法。首先，基于穿戴偏移引起的作用力来源分析，将交互作用力数据分类为主要、辅助、协调和冗余组件类别。然后，提出了一种绑定对齐策略（BAS）以通过结合不同力数据来减少穿戴干扰。此外，我们提出了一种全臂协调机制（FCM），专注于手臂运动意图的两种模式——关节导向和目标导向，以提高在高速运动过程中整个外骨骼控制的灵活性。在此方法中，我们提出了一种算法来区分这两种意图以解决力组件冲突问题。最后，进行了一系列覆盖外骨骼性能各个方面的实验（灵活性、适应性、准确性、速度和疲劳），以证明我们控制框架在全臂外骨骼中的优势。 

---
# Few-shot Sim2Real Based on High Fidelity Rendering with Force Feedback Teleoperation 

**Title (ZH)**: 基于高保真渲染和力反馈遥控的少样本Sim2Real 

**Authors**: Yanwen Zou, Junda Huang, Boyuan Liang, Honghao Guo, Zhengyang Liu, Xin Ma, Jianshu Zhou, Masayoshi Tomizuka  

**Link**: [PDF](https://arxiv.org/pdf/2503.01301)  

**Abstract**: Teleoperation offers a promising approach to robotic data collection and human-robot interaction. However, existing teleoperation methods for data collection are still limited by efficiency constraints in time and space, and the pipeline for simulation-based data collection remains unclear. The problem is how to enhance task performance while minimizing reliance on real-world data. To address this challenge, we propose a teleoperation pipeline for collecting robotic manipulation data in simulation and training a few-shot sim-to-real visual-motor policy. Force feedback devices are integrated into the teleoperation system to provide precise end-effector gripping force feedback. Experiments across various manipulation tasks demonstrate that force feedback significantly improves both success rates and execution efficiency, particularly in simulation. Furthermore, experiments with different levels of visual rendering quality reveal that enhanced visual realism in simulation substantially boosts task performance while reducing the need for real-world data. 

**Abstract (ZH)**: 基于仿真的机器人操作数据采集及少样本视觉-运动策略训练的远程操控pipeline 

---
# Stone Soup Multi-Target Tracking Feature Extraction For Autonomous Search And Track In Deep Reinforcement Learning Environment 

**Title (ZH)**: 石 Soup 多目标跟踪特征提取在深度强化学习环境中的自主搜索与跟踪 

**Authors**: Jan-Hendrik Ewers, Joe Gibbs, David Anderson  

**Link**: [PDF](https://arxiv.org/pdf/2503.01293)  

**Abstract**: Management of sensing resources is a non-trivial problem for future military air assets with future systems deploying heterogeneous sensors to generate information of the battlespace. Machine learning techniques including deep reinforcement learning (DRL) have been identified as promising approaches, but require high-fidelity training environments and feature extractors to generate information for the agent. This paper presents a deep reinforcement learning training approach, utilising the Stone Soup tracking framework as a feature extractor to train an agent for a sensor management task. A general framework for embedding Stone Soup tracker components within a Gymnasium environment is presented, enabling fast and configurable tracker deployments for RL training using Stable Baselines3. The approach is demonstrated in a sensor management task where an agent is trained to search and track a region of airspace utilising track lists generated from Stone Soup trackers. A sample implementation using three neural network architectures in a search-and-track scenario demonstrates the approach and shows that RL agents can outperform simple sensor search and track policies when trained within the Gymnasium and Stone Soup environment. 

**Abstract (ZH)**: 未来的军事空中资产在部署异构传感器生成战场信息时，感知资源管理是一个非平凡的问题。机器学习技术包括深度强化学习（DRL）被认为是有前景的方法，但需要高保真度的训练环境和特征提取器来为代理生成信息。本文提出了一种基于深度强化学习的训练方法，利用Stone Soup跟踪框架作为特征提取器，训练一个用于传感器管理任务的代理。我们介绍了一个将Stone Soup跟踪组件嵌入Gymnasium环境中的通用框架，使得使用Stable Baselines3进行强化学习训练时能够快速且配置灵活地部署跟踪器。该方法在一项传感器管理任务中进行了演示，任务中代理被训练以利用Stone Soup跟踪器生成的航迹列表搜索和跟踪空中区域。在搜索和跟踪场景中，使用三种神经网络架构的示例实现展示了该方法，并且表明在Gymnasium和Stone Soup环境中训练的RL代理能够优于简单的传感器搜索和跟踪策略。 

---
# DnD Filter: Differentiable State Estimation for Dynamic Systems using Diffusion Models 

**Title (ZH)**: DnD Filter: 基于扩散模型的差分状态估计方法 

**Authors**: Ziyu Wan, Lin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.01274)  

**Abstract**: This paper proposes the DnD Filter, a differentiable filter that utilizes diffusion models for state estimation of dynamic systems. Unlike conventional differentiable filters, which often impose restrictive assumptions on process noise (e.g., Gaussianity), DnD Filter enables a nonlinear state update without such constraints by conditioning a diffusion model on both the predicted state and observational data, capitalizing on its ability to approximate complex distributions. We validate its effectiveness on both a simulated task and a real-world visual odometry task, where DnD Filter consistently outperforms existing baselines. Specifically, it achieves a 25\% improvement in estimation accuracy on the visual odometry task compared to state-of-the-art differentiable filters, and even surpasses differentiable smoothers that utilize future measurements. To the best of our knowledge, DnD Filter represents the first successful attempt to leverage diffusion models for state estimation, offering a flexible and powerful framework for nonlinear estimation under noisy measurements. 

**Abstract (ZH)**: 基于扩散模型的可微分滤波器DnD滤波器：一种用于动态系统状态估计的可微分滤波器 

---
# Design and Development of a Locomotion Interface for Virtual Reality Lower-Body Haptic Interaction 

**Title (ZH)**: 虚拟现实下体触觉交互的运动接口设计与开发 

**Authors**: An-Chi He, Jungsoo Park, Benjamin Beiter, Bhaben Kalita, Alexander Leonessa  

**Link**: [PDF](https://arxiv.org/pdf/2503.01271)  

**Abstract**: This work presents the design, build, control, and preliminary user data of a locomotion interface called ForceBot. It delivers lower-body haptic interaction in virtual reality (VR), enabling users to walk in VR while interacting with various simulated terrains. It utilizes two planar gantries to give each foot two degrees of freedom and passive heel-lifting motion. The design used motion capture data with dynamic simulation for ergonomic human-robot workspace and hardware selection. Its system framework uses open-source robotic software and pairs with a custom-built power delivery system that offers EtherCAT communication with a 1,000 Hz soft real-time computation rate. This system features an admittance controller to regulate physical human-robot interaction (pHRI) alongside a walking algorithm to generate walking motion and simulate virtual terrains. The system's performance is explored through three measurements that evaluate the relationship between user input force and output pHRI motion. Overall, this platform presents a unique approach by utilizing planar gantries to realize VR terrain interaction with an extensive workspace, reasonably compact footprint, and preliminary user data. 

**Abstract (ZH)**: 基于平面桁架的ForceBot步行接口设计、构建、控制及初步用户数据 

---
# Impact of Static Friction on Sim2Real in Robotic Reinforcement Learning 

**Title (ZH)**: 静态摩擦力对机器人强化学习中Sim2Real影响的研究 

**Authors**: Xiaoyi Hu, Qiao Sun, Bailin He, Haojie Liu, Xueyi Zhang, Chunpeng lu, Jiangwei Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2503.01255)  

**Abstract**: In robotic reinforcement learning, the Sim2Real gap remains a critical challenge. However, the impact of Static friction on Sim2Real has been underexplored. Conventional domain randomization methods typically exclude Static friction from their parameter space. In our robotic reinforcement learning task, such conventional domain randomization approaches resulted in significantly underperforming real-world models. To address this Sim2Real challenge, we employed Actuator Net as an alternative to conventional domain randomization. While this method enabled successful transfer to flat-ground locomotion, it failed on complex terrains like stairs. To further investigate physical parameters affecting Sim2Real in robotic joints, we developed a control-theoretic joint model and performed systematic parameter identification. Our analysis revealed unexpectedly high friction-torque ratios in our robotic joints. To mitigate its impact, we implemented Static friction-aware domain randomization for Sim2Real. Recognizing the increased training difficulty introduced by friction modeling, we proposed a simple and novel solution to reduce learning complexity. To validate this approach, we conducted comprehensive Sim2Sim and Sim2Real experiments comparing three methods: conventional domain randomization (without Static friction), Actuator Net, and our Static friction-aware domain randomization. All experiments utilized the Rapid Motor Adaptation (RMA) algorithm. Results demonstrated that our method achieved superior adaptive capabilities and overall performance. 

**Abstract (ZH)**: 机器人强化学习中，静态摩擦对Sim2Real的影响研究 

---
# Diffusion Stabilizer Policy for Automated Surgical Robot Manipulations 

**Title (ZH)**: 自动化手术机器人操作的扩散稳定器策略 

**Authors**: Chonlam Ho, Jianshu Hu, Hesheng Wang, Qi Dou, Yutong Ban  

**Link**: [PDF](https://arxiv.org/pdf/2503.01252)  

**Abstract**: Intelligent surgical robots have the potential to revolutionize clinical practice by enabling more precise and automated surgical procedures. However, the automation of such robot for surgical tasks remains under-explored compared to recent advancements in solving household manipulation tasks. These successes have been largely driven by (1) advanced models, such as transformers and diffusion models, and (2) large-scale data utilization. Aiming to extend these successes to the domain of surgical robotics, we propose a diffusion-based policy learning framework, called Diffusion Stabilizer Policy (DSP), which enables training with imperfect or even failed trajectories. Our approach consists of two stages: first, we train the diffusion stabilizer policy using only clean data. Then, the policy is continuously updated using a mixture of clean and perturbed data, with filtering based on the prediction error on actions. Comprehensive experiments conducted in various surgical environments demonstrate the superior performance of our method in perturbation-free settings and its robustness when handling perturbed demonstrations. 

**Abstract (ZH)**: 基于扩散的政策学习框架（Diffusion Stabilizer Policy）：促进手术机器人领域中的鲁棒政策训练 

---
# Catching Spinning Table Tennis Balls in Simulation with End-to-End Curriculum Reinforcement Learning 

**Title (ZH)**: 使用端到端 Curriculum 强化学习在仿真中捕捉旋转乒乓球 

**Authors**: Xiaoyi Hu, Yue Mao, Gang Wang, Qingdu Li, Jianwei Zhang, Yunfeng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2503.01251)  

**Abstract**: The game of table tennis is renowned for its extremely high spin rate, but most table tennis robots today struggle to handle balls with such rapid spin. To address this issue, we have contributed a series of methods, including: 1. Curriculum Reinforcement Learning (RL): This method helps the table tennis robot learn to play table tennis progressively from easy to difficult tasks. 2. Analysis of Spinning Table Tennis Ball Collisions: We have conducted a physics-based analysis to generate more realistic trajectories of spinning table tennis balls after collision. 3. Definition of Trajectory States: The definition of trajectory states aids in setting up the reward function. 4. Selection of Valid Rally Trajectories: We have introduced a valid rally trajectory selection scheme to ensure that the robot's training is not influenced by abnormal trajectories. 5. Reality-to-Simulation (Real2Sim) Transfer: This scheme is employed to validate the trained robot's ability to handle spinning balls in real-world scenarios. With Real2Sim, the deployment costs for robotic reinforcement learning can be further reduced. Moreover, the trajectory-state-based reward function is not limited to table tennis robots; it can be generalized to a wide range of cyclical tasks. To validate our robot's ability to handle spinning balls, the Real2Sim experiments were conducted. For the specific video link of the experiment, please refer to the supplementary materials. 

**Abstract (ZH)**: 乒乓球游戏以其极高的旋转率而闻名，但当今大多数乒乓球机器人难以处理带有快速旋转的球。为了解决这一问题，我们贡献了一系列方法，包括：1. 进阶强化学习（Curriculum Reinforcement Learning, RL）：该方法帮助乒乓球机器人从易到难逐步学习乒乓球技能。2. 旋转乒乓球碰撞分析：我们进行了基于物理学的分析，以生成更真实的碰撞后的旋转乒乓球轨迹。3. 轨迹状态定义：轨迹状态的定义有助于设置奖励函数。4. 有效回球轨迹选择：我们引入了有效回球轨迹选择方案，以确保机器人的训练不受异常轨迹的影响。5. 现实到模拟（Real2Sim）转移：该方案用于验证训练后的机器人在现实世界场景中处理旋转球的能力。使用Real2Sim可以进一步降低机器人强化学习的部署成本。此外，基于轨迹状态的奖励函数不仅适用于乒乓球机器人，还可推广到一系列循环任务。为验证我们机器人处理旋转球的能力，进行了Real2Sim实验。具体实验视频链接请参见补充材料。 

---
# A Taxonomy for Evaluating Generalist Robot Policies 

**Title (ZH)**: 通用机器人政策评估 taxonomy 

**Authors**: Jensen Gao, Suneel Belkhale, Sudeep Dasari, Ashwin Balakrishna, Dhruv Shah, Dorsa Sadigh  

**Link**: [PDF](https://arxiv.org/pdf/2503.01238)  

**Abstract**: Machine learning for robotics promises to unlock generalization to novel tasks and environments. Guided by this promise, many recent works have focused on scaling up robot data collection and developing larger, more expressive policies to achieve this. But how do we measure progress towards this goal of policy generalization in practice? Evaluating and quantifying generalization is the Wild West of modern robotics, with each work proposing and measuring different types of generalization in their own, often difficult to reproduce, settings. In this work, our goal is (1) to outline the forms of generalization we believe are important in robot manipulation in a comprehensive and fine-grained manner, and (2) to provide reproducible guidelines for measuring these notions of generalization. We first propose STAR-Gen, a taxonomy of generalization for robot manipulation structured around visual, semantic, and behavioral generalization. We discuss how our taxonomy encompasses most prior notions of generalization in robotics. Next, we instantiate STAR-Gen with a concrete real-world benchmark based on the widely-used Bridge V2 dataset. We evaluate a variety of state-of-the-art models on this benchmark to demonstrate the utility of our taxonomy in practice. Our taxonomy of generalization can yield many interesting insights into existing models: for example, we observe that current vision-language-action models struggle with various types of semantic generalization, despite the promise of pre-training on internet-scale language datasets. We believe STAR-Gen and our guidelines can improve the dissemination and evaluation of progress towards generalization in robotics, which we hope will guide model design and future data collection efforts. We provide videos and demos at our website this http URL. 

**Abstract (ZH)**: 机器学习在机器人领域的应用有望解锁对新任务和环境的一般化能力。受此启发，许多近期工作集中在扩大机器人数据采集规模并开发更大、更具表达性的策略以实现这一目标。但在实践中，我们如何衡量向这一政策一般化目标的进步呢？在现代机器人领域，评估和量化一般化仍是未开发的领域，每项工作都在其自身难以复现的设置中提出和衡量不同类型的一般化。在本文中，我们的目标是（1）以全面和细腻的方式概述我们认为在机器人操作中重要的各种一般化形式，（2）提供衡量这些一般化概念的可再现指南。我们首先提出STAR-Gen，这是一个基于视觉、语义和行为一般化的机器人操作一般化分类法。我们讨论了我们的分类法如何涵盖机器人学中大多数先前的一般化概念。接下来，我们基于广泛使用的Bridge V2数据集，具体化了STAR-Gen，并提出了一个具体的现实世界基准。我们评估了多种当前最先进的模型在该基准上的表现，以展示我们分类法在实践中的用途。我们的一般化分类法可以对现有模型提供许多有趣的认识：例如，尽管互联网规模语言数据集的预训练承诺，我们观察到当前的视觉-语言-动作模型在各种语义一般化方面存在困难。我们相信STAR-Gen和我们的指南可以改进机器人领域一般化进展的传播和评估，我们希望这将指导模型设计和未来的数据采集努力。我们在网站上提供了视频和演示，链接为 this http URL。 

---
# LLM-Advisor: An LLM Benchmark for Cost-efficient Path Planning across Multiple Terrains 

**Title (ZH)**: LLM-顾问：跨多种地形高效成本路径规划的LLM基准 

**Authors**: Ling Xiao, Toshihiko Yamasaki  

**Link**: [PDF](https://arxiv.org/pdf/2503.01236)  

**Abstract**: Multi-terrain cost-efficient path planning is a crucial task in robot navigation, requiring the identification of a path from the start to the goal that not only avoids obstacles but also minimizes travel costs. This is especially crucial for real-world applications where robots need to navigate diverse terrains in outdoor environments, where recharging or refueling is difficult. However, there is very limited research on this topic. In this paper, we develop a prompt-based approach, LLM-Advisor, which leverages large language models (LLMs) as effective advisors for path planning. The LLM-Advisor selectively provides suggestions, demonstrating its ability to recognize when no modifications are necessary. When suggestions are made, 70.59% of the paths suggested for the A* algorithm, 69.47% for the RRT* algorithm, and 78.70% for the LLM-A* algorithm achieve greater cost efficiency. Since LLM-Advisor may occasionally lack common sense in their suggestions, we propose two hallucination-mitigation strategies. Furthermore, we experimentally verified that GPT-4o performs poorly in zero-shot path planning, even when terrain descriptions are clearly provided, demonstrating its low spatial awareness. We also experimentally demonstrate that using an LLM as an advisor is more effective than directly integrating it into the path-planning loop. Since LLMs may generate hallucinations, using LLMs in the loop of a search-based method (such as A*) may lead to a higher number of failed paths, demonstrating that our proposed LLM-Advisor is a better choice. 

**Abstract (ZH)**: 多地形高效路径规划是机器人导航中的关键任务，要求识别从起点到终点的路径，不仅避免障碍物，还尽量降低成本。这一任务在室外环境中尤为重要，尤其是在地形多样且充电或加油困难的场景下。然而，针对这一主题的研究非常有限。本文开发了一种基于提示的方法——LLM-Advisor，利用大规模语言模型（LLMs）作为路径规划的有效顾问。LLM-Advisor选择性地提供建议，展示了其在识别无需修改情况时的能力。当建议被提出时，70.59%的路径对A*算法，69.47%的路径对RRT*算法，以及78.70%的路径对LLM-A*算法提高了成本效率。由于LLM-Advisor偶尔缺乏常识，我们提出了两种减少幻觉的方法。此外，实验验证表明，即使提供了明确的地形描述，GPT-4o在零样本路径规划中的表现也很差，显示出其较低的空间认知能力。我们还实验性地证明，使用LLM作为顾问比直接将其集成到路径规划循环中更有效。由于LLM可能生成幻觉，在基于搜索的方法（如A*）的循环中使用LLM可能导致更多路径失败，证明了我们提出的LLM-Advisor是一个更好的选择。 

---
# A Single Scale Doesn't Fit All: Adaptive Motion Scaling for Efficient and Precise Teleoperation 

**Title (ZH)**: 不是一成不变的尺度适用一切：自适应运动尺度化以实现高效的精确遥操作 

**Authors**: Jeonghyeon Yoon, Sanghyeok Park, Hyojae Park, Cholin Kim, Sihyeoung Park, Minho Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01216)  

**Abstract**: Teleoperation is increasingly employed in environments where direct human access is difficult, such as hazardous exploration or surgical field. However, if the motion scale factor(MSF) intended to compensate for workspace-size differences is set inappropriately, repeated clutching operations and reduced precision can significantly raise cognitive load. This paper presents a shared controller that dynamically applies the MSF based on the user's intended motion scale. Inspired by human motor skills, the leader arm trajectory is divided into coarse(fast, large-range movements) and fine(precise, small-range movements), with three features extracted to train a fuzzy C-means(FCM) clustering model that probabilistically classifies the user's motion scale. Scaling the robot's motion accordingly reduces unnecessary repetition for large-scale movements and enables more precise control for fine operations. Incorporating recent trajectory data into model updates and offering user feedback for adjusting the MSF range and response speed allows mutual adaptation between user and system. In peg transfer experiments, compared to using a fixed single scale, the proposed approach demonstrated improved task efficiency(number of clutching and task completion time decreased 38.46% and 11.96% respectively), while NASA-TLX scores confirmed a meaningful reduction(58.01% decreased) in cognitive load. This outcome suggests that a user-intent-based motion scale adjustment can effectively enhance both efficiency and precision in teleoperation. 

**Abstract (ZH)**: 基于用户意图的运动比例因子自适应控制器在远程操作中的应用研究 

---
# Action Tokenizer Matters in In-Context Imitation Learning 

**Title (ZH)**: Action Tokenizer 对于在上下文模仿学习中很重要。 

**Authors**: An Dinh Vuong, Minh Nhat Vu, Dong An, Ian Reid  

**Link**: [PDF](https://arxiv.org/pdf/2503.01206)  

**Abstract**: In-context imitation learning (ICIL) is a new paradigm that enables robots to generalize from demonstrations to unseen tasks without retraining. A well-structured action representation is the key to capturing demonstration information effectively, yet action tokenizer (the process of discretizing and encoding actions) remains largely unexplored in ICIL. In this work, we first systematically evaluate existing action tokenizer methods in ICIL and reveal a critical limitation: while they effectively encode action trajectories, they fail to preserve temporal smoothness, which is crucial for stable robotic execution. To address this, we propose LipVQ-VAE, a variational autoencoder that enforces the Lipschitz condition in the latent action space via weight normalization. By propagating smoothness constraints from raw action inputs to a quantized latent codebook, LipVQ-VAE generates more stable and smoother actions. When integrating into ICIL, LipVQ-VAE improves performance by more than 5.3% in high-fidelity simulators, with real-world experiments confirming its ability to produce smoother, more reliable trajectories. Code and checkpoints will be released. 

**Abstract (ZH)**: 基于上下文的模仿学习中动作分词器的Lipschitz正则化变分自编码器研究 

---
# Enhancing Deep Reinforcement Learning-based Robot Navigation Generalization through Scenario Augmentation 

**Title (ZH)**: 基于场景扩充提高深度强化学习robot导航的泛化能力 

**Authors**: Shanze Wang, Mingao Tan, Zhibo Yang, Xianghui Wang, Xiaoyu Shen, Hailong Huang, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01146)  

**Abstract**: This work focuses on enhancing the generalization performance of deep reinforcement learning-based robot navigation in unseen environments. We present a novel data augmentation approach called scenario augmentation, which enables robots to navigate effectively across diverse settings without altering the training scenario. The method operates by mapping the robot's observation into an imagined space, generating an imagined action based on this transformed observation, and then remapping this action back to the real action executed in simulation. Through scenario augmentation, we conduct extensive comparative experiments to investigate the underlying causes of suboptimal navigation behaviors in unseen environments. Our analysis indicates that limited training scenarios represent the primary factor behind these undesired behaviors. Experimental results confirm that scenario augmentation substantially enhances the generalization capabilities of deep reinforcement learning-based navigation systems. The improved navigation framework demonstrates exceptional performance by producing near-optimal trajectories with significantly reduced navigation time in real-world applications. 

**Abstract (ZH)**: 本研究致力于增强基于深度强化学习的机器人导航在未见环境下的泛化性能。我们提出了一种新颖的数据增强方法——情景增强，该方法允许机器人在不改变训练情景的情况下，有效地在多种不同的环境中导航。方法通过将机器人的观察映射到一个想象的空间，基于这个转换后的观察生成想象中的动作，然后将该动作重新映射回模拟中执行的实际动作。通过情景增强，我们进行了广泛的对比实验，以探究在未见环境中产生次优化导航行为的内在原因。分析表明，有限的训练情景是这些不良行为的主要原因。实验结果证实，情景增强显著增强了基于深度强化学习的导航系统的泛化能力。改进的导航框架在实际应用中表现出色，能够生成近乎最优的轨迹，并显著减少导航时间。 

---
# Beyond Visibility Limits: A DRL-Based Navigation Strategy for Unexpected Obstacles 

**Title (ZH)**: 超越可见性限制：基于DRL的意外障碍导航策略 

**Authors**: Mingao Tan, Shanze Wang, Biao Huang, Zhibo Yang, Rongfei Chen, Xiaoyu Shen, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01127)  

**Abstract**: Distance-based reward mechanisms in deep reinforcement learning (DRL) navigation systems suffer from critical safety limitations in dynamic environments, frequently resulting in collisions when visibility is restricted. We propose DRL-NSUO, a novel navigation strategy for unexpected obstacles that leverages the rate of change in LiDAR data as a dynamic environmental perception element. Our approach incorporates a composite reward function with environmental change rate constraints and dynamically adjusted weights through curriculum learning, enabling robots to autonomously balance between path efficiency and safety maximization. We enhance sensitivity to nearby obstacles by implementing short-range feature preprocessing of LiDAR data. Experimental results demonstrate that this method significantly improves both robot and pedestrian safety in complex scenarios compared to traditional DRL-based methods. When evaluated on the BARN navigation dataset, our method achieved superior performance with success rates of 94.0% at 0.5 m/s and 91.0% at 1.0 m/s, outperforming conservative obstacle expansion strategies. These results validate DRL-NSUO's enhanced practicality and safety for human-robot collaborative environments, including intelligent logistics applications. 

**Abstract (ZH)**: 基于距离的奖励机制在深度强化学习导航系统中于动态环境下的安全限制 critical 安全限制，在能见度受限的情况下经常导致碰撞。我们提出了一种新的 DRL-NSUO 导航策略，该策略利用 LiDAR 数据变化率作为动态环境感知元素以应对意外障碍。该方法结合了包含环境变化率约束的复合奖励函数，并通过课程学习动态调整权重，使机器人能够自主在路径效率和安全最大化之间取得平衡。通过实施短距离 LiDAR 数据特征预处理来增强对邻近障碍物的敏感度。实验结果表明，与传统的基于 DRL 的方法相比，该方法在复杂场景中显著提高了机器人和行人的安全性。在评估 BARN 导航数据集时，该方法在0.5 m/s 和 1.0 m/s 速度下分别取得了94.0% 和 91.0%的成功率，优于保守的障碍物扩展策略。这些结果验证了 DRL-NSUO 在人机协作环境中的增强实用性和安全性，包括智能物流应用。 

---
# TACO: General Acrobatic Flight Control via Target-and-Command-Oriented Reinforcement Learning 

**Title (ZH)**: TACO: 基于目标和指令导向的强化学习通用杂技飞行控制 

**Authors**: Zikang Yin, Canlun Zheng, Shiliang Guo, Zhikun Wang, Shiyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.01125)  

**Abstract**: Although acrobatic flight control has been studied extensively, one key limitation of the existing methods is that they are usually restricted to specific maneuver tasks and cannot change flight pattern parameters online. In this work, we propose a target-and-command-oriented reinforcement learning (TACO) framework, which can handle different maneuver tasks in a unified way and allows online parameter changes. Additionally, we propose a spectral normalization method with input-output rescaling to enhance the policy's temporal and spatial smoothness, independence, and symmetry, thereby overcoming the sim-to-real gap. We validate the TACO approach through extensive simulation and real-world experiments, demonstrating its capability to achieve high-speed circular flights and continuous multi-flips. 

**Abstract (ZH)**: 尽管杂技飞行控制已经得到了广泛研究，但现有方法的一个关键局限是通常局限于特定机动任务，不能在线更改飞行模式参数。本文提出了一种目标和指令导向的强化学习（TACO）框架，该框架能够以统一方式处理不同的机动任务，并允许在线参数变化。此外，我们提出了输入输出归一化的谱正则化方法，以增强策略的时间和空间连续性、独立性和对称性，从而克服模拟与现实之间的差距。通过广泛的仿真和实地实验验证了TACO方法的能力，展示了其实现高速圆周飞行和连续多翻转的能力。 

---
# Ground contact and reaction force sensing for linear policy control of quadruped robot 

**Title (ZH)**: quadruped 机器人线性策略控制的地接触与反作用力感知 

**Authors**: Harshita Mhaske, Aniket Mandhare, Jidong Huang, Yu Bai  

**Link**: [PDF](https://arxiv.org/pdf/2503.01102)  

**Abstract**: Designing robots capable of traversing uneven terrain and overcoming physical obstacles has been a longstanding challenge in the field of robotics. Walking robots show promise in this regard due to their agility, redundant DOFs and intermittent ground contact of locomoting appendages. However, the complexity of walking robots and their numerous DOFs make controlling them extremely difficult and computation heavy. Linear policies trained with reinforcement learning have been shown to perform adequately to enable quadrupedal walking, while being computationally light weight. The goal of this research is to study the effect of augmentation of observation space of a linear policy with newer state variables on performance of the policy. Since ground contact and reaction forces are the primary means of robot-environment interaction, they are essential state variables on which the linear policy must be informed. Experimental results show that augmenting the observation space with ground contact and reaction force data trains policies with better survivability, better stability against external disturbances and higher adaptability to untrained conditions. 

**Abstract (ZH)**: 设计能够在不平地形上通行并克服物理障碍的机器人一直是机器人领域的一个长期挑战。行走机器人由于其灵活性、冗余自由度以及移动肢体的间歇性地面接触，在这方面具有潜力。然而，行走机器人的复杂性和众多自由度使其控制变得极其困难且计算量大。使用强化学习训练的线性策略已被证明能够有效实现四足行走，同时具有较低的计算负载。本研究的目的是研究将新的状态变量添加到线性策略的观察空间中对策略性能的影响。由于地面接触和反应力是机器人与环境交互的主要方式，因此这些是线性策略必须获知的重要状态变量。实验结果表明，将地面接触和反应力数据添加到观察空间中训练出来的策略具有更好的生存能力、更好的对外部干扰的稳定性以及更高的对未训练条件的适应性。 

---
# Optimal Trajectory Planning for Cooperative Manipulation with Multiple Quadrotors Using Control Barrier Functions 

**Title (ZH)**: 基于控制屏障函数的多旋翼协同 manipulation 最优轨迹规划 

**Authors**: Arpan Pallar, Guanrui Li, Mrunal Sarvaiya, Giuseppe Loianno  

**Link**: [PDF](https://arxiv.org/pdf/2503.01096)  

**Abstract**: In this paper, we present a novel trajectory planning algorithm for cooperative manipulation with multiple quadrotors using control barrier functions (CBFs). Our approach addresses the complex dynamics of a system in which a team of quadrotors transports and manipulates a cable-suspended rigid-body payload in environments cluttered with obstacles. The proposed algorithm ensures obstacle avoidance for the entire system, including the quadrotors, cables, and the payload in all six degrees of freedom (DoF). We introduce the use of CBFs to enable safe and smooth maneuvers, effectively navigating through cluttered environments while accommodating the system's nonlinear dynamics. To simplify complex constraints, the system components are modeled as convex polytopes, and the Duality theorem is employed to reduce the computational complexity of the optimization problem. We validate the performance of our planning approach both in simulation and real-world environments using multiple quadrotors. The results demonstrate the effectiveness of the proposed approach in achieving obstacle avoidance and safe trajectory generation for cooperative transportation tasks. 

**Abstract (ZH)**: 基于控制屏障函数的多旋翼协作搬运中新颖轨迹规划算法 

---
# KineSoft: Learning Proprioceptive Manipulation Policies with Soft Robot Hands 

**Title (ZH)**: KineSoft: 学习基于软体手的本体感觉操作策略 

**Authors**: Uksang Yoo, Jonathan Francis, Jean Oh, Jeffrey Ichnowski  

**Link**: [PDF](https://arxiv.org/pdf/2503.01078)  

**Abstract**: Underactuated soft robot hands offer inherent safety and adaptability advantages over rigid systems, but developing dexterous manipulation skills remains challenging. While imitation learning shows promise for complex manipulation tasks, traditional approaches struggle with soft systems due to demonstration collection challenges and ineffective state representations. We present KineSoft, a framework enabling direct kinesthetic teaching of soft robotic hands by leveraging their natural compliance as a skill teaching advantage rather than only as a control challenge. KineSoft makes two key contributions: (1) an internal strain sensing array providing occlusion-free proprioceptive shape estimation, and (2) a shape-based imitation learning framework that uses proprioceptive feedback with a low-level shape-conditioned controller to ground diffusion-based policies. This enables human demonstrators to physically guide the robot while the system learns to associate proprioceptive patterns with successful manipulation strategies. We validate KineSoft through physical experiments, demonstrating superior shape estimation accuracy compared to baseline methods, precise shape-trajectory tracking, and higher task success rates compared to baseline imitation learning approaches. 

**Abstract (ZH)**: 柔oad机器人手内在的安全性和适应性优势使其在与刚性系统相比时更具优势，但在开发灵巧操作技能方面仍面临挑战。尽管模仿学习对复杂操作任务充满前景，但传统方法因演示收集难题和无效的状态表示而在软系统上挣扎。我们提出KineSoft框架，该框架通过利用软机器人手的自然顺应性作为一种技能教学优势而非仅作为控制挑战，实现实操教学。KineSoft 的两个主要贡献是：(1) 内部应变传感阵列提供无遮挡的本体感受形状估计，以及 (2) 一种基于形状的模仿学习框架，该框架使用本体感受反馈和低级形状条件控制器来接地扩散基策略。这使人类示范者能够在物理上引导机器人，同时系统学习将本体感受模式与成功的操作策略联系起来。我们通过物理实验验证了KineSoft，展示了与基线方法相比更高的形状估计准确性、精确的形状轨迹跟踪能力以及更高的任务成功率。 

---
# OceanSim: A GPU-Accelerated Underwater Robot Perception Simulation Framework 

**Title (ZH)**: OceanSim：一种基于GPU加速的水下机器人感知仿真框架 

**Authors**: Jingyu Song, Haoyu Ma, Onur Bagoren, Advaith V. Sethuraman, Yiting Zhang, Katherine A. Skinner  

**Link**: [PDF](https://arxiv.org/pdf/2503.01074)  

**Abstract**: Underwater simulators offer support for building robust underwater perception solutions. Significant work has recently been done to develop new simulators and to advance the performance of existing underwater simulators. Still, there remains room for improvement on physics-based underwater sensor modeling and rendering efficiency. In this paper, we propose OceanSim, a high-fidelity GPU-accelerated underwater simulator to address this research gap. We propose advanced physics-based rendering techniques to reduce the sim-to-real gap for underwater image simulation. We develop OceanSim to fully leverage the computing advantages of GPUs and achieve real-time imaging sonar rendering and fast synthetic data generation. We evaluate the capabilities and realism of OceanSim using real-world data to provide qualitative and quantitative results. The project page for OceanSim is this https URL. 

**Abstract (ZH)**: 水下模拟器为构建 robust 的水下感知解决方案提供支持。尽管已经开展了许多工作以开发新的模拟器并推进现有水下模拟器的性能，但在基于物理的水下传感器建模和渲染效率方面仍有改进空间。本文提出了一种高保真度的 GPU 加速水下模拟器 OceanSim，以解决这一研究缺口。我们提出先进的基于物理的渲染技术以减少水下图像模拟的模拟与现实差距。我们开发 OceanSim 以充分利用 GPU 的计算优势，实现实时声呐成像渲染和快速合成数据生成。我们使用真实世界的数据评估 OceanSim 的能力和真实性，提供定性和定量结果。OceanSim 的项目页面为：https://github.com/alibaba/OceanSim。 

---
# Language-Guided Object Search in Agricultural Environments 

**Title (ZH)**: 农业环境中基于语言的物体搜索 

**Authors**: Advaith Balaji, Saket Pradhan, Dmitry Berenson  

**Link**: [PDF](https://arxiv.org/pdf/2503.01068)  

**Abstract**: Creating robots that can assist in farms and gardens can help reduce the mental and physical workload experienced by farm workers. We tackle the problem of object search in a farm environment, providing a method that allows a robot to semantically reason about the location of an unseen target object among a set of previously seen objects in the environment using a Large Language Model (LLM). We leverage object-to-object semantic relationships to plan a path through the environment that will allow us to accurately and efficiently locate our target object while also reducing the overall distance traveled, without needing high-level room or area-level semantic relationships. During our evaluations, we found that our method outperformed a current state-of-the-art baseline and our ablations. Our offline testing yielded an average path efficiency of 84%, reflecting how closely the predicted path aligns with the ideal path. Upon deploying our system on the Boston Dynamics Spot robot in a real-world farm environment, we found that our system had a success rate of 80%, with a success weighted by path length of 0.67, which demonstrates a reasonable trade-off between task success and path efficiency under real-world conditions. The project website can be viewed at this https URL 

**Abstract (ZH)**: 在农场环境中创建能够辅助工作的机器人可以减轻农场工人的心身负担。我们解决了在农场环境中寻找物体的问题，提出了一种方法，利用大型语言模型（LLM）使机器人能够在环境中先前见过的一组物体中对未见过的目标物体的地理位置进行语义推理。我们利用物体间的语义关系进行路径规划，以便准确且高效地定位目标物体，同时减少总体行驶距离，无需依赖高层面的房间或区域级别的语义关系。在评估中，我们的方法优于当前最先进的基线和消融实验。我们的离线测试显示路径效率平均为84%，反映了预测路径与理想路径的接近程度。将系统部署在Boston Dynamics Spot机器人上，在真实农场环境中，系统的成功率为80%，按路径长度加权的成功率为0.67，这表明在实际条件下任务成功和路径效率之间的合理权衡。项目网站可访问：这个https URL。 

---
# General Force Sensation for Tactile Robot 

**Title (ZH)**: 全身力感知的触觉机器人 

**Authors**: Zhuo Chen, Ni Ou, Xuyang Zhang, Zhiyuan Wu, Yongqiang Zhao, Yupeng Wang, Nathan Lepora, Lorenzo Jamone, Jiankang Deng, Shan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2503.01058)  

**Abstract**: Robotic tactile sensors, including vision-based and taxel-based sensors, enable agile manipulation and safe human-robot interaction through force sensation. However, variations in structural configurations, measured signals, and material properties create domain gaps that limit the transferability of learned force sensation across different tactile sensors. Here, we introduce GenForce, a general framework for achieving transferable force sensation across both homogeneous and heterogeneous tactile sensors in robotic systems. By unifying tactile signals into marker-based binary tactile images, GenForce enables the transfer of existing force labels to arbitrary target sensors using a marker-to-marker translation technique with a few paired data. This process equips uncalibrated tactile sensors with force prediction capabilities through spatiotemporal force prediction models trained on the transferred data. Extensive experimental results validate GenForce's generalizability, accuracy, and robustness across sensors with diverse marker patterns, structural designs, material properties, and sensing principles. The framework significantly reduces the need for costly and labor-intensive labeled data collection, enabling the rapid deployment of multiple tactile sensors on robotic hands requiring force sensing capabilities. 

**Abstract (ZH)**: 基于基因力的跨异质触觉传感器可移植力感知框架 

---
# From Vague Instructions to Task Plans: A Feedback-Driven HRC Task Planning Framework based on LLMs 

**Title (ZH)**: 从模糊指令到任务计划：基于LLM的反馈驱动人机协作任务规划框架 

**Authors**: Afagh Mehri Shervedani, Matthew R. Walter, Milos Zefran  

**Link**: [PDF](https://arxiv.org/pdf/2503.01007)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated their potential as planners in human-robot collaboration (HRC) scenarios, offering a promising alternative to traditional planning methods. LLMs, which can generate structured plans by reasoning over natural language inputs, have the ability to generalize across diverse tasks and adapt to human instructions. This paper investigates the potential of LLMs to facilitate planning in the context of human-robot collaborative tasks, with a focus on their ability to reason from high-level, vague human inputs, and fine-tune plans based on real-time feedback. We propose a novel hybrid framework that combines LLMs with human feedback to create dynamic, context-aware task plans. Our work also highlights how a single, concise prompt can be used for a wide range of tasks and environments, overcoming the limitations of long, detailed structured prompts typically used in prior studies. By integrating user preferences into the planning loop, we ensure that the generated plans are not only effective but aligned with human intentions. 

**Abstract (ZH)**: 近期大型语言模型在-human-机器人协作规划中的进展：基于自然语言推理的通用能力和人机反馈动态调优框架 

---
# HWC-Loco: A Hierarchical Whole-Body Control Approach to Robust Humanoid Locomotion 

**Title (ZH)**: HWC-Loco：一种稳健的人形步行的分层次全身控制方法 

**Authors**: Sixu Lin, Guanren Qiao, Yunxin Tai, Ang Li, Kui Jia, Guiliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00923)  

**Abstract**: Humanoid robots, capable of assuming human roles in various workplaces, have become essential to the advancement of embodied intelligence. However, as robots with complex physical structures, learning a control model that can operate robustly across diverse environments remains inherently challenging, particularly under the discrepancies between training and deployment environments. In this study, we propose HWC-Loco, a robust whole-body control algorithm tailored for humanoid locomotion tasks. By reformulating policy learning as a robust optimization problem, HWC-Loco explicitly learns to recover from safety-critical scenarios. While prioritizing safety guarantees, overly conservative behavior can compromise the robot's ability to complete the given tasks. To tackle this challenge, HWC-Loco leverages a hierarchical policy for robust control. This policy can dynamically resolve the trade-off between goal-tracking and safety recovery, guided by human behavior norms and dynamic constraints. To evaluate the performance of HWC-Loco, we conduct extensive comparisons against state-of-the-art humanoid control models, demonstrating HWC-Loco's superior performance across diverse terrains, robot structures, and locomotion tasks under both simulated and real-world environments. 

**Abstract (ZH)**: humanoid机器人，能够在各种工作场所扮演人类角色，已成为推进体现智能的关键。然而，作为具有复杂物理结构的机器人，学习能够在不同环境中稳健运行的控制模型仍然充满挑战，特别是在训练环境与部署环境之间的差异条件下。在本研究中，我们提出HWC-Loco，一种针对人形移动任务的稳健全身控制算法。通过将策略学习重新定义为稳健优化问题，HWC-Loco明确地学习从关键安全场景中恢复。虽然优先考虑安全保证，但过于保守的行为可能损害机器人完成给定任务的能力。为应对这一挑战，HWC-Loco采用分层策略进行稳健控制。该策略可以根据人类行为规范和动态约束动态调节目标跟踪与安全恢复之间的权衡。为了评估HWC-Loco的性能，我们在模拟和真实世界环境中对不同地形、机器人结构和移动任务的现代人形控制模型进行了广泛的比较，结果显示HWC-Loco在各种条件下表现出更优的性能。 

---
# Efficient End-to-end Visual Localization for Autonomous Driving with Decoupled BEV Neural Matching 

**Title (ZH)**: 自主驾驶中基于解耦BEV神经匹配的高效端到端视觉定位 

**Authors**: Jinyu Miao, Tuopu Wen, Ziang Luo, Kangan Qian, Zheng Fu, Yunlong Wang, Kun Jiang, Mengmeng Yang, Jin Huang, Zhihua Zhong, Diange Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00862)  

**Abstract**: Accurate localization plays an important role in high-level autonomous driving systems. Conventional map matching-based localization methods solve the poses by explicitly matching map elements with sensor observations, generally sensitive to perception noise, therefore requiring costly hyper-parameter tuning. In this paper, we propose an end-to-end localization neural network which directly estimates vehicle poses from surrounding images, without explicitly matching perception results with HD maps. To ensure efficiency and interpretability, a decoupled BEV neural matching-based pose solver is proposed, which estimates poses in a differentiable sampling-based matching module. Moreover, the sampling space is hugely reduced by decoupling the feature representation affected by each DoF of poses. The experimental results demonstrate that the proposed network is capable of performing decimeter level localization with mean absolute errors of 0.19m, 0.13m and 0.39 degree in longitudinal, lateral position and yaw angle while exhibiting a 68.8% reduction in inference memory usage. 

**Abstract (ZH)**: 准确的定位在高级自动驾驶系统中发挥着重要作用。面向高效可解释的端到端定位神经网络：通过解耦BEV神经匹配模块直接从周围图像中估计车辆姿态，显著减少推理内存使用并实现分米级定位准确性。 

---
# T3: Multi-modal Tailless Triple-Flapping-Wing Robot for Efficient Aerial and Terrestrial Locomotion 

**Title (ZH)**: T3：多模态无尾三振动翼地面与空中移动机器人 

**Authors**: Xiangyu Xu, Zhi Zheng, Jin Wang, Yikai Chen, Jingyang Huang, Ruixin Wu, Huan Yu, Guodong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00805)  

**Abstract**: Flapping-wing robots offer great versatility; however, achieving efficient multi-modal locomotion remains challenging. This paper presents the design, modeling, and experimentation of T3, a novel tailless flapping-wing robot with three pairs of independently actuated wings. Inspired by juvenile water striders, T3 incorporates bio-inspired elastic passive legs that effectively transmit vibrations generated during wing flapping, enabling ground movement without additional motors. This novel mechanism facilitates efficient multi-modal locomotion while minimizing actuator usage, reducing complexity, and enhancing performance. An SE(3)-based controller ensures precise trajectory tracking and seamless mode transition. To validate T3's effectiveness, we developed a fully functional prototype and conducted targeted modeling, real-world experiments, and benchmark comparisons. The results demonstrate the robot's and controller's outstanding performance, underscoring the potential of multi-modal flapping-wing technologies for future aerial-ground robotic applications. 

**Abstract (ZH)**: 无尾拍翼机器人T3的设计、建模与实验：基于仿生弹性 Legs 的高效多模态运动 

---
# Detecting Heel Strike and toe off Events Using Kinematic Methods and LSTM Models 

**Title (ZH)**: 使用运动学方法和LSTM模型检测足跟打击和脚趾离地事件 

**Authors**: Longbin Zhang, Tsung-Lin Wu, Ananda Sidarta, Xiaoyue Yan, Prayook Jatesiktat, Kailun Yang, Wei Tech Ang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00794)  

**Abstract**: Accurate gait event detection is crucial for gait analysis, rehabilitation, and assistive technology, particularly in exoskeleton control, where precise identification of stance and swing phases is essential. This study evaluated the performance of seven kinematics-based methods and a Long Short-Term Memory (LSTM) model for detecting heel strike and toe-off events across 4363 gait cycles from 588 able-bodied subjects. The results indicated that while the Zeni et al. method achieved the highest accuracy among kinematics-based approaches, other methods exhibited systematic biases or required dataset-specific tuning. The LSTM model performed comparably to Zeni et al., providing a data-driven alternative without systematic bias. These findings highlight the potential of deep learning-based approaches for gait event detection while emphasizing the need for further validation in clinical populations and across diverse gait conditions. Future research will explore the generalizability of these methods in pathological populations, such as individuals with post-stroke conditions and knee osteoarthritis, as well as their robustness across varied gait conditions and data collection settings to enhance their applicability in rehabilitation and exoskeleton control. 

**Abstract (ZH)**: 准确的步伐事件检测对于步态分析、康复和辅助技术，特别是在外骨骼控制中至关重要，其中精确识别支撑相和摆动相是 essentials。本研究评估了七种基于运动学的方法和长短期记忆（LSTM）模型在588名健康受试者4363个步态周期中检测足跟撞击和足趾离地事件的性能。结果表明，尽管Zeni等人方法在基于运动学的方法中实现了最高的准确性，但其他方法存在系统偏差或需要特定数据集的调整。LSTM模型的表现与Zeni等人相当，提供了一种无系统偏差的数据驱动替代方案。这些发现突显了基于深度学习的方法在步态事件检测中的潜在价值，同时强调了这些方法在临床人群和不同步态条件下进一步验证的必要性。未来的研究将继续探索这些方法在外周神经系统疾病患者（如中风后患者和膝关节骨关节炎患者）等病理人群中的一般适用性，以及它们在各种步态条件和数据采集环境下的鲁棒性，以增强其在康复和外骨骼控制中的应用。 

---
# Development of a Five-Fingerd Biomimetic Soft Robotic Hand by 3D Printing the Skin and Skeleton as One Unit 

**Title (ZH)**: 基于3D打印皮肤和骨架一体单位的五指仿生软体手的发展 

**Authors**: Kazuhiro Miyama, Kento Kawaharazuka, Kei Okada, Masayuki Inaba  

**Link**: [PDF](https://arxiv.org/pdf/2503.00789)  

**Abstract**: Robot hands that imitate the shape of the human body have been actively studied, and various materials and mechanisms have been proposed to imitate the human body. Although the use of soft materials is advantageous in that it can imitate the characteristics of the human body's epidermis, it increases the number of parts and makes assembly difficult in order to perform complex movements. In this study, we propose a skin-skeleton integrated robot hand that has 15 degrees of freedom and consists of four parts. The developed robotic hand is mostly composed of a single flexible part produced by a 3D printer, and while it can be easily assembled, it can perform adduction, flexion, and opposition of the thumb, as well as flexion of four fingers. 

**Abstract (ZH)**: 模仿人体形状的机器人手已被积极研究，提出各种材料和机制以模仿人体特征。尽管使用软材料有利于模仿人体表皮的特性，但在进行复杂运动时会增加部件数量并使组装变得困难。本研究提出了一种具有15自由度、由四部分组成的皮肤-骨骼集成机器人手。开发的机器人手主要由3D打印的单个柔性部件组成，不仅易于组装，还可实现拇指的内收、弯曲和对掌以及四指的弯曲。 

---
# FLOAT Drone: A Fully-actuated Coaxial Aerial Robot for Close-Proximity Operations 

**Title (ZH)**: FLOAT无人机：一种全驱动共轴航空机器人，适用于近距离操作 

**Authors**: Junxiao Lin, Shuhang Ji, Yuze Wu, Tianyue Wu, Zhichao Han, Fei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.00785)  

**Abstract**: How to endow aerial robots with the ability to operate in close proximity remains an open problem. The core challenges lie in the propulsion system's dual-task requirement: generating manipulation forces while simultaneously counteracting gravity. These competing demands create dynamic coupling effects during physical interactions. Furthermore, rotor-induced airflow disturbances critically undermine operational reliability. Although fully-actuated unmanned aerial vehicles (UAVs) alleviate dynamic coupling effects via six-degree-of-freedom (6-DoF) force-torque decoupling, existing implementations fail to address the aerodynamic interference between drones and environments. They also suffer from oversized designs, which compromise maneuverability and limit their applications in various operational scenarios. To address these limitations, we present FLOAT Drone (FuLly-actuated cOaxial Aerial roboT), a novel fully-actuated UAV featuring two key structural innovations. By integrating control surfaces into fully-actuated systems for the first time, we significantly suppress lateral airflow disturbances during operations. Furthermore, a coaxial dual-rotor configuration enables a compact size while maintaining high hovering efficiency. Through dynamic modeling, we have developed hierarchical position and attitude controllers that support both fully-actuated and underactuated modes. Experimental validation through comprehensive real-world experiments confirms the system's functional capabilities in close-proximity operations. 

**Abstract (ZH)**: 如何赋予空中机器人在近距离操作的能力仍然是一个开放问题。核心挑战在于推进系统的双重任务要求：在同时对抗重力的情况下产生操作力。这些相互竞争的需求在物理交互过程中产生了动态耦合效应。此外，旋翼诱导的气流扰动严重削弱了操作可靠性。尽管全驱动的无人驾驶飞行器（UAV）通过六自由度（6-DoF）力-力矩解耦可以缓解动态耦合效应，但现有的实现方式未能解决无人机与环境之间的气动干扰问题。同时，它们也受到过大设计的影响，这损害了机动性并限制了在各种操作场景中的应用。为了克服这些局限性，我们提出了一种名为FLOAT Drone（全驱动共轴空中机器人）的创新设计，该设计包含两项关键的结构创新。通过首次将控制面整合进全驱动系统中，显著抑制了操作过程中的侧向气流扰动。此外，共轴双旋翼配置实现了紧凑的尺寸的同时保持了高悬停效率。通过动力学建模，我们开发了支持全驱动和欠驱动模式的分层位置和姿态控制器。全面的实地实验验证了该系统在近距离操作中的功能能力。 

---
# CARIL: Confidence-Aware Regression in Imitation Learning for Autonomous Driving 

**Title (ZH)**: CARIL: 带有置信度感知的 imitation 学习在自动驾驶中的回归关键技术 

**Authors**: Elahe Delavari, Aws Khalil, Jaerock Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2503.00783)  

**Abstract**: End-to-end vision-based imitation learning has demonstrated promising results in autonomous driving by learning control commands directly from expert demonstrations. However, traditional approaches rely on either regressionbased models, which provide precise control but lack confidence estimation, or classification-based models, which offer confidence scores but suffer from reduced precision due to discretization. This limitation makes it challenging to quantify the reliability of predicted actions and apply corrections when necessary. In this work, we introduce a dual-head neural network architecture that integrates both regression and classification heads to improve decision reliability in imitation learning. The regression head predicts continuous driving actions, while the classification head estimates confidence, enabling a correction mechanism that adjusts actions in low-confidence scenarios, enhancing driving stability. We evaluate our approach in a closed-loop setting within the CARLA simulator, demonstrating its ability to detect uncertain actions, estimate confidence, and apply real-time corrections. Experimental results show that our method reduces lane deviation and improves trajectory accuracy by up to 50%, outperforming conventional regression-only models. These findings highlight the potential of classification-guided confidence estimation in enhancing the robustness of vision-based imitation learning for autonomous driving. The source code is available at this https URL. 

**Abstract (ZH)**: 基于视觉的端到端模仿学习通过直接从专家演示中学习控制命令，在自主驾驶领域展示了 promising 的结果。然而，传统方法依赖于提供精确控制但缺乏信心估计的回归模型，或提供信心评分但因离散化而精度降低的分类模型。这一局限使得量化预测动作的可靠性并在必要时进行修正变得具有挑战性。在本工作中，我们提出了一种双头神经网络架构，将回归和分类头集成在一起以提高模仿学习中的决策可靠性。回归头预测连续驾驶动作，而分类头估计信心，使在低信心场景下能够调整动作，从而增强驾驶稳定性。我们在 CARLA 模拟器中闭环环境中评估了我们的方法，展示了其检测不确定动作、估计信心并实时修正的能力。实验结果表明，与仅使用回归模型相比，我们的方法将车道偏离减少，并且轨迹精度提高最多 50%。这些发现突显了分类引导的信心估计在增强基于视觉的模仿学习鲁棒性方面的潜力。源代码可在以下链接获取：this https URL。 

---
# Phantom: Training Robots Without Robots Using Only Human Videos 

**Title (ZH)**: 幻影：仅使用人类视频训练机器人 

**Authors**: Marion Lepert, Jiaying Fang, Jeannette Bohg  

**Link**: [PDF](https://arxiv.org/pdf/2503.00779)  

**Abstract**: Scaling robotics data collection is critical to advancing general-purpose robots. Current approaches often rely on teleoperated demonstrations which are difficult to scale. We propose a novel data collection method that eliminates the need for robotics hardware by leveraging human video demonstrations. By training imitation learning policies on this human data, our approach enables zero-shot deployment on robots without collecting any robot-specific data. To bridge the embodiment gap between human and robot appearances, we utilize a data editing approach on the input observations that aligns the image distributions between training data on humans and test data on robots. Our method significantly reduces the cost of diverse data collection by allowing anyone with an RGBD camera to contribute. We demonstrate that our approach works in diverse, unseen environments and on varied tasks. 

**Abstract (ZH)**: 通过利用人类视频示范来扩展机器人数据收集对于推进通用机器人技术至关重要。我们提出了一种新颖的数据收集方法，该方法通过利用人类视频示范而不依赖于机器人硬件，从而消除了对机器人特定数据的收集需求。为了弥合人类和机器人外观之间的主体差异，我们对输入观察数据进行数据编辑，以使训练数据的人像分布与测试数据的机器人分布对齐。我们的方法通过允许任何人使用RGBD相机贡献数据，显著降低了多样数据收集的成本。我们证明了该方法在未见过的多种环境和任务中有效。 

---
# AffordGrasp: In-Context Affordance Reasoning for Open-Vocabulary Task-Oriented Grasping in Clutter 

**Title (ZH)**: AffordGrasp: 开放词汇任务导向杂乱环境中的获取能力推理 

**Authors**: Yingbo Tang, Shuaike Zhang, Xiaoshuai Hao, Pengwei Wang, Jianlong Wu, Zhongyuan Wang, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00778)  

**Abstract**: Inferring the affordance of an object and grasping it in a task-oriented manner is crucial for robots to successfully complete manipulation tasks. Affordance indicates where and how to grasp an object by taking its functionality into account, serving as the foundation for effective task-oriented grasping. However, current task-oriented methods often depend on extensive training data that is confined to specific tasks and objects, making it difficult to generalize to novel objects and complex scenes. In this paper, we introduce AffordGrasp, a novel open-vocabulary grasping framework that leverages the reasoning capabilities of vision-language models (VLMs) for in-context affordance reasoning. Unlike existing methods that rely on explicit task and object specifications, our approach infers tasks directly from implicit user instructions, enabling more intuitive and seamless human-robot interaction in everyday scenarios. Building on the reasoning outcomes, our framework identifies task-relevant objects and grounds their part-level affordances using a visual grounding module. This allows us to generate task-oriented grasp poses precisely within the affordance regions of the object, ensuring both functional and context-aware robotic manipulation. Extensive experiments demonstrate that AffordGrasp achieves state-of-the-art performance in both simulation and real-world scenarios, highlighting the effectiveness of our method. We believe our approach advances robotic manipulation techniques and contributes to the broader field of embodied AI. Project website: this https URL. 

**Abstract (ZH)**: 基于任务的物体利用能力和抓取：一种利用视觉语言模型进行上下文推理的开放式词汇抓取框架 

---
# Shadow: Leveraging Segmentation Masks for Cross-Embodiment Policy Transfer 

**Title (ZH)**: Shadow: 利用分割掩码进行跨 bodys良政策转移 

**Authors**: Marion Lepert, Ria Doshi, Jeannette Bohg  

**Link**: [PDF](https://arxiv.org/pdf/2503.00774)  

**Abstract**: Data collection in robotics is spread across diverse hardware, and this variation will increase as new hardware is developed. Effective use of this growing body of data requires methods capable of learning from diverse robot embodiments. We consider the setting of training a policy using expert trajectories from a single robot arm (the source), and evaluating on a different robot arm for which no data was collected (the target). We present a data editing scheme termed Shadow, in which the robot during training and evaluation is replaced with a composite segmentation mask of the source and target robots. In this way, the input data distribution at train and test time match closely, enabling robust policy transfer to the new unseen robot while being far more data efficient than approaches that require co-training on large amounts of data from diverse embodiments. We demonstrate that an approach as simple as Shadow is effective both in simulation on varying tasks and robots, and on real robot hardware, where Shadow demonstrates an average of over 2x improvement in success rate compared to the strongest baseline. 

**Abstract (ZH)**: 机器人领域的数据收集分布于多种硬件之上，随着新硬件的开发，这种差异将会增大。有效利用不断增加的数据体需要能够从多种机器人实体中学习的方法。我们考虑从单个机器人手臂（源）的专家轨迹训练一个策略，并在没有收集数据的不同机器人手臂（目标）上进行评估。我们提出了一种名为Shadow的数据编辑方案，在此方案中，训练和评估时的机器人被源机器人和目标机器人的复合分割掩码所替换。这样，训练和测试时的输入数据分布非常接近，能够稳健地将策略转移到新的未见机器人，且相比于需要在大量多样化实体数据上共同训练的方法，这种方法更为数据高效。我们证明，在不同的任务和机器人以及真实机器人硬件上，Shadow方法在成功率上平均优于最强基线约2倍。 

---
# Disturbance Estimation of Legged Robots: Predefined Convergence via Dynamic Gains 

**Title (ZH)**: 腿足机器人扰动估计算法：基于动态增益的预先定义收敛性 

**Authors**: Bolin Li, Peiyuan Cai, Gewei Zuo, Lijun Zhu, Han Ding  

**Link**: [PDF](https://arxiv.org/pdf/2503.00769)  

**Abstract**: In this study, we address the challenge of disturbance estimation in legged robots by introducing a novel continuous-time online feedback-based disturbance observer that leverages measurable variables. The distinct feature of our observer is the integration of dynamic gains and comparison functions, which guarantees predefined convergence of the disturbance estimation error, including ultimately uniformly bounded, asymptotic, and exponential convergence, among various types. The properties of dynamic gains and the sufficient conditions for comparison functions are detailed to guide engineers in designing desired convergence behaviors. Notably, the observer functions effectively without the need for upper bound information of the disturbance or its derivative, enhancing its engineering applicability. An experimental example corroborates the theoretical advancements achieved. 

**Abstract (ZH)**: 本研究通过引入一种新型的连续时间在线反馈型扰动观察器来应对腿式机器人中的扰动估计挑战，该观察器利用可测量变量。观察器的独特之处在于集成了动态增益和比较函数，确保扰动估计误差在不同类型中达到预定义的收敛性，包括终极一致有界、渐近和指数收敛。详细讨论了动态增益的性质和比较函数的充分条件，以指导工程师设计所需的收敛行为。值得注意的是，该观察器在无需扰动及其导数的上界信息的情况下有效运行，增强了其实用性。实验示例验证了理论进展。 

---
# Enhanced $A^{*}$ Algorithm for Mobile Robot Path Planning with Non-Holonomic Constraints 

**Title (ZH)**: 非holonomic约束下增强的$A^{*}$算法在移动机器人路径规划中的应用 

**Authors**: Suraj Kumar, Sudheendra R, Aditya R, Bharat Kumar GVP, Ravi Kumar L  

**Link**: [PDF](https://arxiv.org/pdf/2503.00764)  

**Abstract**: In this paper, a novel method for path planning of mobile robots is proposed, taking into account the non-holonomic turn radius constraints and finite dimensions of the robot. The approach involves rasterizing the environment to generate a 2D map and utilizes an enhanced version of the $A^{*}$ algorithm that incorporates non-holonomic constraints while ensuring collision avoidance. Two new instantiations of the $A^{*}$ algorithm are introduced and tested across various scenarios and environments, with results demonstrating the effectiveness of the proposed method. 

**Abstract (ZH)**: 本文提出了一种考虑非全自由度转向半径约束和机器人有限尺寸的移动机器人路径规划新方法。该方法通过栅格化环境生成2D地图，并利用一种整合非全自由度约束并确保避障的增强版$A^{*}$算法。介绍了两种新的$A^{*}$算法实例，并在多种场景和环境中进行测试，结果表明所提出方法的有效性。 

---
# TRACE: A Self-Improving Framework for Robot Behavior Forecasting with Vision-Language Models 

**Title (ZH)**: TRACE：一种基于视觉语言模型的自我提升机器人行为预测框架 

**Authors**: Gokul Puthumanaillam, Paulo Padrao, Jose Fuentes, Pranay Thangeda, William E. Schafer, Jae Hyuk Song, Karan Jagdale, Leonardo Bobadilla, Melkior Ornik  

**Link**: [PDF](https://arxiv.org/pdf/2503.00761)  

**Abstract**: Predicting the near-term behavior of a reactive agent is crucial in many robotic scenarios, yet remains challenging when observations of that agent are sparse or intermittent. Vision-Language Models (VLMs) offer a promising avenue by integrating textual domain knowledge with visual cues, but their one-shot predictions often miss important edge cases and unusual maneuvers. Our key insight is that iterative, counterfactual exploration--where a dedicated module probes each proposed behavior hypothesis, explicitly represented as a plausible trajectory, for overlooked possibilities--can significantly enhance VLM-based behavioral forecasting. We present TRACE (Tree-of-thought Reasoning And Counterfactual Exploration), an inference framework that couples tree-of-thought generation with domain-aware feedback to refine behavior hypotheses over multiple rounds. Concretely, a VLM first proposes candidate trajectories for the agent; a counterfactual critic then suggests edge-case variations consistent with partial observations, prompting the VLM to expand or adjust its hypotheses in the next iteration. This creates a self-improving cycle where the VLM progressively internalizes edge cases from previous rounds, systematically uncovering not only typical behaviors but also rare or borderline maneuvers, ultimately yielding more robust trajectory predictions from minimal sensor data. We validate TRACE on both ground-vehicle simulations and real-world marine autonomous surface vehicles. Experimental results show that our method consistently outperforms standard VLM-driven and purely model-based baselines, capturing a broader range of feasible agent behaviors despite sparse sensing. Evaluation videos and code are available at this http URL. 

**Abstract (ZH)**: 基于迭代反事实探索的反应性代理近期行为预测 

---
# CLEA: Closed-Loop Embodied Agent for Enhancing Task Execution in Dynamic Environments 

**Title (ZH)**: CLEA: 闭环躯体化代理，以提高动态环境中的任务执行能力 

**Authors**: Mingcong Lei, Ge Wang, Yiming Zhao, Zhixin Mai, Qing Zhao, Yao Guo, Zhen Li, Shuguang Cui, Yatong Han, Jinke Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.00729)  

**Abstract**: Large Language Models (LLMs) exhibit remarkable capabilities in the hierarchical decomposition of complex tasks through semantic reasoning. However, their application in embodied systems faces challenges in ensuring reliable execution of subtask sequences and achieving one-shot success in long-term task completion. To address these limitations in dynamic environments, we propose Closed-Loop Embodied Agent (CLEA) -- a novel architecture incorporating four specialized open-source LLMs with functional decoupling for closed-loop task management. The framework features two core innovations: (1) Interactive task planner that dynamically generates executable subtasks based on the environmental memory, and (2) Multimodal execution critic employing an evaluation framework to conduct a probabilistic assessment of action feasibility, triggering hierarchical re-planning mechanisms when environmental perturbations exceed preset thresholds. To validate CLEA's effectiveness, we conduct experiments in a real environment with manipulable objects, using two heterogeneous robots for object search, manipulation, and search-manipulation integration tasks. Across 12 task trials, CLEA outperforms the baseline model, achieving a 67.3% improvement in success rate and a 52.8% increase in task completion rate. These results demonstrate that CLEA significantly enhances the robustness of task planning and execution in dynamic environments. 

**Abstract (ZH)**: 闭合回路实体化代理（CLEA）：具备功能解耦的新型层级任务管理架构 

---
# From Understanding the World to Intervening in It: A Unified Multi-Scale Framework for Embodied Cognition 

**Title (ZH)**: 从理解世界到干预世界：统一的多层次框架下的体现认知 

**Authors**: Maijunxian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00727)  

**Abstract**: In this paper, we propose AUKAI, an Adaptive Unified Knowledge-Action Intelligence for embodied cognition that seamlessly integrates perception, memory, and decision-making via multi-scale error feedback. Interpreting AUKAI as an embedded world model, our approach simultaneously predicts state transitions and evaluates intervention utility. The framework is underpinned by rigorous theoretical analysis drawn from convergence theory, optimal control, and Bayesian inference, which collectively establish conditions for convergence, stability, and near-optimal performance. Furthermore, we present a hybrid implementation that combines the strengths of neural networks with symbolic reasoning modules, thereby enhancing interpretability and robustness. Finally, we demonstrate the potential of AUKAI through a detailed application in robotic navigation and obstacle avoidance, and we outline comprehensive experimental plans to validate its effectiveness in both simulated and real-world environments. 

**Abstract (ZH)**: 本文提出了一种自适应统一知行智能AUKAI，该智能用于体现认知，通过多尺度误差反馈无缝地整合感知、记忆和决策。将AUKAI视为嵌入式世界模型，我们的方法同时预测状态转换并评估干预效益。该框架基于收敛理论、最优控制和贝叶斯推断的严格理论分析，共同建立了收敛、稳定性和近乎最优性能的条件。此外，我们提出了一种混合实现方法，结合了神经网络和符号推理模块的优势，以增强可解释性和鲁棒性。最后，我们通过详细的机器人导航和障碍避免应用展示了AUKAI的潜力，并概述了全面的实验计划，以在模拟和真实环境中验证其有效性。 

---
# ICanC: Improving Camera-based Object Detection and Energy Consumption in Low-Illumination Environments 

**Title (ZH)**: ICanC: 提高低光照环境中基于相机的目标检测和能耗性能 

**Authors**: Daniel Ma, Ren Zhong, Weisong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2503.00709)  

**Abstract**: This paper introduces ICanC (pronounced "I Can See"), a novel system designed to enhance object detection and optimize energy efficiency in autonomous vehicles (AVs) operating in low-illumination environments. By leveraging the complementary capabilities of LiDAR and camera sensors, ICanC improves detection accuracy under conditions where camera performance typically declines, while significantly reducing unnecessary headlight usage. This approach aligns with the broader objective of promoting sustainable transportation.
ICanC comprises three primary nodes: the Obstacle Detector, which processes LiDAR point cloud data to fit bounding boxes onto detected objects and estimate their position, velocity, and orientation; the Danger Detector, which evaluates potential threats using the information provided by the Obstacle Detector; and the Light Controller, which dynamically activates headlights to enhance camera visibility solely when a threat is detected.
Experiments conducted in physical and simulated environments demonstrate ICanC's robust performance, even in the presence of significant noise interference. The system consistently achieves high accuracy in camera-based object detection when headlights are engaged, while significantly reducing overall headlight energy consumption. These results position ICanC as a promising advancement in autonomous vehicle research, achieving a balance between energy efficiency and reliable object detection. 

**Abstract (ZH)**: 本文介绍了一种新型系统ICanC（发音为“I Can See”），该系统旨在增强自主车辆（AVs）在低照明环境中物体检测的准确性和优化光能效率。ICanC通过结合LiDAR和摄像头传感器的优势，在传统摄像头性能下降的情况下提高检测准确性，同时大幅减少不必要的前灯使用。该方法符合推动可持续交通的总体目标。 

---
# Safe Periodic Trochoidal Paths for Fixed-Wing UAVs in Confined Windy Environments 

**Title (ZH)**: 固定翼无人机在受限多风环境中安全周期 trochoidal 航线规划 

**Authors**: Jaeyoung Lim, David Rohr, Thomas Stastny, Roland Siegwart  

**Link**: [PDF](https://arxiv.org/pdf/2503.00706)  

**Abstract**: Due to their energy-efficient flight characteristics, fixed-wing type UAVs are useful robotic tools for long-range and duration flight applications in large-scale environments. However, flying fixed-wing UAV in confined environments, such as mountainous regions, can be challenging due to their limited maneuverability and sensitivity to uncertain wind conditions. In this work, we first analyze periodic trochoidal paths that can be used to define wind-aware terminal loitering states. We then propose a wind-invariant safe set of trochoidal paths along with a switching strategy for selecting the corresponding minimum-extent periodic path type. Finally, we show that planning with this minimum-extent set allows us to safely reach up to 10 times more locations in mountainous terrain compared to planning with a single, conservative loitering maneuver. 

**Abstract (ZH)**: 由于其能效高的飞行特性，固定翼型无人机是大型环境中长距离和长时间飞行应用有用的 robotic 工具。然而，在山地等受限环境中飞行固定翼型无人机因其有限的机动性和对不确定风况的敏感性而具有挑战性。在此工作中，我们首先分析可用于定义风感知终端盘旋状态的周期性螺旋线路径。然后，我们提出一种风不变的安全周期性螺旋线路径集合以及相应的最小范围周期性路径类型的切换策略。最后，我们证明使用此最小范围集合进行规划能够在山地地形中安全地到达常规盘旋操作可达位置的多达10倍数量的位置。 

---
# Learning Perceptive Humanoid Locomotion over Challenging Terrain 

**Title (ZH)**: 学习穿越挑战性地形的感知 humanoid 运动 

**Authors**: Wandong Sun, Baoshi Cao, Long Chen, Yongbo Su, Yang Liu, Zongwu Xie, Hong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00692)  

**Abstract**: Humanoid robots are engineered to navigate terrains akin to those encountered by humans, which necessitates human-like locomotion and perceptual abilities. Currently, the most reliable controllers for humanoid motion rely exclusively on proprioception, a reliance that becomes both dangerous and unreliable when coping with rugged terrain. Although the integration of height maps into perception can enable proactive gait planning, robust utilization of this information remains a significant challenge, especially when exteroceptive perception is noisy. To surmount these challenges, we propose a solution based on a teacher-student distillation framework. In this paradigm, an oracle policy accesses noise-free data to establish an optimal reference policy, while the student policy not only imitates the teacher's actions but also simultaneously trains a world model with a variational information bottleneck for sensor denoising and state estimation. Extensive evaluations demonstrate that our approach markedly enhances performance in scenarios characterized by unreliable terrain estimations. Moreover, we conducted rigorous testing in both challenging urban settings and off-road environments, the model successfully traverse 2 km of varied terrain without external intervention. 

**Abstract (ZH)**: 类人机器人被设计为能够在类似人类遇到的地形中导航，这需要类似人类的运动能力和感知能力。目前，最可靠的类人运动控制器依赖于本体感受，但在应对崎岖地形时，这种依赖变得既危险又不可靠。虽然将高度图整合到感知中可以实现主动步态规划，但如何稳健地利用这些信息仍然是一个重大挑战，尤其是在外部感知噪声较大的情况下。为了克服这些挑战，我们提出了一种基于教师-学生蒸馏框架的解决方案。在这个框架中，先验策略访问无噪声数据以建立最优参考策略，而学生策略不仅模仿教师的动作，还同时通过变分信息瓶颈训练世界模型进行传感器去噪和状态估计。广泛的评估表明，我们的方法在地形不可靠的情况下显著提高了性能。此外，我们在具有挑战性的城市环境和非道路环境中进行了严格的测试，模型成功地在没有外部干预的情况下穿越了2公里的复杂地形。 

---
# One-Shot Gesture Recognition for Underwater Diver-To-Robot Communication 

**Title (ZH)**: 一次性手势识别以实现水下 diver-to-机器人通信 

**Authors**: Rishikesh Joshi, Junaed Sattar  

**Link**: [PDF](https://arxiv.org/pdf/2503.00676)  

**Abstract**: Reliable human-robot communication is essential for underwater human-robot interaction (U-HRI), yet traditional methods such as acoustic signaling and predefined gesture-based models suffer from limitations in adaptability and robustness. In this work, we propose One-Shot Gesture Recognition (OSG), a novel method that enables real-time, pose-based, temporal gesture recognition underwater from a single demonstration, eliminating the need for extensive dataset collection or model retraining. OSG leverages shape-based classification techniques, including Hu moments, Zernike moments, and Fourier descriptors, to robustly recognize gestures in visually-challenging underwater environments. Our system achieves high accuracy on real-world underwater data and operates efficiently on embedded hardware commonly found on autonomous underwater vehicles (AUVs), demonstrating its feasibility for deployment on-board robots. Compared to deep learning approaches, OSG is lightweight, computationally efficient, and highly adaptable, making it ideal for diver-to-robot communication. We evaluate OSG's performance on an augmented gesture dataset and real-world underwater video data, comparing its accuracy against deep learning methods. Our results show OSG's potential to enhance U-HRI by enabling the immediate deployment of user-defined gestures without the constraints of predefined gesture languages. 

**Abstract (ZH)**: 可靠的 underwater 人类-机器人通信对于水下人类-机器人交互（U-HRI）至关重要，然而传统的声学信号和预定义手势模型方法在适应性和鲁棒性方面存在局限性。在本文中，我们提出了一次性手势识别（OSG），这是一种新型方法，能够在单次示范下实现水下基于姿态的时间手势识别，无需大量数据集收集或模型重新训练。OSG 利用基于形状分类的技术，包括 Hu 矩、Zernike 矩和傅里叶描述符，以在视觉挑战性的水下环境中稳健地识别手势。我们的系统在实际水下数据上实现了高精度，并在自主水下车辆（AUV）常见的嵌入式硬件上高效运行，展示了其在水下机器人上部署的可行性。与深度学习方法相比，OSG 是轻量级的、计算效率高且高度适应性强，使其成为潜水员到机器人通信的理想选择。我们使用增强手势数据集和真实水下视频数据评估了 OSG 的性能，并将其准确性与深度学习方法进行了比较。我们的结果表明，OSG 有可能通过使用户定义的手势能够立即部署而增强 U-HRI，从而摆脱预定义手势语言的限制。 

---
# Autonomous Dissection in Robotic Cholecystectomy 

**Title (ZH)**: 自主机器人胆囊切除术中的自动解剖分离 

**Authors**: Ki-Hwan Oh, Leonardo Borgioli, Miloš Žefran, Valentina Valle, Pier Cristoforo Giulianotti  

**Link**: [PDF](https://arxiv.org/pdf/2503.00666)  

**Abstract**: Robotic surgery offers enhanced precision and adaptability, paving the way for automation in surgical interventions. Cholecystectomy, the gallbladder removal, is particularly well-suited for automation due to its standardized procedural steps and distinct anatomical boundaries. A key challenge in automating this procedure is dissecting with accuracy and adaptability. This paper presents a vision-based autonomous robotic dissection architecture that integrates real-time segmentation, keypoint detection, grasping and stretching the gallbladder with the left arm, and dissecting with the other. We introduce an improved segmentation dataset based on videos of robotic cholecystectomy performed by various surgeons, incorporating a new ``liver bed'' class to enhance boundary tracking after multiple rounds of dissection. Our system employs state-of-the-art segmentation models and an adaptive boundary extraction method that maintains accuracy despite tissue deformations and visual variations. Moreover, we implemented an automated grasping and pulling strategy to optimize tissue tension before dissection upon our previous work. Ex vivo evaluations on porcine livers demonstrate that our framework significantly improves dissection precision and consistency, marking a step toward fully autonomous robotic cholecystectomy. 

**Abstract (ZH)**: 基于视觉的自主机器人解剖架构：用于自动化胆囊切除术的精确和适应性解剖 

---
# CAP: A Connectivity-Aware Hierarchical Coverage Path Planning Algorithm for Unknown Environments using Coverage Guidance Graph 

**Title (ZH)**: CAP：一种基于连通性的分层覆盖路径规划算法用于未知环境的覆盖指导图方法 

**Authors**: Zongyuan Shen, Burhanuddin Shirose, Prasanna Sriganesh, Matthew Travers  

**Link**: [PDF](https://arxiv.org/pdf/2503.00647)  

**Abstract**: Efficient coverage of unknown environments requires robots to adapt their paths in real time based on on-board sensor data. In this paper, we introduce CAP, a connectivity-aware hierarchical coverage path planning algorithm for efficient coverage of unknown environments. During online operation, CAP incrementally constructs a coverage guidance graph to capture essential information about the environment. Based on the updated graph, the hierarchical planner determines an efficient path to maximize global coverage efficiency and minimize local coverage time. The performance of CAP is evaluated and compared with five baseline algorithms through high-fidelity simulations as well as robot experiments. Our results show that CAP yields significant improvements in coverage time, path length, and path overlap ratio. 

**Abstract (ZH)**: 面向连通性的分层覆盖路径规划算法CAP：未知环境的有效覆盖 

---
# Safety-Critical Control for Robotic Manipulators using Collision Cone Control Barrier Functions 

**Title (ZH)**: 基于碰撞圆锥控制Barrier函数的安全关键控制方法用于机器人 manipulator 控制 

**Authors**: Lucas Almeida  

**Link**: [PDF](https://arxiv.org/pdf/2503.00623)  

**Abstract**: This paper presents a comprehensive approach for the safety-critical control of robotic manipulators operating in dynamic environments. Building upon the framework of Control Barrier Functions (CBFs), we extend the collision cone methodology to formulate Collision Cone Control Barrier Functions (C3BFs) specifically tailored for manipulators. In our approach, safety constraints derived from collision cone geometry are seamlessly integrated with Cartesian impedance control to ensure compliant yet safe end-effector behavior. A Quadratic Program (QP)-based controller is developed to minimally modify the nominal control input to enforce safety. Extensive simulation experiments demonstrate the efficacy of the proposed method in various dynamic scenarios. 

**Abstract (ZH)**: 本文提出了一种全面的方法，用于在动态环境中操作的机器人 manipulator 的安全关键控制。基于控制屏障函数（CBFs）的框架，我们扩展了碰撞锥方法，提出了专门针对 manipulator 的碰撞锥控制屏障函数（C3BFs）。在我们的方法中，从碰撞锥几何中导出的安全约束与笛卡尔阻抗控制无缝集成，以确保末端执行器的行为既柔顺又安全。我们开发了一种基于二次规划（QP）的控制器，以最小程度地修改名义控制输入来强制执行安全性。广泛的动力学仿真实验证明了所提方法在各种动态场景中的有效性。 

---
# Sampling-Based Motion Planning with Discrete Configuration-Space Symmetries 

**Title (ZH)**: 基于采样的运动规划与离散配置空间对称性 

**Authors**: Thomas Cohn, Russ Tedrake  

**Link**: [PDF](https://arxiv.org/pdf/2503.00614)  

**Abstract**: When planning motions in a configuration space that has underlying symmetries (e.g. when manipulating one or multiple symmetric objects), the ideal planning algorithm should take advantage of those symmetries to produce shorter trajectories. However, finite symmetries lead to complicated changes to the underlying topology of configuration space, preventing the use of standard algorithms. We demonstrate how the key primitives used for sampling-based planning can be efficiently implemented in spaces with finite symmetries. A rigorous theoretical analysis, building upon a study of the geometry of the configuration space, shows improvements in the sample complexity of several standard algorithms. Furthermore, a comprehensive slate of experiments demonstrates the practical improvements in both path length and runtime. 

**Abstract (ZH)**: 在具有有限对称性的配置空间中高效实施基于采样的运动规划基本原语及其理论分析与实验验证 

---
# ATMO: An Aerially Transforming Morphobot for Dynamic Ground-Aerial Transition 

**Title (ZH)**: ATMO: 一种用于动态地面-空中转换的空中形态变换morphobot 

**Authors**: Ioannis Mandralis, Reza Nemovi, Alireza Ramezani, Richard M. Murray, Morteza Gharib  

**Link**: [PDF](https://arxiv.org/pdf/2503.00609)  

**Abstract**: Designing ground-aerial robots is challenging due to the increased actuation requirements which can lead to added weight and reduced locomotion efficiency. Morphobots mitigate this by combining actuators into multi-functional groups and leveraging ground transformation to achieve different locomotion modes. However, transforming on the ground requires dealing with the complexity of ground-vehicle interactions during morphing, limiting applicability on rough terrain. Mid-air transformation offers a solution to this issue but demands operating near or beyond actuator limits while managing complex aerodynamic forces. We address this problem by introducing the Aerially Transforming Morphobot (ATMO), a robot which transforms near the ground achieving smooth transition between aerial and ground modes. To achieve this, we leverage the near ground aerodynamics, uncovered by experimental load cell testing, and stabilize the system using a model-predictive controller that adapts to ground proximity and body shape. The system is validated through numerous experimental demonstrations. We find that ATMO can land smoothly at body postures past its actuator saturation limits by virtue of the uncovered ground-effect. 

**Abstract (ZH)**: 基于空中变形的变形地面-空中机器人设计克服了因增加的驱动需求而导致的重量增加和移动效率降低的问题。通过将驱动器组合成多功能组并利用地面变形来实现不同的移动模式，形态机器人减轻了这一问题。然而，地面变形受到地面-车辆交互复杂性的限制，限制了其在崎岖地形上的应用。空中变形提供了解决方案，但需要在或接近驱动器极限条件下操作并管理复杂的气动力量。我们通过引入空中变形的形态机器人（ATMO），一种在近地面变形以实现空中和地面模式平滑过渡的机器人，解决了这一问题。我们利用实验荷重测试发现的近地面气动学，并利用适应地面接近度和机体形状的模型预测控制器来稳定系统。该系统通过多次实验演示进行了验证。我们发现，ATMO得益于发现的地效，可以在超过了驱动器饱和极限的机体姿态下平稳着陆。 

---
# Dynamic Collision Avoidance Using VelocityObstacle-based Control Barrier Functions 

**Title (ZH)**: 基于速度障碍的控制障碍函数的动态碰撞 avoidance 控制 

**Authors**: Jihao Huang, Jun Zeng, Xuemin Chi, Koushil Sreenath, Zhitao Liu, Hongye Su  

**Link**: [PDF](https://arxiv.org/pdf/2503.00606)  

**Abstract**: Designing safety-critical controllers for acceleration-controlled unicycle robots is challenging, as control inputs may not appear in the constraints of control Lyapunov functions(CLFs) and control barrier functions (CBFs), leading to invalid controllers. Existing methods often rely on state-feedback-based CLFs and high-order CBFs (HOCBFs), which are computationally expensive to construct and fail to maintain effectiveness in dynamic environments with fast-moving, nearby obstacles. To address these challenges, we propose constructing velocity obstacle-based CBFs (VOCBFs) in the velocity space to enhance dynamic collision avoidance capabilities, instead of relying on distance-based CBFs that require the introduction of HOCBFs. Additionally, by extending VOCBFs using variants of VO, we enable reactive collision avoidance between robots. We formulate a safety-critical controller for acceleration-controlled unicycle robots as a mixed-integer quadratic programming (MIQP), integrating state-feedback-based CLFs for navigation and VOCBFs for collision avoidance. To enhance the efficiency of solving the MIQP, we split the MIQP into multiple sub-optimization problems and employ a decision network to reduce computational costs. Numerical simulations demonstrate that our approach effectively guides the robot to its target while avoiding collisions. Compared to HOCBFs, VOCBFs exhibit significantly improved dynamic obstacle avoidance performance, especially when obstacles are fast-moving and close to the robot. Furthermore, we extend our method to distributed multi-robot systems. 

**Abstract (ZH)**: 基于速度障碍的碰撞避免控制设计：加速控制单轮机器人安全关键控制器的设计挑战 

---
# Space-Time Graphs of Convex Sets for Multi-Robot Motion Planning 

**Title (ZH)**: 凸集的时空图在多机器人运动规划中的应用 

**Authors**: Jingtao Tang, Zining Mao, Lufan Yang, Hang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.00583)  

**Abstract**: We address the Multi-Robot Motion Planning (MRMP) problem of computing collision-free trajectories for multiple robots in shared continuous environments. While existing frameworks effectively decompose MRMP into single-robot subproblems, spatiotemporal motion planning with dynamic obstacles remains challenging, particularly in cluttered or narrow-corridor settings. We propose Space-Time Graphs of Convex Sets (ST-GCS), a novel planner that systematically covers the collision-free space-time domain with convex sets instead of relying on random sampling. By extending Graphs of Convex Sets (GCS) into the time dimension, ST-GCS formulates time-optimal trajectories in a unified convex optimization that naturally accommodates velocity bounds and flexible arrival times. We also propose Exact Convex Decomposition (ECD) to "reserve" trajectories as spatiotemporal obstacles, maintaining a collision-free space-time graph of convex sets for subsequent planning. Integrated into two prioritized-planning frameworks, ST-GCS consistently achieves higher success rates and better solution quality than state-of-the-art sampling-based planners -- often at orders-of-magnitude faster runtimes -- underscoring its benefits for MRMP in challenging settings. 

**Abstract (ZH)**: 多机器人时空路径规划： convex集时空图方法 

---
# Actor-Critic Cooperative Compensation to Model Predictive Control for Off-Road Autonomous Vehicles Under Unknown Dynamics 

**Title (ZH)**: 基于未知动力学的离-road自主车辆模型预测控制的Actor-Critic协同补偿方法 

**Authors**: Prakhar Gupta, Jonathon M Smereka, Yunyi Jia  

**Link**: [PDF](https://arxiv.org/pdf/2503.00577)  

**Abstract**: This study presents an Actor-Critic Cooperative Compensated Model Predictive Controller (AC3MPC) designed to address unknown system dynamics. To avoid the difficulty of modeling highly complex dynamics and ensuring realtime control feasibility and performance, this work uses deep reinforcement learning with a model predictive controller in a cooperative framework to handle unknown dynamics. The model-based controller takes on the primary role as both controllers are provided with predictive information about the other. This improves tracking performance and retention of inherent robustness of the model predictive controller. We evaluate this framework for off-road autonomous driving on unknown deformable terrains that represent sandy deformable soil, sandy and rocky soil, and cohesive clay-like deformable soil. Our findings demonstrate that our controller statistically outperforms standalone model-based and learning-based controllers by upto 29.2% and 10.2%. This framework generalized well over varied and previously unseen terrain characteristics to track longitudinal reference speeds with lower errors. Furthermore, this required significantly less training data compared to purely learning-based controller, while delivering better performance even when under-trained. 

**Abstract (ZH)**: 基于Actor-Critic合作补偿模型预测控制的未知系统动力学处理方法研究 

---
# Enhancing Context-Aware Human Motion Prediction for Efficient Robot Handovers 

**Title (ZH)**: 增强上下文意识的人体运动预测以实现高效的机器人交接 

**Authors**: Gerard Gómez-Izquierdo, Javier Laplaza, Alberto Sanfeliu, Anaís Garrell  

**Link**: [PDF](https://arxiv.org/pdf/2503.00576)  

**Abstract**: Accurate human motion prediction (HMP) is critical for seamless human-robot collaboration, particularly in handover tasks that require real-time adaptability. Despite the high accuracy of state-of-the-art models, their computational complexity limits practical deployment in real-world robotic applications. In this work, we enhance human motion forecasting for handover tasks by leveraging siMLPe [1], a lightweight yet powerful architecture, and introducing key improvements. Our approach, named IntentMotion incorporates intention-aware conditioning, task-specific loss functions, and a novel intention classifier, significantly improving motion prediction accuracy while maintaining efficiency. Experimental results demonstrate that our method reduces body loss error by over 50%, achieves 200x faster inference, and requires only 3% of the parameters compared to existing state-of-the-art HMP models. These advancements establish our framework as a highly efficient and scalable solution for real-time human-robot interaction. 

**Abstract (ZH)**: 准确的人体运动预测（HMP）对于无缝的人机协作至关重要，特别是在需要实时适应性的交接任务中。尽管最先进的模型具有较高的准确性，但其计算复杂性限制了其实用部署在现实机器人应用中的应用。在本文中，我们通过利用轻量但强大的siMLPe架构并引入关键改进，增强了交接任务中的人体运动预测。我们的方法名为IntentMotion， Incorporates意图感知的条件输入、任务特定的损失函数以及一种新型的意图分类器，显著提高了运动预测准确性的同时保持了高效性。实验结果表明，我们的方法将身体损耗错误降低了超过50%，推理速度提高了200倍，并且只需要现有最先进的HMP模型3%的参数。这些进步确立了我们框架作为实时人机交互的高效且可扩展解决方案的地位。 

---
# Dexterous Three-Finger Gripper based on Offset Trimmed Helicoids (OTHs) 

**Title (ZH)**: 基于偏移修剪螺旋面（OTHs）的灵巧三指 gripper 

**Authors**: Qinghua Guan, Hung Hon Cheng, Josie Hughes  

**Link**: [PDF](https://arxiv.org/pdf/2503.00574)  

**Abstract**: This study presents an innovative offset-trimmed helicoids (OTH) structure, featuring a tunable deformation center that emulates the flexibility of human fingers. This design significantly reduces the actuation force needed for larger elastic deformations, particularly when dealing with harder materials like thermoplastic polyurethane (TPU). The incorporation of two helically routed tendons within the finger enables both in-plane bending and lateral out-of-plane transitions, effectively expanding its workspace and allowing for variable curvature along its length. Compliance analysis indicates that the compliance at the fingertip can be fine-tuned by adjusting the mounting placement of the fingers. This customization enhances the gripper's adaptability to a diverse range of objects. By leveraging TPU's substantial elastic energy storage capacity, the gripper is capable of dynamically rotating objects at high speeds, achieving approximately 60 in just 15 milliseconds. The three-finger gripper, with its high dexterity across six degrees of freedom, has demonstrated the capability to successfully perform intricate tasks. One such example is the adept spinning of a rod within the gripper's grasp. 

**Abstract (ZH)**: 一种具有可调变形中心的创新偏移修剪螺旋体（OTH）结构及其应用研究 

---
# PL-VIWO: A Lightweight and Robust Point-Line Monocular Visual Inertial Wheel Odometry 

**Title (ZH)**: PL-VIWO: 一种轻量级稳健的点线单目视觉惯性轮 odometer 

**Authors**: Zhixin Zhang, Wenzhi Bai, Liang Zhao, Pawel Ladosz  

**Link**: [PDF](https://arxiv.org/pdf/2503.00551)  

**Abstract**: This paper presents a novel tightly coupled Filter-based monocular visual-inertial-wheel odometry (VIWO) system for ground robots, designed to deliver accurate and robust localization in long-term complex outdoor navigation scenarios. As an external sensor, the camera enhances localization performance by introducing visual constraints. However, obtaining a sufficient number of effective visual features is often challenging, particularly in dynamic or low-texture environments. To address this issue, we incorporate the line features for additional geometric constraints. Unlike traditional approaches that treat point and line features independently, our method exploits the geometric relationships between points and lines in 2D images, enabling fast and robust line matching and triangulation. Additionally, we introduce Motion Consistency Check (MCC) to filter out potential dynamic points, ensuring the effectiveness of point feature updates. The proposed system was evaluated on publicly available datasets and benchmarked against state-of-the-art methods. Experimental results demonstrate superior performance in terms of accuracy, robustness, and efficiency. The source code is publicly available at: this https URL 

**Abstract (ZH)**: 本文提出了一种新颖的紧密耦合滤波器为基础的单目视觉-惯性-陀螺仪里程计（VIWO）系统，旨在为地面机器人在长期复杂户外导航场景中提供准确可靠的定位。作为外部传感器，摄像机通过引入视觉约束来提升定位性能。然而，在动态或低纹理环境中获得足够的有效视觉特征往往是具有挑战性的。为解决这一问题，我们引入了线特征以提供额外的几何约束。与传统方法将点特征和线特征独立处理不同，我们的方法利用二维图像中点和线之间的几何关系，实现快速稳健的线匹配和三角化。此外，我们引入了运动一致性检查（MCC）来滤除潜在的动态点，确保点特征更新的有效性。所提出系统在公开可用的数据集上进行了评估，并与最先进的方法进行了基准测试。实验结果在准确性、稳健性和效率方面展现了优越性能。源代码可在以下链接获取：this https URL 

---
# Vehicle Top Tag Assisted Vehicle-Road Cooperative Localization For Autonomous Public Buses 

**Title (ZH)**: 基于车辆顶置标签的自动驾驶公交车辆-道路协同定位 

**Authors**: Hao Li, Yifei Sun, Bo Liu, Linbin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00546)  

**Abstract**: Accurate vehicle localization is indispensable to autonomous vehicles, but is difficult to realize in complicated application scenarios. Intersection scenarios that suffer from environmental shielding and crowded dynamic objects are especially crucial and challenging. To handle difficult intersection scenarios, the methodology of vehicle top tag assisted vehicle-road cooperative localization or for short vehicle top tag assisted localization is proposed. The proposed methodology has merits of satisfying all the feasibility, reliability, explainability, society and economy concerns. Concrete solutions of vehicle top tag detection and vehicle top tag localization that instantiate the core part of the proposed methodology are presented. Simulation results are provided to demonstrate effectiveness of the presented solutions. The proposed methodology of vehicle top tag assisted localization also has the potential to be extended to a much wider range of practical applications than our intended ones involving autonomous public buses. 

**Abstract (ZH)**: 精确的车辆定位对于自动驾驶车辆至关重要，但在复杂的应用场景中难以实现。特别是在受环境遮挡和拥挤动态对象影响的交叉场景中，这一问题尤为关键和具有挑战性。为应对复杂的交叉场景，提出了一种车辆顶部标签辅助车辆-道路协同定位的方法，简称车辆顶部标签辅助定位。该方法在可行性和可靠性、可解释性以及社会和经济影响方面具有优势。详细介绍了车辆顶部标签检测和车辆顶部标签定位的具体解决方案，以实例化该方法的核心部分。提供的仿真结果证明了所提解决方案的有效性。该方法还具有扩展到更广泛的实用应用场景的潜力，而不仅仅是我们预期的应用于自动驾驶公交车的场景。 

---
# BodyGen: Advancing Towards Efficient Embodiment Co-Design 

**Title (ZH)**: BodyGen: 向高效身躯联合设计迈进 

**Authors**: Haofei Lu, Zhe Wu, Junliang Xing, Jianshu Li, Ruoyu Li, Zhe Li, Yuanchun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2503.00533)  

**Abstract**: Embodiment co-design aims to optimize a robot's morphology and control policy simultaneously. While prior work has demonstrated its potential for generating environment-adaptive robots, this field still faces persistent challenges in optimization efficiency due to the (i) combinatorial nature of morphological search spaces and (ii) intricate dependencies between morphology and control. We prove that the ineffective morphology representation and unbalanced reward signals between the design and control stages are key obstacles to efficiency. To advance towards efficient embodiment co-design, we propose BodyGen, which utilizes (1) topology-aware self-attention for both design and control, enabling efficient morphology representation with lightweight model sizes; (2) a temporal credit assignment mechanism that ensures balanced reward signals for optimization. With our findings, Body achieves an average 60.03% performance improvement against state-of-the-art baselines. We provide codes and more results on the website: this https URL. 

**Abstract (ZH)**: 基于体态的协同设计旨在同时优化机器人形态和控制策略。尽管先前的工作已经展示了其生成环境适应型机器人的潜力，但由于形态搜索空间的组合性质以及形态与控制之间的复杂依赖关系，该领域仍然面临优化效率的持续挑战。我们证明了无效的形态表示以及设计阶段和控制阶段之间不均衡的奖励信号是效率问题的关键障碍。为了实现高效的基于体态的协同设计，我们提出了BodyGen，该方法利用（1）拓扑感知自注意力机制，以轻量级模型大小实现高效形态表示；（2）时序归因机制以确保优化过程中的奖励信号平衡。基于我们的研究发现，Body在与最新基线方法相比时，平均提高了60.03%的性能。我们在网站上提供了代码和更多结果：this https URL。 

---
# Never too Prim to Swim: An LLM-Enhanced RL-based Adaptive S-Surface Controller for AUVs under Extreme Sea Conditions 

**Title (ZH)**: 永不过时的泳趣：一种增强学习模型辅助的适应性S-表面控制器用于极端海况下的自治 underwater 机器人 

**Authors**: Guanwen Xie, Jingzehua Xu, Yimian Ding, Zhi Zhang, Shuai Zhang, Yi Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00527)  

**Abstract**: The adaptivity and maneuvering capabilities of Autonomous Underwater Vehicles (AUVs) have drawn significant attention in oceanic research, due to the unpredictable disturbances and strong coupling among the AUV's degrees of freedom. In this paper, we developed large language model (LLM)-enhanced reinforcement learning (RL)-based adaptive S-surface controller for AUVs. Specifically, LLMs are introduced for the joint optimization of controller parameters and reward functions in RL training. Using multi-modal and structured explicit task feedback, LLMs enable joint adjustments, balance multiple objectives, and enhance task-oriented performance and adaptability. In the proposed controller, the RL policy focuses on upper-level tasks, outputting task-oriented high-level commands that the S-surface controller then converts into control signals, ensuring cancellation of nonlinear effects and unpredictable external disturbances in extreme sea conditions. Under extreme sea conditions involving complex terrain, waves, and currents, the proposed controller demonstrates superior performance and adaptability in high-level tasks such as underwater target tracking and data collection, outperforming traditional PID and SMC controllers. 

**Abstract (ZH)**: 自主 underwater 车辆 (AUVs) 的适应性和机动能力由于海洋环境中的不可预测干扰和自由度间的强耦合而受到广泛关注。本文提出了一种大型语言模型 (LLM) 增强的强化学习 (RL) 基准自适应 S-表面控制器。具体而言，LLM 用于在 RL 训练过程中联合优化控制器参数和奖励函数。利用多模态和结构化的显式任务反馈，LLM 使参数联合调整、平衡多个目标，并提高任务导向的性能和适应性。在所提出的控制器中，RL 策略关注高层任务，输出任务导向的高层命令，S-表面控制器将其转换为控制信号，以确保在极端海况下消除非线性影响和不可预测的外部干扰。在涉及复杂地形、波浪和流速的极端海况下，所提出的控制器在水下目标跟踪和数据采集等高层任务中表现出色，优于传统的 PID 和 SMC 控制器。 

---
# CAFEs: Cable-driven Collaborative Floating End-Effectors for Agriculture Applications 

**Title (ZH)**: CAFEs: 电缆驱动协作漂浮末端执行器在农业应用中的研究 

**Authors**: Hung Hon Cheng, Josie Hughes  

**Link**: [PDF](https://arxiv.org/pdf/2503.00514)  

**Abstract**: CAFEs (Collaborative Agricultural Floating End-effectors) is a new robot design and control approach to automating large-scale agricultural tasks. Based upon a cable driven robot architecture, by sharing the same roller-driven cable set with modular robotic arms, a fast-switching clamping mechanism allows each CAFE to clamp onto or release from the moving cables, enabling both independent and synchronized movement across the workspace. The methods developed to enable this system include the mechanical design, precise position control and a dynamic model for the spring-mass liked system, ensuring accurate and stable movement of the robotic arms. The system's scalability is further explored by studying the tension and sag in the cables to maintain performance as more robotic arms are deployed. Experimental and simulation results demonstrate the system's effectiveness in tasks including pick-and-place showing its potential to contribute to agricultural automation. 

**Abstract (ZH)**: CAFEs（协作农业浮动末端执行器）是一种用于自动化大规模农业任务的新机器人设计与控制方法。基于绳索驱动机器人架构，通过共享滚子驱动绳索集和模块化机械臂，快速切换夹持机制使每个CAFE能够夹持或释放移动绳索，从而实现在工作空间内的独立和同步运动。为了实现这一系统，开发了一种机械设计、精确位置控制和类似于弹簧-质量系统的动力学模型，确保机器人臂的准确和稳定运动。通过对绳索的张力和下垂进行研究，进一步探讨了系统的可扩展性，以保持性能随着更多机器人臂的部署而稳定。实验和仿真结果表明，该系统在拾取和放置等任务中的有效性，展示了其在农业自动化方面的潜力。 

---
# HGDiffuser: Efficient Task-Oriented Grasp Generation via Human-Guided Grasp Diffusion Models 

**Title (ZH)**: HGDiffuser：高效的任务导向抓取生成通过人类引导的抓取扩散模型 

**Authors**: Dehao Huang, Wenlong Dong, Chao Tang, Hong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00508)  

**Abstract**: Task-oriented grasping (TOG) is essential for robots to perform manipulation tasks, requiring grasps that are both stable and compliant with task-specific constraints. Humans naturally grasp objects in a task-oriented manner to facilitate subsequent manipulation tasks. By leveraging human grasp demonstrations, current methods can generate high-quality robotic parallel-jaw task-oriented grasps for diverse objects and tasks. However, they still encounter challenges in maintaining grasp stability and sampling efficiency. These methods typically rely on a two-stage process: first performing exhaustive task-agnostic grasp sampling in the 6-DoF space, then applying demonstration-induced constraints (e.g., contact regions and wrist orientations) to filter candidates. This leads to inefficiency and potential failure due to the vast sampling space. To address this, we propose the Human-guided Grasp Diffuser (HGDiffuser), a diffusion-based framework that integrates these constraints into a guided sampling process. Through this approach, HGDiffuser directly generates 6-DoF task-oriented grasps in a single stage, eliminating exhaustive task-agnostic sampling. Furthermore, by incorporating Diffusion Transformer (DiT) blocks as the feature backbone, HGDiffuser improves grasp generation quality compared to MLP-based methods. Experimental results demonstrate that our approach significantly improves the efficiency of task-oriented grasp generation, enabling more effective transfer of human grasping strategies to robotic systems. To access the source code and supplementary videos, visit this https URL. 

**Abstract (ZH)**: 面向任务的抓取（TOG）是机器人执行操作任务的关键，要求抓取既稳定又符合特定任务约束。人类在执行后续操作任务时会自然地以任务为导向抓取物体。借助人类抓取示例，当前方法可以生成高质量的机器人平行夹爪面向任务的抓取，适用于多种物体和任务。然而，这些方法在保持抓取稳定性和采样效率方面仍然面临挑战。这些方法通常依赖于两阶段过程：首先在6-DOF空间进行 exhaustive 任务无关的抓取采样，然后应用由示例诱导的约束（如接触区域和手腕方向）来筛选候选抓取。这导致了采样的低效率和潜在的失败。为了解决这一问题，我们提出了由人类指导的抓取扩散器（HGDiffuser），这是一种基于扩散的方法，将这些约束整合到引导采样过程之中。借助此方法，HGDiffuser可以直接在单阶段生成6-DOF面向任务的抓取，去掉了 exhaustive 任务无关的采样。此外，通过将扩散转换器（DiT）块作为特征骨干，HGDiffuser 提高了抓取生成质量，优于基于MLP的方法。实验结果表明，我们的方法显著提高了面向任务的抓取生成效率，使得将人类抓取策略更有效地转移到机器人系统中成为可能。要访问源代码和补充视频，访问此链接：https://this-url 

---
# Interact, Instruct to Improve: A LLM-Driven Parallel Actor-Reasoner Framework for Enhancing Autonomous Vehicle Interactions 

**Title (ZH)**: 交互，指令以提高：一个由LLM驱动的并行行动-推理框架，用于增强自动驾驶车辆交互 

**Authors**: Shiyu Fang, Jiaqi Liu, Chengkai Xu, Chen Lv, Peng Hang, Jian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.00502)  

**Abstract**: Autonomous Vehicles (AVs) have entered the commercialization stage, but their limited ability to interact and express intentions still poses challenges in interactions with Human-driven Vehicles (HVs). Recent advances in large language models (LLMs) enable bidirectional human-machine communication, but the conflict between slow inference speed and the need for real-time decision-making challenges practical deployment. To address these issues, this paper introduces a parallel Actor-Reasoner framework designed to enable explicit bidirectional AV-HV interactions across multiple scenarios. First, by facilitating interactions between the LLM-driven Reasoner and heterogeneous simulated HVs during training, an interaction memory database, referred to as the Actor, is established. Then, by introducing the memory partition module and the two-layer memory retrieval module, the Actor's ability to handle heterogeneous HVs is significantly enhanced. Ablation studies and comparisons with other decision-making methods demonstrate that the proposed Actor-Reasoner framework significantly improves safety and efficiency. Finally, with the combination of the external Human-Machine Interface (eHMI) information derived from Reasoner's reasoning and the feasible action solutions retrieved from the Actor, the effectiveness of the proposed Actor-Reasoner is confirmed in multi-scenario field interactions. Our code is available at this https URL. 

**Abstract (ZH)**: 自主驾驶车辆（AVs）已进入商业化阶段，但其有限的交互和意图表达能力仍给与人类驾驶车辆（HVs）的交互带来挑战。大型语言模型（LLMs）的 Recent 进展使双向人机通信成为可能，但推理速度缓慢与实时决策需求之间的冲突挑战其实际部署。为解决这些问题，本文引入一种并行 Actor-Reasoner 框架，旨在实现多种场景下AV-HV的明确双向交互。首先，在训练过程中通过LLM驱动的Reasoner与异构模拟HV的交互建立一个交互记忆数据库，称为Actor。然后，通过引入记忆分区模块和两层记忆检索模块，显著增强了Actor处理异构HV的能力。消融研究和与其他决策方法的对比表明，提出的钱Actor-Reasoner框架显著提高了安全性和效率。最后，结合来自Reasoner推理的外部人机接口（eHMI）信息和从Actor检索到的可行动作解决方案，在多种场景下的实地交互中验证了所提出的钱Actor-Reasoner的有效性。代码可在以下链接获取。 

---
# Flying on Point Clouds with Reinforcement Learning 

**Title (ZH)**: 基于强化学习在点云上的飞行 

**Authors**: Guangtong Xu, Tianyue Wu, Zihan Wang, Qianhao Wang, Fei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.00496)  

**Abstract**: A long-cherished vision of drones is to autonomously traverse through clutter to reach every corner of the world using onboard sensing and computation. In this paper, we combine onboard 3D lidar sensing and sim-to-real reinforcement learning (RL) to enable autonomous flight in cluttered environments. Compared to vision sensors, lidars appear to be more straightforward and accurate for geometric modeling of surroundings, which is one of the most important cues for successful obstacle avoidance. On the other hand, sim-to-real RL approach facilitates the realization of low-latency control, without the hierarchy of trajectory generation and tracking. We demonstrate that, with design choices of practical significance, we can effectively combine the advantages of 3D lidar sensing and RL to control a quadrotor through a low-level control interface at 50Hz. The key to successfully learn the policy in a lightweight way lies in a specialized surrogate of the lidar's raw point clouds, which simplifies learning while retaining a fine-grained perception to detect narrow free space and thin obstacles. Simulation statistics demonstrate the advantages of the proposed system over alternatives, such as performing easier maneuvers and higher success rates at different speed constraints. With lightweight simulation techniques, the policy trained in the simulator can control a physical quadrotor, where the system can dodge thin obstacles and safely traverse randomly distributed obstacles. 

**Abstract (ZH)**: 无人机自主穿越障碍实现全球无死角飞行的长久愿景是利用机载3D激光雷达感知与计算能力。本文结合机载3D激光雷达感知和仿真到现实的强化学习（RL）方法，实现复杂环境下的自主飞行。相比于视觉传感器，激光雷达在构建周围环境的几何模型方面显得更为简单和准确，这是成功避障的重要线索。另一方面，仿真到现实的RL方法使得低延迟控制得以实现，省去了轨迹生成和跟踪的层次结构。我们展示了通过实际有意义的设计选择，可以有效地结合3D激光雷达感知和RL的优势，通过50Hz的低级别控制接口控制四旋翼无人机。轻量化学习策略的关键在于专门设计的激光雷达原始点云的代理模型，该模型简化了学习过程，同时保持了对狭窄自由空间和薄障碍物的精细感知。仿真统计结果表明，所提出的系统在执行更简单的操作和在不同速度约束下更高的成功率方面优于替代方案。通过轻量级仿真技术，模拟器中训练的策略可以控制物理四旋翼无人机，系统可以避开薄障碍物并安全穿越随机分布的障碍物。 

---
# A Navigation System for ROV's inspection on Fish Net Cage 

**Title (ZH)**: ROV导航系统用于鱼网笼检查 

**Authors**: Zhikang Ge, Fang Yang, Wenwu Lu, Peng Wei, Yibin Ying, Chen Peng  

**Link**: [PDF](https://arxiv.org/pdf/2503.00482)  

**Abstract**: Autonomous Remotely Operated Vehicles (ROVs) offer a promising solution for automating fishnet inspection, reducing labor dependency, and improving operational efficiency. In this paper, we modify an off-the-shelf ROV, the BlueROV2, into a ROS-based framework and develop a localization module, a path planning system, and a control framework. For real-time, local localization, we employ the open-source TagSLAM library. Additionally, we propose a control strategy based on a Nominal Feedback Controller (NFC) to achieve precise trajectory tracking. The proposed system has been implemented and validated through experiments in a controlled laboratory environment, demonstrating its effectiveness for real-world applications. 

**Abstract (ZH)**: 自主远程操作车辆（ROVs）为自动化渔网检查提供了有前景的解决方案，减少对人力的依赖并提升操作效率。本文将商用BlueROV2改造为基于ROS的框架，并开发了定位模块、路径规划系统和控制框架。为实现实时局部定位，我们采用了开源的TagSLAM库。此外，我们提出了一种基于名义反馈控制器（NFC）的控制策略以实现精确轨迹跟踪。所提出的系统已在受控实验室环境中通过实验实施并验证，证明其适用于实际应用。 

---
# Model-based optimisation for the personalisation of robot-assisted gait training 

**Title (ZH)**: 基于模型的优化方法用于机器人辅助步态训练的个性化调整 

**Authors**: Andreas Christou, Daniel F. N. Gordon, Theodoros Stouraitis, Juan C. Moreno, Sethu Vijayakumar  

**Link**: [PDF](https://arxiv.org/pdf/2503.00480)  

**Abstract**: Personalised rehabilitation can be key to promoting gait independence and quality of life. Robots can enhance therapy by systematically delivering support in gait training, but often use one-size-fits-all control methods, which can be suboptimal. Here, we describe a model-based optimisation method for designing and fine-tuning personalised robotic controllers. As a case study, we formulate the objective of providing assistance as needed as an optimisation problem, and we demonstrate how musculoskeletal modelling can be used to develop personalised interventions. Eighteen healthy participants (age = 26 +/- 4) were recruited and the personalised control parameters for each were obtained to provide assistance as needed during a unilateral tracking task. A comparison was carried out between the personalised controller and the non-personalised controller. In simulation, a significant improvement was predicted when the personalised parameters were used. Experimentally, responses varied: six subjects showed significant improvements with the personalised parameters, eight subjects showed no obvious change, while four subjects performed worse. High interpersonal and intra-personal variability was observed with both controllers. This study highlights the importance of personalised control in robot-assisted gait training, and the need for a better estimation of human-robot interaction and human behaviour to realise the benefits of model-based optimisation. 

**Abstract (ZH)**: 基于模型的个性化康复机器人控制器优化方法研究：助力需求为导向的下肢功能训练与生活品质提升 

---
# Bring Your Own Grasp Generator: Leveraging Robot Grasp Generation for Prosthetic Grasping 

**Title (ZH)**: 自带抓取生成器：利用机器人抓取生成技术进行假肢抓取 

**Authors**: Giuseppe Stracquadanio, Federico Vasile, Elisa Maiettini, Nicolò Boccardo, Lorenzo Natale  

**Link**: [PDF](https://arxiv.org/pdf/2503.00466)  

**Abstract**: One of the most important research challenges in upper-limb prosthetics is enhancing the user-prosthesis communication to closely resemble the experience of a natural limb. As prosthetic devices become more complex, users often struggle to control the additional degrees of freedom. In this context, leveraging shared-autonomy principles can significantly improve the usability of these systems. In this paper, we present a novel eye-in-hand prosthetic grasping system that follows these principles. Our system initiates the approach-to-grasp action based on user's command and automatically configures the DoFs of a prosthetic hand. First, it reconstructs the 3D geometry of the target object without the need of a depth camera. Then, it tracks the hand motion during the approach-to-grasp action and finally selects a candidate grasp configuration according to user's intentions. We deploy our system on the Hannes prosthetic hand and test it on able-bodied subjects and amputees to validate its effectiveness. We compare it with a multi-DoF prosthetic control baseline and find that our method enables faster grasps, while simplifying the user experience. Code and demo videos are available online at this https URL. 

**Abstract (ZH)**: 上肢假肢领域的一项重要研究挑战是增强用户与假肢之间的沟通，使其更接近自然肢体的体验。随着假肢设备变得更加复杂，用户往往难以控制额外的自由度。在此背景下，利用共享自治原则可以显著提高这些系统的易用性。本文介绍了一种遵循这些原则的新型眼手协调假肢抓取系统。该系统根据用户的指令启动接近抓取动作，并自动配置假肢手的自由度。首先，它无需深度相机即可重建目标物体的3D几何形状。然后，在接近抓取过程中跟踪手部运动，最后根据用户的意图选择候选抓取配置。我们在Hannes假肢手上部署该系统，并对健全受试者和截肢者进行测试，以验证其有效性。我们将该方法与一个多自由度假肢控制基线进行比较，发现我们的方法能够实现更快的抓取，同时简化用户的体验。代码和演示视频可在以下链接在线获得。 

---
# Floorplan-SLAM: A Real-Time, High-Accuracy, and Long-Term Multi-Session Point-Plane SLAM for Efficient Floorplan Reconstruction 

**Title (ZH)**: 地板平面-SLAM：一种高效地板平面重构的实时、高精度和长时间多会话点-平面SLAM 

**Authors**: Haolin Wang, Zeren Lv, Hao Wei, Haijiang Zhu, Yihong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00397)  

**Abstract**: Floorplan reconstruction provides structural priors essential for reliable indoor robot navigation and high-level scene understanding. However, existing approaches either require time-consuming offline processing with a complete map, or rely on expensive sensors and substantial computational resources. To address the problems, we propose Floorplan-SLAM, which incorporates floorplan reconstruction tightly into a multi-session SLAM system by seamlessly interacting with plane extraction, pose estimation, and back-end optimization, achieving real-time, high-accuracy, and long-term floorplan reconstruction using only a stereo camera. Specifically, we present a robust plane extraction algorithm that operates in a compact plane parameter space and leverages spatially complementary features to accurately detect planar structures, even in weakly textured scenes. Furthermore, we propose a floorplan reconstruction module tightly coupled with the SLAM system, which uses continuously optimized plane landmarks and poses to formulate and solve a novel optimization problem, thereby enabling real-time incremental floorplan reconstruction. Note that by leveraging the map merging capability of multi-session SLAM, our method supports long-term floorplan reconstruction across multiple sessions without redundant data collection. Experiments on the VECtor and the self-collected datasets indicate that Floorplan-SLAM significantly outperforms state-of-the-art methods in terms of plane extraction robustness, pose estimation accuracy, and floorplan reconstruction fidelity and speed, achieving real-time performance at 25-45 FPS without GPU acceleration, which reduces the floorplan reconstruction time for a 1000 square meters scene from over 10 hours to just 9.44 minutes. 

**Abstract (ZH)**: Floorplan-SLAM：结合多时段SLAM系统的紧凑平面参数空间中稳健的平面提取与实时高精度长期楼层平面重建 

---
# Scalable Real2Sim: Physics-Aware Asset Generation Via Robotic Pick-and-Place Setups 

**Title (ZH)**: 可扩展的Real2Sim：基于机器人拾放设置的物理感知资产生成 

**Authors**: Nicholas Pfaff, Evelyn Fu, Jeremy Binagia, Phillip Isola, Russ Tedrake  

**Link**: [PDF](https://arxiv.org/pdf/2503.00370)  

**Abstract**: Simulating object dynamics from real-world perception shows great promise for digital twins and robotic manipulation but often demands labor-intensive measurements and expertise. We present a fully automated Real2Sim pipeline that generates simulation-ready assets for real-world objects through robotic interaction. Using only a robot's joint torque sensors and an external camera, the pipeline identifies visual geometry, collision geometry, and physical properties such as inertial parameters. Our approach introduces a general method for extracting high-quality, object-centric meshes from photometric reconstruction techniques (e.g., NeRF, Gaussian Splatting) by employing alpha-transparent training while explicitly distinguishing foreground occlusions from background subtraction. We validate the full pipeline through extensive experiments, demonstrating its effectiveness across diverse objects. By eliminating the need for manual intervention or environment modifications, our pipeline can be integrated directly into existing pick-and-place setups, enabling scalable and efficient dataset creation. 

**Abstract (ZH)**: 从现实世界感知模拟物体动力学在数字孪生和机器人操作中显示出巨大前景，但往往需要大量的劳动密集型测量和专业知识。我们提出了一种完全自动化的Real2Sim管道，通过机器人交互生成可用于现实世界物体的仿真资产。仅使用机器人关节扭矩传感器和外部摄像头，该管道识别视觉几何、碰撞几何以及惯性参数等物理属性。我们的方法通过采用α透明训练，结合前景遮挡与背景减法明确区分，引入了一种从光度重构技术（例如，NeRF、Gaussian Splatting）中高效提取高质量、以物体为中心的网格的通用方法。我们通过广泛的实验验证了整个管道的有效性，展示了其在多种物体上的适用性。通过消除人工干预或环境修改的需要，该管道可以直接集成到现有的抓取和放置设置中，实现可扩展且高效的数据集创建。 

---
# Legged Robot State Estimation Using Invariant Neural-Augmented Kalman Filter with a Neural Compensator 

**Title (ZH)**: 基于神经补偿器的不变神经增强卡尔曼滤波的腿足机器人状态估计 

**Authors**: Seokju Lee, Hyun-Bin Kim, Kyung-Soo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.00344)  

**Abstract**: This paper presents an algorithm to improve state estimation for legged robots. Among existing model-based state estimation methods for legged robots, the contact-aided invariant extended Kalman filter defines the state on a Lie group to preserve invariance, thereby significantly accelerating convergence. It achieves more accurate state estimation by leveraging contact information as measurements for the update step. However, when the model exhibits strong nonlinearity, the estimation accuracy decreases. Such nonlinearities can cause initial errors to accumulate and lead to large drifts over time. To address this issue, we propose compensating for errors by augmenting the Kalman filter with an artificial neural network serving as a nonlinear function approximator. Furthermore, we design this neural network to respect the Lie group structure to ensure invariance, resulting in our proposed Invariant Neural-Augmented Kalman Filter (InNKF). The proposed algorithm offers improved state estimation performance by combining the strengths of model-based and learning-based approaches. Supplementary Video: this https URL 

**Abstract (ZH)**: 基于接触辅助的李群不变广义Kalman滤波器改进腿式机器人的状态估计算法 

---
# Feasible Force Set Shaping for a Payload-Carrying Platform Consisting of Tiltable Multiple UAVs Connected Via Passive Hinge Joints 

**Title (ZH)**: 基于被动铰链关节连接的可倾斜多旋翼载荷平台的可行力分布形成功能研究 

**Authors**: Takumi Ito, Hayato Kawashima, Riku Funada, Mitsuji Sampei  

**Link**: [PDF](https://arxiv.org/pdf/2503.00341)  

**Abstract**: This paper presents a method for shaping the feasible force set of a payload-carrying platform composed of multiple Unmanned Aerial Vehicles (UAVs) and proposes a control law that leverages the advantages of this shaped force set. The UAVs are connected to the payload through passively rotatable hinge joints. The joint angles are controlled by the differential thrust produced by the rotors, while the total force generated by all the rotors is responsible for controlling the payload. The shape of the set of the total force depends on the tilt angles of the UAVs, which allows us to shape the feasible force set by adjusting these tilt angles. This paper aims to ensure that the feasible force set encompasses the required shape, enabling the platform to generate force redundantly -meaning in various directions. We then propose a control law that takes advantage of this redundancy. 

**Abstract (ZH)**: 基于多无人机平台负载承载的力集整形方法及其控制策略 

---
# Fast Visuomotor Policies via Partial Denoising 

**Title (ZH)**: 部分去噪实现快速视动策略 

**Authors**: Haojun Chen, Minghao Liu, Xiaojian Ma, Zailin Ma, Huimin Wu, Chengdong Ma, Yuanpei Chen, Yifan Zhong, Mingzhi Wang, Qing Li, Yaodong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00339)  

**Abstract**: Diffusion policies are widely adopted in complex visuomotor tasks for their ability to capture multimodal action distributions. However, the multiple sampling steps required for action generation significantly harm real-time inference efficiency, which limits their applicability in long-horizon tasks and real-time decision-making scenarios. Existing acceleration techniques reduce sampling steps by approximating the original denoising process but inevitably introduce unacceptable performance loss. Here we propose Falcon, which mitigates this trade-off and achieves further acceleration. The core insight is that visuomotor tasks exhibit sequential dependencies between actions at consecutive time steps. Falcon leverages this property to avoid denoising from a standard normal distribution at each decision step. Instead, it starts denoising from partial denoised actions derived from historical information to significantly reduce the denoising steps while incorporating current observations to achieve performance-preserving acceleration of action generation. Importantly, Falcon is a training-free algorithm that can be applied as a plug-in to further improve decision efficiency on top of existing acceleration techniques. We validated Falcon in 46 simulated environments, demonstrating a 2-7x speedup with negligible performance degradation, offering a promising direction for efficient visuomotor policy design. 

**Abstract (ZH)**: Falcon：通过利用动作序列依赖性实现高效的视觉-运动策略设计 

---
# Peek into the `White-Box': A Field Study on Bystander Engagement with Urban Robot Uncertainty 

**Title (ZH)**: 窥视“白盒”：一项关于路人参与城市机器人不确定性交互的实地研究 

**Authors**: Xinyan Yu, Marius Hoggenmueller, Tram Thi Minh Tran, Yiyuan Wang, Qiuming Zhang, Martin Tomitsch  

**Link**: [PDF](https://arxiv.org/pdf/2503.00337)  

**Abstract**: Uncertainty inherently exists in the autonomous decision-making process of robots. Involving humans in resolving this uncertainty not only helps robots mitigate it but is also crucial for improving human-robot interactions. However, in public urban spaces filled with unpredictability, robots often face heightened uncertainty without direct human collaborators. This study investigates how robots can engage bystanders for assistance in public spaces when encountering uncertainty and examines how these interactions impact bystanders' perceptions and attitudes towards robots. We designed and tested a speculative `peephole' concept that engages bystanders in resolving urban robot uncertainty. Our design is guided by considerations of non-intrusiveness and eliciting initiative in an implicit manner, considering bystanders' unique role as non-obligated participants in relation to urban robots. Drawing from field study findings, we highlight the potential of involving bystanders to mitigate urban robots' technological imperfections to both address operational challenges and foster public acceptance of urban robots. Furthermore, we offer design implications to encourage bystanders' involvement in mitigating the imperfections. 

**Abstract (ZH)**: 不确定性固存在机器人自主决策过程中。将人类纳入解决这一不确定性不仅有助于机器人减轻这种不确定性，也是改善人机交互的关键。然而，在充满不可预测性的公共城市空间中，机器人在缺乏直接人类合作者的情况下往往面临更高的不确定性。本研究探讨了机器人在公共空间遇到不确定性时如何寻求旁观者协助，并考察了这些互动如何影响旁观者对机器人的感知和态度。我们设计并测试了一个 speculative 的“窥视孔”概念，该概念旨在让旁观者参与到解决城市机器人不确定性中。我们的设计考虑了非侵入性和以隐式方式激发主动性的因素，考虑到旁观者在与城市机器人关系中作为非强制性参与者所扮演的独特角色。基于实地研究的结果，我们强调了让旁观者参与解决城市机器人技术缺陷的潜力，以应对运营挑战并促进公众对城市机器人的接受度。此外，我们提供了设计建议，以鼓励旁观者参与解决这些缺陷。 

---
# XIRVIO: Critic-guided Iterative Refinement for Visual-Inertial Odometry with Explainable Adaptive Weighting 

**Title (ZH)**: XIRVIO：具有可解释自适应加权的批评指导迭代增强视觉-惯性测地术 

**Authors**: Chit Yuen Lam, Ronald Clark, Basaran Bahadir Kocer  

**Link**: [PDF](https://arxiv.org/pdf/2503.00315)  

**Abstract**: We introduce XIRVIO, a transformer-based Generative Adversarial Network (GAN) framework for monocular visual inertial odometry (VIO). By taking sequences of images and 6-DoF inertial measurements as inputs, XIRVIO's generator predicts pose trajectories through an iterative refinement process which are then evaluated by the critic to select the iteration with the optimised prediction. Additionally, the self-emergent adaptive sensor weighting reveals how XIRVIO attends to each sensory input based on contextual cues in the data, making it a promising approach for achieving explainability in safety-critical VIO applications. Evaluations on the KITTI dataset demonstrate that XIRVIO matches well-known state-of-the-art learning-based methods in terms of both translation and rotation errors. 

**Abstract (ZH)**: XIRVIO：一种基于变换器的生成对抗网络框架，用于单目视觉惯性里程计 

---
# A Practical Sensing Interface for Exoskeleton Evaluation in Workplaces using Interface Forces 

**Title (ZH)**: 基于接口力的工作场所外骨骼评估实用传感接口 

**Authors**: Joshua Leong Wei Ren, Thomas M. Kwok  

**Link**: [PDF](https://arxiv.org/pdf/2503.00293)  

**Abstract**: This paper presents a novel approach to evaluating back support exoskeletons (BSEs) in workplace settings addressing the limitations of traditional methods like electromyography (EMG), which are impractical due to their sensitivity to external disturbances and user sweat. Variability in BSE performance among users, often due to joint misalignment and anthropomorphic differences, can lead to discomfort and reduced effectiveness. To overcome these challenges, we propose integrating a compact load cell into the exoskeleton's thigh cuff. This small load cell provides precise force measurements without significantly altering the exoskeleton's kinematics or inertia, enabling real-time assessment of exoskeleton assistance in both laboratory and workplace environments, Experimental validation during load-lifting tasks demonstrated that the load cell effectively captures interface forces between the BSE and human subjects, showing stronger correlations with the user's muscle activity when the BSE provides effective assistance. This innovative sensing interface offers a stable, practical alternative to EMG and respiratory gas measurements, facilitating more accurate and convenient evaluation of BSE performance in real-world industrial and laboratory settings. The proposed method holds promise for enhancing the adoption and effectiveness of BSEs by providing reliable, real-time feedback on their assistance capabilities. 

**Abstract (ZH)**: 本文提出了一种评估工作场所背支撑外骨骼（BSE）的新方法，以克服传统方法如电肌图（EMG）的局限性，这些方法因对外部干扰和用户汗液的敏感性而不切实际。用户之间的BSE性能差异，通常是由于关节对齐不良和体型差异引起的，可能导致不适并降低有效性。为克服这些挑战，我们提出将小型载荷细胞集成到外骨骼的大腿护带上。该小型载荷细胞提供了精确的力测量，而不显著改变外骨骼的运动学或惯性，从而能够在实验室和工作场所环境中实时评估外骨骼的辅助效果。在负载提升任务中的实验验证表明，载荷细胞有效地捕捉了BSE与人类受试者之间的界面力，当BSE提供有效辅助时，这种力测量与用户的肌肉活动显示出更强的相关性。该创新传感界面提供了一种稳定且实用的替代EMG和呼吸气体测量的方法，促进了更准确和便捷的BSE性能评估，特别是在实际工业和实验室环境中。所提出的方法有潜力通过提供关于外骨骼辅助能力的可靠实时反馈来增强BSE的采用和有效性。 

---
# Towards Passive Safe Reinforcement Learning: A Comparative Study on Contact-rich Robotic Manipulation 

**Title (ZH)**: 面向被动安全强化学习：接触丰富的机器人操纵比较研究 

**Authors**: Heng Zhang, Gokhan Solak, Sebastian Hjorth, Arash Ajoudani  

**Link**: [PDF](https://arxiv.org/pdf/2503.00287)  

**Abstract**: Reinforcement learning (RL) has achieved remarkable success in various robotic tasks; however, its deployment in real-world scenarios, particularly in contact-rich environments, often overlooks critical safety and stability aspects. Policies without passivity guarantees can result in system instability, posing risks to robots, their environments, and human operators. In this work, we investigate the limitations of traditional RL policies when deployed in contact-rich tasks and explore the combination of energy-based passive control with safe RL in both training and deployment to answer these challenges. Firstly, we introduce energy-based constraints in our safe RL formulation to train \textit{passivity-aware} RL agents. Secondly, we add a passivity filter on the agent output for \textit{passivity-ensured} control during deployment. We conduct comparative studies on a contact-rich robotic maze exploration task, evaluating the effects of learning passivity-aware policies and the importance of passivity-ensured control. The experiments demonstrate that a passivity-agnostic RL policy easily violates energy constraints in deployment, even though it achieves high task completion in training. The results show that our proposed approach guarantees control stability through passivity filtering and improves the energy efficiency through passivity-aware training. A video of real-world experiments is available as supplementary material. We also release the checkpoint model and offline data for pre-training at \href{this https URL}{Hugging Face} 

**Abstract (ZH)**: 强化学习（RL）在各种机器人任务中取得了显著成功；然而，在实际应用场景中，特别是在接触密集环境中，其部署往往忽视了关键的安全与稳定性方面。没有保证被动性的策略可能导致系统不稳定，对机器人、其环境和人类操作员构成风险。在本工作中，我们探讨了传统RL策略在接触密集任务中的局限性，并探索了在训练和部署中将基于能量的被动控制与安全RL相结合的方法，以应对这些挑战。首先，我们在安全RL框架中引入能量约束以训练具备被动性的RL代理。其次，我们在部署时为代理输出添加被动性滤波器以确保被动性。我们在接触密集的机器人迷宫探索任务中进行对比研究，评估学习被动性意识策略和被动性确保控制的重要性。实验结果表明，不具备被动性的RL策略在部署时容易违反能量约束，尽管其在训练中能够实现高任务完成率。结果表明，我们提出的方法通过被动性滤波确保了控制的稳定性，并通过被动性意识训练提高了能源效率。附有现实世界实验的视频作为补充材料。我们也公布了检查点模型和离线数据以供预训练，在Hugging Face（此 https URL）可下载。 

---
# Human-Robot Collaboration: A Non-Verbal Approach with the NAO Humanoid Robot 

**Title (ZH)**: 人类与机器人协作：基于NAO人形机器人的非言语方法 

**Authors**: Maaz Qureshi, Kerstin Dautenhahn  

**Link**: [PDF](https://arxiv.org/pdf/2503.00284)  

**Abstract**: Humanoid robots, particularly NAO, are gaining prominence for their potential to revolutionize human-robot collaboration, especially in domestic settings like kitchens. Leveraging the advantages of NAO, this research explores non-verbal communications role in enhancing human-robot interaction during meal preparation tasks. By employing gestures, body movements, and visual cues, NAO provides feedback to users, improving comprehension and safety. Our study investigates user perceptions of NAO feedback and its anthropomorphic attributes. Findings suggest that combining various non-verbal cues enhances communication effectiveness, although achieving full anthropomorphic likeness remains a challenge. Insights from this research inform the design of future robotic systems for improved human-robot collaboration. 

**Abstract (ZH)**: 类人机器人，尤其是NAO，正因其在革命人类与机器人协作方面（特别是在厨房等家庭环境中）的潜力而日益受到关注。利用NAO的优势，本研究探讨了非言语沟通在提升烹饪任务中的人机交互中的作用。通过使用手势、身体动作和视觉提示，NAO向用户提供反馈，从而提高理解和安全性。本研究考察了用户对NAO反馈及其类人属性的感知。研究发现，结合多种非言语提示可以增强沟通效果，尽管达到完全类人形态仍面临挑战。本研究的见解为设计未来的人机协作机器人系统提供了指导。 

---
# Xpress: A System For Dynamic, Context-Aware Robot Facial Expressions using Language Models 

**Title (ZH)**: Xpress：一种基于语言模型的动态、上下文感知机器人面部表情系统 

**Authors**: Victor Nikhil Antony, Maia Stiber, Chien-Ming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00283)  

**Abstract**: Facial expressions are vital in human communication and significantly influence outcomes in human-robot interaction (HRI), such as likeability, trust, and companionship. However, current methods for generating robotic facial expressions are often labor-intensive, lack adaptability across contexts and platforms, and have limited expressive ranges--leading to repetitive behaviors that reduce interaction quality, particularly in long-term scenarios. We introduce Xpress, a system that leverages language models (LMs) to dynamically generate context-aware facial expressions for robots through a three-phase process: encoding temporal flow, conditioning expressions on context, and generating facial expression code. We demonstrated Xpress as a proof-of-concept through two user studies (n=15x2) and a case study with children and parents (n=13), in storytelling and conversational scenarios to assess the system's context-awareness, expressiveness, and dynamism. Results demonstrate Xpress's ability to dynamically produce expressive and contextually appropriate facial expressions, highlighting its versatility and potential in HRI applications. 

**Abstract (ZH)**: 面部表情在人类沟通中至关重要，并且显著影响人机交互（HRI）的结果，如好感度、信任度和陪伴感。然而，当前生成机器人面部表情的方法往往劳动密集、缺乏跨情境和平台的适应性，并且表情表达范围有限——导致重复行为，降低了交互质量，尤其是在长期场景中。我们介绍了Xpress系统，该系统利用语言模型（LMs）通过三阶段过程动态生成情境相关的面部表情：编码时间流动、依据情境条件面部表情、生成面部表情编码。我们通过两个用户研究（n=15x2）和一个关于儿童和家长的案例研究（n=13）展示了Xpress系统在故事讲述和对话场景中的情境意识、表达能力和动态性。研究结果表明，Xpress能够动态生成恰当且富有表现力的面部表情，突显其在HRI应用中的 versatility 和潜力。 

---
# Maintaining Plasticity in Reinforcement Learning: A Cost-Aware Framework for Aerial Robot Control in Non-stationary Environments 

**Title (ZH)**: 在非稳定环境中基于成本意识的无人机控制：强化学习中的塑性维持框架 

**Authors**: Ali Tahir Karasahin, Ziniu Wu, Basaran Bahadir Kocer  

**Link**: [PDF](https://arxiv.org/pdf/2503.00282)  

**Abstract**: Reinforcement learning (RL) has demonstrated the ability to maintain the plasticity of the policy throughout short-term training in aerial robot control. However, these policies have been shown to loss of plasticity when extended to long-term learning in non-stationary environments. For example, the standard proximal policy optimization (PPO) policy is observed to collapse in long-term training settings and lead to significant control performance degradation. To address this problem, this work proposes a cost-aware framework that uses a retrospective cost mechanism (RECOM) to balance rewards and losses in RL training with a non-stationary environment. Using a cost gradient relation between rewards and losses, our framework dynamically updates the learning rate to actively train the control policy in a disturbed wind environment. Our experimental results show that our framework learned a policy for the hovering task without policy collapse in variable wind conditions and has a successful result of 11.29% less dormant units than L2 regularization with PPO. 

**Abstract (ZH)**: 强化学习在空中机器人控制中的短期训练中展示了保持策略弹性的能力，但在非stationary环境中进行长期学习时，这些策略显示出弹性的损失。例如，标准的近端策略优化（PPO）策略在长期训练环境中观察到崩溃并导致控制性能显著下降。为了解决这一问题，本文提出了一种成本感知框架，该框架使用回顾性成本机制（RECOM）在非stationary环境中平衡RL训练中的奖励和损失。通过奖励和损失之间的成本梯度关系，我们的框架动态更新学习率，以在受扰动风环境中有活性地训练控制策略。实验结果表明，我们的框架在变风条件下学习悬停任务的策略，未出现策略崩溃，并且与PPO和L2正则化相比，活跃单元比例减少了11.29%。 

---
# CRADMap: Applied Distributed Volumetric Mapping with 5G-Connected Multi-Robots and 4D Radar Sensing 

**Title (ZH)**: CRADMap: 应用5G连接多机器人与4D雷达感测的分布式体积映射技术 

**Authors**: Maaz Qureshi, Alexander Werner, Zhenan Liu, Amir Khajepour, George Shaker, William Melek  

**Link**: [PDF](https://arxiv.org/pdf/2503.00262)  

**Abstract**: Sparse and feature SLAM methods provide robust camera pose estimation. However, they often fail to capture the level of detail required for inspection and scene awareness tasks. Conversely, dense SLAM approaches generate richer scene reconstructions but impose a prohibitive computational load to create 3D maps. We present a novel distributed volumetric mapping framework designated as CRADMap that addresses these issues by extending the state-of-the-art (SOTA) ORBSLAM3 [1] system with the COVINS [2] on the backend for global optimization. Our pipeline for volumetric reconstruction fuses dense keyframes at a centralized server via 5G connectivity, aggregating geometry, and occupancy information from multiple autonomous mobile robots (AMRs) without overtaxing onboard resources. This enables each AMR to independently perform mapping while the backend constructs high-fidelity 3D maps in real time. To overcome the limitation of standard visual nodes we automate a 4D mmWave radar, standalone from CRADMap, to test its capabilities for making extra maps of the hidden metallic object(s) in a cluttered environment. Experimental results Section-IV confirm that our framework yields globally consistent volumetric reconstructions and seamlessly supports applied distributed mapping in complex indoor environments. 

**Abstract (ZH)**: 稀疏特征SLAM方法提供了稳健的相机姿态估计，但往往无法捕捉到检验和场景感知任务所需的细节水平。相反，密集SLAM方法生成更为丰富的场景重建，但会带来创建3D地图的计算负担。我们提出了一种名为CRADMap的新型分布式体积映射框架，通过将最先进的ORBSLAM3系统与COVINS全局优化模块扩展结合，解决了这些问题。我们的体积重建管道利用5G连接在中央服务器上融合密集关键帧，汇总来自多个自主移动机器人（AMRs）的几何和占用信息，而不会过度占用其车载资源。这使得每个AMR能够独立进行测绘，而后端可以实现实时构建高保真的3D地图。为了克服标准视觉节点的限制，我们自动化了一种独立于CRADMap的4D毫米波雷达，以测试其为杂乱环境中的隐藏金属物体生成额外地图的能力。实验结果（第四节）证实，我们的框架能够提供全局一致的体积重建，并能无缝支持复杂室内环境中的分布式映射。 

---
# Robotic Automation in Apparel Manufacturing: A Novel Approach to Fabric Handling and Sewing 

**Title (ZH)**: 服装制造中的机器人自动化：一种新颖的面料处理与缝制方法 

**Authors**: Abhiroop Ajith, Gokul Narayanan, Jonathan Zornow, Carlos Calle, Auralis Herrero Lugo, Jose Luis Susa Rincon, Chengtao Wen, Eugen Solowjow  

**Link**: [PDF](https://arxiv.org/pdf/2503.00249)  

**Abstract**: Sewing garments using robots has consistently posed a research challenge due to the inherent complexities in fabric manipulation. In this paper, we introduce an intelligent robotic automation system designed to address this issue. By employing a patented technique that temporarily stiffens garments, we eliminate the traditional necessity for fabric modeling. Our methodological approach is rooted in a meticulously designed three-stage pipeline: first, an accurate pose estimation of the cut fabric pieces; second, a procedure to temporarily join fabric pieces; and third, a closed-loop visual servoing technique for the sewing process. Demonstrating versatility across various fabric types, our approach has been successfully validated in practical settings, notably with cotton material at the Bluewater Defense production line and denim material at Levi's research facility. The techniques described in this paper integrate robotic mechanisms with traditional sewing machines, devising a real-time sewing algorithm, and providing hands-on validation through a collaborative robot setup. 

**Abstract (ZH)**: 使用机器人缝制 garments 一直由于面料操控的固有复杂性而构成研究挑战。本文介绍了一种智能机器人自动化系统，旨在解决这一问题。通过采用一项专利技术暂时使 garments 硬化，我们消除了传统上需要进行面料建模的必要性。我们的方法学基于精心设计的三阶段管道：首先，精确估算裁剪好的面料片的姿态；其次，一种将面料片临时拼接的程序；最后，一种闭环视觉伺服技术以指导缝制过程。我们的方法在不同类型的面料上展示了其灵活性，并已在Bluewater Defense 生产线的棉质材料和 Levi's 研究中心的牛仔布材料的实际应用中得到验证。本文中描述的技术将机器人机制与传统缝纫机相结合，设计了一种实时缝制算法，并通过协作机器人设置进行了实际验证。 

---
# Tendon-driven Grasper Design for Aerial Robot Perching on Tree Branches 

**Title (ZH)**: 基于树枝着陆的驱动绳索夹持器设计 

**Authors**: Haichuan Li, Ziang Zhao, Ziniu Wu, Parth Potdar, Long Tran, Ali Tahir Karasahin, Shane Windsor, Stephen G. Burrow, Basaran Bahadir Kocer  

**Link**: [PDF](https://arxiv.org/pdf/2503.00214)  

**Abstract**: Protecting and restoring forest ecosystems has become an important conservation issue. Although various robots have been used for field data collection to protect forest ecosystems, the complex terrain and dense canopy make the data collection less efficient. To address this challenge, an aerial platform with bio-inspired behaviour facilitated by a bio-inspired mechanism is proposed. The platform spends minimum energy during data collection by perching on tree branches. A raptor inspired vision algorithm is used to locate a tree trunk, and then a horizontal branch on which the platform can perch is identified. A tendon-driven mechanism inspired by bat claws which requires energy only for actuation, secures the platform onto the branch using the mechanism's passive compliance. Experimental results show that the mechanism can perform perching on branches ranging from 30 mm to 80 mm in diameter. The real-world tests validated the system's ability to select and adapt to target points, and it is expected to be useful in complex forest ecosystems. 

**Abstract (ZH)**: 受生物启发的航空平台在森林生态系统保护与恢复中的应用研究 

---
# SafeAuto: Knowledge-Enhanced Safe Autonomous Driving with Multimodal Foundation Models 

**Title (ZH)**: SafeAuto：基于多模态基础模型的安全自主驾驶知识增强方法 

**Authors**: Jiawei Zhang, Xuan Yang, Taiqi Wang, Yu Yao, Aleksandr Petiushko, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00211)  

**Abstract**: Traditional autonomous driving systems often struggle to integrate high-level reasoning with low-level control, resulting in suboptimal and sometimes unsafe driving behaviors. The emergence of Multimodal Large Language Models (MLLMs), which can process both visual and textual data, presents an opportunity to unify perception and reasoning tasks within a single framework. However, effectively embedding precise safety knowledge into MLLMs for autonomous driving remains a significant challenge. To address this, we propose SafeAuto, a novel framework that enhances MLLM-based autonomous driving systems by incorporating both unstructured and structured knowledge. Specifically, we first introduce the Position-Dependent Cross-Entropy (PDCE) loss function, designed to improve the accuracy of low-level control signal predictions when numerical values are represented as text. Second, to ensure safe autonomous driving by explicitly integrating precise safety knowledge into the MLLM, we develop a reasoning component for SafeAuto. This component translates driving safety regulations into first-order logic rules (e.g., "red light => stop") and incorporates these rules into a probabilistic graphical model, such as a Markov Logic Network (MLN). The MLN is trained to verify the predicted next actions using environmental attributes identified by attribute recognition models (e.g., detecting a red light) to form the predicates. Additionally, we construct a Multimodal RAG model that leverages video data, control signals, and environmental attributes to learn more effectively from past similar driving experiences. By integrating PDCE, MLN, and Multimodal RAG, SafeAuto significantly outperforms existing baselines across multiple datasets. This advancement enables more accurate, reliable, and safer autonomous driving systems that learn from experience, obey traffic laws, and perform precise control actions. 

**Abstract (ZH)**: 传统自动驾驶系统往往难以将高层次推理与低层次控制相结合，导致行为效果不佳甚至存在安全隐患。多模态大型语言模型（MLLM）能够处理视觉和文本数据的出现，为统一感知和推理任务提供了一种可能。然而，将精确的安全知识有效地嵌入到MLLM中以实现自动驾驶仍是一个重大挑战。为解决这一问题，我们提出了SafeAuto框架，该框架通过结合未结构化和结构化知识来增强基于MLLM的自动驾驶系统。具体而言，我们首先引入了位置依赖交叉熵（PDCE）损失函数，旨在提高当数值以文本形式表示时低层次控制信号预测的准确性。其次，为确保通过明确整合精确的安全知识实现安全的自动驾驶，我们为SafeAuto开发了一个推理组件。该组件将驾驶安全规定转化为一阶逻辑规则（例如，“红灯=>停止”），并将这些规则融入概率图模型，如马尔科夫逻辑网络（MLN）。MLN利用属性识别模型识别的环境属性（例如，检测红灯）来构建谓词，并通过这些属性验证预测的下一步动作。此外，我们构建了一个多模态RAG模型，该模型利用视频数据、控制信号和环境属性从过去的类似驾驶经历中更有效地学习。通过整合PDCE、MLN和多模态RAG，SafeAuto在多个数据集上显著优于现有基线。这一进展使得能够实现更准确、可靠和安全的自动驾驶系统，这些系统能够从经验中学习、遵守交通法规并执行精确的控制动作。 

---
# Survival of the fastest -- algorithm-guided evolution of light-powered underwater microrobots 

**Title (ZH)**: Survival of the最快者——算法引导下的水下微机器人光动力进化 

**Authors**: Mikołaj Rogóż, Zofia Dziekan, Piotr Wasylczyk  

**Link**: [PDF](https://arxiv.org/pdf/2503.00204)  

**Abstract**: Depending on environmental conditions, lightweight soft robots can exhibit various modes of locomotion that are difficult to model. As a result, optimizing their performance is complex, especially in small-scale systems characterized by low Reynolds numbers, when multiple aero- and hydrodynamical processes influence their movement. In this work, we study underwater swimmer locomotion by applying experimental results as the fitness function in two evolutionary algorithms: particle swarm optimization and genetic algorithm. Since soft, light-powered robots with different characteristics (phenotypes) can be fabricated quickly, they provide a great platform for optimisation experiments, using physical robots competing to improve swimming speed over consecutive generations. Interestingly, just like in natural evolution, unexpected gene combinations led to surprisingly good results, including several hundred percent increase in speed or the discovery of a self-oscillating underwater locomotion mode. 

**Abstract (ZH)**: 轻质软体机器人的水下游泳运动优化研究 

---
# Unified Video Action Model 

**Title (ZH)**: 统一视频动作模型 

**Authors**: Shuang Li, Yihuai Gao, Dorsa Sadigh, Shuran Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.00200)  

**Abstract**: A unified video and action model holds significant promise for robotics, where videos provide rich scene information for action prediction, and actions provide dynamics information for video prediction. However, effectively combining video generation and action prediction remains challenging, and current video generation-based methods struggle to match the performance of direct policy learning in action accuracy and inference this http URL bridge this gap, we introduce the Unified Video Action model (UVA), which jointly optimizes video and action predictions to achieve both high accuracy and efficient action inference. The key lies in learning a joint video-action latent representation and decoupling video-action decoding. The joint latent representation bridges the visual and action domains, effectively modeling the relationship between video and action sequences. Meanwhile, the decoupled decoding, powered by two lightweight diffusion heads, enables high-speed action inference by bypassing video generation during inference. Such a unified framework further enables versatile functionality through masked input training. By selectively masking actions or videos, a single model can tackle diverse tasks beyond policy learning, such as forward and inverse dynamics modeling and video generation. Via an extensive set of experiments, we demonstrate that UVA can serve as a general-purpose solution for a wide range of robotics tasks, such as policy learning, forward/inverse dynamics and video observation prediction, without compromising performance compared to methods tailored for specific applications. Results are best viewed on this https URL. 

**Abstract (ZH)**: 一种统一的视频和动作模型在机器人领域具有重要潜力，视频提供了丰富的场景信息以进行动作预测，而动作提供了视频预测的动力学信息。然而，有效地结合视频生成和动作预测仍然具有挑战性，当前基于视频生成的方法在动作准确性及推理速度上难以匹配合乎政策学习的方法。为了弥合这一差距，我们提出了统一视频动作模型（UVA），该模型同时优化视频和动作预测，以实现高准确性和高效的动作推理。关键在于学习联合视频-动作潜在表示并解耦视频-动作解码。联合潜在表示连接视觉和动作领域，有效地建模了视频和动作序列之间的关系。同时，由两个轻量级扩散头驱动的解耦解码能够在推理时不生成视频，从而实现高速动作推理。这种统一框架还通过蒙版输入训练增强了多功能性。通过选择性地蒙版动作或视频，单个模型可以解决超出策略学习的各种任务，如前向/逆向动力学建模和视频生成。通过一系列广泛实验，我们证明UVA可以用作解决一系列机器人任务的一般解决方案，例如策略学习、前向/逆向动力学建模和视频观测预测，且与专门针对特定应用的方法相比不牺牲性能。结果请参阅此链接：https://openaccess.thecvf.com/content/CVPR2023/papers/Peng_Unified_Video_Action_Models_for_Robotic_Task_Completion_CVPR_2023_paper.pdf。 

---
# ProDapt: Proprioceptive Adaptation using Long-term Memory Diffusion 

**Title (ZH)**: ProDapt: 使用长期记忆扩散的本体感知适应 

**Authors**: Federico Pizarro Bejarano, Bryson Jones, Daniel Pastor Moreno, Joseph Bowkett, Paul G. Backes, Angela P. Schoellig  

**Link**: [PDF](https://arxiv.org/pdf/2503.00193)  

**Abstract**: Diffusion models have revolutionized imitation learning, allowing robots to replicate complex behaviours. However, diffusion often relies on cameras and other exteroceptive sensors to observe the environment and lacks long-term memory. In space, military, and underwater applications, robots must be highly robust to failures in exteroceptive sensors, operating using only proprioceptive information. In this paper, we propose ProDapt, a method of incorporating long-term memory of previous contacts between the robot and the environment in the diffusion process, allowing it to complete tasks using only proprioceptive data. This is achieved by identifying "keypoints", essential past observations maintained as inputs to the policy. We test our approach using a UR10e robotic arm in both simulation and real experiments and demonstrate the necessity of this long-term memory for task completion. 

**Abstract (ZH)**: 基于长期记忆的扩散模型在仅使用本体感受信息完成任务中的应用 

---
# Learning Vision-Based Neural Network Controllers with Semi-Probabilistic Safety Guarantees 

**Title (ZH)**: 基于视觉的神经网络控制器的学习与半概率安全保证 

**Authors**: Xinhang Ma, Junlin Wu, Hussein Sibai, Yiannis Kantaros, Yevgeniy Vorobeychik  

**Link**: [PDF](https://arxiv.org/pdf/2503.00191)  

**Abstract**: Ensuring safety in autonomous systems with vision-based control remains a critical challenge due to the high dimensionality of image inputs and the fact that the relationship between true system state and its visual manifestation is unknown. Existing methods for learning-based control in such settings typically lack formal safety guarantees. To address this challenge, we introduce a novel semi-probabilistic verification framework that integrates reachability analysis with conditional generative adversarial networks and distribution-free tail bounds to enable efficient and scalable verification of vision-based neural network controllers. Next, we develop a gradient-based training approach that employs a novel safety loss function, safety-aware data-sampling strategy to efficiently select and store critical training examples, and curriculum learning, to efficiently synthesize safe controllers in the semi-probabilistic framework. Empirical evaluations in X-Plane 11 airplane landing simulation, CARLA-simulated autonomous lane following, and F1Tenth lane following in a physical visually-rich miniature environment demonstrate the effectiveness of our method in achieving formal safety guarantees while maintaining strong nominal performance. Our code is available at this https URL. 

**Abstract (ZH)**: 基于视觉控制的自主系统安全性保障仍是一个关键挑战，由于图像输入的高维度以及真实系统状态与其视觉表现之间的关系未知。在这种环境下基于学习的控制方法通常缺乏形式化的安全性保证。为应对这一挑战，我们提出了一种新颖的半概率验证框架，该框架将可达性分析与条件生成对抗网络和分布无约束尾部边界相结合，以实现高效的可扩展性视觉基于神经网络控制器的验证。接下来，我们开发了一种基于梯度的训练方法，该方法采用了一种新型的安全性损失函数、感知安全性数据采样策略以高效选择和存储关键训练示例，并采用渐进学习方法，以在半概率框架中高效合成安全控制器。在X-Plane 11飞机着陆仿真、CARLA模拟的自主车道跟随以及F1Tenth在具有丰富视觉的真实小型环境中的车道跟随中进行的经验评估表明，我们的方法能够在保持强标准性能的同时实现形式化安全性保证。我们的代码可在该网址获取。 

---
# A Magnetic-Actuated Vision-Based Whisker Array for Contact Perception and Grasping 

**Title (ZH)**: 磁驱动基于视觉的触须阵列用于接触感知与抓取 

**Authors**: Zhixian Hu, Juan Wachs, Yu She  

**Link**: [PDF](https://arxiv.org/pdf/2503.00133)  

**Abstract**: Tactile sensing and the manipulation of delicate objects are critical challenges in robotics. This study presents a vision-based magnetic-actuated whisker array sensor that integrates these functions. The sensor features eight whiskers arranged circularly, supported by an elastomer membrane and actuated by electromagnets and permanent magnets. A camera tracks whisker movements, enabling high-resolution tactile this http URL sensor's performance was evaluated through object classification and grasping experiments. In the classification experiment, the sensor approached objects from four directions and accurately identified five distinct objects with a classification accuracy of 99.17% using a Multi-Layer Perceptron model. In the grasping experiment, the sensor tested configurations of eight, four, and two whiskers, achieving the highest success rate of 87% with eight whiskers. These results highlight the sensor's potential for precise tactile sensing and reliable manipulation. 

**Abstract (ZH)**: 基于视觉的磁驱动触须阵列传感器：精细物体的触觉感知与操作的关键挑战及其解决方案 

---
# Navigating the Edge with the State-of-the-Art Insights into Corner Case Identification and Generation for Enhanced Autonomous Vehicle Safety 

**Title (ZH)**: 基于前沿见解的边缘案例识别与生成导航以提升自主车辆安全 

**Authors**: Gabriel Kenji Godoy Shimanuki, Alexandre Moreira Nascimento, Lucio Flavio Vismari, Joao Batista Camargo Junior, Jorge Rady de Almeida Junior, Paulo Sergio Cugnasca  

**Link**: [PDF](https://arxiv.org/pdf/2503.00077)  

**Abstract**: In recent years, there has been significant development of autonomous vehicle (AV) technologies. However, despite the notable achievements of some industry players, a strong and appealing body of evidence that demonstrate AVs are actually safe is lacky, which could foster public distrust in this technology and further compromise the entire development of this industry, as well as related social impacts. To improve the safety of AVs, several techniques are proposed that use synthetic data in virtual simulation. In particular, the highest risk data, known as corner cases (CCs), are the most valuable for developing and testing AV controls, as they can expose and improve the weaknesses of these autonomous systems. In this context, the present paper presents a systematic literature review aiming to comprehensively analyze methodologies for CC identifi cation and generation, also pointing out current gaps and further implications of synthetic data for AV safety and reliability. Based on a selection criteria, 110 studies were picked from an initial sample of 1673 papers. These selected paper were mapped into multiple categories to answer eight inter-linked research questions. It concludes with the recommendation of a more integrated approach focused on safe development among all stakeholders, with active collaboration between industry, academia and regulatory bodies. 

**Abstract (ZH)**: 近年来，自动驾驶（AV）技术取得了显著进展。然而，尽管一些行业参与者取得了显著成就，但仍缺乏强有力且令人信服的证据证明AV实际上安全，这可能会加剧公众对这项技术的不信任，并进一步损害该行业的整体发展及相关社会影响。为了提高AV的安全性，提出了一些使用合成数据在虚拟模拟中应用的技术。特别地，被称为边界案例（CCs）的最高风险数据最为宝贵，它们可用于开发和测试AV控制，暴露并改善这些自主系统的弱点。在此背景下，本文进行了一项系统文献综述，旨在全面分析CC识别与生成的方法学，指出当前的空白，并探讨合成数据对AV安全性和可靠性的进一步影响。基于选择标准，从初选的1673篇论文中筛选出110篇。这些选定的研究被映射到多个类别，以回答八个相互关联的研究问题。结论中建议采取更综合的方法，确保所有利益相关者之间的安全开发，并加强产业、学术界和监管机构之间的合作。 

---
# Stability Analysis of Deep Reinforcement Learning for Multi-Agent Inspection in a Terrestrial Testbed 

**Title (ZH)**: terrestrial测试床中多agent检查的深度强化学习稳定性分析 

**Authors**: Henry Lei, Zachary S. Lippay, Anonto Zaman, Joshua Aurand, Amin Maghareh, Sean Phillips  

**Link**: [PDF](https://arxiv.org/pdf/2503.00056)  

**Abstract**: The design and deployment of autonomous systems for space missions require robust solutions to navigate strict reliability constraints, extended operational duration, and communication challenges. This study evaluates the stability and performance of a hierarchical deep reinforcement learning (DRL) framework designed for multi-agent satellite inspection tasks. The proposed framework integrates a high-level guidance policy with a low-level motion controller, enabling scalable task allocation and efficient trajectory execution. Experiments conducted on the Local Intelligent Network of Collaborative Satellites (LINCS) testbed assess the framework's performance under varying levels of fidelity, from simulated environments to a cyber-physical testbed. Key metrics, including task completion rate, distance traveled, and fuel consumption, highlight the framework's robustness and adaptability despite real-world uncertainties such as sensor noise, dynamic perturbations, and runtime assurance (RTA) constraints. The results demonstrate that the hierarchical controller effectively bridges the sim-to-real gap, maintaining high task completion rates while adapting to the complexities of real-world environments. These findings validate the framework's potential for enabling autonomous satellite operations in future space missions. 

**Abstract (ZH)**: 空间任务中自主系统的设计与部署需要应对严格可靠性的约束、长时间运行以及通信挑战的稳健解决方案。本研究评估了一种用于多智能体卫星检查任务的分层深度强化学习框架的稳定性和性能。提出的框架将高层指导策略与低层运动控制器相结合，实现可扩展的任务分配和高效的轨迹执行。在本地协作卫星的局部智能网络（LINCS）试验台上，从模拟环境到物理-计算试验台的不同保真度水平下进行了实验，评估了该框架的性能。关键指标，包括任务完成率、行驶距离和燃料消耗，展示了框架在真实世界不确定性如传感器噪声、动态扰动和运行时间保证（RTA）约束下的鲁棒性和适应性。实验结果表明，分层控制器有效地弥合了仿真与现实之间的差距，维持了高任务完成率的同时适应了真实世界环境的复杂性。这些发现验证了该框架在未来空间任务中实现自主卫星操作的潜力。 

---
# AI and Semantic Communication for Infrastructure Monitoring in 6G-Driven Drone Swarms 

**Title (ZH)**: AI和语义通信在6G驱动的无人机群基础设施监控中的应用 

**Authors**: Tasnim Ahmed, Salimur Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2503.00053)  

**Abstract**: The adoption of unmanned aerial vehicles to monitor critical infrastructure is gaining momentum in various industrial domains. Organizational imperatives drive this progression to minimize expenses, accelerate processes, and mitigate hazards faced by inspection personnel. However, traditional infrastructure monitoring systems face critical bottlenecks-5G networks lack the latency and reliability for large-scale drone coordination, while manual inspections remain costly and slow. We propose a 6G-enabled drone swarm system that integrates ultra-reliable, low-latency communications, edge AI, and semantic communication to automate inspections. By adopting LLMs for structured output and report generation, our framework is hypothesized to reduce inspection costs and improve fault detection speed compared to existing methods. 

**Abstract (ZH)**: 6G使能的无人机群系统：集成超可靠低延迟通信、边缘AI和语义通信的智能巡检技术 

---
# Glad: A Streaming Scene Generator for Autonomous Driving 

**Title (ZH)**: Glad: 自动驾驶的流式场景生成器 

**Authors**: Bin Xie, Yingfei Liu, Tiancai Wang, Jiale Cao, Xiangyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00045)  

**Abstract**: The generation and simulation of diverse real-world scenes have significant application value in the field of autonomous driving, especially for the corner cases. Recently, researchers have explored employing neural radiance fields or diffusion models to generate novel views or synthetic data under driving scenes. However, these approaches suffer from unseen scenes or restricted video length, thus lacking sufficient adaptability for data generation and simulation. To address these issues, we propose a simple yet effective framework, named Glad, to generate video data in a frame-by-frame style. To ensure the temporal consistency of synthetic video, we introduce a latent variable propagation module, which views the latent features of previous frame as noise prior and injects it into the latent features of current frame. In addition, we design a streaming data sampler to orderly sample the original image in a video clip at continuous iterations. Given the reference frame, our Glad can be viewed as a streaming simulator by generating the videos for specific scenes. Extensive experiments are performed on the widely-used nuScenes dataset. Experimental results demonstrate that our proposed Glad achieves promising performance, serving as a strong baseline for online video generation. We will release the source code and models publicly. 

**Abstract (ZH)**: 生成和模拟多样化的现实场景在自动驾驶领域具有重要的应用价值，尤其是在处理corner cases方面。最近，研究人员探索了利用神经辐射场或扩散模型在驾驶场景下生成新颖视图或合成数据。然而，这些方法受到未见过的场景或视频长度限制的影响，因而缺乏足够的数据生成和模拟适应性。为解决这些问题，我们提出了一种简单有效的框架Glad，以帧级别的方式生成视频数据。为了确保合成视频的时间连贯性，我们引入了一个潜在变量传播模块，将前一帧的潜在特征视为噪声先验，并将其注入当前帧的潜在特征中。此外，我们设计了流式数据采样器，在连续迭代中有序地从视频片段中采样原始图像。给定参考帧，我们的Glad可以被视为一个流式模拟器，通过为特定场景生成视频。我们在广泛使用的nuScenes数据集上进行了 extensive 实验。实验结果表明，我们提出的Glad在在线视频生成方面取得了令人鼓舞的性能，作为在线视频生成的强基线，我们将在未来公开发布源代码和模型。 

---
# Multi-Stage Manipulation with Demonstration-Augmented Reward, Policy, and World Model Learning 

**Title (ZH)**: 演示增强奖励、策略和世界模型的多阶段操控 

**Authors**: Adrià López Escoriza, Nicklas Hansen, Stone Tao, Tongzhou Mu, Hao Su  

**Link**: [PDF](https://arxiv.org/pdf/2503.01837)  

**Abstract**: Long-horizon tasks in robotic manipulation present significant challenges in reinforcement learning (RL) due to the difficulty of designing dense reward functions and effectively exploring the expansive state-action space. However, despite a lack of dense rewards, these tasks often have a multi-stage structure, which can be leveraged to decompose the overall objective into manageable subgoals. In this work, we propose DEMO3, a framework that exploits this structure for efficient learning from visual inputs. Specifically, our approach incorporates multi-stage dense reward learning, a bi-phasic training scheme, and world model learning into a carefully designed demonstration-augmented RL framework that strongly mitigates the challenge of exploration in long-horizon tasks. Our evaluations demonstrate that our method improves data-efficiency by an average of 40% and by 70% on particularly difficult tasks compared to state-of-the-art approaches. We validate this across 16 sparse-reward tasks spanning four domains, including challenging humanoid visual control tasks using as few as five demonstrations. 

**Abstract (ZH)**: 长时_horizon_任务在机器人操作中的强化学习中带来了显著挑战，原因在于密集奖励函数的设计困难以及对广阔状态-动作空间的有效探索。然而，尽管缺乏密集奖励，这些任务通常具有多阶段结构，可以利用这种结构将总体目标分解为可管理的子目标。在本工作中，我们提出了DEMO3框架，该框架利用这种结构从视觉输入中高效地进行学习。具体而言，我们的方法结合了多阶段密集奖励学习、双阶段训练方案和世界模型学习，以精心设计的演示增强 RL 框架，显著减轻了长时_horizon_任务中的探索挑战。我们的评估表明，与最先进的方法相比，我们的方法在数据效率上平均提高了40%，在特别困难的任务上提高了70%。我们在包括最少仅需五次演示的人形视觉控制任务在内的四个领域中的16个稀疏奖励任务中进行了验证。 

---
# CAPS: Context-Aware Priority Sampling for Enhanced Imitation Learning in Autonomous Driving 

**Title (ZH)**: CAPS：面向情境的优先采样方法以增强自主驾驶中的 imitation learning 

**Authors**: Hamidreza Mirkhani, Behzad Khamidehi, Ehsan Ahmadi, Fazel Arasteh, Mohammed Elmahgiubi, Weize Zhang, Umar Rajguru, Kasra Rezaee  

**Link**: [PDF](https://arxiv.org/pdf/2503.01650)  

**Abstract**: In this paper, we introduce CAPS (Context-Aware Priority Sampling), a novel method designed to enhance data efficiency in learning-based autonomous driving systems. CAPS addresses the challenge of imbalanced training datasets in imitation learning by leveraging Vector Quantized Variational Autoencoders (VQ-VAEs). The use of VQ-VAE provides a structured and interpretable data representation, which helps reveal meaningful patterns in the data. These patterns are used to group the data into clusters, with each sample being assigned a cluster ID. The cluster IDs are then used to re-balance the dataset, ensuring that rare yet valuable samples receive higher priority during training. By ensuring a more diverse and informative training set, CAPS improves the generalization of the trained planner across a wide range of driving scenarios. We evaluate our method through closed-loop simulations in the CARLA environment. The results on Bench2Drive scenarios demonstrate that our framework outperforms state-of-the-art methods, leading to notable improvements in model performance. 

**Abstract (ZH)**: 基于上下文感知优先采样的自动驾驶数据效率提升方法（CAPS） 

---
# A Note on the Time Complexity of Using Subdivision Methods for the Approximation of Fibers 

**Title (ZH)**: 关于使用分划方法近似纤维的时间复杂性注记 

**Authors**: Michael M. Bilevich, Dan Halperin  

**Link**: [PDF](https://arxiv.org/pdf/2503.01626)  

**Abstract**: Subdivision methods such as quadtrees, octrees, and higher-dimensional orthrees are standard practice in different domains of computer science. We can use these methods to represent given geometries, such as curves, meshes, or surfaces. This representation is achieved by splitting some bounding voxel recursively while further splitting only sub-voxels that intersect with the given geometry. It is fairly known that subdivision methods are more efficient than traversing a fine-grained voxel grid. In this short note, we propose another outlook on analyzing the construction time complexity of orthrees to represent implicitly defined geometries that are fibers (preimages) of some function. This complexity is indeed asymptotically better than traversing dense voxel grids, under certain conditions, which we specify in the note. In fact, the complexity is output sensitive, and is closely related to the Hausdorff measure and Hausdorff dimension of the resulting geometry. 

**Abstract (ZH)**: 细分方法，如四叉树、八叉树和高维正交树在计算机科学的不同领域都是标准实践。我们可以通过这些方法来表示给定的几何形状，如曲线、网格或曲面。这种表示是通过对一些边界体素进行递归分割，仅进一步分割与给定几何形状相交的子体素来实现的。已知细分方法比遍历细粒度体素网格更高效。在本文简要说明中，我们提出了另一种分析使用正交树表示由某些函数定义的隐式几何（纤维或前image）的构建时间复杂性的视角。在某些条件下，这种复杂性确实比遍历密集体素网格更具渐近优势，我们将在说明中指定这些条件。实际上，这种复杂性对输出敏感，并且与所得几何的豪斯多夫测度和豪斯多夫维数密切相关。 

---
# Category-level Meta-learned NeRF Priors for Efficient Object Mapping 

**Title (ZH)**: 类别级元学习NERF先验用于高效物体映射 

**Authors**: Saad Ejaz, Hriday Bavle, Laura Ribeiro, Holger Voos, Jose Luis Sanchez-Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2503.01582)  

**Abstract**: In 3D object mapping, category-level priors enable efficient object reconstruction and canonical pose estimation, requiring only a single prior per semantic category (e.g., chair, book, laptop). Recently, DeepSDF has predominantly been used as a category-level shape prior, but it struggles to reconstruct sharp geometry and is computationally expensive. In contrast, NeRFs capture fine details but have yet to be effectively integrated with category-level priors in a real-time multi-object mapping framework. To bridge this gap, we introduce PRENOM, a Prior-based Efficient Neural Object Mapper that integrates category-level priors with object-level NeRFs to enhance reconstruction efficiency while enabling canonical object pose estimation. PRENOM gets to know objects on a first-name basis by meta-learning on synthetic reconstruction tasks generated from open-source shape datasets. To account for object category variations, it employs a multi-objective genetic algorithm to optimize the NeRF architecture for each category, balancing reconstruction quality and training time. Additionally, prior-based probabilistic ray sampling directs sampling toward expected object regions, accelerating convergence and improving reconstruction quality under constrained resources. Experimental results on a low-end GPU highlight the ability of PRENOM to achieve high-quality reconstructions while maintaining computational feasibility. Specifically, comparisons with prior-free NeRF-based approaches on a synthetic dataset show a 21% lower Chamfer distance, demonstrating better reconstruction quality. Furthermore, evaluations against other approaches using shape priors on a noisy real-world dataset indicate a 13% improvement averaged across all reconstruction metrics, a boost in rotation estimation accuracy, and comparable translation and size estimation performance, while being trained for 5x less time. 

**Abstract (ZH)**: 基于先验的高效神经对象.mapper：将类别级先验与对象级NeRF集成以增强重建效率并实现标准化对象姿态估计 

---
# Trajectory Planning with Signal Temporal Logic Costs using Deterministic Path Integral Optimization 

**Title (ZH)**: 使用确定性路径积分优化的信号时序逻辑代价轨迹规划 

**Authors**: Patrick Halder, Hannes Homburger, Lothar Kiltz, Johannes Reuter, Matthias Althoff  

**Link**: [PDF](https://arxiv.org/pdf/2503.01476)  

**Abstract**: Formulating the intended behavior of a dynamic system can be challenging. Signal temporal logic (STL) is frequently used for this purpose due to its suitability in formalizing comprehensible, modular, and versatile spatiotemporal specifications. Due to scaling issues with respect to the complexity of the specifications and the potential occurrence of non-differentiable terms, classical optimization methods often solve STL-based problems inefficiently. Smoothing and approximation techniques can alleviate these issues but require changing the optimization problem. This paper proposes a novel sampling-based method based on model predictive path integral control to solve optimal control problems with STL cost functions. We demonstrate the effectiveness of our method on benchmark motion planning problems and compare its performance with state-of-the-art methods. The results show that our method efficiently solves optimal control problems with STL costs. 

**Abstract (ZH)**: 基于模型预测路径积分控制的采样方法求解带有STL成本函数的最优控制问题 

---
# POPGym Arcade: Parallel Pixelated POMDPs 

**Title (ZH)**: POPGym 游戏厅：并行像素化部分可观察马尔可夫决策过程 

**Authors**: Zekang Wang, Zhe He, Edan Toledo, Steven Morad  

**Link**: [PDF](https://arxiv.org/pdf/2503.01450)  

**Abstract**: We introduce POPGym Arcade, a benchmark consisting of 7 pixel-based environments each with three difficulties, utilizing a single observation and action space. Each environment offers both fully observable and partially observable variants, enabling counterfactual studies on partial observability. POPGym Arcade utilizes JIT compilation on hardware accelerators to achieve substantial speedups over CPU-bound environments. Moreover, this enables Podracer-style architectures to further increase hardware utilization and training speed. We evaluate memory models on our environments using a Podracer variant of Q learning, and examine the results. Finally, we generate memory saliency maps, uncovering how memories propagate through policies. Our library is available at this https URL popgym_arcade. 

**Abstract (ZH)**: POPGym Arcade：一个包含7个基于像素的环境的基准，每个环境有三个难度级别，利用单一的观察和动作空间。该基准既包含了完全可观测版本也包含了部分可观测版本的环境，便于进行部分可观测性的反事实研究。POPGym Arcade利用硬件加速器上的JIT编译，实现了相对于CPU受限环境的显著加速，并使得Podracer风格的架构能够进一步提高硬件利用率和训练速度。我们使用Podracer变种的Q学习评估了内存模型，并研究了结果。最后，我们生成了记忆显著性图，揭示了记忆如何在策略中传播。我们的库可在以下链接获取：popgym_arcade。 

---
# Convex Hull-based Algebraic Constraint for Visual Quadric SLAM 

**Title (ZH)**: 基于凸包的代数约束视觉四次多项式SLAM 

**Authors**: Xiaolong Yu, Junqiao Zhao, Shuangfu Song, Zhongyang Zhu, Zihan Yuan, Chen Ye, Tiantian Feng  

**Link**: [PDF](https://arxiv.org/pdf/2503.01254)  

**Abstract**: Using Quadrics as the object representation has the benefits of both generality and closed-form projection derivation between image and world spaces. Although numerous constraints have been proposed for dual quadric reconstruction, we found that many of them are imprecise and provide minimal improvements to this http URL scrutinizing the existing constraints, we introduce a concise yet more precise convex hull-based algebraic constraint for object landmarks, which is applied to object reconstruction, frontend pose estimation, and backend bundle this http URL constraint is designed to fully leverage precise semantic segmentation, effectively mitigating mismatches between complex-shaped object contours and dual this http URL on public datasets demonstrate that our approach is applicable to both monocular and RGB-D SLAM and achieves improved object mapping and localization than existing quadric SLAM methods. The implementation of our method is available at this https URL. 

**Abstract (ZH)**: 将双四面体重建中的对象表示用quadrics表示，兼具通用性和图像空间与世界空间闭式投影推导的优点。尽管为双四面体重建提出了众多约束条件，我们发现其中许多约束不够精确且提供的改进有限。通过对现有约束条件进行仔细分析，我们引入了一种更为精确且简洁的基于凸包的代数约束，将其应用于对象重建、前端姿态估计和后端 bundle 调整。该约束旨在充分利用精确的语义分割，有效缓解复杂形状对象轮廓与双四面体之间的不匹配问题。在公共数据集上的实验表明，我们的方法适用于单目和RGB-D SLAM，并在对象建图和定位方面优于现有四面体SLAM方法。我们的方法实现可从这个 <https://www.example.com> 获取。 

---
# A Multi-Sensor Fusion Approach for Rapid Orthoimage Generation in Large-Scale UAV Mapping 

**Title (ZH)**: 大规模无人机测绘中基于多传感器融合的快速正射影像生成方法 

**Authors**: Jialei He, Zhihao Zhan, Zhituo Tu, Xiang Zhu, Jie Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2503.01202)  

**Abstract**: Rapid generation of large-scale orthoimages from Unmanned Aerial Vehicles (UAVs) has been a long-standing focus of research in the field of aerial mapping. A multi-sensor UAV system, integrating the Global Positioning System (GPS), Inertial Measurement Unit (IMU), 4D millimeter-wave radar and camera, can provide an effective solution to this problem. In this paper, we utilize multi-sensor data to overcome the limitations of conventional orthoimage generation methods in terms of temporal performance, system robustness, and geographic reference accuracy. A prior-pose-optimized feature matching method is introduced to enhance matching speed and accuracy, reducing the number of required features and providing precise references for the Structure from Motion (SfM) process. The proposed method exhibits robustness in low-texture scenes like farmlands, where feature matching is difficult. Experiments show that our approach achieves accurate feature matching orthoimage generation in a short time. The proposed drone system effectively aids in farmland detection and management. 

**Abstract (ZH)**: 基于多传感器的无人机系统快速生成大规模正射影像的研究 

---
# Differentiable Information Enhanced Model-Based Reinforcement Learning 

**Title (ZH)**: 可微信息增强模型导向强化学习 

**Authors**: Xiaoyuan Zhang, Xinyan Cai, Bo Liu, Weidong Huang, Song-Chun Zhu, Siyuan Qi, Yaodong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01178)  

**Abstract**: Differentiable environments have heralded new possibilities for learning control policies by offering rich differentiable information that facilitates gradient-based methods. In comparison to prevailing model-free reinforcement learning approaches, model-based reinforcement learning (MBRL) methods exhibit the potential to effectively harness the power of differentiable information for recovering the underlying physical dynamics. However, this presents two primary challenges: effectively utilizing differentiable information to 1) construct models with more accurate dynamic prediction and 2) enhance the stability of policy training. In this paper, we propose a Differentiable Information Enhanced MBRL method, MB-MIX, to address both challenges. Firstly, we adopt a Sobolev model training approach that penalizes incorrect model gradient outputs, enhancing prediction accuracy and yielding more precise models that faithfully capture system dynamics. Secondly, we introduce mixing lengths of truncated learning windows to reduce the variance in policy gradient estimation, resulting in improved stability during policy learning. To validate the effectiveness of our approach in differentiable environments, we provide theoretical analysis and empirical results. Notably, our approach outperforms previous model-based and model-free methods, in multiple challenging tasks involving controllable rigid robots such as humanoid robots' motion control and deformable object manipulation. 

**Abstract (ZH)**: 不同的环境为基于梯度的方法提供了丰富的可微信息，开启了学习控制策略的新可能性。与当前占主导地位的无模型强化学习方法相比，基于模型的强化学习（MBRL）方法具备利用可微信息恢复潜在物理动态的潜力。然而，这提出了两个主要挑战：有效利用可微信息以1）构建更准确的动力学预测模型，2）提高策略训练的稳定性。本文提出了一种可微信息增强的MBRL方法MB-MIX来解决这两个挑战。首先，我们采用Sobolev模型训练方法惩罚错误的模型梯度输出，提高预测准确性并产生更精确地捕捉系统动力学的模型。其次，我们引入截断学习窗口的混合长度来减少策略梯度估计的方差，从而在策略学习过程中提高稳定性。为了验证我们在可微环境中方法的有效性，我们提供了理论分析和实证结果。值得注意的是，与先前的基于模型和无模型方法相比，我们的方法在涉及可控刚体（如类人机器人动作控制和可变形物体操作）的多个具有挑战性的任务中表现更优。 

---
# FGS-SLAM: Fourier-based Gaussian Splatting for Real-time SLAM with Sparse and Dense Map Fusion 

**Title (ZH)**: 基于傅里叶变换的高斯点云融合实时SLAM：稀疏与密集地图融合 

**Authors**: Yansong Xu, Junlin Li, Wei Zhang, Siyu Chen, Shengyong Zhang, Yuquan Leng, Weijia Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.01109)  

**Abstract**: 3D gaussian splatting has advanced simultaneous localization and mapping (SLAM) technology by enabling real-time positioning and the construction of high-fidelity maps. However, the uncertainty in gaussian position and initialization parameters introduces challenges, often requiring extensive iterative convergence and resulting in redundant or insufficient gaussian representations. To address this, we introduce a novel adaptive densification method based on Fourier frequency domain analysis to establish gaussian priors for rapid convergence. Additionally, we propose constructing independent and unified sparse and dense maps, where a sparse map supports efficient tracking via Generalized Iterative Closest Point (GICP) and a dense map creates high-fidelity visual representations. This is the first SLAM system leveraging frequency domain analysis to achieve high-quality gaussian mapping in real-time. Experimental results demonstrate an average frame rate of 36 FPS on Replica and TUM RGB-D datasets, achieving competitive accuracy in both localization and mapping. 

**Abstract (ZH)**: 基于傅里叶频率域分析的自适应密集化方法在3D高斯求和SLAM中的应用：实现实时高质量高斯映射 

---
# One-Shot Affordance Grounding of Deformable Objects in Egocentric Organizing Scenes 

**Title (ZH)**: 一手构建主观 organizing 场景中变形物体的功能接地 

**Authors**: Wanjun Jia, Fan Yang, Mengfei Duan, Xianchi Chen, Yinxi Wang, Yiming Jiang, Wenrui Chen, Kailun Yang, Zhiyong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.01092)  

**Abstract**: Deformable object manipulation in robotics presents significant challenges due to uncertainties in component properties, diverse configurations, visual interference, and ambiguous prompts. These factors complicate both perception and control tasks. To address these challenges, we propose a novel method for One-Shot Affordance Grounding of Deformable Objects (OS-AGDO) in egocentric organizing scenes, enabling robots to recognize previously unseen deformable objects with varying colors and shapes using minimal samples. Specifically, we first introduce the Deformable Object Semantic Enhancement Module (DefoSEM), which enhances hierarchical understanding of the internal structure and improves the ability to accurately identify local features, even under conditions of weak component information. Next, we propose the ORB-Enhanced Keypoint Fusion Module (OEKFM), which optimizes feature extraction of key components by leveraging geometric constraints and improves adaptability to diversity and visual interference. Additionally, we propose an instance-conditional prompt based on image data and task context, effectively mitigates the issue of region ambiguity caused by prompt words. To validate these methods, we construct a diverse real-world dataset, AGDDO15, which includes 15 common types of deformable objects and their associated organizational actions. Experimental results demonstrate that our approach significantly outperforms state-of-the-art methods, achieving improvements of 6.2%, 3.2%, and 2.9% in KLD, SIM, and NSS metrics, respectively, while exhibiting high generalization performance. Source code and benchmark dataset will be publicly available at this https URL. 

**Abstract (ZH)**: 面向自我中心整理场景的可变形物体一次性gradable用途 grounding 方法（OS-AGDO） 

---
# HiMo: High-Speed Objects Motion Compensation in Point Clouds 

**Title (ZH)**: HiMo: 高速点云中物体运动补偿 

**Authors**: Qingwen Zhang, Ajinkya Khoche, Yi Yang, Li Ling, Sina Sharif Mansouri, Olov Andersson, Patric Jensfelt  

**Link**: [PDF](https://arxiv.org/pdf/2503.00803)  

**Abstract**: LiDAR point clouds often contain motion-induced distortions, degrading the accuracy of object appearances in the captured data. In this paper, we first characterize the underlying reasons for the point cloud distortion and show that this is present in public datasets. We find that this distortion is more pronounced in high-speed environments such as highways, as well as in multi-LiDAR configurations, a common setup for heavy vehicles. Previous work has dealt with point cloud distortion from the ego-motion but fails to consider distortion from the motion of other objects. We therefore introduce a novel undistortion pipeline, HiMo, that leverages scene flow estimation for object motion compensation, correcting the depiction of dynamic objects. We further propose an extension of a state-of-the-art self-supervised scene flow method. Due to the lack of well-established motion distortion metrics in the literature, we also propose two metrics for compensation performance evaluation: compensation accuracy at a point level and shape similarity on objects. To demonstrate the efficacy of our method, we conduct extensive experiments on the Argoverse 2 dataset and a new real-world dataset. Our new dataset is collected from heavy vehicles equipped with multi-LiDARs and on highways as opposed to mostly urban settings in the existing datasets. The source code, including all methods and the evaluation data, will be provided upon publication. See this https URL for more details. 

**Abstract (ZH)**: LiDAR 点云中的运动引起的失真会降低捕获数据中物体外观的准确性。本文首先分析点云失真的根本原因，并表明这种失真存在于公共数据集中。我们发现，这种失真在高速公路等高速环境中以及多LiDAR配置中更为明显，这是重型车辆中常见的配置。以往的工作已经处理了来自自身运动的点云失真，但未能考虑到其他物体运动引起的失真。因此，我们引入了一种名为HiMo的新型解畸变流水线，该流水线利用场景流估计进行物体运动补偿，纠正动态物体的描述。此外，我们提出了最先进的自监督场景流方法的一个扩展。由于文献中缺乏成熟的运动失真评估指标，我们还提出了两种性能评估指标：基于点的补偿准确性和对象的形状相似度。为了证明我们方法的有效性，我们在Argoverse 2数据集和一个新的真实世界数据集上进行了广泛的实验。我们的新数据集源自配备了多LiDAR的重型车辆，并在高速公路上收集，而现有的数据集主要集中在城市环境中。论文发表后，将提供源代码，包括所有方法和评估数据。更多细节请见：见此链接。 

---
# Bridging Spectral-wise and Multi-spectral Depth Estimation via Geometry-guided Contrastive Learning 

**Title (ZH)**: 基于几何引导对比学习的频谱级和多光谱深度估计桥梁方法 

**Authors**: Ukcheol Shin, Kyunghyun Lee, Jean Oh  

**Link**: [PDF](https://arxiv.org/pdf/2503.00793)  

**Abstract**: Deploying depth estimation networks in the real world requires high-level robustness against various adverse conditions to ensure safe and reliable autonomy. For this purpose, many autonomous vehicles employ multi-modal sensor systems, including an RGB camera, NIR camera, thermal camera, LiDAR, or Radar. They mainly adopt two strategies to use multiple sensors: modality-wise and multi-modal fused inference. The former method is flexible but memory-inefficient, unreliable, and vulnerable. Multi-modal fusion can provide high-level reliability, yet it needs a specialized architecture. In this paper, we propose an effective solution, named align-and-fuse strategy, for the depth estimation from multi-spectral images. In the align stage, we align embedding spaces between multiple spectrum bands to learn shareable representation across multi-spectral images by minimizing contrastive loss of global and spatially aligned local features with geometry cue. After that, in the fuse stage, we train an attachable feature fusion module that can selectively aggregate the multi-spectral features for reliable and robust prediction results. Based on the proposed method, a single-depth network can achieve both spectral-invariant and multi-spectral fused depth estimation while preserving reliability, memory efficiency, and flexibility. 

**Abstract (ZH)**: 在实际环境中部署深度估计网络需要高度 robust 性以应对各种不利条件，确保安全可靠地自主运行。为此，许多自动驾驶车辆采用多模态传感器系统，包括 RGB 相机、近红外相机、热相机、激光雷达或雷达。它们主要采用两种策略来利用多个传感器：模态特定和多模态融合推理。前者灵活但内存效率不高，不可靠且易受攻击。多模态融合可以提供高级别的可靠性，但需要专门的架构。本文提出了一种有效的解决方案，即对多谱图像进行深度估计的对齐和融合策略。在对齐阶段，我们通过对齐不同谱带的嵌入空间，通过最小化几何线索引导的全局和空间对齐局部特征的对比损失来学习多谱图像间的共享表示。接着，在融合阶段，我们训练一个可附加的特征融合模块，可以有选择地聚合多谱特征，以获得可靠且 robust 的预测结果。基于本方法，单个深度网络可以同时实现谱不变性和多谱融合深度估计，同时保持可靠性、内存效率和灵活性。 

---
# Unifying Light Field Perception with Field of Parallax 

**Title (ZH)**: 统一视差场与光场感知 

**Authors**: Fei Teng, Buyin Deng, Boyuan Zheng, Kai Luo, Kunyu Peng, Jiaming Zhang, Kailun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00747)  

**Abstract**: Field of Parallax (FoP)}, a spatial field that distills the common features from different LF representations to provide flexible and consistent support for multi-task learning. FoP is built upon three core features--projection difference, adjacency divergence, and contextual consistency--which are essential for cross-task adaptability. To implement FoP, we design a two-step angular adapter: the first step captures angular-specific differences, while the second step consolidates contextual consistency to ensure robust representation. Leveraging the FoP-based representation, we introduce the LFX framework, the first to handle arbitrary LF representations seamlessly, unifying LF multi-task vision. We evaluated LFX across three different tasks, achieving new state-of-the-art results, compared with previous task-specific architectures: 84.74% in mIoU for semantic segmentation on UrbanLF, 0.84% in AP for object detection on PKU, and 0.030 in MAE and 0.026 in MAE for salient object detection on Duftv2 and PKU, respectively. The source code will be made publicly available at this https URL. 

**Abstract (ZH)**: 基于偏移场（FoP）的空间场：一种从不同超分辨率表示中提炼共性特征以提供灵活一致多任务学习支持的场。FoP基于投影差异、相邻差异和上下文一致性三种核心特征构建，这些特征对于跨任务适应性至关重要。为了实现FoP，我们设计了一种两步角度适配器：第一步捕捉角度特定差异，第二步 consolidates 上下文一致性以确保稳健的表示。利用基于FoP的表示，我们引入了LFX框架，这是首个能够无缝处理任意超分辨率表示、统一超分辨率多任务视觉的框架。我们在三个不同任务上评估了LFX，取得了之前任务特定架构的新最佳结果：在UrbanLF语义分割上的mIoU为84.74%，在PKU物体检测上的AP为0.84%，在Duftv2和PKU上的显著对象检测上的MAE分别为0.030和0.026。源代码将在以下网址公开： this https URL。 

---
# Factorized Deep Q-Network for Cooperative Multi-Agent Reinforcement Learning in Victim Tagging 

**Title (ZH)**: 因子分解深度Q网络在受害者标记的协同多agent reinforcement学习中应用 

**Authors**: Maria Ana Cardei, Afsaneh Doryab  

**Link**: [PDF](https://arxiv.org/pdf/2503.00684)  

**Abstract**: Mass casualty incidents (MCIs) are a growing concern, characterized by complexity and uncertainty that demand adaptive decision-making strategies. The victim tagging step in the emergency medical response must be completed quickly and is crucial for providing information to guide subsequent time-constrained response actions. In this paper, we present a mathematical formulation of multi-agent victim tagging to minimize the time it takes for responders to tag all victims. Five distributed heuristics are formulated and evaluated with simulation experiments. The heuristics considered are on-the go, practical solutions that represent varying levels of situational uncertainty in the form of global or local communication capabilities, showcasing practical constraints. We further investigate the performance of a multi-agent reinforcement learning (MARL) strategy, factorized deep Q-network (FDQN), to minimize victim tagging time as compared to baseline heuristics. Extensive simulations demonstrate that between the heuristics, methods with local communication are more efficient for adaptive victim tagging, specifically choosing the nearest victim with the option to replan. Analyzing all experiments, we find that our FDQN approach outperforms heuristics in smaller-scale scenarios, while heuristics excel in more complex scenarios. Our experiments contain diverse complexities that explore the upper limits of MARL capabilities for real-world applications and reveal key insights. 

**Abstract (ZH)**: 大规模伤亡事件中的多agent受害者标记：数学建模与算法优化 

---
# Dur360BEV: A Real-world Single 360-degree Camera Dataset and Benchmark for Bird-Eye View Mapping in Autonomous Driving 

**Title (ZH)**: Dur360BEV：用于自动驾驶中鸟瞰图建图的真实世界单个360度摄像头数据集及基准 

**Authors**: Wenke E, Chao Yuan, Li Li, Yixin Sun, Yona Falinie A. Gaus, Amir Atapour-Abarghouei, Toby P. Breckon  

**Link**: [PDF](https://arxiv.org/pdf/2503.00675)  

**Abstract**: We present Dur360BEV, a novel spherical camera autonomous driving dataset equipped with a high-resolution 128-channel 3D LiDAR and a RTK-refined GNSS/INS system, along with a benchmark architecture designed to generate Bird-Eye-View (BEV) maps using only a single spherical camera. This dataset and benchmark address the challenges of BEV generation in autonomous driving, particularly by reducing hardware complexity through the use of a single 360-degree camera instead of multiple perspective cameras. Within our benchmark architecture, we propose a novel spherical-image-to-BEV (SI2BEV) module that leverages spherical imagery and a refined sampling strategy to project features from 2D to 3D. Our approach also includes an innovative application of Focal Loss, specifically adapted to address the extreme class imbalance often encountered in BEV segmentation tasks. Through extensive experiments, we demonstrate that this application of Focal Loss significantly improves segmentation performance on the Dur360BEV dataset. The results show that our benchmark not only simplifies the sensor setup but also achieves competitive performance. 

**Abstract (ZH)**: Dur360BEV：一种配备高分辨率128通道3D LiDAR和RTK校准的GNSS/INS系统的新型球形 cameras 自动驾驶数据集及单目球形图像到BEV地图生成基准架构 

---
# ExAMPC: the Data-Driven Explainable and Approximate NMPC with Physical Insights 

**Title (ZH)**: ExAMPC：基于数据的可解释性和近似NMPC，融入物理洞察 

**Authors**: Jean Pierre Allamaa, Panagiotis Patrinos, Tong Duy Son  

**Link**: [PDF](https://arxiv.org/pdf/2503.00654)  

**Abstract**: Amidst the surge in the use of Artificial Intelligence (AI) for control purposes, classical and model-based control methods maintain their popularity due to their transparency and deterministic nature. However, advanced controllers like Nonlinear Model Predictive Control (NMPC), despite proven capabilities, face adoption challenges due to their computational complexity and unpredictable closed-loop performance in complex validation systems. This paper introduces ExAMPC, a methodology bridging classical control and explainable AI by augmenting the NMPC with data-driven insights to improve the trustworthiness and reveal the optimization solution and closed-loop performance's sensitivities to physical variables and system parameters. By employing a low-order spline embedding to reduce the open-loop trajectory dimensionality by over 95%, and integrating it with SHAP and Symbolic Regression from eXplainable AI (XAI) for an approximate NMPC, we enable intuitive physical insights into the NMPC's optimization routine. The prediction accuracy of the approximate NMPC is enhanced through physics-inspired continuous-time constraints penalties, reducing the predicted continuous trajectory violations by 93%. ExAMPC enables accurate forecasting of the NMPC's computational requirements with explainable insights on worst-case scenarios. Experimental validation on automated valet parking and autonomous racing with lap-time optimization NMPC, demonstrates the methodology's practical effectiveness in real-world applications. 

**Abstract (ZH)**: 伴随人工智能（AI）在控制领域应用的增加，经典和模型参考控制方法由于其透明性和确定性仍保持受欢迎。然而，先进的控制器如非线性模型预测控制（NMPC），尽管已证明其能力，但由于计算复杂性和在复杂验证系统中的不可预测闭环性能，其应用面临挑战。本文提出了ExAMPC方法，通过将数据驱动见解嵌入NMPC以增强其可信度，并揭示优化解和闭环性能对物理变量和系统参数的灵敏度，从而在经典控制和可解释AI之间架起桥梁。通过采用低阶样条嵌入减少开放轨迹维度超过95%，并与可解释AI（XAI）中的SHAP和符号回归相结合构建近似NMPC，我们使NMPC的优化过程具有直观的物理洞察。通过引入基于物理的连续时间约束惩罚，近似NMPC的预测准确性提高，预测连续轨迹违规减少93%。ExAMPC能够提供关于NMPC计算需求和最坏情况场景的可解释洞察。在自动valet停车和基于圈速优化的自主赛车试验验证中，该方法展示了其实用有效性。 

---
# Inteval Analysis for two spherical functions arising from robust Perspective-n-Lines problem 

**Title (ZH)**: 两球面函数在鲁棒透视-n-线问题中的区间分析 

**Authors**: Xiang Zheng, Haodong Jiang, Junfeng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00400)  

**Abstract**: This report presents a comprehensive interval analysis of two spherical functions derived from the robust Perspective-n-Lines (PnL) problem. The study is motivated by the application of a dimension-reduction technique to achieve global solutions for the robust PnL problem. We establish rigorous theoretical results, supported by detailed proofs, and validate our findings through extensive numerical simulations. 

**Abstract (ZH)**: 本报告对两sphere函数从鲁棒Perspective-n-Lines (PnL)问题导出的区间进行全面分析。该研究受到将降维技术应用于实现鲁棒PnL问题全局解的动机驱动。我们建立了严格的理论结果，并通过广泛的数值仿真验证了这些结果。 

---
# EVLoc: Event-based Visual Localization in LiDAR Maps via Event-Depth Registration 

**Title (ZH)**: EVLoc：基于事件深度注册的LiDAR地图中事件视觉定位 

**Authors**: Kuangyi Chen, Jun Zhang, Friedrich Fraundorfer  

**Link**: [PDF](https://arxiv.org/pdf/2503.00167)  

**Abstract**: Event cameras are bio-inspired sensors with some notable features, including high dynamic range and low latency, which makes them exceptionally suitable for perception in challenging scenarios such as high-speed motion and extreme lighting conditions. In this paper, we explore their potential for localization within pre-existing LiDAR maps, a critical task for applications that require precise navigation and mobile manipulation. Our framework follows a paradigm based on the refinement of an initial pose. Specifically, we first project LiDAR points into 2D space based on a rough initial pose to obtain depth maps, and then employ an optical flow estimation network to align events with LiDAR points in 2D space, followed by camera pose estimation using a PnP solver. To enhance geometric consistency between these two inherently different modalities, we develop a novel frame-based event representation that improves structural clarity. Additionally, given the varying degrees of bias observed in the ground truth poses, we design a module that predicts an auxiliary variable as a regularization term to mitigate the impact of this bias on network convergence. Experimental results on several public datasets demonstrate the effectiveness of our proposed method. To facilitate future research, both the code and the pre-trained models are made available online. 

**Abstract (ZH)**: 事件相机是受生物启发的传感器，具有高动态范围和低延迟等显著特点，使其在高速运动和极端光照等挑战性场景下的感知任务中表现出色。本文探讨了其在已有LiDAR地图中实现定位的潜在应用，这对于需要精确导航和移动操作的应用至关重要。我们的框架基于初始姿态 refinement 的 paradigm。具体而言，我们首先根据粗略的初始姿态将LiDAR点投影到二维空间以获取深度图，然后使用光流估计网络将事件与二维空间中的LiDAR点对齐，最后利用PnP求解器估计相机姿态。为了在这些本质上不同的模态之间增强几何一致性，我们开发了一种新颖的基于帧的事件表示方法，以提高结构清晰度。此外，由于观察到的地面真实姿态偏差程度不一，我们设计了一个模块来预测辅助变量作为正则项，以减轻这种偏差对网络收敛的影响。在多个公开数据集上的实验结果表明了我们所提出方法的有效性，并为未来研究提供了代码和预训练模型。 

---
# CNSv2: Probabilistic Correspondence Encoded Neural Image Servo 

**Title (ZH)**: CNSv2: 聚合概率对应关系的神经图像伺服 

**Authors**: Anzhe Chen, Hongxiang Yu, Shuxin Li, Yuxi Chen, Zhongxiang Zhou, Wentao Sun, Rong Xiong, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00132)  

**Abstract**: Visual servo based on traditional image matching methods often requires accurate keypoint correspondence for high precision control. However, keypoint detection or matching tends to fail in challenging scenarios with inconsistent illuminations or textureless objects, resulting significant performance degradation. Previous approaches, including our proposed Correspondence encoded Neural image Servo policy (CNS), attempted to alleviate these issues by integrating neural control strategies. While CNS shows certain improvement against error correspondence over conventional image-based controllers, it could not fully resolve the limitations arising from poor keypoint detection and matching. In this paper, we continue to address this problem and propose a new solution: Probabilistic Correspondence Encoded Neural Image Servo (CNSv2). CNSv2 leverages probabilistic feature matching to improve robustness in challenging scenarios. By redesigning the architecture to condition on multimodal feature matching, CNSv2 achieves high precision, improved robustness across diverse scenes and runs in real-time. We validate CNSv2 with simulations and real-world experiments, demonstrating its effectiveness in overcoming the limitations of detector-based methods in visual servo tasks. 

**Abstract (ZH)**: 基于传统图像匹配方法的视觉伺服往往需要准确的关键点对应以实现高精度控制。然而，在不一致光照或无纹理物体等具有挑战性的场景中，关键点检测或匹配往往会失败，导致性能显著下降。先前的方法，包括我们提出的编码对应神经图像伺服策略（CNS），试图通过集成神经控制策略来缓解这些问题。尽管CNS在传统基于图像的控制器上对错误对应具有一定的改善，但它无法完全解决由不良关键点检测和匹配引起的限制。在本文中，我们继续解决这一问题并提出一种新的解决方案：概率编码神经图像伺服（CNSv2）。CNSv2利用概率特征匹配以提高在具有挑战性场景中的鲁棒性。通过重新设计架构以基于多模态特征匹配进行条件处理，CNSv2实现了高精度、跨多种场景的增强鲁棒性并能在实时运行。我们通过模拟和真实world实验验证CNSv2，展示了其在视觉伺服任务中克服基于检测的方法限制的有效性。 

---
# CAMETA: Conflict-Aware Multi-Agent Estimated Time of Arrival Prediction for Mobile Robots 

**Title (ZH)**: CAMETA：冲突意识的多机器人预计到达时间预测模型 

**Authors**: Jonas le Fevre Sejersen, Erdal Kayacan  

**Link**: [PDF](https://arxiv.org/pdf/2503.00074)  

**Abstract**: This study presents the conflict-aware multi-agent estimated time of arrival (CAMETA) framework, a novel approach for predicting the arrival times of multiple agents in unstructured environments without predefined road infrastructure. The CAMETA framework consists of three components: a path planning layer generating potential path suggestions, a multi-agent ETA prediction layer predicting the arrival times for all agents based on the paths, and lastly, a path selection layer that calculates the accumulated cost and selects the best path. The novelty of the CAMETA framework lies in the heterogeneous map representation and the heterogeneous graph neural network architecture. As a result of the proposed novel structure, CAMETA improves the generalization capability compared to the state-of-the-art methods that rely on structured road infrastructure and historical data. The simulation results demonstrate the efficiency and efficacy of the multi-agent ETA prediction layer, with a mean average percentage error improvement of 29.5% and 44% when compared to a traditional path planning method (A *) which does not consider conflicts. The performance of the CAMETA framework shows significant improvements in terms of robustness to noise and conflicts as well as determining proficient routes compared to state-of-the-art multi-agent path planners. 

**Abstract (ZH)**: 基于冲突感知的多agent到达时间预测框架（CAMETA）：无预定道路基础设施的不规则环境中的多agent到达时间预测 

---
# Correspondence-Free Pose Estimation with Patterns: A Unified Approach for Multi-Dimensional Vision 

**Title (ZH)**: 无对应关系的位姿估计：一种多维视觉的统一方法 

**Authors**: Quan Quan, Dun Dai  

**Link**: [PDF](https://arxiv.org/pdf/2503.00051)  

**Abstract**: 6D pose estimation is a central problem in robot vision. Compared with pose estimation based on point correspondences or its robust versions, correspondence-free methods are often more flexible. However, existing correspondence-free methods often rely on feature representation alignment or end-to-end regression. For such a purpose, a new correspondence-free pose estimation method and its practical algorithms are proposed, whose key idea is the elimination of unknowns by process of addition to separate the pose estimation from correspondence. By taking the considered point sets as patterns, feature functions used to describe these patterns are introduced to establish a sufficient number of equations for optimization. The proposed method is applicable to nonlinear transformations such as perspective projection and can cover various pose estimations from 3D-to-3D points, 3D-to-2D points, and 2D-to-2D points. Experimental results on both simulation and actual data are presented to demonstrate the effectiveness of the proposed method. 

**Abstract (ZH)**: 无对应点的6D姿态估计是机器人视觉中的一个核心问题。与基于点对应或其稳健版本的姿态估计方法相比，无对应点的方法通常更为灵活。然而，现有的无对应点方法往往依赖于特征表示对齐或端到端回归。为此，提出了一种新的无对应点姿态估计方法及其实际算法，其关键思想是通过加法消除未知数，从而将姿态估计与对应分离。通过将考虑的点集视为模式，引入用于描述这些模式的特征函数以建立足够的方程进行优化。所提出的方法适用于如透视投影等非线性变换，并可以涵盖从3D到3D点、3D到2D点和2D到2D点的各种姿态估计。基于仿真和实际数据的实验结果证明了所提出方法的有效性。 

---
# Observability Investigation for Rotational Calibration of (Global-pose aided) VIO under Straight Line Motion 

**Title (ZH)**: 直行运动下基于全局姿态辅助的VIO旋转校准的可观测性研究 

**Authors**: Junlin Song, Antoine Richard, Miguel Olivares-Mendez  

**Link**: [PDF](https://arxiv.org/pdf/2503.00027)  

**Abstract**: Online extrinsic calibration is crucial for building "power-on-and-go" moving platforms, like robots and AR devices. However, blindly performing online calibration for unobservable parameter may lead to unpredictable results. In the literature, extensive studies have been conducted on the extrinsic calibration between IMU and camera, from theory to practice. It is well-known that the observability of extrinsic parameter can be guaranteed under sufficient motion excitation. Furthermore, the impacts of degenerate motions are also investigated. Despite these successful analyses, we identify an issue regarding the existing observability conclusion. This paper focuses on the observability investigation for straight line motion, which is a common-seen and fundamental degenerate motion in applications. We analytically prove that pure translational straight line motion can lead to the unobservability of the rotational extrinsic parameter between IMU and camera (at least one degree of freedom). By correcting observability conclusion, our novel theoretical finding disseminate more precise principle to the research community and provide explainable calibration guideline for practitioners. Our analysis is validated by rigorous theory and experiments. 

**Abstract (ZH)**: 在线外参标定对于构建“即开即用”移动平台（如机器人和AR设备）至关重要。然而，盲目进行不可观测参数的在线标定可能产生不可预测的结果。在文献中，IMU与相机之间外参标定的理论与实践研究已十分广泛。已知在充分的运动激励下，外参的可观测性可以得到保证。此外，退化运动的影响也得到了研究。尽管已有这些成功的分析，我们识别出现有可观测性结论中存在一个问题。本文专注于直线运动的可观测性研究，这是应用中常见且基本的退化运动。我们从理论上证明，纯平移直线运动会导致IMU与相机之间的旋转外参至少在一个自由度上的不可观测性。通过修正可观测性结论，我们的新理论发现向研究界传播了更加精确的原则，并为实践者提供了可解释的标定指南。我们的分析通过严格的理论与实验得到了验证。 

---
