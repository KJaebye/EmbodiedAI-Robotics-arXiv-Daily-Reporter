# MLA: A Multisensory Language-Action Model for Multimodal Understanding and Forecasting in Robotic Manipulation 

**Title (ZH)**: MLA：一种多感官语言-动作模型，用于机器人操作中的多模态理解和预测。 

**Authors**: Zhuoyang Liu, Jiaming Liu, Jiadong Xu, Nuowei Han, Chenyang Gu, Hao Chen, Kaichen Zhou, Renrui Zhang, Kai Chin Hsieh, Kun Wu, Zhengping Che, Jian Tang, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.26642)  

**Abstract**: Vision-language-action models (VLAs) have shown generalization capabilities in robotic manipulation tasks by inheriting from vision-language models (VLMs) and learning action generation. Most VLA models focus on interpreting vision and language to generate actions, whereas robots must perceive and interact within the spatial-physical world. This gap highlights the need for a comprehensive understanding of robotic-specific multisensory information, which is crucial for achieving complex and contact-rich control. To this end, we introduce a multisensory language-action (MLA) model that collaboratively perceives heterogeneous sensory modalities and predicts future multisensory objectives to facilitate physical world modeling. Specifically, to enhance perceptual representations, we propose an encoder-free multimodal alignment scheme that innovatively repurposes the large language model itself as a perception module, directly interpreting multimodal cues by aligning 2D images, 3D point clouds, and tactile tokens through positional correspondence. To further enhance MLA's understanding of physical dynamics, we design a future multisensory generation post-training strategy that enables MLA to reason about semantic, geometric, and interaction information, providing more robust conditions for action generation. For evaluation, the MLA model outperforms the previous state-of-the-art 2D and 3D VLA methods by 12% and 24% in complex, contact-rich real-world tasks, respectively, while also demonstrating improved generalization to unseen configurations. Project website: this https URL 

**Abstract (ZH)**: 视觉-语言-动作模型（VLAs）通过继承视觉-语言模型（VLMs）的能力并在学习动作生成的同时展示出在机器人操作任务中的泛化能力。大多数VLA模型专注于通过解释视觉和语言来生成动作，而机器人必须感知并与其所处的三维物理世界进行互动。这一差距突显了对特定于机器人多模态信息的全面理解的需求，这对于实现复杂且接触密集的控制至关重要。为此，我们提出了一种多感官语言-动作（MLA）模型，该模型能够协同感知异构的感官模态，并预测未来的多感官目标，以促进物理世界的建模。具体来说，为了增强感知表示，我们提出了一种无需编码器的多模态对齐方案，通过位置对应关系直接利用大型语言模型来解释2D图像、3D点云和触觉标记等多模态提示。为进一步增强MLA对物理动力学的理解，我们设计了一种未来多感官生成后训练策略，使MLA能够在语义、几何和交互信息方面进行推理，从而为动作生成提供更稳健的条件。在评估中，MLA模型在复杂且接触密集的现实世界任务中分别比之前最先进的2D和3D VLA方法展示了12%和24%的性能提升，同时在未见过的配置方面也展示了更好的泛化能力。项目网站：this https URL 

---
# OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction 

**Title (ZH)**: 全方位重定位：保留交互的人形全身动操作与场景交互数据生成 

**Authors**: Lujie Yang, Xiaoyu Huang, Zhen Wu, Angjoo Kanazawa, Pieter Abbeel, Carmelo Sferrazza, C. Karen Liu, Rocky Duan, Guanya Shi  

**Link**: [PDF](https://arxiv.org/pdf/2509.26633)  

**Abstract**: A dominant paradigm for teaching humanoid robots complex skills is to retarget human motions as kinematic references to train reinforcement learning (RL) policies. However, existing retargeting pipelines often struggle with the significant embodiment gap between humans and robots, producing physically implausible artifacts like foot-skating and penetration. More importantly, common retargeting methods neglect the rich human-object and human-environment interactions essential for expressive locomotion and loco-manipulation. To address this, we introduce OmniRetarget, an interaction-preserving data generation engine based on an interaction mesh that explicitly models and preserves the crucial spatial and contact relationships between an agent, the terrain, and manipulated objects. By minimizing the Laplacian deformation between the human and robot meshes while enforcing kinematic constraints, OmniRetarget generates kinematically feasible trajectories. Moreover, preserving task-relevant interactions enables efficient data augmentation, from a single demonstration to different robot embodiments, terrains, and object configurations. We comprehensively evaluate OmniRetarget by retargeting motions from OMOMO, LAFAN1, and our in-house MoCap datasets, generating over 8-hour trajectories that achieve better kinematic constraint satisfaction and contact preservation than widely used baselines. Such high-quality data enables proprioceptive RL policies to successfully execute long-horizon (up to 30 seconds) parkour and loco-manipulation skills on a Unitree G1 humanoid, trained with only 5 reward terms and simple domain randomization shared by all tasks, without any learning curriculum. 

**Abstract (ZH)**: 一种保留交互的全维肢体重定时方法：基于交互网格模型在强化学习中的应用 

---
# Graphite: A GPU-Accelerated Mixed-Precision Graph Optimization Framework 

**Title (ZH)**: Graphite：一种GPU加速的混合精度图优化框架 

**Authors**: Shishir Gopinath, Karthik Dantu, Steven Y. Ko  

**Link**: [PDF](https://arxiv.org/pdf/2509.26581)  

**Abstract**: We present Graphite, a GPU-accelerated nonlinear graph optimization framework. It provides a CUDA C++ interface to enable the sharing of code between a realtime application, such as a SLAM system, and its optimization tasks. The framework supports techniques to reduce memory usage, including in-place optimization, support for multiple floating point types and mixed-precision modes, and dynamically computed Jacobians. We evaluate Graphite on well-known bundle adjustment problems and find that it achieves similar performance to MegBA, a solver specialized for bundle adjustment, while maintaining generality and using less memory. We also apply Graphite to global visual-inertial bundle adjustment on maps generated from stereo-inertial SLAM datasets, and observe speed ups of up to 59x compared to a CPU baseline. Our results indicate that our solver enables faster large-scale optimization on both desktop and resource-constrained devices. 

**Abstract (ZH)**: 我们介绍Graphite：一种基于GPU的非线性图优化框架 

---
# Radio-based Multi-Robot Odometry and Relative Localization 

**Title (ZH)**: 基于无线电的多机器人里程计与相对定位 

**Authors**: Andrés Martínez-Silva, David Alejo, Luis Merino, Fernando Caballero  

**Link**: [PDF](https://arxiv.org/pdf/2509.26558)  

**Abstract**: Radio-based methods such as Ultra-Wideband (UWB) and RAdio Detection And Ranging (radar), which have traditionally seen limited adoption in robotics, are experiencing a boost in popularity thanks to their robustness to harsh environmental conditions and cluttered environments. This work proposes a multi-robot UGV-UAV localization system that leverages the two technologies with inexpensive and readily-available sensors, such as Inertial Measurement Units (IMUs) and wheel encoders, to estimate the relative position of an aerial robot with respect to a ground robot. The first stage of the system pipeline includes a nonlinear optimization framework to trilaterate the location of the aerial platform based on UWB range data, and a radar pre-processing module with loosely coupled ego-motion estimation which has been adapted for a multi-robot scenario. Then, the pre-processed radar data as well as the relative transformation are fed to a pose-graph optimization framework with odometry and inter-robot constraints. The system, implemented for the Robotic Operating System (ROS 2) with the Ceres optimizer, has been validated in Software-in-the-Loop (SITL) simulations and in a real-world dataset. The proposed relative localization module outperforms state-of-the-art closed-form methods which are less robust to noise. Our SITL environment includes a custom Gazebo plugin for generating realistic UWB measurements modeled after real data. Conveniently, the proposed factor graph formulation makes the system readily extensible to full Simultaneous Localization And Mapping (SLAM). Finally, all the code and experimental data is publicly available to support reproducibility and to serve as a common open dataset for benchmarking. 

**Abstract (ZH)**: 基于无线电波的方法，如超宽带（UWB）和雷达（Radar），尽管在机器人领域传统上应用有限，但得益于其在恶劣环境和复杂环境中的鲁棒性，这些技术正逐渐流行起来。本文提出了一种多机器人UGV-UAV localization系统，利用这些技术以及低成本易获得的传感器（如惯性测量单元（IMU）和轮码盘），以估计空中机器人相对于地面机器人的相对位置。系统管道的第一阶段包括一个非线性优化框架，用于基于UWB范围数据进行三角测量，并包括一个适用于多机器人场景的雷达预处理模块，该模块与自我运动估计松散耦合。然后，预处理后的雷达数据以及相对变换被输入一个包含里程计和机器人间约束的姿态图优化框架。该系统基于Robotic Operating System（ROS 2）并通过Ceres优化器实现，并已在Software-in-the-Loop（SITL）仿真和真实世界数据集上得到了验证。所提出的相对定位模块在抗噪性方面优于最先进的闭式解方法。我们的SITL环境包含一个根据真实数据生成真实UWB测量值的自定义Gazebo插件。此外，所提出的因子图表示使系统易于扩展到完整的Simultaneous Localization And Mapping（SLAM）。最后，所有代码和实验数据均公开，以支持可再现性，并作为基准测试的共同开放数据集。 

---
# Memory-Efficient 2D/3D Shape Assembly of Robot Swarms 

**Title (ZH)**: 机器人 swarm 的高效 2D/3D 形状装配 

**Authors**: Shuoyu Yue, Pengpeng Li, Yang Xu, Kunrui Ze, Xingjian Long, Huazi Cao, Guibin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.26518)  

**Abstract**: Mean-shift-based approaches have recently emerged as the most effective methods for robot swarm shape assembly tasks. These methods rely on image-based representations of target shapes to compute local density gradients and perform mean-shift exploration, which constitute their core mechanism. However, such image representations incur substantial memory overhead, which can become prohibitive for high-resolution or 3D shapes. To overcome this limitation, we propose a memory-efficient tree map representation that hierarchically encodes user-specified shapes and is applicable to both 2D and 3D scenarios. Building on this representation, we design a behavior-based distributed controller that enables assignment-free shape assembly. Comparative 2D and 3D simulations against a state-of-the-art mean-shift algorithm demonstrate one to two orders of magnitude lower memory usage and two to three times faster shape entry while maintaining comparable uniformity. Finally, we validate the framework through physical experiments with 6 to 7 UAVs, confirming its real-world practicality. 

**Abstract (ZH)**: 基于均值漂移的方法近年来成为机器人 swarm 形状组装任务中最有效的手段。这些方法依赖于目标形状的图像表示来计算局部密度梯度并执行均值漂移探索，这是其核心机制。然而，这种图像表示会带来巨大的内存开销，对于高分辨率或3D形状可能成为限制因素。为克服这一限制，我们提出了一种内存高效的树状图表示方法，该方法可以分层编码用户指定的形状，并适用于2D和3D场景。基于此表示方法，我们设计了一种基于行为的分布式控制器，以实现无需分配的形状组装。与最先进的均值漂移算法的2D和3D仿真结果表明，该方法的内存使用量低一个到两个数量级，形状进入速度提高两到三倍，同时保持了相当均匀性。最后，通过使用6到7个无人机的物理实验验证了该框架的有效性，证实了其实用性。 

---
# Learning from Hallucinating Critical Points for Navigation in Dynamic Environments 

**Title (ZH)**: 基于生成关键点进行动态环境中导航的学习 

**Authors**: Saad Abdul Ghani, Kameron Lee, Xuesu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2509.26513)  

**Abstract**: Generating large and diverse obstacle datasets to learn motion planning in environments with dynamic obstacles is challenging due to the vast space of possible obstacle trajecto- ries. Inspired by hallucination-based data synthesis approaches, we propose Learning from Hallucinating Critical Points (LfH- CP), a self-supervised framework for creating rich dynamic ob- stacle datasets based on existing optimal motion plans without requiring expensive expert demonstrations or trial-and-error exploration. LfH-CP factorizes hallucination into two stages: first identifying when and where obstacles must appear in order to result in an optimal motion plan, i.e., the critical points, and then procedurally generating diverse trajectories that pass through these points while avoiding collisions. This factorization avoids generative failures such as mode collapse and ensures coverage of diverse dynamic behaviors. We further introduce a diversity metric to quantify dataset richness and show that LfH-CP produces substantially more varied training data than existing baselines. Experiments in simulation demonstrate that planners trained on LfH-CP datasets achieves higher success rates compared to a prior hallucination method. 

**Abstract (ZH)**: 利用幻视关键点学习（LfH-CR）自监督框架生成丰富多样的动态障碍数据集以学习具有动态障碍环境中的运动规划 

---
# Analytic Conditions for Differentiable Collision Detection in Trajectory Optimization 

**Title (ZH)**: 可微碰撞检测在轨迹优化中的分析条件 

**Authors**: Akshay Jaitly, Devesh K. Jha, Kei Ota, Yuki Shirai  

**Link**: [PDF](https://arxiv.org/pdf/2509.26459)  

**Abstract**: Optimization-based methods are widely used for computing fast, diverse solutions for complex tasks such as collision-free movement or planning in the presence of contacts. However, most of these methods require enforcing non-penetration constraints between objects, resulting in a non-trivial and computationally expensive problem. This makes the use of optimization-based methods for planning and control challenging. In this paper, we present a method to efficiently enforce non-penetration of sets while performing optimization over their configuration, which is directly applicable to problems like collision-aware trajectory optimization. We introduce novel differentiable conditions with analytic expressions to achieve this. To enforce non-collision between non-smooth bodies using these conditions, we introduce a method to approximate polytopes as smooth semi-algebraic sets. We present several numerical experiments to demonstrate the performance of the proposed method and compare the performance with other baseline methods recently proposed in the literature. 

**Abstract (ZH)**: 基于优化的方法广泛用于计算复杂任务（如碰撞免费移动或接触条件下的规划）的快速和多样化解决方案。然而，这些方法大多需要在物体之间施加非穿透约束，导致问题非平凡且计算成本高昂。这使得基于优化的方法在规划与控制中的应用具有挑战性。本文提出了一种方法，可以在优化配置的同时高效地施加非穿透约束，该方法直接适用于碰撞感知轨迹优化等问题。我们引入了新颖的可微条件及其解析表达式以实现这一目标。为了使用这些条件来近似非光滑体之间的无碰撞，我们提出了一种将多面体近似为光滑半代数集的方法。我们通过多个数值实验展示了所提出方法的性能，并将其性能与其他近期文献中提出的基线方法进行了比较。 

---
# Unwinding Rotations Reduces VR Sickness in Nonsimulated Immersive Telepresence 

**Title (ZH)**: 解开旋转减少非模拟沉浸式远程呈现中的VR恶心感 

**Authors**: Filip Kulisiewicz, Basak Sakcak, Evan G. Center, Juho Kalliokoski, Katherine J. Mimnaugh, Steven M. LaValle, Timo Ojala  

**Link**: [PDF](https://arxiv.org/pdf/2509.26439)  

**Abstract**: Immersive telepresence, when a user views the video stream of a $360^\circ$ camera in a remote environment using a Head Mounted Display (HMD), has great potential to improve the sense of being in a remote environment. In most cases of immersive robotic telepresence, the camera is mounted on a mobile robot which increases the portion of the environment that the remote user can explore. However, robot motions can induce unpleasant symptoms associated with Virtual Reality (VR) sickness, degrading the overall user experience. Previous research has shown that unwinding the rotations of the robot, that is, decoupling the rotations that the camera undergoes due to robot motions from what is seen by the user, can increase user comfort and reduce VR sickness. However, that work considered a virtual environment and a simulated robot. In this work, to test whether the same hypotheses hold when the video stream from a real camera is used, we carried out a user study $(n=36)$ in which the unwinding rotations method was compared against coupled rotations in a task completed through a panoramic camera mounted on a robotic arm. Furthermore, within an inspection task which involved translations and rotations in three dimensions, we tested whether unwinding the robot rotations impacted the performance of users. The results show that the users found the unwinding rotations method to be more comfortable and preferable, and that a reduced level of VR sickness can be achieved without a significant impact on task performance. 

**Abstract (ZH)**: 基于头戴式显示的浸入式远程 presence，当用户使用360°摄像头在远程环境中的视频流时，具有潜在能力提升远程环境的沉浸感。在大多数沉浸式机器人 presence 情景中，摄像头安装在移动机器人上，增加了远程用户可以探索的环境比例。然而，机器人运动可能会引起与虚拟现实 (VR) 不适相关的症状，降低整体用户体验。先前的研究表明，通过解开由于机器人运动引起的摄像头旋转与用户所见之间的耦合，可以增加用户舒适度并减少 VR 不适。然而，该工作考虑的是虚拟环境和模拟机器人。在此项工作中，为了测试当使用实际摄像头的视频流时，相同假设是否仍然成立，我们在完成通过机械臂上全景摄像头的任务中进行了用户研究（n=36），比较了解开旋转方法与耦合旋转方法。此外，在一项涉及三维平移和旋转的检测任务中，我们测试了解开机器人旋转是否影响用户性能。研究结果表明，用户认为解开旋转方法更舒适且更优，并且可以在不影响任务性能的情况下实现较低水平的 VR 不适。 

---
# Real-time Velocity Profile Optimization for Time-Optimal Maneuvering with Generic Acceleration Constraints 

**Title (ZH)**: 基于通用加速度约束的实时速度剖面优化以实现时间最优机动 

**Authors**: Mattia Piazza, Mattia Piccinini, Sebastiano Taddei, Francesco Biral, Enrico Bertolazzi  

**Link**: [PDF](https://arxiv.org/pdf/2509.26428)  

**Abstract**: The computation of time-optimal velocity profiles along prescribed paths, subject to generic acceleration constraints, is a crucial problem in robot trajectory planning, with particular relevance to autonomous racing. However, the existing methods either support arbitrary acceleration constraints at high computational cost or use conservative box constraints for computational efficiency. We propose FBGA, a new \underline{F}orward-\underline{B}ackward algorithm with \underline{G}eneric \underline{A}cceleration constraints, which achieves both high accuracy and low computation time. FBGA operates forward and backward passes to maximize the velocity profile in short, discretized path segments, while satisfying user-defined performance limits. Tested on five racetracks and two vehicle classes, FBGA handles complex, non-convex acceleration constraints with custom formulations. Its maneuvers and lap times closely match optimal control baselines (within $0.11\%$-$0.36\%$), while being up to three orders of magnitude faster. FBGA maintains high accuracy even with coarse discretization, making it well-suited for online multi-query trajectory planning. Our open-source \texttt{C++} implementation is available at: this https URL. 

**Abstract (ZH)**: 沿指定路径在通用加速度约束下的时间最优速度剖面计算是机器人轨迹规划中的关键问题，特别是在自主赛车领域尤为重要。然而，现有的方法要么在高计算成本下支持任意加速度约束，要么为了计算效率使用保守的盒状约束。我们提出了一种新的FBGA（Forward-Backward算法，通用加速度约束），在保证高精度的同时，实现了低计算时间。FBGA通过正向和反向 passes 在短的离散路径段中最大化速度剖面，同时满足用户定义的性能限制。在五个赛道和两类车辆上测试，FBGA能够处理由自定义公式表示的复杂非凸加速度约束，并且其操作和圈速与最优控制基准（在0.11%-0.36%以内）极为接近，同时比后者快三个数量级。即使离散度粗糙，FBGA也能保持高精度，使其适合在线多查询轨迹规划。我们的开源C++实现可在以下链接获取：this https URL。 

---
# SDA-PLANNER: State-Dependency Aware Adaptive Planner for Embodied Task Planning 

**Title (ZH)**: SDA-PLANNER: awareness of State-Dependency Adaptive Planner for Embodied Task Planning 

**Authors**: Zichao Shen, Chen Gao, Jiaqi Yuan, Tianchen Zhu, Xingcheng Fu, Qingyun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.26375)  

**Abstract**: Embodied task planning requires agents to produce executable actions in a close-loop manner within the environment. With progressively improving capabilities of LLMs in task decomposition, planning, and generalization, current embodied task planning methods adopt LLM-based this http URL, existing LLM-based planners remain limited in three aspects, i.e., fixed planning paradigms, lack of action sequence constraints, and error-agnostic. In this work, we propose SDA-PLANNER, enabling an adaptive planning paradigm, state-dependency aware and error-aware mechanisms for comprehensive embodied task planning. Specifically, SDA-PLANNER introduces a State-Dependency Graph to explicitly model action preconditions and effects, guiding the dynamic revision. To handle execution error, it employs an error-adaptive replanning strategy consisting of Error Backtrack and Diagnosis and Adaptive Action SubTree Generation, which locally reconstructs the affected portion of the plan based on the current environment state. Experiments demonstrate that SDA-PLANNER consistently outperforms baselines in success rate and goal completion, particularly under diverse error conditions. 

**Abstract (ZH)**: 基于LLM的实体任务规划要求代理在环境中以闭环方式产生可执行的动作。尽管LLM在任务分解、规划和泛化方面的逐步增强使其成为当前实体任务规划方法的核心，现有的基于LLM的规划器仍存在固定规划范式、缺少动作序列约束以及错误无感知等局限。本文提出SDA-PLANNER，使其能够实现适应性规划范式，并通过状态依赖性和错误感知机制进行全面优化。具体而言，SDA-PLANNER引入状态依赖图来明确建模动作前提和效果，指导动态修订。为处理执行错误，它采用了错误自适应重规划策略，包括错误回溯和诊断以及自适应动作子树生成，该策略基于当前环境状态局部重构受影响部分的计划。实验表明，SDA-PLANNER在成功率和目标完成方面始终优于基线，并且在多种错误条件下表现出色。 

---
# Kinodynamic Motion Planning for Mobile Robot Navigation across Inconsistent World Models 

**Title (ZH)**: 跨不一致世界模型的移动机器人Kinodynamic运动规划 

**Authors**: Eric R. Damm, Thomas M. Howard  

**Link**: [PDF](https://arxiv.org/pdf/2509.26339)  

**Abstract**: Mobile ground robots lacking prior knowledge of an environment must rely on sensor data to develop a model of their surroundings. In these scenarios, consistent identification of obstacles and terrain features can be difficult due to noise and algorithmic shortcomings, which can make it difficult for motion planning systems to generate safe motions. One particular difficulty to overcome is when regions of the cost map switch between being marked as obstacles and free space through successive planning cycles. One potential solution to this, which we refer to as Valid in Every Hypothesis (VEH), is for the planning system to plan motions that are guaranteed to be safe through a history of world models. Another approach is to track a history of world models, and adjust node costs according to the potential penalty of needing to reroute around previously hazardous areas. This work discusses three major iterations on this idea. The first iteration, called PEH, invokes a sub-search for every node expansion that crosses through a divergence point in the world models. The second and third iterations, called GEH and GEGRH respectively, defer the sub-search until after an edge expands into the goal region. GEGRH uses an additional step to revise the graph based on divergent nodes in each world. Initial results showed that, although PEH and GEH find more optimistic solutions than VEH, they are unable to generate solutions in less than one-second, which exceeds our requirements for field deployment. Analysis of results from a field experiment in an unstructured, off-road environment on a Clearpath Robotics Warthog UGV indicate that GEGRH finds lower cost trajectories and has faster average planning times than VEH. Compared to single-hypothesis (SH) search, where only the latest world model is considered, GEGRH generates more conservative plans with a small increase in average planning time. 

**Abstract (ZH)**: 基于每个假设有效的移动地面机器人规划方法：GEGRH 

---
# LLM-MCoX: Large Language Model-based Multi-robot Coordinated Exploration and Search 

**Title (ZH)**: 基于大型语言模型的多机器人协调探索与搜索 

**Authors**: Ruiyang Wang, Haolun Tsu, David Hunt, Shaocheng Luo, Jiwoo Kim, Miroslav Pajic  

**Link**: [PDF](https://arxiv.org/pdf/2509.26324)  

**Abstract**: Autonomous exploration and object search in unknown indoor environments remain challenging for multi-robot systems (MRS). Traditional approaches often rely on greedy frontier assignment strategies with limited inter-robot coordination. In this work, we introduce LLM-MCoX (LLM-based Multi-robot Coordinated Exploration and Search), a novel framework that leverages Large Language Models (LLMs) for intelligent coordination of both homogeneous and heterogeneous robot teams tasked with efficient exploration and target object search. Our approach combines real-time LiDAR scan processing for frontier cluster extraction and doorway detection with multimodal LLM reasoning (e.g., GPT-4o) to generate coordinated waypoint assignments based on shared environment maps and robot states. LLM-MCoX demonstrates superior performance compared to existing methods, including greedy and Voronoi-based planners, achieving 22.7% faster exploration times and 50% improved search efficiency in large environments with 6 robots. Notably, LLM-MCoX enables natural language-based object search capabilities, allowing human operators to provide high-level semantic guidance that traditional algorithms cannot interpret. 

**Abstract (ZH)**: 基于大型语言模型的多机器人协同探索与搜索（LLM-MCoX）：在未知室内环境中的自主探索和目标搜索 

---
# Anomaly detection for generic failure monitoring in robotic assembly, screwing and manipulation 

**Title (ZH)**: 机器人装配、拧紧和操作中通用故障监测的异常检测 

**Authors**: Niklas Grambow, Lisa-Marie Fenner, Felipe Kempkes, Philip Hotz, Dingyuan Wan, Jörg Krüger, Kevin Haninger  

**Link**: [PDF](https://arxiv.org/pdf/2509.26308)  

**Abstract**: Out-of-distribution states in robot manipulation often lead to unpredictable robot behavior or task failure, limiting success rates and increasing risk of damage. Anomaly detection (AD) can identify deviations from expected patterns in data, which can be used to trigger failsafe behaviors and recovery strategies. Prior work has applied data-driven AD to time series data in specific robotic tasks, but its transferability across control strategies and task types has not been shown. Leveraging time series data, such as force/torque signals, allows to directly capture robot-environment interactions, crucial for manipulation and online failure detection. Their broad availability, high sampling rates, and low dimensionality enable high temporal resolution and efficient processing. As robotic tasks can have widely signal characteristics and requirements, AD methods which can be applied in the same way to a wide range of tasks is needed, ideally with good data efficiency. We examine three industrial robotic tasks, each presenting several anomalies. Test scenarios in robotic cabling, screwing, and sanding are built, and multimodal time series data is gathered. Several autoencoder-based methods are compared, evaluating generalization across tasks and control methods (diffusion policy, position, and impedance control). This allows us to validate the integration of AD in complex tasks involving tighter tolerances and variation from both the robot and its environment. Additionally, we evaluate data efficiency, detection latency, and task characteristics which support robust detection. The results indicate reliable detection with AUROC exceeding 0.93 in failures in the cabling and screwing task, such as incorrect or misaligned parts and obstructed targets. In the polishing task, only severe failures were reliably detected, while more subtle failure types remained undetected. 

**Abstract (ZH)**: 机器人操作中的离分布状态往往会导致不可预测的机器人行为或任务失败，限制了成功率并增加了损坏风险。异常检测（AD）可以通过识别数据中的偏差来触发故障安全行为和恢复策略。以往的工作已经在特定的机器人任务中的时间序列数据上应用了数据驱动的AD，但其在控制策略和任务类型之间的可转移性尚未得到验证。通过利用如力/扭矩信号的时间序列数据，可以直接捕获机器人与环境的相互作用，这对于操作和在线故障检测至关重要。这些数据的广泛可获得性、高采样率和低维度使得时间分辨率高且处理效率高。由于机器人任务可以具有广泛的不同信号特征和要求，需要一种可以在多种任务中以相同方式应用且具有良好的数据效率的AD方法，理想情况下效果良好。我们研究了三个工业机器人任务，每个任务都表现出多种异常。我们构建了在电缆装配、螺丝紧固和打磨中的测试场景，并收集了多模态时间序列数据。比较了几种基于自编码器的方法，评估了其在不同任务和控制方法（扩散策略、位置控制和阻抗控制）下的泛化能力。这使我们能够验证AD在涉及更严格公差和来自机器人及其环境的更大变化的复杂任务中的集成。此外，我们还评估了数据效率、检测延迟以及支持稳健检测的任务特性。结果显示，在电缆装配和螺丝紧固任务中的故障中，AUROC超过0.93，能够可靠地检测出诸如错误部件或对齐不良、目标受阻等情况。在抛光任务中，只有严重的故障得到了可靠检测，而更微妙的故障类型仍未被检测到。 

---
# ISyHand: A Dexterous Multi-finger Robot Hand with an Articulated Palm 

**Title (ZH)**: ISyHand: 一个具有articulated palm的灵巧多指机器人手 

**Authors**: Benjamin A. Richardson, Felix Grüninger, Lukas Mack, Joerg Stueckler, Katherine J. Kuchenbecker  

**Link**: [PDF](https://arxiv.org/pdf/2509.26236)  

**Abstract**: The rapid increase in the development of humanoid robots and customized manufacturing solutions has brought dexterous manipulation to the forefront of modern robotics. Over the past decade, several expensive dexterous hands have come to market, but advances in hardware design, particularly in servo motors and 3D printing, have recently facilitated an explosion of cheaper open-source hands. Most hands are anthropomorphic to allow use of standard human tools, and attempts to increase dexterity often sacrifice anthropomorphism. We introduce the open-source ISyHand (pronounced easy-hand), a highly dexterous, low-cost, easy-to-manufacture, on-joint servo-driven robot hand. Our hand uses off-the-shelf Dynamixel motors, fasteners, and 3D-printed parts, can be assembled within four hours, and has a total material cost of about 1,300 USD. The ISyHands's unique articulated-palm design increases overall dexterity with only a modest sacrifice in anthropomorphism. To demonstrate the utility of the articulated palm, we use reinforcement learning in simulation to train the hand to perform a classical in-hand manipulation task: cube reorientation. Our novel, systematic experiments show that the simulated ISyHand outperforms the two most comparable hands in early training phases, that all three perform similarly well after policy convergence, and that the ISyHand significantly outperforms a fixed-palm version of its own design. Additionally, we deploy a policy trained on cube reorientation on the real hand, demonstrating its ability to perform real-world dexterous manipulation. 

**Abstract (ZH)**: 开源易手（ISyHand）：低成本高灵巧性关节驱动机器人手 

---
# Terrain-Awared LiDAR-Inertial Odometry for Legged-Wheel Robots Based on Radial Basis Function Approximation 

**Title (ZH)**: 基于径向基函数近似的地形aware激光雷达-惯性里程计算法及其在足轮机器人上的应用 

**Authors**: Yizhe Liu, Han Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.26222)  

**Abstract**: An accurate odometry is essential for legged-wheel robots operating in unstructured terrains such as bumpy roads and staircases. Existing methods often suffer from pose drift due to their ignorance of terrain geometry. We propose a terrain-awared LiDAR-Inertial odometry (LIO) framework that approximates the terrain using Radial Basis Functions (RBF) whose centers are adaptively selected and weights are recursively updated. The resulting smooth terrain manifold enables ``soft constraints" that regularize the odometry optimization and mitigates the $z$-axis pose drift under abrupt elevation changes during robot's maneuver. To ensure the LIO's real-time performance, we further evaluate the RBF-related terms and calculate the inverse of the sparse kernel matrix with GPU parallelization. Experiments on unstructured terrains demonstrate that our method achieves higher localization accuracy than the state-of-the-art baselines, especially in the scenarios that have continuous height changes or sparse features when abrupt height changes occur. 

**Abstract (ZH)**: 一种适应地形的LiDAR-惯性里程计框架在不规则地形上的应用：该框架通过径向基函数近似地形，并通过自适应选择中心和递归更新权重实现地形aware的里程计优化，从而在高度突变时减少姿态漂移，实现实时性能并提高定位精度。 

---
# Side Scan Sonar-based SLAM for Autonomous Algae Farm Monitoring 

**Title (ZH)**: 侧扫声纳基于SLAM的自主藻类农场监测 

**Authors**: Julian Valdez, Ignacio Torroba, John Folkesson, Ivan Stenius  

**Link**: [PDF](https://arxiv.org/pdf/2509.26121)  

**Abstract**: The transition of seaweed farming to an alternative food source on an industrial scale relies on automating its processes through smart farming, equivalent to land agriculture. Key to this process are autonomous underwater vehicles (AUVs) via their capacity to automate crop and structural inspections. However, the current bottleneck for their deployment is ensuring safe navigation within farms, which requires an accurate, online estimate of the AUV pose and map of the infrastructure. To enable this, we propose an efficient side scan sonar-based (SSS) simultaneous localization and mapping (SLAM) framework that exploits the geometry of kelp farms via modeling structural ropes in the back-end as sequences of individual landmarks from each SSS ping detection, instead of combining detections into elongated representations. Our method outperforms state of the art solutions in hardware in the loop (HIL) experiments on a real AUV survey in a kelp farm. The framework and dataset can be found at this https URL. 

**Abstract (ZH)**: 大型海藻养殖向替代食物来源的转型依赖于通过智能养殖自动化其过程，相当于陆地农业。这一过程的关键是自主水下车辆（AUV）通过其自动化作物和结构检查的能力。然而，目前部署的瓶颈在于确保在农场内安全导航，这需要准确的在线姿态估计和基础设施的地图。为此，我们提出了一种有效的基于侧扫声呐（SSS）的同步定位与 mapping（SLAM）框架，该框架通过在后端将结构绳索建模为每个 SSS 回波检测的个体地标序列，而不是将检测合并为延长的表示，利用了海藻农场的几何结构。我们的方法在真实 AUV 在海藻农场进行的硬件在环（HIL）实验中优于当前最先进的解决方案。该框架和数据集可访问此处：this https URL。 

---
# Autonomous Multi-Robot Infrastructure for AI-Enabled Healthcare Delivery and Diagnostics 

**Title (ZH)**: 自主多机器人基础设施及其在AI赋能的健康护理交付与诊断中的应用 

**Authors**: Nakhul Kalaivanan, Senthil Arumugam Muthukumaraswamy, Girish Balasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2509.26106)  

**Abstract**: This research presents a multi-robot system for inpatient care, designed using swarm intelligence principles and incorporating wearable health sensors, RF-based communication, and AI-driven decision support. Within a simulated hospital environment, the system adopts a leader-follower swarm configuration to perform patient monitoring, medicine delivery, and emergency assistance. Due to ethical constraints, live patient trials were not conducted; instead, validation was carried out through controlled self-testing with wearable sensors. The Leader Robot acquires key physiological parameters, including temperature, SpO2, heart rate, and fall detection, and coordinates other robots when required. The Assistant Robot patrols corridors for medicine delivery, while a robotic arm provides direct drug administration. The swarm-inspired leader-follower strategy enhanced communication reliability and ensured continuous monitoring, including automated email alerts to healthcare staff. The system hardware was implemented using Arduino, Raspberry Pi, NRF24L01 RF modules, and a HuskyLens AI camera. Experimental evaluation showed an overall sensor accuracy above 94%, a 92% task-level success rate, and a 96% communication reliability rate, demonstrating system robustness. Furthermore, the AI-enabled decision support was able to provide early warnings of abnormal health conditions, highlighting the potential of the system as a cost-effective solution for hospital automation and patient safety. 

**Abstract (ZH)**: 基于 swarm intelligence 的穿戴健康传感器多机器人病房护理系统及其实验评估 

---
# Evolutionary Continuous Adaptive RL-Powered Co-Design for Humanoid Chin-Up Performance 

**Title (ZH)**: 基于进化连续自适应RL的类人引体向上性能协同设计 

**Authors**: Tianyi Jin, Melya Boukheddimi, Rohit Kumar, Gabriele Fadini, Frank Kirchner  

**Link**: [PDF](https://arxiv.org/pdf/2509.26082)  

**Abstract**: Humanoid robots have seen significant advancements in both design and control, with a growing emphasis on integrating these aspects to enhance overall performance. Traditionally, robot design has followed a sequential process, where control algorithms are developed after the hardware is finalized. However, this can be myopic and prevent robots to fully exploit their hardware capabilities. Recent approaches advocate for co-design, optimizing both design and control in parallel to maximize robotic capabilities. This paper presents the Evolutionary Continuous Adaptive RL-based Co-Design (EA-CoRL) framework, which combines reinforcement learning (RL) with evolutionary strategies to enable continuous adaptation of the control policy to the hardware. EA-CoRL comprises two key components: Design Evolution, which explores the hardware choices using an evolutionary algorithm to identify efficient configurations, and Policy Continuous Adaptation, which fine-tunes a task-specific control policy across evolving designs to maximize performance rewards. We evaluate EA-CoRL by co-designing the actuators (gear ratios) and control policy of the RH5 humanoid for a highly dynamic chin-up task, previously unfeasible due to actuator limitations. Comparative results against state-of-the-art RL-based co-design methods show that EA-CoRL achieves higher fitness score and broader design space exploration, highlighting the critical role of continuous policy adaptation in robot co-design. 

**Abstract (ZH)**: humanoid机器人在设计和控制方面的进展显著，对这两者的整合越来越受到重视，以提升整体性能。传统上，机器人设计遵循一个顺序过程，即在硬件最终确定后才开发控制算法。然而，这种方法可能过于狭隘，限制了机器人充分利用其硬件能力。近年来，协同设计方法倡导并行优化设计和控制，以最大化机器人能力。本文提出了一种结合强化学习（RL）与进化策略的进化连续自适应RL基于协同设计（EA-CoRL）框架，该框架能够使控制策略连续适应硬件。EA-CoRL包括两个关键组件：设计进化，利用进化算法探索硬件选择以识别高效配置，以及策略连续适应，针对进化中的设计微调特定任务的控制策略以最大化性能奖励。我们通过协同设计RH5人形机器人的执行器（传动比）和控制策略，对一个此前因执行器限制而不可行的高度动态引体向上任务进行了评估。与最先进的基于RL的协同设计方法相比，EA-CoRL显示出更高的适应度评分和更广泛的硬件配置探索范围，突出了连续策略适应在机器人协同设计中的重要作用。 

---
# Conflict-Based Search and Prioritized Planning for Multi-Agent Path Finding Among Movable Obstacles 

**Title (ZH)**: 基于冲突的搜索与优先级规划的多agent路径寻找算法研究 

**Authors**: Shaoli Hu, Shizhe Zhao, Zhongqiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2509.26050)  

**Abstract**: This paper investigates Multi-Agent Path Finding Among Movable Obstacles (M-PAMO), which seeks collision-free paths for multiple agents from their start to goal locations among static and movable obstacles. M-PAMO arises in logistics and warehouses where mobile robots are among unexpected movable objects. Although Multi-Agent Path Finding (MAPF) and single-agent Path planning Among Movable Obstacles (PAMO) were both studied, M-PAMO remains under-explored. Movable obstacles lead to new fundamental challenges as the state space, which includes both agents and movable obstacles, grows exponentially with respect to the number of agents and movable obstacles. In particular, movable obstacles often closely couple agents together spatially and temporally. This paper makes a first attempt to adapt and fuse the popular Conflict-Based Search (CBS) and Prioritized Planning (PP) for MAPF, and a recent single-agent PAMO planner called PAMO*, together to address M-PAMO. We compare their performance with up to 20 agents and hundreds of movable obstacles, and show the pros and cons of these approaches. 

**Abstract (ZH)**: Multi-Agent Path Finding Among Movable Obstacles (M-PAMO) 

---
# On the Conic Complementarity of Planar Contacts 

**Title (ZH)**: 平面接触的圆锥互补性 

**Authors**: Yann de Mont-Marin, Louis Montaut, Jean Ponce, Martial Hebert, Justin Carpentier  

**Link**: [PDF](https://arxiv.org/pdf/2509.25999)  

**Abstract**: We present a unifying theoretical result that connects two foundational principles in robotics: the Signorini law for point contacts, which underpins many simulation methods for preventing object interpenetration, and the center of pressure (also known as the zero-moment point), a key concept used in, for instance, optimization-based locomotion control. Our contribution is the planar Signorini condition, a conic complementarity formulation that models general planar contacts between rigid bodies. We prove that this formulation is equivalent to enforcing the punctual Signorini law across an entire contact surface, thereby bridging the gap between discrete and continuous contact models. A geometric interpretation reveals that the framework naturally captures three physical regimes -sticking, separating, and tilting-within a unified complementarity structure. This leads to a principled extension of the classical center of pressure, which we refer to as the extended center of pressure. By establishing this connection, our work provides a mathematically consistent and computationally tractable foundation for handling planar contacts, with implications for both the accurate simulation of contact dynamics and the design of advanced control and optimization algorithms in locomotion and manipulation. 

**Abstract (ZH)**: 我们提出一个统一的理论结果，将机器人学中的两大基础原则—— SIGNORINI 法则（用于点接触）和中心压力（零力点），即用于防止物体穿插的许多模拟方法的基础，以及关键概念质心压力在比如基于优化的运动控制中的应用——联系起来。我们的贡献是在刚体之间模拟一般平面接触的平面 SIGNORINI 条件——一种锥互补形式。我们证明这种形式等同于在整个接触表面上强制实施 punctual SIGNORINI 法则，从而弥合了离散和连续接触模型之间的差距。几何解释表明，这种框架自然地捕捉了三种物理机制——黏着、分离和倾斜——并在统一的互补结构中。这导致了一个经典质心压力的原理性扩展，我们称之为扩展中心压力。通过建立这种联系，我们的工作为处理平面接触提供了一个数学上一致且计算上可行的基础，这对接触动力学的准确建模以及运动和操作中高级控制和优化算法的设计都有着重要意义。 

---
# Emotionally Expressive Robots: Implications for Children's Behavior toward Robot 

**Title (ZH)**: 情感表达型机器人：对儿童对机器人行为的影响 

**Authors**: Elisabetta Zibetti, Sureya Waheed Palmer, Rebecca Stower, Salvatore M Anzalone  

**Link**: [PDF](https://arxiv.org/pdf/2509.25986)  

**Abstract**: The growing development of robots with artificial emotional expressiveness raises important questions about their persuasive potential in children's behavior. While research highlights the pragmatic value of emotional expressiveness in human social communication, the extent to which robotic expressiveness can or should influence empathic responses in children is grounds for debate. In a pilot study with 22 children (aged 7-11) we begin to explore the ways in which different levels of embodied expressiveness (body only, face only, body and face) of two basic emotions (happiness and sadness) displayed by an anthropomorphic robot (QTRobot) might modify children's behavior in a child-robot cooperative turn-taking game. We observed that children aligned their behavior to the robot's inferred emotional state. However, higher levels of expressiveness did not result in increased alignment. The preliminary results reported here provide a starting point for reflecting on robotic expressiveness and its role in shaping children's social-emotional behavior toward robots as social peers in the near future. 

**Abstract (ZH)**: 随着具有人工情感表达能力的机器人不断发展，它们在儿童行为中潜在的说服力引起了重要问题。虽然研究强调了情感表达在人类社会交流中的实用价值，但机器人的情感表达在多大程度上或是否应该影响儿童的共情反应尚存争议。在一项针对22名儿童（7-11岁）的试点研究中，我们开始探讨不同水平的身体与面部表达（仅身体、仅面部、身体和面部）以及两种基本情绪（快乐和悲伤）由类人机器人（QTRobot）展示时，如何可能改变儿童在儿童-机器人合作轮流游戏中的行为。我们观察到，儿童会调整自己的行为以匹配机器人推断的情感状态。然而，更高的表达水平并未导致更强的匹配度。这里报告的初步结果为反思机器人的表达性和其对未来儿童与机器人作为社会同伴的社会情感行为塑造作用提供了起点。 

---
# S$^3$E: Self-Supervised State Estimation for Radar-Inertial System 

**Title (ZH)**: S$^3$E：雷达-惯性系统自监督状态估计 

**Authors**: Shengpeng Wang, Yulong Xie, Qing Liao, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.25984)  

**Abstract**: Millimeter-wave radar for state estimation is gaining significant attention for its affordability and reliability in harsh conditions. Existing localization solutions typically rely on post-processed radar point clouds as landmark points. Nonetheless, the inherent sparsity of radar point clouds, ghost points from multi-path effects, and limited angle resolution in single-chirp radar severely degrade state estimation performance. To address these issues, we propose S$^3$E, a \textbf{S}elf-\textbf{S}upervised \textbf{S}tate \textbf{E}stimator that employs more richly informative radar signal spectra to bypass sparse points and fuses complementary inertial information to achieve accurate localization. S$^3$E fully explores the association between \textit{exteroceptive} radar and \textit{proprioceptive} inertial sensor to achieve complementary benefits. To deal with limited angle resolution, we introduce a novel cross-fusion technique that enhances spatial structure information by exploiting subtle rotational shift correlations across heterogeneous data. The experimental results demonstrate our method achieves robust and accurate performance without relying on localization ground truth supervision. To the best of our knowledge, this is the first attempt to achieve state estimation by fusing radar spectra and inertial data in a complementary self-supervised manner. 

**Abstract (ZH)**: 毫米波雷达用于状态估计正因其在恶劣条件下的负担能力和可靠性而获得显著关注。传统定位解决方案通常依赖于后期处理的雷达点云作为地标点。然而，雷达点云的固有稀疏性、多路径效应产生的鬼点以及单脉冲雷达有限的角度分辨率严重降低了状态估计性能。为了解决这些问题，我们提出了一种自监督状态估计器S$^3$E，该方法利用更丰富的雷达信号频谱来避开稀疏点，并融合互补的惯性信息以实现精确的定位。S$^3$E充分利用了外部感知雷达与内感知惯性传感器之间的关联，以实现互补效益。为解决有限的角度分辨率问题，我们引入了一种新颖的跨融合技术，通过利用异构数据中细微旋转偏移的相关性来增强空间结构信息。实验结果表明，我们的方法在无需依赖定位地面 truth 监督的情况下实现了坚固而准确的性能。据我们所知，这是首次尝试通过互补的自监督方式融合雷达频谱和惯性数据以实现状态估计的方法。 

---
# MUVLA: Learning to Explore Object Navigation via Map Understanding 

**Title (ZH)**: MUVLA：通过地图理解学习物体导航探索 

**Authors**: Peilong Han, Fan Jia, Min Zhang, Yutao Qiu, Hongyao Tang, Yan Zheng, Tiancai Wang, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2509.25966)  

**Abstract**: In this paper, we present MUVLA, a Map Understanding Vision-Language-Action model tailored for object navigation. It leverages semantic map abstractions to unify and structure historical information, encoding spatial context in a compact and consistent form. MUVLA takes the current and history observations, as well as the semantic map, as inputs and predicts the action sequence based on the description of goal object. Furthermore, it amplifies supervision through reward-guided return modeling based on dense short-horizon progress signals, enabling the model to develop a detailed understanding of action value for reward maximization. MUVLA employs a three-stage training pipeline: learning map-level spatial understanding, imitating behaviors from mixed-quality demonstrations, and reward amplification. This strategy allows MUVLA to unify diverse demonstrations into a robust spatial representation and generate more rational exploration strategies. Experiments on HM3D and Gibson benchmarks demonstrate that MUVLA achieves great generalization and learns effective exploration behaviors even from low-quality or partially successful trajectories. 

**Abstract (ZH)**: 本文介绍了MUVLA，一种针对物体导航定制的时空图理解视觉语言行动模型。该模型利用语义地图抽象来统一和结构化历史信息，以紧凑且一致的形式编码空间上下文。MUVLA 采用当前和历史观察结果以及语义地图作为输入，并基于目标物体的描述预测行动序列。此外，通过基于密集的短期进展信号进行奖励导向的返回建模来增强监督，从而促使模型发展出详细的行动价值理解以实现奖励最大化。MUVLA 采用三阶段训练管道：学习地图级别空间理解、模仿混合质量的演示行为以及奖励放大。这一策略使 MUVLA 能够将多种多样的演示统一为 robust 的空间表示，并生成更合理的探索策略。在 HM3D 和 Gibson 基准测试中，MUVLA 展示了出色的泛化能力，并能够从低质量或部分成功的轨迹中学习有效的探索行为。 

---
# Towards Intuitive Human-Robot Interaction through Embodied Gesture-Driven Control with Woven Tactile Skins 

**Title (ZH)**: 基于编织触感皮肤的体现手势驱动控制以实现直观的人机交互 

**Authors**: ChunPing Lam, Xiangjia Chen, Chenming Wu, Hao Chen, Binzhi Sun, Guoxin Fang, Charlie C.L. Wang, Chengkai Dai, Yeung Yam  

**Link**: [PDF](https://arxiv.org/pdf/2509.25951)  

**Abstract**: This paper presents a novel human-robot interaction (HRI) framework that enables intuitive gesture-driven control through a capacitance-based woven tactile skin. Unlike conventional interfaces that rely on panels or handheld devices, the woven tactile skin integrates seamlessly with curved robot surfaces, enabling embodied interaction and narrowing the gap between human intent and robot response. Its woven design combines fabric-like flexibility with structural stability and dense multi-channel sensing through the interlaced conductive threads. Building on this capability, we define a gesture-action mapping of 14 single- and multi-touch gestures that cover representative robot commands, including task-space motion and auxiliary functions. A lightweight convolution-transformer model designed for gesture recognition in real time achieves an accuracy of near-100%, outperforming prior baseline approaches. Experiments on robot arm tasks, including pick-and-place and pouring, demonstrate that our system reduces task completion time by up to 57% compared with keyboard panels and teach pendants. Overall, our proposed framework demonstrates a practical pathway toward more natural and efficient embodied HRI. 

**Abstract (ZH)**: 基于电容式编织触觉皮肤的新型人机交互框架：直观手势驱动控制及应用 

---
# State Estimation for Compliant and Morphologically Adaptive Robots 

**Title (ZH)**: 顺应性和形态适应性机器人状态估计 

**Authors**: Valentin Yuryev, Max Polzin, Josie Hughes  

**Link**: [PDF](https://arxiv.org/pdf/2509.25945)  

**Abstract**: Locomotion robots with active or passive compliance can show robustness to uncertain scenarios, which can be promising for agricultural, research and environmental industries. However, state estimation for these robots is challenging due to the lack of rigid-body assumptions and kinematic changes from morphing. We propose a method to estimate typical rigid-body states alongside compliance-related states, such as soft robot shape in different morphologies and locomotion modes. Our neural network-based state estimator uses a history of states and a mechanism to directly influence unreliable sensors. We test our framework on the GOAT platform, a robot capable of passive compliance and active morphing for extreme outdoor terrain. The network is trained on motion capture data in a novel compliance-centric frame that accounts for morphing-related states. Our method predicts shape-related measurements within 4.2% of the robot's size, velocities within 6.3% and 2.4% of the top linear and angular speeds, respectively, and orientation within 1.5 degrees. We also demonstrate a 300% increase in travel range during a motor malfunction when using our estimator for closed-loop autonomous outdoor operation. 

**Abstract (ZH)**: 具有主动或被动顺应性的移动机器人可以在不确定场景中表现出色，这有望应用于农业、研究和环境行业。然而，由于缺乏刚体假设和形态变化带来的运动学变化，这些机器人的状态估计具有挑战性。我们提出了一种方法来估计典型的刚体状态以及与顺应性相关的状态，如不同形态和运动模式下的软体机器人形状。基于神经网络的状态估计器使用状态的历史数据并具有一套机制，可以直接影响不可靠的传感器。我们在GOAT平台上测试了该框架，GOAT平台是一种能够在极端户外地形中表现出被动顺应性和主动变形能力的机器人。网络在一种新的顺应性为中心的新框架下进行了训练，该框架考虑了与形态变化相关的状态。我们的方法预测的形状相关测量值的误差在机器人尺寸的4.2%以内，速度分别在最大线性速度和角速度的6.3%和2.4%以内，姿态误差在1.5度以内。我们还展示了当使用我们的估计器进行闭环自主户外操作时，电机故障期间行程范围提高了300%。 

---
# Reinforced Embodied Planning with Verifiable Reward for Real-World Robotic Manipulation 

**Title (ZH)**: 强化体态规划与可验证奖励的世界机器人操作规划 

**Authors**: Zitong Bo, Yue Hu, Jinming Ma, Mingliang Zhou, Junhui Yin, Yachen Kang, Yuqi Liu, Tong Wu, Diyun Xiang, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.25852)  

**Abstract**: Enabling robots to execute long-horizon manipulation tasks from free-form language instructions remains a fundamental challenge in embodied AI. While vision-language models (VLMs) have shown promise as high-level planners, their deployment in the real world is hindered by two gaps: (i) the scarcity of large-scale, sequential manipulation data that couples natural language with multi-step action plans, and (ii) the absence of dense, interpretable rewards for fine-tuning VLMs on planning objectives. To address these issues, we propose REVER, a framework that empowers VLMs to generate and validate long-horizon manipulation plans from natural language instructions in real-world scenarios. Under REVER we train and release RoboFarseer, a VLM incentivized to emit chain-of-thought that perform temporal and spatial reasoning, ensuring physically plausible and logically coherent plans. To obtain training data, we leverage the Universal Manipulation Interface framework to capture hardware-agnostic demonstrations of atomic skills. An automated annotation engine converts each demonstration into vision-instruction-plan triplet. We introduce a verifiable reward that scores the generated plan by its ordered bipartite matching overlap with the ground-truth skill sequence. At run time, the fine-tuned VLM functions both as a planner and as a monitor, verifying step-wise completion. RoboFarseer matches or exceeds the performance of proprietary models that are orders of magnitude larger, while on open-ended planning it surpasses the best baseline by more than 40%. In real-world, long-horizon tasks, the complete system boosts overall success by roughly 60% compared with the same low-level controller without the planner. We will open-source both the dataset and the trained model upon publication. 

**Abstract (ZH)**: 使机器人能够从自然语言指令执行长期 horizon 的 manipulation 任务仍然是嵌入式 AI 中的一项基本挑战。尽管视觉-语言模型 (VLMs) 在高阶规划中显示出了潜力，但在实际部署中受到了两个差距的阻碍：（i）缺乏将自然语言与多步行动计划耦合的大规模序列 manipulation 数据，（ii）缺乏密集的、可解释的奖励来对 VLMs 进行规划目标的微调。为了解决这些问题，我们提出了 REVER 框架，使 VLMs 能够从自然语言指令生成和验证实际场景中的长期 horizon manipulation 计划。在 REVER 框架下，我们训练并发布了 RoboFarseer，一种激励 VLMs 生成执行时间和空间推理的链式思考的 VLM，确保其生成的计划具有物理合理性和逻辑一致性。为获得训练数据，我们利用 Universal Manipulation Interface 框架来捕捉原子技能的硬件无关的演示。自动化注释引擎将每个演示转换为视觉-指令-计划三元组。我们引入了一个可验证的奖励，通过其有序的二分匹配重叠来评分生成的计划与真实技能序列的匹配度。在运行时，微调后的 VLM 同时作为规划者和监控器，验证每一步的完成情况。RoboFarseer 在性能上匹配或超越了规模大几个数量级的专有模型，而在开放性规划中，它超出了最佳基线 40% 以上。在实际场景中执行长期 horizon 任务时，完整系统将总体成功率提升约 60%，相较于相同低级控制器而无规划器的情况。我们在发布时会开源该数据集和训练模型。 

---
# Act to See, See to Act: Diffusion-Driven Perception-Action Interplay for Adaptive Policies 

**Title (ZH)**: 行动以感知，感知以行动：扩散驱动的感知-行动交互对自适应策略的影响 

**Authors**: Jing Wang, Weiting Peng, Jing Tang, Zeyu Gong, Xihua Wang, Bo Tao, Li Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.25822)  

**Abstract**: Existing imitation learning methods decouple perception and action, which overlooks the causal reciprocity between sensory representations and action execution that humans naturally leverage for adaptive behaviors. To bridge this gap, we introduce Action--Guided Diffusion Policy (DP--AG), a unified representation learning that explicitly models a dynamic interplay between perception and action through probabilistic latent dynamics. DP--AG encodes latent observations into a Gaussian posterior via variational inference and evolves them using an action-guided SDE, where the Vector-Jacobian Product (VJP) of the diffusion policy's noise predictions serves as a structured stochastic force driving latent updates. To promote bidirectional learning between perception and action, we introduce a cycle--consistent contrastive loss that organizes the gradient flow of the noise predictor into a coherent perception--action loop, enforcing mutually consistent transitions in both latent updates and action refinements. Theoretically, we derive a variational lower bound for the action-guided SDE, and prove that the contrastive objective enhances continuity in both latent and action trajectories. Empirically, DP--AG significantly outperforms state--of--the--art methods across simulation benchmarks and real-world UR5 manipulation tasks. As a result, our DP--AG offers a promising step toward bridging biological adaptability and artificial policy learning. 

**Abstract (ZH)**: 行动导向扩散策略：通过概率潜动态建模感知与行动的动态互动 

---
# SAC Flow: Sample-Efficient Reinforcement Learning of Flow-Based Policies via Velocity-Reparameterized Sequential Modeling 

**Title (ZH)**: SAC流：基于速度重参数化序贯建模的样本高效流基于策略的强化学习 

**Authors**: Yixian Zhang, Shu'ang Yu, Tonghe Zhang, Mo Guang, Haojia Hui, Kaiwen Long, Yu Wang, Chao Yu, Wenbo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.25756)  

**Abstract**: Training expressive flow-based policies with off-policy reinforcement learning is notoriously unstable due to gradient pathologies in the multi-step action sampling process. We trace this instability to a fundamental connection: the flow rollout is algebraically equivalent to a residual recurrent computation, making it susceptible to the same vanishing and exploding gradients as RNNs. To address this, we reparameterize the velocity network using principles from modern sequential models, introducing two stable architectures: Flow-G, which incorporates a gated velocity, and Flow-T, which utilizes a decoded velocity. We then develop a practical SAC-based algorithm, enabled by a noise-augmented rollout, that facilitates direct end-to-end training of these policies. Our approach supports both from-scratch and offline-to-online learning and achieves state-of-the-art performance on continuous control and robotic manipulation benchmarks, eliminating the need for common workarounds like policy distillation or surrogate objectives. 

**Abstract (ZH)**: 使用离策学习训练表达性强的流动基策略因多步动作采样的梯度病理问题而 notoriously 不稳定。我们将这种不稳定性追溯到一个基本的联系：流动展开在代数上等价于残差递归计算，使其容易受到与递归神经网络相同的梯度消失和梯度爆炸问题。为了解决这一问题，我们借用现代序列模型的原则重新参数化速度网络，引入了两种稳定的架构：Flow-G，其包含门控速度，Flow-T，其利用解码速度。然后，我们开发了一种实用的基于SAC的算法，该算法通过噪声增强的展开过程启用直接端到端训练这些策略。我们的方法支持从头学习和离线到在线学习，并在连续控制和机器人操作基准测试中实现了最先进的性能，消除了如策略蒸馏或替代目标等常见工作-around 解决方法的需要。 

---
# Best of Sim and Real: Decoupled Visuomotor Manipulation via Learning Control in Simulation and Perception in Real 

**Title (ZH)**: 最佳的模拟与现实：通过模拟学习控制与现实感知的解耦视觉与运动 manipulation 

**Authors**: Jialei Huang, Zhaoheng Yin, Yingdong Hu, Shuo Wang, Xingyu Lin, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.25747)  

**Abstract**: Sim-to-real transfer remains a fundamental challenge in robot manipulation due to the entanglement of perception and control in end-to-end learning. We present a decoupled framework that learns each component where it is most reliable: control policies are trained in simulation with privileged state to master spatial layouts and manipulation dynamics, while perception is adapted only at deployment to bridge real observations to the frozen control policy. Our key insight is that control strategies and action patterns are universal across environments and can be learned in simulation through systematic randomization, while perception is inherently domain-specific and must be learned where visual observations are authentic. Unlike existing end-to-end approaches that require extensive real-world data, our method achieves strong performance with only 10-20 real demonstrations by reducing the complex sim-to-real problem to a structured perception alignment task. We validate our approach on tabletop manipulation tasks, demonstrating superior data efficiency and out-of-distribution generalization compared to end-to-end baselines. The learned policies successfully handle object positions and scales beyond the training distribution, confirming that decoupling perception from control fundamentally improves sim-to-real transfer. 

**Abstract (ZH)**: 基于分解框架的从仿真到现实的机器人 manipulation 转移 

---
# TacRefineNet: Tactile-Only Grasp Refinement Between Arbitrary In-Hand Object Poses 

**Title (ZH)**: 基于触觉的手持物体任意姿态下的抓取 refinement 网络 

**Authors**: Shuaijun Wang, Haoran Zhou, Diyun Xiang, Yangwei You  

**Link**: [PDF](https://arxiv.org/pdf/2509.25746)  

**Abstract**: Despite progress in both traditional dexterous grasping pipelines and recent Vision-Language-Action (VLA) approaches, the grasp execution stage remains prone to pose inaccuracies, especially in long-horizon tasks, which undermines overall performance. To address this "last-mile" challenge, we propose TacRefineNet, a tactile-only framework that achieves fine in-hand pose refinement of known objects in arbitrary target poses using multi-finger fingertip sensing. Our method iteratively adjusts the end-effector pose based on tactile feedback, aligning the object to the desired configuration. We design a multi-branch policy network that fuses tactile inputs from multiple fingers along with proprioception to predict precise control updates. To train this policy, we combine large-scale simulated data from a physics-based tactile model in MuJoCo with real-world data collected from a physical system. Comparative experiments show that pretraining on simulated data and fine-tuning with a small amount of real data significantly improves performance over simulation-only training. Extensive real-world experiments validate the effectiveness of the method, achieving millimeter-level grasp accuracy using only tactile input. To our knowledge, this is the first method to enable arbitrary in-hand pose refinement via multi-finger tactile sensing alone. Project website is available at this https URL 

**Abstract (ZH)**: 尽管在传统灵巧抓取管道和近期视觉-语言-动作（VLA）方法方面取得了进展，但在执行抓取阶段仍容易出现姿态不准确的问题，尤其是在远期任务中，这会阻碍整体性能。为应对这一“最后一公里”挑战，我们提出TacRefineNet，这是一种仅依赖触觉的框架，利用多指指尖感知实现对任意目标姿态已知物体的精细在手姿态校正。该方法根据触觉反馈迭代调整末端执行器姿态，使物体与所需的配置对齐。我们设计了一个多分支策略网络，将多个手指的触觉输入与本体感受融合，以预测精确的控制更新。为训练该策略，我们将基于物理模型的大量模拟数据（来自MuJoCo）与来自实际系统的实时数据结合使用。对比实验表明，仅通过模拟数据预训练并在少量实际数据上进行微调，可以显著提高性能，优于仅基于模拟训练。大量现实世界的实验验证了该方法的有效性，仅通过触觉输入实现了毫米级的抓取精度。据我们所知，这是首个通过单一多指触觉感知实现任意在手姿态校正的方法。项目网站可访问此链接。 

---
# VLA Model Post-Training via Action-Chunked PPO and Self Behavior Cloning 

**Title (ZH)**: 基于动作分块PPO和自我行为克隆的VLA模型后训练 

**Authors**: Si-Cheng Wang, Tian-Yu Xiang, Xiao-Hu Zhou, Mei-Jiang Gui, Xiao-Liang Xie, Shi-Qi Liu, Shuang-Yi Wang, Ao-Qun Jin, Zeng-Guang Hou  

**Link**: [PDF](https://arxiv.org/pdf/2509.25718)  

**Abstract**: Reinforcement learning (RL) is a promising avenue for post-training vision-language-action (VLA) models, but practical deployment is hindered by sparse rewards and unstable training. This work mitigates these challenges by introducing an action chunk based on proximal policy optimization (PPO) with behavior cloning using self-collected demonstrations. Aggregating consecutive actions into chunks improves the temporal consistency of the policy and the density of informative feedback. In addition, an auxiliary behavior cloning loss is applied with a dynamically updated demonstration buffer that continually collects high-quality task trials during training. The relative weight between the action-chunked PPO objective and the self behavior clone auxiliary loss is adapted online to stabilize the post-training process. Experiments on the MetaWorld benchmark indicate improved performance over supervised fine-tuning, achieving a high success rate (0.93) and few steps to success (42.17). These results demonstrate the viability of RL for VLA post-training and help lay the groundwork for downstream VLA applications. 

**Abstract (ZH)**: 基于 proximal policy optimization 的动作块和行为克隆在后训练视觉-语言-行动模型中的应用 

---
# OmniNav: A Unified Framework for Prospective Exploration and Visual-Language Navigation 

**Title (ZH)**: OmniNav：综合框架用于前瞻性探索和视觉语言导航 

**Authors**: Xinda Xue, Junjun Hu, Minghua Luo, Xie Shichao, Jintao Chen, Zixun Xie, Quan Kuichen, Guo Wei, Mu Xu, Zedong Chu  

**Link**: [PDF](https://arxiv.org/pdf/2509.25687)  

**Abstract**: Embodied navigation presents a core challenge for intelligent robots, requiring the comprehension of visual environments, natural language instructions, and autonomous exploration. Existing models often fall short in offering a unified solution across diverse navigation paradigms, resulting in low success rates and limited generalization. We introduce OmniNav, a unified framework addressing instruct-goal, object-goal, point-goal navigation, and frontier-based exploration within a single architecture. Our approach features a lightweight, low-latency policy that accurately predicts continuous-space waypoints (coordinates and orientations). This policy surpasses action-chunk methods in precision and supports real-world deployment at control frequencies up to 5 Hz. Architecturally, OmniNav employs a fast-slow system design: a fast module generates waypoints using short-horizon visual context and subtasks, while a slow module performs deliberative planning with long-horizon observations and candidate frontiers to select subsequent subgoals and subtasks. This collaboration enhances path efficiency and maintains trajectory coherence, particularly in exploration and memory-intensive scenarios. Crucially, we identify that the primary bottleneck isn't merely navigation policy learning, but a robust understanding of general instructions and objects. To boost generalization, OmniNav integrates large-scale, general-purpose training datasets, including those for image captioning and visual recognition, into a joint multi-task regimen. This significantly improves success rates and robustness. Extensive experiments confirm OmniNav's state-of-the-art performance across various navigation benchmarks, with real-world deployment further validating its efficacy. OmniNav provides practical insights for embodied navigation, charting a scalable path towards versatile, highly generalizable robotic intelligence. 

**Abstract (ZH)**: 全方位导航为智能机器人提出了核心挑战，要求其理解视觉环境、自然语言指令并实现自主探索。现有的模型往往在提供跨异构导航范式的统一解决方案方面存在不足，导致成功率低且泛化能力有限。我们提出了一体化框架OmniNav，该框架在单一架构中解决了指令目标、物体目标、点目标导航以及基于前沿的探索问题。我们的方法采用轻量级、低延迟策略，能够准确预测连续空间航点（坐标和方向）。该策略的精度超过了基于动作片段的方法，并支持在高达5 Hz的控制频率下进行实际部署。架构上，OmniNav 采用快慢系统设计：快速模块使用短时视觉上下文和子任务生成航点，而缓慢模块则利用长时间观察和候选前沿进行详尽规划，以选择后续的子目标和子任务。这种协作增强了路径效率并保持了轨迹一致性，尤其是在探索和记忆密集型场景中。最关键的是，我们发现主要瓶颈不仅在于导航策略学习，还在于对通用指令和物体的稳健理解。为了提高泛化能力，OmniNav 将大规模、通用的数据集，包括图像字幕和视觉识别数据集，整合到联合多任务训练中，这显著提高了成功率和鲁棒性。广泛的实验结果表明，OmniNav 在各种导航基准测试中表现出目前最先进的性能，在实际部署中进一步验证了其有效性。OmniNav 为实体导航提供了实用洞察，描绘了一条实现多功能、高度泛化的机器人智能的可扩展路径。 

---
# Hierarchical Diffusion Motion Planning with Task-Conditioned Uncertainty-Aware Priors 

**Title (ZH)**: 层次扩散运动规划：基于任务条件的不确定性感知先验 

**Authors**: Amelie Minji Kim, Anqi Wu, Ye Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.25685)  

**Abstract**: We propose a novel hierarchical diffusion planner that embeds task and motion structure directly in the noise model. Unlike standard diffusion-based planners that use zero-mean, isotropic Gaussian noise, we employ a family of task-conditioned structured Gaussians whose means and covariances are derived from Gaussian Process Motion Planning (GPMP): sparse, task-centric key states or their associated timings (or both) are treated as noisy observations to produce a prior instance. We first generalize the standard diffusion process to biased, non-isotropic corruption with closed-form forward and posterior expressions. Building on this, our hierarchy separates prior instantiation from trajectory denoising: the upper level instantiates a task-conditioned structured Gaussian (mean and covariance), and the lower level denoises the full trajectory under that fixed prior. Experiments on Maze2D goal-reaching and KUKA block stacking show improved success rates, smoother trajectories, and stronger task alignment compared to isotropic baselines. Ablation studies indicate that explicitly structuring the corruption process offers benefits beyond simply conditioning the neural network. Overall, our method concentrates probability mass of prior near feasible, smooth, and semantically meaningful trajectories while maintaining tractability. Our project page is available at this https URL. 

**Abstract (ZH)**: 我们提出了一种新颖的层次扩散规划器，将任务和运动结构直接嵌入噪声模型中。 

---
# dVLA: Diffusion Vision-Language-Action Model with Multimodal Chain-of-Thought 

**Title (ZH)**: dVLA：具备多模态链式思维的扩散视觉语言行动模型 

**Authors**: Junjie Wen, Minjie Zhu, Jiaming Liu, Zhiyuan Liu, Yicun Yang, Linfeng Zhang, Shanghang Zhang, Yichen Zhu, Yi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.25681)  

**Abstract**: Vision-Language-Action (VLA) models are emerging as a next-generation paradigm for robotics. We introduce dVLA, a diffusion-based VLA that leverages a multimodal chain-of-thought to unify visual perception, language reasoning, and robotic control in a single system. dVLA jointly optimizes perception, language understanding, and action under a single diffusion objective, enabling stronger cross-modal reasoning and better generalization to novel instructions and objects. For practical deployment, we mitigate inference latency by incorporating two acceleration strategies, a prefix attention mask and KV caching, yielding up to around times speedup at test-time inference. We evaluate dVLA in both simulation and the real world: on the LIBERO benchmark, it achieves state-of-the-art performance with a 96.4% average success rate, consistently surpassing both discrete and continuous action policies; on a real Franka robot, it succeeds across a diverse task suite, including a challenging bin-picking task that requires multi-step planning, demonstrating robust real-world performance. Together, these results underscore the promise of unified diffusion frameworks for practical, high-performance VLA robotics. 

**Abstract (ZH)**: 基于扩散的视觉-语言-动作模型：统一的多模态推理在机器人学中的应用 

---
# Field Calibration of Hyperspectral Cameras for Terrain Inference 

**Title (ZH)**: 高光谱相机用于地貌推断的场校准 

**Authors**: Nathaniel Hanson, Benjamin Pyatski, Samuel Hibbard, Gary Lvov, Oscar De La Garza, Charles DiMarzio, Kristen L. Dorsey, Taşkın Padır  

**Link**: [PDF](https://arxiv.org/pdf/2509.25663)  

**Abstract**: Intra-class terrain differences such as water content directly influence a vehicle's ability to traverse terrain, yet RGB vision systems may fail to distinguish these properties. Evaluating a terrain's spectral content beyond red-green-blue wavelengths to the near infrared spectrum provides useful information for intra-class identification. However, accurate analysis of this spectral information is highly dependent on ambient illumination. We demonstrate a system architecture to collect and register multi-wavelength, hyperspectral images from a mobile robot and describe an approach to reflectance calibrate cameras under varying illumination conditions. To showcase the practical applications of our system, HYPER DRIVE, we demonstrate the ability to calculate vegetative health indices and soil moisture content from a mobile robot platform. 

**Abstract (ZH)**: 基于多光谱的移动机器人地形内类差异识别系统及其应用 

---
# Exhaustive-Serve-Longest Control for Multi-robot Scheduling Systems 

**Title (ZH)**: 全面服务最长控制策略的多机器人调度系统 

**Authors**: Mohammad Merati, David Castañón  

**Link**: [PDF](https://arxiv.org/pdf/2509.25556)  

**Abstract**: We study online task allocation for multi-robot, multi-queue systems with stochastic arrivals and switching delays. Time is slotted; each location can host at most one robot per slot; service consumes one slot; switching between locations incurs a one-slot travel delay; and arrivals are independent Bernoulli processes. We formulate a discounted-cost Markov decision process and propose Exhaustive-Serve-Longest (ESL), a simple real-time policy that serves exhaustively when the current location is nonempty and, when idle, switches to a longest unoccupied nonempty location, and we prove the optimality of this policy. As baselines, we tune a fixed-dwell cyclic policy via a discrete-time delay expression and implement a first-come-first-serve policy. Across server-to-location ratios and loads, ESL consistently yields lower discounted holding cost and smaller mean queue lengths, with action-time fractions showing more serving and restrained switching. Its simplicity and robustness make ESL a practical default for real-time multi-robot scheduling systems. 

**Abstract (ZH)**: 多机器人、多队列系统中具有随机到达和服务延迟的在线任务分配研究 

---
# Online Mapping for Autonomous Driving: Addressing Sensor Generalization and Dynamic Map Updates in Campus Environments 

**Title (ZH)**: 校园环境中的在线地图构建：应对传感器泛化和动态地图更新问题 

**Authors**: Zihan Zhang, Abhijit Ravichandran, Pragnya Korti, Luobin Wang, Henrik I. Christensen  

**Link**: [PDF](https://arxiv.org/pdf/2509.25542)  

**Abstract**: High-definition (HD) maps are essential for autonomous driving, providing precise information such as road boundaries, lane dividers, and crosswalks to enable safe and accurate navigation. However, traditional HD map generation is labor-intensive, expensive, and difficult to maintain in dynamic environments. To overcome these challenges, we present a real-world deployment of an online mapping system on a campus golf cart platform equipped with dual front cameras and a LiDAR sensor. Our work tackles three core challenges: (1) labeling a 3D HD map for campus environment; (2) integrating and generalizing the SemVecMap model onboard; and (3) incrementally generating and updating the predicted HD map to capture environmental changes. By fine-tuning with campus-specific data, our pipeline produces accurate map predictions and supports continual updates, demonstrating its practical value in real-world autonomous driving scenarios. 

**Abstract (ZH)**: 高分辨率（HD）地图对于自动驾驶至关重要，可以提供精准的道路边界、车道分隔线和人行横道等信息，以实现安全准确的导航。然而，传统的HD地图生成过程耗时、昂贵且难以在动态环境中维护。为克服这些挑战，我们在一个配备了双前摄像头和激光雷达传感器的校园高尔夫车平台上部署了一个在线制图系统。我们的工作解决了三个核心挑战：（1）为校园环境标注3D HD地图；（2）在车载上整合并泛化SemVecMap模型；（3）增量生成和更新预测的HD地图以捕捉环境变化。通过使用特定于校园的数据进行微调，我们的流程生成了准确的地图预测，并支持持续更新，证明了其在实际自动驾驶场景中的实用价值。 

---
# CoTaP: Compliant Task Pipeline and Reinforcement Learning of Its Controller with Compliance Modulation 

**Title (ZH)**: CoTaP: 合成型任务管道及其控制器的顺应性调节强化学习 

**Authors**: Zewen He, Chenyuan Chen, Dilshod Azizov, Yoshihiko Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2509.25443)  

**Abstract**: Humanoid whole-body locomotion control is a critical approach for humanoid robots to leverage their inherent advantages. Learning-based control methods derived from retargeted human motion data provide an effective means of addressing this issue. However, because most current human datasets lack measured force data, and learning-based robot control is largely position-based, achieving appropriate compliance during interaction with real environments remains challenging. This paper presents Compliant Task Pipeline (CoTaP): a pipeline that leverages compliance information in the learning-based structure of humanoid robots. A two-stage dual-agent reinforcement learning framework combined with model-based compliance control for humanoid robots is proposed. In the training process, first a base policy with a position-based controller is trained; then in the distillation, the upper-body policy is combined with model-based compliance control, and the lower-body agent is guided by the base policy. In the upper-body control, adjustable task-space compliance can be specified and integrated with other controllers through compliance modulation on the symmetric positive definite (SPD) manifold, ensuring system stability. We validated the feasibility of the proposed strategy in simulation, primarily comparing the responses to external disturbances under different compliance settings. 

**Abstract (ZH)**: 基于顺应性的类人全身运动控制管道（Compliant Task Pipeline）：一种结合模型驱动顺应控制的双Agent强化学习框架 

---
# Parallel Heuristic Search as Inference for Actor-Critic Reinforcement Learning Models 

**Title (ZH)**: 并行启发式搜索作为演员-评论家强化学习模型的推理 

**Authors**: Hanlan Yang, Itamar Mishani, Luca Pivetti, Zachary Kingston, Maxim Likhachev  

**Link**: [PDF](https://arxiv.org/pdf/2509.25402)  

**Abstract**: Actor-Critic models are a class of model-free deep reinforcement learning (RL) algorithms that have demonstrated effectiveness across various robot learning tasks. While considerable research has focused on improving training stability and data sampling efficiency, most deployment strategies have remained relatively simplistic, typically relying on direct actor policy rollouts. In contrast, we propose \pachs{} (\textit{P}arallel \textit{A}ctor-\textit{C}ritic \textit{H}euristic \textit{S}earch), an efficient parallel best-first search algorithm for inference that leverages both components of the actor-critic architecture: the actor network generates actions, while the critic network provides cost-to-go estimates to guide the search. Two levels of parallelism are employed within the search -- actions and cost-to-go estimates are generated in batches by the actor and critic networks respectively, and graph expansion is distributed across multiple threads. We demonstrate the effectiveness of our approach in robotic manipulation tasks, including collision-free motion planning and contact-rich interactions such as non-prehensile pushing. Visit this http URL for demonstrations and examples. 

**Abstract (ZH)**: Actor-Critic模型是一类模型自由的深度强化学习（RL）算法，已在各种机器人学习任务中展示了有效性。尽管在提高训练稳定性和数据采样效率方面进行了大量研究，但大多数部署策略仍相对简单，通常依赖于直接的actor策略采样。相反，我们提出了一种高效的并行最佳优先搜索算法\pachs{}（Parallel Actor-Critic Heuristic Search），该算法利用了actor-critic架构的两个组成部分：actor网络生成动作，而critic网络提供成本到终点估计以引导搜索。搜索中采用了两个层次的并行性——动作和成本到终点估计分别由actor和critic网络以批次形式生成，图扩展分布到多个线程中。我们通过机器人操作任务展示了该方法的有效性，包括无碰撞运动规划和接触丰富的交互，如非抓握推拉。请访问此网址以获取演示和示例。 

---
# SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation 

**Title (ZH)**: 面向长期 horizon 机器人操作的阶段aware奖励建模 

**Authors**: Qianzhong Chen, Justin Yu, Mac Schwager, Pieter Abbeel, Fred Shentu, Philipp Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.25358)  

**Abstract**: Large-scale robot learning has recently shown promise for enabling robots to perform complex tasks by integrating perception, control, and language understanding. Yet, it struggles with long-horizon, contact-rich manipulation such as deformable object handling, where demonstration quality is inconsistent. Reward modeling offers a natural solution: by providing grounded progress signals, it transforms noisy demonstrations into stable supervision that generalizes across diverse trajectories. We introduce a stage-aware, video-based reward modeling framework that jointly predicts high-level task stages and fine-grained progress. Reward labels are automatically derived from natural language subtask annotations, ensuring consistent progress estimation across variable-length demonstrations. This design overcomes frame-index labeling, which fails in variable-duration tasks like folding a T-shirt. Our reward model demonstrates robustness to variability, generalization to out-of-distribution settings, and strong utility for policy training. Building on it, we propose Reward-Aligned Behavior Cloning (RA-BC), which filters high-quality data and reweights samples by reward. Experiments show the reward model alone outperforms baselines on validation and real robot rollouts. Integrated into RA-BC, our approach achieves 83\% success on folding T-shirts from the flattened state and 67\% from the crumpled state -- far surpassing vanilla behavior cloning, which attains only 8\% and 0\% success. Overall, our results highlight reward modeling as a key enabler for scalable, annotation-efficient, and robust imitation learning in long-horizon manipulation. 

**Abstract (ZH)**: 大规模机器人学习在实现具有感知、控制和语言理解能力的复杂任务方面 Recent Progress：一种阶段意识的视频基奖励建模框架及其在长时 horizon 操作中的应用 

---
# SRMP: Search-Based Robot Motion Planning Library 

**Title (ZH)**: 基于搜索的机器人运动规划库SRMP 

**Authors**: Itamar Mishani, Yorai Shaoul, Ramkumar Natarajan, Jiaoyang Li, Maxim Likhachev  

**Link**: [PDF](https://arxiv.org/pdf/2509.25352)  

**Abstract**: Motion planning is a critical component in any robotic system. Over the years, powerful tools like the Open Motion Planning Library (OMPL) have been developed, offering numerous motion planning algorithms. However, existing frameworks often struggle to deliver the level of predictability and repeatability demanded by high-stakes applications -- ranging from ensuring safety in industrial environments to the creation of high-quality motion datasets for robot learning. Complementing existing tools, we introduce SRMP (Search-based Robot Motion Planning), a new software framework tailored for robotic manipulation. SRMP distinguishes itself by generating consistent and reliable trajectories, and is the first software tool to offer motion planning algorithms for multi-robot manipulation tasks. SRMP easily integrates with major simulators, including MuJoCo, Sapien, Genesis, and PyBullet via a Python and C++ API. SRMP includes a dedicated MoveIt! plugin that enables immediate deployment on robot hardware and seamless integration with existing pipelines. Through extensive evaluations, we demonstrate in this paper that SRMP not only meets the rigorous demands of industrial and safety-critical applications but also sets a new standard for consistency in motion planning across diverse robotic systems. Visit this http URL for SRMP documentation and tutorials. 

**Abstract (ZH)**: 基于搜索的机器人运动规划SRMP：一种针对机器人操作的新软件框架 

---
# BEV-VLM: Trajectory Planning via Unified BEV Abstraction 

**Title (ZH)**: BEV-VLM：通过统一的BEV抽象进行轨迹规划 

**Authors**: Guancheng Chen, Sheng Yang, Tong Zhan, Jian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.25249)  

**Abstract**: This paper introduces BEV-VLM, a novel framework for trajectory planning in autonomous driving that leverages Vision-Language Models (VLMs) with Bird's-Eye View (BEV) feature maps as visual inputs. Unlike conventional approaches that rely solely on raw visual data such as camera images, our method utilizes highly compressed and informative BEV representations, which are generated by fusing multi-modal sensor data (e.g., camera and LiDAR) and aligning them with HD Maps. This unified BEV-HD Map format provides a geometrically consistent and rich scene description, enabling VLMs to perform accurate trajectory planning. Experimental results on the nuScenes dataset demonstrate 44.8% improvements in planning accuracy and complete collision avoidance. Our work highlights that VLMs can effectively interpret processed visual representations like BEV features, expanding their applicability beyond raw images in trajectory planning. 

**Abstract (ZH)**: BEV-VLM：一种利用Bird's-Eye View特征图和Vision-Language模型进行自主驾驶轨迹规划的新框架 

---
# When and How to Express Empathy in Human-Robot Interaction Scenarios 

**Title (ZH)**: 在人机交互场景中何时及如何表达同理心 

**Authors**: Christian Arzate Cruz, Edwin C. Montiel-Vazquez, Chikara Maeda, Randy Gomez  

**Link**: [PDF](https://arxiv.org/pdf/2509.25200)  

**Abstract**: Incorporating empathetic behavior into robots can improve their social effectiveness and interaction quality. In this paper, we present whEE (when and how to express empathy), a framework that enables social robots to detect when empathy is needed and generate appropriate responses. Using large language models, whEE identifies key behavioral empathy cues in human interactions. We evaluate it in human-robot interaction scenarios with our social robot, Haru. Results show that whEE effectively identifies and responds to empathy cues, providing valuable insights for designing social robots capable of adaptively modulating their empathy levels across various interaction contexts. 

**Abstract (ZH)**: 将共情行为融入机器人可以提高其社会效果和交互质量。本文提出了一种名为whEE（何时以及如何表达共情）的框架，使社会机器人能够检测出何时需要表达共情，并生成合适的响应。通过使用大规模语言模型，whEE识别出人类互动中的关键共情行为线索。我们在与我们的社会机器人Haru的人机交互场景中对其进行评估。结果显示，whEE有效识别并响应共情行为线索，为设计能够在各种交互情境中适配性调节共情水平的社会机器人提供了宝贵见解。 

---
# Benchmarking Egocentric Visual-Inertial SLAM at City Scale 

**Title (ZH)**: 基于城市规模的主观视角视觉-惯性SLAM基准测试 

**Authors**: Anusha Krishnan, Shaohui Liu, Paul-Edouard Sarlin, Oscar Gentilhomme, David Caruso, Maurizio Monge, Richard Newcombe, Jakob Engel, Marc Pollefeys  

**Link**: [PDF](https://arxiv.org/pdf/2509.26639)  

**Abstract**: Precise 6-DoF simultaneous localization and mapping (SLAM) from onboard sensors is critical for wearable devices capturing egocentric data, which exhibits specific challenges, such as a wider diversity of motions and viewpoints, prevalent dynamic visual content, or long sessions affected by time-varying sensor calibration. While recent progress on SLAM has been swift, academic research is still driven by benchmarks that do not reflect these challenges or do not offer sufficiently accurate ground truth poses. In this paper, we introduce a new dataset and benchmark for visual-inertial SLAM with egocentric, multi-modal data. We record hours and kilometers of trajectories through a city center with glasses-like devices equipped with various sensors. We leverage surveying tools to obtain control points as indirect pose annotations that are metric, centimeter-accurate, and available at city scale. This makes it possible to evaluate extreme trajectories that involve walking at night or traveling in a vehicle. We show that state-of-the-art systems developed by academia are not robust to these challenges and we identify components that are responsible for this. In addition, we design tracks with different levels of difficulty to ease in-depth analysis and evaluation of less mature approaches. The dataset and benchmark are available at this https URL. 

**Abstract (ZH)**: 基于穿戴设备的六自由度同时定位与建图：从车载传感器精确捕捉第一人称多模态数据的独特挑战及新数据集与基准 

---
# TimeRewarder: Learning Dense Reward from Passive Videos via Frame-wise Temporal Distance 

**Title (ZH)**: TimeRewarder: 通过帧级时间距离从被动视频中学习密集奖励 

**Authors**: Yuyang Liu, Chuan Wen, Yihang Hu, Dinesh Jayaraman, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.26627)  

**Abstract**: Designing dense rewards is crucial for reinforcement learning (RL), yet in robotics it often demands extensive manual effort and lacks scalability. One promising solution is to view task progress as a dense reward signal, as it quantifies the degree to which actions advance the system toward task completion over time. We present TimeRewarder, a simple yet effective reward learning method that derives progress estimation signals from passive videos, including robot demonstrations and human videos, by modeling temporal distances between frame pairs. We then demonstrate how TimeRewarder can supply step-wise proxy rewards to guide reinforcement learning. In our comprehensive experiments on ten challenging Meta-World tasks, we show that TimeRewarder dramatically improves RL for sparse-reward tasks, achieving nearly perfect success in 9/10 tasks with only 200,000 interactions per task with the environment. This approach outperformed previous methods and even the manually designed environment dense reward on both the final success rate and sample efficiency. Moreover, we show that TimeRewarder pretraining can exploit real-world human videos, highlighting its potential as a scalable approach path to rich reward signals from diverse video sources. 

**Abstract (ZH)**: 设计密集奖励对于强化学习至关重要，但在机器人领域往往需要大量手动努力且缺乏可扩展性。一种有前景的解决方案是将任务进度视为密集奖励信号，因为它量化了动作随时间推进系统完成任务的程度。我们提出了TimeRewarder，这是一种简单而有效的方法，通过模型帧对之间的时序距离从被动视频中提取进度估计算法，包括机器人演示和人类视频，以学习奖励函数。然后，我们展示了如何使用TimeRewarder为强化学习提供逐步代理奖励以进行引导。在对十个具有挑战性的Meta-World任务进行全面实验中，我们表明TimeRewarder显著提高了稀疏奖励任务的强化学习效果，仅通过每任务200,000次环境交互实现了9/10任务近完美的成功率。该方法在最终成功率和样本效率上都优于先前的方法，甚至超过了手工设计的密集环境奖励。此外，我们展示了TimeRewarder预训练可以利用现实世界的人类视频，突显了其从多种视频来源获取丰富奖励信号的潜在可扩展性路径。 

---
# The Trajectory Bundle Method: Unifying Sequential-Convex Programming and Sampling-Based Trajectory Optimization 

**Title (ZH)**: 轨迹束方法：序列凸规划与基于采样的轨迹优化统一方法 

**Authors**: Kevin Tracy, John Z. Zhang, Jon Arrizabalaga, Stefan Schaal, Yuval Tassa, Tom Erez, Zachary Manchester  

**Link**: [PDF](https://arxiv.org/pdf/2509.26575)  

**Abstract**: We present a unified framework for solving trajectory optimization problems in a derivative-free manner through the use of sequential convex programming. Traditionally, nonconvex optimization problems are solved by forming and solving a sequence of convex optimization problems, where the cost and constraint functions are approximated locally through Taylor series expansions. This presents a challenge for functions where differentiation is expensive or unavailable. In this work, we present a derivative-free approach to form these convex approximations by computing samples of the dynamics, cost, and constraint functions and letting the solver interpolate between them. Our framework includes sample-based trajectory optimization techniques like model-predictive path integral (MPPI) control as a special case and generalizes them to enable features like multiple shooting and general equality and inequality constraints that are traditionally associated with derivative-based sequential convex programming methods. The resulting framework is simple, flexible, and capable of solving a wide variety of practical motion planning and control problems. 

**Abstract (ZH)**: 我们提出了一种统一框架，通过顺序凸规划以无导数的方式求解轨迹优化问题。传统上，非凸优化问题通过形成和求解一系列凸优化问题来解决，其中通过泰勒级数展开局部逼近代价和约束函数。对于导数昂贵或不可用的情况，这提出了挑战。在本工作中，我们提出了一种无导数的方法来形成这些凸逼近：通过计算动力学、代价和约束函数的样本，并让求解器在它们之间进行插值。该框架将模型预测路径积分（MPPI）控制等基于样本的轨迹优化技术作为特殊情况，并将其推广以启用多项射击以及传统上与基于导数的顺序凸规划方法相关的多种等式和不等式约束。所提出的方法简单、灵活，并能够解决广泛的实践运动规划和控制问题。 

---
# OceanGym: A Benchmark Environment for Underwater Embodied Agents 

**Title (ZH)**: OceanGym: 水下 embodiable 代理的基准环境 

**Authors**: Yida Xue, Mingjun Mao, Xiangyuan Ru, Yuqi Zhu, Baochang Ren, Shuofei Qiao, Mengru Wang, Shumin Deng, Xinyu An, Ningyu Zhang, Ying Chen, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.26536)  

**Abstract**: We introduce OceanGym, the first comprehensive benchmark for ocean underwater embodied agents, designed to advance AI in one of the most demanding real-world environments. Unlike terrestrial or aerial domains, underwater settings present extreme perceptual and decision-making challenges, including low visibility, dynamic ocean currents, making effective agent deployment exceptionally difficult. OceanGym encompasses eight realistic task domains and a unified agent framework driven by Multi-modal Large Language Models (MLLMs), which integrates perception, memory, and sequential decision-making. Agents are required to comprehend optical and sonar data, autonomously explore complex environments, and accomplish long-horizon objectives under these harsh conditions. Extensive experiments reveal substantial gaps between state-of-the-art MLLM-driven agents and human experts, highlighting the persistent difficulty of perception, planning, and adaptability in ocean underwater environments. By providing a high-fidelity, rigorously designed platform, OceanGym establishes a testbed for developing robust embodied AI and transferring these capabilities to real-world autonomous ocean underwater vehicles, marking a decisive step toward intelligent agents capable of operating in one of Earth's last unexplored frontiers. The code and data are available at this https URL. 

**Abstract (ZH)**: 我们介绍OceanGym，这是首个全面的海洋水下 embodiable代理基准，旨在推动AI在最具挑战性的现实环境之一中的发展。不同于陆地或空中领域，水下环境呈现极端的感受与决策挑战，包括低能见度、动态海洋流，使有效代理部署异常困难。OceanGym包含八个现实的任务领域，并采用由多模态大型语言模型（MLLMs）驱动的统一代理框架，整合了感知、记忆和顺序决策。代理需要在这些恶劣条件下理解光学和声纳数据，自主探索复杂环境，并实现长期目标。大量实验表明，由最先进MLLM驱动的代理与人类专家之间存在显著差距，突显了在海洋水下环境中感知、规划和适应性的持续困难。通过提供一个高保真、精心设计的平台，OceanGym建立了开发鲁棒的 embodiable AI的测验平台，并将这些能力转移到真实的自主海洋水下车辆上，标志着向能够在此地球最后未开发的前沿领域中运行的智能代理迈出决定性的一步。代码和数据请访问此网址。 

---
# Towards Human Engagement with Realistic AI Combat Pilots 

**Title (ZH)**: 面向现实AI战斗机飞行员的人机互动研究 

**Authors**: Ardian Selmonaj, Giacomo Del Rio, Adrian Schneider, Alessandro Antonucci  

**Link**: [PDF](https://arxiv.org/pdf/2509.26002)  

**Abstract**: We present a system that enables real-time interaction between human users and agents trained to control fighter jets in simulated 3D air combat scenarios. The agents are trained in a dedicated environment using Multi-Agent Reinforcement Learning. A communication link is developed to allow seamless deployment of trained agents into VR-Forces, a widely used defense simulation tool for realistic tactical scenarios. This integration allows mixed simulations where human-controlled entities engage with intelligent agents exhibiting distinct combat behaviors. Our interaction model creates new opportunities for human-agent teaming, immersive training, and the exploration of innovative tactics in defense contexts. 

**Abstract (ZH)**: 我们提出了一种系统，能够在模拟3D空战场景中实现人类用户与通过多智能体强化学习训练来控制战斗机的代理之间的实时交互。这些代理在专用环境中训练，并开发了通信链接以便无缝部署至广泛使用的防御仿真工具VR-Forces中，用于现实战术场景。这种整合使得人类控制的实体能够与展示不同战斗行为的智能代理进行混合仿真。我们的交互模型为人类-代理团队合作、沉浸式训练以及在防御背景下探索创新战术提供了新机遇。 

---
# Preemptive Spatiotemporal Trajectory Adjustment for Heterogeneous Vehicles in Highway Merging Zones 

**Title (ZH)**: 高速公路汇合区异质车辆的预emption时空轨迹调整 

**Authors**: Yuan Li, Xiaoxue Xu, Xiang Dong, Junfeng Hao, Tao Li, Sana Ullaha, Chuangrui Huang, Junjie Niu, Ziyan Zhao, Ting Peng  

**Link**: [PDF](https://arxiv.org/pdf/2509.25929)  

**Abstract**: Aiming at the problem of driver's perception lag and low utilization efficiency of space-time resources in expressway ramp confluence area, based on the preemptive spatiotemporal trajectory Adjustment system, from the perspective of coordinating spatiotemporal resources, the reasonable value of safe space-time distance in trajectory pre-preparation is quantitatively analyzed. The minimum safety gap required for ramp vehicles to merge into the mainline is analyzed by introducing double positioning error and spatiotemporal trajectory tracking error. A merging control strategy for autonomous driving heterogeneous vehicles is proposed, which integrates vehicle type, driving intention, and safety spatiotemporal distance. The specific confluence strategies of ramp target vehicles and mainline cooperative vehicles under different vehicle types are systematically expounded. A variety of traffic flow and speed scenarios are used for full combination simulation. By comparing the time-position-speed diagram, the vehicle operation characteristics and the dynamic difference of confluence are qualitatively analyzed, and the average speed and average delay are used as the evaluation indices to quantitatively evaluate the performance advantages of the preemptive cooperative confluence control strategy. The results show that the maximum average delay improvement rates of mainline and ramp vehicles are 90.24 % and 74.24 %, respectively. The proposed strategy can effectively avoid potential vehicle conflicts and emergency braking behaviors, improve driving safety in the confluence area, and show significant advantages in driving stability and overall traffic efficiency optimization. 

**Abstract (ZH)**: 基于预emption时空轨迹调整系统的匝道合流区时空资源协调控制研究 

---
# Boundary-to-Region Supervision for Offline Safe Reinforcement Learning 

**Title (ZH)**: 边界到区域的监督学习用于离线安全强化学习 

**Authors**: Huikang Su, Dengyun Peng, Zifeng Zhuang, YuHan Liu, Qiguang Chen, Donglin Wang, Qinghe Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.25727)  

**Abstract**: Offline safe reinforcement learning aims to learn policies that satisfy predefined safety constraints from static datasets. Existing sequence-model-based methods condition action generation on symmetric input tokens for return-to-go and cost-to-go, neglecting their intrinsic asymmetry: return-to-go (RTG) serves as a flexible performance target, while cost-to-go (CTG) should represent a rigid safety boundary. This symmetric conditioning leads to unreliable constraint satisfaction, especially when encountering out-of-distribution cost trajectories. To address this, we propose Boundary-to-Region (B2R), a framework that enables asymmetric conditioning through cost signal realignment . B2R redefines CTG as a boundary constraint under a fixed safety budget, unifying the cost distribution of all feasible trajectories while preserving reward structures. Combined with rotary positional embeddings , it enhances exploration within the safe region. Experimental results show that B2R satisfies safety constraints in 35 out of 38 safety-critical tasks while achieving superior reward performance over baseline methods. This work highlights the limitations of symmetric token conditioning and establishes a new theoretical and practical approach for applying sequence models to safe RL. Our code is available at this https URL. 

**Abstract (ZH)**: 离线安全强化学习旨在从静态数据集中学习满足预定义安全约束的策略。现有的序列模型方法通过对返回剩余奖励和成本剩余成本的对称输入令牌进行条件处理来生成动作，忽视了它们的内在不对称性：返回剩余奖励（RTG）作为灵活的性能目标，而成本剩余成本（CTG）则应表示一个刚性安全边界。这种对称条件处理会导致约束满足不可靠，尤其是在遇到分布外成本轨迹时。为了解决这一问题，我们提出了边界到区域（B2R）框架，通过成本信号重新对齐实现不对称条件处理。B2R将CTG重新定义为在固定安全预算下的边界约束，统一了所有可行轨迹的成本分布，同时保留奖励结构。结合旋转位置嵌入，它增强了在安全区域内的探索。实验结果表明，B2R在38个安全关键任务中的35个任务中满足了安全约束，并在基线方法上达到了更好的奖励性能。这项工作突显了对称令牌条件处理的局限性，并建立了将序列模型应用于安全RL的一种新的理论和实践方法。我们的代码可在以下链接获取：this https URL。 

---
# MoReFlow: Motion Retargeting Learning through Unsupervised Flow Matching 

**Title (ZH)**: MoReFlow: 无监督流匹配下的运动重定位学习 

**Authors**: Wontaek Kim, Tianyu Li, Sehoon Ha  

**Link**: [PDF](https://arxiv.org/pdf/2509.25600)  

**Abstract**: Motion retargeting holds a premise of offering a larger set of motion data for characters and robots with different morphologies. Many prior works have approached this problem via either handcrafted constraints or paired motion datasets, limiting their applicability to humanoid characters or narrow behaviors such as locomotion. Moreover, they often assume a fixed notion of retargeting, overlooking domain-specific objectives like style preservation in animation or task-space alignment in robotics. In this work, we propose MoReFlow, Motion Retargeting via Flow Matching, an unsupervised framework that learns correspondences between characters' motion embedding spaces. Our method consists of two stages. First, we train tokenized motion embeddings for each character using a VQ-VAE, yielding compact latent representations. Then, we employ flow matching with conditional coupling to align the latent spaces across characters, which simultaneously learns conditioned and unconditioned matching to achieve robust but flexible retargeting. Once trained, MoReFlow enables flexible and reversible retargeting without requiring paired data. Experiments demonstrate that MoReFlow produces high-quality motions across diverse characters and tasks, offering improved controllability, generalization, and motion realism compared to the baselines. 

**Abstract (ZH)**: 基于流动匹配的动作重定向持有为具有不同形态的角色和机器人提供更大动作数据集的前提。许多先前的工作通过手工构建的约束或配对的动作数据集来解决这个问题，这限制了它们在人形角色或狭窄行为如行走方面的适用性。此外，它们通常假设动作重定向的固定概念，忽视了诸如动画中的风格保持或机器人中的任务空间对齐等领域特定目标。在本文中，我们提出了一种基于流动匹配的动作重定向方法MoReFlow，这是一种无监督框架，用于学习角色动作嵌入空间之间的对应关系。该方法包括两个阶段。首先，我们使用VQ-VAE为每个角色训练标记化动作嵌入，产生紧凑的潜在表示。然后，我们使用条件耦合的流动匹配来对齐角色之间的潜在空间，同时学习条件和无条件匹配，以实现稳健且灵活的动作重定向。训练完成后，MoReFlow能够实现灵活且可逆的动作重定向，无需配对数据。实验结果表明，MoReFlow在多种角色和任务中生成高质量的动作，相比基线方法提供了更好的可控性、泛化能力和动作真实性。 

---
# Integrator Forwading Design for Unicycles with Constant and Actuated Velocity in Polar Coordinates 

**Title (ZH)**: 极坐标系中具有恒定和可控速度单轮车的积分前向设计 

**Authors**: Miroslav Krstic, Velimir Todorovski, Kwang Hak Kim, Alessandro Astolfi  

**Link**: [PDF](https://arxiv.org/pdf/2509.25579)  

**Abstract**: In a companion paper, we present a modular framework for unicycle stabilization in polar coordinates that provides smooth steering laws through backstepping. Surprisingly, the same problem also allows the application of integrator forwarding. In this work, we leverage this feature and construct new smooth steering laws together with control Lyapunov functions (CLFs), expanding the set of CLFs available for inverse optimal control design. In the case of constant forward velocity (Dubins car), backstepping produces finite-time (deadbeat) parking, and we show that integrator forwarding yields the very same class of solutions. This reveals a fundamental connection between backstepping and forwarding in addressing both the unicycle and, the Dubins car parking problems. 

**Abstract (ZH)**: 伴随论文中，我们提出了一种基于极坐标下的单轮车稳定模块化框架，通过递归回步方法提供了平滑的转向法则。令人意外的是，相同的问题也允许应用积分前馈方法。在本工作中，我们利用这一特性，构建新的平滑转向法则与控制李亚普诺夫函数（CLFs），扩展了可用于逆最优控制设计的CLFs集合。对于恒定前进速度的情况（杜宾车），递归回步方法产生有限时间（瞬态）停车，并且我们证明积分前馈方法同样产生同一类解。这揭示了递归回步方法与前馈方法在解决单轮车和杜宾车停车问题时的基本联系。 

---
# Modular Design of Strict Control Lyapunov Functions for Global Stabilization of the Unicycle in Polar Coordinates 

**Title (ZH)**: 极坐标系中独轮车全局稳定性的严格控制李亚普诺夫函数模块化设计 

**Authors**: Velimir Todorovski, Kwang Hak Kim, Miroslav Krstic  

**Link**: [PDF](https://arxiv.org/pdf/2509.25575)  

**Abstract**: Since the mid-1990s, it has been known that, unlike in Cartesian form where Brockett's condition rules out static feedback stabilization, the unicycle is globally asymptotically stabilizable by smooth feedback in polar coordinates. In this note, we introduce a modular framework for designing smooth feedback laws that achieve global asymptotic stabilization in polar coordinates. These laws are bidirectional, enabling efficient parking maneuvers, and are paired with families of strict control Lyapunov functions (CLFs) constructed in a modular fashion. The resulting CLFs guarantee global asymptotic stability with explicit convergence rates and include barrier variants that yield "almost global" stabilization, excluding only zero-measure subsets of the rotation manifolds. The strictness of the CLFs is further leveraged in our companion paper, where we develop inverse-optimal redesigns with meaningful cost functions and infinite gain margins. 

**Abstract (ZH)**: 自20世纪90年代中期以来，人们知道，与笛卡尔坐标系中布罗克特条件排除静态反馈稳定化不同，独轮车可以通过光滑反馈在极坐标系中实现全局渐近稳定化。本文介绍了一种模块化框架，用于设计能够在极坐标系中实现全局渐近稳定化的光滑反馈法则。这些法则具有双向性，便于高效执行停车操作，并且与模块化构造的严格控制李雅普(Ny)诺夫函数(CLFs)配对。这些CLFs确保全局渐近稳定，并具有显式的收敛速率，包括作为旋转流形上零测度子集的补充，可实现“几乎全局”稳定化。在我们姊妹论文中，进一步利用这些CLFs的严格性，开发了具有实际意义的成本函数和无限增益边际的逆最优重新设计方法。 

---
# LLM-RG: Referential Grounding in Outdoor Scenarios using Large Language Models 

**Title (ZH)**: LLM-RG：大规模语言模型在户外场景中的指 Referential 地指grounding 场景中的应用 

**Authors**: Pranav Saxena, Avigyan Bhattacharya, Ji Zhang, Wenshan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.25528)  

**Abstract**: Referential grounding in outdoor driving scenes is challenging due to large scene variability, many visually similar objects, and dynamic elements that complicate resolving natural-language references (e.g., "the black car on the right"). We propose LLM-RG, a hybrid pipeline that combines off-the-shelf vision-language models for fine-grained attribute extraction with large language models for symbolic reasoning. LLM-RG processes an image and a free-form referring expression by using an LLM to extract relevant object types and attributes, detecting candidate regions, generating rich visual descriptors with a VLM, and then combining these descriptors with spatial metadata into natural-language prompts that are input to an LLM for chain-of-thought reasoning to identify the referent's bounding box. Evaluated on the Talk2Car benchmark, LLM-RG yields substantial gains over both LLM and VLM-based baselines. Additionally, our ablations show that adding 3D spatial cues further improves grounding. Our results demonstrate the complementary strengths of VLMs and LLMs, applied in a zero-shot manner, for robust outdoor referential grounding. 

**Abstract (ZH)**: 基于视觉-语言模型和大型语言模型的LLM-RG在户外驾驶场景中的引用定位具有挑战性，由于场景高度变化、视觉相似的物体众多以及动态元素使得自然语言引用解析复杂化（例如，“右边的黑色汽车”）。我们提出了LLM-RG，这是一种混合管道，结合了现成的视觉-语言模型进行细粒度属性提取，以及大型语言模型进行符号推理。LLM-RG通过使用LLM提取相关对象类型和属性、检测候选区域、利用VLM生成丰富的视觉描述符，然后将这些描述符与空间元数据结合成自然语言提示输入LLM进行链式推理，以识别引用的边界框。在Talk2Car基准测试上，LLM-RG在基于LLM和VLM的方法基线上取得了显著的进步。此外，我们的消融实验表明，增加三维空间线索进一步提高了引用定位的效果。我们的结果展示了视觉-语言模型和大型语言模型在零样本情况下结合应用的互补优势，用于鲁棒的户外引用定位。 

---
# Robust Visual Localization in Compute-Constrained Environments by Salient Edge Rendering and Weighted Hamming Similarity 

**Title (ZH)**: 受限计算环境中的显著边缘渲染与加权汉明相似性稳健视觉定位 

**Authors**: Tu-Hoa Pham, Philip Bailey, Daniel Posada, Georgios Georgakis, Jorge Enriquez, Surya Suresh, Marco Dolci, Philip Twu  

**Link**: [PDF](https://arxiv.org/pdf/2509.25520)  

**Abstract**: We consider the problem of vision-based 6-DoF object pose estimation in the context of the notional Mars Sample Return campaign, in which a robotic arm would need to localize multiple objects of interest for low-clearance pickup and insertion, under severely constrained hardware. We propose a novel localization algorithm leveraging a custom renderer together with a new template matching metric tailored to the edge domain to achieve robust pose estimation using only low-fidelity, textureless 3D models as inputs. Extensive evaluations on synthetic datasets as well as from physical testbeds on Earth and in situ Mars imagery shows that our method consistently beats the state of the art in compute and memory-constrained localization, both in terms of robustness and accuracy, in turn enabling new possibilities for cheap and reliable localization on general-purpose hardware. 

**Abstract (ZH)**: 基于视觉的6自由度物体姿态估计在设想中的火星样本返回任务中，利用定制渲染器和边缘域定制模板匹配度量实现受硬件限制条件下的鲁棒姿态估计 

---
# World Model for AI Autonomous Navigation in Mechanical Thrombectomy 

**Title (ZH)**: AI自主导航的机械取栓世界模型 

**Authors**: Harry Robertshaw, Han-Ru Wu, Alejandro Granados, Thomas C Booth  

**Link**: [PDF](https://arxiv.org/pdf/2509.25518)  

**Abstract**: Autonomous navigation for mechanical thrombectomy (MT) remains a critical challenge due to the complexity of vascular anatomy and the need for precise, real-time decision-making. Reinforcement learning (RL)-based approaches have demonstrated potential in automating endovascular navigation, but current methods often struggle with generalization across multiple patient vasculatures and long-horizon tasks. We propose a world model for autonomous endovascular navigation using TD-MPC2, a model-based RL algorithm. We trained a single RL agent across multiple endovascular navigation tasks in ten real patient vasculatures, comparing performance against the state-of-the-art Soft Actor-Critic (SAC) method. Results indicate that TD-MPC2 significantly outperforms SAC in multi-task learning, achieving a 65% mean success rate compared to SAC's 37%, with notable improvements in path ratio. TD-MPC2 exhibited increased procedure times, suggesting a trade-off between success rate and execution speed. These findings highlight the potential of world models for improving autonomous endovascular navigation and lay the foundation for future research in generalizable AI-driven robotic interventions. 

**Abstract (ZH)**: 自主机械取栓的自主导航仍然是一个关键挑战，由于血管解剖结构的复杂性和需要进行精确的实时决策。基于强化学习的方法在自动化血管内导航中显示出潜力，但目前的方法在跨多个患者血管结构的泛化能力和长期任务处理上常常表现不足。我们提出了一种使用TD-MPC2（基于模型的RL算法）的世界模型来进行自主血管内导航。我们在十个真实患者的血管结构中训练了一个单一的RL代理，并与最先进的Soft Actor-Critic（SAC）方法进行了性能比较。结果表明，TD-MPC2在多任务学习中显著优于SAC，成功率达到65%，而SAC仅为37%，并且在路径比率上有明显的提升。TD-MPC2的程序时间有所增加，表明了成功率和执行速度之间的权衡。这些发现突显了世界模型在提升自主血管内导航中的潜在价值，并为未来通用AI驱动的机器人干预研究奠定了基础。 

---
# Message passing-based inference in an autoregressive active inference agent 

**Title (ZH)**: 基于消息传递的自回归主动求知代理推断 

**Authors**: Wouter M. Kouw, Tim N. Nisslbeck, Wouter L.N. Nuijten  

**Link**: [PDF](https://arxiv.org/pdf/2509.25482)  

**Abstract**: We present the design of an autoregressive active inference agent in the form of message passing on a factor graph. Expected free energy is derived and distributed across a planning graph. The proposed agent is validated on a robot navigation task, demonstrating exploration and exploitation in a continuous-valued observation space with bounded continuous-valued actions. Compared to a classical optimal controller, the agent modulates action based on predictive uncertainty, arriving later but with a better model of the robot's dynamics. 

**Abstract (ZH)**: 一种基于因子图消息传递的自回归主动推断代理的设计及其在机器人导航任务中的验证 

---
# Infrastructure Sensor-enabled Vehicle Data Generation using Multi-Sensor Fusion for Proactive Safety Applications at Work Zone 

**Title (ZH)**: 基于多传感器融合的工作区主动安全应用的基础设施传感器启用车辆数据生成 

**Authors**: Suhala Rabab Saba, Sakib Khan, Minhaj Uddin Ahmad, Jiahe Cao, Mizanur Rahman, Li Zhao, Nathan Huynh, Eren Erman Ozguven  

**Link**: [PDF](https://arxiv.org/pdf/2509.25452)  

**Abstract**: Infrastructure-based sensing and real-time trajectory generation show promise for improving safety in high-risk roadway segments such as work zones, yet practical deployments are hindered by perspective distortion, complex geometry, occlusions, and costs. This study tackles these barriers by integrating roadside camera and LiDAR sensors into a cosimulation environment to develop a scalable, cost-effective vehicle detection and localization framework, and employing a Kalman Filter-based late fusion strategy to enhance trajectory consistency and accuracy. In simulation, the fusion algorithm reduced longitudinal error by up to 70 percent compared to individual sensors while preserving lateral accuracy within 1 to 3 meters. Field validation in an active work zone, using LiDAR, a radar-camera rig, and RTK-GPS as ground truth, demonstrated that the fused trajectories closely match real vehicle paths, even when single-sensor data are intermittent or degraded. These results confirm that KF based sensor fusion can reliably compensate for individual sensor limitations, providing precise and robust vehicle tracking capabilities. Our approach thus offers a practical pathway to deploy infrastructure-enabled multi-sensor systems for proactive safety measures in complex traffic environments. 

**Abstract (ZH)**: 基于基础设施的传感器与实时轨迹生成在高风险道路路段（如工作区）提高安全性方面具有潜力，但实际部署受到视角失真、复杂几何结构、遮挡和成本的阻碍。本研究通过将路边摄像头和LiDAR传感器整合到协同仿真环境中，开发了一种可扩展且成本效益高的车辆检测与定位框架，并采用卡尔曼滤波器为基础的后融合策略以增强轨迹的一致性和准确性。在仿真中，融合算法将纵向误差降低了多达70%，同时保持了横向准确性在1至3米范围内。在现场验证中，在一个活跃的工作区内，使用LiDAR、雷达-摄像头组合和RTK-GPS作为地面真实值，结果显示融合轨迹与实际车辆路径高度一致，即使单传感器数据间断或降级也是如此。这些结果证实了基于卡尔曼滤波器的传感器融合可以可靠地弥补单传感器的局限性，提供精确且稳健的车辆追踪能力。因此，本研究为在复杂交通环境中部署基础设施支持的多传感器系统以实现主动安全性提供了一条实用途径。 

---
# Sensor optimization for urban wind estimation with cluster-based probabilistic framework 

**Title (ZH)**: 基于聚类的概率框架下的城市风速估计传感器优化 

**Authors**: Yutong Liang, Chang Hou, Guy Y. Cornejo Maceda, Andrea Ianiro, Stefano Discetti, Andrea Meilán-Vila, Didier Sornette, Sandro Claudio Lera, Jialong Chen, Xiaozhou He, Bernd R. Noack  

**Link**: [PDF](https://arxiv.org/pdf/2509.25222)  

**Abstract**: We propose a physics-informed machine-learned framework for sensor-based flow estimation for drone trajectories in complex urban terrain. The input is a rich set of flow simulations at many wind conditions. The outputs are velocity and uncertainty estimates for a target domain and subsequent sensor optimization for minimal uncertainty. The framework has three innovations compared to traditional flow estimators. First, the algorithm scales proportionally to the domain complexity, making it suitable for flows that are too complex for any monolithic reduced-order representation. Second, the framework extrapolates beyond the training data, e.g., smaller and larger wind velocities. Last, and perhaps most importantly, the sensor location is a free input, significantly extending the vast majority of the literature. The key enablers are (1) a Reynolds number-based scaling of the flow variables, (2) a physics-based domain decomposition, (3) a cluster-based flow representation for each subdomain, (4) an information entropy correlating the subdomains, and (5) a multi-variate probability function relating sensor input and targeted velocity estimates. This framework is demonstrated using drone flight paths through a three-building cluster as a simple example. We anticipate adaptations and applications for estimating complete cities and incorporating weather input. 

**Abstract (ZH)**: 基于传感器的无人机轨迹流估计的物理知情机器学习框架：面向复杂城市地形的风速与不确定性估计及传感器优化 

---
