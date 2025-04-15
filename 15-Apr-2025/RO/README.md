# Co-optimizing Physical Reconfiguration Parameters and Controllers for an Origami-inspired Reconfigurable Manipulator 

**Title (ZH)**: 基于 Origami 灵感的可重构 manipulator 的物理重构参数与控制器的协同优化 

**Authors**: Zhe Chen, Li Chen, Hao Zhang, Jianguo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.10474)  

**Abstract**: Reconfigurable robots that can change their physical configuration post-fabrication have demonstrate their potential in adapting to different environments or tasks. However, it is challenging to determine how to optimally adjust reconfigurable parameters for a given task, especially when the controller depends on the robot's configuration. In this paper, we address this problem using a tendon-driven reconfigurable manipulator composed of multiple serially connected origami-inspired modules as an example. Under tendon actuation, these modules can achieve different shapes and motions, governed by joint stiffnesses (reconfiguration parameters) and the tendon displacements (control inputs). We leverage recent advances in co-optimization of design and control for robotic system to treat reconfiguration parameters as design variables and optimize them using reinforcement learning techniques. We first establish a forward model based on the minimum potential energy method to predict the shape of the manipulator under tendon actuations. Using the forward model as the environment dynamics, we then co-optimize the control policy (on the tendon displacements) and joint stiffnesses of the modules for goal reaching tasks while ensuring collision avoidance. Through co-optimization, we obtain optimized joint stiffness and the corresponding optimal control policy to enable the manipulator to accomplish the task that would be infeasible with fixed reconfiguration parameters (i.e., fixed joint stiffness). We envision the co-optimization framework can be extended to other reconfigurable robotic systems, enabling them to optimally adapt their configuration and behavior for diverse tasks and environments. 

**Abstract (ZH)**: 可重构机器人在后加工可改变物理配置的情况下展现出适应不同环境或任务的潜力。然而，对于给定任务如何最优调整可重构参数仍然具有挑战性，尤其是在控制器依赖于机器人配置的情况下。本文利用一个由多个串联的 Origami 启发模块组成的缆索驱动可重构 manipulator 为例，解决这一问题。在缆索驱动下，这些模块可以实现不同的形状和运动，由连接处的刚度（重构参数）和缆索位移（控制输入）控制。我们利用最近在机器人系统的设计与控制协同优化方面的进展，将重构参数作为设计变量，并采用强化学习技术对其进行优化。我们首先基于最小势能法建立一个前向模型来预测缆索驱动下的 manipulator 形状。利用前向模型作为环境动力学，我们进一步协同优化模块的控制策略（缆索位移）和关节刚度，以实现目标获取任务的同时避免碰撞。通过协同优化，我们获得最优关节刚度和相应的最优控制策略，使 manipulator 能够完成固定重构参数下无法实现的任务。我们设想这种协同优化框架可以扩展到其他可重构机器人系统，使它们能够最优地调整其配置和行为以适应多样化的任务和环境。 

---
# Region Based SLAM-Aware Exploration: Efficient and Robust Autonomous Mapping Strategy That Can Scale 

**Title (ZH)**: 区域导向的SLAM感知探索：一种高效稳健且可扩展的自主建图策略 

**Authors**: Megha Maheshwari, Sadeigh Rabiee, He Yin, Martin Labrie, Hang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.10416)  

**Abstract**: Autonomous exploration for mapping unknown large scale environments is a fundamental challenge in robotics, with efficiency in time, stability against map corruption and computational resources being crucial. This paper presents a novel approach to indoor exploration that addresses these key issues in existing methods. We introduce a Simultaneous Localization and Mapping (SLAM)-aware region-based exploration strategy that partitions the environment into discrete regions, allowing the robot to incrementally explore and stabilize each region before moving to the next one. This approach significantly reduces redundant exploration and improves overall efficiency. As the device finishes exploring a region and stabilizes it, we also perform SLAM keyframe marginalization, a technique which reduces problem complexity by eliminating variables, while preserving their essential information. To improves robustness and further enhance efficiency, we develop a check- point system that enables the robot to resume exploration from the last stable region in case of failures, eliminating the need for complete re-exploration. Our method, tested in real homes, office and simulations, outperforms state-of-the-art approaches. The improvements demonstrate substantial enhancements in various real world environments, with significant reductions in keyframe usage (85%), submap usage (50% office, 32% home), pose graph optimization time (78-80%), and exploration duration (10-15%). This region-based strategy with keyframe marginalization offers an efficient solution for autonomous robotic mapping. 

**Abstract (ZH)**: 自主探索未知大规模环境是机器人领域的一项基本挑战，时间效率、地图稳定性和计算资源的利用至关重要。本文提出了一种针对现有方法中关键问题的室内探索新策略。我们引入了一种Simultaneous Localization and Mapping (SLAM)-aware区域基探索策略，将环境划分为离散区域，使机器人能够在探索并稳定每个区域后才转移到下一个区域，从而显著减少了重复探索并提升了整体效率。随着设备完成对一个区域的探索和稳定，我们还执行了SLAM关键帧边缘化，该技术通过消除变量来减少问题复杂性，同时保留其本质信息。为了提高鲁棒性并进一步增强效率，我们开发了一种检查点系统，以便在出现故障时机器人可以从最后一个稳定区域继续探索，无需进行全面重探索。我们的方法在实际家居、办公室和模拟环境中测试，优于现有最先进的方法。改进在各种实际环境中的表现显著提升，关键帧使用量减少了85%，办公室分图使用量减少了50%，家庭分图使用量减少了32%，姿态图优化时间减少了78-80%，探索时间缩短了10-15%。区域基策略结合关键帧边缘化提供了自主机器人建图的一种高效解决方案。 

---
# Teacher Motion Priors: Enhancing Robot Locomotion over Challenging Terrain 

**Title (ZH)**: 教师动作先验：提升机器人在复杂地形上的运动能力 

**Authors**: Fangcheng Jin, Yuqi Wang, Peixin Ma, Guodong Yang, Pan Zhao, En Li, Zhengtao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10390)  

**Abstract**: Achieving robust locomotion on complex terrains remains a challenge due to high dimensional control and environmental uncertainties. This paper introduces a teacher prior framework based on the teacher student paradigm, integrating imitation and auxiliary task learning to improve learning efficiency and generalization. Unlike traditional paradigms that strongly rely on encoder-based state embeddings, our framework decouples the network design, simplifying the policy network and deployment. A high performance teacher policy is first trained using privileged information to acquire generalizable motion skills. The teacher's motion distribution is transferred to the student policy, which relies only on noisy proprioceptive data, via a generative adversarial mechanism to mitigate performance degradation caused by distributional shifts. Additionally, auxiliary task learning enhances the student policy's feature representation, speeding up convergence and improving adaptability to varying terrains. The framework is validated on a humanoid robot, showing a great improvement in locomotion stability on dynamic terrains and significant reductions in development costs. This work provides a practical solution for deploying robust locomotion strategies in humanoid robots. 

**Abstract (ZH)**: 基于教师学生范式的教师先验框架：通过模仿和辅助任务学习提高复杂地形上稳健运动控制的学习效率和泛化能力 

---
# Flying Hand: End-Effector-Centric Framework for Versatile Aerial Manipulation Teleoperation and Policy Learning 

**Title (ZH)**: 飞行手：以末端执行器为中心的通用空中操作与策略学习框架 

**Authors**: Guanqi He, Xiaofeng Guo, Luyi Tang, Yuanhang Zhang, Mohammadreza Mousaei, Jiahe Xu, Junyi Geng, Sebastian Scherer, Guanya Shi  

**Link**: [PDF](https://arxiv.org/pdf/2504.10334)  

**Abstract**: Aerial manipulation has recently attracted increasing interest from both industry and academia. Previous approaches have demonstrated success in various specific tasks. However, their hardware design and control frameworks are often tightly coupled with task specifications, limiting the development of cross-task and cross-platform algorithms. Inspired by the success of robot learning in tabletop manipulation, we propose a unified aerial manipulation framework with an end-effector-centric interface that decouples high-level platform-agnostic decision-making from task-agnostic low-level control. Our framework consists of a fully-actuated hexarotor with a 4-DoF robotic arm, an end-effector-centric whole-body model predictive controller, and a high-level policy. The high-precision end-effector controller enables efficient and intuitive aerial teleoperation for versatile tasks and facilitates the development of imitation learning policies. Real-world experiments show that the proposed framework significantly improves end-effector tracking accuracy, and can handle multiple aerial teleoperation and imitation learning tasks, including writing, peg-in-hole, pick and place, changing light bulbs, etc. We believe the proposed framework provides one way to standardize and unify aerial manipulation into the general manipulation community and to advance the field. Project website: this https URL. 

**Abstract (ZH)**: 空中操作 recently 吸引了工业和学术界的广泛关注。尽管以往的方法在各种具体任务中取得了成功，但它们的硬件设计和控制框架往往紧密耦合于特定任务的要求，限制了跨任务和跨平台算法的发展。受桌面操作中机器人学习成功经验的启发，我们提出了一种以效应器为中心的统一空中操作框架，该框架将高层的平台无关决策与任务无关的低层控制相分离。该框架包括一个全驱动六旋翼无人机和一个四自由度的机械臂、一个以效应器为中心的全身模型预测控制器以及一个高层策略。高精度效应器控制器使得高效的直观空中遥控操作成为可能，并促进了模仿学习策略的发展。实验证明，提出的方法显著提高了末端执行器的跟踪准确性，并能处理包括书写、孔插针、取放、更换灯泡等多种空中遥控操作和模仿学习任务。我们相信，提出的框架为将空中操作标准化和统一到一般操作社区提供了途径，并推动了该领域的发展。项目网站: [this URL](this https URL)。 

---
# Siamese Network with Dual Attention for EEG-Driven Social Learning: Bridging the Human-Robot Gap in Long-Tail Autonomous Driving 

**Title (ZH)**: 基于双注意力机制的孪生网络在长尾自主驾驶中的人机协作社会学习：缩小人机器人差距 

**Authors**: Xiaoshan Zhou, Carol C. Menassa, Vineet R. Kamat  

**Link**: [PDF](https://arxiv.org/pdf/2504.10296)  

**Abstract**: Robots with wheeled, quadrupedal, or humanoid forms are increasingly integrated into built environments. However, unlike human social learning, they lack a critical pathway for intrinsic cognitive development, namely, learning from human feedback during interaction. To understand human ubiquitous observation, supervision, and shared control in dynamic and uncertain environments, this study presents a brain-computer interface (BCI) framework that enables classification of Electroencephalogram (EEG) signals to detect cognitively demanding and safety-critical events. As a timely and motivating co-robotic engineering application, we simulate a human-in-the-loop scenario to flag risky events in semi-autonomous robotic driving-representative of long-tail cases that pose persistent bottlenecks to the safety performance of smart mobility systems and robotic vehicles. Drawing on recent advances in few-shot learning, we propose a dual-attention Siamese convolutional network paired with Dynamic Time Warping Barycenter Averaging approach to generate robust EEG-encoded signal representations. Inverse source localization reveals activation in Broadman areas 4 and 9, indicating perception-action coupling during task-relevant mental imagery. The model achieves 80% classification accuracy under data-scarce conditions and exhibits a nearly 100% increase in the utility of salient features compared to state-of-the-art methods, as measured through integrated gradient attribution. Beyond performance, this study contributes to our understanding of the cognitive architecture required for BCI agents-particularly the role of attention and memory mechanisms-in categorizing diverse mental states and supporting both inter- and intra-subject adaptation. Overall, this research advances the development of cognitive robotics and socially guided learning for service robots in complex built environments. 

**Abstract (ZH)**: 具有轮式、四足或人形形式的机器人越来越多地集成到建筑物环境中。然而，与人类社会学习不同，它们缺乏一个关键的认知发展途径，即在互动中从人类反馈中学习。为了理解人类在动态和不确定环境中的普遍观察、监督和共享控制，本研究提出了一个脑-机接口（BCI）框架，以实现脑电图（EEG）信号分类，检测认知要求高和安全关键事件。作为及时且富有动力的协作机器人工程应用，我们模拟了一种循环人类在环的场景，以标记出在半自主机器人驾驶中代表大量尾部案例的风险事件，这些案例持续阻碍着智能移动系统和机器人车辆的安全性能。基于近期在少样本学习方面的进展，我们提出了一种双注意结构Siamese卷积网络配以动态时间战争.DisplayNameAveraging方法，以生成稳健的EEG编码信号表示。逆源定位显示了布罗dmann区域4和9的激活，表明在任务相关心智成像期间存在感知-行动 coupling。该模型在数据稀缺条件下实现了80%的分类准确率，并且与最先进的方法相比，其显著特征的实用性提高了近100%，这通过综合梯度归因进行测量。超越性能，本研究还扩展了我们对BCI代理所需认知架构的理解，尤其是注意和记忆机制在分类不同心状状态和支持跨内个体适应方面的作用。总体而言，这项研究推进了认知机器人和社交引导服务机器人在复杂建筑物环境中的发展。 

---
# Ankle Exoskeletons in Walking and Load-Carrying Tasks: Insights into Biomechanics and Human-Robot Interaction 

**Title (ZH)**: 膝踝矫形器在行走和负重任务中的研究：生物力学与人机交互见解 

**Authors**: J.F. Almeida, J. André, C.P. Santos  

**Link**: [PDF](https://arxiv.org/pdf/2504.10294)  

**Abstract**: Background: Lower limb exoskeletons can enhance quality of life, but widespread adoption is limited by the lack of frameworks to assess their biomechanical and human-robot interaction effects, which are essential for developing adaptive and personalized control strategies. Understanding impacts on kinematics, muscle activity, and HRI dynamics is key to achieve improved usability of wearable robots. Objectives: We propose a systematic methodology evaluate an ankle exoskeleton's effects on human movement during walking and load-carrying (10 kg front pack), focusing on joint kinematics, muscle activity, and HRI torque signals. Materials and Methods: Using Xsens MVN (inertial motion capture), Delsys EMG, and a unilateral exoskeleton, three experiments were conducted: (1) isolated dorsiflexion/plantarflexion; (2) gait analysis (two subjects, passive/active modes); and (3) load-carrying under assistance. Results and Conclusions: The first experiment confirmed that the HRI sensor captured both voluntary and involuntary torques, providing directional torque insights. The second experiment showed that the device slightly restricted ankle range of motion (RoM) but supported normal gait patterns across all assistance modes. The exoskeleton reduced muscle activity, particularly in active mode. HRI torque varied according to gait phases and highlighted reduced synchronization, suggesting a need for improved support. The third experiment revealed that load-carrying increased GM and TA muscle activity, but the device partially mitigated user effort by reducing muscle activity compared to unassisted walking. HRI increased during load-carrying, providing insights into user-device dynamics. These results demonstrate the importance of tailoring exoskeleton evaluation methods to specific devices and users, while offering a framework for future studies on exoskeleton biomechanics and HRI. 

**Abstract (ZH)**: 背景：下肢外骨骼可以提升生活质量，但其广泛应用受限于缺乏评估其生物力学和人机器人交互效果的框架，这些是开发适应性和个性化控制策略的关键。理解运动学、肌肉活动和人机器人交互动力学的影响是实现可穿戴机器人更好易用性的关键。目的：我们提出了一种系统性方法，评估踝关节外骨骼在行走和负重（10 kg 前背包）时对外力作用对人体运动的影响，重点在于关节运动学、肌肉活动和人机器人交互扭矩信号。材料与方法：使用Xsens MVN（惯性动作捕捉）、Delsys EMG和单侧外骨骼进行了三项实验：（1）孤立的背屈/跖屈；（2）步态分析（两名受试者，被动/主动模式）；（3）辅助负重。结果与结论：第一项实验证实了人机器人交互传感器捕获了自愿和不自愿的扭矩，并提供了方向扭矩的见解。第二项实验表明，该设备轻微限制了踝关节运动范围，但支持所有辅助模式下的正常步态模式。外骨骼减少了肌肉活动，特别是在主动模式下。人机器人交互扭矩随着步行周期的不同而变化，突显了减少同步的现象，表明需要改进支持。第三项实验发现，负重增加了股四头肌和胫骨前肌的活动，但该设备通过减少与未辅助行走相比的肌肉活动部分缓解了用户的用力情况。负重状态下人机器人交互扭矩增加，提供了用户-设备动力学的见解。这些结果表明，对外骨骼评价方法进行定制以适应特定设备和用户的重要性，同时为未来外骨骼生物力学和人机器人交互的研究提供了一个框架。 

---
# Look-to-Touch: A Vision-Enhanced Proximity and Tactile Sensor for Distance and Geometry Perception in Robotic Manipulation 

**Title (ZH)**: 视知觉增强的近距离和触觉传感器：用于 robotic 操作的距离与几何感知 

**Authors**: Yueshi Dong, Jieji Ren, Zhenle Liu, Zhanxuan Peng, Zihao Yuan, Ningbin Zhang, Guoying Gu  

**Link**: [PDF](https://arxiv.org/pdf/2504.10280)  

**Abstract**: Camera-based tactile sensors provide robots with a high-performance tactile sensing approach for environment perception and dexterous manipulation. However, achieving comprehensive environmental perception still requires cooperation with additional sensors, which makes the system bulky and limits its adaptability to unstructured environments. In this work, we present a vision-enhanced camera-based dual-modality sensor, which realizes full-scale distance sensing from 50 cm to -3 mm while simultaneously keeping ultra-high-resolution texture sensing and reconstruction capabilities. Unlike conventional designs with fixed opaque gel layers, our sensor features a partially transparent sliding window, enabling mechanical switching between tactile and visual modes. For each sensing mode, a dynamic distance sensing model and a contact geometry reconstruction model are proposed. Through integration with soft robotic fingers, we systematically evaluate the performance of each mode, as well as in their synergistic operation. Experimental results show robust distance tracking across various speeds, nanometer-scale roughness detection, and sub-millimeter 3D texture reconstruction. The combination of both modalities improves the robot's efficiency in executing grasping tasks. Furthermore, the embedded mechanical transmission in the sensor allows for fine-grained intra-hand adjustments and precise manipulation, unlocking new capabilities for soft robotic hands. 

**Abstract (ZH)**: 基于视觉增强的相机双模态传感器实现全面距离感知与超高清纹理感测与重建 

---
# Vision based driving agent for race car simulation environments 

**Title (ZH)**: 基于视觉的赛车模拟环境驾驶代理 

**Authors**: Gergely Bári, László Palkovics  

**Link**: [PDF](https://arxiv.org/pdf/2504.10266)  

**Abstract**: In recent years, autonomous driving has become a popular field of study. As control at tire grip limit is essential during emergency situations, algorithms developed for racecars are useful for road cars too. This paper examines the use of Deep Reinforcement Learning (DRL) to solve the problem of grip limit driving in a simulated environment. Proximal Policy Optimization (PPO) method is used to train an agent to control the steering wheel and pedals of the vehicle, using only visual inputs to achieve professional human lap times. The paper outlines the formulation of the task of time optimal driving on a race track as a deep reinforcement learning problem, and explains the chosen observations, actions, and reward functions. The results demonstrate human-like learning and driving behavior that utilize maximum tire grip potential. 

**Abstract (ZH)**: 近年来，自动驾驶已成为一个热门的研究领域。由于在紧急情况下轮胎抓地力极限的控制至关重要，用于赛车的算法同样适用于普通车辆。本文探讨了使用深度强化学习（DRL）在模拟环境中解决轮胎抓地力极限驾驶问题的方法。采用 proximal policy optimization (PPO) 方法训练一个代理，使其仅通过视觉输入控制车辆的方向盘和踏板，实现专业级的人际圈速。本文阐述了将赛道上时间最优驾驶任务建模为深度强化学习问题的方法，并解释了所选择的观测、行动和奖励函数。实验结果展示了利用轮胎最大抓地力潜力的人类似驾驶行为。 

---
# A Quasi-Steady-State Black Box Simulation Approach for the Generation of g-g-g-v Diagrams 

**Title (ZH)**: 擬稳态黑箱仿真方法生成g-g-g-v图示 

**Authors**: Frederik Werner, Simon Sagmeister, Mattia Piccinini, Johannes Betz  

**Link**: [PDF](https://arxiv.org/pdf/2504.10225)  

**Abstract**: The classical g-g diagram, representing the achievable acceleration space for a vehicle, is commonly used as a constraint in trajectory planning and control due to its computational simplicity. To address non-planar road geometries, this concept can be extended to incorporate g-g constraints as a function of vehicle speed and vertical acceleration, commonly referred to as g-g-g-v diagrams. However, the estimation of g-g-g-v diagrams is an open problem. Existing simulation-based approaches struggle to isolate non-transient, open-loop stable states across all combinations of speed and acceleration, while optimization-based methods often require simplified vehicle equations and have potential convergence issues. In this paper, we present a novel, open-source, quasi-steady-state black box simulation approach that applies a virtual inertial force in the longitudinal direction. The method emulates the load conditions associated with a specified longitudinal acceleration while maintaining constant vehicle speed, enabling open-loop steering ramps in a purely QSS manner. Appropriate regulation of the ramp steer rate inherently mitigates transient vehicle dynamics when determining the maximum feasible lateral acceleration. Moreover, treating the vehicle model as a black box eliminates model mismatch issues, allowing the use of high-fidelity or proprietary vehicle dynamics models typically unsuited for optimization approaches. An open-source version of the proposed method is available at: this https URL 

**Abstract (ZH)**: 经典g-g图，表示车辆可实现的加速度空间，由于其计算简单常被用作轨迹规划和控制中的约束条件。为了应对非平面路形几何结构，该概念可以扩展为结合速度和垂向加速度的g-g约束，通常称为g-g-g-v图。然而，g-g-g-v图的估算仍是一个开放问题。现有基于仿真的方法难以在所有速度和加速度组合下隔离出非瞬态、开环稳定状态，而基于优化的方法通常需要简化车辆方程，且存在收敛性问题。在本文中，我们提出了一种新颖的开源准稳态黑盒仿真方法，在纵向方向应用虚拟惯性力。该方法模拟了指定纵向加速度关联的载荷条件，同时保持恒定车速，以纯准稳态方式实现开环转向坡道。合理调节坡道转向速率本身可以缓解确定最大可行侧向加速度时的瞬态车辆动力学。此外，将车辆模型视为黑盒消除了模型不匹配问题，允许使用通常不适用于优化方法的高保真或专有车辆动力学模型。所提方法的开源版本可在以下链接获取：this https URL 

---
# Shoulder Range of Motion Rehabilitation Robot Incorporating Scapulohumeral Rhythm for Frozen Shoulder 

**Title (ZH)**: 肩关节活动度康复机器人结合肩锁关节运动规律用于治疗粘连性肩关节囊炎 

**Authors**: Hyunbum Cho, Sungmoon Hur, Joowan Kim, Keewon Kim, Jaeheung Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.10163)  

**Abstract**: This paper presents a novel rehabilitation robot designed to address the challenges of passive range of motion (PROM) exercises for frozen shoulder patients by integrating advanced scapulohumeral rhythm stabilization. Frozen shoulder is characterized by limited glenohumeral motion and disrupted scapulohumeral rhythm, with therapist-assisted interventions being highly effective for restoring normal shoulder function. While existing robotic solutions replicate natural shoulder biomechanics, they lack the ability to stabilize compensatory movements, such as shoulder shrugging, which are critical for effective rehabilitation. Our proposed device features a 6 degrees of freedom (DoF) mechanism, including 5 DoF for shoulder motion and an innovative 1 DoF Joint press for scapular stabilization. The robot employs a personalized two-phase operation: recording normal shoulder movement patterns from the unaffected side and applying them to guide the affected side. Experimental results demonstrated the robot's ability to replicate recorded motion patterns with high precision, with root mean square error (RMSE) values consistently below 1 degree. In simulated frozen shoulder conditions, the robot effectively suppressed scapular elevation, delaying the onset of compensatory movements and guiding the affected shoulder to move more closely in alignment with normal shoulder motion, particularly during arm elevation movements such as abduction and flexion. These findings confirm the robot's potential as a rehabilitation tool capable of automating PROM exercises while correcting compensatory movements. The system provides a foundation for advanced, personalized rehabilitation for patients with frozen shoulders. 

**Abstract (ZH)**: 一种用于治疗冻结肩患者被动关节活动范围锻炼的新型康复机器人及其应用研究 

---
# A Human-Sensitive Controller: Adapting to Human Ergonomics and Physical Constraints via Reinforcement Learning 

**Title (ZH)**: 人性化控制器：强化学习适应人类人体工学和物理约束 

**Authors**: Vitor Martins, Sara M. Cerqueira, Mercedes Balcells, Elazer R Edelman, Cristina P. Santos  

**Link**: [PDF](https://arxiv.org/pdf/2504.10102)  

**Abstract**: Work-Related Musculoskeletal Disorders continue to be a major challenge in industrial environments, leading to reduced workforce participation, increased healthcare costs, and long-term disability. This study introduces a human-sensitive robotic system aimed at reintegrating individuals with a history of musculoskeletal disorders into standard job roles, while simultaneously optimizing ergonomic conditions for the broader workforce. This research leverages reinforcement learning to develop a human-aware control strategy for collaborative robots, focusing on optimizing ergonomic conditions and preventing pain during task execution. Two RL approaches, Q-Learning and Deep Q-Network (DQN), were implemented and tested to personalize control strategies based on individual user characteristics. Although experimental results revealed a simulation-to-real gap, a fine-tuning phase successfully adapted the policies to real-world conditions. DQN outperformed Q-Learning by completing tasks faster while maintaining zero pain risk and safe ergonomic levels. The structured testing protocol confirmed the system's adaptability to diverse human anthropometries, underscoring the potential of RL-driven cobots to enable safer, more inclusive workplaces. 

**Abstract (ZH)**: 工关联肌骨骼紊乱仍然是工业环境中的一项重大挑战，导致劳动力参与率降低、医疗成本增加和长期残疾。本研究介绍了一种以人为本的机器人系统，旨在将具有肌骨骼疾病史的个体重新融入标准工作岗位，同时优化更广泛的劳动力的工效条件。该研究利用强化学习开发了一种以人为本的协作机器人控制策略，侧重于优化工效条件并在执行任务时预防疼痛。实现了两种RL方法，即Q-Learning和深度Q网络（DQN），并进行了测试，以根据个体用户特性个性化控制策略。尽管实验结果表明存在仿真到现实的差距，但细调阶段成功地将策略适应了实际条件。DQN在完成任务速度更快的同时保持了零疼痛风险和安全工效水平，结构化测试协议证实了该系统对多样的人体测量的适应性，强调了RL驱动的协作机器人在实现更安全、更包容的工作场所方面的潜力。 

---
# A Framework for Adaptive Load Redistribution in Human-Exoskeleton-Cobot Systems 

**Title (ZH)**: 人类-外骨骼-协作机器人系统中自适应负载重新分布的框架 

**Authors**: Emir Mobedi, Gokhan Solak, Arash Ajoudani  

**Link**: [PDF](https://arxiv.org/pdf/2504.10066)  

**Abstract**: Wearable devices like exoskeletons are designed to reduce excessive loads on specific joints of the body. Specifically, single- or two-degrees-of-freedom (DOF) upper-body industrial exoskeletons typically focus on compensating for the strain on the elbow and shoulder joints. However, during daily activities, there is no assurance that external loads are correctly aligned with the supported joints. Optimizing work processes to ensure that external loads are primarily (to the extent that they can be compensated by the exoskeleton) directed onto the supported joints can significantly enhance the overall usability of these devices and the ergonomics of their users. Collaborative robots (cobots) can play a role in this optimization, complementing the collaborative aspects of human work. In this study, we propose an adaptive and coordinated control system for the human-cobot-exoskeleton interaction. This system adjusts the task coordinates to maximize the utilization of the supported joints. When the torque limits of the exoskeleton are exceeded, the framework continuously adapts the task frame, redistributing excessive loads to non-supported body joints to prevent overloading the supported ones. We validated our approach in an equivalent industrial painting task involving a single-DOF elbow exoskeleton, a cobot, and four subjects, each tested in four different initial arm configurations with five distinct optimisation weight matrices and two different payloads. 

**Abstract (ZH)**: 可穿戴设备如外骨骼旨在减轻身体特定关节的过度负荷。具体而言，单自由度或双自由度上肢工业外骨骼通常侧重于缓解肘关节和肩关节的紧张。然而，在日常活动中，并不能保证外部负荷正确对准支持的关节。通过优化工作流程，确保外部负荷主要（在可以被外骨骼补偿的范围内）作用在支持的关节上，可以显著提升这些设备的整体可用性和用户的工效学。协作机器人（ cobots）可以在这一优化中发挥作用，补充人类工作的协作性。在本研究中，我们提出了一种适应性和协调性的控制系统，用于人类-协作机器人-外骨骼交互。该系统调整任务坐标，以最大化利用支持的关节。当外骨骼的扭矩限制被超过时，该框架会持续适应任务框架，重新分配过度的负荷至非支持的体关节，以防止过度负担支持的关节。我们通过一项包含单自由度肘关节外骨骼、一个协作机器人和四名受试者的等效工业喷漆任务进行了验证，每位受试者在四种不同的初始臂配置下，分别使用五个不同的优化权重矩阵和两种不同负载进行测试。 

---
# Joint Action Language Modelling for Transparent Policy Execution 

**Title (ZH)**: 联合动作语言建模以实现透明政策执行 

**Authors**: Theodor Wulff, Rahul Singh Maharjan, Xinyun Chi, Angelo Cangelosi  

**Link**: [PDF](https://arxiv.org/pdf/2504.10055)  

**Abstract**: An agent's intention often remains hidden behind the black-box nature of embodied policies. Communication using natural language statements that describe the next action can provide transparency towards the agent's behavior. We aim to insert transparent behavior directly into the learning process, by transforming the problem of policy learning into a language generation problem and combining it with traditional autoregressive modelling. The resulting model produces transparent natural language statements followed by tokens representing the specific actions to solve long-horizon tasks in the Language-Table environment. Following previous work, the model is able to learn to produce a policy represented by special discretized tokens in an autoregressive manner. We place special emphasis on investigating the relationship between predicting actions and producing high-quality language for a transparent agent. We find that in many cases both the quality of the action trajectory and the transparent statement increase when they are generated simultaneously. 

**Abstract (ZH)**: 一种代理的意图往往被其体内策略的黑箱性质所隐藏。使用自然语言语句描述下一个动作的通信可以提供对代理行为的透明度。我们旨在通过将策略学习问题转变为语言生成问题，并结合传统的自回归建模，直接在学习过程中插入透明行为。该模型生成透明的自然语言语句，随后是表示具体动作的标记，以解决Language-Table环境中的长期任务。借鉴先前的工作，该模型能够以自回归方式学习产生由特殊离散标记表示的策略。我们特别关注预测动作与生成高质量透明语言之间关系的研究。我们发现，在许多情况下，当动作轨迹和透明语句同时生成时，它们的质量都会提高。 

---
# Prior Does Matter: Visual Navigation via Denoising Diffusion Bridge Models 

**Title (ZH)**: Prior Does Matter: Visual Navigation via Denoising Diffusion Bridge Models 

**Authors**: Hao Ren, Yiming Zeng, Zetong Bi, Zhaoliang Wan, Junlong Huang, Hui Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.10041)  

**Abstract**: Recent advancements in diffusion-based imitation learning, which show impressive performance in modeling multimodal distributions and training stability, have led to substantial progress in various robot learning tasks. In visual navigation, previous diffusion-based policies typically generate action sequences by initiating from denoising Gaussian noise. However, the target action distribution often diverges significantly from Gaussian noise, leading to redundant denoising steps and increased learning complexity. Additionally, the sparsity of effective action distributions makes it challenging for the policy to generate accurate actions without guidance. To address these issues, we propose a novel, unified visual navigation framework leveraging the denoising diffusion bridge models named NaviBridger. This approach enables action generation by initiating from any informative prior actions, enhancing guidance and efficiency in the denoising process. We explore how diffusion bridges can enhance imitation learning in visual navigation tasks and further examine three source policies for generating prior actions. Extensive experiments in both simulated and real-world indoor and outdoor scenarios demonstrate that NaviBridger accelerates policy inference and outperforms the baselines in generating target action sequences. Code is available at this https URL. 

**Abstract (ZH)**: 基于弥散的 imitation 学习 Recent 进展展示了在建模多模态分布和训练稳定性方面的出色性能，这在多种机器人学习任务中取得了显著进步。在视觉导航中，先前的基于弥散的策略通常从去噪高斯噪声开始生成动作序列。然而，目标动作分布往往与高斯噪声有显著差异，导致不必要的去噪步骤并增加了学习复杂性。此外，有效动作分布的稀疏性使策略在没有引导的情况下生成准确动作变得极具挑战性。为解决这些问题，我们提出了一种新颖的统一视觉导航框架，利用去噪弥散桥梁模型 NaviBridger。此方法允许从任何信息性先验动作开始生成动作，增强去噪过程中的引导和效率。我们探讨了弥散桥梁如何在视觉导航任务中增强 imitation 学习，并进一步研究了三种用于生成先验动作的源策略。在仿真和真实世界室内外场景中的 extensive 实验表明，NaviBridger 加快了策略推理并优于基线方法，在生成目标动作序列方面表现更优。代码可在以下链接获取：this https URL。 

---
# EmbodiedAgent: A Scalable Hierarchical Approach to Overcome Practical Challenge in Multi-Robot Control 

**Title (ZH)**: embodiable代理:一种克服多机器人控制实用挑战的可扩展分层方法 

**Authors**: Hanwen Wan, Yifei Chen, Zeyu Wei, Dongrui Li, Zexin Lin, Donghao Wu, Jiu Cheng, Yuxiang Zhang, Xiaoqiang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2504.10030)  

**Abstract**: This paper introduces EmbodiedAgent, a hierarchical framework for heterogeneous multi-robot control. EmbodiedAgent addresses critical limitations of hallucination in impractical tasks. Our approach integrates a next-action prediction paradigm with a structured memory system to decompose tasks into executable robot skills while dynamically validating actions against environmental constraints. We present MultiPlan+, a dataset of more than 18,000 annotated planning instances spanning 100 scenarios, including a subset of impractical cases to mitigate hallucination. To evaluate performance, we propose the Robot Planning Assessment Schema (RPAS), combining automated metrics with LLM-aided expert grading. Experiments demonstrate EmbodiedAgent's superiority over state-of-the-art models, achieving 71.85% RPAS score. Real-world validation in an office service task highlights its ability to coordinate heterogeneous robots for long-horizon objectives. 

**Abstract (ZH)**: 本文介绍了EmbodiedAgent，这是一种分层框架，用于异质多机器人控制。EmbodiedAgent解决了在不切实际任务中幻觉的关键限制。我们的方法结合了下一动作预测范式和结构化记忆系统，将任务分解为可执行的机器人技能，并动态验证动作是否符合环境约束。我们提出了一个多plan+数据集，其中包括超过18,000个标注的规划实例，涵盖100个场景，其中包括一些不切实际的案例以减轻幻觉。为了评估性能，我们提出了一种机器人规划评估方案（RPAS），结合了自动化指标和LLM辅助专家评分。实验结果表明，EmbodiedAgent在与现有最佳模型相比时表现出优越性，获得了71.85%的RPAS评分。在办公室服务任务中的实际验证展示了其协调异构机器人实现长时目标的能力。 

---
# KeyMPs: One-Shot Vision-Language Guided Motion Generation by Sequencing DMPs for Occlusion-Rich Tasks 

**Title (ZH)**: KeyMPs: 通过序列化DMPs实现一-shot视觉-语言引导的运动生成，用于遮挡丰富的任务 

**Authors**: Edgar Anarossi, Yuhwan Kwon, Hirotaka Tahara, Shohei Tanaka, Keisuke Shirai, Masashi Hamaya, Cristian C. Beltran-Hernandez, Atsushi Hashimoto, Takamitsu Matsubara  

**Link**: [PDF](https://arxiv.org/pdf/2504.10011)  

**Abstract**: Dynamic Movement Primitives (DMPs) provide a flexible framework wherein smooth robotic motions are encoded into modular parameters. However, they face challenges in integrating multimodal inputs commonly used in robotics like vision and language into their framework. To fully maximize DMPs' potential, enabling them to handle multimodal inputs is essential. In addition, we also aim to extend DMPs' capability to handle object-focused tasks requiring one-shot complex motion generation, as observation occlusion could easily happen mid-execution in such tasks (e.g., knife occlusion in cake icing, hand occlusion in dough kneading, etc.). A promising approach is to leverage Vision-Language Models (VLMs), which process multimodal data and can grasp high-level concepts. However, they typically lack enough knowledge and capabilities to directly infer low-level motion details and instead only serve as a bridge between high-level instructions and low-level control. To address this limitation, we propose Keyword Labeled Primitive Selection and Keypoint Pairs Generation Guided Movement Primitives (KeyMPs), a framework that combines VLMs with sequencing of DMPs. KeyMPs use VLMs' high-level reasoning capability to select a reference primitive through keyword labeled primitive selection and VLMs' spatial awareness to generate spatial scaling parameters used for sequencing DMPs by generalizing the overall motion through keypoint pairs generation, which together enable one-shot vision-language guided motion generation that aligns with the intent expressed in the multimodal input. We validate our approach through an occlusion-rich manipulation task, specifically object cutting experiments in both simulated and real-world environments, demonstrating superior performance over other DMP-based methods that integrate VLMs support. 

**Abstract (ZH)**: 动态运动模块（KeyMPs）结合视觉语言模型实现多模态输入的复杂运动一次生成 

---
# NaviDiffusor: Cost-Guided Diffusion Model for Visual Navigation 

**Title (ZH)**: NaviDiffusor：成本导向的视觉导航扩散模型 

**Authors**: Yiming Zeng, Hao Ren, Shuhang Wang, Junlong Huang, Hui Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.10003)  

**Abstract**: Visual navigation, a fundamental challenge in mobile robotics, demands versatile policies to handle diverse environments. Classical methods leverage geometric solutions to minimize specific costs, offering adaptability to new scenarios but are prone to system errors due to their multi-modular design and reliance on hand-crafted rules. Learning-based methods, while achieving high planning success rates, face difficulties in generalizing to unseen environments beyond the training data and often require extensive training. To address these limitations, we propose a hybrid approach that combines the strengths of learning-based methods and classical approaches for RGB-only visual navigation. Our method first trains a conditional diffusion model on diverse path-RGB observation pairs. During inference, it integrates the gradients of differentiable scene-specific and task-level costs, guiding the diffusion model to generate valid paths that meet the constraints. This approach alleviates the need for retraining, offering a plug-and-play solution. Extensive experiments in both indoor and outdoor settings, across simulated and real-world scenarios, demonstrate zero-shot transfer capability of our approach, achieving higher success rates and fewer collisions compared to baseline methods. Code will be released at this https URL. 

**Abstract (ZH)**: 视觉导航，移动机器人领域的基本挑战，要求具有多样性的策略以应对不同的环境。经典方法利用几何解决方案以最小化特定成本，虽然能适应新的场景，但由于其多模块设计和依赖人工设计的规则，容易出现系统错误。基于学习的方法虽然能实现高规划成功率，但在处理超出训练数据之外的未见过的环境时面临泛化问题，且通常需要大量的训练。为解决这些问题，我们提出了一种结合基于学习方法和经典方法的混合方法，用于仅RGB视觉导航。该方法首先在一个包含多种路径-RGB观测对的数据集上训练一个条件扩散模型。在推理过程中，它整合了可微场景特定和任务级成本的梯度，引导扩散模型生成满足约束的有效路径。这种方法减轻了重新训练的需求，提供了一种即插即用的解决方案。在室内和室外环境、模拟和现实场景中的 extensive 实验表明，该方法在零样本迁移能力方面优于基线方法，实现更高的成功率和更少的碰撞。代码将在以下链接发布：this https URL。 

---
# FLoRA: Sample-Efficient Preference-based RL via Low-Rank Style Adaptation of Reward Functions 

**Title (ZH)**: FLoRA: 基于偏好低秩风格适配奖励函数的样本高效强化学习 

**Authors**: Daniel Marta, Simon Holk, Miguel Vasco, Jens Lundell, Timon Homberger, Finn Busch, Olov Andersson, Danica Kragic, Iolanda Leite  

**Link**: [PDF](https://arxiv.org/pdf/2504.10002)  

**Abstract**: Preference-based reinforcement learning (PbRL) is a suitable approach for style adaptation of pre-trained robotic behavior: adapting the robot's policy to follow human user preferences while still being able to perform the original task. However, collecting preferences for the adaptation process in robotics is often challenging and time-consuming. In this work we explore the adaptation of pre-trained robots in the low-preference-data regime. We show that, in this regime, recent adaptation approaches suffer from catastrophic reward forgetting (CRF), where the updated reward model overfits to the new preferences, leading the agent to become unable to perform the original task. To mitigate CRF, we propose to enhance the original reward model with a small number of parameters (low-rank matrices) responsible for modeling the preference adaptation. Our evaluation shows that our method can efficiently and effectively adjust robotic behavior to human preferences across simulation benchmark tasks and multiple real-world robotic tasks. 

**Abstract (ZH)**: 基于偏好增强学习的预训练机器人适应性研究：低偏好数据下的行为调整 

---
# GenTe: Generative Real-world Terrains for General Legged Robot Locomotion Control 

**Title (ZH)**: GenTe: 生成的现实地形用于通用腿式机器人运动控制 

**Authors**: Hanwen Wan, Mengkang Li, Donghao Wu, Yebin Zhong, Yixuan Deng, Zhenglong Sun, Xiaoqiang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2504.09997)  

**Abstract**: Developing bipedal robots capable of traversing diverse real-world terrains presents a fundamental robotics challenge, as existing methods using predefined height maps and static environments fail to address the complexity of unstructured landscapes. To bridge this gap, we propose GenTe, a framework for generating physically realistic and adaptable terrains to train generalizable locomotion policies. GenTe constructs an atomic terrain library that includes both geometric and physical terrains, enabling curriculum training for reinforcement learning-based locomotion policies. By leveraging function-calling techniques and reasoning capabilities of Vision-Language Models (VLMs), GenTe generates complex, contextually relevant terrains from textual and graphical inputs. The framework introduces realistic force modeling for terrain interactions, capturing effects such as soil sinkage and hydrodynamic resistance. To the best of our knowledge, GenTe is the first framework that systemically generates simulation environments for legged robot locomotion control. Additionally, we introduce a benchmark of 100 generated terrains. Experiments demonstrate improved generalization and robustness in bipedal robot locomotion. 

**Abstract (ZH)**: 开发能够穿越多样化现实地形的两足机器人呈现了一项基础的机器人学挑战，现有方法依赖预定义的高度图和静态环境无法应对不规则地形的复杂性。为了弥合这一差距，我们提出了GenTe框架，该框架用于生成物理上真实且可适应的地形以训练通用的运动策略。GenTe构建了一个包含几何和物理地形的原子地形库，使基于强化学习的运动策略能够进行阶梯式训练。通过利用函数调用技术和视觉语言模型（VLMs）的推理能力，GenTe能够从文本和图形输入中生成复杂且上下文相关的真实地形。该框架引入了地形交互的现实力模型，捕捉诸如土壤下陷和水动力阻力等效果。据我们所知，GenTe是首个系统性生成用于腿部机器人运动控制模拟环境的框架。此外，我们还引入了一个包含100种生成地形的基准测试集。实验结果表明，这对两足机器人运动的泛化能力和 robustness 有所提升。 

---
# Efficient Task-specific Conditional Diffusion Policies: Shortcut Model Acceleration and SO(3) Optimization 

**Title (ZH)**: 任务特定条件扩散策略的高效加速模型与SO(3)优化 

**Authors**: Haiyong Yu, Yanqiong Jin, Yonghao He, Wei Sui  

**Link**: [PDF](https://arxiv.org/pdf/2504.09927)  

**Abstract**: Imitation learning, particularly Diffusion Policies based methods, has recently gained significant traction in embodied AI as a powerful approach to action policy generation. These models efficiently generate action policies by learning to predict noise. However, conventional Diffusion Policy methods rely on iterative denoising, leading to inefficient inference and slow response times, which hinder real-time robot control. To address these limitations, we propose a Classifier-Free Shortcut Diffusion Policy (CF-SDP) that integrates classifier-free guidance with shortcut-based acceleration, enabling efficient task-specific action generation while significantly improving inference speed. Furthermore, we extend diffusion modeling to the SO(3) manifold in shortcut model, defining the forward and reverse processes in its tangent space with an isotropic Gaussian distribution. This ensures stable and accurate rotational estimation, enhancing the effectiveness of diffusion-based control. Our approach achieves nearly 5x acceleration in diffusion inference compared to DDIM-based Diffusion Policy while maintaining task performance. Evaluations both on the RoboTwin simulation platform and real-world scenarios across various tasks demonstrate the superiority of our method. 

**Abstract (ZH)**: 无监督学习，尤其是基于扩散策略的方法，在体现人工智能中的动作策略生成方面 recently gained significant traction. 这些模型通过学习预测噪声高效地生成动作策略。然而，传统的扩散策略方法依赖于迭代去噪，导致推断效率低下和响应时间缓慢，这阻碍了实时机器人控制。为了解决这些限制，我们提出了一种无分类器快捷扩散策略（CF-SDP），该方法将无分类器引导与基于快捷路径的加速相结合，在实现任务特定动作生成的同时显著提高推断速度。此外，我们将扩散建模扩展到SO(3)流形，在快捷模型中定义其切空间中的前向和反向过程，使用各向同性高斯分布。这确保了稳定的准确的旋转估计，增强了基于扩散的控制的有效性。我们的方法在与DDIM基于的扩散策略相比，实现了约5倍的扩散推断加速，同时保持任务性能。在RoboTwin仿真平台及各类真实场景下进行的评估均证明了该方法的优势。 

---
# LangPert: Detecting and Handling Task-level Perturbations for Robust Object Rearrangement 

**Title (ZH)**: LangPert：检测和处理任务级扰动以实现稳健的物体重新排列 

**Authors**: Xu Yin, Min-Sung Yoon, Yuchi Huo, Kang Zhang, Sung-Eui Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2504.09893)  

**Abstract**: Task execution for object rearrangement could be challenged by Task-Level Perturbations (TLP), i.e., unexpected object additions, removals, and displacements that can disrupt underlying visual policies and fundamentally compromise task feasibility and progress. To address these challenges, we present LangPert, a language-based framework designed to detect and mitigate TLP situations in tabletop rearrangement tasks. LangPert integrates a Visual Language Model (VLM) to comprehensively monitor policy's skill execution and environmental TLP, while leveraging the Hierarchical Chain-of-Thought (HCoT) reasoning mechanism to enhance the Large Language Model (LLM)'s contextual understanding and generate adaptive, corrective skill-execution plans. Our experimental results demonstrate that LangPert handles diverse TLP situations more effectively than baseline methods, achieving higher task completion rates, improved execution efficiency, and potential generalization to unseen scenarios. 

**Abstract (ZH)**: 基于语言的框架LangPert在桌面上对象重排任务中检测和缓解任务级扰动（TLP）的能力 

---
# SIO-Mapper: A Framework for Lane-Level HD Map Construction Using Satellite Images and OpenStreetMap with No On-Site Visits 

**Title (ZH)**: SIO-Mapper：基于卫星图像和OpenStreetMap的无需现场勘查的车道级高清地图构建框架 

**Authors**: Younghun Cho, Jee-Hwan Ryu  

**Link**: [PDF](https://arxiv.org/pdf/2504.09882)  

**Abstract**: High-definition (HD) maps, particularly those containing lane-level information regarded as ground truth, are crucial for vehicle localization research. Traditionally, constructing HD maps requires highly accurate sensor measurements collection from the target area, followed by manual annotation to assign semantic information. Consequently, HD maps are limited in terms of geographic coverage. To tackle this problem, in this paper, we propose SIO-Mapper, a novel lane-level HD map construction framework that constructs city-scale maps without physical site visits by utilizing satellite images and OpenStreetmap data. One of the key contributions of SIO-Mapper is its ability to extract lane information more accurately by introducing SIO-Net, a novel deep learning network that integrates features from satellite image and OpenStreetmap using both Transformer-based and convolution-based encoders. Furthermore, to overcome challenges in merging lanes over large areas, we introduce a novel lane integration methodology that combines cluster-based and graph-based approaches. This algorithm ensures the seamless aggregation of lane segments with high accuracy and coverage, even in complex road environments. We validated SIO-Mapper on the Naver Labs Open Dataset and NuScenes dataset, demonstrating better performance in various environments including Korea, the United States, and Singapore compared to the state-of-the-art lane-level HD mapconstruction methods. 

**Abstract (ZH)**: 高-definition (HD) 地图，特别是包含车道级信息且作为ground truth的HD地图，对于车辆定位研究至关重要。传统的HD地图构建方法需要在目标区域收集高精度传感器测量数据，并进行人工标注以分配语义信息。因此，HD地图在地理覆盖范围上受到限制。为解决这一问题，本文提出了一种名为SIO-Mapper的新颖车道级HD地图构建框架，该框架通过利用卫星图像和OpenStreetMap数据构建城市规模的地图，而无需进行实地考察。SIO-Mapper的一个关键贡献是通过引入结合卫星图像和OpenStreetMap特征的新型深度学习网络SIO-Net来更准确地提取车道信息，该网络集成了基于Transformer和卷积编码器。此外，为了克服大规模区域车道合并的挑战，我们引入了一种新的车道合并方法，结合了基于聚类和基于图的方法。该算法确保即使在复杂道路环境中也能以高度准确和广泛的精度无缝聚合车道片段。我们在Naver Labs Open Dataset和NuScenes数据集上验证了SIO-Mapper，结果表明在韩国、美国和新加坡等不同环境中，其性能优于最先进的车道级HD地图构建方法。 

---
# NeRF-Based Transparent Object Grasping Enhanced by Shape Priors 

**Title (ZH)**: 基于NeRF的形状先验增强透明物体抓取 

**Authors**: Yi Han, Zixin Lin, Dongjie Li, Lvping Chen, Yongliang Shi, Gan Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.09868)  

**Abstract**: Transparent object grasping remains a persistent challenge in robotics, largely due to the difficulty of acquiring precise 3D information. Conventional optical 3D sensors struggle to capture transparent objects, and machine learning methods are often hindered by their reliance on high-quality datasets. Leveraging NeRF's capability for continuous spatial opacity modeling, our proposed architecture integrates a NeRF-based approach for reconstructing the 3D information of transparent objects. Despite this, certain portions of the reconstructed 3D information may remain incomplete. To address these deficiencies, we introduce a shape-prior-driven completion mechanism, further refined by a geometric pose estimation method we have developed. This allows us to obtain a complete and reliable 3D information of transparent objects. Utilizing this refined data, we perform scene-level grasp prediction and deploy the results in real-world robotic systems. Experimental validation demonstrates the efficacy of our architecture, showcasing its capability to reliably capture 3D information of various transparent objects in cluttered scenes, and correspondingly, achieve high-quality, stables, and executable grasp predictions. 

**Abstract (ZH)**: 透明物体抓取在机器人技术中 remains a persistent challenge largely due to the difficulty of acquiring precise 3D information. 翻译为：

透明物体抓取在机器人技术中仍然是一个持久的挑战，主要是由于难以获取精确的3D信息。 

---
# PreCi: Pretraining and Continual Improvement of Humanoid Locomotion via Model-Assumption-Based Regularization 

**Title (ZH)**: 基于模型假设正则化的预训练与持续改进 humanoid 行走研究 

**Authors**: Hyunyoung Jung, Zhaoyuan Gu, Ye Zhao, Hae-Won Park, Sehoon Ha  

**Link**: [PDF](https://arxiv.org/pdf/2504.09833)  

**Abstract**: Humanoid locomotion is a challenging task due to its inherent complexity and high-dimensional dynamics, as well as the need to adapt to diverse and unpredictable environments. In this work, we introduce a novel learning framework for effectively training a humanoid locomotion policy that imitates the behavior of a model-based controller while extending its capabilities to handle more complex locomotion tasks, such as more challenging terrain and higher velocity commands. Our framework consists of three key components: pre-training through imitation of the model-based controller, fine-tuning via reinforcement learning, and model-assumption-based regularization (MAR) during fine-tuning. In particular, MAR aligns the policy with actions from the model-based controller only in states where the model assumption holds to prevent catastrophic forgetting. We evaluate the proposed framework through comprehensive simulation tests and hardware experiments on a full-size humanoid robot, Digit, demonstrating a forward speed of 1.5 m/s and robust locomotion across diverse terrains, including slippery, sloped, uneven, and sandy terrains. 

**Abstract (ZH)**: 基于模型控制器的模仿与假设正则化的类人运动学习框架 

---
# RoboCup Rescue 2025 Team Description Paper UruBots 

**Title (ZH)**: 2025年RoboCup救援赛队伍描述论文 UruBots 

**Authors**: Kevin Farias, Pablo Moraes, Igor Nunes, Juan Deniz, Sebastian Barcelona, Hiago Sodre, William Moraes, Monica Rodriguez, Ahilen Mazondo, Vincent Sandin, Gabriel da Silva, Victoria Saravia, Vinicio Melgar, Santiago Fernandez, Ricardo Grando  

**Link**: [PDF](https://arxiv.org/pdf/2504.09778)  

**Abstract**: This paper describes the approach used by Team UruBots for participation in the 2025 RoboCup Rescue Robot League competition. Our team aims to participate for the first time in this competition at RoboCup, using experience learned from previous competitions and research. We present our vehicle and our approach to tackle the task of detecting and finding victims in search and rescue environments. Our approach contains known topics in robotics, such as ROS, SLAM, Human Robot Interaction and segmentation and perception. Our proposed approach is open source, available to the RoboCup Rescue community, where we aim to learn and contribute to the league. 

**Abstract (ZH)**: 乌兔邦队参加2025年机器人世界杯救援机器人联赛的方法描述 

---
# UruBots RoboCup Work Team Description Paper 

**Title (ZH)**: 乌鲁机器人杯 RoboCup 工作组描述论文 

**Authors**: Hiago Sodre, Juan Deniz, Pablo Moraes, William Moraes, Igor Nunes, Vincent Sandin, Ahilen Mazondo, Santiago Fernandez, Gabriel da Silva, Monica Rodriguez, Sebastian Barcelona, Ricardo Grando  

**Link**: [PDF](https://arxiv.org/pdf/2504.09755)  

**Abstract**: This work presents a team description paper for the RoboCup Work League. Our team, UruBots, has been developing robots and projects for research and competitions in the last three years, attending robotics competitions in Uruguay and around the world. In this instance, we aim to participate and contribute to the RoboCup Work category, hopefully making our debut in this prestigious competition. For that, we present an approach based on the Limo robot, whose main characteristic is its hybrid locomotion system with wheels and tracks, with some extras added by the team to complement the robot's functionalities. Overall, our approach allows the robot to efficiently and autonomously navigate a Work scenario, with the ability to manipulate objects, perform autonomous navigation, and engage in a simulated industrial environment. 

**Abstract (ZH)**: 本研究论文介绍了参加RoboCup工作组的UruBots团队。在过去三年里，我们一直在研发机器人和项目，参加乌拉圭及世界各地的机器人竞赛。此次，我们旨在参加RoboCup工作类别竞赛，希望能首次亮相这一 prestigioius 比赛。为此，我们提出了基于Limo机器人的方法，该方法的主要特点是其轮式和履带式混合移动系统，并且团队添加了一些额外功能以补充机器人的功能。总体而言，我们的方法使机器人能够在工作场景中高效自主导航，具备操控物体、自主导航以及模拟工业环境的能力。 

---
# Adapting Robot's Explanation for Failures Based on Observed Human Behavior in Human-Robot Collaboration 

**Title (ZH)**: 基于人类行为观察调整机器人在人机协作中失败解释 

**Authors**: Andreas Naoum, Parag Khanna, Elmira Yadollahi, Mårten Björkman, Christian Smith  

**Link**: [PDF](https://arxiv.org/pdf/2504.09717)  

**Abstract**: This work aims to interpret human behavior to anticipate potential user confusion when a robot provides explanations for failure, allowing the robot to adapt its explanations for more natural and efficient collaboration. Using a dataset that included facial emotion detection, eye gaze estimation, and gestures from 55 participants in a user study, we analyzed how human behavior changed in response to different types of failures and varying explanation levels. Our goal is to assess whether human collaborators are ready to accept less detailed explanations without inducing confusion. We formulate a data-driven predictor to predict human confusion during robot failure explanations. We also propose and evaluate a mechanism, based on the predictor, to adapt the explanation level according to observed human behavior. The promising results from this evaluation indicate the potential of this research in adapting a robot's explanations for failures to enhance the collaborative experience. 

**Abstract (ZH)**: 本研究旨在解释人类行为，以预见当机器人在故障解释时可能引起的潜在用户困惑，并使机器人能够根据人类行为的反馈调整其解释以实现更为自然和高效的协作。通过一项用户研究，该研究包括55名参与者的眼部情感检测、凝视估计和手势数据，分析了人类行为在面对不同类型的故障和不同解释水平时的变化。我们的目标是评估人类合作者在不受困惑影响的情况下接受简化解释的意愿。我们构建了一种基于数据的预测器，以预测机器人故障解释过程中的人类困惑。此外，我们提出了并评估了一种机制，根据预测器和观察到的人类行为调整解释水平。这一评估的有利结果表明，此研究有可能通过适应机器人在故障时的解释来增强合作体验。 

---
# From Movement Primitives to Distance Fields to Dynamical Systems 

**Title (ZH)**: 从运动原型到距离场再到动力学系统 

**Authors**: Yiming Li, Sylvain Calinon  

**Link**: [PDF](https://arxiv.org/pdf/2504.09705)  

**Abstract**: Developing autonomous robots capable of learning and reproducing complex motions from demonstrations remains a fundamental challenge in robotics. On the one hand, movement primitives (MPs) provide a compact and modular representation of continuous trajectories. On the other hand, autonomous systems provide control policies that are time independent. We propose in this paper a simple and flexible approach that gathers the advantages of both representations by transforming MPs into autonomous systems. The key idea is to transform the explicit representation of a trajectory as an implicit shape encoded as a distance field. This conversion from a time-dependent motion to a spatial representation enables the definition of an autonomous dynamical system with modular reactions to perturbation. Asymptotic stability guarantees are provided by using Bernstein basis functions in the MPs, representing trajectories as concatenated quadratic Bézier curves, which provide an analytical method for computing distance fields. This approach bridges conventional MPs with distance fields, ensuring smooth and precise motion encoding, while maintaining a continuous spatial representation. By simply leveraging the analytic gradients of the curve and its distance field, a stable dynamical system can be computed to reproduce the demonstrated trajectories while handling perturbations, without requiring a model of the dynamical system to be estimated. Numerical simulations and real-world robotic experiments validate our method's ability to encode complex motion patterns while ensuring trajectory stability, together with the flexibility of designing the desired reaction to perturbations. An interactive project page demonstrating our approach is available at this https URL. 

**Abstract (ZH)**: 开发能够从演示中学习和再现复杂运动的自主机器人仍然是机器人学中的一个基本挑战。本文提出了一种简单灵活的方法，通过将运动基元（MPs）转换为自主系统，集成了两种表示形式的优势。关键思想是将轨迹的显式表示转换为由距离场编码的隐式形状。这种从时间依赖运动到空间表示的转换使定义具有模块化对扰动反应的自主动力学系统成为可能。通过使用伯恩斯坦基函数表示轨迹（作为连接的二次贝塞尔曲线），并提供距离场的解析计算方法，保证了渐近稳定性。该方法将传统的运动基元与距离场相结合，确保了平滑和精确的运动编码，同时保持连续的空间表示。仅通过利用曲线及其距离场的解析梯度，可以计算出稳定的动力学系统以再现演示的轨迹并处理扰动，而无需估计动力学系统的模型。数值仿真和现实世界的机器人实验验证了该方法在确保轨迹稳定性的同时能够编码复杂运动模式的能力，并且具有灵活性，可以设计期望的对扰动的反应。我们的方法示例可在以下网址查看：此 https URL。 

---
# A highly maneuverable flying squirrel drone with agility-improving foldable wings 

**Title (ZH)**: 一种配备增强机动性的折叠翼微型飞行灵 squirrel 无人机 

**Authors**: Dohyeon Lee, Jun-Gill Kang, Soohee Han  

**Link**: [PDF](https://arxiv.org/pdf/2504.09609)  

**Abstract**: Drones, like most airborne aerial vehicles, face inherent disadvantages in achieving agile flight due to their limited thrust capabilities. These physical constraints cannot be fully addressed through advancements in control algorithms alone. Drawing inspiration from the winged flying squirrel, this paper proposes a highly maneuverable drone equipped with agility-enhancing foldable wings. By leveraging collaborative control between the conventional propeller system and the foldable wings-coordinated through the Thrust-Wing Coordination Control (TWCC) framework-the controllable acceleration set is expanded, enabling the generation of abrupt vertical forces that are unachievable with traditional wingless drones. The complex aerodynamics of the foldable wings are modeled using a physics-assisted recurrent neural network (paRNN), which calibrates the angle of attack (AOA) to align with the real aerodynamic behavior of the wings. The additional air resistance generated by appropriately deploying these wings significantly improves the tracking performance of the proposed "flying squirrel" drone. The model is trained on real flight data and incorporates flat-plate aerodynamic principles. Experimental results demonstrate that the proposed flying squirrel drone achieves a 13.1% improvement in tracking performance, as measured by root mean square error (RMSE), compared to a conventional wingless drone. A demonstration video is available on YouTube: this https URL. 

**Abstract (ZH)**: 无人机，类似于大多数空中飞行器，由于推力限制，在实现敏捷飞行方面固有地处于不利地位。这些物理限制仅靠控制算法的进步无法完全解决。受有翼滑翔quirrel启发，本文提出了一种配备增稳可折叠翼的高机动性无人机。通过利用传统推进系统与可折叠翼之间的协作控制（协调通过推进翼协调控制[TWCC]框架实现），扩展可控加速度集合，从而能够生成传统无翼无人机无法实现的突然垂直力。利用物理辅助循环神经网络（paRNN）建模可折叠翼的复杂空气动力学特性，校正迎角（AOA），使其与翼的实际空气动力学行为相吻合。适当地展开这些翼所产生的额外空气阻力显著提高了所提“滑翔quirrel”无人机的跟踪性能。该模型使用实际飞行数据训练，并结合平板空气动力学原理。实验结果表明，与传统无翼无人机相比，所提滑翔quirrel无人机的跟踪性能通过均方根误差（RMSE）衡量提高了13.1%。完整演示视频可在YouTube上查看：this https URL。 

---
# GeoNav: Empowering MLLMs with Explicit Geospatial Reasoning Abilities for Language-Goal Aerial Navigation 

**Title (ZH)**: GeoNav: 为语言目标航空导航增强显式地理空间推理能力的MLLMs 

**Authors**: Haotian Xu, Yue Hu, Chen Gao, Zhengqiu Zhu, Yong Zhao, Yong Li, Quanjun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2504.09587)  

**Abstract**: Language-goal aerial navigation is a critical challenge in embodied AI, requiring UAVs to localize targets in complex environments such as urban blocks based on textual specification. Existing methods, often adapted from indoor navigation, struggle to scale due to limited field of view, semantic ambiguity among objects, and lack of structured spatial reasoning. In this work, we propose GeoNav, a geospatially aware multimodal agent to enable long-range navigation. GeoNav operates in three phases-landmark navigation, target search, and precise localization-mimicking human coarse-to-fine spatial strategies. To support such reasoning, it dynamically builds two different types of spatial memory. The first is a global but schematic cognitive map, which fuses prior textual geographic knowledge and embodied visual cues into a top-down, annotated form for fast navigation to the landmark region. The second is a local but delicate scene graph representing hierarchical spatial relationships between blocks, landmarks, and objects, which is used for definite target localization. On top of this structured representation, GeoNav employs a spatially aware, multimodal chain-of-thought prompting mechanism to enable multimodal large language models with efficient and interpretable decision-making across stages. On the CityNav urban navigation benchmark, GeoNav surpasses the current state-of-the-art by up to 12.53% in success rate and significantly improves navigation efficiency, even in hard-level tasks. Ablation studies highlight the importance of each module, showcasing how geospatial representations and coarse-to-fine reasoning enhance UAV navigation. 

**Abstract (ZH)**: 基于地理意识的多模态导航：复杂环境中的语言目标空中导航 

---
# AirVista-II: An Agentic System for Embodied UAVs Toward Dynamic Scene Semantic Understanding 

**Title (ZH)**: AirVista-II：面向动态场景语义理解的自主体无人机系统 

**Authors**: Fei Lin, Yonglin Tian, Tengchao Zhang, Jun Huang, Sangtian Guan, Fei-Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09583)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) are increasingly important in dynamic environments such as logistics transportation and disaster response. However, current tasks often rely on human operators to monitor aerial videos and make operational decisions. This mode of human-machine collaboration suffers from significant limitations in efficiency and adaptability. In this paper, we present AirVista-II -- an end-to-end agentic system for embodied UAVs, designed to enable general-purpose semantic understanding and reasoning in dynamic scenes. The system integrates agent-based task identification and scheduling, multimodal perception mechanisms, and differentiated keyframe extraction strategies tailored for various temporal scenarios, enabling the efficient capture of critical scene information. Experimental results demonstrate that the proposed system achieves high-quality semantic understanding across diverse UAV-based dynamic scenarios under a zero-shot setting. 

**Abstract (ZH)**: 无人自治系统AirVista-II：面向动态场景的通用语义理解与推理 

---
# Embodied Chain of Action Reasoning with Multi-Modal Foundation Model for Humanoid Loco-manipulation 

**Title (ZH)**: 基于多模态基础模型的 humanoid 执行操作推理结合身体约束 

**Authors**: Yu Hao, Geeta Chandra Raju Bethala, Niraj Pudasaini, Hao Huang, Shuaihang Yuan, Congcong Wen, Baoru Huang, Anh Nguyen, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09532)  

**Abstract**: Enabling humanoid robots to autonomously perform loco-manipulation tasks in complex, unstructured environments poses significant challenges. This entails equipping robots with the capability to plan actions over extended horizons while leveraging multi-modality to bridge gaps between high-level planning and actual task execution. Recent advancements in multi-modal foundation models have showcased substantial potential in enhancing planning and reasoning abilities, particularly in the comprehension and processing of semantic information for robotic control tasks. In this paper, we introduce a novel framework based on foundation models that applies the embodied chain of action reasoning methodology to autonomously plan actions from textual instructions for humanoid loco-manipulation. Our method integrates humanoid-specific chain of thought methodology, including detailed affordance and body movement analysis, which provides a breakdown of the task into a sequence of locomotion and manipulation actions. Moreover, we incorporate spatial reasoning based on the observation and target object properties to effectively navigate where target position may be unseen or occluded. Through rigorous experimental setups on object rearrangement, manipulations and loco-manipulation tasks on a real-world environment, we evaluate our method's efficacy on the decoupled upper and lower body control and demonstrate the effectiveness of the chain of robotic action reasoning strategies in comprehending human instructions. 

**Abstract (ZH)**: 自主在复杂非结构化环境中执行类人机器人动作操作任务面临重大挑战：通过本体行动推理方法基于基础模型自主规划文本指令下的类人体操机动操作任务 

---
# Towards Intuitive Drone Operation Using a Handheld Motion Controller 

**Title (ZH)**: 基于手持运动控制器的直观无人机操作研究 

**Authors**: Daria Trinitatova, Sofia Shevelo, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2504.09510)  

**Abstract**: We present an intuitive human-drone interaction system that utilizes a gesture-based motion controller to enhance the drone operation experience in real and simulated environments. The handheld motion controller enables natural control of the drone through the movements of the operator's hand, thumb, and index finger: the trigger press manages the throttle, the tilt of the hand adjusts pitch and roll, and the thumbstick controls yaw rotation. Communication with drones is facilitated via the ExpressLRS radio protocol, ensuring robust connectivity across various frequencies. The user evaluation of the flight experience with the designed drone controller using the UEQ-S survey showed high scores for both Pragmatic (mean=2.2, SD = 0.8) and Hedonic (mean=2.3, SD = 0.9) Qualities. This versatile control interface supports applications such as research, drone racing, and training programs in real and simulated environments, thereby contributing to advances in the field of human-drone interaction. 

**Abstract (ZH)**: 一种基于手势的直观人机操控系统：增强无人机操作体验的研究 

---
# Debiasing 6-DOF IMU via Hierarchical Learning of Continuous Bias Dynamics 

**Title (ZH)**: 基于层次学习连续偏置动态的6-DOF IMU去偏差化 

**Authors**: Ben Liu, Tzu-Yuan Lin, Wei Zhang, Maani Ghaffari  

**Link**: [PDF](https://arxiv.org/pdf/2504.09495)  

**Abstract**: This paper develops a deep learning approach to the online debiasing of IMU gyroscopes and accelerometers. Most existing methods rely on implicitly learning a bias term to compensate for raw IMU data. Explicit bias learning has recently shown its potential as a more interpretable and motion-independent alternative. However, it remains underexplored and faces challenges, particularly the need for ground truth bias data, which is rarely available. To address this, we propose a neural ordinary differential equation (NODE) framework that explicitly models continuous bias dynamics, requiring only pose ground truth, often available in datasets. This is achieved by extending the canonical NODE framework to the matrix Lie group for IMU kinematics with a hierarchical training strategy. The validation on two public datasets and one real-world experiment demonstrates significant accuracy improvements in IMU measurements, reducing errors in both pure IMU integration and visual-inertial odometry. 

**Abstract (ZH)**: 基于神经常微分方程的IMU陀螺仪和加速度计在线去偏见的深度学习方法 

---
# A highly maneuverable flying squirrel drone with controllable foldable wings 

**Title (ZH)**: 可操控折叠翅膀的高机动飞行豚鼠无人机 

**Authors**: Jun-Gill Kang, Dohyeon Lee, Soohee Han  

**Link**: [PDF](https://arxiv.org/pdf/2504.09478)  

**Abstract**: Typical drones with multi rotors are generally less maneuverable due to unidirectional thrust, which may be unfavorable to agile flight in very narrow and confined spaces. This paper suggests a new bio-inspired drone that is empowered with high maneuverability in a lightweight and easy-to-carry way. The proposed flying squirrel inspired drone has controllable foldable wings to cover a wider range of flight attitudes and provide more maneuverable flight capability with stable tracking performance. The wings of a drone are fabricated with silicone membranes and sophisticatedly controlled by reinforcement learning based on human-demonstrated data. Specially, such learning based wing control serves to capture even the complex aerodynamics that are often impossible to model mathematically. It is shown through experiment that the proposed flying squirrel drone intentionally induces aerodynamic drag and hence provides the desired additional repulsive force even under saturated mechanical thrust. This work is very meaningful in demonstrating the potential of biomimicry and machine learning for realizing an animal-like agile drone. 

**Abstract (ZH)**: 仿鼯鼠灵感的高机动性轻量化无人机及其基于强化学习的可控折叠翼设计 

---
# ADDT -- A Digital Twin Framework for Proactive Safety Validation in Autonomous Driving Systems 

**Title (ZH)**: ADDT -- 一种用于自主驾驶系统前瞻性安全验证的数字孪生框架 

**Authors**: Bo Yu, Chaoran Yuan, Zishen Wan, Jie Tang, Fadi Kurdahi, Shaoshan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.09461)  

**Abstract**: Autonomous driving systems continue to face safety-critical failures, often triggered by rare and unpredictable corner cases that evade conventional testing. We present the Autonomous Driving Digital Twin (ADDT) framework, a high-fidelity simulation platform designed to proactively identify hidden faults, evaluate real-time performance, and validate safety before deployment. ADDT combines realistic digital models of driving environments, vehicle dynamics, sensor behavior, and fault conditions to enable scalable, scenario-rich stress-testing under diverse and adverse conditions. It supports adaptive exploration of edge cases using reinforcement-driven techniques, uncovering failure modes that physical road testing often misses. By shifting from reactive debugging to proactive simulation-driven validation, ADDT enables a more rigorous and transparent approach to autonomous vehicle safety engineering. To accelerate adoption and facilitate industry-wide safety improvements, the entire ADDT framework has been released as open-source software, providing developers with an accessible and extensible tool for comprehensive safety testing at scale. 

**Abstract (ZH)**: 自主驾驶系统继续面临安全关键性故障，这些故障通常由传统测试难以覆盖的罕见且不可预测的边缘情况触发。我们提出了自主驾驶数字双胞胎（ADDT）框架，这是一个高保真度的仿真平台，旨在主动识别隐藏故障、评估实时性能并在部署前验证安全性。ADDT 结合了驾驶环境、车辆动力学、传感器行为和故障条件的现实数字模型，以实现多样化且恶劣条件下的可扩展、场景丰富的压力测试。它支持使用强化学习驱动技术进行边缘案例的自适应探索，揭示物理道路测试往往难以发现的故障模式。通过从被动调试转向主动的仿真驱动验证，ADDT 使得自主车辆安全工程更加严谨和透明。为了加速采用并推动全行业的安全改进，整个ADDT框架已被开源发布，提供了开发人员进行全面安全测试的可访问且可扩展的工具。 

---
# DoorBot: Closed-Loop Task Planning and Manipulation for Door Opening in the Wild with Haptic Feedback 

**Title (ZH)**: DoorBot: 在野外观门操作的闭环任务规划与执行及触觉反馈 

**Authors**: Zhi Wang, Yuchen Mo, Shengmiao Jin, Wenzhen Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2504.09358)  

**Abstract**: Robots operating in unstructured environments face significant challenges when interacting with everyday objects like doors. They particularly struggle to generalize across diverse door types and conditions. Existing vision-based and open-loop planning methods often lack the robustness to handle varying door designs, mechanisms, and push/pull configurations. In this work, we propose a haptic-aware closed-loop hierarchical control framework that enables robots to explore and open different unseen doors in the wild. Our approach leverages real-time haptic feedback, allowing the robot to adjust its strategy dynamically based on force feedback during manipulation. We test our system on 20 unseen doors across different buildings, featuring diverse appearances and mechanical types. Our framework achieves a 90% success rate, demonstrating its ability to generalize and robustly handle varied door-opening tasks. This scalable solution offers potential applications in broader open-world articulated object manipulation tasks. 

**Abstract (ZH)**: 具有触觉意识的闭环层次控制框架：使机器人在不同环境下探索和开启未见门的比例高效解决方案 

---
# Adaptive Planning Framework for UAV-Based Surface Inspection in Partially Unknown Indoor Environments 

**Title (ZH)**: 基于部分未知室内环境的无人机表面检测自适应规划框架 

**Authors**: Hanyu Jin, Zhefan Xu, Haoyu Shen, Xinming Han, Kanlong Ye, Kenji Shimada  

**Link**: [PDF](https://arxiv.org/pdf/2504.09294)  

**Abstract**: Inspecting indoor environments such as tunnels, industrial facilities, and construction sites is essential for infrastructure monitoring and maintenance. While manual inspection in these environments is often time-consuming and potentially hazardous, Unmanned Aerial Vehicles (UAVs) can improve efficiency by autonomously handling inspection tasks. Such inspection tasks usually rely on reference maps for coverage planning. However, in industrial applications, only the floor plans are typically available. The unforeseen obstacles not included in the floor plans will result in outdated reference maps and inefficient or unsafe inspection trajectories. In this work, we propose an adaptive inspection framework that integrates global coverage planning with local reactive adaptation to improve the coverage and efficiency of UAV-based inspection in partially unknown indoor environments. Experimental results in structured indoor scenarios demonstrate the effectiveness of the proposed approach in inspection efficiency and achieving high coverage rates with adaptive obstacle handling, highlighting its potential for enhancing the efficiency of indoor facility inspection. 

**Abstract (ZH)**: 基于无人机的半未知室内环境适应性检测框架：结合全局覆盖规划与局部反应性适应提高检测效率和覆盖率 

---
# REALM: Real-Time Estimates of Assistance for Learned Models in Human-Robot Interaction 

**Title (ZH)**: REALM: 人类与机器人交互中学习模型的实时辅助估计 

**Authors**: Michael Hagenow, Julie A. Shah  

**Link**: [PDF](https://arxiv.org/pdf/2504.09243)  

**Abstract**: There are a variety of mechanisms (i.e., input types) for real-time human interaction that can facilitate effective human-robot teaming. For example, previous works have shown how teleoperation, corrective, and discrete (i.e., preference over a small number of choices) input can enable robots to complete complex tasks. However, few previous works have looked at combining different methods, and in particular, opportunities for a robot to estimate and elicit the most effective form of assistance given its understanding of a task. In this paper, we propose a method for estimating the value of different human assistance mechanisms based on the action uncertainty of a robot policy. Our key idea is to construct mathematical expressions for the expected post-interaction differential entropy (i.e., uncertainty) of a stochastic robot policy to compare the expected value of different interactions. As each type of human input imposes a different requirement for human involvement, we demonstrate how differential entropy estimates can be combined with a likelihood penalization approach to effectively balance feedback informational needs with the level of required input. We demonstrate evidence of how our approach interfaces with emergent learning models (e.g., a diffusion model) to produce accurate assistance value estimates through both simulation and a robot user study. Our user study results indicate that the proposed approach can enable task completion with minimal human feedback for uncertain robot behaviors. 

**Abstract (ZH)**: 实时人类交互机制在促进人机团队协作中的多样性及其价值估计：一种基于机器人策略动作不确定性的方法 

---
# Development of a PPO-Reinforcement Learned Walking Tripedal Soft-Legged Robot using SOFA 

**Title (ZH)**: 基于SOFA的PPO强化学习三足软腿行走机器人开发 

**Authors**: Yomna Mokhtar, Tarek Shohdy, Abdallah A. Hassan, Mostafa Eshra, Omar Elmenawy, Osama Khalil, Haitham El-Hussieny  

**Link**: [PDF](https://arxiv.org/pdf/2504.09242)  

**Abstract**: Rigid robots were extensively researched, whereas soft robotics remains an underexplored field. Utilizing soft-legged robots in performing tasks as a replacement for human beings is an important stride to take, especially under harsh and hazardous conditions over rough terrain environments. For the demand to teach any robot how to behave in different scenarios, a real-time physical and visual simulation is essential. When it comes to soft robots specifically, a simulation framework is still an arduous problem that needs to be disclosed. Using the simulation open framework architecture (SOFA) is an advantageous step. However, neither SOFA's manual nor prior public SOFA projects show its maximum capabilities the users can reach. So, we resolved this by establishing customized settings and handling the framework components appropriately. Settling on perfect, fine-tuned SOFA parameters has stimulated our motivation towards implementing the state-of-the-art (SOTA) reinforcement learning (RL) method of proximal policy optimization (PPO). The final representation is a well-defined, ready-to-deploy walking, tripedal, soft-legged robot based on PPO-RL in a SOFA environment. Robot navigation performance is a key metric to be considered for measuring the success resolution. Although in the simulated soft robots case, an 82\% success rate in reaching a single goal is a groundbreaking output, we pushed the boundaries to further steps by evaluating the progress under assigning a sequence of goals. While trailing the platform steps, outperforming discovery has been observed with an accumulative squared error deviation of 19 mm. The full code is publicly available at \href{this https URL}{this http URL\textunderscore$SOFA$\textunderscore$Soft$\textunderscore$Legged$\textunderscore$ this http URL} 

**Abstract (ZH)**: 柔体机器人受到了广泛研究，而软体机器人领域仍然未被充分探索。利用具备软腿的机器人执行任务，替代人类尤为重要，尤其是在恶劣和危险的崎岖地形环境中。为了满足任何机器人在不同场景下学习如何行为的需求，实时物理和视觉模拟是必不可少的。对于软体机器人而言，尚未有现成的模拟框架满足需求，因此使用模拟开放框架架构（SOFA）是一个有益的步骤。然而，SOFA的手册及先前公开的SOFA项目均未能充分展示其全部功能。因此，我们通过对框架进行定制化设置并适当处理框架组件，解决了这一问题。优化SOFA参数激发了我们利用最先进的强化学习方法（PPO）进行实施的动力。最终成果是在SOFA环境中基于PPO-RL实现了一款定义明确且可部署的三足软腿行走机器人。机器人的导航性能是衡量成功的关键指标。尽管在模拟软体机器人的情况下，达到单个目标的成功率为82%已是突破性成果，我们进一步通过评估一系列目标来提升其性能，在跟踪平台步骤时，累计平方误差偏差达到了19毫米。完整代码已公开，可通过[this http URL](this http URL_SOFA_SoftLegged)访问。 

---
# Concurrent-Allocation Task Execution for Multi-Robot Path-Crossing-Minimal Navigation in Obstacle Environments 

**Title (ZH)**: 多机器人在障碍环境中的路径交叉最小化并发分配任务执行导航 

**Authors**: Bin-Bin Hu, Weijia Yao, Yanxin Zhou, Henglai Wei, Chen Lv  

**Link**: [PDF](https://arxiv.org/pdf/2504.09230)  

**Abstract**: Reducing undesirable path crossings among trajectories of different robots is vital in multi-robot navigation missions, which not only reduces detours and conflict scenarios, but also enhances navigation efficiency and boosts productivity. Despite recent progress in multi-robot path-crossing-minimal (MPCM) navigation, the majority of approaches depend on the minimal squared-distance reassignment of suitable desired points to robots directly. However, if obstacles occupy the passing space, calculating the actual robot-point distances becomes complex or intractable, which may render the MPCM navigation in obstacle environments inefficient or even infeasible.
In this paper, the concurrent-allocation task execution (CATE) algorithm is presented to address this problem (i.e., MPCM navigation in obstacle environments). First, the path-crossing-related elements in terms of (i) robot allocation, (ii) desired-point convergence, and (iii) collision and obstacle avoidance are encoded into integer and control barrier function (CBF) constraints. Then, the proposed constraints are used in an online constrained optimization framework, which implicitly yet effectively minimizes the possible path crossings and trajectory length in obstacle environments by minimizing the desired point allocation cost and slack variables in CBF constraints simultaneously. In this way, the MPCM navigation in obstacle environments can be achieved with flexible spatial orderings. Note that the feasibility of solutions and the asymptotic convergence property of the proposed CATE algorithm in obstacle environments are both guaranteed, and the calculation burden is also reduced by concurrently calculating the optimal allocation and the control input directly without the path planning process. 

**Abstract (ZH)**: 在障碍环境下的多机器人路径交叉最小化导航 

---
# Compliant Explicit Reference Governor for Contact Friendly Robotic Manipulators 

**Title (ZH)**: 友接触柔顺显式引用控制器 

**Authors**: Yaashia Gautam, Nataliya Nechyporenko, Chi-Hui Lin, Alessandro Roncone, Marco M. Nicotra  

**Link**: [PDF](https://arxiv.org/pdf/2504.09188)  

**Abstract**: This paper introduces the Compliant Explicit Reference Governor (C-ERG), an extension of the Explicit Reference Governor that allows the robot to operate safely while in contact with the environment.
The C-ERG is an intermediate layer that can be placed between a high-level planner and a low-level controller: its role is to enforce operational constraints and to enable the smooth transition between free-motion and contact operations. The C-ERG ensures safety by limiting the total energy available to the robotic arm at the time of contact. In the absence of contact, however, the C-ERG does not penalize the system performance.
Numerical examples showcase the behavior of the C-ERG for increasingly complex systems. 

**Abstract (ZH)**: 这种论文介绍了一种扩展的明确参考 governor（C-ERG），即可穿戴式明确参考 governor，允许机器人在与环境接触时安全运行。 

---
# Steady-State Drifting Equilibrium Analysis of Single-Track Two-Wheeled Robots for Controller Design 

**Title (ZH)**: 单轨道两轮机器人稳态漂移平衡分析及其控制器设计 

**Authors**: Feilong Jing, Yang Deng, Boyi Wang, Xudong Zheng, Yifan Sun, Zhang Chen, Bin Liang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09134)  

**Abstract**: Drifting is an advanced driving technique where the wheeled robot's tire-ground interaction breaks the common non-holonomic pure rolling constraint. This allows high-maneuverability tasks like quick cornering, and steady-state drifting control enhances motion stability under lateral slip conditions. While drifting has been successfully achieved in four-wheeled robot systems, its application to single-track two-wheeled (STTW) robots, such as unmanned motorcycles or bicycles, has not been thoroughly studied. To bridge this gap, this paper extends the drifting equilibrium theory to STTW robots and reveals the mechanism behind the steady-state drifting maneuver. Notably, the counter-steering drifting technique used by skilled motorcyclists is explained through this theory. In addition, an analytical algorithm based on intrinsic geometry and kinematics relationships is proposed, reducing the computation time by four orders of magnitude while maintaining less than 6% error compared to numerical methods. Based on equilibrium analysis, a model predictive controller (MPC) is designed to achieve steady-state drifting and equilibrium points transition, with its effectiveness and robustness validated through simulations. 

**Abstract (ZH)**: 滚动是先进的驾驶技术，其中轮式机器人轮胎与地面的相互作用打破了常见的非全动纯滚动约束。这使得快速过弯等高机动任务成为可能，而稳态漂移控制则在侧滑条件下增强了运动稳定性。虽然滚动技术已在四轮机器人系统中成功实现，但对于单轨两轮（STTW）机器人，如无人驾驶摩托车或自行车，其应用尚未进行充分研究。为了填补这一空白，本文将滚动平衡理论扩展到STTW机器人，并揭示了稳态漂移机动背后的机制。此外，本文提出了一种基于固有几何学和运动学关系的解析算法，相比数值方法，该算法将计算时间缩短了四个数量级，同时保留了不到6%的误差。基于平衡分析，设计了一个模型预测控制器（MPC）来实现稳态漂移和平衡点切换，并通过仿真验证了其有效性和鲁棒性。 

---
# Haptic Perception via the Dynamics of Flexible Body Inspired by an Ostrich's Neck 

**Title (ZH)**: 基于鸵鸟颈部动力学的柔性体触觉感知 

**Authors**: Kazashi Nakano, Katsuma Inoue, Yasuo Kuniyoshi, Kohei Nakajima  

**Link**: [PDF](https://arxiv.org/pdf/2504.09131)  

**Abstract**: In biological systems, haptic perception is achieved through both flexible skin and flexible body. In fully soft robots, the fragility of their bodies and the time delays in sensory processing pose significant challenges. The musculoskeletal system possesses both the deformability inherent in soft materials and the durability of rigid-body robots. Additionally, by outsourcing part of the intelligent information processing to the morphology of the musculoskeletal system, applications for dynamic tasks are expected. This study focuses on the pecking movements of birds, which achieve precise haptic perception through the musculoskeletal system of their flexible neck. Physical reservoir computing is applied to flexible structures inspired by an ostrich neck to analyze the relationship between haptic perception and physical characteristics. Combined experiments using both an actual robot and simulations demonstrate that, under appropriate body viscoelasticity, the flexible structure can distinguish objects of varying softness and memorize this information as behaviors. Drawing on these findings and anatomical insights from the ostrich neck, a haptic sensing system is proposed that possesses separability and this behavioral memory in flexible structures, enabling rapid learning and real-time inference. The results demonstrate that through the dynamics of flexible structures, diverse functions can emerge beyond their original design as manipulators. 

**Abstract (ZH)**: 在生物系统中，触觉感知通过灵活的皮肤和身体实现。在全软机器人中，其脆弱的身体和感觉处理中的时间延迟带来了重大挑战。肌骨骼系统兼具软材料的柔韧性和刚性机器人耐用性的特点。此外，通过将部分智能信息处理外包给肌骨骼系统的形态，期望应用于动态任务。本研究聚焦于鸟类的啄食动作，通过其灵活颈部的肌骨骼系统实现精确的触觉感知。物理流形计算应用于受鸵鸟颈部启发的柔性结构，以分析触觉感知与物理特性之间的关系。结合使用实际机器人和模拟的实验表明，在适当的体部粘弹性条件下，柔性结构能够区分不同柔软度的物体并将其信息作为行为进行记忆。借鉴这些发现和鸵鸟颈部的解剖学见解，提出了一种具备柔性结构分离性和行为记忆的触觉感知系统，使其能够实现快速学习和实时推理。研究结果表明，通过柔性结构的动力学，可以超越其原始设计功能发挥多种功能。 

---
# IMPACT: Behavioral Intention-aware Multimodal Trajectory Prediction with Adaptive Context Trimming 

**Title (ZH)**: IMPACT: 基于行为意图的自适应背景裁剪多模态轨迹预测 

**Authors**: Jiawei Sun, Xibin Yue, Jiahui Li, Tianle Shen, Chengran Yuan, Shuo Sun, Sheng Guo, Quanyun Zhou, Marcelo H Ang Jr  

**Link**: [PDF](https://arxiv.org/pdf/2504.09103)  

**Abstract**: While most prior research has focused on improving the precision of multimodal trajectory predictions, the explicit modeling of multimodal behavioral intentions (e.g., yielding, overtaking) remains relatively underexplored. This paper proposes a unified framework that jointly predicts both behavioral intentions and trajectories to enhance prediction accuracy, interpretability, and efficiency. Specifically, we employ a shared context encoder for both intention and trajectory predictions, thereby reducing structural redundancy and information loss. Moreover, we address the lack of ground-truth behavioral intention labels in mainstream datasets (Waymo, Argoverse) by auto-labeling these datasets, thus advancing the community's efforts in this direction. We further introduce a vectorized occupancy prediction module that infers the probability of each map polyline being occupied by the target vehicle's future trajectory. By leveraging these intention and occupancy prediction priors, our method conducts dynamic, modality-dependent pruning of irrelevant agents and map polylines in the decoding stage, effectively reducing computational overhead and mitigating noise from non-critical elements. Our approach ranks first among LiDAR-free methods on the Waymo Motion Dataset and achieves first place on the Waymo Interactive Prediction Dataset. Remarkably, even without model ensembling, our single-model framework improves the soft mean average precision (softmAP) by 10 percent compared to the second-best method in the Waymo Interactive Prediction Leaderboard. Furthermore, the proposed framework has been successfully deployed on real vehicles, demonstrating its practical effectiveness in real-world applications. 

**Abstract (ZH)**: 一种统一框架：联合预测行为意图和轨迹以提高多模态轨迹预测的准确性、可解释性和效率 

---
# agriFrame: Agricultural framework to remotely control a rover inside a greenhouse environment 

**Title (ZH)**: 农业框架：用于控制温室环境内 rover 的远程框架 

**Authors**: Saail Narvekar, Soofiyan Atar, Vishal Gupta, Lohit Penubaku, Kavi Arya  

**Link**: [PDF](https://arxiv.org/pdf/2504.09079)  

**Abstract**: The growing demand for innovation in agriculture is essential for food security worldwide and more implicit in developing countries. With growing demand comes a reduction in rapid development time. Data collection and analysis are essential in agriculture. However, considering a given crop, its cycle comes once a year, and researchers must wait a few months before collecting more data for the given crop. To overcome this hurdle, researchers are venturing into digital twins for agriculture. Toward this effort, we present an agricultural framework(agriFrame). Here, we introduce a simulated greenhouse environment for testing and controlling a robot and remotely controlling/implementing the algorithms in the real-world greenhouse setup. This work showcases the importance/interdependence of network setup, remotely controllable rover, and messaging protocol. The sophisticated yet simple-to-use agriFrame has been optimized for the simulator on minimal laptop/desktop specifications. 

**Abstract (ZH)**: 随着农业创新需求的增长，保障全球粮食安全变得至关重要，尤其对于发展中国家而言。随之而来的快速开发时间减少要求数据的收集与分析成为关键。然而，对于特定作物，其生长周期每年仅一次，研究人员需要等待几个月才能收集更多数据。为克服这一挑战，研究者们正致力于农业数字孪生技术。为此，我们提出一种农业框架(agriFrame)。该框架介绍了一个模拟温室环境，用于测试和控制机器人，并远程控制/实施实温室环境中的算法。本工作展示了网络配置、远程可控漫游车和消息协议的重要性/相互依赖性。agriFrame既先进又易于使用，已经针对最少的笔记本电脑/桌面规格进行了优化。 

---
# Multi-Robot Coordination with Adversarial Perception 

**Title (ZH)**: 多方机器人对抗感知下的协同控制 

**Authors**: Rayan Bahrami, Hamidreza Jafarnejadsani  

**Link**: [PDF](https://arxiv.org/pdf/2504.09047)  

**Abstract**: This paper investigates the resilience of perception-based multi-robot coordination with wireless communication to online adversarial perception. A systematic study of this problem is essential for many safety-critical robotic applications that rely on the measurements from learned perception modules. We consider a (small) team of quadrotor robots that rely only on an Inertial Measurement Unit (IMU) and the visual data measurements obtained from a learned multi-task perception module (e.g., object detection) for downstream tasks, including relative localization and coordination. We focus on a class of adversarial perception attacks that cause misclassification, mislocalization, and latency. We propose that the effects of adversarial misclassification and mislocalization can be modeled as sporadic (intermittent) and spurious measurement data for the downstream tasks. To address this, we present a framework for resilience analysis of multi-robot coordination with adversarial measurements. The framework integrates data from Visual-Inertial Odometry (VIO) and the learned perception model for robust relative localization and state estimation in the presence of adversarially sporadic and spurious measurements. The framework allows for quantifying the degradation in system observability and stability in relation to the success rate of adversarial perception. Finally, experimental results on a multi-robot platform demonstrate the real-world applicability of our methodology for resource-constrained robotic platforms. 

**Abstract (ZH)**: 基于感知的多机器人协调在无线通信下的抗在线敌对感知攻击鲁棒性研究 

---
# Nonconvex Obstacle Avoidance using Efficient Sampling-Based Distance Functions 

**Title (ZH)**: 非凸障碍避免的高效采样基距函数方法 

**Authors**: Paul Lutkus, Michelle S. Chong, Lars Lindemann  

**Link**: [PDF](https://arxiv.org/pdf/2504.09038)  

**Abstract**: We consider nonconvex obstacle avoidance where a robot described by nonlinear dynamics and a nonconvex shape has to avoid nonconvex obstacles. Obstacle avoidance is a fundamental problem in robotics and well studied in control. However, existing solutions are computationally expensive (e.g., model predictive controllers), neglect nonlinear dynamics (e.g., graph-based planners), use diffeomorphic transformations into convex domains (e.g., for star shapes), or are conservative due to convex overapproximations. The key challenge here is that the computation of the distance between the shapes of the robot and the obstacles is a nonconvex problem. We propose efficient computation of this distance via sampling-based distance functions. We quantify the sampling error and show that, for certain systems, such sampling-based distance functions are valid nonsmooth control barrier functions. We also study how to deal with disturbances on the robot dynamics in our setting. Finally, we illustrate our method on a robot navigation task involving an omnidirectional robot and nonconvex obstacles. We also analyze performance and computational efficiency of our controller as a function of the number of samples. 

**Abstract (ZH)**: 非凸障碍避让中的非线性动力学机器人问题研究 

---
# CL-CoTNav: Closed-Loop Hierarchical Chain-of-Thought for Zero-Shot Object-Goal Navigation with Vision-Language Models 

**Title (ZH)**: CL-CoTNav: 闭合环路层次链式思考在视觉语言模型支持下的零样本物体目标导航 

**Authors**: Yuxin Cai, Xiangkun He, Maonan Wang, Hongliang Guo, Wei-Yun Yau, Chen Lv  

**Link**: [PDF](https://arxiv.org/pdf/2504.09000)  

**Abstract**: Visual Object Goal Navigation (ObjectNav) requires a robot to locate a target object in an unseen environment using egocentric observations. However, decision-making policies often struggle to transfer to unseen environments and novel target objects, which is the core generalization problem. Traditional end-to-end learning methods exacerbate this issue, as they rely on memorizing spatial patterns rather than employing structured reasoning, limiting their ability to generalize effectively. In this letter, we introduce Closed-Loop Hierarchical Chain-of-Thought Navigation (CL-CoTNav), a vision-language model (VLM)-driven ObjectNav framework that integrates structured reasoning and closed-loop feedback into navigation decision-making. To enhance generalization, we fine-tune a VLM using multi-turn question-answering (QA) data derived from human demonstration trajectories. This structured dataset enables hierarchical Chain-of-Thought (H-CoT) prompting, systematically extracting compositional knowledge to refine perception and decision-making, inspired by the human cognitive process of locating a target object through iterative reasoning steps. Additionally, we propose a Closed-Loop H-CoT mechanism that incorporates detection and reasoning confidence scores into training. This adaptive weighting strategy guides the model to prioritize high-confidence data pairs, mitigating the impact of noisy inputs and enhancing robustness against hallucinated or incorrect reasoning. Extensive experiments in the AI Habitat environment demonstrate CL-CoTNav's superior generalization to unseen scenes and novel object categories. Our method consistently outperforms state-of-the-art approaches in navigation success rate (SR) and success weighted by path length (SPL) by 22.4\%. We release our datasets, models, and supplementary videos on our project page. 

**Abstract (ZH)**: 基于闭合环层次链式思维的视觉语言目标导航（CL-CoTNav） 

---
# Anti-Slip AI-Driven Model-Free Control with Global Exponential Stability in Skid-Steering Robots 

**Title (ZH)**: 基于滑移 steering 机器人全局指数稳定性的AI驱动无模型防滑控制 

**Authors**: Mehdi Heydari Shahna, Pauli Mustalahti, Jouni Mattila  

**Link**: [PDF](https://arxiv.org/pdf/2504.08831)  

**Abstract**: Undesired lateral and longitudinal wheel slippage can disrupt a mobile robot's heading angle, traction, and, eventually, desired motion. This issue makes the robotization and accurate modeling of heavy-duty machinery very challenging because the application primarily involves off-road terrains, which are susceptible to uneven motion and severe slippage. As a step toward robotization in skid-steering heavy-duty robot (SSHDR), this paper aims to design an innovative robust model-free control system developed by neural networks to strongly stabilize the robot dynamics in the presence of a broad range of potential wheel slippages. Before the control design, the dynamics of the SSHDR are first investigated by mathematically incorporating slippage effects, assuming that all functional modeling terms of the system are unknown to the control system. Then, a novel tracking control framework to guarantee global exponential stability of the SSHDR is designed as follows: 1) the unknown modeling of wheel dynamics is approximated using radial basis function neural networks (RBFNNs); and 2) a new adaptive law is proposed to compensate for slippage effects and tune the weights of the RBFNNs online during execution. Simulation and experimental results verify the proposed tracking control performance of a 4,836 kg SSHDR operating on slippery terrain. 

**Abstract (ZH)**: 未期望的横向和纵向车轮打滑会破坏移动机器人航向角、附着性能，并最终影响其期望运动。这一问题使得重型机械的机器人化及其精确建模极具挑战性，因为应用主要涉及易发生不规则运动和严重打滑的非铺装地形。为向滑移转向重型机器人（SSHDR）的机器人化方向迈出一步，本文旨在设计一种基于神经网络的创新鲁棒无模型控制系统，以在广泛范围的潜在车轮打滑条件下，强烈稳定机器人动力学。在控制设计之前，首先通过数学上包含打滑效应的方法研究SSHDR的动力学，并假定控制系统对系统的所有功能建模项均未知。然后提出了一种新型跟踪控制框架，以确保SSHDR的全局指数稳定：1) 使用径向基函数神经网络（RBFNNs）近似未知的车轮动力学模型；2) 提出了一种新的自适应律，在执行过程中补偿打滑效应并在线调整RBFNNs的权重。仿真和实验结果验证了该跟踪控制方法在滑溜地面上运行4836 kg SSHDR的性能。 

---
# MonoDiff9D: Monocular Category-Level 9D Object Pose Estimation via Diffusion Model 

**Title (ZH)**: MonoDiff9D：基于扩散模型的单目类别级9D物体姿态估计 

**Authors**: Jian Liu, Wei Sun, Hui Yang, Jin Zheng, Zichen Geng, Hossein Rahmani, Ajmal Mian  

**Link**: [PDF](https://arxiv.org/pdf/2504.10433)  

**Abstract**: Object pose estimation is a core means for robots to understand and interact with their environment. For this task, monocular category-level methods are attractive as they require only a single RGB camera. However, current methods rely on shape priors or CAD models of the intra-class known objects. We propose a diffusion-based monocular category-level 9D object pose generation method, MonoDiff9D. Our motivation is to leverage the probabilistic nature of diffusion models to alleviate the need for shape priors, CAD models, or depth sensors for intra-class unknown object pose estimation. We first estimate coarse depth via DINOv2 from the monocular image in a zero-shot manner and convert it into a point cloud. We then fuse the global features of the point cloud with the input image and use the fused features along with the encoded time step to condition MonoDiff9D. Finally, we design a transformer-based denoiser to recover the object pose from Gaussian noise. Extensive experiments on two popular benchmark datasets show that MonoDiff9D achieves state-of-the-art monocular category-level 9D object pose estimation accuracy without the need for shape priors or CAD models at any stage. Our code will be made public at this https URL. 

**Abstract (ZH)**: 基于扩散模型的单目类别级9D物体姿态生成方法：MonoDiff9D 

---
# ST-Booster: An Iterative SpatioTemporal Perception Booster for Vision-and-Language Navigation in Continuous Environments 

**Title (ZH)**: ST-增强器：连续环境中基于时空感知的迭代视觉-语言导航增强方法 

**Authors**: Lu Yue, Dongliang Zhou, Liang Xie, Erwei Yin, Feitian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09843)  

**Abstract**: Vision-and-Language Navigation in Continuous Environments (VLN-CE) requires agents to navigate unknown, continuous spaces based on natural language instructions. Compared to discrete settings, VLN-CE poses two core perception challenges. First, the absence of predefined observation points leads to heterogeneous visual memories and weakened global spatial correlations. Second, cumulative reconstruction errors in three-dimensional scenes introduce structural noise, impairing local feature perception. To address these challenges, this paper proposes ST-Booster, an iterative spatiotemporal booster that enhances navigation performance through multi-granularity perception and instruction-aware reasoning. ST-Booster consists of three key modules -- Hierarchical SpatioTemporal Encoding (HSTE), Multi-Granularity Aligned Fusion (MGAF), and ValueGuided Waypoint Generation (VGWG). HSTE encodes long-term global memory using topological graphs and captures shortterm local details via grid maps. MGAF aligns these dualmap representations with instructions through geometry-aware knowledge fusion. The resulting representations are iteratively refined through pretraining tasks. During reasoning, VGWG generates Guided Attention Heatmaps (GAHs) to explicitly model environment-instruction relevance and optimize waypoint selection. Extensive comparative experiments and performance analyses are conducted, demonstrating that ST-Booster outperforms existing state-of-the-art methods, particularly in complex, disturbance-prone environments. 

**Abstract (ZH)**: 连续环境中的视觉语言导航（VLN-CE）要求代理基于自然语言指令在未知的连续空间中进行导航。与离散环境相比，VLN-CE 提出了两个核心感知挑战。首先，缺乏预定义的观测点导致视觉记忆异质化和全局空间相关性减弱。其次，在三维场景中的累积重构误差引入了结构噪声，损害了局部特征感知能力。为应对这些挑战，本文提出了一种迭代时空增强器 ST-Booster，通过多层次感知和指令感知推理来提升导航性能。ST-Booster 包含三个关键模块——层次时空编码（HSTE）、多粒度对齐融合（MGAF）和价值引导的航点生成（VGWG）。HSTE 使用拓扑图来编码长期的全局记忆，并通过格网地图捕捉短期的局部细节。MGAF 通过几何感知知识融合将这两种地图表示与指令对齐，结果表示通过预训练任务进行迭代细化。在推理期间，VGWG 生成引导注意力热点图（GAHs）以明确建模环境-指令相关性并优化航点选择。通过广泛的对比实验和性能分析，证明了 ST-Booster 在复杂且干扰性强的环境中的表现优于现有最先进的方法。 

---
# Score Matching Diffusion Based Feedback Control and Planning of Nonlinear Systems 

**Title (ZH)**: 基于评分匹配扩散的非线性系统反馈控制与规划 

**Authors**: Karthik Elamvazhuthi, Darshan Gadginmath, Fabio Pasqualetti  

**Link**: [PDF](https://arxiv.org/pdf/2504.09836)  

**Abstract**: We propose a novel control-theoretic framework that leverages principles from generative modeling -- specifically, Denoising Diffusion Probabilistic Models (DDPMs) -- to stabilize control-affine systems with nonholonomic constraints. Unlike traditional stochastic approaches, which rely on noise-driven dynamics in both forward and reverse processes, our method crucially eliminates the need for noise in the reverse phase, making it particularly relevant for control applications. We introduce two formulations: one where noise perturbs all state dimensions during the forward phase while the control system enforces time reversal deterministically, and another where noise is restricted to the control channels, embedding system constraints directly into the forward process.
For controllable nonlinear drift-free systems, we prove that deterministic feedback laws can exactly reverse the forward process, ensuring that the system's probability density evolves correctly without requiring artificial diffusion in the reverse phase. Furthermore, for linear time-invariant systems, we establish a time-reversal result under the second formulation. By eliminating noise in the backward process, our approach provides a more practical alternative to machine learning-based denoising methods, which are unsuitable for control applications due to the presence of stochasticity. We validate our results through numerical simulations on benchmark systems, including a unicycle model in a domain with obstacles, a driftless five-dimensional system, and a four-dimensional linear system, demonstrating the potential for applying diffusion-inspired techniques in linear, nonlinear, and settings with state space constraints. 

**Abstract (ZH)**: 我们提出了一种新颖的控制理论框架，利用生成模型原理——特别是去噪扩散概率模型（DDPMs）——来稳定具有非完整约束的控制协调系统。与依赖正向和逆向过程中的噪声驱动动力学的传统随机方法不同，我们的方法在逆向阶段完全消除了噪声的需要，使其特别适用于控制应用。我们引入了两种形式：一种是噪声在正向阶段扰动所有状态维度，而控制系统在时间反演过程中确定性地起作用；另一种是噪声仅限于控制通道，在正向过程中直接嵌入系统约束。对于可控的无漂移非线性系统，我们证明了确定性的反馈法则可以在正向过程中精确地实现时间反演，从而确保系统的概率密度正确演化，而不必在逆向阶段引入人工扩散。此外，对于线性时不变系统，在第二种形式下，我们建立了时间反演结果。通过在逆向过程中消除噪声，我们的方法提供了一种比基于机器学习的去噪方法更实用的替代方案，这类方法由于存在随机性而不适用于控制应用。我们通过在基准系统上的数值模拟验证了这些结果，包括具有障碍物的单轮车模型、无漂移的五维系统和四维线性系统，展示了以扩散启发技术应用于线性、非线性和状态空间约束设置中的潜力。 

---
# RoboComm: A DID-based scalable and privacy-preserving Robot-to-Robot interaction over state channels 

**Title (ZH)**: RoboComm：基于DID的可扩展且隐私保护的机器人间状态通道交互 

**Authors**: Roshan Singh, Sushant Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2504.09517)  

**Abstract**: In a multi robot system establishing trust amongst untrusted robots from different organisations while preserving a robot's privacy is a challenge. Recently decentralized technologies such as smart contract and blockchain are being explored for applications in robotics. However, the limited transaction processing and high maintenance cost hinder the widespread adoption of such approaches. Moreover, blockchain transactions be they on public or private permissioned blockchain are publically readable which further fails to preserve the confidentiality of the robot's data and privacy of the robot.
In this work, we propose RoboComm a Decentralized Identity based approach for privacy-preserving interaction between robots. With DID a component of Self-Sovereign Identity; robots can authenticate each other independently without relying on any third-party service. Verifiable Credentials enable private data associated with a robot to be stored within the robot's hardware, unlike existing blockchain based approaches where the data has to be on the blockchain. We improve throughput by allowing message exchange over state channels. Being a blockchain backed solution RoboComm provides a trustworthy system without relying on a single party. Moreover, we implement our proposed approach to demonstrate the feasibility of our solution. 

**Abstract (ZH)**: 基于分布式身份的身份保护型机器人通信方法 

---
# ES-HPC-MPC: Exponentially Stable Hybrid Perception Constrained MPC for Quadrotor with Suspended Payloads 

**Title (ZH)**: ES-HPC-MPC: 悬挂载荷旋翼无人机的指数稳定混合感知约束模型预测控制 

**Authors**: Luis F. Recalde, Mrunal Sarvaiya, Giuseppe Loianno, Guanrui Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.08841)  

**Abstract**: Aerial transportation using quadrotors with cable-suspended payloads holds great potential for applications in disaster response, logistics, and infrastructure maintenance. However, their hybrid and underactuated dynamics pose significant control and perception challenges. Traditional approaches often assume a taut cable condition, limiting their effectiveness in real-world applications where slack-to-taut transitions occur due to disturbances. We introduce ES-HPC-MPC, a model predictive control framework that enforces exponential stability and perception-constrained control under hybrid dynamics.
Our method leverages Exponentially Stabilizing Control Lyapunov Functions (ES-CLFs) to enforce stability during the tasks and Control Barrier Functions (CBFs) to maintain the payload within the onboard camera's field of view (FoV). We validate our method through both simulation and real-world experiments, demonstrating stable trajectory tracking and reliable payload perception. We validate that our method maintains stability and satisfies perception constraints while tracking dynamically infeasible trajectories and when the system is subjected to hybrid mode transitions caused by unexpected disturbances. 

**Abstract (ZH)**: 基于缆绳悬吊载荷的四旋翼无人机空中运输在灾害响应、物流和基础设施维护中有巨大的应用潜力。然而，其混合和欠驱动动力学给控制和感知带来了显著挑战。传统的控制方法通常假设缆绳紧绷状态，限定了其在因干扰导致松弛到紧绷状态过渡的实际应用场景中的有效性。我们引入了ES-HPC-MPC模型预测控制框架，在混合动力学下保证指数稳定性并受感知约束的控制。 

---
# Endowing Embodied Agents with Spatial Reasoning Capabilities for Vision-and-Language Navigation 

**Title (ZH)**: 为视觉-语言导航赋予实体代理空间推理能力 

**Authors**: Luo Ling, Bai Qianqian  

**Link**: [PDF](https://arxiv.org/pdf/2504.08806)  

**Abstract**: Enhancing the spatial perception capabilities of mobile robots is crucial for achieving embodied Vision-and-Language Navigation (VLN). Although significant progress has been made in simulated environments, directly transferring these capabilities to real-world scenarios often results in severe hallucination phenomena, causing robots to lose effective spatial awareness. To address this issue, we propose BrainNav, a bio-inspired spatial cognitive navigation framework inspired by biological spatial cognition theories and cognitive map theory. BrainNav integrates dual-map (coordinate map and topological map) and dual-orientation (relative orientation and absolute orientation) strategies, enabling real-time navigation through dynamic scene capture and path planning. Its five core modules-Hippocampal Memory Hub, Visual Cortex Perception Engine, Parietal Spatial Constructor, Prefrontal Decision Center, and Cerebellar Motion Execution Unit-mimic biological cognitive functions to reduce spatial hallucinations and enhance adaptability. Validated in a zero-shot real-world lab environment using the Limo Pro robot, BrainNav, compatible with GPT-4, outperforms existing State-of-the-Art (SOTA) Vision-and-Language Navigation in Continuous Environments (VLN-CE) methods without fine-tuning. 

**Abstract (ZH)**: 增强移动机器人在视觉-语言导航中的空间感知能力对于实现具身的视觉-语言导航(VLN)至关重要。尽管在模拟环境中已经取得了显著进展，但这些能力直接转移到真实世界场景中时，往往会引发严重的幻觉现象，导致机器人失去有效的空间意识。为了解决这一问题，我们提出了一种基于生物空间认知理论和认知地图理论的神经启发式空间认知导航框架BrainNav。BrainNav结合了双地图（坐标地图和拓扑地图）和双方向（相对方向和绝对方向）策略，通过动态场景捕获和路径规划实现实时导航。其五个核心模块——海马体记忆中心、视觉皮层感知引擎、顶叶空间构造器、前额叶决策中心和小脑运动执行单元——模拟了生物认知功能，以减少空间幻觉并增强适应性。在使用Limo Pro机器人进行的零样本真实世界实验室环境中验证，兼容GPT-4的BrainNav在无需微调的情况下优于现有的视觉-语言导航(SOTA)方法。 

---
