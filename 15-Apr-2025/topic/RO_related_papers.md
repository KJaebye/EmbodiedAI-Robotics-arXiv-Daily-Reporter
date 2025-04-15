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
# Ankle Exoskeletons in Walking and Load-Carrying Tasks: Insights into Biomechanics and Human-Robot Interaction 

**Title (ZH)**: 膝踝矫形器在行走和负重任务中的研究：生物力学与人机交互见解 

**Authors**: J.F. Almeida, J. André, C.P. Santos  

**Link**: [PDF](https://arxiv.org/pdf/2504.10294)  

**Abstract**: Background: Lower limb exoskeletons can enhance quality of life, but widespread adoption is limited by the lack of frameworks to assess their biomechanical and human-robot interaction effects, which are essential for developing adaptive and personalized control strategies. Understanding impacts on kinematics, muscle activity, and HRI dynamics is key to achieve improved usability of wearable robots. Objectives: We propose a systematic methodology evaluate an ankle exoskeleton's effects on human movement during walking and load-carrying (10 kg front pack), focusing on joint kinematics, muscle activity, and HRI torque signals. Materials and Methods: Using Xsens MVN (inertial motion capture), Delsys EMG, and a unilateral exoskeleton, three experiments were conducted: (1) isolated dorsiflexion/plantarflexion; (2) gait analysis (two subjects, passive/active modes); and (3) load-carrying under assistance. Results and Conclusions: The first experiment confirmed that the HRI sensor captured both voluntary and involuntary torques, providing directional torque insights. The second experiment showed that the device slightly restricted ankle range of motion (RoM) but supported normal gait patterns across all assistance modes. The exoskeleton reduced muscle activity, particularly in active mode. HRI torque varied according to gait phases and highlighted reduced synchronization, suggesting a need for improved support. The third experiment revealed that load-carrying increased GM and TA muscle activity, but the device partially mitigated user effort by reducing muscle activity compared to unassisted walking. HRI increased during load-carrying, providing insights into user-device dynamics. These results demonstrate the importance of tailoring exoskeleton evaluation methods to specific devices and users, while offering a framework for future studies on exoskeleton biomechanics and HRI. 

**Abstract (ZH)**: 背景：下肢外骨骼可以提升生活质量，但其广泛应用受限于缺乏评估其生物力学和人机器人交互效果的框架，这些是开发适应性和个性化控制策略的关键。理解运动学、肌肉活动和人机器人交互动力学的影响是实现可穿戴机器人更好易用性的关键。目的：我们提出了一种系统性方法，评估踝关节外骨骼在行走和负重（10 kg 前背包）时对外力作用对人体运动的影响，重点在于关节运动学、肌肉活动和人机器人交互扭矩信号。材料与方法：使用Xsens MVN（惯性动作捕捉）、Delsys EMG和单侧外骨骼进行了三项实验：（1）孤立的背屈/跖屈；（2）步态分析（两名受试者，被动/主动模式）；（3）辅助负重。结果与结论：第一项实验证实了人机器人交互传感器捕获了自愿和不自愿的扭矩，并提供了方向扭矩的见解。第二项实验表明，该设备轻微限制了踝关节运动范围，但支持所有辅助模式下的正常步态模式。外骨骼减少了肌肉活动，特别是在主动模式下。人机器人交互扭矩随着步行周期的不同而变化，突显了减少同步的现象，表明需要改进支持。第三项实验发现，负重增加了股四头肌和胫骨前肌的活动，但该设备通过减少与未辅助行走相比的肌肉活动部分缓解了用户的用力情况。负重状态下人机器人交互扭矩增加，提供了用户-设备动力学的见解。这些结果表明，对外骨骼评价方法进行定制以适应特定设备和用户的重要性，同时为未来外骨骼生物力学和人机器人交互的研究提供了一个框架。 

---
# Vision based driving agent for race car simulation environments 

**Title (ZH)**: 基于视觉的赛车模拟环境驾驶代理 

**Authors**: Gergely Bári, László Palkovics  

**Link**: [PDF](https://arxiv.org/pdf/2504.10266)  

**Abstract**: In recent years, autonomous driving has become a popular field of study. As control at tire grip limit is essential during emergency situations, algorithms developed for racecars are useful for road cars too. This paper examines the use of Deep Reinforcement Learning (DRL) to solve the problem of grip limit driving in a simulated environment. Proximal Policy Optimization (PPO) method is used to train an agent to control the steering wheel and pedals of the vehicle, using only visual inputs to achieve professional human lap times. The paper outlines the formulation of the task of time optimal driving on a race track as a deep reinforcement learning problem, and explains the chosen observations, actions, and reward functions. The results demonstrate human-like learning and driving behavior that utilize maximum tire grip potential. 

**Abstract (ZH)**: 近年来，自动驾驶已成为一个热门的研究领域。由于在紧急情况下轮胎抓地力极限的控制至关重要，用于赛车的算法同样适用于普通车辆。本文探讨了使用深度强化学习（DRL）在模拟环境中解决轮胎抓地力极限驾驶问题的方法。采用 proximal policy optimization (PPO) 方法训练一个代理，使其仅通过视觉输入控制车辆的方向盘和踏板，实现专业级的人际圈速。本文阐述了将赛道上时间最优驾驶任务建模为深度强化学习问题的方法，并解释了所选择的观测、行动和奖励函数。实验结果展示了利用轮胎最大抓地力潜力的人类似驾驶行为。 

---
# Shoulder Range of Motion Rehabilitation Robot Incorporating Scapulohumeral Rhythm for Frozen Shoulder 

**Title (ZH)**: 肩关节活动度康复机器人结合肩锁关节运动规律用于治疗粘连性肩关节囊炎 

**Authors**: Hyunbum Cho, Sungmoon Hur, Joowan Kim, Keewon Kim, Jaeheung Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.10163)  

**Abstract**: This paper presents a novel rehabilitation robot designed to address the challenges of passive range of motion (PROM) exercises for frozen shoulder patients by integrating advanced scapulohumeral rhythm stabilization. Frozen shoulder is characterized by limited glenohumeral motion and disrupted scapulohumeral rhythm, with therapist-assisted interventions being highly effective for restoring normal shoulder function. While existing robotic solutions replicate natural shoulder biomechanics, they lack the ability to stabilize compensatory movements, such as shoulder shrugging, which are critical for effective rehabilitation. Our proposed device features a 6 degrees of freedom (DoF) mechanism, including 5 DoF for shoulder motion and an innovative 1 DoF Joint press for scapular stabilization. The robot employs a personalized two-phase operation: recording normal shoulder movement patterns from the unaffected side and applying them to guide the affected side. Experimental results demonstrated the robot's ability to replicate recorded motion patterns with high precision, with root mean square error (RMSE) values consistently below 1 degree. In simulated frozen shoulder conditions, the robot effectively suppressed scapular elevation, delaying the onset of compensatory movements and guiding the affected shoulder to move more closely in alignment with normal shoulder motion, particularly during arm elevation movements such as abduction and flexion. These findings confirm the robot's potential as a rehabilitation tool capable of automating PROM exercises while correcting compensatory movements. The system provides a foundation for advanced, personalized rehabilitation for patients with frozen shoulders. 

**Abstract (ZH)**: 一种用于治疗冻结肩患者被动关节活动范围锻炼的新型康复机器人及其应用研究 

---
# A Framework for Adaptive Load Redistribution in Human-Exoskeleton-Cobot Systems 

**Title (ZH)**: 人类-外骨骼-协作机器人系统中自适应负载重新分布的框架 

**Authors**: Emir Mobedi, Gokhan Solak, Arash Ajoudani  

**Link**: [PDF](https://arxiv.org/pdf/2504.10066)  

**Abstract**: Wearable devices like exoskeletons are designed to reduce excessive loads on specific joints of the body. Specifically, single- or two-degrees-of-freedom (DOF) upper-body industrial exoskeletons typically focus on compensating for the strain on the elbow and shoulder joints. However, during daily activities, there is no assurance that external loads are correctly aligned with the supported joints. Optimizing work processes to ensure that external loads are primarily (to the extent that they can be compensated by the exoskeleton) directed onto the supported joints can significantly enhance the overall usability of these devices and the ergonomics of their users. Collaborative robots (cobots) can play a role in this optimization, complementing the collaborative aspects of human work. In this study, we propose an adaptive and coordinated control system for the human-cobot-exoskeleton interaction. This system adjusts the task coordinates to maximize the utilization of the supported joints. When the torque limits of the exoskeleton are exceeded, the framework continuously adapts the task frame, redistributing excessive loads to non-supported body joints to prevent overloading the supported ones. We validated our approach in an equivalent industrial painting task involving a single-DOF elbow exoskeleton, a cobot, and four subjects, each tested in four different initial arm configurations with five distinct optimisation weight matrices and two different payloads. 

**Abstract (ZH)**: 可穿戴设备如外骨骼旨在减轻身体特定关节的过度负荷。具体而言，单自由度或双自由度上肢工业外骨骼通常侧重于缓解肘关节和肩关节的紧张。然而，在日常活动中，并不能保证外部负荷正确对准支持的关节。通过优化工作流程，确保外部负荷主要（在可以被外骨骼补偿的范围内）作用在支持的关节上，可以显著提升这些设备的整体可用性和用户的工效学。协作机器人（ cobots）可以在这一优化中发挥作用，补充人类工作的协作性。在本研究中，我们提出了一种适应性和协调性的控制系统，用于人类-协作机器人-外骨骼交互。该系统调整任务坐标，以最大化利用支持的关节。当外骨骼的扭矩限制被超过时，该框架会持续适应任务框架，重新分配过度的负荷至非支持的体关节，以防止过度负担支持的关节。我们通过一项包含单自由度肘关节外骨骼、一个协作机器人和四名受试者的等效工业喷漆任务进行了验证，每位受试者在四种不同的初始臂配置下，分别使用五个不同的优化权重矩阵和两种不同负载进行测试。 

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
# From Movement Primitives to Distance Fields to Dynamical Systems 

**Title (ZH)**: 从运动原型到距离场再到动力学系统 

**Authors**: Yiming Li, Sylvain Calinon  

**Link**: [PDF](https://arxiv.org/pdf/2504.09705)  

**Abstract**: Developing autonomous robots capable of learning and reproducing complex motions from demonstrations remains a fundamental challenge in robotics. On the one hand, movement primitives (MPs) provide a compact and modular representation of continuous trajectories. On the other hand, autonomous systems provide control policies that are time independent. We propose in this paper a simple and flexible approach that gathers the advantages of both representations by transforming MPs into autonomous systems. The key idea is to transform the explicit representation of a trajectory as an implicit shape encoded as a distance field. This conversion from a time-dependent motion to a spatial representation enables the definition of an autonomous dynamical system with modular reactions to perturbation. Asymptotic stability guarantees are provided by using Bernstein basis functions in the MPs, representing trajectories as concatenated quadratic Bézier curves, which provide an analytical method for computing distance fields. This approach bridges conventional MPs with distance fields, ensuring smooth and precise motion encoding, while maintaining a continuous spatial representation. By simply leveraging the analytic gradients of the curve and its distance field, a stable dynamical system can be computed to reproduce the demonstrated trajectories while handling perturbations, without requiring a model of the dynamical system to be estimated. Numerical simulations and real-world robotic experiments validate our method's ability to encode complex motion patterns while ensuring trajectory stability, together with the flexibility of designing the desired reaction to perturbations. An interactive project page demonstrating our approach is available at this https URL. 

**Abstract (ZH)**: 开发能够从演示中学习和再现复杂运动的自主机器人仍然是机器人学中的一个基本挑战。本文提出了一种简单灵活的方法，通过将运动基元（MPs）转换为自主系统，集成了两种表示形式的优势。关键思想是将轨迹的显式表示转换为由距离场编码的隐式形状。这种从时间依赖运动到空间表示的转换使定义具有模块化对扰动反应的自主动力学系统成为可能。通过使用伯恩斯坦基函数表示轨迹（作为连接的二次贝塞尔曲线），并提供距离场的解析计算方法，保证了渐近稳定性。该方法将传统的运动基元与距离场相结合，确保了平滑和精确的运动编码，同时保持连续的空间表示。仅通过利用曲线及其距离场的解析梯度，可以计算出稳定的动力学系统以再现演示的轨迹并处理扰动，而无需估计动力学系统的模型。数值仿真和现实世界的机器人实验验证了该方法在确保轨迹稳定性的同时能够编码复杂运动模式的能力，并且具有灵活性，可以设计期望的对扰动的反应。我们的方法示例可在以下网址查看：此 https URL。 

---
# A highly maneuverable flying squirrel drone with agility-improving foldable wings 

**Title (ZH)**: 具有 agility 提升可折叠翼的高机动飞行毛猬无人机 

**Authors**: Dohyeon Lee, Jun-Gill Kang, Soohee Han  

**Link**: [PDF](https://arxiv.org/pdf/2504.09609)  

**Abstract**: Drones, like most airborne aerial vehicles, face inherent disadvantages in achieving agile flight due to their limited thrust capabilities. These physical constraints cannot be fully addressed through advancements in control algorithms alone. Drawing inspiration from the winged flying squirrel, this paper proposes a highly maneuverable drone equipped with agility-enhancing foldable wings. By leveraging collaborative control between the conventional propeller system and the foldable wings-coordinated through the Thrust-Wing Coordination Control (TWCC) framework-the controllable acceleration set is expanded, enabling the generation of abrupt vertical forces that are unachievable with traditional wingless drones. The complex aerodynamics of the foldable wings are modeled using a physics-assisted recurrent neural network (paRNN), which calibrates the angle of attack (AOA) to align with the real aerodynamic behavior of the wings. The additional air resistance generated by appropriately deploying these wings significantly improves the tracking performance of the proposed "flying squirrel" drone. The model is trained on real flight data and incorporates flat-plate aerodynamic principles. Experimental results demonstrate that the proposed flying squirrel drone achieves a 13.1% improvement in tracking performance, as measured by root mean square error (RMSE), compared to a conventional wingless drone. A demonstration video is available on YouTube: this https URL. 

**Abstract (ZH)**: 基于仿翼 squirrel 飞行器的敏捷无人机及其控制方法 

---
# Towards Intuitive Drone Operation Using a Handheld Motion Controller 

**Title (ZH)**: 基于手持运动控制器的直观无人机操作研究 

**Authors**: Daria Trinitatova, Sofia Shevelo, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2504.09510)  

**Abstract**: We present an intuitive human-drone interaction system that utilizes a gesture-based motion controller to enhance the drone operation experience in real and simulated environments. The handheld motion controller enables natural control of the drone through the movements of the operator's hand, thumb, and index finger: the trigger press manages the throttle, the tilt of the hand adjusts pitch and roll, and the thumbstick controls yaw rotation. Communication with drones is facilitated via the ExpressLRS radio protocol, ensuring robust connectivity across various frequencies. The user evaluation of the flight experience with the designed drone controller using the UEQ-S survey showed high scores for both Pragmatic (mean=2.2, SD = 0.8) and Hedonic (mean=2.3, SD = 0.9) Qualities. This versatile control interface supports applications such as research, drone racing, and training programs in real and simulated environments, thereby contributing to advances in the field of human-drone interaction. 

**Abstract (ZH)**: 一种基于手势的直观人机操控系统：增强无人机操作体验的研究 

---
# A highly maneuverable flying squirrel drone with controllable foldable wings 

**Title (ZH)**: 可折叠翼片高度机动的飞鼠无人机 

**Authors**: Jun-Gill Kang, Dohyeon Lee, Soohee Han  

**Link**: [PDF](https://arxiv.org/pdf/2504.09478)  

**Abstract**: Typical drones with multi rotors are generally less maneuverable due to unidirectional thrust, which may be unfavorable to agile flight in very narrow and confined spaces. This paper suggests a new bio-inspired drone that is empowered with high maneuverability in a lightweight and easy-to-carry way. The proposed flying squirrel inspired drone has controllable foldable wings to cover a wider range of flight attitudes and provide more maneuverable flight capability with stable tracking performance. The wings of a drone are fabricated with silicone membranes and sophisticatedly controlled by reinforcement learning based on human-demonstrated data. Specially, such learning based wing control serves to capture even the complex aerodynamics that are often impossible to model mathematically. It is shown through experiment that the proposed flying squirrel drone intentionally induces aerodynamic drag and hence provides the desired additional repulsive force even under saturated mechanical thrust. This work is very meaningful in demonstrating the potential of biomimicry and machine learning for realizing an animal-like agile drone. 

**Abstract (ZH)**: 受松鼠启发的生物灵感无人机：轻量化高性能机动飞行技术 

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
# Anti-Slip AI-Driven Model-Free Control with Global Exponential Stability in Skid-Steering Robots 

**Title (ZH)**: 基于滑移 steering 机器人全局指数稳定性的AI驱动无模型防滑控制 

**Authors**: Mehdi Heydari Shahna, Pauli Mustalahti, Jouni Mattila  

**Link**: [PDF](https://arxiv.org/pdf/2504.08831)  

**Abstract**: Undesired lateral and longitudinal wheel slippage can disrupt a mobile robot's heading angle, traction, and, eventually, desired motion. This issue makes the robotization and accurate modeling of heavy-duty machinery very challenging because the application primarily involves off-road terrains, which are susceptible to uneven motion and severe slippage. As a step toward robotization in skid-steering heavy-duty robot (SSHDR), this paper aims to design an innovative robust model-free control system developed by neural networks to strongly stabilize the robot dynamics in the presence of a broad range of potential wheel slippages. Before the control design, the dynamics of the SSHDR are first investigated by mathematically incorporating slippage effects, assuming that all functional modeling terms of the system are unknown to the control system. Then, a novel tracking control framework to guarantee global exponential stability of the SSHDR is designed as follows: 1) the unknown modeling of wheel dynamics is approximated using radial basis function neural networks (RBFNNs); and 2) a new adaptive law is proposed to compensate for slippage effects and tune the weights of the RBFNNs online during execution. Simulation and experimental results verify the proposed tracking control performance of a 4,836 kg SSHDR operating on slippery terrain. 

**Abstract (ZH)**: 未期望的横向和纵向车轮打滑会破坏移动机器人航向角、附着性能，并最终影响其期望运动。这一问题使得重型机械的机器人化及其精确建模极具挑战性，因为应用主要涉及易发生不规则运动和严重打滑的非铺装地形。为向滑移转向重型机器人（SSHDR）的机器人化方向迈出一步，本文旨在设计一种基于神经网络的创新鲁棒无模型控制系统，以在广泛范围的潜在车轮打滑条件下，强烈稳定机器人动力学。在控制设计之前，首先通过数学上包含打滑效应的方法研究SSHDR的动力学，并假定控制系统对系统的所有功能建模项均未知。然后提出了一种新型跟踪控制框架，以确保SSHDR的全局指数稳定：1) 使用径向基函数神经网络（RBFNNs）近似未知的车轮动力学模型；2) 提出了一种新的自适应律，在执行过程中补偿打滑效应并在线调整RBFNNs的权重。仿真和实验结果验证了该跟踪控制方法在滑溜地面上运行4836 kg SSHDR的性能。 

---
# ES-HPC-MPC: Exponentially Stable Hybrid Perception Constrained MPC for Quadrotor with Suspended Payloads 

**Title (ZH)**: ES-HPC-MPC: 悬挂载荷旋翼无人机的指数稳定混合感知约束模型预测控制 

**Authors**: Luis F. Recalde, Mrunal Sarvaiya, Giuseppe Loianno, Guanrui Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.08841)  

**Abstract**: Aerial transportation using quadrotors with cable-suspended payloads holds great potential for applications in disaster response, logistics, and infrastructure maintenance. However, their hybrid and underactuated dynamics pose significant control and perception challenges. Traditional approaches often assume a taut cable condition, limiting their effectiveness in real-world applications where slack-to-taut transitions occur due to disturbances. We introduce ES-HPC-MPC, a model predictive control framework that enforces exponential stability and perception-constrained control under hybrid dynamics.
Our method leverages Exponentially Stabilizing Control Lyapunov Functions (ES-CLFs) to enforce stability during the tasks and Control Barrier Functions (CBFs) to maintain the payload within the onboard camera's field of view (FoV). We validate our method through both simulation and real-world experiments, demonstrating stable trajectory tracking and reliable payload perception. We validate that our method maintains stability and satisfies perception constraints while tracking dynamically infeasible trajectories and when the system is subjected to hybrid mode transitions caused by unexpected disturbances. 

**Abstract (ZH)**: 基于缆绳悬吊载荷的四旋翼无人机空中运输在灾害响应、物流和基础设施维护中有巨大的应用潜力。然而，其混合和欠驱动动力学给控制和感知带来了显著挑战。传统的控制方法通常假设缆绳紧绷状态，限定了其在因干扰导致松弛到紧绷状态过渡的实际应用场景中的有效性。我们引入了ES-HPC-MPC模型预测控制框架，在混合动力学下保证指数稳定性并受感知约束的控制。 

---
