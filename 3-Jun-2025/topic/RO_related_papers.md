# Feel the Force: Contact-Driven Learning from Humans 

**Title (ZH)**: 感受力量：由接触驱动的人机学习 

**Authors**: Ademi Adeniji, Zhuoran Chen, Vincent Liu, Venkatesh Pattabiraman, Raunaq Bhirangi, Siddhant Haldar, Pieter Abbeel, Lerrel Pinto  

**Link**: [PDF](https://arxiv.org/pdf/2506.01944)  

**Abstract**: Controlling fine-grained forces during manipulation remains a core challenge in robotics. While robot policies learned from robot-collected data or simulation show promise, they struggle to generalize across the diverse range of real-world interactions. Learning directly from humans offers a scalable solution, enabling demonstrators to perform skills in their natural embodiment and in everyday environments. However, visual demonstrations alone lack the information needed to infer precise contact forces. We present FeelTheForce (FTF): a robot learning system that models human tactile behavior to learn force-sensitive manipulation. Using a tactile glove to measure contact forces and a vision-based model to estimate hand pose, we train a closed-loop policy that continuously predicts the forces needed for manipulation. This policy is re-targeted to a Franka Panda robot with tactile gripper sensors using shared visual and action representations. At execution, a PD controller modulates gripper closure to track predicted forces-enabling precise, force-aware control. Our approach grounds robust low-level force control in scalable human supervision, achieving a 77% success rate across 5 force-sensitive manipulation tasks. Code and videos are available at this https URL. 

**Abstract (ZH)**: 操纵过程中精细力量的控制仍然是机器人技术中的核心挑战。通过人类直接学习提供的解决方案可扩展，使演示者能够在自然身体形态和日常环境中执行技能。然而，仅通过视觉演示无法提供推断精确接触力所需的信息。我们提出FeelTheForce (FTF)：一种机器人学习系统，用于模仿人类触觉行为以学习力敏感操纵。借助触觉手套测量接触力，并使用基于视觉的模型估计手部姿态，我们训练了一个闭环策略，该策略能够持续预测操纵所需的力。该策略经重新调整以适用于配备触觉 gripper 传感器的 Franka Panda 机器人，采用共享的视觉和动作表征。在执行过程中，PD 控制器调节 gripper 的闭合以追踪预测的力，实现精确的力感知控制。我们的方法将鲁棒的低级力控制与可扩展的人类监督相结合，在 5 项力敏感操纵任务中实现了 77% 的成功率。更多信息请参见此链接。 

---
# FreeTacMan: Robot-free Visuo-Tactile Data Collection System for Contact-rich Manipulation 

**Title (ZH)**: FreeTacMan: 无需机器人的一种触觉-视觉数据采集系统用于接触丰富的操作任务 

**Authors**: Longyan Wu, Checheng Yu, Jieji Ren, Li Chen, Ran Huang, Guoying Gu, Hongyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.01941)  

**Abstract**: Enabling robots with contact-rich manipulation remains a pivotal challenge in robot learning, which is substantially hindered by the data collection gap, including its inefficiency and limited sensor setup. While prior work has explored handheld paradigms, their rod-based mechanical structures remain rigid and unintuitive, providing limited tactile feedback and posing challenges for human operators. Motivated by the dexterity and force feedback of human motion, we propose FreeTacMan, a human-centric and robot-free data collection system for accurate and efficient robot manipulation. Concretely, we design a wearable data collection device with dual visuo-tactile grippers, which can be worn by human fingers for intuitive and natural control. A high-precision optical tracking system is introduced to capture end-effector poses, while synchronizing visual and tactile feedback simultaneously. FreeTacMan achieves multiple improvements in data collection performance compared to prior works, and enables effective policy learning for contact-rich manipulation tasks with the help of the visuo-tactile information. We will release the work to facilitate reproducibility and accelerate research in visuo-tactile manipulation. 

**Abstract (ZH)**: 基于接触丰富的 mão 动作的自适应机器人数据收集系统：FreeTacMan 

---
# Riemannian Time Warping: Multiple Sequence Alignment in Curved Spaces 

**Title (ZH)**: 黎曼流形时间扭曲：弯曲空间中的多重序列对齐 

**Authors**: Julian Richter, Christopher Erdös, Christian Scheurer, Jochen J. Steil, Niels Dehio  

**Link**: [PDF](https://arxiv.org/pdf/2506.01635)  

**Abstract**: Temporal alignment of multiple signals through time warping is crucial in many fields, such as classification within speech recognition or robot motion learning. Almost all related works are limited to data in Euclidean space. Although an attempt was made in 2011 to adapt this concept to unit quaternions, a general extension to Riemannian manifolds remains absent. Given its importance for numerous applications in robotics and beyond, we introduce Riemannian Time Warping~(RTW). This novel approach efficiently aligns multiple signals by considering the geometric structure of the Riemannian manifold in which the data is embedded. Extensive experiments on synthetic and real-world data, including tests with an LBR iiwa robot, demonstrate that RTW consistently outperforms state-of-the-art baselines in both averaging and classification tasks. 

**Abstract (ZH)**: 基于黎曼几何的时间扭曲多信号对齐方法 

---
# RoboTwin: A Robotic Teleoperation Framework Using Digital Twins 

**Title (ZH)**: RoboTwin: 基于数字孪生的机器人远程操作框架 

**Authors**: Harsha Yelchuri, Diwakar Kumar Singh, Nithish Krishnabharathi Gnani, T V Prabhakar, Chandramani Singh  

**Link**: [PDF](https://arxiv.org/pdf/2506.01027)  

**Abstract**: Robotic surgery imposes a significant cognitive burden on the surgeon. This cognitive burden increases in the case of remote robotic surgeries due to latency between entities and thus might affect the quality of surgery. Here, the patient side and the surgeon side are geographically separated by hundreds to thousands of kilometres. Real-time teleoperation of robots requires strict latency bounds for control and feedback. We propose a dual digital twin (DT) framework and explain the simulation environment and teleoperation framework. Here, the doctor visually controls the locally available DT of the patient side and thus experiences minimum latency. The second digital twin serves two purposes. Firstly, it provides a layer of safety for operator-related mishaps, and secondly, it conveys the coordinates of known and unknown objects back to the operator's side digital twin. We show that teleoperation accuracy and user experience are enhanced with our approach. Experimental results using the NASA-TLX metric show that the quality of surgery is vastly improved with DT, perhaps due to reduced cognitive burden. The network data rate for identifying objects at the operator side is 25x lower than normal. 

**Abstract (ZH)**: 机器人手术对外科医生产生了显著的认知负担。远程机器人手术由于实体之间的延迟增加，这种认知负担可能会进一步影响手术质量。在这种情况下，患者端和医生端相隔数百至数千公里。实时遥控机器人需要严格控制延迟和反馈的时限要求。我们提出了一种双数字孪生（DT）框架，并解释了模拟环境和遥控框架。医生通过远程控制患者端的本地可用数字孪生体，从而体验最小的延迟。第二个数字孪生体有两个作用：首先，为操作员相关失误提供一层安全保护；其次，将已知和未知物体的坐标信息传回到操作员端的数字孪生体。我们的方法提高了遥控操作的准确性和用户体验。使用NASA-TLX指标的实验结果表明，使用数字孪生体提高了手术质量，可能是由于减少了认知负担。操作端识别物体的网络数据传输速率降低了25倍。 

---
# Multi-Objective Neural Network Assisted Design Optimization of Soft Fin-Ray Grippers for Enhanced Grasping Performance 

**Title (ZH)**: 软鳍肋夹持器多目标神经网络辅助设计优化以增强抓取性能 

**Authors**: Ali Ghanizadeh, Ali Ahmadi, Arash Bahrami  

**Link**: [PDF](https://arxiv.org/pdf/2506.00494)  

**Abstract**: Soft Fin-Ray grippers can perform delicate and careful manipulation, which has caused notable attention in different fields. These grippers can handle objects of various forms and sizes safely. The internal structure of the Fin-Ray finger plays a significant role in its adaptability and grasping performance. However, modeling the non-linear grasp force and deformation behaviors for design purposes is challenging. Moreover, when the Fin-Ray finger becomes more rigid and capable of exerting higher forces, it becomes less delicate in handling objects. The contrast between these two objectives gives rise to a multi-objective optimization problem. In this study, we employ finite element method (FEM) to estimate the deflections and contact forces of the Fin-Ray, grasping cylindrical objects. This dataset is then used to construct a multilayer perception (MLP) for prediction of the contact force and the tip displacement. The FEM dataset consists of three input and four target features. The three input features of the MLP and optimization design variables are the thickness of the front and supporting beams, the thickness of the cross beams, and the equal spacing between the cross beams. In addition, the target features are the maximum contact forces and maximum tip displacements in x- and y-directions. The magnitude of maximum contact force and magnitude of maximum tip displacement are the two objectives, showing the trade-off between force and delicate manipulation in soft Fin-Ray grippers. Furthermore, the optimized set of solutions are found using multi-objective optimal techniques. We use non-dominated sorting genetic algorithm (NSGA-II) method for this purpose. Our findings demonstrate that our methodologies can be used to improve the design and gripping performance of soft robotic grippers, helping us to choose a design not only for delicate grasping but also for high-force applications. 

**Abstract (ZH)**: Soft Fin-Ray 夹持器可以进行精细和谨慎的操作，已在不同领域引起广泛关注。这些夹持器能安全地处理各种形状和大小的物体。Fin-Ray 指夹的内部结构在它的适应性和夹持性能中起着重要作用。然而，为了设计目的，对非线性夹持力和变形行为建模具有挑战性。此外，当 Fin-Ray 指夹变得更刚硬并能够施加更大的力时，它在操作物体时会变得不够细致。这两种目标之间的对比产生了多目标优化问题。在本研究中，我们采用有限元方法（FEM）估计 Fin-Ray 对圆柱形物体进行夹持时的位移和接触力，然后利用此数据集构建多层感知器（MLP）以预测接触力和末端位移。FEM 数据集包括三个输入特征和四个目标特征。MLP 和优化设计变量的三个输入特征为前梁和支撑梁的厚度、横梁的厚度以及横梁之间的等间距。此外，目标特征为 x- 和 y-方向的最大接触力和最大末端位移。最大接触力的大小和最大末端位移的大小是两个目标，表明软 Fin-Ray 夹持器中力量与精细操作之间的权衡。进一步使用多目标优化技术找到优化解集。我们使用非支配排序遗传算法（NSGA-II）方法进行此目的。我们的研究结果表明，我们的方法可用于改进软柔体夹持器的设计和夹持性能，帮助我们在精细夹持和高力应用之间做出设计选择。 

---
# Tunable Virtual IMU Frame by Weighted Averaging of Multiple Non-Collocated IMUs 

**Title (ZH)**: 可调虚拟IMU框架通过多个非对齐IMU加权平均实现 

**Authors**: Yizhou Gao, Tim Barfoot  

**Link**: [PDF](https://arxiv.org/pdf/2506.00371)  

**Abstract**: We present a new method to combine several rigidly connected but physically separated IMUs through a weighted average into a single virtual IMU (VIMU). This has the benefits of (i) reducing process noise through averaging, and (ii) allowing for tuning the location of the VIMU. The VIMU can be placed to be coincident with, for example, a camera frame or GNSS frame, thereby offering a quality-of-life improvement for users. Specifically, our VIMU removes the need to consider any lever-arm terms in the propagation model. We also present a quadratic programming method for selecting the weights to minimize the noise of the VIMU while still selecting the placement of its reference frame. We tested our method in simulation and validated it on a real dataset. The results show that our averaging technique works for IMUs with large separation and performance gain is observed in both the simulation and the real experiment compared to using only a single IMU. 

**Abstract (ZH)**: 我们提出了一种通过加权平均将多个物理上分离但刚性连接的IMU组合成单一虚拟IMU（VIMU）的新方法。这种方法的好处包括：（i）通过平均减少过程噪声，（ii）允许调整VIMU的位置。VIMU可以放置为与，例如，相机框架或GNSS框架重合，从而为用户提供生活质量的改善。具体而言，我们的VIMU消除了在传播模型中考虑任何力臂项的需求。我们还提出了一种二次规划方法来选择权重，以最小化VIMU的噪声同时选择其参考框架的位置。我们在仿真中测试了该方法并在实际数据集上进行了验证。结果显示，对于具有较大分离度的IMU，我们的平均技术是有效的，在仿真和实际实验中都观察到了使用单一IMU的性能提升。 

---
# Haptic Rapidly-Exploring Random Trees: A Sampling-based Planner for Quasi-static Manipulation Tasks 

**Title (ZH)**: 触觉快速探索随机树：一种基于采样的准静态操作任务规划器 

**Authors**: Lin Yang, Huu-Thiet Nguyen, Donghan Yu, Chen Lv, Domenico Campolo  

**Link**: [PDF](https://arxiv.org/pdf/2506.00351)  

**Abstract**: In this work, we explore how conventional motion planning algorithms can be reapplied to contact-rich manipulation tasks. Rather than focusing solely on efficiency, we investigate how manipulation aspects can be recast in terms of conventional motion-planning algorithms. Conventional motion planners, such as Rapidly-Exploring Random Trees (RRT), typically compute collision-free paths in configuration space. However, in manipulation tasks, intentional contact is often necessary. For example, when dealing with a crowded bookshelf, a robot must strategically push books aside before inserting a new one. In such scenarios, classical motion planners often fail because of insufficient space. As such, we presents Haptic Rapidly-Exploring Random Trees (HapticRRT), a planning algorithm that incorporates a recently proposed optimality measure in the context of \textit{quasi-static} manipulation, based on the (squared) Hessian of manipulation potential. The key contributions are i) adapting classical RRT to a framework that re-frames quasi-static manipulation as a planning problem on an implicit equilibrium manifold; ii) discovering multiple manipulation strategies, corresponding to branches of the equilibrium manifold. iii) providing deeper insight to haptic obstacle and haptic metric, enhancing interpretability. We validate our approach on a simulated pendulum and a real-world crowded bookshelf task, demonstrating its ability to autonomously discover strategic wedging-in policies and multiple branches. The video can be found at this https URL 

**Abstract (ZH)**: 基于触觉的快速扩展随机树在接触富有的 manipulation 任务中的应用 

---
