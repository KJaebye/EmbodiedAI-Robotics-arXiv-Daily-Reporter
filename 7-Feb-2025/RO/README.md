# DexterityGen: Foundation Controller for Unprecedented Dexterity 

**Title (ZH)**: 《DexterityGen：前所未有的灵巧性基础控制器》

这个标题翻译成中文时，保持了原文的结构和学术规范，同时确保了意思的准确传达。“Dexterity”在这里指的是操作或使用物体的灵巧能力，“Foundation Controller”可以理解为基础控制器或者基础控制模块。 

**Authors**: Zhao-Heng Yin, Changhao Wang, Luis Pineda, Francois Hogan, Krishna Bodduluri, Akash Sharma, Patrick Lancaster, Ishita Prasad, Mrinal Kalakrishnan, Jitendra Malik, Mike Lambeta, Tingfan Wu, Pieter Abbeel, Mustafa Mukadam  

**Link**: [PDF](https://arxiv.org/pdf/2502.04307)  

**Abstract**: Teaching robots dexterous manipulation skills, such as tool use, presents a significant challenge. Current approaches can be broadly categorized into two strategies: human teleoperation (for imitation learning) and sim-to-real reinforcement learning. The first approach is difficult as it is hard for humans to produce safe and dexterous motions on a different embodiment without touch feedback. The second RL-based approach struggles with the domain gap and involves highly task-specific reward engineering on complex tasks. Our key insight is that RL is effective at learning low-level motion primitives, while humans excel at providing coarse motion commands for complex, long-horizon tasks. Therefore, the optimal solution might be a combination of both approaches. In this paper, we introduce DexterityGen (DexGen), which uses RL to pretrain large-scale dexterous motion primitives, such as in-hand rotation or translation. We then leverage this learned dataset to train a dexterous foundational controller. In the real world, we use human teleoperation as a prompt to the controller to produce highly dexterous behavior. We evaluate the effectiveness of DexGen in both simulation and real world, demonstrating that it is a general-purpose controller that can realize input dexterous manipulation commands and significantly improves stability by 10-100x measured as duration of holding objects across diverse tasks. Notably, with DexGen we demonstrate unprecedented dexterous skills including diverse object reorientation and dexterous tool use such as pen, syringe, and screwdriver for the first time. 

**Abstract (ZH)**: 教授机器人灵巧操作技能，例如工具使用，是一项重大挑战。当前的方法可以大致分为两种策略：人类远程操作（模仿学习）和从仿真到现实的强化学习。第一种方法难度较大，因为人类在没有触觉反馈的情况下，很难在不同的实体上产生安全而灵巧的动作。而基于第二种RL的方法则面临着领域差异的问题，并且在复杂任务中需要高度任务特定的奖励工程。我们的关键见解是，强化学习在学习低级运动基元方面非常有效，而人类则在提供复杂、长期任务的粗略运动指令方面表现出色。因此，最佳解决方案可能是这两种方法的结合。在这项研究中，我们引入了DexterityGen（DexGen），使用强化学习预训练大规模灵巧运动基元，如手掌握持旋转或平移。然后，我们利用这个学习的数据集来训练一个基础的手灵巧控制器。在现实世界中，我们使用人类远程操作来作为控制器的提示，使其产生高度灵巧的行为。我们在仿真和现实世界中评估了DexGen的有效性，证明它是一个通用控制器，可以实现输入的灵巧操作指令，并且在多种任务下通过对象抓握持续时间的提高，提升了稳定性的10到100倍。特别是，通过DexGen，我们首次展示了前所未有的灵巧技能，包括各种物体重定向和灵巧工具使用，例如笔、针筒和螺丝刀等。 

---
# Learning Real-World Action-Video Dynamics with Heterogeneous Masked Autoregression 

**Title (ZH)**: 使用异质遮蔽自回归学习真实世界动作视频动力学 

**Authors**: Lirui Wang, Kevin Zhao, Chaoqi Liu, Xinlei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.04296)  

**Abstract**: We propose Heterogeneous Masked Autoregression (HMA) for modeling action-video dynamics to generate high-quality data and evaluation in scaling robot learning. Building interactive video world models and policies for robotics is difficult due to the challenge of handling diverse settings while maintaining computational efficiency to run in real time. HMA uses heterogeneous pre-training from observations and action sequences across different robotic embodiments, domains, and tasks. HMA uses masked autoregression to generate quantized or soft tokens for video predictions. \ourshort achieves better visual fidelity and controllability than the previous robotic video generation models with 15 times faster speed in the real world. After post-training, this model can be used as a video simulator from low-level action inputs for evaluating policies and generating synthetic data. See this link this https URL for more information. 

**Abstract (ZH)**: 以下是经过学术规范翻译的内容：

我们提出了异质掩码自回归(Heterogeneous Masked Autoregression, HMA)方法，用于建模动作-视频动态，以生成高质量的数据和评估，从而在缩放机器人学习时提供更有效的解决方案。由于在保持实时运行效率的同时处理多种环境设定具有挑战性，因此为机器人构建交互式视频世界模型和策略颇具难度。HMA 通过在不同机器人实体、领域和任务的观测和动作序列上进行异质预训练，来克服这一挑战。HMA 利用掩码自回归生成视频预测中的量化或软标记。我们的模型在实际世界中比之前的机器人视频生成模型具有15倍的速度优势，并且在视觉保真度和可控性方面表现更佳。经过后训练后，该模型可以作为从低级动作输入中生成视频模拟器，用于评估策略并生成合成数据。有关更多信息，请访问以下链接：[请提供链接] 

---
# Compliant Beaded-String Jamming For Variable Stiffness Anthropomorphic Fingers 

**Title (ZH)**: 符合学术规范的翻译如下：

可变形珠状绳索夹紧技术用于仿人手指可变 stiffness 设计 

**Authors**: Maximilian Westermann, Marco Pontin, Leone Costi, Alessandro Albini, Perla Maiolino  

**Link**: [PDF](https://arxiv.org/pdf/2502.04190)  

**Abstract**: Achieving human-like dexterity in robotic grippers remains an open challenge, particularly in ensuring robust manipulation in uncertain environments. Soft robotic hands try to address this by leveraging passive compliance, a characteristic that is crucial to the adaptability of the human hand, to achieve more robust manipulation while reducing reliance on high-resolution sensing and complex control. Further improvements in terms of precision and postural stability in manipulation tasks are achieved through the integration of variable stiffness mechanisms, but these tend to lack residual compliance, be bulky and have slow response times. To address these limitations, this work introduces a Compliant Joint Jamming mechanism for anthropomorphic fingers that exhibits passive residual compliance and adjustable stiffness, while achieving a range of motion in line with that of human interphalangeal joints. The stiffness range provided by the mechanism is controllable from 0.48 Nm/rad to 1.95 Nm/rad (a 4x increase). Repeatability, hysteresis and stiffness were also characterized as a function of the jamming force. To demonstrate the importance of the passive residual compliance afforded by the proposed system, a peg-in-hole task was conducted, which showed a 60% higher success rate for a gripper integrating our joint design when compared to a rigid one. 

**Abstract (ZH)**: 在机器人抓手上实现类似人类的灵巧性仍然是一个开放的挑战，特别是在确保在不确定环境中进行稳健操作方面。软体手通过利用被动顺应性这一有助于人类手部适应性的特性，尝试解决这一问题，从而实现更为稳健的操作，同时减少对高分辨率传感和复杂控制的依赖。通过集成可变刚度机制，进一步提高了操作任务中的精度和姿势稳定性，但这些机制往往缺乏剩余顺应性，体积较大且响应时间较长。为解决这些局限性，本研究引入了一种适用于类人手指的顺应关节卡紧机制，该机制展现出被动剩余顺应性和可调刚度，并能达到与人类指间关节相当的运动范围。该机制提供的刚度范围从0.48 Nm/弧度到1.95 Nm/弧度（提高了4倍）。还对卡紧力与重复性、滞回现象和刚度之间的关系进行了描述。为展示所提系统提供的被动剩余顺应性的重要性，进行了一个孔中插入钉子的任务，结果显示，集成我们关节设计的夹爪的成功率为60%高于刚性夹爪的夹爪。 

---
# Dense Fixed-Wing Swarming using Receding-Horizon NMPC 

**Title (ZH)**: 使用预见 horizon NMPC 的密集固定翼无人机编队飞行 

**Authors**: Varun Madabushi, Yocheved Kopel, Adam Polevoy, Joseph Moore  

**Link**: [PDF](https://arxiv.org/pdf/2502.04174)  

**Abstract**: In this paper, we present an approach for controlling a team of agile fixed-wing aerial vehicles in close proximity to one another. Our approach relies on receding-horizon nonlinear model predictive control (NMPC) to plan maneuvers across an expanded flight envelope to enable inter-agent collision avoidance. To facilitate robust collision avoidance and characterize the likelihood of inter-agent collisions, we compute a statistical bound on the probability of the system leaving a tube around the planned nominal trajectory. Finally, we propose a metric for evaluating highly dynamic swarms and use this metric to evaluate our approach. We successfully demonstrated our approach through both simulation and hardware experiments, and to our knowledge, this the first time close-quarters swarming has been achieved with physical aerobatic fixed-wing vehicles. 

**Abstract (ZH)**: 在本文中，我们提出了一种控制紧密靠近飞行的敏捷固定翼飞行器编队的方法。该方法依赖于动态规划非线性模型预测控制（NMPC），以扩展飞行包线来进行机动控制，从而实现-agent间的碰撞避免。为了实现稳健的碰撞避免并量化-agent间碰撞的可能性，我们计算了系统偏离计划轨迹管的概率边界。最后，我们提出了一种用于评估高度动态集群的指标，并使用该指标评估了我们的方法。我们通过仿真和硬件实验成功地验证了该方法，并且据我们所知，这是首次使用实际的空中表演固定翼飞行器实现紧密空间内的集群飞行。 

---
# From Configuration-Space Clearance to Feature-Space Margin: Sample Complexity in Learning-Based Collision Detection 

**Title (ZH)**: 从配置空间 clearance 到特征空间 margin：基于学习的碰撞检测中的样本复杂度 

**Authors**: Sapir Tubul, Aviv Tamar, Kiril Solovey, Oren Salzman  

**Link**: [PDF](https://arxiv.org/pdf/2502.04170)  

**Abstract**: Motion planning is a central challenge in robotics, with learning-based approaches gaining significant attention in recent years. Our work focuses on a specific aspect of these approaches: using machine-learning techniques, particularly Support Vector Machines (SVM), to evaluate whether robot configurations are collision free, an operation termed ``collision detection''. Despite the growing popularity of these methods, there is a lack of theory supporting their efficiency and prediction accuracy. This is in stark contrast to the rich theoretical results of machine-learning methods in general and of SVMs in particular. Our work bridges this gap by analyzing the sample complexity of an SVM classifier for learning-based collision detection in motion planning. We bound the number of samples needed to achieve a specified accuracy at a given confidence level. This result is stated in terms relevant to robot motion-planning such as the system's clearance. Building on these theoretical results, we propose a collision-detection algorithm that can also provide statistical guarantees on the algorithm's error in classifying robot configurations as collision-free or not. 

**Abstract (ZH)**: 运动规划是机器人技术中的一个核心挑战，近年来基于学习的方法受到了广泛关注。我们的研究集中在这些方法的一个特定方面：利用机器学习技术，尤其是支持向量机（SVM），评估机器人配置是否会发生碰撞，这一操作被称为“碰撞检测”。尽管这些方法的 popularity 不断增长，但仍然缺乏支持其效率和预测准确性的理论基础。这与机器学习方法以及 SVM 的丰富理论结果形成了鲜明对比。我们的研究通过分析 SVM 分类器在基于学习的碰撞检测中的样本复杂性，来弥补这一差距。我们界定了在给定置信水平下达到特定准确性的样本数量。这一结果以机器人运动规划相关的系统裕度等术语表述。基于这些理论结果，我们提出了一个碰撞检测算法，该算法还能够对算法分类机器人配置为碰撞-free 或非碰撞-free 时的误差提供统计上的保证。 

---
# Safe Quadrotor Navigation using Composite Control Barrier Functions 

**Title (ZH)**: 使用复合控制屏障函数实现安全四旋翼导航 

**Authors**: Marvin Harms, Martin Jacquet, Kostas Alexis  

**Link**: [PDF](https://arxiv.org/pdf/2502.04101)  

**Abstract**: This paper introduces a safety filter to ensure collision avoidance for multirotor aerial robots. The proposed formalism leverages a single Composite Control Barrier Function from all position constraints acting on a third-order nonlinear representation of the robot's dynamics. We analyze the recursive feasibility of the safety filter under the composite constraint and demonstrate that the infeasible set is negligible. The proposed method allows computational scalability against thousands of constraints and, thus, complex scenes with numerous obstacles. We experimentally demonstrate its ability to guarantee the safety of a quadrotor with an onboard LiDAR, operating in both indoor and outdoor cluttered environments against both naive and adversarial nominal policies. 

**Abstract (ZH)**: 本文介绍了安全滤波器，以确保多旋翼飞行器在运动过程中的碰撞避免。所提出的表述方式利用了一个综合控制屏障函数，该函数源自作用于机器人动态三阶非线性表示的所有位置约束。我们分析了在综合约束下安全滤波器的递归可行性，并证明不可行集可以忽略不计。所提出的方法在面对成千上万条约束时仍具有计算扩展性，从而适用于具有众多障碍物的复杂场景。实验结果表明，该方法能够确保装备有机载激光雷达（LiDAR）的四旋翼无人机在室内和室外有障碍物的环境下安全运行，并能够抵抗简单和恶意的名义策略。 

---
# Soft and Highly-Integrated Optical Fiber Bending Sensors for Proprioception in Multi-Material 3D Printed Fingers 

**Title (ZH)**: 用于多材料3D打印手指本体感受性的柔软且高度集成的光纤弯曲传感器 

**Authors**: Ellis Capp, Marco Pontin, Peter Walters, Perla Maiolino  

**Link**: [PDF](https://arxiv.org/pdf/2502.04094)  

**Abstract**: Accurate shape sensing, only achievable through distributed proprioception, is a key requirement for closed-loop control of soft robots. Low-cost power efficient optoelectronic sensors manufactured from flexible materials represent a natural choice as they can cope with the large deformations of soft robots without loss of performance. However, existing integration approaches are cumbersome and require manual steps and complex assembly. We propose a semi-automated printing process where plastic optical fibers are embedded with readout electronics in 3D printed flexures. The fibers become locked in place and the readout electronics remain optically coupled to them while the flexures undergo large bending deformations, creating a repeatable, monolithically manufactured bending transducer with only 10 minutes required in total for the manual embedding steps. We demonstrate the process by manufacturing multi-material 3D printed fingers and extensively evaluating the performance of each proprioceptive joint. The sensors achieve 70% linearity and 4.81° RMS error on average. Furthermore, the distributed architecture allows for maintaining an average fingertip position estimation accuracy of 12 mm in the presence of external static forces. To demonstrate the potential of the distributed sensor architecture in robotics applications, we build a data-driven model independent of actuation feedback to detect contact with objects in the environment. 

**Abstract (ZH)**: 精确的形状感知是软机器人闭环控制的关键要求，而这种精确感知仅可通过分布式的本体感受实现。低成本、高效的光学电子传感器利用柔性材料制造，成为一种自然的选择，它们可以在不影响性能的情况下应对软机器人的大形变。然而，现有的集成方法繁琐且需要手动步骤和复杂的组装过程。我们提出了一种半自动化打印过程，其中将塑料光学纤维及其读出电子学嵌入3D打印的柔性结构中。当这些结构经历大幅度弯曲变形时，光纤被固定在适当位置，而读出电子学仍然保持光学耦合状态，从而实现了重复性极佳的集成弯曲传感器，仅需10分钟的手动嵌入步骤即可完成。我们通过制造多材料3D打印手指并广泛评估每个本体感受关节的性能来演示此过程。传感器展现出70%的线性度和4.81°的均方根误差。此外，分布式架构使传感器能够在存在外部静载荷的情况下，保持指尖位置估计的平均精度为12毫米。为了展示分布式传感架构在机器人应用中的潜力，我们构建了一个不依赖于执行反馈的数据驱动模型，用于检测环境中的接触物体。 

---
# Malleable Robots 

**Title (ZH)**: 可塑机器人 

**Authors**: Angus B. Clark, Xinran Wang, Alex Ranne, Nicolas Rojas  

**Link**: [PDF](https://arxiv.org/pdf/2502.04012)  

**Abstract**: This chapter is about the fundamentals of fabrication, control, and human-robot interaction of a new type of collaborative robotic manipulators, called malleable robots, which are based on adjustable architectures of varying stiffness for achieving high dexterity with lower mobility arms. Collaborative robots, or cobots, commonly integrate six or more degrees of freedom (DOF) in a serial arm in order to allow positioning in constrained spaces and adaptability across tasks. Increasing the dexterity of robotic arms has been indeed traditionally accomplished by increasing the number of degrees of freedom of the system; however, once a robotic task has been established (e.g., a pick-and-place operation), the motion of the end-effector can be normally achieved using less than 6-DOF (i.e., lower mobility). The aim of malleable robots is to close the technological gap that separates current cobots from achieving flexible, accessible manufacturing automation with a reduced number of actuators. 

**Abstract (ZH)**: 本章讨论了一种新型协作机器人 manipulator——可调柔顺机器人（malleable robots）的基础原理、控制方法及其与人类的交互技术。这种机器人基于不同刚度的可调架构，以实现较低的活动臂长和较高的灵活性。协作机器人（即协作机器人或cobots）通常通过在其串行臂中集成六自由度（DOF）或以上，以便在受限空间中进行定位并适应不同任务。确实，传统上增加机器人臂的灵活性是通过增加系统的自由度来实现的；然而，一旦确定了机器人任务（例如取放操作），末端执行器的运动通常可以使用少于6-DOF（即较低的活动度）来实现。可调柔顺机器人的目标是弥合当前协作机器人与通过减少执行器数量实现灵活、易于接近的制造自动化之间的技术差距。 

---
# Bilevel Multi-Armed Bandit-Based Hierarchical Reinforcement Learning for Interaction-Aware Self-Driving at Unsignalized Intersections 

**Title (ZH)**: 基于多层次拉姆齐特老虎机的 bilevel 递归强化学习方法：应用于无信号交叉口的交互感知自动驾驶 

**Authors**: Zengqi Peng, Yubin Wang, Lei Zheng, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2502.03960)  

**Abstract**: In this work, we present BiM-ACPPO, a bilevel multi-armed bandit-based hierarchical reinforcement learning framework for interaction-aware decision-making and planning at unsignalized intersections. Essentially, it proactively takes the uncertainties associated with surrounding vehicles (SVs) into consideration, which encompass those stemming from the driver's intention, interactive behaviors, and the varying number of SVs. Intermediate decision variables are introduced to enable the high-level RL policy to provide an interaction-aware reference, for guiding low-level model predictive control (MPC) and further enhancing the generalization ability of the proposed framework. By leveraging the structured nature of self-driving at unsignalized intersections, the training problem of the RL policy is modeled as a bilevel curriculum learning task, which is addressed by the proposed Exp3.S-based BiMAB algorithm. It is noteworthy that the training curricula are dynamically adjusted, thereby facilitating the sample efficiency of the RL training process. Comparative experiments are conducted in the high-fidelity CARLA simulator, and the results indicate that our approach achieves superior performance compared to all baseline methods. Furthermore, experimental results in two new urban driving scenarios clearly demonstrate the commendable generalization performance of the proposed method. 

**Abstract (ZH)**: 在本研究中，我们提出了一种基于 bilevel 多臂老虎机的分层强化学习框架 BiM-ACPPO，该框架用于处理无信号交叉路口的交互感知决策和规划。该框架能够主动考虑周围车辆（SVs）带来的不确定性，这些不确定性包括驾驶员意图、互动行为以及 SVs 的数量变化。引入了中间决策变量，使高层的 RL 策略能够提供一个交互感知的参考，指导低层级模型预测控制（MPC），并进一步提升所提出框架的泛化能力。通过利用无信号交叉路口自动驾驶的结构化特性，RL 策略的训练问题被建模为一个分层的学习任务，使用所提出的基于 Exp3.S 的 BiMAB 算法来解决。值得注意的是，训练课程是动态调整的，从而提高了 RL 训练过程的样本效率。在高保真 CARLA 仿真器中进行的比较实验表明，我们的方法在所有基线方法中表现出更优的性能。此外，在两个新的城市驾驶场景中的实验结果进一步证明了所提出方法的出色泛化性能。 

---
# Adaptation of Task Goal States from Prior Knowledge 

**Title (ZH)**: 从前验知识适应任务目标状态 

**Authors**: Andrei Costinescu, Darius Burschka  

**Link**: [PDF](https://arxiv.org/pdf/2502.03918)  

**Abstract**: This paper presents a framework to define a task with freedom and variability in its goal state. A robot could use this to observe the execution of a task and target a different goal from the observed one; a goal that is still compatible with the task description but would be easier for the robot to execute. We define the model of an environment state and an environment variation, and present experiments on how to interactively create the variation from a single task demonstration and how to use this variation to create an execution plan for bringing any environment into the goal state. 

**Abstract (ZH)**: 本文提出了一种框架，用于定义具有自由度和变化性目标状态的任务。机器人可以使用此框架观察任务的执行过程，并设定与所观察的目标不同的新目标；该新目标与任务描述兼容，但对机器人执行更为简便。本文定义了环境状态模型和环境变化模型，并呈现了如何从单一任务演示中互动地创建变化以及如何利用这些变化来制定将任何环境带入目标状态的执行计划。 

---
# A Flexible FBG-Based Contact Force Sensor for Robotic Gripping Systems 

**Title (ZH)**: 基于光纤布拉格光栅的可调接触力传感器及其在机器人抓持系统中的应用 

**Authors**: Wenjie Lai, Huu Duoc Nguyen, Jiajun Liu, Xingyu Chen, Soo Jay Phee  

**Link**: [PDF](https://arxiv.org/pdf/2502.03914)  

**Abstract**: Soft robotic grippers demonstrate great potential for gently and safely handling objects; however, their full potential for executing precise and secure grasping has been limited by the lack of integrated sensors, leading to problems such as slippage and excessive force exertion. To address this challenge, we present a small and highly sensitive Fiber Bragg Grating-based force sensor designed for accurate contact force measurement. The flexible force sensor comprises a 3D-printed TPU casing with a small bump and uvula structure, a dual FBG array, and a protective tube. A series of tests have been conducted to evaluate the effectiveness of the proposed force sensor, including force calibration, repeatability test, hysteresis study, force measurement comparison, and temperature calibration and compensation tests. The results demonstrated good repeatability, with a force measurement range of 4.69 N, a high sensitivity of approximately 1169.04 pm/N, a root mean square error (RMSE) of 0.12 N, and a maximum hysteresis of 4.83%. When compared to a commercial load cell, the sensor showed a percentage error of 2.56% and an RMSE of 0.14 N. Besides, the proposed sensor validated its temperature compensation effectiveness, with a force RMSE of 0.01 N over a temperature change of 11 Celsius degree. The sensor was integrated with a soft grow-and-twine gripper to monitor interaction forces between different objects and the robotic gripper. Closed-loop force control was applied during automated pick-and-place tasks and significantly improved gripping stability, as demonstrated in tests. This force sensor can be used across manufacturing, agriculture, healthcare (like prosthetic hands), logistics, and packaging, to provide situation awareness and higher operational efficiency. 

**Abstract (ZH)**: 软体夹爪展示了轻柔且安全处理物体的巨大潜力，但它们在执行精确和安全夹持方面的全部潜力受到了集成传感器缺乏的限制，导致诸如打滑和施加过大力量等问题。为解决这一挑战，我们提出了一种小型且高灵敏度的基于光纤布拉格光栅（FBG）的力传感器，用于精确接触力测量。该柔性力传感器包括一个带有小突起和悬雍垂结构的3D打印TPU外壳、双FBG阵列和一个保护管。一系列测试已被用于评估所提出的力传感器的有效性，包括力校准、重复性测试、迟滞研究、力测量比较以及温度校准和补偿测试。结果表明，该传感器具有良好的重复性，其力测量范围为4.69牛顿，灵敏度约为1169.04 pm/N，均方根误差（RMSE）为0.12牛顿，最大迟滞性为4.83%。与商用载荷细胞相比，该传感器的百分比误差为2.56%，RMSE为0.14牛顿。此外，所提出的传感器验证了其温度补偿效果，在温度变化11摄氏度的情况下，力的RMSE为0.01牛顿。该传感器已被集成到一个软体生长和缠绕夹爪中，以监测不同物体与机器人夹爪之间的相互作用力。在自动化拾放任务中应用闭环力控制，并显著提高了夹持稳定性，测试结果表明了这一点。该力传感器在制造、农业、医疗保健（如假肢手）、物流和包装等领域中具有广泛的应用前景，可提供情况感知并提高操作效率。 

---
# Dynamic Rank Adjustment in Diffusion Policies for Efficient and Flexible Training 

**Title (ZH)**: 动态调整扩散政策中的排名，在高效灵活训练中的应用 

**Authors**: Xiatao Sun, Shuo Yang, Yinxing Chen, Francis Fan, Yiyan, Liang, Daniel Rakita  

**Link**: [PDF](https://arxiv.org/pdf/2502.03822)  

**Abstract**: Diffusion policies trained via offline behavioral cloning have recently gained traction in robotic motion generation. While effective, these policies typically require a large number of trainable parameters. This model size affords powerful representations but also incurs high computational cost during training. Ideally, it would be beneficial to dynamically adjust the trainable portion as needed, balancing representational power with computational efficiency. For example, while overparameterization enables diffusion policies to capture complex robotic behaviors via offline behavioral cloning, the increased computational demand makes online interactive imitation learning impractical due to longer training time. To address this challenge, we present a framework, called DRIFT, that uses the Singular Value Decomposition to enable dynamic rank adjustment during diffusion policy training. We implement and demonstrate the benefits of this framework in DRIFT-DAgger, an imitation learning algorithm that can seamlessly slide between an offline bootstrapping phase and an online interactive phase. We perform extensive experiments to better understand the proposed framework, and demonstrate that DRIFT-DAgger achieves improved sample efficiency and faster training with minimal impact on model performance. 

**Abstract (ZH)**: 通过离线行为克隆训练的扩散策略最近在机器人运动生成中引起了广泛关注。虽然这些策略非常有效，但它们通常需要大量的可训练参数。这种模型大小赋予了强大的表示能力，但也带来了在训练期间高昂的计算成本。理想情况下，可以根据需要动态调整可训练部分，平衡表示能力和计算效率。例如，虽然过参数化使扩散策略能够通过离线行为克隆捕捉复杂的机器人行为，但增加的计算需求使得在线交互式模仿学习变得不实用，因为训练时间延长。为了解决这一挑战，我们提出了一种称为DRIFT的框架，该框架利用奇异值分解，在扩散策略训练期间实现动态秩调整。我们通过DRIFT-DAgger，一个可以在离线自我强化阶段和在线交互式阶段之间平滑过渡的模仿学习算法来实现和展示这一框架的好处。我们进行了广泛的实验，以更好地了解所提出的框架，并证明DRIFT-DAgger实现了更高的样本效率和更快的训练速度，且对模型性能的影响最小。 

---
# Large Language Models for Multi-Robot Systems: A Survey 

**Title (ZH)**: 大型语言模型在多机器人系统中的应用：一个综述 

**Authors**: Peihan Li, Zijian An, Shams Abrar, Lifeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.03814)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has opened new possibilities in Multi-Robot Systems (MRS), enabling enhanced communication, task planning, and human-robot interaction. Unlike traditional single-robot and multi-agent systems, MRS poses unique challenges, including coordination, scalability, and real-world adaptability. This survey provides the first comprehensive exploration of LLM integration into MRS. It systematically categorizes their applications across high-level task allocation, mid-level motion planning, low-level action generation, and human intervention. We highlight key applications in diverse domains, such as household robotics, construction, formation control, target tracking, and robot games, showcasing the versatility and transformative potential of LLMs in MRS. Furthermore, we examine the challenges that limit adapting LLMs in MRS, including mathematical reasoning limitations, hallucination, latency issues, and the need for robust benchmarking systems. Finally, we outline opportunities for future research, emphasizing advancements in fine-tuning, reasoning techniques, and task-specific models. This survey aims to guide researchers in the intelligence and real-world deployment of MRS powered by LLMs. Based on the fast-evolving nature of research in the field, we keep updating the papers in the open-source Github repository. 

**Abstract (ZH)**: 大型语言模型（LLMs）的快速进步为多机器人系统（MRS）开启了新的可能性，使其在通信、任务规划和人机交互方面得到了增强。与传统的单机器人系统和多智能体系统不同，MRS提出了独特的挑战，包括协调、可扩展性和现实世界的适应性。本文综述了LLMs在MRS中集成的第一项全面探索。文章系统地分类了LLMs在高层任务分配、中层运动规划、低层动作生成以及人类干预方面的应用。我们强调了多样化领域的关键应用，如家庭机器人、建筑施工、队形控制、目标跟踪和机器人游戏，展示了LLMs在MRS中的多样性和变革潜力。此外，我们探讨了限制LLMs适应MRS的挑战，包括数学推理限制、幻觉、延迟问题以及需要稳健的基准测试系统。最后，我们概述了未来研究的机会，强调了微调方法、推理技术以及任务特定模型的进步。本文综述旨在指导研究人员利用LLMs增强MRS的智能化和实际部署。鉴于领域研究的快速演进，我们会在开源Github仓库中不断更新论文。 

---
# Action-Free Reasoning for Policy Generalization 

**Title (ZH)**: 动作无关推理在策略泛化中的应用 

**Authors**: Jaden Clark, Suvir Mirchandani, Dorsa Sadigh, Suneel Belkhale  

**Link**: [PDF](https://arxiv.org/pdf/2502.03729)  

**Abstract**: End-to-end imitation learning offers a promising approach for training robot policies. However, generalizing to new settings remains a significant challenge. Although large-scale robot demonstration datasets have shown potential for inducing generalization, they are resource-intensive to scale. In contrast, human video data is abundant and diverse, presenting an attractive alternative. Yet, these human-video datasets lack action labels, complicating their use in imitation learning. Existing methods attempt to extract grounded action representations (e.g., hand poses), but resulting policies struggle to bridge the embodiment gap between human and robot actions. We propose an alternative approach: leveraging language-based reasoning from human videos-essential for guiding robot actions-to train generalizable robot policies. Building on recent advances in reasoning-based policy architectures, we introduce Reasoning through Action-free Data (RAD). RAD learns from both robot demonstration data (with reasoning and action labels) and action-free human video data (with only reasoning labels). The robot data teaches the model to map reasoning to low-level actions, while the action-free data enhances reasoning capabilities. Additionally, we will release a new dataset of 3,377 human-hand demonstrations with reasoning annotations compatible with the Bridge V2 benchmark and aimed at facilitating future research on reasoning-driven robot learning. Our experiments show that RAD enables effective transfer across the embodiment gap, allowing robots to perform tasks seen only in action-free data. Furthermore, scaling up action-free reasoning data significantly improves policy performance and generalization to novel tasks. These results highlight the promise of reasoning-driven learning from action-free datasets for advancing generalizable robot control. Project page: this https URL 

**Abstract (ZH)**: 端到端的模仿学习为训练机器人策略提供了一种颇具前景的方法。然而，将策略应用于新的环境仍然是一个显著的挑战。尽管大规模的机器人演示数据集展示了潜在的泛化能力，但扩大量级需要大量的资源。相比之下，人类视频数据丰富多样，提供了一个有吸引力的替代方案。然而，这些人类视频数据缺少行动标签，给模仿学习的应用带来了复杂性。现有的方法试图提取基于语言的动作表示（如手部姿态），但这些策略在弥合人类和机器人行动之间的身体差异方面表现出困难。我们提出了一种替代方法：利用人类视频中的语言推理来引导机器人动作，从而训练可泛化的机器人策略。基于近期在基于推理的策略架构方面的进展，我们引入了“基于行动的推理数据”（RAD）。RAD 从包含推理和行动标签的机器人演示数据以及仅包含推理标签的行动空白人类视频数据中学习。机器人数据教会模型将推理映射到低级行动，而行动空白数据则增强了解释能力。此外，我们将发布一个包含3,377个人手演示的新数据集，这些演示数据适用于Bridge V2基准，并旨在促进基于推理的机器人学习的未来研究。我们的实验表明，RAD 可以有效地跨越身体差异进行迁移学习，使机器人能够执行仅在行动空白数据中见过的任务。此外，增加行动空白数据的推理规模显著提升了策略性能并提高了对新任务的泛化能力。这些结果凸显了从行动空白数据中基于推理学习在推动机器人控制的泛化方面具有前景。项目页面：this https URL 

---
# Efficiently Generating Expressive Quadruped Behaviors via Language-Guided Preference Learning 

**Title (ZH)**: 通过语言引导的偏好学习高效生成丰富的四足运动行为 

**Authors**: Jaden Clark, Joey Hejna, Dorsa Sadigh  

**Link**: [PDF](https://arxiv.org/pdf/2502.03717)  

**Abstract**: Expressive robotic behavior is essential for the widespread acceptance of robots in social environments. Recent advancements in learned legged locomotion controllers have enabled more dynamic and versatile robot behaviors. However, determining the optimal behavior for interactions with different users across varied scenarios remains a challenge. Current methods either rely on natural language input, which is efficient but low-resolution, or learn from human preferences, which, although high-resolution, is sample inefficient. This paper introduces a novel approach that leverages priors generated by pre-trained LLMs alongside the precision of preference learning. Our method, termed Language-Guided Preference Learning (LGPL), uses LLMs to generate initial behavior samples, which are then refined through preference-based feedback to learn behaviors that closely align with human expectations. Our core insight is that LLMs can guide the sampling process for preference learning, leading to a substantial improvement in sample efficiency. We demonstrate that LGPL can quickly learn accurate and expressive behaviors with as few as four queries, outperforming both purely language-parameterized models and traditional preference learning approaches. Website with videos: this https URL 

**Abstract (ZH)**: 表达性的人形机器人行为对于机器人在社会环境中被广泛接受至关重要。近年来，学习型腿部运动控制器的进步使得机器人行为更加动态和多变。然而，确定不同用户在各种场景下的最佳行为仍然是一个挑战。目前的方法要么依赖自然语言输入，这虽然高效但分辨率较低；要么从人类偏好中学习，尽管分辨率较高，但样本效率较低。本文提出了一种新的方法，利用预训练的LLM生成的先验知识与偏好学习的精确性相结合。我们的方法称为语言引导的偏好学习（LGPL），使用LLM生成初始行为样本，并通过基于偏好的反馈进一步完善这些样本，从而学习与人类期望高度一致的行为。我们核心的洞察是，LLM可以指导偏好学习的采样过程，从而显著提高样本效率。我们证明，LGPL可以在最少四次查询的情况下快速学习出准确且具有表现力的行为，并且在性能上优于纯语言参数化模型和传统的偏好学习方法。视频网站：https://this-url 

---
# Reduce Lap Time for Autonomous Racing with Curvature-Integrated MPCC Local Trajectory Planning Method 

**Title (ZH)**: 使用曲率整合的MPCC局部轨迹规划方法减少自主赛车的过弯时间 

**Authors**: Zhouheng Li, Lei Xie, Cheng Hu, Hongye Su  

**Link**: [PDF](https://arxiv.org/pdf/2502.03695)  

**Abstract**: The widespread application of autonomous driving technology has significantly advanced the field of autonomous racing. Model Predictive Contouring Control (MPCC) is a highly effective local trajectory planning method for autonomous racing. However, the traditional MPCC method struggles with racetracks that have significant curvature changes, limiting the performance of the vehicle during autonomous racing. To address this issue, we propose a curvature-integrated MPCC (CiMPCC) local trajectory planning method for autonomous racing. This method optimizes the velocity of the local trajectory based on the curvature of the racetrack centerline. The specific implementation involves mapping the curvature of the racetrack centerline to a reference velocity profile, which is then incorporated into the cost function for optimizing the velocity of the local trajectory. This reference velocity profile is created by normalizing and mapping the curvature of the racetrack centerline, thereby ensuring efficient and performance-oriented local trajectory planning in racetracks with significant curvature. The proposed CiMPCC method has been experimented on a self-built 1:10 scale F1TENTH racing vehicle deployed with ROS platform. The experimental results demonstrate that the proposed method achieves outstanding results on a challenging racetrack with sharp curvature, improving the overall lap time by 11.4%-12.5% compared to other autonomous racing trajectory planning methods. Our code is available at this https URL. 

**Abstract (ZH)**: 自动驾驶技术的广泛应用显著推动了自主赛车领域的进步。模型预测轮廓控制（Model Predictive Contouring Control, MPCC）是一种在自主赛车中非常有效的局部轨迹规划方法。然而，传统的MPCC方法在面对具有显著曲率变化的赛道时表现不佳，限制了自主赛车的性能。为了解决这一问题，我们提出了一种曲率整合的MPCC（Curvature-integrated MPCC, CiMPCC）局部轨迹规划方法，适用于自主赛车。该方法根据赛道中心线的曲率优化局部轨迹的速度。具体实施中，通过将赛道中心线的曲率映射到一个参考速度轮廓，然后将该参考速度轮廓整合到成本函数中，以优化局部轨迹的速度。参考速度轮廓通过标准化和映射赛道中心线的曲率生成，从而确保在具有显著曲率变化的赛道上实现高效且性能导向的局部轨迹规划。我们已经在ROS平台上构建的自用1:10比例尺F1TENTH赛车上对提出的CiMPCC方法进行了实验。实验结果表明，该方法在具有尖锐曲率的挑战性赛道上取得了出色的表现，与其它自主赛车轨迹规划方法相比，整体圈速提高了11.4%-12.5%。我们的代码可以在以下链接访问：[在此处链接]。 

---
# Anytime Planning for End-Effector Trajectory Tracking 

**Title (ZH)**: 任意时间规划用于末端执行器轨迹跟踪 

**Authors**: Yeping Wang, Michael Gleicher  

**Link**: [PDF](https://arxiv.org/pdf/2502.03676)  

**Abstract**: End-effector trajectory tracking algorithms find joint motions that drive robot manipulators to track reference trajectories. In practical scenarios, anytime algorithms are preferred for their ability to quickly generate initial motions and continuously refine them over time. In this paper, we present an algorithmic framework that adapts common graph-based trajectory tracking algorithms to be anytime and enhances their efficiency and effectiveness. Our key insight is to identify guide paths that approximately track the reference trajectory and strategically bias sampling toward the guide paths. We demonstrate the effectiveness of the proposed framework by restructuring two existing graph-based trajectory tracking algorithms and evaluating the updated algorithms in three experiments. 

**Abstract (ZH)**: 末端执行器轨迹跟踪算法通过驱动机器人 manipulator 的关节运动使其跟踪参考轨迹。在实际应用中，任意时间（anytime）算法因其能够快速生成初始运动并随时间不断优化受到青睐。本文介绍了一种算法框架，能够将常见的图基轨迹跟踪算法调整为任意时间类型，并提高了这些算法的效率和有效性。我们的关键见解是识别出能够大约跟踪参考轨迹的引导路径，并战略性地偏置采样向这些引导路径靠拢。通过重构两种现有的图基轨迹跟踪算法并在三个实验中评估更新后的算法，我们展示了该框架的有效性。 

---
# Discrete GCBF Proximal Policy Optimization for Multi-agent Safe Optimal Control 

**Title (ZH)**: 离散GCBF近端策略优化在多智能体安全最优控制中的应用 

**Authors**: Songyuan Zhang, Oswin So, Mitchell Black, Chuchu Fan  

**Link**: [PDF](https://arxiv.org/pdf/2502.03640)  

**Abstract**: Control policies that can achieve high task performance and satisfy safety constraints are desirable for any system, including multi-agent systems (MAS). One promising technique for ensuring the safety of MAS is distributed control barrier functions (CBF). However, it is difficult to design distributed CBF-based policies for MAS that can tackle unknown discrete-time dynamics, partial observability, changing neighborhoods, and input constraints, especially when a distributed high-performance nominal policy that can achieve the task is unavailable. To tackle these challenges, we propose DGPPO, a new framework that simultaneously learns both a discrete graph CBF which handles neighborhood changes and input constraints, and a distributed high-performance safe policy for MAS with unknown discrete-time dynamics. We empirically validate our claims on a suite of multi-agent tasks spanning three different simulation engines. The results suggest that, compared with existing methods, our DGPPO framework obtains policies that achieve high task performance (matching baselines that ignore the safety constraints), and high safety rates (matching the most conservative baselines), with a constant set of hyperparameters across all environments. 

**Abstract (ZH)**: 为了实现高任务性能并满足安全约束，任何系统，包括多Agent系统（MAS），都需要合适的控制策略。保证MAS安全的一种有前景的技术是分布式控制屏障函数（CBF）。然而，在设计能够处理未知离散时间动力学、部分可观测性、变化的邻居关系以及输入约束的分布式CBF基于策略时，存在诸多困难，尤其是在没有一种能够完成任务的分布式高性能名义策略的情况下更为困难。为解决这些问题，我们提出了一种新的框架DGPPO，该框架同时学习能够处理邻居关系变化和输入约束的离散图CBF，以及能够在未知离散时间动力学的情况下为MAS学习一种分布式高 performance 安全策略。我们在三个不同仿真引擎上的多Agent任务中进行了实证验证，结果表明，与现有的方法相比，我们的DGPPO框架能够生成既实现高任务性能（与忽略安全约束的基线相当）又具有高安全性（与最保守的基线相当）的策略，且在所有环境中使用的是相同的一组超参数。 

---
# Simultaneous Multi-Robot Motion Planning with Projected Diffusion Models 

**Title (ZH)**: 使用投影扩散模型的多机器人同时 motion 计划 

**Authors**: Jinhao Liang, Jacob K Christopher, Sven Koenig, Ferdinando Fioretto  

**Link**: [PDF](https://arxiv.org/pdf/2502.03607)  

**Abstract**: Recent advances in diffusion models hold significant potential in robotics, enabling the generation of diverse and smooth trajectories directly from raw representations of the environment. Despite this promise, applying diffusion models to motion planning remains challenging due to their difficulty in enforcing critical constraints, such as collision avoidance and kinematic feasibility. These limitations become even more pronounced in Multi-Robot Motion Planning (MRMP), where multiple robots must coordinate in shared spaces. To address this challenge, this work proposes Simultaneous MRMP Diffusion (SMD), a novel approach integrating constrained optimization into the diffusion sampling process to produce collision-free, kinematically feasible trajectories. Additionally, the paper introduces a comprehensive MRMP benchmark to evaluate trajectory planning algorithms across scenarios with varying robot densities, obstacle complexities, and motion constraints. Experimental results show SMD consistently outperforms classical and learning-based motion planners, achieving higher success rates and efficiency in complex multi-robot environments. 

**Abstract (ZH)**: 近年来，扩散模型在机器人领域取得了重要进展，能够直接从环境的原始表示中生成多样且平滑的轨迹。尽管具有这一潜力，但将扩散模型应用于运动规划仍然具有挑战性，因为它们在强制执行关键约束（如避障和运动学可行性）方面存在困难。这些限制在多机器人运动规划（MRMP）中表现得更为明显，在多机器人必须在共享空间中协调运动时尤为突出。为应对这一挑战，本文提出了一种名为Simultaneous MRMP Diffusion (SMD)的新方法，该方法将约束优化集成到扩散采样过程中，以生成无障碍且运动学可行的轨迹。此外，本文还引入了一个综合的MRMP基准，用于评估在不同机器人密度、障碍复杂性和运动约束条件下的轨迹规划算法。实验结果表明，SMD在复杂多机器人环境中的一致表现优于经典和基于学习的运动规划方法，表现出更高的成功率和效率。 

---
# SMART: Advancing Scalable Map Priors for Driving Topology Reasoning 

**Title (ZH)**: SMART：推动驾驶拓扑推理的可扩展地图先验技术 

**Authors**: Junjie Ye, David Paz, Hengyuan Zhang, Yuliang Guo, Xinyu Huang, Henrik I. Christensen, Yue Wang, Liu Ren  

**Link**: [PDF](https://arxiv.org/pdf/2502.04329)  

**Abstract**: Topology reasoning is crucial for autonomous driving as it enables comprehensive understanding of connectivity and relationships between lanes and traffic elements. While recent approaches have shown success in perceiving driving topology using vehicle-mounted sensors, their scalability is hindered by the reliance on training data captured by consistent sensor configurations. We identify that the key factor in scalable lane perception and topology reasoning is the elimination of this sensor-dependent feature. To address this, we propose SMART, a scalable solution that leverages easily available standard-definition (SD) and satellite maps to learn a map prior model, supervised by large-scale geo-referenced high-definition (HD) maps independent of sensor settings. Attributed to scaled training, SMART alone achieves superior offline lane topology understanding using only SD and satellite inputs. Extensive experiments further demonstrate that SMART can be seamlessly integrated into any online topology reasoning methods, yielding significant improvements of up to 28% on the OpenLane-V2 benchmark. 

**Abstract (ZH)**: 自驾车领域中的拓扑推理对于实现全面理解和感知车道及交通元素之间的连接性和关系至关重要。虽然最近的方法已经证明了利用车载传感器感知驾驶拓扑的成功，但它们的扩展性受到依赖于一致传感器配置采集的训练数据的限制。我们发现，可扩展的车道感知和拓扑推理的关键因素是消除这种依赖传感器特征的方法。为了解决这一问题，我们提出了SMART（可扩展的多源地图信息推理模型），它利用易于获取的标准分辨率（SD）地图和卫星地图来学习先验地图模型，该模型通过大规模地理参考的高分辨率（HD）地图进行监督，而这些高分辨率地图与传感器配置无关。得益于扩展训练，仅使用SD和卫星地图输入，SMART即可实现优越的离线车道拓扑理解。进一步的实验表明，SMART可以无缝集成到任何在线拓扑推理方法中，在OpenLane-V2基准测试中可实现高达28%的显著性能提升。 

---
# Making Sense of Touch: Unsupervised Shapelet Learning in Bag-of-words Sense 

**Title (ZH)**: 理解触感：Bag-of-words表示下的无监督形状学习 

**Authors**: Zhicong Xian, Tabish Chaudhary, Jürgen Bock  

**Link**: [PDF](https://arxiv.org/pdf/2502.04167)  

**Abstract**: This paper introduces NN-STNE, a neural network using t-distributed stochastic neighbor embedding (t-SNE) as a hidden layer to reduce input dimensions by mapping long time-series data into shapelet membership probabilities. A Gaussian kernel-based mean square error preserves local data structure, while K-means initializes shapelet candidates due to the non-convex optimization challenge. Unlike existing methods, our approach uses t-SNE to address crowding in low-dimensional space and applies L1-norm regularization to optimize shapelet length. Evaluations on the UCR dataset and an electrical component manipulation task, like switching on, demonstrate improved clustering accuracy over state-of-the-art feature-learning methods in robotics. 

**Abstract (ZH)**: 本文介绍了NN-STNE，这是一种使用t分布随机邻嵌入（t-SNE）作为隐藏层的神经网络，通过将长时序数据映射为形状元成员概率来减少输入维度。基于高斯核的均方误差保留局部数据结构，而K-means由于非凸优化难题用于初始化形状元候选。与现有方法不同，我们的方法使用t-SNE解决低维空间中的数据拥挤问题，并应用L1范数正则化优化形状元长度。在UCR数据集和类似电气元件开关等任务上的评估表明，与机器人领域的先进特征学习方法相比，我们的方法能够实现更高的聚类准确性。 

---
# Enhancing people localisation in drone imagery for better crowd management by utilising every pixel in high-resolution images 

**Title (ZH)**: 通过利用高分辨率图像中的每个像素来提升无人机图像中的人群定位，以更好地进行人群管理 

**Authors**: Bartosz Ptak, Marek Kraft  

**Link**: [PDF](https://arxiv.org/pdf/2502.04014)  

**Abstract**: Accurate people localisation using drones is crucial for effective crowd management, not only during massive events and public gatherings but also for monitoring daily urban crowd flow. Traditional methods for tiny object localisation using high-resolution drone imagery often face limitations in precision and efficiency, primarily due to constraints in image scaling and sliding window techniques. To address these challenges, a novel approach dedicated to point-oriented object localisation is proposed. Along with this approach, the Pixel Distill module is introduced to enhance the processing of high-definition images by extracting spatial information from individual pixels at once. Additionally, a new dataset named UP-COUNT, tailored to contemporary drone applications, is shared. It addresses a wide range of challenges in drone imagery, such as simultaneous camera and object movement during the image acquisition process, pushing forward the capabilities of crowd management applications. A comprehensive evaluation of the proposed method on the proposed dataset and the commonly used DroneCrowd dataset demonstrates the superiority of our approach over existing methods and highlights its efficacy in drone-based crowd object localisation tasks. These improvements markedly increase the algorithm's applicability to operate in real-world scenarios, enabling more reliable localisation and counting of individuals in dynamic environments. 

**Abstract (ZH)**: 使用无人机进行精确人群定位对于有效的 crowd 管理至关重要，不仅在大规模活动和公共集会期间，还在日常城市人群流动的监测中也是如此。传统方法利用高分辨率无人机图像进行小目标定位往往在精度和效率方面存在限制，主要原因是图像缩放和滑动窗口技术的约束。为了解决这些问题，提出了一种专门针对点目标定位的新方法。同时，引入了 Pixel Distill 模块以通过一次从单个像素提取空间信息来增强高清图像的处理能力。此外，还共享了一个名为 UP-COUNT 的新数据集，专门针对现代无人机应用。该数据集解决了无人机图像中的各种挑战，如图像获取过程中相机和目标的同时移动，从而推动了人群管理应用的能力。在所提出的数据集和常用的 DroneCrowd 数据集上对所提出方法进行全面评估，表明其在现有方法中的优越性，并突出其在基于无人机的人群目标定位任务中的有效性。这些改进显著提高了算法在实际场景中的适用性，使其能够在动态环境中实现更可靠的个体定位和计数。 

---
# LeAP: Consistent multi-domain 3D labeling using Foundation Models 

**Title (ZH)**: LeAP：一致的多领域三维标记方法基于基础模型 

**Authors**: Simon Gebraad, Andras Palffy, Holger Caesar  

**Link**: [PDF](https://arxiv.org/pdf/2502.03901)  

**Abstract**: Availability of datasets is a strong driver for research on 3D semantic understanding, and whilst obtaining unlabeled 3D point cloud data is straightforward, manually annotating this data with semantic labels is time-consuming and costly. Recently, Vision Foundation Models (VFMs) enable open-set semantic segmentation on camera images, potentially aiding automatic labeling. However,VFMs for 3D data have been limited to adaptations of 2D models, which can introduce inconsistencies to 3D labels. This work introduces Label Any Pointcloud (LeAP), leveraging 2D VFMs to automatically label 3D data with any set of classes in any kind of application whilst ensuring label consistency. Using a Bayesian update, point labels are combined into voxels to improve spatio-temporal consistency. A novel 3D Consistency Network (3D-CN) exploits 3D information to further improve label quality. Through various experiments, we show that our method can generate high-quality 3D semantic labels across diverse fields without any manual labeling. Further, models adapted to new domains using our labels show up to a 34.2 mIoU increase in semantic segmentation tasks. 

**Abstract (ZH)**: 数据集的可用性是推进三维语义理解研究的重要驱动力。虽然获取未标记的三维点云数据相对简单，但手动为这些数据添加语义标签既耗时又昂贵。近年来，视觉基础模型（VFMs）能够实现开放集的图像语义分割，这可能会帮助自动标注。然而，针对三维数据的VFMs主要限于2D模型的改编，这可能会引入三维标签的一致性问题。本文引入了Label Any Pointcloud（LeAP），利用2D VFMs自动生成任何类别的三维数据标签，同时确保标签的一致性。通过贝叶斯更新，点标签被结合成体素，以提高时空一致性。我们提出了一种新颖的三维一致性网络（3D-CN），利用三维信息进一步提高标签质量。通过各种实验，我们证明了该方法可以在无需任何手动标注的情况下，生成跨多个领域的高质量三维语义标签。此外，使用我们的标签适应到新领域的模型，在语义分割任务中的mean IoU上可提高最多34.2%。 

---
# How vulnerable is my policy? Adversarial attacks on modern behavior cloning policies 

**Title (ZH)**: 我的政策有多脆弱？对现代行为克隆策略的对抗性攻击 

**Authors**: Basavasagar Patil, Akansha Kalra, Guanhong Tao, Daniel S. Brown  

**Link**: [PDF](https://arxiv.org/pdf/2502.03698)  

**Abstract**: Learning from Demonstration (LfD) algorithms have shown promising results in robotic manipulation tasks, but their vulnerability to adversarial attacks remains underexplored. This paper presents a comprehensive study of adversarial attacks on both classic and recently proposed algorithms, including Behavior Cloning (BC), LSTM-GMM, Implicit Behavior Cloning (IBC), Diffusion Policy (DP), and VQ-Behavior Transformer (VQ-BET). We study the vulnerability of these methods to untargeted, targeted and universal adversarial perturbations. While explicit policies, such as BC, LSTM-GMM and VQ-BET can be attacked in the same manner as standard computer vision models, we find that attacks for implicit and denoising policy models are nuanced and require developing novel attack methods. Our experiments on several simulated robotic manipulation tasks reveal that most of the current methods are highly vulnerable to adversarial perturbations. We also show that these attacks are transferable across algorithms, architectures, and tasks, raising concerning security vulnerabilities with potentially a white-box threat model. In addition, we test the efficacy of a randomized smoothing, a widely used adversarial defense technique, and highlight its limitation in defending against attacks on complex and multi-modal action distribution common in complex control tasks. In summary, our findings highlight the vulnerabilities of modern BC algorithms, paving way for future work in addressing such limitations. 

**Abstract (ZH)**: 学习示范（LfD）算法在机器人操作任务中展现了广阔的应用前景，但它们对对抗攻击的脆弱性仍未得到充分研究。本文全面探讨了对抗攻击对经典和近年提出的各种算法的影响，包括行为克隆（BC）、LSTM-GMM、隐式行为克隆（IBC）、扩散策略（DP）和变量子行为变换器（VQ-BET）。我们研究了这些方法对非目标攻击、目标攻击和通用对抗扰动的脆弱性。尽管显式策略，如BC、LSTM-GMM和VQ-BET可与标准计算机视觉模型一样进行攻击，但我们将发现，对隐式和去噪策略模型的攻击更为复杂，需要开发新的攻击方法。在几个模拟的机器人操作任务上的实验表明，当前大多数方法对对抗扰动非常脆弱。同时，我们还展示了这些攻击在不同算法、架构和任务间的可转移性，这表明存在潜在的白盒威胁模型，揭示出重大的安全漏洞。此外，我们测试了随机平滑——一种常用的对抗防御技术的有效性，揭示了其在应对复杂且多模态动作分布方面的局限性，这些常见于复杂控制任务中的攻击。总之，本文的研究结果指出现代BC算法的脆弱性，为未来工作解决此类局限性指明了方向。 

---
# TD-M(PC)$^2$: Improving Temporal Difference MPC Through Policy Constraint 

**Title (ZH)**: TD-M(PC)$^2$: 通过策略约束提高时差蒙特卡洛模型预测控制 

**Authors**: Haotian Lin, Pengcheng Wang, Jeff Schneider, Guanya Shi  

**Link**: [PDF](https://arxiv.org/pdf/2502.03550)  

**Abstract**: Model-based reinforcement learning algorithms that combine model-based planning and learned value/policy prior have gained significant recognition for their high data efficiency and superior performance in continuous control. However, we discover that existing methods that rely on standard SAC-style policy iteration for value learning, directly using data generated by the planner, often result in \emph{persistent value overestimation}. Through theoretical analysis and experiments, we argue that this issue is deeply rooted in the structural policy mismatch between the data generation policy that is always bootstrapped by the planner and the learned policy prior. To mitigate such a mismatch in a minimalist way, we propose a policy regularization term reducing out-of-distribution (OOD) queries, thereby improving value learning. Our method involves minimum changes on top of existing frameworks and requires no additional computation. Extensive experiments demonstrate that the proposed approach improves performance over baselines such as TD-MPC2 by large margins, particularly in 61-DoF humanoid tasks. View qualitative results at this https URL. 

**Abstract (ZH)**: 基于模型的强化学习算法结合了基于模型的规划和通过学习获得的价值/策略先验，已经在连续控制领域获得了显著的认同，特别是在数据效率和性能方面表现出色。然而，我们发现依赖于标准SAC风格价值学习的策略迭代方法，直接使用规划器生成的数据，经常会导致\emph{持续的价值过估计}。通过理论分析和实验，我们指出这个问题源于数据生成策略与学习到的策略先验之间固有的结构性策略不匹配。为了以简约的方式缓解这种不匹配，我们提出了一种减少离群查询（Out-of-Distribution, OOD）的策略正则化项，从而改善价值学习。我们的方法仅在现有框架的基础上做少量改动，不需要额外的计算。广泛的实验表明，与TD-MPC2等基准方法相比，所提出的方法在61个自由度的人形任务中显著提高了性能。请查看定性结果：[点击这里](https://example.com)。 

---
