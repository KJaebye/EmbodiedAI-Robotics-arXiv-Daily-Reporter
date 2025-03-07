# 3HANDS Dataset: Learning from Humans for Generating Naturalistic Handovers with Supernumerary Robotic Limbs 

**Title (ZH)**: 3HANDS数据集：从人类学习以生成具有辅助肢体的自然传递动作 

**Authors**: Artin Saberpour Abadian, Yi-Chi Liao, Ata Otaran, Rishabh Dabral, Marie Muehlhaus, Christian Theobalt, Martin Schmitz, Jürgen Steimle  

**Link**: [PDF](https://arxiv.org/pdf/2503.04635)  

**Abstract**: Supernumerary robotic limbs (SRLs) are robotic structures integrated closely with the user's body, which augment human physical capabilities and necessitate seamless, naturalistic human-machine interaction. For effective assistance in physical tasks, enabling SRLs to hand over objects to humans is crucial. Yet, designing heuristic-based policies for robots is time-consuming, difficult to generalize across tasks, and results in less human-like motion. When trained with proper datasets, generative models are powerful alternatives for creating naturalistic handover motions. We introduce 3HANDS, a novel dataset of object handover interactions between a participant performing a daily activity and another participant enacting a hip-mounted SRL in a naturalistic manner. 3HANDS captures the unique characteristics of SRL interactions: operating in intimate personal space with asymmetric object origins, implicit motion synchronization, and the user's engagement in a primary task during the handover. To demonstrate the effectiveness of our dataset, we present three models: one that generates naturalistic handover trajectories, another that determines the appropriate handover endpoints, and a third that predicts the moment to initiate a handover. In a user study (N=10), we compare the handover interaction performed with our method compared to a baseline. The findings show that our method was perceived as significantly more natural, less physically demanding, and more comfortable. 

**Abstract (ZH)**: Supernumerary Robotic Limbs (SRLs) 数据集 3HANDS：自然递物交互的研究 

---
# Whole-Body Model-Predictive Control of Legged Robots with MuJoCo 

**Title (ZH)**: 基于MuJoCo的腿式机器人全身模型预测控制 

**Authors**: John Z. Zhang, Taylor A. Howell, Zeji Yi, Chaoyi Pan, Guanya Shi, Guannan Qu, Tom Erez, Yuval Tassa, Zachary Manchester  

**Link**: [PDF](https://arxiv.org/pdf/2503.04613)  

**Abstract**: We demonstrate the surprising real-world effectiveness of a very simple approach to whole-body model-predictive control (MPC) of quadruped and humanoid robots: the iterative LQR (iLQR) algorithm with MuJoCo dynamics and finite-difference approximated derivatives. Building upon the previous success of model-based behavior synthesis and control of locomotion and manipulation tasks with MuJoCo in simulation, we show that these policies can easily generalize to the real world with few sim-to-real considerations. Our baseline method achieves real-time whole-body MPC on a variety of hardware experiments, including dynamic quadruped locomotion, quadruped walking on two legs, and full-sized humanoid bipedal locomotion. We hope this easy-to-reproduce hardware baseline lowers the barrier to entry for real-world whole-body MPC research and contributes to accelerating research velocity in the community. Our code and experiment videos will be available online at:this https URL 

**Abstract (ZH)**: 我们展示了在四足机器人和类人机器人全身模型预测控制(MPC)中，一个极为简单的迭代线性二次调节(iLQR)算法结合MuJoCo动力学和有限差分逼近导数的惊人实际效果。基于MuJoCo在仿真中成功实现基于模型的行为合成与控制，特别是在运动和操作任务中的表现，我们证明了这些策略在现实世界中只需少量的仿真到现实世界的转换即可轻松泛化。我们的基准方法在多种硬件实验中实现了实时的全身MPC，包括动态四足运动、两足行走以及全尺寸类人双足运动。我们希望这个易于复现的硬件基准可以降低实际世界中全身MPC研究的门槛，并加速社区中的研究进度。我们的代码和实验视频将在以下网址在线提供：this https URL 

---
# ExoNav II: Design of a Robotic Tool with Follow-the-Leader Motion Capability for Lateral and Ventral Spinal Cord Stimulation (SCS) 

**Title (ZH)**: ExoNav II：具有跟随领导者运动能力的外骨骼导航工具设计，用于侧方和腹侧脊髓电刺激（SCS） 

**Authors**: Behnam Moradkhani, Pejman Kheradmand, Harshith Jella, Joseph Klein, Ajmal Zemmar, Yash Chitalia  

**Link**: [PDF](https://arxiv.org/pdf/2503.04603)  

**Abstract**: Spinal cord stimulation (SCS) electrodes are traditionally placed in the dorsal epidural space to stimulate the dorsal column fibers for pain therapy. Recently, SCS has gained attention in restoring gait. However, the motor fibers triggering locomotion are located in the ventral and lateral spinal cord. Currently, SCS electrodes are steered manually, making it difficult to navigate them to the lateral and ventral motor fibers in the spinal cord. In this work, we propose a helically micro-machined continuum robot that can bend in a helical shape when subjected to actuation tendon forces. Using a stiff outer tube and adding translational and rotational degrees of freedom, this helical continuum robot can perform follow-the-leader (FTL) motion. We propose a kinematic model to relate tendon stroke and geometric parameters of the robot's helical shape to its acquired trajectory and end-effector position. We evaluate the proposed kinematic model and the robot's FTL motion capability experimentally. The stroke-based method, which links tendon stroke values to the robot's shape, showed inaccuracies with a 19.84 mm deviation and an RMSE of 14.42 mm for 63.6 mm of robot's length bending. The position-based method, using kinematic equations to map joint space to task space, performed better with a 10.54 mm deviation and an RMSE of 8.04 mm. Follow-the-leader experiments showed deviations of 11.24 mm and 7.32 mm, with RMSE values of 8.67 mm and 5.18 mm for the stroke-based and position-based methods, respectively. Furthermore, end-effector trajectories in two FTL motion trials are compared to confirm the robot's repeatable behavior. Finally, we demonstrate the robot's operation on a 3D-printed spinal cord phantom model. 

**Abstract (ZH)**: 基于螺线微加工连续体机器人的脊髓刺激电极导航方法研究 

---
# DogLegs: Robust Proprioceptive State Estimation for Legged Robots Using Multiple Leg-Mounted IMUs 

**Title (ZH)**: DogLegs：使用多个腿式IMU的腿部机器人 proprioceptive 状态估计的鲁棒方法 

**Authors**: Yibin Wu, Jian Kuang, Shahram Khorshidi, Xiaoji Niu, Lasse Klingbeil, Maren Bennewitz, Heiner Kuhlmann  

**Link**: [PDF](https://arxiv.org/pdf/2503.04580)  

**Abstract**: Robust and accurate proprioceptive state estimation of the main body is crucial for legged robots to execute tasks in extreme environments where exteroceptive sensors, such as LiDARs and cameras may become unreliable. In this paper, we propose DogLegs, a state estimation system for legged robots that fuses the measurements from a body-mounted inertial measurement unit (Body-IMU), joint encoders, and multiple leg-mounted IMUs (Leg-IMU) using an extended Kalman filter (EKF). The filter system contains the error states of all IMU frames. The Leg-IMUs are used to detect foot contact, thereby providing zero velocity measurements to update the state of the Leg-IMU frames. Additionally, we compute the relative position constraints between the Body-IMU and Leg-IMUs by the leg kinematics and use them to update the main body state and reduce the error drift of the individual IMU frames. Field experimental results have shown that our proposed system can achieve better state estimation accuracy compared to the traditional leg odometry method (using only Body-IMU and joint encoders) across different terrains. We make our datasets publicly available to benefit the research community. 

**Abstract (ZH)**: 腿部机器人在极端环境中的稳健且准确的本体状态估计对于执行任务至关重要，其中外部传感器如激光雷达和摄像头可能变得不可靠。本文提出DogLegs，一种使用扩展卡尔曼滤波器融合装在身体上的惯性测量单元（Body-IMU）、关节编码器和多个装在腿部的惯性测量单元（Leg-IMU）的状态估计系统。滤波器系统包含了所有惯性测量单元坐标系的误差状态。腿部惯性测量单元用于检测足部接触，从而提供零速度测量以更新腿部惯性测量单元坐标系的状态。此外，我们通过腿部运动学计算身体惯性测量单元和腿部惯性测量单元之间的相对位置约束，并使用它们来更新主体状态并减少单个惯性测量单元坐标系的误差漂移。实地实验结果表明，与仅使用身体惯性测量单元和关节编码器的传统腿部里程计方法相比，本提出系统在不同地形上可以获得更好的状态估计精度。我们公开了我们的数据集以造福研究社区。 

---
# Data-augmented Learning of Geodesic Distances in Irregular Domains through Soner Boundary Conditions 

**Title (ZH)**: 通过Sonner边界条件在不规则域中增强数据学习测地距离 

**Authors**: Rafael I. Cabral Muchacho, Florian T. Pokorny  

**Link**: [PDF](https://arxiv.org/pdf/2503.04579)  

**Abstract**: Geodesic distances play a fundamental role in robotics, as they efficiently encode global geometric information of the domain. Recent methods use neural networks to approximate geodesic distances by solving the Eikonal equation through physics-informed approaches. While effective, these approaches often suffer from unstable convergence during training in complex environments. We propose a framework to learn geodesic distances in irregular domains by using the Soner boundary condition, and systematically evaluate the impact of data losses on training stability and solution accuracy. Our experiments demonstrate that incorporating data losses significantly improves convergence robustness, reducing training instabilities and sensitivity to initialization. These findings suggest that hybrid data-physics approaches can effectively enhance the reliability of learning-based geodesic distance solvers with sparse data. 

**Abstract (ZH)**: 不规则域中基于Soner边条件学习测地距离的框架及数据损失对训练稳定性和解的准确性的系统评估 

---
# Occlusion-Aware Consistent Model Predictive Control for Robot Navigation in Occluded Obstacle-Dense Environments 

**Title (ZH)**: 考虑遮挡的一致模型预测控制在稠密障碍物遮挡环境中的机器人导航 

**Authors**: Minzhe Zheng, Lei Zheng, Lei Zhu, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.04563)  

**Abstract**: Ensuring safety and motion consistency for robot navigation in occluded, obstacle-dense environments is a critical challenge. In this context, this study presents an occlusion-aware Consistent Model Predictive Control (CMPC) strategy. To account for the occluded obstacles, it incorporates adjustable risk regions that represent their potential future locations. Subsequently, dynamic risk boundary constraints are developed online to ensure safety. The CMPC then constructs multiple locally optimal trajectory branches (each tailored to different risk regions) to balance between exploitation and exploration. A shared consensus trunk is generated to ensure smooth transitions between branches without significant velocity fluctuations, further preserving motion consistency. To facilitate high computational efficiency and ensure coordination across local trajectories, we use the alternating direction method of multipliers (ADMM) to decompose the CMPC into manageable sub-problems for parallel solving. The proposed strategy is validated through simulation and real-world experiments on an Ackermann-steering robot platform. The results demonstrate the effectiveness of the proposed CMPC strategy through comparisons with baseline approaches in occluded, obstacle-dense environments. 

**Abstract (ZH)**: 确保机器人在遮挡和障碍物密集环境中导航的安全性和运动一致性是一项关键挑战。在此背景下，本文提出了一种 Awareness-Occlusion 的一致模型预测控制（CMPC）策略。为了考虑遮挡的障碍物，它引入了可调节的风险区域来表示它们的潜在未来位置。随后，开发了在线动态风险边界约束以确保安全。CMPC 构建了多个局部最优轨迹分支（每个分支针对不同的风险区域），以在利用和探索之间取得平衡。生成了一个共享共识主干来确保分支之间的平滑过渡，同时避免显著的速度波动，从而进一步保持运动一致性。为了实现高效的计算效率并确保局部轨迹之间的协调，我们使用交替方向乘子法（ADMM）将 CMPC 分解为可并行求解的子问题。所提出的策略通过仿真实验和在 Ackermann 转向机器人平台上的实地试验得到了验证。结果表明，在遮挡和障碍物密集环境中，所提出的 CMPC 策略优于基线方法。 

---
# Learning Generalizable Language-Conditioned Cloth Manipulation from Long Demonstrations 

**Title (ZH)**: 基于长示范学习可泛化的语言条件化布料 manipulation 

**Authors**: Hanyi Zhao, Jinxuan Zhu, Zihao Yan, Yichen Li, Yuhong Deng, Xueqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04557)  

**Abstract**: Multi-step cloth manipulation is a challenging problem for robots due to the high-dimensional state spaces and the dynamics of cloth. Despite recent significant advances in end-to-end imitation learning for multi-step cloth manipulation skills, these methods fail to generalize to unseen tasks. Our insight in tackling the challenge of generalizable multi-step cloth manipulation is decomposition. We propose a novel pipeline that autonomously learns basic skills from long demonstrations and composes learned basic skills to generalize to unseen tasks. Specifically, our method first discovers and learns basic skills from the existing long demonstration benchmark with the commonsense knowledge of a large language model (LLM). Then, leveraging a high-level LLM-based task planner, these basic skills can be composed to complete unseen tasks. Experimental results demonstrate that our method outperforms baseline methods in learning multi-step cloth manipulation skills for both seen and unseen tasks. 

**Abstract (ZH)**: 多步骤布料 manipulation 的自主分解学习方法 

---
# ViT-VS: On the Applicability of Pretrained Vision Transformer Features for Generalizable Visual Servoing 

**Title (ZH)**: ViT-VS：预训练视觉变换器特征在通用视觉 servoing 中的应用探索 

**Authors**: Alessandro Scherl, Stefan Thalhammer, Bernhard Neuberger, Wilfried Wöber, José Gracía-Rodríguez  

**Link**: [PDF](https://arxiv.org/pdf/2503.04545)  

**Abstract**: Visual servoing enables robots to precisely position their end-effector relative to a target object. While classical methods rely on hand-crafted features and thus are universally applicable without task-specific training, they often struggle with occlusions and environmental variations, whereas learning-based approaches improve robustness but typically require extensive training. We present a visual servoing approach that leverages pretrained vision transformers for semantic feature extraction, combining the advantages of both paradigms while also being able to generalize beyond the provided sample. Our approach achieves full convergence in unperturbed scenarios and surpasses classical image-based visual servoing by up to 31.2\% relative improvement in perturbed scenarios. Even the convergence rates of learning-based methods are matched despite requiring no task- or object-specific training. Real-world evaluations confirm robust performance in end-effector positioning, industrial box manipulation, and grasping of unseen objects using only a reference from the same category. Our code and simulation environment are available at: this https URL 

**Abstract (ZH)**: 视觉伺服使机器人能够精确定位其末端执行器相对于目标物体的位置。虽然经典的视觉伺服方法依赖于手动设计的特征，因此不需要针对具体任务进行训练即可广泛适用，但它们往往难以应对遮挡和环境变化，而基于学习的方法则提高了鲁棒性，但通常需要大量的训练。我们提出了一种视觉伺服方法，利用预训练的视觉变换器进行语义特征提取，结合了两种范式的优点，并且能够在提供的样本之外泛化。在这种方法中，可以在未受干扰的场景下实现完全收敛，并在受干扰场景下相对于基于图像的经典视觉伺服方法实现了高达31.2%的相对改进。即使在不需要针对特定任务或物体进行训练的情况下，基于学习的方法的收敛速率也得到了匹配。实验证明，该方法在末端执行器定位、工业箱体操作以及从未见过的物体抓取方面表现出稳健的性能，并仅需同一类别的参考图像。我们的代码和模拟环境可在以下链接获取：this https URL。 

---
# SRSA: Skill Retrieval and Adaptation for Robotic Assembly Tasks 

**Title (ZH)**: SRSA: 技能检索与适应在机器人装配任务中的应用 

**Authors**: Yijie Guo, Bingjie Tang, Iretiayo Akinola, Dieter Fox, Abhishek Gupta, Yashraj Narang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04538)  

**Abstract**: Enabling robots to learn novel tasks in a data-efficient manner is a long-standing challenge. Common strategies involve carefully leveraging prior experiences, especially transition data collected on related tasks. Although much progress has been made for general pick-and-place manipulation, far fewer studies have investigated contact-rich assembly tasks, where precise control is essential. We introduce SRSA (Skill Retrieval and Skill Adaptation), a novel framework designed to address this problem by utilizing a pre-existing skill library containing policies for diverse assembly tasks. The challenge lies in identifying which skill from the library is most relevant for fine-tuning on a new task. Our key hypothesis is that skills showing higher zero-shot success rates on a new task are better suited for rapid and effective fine-tuning on that task. To this end, we propose to predict the transfer success for all skills in the skill library on a novel task, and then use this prediction to guide the skill retrieval process. We establish a framework that jointly captures features of object geometry, physical dynamics, and expert actions to represent the tasks, allowing us to efficiently learn the transfer success predictor. Extensive experiments demonstrate that SRSA significantly outperforms the leading baseline. When retrieving and fine-tuning skills on unseen tasks, SRSA achieves a 19% relative improvement in success rate, exhibits 2.6x lower standard deviation across random seeds, and requires 2.4x fewer transition samples to reach a satisfactory success rate, compared to the baseline. Furthermore, policies trained with SRSA in simulation achieve a 90% mean success rate when deployed in the real world. Please visit our project webpage this https URL. 

**Abstract (ZH)**: 使机器人能够以数据高效的方式学习新型任务是一项长期挑战。SRSA（技能检索与技能适应）框架通过利用包含多种装配任务策略的预存技能库，旨在解决这一问题。我们假设在新任务上零样本成功率较高的技能更适合快速有效地对新任务进行微调。为此，我们提出了一种预测技能库中所有技能在新任务上的转移成功率的方法，并利用预测结果指导技能检索过程。我们建立了一个框架，共同捕捉对象几何、物理动力学和专家行动的特征来表示任务，这使我们能够高效地学习转移成功率预测器。广泛的实验表明，SRSA 显著优于领先基线。与基线相比，SRSA 在检索和微调未见过的任务时，成功率相对提高19%，标准偏差降低2.6倍，所需过渡样本数减少2.4倍以达到满意的成功率。此外，使用SRSA在仿真中训练的策略在实际部署中平均成功率为90%。请访问我们的项目网页：[此网址]。 

---
# PALo: Learning Posture-Aware Locomotion for Quadruped Robots 

**Title (ZH)**: PALo：学习姿态感知四足机器人运动控制 

**Authors**: Xiangyu Miao, Jun Sun, Hang Lai, Xinpeng Di, Jiahang Cao, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04462)  

**Abstract**: With the rapid development of embodied intelligence, locomotion control of quadruped robots on complex terrains has become a research hotspot. Unlike traditional locomotion control approaches focusing solely on velocity tracking, we pursue to balance the agility and robustness of quadruped robots on diverse and complex terrains. To this end, we propose an end-to-end deep reinforcement learning framework for posture-aware locomotion named PALo, which manages to handle simultaneous linear and angular velocity tracking and real-time adjustments of body height, pitch, and roll angles. In PALo, the locomotion control problem is formulated as a partially observable Markov decision process, and an asymmetric actor-critic architecture is adopted to overcome the sim-to-real challenge. Further, by incorporating customized training curricula, PALo achieves agile posture-aware locomotion control in simulated environments and successfully transfers to real-world settings without fine-tuning, allowing real-time control of the quadruped robot's locomotion and body posture across challenging terrains. Through in-depth experimental analysis, we identify the key components of PALo that contribute to its performance, further validating the effectiveness of the proposed method. The results of this study provide new possibilities for the low-level locomotion control of quadruped robots in higher dimensional command spaces and lay the foundation for future research on upper-level modules for embodied intelligence. 

**Abstract (ZH)**: 随着嵌入式智能的快速发展，四足机器人在复杂地形上的运动控制已成为研究热点。与传统的仅仅专注于速度跟踪的运动控制方法不同，我们致力于在多样且复杂的地形上平衡四足机器人的敏捷性和鲁棒性。为此，我们提出了一种名为PALo的端到端深度强化学习框架，该框架能够同时处理线性速度和角速度的跟踪，并实时调整身体高度、俯仰角和滚转角，实现姿态感知的运动控制。在PALo中，运动控制问题被建模为部分可观测马尔可夫决策过程，并采用不对称的演员-评论家架构来克服仿真实验到真实环境的转化挑战。通过集成定制化的训练课程，PALo在模拟环境中实现了敏捷的姿态感知运动控制，并能够在无需微调的情况下成功转移到实际应用场景，实现了四足机器人在挑战性地形上的实时运动和姿态控制。通过深入的实验分析，我们确定了PALo的关键组件，进一步验证了所提出方法的有效性。本研究的结果为四足机器人在高维命令空间中的低级运动控制提供了新的可能性，并为未来基于代理智能的高级模块研究奠定了基础。 

---
# EvidMTL: Evidential Multi-Task Learning for Uncertainty-Aware Semantic Surface Mapping from Monocular RGB Images 

**Title (ZH)**: EvidMTL: 证据多任务学习在单目RGB图像中带不确定性意识语义表面映射中的应用 

**Authors**: Rohit Menon, Nils Dengler, Sicong Pan, Gokul Krishna Chenchani, Maren Bennewitz  

**Link**: [PDF](https://arxiv.org/pdf/2503.04441)  

**Abstract**: For scene understanding in unstructured environments, an accurate and uncertainty-aware metric-semantic mapping is required to enable informed action selection by autonomous this http URL mapping methods often suffer from overconfident semantic predictions, and sparse and noisy depth sensing, leading to inconsistent map representations. In this paper, we therefore introduce EvidMTL, a multi-task learning framework that uses evidential heads for depth estimation and semantic segmentation, enabling uncertainty-aware inference from monocular RGB images. To enable uncertainty-calibrated evidential multi-task learning, we propose a novel evidential depth loss function that jointly optimizes the belief strength of the depth prediction in conjunction with evidential segmentation loss. Building on this, we present EvidKimera, an uncertainty-aware semantic surface mapping framework, which uses evidential depth and semantics prediction for improved 3D metric-semantic consistency. We train and evaluate EvidMTL on the NYUDepthV2 and assess its zero-shot performance on ScanNetV2, demonstrating superior uncertainty estimation compared to conventional approaches while maintaining comparable depth estimation and semantic segmentation. In zero-shot mapping tests on ScanNetV2, EvidKimera outperforms Kimera in semantic surface mapping accuracy and consistency, highlighting the benefits of uncertainty-aware mapping and underscoring its potential for real-world robotic applications. 

**Abstract (ZH)**: 面向非结构化环境的场景理解需要一种准确且aware不确定性的度量语义映射，以支持自主系统进行有根据的动作选择。现有的映射方法往往受到自信心过强的语义预测和稀疏的噪声深度传感的影响，导致地图表示不一致。因此，本文提出了一种使用证据头部进行深度估计和语义分割的多任务学习框架EvidMTL，使从单目RGB图像中进行aware不确定性的推理成为可能。为实现校准不确定性的证据多任务学习，我们提出了一种新颖的证据深度损失函数，该函数可以联合优化深度预测的信念强度以及语义分割损失。在此基础上，我们介绍了EvidKimera，这是一种aware不确定性的语义表面映射框架，使用证据深度和语义预测以提高三维度量语义一致性。我们在NYUDepthV2上训练并评估了EvidMTL，并在ScanNetV2上评估其零样本性能，其不确定性的估计优于传统方法，同时保持了类似深度估计和语义分割的性能。在ScanNetV2上的零样本映射测试中，EvidKimera在语义表面映射的准确性和一致性方面优于Kimera，强调了aware不确定性映射的优势，并突显了其在实际机器人应用中的潜力。 

---
# On the Analysis of Stability, Sensitivity and Transparency in Variable Admittance Control for pHRI Enhanced by Virtual Fixtures 

**Title (ZH)**: 基于虚拟fixtures增强的pHRI可变阻抗控制中的稳定性、灵敏度和透明度分析 

**Authors**: Davide Tebaldi, Dario Onfiani, Luigi Biagiotti  

**Link**: [PDF](https://arxiv.org/pdf/2503.04414)  

**Abstract**: The interest in Physical Human-Robot Interaction (pHRI) has significantly increased over the last two decades thanks to the availability of collaborative robots that guarantee user safety during force exchanges. For this reason, stability concerns have been addressed extensively in the literature while proposing new control schemes for pHRI applications. Because of the nonlinear nature of robots, stability analyses generally leverage passivity concepts. On the other hand, the proposed algorithms generally consider ideal models of robot manipulators. For this reason, the primary objective of this paper is to conduct a detailed analysis of the sources of instability for a class of pHRI control schemes, namely proxy-based constrained admittance controllers, by considering parasitic effects such as transmission elasticity, motor velocity saturation, and actuation delay. Next, a sensitivity analysis supported by experimental results is carried out, in order to identify how the control parameters affect the stability of the overall system. Finally, an adaptation technique for the proxy parameters is proposed with the goal of maximizing transparency in pHRI. The proposed adaptation method is validated through both simulations and experimental tests. 

**Abstract (ZH)**: 基于代理的约束阻抗控制器下物理人机互动控制方案稳定性分析及其参数调整方法 

---
# SeGMan: Sequential and Guided Manipulation Planner for Robust Planning in 2D Constrained Environments 

**Title (ZH)**: SeGMan: 用于2D约束环境稳健规划的序列引导操作规划器 

**Authors**: Cankut Bora Tuncer, Dilruba Sultan Haliloglu, Ozgur S. Oguz  

**Link**: [PDF](https://arxiv.org/pdf/2503.04409)  

**Abstract**: In this paper, we present SeGMan, a hybrid motion planning framework that integrates sampling-based and optimization-based techniques with a guided forward search to address complex, constrained sequential manipulation challenges, such as pick-and-place puzzles. SeGMan incorporates an adaptive subgoal selection method that adjusts the granularity of subgoals, enhancing overall efficiency. Furthermore, proposed generalizable heuristics guide the forward search in a more targeted manner. Extensive evaluations in maze-like tasks populated with numerous objects and obstacles demonstrate that SeGMan is capable of generating not only consistent and computationally efficient manipulation plans but also outperform state-of-the-art approaches. 

**Abstract (ZH)**: 本文提出了一种名为SeGMan的混合运动规划框架，将基于采样的技术和基于优化的方法与引导式前向搜索相结合，以应对复杂的受约束的顺序操作挑战，如拾取和放置 puzzle。SeGMan 汇入了一种自适应子目标选择方法，可通过调整子目标的粒度来提高整体效率。此外，提出的可泛化的启发式方法以更针对性的方式引导前向搜索。在充满大量物体和障碍物的迷宫式任务中进行的广泛评估表明，SeGMan 不仅有能力生成一致且计算效率高的操作计划，还在性能上超越了现有最先进的方法。 

---
# Energy Consumption of Robotic Arm with the Local Reduction Method 

**Title (ZH)**: 基于局部缩减方法的机器人手臂能耗研究 

**Authors**: Halima Ibrahim Kure, Jishna Retnakumari, Lucian Nita, Saeed Sharif, Hamed Balogun, Augustine O. Nwajana  

**Link**: [PDF](https://arxiv.org/pdf/2503.04340)  

**Abstract**: Energy consumption in robotic arms is a significant concern in industrial automation due to rising operational costs and environmental impact. This study investigates the use of a local reduction method to optimize energy efficiency in robotic systems without compromising performance. The approach refines movement parameters, minimizing energy use while maintaining precision and operational reliability. A three-joint robotic arm model was tested using simulation over a 30-second period for various tasks, including pick-and-place and trajectory-following operations. The results revealed that the local reduction method reduced energy consumption by up to 25% compared to traditional techniques such as Model Predictive Control (MPC) and Genetic Algorithms (GA). Unlike MPC, which requires significant computational resources, and GA, which has slow convergence rates, the local reduction method demonstrated superior adaptability and computational efficiency in real-time applications. The study highlights the scalability and simplicity of the local reduction approach, making it an attractive option for industries seeking sustainable and cost-effective solutions. Additionally, this method can integrate seamlessly with emerging technologies like Artificial Intelligence (AI), further enhancing its application in dynamic and complex environments. This research underscores the potential of the local reduction method as a practical tool for optimizing robotic arm operations, reducing energy demands, and contributing to sustainability in industrial automation. Future work will focus on extending the approach to real-world scenarios and incorporating AI-driven adjustments for more dynamic adaptability. 

**Abstract (ZH)**: 机器人手臂的能效优化：局部减少方法的研究 

---
# Shaken, Not Stirred: A Novel Dataset for Visual Understanding of Glasses in Human-Robot Bartending Tasks 

**Title (ZH)**: 摇而不搅：一种新型数据集，用于人类-机器人调酒任务中的玻璃视觉理解 

**Authors**: Lukáš Gajdošech, Hassan Ali, Jan-Gerrit Habekost, Martin Madaras, Matthias Kerzel, Stefan Wermter  

**Link**: [PDF](https://arxiv.org/pdf/2503.04308)  

**Abstract**: Datasets for object detection often do not account for enough variety of glasses, due to their transparent and reflective properties. Specifically, open-vocabulary object detectors, widely used in embodied robotic agents, fail to distinguish subclasses of glasses. This scientific gap poses an issue to robotic applications that suffer from accumulating errors between detection, planning, and action execution. The paper introduces a novel method for the acquisition of real-world data from RGB-D sensors that minimizes human effort. We propose an auto-labeling pipeline that generates labels for all the acquired frames based on the depth measurements. We provide a novel real-world glass object dataset that was collected on the Neuro-Inspired COLlaborator (NICOL), a humanoid robot platform. The data set consists of 7850 images recorded from five different cameras. We show that our trained baseline model outperforms state-of-the-art open-vocabulary approaches. In addition, we deploy our baseline model in an embodied agent approach to the NICOL platform, on which it achieves a success rate of 81% in a human-robot bartending scenario. 

**Abstract (ZH)**: 物体检测数据集往往未能充分考虑到眼镜的多样性，鉴于它们的透明和反射特性。具体而言，广泛应用于体内机器人代理的开放词汇物体检测器无法区分眼镜的子类别。这一科学缺口影响了因检测、规划和执行动作之间的累积误差而受到影响的机器人应用。本文介绍了一种新方法，用于从RGB-D传感器获取现实世界数据，以最小化人力投入。我们提出了一种自动标注流水线，根据深度测量生成所有获取帧的标签。我们提供了一个由神经启发式协作机器人（NICOL）收集的新颖现实世界眼镜对象数据集，NICOL是一个类人机器人平台。数据集包含从五个不同摄像头记录的7850张图像。我们表明，我们训练的基础模型在开放词汇方法中表现优于现有最佳方法。此外，我们在NICOL平台上部署了基础模型，用于体内代理在人类-机器人调酒场景中实现了81%的成功率。 

---
# Manipulation of Elasto-Flexible Cables with Single or Multiple UAVs 

**Title (ZH)**: 使用单个或多个无人机操纵弹性柔性缆线 

**Authors**: Chiara Gabellieri, Lars Teeuwen, Yaolei Shen, Antonio Franchi  

**Link**: [PDF](https://arxiv.org/pdf/2503.04304)  

**Abstract**: This work considers a large class of systems composed of multiple quadrotors manipulating deformable and extensible cables. The cable is described via a discretized representation, which decomposes it into linear springs interconnected through lumped-mass passive spherical joints. Sets of flat outputs are found for the systems. Numerical simulations support the findings by showing cable manipulation relying on flatness-based trajectories. Eventually, we present an experimental validation of the effectiveness of the proposed discretized cable model for a two-robot example. Moreover, a closed-loop controller based on the identified model and using cable-output feedback is experimentally tested. 

**Abstract (ZH)**: 本研究考虑了一类由多个 quadrotor 操纵可变形和可伸展缆线的系统。缆线通过离散化表示进行描述，将其分解为通过集中质量被动球关节连接的线性弹簧。找到了系统的集控输出集。数值模拟支持这些发现，展示了基于平坦性轨迹的缆线操纵。最后，我们介绍了所提出的离散化缆线模型在两机器人示例中的实验验证，并基于识别的模型设计了一个闭环控制器，使用缆线输出反馈进行了实验测试。 

---
# Towards Autonomous Reinforcement Learning for Real-World Robotic Manipulation with Large Language Models 

**Title (ZH)**: 面向大型语言模型的自主强化学习在实际机器人操作中的应用 

**Authors**: Niccolò Turcato, Matteo Iovino, Aris Synodinos, Alberto Dalla Libera, Ruggero Carli, Pietro Falco  

**Link**: [PDF](https://arxiv.org/pdf/2503.04280)  

**Abstract**: Recent advancements in Large Language Models (LLMs) and Visual Language Models (VLMs) have significantly impacted robotics, enabling high-level semantic motion planning applications. Reinforcement Learning (RL), a complementary paradigm, enables agents to autonomously optimize complex behaviors through interaction and reward signals. However, designing effective reward functions for RL remains challenging, especially in real-world tasks where sparse rewards are insufficient and dense rewards require elaborate design. In this work, we propose Autonomous Reinforcement learning for Complex HumanInformed Environments (ARCHIE), an unsupervised pipeline leveraging GPT-4, a pre-trained LLM, to generate reward functions directly from natural language task descriptions. The rewards are used to train RL agents in simulated environments, where we formalize the reward generation process to enhance feasibility. Additionally, GPT-4 automates the coding of task success criteria, creating a fully automated, one-shot procedure for translating human-readable text into deployable robot skills. Our approach is validated through extensive simulated experiments on single-arm and bi-manual manipulation tasks using an ABB YuMi collaborative robot, highlighting its practicality and effectiveness. Tasks are demonstrated on the real robot setup. 

**Abstract (ZH)**: 近期大规模语言模型（LLMs）和视觉语言模型（VLMs）的发展显著影响了机器人技术，使其能够实现高级语义运动规划应用。强化学习（RL）作为一种互补范式，可通过交互和奖励信号使智能体自主优化复杂行为。然而，为RL设计有效的奖励函数仍然是一个挑战，尤其是在实际任务中稀疏奖励不足，密集奖励需要复杂的定制设计。在此工作中，我们提出了自主强化学习用于复杂人类导向环境的架构（ARCHIE），这是一种无监督流程，利用预训练的LLM GPT-4直接从自然语言任务描述生成奖励函数。这些奖励用于在模拟环境中训练RL智能体，并正式化奖励生成过程以增强其实现性。此外，GPT-4 自动化了任务成功标准的编码，创造了将可读文本一键转化为可部署机器人技能的全自动流程。通过使用ABB YuMi协作机器人进行广泛的单臂和双臂操作任务的模拟实验，验证了该方法的实际可行性和有效性，并在实际机器人设置中演示了任务。 

---
# VLA Model-Expert Collaboration for Bi-directional Manipulation Learning 

**Title (ZH)**: VLA模型与专家协作的双向操作学习 

**Authors**: Tian-Yu Xiang, Ao-Qun Jin, Xiao-Hu Zhou, Mei-Jiang Gui, Xiao-Liang Xie, Shi-Qi Liu, Shuang-Yi Wang, Sheng-Bin Duang, Si-Cheng Wang, Zheng Lei, Zeng-Guang Hou  

**Link**: [PDF](https://arxiv.org/pdf/2503.04163)  

**Abstract**: The emergence of vision-language-action (VLA) models has given rise to foundation models for robot manipulation. Although these models have achieved significant improvements, their generalization in multi-task manipulation remains limited. This study proposes a VLA model-expert collaboration framework that leverages a limited number of expert actions to enhance VLA model performance. This approach reduces expert workload relative to manual operation while simultaneously improving the reliability and generalization of VLA models. Furthermore, manipulation data collected during collaboration can further refine the VLA model, while human participants concurrently enhance their skills. This bi-directional learning loop boosts the overall performance of the collaboration system. Experimental results across various VLA models demonstrate the effectiveness of the proposed system in collaborative manipulation and learning, as evidenced by improved success rates across tasks. Additionally, validation using a brain-computer interface (BCI) indicates that the collaboration system enhances the efficiency of low-speed action systems by involving VLA model during manipulation. These promising results pave the way for advancing human-robot interaction in the era of foundation models for robotics. (Project website: this https URL) 

**Abstract (ZH)**: 视觉-语言-动作（VLA）模型的崛起为机器人操作提供了基础模型。尽管这些模型已经取得了显著的进步，但在多任务操作中的泛化能力仍有限。本研究提出了一种VLA模型-专家协作框架，利用少量专家操作来提升VLA模型的性能。这一方法相对于手动操作减少了专家的工作负担，同时提高了VLA模型的可靠性和泛化能力。此外，在协作过程中收集的操纵数据可以进一步细化VLA模型，同时参与的人类参与者也同步提升了他们的技能。这种双向学习循环提升了协作系统的整体性能。在多种VLA模型上的实验结果表明，所提出的系统在协作操纵和学习方面具有有效性，表现为任务成功率的提高。此外，通过脑-机接口（BCI）验证表明，协作系统通过在操作过程中包含VLA模型，提高了低速操作系统的效率。这些有前景的结果为在机器人基础模型时代推进人机交互 paved the way。 (项目网站: [this https URL]) 

---
# Simulation-based Analysis Of Highway Trajectory Planning Using High-Order Polynomial For Highly Automated Driving Function 

**Title (ZH)**: 基于高次多项式的高速公路轨迹规划simulations分析：面向高层次自动驾驶功能 

**Authors**: Milin Patel, Marzana Khatun, Rolf Jung, Michael Glaß  

**Link**: [PDF](https://arxiv.org/pdf/2503.04159)  

**Abstract**: One of the fundamental tasks of autonomous driving is safe trajectory planning, the task of deciding where the vehicle needs to drive, while avoiding obstacles, obeying safety rules, and respecting the fundamental limits of road. Real-world application of such a method involves consideration of surrounding environment conditions and movements such as Lane Change, collision avoidance, and lane merge. The focus of the paper is to develop and implement safe collision free highway Lane Change trajectory using high order polynomial for Highly Automated Driving Function (HADF). Planning is often considered as a higher-level process than control. Behavior Planning Module (BPM) is designed that plans the high-level driving actions like Lane Change maneuver to safely achieve the functionality of transverse guidance ensuring safety of the vehicle using motion planning in a scenario including environmental situation. Based on the recommendation received from the (BPM), the function will generate a desire corresponding trajectory. The proposed planning system is situation specific with polynomial based algorithm for same direction two lane highway scenario. To support the trajectory system polynomial curve can be used to reduces overall complexity and thereby allows rapid computation. The proposed Lane Change scenario is modeled, and results has been analyzed (verified and validate) through the MATLAB simulation environment. The method proposed in this paper has achieved a significant improvement in safety and stability of Lane Changing maneuver. 

**Abstract (ZH)**: 基于高次多项式的适配行驶车道变道无碰撞安全轨迹规划方法 

---
# Real-time Spatial-temporal Traversability Assessment via Feature-based Sparse Gaussian Process 

**Title (ZH)**: 基于特征的稀疏高斯过程实时时空可通行性评估 

**Authors**: Senming Tan, Zhenyu Hou, Zhihao Zhang, Long Xu, Mengke Zhang, Zhaoqi He, Chao Xu, Fei Gao, Yanjun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04134)  

**Abstract**: Terrain analysis is critical for the practical application of ground mobile robots in real-world tasks, especially in outdoor unstructured environments. In this paper, we propose a novel spatial-temporal traversability assessment method, which aims to enable autonomous robots to effectively navigate through complex terrains. Our approach utilizes sparse Gaussian processes (SGP) to extract geometric features (curvature, gradient, elevation, etc.) directly from point cloud scans. These features are then used to construct a high-resolution local traversability map. Then, we design a spatial-temporal Bayesian Gaussian kernel (BGK) inference method to dynamically evaluate traversability scores, integrating historical and real-time data while considering factors such as slope, flatness, gradient, and uncertainty metrics. GPU acceleration is applied in the feature extraction step, and the system achieves real-time performance. Extensive simulation experiments across diverse terrain scenarios demonstrate that our method outperforms SOTA approaches in both accuracy and computational efficiency. Additionally, we develop an autonomous navigation framework integrated with the traversability map and validate it with a differential driven vehicle in complex outdoor environments. Our code will be open-source for further research and development by the community, this https URL. 

**Abstract (ZH)**: 地形分析对于地面移动机器人在实际任务中的应用至关重要，特别是在户外未结构化环境中。本文提出了一种新颖的时空通达性评估方法，旨在使自主机器人能够有效导航通过复杂地形。我们的方法利用稀疏高斯过程（SGP）直接从点云扫描中提取几何特征（曲率、坡度、高程等），然后利用这些特征构建高分辨率局部通达性地图。随后，我们设计了一种时空贝叶斯高斯核（BGK）推断方法，以动态评估通达性评分，综合历史和实时数据，并考虑坡度、平坦度、坡度和不确定性指标等因素。在特征提取步骤中应用了GPU加速，系统实现了实时性能。在各种地形场景的广泛模拟实验中，我们的方法在准确性和计算效率方面均优于当前最佳方法。此外，我们开发了一种集成了通达性地图的自主导航框架，并在复杂户外环境中用差速驱动车辆进行了验证。我们的代码将开源供社区进一步研究和开发使用，this https URL。 

---
# DVM-SLAM: Decentralized Visual Monocular Simultaneous Localization and Mapping for Multi-Agent Systems 

**Title (ZH)**: 多代理系统中去中心化视觉单目同时定位与建图 

**Authors**: Joshua Bird, Jan Blumenkamp, Amanda Prorok  

**Link**: [PDF](https://arxiv.org/pdf/2503.04126)  

**Abstract**: Cooperative Simultaneous Localization and Mapping (C-SLAM) enables multiple agents to work together in mapping unknown environments while simultaneously estimating their own positions. This approach enhances robustness, scalability, and accuracy by sharing information between agents, reducing drift, and enabling collective exploration of larger areas. In this paper, we present Decentralized Visual Monocular SLAM (DVM-SLAM), the first open-source decentralized monocular C-SLAM system. By only utilizing low-cost and light-weight monocular vision sensors, our system is well suited for small robots and micro aerial vehicles (MAVs). DVM-SLAM's real-world applicability is validated on physical robots with a custom collision avoidance framework, showcasing its potential in real-time multi-agent autonomous navigation scenarios. We also demonstrate comparable accuracy to state-of-the-art centralized monocular C-SLAM systems. We open-source our code and provide supplementary material online. 

**Abstract (ZH)**: 协作 simultaneous localization and mapping (C-SLAM) 允许多个代理在测绘未知环境的同时共同估计各自的定位。该方法通过共享信息、减少漂移和实现更大区域的集体探索，增强了鲁棒性、可扩展性和准确性。在本文中，我们提出了去中心化单目视觉 SLAM (DVM-SLAM)，这是首个开源的去中心化单目 C-SLAM 系统。仅通过使用低成本和轻量级的单目视觉传感器，我们的系统非常适合小型机器人和微空中车辆 (MAVs)。DVM-SLAM 在具有自定义避碰框架的实际机器人上验证了其现实世界的应用性，并展示了其在实时多代理自主导航场景中的潜力。我们还展示了与最先进的集中式单目 C-SLAM 系统相当的准确性。我们在开源代码并在线提供补充材料。 

---
# GAGrasp: Geometric Algebra Diffusion for Dexterous Grasping 

**Title (ZH)**: GAGrasp: 几何代数扩散用于灵巧抓取 

**Authors**: Tao Zhong, Christine Allen-Blanchette  

**Link**: [PDF](https://arxiv.org/pdf/2503.04123)  

**Abstract**: We propose GAGrasp, a novel framework for dexterous grasp generation that leverages geometric algebra representations to enforce equivariance to SE(3) transformations. By encoding the SE(3) symmetry constraint directly into the architecture, our method improves data and parameter efficiency while enabling robust grasp generation across diverse object poses. Additionally, we incorporate a differentiable physics-informed refinement layer, which ensures that generated grasps are physically plausible and stable. Extensive experiments demonstrate the model's superior performance in generalization, stability, and adaptability compared to existing methods. Additional details at this https URL 

**Abstract (ZH)**: 我们提出了一种新的灵巧抓取生成框架GAGrasp，该框架利用几何代数表示来约束SE(3)变换的等变性。通过直接将SE(3)对称性约束编码到架构中，我们的方法在提高数据和参数效率的同时，能够在多种物体姿态下生成稳健的抓取。此外，我们还引入了一个可微物理导向的精修层，确保生成的抓取是物理上可实现且稳定的。广泛的经验表明，与现有方法相比，该模型在泛化能力、稳定性和适应性方面表现出更优的性能。更多信息请参见此链接：this https URL 

---
# The Spinning Blimp: Design and Control of a Novel Minimalist Aerial Vehicle Leveraging Rotational Dynamics and Locomotion 

**Title (ZH)**: 旋翼气球：基于旋转动力学和运动的新型 minimalist 航空车辆设计与控制 

**Authors**: Leonardo Santens, Diego S. D'Antonio, Shuhang Hou, David Saldaña  

**Link**: [PDF](https://arxiv.org/pdf/2503.04112)  

**Abstract**: This paper presents the Spinning Blimp, a novel lighter-than-air (LTA) aerial vehicle designed for low-energy stable flight. Utilizing an oblate spheroid helium balloon for buoyancy, the vehicle achieves minimal energy consumption while maintaining prolonged airborne states. The unique and low-cost design employs a passively arranged wing coupled with a propeller to induce a spinning behavior, providing inherent pendulum-like stabilization. We propose a control strategy that takes advantage of the continuous revolving nature of the spinning blimp to control translational motion. The cost-effectiveness of the vehicle makes it highly suitable for a variety of applications, such as patrolling, localization, air and turbulence monitoring, and domestic surveillance. Experimental evaluations affirm the design's efficacy and underscore its potential as a versatile and economically viable solution for aerial applications. 

**Abstract (ZH)**: 基于旋转行为的新型低能耗轻于空气航空器设计与控制研究 

---
# Image-Based Relocalization and Alignment for Long-Term Monitoring of Dynamic Underwater Environments 

**Title (ZH)**: 基于图像的再定位与对准在动态水下环境长期监测中的应用 

**Authors**: Beverley Gorry, Tobias Fischer, Michael Milford, Alejandro Fontan  

**Link**: [PDF](https://arxiv.org/pdf/2503.04096)  

**Abstract**: Effective monitoring of underwater ecosystems is crucial for tracking environmental changes, guiding conservation efforts, and ensuring long-term ecosystem health. However, automating underwater ecosystem management with robotic platforms remains challenging due to the complexities of underwater imagery, which pose significant difficulties for traditional visual localization methods. We propose an integrated pipeline that combines Visual Place Recognition (VPR), feature matching, and image segmentation on video-derived images. This method enables robust identification of revisited areas, estimation of rigid transformations, and downstream analysis of ecosystem changes. Furthermore, we introduce the SQUIDLE+ VPR Benchmark-the first large-scale underwater VPR benchmark designed to leverage an extensive collection of unstructured data from multiple robotic platforms, spanning time intervals from days to years. The dataset encompasses diverse trajectories, arbitrary overlap and diverse seafloor types captured under varying environmental conditions, including differences in depth, lighting, and turbidity. Our code is available at: this https URL 

**Abstract (ZH)**: 有效的水下生态系统监测对于追踪环境变化、指导保护努力并确保长期生态系统健康至关重要。然而，由于水下图像的复杂性给传统视觉定位方法带来了巨大挑战，使用机器人平台自动管理水下生态系统仍然具有挑战性。我们提出了一种结合视觉地方识别（VPR）、特征匹配和图像分割的集成管道。该方法能够稳健地识别 revisit 区域、估计刚性变换，并进行生态系统变化的下游分析。此外，我们引入了 SQUIDLE+ VPR 基准——首个利用多种机器人平台多年数据的大型水下 VPR 基准，涵盖从几天到几年的不同时段。数据集包括在不同环境条件下（包括深度、光照和浑浊度差异）捕捉到的多样轨迹和任意重叠的海床类型。我们的代码可在以下链接获取：this https URL。 

---
# OPG-Policy: Occluded Push-Grasp Policy Learning with Amodal Segmentation 

**Title (ZH)**: OPG-策略：基于无掩码分割的遮挡推握策略学习 

**Authors**: Hao Ding, Yiming Zeng, Zhaoliang Wan, Hui Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.04089)  

**Abstract**: Goal-oriented grasping in dense clutter, a fundamental challenge in robotics, demands an adaptive policy to handle occluded target objects and diverse configurations. Previous methods typically learn policies based on partially observable segments of the occluded target to generate motions. However, these policies often struggle to generate optimal motions due to uncertainties regarding the invisible portions of different occluded target objects across various scenes, resulting in low motion efficiency. To this end, we propose OPG-Policy, a novel framework that leverages amodal segmentation to predict occluded portions of the target and develop an adaptive push-grasp policy for cluttered scenarios where the target object is partially observed. Specifically, our approach trains a dedicated amodal segmentation module for diverse target objects to generate amodal masks. These masks and scene observations are mapped to the future rewards of grasp and push motion primitives via deep Q-learning to learn the motion critic. Afterward, the push and grasp motion candidates predicted by the critic, along with the relevant domain knowledge, are fed into the coordinator to generate the optimal motion implemented by the robot. Extensive experiments conducted in both simulated and real-world environments demonstrate the effectiveness of our approach in generating motion sequences for retrieving occluded targets, outperforming other baseline methods in success rate and motion efficiency. 

**Abstract (ZH)**: 目标导向的密集杂件抓取：一种机器人领域中的基本挑战，要求一种适应性的策略来处理遮挡的目标物体和多变的配置。"]), 一种新颖的方法：基于非可视分割的适应性推抓策略(OPG-Policy) 

---
# Music-Driven Legged Robots: Synchronized Walking to Rhythmic Beats 

**Title (ZH)**: 音乐驱动的 legged 机器人：与节奏节拍同步行走 

**Authors**: Taixian Hou, Yueqi Zhang, Xiaoyi Wei, Zhiyan Dong, Jiafu Yi, Peng Zhai, Lihua Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04063)  

**Abstract**: We address the challenge of effectively controlling the locomotion of legged robots by incorporating precise frequency and phase characteristics, which is often ignored in locomotion policies that do not account for the periodic nature of walking. We propose a hierarchical architecture that integrates a low-level phase tracker, oscillators, and a high-level phase modulator. This controller allows quadruped robots to walk in a natural manner that is synchronized with external musical rhythms. Our method generates diverse gaits across different frequencies and achieves real-time synchronization with music in the physical world. This research establishes a foundational framework for enabling real-time execution of accurate rhythmic motions in legged robots. Video is available at website: this https URL. 

**Abstract (ZH)**: 我们提出了一种分层架构，结合低层次相位追踪器、振荡器和高层次相位调节器，以有效地控制腿足机器人行走的运动，克服了未考虑行走周期性特征的运动策略中存在的问题。我们的方法在不同频率下生成多种步态，并在物理世界中实现了与音乐的实时同步。本研究为在腿足机器人中实现准确节拍运动的实时执行建立了基础框架。更多详情请参见网址：this https URL。 

---
# RA-DP: Rapid Adaptive Diffusion Policy for Training-Free High-frequency Robotics Replanning 

**Title (ZH)**: RA-DP: 快速自适应扩散策略用于无需训练的高频机器人重规划 

**Authors**: Xi Ye, Rui Heng Yang, Jun Jin, Yinchuan Li, Amir Rasouli  

**Link**: [PDF](https://arxiv.org/pdf/2503.04051)  

**Abstract**: Diffusion models exhibit impressive scalability in robotic task learning, yet they struggle to adapt to novel, highly dynamic environments. This limitation primarily stems from their constrained replanning ability: they either operate at a low frequency due to a time-consuming iterative sampling process, or are unable to adapt to unforeseen feedback in case of rapid replanning. To address these challenges, we propose RA-DP, a novel diffusion policy framework with training-free high-frequency replanning ability that solves the above limitations in adapting to unforeseen dynamic environments. Specifically, our method integrates guidance signals which are often easily obtained in the new environment during the diffusion sampling process, and utilizes a novel action queue mechanism to generate replanned actions at every denoising step without retraining, thus forming a complete training-free framework for robot motion adaptation in unseen environments. Extensive evaluations have been conducted in both well-recognized simulation benchmarks and real robot tasks. Results show that RA-DP outperforms the state-of-the-art diffusion-based methods in terms of replanning frequency and success rate. Moreover, we show that our framework is theoretically compatible with any training-free guidance signal. 

**Abstract (ZH)**: Diffusion模型在机器人任务学习中展现了令人 Impressive 的可扩展性，但它们在适应新型高度动态环境方面存在困难。这种限制主要源于它们受限的重新规划能力：它们要么因耗时的迭代采样过程而以低频运行，要么在需要快速重新规划的情况下无法适应未预见的反馈。为解决这些挑战，我们提出了RA-DP，这是一种具有无训练高频率重新规划能力的新型扩散策略框架，能够解决在适应未预见动态环境方面的上述限制。具体而言，我们的方法在扩散采样过程中整合了通常容易在新环境中获得的引导信号，并利用一种新的动作队列机制在去噪的每一步生成重新规划的动作，从而形成一个完整的无训练框架，用于机器人在未见过的环境中的动作适应。在公认的仿真基准和实际机器人任务中进行了广泛的评估。结果表明，RA-DP在重新规划频率和成功率方面优于现有的基于扩散的方法。此外，我们展示了我们的框架在理论上与任何无训练引导信号兼容。 

---
# Object State Estimation Through Robotic Active Interaction for Biological Autonomous Drilling 

**Title (ZH)**: 基于机器人主动交互的生物自主钻探状态估计 

**Authors**: Xiaofeng Lin, Enduo Zhao, Saúl Alexis Heredia Pérez, Kanako Harada  

**Link**: [PDF](https://arxiv.org/pdf/2503.04043)  

**Abstract**: Estimating the state of biological specimens is challenging due to limited observation through microscopic vision. For instance, during mouse skull drilling, the appearance alters little when thinning bone tissue because of its semi-transparent property and the high-magnification microscopic vision. To obtain the object's state, we introduce an object state estimation method for biological specimens through active interaction based on the deflection. The method is integrated to enhance the autonomous drilling system developed in our previous work. The method and integrated system were evaluated through 12 autonomous eggshell drilling experiment trials. The results show that the system achieved a 91.7% successful ratio and 75% detachable ratio, showcasing its potential applicability in more complex surgical procedures such as mouse skull craniotomy. This research paves the way for further development of autonomous robotic systems capable of estimating the object's state through active interaction. 

**Abstract (ZH)**: 通过主动交互基于偏转的生物标本状态估计方法及其在自主钻孔系统中的应用 

---
# Autonomous Robotic Bone Micro-Milling System with Automatic Calibration and 3D Surface Fitting 

**Title (ZH)**: 自主机器人骨微铣系统及其自动校准与三维表面拟合 

**Authors**: Enduo Zhao, Xiaofeng Lin, Yifan Wang, Kanako Harada  

**Link**: [PDF](https://arxiv.org/pdf/2503.04038)  

**Abstract**: Automating bone micro-milling using a robotic system presents challenges due to the uncertainties in both the external and internal features of bone tissue. For example, during a mouse cranial window creation, a circular path with a radius of 2 to 4 mm needs to be milled on the mouse skull using a microdrill. The uneven surface and non-uniform thickness of the mouse skull make it difficult to fully automate this process, requiring the system to possess advanced perceptual and adaptive capabilities. In this study, we propose an automatic calibration and 3D surface fitting method and integrate it into an autonomous robotic bone micro-milling system, enabling it to quickly, in real-time, and accurately perceive and adapt to the uneven surface and non-uniform thickness of the target without human assistance. Validation experiments on euthanized mice demonstrate that the improved system achieves a success rate of 85.7 % and an average milling time of 2.1 minutes, showing not only significant performance improvements over the previous system but also exceptional accuracy, speed, and stability compared to human operators. 

**Abstract (ZH)**: 利用机器人系统自动化骨微 milling 遇到了由于骨组织内外特征的不确定性所带来的挑战。例如，在创建小鼠颅窗时，需要使用微型钻在小鼠颅骨上铣出直径为 2 至 4 毫米的圆路径。小鼠颅骨不均匀的表面和非均匀的厚度使得完全自动化此过程变得困难，因此系统需要具备先进的感知和自适应能力。在本研究中，我们提出了一种自动校准和三维表面拟合方法，并将其集成到自主机器人骨微 milling 系统中，使其能够在没有人工辅助的情况下快速、实时地感知和适应目标的不均匀表面和非均匀厚度。实验表明，改进后的系统在麻醉小鼠上的成功率达到了 85.7%，平均加工时间为 2.1 分钟，不仅在性能上显著优于之前系统，在准确度、速度和稳定性方面也超过了人工操作者。 

---
# Dexterous Hand Manipulation via Efficient Imitation-Bootstrapped Online Reinforcement Learning 

**Title (ZH)**: Dexterous手部 manipulation 通过高效模仿-bootstrap在线强化学习 

**Authors**: Dongchi Huang, Tianle Zhang, Yihang Li, Ling Zhao, Jiayi Li, Zhirui Fang, Chunhe Xia, Lusong Li, Xiaodong He  

**Link**: [PDF](https://arxiv.org/pdf/2503.04014)  

**Abstract**: Dexterous hand manipulation in real-world scenarios presents considerable challenges due to its demands for both dexterity and precision. While imitation learning approaches have thoroughly examined these challenges, they still require a significant number of expert demonstrations and are limited by a constrained performance upper bound. In this paper, we propose a novel and efficient Imitation-Bootstrapped Online Reinforcement Learning (IBORL) method tailored for robotic dexterous hand manipulation in real-world environments. Specifically, we pretrain the policy using a limited set of expert demonstrations and subsequently finetune this policy through direct reinforcement learning in the real world. To address the catastrophic forgetting issues that arise from the distribution shift between expert demonstrations and real-world environments, we design a regularization term that balances the exploration of novel behaviors with the preservation of the pretrained policy. Our experiments with real-world tasks demonstrate that our method significantly outperforms existing approaches, achieving an almost 100% success rate and a 23% improvement in cycle time. Furthermore, by finetuning with online reinforcement learning, our method surpasses expert demonstrations and uncovers superior policies. Our code and empirical results are available in this https URL. 

**Abstract (ZH)**: 现实场景中灵巧手操作面临的挑战在于其对灵巧性和精确性的要求。尽管模仿学习方法已经充分研究了这些挑战，但它们仍然需要大量专家示范，并受到受限的性能上限的限制。本文提出了一种新的高效模仿引导在线强化学习（IBORL）方法，专门针对现实环境中灵巧手操作的机器人应用。具体而言，我们使用有限的专家示范进行策略预训练，并通过直接在现实世界中进行强化学习进行策略微调。为了解决由专家示范与现实环境分布变化引起的灾难性遗忘问题，我们设计了一种正则化项，该项平衡了探索新的行为与保留预训练策略之间的关系。我们的实验证明，该方法显著优于现有方法，成功率达到几乎100%，循环时间提高了23%。此外，通过在线强化学习进行策略微调，该方法超越了专家示范并发现了更优的策略。我们的代码和实验结果可通过以下链接访问。 

---
# Planning and Control for Deformable Linear Object Manipulation 

**Title (ZH)**: 可变形线性物体操作的规划与控制 

**Authors**: Burak Aksoy, John Wen  

**Link**: [PDF](https://arxiv.org/pdf/2503.04007)  

**Abstract**: Manipulating a deformable linear object (DLO) such as wire, cable, and rope is a common yet challenging task due to their high degrees of freedom and complex deformation behaviors, especially in an environment with obstacles. Existing local control methods are efficient but prone to failure in complex scenarios, while precise global planners are computationally intensive and difficult to deploy. This paper presents an efficient, easy-to-deploy framework for collision-free DLO manipulation using mobile manipulators. We demonstrate the effectiveness of leveraging standard planning tools for high-dimensional DLO manipulation without requiring custom planners or extensive data-driven models. Our approach combines an off-the-shelf global planner with a real-time local controller. The global planner approximates the DLO as a series of rigid links connected by spherical joints, enabling rapid path planning without the need for problem-specific planners or large datasets. The local controller employs control barrier functions (CBFs) to enforce safety constraints, maintain the DLO integrity, prevent overstress, and handle obstacle avoidance. It compensates for modeling inaccuracies by using a state-of-the-art position-based dynamics technique that approximates physical properties like Young's and shear moduli. We validate our framework through extensive simulations and real-world demonstrations. In complex obstacle scenarios-including tent pole transport, corridor navigation, and tasks requiring varied stiffness-our method achieves a 100% success rate over thousands of trials, with significantly reduced planning times compared to state-of-the-art techniques. Real-world experiments include transportation of a tent pole and a rope using mobile manipulators. We share our ROS-based implementation to facilitate adoption in various applications. 

**Abstract (ZH)**: 基于移动操作器的无障碍变形线性对象操纵高效易部署框架 

---
# Robotic Compliant Object Prying Using Diffusion Policy Guided by Vision and Force Observations 

**Title (ZH)**: 视觉和力感知引导的扩散策略用于柔顺物体撬取的机器人技术 

**Authors**: Jeon Ho Kang, Sagar Joshi, Ruopeng Huang, Satyandra K. Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2503.03998)  

**Abstract**: The growing adoption of batteries in the electric vehicle industry and various consumer products has created an urgent need for effective recycling solutions. These products often contain a mix of compliant and rigid components, making robotic disassembly a critical step toward achieving scalable recycling processes. Diffusion policy has emerged as a promising approach for learning low-level skills in robotics. To effectively apply diffusion policy to contact-rich tasks, incorporating force as feedback is essential. In this paper, we apply diffusion policy with vision and force in a compliant object prying task. However, when combining low-dimensional contact force with high-dimensional image, the force information may be diluted. To address this issue, we propose a method that effectively integrates force with image data for diffusion policy observations. We validate our approach on a battery prying task that demands high precision and multi-step execution. Our model achieves a 96\% success rate in diverse scenarios, marking a 57\% improvement over the vision-only baseline. Our method also demonstrates zero-shot transfer capability to handle unseen objects and battery types. Supplementary videos and implementation codes are available on our project website. this https URL 

**Abstract (ZH)**: 电动汽车行业和各种消费产品的电池应用越来越多，迫切需要有效的回收解决方案。这些产品通常包含柔性和刚性部件的混合，因此机器人拆卸成为实现可扩展回收过程的关键步骤。扩散策略已 emerges 作为一种有潜力的方法来学习机器人的低级技能。为了有效将扩散策略应用于富含接触的任务，引入力反馈是必不可少的。在本文中，我们在柔体物体分拣任务中应用结合视觉和力的扩散策略。然而，当将低维接触力与高维图像结合时，力信息可能会被稀释。为了解决这一问题，我们提出了一种有效整合力和图像数据的方法，以改善扩散策略的观察效果。我们通过一项需要高精度和多步骤执行的电池分拣任务验证了该方法。我们的模型在多种场景下实现了96%的成功率，比仅使用视觉的基线提高了57%。此外，我们的方法还展示了处理未见过的物体和电池类型的零样本迁移能力。有关补充视频和实现代码，可访问我们的项目网站：this https URL。 

---
# GeoFIK: A Fast and Reliable Geometric Solver for the IK of the Franka Arm based on Screw Theory Enabling Multiple Redundancy Parameters 

**Title (ZH)**: GeoFIK: 基于螺杆理论的Franka臂的快速可靠几何求解器及其在处理冗余参数方面的应用 

**Authors**: Pablo C. Lopez-Custodio, Yuhe Gong, Luis F.C. Figueredo  

**Link**: [PDF](https://arxiv.org/pdf/2503.03992)  

**Abstract**: Modern robotics applications require an inverse kinematics (IK) solver that is fast, robust and consistent, and that provides all possible solutions. Currently, the Franka robot arm is the most widely used manipulator in robotics research. With 7 DOFs, the IK of this robot is not only complex due to its 1-DOF redundancy, but also due to the link offsets at the wrist and elbow. Due to this complexity, none of the Franka IK solvers available in the literature provide satisfactory results when used in real-world applications. Therefore, in this paper we introduce GeoFIK (Geometric Franka IK), an analytical IK solver that allows the use of different joint variables to resolve the redundancy. The approach uses screw theory to describe the entire geometry of the robot, allowing the computation of the Jacobian matrix prior to computation of joint angles. All singularities are identified and handled. As an example of how the geometric elements obtained by the IK can be exploited, a solver with the swivel angle as the free variable is provided. Several experiments are carried out to validate the speed, robustness and reliability of the GeoFIK against two state-of-the-art solvers. 

**Abstract (ZH)**: 现代机器人应用需要一个快速、稳健且一致的逆运动学（IK）求解器，并能提供所有可能的解。目前， Franka 机器人臂是机器人研究中应用最广泛的 manipulator。由于其 1-DOF 冗余以及手腕和肘部的连杆偏移，即使有 7 个自由度，Franka 机器人的 IK 也是复杂的。由于这种复杂性，在文献中可用的所有 Franka IK 求解器在实际应用中都无法提供满意的结果。因此，本文提出了一种名为 GeoFIK（几何 Franka 逆运动学）的解析 IK 求解器，该求解器允许使用不同的关节变量来解决冗余问题。该方法使用螺丝理论描述整个机器人的几何结构，允许在计算关节角之前计算雅可比矩阵。所有奇异性都被识别并处理。作为通过 IK 所获得的几何元素的应用示例，提供了一个以俯仰角作为自由变量的求解器。进行了多项实验，以验证 GeoFIK 在速度、稳健性和可靠性方面与两个最先进的求解器之间的对比。 

---
# GRaD-Nav: Efficiently Learning Visual Drone Navigation with Gaussian Radiance Fields and Differentiable Dynamics 

**Title (ZH)**: GRaD-Nav: 通过高斯辐射场和可微动力学高效学习视觉无人机导航 

**Authors**: Qianzhong Chen, Jiankai Sun, Naixiang Gao, JunEn Low, Timothy Chen, Mac Schwager  

**Link**: [PDF](https://arxiv.org/pdf/2503.03984)  

**Abstract**: Autonomous visual navigation is an essential element in robot autonomy. Reinforcement learning (RL) offers a promising policy training paradigm. However existing RL methods suffer from high sample complexity, poor sim-to-real transfer, and limited runtime adaptability to navigation scenarios not seen during training. These problems are particularly challenging for drones, with complex nonlinear and unstable dynamics, and strong dynamic coupling between control and perception. In this paper, we propose a novel framework that integrates 3D Gaussian Splatting (3DGS) with differentiable deep reinforcement learning (DDRL) to train vision-based drone navigation policies. By leveraging high-fidelity 3D scene representations and differentiable simulation, our method improves sample efficiency and sim-to-real transfer. Additionally, we incorporate a Context-aided Estimator Network (CENet) to adapt to environmental variations at runtime. Moreover, by curriculum training in a mixture of different surrounding environments, we achieve in-task generalization, the ability to solve new instances of a task not seen during training. Drone hardware experiments demonstrate our method's high training efficiency compared to state-of-the-art RL methods, zero shot sim-to-real transfer for real robot deployment without fine tuning, and ability to adapt to new instances within the same task class (e.g. to fly through a gate at different locations with different distractors in the environment). 

**Abstract (ZH)**: 自主视觉导航是机器人自主性的重要组成部分。深度可微强化学习（DDRL）结合3D高保真场景表示与3D高斯点扩散（3DGS）的框架为基于视觉的无人机导航策略训练提供了有前途的方法。通过利用高保真3D场景表示和可微分模拟，我们的方法提高了样本效率并增强了从仿真到现实的转移能力。此外，我们引入了上下文辅助估计网络（CENet）以实现运行时的环境适应性。通过在不同环境混合中进行递增训练，我们实现了任务内泛化，即在训练中未见过的新任务实例上的解决能力。无人机硬件实验表明，我们的方法相较于最先进的RL方法具有更高的训练效率，不需要微调即可实现零样本从仿真到现实的转移部署，并且能够在相同任务类别中适应新的实例（例如，在不同位置且环境中存在不同干扰物的情况下通过门）。 

---
# Equivariant Filter Design for Range-only SLAM 

**Title (ZH)**: 只范围测程SLAM的等变滤波器设计 

**Authors**: Yixiao Ge, Arthur Pearce, Pieter van Goor, Robert Mahony  

**Link**: [PDF](https://arxiv.org/pdf/2503.03973)  

**Abstract**: Range-only Simultaneous Localisation and Mapping (RO-SLAM) is of interest due to its practical applications in ultra-wideband (UWB) and Bluetooth Low Energy (BLE) localisation in terrestrial and aerial applications and acoustic beacon localisation in submarine applications. In this work, we consider a mobile robot equipped with an inertial measurement unit (IMU) and a range sensor that measures distances to a collection of fixed landmarks. We derive an equivariant filter (EqF) for the RO-SLAM problem based on a symmetry Lie group that is compatible with the range measurements. The proposed filter does not require bootstrapping or initialisation of landmark positions, and demonstrates robustness to the no-prior situation. The filter is demonstrated on a real-world dataset, and it is shown to significantly outperform a state-of-the-art EKF alternative in terms of both accuracy and robustness. 

**Abstract (ZH)**: 仅距离同时定位与建图（RO-SLAM）在超宽带（UWB）和蓝牙低功耗（BLE）定位及水下声学信标定位等领域的实际应用中引起了兴趣。在此工作中，我们考虑配备惯性测量单元（IMU）和距离传感器的移动机器人，该传感器测量到一组固定地标点的距离。我们基于与距离测量相兼容的对称李群推导了仅距离同时定位与建图问题的不变滤波器（EqF）。所提出的滤波器不需要地标位置的回环检测或初始化，并展示了在无先验情况下具有良好的鲁棒性。该滤波器在真实数据集上进行了演示，并且在准确性和鲁棒性方面均显著优于最先进的扩展卡尔曼滤波器（EKF）替代方案。 

---
# Enhancing Autonomous Driving Safety with Collision Scenario Integration 

**Title (ZH)**: 基于碰撞场景集成的自动驾驶安全性提升 

**Authors**: Zi Wang, Shiyi Lan, Xinglong Sun, Nadine Chang, Zhenxin Li, Zhiding Yu, Jose M. Alvarez  

**Link**: [PDF](https://arxiv.org/pdf/2503.03957)  

**Abstract**: Autonomous vehicle safety is crucial for the successful deployment of self-driving cars. However, most existing planning methods rely heavily on imitation learning, which limits their ability to leverage collision data effectively. Moreover, collecting collision or near-collision data is inherently challenging, as it involves risks and raises ethical and practical concerns. In this paper, we propose SafeFusion, a training framework to learn from collision data. Instead of over-relying on imitation learning, SafeFusion integrates safety-oriented metrics during training to enable collision avoidance learning. In addition, to address the scarcity of collision data, we propose CollisionGen, a scalable data generation pipeline to generate diverse, high-quality scenarios using natural language prompts, generative models, and rule-based filtering. Experimental results show that our approach improves planning performance in collision-prone scenarios by 56\% over previous state-of-the-art planners while maintaining effectiveness in regular driving situations. Our work provides a scalable and effective solution for advancing the safety of autonomous driving systems. 

**Abstract (ZH)**: 自主驾驶车辆的安全性对于其成功部署至关重要。然而，现有大多数规划方法严重依赖于模仿学习，这限制了它们有效利用碰撞数据的能力。此外，收集碰撞或近碰撞数据固有地具有挑战性，因为它涉及风险，并引发伦理和实际问题。在这项研究中，我们提出SafeFusion，一种从碰撞数据中学习的训练框架。SafeFusion 不过分依赖模仿学习，而是通过训练整合安全导向的指标，以实现碰撞避免学习。此外，为了解决碰撞数据稀缺的问题，我们提出了CollisionGen，这是一种可扩展的数据生成管道，利用自然语言提示、生成模型和基于规则的过滤生成多样化、高质量的场景。实验结果表明，与之前最先进的规划器相比，我们的方法在碰撞频发场景中的规划性能提高了56%，同时在常规驾驶情况下保持了有效性。我们的工作提供了一种可扩展且有效的解决方案，以促进自主驾驶系统的安全性。 

---
# CREStE: Scalable Mapless Navigation with Internet Scale Priors and Counterfactual Guidance 

**Title (ZH)**: CREStE: 基于互联网规模先验和假设性指导的可扩展无地图导航方法 

**Authors**: Arthur Zhang, Harshit Sikchi, Amy Zhang, Joydeep Biswas  

**Link**: [PDF](https://arxiv.org/pdf/2503.03921)  

**Abstract**: We address the long-horizon mapless navigation problem: enabling robots to traverse novel environments without relying on high-definition maps or precise waypoints that specify exactly where to navigate. Achieving this requires overcoming two major challenges -- learning robust, generalizable perceptual representations of the environment without pre-enumerating all possible navigation factors and forms of perceptual aliasing and utilizing these learned representations to plan human-aligned navigation paths. Existing solutions struggle to generalize due to their reliance on hand-curated object lists that overlook unforeseen factors, end-to-end learning of navigation features from scarce large-scale robot datasets, and handcrafted reward functions that scale poorly to diverse scenarios. To overcome these limitations, we propose CREStE, the first method that learns representations and rewards for addressing the full mapless navigation problem without relying on large-scale robot datasets or manually curated features. CREStE leverages visual foundation models trained on internet-scale data to learn continuous bird's-eye-view representations capturing elevation, semantics, and instance-level features. To utilize learned representations for planning, we propose a counterfactual-based loss and active learning procedure that focuses on the most salient perceptual cues by querying humans for counterfactual trajectory annotations in challenging scenes. We evaluate CREStE in kilometer-scale navigation tasks across six distinct urban environments. CREStE significantly outperforms all state-of-the-art approaches with 70% fewer human interventions per mission, including a 2-kilometer mission in an unseen environment with just 1 intervention; showcasing its robustness and effectiveness for long-horizon mapless navigation. For videos and additional materials, see this https URL . 

**Abstract (ZH)**: 无地图长期规划导航问题：使机器人能够在不依赖高精度地图或精确航点的情况下穿越未知环境 

---
# GO-VMP: Global Optimization for View Motion Planning in Fruit Mapping 

**Title (ZH)**: GO-VMP: 全局优化在水果测绘中的视图运动规划 

**Authors**: Allen Isaac Jose, Sicong Pan, Tobias Zaenker, Rohit Menon, Sebastian Houben, Maren Bennewitz  

**Link**: [PDF](https://arxiv.org/pdf/2503.03912)  

**Abstract**: Automating labor-intensive tasks such as crop monitoring with robots is essential for enhancing production and conserving resources. However, autonomously monitoring horticulture crops remains challenging due to their complex structures, which often result in fruit occlusions. Existing view planning methods attempt to reduce occlusions but either struggle to achieve adequate coverage or incur high robot motion costs. We introduce a global optimization approach for view motion planning that aims to minimize robot motion costs while maximizing fruit coverage. To this end, we leverage coverage constraints derived from the set covering problem (SCP) within a shortest Hamiltonian path problem (SHPP) formulation. While both SCP and SHPP are well-established, their tailored integration enables a unified framework that computes a global view path with minimized motion while ensuring full coverage of selected targets. Given the NP-hard nature of the problem, we employ a region-prior-based selection of coverage targets and a sparse graph structure to achieve effective optimization outcomes within a limited time. Experiments in simulation demonstrate that our method detects more fruits, enhances surface coverage, and achieves higher volume accuracy than the motion-efficient baseline with a moderate increase in motion cost, while significantly reducing motion costs compared to the coverage-focused baseline. Real-world experiments further confirm the practical applicability of our approach. 

**Abstract (ZH)**: 利用机器人自动化劳动密集型任务如作物监测对于提升生产效率和节约资源至关重要，但由于园艺作物结构复杂常导致果实遮挡，自主监测园艺作物仍然具有挑战性。我们提出了一种全局优化视角运动规划方法，旨在最小化机器人运动成本同时最大化果实覆盖率。为此，我们在最短哈密尔顿路径问题（SHPP）建模中利用来自集合覆盖问题（SCP）的覆盖约束。尽管SCP和SHPP都是成熟的数学模型，但它们的结合使我们能够提出一个统一框架，该框架能在确保覆盖选定目标的同时计算出最小化运动成本的全局视角路径。由于问题是NP难的，我们采用了基于区域优先的选择覆盖目标和稀疏图结构，以在有限时间内实现有效的优化效果。模拟实验结果显示，与运动效率基线相比，我们的方法能够在适度增加运动成本的同时检测更多果实、提高表面覆盖度和实现更高的体积精度，而且与以覆盖为目标的基线相比，显著降低了运动成本。实地实验进一步验证了我们方法的实用性。 

---
# Safe LLM-Controlled Robots with Formal Guarantees via Reachability Analysis 

**Title (ZH)**: 具有可达性分析形式保障的safe LLM控制机器人 

**Authors**: Ahmad Hafez, Alireza Naderi Akhormeh, Amr Hegazy, Amr Alanwar  

**Link**: [PDF](https://arxiv.org/pdf/2503.03911)  

**Abstract**: The deployment of Large Language Models (LLMs) in robotic systems presents unique safety challenges, particularly in unpredictable environments. Although LLMs, leveraging zero-shot learning, enhance human-robot interaction and decision-making capabilities, their inherent probabilistic nature and lack of formal guarantees raise significant concerns for safety-critical applications. Traditional model-based verification approaches often rely on precise system models, which are difficult to obtain for real-world robotic systems and may not be fully trusted due to modeling inaccuracies, unmodeled dynamics, or environmental uncertainties. To address these challenges, this paper introduces a safety assurance framework for LLM-controlled robots based on data-driven reachability analysis, a formal verification technique that ensures all possible system trajectories remain within safe operational limits. Our framework specifically investigates the problem of instructing an LLM to navigate the robot to a specified goal and assesses its ability to generate low-level control actions that successfully guide the robot safely toward that goal. By leveraging historical data to construct reachable sets of states for the robot-LLM system, our approach provides rigorous safety guarantees against unsafe behaviors without relying on explicit analytical models. We validate the framework through experimental case studies in autonomous navigation and task planning, demonstrating its effectiveness in mitigating risks associated with LLM-generated commands. This work advances the integration of formal methods into LLM-based robotics, offering a principled and practical approach to ensuring safety in next-generation autonomous systems. 

**Abstract (ZH)**: 大型语言模型在机器人系统中的部署提出了独特的安全挑战，尤其是在不可预测的环境中。虽然大型语言模型利用零-shot学习增强人机交互和决策能力，但它们固有的概率性质和缺乏正式保证使得它们在关键安全应用中存在重大隐患。传统的基于模型的验证方法通常依赖于精确的系统模型，但在实际的机器人系统中很难获取这些模型，并且由于建模不准确、未建模的动力学或环境不确定性，这些模型可能无法完全信赖。为了解决这些挑战，本文提出了一种基于数据驱动可达性分析的安全保证框架，这是一种形式验证技术，确保所有可能的系统轨迹都在安全操作范围内。该框架具体研究了指导大型语言模型导航机器人至指定目标的问题，并评估了其生成低级控制动作以安全引导机器人至目标的能力。通过利用历史数据构建机器人-大型语言模型系统的可达状态集，我们的方法在不依赖显式分析模型的情况下提供了严格的安全保证，防止出现不安全行为。我们通过自主导航和任务规划的实验案例研究验证了该框架，展示了其在缓解由大型语言模型生成的命令引起的风险方面的有效性。这项工作推进了形式方法在大型语言模型驱动的机器人中的集成，提供了一种原理性和实用的方法确保下一代自主系统的安全性。 

---
# LensDFF: Language-enhanced Sparse Feature Distillation for Efficient Few-Shot Dexterous Manipulation 

**Title (ZH)**: LensDFF: 语言增强的稀疏特征蒸馏用于高效的少样本灵巧 manipulation 

**Authors**: Qian Feng, David S. Martinez Lema, Jianxiang Feng, Zhaopeng Chen, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2503.03890)  

**Abstract**: Learning dexterous manipulation from few-shot demonstrations is a significant yet challenging problem for advanced, human-like robotic systems. Dense distilled feature fields have addressed this challenge by distilling rich semantic features from 2D visual foundation models into the 3D domain. However, their reliance on neural rendering models such as Neural Radiance Fields (NeRF) or Gaussian Splatting results in high computational costs. In contrast, previous approaches based on sparse feature fields either suffer from inefficiencies due to multi-view dependencies and extensive training or lack sufficient grasp dexterity. To overcome these limitations, we propose Language-ENhanced Sparse Distilled Feature Field (LensDFF), which efficiently distills view-consistent 2D features onto 3D points using our novel language-enhanced feature fusion strategy, thereby enabling single-view few-shot generalization. Based on LensDFF, we further introduce a few-shot dexterous manipulation framework that integrates grasp primitives into the demonstrations to generate stable and highly dexterous grasps. Moreover, we present a real2sim grasp evaluation pipeline for efficient grasp assessment and hyperparameter tuning. Through extensive simulation experiments based on the real2sim pipeline and real-world experiments, our approach achieves competitive grasping performance, outperforming state-of-the-art approaches. 

**Abstract (ZH)**: 从少量示范中学习灵巧操作是先进、类人机器人系统面临的一个重要而具有挑战性的问题。密集提炼特征场通过将丰富的语义特征从2D视觉基础模型提炼到3D领域，解决了这一挑战。然而，它们依赖于神经渲染模型如神经辐射场（NeRF）或高斯点积带来的高计算成本。相比之下，基于稀疏特征场的先前方法要么由于多视图依赖性和广泛的训练而导致效率低下，要么缺乏足够的抓取灵巧性。为克服这些限制，我们提出了语言增强稀疏提炼特征场（LensDFF），该方法使用我们新颖的语言增强特征融合策略，高效地将一致的2D特征投射到3D点上，从而实现单视图少量示范泛化。基于LensDFF，我们进一步提出了一种整合抓取原始操作的少量示范灵巧操作框架，以生成稳定且高度灵巧的抓取。此外，我们提出了一个高效的抓取评估和超参数调整的实2仿抓取评估管道。通过基于实2仿管道和真实世界实验的广泛模拟实验，我们的方法实现了竞争性的抓取性能，并且在某些方面优于现有最佳方法。 

---
# Pretrained LLMs as Real-Time Controllers for Robot Operated Serial Production Line 

**Title (ZH)**: 预训练大语言模型作为实时控制器用于机器人操作的连续生产线 

**Authors**: Muhammad Waseem, Kshitij Bhatta, Chen Li, Qing Chang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03889)  

**Abstract**: The manufacturing industry is undergoing a transformative shift, driven by cutting-edge technologies like 5G, AI, and cloud computing. Despite these advancements, effective system control, which is crucial for optimizing production efficiency, remains a complex challenge due to the intricate, knowledge-dependent nature of manufacturing processes and the reliance on domain-specific expertise. Conventional control methods often demand heavy customization, considerable computational resources, and lack transparency in decision-making. In this work, we investigate the feasibility of using Large Language Models (LLMs), particularly GPT-4, as a straightforward, adaptable solution for controlling manufacturing systems, specifically, mobile robot scheduling. We introduce an LLM-based control framework to assign mobile robots to different machines in robot assisted serial production lines, evaluating its performance in terms of system throughput. Our proposed framework outperforms traditional scheduling approaches such as First-Come-First-Served (FCFS), Shortest Processing Time (SPT), and Longest Processing Time (LPT). While it achieves performance that is on par with state-of-the-art methods like Multi-Agent Reinforcement Learning (MARL), it offers a distinct advantage by delivering comparable throughput without the need for extensive retraining. These results suggest that the proposed LLM-based solution is well-suited for scenarios where technical expertise, computational resources, and financial investment are limited, while decision transparency and system scalability are critical concerns. 

**Abstract (ZH)**: 制造业正经历由5G、AI和云计算等前沿技术驱动的转型。尽管取得了这些进展，有效系统控制——这对于优化生产效率至关重要——依然是一个复杂挑战，因其制造业过程的复杂性和对领域专长的依赖。传统控制方法往往需要大量定制、较多的计算资源，并且在决策透明度方面存在不足。在本研究中，我们探讨使用大型语言模型（LLMs），特别是GPT-4，作为控制制造系统的简便且具有弹性的解决方案，特别应用于协作机器人调度。我们提出了一种基于LLM的控制框架，用于在协作机器人辅助的流水线中分配移动机器人，并评估其在系统吞吐量方面的性能。我们提出的框架在系统吞吐量方面优于传统的调度方法，如先到先服务（FCFS）、最短处理时间（SPT）和最长处理时间（LPT）。它在性能上与最先进的方法（如多代理强化学习MARL）相当，但通过不需大量重新训练即可提供相当的吞吐量，展现出明显的优势。这些结果表明，所提出的基于LLM的解决方案适用于技术专长、计算资源和财务投资有限而决策透明度和系统扩展性是关键考虑的场景。 

---
# Floxels: Fast Unsupervised Voxel Based Scene Flow Estimation 

**Title (ZH)**: Floxels: 快速无监督体素基场景流估计 

**Authors**: David T. Hoffmann, Syed Haseeb Raza, Hanqiu Jiang, Denis Tananaev, Steffen Klingenhoefer, Martin Meinke  

**Link**: [PDF](https://arxiv.org/pdf/2503.04718)  

**Abstract**: Scene flow estimation is a foundational task for many robotic applications, including robust dynamic object detection, automatic labeling, and sensor synchronization. Two types of approaches to the problem have evolved: 1) Supervised and 2) optimization-based methods. Supervised methods are fast during inference and achieve high-quality results, however, they are limited by the need for large amounts of labeled training data and are susceptible to domain gaps. In contrast, unsupervised test-time optimization methods do not face the problem of domain gaps but usually suffer from substantial runtime, exhibit artifacts, or fail to converge to the right solution. In this work, we mitigate several limitations of existing optimization-based methods. To this end, we 1) introduce a simple voxel grid-based model that improves over the standard MLP-based formulation in multiple dimensions and 2) introduce a new multiframe loss formulation. 3) We combine both contributions in our new method, termed Floxels. On the Argoverse 2 benchmark, Floxels is surpassed only by EulerFlow among unsupervised methods while achieving comparable performance at a fraction of the computational cost. Floxels achieves a massive speedup of more than ~60 - 140x over EulerFlow, reducing the runtime from a day to 10 minutes per sequence. Over the faster but low-quality baseline, NSFP, Floxels achieves a speedup of ~14x. 

**Abstract (ZH)**: 基于场景流估计的 rob 领域应用基础任务，包括稳健的动力学对象检测、自动标注和传感器同步。该问题演化出两种方法：1）监督学习方法和2）优化基方法。监督学习方法在推理时速度快，能够达到高质量的结果，但需要大量标注训练数据，并且容易受到领域差异的影响。相比之下，无监督的测试时优化方法不受领域差异问题的困扰，但通常运行时耗时较长、产生伪影或未能收敛到正确的解。在此项工作中，我们减轻了现有优化基方法的若干局限性。为此，我们1）引入一种基于简单体素格网的模型，该模型在多个维度上改进了标准的基于MLP的表述形式；2）引入一种新的多帧损失表述形式；3）将上述两个贡献结合在一起，提出了一种称为Floxeles的新方法。在Argoverse 2基准测试中，Floxeles在无监督方法中仅次于EulerFlow，而在计算成本仅为后者的一小部分的情况下，实现了可比的性能。Floxeles相比EulerFlow实现了超过60-140倍的加速，将运行时间从一天缩短到每个序列10分钟。与更快但质量较低的基线NSFP相比，Floxeles实现了约14倍的加速。 

---
# Multi-Agent Inverse Q-Learning from Demonstrations 

**Title (ZH)**: 基于演示的多智能体逆Q学习 

**Authors**: Nathaniel Haynam, Adam Khoja, Dhruv Kumar, Vivek Myers, Erdem Bıyık  

**Link**: [PDF](https://arxiv.org/pdf/2503.04679)  

**Abstract**: When reward functions are hand-designed, deep reinforcement learning algorithms often suffer from reward misspecification, causing them to learn suboptimal policies in terms of the intended task objectives. In the single-agent case, inverse reinforcement learning (IRL) techniques attempt to address this issue by inferring the reward function from expert demonstrations. However, in multi-agent problems, misalignment between the learned and true objectives is exacerbated due to increased environment non-stationarity and variance that scales with multiple agents. As such, in multi-agent general-sum games, multi-agent IRL algorithms have difficulty balancing cooperative and competitive objectives. To address these issues, we propose Multi-Agent Marginal Q-Learning from Demonstrations (MAMQL), a novel sample-efficient framework for multi-agent IRL. For each agent, MAMQL learns a critic marginalized over the other agents' policies, allowing for a well-motivated use of Boltzmann policies in the multi-agent context. We identify a connection between optimal marginalized critics and single-agent soft-Q IRL, allowing us to apply a direct, simple optimization criterion from the single-agent domain. Across our experiments on three different simulated domains, MAMQL significantly outperforms previous multi-agent methods in average reward, sample efficiency, and reward recovery by often more than 2-5x. We make our code available at this https URL . 

**Abstract (ZH)**: 多智能体边际Q学习从演示中发现奖励（MAMQL）：一种多智能体逆强化学习的新范式 

---
# Omnidirectional Multi-Object Tracking 

**Title (ZH)**: 全方位多目标跟踪 

**Authors**: Kai Luo, Hao Shi, Sheng Wu, Fei Teng, Mengfei Duan, Chang Huang, Yuhang Wang, Kaiwei Wang, Kailun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04565)  

**Abstract**: Panoramic imagery, with its 360° field of view, offers comprehensive information to support Multi-Object Tracking (MOT) in capturing spatial and temporal relationships of surrounding objects. However, most MOT algorithms are tailored for pinhole images with limited views, impairing their effectiveness in panoramic settings. Additionally, panoramic image distortions, such as resolution loss, geometric deformation, and uneven lighting, hinder direct adaptation of existing MOT methods, leading to significant performance degradation. To address these challenges, we propose OmniTrack, an omnidirectional MOT framework that incorporates Tracklet Management to introduce temporal cues, FlexiTrack Instances for object localization and association, and the CircularStatE Module to alleviate image and geometric distortions. This integration enables tracking in large field-of-view scenarios, even under rapid sensor motion. To mitigate the lack of panoramic MOT datasets, we introduce the QuadTrack dataset--a comprehensive panoramic dataset collected by a quadruped robot, featuring diverse challenges such as wide fields of view, intense motion, and complex environments. Extensive experiments on the public JRDB dataset and the newly introduced QuadTrack benchmark demonstrate the state-of-the-art performance of the proposed framework. OmniTrack achieves a HOTA score of 26.92% on JRDB, representing an improvement of 3.43%, and further achieves 23.45% on QuadTrack, surpassing the baseline by 6.81%. The dataset and code will be made publicly available at this https URL. 

**Abstract (ZH)**: 全景图像因其360°的视野，提供了支持多目标跟踪（MOT）所需的全面信息，用于捕捉周围对象的空间和时间关系。然而，大多数MOT算法都是针对具有有限视角的针孔图像定制的，在全景设置中的效果不佳。此外，全景图像失真，如分辨率损失、几何变形和不均匀光照，阻碍了现有MOT方法的直接应用，导致性能显著下降。为了解决这些挑战，我们提出了 OmniTrack，这是一种结合了 Tracklet 管理引入时间线索、FlexiTrack 实例进行目标定位和关联以及 CircularStatE 模块以缓解图像和几何失真的全方位MOT框架。这种集成使得即使在传感器快速移动的情况下也能在大视野场景中进行跟踪。为了解决全景MOT数据集缺乏的问题，我们引入了QuadTrack数据集——由四足机器人收集的全面全景数据集，包含广泛的挑战，如宽广的视野、剧烈的运动和复杂的环境。在公共JRDB数据集和新引入的QuadTrack基准上的广泛实验展示了所提出框架的最先进的性能。OmniTrack在JRDB上的HOTA得分为26.92%，相比基线提高了3.43%，进一步在QuadTrack上达到23.45%，超过了基线6.81%。数据集和代码将在以下网址公开：this https URL。 

---
# ForestLPR: LiDAR Place Recognition in Forests Attentioning Multiple BEV Density Images 

**Title (ZH)**: ForestLPR：关注多BEV密度图像的森林LiDAR场所识别 

**Authors**: Yanqing Shen, Turcan Tuna, Marco Hutter, Cesar Cadena, Nanning Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.04475)  

**Abstract**: Place recognition is essential to maintain global consistency in large-scale localization systems. While research in urban environments has progressed significantly using LiDARs or cameras, applications in natural forest-like environments remain largely under-explored. Furthermore, forests present particular challenges due to high self-similarity and substantial variations in vegetation growth over time. In this work, we propose a robust LiDAR-based place recognition method for natural forests, ForestLPR. We hypothesize that a set of cross-sectional images of the forest's geometry at different heights contains the information needed to recognize revisiting a place. The cross-sectional images are represented by \ac{bev} density images of horizontal slices of the point cloud at different heights. Our approach utilizes a visual transformer as the shared backbone to produce sets of local descriptors and introduces a multi-BEV interaction module to attend to information at different heights adaptively. It is followed by an aggregation layer that produces a rotation-invariant place descriptor. We evaluated the efficacy of our method extensively on real-world data from public benchmarks as well as robotic datasets and compared it against the state-of-the-art (SOTA) methods. The results indicate that ForestLPR has consistently good performance on all evaluations and achieves an average increase of 7.38\% and 9.11\% on Recall@1 over the closest competitor on intra-sequence loop closure detection and inter-sequence re-localization, respectively, validating our hypothesis 

**Abstract (ZH)**: 基于LiDAR的自然森林场所识别方法ForestLPR 

---
# Instrument-Splatting: Controllable Photorealistic Reconstruction of Surgical Instruments Using Gaussian Splatting 

**Title (ZH)**: 手术器械渲染：基于高斯点云的可控逼真重建 

**Authors**: Shuojue Yang, Zijian Wu, Mingxuan Hong, Qian Li, Daiyun Shen, Septimiu E. Salcudean, Yueming Jin  

**Link**: [PDF](https://arxiv.org/pdf/2503.04082)  

**Abstract**: Real2Sim is becoming increasingly important with the rapid development of surgical artificial intelligence (AI) and autonomy. In this work, we propose a novel Real2Sim methodology, \textit{Instrument-Splatting}, that leverages 3D Gaussian Splatting to provide fully controllable 3D reconstruction of surgical instruments from monocular surgical videos. To maintain both high visual fidelity and manipulability, we introduce a geometry pre-training to bind Gaussian point clouds on part mesh with accurate geometric priors and define a forward kinematics to control the Gaussians as flexible as real instruments. Afterward, to handle unposed videos, we design a novel instrument pose tracking method leveraging semantics-embedded Gaussians to robustly refine per-frame instrument poses and joint states in a render-and-compare manner, which allows our instrument Gaussian to accurately learn textures and reach photorealistic rendering. We validated our method on 2 publicly released surgical videos and 4 videos collected on ex vivo tissues and green screens. Quantitative and qualitative evaluations demonstrate the effectiveness and superiority of the proposed method. 

**Abstract (ZH)**: Real2Sim 方法在单目手术视频中的手术器械三维重建：Instrument-Splatting 

---
# COARSE: Collaborative Pseudo-Labeling with Coarse Real Labels for Off-Road Semantic Segmentation 

**Title (ZH)**: COARSE: 基于粗粒度真实标签的协作伪标签生成方法在离路语义分割中的应用 

**Authors**: Aurelio Noca, Xianmei Lei, Jonathan Becktor, Jeffrey Edlund, Anna Sabel, Patrick Spieler, Curtis Padgett, Alexandre Alahi, Deegan Atha  

**Link**: [PDF](https://arxiv.org/pdf/2503.03947)  

**Abstract**: Autonomous off-road navigation faces challenges due to diverse, unstructured environments, requiring robust perception with both geometric and semantic understanding. However, scarce densely labeled semantic data limits generalization across domains. Simulated data helps, but introduces domain adaptation issues. We propose COARSE, a semi-supervised domain adaptation framework for off-road semantic segmentation, leveraging sparse, coarse in-domain labels and densely labeled out-of-domain data. Using pretrained vision transformers, we bridge domain gaps with complementary pixel-level and patch-level decoders, enhanced by a collaborative pseudo-labeling strategy on unlabeled data. Evaluations on RUGD and Rellis-3D datasets show significant improvements of 9.7\% and 8.4\% respectively, versus only using coarse data. Tests on real-world off-road vehicle data in a multi-biome setting further demonstrate COARSE's applicability. 

**Abstract (ZH)**: 野外自主导航面临的挑战在于多样且结构不一的环境，要求具有几何和语义理解的稳健感知。然而，稀少且密集标注的语义数据限制了其跨领域的泛化能力。模拟数据有助于此问题，但会引入领域适应问题。我们提出COARSE，一种利用稀疏但在域数据和密集标注的跨域数据的半监督领域适应框架，通过预训练的视觉变换器和互补的像素级和块级解码器，结合协作性的伪标签策略，在未标注数据上增强领域间隙的桥梁。在RUGD和Rellis-3D数据集上的评估结果显示，与仅使用稀疏数据相比，精度分别提升了9.7%和8.4%。在多种生物群落的实地车辆数据测试中进一步证明了COARSE的应用前景。 

---
# Endpoint-Explicit Differential Dynamic Programming via Exact Resolution 

**Title (ZH)**: 端点显式差分动态规划通过精确分辨率 

**Authors**: Maria Parilli, Sergi Martinez, Carlos Mastalli  

**Link**: [PDF](https://arxiv.org/pdf/2503.03897)  

**Abstract**: We introduce a novel method for handling endpoint constraints in constrained differential dynamic programming (DDP). Unlike existing approaches, our method guarantees quadratic convergence and is exact, effectively managing rank deficiencies in both endpoint and stagewise equality constraints. It is applicable to both forward and inverse dynamics formulations, making it particularly well-suited for model predictive control (MPC) applications and for accelerating optimal control (OC) solvers. We demonstrate the efficacy of our approach across a broad range of robotics problems and provide a user-friendly open-source implementation within CROCODDYL. 

**Abstract (ZH)**: 我们提出了一种在约束差分动力规划（DDP）中处理端点约束的新方法。与现有方法不同，该方法保证了二次收敛性和精确性，有效管理了端点和阶段等式约束中的秩缺陷。该方法适用于正向和逆向动力学 formulations，特别适合模型预测控制（MPC）应用，并能加速最优控制（OC）求解器。我们通过广泛领域的机器人学问题展示了该方法的有效性，并在CROCODDYL中提供了用户友好的开源实现。 

---
# Human Implicit Preference-Based Policy Fine-tuning for Multi-Agent Reinforcement Learning in USV Swarm 

**Title (ZH)**: 基于人类隐性偏好的多Agent强化学习USV群集策略微调 

**Authors**: Hyeonjun Kim, Kanghoon Lee, Junho Park, Jiachen Li, Jinkyoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2503.03796)  

**Abstract**: Multi-Agent Reinforcement Learning (MARL) has shown promise in solving complex problems involving cooperation and competition among agents, such as an Unmanned Surface Vehicle (USV) swarm used in search and rescue, surveillance, and vessel protection. However, aligning system behavior with user preferences is challenging due to the difficulty of encoding expert intuition into reward functions. To address the issue, we propose a Reinforcement Learning with Human Feedback (RLHF) approach for MARL that resolves credit-assignment challenges through an Agent-Level Feedback system categorizing feedback into intra-agent, inter-agent, and intra-team types. To overcome the challenges of direct human feedback, we employ a Large Language Model (LLM) evaluator to validate our approach using feedback scenarios such as region constraints, collision avoidance, and task allocation. Our method effectively refines USV swarm policies, addressing key challenges in multi-agent systems while maintaining fairness and performance consistency. 

**Abstract (ZH)**: 多代理 reinforcement 学习 (MARL) 在解决涉及代理之间合作与竞争的复杂问题（如用于搜索与救援、 surveillance 和船只保护的无人水面舰艇群）方面展现了潜力。然而，将系统行为与用户偏好对齐具有挑战性，因为将专家直觉编码到奖励函数中很难。为此，我们提出了一种基于人类反馈的 reinforcement 学习 (RLHF) 方法，该方法通过代理级反馈系统对反馈进行分类，分为代理内、代理间和团队内类型，以解决信用分配难题。为克服直接人类反馈的挑战，我们采用大型语言模型 (LLM) 评估器，使用区域约束、避碰和任务分配等反馈场景来验证我们的方法。该方法有效地优化了无人水面舰艇群的策略，解决了多代理系统中的关键挑战，同时保持了公平性和性能一致性。 

---
# Accelerating Focal Search in Multi-Agent Path Finding with Tighter Lower Bounds 

**Title (ZH)**: 基于更紧的下界加速多智能体路径寻找中的焦点搜索 

**Authors**: Yimin Tang, Zhenghong Yu, Jiaoyang Li, Sven Koenig  

**Link**: [PDF](https://arxiv.org/pdf/2503.03779)  

**Abstract**: Multi-Agent Path Finding (MAPF) involves finding collision-free paths for multiple agents while minimizing a cost function--an NP-hard problem. Bounded suboptimal methods like Enhanced Conflict-Based Search (ECBS) and Explicit Estimation CBS (EECBS) balance solution quality with computational efficiency using focal search mechanisms. While effective, traditional focal search faces a limitation: the lower bound (LB) value determining which nodes enter the FOCAL list often increases slowly in early search stages, resulting in a constrained search space that delays finding valid solutions. In this paper, we propose a novel bounded suboptimal algorithm, double-ECBS (DECBS), to address this issue by first determining the maximum LB value and then employing a best-first search guided by this LB to find a collision-free path. Experimental results demonstrate that DECBS outperforms ECBS in most test cases and is compatible with existing optimization techniques. DECBS can reduce nearly 30% high-level CT nodes and 50% low-level focal search nodes. When agent density is moderate to high, DECBS achieves a 23.5% average runtime improvement over ECBS with identical suboptimality bounds and optimizations. 

**Abstract (ZH)**: 多Agent路径规划(Multi-Agent Path Finding, MAPF)涉及在保证碰撞自由的同时最小化成本函数——一个NP难问题。增强冲突基于搜索(Enhanced Conflict-Based Search, ECBS)和显式估计冲突基于搜索(Explicit Estimation Conflict-Based Search, EECS)等受限次优方法通过焦点搜索机制平衡解的质量与计算效率。虽然这些方法有效，但传统的焦点搜索方法面临一个问题：决定哪些节点进入FOCAL列表的下界(LB)值在早期搜索阶段往往增长缓慢，导致搜索空间受限，从而延迟找到有效解。本文提出了一种新的受限次优算法双ECBS(Double-ECBS)，首先确定最大LB值，然后利用该LB值引导的最佳优先搜索来寻找碰撞自由路径。实验结果表明，DECBS在大多数测试案例中优于ECBS，并且与现有优化技术兼容。DECBS可以减少约30%的高层次CT节点和50%的低层次焦点搜索节点。当代理密度中等到较高时，与ECBS相比，DECBS在相同的次优性约束和优化条件下，平均可实现23.5%的运行时间改进。 

---
# Fair Play in the Fast Lane: Integrating Sportsmanship into Autonomous Racing Systems 

**Title (ZH)**: 快车道中的公平竞争：将运动精神整合到自主赛车系统中 

**Authors**: Zhenmin Huang, Ce Hao, Wei Zhan, Jun Ma, Masayoshi Tomizuka  

**Link**: [PDF](https://arxiv.org/pdf/2503.03774)  

**Abstract**: Autonomous racing has gained significant attention as a platform for high-speed decision-making and motion control. While existing methods primarily focus on trajectory planning and overtaking strategies, the role of sportsmanship in ensuring fair competition remains largely unexplored. In human racing, rules such as the one-motion rule and the enough-space rule prevent dangerous and unsportsmanlike behavior. However, autonomous racing systems often lack mechanisms to enforce these principles, potentially leading to unsafe maneuvers. This paper introduces a bi-level game-theoretic framework to integrate sportsmanship (SPS) into versus racing. At the high level, we model racing intentions using a Stackelberg game, where Monte Carlo Tree Search (MCTS) is employed to derive optimal strategies. At the low level, vehicle interactions are formulated as a Generalized Nash Equilibrium Problem (GNEP), ensuring that all agents follow sportsmanship constraints while optimizing their trajectories. Simulation results demonstrate the effectiveness of the proposed approach in enforcing sportsmanship rules while maintaining competitive performance. We analyze different scenarios where attackers and defenders adhere to or disregard sportsmanship rules and show how knowledge of these constraints influences strategic decision-making. This work highlights the importance of balancing competition and fairness in autonomous racing and provides a foundation for developing ethical and safe AI-driven racing systems. 

**Abstract (ZH)**: 自主赛车比赛作为高速决策和运动控制的平台受到了广泛关注。虽然现有方法主要关注轨迹规划和超越策略，但公平竞争中的体育精神作用仍很少被探讨。在人类赛车中，如一次动作原则和足够空间原则等规则可防止危险和不体育行为。然而，自主赛车系统往往缺乏执行这些原则的机制，可能导致不安全的操作。本文提出了一种双层博弈理论框架，将体育精神(SPS)整合到对抗赛车中。在高层次上，我们使用Stackelberg博弈建模赛车意图，并利用蒙特卡洛树搜索(MCTS)获取最优策略。在低层次上，车辆交互被形式化为广义纳什均衡问题(GNEP)，确保所有代理遵守体育精神约束条件并优化其轨迹。仿真结果证明了所提出方法在执行体育精神规则的同时保持竞争力的有效性。我们分析了攻击者和防守者遵守或不遵守体育精神规则的不同情景，并展示了这些约束条件如何影响战略决策。本文突出了在自主赛车中平衡竞争与公平的重要性，并为开发道德且安全的AI驱动赛车系统奠定了基础。 

---
