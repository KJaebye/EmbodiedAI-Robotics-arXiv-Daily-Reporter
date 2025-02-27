# Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models 

**Title (ZH)**: HI 机器人：基于层次视觉-语言-动作模型的开放性指令跟随 

**Authors**: Lucy Xiaoyang Shi, Brian Ichter, Michael Equi, Liyiming Ke, Karl Pertsch, Quan Vuong, James Tanner, Anna Walling, Haohuan Wang, Niccolo Fusai, Adrian Li-Bell, Danny Driess, Lachy Groom, Sergey Levine, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2502.19417)  

**Abstract**: Generalist robots that can perform a range of different tasks in open-world settings must be able to not only reason about the steps needed to accomplish their goals, but also process complex instructions, prompts, and even feedback during task execution. Intricate instructions (e.g., "Could you make me a vegetarian sandwich?" or "I don't like that one") require not just the ability to physically perform the individual steps, but the ability to situate complex commands and feedback in the physical world. In this work, we describe a system that uses vision-language models in a hierarchical structure, first reasoning over complex prompts and user feedback to deduce the most appropriate next step to fulfill the task, and then performing that step with low-level actions. In contrast to direct instruction following methods that can fulfill simple commands ("pick up the cup"), our system can reason through complex prompts and incorporate situated feedback during task execution ("that's not trash"). We evaluate our system across three robotic platforms, including single-arm, dual-arm, and dual-arm mobile robots, demonstrating its ability to handle tasks such as cleaning messy tables, making sandwiches, and grocery shopping. 

**Abstract (ZH)**: 通用机器人能够在开放环境中执行多种不同任务，必须能够不仅推理出完成目标所需的步骤，还能处理任务执行过程中的复杂指令、提示甚至反馈。复杂的指令（例如，“你能为我做一份素食三明治吗？”或“我不喜欢那个”）不仅需要物理执行个体步骤的能力，还需要能够将复杂命令和反馈置于物理世界中。在本文中，我们描述了一个采用分层结构的视觉语言模型系统，首先通过推理复杂的提示和用户反馈来推断出最合适的下一步以完成任务，然后通过低层次操作执行该步骤。与能够执行简单命令（如“拿起杯子”）的直接指令遵循方法不同，我们的系统能够通过复杂的提示进行推理，并在任务执行过程中结合情境反馈（如“那不是垃圾”）。我们在包括单臂、双臂和双臂移动机器人在内的三种机器人平台上对系统进行了评估，展示了其处理诸如清洁杂乱的桌子、制作三明治和购物等任务的能力。 

---
# ARENA: Adaptive Risk-aware and Energy-efficient NAvigation for Multi-Objective 3D Infrastructure Inspection with a UAV 

**Title (ZH)**: ARENA：自适应风险意识和能效导航的多目标3D基础设施无人机检测 

**Authors**: David-Alexandre Poissant, Alexis Lussier Desbiens, François Ferland, Louis Petit  

**Link**: [PDF](https://arxiv.org/pdf/2502.19401)  

**Abstract**: Autonomous robotic inspection missions require balancing multiple conflicting objectives while navigating near costly obstacles. Current multi-objective path planning (MOPP) methods struggle to adapt to evolving risks like localization errors, weather, battery state, and communication issues. This letter presents an Adaptive Risk-aware and Energy-efficient NAvigation (ARENA) MOPP approach for UAVs in complex 3D environments. Our method enables online trajectory adaptation by optimizing safety, time, and energy using 4D NURBS representation and a genetic-based algorithm to generate the Pareto front. A novel risk-aware voting algorithm ensures adaptivity. Simulations and real-world tests demonstrate the planner's ability to produce diverse, optimized trajectories covering 95% or more of the range defined by single-objective benchmarks and its ability to estimate power consumption with a mean error representing 14% of the full power range. The ARENA framework enhances UAV autonomy and reliability in critical, evolving 3D missions. 

**Abstract (ZH)**: 自主巡检机器人任务要求在接近昂贵障碍物时权衡多个相互冲突的目标。当前的多目标路径规划方法难以适应如定位误差、天气、电池状态和通信问题等不断变化的风险。本信提出了一种适应风险感知和能量高效导航（ARENA）的多目标路径规划方法，适用于具有复杂3D环境的无人机。该方法通过使用4D NURBS表示和基于遗传的算法优化安全、时间与能量，并生成帕累托前沿，实现了在线轨迹适应。一种新颖的风险感知投票算法确保了系统的适应性。仿真和实地测试展示了该规划器生成多样化、优化轨迹的能力，覆盖单目标基准定义范围的95%以上，并且能够以接近14%的最大功率范围误差估计功率消耗。ARENA框架提升了无人机在关键、动态3D任务中的自主性和可靠性。 

---
# Surface-Based Manipulation 

**Title (ZH)**: 基于表面的操作 

**Authors**: Ziqiao Wang, Serhat Demirtas, Fabio Zuliani, Jamie Paik  

**Link**: [PDF](https://arxiv.org/pdf/2502.19389)  

**Abstract**: Intelligence lies not only in the brain but in the body. The shape of our bodies can influence how we think and interact with the physical world. In robotics research, interacting with the physical world is crucial as it allows robots to manipulate objects in various real-life scenarios. Conventional robotic manipulation strategies mainly rely on finger-shaped end effectors. However, achieving stable grasps on fragile, deformable, irregularly shaped, or slippery objects is challenging due to difficulties in establishing stable force or geometric constraints.
Here, we present surface-based manipulation strategies that diverge from classical grasping approaches, using with flat surfaces as minimalist end-effectors. By changing the position and orientation of these surfaces, objects can be translated, rotated and even flipped across the surface using closed-loop control strategies. Since this method does not rely on stable grasp, it can adapt to objects of various shapes, sizes, and stiffness levels, even enabling the manipulation the shape of deformable objects. Our results provide a new perspective for solving complex manipulation problems. 

**Abstract (ZH)**: 智能不仅存在于大脑中，也存在于身体中。我们的身体形状可以影响我们的思维和与物理世界的互动。在机器人研究中，与物理世界的交互至关重要，因为它使机器人能够在各种现实情境中操作物体。传统的机器人操作策略主要依赖于手指状的末端执行器。然而，对脆弱、可变形、形状不规则或滑溜的物体实现稳定的抓握极具挑战性，原因在于难以建立稳定的力或几何约束。

在这里，我们提出了基于表面的操作策略，这些策略偏离了传统的抓取方法，使用平坦表面作为简约的末端执行器。通过改变这些表面的位置和方向，可以利用闭环控制策略将物体在表面进行平移、旋转甚至翻转。由于这种方法无需依赖稳定的抓握，因此它可以适应各种形状、大小和刚度级别的物体，甚至能够操作可变形物体的形状。我们的研究结果为解决复杂操作问题提供了新的视角。 

---
# LiDAR Registration with Visual Foundation Models 

**Title (ZH)**: 基于视觉基础模型的LiDAR注册 

**Authors**: Niclas Vödisch, Giovanni Cioffi, Marco Cannici, Wolfram Burgard, Davide Scaramuzza  

**Link**: [PDF](https://arxiv.org/pdf/2502.19374)  

**Abstract**: LiDAR registration is a fundamental task in robotic mapping and localization. A critical component of aligning two point clouds is identifying robust point correspondences using point descriptors. This step becomes particularly challenging in scenarios involving domain shifts, seasonal changes, and variations in point cloud structures. These factors substantially impact both handcrafted and learning-based approaches. In this paper, we address these problems by proposing to use DINOv2 features, obtained from surround-view images, as point descriptors. We demonstrate that coupling these descriptors with traditional registration algorithms, such as RANSAC or ICP, facilitates robust 6DoF alignment of LiDAR scans with 3D maps, even when the map was recorded more than a year before. Although conceptually straightforward, our method substantially outperforms more complex baseline techniques. In contrast to previous learning-based point descriptors, our method does not require domain-specific retraining and is agnostic to the point cloud structure, effectively handling both sparse LiDAR scans and dense 3D maps. We show that leveraging the additional camera data enables our method to outperform the best baseline by +24.8 and +17.3 registration recall on the NCLT and Oxford RobotCar datasets. We publicly release the registration benchmark and the code of our work on this https URL. 

**Abstract (ZH)**: 基于DINOv2特征的LiDAR点云配准方法 

---
# Hybrid Robot Learning for Automatic Robot Motion Planning in Manufacturing 

**Title (ZH)**: 制造领域自动机器人运动规划的混合机器人学习方法 

**Authors**: Siddharth Singh, Tian Yu, Qing Chang, John Karigiannis, Shaopeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.19340)  

**Abstract**: Industrial robots are widely used in diverse manufacturing environments. Nonetheless, how to enable robots to automatically plan trajectories for changing tasks presents a considerable challenge. Further complexities arise when robots operate within work cells alongside machines, humans, or other robots. This paper introduces a multi-level hybrid robot motion planning method combining a task space Reinforcement Learning-based Learning from Demonstration (RL-LfD) agent and a joint-space based Deep Reinforcement Learning (DRL) based agent. A higher level agent learns to switch between the two agents to enable feasible and smooth motion. The feasibility is computed by incorporating reachability, joint limits, manipulability, and collision risks of the robot in the given environment. Therefore, the derived hybrid motion planning policy generates a feasible trajectory that adheres to task constraints. The effectiveness of the method is validated through sim ulated robotic scenarios and in a real-world setup. 

**Abstract (ZH)**: 工业机器人在多样化制造环境中广泛应用，然而如何使机器人能够自动规划适应变化任务的轨迹仍是一项重大挑战。当机器人在工作单元中与机器、人类或其他机器人协同工作时，这一挑战变得更加复杂。本文提出了一种多层混合机器人运动规划方法，结合了基于任务空间的Reinforcement Learning-基于演示学习（RL-LfD）代理和基于关节空间的深度强化学习（DRL）代理。高层代理学习在两者之间切换，以实现可行且平滑的运动。可行性的计算通过综合考虑给定环境中机器人的可达性、关节极限、操作性和碰撞风险来实现。因此，所提取的混合运动规划策略生成了一条符合任务约束的可行轨迹。该方法的有效性通过模拟的机器人场景和实际应用中的验证得到了证明。 

---
# ObjectVLA: End-to-End Open-World Object Manipulation Without Demonstration 

**Title (ZH)**: ObjectVLA: 无演示的全程开放世界对象操作 

**Authors**: Minjie Zhu, Yichen Zhu, Jinming Li, Zhongyi Zhou, Junjie Wen, Xiaoyu Liu, Chaomin Shen, Yaxin Peng, Feifei Feng  

**Link**: [PDF](https://arxiv.org/pdf/2502.19250)  

**Abstract**: Imitation learning has proven to be highly effective in teaching robots dexterous manipulation skills. However, it typically relies on large amounts of human demonstration data, which limits its scalability and applicability in dynamic, real-world environments. One key challenge in this context is object generalization, where a robot trained to perform a task with one object, such as "hand over the apple," struggles to transfer its skills to a semantically similar but visually different object, such as "hand over the peach." This gap in generalization to new objects beyond those in the same category has yet to be adequately addressed in previous work on end-to-end visuomotor policy learning. In this paper, we present a simple yet effective approach for achieving object generalization through Vision-Language-Action (VLA) models, referred to as \textbf{ObjectVLA}. Our model enables robots to generalize learned skills to novel objects without requiring explicit human demonstrations for each new target object. By leveraging vision-language pair data, our method provides a lightweight and scalable way to inject knowledge about the target object, establishing an implicit link between the object and the desired action. We evaluate ObjectVLA on a real robotic platform, demonstrating its ability to generalize across 100 novel objects with a 64\% success rate in selecting objects not seen during training. Furthermore, we propose a more accessible method for enhancing object generalization in VLA models, using a smartphone to capture a few images and fine-tune the pre-trained model. These results highlight the effectiveness of our approach in enabling object-level generalization and reducing the need for extensive human demonstrations, paving the way for more flexible and scalable robotic learning systems. 

**Abstract (ZH)**: 模仿学习已被证明在教学徒机器人灵巧操作技能方面非常有效。然而，它通常依赖大量的人类演示数据，这限制了其在动态现实环境中的可扩展性和适用性。在这种背景下，一个关键挑战是对象泛化，即一个被训练执行一项任务（例如“把苹果递过来”）的机器人，在面对语义相似但外观不同的对象（例如“把桃子递过来”）时难以将技能迁移到新的对象上。在先前的端到端视觉-运动策略学习工作中，这一新类别对象的泛化问题尚未得到充分解决。在本文中，我们提出了一种通过视觉-语言-动作（VLA）模型实现对象泛化的简单有效方法，称为**ObjectVLA**。我们的模型使机器人能够在不需要为每个新目标对象提供明确的人类演示数据的情况下，将已学习的技能泛化到新对象上。通过利用视觉-语言配对数据，我们的方法提供了一种轻量级且可扩展的方式来注入目标对象的知识，从而在对象和所需动作之间建立隐式联系。我们在真实的机器人平台上评估了ObjectVLA，结果显示它能够在选择训练期间未见过的100个新对象中成功率达到64%。此外，我们提出了一种更简便的方法来增强VLA模型的对象泛化能力，使用智能手机捕捉几张图像并对预训练模型进行微调。这些结果突显了我们方法在实现对象级别泛化和减少大量人类演示需求方面的有效性，为更灵活和可扩展的机器人学习系统铺平了道路。 

---
# BEV-LIO(LC): BEV Image Assisted LiDAR-Inertial Odometry with Loop Closure 

**Title (ZH)**: BEV-LIO(LC): 基于BEV图像的 LiDAR-惯性里程计环路闭合辅助定位 

**Authors**: Haoxin Cai, Shenghai Yuan, Xinyi Li, Junfeng Guo, Jianqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.19242)  

**Abstract**: This work introduces BEV-LIO(LC), a novel LiDAR-Inertial Odometry (LIO) framework that combines Bird's Eye View (BEV) image representations of LiDAR data with geometry-based point cloud registration and incorporates loop closure (LC) through BEV image features. By normalizing point density, we project LiDAR point clouds into BEV images, thereby enabling efficient feature extraction and matching. A lightweight convolutional neural network (CNN) based feature extractor is employed to extract distinctive local and global descriptors from the BEV images. Local descriptors are used to match BEV images with FAST keypoints for reprojection error construction, while global descriptors facilitate loop closure detection. Reprojection error minimization is then integrated with point-to-plane registration within an iterated Extended Kalman Filter (iEKF). In the back-end, global descriptors are used to create a KD-tree-indexed keyframe database for accurate loop closure detection. When a loop closure is detected, Random Sample Consensus (RANSAC) computes a coarse transform from BEV image matching, which serves as the initial estimate for Iterative Closest Point (ICP). The refined transform is subsequently incorporated into a factor graph along with odometry factors, improving the global consistency of localization. Extensive experiments conducted in various scenarios with different LiDAR types demonstrate that BEV-LIO(LC) outperforms state-of-the-art methods, achieving competitive localization accuracy. Our code, video and supplementary materials can be found at this https URL. 

**Abstract (ZH)**: BEV-LIO(LC)：一种结合鸟瞰图表示的环视闭合LiDAR-惯性里程计框架 

---
# Leg Exoskeleton Odometry using a Limited FOV Depth Sensor 

**Title (ZH)**: 基于有限视野深度传感器的下肢外骨骼里程计 

**Authors**: Fabio Elnecave Xavier, Matis Viozelange, Guillaume Burger, Marine Pétriaux, Jean-Emmanuel Deschaud, François Goulette  

**Link**: [PDF](https://arxiv.org/pdf/2502.19237)  

**Abstract**: For leg exoskeletons to operate effectively in real-world environments, they must be able to perceive and understand the terrain around them. However, unlike other legged robots, exoskeletons face specific constraints on where depth sensors can be mounted due to the presence of a human user. These constraints lead to a limited Field Of View (FOV) and greater sensor motion, making odometry particularly challenging. To address this, we propose a novel odometry algorithm that integrates proprioceptive data from the exoskeleton with point clouds from a depth camera to produce accurate elevation maps despite these limitations. Our method builds on an extended Kalman filter (EKF) to fuse kinematic and inertial measurements, while incorporating a tailored iterative closest point (ICP) algorithm to register new point clouds with the elevation map. Experimental validation with a leg exoskeleton demonstrates that our approach reduces drift and enhances the quality of elevation maps compared to a purely proprioceptive baseline, while also outperforming a more traditional point cloud map-based variant. 

**Abstract (ZH)**: 基于深度相机点云与 proprioceptive 数据的新型腿部外骨骼 odometer 算法 

---
# CPG-Based Manipulation with Multi-Module Origami Robot Surface 

**Title (ZH)**: 基于CPG的多模块Origami机器人表面 manipulation 

**Authors**: Yuhao Jiang, Serge El Asmar, Ziqiao Wang, Serhat Demirtas, Jamie Paik  

**Link**: [PDF](https://arxiv.org/pdf/2502.19218)  

**Abstract**: Robotic manipulators often face challenges in handling objects of different sizes and materials, limiting their effectiveness in practical applications. This issue is particularly pronounced when manipulating meter-scale objects or those with varying stiffness, as traditional gripping techniques and strategies frequently prove inadequate. In this letter, we introduce a novel surface-based multi-module robotic manipulation framework that utilizes a Central Pattern Generator (CPG)-based motion generator, combined with a simulation-based optimization method to determine the optimal manipulation parameters for a multi-module origami robotic surface (Ori-Pixel). This approach allows for the manipulation of objects ranging from centimeters to meters in size, with varying stiffness and shape. The optimized CPG parameters are tested through both dynamic simulations and a series of prototype experiments involving a wide range of objects differing in size, weight, shape, and material, demonstrating robust manipulation capabilities. 

**Abstract (ZH)**: 基于表面的多模块机器人 manipulation 框架：用于不同大小和刚度物体的优化 Central Pattern Generator 参数研究 

---
# Increasing the Task Flexibility of Heavy-Duty Manipulators Using Visual 6D Pose Estimation of Objects 

**Title (ZH)**: 使用物体6D姿态估计提高重型 manipulator 作业灵活性 

**Authors**: Petri Mäkinen, Pauli Mustalahti, Tuomo Kivelä, Jouni Mattila  

**Link**: [PDF](https://arxiv.org/pdf/2502.19169)  

**Abstract**: Recent advances in visual 6D pose estimation of objects using deep neural networks have enabled novel ways of vision-based control for heavy-duty robotic applications. In this study, we present a pipeline for the precise tool positioning of heavy-duty, long-reach (HDLR) manipulators using advanced machine vision. A camera is utilized in the so-called eye-in-hand configuration to estimate directly the poses of a tool and a target object of interest (OOI). Based on the pose error between the tool and the target, along with motion-based calibration between the camera and the robot, precise tool positioning can be reliably achieved using conventional robotic modeling and control methods prevalent in the industry. The proposed methodology comprises orientation and position alignment based on the visually estimated OOI poses, whereas camera-to-robot calibration is conducted based on motion utilizing visual SLAM. The methods seek to avert the inaccuracies resulting from rigid-body--based kinematics of structurally flexible HDLR manipulators via image-based algorithms. To train deep neural networks for OOI pose estimation, only synthetic data are utilized. The methods are validated in a real-world setting using an HDLR manipulator with a 5 m reach. The experimental results demonstrate that an image-based average tool positioning error of less than 2 mm along the non-depth axes is achieved, which facilitates a new way to increase the task flexibility and automation level of non-rigid HDLR manipulators. 

**Abstract (ZH)**: 基于深度神经网络的物体视化6D姿态估计在重型长臂 manipulator 精确工具定位中的应用研究 

---
# Ground-level Viewpoint Vision-and-Language Navigation in Continuous Environments 

**Title (ZH)**: 地面视角的连续环境视觉-语言导航 

**Authors**: Zerui Li, Gengze Zhou, Haodong Hong, Yanyan Shao, Wenqi Lyu, Yanyuan Qiao, Qi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.19024)  

**Abstract**: Vision-and-Language Navigation (VLN) empowers agents to associate time-sequenced visual observations with corresponding instructions to make sequential decisions. However, generalization remains a persistent challenge, particularly when dealing with visually diverse scenes or transitioning from simulated environments to real-world deployment. In this paper, we address the mismatch between human-centric instructions and quadruped robots with a low-height field of view, proposing a Ground-level Viewpoint Navigation (GVNav) approach to mitigate this issue. This work represents the first attempt to highlight the generalization gap in VLN across varying heights of visual observation in realistic robot deployments. Our approach leverages weighted historical observations as enriched spatiotemporal contexts for instruction following, effectively managing feature collisions within cells by assigning appropriate weights to identical features across different viewpoints. This enables low-height robots to overcome challenges such as visual obstructions and perceptual mismatches. Additionally, we transfer the connectivity graph from the HM3D and Gibson datasets as an extra resource to enhance spatial priors and a more comprehensive representation of real-world scenarios, leading to improved performance and generalizability of the waypoint predictor in real-world environments. Extensive experiments demonstrate that our Ground-level Viewpoint Navigation (GVnav) approach significantly improves performance in both simulated environments and real-world deployments with quadruped robots. 

**Abstract (ZH)**: 地面视角导航（Ground-level Viewpoint Navigation, GVNav）：缓解视觉语言导航中的身高差异通用性差距 

---
# SLAM in the Dark: Self-Supervised Learning of Pose, Depth and Loop-Closure from Thermal Images 

**Title (ZH)**: 暗中SLAM：基于热成像的自监督姿态、深度和环回闭合学习 

**Authors**: Yangfan Xu, Qu Hao, Lilian Zhang, Jun Mao, Xiaofeng He, Wenqi Wu, Changhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18932)  

**Abstract**: Visual SLAM is essential for mobile robots, drone navigation, and VR/AR, but traditional RGB camera systems struggle in low-light conditions, driving interest in thermal SLAM, which excels in such environments. However, thermal imaging faces challenges like low contrast, high noise, and limited large-scale annotated datasets, restricting the use of deep learning in outdoor scenarios. We present DarkSLAM, a noval deep learning-based monocular thermal SLAM system designed for large-scale localization and reconstruction in complex lighting this http URL approach incorporates the Efficient Channel Attention (ECA) mechanism in visual odometry and the Selective Kernel Attention (SKA) mechanism in depth estimation to enhance pose accuracy and mitigate thermal depth degradation. Additionally, the system includes thermal depth-based loop closure detection and pose optimization, ensuring robust performance in low-texture thermal scenes. Extensive outdoor experiments demonstrate that DarkSLAM significantly outperforms existing methods like SC-Sfm-Learner and Shin et al., delivering precise localization and 3D dense mapping even in challenging nighttime environments. 

**Abstract (ZH)**: 基于深度学习的无光温差SLAM系统：暗光环境下的大规模定位与重建 

---
# Think on your feet: Seamless Transition between Human-like Locomotion in Response to Changing Commands 

**Title (ZH)**: 随机应变：根据变化的指令实现类人的运动无缝过渡 

**Authors**: Huaxing Huang, Wenhao Cui, Tonghe Zhang, Shengtao Li, Jinchao Han, Bangyu Qin, Tianchu Zhang, Liang Zheng, Ziyang Tang, Chenxu Hu, Ning Yan, Jiahao Chen, Shipu Zhang, Zheyuan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18901)  

**Abstract**: While it is relatively easier to train humanoid robots to mimic specific locomotion skills, it is more challenging to learn from various motions and adhere to continuously changing commands. These robots must accurately track motion instructions, seamlessly transition between a variety of movements, and master intermediate motions not present in their reference data. In this work, we propose a novel approach that integrates human-like motion transfer with precise velocity tracking by a series of improvements to classical imitation learning. To enhance generalization, we employ the Wasserstein divergence criterion (WGAN-div). Furthermore, a Hybrid Internal Model provides structured estimates of hidden states and velocity to enhance mobile stability and environment adaptability, while a curiosity bonus fosters exploration. Our comprehensive method promises highly human-like locomotion that adapts to varying velocity requirements, direct generalization to unseen motions and multitasking, as well as zero-shot transfer to the simulator and the real world across different terrains. These advancements are validated through simulations across various robot models and extensive real-world experiments. 

**Abstract (ZH)**: 一种结合人类运动转移与精确速度跟踪的改进 imitation 学习方法：适应变化速度要求、未见动作的直接泛化及多任务学习的研究 

---
# RL-OGM-Parking: Lidar OGM-Based Hybrid Reinforcement Learning Planner for Autonomous Parking 

**Title (ZH)**: 基于激光雷达OGM的混合强化学习自主泊车规划方法 

**Authors**: Zhitao Wang, Zhe Chen, Mingyang Jiang, Tong Qin, Ming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18846)  

**Abstract**: Autonomous parking has become a critical application in automatic driving research and development. Parking operations often suffer from limited space and complex environments, requiring accurate perception and precise maneuvering. Traditional rule-based parking algorithms struggle to adapt to diverse and unpredictable conditions, while learning-based algorithms lack consistent and stable performance in various scenarios. Therefore, a hybrid approach is necessary that combines the stability of rule-based methods and the generalizability of learning-based methods. Recently, reinforcement learning (RL) based policy has shown robust capability in planning tasks. However, the simulation-to-reality (sim-to-real) transfer gap seriously blocks the real-world deployment. To address these problems, we employ a hybrid policy, consisting of a rule-based Reeds-Shepp (RS) planner and a learning-based reinforcement learning (RL) planner. A real-time LiDAR-based Occupancy Grid Map (OGM) representation is adopted to bridge the sim-to-real gap, leading the hybrid policy can be applied to real-world systems seamlessly. We conducted extensive experiments both in the simulation environment and real-world scenarios, and the result demonstrates that the proposed method outperforms pure rule-based and learning-based methods. The real-world experiment further validates the feasibility and efficiency of the proposed method. 

**Abstract (ZH)**: 自主泊车已成为自动驾驶研究与开发中的关键应用。泊车操作常受限于有限的空间和复杂的环境，需要准确的感知和精确的操作。传统的基于规则的泊车算法难以适应多变且不可预测的条件，而基于学习的算法在各种场景中缺乏一致且稳定的表现。因此，有必要结合基于规则方法的稳定性和基于学习方法的一般性，采用一种混合策略。最近，基于强化学习（RL）的策略在规划任务中显示出强大的能力。然而，模拟到现实（sim-to-real）的转移差距严重阻碍了其实用性部署。为解决这些问题，我们采用了一种混合策略，该策略结合了基于规则的Reeds-Shepp（RS）规划器和基于学习的强化学习（RL）规划器。采用实时LiDAR基于的占用网格地图（OGM）表示来弥合模拟到现实的差距，使得混合策略能够无缝应用于现实系统。我们在模拟环境和真实场景中进行了广泛的实验，结果表明所提出的方法优于纯基于规则和基于学习的方法。进一步的真实场景实验验证了所提出方法的可行性和效率。标题：

一种结合Reeds-Shepp规划器和强化学习规划器的混合自主泊车方法 

---
# Attention-Guided Integration of CLIP and SAM for Precise Object Masking in Robotic Manipulation 

**Title (ZH)**: 基于注意力引导的CLIP与SAM融合方法实现精确物体掩码在机器人操作中的应用 

**Authors**: Muhammad A. Muttaqien, Tomohiro Motoda, Ryo Hanai, Domae Yukiyasu  

**Link**: [PDF](https://arxiv.org/pdf/2502.18842)  

**Abstract**: This paper introduces a novel pipeline to enhance the precision of object masking for robotic manipulation within the specific domain of masking products in convenience stores. The approach integrates two advanced AI models, CLIP and SAM, focusing on their synergistic combination and the effective use of multimodal data (image and text). Emphasis is placed on utilizing gradient-based attention mechanisms and customized datasets to fine-tune performance. While CLIP, SAM, and Grad- CAM are established components, their integration within this structured pipeline represents a significant contribution to the field. The resulting segmented masks, generated through this combined approach, can be effectively utilized as inputs for robotic systems, enabling more precise and adaptive object manipulation in the context of convenience store products. 

**Abstract (ZH)**: 本文介绍了一种新颖的工作流程，以提高机器人在便利商店产品掩码领域中操作时对象掩码的准确性。该方法结合了两种先进的AI模型——CLIP和SAM，重点在于它们的协同作用以及多模态数据（图像和文本）的有效利用。强调利用梯度基注意力机制和定制化数据集来 fine-tune 性能。尽管 CLIP、SAM 和 Grad-CAM 是成熟的技术组件，但它们在这结构化工作流程中的集成对领域做出了重要贡献。通过这种综合方法生成的分割掩码可以有效地作为机器人系统的输入，从而在便利商店产品上下文中实现更精确和适应性强的对象操作。 

---
# Efficient and Distributed Large-Scale Point Cloud Bundle Adjustment via Majorization-Minimization 

**Title (ZH)**: 大规模点云_bundle_调整的高效分布式majorization-minimization方法 

**Authors**: Rundong Li, Zheng Liu, Hairuo Wei, Yixi Cai, Haotian Li, Fu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18801)  

**Abstract**: Point cloud bundle adjustment is critical in large-scale point cloud mapping. However, it is both computationally and memory intensive, with its complexity growing cubically as the number of scan poses increases. This paper presents BALM3.0, an efficient and distributed large-scale point cloud bundle adjustment method. The proposed method employs the majorization-minimization algorithm to decouple the scan poses in the bundle adjustment process, thus performing the point cloud bundle adjustment on large-scale data with improved computational efficiency. The key difficulty of applying majorization-minimization on bundle adjustment is to identify the proper surrogate cost function. In this paper, the proposed surrogate cost function is based on the point-to-plane distance. The primary advantages of decoupling the scan poses via a majorization-minimization algorithm stem from two key aspects. First, the decoupling of scan poses reduces the optimization time complexity from cubic to linear, significantly enhancing the computational efficiency of the bundle adjustment process in large-scale environments. Second, it lays the theoretical foundation for distributed bundle adjustment. By distributing both data and computation across multiple devices, this approach helps overcome the limitations posed by large memory and computational requirements, which may be difficult for a single device to handle. The proposed method is extensively evaluated in both simulated and real-world environments. The results demonstrate that the proposed method achieves the same optimal residual with comparable accuracy while offering up to 704 times faster optimization speed and reducing memory usage to 1/8. Furthermore, this paper also presented and implemented a distributed bundle adjustment framework and successfully optimized large-scale data (21,436 poses with 70 GB point clouds) with four consumer-level laptops. 

**Abstract (ZH)**: 大规模点云Bundle调整的BALM3.0高效分布式方法 

---
# Learning Autonomy: Off-Road Navigation Enhanced by Human Input 

**Title (ZH)**: 自主学习：增强型离线导航由人类输入辅助 

**Authors**: Akhil Nagariya, Dimitar Filev, Srikanth Saripalli, Gaurav Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2502.18760)  

**Abstract**: In the area of autonomous driving, navigating off-road terrains presents a unique set of challenges, from unpredictable surfaces like grass and dirt to unexpected obstacles such as bushes and puddles. In this work, we present a novel learning-based local planner that addresses these challenges by directly capturing human driving nuances from real-world demonstrations using only a monocular camera. The key features of our planner are its ability to navigate in challenging off-road environments with various terrain types and its fast learning capabilities. By utilizing minimal human demonstration data (5-10 mins), it quickly learns to navigate in a wide array of off-road conditions. The local planner significantly reduces the real world data required to learn human driving preferences. This allows the planner to apply learned behaviors to real-world scenarios without the need for manual fine-tuning, demonstrating quick adjustment and adaptability in off-road autonomous driving technology. 

**Abstract (ZH)**: 在自主驾驶领域，穿越非沥青路面地形呈现一系列独特挑战，从不可预测的草地和泥土表面到突如其来的灌木丛和水坑等障碍。本文提出了一种新颖的学习型局部规划器，通过仅使用单目摄像头直接从实际场景演示中捕捉人类驾驶的细微之处来应对这些挑战。该规划器的关键特性在于其能够在多种地形类型的复杂非沥青路面上导航，并且具有快速学习能力。通过利用少量的人类演示数据（5-10分钟），它能够迅速学习在各种非沥青路面上的导航技巧。该局部规划器显著减少了学习人类驾驶偏好的所需实际场景数据量。这使得规划器能够在不需要手动微调的情况下，将学习到的行为应用到实际场景中，展示了在非沥青路面自主驾驶技术中的快速调整和适应性。 

---
# Simulating Safe Bite Transfer in Robot-Assisted Feeding with a Soft Head and Articulated Jaw 

**Title (ZH)**: 基于柔软头部和articulated颚的机器人辅助喂食中安全咬合转移的模拟 

**Authors**: Yi Heng San, Vasanthamaran Ravichandram, J-Anne Yow, Sherwin Stephen Chan, Yifan Wang, Wei Tech Ang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18749)  

**Abstract**: Ensuring safe and comfortable bite transfer during robot-assisted feeding is challenging due to the close physical human-robot interaction required. This paper presents a novel approach to modeling physical human-robot interaction in a physics-based simulator (MuJoCo) using soft-body dynamics. We integrate a flexible head model with a rigid skeleton while accounting for internal dynamics, enabling the flexible model to be actuated by the skeleton. Incorporating realistic soft-skin contact dynamics in simulation allows for systematically evaluating bite transfer parameters, such as insertion depth and entry angle, and their impact on user safety and comfort. Our findings suggest that a straight-in-straight-out strategy minimizes forces and enhances user comfort in robot-assisted feeding, assuming a static head. This simulation-based approach offers a safer and more controlled alternative to real-world experimentation. Supplementary videos can be found at: this https URL. 

**Abstract (ZH)**: 确保机器人辅助喂食过程中安全舒适的咬合转移具有挑战性，因为需要进行紧密的物理人机交互。本文提出了一种在基于物理的模拟器（MuJoCo）中使用软体动力学建模物理人机交互的新型方法。我们整合了一个可挠头模型和刚性骨架，并考虑了内部动力学，使柔性模型可以由骨架驱动。在仿真中引入现实的软皮肤接触动力学，可以系统地评估咬合转移参数，如插入深度和进入角度，以及它们对用户安全性和舒适性的影响。我们的研究结果表明，在假设头部静止的情况下，直线进出策略可以最小化力并提高机器人辅助喂食的用户舒适度。基于仿真的方法为实际实验提供了更安全、更可控的替代方案。有关补充视频，请参阅：this https URL。 

---
# MaskPlanner: Learning-Based Object-Centric Motion Generation from 3D Point Clouds 

**Title (ZH)**: MaskPlanner: 基于学习的物体中心运动生成从3D点云 

**Authors**: Gabriele Tiboni, Raffaello Camoriano, Tatiana Tommasi  

**Link**: [PDF](https://arxiv.org/pdf/2502.18745)  

**Abstract**: Object-Centric Motion Generation (OCMG) plays a key role in a variety of industrial applications$\unicode{x2014}$such as robotic spray painting and welding$\unicode{x2014}$requiring efficient, scalable, and generalizable algorithms to plan multiple long-horizon trajectories over free-form 3D objects. However, existing solutions rely on specialized heuristics, expensive optimization routines, or restrictive geometry assumptions that limit their adaptability to real-world scenarios. In this work, we introduce a novel, fully data-driven framework that tackles OCMG directly from 3D point clouds, learning to generalize expert path patterns across free-form surfaces. We propose MaskPlanner, a deep learning method that predicts local path segments for a given object while simultaneously inferring "path masks" to group these segments into distinct paths. This design induces the network to capture both local geometric patterns and global task requirements in a single forward pass. Extensive experimentation on a realistic robotic spray painting scenario shows that our approach attains near-complete coverage (above 99%) for unseen objects, while it remains task-agnostic and does not explicitly optimize for paint deposition. Moreover, our real-world validation on a 6-DoF specialized painting robot demonstrates that the generated trajectories are directly executable and yield expert-level painting quality. Our findings crucially highlight the potential of the proposed learning method for OCMG to reduce engineering overhead and seamlessly adapt to several industrial use cases. 

**Abstract (ZH)**: 对象中心的运动生成（OCMG）在各种工业应用中发挥着关键作用，如机器人喷涂和焊接，需要高效的、可扩展的和通用的算法来规划自由形式3D物体上的长时_horizon_轨迹。然而，现有解决方案依赖于专门的启发式方法、昂贵的优化过程或限制性的几何假设，这限制了它们在实际场景中的适应性。在本文中，我们提出了一种全新的、完全数据驱动的框架，直接从3D点云中处理OCMG问题，学习在自由形态表面上泛化专家路径模式。我们提出了一种名为MaskPlanner的深度学习方法，该方法在给定对象的情况下预测局部路径片段，同时推断“路径掩码”将这些片段聚合成不同的路径。这种设计促使网络在一个前向传递过程中同时捕捉局部几何模式和全局任务需求。在现实的机器人喷涂场景中的广泛实验表明，我们的方法对于未见过的对象可以获得接近完全的覆盖（超过99%），并且它是任务无关的，未明确针对涂料沉积进行优化。此外，在一个六自由度专业喷涂机器人上的实际验证表明，生成的轨迹可以直接执行并达到专家级的喷涂质量。我们的研究结果关键性地突显了所提出的学习方法在OCMG中的潜力，可以减少工程开销并无缝适应多个工业应用场景。 

---
# QueryAdapter: Rapid Adaptation of Vision-Language Models in Response to Natural Language Queries 

**Title (ZH)**: QueryAdapter: 面向自然语言查询的视觉语言模型快速适配 

**Authors**: Nicolas Harvey Chapman, Feras Dayoub, Will Browne, Christopher Lehnert  

**Link**: [PDF](https://arxiv.org/pdf/2502.18735)  

**Abstract**: A domain shift exists between the large-scale, internet data used to train a Vision-Language Model (VLM) and the raw image streams collected by a robot. Existing adaptation strategies require the definition of a closed-set of classes, which is impractical for a robot that must respond to diverse natural language queries. In response, we present QueryAdapter; a novel framework for rapidly adapting a pre-trained VLM in response to a natural language query. QueryAdapter leverages unlabelled data collected during previous deployments to align VLM features with semantic classes related to the query. By optimising learnable prompt tokens and actively selecting objects for training, an adapted model can be produced in a matter of minutes. We also explore how objects unrelated to the query should be dealt with when using real-world data for adaptation. In turn, we propose the use of object captions as negative class labels, helping to produce better calibrated confidence scores during adaptation. Extensive experiments on ScanNet++ demonstrate that QueryAdapter significantly enhances object retrieval performance compared to state-of-the-art unsupervised VLM adapters and 3D scene graph methods. Furthermore, the approach exhibits robust generalization to abstract affordance queries and other datasets, such as Ego4D. 

**Abstract (ZH)**: 一个大型互联网数据训练的视觉-语言模型与机器人收集的原始图像流之间存在领域转换问题。现有的适应策略需要定义一个封闭类集，这对于必须响应 diverse 自然语言查询的机器人来说是不切实际的。为此，我们提出 QueryAdapter；一种针对自然语言查询快速适应预训练视觉-语言模型的新型框架。QueryAdapter 利用先前部署中收集的未标注数据，将 VLM 特征对齐到与查询相关的语义类。通过优化可学习的提示标记并主动选择用于训练的对象，可以在几分钟内生成适应模型。我们还探讨了在使用真实世界数据进行适应时，如何处理与查询无关的对象。为此，我们建议使用对象描述作为负面类标签，有助于在适应过程中产生更好的校准置信分数。广泛的 ScanNet++ 实验表明，QueryAdapter 在对象检索性能上显著优于最先进的无监督 VLM 调适方法和 3D 场景图方法。此外，该方法在抽象用法查询和其他数据集（如 Ego4D）上表现出稳健的泛化能力。 

---
# Rapidly Built Medical Crash Cart! Lessons Learned and Impacts on High-Stakes Team Collaboration in the Emergency Room 

**Title (ZH)**: 快速构建的医疗抢救车！紧急室高 stakes 团队协作中的经验教训及影响 

**Authors**: Angelique Taylor, Tauhid Tanjim, Michael Joseph Sack, Maia Hirsch, Kexin Cheng, Kevin Ching, Jonathan St. George, Thijs Roumen, Malte F. Jung, Hee Rin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.18688)  

**Abstract**: Designing robots to support high-stakes teamwork in emergency settings presents unique challenges, including seamless integration into fast-paced environments, facilitating effective communication among team members, and adapting to rapidly changing situations. While teleoperated robots have been successfully used in high-stakes domains such as firefighting and space exploration, autonomous robots that aid highs-takes teamwork remain underexplored. To address this gap, we conducted a rapid prototyping process to develop a series of seemingly autonomous robot designed to assist clinical teams in the Emergency Room. We transformed a standard crash cart--which stores medical equipment and emergency supplies into a medical robotic crash cart (MCCR). The MCCR was evaluated through field deployments to assess its impact on team workload and usability, identified taxonomies of failure, and refined the MCCR in collaboration with healthcare professionals. Our work advances the understanding of robot design for high-stakes, time-sensitive settings, providing insights into useful MCCR capabilities and considerations for effective human-robot collaboration. By publicly disseminating our MCCR tutorial, we hope to encourage HRI researchers to explore the design of robots for high-stakes teamwork. 

**Abstract (ZH)**: 设计用于紧急情况下高风险团队支持的机器人面临独特挑战，包括无缝融入快节奏环境、促进团队成员间有效沟通以及适应快速变化的情况。虽然远程操作机器人已在灭火和太空探索等高风险领域成功应用，但辅助高风险团队协作的自主机器人仍处于探索阶段。为解决这一缺口，我们通过快速原型设计过程开发了一系列看似自主的机器人，旨在协助急诊室临床团队。我们将标准的急救车转变为医疗机器人急救车（MCCR）。通过实地部署评估MCCR对团队工作负荷的影响和易用性，识别失败模式，并与医疗专业人员合作改进MCCR。我们的工作推进了对高风险、时间敏感环境中机器人设计的理解，提供了有关有用MCCR功能和有效人机协作考虑的见解。通过公开发布MCCR教程，我们希望鼓励HRI研究人员探索高风险团队协作中机器人的设计。 

---
# A Distributional Treatment of Real2Sim2Real for Vision-Driven Deformable Linear Object Manipulation 

**Title (ZH)**: 基于视觉驱动的可变形线性物体操纵的分布处理从真实到模拟再到真实的转变 

**Authors**: Georgios Kamaras, Subramanian Ramamoorthy  

**Link**: [PDF](https://arxiv.org/pdf/2502.18615)  

**Abstract**: We present an integrated (or end-to-end) framework for the Real2Sim2Real problem of manipulating deformable linear objects (DLOs) based on visual perception. Working with a parameterised set of DLOs, we use likelihood-free inference (LFI) to compute the posterior distributions for the physical parameters using which we can approximately simulate the behaviour of each specific DLO. We use these posteriors for domain randomisation while training, in simulation, object-specific visuomotor policies for a visuomotor DLO reaching task, using model-free reinforcement learning. We demonstrate the utility of this approach by deploying sim-trained DLO manipulation policies in the real world in a zero-shot manner, i.e. without any further fine-tuning. In this context, we evaluate the capacity of a prominent LFI method to perform fine classification over the parametric set of DLOs, using only visual and proprioceptive data obtained in a dynamic manipulation trajectory. We then study the implications of the resulting domain distributions in sim-based policy learning and real-world performance. 

**Abstract (ZH)**: 基于视觉感知的柔体线性对象 manipulatiion 的端到端框架：从真实世界到模拟再到现实世界 

---
# Autonomous Vision-Guided Resection of Central Airway Obstruction 

**Title (ZH)**: 自主视觉引导中央气道阻塞切除术 

**Authors**: M. E. Smith, N. Yilmaz, T. Watts, P. M. Scheikl, J. Ge, A. Deguet, A. Kuntz, A. Krieger  

**Link**: [PDF](https://arxiv.org/pdf/2502.18586)  

**Abstract**: Existing tracheal tumor resection methods often lack the precision required for effective airway clearance, and robotic advancements offer new potential for autonomous resection. We present a vision-guided, autonomous approach for palliative resection of tracheal tumors. This system models the tracheal surface with a fifth-degree polynomial to plan tool trajectories, while a custom Faster R-CNN segmentation pipeline identifies the trachea and tumor boundaries. The electrocautery tool angle is optimized using handheld surgical demonstrations, and trajectories are planned to maintain a 1 mm safety clearance from the tracheal surface. We validated the workflow successfully in five consecutive experiments on ex-vivo animal tissue models, successfully clearing the airway obstruction without trachea perforation in all cases (with more than 90% volumetric tumor removal). These results support the feasibility of an autonomous resection platform, paving the way for future developments in minimally-invasive autonomous resection. 

**Abstract (ZH)**: 基于视觉引导的自动气管肿瘤消融方法 

---
# Embodying mechano-fluidic memory in soft machines to program behaviors upon interactions 

**Title (ZH)**: 在软机器中嵌入 mechano-fluidic 记忆以编程交互行为 

**Authors**: Alberto Comoretto, Tanaya Mandke, Johannes T.B. Overvelde  

**Link**: [PDF](https://arxiv.org/pdf/2502.19192)  

**Abstract**: Soft machines display shape adaptation to external circumstances due to their intrinsic compliance. To achieve increasingly more responsive behaviors upon interactions without relying on centralized computation, embodying memory directly in the machines' structure is crucial. Here, we harness the bistability of elastic shells to alter the fluidic properties of an enclosed cavity, thereby switching between stable frequency states of a locomoting self-oscillating machine. To program these memory states upon interactions, we develop fluidic circuits surrounding the bistable shell, with soft tubes that kink and unkink when externally touched. We implement circuits for both long-term and short-term memory in a soft machine that switches behaviors in response to a human user and that autonomously changes direction after detecting a wall. By harnessing only geometry and elasticity, embodying memory allows physical structures without a central brain to exhibit autonomous feats that are typically reserved for computer-based robotic systems. 

**Abstract (ZH)**: 软机器通过其固有的顺应性对外部环境进行形状适应。为了在不依赖集中计算的情况下实现更加响应性的行为，直接在机器的结构中体现记忆至关重要。在这里，我们利用弹性薄壳的双稳定特性改变其包围腔体的流体属性，从而在行进的自振荡机器之间切换稳定频率状态。为了在交互中编程这些记忆状态，我们开发了围绕双稳态壳体的流体电路，其中软管在外部触碰时会弯曲和恢复。我们在一个能够根据人类用户的交互表现出不同行为并在检测到墙壁后自主改变方向的软机器中实现长期和短期记忆的电路。仅仅利用几何和弹性，体现记忆使得没有中央大脑的物理结构能够展示出通常只能由基于计算机的机器人系统实现的自主行为。 

---
# PlantPal: Leveraging Precision Agriculture Robots to Facilitate Remote Engagement in Urban Gardening 

**Title (ZH)**: PlantPal: 利用精准农业机器人促进城市园艺的远程参与 

**Authors**: Albin Zeqiri, Julian Britten, Clara Schramm, Pascal Jansen, Michael Rietzler, Enrico Rukzio  

**Link**: [PDF](https://arxiv.org/pdf/2502.19171)  

**Abstract**: Urban gardening is widely recognized for its numerous health and environmental benefits. However, the lack of suitable garden spaces, demanding daily schedules and limited gardening expertise present major roadblocks for citizens looking to engage in urban gardening. While prior research has explored smart home solutions to support urban gardeners, these approaches currently do not fully address these practical barriers. In this paper, we present PlantPal, a system that enables the cultivation of garden spaces irrespective of one's location, expertise level, or time constraints. PlantPal enables the shared operation of a precision agriculture robot (PAR) that is equipped with garden tools and a multi-camera system. Insights from a 3-week deployment (N=18) indicate that PlantPal facilitated the integration of gardening tasks into daily routines, fostered a sense of connection with one's field, and provided an engaging experience despite the remote setting. We contribute design considerations for future robot-assisted urban gardening concepts. 

**Abstract (ZH)**: 城市园艺因其众多的健康和环境益处而广受认可。然而，缺乏合适的花园空间、严苛的日常安排和有限的园艺技能是市民参与城市园艺的主要障碍。尽管先前的研究探讨了支持城市园艺者的智能家居解决方案，但这些方法目前尚未充分解决这些实际障碍。本文中，我们介绍了PlantPal系统，该系统能够在Regardless of 一个人的位置、技能水平或时间限制的情况下开展园艺活动。PlantPal通过一种配备有园艺工具和多摄像头系统的精确农业机器人（PAR），实现了多个园艺空间的操作共享。为期三周的部署（N=18）表明，PlantPal使得园艺任务能够融入日常生活，强化了与园地的联系，并提供了一种引人入胜的体验，即使在远程情况下也是如此。我们为未来的辅助机器人城市园艺概念提供了设计考虑。 

---
# A Temporal Planning Framework for Multi-Agent Systems via LLM-Aided Knowledge Base Management 

**Title (ZH)**: 基于LLM辅助知识库管理的多Agent系统时间规划框架 

**Authors**: Enrico Saccon, Ahmet Tikna, Davide De Martini, Edoardo Lamon, Luigi Palopoli, Marco Roveri  

**Link**: [PDF](https://arxiv.org/pdf/2502.19135)  

**Abstract**: This paper presents a novel framework, called PLANTOR (PLanning with Natural language for Task-Oriented Robots), that integrates Large Language Models (LLMs) with Prolog-based knowledge management and planning for multi-robot tasks. The system employs a two-phase generation of a robot-oriented knowledge base, ensuring reusability and compositional reasoning, as well as a three-step planning procedure that handles temporal dependencies, resource constraints, and parallel task execution via mixed-integer linear programming. The final plan is converted into a Behaviour Tree for direct use in ROS2. We tested the framework in multi-robot assembly tasks within a block world and an arch-building scenario. Results demonstrate that LLMs can produce accurate knowledge bases with modest human feedback, while Prolog guarantees formal correctness and explainability. This approach underscores the potential of LLM integration for advanced robotics tasks requiring flexible, scalable, and human-understandable planning. 

**Abstract (ZH)**: PLANTOR：面向任务机器人规划的自然语言与Prolog知识管理框架 

---
# Interpretable Data-Driven Ship Dynamics Model: Enhancing Physics-Based Motion Prediction with Parameter Optimization 

**Title (ZH)**: 可解释的数据驱动船舶动力学模型：基于参数优化的物理基础运动预测增强 

**Authors**: Papandreou Christos, Mathioudakis Michail, Stouraitis Theodoros, Iatropoulos Petros, Nikitakis Antonios, Stavros Paschalakis, Konstantinos Kyriakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2502.18696)  

**Abstract**: The deployment of autonomous navigation systems on ships necessitates accurate motion prediction models tailored to individual vessels. Traditional physics-based models, while grounded in hydrodynamic principles, often fail to account for ship-specific behaviors under real-world conditions. Conversely, purely data-driven models offer specificity but lack interpretability and robustness in edge cases. This study proposes a data-driven physics-based model that integrates physics-based equations with data-driven parameter optimization, leveraging the strengths of both approaches to ensure interpretability and adaptability. The model incorporates physics-based components such as 3-DoF dynamics, rudder, and propeller forces, while parameters such as resistance curve and rudder coefficients are optimized using synthetic data. By embedding domain knowledge into the parameter optimization process, the fitted model maintains physical consistency. Validation of the approach is realized with two container ships by comparing, both qualitatively and quantitatively, predictions against ground-truth trajectories. The results demonstrate significant improvements, in predictive accuracy and reliability, of the data-driven physics-based models over baseline physics-based models tuned with traditional marine engineering practices. The fitted models capture ship-specific behaviors in diverse conditions with their predictions being, 51.6% (ship A) and 57.8% (ship B) more accurate, 72.36% (ship A) and 89.67% (ship B) more consistent. 

**Abstract (ZH)**: 自主导航系统在船舶上的部署需要针对单艘船舶定制的准确运动预测模型。传统基于物理的模型虽然基于水动力原理，但在实际条件下往往无法准确反映船舶特定行为。相比之下，纯数据驱动的模型虽然具有特定性，但在边缘情况下缺乏可解释性和鲁棒性。本研究提出了一种结合基于物理的方程与数据驱动参数优化的基于物理的数据驱动模型，利用两者的优点确保模型的可解释性和适应性。该模型整合了基于物理的组件如3-DoF动力学、舵力和推进力，而阻力曲线参数和舵系数则通过合成数据进行优化。通过将领域知识嵌入参数优化过程，拟合模型保持了物理一致性。通过将两种集装箱船的实际轨迹与预测结果进行定性和定量比较，验证了该方法的有效性。结果表明，基于物理的数据驱动模型在预测准确性和可靠性方面显著优于传统海洋工程实践中调优的传统基于物理的模型。拟合模型在不同条件下捕捉到船特有的行为，预测准确性和一致性分别提高了51.6%（船A）和57.8%（船B），72.36%（船A）和89.67%（船B）。 

---
# Hybrid Voting-Based Task Assignment in Role-Playing Games 

**Title (ZH)**: 基于角色扮演游戏中混合投票的任务分配 

**Authors**: Daniel Weiner, Raj Korpan  

**Link**: [PDF](https://arxiv.org/pdf/2502.18690)  

**Abstract**: In role-playing games (RPGs), the level of immersion is critical-especially when an in-game agent conveys tasks, hints, or ideas to the player. For an agent to accurately interpret the player's emotional state and contextual nuances, a foundational level of understanding is required, which can be achieved using a Large Language Model (LLM). Maintaining the LLM's focus across multiple context changes, however, necessitates a more robust approach, such as integrating the LLM with a dedicated task allocation model to guide its performance throughout gameplay. In response to this need, we introduce Voting-Based Task Assignment (VBTA), a framework inspired by human reasoning in task allocation and completion. VBTA assigns capability profiles to agents and task descriptions to tasks, then generates a suitability matrix that quantifies the alignment between an agent's abilities and a task's requirements. Leveraging six distinct voting methods, a pre-trained LLM, and integrating conflict-based search (CBS) for path planning, VBTA efficiently identifies and assigns the most suitable agent to each task. While existing approaches focus on generating individual aspects of gameplay, such as single quests, or combat encounters, our method shows promise when generating both unique combat encounters and narratives because of its generalizable nature. 

**Abstract (ZH)**: 在角色扮演游戏中，沉浸感的水平至关重要，特别是在游戏代理向玩家传达任务、提示或理念时。为了使代理能够准确地解读玩家的情感状态和情境细微差别，需要具备一定的理解基础，这可以通过使用大规模语言模型（LLM）来实现。然而，维持LLM在多次情境变化中的专注度则需要更为 robust 的方法，比如将LLM与专门的任务分配模型集成，以引导其在游戏过程中的表现。为了应对这一需求，我们引入了一种基于投票的任务分配框架（VBTA），该框架受到人类任务分配与完成过程中推理的启发。VBTA为代理分配能力配置文件，并为任务分配任务描述，然后生成一个适合性矩阵，该矩阵量化了代理能力与任务需求之间的对齐程度。利用六种不同的投票方法、一个预训练的LLM以及结合冲突基于搜索（CBS）进行路径规划，VBTA能够有效地识别并分配最适合的代理来执行每个任务。尽管现有方法侧重于生成游戏的单一方面，如单个任务或战斗遭遇，我们的方法因其通用性，在生成独特的战斗遭遇和叙事方面显示出潜力。 

---
# ARACNE: An LLM-Based Autonomous Shell Pentesting Agent 

**Title (ZH)**: ARACNE: 一个基于LLM的自主Shell渗透测试代理 

**Authors**: Tomas Nieponice, Veronica Valeros, Sebastian Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2502.18528)  

**Abstract**: We introduce ARACNE, a fully autonomous LLM-based pentesting agent tailored for SSH services that can execute commands on real Linux shell systems. Introduces a new agent architecture with multi-LLM model support. Experiments show that ARACNE can reach a 60\% success rate against the autonomous defender ShelLM and a 57.58\% success rate against the Over The Wire Bandit CTF challenges, improving over the state-of-the-art. When winning, the average number of actions taken by the agent to accomplish the goals was less than 5. The results show that the use of multi-LLM is a promising approach to increase accuracy in the actions. 

**Abstract (ZH)**: 基于LLM的针对SSH服务的自主_ARACNE_渗透测试代理及其多LLM架构研究 

---
