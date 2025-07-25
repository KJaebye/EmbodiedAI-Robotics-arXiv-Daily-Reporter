# Safety-Aware Optimal Scheduling for Autonomous Masonry Construction using Collaborative Heterogeneous Aerial Robots 

**Title (ZH)**: 基于协作异构 aerial 机器人的自働砌筑作业安全意识最优调度 

**Authors**: Marios-Nektarios Stamatopoulos, Shridhar Velhal, Avijit Banerjee, George Nikolakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2506.18697)  

**Abstract**: This paper presents a novel high-level task planning and optimal coordination framework for autonomous masonry construction, using a team of heterogeneous aerial robotic workers, consisting of agents with separate skills for brick placement and mortar application. This introduces new challenges in scheduling and coordination, particularly due to the mortar curing deadline required for structural bonding and ensuring the safety constraints among UAVs operating in parallel. To address this, an automated pipeline generates the wall construction plan based on the available bricks while identifying static structural dependencies and potential conflicts for safe operation. The proposed framework optimizes UAV task allocation and execution timing by incorporating dynamically coupled precedence deadline constraints that account for the curing process and static structural dependency constraints, while enforcing spatio-temporal constraints to prevent collisions and ensure safety. The primary objective of the scheduler is to minimize the overall construction makespan while minimizing logistics, traveling time between tasks, and the curing time to maintain both adhesion quality and safe workspace separation. The effectiveness of the proposed method in achieving coordinated and time-efficient aerial masonry construction is extensively validated through Gazebo simulated missions. The results demonstrate the framework's capability to streamline UAV operations, ensuring both structural integrity and safety during the construction process. 

**Abstract (ZH)**: 基于异构空中机器人工作者的自主砌体施工高阶任务规划与最优协调框架 

---
# NOVA: Navigation via Object-Centric Visual Autonomy for High-Speed Target Tracking in Unstructured GPS-Denied Environments 

**Title (ZH)**: NOVA：面向无结构GPS deny环境高速目标跟踪的以对象为中心的视觉自主导航 

**Authors**: Alessandro Saviolo, Giuseppe Loianno  

**Link**: [PDF](https://arxiv.org/pdf/2506.18689)  

**Abstract**: Autonomous aerial target tracking in unstructured and GPS-denied environments remains a fundamental challenge in robotics. Many existing methods rely on motion capture systems, pre-mapped scenes, or feature-based localization to ensure safety and control, limiting their deployment in real-world conditions. We introduce NOVA, a fully onboard, object-centric framework that enables robust target tracking and collision-aware navigation using only a stereo camera and an IMU. Rather than constructing a global map or relying on absolute localization, NOVA formulates perception, estimation, and control entirely in the target's reference frame. A tightly integrated stack combines a lightweight object detector with stereo depth completion, followed by histogram-based filtering to infer robust target distances under occlusion and noise. These measurements feed a visual-inertial state estimator that recovers the full 6-DoF pose of the robot relative to the target. A nonlinear model predictive controller (NMPC) plans dynamically feasible trajectories in the target frame. To ensure safety, high-order control barrier functions are constructed online from a compact set of high-risk collision points extracted from depth, enabling real-time obstacle avoidance without maps or dense representations. We validate NOVA across challenging real-world scenarios, including urban mazes, forest trails, and repeated transitions through buildings with intermittent GPS loss and severe lighting changes that disrupt feature-based localization. Each experiment is repeated multiple times under similar conditions to assess resilience, showing consistent and reliable performance. NOVA achieves agile target following at speeds exceeding 50 km/h. These results show that high-speed vision-based tracking is possible in the wild using only onboard sensing, with no reliance on external localization or environment assumptions. 

**Abstract (ZH)**: 自主无人机在无结构和GPS受限环境中的目标跟踪仍然是机器人技术中的一个基本挑战。我们介绍了NOVA，一种完全机载、目标中心化的框架，仅使用立体相机和IMU即可实现鲁棒的目标跟踪和避障导航。NOVA 不构建全局地图或依赖绝对定位，而是将感知、估计和控制全部基于目标的参考帧进行。紧凑集成的堆栈结合了轻量级物体检测和立体深度补全，随后通过基于直方图的过滤器来推断在遮挡和噪声下的稳健目标距离。这些测量值输入视觉-惯性状态估计器，恢复机器人相对于目标的全6自由度姿态。非线性模型预测控制器（NMPC）在目标参考帧中规划动态可行的轨迹。为确保安全，NOVA 在线构建高阶控制障碍函数，基于深度信息提取的一组紧凑高风险碰撞点，实现实时障碍物规避，无需地图或密集表示。在城市迷宫、森林小径和间歇性GPS丢失及严重光照变化等具有挑战性的实际场景中验证NOVA，展示了其鲁棒性和可靠性。NOVA 实现了超过50 km/h的敏捷目标跟随。这些结果表明，仅依赖机载传感器，无需外部定位或环境假设，高速基于视觉的目标跟踪在野外是可行的。 

---
# Design, fabrication and control of a cable-driven parallel robot 

**Title (ZH)**: 基于缆索驱动的并联机器人设计、制造与控制 

**Authors**: Dhruv Sorathiya, Sarthak Sahoo, Vivek Natarajan  

**Link**: [PDF](https://arxiv.org/pdf/2506.18526)  

**Abstract**: In cable driven parallel robots (CDPRs), the payload is suspended using a network of cables whose length can be controlled to maneuver the payload within the workspace. Compared to rigid link robots, CDPRs provide better maneuverability due to the flexibility of the cables and consume lesser power due to the high strength-to-weight ratio of the cables. However, amongst other things, the flexibility of the cables and the fact that they can only pull (and not push) render the dynamics of CDPRs complex. Hence advanced modelling paradigms and control algorithms must be developed to fully utilize the potential of CDPRs. Furthermore, given the complex dynamics of CDPRs, the models and control algorithms proposed for them must be validated on experimental setups to ascertain their efficacy in practice. We have recently developed an elaborate experimental setup for a CDPR with three cables and validated elementary open-loop motion planning algorithms on it. In this paper, we describe several aspects of the design and fabrication of our setup, including component selection and assembly, and present our experimental results. Our setup can reproduce complex phenomenon such as the transverse vibration of the cables seen in large CDPRs and will in the future be used to model and control such phenomenon and also to validate more sophisticated motion planning algorithms. 

**Abstract (ZH)**: 基于缆索驱动并联机器人的设计与实验研究 

---
# Radar and Event Camera Fusion for Agile Robot Ego-Motion Estimation 

**Title (ZH)**: 雷达与事件摄像头融合用于敏捷机器人自我运动估计 

**Authors**: Yang Lyu, Zhenghao Zou, Yanfeng Li, Chunhui Zhao, Quan Pan  

**Link**: [PDF](https://arxiv.org/pdf/2506.18443)  

**Abstract**: Achieving reliable ego motion estimation for agile robots, e.g., aerobatic aircraft, remains challenging because most robot sensors fail to respond timely and clearly to highly dynamic robot motions, often resulting in measurement blurring, distortion, and delays. In this paper, we propose an IMU-free and feature-association-free framework to achieve aggressive ego-motion velocity estimation of a robot platform in highly dynamic scenarios by combining two types of exteroceptive sensors, an event camera and a millimeter wave radar, First, we used instantaneous raw events and Doppler measurements to derive rotational and translational velocities directly. Without a sophisticated association process between measurement frames, the proposed method is more robust in texture-less and structureless environments and is more computationally efficient for edge computing devices. Then, in the back-end, we propose a continuous-time state-space model to fuse the hybrid time-based and event-based measurements to estimate the ego-motion velocity in a fixed-lagged smoother fashion. In the end, we validate our velometer framework extensively in self-collected experiment datasets. The results indicate that our IMU-free and association-free ego motion estimation framework can achieve reliable and efficient velocity output in challenging environments. The source code, illustrative video and dataset are available at this https URL. 

**Abstract (ZH)**: 实现敏捷机器人（例如，特技飞行器）可靠的自我运动估计仍然具有挑战性，因为大多数机器人传感器难以及时清晰地响应高度动态的机器人运动，经常导致测量模糊、失真和延迟。本文提出了一种IMU-free和特征关联free的框架，通过结合事件摄像头和毫米波雷达两种外部传感器，来实现高度动态场景下机器人平台的激进自我运动速度估计。首先，我们使用瞬时原始事件和多普勒测量直接推导出旋转和平移速度，避免了高级别的测量帧关联过程，该方法在无纹理和无结构环境中更加鲁棒，并且对于边缘计算设备具有更高的计算效率。然后，在后端，我们提出了一种连续时间状态空间模型，用于融合基于时间和事件的混合测量，以滞后固定滤波方式估计自我运动速度。最后，我们在自行收集的实验数据集中广泛验证了我们提出的velometer框架。结果表明，我们提出的IMU-free和关联free的自我运动估计框架可以在具有挑战性的环境中实现可靠的、高效的速度输出。源代码、示例视频和数据集可在以下链接获取：this https URL。 

---
# Integrating Maneuverable Planning and Adaptive Control for Robot Cart-Pushing under Disturbances 

**Title (ZH)**: 扰动下移动规划与自适应控制集成的机器人杆车推送方法 

**Authors**: Zhe Zhang, Peijia Xie, Zhirui Sun, Bingyi Xia, Bi-Ke Zhu, Jiankun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18410)  

**Abstract**: Precise and flexible cart-pushing is a challenging task for mobile robots. The motion constraints during cart-pushing and the robot's redundancy lead to complex motion planning problems, while variable payloads and disturbances present complicated dynamics. In this work, we propose a novel planning and control framework for flexible whole-body coordination and robust adaptive control. Our motion planning method employs a local coordinate representation and a novel kinematic model to solve a nonlinear optimization problem, thereby enhancing motion maneuverability by generating feasible and flexible push poses. Furthermore, we present a disturbance rejection control method to resist disturbances and reduce control errors for the complex control problem without requiring an accurate dynamic model. We validate our method through extensive experiments in simulation and real-world settings, demonstrating its superiority over existing approaches. To the best of our knowledge, this is the first work to systematically evaluate the flexibility and robustness of cart-pushing methods in experiments. The video supplement is available at this https URL. 

**Abstract (ZH)**: 精确灵活的载车推举是移动机器人面临的具有挑战性的任务。载车推举过程中的运动约束和机器人的冗余性导致了复杂的运动规划问题，而可变负载和干扰使得动力学特性变得复杂。在本工作中，我们提出了一种新颖的整体运动规划与鲁棒自适应控制框架。我们的运动规划方法采用局部坐标表示和新型动力学模型来求解非线性优化问题，从而通过生成可行且灵活的推举姿态来增强运动灵活性。此外，我们提出了一种抗干扰控制方法，能够在不需要精确动力学模型的情况下抵抗干扰并减少控制误差。我们通过广泛的仿真和现实环境中实验验证了该方法的优越性，据我们所知，这是首次系统评估载车推举方法灵活性和鲁棒性的实验工作。补充视频可在以下链接获取：this https URL。 

---
# Robotic Manipulation of a Rotating Chain with Bottom End Fixed 

**Title (ZH)**: 固定底端旋转链条的机器人操作 

**Authors**: Qi Jing Chen, Shilin Shan, Quang-Cuong Pham  

**Link**: [PDF](https://arxiv.org/pdf/2506.18355)  

**Abstract**: This paper studies the problem of using a robot arm to manipulate a uniformly rotating chain with its bottom end fixed. Existing studies have investigated ideal rotational shapes for practical applications, yet they do not discuss how these shapes can be consistently achieved through manipulation planning. Our work presents a manipulation strategy for stable and consistent shape transitions. We find that the configuration space of such a chain is homeomorphic to a three-dimensional cube. Using this property, we suggest a strategy to manipulate the chain into different configurations, specifically from one rotation mode to another, while taking stability and feasibility into consideration. We demonstrate the effectiveness of our strategy in physical experiments by successfully transitioning from rest to the first two rotation modes. The concepts explored in our work has critical applications in ensuring safety and efficiency of drill string and yarn spinning operations. 

**Abstract (ZH)**: 本文研究了使用机器人手臂操纵底端固定的均匀旋转链条的问题。现有研究探讨了这些链条在实际应用中的理想旋转形状，但未讨论如何通过操作规划一致地实现这些形状。本文提出了一种稳定且一致的形状转换操作策略。我们发现此类链条的配置空间同胚于三维立方体。利用这一性质，我们提出了一种策略，在考虑稳定性和可行性的情况下，将链条从一种旋转模式过渡到另一种旋转模式。我们通过物理实验成功地从静止状态过渡到前两种旋转模式，证明了该策略的有效性。本文所探索的概念对于确保钻具和纺纱操作的安全性和效率具有关键应用价值。 

---
# TritonZ: A Remotely Operated Underwater Rover with Manipulator Arm for Exploration and Rescue Operations 

**Title (ZH)**: TritonZ：一种配备 manipulator arm 的远程操作水下无人驾驶探测与救援載具 

**Authors**: Kawser Ahmed, Mir Shahriar Fardin, Md Arif Faysal Nayem, Fahim Hafiz, Swakkhar Shatabda  

**Link**: [PDF](https://arxiv.org/pdf/2506.18343)  

**Abstract**: The increasing demand for underwater exploration and rescue operations enforces the development of advanced wireless or semi-wireless underwater vessels equipped with manipulator arms. This paper presents the implementation of a semi-wireless underwater vehicle, "TritonZ" equipped with a manipulator arm, tailored for effective underwater exploration and rescue operations. The vehicle's compact design enables deployment in different submarine surroundings, addressing the need for wireless systems capable of navigating challenging underwater terrains. The manipulator arm can interact with the environment, allowing the robot to perform sophisticated tasks during exploration and rescue missions in emergency situations. TritonZ is equipped with various sensors such as Pi-Camera, Humidity, and Temperature sensors to send real-time environmental data. Our underwater vehicle controlled using a customized remote controller can navigate efficiently in the water where Pi-Camera enables live streaming of the surroundings. Motion control and video capture are performed simultaneously using this camera. The manipulator arm is designed to perform various tasks, similar to grasping, manipulating, and collecting underwater objects. Experimental results shows the efficacy of the proposed remotely operated vehicle in performing a variety of underwater exploration and rescue tasks. Additionally, the results show that TritonZ can maintain an average of 13.5cm/s with a minimal delay of 2-3 seconds. Furthermore, the vehicle can sustain waves underwater by maintaining its position as well as average velocity. The full project details and source code can be accessed at this link: this https URL 

**Abstract (ZH)**: 一种配备 manipulator arm 的半无线水下机器人“TritonZ”在水下探测与救援中的实施与应用 

---
# Robot Tactile Gesture Recognition Based on Full-body Modular E-skin 

**Title (ZH)**: 基于全身模块化电子皮肤的机器人触觉手势识别 

**Authors**: Shuo Jiang, Boce Hu, Linfeng Zhao, Lawson L.S. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2506.18256)  

**Abstract**: With the development of robot electronic skin technology, various tactile sensors, enhanced by AI, are unlocking a new dimension of perception for robots. In this work, we explore how robots equipped with electronic skin can recognize tactile gestures and interpret them as human commands. We developed a modular robot E-skin, composed of multiple irregularly shaped skin patches, which can be assembled to cover the robot's body while capturing real-time pressure and pose data from thousands of sensing points. To process this information, we propose an equivariant graph neural network-based recognizer that efficiently and accurately classifies diverse tactile gestures, including poke, grab, stroke, and double-pat. By mapping the recognized gestures to predefined robot actions, we enable intuitive human-robot interaction purely through tactile input. 

**Abstract (ZH)**: 随着机器人电子皮肤技术的发展，增强人工智能的各种触觉传感器正在为机器人解锁新的感知维度。在本研究中，我们探讨了装备有电子皮肤的机器人如何识别触觉手势并将其解释为人机命令。我们开发了一种模块化电子皮肤机器人，由多个不规则形状的皮肤贴片组成，可以组装覆盖机器人身体，并从数千个传感点捕获实时压力和姿态数据。为了处理这些信息，我们提出了一种基于等变图神经网络的识别器，能够高效准确地分类包括戳、抓取、划过和双击等各种触觉手势。通过将识别的手势映射到预定义的机器人动作，我们实现了纯基于触觉输入的人机直观交互。 

---
# Automated Plan Refinement for Improving Efficiency of Robotic Layup of Composite Sheets 

**Title (ZH)**: 自动计划细化以提高复合板材铺层机器人作业效率 

**Authors**: Rutvik Patel, Alec Kanyuck, Zachary McNulty, Zeren Yu, Lisa Carlson, Vann Heng, Brice Johnson, Satyandra K. Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.18160)  

**Abstract**: The automation of composite sheet layup is essential to meet the increasing demand for composite materials in various industries. However, draping plans for the robotic layup of composite sheets are not robust. A plan that works well under a certain condition does not work well in a different condition. Changes in operating conditions due to either changes in material properties or working environment may lead a draping plan to exhibit suboptimal performance. In this paper, we present a comprehensive framework aimed at refining plans based on the observed execution performance. Our framework prioritizes the minimization of uncompacted regions while simultaneously improving time efficiency. To achieve this, we integrate human expertise with data-driven decision-making to refine expert-crafted plans for diverse production environments. We conduct experiments to validate the effectiveness of our approach, revealing significant reductions in the number of corrective paths required compared to initial expert-crafted plans. Through a combination of empirical data analysis, action-effectiveness modeling, and search-based refinement, our system achieves superior time efficiency in robotic layup. Experimental results demonstrate the efficacy of our approach in optimizing the layup process, thereby advancing the state-of-the-art in composite manufacturing automation. 

**Abstract (ZH)**: 复合板材铺层的自动化是满足各行业对复合材料日益增长需求的关键。然而，复合板材机器人铺层的成型方案不够 robust，在不同条件下表现不佳。由于材料属性或工作环境的变化导致的操作条件变化，可能会使成型方案表现出次优化性能。本文提出了一种全面的框架，旨在根据实际执行性能优化成型方案。该框架优先最小化未压实区域，同时提高时间效率。为此，我们将专业知识与数据驱动的决策相结合，为多种生产环境优化专家设计的成型方案。通过实验验证了该方法的有效性，结果显示所需纠正路径数量显著减少，相比初始专家设计的方案。通过结合经验数据分析、行动有效性建模和基于搜索的优化，该系统在机器人铺层中实现了更高的时间效率。实验结果证明了该方法在优化铺层过程方面的有效性，从而推动了复合材料制造自动化技术的发展。 

---
# RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation 

**Title (ZH)**: RoboTwin 2.0：一种具有强领域随机化的大规模数据生成器和基准测试平台，用于稳健的双臂机器人操作 

**Authors**: Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Qiwei Liang, Zixuan Li, Xianliang Lin, Yiheng Ge, Zhenyu Gu, Weiliang Deng, Yubin Guo, Tian Nian, Xuanbing Xie, Qiangyu Chen, Kailun Su, Tianling Xu, Guodong Liu, Mengkang Hu, Huan-ang Gao, Kaixuan Wang, Zhixuan Liang, Yusen Qin, Xiaokang Yang, Ping Luo, Yao Mu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18088)  

**Abstract**: Simulation-based data synthesis has emerged as a powerful paradigm for enhancing real-world robotic manipulation. However, existing synthetic datasets remain insufficient for robust bimanual manipulation due to two challenges: (1) the lack of an efficient, scalable data generation method for novel tasks, and (2) oversimplified simulation environments that fail to capture real-world complexity. We present RoboTwin 2.0, a scalable simulation framework that enables automated, large-scale generation of diverse and realistic data, along with unified evaluation protocols for dual-arm manipulation. We first construct RoboTwin-OD, a large-scale object library comprising 731 instances across 147 categories, each annotated with semantic and manipulation-relevant labels. Building on this foundation, we develop an expert data synthesis pipeline that combines multimodal large language models (MLLMs) with simulation-in-the-loop refinement to generate task-level execution code automatically. To improve sim-to-real transfer, RoboTwin 2.0 incorporates structured domain randomization along five axes: clutter, lighting, background, tabletop height and language instructions, thereby enhancing data diversity and policy robustness. We instantiate this framework across 50 dual-arm tasks spanning five robot embodiments, and pre-collect over 100,000 domain-randomized expert trajectories. Empirical results show a 10.9% gain in code generation success and improved generalization to novel real-world scenarios. A VLA model fine-tuned on our dataset achieves a 367% relative improvement (42.0% vs. 9.0%) on unseen scene real-world tasks, while zero-shot models trained solely on our synthetic data achieve a 228% relative gain, highlighting strong generalization without real-world supervision. We release the data generator, benchmark, dataset, and code to support scalable research in robust bimanual manipulation. 

**Abstract (ZH)**: 基于模拟的数据合成已成为增强现实世界双臂操作的强大范式。然而，现有的合成数据集由于两个挑战而不足以实现稳健的双臂操作：(1) 缺乏一种高效且可扩展的方法来生成新型任务的数据；(2) 简化的模拟环境无法捕捉现实世界的复杂性。我们提出了RoboTwin 2.0，一种可扩展的模拟框架，能够自动化、大规模生成多样且逼真的数据，并提供统一的双臂操作评估协议。首先，我们构建了RoboTwin-OD，一个包含731个实例的大型物体库，覆盖147个类别，并为每个实例标注了语义和操作相关的标签。在此基础上，我们开发了一个专家级数据合成流水线，该流水线结合了多模态大规模语言模型（MLLMs）和仿真在环改进，以自动生成任务级执行代码。为了提高模拟到现实的转移，RoboTwin 2.0引入了沿五个维度的结构化领域随机化：杂乱、照明、背景、桌面高度和语言指令，从而增强数据多样性并提高策略鲁棒性。我们在50个双臂任务上实例化了该框架，覆盖了五个机器人实体，并预先收集了超过100,000条领域随机化的专家轨迹。实证结果显示代码生成成功率提高了10.9%，并且在新颖的实际场景中具有更好的泛化能力。在我们数据集上微调的VLA模型在未见过的场景实际任务中实现了367%的相对改进（42.0% vs. 9.0%），而仅在我们合成数据上训练的零样本模型实现了228%的相对改进，突显了在没有现实世界监督的情况下强大的泛化能力。我们发布了数据生成器、基准测试、数据集和代码，以支持在稳健双臂操作方面的可扩展研究。 

---
# StereoTacTip: Vision-based Tactile Sensing with Biomimetic Skin-Marker Arrangements 

**Title (ZH)**: StereoTacTip：基于生物模仿皮肤标记排列的视觉触觉感知 

**Authors**: Chenghua Lu, Kailuan Tang, Xueming Hui, Haoran Li, Saekwang Nam, Nathan F. Lepora  

**Link**: [PDF](https://arxiv.org/pdf/2506.18040)  

**Abstract**: Vision-Based Tactile Sensors (VBTSs) stand out for their superior performance due to their high-information content output. Recently, marker-based VBTSs have been shown to give accurate geometry reconstruction when using stereo cameras. \uhl{However, many marker-based VBTSs use complex biomimetic skin-marker arrangements, which presents issues for the geometric reconstruction of the skin surface from the markers}. Here we investigate how the marker-based skin morphology affects stereo vision-based tactile sensing, using a novel VBTS called the StereoTacTip. To achieve accurate geometry reconstruction, we introduce: (i) stereo marker matching and tracking using a novel Delaunay-Triangulation-Ring-Coding algorithm; (ii) a refractive depth correction model that corrects the depth distortion caused by refraction in the internal media; (iii) a skin surface correction model from the marker positions, relying on an inverse calculation of normals to the skin surface; and (iv)~methods for geometry reconstruction over multiple contacts. To demonstrate these findings, we reconstruct topographic terrains on a large 3D map. Even though contributions (i) and (ii) were developed for biomimetic markers, they should improve the performance of all marker-based VBTSs. Overall, this work illustrates that a thorough understanding and evaluation of the morphologically-complex skin and marker-based tactile sensor principles are crucial for obtaining accurate geometric information. 

**Abstract (ZH)**: 基于视觉的触觉传感器（VBTS）由于其高信息含量的输出而表现出色。最近，基于标记的VBTS在使用立体摄像头时已被证明可以提供准确的几何重构。然而，许多基于标记的VBTS使用复杂的仿生标记布局，这给从标记重构皮肤表面带来了问题。我们通过一种新颖的立体触觉传感装置StereoTacTip研究了基于标记的皮肤形态对立体视觉触觉感知的影响。为了实现准确的几何重构，我们引入了：（i）使用新颖的Delaunay-Triangulation-Ring-Coding算法实现的立体标记匹配和跟踪；（ii）折射深度矫正模型，用于校正由内部介质折射引起的深度失真；（iii）基于标记位置的皮肤表面矫正模型，依赖于对皮肤表面法线的逆计算；（iv）在多接触点进行几何重构的方法。通过在大尺寸3D地图上重构地形来验证这些发现，尽管（i）和（ii）的贡献最初是为仿生标记开发的，但它们也应改善所有基于标记的VBTS的性能。总体而言，本工作表明，对形态复杂皮肤和基于标记的触觉传感器原理进行彻底的理解和评估对于获取准确的几何信息至关重要。 

---
# Newtonian and Lagrangian Neural Networks: A Comparison Towards Efficient Inverse Dynamics Identification 

**Title (ZH)**: 牛顿 Neural 网络和拉格朗日 Neural 网络：高效逆动力学识别的比较 

**Authors**: Minh Trinh, Andreas René Geist, Josefine Monnet, Stefan Vilceanu, Sebastian Trimpe, Christian Brecher  

**Link**: [PDF](https://arxiv.org/pdf/2506.17994)  

**Abstract**: Accurate inverse dynamics models are essential tools for controlling industrial robots. Recent research combines neural network regression with inverse dynamics formulations of the Newton-Euler and the Euler-Lagrange equations of motion, resulting in so-called Newtonian neural networks and Lagrangian neural networks, respectively. These physics-informed models seek to identify unknowns in the analytical equations from data. Despite their potential, current literature lacks guidance on choosing between Lagrangian and Newtonian networks. In this study, we show that when motor torques are estimated instead of directly measuring joint torques, Lagrangian networks prove less effective compared to Newtonian networks as they do not explicitly model dissipative torques. The performance of these models is compared to neural network regression on data of a MABI MAX 100 industrial robot. 

**Abstract (ZH)**: 准确的动力学逆模型是控制工业机器人的关键工具。最近的研究结合了神经网络回归与牛顿-欧拉和欧拉-拉格朗日运动方程的动力学逆形式，分别产生了所谓的牛顿神经网络和拉格朗日神经网络。这些基于物理的模型旨在从数据中识别解析方程中的未知量。尽管具有潜在优势，当前文献缺乏关于选择拉格朗日网络和牛顿网络之间差异的指导。在本研究中，我们表明，在估计电机扭矩而非直接测量关节扭矩时，拉格朗日网络的效果不如牛顿网络，因为它们没有明确 modeling 消散扭矩。这些模型的性能与针对 MABI MAX 100 工业机器人的数据进行的神经网络回归进行比较。 

---
# Embedded Flexible Circumferential Sensing for Real-Time Intraoperative Environmental Perception in Continuum Robots 

**Title (ZH)**: Continuum 机器人实时内术中环境感知的嵌入式柔性环形传感 

**Authors**: Peiyu Luo, Shilong Yao, Yuhan Chen, Max Q.-H. Meng  

**Link**: [PDF](https://arxiv.org/pdf/2506.17902)  

**Abstract**: Continuum robots have been widely adopted in robot-assisted minimally invasive surgery (RMIS) because of their compact size and high flexibility. However, their proprioceptive capabilities remain limited, particularly in narrow lumens, where lack of environmental awareness can lead to unintended tissue contact and surgical risks. To address this challenge, this work proposes a flexible annular sensor structure integrated around the vertebral disks of continuum robots. The proposed design enables real-time environmental mapping by estimating the distance between the robotic disks and the surrounding tissue, thereby facilitating safer operation through advanced control strategies. The experiment has proven that its accuracy in obstacle detection can reach 0.19 mm. Fabricated using flexible printed circuit (FPC) technology, the sensor demonstrates a modular and cost-effective design with compact dimensions and low noise interference. Its adaptable parameters allow compatibility with various continuum robot architectures, offering a promising solution for enhancing intraoperative perception and control in surgical robotics. 

**Abstract (ZH)**: 连续体机器人在辅助微创手术中的紧凑尺寸和高柔性使其得以广泛应用。然而，它们的本体感受能力仍然有限，特别是在狭窄的管腔中，缺乏环境感知会导致意外组织接触和手术风险。为解决这一挑战，本工作提出了一种集成在连续体机器人椎间盘周围的柔性环形传感器结构。所提出的设计通过估计机器人椎间盘与周围组织之间的距离，实现实时环境映射，从而通过先进的控制策略促进更安全的操作。实验结果证明，其在障碍物检测中的精度可达0.19毫米。该传感器基于柔性印刷电路（FPC）技术制造，具有模块化、成本效益高、紧凑尺寸和低噪声干扰的特点。其可调参数使其能够与各种连续体机器人架构兼容，为增强外科手术机器人内的感知和控制提供了前景广阔的有效解决方案。 

---
# Generative Grasp Detection and Estimation with Concept Learning-based Safety Criteria 

**Title (ZH)**: 基于概念学习的安全准则的生成性抓取检测与估计 

**Authors**: Al-Harith Farhad, Khalil Abuibaid, Christiane Plociennik, Achim Wagner, Martin Ruskowski  

**Link**: [PDF](https://arxiv.org/pdf/2506.17842)  

**Abstract**: Neural networks are often regarded as universal equations that can estimate any function. This flexibility, however, comes with the drawback of high complexity, rendering these networks into black box models, which is especially relevant in safety-centric applications. To that end, we propose a pipeline for a collaborative robot (Cobot) grasping algorithm that detects relevant tools and generates the optimal grasp. To increase the transparency and reliability of this approach, we integrate an explainable AI method that provides an explanation for the underlying prediction of a model by extracting the learned features and correlating them to corresponding classes from the input. These concepts are then used as additional criteria to ensure the safe handling of work tools. In this paper, we show the consistency of this approach and the criterion for improving the handover position. This approach was tested in an industrial environment, where a camera system was set up to enable a robot to pick up certain tools and objects. 

**Abstract (ZH)**: 神经网络通常被视为万能方程，能够估计任何函数。然而，这种灵活性伴随着高复杂性的缺点，使这些网络成为黑盒模型，特别是在以安全为中心的应用中更为突出。为此，我们提出了一种协作机器人（Cobot）抓取算法的管道，用于检测相关工具并生成最优抓取方式。为了提高该方法的透明度和可靠性，我们集成了一种可解释的AI方法，通过提取学习特征并与输入中的相应类别进行关联，为模型的底层预测提供解释。这些概念随后被用作额外的标准，以确保工具的安全处理。在本文中，我们展示了该方法的一致性及其改进交换单元位置的准则。该方法在工业环境中进行了测试，设置了一个摄像头系统，使机器人能够拾取特定的工具和物体。 

---
# Leveling the Playing Field: Carefully Comparing Classical and Learned Controllers for Quadrotor Trajectory Tracking 

**Title (ZH)**: 平衡竞争環境：仔细比较经典控制器与学习控制器在四旋翼轨迹跟踪中的表现 

**Authors**: Pratik Kunapuli, Jake Welde, Dinesh Jayaraman, Vijay Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.17832)  

**Abstract**: Learning-based control approaches like reinforcement learning (RL) have recently produced a slew of impressive results for tasks like quadrotor trajectory tracking and drone racing. Naturally, it is common to demonstrate the advantages of these new controllers against established methods like analytical controllers. We observe, however, that reliably comparing the performance of such very different classes of controllers is more complicated than might appear at first sight. As a case study, we take up the problem of agile tracking of an end-effector for a quadrotor with a fixed arm. We develop a set of best practices for synthesizing the best-in-class RL and geometric controllers (GC) for benchmarking. In the process, we resolve widespread RL-favoring biases in prior studies that provide asymmetric access to: (1) the task definition, in the form of an objective function, (2) representative datasets, for parameter optimization, and (3) feedforward information, describing the desired future trajectory. The resulting findings are the following: our improvements to the experimental protocol for comparing learned and classical controllers are critical, and each of the above asymmetries can yield misleading conclusions. Prior works have claimed that RL outperforms GC, but we find the gaps between the two controller classes are much smaller than previously published when accounting for symmetric comparisons. Geometric control achieves lower steady-state error than RL, while RL has better transient performance, resulting in GC performing better in relatively slow or less agile tasks, but RL performing better when greater agility is required. Finally, we open-source implementations of geometric and RL controllers for these aerial vehicles, implementing best practices for future development. Website and code is available at this https URL 

**Abstract (ZH)**: 基于学习的控制方法，如强化学习（RL），最近在四旋翼轨迹跟踪和无人机竞速等任务中取得了令人印象深刻的结果。然而，将这些新型控制器与传统方法如分析控制器进行对比时，可靠地比较不同类别的控制器性能要比表面看起来复杂得多。作为案例研究，我们探讨了四旋翼固定臂末端执行器敏捷跟踪的问题。我们开发了一套最佳实践，用于合成用于基准测试的最佳强化学习（RL）和几何控制器（GC）。在这一过程中，我们解决了先前研究中广泛存在的RL有利倾向偏差，这些偏差在以下方面提供了不对等的访问权限：（1）任务定义形式的目标函数，（2）代表性数据集用于参数优化，以及（3）前馈信息，描述期望的未来轨迹。研究结果如下：改进实验协议对于比较学习和经典控制器至关重要，上述每一项不对等均可导致误导性结论。先前的研究声称RL优于GC，但我们将对称比较后发现，两类控制器之间的差距要比之前公布的要小得多。几何控制在稳态误差方面优于RL，而RL在瞬态性能方面更佳，因此在相对缓慢或不太敏捷的任务中，几何控制表现更优；但在需要更高敏捷性的任务中，RL表现更佳。最后，我们开源了这些空中车辆的几何控制和RL控制实施代码，遵循最佳实践，以促进未来的发展。相关网站和代码可在以下链接访问：this https URL。 

---
# Learning to Dock: A Simulation-based Study on Closing the Sim2Real Gap in Autonomous Underwater Docking 

**Title (ZH)**: 基于模拟的学习泊靠：自主水下泊靠的模拟到现实差距研究 

**Authors**: Kevin Chang, Rakesh Vivekanandan, Noah Pragin, Sean Bullock, Geoffrey Hollinger  

**Link**: [PDF](https://arxiv.org/pdf/2506.17823)  

**Abstract**: Autonomous Underwater Vehicle (AUV) docking in dynamic and uncertain environments is a critical challenge for underwater robotics. Reinforcement learning is a promising method for developing robust controllers, but the disparity between training simulations and the real world, or the sim2real gap, often leads to a significant deterioration in performance. In this work, we perform a simulation study on reducing the sim2real gap in autonomous docking through training various controllers and then evaluating them under realistic disturbances. In particular, we focus on the real-world challenge of docking under different payloads that are potentially outside the original training distribution. We explore existing methods for improving robustness including randomization techniques and history-conditioned controllers. Our findings provide insights into mitigating the sim2real gap when training docking controllers. Furthermore, our work indicates areas of future research that may be beneficial to the marine robotics community. 

**Abstract (ZH)**: 自主水下车辆（AUV）在动态和不确定环境下的自主对接是一项关键挑战，对于水下机器人技术而言。强化学习是开发稳健控制器的一种有前景的方法，但训练模拟与现实世界之间的差距或sim2real鸿沟通常会导致性能显著下降。在本文中，我们通过训练各种控制器并在现实扰动下评估它们，来研究减少自主对接中sim2real鸿沟的仿真研究。特别地，我们专注于不同载荷下的对接现实世界挑战，这些载荷可能是原始训练分布之外的。我们探讨了提高稳健性的现有方法，包括随机化技术及基于历史条件的控制器。我们的研究结果为训练对接控制器时缓解sim2real鸿沟提供了见解。此外，我们的工作还指出了对未来研究有益于海洋机器人社区的研究方向。 

---
# Online Adaptation for Flying Quadrotors in Tight Formations 

**Title (ZH)**: 在线适应性控制在紧凑编队中的四旋翼飞行器 

**Authors**: Pei-An Hsieh, Kong Yao Chee, M. Ani Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2506.17488)  

**Abstract**: The task of flying in tight formations is challenging for teams of quadrotors because the complex aerodynamic wake interactions can destabilize individual team members as well as the team. Furthermore, these aerodynamic effects are highly nonlinear and fast-paced, making them difficult to model and predict. To overcome these challenges, we present L1 KNODE-DW MPC, an adaptive, mixed expert learning based control framework that allows individual quadrotors to accurately track trajectories while adapting to time-varying aerodynamic interactions during formation flights. We evaluate L1 KNODE-DW MPC in two different three-quadrotor formations and show that it outperforms several MPC baselines. Our results show that the proposed framework is capable of enabling the three-quadrotor team to remain vertically aligned in close proximity throughout the flight. These findings show that the L1 adaptive module compensates for unmodeled disturbances most effectively when paired with an accurate dynamics model. A video showcasing our framework and the physical experiments is available here: this https URL 

**Abstract (ZH)**: 适用于编队飞行的四旋翼无人机紧密编队飞行任务具有挑战性，因为复杂的气动尾流相互作用可能会导致个体成员乃至整个团队失稳。此外，这些气动效应是非线性的且变化快速，难以建模和预测。为克服这些挑战，我们提出了一种适应性混合专家学习控制框架L1 KNODE-DW MPC，它使个体四旋翼无人机能够在编队飞行过程中适应时间变化的气动相互作用的同时精确跟踪轨迹。我们在两种不同的三四旋翼无人机编队中评估了L1 KNODE-DW MPC，并证明它优于几种MPC基准算法。我们的结果表明，所提出的框架能够使三四旋翼无人机团队在整个飞行过程中保持垂直对齐但紧密接近。这些发现表明，L1自适应模块与准确的动力学模型配对时，能够最有效地补偿未建模的干扰。展示我们框架和物理实验的视频可在以下链接查看：this https URL。 

---
# Kinematic Model Optimization via Differentiable Contact Manifold for In-Space Manipulation 

**Title (ZH)**: 基于可微接触流形的空间操作动力学模型优化 

**Authors**: Abhay Negi, Omey M. Manyar, Satyandra K. Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.17458)  

**Abstract**: Robotic manipulation in space is essential for emerging applications such as debris removal and in-space servicing, assembly, and manufacturing (ISAM). A key requirement for these tasks is the ability to perform precise, contact-rich manipulation under significant uncertainty. In particular, thermal-induced deformation of manipulator links and temperature-dependent encoder bias introduce kinematic parameter errors that significantly degrade end-effector accuracy. Traditional calibration techniques rely on external sensors or dedicated calibration procedures, which can be infeasible or risky in dynamic, space-based operational scenarios.
This paper proposes a novel method for kinematic parameter estimation that only requires encoder measurements and binary contact detection. The approach focuses on estimating link thermal deformation strain and joint encoder biases by leveraging information of the contact manifold - the set of relative SE(3) poses at which contact between the manipulator and environment occurs. We present two core contributions: (1) a differentiable, learning-based model of the contact manifold, and (2) an optimization-based algorithm for estimating kinematic parameters from encoder measurements at contact instances. By enabling parameter estimation using only encoder measurements and contact detection, this method provides a robust, interpretable, and data-efficient solution for safe and accurate manipulation in the challenging conditions of space. 

**Abstract (ZH)**: 空间机器人操作对于新兴应用如太空碎片移除、在轨服务、组装与制造（ISAM）至关重要。这些任务的关键要求是在显著不确定性条件下进行精确的、接触丰富的操作。特别是在空间操作场景中，由热效应引起的机械臂支链变形和温度依赖性的编码器偏移引入了会造成末端执行器精度显著下降的运动学参数误差。传统校准技术依赖于外部传感器或专门的校准程序，但在动态的空间操作场景中，这些方法可能不切实际或存在风险。

本文提出了一种仅需使用编码器测量和二元接触检测的新方法，用于估计运动学参数。该方法侧重于通过利用接触流形（即机械臂与环境接触所发生相对SE(3)姿态的集合）信息，估计支链的热变形应变和关节编码器偏移。本文的核心贡献包括：（1）一种可微的学习型接触流形模型，（2）一种基于优化的算法，用于估计接触时刻的编码器测量值所对应的动力学参数。通过仅使用编码器测量和接触检测来估计参数，该方法为在复杂空间条件下实现安全和精确的操作提供了鲁棒、可解释且数据高效的解决方案。 

---
# CFTel: A Practical Architecture for Robust and Scalable Telerobotics with Cloud-Fog Automation 

**Title (ZH)**: CFTel：一种具有云-雾自动化技术的稳健可扩展远程机器人系统架构 

**Authors**: Thien Tran, Jonathan Kua, Minh Tran, Honghao Lyu, Thuong Hoang, Jiong Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.17991)  

**Abstract**: Telerobotics is a key foundation in autonomous Industrial Cyber-Physical Systems (ICPS), enabling remote operations across various domains. However, conventional cloud-based telerobotics suffers from latency, reliability, scalability, and resilience issues, hindering real-time performance in critical applications. Cloud-Fog Telerobotics (CFTel) builds on the Cloud-Fog Automation (CFA) paradigm to address these limitations by leveraging a distributed Cloud-Edge-Robotics computing architecture, enabling deterministic connectivity, deterministic connected intelligence, and deterministic networked computing. This paper synthesizes recent advancements in CFTel, aiming to highlight its role in facilitating scalable, low-latency, autonomous, and AI-driven telerobotics. We analyze architectural frameworks and technologies that enable them, including 5G Ultra-Reliable Low-Latency Communication, Edge Intelligence, Embodied AI, and Digital Twins. The study demonstrates that CFTel has the potential to enhance real-time control, scalability, and autonomy while supporting service-oriented solutions. We also discuss practical challenges, including latency constraints, cybersecurity risks, interoperability issues, and standardization efforts. This work serves as a foundational reference for researchers, stakeholders, and industry practitioners in future telerobotics research. 

**Abstract (ZH)**: 云雾协作远程操作（云雾远程操作，CFTel）是自主工业网络物理系统（ICPS）的关键基础，支持各领域内的远程操作。然而，传统的基于云的远程操作面临延迟、可靠性和扩展性等问题，限制了其在关键应用中的实时性能。云雾协作远程操作（CFTel）基于云雾自动化（CFA）范式，通过利用分布式云-边缘-机器人计算架构，解决这些限制，实现了确定性连接、确定性连接智能和确定性网络计算。本文综合了CFTel领域的最新进展，旨在突出其在实现可扩展、低延迟、自主和AI驱动的远程操作中的作用。本文分析了使其实现的架构框架和技术，包括5G超可靠低延迟通信、边缘智能、具身AI和数字孪生。研究证明，CFTel有望提高实时控制、可扩展性和自主性，同时支持面向服务的解决方案。同时，本文讨论了实际挑战，包括延迟限制、网络安全风险、互操作性问题和标准化努力。本文为未来远程操作研究中的研究人员、利益相关者和行业 practitioner 提供了一个基础参考。 

---
# Conformal Safety Shielding for Imperfect-Perception Agents 

**Title (ZH)**: Imperfect-Perception 代理的齐性安全性屏蔽 

**Authors**: William Scarbro, Calum Imrie, Sinem Getir Yaman, Kavan Fatehi, Corina S. Pasareanu, Radu Calinescu, Ravi Mangal  

**Link**: [PDF](https://arxiv.org/pdf/2506.17275)  

**Abstract**: We consider the problem of safe control in discrete autonomous agents that use learned components for imperfect perception (or more generally, state estimation) from high-dimensional observations. We propose a shield construction that provides run-time safety guarantees under perception errors by restricting the actions available to an agent, modeled as a Markov decision process, as a function of the state estimates. Our construction uses conformal prediction for the perception component, which guarantees that for each observation, the predicted set of estimates includes the actual state with a user-specified probability. The shield allows an action only if it is allowed for all the estimates in the predicted set, resulting in a local safety guarantee. We also articulate and prove a global safety property of existing shield constructions for perfect-perception agents bounding the probability of reaching unsafe states if the agent always chooses actions prescribed by the shield. We illustrate our approach with a case-study of an experimental autonomous system that guides airplanes on taxiways using high-dimensional perception DNNs. 

**Abstract (ZH)**: 我们考虑具有 learned 组件进行不完美感知（或更一般地，状态估计）的离散自主代理的安全控制问题。我们提出了一种防护构造，该构造通过根据状态估计限制代理可用的动作，为感知错误提供运行时的安全保证，将代理建模为马尔可夫决策过程。该构造使用符合预测方法进行感知组件，保证对每个观测，预测的状态集包含实际状态的概率由用户指定。防护构造仅允许如果预测集中所有估计都允许该动作，从而实现局部安全性保证。我们还阐述并证明了现有完美感知代理防护构造的全局安全性属性，该属性界定了如果代理始终选择防护构造指定的动作，则到达不安全状态的概率。我们通过一个实验自主系统的案例研究说明了这种方法，该系统使用高维感知 DNN 引导飞机在滑行道上航行。 

---
