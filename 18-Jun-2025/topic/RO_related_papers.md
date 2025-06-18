# Factor-Graph-Based Passive Acoustic Navigation for Decentralized Cooperative Localization Using Bearing Elevation Depth Difference 

**Title (ZH)**: 基于因子图的被动声纳导航用于基于方位仰角深度差的分布式协同定位 

**Authors**: Kalliyan Velasco, Timothy W. McLain, Joshua G. Mangelson  

**Link**: [PDF](https://arxiv.org/pdf/2506.14690)  

**Abstract**: Accurate and scalable underwater multi-agent localization remains a critical challenge due to the constraints of underwater communication. In this work, we propose a multi-agent localization framework using a factor-graph representation that incorporates bearing, elevation, and depth difference (BEDD). Our method leverages inverted ultra-short baseline (inverted-USBL) derived azimuth and elevation measurements from incoming acoustic signals and relative depth measurements to enable cooperative localization for a multi-robot team of autonomous underwater vehicles (AUVs). We validate our approach in the HoloOcean underwater simulator with a fleet of AUVs, demonstrating improved localization accuracy compared to dead reckoning. Additionally, we investigate the impact of azimuth and elevation measurement outliers, highlighting the need for robust outlier rejection techniques for acoustic signals. 

**Abstract (ZH)**: 基于bearing、elevation和depth difference的可扩展水下多agent定位框架 

---
# GAMORA: A Gesture Articulated Meta Operative Robotic Arm for Hazardous Material Handling in Containment-Level Environments 

**Title (ZH)**: GAMORA：一种用于限制级环境危险材料处理的 gesture articulated meta-operative 机械臂 

**Authors**: Farha Abdul Wasay, Mohammed Abdul Rahman, Hania Ghouse  

**Link**: [PDF](https://arxiv.org/pdf/2506.14513)  

**Abstract**: The convergence of robotics and virtual reality (VR) has enabled safer and more efficient workflows in high-risk laboratory settings, particularly virology labs. As biohazard complexity increases, minimizing direct human exposure while maintaining precision becomes essential. We propose GAMORA (Gesture Articulated Meta Operative Robotic Arm), a novel VR-guided robotic system that enables remote execution of hazardous tasks using natural hand gestures. Unlike existing scripted automation or traditional teleoperation, GAMORA integrates the Oculus Quest 2, NVIDIA Jetson Nano, and Robot Operating System (ROS) to provide real-time immersive control, digital twin simulation, and inverse kinematics-based articulation. The system supports VR-based training and simulation while executing precision tasks in physical environments via a 3D-printed robotic arm. Inverse kinematics ensure accurate manipulation for delicate operations such as specimen handling and pipetting. The pipeline includes Unity-based 3D environment construction, real-time motion planning, and hardware-in-the-loop testing. GAMORA achieved a mean positional discrepancy of 2.2 mm (improved from 4 mm), pipetting accuracy within 0.2 mL, and repeatability of 1.2 mm across 50 trials. Integrated object detection via YOLOv8 enhances spatial awareness, while energy-efficient operation (50% reduced power output) ensures sustainable deployment. The system's digital-physical feedback loop enables safe, precise, and repeatable automation of high-risk lab tasks. GAMORA offers a scalable, immersive solution for robotic control and biosafety in biomedical research environments. 

**Abstract (ZH)**: 机器人技术和虚拟现实的融合已在高风险实验室环境中，尤其是病毒学实验室中实现了更安全、更高效的 workflows。随着生物危害的复杂性增加，如何在保持精准度的同时最小化直接的人体暴露变得至关重要。我们提出了一种名为GAMORA（Gesture Articulated Meta Operative Robotic Arm）的新型VR指导机器人系统，该系统利用自然手势远程执行危险任务。GAMORA集成了Oculus Quest 2、NVIDIA Jetson Nano和Robot Operating System (ROS)，提供实时沉浸式控制、数字孪生模拟和基于逆运动学的操作。该系统支持基于VR的培训和模拟，并通过3D打印的机械臂在物理环境中执行精准任务。逆运动学确保了复杂操作中的精确操作，如样本处理和移液。管道包括基于Unity的3D环境构建、实时运动规划和硬件在环测试。GAMORA实现了2.2毫米的平均位置偏差（改进前为4毫米）、移液精度在0.2毫升以内，并且在50次试验中重复性为1.2毫米。通过YOLOv8集成对象检测增强了空间意识，而高效的能源操作（功率输出减少50%）确保了可持续部署。该系统的数字-物理反馈回路使高风险实验室任务的安全、精准和可重复自动化成为可能。GAMORA为生物医药研究环境中的机器人控制和生物安全提供了一种可扩展的沉浸式解决方案。 

---
# ros2 fanuc interface: Design and Evaluation of a Fanuc CRX Hardware Interface in ROS2 

**Title (ZH)**: ROS2 Fanuc 接口：ROS2 中 Fanuc CRX 硬件接口的设计与评估 

**Authors**: Paolo Franceschi, Marco Faroni, Stefano Baraldo, Anna Valente  

**Link**: [PDF](https://arxiv.org/pdf/2506.14487)  

**Abstract**: This paper introduces the ROS2 control and the Hardware Interface (HW) integration for the Fanuc CRX- robot family. It explains basic implementation details and communication protocols, and its integration with the Moveit2 motion planning library. We conducted a series of experiments to evaluate relevant performances in the robotics field. We tested the developed ros2_fanuc_interface for four relevant robotics cases: step response, trajectory tracking, collision avoidance integrated with Moveit2, and dynamic velocity scaling, respectively. Results show that, despite a non-negligible delay between command and feedback, the robot can track the defined path with negligible errors (if it complies with joint velocity limits), ensuring collision avoidance. Full code is open source and available at this https URL. 

**Abstract (ZH)**: ROS2控制与Hardware Interface (HW)集成在Fanuc CRX-机器人家族中的实现与研究：基于Moveit2的运动规划库集成实验 

---
# Automatic Cannulation of Femoral Vessels in a Porcine Shock Model 

**Title (ZH)**: 猪休克模型中自动股血管穿刺的研究 

**Authors**: Nico Zevallos, Cecilia G. Morales, Andrew Orekhov, Tejas Rane, Hernando Gomez, Francis X. Guyette, Michael R. Pinsky, John Galeotti, Artur Dubrawski, Howie Choset  

**Link**: [PDF](https://arxiv.org/pdf/2506.14467)  

**Abstract**: Rapid and reliable vascular access is critical in trauma and critical care. Central vascular catheterization enables high-volume resuscitation, hemodynamic monitoring, and advanced interventions like ECMO and REBOA. While peripheral access is common, central access is often necessary but requires specialized ultrasound-guided skills, posing challenges in prehospital settings. The complexity arises from deep target vessels and the precision needed for needle placement. Traditional techniques, like the Seldinger method, demand expertise to avoid complications. Despite its importance, ultrasound-guided central access is underutilized due to limited field expertise. While autonomous needle insertion has been explored for peripheral vessels, only semi-autonomous methods exist for femoral access. This work advances toward full automation, integrating robotic ultrasound for minimally invasive emergency procedures. Our key contribution is the successful femoral vein and artery cannulation in a porcine hemorrhagic shock model. 

**Abstract (ZH)**: rapid 和可靠的血管通路在创伤和重症 care 中至关重要。中心静脉导管化使高体积复苏、血流动力学监测及体外膜氧合（ECMO）和全身性旁路（REBOA）等高级干预成为可能。虽然外周通路较为常见，但中心通路在必要时是必需的，这需要特殊的超声引导技能，在院前环境中提出了挑战。复杂性源于深部目标血管以及针头放置所需的精度。传统技术，如西德林方法，要求有专业知识以避免并发症。尽管其重要性，由于现场专业人员有限，超声引导下中心通路的应用仍然不足。虽然外周血管的自主穿刺已有所探索，但仅存在半自主的股静脉穿刺方法。本研究朝着完全自动化方向迈进，结合了微创紧急程序中的机器人超声技术。我们的主要贡献是在一种猪实验性失血性休克模型中成功实现了股静脉和动脉穿刺。 

---
# Data Driven Approach to Input Shaping for Vibration Suppression in a Flexible Robot Arm 

**Title (ZH)**: 基于数据驱动的输入成型方法用于柔性机器人臂的振动抑制 

**Authors**: Jarkko Kotaniemi, Janne Saukkoriipi, Shuai Li, Markku Suomalainen  

**Link**: [PDF](https://arxiv.org/pdf/2506.14405)  

**Abstract**: This paper presents a simple and effective method for setting parameters for an input shaper to suppress the residual vibrations in flexible robot arms using a data-driven approach. The parameters are adaptively tuned in the workspace of the robot by interpolating previously measured data of the robot's residual vibrations. Input shaping is a simple and robust technique to generate vibration-reduced shaped commands by a convolution of an impulse sequence with the desired input command. The generated impulses create waves in the material countering the natural vibrations of the system. The method is demonstrated with a flexible 3D-printed robot arm with multiple different materials, achieving a significant reduction in the residual vibrations. 

**Abstract (ZH)**: 基于数据驱动的方法在柔性机器人手臂中抑制残余振动的参数设置方法 

---
# Robust Adaptive Time-Varying Control Barrier Function with Application to Robotic Surface Treatment 

**Title (ZH)**: 鲁棒自适应时变控制屏障函数及其在机器人表面处理中的应用 

**Authors**: Yitaek Kim, Christoffer Sloth  

**Link**: [PDF](https://arxiv.org/pdf/2506.14249)  

**Abstract**: Set invariance techniques such as control barrier functions (CBFs) can be used to enforce time-varying constraints such as keeping a safe distance from dynamic objects. However, existing methods for enforcing time-varying constraints often overlook model uncertainties. To address this issue, this paper proposes a CBFs-based robust adaptive controller design endowing time-varying constraints while considering parametric uncertainty and additive disturbances. To this end, we first leverage Robust adaptive Control Barrier Functions (RaCBFs) to handle model uncertainty, along with the concept of Input-to-State Safety (ISSf) to ensure robustness towards input disturbances. Furthermore, to alleviate the inherent conservatism in robustness, we also incorporate a set membership identification scheme. We demonstrate the proposed method on robotic surface treatment that requires time-varying force bounds to ensure uniform quality, in numerical simulation and real robotic setup, showing that the quality is formally guaranteed within an acceptable range. 

**Abstract (ZH)**: 基于控制障碍函数的鲁棒自适应控制器设计：同时考虑时间varying约束、参数不确定性及增广扰动 

---
# Pose State Perception of Interventional Robot for Cardio-cerebrovascular Procedures 

**Title (ZH)**: 介入机器人在心血管和脑血管手术中的姿态状态感知 

**Authors**: Shunhan Ji, Yanxi Chen, Zhongyu Yang, Quan Zhang, Xiaohang Nie, Jingqian Sun, Yichao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14201)  

**Abstract**: In response to the increasing demand for cardiocerebrovascular interventional surgeries, precise control of interventional robots has become increasingly important. Within these complex vascular scenarios, the accurate and reliable perception of the pose state for interventional robots is particularly crucial. This paper presents a novel vision-based approach without the need of additional sensors or markers. The core of this paper's method consists of a three-part framework: firstly, a dual-head multitask U-Net model for simultaneous vessel segment and interventional robot detection; secondly, an advanced algorithm for skeleton extraction and optimization; and finally, a comprehensive pose state perception system based on geometric features is implemented to accurately identify the robot's pose state and provide strategies for subsequent control. The experimental results demonstrate the proposed method's high reliability and accuracy in trajectory tracking and pose state perception. 

**Abstract (ZH)**: 响应心脏血管介入手术需求的增加，介入机器人精确控制变得日益重要。在这些复杂的血管场景中，介入机器人的姿态状态准确可靠感知尤为关键。本文提出了一种无需额外传感器或标记的新型视觉方法。该方法的核心由三部分框架组成：首先，一种双重头多任务U-Net模型进行血管段和介入机器人的同时检测；其次，一种先进的骨架提取和优化算法；最后，基于几何特征实现全面的姿态状态感知系统，以准确识别机器人的姿态状态并为后续控制提供策略。实验结果表明，所提出的方法在轨迹跟踪和姿态状态感知方面具有高可靠性和准确性。 

---
# TACS-Graphs: Traversability-Aware Consistent Scene Graphs for Ground Robot Indoor Localization and Mapping 

**Title (ZH)**: TACS-图：考虑可达性的一致场景图，用于地面机器人室内定位与建图 

**Authors**: Jeewon Kim, Minho Oh, Hyun Myung  

**Link**: [PDF](https://arxiv.org/pdf/2506.14178)  

**Abstract**: Scene graphs have emerged as a powerful tool for robots, providing a structured representation of spatial and semantic relationships for advanced task planning. Despite their potential, conventional 3D indoor scene graphs face critical limitations, particularly under- and over-segmentation of room layers in structurally complex environments. Under-segmentation misclassifies non-traversable areas as part of a room, often in open spaces, while over-segmentation fragments a single room into overlapping segments in complex environments. These issues stem from naive voxel-based map representations that rely solely on geometric proximity, disregarding the structural constraints of traversable spaces and resulting in inconsistent room layers within scene graphs. To the best of our knowledge, this work is the first to tackle segmentation inconsistency as a challenge and address it with Traversability-Aware Consistent Scene Graphs (TACS-Graphs), a novel framework that integrates ground robot traversability with room segmentation. By leveraging traversability as a key factor in defining room boundaries, the proposed method achieves a more semantically meaningful and topologically coherent segmentation, effectively mitigating the inaccuracies of voxel-based scene graph approaches in complex environments. Furthermore, the enhanced segmentation consistency improves loop closure detection efficiency in the proposed Consistent Scene Graph-leveraging Loop Closure Detection (CoSG-LCD) leading to higher pose estimation accuracy. Experimental results confirm that the proposed approach outperforms state-of-the-art methods in terms of scene graph consistency and pose graph optimization performance. 

**Abstract (ZH)**: 基于可通行性一致场景图的时空结构表示与任务规划 

---
# Lasso Gripper: A String Shooting-Retracting Mechanism for Shape-Adaptive Grasping 

**Title (ZH)**: Lasso夹爪：一种形状自适应抓取的绳射收回机制 

**Authors**: Qiyuan Qiao, Yu Wang, Xiyu Fan, Peng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14163)  

**Abstract**: Handling oversized, variable-shaped, or delicate objects in transportation, grasping tasks is extremely challenging, mainly due to the limitations of the gripper's shape and size. This paper proposes a novel gripper, Lasso Gripper. Inspired by traditional tools like the lasso and the uurga, Lasso Gripper captures objects by launching and retracting a string. Contrary to antipodal grippers, which concentrate force on a limited area, Lasso Gripper applies uniform pressure along the length of the string for a more gentle grasp. The gripper is controlled by four motors-two for launching the string inward and two for launching it outward. By adjusting motor speeds, the size of the string loop can be tuned to accommodate objects of varying sizes, eliminating the limitations imposed by the maximum gripper separation distance. To address the issue of string tangling during rapid retraction, a specialized mechanism was incorporated. Additionally, a dynamic model was developed to estimate the string's curve, providing a foundation for the kinematic analysis of the workspace. In grasping experiments, Lasso Gripper, mounted on a robotic arm, successfully captured and transported a range of objects, including bull and horse figures as well as delicate vegetables. The demonstration video is available here: this https URL. 

**Abstract (ZH)**: 处理运输过程中尺寸过大、形状变化或易碎物体的抓取任务极为挑战，主要由于 gripper 的形状和尺寸限制。本文提出了一种新的 gripper，Lasso Gripper。受传统的牛仔绳圈和uurga工具的启发，Lasso Gripper 通过发射和回收一根绳子来捕捉物体。与集中力作用在有限区域的对称 gripper 不同，Lasso Gripper 在绳子的长度上均匀施压，以实现更温和的抓取。该 gripper 由四个电机控制——两个用于向内发射绳子，两个用于向外发射。通过调整电机速度，可以调整绳子环的大小，以适应不同尺寸的物体，从而消除 gripper 分离距离最大值的限制。为解决快速回收过程中绳子缠绕的问题，引入了一种专门的机制。此外，还开发了一个动态模型来估算绳子的曲线，为工作空间的运动学分析提供基础。在抓取实验中，安装在机器人臂上的 Lasso Gripper 成功捕捉并运输了包括牛仔和马具模型以及易碎蔬菜在内的多种物体。演示视频请参阅：this https URL。 

---
# Haptic-Based User Authentication for Tele-robotic System 

**Title (ZH)**: 基于触觉的用户认证方法用于远程机器人系统 

**Authors**: Rongyu Yu, Kan Chen, Zeyu Deng, Chen Wang, Burak Kizilkaya, Liying Emma Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.14116)  

**Abstract**: Tele-operated robots rely on real-time user behavior mapping for remote tasks, but ensuring secure authentication remains a challenge. Traditional methods, such as passwords and static biometrics, are vulnerable to spoofing and replay attacks, particularly in high-stakes, continuous interactions. This paper presents a novel anti-spoofing and anti-replay authentication approach that leverages distinctive user behavioral features extracted from haptic feedback during human-robot interactions. To evaluate our authentication approach, we collected a time-series force feedback dataset from 15 participants performing seven distinct tasks. We then developed a transformer-based deep learning model to extract temporal features from the haptic signals. By analyzing user-specific force dynamics, our method achieves over 90 percent accuracy in both user identification and task classification, demonstrating its potential for enhancing access control and identity assurance in tele-robotic systems. 

**Abstract (ZH)**: 遥操作机器人依赖于远程任务的实时用户行为映射，但确保安全认证仍是一项挑战。传统方法，如密码和静态生物特征识别，容易受到欺骗和重放攻击，特别是在高风险的连续交互中。本文提出了一种新颖的防欺骗和防重放认证方法，该方法利用从人类与机器人交互中提取的独特用户行为特征。为了评估我们的认证方法，我们从15名参与者完成的七个不同任务中收集了时间序列力反馈数据集。然后，我们开发了一个基于Transformer的深度学习模型来从触觉信号中提取时间特征。通过分析用户的特定力动力学，我们的方法在用户识别和任务分类中的准确率均超过90%，表明其在提高远程机器人系统访问控制和身份验证方面的潜力。 

---
# ReLCP: Scalable Complementarity-Based Collision Resolution for Smooth Rigid Bodies 

**Title (ZH)**: ReLCP: 基于互补性平滑刚体碰撞分辨率的可扩展方法 

**Authors**: Bryce Palmer, Hasan Metin Aktulga, Tong Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.14097)  

**Abstract**: We present a complementarity-based collision resolution algorithm for smooth, non-spherical, rigid bodies. Unlike discrete surface representation approaches, which approximate surfaces using discrete elements (e.g., tessellations or sub-spheres) with constraints between nearby faces, edges, nodes, or sub-objects, our algorithm solves a recursively generated linear complementarity problem (ReLCP) to adaptively identify potential collision locations during the collision resolution procedure. Despite adaptively and in contrast to Newton-esque schemes, we prove conditions under which the resulting solution exists and the center of mass translational and rotational dynamics are unique. Our ReLCP also converges to classical LCP-based collision resolution for sufficiently small timesteps. Because increasing the surface resolution in discrete representation methods necessitates subdividing geometry into finer elements -- leading to a super-linear increase in the number of collision constraints -- these approaches scale poorly with increased surface resolution. In contrast, our adaptive ReLCP framework begins with a single constraint per pair of nearby bodies and introduces new constraints only when unconstrained motion would lead to overlap, circumventing the oversampling required by discrete methods. By requiring one to two orders of magnitude fewer collision constraints to achieve the same surface resolution, we observe 10-100x speedup in densely packed applications. We validate our ReLCP method against multisphere and single-constraint methods, comparing convergence in a two-ellipsoid collision test, scalability and performance in a compacting ellipsoid suspension and growing bacterial colony, and stability in a taut chainmail network, highlighting our ability to achieve high-fidelity surface representations without suffering from poor scalability or artificial surface roughness. 

**Abstract (ZH)**: 基于互补性原则的光滑非球形刚体碰撞解决算法 

---
# Quadrotor Morpho-Transition: Learning vs Model-Based Control Strategies 

**Title (ZH)**: 四旋翼形态变换：学习导向 vs 模型基控制策略 

**Authors**: Ioannis Mandralis, Richard M. Murray, Morteza Gharib  

**Link**: [PDF](https://arxiv.org/pdf/2506.14039)  

**Abstract**: Quadrotor Morpho-Transition, or the act of transitioning from air to ground through mid-air transformation, involves complex aerodynamic interactions and a need to operate near actuator saturation, complicating controller design. In recent work, morpho-transition has been studied from a model-based control perspective, but these approaches remain limited due to unmodeled dynamics and the requirement for planning through contacts. Here, we train an end-to-end Reinforcement Learning (RL) controller to learn a morpho-transition policy and demonstrate successful transfer to hardware. We find that the RL control policy achieves agile landing, but only transfers to hardware if motor dynamics and observation delays are taken into account. On the other hand, a baseline MPC controller transfers out-of-the-box without knowledge of the actuator dynamics and delays, at the cost of reduced recovery from disturbances in the event of unknown actuator failures. Our work opens the way for more robust control of agile in-flight quadrotor maneuvers that require mid-air transformation. 

**Abstract (ZH)**: 四旋翼机形态转换：从空中到地面的中间形态变化涉及复杂的气动相互作用，并且需要在接近效应器饱和的情况下操作，这增加了控制器设计的复杂性。近期工作中，形态转换从基于模型的控制角度进行了研究，但这些方法由于未建模的动力学和需要通过接触进行规划而受到限制。在此，我们训练了一个端到端的强化学习（RL）控制器来学习形态转换策略，并展示了其成功转移至硬件。我们发现，RL控制策略能够实现灵活着陆，但在转移到硬件时，需要考虑电机动力学和观察延迟。相反，基准的模型预测控制（MPC）控制器在不了解效应器动力学和延迟的情况下也能直接转移，但在未知效应器故障导致的干扰情况下恢复能力较低。我们的工作为需要空中转换的敏捷空中四旋翼机动提供了更稳健的控制方法。 

---
# A Cooperative Contactless Object Transport with Acoustic Robots 

**Title (ZH)**: 协作式非接触物体传输 acoustic 机器人 

**Authors**: Narsimlu Kemsaram, Akin Delibasi, James Hardwick, Bonot Gautam, Diego Martinez Plasencia, Sriram Subramanian  

**Link**: [PDF](https://arxiv.org/pdf/2506.13957)  

**Abstract**: Cooperative transport, the simultaneous movement of an object by multiple agents, has been widely observed in biological systems such as ant colonies, which improve efficiency and adaptability in dynamic environments. Inspired by these natural phenomena, we present a novel acoustic robotic system for the transport of contactless objects in mid-air. Our system leverages phased ultrasonic transducers and a robotic control system onboard to generate localized acoustic pressure fields, enabling precise manipulation of airborne particles and robots. We categorize contactless object-transport strategies into independent transport (uncoordinated) and forward-facing cooperative transport (coordinated), drawing parallels with biological systems to optimize efficiency and robustness. The proposed system is experimentally validated by evaluating levitation stability using a microphone in the measurement lab, transport efficiency through a phase-space motion capture system, and clock synchronization accuracy via an oscilloscope. The results demonstrate the feasibility of both independent and cooperative airborne object transport. This research contributes to the field of acoustophoretic robotics, with potential applications in contactless material handling, micro-assembly, and biomedical applications. 

**Abstract (ZH)**: 基于声学的接触less空中物体运输的协作传输：一种新的机器人系统 

---
# Socially-aware Object Transportation by a Mobile Manipulator in Static Planar Environments with Obstacles 

**Title (ZH)**: 静态平面环境中有障碍物的移动 manipulator 社会意识物体运输 

**Authors**: Caio C. G. Ribeiro, Leonardo R. D. Paes, Douglas G. Macharet  

**Link**: [PDF](https://arxiv.org/pdf/2506.13953)  

**Abstract**: Socially-aware robotic navigation is essential in environments where humans and robots coexist, ensuring both safety and comfort. However, most existing approaches have been primarily developed for mobile robots, leaving a significant gap in research that addresses the unique challenges posed by mobile manipulators. In this paper, we tackle the challenge of navigating a robotic mobile manipulator, carrying a non-negligible load, within a static human-populated environment while adhering to social norms. Our goal is to develop a method that enables the robot to simultaneously manipulate an object and navigate between locations in a socially-aware manner. We propose an approach based on the Risk-RRT* framework that enables the coordinated actuation of both the mobile base and manipulator. This approach ensures collision-free navigation while adhering to human social preferences. We compared our approach in a simulated environment to socially-aware mobile-only methods applied to a mobile manipulator. The results highlight the necessity for mobile manipulator-specific techniques, with our method outperforming mobile-only approaches. Our method enabled the robot to navigate, transport an object, avoid collisions, and minimize social discomfort effectively. 

**Abstract (ZH)**: 社交意识的移动操作器导航在人机共存环境中至关重要，既保障安全又提升舒适度。然而，现有大多数方法主要针对移动机器人开发，忽略了移动操作器面临的独特挑战。在本文中，我们解决了一个带有非轻载荷的移动操作器在静态人群环境中的导航问题，同时遵守社会规范。我们的目标是开发一种方法，使机器人能够同时进行操作和在社交感念下导航。我们提出了一种基于Risk-RRT*框架的方法，能够同时协调移动基座和操作器的动作。该方法确保了无碰撞导航，并遵守人类的社会偏好。我们将我们的方法与应用到移动操作器上的仅移动模式的社交意识方法在模拟环境中进行了比较，结果强调了针对移动操作器的具体技术的必要性，我们的方法优于仅移动模式的方法。我们的方法使机器人能够有效导航、运输物体、避免碰撞并最小化社交不适。 

---
# TUM Teleoperation: Open Source Software for Remote Driving and Assistance of Automated Vehicles 

**Title (ZH)**: TUM远程驾驶与自动驾驶车辆辅助开源软件 

**Authors**: Tobias Kerbl, David Brecht, Nils Gehrke, Nijinshan Karunainayagam, Niklas Krauss, Florian Pfab, Richard Taupitz, Ines Trautmannsheimer, Xiyan Su, Maria-Magdalena Wolf, Frank Diermeyer  

**Link**: [PDF](https://arxiv.org/pdf/2506.13933)  

**Abstract**: Teleoperation is a key enabler for future mobility, supporting Automated Vehicles in rare and complex scenarios beyond the capabilities of their automation. Despite ongoing research, no open source software currently combines Remote Driving, e.g., via steering wheel and pedals, Remote Assistance through high-level interaction with automated driving software modules, and integration with a real-world vehicle for practical testing. To address this gap, we present a modular, open source teleoperation software stack that can interact with an automated driving software, e.g., Autoware, enabling Remote Assistance and Remote Driving. The software featuresstandardized interfaces for seamless integration with various real-world and simulation platforms, while allowing for flexible design of the human-machine interface. The system is designed for modularity and ease of extension, serving as a foundation for collaborative development on individual software components as well as realistic testing and user studies. To demonstrate the applicability of our software, we evaluated the latency and performance of different vehicle platforms in simulation and real-world. The source code is available on GitHub 

**Abstract (ZH)**: 远程操控是未来移动性的关键使能器，支持自动驾驶车辆在超出其自身自动化能力的罕见和复杂场景中的应用。尽管研究持续进行，目前尚无开源软件集成了远程驾驶（例如通过方向盘和踏板）和高级交互式的远程协助功能，并与真实世界中的车辆集成进行实践测试。为填补这一空白，我们提出了一种模块化、开源的远程操控软件栈，能够与自动驾驶软件（如Autoware）交互，支持远程协助和远程驾驶。该软件提供了标准化接口，以便无缝集成到各种现实世界和仿真平台，同时也允许灵活设计人机接口。该系统设计为模块化和易于扩展，作为单独软件组件协作开发、现实测试及用户研究的基础。为了展示我们软件的应用性，我们在仿真和真实世界中评估了不同车辆平台的延迟和性能。源代码可在GitHub上获取。 

---
# Sequence Modeling for Time-Optimal Quadrotor Trajectory Optimization with Sampling-based Robustness Analysis 

**Title (ZH)**: 基于采样法鲁棒性分析的四旋翼时空最优轨迹优化序列建模 

**Authors**: Katherine Mao, Hongzhan Yu, Ruipeng Zhang, Igor Spasojevic, M Ani Hsieh, Sicun Gao, Vijay Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.13915)  

**Abstract**: Time-optimal trajectories drive quadrotors to their dynamic limits, but computing such trajectories involves solving non-convex problems via iterative nonlinear optimization, making them prohibitively costly for real-time applications. In this work, we investigate learning-based models that imitate a model-based time-optimal trajectory planner to accelerate trajectory generation. Given a dataset of collision-free geometric paths, we show that modeling architectures can effectively learn the patterns underlying time-optimal trajectories. We introduce a quantitative framework to analyze local analytic properties of the learned models, and link them to the Backward Reachable Tube of the geometric tracking controller. To enhance robustness, we propose a data augmentation scheme that applies random perturbations to the input paths. Compared to classical planners, our method achieves substantial speedups, and we validate its real-time feasibility on a hardware quadrotor platform. Experiments demonstrate that the learned models generalize to previously unseen path lengths. The code for our approach can be found here: this https URL 

**Abstract (ZH)**: 基于学习的模型加速四旋翼无人机的最优轨迹生成，但计算此类轨迹涉及通过迭代非线性优化求解非凸问题，这使得它们在实时应用中成本高昂。本研究探讨了学习型模型，这些模型模仿基于模型的最优时间轨迹规划器以加速轨迹生成。给定一组无碰撞几何路径数据集，我们表明建模架构可以有效学习最优时间轨迹背后的模式。我们引入了一个定量框架来分析所学模型的局部分析性质，并将其与几何跟踪控制器的后向可达管联系起来。为了提高鲁棒性，我们提出了一种数据增强方案，该方案对输入路径应用随机扰动。与经典规划器相比，我们的方法实现了显著的速度提升，并在硬件四旋翼平台上的实时可行性得到验证。实验表明，所学模型可以泛化到未见过的路径长度。我们的方法代码可以在这里找到：this https URL。 

---
# VisLanding: Monocular 3D Perception for UAV Safe Landing via Depth-Normal Synergy 

**Title (ZH)**: VisLanding：基于深度法线协同的单目3D视觉感知的无人机安全着陆 

**Authors**: Zhuoyue Tan, Boyong He, Yuxiang Ji, Liaoni Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14525)  

**Abstract**: This paper presents VisLanding, a monocular 3D perception-based framework for safe UAV (Unmanned Aerial Vehicle) landing. Addressing the core challenge of autonomous UAV landing in complex and unknown environments, this study innovatively leverages the depth-normal synergy prediction capabilities of the Metric3D V2 model to construct an end-to-end safe landing zones (SLZ) estimation framework. By introducing a safe zone segmentation branch, we transform the landing zone estimation task into a binary semantic segmentation problem. The model is fine-tuned and annotated using the WildUAV dataset from a UAV perspective, while a cross-domain evaluation dataset is constructed to validate the model's robustness. Experimental results demonstrate that VisLanding significantly enhances the accuracy of safe zone identification through a depth-normal joint optimization mechanism, while retaining the zero-shot generalization advantages of Metric3D V2. The proposed method exhibits superior generalization and robustness in cross-domain testing compared to other approaches. Furthermore, it enables the estimation of landing zone area by integrating predicted depth and normal information, providing critical decision-making support for practical applications. 

**Abstract (ZH)**: 基于单目3D感知的VisLanding无人机安全着陆框架 

---
# Adaptive Reinforcement Learning for Unobservable Random Delays 

**Title (ZH)**: 自适应强化学习应对不可观测的随机延迟 

**Authors**: John Wikman, Alexandre Proutiere, David Broman  

**Link**: [PDF](https://arxiv.org/pdf/2506.14411)  

**Abstract**: In standard Reinforcement Learning (RL) settings, the interaction between the agent and the environment is typically modeled as a Markov Decision Process (MDP), which assumes that the agent observes the system state instantaneously, selects an action without delay, and executes it immediately. In real-world dynamic environments, such as cyber-physical systems, this assumption often breaks down due to delays in the interaction between the agent and the system. These delays can vary stochastically over time and are typically unobservable, meaning they are unknown when deciding on an action. Existing methods deal with this uncertainty conservatively by assuming a known fixed upper bound on the delay, even if the delay is often much lower. In this work, we introduce the interaction layer, a general framework that enables agents to adaptively and seamlessly handle unobservable and time-varying delays. Specifically, the agent generates a matrix of possible future actions to handle both unpredictable delays and lost action packets sent over networks. Building on this framework, we develop a model-based algorithm, Actor-Critic with Delay Adaptation (ACDA), which dynamically adjusts to delay patterns. Our method significantly outperforms state-of-the-art approaches across a wide range of locomotion benchmark environments. 

**Abstract (ZH)**: 在标准强化学习设置中，智能体与环境的交互通常被建模为马尔科夫决策过程（MDP），假设智能体能够即时观察系统状态，无延迟地选择动作并立即执行。在现实世界中的动态环境中，如网络物理系统中，由于智能体与系统之间存在交互延迟，这一假设经常失效。这些延迟随着时间随机变化且通常是不可观测的，在决定动作时无法得知。现有方法通过假设已知固定的延迟上限保守地应对这种不确定性，即使延迟往往远低于此上限。在本工作中，我们引入了交互层，这是一种通用框架，使智能体能够适应性且无缝地处理不可观测和时间变化的延迟。具体而言，智能体生成一个可能未来动作的矩阵，以应对不可预测的延迟和网络中丢失的动作包。在此框架的基础上，我们开发了基于模型的算法——延迟适应的Actor- Critic（ACDA），该算法能够动态调整以适应延迟模式。我们的方法在一系列移动基准环境中的性能显著优于现有最先进的方法。 

---
