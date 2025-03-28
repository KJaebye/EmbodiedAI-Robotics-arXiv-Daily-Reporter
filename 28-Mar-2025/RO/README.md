# Enhancing Underwater Navigation through Cross-Correlation-Aware Deep INS/DVL Fusion 

**Title (ZH)**: 基于交叉相关性意识的深耦合INS/DVL融合 underwater navigation enhancement 

**Authors**: Nadav Cohen, Itzik Klein  

**Link**: [PDF](https://arxiv.org/pdf/2503.21727)  

**Abstract**: The accurate navigation of autonomous underwater vehicles critically depends on the precision of Doppler velocity log (DVL) velocity measurements. Recent advancements in deep learning have demonstrated significant potential in improving DVL outputs by leveraging spatiotemporal dependencies across multiple sensor modalities. However, integrating these estimates into model-based filters, such as the extended Kalman filter, introduces statistical inconsistencies, most notably, cross-correlations between process and measurement noise. This paper addresses this challenge by proposing a cross-correlation-aware deep INS/DVL fusion framework. Building upon BeamsNet, a convolutional neural network designed to estimate AUV velocity using DVL and inertial data, we integrate its output into a navigation filter that explicitly accounts for the cross-correlation induced between the noise sources. This approach improves filter consistency and better reflects the underlying sensor error structure. Evaluated on two real-world underwater trajectories, the proposed method outperforms both least squares and cross-correlation-neglecting approaches in terms of state uncertainty. Notably, improvements exceed 10% in velocity and misalignment angle confidence metrics. Beyond demonstrating empirical performance, this framework provides a theoretically principled mechanism for embedding deep learning outputs within stochastic filters. 

**Abstract (ZH)**: 精确自主水下车辆的导航依赖于Doppler速度计(DVL)速度测量的精确性。基于深度学习的 Recent 进展展示了通过利用多传感器模态的空间时间依赖性来改进 DVL 输出的巨大潜力。然而，将这些估计值整合到模型滤波器中，如扩展卡尔曼滤波器中，会引入统计不一致性，尤其是在过程噪声和测量噪声之间的交叉相关性。本文通过提出一种考虑交叉相关的深度 INS/DVL 融合框架来应对这一挑战。在此基础上，我们利用为利用 DVL 和惯性数据估计 AUV 速度而设计的 BeamsNet 卷积神经网络的输出，将其整合进一个导航滤波器中，该滤波器明确考虑了由噪声源引起的交叉相关性。该方法提高了滤波器的一致性，更好地反映了传感器误差结构。在两个实际水下轨迹上的评估表明，与最小二乘法和忽视交叉相关性的方法相比，所提出的方法在状态不确定性上表现出更佳性能。特别是在速度和偏移角度的置信度指标上，改进超过10%。除了展示实证性能外，该框架还提供了一种在随机滤波器中嵌入深度学习输出的理论基础机制。 

---
# Dataset and Analysis of Long-Term Skill Acquisition in Robot-Assisted Minimally Invasive Surgery 

**Title (ZH)**: 基于机器人辅助微创手术的长期技能获取数据集与分析 

**Authors**: Yarden Sharon, Alex Geftler, Hanna Kossowsky Lev, Ilana Nisky  

**Link**: [PDF](https://arxiv.org/pdf/2503.21591)  

**Abstract**: Objective: We aim to investigate long-term robotic surgical skill acquisition among surgical residents and the effects of training intervals and fatigue on performance. Methods: For six months, surgical residents participated in three training sessions once a month, surrounding a single 26-hour hospital shift. In each shift, they participated in training sessions scheduled before, during, and after the shift. In each training session, they performed three dry-lab training tasks: Ring Tower Transfer, Knot-Tying, and Suturing. We collected a comprehensive dataset, including videos synchronized with kinematic data, activity tracking, and scans of the suturing pads. Results: We collected a dataset of 972 trials performed by 18 residents of different surgical specializations. Participants demonstrated consistent performance improvement across all tasks. In addition, we found variations in between-shift learning and forgetting across metrics and tasks, and hints for possible effects of fatigue. Conclusion: The findings from our first analysis shed light on the long-term learning processes of robotic surgical skills with extended intervals and varying levels of fatigue. Significance: This study lays the groundwork for future research aimed at optimizing training protocols and enhancing AI applications in surgery, ultimately contributing to improved patient outcomes. The dataset will be made available upon acceptance of our journal submission. 

**Abstract (ZH)**: 研究目标：我们旨在调查手术 residents 在长期机器人手术技能学习中的表现，并研究训练间隔和疲劳对表现的影响。方法：六个月内，手术 residents 每月参与三次围绕26小时住院轮班的培训，每次轮班中，他们分别在轮班前后及期间参与培训。在每次培训中，他们执行三个干实验培训任务：Ring Tower Transfer、Knot-Tying 和 Suturing。我们收集了包括与运动数据同步的视频、活动跟踪以及缝合垫扫描在内的全面数据集。结果：我们收集了18名来自不同手术专科的 residents 执行的972次试验数据。参与者在所有任务上展示了持续的表现改进。此外，我们还发现学习和遗忘在不同指标和任务上的差异，并暗示了疲劳可能的影响。结论：首次分析的发现揭示了在长周期间隔和不同疲劳水平下机器人手术技能学习的长期过程。意义：本研究为未来旨在优化培训方案和增强手术中人工智能应用的研究奠定了基础，最终有助于改善患者预后。数据集将在期刊投稿接受后提供。 

---
# Cooking Task Planning using LLM and Verified by Graph Network 

**Title (ZH)**: 使用LLM进行烹饪任务规划并由图网络验证 

**Authors**: Ryunosuke Takebayashi, Vitor Hideyo Isume, Takuya Kiyokawa, Weiwei Wan, Kensuke Harada  

**Link**: [PDF](https://arxiv.org/pdf/2503.21564)  

**Abstract**: Cooking tasks remain a challenging problem for robotics due to their complexity. Videos of people cooking are a valuable source of information for such task, but introduces a lot of variability in terms of how to translate this data to a robotic environment. This research aims to streamline this process, focusing on the task plan generation step, by using a Large Language Model (LLM)-based Task and Motion Planning (TAMP) framework to autonomously generate cooking task plans from videos with subtitles, and execute them. Conventional LLM-based task planning methods are not well-suited for interpreting the cooking video data due to uncertainty in the videos, and the risk of hallucination in its output. To address both of these problems, we explore using LLMs in combination with Functional Object-Oriented Networks (FOON), to validate the plan and provide feedback in case of failure. This combination can generate task sequences with manipulation motions that are logically correct and executable by a robot. We compare the execution of the generated plans for 5 cooking recipes from our approach against the plans generated by a few-shot LLM-only approach for a dual-arm robot setup. It could successfully execute 4 of the plans generated by our approach, whereas only 1 of the plans generated by solely using the LLM could be executed. 

**Abstract (ZH)**: 基于大型语言模型的任务规划框架在从烹饪视频生成可执行任务计划中的应用 

---
# Data-Driven Contact-Aware Control Method for Real-Time Deformable Tool Manipulation: A Case Study in the Environmental Swabbing 

**Title (ZH)**: 基于数据驱动的接触感知控制方法在实时可变形工具操作中的应用：以环境拭子为例 

**Authors**: Siavash Mahmoudi, Amirreza Davar, Dongyi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21491)  

**Abstract**: Deformable Object Manipulation (DOM) remains a critical challenge in robotics due to the complexities of developing suitable model-based control strategies. Deformable Tool Manipulation (DTM) further complicates this task by introducing additional uncertainties between the robot and its environment. While humans effortlessly manipulate deformable tools using touch and experience, robotic systems struggle to maintain stability and precision. To address these challenges, we present a novel State-Adaptive Koopman LQR (SA-KLQR) control framework for real-time deformable tool manipulation, demonstrated through a case study in environmental swab sampling for food safety. This method leverages Koopman operator-based control to linearize nonlinear dynamics while adapting to state-dependent variations in tool deformation and contact forces. A tactile-based feedback system dynamically estimates and regulates the swab tool's angle, contact pressure, and surface coverage, ensuring compliance with food safety standards. Additionally, a sensor-embedded contact pad monitors force distribution to mitigate tool pivoting and deformation, improving stability during dynamic interactions. Experimental results validate the SA-KLQR approach, demonstrating accurate contact angle estimation, robust trajectory tracking, and reliable force regulation. The proposed framework enhances precision, adaptability, and real-time control in deformable tool manipulation, bridging the gap between data-driven learning and optimal control in robotic interaction tasks. 

**Abstract (ZH)**: 变形物体操作（DOM）仍然是机器人技术中的一个关键挑战，由于开发合适的模型基于控制策略复杂性高。变形工具操作（DTM）进一步增加了这一任务的复杂性，引入了更多机器人与其环境之间的不确定性。尽管人类能够通过触觉和经验轻松操作变形工具，但机器人系统在保持稳定性和精确性方面面临困难。为应对这些挑战，我们提出了一种新的状态自适应Koopman LQR（SA-KLQR）控制框架，用于实时变形工具操作，并通过食品安全环境采样案例研究进行了演示。该方法利用Koopman算子控制来线性化非线性动态模型，并适应工具变形和接触力的状态依赖性变化。基于触觉的反馈系统动态估计和调节拭子工具的角度、接触压力和表面覆盖，确保符合食品安全标准。此外，嵌入传感器的接触垫监控力分布，以减少工具偏转和变形，提高动态交互过程中的稳定性。实验结果验证了SA-KLQR方法，展示了准确的接触角度估计、鲁棒的轨迹跟踪和可靠的力调节。所提出的框架增强了变形工具操作的精度、适应性和实时控制能力，实现了数据驱动学习与机器人交互任务最优控制之间的桥梁。 

---
# STAMICS: Splat, Track And Map with Integrated Consistency and Semantics for Dense RGB-D SLAM 

**Title (ZH)**: STAMICS: 喷射、追踪和建图并与一致性及语义集成的密集RGB-D SLAM 

**Authors**: Yongxu Wang, Xu Cao, Weiyun Yi, Zhaoxin Fan  

**Link**: [PDF](https://arxiv.org/pdf/2503.21425)  

**Abstract**: Simultaneous Localization and Mapping (SLAM) is a critical task in robotics, enabling systems to autonomously navigate and understand complex environments. Current SLAM approaches predominantly rely on geometric cues for mapping and localization, but they often fail to ensure semantic consistency, particularly in dynamic or densely populated scenes. To address this limitation, we introduce STAMICS, a novel method that integrates semantic information with 3D Gaussian representations to enhance both localization and mapping accuracy. STAMICS consists of three key components: a 3D Gaussian-based scene representation for high-fidelity reconstruction, a graph-based clustering technique that enforces temporal semantic consistency, and an open-vocabulary system that allows for the classification of unseen objects. Extensive experiments show that STAMICS significantly improves camera pose estimation and map quality, outperforming state-of-the-art methods while reducing reconstruction errors. Code will be public available. 

**Abstract (ZH)**: 基于语义信息的3D高斯表示 simultaneously localization and mapping (STAMICS): 同时定位与建图 

---
# AcL: Action Learner for Fault-Tolerant Quadruped Locomotion Control 

**Title (ZH)**: 故障 tolerant 四足行走控制的行动学习者 

**Authors**: Tianyu Xu, Yaoyu Cheng, Pinxi Shen, Lin Zhao, Electrical, Computer Engineering, National University of Singapore, Singapore, Mechanical Engineering, National University of Singapore, Singapore  

**Link**: [PDF](https://arxiv.org/pdf/2503.21401)  

**Abstract**: Quadrupedal robots can learn versatile locomotion skills but remain vulnerable when one or more joints lose power. In contrast, dogs and cats can adopt limping gaits when injured, demonstrating their remarkable ability to adapt to physical conditions. Inspired by such adaptability, this paper presents Action Learner (AcL), a novel teacher-student reinforcement learning framework that enables quadrupeds to autonomously adapt their gait for stable walking under multiple joint faults. Unlike conventional teacher-student approaches that enforce strict imitation, AcL leverages teacher policies to generate style rewards, guiding the student policy without requiring precise replication. We train multiple teacher policies, each corresponding to a different fault condition, and subsequently distill them into a single student policy with an encoder-decoder architecture. While prior works primarily address single-joint faults, AcL enables quadrupeds to walk with up to four faulty joints across one or two legs, autonomously switching between different limping gaits when faults occur. We validate AcL on a real Go2 quadruped robot under single- and double-joint faults, demonstrating fault-tolerant, stable walking, smooth gait transitions between normal and lamb gaits, and robustness against external disturbances. 

**Abstract (ZH)**: 四足机器人可以通过学习获得多样的运动技能，但在某些关节失效时仍然脆弱。相比之下，狗和猫在受伤时可以采用跛行步态，显示出它们适应物理条件的出色能力。受这种适应性启发，本文提出了一种新的教师-学生强化学习框架Action Learner (AcL)，该框架使四足机器人能够在多个关节故障的情况下自主调整步态以实现稳定行走。与传统的严格 imitation 方法不同，AcL 利用教师策略生成风格奖励，引导学生策略而不需精确复制。我们训练了多个针对不同故障条件的教师策略，并通过编码器-解码器架构将它们提炼成一个学生策略。相比于以往主要解决单关节故障的研究，AcL 允许四足机器人在一条或多条腿有多个故障关节的情况下自主切换不同的跛行步态以实现稳定行走。我们在单关节和双关节故障的真实 Go2 四足机器人上验证了 AcL，展示了其在面对外部干扰时的容错性、稳定行走能力以及正常步态与羔羊步态之间的平滑过渡。 

---
# A Data-Driven Method for INS/DVL Alignment 

**Title (ZH)**: 基于数据驱动的方法实现INS/DVL对准 

**Authors**: Guy Damari, Itzik Klein  

**Link**: [PDF](https://arxiv.org/pdf/2503.21350)  

**Abstract**: Autonomous underwater vehicles (AUVs) are sophisticated robotic platforms crucial for a wide range of applications. The accuracy of AUV navigation systems is critical to their success. Inertial sensors and Doppler velocity logs (DVL) fusion is a promising solution for long-range underwater navigation. However, the effectiveness of this fusion depends heavily on an accurate alignment between the inertial sensors and the DVL. While current alignment methods show promise, there remains significant room for improvement in terms of accuracy, convergence time, and alignment trajectory efficiency. In this research we propose an end-to-end deep learning framework for the alignment process. By leveraging deep-learning capabilities, such as noise reduction and capture of nonlinearities in the data, we show using simulative data, that our proposed approach enhances both alignment accuracy and reduces convergence time beyond current model-based methods. 

**Abstract (ZH)**: 自主水下车辆（AUVs）是广泛应用场景中有重要地位的复杂机器人平台。AUV导航系统的准确性对其成功至关重要。惯性传感器和多普勒 velocity 记录仪（DVL）的融合是一种长距离水下导航的有前景解决方案。然而，这种融合的有效性在很大程度上取决于惯性传感器和DVL之间的精确对准。尽管现有的对准方法显示出潜力，但在准确度、收敛时间和对准轨迹效率方面仍有很大的改进空间。在本研究中，我们提出了一种端到端的深度学习框架用于对准过程。通过利用深度学习能力，如噪声抑制和数据中的非线性特征捕获，我们使用模拟数据表明，我们提出的方法在提高对准准确度和减少收敛时间方面超越了当前基于模型的方法。 

---
# Lidar-only Odometry based on Multiple Scan-to-Scan Alignments over a Moving Window 

**Title (ZH)**: 基于移动窗口多扫描匹配对齐的lidar-only里程计 

**Authors**: Aaron Kurda, Simon Steuernagel, Marcus Baum  

**Link**: [PDF](https://arxiv.org/pdf/2503.21293)  

**Abstract**: Lidar-only odometry considers the pose estimation of a mobile robot based on the accumulation of motion increments extracted from consecutive lidar scans. Many existing approaches to the problem use a scan-to-map registration, which neglects the accumulation of errors within the maintained map due to drift. Other methods use a refinement step that jointly optimizes the local map on a feature basis. We propose a solution that avoids this by using multiple independent scan-to-scan Iterative Closest Points (ICP) registrations to previous scans in order to derive constraints for a pose graph. The optimization of the pose graph then not only yields an accurate estimate for the latest pose, but also enables the refinement of previous scans in the optimization window. By avoiding the need to recompute the scan-to-scan alignments, the computational load is minimized. Extensive evaluation on the public KITTI and MulRan datasets as well as on a custom automotive lidar dataset is carried out. Results show that the proposed approach achieves state-of-the-art estimation accuracy, while alleviating the mentioned issues. 

**Abstract (ZH)**: 基于lidar-only的里程计考虑的是通过连续lidar扫描提取的运动增量累积进行移动机器人姿态估计。现有的许多方法使用扫描到地图的注册，忽视了在维护地图过程中累积的由漂移引起的误差。其他方法采用在特征基础上联合优化局部地图的精化步骤。我们提出了一种解决方案，通过使用与之前扫描多次独立的Iterative Closest Points (ICP)注册来推导姿态图的约束条件，从而避免了上述方法。然后对姿态图的优化不仅提供了最新姿态的准确估计，还使优化窗口内的先前扫描得到精化。通过避免重新计算扫描到扫描的对齐，计算负载得以最小化。我们在公共的KITTIX和MulRan数据集以及一个定制的汽车lidar数据集上进行了广泛的评估。结果显示，所提出的方案在估计准确性上达到了最新水平，并解决了上述提到的问题。 

---
# An analysis of higher-order kinematics formalisms for an innovative surgical parallel robot 

**Title (ZH)**: 创新手术并联机器人高阶运动学 formalisms 分析 

**Authors**: Calin Vaida, Iosif Birlescu, Bogdan Gherman, Daniel Condurache, Damien Chablat, Doina Pisla  

**Link**: [PDF](https://arxiv.org/pdf/2503.21291)  

**Abstract**: The paper presents a novel modular hybrid parallel robot for pancreatic surgery and its higher-order kinematics derived based on various formalisms. The classical vector, homogeneous transformation matrices and dual quaternion approaches are studied for the kinematic functions using both classical differentiation and multidual algebra. The algorithms for inverse kinematics for all three studied formalisms are presented for both differentiation and multidual algebra approaches. Furthermore, these algorithms are compared based on numerical stability, execution times and number and type of mathematical functions and operators contained in each algorithm. A statistical analysis shows that there is significant improvement in execution time for the algorithms implemented using multidual algebra, while the numerical stability is appropriate for all algorithms derived based on differentiation and multidual algebra. While the implementation of the kinematic algorithms using multidual algebra shows positive results when benchmarked on a standard PC, further work is required to evaluate the multidual algorithms on hardware/software used for the modular parallel robot command and control. 

**Abstract (ZH)**: 基于多种表示形式的胰腺手术模块化混合并行机器人及其高阶运动学研究 

---
# Haptic bilateral teleoperation system for free-hand dental procedures 

**Title (ZH)**: 基于自由手操作的触觉双边遥操作系统 

**Authors**: Lorenzo Pagliara, Enrico Ferrentino, Andrea Chiacchio, Giovanni Russo  

**Link**: [PDF](https://arxiv.org/pdf/2503.21288)  

**Abstract**: Free-hand dental procedures are typically repetitive, time-consuming and require high precision and manual dexterity. Dental robots can play a key role in improving procedural accuracy and safety, enhancing patient comfort, and reducing operator workload. However, robotic solutions for free-hand procedures remain limited or completely lacking, and their acceptance is still low. To address this gap, we develop a haptic bilateral teleoperation system (HBTS) for free-hand dental procedures. The system includes a dedicated mechanical end-effector, compatible with standard clinical tools, and equipped with an endoscopic camera for improved visibility of the intervention site. By ensuring motion and force correspondence between the operator's actions and the robot's movements, monitored through visual feedback, we enhance the operator's sensory awareness and motor accuracy. Furthermore, recognizing the need to ensure procedural safety, we limit interaction forces by scaling the motion references provided to the admittance controller based solely on measured contact forces. This ensures effective force limitation in all contact states without requiring prior knowledge of the environment. The proposed HBTS is validated in a dental scaling procedure using a dental phantom. The results show that the system improves the naturalness, safety, and accuracy of teleoperation, highlighting its potential to enhance free-hand dental procedures. 

**Abstract (ZH)**: 自由手口腔手术的触觉双边遥操作系统：提高手术准确性与安全性 

---
# Output-Feedback Boundary Control of Thermally and Flow-Induced Vibrations in Slender Timoshenko Beams 

**Title (ZH)**: 基于输出反馈的 slender 瑞芝诺梁由于热和流诱导振动的边界控制 

**Authors**: Chengyi Wang, Ji Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21281)  

**Abstract**: This work is motivated by the engineering challenge of suppressing vibrations in turbine blades of aero engines, which often operate under extreme thermal conditions and high-Mach aerodynamic environments that give rise to complex vibration phenomena, commonly referred to as thermally-induced and flow-induced vibrations. Using Hamilton's variational principle, the system is modeled as a rotating slender Timoshenko beam under thermal and aerodynamic loads, described by a mixed hyperbolic-parabolic PDE system where instabilities occur both within the PDE domain and at the uncontrolled boundary, and the two types of PDEs are cascaded in the domain. For such a system, we present the state-feedback control design based on the PDE backstepping method. Recognizing that the distributed temperature gradients and structural vibrations in the Timoshenko beam are typically unmeasurable in practice, we design a state observer for the mixed hyperbolic-parabolic PDE system. Based on this observer, an output-feedback controller is then built to regulate the overall system using only available boundary measurements. In the closed-loop system, the state of the uncontrolled boundary, i.e., the furthest state from the control input, is proved to be exponentially convergent to zero, and all signals are proved as uniformly ultimately bounded. The proposed control design is validated on an aero-engine flexible blade under extreme thermal and aerodynamic conditions. 

**Abstract (ZH)**: 基于涡轮发动机热力和气动环境下的叶片振动抑制的PDE反馈控制设计 

---
# OminiAdapt: Learning Cross-Task Invariance for Robust and Environment-Aware Robotic Manipulation 

**Title (ZH)**: OminiAdapt: 学习跨任务不变性以实现鲁棒且环境意识的机器人 manipuloration 

**Authors**: Yongxu Wang, Weiyun Yi, Xinhao Kong, Wanting Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.21257)  

**Abstract**: With the rapid development of embodied intelligence, leveraging large-scale human data for high-level imitation learning on humanoid robots has become a focal point of interest in both academia and industry. However, applying humanoid robots to precision operation domains remains challenging due to the complexities they face in perception and control processes, the long-standing physical differences in morphology and actuation mechanisms between humanoid robots and humans, and the lack of task-relevant features obtained from egocentric vision. To address the issue of covariate shift in imitation learning, this paper proposes an imitation learning algorithm tailored for humanoid robots. By focusing on the primary task objectives, filtering out background information, and incorporating channel feature fusion with spatial attention mechanisms, the proposed algorithm suppresses environmental disturbances and utilizes a dynamic weight update strategy to significantly improve the success rate of humanoid robots in accomplishing target tasks. Experimental results demonstrate that the proposed method exhibits robustness and scalability across various typical task scenarios, providing new ideas and approaches for autonomous learning and control in humanoid robots. The project will be open-sourced on GitHub. 

**Abstract (ZH)**: 随着嵌入式智能的快速发展，利用大规模人类数据进行人形机器人高层模仿学习已成为学术界和工业界的焦点。然而，将人形机器人应用于精确操作领域仍然面临着感知和控制过程的复杂性、人形机器人与人类在形态和驱动机制上的长期先天差异以及从第一人称视觉中获得的相关任务特征不足的挑战。为了应对模仿学习中的协变量移位问题，本文提出了一种针对人形机器人的模仿学习算法。该算法通过聚焦主要任务目标、过滤背景信息，并结合通道特征融合与空间注意力机制，抑制环境干扰，并采用动态权重更新策略，显著提高了人形机器人完成目标任务的成功率。实验结果表明，所提出的方法在各种典型任务场景中表现出较强的鲁棒性和扩展性，为人形机器人的自主学习和控制提供了新的思路和方法。该项目将在GitHub上开源。 

---
# Dimensional optimization of single-DOF planar rigid link-flapping mechanisms for high lift and low power 

**Title (ZH)**: 单自由度平面刚性连杆拍动机制的维度优化以实现高升力和低功率 

**Authors**: Shyam Sunder Nishad, Anupam Saxena  

**Link**: [PDF](https://arxiv.org/pdf/2503.21204)  

**Abstract**: Rigid link flapping mechanisms remain the most practical choice for flapping wing micro-aerial vehicles (MAVs) to carry useful payloads and onboard batteries for free flight due to their long-term durability and reliability. However, to achieve high agility and maneuverability-like insects-MAVs with these mechanisms require significant weight reduction. One approach involves using single-DOF planar rigid linkages, which are rarely optimized dimensionally for high lift and low power so that smaller motors and batteries could be used. We integrated a mechanism simulator based on a quasistatic nonlinear finite element method with an unsteady vortex lattice method-based aerodynamic analysis tool within an optimization routine. We optimized three different mechanism topologies from the literature. As a result, significant power savings were observed up to 42% in some cases, due to increased amplitude and higher lift coefficients resulting from optimized asymmetric sweeping velocity profiles. We also conducted an uncertainty analysis that revealed the need for high manufacturing tolerances to ensure reliable mechanism performance. The presented unified computational tool also facilitates the optimal selection of MAV components based on the payload and flight time requirements. 

**Abstract (ZH)**: 柔性连接摆动机制仍然是飞行时间长且可靠的微型空中车辆（MAVs）携带有效载荷和机载电池进行自由飞行的最实用选择。然而，为了实现类似于昆虫的高敏捷性和机动性，具有这些机制的MAVs需要显著减轻重量。一种方法是使用单自由度平面刚性连杆，这些连杆很少从高升力和低功率的角度优化尺寸，以便可以使用更小的电机和电池。我们结合了一种基于拟静态非线性有限元法的机制模拟器和一种基于不均匀漩涡网法的气动分析工具，用于优化过程。我们优化了来自文献的三种不同的机制拓扑结构。结果表明，由于优化的非对称扫掠速度剖面导致升幅增加和更高的升力系数，在某些情况下观察到高达42%的功率节省。我们还进行了不确定性分析，揭示了为了确保机制性能的可靠性，需要高制造公差。所展示的统一计算工具也有助于根据有效载荷和飞行时间要求来优化MAV组件的选择。 

---
# TAGA: A Tangent-Based Reactive Approach for Socially Compliant Robot Navigation Around Human Groups 

**Title (ZH)**: 基于切线的反应式方法：面向人类群体的社会合规机器人导航 

**Authors**: Utsha Kumar Roy, Sejuti Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2503.21168)  

**Abstract**: Robot navigation in densely populated environments presents significant challenges, particularly regarding the interplay between individual and group dynamics. Current navigation models predominantly address interactions with individual pedestrians while failing to account for human groups that naturally form in real-world settings. Conversely, the limited models implementing group-aware navigation typically prioritize group dynamics at the expense of individual interactions, both of which are essential for socially appropriate navigation. This research extends an existing simulation framework to incorporate both individual pedestrians and human groups. We present Tangent Action for Group Avoidance (TAGA), a modular reactive mechanism that can be integrated with existing navigation frameworks to enhance their group-awareness capabilities. TAGA dynamically modifies robot trajectories using tangent action-based avoidance strategies while preserving the underlying model's capacity to navigate around individuals. Additionally, we introduce Group Collision Rate (GCR), a novel metric to quantitatively assess how effectively robots maintain group integrity during navigation. Through comprehensive simulation-based benchmarking, we demonstrate that integrating TAGA with state-of-the-art navigation models (ORCA, Social Force, DS-RNN, and AG-RL) reduces group intrusions by 45.7-78.6% while maintaining comparable success rates and navigation efficiency. Future work will focus on real-world implementation and validation of this approach. 

**Abstract (ZH)**: 密人群体中机器人的导航面临显著挑战，特别是个体与群体动力学之间的相互作用。当前的导航模型主要关注与单独行人的互动，而忽略了真实环境中自然形成的行人群体。相比之下，少数采用群体意识导航的模型往往在个体互动方面有所忽视，而个体互动对于社会适配的导航同样至关重要。本研究扩展了一个现有的仿真框架，以同时考虑单独行人的行为和人类群体。我们提出了Tangent Action for Group Avoidance (TAGA)，这是一种模块化的反应机制，可以根据现有的导航框架增强其群体意识能力。TAGA通过基于切线动作的规避策略动态修改机器人轨迹，同时保持基础模型绕过单独行人导航的能力。此外，我们引入了Group Collision Rate (GCR) 作为新的度量标准，以定量评估机器人在导航过程中维持群体完整性的有效性。通过全面的基于仿真的基准测试，我们证明将TAGA与最先进的导航模型（ORCA、社会力模型、DS-RNN和AG-RL）集成后，可以将群体入侵率降低45.7%-78.6%，同时保持相似的成功率和导航效率。未来的工作将集中在该方法的实际应用和验证上。 

---
# Safe Human Robot Navigation in Warehouse Scenario 

**Title (ZH)**: 仓储场景中安全的人机导航 

**Authors**: Seth Farrell, Chenghao Li, Hongzhan Yu, Ryo Yoshimitsu, Sicun Gao, Henrik I. Christensen  

**Link**: [PDF](https://arxiv.org/pdf/2503.21141)  

**Abstract**: The integration of autonomous mobile robots (AMRs) in industrial environments, particularly warehouses, has revolutionized logistics and operational efficiency. However, ensuring the safety of human workers in dynamic, shared spaces remains a critical challenge. This work proposes a novel methodology that leverages control barrier functions (CBFs) to enhance safety in warehouse navigation. By integrating learning-based CBFs with the Open Robotics Middleware Framework (OpenRMF), the system achieves adaptive and safety-enhanced controls in multi-robot, multi-agent scenarios. Experiments conducted using various robot platforms demonstrate the efficacy of the proposed approach in avoiding static and dynamic obstacles, including human pedestrians. Our experiments evaluate different scenarios in which the number of robots, robot platforms, speed, and number of obstacles are varied, from which we achieve promising performance. 

**Abstract (ZH)**: 自主移动机器人（AMRs）在工业环境中的集成，尤其是在仓库中的应用，已经革新了物流和运营效率。然而，在动态共享空间中确保人类工人的安全仍然是一个关键挑战。本工作提出了一种新颖的方法，利用控制障碍函数（CBFs）来增强仓库导航中的安全性。通过将基于学习的CBFs与Open Robotics Middleware Framework（OpenRMF）集成，系统在多机器人、多代理场景中实现了适应性和增强的安全控制。使用各种机器人平台进行的实验展示了所提方法在避免静态和动态障碍物（包括人类行人）方面的有效性。我们的实验评估了不同场景下的性能，包括机器人数量、平台类型、速度和障碍物数量的变化，取得了令人鼓舞的结果。 

---
# Fuzzy-Logic-based model predictive control: A paradigm integrating optimal and common-sense decision making 

**Title (ZH)**: 基于模糊逻辑的模型预测控制：一种结合最优与常识决策的范式 

**Authors**: Filip Surma, Anahita Jamshidnejad  

**Link**: [PDF](https://arxiv.org/pdf/2503.21065)  

**Abstract**: This paper introduces a novel concept, fuzzy-logic-based model predictive control (FLMPC), along with a multi-robot control approach for exploring unknown environments and locating targets. Traditional model predictive control (MPC) methods rely on Bayesian theory to represent environmental knowledge and optimize a stochastic cost function, often leading to high computational costs and lack of effectiveness in locating all the targets. Our approach instead leverages FLMPC and extends it to a bi-level parent-child architecture for enhanced coordination and extended decision making horizon. Extracting high-level information from probability distributions and local observations, FLMPC simplifies the optimization problem and significantly extends its operational horizon compared to other MPC methods. We conducted extensive simulations in unknown 2-dimensional environments with randomly placed obstacles and humans. We compared the performance and computation time of FLMPC against MPC with a stochastic cost function, then evaluated the impact of integrating the high-level parent FLMPC layer. The results indicate that our approaches significantly improve both performance and computation time, enhancing coordination of robots and reducing the impact of uncertainty in large-scale search and rescue environments. 

**Abstract (ZH)**: 基于模糊逻辑的模型预测控制多机器人未知环境探索与目标定位方法 

---
# Pellet-based 3D Printing of Soft Thermoplastic Elastomeric Membranes for Soft Robotic Applications 

**Title (ZH)**: 基于颗粒的3D打印软热塑性弹性体膜材料在软机器人应用中的研究 

**Authors**: Nick Willemstein, Herman van der Kooij, Ali Sadeghi  

**Link**: [PDF](https://arxiv.org/pdf/2503.20957)  

**Abstract**: Additive Manufacturing (AM) is a promising solution for handling the complexity of fabricating soft robots. However, the AM of hyperelastic materials is still challenging with limited material types. Within this work, pellet-based 3D printing of very soft thermoplastic elastomers (TPEs) was explored. Our results show that TPEs can have similar engineering stress and maximum strain as Ecoflex OO-10. These TPEs were used to 3D-print airtight thin membranes (0.2-1.2 mm), which could inflate up to a stretch of 1320\%. Combining the membrane's large expansion and softness with the 3D printing of hollow structures simplified the design of a bending actuator that can bend 180 degrees and reach a blocked force of 238 times its weight. In addition, by 3D printing TPE pellets and rigid filaments, the soft membrane could grasp objects by enveloping an object or as a sensorized sucker, which relied on the TPE's softness to conform to the object or act as a seal. In addition, the membrane of the sucker was utilized as a tactile sensor to detect an object before adhesion. These results suggest the feasibility of 3D printing soft robots by using soft TPEs and membranes as an interesting class of materials and sensorized actuators, respectively. 

**Abstract (ZH)**: 基于颗粒的3D打印超软热塑性弹性体以制造软机器人：一种有前景的解决方案 

---
# A Study of Perceived Safety for Soft Robotics in Caregiving Tasks 

**Title (ZH)**: 软机器人在照护任务中感知安全性研究 

**Authors**: Cosima du Pasquier, Jennifer Grannen, Chuer Pan, Serin L. Huber, Aliyah Smith, Monroe Kennedy, Shuran Song, Dorsa Sadigh, Allison M. Okamura  

**Link**: [PDF](https://arxiv.org/pdf/2503.20916)  

**Abstract**: In this project, we focus on human-robot interaction in caregiving scenarios like bathing, where physical contact is inevitable and necessary for proper task execution because force must be applied to the skin. Using finite element analysis, we designed a 3D-printed gripper combining positive and negative pressure for secure yet compliant handling. Preliminary tests showed it exerted a lower, more uniform pressure profile than a standard rigid gripper. In a user study, participants' trust in robots significantly increased after they experienced a brief bathing demonstration performed by a robotic arm equipped with the soft gripper. These results suggest that soft robotics can enhance perceived safety and acceptance in intimate caregiving scenarios. 

**Abstract (ZH)**: 本研究专注于沐浴等护理场景中的机器人交互，通过有限元分析设计了一个结合正压和负压的3D打印夹爪，以实现安全且柔顺的抓持。初步测试显示，该夹爪施加的压力分布更为均匀且较小，低于标准刚性夹爪。在用户研究中，参与者在体验配备软夹爪的机器人手臂进行简短沐浴演示后，对机器人的信任度显著增加。这些结果表明，软体机器人可以在亲密护理场景中提高感知安全性和接受度。 

---
# Anti Robot Speciesism 

**Title (ZH)**: 反机器人种主义 

**Authors**: Julian De Freitas, Noah Castelo, Bernd Schmitt, Miklos Sarvary  

**Link**: [PDF](https://arxiv.org/pdf/2503.20842)  

**Abstract**: Humanoid robots are a form of embodied artificial intelligence (AI) that looks and acts more and more like humans. Powered by generative AI and advances in robotics, humanoid robots can speak and interact with humans rather naturally but are still easily recognizable as robots. But how will we treat humanoids when they seem indistinguishable from humans in appearance and mind? We find a tendency (called "anti-robot" speciesism) to deny such robots humanlike capabilities, driven by motivations to accord members of the human species preferential treatment. Six experiments show that robots are denied humanlike attributes, simply because they are not biological beings and because humans want to avoid feelings of cognitive dissonance when utilizing such robots for unsavory tasks. Thus, people do not rationally attribute capabilities to perfectly humanlike robots but deny them capabilities as it suits them. 

**Abstract (ZH)**: 拟人机器人是一种体现式的人工智能（AI），在外貌和行为上越来越像人类。依靠生成式AI和机器人技术的进步，拟人机器人能够以相当自然的方式与人类交谈和互动，但仍很容易被辨认出是机器人。但在拟人机器人在外貌和心智上几乎与人类无异时，我们又将如何对待它们？我们发现了一种倾向（称为“反机器人”物种主义），即否认这些机器人类似人类的能力，这种倾向的动力是给予人类物种成员优先待遇的动机。六项实验表明，机器人被认为不具备类似人类的属性，仅仅是因为它们不是生物体，而人类在利用这些机器人完成令人不快的任务时想要避免认知失调的感觉。因此，人们并不理性地将能力赋予完全类似人类的机器人，而是出于自身利益拒绝赋予它们这些能力。 

---
# TAR: Teacher-Aligned Representations via Contrastive Learning for Quadrupedal Locomotion 

**Title (ZH)**: TAR: 通过对比学习实现与教师对齐的表示以应用于四足行走 

**Authors**: Amr Mousa, Neil Karavis, Michele Caprio, Wei Pan, Richard Allmendinger  

**Link**: [PDF](https://arxiv.org/pdf/2503.20839)  

**Abstract**: Quadrupedal locomotion via Reinforcement Learning (RL) is commonly addressed using the teacher-student paradigm, where a privileged teacher guides a proprioceptive student policy. However, key challenges such as representation misalignment between the privileged teacher and the proprioceptive-only student, covariate shift due to behavioral cloning, and lack of deployable adaptation lead to poor generalization in real-world scenarios. We propose Teacher-Aligned Representations via Contrastive Learning (TAR), a framework that leverages privileged information with self-supervised contrastive learning to bridge this gap. By aligning representations to a privileged teacher in simulation via contrastive objectives, our student policy learns structured latent spaces and exhibits robust generalization to Out-of-Distribution (OOD) scenarios, surpassing the fully privileged "Teacher". Results showed accelerated training by 2x compared to state-of-the-art baselines to achieve peak performance. OOD scenarios showed better generalization by 40 percent on average compared to existing methods. Additionally, TAR transitions seamlessly into learning during deployment without requiring privileged states, setting a new benchmark in sample-efficient, adaptive locomotion and enabling continual fine-tuning in real-world scenarios. Open-source code and videos are available at this https URL. 

**Abstract (ZH)**: 基于对比学习的教师对齐表示（TAR）：通过强化学习实现四足运动 

---
# Benchmarking Multi-Object Grasping 

**Title (ZH)**: 多目标抓取基准测试 

**Authors**: Tianze Chen, Ricardo Frumento, Giulia Pagnanelli, Gianmarco Cei, Villa Keth, Shahadding Gafarov, Jian Gong, Zihe Ye, Marco Baracca, Salvatore D'Avella, Matteo Bianchi, Yu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.20820)  

**Abstract**: In this work, we describe a multi-object grasping benchmark to evaluate the grasping and manipulation capabilities of robotic systems in both pile and surface scenarios. The benchmark introduces three robot multi-object grasping benchmarking protocols designed to challenge different aspects of robotic manipulation. These protocols are: 1) the Only-Pick-Once protocol, which assesses the robot's ability to efficiently pick multiple objects in a single attempt; 2) the Accurate pick-trnsferring protocol, which evaluates the robot's capacity to selectively grasp and transport a specific number of objects from a cluttered environment; and 3) the Pick-transferring-all protocol, which challenges the robot to clear an entire scene by sequentially grasping and transferring all available objects. These protocols are intended to be adopted by the broader robotics research community, providing a standardized method to assess and compare robotic systems' performance in multi-object grasping tasks. We establish baselines for these protocols using standard planning and perception algorithms on a Barrett hand, Robotiq parallel jar gripper, and the Pisa/IIT Softhand-2, which is a soft underactuated robotic hand. We discuss the results in relation to human performance in similar tasks we well. 

**Abstract (ZH)**: 本研究介绍了多目标抓取基准测试，以评估机器人系统在堆叠和表面场景中的抓取和操作能力。该基准测试提出了三种旨在挑战机器人操作不同方面的抓取与操作协议。这些协议包括：1）一次性抓取协议，评估机器人一次尝试中高效抓取多个对象的能力；2）精确抓取转移协议，评估机器人在杂乱环境中选择性抓取和运输特定数量对象的能力；3）全面抓取转移协议，挑战机器人依次抓取并转移场景中所有可用对象的能力。这些协议旨在被更广泛的机器人研究社区采用，提供一种标准化方法来评估和比较机器人系统在多目标抓取任务中的性能。我们使用标准规划和感知算法在Barrett手、Robotiq平行夹爪 gripper以及Pisa/IIT SoftHand-2（一种软欠驱动机器人手）上建立了这些协议的基础性能。我们还将结果与人类在类似任务中的表现进行了比较。 

---
# Neuro-Symbolic Imitation Learning: Discovering Symbolic Abstractions for Skill Learning 

**Title (ZH)**: 神经符号模仿学习：发现技能学习中的符号抽象 

**Authors**: Leon Keller, Daniel Tanneberg, Jan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2503.21406)  

**Abstract**: Imitation learning is a popular method for teaching robots new behaviors. However, most existing methods focus on teaching short, isolated skills rather than long, multi-step tasks. To bridge this gap, imitation learning algorithms must not only learn individual skills but also an abstract understanding of how to sequence these skills to perform extended tasks effectively. This paper addresses this challenge by proposing a neuro-symbolic imitation learning framework. Using task demonstrations, the system first learns a symbolic representation that abstracts the low-level state-action space. The learned representation decomposes a task into easier subtasks and allows the system to leverage symbolic planning to generate abstract plans. Subsequently, the system utilizes this task decomposition to learn a set of neural skills capable of refining abstract plans into actionable robot commands. Experimental results in three simulated robotic environments demonstrate that, compared to baselines, our neuro-symbolic approach increases data efficiency, improves generalization capabilities, and facilitates interpretability. 

**Abstract (ZH)**: 模仿学习是教学机器人新行为的一种流行方法。然而，现有方法主要集中在教授短的、孤立的技能，而不是长的、多步骤的任务。为弥合这一差距，模仿学习算法不仅要学习个体技能，还需要获得如何序列化这些技能以有效地执行扩展任务的抽象理解。本文通过提出一种神经-符号模仿学习框架来应对这一挑战。利用任务演示，系统首先学习一个符号表示，抽象了低层的状态-动作空间。learned的表示将任务分解为更易于执行的子任务，并允许系统利用符号规划生成抽象计划。随后，系统利用这种任务分解来学习一组神经技能，这些技能能够将抽象计划细化为可执行的机器人命令。在三个模拟机器人环境中的实验结果表明，与基线方法相比，我们的神经-符号方法提高了数据效率、增强了泛化能力和增强了可解释性。 

---
# UGNA-VPR: A Novel Training Paradigm for Visual Place Recognition Based on Uncertainty-Guided NeRF Augmentation 

**Title (ZH)**: UGNA-VPR：基于不确定性引导的NeRF增强的视觉地方识别新型训练范式 

**Authors**: Yehui Shen, Lei Zhang, Qingqiu Li, Xiongwei Zhao, Yue Wang, Huimin Lu, Xieyuanli Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.21338)  

**Abstract**: Visual place recognition (VPR) is crucial for robots to identify previously visited locations, playing an important role in autonomous navigation in both indoor and outdoor environments. However, most existing VPR datasets are limited to single-viewpoint scenarios, leading to reduced recognition accuracy, particularly in multi-directional driving or feature-sparse scenes. Moreover, obtaining additional data to mitigate these limitations is often expensive. This paper introduces a novel training paradigm to improve the performance of existing VPR networks by enhancing multi-view diversity within current datasets through uncertainty estimation and NeRF-based data augmentation. Specifically, we initially train NeRF using the existing VPR dataset. Then, our devised self-supervised uncertainty estimation network identifies places with high uncertainty. The poses of these uncertain places are input into NeRF to generate new synthetic observations for further training of VPR networks. Additionally, we propose an improved storage method for efficient organization of augmented and original training data. We conducted extensive experiments on three datasets and tested three different VPR backbone networks. The results demonstrate that our proposed training paradigm significantly improves VPR performance by fully utilizing existing data, outperforming other training approaches. We further validated the effectiveness of our approach on self-recorded indoor and outdoor datasets, consistently demonstrating superior results. Our dataset and code have been released at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 视觉场所识别（VPR）对于机器人识别之前访问的位置至关重要，对于室内外环境的自主导航起着重要作用。然而，现有的大多数VPR数据集局限于单视角场景，导致识别准确性降低，特别是在多方向驾驶或特征稀疏场景中。此外，获得额外数据以应对这些局限性往往代价昂贵。本文提出了一种新的训练 paradigm，通过在当前数据集中增强多视角多样性来提高现有VPR网络的性能，这种方法利用不确定性估计和基于NeRF的数据增强。我们首先使用现有VPR数据集训练NeRF。然后，我们设计的自监督不确定性估计网络识别出高不确定性的地方。将这些不确定地点的姿态输入NeRF生成新的合成观测数据，进一步用于VPR网络的训练。此外，我们提出了改进的数据存储方法，以高效组织增强和原始训练数据。我们在三个数据集上进行了广泛的实验，并测试了三种不同的VPR骨干网络。结果表明，我们提出的训练 paradigm 显着提高了VPR性能，充分利用了现有数据，优于其他训练方法。我们在自行录制的室内外数据集上进一步验证了我们方法的有效性，结果一致表现出色。我们的数据集和代码已发布在 \href{this https URL}{this https URL}。 

---
# Exploring Interference between Concurrent Skin Stretches 

**Title (ZH)**: 探索并发皮肤拉伸之间的干扰 

**Authors**: Ching Hei Cheng, Jonathan Eden, Denny Oetomo, Ying Tan  

**Link**: [PDF](https://arxiv.org/pdf/2503.21044)  

**Abstract**: Proprioception is essential for coordinating human movements and enhancing the performance of assistive robotic devices. Skin stretch feedback, which closely aligns with natural proprioception mechanisms, presents a promising method for conveying proprioceptive information. To better understand the impact of interference on skin stretch perception, we conducted a user study with 30 participants that evaluated the effect of two simultaneous skin stretches on user perception. We observed that when participants experience simultaneous skin stretch stimuli, a masking effect occurs which deteriorates perception performance in the collocated skin stretch configurations. However, the perceived workload stays the same. These findings show that interference can affect the perception of skin stretch such that multi-channel skin stretch feedback designs should avoid locating modules in close proximity. 

**Abstract (ZH)**: 本体感觉对于协调人类运动和增强辅助机器人设备性能至关重要。与自然本体感觉机制紧密吻合的皮肤拉伸反馈提供了一种有潜力的方法来传达本体感觉信息。为了更好地理解干扰对面部皮肤拉伸感知的影响，我们进行了一个包含30名参与者的用户研究，评估了双通道皮肤拉伸对用户感知的影响。我们发现，当参与者同时经历来自同一位置的皮肤拉伸刺激时，会出现掩蔽效应，从而恶化感知性能。然而，感知的工作负载保持不变。这些发现表明，干扰可以影响皮肤拉伸的感知，因此多通道皮肤拉伸反馈设计应避免将模块放置在相近位置。 

---
# Unified Multimodal Discrete Diffusion 

**Title (ZH)**: 统一多模态离散扩散 

**Authors**: Alexander Swerdlow, Mihir Prabhudesai, Siddharth Gandhi, Deepak Pathak, Katerina Fragkiadaki  

**Link**: [PDF](https://arxiv.org/pdf/2503.20853)  

**Abstract**: Multimodal generative models that can understand and generate across multiple modalities are dominated by autoregressive (AR) approaches, which process tokens sequentially from left to right, or top to bottom. These models jointly handle images, text, video, and audio for various tasks such as image captioning, question answering, and image generation. In this work, we explore discrete diffusion models as a unified generative formulation in the joint text and image domain, building upon their recent success in text generation. Discrete diffusion models offer several advantages over AR models, including improved control over quality versus diversity of generated samples, the ability to perform joint multimodal inpainting (across both text and image domains), and greater controllability in generation through guidance. Leveraging these benefits, we present the first Unified Multimodal Discrete Diffusion (UniDisc) model which is capable of jointly understanding and generating text and images for a variety of downstream tasks. We compare UniDisc to multimodal AR models, performing a scaling analysis and demonstrating that UniDisc outperforms them in terms of both performance and inference-time compute, enhanced controllability, editability, inpainting, and flexible trade-off between inference time and generation quality. Code and additional visualizations are available at this https URL. 

**Abstract (ZH)**: 多模态生成模型能够在多个模态之间进行理解和生成，目前主要依赖自回归（AR）方法，这些方法按从左到右或从上到下的顺序处理令牌。这些模型用于各种任务，包括图像字幕、问答和图像生成。在本文中，我们探讨了在联合文本和图像域中作为统一生成框架的离散扩散模型，并利用其在文本生成方面的最近成功。离散扩散模型相对于AR模型具有多方面优势，包括在质量和多样性之间改进的控制能力、跨文本和图像域的联合多模态修补能力，以及生成过程中的更大可控性。利用这些优势，我们提出了第一个统一多模态离散扩散（UniDisc）模型，该模型能够联合理解和生成文本和图像以供多种下游任务使用。我们将在多模态AR模型上比较UniDisc，并进行规模分析，证明在性能、推理时延计算、可控性编辑、修补以及推断时间和生成质量的可调性方面UniDisc均优于它们。代码和额外的可视化内容可在此处获取。 

---
# Robust Deep Reinforcement Learning in Robotics via Adaptive Gradient-Masked Adversarial Attacks 

**Title (ZH)**: 机器人领域健壮的深度强化学习通过自适应梯度蒙蔽对抗攻击 

**Authors**: Zongyuan Zhang, Tianyang Duan, Zheng Lin, Dong Huang, Zihan Fang, Zekai Sun, Ling Xiong, Hongbin Liang, Heming Cui, Yong Cui, Yue Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.20844)  

**Abstract**: Deep reinforcement learning (DRL) has emerged as a promising approach for robotic control, but its realworld deployment remains challenging due to its vulnerability to environmental perturbations. Existing white-box adversarial attack methods, adapted from supervised learning, fail to effectively target DRL agents as they overlook temporal dynamics and indiscriminately perturb all state dimensions, limiting their impact on long-term rewards. To address these challenges, we propose the Adaptive Gradient-Masked Reinforcement (AGMR) Attack, a white-box attack method that combines DRL with a gradient-based soft masking mechanism to dynamically identify critical state dimensions and optimize adversarial policies. AGMR selectively allocates perturbations to the most impactful state features and incorporates a dynamic adjustment mechanism to balance exploration and exploitation during training. Extensive experiments demonstrate that AGMR outperforms state-of-the-art adversarial attack methods in degrading the performance of the victim agent and enhances the victim agent's robustness through adversarial defense mechanisms. 

**Abstract (ZH)**: 基于梯度的自适应掩蔽强化学习（AGMR）攻击：一种白盒对抗攻击方法 

---
# In vitro 2 In vivo : Bidirectional and High-Precision Generation of In Vitro and In Vivo Neuronal Spike Data 

**Title (ZH)**: 体外与体内：双向高精度生成体外与体内神经元放电数据 

**Authors**: Masanori Shimono  

**Link**: [PDF](https://arxiv.org/pdf/2503.20841)  

**Abstract**: Neurons encode information in a binary manner and process complex signals. However, predicting or generating diverse neural activity patterns remains challenging. In vitro and in vivo studies provide distinct advantages, yet no robust computational framework seamlessly integrates both data types. We address this by applying the Transformer model, widely used in large-scale language models, to neural data. To handle binary data, we introduced Dice loss, enabling accurate cross-domain neural activity generation. Structural analysis revealed how Dice loss enhances learning and identified key brain regions facilitating high-precision data generation. Our findings support the 3Rs principle in animal research, particularly Replacement, and establish a mathematical framework bridging animal experiments and human clinical studies. This work advances data-driven neuroscience and neural activity modeling, paving the way for more ethical and effective experimental methodologies. 

**Abstract (ZH)**: 神经元以二进制方式编码信息并处理复杂信号。然而，预测或生成多样的神经活动模式仍然具有挑战性。体外和体内研究各有优势，但尚未有强大的计算框架能够无缝整合这两种数据类型。我们通过将广泛应用于大规模语言模型的Transformer模型应用于神经数据来解决这一问题。为了处理二进制数据，我们引入了Dice损失，使其能够准确地生成跨域神经活动。结构分析揭示了Dice损失如何增强学习，并确定了有助于高精度数据生成的关键脑区。我们的发现支持动物实验中的3Rs原则，特别是在替代方面，并建立了连接动物实验和人类临床研究的数学框架。这项工作推进了数据驱动的神经科学和神经活动建模，为更伦理和有效的实验方法奠定了基础。 

---
