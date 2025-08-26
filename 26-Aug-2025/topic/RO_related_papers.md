# DANCeRS: A Distributed Algorithm for Negotiating Consensus in Robot Swarms with Gaussian Belief Propagation 

**Title (ZH)**: DANCeRS：基于高斯信念传播的机器人 swarm 中一致性协商的分布式算法 

**Authors**: Aalok Patwardhan, Andrew J. Davison  

**Link**: [PDF](https://arxiv.org/pdf/2508.18153)  

**Abstract**: Robot swarms require cohesive collective behaviour to address diverse challenges, including shape formation and decision-making. Existing approaches often treat consensus in discrete and continuous decision spaces as distinct problems. We present DANCeRS, a unified, distributed algorithm leveraging Gaussian Belief Propagation (GBP) to achieve consensus in both domains. By representing a swarm as a factor graph our method ensures scalability and robustness in dynamic environments, relying on purely peer-to-peer message passing. We demonstrate the effectiveness of our general framework through two applications where agents in a swarm must achieve consensus on global behaviour whilst relying on local communication. In the first, robots must perform path planning and collision avoidance to create shape formations. In the second, we show how the same framework can be used by a group of robots to form a consensus over a set of discrete decisions. Experimental results highlight our method's scalability and efficiency compared to recent approaches to these problems making it a promising solution for multi-robot systems requiring distributed consensus. We encourage the reader to see the supplementary video demo. 

**Abstract (ZH)**: 机器人群需要具有汇聚的集体行为以应对多样化的挑战，包括形状形成和决策制定。现有方法往往将离散和连续决策空间中的共识视为两个独立的问题。我们提出了DANCeRS统一分布式算法，利用高斯信念传播（GBP）在两个领域中实现共识。通过将群簇表示为因子图，我们的方法确保在动态环境中具有可扩展性和鲁棒性，依赖于纯粹的点对点消息传递。我们通过两个应用展示了我们通用框架的有效性，在这些应用中，群中的代理必须依靠局部通信实现对全局行为的一致性。在第一个应用中，机器人必须进行路径规划和碰撞避免以形成形状。在第二个应用中，我们展示了同一框架如何用于一组机器人以形成对一组离散决策的一致性共识。实验结果突出了我们方法与解决这些问题的近期方法相比的可扩展性和效率，使其成为需要分布式一致性的多机器人系统的一种有前景的解决方案。我们鼓励读者观看补充视频演示。 

---
# Analysis of Harpy's Constrained Trotting and Jumping Maneuver 

**Title (ZH)**: 分析哈皮鸟的受限行走和跳跃机动 Maneuver 

**Authors**: Prathima Ananda Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2508.18139)  

**Abstract**: This study presents an analysis of experimental data from Harpy, a thruster-assisted bipedal robot developed at Northeastern University. The study examines data sets from trotting and jumping experiments to understand the fundamental principles governing hybrid leg-thruster locomotion. Through data analysis across multiple locomotion modes, this research reveals that Harpy achieves stable locomotion with bounded trajectories and consistent foot placement through strategic leg-thruster synergy. The results demonstrate controlled joint behavior with low torques and symmetric tracking, accurate foot placement within kinematic constraints despite phase-transition perturbations, and underactuated degree-of-freedom stability without divergence. Energy level analysis reveals that legs provide primary propulsion, while the thrusters enable additional aerial phase control. The analysis identifies critical body-leg coupling dynamics during aerial phases that require phase-specific control strategies. Consistent repeatability and symmetry across experiments validate the robustness of the hybrid actuation approach. 

**Abstract (ZH)**: 本研究分析了东北大学开发的Harpy助推双足机器人实验数据，探讨了跑步和跳跃实验数据，以理解混合腿-助推器运动原理。通过跨多种运动模式的数据分析，本研究揭示Harpy通过策略性腿部-助推器协同作用实现了稳定的有限轨迹和一致的脚部定位。结果表明，关节行为受到控制且扭矩较低、跟踪对称，尽管有相位过渡干扰，脚部仍在运动学约束内精确定位，且欠驱动自由度保持稳定而不发散。能量水平分析显示，腿部提供主要推力，助推器允许额外的空中相位控制。分析识别了空中相位期间的关键身体-腿部耦合动力学，需要特定相位的控制策略。一致的重复性和对称性验证了混合驱动方法的稳健性。 

---
# The Effects of Communication Delay on Human Performance and Neurocognitive Responses in Mobile Robot Teleoperation 

**Title (ZH)**: 通信延迟对移动机器人遥控中人类表现及神经认知反应的影响 

**Authors**: Zhaokun Chen, Wenshuo Wang, Wenzhuo Liu, Yichen Liu, Junqiang Xi  

**Link**: [PDF](https://arxiv.org/pdf/2508.18074)  

**Abstract**: Communication delays in mobile robot teleoperation adversely affect human-machine collaboration. Understanding delay effects on human operational performance and neurocognition is essential for resolving this issue. However, no previous research has explored this. To fill this gap, we conduct a human-in-the-loop experiment involving 10 participants, integrating electroencephalography (EEG) and robot behavior data under varying delays (0-500 ms in 100 ms increments) to systematically investigate these effects. Behavior analysis reveals significant performance degradation at 200-300 ms delays, affecting both task efficiency and accuracy. EEG analysis discovers features with significant delay dependence: frontal $\theta/\beta$-band and parietal $\alpha$-band power. We also identify a threshold window (100-200 ms) for early perception of delay in humans, during which these EEG features first exhibit significant differences. When delay exceeds 400 ms, all features plateau, indicating saturation of cognitive resource allocation at physiological limits. These findings provide the first evidence of perceptual and cognitive delay thresholds during teleoperation tasks in humans, offering critical neurocognitive insights for the design of delay compensation strategies. 

**Abstract (ZH)**: 移动机器人远程操作中的通信延迟影响人机协作。探索延迟对人类操作性能和神经认知影响的规律对于解决这一问题至关重要。然而，此前的研究尚未涉及此领域。为填补这一空白，我们进行了10名参与者参与的包含电encephalography（EEG）和机器人行为数据的人机闭环实验，探索不同延迟（0-500 ms，间隔100 ms）下的这些影响。行为分析显示，在200-300 ms延迟时，任务效率和准确性显著下降。EEG分析发现具有显著延迟依赖性的特征：前额θ/β频段和顶叶α频段的功率。我们还确定了一个感知延迟的阈值窗口（100-200 ms），在此期间这些EEG特征首次表现出显著差异。当延迟超过400 ms时，所有特征均达到饱和状态，表明认知资源分配在生理极限处饱和。这些发现提供了人类在远程操作任务中感知和认知延迟阈值的首个证据，为设计延迟补偿策略提供了关键的神经认知洞察。 

---
# Modeling and Control Framework for Autonomous Space Manipulator Handover Operations 

**Title (ZH)**: 自主太空 manipulator 手蒯交接操作的建模与控制框架 

**Authors**: Diego Quevedo, Sarah Hudson, Donghoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.18039)  

**Abstract**: Autonomous space robotics is poised to play a vital role in future space missions, particularly for In-space Servicing, Assembly, and Manufacturing (ISAM). A key capability in such missions is the Robot-to-Robot (R2R) handover of mission-critical objects. This work presents a dynamic model of a dual-arm space manipulator system and compares various tracking control laws. The key contributions of this work are the development of a cooperative manipulator dynamic model and the comparative analysis of control laws to support autonomous R2R handovers in ISAM scenarios. 

**Abstract (ZH)**: 自主空间机器人在future空间任务中的关键作用，特别是在空间服务、组装与制造（ISAM）领域，尤其体现在机器人到机器人（R2R）任务关键对象的手递手操作中。本文提出了一个双臂空间操作器系统的动力学模型，并比较了多种跟踪控制律。本文的主要贡献在于开发了协同操作器动力学模型，并对支持ISAM场景中自主R2R手递手操作的控制律进行了比较分析。 

---
# No Need to Look! Locating and Grasping Objects by a Robot Arm Covered with Sensitive Skin 

**Title (ZH)**: 无需观察！基于敏感皮肤覆盖的机器人手臂定位与抓取物体 

**Authors**: Karel Bartunek, Lukas Rustler, Matej Hoffmann  

**Link**: [PDF](https://arxiv.org/pdf/2508.17986)  

**Abstract**: Locating and grasping of objects by robots is typically performed using visual sensors. Haptic feedback from contacts with the environment is only secondary if present at all. In this work, we explored an extreme case of searching for and grasping objects in complete absence of visual input, relying on haptic feedback only. The main novelty lies in the use of contacts over the complete surface of a robot manipulator covered with sensitive skin. The search is divided into two phases: (1) coarse workspace exploration with the complete robot surface, followed by (2) precise localization using the end-effector equipped with a force/torque sensor. We systematically evaluated this method in simulation and on the real robot, demonstrating that diverse objects can be located, grasped, and put in a basket. The overall success rate on the real robot for one object was 85.7\% with failures mainly while grasping specific objects. The method using whole-body contacts is six times faster compared to a baseline that uses haptic feedback only on the end-effector. We also show locating and grasping multiple objects on the table. This method is not restricted to our specific setup and can be deployed on any platform with the ability of sensing contacts over the entire body surface. This work holds promise for diverse applications in areas with challenging visual perception (due to lighting, dust, smoke, occlusion) such as in agriculture when fruits or vegetables need to be located inside foliage and picked. 

**Abstract (ZH)**: 基于全身体感反馈的物体定位与抓取 

---
# CubeDN: Real-time Drone Detection in 3D Space from Dual mmWave Radar Cubes 

**Title (ZH)**: CubeDN：来自双毫米波雷达立方体的三维空间实时无人机检测 

**Authors**: Yuan Fang, Fangzhan Shi, Xijia Wei, Qingchao Chen, Kevin Chetty, Simon Julier  

**Link**: [PDF](https://arxiv.org/pdf/2508.17831)  

**Abstract**: As drone use has become more widespread, there is a critical need to ensure safety and security. A key element of this is robust and accurate drone detection and localization. While cameras and other optical sensors like LiDAR are commonly used for object detection, their performance degrades under adverse lighting and environmental conditions. Therefore, this has generated interest in finding more reliable alternatives, such as millimeter-wave (mmWave) radar. Recent research on mmWave radar object detection has predominantly focused on 2D detection of road users. Although these systems demonstrate excellent performance for 2D problems, they lack the sensing capability to measure elevation, which is essential for 3D drone detection. To address this gap, we propose CubeDN, a single-stage end-to-end radar object detection network specifically designed for flying drones. CubeDN overcomes challenges such as poor elevation resolution by utilizing a dual radar configuration and a novel deep learning pipeline. It simultaneously detects, localizes, and classifies drones of two sizes, achieving decimeter-level tracking accuracy at closer ranges with overall $95\%$ average precision (AP) and $85\%$ average recall (AR). Furthermore, CubeDN completes data processing and inference at 10Hz, making it highly suitable for practical applications. 

**Abstract (ZH)**: 随着无人机使用范围的扩大，确保安全与security的需要变得至关重要。一个重要方面是 robust 和 accurate 的无人机检测与定位。尽管相机和其他光学传感器如 LiDAR 广泛用于目标检测，但在不良光照和环境条件下，它们的表现会下降。因此，寻找更可靠替代方案，如毫米波（mmWave）雷达，引起了人们的兴趣。最近关于 mmWave 雷达目标检测的研究主要集中在2D检测道路使用者上。虽然这些系统在2D问题上表现出色，但缺乏测量海拔的能力，这是3D无人机检测必不可少的。为了解决这个差距，我们提出 CubeDN，这是一种专门用于飞行无人机的一阶段端到端雷达目标检测网络。CubeDN 通过使用双雷达配置和一种新颖的深度学习流水线，克服了低海拔分辨率等问题。它同时检测、定位和分类两种尺寸的无人机，在近距离范围内实现了分米级的跟踪精度，总体平均精度（AP）为95%，平均召回率（AR）为85%。此外，CubeDN 在10Hz 的数据处理和推理速度下运行，使其非常适用于实际应用。 

---
# Effect of Performance Feedback Timing on Motor Learning for a Surgical Training Task 

**Title (ZH)**: 手术训练任务中绩效反馈时间对运动学习的影响 

**Authors**: Mary Kate Gale, Kailana Baker-Matsuoka, Ilana Nisky, Allison Okamura  

**Link**: [PDF](https://arxiv.org/pdf/2508.17830)  

**Abstract**: Objective: Robot-assisted minimally invasive surgery (RMIS) has become the gold standard for a variety of surgical procedures, but the optimal method of training surgeons for RMIS is unknown. We hypothesized that real-time, rather than post-task, error feedback would better increase learning speed and reduce errors. Methods: Forty-two surgical novices learned a virtual version of the ring-on-wire task, a canonical task in RMIS training. We investigated the impact of feedback timing with multi-sensory (haptic and visual) cues in three groups: (1) real-time error feedback, (2) trial replay with error feedback, and (3) no error feedback. Results: Participant performance was evaluated based on the accuracy of ring position and orientation during the task. Participants who received real-time feedback outperformed other groups in ring orientation. Additionally, participants who received feedback in replay outperformed participants who did not receive any error feedback on ring orientation during long, straight path sections. There were no significant differences between groups for ring position overall, but participants who received real-time feedback outperformed the other groups in positional accuracy on tightly curved path sections. Conclusion: The addition of real-time haptic and visual error feedback improves learning outcomes in a virtual surgical task over error feedback in replay or no error feedback at all. Significance: This work demonstrates that multi-sensory error feedback delivered in real time leads to better training outcomes as compared to the same feedback delivered after task completion. This novel method of training may enable surgical trainees to develop skills with greater speed and accuracy. 

**Abstract (ZH)**: 客观目标：机器人辅助微创手术（RMIS）已成为多种外科手术的标准，但最佳的外科医生培训方法尚未确定。我们假设实时错误反馈而非任务后错误反馈能够更好地提高学习速度并减少错误。方法：42名手术初学者学习了一个虚拟的环在绳上的任务，这是RMIS培训中的一个经典任务。我们通过多感官（触觉和视觉）提示在三个组中研究了反馈时间的影响：（1）实时错误反馈，（2）试次回放带有错误反馈，（3）无错误反馈。结果：根据任务中环的位置和方向准确性评估参与者的表现。接受实时反馈的参与者在环的方向上表现最好。此外，在长直线路径段上，接受回放反馈的参与者在环的方向上表现优于未接受错误反馈的参与者。总体上，各组在环的位置上没有显著差异，但接受实时反馈的参与者在紧弯路径段上的位置准确性上表现最好。结论：实时的触觉和视觉错误反馈在虚拟外科任务中的学习效果优于任务后错误反馈或完全没有错误反馈。意义：这项工作表明，与任务完成后提供的反馈相比，实时的多感官错误反馈能够更好地提高培训效果。这一新的培训方法可以使外科训练生以更快和更准确的方式发展技能。 

---
# SEBVS: Synthetic Event-based Visual Servoing for Robot Navigation and Manipulation 

**Title (ZH)**: SEBVS: 合成事件驱动的视觉伺服在机器人导航与操作中的应用 

**Authors**: Krishna Vinod, Prithvi Jai Ramesh, Pavan Kumar B N, Bharatesh Chakravarthi  

**Link**: [PDF](https://arxiv.org/pdf/2508.17643)  

**Abstract**: Event cameras offer microsecond latency, high dynamic range, and low power consumption, making them ideal for real-time robotic perception under challenging conditions such as motion blur, occlusion, and illumination changes. However, despite their advantages, synthetic event-based vision remains largely unexplored in mainstream robotics simulators. This lack of simulation setup hinders the evaluation of event-driven approaches for robotic manipulation and navigation tasks. This work presents an open-source, user-friendly v2e robotics operating system (ROS) package for Gazebo simulation that enables seamless event stream generation from RGB camera feeds. The package is used to investigate event-based robotic policies (ERP) for real-time navigation and manipulation. Two representative scenarios are evaluated: (1) object following with a mobile robot and (2) object detection and grasping with a robotic manipulator. Transformer-based ERPs are trained by behavior cloning and compared to RGB-based counterparts under various operating conditions. Experimental results show that event-guided policies consistently deliver competitive advantages. The results highlight the potential of event-driven perception to improve real-time robotic navigation and manipulation, providing a foundation for broader integration of event cameras into robotic policy learning. The GitHub repo for the dataset and code: this https URL 

**Abstract (ZH)**: 事件相机提供了微秒级延迟、高动态范围和低功耗，使其在运动模糊、遮挡和光照变化等挑战条件下进行实时机器人感知的理想选择。然而，尽管具有这些优点，合成事件驱动视觉在主流机器人模拟器中仍 largely unexplored。本工作 presents 一个开源、用户友好的从RGB相机馈送生成事件流的v2e机器人操作系统(ROS)包，以用于Gazebo模拟。该包用于研究事件驱动机器人策略(ERP)在实时导航和 manipulation 任务中的应用。评估了两个代表性场景：（1）移动机器人物体跟随和（2）机器人 manipulator 物体检测与抓取。基于Transformer的 ERP 通过行为克隆训练，并在不同操作条件下与基于RGB的同类进行比较。实验结果表明，事件引导策略一致地提供了竞争力的优势。结果强调了事件驱动感知在提高实时机器人导航和 manipulation 方面的潜力，为更广泛地将事件相机整合到机器人策略学习中奠定了基础。GitHub 数据集和代码仓库: this https URL。 

---
# Robotic Manipulation via Imitation Learning: Taxonomy, Evolution, Benchmark, and Challenges 

**Title (ZH)**: 基于模仿学习的机器人操作：分类、演进、基准与挑战 

**Authors**: Zezeng Li, Alexandre Chapin, Enda Xiang, Rui Yang, Bruno Machado, Na Lei, Emmanuel Dellandrea, Di Huang, Liming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.17449)  

**Abstract**: Robotic Manipulation (RM) is central to the advancement of autonomous robots, enabling them to interact with and manipulate objects in real-world environments. This survey focuses on RM methodologies that leverage imitation learning, a powerful technique that allows robots to learn complex manipulation skills by mimicking human demonstrations. We identify and analyze the most influential studies in this domain, selected based on community impact and intrinsic quality. For each paper, we provide a structured summary, covering the research purpose, technical implementation, hierarchical classification, input formats, key priors, strengths and limitations, and citation metrics. Additionally, we trace the chronological development of imitation learning techniques within RM policy (RMP), offering a timeline of key technological advancements. Where available, we report benchmark results and perform quantitative evaluations to compare existing methods. By synthesizing these insights, this review provides a comprehensive resource for researchers and practitioners, highlighting both the state of the art and the challenges that lie ahead in the field of robotic manipulation through imitation learning. 

**Abstract (ZH)**: 机器人操作中的模仿学习方法综述：从实世界环境中物体的交互与操作到基于模仿学习的机器人操作方法学的研究 

---
# A Rapid Iterative Trajectory Planning Method for Automated Parking through Differential Flatness 

**Title (ZH)**: 基于微分平坦性的快速迭代轨迹规划方法用于自动停车 

**Authors**: Zhouheng Li, Lei Xie, Cheng Hu, Hongye Su  

**Link**: [PDF](https://arxiv.org/pdf/2508.17038)  

**Abstract**: As autonomous driving continues to advance, automated parking is becoming increasingly essential. However, significant challenges arise when implementing path velocity decomposition (PVD) trajectory planning for automated parking. The primary challenge is ensuring rapid and precise collision-free trajectory planning, which is often in conflict. The secondary challenge involves maintaining sufficient control feasibility of the planned trajectory, particularly at gear shifting points (GSP). This paper proposes a PVD-based rapid iterative trajectory planning (RITP) method to solve the above challenges. The proposed method effectively balances the necessity for time efficiency and precise collision avoidance through a novel collision avoidance framework. Moreover, it enhances the overall control feasibility of the planned trajectory by incorporating the vehicle kinematics model and including terminal smoothing constraints (TSC) at GSP during path planning. Specifically, the proposed method leverages differential flatness to ensure the planned path adheres to the vehicle kinematic model. Additionally, it utilizes TSC to maintain curvature continuity at GSP, thereby enhancing the control feasibility of the overall trajectory. The simulation results demonstrate superior time efficiency and tracking errors compared to model-integrated and other iteration-based trajectory planning methods. In the real-world experiment, the proposed method was implemented and validated on a ROS-based vehicle, demonstrating the applicability of the RITP method for real vehicles. 

**Abstract (ZH)**: 随着自动驾驶技术的不断进步，自动泊车变得 increasingly 重要。然而，在实施路径速度分解（PVD）轨迹规划时，自动泊车面临着显著的挑战。主要挑战是确保快速而精确的无碰撞轨迹规划，这往往存在冲突。次要挑战在于在换挡点（GSP）保持计划轨迹的充分控制可行性。本文提出了一种基于PVD的快速迭代轨迹规划（RITP）方法来解决上述挑战。所提出的方法通过一种新的碰撞规避框架有效地平衡了时间效率和精确碰撞规避的必要性。此外，通过结合车辆运动学模型并在路径规划过程中在换挡点（GSP）纳入终端平滑约束（TSC），该方法还增强了计划轨迹的整体控制可行性。具体而言，所提出的方法利用微分平坦性确保计划路径遵循车辆运动学模型。此外，利用TSC保持换挡点处的曲率连续性，从而提高整体轨迹的控制可行性。仿真结果表明，与集成模型和其他基于迭代的轨迹规划方法相比，该方法具有更好的时间效率和跟踪误差。在实际试验中，所提出的方法已在基于ROS的车辆上实现并验证，展示了RITP方法在实际车辆上的适用性。 

---
# Relative Navigation and Dynamic Target Tracking for Autonomous Underwater Proximity Operations 

**Title (ZH)**: 自主水下近距离操作中的相对导航与动态目标跟踪 

**Authors**: David Baxter, Aldo Terán Espinoza, Antonio Terán Espinoza, Amy Loutfi, John Folkesson, Peter Sigray, Stephanie Lowry, Jakob Kuttenkeuler  

**Link**: [PDF](https://arxiv.org/pdf/2508.16901)  

**Abstract**: Estimating a target's 6-DoF motion in underwater proximity operations is difficult because the chaser lacks target-side proprioception and the available relative observations are sparse, noisy, and often partial (e.g., Ultra-Short Baseline (USBL) positions). Without a motion prior, factor-graph maximum a posteriori estimation is underconstrained: consecutive target states are weakly linked and orientation can drift. We propose a generalized constant-twist motion prior defined on the tangent space of Lie groups that enforces temporally consistent trajectories across all degrees of freedom; in SE(3) it couples translation and rotation in the body frame. We present a ternary factor and derive its closed-form Jacobians based on standard Lie group operations, enabling drop-in use for trajectories on arbitrary Lie groups. We evaluate two deployment modes: (A) an SE(3)-only representation that regularizes orientation even when only position is measured, and (B) a mode with boundary factors that switches the target representation between SE(3) and 3D position while applying the same generalized constant-twist prior across representation changes. Validation on a real-world dynamic docking scenario dataset shows consistent ego-target trajectory estimation through USBL-only and optical relative measurement segments with an improved relative tracking accuracy compared to the noisy measurements to the target. Because the construction relies on standard Lie group primitives, it is portable across state manifolds and sensing modalities. 

**Abstract (ZH)**: 基于Lie群切空间的通用固定旋扭转运动先验在水下近程操作中目标6-DOF运动估计 

---
# Autonomous UAV Flight Navigation in Confined Spaces: A Reinforcement Learning Approach 

**Title (ZH)**: 自主无人机在受限空间内的飞行导航：一种强化学习方法 

**Authors**: Marco S. Tayar, Lucas K. de Oliveira, Juliano D. Negri, Thiago H. Segreto, Ricardo V. Godoy, Marcelo Becker  

**Link**: [PDF](https://arxiv.org/pdf/2508.16807)  

**Abstract**: Inspecting confined industrial infrastructure, such as ventilation shafts, is a hazardous and inefficient task for humans. Unmanned Aerial Vehicles (UAVs) offer a promising alternative, but GPS-denied environments require robust control policies to prevent collisions. Deep Reinforcement Learning (DRL) has emerged as a powerful framework for developing such policies, and this paper provides a comparative study of two leading DRL algorithms for this task: the on-policy Proximal Policy Optimization (PPO) and the off-policy Soft Actor-Critic (SAC). The training was conducted with procedurally generated duct environments in Genesis simulation environment. A reward function was designed to guide a drone through a series of waypoints while applying a significant penalty for collisions. PPO learned a stable policy that completed all evaluation episodes without collision, producing smooth trajectories. By contrast, SAC consistently converged to a suboptimal behavior that traversed only the initial segments before failure. These results suggest that, in hazard-dense navigation, the training stability of on-policy methods can outweigh the nominal sample efficiency of off-policy algorithms. More broadly, the study provides evidence that procedurally generated, high-fidelity simulations are effective testbeds for developing and benchmarking robust navigation policies. 

**Abstract (ZH)**: 基于高性能模拟环境的深度强化学习在受限工业基础设施检测中的比较研究：以通风管道为例 

---
# A Dataset and Benchmark for Robotic Cloth Unfolding Grasp Selection: The ICRA 2024 Cloth Competition 

**Title (ZH)**: 用于机器人布料展开抓取选择的数据集与基准：ICRA 2024 布料竞赛 

**Authors**: Victor-Louis De Gusseme, Thomas Lips, Remko Proesmans, Julius Hietala, Giwan Lee, Jiyoung Choi, Jeongil Choi, Geon Kim, Phayuth Yonrith, Domen Tabernik, Andrej Gams, Peter Nimac, Matej Urbas, Jon Muhovič, Danijel Skočaj, Matija Mavsar, Hyojeong Yu, Minseo Kwon, Young J. Kim, Yang Cong, Ronghan Chen, Yu Ren, Supeng Diao, Jiawei Weng, Jiayue Liu, Haoran Sun, Linhan Yang, Zeqing Zhang, Ning Guo, Lei Yang, Fang Wan, Chaoyang Song, Jia Pan, Yixiang Jin, Yong A, Jun Shi, Dingzhe Li, Yong Yang, Kakeru Yamasaki, Takumi Kajiwara, Yuki Nakadera, Krati Saxena, Tomohiro Shibata, Chongkun Xia, Kai Mo, Yanzhao Yu, Qihao Lin, Binqiang Ma, Uihun Sagong, JungHyun Choi, JeongHyun Park, Dongwoo Lee, Yeongmin Kim, Myun Joong Hwang, Yusuke Kuribayashi, Naoki Hiratsuka, Daisuke Tanaka, Solvi Arnold, Kimitoshi Yamazaki, Carlos Mateo-Agullo, Andreas Verleysen, Francis Wyffels  

**Link**: [PDF](https://arxiv.org/pdf/2508.16749)  

**Abstract**: Robotic cloth manipulation suffers from a lack of standardized benchmarks and shared datasets for evaluating and comparing different approaches. To address this, we created a benchmark and organized the ICRA 2024 Cloth Competition, a unique head-to-head evaluation focused on grasp pose selection for in-air robotic cloth unfolding. Eleven diverse teams participated in the competition, utilizing our publicly released dataset of real-world robotic cloth unfolding attempts and a variety of methods to design their unfolding approaches. Afterwards, we also expanded our dataset with 176 competition evaluation trials, resulting in a dataset of 679 unfolding demonstrations across 34 garments. Analysis of the competition results revealed insights about the trade-off between grasp success and coverage, the surprisingly strong achievements of hand-engineered methods and a significant discrepancy between competition performance and prior work, underscoring the importance of independent, out-of-the-lab evaluation in robotic cloth manipulation. The associated dataset is a valuable resource for developing and evaluating grasp selection methods, particularly for learning-based approaches. We hope that our benchmark, dataset and competition results can serve as a foundation for future benchmarks and drive further progress in data-driven robotic cloth manipulation. The dataset and benchmarking code are available at this https URL. 

**Abstract (ZH)**: 机器人布料操作缺乏标准化基准和共享数据集来评估和比较不同方法。为解决这一问题，我们创建了一个基准并组织了ICRA 2024布料竞赛，这是一个专注于空中机器人布料展开中抓取姿态选择的unique头对头评估。来自世界各地的十一支队伍参与了竞赛，利用我们公开发布的实际机器人布料展开数据集以及各种方法设计其展开策略。随后，我们还扩充了数据集，增加了176次竞赛评估试验，共包含34件服装的679次展开演示。竞赛结果分析揭示了抓取成功率与覆盖范围之间的权衡，手工程设计方法的惊人表现，以及竞赛表现与之前工作之间的显著差异，强调了在机器人布料操作中独立于实验室外评估的重要性。该相关数据集是开发和评估抓取选择方法（特别是基于学习的方法）的重要资源。我们希望我们的基准、数据集和竞赛结果能为未来基准的建立提供基础，并推动数据驱动的机器人布料操作的进一步进步。数据集和基准代码可在以下链接获取：this https URL。 

---
# BirdRecorder's AI on Sky: Safeguarding birds of prey by detection and classification of tiny objects around wind turbines 

**Title (ZH)**: BirdRecorder的AI翱翔于天空：通过风力发电机周围小型物体的检测与分类保护猛禽 

**Authors**: Nico Klar, Nizam Gifary, Felix P. G. Ziegler, Frank Sehnke, Anton Kaifel, Eric Price, Aamir Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2508.18136)  

**Abstract**: The urgent need for renewable energy expansion, particularly wind power, is hindered by conflicts with wildlife conservation. To address this, we developed BirdRecorder, an advanced AI-based anti-collision system to protect endangered birds, especially the red kite (Milvus milvus). Integrating robotics, telemetry, and high-performance AI algorithms, BirdRecorder aims to detect, track, and classify avian species within a range of 800 m to minimize bird-turbine collisions.
BirdRecorder integrates advanced AI methods with optimized hardware and software architectures to enable real-time image processing. Leveraging Single Shot Detector (SSD) for detection, combined with specialized hardware acceleration and tracking algorithms, our system achieves high detection precision while maintaining the speed necessary for real-time decision-making. By combining these components, BirdRecorder outperforms existing approaches in both accuracy and efficiency.
In this paper, we summarize results on field tests and performance of the BirdRecorder system. By bridging the gap between renewable energy expansion and wildlife conservation, BirdRecorder contributes to a more sustainable coexistence of technology and nature. 

**Abstract (ZH)**: 迫切需要扩展可再生能源，尤其是风能，但受到了与野生动物保护之间的冲突的阻碍。为了解决这一问题，我们开发了BirdRecorder，一种基于先进AI的防撞系统，旨在保护濒危鸟类，尤其是红尾鹰（Milvus milvus）。BirdRecorder 结合了机器人技术、遥感和高性能AI算法，旨在在800米范围内检测、跟踪和分类鸟类，以最小化鸟与风力涡轮机的碰撞。

BirdRecorder 将先进的AI方法与优化的硬件和软件架构相结合，实现实时图像处理。利用单-shot检测器（SSD）进行检测，并结合专门的硬件加速和跟踪算法，我们的系统在保持足够速度以进行即时决策的同时，实现了高检测精度。通过结合这些组件，BirdRecorder 在准确性和效率上均优于现有方法。

在本文中，我们总结了BirdRecorder系统在实地测试中的结果和性能。通过弥合可再生能源扩展与野生动物保护之间的差距，BirdRecorder 促进了技术与自然之间更可持续的共存。 

---
# Dimension-Decomposed Learning for Quadrotor Geometric Attitude Control with Almost Global Exponential Convergence on SO(3) 

**Title (ZH)**: 四旋翼几何姿态控制的维度分解学习及其在SO(3)上的几乎全局指数收敛 

**Authors**: Tianhua Gao, Masashi Izumita, Kohji Tomita, Akiya Kamimura  

**Link**: [PDF](https://arxiv.org/pdf/2508.14422)  

**Abstract**: This paper introduces a lightweight and interpretable online learning approach called Dimension-Decomposed Learning (DiD-L) for disturbance identification in quadrotor geometric attitude control. As a module instance of DiD-L, we propose the Sliced Adaptive-Neuro Mapping (SANM). Specifically, to address underlying underfitting problems, the high-dimensional mapping for online identification is axially ``sliced" into multiple low-dimensional submappings (slices). In this way, the complex high-dimensional problem is decomposed into a set of simple low-dimensional subtasks addressed by shallow neural networks and adaptive laws. These neural networks and adaptive laws are updated online via Lyapunov-based adaptation without the persistent excitation (PE) condition. To enhance the interpretability of the proposed approach, we prove that the state solution of the rotational error dynamics exponentially converges into an arbitrarily small ball within an almost global attraction domain, despite time-varying disturbances and inertia uncertainties. This result is novel as it demonstrates exponential convergence without requiring pre-training for unseen disturbances and specific knowledge of the model. To our knowledge in the quadrotor control field, DiD-L is the first online learning approach that is lightweight enough to run in real-time at 400 Hz on microcontroller units (MCUs) such as STM32, and has been validated through real-world experiments. 

**Abstract (ZH)**: 一种轻量级可解释的在线学习方法——维度分解学习（DiD-L）及其在四旋翼几何姿态控制中的扰动识别应用 

---
# AQ-PCDSys: An Adaptive Quantized Planetary Crater Detection System for Autonomous Space Exploration 

**Title (ZH)**: AQ-PCDSys: 一种自适应量化 planetary 衡星环形坑检测系统用于自主太空探索 

**Authors**: Aditri Paul, Archan Paul  

**Link**: [PDF](https://arxiv.org/pdf/2508.18025)  

**Abstract**: Autonomous planetary exploration missions are critically dependent on real-time, accurate environmental perception for navigation and hazard avoidance. However, deploying deep learning models on the resource-constrained computational hardware of planetary exploration platforms remains a significant challenge. This paper introduces the Adaptive Quantized Planetary Crater Detection System (AQ-PCDSys), a novel framework specifically engineered for real-time, onboard deployment in the computationally constrained environments of space exploration missions. AQ-PCDSys synergistically integrates a Quantized Neural Network (QNN) architecture, trained using Quantization-Aware Training (QAT), with an Adaptive Multi-Sensor Fusion (AMF) module. The QNN architecture significantly optimizes model size and inference latency suitable for real-time onboard deployment in space exploration missions, while preserving high accuracy. The AMF module intelligently fuses data from Optical Imagery (OI) and Digital Elevation Models (DEMs) at the feature level, utilizing an Adaptive Weighting Mechanism (AWM) to dynamically prioritize the most relevant and reliable sensor modality based on planetary ambient conditions. This approach enhances detection robustness across diverse planetary landscapes. Paired with Multi-Scale Detection Heads specifically designed for robust and efficient detection of craters across a wide range of sizes, AQ-PCDSys provides a computationally efficient, reliable and accurate solution for planetary crater detection, a critical capability for enabling the next generation of autonomous planetary landing, navigation, and scientific exploration. 

**Abstract (ZH)**: 自适应量化行星撞击坑检测系统（AQ-PCDSys）：一种适用于太空探索任务计算约束环境的实时机载部署框架 

---
