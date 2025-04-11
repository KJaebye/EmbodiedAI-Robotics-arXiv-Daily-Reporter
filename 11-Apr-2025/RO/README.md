# Echo: An Open-Source, Low-Cost Teleoperation System with Force Feedback for Dataset Collection in Robot Learning 

**Title (ZH)**: Echo：一种用于机器人学习数据集收集的低成本力反馈远程操作开源系统 

**Authors**: Artem Bazhenov, Sergei Satsevich, Sergei Egorov, Farit Khabibullin, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2504.07939)  

**Abstract**: In this article, we propose Echo, a novel joint-matching teleoperation system designed to enhance the collection of datasets for manual and bimanual tasks. Our system is specifically tailored for controlling the UR manipulator and features a custom controller with force feedback and adjustable sensitivity modes, enabling precise and intuitive operation. Additionally, Echo integrates a user-friendly dataset recording interface, simplifying the process of collecting high-quality training data for imitation learning. The system is designed to be reliable, cost-effective, and easily reproducible, making it an accessible tool for researchers, laboratories, and startups passionate about advancing robotics through imitation learning. Although the current implementation focuses on the UR manipulator, Echo architecture is reconfigurable and can be adapted to other manipulators and humanoid systems. We demonstrate the effectiveness of Echo through a series of experiments, showcasing its ability to perform complex bimanual tasks and its potential to accelerate research in the field. We provide assembly instructions, a hardware description, and code at this https URL. 

**Abstract (ZH)**: 本文提出Echo，一种新的联合匹配远程操作系统，旨在增强手动和双臂任务数据集的采集。该系统特别针对控制UR操作器设计，并配备了一个具备力反馈和可调灵敏度模式的定制控制器，使得操作既精确又直观。此外，Echo集成了用户友好的数据集记录界面，简化了高质量训练数据的采集过程，用于模仿学习。系统设计注重可靠、经济，并易于复制，使其成为研究人员、实验室和致力于通过模仿学习推进机器人技术的初创公司的一个可访问工具。尽管当前实现主要针对UR操作器，但Echo架构具有可重构性，并可适应其他操作器和类人系统。我们通过一系列实验展示了Echo的有效性，展示了其完成复杂双臂任务的能力及其在推动该领域研究方面的潜力。在此网址<https://>提供了组装说明、硬件描述和代码。 

---
# Experimental Analysis of Quadcopter Drone Hover Constraints for Localization Improvements 

**Title (ZH)**: 四旋翼无人机悬停约束的实验分析及其对定位改进的影响 

**Authors**: Uthman Olawoye, David Akhihiero, Jason N. Gross  

**Link**: [PDF](https://arxiv.org/pdf/2504.07843)  

**Abstract**: In this work, we evaluate the use of aerial drone hover constraints in a multisensor fusion of ground robot and drone data to improve the localization performance of a drone. In particular, we build upon our prior work on cooperative localization between an aerial drone and ground robot that fuses data from LiDAR, inertial navigation, peer-to-peer ranging, altimeter, and stereo-vision and evaluate the incorporation knowledge from the autopilot regarding when the drone is hovering. This control command data is leveraged to add constraints on the velocity state. Hover constraints can be considered important dynamic model information, such as the exploitation of zero-velocity updates in pedestrian navigation. We analyze the benefits of these constraints using an incremental factor graph optimization. Experimental data collected in a motion capture faculty is used to provide performance insights and assess the benefits of hover constraints. 

**Abstract (ZH)**: 本研究评估了在地面机器人和无人机数据多传感器融合中使用空中无人机悬停约束以提高无人机定位性能的应用。在 particular，我们基于以前关于空中无人机与地面机器人协同定位的工作，该工作融合了来自 LiDAR、惯性导航、点对点测距、气压计和立体视觉的数据，并评估了自动驾驶仪有关无人机悬停时的知识整合。通过利用该控制命令数据，我们增加了对速度状态的约束。悬停约束可以视为重要的动态模型信息，例如在行人导航中利用零速度更新。我们使用增量因子图优化分析了这些约束的优势。实验数据采集自运动捕捉实验室，用于提供性能洞察并评估悬停约束的优势。 

---
# Cable Optimization and Drag Estimation for Tether-Powered Multirotor UAVs 

**Title (ZH)**: 脐带供电多旋翼无人机的电缆优化与拽引力估计 

**Authors**: Max Beffert, Andreas Zell  

**Link**: [PDF](https://arxiv.org/pdf/2504.07802)  

**Abstract**: The flight time of multirotor unmanned aerial vehicles (UAVs) is typically constrained by their high power consumption. Tethered power systems present a viable solution to extend flight times while maintaining the advantages of multirotor UAVs, such as hover capability and agility. This paper addresses the critical aspect of cable selection for tether-powered multirotor UAVs, considering both hover and forward flight. Existing research often overlooks the trade-offs between cable mass, power losses, and system constraints. We propose a novel methodology to optimize cable selection, accounting for thrust requirements and power efficiency across various flight conditions. The approach combines physics-informed modeling with system identification to combine hover and forward flight dynamics, incorporating factors such as motor efficiency, tether resistance, and aerodynamic drag. This work provides an intuitive and practical framework for optimizing tethered UAV designs, ensuring efficient power transmission and flight performance. Thus allowing for better, safer, and more efficient tethered drones. 

**Abstract (ZH)**: 多旋翼无人机的缆绳选择优化：考虑悬停和前进飞行的动力学 

---
# TOCALib: Optimal control library with interpolation for bimanual manipulation and obstacles avoidance 

**Title (ZH)**: TOCALib: 用于双臂操作和避障的插值优化控制库 

**Authors**: Yulia Danik, Dmitry Makarov, Aleksandra Arkhipova, Sergei Davidenko, Aleksandr Panov  

**Link**: [PDF](https://arxiv.org/pdf/2504.07708)  

**Abstract**: The paper presents a new approach for constructing a library of optimal trajectories for two robotic manipulators, Two-Arm Optimal Control and Avoidance Library (TOCALib). The optimisation takes into account kinodynamic and other constraints within the FROST framework. The novelty of the method lies in the consideration of collisions using the DCOL method, which allows obtaining symbolic expressions for assessing the presence of collisions and using them in gradient-based optimization control methods. The proposed approach allowed the implementation of complex bimanual manipulations. In this paper we used Mobile Aloha as an example of TOCALib application. The approach can be extended to other bimanual robots, as well as to gait control of bipedal robots. It can also be used to construct training data for machine learning tasks for manipulation. 

**Abstract (ZH)**: 基于FROST框架的双臂最优轨迹库TOCALib的构建方法：考虑碰撞的优化控制与避免 

---
# Transformer-Based Robust Underwater Inertial Navigation in Prolonged Doppler Velocity Log Outages 

**Title (ZH)**: 基于变压器的鲁棒水下惯性导航在延长的多普勒速度日志中断期间 

**Authors**: Zeev Yampolsky, Nadav Cohen, Itzik Klein  

**Link**: [PDF](https://arxiv.org/pdf/2504.07697)  

**Abstract**: Autonomous underwater vehicles (AUV) have a wide variety of applications in the marine domain, including exploration, surveying, and mapping. Their navigation systems rely heavily on fusing data from inertial sensors and a Doppler velocity log (DVL), typically via nonlinear filtering. The DVL estimates the AUV's velocity vector by transmitting acoustic beams to the seabed and analyzing the Doppler shift from the reflected signals. However, due to environmental challenges, DVL beams can deflect or fail in real-world settings, causing signal outages. In such cases, the AUV relies solely on inertial data, leading to accumulated navigation errors and mission terminations. To cope with these outages, we adopted ST-BeamsNet, a deep learning approach that uses inertial readings and prior DVL data to estimate AUV velocity during isolated outages. In this work, we extend ST-BeamsNet to address prolonged DVL outages and evaluate its impact within an extended Kalman filter framework. Experiments demonstrate that the proposed framework improves velocity RMSE by up to 63% and reduces final position error by up to 95% compared to pure inertial navigation. This is in scenarios involving up to 50 seconds of complete DVL outage. 

**Abstract (ZH)**: 自主水下车辆（AUV）在海洋领域中有广泛的应用，包括探索、调查和制图。其导航系统通常通过非线性滤波融合惯性传感器和多普勒声速计（DVL）的数据。DVL通过向海底发射声波束并分析反射信号的多普勒移频来估计AUV的速度矢量。然而，由于环境挑战，在实际应用中，DVL声波束可能会发生偏移或失效，导致信号中断。在这种情况下，AUV仅依赖惯性数据，导致累积的导航误差并可能终止任务。为了应对这些中断，我们采用了ST-BeamsNet这种方法，它使用惯性读数和先前的DVL数据在孤立的中断期间估计AUV的速度。在本文中，我们将ST-BeamsNet扩展以应对持续时间更长的DVL中断，并在扩展卡尔曼滤波框架内评估其影响。实验表明，所提出的框架在纯惯性导航方案中将速度RMSE提高了最高达63%，并将最终位置误差降低了最高达95%，适用于包括50秒完全DVL中断在内的场景。 

---
# Sim-to-Real Transfer in Reinforcement Learning for Maneuver Control of a Variable-Pitch MAV 

**Title (ZH)**: Sim-to-Real Transfer in Reinforcement Learning for Maneuver Control of a Variable-Pitch MAV 

**Authors**: Zhikun Wang, Shiyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.07694)  

**Abstract**: Reinforcement learning (RL) algorithms can enable high-maneuverability in unmanned aerial vehicles (MAVs), but transferring them from simulation to real-world use is challenging. Variable-pitch propeller (VPP) MAVs offer greater agility, yet their complex dynamics complicate the sim-to-real transfer. This paper introduces a novel RL framework to overcome these challenges, enabling VPP MAVs to perform advanced aerial maneuvers in real-world settings. Our approach includes real-to-sim transfer techniques-such as system identification, domain randomization, and curriculum learning to create robust training simulations and a sim-to-real transfer strategy combining a cascade control system with a fast-response low-level controller for reliable deployment. Results demonstrate the effectiveness of this framework in achieving zero-shot deployment, enabling MAVs to perform complex maneuvers such as flips and wall-backtracking. 

**Abstract (ZH)**: 基于强化学习的可变桨距旋翼无人机从仿真到现实世界的高效操控方法 

---
# Localization Meets Uncertainty: Uncertainty-Aware Multi-Modal Localization 

**Title (ZH)**: 定位遇见不确定性：不确定性感知多模态定位 

**Authors**: Hye-Min Won, Jieun Lee, Jiyong Oh  

**Link**: [PDF](https://arxiv.org/pdf/2504.07677)  

**Abstract**: Reliable localization is critical for robot navigation in complex indoor environments. In this paper, we propose an uncertainty-aware localization method that enhances the reliability of localization outputs without modifying the prediction model itself. This study introduces a percentile-based rejection strategy that filters out unreliable 3-DoF pose predictions based on aleatoric and epistemic uncertainties the network estimates. We apply this approach to a multi-modal end-to-end localization that fuses RGB images and 2D LiDAR data, and we evaluate it across three real-world datasets collected using a commercialized serving robot. Experimental results show that applying stricter uncertainty thresholds consistently improves pose accuracy. Specifically, the mean position error is reduced by 41.0%, 56.7%, and 69.4%, and the mean orientation error by 55.6%, 65.7%, and 73.3%, when applying 90%, 80%, and 70% thresholds, respectively. Furthermore, the rejection strategy effectively removes extreme outliers, resulting in better alignment with ground truth trajectories. To the best of our knowledge, this is the first study to quantitatively demonstrate the benefits of percentile-based uncertainty rejection in multi-modal end-to-end localization tasks. Our approach provides a practical means to enhance the reliability and accuracy of localization systems in real-world deployments. 

**Abstract (ZH)**: 可靠的位置定位对于机器人在复杂室内环境中的导航至关重要。本文提出了一种不确定性感知的位置定位方法，该方法在不修改预测模型本身的情况下，增强位置定位输出的可靠性。本研究引入了一种基于百分位数的拒绝策略，该策略根据网络估计的 aleatoric 和 epistemic 不确定性，过滤掉不可靠的 3-DoF 姿态预测。我们将此方法应用于融合 RGB 图像和 2D LiDAR 数据的端到端多模态定位中，并在使用商用服务机器人收集的三个现实世界数据集中进行了评估。实验结果显示，应用更严格的不确定性阈值可以一致地提高姿态准确性。具体而言，当应用 90%、80% 和 70% 的阈值时，位置误差的均值分别减少了 41.0%、56.7% 和 69.4%，姿态误差的均值分别减少了 55.6%、65.7% 和 73.3%。此外，拒绝策略有效去除极端异常值，从而更好地与真实轨迹对齐。据我们所知，这是首次定量证明多模态端到端定位任务中基于百分位数不确定性拒绝策略益处的研究。我们的方法为在实际部署中提升定位系统的可靠性和准确性提供了一种实用手段。 

---
# UWB Anchor Based Localization of a Planetary Rover 

**Title (ZH)**: 基于UWB锚点的行星车定位 

**Authors**: Andreas Nüchter, Lennart Werner, Martin Hesse, Dorit Borrmann, Thomas Walter, Sergio Montenegro, Gernot Grömer  

**Link**: [PDF](https://arxiv.org/pdf/2504.07658)  

**Abstract**: Localization of an autonomous mobile robot during planetary exploration is challenging due to the unknown terrain, the difficult lighting conditions and the lack of any global reference such as satellite navigation systems. We present a novel approach for robot localization based on ultra-wideband (UWB) technology. The robot sets up its own reference coordinate system by distributing UWB anchor nodes in the environment via a rocket-propelled launcher system. This allows the creation of a localization space in which UWB measurements are employed to supplement traditional SLAM-based techniques. The system was developed for our involvement in the ESA-ESRIC challenge 2021 and the AMADEE-24, an analog Mars simulation in Armenia by the Austrian Space Forum (ÖWF). 

**Abstract (ZH)**: 行星探测中自主移动机器人的定位因其未知地形、恶劣光照条件以及缺乏如卫星导航系统等全球参考而具有挑战性。基于超宽带(UWB)技术的机器人定位方法研究：通过火箭推进发射系统布置UWB锚节点建立自主参考坐标系，以补充传统SLAM技术并构建局部定位空间。该系统用于参与ESA-ESRIC 2021挑战赛及奥利地空间论坛(OEF)在 Armenian 的模拟火星环境AMADEE-24项目。 

---
# Learning Long Short-Term Intention within Human Daily Behaviors 

**Title (ZH)**: 学习人体日常行为中的长期短期意图 

**Authors**: Zhe Sun, Rujie Wu, Xiaodong Yang, Hongzhao Xie, Haiyan Jiang, Junda Bi, Zhenliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07597)  

**Abstract**: In the domain of autonomous household robots, it is of utmost importance for robots to understand human behaviors and provide appropriate services. This requires the robots to possess the capability to analyze complex human behaviors and predict the true intentions of humans. Traditionally, humans are perceived as flawless, with their decisions acting as the standards that robots should strive to align with. However, this raises a pertinent question: What if humans make mistakes? In this research, we present a unique task, termed "long short-term intention prediction". This task requires robots can predict the long-term intention of humans, which aligns with human values, and the short term intention of humans, which reflects the immediate action intention. Meanwhile, the robots need to detect the potential non-consistency between the short-term and long-term intentions, and provide necessary warnings and suggestions. To facilitate this task, we propose a long short-term intention model to represent the complex intention states, and build a dataset to train this intention model. Then we propose a two-stage method to integrate the intention model for robots: i) predicting human intentions of both value-based long-term intentions and action-based short-term intentions; and 2) analyzing the consistency between the long-term and short-term intentions. Experimental results indicate that the proposed long short-term intention model can assist robots in comprehending human behavioral patterns over both long-term and short-term durations, which helps determine the consistency between long-term and short-term intentions of humans. 

**Abstract (ZH)**: 自主家庭机器人领域中，机器人理解和预测人类行为并提供恰当服务至关重要。这要求机器人具备分析复杂人类行为和预测人类真实意图的能力。传统上，人类被视为完美无缺的，其决策被视为机器人应力求一致的标准。然而，如果人类犯错误怎么办？本研究提出一项独特的任务，称为“长短期意图预测”。该任务要求机器人能够预测与人类价值观相符的长期意图和反映即时动作意图的短期意图。同时，机器人需要检测短期意图与长期意图之间的潜在不一致性，并提供必要的警示和建议。为了完成这一任务，我们提出了一种长短期意图模型来表示复杂的意图状态，并构建了一个数据集来训练该意图模型。然后，我们提出了一种两阶段方法来整合意图模型：首先，预测基于价值的长期意图和基于行动的短期意图；其次，分析长期意图与短期意图的一致性。实验结果表明，所提出的长短期意图模型能够帮助机器人理解人类行为模式，无论是长期的还是短期的，从而确定人类长期意图和短期意图之间的一致性。 

---
# Efficient Swept Volume-Based Trajectory Generation for Arbitrary-Shaped Ground Robot Navigation 

**Title (ZH)**: 基于扫掠体积的任意形状地面机器人轨迹生成高效算法 

**Authors**: Yisheng Li, Longji Yin, Yixi Cai, Jianheng Liu, Haotian Li, Fu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07554)  

**Abstract**: Navigating an arbitrary-shaped ground robot safely in cluttered environments remains a challenging problem. The existing trajectory planners that account for the robot's physical geometry severely suffer from the intractable runtime. To achieve both computational efficiency and Continuous Collision Avoidance (CCA) of arbitrary-shaped ground robot planning, we proposed a novel coarse-to-fine navigation framework that significantly accelerates planning. In the first stage, a sampling-based method selectively generates distinct topological paths that guarantee a minimum inflated margin. In the second stage, a geometry-aware front-end strategy is designed to discretize these topologies into full-state robot motion sequences while concurrently partitioning the paths into SE(2) sub-problems and simpler R2 sub-problems for back-end optimization. In the final stage, an SVSDF-based optimizer generates trajectories tailored to these sub-problems and seamlessly splices them into a continuous final motion plan. Extensive benchmark comparisons show that the proposed method is one to several orders of magnitude faster than the cutting-edge methods in runtime while maintaining a high planning success rate and ensuring CCA. 

**Abstract (ZH)**: 导航具有任意形状的地面机器人在复杂环境中的安全路径规划仍然是一个具有挑战性的问题。为同时实现计算效率和任意形状地面机器人的连续碰撞避免，我们提出了一种新的粗细结合导航框架，显著加速路径规划。该框架的首个阶段通过采样方法选择性生成保证最小膨胀边界的独特拓扑路径。第二个阶段设计了一个几何感知的前端策略，将这些拓扑路径离散化为完整的机器人状态运动序列，并同时将路径分割为SE(2)子问题和更简单的R2子问题用于后端优化。最终阶段使用基于SVSDF的优化器为这些子问题生成定制轨迹，并无缝拼接成连续的最终运动计划。广泛的基准比较结果表明，与最先进的方法相比，所提出的方法在运行时间上快一个到几个数量级，同时保持高规划成功率并确保连续碰撞避免。 

---
# Drive in Corridors: Enhancing the Safety of End-to-end Autonomous Driving via Corridor Learning and Planning 

**Title (ZH)**: 基于走廊学习与规划的端到端自动驾驶安全性提升 

**Authors**: Zhiwei Zhang, Ruichen Yang, Ke Wu, Zijun Xu, Jingchu Liu, Lisen Mu, Zhongxue Gan, Wenchao Ding  

**Link**: [PDF](https://arxiv.org/pdf/2504.07507)  

**Abstract**: Safety remains one of the most critical challenges in autonomous driving systems. In recent years, the end-to-end driving has shown great promise in advancing vehicle autonomy in a scalable manner. However, existing approaches often face safety risks due to the lack of explicit behavior constraints. To address this issue, we uncover a new paradigm by introducing the corridor as the intermediate representation. Widely adopted in robotics planning, the corridors represents spatio-temporal obstacle-free zones for the vehicle to traverse. To ensure accurate corridor prediction in diverse traffic scenarios, we develop a comprehensive learning pipeline including data annotation, architecture refinement and loss formulation. The predicted corridor is further integrated as the constraint in a trajectory optimization process. By extending the differentiability of the optimization, we enable the optimized trajectory to be seamlessly trained within the end-to-end learning framework, improving both safety and interpretability. Experimental results on the nuScenes dataset demonstrate state-of-the-art performance of our approach, showing a 66.7% reduction in collisions with agents and a 46.5% reduction with curbs, significantly enhancing the safety of end-to-end driving. Additionally, incorporating the corridor contributes to higher success rates in closed-loop evaluations. 

**Abstract (ZH)**: 自主驾驶系统中安全性仍然是最关键的挑战之一。近年来，端到端驾驶展示了在规模化提升车辆自主性方面的巨大潜力。然而，现有方法往往由于缺乏显式的行为约束而面临安全风险。为解决这一问题，我们通过引入走廊作为中间表示，提出了一种新的范式。走廊在机器人规划中广泛采用，代表了车辆可以穿越的空间-时间无障碍区域。为了确保在不同交通场景下准确预测走廊，我们开发了一个全面的学习管道，包括数据标注、架构优化和损失函数设计。预测的走廊进一步被作为约束整合到轨迹优化过程中。通过扩展优化的可微性，我们使优化的轨迹能够在端到端学习框架中无缝训练，从而同时提高安全性和可解释性。实验结果表明，我们的方法在nuScenes数据集上的性能达到最新水平，碰撞代理减少66.7%，撞缘减少46.5%，显著提升了端到端驾驶的安全性。此外，将走廊纳入还提高了闭环评估的成功率。 

---
# Adaptive Vision-Guided Robotic Arm Control for Precision Pruning in Dynamic Orchard Environments 

**Title (ZH)**: 动态果园环境中自适应视觉引导机器人手臂控制用于精准修剪 

**Authors**: Dawood Ahmed, Basit Muhammad Imran, Martin Churuvija, Manoj Karkee  

**Link**: [PDF](https://arxiv.org/pdf/2504.07309)  

**Abstract**: This study presents a vision-guided robotic control system for automated fruit tree pruning applications. Traditional agricultural practices rely on labor-intensive tasks and processes that lack scalability and efficiency, creating a pressing need for automation research to address growing demands for higher crop yields, scalable operations, and reduced manual labor. To this end, this paper proposes a novel algorithm for robust and automated fruit pruning in dense orchards. The proposed algorithm utilizes CoTracker, that is designed to track 2D feature points in video sequences with significant robustness and accuracy, while leveraging joint attention mechanisms to account for inter-point dependencies, enabling robust and precise tracking under challenging and sophisticated conditions. To validate the efficacy of CoTracker, a Universal Robots manipulator UR5e is employed in a Gazebo simulation environment mounted on ClearPath Robotics Warthog robot featuring an Intel RealSense D435 camera. The system achieved a 93% success rate in pruning trials and with an average end trajectory error of 0.23 mm. The vision controller demonstrated robust performance in handling occlusions and maintaining stable trajectories as the arm move towards the target point. The results validate the effectiveness of integrating vision-based tracking with kinematic control for precision agricultural tasks. Future work will focus on real-world implementation and the integration of 3D reconstruction techniques for enhanced adaptability in dynamic environments. 

**Abstract (ZH)**: 基于视觉引导的机器人控制系统在自动化果树修剪应用中的研究 

---
# Data-Enabled Neighboring Extremal: Case Study on Model-Free Trajectory Tracking for Robotic Arm 

**Title (ZH)**: 数据驱动邻域极值方法：基于模型自由轨迹跟踪的机器人手臂案例研究 

**Authors**: Amin Vahidi-Moghaddam, Keyi Zhu, Kaixiang Zhang, Ziyou Song, Zhaojian Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.07292)  

**Abstract**: Data-enabled predictive control (DeePC) has recently emerged as a powerful data-driven approach for efficient system controls with constraints handling capabilities. It performs optimal controls by directly harnessing input-output (I/O) data, bypassing the process of explicit model identification that can be costly and time-consuming. However, its high computational complexity, driven by a large-scale optimization problem (typically in a higher dimension than its model-based counterpart--Model Predictive Control), hinders real-time applications. To overcome this limitation, we propose the data-enabled neighboring extremal (DeeNE) framework, which significantly reduces computational cost while preserving control performance. DeeNE leverages first-order optimality perturbation analysis to efficiently update a precomputed nominal DeePC solution in response to changes in initial conditions and reference trajectories. We validate its effectiveness on a 7-DoF KINOVA Gen3 robotic arm, demonstrating substantial computational savings and robust, data-driven control performance. 

**Abstract (ZH)**: 基于数据的邻域极值数据驱动预测控制（DeeNE）框架：高性能与低计算成本的平衡 

---
# Bridging Deep Reinforcement Learning and Motion Planning for Model-Free Navigation in Cluttered Environments 

**Title (ZH)**: 深度强化学习与运动规划在受阻环境下的模型自由导航桥梁构建 

**Authors**: Licheng Luo, Mingyu Cai  

**Link**: [PDF](https://arxiv.org/pdf/2504.07283)  

**Abstract**: Deep Reinforcement Learning (DRL) has emerged as a powerful model-free paradigm for learning optimal policies. However, in real-world navigation tasks, DRL methods often suffer from insufficient exploration, particularly in cluttered environments with sparse rewards or complex dynamics under system disturbances. To address this challenge, we bridge general graph-based motion planning with DRL, enabling agents to explore cluttered spaces more effectively and achieve desired navigation performance. Specifically, we design a dense reward function grounded in a graph structure that spans the entire state space. This graph provides rich guidance, steering the agent toward optimal strategies. We validate our approach in challenging environments, demonstrating substantial improvements in exploration efficiency and task success rates. The project website is available at: this https URL 

**Abstract (ZH)**: 深度 reinforcement learning (DRL) 已经成为一种强大的无模型范式，用于学习最优策略。然而，在现实世界的导航任务中，DRL 方法往往由于探索不足而在复杂环境或系统干扰下的稀疏奖励或复杂动力学中表现出不佳。为了应对这一挑战，我们将通用图基运动规划与 DRL 结合起来，使代理能够在更复杂的环境中更有效地探索，并实现期望的导航性能。具体而言，我们设计了一个基于图结构的密集奖励函数，该图结构覆盖整个状态空间。这个图提供了丰富的指导，引导代理采取最优策略。我们在具有挑战性的环境中验证了这种方法，展示了在探索效率和任务成功率方面显著的改善。项目网站可访问：this https URL。 

---
# Expectations, Explanations, and Embodiment: Attempts at Robot Failure Recovery 

**Title (ZH)**: 期望、解释与 embodient: 机器人故障恢复的尝试 

**Authors**: Elmira Yadollahi, Fethiye Irmak Dogan, Yujing Zhang, Beatriz Nogueira, Tiago Guerreiro, Shelly Levy Tzedek, Iolanda Leite  

**Link**: [PDF](https://arxiv.org/pdf/2504.07266)  

**Abstract**: Expectations critically shape how people form judgments about robots, influencing whether they view failures as minor technical glitches or deal-breaking flaws. This work explores how high and low expectations, induced through brief video priming, affect user perceptions of robot failures and the utility of explanations in HRI. We conducted two online studies ($N=600$ total participants); each replicated two robots with different embodiments, Furhat and Pepper. In our first study, grounded in expectation theory, participants were divided into two groups, one primed with positive and the other with negative expectations regarding the robot's performance, establishing distinct expectation frameworks. This validation study aimed to verify whether the videos could reliably establish low and high-expectation profiles. In the second study, participants were primed using the validated videos and then viewed a new scenario in which the robot failed at a task. Half viewed a version where the robot explained its failure, while the other half received no explanation. We found that explanations significantly improved user perceptions of Furhat, especially when participants were primed to have lower expectations. Explanations boosted satisfaction and enhanced the robot's perceived expressiveness, indicating that effectively communicating the cause of errors can help repair user trust. By contrast, Pepper's explanations produced minimal impact on user attitudes, suggesting that a robot's embodiment and style of interaction could determine whether explanations can successfully offset negative impressions. Together, these findings underscore the need to consider users' expectations when tailoring explanation strategies in HRI. When expectations are initially low, a cogent explanation can make the difference between dismissing a failure and appreciating the robot's transparency and effort to communicate. 

**Abstract (ZH)**: 期望对机器人判断形成过程至关重要，影响人们对机器人失败的解读，是视为微不足道的技术问题还是致命缺陷。本研究探讨了通过短暂视频引导建立的高期待和低期待如何影响用户对机器人失败的感知及其在人机交互中的解释效用。我们进行了两项在线研究（总共600名参与者），每项研究使用了不同的机器人实体，Furhat和Pepper。在第一项研究中，基于期望理论，参与者被分为两组，一组被引导形成对机器人性能的正面期待，另一组形成负面期待，从而建立不同的期望框架。这项验证性研究旨在验证引导视频能否可靠地建立低期待和高期待的用户画像。在第二项研究中，使用了经过验证的视频引导参与者，并观察机器人在完成任务时出现故障的新场景。一半参与者看到了机器人解释故障的版本，另一半没有收到解释。研究发现，当参与者被引导形成较低的期待时，解释显著改善了他们对Furhat的感知，尤其是在满足度和增强机器人表达能力方面。相比之下，对Pepper的解释几乎没有影响，表明机器人的载体形式和交互方式可能决定了解释能否成功抵消负面印象。这些发现强调了在人机交互中考虑用户期待，制定合适的解释策略的重要性。当初期期待较低时，一个合理的解释可以将失败视为机器人透明性和沟通努力的表现而被理解。 

---
# Analysis of the Unscented Transform for Cooperative Localization with Ranging-Only Information 

**Title (ZH)**: 基于只含有测距信息的合作定位中无中心变换的分析 

**Authors**: Uthman Olawoye, Cagri Kilic, Jason N Gross  

**Link**: [PDF](https://arxiv.org/pdf/2504.07242)  

**Abstract**: Cooperative localization in multi-agent robotic systems is challenging, especially when agents rely on limited information, such as only peer-to-peer range measurements. Two key challenges arise: utilizing this limited information to improve position estimation; handling uncertainties from sensor noise, nonlinearity, and unknown correlations between agents measurements; and avoiding information reuse. This paper examines the use of the Unscented Transform (UT) for state estimation for a case in which range measurement between agents and covariance intersection (CI) is used to handle unknown correlations. Unlike Kalman Filter approaches, CI methods fuse complete state and covariance estimates. This makes formulating a CI approach with ranging-only measurements a challenge. To overcome this, UT is used to handle uncertainties and formulate a cooperative state update using range measurements and current cooperative state estimates. This introduces information reuse in the measurement update. Therefore, this work aims to evaluate the limitations and utility of this formulation when faced with various levels of state measurement uncertainty and errors. 

**Abstract (ZH)**: 多机器人系统中基于有限信息的合作定位具有挑战性：利用有限信息改进位置估计；处理来自传感器噪声、非线性以及未知测量间相关性的不确定性；避免信息重复使用。本文探讨使用无偏变换（UT）进行状态估计，并结合协方差交叠（CI）处理未知相关性。不同于卡尔曼滤波方法，CI方法融合完整的状态和协方差估计。这使得使用仅基于距离的测量来制定CI方法成为一个挑战。为克服这一问题，本文使用UT处理不确定性，并结合距离测量和当前合作状态估计来制定合作状态更新，从而在测量更新中引入信息重复使用。因此，本文旨在评估在不同状态测量不确定性水平和误差下的这种形式的局限性和实用性。 

---
# A Pointcloud Registration Framework for Relocalization in Subterranean Environments 

**Title (ZH)**: 地下环境重定位的点云配准框架 

**Authors**: David Akhihiero, Jason N. Gross  

**Link**: [PDF](https://arxiv.org/pdf/2504.07231)  

**Abstract**: Relocalization, the process of re-establishing a robot's position within an environment, is crucial for ensuring accurate navigation and task execution when external positioning information, such as GPS, is unavailable or has been lost. Subterranean environments present significant challenges for relocalization due to limited external positioning information, poor lighting that affects camera localization, irregular and often non-distinct surfaces, and dust, which can introduce noise and occlusion in sensor data. In this work, we propose a robust, computationally friendly framework for relocalization through point cloud registration utilizing a prior point cloud map. The framework employs Intrinsic Shape Signatures (ISS) to select feature points in both the target and prior point clouds. The Fast Point Feature Histogram (FPFH) algorithm is utilized to create descriptors for these feature points, and matching these descriptors yields correspondences between the point clouds. A 3D transformation is estimated using the matched points, which initializes a Normal Distribution Transform (NDT) registration. The transformation result from NDT is further refined using the Iterative Closest Point (ICP) registration algorithm. This framework enhances registration accuracy even in challenging conditions, such as dust interference and significant initial transformations between the target and source, making it suitable for autonomous robots operating in underground mines and tunnels. This framework was validated with experiments in simulated and real-world mine datasets, demonstrating its potential for improving relocalization. 

**Abstract (ZH)**: 基于点云配准的鲁棒重本地化方法：利用先验点云地图在地下环境中重获机器人位置 

---
# Fast Adaptation with Behavioral Foundation Models 

**Title (ZH)**: 快速适应行为基础模型 

**Authors**: Harshit Sikchi, Andrea Tirinzoni, Ahmed Touati, Yingchen Xu, Anssi Kanervisto, Scott Niekum, Amy Zhang, Alessandro Lazaric, Matteo Pirotta  

**Link**: [PDF](https://arxiv.org/pdf/2504.07896)  

**Abstract**: Unsupervised zero-shot reinforcement learning (RL) has emerged as a powerful paradigm for pretraining behavioral foundation models (BFMs), enabling agents to solve a wide range of downstream tasks specified via reward functions in a zero-shot fashion, i.e., without additional test-time learning or planning. This is achieved by learning self-supervised task embeddings alongside corresponding near-optimal behaviors and incorporating an inference procedure to directly retrieve the latent task embedding and associated policy for any given reward function. Despite promising results, zero-shot policies are often suboptimal due to errors induced by the unsupervised training process, the embedding, and the inference procedure. In this paper, we focus on devising fast adaptation strategies to improve the zero-shot performance of BFMs in a few steps of online interaction with the environment while avoiding any performance drop during the adaptation process. Notably, we demonstrate that existing BFMs learn a set of skills containing more performant policies than those identified by their inference procedure, making them well-suited for fast adaptation. Motivated by this observation, we propose both actor-critic and actor-only fast adaptation strategies that search in the low-dimensional task-embedding space of the pre-trained BFM to rapidly improve the performance of its zero-shot policies on any downstream task. Notably, our approach mitigates the initial "unlearning" phase commonly observed when fine-tuning pre-trained RL models. We evaluate our fast adaptation strategies on top of four state-of-the-art zero-shot RL methods in multiple navigation and locomotion domains. Our results show that they achieve 10-40% improvement over their zero-shot performance in a few tens of episodes, outperforming existing baselines. 

**Abstract (ZH)**: 无监督零样本强化学习（RL）已成为预训练行为基础模型（BFMs）的一种强大范式，使代理能够通过奖励函数指定的广泛下游任务以零样本方式求解，即无需额外的测试时学习或规划。这通过在学习自监督任务嵌入的同时学习相应的近最优行为，以及结合推断过程来直接检索任何给定奖励函数的潜在任务嵌入及其关联策略来实现。尽管取得了有希望的结果，但零样本策略通常由于无监督训练过程、嵌入和推断过程引起的误差而不够优化。在本文中，我们专注于设计快速适应策略，以在少量与环境的在线交互步骤中改进BFMs的零样本性能，同时避免适应过程中性能下降。值得注意的是，我们证明现有的BFMs学会了一组技能，其中包含由其推断过程识别出的性能更好的策略，使它们适合快速适应。受此观察的启发，我们提出了基于演员-批评家和仅演员的快速适应策略，这些策略在预训练BFM的任务嵌入低维空间中搜索，以迅速改进其零样本策略在任何下游任务上的性能。值得注意的是，我们的方法减轻了在微调预训练RL模型时通常观察到的最初的“反学习”阶段。我们在四个最先进的零样本RL方法上评估了我们的快速适应策略，并且在多个导航和运动领域中，结果显示它们在少量回合中实现了10-40%的性能提升，超越了现有基线。 

---
# Joint Travel Route Optimization Framework for Platooning 

**Title (ZH)**: 编队行驶的联合旅游路线优化框架 

**Authors**: Akif Adas, Stefano Arrigoni, Mattia Brambilla, Monica Barbara Nicoli, Edoardo Sabbioni  

**Link**: [PDF](https://arxiv.org/pdf/2504.07623)  

**Abstract**: Platooning represents an advanced driving technology designed to assist drivers in traffic convoys of varying lengths, enhancing road safety, reducing driver fatigue, and improving fuel efficiency. Sophisticated automated driving assistance systems have facilitated this innovation. Recent advancements in platooning emphasize cooperative mechanisms within both centralized and decentralized architectures enabled by vehicular communication technologies. This study introduces a cooperative route planning optimization framework aimed at promoting the adoption of platooning through a centralized platoon formation strategy at the system level. This approach is envisioned as a transitional phase from individual (ego) driving to fully collaborative driving. Additionally, this research formulates and incorporates travel cost metrics related to fuel consumption, driver fatigue, and travel time, considering regulatory constraints on consecutive driving durations. The performance of these cost metrics has been evaluated using Dijkstra's and A* shortest path algorithms within a network graph framework. The results indicate that the proposed architecture achieves an average cost improvement of 14 % compared to individual route planning for long road trips. 

**Abstract (ZH)**: Platooning代表了一种先进的驾驶技术，旨在协助驾驶者在不同长度的车队中行驶，提升道路安全，减少驾驶疲劳，并提高燃油效率。复杂的自动化驾驶辅助系统促进了这一创新。近期的platooning发展强调了由车辆通信技术支撑的集中式和分布式架构中的协同机制。本研究提出了一种协同路线规划优化框架，旨在通过系统层面的集中式编队策略促进platooning的普及。该方法被视为从个体（ ego）驾驶向完全协作驾驶过渡的阶段。此外，本研究制定了并融入了与燃料消耗、驾驶疲劳和旅行时间相关的旅行成本指标，并考虑了连续驾驶时间的监管限制。这些成本指标的应用通过网络图框架中的Dijkstra和A*最短路径算法进行评估。结果显示，所提出的架构在长距离旅行中平均成本改进了14%。 

---
# Personalized and Demand-Based Education Concept: Practical Tools for Control Engineers 

**Title (ZH)**: 个性化和需求导向的教育理念：控制工程师的实践工具 

**Authors**: Balint Varga, Lars Fischer, Levente Kovacs  

**Link**: [PDF](https://arxiv.org/pdf/2504.07466)  

**Abstract**: This paper presents a personalized lecture concept using educational blocks and its demonstrative application in a new university lecture. Higher education faces daily challenges: deep and specialized knowledge is available from everywhere and accessible to almost everyone. University lecturers of specialized master courses confront the problem that their lectures are either too boring or too complex for the attending students. Additionally, curricula are changing more rapidly than they have in the past 10-30 years. The German education system comprises different educational forms, with universities providing less practical content. Consequently, many university students do not obtain the practical skills they should ideally gain through university lectures. Therefore, in this work, a new lecture concept is proposed based on the extension of the just-in-time teaching paradigm: Personalized and Demand-Based Education. This concept includes: 1) an initial assessment of students' backgrounds, 2) selecting the appropriate educational blocks, and 3) collecting ongoing feedback during the semester. The feedback was gathered via Pingo, ensuring anonymity for the students. Our concept was exemplarily tested in the new lecture "Practical Tools for Control Engineers" at the Karlsruhe Institute of Technology. The initial results indicate that our proposed concept could be beneficial in addressing the current challenges in higher education. 

**Abstract (ZH)**: 基于教育模块的个性化讲授概念及其在新大学课程中的示范应用：面向控制工程师的实践工具讲授概念 

---
# Multi-Object Tracking for Collision Avoidance Using Multiple Cameras in Open RAN Networks 

**Title (ZH)**: 使用开放无线接入网络中多摄像头进行多目标跟踪以避免碰撞 

**Authors**: Jordi Serra, Anton Aguilar, Ebrahim Abu-Helalah, Raúl Parada, Paolo Dini  

**Link**: [PDF](https://arxiv.org/pdf/2504.07163)  

**Abstract**: This paper deals with the multi-object detection and tracking problem, within the scope of open Radio Access Network (RAN), for collision avoidance in vehicular scenarios. To this end, a set of distributed intelligent agents collocated with cameras are considered. The fusion of detected objects is done at an edge service, considering Open RAN connectivity. Then, the edge service predicts the objects trajectories for collision avoidance. Compared to the related work a more realistic Open RAN network is implemented and multiple cameras are used. 

**Abstract (ZH)**: 基于开放无线接入网络的车载场景中多目标检测与跟踪及其碰撞避免方法 

---
