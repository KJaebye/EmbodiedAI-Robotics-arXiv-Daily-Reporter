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
# UWB Anchor Based Localization of a Planetary Rover 

**Title (ZH)**: 基于UWB锚点的行星车定位 

**Authors**: Andreas Nüchter, Lennart Werner, Martin Hesse, Dorit Borrmann, Thomas Walter, Sergio Montenegro, Gernot Grömer  

**Link**: [PDF](https://arxiv.org/pdf/2504.07658)  

**Abstract**: Localization of an autonomous mobile robot during planetary exploration is challenging due to the unknown terrain, the difficult lighting conditions and the lack of any global reference such as satellite navigation systems. We present a novel approach for robot localization based on ultra-wideband (UWB) technology. The robot sets up its own reference coordinate system by distributing UWB anchor nodes in the environment via a rocket-propelled launcher system. This allows the creation of a localization space in which UWB measurements are employed to supplement traditional SLAM-based techniques. The system was developed for our involvement in the ESA-ESRIC challenge 2021 and the AMADEE-24, an analog Mars simulation in Armenia by the Austrian Space Forum (ÖWF). 

**Abstract (ZH)**: 行星探测中自主移动机器人的定位因其未知地形、恶劣光照条件以及缺乏如卫星导航系统等全球参考而具有挑战性。基于超宽带(UWB)技术的机器人定位方法研究：通过火箭推进发射系统布置UWB锚节点建立自主参考坐标系，以补充传统SLAM技术并构建局部定位空间。该系统用于参与ESA-ESRIC 2021挑战赛及奥利地空间论坛(OEF)在 Armenian 的模拟火星环境AMADEE-24项目。 

---
# Efficient Swept Volume-Based Trajectory Generation for Arbitrary-Shaped Ground Robot Navigation 

**Title (ZH)**: 基于扫掠体积的任意形状地面机器人轨迹生成高效算法 

**Authors**: Yisheng Li, Longji Yin, Yixi Cai, Jianheng Liu, Haotian Li, Fu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07554)  

**Abstract**: Navigating an arbitrary-shaped ground robot safely in cluttered environments remains a challenging problem. The existing trajectory planners that account for the robot's physical geometry severely suffer from the intractable runtime. To achieve both computational efficiency and Continuous Collision Avoidance (CCA) of arbitrary-shaped ground robot planning, we proposed a novel coarse-to-fine navigation framework that significantly accelerates planning. In the first stage, a sampling-based method selectively generates distinct topological paths that guarantee a minimum inflated margin. In the second stage, a geometry-aware front-end strategy is designed to discretize these topologies into full-state robot motion sequences while concurrently partitioning the paths into SE(2) sub-problems and simpler R2 sub-problems for back-end optimization. In the final stage, an SVSDF-based optimizer generates trajectories tailored to these sub-problems and seamlessly splices them into a continuous final motion plan. Extensive benchmark comparisons show that the proposed method is one to several orders of magnitude faster than the cutting-edge methods in runtime while maintaining a high planning success rate and ensuring CCA. 

**Abstract (ZH)**: 导航具有任意形状的地面机器人在复杂环境中的安全路径规划仍然是一个具有挑战性的问题。为同时实现计算效率和任意形状地面机器人的连续碰撞避免，我们提出了一种新的粗细结合导航框架，显著加速路径规划。该框架的首个阶段通过采样方法选择性生成保证最小膨胀边界的独特拓扑路径。第二个阶段设计了一个几何感知的前端策略，将这些拓扑路径离散化为完整的机器人状态运动序列，并同时将路径分割为SE(2)子问题和更简单的R2子问题用于后端优化。最终阶段使用基于SVSDF的优化器为这些子问题生成定制轨迹，并无缝拼接成连续的最终运动计划。广泛的基准比较结果表明，与最先进的方法相比，所提出的方法在运行时间上快一个到几个数量级，同时保持高规划成功率并确保连续碰撞避免。 

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
# Joint Travel Route Optimization Framework for Platooning 

**Title (ZH)**: 编队行驶的联合旅游路线优化框架 

**Authors**: Akif Adas, Stefano Arrigoni, Mattia Brambilla, Monica Barbara Nicoli, Edoardo Sabbioni  

**Link**: [PDF](https://arxiv.org/pdf/2504.07623)  

**Abstract**: Platooning represents an advanced driving technology designed to assist drivers in traffic convoys of varying lengths, enhancing road safety, reducing driver fatigue, and improving fuel efficiency. Sophisticated automated driving assistance systems have facilitated this innovation. Recent advancements in platooning emphasize cooperative mechanisms within both centralized and decentralized architectures enabled by vehicular communication technologies. This study introduces a cooperative route planning optimization framework aimed at promoting the adoption of platooning through a centralized platoon formation strategy at the system level. This approach is envisioned as a transitional phase from individual (ego) driving to fully collaborative driving. Additionally, this research formulates and incorporates travel cost metrics related to fuel consumption, driver fatigue, and travel time, considering regulatory constraints on consecutive driving durations. The performance of these cost metrics has been evaluated using Dijkstra's and A* shortest path algorithms within a network graph framework. The results indicate that the proposed architecture achieves an average cost improvement of 14 % compared to individual route planning for long road trips. 

**Abstract (ZH)**: Platooning代表了一种先进的驾驶技术，旨在协助驾驶者在不同长度的车队中行驶，提升道路安全，减少驾驶疲劳，并提高燃油效率。复杂的自动化驾驶辅助系统促进了这一创新。近期的platooning发展强调了由车辆通信技术支撑的集中式和分布式架构中的协同机制。本研究提出了一种协同路线规划优化框架，旨在通过系统层面的集中式编队策略促进platooning的普及。该方法被视为从个体（ ego）驾驶向完全协作驾驶过渡的阶段。此外，本研究制定了并融入了与燃料消耗、驾驶疲劳和旅行时间相关的旅行成本指标，并考虑了连续驾驶时间的监管限制。这些成本指标的应用通过网络图框架中的Dijkstra和A*最短路径算法进行评估。结果显示，所提出的架构在长距离旅行中平均成本改进了14%。 

---
