# Hydra: Marker-Free RGB-D Hand-Eye Calibration 

**Title (ZH)**: Hydra: 无标记点的RGB-D 手眼标定 

**Authors**: Martin Huber, Huanyu Tian, Christopher E. Mower, Lucas-Raphael Müller, Sébastien Ourselin, Christos Bergeles, Tom Vercauteren  

**Link**: [PDF](https://arxiv.org/pdf/2504.20584)  

**Abstract**: This work presents an RGB-D imaging-based approach to marker-free hand-eye calibration using a novel implementation of the iterative closest point (ICP) algorithm with a robust point-to-plane (PTP) objective formulated on a Lie algebra. Its applicability is demonstrated through comprehensive experiments using three well known serial manipulators and two RGB-D cameras. With only three randomly chosen robot configurations, our approach achieves approximately 90% successful calibrations, demonstrating 2-3x higher convergence rates to the global optimum compared to both marker-based and marker-free baselines. We also report 2 orders of magnitude faster convergence time (0.8 +/- 0.4 s) for 9 robot configurations over other marker-free methods. Our method exhibits significantly improved accuracy (5 mm in task space) over classical approaches (7 mm in task space) whilst being marker-free. The benchmarking dataset and code are open sourced under Apache 2.0 License, and a ROS 2 integration with robot abstraction is provided to facilitate deployment. 

**Abstract (ZH)**: 基于RGB-D成像的无标记手眼标定方法：结合Lie代数上鲁棒的点到平面目标函数的迭代最近点算法新实现 

---
# SPARK Hand: Scooping-Pinching Adaptive Robotic Hand with Kempe Mechanism for Vertical Passive Grasp in Environmental Constraints 

**Title (ZH)**: SPARK 手爪：基于 Kempe 机构的适应性抓取手爪，适用于环境约束下的垂直被动抓取 

**Authors**: Jiaqi Yin, Tianyi Bi, Wenzeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.20506)  

**Abstract**: This paper presents the SPARK finger, an innovative passive adaptive robotic finger capable of executing both parallel pinching and scooping grasps. The SPARK finger incorporates a multi-link mechanism with Kempe linkages to achieve a vertical linear fingertip trajectory. Furthermore, a parallelogram linkage ensures the fingertip maintains a fixed orientation relative to the base, facilitating precise and stable manipulation. By integrating these mechanisms with elastic elements, the design enables effective interaction with surfaces, such as tabletops, to handle challenging objects. The finger employs a passive switching mechanism that facilitates seamless transitions between pinching and scooping modes, adapting automatically to various object shapes and environmental constraints without additional actuators. To demonstrate its versatility, the SPARK Hand, equipped with two SPARK fingers, has been developed. This system exhibits enhanced grasping performance and stability for objects of diverse sizes and shapes, particularly thin and flat objects that are traditionally challenging for conventional grippers. Experimental results validate the effectiveness of the SPARK design, highlighting its potential for robotic manipulation in constrained and dynamic environments. 

**Abstract (ZH)**: SPARK指尖：一种创新的被动自适应机械手指，兼具平行夹持和舀取抓握能力 

---
# Combining Quality of Service and System Health Metrics in MAPE-K based ROS Systems through Behavior Trees 

**Title (ZH)**: 基于行为树结合服务质量与系统健康指标的MAPE-K机制在ROS系统中的应用 

**Authors**: Andreas Wiedholz, Rafael Paintner, Julian Gleißner, Alwin Hoffmann  

**Link**: [PDF](https://arxiv.org/pdf/2504.20477)  

**Abstract**: In recent years, the field of robotics has witnessed a significant shift from operating in structured environments to handling dynamic and unpredictable settings. To tackle these challenges, methodologies from the field of self-adaptive systems enabling these systems to react to unforeseen circumstances during runtime have been applied. The Monitoring-Analysis- Planning-Execution over Knowledge (MAPE-K) feedback loop model is a popular approach, often implemented in a managing subsystem, responsible for monitoring and adapting a managed subsystem. This work explores the implementation of the MAPE- K feedback loop based on Behavior Trees (BTs) within the Robot Operating System 2 (ROS2) framework. By delineating the managed and managing subsystems, our approach enhances the flexibility and adaptability of ROS-based systems, ensuring they not only meet Quality-of-Service (QoS), but also system health metric requirements, namely availability of ROS nodes and communication channels. Our implementation allows for the application of the method to new managed subsystems without needing custom BT nodes as the desired behavior can be configured within a specific rule set. We demonstrate the effectiveness of our method through various experiments on a system showcasing an aerial perception use case. By evaluating different failure cases, we show both an increased perception quality and a higher system availability. Our code is open source 

**Abstract (ZH)**: 近年来，机器人领域见证了从操作结构化环境向处理动态和不可预测环境的转变。为了应对这些挑战，来自自适应系统领域的方法被应用，使这些系统能够在运行时对不可预见的情况作出反应。MAPE-K反馈循环模型是一种流行的方法，通常在管理子系统中实现，负责监控和适应被管理的子系统。本文探讨了基于行为树（Behavior Trees，BTs）在Robot Operating System 2（ROS2）框架中实现MAPE-K反馈循环的方法。通过界定被管理子系统和管理子系统，我们的方法增强了基于ROS的系统的灵活性和适应性，确保它们不仅满足服务质量（QoS）要求，还满足系统健康指标要求，即ROS节点和通信通道的可用性。我们的实现允许将该方法应用于新的被管理子系统，无需为所需行为创建自定义的行为树节点，只需配置特定的规则集即可。通过在展示空中感知应用案例的系统上进行各种实验，我们展示了该方法的有效性。通过评估不同的故障情况，我们表明了感知质量的提高和系统可用性的增加。我们的代码是开源的。标题：基于行为树在ROS2框架中实现MAPE-K反馈循环的方法 

---
# DRO: Doppler-Aware Direct Radar Odometry 

**Title (ZH)**: DRO：多普勒感知直接雷达里程计 

**Authors**: Cedric Le Gentil, Leonardo Brizi, Daniil Lisus, Xinyuan Qiao, Giorgio Grisetti, Timothy D. Barfoot  

**Link**: [PDF](https://arxiv.org/pdf/2504.20339)  

**Abstract**: A renaissance in radar-based sensing for mobile robotic applications is underway. Compared to cameras or lidars, millimetre-wave radars have the ability to `see' through thin walls, vegetation, and adversarial weather conditions such as heavy rain, fog, snow, and dust. In this paper, we propose a novel SE(2) odometry approach for spinning frequency-modulated continuous-wave radars. Our method performs scan-to-local-map registration of the incoming radar data in a direct manner using all the radar intensity information without the need for feature or point cloud extraction. The method performs locally continuous trajectory estimation and accounts for both motion and Doppler distortion of the radar scans. If the radar possesses a specific frequency modulation pattern that makes radial Doppler velocities observable, an additional Doppler-based constraint is formulated to improve the velocity estimate and enable odometry in geometrically feature-deprived scenarios (e.g., featureless tunnels). Our method has been validated on over 250km of on-road data sourced from public datasets (Boreas and MulRan) and collected using our automotive platform. With the aid of a gyroscope, it outperforms state-of-the-art methods and achieves an average relative translation error of 0.26% on the Boreas leaderboard. When using data with the appropriate Doppler-enabling frequency modulation pattern, the translation error is reduced to 0.18% in similar environments. We also benchmarked our algorithm using 1.5 hours of data collected with a mobile robot in off-road environments with various levels of structure to demonstrate its versatility. Our real-time implementation is publicly available: this https URL. 

**Abstract (ZH)**: 基于雷达的移动机器人应用传感复兴正在进行中。与摄像头或激光雷达相比，毫米波雷达能够在透过薄墙、植被以及恶劣天气（如大雨、雾、雪和尘埃）的情况下“看”到目标。本文提出了一种针对旋转频率调制连续波雷达的新型SE(2)里程计方法。该方法直接利用所有雷达强度信息进行进来雷达数据与局部地图的配准，无需提取特征或点云。方法实现了局部连续轨迹估计，并同时考虑了雷达扫描的运动和多普勒失真。如果雷达具有特定的频率调制模式使其径向多普勒速度可观测，将额外引入多普勒约束以改进速度估计，并在几何特征缺乏的场景中（如无特征隧道）实现里程计。该方法已在公共数据集（Boreas和MulRan）的超过250公里路面上数据以及使用我们的汽车平台收集的数据上进行了验证。借助陀螺仪，该方法在Boreas排行榜上优于现有最佳方法，平均相对位移误差为0.26%。在具有适当多普勒使能频率调制模式的数据中，相同环境下的位移误差可降低至0.18%。我们还在不同结构水平的野外环境中用移动机器人收集了1.5小时的数据，以此来验证其通用性。我们的实时实现已公开：this https URL。 

---
# NMPC-based Unified Posture Manipulation and Thrust Vectoring for Agile and Fault-Tolerant Flight of a Morphing Aerial Robot 

**Title (ZH)**: 基于NMPC的形态变化航空机器人的一体化姿态操控与推力矢量控制敏捷及容错飞行方法 

**Authors**: Shashwat Pandya  

**Link**: [PDF](https://arxiv.org/pdf/2504.20326)  

**Abstract**: This thesis presents a unified control framework for agile and fault-tolerant flight of the Multi-Modal Mobility Morphobot (M4) in aerial mode. The M4 robot is capable of transitioning between ground and aerial locomotion. The articulated legs enable more dynamic maneuvers than a standard quadrotor platform. A nonlinear model predictive control (NMPC) approach is developed to simultaneously plan posture manipulation and thrust vectoring actions, allowing the robot to execute sharp turns and dynamic flight trajectories. The framework integrates an agile and fault-tolerant control logic that enables precise tracking under aggressive maneuvers while compensating for actuator failures, ensuring continued operation without significant performance degradation. Simulation results validate the effectiveness of the proposed method, demonstrating accurate trajectory tracking and robust recovery from faults, contributing to resilient autonomous flight in complex environments. 

**Abstract (ZH)**: 本论文提出了一种统一的控制框架，用于Multi-Modal Mobility Morphobot (M4) 无人机模式下的敏捷和容错飞行控制。M4 机器人能够在地面和空中运动间转换。其 articulated 腿使得动作更为动态，超过了标准四旋翼平台。开发了一种非线性模型预测控制（NMPC）方法，同时规划姿态操作和推力矢量动作，使机器人能够执行快速转弯和动态飞行轨迹。该框架整合了敏捷和容错控制逻辑，能够在剧烈机动下实现精确跟踪，并补偿执行器故障，确保在不显著性能下降的情况下持续运行。仿真结果验证了所提出方法的有效性，展示了精确的轨迹跟踪和从故障中稳健恢复的能力，从而为复杂环境下的可靠自主飞行做出了贡献。 

---
# GenGrid: A Generalised Distributed Experimental Environmental Grid for Swarm Robotics 

**Title (ZH)**: GenGrid: 一种通用的分布式实验环境网格在 swarm 机器人中的应用 

**Authors**: Pranav Kedia, Madhav Rao  

**Link**: [PDF](https://arxiv.org/pdf/2504.20071)  

**Abstract**: GenGrid is a novel comprehensive open-source, distributed platform intended for conducting extensive swarm robotic experiments. The modular platform is designed to run swarm robotics experiments that are compatible with different types of mobile robots ranging from Colias, Kilobot, and E puck. The platform offers programmable control over the experimental setup and its parameters and acts as a tool to collect swarm robot data, including localization, sensory feedback, messaging, and interaction. GenGrid is designed as a modular grid of attachable computing nodes that offers bidirectional communication between the robotic agent and grid nodes and within grids. The paper describes the hardware and software architecture design of the GenGrid system. Further, it discusses some common experimental studies covering multi-robot and swarm robotics to showcase the platform's use. GenGrid of 25 homogeneous cells with identical sensing and communication characteristics with a footprint of 37.5 cm X 37.5 cm, exhibits multiple capabilities with minimal resources. The open-source hardware platform is handy for running swarm experiments, including robot hopping based on multiple gradients, collective transport, shepherding, continuous pheromone deposition, and subsequent evaporation. The low-cost, modular, and open-source platform is significant in the swarm robotics research community, which is currently driven by commercial platforms that allow minimal modifications. 

**Abstract (ZH)**: GenGrid是一种新型综合开源分布式平台，旨在进行广泛的 swarm 机器人实验。 

---
