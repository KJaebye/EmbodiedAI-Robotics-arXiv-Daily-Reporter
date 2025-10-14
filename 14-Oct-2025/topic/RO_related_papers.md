# Calibrated Dynamic Modeling for Force and Payload Estimation in Hydraulic Machinery 

**Title (ZH)**: 液压机械中力和载荷估算的校准动态建模 

**Authors**: Lennart Werner, Pol Eyschen, Sean Costello, Pierluigi Micarelli, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2510.11574)  

**Abstract**: Accurate real-time estimation of end effector interaction forces in hydraulic excavators is a key enabler for advanced automation in heavy machinery. Accurate knowledge of these forces allows improved, precise grading and digging maneuvers. To address these challenges, we introduce a high-accuracy, retrofittable 2D force- and payload estimation algorithm that does not impose additional requirements on the operator regarding trajectory, acceleration or the use of the slew joint. The approach is designed for retrofittability, requires minimal calibration and no prior knowledge of machine-specific dynamic characteristics. Specifically, we propose a method for identifying a dynamic model, necessary to estimate both end effector interaction forces and bucket payload during normal operation. Our optimization-based payload estimation achieves a full-scale payload accuracy of 1%. On a standard 25 t excavator, the online force measurement from pressure and inertial measurements achieves a direction accuracy of 13 degree and a magnitude accuracy of 383 N. The method's accuracy and generalization capability are validated on two excavator platforms of different type and weight classes. We benchmark our payload estimation against a classical quasistatic method and a commercially available system. Our system outperforms both in accuracy and precision. 

**Abstract (ZH)**: 液压挖掘机末端执行器实时交互力的高精度估算对于重型机械的高级自动化是一个关键 enabler。了解这些力的准确信息能够改善和精确控制铲掘操作。为应对这些挑战，我们提出了一种高精度、可 retrofit 的二维力和负载估算算法，该算法不对操作员的轨迹、加速度或 slew 联轴器的使用提出额外要求。该方法设计用于 retrofit，需要最少的校准且无需了解特定机器的动态特性。具体而言，我们提出了一种识别动态模型的方法，该模型对于估算正常操作过程中的末端执行器交互力和铲斗负载是必要的。基于优化的负载估算达到了 1% 的全尺度负载精度。在标准 25 吨挖掘机上，基于压力和惯性测量的在线力测量在方向上的精度为 13 度，在大小上的精度为 383 N。该方法的准确性和泛化能力在两类不同类型的挖掘机平台上得到了验证。我们将负载估算与经典准静态方法和商用系统进行了基准测试，我们的系统在准确性和精度方面均优于两者。 

---
# Robot Soccer Kit: Omniwheel Tracked Soccer Robots for Education 

**Title (ZH)**: 全自动轮足球机器人套件：用于教育的全方位行走足球机器人 

**Authors**: Gregoire Passault, Clement Gaspard, Olivier Ly  

**Link**: [PDF](https://arxiv.org/pdf/2510.11552)  

**Abstract**: Recent developments of low cost off-the-shelf programmable components, their modularity, and also rapid prototyping made educational robotics flourish, as it is accessible in most schools today. They allow to illustrate and embody theoretical problems in practical and tangible applications, and gather multidisciplinary skills. They also give a rich natural context for project-oriented pedagogy. However, most current robot kits all are limited to egocentric aspect of the robots perception. This makes it difficult to access more high-level problems involving e.g. coordinates or navigation. In this paper we introduce an educational holonomous robot kit that comes with an external tracking system, which lightens the constraint on embedded systems, but allows in the same time to discover high-level aspects of robotics, otherwise unreachable. 

**Abstract (ZH)**: 低成本即插即用可编程组件及其模块化和快速原型制作技术促进了教育机器人的发展，使其在今天的大多数学校中变得可行。它们能够将理论问题通过实际和具体的应用进行展示和体现，汇集多学科技能，并为项目导向的教学提供丰富的自然背景。然而，当前大多数机器人套件仅限于机器人自身的视角感知，这使得访问涉及坐标或导航等更高层次的问题变得困难。在本文中，我们介绍了一种配备外部跟踪系统的教育全向机器人套件，该套件减轻了嵌入式系统的约束，同时允许探索其他难以触及的高级机器人方面。 

---
# DQ-NMPC: Dual-Quaternion NMPC for Quadrotor Flight 

**Title (ZH)**: DQ-NMPC: 双四元数NMPC在四旋翼飞行中的应用 

**Authors**: Luis F. Recalde, Dhruv Agrawal, Jon Arrizabalaga, Guanrui Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.11525)  

**Abstract**: MAVs have great potential to assist humans in complex tasks, with applications ranging from logistics to emergency response. Their agility makes them ideal for operations in complex and dynamic environments. However, achieving precise control in agile flights remains a significant challenge, particularly due to the underactuated nature of quadrotors and the strong coupling between their translational and rotational dynamics. In this work, we propose a novel NMPC framework based on dual-quaternions (DQ-NMPC) for quadrotor flight. By representing both quadrotor dynamics and the pose error directly on the dual-quaternion manifold, our approach enables a compact and globally non-singular formulation that captures the quadrotor coupled dynamics. We validate our approach through simulations and real-world experiments, demonstrating better numerical conditioning and significantly improved tracking performance, with reductions in position and orientation errors of up to 56.11% and 56.77%, compared to a conventional baseline NMPC method. Furthermore, our controller successfully handles aggressive trajectories, reaching maximum speeds up to 13.66 m/s and accelerations reaching 4.2 g within confined space conditions of dimensions 11m x 4.5m x 3.65m under which the baseline controller fails. 

**Abstract (ZH)**: MAVs在复杂任务中的潜在应用从物流到应急响应广泛存在，其敏捷性使它们成为复杂和动态环境操作的理想选择。然而，在敏捷飞行中实现精确控制依然面临重大挑战，特别是在四旋翼机欠驱动的特性和其平移和旋转动力学的强耦合的影响下。本文提出了一种基于双四元数（DQ-NMPC）的新颖NMPC框架，用于四旋翼机飞行。通过在双四元数流形上直接表示四旋翼机动力学和姿态误差，我们的方法能够提供一个紧凑且全局非奇异的公式，从而捕捉四旋翼机的耦合动力学。通过仿真和实际实验验证了我们方法的有效性，展示了更好的数值条件和显著提高的跟踪性能，位置和姿态误差分别减少了56.11%和56.77%，与传统的基准NMPC方法相比。此外，在受限空间条件下，我们的控制器成功处理了激进的轨迹，最大速度达到13.66 m/s，加速度达到4.2 g，而基准控制器在此条件下失效。 

---
# Coordinated Strategies in Realistic Air Combat by Hierarchical Multi-Agent Reinforcement Learning 

**Title (ZH)**: 现实空中格斗中的分层次多agent强化学习协同策略 

**Authors**: Ardian Selmonaj, Giacomo Del Rio, Adrian Schneider, Alessandro Antonucci  

**Link**: [PDF](https://arxiv.org/pdf/2510.11474)  

**Abstract**: Achieving mission objectives in a realistic simulation of aerial combat is highly challenging due to imperfect situational awareness and nonlinear flight dynamics. In this work, we introduce a novel 3D multi-agent air combat environment and a Hierarchical Multi-Agent Reinforcement Learning framework to tackle these challenges. Our approach combines heterogeneous agent dynamics, curriculum learning, league-play, and a newly adapted training algorithm. To this end, the decision-making process is organized into two abstraction levels: low-level policies learn precise control maneuvers, while high-level policies issue tactical commands based on mission objectives. Empirical results show that our hierarchical approach improves both learning efficiency and combat performance in complex dogfight scenarios. 

**Abstract (ZH)**: 在现实istic模拟空中战斗中实现任务目标由于情况认知不完整和非线性飞行动力学而极具挑战性。本文介绍了一种新颖的3D多agent空中 combat环境和层次化多agent强化学习框架以应对这些挑战。我们的方法结合了异构agent动力学、阶梯式学习、联赛对战以及一种新的训练算法。在此基础上，决策过程组织为两个抽象层次：低层策略学习精确的控制机动，而高层策略基于任务目标发布战术命令。实验证明，我们的层次化方法在复杂缠斗场景中提高了学习效率和战斗性能。 

---
# Path and Motion Optimization for Efficient Multi-Location Inspection with Humanoid Robots 

**Title (ZH)**: 人形机器人多地点高效检查的路径与运动优化 

**Authors**: Jiayang Wu, Jiongye Li, Shibowen Zhang, Zhicheng He, Zaijin Wang, Xiaokun Leng, Hangxin Liu, Jingwen Zhang, Jiayi Wang, Song-Chun Zhu, Yao Su  

**Link**: [PDF](https://arxiv.org/pdf/2510.11401)  

**Abstract**: This paper proposes a novel framework for humanoid robots to execute inspection tasks with high efficiency and millimeter-level precision. The approach combines hierarchical planning, time-optimal standing position generation, and integrated \ac{mpc} to achieve high speed and precision. A hierarchical planning strategy, leveraging \ac{ik} and \ac{mip}, reduces computational complexity by decoupling the high-dimensional planning problem. A novel MIP formulation optimizes standing position selection and trajectory length, minimizing task completion time. Furthermore, an MPC system with simplified kinematics and single-step position correction ensures millimeter-level end-effector tracking accuracy. Validated through simulations and experiments on the Kuavo 4Pro humanoid platform, the framework demonstrates low time cost and a high success rate in multi-location tasks, enabling efficient and precise execution of complex industrial operations. 

**Abstract (ZH)**: This paper提出了一种新的框架，用于使类人机器人以高效率和毫米级精度执行检查任务。该方法结合了层次规划、最优站立位置生成和集成的模型预测控制（MPC），以实现高速度和高精度。通过利用逆运动学（IK）和混合整数规划（MIP）的层次规划策略，减少计算复杂性，分解高维规划问题。一种新颖的MIP公式优化站立位置选择和轨迹长度，最小化任务完成时间。此外，简化动力学的MPC系统和单步位置校正确保末端执行器跟踪精度达到毫米级。通过在Kuavo 4Pro类人平台上进行仿真和实验验证，该框架在多位置任务中显示出低时间成本和高成功率，从而实现复杂工业操作的高效和精确执行。 

---
# Adap-RPF: Adaptive Trajectory Sampling for Robot Person Following in Dynamic Crowded Environments 

**Title (ZH)**: Adap-RPF: 动态拥挤环境中适应性人体跟随轨迹采样算法 

**Authors**: Weixi Situ, Hanjing Ye, Jianwei Peng, Yu Zhan, Hong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11308)  

**Abstract**: Robot person following (RPF) is a core capability in human-robot interaction, enabling robots to assist users in daily activities, collaborative work, and other service scenarios. However, achieving practical RPF remains challenging due to frequent occlusions, particularly in dynamic and crowded environments. Existing approaches often rely on fixed-point following or sparse candidate-point selection with oversimplified heuristics, which cannot adequately handle complex occlusions caused by moving obstacles such as pedestrians. To address these limitations, we propose an adaptive trajectory sampling method that generates dense candidate points within socially aware zones and evaluates them using a multi-objective cost function. Based on the optimal point, a person-following trajectory is estimated relative to the predicted motion of the target. We further design a prediction-aware model predictive path integral (MPPI) controller that simultaneously tracks this trajectory and proactively avoids collisions using predicted pedestrian motions. Extensive experiments show that our method outperforms state-of-the-art baselines in smoothness, safety, robustness, and human comfort, with its effectiveness further demonstrated on a mobile robot in real-world scenarios. 

**Abstract (ZH)**: 基于社交aware区域的自适应轨迹采样方法及其在人机交互中的应用研究：一种用于机器人跟随的人群跟踪方法 

---
# Rotor-Failure-Aware Quadrotors Flight in Unknown Environments 

**Title (ZH)**: 基于未知环境中的旋翼无人机故障感知飞行 

**Authors**: Xiaobin Zhou, Miao Wang, Chengao Li, Can Cui, Ruibin Zhang, Yongchao Wang, Chao Xu, Fei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2510.11306)  

**Abstract**: Rotor failures in quadrotors may result in high-speed rotation and vibration due to rotor imbalance, which introduces significant challenges for autonomous flight in unknown environments. The mainstream approaches against rotor failures rely on fault-tolerant control (FTC) and predefined trajectory tracking. To the best of our knowledge, online failure detection and diagnosis (FDD), trajectory planning, and FTC of the post-failure quadrotors in unknown and complex environments have not yet been achieved. This paper presents a rotor-failure-aware quadrotor navigation system designed to mitigate the impacts of rotor imbalance. First, a composite FDD-based nonlinear model predictive controller (NMPC), incorporating motor dynamics, is designed to ensure fast failure detection and flight stability. Second, a rotor-failure-aware planner is designed to leverage FDD results and spatial-temporal joint optimization, while a LiDAR-based quadrotor platform with four anti-torque plates is designed to enable reliable perception under high-speed rotation. Lastly, extensive benchmarks against state-of-the-art methods highlight the superior performance of the proposed approach in addressing rotor failures, including propeller unloading and motor stoppage. The experimental results demonstrate, for the first time, that our approach enables autonomous quadrotor flight with rotor failures in challenging environments, including cluttered rooms and unknown forests. 

**Abstract (ZH)**: 四旋翼飞行器转子故障感知导航系统设计与实现 

---
# Design and Koopman Model Predictive Control of A Soft Exoskeleton Based on Origami-Inspired Pneumatic Actuator for Knee Rehabilitation 

**Title (ZH)**: 基于 Origami 风格气动执行器的软外骨骼设计与折纸启发式 Koopman 模型预测控制研究：用于膝关节康复 

**Authors**: Junxiang Wang, Han Zhang, Zehao Wang, Huaiyuan Chen, Pu Wang, Weidong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.11094)  

**Abstract**: Effective rehabilitation methods are essential for the recovery of lower limb dysfunction caused by stroke. Nowadays, robotic exoskeletons have shown great potentials in rehabilitation. Nevertheless, traditional rigid exoskeletons are usually heavy and need a lot of work to help the patients to put them on. Moreover, it also requires extra compliance control to guarantee the safety. In contrast, soft exoskeletons are easy and comfortable to wear and have intrinsic compliance, but their complex nonlinear human-robot interaction dynamics would pose significant challenges for control. In this work, based on the pneumatic actuators inspired by origami, we design a rehabilitation exoskeleton for knee that is easy and comfortable to wear. To guarantee the control performance and enable a nice human-robot interaction, we first use Deep Koopman Network to model the human-robot interaction dynamics. In particular, by viewing the electromyography (EMG) signals and the duty cycle of the PWM wave that controls the pneumatic robot's valves and pump as the inputs, the linear Koopman model accurately captures the complex human-robot interaction dynamics. Next, based on the obtained Koopman model, we further use Model Predictive Control (MPC) to control the soft robot and help the user to do rehabilitation training in real-time. The goal of the rehabilitation training is to track a given reference signal shown on the screen. Experiments show that by integrating the EMG signals into the Koopman model, we have improved the model accuracy to great extent. In addition, a personalized Koopman model trained from the individual's own data performs better than the non-personalized model. Consequently, our control framework outperforms the traditional PID control in both passive and active training modes. Hence the proposed method provides a new control framework for soft rehabilitation robots. 

**Abstract (ZH)**: 基于Origami启发的气动 actuators设计的易于穿戴的膝关节康复外骨骼：基于Koopman网络的模型预测控制方法 

---
# XGrasp: Gripper-Aware Grasp Detection with Multi-Gripper Data Generation 

**Title (ZH)**: XGrasp: 多爪夹数据生成的夹持检测与爪夹aware方法 

**Authors**: Yeonseo Lee, Jungwook Mun, Hyosup Shin, Guebin Hwang, Junhee Nam, Taeyeop Lee, Sungho Jo  

**Link**: [PDF](https://arxiv.org/pdf/2510.11036)  

**Abstract**: Most robotic grasping methods are typically designed for single gripper types, which limits their applicability in real-world scenarios requiring diverse end-effectors. We propose XGrasp, a real-time gripper-aware grasp detection framework that efficiently handles multiple gripper configurations. The proposed method addresses data scarcity by systematically augmenting existing datasets with multi-gripper annotations. XGrasp employs a hierarchical two-stage architecture. In the first stage, a Grasp Point Predictor (GPP) identifies optimal locations using global scene information and gripper specifications. In the second stage, an Angle-Width Predictor (AWP) refines the grasp angle and width using local features. Contrastive learning in the AWP module enables zero-shot generalization to unseen grippers by learning fundamental grasping characteristics. The modular framework integrates seamlessly with vision foundation models, providing pathways for future vision-language capabilities. The experimental results demonstrate competitive grasp success rates across various gripper types, while achieving substantial improvements in inference speed compared to existing gripper-aware methods. Project page: this https URL 

**Abstract (ZH)**: 多 gripper 配置aware 的实时夹持检测框架 XGrasp 

---
# AMO-HEAD: Adaptive MARG-Only Heading Estimation for UAVs under Magnetic Disturbances 

**Title (ZH)**: AMO-HEAD：在磁场干扰下适用于无人机的自适应磁罗盘-only航向估计方法 

**Authors**: Qizhi Guo, Siyuan Yang, Junning Lyu, Jianjun Sun, Defu Lin, Shaoming He  

**Link**: [PDF](https://arxiv.org/pdf/2510.10979)  

**Abstract**: Accurate and robust heading estimation is crucial for unmanned aerial vehicles (UAVs) when conducting indoor inspection tasks. However, the cluttered nature of indoor environments often introduces severe magnetic disturbances, which can significantly degrade heading accuracy. To address this challenge, this paper presents an Adaptive MARG-Only Heading (AMO-HEAD) estimation approach for UAVs operating in magnetically disturbed environments. AMO-HEAD is a lightweight and computationally efficient Extended Kalman Filter (EKF) framework that leverages inertial and magnetic sensors to achieve reliable heading estimation. In the proposed approach, gyroscope angular rate measurements are integrated to propagate the quaternion state, which is subsequently corrected using accelerometer and magnetometer data. The corrected quaternion is then used to compute the UAV's heading. An adaptive process noise covariance method is introduced to model and compensate for gyroscope measurement noise, bias drift, and discretization errors arising from the Euler method integration. To mitigate the effects of external magnetic disturbances, a scaling factor is applied based on real-time magnetic deviation detection. A theoretical observability analysis of the proposed AMO-HEAD is performed using the Lie derivative. Extensive experiments were conducted in real world indoor environments with customized UAV platforms. The results demonstrate the effectiveness of the proposed algorithm in providing precise heading estimation under magnetically disturbed conditions. 

**Abstract (ZH)**: 磁干扰环境下无人机精确鲁棒航向估计的自适应MARG-only方法 

---
# QuayPoints: A Reasoning Framework to Bridge the Information Gap Between Global and Local Planning in Autonomous Racing 

**Title (ZH)**: 岸线点：一种连接自主赛车全局规划与局部规划信息缺口的推理框架 

**Authors**: Yashom Dighe, Youngjin Kim, Karthik Dantu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10886)  

**Abstract**: Autonomous racing requires tight integration between perception, planning and control to minimize latency as well as timely decision making. A standard autonomy pipeline comprising a global planner, local planner, and controller loses information as the higher-level racing context is sequentially propagated downstream into specific task-oriented context. In particular, the global planner's understanding of optimality is typically reduced to a sparse set of waypoints, leaving the local planner to make reactive decisions with limited context. This paper investigates whether additional global insights, specifically time-optimality information, can be meaningfully passed to the local planner to improve downstream decisions. We introduce a framework that preserves essential global knowledge and conveys it to the local planner through QuayPoints regions where deviations from the optimal raceline result in significant compromises to optimality. QuayPoints enable local planners to make more informed global decisions when deviating from the raceline, such as during strategic overtaking. To demonstrate this, we integrate QuayPoints into an existing planner and show that it consistently overtakes opponents traveling at up to 75% of the ego vehicle's speed across four distinct race tracks. 

**Abstract (ZH)**: 自主赛车需要在感知、规划和控制之间实现紧密集成，以最大限度地减少延迟并及时做出决策。一个标准的自主驾驶流水线包括全局规划器、局部规划器和控制器，但在将高层次的赛车上下文逐次传递到特定任务导向的上下文时会丢失信息。特别是，全局规划器对最优性的理解通常仅限于稀疏的 waypoints，从而使局部规划器只能在有限的上下文中作出反应性决策。本文探讨是否可以通过传递额外的全局洞察，特别是时间最优性信息，来改善下游决策。我们提出了一种框架，通过保留关键的全局知识并在 QuayPoints 区域内传达这些信息，使得局部规划器在偏离赛道线时能够做出更明智的全局决策，例如在战略性超越时。为了证明这一点，我们将 QuayPoints 集成到现有的规划器中，并展示了该方法在四条不同赛道上持续超越速度最高为其自身速度75%的对手的有效性。 

---
# Contact Sensing via Joint Torque Sensors and a Force/Torque Sensor for Legged Robots 

**Title (ZH)**: 基于关节扭矩传感器和力/扭矩传感器的接触感知 

**Authors**: Jared Grinberg, Yanran Ding  

**Link**: [PDF](https://arxiv.org/pdf/2510.10843)  

**Abstract**: This paper presents a method for detecting and localizing contact along robot legs using distributed joint torque sensors and a single hip-mounted force-torque (FT) sensor using a generalized momentum-based observer framework. We designed a low-cost strain-gauge-based joint torque sensor that can be installed on every joint to provide direct torque measurements, eliminating the need for complex friction models and providing more accurate torque readings than estimation based on motor current. Simulation studies on a floating-based 2-DoF robot leg verified that the proposed framework accurately recovers contact force and location along the thigh and shin links. Through a calibration procedure, our torque sensor achieved an average 96.4% accuracy relative to ground truth measurements. Building upon the torque sensor, we performed hardware experiments on a 2-DoF manipulator, which showed sub-centimeter contact localization accuracy and force errors below 0.2 N. 

**Abstract (ZH)**: 本文提出了一种使用分布式关节扭矩传感器和单个臀部安装的力-扭矩传感器，结合广义动量观察器框架来检测和定位机器人腿部接触的方法。 

---
# Representing Data in Robotic Tactile Perception -- A Review 

**Title (ZH)**: 机器人触觉感知中的数据表示——一个综述 

**Authors**: Alessandro Albini, Mohsen Kaboli, Giorgio Cannata, Perla Maiolino  

**Link**: [PDF](https://arxiv.org/pdf/2510.10804)  

**Abstract**: Robotic tactile perception is a complex process involving several computational steps performed at different levels. Tactile information is shaped by the interplay of robot actions, the mechanical properties of its body, and the software that processes the data. In this respect, high-level computation, required to process and extract information, is commonly performed by adapting existing techniques from other domains, such as computer vision, which expects input data to be properly structured. Therefore, it is necessary to transform tactile sensor data to match a specific data structure. This operation directly affects the tactile information encoded and, as a consequence, the task execution. This survey aims to address this specific aspect of the tactile perception pipeline, namely Data Representation. The paper first clearly defines its contributions to the perception pipeline and then reviews how previous studies have dealt with the problem of representing tactile information, investigating the relationships among hardware, representations, and high-level computation methods. The analysis has led to the identification of six structures commonly used in the literature to represent data. The manuscript provides discussions and guidelines for properly selecting a representation depending on operating conditions, including the available hardware, the tactile information required to be encoded, and the task at hand. 

**Abstract (ZH)**: 机器人触觉感知是一个涉及多个在不同层次上进行的计算步骤的复杂过程。触觉信息受到机器人动作、其身体的机械特性以及处理数据的软件的相互作用影响。在这方面，为了处理和提取信息所需的高层计算通常是通过适应来自其他领域（如计算机视觉）的现有技术来进行的，这些技术期望输入数据结构化。因此，有必要将触觉传感器数据转换为特定的数据结构。此操作直接影触觉信息编码，并且作为结果影响任务执行。本文综述旨在解决触觉感知管道中的这一特定方面，即数据表示。论文首先明确其对感知管道的贡献，然后回顾了先前研究如何处理表示触觉信息的问题，探讨了硬件、表示方法与高层计算方法之间的关系。分析工作已识别出文献中常用的基本结构六个。文章还提供了关于根据操作条件正确选择表示方法的讨论和指南，包括可用的硬件、需要编码的触觉信息以及当前任务。 

---
# Two-Layer Voronoi Coverage Control for Hybrid Aerial-Ground Robot Teams in Emergency Response: Implementation and Analysis 

**Title (ZH)**: 应急响应中混合 aerial-地面机器人团队的两层 Voronoi 覆盖控制：实现与分析 

**Authors**: Douglas Hutchings, Luai Abuelsamen, Karthik Rajgopal  

**Link**: [PDF](https://arxiv.org/pdf/2510.10781)  

**Abstract**: We present a comprehensive two-layer Voronoi coverage control approach for coordinating hybrid aerial-ground robot teams in hazardous material emergency response scenarios. Traditional Voronoi coverage control methods face three critical limitations in emergency contexts: heterogeneous agent capabilities with vastly different velocities, clustered initial deployment configurations, and urgent time constraints requiring rapid response rather than eventual convergence. Our method addresses these challenges through a decoupled two-layer architecture that separately optimizes aerial and ground robot positioning, with aerial agents delivering ground sensors via airdrop to high-priority locations. We provide detailed implementation of bounded Voronoi cell computation, efficient numerical integration techniques for importance-weighted centroids, and robust control strategies that prevent agent trapping. Simulation results demonstrate an 88% reduction in response time, achieving target sensor coverage (18.5% of initial sensor loss) in 25 seconds compared to 220 seconds for ground-only deployment. Complete implementation code is available at this https URL. 

**Abstract (ZH)**: 我们提出了一种全面的两层Voronoi覆盖控制方法，用于协调具有不同能力的空地机器人团队在危险材料应急响应场景中的协同工作。传统Voronoi覆盖控制方法在应急情境下面临三大关键限制：异质性代理具有显著不同的速度、初始部署配置的集群化以及紧急的时间约束要求快速响应而非最终收敛。我们的方法通过解耦的两层架构分别优化空中和地面机器人的定位，其中空中代理通过空投将地面传感器部署到高优先级位置。我们详细阐述了有界Voronoi单元计算、重要加权质心的有效数值积分技术和防滞困的鲁棒控制策略。仿真结果表明，响应时间减少了88%，在25秒内实现了目标传感器覆盖（初始传感器损失的18.5%），而仅地面部署需要220秒。完整的实施代码可在以下链接获得：this https URL。 

---
# Decoupled Scaling 4ch Bilateral Control on the Cartesian coordinate by 6-DoF Manipulator using Rotation Matrix 

**Title (ZH)**: 6自由度 manipulator 在笛卡尔坐标系中基于旋转矩阵的解耦四通道双侧控制缩放 

**Authors**: Koki Yamane, Sho Sakaino, Toshiaki Tsuji  

**Link**: [PDF](https://arxiv.org/pdf/2510.10545)  

**Abstract**: Four-channel bilateral control is a method for achieving remote control with force feedback and adjustment operability by synchronizing the positions and forces of two manipulators. This is expected to significantly improve the operability of the remote control in contact-rich tasks. Among these, 4-channel bilateral control on the Cartesian coordinate system is advantageous owing to its suitability for manipulators with different structures and because it allows the dynamics in the Cartesian coordinate system to be adjusted by adjusting the control parameters, thus achieving intuitive operability for humans. This paper proposes a 4-channel bilateral control method that achieves the desired dynamics by decoupling each dimension in the Cartesian coordinate system regardless of the scaling factor. 

**Abstract (ZH)**: 四通道双边控制是一种通过同步两个机器人操作器的位置和力实现远程控制并具有力反馈和调整操作性的方法。这种方法有望显著改善在接触密集任务中远程控制的操作性。其中，基于笛卡尔坐标系的四通道双边控制因其适用于不同结构的操作器，并可以通过调整控制参数来调整笛卡尔坐标系中的动力学，从而实现直观的人机操作性而具有优势。本文提出了一种四通道双边控制方法，通过解耦笛卡尔坐标系中的每个维度（不考虑比例因子）来实现期望的动力学。 

---
# Galilean Symmetry in Robotics 

**Title (ZH)**: 伽利略对称性在机器人学中的应用 

**Authors**: Robert Mahony, Jonathan Kelly, Stephan Weiss  

**Link**: [PDF](https://arxiv.org/pdf/2510.10468)  

**Abstract**: Galilean symmetry is the natural symmetry of inertial motion that underpins Newtonian physics. Although rigid-body symmetry is one of the most established and fundamental tools in robotics, there appears to be no comparable treatment of Galilean symmetry for a robotics audience. In this paper, we present a robotics-tailored exposition of Galilean symmetry that leverages the community's familiarity with and understanding of rigid-body transformations and pose representations. Our approach contrasts with common treatments in the physics literature that introduce Galilean symmetry as a stepping stone to Einstein's relativity. A key insight is that the Galilean matrix Lie group can be used to describe two different pose representations, Galilean frames, that use inertial velocity in the state definition, and extended poses, that use coordinate velocity. We provide three examples where applying the Galilean matrix Lie-group algebra to robotics problems is straightforward and yields significant insights: inertial navigation above the rotating Earth, manipulator kinematics, and sensor data fusion under temporal uncertainty. We believe that the time is right for the robotics community to benefit from rediscovering and extending this classical material and applying it to modern problems. 

**Abstract (ZH)**: 伽利略对称性是支撑牛顿物理学的惯性运动的自然对称性。尽管刚体对称性是机器人学中最成熟和基础的工具之一，但似乎还没有针对机器人学受众的伽利略对称性相似处理。本文为机器人学受众呈现了一种基于对刚体变换和姿态度量熟悉性的伽利略对称性的解释方法。我们的方法与物理学文献中常见的将伽利略对称性作为爱因斯坦相对论过渡工具的介绍不同。关键见解是伽利略矩阵李群可以用来描述两种不同的姿态度量：伽利略框架和扩展姿态度量，它们分别在状态定义中使用惯性速度和坐标速度。我们提供了三个例子，说明将伽利略矩阵李群代数应用于机器人学问题不仅直观而且能得出重要见解：在旋转地球上方的惯性导航、 manipulator 机械学以及基于时间不确定性下的传感器数据融合。我们认为，是时候让机器人学社区从重新发现和扩展这一经典材料并将其应用于现代问题中获益了。 

---
# MicroRoboScope: A Portable and Integrated Mechatronic Platform for Magnetic and Acoustic Microrobotic Experimentation 

**Title (ZH)**: MicroRoboScope：便携式集成机电平台，用于磁性和声学微机器人实验研究 

**Authors**: Max Sokolich, Yanda Yang, Subrahmanyam Cherukumilli, Fatma Ceren Kirmizitas, Sambeeta Das  

**Link**: [PDF](https://arxiv.org/pdf/2510.10392)  

**Abstract**: This paper presents MicroRoboScope, a portable, compact, and versatile microrobotic experimentation platform designed for real-time, closed-loop control of both magnetic and acoustic microrobots. The system integrates an embedded computer, microscope, power supplies, and control circuitry into a single, low-cost and fully integrated apparatus. Custom control software developed in Python and Arduino C++ handles live video acquisition, microrobot tracking, and generation of control signals for electromagnetic coils and acoustic transducers. The platform's multi-modal actuation, accessibility, and portability make it suitable not only for specialized research laboratories but also for educational and outreach settings. By lowering the barrier to entry for microrobotic experimentation, this system enables new opportunities for research, education, and translational applications in biomedicine, tissue engineering, and robotics. 

**Abstract (ZH)**: MicroRoboScope：一款便携式、紧凑且多功能的微机器人实验平台，用于实时、闭环控制磁性和声学微机器人 

---
# Integration of the TIAGo Robot into Isaac Sim with Mecanum Drive Modeling and Learned S-Curve Velocity Profiles 

**Title (ZH)**: TIAGo 机器人在 Isaac Sim 中的集成：配备麦克斯韦驱动模型和学习得到的 S 曲线速度轮廓 

**Authors**: Vincent Schoenbach, Marvin Wiedemann, Raphael Memmesheimer, Malte Mosbach, Sven Behnke  

**Link**: [PDF](https://arxiv.org/pdf/2510.10273)  

**Abstract**: Efficient physics simulation has significantly accelerated research progress in robotics applications such as grasping and assembly. The advent of GPU-accelerated simulation frameworks like Isaac Sim has particularly empowered learning-based methods, enabling them to tackle increasingly complex tasks. The PAL Robotics TIAGo++ Omni is a versatile mobile manipulator equipped with a mecanum-wheeled base, allowing omnidirectional movement and a wide range of task capabilities. However, until now, no model of the robot has been available in Isaac Sim. In this paper, we introduce such a model, calibrated to approximate the behavior of the real robot, with a focus on its omnidirectional drive dynamics. We present two control models for the omnidirectional drive: a physically accurate model that replicates real-world wheel dynamics and a lightweight velocity-based model optimized for learning-based applications. With these models, we introduce a learning-based calibration approach to approximate the real robot's S-shaped velocity profile using minimal trajectory data recordings. This simulation should allow researchers to experiment with the robot and perform efficient learning-based control in diverse environments. We provide the integration publicly at this https URL. 

**Abstract (ZH)**: 高效的物理模拟显著加速了机器人应用如抓取和装配的研究进展。加速器GPU驱动的仿真框架Isaac Sim的出现尤其增强了基于学习的方法，使其能够应对更为复杂的任务。PAL Robotics TIAGo++ Omni是一款多功能的移动执行器，配备多向轮基座，允许全方位移动和广泛的作业能力。然而，直到现在，Isaac Sim中尚无该机器人的模型。本文介绍了一个校准过的机器人模型，旨在近似其真实行为，重点关注其全方位驱动动态。我们介绍了两种全方位驱动控制模型：一种物理上精确的模型，能够复制现实世界的车轮动力学；一种轻量级的速度为基础的模型，针对基于学习的应用进行了优化。通过这些模型，我们提出了一种基于学习的校准方法，使用最少的轨迹数据记录来近似真实机器人S形的速度曲线。该仿真应允许研究人员对该机器人进行实验，并在各种环境中进行高效的基于学习的控制。我们已在此公开提供了该模型的集成：https://this-url。 

---
# LOMORO: Long-term Monitoring of Dynamic Targets with Minimum Robotic Fleet under Resource Constraints 

**Title (ZH)**: LOMORO: 在资源约束条件下最小化机器人车队进行动态目标长期监测 

**Authors**: Mingke Lu, Shuaikang Wang, Meng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2510.10046)  

**Abstract**: Long-term monitoring of numerous dynamic targets can be tedious for a human operator and infeasible for a single robot, e.g., to monitor wild flocks, detect intruders, search and rescue. Fleets of autonomous robots can be effective by acting collaboratively and concurrently. However, the online coordination is challenging due to the unknown behaviors of the targets and the limited perception of each robot. Existing work often deploys all robots available without minimizing the fleet size, or neglects the constraints on their resources such as battery and memory. This work proposes an online coordination scheme called LOMORO for collaborative target monitoring, path routing and resource charging. It includes three core components: (I) the modeling of multi-robot task assignment problem under the constraints on resources and monitoring intervals; (II) the resource-aware task coordination algorithm iterates between the high-level assignment of dynamic targets and the low-level multi-objective routing via the Martin's algorithm; (III) the online adaptation algorithm in case of unpredictable target behaviors and robot failures. It ensures the explicitly upper-bounded monitoring intervals for all targets and the lower-bounded resource levels for all robots, while minimizing the average number of active robots. The proposed methods are validated extensively via large-scale simulations against several baselines, under different road networks, robot velocities, charging rates and monitoring intervals. 

**Abstract (ZH)**: 基于在线协调的多机器人动态目标协同监测路径规划与资源充电方法 

---
# Hybrid Robotic Meta-gripper for Tomato Harvesting: Analysis of Auxetic Structures with Lattice Orientation Variations 

**Title (ZH)**: 具有 lattice 方向变异的Auxetic结构混合机器人Meta-gripper在番茄采摘中的分析 

**Authors**: Shahid Ansari, Vivek Gupta, Bishakh Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2510.10016)  

**Abstract**: The agricultural sector is rapidly evolving to meet growing global food demands, yet tasks like fruit and vegetable handling remain labor-intensive, causing inefficiencies and post-harvest losses. Automation, particularly selective harvesting, offers a viable solution, with soft robotics emerging as a key enabler. This study introduces a novel hybrid gripper for tomato harvesting, incorporating a rigid outer frame with a soft auxetic internal lattice. The six-finger, 3D caging-effect design enables gentle yet secure grasping in unstructured environments. Uniquely, the work investigates the effect of auxetic lattice orientation on grasping conformability, combining experimental validation with 2D Digital Image Correlation (DIC) and nonlinear finite element analysis (FEA). Auxetic configurations with unit cell inclinations of 0 deg, 30 deg, 45 deg, and 60 deg are evaluated, and their grasping forces, deformation responses, and motor torque requirements are systematically compared. Results demonstrate that lattice orientation strongly influences compliance, contact forces, and energy efficiency, with distinct advantages across configurations. This comparative framework highlights the novelty of tailoring auxetic geometries to optimize robotic gripper performance. The findings provide new insights into soft-rigid hybrid gripper design, advancing automation strategies for precision agriculture while minimizing crop damage. 

**Abstract (ZH)**: 一种新颖的番茄采摘混合夹持器及其异质 lattice 结构对抓取柔顺性的影响研究 

---
# FORM: Fixed-Lag Odometry with Reparative Mapping utilizing Rotating LiDAR Sensors 

**Title (ZH)**: 固定滞后_odometry_with_修正性_mapping_利用旋转LiDAR传感器 

**Authors**: Easton R. Potokar, Taylor Pool, Daniel McGann, Michael Kaess  

**Link**: [PDF](https://arxiv.org/pdf/2510.09966)  

**Abstract**: Light Detection and Ranging (LiDAR) sensors have become a de-facto sensor for many robot state estimation tasks, spurring development of many LiDAR Odometry (LO) methods in recent years. While some smoothing-based LO methods have been proposed, most require matching against multiple scans, resulting in sub-real-time performance. Due to this, most prior works estimate a single state at a time and are ``submap''-based. This architecture propagates any error in pose estimation to the fixed submap and can cause jittery trajectories and degrade future registrations. We propose Fixed-Lag Odometry with Reparative Mapping (FORM), a LO method that performs smoothing over a densely connected factor graph while utilizing a single iterative map for matching. This allows for both real-time performance and active correction of the local map as pose estimates are further refined. We evaluate on a wide variety of datasets to show that FORM is robust, accurate, real-time, and provides smooth trajectory estimates when compared to prior state-of-the-art LO methods. 

**Abstract (ZH)**: 基于约束图的固定滞后里程计与修复制图方法 

---
# Context-Aware Model-Based Reinforcement Learning for Autonomous Racing 

**Title (ZH)**: 基于上下文的模型导向强化学习在自主赛车中的应用 

**Authors**: Emran Yasser Moustafa, Ivana Dusparic  

**Link**: [PDF](https://arxiv.org/pdf/2510.11501)  

**Abstract**: Autonomous vehicles have shown promising potential to be a groundbreaking technology for improving the safety of road users. For these vehicles, as well as many other safety-critical robotic technologies, to be deployed in real-world applications, we require algorithms that can generalize well to unseen scenarios and data. Model-based reinforcement learning algorithms (MBRL) have demonstrated state-of-the-art performance and data efficiency across a diverse set of domains. However, these algorithms have also shown susceptibility to changes in the environment and its transition dynamics.
In this work, we explore the performance and generalization capabilities of MBRL algorithms for autonomous driving, specifically in the simulated autonomous racing environment, Roboracer (formerly F1Tenth). We frame the head-to-head racing task as a learning problem using contextual Markov decision processes and parameterize the driving behavior of the adversaries using the context of the episode, thereby also parameterizing the transition and reward dynamics. We benchmark the behavior of MBRL algorithms in this environment and propose a novel context-aware extension of the existing literature, cMask. We demonstrate that context-aware MBRL algorithms generalize better to out-of-distribution adversary behaviors relative to context-free approaches. We also demonstrate that cMask displays strong generalization capabilities, as well as further performance improvement relative to other context-aware MBRL approaches when racing against adversaries with in-distribution behaviors. 

**Abstract (ZH)**: 自主车辆在提高道路用户安全方面展示了突破性的潜力。为了使这些车辆以及许多其他关键安全机器人技术能够在实际应用中部署，我们要求使用能够良好泛化到未见过的场景和数据集中的算法。基于模型的强化学习算法（MBRL）在多种领域中表现出最先进的性能和数据效率。然而，这些算法也显示出对环境变化及其转移动力学的敏感性。在本研究中，我们探讨了MBRL算法在自主驾驶中的性能和泛化能力，特别是在模拟自主赛车环境Roboracer（原F1Tenth）中的表现。我们将一对一赛车任务表述为基于上下文的马尔可夫决策过程的学习问题，并使用当前段落的上下文来参数化对手的驾驶行为，从而也参数化转移和奖励动力学。我们在该环境中测试了MBRL算法的行为，并提出了一种上下文感知的扩展方法cMask。我们证明，上下文感知的MBRL算法在面对未见过的对手行为时具有更好的泛化能力，相比非上下文感知的方法。此外，我们还展示了cMask在面对分布内行为对手时表现出强泛化能力和相对于其他上下文感知的MBRL方法进一步的性能提升。 

---
# Computing Safe Control Inputs using Discrete-Time Matrix Control Barrier Functions via Convex Optimization 

**Title (ZH)**: 使用凸优化计算离散时间矩阵控制屏障函数的安全控制输入 

**Authors**: James Usevitch, Juan Augusto Paredes Salazar, Ankit Goel  

**Link**: [PDF](https://arxiv.org/pdf/2510.09925)  

**Abstract**: Control barrier functions (CBFs) have seen widespread success in providing forward invariance and safety guarantees for dynamical control systems. A crucial limitation of discrete-time formulations is that CBFs that are nonconcave in their argument require the solution of nonconvex optimization problems to compute safety-preserving control inputs, which inhibits real-time computation of control inputs guaranteeing forward invariance. This paper presents a novel method for computing safety-preserving control inputs for discrete-time systems with nonconvex safety sets, utilizing convex optimization and the recently developed class of matrix control barrier function techniques. The efficacy of our methods is demonstrated through numerical simulations on a bicopter system. 

**Abstract (ZH)**: 非凸安全集下离散时间系统的安全保持控制输入计算方法：利用凸优化和矩阵控制障碍函数技术 

---
