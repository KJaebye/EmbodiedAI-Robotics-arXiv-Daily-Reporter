# Adaptive Fault-tolerant Control of Underwater Vehicles with Thruster Failures 

**Title (ZH)**: 基于推进器故障自适应容错控制的水下车辆控制 

**Authors**: Haolin Liu, Shiliang Zhang, Shangbin Jiao, Xiaohui Zhang, Xuehui Ma, Yan Yan, Wenchuan Cui, Youmin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.16037)  

**Abstract**: This paper presents a fault-tolerant control for the trajectory tracking of autonomous underwater vehicles (AUVs) against thruster failures. We formulate faults in AUV thrusters as discrete switching events during a UAV mission, and develop a soft-switching approach in facilitating shift of control strategies across fault scenarios. We mathematically define AUV thruster fault scenarios, and develop the fault-tolerant control that captures the fault scenario via Bayesian approach. Particularly, when the AUV fault type switches from one to another, the developed control captures the fault states and maintains the control by a linear quadratic tracking controller. With the captured fault states by Bayesian approach, we derive the control law by aggregating the control outputs for individual fault scenarios weighted by their Bayesian posterior probability. The developed fault-tolerant control works in an adaptive way and guarantees soft-switching across fault scenarios, and requires no complicated fault detection dedicated to different type of faults. The entailed soft-switching ensures stable AUV trajectory tracking when fault type shifts, which otherwise leads to reduced control under hard-switching control strategies. We conduct numerical simulations with diverse AUV thruster fault settings. The results demonstrate that the proposed control can provide smooth transition across thruster failures, and effectively sustain AUV trajectory tracking control in case of thruster failures and failure shifts. 

**Abstract (ZH)**: 针对推进器故障的自主水下车辆轨迹跟踪容错控制 

---
# Blimp-based Crime Scene Analysis 

**Title (ZH)**: 基于Blimp的犯罪现场分析 

**Authors**: Martin Cooney, Fernando Alonso-Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2504.15962)  

**Abstract**: To tackle the crucial problem of crime, evidence at indoor crime scenes must be analyzed before it becomes contaminated or degraded. Here, as an application of artificial intelligence (AI), computer vision, and robotics, we explore how a blimp could be designed as a kind of "floating camera" to drift over and record evidence with minimal disturbance. In particular, rapid prototyping is used to develop a proof-of-concept to gain insight into what such blimps could do, manually piloted or semi-autonomously. As a result, we show the feasibility of attaching various components to an indoor blimp, and confirm our basic premise, that blimps can sense evidence without producing much wind. Some additional suggestions--regarding mapping, sensing, and path-finding--aim to stimulate the flow of ideas for further exploration. 

**Abstract (ZH)**: 基于人工智能、计算机视觉和机器人技术的轻气球在室内犯罪现场勘查中的应用探索：从快速原型设计到概念验证 

---
# Embedded Safe Reactive Navigation for Multirotors Systems using Control Barrier Functions 

**Title (ZH)**: 使用控制屏障函数的多旋翼系统嵌入式安全反应导航 

**Authors**: Nazar Misyats, Marvin Harms, Morten Nissov, Martin Jacquet, Kostas Alexis  

**Link**: [PDF](https://arxiv.org/pdf/2504.15850)  

**Abstract**: Aiming to promote the wide adoption of safety filters for autonomous aerial robots, this paper presents a safe control architecture designed for seamless integration into widely used open-source autopilots. Departing from methods that require consistent localization and mapping, we formalize the obstacle avoidance problem as a composite control barrier function constructed only from the online onboard range measurements. The proposed framework acts as a safety filter, modifying the acceleration references derived by the nominal position/velocity control loops, and is integrated into the PX4 autopilot stack. Experimental studies using a small multirotor aerial robot demonstrate the effectiveness and performance of the solution within dynamic maneuvering and unknown environments. 

**Abstract (ZH)**: 旨在促进自主 aerial 机器人安全过滤器的广泛应用，本文提出了一种安全控制架构，该架构便于无缝集成到广泛使用的开源自动驾驶仪中。不同于需要一致定位和建图的方法，我们将避障问题形式化为仅基于在线机载范围测量构建的复合控制屏障函数。所提出的框架作为安全过滤器，在名义位置/速度控制环路获取的加速度参考上进行修改，并集成到PX4自动驾驶仪栈中。使用小型多旋翼 aerial 机器人进行的实验研究验证了该解决方案在动态机动和未知环境中的有效性和性能。 

---
# CaRoSaC: A Reinforcement Learning-Based Kinematic Control of Cable-Driven Parallel Robots by Addressing Cable Sag through Simulation 

**Title (ZH)**: CaRoSaC：一种通过仿真解决缆线松弛的基于强化学习的缆索驱动并联机器人运动控制方法 

**Authors**: Rohit Dhakate, Thomas Jantos, Eren Allak, Stephan Weiss, Jan Steinbrener  

**Link**: [PDF](https://arxiv.org/pdf/2504.15740)  

**Abstract**: This paper introduces the Cable Robot Simulation and Control (CaRoSaC) Framework, which integrates a simulation environment with a model-free reinforcement learning control methodology for suspended Cable-Driven Parallel Robots (CDPRs), accounting for cable sag. Our approach seeks to bridge the knowledge gap of the intricacies of CDPRs due to aspects such as cable sag and precision control necessities by establishing a simulation platform that captures the real-world behaviors of CDPRs, including the impacts of cable sag. The framework offers researchers and developers a tool to further develop estimation and control strategies within the simulation for understanding and predicting the performance nuances, especially in complex operations where cable sag can be significant. Using this simulation framework, we train a model-free control policy in Reinforcement Learning (RL). This approach is chosen for its capability to adaptively learn from the complex dynamics of CDPRs. The policy is trained to discern optimal cable control inputs, ensuring precise end-effector positioning. Unlike traditional feedback-based control methods, our RL control policy focuses on kinematic control and addresses the cable sag issues without being tethered to predefined mathematical models. We also demonstrate that our RL-based controller, coupled with the flexible cable simulation, significantly outperforms the classical kinematics approach, particularly in dynamic conditions and near the boundary regions of the workspace. The combined strength of the described simulation and control approach offers an effective solution in manipulating suspended CDPRs even at workspace boundary conditions where traditional approach fails, as proven from our experiments, ensuring that CDPRs function optimally in various applications while accounting for the often neglected but critical factor of cable sag. 

**Abstract (ZH)**: Cable Robot Simulation and Control (CaRoSaC)框架：考虑电缆下垂的悬空缆驱动并联机器人建模与控制方法 

---
# Autonomous Control of Redundant Hydraulic Manipulator Using Reinforcement Learning with Action Feedback 

**Title (ZH)**: 使用带有动作反馈的强化学习控制冗余液压 manipulator 的自主控制 

**Authors**: Rohit Dhakate, Christian Brommer, Christoph Böhm, Stephan Weiss, Jan Steinbrener  

**Link**: [PDF](https://arxiv.org/pdf/2504.15714)  

**Abstract**: This article presents an entirely data-driven approach for autonomous control of redundant manipulators with hydraulic actuation. The approach only requires minimal system information, which is inherited from a simulation model. The non-linear hydraulic actuation dynamics are modeled using actuator networks from the data gathered during the manual operation of the manipulator to effectively emulate the real system in a simulation environment. A neural network control policy for autonomous control, based on end-effector (EE) position tracking is then learned using Reinforcement Learning (RL) with Ornstein-Uhlenbeck process noise (OUNoise) for efficient exploration. The RL agent also receives feedback based on supervised learning of the forward kinematics which facilitates selecting the best suitable action from exploration. The control policy directly provides the joint variables as outputs based on provided target EE position while taking into account the system dynamics. The joint variables are then mapped to the hydraulic valve commands, which are then fed to the system without further modifications. The proposed approach is implemented on a scaled hydraulic forwarder crane with three revolute and one prismatic joint to track the desired position of the EE in 3-Dimensional (3D) space. With the emulated dynamics and extensive learning in simulation, the results demonstrate the feasibility of deploying the learned controller directly on the real system. 

**Abstract (ZH)**: 一种基于数据驱动的冗余液压 manipulator 自主控制方法 

---
# VibeCheck: Using Active Acoustic Tactile Sensing for Contact-Rich Manipulation 

**Title (ZH)**: VibeCheck: 使用主动声触觉传感进行高接触操作 

**Authors**: Kaidi Zhang, Do-Gon Kim, Eric T. Chang, Hua-Hsuan Liang, Zhanpeng He, Kathryn Lampo, Philippe Wu, Ioannis Kymissis, Matei Ciocarlie  

**Link**: [PDF](https://arxiv.org/pdf/2504.15535)  

**Abstract**: The acoustic response of an object can reveal a lot about its global state, for example its material properties or the extrinsic contacts it is making with the world. In this work, we build an active acoustic sensing gripper equipped with two piezoelectric fingers: one for generating signals, the other for receiving them. By sending an acoustic vibration from one finger to the other through an object, we gain insight into an object's acoustic properties and contact state. We use this system to classify objects, estimate grasping position, estimate poses of internal structures, and classify the types of extrinsic contacts an object is making with the environment. Using our contact type classification model, we tackle a standard long-horizon manipulation problem: peg insertion. We use a simple simulated transition model based on the performance of our sensor to train an imitation learning policy that is robust to imperfect predictions from the classifier. We finally demonstrate the policy on a UR5 robot with active acoustic sensing as the only feedback. 

**Abstract (ZH)**: 具有主动声学传感的抓手：通过声学响应揭示物体的全局状态与其接触状态 

---
# Field Report on Ground Penetrating Radar for Localization at the Mars Desert Research Station 

**Title (ZH)**: 火星沙漠研究站地下雷达定位现场报告 

**Authors**: Anja Sheppard, Katherine A. Skinner  

**Link**: [PDF](https://arxiv.org/pdf/2504.15455)  

**Abstract**: In this field report, we detail the lessons learned from our field expedition to collect Ground Penetrating Radar (GPR) data in a Mars analog environment for the purpose of validating GPR localization techniques in rugged environments. Planetary rovers are already equipped with GPR for geologic subsurface characterization. GPR has been successfully used to localize vehicles on Earth, but it has not yet been explored as another modality for localization on a planetary rover. Leveraging GPR for localization can aid in efficient and robust rover pose estimation. In order to demonstrate localizing GPR in a Mars analog environment, we collected over 50 individual survey trajectories during a two-week period at the Mars Desert Research Station (MDRS). In this report, we discuss our methodology, lessons learned, and opportunities for future work. 

**Abstract (ZH)**: 在本实地报告中，我们详细介绍了在火星模拟环境中采集地面穿透雷达（GPR）数据的实地探险经验，旨在验证GPR定位技术在崎岖环境中的有效性。行星探测车已经配备了GPR用于地质地下层表征。GPR在地球上已被成功用于车辆定位，但在行星探测车上作为另一种定位模式的应用尚未被探索。利用GPR进行定位可以辅助实现高效的稳健的探测车姿态估计。为了展示在火星模拟环境中定位GPR，我们在火星沙漠研究站（MDRS）为期两周的时间内收集了超过50个独立的调查轨迹。在本报告中，我们讨论了我们的方法论、经验教训以及未来工作的机会。 

---
# Efficient and Safe Planner for Automated Driving on Ramps Considering Unsatisfication 

**Title (ZH)**: 考虑不满意情况的匝道自动驾驶高效安全规划器 

**Authors**: Qinghao Li, Zhen Tian, Xiaodan Wang, Jinming Yang, Zhihao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15320)  

**Abstract**: Automated driving on ramps presents significant challenges due to the need to balance both safety and efficiency during lane changes. This paper proposes an integrated planner for automated vehicles (AVs) on ramps, utilizing an unsatisfactory level metric for efficiency and arrow-cluster-based sampling for safety. The planner identifies optimal times for the AV to change lanes, taking into account the vehicle's velocity as a key factor in efficiency. Additionally, the integrated planner employs arrow-cluster-based sampling to evaluate collision risks and select an optimal lane-changing curve. Extensive simulations were conducted in a ramp scenario to verify the planner's efficient and safe performance. The results demonstrate that the proposed planner can effectively select an appropriate lane-changing time point and a safe lane-changing curve for AVs, without incurring any collisions during the maneuver. 

**Abstract (ZH)**: 匝道上自动驾驶车辆的结合规划器：利用效率不足等级度量和箭头簇基于采样确保安全与效率并发 

---
# SLAM-Based Navigation and Fault Resilience in a Surveillance Quadcopter with Embedded Vision Systems 

**Title (ZH)**: 基于SLAM的监视四旋翼无人机嵌入式视觉系统导航与故障韧性研究 

**Authors**: Abhishek Tyagi, Charu Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2504.15305)  

**Abstract**: We present an autonomous aerial surveillance platform, Veg, designed as a fault-tolerant quadcopter system that integrates visual SLAM for GPS-independent navigation, advanced control architecture for dynamic stability, and embedded vision modules for real-time object and face recognition. The platform features a cascaded control design with an LQR inner-loop and PD outer-loop trajectory control. It leverages ORB-SLAM3 for 6-DoF localization and loop closure, and supports waypoint-based navigation through Dijkstra path planning over SLAM-derived maps. A real-time Failure Detection and Identification (FDI) system detects rotor faults and executes emergency landing through re-routing. The embedded vision system, based on a lightweight CNN and PCA, enables onboard object detection and face recognition with high precision. The drone operates fully onboard using a Raspberry Pi 4 and Arduino Nano, validated through simulations and real-world testing. This work consolidates real-time localization, fault recovery, and embedded AI on a single platform suitable for constrained environments. 

**Abstract (ZH)**: 一种基于视觉SLAM的鲁棒四旋翼自主空中 surveillance 平台：融合故障检测与容错的视觉导航与识别系统 

---
# An ACO-MPC Framework for Energy-Efficient and Collision-Free Path Planning in Autonomous Maritime Navigation 

**Title (ZH)**: 基于自主海洋导航的节能无碰撞路径规划的蚁群优化-模型预测控制框架 

**Authors**: Yaoze Liu, Zhen Tian, Qifan Zhou, Zixuan Huang, Hongyu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.15611)  

**Abstract**: Automated driving on ramps presents significant challenges due to the need to balance both safety and efficiency during lane changes. This paper proposes an integrated planner for automated vehicles (AVs) on ramps, utilizing an unsatisfactory level metric for efficiency and arrow-cluster-based sampling for safety. The planner identifies optimal times for the AV to change lanes, taking into account the vehicle's velocity as a key factor in efficiency. Additionally, the integrated planner employs arrow-cluster-based sampling to evaluate collision risks and select an optimal lane-changing curve. Extensive simulations were conducted in a ramp scenario to verify the planner's efficient and safe performance. The results demonstrate that the proposed planner can effectively select an appropriate lane-changing time point and a safe lane-changing curve for AVs, without incurring any collisions during the maneuver. 

**Abstract (ZH)**: 自动化车辆在匝道上的车道变更规划：综合效率不满足度度量与箭头簇采样的集成规划 

---
