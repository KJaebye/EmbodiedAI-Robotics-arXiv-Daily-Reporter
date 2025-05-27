# URPlanner: A Universal Paradigm For Collision-Free Robotic Motion Planning Based on Deep Reinforcement Learning 

**Title (ZH)**: URPlanner: 一种基于深度强化学习的通用无碰撞机器人运动规划范式 

**Authors**: Fengkang Ying, Hanwen Zhang, Haozhe Wang, Huishi Huang, Marcelo H. Ang Jr  

**Link**: [PDF](https://arxiv.org/pdf/2505.20175)  

**Abstract**: Collision-free motion planning for redundant robot manipulators in complex environments is yet to be explored. Although recent advancements at the intersection of deep reinforcement learning (DRL) and robotics have highlighted its potential to handle versatile robotic tasks, current DRL-based collision-free motion planners for manipulators are highly costly, hindering their deployment and application. This is due to an overreliance on the minimum distance between the manipulator and obstacles, inadequate exploration and decision-making by DRL, and inefficient data acquisition and utilization. In this article, we propose URPlanner, a universal paradigm for collision-free robotic motion planning based on DRL. URPlanner offers several advantages over existing approaches: it is platform-agnostic, cost-effective in both training and deployment, and applicable to arbitrary manipulators without solving inverse kinematics. To achieve this, we first develop a parameterized task space and a universal obstacle avoidance reward that is independent of minimum distance. Second, we introduce an augmented policy exploration and evaluation algorithm that can be applied to various DRL algorithms to enhance their performance. Third, we propose an expert data diffusion strategy for efficient policy learning, which can produce a large-scale trajectory dataset from only a few expert demonstrations. Finally, the superiority of the proposed methods is comprehensively verified through experiments. 

**Abstract (ZH)**: 冗余机器人 manipulator 在复杂环境下的无碰撞运动规划尚未被充分探索。尽管深度强化学习（DRL）与机器人技术的交叉领域近期突显了其处理多种机器人任务的潜力，但当前基于 DRL 的无碰撞运动规划器成本高昂，阻碍了其部署和应用。这主要是因为过于依赖 manipulator 与障碍物之间的最小距离、DRL 资源探索和决策不足以及数据获取和利用效率低下。本文提出了一种基于 DRL 的通用无碰撞机器人运动规划框架 URPlanner。URPlanner 具有以下优势：平台无关、训练和部署成本低，并且能够应用于任意 manipulator 而无需求解逆运动学。为此，我们首先开发了一个参数化任务空间及相关联的通用障碍物回避奖励，该奖励独立于最小距离。其次，我们引入了一种增强策略探索和评估算法，该算法可应用于各种 DRL 算法以增强其性能。第三，我们提出了一种专家数据扩散策略，以实现高效策略学习，该策略可以从少量专家演示生成大规模轨迹数据集。最后，通过实验全面验证了所提方法的优越性。 

---
# Target Tracking via LiDAR-RADAR Sensor Fusion for Autonomous Racing 

**Title (ZH)**: 基于LiDAR-RADAR传感器融合的自主赛车目标跟踪 

**Authors**: Marcello Cellina, Matteo Corno, Sergio Matteo Savaresi  

**Link**: [PDF](https://arxiv.org/pdf/2505.20043)  

**Abstract**: High Speed multi-vehicle Autonomous Racing will increase the safety and performance of road-going Autonomous Vehicles. Precise vehicle detection and dynamics estimation from a moving platform is a key requirement for planning and executing complex autonomous overtaking maneuvers. To address this requirement, we have developed a Latency-Aware EKF-based Multi Target Tracking algorithm fusing LiDAR and RADAR measurements. The algorithm explots the different sensor characteristics by explicitly integrating the Range Rate in the EKF Measurement Function, as well as a-priori knowledge of the racetrack during state prediction. It can handle Out-Of-Sequence Measurements via Reprocessing using a double State and Measurement Buffer, ensuring sensor delay compensation with no information loss. This algorithm has been implemented on Team PoliMOVE's autonomous racecar, and was proved experimentally by completing a number of fully autonomous overtaking maneuvers at speeds up to 275 km/h. 

**Abstract (ZH)**: 高速多车自主赛车比赛将提高道路上自主车辆的安全性和性能。基于LiDAR和RADAR测量的延迟感知EKF多目标跟踪算法对于规划和执行复杂的自主超车机动至关重要。 

---
# A Cooperative Aerial System of A Payload Drone Equipped with Dexterous Rappelling End Droid for Cluttered Space Pickup 

**Title (ZH)**: 装备有灵巧绳降末端执行器的载荷无人机协同空中系统在复杂空间拾取操作 

**Authors**: Wenjing Ren, Xin Dong, Yangjie Cui, Binqi Yang, Haoze Li, Tao Yu, Jinwu Xiang, Daochun Li, Zhan Tu  

**Link**: [PDF](https://arxiv.org/pdf/2505.19980)  

**Abstract**: In cluttered spaces, such as forests, drone picking up a payload via an abseil claw is an open challenge, as the cable is likely tangled and blocked by the branches and obstacles. To address such a challenge, in this work, a cooperative aerial system is proposed, which consists of a payload drone and a dexterous rappelling end droid. The two ends are linked via a Kevlar tether cable. The end droid is actuated by four propellers, which enable mid-air dexterous adjustment of clawing angle and guidance of cable movement. To avoid tanglement and rappelling obstacles, a trajectory optimization method that integrates cable length constraints and dynamic feasibility is developed, which guarantees safe pickup. A tether cable dynamic model is established to evaluate real-time cable status, considering both taut and sagging conditions. Simulation and real-world experiments are conducted to demonstrate that the proposed system is capable of picking up payload in cluttered spaces. As a result, the end droid can reach the target point successfully under cable constraints and achieve passive retrieval during the lifting phase without propulsion, which enables effective and efficient aerial manipulation. 

**Abstract (ZH)**: 在 cluttered 空间中，如森林中，无人机通过悬降爪拾取载荷是一个开放性挑战，因为缆绳容易与树枝和障碍物缠绕和堵塞。为应对这一挑战，本工作提出了一种协同空中系统，该系统由一个载荷无人机和一个灵巧悬降末端机器人组成。两端通过凯夫拉绳索连接。末端机器人由四个推进器驱动，能够在空中灵活调整爪子的角度和引导缆绳的移动。为避免缠绕和悬降障碍，开发了一种结合缆绳长度约束和动态可行性的路径优化方法，以保证安全拾取。建立了一种缆绳动力学模型，以实时刻缆状态评估，考虑了绷紧和下垂两种情况。通过仿真和实际实验展示了所提出系统能够在 cluttered 空间中拾取载荷的能力，结果表明，在缆绳限制条件下，末端机器人可以成功到达目标点，并在提升阶段实现无推进的被动拾取，从而实现有效的空中操作。 

---
# GeoPF: Infusing Geometry into Potential Fields for Reactive Planning in Non-trivial Environments 

**Title (ZH)**: GeoPF：将几何学融入潜在场以在复杂环境中进行反应性规划 

**Authors**: Yuhe Gong, Riddhiman Laha, Luis Figueredo  

**Link**: [PDF](https://arxiv.org/pdf/2505.19688)  

**Abstract**: Reactive intelligence remains one of the cornerstones of versatile robotics operating in cluttered, dynamic, and human-centred environments. Among reactive approaches, potential fields (PF) continue to be widely adopted due to their simplicity and real-time applicability. However, existing PF methods typically oversimplify environmental representations by relying on isotropic, point- or sphere-based obstacle approximations. In human-centred settings, this simplification results in overly conservative paths, cumbersome tuning, and computational overhead -- even breaking real-time requirements. In response, we propose the Geometric Potential Field (GeoPF), a reactive motion-planning framework that explicitly infuses geometric primitives - points, lines, planes, cubes, and cylinders - into real-time planning. By leveraging precise closed-form distance functions, GeoPF significantly reduces computational complexity and parameter tuning effort. Extensive quantitative analyses consistently show GeoPF's higher success rates, reduced tuning complexity (a single parameter set across experiments), and substantially lower computational costs (up to 2 orders of magnitude) compared to traditional PF methods. Real-world experiments further validate GeoPF's robustness and practical ease of deployment. GeoPF provides a fresh perspective on reactive planning problems driving geometric-aware temporal motion generation, enabling flexible and low-latency motion planning suitable for modern robotic applications. 

**Abstract (ZH)**: 几何势场：一种面向几何感知的实时motion规划框架 

---
# Autonomous Flights inside Narrow Tunnels 

**Title (ZH)**: 自主窄隧道内飞行 

**Authors**: Luqi Wang, Yan Ning, Hongming Chen, Peize Liu, Yang Xu, Hao Xu, Ximin Lyu, Shaojie Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.19657)  

**Abstract**: Multirotors are usually desired to enter confined narrow tunnels that are barely accessible to humans in various applications including inspection, search and rescue, and so on. This task is extremely challenging since the lack of geometric features and illuminations, together with the limited field of view, cause problems in perception; the restricted space and significant ego airflow disturbances induce control issues. This paper introduces an autonomous aerial system designed for navigation through tunnels as narrow as 0.5 m in diameter. The real-time and online system includes a virtual omni-directional perception module tailored for the mission and a novel motion planner that incorporates perception and ego airflow disturbance factors modeled using camera projections and computational fluid dynamics analyses, respectively. Extensive flight experiments on a custom-designed quadrotor are conducted in multiple realistic narrow tunnels to validate the superior performance of the system, even over human pilots, proving its potential for real applications. Additionally, a deployment pipeline on other multirotor platforms is outlined and open-source packages are provided for future developments. 

**Abstract (ZH)**: 多旋翼无人机在狭窄隧道中的自主导航：从0.5米直径的隧道出发 

---
# Indoor Air Quality Detection Robot Model Based on the Internet of Things (IoT) 

**Title (ZH)**: 基于物联网(IoT)的室内空气质量检测机器人模型 

**Authors**: Anggiat Mora Simamora, Asep Denih, Mohamad Iqbal Suriansyah  

**Link**: [PDF](https://arxiv.org/pdf/2505.19600)  

**Abstract**: This paper presents the design, implementation, and evaluation of an IoT-based robotic system for mapping and monitoring indoor air quality. The primary objective was to develop a mobile robot capable of autonomously mapping a closed environment, detecting concentrations of CO$_2$, volatile organic compounds (VOCs), smoke, temperature, and humidity, and transmitting real-time data to a web interface. The system integrates a set of sensors (SGP30, MQ-2, DHT11, VL53L0X, MPU6050) with an ESP32 microcontroller. It employs a mapping algorithm for spatial data acquisition and utilizes a Mamdani fuzzy logic system for air quality classification. Empirical tests in a model room demonstrated average localization errors below $5\%$, actuator motion errors under $2\%$, and sensor measurement errors within $12\%$ across all modalities. The contributions of this work include: (1) a low-cost, integrated IoT robotic platform for simultaneous mapping and air quality detection; (2) a web-based user interface for real-time visualization and control; and (3) validation of system accuracy under laboratory conditions. 

**Abstract (ZH)**: 基于物联网的室内空气质量测绘与监控机器人系统的设计、实现与评估 

---
# LLA-MPC: Fast Adaptive Control for Autonomous Racing 

**Title (ZH)**: LLA-MPC: 快速自适应控制用于自主赛车 

**Authors**: Maitham F. AL-Sunni, Hassan Almubarak, Katherine Horng, John M. Dolan  

**Link**: [PDF](https://arxiv.org/pdf/2505.19512)  

**Abstract**: We present Look-Back and Look-Ahead Adaptive Model Predictive Control (LLA-MPC), a real-time adaptive control framework for autonomous racing that addresses the challenge of rapidly changing tire-surface interactions. Unlike existing approaches requiring substantial data collection or offline training, LLA-MPC employs a model bank for immediate adaptation without a learning period. It integrates two key mechanisms: a look-back window that evaluates recent vehicle behavior to select the most accurate model and a look-ahead horizon that optimizes trajectory planning based on the identified dynamics. The selected model and estimated friction coefficient are then incorporated into a trajectory planner to optimize reference paths in real-time. Experiments across diverse racing scenarios demonstrate that LLA-MPC outperforms state-of-the-art methods in adaptation speed and handling, even during sudden friction transitions. Its learning-free, computationally efficient design enables rapid adaptation, making it ideal for high-speed autonomous racing in multi-surface environments. 

**Abstract (ZH)**: 回顾前瞻自适应模型预测控制（LLA-MPC）：一种用于自主赛车的实时自适应控制框架 

---
# Passive Vibration Control of a 3-D Printer Gantry 

**Title (ZH)**: 3D打印龙门架的被动振动控制 

**Authors**: Maharshi A. Sharma, Albert E. Patterson  

**Link**: [PDF](https://arxiv.org/pdf/2505.19311)  

**Abstract**: Improved additive manufacturing capabilities are vital for the future development and improvement of ubiquitous robotic systems. These machines can be integrated into existing robotic systems to allow manufacturing and repair of components, as well as fabrication of custom parts for the robots themselves. The fused filament fabrication (FFF) process is one of the most common and well-developed AM processes but suffers from the effects of vibration-induced position error, particularly as the printing speed is raised. This project adapted and expanded a dynamic model of an FFF gantry system to include a passive spring-mass-damper system controller attached to the extruder carriage and tuned using optimal parameters. A case study was conducted to demonstrate the effects and generate recommendations for implementation. This work is also valuable for other mechatronic systems which operate using an open-loop control system and which suffer from vibration, including numerous robotic systems, pick-and-place machines, positioners, and similar. 

**Abstract (ZH)**: 改进的增材制造能力对于未来通用机器人系统的开发与改进至关重要。这些机器可以整合到现有的机器人系统中，以实现组件的制造与修复，以及为机器人自身制作定制部件。熔融沉积成型（FFF）工艺是应用最广泛和最成熟的增材制造工艺之一，但随着打印速度的提高，会受到振动引起的定位误差的影响。本项目改编并扩展了一个FFF龙门系统的动态模型，包括将一个被动弹簧-质量-阻尼系统控制器附加到挤出机车架上，并通过最优参数进行调整。进行了案例研究以展示其效果并提出实施建议。此项工作对于其他采用开环控制系统且受振动影响的机电系统也具有重要意义，包括多种机器人系统、取放机、定位器等。 

---
# Designing Pin-pression Gripper and Learning its Dexterous Grasping with Online In-hand Adjustment 

**Title (ZH)**: 设计针压式夹持器并实现其灵巧抓取的在线手内调整方法 

**Authors**: Hewen Xiao, Xiuping Liu, Hang Zhao, Jian Liu, Kai Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18994)  

**Abstract**: We introduce a novel design of parallel-jaw grippers drawing inspiration from pin-pression toys. The proposed pin-pression gripper features a distinctive mechanism in which each finger integrates a 2D array of pins capable of independent extension and retraction. This unique design allows the gripper to instantaneously customize its finger's shape to conform to the object being grasped by dynamically adjusting the extension/retraction of the pins. In addition, the gripper excels in in-hand re-orientation of objects for enhanced grasping stability again via dynamically adjusting the pins. To learn the dynamic grasping skills of pin-pression grippers, we devise a dedicated reinforcement learning algorithm with careful designs of state representation and reward shaping. To achieve a more efficient grasp-while-lift grasping mode, we propose a curriculum learning scheme. Extensive evaluations demonstrate that our design, together with the learned skills, leads to highly flexible and robust grasping with much stronger generality to unseen objects than alternatives. We also highlight encouraging physical results of sim-to-real transfer on a physically manufactured pin-pression gripper, demonstrating the practical significance of our novel gripper design and grasping skill. Demonstration videos for this paper are available at this https URL. 

**Abstract (ZH)**: 基于 pin-pression 玩具启发的并指夹持器新型设计及动态抓取技能学习 

---
# Mobile Manipulation Planning for Tabletop Rearrangement 

**Title (ZH)**: 桌面重排中的移动 manipulate 计划 

**Authors**: Jiaming Hu, Jiawei Wang, Henrik I Christensen  

**Link**: [PDF](https://arxiv.org/pdf/2505.18732)  

**Abstract**: Efficient tabletop rearrangement planning seeks to find high-quality solutions while minimizing total cost. However, the task is challenging due to object dependencies and limited buffer space for temporary placements. The complexity increases for mobile robots, which must navigate around the table with restricted access. A*-based methods yield high-quality solutions, but struggle to scale as the number of objects increases. Monte Carlo Tree Search (MCTS) has been introduced as an anytime algorithm, but its convergence speed to high-quality solutions remains slow. Previous work~\cite{strap2024} accelerated convergence but required the robot to move to the closest position to the object for each pick and place operation, leading to inefficiencies. To address these limitations, we extend the planner by introducing a more efficient strategy for mobile robots. Instead of selecting the nearest available location for each action, our approach allows multiple operations (e.g., pick-and-place) from a single standing position, reducing unnecessary movement. Additionally, we incorporate state re-exploration to further improve plan quality. Experimental results show that our planner outperforms existing planners both in terms of solution quality and planning time. 

**Abstract (ZH)**: 桌面布局高效规划旨在找到高质量解决方案的同时最小化总成本。但由于对象依赖性和有限的临时放置缓冲空间，该任务极具挑战性。对于移动机器人来说，任务更为复杂，因为它们必须在有限的访问范围内导航。基于A*的方法能够提供高质量的解决方案，但随着物体数量的增加，难以扩展。蒙特卡洛树搜索（MCTS）作为一种随时可用的算法被引入，但其收敛到高质量解决方案的速度仍然较慢。先前的工作~\cite{strap2024}加速了收敛速度，但要求机器人在每次抓取和放置操作时移动到物体的最接近位置，导致效率低下。为解决这些限制，我们通过引入一种更高效的移动机器人策略来扩展规划器。我们的方法允许从单个站立位置执行多个操作（例如抓取-放置），从而减少不必要的移动。此外，我们还引入了状态重新探索，以进一步提高规划质量。实验结果表明，我们的规划器在解决方案质量和规划时间方面均优于现有规划器。 

---
# Coordinated guidance and control for multiple parafoil system landing 

**Title (ZH)**: 多伞系统着陆的协同指导与控制 

**Authors**: Zhenyu Wei, Zhijiang Shao, Lorenz T. Biegler  

**Link**: [PDF](https://arxiv.org/pdf/2505.18691)  

**Abstract**: Multiple parafoil landing is an enabling technology for massive supply delivery missions. However, it is still an open question to design a collision-free, computation-efficient guidance and control method for unpowered parafoils. To address this issue, this paper proposes a coordinated guidance and control method for multiple parafoil landing. First, the multiple parafoil landing process is formulated as a trajectory optimization problem. Then, the landing point allocation algorithm is designed to assign the landing point to each parafoil. In order to guarantee flight safety, the collision-free trajectory replanning algorithm is designed. On this basis, the nonlinear model predictive control algorithm is adapted to leverage the nonlinear dynamics model for trajectory tracking. Finally, the parafoil kinematic model is utilized to reduce the computational burden of trajectory calculation, and kinematic model is updated by the moving horizon correction algorithm to improve the trajectory accuracy. Simulation results demonstrate the effectiveness and computational efficiency of the proposed coordinated guidance and control method for the multiple parafoil landing. 

**Abstract (ZH)**: 多副伞降落协调引导与控制方法研究 

---
# Optimization-Based Trajectory Planning for Tractor-Trailer Vehicles on Curvy Roads: A Progressively Increasing Sampling Number Method 

**Title (ZH)**: 基于优化的曲线路段铰接车辆轨迹规划：递增采样数量方法 

**Authors**: Zehao Wang, Han Zhang, Jingchuan Wang, Weidong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.18590)  

**Abstract**: In this work, we propose an optimization-based trajectory planner for tractor-trailer vehicles on curvy roads. The lack of analytical expression for the trailer's errors to the center line pose a great challenge to the trajectory planning for tractor-trailer vehicles. To address this issue, we first use geometric representations to characterize the lateral and orientation errors in Cartesian frame, where the errors would serve as the components of the cost function and the road edge constraints within our optimization process. Next, we generate a coarse trajectory to warm-start the subsequent optimization problems. On the other hand, to achieve a good approximation of the continuous-time kinematics, optimization-based methods usually discretize the kinematics with a large sampling number. This leads to an increase in the number of the variables and constraints, thus making the optimization problem difficult to solve. To address this issue, we design a Progressively Increasing Sampling Number Optimization (PISNO) framework. More specifically, we first find a nearly feasible trajectory with a small sampling number to warm-start the optimization process. Then, the sampling number is progressively increased, and the corresponding intermediate Optimal Control Problem (OCP) is solved in each iteration. Next, we further resample the obtained solution into a finer sampling period, and then use it to warm-start the intermediate OCP in next iteration. This process is repeated until reaching a threshold sampling number. Simulation and experiment results show the proposed method exhibits a good performance and less computational consumption over the benchmarks. 

**Abstract (ZH)**: 基于优化的曲线路况下挂车车辆轨迹规划方法 

---
# ManiFeel: Benchmarking and Understanding Visuotactile Manipulation Policy Learning 

**Title (ZH)**: ManiFeel：评估与理解基于视觉-触觉操控策略学习 

**Authors**: Quan Khanh Luu, Pokuang Zhou, Zhengtong Xu, Zhiyuan Zhang, Qiang Qiu, Yu She  

**Link**: [PDF](https://arxiv.org/pdf/2505.18472)  

**Abstract**: Supervised visuomotor policies have shown strong performance in robotic manipulation but often struggle in tasks with limited visual input, such as operations in confined spaces, dimly lit environments, or scenarios where perceiving the object's properties and state is critical for task success. In such cases, tactile feedback becomes essential for manipulation. While the rapid progress of supervised visuomotor policies has benefited greatly from high-quality, reproducible simulation benchmarks in visual imitation, the visuotactile domain still lacks a similarly comprehensive and reliable benchmark for large-scale and rigorous evaluation. To address this, we introduce ManiFeel, a reproducible and scalable simulation benchmark for studying supervised visuotactile manipulation policies across a diverse set of tasks and scenarios. ManiFeel presents a comprehensive benchmark suite spanning a diverse set of manipulation tasks, evaluating various policies, input modalities, and tactile representation methods. Through extensive experiments, our analysis reveals key factors that influence supervised visuotactile policy learning, identifies the types of tasks where tactile sensing is most beneficial, and highlights promising directions for future research in visuotactile policy learning. ManiFeel aims to establish a reproducible benchmark for supervised visuotactile policy learning, supporting progress in visuotactile manipulation and perception. To facilitate future research and ensure reproducibility, we will release our codebase, datasets, training logs, and pretrained checkpoints. Please visit the project website for more details: this https URL 

**Abstract (ZH)**: 监督视觉-运动策略在机器人操作任务中表现强劲，但在视觉输入有限的任务中往往表现不佳，如受限空间操作、光线不足的环境中，或需要感知物体特性和状态才能成功完成任务的场景。在这种情况下，触觉反馈对于操作变得至关重要。尽管监督视觉-运动策略的快速发展得益于视觉模仿中的高质量、可重复的仿真基准，但在触觉-视觉领域仍缺乏类似全面和可靠的基准，用于大规模和严格的评估。为解决这一问题，我们引入了ManiFeel，一个可重复且可扩展的仿真基准，用于跨多种任务和场景研究监督触觉-视觉操作策略。ManiFeel提供了一个涵盖广泛操作任务的基准套件，评估各种策略、输入模态和触觉表示方法。通过大量实验，我们的分析揭示了影响监督触觉-视觉策略学习的关键因素，确定了触觉传感最为有益的任务类型，并指出了触觉策略学习未来研究的有希望方向。ManiFeel旨在建立一个监督触觉-视觉策略学习的可重复基准，促进触觉-视觉操作和感知的进步。为了便于未来研究并确保可重复性，我们将公开我们的代码库、数据集、训练日志和预训练检查点。更多详情请访问项目网站：this https URL。 

---
# MorphEUS: Morphable Omnidirectional Unmanned System 

**Title (ZH)**: morphEUS: 变形的全向无人驾驶系统 

**Authors**: Ivan Bao, José C. Díaz Peón González Pacheco, Atharva Navsalkar, Andrew Scheffer, Sashreek Shankar, Andrew Zhao, Hongyu Zhou, Vasileios Tzoumas  

**Link**: [PDF](https://arxiv.org/pdf/2505.18270)  

**Abstract**: Omnidirectional aerial vehicles (OMAVs) have opened up a wide range of possibilities for inspection, navigation, and manipulation applications using drones. In this paper, we introduce MorphEUS, a morphable co-axial quadrotor that can control position and orientation independently with high efficiency. It uses a paired servo motor mechanism for each rotor arm, capable of pointing the vectored-thrust in any arbitrary direction. As compared to the \textit{state-of-the-art} OMAVs, we achieve higher and more uniform force/torque reachability with a smaller footprint and minimum thrust cancellations. The overactuated nature of the system also results in resiliency to rotor or servo-motor failures. The capabilities of this quadrotor are particularly well-suited for contact-based infrastructure inspection and close-proximity imaging of complex geometries. In the accompanying control pipeline, we present theoretical results for full controllability, almost-everywhere exponential stability, and thrust-energy optimality. We evaluate our design and controller on high-fidelity simulations showcasing the trajectory-tracking capabilities of the vehicle during various tasks. Supplementary details and experimental videos are available on the project webpage. 

**Abstract (ZH)**: 全向空中车辆（OMAVs）为使用无人机进行检查、导航和操作应用开辟了广泛的可能性。本文介绍了一种可重塑的共轴四旋翼机MorphEUS，它可以高效地独立控制位置和姿态。该四旋翼机采用每旋臂配对的伺服电机机制，能够将矢量推力指向任意方向。与当前最先进的OMAVs相比，我们实现了更高的且更为均匀的力量/力矩可达性，并具有更小的占地面积和最少的推力抵消。系统的过驱动特性还使其对旋翼或伺服电机故障具有韧性。该四旋翼机特别适合接触式的基础设施检查以及复杂几何结构的近距离成像。在配套的控制管道中，我们提出了全可控性、几乎处处指数稳定性和推力能量优化的理论结果。我们在高保真模拟中评估了该设计和控制器在各类任务中的路径追踪能力，并展示了车辆轨迹跟踪能力。更多补充细节和实验视频可在项目网页上获取。 

---
