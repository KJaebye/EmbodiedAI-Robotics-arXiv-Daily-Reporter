# Edge Nearest Neighbor in Sampling-Based Motion Planning 

**Title (ZH)**: 基于采样法运动规划中的边最近邻方法 

**Authors**: Stav Ashur, Nancy M. Amato, Sariel Har-Peled  

**Link**: [PDF](https://arxiv.org/pdf/2506.13753)  

**Abstract**: Neighborhood finders and nearest neighbor queries are fundamental parts of sampling based motion planning algorithms. Using different distance metrics or otherwise changing the definition of a neighborhood produces different algorithms with unique empiric and theoretical properties. In \cite{l-pa-06} LaValle suggests a neighborhood finder for the Rapidly-exploring Random Tree RRT
algorithm \cite{l-rrtnt-98} which finds the nearest neighbor of the sampled point on the swath of the tree, that is on the set of all of the points on the tree edges, using a hierarchical data structure. In this paper we implement such a neighborhood finder and show, theoretically and experimentally, that this results in more efficient algorithms, and suggest a variant of the Rapidly-exploring Random Graph RRG algorithm \cite{f-isaom-10} that better exploits the exploration properties of the newly described subroutine for finding narrow passages. 

**Abstract (ZH)**: 基于采样方法的运动规划算法中，邻居查找器和最近邻查询是基本组成部分。使用不同的距离度量或重新定义邻居的定义会产生具有独特经验和理论性质的不同算法。在LaValle的《Probabilistic Robotics》（2006）中，他建议了一种适用于快速探索随机树RRT（1998）算法的邻居查找器，该查找器使用层次数据结构在树的路径上的所有点集中找到采样点的最近邻。本文实现了这样的邻居查找器，并从理论和实验上证明这可以提高算法效率，并提出了一种改进的快速探索随机图RRG算法（2010），更好地利用了新描述的查找狭窄通道的子程序的探索性质。 

---
# HARMONI: Haptic-Guided Assistance for Unified Robotic Tele-Manipulation and Tele-Navigation 

**Title (ZH)**: HARMONI: 耦合触觉引导的统一机器人远程操作与导航辅助 

**Authors**: V. Sripada, A. Khan, J. Föcker, S. Parsa, Susmitha P, H Maior, A. Ghalamzan-E  

**Link**: [PDF](https://arxiv.org/pdf/2506.13704)  

**Abstract**: Shared control, which combines human expertise with autonomous assistance, is critical for effective teleoperation in complex environments. While recent advances in haptic-guided teleoperation have shown promise, they are often limited to simplified tasks involving 6- or 7-DoF manipulators and rely on separate control strategies for navigation and manipulation. This increases both cognitive load and operational overhead. In this paper, we present a unified tele-mobile manipulation framework that leverages haptic-guided shared control. The system integrates a 9-DoF follower mobile manipulator and a 7-DoF leader robotic arm, enabling seamless transitions between tele-navigation and tele-manipulation through real-time haptic feedback. A user study with 20 participants under real-world conditions demonstrates that our framework significantly improves task accuracy and efficiency without increasing cognitive load. These findings highlight the potential of haptic-guided shared control for enhancing operator performance in demanding teleoperation scenarios. 

**Abstract (ZH)**: 基于触觉引导的协同控制的统一远程移动操作框架 

---
# Disturbance-aware minimum-time planning strategies for motorsport vehicles with probabilistic safety certificates 

**Title (ZH)**: 基于扰动感知的最短时间规划策略及其概率安全证书研究 

**Authors**: Martino Gulisano, Matteo Masoni, Marco Gabiccini, Massimo Guiggiani  

**Link**: [PDF](https://arxiv.org/pdf/2506.13622)  

**Abstract**: This paper presents a disturbance-aware framework that embeds robustness into minimum-lap-time trajectory optimization for motorsport. Two formulations are introduced. (i) Open-loop, horizon-based covariance propagation uses worst-case uncertainty growth over a finite window to tighten tire-friction and track-limit constraints. (ii) Closed-loop, covariance-aware planning incorporates a time-varying LQR feedback law in the optimizer, providing a feedback-consistent estimate of disturbance attenuation and enabling sharper yet reliable constraint tightening. Both methods yield reference trajectories for human or artificial drivers: in autonomous applications the modelled controller can replicate the on-board implementation, while for human driving accuracy increases with the extent to which the driver can be approximated by the assumed time-varying LQR policy. Computational tests on a representative Barcelona-Catalunya sector show that both schemes meet the prescribed safety probability, yet the closed-loop variant incurs smaller lap-time penalties than the more conservative open-loop solution, while the nominal (non-robust) trajectory remains infeasible under the same uncertainties. By accounting for uncertainty growth and feedback action during planning, the proposed framework delivers trajectories that are both performance-optimal and probabilistically safe, advancing minimum-time optimization toward real-world deployment in high-performance motorsport and autonomous racing. 

**Abstract (ZH)**: 基于干扰感知的最小圈时轨迹优化鲁棒性框架：开环与闭环方法 

---
# JENGA: Object selection and pose estimation for robotic grasping from a stack 

**Title (ZH)**: JENGA: 从堆积物中进行机器人抓取的对象选择与姿态估计 

**Authors**: Sai Srinivas Jeevanandam, Sandeep Inuganti, Shreedhar Govil, Didier Stricker, Jason Rambach  

**Link**: [PDF](https://arxiv.org/pdf/2506.13425)  

**Abstract**: Vision-based robotic object grasping is typically investigated in the context of isolated objects or unstructured object sets in bin picking scenarios. However, there are several settings, such as construction or warehouse automation, where a robot needs to interact with a structured object formation such as a stack. In this context, we define the problem of selecting suitable objects for grasping along with estimating an accurate 6DoF pose of these objects. To address this problem, we propose a camera-IMU based approach that prioritizes unobstructed objects on the higher layers of stacks and introduce a dataset for benchmarking and evaluation, along with a suitable evaluation metric that combines object selection with pose accuracy. Experimental results show that although our method can perform quite well, this is a challenging problem if a completely error-free solution is needed. Finally, we show results from the deployment of our method for a brick-picking application in a construction scenario. 

**Abstract (ZH)**: 基于视觉的机器人物体抓取通常在孤立物体或未结构化的物体集合的拾取场景中进行研究。然而，在建筑或仓库自动化等场景中，机器人需要与结构化的物体堆叠进行交互。在这种情况下，我们定义了选择适合抓取的物体并估算这些物体的准确6DoF姿态的问题。为了解决这个问题，我们提出了一种基于摄像头-IMU的方法，优先选择堆叠高层上的无遮挡物体，并介绍了一个用于基准测试和评估的数据集以及一个结合物体选择与姿态准确性的合适评价指标。实验结果表明，尽管我们的方法表现不错，但如果需要完全无误差的解决方案，则这是一个极具挑战性的问题。最后，我们在建筑场景中的砖块拾取应用中部署了我们的方法并展示了结果。 

---
# Delayed Expansion AGT: Kinodynamic Planning with Application to Tractor-Trailer Parking 

**Title (ZH)**: 延迟扩张AGT：动力学规划及其在牵引车-挂车泊车中的应用 

**Authors**: Dongliang Zheng, Yebin Wang, Stefano Di Cairano, Panagiotis Tsiotras  

**Link**: [PDF](https://arxiv.org/pdf/2506.13421)  

**Abstract**: Kinodynamic planning of articulated vehicles in cluttered environments faces additional challenges arising from high-dimensional state space and complex system dynamics. Built upon [1],[2], this work proposes the DE-AGT algorithm that grows a tree using pre-computed motion primitives (MPs) and A* heuristics. The first feature of DE-AGT is a delayed expansion of MPs. In particular, the MPs are divided into different modes, which are ranked online. With the MP classification and prioritization, DE-AGT expands the most promising mode of MPs first, which eliminates unnecessary computation and finds solutions faster. To obtain the cost-to-go heuristic for nonholonomic articulated vehicles, we rely on supervised learning and train neural networks for fast and accurate cost-to-go prediction. The learned heuristic is used for online mode ranking and node selection. Another feature of DE-AGT is the improved goal-reaching. Exactly reaching a goal state usually requires a constant connection checking with the goal by solving steering problems -- non-trivial and time-consuming for articulated vehicles. The proposed termination scheme overcomes this challenge by tightly integrating a light-weight trajectory tracking controller with the search process. DE-AGT is implemented for autonomous parking of a general car-like tractor with 3-trailer. Simulation results show an average of 10x acceleration compared to a previous method. 

**Abstract (ZH)**: 带有延迟扩展的MPs及其优先级化的DE-AGT算法在复杂环境中的铰接车辆机动规划 

---
# Observability-Aware Active Calibration of Multi-Sensor Extrinsics for Ground Robots via Online Trajectory Optimization 

**Title (ZH)**: 基于在线轨迹优化的地面机器人多传感器外参观测导向的主动标定 

**Authors**: Jiang Wang, Yaozhong Kang, Linya Fu, Kazuhiro Nakadai, He Kong  

**Link**: [PDF](https://arxiv.org/pdf/2506.13420)  

**Abstract**: Accurate calibration of sensor extrinsic parameters for ground robotic systems (i.e., relative poses) is crucial for ensuring spatial alignment and achieving high-performance perception. However, existing calibration methods typically require complex and often human-operated processes to collect data. Moreover, most frameworks neglect acoustic sensors, thereby limiting the associated systems' auditory perception capabilities. To alleviate these issues, we propose an observability-aware active calibration method for ground robots with multimodal sensors, including a microphone array, a LiDAR (exteroceptive sensors), and wheel encoders (proprioceptive sensors). Unlike traditional approaches, our method enables active trajectory optimization for online data collection and calibration, contributing to the development of more intelligent robotic systems. Specifically, we leverage the Fisher information matrix (FIM) to quantify parameter observability and adopt its minimum eigenvalue as an optimization metric for trajectory generation via B-spline curves. Through planning and replanning of robot trajectory online, the method enhances the observability of multi-sensor extrinsic parameters. The effectiveness and advantages of our method have been demonstrated through numerical simulations and real-world experiments. For the benefit of the community, we have also open-sourced our code and data at this https URL. 

**Abstract (ZH)**: 地面机器人多模态传感器外参的可观性感知主动标定方法对确保空间对齐和实现高性能感知至关重要。然而，现有的标定方法通常需要复杂且往往依赖人工的数据采集过程。此外，大多数框架忽视了声学传感器，从而限制了相关系统的声音感知能力。为解决这些问题，我们提出了一种地面机器人多模态传感器（包括麦克风阵列、激光雷达（外部传感器）和编码器（内部传感器））的可观性感知主动标定方法。与传统方法不同，我们的方法能够实现主动轨迹优化以在线数据采集和标定，从而推动更智能的机器人系统的研发。具体而言，我们利用 Fisher 信息矩阵（FIM）量化参数可观性，并采用其最小特征值作为通过 B-样条曲线生成轨迹的优化指标。通过在线规划和重新规划机器人的轨迹，该方法增强了多传感器外参的可观性。我们的方法的有效性和优势已在数值仿真和实际实验中得到验证。为了社区的益处，我们已在该 URL 开源了我们的代码和数据。 

---
# C2TE: Coordinated Constrained Task Execution Design for Ordering-Flexible Multi-Vehicle Platoon Merging 

**Title (ZH)**: C2TE: 协调约束任务执行设计用于订单灵活的多车辆编队合并 

**Authors**: Bin-Bin Hu, Yanxin Zhou, Henglai Wei, Shuo Cheng, Chen Lv  

**Link**: [PDF](https://arxiv.org/pdf/2506.13202)  

**Abstract**: In this paper, we propose a distributed coordinated constrained task execution (C2TE) algorithm that enables a team of vehicles from different lanes to cooperatively merge into an {\it ordering-flexible platoon} maneuvering on the desired lane. Therein, the platoon is flexible in the sense that no specific spatial ordering sequences of vehicles are predetermined. To attain such a flexible platoon, we first separate the multi-vehicle platoon (MVP) merging mission into two stages, namely, pre-merging regulation and {\it ordering-flexible platoon} merging, and then formulate them into distributed constraint-based optimization problems. Particularly, by encoding longitudinal-distance regulation and same-lane collision avoidance subtasks into the corresponding control barrier function (CBF) constraints, the proposed algorithm in Stage 1 can safely enlarge sufficient longitudinal distances among adjacent vehicles. Then, by encoding lateral convergence, longitudinal-target attraction, and neighboring collision avoidance subtasks into CBF constraints, the proposed algorithm in Stage~2 can efficiently achieve the {\it ordering-flexible platoon}. Note that the {\it ordering-flexible platoon} is realized through the interaction of the longitudinal-target attraction and time-varying neighboring collision avoidance constraints simultaneously. Feasibility guarantee and rigorous convergence analysis are both provided under strong nonlinear couplings induced by flexible orderings. Finally, experiments using three autonomous mobile vehicles (AMVs) are conducted to verify the effectiveness and flexibility of the proposed algorithm, and extensive simulations are performed to demonstrate its robustness, adaptability, and scalability when tackling vehicles' sudden breakdown, new appearing, different number of lanes, mixed autonomy, and large-scale scenarios, respectively. 

**Abstract (ZH)**: 分布式协同约束任务执行算法：不同车道车辆团队的可排序灵活车队合并方法 

---
# Equilibrium-Driven Smooth Separation and Navigation of Marsupial Robotic Systems 

**Title (ZH)**: 袋鼠型机器人系统均衡驱动平滑分离与导航 

**Authors**: Bin-Bin Hu, Bayu Jayawardhana, Ming Cao  

**Link**: [PDF](https://arxiv.org/pdf/2506.13198)  

**Abstract**: In this paper, we propose an equilibrium-driven controller that enables a marsupial carrier-passenger robotic system to achieve smooth carrier-passenger separation and then to navigate the passenger robot toward a predetermined target point. Particularly, we design a potential gradient in the form of a cubic polynomial for the passenger's controller as a function of the carrier-passenger and carrier-target distances in the moving carrier's frame. This introduces multiple equilibrium points corresponding to the zero state of the error dynamic system during carrier-passenger separation. The change of equilibrium points is associated with the change in their attraction regions, enabling smooth carrier-passenger separation and afterwards seamless navigation toward the target. Finally, simulations demonstrate the effectiveness and adaptability of the proposed controller in environments containing obstacles. 

**Abstract (ZH)**: 本文提出一个均衡驱动控制器，使育儿袋式载运机器人能够在实现平稳的载运机器人与乘客机器人分离后，导航乘客机器人前往预设目标点。特别地，我们在移动载体坐标系中，基于载运机器人与乘客机器人及载运机器人与目标点的距离，设计了一个立方多项式的潜在梯度作为乘客机器人的控制器。这引入了多个均衡点，对应于载运机器人与乘客机器人分离过程中误差动态系统的零状态。均衡点的变化与其吸引力区域的变化相关，从而实现平稳的载运机器人与乘客机器人分离，并在之后无缝地导航至目标点。最后，仿真实验展示了所提控制器在包含障碍物环境中的有效性和适应性。 

---
# Autonomous 3D Moving Target Encirclement and Interception with Range measurement 

**Title (ZH)**: 自主三维移动目标环围与拦截（基于距离测量） 

**Authors**: Fen Liu, Shenghai Yuan, Thien-Minh Nguyen, Rong Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.13106)  

**Abstract**: Commercial UAVs are an emerging security threat as they are capable of carrying hazardous payloads or disrupting air traffic. To counter UAVs, we introduce an autonomous 3D target encirclement and interception strategy. Unlike traditional ground-guided systems, this strategy employs autonomous drones to track and engage non-cooperative hostile UAVs, which is effective in non-line-of-sight conditions, GPS denial, and radar jamming, where conventional detection and neutralization from ground guidance fail. Using two noisy real-time distances measured by drones, guardian drones estimate the relative position from their own to the target using observation and velocity compensation methods, based on anti-synchronization (AS) and an X$-$Y circular motion combined with vertical jitter. An encirclement control mechanism is proposed to enable UAVs to adaptively transition from encircling and protecting a target to encircling and monitoring a hostile target. Upon breaching a warning threshold, the UAVs may even employ a suicide attack to neutralize the hostile target. We validate this strategy through real-world UAV experiments and simulated analysis in MATLAB, demonstrating its effectiveness in detecting, encircling, and intercepting hostile drones. More details: this https URL. 

**Abstract (ZH)**: 商用无人机是一种新兴的安全威胁，因为它们能够携带危险载荷或干扰空中交通。为此，我们提出了一种自主三维目标包围和拦截策略。该策略不同于传统的地面引导系统，它利用自主无人机追踪并对抗不合作的敌对无人机，在视线外、GPS拒绝和雷达干扰等条件下表现出色，而传统从地面引导进行检测和中和会失效。基于反同步（AS）以及结合XY圆周运动和垂直抖动的观测和速度补偿方法，利用两架无人机测得的带噪声的实时距离，守护无人机估计到目标的相对位置，并提出了一种包围控制机制，使无人机能够从保护目标转向包围监视敌对目标。当突破警告阈值时，无人机甚至可以采取自杀袭击来中和敌对目标。通过实际无人机实验和MATLAB模拟分析验证了该策略，在检测、包围和拦截敌对无人机方面显示了有效性。更多细节：请访问此链接。 

---
# Underwater target 6D State Estimation via UUV Attitude Enhance Observability 

**Title (ZH)**: 基于UUV姿态增强可观测性的水下目标6维状态估计 

**Authors**: Fen Liu, Chengfeng Jia, Na Zhang, Shenghai Yuan, Rong Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.13105)  

**Abstract**: Accurate relative state observation of Unmanned Underwater Vehicles (UUVs) for tracking uncooperative targets remains a significant challenge due to the absence of GPS, complex underwater dynamics, and sensor limitations. Existing localization approaches rely on either global positioning infrastructure or multi-UUV collaboration, both of which are impractical for a single UUV operating in large or unknown environments. To address this, we propose a novel persistent relative 6D state estimation framework that enables a single UUV to estimate its relative motion to a non-cooperative target using only successive noisy range measurements from two monostatic sonar sensors. Our key contribution is an observability-enhanced attitude control strategy, which optimally adjusts the UUV's orientation to improve the observability of relative state estimation using a Kalman filter, effectively mitigating the impact of sensor noise and drift accumulation. Additionally, we introduce a rigorously proven Lyapunov-based tracking control strategy that guarantees long-term stability by ensuring that the UUV maintains an optimal measurement range, preventing localization errors from diverging over time. Through theoretical analysis and simulations, we demonstrate that our method significantly improves 6D relative state estimation accuracy and robustness compared to conventional approaches. This work provides a scalable, infrastructure-free solution for UUVs tracking uncooperative targets underwater. 

**Abstract (ZH)**: 基于单艇相对6D状态估计的无人驾驶水下车辆追踪非合作目标方法 

---
# IKDiffuser: Fast and Diverse Inverse Kinematics Solution Generation for Multi-arm Robotic Systems 

**Title (ZH)**: IKDiffuser：多臂机器人系统快速多样逆运动学解决方案生成 

**Authors**: Zeyu Zhang, Ziyuan Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2506.13087)  

**Abstract**: Solving Inverse Kinematics (IK) problems is fundamental to robotics, but has primarily been successful with single serial manipulators. For multi-arm robotic systems, IK remains challenging due to complex self-collisions, coupled joints, and high-dimensional redundancy. These complexities make traditional IK solvers slow, prone to failure, and lacking in solution diversity. In this paper, we present IKDiffuser, a diffusion-based model designed for fast and diverse IK solution generation for multi-arm robotic systems. IKDiffuser learns the joint distribution over the configuration space, capturing complex dependencies and enabling seamless generalization to multi-arm robotic systems of different structures. In addition, IKDiffuser can incorporate additional objectives during inference without retraining, offering versatility and adaptability for task-specific requirements. In experiments on 6 different multi-arm systems, the proposed IKDiffuser achieves superior solution accuracy, precision, diversity, and computational efficiency compared to existing solvers. The proposed IKDiffuser framework offers a scalable, unified approach to solving multi-arm IK problems, facilitating the potential of multi-arm robotic systems in real-time manipulation tasks. 

**Abstract (ZH)**: 基于扩散模型的多臂机器人逆动力学快速多样求解 

---
# Constrained Optimal Planning to Minimize Battery Degradation of Autonomous Mobile Robots 

**Title (ZH)**: 自主移动机器人电池退化最小化的受约束最优规划 

**Authors**: Jiachen Li, Jian Chu, Feiyang Zhao, Shihao Li, Wei Li, Dongmei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.13019)  

**Abstract**: This paper proposes an optimization framework that addresses both cycling degradation and calendar aging of batteries for autonomous mobile robot (AMR) to minimize battery degradation while ensuring task completion. A rectangle method of piecewise linear approximation is employed to linearize the bilinear optimization problem. We conduct a case study to validate the efficiency of the proposed framework in achieving an optimal path planning for AMRs while reducing battery aging. 

**Abstract (ZH)**: 本文提出了一种优化框架，旨在同时解决自主移动机器人(AMR)电池的充放电退化和日历老化问题，以最小化电池退化并确保任务完成。通过采用分段线性逼近的矩形法将双线性优化问题线性化。通过案例研究验证了所提出框架在实现AMR最优路径规划的同时减少电池老化方面的有效性。 

---
# On-board Sonar Data Classification for Path Following in Underwater Vehicles using Fast Interval Type-2 Fuzzy Extreme Learning Machine 

**Title (ZH)**: 使用快速区间型2模糊极限学习机的水下车辆路径跟随声纳数据分类 

**Authors**: Adrian Rubio-Solis, Luciano Nava-Balanzar, Tomas Salgado-Jimenez  

**Link**: [PDF](https://arxiv.org/pdf/2506.12762)  

**Abstract**: In autonomous underwater missions, the successful completion of predefined paths mainly depends on the ability of underwater vehicles to recognise their surroundings. In this study, we apply the concept of Fast Interval Type-2 Fuzzy Extreme Learning Machine (FIT2-FELM) to train a Takagi-Sugeno-Kang IT2 Fuzzy Inference System (TSK IT2-FIS) for on-board sonar data classification using an underwater vehicle called BlueROV2. The TSK IT2-FIS is integrated into a Hierarchical Navigation Strategy (HNS) as the main navigation engine to infer local motions and provide the BlueROV2 with full autonomy to follow an obstacle-free trajectory in a water container of 2.5m x 2.5m x 3.5m. Compared to traditional navigation architectures, using the proposed method, we observe a robust path following behaviour in the presence of uncertainty and noise. We found that the proposed approach provides the BlueROV with a more complete sensory picture about its surroundings while real-time navigation planning is performed by the concurrent execution of two or more tasks. 

**Abstract (ZH)**: 基于Fast Interval Type-2 Fuzzy Extreme Learning Machine的BlueROV2 underwater车辆自主水下任务中声纳数据分类与导航研究 

---
# Deep Fusion of Ultra-Low-Resolution Thermal Camera and Gyroscope Data for Lighting-Robust and Compute-Efficient Rotational Odometry 

**Title (ZH)**: 基于超高分辨率热敏相机和陀螺仪数据的深度融合：面向照明鲁棒性和计算效率的旋转里程计 

**Authors**: Farida Mohsen, Ali Safa  

**Link**: [PDF](https://arxiv.org/pdf/2506.12536)  

**Abstract**: Accurate rotational odometry is crucial for autonomous robotic systems, particularly for small, power-constrained platforms such as drones and mobile robots. This study introduces thermal-gyro fusion, a novel sensor fusion approach that integrates ultra-low-resolution thermal imaging with gyroscope readings for rotational odometry. Unlike RGB cameras, thermal imaging is invariant to lighting conditions and, when fused with gyroscopic data, mitigates drift which is a common limitation of inertial sensors. We first develop a multimodal data acquisition system to collect synchronized thermal and gyroscope data, along with rotational speed labels, across diverse environments. Subsequently, we design and train a lightweight Convolutional Neural Network (CNN) that fuses both modalities for rotational speed estimation. Our analysis demonstrates that thermal-gyro fusion enables a significant reduction in thermal camera resolution without significantly compromising accuracy, thereby improving computational efficiency and memory utilization. These advantages make our approach well-suited for real-time deployment in resource-constrained robotic systems. Finally, to facilitate further research, we publicly release our dataset as supplementary material. 

**Abstract (ZH)**: 准确的旋转里程计对于自主机器人系统至关重要，特别适用于如无人机和移动机器人等小型、功率受限的平台。本研究介绍了一种热敏-陀螺仪融合方法，该方法结合了超低分辨率热成像与陀螺仪读数以实现旋转里程计。与RGB相机不同，热成像对光照条件不敏感，并且与陀螺仪数据融合可以缓解由惯性传感器常见的漂移问题。我们首先开发了一种多模态数据采集系统，以同步收集热成像和陀螺仪数据以及旋转速度标签，适用于多种环境。随后，我们设计并训练了一种轻量级卷积神经网络（CNN），用于融合这两种模态以估计旋转速度。我们的分析表明，热敏-陀螺仪融合能够在显著降低热成像分辨率的情况下，不显著牺牲准确性，从而提高计算效率和内存利用率。这些优势使我们的方法非常适合在资源受限的机器人系统中进行实时部署。最后，为了促进进一步的研究，我们公开发布了我们的数据集作为补充材料。 

---
# A Spatial Relationship Aware Dataset for Robotics 

**Title (ZH)**: 空间关系感知数据集 для 机器人学 

**Authors**: Peng Wang, Minh Huy Pham, Zhihao Guo, Wei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.12525)  

**Abstract**: Robotic task planning in real-world environments requires not only object recognition but also a nuanced understanding of spatial relationships between objects. We present a spatial-relationship-aware dataset of nearly 1,000 robot-acquired indoor images, annotated with object attributes, positions, and detailed spatial relationships. Captured using a Boston Dynamics Spot robot and labelled with a custom annotation tool, the dataset reflects complex scenarios with similar or identical objects and intricate spatial arrangements. We benchmark six state-of-the-art scene-graph generation models on this dataset, analysing their inference speed and relational accuracy. Our results highlight significant differences in model performance and demonstrate that integrating explicit spatial relationships into foundation models, such as ChatGPT 4o, substantially improves their ability to generate executable, spatially-aware plans for robotics. The dataset and annotation tool are publicly available at this https URL, supporting further research in spatial reasoning for robotics. 

**Abstract (ZH)**: 实时环境中的机器人任务规划不仅需要物体识别，还需要对物体之间空间关系的深刻理解。我们呈现了一个包含近1,000张室内图像的空间关系感知数据集，这些图像由机器人 annotation 并标注了物体属性、位置和详细的空间关系。使用波士顿动力公司的 Spot 机器人采集，并使用自定义标注工具进行标记，该数据集反映了具有相似或相同物体和复杂空间布局的复杂场景。我们在该数据集上基准测试了六种最先进的场景图生成模型，分析了它们的推理速度和关系准确性。我们的结果强调了不同模型性能的显著差异，并表明将明确的空间关系集成到基础模型（如ChatGPT 4o）中，可以显著提高其生成可执行且空间感知的机器人计划的能力。数据集和标注工具在此httpsURL公开，支持进一步的空间推理研究用于机器人领域。 

---
# Design and Development of a Robotic Transcatheter Delivery System for Aortic Valve Replacement 

**Title (ZH)**: 基于经导管输送系统的主动脉瓣置换机器人设计与开发 

**Authors**: Harith S. Gallage, Bailey F. De Sousa, Benjamin I. Chesnik, Chaikel G. Brownstein, Anson Paul, Ronghuai Qi  

**Link**: [PDF](https://arxiv.org/pdf/2506.12082)  

**Abstract**: Minimally invasive transcatheter approaches are increasingly adopted for aortic stenosis treatment, where optimal commissural and coronary alignment is important. Achieving precise alignment remains clinically challenging, even with contemporary robotic transcatheter aortic valve replacement (TAVR) devices, as this task is still performed manually. This paper proposes the development of a robotic transcatheter delivery system featuring an omnidirectional bending joint and an actuation system designed to enhance positional accuracy and precision in TAVR procedures. The preliminary experimental results validate the functionality of this novel robotic system. 

**Abstract (ZH)**: 微创经导管 Approaches for 二尖瓣狭窄 治疗中，最佳隔膜和冠状动脉对齐至关重要。尽管使用当今的机器人经导管主动脉瓣置换（TAVR）设备，实现精确对齐仍然具有临床挑战性，因为这项任务仍然需要手动完成。本文提出开发一种配备全景弯曲关节和旨在增强 TAVR 程序中位置准确性和精度的驱动系统的机器人经导管输送系统。初步的实验结果验证了该新型机器人系统的功能。 

---
# Real Time Self-Tuning Adaptive Controllers on Temperature Control Loops using Event-based Game Theory 

**Title (ZH)**: 基于事件驱动博弈论的实时自调适应温度控制环控制器 

**Authors**: Steve Yuwono, Muhammad Uzair Rana, Dorothea Schwung, Andreas Schwung  

**Link**: [PDF](https://arxiv.org/pdf/2506.13164)  

**Abstract**: This paper presents a novel method for enhancing the adaptability of Proportional-Integral-Derivative (PID) controllers in industrial systems using event-based dynamic game theory, which enables the PID controllers to self-learn, optimize, and fine-tune themselves. In contrast to conventional self-learning approaches, our proposed framework offers an event-driven control strategy and game-theoretic learning algorithms. The players collaborate with the PID controllers to dynamically adjust their gains in response to set point changes and disturbances. We provide a theoretical analysis showing sound convergence guarantees for the game given suitable stability ranges of the PID controlled loop. We further introduce an automatic boundary detection mechanism, which helps the players to find an optimal initialization of action spaces and significantly reduces the exploration time. The efficacy of this novel methodology is validated through its implementation in the temperature control loop of a printing press machine. Eventually, the outcomes of the proposed intelligent self-tuning PID controllers are highly promising, particularly in terms of reducing overshoot and settling time. 

**Abstract (ZH)**: 本文提出了一种使用事件驱动动态博弈理论增强比例积分微分（PID）控制器适应性的新型方法，使PID控制器能够自我学习、优化和精调。与传统的自我学习方法相比，我们提出的框架提供了基于事件的控制策略和博弈论学习算法。博弈中的玩家与PID控制器协作，动态调整增益以响应设定点变化和干扰。我们提供了理论分析，证明在PID控制环具有合适稳定范围的情况下，博弈具有合理的收敛保证。此外，我们引入了一种自动边界检测机制，有助于玩家找到最优的动作空间初始化，并显著减少探索时间。该新颖方法的有效性通过在印刷机温控回路中的实现得以验证。最终，所提出的智能自调谐PID控制器的成果极具前景，特别是在减少超调和稳态时间方面表现突出。 

---
