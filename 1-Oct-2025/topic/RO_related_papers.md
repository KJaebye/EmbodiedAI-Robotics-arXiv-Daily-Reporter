# Radio-based Multi-Robot Odometry and Relative Localization 

**Title (ZH)**: 基于无线电的多机器人里程计与相对定位 

**Authors**: Andrés Martínez-Silva, David Alejo, Luis Merino, Fernando Caballero  

**Link**: [PDF](https://arxiv.org/pdf/2509.26558)  

**Abstract**: Radio-based methods such as Ultra-Wideband (UWB) and RAdio Detection And Ranging (radar), which have traditionally seen limited adoption in robotics, are experiencing a boost in popularity thanks to their robustness to harsh environmental conditions and cluttered environments. This work proposes a multi-robot UGV-UAV localization system that leverages the two technologies with inexpensive and readily-available sensors, such as Inertial Measurement Units (IMUs) and wheel encoders, to estimate the relative position of an aerial robot with respect to a ground robot. The first stage of the system pipeline includes a nonlinear optimization framework to trilaterate the location of the aerial platform based on UWB range data, and a radar pre-processing module with loosely coupled ego-motion estimation which has been adapted for a multi-robot scenario. Then, the pre-processed radar data as well as the relative transformation are fed to a pose-graph optimization framework with odometry and inter-robot constraints. The system, implemented for the Robotic Operating System (ROS 2) with the Ceres optimizer, has been validated in Software-in-the-Loop (SITL) simulations and in a real-world dataset. The proposed relative localization module outperforms state-of-the-art closed-form methods which are less robust to noise. Our SITL environment includes a custom Gazebo plugin for generating realistic UWB measurements modeled after real data. Conveniently, the proposed factor graph formulation makes the system readily extensible to full Simultaneous Localization And Mapping (SLAM). Finally, all the code and experimental data is publicly available to support reproducibility and to serve as a common open dataset for benchmarking. 

**Abstract (ZH)**: 基于无线电波的方法，如超宽带（UWB）和雷达（Radar），尽管在机器人领域传统上应用有限，但得益于其在恶劣环境和复杂环境中的鲁棒性，这些技术正逐渐流行起来。本文提出了一种多机器人UGV-UAV localization系统，利用这些技术以及低成本易获得的传感器（如惯性测量单元（IMU）和轮码盘），以估计空中机器人相对于地面机器人的相对位置。系统管道的第一阶段包括一个非线性优化框架，用于基于UWB范围数据进行三角测量，并包括一个适用于多机器人场景的雷达预处理模块，该模块与自我运动估计松散耦合。然后，预处理后的雷达数据以及相对变换被输入一个包含里程计和机器人间约束的姿态图优化框架。该系统基于Robotic Operating System（ROS 2）并通过Ceres优化器实现，并已在Software-in-the-Loop（SITL）仿真和真实世界数据集上得到了验证。所提出的相对定位模块在抗噪性方面优于最先进的闭式解方法。我们的SITL环境包含一个根据真实数据生成真实UWB测量值的自定义Gazebo插件。此外，所提出的因子图表示使系统易于扩展到完整的Simultaneous Localization And Mapping（SLAM）。最后，所有代码和实验数据均公开，以支持可再现性，并作为基准测试的共同开放数据集。 

---
# Memory-Efficient 2D/3D Shape Assembly of Robot Swarms 

**Title (ZH)**: 机器人 swarm 的高效 2D/3D 形状装配 

**Authors**: Shuoyu Yue, Pengpeng Li, Yang Xu, Kunrui Ze, Xingjian Long, Huazi Cao, Guibin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.26518)  

**Abstract**: Mean-shift-based approaches have recently emerged as the most effective methods for robot swarm shape assembly tasks. These methods rely on image-based representations of target shapes to compute local density gradients and perform mean-shift exploration, which constitute their core mechanism. However, such image representations incur substantial memory overhead, which can become prohibitive for high-resolution or 3D shapes. To overcome this limitation, we propose a memory-efficient tree map representation that hierarchically encodes user-specified shapes and is applicable to both 2D and 3D scenarios. Building on this representation, we design a behavior-based distributed controller that enables assignment-free shape assembly. Comparative 2D and 3D simulations against a state-of-the-art mean-shift algorithm demonstrate one to two orders of magnitude lower memory usage and two to three times faster shape entry while maintaining comparable uniformity. Finally, we validate the framework through physical experiments with 6 to 7 UAVs, confirming its real-world practicality. 

**Abstract (ZH)**: 基于均值漂移的方法近年来成为机器人 swarm 形状组装任务中最有效的手段。这些方法依赖于目标形状的图像表示来计算局部密度梯度并执行均值漂移探索，这是其核心机制。然而，这种图像表示会带来巨大的内存开销，对于高分辨率或3D形状可能成为限制因素。为克服这一限制，我们提出了一种内存高效的树状图表示方法，该方法可以分层编码用户指定的形状，并适用于2D和3D场景。基于此表示方法，我们设计了一种基于行为的分布式控制器，以实现无需分配的形状组装。与最先进的均值漂移算法的2D和3D仿真结果表明，该方法的内存使用量低一个到两个数量级，形状进入速度提高两到三倍，同时保持了相当均匀性。最后，通过使用6到7个无人机的物理实验验证了该框架的有效性，证实了其实用性。 

---
# Unwinding Rotations Reduces VR Sickness in Nonsimulated Immersive Telepresence 

**Title (ZH)**: 解开旋转减少非模拟沉浸式远程呈现中的VR恶心感 

**Authors**: Filip Kulisiewicz, Basak Sakcak, Evan G. Center, Juho Kalliokoski, Katherine J. Mimnaugh, Steven M. LaValle, Timo Ojala  

**Link**: [PDF](https://arxiv.org/pdf/2509.26439)  

**Abstract**: Immersive telepresence, when a user views the video stream of a $360^\circ$ camera in a remote environment using a Head Mounted Display (HMD), has great potential to improve the sense of being in a remote environment. In most cases of immersive robotic telepresence, the camera is mounted on a mobile robot which increases the portion of the environment that the remote user can explore. However, robot motions can induce unpleasant symptoms associated with Virtual Reality (VR) sickness, degrading the overall user experience. Previous research has shown that unwinding the rotations of the robot, that is, decoupling the rotations that the camera undergoes due to robot motions from what is seen by the user, can increase user comfort and reduce VR sickness. However, that work considered a virtual environment and a simulated robot. In this work, to test whether the same hypotheses hold when the video stream from a real camera is used, we carried out a user study $(n=36)$ in which the unwinding rotations method was compared against coupled rotations in a task completed through a panoramic camera mounted on a robotic arm. Furthermore, within an inspection task which involved translations and rotations in three dimensions, we tested whether unwinding the robot rotations impacted the performance of users. The results show that the users found the unwinding rotations method to be more comfortable and preferable, and that a reduced level of VR sickness can be achieved without a significant impact on task performance. 

**Abstract (ZH)**: 基于头戴式显示的浸入式远程 presence，当用户使用360°摄像头在远程环境中的视频流时，具有潜在能力提升远程环境的沉浸感。在大多数沉浸式机器人 presence 情景中，摄像头安装在移动机器人上，增加了远程用户可以探索的环境比例。然而，机器人运动可能会引起与虚拟现实 (VR) 不适相关的症状，降低整体用户体验。先前的研究表明，通过解开由于机器人运动引起的摄像头旋转与用户所见之间的耦合，可以增加用户舒适度并减少 VR 不适。然而，该工作考虑的是虚拟环境和模拟机器人。在此项工作中，为了测试当使用实际摄像头的视频流时，相同假设是否仍然成立，我们在完成通过机械臂上全景摄像头的任务中进行了用户研究（n=36），比较了解开旋转方法与耦合旋转方法。此外，在一项涉及三维平移和旋转的检测任务中，我们测试了解开机器人旋转是否影响用户性能。研究结果表明，用户认为解开旋转方法更舒适且更优，并且可以在不影响任务性能的情况下实现较低水平的 VR 不适。 

---
# Real-time Velocity Profile Optimization for Time-Optimal Maneuvering with Generic Acceleration Constraints 

**Title (ZH)**: 基于通用加速度约束的实时速度剖面优化以实现时间最优机动 

**Authors**: Mattia Piazza, Mattia Piccinini, Sebastiano Taddei, Francesco Biral, Enrico Bertolazzi  

**Link**: [PDF](https://arxiv.org/pdf/2509.26428)  

**Abstract**: The computation of time-optimal velocity profiles along prescribed paths, subject to generic acceleration constraints, is a crucial problem in robot trajectory planning, with particular relevance to autonomous racing. However, the existing methods either support arbitrary acceleration constraints at high computational cost or use conservative box constraints for computational efficiency. We propose FBGA, a new \underline{F}orward-\underline{B}ackward algorithm with \underline{G}eneric \underline{A}cceleration constraints, which achieves both high accuracy and low computation time. FBGA operates forward and backward passes to maximize the velocity profile in short, discretized path segments, while satisfying user-defined performance limits. Tested on five racetracks and two vehicle classes, FBGA handles complex, non-convex acceleration constraints with custom formulations. Its maneuvers and lap times closely match optimal control baselines (within $0.11\%$-$0.36\%$), while being up to three orders of magnitude faster. FBGA maintains high accuracy even with coarse discretization, making it well-suited for online multi-query trajectory planning. Our open-source \texttt{C++} implementation is available at: this https URL. 

**Abstract (ZH)**: 沿指定路径在通用加速度约束下的时间最优速度剖面计算是机器人轨迹规划中的关键问题，特别是在自主赛车领域尤为重要。然而，现有的方法要么在高计算成本下支持任意加速度约束，要么为了计算效率使用保守的盒状约束。我们提出了一种新的FBGA（Forward-Backward算法，通用加速度约束），在保证高精度的同时，实现了低计算时间。FBGA通过正向和反向 passes 在短的离散路径段中最大化速度剖面，同时满足用户定义的性能限制。在五个赛道和两类车辆上测试，FBGA能够处理由自定义公式表示的复杂非凸加速度约束，并且其操作和圈速与最优控制基准（在0.11%-0.36%以内）极为接近，同时比后者快三个数量级。即使离散度粗糙，FBGA也能保持高精度，使其适合在线多查询轨迹规划。我们的开源C++实现可在以下链接获取：this https URL。 

---
# Kinodynamic Motion Planning for Mobile Robot Navigation across Inconsistent World Models 

**Title (ZH)**: 跨不一致世界模型的移动机器人Kinodynamic运动规划 

**Authors**: Eric R. Damm, Thomas M. Howard  

**Link**: [PDF](https://arxiv.org/pdf/2509.26339)  

**Abstract**: Mobile ground robots lacking prior knowledge of an environment must rely on sensor data to develop a model of their surroundings. In these scenarios, consistent identification of obstacles and terrain features can be difficult due to noise and algorithmic shortcomings, which can make it difficult for motion planning systems to generate safe motions. One particular difficulty to overcome is when regions of the cost map switch between being marked as obstacles and free space through successive planning cycles. One potential solution to this, which we refer to as Valid in Every Hypothesis (VEH), is for the planning system to plan motions that are guaranteed to be safe through a history of world models. Another approach is to track a history of world models, and adjust node costs according to the potential penalty of needing to reroute around previously hazardous areas. This work discusses three major iterations on this idea. The first iteration, called PEH, invokes a sub-search for every node expansion that crosses through a divergence point in the world models. The second and third iterations, called GEH and GEGRH respectively, defer the sub-search until after an edge expands into the goal region. GEGRH uses an additional step to revise the graph based on divergent nodes in each world. Initial results showed that, although PEH and GEH find more optimistic solutions than VEH, they are unable to generate solutions in less than one-second, which exceeds our requirements for field deployment. Analysis of results from a field experiment in an unstructured, off-road environment on a Clearpath Robotics Warthog UGV indicate that GEGRH finds lower cost trajectories and has faster average planning times than VEH. Compared to single-hypothesis (SH) search, where only the latest world model is considered, GEGRH generates more conservative plans with a small increase in average planning time. 

**Abstract (ZH)**: 基于每个假设有效的移动地面机器人规划方法：GEGRH 

---
# Side Scan Sonar-based SLAM for Autonomous Algae Farm Monitoring 

**Title (ZH)**: 侧扫声纳基于SLAM的自主藻类农场监测 

**Authors**: Julian Valdez, Ignacio Torroba, John Folkesson, Ivan Stenius  

**Link**: [PDF](https://arxiv.org/pdf/2509.26121)  

**Abstract**: The transition of seaweed farming to an alternative food source on an industrial scale relies on automating its processes through smart farming, equivalent to land agriculture. Key to this process are autonomous underwater vehicles (AUVs) via their capacity to automate crop and structural inspections. However, the current bottleneck for their deployment is ensuring safe navigation within farms, which requires an accurate, online estimate of the AUV pose and map of the infrastructure. To enable this, we propose an efficient side scan sonar-based (SSS) simultaneous localization and mapping (SLAM) framework that exploits the geometry of kelp farms via modeling structural ropes in the back-end as sequences of individual landmarks from each SSS ping detection, instead of combining detections into elongated representations. Our method outperforms state of the art solutions in hardware in the loop (HIL) experiments on a real AUV survey in a kelp farm. The framework and dataset can be found at this https URL. 

**Abstract (ZH)**: 大型海藻养殖向替代食物来源的转型依赖于通过智能养殖自动化其过程，相当于陆地农业。这一过程的关键是自主水下车辆（AUV）通过其自动化作物和结构检查的能力。然而，目前部署的瓶颈在于确保在农场内安全导航，这需要准确的在线姿态估计和基础设施的地图。为此，我们提出了一种有效的基于侧扫声呐（SSS）的同步定位与 mapping（SLAM）框架，该框架通过在后端将结构绳索建模为每个 SSS 回波检测的个体地标序列，而不是将检测合并为延长的表示，利用了海藻农场的几何结构。我们的方法在真实 AUV 在海藻农场进行的硬件在环（HIL）实验中优于当前最先进的解决方案。该框架和数据集可访问此处：this https URL。 

---
# On the Conic Complementarity of Planar Contacts 

**Title (ZH)**: 平面接触的圆锥互补性 

**Authors**: Yann de Mont-Marin, Louis Montaut, Jean Ponce, Martial Hebert, Justin Carpentier  

**Link**: [PDF](https://arxiv.org/pdf/2509.25999)  

**Abstract**: We present a unifying theoretical result that connects two foundational principles in robotics: the Signorini law for point contacts, which underpins many simulation methods for preventing object interpenetration, and the center of pressure (also known as the zero-moment point), a key concept used in, for instance, optimization-based locomotion control. Our contribution is the planar Signorini condition, a conic complementarity formulation that models general planar contacts between rigid bodies. We prove that this formulation is equivalent to enforcing the punctual Signorini law across an entire contact surface, thereby bridging the gap between discrete and continuous contact models. A geometric interpretation reveals that the framework naturally captures three physical regimes -sticking, separating, and tilting-within a unified complementarity structure. This leads to a principled extension of the classical center of pressure, which we refer to as the extended center of pressure. By establishing this connection, our work provides a mathematically consistent and computationally tractable foundation for handling planar contacts, with implications for both the accurate simulation of contact dynamics and the design of advanced control and optimization algorithms in locomotion and manipulation. 

**Abstract (ZH)**: 我们提出一个统一的理论结果，将机器人学中的两大基础原则—— SIGNORINI 法则（用于点接触）和中心压力（零力点），即用于防止物体穿插的许多模拟方法的基础，以及关键概念质心压力在比如基于优化的运动控制中的应用——联系起来。我们的贡献是在刚体之间模拟一般平面接触的平面 SIGNORINI 条件——一种锥互补形式。我们证明这种形式等同于在整个接触表面上强制实施 punctual SIGNORINI 法则，从而弥合了离散和连续接触模型之间的差距。几何解释表明，这种框架自然地捕捉了三种物理机制——黏着、分离和倾斜——并在统一的互补结构中。这导致了一个经典质心压力的原理性扩展，我们称之为扩展中心压力。通过建立这种联系，我们的工作为处理平面接触提供了一个数学上一致且计算上可行的基础，这对接触动力学的准确建模以及运动和操作中高级控制和优化算法的设计都有着重要意义。 

---
# State Estimation for Compliant and Morphologically Adaptive Robots 

**Title (ZH)**: 顺应性和形态适应性机器人状态估计 

**Authors**: Valentin Yuryev, Max Polzin, Josie Hughes  

**Link**: [PDF](https://arxiv.org/pdf/2509.25945)  

**Abstract**: Locomotion robots with active or passive compliance can show robustness to uncertain scenarios, which can be promising for agricultural, research and environmental industries. However, state estimation for these robots is challenging due to the lack of rigid-body assumptions and kinematic changes from morphing. We propose a method to estimate typical rigid-body states alongside compliance-related states, such as soft robot shape in different morphologies and locomotion modes. Our neural network-based state estimator uses a history of states and a mechanism to directly influence unreliable sensors. We test our framework on the GOAT platform, a robot capable of passive compliance and active morphing for extreme outdoor terrain. The network is trained on motion capture data in a novel compliance-centric frame that accounts for morphing-related states. Our method predicts shape-related measurements within 4.2% of the robot's size, velocities within 6.3% and 2.4% of the top linear and angular speeds, respectively, and orientation within 1.5 degrees. We also demonstrate a 300% increase in travel range during a motor malfunction when using our estimator for closed-loop autonomous outdoor operation. 

**Abstract (ZH)**: 具有主动或被动顺应性的移动机器人可以在不确定场景中表现出色，这有望应用于农业、研究和环境行业。然而，由于缺乏刚体假设和形态变化带来的运动学变化，这些机器人的状态估计具有挑战性。我们提出了一种方法来估计典型的刚体状态以及与顺应性相关的状态，如不同形态和运动模式下的软体机器人形状。基于神经网络的状态估计器使用状态的历史数据并具有一套机制，可以直接影响不可靠的传感器。我们在GOAT平台上测试了该框架，GOAT平台是一种能够在极端户外地形中表现出被动顺应性和主动变形能力的机器人。网络在一种新的顺应性为中心的新框架下进行了训练，该框架考虑了与形态变化相关的状态。我们的方法预测的形状相关测量值的误差在机器人尺寸的4.2%以内，速度分别在最大线性速度和角速度的6.3%和2.4%以内，姿态误差在1.5度以内。我们还展示了当使用我们的估计器进行闭环自主户外操作时，电机故障期间行程范围提高了300%。 

---
# TacRefineNet: Tactile-Only Grasp Refinement Between Arbitrary In-Hand Object Poses 

**Title (ZH)**: 基于触觉的手持物体任意姿态下的抓取 refinement 网络 

**Authors**: Shuaijun Wang, Haoran Zhou, Diyun Xiang, Yangwei You  

**Link**: [PDF](https://arxiv.org/pdf/2509.25746)  

**Abstract**: Despite progress in both traditional dexterous grasping pipelines and recent Vision-Language-Action (VLA) approaches, the grasp execution stage remains prone to pose inaccuracies, especially in long-horizon tasks, which undermines overall performance. To address this "last-mile" challenge, we propose TacRefineNet, a tactile-only framework that achieves fine in-hand pose refinement of known objects in arbitrary target poses using multi-finger fingertip sensing. Our method iteratively adjusts the end-effector pose based on tactile feedback, aligning the object to the desired configuration. We design a multi-branch policy network that fuses tactile inputs from multiple fingers along with proprioception to predict precise control updates. To train this policy, we combine large-scale simulated data from a physics-based tactile model in MuJoCo with real-world data collected from a physical system. Comparative experiments show that pretraining on simulated data and fine-tuning with a small amount of real data significantly improves performance over simulation-only training. Extensive real-world experiments validate the effectiveness of the method, achieving millimeter-level grasp accuracy using only tactile input. To our knowledge, this is the first method to enable arbitrary in-hand pose refinement via multi-finger tactile sensing alone. Project website is available at this https URL 

**Abstract (ZH)**: 尽管在传统灵巧抓取管道和近期视觉-语言-动作（VLA）方法方面取得了进展，但在执行抓取阶段仍容易出现姿态不准确的问题，尤其是在远期任务中，这会阻碍整体性能。为应对这一“最后一公里”挑战，我们提出TacRefineNet，这是一种仅依赖触觉的框架，利用多指指尖感知实现对任意目标姿态已知物体的精细在手姿态校正。该方法根据触觉反馈迭代调整末端执行器姿态，使物体与所需的配置对齐。我们设计了一个多分支策略网络，将多个手指的触觉输入与本体感受融合，以预测精确的控制更新。为训练该策略，我们将基于物理模型的大量模拟数据（来自MuJoCo）与来自实际系统的实时数据结合使用。对比实验表明，仅通过模拟数据预训练并在少量实际数据上进行微调，可以显著提高性能，优于仅基于模拟训练。大量现实世界的实验验证了该方法的有效性，仅通过触觉输入实现了毫米级的抓取精度。据我们所知，这是首个通过单一多指触觉感知实现任意在手姿态校正的方法。项目网站可访问此链接。 

---
# SRMP: Search-Based Robot Motion Planning Library 

**Title (ZH)**: 基于搜索的机器人运动规划库SRMP 

**Authors**: Itamar Mishani, Yorai Shaoul, Ramkumar Natarajan, Jiaoyang Li, Maxim Likhachev  

**Link**: [PDF](https://arxiv.org/pdf/2509.25352)  

**Abstract**: Motion planning is a critical component in any robotic system. Over the years, powerful tools like the Open Motion Planning Library (OMPL) have been developed, offering numerous motion planning algorithms. However, existing frameworks often struggle to deliver the level of predictability and repeatability demanded by high-stakes applications -- ranging from ensuring safety in industrial environments to the creation of high-quality motion datasets for robot learning. Complementing existing tools, we introduce SRMP (Search-based Robot Motion Planning), a new software framework tailored for robotic manipulation. SRMP distinguishes itself by generating consistent and reliable trajectories, and is the first software tool to offer motion planning algorithms for multi-robot manipulation tasks. SRMP easily integrates with major simulators, including MuJoCo, Sapien, Genesis, and PyBullet via a Python and C++ API. SRMP includes a dedicated MoveIt! plugin that enables immediate deployment on robot hardware and seamless integration with existing pipelines. Through extensive evaluations, we demonstrate in this paper that SRMP not only meets the rigorous demands of industrial and safety-critical applications but also sets a new standard for consistency in motion planning across diverse robotic systems. Visit this http URL for SRMP documentation and tutorials. 

**Abstract (ZH)**: 基于搜索的机器人运动规划SRMP：一种针对机器人操作的新软件框架 

---
# Preemptive Spatiotemporal Trajectory Adjustment for Heterogeneous Vehicles in Highway Merging Zones 

**Title (ZH)**: 高速公路汇合区异质车辆的预emption时空轨迹调整 

**Authors**: Yuan Li, Xiaoxue Xu, Xiang Dong, Junfeng Hao, Tao Li, Sana Ullaha, Chuangrui Huang, Junjie Niu, Ziyan Zhao, Ting Peng  

**Link**: [PDF](https://arxiv.org/pdf/2509.25929)  

**Abstract**: Aiming at the problem of driver's perception lag and low utilization efficiency of space-time resources in expressway ramp confluence area, based on the preemptive spatiotemporal trajectory Adjustment system, from the perspective of coordinating spatiotemporal resources, the reasonable value of safe space-time distance in trajectory pre-preparation is quantitatively analyzed. The minimum safety gap required for ramp vehicles to merge into the mainline is analyzed by introducing double positioning error and spatiotemporal trajectory tracking error. A merging control strategy for autonomous driving heterogeneous vehicles is proposed, which integrates vehicle type, driving intention, and safety spatiotemporal distance. The specific confluence strategies of ramp target vehicles and mainline cooperative vehicles under different vehicle types are systematically expounded. A variety of traffic flow and speed scenarios are used for full combination simulation. By comparing the time-position-speed diagram, the vehicle operation characteristics and the dynamic difference of confluence are qualitatively analyzed, and the average speed and average delay are used as the evaluation indices to quantitatively evaluate the performance advantages of the preemptive cooperative confluence control strategy. The results show that the maximum average delay improvement rates of mainline and ramp vehicles are 90.24 % and 74.24 %, respectively. The proposed strategy can effectively avoid potential vehicle conflicts and emergency braking behaviors, improve driving safety in the confluence area, and show significant advantages in driving stability and overall traffic efficiency optimization. 

**Abstract (ZH)**: 基于预emption时空轨迹调整系统的匝道合流区时空资源协调控制研究 

---
# Integrator Forwading Design for Unicycles with Constant and Actuated Velocity in Polar Coordinates 

**Title (ZH)**: 极坐标系中具有恒定和可控速度单轮车的积分前向设计 

**Authors**: Miroslav Krstic, Velimir Todorovski, Kwang Hak Kim, Alessandro Astolfi  

**Link**: [PDF](https://arxiv.org/pdf/2509.25579)  

**Abstract**: In a companion paper, we present a modular framework for unicycle stabilization in polar coordinates that provides smooth steering laws through backstepping. Surprisingly, the same problem also allows the application of integrator forwarding. In this work, we leverage this feature and construct new smooth steering laws together with control Lyapunov functions (CLFs), expanding the set of CLFs available for inverse optimal control design. In the case of constant forward velocity (Dubins car), backstepping produces finite-time (deadbeat) parking, and we show that integrator forwarding yields the very same class of solutions. This reveals a fundamental connection between backstepping and forwarding in addressing both the unicycle and, the Dubins car parking problems. 

**Abstract (ZH)**: 伴随论文中，我们提出了一种基于极坐标下的单轮车稳定模块化框架，通过递归回步方法提供了平滑的转向法则。令人意外的是，相同的问题也允许应用积分前馈方法。在本工作中，我们利用这一特性，构建新的平滑转向法则与控制李亚普诺夫函数（CLFs），扩展了可用于逆最优控制设计的CLFs集合。对于恒定前进速度的情况（杜宾车），递归回步方法产生有限时间（瞬态）停车，并且我们证明积分前馈方法同样产生同一类解。这揭示了递归回步方法与前馈方法在解决单轮车和杜宾车停车问题时的基本联系。 

---
# Modular Design of Strict Control Lyapunov Functions for Global Stabilization of the Unicycle in Polar Coordinates 

**Title (ZH)**: 极坐标系中独轮车全局稳定性的严格控制李亚普诺夫函数模块化设计 

**Authors**: Velimir Todorovski, Kwang Hak Kim, Miroslav Krstic  

**Link**: [PDF](https://arxiv.org/pdf/2509.25575)  

**Abstract**: Since the mid-1990s, it has been known that, unlike in Cartesian form where Brockett's condition rules out static feedback stabilization, the unicycle is globally asymptotically stabilizable by smooth feedback in polar coordinates. In this note, we introduce a modular framework for designing smooth feedback laws that achieve global asymptotic stabilization in polar coordinates. These laws are bidirectional, enabling efficient parking maneuvers, and are paired with families of strict control Lyapunov functions (CLFs) constructed in a modular fashion. The resulting CLFs guarantee global asymptotic stability with explicit convergence rates and include barrier variants that yield "almost global" stabilization, excluding only zero-measure subsets of the rotation manifolds. The strictness of the CLFs is further leveraged in our companion paper, where we develop inverse-optimal redesigns with meaningful cost functions and infinite gain margins. 

**Abstract (ZH)**: 自20世纪90年代中期以来，人们知道，与笛卡尔坐标系中布罗克特条件排除静态反馈稳定化不同，独轮车可以通过光滑反馈在极坐标系中实现全局渐近稳定化。本文介绍了一种模块化框架，用于设计能够在极坐标系中实现全局渐近稳定化的光滑反馈法则。这些法则具有双向性，便于高效执行停车操作，并且与模块化构造的严格控制李雅普(Ny)诺夫函数(CLFs)配对。这些CLFs确保全局渐近稳定，并具有显式的收敛速率，包括作为旋转流形上零测度子集的补充，可实现“几乎全局”稳定化。在我们姊妹论文中，进一步利用这些CLFs的严格性，开发了具有实际意义的成本函数和无限增益边际的逆最优重新设计方法。 

---
# Infrastructure Sensor-enabled Vehicle Data Generation using Multi-Sensor Fusion for Proactive Safety Applications at Work Zone 

**Title (ZH)**: 基于多传感器融合的工作区主动安全应用的基础设施传感器启用车辆数据生成 

**Authors**: Suhala Rabab Saba, Sakib Khan, Minhaj Uddin Ahmad, Jiahe Cao, Mizanur Rahman, Li Zhao, Nathan Huynh, Eren Erman Ozguven  

**Link**: [PDF](https://arxiv.org/pdf/2509.25452)  

**Abstract**: Infrastructure-based sensing and real-time trajectory generation show promise for improving safety in high-risk roadway segments such as work zones, yet practical deployments are hindered by perspective distortion, complex geometry, occlusions, and costs. This study tackles these barriers by integrating roadside camera and LiDAR sensors into a cosimulation environment to develop a scalable, cost-effective vehicle detection and localization framework, and employing a Kalman Filter-based late fusion strategy to enhance trajectory consistency and accuracy. In simulation, the fusion algorithm reduced longitudinal error by up to 70 percent compared to individual sensors while preserving lateral accuracy within 1 to 3 meters. Field validation in an active work zone, using LiDAR, a radar-camera rig, and RTK-GPS as ground truth, demonstrated that the fused trajectories closely match real vehicle paths, even when single-sensor data are intermittent or degraded. These results confirm that KF based sensor fusion can reliably compensate for individual sensor limitations, providing precise and robust vehicle tracking capabilities. Our approach thus offers a practical pathway to deploy infrastructure-enabled multi-sensor systems for proactive safety measures in complex traffic environments. 

**Abstract (ZH)**: 基于基础设施的传感器与实时轨迹生成在高风险道路路段（如工作区）提高安全性方面具有潜力，但实际部署受到视角失真、复杂几何结构、遮挡和成本的阻碍。本研究通过将路边摄像头和LiDAR传感器整合到协同仿真环境中，开发了一种可扩展且成本效益高的车辆检测与定位框架，并采用卡尔曼滤波器为基础的后融合策略以增强轨迹的一致性和准确性。在仿真中，融合算法将纵向误差降低了多达70%，同时保持了横向准确性在1至3米范围内。在现场验证中，在一个活跃的工作区内，使用LiDAR、雷达-摄像头组合和RTK-GPS作为地面真实值，结果显示融合轨迹与实际车辆路径高度一致，即使单传感器数据间断或降级也是如此。这些结果证实了基于卡尔曼滤波器的传感器融合可以可靠地弥补单传感器的局限性，提供精确且稳健的车辆追踪能力。因此，本研究为在复杂交通环境中部署基础设施支持的多传感器系统以实现主动安全性提供了一条实用途径。 

---
