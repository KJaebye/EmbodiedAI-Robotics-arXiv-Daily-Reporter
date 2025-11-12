# Safe and Optimal Learning from Preferences via Weighted Temporal Logic with Applications in Robotics and Formula 1 

**Title (ZH)**: 安全且最优地从偏好中学习：基于加权时序逻辑的方法及其在机器人技术与一级方程式中的应用 

**Authors**: Ruya Karagulle, Cristian-Ioan Vasile, Necmiye Ozay  

**Link**: [PDF](https://arxiv.org/pdf/2511.08502)  

**Abstract**: Autonomous systems increasingly rely on human feedback to align their behavior, expressed as pairwise comparisons, rankings, or demonstrations. While existing methods can adapt behaviors, they often fail to guarantee safety in safety-critical domains. We propose a safety-guaranteed, optimal, and efficient approach to solve the learning problem from preferences, rankings, or demonstrations using Weighted Signal Temporal Logic (WSTL). WSTL learning problems, when implemented naively, lead to multi-linear constraints in the weights to be learned. By introducing structural pruning and log-transform procedures, we reduce the problem size and recast the problem as a Mixed-Integer Linear Program while preserving safety guarantees. Experiments on robotic navigation and real-world Formula 1 data demonstrate that the method effectively captures nuanced preferences and models complex task objectives. 

**Abstract (ZH)**: 自主系统日益依赖人类反馈来调整其行为，这些反馈可以是成对比较、排名或示范。虽然现有方法可以适应行为，但在安全关键领域往往无法确保安全性。我们提出了一种安全保证、最优且高效的基于加权信号时序逻辑（WSTL）从偏好、排名或示范中学习的方法。通过引入结构化剪枝和对数变换程序，我们减少了问题规模，并将问题重新表述为混合整数线性规划问题，同时保留了安全保证。实验结果显示，该方法能够有效捕捉细腻的偏好并建模复杂的任务目标。 

---
# A CODECO Case Study and Initial Validation for Edge Orchestration of Autonomous Mobile Robots 

**Title (ZH)**: CODECO 案例研究及边缘 orchestration 自主移动机器人初步验证 

**Authors**: H. Zhu, T. Samizadeh, R. C. Sofia  

**Link**: [PDF](https://arxiv.org/pdf/2511.08354)  

**Abstract**: Autonomous Mobile Robots (AMRs) increasingly adopt containerized micro-services across the Edge-Cloud continuum. While Kubernetes is the de-facto orchestrator for such systems, its assumptions of stable networks, homogeneous resources, and ample compute capacity do not fully hold in mobile, resource-constrained robotic environments.
This paper describes a case study on smart-manufacturing AMRs and performs an initial comparison between CODECO orchestration and standard Kubernetes using a controlled KinD environment. Metrics include pod deployment and deletion times, CPU and memory usage, and inter-pod data rates. The observed results indicate that CODECO offers reduced CPU consumption and more stable communication patterns, at the cost of modest memory overhead (10-15%) and slightly increased pod lifecycle latency due to secure overlay initialization. 

**Abstract (ZH)**: 自主移动机器人（AMRs）越来越多地采用边缘-云连续体中的容器化微服务。虽然Kubernetes是此类系统的事实上的编排器，但其对稳定网络、均质资源和充足计算能力的假设并不完全适用于移动且资源受限的机器人环境。

本文描述了一项关于智能制造AMRs的案例研究，并在受控的KinD环境中对CODECO编排与标准Kubernetes进行了初步比较。评估指标包括部署和删除Pod的时间、CPU和内存使用情况，以及Pod之间数据传输率。观察结果显示，CODECO在消耗CPU方面有所减少，并提供了更稳定的通信模式，但伴随着10-15%的内存开销增加和轻微增加的Pod生命周期延迟（由于安全覆盖网络初始化）。 

---
# Real-Time Performance Analysis of Multi-Fidelity Residual Physics-Informed Neural Process-Based State Estimation for Robotic Systems 

**Title (ZH)**: 基于多保真剩余物理知情神经过程的状态估计的机器人系统实时性能分析 

**Authors**: Devin Hunter, Chinwendu Enyioha  

**Link**: [PDF](https://arxiv.org/pdf/2511.08231)  

**Abstract**: Various neural network architectures are used in many of the state-of-the-art approaches for real-time nonlinear state estimation. With the ever-increasing incorporation of these data-driven models into the estimation domain, model predictions with reliable margins of error are a requirement -- especially for safety-critical applications. This paper discusses the application of a novel real-time, data-driven estimation approach based on the multi-fidelity residual physics-informed neural process (MFR-PINP) toward the real-time state estimation of a robotic system. Specifically, we address the model-mismatch issue of selecting an accurate kinematic model by tasking the MFR-PINP to also learn the residuals between simple, low-fidelity predictions and complex, high-fidelity ground-truth dynamics. To account for model uncertainty present in a physical implementation, robust uncertainty guarantees from the split conformal (SC) prediction framework are modeled in the training and inference paradigms. We provide implementation details of our MFR-PINP-based estimator for a hybrid online learning setting to validate our model's usage in real-time applications. Experimental results of our approach's performance in comparison to the state-of-the-art variants of the Kalman filter (i.e. unscented Kalman filter and deep Kalman filter) in estimation scenarios showed promising results for the MFR-PINP model as a viable option in real-time estimation tasks. 

**Abstract (ZH)**: 基于多保真度残差物理知情神经过程的实时数据驱动状态估计算法在机器人系统的应用 

---
# AVOID-JACK: Avoidance of Jackknifing for Swarms of Long Heavy Articulated Vehicles 

**Title (ZH)**: AVOID-JACK: 避免套索法对长重型 articulated 车辆群的处理 

**Authors**: Adrian Schönnagel, Michael Dubé, Christoph Steup, Felix Keppler, Sanaz Mostaghim  

**Link**: [PDF](https://arxiv.org/pdf/2511.08016)  

**Abstract**: This paper presents a novel approach to avoiding jackknifing and mutual collisions in Heavy Articulated Vehicles (HAVs) by leveraging decentralized swarm intelligence. In contrast to typical swarm robotics research, our robots are elongated and exhibit complex kinematics, introducing unique challenges. Despite its relevance to real-world applications such as logistics automation, remote mining, airport baggage transport, and agricultural operations, this problem has not been addressed in the existing literature.
To tackle this new class of swarm robotics problems, we propose a purely reaction-based, decentralized swarm intelligence strategy tailored to automate elongated, articulated vehicles. The method presented in this paper prioritizes jackknifing avoidance and establishes a foundation for mutual collision avoidance. We validate our approach through extensive simulation experiments and provide a comprehensive analysis of its performance. For the experiments with a single HAV, we observe that for 99.8% jackknifing was successfully avoided and that 86.7% and 83.4% reach their first and second goals, respectively. With two HAVs interacting, we observe 98.9%, 79.4%, and 65.1%, respectively, while 99.7% of the HAVs do not experience mutual collisions. 

**Abstract (ZH)**: 本文提出了一种通过利用分散化的 swarm 智能来避免重型铰接车辆（HAVs）侧翻和相互碰撞的新型方法。为了应对这类新的 swarm 机器人问题，我们提出了一种专门为长形和具有复杂运动学的铰接车辆设计的纯反应式分散化 swarm 智能策略。该方法优先避免侧翻并为相互碰撞的避免奠定基础。通过广泛的仿真实验验证了该方法的有效性，并对其性能进行了全面分析。对于单一 HAV 的实验，发现99.8% 的侧翻情况被成功避免，并且分别有86.7% 和83.4% 达到第一个和第二个目标。对于两个 HAV 交互的实验，分别有98.9%、79.4% 和65.1%，同时99.7% 的HAV 没有经历相互碰撞。 

---
# A Two-Layer Electrostatic Film Actuator with High Actuation Stress and Integrated Brake 

**Title (ZH)**: 具有高驱动应力和集成制动的两层静电薄膜驱动器 

**Authors**: Huacen Wang, Hongqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.08005)  

**Abstract**: Robotic systems driven by conventional motors often suffer from challenges such as large mass, complex control algorithms, and the need for additional braking mechanisms, which limit their applications in lightweight and compact robotic platforms. Electrostatic film actuators offer several advantages, including thinness, flexibility, lightweight construction, and high open-loop positioning accuracy. However, the actuation stress exhibited by conventional actuators in air still needs improvement, particularly for the widely used three-phase electrode design. To enhance the output performance of actuators, this paper presents a two-layer electrostatic film actuator with an integrated brake. By alternately distributing electrodes on both the top and bottom layers, a smaller effective electrode pitch is achieved under the same fabrication constraints, resulting in an actuation stress of approximately 241~N/m$^2$, representing a 90.5\% improvement over previous three-phase actuators operating in air. Furthermore, its integrated electrostatic adhesion mechanism enables load retention under braking mode. Several demonstrations, including a tug-of-war between a conventional single-layer actuator and the proposed two-layer actuator, a payload operation, a one-degree-of-freedom robotic arm, and a dual-mode gripper, were conducted to validate the actuator's advantageous capabilities in both actuation and braking modes. 

**Abstract (ZH)**: 基于传统电机驱动的机器人系统常常面临质量大、控制算法复杂以及需要额外制动机制等挑战，这限制了它们在轻量紧凑型机器人平台中的应用。静电薄膜执行器具有薄、柔、轻以及开环定位精度高的优点。然而，传统执行器在空气中的驱动应力仍需提升，特别是对于广泛使用的三相电极设计。为了增强执行器的输出性能，本文提出了一种具有集成制动机制的两层静电薄膜执行器。通过在上下层交替分布电极，使其在相同制备约束条件下实现了更小的有效电极间距，在空气中获得了约241~N/m$^2$的驱动应力，相比之前的三相执行器提升了90.5%。此外，其集成的静电吸附机制能够在制动模式下保持负载。通过多项演示，包括传统单层执行器与所提双层执行器的拔河对比、负载操作、单自由度机械臂和双模式夹爪等，验证了该执行器在驱动和制动模式下的优越性能。 

---
# Effective Game-Theoretic Motion Planning via Nested Search 

**Title (ZH)**: 基于嵌套搜索的有效博弈论运动规划 

**Authors**: Avishav Engle, Andrey Zhitnikov, Oren Salzman, Omer Ben-Porat, Kiril Solovey  

**Link**: [PDF](https://arxiv.org/pdf/2511.08001)  

**Abstract**: To facilitate effective, safe deployment in the real world, individual robots must reason about interactions with other agents, which often occur without explicit communication. Recent work has identified game theory, particularly the concept of Nash Equilibrium (NE), as a key enabler for behavior-aware decision-making. Yet, existing work falls short of fully unleashing the power of game-theoretic reasoning. Specifically, popular optimization-based methods require simplified robot dynamics and tend to get trapped in local minima due to convexification. Other works that rely on payoff matrices suffer from poor scalability due to the explicit enumeration of all possible trajectories. To bridge this gap, we introduce Game-Theoretic Nested Search (GTNS), a novel, scalable, and provably correct approach for computing NEs in general dynamical systems. GTNS efficiently searches the action space of all agents involved, while discarding trajectories that violate the NE constraint (no unilateral deviation) through an inner search over a lower-dimensional space. Our algorithm enables explicit selection among equilibria by utilizing a user-specified global objective, thereby capturing a rich set of realistic interactions. We demonstrate the approach on a variety of autonomous driving and racing scenarios where we achieve solutions in mere seconds on commodity hardware. 

**Abstract (ZH)**: 基于博弈论的嵌套搜索方法：一种通用动力系统中纳什均衡的高效可证明正确计算方法 

---
# USV Obstacles Detection and Tracking in Marine Environments 

**Title (ZH)**: USV 海洋环境中的障碍物检测与跟踪 

**Authors**: Yara AlaaEldin, Enrico Simetti, Francesca Odone  

**Link**: [PDF](https://arxiv.org/pdf/2511.07950)  

**Abstract**: Developing a robust and effective obstacle detection and tracking system for Unmanned Surface Vehicle (USV) at marine environments is a challenging task. Research efforts have been made in this area during the past years by GRAAL lab at the university of Genova that resulted in a methodology for detecting and tracking obstacles on the image plane and, then, locating them in the 3D LiDAR point cloud. In this work, we continue on the developed system by, firstly, evaluating its performance on recently published marine datasets. Then, we integrate the different blocks of the system on ROS platform where we could test it in real-time on synchronized LiDAR and camera data collected in various marine conditions available in the MIT marine datasets. We present a thorough experimental analysis of the results obtained using two approaches; one that uses sensor fusion between the camera and LiDAR to detect and track the obstacles and the other uses only the LiDAR point cloud for the detection and tracking. In the end, we propose a hybrid approach that merges the advantages of both approaches to build an informative obstacles map of the surrounding environment to the USV. 

**Abstract (ZH)**: 在海洋环境中开发一种稳健有效的障碍检测与跟踪系统是具有挑战性的任务。Genova大学GRAAL实验室在过去几年中进行了这方面的研究工作，提出了一种在图像平面上检测和跟踪障碍物的方法，并进一步将它们定位在3D LiDAR点云中。在此基础上，我们首先评估了该系统在最近发布的海洋数据集上的性能。然后，我们将系统中的不同模块集成到ROS平台上，在MIT海洋数据集中同步的LiDAR和相机数据的不同海洋条件下进行实时测试。我们使用两种方法进行了详细的实验分析；一种方法结合了相机和LiDAR传感器融合来检测和跟踪障碍物，另一种方法仅使用LiDAR点云来检测和跟踪。最后，我们提出了一种混合方法，将两种方法的优势结合起来，为USV构建周边环境的信息障碍地图。 

---
# Local Path Planning with Dynamic Obstacle Avoidance in Unstructured Environments 

**Title (ZH)**: 不结构化环境中具有动态障碍物避让的局部路径规划 

**Authors**: Okan Arif Guvenkaya, Selim Ahmet Iz, Mustafa Unel  

**Link**: [PDF](https://arxiv.org/pdf/2511.07927)  

**Abstract**: Obstacle avoidance and path planning are essential for guiding unmanned ground vehicles (UGVs) through environments that are densely populated with dynamic obstacles. This paper develops a novel approach that combines tangentbased path planning and extrapolation methods to create a new decision-making algorithm for local path planning. In the assumed scenario, a UGV has a prior knowledge of its initial and target points within the dynamic environment. A global path has already been computed, and the robot is provided with waypoints along this path. As the UGV travels between these waypoints, the algorithm aims to avoid collisions with dynamic obstacles. These obstacles follow polynomial trajectories, with their initial positions randomized in the local map and velocities randomized between O and the allowable physical velocity limit of the robot, along with some random accelerations. The developed algorithm is tested in several scenarios where many dynamic obstacles move randomly in the environment. Simulation results show the effectiveness of the proposed local path planning strategy by gradually generating a collision free path which allows the robot to navigate safely between initial and the target locations. 

**Abstract (ZH)**: 基于切线的路径规划和外推方法相结合的无人地面车辆避障与路径规划新方法 

---
# Dual-MPC Footstep Planning for Robust Quadruped Locomotion 

**Title (ZH)**: 双模型预测控制足步规划以实现稳健的四足运动 

**Authors**: Byeong-Il Ham, Hyun-Bin Kim, Jeonguk Kang, Keun Ha Choi, Kyung-Soo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.07921)  

**Abstract**: In this paper, we propose a footstep planning strategy based on model predictive control (MPC) that enables robust regulation of body orientation against undesired body rotations by optimizing footstep placement. Model-based locomotion approaches typically adopt heuristic methods or planning based on the linear inverted pendulum model. These methods account for linear velocity in footstep planning, while excluding angular velocity, which leads to angular momentum being handled exclusively via ground reaction force (GRF). Footstep planning based on MPC that takes angular velocity into account recasts the angular momentum control problem as a dual-input approach that coordinates GRFs and footstep placement, instead of optimizing GRFs alone, thereby improving tracking performance. A mutual-feedback loop couples the footstep planner and the GRF MPC, with each using the other's solution to iteratively update footsteps and GRFs. The use of optimal solutions reduces body oscillation and enables extended stance and swing phases. The method is validated on a quadruped robot, demonstrating robust locomotion with reduced oscillations, longer stance and swing phases across various terrains. 

**Abstract (ZH)**: 基于模型预测控制的步足规划策略：考虑角速度的体姿态 robust 调节 

---
# A Comprehensive Experimental Characterization of Mechanical Layer Jamming Systems 

**Title (ZH)**: 全面的机械层卡滞系统实验表征 

**Authors**: Jessica Gumowski, Krishna Manaswi Digumarti, David Howard  

**Link**: [PDF](https://arxiv.org/pdf/2511.07882)  

**Abstract**: Organisms in nature, such as Cephalopods and Pachyderms, exploit stiffness modulation to achieve amazing dexterity in the control of their appendages. In this paper, we explore the phenomenon of layer jamming, which is a popular stiffness modulation mechanism that provides an equivalent capability for soft robots. More specifically, we focus on mechanical layer jamming, which we realise through two-layer multi material structure with tooth-like protrusions. We identify key design parameters for mechanical layer jamming systems, including the ability to modulate stiffness, and perform a variety of comprehensive tests placing the specimens under bending and torsional loads to understand the influence of our selected design parameters (mainly tooth geometry) on the performance of the jammed structures. We note the ability of these structures to produce a peak change in stiffness of 5 times in bending and 3.2 times in torsion. We also measure the force required to separate the two jammed layers, an often ignored parameter in the study of jamming-induced stiffness change. This study aims to shed light on the principled design of mechanical layer jammed systems and guide researchers in the selection of appropriate designs for their specific application domains. 

**Abstract (ZH)**: 自然界中的生物，如头足类和大型哺乳动物，利用刚度调节实现其附肢控制的惊人灵活性。本文探讨了层状阻塞现象，这是一种常见的刚度调节机制，为软体机器人提供了等效功能。具体而言，我们关注机械层状阻塞，通过两层多材料结构结合齿状突起实现。我们确定了机械层状阻塞系统的关键设计参数，包括刚度调节能力，并通过一系列综合测试，将标本置于弯曲和扭转负载下，以理解选定设计参数（主要是齿状结构几何）对阻塞结构性能的影响。我们注意到这些结构在弯曲时可产生5倍的刚度峰值变化，在扭转时可产生3.2倍的刚度峰值变化。我们还测量了分离两层阻塞所需的力，这是一个在研究阻塞引起的刚度变化中经常被忽略的参数。本研究旨在阐明机械层状阻塞系统的原理设计，并指导研究人员为其特定的应用领域选择合适的结构设计。 

---
# Occlusion-Aware Ground Target Search by a UAV in an Urban Environment 

**Title (ZH)**: 城市环境中无人机基于遮挡意识的地面目标搜索 

**Authors**: Collin Hague, Artur Wolek  

**Link**: [PDF](https://arxiv.org/pdf/2511.07822)  

**Abstract**: This paper considers the problem of searching for a point of interest (POI) moving along an urban road network with an uncrewed aerial vehicle (UAV). The UAV is modeled as a variable-speed Dubins vehicle with a line-of-sight sensor in an urban environment that may occlude the sensor's view of the POI. A search strategy is proposed that exploits a probabilistic visibility volume (VV) to plan its future motion with iterative deepening $A^\ast$. The probabilistic VV is a time-varying three-dimensional representation of the sensing constraints for a particular distribution of the POI's state. To find the path most likely to view the POI, the planner uses a heuristic to optimistically estimate the probability of viewing the POI over a time horizon. The probabilistic VV is max-pooled to create a variable-timestep planner that reduces the search space and balances long-term and short-term planning. The proposed path planning method is compared to prior work with a Monte-Carlo simulation and is shown to outperform the baseline methods in cluttered environments when the UAV's sensor has a higher false alarm probability. 

**Abstract (ZH)**: 本文考虑了使用无人驾驶航空车辆（UAV）在城市道路网络中搜寻沿道路移动的兴趣点（POI）的问题。UAV被建模为具有视线传感器的可变速度杜宾车，在可能遮挡传感器视域的城市环境中对POI进行搜索。提出了一种策略，该策略利用概率可视体积（VV）并通过迭代加深$A^\ast$算法规划其未来的运动。概率可视体积是一个时间变化的三维表示，体现了特定POI状态分布下的传感约束。为了找到最有可能观察到POI的路径，规划器使用启发式方法乐观地估计在时间跨度内观察到POI的概率。概率可视体积通过最大池化创建了一个可变时间步长的规划器，从而减少搜索空间并平衡长期和短期规划。与之前的路径规划方法进行了 Monte-Carlo 模拟对比，并在UAV传感器具有较高误报概率的复杂环境中展示了所提出方法的优越性。 

---
# Benchmarking Resilience and Sensitivity of Polyurethane-Based Vision-Based Tactile Sensors 

**Title (ZH)**: 基于聚氨酯的视觉触觉传感器的鲁棒性和敏感性基准测试 

**Authors**: Benjamin Davis, Hannah Stuart  

**Link**: [PDF](https://arxiv.org/pdf/2511.07797)  

**Abstract**: Vision-based tactile sensors (VBTSs) are a promising technology for robots, providing them with dense signals that can be translated into an understanding of normal and shear load, contact region, texture classification, and more. However, existing VBTS tactile surfaces make use of silicone gels, which provide high sensitivity but easily deteriorate from loading and surface wear. We propose that polyurethane rubber, used for high-load applications like shoe soles, rubber wheels, and industrial gaskets, may provide improved physical gel resilience, potentially at the cost of sensitivity. To compare the resilience and sensitivity of silicone and polyurethane VBTS gels, we propose a series of standard evaluation benchmarking protocols. Our resilience tests assess sensor durability across normal loading, shear loading, and abrasion. For sensitivity, we introduce model-free assessments of force and spatial sensitivity to directly measure the physical capabilities of each gel without effects introduced from data and model quality. Finally, we include a bottle cap loosening and tightening demonstration as an example where polyurethane gels provide an advantage over their silicone counterparts. 

**Abstract (ZH)**: 基于视觉的触觉传感器（VBTS）是机器人技术的一种有前景的技术，能够提供密集的信号以理解正压力、切向压力、接触区域、纹理分类等。然而，现有的VBTS触觉表面大多采用硅胶，虽然灵敏度高，但容易因负载和表面磨损而退化。我们提出使用聚氨酯橡胶作为材料，聚氨酯橡胶常用于鞋底、橡胶轮子和工业垫片等高负载应用，可能会提供更好的物理胶体韧性，尽管灵敏度可能会有所降低。为了比较硅胶和聚氨酯VBTS胶体的韧性与灵敏度，我们提出了系列标准评估基准测试协议。我们的韧性测试评估传感器在正压力、切向压力和磨损条件下的耐用性。对于灵敏度，我们引入了无模型评估方法，直接测量每种胶体的力学能力，不受数据质量和模型效果的影响。最后，我们通过瓶盖松紧试验示例展示了聚氨酯胶体相对于硅胶的优势。 

---
# Testing and Evaluation of Underwater Vehicle Using Hardware-In-The-Loop Simulation with HoloOcean 

**Title (ZH)**: 基于HoloOcean的硬件在环仿真的水下车辆测试与评估 

**Authors**: Braden Meyers, Joshua G. Mangelson  

**Link**: [PDF](https://arxiv.org/pdf/2511.07687)  

**Abstract**: Testing marine robotics systems in controlled environments before field tests is challenging, especially when acoustic-based sensors and control surfaces only function properly underwater. Deploying robots in indoor tanks and pools often faces space constraints that complicate testing of control, navigation, and perception algorithms at scale. Recent developments of high-fidelity underwater simulation tools have the potential to address these problems. We demonstrate the utility of the recently released HoloOcean 2.0 simulator with improved dynamics for torpedo AUV vehicles and a new ROS 2 interface. We have successfully demonstrated a Hardware-in-the-Loop (HIL) and Software-in-the-Loop (SIL) setup for testing and evaluating a CougUV torpedo autonomous underwater vehicle (AUV) that was built and developed in our lab. With this HIL and SIL setup, simulations are run in HoloOcean using a ROS 2 bridge such that simulated sensor data is sent to the CougUV (mimicking sensor drivers) and control surface commands are sent back to the simulation, where vehicle dynamics and sensor data are calculated. We compare our simulated results to real-world field trial results. 

**Abstract (ZH)**: 在可控环境中测试基于声学传感器和控制面的水下机器人系统在野外试验前的挑战性测试，尤其是在只有在水下这些传感器和控制面才能正常工作的情况下。将机器人部署在室内水箱和游泳池中通常会受到空间限制，这使得大规模测试控制、导航和感知算法变得复杂。最近开发的高保真水下模拟工具有可能解决这些问题。我们展示了最近发布的HoloOcean 2.0仿真器及其改进的鱼雷AUV动力学模型和新的ROS 2接口的实用性。我们成功地为在我们实验室设计和开发的CougUV鱼雷自主水下车辆（AUV）建立了一个硬件在环（HIL）和软件在环（SIL）测试与评估设置。通过这种方式，在HoloOcean中使用ROS 2桥运行仿真，发送模拟传感器数据到CougUV（模拟传感器驱动程序）并与模拟器互动，发送回控制面指令，在此过程中计算车辆动力学和传感器数据。我们将我们的模拟结果与实际野外试验结果进行了比较。 

---
# Work-in-Progress: Function-as-Subtask API Replacing Publish/Subscribe for OS-Native DAG Scheduling 

**Title (ZH)**: 工作进展：将函数作为子任务API替换发布/订阅进行OS原生DAG调度 

**Authors**: Takahiro Ishikawa-Aso, Atsushi Yano, Yutaro Kobayashi, Takumi Jin, Yuuki Takano, Shinpei Kato  

**Link**: [PDF](https://arxiv.org/pdf/2511.08297)  

**Abstract**: The Directed Acyclic Graph (DAG) task model for real-time scheduling finds its primary practical target in Robot Operating System 2 (ROS 2). However, ROS 2's publish/subscribe API leaves DAG precedence constraints unenforced: a callback may publish mid-execution, and multi-input callbacks let developers choose topic-matching policies. Thus preserving DAG semantics relies on conventions; once violated, the model collapses. We propose the Function-as-Subtask (FasS) API, which expresses each subtask as a function whose arguments/return values are the subtask's incoming/outgoing edges. By minimizing description freedom, DAG semantics is guaranteed at the API rather than by programmer discipline. We implement a DAG-native scheduler using FasS on a Rust-based experimental kernel and evaluate its semantic fidelity, and we outline design guidelines for applying FasS to Linux Linux sched_ext. 

**Abstract (ZH)**: 面向实时调度的有向无环图（DAG）任务模型在Robot Operating System 2（ROS 2）中找到了主要的实际目标。然而，ROS 2的发布/订阅API未能强制执行DAG的优先级约束：回调函数可能在执行过程中发布数据，多输入回调函数允许开发人员选择主题匹配策略。因此，保持DAG语义依赖于约定；一旦被违反，模型将失效。我们提出了一种Function-as-Subtask（FasS）API，通过将每个子任务表示为一个函数，其参数/返回值为子任务的入边/出边，从而最小化描述自由度，在API层面而非开发人员的自律保证DAG语义。我们使用基于Rust的实验内核实现了DAG原生调度器，并评估了其语义准确性，并概述了将FasS应用于Linux sched_ext的设计指南。 

---
# ARGUS: A Framework for Risk-Aware Path Planning in Tactical UGV Operations 

**Title (ZH)**: ARGUS：战术UGV操作中风险感知路径规划的框架 

**Authors**: Nuno Soares, António Grilo  

**Link**: [PDF](https://arxiv.org/pdf/2511.07565)  

**Abstract**: This thesis presents the development of ARGUS, a framework for mission planning for Unmanned Ground Vehicles (UGVs) in tactical environments. The system is designed to translate battlefield complexity and the commander's intent into executable action plans. To this end, ARGUS employs a processing pipeline that takes as input geospatial terrain data, military intelligence on existing threats and their probable locations, and mission priorities defined by the commander. Through a set of integrated modules, the framework processes this information to generate optimized trajectories that balance mission objectives against the risks posed by threats and terrain characteristics. A fundamental capability of ARGUS is its dynamic nature, which allows it to adapt plans in real-time in response to unforeseen events, reflecting the fluid nature of the modern battlefield. The system's interoperability were validated in a practical exercise with the Portuguese Army, where it was successfully demonstrated that the routes generated by the model can be integrated and utilized by UGV control systems. The result is a decision support tool that not only produces an optimal trajectory but also provides the necessary insights for its execution, thereby contributing to greater effectiveness and safety in the employment of autonomous ground systems. 

**Abstract (ZH)**: ARGUS：战术环境中无人地面车辆任务规划框架的发展 

---
