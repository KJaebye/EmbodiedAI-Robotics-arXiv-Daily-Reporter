# V*: An Efficient Motion Planning Algorithm for Autonomous Vehicles 

**Title (ZH)**: V*: 自动驾驶车辆高效路径规划算法 

**Authors**: Abdullah Zareh Andaryan, Michael G.H. Bell, Mohsen Ramezani, Glenn Geers  

**Link**: [PDF](https://arxiv.org/pdf/2508.06404)  

**Abstract**: Autonomous vehicle navigation in structured environments requires planners capable of generating time-optimal, collision-free trajectories that satisfy dynamic and kinematic constraints. We introduce V*, a graph-based motion planner that represents speed and direction as explicit state variables within a discretised space-time-velocity lattice. Unlike traditional methods that decouple spatial search from dynamic feasibility or rely on post-hoc smoothing, V* integrates both motion dimensions directly into graph construction through dynamic graph generation during search expansion. To manage the complexity of high-dimensional search, we employ a hexagonal discretisation strategy and provide formal mathematical proofs establishing optimal waypoint spacing and minimal node redundancy under constrained heading transitions for velocity-aware motion planning. We develop a mathematical formulation for transient steering dynamics in the kinematic bicycle model, modelling steering angle convergence with exponential behaviour, and deriving the relationship for convergence rate parameters. This theoretical foundation, combined with geometric pruning strategies that eliminate expansions leading to infeasible steering configurations, enables V* to evaluate dynamically admissible manoeuvres, ensuring each trajectory is physically realisable without further refinement. We further demonstrate V*'s performance in simulation studies with cluttered and dynamic environments involving moving obstacles, showing its ability to avoid conflicts, yield proactively, and generate safe, efficient trajectories with temporal reasoning capabilities for waiting behaviours and dynamic coordination. 

**Abstract (ZH)**: 自主车辆在结构化环境中的导航需要能够生成时间最优、无碰撞轨迹的规划器，这些轨迹同时满足动力学和运动学约束。我们引入了一种基于图的运动规划器V*，它在离散化的时空速度晶格中将速度和方向表示为显式的状态变量。与传统方法将空间搜索与动态可行性分离或依赖于事后平滑的方法不同，V*在搜索扩展过程中通过动态图生成直接将运动维度整合到图构建中。为了管理高维搜索的复杂性，我们采用六边形离散化策略，并提供了正式的数学证明，建立了速度感知运动规划中最优航点间距和最小节点冗余性，特别是在方向转换受限制的情况下。我们为刚性自行车模型中的瞬态转向动力学开发了一个数学公式，模型转向角收敛表现出指数行为，并推导出收敛速率参数的关系。理论基础与几何修剪策略相结合，可以消除导致不可行转向配置的扩展，从而使V*能够评估动态可行的操作，确保每条轨迹在无需进一步细化的情况下都能实现。此外，我们在包含移动障碍物的拥挤和动态环境中进行的模拟研究进一步展示了V*的能力，能够避免冲突、主动让行，并生成具有时间推理能力的等待行为和动态协调的高效、安全轨迹。 

---
# L2Calib: $SE(3)$-Manifold Reinforcement Learning for Robust Extrinsic Calibration with Degenerate Motion Resilience 

**Title (ZH)**: L2Calib: $SE(3)$流形强化学习及其在退化运动鲁棒外在标定中的应用 

**Authors**: Baorun Li, Chengrui Zhu, Siyi Du, Bingran Chen, Jie Ren, Wenfei Wang, Yong Liu, Jiajun Lv  

**Link**: [PDF](https://arxiv.org/pdf/2508.06330)  

**Abstract**: Extrinsic calibration is essential for multi-sensor fusion, existing methods rely on structured targets or fully-excited data, limiting real-world applicability. Online calibration further suffers from weak excitation, leading to unreliable estimates. To address these limitations, we propose a reinforcement learning (RL)-based extrinsic calibration framework that formulates extrinsic calibration as a decision-making problem, directly optimizes $SE(3)$ extrinsics to enhance odometry accuracy. Our approach leverages a probabilistic Bingham distribution to model 3D rotations, ensuring stable optimization while inherently retaining quaternion symmetry. A trajectory alignment reward mechanism enables robust calibration without structured targets by quantitatively evaluating estimated tightly-coupled trajectory against a reference trajectory. Additionally, an automated data selection module filters uninformative samples, significantly improving efficiency and scalability for large-scale datasets. Extensive experiments on UAVs, UGVs, and handheld platforms demonstrate that our method outperforms traditional optimization-based approaches, achieving high-precision calibration even under weak excitation conditions. Our framework simplifies deployment on diverse robotic platforms by eliminating the need for high-quality initial extrinsics and enabling calibration from routine operating data. The code is available at this https URL. 

**Abstract (ZH)**: 基于强化学习的外参标定框架：解决多传感器融合中的外参标定问题 

---
# Surrogate-Enhanced Modeling and Adaptive Modular Control of All-Electric Heavy-Duty Robotic Manipulators 

**Title (ZH)**: 增强代理模型与自适应模块化控制的全电气重型机器人 manipulator 系统建模与控制 

**Authors**: Amir Hossein Barjini, Mohammad Bahari, Mahdi Hejrati, Jouni Mattila  

**Link**: [PDF](https://arxiv.org/pdf/2508.06313)  

**Abstract**: This paper presents a unified system-level modeling and control framework for an all-electric heavy-duty robotic manipulator (HDRM) driven by electromechanical linear actuators (EMLAs). A surrogate-enhanced actuator model, combining integrated electromechanical dynamics with a neural network trained on a dedicated testbed, is integrated into an extended virtual decomposition control (VDC) architecture augmented by a natural adaptation law. The derived analytical HDRM model supports a hierarchical control structure that seamlessly maps high-level force and velocity objectives to real-time actuator commands, accompanied by a Lyapunov-based stability proof. In multi-domain simulations of both cubic and a custom planar triangular trajectory, the proposed adaptive modular controller achieves sub-centimeter Cartesian tracking accuracy. Experimental validation of the same 1-DoF platform under realistic load emulation confirms the efficacy of the proposed control strategy. These findings demonstrate that a surrogate-enhanced EMLA model embedded in the VDC approach can enable modular, real-time control of an all-electric HDRM, supporting its deployment in next-generation mobile working machines. 

**Abstract (ZH)**: 本文提出了一个统一的系统级建模与控制框架，用于由电磁线性执行器驱动的全电动重型机器人 manipulator (HDRM)。该框架结合了集成机电动力学与专用测试床训练的神经网络的代理增强执行器模型，并集成到扩展的虚拟分解控制 (VDC) 架构中，该架构带有自然自适应定律。从中推导出的分析型 HDRM 模型支持分层控制结构，能够无缝地将高层力和速度目标映射到实时执行器命令，并伴有基于李雅普诺夫的稳定性证明。在对三维和自定义平面三角形轨迹的多领域仿真中，所提出的自适应模块化控制器实现了亚毫米级笛卡尔跟踪精度。在现实负载模拟下的同一 1-DoF 平台的实验验证确认了所提控制策略的有效性。这些发现表明，嵌入 VDC 方法中的代理增强型 EMLA 模型能够实现全电动 HDRM 的模块化、实时控制，支持其在下一代移动作业机械中的部署。 

---
# EcBot: Data-Driven Energy Consumption Open-Source MATLAB Library for Manipulators 

**Title (ZH)**: EcBot: 数据驱动的操纵器开源MATLAB能耗库 

**Authors**: Juan Heredia, Christian Schlette, Mikkel Baun Kjærgaard  

**Link**: [PDF](https://arxiv.org/pdf/2508.06276)  

**Abstract**: Existing literature proposes models for estimating the electrical power of manipulators, yet two primary limitations prevail. First, most models are predominantly tested using traditional industrial robots. Second, these models often lack accuracy. To address these issues, we introduce an open source Matlab-based library designed to automatically generate \ac{ec} models for manipulators. The necessary inputs for the library are Denavit-Hartenberg parameters, link masses, and centers of mass. Additionally, our model is data-driven and requires real operational data, including joint positions, velocities, accelerations, electrical power, and corresponding timestamps. We validated our methodology by testing on four lightweight robots sourced from three distinct manufacturers: Universal Robots, Franka Emika, and Kinova. The model underwent testing, and the results demonstrated an RMSE ranging from 1.42 W to 2.80 W for the training dataset and from 1.45 W to 5.25 W for the testing dataset. 

**Abstract (ZH)**: 现有的文献提出了估计 manipulator 电气功率的模型，但主要存在两个局限性。首先，大多数模型主要在传统工业机器人上进行测试。其次，这些模型往往缺乏准确性。为解决这些问题，我们介绍了一个基于 Matlab 的开源库，该库旨在自动为 manipulator 生成 \ac{ec} 模型。该库所需的输入包括 Denavit-Hartenberg 参数、连杆质量以及质心。此外，我们的模型是数据驱动的，需要实际操作数据，包括关节位置、速度、加速度、电气功率及其对应的timestamp。我们通过在三个不同制造商的四款轻量化机器人（Universal Robots、Franka Emika 和 Kinova）上进行测试验证了该方法。模型进行了测试，结果显示训练数据集的 RMSE 范围为 1.42 W 至 2.80 W，测试数据集的 RMSE 范围为 1.45 W 至 5.25 W。 

---
# Dynamical Trajectory Planning of Disturbance Consciousness for Air-Land Bimodal Unmanned Aerial Vehicles 

**Title (ZH)**: 扰动意识导向的空地两用无人航空器动态轨迹规划 

**Authors**: Shaoting Liu, Zhou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05972)  

**Abstract**: Air-land bimodal vehicles provide a promising solution for navigating complex environments by combining the flexibility of aerial locomotion with the energy efficiency of ground mobility. To enhance the robustness of trajectory planning under environmental disturbances, this paper presents a disturbance-aware planning framework that incorporates real-time disturbance estimation into both path searching and trajectory optimization. A key component of the framework is a disturbance-adaptive safety boundary adjustment mechanism, which dynamically modifies the vehicle's feasible dynamic boundaries based on estimated disturbances to ensure trajectory feasibility. Leveraging the dynamics model of the bimodal vehicle, the proposed approach achieves adaptive and reliable motion planning across different terrains and operating conditions. A series of real-world experiments and benchmark comparisons on a custom-built platform validate the effectiveness and robustness of the method, demonstrating improvements in tracking accuracy, task efficiency, and energy performance under both ground and aerial disturbances. 

**Abstract (ZH)**: 空气-地面双模车辆通过结合空中运动的灵活性和地面行驶的能效性，提供了一种应对复杂环境的有前景的解决方案。为了在环境干扰下增强轨迹规划的鲁棒性，本文提出了一种干扰感知规划框架，该框架将实时干扰估计融入路径搜索和轨迹优化中。该框架的关键组件是一个干扰自适应安全边界调整机制，该机制根据估计到的干扰动态修改车辆的可行动态边界，以确保轨迹的可行性。利用双模车辆的动力学模型，所提出的方法实现了在不同地形和操作条件下的自适应和可靠的运动规划。在自建平台上进行的一系列实际实验和基准比较验证了该方法的有效性和鲁棒性，展示了在地面和空中干扰下跟踪准确性、任务效率和能源性能的提升。 

---
# Affordance-Guided Dual-Armed Disassembly Teleoperation for Mating Parts 

**Title (ZH)**: 面向功能的双臂拆装远程操作以实现零件对接 

**Authors**: Gen Sako, Takuya Kiyokawa, Kensuke Harada, Tomoki Ishikura, Naoya Miyaji, Genichiro Matsuda  

**Link**: [PDF](https://arxiv.org/pdf/2508.05937)  

**Abstract**: Robotic non-destructive disassembly of mating parts remains challenging due to the need for flexible manipulation and the limited visibility of internal structures. This study presents an affordance-guided teleoperation system that enables intuitive human demonstrations for dual-arm fix-and-disassemble tasks for mating parts. The system visualizes feasible grasp poses and disassembly directions in a virtual environment, both derived from the object's geometry, to address occlusions and structural complexity. To prevent excessive position tracking under load when following the affordance, we integrate a hybrid controller that combines position and impedance control into the teleoperated disassembly arm. Real-world experiments validate the effectiveness of the proposed system, showing improved task success rates and reduced object pose deviation. 

**Abstract (ZH)**: 基于 affordance 引导的遥操作系统：用于配合部件的固定与拆卸任务的直观人机示范方法 

---
# Modular Vacuum-Based Fixturing System for Adaptive Disassembly Workspace Integration 

**Title (ZH)**: 基于模块化真空fixture的自适应拆解工作空间集成系统 

**Authors**: Haohui Pan, Takuya Kiyokawa, Tomoki Ishikura, Shingo Hamada, Genichiro Matsuda, Kensuke Harada  

**Link**: [PDF](https://arxiv.org/pdf/2508.05936)  

**Abstract**: The disassembly of small household appliances poses significant challenges due to their complex and curved geometries, which render traditional rigid fixtures inadequate. In this paper, we propose a modular vacuum-based fixturing system that leverages commercially available balloon-type soft grippers to conform to arbitrarily shaped surfaces and provide stable support during screw-removal tasks. To enable a reliable deployment of the system, we develop a stability-aware planning framework that samples the bottom surface of the target object, filters candidate contact points based on geometric continuity, and evaluates support configurations using convex hull-based static stability criteria. We compare the quality of object placement under different numbers and configurations of balloon hands. In addition, real-world experiments were conducted to compare the success rates of traditional rigid fixtures with our proposed system. The results demonstrate that our method consistently achieves higher success rates and superior placement stability during screw removal tasks. 

**Abstract (ZH)**: 小家电的拆解由于其复杂的曲面几何结构带来了显著挑战，传统的刚性夹具已不足以应对。本文提出了一种模块化真空基座夹具系统，利用商用气球型柔软夹爪来适应任意形状的表面，并在拆卸螺钉任务中提供稳定的支撑。为了确保系统的有效部署，我们开发了一种基于稳定性的规划框架，通过对目标物体底部表面取样、基于几何连续性筛选候选接触点，并使用基于凸包的静态稳定准则评估支撑配置。我们比较了不同数量和配置的气球手对物体放置质量的影响。此外，我们还在现实世界中进行了实验，将传统刚性夹具的成功率与我们提出的系统进行了比较。结果表明，我们的方法在拆卸螺钉任务中始终能实现更高的成功率和更好的放置稳定性。 

---
# GPU-Accelerated Barrier-Rate Guided MPPI Control for Tractor-Trailer Systems 

**Title (ZH)**: 基于GPU加速的基于速率引导的MPPI控制算法在拖拉机- 牵引车系统中的应用 eSports 

**Authors**: Keyvan Majd, Hardik Parwana, Bardh Hoxha, Steven Hong, Hideki Okamoto, Georgios Fainekos  

**Link**: [PDF](https://arxiv.org/pdf/2508.05773)  

**Abstract**: Articulated vehicles such as tractor-trailers, yard trucks, and similar platforms must often reverse and maneuver in cluttered spaces where pedestrians are present. We present how Barrier-Rate guided Model Predictive Path Integral (BR-MPPI) control can solve navigation in such challenging environments. BR-MPPI embeds Control Barrier Function (CBF) constraints directly into the path-integral update. By steering the importance-sampling distribution toward collision-free, dynamically feasible trajectories, BR-MPPI enhances the exploration strength of MPPI and improves robustness of resulting trajectories. The method is evaluated in the high-fidelity CarMaker simulator on a 12 [m] tractor-trailer tasked with reverse and forward parking in a parking lot. BR-MPPI computes control inputs in above 100 [Hz] on a single GPU (for scenarios with eight obstacles) and maintains better parking clearance than a standard MPPI baseline and an MPPI with collision cost baseline. 

**Abstract (ZH)**: articulated车辆（如拖拉机-挂车、场内卡车等）常常需要在行人存在的拥挤空间中倒车和机动。我们展示了Barrier-Rate引导的模型预测路径积分（BR-MPPI）控制如何解决在这种具有挑战性的环境中的导航问题。BR-MPPI将控制障碍函数（CBF）约束直接嵌入到路径积分更新中。通过使重要性抽样分布朝向无碰撞的动态可行轨迹，BR-MPPI增强了MPPI的探索强度并提高了结果轨迹的鲁棒性。该方法在高保真CarMaker模拟器中对一辆12米长的拖拉机-挂车进行了评估，该拖车在停车场中进行倒车和前进停车。在包含八个障碍物的场景中，BR-MPPI在单个GPU上以超过100 Hz的频率计算控制输入，并在停车间隙方面优于标准的MPPI基线和带有碰撞成本的MPPI基线。 

---
# Graph-based Robot Localization Using a Graph Neural Network with a Floor Camera and a Feature Rich Industrial Floor 

**Title (ZH)**: 基于图神经网络的.floor摄像机和丰富特征工业-floor的图表示机器人定位 

**Authors**: Dominik Brämer, Diana Kleingarn, Oliver Urbann  

**Link**: [PDF](https://arxiv.org/pdf/2508.06177)  

**Abstract**: Accurate localization represents a fundamental challenge in
robotic navigation. Traditional methodologies, such as Lidar or QR-code based systems, suffer from inherent scalability and adaptability con straints, particularly in complex environments. In this work, we propose
an innovative localization framework that harnesses flooring characteris tics by employing graph-based representations and Graph Convolutional
Networks (GCNs). Our method uses graphs to represent floor features,
which helps localize the robot more accurately (0.64cm error) and more
efficiently than comparing individual image features. Additionally, this
approach successfully addresses the kidnapped robot problem in every
frame without requiring complex filtering processes. These advancements
open up new possibilities for robotic navigation in diverse environments. 

**Abstract (ZH)**: 精确定位 localization 代表了机器人导航中的一项基本挑战 argent localization represents a fundamental challenge in robotic navigation. Patio présenté une nouvelle cadre framework 创新型的定位框架stdbool Localization Framework déco � santéитесь 使用图论基 based方法-tags 和图卷积网络 GCNs tô 表示地面特征，使得机器人定位更加准确 kinois
is (精度达到6.64cm) � 更加高效 �并且成功解决了复杂过滤过程中的机器人 kidnapping盗贼问题。is进步为机器人在各种环境中应用is提供了可能性。这些进步为机器人 différentis 在érrent是环境 kinoisisis提供了可能性 editText Localization vidé 新环境 mexico提供了可能性。 

---
# Unsupervised Partner Design Enables Robust Ad-hoc Teamwork 

**Title (ZH)**: 无监督伙伴设计实现稳健的临时 teamwork 

**Authors**: Constantin Ruhdorfer, Matteo Bortoletto, Victor Oei, Anna Penzkofer, Andreas Bulling  

**Link**: [PDF](https://arxiv.org/pdf/2508.06336)  

**Abstract**: We introduce Unsupervised Partner Design (UPD) - a population-free, multi-agent reinforcement learning framework for robust ad-hoc teamwork that adaptively generates training partners without requiring pretrained partners or manual parameter tuning. UPD constructs diverse partners by stochastically mixing an ego agent's policy with biased random behaviours and scores them using a variance-based learnability metric that prioritises partners near the ego agent's current learning frontier. We show that UPD can be integrated with unsupervised environment design, resulting in the first method enabling fully unsupervised curricula over both level and partner distributions in a cooperative setting. Through extensive evaluations on Overcooked-AI and the Overcooked Generalisation Challenge, we demonstrate that this dynamic partner curriculum is highly effective: UPD consistently outperforms both population-based and population-free baselines as well as ablations. In a user study, we further show that UPD achieves higher returns than all baselines and was perceived as significantly more adaptive, more human-like, a better collaborator, and less frustrating. 

**Abstract (ZH)**: 无监督伙伴设计：一种无需人群的多智能体强化学习框架，用于自适应生成训练伙伴以实现鲁棒的即兴团队合作 

---
