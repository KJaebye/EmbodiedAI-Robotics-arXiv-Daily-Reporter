# Real-time Terrain Analysis for Off-road Autonomous Vehicles 

**Title (ZH)**: 离线自主车辆的实时地形分析 

**Authors**: Edwina Lewis, Aditya Parameshwaran, Laura Redmond, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21347)  

**Abstract**: This research addresses critical autonomous vehicle control challenges arising from road roughness variation, which induces course deviations and potential loss of road contact during steering operations. We present a novel real-time road roughness estimation system employing Bayesian calibration methodology that processes axle accelerations to predict terrain roughness with quantifiable confidence measures. The technical framework integrates a Gaussian process surrogate model with a simulated half-vehicle model, systematically processing vehicle velocity and road surface roughness parameters to generate corresponding axle acceleration responses. The Bayesian calibration routine performs inverse estimation of road roughness from observed accelerations and velocities, yielding posterior distributions that quantify prediction uncertainty for adaptive risk management. Training data generation utilizes Latin Hypercube sampling across comprehensive velocity and roughness parameter spaces, while the calibrated model integrates seamlessly with a Simplex controller architecture to dynamically adjust velocity limits based on real-time roughness predictions. Experimental validation on stochastically generated surfaces featuring varying roughness regions demonstrates robust real-time characterization capabilities, with the integrated Simplex control strategy effectively enhancing autonomous vehicle operational safety through proactive surface condition response. This innovative Bayesian framework establishes a comprehensive foundation for mitigating roughness-related operational risks while simultaneously improving efficiency and safety margins in autonomous vehicle systems. 

**Abstract (ZH)**: 这篇研究着眼于路面粗糙度变化引发的关键自动驾驶车辆控制挑战，这些挑战会在转向操作过程中导致路径偏差和路面接触丢失。我们提出了一种新型的实时路面粗糙度估计系统，该系统采用贝叶斯校准方法处理轴加速度，以定量置信度预测地形粗糙度。该技术框架结合了高斯过程代理模型和模拟半车辆模型，系统地处理车辆速度和路面粗糙度参数，生成相应的轴加速度响应。贝叶斯校准流程从观测的加速度和速度反向估计路面粗糙度，生成后验分布以量化预测不确定性，从而实现自适应风险管理。训练数据生成利用拉丁超立方抽样覆盖全面的速度和粗糙度参数空间，而校准模型无缝集成到简单形控制器架构中，基于实时粗糙度预测动态调整速度限制。在包含不同粗糙度区域的随机生成路面上的实验验证展示了 robust 的实时表征能力，而集成的简单形控制策略通过主动应对路面条件有效提升了自动驾驶车辆的操作安全性。这一创新的贝叶斯框架为减轻与粗糙度相关的操作风险奠定了全面的基础，同时提高了自动驾驶车辆系统的工作效率和安全裕度。 

---
# Active Disturbance Rejection Control for Trajectory Tracking of a Seagoing USV: Design, Simulation, and Field Experiments 

**Title (ZH)**: 基于主动干扰 rejection 控制的海上USV轨迹跟踪控制设计、仿真与场试验 

**Authors**: Jelmer van der Saag, Elia Trevisan, Wouter Falkena, Javier Alonso-Mora  

**Link**: [PDF](https://arxiv.org/pdf/2506.21265)  

**Abstract**: Unmanned Surface Vessels (USVs) face significant control challenges due to uncertain environmental disturbances like waves and currents. This paper proposes a trajectory tracking controller based on Active Disturbance Rejection Control (ADRC) implemented on the DUS V2500. A custom simulation incorporating realistic waves and current disturbances is developed to validate the controller's performance, supported by further validation through field tests in the harbour of Scheveningen, the Netherlands, and at sea. Simulation results demonstrate that ADRC significantly reduces cross-track error across all tested conditions compared to a baseline PID controller but increases control effort and energy consumption. Field trials confirm these findings while revealing a further increase in energy consumption during sea trials compared to the baseline. 

**Abstract (ZH)**: 无人水面船舶（USVs）面对来自波浪和洋流等不确定性环境干扰的显著控制挑战。本文提出了一种基于活性干扰拒绝控制（ADRC）的轨迹跟踪控制器，并在DUS V2500上进行了实现。一个包含真实波浪和水流干扰的自定义仿真被开发出来以验证控制器的性能，并通过在荷兰斯赫维宁恩港和海上进行的进一步现场试验来验证。仿真结果表明，与基线PID控制器相比，ADRC在所有测试条件下都能显著降低横向偏差，但会增加控制努力和能耗。现场试验证实了这些发现，并揭示了与基线相比，在海上试验中能耗进一步增加。 

---
# Dynamic Risk-Aware MPPI for Mobile Robots in Crowds via Efficient Monte Carlo Approximations 

**Title (ZH)**: 基于高效蒙特卡洛近似的动态风险意识MPPI在人群中的移动机器人路径规划 

**Authors**: Elia Trevisan, Khaled A. Mustafa, Godert Notten, Xinwei Wang, Javier Alonso-Mora  

**Link**: [PDF](https://arxiv.org/pdf/2506.21205)  

**Abstract**: Deploying mobile robots safely among humans requires the motion planner to account for the uncertainty in the other agents' predicted trajectories. This remains challenging in traditional approaches, especially with arbitrarily shaped predictions and real-time constraints. To address these challenges, we propose a Dynamic Risk-Aware Model Predictive Path Integral control (DRA-MPPI), a motion planner that incorporates uncertain future motions modelled with potentially non-Gaussian stochastic predictions. By leveraging MPPI's gradient-free nature, we propose a method that efficiently approximates the joint Collision Probability (CP) among multiple dynamic obstacles for several hundred sampled trajectories in real-time via a Monte Carlo (MC) approach. This enables the rejection of samples exceeding a predefined CP threshold or the integration of CP as a weighted objective within the navigation cost function. Consequently, DRA-MPPI mitigates the freezing robot problem while enhancing safety. Real-world and simulated experiments with multiple dynamic obstacles demonstrate DRA-MPPI's superior performance compared to state-of-the-art approaches, including Scenario-based Model Predictive Control (S-MPC), Frenet planner, and vanilla MPPI. 

**Abstract (ZH)**: 部署移动机器人在人类环境中安全移动需要运动规划器考虑其他代理预测轨迹中的不确定性。传统方法在处理任意形状的预测和实时约束时仍面临挑战。为解决这些挑战，我们提出了一种动态风险意识模型预测路径积分控制（DRA-MPPI），这是一种将潜在非高斯随机预测建模的不确定未来运动纳入考量的运动规划器。通过利用MPPI的无导数性质，我们提出了一种方法，能够在几秒钟内通过蒙特卡洛方法高效地近似多个动态障碍物之间的联合碰撞概率（CP），从而在多个采样轨迹中实时估算碰撞概率。这使得DRA-MPPI能够拒绝超出预定义CP阈值的样本，或将CP作为加权目标集成到导航成本函数中。因此，DRA-MPPI减轻了机器人冻结问题，增强了安全性。多动态障碍物的现实世界和模拟实验表明，DRA-MPPI在性能上优于现有最先进的方法，包括基于场景的模型预测控制（S-MPC）、Frenet规划器和传统的MPPI。 

---
# Control of Marine Robots in the Era of Data-Driven Intelligence 

**Title (ZH)**: 数据驱动智能时代的海洋机器人控制 

**Authors**: Lin Hong, Lu Liu, Zhouhua Peng, Fumin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21063)  

**Abstract**: The control of marine robots has long relied on model-based methods grounded in classical and modern control theory. However, the nonlinearity and uncertainties inherent in robot dynamics, coupled with the complexity of marine environments, have revealed the limitations of conventional control methods. The rapid evolution of machine learning has opened new avenues for incorporating data-driven intelligence into control strategies, prompting a paradigm shift in the control of marine robots. This paper provides a review of recent progress in marine robot control through the lens of this emerging paradigm. The review covers both individual and cooperative marine robotic systems, highlighting notable achievements in data-driven control of marine robots and summarizing open-source resources that support the development and validation of advanced control methods. Finally, several future perspectives are outlined to guide research toward achieving high-level autonomy for marine robots in real-world applications. This paper aims to serve as a roadmap toward the next-generation control framework of marine robots in the era of data-driven intelligence. 

**Abstract (ZH)**: 基于数据驱动智能的海洋机器人控制进展reviews the recent progress in marine robot control through the emerging paradigm of data-driven intelligence. 

---
# Cooperative Circumnavigation for Multi-Quadrotor Systems via Onboard Sensing 

**Title (ZH)**: 多旋翼系统基于机载传感的协同巡飞 

**Authors**: Xueming Liu, Lin Li, Xiang Zhou, Qingrui Zhang, Tianjiang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20954)  

**Abstract**: A cooperative circumnavigation framework is proposed for multi-quadrotor systems to enclose and track a moving target without reliance on external localization systems. The distinct relationships between quadrotor-quadrotor and quadrotor-target interactions are evaluated using a heterogeneous perception strategy and corresponding state estimation algorithms. A modified Kalman filter is developed to fuse visual-inertial odometry with range measurements to enhance the accuracy of inter-quadrotor relative localization. An event-triggered distributed Kalman filter is designed to achieve robust target state estimation under visual occlusion by incorporating neighbor measurements and estimated inter-quadrotor relative positions. Using the estimation results, a cooperative circumnavigation controller is constructed, leveraging an oscillator-based autonomous formation flight strategy. We conduct extensive indoor and outdoor experiments to validate the efficiency of the proposed circumnavigation framework in occluded environments. Furthermore, a quadrotor failure experiment highlights the inherent fault tolerance property of the proposed framework, underscoring its potential for deployment in search-and-rescue operations. 

**Abstract (ZH)**: 一种多旋翼系统协作环航框架：在无需外部定位系统的情况下包围和跟踪移动目标，并评估旋翼机-旋翼机和旋翼机-目标之间的异构感知关系和相应状态估计算法，开发改进的卡尔曼滤波器融合视觉-惯性里程计与距离测量以提高旋翼机间相对定位的精度，设计事件触发的分布式卡尔曼滤波器以在视觉遮挡下实现鲁棒的目标状态估计，利用振荡器为基础的自主编队飞行策略构建协作环航控制器。通过广泛的室内和室外实验验证所提出环航框架在遮挡环境中的效率，并通过旋翼机故障实验突出所提出框架的固有容错特性，强调其在搜索与救援操作中的潜力。 

---
# Model-Based Real-Time Pose and Sag Estimation of Overhead Power Lines Using LiDAR for Drone Inspection 

**Title (ZH)**: 基于模型的无人机巡检中基于LiDAR的架空输电线路实时姿态和弛度估计 

**Authors**: Alexandre Girard, Steven A. Parkison, Philippe Hamelin  

**Link**: [PDF](https://arxiv.org/pdf/2506.20812)  

**Abstract**: Drones can inspect overhead power lines while they remain energized, significantly simplifying the inspection process. However, localizing a drone relative to all conductors using an onboard LiDAR sensor presents several challenges: (1) conductors provide minimal surface for LiDAR beams limiting the number of conductor points in a scan, (2) not all conductors are consistently detected, and (3) distinguishing LiDAR points corresponding to conductors from other objects, such as trees and pylons, is difficult. This paper proposes an estimation approach that minimizes the error between LiDAR measurements and a single geometric model representing the entire conductor array, rather than tracking individual conductors separately. Experimental results, using data from a power line drone inspection, demonstrate that this method achieves accurate tracking, with a solver converging under 50 ms per frame, even in the presence of partial observations, noise, and outliers. A sensitivity analysis shows that the estimation approach can tolerate up to twice as many outlier points as valid conductors measurements. 

**Abstract (ZH)**: 无人机可以带电检测架空输电线路，显著简化检测过程。然而，使用机载LiDAR传感器相对于所有导线进行定位存在多项挑战：导线对LiDAR光束提供的表面极少，限制了扫描中的导线点数量；并非所有导线都能一致检测；区分与导线对应的LiDAR点与其他物体，如树木和电塔，是困难的。本文提出了一种估计方法，力求最小化LiDAR测量值与表示整个导线阵列的单一几何模型之间的误差，而不是单独追踪每根导线。实验结果，使用输电线路无人机检测数据，显示该方法即使在部分观测、噪声和离群点存在的条件下也能实现准确跟踪，求解器在每帧50毫秒内收敛。敏感性分析表明，该估计方法可以容忍的离群点数量是有效导线测量数量的两倍。 

---
# Online Planning for Cooperative Air-Ground Robot Systems with Unknown Fuel Requirements 

**Title (ZH)**: 具有未知燃料需求的协同空地机器人系统在线规划 

**Authors**: Ritvik Agarwal, Behnoushsadat Hatami, Alvika Gautam, Parikshit Maini  

**Link**: [PDF](https://arxiv.org/pdf/2506.20804)  

**Abstract**: We consider an online variant of the fuel-constrained UAV routing problem with a ground-based mobile refueling station (FCURP-MRS), where targets incur unknown fuel costs. We develop a two-phase solution: an offline heuristic-based planner computes initial UAV and UGV paths, and a novel online planning algorithm that dynamically adjusts rendezvous points based on real-time fuel consumption during target processing. Preliminary Gazebo simulations demonstrate the feasibility of our approach in maintaining UAV-UGV path validity, ensuring mission completion. Link to video: this https URL 

**Abstract (ZH)**: 基于地面移动加油站的在线燃料约束无人机路由问题（FCURP-MRS）：一种两阶段解决方案 

---
# IMA-Catcher: An IMpact-Aware Nonprehensile Catching Framework based on Combined Optimization and Learning 

**Title (ZH)**: IMA-Catcher：一种基于联合优化与学习的感知作用非抓取捕捉框架 

**Authors**: Francesco Tassi, Jianzhuang Zhao, Gustavo J. G. Lahr, Luna Gava, Marco Monforte, Arren Glover, Chiara Bartolozzi, Arash Ajoudani  

**Link**: [PDF](https://arxiv.org/pdf/2506.20801)  

**Abstract**: Robotic catching of flying objects typically generates high impact forces that might lead to task failure and potential hardware damages. This is accentuated when the object mass to robot payload ratio increases, given the strong inertial components characterizing this task. This paper aims to address this problem by proposing an implicitly impact-aware framework that accomplishes the catching task in both pre- and post-catching phases. In the first phase, a motion planner generates optimal trajectories that minimize catching forces, while in the second, the object's energy is dissipated smoothly, minimizing bouncing. In particular, in the pre-catching phase, a real-time optimal planner is responsible for generating trajectories of the end-effector that minimize the velocity difference between the robot and the object to reduce impact forces during catching. In the post-catching phase, the robot's position, velocity, and stiffness trajectories are generated based on human demonstrations when catching a series of free-falling objects with unknown masses. A hierarchical quadratic programming-based controller is used to enforce the robot's constraints (i.e., joint and torque limits) and create a stack of tasks that minimizes the reflected mass at the end-effector as a secondary objective. The initial experiments isolate the problem along one dimension to accurately study the effects of each contribution on the metrics proposed. We show how the same task, without velocity matching, would be infeasible due to excessive joint torques resulting from the impact. The addition of reflected mass minimization is then investigated, and the catching height is increased to evaluate the method's robustness. Finally, the setup is extended to catching along multiple Cartesian axes, to prove its generalization in space. 

**Abstract (ZH)**: 机器人捕获飞行物体通常会产生高冲击力，可能导致任务失败和潜在的硬件损坏。当物体质量与机器人负载比例增加时，这一问题更加明显，因为该任务具有强烈的惯性特征。本文提出了一种隐式冲击感知框架，该框架在捕获前、后阶段均能够完成捕获任务。在捕获前阶段，运动规划器生成可最小化捕获力的最优轨迹；在捕获后阶段，物体的能量被平滑消散，以最小化反弹。具体而言，在捕获前阶段，实时最优规划器负责生成末端执行器的轨迹，以最小化机器人与物体之间的速度差异，从而减少捕获过程中的冲击力。在捕获后阶段，基于人类示范，机器人的位置、速度和刚度轨迹根据一系列未知质量的自由落体物体的捕获生成。基于分层二次规划的控制器被用于强制执行机器人的约束（即关节和扭矩限制），并创建一个任务堆栈，将末端执行器上反射质量最小化作为次要目标。初始实验将问题沿着一个维度隔离，以准确研究每种贡献对所提指标的影响。我们展示了在没有速度匹配的情况下，同样的任务由于冲击导致的关节扭矩过度而变得不可行。随后研究了引入反射质量最小化的有效性，并提高捕获高度以评估该方法的鲁棒性。最后，该设置扩展到多个笛卡尔轴上的捕获，以证明其在空间上的通用性。 

---
# Complex Model Transformations by Reinforcement Learning with Uncertain Human Guidance 

**Title (ZH)**: 基于不确定性人类指导的强化学习复杂模型转换 

**Authors**: Kyanna Dagenais, Istvan David  

**Link**: [PDF](https://arxiv.org/pdf/2506.20883)  

**Abstract**: Model-driven engineering problems often require complex model transformations (MTs), i.e., MTs that are chained in extensive sequences. Pertinent examples of such problems include model synchronization, automated model repair, and design space exploration. Manually developing complex MTs is an error-prone and often infeasible process. Reinforcement learning (RL) is an apt way to alleviate these issues. In RL, an autonomous agent explores the state space through trial and error to identify beneficial sequences of actions, such as MTs. However, RL methods exhibit performance issues in complex problems. In these situations, human guidance can be of high utility. In this paper, we present an approach and technical framework for developing complex MT sequences through RL, guided by potentially uncertain human advice. Our framework allows user-defined MTs to be mapped onto RL primitives, and executes them as RL programs to find optimal MT sequences. Our evaluation shows that human guidance, even if uncertain, substantially improves RL performance, and results in more efficient development of complex MTs. Through a trade-off between the certainty and timeliness of human advice, our method takes a step towards RL-driven human-in-the-loop engineering methods. 

**Abstract (ZH)**: 通过受不确定人类指导约束的强化学习开发复杂模型转换序列：迈向RL驱动的人机交互工程方法 

---
