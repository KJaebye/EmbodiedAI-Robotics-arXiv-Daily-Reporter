# Fabrica: Dual-Arm Assembly of General Multi-Part Objects via Integrated Planning and Learning 

**Title (ZH)**: Fabrica：通过集成规划与学习的双臂通用多部件对象装配 

**Authors**: Yunsheng Tian, Joshua Jacob, Yijiang Huang, Jialiang Zhao, Edward Gu, Pingchuan Ma, Annan Zhang, Farhad Javid, Branden Romero, Sachin Chitta, Shinjiro Sueda, Hui Li, Wojciech Matusik  

**Link**: [PDF](https://arxiv.org/pdf/2506.05168)  

**Abstract**: Multi-part assembly poses significant challenges for robots to execute long-horizon, contact-rich manipulation with generalization across complex geometries. We present Fabrica, a dual-arm robotic system capable of end-to-end planning and control for autonomous assembly of general multi-part objects. For planning over long horizons, we develop hierarchies of precedence, sequence, grasp, and motion planning with automated fixture generation, enabling general multi-step assembly on any dual-arm robots. The planner is made efficient through a parallelizable design and is optimized for downstream control stability. For contact-rich assembly steps, we propose a lightweight reinforcement learning framework that trains generalist policies across object geometries, assembly directions, and grasp poses, guided by equivariance and residual actions obtained from the plan. These policies transfer zero-shot to the real world and achieve 80% successful steps. For systematic evaluation, we propose a benchmark suite of multi-part assemblies resembling industrial and daily objects across diverse categories and geometries. By integrating efficient global planning and robust local control, we showcase the first system to achieve complete and generalizable real-world multi-part assembly without domain knowledge or human demonstrations. Project website: this http URL 

**Abstract (ZH)**: 多部件装配对机器人执行长期、富含接触的通用操纵任务构成显著挑战。我们提出Fabrica，一个双臂机器人系统，具备端到端规划和控制能力，用于自主装配通用多部件物体。通过长时间规划，我们开发了优先级、序列、抓取和运动规划的层次结构，并自动生成夹具，使任意双臂机器人能够进行通用多步装配。该规划器通过并行化设计变得高效，并优化了下游控制稳定性。对于富含接触的装配步骤，我们提出了一种轻量级的强化学习框架，训练出针对不同物体几何形状、装配方向和抓取姿态的通用策略，这些策略通过计划中的不变性和残差动作进行引导。这些策略在没有领域知识或人类示范的情况下，在实际世界中实现零样本转移，并实现80%的成功步骤。为系统评估，我们提出了一个基准测试套件，包括类似工业和日常物体的多部件装配，涵盖多种类别和几何形状。通过整合高效的全局规划和鲁棒的局部控制，我们展示了第一个能够在实际世界中实现完整且可泛化的多部件装配的系统。项目网站：这个链接 

---
# EDEN: Efficient Dual-Layer Exploration Planning for Fast UAV Autonomous Exploration in Large 3-D Environments 

**Title (ZH)**: EDEN:高效双层探索规划以实现大型三维环境快速无人机自主探索 

**Authors**: Qianli Dong, Xuebo Zhang, Shiyong Zhang, Ziyu Wang, Zhe Ma, Haobo Xi  

**Link**: [PDF](https://arxiv.org/pdf/2506.05106)  

**Abstract**: Efficient autonomous exploration in large-scale environments remains challenging due to the high planning computational cost and low-speed maneuvers. In this paper, we propose a fast and computationally efficient dual-layer exploration planning method. The insight of our dual-layer method is efficiently finding an acceptable long-term region routing and greedily exploring the target in the region of the first routing area with high speed. Specifically, the proposed method finds the long-term area routing through an approximate algorithm to ensure real-time planning in large-scale environments. Then, the viewpoint in the first routing region with the lowest curvature-penalized cost, which can effectively reduce decelerations caused by sharp turn motions, will be chosen as the next exploration target. To further speed up the exploration, we adopt an aggressive and safe exploration-oriented trajectory to enhance exploration continuity. The proposed method is compared to state-of-the-art methods in challenging simulation environments. The results show that the proposed method outperforms other methods in terms of exploration efficiency, computational cost, and trajectory speed. We also conduct real-world experiments to validate the effectiveness of the proposed method. The code will be open-sourced. 

**Abstract (ZH)**: 大规模环境中的高效自主探索仍然具有挑战性，因为规划计算成本高且操纵速度低。本文提出了一种快速且计算高效的双层探索规划方法。我们双层方法的见解是高效地找到一个可接受的长期区域路线，并在第一个路由区域中以高速贪婪地探索目标。具体来说，提出的算法通过近似算法找到长期区域路线，以确保在大规模环境中实现实时规划。然后，选择第一个路由区域中曲率惩罚成本最低的视角作为下一个探索目标，以有效减少因急转弯运动引起的减速。为了进一步加快探索，我们采用了一种积极且安全的探索导向轨迹来提高探索连续性。我们将提出的方法与最先进的方法在具有挑战性的仿真环境中进行了比较。结果表明，提出的方法在探索效率、计算成本和轨迹速度方面优于其他方法。我们还进行了实地实验以验证提出方法的有效性。代码将开源。 

---
# A Unified Framework for Simulating Strongly-Coupled Fluid-Robot Multiphysics 

**Title (ZH)**: 统一的强耦合流体-机器人多物理场仿真框架 

**Authors**: Jeong Hun Lee, Junzhe Hu, Sofia Kwok, Carmel Majidi, Zachary Manchester  

**Link**: [PDF](https://arxiv.org/pdf/2506.05012)  

**Abstract**: We present a framework for simulating fluid-robot multiphysics as a single, unified optimization problem. The coupled manipulator and incompressible Navier-Stokes equations governing the robot and fluid dynamics are derived together from a single Lagrangian using the principal of least action. We then employ discrete variational mechanics to derive a stable, implicit time-integration scheme for jointly simulating both the fluid and robot dynamics, which are tightly coupled by a constraint that enforces the no-slip boundary condition at the fluid-robot interface. Extending the classical immersed boundary method, we derive a new formulation of the no-slip constraint that is numerically well-conditioned and physically accurate for multibody systems commonly found in robotics. We demonstrate our approach's physical accuracy on benchmark computational fluid-dynamics problems, including Poiseuille flow and a disc in free stream. We then design a locomotion policy for a novel swimming robot in simulation and validate results on real-world hardware, showcasing our framework's sim-to-real capability for robotics tasks. 

**Abstract (ZH)**: 我们提出了一种将流体-机器人多物理模拟作为单一统一优化问题的框架。通过单一拉格朗日函数并利用最小作用原理，结合推导出机器人和流体力学的耦合操作器及不可压缩纳维-斯托克斯方程。随后，我们运用离散变分力学推导出一种稳定且隐式的时间积分方案来联合模拟流体和机器人动力学，二者通过确保流体-机器人界面无滑移边界条件的约束紧密耦合。在此基础上，我们扩展了经典的浸入边界方法，推导出一种新的无滑移约束形式，其在机器人中常见的多体系统中具有良好的数值条件稳定性和物理准确性。我们在基准计算流体力学问题上展示了该方法的物理准确性，包括泊流流和自由流中的盘体。然后，我们在模拟中为一种新型游泳机器人设计了运动策略，并在实际硬件上验证结果，展示了该框架在机器人任务中的从仿真到现实的能力。 

---
# A Pillbug-Inspired Morphing Mechanism Covered with Sliding Shells 

**Title (ZH)**: 基于滑动壳覆盖的 Pill虫启发自形态机制 

**Authors**: Jieyu Wang, Yingzhong Tian, Fengfeng Xi, Damien Chablat, Jianing Lin, Gaoke Ren, Yinjun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.04942)  

**Abstract**: This research proposes a novel morphing structure with shells inspired by the movement of pillbugs. Instead of the pillbug body, a loopcoupled mechanism based on slider-crank mechanisms is utilized to achieve the rolling up and spreading motion. This mechanism precisely imitates three distinct curves that mimic the shape morphing of a pillbug. To decrease the degree-of-freedom (DOF) of the mechanism to one, scissor mechanisms are added. 3D curved shells are then attached to the tracer points of the morphing mechanism to safeguard it from attacks while allowing it to roll. Through type and dimensional synthesis, a complete system that includes shells and an underlying morphing mechanism is developed. A 3D model is created and tested to demonstrate the proposed system's shape-changing capability. Lastly, a robot with two modes is developed based on the proposed mechanism, which can curl up to roll down hills and can spread to move in a straight line via wheels. 

**Abstract (ZH)**: 基于多足虫运动灵感的新型折纸结构及其变形机制研究 

---
# Tire Wear Aware Trajectory Tracking Control for Multi-axle Swerve-drive Autonomous Mobile Robots 

**Title (ZH)**: 基于胎纹磨损的多轴转向驱动自主移动机器人轨迹跟踪控制 

**Authors**: Tianxin Hu, Xinhang Xu, Thien-Minh Nguyen, Fen Liu, Shenghai Yuan, Lihua Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.04752)  

**Abstract**: Multi-axle Swerve-drive Autonomous Mobile Robots (MS-AGVs) equipped with independently steerable wheels are commonly used for high-payload transportation. In this work, we present a novel model predictive control (MPC) method for MS-AGV trajectory tracking that takes tire wear minimization consideration in the objective function. To speed up the problem-solving process, we propose a hierarchical controller design and simplify the dynamic model by integrating the \textit{magic formula tire model} and \textit{simplified tire wear model}. In the experiment, the proposed method can be solved by simulated annealing in real-time on a normal personal computer and by incorporating tire wear into the objective function, tire wear is reduced by 19.19\% while maintaining the tracking accuracy in curve-tracking experiments. In the more challenging scene: the desired trajectory is offset by 60 degrees from the vehicle's heading, the reduction in tire wear increased to 65.20\% compared to the kinematic model without considering the tire wear optimization. 

**Abstract (ZH)**: 带有独立转向轮的多轴偏转驱动自主移动机器人（MS-AGVs）常用于高载重运输。本文提出了一种新的考虑轮胎磨损最小化的模型预测控制（MPC）方法，用于MS-AGV轨迹跟踪。为了加快问题求解速度，我们提出了分层控制器设计，并通过集成“魔法公式轮胎模型”和“简化轮胎磨损模型”简化动态模型。在实验中，所提出的方法可以在普通个人计算机上通过模拟退火在实时下求解，并通过将轮胎磨损纳入目标函数，曲轨实验中的轮胎磨损降低了19.19%，同时保持了跟踪精度。在更具挑战性的场景中，期望轨迹相对于车辆航向偏移60度，与不考虑轮胎磨损优化的动力学模型相比，轮胎磨损减少了65.20%。 

---
# Real-Time LPV-Based Non-Linear Model Predictive Control for Robust Trajectory Tracking in Autonomous Vehicles 

**Title (ZH)**: 基于实时LPV的非线性模型预测控制在自主车辆 robust 轨迹跟踪中的应用 

**Authors**: Nitish Kumar, Rajalakshmi Pachamuthu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04684)  

**Abstract**: This paper presents the development and implementation of a Model Predictive Control (MPC) framework for trajectory tracking in autonomous vehicles under diverse driving conditions. The proposed approach incorporates a modular architecture that integrates state estimation, vehicle dynamics modeling, and optimization to ensure real-time performance. The state-space equations are formulated in a Linear Parameter Varying (LPV) form, and a curvature-based tuning method is introduced to optimize weight matrices for varying trajectories. The MPC framework is implemented using the Robot Operating System (ROS) for parallel execution of state estimation and control optimization, ensuring scalability and minimal latency. Extensive simulations and real-time experiments were conducted on multiple predefined trajectories, demonstrating high accuracy with minimal cross-track and orientation errors, even under aggressive maneuvers and high-speed conditions. The results highlight the robustness and adaptability of the proposed system, achieving seamless alignment between simulated and real-world performance. This work lays the foundation for dynamic weight tuning and integration into cooperative autonomous navigation systems, paving the way for enhanced safety and efficiency in autonomous driving applications. 

**Abstract (ZH)**: 本文提出了一种用于在各种驾驶条件下自主车辆轨迹跟踪的模型预测控制(MPC)框架的发展与实施。该提出的方案结合了模块化架构，该架构集成了状态估计、车辆动力学建模和优化，以确保实时性能。状态空间方程被形式化为线性参数变化(LPV)形式，并引入了一种基于曲率的调优方法以优化不同轨迹的权重矩阵。MPC框架使用机器人操作系统(ROS)实现，并行执行状态估计和控制优化，确保可扩展性并最小化延迟。在多种预定义轨迹上进行了广泛的仿真和实时实验，即使在激进操作和高速条件下也表现出高精度，且侧向和姿态误差较小。研究结果强调了所提出系统的稳健性和适应性，实现了模拟与实际性能的无缝对接。本文为动态权重调优及其在协同自主导航系统中的集成奠定了基础，为进一步提高自动驾驶应用的安全性和效率铺平了道路。 

---
# Application of SDRE to Achieve Gait Control in a Bipedal Robot for Knee-Type Exoskeleton Testing 

**Title (ZH)**: 基于SDRE的膝式外骨骼测试 bipedal机器人步态控制应用 

**Authors**: Ping-Kong Huang, Chien-Wu Lan, Chin-Tien Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04680)  

**Abstract**: Exoskeletons are widely used in rehabilitation and industrial applications to assist human motion. However, direct human testing poses risks due to possible exoskeleton malfunctions and inconsistent movement replication. To provide a safer and more repeatable testing environment, this study employs a bipedal robot platform to reproduce human gait, allowing for controlled exoskeleton evaluations. A control strategy based on the State-Dependent Riccati Equation (SDRE) is formulated to achieve optimal torque control for accurate gait replication. The bipedal robot dynamics are represented using double pendulum model, where SDRE-optimized control inputs minimize deviations from human motion trajectories. To align with motor behavior constraints, a parameterized control method is introduced to simplify the control process while effectively replicating human gait. The proposed approach initially adopts a ramping trapezoidal velocity model, which is then adapted into a piecewise linear velocity-time representation through motor command overwriting. This modification enables finer control over gait phase transitions while ensuring compatibility with motor dynamics. The corresponding cost function optimizes the control parameters to minimize errors in joint angles, velocities, and torques relative to SDRE control result. By structuring velocity transitions in accordance with motor limitations, the method reduce the computational load associated with real-time control. Experimental results verify the feasibility of the proposed parameterized control method in reproducing human gait. The bipedal robot platform provides a reliable and repeatable testing mechanism for knee-type exoskeletons, offering insights into exoskeleton performance under controlled conditions. 

**Abstract (ZH)**: 外骨骼在康复和工业应用中广泛用于辅助人体运动。然而，直接人体测试存在风险，因为可能出现外骨骼故障和运动复制不一致的问题。为了提供一个更安全和更具可重复性的测试环境，本研究采用双足机器人平台来再现人类步态，从而实现对外骨骼的受控评估。基于状态依赖型里卡提方程（SDRE）的控制策略被制定出来，以实现最优扭矩控制，准确再现步态。双足机器人动力学通过双摆模型表示，其中SDRE优化的控制输入最小化与人类运动轨迹的偏差。为了符合电机行为约束，引入了一种参数化控制方法来简化控制过程同时有效再现人类步态。所提出的方法最初采用梯形加速度模型，然后通过电机命令覆盖将其改编为分段线性速度-时间表示。这种修改在细控步态相位转换的同时，确保与电机动力学的兼容性。相应的代价函数优化控制参数，以最小化关节角度、速度和扭矩相对于SDRE控制结果的误差。通过根据电机限制结构化速度过渡，该方法减少了实时控制相关的计算负担。实验结果验证了所提出参数化控制方法在再现人类步态方面的可行性。双足机器人平台为膝型外骨骼提供了一种可靠且可重复的测试机制，在受控条件下揭示了外骨骼的性能。 

---
# ActivePusher: Active Learning and Planning with Residual Physics for Nonprehensile Manipulation 

**Title (ZH)**: 主动推� Zhang: 基于残差物理的主动学习与计划在非挟持操作中 

**Authors**: Zhuoyun Zhong, Seyedali Golestaneh, Constantinos Chamzas  

**Link**: [PDF](https://arxiv.org/pdf/2506.04646)  

**Abstract**: Planning with learned dynamics models offers a promising approach toward real-world, long-horizon manipulation, particularly in nonprehensile settings such as pushing or rolling, where accurate analytical models are difficult to obtain. Although learning-based methods hold promise, collecting training data can be costly and inefficient, as it often relies on randomly sampled interactions that are not necessarily the most informative. To address this challenge, we propose ActivePusher, a novel framework that combines residual-physics modeling with kernel-based uncertainty-driven active learning to focus data acquisition on the most informative skill parameters. Additionally, ActivePusher seamlessly integrates with model-based kinodynamic planners, leveraging uncertainty estimates to bias control sampling toward more reliable actions. We evaluate our approach in both simulation and real-world environments and demonstrate that it improves data efficiency and planning success rates compared to baseline methods. 

**Abstract (ZH)**: 基于学习动态模型的规划为实现真实世界中的长期操作提供了有前景的方法，特别是在推搡或滚动等非受握操作中，准确的分析模型难以获得。尽管基于学习的方法具有潜力，但收集训练数据可能代价高昂且效率低下，因为这通常依赖于随机采样的交互，未必是最具信息性的。为应对这一挑战，我们提出了ActivePusher，一种结合残差物理建模与核基于不确定性驱动的主动学习的新框架，专注于最信息性的技能参数的数据获取。此外，ActivePusher 无缝集成到基于模型的运动动力学规划器中，利用不确定性估计对控制采样进行偏置，以更倾向于可靠的操作。我们在模拟和真实世界环境中评估了这种方法，并证明其在数据效率和规划成功率方面优于基准方法。 

---
# Multimodal Limbless Crawling Soft Robot with a Kirigami Skin 

**Title (ZH)**: kirigami 基皮肤的多模态无肢爬行软机器人 

**Authors**: Jonathan Tirado, Aida Parvaresh, Burcu Seyidoğlu, Darryl A. Bedford, Jonas Jørgensen, Ahmad Rafsanjani  

**Link**: [PDF](https://arxiv.org/pdf/2506.04547)  

**Abstract**: Limbless creatures can crawl on flat surfaces by deforming their bodies and interacting with asperities on the ground, offering a biological blueprint for designing efficient limbless robots. Inspired by this natural locomotion, we present a soft robot capable of navigating complex terrains using a combination of rectilinear motion and asymmetric steering gaits. The robot is made of a pair of antagonistic inflatable soft actuators covered with a flexible kirigami skin with asymmetric frictional properties. The robot's rectilinear locomotion is achieved through cyclic inflation of internal chambers with precise phase shifts, enabling forward progression. Steering is accomplished using an asymmetric gait, allowing for both in-place rotation and wide turns. To validate its mobility in obstacle-rich environments, we tested the robot in an arena with coarse substrates and multiple obstacles. Real-time feedback from onboard proximity sensors, integrated with a human-machine interface (HMI), allowed adaptive control to avoid collisions. This study highlights the potential of bioinspired soft robots for applications in confined or unstructured environments, such as search-and-rescue operations, environmental monitoring, and industrial inspections. 

**Abstract (ZH)**: 无肢生物通过变形身体并利用地面凸起进行爬行，为设计高效的无肢机器人提供了生物学蓝图。受此自然运动方式启发，我们提出了一种可以利用直线运动与不对称转向步态组合方式进行复杂地形导航的软机器人。该机器人由一对对抗式充气软执行器组成，表面覆盖着具有不对称摩擦性质的柔性 kirigami 外皮。机器人的直线爬行通过内部隔室的循环充气和精确的相位移实现，从而实现前进。转向则通过不对称步态实现，允许原地旋转和大范围转弯。为了验证其在多障碍环境中的移动性，我们在此类地形中具有粗糙材质的竞技场中对机器人进行了测试。通过集成内置接近传感器和人机界面（HMI），实现实时反馈并进行适应性控制以避免碰撞。本研究突显了受生物启发的软机器人在受限或未结构化环境中的应用潜力，如搜救、环境监测和工业检查等领域。 

---
# Olfactory Inertial Odometry: Sensor Calibration and Drift Compensation 

**Title (ZH)**: 嗅觉惯性里程计：传感器标定与漂移补偿 

**Authors**: Kordel K. France, Ovidiu Daescu, Anirban Paul, Shalini Prasad  

**Link**: [PDF](https://arxiv.org/pdf/2506.04539)  

**Abstract**: Visual inertial odometry (VIO) is a process for fusing visual and kinematic data to understand a machine's state in a navigation task. Olfactory inertial odometry (OIO) is an analog to VIO that fuses signals from gas sensors with inertial data to help a robot navigate by scent. Gas dynamics and environmental factors introduce disturbances into olfactory navigation tasks that can make OIO difficult to facilitate. With our work here, we define a process for calibrating a robot for OIO that generalizes to several olfaction sensor types. Our focus is specifically on calibrating OIO for centimeter-level accuracy in localizing an odor source on a slow-moving robot platform to demonstrate use cases in robotic surgery and touchless security screening. We demonstrate our process for OIO calibration on a real robotic arm and show how this calibration improves performance over a cold-start olfactory navigation task. 

**Abstract (ZH)**: 视觉惯性里程计（VIO）是一种融合视觉和动力学数据以理解机器在导航任务中状态的过程。气味惯性里程计（OIO）是一种将气体传感器信号与惯性数据融合以助机器人靠气味导航的技术。气体动力学和环境因素会对气味导航任务引入干扰，使OIO难以实现。通过我们的研究，我们定义了一种适用于多种气味传感器类型的校准过程，重点在于为缓慢移动的机器人平台校准OIO，以实现厘米级精度的气味源定位，应用于机器人手术和无接触安全筛查等场景。我们在实际的机器人手臂上展示了OIO校准过程，并展示了这种校准如何提升从冷启动开始的气味导航任务的性能。 

---
