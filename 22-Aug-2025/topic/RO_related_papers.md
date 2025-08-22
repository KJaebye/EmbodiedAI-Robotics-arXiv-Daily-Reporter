# Understanding and Utilizing Dynamic Coupling in Free-Floating Space Manipulators for On-Orbit Servicing 

**Title (ZH)**: 理解与利用自由浮动空间操作器中的动态耦合进行在轨服务 

**Authors**: Gargi Das, Daegyun Choi, Donghoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.15732)  

**Abstract**: This study proposes a dynamic coupling-informed trajectory optimization algorithm for free-floating space manipulator systems (SMSs). Dynamic coupling between the base and the manipulator arms plays a critical role in influencing the system's behavior. While prior research has predominantly focused on minimizing this coupling, often overlooking its potential advantages, this work investigates how dynamic coupling can instead be leveraged to improve trajectory planning. Singular value decomposition (SVD) of the dynamic coupling matrix is employed to identify the dominant components governing coupling behavior. A quantitative metric is then formulated to characterize the strength and directionality of the coupling and is incorporated into a trajectory optimization framework. To assess the feasibility of the optimized trajectory, a sliding mode control-based tracking controller is designed to generate the required joint torque inputs. Simulation results demonstrate that explicitly accounting for dynamic coupling in trajectory planning enables more informed and potentially more efficient operation, offering new directions for the control of free-floating SMSs. 

**Abstract (ZH)**: 本文提出了一种动态耦合指导下的自由浮动空间 manipulator 系统轨迹优化算法。基座与 manipulator 臂之间的动态耦合对系统行为起着关键作用。尽管以往研究主要侧重于减小这种耦合，而忽视了其潜在优势，本研究探讨了如何利用动态耦合来改进轨迹规划。通过动态耦合矩阵的奇异值分解 (SVD) 来识别主导耦合行为的组件。然后，构建一个量化指标来表征耦合的强度和方向性，并将其纳入轨迹优化框架。为了评估优化轨迹的可行性，设计了一种滑模控制跟踪控制器来生成所需的关节扭矩输入。仿真结果表明，在轨迹规划中显式考虑动态耦合能够实现更加明智且可能更高效的运行，为自由浮动空间 manipulator 系统的控制提供了新的方向。 

---
# Mag-Match: Magnetic Vector Field Features for Map Matching and Registration 

**Title (ZH)**: 磁匹配：磁场矢量场特征在地图匹配和配准中的应用 

**Authors**: William McDonald, Cedric Le Gentil, Jennifer Wakulicz, Teresa Vidal-Calleja  

**Link**: [PDF](https://arxiv.org/pdf/2508.15300)  

**Abstract**: Map matching and registration are essential tasks in robotics for localisation and integration of multi-session or multi-robot data. Traditional methods rely on cameras or LiDARs to capture visual or geometric information but struggle in challenging conditions like smoke or dust. Magnetometers, on the other hand, detect magnetic fields, revealing features invisible to other sensors and remaining robust in such environments. In this paper, we introduce Mag-Match, a novel method for extracting and describing features in 3D magnetic vector field maps to register different maps of the same area. Our feature descriptor, based on higher-order derivatives of magnetic field maps, is invariant to global orientation, eliminating the need for gravity-aligned mapping. To obtain these higher-order derivatives map-wide given point-wise magnetometer data, we leverage a physics-informed Gaussian Process to perform efficient and recursive probabilistic inference of both the magnetic field and its derivatives. We evaluate Mag-Match in simulated and real-world experiments against a SIFT-based approach, demonstrating accurate map-to-map, robot-to-map, and robot-to-robot transformations - even without initial gravitational alignment. 

**Abstract (ZH)**: 磁匹配和注册是机器人学中用于局部化和多会话或多机器人数据整合的重要任务。传统方法依赖于摄像头或激光雷达捕获视觉或几何信息，但在烟雾或灰尘等挑战性条件下表现不佳。相比之下，磁强计检测磁场，揭示其他传感器看不见的特征，并且能够在恶劣环境中保持稳健性。本文介绍了Mag-Match，一种通过提取和描述3D磁场矢量场图中的特征来注册相同区域不同地图的新型方法。我们的特征描述符基于磁场图的高阶导数，具有全局方向的不变性，消除了重力对齐映射的需要。为了从局部磁强计数据获得整个区域的高阶导数图，我们利用物理启发的高斯过程进行高效且递归的概率推断，以获取磁场及其导数。在模拟和实际实验中，Mag-Match与基于SIFT的方法进行比较，展示了准确的地图间、机器人间以及机器人到地图的变换，即使没有初始重力对齐。 

---
# Decentralized Vision-Based Autonomous Aerial Wildlife Monitoring 

**Title (ZH)**: 基于视觉的分布式自主航空野生动物监控 

**Authors**: Makram Chahine, William Yang, Alaa Maalouf, Justin Siriska, Ninad Jadhav, Daniel Vogt, Stephanie Gil, Robert Wood, Daniela Rus  

**Link**: [PDF](https://arxiv.org/pdf/2508.15038)  

**Abstract**: Wildlife field operations demand efficient parallel deployment methods to identify and interact with specific individuals, enabling simultaneous collective behavioral analysis, and health and safety interventions. Previous robotics solutions approach the problem from the herd perspective, or are manually operated and limited in scale. We propose a decentralized vision-based multi-quadrotor system for wildlife monitoring that is scalable, low-bandwidth, and sensor-minimal (single onboard RGB camera). Our approach enables robust identification and tracking of large species in their natural habitat. We develop novel vision-based coordination and tracking algorithms designed for dynamic, unstructured environments without reliance on centralized communication or control. We validate our system through real-world experiments, demonstrating reliable deployment in diverse field conditions. 

**Abstract (ZH)**: 野生动物实地操作需要高效的并行部署方法以识别和互动特定个体，实现同时进行集体行为分析和健康与安全干预。以往的机器人解决方案从 herd 角度入手，或者手动操作且规模有限。我们提出了一种去中心化的基于视觉的多旋翼无人机系统，用于野生动物监控，该系统可扩展、低带宽且传感器最小化（单个机载 RGB 摄像头）。我们的方法能使我们在自然栖息地可靠地识别和跟踪大型物种。我们开发了一种新的基于视觉的协调和跟踪算法，适用于动态且结构不佳的环境，并且不依赖于中心化的通信或控制。我们通过实地试验验证了该系统，展示了其在多种现场条件下的可靠部署能力。 

---
# GraspQP: Differentiable Optimization of Force Closure for Diverse and Robust Dexterous Grasping 

**Title (ZH)**: GraspQP: 力闭合的可微优化方法实现多样且稳健的灵巧抓取 

**Authors**: René Zurbrügg, Andrei Cramariuc, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2508.15002)  

**Abstract**: Dexterous robotic hands enable versatile interactions due to the flexibility and adaptability of multi-fingered designs, allowing for a wide range of task-specific grasp configurations in diverse environments. However, to fully exploit the capabilities of dexterous hands, access to diverse and high-quality grasp data is essential -- whether for developing grasp prediction models from point clouds, training manipulation policies, or supporting high-level task planning with broader action options. Existing approaches for dataset generation typically rely on sampling-based algorithms or simplified force-closure analysis, which tend to converge to power grasps and often exhibit limited diversity. In this work, we propose a method to synthesize large-scale, diverse, and physically feasible grasps that extend beyond simple power grasps to include refined manipulations, such as pinches and tri-finger precision grasps. We introduce a rigorous, differentiable energy formulation of force closure, implicitly defined through a Quadratic Program (QP). Additionally, we present an adjusted optimization method (MALA*) that improves performance by dynamically rejecting gradient steps based on the distribution of energy values across all samples. We extensively evaluate our approach and demonstrate significant improvements in both grasp diversity and the stability of final grasp predictions. Finally, we provide a new, large-scale grasp dataset for 5,700 objects from DexGraspNet, comprising five different grippers and three distinct grasp types.
Dataset and Code:this https URL 

**Abstract (ZH)**: 灵巧机械手的手指灵活性和多指设计的适应性使其能够进行多样的交互，从而在各种环境中实现多种任务特定的抓持配置。然而，为了充分利用灵巧手的 capabilities，获取多样且高质量的抓持数据是必不可少的——无论是开发从点云预测抓持模型，训练操作策略，还是为高级任务规划提供更多操作选项。现有的数据集生成方法通常依赖于基于采样的算法或简化的力量闭合分析，这些方法往往会收敛于功率抓持，并表现出有限的多样性。在这项工作中，我们提出了一种合成大规模、多样且物理上可行的抓持的数据方法，这种方法不仅包括简单的功率抓持，还扩展到包括精细操作，如捏握和三指精度抓持。我们引入了一种严格的、可通过二次规划（QP）隐式定义的能量形式的力量闭合差分方法。此外，我们提出了一种调整的优化方法（MALA*），该方法通过基于所有样本的能量值分布动态拒绝梯度步骤来提高性能。我们广泛评估了我们的方法，并展示了在抓持多样性和最终抓持预测稳定性方面的显著改进。最后，我们提供了来自DexGraspNet的5,700个物体的新大规模抓持数据集，包含五种不同的夹爪和三种不同的抓取类型。 Dataset and Code:https://... 

---
# A Vision-Based Shared-Control Teleoperation Scheme for Controlling the Robotic Arm of a Four-Legged Robot 

**Title (ZH)**: 基于视觉的四足机器人臂共享控制远程操作方案 

**Authors**: Murilo Vinicius da Silva, Matheus Hipolito Carvalho, Juliano Negri, Thiago Segreto, Gustavo J. G. Lahr, Ricardo V. Godoy, Marcelo Becker  

**Link**: [PDF](https://arxiv.org/pdf/2508.14994)  

**Abstract**: In hazardous and remote environments, robotic systems perform critical tasks demanding improved safety and efficiency. Among these, quadruped robots with manipulator arms offer mobility and versatility for complex operations. However, teleoperating quadruped robots is challenging due to the lack of integrated obstacle detection and intuitive control methods for the robotic arm, increasing collision risks in confined or dynamically changing workspaces. Teleoperation via joysticks or pads can be non-intuitive and demands a high level of expertise due to its complexity, culminating in a high cognitive load on the operator. To address this challenge, a teleoperation approach that directly maps human arm movements to the robotic manipulator offers a simpler and more accessible solution. This work proposes an intuitive remote control by leveraging a vision-based pose estimation pipeline that utilizes an external camera with a machine learning-based model to detect the operator's wrist position. The system maps these wrist movements into robotic arm commands to control the robot's arm in real-time. A trajectory planner ensures safe teleoperation by detecting and preventing collisions with both obstacles and the robotic arm itself. The system was validated on the real robot, demonstrating robust performance in real-time control. This teleoperation approach provides a cost-effective solution for industrial applications where safety, precision, and ease of use are paramount, ensuring reliable and intuitive robotic control in high-risk environments. 

**Abstract (ZH)**: 在恶劣和偏远环境中，机器人系统执行需要改进安全性和效率的关键任务。其中，具有 manipulator 臂的四足机器人提供复杂操作所需的机动性和多功能性。然而，远程操作四足机器人由于缺乏集成的障碍检测和直观的控制方法而具有挑战性，增加了在受限或动态变化的工作空间中发生碰撞的风险。使用操纵杆或按键进行远程操控可能缺乏直观性，并且由于其复杂性要求高度的专业技能，从而给操作者带来较高的认知负荷。为解决这一挑战，一种直接将人类手臂运动映射到机器人 manipulator 的远程操控方法提供了一种更简单、更易操作的解决方案。本研究提出了一种直观的远程控制方法，利用基于视觉的姿态估计流水线，借助外部摄像头和基于机器学习的模型来检测操作者的腕部位置。系统将这些手腕运动映射为机器人手臂命令，以实现实时控制。轨迹规划器通过检测并与障碍物及机器人手臂本身发生碰撞来进行安全远程操控。该系统已在真实机器人上进行了验证，显示了在实时控制中表现出的稳健性能。这种远程操控方法为工业应用提供了一种经济有效的解决方案，其中安全、精度和易用性至关重要，确保在高风险环境中实现可靠且直观的机器人控制。 

---
# Discrete VHCs for Propeller Motion of a Devil-Stick using purely Impulsive Inputs 

**Title (ZH)**: 离散化VHCs在devil-stick桨动中的纯冲量输入研究 

**Authors**: Aakash Khandelwal, Ranjan Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2508.15040)  

**Abstract**: The control problem of realizing propeller motion of a devil-stick in the vertical plane using impulsive forces applied normal to the stick is considered. This problem is an example of underactuated robotic juggling and has not been considered in the literature before. Inspired by virtual holonomic constraints, the concept of discrete virtual holonomic constraints (DVHC) is introduced for the first time to solve this orbital stabilization problem. At the discrete instants when impulsive inputs are applied, the location of the center-of-mass of the devil-stick is specified in terms of its orientation angle. This yields the discrete zero dynamics (DZD), which provides conditions for stable propeller motion. In the limiting case, when the rotation angle between successive applications of impulsive inputs is chosen to be arbitrarily small, the problem reduces to that of propeller motion under continuous forcing. A controller that enforces the DVHC, and an orbit stabilizing controller based on the impulse controlled Poincaré map approach are presented. The efficacy of the approach to trajectory design and stabilization is validated through simulations. 

**Abstract (ZH)**: 使用作用于魔棍上的脉冲力在垂直平面内实现魔棍旋翼运动的控制问题：基于离散虚拟约束的概念解决轨道稳定问题 

---
