# Coordinated Motion Planning of a Wearable Multi-Limb System for Enhanced Human-Robot Interaction 

**Title (ZH)**: 可穿戴多肢系统协调运动规划以增强人机交互 

**Authors**: Chaerim Moon, Joohyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.10444)  

**Abstract**: Supernumerary Robotic Limbs (SRLs) can enhance human capability within close proximity. However, as a wearable device, the generated moment from its operation acts on the human body as an external torque. When the moments increase, more muscle units are activated for balancing, and it can result in reduced muscular null space. Therefore, this paper suggests a concept of a motion planning layer that reduces the generated moment for enhanced Human-Robot Interaction. It modifies given trajectories with desirable angular acceleration and position deviation limits. Its performance to reduce the moment is demonstrated through the simulation, which uses simplified human and robotic system models. 

**Abstract (ZH)**: 超额机器人肢体（SRLs）在近距离内可增强人类能力。然而，作为可穿戴设备，其操作产生的力矩作用于人体作为外部扭距。当力矩增加时，更多的肌肉单元被激活以维持平衡，这可能导致肌肉余度空间减少。因此，本文提出了一种运动规划层的概念，以减少生成的力矩，从而提升人机交互。该概念通过修改给定轨迹以满足期望的角加速度和位置偏差限制来实现。其减少力矩的性能通过使用简化的仿人和机器人系统模型进行的仿真来验证。 

---
# Acetrans: An Autonomous Corridor-Based and Efficient UAV Suspended Transport System 

**Title (ZH)**: Acetrans: 一种自主走廊基高效无人机悬挂运输系统 

**Authors**: Weiyan Lu, Huizhe Li, Yuhao Fang, Zhexuan Zhou, Junda Wu, Yude Li, Youmin Gong, Jie Mei  

**Link**: [PDF](https://arxiv.org/pdf/2509.10349)  

**Abstract**: Unmanned aerial vehicles (UAVs) with suspended payloads offer significant advantages for aerial transportation in complex and cluttered environments. However, existing systems face critical limitations, including unreliable perception of the cable-payload dynamics, inefficient planning in large-scale environments, and the inability to guarantee whole-body safety under cable bending and external disturbances. This paper presents Acetrans, an Autonomous, Corridor-based, and Efficient UAV suspended transport system that addresses these challenges through a unified perception, planning, and control framework. A LiDAR-IMU fusion module is proposed to jointly estimate both payload pose and cable shape under taut and bent modes, enabling robust whole-body state estimation and real-time filtering of cable point clouds. To enhance planning scalability, we introduce the Multi-size-Aware Configuration-space Iterative Regional Inflation (MACIRI) algorithm, which generates safe flight corridors while accounting for varying UAV and payload geometries. A spatio-temporal, corridor-constrained trajectory optimization scheme is then developed to ensure dynamically feasible and collision-free trajectories. Finally, a nonlinear model predictive controller (NMPC) augmented with cable-bending constraints provides robust whole-body safety during execution. Simulation and experimental results validate the effectiveness of Acetrans, demonstrating substantial improvements in perception accuracy, planning efficiency, and control safety compared to state-of-the-art methods. 

**Abstract (ZH)**: 自主基于走廊高效悬吊运输的自主无人机系统（Acetrans） 

---
# DiffAero: A GPU-Accelerated Differentiable Simulation Framework for Efficient Quadrotor Policy Learning 

**Title (ZH)**: DiffAero：一种基于GPU加速的可微分模拟框架，用于高效 quadrotor 策略学习 

**Authors**: Xinhong Zhang, Runqing Wang, Yunfan Ren, Jian Sun, Hao Fang, Jie Chen, Gang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10247)  

**Abstract**: This letter introduces DiffAero, a lightweight, GPU-accelerated, and fully differentiable simulation framework designed for efficient quadrotor control policy learning. DiffAero supports both environment-level and agent-level parallelism and integrates multiple dynamics models, customizable sensor stacks (IMU, depth camera, and LiDAR), and diverse flight tasks within a unified, GPU-native training interface. By fully parallelizing both physics and rendering on the GPU, DiffAero eliminates CPU-GPU data transfer bottlenecks and delivers orders-of-magnitude improvements in simulation throughput. In contrast to existing simulators, DiffAero not only provides high-performance simulation but also serves as a research platform for exploring differentiable and hybrid learning algorithms. Extensive benchmarks and real-world flight experiments demonstrate that DiffAero and hybrid learning algorithms combined can learn robust flight policies in hours on consumer-grade hardware. The code is available at this https URL. 

**Abstract (ZH)**: 这篇通信介绍了DiffAero，一个轻量级、GPU加速且完全可微的仿真框架，用于高效地学习四旋翼飞行器控制策略。DiffAero支持环境级和代理级并行性，并在统一的、GPU原生训练接口中集成了多种动力学模型、可定制的传感器堆栈（IMU、深度相机和LiDAR）以及多样化的飞行任务。通过在GPU上完全并行化物理仿真和渲染，DiffAero消除了CPU-GPU数据传输瓶颈，显著提高了仿真吞吐量。与现有仿真器不同，DiffAero不仅提供了高性能仿真，还作为一个研究平台，用于探索可微分和混合学习算法。广泛的基准测试和实际飞行实验表明，DiffAero和混合学习算法结合可以在消费级硬件上几小时内学习到鲁棒的飞行策略。代码可在以下链接获取。 

---
# Prespecified-Performance Kinematic Tracking Control for Aerial Manipulation 

**Title (ZH)**: 预设性能运动跟踪控制在无人机操作中 

**Authors**: Hauzi Cao, Jiahao Shen, Zhengzhen Li, Qinquan Ren, Shiyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.10065)  

**Abstract**: This paper studies the kinematic tracking control problem for aerial manipulators. Existing kinematic tracking control methods, which typically employ proportional-derivative feedback or tracking-error-based feedback strategies, may fail to achieve tracking objectives within specified time constraints. To address this limitation, we propose a novel control framework comprising two key components: end-effector tracking control based on a user-defined preset trajectory and quadratic programming-based reference allocation. Compared with state-of-the-art approaches, the proposed method has several attractive features. First, it ensures that the end-effector reaches the desired position within a preset time while keeping the tracking error within a performance envelope that reflects task requirements. Second, quadratic programming is employed to allocate the references of the quadcopter base and the Delta arm, while considering the physical constraints of the aerial manipulator, thus preventing solutions that may violate physical limitations. The proposed approach is validated through three experiments. Experimental results demonstrate the effectiveness of the proposed algorithm and its capability to guarantee that the target position is reached within the preset time. 

**Abstract (ZH)**: 本文研究了空中 manipulator 的运动跟踪控制问题。现有的运动跟踪控制方法通常采用比例-微分反馈或基于跟踪误差的反馈策略，可能无法在指定的时间范围内实现跟踪目标。为解决这一限制，我们提出了一种新颖的控制框架，包含两个关键组成部分：基于用户定义预设轨迹的末端执行器跟踪控制和基于二次规划的参考分配。与现有方法相比，所提出的方法具有几个吸引人的特点。首先，它确保末端执行器在预设时间内达到期望位置，同时将跟踪误差保持在反映任务要求的性能包络内。其次，二次规划用于分配四旋翼底座和Delta臂的参考值，并考虑空中 manipulator 的物理约束，从而防止违反物理限制的解。所提出的方法通过三个实验得到验证。实验结果表明了所提算法的有效性及其在预设时间内达到目标位置的能力。 

---
# TwinTac: A Wide-Range, Highly Sensitive Tactile Sensor with Real-to-Sim Digital Twin Sensor Model 

**Title (ZH)**: TwinTac: 一种宽范围高性能触觉传感器及其实时到仿真数字孪生传感器模型 

**Authors**: Xiyan Huang, Zhe Xu, Chenxi Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2509.10063)  

**Abstract**: Robot skill acquisition processes driven by reinforcement learning often rely on simulations to efficiently generate large-scale interaction data. However, the absence of simulation models for tactile sensors has hindered the use of tactile sensing in such skill learning processes, limiting the development of effective policies driven by tactile perception. To bridge this gap, we present TwinTac, a system that combines the design of a physical tactile sensor with its digital twin model. Our hardware sensor is designed for high sensitivity and a wide measurement range, enabling high quality sensing data essential for object interaction tasks. Building upon the hardware sensor, we develop the digital twin model using a real-to-sim approach. This involves collecting synchronized cross-domain data, including finite element method results and the physical sensor's outputs, and then training neural networks to map simulated data to real sensor responses. Through experimental evaluation, we characterized the sensitivity of the physical sensor and demonstrated the consistency of the digital twin in replicating the physical sensor's output. Furthermore, by conducting an object classification task, we showed that simulation data generated by our digital twin sensor can effectively augment real-world data, leading to improved accuracy. These results highlight TwinTac's potential to bridge the gap in cross-domain learning tasks. 

**Abstract (ZH)**: 基于孪生模型的物理触觉传感器及其数字孪生系统在强化学习驱动的机器人技能获取中的应用 

---
# Design and Evaluation of Two Spherical Systems for Mobile 3D Mapping 

**Title (ZH)**: 设计与评估两种球面系统在移动3D测绘中的应用 

**Authors**: Marawan Khalil, Fabian Arzberger, Andreas Nüchter  

**Link**: [PDF](https://arxiv.org/pdf/2509.10032)  

**Abstract**: Spherical robots offer unique advantages for mapping applications in hazardous or confined environments, thanks to their protective shells and omnidirectional mobility. This work presents two complementary spherical mapping systems: a lightweight, non-actuated design and an actuated variant featuring internal pendulum-driven locomotion. Both systems are equipped with a Livox Mid-360 solid-state LiDAR sensor and run LiDAR-Inertial Odometry (LIO) algorithms on resource-constrained hardware. We assess the mapping accuracy of these systems by comparing the resulting 3D point-clouds from the LIO algorithms to a ground truth map. The results indicate that the performance of state-of-the-art LIO algorithms deteriorates due to the high dynamic movement introduced by the spherical locomotion, leading to globally inconsistent maps and sometimes unrecoverable drift. 

**Abstract (ZH)**: 球形机器人因其防护壳和全方位移动优势，特别适合在危险或受限环境中进行测绘应用。本研究提出了两种互补的球形测绘系统：一种轻量化非驱动设计和一种配有内部摆轮驱动移动的驱动变体。这两种系统均配备了Livox Mid-360 固态激光雷达传感器，并在资源受限硬件上运行激光雷达-惯性里程计（LIO）算法。通过将LIO算法生成的3D点云与真实地图进行比较，评估这两种系统的测绘精度。结果表明，由于球形运动引入的高动态移动导致最先进的LIO算法性能下降，造成全局不一致的测绘结果，并且有时会导致无法恢复的漂移。 

---
# Towards simulation-based optimization of compliant fingers for high-speed connector assembly 

**Title (ZH)**: 基于仿真的柔性手指优化以实现高速连接器装配 

**Authors**: Richard Matthias Hartisch, Alexander Rother, Jörg Krüger, Kevin Haninger  

**Link**: [PDF](https://arxiv.org/pdf/2509.10012)  

**Abstract**: Mechanical compliance is a key design parameter for dynamic contact-rich manipulation, affecting task success and safety robustness over contact geometry variation. Design of soft robotic structures, such as compliant fingers, requires choosing design parameters which affect geometry and stiffness, and therefore manipulation performance and robustness. Today, these parameters are chosen through either hardware iteration, which takes significant development time, or simplified models (e.g. planar), which can't address complex manipulation task objectives. Improvements in dynamic simulation, especially with contact and friction modeling, present a potential design tool for mechanical compliance. We propose a simulation-based design tool for compliant mechanisms which allows design with respect to task-level objectives, such as success rate. This is applied to optimize design parameters of a structured compliant finger to reduce failure cases inside a tolerance window in insertion tasks. The improvement in robustness is then validated on a real robot using tasks from the benchmark NIST task board. The finger stiffness affects the tolerance window: optimized parameters can increase tolerable ranges by a factor of 2.29, with workpiece variation up to 8.6 mm being compensated. However, the trends remain task-specific. In some tasks, the highest stiffness yields the widest tolerable range, whereas in others the opposite is observed, motivating need for design tools which can consider application-specific geometry and dynamics. 

**Abstract (ZH)**: 机械顺应性是动态接触丰富操作的关键设计参数，影响任务成功率和接触几何变化下的安全性鲁棒性。软体机器人结构（如顺应手指）的设计需要选择影响几何和刚度的设计参数，从而影响操作性能和鲁棒性。当前，这些参数通过硬件迭代选择，耗时长，或者通过简化模型（如平面模型）选择，无法解决复杂的操作任务目标。动态仿真改进，尤其是接触和摩擦建模，为机械顺应性提供了一个潜在的设计工具。我们提出了一个基于仿真设计工具，用于顺应机制的设计，可以以任务级目标（如成功率）为导向进行设计。该工具应用于优化插入任务中结构化顺应手指的设计参数，以在容差窗口内减少失效情况。然后，通过在基准NIST任务板上的真实机器人任务中验证，验证了鲁棒性的提升。手指刚度影响容差窗口：优化参数可以将可容忍范围增加2.29倍，即使工件变化达到8.6毫米也能进行补偿。然而，趋势仍然具有任务特异性。在某些任务中，最高刚度提供最宽的可容忍范围，而在其他任务中则相反，这表明需要能够考虑应用特定几何和动态的设计工具。 

---
# Gaussian path model library for intuitive robot motion programming by demonstration 

**Title (ZH)**: 高斯路径模型库：基于示范的直观机器人运动编程 

**Authors**: Samuli Soutukorva, Markku Suomalainen, Martin Kollingbaum, Tapio Heikkilä  

**Link**: [PDF](https://arxiv.org/pdf/2509.10007)  

**Abstract**: This paper presents a system for generating Gaussian path models from teaching data representing the path shape. In addition, methods for using these path models to classify human demonstrations of paths are introduced. By generating a library of multiple Gaussian path models of various shapes, human demonstrations can be used for intuitive robot motion programming. A method for modifying existing Gaussian path models by demonstration through geometric analysis is also presented. 

**Abstract (ZH)**: 本文提出了一种从表示路径形状的教学数据生成高斯路径模型的系统。此外，介绍了使用这些路径模型对人类路径演示进行分类的方法。通过生成多种不同形状的高斯路径模型库，人类演示可以用于直观的机器人运动编程。还提出了通过几何分析修改现有高斯路径模型的方法。 

---
# Self-Augmented Robot Trajectory: Efficient Imitation Learning via Safe Self-augmentation with Demonstrator-annotated Precision 

**Title (ZH)**: 自我增强机器人轨迹：通过安全自我扩充实现的高效模仿学习 

**Authors**: Hanbit Oh, Masaki Murooka, Tomohiro Motoda, Ryoichi Nakajo, Yukiyasu Domae  

**Link**: [PDF](https://arxiv.org/pdf/2509.09893)  

**Abstract**: Imitation learning is a promising paradigm for training robot agents; however, standard approaches typically require substantial data acquisition -- via numerous demonstrations or random exploration -- to ensure reliable performance. Although exploration reduces human effort, it lacks safety guarantees and often results in frequent collisions -- particularly in clearance-limited tasks (e.g., peg-in-hole) -- thereby, necessitating manual environmental resets and imposing additional human burden. This study proposes Self-Augmented Robot Trajectory (SART), a framework that enables policy learning from a single human demonstration, while safely expanding the dataset through autonomous augmentation. SART consists of two stages: (1) human teaching only once, where a single demonstration is provided and precision boundaries -- represented as spheres around key waypoints -- are annotated, followed by one environment reset; (2) robot self-augmentation, where the robot generates diverse, collision-free trajectories within these boundaries and reconnects to the original demonstration. This design improves the data collection efficiency by minimizing human effort while ensuring safety. Extensive evaluations in simulation and real-world manipulation tasks show that SART achieves substantially higher success rates than policies trained solely on human-collected demonstrations. Video results available at this https URL . 

**Abstract (ZH)**: 自我增强机器人轨迹（SART）：一种从单个人机演示中学习的框架 

---
