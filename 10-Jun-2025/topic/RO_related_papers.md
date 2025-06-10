# Design and Implementation of a Peer-to-Peer Communication, Modular and Decentral YellowCube UUV 

**Title (ZH)**: 面向模块化与去中心化的Peer-to-Peer通信黄立方无人潜水器设计与实现 

**Authors**: Zhizun Xu, Baozhu Jia, Weichao Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07924)  

**Abstract**: The underwater Unmanned Vehicles(UUVs) are pivot tools for offshore engineering and oceanographic research. Most existing UUVs do not facilitate easy integration of new or upgraded sensors. A solution to this problem is to have a modular UUV system with changeable payload sections capable of carrying different sensor to suite different missions. The design and implementation of a modular and decentral UUV named YellowCube is presented in the paper. Instead a centralised software architecture which is adopted by the other modular underwater vehicles designs, a Peer-To-Peer(P2P) communication mechanism is implemented among the UUV's modules. The experiments in the laboratory and sea trials have been executed to verify the performances of the UUV. 

**Abstract (ZH)**: 模块化和去中心化的水下无人车辆YellowCube的设计与实现 

---
# Primal-Dual iLQR for GPU-Accelerated Learning and Control in Legged Robots 

**Title (ZH)**: 基于原始对偶iLQR的GPU加速腿足机器人学习与控制 

**Authors**: Lorenzo Amatucci, João Sousa-Pinto, Giulio Turrisi, Dominique Orban, Victor Barasuol, Claudio Semini  

**Link**: [PDF](https://arxiv.org/pdf/2506.07823)  

**Abstract**: This paper introduces a novel Model Predictive Control (MPC) implementation for legged robot locomotion that leverages GPU parallelization. Our approach enables both temporal and state-space parallelization by incorporating a parallel associative scan to solve the primal-dual Karush-Kuhn-Tucker (KKT) system. In this way, the optimal control problem is solved in $\mathcal{O}(n\log{N} + m)$ complexity, instead of $\mathcal{O}(N(n + m)^3)$, where $n$, $m$, and $N$ are the dimension of the system state, control vector, and the length of the prediction horizon. We demonstrate the advantages of this implementation over two state-of-the-art solvers (acados and crocoddyl), achieving up to a 60\% improvement in runtime for Whole Body Dynamics (WB)-MPC and a 700\% improvement for Single Rigid Body Dynamics (SRBD)-MPC when varying the prediction horizon length. The presented formulation scales efficiently with the problem state dimensions as well, enabling the definition of a centralized controller for up to 16 legged robots that can be computed in less than 25 ms. Furthermore, thanks to the JAX implementation, the solver supports large-scale parallelization across multiple environments, allowing the possibility of performing learning with the MPC in the loop directly in GPU. 

**Abstract (ZH)**: 一种基于GPU并行化的腿式机器人运动新型模型预测控制实现 

---
# SMaRCSim: Maritime Robotics Simulation Modules 

**Title (ZH)**: SMaRCSim: 海洋机器人仿真模块 

**Authors**: Mart Kartašev, David Dörner, Özer Özkahraman, Petter Ögren, Ivan Stenius, John Folkesson  

**Link**: [PDF](https://arxiv.org/pdf/2506.07781)  

**Abstract**: Developing new functionality for underwater robots and testing them in the real world is time-consuming and resource-intensive. Simulation environments allow for rapid testing before field deployment. However, existing tools lack certain functionality for use cases in our project: i) developing learning-based methods for underwater vehicles; ii) creating teams of autonomous underwater, surface, and aerial vehicles; iii) integrating the simulation with mission planning for field experiments. A holistic solution to these problems presents great potential for bringing novel functionality into the underwater domain. In this paper we present SMaRCSim, a set of simulation packages that we have developed to help us address these issues. 

**Abstract (ZH)**: 开发水下机器人的新功能并在实际环境中测试是耗时且资源密集的。仿真环境允许在实地部署前快速测试。然而，现有工具缺乏我们项目中某些使用案例所需的特定功能：i) 开发基于学习的方法用于水下车辆；ii) 创建自主水下、水面和空中车辆的团队；iii) 将仿真与实地试验的使命规划集成。为这些难题提供一个整体解决方案具有将新颖功能引入水下领域的巨大潜力。在本文中，我们介绍了我们开发的一套仿真包SMaRCSim，以帮助我们解决这些问题。 

---
# RAPID Hand: A Robust, Affordable, Perception-Integrated, Dexterous Manipulation Platform for Generalist Robot Autonomy 

**Title (ZH)**: RAPID 手：一种 robust、经济、感知集成、灵巧的操作平台，适用于通用机器人自主性 

**Authors**: Zhaoliang Wan, Zetong Bi, Zida Zhou, Hao Ren, Yiming Zeng, Yihan Li, Lu Qi, Xu Yang, Ming-Hsuan Yang, Hui Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.07490)  

**Abstract**: This paper addresses the scarcity of low-cost but high-dexterity platforms for collecting real-world multi-fingered robot manipulation data towards generalist robot autonomy. To achieve it, we propose the RAPID Hand, a co-optimized hardware and software platform where the compact 20-DoF hand, robust whole-hand perception, and high-DoF teleoperation interface are jointly designed. Specifically, RAPID Hand adopts a compact and practical hand ontology and a hardware-level perception framework that stably integrates wrist-mounted vision, fingertip tactile sensing, and proprioception with sub-7 ms latency and spatial alignment. Collecting high-quality demonstrations on high-DoF hands is challenging, as existing teleoperation methods struggle with precision and stability on complex multi-fingered systems. We address this by co-optimizing hand design, perception integration, and teleoperation interface through a universal actuation scheme, custom perception electronics, and two retargeting constraints. We evaluate the platform's hardware, perception, and teleoperation interface. Training a diffusion policy on collected data shows superior performance over prior works, validating the system's capability for reliable, high-quality data collection. The platform is constructed from low-cost and off-the-shelf components and will be made public to ensure reproducibility and ease of adoption. 

**Abstract (ZH)**: 本文解决了低本钱但高灵巧度平台收集现实世界多指机器人操作数据以实现通用机器人自主性的稀缺问题。为此，我们提出了一种协同优化硬件和软件平台——RAPID手，该平台集成了紧凑的20自由度手、稳健的整体手部感知以及高自由度远程操作界面。具体而言，RAPID手采用了紧凑且实用的手部本体论和硬件级感知框架，该框架能够稳定地整合腕部摄像头、指尖触觉感知和 proprioception，且具有亚7毫秒的延迟和空间对齐。在高自由度手上收集高质量演示动作具有挑战性，现有远程操作方法在复杂多指系统上难以实现精确性和稳定性。我们通过通用驱动方案、定制感知电子学以及两套适配约束，协同优化了手部设计、感知集成和远程操作界面。我们评估了该平台的硬件、感知能力和远程操作界面。基于收集的数据训练扩散策略显示了优于现有工作的性能，验证了该系统可靠地收集高质量数据的能力。该平台由低成本和现成组件构建，并将公开发布以确保可重复性和易于采用。 

---
# UruBots Autonomous Cars Challenge Pro Team Description Paper for FIRA 2025 

**Title (ZH)**: UruBots自主汽车挑战专业团队描述论文：FIRA 2025 

**Authors**: Pablo Moraes, Mónica Rodríguez, Sebastian Barcelona, Angel Da Silva, Santiago Fernandez, Hiago Sodre, Igor Nunes, Bruna Guterres, Ricardo Grando  

**Link**: [PDF](https://arxiv.org/pdf/2506.07348)  

**Abstract**: This paper describes the development of an autonomous car by the UruBots team for the 2025 FIRA Autonomous Cars Challenge (Pro). The project involves constructing a compact electric vehicle, approximately the size of an RC car, capable of autonomous navigation through different tracks. The design incorporates mechanical and electronic components and machine learning algorithms that enable the vehicle to make real-time navigation decisions based on visual input from a camera. We use deep learning models to process camera images and control vehicle movements. Using a dataset of over ten thousand images, we trained a Convolutional Neural Network (CNN) to drive the vehicle effectively, through two outputs, steering and throttle. The car completed the track in under 30 seconds, achieving a pace of approximately 0.4 meters per second while avoiding obstacles. 

**Abstract (ZH)**: 乌鲁机器人团队2025年FIRA自主汽车挑战赛（专业组）的自主汽车开发研究 

---
# Reproducibility in the Control of Autonomous Mobility-on-Demand Systems 

**Title (ZH)**: 自主出行系统控制中的可重复性研究 

**Authors**: Xinling Li, Meshal Alharbi, Daniele Gammelli, James Harrison, Filipe Rodrigues, Maximilian Schiffer, Marco Pavone, Emilio Frazzoli, Jinhua Zhao, Gioele Zardini  

**Link**: [PDF](https://arxiv.org/pdf/2506.07345)  

**Abstract**: Autonomous Mobility-on-Demand (AMoD) systems, powered by advances in robotics, control, and Machine Learning (ML), offer a promising paradigm for future urban transportation. AMoD offers fast and personalized travel services by leveraging centralized control of autonomous vehicle fleets to optimize operations and enhance service performance. However, the rapid growth of this field has outpaced the development of standardized practices for evaluating and reporting results, leading to significant challenges in reproducibility. As AMoD control algorithms become increasingly complex and data-driven, a lack of transparency in modeling assumptions, experimental setups, and algorithmic implementation hinders scientific progress and undermines confidence in the results. This paper presents a systematic study of reproducibility in AMoD research. We identify key components across the research pipeline, spanning system modeling, control problems, simulation design, algorithm specification, and evaluation, and analyze common sources of irreproducibility. We survey prevalent practices in the literature, highlight gaps, and propose a structured framework to assess and improve reproducibility. Specifically, concrete guidelines are offered, along with a "reproducibility checklist", to support future work in achieving replicable, comparable, and extensible results. While focused on AMoD, the principles and practices we advocate generalize to a broader class of cyber-physical systems that rely on networked autonomy and data-driven control. This work aims to lay the foundation for a more transparent and reproducible research culture in the design and deployment of intelligent mobility systems. 

**Abstract (ZH)**: 基于自主机器人、控制和机器学习技术的按需自主移动（AMoD）系统为未来城市交通提供了有希望的范式。本文系统研究了AMoD研究中的可重复性问题，识别了研究管道中的关键组件，包括系统建模、控制问题、仿真设计、算法规范和评估，并分析了不可重复性的常见来源。本文还概述了具体的指导意见，并提出了“可重复性检查表”来支持未来工作中实现可重复、可比较和可扩展的结果。虽然重点是AMoD系统，但所倡导的原则和实践适用于更广泛依赖网络自主性和数据驱动控制的网络物理系统。这项工作旨在为智能移动系统的设计和部署建立一个更透明和可重复的研究文化奠定基础。 

---
# Very Large-scale Multi-Robot Task Allocation in Challenging Environments via Robot Redistribution 

**Title (ZH)**: 挑战环境下的大规模多机器人任务分配通过机器人重新分配 

**Authors**: Seabin Lee, Joonyeol Sim, Changjoo Nam  

**Link**: [PDF](https://arxiv.org/pdf/2506.07293)  

**Abstract**: We consider the Multi-Robot Task Allocation (MRTA) problem that aims to optimize an assignment of multiple robots to multiple tasks in challenging environments which are with densely populated obstacles and narrow passages. In such environments, conventional methods optimizing the sum-of-cost are often ineffective because the conflicts between robots incur additional costs (e.g., collision avoidance, waiting). Also, an allocation that does not incorporate the actual robot paths could cause deadlocks, which significantly degrade the collective performance of the robots.
We propose a scalable MRTA method that considers the paths of the robots to avoid collisions and deadlocks which result in a fast completion of all tasks (i.e., minimizing the \textit{makespan}). To incorporate robot paths into task allocation, the proposed method constructs a roadmap using a Generalized Voronoi Diagram. The method partitions the roadmap into several components to know how to redistribute robots to achieve all tasks with less conflicts between the robots. In the redistribution process, robots are transferred to their final destinations according to a push-pop mechanism with the first-in first-out principle. From the extensive experiments, we show that our method can handle instances with hundreds of robots in dense clutter while competitors are unable to compute a solution within a time limit. 

**Abstract (ZH)**: 多机器人任务分配中的路径考虑方法：避免碰撞与死锁以优化完成任务时间 

---
# Machine Learning-Based Self-Localization Using Internal Sensors for Automating Bulldozers 

**Title (ZH)**: 基于机器学习的内部传感器自我定位方法及其在自动化推土机中的应用 

**Authors**: Hikaru Sawafuji, Ryota Ozaki, Takuto Motomura, Toyohisa Matsuda, Masanori Tojima, Kento Uchida, Shinichi Shirakawa  

**Link**: [PDF](https://arxiv.org/pdf/2506.07271)  

**Abstract**: Self-localization is an important technology for automating bulldozers. Conventional bulldozer self-localization systems rely on RTK-GNSS (Real Time Kinematic-Global Navigation Satellite Systems). However, RTK-GNSS signals are sometimes lost in certain mining conditions. Therefore, self-localization methods that do not depend on RTK-GNSS are required. In this paper, we propose a machine learning-based self-localization method for bulldozers. The proposed method consists of two steps: estimating local velocities using a machine learning model from internal sensors, and incorporating these estimates into an Extended Kalman Filter (EKF) for global localization. We also created a novel dataset for bulldozer odometry and conducted experiments across various driving scenarios, including slalom, excavation, and driving on slopes. The result demonstrated that the proposed self-localization method suppressed the accumulation of position errors compared to kinematics-based methods, especially when slip occurred. Furthermore, this study showed that bulldozer-specific sensors, such as blade position sensors and hydraulic pressure sensors, contributed to improving self-localization accuracy. 

**Abstract (ZH)**: 基于机器学习的推土机自定位方法 

---
# MorphoCopter: Design, Modeling, and Control of a New Transformable Quad-Bi Copter 

**Title (ZH)**: MorphoCopter：新型变形四轴双旋翼飞行器的设计、建模与控制 

**Authors**: Harsh Modi, Hao Su, Xiao Liang, Minghui Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.07204)  

**Abstract**: This paper presents a novel morphing quadrotor, named MorphoCopter, covering its design, modeling, control, and experimental tests. It features a unique single rotary joint that enables rapid transformation into an ultra-narrow profile. Although quadrotors have seen widespread adoption in applications such as cinematography, agriculture, and disaster management with increasingly sophisticated control systems, their hardware configurations have remained largely unchanged, limiting their capabilities in certain environments. Our design addresses this by enabling the hardware configuration to change on the fly when required. In standard flight mode, the MorphoCopter adopts an X configuration, functioning as a traditional quadcopter, but can quickly fold into a stacked bicopters arrangement or any configuration in between. Existing morphing designs often sacrifice controllability in compact configurations or rely on complex multi-joint systems. Moreover, our design achieves a greater width reduction than any existing solution. We develop a new inertia and control-action aware adaptive control system that maintains robust performance across all rotary-joint configurations. The prototype can reduce its width from 447 mm to 138 mm (nearly 70\% reduction) in just a few seconds. We validated the MorphoCopter through rigorous simulations and a comprehensive series of flight experiments, including robustness tests, trajectory tracking, and narrow-gap passing tests. 

**Abstract (ZH)**: 一种新型变形四旋翼机MorphoCopter的设计、建模、控制及实验研究 

---
# RF-Source Seeking with Obstacle Avoidance using Real-time Modified Artificial Potential Fields in Unknown Environments 

**Title (ZH)**: 基于实时修改人工势场的未知环境中的RF-源搜索与避障 

**Authors**: Shahid Mohammad Mulla, Aryan Kanakapudi, Lakshmi Narasimhan, Anuj Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2506.06811)  

**Abstract**: Navigation of UAVs in unknown environments with obstacles is essential for applications in disaster response and infrastructure monitoring. However, existing obstacle avoidance algorithms, such as Artificial Potential Field (APF) are unable to generalize across environments with different obstacle configurations. Furthermore, the precise location of the final target may not be available in applications such as search and rescue, in which case approaches such as RF source seeking can be used to align towards the target location. This paper proposes a real-time trajectory planning method, which involves real-time adaptation of APF through a sampling-based approach. The proposed approach utilizes only the bearing angle of the target without its precise location, and adjusts the potential field parameters according to the environment with new obstacle configurations in real time. The main contributions of the article are i) an RF source seeking algorithm to provide a bearing angle estimate using RF signal calculations based on antenna placement, and ii) a modified APF for adaptable collision avoidance in changing environments, which are evaluated separately in the simulation software Gazebo, using ROS2 for communication. Simulation results show that the RF source-seeking algorithm achieves high accuracy, with an average angular error of just 1.48 degrees, and with this estimate, the proposed navigation algorithm improves the success rate of reaching the target by 46% and reduces the trajectory length by 1.2% compared to standard potential fields. 

**Abstract (ZH)**: 无人机在未知环境中的障碍物导航对于灾难响应和基础设施监测应用至关重要。然而，现有的障碍物规避算法，如人工势场法（APF），在不同障碍配置的环境中无法泛化。此外，在搜索与救援等应用场景中，目标的精确位置可能不可得，此时可以使用RF信号追踪方法对准目标位置。本文提出了一种实时轨迹规划方法，该方法通过采样方法实时调整APF。所提出的方法仅使用目标的方向角而无需其精确位置，并根据新障碍配置的环境实时调整势场参数。文章的主要贡献包括：i) 一种RF信号追踪算法，基于天线布局进行RF信号计算以提供方向角估计；ii) 一种修改的APF方法，用于适应性碰撞规避以应对变化的环境。这些方法在仿真软件Gazebo中分别进行评估，并使用ROS2进行通信。仿真结果显示，RF信号追踪算法具有高精度，平均角误差仅为1.48度，使用该估计值，所提出的导航算法将到达目标的成功率提高了46%，并使轨迹长度减少了1.2%。 

---
# SARAL-Bot: Autonomous Robot for Strawberry Plant Care 

**Title (ZH)**: SARAL-Bot: 自主草莓植物护理机器人 

**Authors**: Arif Ahmed, Ritvik Agarwal, Gaurav Srikar, Nathaniel Rose, Parikshit Maini  

**Link**: [PDF](https://arxiv.org/pdf/2506.06798)  

**Abstract**: Strawberry farming demands intensive labor for monitoring and maintaining plant health. To address this, Team SARAL develops an autonomous robot for the 2024 ASABE Student Robotics Challenge, capable of navigation, unhealthy leaf detection, and removal. The system addresses labor shortages, reduces costs, and supports sustainable farming through vision-based plant assessment. This work demonstrates the potential of robotics to modernize strawberry cultivation and enable scalable, intelligent agricultural solutions. 

**Abstract (ZH)**: 草莓种植需要密集的人力来监测和维持植物健康。为了解决这一问题，SARAL团队开发了一款 autonomous 机器人参加2024年ASABE Student Robotics Challenge，该机器人具备导航、不健康叶片检测和移除功能。该系统解决了劳动力短缺问题，降低了成本，并通过基于视觉的植物评估支持可持续农业。这项工作展示了机器人技术在现代草莓种植中的潜力及其在可扩展、智能农业解决方案中的应用。 

---
# NeSyPack: A Neuro-Symbolic Framework for Bimanual Logistics Packing 

**Title (ZH)**: NeSyPack: 一种双臂物流包装的神经符号框架 

**Authors**: Bowei Li, Peiqi Yu, Zhenran Tang, Han Zhou, Yifan Sun, Ruixuan Liu, Changliu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06567)  

**Abstract**: This paper presents NeSyPack, a neuro-symbolic framework for bimanual logistics packing. NeSyPack combines data-driven models and symbolic reasoning to build an explainable hierarchical system that is generalizable, data-efficient, and reliable. It decomposes a task into subtasks via hierarchical reasoning, and further into atomic skills managed by a symbolic skill graph. The graph selects skill parameters, robot configurations, and task-specific control strategies for execution. This modular design enables robustness, adaptability, and efficient reuse - outperforming end-to-end models that require large-scale retraining. Using NeSyPack, our team won the First Prize in the What Bimanuals Can Do (WBCD) competition at the 2025 IEEE International Conference on Robotics and Automation. 

**Abstract (ZH)**: 本论文介绍了NeSyPack，一种用于双臂物流包装的神经符号框架。NeSyPack 结合数据驱动模型和符号推理，构建了一个可解释的分层系统，该系统具有通用性、数据高效性和可靠性。该框架通过分层推理将任务分解为子任务，进一步分解为由符号技能图管理的基本技能。该图选择技能参数、机器人配置和任务特定的控制策略以供执行。这种模块化设计增强了系统的鲁棒性、适应性和高效复用性，超越了需要大规模重新训练的端到端模型。使用NeSyPack，我们的团队在2025年IEEE国际机器人与自动化会议上举办的What Bimanuals Can Do (WBCD) 竞赛中获得了第一名。 

---
# Tactile MNIST: Benchmarking Active Tactile Perception 

**Title (ZH)**: 触觉MNIST：活性触觉感知基准测试 

**Authors**: Tim Schneider, Guillaume Duret, Cristiana de Farias, Roberto Calandra, Liming Chen, Jan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2506.06361)  

**Abstract**: Tactile perception has the potential to significantly enhance dexterous robotic manipulation by providing rich local information that can complement or substitute for other sensory modalities such as vision. However, because tactile sensing is inherently local, it is not well-suited for tasks that require broad spatial awareness or global scene understanding on its own. A human-inspired strategy to address this issue is to consider active perception techniques instead. That is, to actively guide sensors toward regions with more informative or significant features and integrate such information over time in order to understand a scene or complete a task. Both active perception and different methods for tactile sensing have received significant attention recently. Yet, despite advancements, both fields lack standardized benchmarks. To bridge this gap, we introduce the Tactile MNIST Benchmark Suite, an open-source, Gymnasium-compatible benchmark specifically designed for active tactile perception tasks, including localization, classification, and volume estimation. Our benchmark suite offers diverse simulation scenarios, from simple toy environments all the way to complex tactile perception tasks using vision-based tactile sensors. Furthermore, we also offer a comprehensive dataset comprising 13,500 synthetic 3D MNIST digit models and 153,600 real-world tactile samples collected from 600 3D printed digits. Using this dataset, we train a CycleGAN for realistic tactile simulation rendering. By providing standardized protocols and reproducible evaluation frameworks, our benchmark suite facilitates systematic progress in the fields of tactile sensing and active perception. 

**Abstract (ZH)**: 触觉感知有潜力通过提供丰富的本地信息来显著增强灵巧的机器人操作，这些信息可以补充或替代其他感官模态（如视觉）。然而，由于触觉感知本质上是局部的，它单独使用时并不适合需要广泛空间意识或全局场景理解的任务。受人类策略启发的解决思路是考虑主动感知技术，即主动引导传感器朝向更有信息量或更显著特征的区域，并通过时间上的信息整合来理解和完成任务。主动感知和不同类型的触觉传感方法近年来受到了广泛关注。尽管取得了进展，但两个领域仍缺乏标准基准。为填补这一空白，我们引入了触觉MNIST基准套件，这是一个开源的、兼容Gymnasium的基准，专门针对主动触觉感知任务，包括定位、分类和体积估计。我们的基准套件提供了多样化的模拟场景，从简单的玩具环境到基于视觉的触觉传感器进行的复杂触觉感知任务。此外，我们还提供了一个详尽的数据集，包含13,500个3D合成MNIST数字模型和153,600个来自600个3D打印数字的真实世界的触觉样本。利用该数据集，我们训练了一个CycleGAN进行逼真的触觉仿真渲染。通过提供标准化协议和可重复的评估框架，我们的基准套件促进了触觉传感和主动感知领域的系统性进步。 

---
# Towards Data-Driven Model-Free Safety-Critical Control 

**Title (ZH)**: 基于数据驱动的模型自由的安全关键控制 

**Authors**: Zhe Shen, Yitaek Kim, Christoffer Sloth  

**Link**: [PDF](https://arxiv.org/pdf/2506.06931)  

**Abstract**: This paper presents a framework for enabling safe velocity control of general robotic systems using data-driven model-free Control Barrier Functions (CBFs). Model-free CBFs rely on an exponentially stable velocity controller and a design parameter (e.g. alpha in CBFs); this design parameter depends on the exponential decay rate of the controller. However, in practice, the decay rate is often unavailable, making it non-trivial to use model-free CBFs, as it requires manual tuning for alpha. To address this, a Neural Network is used to learn the Lyapunov function from data, and the maximum decay rate of the systems built-in velocity controller is subsequently estimated. Furthermore, to integrate the estimated decay rate with model-free CBFs, we derive a probabilistic safety condition that incorporates a confidence bound on the violation rate of the exponential stability condition, using Chernoff bound. This enhances robustness against uncertainties in stability violations. The proposed framework has been tested on a UR5e robot in multiple experimental settings, and its effectiveness in ensuring safe velocity control with model-free CBFs has been demonstrated. 

**Abstract (ZH)**: 基于数据驱动的控Barrier函数框架：实现通用机器人系统的安全速度控制 

---
# DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning 

**Title (ZH)**: DriveSuprim:向精确轨迹选择的端到端规划迈进 

**Authors**: Wenhao Yao, Zhenxin Li, Shiyi Lan, Zi Wang, Xinglong Sun, Jose M. Alvarez, Zuxuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06659)  

**Abstract**: In complex driving environments, autonomous vehicles must navigate safely. Relying on a single predicted path, as in regression-based approaches, usually does not explicitly assess the safety of the predicted trajectory. Selection-based methods address this by generating and scoring multiple trajectory candidates and predicting the safety score for each, but face optimization challenges in precisely selecting the best option from thousands of possibilities and distinguishing subtle but safety-critical differences, especially in rare or underrepresented scenarios. We propose DriveSuprim to overcome these challenges and advance the selection-based paradigm through a coarse-to-fine paradigm for progressive candidate filtering, a rotation-based augmentation method to improve robustness in out-of-distribution scenarios, and a self-distillation framework to stabilize training. DriveSuprim achieves state-of-the-art performance, reaching 93.5% PDMS in NAVSIM v1 and 87.1% EPDMS in NAVSIM v2 without extra data, demonstrating superior safetycritical capabilities, including collision avoidance and compliance with rules, while maintaining high trajectory quality in various driving scenarios. 

**Abstract (ZH)**: 在复杂驾驶环境中，自动驾驶车辆必须安全导航。依赖单一预测路径的方法，如基于回归的方法，通常不会明确评估预测轨迹的安全性。选择性方法通过生成和评分多个轨迹候选方案，并为每个方案预测安全得分来解决这一问题，但在从成千上万种可能性中精确选择最佳选项和区分细微但至关安全的差异方面面临优化挑战，尤其是在罕见或未充分代表的场景中。我们提出DriveSuprim以克服这些挑战并通过粗到细的渐进候选过滤 paradigmm、基于旋转的增强方法提高分布外场景的鲁棒性以及自我蒸馏框架稳定训练来推进选择性方法的范式。DriveSuprim在不使用额外数据的情况下达到了NAVSIM v1的93.5% PDMS和NAVSIM v2的87.1% EPDMS，展示出了优越的安全关键能力，包括碰撞避免和规则遵守，并在各种驾驶场景中保持了高质量的轨迹。 

---
