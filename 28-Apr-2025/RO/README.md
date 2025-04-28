# Boxi: Design Decisions in the Context of Algorithmic Performance for Robotics 

**Title (ZH)**: 盒智能机器人算法性能视角下的设计决策 

**Authors**: Jonas Frey, Turcan Tuna, Lanke Frank Tarimo Fu, Cedric Weibel, Katharine Patterson, Benjamin Krummenacher, Matthias Müller, Julian Nubert, Maurice Fallon, Cesar Cadena, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2504.18500)  

**Abstract**: Achieving robust autonomy in mobile robots operating in complex and unstructured environments requires a multimodal sensor suite capable of capturing diverse and complementary information. However, designing such a sensor suite involves multiple critical design decisions, such as sensor selection, component placement, thermal and power limitations, compute requirements, networking, synchronization, and calibration. While the importance of these key aspects is widely recognized, they are often overlooked in academia or retained as proprietary knowledge within large corporations. To improve this situation, we present Boxi, a tightly integrated sensor payload that enables robust autonomy of robots in the wild. This paper discusses the impact of payload design decisions made to optimize algorithmic performance for downstream tasks, specifically focusing on state estimation and mapping. Boxi is equipped with a variety of sensors: two LiDARs, 10 RGB cameras including high-dynamic range, global shutter, and rolling shutter models, an RGB-D camera, 7 inertial measurement units (IMUs) of varying precision, and a dual antenna RTK GNSS system. Our analysis shows that time synchronization, calibration, and sensor modality have a crucial impact on the state estimation performance. We frame this analysis in the context of cost considerations and environment-specific challenges. We also present a mobile sensor suite `cookbook` to serve as a comprehensive guideline, highlighting generalizable key design considerations and lessons learned during the development of Boxi. Finally, we demonstrate the versatility of Boxi being used in a variety of applications in real-world scenarios, contributing to robust autonomy. More details and code: this https URL 

**Abstract (ZH)**: 在复杂且未结构化的环境中实现移动机器人 robust autonomy 需要一个能够捕捉多样且互补信息的多模态传感器套件。然而，设计这样一个传感器套件涉及多个关键设计决策，例如传感器选择、组件布置、热管理和电源限制、计算要求、网络连接、同步和标定。尽管这些关键方面的重要性得到广泛认可，但在学术界常常被忽视，或者在大型公司中保留为专有知识。为改善这一情况，我们提出了 Boxi，一个高度集成的传感器负载，能够在野外实现机器人的 robust autonomy。本文讨论了为优化下游任务的算法性能所做出的负载设计决策，特别是针对状态估计和制图。Boxi 配备了多种传感器：两个 LiDAR，10 个 RGB 相机（包括高动态范围、全局快门和滚动快门模型），一个 RGB-D 相机，7 个不同精度的惯性测量单元（IMU），以及一个双天线 RTK GNSS 系统。我们的分析表明，时间同步、标定和传感器模态对状态估计性能至关重要。我们将此类分析置于成本考虑和环境特定挑战的背景下。此外，我们还提供了一个移动传感器套件 `cookbook`，旨在提供全面的指南，并强调在开发 Boxi 过程中获得的一般可推广的关键设计考虑和经验教训。最后，我们展示了 Boxi 在多种实际应用场景中的多功能性，为实现 robust autonomy 做出贡献。更多细节和代码：见此链接。 

---
# Instrumentation for Better Demonstrations: A Case Study 

**Title (ZH)**: 更好的演示所需的仪器：一个案例研究 

**Authors**: Remko Proesmans, Thomas Lips, Francis wyffels  

**Link**: [PDF](https://arxiv.org/pdf/2504.18481)  

**Abstract**: Learning from demonstrations is a powerful paradigm for robot manipulation, but its effectiveness hinges on both the quantity and quality of the collected data. In this work, we present a case study of how instrumentation, i.e. integration of sensors, can improve the quality of demonstrations and automate data collection. We instrument a squeeze bottle with a pressure sensor to learn a liquid dispensing task, enabling automated data collection via a PI controller. Transformer-based policies trained on automated demonstrations outperform those trained on human data in 78% of cases. Our findings indicate that instrumentation not only facilitates scalable data collection but also leads to better-performing policies, highlighting its potential in the pursuit of generalist robotic agents. 

**Abstract (ZH)**: 基于演示学习是机器人操作的一种强大范式，但其效果取决于收集数据的数量和质量。在本研究中，我们通过传感器整合即传感器集成，探讨了如何提高演示质量并自动化数据收集。我们将压力传感器集成到挤压瓶中，以学习液体分配任务，并通过PI控制器实现自动化数据收集。基于变压器的策略在自动采集的演示数据上训练的效果，在78%的情况下优于在人类数据上训练的效果。我们的研究结果表明，传感器整合不仅促进了可扩展的数据收集，还导致了性能更好的策略，突显了其在追求通用机器人代理方面的潜力。 

---
# Action Flow Matching for Continual Robot Learning 

**Title (ZH)**: 连续机器人学习中的动作流程匹配 

**Authors**: Alejandro Murillo-Gonzalez, Lantao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.18471)  

**Abstract**: Continual learning in robotics seeks systems that can constantly adapt to changing environments and tasks, mirroring human adaptability. A key challenge is refining dynamics models, essential for planning and control, while addressing issues such as safe adaptation, catastrophic forgetting, outlier management, data efficiency, and balancing exploration with exploitation -- all within task and onboard resource constraints. Towards this goal, we introduce a generative framework leveraging flow matching for online robot dynamics model alignment. Rather than executing actions based on a misaligned model, our approach refines planned actions to better match with those the robot would take if its model was well aligned. We find that by transforming the actions themselves rather than exploring with a misaligned model -- as is traditionally done -- the robot collects informative data more efficiently, thereby accelerating learning. Moreover, we validate that the method can handle an evolving and possibly imperfect model while reducing, if desired, the dependency on replay buffers or legacy model snapshots. We validate our approach using two platforms: an unmanned ground vehicle and a quadrotor. The results highlight the method's adaptability and efficiency, with a record 34.2\% higher task success rate, demonstrating its potential towards enabling continual robot learning. Code: this https URL. 

**Abstract (ZH)**: 机器人领域的持续学习旨在寻求能够在不断变化的环境和任务中不断适应的系统，模仿人类的适应能力。一个关键挑战是如何在确保规划和控制的同时精炼动力学模型，并解决安全适应、灾难性遗忘、离群值管理、数据效率以及在任务和机载资源约束下平衡探索与利用的问题。为此，我们提出了一种利用流匹配的生成框架，进行在线机器人动力学模型对齐。我们的方法通过改进计划动作以更好地匹配机器人实际应采取的动作，而非基于错位模型执行动作，从而更高效地收集信息性数据，加速学习过程。此外，我们验证了该方法能够处理动态变化且可能存在缺陷的模型，并在必要时减少对回放缓冲区或遗留模型快照的依赖。我们使用两个平台——无人地面车辆和四旋翼无人机——验证了该方法，并展示了其适应性和效率，任务成功率提高了34.2%，展示了其在持续机器人学习方面的潜力。代码：https://this-url.com。 

---
# The Autonomous Software Stack of the FRED-003C: The Development That Led to Full-Scale Autonomous Racing 

**Title (ZH)**: FRED-003C自主软件堆栈：通往全方位自主赛车的开发历程 

**Authors**: Zalán Demeter, Levente Puskás, Balázs Kovács, Ádám Matkovics, Martin Nádas, Balázs Tuba, Zsolt Farkas, Ármin Bogár-Németh, Gergely Bári  

**Link**: [PDF](https://arxiv.org/pdf/2504.18439)  

**Abstract**: Scientific development often takes place in the context of research projects carried out by dedicated students during their time at university. In the field of self-driving software research, the Formula Student Driverless competitions are an excellent platform to promote research and attract young engineers. This article presents the software stack developed by BME Formula Racing Team, that formed the foundation of the development that ultimately led us to full-scale autonomous racing. The experience we gained here contributes greatly to our successful participation in the Abu Dhabi Autonomous Racing League. We therefore think it is important to share the system we used, providing a valuable starting point for other ambitious students. We provide a detailed description of the software pipeline we used, including a brief description of the hardware-software architecture. Furthermore, we introduce the methods that we developed for the modules that implement perception; localisation and mapping, planning, and control tasks. 

**Abstract (ZH)**: 科学的发展往往在大学生们在大学期间进行的专业研究项目中得以推进。在自动驾驶软件研究领域，Formula Student Driverless竞赛是推广研究和吸引年轻工程师的优秀平台。本文介绍了布达佩斯科技与经济大学Formula Racing队开发的软件栈，奠定了实现全规模自动驾驶赛车的基础。我们在这里获得的经验极大地促进了我们参与阿布扎比自动驾驶赛车联盟的顺利进行。因此，我们认为分享我们所使用系统的重要性，为其他志存高远的学生提供一个宝贵的研究起点。本文详细描述了我们使用的软件流水线，包括硬件-software架构的简要说明。此外，我们还介绍了为感知、定位与建图、规划和控制模块开发的方法。 

---
# Enhancing System Self-Awareness and Trust of AI: A Case Study in Trajectory Prediction and Planning 

**Title (ZH)**: 增强系统自我意识和对AI的信任：轨迹预测与规划案例研究 

**Authors**: Lars Ullrich, Zurab Mujirishvili, Knut Graichen  

**Link**: [PDF](https://arxiv.org/pdf/2504.18421)  

**Abstract**: In the trajectory planning of automated driving, data-driven statistical artificial intelligence (AI) methods are increasingly established for predicting the emergent behavior of other road users. While these methods achieve exceptional performance in defined datasets, they usually rely on the independent and identically distributed (i.i.d.) assumption and thus tend to be vulnerable to distribution shifts that occur in the real world. In addition, these methods lack explainability due to their black box nature, which poses further challenges in terms of the approval process and social trustworthiness. Therefore, in order to use the capabilities of data-driven statistical AI methods in a reliable and trustworthy manner, the concept of TrustMHE is introduced and investigated in this paper. TrustMHE represents a complementary approach, independent of the underlying AI systems, that combines AI-driven out-of-distribution detection with control-driven moving horizon estimation (MHE) to enable not only detection and monitoring, but also intervention. The effectiveness of the proposed TrustMHE is evaluated and proven in three simulation scenarios. 

**Abstract (ZH)**: 在自动驾驶的轨迹规划中，基于数据驱动的统计人工智能方法逐渐建立起来以预测其他道路用户的行为。尽管这些方法在定义的数据集上表现出色，但通常依赖独立同分布（i.i.d.）假设，因此在现实世界中容易受到分布变化的影响。此外，由于其黑盒性质，这些方法缺乏可解释性，这在审批过程和社交信任方面提出了进一步的挑战。因此，为了可靠且可信地利用数据驱动的统计人工智能方法的能力，本文引介入一种名为TrustMHE的概念，并对其进行研究。TrustMHE代表一种独立于底层AI系统的补充方法，结合了基于AI的异常检测和基于控制的移动窗口估计（MHE），不仅实现检测和监控，还实现了干预。提出的TrustMHE在三个仿真场景中进行了评估并得到验证。 

---
# Optimal Control of Sensor-Induced Illusions on Robotic Agents 

**Title (ZH)**: 传感器诱导错觉下类人机器人的最优控制 

**Authors**: Lorenzo Medici, Steven M. LaValle, Basak Sakcak  

**Link**: [PDF](https://arxiv.org/pdf/2504.18339)  

**Abstract**: This paper presents a novel problem of creating and regulating localization and navigation illusions considering two agents: a receiver and a producer. A receiver is moving on a plane localizing itself using the intensity of signals from three known towers observed at its position. Based on this position estimate, it follows a simple policy to reach its goal. The key idea is that a producer alters the signal intensities to alter the position estimate of the receiver while ensuring it reaches a different destination with the belief that it reached its goal. We provide a precise mathematical formulation of this problem and show that it allows standard techniques from control theory to be applied to generate localization and navigation illusions that result in a desired receiver behavior. 

**Abstract (ZH)**: 本文提出了一种关于考虑两个代理（接收者和生产者）的定位和导航幻象创建与调控的新问题。接收者在平面上通过估计来自三个已知基站的信号强度来进行自我定位，并基于该位置估计遵循简单策略以实现其目标。关键思想是，生产者通过改变信号强度来干扰接收者的定位估计，同时确保接收者在错误的定位下到达不同的目的地。我们对该问题进行了精确的数学建模，并展示了如何应用控制理论中的标准技术来生成定位和导航幻象，从而实现期望的接收者行为。 

---
# Design and Evaluation of a UGV-Based Robotic Platform for Precision Soil Moisture Remote Sensing 

**Title (ZH)**: 基于UGV的精准土壤水分遥感机器人平台的设计与评估 

**Authors**: Ilektra Tsimpidi, Ilias Tevetzidis, Vidya Sumathy, George Nikolakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2504.18284)  

**Abstract**: This extended abstract presents the design and evaluation of AgriOne, an automated unmanned ground vehicle (UGV) platform for high precision sensing of soil moisture in large agricultural fields. The developed robotic system is equipped with a volumetric water content (VWC) sensor mounted on a robotic manipulator and utilizes a surface-aware data collection framework to ensure accurate measurements in heterogeneous terrains. The framework identifies and removes invalid data points where the sensor fails to penetrate the soil, ensuring data reliability. Multiple field experiments were conducted to validate the platform's performance, while the obtained results demonstrate the efficacy of the AgriOne robot in real-time data acquisition, reducing the need for permanent sensors and labor-intensive methods. 

**Abstract (ZH)**: 扩展摘要：AgriOne自动化无人地面车辆平台在大田土壤水分高精度感知的设计与评估 

---
# Depth-Constrained ASV Navigation with Deep RL and Limited Sensing 

**Title (ZH)**: 深度受限的ASV导航与深度强化学习及有限感知 

**Authors**: Amirhossein Zhalehmehrabi, Daniele Meli, Francesco Dal Santo, Francesco Trotti, Alessandro Farinelli  

**Link**: [PDF](https://arxiv.org/pdf/2504.18253)  

**Abstract**: Autonomous Surface Vehicles (ASVs) play a crucial role in maritime operations, yet their navigation in shallow-water environments remains challenging due to dynamic disturbances and depth constraints. Traditional navigation strategies struggle with limited sensor information, making safe and efficient operation difficult. In this paper, we propose a reinforcement learning (RL) framework for ASV navigation under depth constraints, where the vehicle must reach a target while avoiding unsafe areas with only a single depth measurement per timestep from a downward-facing Single Beam Echosounder (SBES). To enhance environmental awareness, we integrate Gaussian Process (GP) regression into the RL framework, enabling the agent to progressively estimate a bathymetric depth map from sparse sonar readings. This approach improves decision-making by providing a richer representation of the environment. Furthermore, we demonstrate effective sim-to-real transfer, ensuring that trained policies generalize well to real-world aquatic conditions. Experimental results validate our method's capability to improve ASV navigation performance while maintaining safety in challenging shallow-water environments. 

**Abstract (ZH)**: 基于深度约束的自主水面车辆导航的强化学习框架 

---
# Implementation Analysis of Collaborative Robot Digital Twins in Physics Engines 

**Title (ZH)**: 物理引擎中协作机器人数字孪生的实现分析 

**Authors**: Christian König, Jan Petershans, Jan Herbst, Matthias Rüb, Dennis Krummacker, Eric Mittag, Hand D. Schooten  

**Link**: [PDF](https://arxiv.org/pdf/2504.18200)  

**Abstract**: This paper presents a Digital Twin (DT) of a 6G communications system testbed that integrates two robotic manipulators with a high-precision optical infrared tracking system in Unreal Engine 5. Practical details of the setup and implementation insights provide valuable guidance for users aiming to replicate such systems, an endeavor that is crucial to advancing DT applications within the scientific community. Key topics discussed include video streaming, integration within the Robot Operating System 2 (ROS 2), and bidirectional communication. The insights provided are intended to support the development and deployment of DTs in robotics and automation research. 

**Abstract (ZH)**: 本文 presents a数字孪生(Digital Twin, DT)实验床，该实验床在Unreal Engine 5中集成了两个机器人 manipulator 和一个高精度光学红外跟踪系统。实际搭建细节和实施 insights 为希望复制此类系统的用户提供了宝贵指导，这对于在科学界推进数字孪生应用至关重要。文中讨论的关键主题包括视频流传输、与Robot Operating System 2 (ROS 2)的集成以及双向通信。提供的见解旨在支持机器人与自动化研究中数字孪生的开发与部署。 

---
# Sampling-Based Grasp and Collision Prediction for Assisted Teleoperation 

**Title (ZH)**: 基于采样的抓取和碰撞预测辅助远程操作 

**Authors**: Simon Manschitz, Berk Gueler, Wei Ma, Dirk Ruiken  

**Link**: [PDF](https://arxiv.org/pdf/2504.18186)  

**Abstract**: Shared autonomy allows for combining the global planning capabilities of a human operator with the strengths of a robot such as repeatability and accurate control. In a real-time teleoperation setting, one possibility for shared autonomy is to let the human operator decide for the rough movement and to let the robot do fine adjustments, e.g., when the view of the operator is occluded. We present a learning-based concept for shared autonomy that aims at supporting the human operator in a real-time teleoperation setting. At every step, our system tracks the target pose set by the human operator as accurately as possible while at the same time satisfying a set of constraints which influence the robot's behavior. An important characteristic is that the constraints can be dynamically activated and deactivated which allows the system to provide task-specific assistance. Since the system must generate robot commands in real-time, solving an optimization problem in every iteration is not feasible. Instead, we sample potential target configurations and use Neural Networks for predicting the constraint costs for each configuration. By evaluating each configuration in parallel, our system is able to select the target configuration which satisfies the constraints and has the minimum distance to the operator's target pose with minimal delay. We evaluate the framework with a pick and place task on a bi-manual setup with two Franka Emika Panda robot arms with Robotiq grippers. 

**Abstract (ZH)**: 基于学习的共享自主概念：支持实时遥操作的人机协作 

---
# RL-Driven Data Generation for Robust Vision-Based Dexterous Grasping 

**Title (ZH)**: 基于RL驱动的数据生成的稳健视觉引导灵巧抓取 

**Authors**: Atsushi Kanehira, Naoki Wake, Kazuhiro Sasabuchi, Jun Takamatsu, Katsushi Ikeuchi  

**Link**: [PDF](https://arxiv.org/pdf/2504.18084)  

**Abstract**: This work presents reinforcement learning (RL)-driven data augmentation to improve the generalization of vision-action (VA) models for dexterous grasping. While real-to-sim-to-real frameworks, where a few real demonstrations seed large-scale simulated data, have proven effective for VA models, applying them to dexterous settings remains challenging: obtaining stable multi-finger contacts is nontrivial across diverse object shapes. To address this, we leverage RL to generate contact-rich grasping data across varied geometries. In line with the real-to-sim-to-real paradigm, the grasp skill is formulated as a parameterized and tunable reference trajectory refined by a residual policy learned via RL. This modular design enables trajectory-level control that is both consistent with real demonstrations and adaptable to diverse object geometries. A vision-conditioned policy trained on simulation-augmented data demonstrates strong generalization to unseen objects, highlighting the potential of our approach to alleviate the data bottleneck in training VA models. 

**Abstract (ZH)**: 基于强化学习的数据增强方法以提高视-动模型在灵巧抓取任务中的泛化能力 

---
# AllTact Fin Ray: A Compliant Robot Gripper with Omni-Directional Tactile Sensing 

**Title (ZH)**: AllTact Fin Ray：一种全向触觉感知的 compliant 机器人抓取器 

**Authors**: Siwei Liang, Yixuan Guan, Jing Xu, Hongyu Qian, Xiangjun Zhang, Dan Wu, Wenbo Ding, Rui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.18064)  

**Abstract**: Tactile sensing plays a crucial role in robot grasping and manipulation by providing essential contact information between the robot and the environment. In this paper, we present AllTact Fin Ray, a novel compliant gripper design with omni-directional and local tactile sensing capabilities. The finger body is unibody-casted using transparent elastic silicone, and a camera positioned at the base of the finger captures the deformation of the whole body and the contact face. Due to the global deformation of the adaptive structure, existing vision-based tactile sensing approaches that assume constant illumination are no longer applicable. To address this, we propose a novel sensing method where the global deformation is first reconstructed from the image using edge features and spatial constraints. Then, detailed contact geometry is computed from the brightness difference against a dynamically retrieved reference image. Extensive experiments validate the effectiveness of our proposed gripper design and sensing method in contact detection, force estimation, object grasping, and precise manipulation. 

**Abstract (ZH)**: 触觉感知在机器人抓取和操作中起着至关重要的作用，通过提供机器人与环境之间的关键接触信息。本文介绍了AllTact Fin Ray，一种集全方位和局部触觉感知能力于一身的新型顺应式 gripper 设计。手指主体通过透明弹性硅胶整体铸造，并在手指基部配置相机以捕捉全身和接触面的变形。由于适应性结构的整体变形，现有的基于视觉的触觉感知方法假设恒定照明不再适用。为了解决这一问题，我们提出了一种新型的感知方法，首先通过边缘特征和空间约束从图像重建整体变形，然后从与动态检索的参考图像对比的亮度差异中计算出详细的接触几何。广泛的实验证明了我们提出的 gripper 设计和感知方法在接触检测、力估计、物体抓取和精确操作中的有效性。 

---
# Opportunistic Collaborative Planning with Large Vision Model Guided Control and Joint Query-Service Optimization 

**Title (ZH)**: 基于大型视觉模型引导控制与联合查询-服务优化的机会性协作规划 

**Authors**: Jiayi Chen, Shuai Wang, Guoliang Li, Wei Xu, Guangxu Zhu, Derrick Wing Kwan Ng, Chengzhong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.18057)  

**Abstract**: Navigating autonomous vehicles in open scenarios is a challenge due to the difficulties in handling unseen objects. Existing solutions either rely on small models that struggle with generalization or large models that are resource-intensive. While collaboration between the two offers a promising solution, the key challenge is deciding when and how to engage the large model. To address this issue, this paper proposes opportunistic collaborative planning (OCP), which seamlessly integrates efficient local models with powerful cloud models through two key innovations. First, we propose large vision model guided model predictive control (LVM-MPC), which leverages the cloud for LVM perception and decision making. The cloud output serves as a global guidance for a local MPC, thereby forming a closed-loop perception-to-control system. Second, to determine the best timing for large model query and service, we propose collaboration timing optimization (CTO), including object detection confidence thresholding (ODCT) and cloud forward simulation (CFS), to decide when to seek cloud assistance and when to offer cloud service. Extensive experiments show that the proposed OCP outperforms existing methods in terms of both navigation time and success rate. 

**Abstract (ZH)**: 在开放场景中导航自主车辆是一项挑战，因为难以处理未见过的对象。现有的解决方案要么依赖于小型模型但泛化能力差，要么依赖于资源密集型的大模型。尽管两者之间的协作提供了有希望的解决方案，但关键挑战是如何决定何时以及如何激活大模型。为解决这个问题，本文提出了机会性协作规划（OCP），通过两种关键创新无缝地将高效的本地模型与强大的云模型相结合。首先，我们提出了大视图模型引导模型预测控制（LVM-MPC），该模型利用云进行LVM感知和决策。云端输出作为局部MPC的全局指导，从而形成一个从感知到控制的闭环系统。其次，为了确定大型模型查询和服务的最佳时机，我们提出了协作时间优化（CTO），包括对象检测置信阈值（ODCT）和云前向仿真（CFS），以决定何时寻求云辅助以及何时提供云服务。广泛的实验结果表明，提出的OCP在导航时间和成功率方面均优于现有方法。 

---
# Range-based 6-DoF Monte Carlo SLAM with Gradient-guided Particle Filter on GPU 

**Title (ZH)**: 基于范围测量的6-DoF蒙特卡洛SLAM与梯度引导粒子滤波在GPU上的实现 

**Authors**: Takumi Nakao, Kenji Koide, Aoki Takanose, Shuji Oishi, Masashi Yokozuka, Hisashi Date  

**Link**: [PDF](https://arxiv.org/pdf/2504.18056)  

**Abstract**: This paper presents range-based 6-DoF Monte Carlo SLAM with a gradient-guided particle update strategy. While non-parametric state estimation methods, such as particle filters, are robust in situations with high ambiguity, they are known to be unsuitable for high-dimensional problems due to the curse of dimensionality. To address this issue, we propose a particle update strategy that improves the sampling efficiency by using the gradient information of the likelihood function to guide particles toward its mode. Additionally, we introduce a keyframe-based map representation that represents the global map as a set of past frames (i.e., keyframes) to mitigate memory consumption. The keyframe poses for each particle are corrected using a simple loop closure method to maintain trajectory consistency. The combination of gradient information and keyframe-based map representation significantly enhances sampling efficiency and reduces memory usage compared to traditional RBPF approaches. To process a large number of particles (e.g., 100,000 particles) in real-time, the proposed framework is designed to fully exploit GPU parallel processing. Experimental results demonstrate that the proposed method exhibits extreme robustness to state ambiguity and can even deal with kidnapping situations, such as when the sensor moves to different floors via an elevator, with minimal heuristics. 

**Abstract (ZH)**: 基于范围的6自由度蒙特卡洛SLAM结合梯度导向粒子更新策略 

---
# Sky-Drive: A Distributed Multi-Agent Simulation Platform for Socially-Aware and Human-AI Collaborative Future Transportation 

**Title (ZH)**: Sky-Drive：一种面向社会意识和人机协同未来交通的分布式多代理模拟平台 

**Authors**: Zilin Huang, Zihao Sheng, Zhengyang Wan, Yansong Qu, Yuhao Luo, Boyue Wang, Pei Li, Yen-Jung Chen, Jiancong Chen, Keke Long, Jiayi Meng, Yue Leng, Sikai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.18010)  

**Abstract**: Recent advances in autonomous system simulation platforms have significantly enhanced the safe and scalable testing of driving policies. However, existing simulators do not yet fully meet the needs of future transportation research, particularly in modeling socially-aware driving agents and enabling effective human-AI collaboration. This paper introduces Sky-Drive, a novel distributed multi-agent simulation platform that addresses these limitations through four key innovations: (a) a distributed architecture for synchronized simulation across multiple terminals; (b) a multi-modal human-in-the-loop framework integrating diverse sensors to collect rich behavioral data; (c) a human-AI collaboration mechanism supporting continuous and adaptive knowledge exchange; and (d) a digital twin (DT) framework for constructing high-fidelity virtual replicas of real-world transportation environments. Sky-Drive supports diverse applications such as autonomous vehicle (AV)-vulnerable road user (VRU) interaction modeling, human-in-the-loop training, socially-aware reinforcement learning, personalized driving policy, and customized scenario generation. Future extensions will incorporate foundation models for context-aware decision support and hardware-in-the-loop (HIL) testing for real-world validation. By bridging scenario generation, data collection, algorithm training, and hardware integration, Sky-Drive has the potential to become a foundational platform for the next generation of socially-aware and human-centered autonomous transportation research. The demo video and code are available at:this https URL 

**Abstract (ZH)**: 近期自主系统仿真平台的发展显著增强了驾驶策略的安全和可扩展测试。然而，现有的仿真器尚未完全满足未来交通研究的需求，尤其是在建模社会意识较强的驾驶代理和促进有效的人工智能协作方面。本文介绍了Sky-Drive，这是一种通过四项关键创新来解决这些限制的新型分布式多代理仿真平台：(a) 多终端同步仿真分布式架构；(b) 多模式人机在环框架，集成多种传感器以收集丰富的行为数据；(c) 支持持续和适应性知识交流的人工智能协作机制；以及(d) 数字孪生（DT）框架，用于构建高度保真的现实世界交通环境的虚拟副本。Sky-Drive 支持诸如自主车辆（AV）与脆弱道路使用者（VRU）互动建模、人机在环训练、社会意识强化学习、个性化驾驶策略和定制场景生成等多种应用。未来扩展将包括上下文意识基础模型以提供决策支持和实物在环（HIL）测试以进行实际验证。通过连接场景生成、数据收集、算法训练和硬件集成，Sky-Drive 有望成为下一代社会意识强且以人为本的自主交通研究的基础平台。演示视频和代码可在以下链接获取：this https URL。 

---
# Fuzzy-RRT for Obstacle Avoidance in a 2-DOF Semi-Autonomous Surgical Robotic Arm 

**Title (ZH)**: 模糊-RRT算法在两自由度半自主手术机器人避障中的应用 

**Authors**: Kaaustaaub Shankar, Wilhelm Louw, Bharadwaj Dogga, Nick Ernest, Tim Arnett, Kelly Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2504.17979)  

**Abstract**: AI-driven semi-autonomous robotic surgery is essential for addressing the medical challenges of long-duration interplanetary missions, where limited crew sizes and communication delays restrict traditional surgical approaches. Current robotic surgery systems require full surgeon control, demanding extensive expertise and limiting feasibility in space. We propose a novel adaptation of the Fuzzy Rapidly-exploring Random Tree algorithm for obstacle avoidance and collaborative control in a two-degree-of-freedom robotic arm modeled on the Miniaturized Robotic-Assisted surgical system. It was found that the Fuzzy Rapidly-exploring Random Tree algorithm resulted in an 743 percent improvement to path search time and 43 percent improvement to path cost. 

**Abstract (ZH)**: 基于AI的半自主机器人外科手术对于应对长期星际任务的医疗挑战至关重要，其中有限的crew数量和通信延迟限制了传统外科手术方法。当前的机器人外科手术系统需要完全的外科医生控制，这要求广泛的专门知识并限制了其在太空中的可行性。我们提出了一种针对微型机器人辅助外科系统两自由度机器人臂的Fuzzy Rapidly-exploring Random Tree算法的新适应方案，用于避障和协作控制。研究发现，Fuzzy Rapidly-exploring Random Tree算法将路径搜索时间提高了743%，路径成本降低了43%。 

---
# Virtual Roads, Smarter Safety: A Digital Twin Framework for Mixed Autonomous Traffic Safety Analysis 

**Title (ZH)**: 虚拟道路，更智慧的安全：一种混合自主交通安全性分析的数字孪生框架 

**Authors**: Hao Zhang, Ximin Yue, Kexin Tian, Sixu Li, Keshu Wu, Zihao Li, Dominique Lord, Yang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.17968)  

**Abstract**: This paper presents a digital-twin platform for active safety analysis in mixed traffic environments. The platform is built using a multi-modal data-enabled traffic environment constructed from drone-based aerial LiDAR, OpenStreetMap, and vehicle sensor data (e.g., GPS and inclinometer readings). High-resolution 3D road geometries are generated through AI-powered semantic segmentation and georeferencing of aerial LiDAR data. To simulate real-world driving scenarios, the platform integrates the CAR Learning to Act (CARLA) simulator, Simulation of Urban MObility (SUMO) traffic model, and NVIDIA PhysX vehicle dynamics engine. CARLA provides detailed micro-level sensor and perception data, while SUMO manages macro-level traffic flow. NVIDIA PhysX enables accurate modeling of vehicle behaviors under diverse conditions, accounting for mass distribution, tire friction, and center of mass. This integrated system supports high-fidelity simulations that capture the complex interactions between autonomous and conventional vehicles. Experimental results demonstrate the platform's ability to reproduce realistic vehicle dynamics and traffic scenarios, enhancing the analysis of active safety measures. Overall, the proposed framework advances traffic safety research by enabling in-depth, physics-informed evaluation of vehicle behavior in dynamic and heterogeneous traffic environments. 

**Abstract (ZH)**: 基于多模态数据的混合交通环境主动安全分析数字孪生平台 

---
# Plug-and-Play Physics-informed Learning using Uncertainty Quantified Port-Hamiltonian Models 

**Title (ZH)**: 使用不确定性量化哈密尔顿模型的即插即用物理约束学习 

**Authors**: Kaiyuan Tan, Peilun Li, Jun Wang, Thomas Beckers  

**Link**: [PDF](https://arxiv.org/pdf/2504.17966)  

**Abstract**: The ability to predict trajectories of surrounding agents and obstacles is a crucial component in many robotic applications. Data-driven approaches are commonly adopted for state prediction in scenarios where the underlying dynamics are unknown. However, the performance, reliability, and uncertainty of data-driven predictors become compromised when encountering out-of-distribution observations relative to the training data. In this paper, we introduce a Plug-and-Play Physics-Informed Machine Learning (PnP-PIML) framework to address this challenge. Our method employs conformal prediction to identify outlier dynamics and, in that case, switches from a nominal predictor to a physics-consistent model, namely distributed Port-Hamiltonian systems (dPHS). We leverage Gaussian processes to model the energy function of the dPHS, enabling not only the learning of system dynamics but also the quantification of predictive uncertainty through its Bayesian nature. In this way, the proposed framework produces reliable physics-informed predictions even for the out-of-distribution scenarios. 

**Abstract (ZH)**: 基于物理信息的插件式机器学习框架（PnP-PIML）：处理分布外观测的可靠轨迹预测 

---
# CIVIL: Causal and Intuitive Visual Imitation Learning 

**Title (ZH)**: 因果直观视觉imitation学习：CIVIL 

**Authors**: Yinlong Dai, Robert Ramirez Sanchez, Ryan Jeronimus, Shahabedin Sagheb, Cara M. Nunez, Heramb Nemlekar, Dylan P. Losey  

**Link**: [PDF](https://arxiv.org/pdf/2504.17959)  

**Abstract**: Today's robots learn new tasks by imitating human examples. However, this standard approach to visual imitation learning is fundamentally limited: the robot observes what the human does, but not why the human chooses those behaviors. Without understanding the features that factor into the human's decisions, robot learners often misinterpret the data and fail to perform the task when the environment changes. We therefore propose a shift in perspective: instead of asking human teachers just to show what actions the robot should take, we also enable humans to indicate task-relevant features using markers and language prompts. Our proposed algorithm, CIVIL, leverages this augmented data to filter the robot's visual observations and extract a feature representation that causally informs human actions. CIVIL then applies these causal features to train a transformer-based policy that emulates human behaviors without being confused by visual distractors. Our simulations, real-world experiments, and user study demonstrate that robots trained with CIVIL can learn from fewer human demonstrations and perform better than state-of-the-art baselines, especially in previously unseen scenarios. See videos at our project website: this https URL 

**Abstract (ZH)**: 今天的手动机器人通过模仿人类示例学习新任务。然而，这种视觉模仿学习的标准方法存在根本性的局限性：机器人只能观察人类做了什么，但无法了解人类为什么会选择这些行为。没有理解影响人类决策的特征，机器人往往误读数据，在环境变化时无法完成任务。因此，我们提出一种新的视角：不再只是要求人类教师展示机器人应采取的动作，我们还允许人类使用标记和语言提示来指出与任务相关的特征。我们提出的算法CIVIL利用这种增强的数据来过滤机器人的视觉观察，并提取一个因果指示人类行为的特征表示。CIVIL随后利用这些因果特征训练基于变换器的策略，使其模仿人类行为而不被视觉干扰所迷惑。我们的模拟、真实世界实验和用户研究显示，使用CIVIL训练的机器人可以从更少的人类示范中学习，并在前所未见的场景中表现优于最先进的基线方法。详见项目网站上的视频：this https URL 

---
# Learning Attentive Neural Processes for Planning with Pushing Actions 

**Title (ZH)**: 学习注意力神经过程以执行推行动作的规划 

**Authors**: Atharv Jain, Seiji Shaw, Nicholas Roy  

**Link**: [PDF](https://arxiv.org/pdf/2504.17924)  

**Abstract**: Our goal is to enable robots to plan sequences of tabletop actions to push a block with unknown physical properties to a desired goal pose on the table. We approach this problem by learning the constituent models of a Partially-Observable Markov Decision Process (POMDP), where the robot can observe the outcome of a push, but the physical properties of the block that govern the dynamics remain unknown. The pushing problem is a difficult POMDP to solve due to the challenge of state estimation. The physical properties have a nonlinear relationship with the outcomes, requiring computationally expensive methods, such as particle filters, to represent beliefs. Leveraging the Attentive Neural Process architecture, we propose to replace the particle filter with a neural network that learns the inference computation over the physical properties given a history of actions. This Neural Process is integrated into planning as the Neural Process Tree with Double Progressive Widening (NPT-DPW). Simulation results indicate that NPT-DPW generates more effective plans faster than traditional particle filter methods, even in complex pushing scenarios. 

**Abstract (ZH)**: 我们的目标是使机器人能够计划一序列桌面操作，以推动一个具有未知物理特性的立方体到达桌子上的目标姿态。我们通过学习部分可观测马尔可夫决策过程（POMDP）的组成部分模型来解决这个问题，在该过程中，机器人可以观察到推动的结果，但控制动力学的物理特性仍然未知。由于状态估计的挑战，推动问题是解决POMDP的一个难题。物理特性与结果之间存在非线性关系，需要使用计算密集型方法，例如粒子滤波器，来表示信念。利用注意力神经过程架构，我们提出用一个神经网络替换粒子滤波器，该神经网络可以在给定动作历史的情况下学习推断计算物理属性的过程。该神经过程与双重逐步扩展的神经过程树（NPT-DPW）结合，用于规划。模拟结果表明，与传统的粒子滤波器方法相比，NPT-DPW即使在复杂推动场景中也能更快生成更有效的规划。 

---
# Beyond Task and Motion Planning: Hierarchical Robot Planning with General-Purpose Policies 

**Title (ZH)**: 超越任务与运动规划：具有通用策略的分层机器人规划 

**Authors**: Benned Hedegaard, Ziyi Yang, Yichen Wei, Ahmed Jaafar, Stefanie Tellex, George Konidaris, Naman Shah  

**Link**: [PDF](https://arxiv.org/pdf/2504.17901)  

**Abstract**: Task and motion planning is a well-established approach for solving long-horizon robot planning problems. However, traditional methods assume that each task-level robot action, or skill, can be reduced to kinematic motion planning. In this work, we address the challenge of planning with both kinematic skills and closed-loop motor controllers that go beyond kinematic considerations. We propose a novel method that integrates these controllers into motion planning using Composable Interaction Primitives (CIPs), enabling the use of diverse, non-composable pre-learned skills in hierarchical robot planning. Toward validating our Task and Skill Planning (TASP) approach, we describe ongoing robot experiments in real-world scenarios designed to demonstrate how CIPs can allow a mobile manipulator robot to effectively combine motion planning with general-purpose skills to accomplish complex tasks. 

**Abstract (ZH)**: 基于任务和运动规划的技能计划 

---
# Terrain-Aware Kinodynamic Planning with Efficiently Adaptive State Lattices for Mobile Robot Navigation in Off-Road Environments 

**Title (ZH)**: 基于地形感知的自适应状态格网高效适配 kino-dynamic 规划在非结构化环境中的移动机器人导航 

**Authors**: Eric R. Damm, Jason M. Gregory, Eli S. Lancaster, Felix A. Sanchez, Daniel M. Sahu, Thomas M. Howard  

**Link**: [PDF](https://arxiv.org/pdf/2504.17889)  

**Abstract**: To safely traverse non-flat terrain, robots must account for the influence of terrain shape in their planned motions. Terrain-aware motion planners use an estimate of the vehicle roll and pitch as a function of pose, vehicle suspension, and ground elevation map to weigh the cost of edges in the search space. Encoding such information in a traditional two-dimensional cost map is limiting because it is unable to capture the influence of orientation on the roll and pitch estimates from sloped terrain. The research presented herein addresses this problem by encoding kinodynamic information in the edges of a recombinant motion planning search space based on the Efficiently Adaptive State Lattice (EASL). This approach, which we describe as a Kinodynamic Efficiently Adaptive State Lattice (KEASL), differs from the prior representation in two ways. First, this method uses a novel encoding of velocity and acceleration constraints and vehicle direction at expanded nodes in the motion planning graph. Second, this approach describes additional steps for evaluating the roll, pitch, constraints, and velocities associated with poses along each edge during search in a manner that still enables the graph to remain recombinant. Velocities are computed using an iterative bidirectional method using Eulerian integration that more accurately estimates the duration of edges that are subject to terrain-dependent velocity limits. Real-world experiments on a Clearpath Robotics Warthog Unmanned Ground Vehicle were performed in a non-flat, unstructured environment. Results from 2093 planning queries from these experiments showed that KEASL provided a more efficient route than EASL in 83.72% of cases when EASL plans were adjusted to satisfy terrain-dependent velocity constraints. An analysis of relative runtimes and differences between planned routes is additionally presented. 

**Abstract (ZH)**: 面向非平坦地形的机器人安全穿越路径规划中需考虑地形形状的影响。基于Efficiently Adaptive State Lattice (EASL)的Kinodynamic Efficiently Adaptive State Lattice (KEASL)在重组合运动规划搜索空间的边缘中编码动力学信息以解决此问题。 

---
# Autonomous Navigation Of Quadrupeds Using Coverage Path Planning 

**Title (ZH)**: 四足机器人的覆盖路径规划自主导航 

**Authors**: Alexander James Becoy, Kseniia Khomenko, Luka Peternel, Raj Thilak Rajan  

**Link**: [PDF](https://arxiv.org/pdf/2504.17880)  

**Abstract**: This paper proposes a novel method of coverage path planning for the purpose of scanning an unstructured environment autonomously. The method uses the morphological skeleton of the prior 2D navigation map via SLAM to generate a sequence of points of interest (POIs). This sequence is then ordered to create an optimal path given the robot's current position. To control the high-level operation, a finite state machine is used to switch between two modes: navigating towards a POI using Nav2, and scanning the local surrounding. We validate the method in a leveled indoor obstacle-free non-convex environment on time efficiency and reachability over five trials. The map reader and the path planner can quickly process maps of width and height ranging between [196,225] pixels and [185,231] pixels in 2.52 ms/pixel and 1.7 ms/pixel, respectively, where their computation time increases with 22.0 ns/pixel and 8.17 $\mu$s/pixel, respectively. The robot managed to reach 86.5\% of all waypoints over all five runs. The proposed method suffers from drift occurring in the 2D navigation map. 

**Abstract (ZH)**: 本文提出了一种新型的覆盖率路径规划方法，用于自主扫描未结构化环境。该方法通过SLAM之前的2D导航地图的形态骨架生成一系列兴趣点（POIs），并根据机器人当前位置对这些点进行排序以生成最优路径。为了控制高级操作，使用有限状态机在两种模式之间切换：使用Nav2朝向POI导航，以及扫描局部周围环境。我们在一个平坦的室内无障碍非凸环境中进行了五次试验，验证了该方法在时间效率和可达性方面的表现。地图读取器和路径规划器分别以每像素2.52毫秒和1.7毫秒的速度处理宽度和高度在[196,225]像素和[185,231]像素之间的地图，其计算时间分别增加22.0纳秒/像素和8.17微秒/像素。机器人在五次运行中成功到达了所有航点的86.5%。提出的路径规划方法受到2D导航地图中漂移的影响。 

---
# Flow Matching Ergodic Coverage 

**Title (ZH)**: 流匹配遍历覆盖 

**Authors**: Max Muchen Sun, Allison Pinosky, Todd Murphey  

**Link**: [PDF](https://arxiv.org/pdf/2504.17872)  

**Abstract**: Ergodic coverage effectively generates exploratory behaviors for embodied agents by aligning the spatial distribution of the agent's trajectory with a target distribution, where the difference between these two distributions is measured by the ergodic metric. However, existing ergodic coverage methods are constrained by the limited set of ergodic metrics available for control synthesis, fundamentally limiting their performance. In this work, we propose an alternative approach to ergodic coverage based on flow matching, a technique widely used in generative inference for efficient and scalable sampling. We formally derive the flow matching problem for ergodic coverage and show that it is equivalent to a linear quadratic regulator problem with a closed-form solution. Our formulation enables alternative ergodic metrics from generative inference that overcome the limitations of existing ones. These metrics were previously infeasible for control synthesis but can now be supported with no computational overhead. Specifically, flow matching with the Stein variational gradient flow enables control synthesis directly over the score function of the target distribution, improving robustness to the unnormalized distributions; on the other hand, flow matching with the Sinkhorn divergence flow enables an optimal transport-based ergodic metric, improving coverage performance on non-smooth distributions with irregular supports. We validate the improved performance and competitive computational efficiency of our method through comprehensive numerical benchmarks and across different nonlinear dynamics. We further demonstrate the practicality of our method through a series of drawing and erasing tasks on a Franka robot. 

**Abstract (ZH)**: 基于流匹配的遍历覆盖有效地通过将执行体代理的轨迹空间分布与目标分布对齐来生成探索行为，其中这两种分布之间的差异通过遍历度量进行度量。然而，现有的遍历覆盖方法受到可用的遍历度量集的限制，从根本上限制了其性能。在本工作中，我们提出了一种基于流匹配的替代遍历覆盖方法，这是一种在生成性推断中广泛用于高效可扩展采样的技术。我们正式推导了遍历覆盖下的流匹配问题，并证明它等价于一个具有闭式解的线性二次调节器问题。我们的表述使得可以从生成性推断中获得替代的遍历度量，从而克服了现有方法的限制。这些度量以前因控制合成的限制而不可行，但现在可以通过无需计算开销的方式支持。具体来说，基于Stein变分梯度流的流匹配直接在目标分布的评分函数上进行控制合成，增强了对未规范化分布的鲁棒性；另一方面，基于Sinkhorn散度流的流匹配提供了基于最优传输的遍历度量，提高了具有不规则支持的非光滑分布的覆盖性能。我们通过全面的数值基准和不同非线性动力学验证了我们方法的改进性能和具有竞争力的计算效率。我们进一步通过Franka机器人上的绘图和擦除任务展示了我们方法的实用性。 

---
# Set Phasers to Stun: Beaming Power and Control to Mobile Robots with Laser Light 

**Title (ZH)**: 设置phasers至眩晕：通过激光光束为移动机器人传输能量与控制 

**Authors**: Charles J. Carver, Hadleigh Schwartz, Toma Itagaki, Zachary Englhardt, Kechen Liu, Megan Graciela Nauli Manik, Chun-Cheng Chang, Vikram Iyer, Brian Plancher, Xia Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.17865)  

**Abstract**: We present Phaser, a flexible system that directs narrow-beam laser light to moving robots for concurrent wireless power delivery and communication. We design a semi-automatic calibration procedure to enable fusion of stereo-vision-based 3D robot tracking with high-power beam steering, and a low-power optical communication scheme that reuses the laser light as a data channel. We fabricate a Phaser prototype using off-the-shelf hardware and evaluate its performance with battery-free autonomous robots. Phaser delivers optical power densities of over 110 mW/cm$^2$ and error-free data to mobile robots at multi-meter ranges, with on-board decoding drawing 0.3 mA (97\% less current than Bluetooth Low Energy). We demonstrate Phaser fully powering gram-scale battery-free robots to nearly 2x higher speeds than prior work while simultaneously controlling them to navigate around obstacles and along paths. Code, an open-source design guide, and a demonstration video of Phaser is available at this https URL. 

**Abstract (ZH)**: Phaser：一种针对移动机器人实现 concurrent 无线功率传输和通信的灵活系统 

---
# Generalization Capability for Imitation Learning 

**Title (ZH)**: 模仿学习的泛化能力 

**Authors**: Yixiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18538)  

**Abstract**: Imitation learning holds the promise of equipping robots with versatile skills by learning from expert demonstrations. However, policies trained on finite datasets often struggle to generalize beyond the training distribution. In this work, we present a unified perspective on the generalization capability of imitation learning, grounded in both information theorey and data distribution property. We first show that the generalization gap can be upper bounded by (i) the conditional information bottleneck on intermediate representations and (ii) the mutual information between the model parameters and the training dataset. This characterization provides theoretical guidance for designing effective training strategies in imitation learning, particularly in determining whether to freeze, fine-tune, or train large pretrained encoders (e.g., vision-language models or vision foundation models) from scratch to achieve better generalization. Furthermore, we demonstrate that high conditional entropy from input to output induces a flatter likelihood landscape, thereby reducing the upper bound on the generalization gap. In addition, it shortens the stochastic gradient descent (SGD) escape time from sharp local minima, which may increase the likelihood of reaching global optima under fixed optimization budgets. These insights explain why imitation learning often exhibits limited generalization and underscore the importance of not only scaling the diversity of input data but also enriching the variability of output labels conditioned on the same input. 

**Abstract (ZH)**: 模仿学习通过从专家示范中学习赋予机器人多样的技能充满了可能性。然而，基于有限数据集训练的策略往往难以泛化到训练分布之外。在这项工作中，我们从信息理论和数据分布特性出发，提供了一种统一的模仿学习泛化能力视角。我们首先表明，泛化差距可以上界表示为（i）中间表示的条件信息瓶颈和（ii）模型参数与训练数据集之间的互信息。这种表征为设计有效的模仿学习训练策略提供了理论指导，特别是在决定是否冻结、微调或从头训练大型预训练编码器（例如，视觉-语言模型或视觉基础模型）以实现更好的泛化方面。此外，我们证明从输入到输出的高条件熵会导致更平坦的似然景观，从而减少泛化差距的上界。另外，这缩短了从尖锐局部极小值的随机梯度下降（SGD）逃逸时间，从而在固定优化预算下增加达到全局最优的可能性。这些见解解释了为什么模仿学习往往表现出有限的泛化能力，并强调了不仅扩大输入数据多样性，还要在相同输入下丰富输出标签多样性的的重要性。 

---
# E-VLC: A Real-World Dataset for Event-based Visible Light Communication And Localization 

**Title (ZH)**: E-VLC：事件驱动型可见光通信及定位的现实世界数据集 

**Authors**: Shintaro Shiba, Quan Kong, Norimasa Kobori  

**Link**: [PDF](https://arxiv.org/pdf/2504.18521)  

**Abstract**: Optical communication using modulated LEDs (e.g., visible light communication) is an emerging application for event cameras, thanks to their high spatio-temporal resolutions. Event cameras can be used simply to decode the LED signals and also to localize the camera relative to the LED marker positions. However, there is no public dataset to benchmark the decoding and localization in various real-world settings. We present, to the best of our knowledge, the first public dataset that consists of an event camera, a frame camera, and ground-truth poses that are precisely synchronized with hardware triggers. It provides various camera motions with various sensitivities in different scene brightness settings, both indoor and outdoor. Furthermore, we propose a novel method of localization that leverages the Contrast Maximization framework for motion estimation and compensation. The detailed analysis and experimental results demonstrate the advantages of LED-based localization with events over the conventional AR-marker--based one with frames, as well as the efficacy of the proposed method in localization. We hope that the proposed dataset serves as a future benchmark for both motion-related classical computer vision tasks and LED marker decoding tasks simultaneously, paving the way to broadening applications of event cameras on mobile devices. this https URL 

**Abstract (ZH)**: 基于调制LED的光通信事件相机应用：事件相机用于解码LED信号和定位LED标记位置，但由于缺乏公共数据集以在各种实际场景中 benchmark 解码和定位性能，我们首次推出了一个包含事件相机、帧相机和精确同步的地面真相姿态的数据集。该数据集提供了不同场景亮度设置下、室内和室外的各种相机运动和不同的灵敏度。此外，我们提出了一种新的基于对比最大化框架的定位方法，用于运动估计和补偿。详细的分析和实验结果表明，基于LED的事件定位方法在运动估计和补偿方面比基于帧的传统AR标记方法更优，同时展示了所提出方法在定位任务中的有效性。我们希望该数据集能够成为未来经典计算机视觉中的运动相关任务和LED标记解码任务的基准，为事件相机在移动设备中的广泛应用铺平道路。这个 https://doi.org/10.5281/zenodo.6543219 

---
# A Taylor Series Approach to Correction of Input Errors in Gaussian Process Regression 

**Title (ZH)**: 高斯过程回归中输入误差矫正的泰勒级数方法 

**Authors**: Muzaffar Qureshi, Tochukwu Elijah Ogri, Zachary I. Bell, Wanjiku A. Makumi, Rushikesh Kamalapurkar  

**Link**: [PDF](https://arxiv.org/pdf/2504.18463)  

**Abstract**: Gaussian Processes (GPs) are widely recognized as powerful non-parametric models for regression and classification. Traditional GP frameworks predominantly operate under the assumption that the inputs are either accurately known or subject to zero-mean noise. However, several real-world applications such as mobile sensors have imperfect localization, leading to inputs with biased errors. These biases can typically be estimated through measurements collected over time using, for example, Kalman filters. To avoid recomputation of the entire GP model when better estimates of the inputs used in the training data become available, we introduce a technique for updating a trained GP model to incorporate updated estimates of the inputs. By leveraging the differentiability of the mean and covariance functions derived from the squared exponential kernel, a second-order correction algorithm is developed to update the trained GP models. Precomputed Jacobians and Hessians of kernels enable real-time refinement of the mean and covariance predictions. The efficacy of the developed approach is demonstrated using two simulation studies, with error analyses revealing improvements in both predictive accuracy and uncertainty quantification. 

**Abstract (ZH)**: 高斯过程（GPs）被认为是回归和分类的强非参数模型。传统的GP框架主要假设输入要么准确已知，要么受到零均值噪声的影响。然而，如移动传感器等实际应用场景可能具有不完美的定位，导致输入带有有偏误差。这些偏差可以通过时间序列测量，例如使用卡尔曼滤波器进行估计。为了避免在获得更好的输入估计值时重新计算整个GP模型，我们提出了一种更新训练好的GP模型以融入更新的输入估计值的技术。通过利用来自指数平方核的均值和协方差函数的可微性，开发了一种二次校正算法来更新训练好的GP模型。预计算的核的雅可比行列式和海森矩阵使对均值和协方差预测的实时优化成为可能。通过两个模拟研究验证了该方法的有效性，误差分析显示预测准确性和不确定性量化均有提升。 

---
# A Multimodal Hybrid Late-Cascade Fusion Network for Enhanced 3D Object Detection 

**Title (ZH)**: 增强3D对象检测的多模态混合滞后级融合网络 

**Authors**: Carlo Sgaravatti, Roberto Basla, Riccardo Pieroni, Matteo Corno, Sergio M. Savaresi, Luca Magri, Giacomo Boracchi  

**Link**: [PDF](https://arxiv.org/pdf/2504.18419)  

**Abstract**: We present a new way to detect 3D objects from multimodal inputs, leveraging both LiDAR and RGB cameras in a hybrid late-cascade scheme, that combines an RGB detection network and a 3D LiDAR detector. We exploit late fusion principles to reduce LiDAR False Positives, matching LiDAR detections with RGB ones by projecting the LiDAR bounding boxes on the image. We rely on cascade fusion principles to recover LiDAR False Negatives leveraging epipolar constraints and frustums generated by RGB detections of separate views. Our solution can be plugged on top of any underlying single-modal detectors, enabling a flexible training process that can take advantage of pre-trained LiDAR and RGB detectors, or train the two branches separately. We evaluate our results on the KITTI object detection benchmark, showing significant performance improvements, especially for the detection of Pedestrians and Cyclists. 

**Abstract (ZH)**: 我们提出了一种新的多模态输入三维物体检测方法，结合了LiDAR和RGB相机在混合晚融方案中，该方案结合了RGB检测网络和3D LiDAR检测器。我们利用晚融原理减少LiDAR的假阳性结果，通过将LiDAR边框投影到图像上，对齐LiDAR检测结果和RGB检测结果。我们依赖级联融合理论，利用视图间RGB检测生成的束状图和极线约束来恢复LiDAR的假阴性结果。该解决方案可以置于任何底层单模态检测器之上，提供一个灵活的训练过程，可以利用预训练的LiDAR和RGB检测器，或者分别训练两个分支。我们在KITTI物体检测基准上评估了我们的方法，显示出显著的性能提升，特别是在行人和骑自行车者检测方面。 

---
# Interpretable Affordance Detection on 3D Point Clouds with Probabilistic Prototypes 

**Title (ZH)**: 基于概率原型的可解释的3D点云功能检测 

**Authors**: Maximilian Xiling Li, Korbinian Rudolf, Nils Blank, Rudolf Lioutikov  

**Link**: [PDF](https://arxiv.org/pdf/2504.18355)  

**Abstract**: Robotic agents need to understand how to interact with objects in their environment, both autonomously and during human-robot interactions. Affordance detection on 3D point clouds, which identifies object regions that allow specific interactions, has traditionally relied on deep learning models like PointNet++, DGCNN, or PointTransformerV3. However, these models operate as black boxes, offering no insight into their decision-making processes. Prototypical Learning methods, such as ProtoPNet, provide an interpretable alternative to black-box models by employing a "this looks like that" case-based reasoning approach. However, they have been primarily applied to image-based tasks. In this work, we apply prototypical learning to models for affordance detection on 3D point clouds. Experiments on the 3D-AffordanceNet benchmark dataset show that prototypical models achieve competitive performance with state-of-the-art black-box models and offer inherent interpretability. This makes prototypical models a promising candidate for human-robot interaction scenarios that require increased trust and safety. 

**Abstract (ZH)**: 机器人代理需要理解如何自主地与环境中的物体交互，以及在人机交互过程中与物体交互。物体区域的可用性检测（affordance detection）基于3D点云，传统上依赖于PointNet++、DGCNN或PointTransformerV3等深度学习模型。然而，这些模型作为黑盒操作，不提供其决策过程的见解。原型学习方法，如ProtoPNet，通过采用“看起来像”的案例推理方法提供了对黑盒模型的可解释替代方案。然而，它们主要应用于基于图像的任务。在本工作中，我们将原型学习应用于3D点云上的可用性检测模型。实验表明，原型模型在3D-AffordanceNet基准数据集上达到了与最先进的黑盒模型相当的性能，并具有内在的可解释性。这使得原型模型成为需要增加信任和安全的人机交互场景的有前景候选者。 

---
# BiasBench: A reproducible benchmark for tuning the biases of event cameras 

**Title (ZH)**: BiasBench: 一种可再现的事件摄像机偏置调整基准测试 

**Authors**: Andreas Ziegler, David Joseph, Thomas Gossard, Emil Moldovan, Andreas Zell  

**Link**: [PDF](https://arxiv.org/pdf/2504.18235)  

**Abstract**: Event-based cameras are bio-inspired sensors that detect light changes asynchronously for each pixel. They are increasingly used in fields like computer vision and robotics because of several advantages over traditional frame-based cameras, such as high temporal resolution, low latency, and high dynamic range. As with any camera, the output's quality depends on how well the camera's settings, called biases for event-based cameras, are configured. While frame-based cameras have advanced automatic configuration algorithms, there are very few such tools for tuning these biases. A systematic testing framework would require observing the same scene with different biases, which is tricky since event cameras only generate events when there is movement. Event simulators exist, but since biases heavily depend on the electrical circuit and the pixel design, available simulators are not well suited for bias tuning. To allow reproducibility, we present BiasBench, a novel event dataset containing multiple scenes with settings sampled in a grid-like pattern. We present three different scenes, each with a quality metric of the downstream application. Additionally, we present a novel, RL-based method to facilitate online bias adjustments. 

**Abstract (ZH)**: 基于事件的相机是受生物学启发的传感器，能够异步检测每个像素的光照变化。由于与传统的基于帧的相机相比具有高时间分辨率、低延迟和高动态范围等优势，它们在计算机视觉和机器人技术等领域越来越受到重视。类似于任何相机，输出的质量取决于相机设置（对于基于事件的相机称为偏置）的配置情况。尽管基于帧的相机已经拥有先进的自动配置算法，调整这些偏置的工具却相对较少。系统性的测试框架需要使用不同的偏置观测相同的场景，但由于事件相机仅在检测到运动时才生成事件，因此这颇具挑战性。现有的事件模拟器存在，但由于偏置高度依赖于电子电路和像素设计，这些模拟器不适用于偏置调整。为了保证可重复性，我们提出了BiasBench，这是一个新颖的基于事件的数据集，包含多个采用网格采样方法设置的场景。我们还呈现了三种不同的场景，并提供了每个场景下游应用的质量度量。此外，我们还介绍了一种新颖的基于强化学习的方法，以促进在线偏置调整。 

---
# Offline Learning of Controllable Diverse Behaviors 

**Title (ZH)**: 离线学习可控多样行为 

**Authors**: Mathieu Petitbois, Rémy Portelas, Sylvain Lamprier, Ludovic Denoyer  

**Link**: [PDF](https://arxiv.org/pdf/2504.18160)  

**Abstract**: Imitation Learning (IL) techniques aim to replicate human behaviors in specific tasks. While IL has gained prominence due to its effectiveness and efficiency, traditional methods often focus on datasets collected from experts to produce a single efficient policy. Recently, extensions have been proposed to handle datasets of diverse behaviors by mainly focusing on learning transition-level diverse policies or on performing entropy maximization at the trajectory level. While these methods may lead to diverse behaviors, they may not be sufficient to reproduce the actual diversity of demonstrations or to allow controlled trajectory generation. To overcome these drawbacks, we propose a different method based on two key features: a) Temporal Consistency that ensures consistent behaviors across entire episodes and not just at the transition level as well as b) Controllability obtained by constructing a latent space of behaviors that allows users to selectively activate specific behaviors based on their requirements. We compare our approach to state-of-the-art methods over a diverse set of tasks and environments. Project page: this https URL 

**Abstract (ZH)**: 基于时间和可控性的行为模仿学习方法：处理多样化行为的数据集 

---
# A Large Vision-Language Model based Environment Perception System for Visually Impaired People 

**Title (ZH)**: 基于大型视觉-语言模型的视觉障碍人群环境感知系统 

**Authors**: Zezhou Chen, Zhaoxiang Liu, Kai Wang, Kohou Wang, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2504.18027)  

**Abstract**: It is a challenging task for visually impaired people to perceive their surrounding environment due to the complexity of the natural scenes. Their personal and social activities are thus highly limited. This paper introduces a Large Vision-Language Model(LVLM) based environment perception system which helps them to better understand the surrounding environment, by capturing the current scene they face with a wearable device, and then letting them retrieve the analysis results through the device. The visually impaired people could acquire a global description of the scene by long pressing the screen to activate the LVLM output, retrieve the categories of the objects in the scene resulting from a segmentation model by tapping or swiping the screen, and get a detailed description of the objects they are interested in by double-tapping the screen. To help visually impaired people more accurately perceive the world, this paper proposes incorporating the segmentation result of the RGB image as external knowledge into the input of LVLM to reduce the LVLM's hallucination. Technical experiments on POPE, MME and LLaVA-QA90 show that the system could provide a more accurate description of the scene compared to Qwen-VL-Chat, exploratory experiments show that the system helps visually impaired people to perceive the surrounding environment effectively. 

**Abstract (ZH)**: 由于自然场景的复杂性，盲人识别周围环境是一项具有挑战的任务，极大地限制了他们的个人和社会活动。本文介绍了一种基于大型视觉-语言模型（LVLM）的环境感知系统，通过穿戴设备捕捉当前场景，并让使用者通过设备检索分析结果，从而帮助他们更好地理解周围环境。盲人可以通过长按屏幕激活LVLM输出以获取场景的全局描述，通过轻触或滑动屏幕检索由分割模型产生的场景中物体的类别，并通过双击屏幕获取对感兴趣物体的详细描述。为了帮助盲人更准确地感知世界，本文提出将RGB图像的分割结果作为外部知识整合到LVLM的输入中以减少LVLM的幻觉。在POPE、MME和LLaVA-QA90上的技术实验表明，该系统能够比Qwen-VL-Chat提供更准确的场景描述。探索性实验表明，该系统能有效地帮助盲人感知周围环境。 

---
# RSRNav: Reasoning Spatial Relationship for Image-Goal Navigation 

**Title (ZH)**: RSRNav: 基于空间关系推理的图像目标导航 

**Authors**: Zheng Qin, Le Wang, Yabing Wang, Sanping Zhou, Gang Hua, Wei Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.17991)  

**Abstract**: Recent image-goal navigation (ImageNav) methods learn a perception-action policy by separately capturing semantic features of the goal and egocentric images, then passing them to a policy network. However, challenges remain: (1) Semantic features often fail to provide accurate directional information, leading to superfluous actions, and (2) performance drops significantly when viewpoint inconsistencies arise between training and application. To address these challenges, we propose RSRNav, a simple yet effective method that reasons spatial relationships between the goal and current observations as navigation guidance. Specifically, we model the spatial relationship by constructing correlations between the goal and current observations, which are then passed to the policy network for action prediction. These correlations are progressively refined using fine-grained cross-correlation and direction-aware correlation for more precise navigation. Extensive evaluation of RSRNav on three benchmark datasets demonstrates superior navigation performance, particularly in the "user-matched goal" setting, highlighting its potential for real-world applications. 

**Abstract (ZH)**: Recent Image-Goal Navigation Methods Learn a Perception-Action Policy by Reasoning Spatial Relationships 

---
# Quaternion Domain Super MDS for 3D Localization 

**Title (ZH)**: 三维定位领域的四元数域超MDS方法 

**Authors**: Keigo Masuoka, Takumi Takahashi, Giuseppe Thadeu Freitas de Abreu, Hideki Ochiai  

**Link**: [PDF](https://arxiv.org/pdf/2504.17890)  

**Abstract**: We propose a novel low-complexity three-dimensional (3D) localization algorithm for wireless sensor networks, termed quaternion-domain super multidimensional scaling (QD-SMDS). This algorithm reformulates the conventional SMDS, which was originally developed in the real domain, into the quaternion domain. By representing 3D coordinates as quaternions, the method enables the construction of a rank-1 Gram edge kernel (GEK) matrix that integrates both relative distance and angular (phase) information between nodes, maximizing the noise reduction effect achieved through low-rank truncation via singular value decomposition (SVD). The simulation results indicate that the proposed method demonstrates a notable enhancement in localization accuracy relative to the conventional SMDS algorithm, particularly in scenarios characterized by substantial measurement errors. 

**Abstract (ZH)**: Quaternion域超多维标度量化三维无线传感器网络定位算法（QD-SMDS） 

---
# High-Performance Reinforcement Learning on Spot: Optimizing Simulation Parameters with Distributional Measures 

**Title (ZH)**: Spot上高性能强化学习：基于分布性度量优化模拟参数 

**Authors**: A. J Miller, Fangzhou Yu, Michael Brauckmann, Farbod Farshidian  

**Link**: [PDF](https://arxiv.org/pdf/2504.17857)  

**Abstract**: This work presents an overview of the technical details behind a high performance reinforcement learning policy deployment with the Spot RL Researcher Development Kit for low level motor access on Boston Dynamics Spot. This represents the first public demonstration of an end to end end reinforcement learning policy deployed on Spot hardware with training code publicly available through Nvidia IsaacLab and deployment code available through Boston Dynamics. We utilize Wasserstein Distance and Maximum Mean Discrepancy to quantify the distributional dissimilarity of data collected on hardware and in simulation to measure our sim2real gap. We use these measures as a scoring function for the Covariance Matrix Adaptation Evolution Strategy to optimize simulated parameters that are unknown or difficult to measure from Spot. Our procedure for modeling and training produces high quality reinforcement learning policies capable of multiple gaits, including a flight phase. We deploy policies capable of over 5.2ms locomotion, more than triple Spots default controller maximum speed, robustness to slippery surfaces, disturbance rejection, and overall agility previously unseen on Spot. We detail our method and release our code to support future work on Spot with the low level API. 

**Abstract (ZH)**: 基于Boston Dynamics Spot的高性能强化学习策略部署技术细节综述：低层次电机访问下的Spot RL Researcher Development Kit首次公开展示及Sim2Real差距量化 

---
# CaRL: Learning Scalable Planning Policies with Simple Rewards 

**Title (ZH)**: CaRL：学习可扩展的规划策略的简单奖励方法 

**Authors**: Bernhard Jaeger, Daniel Dauner, Jens Beißwenger, Simon Gerstenecker, Kashyap Chitta, Andreas Geiger  

**Link**: [PDF](https://arxiv.org/pdf/2504.17838)  

**Abstract**: We investigate reinforcement learning (RL) for privileged planning in autonomous driving. State-of-the-art approaches for this task are rule-based, but these methods do not scale to the long tail. RL, on the other hand, is scalable and does not suffer from compounding errors like imitation learning. Contemporary RL approaches for driving use complex shaped rewards that sum multiple individual rewards, \eg~progress, position, or orientation rewards. We show that PPO fails to optimize a popular version of these rewards when the mini-batch size is increased, which limits the scalability of these approaches. Instead, we propose a new reward design based primarily on optimizing a single intuitive reward term: route completion. Infractions are penalized by terminating the episode or multiplicatively reducing route completion. We find that PPO scales well with higher mini-batch sizes when trained with our simple reward, even improving performance. Training with large mini-batch sizes enables efficient scaling via distributed data parallelism. We scale PPO to 300M samples in CARLA and 500M samples in nuPlan with a single 8-GPU node. The resulting model achieves 64 DS on the CARLA longest6 v2 benchmark, outperforming other RL methods with more complex rewards by a large margin. Requiring only minimal adaptations from its use in CARLA, the same method is the best learning-based approach on nuPlan. It scores 91.3 in non-reactive and 90.6 in reactive traffic on the Val14 benchmark while being an order of magnitude faster than prior work. 

**Abstract (ZH)**: 我们研究自主驾驶中特权规划的强化学习方法。尽管最先进的方法基于规则，但这些方法难以处理长尾问题。相比之下，强化学习具有可扩展性，避免了模仿学习中的累积误差。用于驾驶的当今强化学习方法使用复杂的复合奖励，这些奖励由多个个体奖励相加而成，例如进度、位置或方向奖励。我们发现，当批量大小增加时，PPO 难以优化这些奖励的一个流行版本，这限制了这些方法的可扩展性。相反，我们提出了一种新的奖励设计，主要基于优化单一直观的奖励项：路线完成度。违规行为通过终止episode或乘性减少路线完成度来进行惩罚。我们发现，当我们使用我们简单的奖励训练PPO时，随着批量大小的增加，PPO具有良好的可扩展性，甚至可以提高性能。使用大型批量大小进行训练能够通过分布数据并行性实现高效扩展。我们在CARLA中将PPO扩展到3亿样本，在nuPlan中扩展到5亿样本，仅使用一个8-GPU节点。生成的模型在CARLA longest6 v2基准测试中达到64 DS，明显优于其他使用更复杂奖励的RL方法。仅需对其在CARLA中的使用进行少量适应，相同的方法在nuPlan中是最佳的数据驱动方法，分别在Val14基准测试中的非反应性交通和反应性交通中得分91.3和90.6，性能比先前工作快一个数量级。 

---
# Learning Underwater Active Perception in Simulation 

**Title (ZH)**: 基于仿真的水下主动感知学习 

**Authors**: Alexandre Cardaillac, Donald G. Dansereau  

**Link**: [PDF](https://arxiv.org/pdf/2504.17817)  

**Abstract**: When employing underwater vehicles for the autonomous inspection of assets, it is crucial to consider and assess the water conditions. Indeed, they have a significant impact on the visibility, which also affects robotic operations. Turbidity can jeopardise the whole mission as it may prevent correct visual documentation of the inspected structures. Previous works have introduced methods to adapt to turbidity and backscattering, however, they also include manoeuvring and setup constraints. We propose a simple yet efficient approach to enable high-quality image acquisition of assets in a broad range of water conditions. This active perception framework includes a multi-layer perceptron (MLP) trained to predict image quality given a distance to a target and artificial light intensity. We generated a large synthetic dataset including ten water types with different levels of turbidity and backscattering. For this, we modified the modelling software Blender to better account for the underwater light propagation properties. We validated the approach in simulation and showed significant improvements in visual coverage and quality of imagery compared to traditional approaches. The project code is available on our project page at this https URL. 

**Abstract (ZH)**: 使用水下车辆进行资产自主检查时，考虑和评估水况至关重要。确实，水况对能见度有重大影响，也影响着机器人的操作。悬浮物质可能会危及整个任务，因为它可能妨碍对检查结构的正确视觉记录。先前的工作引入了适应悬浮物质和后向散射的方法，然而这些方法也包括机动和设置约束。我们提出了一种简单而有效的方法，可以在广泛水况条件下实现高质量的图像获取。该主动感知框架包括一个训练用于根据目标距离和人工光强度预测图像质量的多层感知器（MLP）。我们生成了一个包含不同悬浮物质和后向散射水平的十种水类型的大型合成数据集。为此，我们修改了建模软件Blender，以更好地考虑水下光传播特性。我们在模拟中验证了该方法，并展示了与传统方法相比在视觉覆盖范围和图像质量上显著改进。项目代码可在我们的项目页面（此链接）获取。 

---
# Near-Driven Autonomous Rover Navigation in Complex Environments: Extensions to Urban Search-and-Rescue and Industrial Inspection 

**Title (ZH)**: 近地驱动自主探测车在复杂环境中的自主导航：面向城市搜索与救援及工业检查的应用扩展 

**Authors**: Dhadkan Shrestha, Lincoln Bhattarai  

**Link**: [PDF](https://arxiv.org/pdf/2504.17794)  

**Abstract**: This paper explores the use of an extended neuroevolutionary approach, based on NeuroEvolution of Augmenting Topologies (NEAT), for autonomous robots in dynamic environments associated with hazardous tasks like firefighting, urban search-and-rescue (USAR), and industrial inspections. Building on previous research, it expands the simulation environment to larger and more complex settings, demonstrating NEAT's adaptability across different applications. By integrating recent advancements in NEAT and reinforcement learning, the study uses modern simulation frameworks for realism and hybrid algorithms for optimization. Experimental results show that NEAT-evolved controllers achieve success rates comparable to state-of-the-art deep reinforcement learning methods, with superior structural adaptability. The agents reached ~80% success in outdoor tests, surpassing baseline models. The paper also highlights the benefits of transfer learning among tasks and evaluates the effectiveness of NEAT in complex 3D navigation. Contributions include evaluating NEAT for diverse autonomous applications and discussing real-world deployment considerations, emphasizing the approach's potential as an alternative or complement to deep reinforcement learning in autonomous navigation tasks. 

**Abstract (ZH)**: 基于NEAT扩展神经进化方法在动态环境自主机器人中的应用：以消防、都市搜救和工业检查为例 

---
