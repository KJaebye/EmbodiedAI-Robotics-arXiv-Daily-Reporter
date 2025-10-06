# Simulation to Rules: A Dual-VLM Framework for Formal Visual Planning 

**Title (ZH)**: 模拟到规则：一种形式视觉规划的双多模视域框架 

**Authors**: Yilun Hao, Yongchao Chen, Chuchu Fan, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03182)  

**Abstract**: Vision Language Models (VLMs) show strong potential for visual planning but struggle with precise spatial and long-horizon reasoning. In contrast, Planning Domain Definition Language (PDDL) planners excel at long-horizon formal planning, but cannot interpret visual inputs. Recent works combine these complementary advantages by enabling VLMs to turn visual planning problems into PDDL files for formal planning. However, while VLMs can generate PDDL problem files satisfactorily, they struggle to accurately generate the PDDL domain files, which describe all the planning rules. As a result, prior methods rely on human experts to predefine domain files or on constant environment access for refinement. We propose VLMFP, a Dual-VLM-guided framework that can autonomously generate both PDDL problem and domain files for formal visual planning. VLMFP introduces two VLMs to ensure reliable PDDL file generation: A SimVLM that simulates action consequences based on input rule descriptions, and a GenVLM that generates and iteratively refines PDDL files by comparing the PDDL and SimVLM execution results. VLMFP unleashes multiple levels of generalizability: The same generated PDDL domain file works for all the different instances under the same problem, and VLMs generalize to different problems with varied appearances and rules. We evaluate VLMFP with 6 grid-world domains and test its generalization to unseen instances, appearance, and game rules. On average, SimVLM accurately describes 95.5%, 82.6% of scenarios, simulates 85.5%, 87.8% of action sequence, and judges 82.4%, 85.6% goal reaching for seen and unseen appearances, respectively. With the guidance of SimVLM, VLMFP can generate PDDL files to reach 70.0%, 54.1% valid plans for unseen instances in seen and unseen appearances, respectively. Project page: this https URL. 

**Abstract (ZH)**: 基于双VLM的视觉规划PDDL文件自动生成框架 

---
# Optimal Smooth Coverage Trajectory Planning for Quadrotors in Cluttered Environment 

**Title (ZH)**: 面向杂乱环境四旋翼的最优平滑覆盖轨迹规划 

**Authors**: Duanjiao Li, Yun Chen, Ying Zhang, Junwen Yao, Dongyue Huang, Jianguo Zhang, Ning Ding  

**Link**: [PDF](https://arxiv.org/pdf/2510.03169)  

**Abstract**: For typical applications of UAVs in power grid scenarios, we construct the problem as planning UAV trajectories for coverage in cluttered environments. In this paper, we propose an optimal smooth coverage trajectory planning algorithm. The algorithm consists of two stages. In the front-end, a Genetic Algorithm (GA) is employed to solve the Traveling Salesman Problem (TSP) for Points of Interest (POIs), generating an initial sequence of optimized visiting points. In the back-end, the sequence is further optimized by considering trajectory smoothness, time consumption, and obstacle avoidance. This is formulated as a nonlinear least squares problem and solved to produce a smooth coverage trajectory that satisfies these constraints. Numerical simulations validate the effectiveness of the proposed algorithm, ensuring UAVs can smoothly cover all POIs in cluttered environments. 

**Abstract (ZH)**: 对于电力电网场景中典型无人机应用，我们将问题定义为在复杂环境下的无人机覆盖轨迹规划。在本文中，我们提出了一种最优平滑覆盖轨迹规划算法。该算法分为两个阶段。前端采用遗传算法（GA）求解兴趣点（POIs）的旅行商问题（TSP），生成初始的优化访问序列。后端进一步优化序列，考虑轨迹平滑性、时间消耗和障碍物规避。该问题被形式化为非线性最小二乘问题并求解，生成满足这些约束的平滑覆盖轨迹。数值仿真验证了所提算法的有效性，确保无人机能够顺利覆盖所有POIs在复杂环境中的所有位置。 

---
# MM-Nav: Multi-View VLA Model for Robust Visual Navigation via Multi-Expert Learning 

**Title (ZH)**: MM-Nav: 多视图多专家学习的鲁棒视觉导航模型 

**Authors**: Tianyu Xu, Jiawei Chen, Jiazhao Zhang, Wenyao Zhang, Zekun Qi, Minghan Li, Zhizheng Zhang, He Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03142)  

**Abstract**: Visual navigation policy is widely regarded as a promising direction, as it mimics humans by using egocentric visual observations for navigation. However, optical information of visual observations is difficult to be explicitly modeled like LiDAR point clouds or depth maps, which subsequently requires intelligent models and large-scale data. To this end, we propose to leverage the intelligence of the Vision-Language-Action (VLA) model to learn diverse navigation capabilities from synthetic expert data in a teacher-student manner. Specifically, we implement the VLA model, MM-Nav, as a multi-view VLA (with 360 observations) based on pretrained large language models and visual foundation models. For large-scale navigation data, we collect expert data from three reinforcement learning (RL) experts trained with privileged depth information in three challenging tailor-made environments for different navigation capabilities: reaching, squeezing, and avoiding. We iteratively train our VLA model using data collected online from RL experts, where the training ratio is dynamically balanced based on performance on individual capabilities. Through extensive experiments in synthetic environments, we demonstrate that our model achieves strong generalization capability. Moreover, we find that our student VLA model outperforms the RL teachers, demonstrating the synergistic effect of integrating multiple capabilities. Extensive real-world experiments further confirm the effectiveness of our method. 

**Abstract (ZH)**: 视觉导航策略被视为一个有前景的方向，因为它通过第一人称视觉观察进行导航，模仿人类行为。然而，视觉观察的光学信息难以像激光雷达点云或深度图那样明确建模，这需要智能模型和大规模数据。为此，我们提出利用Vision-Language-Action（VLA）模型的智能，在教师-学生模式下从合成专家数据中学习多样的导航能力。具体而言，我们基于预训练的大语言模型和视觉基础模型实现了多视角VLA模型（具有360度观察视角）MM-Nav。对于大规模导航数据，我们从三位使用优先级深度信息训练的强化学习（RL）专家在三个针对不同导航能力定制的挑战环境中收集专家数据，分别用于接近、挤入和避开。我们从RL专家在线收集的数据中迭代训练我们的VLA模型，在不同能力上的训练比例基于性能动态平衡。通过在合成环境中的广泛实验，我们展示了我们的模型具备强大的泛化能力。此外，我们发现我们的学生VLA模型在导航能力上优于RL教师，证明了整合多种能力的协同效应。进一步的现实世界实验也证实了我们方法的有效性。 

---
# Learning Stability Certificate for Robotics in Real-World Environments 

**Title (ZH)**: 机器人在实际环境中的稳定性证书学习 

**Authors**: Zhe Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.03123)  

**Abstract**: Stability certificates play a critical role in ensuring the safety and reliability of robotic systems. However, deriving these certificates for complex, unknown systems has traditionally required explicit knowledge of system dynamics, often making it a daunting task. This work introduces a novel framework that learns a Lyapunov function directly from trajectory data, enabling the certification of stability for autonomous systems without needing detailed system models. By parameterizing the Lyapunov candidate using a neural network and ensuring positive definiteness through Cholesky factorization, our approach automatically identifies whether the system is stable under the given trajectory. To address the challenges posed by noisy, real-world data, we allow for controlled violations of the stability condition, focusing on maintaining high confidence in the stability certification process. Our results demonstrate that this framework can provide data-driven stability guarantees, offering a robust method for certifying the safety of robotic systems in dynamic, real-world environments. This approach works without access to the internal control algorithms, making it applicable even in situations where system behavior is opaque or proprietary. The tool for learning the stability proof is open-sourced by this research: this https URL. 

**Abstract (ZH)**: 稳定性证书在确保机器人系统的安全性和可靠性中发挥着关键作用。然而，为复杂且未知的系统推导这些证书通常需要明确了解系统的动力学，这往往是一个艰巨的任务。本工作引入了一个新的框架，可以直接从轨迹数据中学习李雅普诺夫函数，从而使人们能够在无需详细系统模型的情况下为自主系统验证稳定性。通过使用神经网络参数化李雅普诺夫候选函数，并通过乔莱斯基分解确保正定性，我们的方法可以自动确定系统在给定轨迹下的稳定性。为了应对实际数据中的噪声挑战，我们允许在保持稳定性认证过程高置信度的前提下对稳定性条件进行可控的违反。我们的结果显示，该框架可以提供基于数据的稳定性保证，从而为动态的现实环境中的机器人系统安全认证提供一种稳健的方法。此方法无需访问内部控制算法，使其在系统行为不透明或专有的情况下同样适用。研究团队开源了用于学习稳定性证明的工具：this https URL。 

---
# Whisker-based Tactile Flight for Tiny Drones 

**Title (ZH)**: 基于 whisker 的/tiny无人机触觉飞行 

**Authors**: Chaoxiang Ye, Guido de Croon, Salua Hamaza  

**Link**: [PDF](https://arxiv.org/pdf/2510.03119)  

**Abstract**: Tiny flying robots hold great potential for search-and-rescue, safety inspections, and environmental monitoring, but their small size limits conventional sensing-especially with poor-lighting, smoke, dust or reflective obstacles. Inspired by nature, we propose a lightweight, 3.2-gram, whisker-based tactile sensing apparatus for tiny drones, enabling them to navigate and explore through gentle physical interaction. Just as rats and moles use whiskers to perceive surroundings, our system equips drones with tactile perception in flight, allowing obstacle sensing even in pitch-dark conditions. The apparatus uses barometer-based whisker sensors to detect obstacle locations while minimising destabilisation. To address sensor noise and drift, we develop a tactile depth estimation method achieving sub-6 mm accuracy. This enables drones to navigate, contour obstacles, and explore confined spaces solely through touch-even in total darkness along both soft and rigid surfaces. Running fully onboard a 192-KB RAM microcontroller, the system supports autonomous tactile flight and is validated in both simulation and real-world tests. Our bio-inspired approach redefines vision-free navigation, opening new possibilities for micro aerial vehicles in extreme environments. 

**Abstract (ZH)**: 基于触觉传感的小型无人机轻量化探知装置及其在低光照环境下的导航探索技术 

---
# Embracing Evolution: A Call for Body-Control Co-Design in Embodied Humanoid Robot 

**Title (ZH)**: 拥抱进化：关于类人机器人身体与控制协同设计的呼吁 

**Authors**: Guiliang Liu, Bo Yue, Yi Jin Kim, Kui Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.03081)  

**Abstract**: Humanoid robots, as general-purpose physical agents, must integrate both intelligent control and adaptive morphology to operate effectively in diverse real-world environments. While recent research has focused primarily on optimizing control policies for fixed robot structures, this position paper argues for evolving both control strategies and humanoid robots' physical structure under a co-design mechanism. Inspired by biological evolution, this approach enables robots to iteratively adapt both their form and behavior to optimize performance within task-specific and resource-constrained contexts. Despite its promise, co-design in humanoid robotics remains a relatively underexplored domain, raising fundamental questions about its feasibility and necessity in achieving true embodied intelligence. To address these challenges, we propose practical co-design methodologies grounded in strategic exploration, Sim2Real transfer, and meta-policy learning. We further argue for the essential role of co-design by analyzing it from methodological, application-driven, and community-oriented perspectives. Striving to guide and inspire future studies, we present open research questions, spanning from short-term innovations to long-term goals. This work positions co-design as a cornerstone for developing the next generation of intelligent and adaptable humanoid agents. 

**Abstract (ZH)**: 人形机器人作为通用物理代理，必须整合智能控制与自适应形态，以在多样的现实环境中有效运作。虽然近期研究主要集中在优化固定机器人结构的控制策略上，本立场论文主张在共融设计机制下同时优化控制策略和人形机器人的物理结构。受到生物进化机制的启发，这种方法使得机器人能够迭代地调整其形态和行为，以在特定任务和资源受限的背景下优化性能。尽管共融设计在人形机器人领域具有巨大潜力，但仍是一个相对未被充分探索的领域，引发了关于其可行性和必要性的基本问题，以实现真正的实体智能。为了应对这些挑战，我们提出了基于战略探索、Sim2Real转移和元策略学习的实用共融设计方法，并从方法论、应用驱动和社区导向等多个视角论证共融设计的必要性。我们进一步提出了一系列开放性研究问题，涵盖短期创新和长期目标，旨在引导和启发未来的研究。本工作将共融设计定位为开发新一代智能且适应性强的人形代理的基础。 

---
# Long-Term Human Motion Prediction Using Spatio-Temporal Maps of Dynamics 

**Title (ZH)**: 基于动力学时空图的长期人体运动预测 

**Authors**: Yufei Zhu, Andrey Rudenko, Tomasz P. Kucner, Achim J. Lilienthal, Martin Magnusson  

**Link**: [PDF](https://arxiv.org/pdf/2510.03031)  

**Abstract**: Long-term human motion prediction (LHMP) is important for the safe and efficient operation of autonomous robots and vehicles in environments shared with humans. Accurate predictions are important for applications including motion planning, tracking, human-robot interaction, and safety monitoring. In this paper, we exploit Maps of Dynamics (MoDs), which encode spatial or spatio-temporal motion patterns as environment features, to achieve LHMP for horizons of up to 60 seconds. We propose an MoD-informed LHMP framework that supports various types of MoDs and includes a ranking method to output the most likely predicted trajectory, improving practical utility in robotics. Further, a time-conditioned MoD is introduced to capture motion patterns that vary across different times of day. We evaluate MoD-LHMP instantiated with three types of MoDs. Experiments on two real-world datasets show that MoD-informed method outperforms learning-based ones, with up to 50\% improvement in average displacement error, and the time-conditioned variant achieves the highest accuracy overall. Project code is available at this https URL 

**Abstract (ZH)**: 长期人类运动预测（LHMP）对于在与人类共享环境中的自主机器人和车辆的安全和高效运行至关重要。精确的预测对于运动规划、跟踪、人机交互和安全监控等应用非常重要。在本文中，我们利用动力学地图（MoDs）来实现长达60秒的LHMP，MoDs将空间或时空运动模式编码为环境特征。我们提出了一种基于MoDs的LHMP框架，支持多种类型的MoDs，并包括一种排名方法来输出最可能的预测轨迹，从而提高其实用性。此外，我们引入了时间条件下的MoDs以捕捉不同时间段变化的运动模式。我们用三种类型的MoDs实例化MoD-LHMP，并在两个真实世界数据集上的实验表明，基于MoDs的方法优于基于学习的方法，在平均位移误差上最多可提高50%，时间条件下的变体总体上达到最高精度。源代码可通过以下链接获取：this https URL。 

---
# HumanoidExo: Scalable Whole-Body Humanoid Manipulation via Wearable Exoskeleton 

**Title (ZH)**: HumanoidExo:基于可穿戴外骨骼的可扩展全身人形机器人操纵 

**Authors**: Rui Zhong, Yizhe Sun, Junjie Wen, Jinming Li, Chuang Cheng, Wei Dai, Zhiwen Zeng, Huimin Lu, Yichen Zhu, Yi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03022)  

**Abstract**: A significant bottleneck in humanoid policy learning is the acquisition of large-scale, diverse datasets, as collecting reliable real-world data remains both difficult and cost-prohibitive. To address this limitation, we introduce HumanoidExo, a novel system that transfers human motion to whole-body humanoid data. HumanoidExo offers a high-efficiency solution that minimizes the embodiment gap between the human demonstrator and the robot, thereby tackling the scarcity of whole-body humanoid data. By facilitating the collection of more voluminous and diverse datasets, our approach significantly enhances the performance of humanoid robots in dynamic, real-world scenarios. We evaluated our method across three challenging real-world tasks: table-top manipulation, manipulation integrated with stand-squat motions, and whole-body manipulation. Our results empirically demonstrate that HumanoidExo is a crucial addition to real-robot data, as it enables the humanoid policy to generalize to novel environments, learn complex whole-body control from only five real-robot demonstrations, and even acquire new skills (i.e., walking) solely from HumanoidExo data. 

**Abstract (ZH)**: 人形机器人政策学习中的一个重要瓶颈是获取大规模、多样化的数据集，因为可靠的真实世界数据的收集既困难又成本高昂。为了解决这一限制，我们介绍了HumanoidExo系统，该系统将人类运动转移为全身人形数据。HumanoidExo提供了一种高效解决方案，最小化了人类示范者与机器人之间的实体差距，从而解决了全身人形数据稀缺的问题。通过促进更大规模和多样化的数据集收集，我们的方法显著提高了人形机器人在动态真实世界场景中的性能。我们在三个极具挑战性的实际任务中评估了我们的方法：桌面操作、结合站立蹲下动作的操作，以及全身操作。我们的结果实证展示了HumanoidExo对于真实机器人数据的重要补充作用，它使人为策略能够泛化到新环境，仅从五个真实机器人示范中学习复杂的全身控制，甚至仅通过HumanoidExo数据就能获取新技能（如走路）。 

---
# 3D-CovDiffusion: 3D-Aware Diffusion Policy for Coverage Path Planning 

**Title (ZH)**: 3D-CovDiffusion: 三维导向的覆盖路径规划扩散策略 

**Authors**: Chenyuan Chen, Haoran Ding, Ran Ding, Tianyu Liu, Zewen He, Anqing Duan, Dezhen Song, Xiaodan Liang, Yoshihiko Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2510.03011)  

**Abstract**: Diffusion models, as a class of deep generative models, have recently emerged as powerful tools for robot skills by enabling stable training with reliable convergence. In this paper, we present an end-to-end framework for generating long, smooth trajectories that explicitly target high surface coverage across various industrial tasks, including polishing, robotic painting, and spray coating. The conventional methods are always fundamentally constrained by their predefined functional forms, which limit the shapes of the trajectories they can represent and make it difficult to handle complex and diverse tasks. Moreover, their generalization is poor, often requiring manual redesign or extensive parameter tuning when applied to new scenarios. These limitations highlight the need for more expressive generative models, making diffusion-based approaches a compelling choice for trajectory generation. By iteratively denoising trajectories with carefully learned noise schedules and conditioning mechanisms, diffusion models not only ensure smooth and consistent motion but also flexibly adapt to the task context. In experiments, our method improves trajectory continuity, maintains high coverage, and generalizes to unseen shapes, paving the way for unified end-to-end trajectory learning across industrial surface-processing tasks without category-specific models. On average, our approach improves Point-wise Chamfer Distance by 98.2\% and smoothness by 97.0\%, while increasing surface coverage by 61\% compared to prior methods. The link to our code can be found \href{this https URL}{here}. 

**Abstract (ZH)**: 基于扩散模型的端到端轨迹生成框架：面向工业任务的高表面覆盖率平滑轨迹生成 

---
# Real-Time Nonlinear Model Predictive Control of Heavy-Duty Skid-Steered Mobile Platform for Trajectory Tracking Tasks 

**Title (ZH)**: 重型履带滑移转向移动平台的实时非线性模型预测控制及其轨迹跟踪任务 

**Authors**: Alvaro Paz, Pauli Mustalahti, Mohammad Dastranj, Jouni Mattila  

**Link**: [PDF](https://arxiv.org/pdf/2510.02976)  

**Abstract**: This paper presents a framework for real-time optimal controlling of a heavy-duty skid-steered mobile platform for trajectory tracking. The importance of accurate real-time performance of the controller lies in safety considerations of situations where the dynamic system under control is affected by uncertainties and disturbances, and the controller should compensate for such phenomena in order to provide stable performance. A multiple-shooting nonlinear model-predictive control framework is proposed in this paper. This framework benefits from suitable algorithm along with readings from various sensors for genuine real-time performance with extremely high accuracy. The controller is then tested for tracking different trajectories where it demonstrates highly desirable performance in terms of both speed and accuracy. This controller shows remarkable improvement when compared to existing nonlinear model-predictive controllers in the literature that were implemented on skid-steered mobile platforms. 

**Abstract (ZH)**: 本文提出了一种用于轨迹跟踪的重载式滑移转向移动平台实时最优控制框架。该控制器的准确实时性能对于受不确定性和干扰影响的动力系统安全至关重要，控制器需补偿此类现象以提供稳定的性能。本文提出了一种多段射击非线性模型预测控制框架，该框架结合了合适的算法和各种传感器读数，实现了极高的实时准确性能。该控制器被测试用于跟踪不同的轨迹，展示了在速度和精度方面的出色性能。与文献中已在滑移转向移动平台上实现的非线性模型预测控制器相比，该控制器显示出显著的改进。 

---
# AI-Enhanced Kinematic Modeling of Flexible Manipulators Using Multi-IMU Sensor Fusion 

**Title (ZH)**: 基于多IMU传感器融合的智能增强柔性 manipulator 动态建模 

**Authors**: Amir Hossein Barjini, Jouni Mattila  

**Link**: [PDF](https://arxiv.org/pdf/2510.02975)  

**Abstract**: This paper presents a novel framework for estimating the position and orientation of flexible manipulators undergoing vertical motion using multiple inertial measurement units (IMUs), optimized and calibrated with ground truth data. The flexible links are modeled as a series of rigid segments, with joint angles estimated from accelerometer and gyroscope measurements acquired by cost-effective IMUs. A complementary filter is employed to fuse the measurements, with its parameters optimized through particle swarm optimization (PSO) to mitigate noise and delay. To further improve estimation accuracy, residual errors in position and orientation are compensated using radial basis function neural networks (RBFNN). Experimental results validate the effectiveness of the proposed intelligent multi-IMU kinematic estimation method, achieving root mean square errors (RMSE) of 0.00021~m, 0.00041~m, and 0.00024~rad for $y$, $z$, and $\theta$, respectively. 

**Abstract (ZH)**: 本文提出了一种基于多惯性测量单元（IMU）并使用地面truth数据优化与校准的柔性 manipulator 在垂直运动中位置和姿态估计的新型框架。柔性连杆被建模为一系列刚性段，关节角度从低成本IMU获取的加速度计和陀螺仪测量值中估计。采用互补滤波器融合测量值，并通过粒子群优化（PSO）优化参数以减轻噪声和延迟。为进一步提高估计精度，位置和姿态的剩余误差通过径向基函数神经网络（RBFNN）进行补偿。实验结果验证了所提出的智能多IMU运动学估计方法的有效性，分别实现了根 mean square error（RMSE）为0.00021 m、0.00041 m和0.00024 rad的误差。 

---
# YawSitter: Modeling and Controlling a Tail-Sitter UAV with Enhanced Yaw Control 

**Title (ZH)**: YawSitter: 建模与控制一种增强Yaw控制的Tail-Sitter无人飞行器 

**Authors**: Amir Habel, Fawad Mehboob, Jeffrin Sam, Clement Fortin, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2510.02968)  

**Abstract**: Achieving precise lateral motion modeling and decoupled control in hover remains a significant challenge for tail-sitter Unmanned Aerial Vehicles (UAVs), primarily due to complex aerodynamic couplings and the absence of welldefined lateral dynamics. This paper presents a novel modeling and control strategy that enhances yaw authority and lateral motion by introducing a sideslip force model derived from differential propeller slipstream effects acting on the fuselage under differential thrust. The resulting lateral force along the body y-axis enables yaw-based lateral position control without inducing roll coupling. The control framework employs a YXZ Euler rotation formulation to accurately represent attitude and incorporate gravitational components while directly controlling yaw in the yaxis, thereby improving lateral dynamic behavior and avoiding singularities. The proposed approach is validated through trajectory-tracking simulations conducted in a Unity-based environment. Tests on both rectangular and circular paths in hover mode demonstrate stable performance, with low mean absolute position errors and yaw deviations constrained within 5.688 degrees. These results confirm the effectiveness of the proposed lateral force generation model and provide a foundation for the development of agile, hover-capable tail-sitter UAVs. 

**Abstract (ZH)**: 实现悬停状态下尾坐无人空中车辆精确的横向运动建模与解耦控制仍然是一个重大挑战，主要是由于复杂的气动耦合和缺乏明确的横向动力学定义。本文提出了一种新的建模和控制策略，通过引入由差动推力引起的旋翼滑流效应作用在机身上的侧滑力模型来增强航向权限和横向运动。由此产生的沿机身y轴的横向力使横向位置控制基于航向，而不引起滚转耦合。控制框架采用YXZ欧拉旋转表示法，准确表示姿态并包含重力组件，同时直接控制y轴上的航向，从而改善横向动力学行为并避免奇点。所提出的方法通过在基于Unity的环境中进行轨迹跟踪仿真得到验证。悬停模式下沿矩形和圆形轨迹的测试显示稳定性能，平均绝对位置误差低，航向偏差限于5.688度以内。这些结果证实了所提出横向力生成模型的有效性，并为开发敏捷悬停尾坐无人空中车辆奠定了基础。 

---
# Single-Rod Brachiation Robot: Mechatronic Control Design and Validation of Prejump Phases 

**Title (ZH)**: 单棒前空翻机器人：机械电子控制设计与预翻转阶段验证 

**Authors**: Juraj Lieskovský, Hijiri Akahane, Aoto Osawa, Jaroslav Bušek, Ikuo Mizuuchi, Tomáš Vyhlídal  

**Link**: [PDF](https://arxiv.org/pdf/2510.02946)  

**Abstract**: A complete mechatronic design of a minimal configuration brachiation robot is presented. The robot consists of a single rigid rod with gripper mechanisms attached to both ends. The grippers are used to hang the robot on a horizontal bar on which it swings or rotates. The motion is imposed by repositioning the robot's center of mass, which is performed using a crank-slide mechanism. Based on a non-linear model, an optimal control strategy is proposed, for repositioning the center of mass in a bang-bang manner. Consequently, utilizing the concept of input-output linearization, a continuous control strategy is proposed that takes into account the limited torque of the crank-slide mechanism and its geometry. An increased attention is paid to energy accumulation towards the subsequent jump stage of the brachiation. These two strategies are validated and compared in simulations. The continuous control strategy is then also implemented within a low-cost STM32-based control system, and both the swing and rotation stages of the brachiation motion are experimentally validated. 

**Abstract (ZH)**: 一种minimal configuration brachiation机器人的完整机电设计被提出。该机器人由一根带有 gripper 机制的两端的刚性杆组成。抓取器用于使机器人挂在水平横杆上进行摆动或旋转。通过重新定位机器人的质心来施加运动，该动作由凸轮-滑块机构完成。基于非线性模型，提出了一种最优控制策略，以bang-bang方式重新定位质心。随后，利用输入-输出线性化概念，提出了一种考虑凸轮-滑块机构的有限转矩及其几何形状的连续控制策略。特别关注于能量积累以改善后续跳跃阶段的 brachiation 性能。这两种策略在仿真中得到验证和比较。随后，连续控制策略在基于低成本STM32的控制系统中实现，并且摆动和旋转阶段的brachiation运动也在实验中得到了验证。 

---
# Metrics vs Surveys: Can Quantitative Measures Replace Human Surveys in Social Robot Navigation? A Correlation Analysis 

**Title (ZH)**: 度量指标 vs 调查问卷：定量指标能否替代人类调查在社会机器人导航中的作用？相关性分析 

**Authors**: Stefano Trepella, Mauro Martini, Noé Pérez-Higueras, Andrea Ostuni, Fernando Caballero, Luis Merino, Marcello Chiaberge  

**Link**: [PDF](https://arxiv.org/pdf/2510.02941)  

**Abstract**: Social, also called human-aware, navigation is a key challenge for the integration of mobile robots into human environments. The evaluation of such systems is complex, as factors such as comfort, safety, and legibility must be considered. Human-centered assessments, typically conducted through surveys, provide reliable insights but are costly, resource-intensive, and difficult to reproduce or compare across systems. Alternatively, numerical social navigation metrics are easy to compute and facilitate comparisons, yet the community lacks consensus on a standard set of metrics.
This work explores the relationship between numerical metrics and human-centered evaluations to identify potential correlations. If specific quantitative measures align with human perceptions, they could serve as standardized evaluation tools, reducing the dependency on surveys. Our results indicate that while current metrics capture some aspects of robot navigation behavior, important subjective factors remain insufficiently represented and new metrics are necessary. 

**Abstract (ZH)**: 社会导向的导航是将移动机器人融入人类环境中的关键挑战。此类系统的评估具有复杂性，因为舒适度、安全性和可读性等因素必须考虑。以人类为中心的评估通常通过调查进行，提供了可靠的观点，但成本高、资源密集且难以跨系统复制和比较。相反，数值社会导航指标易于计算并促进比较，然而社区尚未就标准指标集达成共识。

本文探索数值指标与以人类为中心的评估之间的关系，以识别潜在的相关性。如果特定的定量措施与人的感知相一致，它们可以作为标准化评估工具，减少对调查的依赖。我们的结果显示，尽管当前的指标捕捉了部分机器人导航行为的方面，但重要的主观因素仍缺乏充分代表，因此需要新的指标。 

---
# Point Cloud-Based Control Barrier Functions for Model Predictive Control in Safety-Critical Navigation of Autonomous Mobile Robots 

**Title (ZH)**: 基于点云的控制屏障函数在自主移动机器人安全关键导航中的模型预测控制 

**Authors**: Faduo Liang, Yunfeng Yang, Shi-Lu Dai  

**Link**: [PDF](https://arxiv.org/pdf/2510.02885)  

**Abstract**: In this work, we propose a novel motion planning algorithm to facilitate safety-critical navigation for autonomous mobile robots. The proposed algorithm integrates a real-time dynamic obstacle tracking and mapping system that categorizes point clouds into dynamic and static components. For dynamic point clouds, the Kalman filter is employed to estimate and predict their motion states. Based on these predictions, we extrapolate the future states of dynamic point clouds, which are subsequently merged with static point clouds to construct the forward-time-domain (FTD) map. By combining control barrier functions (CBFs) with nonlinear model predictive control, the proposed algorithm enables the robot to effectively avoid both static and dynamic obstacles. The CBF constraints are formulated based on risk points identified through collision detection between the predicted future states and the FTD map. Experimental results from both simulated and real-world scenarios demonstrate the efficacy of the proposed algorithm in complex environments. In simulation experiments, the proposed algorithm is compared with two baseline approaches, showing superior performance in terms of safety and robustness in obstacle avoidance. The source code is released for the reference of the robotics community. 

**Abstract (ZH)**: 本研究提出了一种新型运动规划算法，以促进自主移动机器人在安全关键导航中的应用。该提出的算法整合了一个实时动态障碍跟踪和制图系统，将点云分类为动态和静态组件。对于动态点云，使用卡尔曼滤波器来估计和预测其运动状态。基于这些预测，我们外推动态点云的未来状态，并将这些状态与静态点云合并以构建未来时间域（FTD）地图。通过结合控制屏障函数（CBFs）与非线性模型预测控制，提出的算法使机器人能够有效避免静态和动态障碍。CBF约束是基于风险点制定的，这些风险点是通过检测预测未来状态与FTD地图之间的碰撞检测来识别的。实验结果来自模拟和真实场景，证明了该算法在复杂环境中的有效性。在模拟实验中，该算法与两种基准方法进行了比较，显示出在障碍物避免方面的优越的安全性和鲁棒性。源代码已发布供机器人社区参考。 

---
# Novel UWB Synthetic Aperture Radar Imaging for Mobile Robot Mapping 

**Title (ZH)**: 基于新型UWB合成孔径雷达成像的移动机器人建图 

**Authors**: Charith Premachandra, U-Xuan Tan  

**Link**: [PDF](https://arxiv.org/pdf/2510.02874)  

**Abstract**: Traditional exteroceptive sensors in mobile robots, such as LiDARs and cameras often struggle to perceive the environment in poor visibility conditions. Recently, radar technologies, such as ultra-wideband (UWB) have emerged as potential alternatives due to their ability to see through adverse environmental conditions (e.g. dust, smoke and rain). However, due to the small apertures with low directivity, the UWB radars cannot reconstruct a detailed image of its field of view (FOV) using a single scan. Hence, a virtual large aperture is synthesized by moving the radar along a mobile robot path. The resulting synthetic aperture radar (SAR) image is a high-definition representation of the surrounding environment. Hence, this paper proposes a pipeline for mobile robots to incorporate UWB radar-based SAR imaging to map an unknown environment. Finally, we evaluated the performance of classical feature detectors: SIFT, SURF, BRISK, AKAZE and ORB to identify loop closures using UWB SAR images. The experiments were conducted emulating adverse environmental conditions. The results demonstrate the viability and effectiveness of UWB SAR imaging for high-resolution environmental mapping and loop closure detection toward more robust and reliable robotic perception systems. 

**Abstract (ZH)**: 移动机器人中基于UWB雷达的合成孔径雷达成像方法及其在高分辨率环境映射和回环检测中的应用 

---
# Action Deviation-Aware Inference for Low-Latency Wireless Robots 

**Title (ZH)**: 基于动作偏差的低-latency无线机器人推断 

**Authors**: Jeyoung Park, Yeonsub Lim, Seungeun Oh, Jihong Park, Jinho Choi, Seong-Lyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.02851)  

**Abstract**: To support latency-sensitive AI applications ranging from autonomous driving to industrial robot manipulation, 6G envisions distributed ML, connecting distributed computational resources in edge and cloud over hyper-reliable low-latency communication (HRLLC). In this setting, speculative decoding can facilitate collaborative inference of models distributively deployed: an on-device draft model locally generates drafts and a remote server-based target model verifies and corrects them, resulting lower latency. However, unlike autoregressive text generation, behavior cloning policies, typically used for embodied AI applications like robot manipulation and autonomous driving, cannot parallelize verification and correction for multiple drafts as each action depends on observation which needs to be updated by a previous action. To this end, we propose Action Deviation-Aware Hybrid Inference, wherein the draft model estimates an action's need for verification and correction by the target model and selectively skips communication and computation for server operations. Action deviation shows a strong correlation with action's rejection probability by the target model, enabling selective skipping. We derive the path deviation threshold that balances the transmission rate and the inference performance, and we empirically show that action deviation-aware hybrid inference reduces uplink transmission and server operation by 40%, while lowering end-to-end latency by 33.32% relative to hybrid inference without skipping and achieving task success rate up to 97.03% of that of target model only inference. 

**Abstract (ZH)**: 面向自主驾驶到工业机器人操作等时延敏感AI应用的6G分布式ML：基于行为偏差的混合推理方法 

---
# Assist-as-needed Control for FES in Foot Drop Management 

**Title (ZH)**: 按需辅助控制在足下垂管理中的应用 

**Authors**: Andreas Christou, Elliot Lister, Georgia Andreopoulou, Don Mahad, Sethu Vijayakumar  

**Link**: [PDF](https://arxiv.org/pdf/2510.02808)  

**Abstract**: Foot drop is commonly managed using Functional Electrical Stimulation (FES), typically delivered via open-loop controllers with fixed stimulation intensities. While users may manually adjust the intensity through external controls, this approach risks overstimulation, leading to muscle fatigue and discomfort, or understimulation, which compromises dorsiflexion and increases fall risk. In this study, we propose a novel closed-loop FES controller that dynamically adjusts the stimulation intensity based on real-time toe clearance, providing "assistance as needed". We evaluate this system by inducing foot drop in healthy participants and comparing the effects of the closed-loop controller with a traditional open-loop controller across various walking conditions, including different speeds and surface inclinations. Kinematic data reveal that our closed-loop controller maintains adequate toe clearance without significantly affecting the joint angles of the hips, the knees, and the ankles, and while using significantly lower stimulation intensities compared to the open-loop controller. These findings suggest that the proposed method not only matches the effectiveness of existing systems but also offers the potential for reduced muscle fatigue and improved long-term user comfort and adherence. 

**Abstract (ZH)**: 基于实时趾 Clearance的闭合环路功能性电刺激控制器管理足下垂研究 

---
# Work Zones challenge VLM Trajectory Planning: Toward Mitigation and Robust Autonomous Driving 

**Title (ZH)**: 工作区挑战基于视觉学习的轨迹规划：向缓解和稳健自动驾驶迈进 

**Authors**: Yifan Liao, Zhen Sun, Xiaoyun Qiu, Zixiao Zhao, Wenbing Tang, Xinlei He, Xinhu Zheng, Tianwei Zhang, Xinyi Huang, Xingshuo Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.02803)  

**Abstract**: Visual Language Models (VLMs), with powerful multimodal reasoning capabilities, are gradually integrated into autonomous driving by several automobile manufacturers to enhance planning capability in challenging environments. However, the trajectory planning capability of VLMs in work zones, which often include irregular layouts, temporary traffic control, and dynamically changing geometric structures, is still unexplored. To bridge this gap, we conduct the \textit{first} systematic study of VLMs for work zone trajectory planning, revealing that mainstream VLMs fail to generate correct trajectories in $68.0%$ of cases. To better understand these failures, we first identify candidate patterns via subgraph mining and clustering analysis, and then confirm the validity of $8$ common failure patterns through human verification. Building on these findings, we propose REACT-Drive, a trajectory planning framework that integrates VLMs with Retrieval-Augmented Generation (RAG). Specifically, REACT-Drive leverages VLMs to convert prior failure cases into constraint rules and executable trajectory planning code, while RAG retrieves similar patterns in new scenarios to guide trajectory generation. Experimental results on the ROADWork dataset show that REACT-Drive yields a reduction of around $3\times$ in average displacement error relative to VLM baselines under evaluation with Qwen2.5-VL. In addition, REACT-Drive yields the lowest inference time ($0.58$s) compared with other methods such as fine-tuning ($17.90$s). We further conduct experiments using a real vehicle in 15 work zone scenarios in the physical world, demonstrating the strong practicality of REACT-Drive. 

**Abstract (ZH)**: 视觉语言模型（VLMs）在工作区路径规划中的系统研究：整合检索增强生成（RAG）以提升自动驾驶能力 

---
# Flow with the Force Field: Learning 3D Compliant Flow Matching Policies from Force and Demonstration-Guided Simulation Data 

**Title (ZH)**: 遵循力场流动：从力和示范指导的模拟数据中学习3D顺应性流匹配策略 

**Authors**: Tianyu Li, Yihan Li, Zizhe Zhang, Nadia Figueroa  

**Link**: [PDF](https://arxiv.org/pdf/2510.02738)  

**Abstract**: While visuomotor policy has made advancements in recent years, contact-rich tasks still remain a challenge. Robotic manipulation tasks that require continuous contact demand explicit handling of compliance and force. However, most visuomotor policies ignore compliance, overlooking the importance of physical interaction with the real world, often leading to excessive contact forces or fragile behavior under uncertainty. Introducing force information into vision-based imitation learning could help improve awareness of contacts, but could also require a lot of data to perform well. One remedy for data scarcity is to generate data in simulation, yet computationally taxing processes are required to generate data good enough not to suffer from the Sim2Real gap. In this work, we introduce a framework for generating force-informed data in simulation, instantiated by a single human demonstration, and show how coupling with a compliant policy improves the performance of a visuomotor policy learned from synthetic data. We validate our approach on real-robot tasks, including non-prehensile block flipping and a bi-manual object moving, where the learned policy exhibits reliable contact maintenance and adaptation to novel conditions. Project Website: this https URL 

**Abstract (ZH)**: 尽管近年来视知觉运动策略取得了进展，但富含接触的任务仍然是一项挑战。要求连续接触的机器人操作任务需要明确处理顺应性和力。然而，大多数视知觉运动策略忽视了顺应性，忽略了与真实世界物理互动的重要性，常常导致不确定情况下的接触力过大或行为脆弱。将力信息引入基于视觉的imitation learning可以帮助提高对接触的意识，但也可能需要大量数据才能表现良好。数据稀缺的一个解决方案是在仿真中生成数据，但生成足够高质量的数据以避免Sim2Real差距需要大量的计算工作。在这项工作中，我们介绍了一种基于单个人类示范生成力导向数据的框架，并展示了与顺应性策略耦合如何提高从合成数据中学习的视知觉运动策略的性能。我们在实际机器人任务上验证了这种方法，包括非抓取积木翻转和双臂物体搬运，其中学习到的策略表现出可靠的动力接触维持和对新条件的适应。项目网站：this https URL 

---
# Team Xiaomi EV-AD VLA: Caption-Guided Retrieval System for Cross-Modal Drone Navigation - Technical Report for IROS 2025 RoboSense Challenge Track 4 

**Title (ZH)**: 小米团队EV-AD VLA：基于Caption引导的跨模态无人机导航检索系统 - IROS 2025 RoboSense挑战赛赛道4技术报告 

**Authors**: Lingfeng Zhang, Erjia Xiao, Yuchen Zhang, Haoxiang Fu, Ruibin Hu, Yanbiao Ma, Wenbo Ding, Long Chen, Hangjun Ye, Xiaoshuai Hao  

**Link**: [PDF](https://arxiv.org/pdf/2510.02728)  

**Abstract**: Cross-modal drone navigation remains a challenging task in robotics, requiring efficient retrieval of relevant images from large-scale databases based on natural language descriptions. The RoboSense 2025 Track 4 challenge addresses this challenge, focusing on robust, natural language-guided cross-view image retrieval across multiple platforms (drones, satellites, and ground cameras). Current baseline methods, while effective for initial retrieval, often struggle to achieve fine-grained semantic matching between text queries and visual content, especially in complex aerial scenes. To address this challenge, we propose a two-stage retrieval refinement method: Caption-Guided Retrieval System (CGRS) that enhances the baseline coarse ranking through intelligent reranking. Our method first leverages a baseline model to obtain an initial coarse ranking of the top 20 most relevant images for each query. We then use Vision-Language-Model (VLM) to generate detailed captions for these candidate images, capturing rich semantic descriptions of their visual content. These generated captions are then used in a multimodal similarity computation framework to perform fine-grained reranking of the original text query, effectively building a semantic bridge between the visual content and natural language descriptions. Our approach significantly improves upon the baseline, achieving a consistent 5\% improvement across all key metrics (Recall@1, Recall@5, and Recall@10). Our approach win TOP-2 in the challenge, demonstrating the practical value of our semantic refinement strategy in real-world robotic navigation scenarios. 

**Abstract (ZH)**: RoboSense 2025Track 4挑战：基于自然语言引导的多平台跨视图图像检索 

---
# A $1000\times$ Faster LLM-enhanced Algorithm For Path Planning in Large-scale Grid Maps 

**Title (ZH)**: 一种基于大尺度网格地图的快速路径规划算法，速度提升1000倍，利用LLM增强 

**Authors**: Junlin Zeng, Xin Zhang, Xiang Zhao, Yan Pan  

**Link**: [PDF](https://arxiv.org/pdf/2510.02716)  

**Abstract**: Path planning in grid maps, arising from various applications, has garnered significant attention. Existing methods, such as A*, Dijkstra, and their variants, work well for small-scale maps but fail to address large-scale ones due to high search time and memory consumption. Recently, Large Language Models (LLMs) have shown remarkable performance in path planning but still suffer from spatial illusion and poor planning performance. Among all the works, LLM-A* \cite{meng2024llm} leverages LLM to generate a series of waypoints and then uses A* to plan the paths between the neighboring waypoints. In this way, the complete path is constructed. However, LLM-A* still suffers from high computational time for large-scale maps. To fill this gap, we conducted a deep investigation into LLM-A* and found its bottleneck, resulting in limited performance. Accordingly, we design an innovative LLM-enhanced algorithm, abbr. as iLLM-A*. iLLM-A* includes 3 carefully designed mechanisms, including the optimization of A*, an incremental learning method for LLM to generate high-quality waypoints, and the selection of the appropriate waypoints for A* for path planning. Finally, a comprehensive evaluation on various grid maps shows that, compared with LLM-A*, iLLM-A* \textbf{1) achieves more than $1000\times$ speedup on average, and up to $2349.5\times$ speedup in the extreme case, 2) saves up to $58.6\%$ of the memory cost, 3) achieves both obviously shorter path length and lower path length standard deviation.} 

**Abstract (ZH)**: 基于网格地图的路径规划由于来自各种应用的需求已引起广泛关注。现有方法如A*、迪杰斯特拉及其变种在小规模地图上表现良好，但在大规模地图上因搜索时间和内存消耗高而失效。最近，大型语言模型（LLMs）在路径规划方面显示出卓越的性能，但仍存在空间幻觉和规划性能差的问题。在所有工作中，LLM-A* [meng2024llm] 利用LLM生成一系列航点，然后使用A*在相邻航点之间规划路径。通过这种方式构造完整的路径。然而，LLM-A*在大规模地图上的计算时间仍然很高。为解决这一问题，我们深入研究了LLM-A*并找到了其瓶颈，从而限制了其性能。据此，我们设计了一个创新的LLM增强算法，简称iLLM-A*。iLLM-A*包括3个精心设计的机制，包括A*的优化、增量学习方法生成高质量航点以及为路径规划选择合适的航点。最后，在各种网格地图上的综合评估显示，与LLM-A*相比，iLLM-A* 1) 在平均情况下实现了超过1000倍的加速，在极端情况下达到2349.5倍的加速；2) 最多节省了58.6%的内存成本；3) 实现了明显更短的路径长度和更低的路径长度标准差。 

---
# A Trajectory Generator for High-Density Traffic and Diverse Agent-Interaction Scenarios 

**Title (ZH)**: 高密度交通及多样交互场景的轨迹生成器 

**Authors**: Ruining Yang, Yi Xu, Yixiao Chen, Yun Fu, Lili Su  

**Link**: [PDF](https://arxiv.org/pdf/2510.02627)  

**Abstract**: Accurate trajectory prediction is fundamental to autonomous driving, as it underpins safe motion planning and collision avoidance in complex environments. However, existing benchmark datasets suffer from a pronounced long-tail distribution problem, with most samples drawn from low-density scenarios and simple straight-driving behaviors. This underrepresentation of high-density scenarios and safety critical maneuvers such as lane changes, overtaking and turning is an obstacle to model generalization and leads to overly optimistic evaluations. To address these challenges, we propose a novel trajectory generation framework that simultaneously enhances scenarios density and enriches behavioral diversity. Specifically, our approach converts continuous road environments into a structured grid representation that supports fine-grained path planning, explicit conflict detection, and multi-agent coordination. Built upon this representation, we introduce behavior-aware generation mechanisms that combine rule-based decision triggers with Frenet-based trajectory smoothing and dynamic feasibility constraints. This design allows us to synthesize realistic high-density scenarios and rare behaviors with complex interactions that are often missing in real data. Extensive experiments on the large-scale Argoverse 1 and Argoverse 2 datasets demonstrate that our method significantly improves both agent density and behavior diversity, while preserving motion realism and scenario-level safety. Our synthetic data also benefits downstream trajectory prediction models and enhances performance in challenging high-density scenarios. 

**Abstract (ZH)**: 准确的轨迹预测是自动驾驶的核心，因为它支撑着在复杂环境中安全运动规划和碰撞避免。然而，现有的基准数据集存在明显的长尾分布问题，大多数样本来自低密度场景和简单的直行行为。这种高密度场景和安全性关键操作（如变道、超车和转弯）的不足表示在数据中，阻碍了模型的泛化能力，并导致了过于乐观的评估。为了解决这些问题，我们提出了一种新的轨迹生成框架，同时增强了场景密度并丰富了行为多样性。具体来说，我们的方法将连续的道路环境转换为支持精细路径规划、明确冲突检测和多智能体协调的结构化网格表示。在此表示基础上，我们引入了行为感知的生成机制，将基于规则的决策触发与Frenet轨迹平滑和动态可行性约束相结合。这一设计使我们能够合成具有复杂交互的真实高密度场景和罕见行为，这些行为在真实数据中往往缺失。大规模Argoverse 1和Argoverse 2数据集上的实验结果显示，我们的方法在提高代理密度和行为多样性的同时，保持了运动的真实性和场景级的安全性。我们合成的数据也改善了下游轨迹预测模型的表现，并在具有挑战性的高密度场景中提高了性能。 

---
# Multi-robot Rigid Formation Navigation via Synchronous Motion and Discrete-time Communication-Control Optimization 

**Title (ZH)**: 基于同步运动与离散时间通信-控制优化的多机器人刚性编队导航 

**Authors**: Qun Yang, Soung Chang Liew  

**Link**: [PDF](https://arxiv.org/pdf/2510.02624)  

**Abstract**: Rigid-formation navigation of multiple robots is essential for applications such as cooperative transportation. This process involves a team of collaborative robots maintaining a predefined geometric configuration, such as a square, while in motion. For untethered collaborative motion, inter-robot communication must be conducted through a wireless network. Notably, few existing works offer a comprehensive solution for multi-robot formation navigation executable on microprocessor platforms via wireless networks, particularly for formations that must traverse complex curvilinear paths. To address this gap, we introduce a novel "hold-and-hit" communication-control framework designed to work seamlessly with the widely-used Robotic Operating System (ROS) platform. The hold-and-hit framework synchronizes robot movements in a manner robust against wireless network delays and packet loss. It operates over discrete-time communication-control cycles, making it suitable for implementation on contemporary microprocessors. Complementary to hold-and-hit, we propose an intra-cycle optimization approach that enables rigid formations to closely follow desired curvilinear paths, even under the nonholonomic movement constraints inherent to most vehicular robots. The combination of hold-and-hit and intra-cycle optimization ensures precise and reliable navigation even in challenging scenarios. Simulations in a virtual environment demonstrate the superiority of our method in maintaining a four-robot square formation along an S-shaped path, outperforming two existing approaches. Furthermore, real-world experiments validate the effectiveness of our framework: the robots maintained an inter-distance error within $\pm 0.069m$ and an inter-angular orientation error within $\pm19.15^{\circ}$ while navigating along an S-shaped path at a fixed linear velocity of $0.1 m/s$. 

**Abstract (ZH)**: 多机器人刚性编队无线网络导航方法研究 

---
# Reachable Predictive Control: A Novel Control Algorithm for Nonlinear Systems with Unknown Dynamics and its Practical Applications 

**Title (ZH)**: 可达预测控制：一种用于未知动态非线性系统的新型控制算法及其实际应用 

**Authors**: Taha Shafa, Yiming Meng, Melkior Ornik  

**Link**: [PDF](https://arxiv.org/pdf/2510.02623)  

**Abstract**: This paper proposes an algorithm capable of driving a system to follow a piecewise linear trajectory without prior knowledge of the system dynamics. Motivated by a critical failure scenario in which a system can experience an abrupt change in its dynamics, we demonstrate that it is possible to follow a set of waypoints comprised of states analytically proven to be reachable despite not knowing the system dynamics. The proposed algorithm first applies small perturbations to locally learn the system dynamics around the current state, then computes the set of states that are provably reachable using the locally learned dynamics and their corresponding maximum growth-rate bounds, and finally synthesizes a control action that navigates the system to a guaranteed reachable state. 

**Abstract (ZH)**: 本文提出了一种算法，能够在未知系统动力学的情况下，驱动系统跟随分段线性轨迹。受系统动力学突然改变的关键故障场景启发，我们证明即使不知道系统动力学，也可以跟随由可证可到达状态组成的航点集。该提议的算法首先通过对当前状态进行小扰动来局部学习系统动力学，然后使用局部学习到的动力学及其相应的最大增长速率界来计算可证可到达的状态集，最后合成一种控制动作，使系统导航到一个有保证可到达的状态。 

---
# RSV-SLAM: Toward Real-Time Semantic Visual SLAM in Indoor Dynamic Environments 

**Title (ZH)**: RSV-SLAM: 面向室内动态环境的实时语义视觉SLAM 

**Authors**: Mobin Habibpour, Alireza Nemati, Ali Meghdari, Alireza Taheri, Shima Nazari  

**Link**: [PDF](https://arxiv.org/pdf/2510.02616)  

**Abstract**: Simultaneous Localization and Mapping (SLAM) plays an important role in many robotics fields, including social robots. Many of the available visual SLAM methods are based on the assumption of a static world and struggle in dynamic environments. In the current study, we introduce a real-time semantic RGBD SLAM approach designed specifically for dynamic environments. Our proposed system can effectively detect moving objects and maintain a static map to ensure robust camera tracking. The key innovation of our approach is the incorporation of deep learning-based semantic information into SLAM systems to mitigate the impact of dynamic objects. Additionally, we enhance the semantic segmentation process by integrating an Extended Kalman filter to identify dynamic objects that may be temporarily idle. We have also implemented a generative network to fill in the missing regions of input images belonging to dynamic objects. This highly modular framework has been implemented on the ROS platform and can achieve around 22 fps on a GTX1080. Benchmarking the developed pipeline on dynamic sequences from the TUM dataset suggests that the proposed approach delivers competitive localization error in comparison with the state-of-the-art methods, all while operating in near real-time. The source code is publicly available. 

**Abstract (ZH)**: 实时语义RGBD SLAM方法在动态环境中的应用 

---
# UMI-on-Air: Embodiment-Aware Guidance for Embodiment-Agnostic Visuomotor Policies 

**Title (ZH)**: UMI-on-Air: 体态感知指导下的体态无关视觉运动策略 

**Authors**: Harsh Gupta, Xiaofeng Guo, Huy Ha, Chuer Pan, Muqing Cao, Dongjae Lee, Sebastian Sherer, Shuran Song, Guanya Shi  

**Link**: [PDF](https://arxiv.org/pdf/2510.02614)  

**Abstract**: We introduce UMI-on-Air, a framework for embodiment-aware deployment of embodiment-agnostic manipulation policies. Our approach leverages diverse, unconstrained human demonstrations collected with a handheld gripper (UMI) to train generalizable visuomotor policies. A central challenge in transferring these policies to constrained robotic embodiments-such as aerial manipulators-is the mismatch in control and robot dynamics, which often leads to out-of-distribution behaviors and poor execution. To address this, we propose Embodiment-Aware Diffusion Policy (EADP), which couples a high-level UMI policy with a low-level embodiment-specific controller at inference time. By integrating gradient feedback from the controller's tracking cost into the diffusion sampling process, our method steers trajectory generation towards dynamically feasible modes tailored to the deployment embodiment. This enables plug-and-play, embodiment-aware trajectory adaptation at test time. We validate our approach on multiple long-horizon and high-precision aerial manipulation tasks, showing improved success rates, efficiency, and robustness under disturbances compared to unguided diffusion baselines. Finally, we demonstrate deployment in previously unseen environments, using UMI demonstrations collected in the wild, highlighting a practical pathway for scaling generalizable manipulation skills across diverse-and even highly constrained-embodiments. All code, data, and checkpoints will be publicly released after acceptance. Result videos can be found at this http URL. 

**Abstract (ZH)**: UMI-on-Air：一种基于体态感知的多功能操作策略部署框架 

---
# SubSense: VR-Haptic and Motor Feedback for Immersive Control in Subsea Telerobotics 

**Title (ZH)**: SubSense: 海底远程操控中的VR触觉与运动反馈沉浸式控制 

**Authors**: Ruo Chen, David Blow, Adnan Abdullah, Md Jahidul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2510.02594)  

**Abstract**: This paper investigates the integration of haptic feedback and virtual reality (VR) control interfaces to enhance teleoperation and telemanipulation of underwater ROVs (remotely operated vehicles). Traditional ROV teleoperation relies on low-resolution 2D camera feeds and lacks immersive and sensory feedback, which diminishes situational awareness in complex subsea environments. We propose SubSense -- a novel VR-Haptic framework incorporating a non-invasive feedback interface to an otherwise 1-DOF (degree of freedom) manipulator, which is paired with the teleoperator's glove to provide haptic feedback and grasp status. Additionally, our framework integrates end-to-end software for managing control inputs and displaying immersive camera views through a VR platform. We validate the system through comprehensive experiments and user studies, demonstrating its effectiveness over conventional teleoperation interfaces, particularly for delicate manipulation tasks. Our results highlight the potential of multisensory feedback in immersive virtual environments to significantly improve remote situational awareness and mission performance, offering more intuitive and accessible ROV operations in the field. 

**Abstract (ZH)**: 本研究探讨了将触感反馈与虚拟现实（VR）控制接口集成以增强遥控潜水器（ROV）的远程操作和远程操控。传统ROV远程操作依赖于低分辨率的2D摄像头馈送，并缺乏沉浸式和感官反馈，这在复杂的海底环境中降低了态势感知能力。我们提出了一种名为SubSense的新颖VR-触感框架，该框架结合了一个非侵入性反馈接口，用于配对手动执行器，该执行器与遥控操作员的手套配对，提供触感反馈和抓取状态。此外，我们的框架通过VR平台集成了端到端的软件，用于管理和显示沉浸式摄像机视图。通过全面的实验和用户研究，我们验证了该系统的有效性和比传统远程操作界面的优势，特别是在精细操控任务方面。研究结果表明，多感官反馈在沉浸式虚拟环境中具有显著增强远程态势感知和任务性能的潜力，为现场遥控潜水器操作提供更直观和易用的服务。 

---
# Efficient Optimal Path Planning in Dynamic Environments Using Koopman MPC 

**Title (ZH)**: 使用库曼 MPC 在动态环境中进行高效最优路径规划 

**Authors**: Mohammad Abtahi, Navid Mojahed, Shima Nazari  

**Link**: [PDF](https://arxiv.org/pdf/2510.02584)  

**Abstract**: This paper presents a data-driven model predictive control framework for mobile robots navigating in dynamic environments, leveraging Koopman operator theory. Unlike the conventional Koopman-based approaches that focus on the linearization of system dynamics only, our work focuses on finding a global linear representation for the optimal path planning problem that includes both the nonlinear robot dynamics and collision-avoidance constraints. We deploy extended dynamic mode decomposition to identify linear and bilinear Koopman realizations from input-state data. Our open-loop analysis demonstrates that only the bilinear Koopman model can accurately capture nonlinear state-input couplings and quadratic terms essential for collision avoidance, whereas linear realizations fail to do so. We formulate a quadratic program for the robot path planning in the presence of moving obstacles in the lifted space and determine the optimal robot action in an MPC framework. Our approach is capable of finding the safe optimal action 320 times faster than a nonlinear MPC counterpart that solves the path planning problem in the original state space. Our work highlights the potential of bilinear Koopman realizations for linearization of highly nonlinear optimal control problems subject to nonlinear state and input constraints to achieve computational efficiency similar to linear problems. 

**Abstract (ZH)**: 基于Koopman算子理论的数据驱动模型预测控制框架：移动机器人在动态环境中的路径规划 

---
# A Recipe for Efficient Sim-to-Real Transfer in Manipulation with Online Imitation-Pretrained World Models 

**Title (ZH)**: 高效的 manipulatation 模拟到现实转移食谱：基于在线模仿预训练世界模型的方法 

**Authors**: Yilin Wang, Shangzhe Li, Haoyi Niu, Zhiao Huang, Weitong Zhang, Hao Su  

**Link**: [PDF](https://arxiv.org/pdf/2510.02538)  

**Abstract**: We are interested in solving the problem of imitation learning with a limited amount of real-world expert data. Existing offline imitation methods often struggle with poor data coverage and severe performance degradation. We propose a solution that leverages robot simulators to achieve online imitation learning. Our sim-to-real framework is based on world models and combines online imitation pretraining with offline finetuning. By leveraging online interactions, our approach alleviates the data coverage limitations of offline methods, leading to improved robustness and reduced performance degradation during finetuning. It also enhances generalization during domain transfer. Our empirical results demonstrate its effectiveness, improving success rates by at least 31.7% in sim-to-sim transfer and 23.3% in sim-to-real transfer over existing offline imitation learning baselines. 

**Abstract (ZH)**: 我们感兴趣的是在线上使用有限的真实世界专家数据解决模拟学习问题。现有离线模拟方法往往面临数据覆盖不足和严重性能退化的挑战。我们提出了一种利用机器人模拟器实现在线模拟学习的解决方案。我们的在线转现实框架基于世界模型，并结合了在线模拟预训练和离线微调。通过利用在线交互，我们的方法缓解了离线方法的数据覆盖限制，从而在微调过程中提高了鲁棒性并减少了性能退化。它还在领域迁移中增强了泛化能力。我们的实证结果证明了其有效性，在模拟到模拟转移中将成功率至少提高31.7%，在模拟到现实转移中将成功率提高23.3%，超越了现有的离线模拟学习基线。 

---
# U-LAG: Uncertainty-Aware, Lag-Adaptive Goal Retargeting for Robotic Manipulation 

**Title (ZH)**: U-LAG：具有不确定性意识和lag自适应的目标重定位于机器人操作中 

**Authors**: Anamika J H, Anujith Muraleedharan  

**Link**: [PDF](https://arxiv.org/pdf/2510.02526)  

**Abstract**: Robots manipulating in changing environments must act on percepts that are late, noisy, or stale. We present U-LAG, a mid-execution goal-retargeting layer that leaves the low-level controller unchanged while re-aiming task goals (pre-contact, contact, post) as new observations arrive. Unlike motion retargeting or generic visual servoing, U-LAG treats in-flight goal re-aiming as a first-class, pluggable module between perception and control. Our main technical contribution is UAR-PF, an uncertainty-aware retargeter that maintains a distribution over object pose under sensing lag and selects goals that maximize expected progress. We instantiate a reproducible Shift x Lag stress test in PyBullet/PandaGym for pick, push, stacking, and peg insertion, where the object undergoes abrupt in-plane shifts while synthetic perception lag is injected during approach. Across 0-10 cm shifts and 0-400 ms lags, UAR-PF and ICP degrade gracefully relative to a no-retarget baseline, achieving higher success with modest end-effector travel and fewer aborts; simple operational safeguards further improve stability. Contributions: (1) UAR-PF for lag-adaptive, uncertainty-aware goal retargeting; (2) a pluggable retargeting interface; and (3) a reproducible Shift x Lag benchmark with evaluation on pick, push, stacking, and peg insertion. 

**Abstract (ZH)**: robots在变化环境中的操作必须基于延迟、噪声或过时的感知信息。我们提出U-LAG，在保持低级控制器不变的情况下，重新瞄准任务目标（接触前、接触中、接触后），并随新观察信息的到达而调整。不同于运动重定或通用视觉伺服，U-LAG 将飞行中的目标重定作为感知与控制之间可插拔的一等模块。我们的主要技术贡献是UAR-PF，一种具有不确定性意识的重定器，在感知滞后下维护物体姿态分布，并选择最大化预期进度的目标。我们在PyBullet/PandaGym中实例化了一个可重现的Shift x Lag压力测试，用于捡拾、推移、堆叠和针插入任务，其中物体在接近过程中注入合成感知滞后，并经历平面内的突然变化。在0-10 cm的位移和0-400 ms的滞后下，与无重定基线相比，UAR-PF和ICP表现得更为稳健，具有较小的末端执行器移动和较少的终止事件；简单的操作保护措施进一步提高了稳定性。贡献包括：(1) UAR-PF用于滞后适应性和不确定性意识的目标重定；(2) 一种可插拔的重定接口；以及(3) 一个具有捡拾、推移、堆叠和针插入评估的可重现Shift x Lag基准测试。 

---
# SIMSplat: Predictive Driving Scene Editing with Language-aligned 4D Gaussian Splatting 

**Title (ZH)**: SIMSplat: 基于语言对齐的4D高斯点云的预测性驾驶场景编辑 

**Authors**: Sung-Yeon Park, Adam Lee, Juanwu Lu, Can Cui, Luyang Jiang, Rohit Gupta, Kyungtae Han, Ahmadreza Moradipari, Ziran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02469)  

**Abstract**: Driving scene manipulation with sensor data is emerging as a promising alternative to traditional virtual driving simulators. However, existing frameworks struggle to generate realistic scenarios efficiently due to limited editing capabilities. To address these challenges, we present SIMSplat, a predictive driving scene editor with language-aligned Gaussian splatting. As a language-controlled editor, SIMSplat enables intuitive manipulation using natural language prompts. By aligning language with Gaussian-reconstructed scenes, it further supports direct querying of road objects, allowing precise and flexible editing. Our method provides detailed object-level editing, including adding new objects and modifying the trajectories of both vehicles and pedestrians, while also incorporating predictive path refinement through multi-agent motion prediction to generate realistic interactions among all agents in the scene. Experiments on the Waymo dataset demonstrate SIMSplat's extensive editing capabilities and adaptability across a wide range of scenarios. Project page: this https URL 

**Abstract (ZH)**: 基于传感器数据的驾驶场景操纵正在成为传统虚拟驾驶模拟器的有前景的替代方案。然而，现有框架由于编辑能力有限，难以高效生成现实主义场景。为应对这些挑战，我们提出了SIMSplat，一种基于语言对齐的高斯点云预测驾驶场景编辑器。作为语言控制的编辑器，SIMSplat支持使用自然语言提示进行直观的操作。通过将语言与高斯重建的场景对齐，它还可以直接查询道路物体，实现精确和灵活的编辑。我们的方法提供了详细的对象级编辑能力，包括添加新物体和修改车辆和行人的轨迹，同时还通过多智能体运动预测进行预测路径细化，生成场景中所有智能体之间的现实主义交互。基于Waymo数据集的实验展示了SIMSplat广泛的编辑能力和在多种场景下的适应性。项目页面：this https URL。 

---
# ERUPT: An Open Toolkit for Interfacing with Robot Motion Planners in Extended Reality 

**Title (ZH)**: ERUPT：扩展现实环境中与机器人运动规划器接口的开源工具包 

**Authors**: Isaac Ngui, Courtney McBeth, André Santos, Grace He, Katherine J. Mimnaugh, James D. Motes, Luciano Soares, Marco Morales, Nancy M. Amato  

**Link**: [PDF](https://arxiv.org/pdf/2510.02464)  

**Abstract**: We propose the Extended Reality Universal Planning Toolkit (ERUPT), an extended reality (XR) system for interactive motion planning. Our system allows users to create and dy- namically reconfigure environments while they plan robot paths. In immersive three-dimensional XR environments, users gain a greater spatial understanding. XR also unlocks a broader range of natural interaction capabilities, allowing users to grab and adjust objects in the environment similarly to the real world, rather than using a mouse and keyboard with the scene projected onto a two-dimensional computer screen. Our system integrates with MoveIt, a manipulation planning framework, allowing users to send motion planning requests and visualize the resulting robot paths in virtual or augmented reality. We provide a broad range of interaction modalities, allowing users to modify objects in the environment and interact with a virtual robot. Our system allows operators to visualize robot motions, ensuring desired behavior as it moves throughout the environment, without risk of collisions within a virtual space, and to then deploy planned paths on physical robots in the real world. 

**Abstract (ZH)**: 扩展现实通用规划工具包（ERUPT）：一种交互运动规划的扩展现实系统 

---
# Improving Cooperation in Collaborative Embodied AI 

**Title (ZH)**: 提高协作型实体AI中的合作水平 

**Authors**: Hima Jacob Leven Suprabha, Laxmi Nag Laxminarayan Nagesh, Ajith Nair, Alvin Reuben Amal Selvaster, Ayan Khan, Raghuram Damarla, Sanju Hannah Samuel, Sreenithi Saravana Perumal, Titouan Puech, Venkataramireddy Marella, Vishal Sonar, Alessandro Suglia, Oliver Lemon  

**Link**: [PDF](https://arxiv.org/pdf/2510.03153)  

**Abstract**: The integration of Large Language Models (LLMs) into multiagent systems has opened new possibilities for collaborative reasoning and cooperation with AI agents. This paper explores different prompting methods and evaluates their effectiveness in enhancing agent collaborative behaviour and decision-making. We enhance CoELA, a framework designed for building Collaborative Embodied Agents that leverage LLMs for multi-agent communication, reasoning, and task coordination in shared virtual spaces. Through systematic experimentation, we examine different LLMs and prompt engineering strategies to identify optimised combinations that maximise collaboration performance. Furthermore, we extend our research by integrating speech capabilities, enabling seamless collaborative voice-based interactions. Our findings highlight the effectiveness of prompt optimisation in enhancing collaborative agent performance; for example, our best combination improved the efficiency of the system running with Gemma3 by 22% compared to the original CoELA system. In addition, the speech integration provides a more engaging user interface for iterative system development and demonstrations. 

**Abstract (ZH)**: 大型语言模型（LLMs）集成到多智能体系统中的进展为与AI代理的协同推理和合作开辟了新可能性。本文探讨了不同的提示方法，并评估了它们在增强智能体协同行为和决策能力方面的有效性。我们改进了CoELA框架，该框架旨在构建利用LLMs进行多智能体通信、推理和任务协调的协作体感知智能体，在共享虚拟空间中进行构建。通过系统的实验，我们检查了不同的LLMs和提示工程策略，以确定最大化合作性能的最佳组合。此外，我们通过整合语音能力，使无缝协作语音交互成为可能。我们的研究发现提示优化在提升协作智能体性能方面的有效性；例如，我们最佳组合在与Gemma3协同工作的系统中提高了22%的效率，与原始的CoELA系统相比。此外，语音集成还为迭代系统开发和演示提供了更具吸引力的用户界面。 

---
# Mask2IV: Interaction-Centric Video Generation via Mask Trajectories 

**Title (ZH)**: Mask2IV: 基于交互的视频生成通过掩码轨迹 

**Authors**: Gen Li, Bo Zhao, Jianfei Yang, Laura Sevilla-Lara  

**Link**: [PDF](https://arxiv.org/pdf/2510.03135)  

**Abstract**: Generating interaction-centric videos, such as those depicting humans or robots interacting with objects, is crucial for embodied intelligence, as they provide rich and diverse visual priors for robot learning, manipulation policy training, and affordance reasoning. However, existing methods often struggle to model such complex and dynamic interactions. While recent studies show that masks can serve as effective control signals and enhance generation quality, obtaining dense and precise mask annotations remains a major challenge for real-world use. To overcome this limitation, we introduce Mask2IV, a novel framework specifically designed for interaction-centric video generation. It adopts a decoupled two-stage pipeline that first predicts plausible motion trajectories for both actor and object, then generates a video conditioned on these trajectories. This design eliminates the need for dense mask inputs from users while preserving the flexibility to manipulate the interaction process. Furthermore, Mask2IV supports versatile and intuitive control, allowing users to specify the target object of interaction and guide the motion trajectory through action descriptions or spatial position cues. To support systematic training and evaluation, we curate two benchmarks covering diverse action and object categories across both human-object interaction and robotic manipulation scenarios. Extensive experiments demonstrate that our method achieves superior visual realism and controllability compared to existing baselines. 

**Abstract (ZH)**: Mask2IV：一种用于交互中心视频生成的新型框架 

---
# Geometry Meets Vision: Revisiting Pretrained Semantics in Distilled Fields 

**Title (ZH)**: 几何与视觉相遇：重返蒸馏字段中的预训练语义 

**Authors**: Zhiting Mei, Ola Shorinwa, Anirudha Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2510.03104)  

**Abstract**: Semantic distillation in radiance fields has spurred significant advances in open-vocabulary robot policies, e.g., in manipulation and navigation, founded on pretrained semantics from large vision models. While prior work has demonstrated the effectiveness of visual-only semantic features (e.g., DINO and CLIP) in Gaussian Splatting and neural radiance fields, the potential benefit of geometry-grounding in distilled fields remains an open question. In principle, visual-geometry features seem very promising for spatial tasks such as pose estimation, prompting the question: Do geometry-grounded semantic features offer an edge in distilled fields? Specifically, we ask three critical questions: First, does spatial-grounding produce higher-fidelity geometry-aware semantic features? We find that image features from geometry-grounded backbones contain finer structural details compared to their counterparts. Secondly, does geometry-grounding improve semantic object localization? We observe no significant difference in this task. Thirdly, does geometry-grounding enable higher-accuracy radiance field inversion? Given the limitations of prior work and their lack of semantics integration, we propose a novel framework SPINE for inverting radiance fields without an initial guess, consisting of two core components: coarse inversion using distilled semantics, and fine inversion using photometric-based optimization. Surprisingly, we find that the pose estimation accuracy decreases with geometry-grounded features. Our results suggest that visual-only features offer greater versatility for a broader range of downstream tasks, although geometry-grounded features contain more geometric detail. Notably, our findings underscore the necessity of future research on effective strategies for geometry-grounding that augment the versatility and performance of pretrained semantic features. 

**Abstract (ZH)**: 在辐射场中的语义蒸馏促进了基于大型视觉模型预训练语义的开放词汇机器人策略的重大进展，如操作和导航方面。虽然前期工作表明仅视觉语义特征（如DINO和CLIP）在Gaussian Splatting和神经辐射场中的有效性，但在蒸馏场中几何约束潜在的好处仍然是一个开放问题。鉴于视觉-几何特征在空间任务如姿态估计方面的前景，我们提出了一个问题：几何约束的语义特征是否在蒸馏场中更具优势？具体而言，我们提出了三个关键问题：首先，空间约束是否能产生更高保真度的几何感知语义特征？我们发现，基于几何约束的骨干网络的图像特征比其对应特征包含更精细的结构细节。其次，几何约束是否能提高语义物体定位的准确性？我们没有观察到显著差异。最后，几何约束是否能提高辐射场反转的准确性？鉴于前期工作的局限性和缺乏语义集成，我们提出了一种新的框架SPINE，用于在没有初始猜测的情况下反转辐射场，该框架由两个核心组件组成：粗略反转使用蒸馏的语义，精细反转使用基于光度的优化。令人惊讶的是，我们发现姿态估计精度随着几何约束特征的增加而降低。我们的结果表明，仅视觉特征在更广泛下游任务中更具灵活性，尽管几何约束特征包含更多的几何细节。值得注意的是，我们的发现突显了未来研究有效几何约束策略的必要性，以增强预训练语义特征的灵活性和性能。 

---
# A Dimension-Decomposed Learning Framework for Online Disturbance Identification in Quadrotor SE(3) Control 

**Title (ZH)**: 基于四旋翼SE(3)控制在线干扰识别的维度分解学习框架 

**Authors**: Tianhua Gao  

**Link**: [PDF](https://arxiv.org/pdf/2510.03100)  

**Abstract**: Quadrotor stability under complex dynamic disturbances and model uncertainties poses significant challenges. One of them remains the underfitting problem in high-dimensional features, which limits the identification capability of current learning-based methods. To address this, we introduce a new perspective: Dimension-Decomposed Learning (DiD-L), from which we develop the Sliced Adaptive-Neuro Mapping (SANM) approach for geometric control. Specifically, the high-dimensional mapping for identification is axially ``sliced" into multiple low-dimensional submappings (``slices"). In this way, the complex high-dimensional problem is decomposed into a set of simple low-dimensional tasks addressed by shallow neural networks and adaptive laws. These neural networks and adaptive laws are updated online via Lyapunov-based adaptation without any pre-training or persistent excitation (PE) condition. To enhance the interpretability of the proposed approach, we prove that the full-state closed-loop system exhibits arbitrarily close to exponential stability despite multi-dimensional time-varying disturbances and model uncertainties. This result is novel as it demonstrates exponential convergence without requiring pre-training for unknown disturbances and specific knowledge of the model. 

**Abstract (ZH)**: Quadrotor 几何控制下的复杂动态扰动和模型不确定性下的欠拟合问题提出了显著挑战：基于维度分解学习的新视角及 Sliced 自适应神经映射方法 

---
# VERNIER: an open-source software pushing marker pose estimation down to the micrometer and nanometer scales 

**Title (ZH)**: VERNIER：一款将标记物姿态估计精度推向微米和纳米尺度的开源软件 

**Authors**: Patrick Sandoz, Antoine N. André, Guillaume J. Laurent  

**Link**: [PDF](https://arxiv.org/pdf/2510.02791)  

**Abstract**: Pose estimation is still a challenge at the small scales. Few solutions exist to capture the 6 degrees of freedom of an object with nanometric and microradians resolutions over relatively large ranges. Over the years, we have proposed several fiducial marker and pattern designs to achieve reliable performance for various microscopy applications. Centimeter ranges are possible using pattern encoding methods, while nanometer resolutions can be achieved using phase processing of the periodic frames. This paper presents VERNIER, an open source phase processing software designed to provide fast and reliable pose measurement based on pseudo-periodic patterns. Thanks to a phase-based local thresholding algorithm, the software has proven to be particularly robust to noise, defocus and occlusion. The successive steps of the phase processing are presented, as well as the different types of patterns that address different application needs. The implementation procedure is illustrated with synthetic and experimental images. Finally, guidelines are given for selecting the appropriate pattern design and microscope magnification lenses as a function of the desired performance. 

**Abstract (ZH)**: 小尺度下姿态估计仍是一项挑战。很少有解决方案能以纳米级和微弧度分辨率，在相对大范围内捕获对象的6自由度。多年来，我们提出了几种标记和模式设计，以实现各种显微镜应用的可靠性能。利用模式编码方法可实现厘米范围，而利用周期性帧的相位处理可实现纳米级分辨率。本文介绍了VERNIER，一款基于伪周期性模式设计的开源相位处理软件，提供基于相位的快速可靠姿态测量。得益于基于相位的局部阈值算法，该软件对噪声、焦距偏移和遮挡具有特别的鲁棒性。相位处理的各个步骤以及适应不同应用需求的不同类型模式被呈现，同时用合成和实验图像说明了其实现过程。最后，给出了根据期望的性能选择合适的模式设计和显微镜放大镜的指南。 

---
# Periodic Event-Triggered Prescribed Time Control of Euler-Lagrange Systems under State and Input Constraints 

**Title (ZH)**: 基于状态和输入约束的欧拉-拉格朗日系统在预定时间内的周期事件触发控制 

**Authors**: Chidre Shravista Kashyap, Karnan A, Pushpak Jagtap, Jishnu Keshavan  

**Link**: [PDF](https://arxiv.org/pdf/2510.02769)  

**Abstract**: This article proposes a periodic event-triggered adaptive barrier control policy for the trajectory tracking problem of perturbed Euler-Lagrangian systems with state, input, and temporal (SIT) constraints. In particular, an approximation-free adaptive-barrier control architecture is designed to ensure prescribed-time convergence of the tracking error to a prescribed bound while rejecting exogenous disturbances. In contrast to existing approaches that necessitate continuous real-time control action, the proposed controller generates event-based updates through periodic evaluation of the triggering condition. Additionally, we derive an upper bound on the monitoring period by analysing the performance degradation of the filtered tracking error to facilitate periodic evaluation of the event-triggered strategy. To this end, a time-varying threshold function is considered in the triggering mechanism to reduce the number of triggers during the transient phase of system behaviour. Notably, the proposed design avoids Zeno behaviour and precludes the need for continuous monitoring of the triggering condition. A simulation and experimental study is undertaken to demonstrate the efficacy of the proposed control scheme. 

**Abstract (ZH)**: 本文提出了一种周期事件触发自适应障碍控制器策略，用于受状态、输入和时间（SIT）约束的受扰欧拉-拉格朗日系统轨迹跟踪问题。特别地，设计了一种无近似自适应障碍控制器架构，以确保跟踪误差在预定时间内收敛到预定界，并拒绝外部干扰。与现有需要连续实时控制作用的方法不同，所提出的控制器通过周期性评估触发条件来生成事件驱动的更新。此外，通过分析过滤后的跟踪误差性能退化来推导出触发策略的监测周期的上界，以促进事件驱动策略的周期性评估。在此过程中，在触发机制中考虑了一个时间变阈值函数，以减少系统行为瞬态阶段的触发次数。值得注意的是，所提设计避开了Zeno行为，并消除了对触发条件连续监控的需要。进行了仿真和实验研究以证明所提出控制方案的有效性。 

---
# Conceptualizing and Modeling Communication-Based Cyberattacks on Automated Vehicles 

**Title (ZH)**: 基于通信的对自动车辆的网络攻击的概念构建与建模 

**Authors**: Tianyi Li, Tianyu Liu, Yicheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02364)  

**Abstract**: Adaptive Cruise Control (ACC) is rapidly proliferating across electric vehicles (EVs) and internal combustion engine (ICE) vehicles, enhancing traffic flow while simultaneously expanding the attack surface for communication-based cyberattacks. Because the two powertrains translate control inputs into motion differently, their cyber-resilience remains unquantified. Therefore, we formalize six novel message-level attack vectors and implement them in a ring-road simulation that systematically varies the ACC market penetration rates (MPRs) and the spatial pattern of compromised vehicles. A three-tier risk taxonomy converts disturbance metrics into actionable defense priorities for practitioners. Across all simulation scenarios, EV platoons exhibit lower velocity standard deviation, reduced spacing oscillations, and faster post-attack recovery compared to ICE counterparts, revealing an inherent stability advantage. These findings clarify how controller-to-powertrain coupling influences vulnerability and offer quantitative guidance for the detection and mitigation of attacks in mixed automated traffic. 

**Abstract (ZH)**: 自适应巡航控制（ACC）在电动汽车（EV）和内燃机 vehicle（ICE）车辆中的应用迅速增长，提高了交通流量的同时，也为基于通信的网络攻击扩大了攻击面。由于两种动力系统的控制输入转化为运动的方式不同，其在网络攻击中的抵抗力尚未量化。因此，本文形式化了六种新颖的消息级攻击向量，并在根据不同 ACC 市场渗透率（MPRs）和受攻击车辆的空间模式进行系统性变化的环形道路仿真中实施这些攻击向量。三级风险分类将干扰指标转化为可操作的防御优先级，供实践者参考。在整个仿真场景中，EV 车队在速度标准偏差、间距振荡和后攻击恢复速度方面均优于 ICE 对手，表明存在固有的稳定性优势。这些发现阐明了控制器与动力系统耦合如何影响脆弱性，并为混合自动化交通中的攻击检测与缓解提供了定量指导。 

---
