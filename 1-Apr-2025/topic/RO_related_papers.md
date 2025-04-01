# Pro-Routing: Proactive Routing of Autonomous Multi-Capacity Robots for Pickup-and-Delivery Tasks 

**Title (ZH)**: 促进路由：自主多容量机器人前瞻性的拣取和配送任务路由算法 

**Authors**: Daniel Garces, Stephanie Gil  

**Link**: [PDF](https://arxiv.org/pdf/2503.24325)  

**Abstract**: We consider a multi-robot setting, where we have a fleet of multi-capacity autonomous robots that must service spatially distributed pickup-and-delivery requests with fixed maximum wait times. Requests can be either scheduled ahead of time or they can enter the system in real-time. In this setting, stability for a routing policy is defined as the cost of the policy being uniformly bounded over time. Most previous work either solve the problem offline to theoretically maintain stability or they consider dynamically arriving requests at the expense of the theoretical guarantees on stability. In this paper, we aim to bridge this gap by proposing a novel proactive rollout-based routing framework that adapts to real-time demand while still provably maintaining the stability of the learned routing policy. We derive provable stability guarantees for our method by proposing a fleet sizing algorithm that obtains a sufficiently large fleet that ensures stability by construction. To validate our theoretical results, we consider a case study on real ride requests for Harvard's evening Van System. We also evaluate the performance of our framework using the currently deployed smaller fleet size. In this smaller setup, we compare against the currently deployed routing algorithm, greedy heuristics, and Monte-Carlo-Tree-Search-based algorithms. Our empirical results show that our framework maintains stability when we use the sufficiently large fleet size found in our theoretical results. For the smaller currently deployed fleet size, our method services 6% more requests than the closest baseline while reducing median passenger wait times by 33%. 

**Abstract (ZH)**: 我们考虑一个多机器人环境，其中有一支多容量自主机器人队列，需要服务具有固定最大等待时间的空间分布式取送请求。请求既可以提前调度，也可以实时进入系统。在这种环境中，路由策略的稳定性定义为策略的成本在时间上均匀有界。大多数先前的工作要么在离线状态下求解问题以理论上保持稳定性，要么考虑动态到达的请求，但会牺牲理论上关于稳定性的保证。在本文中，我们通过提出一种新型的前瞻性展开为基础的路由框架来弥补这一差距，该框架能够适应实时需求，同时证明能够维持学习到的路由策略的稳定性。我们通过提出一种车队规模算法来推导我们的方法的可验证稳定性保证，该算法通过构造获得足够大的车队以确保稳定性。为了验证我们的理论结果，我们在一个实际案例研究中考虑哈佛大学夜间车系统的真实乘车请求。我们也使用当前部署的较小车队规模评估我们框架的性能。在较小的配置中，我们将方法与当前部署的路由算法、贪婪启发式算法以及基于蒙特卡洛树搜索的算法进行对比。实验结果显示，当使用我们在理论结果中找到的足够大的车队规模时，我们的框架能够维持稳定性。对于当前部署的较小车队规模，我们的方法比最近的基线算法多服务6%的请求，同时将中位乘客等待时间减少了33%。 

---
# Pseudo-Random UAV Test Generation Using Low-Fidelity Path Simulator 

**Title (ZH)**: 使用低保真路径模拟器的伪随机无人机测试生成 

**Authors**: Anas Shrinah, Kerstin Eder  

**Link**: [PDF](https://arxiv.org/pdf/2503.24172)  

**Abstract**: Simulation-based testing provides a safe and cost-effective environment for verifying the safety of Uncrewed Aerial Vehicles (UAVs). However, simulation can be resource-consuming, especially when High-Fidelity Simulators (HFS) are used. To optimise simulation resources, we propose a pseudo-random test generator that uses a Low-Fidelity Simulator (LFS) to estimate UAV flight paths. This work simplifies the PX4 autopilot HFS to develop a LFS, which operates one order of magnitude faster than the this http URL cases predicted to cause safety violations in the LFS are subsequently validated using the HFS. 

**Abstract (ZH)**: 基于仿真的测试提供了一种安全且经济有效的环境，用于验证无人机(UAV)的安全性。然而，仿真的资源消耗极大，特别是在使用高保真模拟器(HFS)的情况下。为了优化仿真资源，我们提出了一种伪随机测试生成器，该生成器利用低保真模拟器(LFS)估计无人机飞行路径。本工作简化了PX4自动驾驶仪的HFS，开发了一个LFS，后者的速度比HFS快一个数量级。通过LFS预测可能导致安全违规的情况随后在HFS中进行验证。 

---
# Less is More: Contextual Sampling for Nonlinear Data-Enabled Predictive Control 

**Title (ZH)**: 少即是多：上下文采样在非线性数据驱动预测控制中的应用 

**Authors**: Julius Beerwerth, Bassam Alrifaee  

**Link**: [PDF](https://arxiv.org/pdf/2503.23890)  

**Abstract**: Data-enabled Predictive Control (DeePC) is a powerful data-driven approach for predictive control without requiring an explicit system model. However, its high computational cost limits its applicability to real-time robotic systems. For robotic applications such as motion planning and trajectory tracking, real-time control is crucial. Nonlinear DeePC either relies on large datasets or learning the nonlinearities to ensure predictive accuracy, leading to high computational complexity. This work introduces contextual sampling, a novel data selection strategy to handle nonlinearities for DeePC by dynamically selecting the most relevant data at each time step. By reducing the dataset size while preserving prediction accuracy, our method improves computational efficiency, of DeePC for real-time robotic applications. We validate our approach for autonomous vehicle motion planning. For a dataset size of 100 sub-trajectories, Contextual sampling DeePC reduces tracking error by 53.2 % compared to Leverage Score sampling. Additionally, Contextual sampling reduces max computation time by 87.2 % compared to using the full dataset of 491 sub-trajectories while achieving comparable tracking performance. These results highlight the potential of Contextual sampling to enable real-time, data-driven control for robotic systems. 

**Abstract (ZH)**: 基于上下文的数据选择策略增强的DeePC实时机器人应用中的预测控制 

---
# Disambiguate Gripper State in Grasp-Based Tasks: Pseudo-Tactile as Feedback Enables Pure Simulation Learning 

**Title (ZH)**: 基于抓取任务中指尖状态消歧：伪触觉反馈使纯仿真学习成为可能 

**Authors**: Yifei Yang, Lu Chen, Zherui Song, Yenan Chen, Wentao Sun, Zhongxiang Zhou, Rong Xiong, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23835)  

**Abstract**: Grasp-based manipulation tasks are fundamental to robots interacting with their environments, yet gripper state ambiguity significantly reduces the robustness of imitation learning policies for these tasks. Data-driven solutions face the challenge of high real-world data costs, while simulation data, despite its low costs, is limited by the sim-to-real gap. We identify the root cause of gripper state ambiguity as the lack of tactile feedback. To address this, we propose a novel approach employing pseudo-tactile as feedback, inspired by the idea of using a force-controlled gripper as a tactile sensor. This method enhances policy robustness without additional data collection and hardware involvement, while providing a noise-free binary gripper state observation for the policy and thus facilitating pure simulation learning to unleash the power of simulation. Experimental results across three real-world grasp-based tasks demonstrate the necessity, effectiveness, and efficiency of our approach. 

**Abstract (ZH)**: 基于抓取的 manipulation 任务是机器人与环境交互的基础，但由于夹持器状态的不确定性显著降低了这些任务的imitation learning策略的鲁棒性。数据驱动的方法面临现实世界数据成本高的挑战，而模拟数据虽然成本低，但也受限于模拟与现实之间的差距。我们识别夹持器状态不确定性根源为缺乏触觉反馈。为了解决这一问题，我们提出了一种新的方法，采用伪触觉作为反馈，灵感来源于将力控夹持器用作触觉传感器的想法。该方法在不需要额外数据收集和硬件投入的情况下增强策略的鲁棒性，为策略提供无噪声的二元夹持器状态观察，从而促进纯模拟学习，发挥模拟的作用。在三个实际抓取任务上的实验结果证明了该方法的必要性、有效性和效率。 

---
# Design and Experimental Validation of an Autonomous USV for Sensor Fusion-Based Navigation in GNSS-Denied Environments 

**Title (ZH)**: 基于传感器融合导航的自主USV设计与GNSS forbidden环境下的实验验证 

**Authors**: Samuel Cohen-Salmon, Itzik Klein  

**Link**: [PDF](https://arxiv.org/pdf/2503.23445)  

**Abstract**: This paper presents the design, development, and experimental validation of MARVEL, an autonomous unmanned surface vehicle built for real-world testing of sensor fusion-based navigation algorithms in GNSS-denied environments. MARVEL was developed under strict constraints of cost-efficiency, portability, and seaworthiness, with the goal of creating a modular, accessible platform for high-frequency data acquisition and experimental learning. It integrates electromagnetic logs, Doppler velocity logs, inertial sensors, and real-time kinematic GNSS positioning. MARVEL enables real-time, in-situ validation of advanced navigation and AI-driven algorithms using redundant, synchronized sensors. Field experiments demonstrate the system's stability, maneuverability, and adaptability in challenging sea conditions. The platform offers a novel, scalable approach for researchers seeking affordable, open-ended tools to evaluate sensor fusion techniques under real-world maritime constraints. 

**Abstract (ZH)**: MARVEL：一种面向GNSS遮挡环境下的传感器融合导航算法实时验证的自主水面无人驾驶车辆设计与实验验证 

---
# Meta-Ori: monolithic meta-origami for nonlinear inflatable soft actuators 

**Title (ZH)**: 元纸艺：整体非线性可充气软执行器的元 origami 结构 

**Authors**: Hugo de Souza Oliveira, Xin Li, Johannes Frey, Edoardo Milana  

**Link**: [PDF](https://arxiv.org/pdf/2503.23375)  

**Abstract**: The nonlinear mechanical response of soft materials and slender structures is purposefully harnessed to program functions by design in soft robotic actuators, such as sequencing, amplified response, fast energy release, etc. However, typical designs of nonlinear actuators - e.g. balloons, inverted membranes, springs - have limited design parameters space and complex fabrication processes, hindering the achievement of more elaborated functions. Mechanical metamaterials, on the other hand, have very large design parameter spaces, which allow fine-tuning of nonlinear behaviours. In this work, we present a novel approach to fabricate nonlinear inflatables based on metamaterials and origami (Meta-Ori) as monolithic parts that can be fully 3D printed via Fused Deposition Modeling (FDM) using thermoplastic polyurethane (TPU) commercial filaments. Our design consists of a metamaterial shell with cylindrical topology and nonlinear mechanical response combined with a Kresling origami inflatable acting as a pneumatic transmitter. We develop and release a design tool in the visual programming language Grasshopper to interactively design our Meta-Ori. We characterize the mechanical response of the metashell and the origami, and the nonlinear pressure-volume curve of the Meta-Ori inflatable and, lastly, we demonstrate the actuation sequencing of a bi-segment monolithic Meta-Ori soft actuator. 

**Abstract (ZH)**: 基于机械超材料和 origami 的新型非线性可变形体的设计与制造：一种融合 FDM 3D 打印的技术 

---
# MagicGel: A Novel Visual-Based Tactile Sensor Design with MagneticGel 

**Title (ZH)**: MagicGel：一种基于视觉的新型磁性凝胶触觉传感器设计 

**Authors**: Jianhua Shan, Jie Zhao, Jiangduo Liu, Xiangbo Wang, Ziwei Xia, Guangyuan Xu, Bin Fang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23345)  

**Abstract**: Force estimation is the core indicator for evaluating the performance of tactile sensors, and it is also the key technical path to achieve precise force feedback mechanisms. This study proposes a design method for a visual tactile sensor (VBTS) that integrates a magnetic perception mechanism, and develops a new tactile sensor called MagicGel. The sensor uses strong magnetic particles as markers and captures magnetic field changes in real time through Hall sensors. On this basis, MagicGel achieves the coordinated optimization of multimodal perception capabilities: it not only has fast response characteristics, but also can perceive non-contact status information of home electronic products. Specifically, MagicGel simultaneously analyzes the visual characteristics of magnetic particles and the multimodal data of changes in magnetic field intensity, ultimately improving force estimation capabilities. 

**Abstract (ZH)**: 基于磁感知机制的视觉触觉传感器（VBTS）设计方法及MagicGel触觉传感器的研究 

---
# Localized Graph-Based Neural Dynamics Models for Terrain Manipulation 

**Title (ZH)**: 基于局部图的神经动力学模型在地形操控中的应用 

**Authors**: Chaoqi Liu, Yunzhu Li, Kris Hauser  

**Link**: [PDF](https://arxiv.org/pdf/2503.23270)  

**Abstract**: Predictive models can be particularly helpful for robots to effectively manipulate terrains in construction sites and extraterrestrial surfaces. However, terrain state representations become extremely high-dimensional especially to capture fine-resolution details and when depth is unknown or unbounded. This paper introduces a learning-based approach for terrain dynamics modeling and manipulation, leveraging the Graph-based Neural Dynamics (GBND) framework to represent terrain deformation as motion of a graph of particles. Based on the principle that the moving portion of a terrain is usually localized, our approach builds a large terrain graph (potentially millions of particles) but only identifies a very small active subgraph (hundreds of particles) for predicting the outcomes of robot-terrain interaction. To minimize the size of the active subgraph we introduce a learning-based approach that identifies a small region of interest (RoI) based on the robot's control inputs and the current scene. We also introduce a novel domain boundary feature encoding that allows GBNDs to perform accurate dynamics prediction in the RoI interior while avoiding particle penetration through RoI boundaries. Our proposed method is both orders of magnitude faster than naive GBND and it achieves better overall prediction accuracy. We further evaluated our framework on excavation and shaping tasks on terrain with different granularity. 

**Abstract (ZH)**: 基于图的神经动力学学习驱动的地形建模与操控 

---
# Dexterous Non-Prehensile Manipulation for Ungraspable Object via Extrinsic Dexterity 

**Title (ZH)**: 不可抓握物体的外在灵巧 manipulate 技术 

**Authors**: Yuhan Wang, Yu Li, Yaodong Yang, Yuanpei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.23120)  

**Abstract**: Objects with large base areas become ungraspable when they exceed the end-effector's maximum aperture. Existing approaches address this limitation through extrinsic dexterity, which exploits environmental features for non-prehensile manipulation. While grippers have shown some success in this domain, dexterous hands offer superior flexibility and manipulation capabilities that enable richer environmental interactions, though they present greater control challenges. Here we present ExDex, a dexterous arm-hand system that leverages reinforcement learning to enable non-prehensile manipulation for grasping ungraspable objects. Our system learns two strategic manipulation sequences: relocating objects from table centers to edges for direct grasping, or to walls where extrinsic dexterity enables grasping through environmental interaction. We validate our approach through extensive experiments with dozens of diverse household objects, demonstrating both superior performance and generalization capabilities with novel objects. Furthermore, we successfully transfer the learned policies from simulation to a real-world robot system without additional training, further demonstrating its applicability in real-world scenarios. Project website: this https URL. 

**Abstract (ZH)**: 一种利用强化学习实现非抱握操作的灵巧臂手系统：ExDex 

---
# Predictive Traffic Rule Compliance using Reinforcement Learning 

**Title (ZH)**: 基于强化学习的预测性交通规则遵守研究 

**Authors**: Yanliang Huang, Sebastian Mair, Zhuoqi Zeng, Amr Alanwar, Matthias Althoff  

**Link**: [PDF](https://arxiv.org/pdf/2503.22925)  

**Abstract**: Autonomous vehicle path planning has reached a stage where safety and regulatory compliance are crucial. This paper presents a new approach that integrates a motion planner with a deep reinforcement learning model to predict potential traffic rule violations. In this setup, the predictions of the critic directly affect the cost function of the motion planner, guiding the choices of the trajectory. We incorporate key interstate rules from the German Road Traffic Regulation into a rule book and use a graph-based state representation to handle complex traffic information. Our main innovation is replacing the standard actor network in an actor-critic setup with a motion planning module, which ensures both predictable trajectory generation and prevention of long-term rule violations. Experiments on an open German highway dataset show that the model can predict and prevent traffic rule violations beyond the planning horizon, significantly increasing safety in challenging traffic conditions. 

**Abstract (ZH)**: 自主驾驶车辆路径规划已达到一个关键阶段，安全性和法规遵从性至关重要。本文提出了一种新的方法，将运动规划器与深度强化学习模型集成，以预测潜在的交通规则违规行为。在这种设置中，评论家的预测直接影响运动规划器的成本函数，指导轨迹的选择。我们将德国道路交通法规中的关键跨州规则纳入规则书中，并使用基于图的状态表示来处理复杂交通信息。我们的主要创新之处在于，在演员-评论家架构中用运动规划模块替换标准的演员网络，从而确保轨迹生成的可预测性和长期规则违规的预防。在开放的德国高速公路数据集上的实验表明，该模型可以预测并防止超出规划范围的交通规则违规行为，在复杂交通条件下显著提高安全性。 

---
# LiDAR-based Quadrotor Autonomous Inspection System in Cluttered Environments 

**Title (ZH)**: 基于LiDAR的四旋翼无人机复杂环境自主巡检系统 

**Authors**: Wenyi Liu, Huajie Wu, Liuyu Shi, Fangcheng Zhu, Yuying Zou, Fanze Kong, Fu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.22921)  

**Abstract**: In recent years, autonomous unmanned aerial vehicle (UAV) technology has seen rapid advancements, significantly improving operational efficiency and mitigating risks associated with manual tasks in domains such as industrial inspection, agricultural monitoring, and search-and-rescue missions. Despite these developments, existing UAV inspection systems encounter two critical challenges: limited reliability in complex, unstructured, and GNSS-denied environments, and a pronounced dependency on skilled operators. To overcome these limitations, this study presents a LiDAR-based UAV inspection system employing a dual-phase workflow: human-in-the-loop inspection and autonomous inspection. During the human-in-the-loop phase, untrained pilots are supported by autonomous obstacle avoidance, enabling them to generate 3D maps, specify inspection points, and schedule tasks. Inspection points are then optimized using the Traveling Salesman Problem (TSP) to create efficient task sequences. In the autonomous phase, the quadrotor autonomously executes the planned tasks, ensuring safe and efficient data acquisition. Comprehensive field experiments conducted in various environments, including slopes, landslides, agricultural fields, factories, and forests, confirm the system's reliability and flexibility. Results reveal significant enhancements in inspection efficiency, with autonomous operations reducing trajectory length by up to 40\% and flight time by 57\% compared to human-in-the-loop operations. These findings underscore the potential of the proposed system to enhance UAV-based inspections in safety-critical and resource-constrained scenarios. 

**Abstract (ZH)**: 近年来，自主无人机（UAV）技术取得了 rapid advancements，显著提高了工业检测、农业监测和搜救任务等领域的操作效率，并减少了与手动任务相关的风险。尽管取得了这些进展，现有的无人机检测系统仍面临两大关键挑战：在复杂、未结构化和GPS拒止环境中的有限可靠性，以及对熟练操作员的明显依赖。为克服这些限制，本研究提出了一种基于LiDAR的无人机检测系统，采用双阶段工作流：人工在环检测和自主检测。在人工在环阶段，未受过训练的飞行员通过自主障碍回避的支持，生成3D地图、指定检测点并安排任务。然后使用旅行商问题（TSP）优化检测点，以创建高效的任务序列。在自主阶段，四旋翼机自主执行计划任务，确保安全和高效的数据采集。在各种环境中（包括斜坡、滑坡、农业用地、工厂和森林）进行的综合实地试验证实了系统的可靠性和灵活性。结果表明，在检测效率方面有显著提升，与人工在环操作相比，自主操作可将轨迹长度减少40%以上，飞行时间减少57%。这些发现突显了所提出系统在安全关键和资源受限场景中增强无人机检测的潜力。 

---
# A reduced-scale autonomous morphing vehicle prototype with enhanced aerodynamic efficiency 

**Title (ZH)**: 一种增强气动效率的缩小比例自主形态变化车辆原型 

**Authors**: Peng Zhang, Branson Blaylock  

**Link**: [PDF](https://arxiv.org/pdf/2503.22777)  

**Abstract**: Road vehicles contribute to significant levels of greenhouse gas (GHG) emissions. A potential strategy for improving their aerodynamic efficiency and reducing emissions is through active adaptation of their exterior shapes to the aerodynamic environment. In this study, we present a reduced-scale morphing vehicle prototype capable of actively interacting with the aerodynamic environment to enhance fuel economy. Morphing is accomplished by retrofitting a deformable structure actively actuated by built-in motors. The morphing vehicle prototype is integrated with an optimization algorithm that can autonomously identify the structural shape that minimizes aerodynamic drag. The performance of the morphing vehicle prototype is investigated through an extensive experimental campaign in a large-scale wind tunnel facility. The autonomous optimization algorithm identifies an optimal morphing shape that can elicit an 8.5% reduction in the mean drag force. Our experiments provide a comprehensive dataset that validates the efficiency of shape morphing, demonstrating a clear and consistent decrease in the drag force as the vehicle transitions from a suboptimal to the optimal shape. Insights gained from experiments on scaled-down models provide valuable guidelines for the design of full-size morphing vehicles, which could lead to appreciable energy savings and reductions in GHG emissions. This study highlights the feasibility and benefits of real-time shape morphing under conditions representative of realistic road environments, paving the way for the realization of full-scale morphing vehicles with enhanced aerodynamic efficiency and reduced GHG emissions. 

**Abstract (ZH)**: 基于主动形态变化的车辆在典型道路环境下的减阻研究及其对能源节约和温室气体减排的潜在影响 

---
# Co-design of materials, structures and stimuli for magnetic soft robots with large deformation and dynamic contacts 

**Title (ZH)**: 磁软机器人中大变形和动态接触的材料、结构与刺激共设计 

**Authors**: Liwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.22767)  

**Abstract**: Magnetic soft robots embedded with hard magnetic particles enable untethered actuation via external magnetic fields, offering remote, rapid, and precise control, which is highly promising for biomedical applications. However, designing such systems is challenging due to the complex interplay of magneto-elastic dynamics, large deformation, solid contacts, time-varying stimuli, and posture-dependent loading. As a result, most existing research relies on heuristics and trial-and-error methods or focuses on the independent design of stimuli or structures under static conditions. We propose a topology optimization framework for magnetic soft robots that simultaneously designs structures, location-specific material magnetization and time-varying magnetic stimuli, accounting for large deformations, dynamic motion, and solid contacts. This is achieved by integrating generalized topology optimization with the magneto-elastic material point method, which supports GPU-accelerated parallel simulations and auto-differentiation for sensitivity analysis. We applied this framework to design magnetic robots for various tasks, including multi-task shape morphing and locomotion, in both 2D and 3D. The method autonomously generates optimized robotic systems to achieve target behaviors without requiring human intervention. Despite the nonlinear physics and large design space, it demonstrates exceptional efficiency, completing all cases within minutes. This proposed framework represents a significant step toward the automatic co-design of magnetic soft robots for applications such as metasurfaces, drug delivery, and minimally invasive procedures. 

**Abstract (ZH)**: 嵌入硬磁颗粒的磁软机器人通过外部磁场实现无缆驱动，提供远程、快速、精确控制，极具生物医学应用前景。然而，由于磁弹性动力学、大变形、固体接触、时间变化刺激和姿态依赖载荷的复杂相互作用，设计此类系统具有挑战性。因此，现有大多数研究依赖于经验方法或仅在静态条件下独立设计刺激或结构。我们提出了一种拓扑优化框架，用于同时设计磁软机器人的结构、位置特定的材料磁化和时间变化的磁场刺激，考虑大变形、动态运动和固体接触。通过将广义拓扑优化与磁弹性物质点法结合，该框架支持GPU加速并行仿真和自动求导以进行灵敏度分析。我们将此框架应用于设计用于各种任务的磁驱动机器人，包括2D和3D环境下的多任务形状变形和移动。该方法能自主生成优化的机器人系统以实现目标行为，无需人为干预。尽管涉及非线性物理和大的设计空间，该方法表现出色，能在几分钟内完成所有案例。该提议框架代表了自动协同设计磁软机器人以应用于超表面、药物递送和微创手术等应用的重要步骤。 

---
# Strategies for decentralised UAV-based collisions monitoring in rugby 

**Title (ZH)**: 基于橄榄球的去中心化无人机碰撞监测策略 

**Authors**: Yu Cheng, Harun Šiljak  

**Link**: [PDF](https://arxiv.org/pdf/2503.22757)  

**Abstract**: Recent advancements in unmanned aerial vehicle (UAV) technology have opened new avenues for dynamic data collection in challenging environments, such as sports fields during fast-paced sports action. For the purposes of monitoring sport events for dangerous injuries, we envision a coordinated UAV fleet designed to capture high-quality, multi-view video footage of collision events in real-time. The extracted video data is crucial for analyzing athletes' motions and investigating the probability of sports-related traumatic brain injuries (TBI) during impacts. This research implemented a UAV fleet system on the NetLogo platform, utilizing custom collision detection algorithms to compare against traditional TV-coverage strategies. Our system supports decentralized data capture and autonomous processing, providing resilience in the rapidly evolving dynamics of sports collisions.
The collaboration algorithm integrates both shared and local data to generate multi-step analyses aimed at determining the efficacy of custom methods in enhancing the accuracy of TBI prediction models. Missions are simulated in real-time within a two-dimensional model, focusing on the strategic capture of collision events that could lead to TBI, while considering operational constraints such as rapid UAV maneuvering and optimal positioning. Preliminary results from the NetLogo simulations suggest that custom collision detection methods offer superior performance over standard TV-coverage strategies by enabling more precise and timely data capture. This comparative analysis highlights the advantages of tailored algorithmic approaches in critical sports safety applications. 

**Abstract (ZH)**: 近期无人机（UAV）技术的进步为在快节奏体育比赛中采集动态数据开辟了新的途径，特别是在体育场地上。为了监测体育赛事中的危险伤害，我们设想了一套协同无人机舰队，旨在实时捕捉碰撞事件的高质量多视角视频。提取的视频数据对于分析运动员的动作并调查运动相关创伤性脑损伤（TBI）的概率至关重要。该研究在NetLogo平台上实施了一个无人机舰队系统，利用定制的碰撞检测算法与传统的电视直播策略进行对比。该系统支持分散的数据采集和自主处理，提供了在体育碰撞动态快速演变中的弹性。合作算法结合共享和本地数据，生成多步分析，以确定自定义方法在提高TBI预测模型准确性方面的有效性。研究在二维模型中实时模拟任务，集中在战略捕捉可能导致TBI的碰撞事件上，同时考虑操作约束，如快速无人机机动和最佳定位。NetLogo模拟的初步结果显示，自定义碰撞检测方法在数据捕获的精确性和及时性方面优于标准的电视直播策略，这突显了定制算法方法在关键体育安全应用中的优势。 

---
# Evaluation of Remote Driver Performance in Urban Environment Operational Design Domains 

**Title (ZH)**: 城市环境操作设计域中远程驾驶性能评价 

**Authors**: Ole Hans, Benedikt Walter, Jürgen Adamy  

**Link**: [PDF](https://arxiv.org/pdf/2503.22992)  

**Abstract**: Remote driving has emerged as a solution for enabling human intervention in scenarios where Automated Driving Systems (ADS) face challenges, particularly in urban Operational Design Domains (ODDs). This study evaluates the performance of Remote Drivers (RDs) of passenger cars in a representative urban ODD in Las Vegas, focusing on the influence of cumulative driving experience and targeted training approaches. Using performance metrics such as efficiency, braking, acceleration, and steering, the study shows that driving experience can lead to noticeable improvements of RDs and demonstrates how experience up to 600 km correlates with improved vehicle control. In addition, driving efficiency exhibited a positive trend with increasing kilometers, particularly during the first 300 km of experience, which reaches a plateau from 400 km within a range of 0.35 to 0.42 km/min in the defined ODD. The research further compares ODD-specific training methods, where the detailed ODD training approaches attains notable advantages over other training approaches. The findings underscore the importance of tailored ODD training in enhancing RD performance, safety, and scalability for Remote Driving System (RDS) in real-world applications, while identifying opportunities for optimizing training protocols to address both routine and extreme scenarios. The study provides a robust foundation for advancing RDS deployment within urban environments, contributing to the development of scalable and safety-critical remote operation standards. 

**Abstract (ZH)**: 远程驾驶在应对自动化驾驶系统在城市操作设计领域（ODD）面临的挑战中 emerged as a解决方案。本研究评估了在拉斯维加斯一个代表性城市ODD中远程驾驶员（RDs）的表现，重点关注累计驾驶经验与目标培训方法的影响。通过使用效率、刹车、加速和转向等性能指标，研究显示驾驶经验可以显著提高远程驾驶员的表现，并表明累计驾驶600公里的经验与车辆控制的改善相关。此外，随经验增加的驾驶效率呈现出积极趋势，特别是在经验最初的300公里中，效率在定义的ODD范围内于400公里左右达到0.35至0.42公里/分钟的 plateau。研究进一步比较了ODD特定的培训方法，其中详细的ODD培训方法表现出显著的优势。研究结果强调了为远程驾驶系统（RDS）在实际应用中增强远程驾驶员的表现、安全性和可扩展性，而定制的ODD培训的重要性，并指出了优化培训协议以应对常规和极端情况的机会。该研究为在城市环境中推进远程驾驶系统的部署奠定了坚实的基础，促进了可扩展且安全关键的远程操作标准的发展。 

---
# A Multiple Artificial Potential Functions Approach for Collision Avoidance in UAV Systems 

**Title (ZH)**: 基于多个人工势能函数的方法在无人机系统中的碰撞避免 

**Authors**: Oscar F. Archila, Alain Vande Wouwer, Johannes Schiffer  

**Link**: [PDF](https://arxiv.org/pdf/2503.22830)  

**Abstract**: Collision avoidance is a problem largely studied in robotics, particularly in unmanned aerial vehicle (UAV) applications. Among the main challenges in this area are hardware limitations, the need for rapid response, and the uncertainty associated with obstacle detection. Artificial potential functions (APOFs) are a prominent method to address these challenges. However, existing solutions lack assurances regarding closed-loop stability and may result in chattering effects. Motivated by this, we propose a control method for static obstacle avoidance based on multiple artificial potential functions (MAPOFs). We derive tuning conditions on the control parameters that ensure the stability of the final position. The stability proof is established by analyzing the closed-loop system using tools from hybrid systems theory. Furthermore, we validate the performance of the MAPOF control through simulations, showcasing its effectiveness in avoiding static obstacles. 

**Abstract (ZH)**: 基于多人工势函数的静态障碍物规避控制方法及其稳定性分析 

---
# Incorporating GNSS Information with LIDAR-Inertial Odometry for Accurate Land-Vehicle Localization 

**Title (ZH)**: 融合GNSS信息的机载激光雷达-惯性里程计定位方法用于精确的土地车辆定位 

**Authors**: Jintao Cheng, Bohuan Xue, Shiyang Chen, Qiuchi Xiang, Xiaoyu Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23199)  

**Abstract**: Currently, visual odometry and LIDAR odometry are performing well in pose estimation in some typical environments, but they still cannot recover the localization state at high speed or reduce accumulated drifts. In order to solve these problems, we propose a novel LIDAR-based localization framework, which achieves high accuracy and provides robust localization in 3D pointcloud maps with information of multi-sensors. The system integrates global information with LIDAR-based odometry to optimize the localization state. To improve robustness and enable fast resumption of localization, this paper uses offline pointcloud maps for prior knowledge and presents a novel registration method to speed up the convergence rate. The algorithm is tested on various maps of different data sets and has higher robustness and accuracy than other localization algorithms. 

**Abstract (ZH)**: 基于激光雷达的高精度三维点云地图定位框架 

---
