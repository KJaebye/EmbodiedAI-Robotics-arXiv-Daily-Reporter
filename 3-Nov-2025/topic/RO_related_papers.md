# A Modular and Scalable System Architecture for Heterogeneous UAV Swarms Using ROS 2 and PX4-Autopilot 

**Title (ZH)**: 基于ROS 2和PX4自动驾驶器的模块化和可扩展异构无人机群系统架构 

**Authors**: Robert Pommeranz, Kevin Tebbe, Ralf Heynicke, Gerd Scholl  

**Link**: [PDF](https://arxiv.org/pdf/2510.27327)  

**Abstract**: In this paper a modular and scalable architecture for heterogeneous swarm-based Counter Unmanned Aerial Systems (C-UASs) built on PX4-Autopilot and Robot Operating System 2 (ROS 2) framework is presented. The proposed architecture emphasizes seamless integration of hardware components by introducing independent ROS 2 nodes for each component of a Unmanned Aerial Vehicle (UAV). Communication between swarm participants is abstracted in software, allowing the use of various technologies without architectural changes. Key functionalities are supported, e.g. leader following and formation flight to maneuver the swarm. The system also allows computer vision algorithms to be integrated for the detection and tracking of UAVs. Additionally, a ground station control is integrated for the coordination of swarm operations. Swarm-based Unmanned Aerial System (UAS) architecture is verified within a Gazebo simulation environment but also in real-world demonstrations. 

**Abstract (ZH)**: 基于PX4-Autopilot和ROS 2的异构 swarm-Based C-UAS模块化可扩展架构 

---
# Hybrid Gripper Finger Enabling In-Grasp Friction Modulation Using Inflatable Silicone Pockets 

**Title (ZH)**: 具有可调节摩擦的混合夹持器手指设计：基于可充气硅胶囊袋 

**Authors**: Hoang Hiep Ly, Cong-Nhat Nguyen, Doan-Quang Tran, Quoc-Khanh Dang, Ngoc Duy Tran, Thi Thoa Mac, Anh Nguyen, Xuan-Thuan Nguyen, Tung D. Ta  

**Link**: [PDF](https://arxiv.org/pdf/2510.27184)  

**Abstract**: Grasping objects with diverse mechanical properties, such as heavy, slippery, or fragile items, remains a significant challenge in robotics. Conventional grippers often rely on applying high normal forces, which can cause damage to objects. To address this limitation, we present a hybrid gripper finger that combines a rigid structural shell with a soft, inflatable silicone pocket. The gripper finger can actively modulate its surface friction by controlling the internal air pressure of the silicone pocket. Results from fundamental experiments indicate that increasing the internal pressure results in a proportional increase in the effective coefficient of friction. This enables the gripper to stably lift heavy and slippery objects without increasing the gripping force and to handle fragile or deformable objects, such as eggs, fruits, and paper cups, with minimal damage by increasing friction rather than applying excessive force. The experimental results demonstrate that the hybrid gripper finger with adaptable friction provides a robust and safer alternative to relying solely on high normal forces, thereby enhancing the gripper flexibility in handling delicate, fragile, and diverse objects. 

**Abstract (ZH)**: 具有可调节摩擦力的混合手指抓取具有多样化机械性能的物体：一种减少高正压力依赖性的替代方案 

---
# MobiDock: Design and Control of A Modular Self Reconfigurable Bimanual Mobile Manipulator via Robotic Docking 

**Title (ZH)**: MobiDock：模块化自重构双臂移动 manipulator 的设计与控制研究通过机器人对接 

**Authors**: Xuan-Thuan Nguyen, Khac Nam Nguyen, Ngoc Duy Tran, Thi Thoa Mac, Anh Nguyen, Hoang Hiep Ly, Tung D. Ta  

**Link**: [PDF](https://arxiv.org/pdf/2510.27178)  

**Abstract**: Multi-robot systems, particularly mobile manipulators, face challenges in control coordination and dynamic stability when working together. To address this issue, this study proposes MobiDock, a modular self-reconfigurable mobile manipulator system that allows two independent robots to physically connect and form a unified mobile bimanual platform. This process helps transform a complex multi-robot control problem into the management of a simpler, single system. The system utilizes an autonomous docking strategy based on computer vision with AprilTag markers and a new threaded screw-lock mechanism. Experimental results show that the docked configuration demonstrates better performance in dynamic stability and operational efficiency compared to two independently cooperating robots. Specifically, the unified system has lower Root Mean Square (RMS) Acceleration and Jerk values, higher angular precision, and completes tasks significantly faster. These findings confirm that physical reconfiguration is a powerful design principle that simplifies cooperative control, improving stability and performance for complex tasks in real-world environments. 

**Abstract (ZH)**: 多机器人系统，特别是移动 manipulator，在协同工作时面临控制协调和动态稳定性方面的挑战。为解决这一问题，本研究提出了一种模块化自重构移动 manipulator 系统 MobiDock，该系统允许两个独立的机器人物理连接并形成一个统一的双臂移动平台。这一过程有助于将复杂的多机器人控制问题转化为单一系统的管理问题。该系统采用基于计算机视觉和 AprilTag 标记的自主对接策略以及一种新的带有螺纹螺母锁紧机制。实验结果表明，对接配置在动态稳定性和操作效率方面优于两个独立合作的机器人。具体而言，统一系统具有较低的均方根加速度和冲击值，更高的角度精度，且能显著更快地完成任务。这些发现证实了物理重构是一种强大的设计原则，能够简化协作控制，提高复杂任务在实际环境中的稳定性和性能。 

---
# Confined Space Underwater Positioning Using Collaborative Robots 

**Title (ZH)**: 受限空间水下定位 Using协作机器人 

**Authors**: Xueliang Cheng, Kanzhong Yao, Andrew West, Ognjen Marjanovic, Barry Lennox, Keir Groves  

**Link**: [PDF](https://arxiv.org/pdf/2510.27151)  

**Abstract**: Positioning of underwater robots in confined and cluttered spaces remains a key challenge for field operations. Existing systems are mostly designed for large, open-water environments and struggle in industrial settings due to poor coverage, reliance on external infrastructure, and the need for feature-rich surroundings. Multipath effects from continuous sound reflections further degrade signal quality, reducing accuracy and reliability. Accurate and easily deployable positioning is essential for repeatable autonomous missions; however, this requirement has created a technological bottleneck limiting underwater robotic deployment. This paper presents the Collaborative Aquatic Positioning (CAP) system, which integrates collaborative robotics and sensor fusion to overcome these limitations. Inspired by the "mother-ship" concept, the surface vehicle acts as a mobile leader to assist in positioning a submerged robot, enabling localization even in GPS-denied and highly constrained environments. The system is validated in a large test tank through repeatable autonomous missions using CAP's position estimates for real-time trajectory control. Experimental results demonstrate a mean Euclidean distance (MED) error of 70 mm, achieved in real time without requiring fixed infrastructure, extensive calibration, or environmental features. CAP leverages advances in mobile robot sensing and leader-follower control to deliver a step change in accurate, practical, and infrastructure-free underwater localization. 

**Abstract (ZH)**: 水下机器人在受限和拥挤空间中的定位仍然是现场作业中的一个关键挑战。现有的系统主要设计用于开阔水域环境，在工业环境中由于覆盖范围差、依赖外部基础设施以及需要特征丰富的周围环境而难以适用。连续的声波反射导致的多路径效应进一步恶化信号质量，降低定位的准确性与可靠性。准确且易于部署的定位对于重复的自治任务至关重要；然而，这一要求形成了技术瓶颈，限制了水下机器人部署。本文提出了协作式水下定位（CAP）系统，该系统结合了协作机器人和传感器融合以克服这些限制。受“母舰”概念的启发，水面车辆作为移动的领导者协助定位潜航器，使得即使在GPS受限和高度受限的环境中也能实现定位。该系统通过在大型试验水箱中执行重复的自治任务并使用CAP的位置估计进行实时轨迹控制得到了验证。实验结果表明，CAP实现了实时平均欧几里得距离误差为70毫米，无需固定基础设施、大量校准或环境特征。CAP利用移动机器人感知和领导者-跟随者控制的最新进展，实现了准确、实用且无需基础设施的水下定位的跨越式进步。 

---
# A Hermetic, Transparent Soft Growing Vine Robot System for Pipe Inspection 

**Title (ZH)**: 密闭透明的软生长藤条机器人系统用于管道检查 

**Authors**: William E. Heap, Yimeng Qin, Kai Hammond, Anish Bayya, Haonon Kong, Allison M. Okamura  

**Link**: [PDF](https://arxiv.org/pdf/2510.27010)  

**Abstract**: Rehabilitation of aging pipes requires accurate condition assessment and mapping far into the pipe interiors. Soft growing vine robot systems are particularly promising for navigating confined, sinuous paths such as in pipes, but are currently limited by complex subsystems and a lack of validation in real-world industrial settings. In this paper, we introduce the concept and implementation of a hermetic and transparent vine robot system for visual condition assessment and mapping within non-branching pipes. This design encloses all mechanical and electrical components within the vine robot's soft, airtight, and transparent body, protecting them from environmental interference while enabling visual sensing. Because this approach requires an enclosed mechanism for transporting sensors, we developed, modeled, and tested a passively adapting enclosed tip mount. Finally, we validated the hermetic and transparent vine robot system concept through a real-world condition assessment and mapping task in a wastewater pipe. This work advances the use of soft-growing vine robots in pipe inspection by developing and demonstrating a robust, streamlined, field-validated system suitable for continued development and deployment. 

**Abstract (ZH)**: 老化管道的康复需要准确的状况评估和深入管道内部的测绘。软生长藤蔓机器人系统尤其适用于导航狭小蜿蜒的管道路径，但目前受限于复杂的子系统和缺乏实际工业环境下的验证。本文介绍并实施了一种密封且透明的藤蔓机器人系统，用于非分叉管道内部的视觉状况评估和测绘。该设计将所有机械和电气组件封装在藤蔓机器人的软性、密封且透明的身体中，以防止外部干扰并允许视觉传感。由于该方法需要一个封闭机制来运输传感器，我们开发、建模并测试了被动适应的封闭末端装卡。最后，我们通过实际的污水管道状况评估和测绘任务验证了密封且透明的藤蔓机器人系统概念。这项工作通过开发并展示了一个坚实、简洁且在现场验证的系统，促进了软生长藤蔓机器人的管道检测应用，适合继续发展和部署。 

---
# Design for One, Deploy for Many: Navigating Tree Mazes with Multiple Agents 

**Title (ZH)**: 为一设计，为多部署：多智能体导航树迷宫 

**Authors**: Jahir Argote-Gerald, Genki Miyauchi, Julian Rau, Paul Trodden, Roderich Gross  

**Link**: [PDF](https://arxiv.org/pdf/2510.26900)  

**Abstract**: Maze-like environments, such as cave and pipe networks, pose unique challenges for multiple robots to coordinate, including communication constraints and congestion. To address these challenges, we propose a distributed multi-agent maze traversal algorithm for environments that can be represented by acyclic graphs. It uses a leader-switching mechanism where one agent, assuming a head role, employs any single-agent maze solver while the other agents each choose an agent to follow. The head role gets transferred to neighboring agents where necessary, ensuring it follows the same path as a single agent would. The multi-agent maze traversal algorithm is evaluated in simulations with groups of up to 300 agents, various maze sizes, and multiple single-agent maze solvers. It is compared against strategies that are naïve, or assume either global communication or full knowledge of the environment. The algorithm outperforms the naïve strategy in terms of makespan and sum-of-fuel. It is superior to the global-communication strategy in terms of makespan but is inferior to it in terms of sum-of-fuel. The findings suggest it is asymptotically equivalent to the full-knowledge strategy with respect to either metric. Moreover, real-world experiments with up to 20 Pi-puck robots confirm the feasibility of the approach. 

**Abstract (ZH)**: 迷宫型环境中的多机器人分布式迷宫穿越算法：适用于有向图表示的洞穴和管道网络环境 

---
# Force Characterization of Insect-Scale Aquatic Propulsion Based on Fluid-Structure Interaction 

**Title (ZH)**: 基于流固耦合的微型昆虫 aquatic 推进力特性研究 

**Authors**: Conor K. Trygstad, Nestor O. Perez-Arancibia  

**Link**: [PDF](https://arxiv.org/pdf/2510.26837)  

**Abstract**: We present force characterizations of two newly developed insect-scale propulsors--one single-tailed and one double-tailed--for microrobotic swimmers that leverage fluid-structure interaction (FSI) to generate thrust. The designs of these two devices were inspired by anguilliform swimming and are driven by soft tails excited by high-work-density (HWD) actuators powered by shape-memory alloy (SMA) wires. While these propulsors have been demonstrated to be suitable for microrobotic aquatic locomotion and controllable with simple architectures for trajectory tracking in the two-dimensional (2D) space, the characteristics and magnitudes of the associated forces have not been studied systematically. In the research presented here, we adopted a theoretical framework based on the notion of reactive forces and obtained experimental data for characterization using a custom-built micro-N-resolution force sensor. We measured maximum and cycle-averaged force values with multi-test means of respectively 0.45 mN and 2.97 micro-N, for the tested single-tail propulsor. For the dual-tail propulsor, we measured maximum and cycle-averaged force values with multi-test means of 0.61 mN and 22.6 micro-N, respectively. These results represent the first measurements of the instantaneous thrust generated by insect-scale propulsors of this type and provide insights into FSI for efficient microrobotic propulsion. 

**Abstract (ZH)**: 我们对两种新型昆虫尺度推进器进行了力特性分析——一种单尾推进器和一种双尾推进器，这些推进器利用流固交互（FSI）产生推力，用于微小型水下机器人游动。这两种装置的设计灵感来源于鳗iform游泳，并由软尾部驱动，软尾部通过形状记忆合金（SMA）线圈驱动的高功密度（HWD）执行器激发。尽管这两种推进器已经在二维空间的轨迹跟踪控制中被证明适合微小型水下机器人游动，但它们的力特性和力的大小尚未系统研究。在本研究中，我们采用基于反作用力概念的理论框架，并使用自制的微N级力传感器获取实验数据。我们测得测试的单尾推进器的最大力和循环平均力分别为0.45 mN和2.97 μN。对于双尾推进器，测得的最大力和循环平均力分别为0.61 mN和22.6 μN。这些结果代表了这种尺度推进器瞬时推力的首次测量，并为高效的微小型机器人推进提供了流固交互的见解。 

---
# Reinforcement Learning for Accelerator Beamline Control: a simulation-based approach 

**Title (ZH)**: 基于仿真的一种加速器束线控制的强化学习方法 

**Authors**: Anwar Ibrahim, Alexey Petrenko, Maxim Kaledin, Ehab Suleiman, Fedor Ratnikov, Denis Derkach  

**Link**: [PDF](https://arxiv.org/pdf/2510.26805)  

**Abstract**: Particle accelerators play a pivotal role in advancing scientific research, yet optimizing beamline configurations to maximize particle transmission remains a labor-intensive task requiring expert intervention. In this work, we introduce RLABC (Reinforcement Learning for Accelerator Beamline Control), a Python-based library that reframes beamline optimization as a reinforcement learning (RL) problem. Leveraging the Elegant simulation framework, RLABC automates the creation of an RL environment from standard lattice and element input files, enabling sequential tuning of magnets to minimize particle losses. We define a comprehensive state representation capturing beam statistics, actions for adjusting magnet parameters, and a reward function focused on transmission efficiency. Employing the Deep Deterministic Policy Gradient (DDPG) algorithm, we demonstrate RLABC's efficacy on two beamlines, achieving transmission rates of 94% and 91%, comparable to expert manual optimizations. This approach bridges accelerator physics and machine learning, offering a versatile tool for physicists and RL researchers alike to streamline beamline tuning. 

**Abstract (ZH)**: 粒子加速器在推动科学研究中发挥着关键作用，然而，优化束线配置以最大化粒子传输效率仍是一项劳-intensive的任务，需要专家干预。本文介绍了一种基于Python的RLABC（Reinforcement Learning for Accelerator Beamline Control）库，将束线优化重新定义为一个强化学习（RL）问题。借助Elegant仿真框架，RLABC自动从标准束线和元件输入文件创建RL环境，实现对磁铁的顺序调谐以最小化粒子损失。我们定义了一个全面的状态表示，包括束流统计、调整磁铁参数的动作以及专注于传输效率的奖励函数。采用深度确定性策略梯度（DDPG）算法，我们在两个束线上展示了RLABC的有效性，分别实现了94%和91%的传输率，与专家手动优化相当。该方法将加速器物理与机器学习相结合，为物理学家和RL研究人员提供了一个灵活的工具，用于简化束线调谐。 

---
