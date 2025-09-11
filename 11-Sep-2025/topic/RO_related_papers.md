# TANGO: Traversability-Aware Navigation with Local Metric Control for Topological Goals 

**Title (ZH)**: TANGO: 基于通行性感知的局部度量控制拓扑目标导航 

**Authors**: Stefan Podgorski, Sourav Garg, Mehdi Hosseinzadeh, Lachlan Mares, Feras Dayoub, Ian Reid  

**Link**: [PDF](https://arxiv.org/pdf/2509.08699)  

**Abstract**: Visual navigation in robotics traditionally relies on globally-consistent 3D maps or learned controllers, which can be computationally expensive and difficult to generalize across diverse environments. In this work, we present a novel RGB-only, object-level topometric navigation pipeline that enables zero-shot, long-horizon robot navigation without requiring 3D maps or pre-trained controllers. Our approach integrates global topological path planning with local metric trajectory control, allowing the robot to navigate towards object-level sub-goals while avoiding obstacles. We address key limitations of previous methods by continuously predicting local trajectory using monocular depth and traversability estimation, and incorporating an auto-switching mechanism that falls back to a baseline controller when necessary. The system operates using foundational models, ensuring open-set applicability without the need for domain-specific fine-tuning. We demonstrate the effectiveness of our method in both simulated environments and real-world tests, highlighting its robustness and deployability. Our approach outperforms existing state-of-the-art methods, offering a more adaptable and effective solution for visual navigation in open-set environments. The source code is made publicly available: this https URL. 

**Abstract (ZH)**: 视觉导向在机器人技术中传统上依赖于全局一致的3D地图或学习控制器，这可能会产生计算上的负担并在不同环境间难以泛化。在这项工作中，我们提出了一种新颖的仅基于RGB图像、基于对象级的拓扑导航流水线，该流水线能够在无需3D地图或预训练控制器的情况下实现零样本、长时距的机器人导航。我们的方法结合了全局拓扑路径规划与局部度量轨迹控制，使机器人能够导航至基于对象的中继目标同时避开障碍物。我们通过连续预测局部轨迹并利用单目深度估计和通过性估计来解决先前方法的关键限制，还引入了一个自动切换机制，在必要时切换回基础控制器。该系统采用基础模型运行，确保在无需领域特定微调的情况下具有开放集适用性。我们在模拟环境和实际测试中展示了我们方法的有效性，强调了其鲁棒性和部署性。我们的方法优于现有最先进的方法，提供了在开放集环境中更具适应性和有效性的一种视觉导航解决方案。源代码已公开：this https URL。 

---
# RoboMatch: A Mobile-Manipulation Teleoperation Platform with Auto-Matching Network Architecture for Long-Horizon Manipulation 

**Title (ZH)**: RoboMatch：一种具有自动匹配网络架构的移动 manipulator 远程操作平台，用于长时 horizon 操作 

**Authors**: Hanyu Liu, Yunsheng Ma, Jiaxin Huang, Keqiang Ren, Jiayi Wen, Yilin Zheng, Baishu Wan, Pan Li, Jiejun Hou, Haoru Luan, Zhihua Wang, Zhigong Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.08522)  

**Abstract**: This paper presents RoboMatch, a novel unified teleoperation platform for mobile manipulation with an auto-matching network architecture, designed to tackle long-horizon tasks in dynamic environments. Our system enhances teleoperation performance, data collection efficiency, task accuracy, and operational stability. The core of RoboMatch is a cockpit-style control interface that enables synchronous operation of the mobile base and dual arms, significantly improving control precision and data collection. Moreover, we introduce the Proprioceptive-Visual Enhanced Diffusion Policy (PVE-DP), which leverages Discrete Wavelet Transform (DWT) for multi-scale visual feature extraction and integrates high-precision IMUs at the end-effector to enrich proprioceptive feedback, substantially boosting fine manipulation performance. Furthermore, we propose an Auto-Matching Network (AMN) architecture that decomposes long-horizon tasks into logical sequences and dynamically assigns lightweight pre-trained models for distributed inference. Experimental results demonstrate that our approach improves data collection efficiency by over 20%, increases task success rates by 20-30% with PVE-DP, and enhances long-horizon inference performance by approximately 40% with AMN, offering a robust solution for complex manipulation tasks. 

**Abstract (ZH)**: RoboMatch：一种用于移动操控的新型统一平台及其自匹配网络架构 

---
# FMT$^{x}$: An Efficient and Asymptotically Optimal Extension of the Fast Marching Tree for Dynamic Replanning 

**Title (ZH)**: FMT$^{x}$: 一种高效的动态重规划的快速推进树扩展算法 

**Authors**: Soheil Espahbodini Nia  

**Link**: [PDF](https://arxiv.org/pdf/2509.08521)  

**Abstract**: Path planning in dynamic environments remains a core challenge in robotics, especially as autonomous systems are deployed in unpredictable spaces such as warehouses and public roads. While algorithms like Fast Marching Tree (FMT$^{*}$) offer asymptotically optimal solutions in static settings, their single-pass design prevents path revisions which are essential for real-time adaptation. On the other hand, full replanning is often too computationally expensive. This paper introduces FMT$^{x}$, an extension of the Fast Marching Tree algorithm that enables efficient and consistent replanning in dynamic environments. We revisit the neighbor selection rule of FMT$^{*}$ and demonstrate that a minimal change overcomes its single-pass limitation, enabling the algorithm to update cost-to-come values upon discovering better connections without sacrificing asymptotic optimality or computational efficiency. By maintaining a cost-ordered priority queue and applying a selective update condition that uses an expanding neighbor to identify and trigger the re-evaluation of any node with a potentially suboptimal path, FMT$^{x}$ ensures that suboptimal routes are efficiently repaired as the environment evolves. This targeted strategy preserves the inherent efficiency of FMT$^{*}$ while enabling robust adaptation to changes in obstacle configuration. FMT$^{x}$ is proven to recover an asymptotically optimal solution after environmental changes. Experimental results demonstrate that FMT$^{x}$ outperforms the influential replanner RRT$^{x}$, reacting more swiftly to dynamic events with lower computational overhead and thus offering a more effective solution for real-time robotic navigation in unpredictable worlds. 

**Abstract (ZH)**: 动态环境中的路径规划仍然是机器人技术中的核心挑战，尤其是在仓库和公共道路上等不可预测的空间中部署自主系统时。虽然像快速前行树(Fast Marching Tree, FMT$^{*}$)这样的算法在静态环境中提供渐近最优的解决方案，但它们的一次性设计限制了路径的修订，这在实时适应中是必不可少的。另一方面，完全重新规划往往计算成本过高。本文介绍了FMT$^{x}$，这是一种FMT$^{*}$算法的扩展，能够支持动态环境中的高效且一致的重新规划。我们重新审视了FMT$^{*}$的邻居选择规则，并证明通过最小的变化克服了其一次性设计的局限性，使算法能够在发现更好的连接时更新成本到终点值，而不会牺牲渐近最优性或计算效率。通过维护一个按成本排序的优先队列，并应用选择性更新条件，使用扩展邻居来识别并触发任何潜在次优路径的节点的重新评估，FMT$^{x}$确保随着环境的变化，次优路径能够高效地得到修复。这种定向策略保留了FMT$^{*}$固有的高效性，同时能够稳健地适应障碍配置的变化。FMT$^{x}$能够在环境变化后恢复渐近最优的解决方案。实验结果表明，FMT$^{x}$在应对动态事件方面比有影响力的重新规划算法RRT$^{x}$更快捷，计算开销更低，因此为不可预测环境中的实时机器人导航提供了更有效的解决方案。 

---
# Facilitating the Emergence of Assistive Robots to Support Frailty: Psychosocial and Environmental Realities 

**Title (ZH)**: 促进辅助机器人在应对衰弱方面的 emergence，以考虑心理社会和环境现实 

**Authors**: Angela Higgins, Stephen Potter, Mauro Dragone, Mark Hawley, Farshid Amirabdollahian, Alessandro Di Nuovo, Praminda Caleb-Solly  

**Link**: [PDF](https://arxiv.org/pdf/2509.08510)  

**Abstract**: While assistive robots have much potential to help older people with frailty-related needs, there are few in use. There is a gap between what is developed in laboratories and what would be viable in real-world contexts. Through a series of co-design workshops (61 participants across 7 sessions) including those with lived experience of frailty, their carers, and healthcare professionals, we gained a deeper understanding of everyday issues concerning the place of new technologies in their lives. A persona-based approach surfaced emotional, social, and psychological issues. Any assistive solution must be developed in the context of this complex interplay of psychosocial and environmental factors. Our findings, presented as design requirements in direct relation to frailty, can help promote design thinking that addresses people's needs in a more pragmatic way to move assistive robotics closer to real-world use. 

**Abstract (ZH)**: 辅助机器人虽有潜力帮助体弱老人，但在实际应用中却鲜有使用。实验室中的开发与现实应用之间存在差距。通过一系列共设工作坊（共计7次，61名参与者，包括体弱经验者、照护者及医疗专业人员），我们深入理解了新技术在他们生活中的日常问题。基于角色的方法揭示了情感、社会和心理问题。任何辅助解决方案都必须考虑到心理社会和环境因素的复杂交互。我们的研究成果，以直接关联体弱的需求形式呈现，有助于促进更实际的设计思维，推动辅助机器人技术更接近实际应用。 

---
# CLAP: Clustering to Localize Across n Possibilities, A Simple, Robust Geometric Approach in the Presence of Symmetries 

**Title (ZH)**: CLAP：在存在对称性的条件下定位多种可能性的聚类方法，一种简单的稳健几何approach 

**Authors**: Gabriel I. Fernandez, Ruochen Hou, Alex Xu, Colin Togashi, Dennis W. Hong  

**Link**: [PDF](https://arxiv.org/pdf/2509.08495)  

**Abstract**: In this paper, we present our localization method called CLAP, Clustering to Localize Across $n$ Possibilities, which helped us win the RoboCup 2024 adult-sized autonomous humanoid soccer competition. Competition rules limited our sensor suite to stereo vision and an inertial sensor, similar to humans. In addition, our robot had to deal with varying lighting conditions, dynamic feature occlusions, noise from high-impact stepping, and mistaken features from bystanders and neighboring fields. Therefore, we needed an accurate, and most importantly robust localization algorithm that would be the foundation for our path-planning and game-strategy algorithms. CLAP achieves these requirements by clustering estimated states of our robot from pairs of field features to localize its global position and orientation. Correct state estimates naturally cluster together, while incorrect estimates spread apart, making CLAP resilient to noise and incorrect inputs. CLAP is paired with a particle filter and an extended Kalman filter to improve consistency and smoothness. Tests of CLAP with other landmark-based localization methods showed similar accuracy. However, tests with increased false positive feature detection showed that CLAP outperformed other methods in terms of robustness with very little divergence and velocity jumps. Our localization performed well in competition, allowing our robot to shoot faraway goals and narrowly defend our goal. 

**Abstract (ZH)**: 基于聚类的多可能性定位方法CLAP：RoboCup 2024成人组自主人形足球竞赛中的定位技术 

---
# Augmenting Neural Networks-based Model Approximators in Robotic Force-tracking Tasks 

**Title (ZH)**: 基于神经网络的模型逼近器在机器人力跟踪任务中的增强 

**Authors**: Kevin Saad, Vincenzo Petrone, Enrico Ferrentino, Pasquale Chiacchio, Francesco Braghin, Loris Roveda  

**Link**: [PDF](https://arxiv.org/pdf/2509.08440)  

**Abstract**: As robotics gains popularity, interaction control becomes crucial for ensuring force tracking in manipulator-based tasks. Typically, traditional interaction controllers either require extensive tuning, or demand expert knowledge of the environment, which is often impractical in real-world applications. This work proposes a novel control strategy leveraging Neural Networks (NNs) to enhance the force-tracking behavior of a Direct Force Controller (DFC). Unlike similar previous approaches, it accounts for the manipulator's tangential velocity, a critical factor in force exertion, especially during fast motions. The method employs an ensemble of feedforward NNs to predict contact forces, then exploits the prediction to solve an optimization problem and generate an optimal residual action, which is added to the DFC output and applied to an impedance controller. The proposed Velocity-augmented Artificial intelligence Interaction Controller for Ambiguous Models (VAICAM) is validated in the Gazebo simulator on a Franka Emika Panda robot. Against a vast set of trajectories, VAICAM achieves superior performance compared to two baseline controllers. 

**Abstract (ZH)**: 随着机器人技术的流行，交互控制对于确保基于操作器的任务中的力跟踪变得至关重要。传统交互控制器通常要么需要大量的调整，要么需要专家对环境的了解，这在实际应用中往往是不现实的。本工作提出了一种新的控制策略，利用神经网络（NNs）以增强直接力控制器（DFC）的力跟踪行为。与之前的类似方法不同，该方法考虑了操作器的切向速度，这是力施加的一个关键因素，尤其是在快速运动中。该方法使用前向神经网络的集成来预测接触力，然后利用预测来解决优化问题并生成最优残差动作，该动作被添加到DFC的输出并应用于阻抗控制器。提出的基于速度增强的人工智能交互控制器 for 不确定模型（VAICAM）在 Franka Emika Panda 机器人上使用 Gazebo 模拟器进行了验证。在大量轨迹上，VAICAM 的性能优于两个基线控制器。 

---
# Input-gated Bilateral Teleoperation: An Easy-to-implement Force Feedback Teleoperation Method for Low-cost Hardware 

**Title (ZH)**: 输入门控双边遥控：一种易实现的低成本硬件力反馈遥控方法 

**Authors**: Yoshiki Kanai, Akira Kanazawa, Hideyuki Ichiwara, Hiroshi Ito, Naoaki Noguchi, Tetsuya Ogata  

**Link**: [PDF](https://arxiv.org/pdf/2509.08226)  

**Abstract**: Effective data collection in contact-rich manipulation requires force feedback during teleoperation, as accurate perception of contact is crucial for stable control. However, such technology remains uncommon, largely because bilateral teleoperation systems are complex and difficult to implement. To overcome this, we propose a bilateral teleoperation method that relies only on a simple feedback controller and does not require force sensors. The approach is designed for leader-follower setups using low-cost hardware, making it broadly applicable. Through numerical simulations and real-world experiments, we demonstrate that the method requires minimal parameter tuning, yet achieves both high operability and contact stability, outperforming conventional approaches. Furthermore, we show its high robustness: even at low communication cycle rates between leader and follower, control performance degradation is minimal compared to high-speed operation. We also prove our method can be implemented on two types of commercially available low-cost hardware with zero parameter adjustments. This highlights its high ease of implementation and versatility. We expect this method will expand the use of force feedback teleoperation systems on low-cost hardware. This will contribute to advancing contact-rich task autonomy in imitation learning. 

**Abstract (ZH)**: 有效的接触丰富操作中的数据采集需要遥操作过程中提供力反馈，以确保稳定的控制，而准确感知接触至关重要。然而，这项技术仍然相对罕见，部分原因是双边遥操作系统复杂且难以实现。为克服这一问题，我们提出了一种仅依赖于简单反馈控制器且不需要力传感器的双边遥操作方法。该方法适用于低成本硬件的领导者-追随者设置，具有广泛适用性。通过数值仿真和实际实验，我们展示了该方法需要极少的参数调整，但仍然实现了高操作性和接触稳定性，优于传统方法。此外，我们还表明其具有高鲁棒性：即使在领导者和追随者之间通信周期率较低的情况下，其控制性能的下降也极为有限，与高速操作相比几乎没有性能损失。我们还证明该方法可以在两种商用低成本硬件上实现，无需参数调整。这凸显了其实施的简便性和通用性。我们预期该方法将促进力反馈遥操作系统在低成本硬件上的应用，为模仿学习中的接触丰富任务自主性发展做出贡献。 

---
# Mean Field Game-Based Interactive Trajectory Planning Using Physics-Inspired Unified Potential Fields 

**Title (ZH)**: 基于物理启发统一 Potential Fields 的均场游戏化交互轨迹规划 

**Authors**: Zhen Tian, Fujiang Yuan, Chunhong Yuan, Yanhong Peng  

**Link**: [PDF](https://arxiv.org/pdf/2509.08147)  

**Abstract**: Interactive trajectory planning in autonomous driving must balance safety, efficiency, and scalability under heterogeneous driving behaviors. Existing methods often face high computational cost or rely on external safety critics. To address this, we propose an Interaction-Enriched Unified Potential Field (IUPF) framework that fuses style-dependent benefit and risk fields through a physics-inspired variational model, grounded in mean field game theory. The approach captures conservative, aggressive, and cooperative behaviors without additional safety modules, and employs stochastic differential equations to guarantee Nash equilibrium with exponential convergence. Simulations on lane changing and overtaking scenarios show that IUPF ensures safe distances, generates smooth and efficient trajectories, and outperforms traditional optimization and game-theoretic baselines in both adaptability and computational efficiency. 

**Abstract (ZH)**: 自主驾驶中的交互轨迹规划必须在异构驾驶行为下平衡安全、效率和扩展性。为此，我们提出了一种基于物理启发的变分模型融合风格依赖的益处和风险场的交互增强统一势场（IUPF）框架，该框架基于均场博弈理论，无需额外的安全模块即可捕捉保守、激进和合作行为，并利用随机微分方程确保指数收敛的纳什均衡。模拟结果显示，IUPF能够确保安全距离，生成平滑高效的轨迹，并在适应性和计算效率方面优于传统优化和博弈论baseline方法。 

---
# Real-Time Obstacle Avoidance for a Mobile Robot Using CNN-Based Sensor Fusion 

**Title (ZH)**: 基于CNN的传感器融合的移动机器人实时避障研究 

**Authors**: Lamiaa H. Zain, Raafat E. Shalaby  

**Link**: [PDF](https://arxiv.org/pdf/2509.08095)  

**Abstract**: Obstacle avoidance is a critical component of the navigation stack required for mobile robots to operate effectively in complex and unknown environments. In this research, three end-to-end Convolutional Neural Networks (CNNs) were trained and evaluated offline and deployed on a differential-drive mobile robot for real-time obstacle avoidance to generate low-level steering commands from synchronized color and depth images acquired by an Intel RealSense D415 RGB-D camera in diverse environments. Offline evaluation showed that the NetConEmb model achieved the best performance with a notably low MedAE of $0.58 \times 10^{-3}$ rad/s. In comparison, the lighter NetEmb architecture adopted in this study, which reduces the number of trainable parameters by approximately 25\% and converges faster, produced comparable results with an RMSE of $21.68 \times 10^{-3}$ rad/s, close to the $21.42 \times 10^{-3}$ rad/s obtained by NetConEmb. Real-time navigation further confirmed NetConEmb's robustness, achieving a 100\% success rate in both known and unknown environments, while NetEmb and NetGated succeeded only in navigating the known environment. 

**Abstract (ZH)**: 移动机器人在复杂未知环境中的障碍避让是导航堆栈的关键组成部分。本研究中，针对Intel RealSense D415 RGB-D相机获取的同步颜色和深度图像，在多种环境中训练并评估了三个端到端卷积神经网络（CNN），并在差速驱动移动机器人上实时部署以生成低级转向命令，实现障碍避让。在线下评估中，NetConEmb模型性能最优，其中位绝对误差（MedAE）为$0.58 \times 10^{-3}$ rad/s。相比之下，本研究采用的更轻量级的NetEmb架构通过减少约25%的可训练参数并更快收敛，其均方根误差（RMSE）为$21.68 \times 10^{-3}$ rad/s，接近NetConEmb的$21.42 \times 10^{-3}$ rad/s。实时导航进一步验证了NetConEmb的 robust性，在已知和未知环境中均实现了100%的成功率，而NetEmb和NetGated仅能在已知环境中导航成功。 

---
# A Novel Theoretical Approach on Micro-Nano Robotic Networks Based on Density Matrices and Swarm Quantum Mechanics 

**Title (ZH)**: 基于密度矩阵和群量子力学的新型微纳米机器人网络理论方法 

**Authors**: Maria Mannone, Mahathi Anand, Peppino Fazio, Abdalla Swikir  

**Link**: [PDF](https://arxiv.org/pdf/2509.08002)  

**Abstract**: In a robotic swarm, parameters such as position and proximity to the target can be described in terms of probability amplitudes. This idea led to recent studies on a quantum approach to the definition of the swarm, including a block-matrix representation. Here, we propose an advancement of the idea, defining a swarm as a mixed quantum state, to be described with a density matrix, whose size does not change with the number of robots. We end the article with some directions for future research. 

**Abstract (ZH)**: 在机器人集群中，位置和目标的接近程度等参数可以用概率幅来描述。这一思想导致了对基于量子方法定义集群的研究，包括使用块矩阵表示法。在此，我们提出进一步的发展，将集群定义为混合量子态，并用密度矩阵描述，其尺寸不随机器人数量的变化而变化。文章最后提出了未来研究的方向。 

---
# Planar Juggling of a Devil-Stick using Discrete VHCs 

**Title (ZH)**: 使用离散VHCs的平面 Baton 魔术抛接 

**Authors**: Aakash Khandelwal, Ranjan Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2509.08085)  

**Abstract**: Planar juggling of a devil-stick using impulsive inputs is addressed using the concept of discrete virtual holonomic constraints (DVHC). The location of the center-of-mass of the devil-stick is specified in terms of its orientation at the discrete instants when impulsive control inputs are applied. The discrete zero dynamics (DZD) resulting from the choice of DVHC provides conditions for stable juggling. A control design that enforces the DVHC and an orbit stabilizing controller are presented. The approach is validated in simulation. 

**Abstract (ZH)**: 使用离散虚拟 holonomic 约束（DVHC）的鞭花样棒平面杂耍的冲量控制 

---
