# Loop closure grasping: Topological transformations enable strong, gentle, and versatile grasps 

**Title (ZH)**: 环状闭合抓取：拓扑变换实现稳健、温柔且多功能的抓取 

**Authors**: Kentaro Barhydt, O. Godson Osele, Sreela Kodali, Cosima du Pasquier, Chase M. Hartquist, H. Harry Asada, Allison M. Okamura  

**Link**: [PDF](https://arxiv.org/pdf/2505.10552)  

**Abstract**: Grasping mechanisms must both create and subsequently hold grasps that permit safe and effective object manipulation. Existing mechanisms address the different functional requirements of grasp creation and grasp holding using a single morphology, but have yet to achieve the simultaneous strength, gentleness, and versatility needed for many applications. We present "loop closure grasping", a class of robotic grasping that addresses these different functional requirements through topological transformations between open-loop and closed-loop morphologies. We formalize these morphologies for grasping, formulate the loop closure grasping method, and present principles and a design architecture that we implement using soft growing inflated beams, winches, and clamps. The mechanisms' initial open-loop topology enables versatile grasp creation via unencumbered tip movement, and closing the loop enables strong and gentle holding with effectively infinite bending compliance. Loop closure grasping circumvents the tradeoffs of single-morphology designs, enabling grasps involving historically challenging objects, environments, and configurations. 

**Abstract (ZH)**: 基于拓扑变换的闭环抓取 

---
# AutoCam: Hierarchical Path Planning for an Autonomous Auxiliary Camera in Surgical Robotics 

**Title (ZH)**: AutoCam: 外科机器人中自主辅助相机的分层路径规划 

**Authors**: Alexandre Banks, Randy Moore, Sayem Nazmuz Zaman, Alaa Eldin Abdelaal, Septimiu E. Salcudean  

**Link**: [PDF](https://arxiv.org/pdf/2505.10398)  

**Abstract**: Incorporating an autonomous auxiliary camera into robot-assisted minimally invasive surgery (RAMIS) enhances spatial awareness and eliminates manual viewpoint control. Existing path planning methods for auxiliary cameras track two-dimensional surgical features but do not simultaneously account for camera orientation, workspace constraints, and robot joint limits. This study presents AutoCam: an automatic auxiliary camera placement method to improve visualization in RAMIS. Implemented on the da Vinci Research Kit, the system uses a priority-based, workspace-constrained control algorithm that combines heuristic geometric placement with nonlinear optimization to ensure robust camera tracking. A user study (N=6) demonstrated that the system maintained 99.84% visibility of a salient feature and achieved a pose error of 4.36 $\pm$ 2.11 degrees and 1.95 $\pm$ 5.66 mm. The controller was computationally efficient, with a loop time of 6.8 $\pm$ 12.8 ms. An additional pilot study (N=6), where novices completed a Fundamentals of Laparoscopic Surgery training task, suggests that users can teleoperate just as effectively from AutoCam's viewpoint as from the endoscope's while still benefiting from AutoCam's improved visual coverage of the scene. These results indicate that an auxiliary camera can be autonomously controlled using the da Vinci patient-side manipulators to track a salient feature, laying the groundwork for new multi-camera visualization methods in RAMIS. 

**Abstract (ZH)**: 将自主辅助相机集成到机器人辅助微创手术（RAMIS）中，增强空间感知并消除手动视角控制。现有的辅助相机路径规划方法追踪二维手术特征，但未同时考虑相机姿态、工作空间约束和机器人关节限制。本研究提出AutoCam：一种自动辅助相机定位方法，以提高RAMIS中的可视化效果。该系统基于达芬奇研究套件实现，使用基于优先级的工作空间约束控制算法，结合启发式几何定位和非线性优化，确保相机跟踪的鲁棒性。用户研究（N=6）表明，系统保持了99.84%的重要特征可视性，实现了姿态误差4.36±2.11度和1.95±5.66毫米。控制器计算效率高，循环时间为6.8±12.8毫秒。此外，初步试验（N=6）显示，新手可以在使用AutoCam视角进行腹腔镜手术基础培训任务时，依然能够有效地进行远程操作，同时受益于AutoCam改善的场景视觉覆盖。这些结果表明，可以通过达芬奇术野 manipulators 自主控制辅助相机跟踪重要特征，为RAMIS中的多相机可视化方法奠定了基础。 

---
# pc-dbCBS: Kinodynamic Motion Planning of Physically-Coupled Robot Teams 

**Title (ZH)**: 物理耦合机器人团队的 kinodynamic 运动规划 

**Authors**: Khaled Wahba, Wolfgang Hönig  

**Link**: [PDF](https://arxiv.org/pdf/2505.10355)  

**Abstract**: Motion planning problems for physically-coupled multi-robot systems in cluttered environments are challenging due to their high dimensionality. Existing methods combining sampling-based planners with trajectory optimization produce suboptimal results and lack theoretical guarantees. We propose Physically-coupled discontinuity-bounded Conflict-Based Search (pc-dbCBS), an anytime kinodynamic motion planner, that extends discontinuity-bounded CBS to rigidly-coupled systems. Our approach proposes a tri-level conflict detection and resolution framework that includes the physical coupling between the robots. Moreover, pc-dbCBS alternates iteratively between state space representations, thereby preserving probabilistic completeness and asymptotic optimality while relying only on single-robot motion primitives. Across 25 simulated and six real-world problems involving multirotors carrying a cable-suspended payload and differential-drive robots linked by rigid rods, pc-dbCBS solves up to 92% more instances than a state-of-the-art baseline and plans trajectories that are 50-60% faster while reducing planning time by an order of magnitude. 

**Abstract (ZH)**: 物理耦合多机器人系统在复杂环境中的运动规划问题由于其高维性具有挑战性。现有方法结合基于采样的规划器与轨迹优化会产生次优化的结果并缺乏理论保证。我们提出了物理耦合的断点受限冲突基于搜索（pc-dbCBS），这是一种可随时使用的动力学运动规划器，将断点受限冲突基于搜索扩展到刚性耦合系统。我们的方法提出了一种包含机器人之间物理耦合的三级冲突检测与解决框架。此外，pc-dbCBS 通过迭代地交替使用状态空间表示，从而保持概率完备性和渐近优化性，同时仅依赖单机器人运动基元。在包括携带悬挂载荷的多旋翼和通过刚性杆链接的差分驱动机器人在内的25个模拟问题和6个真实世界问题中，pc-dbCBS 比最先进的基线解决了多出92%的实例，并规划了快50-60%的路径，同时将规划时间缩短了一个数量级。 

---
# Context-aware collaborative pushing of heavy objects using skeleton-based intention prediction 

**Title (ZH)**: 基于骨架意图预测的上下文感知重物协作推拿 

**Authors**: Gokhan Solak, Gustavo J. G. Lahr, Idil Ozdamar, Arash Ajoudani  

**Link**: [PDF](https://arxiv.org/pdf/2505.10239)  

**Abstract**: In physical human-robot interaction, force feedback has been the most common sensing modality to convey the human intention to the robot. It is widely used in admittance control to allow the human to direct the robot. However, it cannot be used in scenarios where direct force feedback is not available since manipulated objects are not always equipped with a force sensor. In this work, we study one such scenario: the collaborative pushing and pulling of heavy objects on frictional surfaces, a prevalent task in industrial settings. When humans do it, they communicate through verbal and non-verbal cues, where body poses, and movements often convey more than words. We propose a novel context-aware approach using Directed Graph Neural Networks to analyze spatio-temporal human posture data to predict human motion intention for non-verbal collaborative physical manipulation. Our experiments demonstrate that robot assistance significantly reduces human effort and improves task efficiency. The results indicate that incorporating posture-based context recognition, either together with or as an alternative to force sensing, enhances robot decision-making and control efficiency. 

**Abstract (ZH)**: 基于物理的人机交互中，力反馈一直是最常用的传感模态以传达人类意图给机器人。它广泛应用于顺应控制，允许人类引导机器人。然而，在无法直接提供力反馈的场景中，由于操纵的对象并不总是配备有力传感器，这一方法无法使用。在本工作中，我们研究了这样一个场景：在摩擦表面上协作推拉重物，这是一个广泛存在于工业环境中的任务。当人类进行这项任务时，他们通过口头和非口头的线索进行沟通，肢体姿势和动作往往传达了更多意义。我们提出了一种新颖的基于上下文的定向图神经网络方法，以分析时空人类姿势数据来预测人类运动意图，以实现非言语协作物理操控。实验结果表明，机器人的协助显著减少了人类的努力，并提高了任务的效率。结果表明，结合或替代基于姿态的上下文识别，可以增强机器人的决策能力和控制效率。 

---
# Quad-LCD: Layered Control Decomposition Enables Actuator-Feasible Quadrotor Trajectory Planning 

**Title (ZH)**: Quad-LCD：分层控制分解实现可行的四旋翼飞行器轨迹规划 

**Authors**: Anusha Srikanthan, Hanli Zhang, Spencer Folk, Vijay Kumar, Nikolai Matni  

**Link**: [PDF](https://arxiv.org/pdf/2505.10228)  

**Abstract**: In this work, we specialize contributions from prior work on data-driven trajectory generation for a quadrotor system with motor saturation constraints. When motors saturate in quadrotor systems, there is an ``uncontrolled drift" of the vehicle that results in a crash. To tackle saturation, we apply a control decomposition and learn a tracking penalty from simulation data consisting of low, medium and high-cost reference trajectories. Our approach reduces crash rates by around $49\%$ compared to baselines on aggressive maneuvers in simulation. On the Crazyflie hardware platform, we demonstrate feasibility through experiments that lead to successful flights. Motivated by the growing interest in data-driven methods to quadrotor planning, we provide open-source lightweight code with an easy-to-use abstraction of hardware platforms. 

**Abstract (ZH)**: 本研究专注于数据驱动的轨迹生成对于四旋翼系统在电机饱和约束下的贡献，特别是处理电机饱和问题。通过控制分解和从包含低成本、中成本和高成本参考轨迹的模拟数据中学习跟踪惩罚，我们降低了四旋翼系统在模拟中激进操作下的相撞率约49%。在疯狂flie硬件平台上，通过实验展示了其实现可行性并成功飞行。为推动四旋翼飞行器规划中数据驱动方法的兴趣增长，我们提供了开源轻量级代码，并提供了易于使用的硬件平台抽象。 

---
# Multi-Robot Task Allocation for Homogeneous Tasks with Collision Avoidance via Spatial Clustering 

**Title (ZH)**: 基于空间聚类的同质任务多机器人任务分配与碰撞避免 

**Authors**: Rathin Chandra Shit, Sharmila Subudhi  

**Link**: [PDF](https://arxiv.org/pdf/2505.10073)  

**Abstract**: In this paper, a novel framework is presented that achieves a combined solution based on Multi-Robot Task Allocation (MRTA) and collision avoidance with respect to homogeneous measurement tasks taking place in industrial environments. The spatial clustering we propose offers to simultaneously solve the task allocation problem and deal with collision risks by cutting the workspace into distinguishable operational zones for each robot. To divide task sites and to schedule robot routes within corresponding clusters, we use K-means clustering and the 2-Opt algorithm. The presented framework shows satisfactory performance, where up to 93\% time reduction (1.24s against 17.62s) with a solution quality improvement of up to 7\% compared to the best performing method is demonstrated. Our method also completely eliminates collision points that persist in comparative methods in a most significant sense. Theoretical analysis agrees with the claim that spatial partitioning unifies the apparently disjoint tasks allocation and collision avoidance problems under conditions of many identical tasks to be distributed over sparse geographical areas. Ultimately, the findings in this work are of substantial importance for real world applications where both computational efficiency and operation free from collisions is of paramount importance. 

**Abstract (ZH)**: 一种结合多机器人任务分配与避撞的工业环境中同质测量任务框架 

---
# Evaluating Robustness of Deep Reinforcement Learning for Autonomous Surface Vehicle Control in Field Tests 

**Title (ZH)**: 评估深度强化学习在田间试验中对自主水面车辆控制健壮性的性能 

**Authors**: Luis F. W. Batista, Stéphanie Aravecchia, Seth Hutchinson, Cédric Pradalier  

**Link**: [PDF](https://arxiv.org/pdf/2505.10033)  

**Abstract**: Despite significant advancements in Deep Reinforcement Learning (DRL) for Autonomous Surface Vehicles (ASVs), their robustness in real-world conditions, particularly under external disturbances, remains insufficiently explored. In this paper, we evaluate the resilience of a DRL-based agent designed to capture floating waste under various perturbations. We train the agent using domain randomization and evaluate its performance in real-world field tests, assessing its ability to handle unexpected disturbances such as asymmetric drag and an off-center payload. We assess the agent's performance under these perturbations in both simulation and real-world experiments, quantifying performance degradation and benchmarking it against an MPC baseline. Results indicate that the DRL agent performs reliably despite significant disturbances. Along with the open-source release of our implementation, we provide insights into effective training strategies, real-world challenges, and practical considerations for deploying DRLbased ASV controllers. 

**Abstract (ZH)**: 尽管在自主水面车辆（ASVs）的深度强化学习（DRL）方面取得了显著进展，但它们在现实世界条件下的稳健性，尤其是在外来干扰下的表现，仍缺乏充分探索。本文评估了一种基于DRL的代理在各种干扰下的恢复能力，该代理旨在捕捉漂浮废弃物。我们使用领域随机化进行训练，并在实地测试中评估其性能，评估其处理非对称阻力和偏心载荷等意外干扰的能力。我们在模拟和实际实验中评估代理在这些干扰下的性能，量化性能退化，并将其基准与MPC基线进行比较。结果表明，即使在显著干扰下，DRL代理也能可靠地工作。除了开源发布我们的实现外，我们还提供了有效训练策略、现实挑战和部署基于DRL的ASV控制器的实用考虑的见解。 

---
# Fast Heuristic Scheduling and Trajectory Planning for Robotic Fruit Harvesters with Multiple Cartesian Arms 

**Title (ZH)**: 多 Cartesian 腕机器人水果收获机的快速启发式调度与轨迹规划 

**Authors**: Yuankai Zhu, Stavros Vougioukas  

**Link**: [PDF](https://arxiv.org/pdf/2505.10028)  

**Abstract**: This work proposes a fast heuristic algorithm for the coupled scheduling and trajectory planning of multiple Cartesian robotic arms harvesting fruits. Our method partitions the workspace, assigns fruit-picking sequences to arms, determines tight and feasible fruit-picking schedules and vehicle travel speed, and generates smooth, collision-free arm trajectories. The fruit-picking throughput achieved by the algorithm was assessed using synthetically generated fruit coordinates and a harvester design featuring up to 12 arms. The throughput increased monotonically as more arms were added. Adding more arms when fruit densities were low resulted in diminishing gains because it took longer to travel from one fruit to another. However, when there were enough fruits, the proposed algorithm achieved a linear speedup as the number of arms increased. 

**Abstract (ZH)**: 本研究提出了一种快速启发式算法，用于多笛卡尔机器人手臂联合调度和轨迹规划的果实采摘。该方法将工作空间划分为多个区域，分配果实采摘顺序给各手臂，确定紧致且可行的果实采摘时间表和车辆行驶速度，并生成平滑且无碰撞的手臂轨迹。通过合成生成的果实坐标和最多配备12个手臂的采摘器设计，评估了算法实现的果实采摘 throughput。随着手臂数量的增加， throughput 呈单调增加趋势。在果实密度较低时，增加更多手臂会导致效益递减，因为从一个果实到另一个果实的行驶时间变长。然而，当果实数量足够时，所提出的算法在手臂数量增加时实现了线性加速。 

---
# Hyper Yoshimura: How a slight tweak on a classical folding pattern unleashes meta-stability for deployable robots 

**Title (ZH)**: Hyper Yoshimura: 一个经典折叠图案的微小调整如何释放可部署机器人的亚稳态 

**Authors**: Ziyang Zhou, Yogesh Phalak, Vishrut Deshpande, Ian Walker, Suyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.09919)  

**Abstract**: Deployable structures inspired by origami offer lightweight, compact, and reconfigurable solutions for robotic and architectural applications. We present a geometric and mechanical framework for Yoshimura-Ori modules that supports a diverse set of metastable states, including newly identified asymmetric "pop-out" and "hyperfolded" configurations. These states are governed by three parameters -- tilt angle, phase shift, and slant height -- and enable discrete, programmable transformations. Using this model, we develop forward and inverse kinematic strategies to stack modules into deployable booms that approximate complex 3D shapes. We validate our approach through mechanical tests and demonstrate a tendon- and pneumatically-actuated Yoshimura Space Crane capable of object manipulation, solar tracking, and high load-bearing performance. A meter-scale solar charging station further illustrates the design's scalability. These results establish Yoshimura-Ori structures as a promising platform for adaptable, multifunctional deployable systems in both terrestrial and space environments. 

**Abstract (ZH)**: 仿 Origami 的可变形结构因其轻量化、紧凑化和可重构特性，在机器人和建筑应用中提供了解决方案。我们提出了一个几何和力学框架，支持 Yoshimura-Ori 模块的多种亚稳态配置，包括新发现的不对称“弹出”和“超折叠”配置。这些状态由三个参数——倾角、相位偏移和斜高——控制，并能使模块实现离散的、可编程的变换。使用此模型，我们开发了前向和逆向运动策略，将模块堆叠成可变形的臂架，以近似复杂的三维形状。我们通过机械测试验证了这种方法，并展示了可实现物体操作、太阳跟踪和高承载性能的腱驱动和气动驱动的 Yoshimura 空间起重机。进一步，一米规模的太阳能充电站演示了该设计的可扩展性。这些结果确立了 Yoshimura-Ori 结构作为适应性强、多用途可变形系统平台，在陆地和太空环境中的潜力。 

---
# EdgeAI Drone for Autonomous Construction Site Demonstrator 

**Title (ZH)**: 边缘AI无人机自主建筑工地演示器 

**Authors**: Emre Girgin, Arda Taha Candan, Coşkun Anıl Zaman  

**Link**: [PDF](https://arxiv.org/pdf/2505.09837)  

**Abstract**: The fields of autonomous systems and robotics are receiving considerable attention in civil applications such as construction, logistics, and firefighting. Nevertheless, the widespread adoption of these technologies is hindered by the necessity for robust processing units to run AI models. Edge-AI solutions offer considerable promise, enabling low-power, cost-effective robotics that can automate civil services, improve safety, and enhance sustainability. This paper presents a novel Edge-AI-enabled drone-based surveillance system for autonomous multi-robot operations at construction sites. Our system integrates a lightweight MCU-based object detection model within a custom-built UAV platform and a 5G-enabled multi-agent coordination infrastructure. We specifically target the real-time obstacle detection and dynamic path planning problem in construction environments, providing a comprehensive dataset specifically created for MCU-based edge applications. Field experiments demonstrate practical viability and identify optimal operational parameters, highlighting our approach's scalability and computational efficiency advantages compared to existing UAV solutions. The present and future roles of autonomous vehicles on construction sites are also discussed, as well as the effectiveness of edge-AI solutions. We share our dataset publicly at this http URL 

**Abstract (ZH)**: 自主系统和机器人技术在建筑、物流和消防等民用应用领域受到广泛关注。然而，这些技术的广泛采用受限于运行AI模型所需的 robust 处理单元。边缘AI解决方案显示出巨大潜力，能够实现低功耗、低成本的机器人，从而自动化公共服务、提高安全性和增强可持续性。本文介绍了用于建筑工地自主多机器人操作的新型边缘AI赋能无人机监视系统。我们的系统在自定义构建的无人机平台和5G使能的多智能体协调基础设施中集成了一个轻量级MCU基础的对象检测模型。我们特别针对建筑环境中的实时障碍检测和动态路径规划问题，提供了专门为MCU基础边缘应用创建的全面数据集。实地实验展示了其实用可行性，确定了最佳操作参数，并突出了与现有无人机解决方案相比，我们的方法在规模性和计算效率方面的优势。本文还讨论了自主车辆在建筑工地当前和未来的作用，以及边缘AI解决方案的有效性。我们已将数据集在此网址公开：[http://example.com]。 

---
# Learning Rock Pushability on Rough Planetary Terrain 

**Title (ZH)**: 在粗糙行星地形中学习岩石推移性 

**Authors**: Tuba Girgin, Emre Girgin, Cagri Kilic  

**Link**: [PDF](https://arxiv.org/pdf/2505.09833)  

**Abstract**: In the context of mobile navigation in unstructured environments, the predominant approach entails the avoidance of obstacles. The prevailing path planning algorithms are contingent upon deviating from the intended path for an indefinite duration and returning to the closest point on the route after the obstacle is left behind spatially. However, avoiding an obstacle on a path that will be used repeatedly by multiple agents can hinder long-term efficiency and lead to a lasting reliance on an active path planning system. In this study, we propose an alternative approach to mobile navigation in unstructured environments by leveraging the manipulation capabilities of a robotic manipulator mounted on top of a mobile robot. Our proposed framework integrates exteroceptive and proprioceptive feedback to assess the push affordance of obstacles, facilitating their repositioning rather than avoidance. While our preliminary visual estimation takes into account the characteristics of both the obstacle and the surface it relies on, the push affordance estimation module exploits the force feedback obtained by interacting with the obstacle via a robotic manipulator as the guidance signal. The objective of our navigation approach is to enhance the efficiency of routes utilized by multiple agents over extended periods by reducing the overall time spent by a fleet in environments where autonomous infrastructure development is imperative, such as lunar or Martian surfaces. 

**Abstract (ZH)**: 在非结构化环境中的移动导航中，主流的方法是避免障碍物。现有的路径规划算法依赖于偏离预定路径一段时间，并在离开障碍物后返回路径上的最近点。然而，对于将被多个代理反复使用的路径上的障碍物避障，可能会阻碍长期效率并导致对活跃路径规划系统的依赖。在本研究中，我们提出了一种利用安装在移动机器人顶部的机器人操作器操作能力的替代移动导航方法。我们提出的框架结合外部和内部反馈来评估障碍物的推搡可行性，从而促进障碍物的重新定位而非避免。虽然我们初步的视觉估计考虑了障碍物及其支撑表面的特性，但推搡可行性估计模块则利用与障碍物交互时通过机器人操作器获得的力反馈作为导向信号。我们的导航方法旨在通过减少在诸如月球或火星表面等需要自主基础设施发展的环境中，车队所花费的总体时间，从而提高多代理长期使用的路径的效率。 

---
# Grasp EveryThing (GET): 1-DoF, 3-Fingered Gripper with Tactile Sensing for Robust Grasping 

**Title (ZH)**: 全方位抓取（GET）：具备触觉感知的1-DoF三指 gripper 及其稳健抓取技术 

**Authors**: Michael Burgess, Edward H. Adelson  

**Link**: [PDF](https://arxiv.org/pdf/2505.09771)  

**Abstract**: We introduce the Grasp EveryThing (GET) gripper, a novel 1-DoF, 3-finger design for securely grasping objects of many shapes and sizes. Mounted on a standard parallel jaw actuator, the design features three narrow, tapered fingers arranged in a two-against-one configuration, where the two fingers converge into a V-shape. The GET gripper is more capable of conforming to object geometries and forming secure grasps than traditional designs with two flat fingers. Inspired by the principle of self-similarity, these V-shaped fingers enable secure grasping across a wide range of object sizes. Further to this end, fingers are parametrically designed for convenient resizing and interchangeability across robotic embodiments with a parallel jaw gripper. Additionally, we incorporate a rigid fingernail to enhance small object manipulation. Tactile sensing can be integrated into the standalone finger via an externally-mounted camera. A neural network was trained to estimate normal force from tactile images with an average validation error of 1.3~N across a diverse set of geometries. In grasping 15 objects and performing 3 tasks via teleoperation, the GET fingers consistently outperformed standard flat fingers. Finger designs for use with multiple robotic embodiments are available on GitHub. 

**Abstract (ZH)**: 一种新颖的单自由度三指夹持器：Grasp EveryThing (GET) 夹持器的设计与应用 

---
# Trailblazer: Learning offroad costmaps for long range planning 

**Title (ZH)**: Trailblazer: 学习用于长距离规划的离路成本图 

**Authors**: Kasi Viswanath, Felix Sanchez, Timothy Overbye, Jason M. Gregory, Srikanth Saripalli  

**Link**: [PDF](https://arxiv.org/pdf/2505.09739)  

**Abstract**: Autonomous navigation in off-road environments remains a significant challenge in field robotics, particularly for Unmanned Ground Vehicles (UGVs) tasked with search and rescue, exploration, and surveillance. Effective long-range planning relies on the integration of onboard perception systems with prior environmental knowledge, such as satellite imagery and LiDAR data. This work introduces Trailblazer, a novel framework that automates the conversion of multi-modal sensor data into costmaps, enabling efficient path planning without manual tuning. Unlike traditional approaches, Trailblazer leverages imitation learning and a differentiable A* planner to learn costmaps directly from expert demonstrations, enhancing adaptability across diverse terrains. The proposed methodology was validated through extensive real-world testing, achieving robust performance in dynamic and complex environments, demonstrating Trailblazer's potential for scalable, efficient autonomous navigation. 

**Abstract (ZH)**: 自主导航在非道路环境中的实现仍然是领域机器人领域的重大挑战，特别是在无人驾驶地面车辆（UGVs）进行搜索与救援、探索和监控任务时。有效的长距离规划依赖于车载感知系统与先验环境知识的集成，如卫星影像和LiDAR数据。本文介绍了Trailblazer，这是一种新型框架，能够自动化多模态传感器数据到成本地图的转换，从而实现高效的路径规划而无需手动调参。与传统方法不同，Trailblazer利用模仿学习和可微分A*规划器直接从专家演示中学习成本地图，增强了其在多样化地形上的适应性。所提出的方法通过广泛的实地测试得到了验证，表现出在动态和复杂环境中的稳健性能，证明了Trailblazer在可扩展和高效自主导航方面的潜力。 

---
