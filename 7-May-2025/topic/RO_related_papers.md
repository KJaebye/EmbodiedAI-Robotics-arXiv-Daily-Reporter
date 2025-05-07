# PyRoki: A Modular Toolkit for Robot Kinematic Optimization 

**Title (ZH)**: PyRoki: 一种模块化机器人运动学优化工具包 

**Authors**: Chung Min Kim, Brent Yi, Hongsuk Choi, Yi Ma, Ken Goldberg, Angjoo Kanazawa  

**Link**: [PDF](https://arxiv.org/pdf/2505.03728)  

**Abstract**: Robot motion can have many goals. Depending on the task, we might optimize for pose error, speed, collision, or similarity to a human demonstration. Motivated by this, we present PyRoki: a modular, extensible, and cross-platform toolkit for solving kinematic optimization problems. PyRoki couples an interface for specifying kinematic variables and costs with an efficient nonlinear least squares optimizer. Unlike existing tools, it is also cross-platform: optimization runs natively on CPU, GPU, and TPU. In this paper, we present (i) the design and implementation of PyRoki, (ii) motion retargeting and planning case studies that highlight the advantages of PyRoki's modularity, and (iii) optimization benchmarking, where PyRoki can be 1.4-1.7x faster and converges to lower errors than cuRobo, an existing GPU-accelerated inverse kinematics library. 

**Abstract (ZH)**: 机器人运动可以有多种目标。根据任务的不同，我们可能需要优化姿态误差、速度、碰撞或者与人类演示的相似度。基于这一点，我们介绍了PyRoki：一种模块化、扩展性强且跨平台的工具包，用于求解运动学优化问题。PyRoki 结合了一个用于指定运动学变量和代价的功能接口，以及一个高效非线性最小二乘优化器。与现有的工具不同，它也是跨平台的：优化可以直接在CPU、GPU和TPU上本地运行。在本文中，我们介绍了(i) PyRoki 的设计和实现，(ii) 动作重定位和规划案例研究，展示了PyRoki模块性的优势，以及(iii) 优化基准测试，在这些测试中，PyRoki 比现有的GPU加速逆运动学库cuRobo 快1.4-1.7倍，并且收敛到较低的误差。 

---
# Frenet Corridor Planner: An Optimal Local Path Planning Framework for Autonomous Driving 

**Title (ZH)**: Frenet 栅栏规划器：自主驾驶的最优局部路径规划框架 

**Authors**: Faizan M. Tariq, Zheng-Hang Yeh, Avinash Singh, David Isele, Sangjae Bae  

**Link**: [PDF](https://arxiv.org/pdf/2505.03695)  

**Abstract**: Motivated by the requirements for effectiveness and efficiency, path-speed decomposition-based trajectory planning methods have widely been adopted for autonomous driving applications. While a global route can be pre-computed offline, real-time generation of adaptive local paths remains crucial. Therefore, we present the Frenet Corridor Planner (FCP), an optimization-based local path planning strategy for autonomous driving that ensures smooth and safe navigation around obstacles. Modeling the vehicles as safety-augmented bounding boxes and pedestrians as convex hulls in the Frenet space, our approach defines a drivable corridor by determining the appropriate deviation side for static obstacles. Thereafter, a modified space-domain bicycle kinematics model enables path optimization for smoothness, boundary clearance, and dynamic obstacle risk minimization. The optimized path is then passed to a speed planner to generate the final trajectory. We validate FCP through extensive simulations and real-world hardware experiments, demonstrating its efficiency and effectiveness. 

**Abstract (ZH)**: 基于路径-速度分解的轨迹规划方法旨在满足自主驾驶应用中效果和效率的要求。尽管全局路线可以在线预计算，实时生成适应性的局部路径仍然至关重要。因此，我们提出了Frenét走廊规划器（FCP），这是一种基于优化的局部路径规划策略，确保在避开障碍物时实现平滑和安全的导航。在Frenét空间中，我们将车辆建模为安全增强的边界框，行人建模为凸包，我们的方法通过确定静态障碍物的适当偏离侧来定义可通行走廊。之后，修改后的空间域自行车动力学模型使路径优化能够实现平滑性、边界 clearance 和动态障碍物风险最小化。优化后的路径随后传递给速度规划器以生成最终轨迹。通过对FCP进行广泛的仿真实验和实地硬件实验，我们验证了其效率和有效性。 

---
# Demonstrating ViSafe: Vision-enabled Safety for High-speed Detect and Avoid 

**Title (ZH)**: 基于视觉的安全性展示：高speed检测与避免 

**Authors**: Parv Kapoor, Ian Higgins, Nikhil Keetha, Jay Patrikar, Brady Moon, Zelin Ye, Yao He, Ivan Cisneros, Yaoyu Hu, Changliu Liu, Eunsuk Kang, Sebastian Scherer  

**Link**: [PDF](https://arxiv.org/pdf/2505.03694)  

**Abstract**: Assured safe-separation is essential for achieving seamless high-density operation of airborne vehicles in a shared airspace. To equip resource-constrained aerial systems with this safety-critical capability, we present ViSafe, a high-speed vision-only airborne collision avoidance system. ViSafe offers a full-stack solution to the Detect and Avoid (DAA) problem by tightly integrating a learning-based edge-AI framework with a custom multi-camera hardware prototype designed under SWaP-C constraints. By leveraging perceptual input-focused control barrier functions (CBF) to design, encode, and enforce safety thresholds, ViSafe can provide provably safe runtime guarantees for self-separation in high-speed aerial operations. We evaluate ViSafe's performance through an extensive test campaign involving both simulated digital twins and real-world flight scenarios. By independently varying agent types, closure rates, interaction geometries, and environmental conditions (e.g., weather and lighting), we demonstrate that ViSafe consistently ensures self-separation across diverse scenarios. In first-of-its-kind real-world high-speed collision avoidance tests with closure rates reaching 144 km/h, ViSafe sets a new benchmark for vision-only autonomous collision avoidance, establishing a new standard for safety in high-speed aerial navigation. 

**Abstract (ZH)**: 确保安全分离是实现空中车辆在共享空域中无缝高密度运行的关键。为使资源受限的航空系统具备这一安全关键能力，我们提出了ViSafe，一种高速视觉_ONLY_空中碰撞避险系统。ViSafe通过将基于学习的边缘AI框架与在SWaP-C约束下设计的定制多摄像头硬件原型紧密集成，提供了一整套解决探测与避险(DAA)问题的方案。通过利用基于感知输入的关键障碍函数(CBF)进行设计、编码和实施安全阈值，ViSafe可以为高速空域操作中的自我分离提供可证明的安全运行保证。我们通过涵盖模拟数字孪生和真实飞行场景的广泛测试活动来评估ViSafe的性能。通过独立变化代理类型、闭合率、相互作用几何形状以及环境条件（例如天气和照明），我们证明了ViSafe能够在多种场景中一致地确保自我分离。在首次实现实高速碰撞避险测试中，闭合率达到144 km/h，ViSafe确立了视觉_ONLY_自主避险的新基准，并为高速空中导航建立了新的安全标准。 

---
# Thermal-LiDAR Fusion for Robust Tunnel Localization in GNSS-Denied and Low-Visibility Conditions 

**Title (ZH)**: GNSS受限制和低能见度条件下的热成像-LiDAR融合隧道稳固定位 

**Authors**: Lukas Schichler, Karin Festl, Selim Solmaz, Daniel Watzenig  

**Link**: [PDF](https://arxiv.org/pdf/2505.03565)  

**Abstract**: Despite significant progress in autonomous navigation, a critical gap remains in ensuring reliable localization in hazardous environments such as tunnels, urban disaster zones, and underground structures. Tunnels present a uniquely difficult scenario: they are not only prone to GNSS signal loss, but also provide little features for visual localization due to their repetitive walls and poor lighting. These conditions degrade conventional vision-based and LiDAR-based systems, which rely on distinguishable environmental features. To address this, we propose a novel sensor fusion framework that integrates a thermal camera with a LiDAR to enable robust localization in tunnels and other perceptually degraded environments. The thermal camera provides resilience in low-light or smoke conditions, while the LiDAR delivers precise depth perception and structural awareness. By combining these sensors, our framework ensures continuous and accurate localization across diverse and dynamic environments. We use an Extended Kalman Filter (EKF) to fuse multi-sensor inputs, and leverages visual odometry and SLAM (Simultaneous Localization and Mapping) techniques to process the sensor data, enabling robust motion estimation and mapping even in GNSS-denied environments. This fusion of sensor modalities not only enhances system resilience but also provides a scalable solution for cyber-physical systems in connected and autonomous vehicles (CAVs). To validate the framework, we conduct tests in a tunnel environment, simulating sensor degradation and visibility challenges. The results demonstrate that our method sustains accurate localization where standard approaches deteriorate due to the tunnels featureless geometry. The frameworks versatility makes it a promising solution for autonomous vehicles, inspection robots, and other cyber-physical systems operating in constrained, perceptually poor environments. 

**Abstract (ZH)**: 尽管在自主导航方面取得了显著进展，但在隧道、城市灾难区域和地下结构等危险环境中的可靠定位仍存在关键差距。隧道呈现了一个独特的难题：它们不仅容易失去GNSS信号，而且由于重复的墙面和照明不良，几乎没有可用于视觉定位的特征。这些条件会削弱依赖可区分环境特征的视觉和LiDAR系统。为了解决这一问题，我们提出了一种新颖的传感器融合框架，将热成像相机与LiDAR集成，以在隧道和其他感知退化的环境中实现稳健的定位。热成像相机在低光或烟雾条件下提供鲁棒性，而LiDAR提供精确的深度感知和结构意识。通过结合这些传感器，我们的框架确保在多种动态环境中持续且准确的定位。我们使用扩展卡尔曼滤波器（EKF）来融合多传感器输入，并利用视觉里程计和SLAM（同时定位与 mapping）技术处理传感器数据，使在GNSS受限环境中也能实现稳健的运动估计和建图。通过将不同类型的传感器数据相融合，该框架不仅提升了系统的鲁棒性，还为连接和自主车辆中的赛博物理系统提供了一个可扩展的解决方案。为了验证该框架，我们在隧道环境中进行了测试，模拟了传感器降级和能见度挑战。结果表明，我们的方法在隧道特征较少的几何结构中能够保持准确的定位，而标准方法则因隧道的特征较少而表现下降。该框架的通用性使其成为在约束和感知较差环境中操作的自主车辆、检查机器人和其他赛博物理系统的有前景的解决方案。 

---
# Automated Action Generation based on Action Field for Robotic Garment Manipulation 

**Title (ZH)**: 基于动作域的机器人服装操作自动动作生成 

**Authors**: Hu Cheng, Fuyuki Tokuda, Kazuhiro Kosuge  

**Link**: [PDF](https://arxiv.org/pdf/2505.03537)  

**Abstract**: Garment manipulation using robotic systems is a challenging task due to the diverse shapes and deformable nature of fabric. In this paper, we propose a novel method for robotic garment manipulation that significantly improves the accuracy while reducing computational time compared to previous approaches. Our method features an action generator that directly interprets scene images and generates pixel-wise end-effector action vectors using a neural network. The network also predicts a manipulation score map that ranks potential actions, allowing the system to select the most effective action. Extensive simulation experiments demonstrate that our method achieves higher unfolding and alignment performances and faster computation time than previous approaches. Real-world experiments show that the proposed method generalizes well to different garment types and successfully flattens garments. 

**Abstract (ZH)**: 使用机器人系统进行衣物 manipulation 是一项具有挑战性的任务，因为面料具有多样的形状和可变形的性质。本文提出了一种新的机器人衣物 manipulation 方法，与之前的approaches相比，该方法显著提高了准确性并减少了计算时间。该方法包括一个动作生成器，该生成器直接解释场景图像并使用神经网络生成像素级末端执行器动作向量。网络还预测一个操作分数图，对潜在操作进行排名，从而使系统能够选择最有效的操作。广泛的仿真实验表明，与之前的approaches相比，本文方法在开衣和对齐性能上更高，并且计算时间更快。实验证明，所提出的方法能够很好地适应不同类型的衣物，并成功使其展平。 

---
# Miniature multihole airflow sensor for lightweight aircraft over wide speed and angular range 

**Title (ZH)**: 轻型航空器宽速域和角范围微型多孔气流传感器 

**Authors**: Lukas Stuber, Simon Jeger, Raphael Zufferey, Dario Floreano  

**Link**: [PDF](https://arxiv.org/pdf/2505.03331)  

**Abstract**: An aircraft's airspeed, angle of attack, and angle of side slip are crucial to its safety, especially when flying close to the stall regime. Various solutions exist, including pitot tubes, angular vanes, and multihole pressure probes. However, current sensors are either too heavy (>30 g) or require large airspeeds (>20 m/s), making them unsuitable for small uncrewed aerial vehicles. We propose a novel multihole pressure probe, integrating sensing electronics in a single-component structure, resulting in a mechanically robust and lightweight sensor (9 g), which we released to the public domain. Since there is no consensus on two critical design parameters, tip shape (conical vs spherical) and hole spacing (distance between holes), we provide a study on measurement accuracy and noise generation using wind tunnel experiments. The sensor is calibrated using a multivariate polynomial regression model over an airspeed range of 3-27 m/s and an angle of attack/sideslip range of +-35°, achieving a mean absolute error of 0.44 m/s and 0.16°. Finally, we validated the sensor in outdoor flights near the stall regime. Our probe enabled accurate estimations of airspeed, angle of attack and sideslip during different acrobatic manoeuvres. Due to its size and weight, this sensor will enable safe flight for lightweight, uncrewed aerial vehicles flying at low speeds close to the stall regime. 

**Abstract (ZH)**: 一种新型多孔压力探头及其在轻小型无人机低速临近失速飞行中的气动参数精确测量与验证 

---
# Systematic Evaluation of Initial States and Exploration-Exploitation Strategies in PID Auto-Tuning: A Framework-Driven Approach Applied on Mobile Robots 

**Title (ZH)**: 基于框架驱动方法对PID自动 tuning初始状态和探索-利用策略的系统性评估：应用于移动机器人 

**Authors**: Zaid Ghazal, Ali Al-Bustami, Khouloud Gaaloul, Jaerock Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2505.03159)  

**Abstract**: PID controllers are widely used in control systems because of their simplicity and effectiveness. Although advanced optimization techniques such as Bayesian Optimization and Differential Evolution have been applied to address the challenges of automatic tuning of PID controllers, the influence of initial system states on convergence and the balance between exploration and exploitation remains underexplored. Moreover, experimenting the influence directly on real cyber-physical systems such as mobile robots is crucial for deriving realistic insights. In the present paper, a novel framework is introduced to evaluate the impact of systematically varying these factors on the PID auto-tuning processes that utilize Bayesian Optimization and Differential Evolution. Testing was conducted on two distinct PID-controlled robotic platforms, an omnidirectional robot and a differential drive mobile robot, to assess the effects on convergence rate, settling time, rise time, and overshoot percentage. As a result, the experimental outcomes yield evidence on the effects of the systematic variations, thereby providing an empirical basis for future research studies in the field. 

**Abstract (ZH)**: 一种新型框架：评估系统变化因素对基于贝叶斯优化和差分进化的PID自调谐过程的影响 

---
# Fabrication and Characterization of Additively Manufactured Stretchable Strain Sensors Towards the Shape Sensing of Continuum Robots 

**Title (ZH)**: 基于增材制造的可拉伸应变传感器制备及其在连续机器人形态感知中的应用 

**Authors**: Daniel C. Moyer, Wenpeng Wang, Logan S. Karschner, Loris Fichera, Pratap M. Rao  

**Link**: [PDF](https://arxiv.org/pdf/2505.03087)  

**Abstract**: This letter describes the manufacturing and experimental characterization of novel stretchable strain sensors for continuum robots. The overarching goal of this research is to provide a new solution for the shape sensing of these devices. The sensors are fabricated via direct ink writing, an extrusion-based additive manufacturing technique. Electrically conductive material (i.e., the \textit{ink}) is printed into traces whose electrical resistance varies in response to mechanical deformation. The principle of operation of stretchable strain sensors is analogous to that of conventional strain gauges, but with a significantly larger operational window thanks to their ability to withstand larger strain. Among the different conductive materials considered for this study, we opted to fabricate the sensors with a high-viscosity eutectic Gallium-Indium ink, which in initial testing exhibited high linearity ($R^2 \approx$ 0.99), gauge factor $\approx$ 1, and negligible drift. Benefits of the proposed sensors include (i) ease of fabrication, as they can be conveniently printed in a matter of minutes; (ii) ease of installation, as they can simply be glued to the outside body of a robot; (iii) ease of miniaturization, which enables integration into millimiter-sized continuum robots. 

**Abstract (ZH)**: 这种信件描述了新型可拉伸应变传感器的制造与实验表征，这些传感器用于连续机器人。本研究的总体目标是为这些设备提供新的形状感知解决方案。传感器通过直接墨水书写制造，这是一种基于挤出的增材制造技术。电导材料（即“墨水”）被打印成痕迹，其电气电阻会根据机械变形进行变化。可拉伸应变传感器的工作原理类似于传统的应变片，但由于其能够承受更大的应变，因此具有显著更大的工作窗口。在为本研究考虑的不同导电材料中，我们选择使用高黏度共晶镓铟墨水来制造传感器，在初始测试中，该传感器表现出高线性度（$R^2 \approx$ 0.99）、约1的灵敏度因子和可忽略不计的漂移。所提出传感器的优点包括：（i）易于制造，可以在几分钟内方便地打印；（ii）易于安装，可以直接粘贴在机器人外部；（iii）易于微型化，使其能够集成到毫米级的连续机器人中。 

---
# A Modal-Space Formulation for Momentum Observer Contact Estimation and Effects of Uncertainty for Continuum Robots 

**Title (ZH)**: 模态空间表示法在连续机器人接触估测中的动量观测器及其不确定性影响研究 

**Authors**: Garrison L.H. Johnston, Neel Shihora, Nabil Simaan  

**Link**: [PDF](https://arxiv.org/pdf/2505.03044)  

**Abstract**: Contact detection for continuum and soft robots has been limited in past works to statics or kinematics-based methods with assumed circular bending curvature or known bending profiles. In this paper, we adapt the generalized momentum observer contact estimation method to continuum robots. This is made possible by leveraging recent results for real-time shape sensing of continuum robots along with a modal-space representation of the robot dynamics. In addition to presenting an approach for estimating the generalized forces due to contact via a momentum observer, we present a constrained optimization method to identify the wrench imparted on the robot during contact. We also present an approach for investigating the effects of unmodeled deviations in the robot's dynamic state on the contact detection method and we validate our algorithm by simulations and experiments. We also compare the performance of the momentum observer to the joint force deviation method, a direct estimation approach using the robot's full dynamic model. We also demonstrate a basic extension of the method to multisegment continuum robots. Results presented in this work extend dynamic contact detection to the domain of continuum and soft robots and can be used to improve the safety of large-scale continuum robots for human-robot collaboration. 

**Abstract (ZH)**: 连续体和软体机器人中的接触检测在过去的研究中主要局限于基于静态或运动学的方法，这些方法假定弯曲曲率为圆形或已知弯曲轮廓。本文中，我们通过利用连续体机器人实时形状感知的最新结果以及机器人动力学的模态空间表示，将广义动量观察器接触估计方法应用于连续体机器人。除了提出一种通过动量观察器估计接触引起的广义力的方法外，我们还提出了一种约束优化方法，用于识别接触期间施加在机器人上的 wrench。此外，我们提出了一种方法来研究机器人动力学状态中未建模偏差对接触检测方法的影响，并通过仿真和实验验证了我们的算法。我们还将动量观察器的性能与关节力偏差方法进行了比较，后者是一种直接使用机器人完整动力学模型的估计方法。我们还展示了该方法的基本扩展，适用于多节段的连续体机器人。本文中提出的结果将动态接触检测扩展到了连续体和软体机器人的领域，可以用于提高大规模连续体机器人在人机协作中的安全性。 

---
# Zero-shot Sim2Real Transfer for Magnet-Based Tactile Sensor on Insertion Tasks 

**Title (ZH)**: 基于磁传感器的插入任务零样本Sim2Real迁移学习 

**Authors**: Beining Han, Abhishek Joshi, Jia Deng  

**Link**: [PDF](https://arxiv.org/pdf/2505.02915)  

**Abstract**: Tactile sensing is an important sensing modality for robot manipulation. Among different types of tactile sensors, magnet-based sensors, like u-skin, balance well between high durability and tactile density. However, the large sim-to-real gap of tactile sensors prevents robots from acquiring useful tactile-based manipulation skills from simulation data, a recipe that has been successful for achieving complex and sophisticated control policies. Prior work has implemented binarization techniques to bridge the sim-to-real gap for dexterous in-hand manipulation. However, binarization inherently loses much information that is useful in many other tasks, e.g., insertion. In our work, we propose GCS, a novel sim-to-real technique to learn contact-rich skills with dense, distributed, 3-axis tactile readings. We evaluate our approach on blind insertion tasks and show zero-shot sim-to-real transfer of RL policies with raw tactile reading as input. 

**Abstract (ZH)**: 基于触觉感知的机器人 manipulation 是一个重要传感模态。在不同类型的触觉传感器中，像 u-skin 这样的基于磁性的传感器在耐用性和触觉密度之间取得了良好的平衡。然而，触觉传感器中的仿真到现实的巨大差距阻碍了机器人通过仿真数据获取有用的基于触觉的 manipulation 技能，这一点在实现复杂和高级的控制策略方面已被证明是有效的。先前的工作通过实现二值化技术来解决灵巧的在手 manipulation 中的仿真到现实差距。然而，二值化会内在地丢失许多在其他任务中很重要的信息，例如插入。在我们的工作中，我们提出了 GCS，这是一种新颖的仿真到现实的技术，用于学习富含接触的技能，这些技能具有密集且分布在三轴触觉读取。我们在盲插入任务上评估了该方法，并展示了使用原始触觉读取作为输入的 RL 政策的零样本仿真到现实转移。 

---
# Model Predictive Fuzzy Control: A Hierarchical Multi-Agent Control Architecture for Outdoor Search-and-Rescue Robots 

**Title (ZH)**: 基于模型预测模糊控制的户外搜索救援机器人分层多代理控制架构 

**Authors**: Craig Maxwell, Mirko Baglioni, Anahita Jamshidnejad  

**Link**: [PDF](https://arxiv.org/pdf/2505.03257)  

**Abstract**: Autonomous robots deployed in unknown search-and-rescue (SaR) environments can significantly improve the efficiency of the mission by assisting in fast localisation and rescue of the trapped victims. We propose a novel integrated hierarchical control architecture, called model predictive fuzzy control (MPFC), for autonomous mission planning of multi-robot SaR systems that should efficiently map an unknown environment: We combine model predictive control (MPC) and fuzzy logic control (FLC), where the robots are locally controlled by computationally efficient FLC controllers, and the parameters of these local controllers are tuned via a centralised MPC controller, in a regular or event-triggered manner. The proposed architecture provides three main advantages: (1) The control decisions are made by the FLC controllers, thus the real-time computation time is affordable. (2) The centralised MPC controller optimises the performance criteria with a global and predictive vision of the system dynamics, and updates the parameters of the FLC controllers accordingly. (3) FLC controllers are heuristic by nature and thus do not take into account optimality in their decisions, while the tuned parameters via the MPC controller can indirectly incorporate some level of optimality in local decisions of the robots. A simulation environment for victim detection in a disaster environment was designed in MATLAB using discrete, 2-D grid-based models. While being comparable from the point of computational efficiency, the integrated MPFC architecture improves the performance of the multi-robot SaR system compared to decentralised FLC controllers. Moreover, the performance of MPFC is comparable to the performance of centralised MPC for path planning of SaR robots, whereas MPFC requires significantly less computational resources, since the number of the optimisation variables in the control problem are reduced. 

**Abstract (ZH)**: 自主部署在未知搜索与救援环境中的机器人可以通过快速定位和救援被困受害者显著提高任务效率。我们提出了一种新颖的集成分层控制架构，称为模型预测模糊控制（MPFC），用于多机器人搜索与救援系统自主任务规划，以高效地映射未知环境。该架构结合了模型预测控制（MPC）和模糊逻辑控制（FLC），其中机器人由计算高效的FLC控制器局部控制，这些局部控制器的参数通过集中式的MPC控制器进行定期或事件触发式的调优。提出的架构具有以下三大优势：（1）控制决策由FLC控制器做出，因此实时计算时间可承受。（2）集中式的MPC控制器以全局和预测的方式优化系统动力学的性能指标，并相应地调整FLC控制器的参数。（3）FLC控制器本质上是启发式的，因此在其决策中不考虑最优性，而通过MPC控制器调优后的参数可以在一定程度上间接地将最优性纳入到机器人局部决策中。在MATLAB中使用离散的二维网格模型设计了一个用于灾后受害者检测的仿真环境。虽然在计算效率上具有可比性，但集成的MPFC架构在多机器人搜索与救援系统的性能上优于分布式FLC控制器。此外，MPFC在搜索与救援机器人路径规划上的性能与集中式MPC相当，但需要显著较少的计算资源，因为控制问题中的优化变量数量减少了。 

---
