# Fast Bilateral Teleoperation and Imitation Learning Using Sensorless Force Control via Accurate Dynamics Model 

**Title (ZH)**: 基于准确动力学模型的无传感器力控制快速双边远程操作及模仿学习 

**Authors**: Koki Yamane, Yunhan Li, Masashi Konosu, Koki Inami, Junji Oaki, Sho Sakaino, Toshiaki Tsuji  

**Link**: [PDF](https://arxiv.org/pdf/2507.06174)  

**Abstract**: In recent years, the advancement of imitation learning has led to increased interest in teleoperating low-cost manipulators to collect demonstration data. However, most existing systems rely on unilateral control, which only transmits target position values. While this approach is easy to implement and suitable for slow, non-contact tasks, it struggles with fast or contact-rich operations due to the absence of force feedback. This work demonstrates that fast teleoperation with force feedback is feasible even with force-sensorless, low-cost manipulators by leveraging 4-channel bilateral control. Based on accurately identified manipulator dynamics, our method integrates nonlinear terms compensation, velocity and external force estimation, and variable gain corresponding to inertial variation. Furthermore, using data collected by 4-channel bilateral control, we show that incorporating force information into both the input and output of learned policies improves performance in imitation learning. These results highlight the practical effectiveness of our system for high-fidelity teleoperation and data collection on affordable hardware. 

**Abstract (ZH)**: 近年来，模拟学习的进步激发了对利用低成本操作机构进行演示数据收集的远程操作技术的兴趣。然而，现有的大多数系统依赖于单向控制，仅传输目标位置值。尽管这种方法易于实现且适合缓慢的非接触任务，但在处理快速或接触密集的操作时会因缺乏力反馈而显得力不从心。本文展示了即使使用无传感器、低成本的操作机构，通过利用四通道双向控制，也能够实现带有力反馈的快速远程操作。基于精确识别的操作机构动态，该方法结合了非线性项补偿、速度和外部力估计以及与惯性变化相应的可变增益。此外，通过四通道双向控制收集的数据表明，在学习策略的输入和输出中融入力信息可以提高模拟学习的性能。这些结果突显了该系统在经济型硬件上实现高保真远程操作和数据收集的实际有效性。 

---
# Learning Agile Tensile Perching for Aerial Robots from Demonstrations 

**Title (ZH)**: 从示范学习敏捷拉伸着陆的空中机器人 

**Authors**: Kangle Yuan, Atar Babgei, Luca Romanello, Hai-Nguyen Nguyen, Ronald Clark, Mirko Kovac, Sophie F. Armanini, Basaran Bahadir Kocer  

**Link**: [PDF](https://arxiv.org/pdf/2507.06172)  

**Abstract**: Perching on structures such as trees, beams, and ledges is essential for extending the endurance of aerial robots by enabling energy conservation in standby or observation modes. A tethered tensile perching mechanism offers a simple, adaptable solution that can be retrofitted to existing robots and accommodates a variety of structure sizes and shapes. However, tethered tensile perching introduces significant modelling challenges which require precise management of aerial robot dynamics, including the cases of tether slack & tension, and momentum transfer. Achieving smooth wrapping and secure anchoring by targeting a specific tether segment adds further complexity. In this work, we present a novel trajectory framework for tethered tensile perching, utilizing reinforcement learning (RL) through the Soft Actor-Critic from Demonstrations (SACfD) algorithm. By incorporating both optimal and suboptimal demonstrations, our approach enhances training efficiency and responsiveness, achieving precise control over position and velocity. This framework enables the aerial robot to accurately target specific tether segments, facilitating reliable wrapping and secure anchoring. We validate our framework through extensive simulation and real-world experiments, and demonstrate effectiveness in achieving agile and reliable trajectory generation for tensile perching. 

**Abstract (ZH)**: 树木、梁柱和凸出部位等结构上的悬挂对于扩展空中机器人的续航能力至关重要，可以通过在待机或观察模式下节约能量来实现。受限于缆绳的张力悬挂机制提供了一种简单的、可适应现有机器人并能兼容多种结构尺寸和形状的解决方案。然而，受限于缆绳的张力悬挂带来了显著的建模挑战，需要精确管理空中机器人的动力学，包括缆绳松弛与张紧、动量传递等情况。通过瞄准特定的缆绳段进行平滑包裹和牢固锚定增加了进一步的复杂性。在本文中，我们提出了一种新的轨迹框架，利用Soft Actor-Critic from Demonstrations (SACfD) 算法通过强化学习（RL）来实现受限张力悬挂。通过结合最优和次优示范，我们的方法提高了训练效率和响应性，实现了对位置和速度的精确控制。该框架使空中机器人能够准确瞄准特定的缆绳段，促进可靠的包裹和牢固锚定。我们通过广泛的仿真和实际实验验证了该框架，并展示了其在实现敏捷可靠的受限张力悬挂轨迹生成方面的有效性。 

---
# Fast and Accurate Collision Probability Estimation for Autonomous Vehicles using Adaptive Sigma-Point Sampling 

**Title (ZH)**: 基于自适应sigma点采样的自主车辆快速准确碰撞概率估计 

**Authors**: Charles Champagne Cossette, Taylor Scott Clawson, Andrew Feit  

**Link**: [PDF](https://arxiv.org/pdf/2507.06149)  

**Abstract**: A novel algorithm is presented for the estimation of collision probabilities between dynamic objects with uncertain trajectories, where the trajectories are given as a sequence of poses with Gaussian distributions. We propose an adaptive sigma-point sampling scheme, which ultimately produces a fast, simple algorithm capable of estimating the collision probability with a median error of 3.5%, and a median runtime of 0.21ms, when measured on an Intel Xeon Gold 6226R Processor. Importantly, the algorithm explicitly accounts for the collision probability's temporal dependence, which is often neglected in prior work and otherwise leads to an overestimation of the collision probability. Finally, the method is tested on a diverse set of relevant real-world scenarios, consisting of 400 6-second snippets of autonomous vehicle logs, where the accuracy and latency is rigorously evaluated. 

**Abstract (ZH)**: 一种新的算法用于估计具有不确定轨迹的动力物体之间的碰撞概率，其中轨迹表示为具有高斯分布的姿态序列。我们提出了一种自适应sigma点抽样方案，最终产生一个快速、简单的算法，能够在Intel Xeon Gold 6226R处理器上以中位数误差3.5%和中位数运行时间0.21ms的速度估计碰撞概率。该算法显式地考虑了碰撞概率的时间依赖性，这在先前的工作中经常被忽略，否则会导致碰撞概率的高估。最后，该方法在400个包含自主车辆日志的6秒片段的多样化实际场景中进行了测试，其中准确性和延迟得到了严格的评估。 

---
# AURA-CVC: Autonomous Ultrasound-guided Robotic Assistance for Central Venous Catheterization 

**Title (ZH)**: AURA-CVC: 自主超声引导机器人辅助中心静脉导管置入 

**Authors**: Deepak Raina, Lidia Al-Zogbi, Brian Teixeira, Vivek Singh, Ankur Kapoor, Thorsten Fleiter, Muyinatu A. Lediju Bell, Vinciya Pandian, Axel Krieger  

**Link**: [PDF](https://arxiv.org/pdf/2507.05979)  

**Abstract**: Purpose: Central venous catheterization (CVC) is a critical medical procedure for vascular access, hemodynamic monitoring, and life-saving interventions. Its success remains challenging due to the need for continuous ultrasound-guided visualization of a target vessel and approaching needle, which is further complicated by anatomical variability and operator dependency. Errors in needle placement can lead to life-threatening complications. While robotic systems offer a potential solution, achieving full autonomy remains challenging. In this work, we propose an end-to-end robotic-ultrasound-guided CVC pipeline, from scan initialization to needle insertion. Methods: We introduce a deep-learning model to identify clinically relevant anatomical landmarks from a depth image of the patient's neck, obtained using RGB-D camera, to autonomously define the scanning region and paths. Then, a robot motion planning framework is proposed to scan, segment, reconstruct, and localize vessels (veins and arteries), followed by the identification of the optimal insertion zone. Finally, a needle guidance module plans the insertion under ultrasound guidance with operator's feedback. This pipeline was validated on a high-fidelity commercial phantom across 10 simulated clinical scenarios. Results: The proposed pipeline achieved 10 out of 10 successful needle placements on the first attempt. Vessels were reconstructed with a mean error of 2.15 \textit{mm}, and autonomous needle insertion was performed with an error less than or close to 1 \textit{mm}. Conclusion: To our knowledge, this is the first robotic CVC system demonstrated on a high-fidelity phantom with integrated planning, scanning, and insertion. Experimental results show its potential for clinical translation. 

**Abstract (ZH)**: 目的：中心静脉导管放置（CVC）是血管通路、血流动力学监测和生命挽救性干预的 critical 医疗程序。由于需要持续的超声引导以可视化目标血管和进针路径，其成功率受到解剖变异性和操作者依赖性的挑战。针头放置错误可能导致危及生命的风险。虽然机器人系统可能提供解决方案，但实现完全自主操作仍具有挑战性。本文提出了一种从扫描初始化到针头插入的端到端的机器人-超声引导CVC流程。 

---
# FineGrasp: Towards Robust Grasping for Delicate Objects 

**Title (ZH)**: FineGrasp: 向向精细物件抓取的稳健性迈进 

**Authors**: Yun Du, Mengao Zhao, Tianwei Lin, Yiwei Jin, Chaodong Huang, Zhizhong Su  

**Link**: [PDF](https://arxiv.org/pdf/2507.05978)  

**Abstract**: Recent advancements in robotic grasping have led to its integration as a core module in many manipulation systems. For instance, language-driven semantic segmentation enables the grasping of any designated object or object part. However, existing methods often struggle to generate feasible grasp poses for small objects or delicate components, potentially causing the entire pipeline to fail. To address this issue, we propose a novel grasping method, FineGrasp, which introduces improvements in three key aspects. First, we introduce multiple network modifications to enhance the ability of to handle delicate regions. Second, we address the issue of label imbalance and propose a refined graspness label normalization strategy. Third, we introduce a new simulated grasp dataset and show that mixed sim-to-real training further improves grasp performance. Experimental results show significant improvements, especially in grasping small objects, and confirm the effectiveness of our system in semantic grasping. 

**Abstract (ZH)**: Recent advancements in robotic grasping have led to its integration as a core module in many manipulation systems. FineGrasp: Improvements in Handling Delicate Regions, Addressing Label Imbalance, and Enhancing Sim-to-Real Training for Semantic Grasping 

---
# Comparison of Path Planning Algorithms for Autonomous Vehicle Navigation Using Satellite and Airborne LiDAR Data 

**Title (ZH)**: 基于卫星和机载LiDAR数据的自主车辆导航路径规划算法比较 

**Authors**: Chang Liu, Zhexiong Xue, Tamas Sziranyi  

**Link**: [PDF](https://arxiv.org/pdf/2507.05884)  

**Abstract**: Autonomous vehicle navigation in unstructured environments, such as forests and mountainous regions, presents significant challenges due to irregular terrain and complex road conditions. This work provides a comparative evaluation of mainstream and well-established path planning algorithms applied to weighted pixel-level road networks derived from high-resolution satellite imagery and airborne LiDAR data. For 2D road-map navigation, where the weights reflect road conditions and terrain difficulty, A*, Dijkstra, RRT*, and a Novel Improved Ant Colony Optimization Algorithm (NIACO) are tested on the DeepGlobe satellite dataset. For 3D road-map path planning, 3D A*, 3D Dijkstra, RRT-Connect, and NIACO are evaluated using the Hamilton airborne LiDAR dataset, which provides detailed elevation information. All algorithms are assessed under identical start and end point conditions, focusing on path cost, computation time, and memory consumption. Results demonstrate that Dijkstra consistently offers the most stable and efficient performance in both 2D and 3D scenarios, particularly when operating on dense, pixel-level geospatial road-maps. These findings highlight the reliability of Dijkstra-based planning for static terrain navigation and establish a foundation for future research on dynamic path planning under complex environmental constraints. 

**Abstract (ZH)**: 无结构环境中自主车辆导航存在显著挑战，例如在森林和山区地区，由于不规则地形和复杂道路条件。本研究对主流且成熟的路径规划算法在高分辨率卫星图像和机载LiDAR数据衍生的加权像素级道路网络上的应用进行了比较评估。对于2D道路图导航，其中权重反映了道路条件和地形难度，A*、Dijkstra、RRT*和一种新型改进蚁群优化算法（NIACO）在DeepGlobe卫星数据集上进行测试。对于3D道路图路径规划，3D A*、3D Dijkstra、RRT-Connect和NIACO在提供详细高程信息的Hamilton机载LiDAR数据集上进行评估。所有算法均在相同的起始和终点条件下进行评估，重点关注路径成本、计算时间和内存消耗。研究结果表明，Dijkstra在2D和3D场景中均表现出最稳定和高效的性能，特别是在密集的像素级地理道路图上操作时。这些发现强调了基于Dijkstra的规划方法在静态地形导航中的可靠性，并为在复杂环境约束下的动态路径规划研究奠定了基础。 

---
# A Learning-based Planning and Control Framework for Inertia Drift Vehicles 

**Title (ZH)**: 基于学习的规划与控制框架：用于惯性漂移车辆 

**Authors**: Bei Zhou, Zhouheng Li, Lei Xie, Hongye Su, Johannes Betz  

**Link**: [PDF](https://arxiv.org/pdf/2507.05748)  

**Abstract**: Inertia drift is a transitional maneuver between two sustained drift stages in opposite directions, which provides valuable insights for navigating consecutive sharp corners for autonomous this http URL, this can be a challenging scenario for the drift controller to handle rapid transitions between opposing sideslip angles while maintaining accurate path tracking. Moreover, accurate drift control depends on a high-fidelity vehicle model to derive drift equilibrium points and predict vehicle states, but this is often compromised by the strongly coupled longitudinal-lateral drift dynamics and unpredictable environmental variations. To address these challenges, this paper proposes a learning-based planning and control framework utilizing Bayesian optimization (BO), which develops a planning logic to ensure a smooth transition and minimal velocity loss between inertia and sustained drift phases. BO is further employed to learn a performance-driven control policy that mitigates modeling errors for enhanced system performance. Simulation results on an 8-shape reference path demonstrate that the proposed framework can achieve smooth and stable inertia drift through sharp corners. 

**Abstract (ZH)**: 基于贝叶斯优化的自主过渡漂移规划与控制框架 

---
# Simultaneous Triggering and Synchronization of Sensors and Onboard Computers 

**Title (ZH)**: 传感器和机载计算机的同步触发与同步 

**Authors**: Morten Nissov, Nikhil Khedekar, Kostas Alexis  

**Link**: [PDF](https://arxiv.org/pdf/2507.05717)  

**Abstract**: High fidelity estimation algorithms for robotics require accurate data. However, timestamping of sensor data is a key issue that rarely receives the attention it deserves. Inaccurate timestamping can be compensated for in post-processing but is imperative for online estimation. Simultaneously, even online mitigation of timing issues can be achieved through a relaxation of the tuning parameters from their otherwise more performative optimal values, but at a detriment to performance. To address the need for real-time, low-cost timestamping, a versatile system which utilizes readily-available components and established methods for synchronization is introduced. The synchronization and triggering (of both high- and low-rate sensors) capabilities of the system are demonstrated. 

**Abstract (ZH)**: 基于机器人应用的高保真估计算法需要准确的数据。然而，传感器数据的时间戳标记是经常被忽视的关键问题。不准确的时间戳可以通过后处理进行补偿，但对于在线估计却至关重要。同时，即使通过放松从最优值更高效的调参值来在线缓解时间问题，性能也会受到影响。为了满足实时、低成本的时间戳需求，介绍了一种灵活的系统，该系统利用 readily-available 组件和已建立的同步方法。展示了该系统的同步和触发（高和低速率传感器）能力。 

---
# Stable Tracking-in-the-Loop Control of Cable-Driven Surgical Manipulators under Erroneous Kinematic Chains 

**Title (ZH)**: 基于错误运动链的电缆驱动手术 manipulators 在环稳定跟踪控制 

**Authors**: Neelay Joglekar, Fei Liu, Florian Richter, Michael C. Yip  

**Link**: [PDF](https://arxiv.org/pdf/2507.05663)  

**Abstract**: Remote Center of Motion (RCM) robotic manipulators have revolutionized Minimally Invasive Surgery, enabling precise, dexterous surgical manipulation within the patient's body cavity without disturbing the insertion point on the patient. Accurate RCM tool control is vital for incorporating autonomous subtasks like suturing, blood suction, and tumor resection into robotic surgical procedures, reducing surgeon fatigue and improving patient outcomes. However, these cable-driven systems are subject to significant joint reading errors, corrupting the kinematics computation necessary to perform control. Although visual tracking with endoscopic cameras can correct errors on in-view joints, errors in the kinematic chain prior to the insertion point are irreparable because they remain out of view. No prior work has characterized the stability of control under these conditions. We fill this gap by designing a provably stable tracking-in-the-loop controller for the out-of-view portion of the RCM manipulator kinematic chain. We additionally incorporate this controller into a bilevel control scheme for the full kinematic chain. We rigorously benchmark our method in simulated and real world settings to verify our theoretical findings. Our work provides key insights into the next steps required for the transition from teleoperated to autonomous surgery. 

**Abstract (ZH)**: Remote Center of Motion (RCM) 轨道 manipulator 控制的稳健性：从可视化关节到不可视化关节的端到端自主手术控制 

---
# Feature Geometry for Stereo Sidescan and Forward-looking Sonar 

**Title (ZH)**: 声纳侧视和前视声纳的特征几何学 

**Authors**: Kalin Norman, Joshua G. Mangelson  

**Link**: [PDF](https://arxiv.org/pdf/2507.05410)  

**Abstract**: In this paper, we address stereo acoustic data fusion for marine robotics and propose a geometry-based method for projecting observed features from one sonar to another for a cross-modal stereo sonar setup that consists of both a forward-looking and a sidescan sonar. Our acoustic geometry for sidescan and forward-looking sonar is inspired by the epipolar geometry for stereo cameras, and we leverage relative pose information to project where an observed feature in one sonar image will be found in the image of another sonar. Additionally, we analyze how both the feature location relative to the sonar and the relative pose between the two sonars impact the projection. From simulated results, we identify desirable stereo configurations for applications in field robotics like feature correspondence and recovery of the 3D information of the feature. 

**Abstract (ZH)**: 本文针对海洋机器人提出了一种立体声学数据融合方法，并提出了一种基于几何的方法，将前视声纳和侧扫声纳（一个向前观测，一个向侧面观测）组成的跨模态立体声纳系统中观测到的特征投影到另一声纳的图像上。我们的声纳几何学受到立体摄像机的极线几何学的启发，并利用相对姿态信息预测一个声纳图像中观测到的特征在另一个声纳图像中的位置。此外，我们分析了特征相对于声纳的位置及其俩声纳之间的相对姿态对投影的影响。通过仿真结果，我们确定了适用于场机器人领域的特征对应和特征三维信息恢复的理想立体声纳配置。 

---
# Assessing Linear Control Strategies for Zero-Speed Fin Roll Damping 

**Title (ZH)**: 评估零速度鳍卷阻尼的线性控制策略 

**Authors**: Nikita Savin, Elena Ambrosovskaya, Dmitry Romaev, Anton Proskurnikov  

**Link**: [PDF](https://arxiv.org/pdf/2507.05867)  

**Abstract**: Roll stabilization is a critical aspect of ship motion control, particularly for vessels operating in low-speed or zero-speed conditions, where traditional hydrodynamic fins lose their effectiveness. In this paper, we consider a roll damping system, developed by Navis JSC, based on two actively controlled zero-speed fins. Unlike conventional fin stabilizers, zero-speed fins employ a drag-based mechanism and active oscillations to generate stabilizing forces even when the vessel is stationary. We propose a simple linear control architecture that, however, accounts for nonlinear drag forces and actuator limitations. Simulation results on a high-fidelity vessel model used for HIL testing demonstrate the effectiveness of the proposed approach. 

**Abstract (ZH)**: 基于零速鳍片的卷制衰减系统在船舶低速或静止状态下的横摇抑制 

---
# Robotic System with AI for Real Time Weed Detection, Canopy Aware Spraying, and Droplet Pattern Evaluation 

**Title (ZH)**: 基于AI的实时杂草检测、冠层意识喷洒及液滴模式评估的机器人系统 

**Authors**: Inayat Rasool, Pappu Kumar Yadav, Amee Parmar, Hasan Mirzakhaninafchi, Rikesh Budhathoki, Zain Ul Abideen Usmani, Supriya Paudel, Ivan Perez Olivera, Eric Jone  

**Link**: [PDF](https://arxiv.org/pdf/2507.05432)  

**Abstract**: Uniform and excessive herbicide application in modern agriculture contributes to increased input costs, environmental pollution, and the emergence of herbicide resistant weeds. To address these challenges, we developed a vision guided, AI-driven variable rate sprayer system capable of detecting weed presence, estimating canopy size, and dynamically adjusting nozzle activation in real time. The system integrates lightweight YOLO11n and YOLO11n-seg deep learning models, deployed on an NVIDIA Jetson Orin Nano for onboard inference, and uses an Arduino Uno-based relay interface to control solenoid actuated nozzles based on canopy segmentation results. Indoor trials were conducted using 15 potted Hibiscus rosa sinensis plants of varying canopy sizes to simulate a range of weed patch scenarios. The YOLO11n model achieved a mean average precision (mAP@50) of 0.98, with a precision of 0.99 and a recall close to 1.0. The YOLO11n-seg segmentation model achieved a mAP@50 of 0.48, precision of 0.55, and recall of 0.52. System performance was validated using water sensitive paper, which showed an average spray coverage of 24.22% in zones where canopy was present. An upward trend in mean spray coverage from 16.22% for small canopies to 21.46% and 21.65% for medium and large canopies, respectively, demonstrated the system's capability to adjust spray output based on canopy size in real time. These results highlight the potential of combining real time deep learning with low-cost embedded hardware for selective herbicide application. Future work will focus on expanding the detection capabilities to include three common weed species in South Dakota: water hemp (Amaranthus tuberculatus), kochia (Bassia scoparia), and foxtail (Setaria spp.), followed by further validation in both indoor and field trials within soybean and corn production systems. 

**Abstract (ZH)**: 均匀且过度使用除草剂在现代农业中增加了输入成本、环境污染并促进了抗除草剂杂草的出现。为应对这些挑战，我们开发了一种基于视觉引导和AI驱动的变量率喷洒系统，能够检测杂草的存在、估计冠层大小，并实时动态调整喷头激活。该系统结合了轻量级YOLO11n和YOLO11n-seg深度学习模型，并在NVIDIA Jetson Orin Nano上进行板载推断，使用基于Arduino Uno的继电器接口根据冠层分割结果控制电磁阀喷头。室内试验使用15株不同冠层大小的木槿（Hibiscus rosa-sinensis）盆栽植物模拟了一系列杂草斑块情景。YOLO11n模型的平均平均精度（mAP@50）为0.98，精确度为0.99，召回率接近1.0。YOLO11n-seg分割模型的mAP@50为0.48，精确度为0.55，召回率为0.52。系统性能使用水敏纸验证，显示在冠层区域平均喷洒覆盖率约为24.22%。冠层从小到大（分别为小冠层16.22%、中冠层21.46%和大冠层21.65%）喷洒覆盖率的上升趋势表明该系统能够根据冠层大小实时调整喷洒输出。这些结果突显了结合实时深度学习和低成本嵌入式硬件进行选择性除草剂应用的潜力。未来工作将致力于扩大检测能力，纳入南达科他州三种常见杂草物种：水棘豆（Amaranthus tuberculatus）、emade草（Bassia scoparia）和狗尾草（Setaria spp.），并通过进一步在大豆和玉米生产系统中的室内和田间试验进行验证。 

---
