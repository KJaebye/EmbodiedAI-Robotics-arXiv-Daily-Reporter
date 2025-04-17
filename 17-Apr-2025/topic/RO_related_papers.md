# GripMap: An Efficient, Spatially Resolved Constraint Framework for Offline and Online Trajectory Planning in Autonomous Racing 

**Title (ZH)**: GripMap：一种高效的空间解析约束框架，用于自主赛车的离线和在线轨迹规划 

**Authors**: Frederik Werner, Ann-Kathrin Schwehn, Markus Lienkamp, Johannes Betz  

**Link**: [PDF](https://arxiv.org/pdf/2504.12115)  

**Abstract**: Conventional trajectory planning approaches for autonomous vehicles often assume a fixed vehicle model that remains constant regardless of the vehicle's location. This overlooks the critical fact that the tires and the surface are the two force-transmitting partners in vehicle dynamics; while the tires stay with the vehicle, surface conditions vary with location. Recognizing these challenges, this paper presents a novel framework for spatially resolving dynamic constraints in both offline and online planning algorithms applied to autonomous racing. We introduce the GripMap concept, which provides a spatial resolution of vehicle dynamic constraints in the Frenet frame, allowing adaptation to locally varying grip conditions. This enables compensation for location-specific effects, more efficient vehicle behavior, and increased safety, unattainable with spatially invariant vehicle models. The focus is on low storage demand and quick access through perfect hashing. This framework proved advantageous in real-world applications in the presented form. Experiments inspired by autonomous racing demonstrate its effectiveness. In future work, this framework can serve as a foundational layer for developing future interpretable learning algorithms that adjust to varying grip conditions in real-time. 

**Abstract (ZH)**: 基于空间分辨率的动力学约束框架在自主竞速中的应用 

---
# An Extended Generalized Prandtl-Ishlinskii Hysteresis Model for I2RIS Robot 

**Title (ZH)**: 扩展的广义普朗特-伊谢林斯基滞回模型用于I2RIS机器人 

**Authors**: Yiyao Yue, Mojtaba Esfandiari, Pengyuan Du, Peter Gehlbach, Makoto Jinno, Adnan Munawar, Peter Kazanzides, Iulian Iordachita  

**Link**: [PDF](https://arxiv.org/pdf/2504.12114)  

**Abstract**: Retinal surgery requires extreme precision due to constrained anatomical spaces in the human retina. To assist surgeons achieve this level of accuracy, the Improved Integrated Robotic Intraocular Snake (I2RIS) with dexterous capability has been developed. However, such flexible tendon-driven robots often suffer from hysteresis problems, which significantly challenges precise control and positioning. In particular, we observed multi-stage hysteresis phenomena in the small-scale I2RIS. In this paper, we propose an Extended Generalized Prandtl-Ishlinskii (EGPI) model to increase the fitting accuracy of the hysteresis. The model incorporates a novel switching mechanism that enables it to describe multi-stage hysteresis in the regions of monotonic input. Experimental validation on I2RIS data demonstrate that the EGPI model outperforms the conventional Generalized Prandtl-Ishlinskii (GPI) model in terms of RMSE, NRMSE, and MAE across multiple motor input directions. The EGPI model in our study highlights the potential in modeling multi-stage hysteresis in minimally invasive flexible robots. 

**Abstract (ZH)**: 微创柔性手术机器人中多阶段滞回现象的扩展广义普朗特-伊尔欣斯基模型研究 

---
# Self-Supervised Traversability Learning with Online Prototype Adaptation for Off-Road Autonomous Driving 

**Title (ZH)**: 基于在线原型适应的自监督通过性学习在非道路自主驾驶中的应用 

**Authors**: Yafeng Bu, Zhenping Sun, Xiaohui Li, Jun Zeng, Xin Zhang, Hui Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.12109)  

**Abstract**: Achieving reliable and safe autonomous driving in off-road environments requires accurate and efficient terrain traversability analysis. However, this task faces several challenges, including the scarcity of large-scale datasets tailored for off-road scenarios, the high cost and potential errors of manual annotation, the stringent real-time requirements of motion planning, and the limited computational power of onboard units. To address these challenges, this paper proposes a novel traversability learning method that leverages self-supervised learning, eliminating the need for manual annotation. For the first time, a Birds-Eye View (BEV) representation is used as input, reducing computational burden and improving adaptability to downstream motion planning. During vehicle operation, the proposed method conducts online analysis of traversed regions and dynamically updates prototypes to adaptively assess the traversability of the current environment, effectively handling dynamic scene changes. We evaluate our approach against state-of-the-art benchmarks on both public datasets and our own dataset, covering diverse seasons and geographical locations. Experimental results demonstrate that our method significantly outperforms recent approaches. Additionally, real-world vehicle experiments show that our method operates at 10 Hz, meeting real-time requirements, while a 5.5 km autonomous driving experiment further validates the generated traversability cost maps compatibility with downstream motion planning. 

**Abstract (ZH)**: 在离线环境中实现可靠和安全的自主驾驶需要准确且高效地进行地形穿越性分析。然而，这一任务面临着多个挑战，包括专门针对离线场景的大规模数据集稀缺、手动标注的高成本和潜在错误、运动规划的严格实时要求以及车载单元的计算能力有限。为应对这些挑战，本文提出了一种新颖的自监督学习穿越性学习方法，无需手动标注。首次使用空中鸟瞰视图（BEV）表示法作为输入，减少了计算负担并增强了对下游运动规划的适应性。在车辆运行过程中，所提出的方法进行在线分析已穿越区域并动态更新原型，以适应性评估当前环境的穿越性，有效应对动态场景变化。我们在公共数据集和我们自己的数据集上（涵盖不同季节和地理区域）对我们的方法与最新基准进行了评估。实验结果表明，我们的方法在性能上显著优于最近的方法。此外，实际车辆实验表明，我们的方法以10 Hz 的速度运行，满足了实时要求，而长达5.5 km 的自主驾驶试验进一步验证了生成的穿越性成本地图与下游运动规划的一致性。 

---
# Real-Time Shape Estimation of Tensegrity Structures Using Strut Inclination Angles 

**Title (ZH)**: 基于杆件倾斜角的 tensegrity 结构实时形状估计方法 

**Authors**: Tufail Ahmad Bhat, Yuhei Yoshimitsu, Kazuki Wada, Shuhei Ikemoto  

**Link**: [PDF](https://arxiv.org/pdf/2504.11868)  

**Abstract**: Tensegrity structures are becoming widely used in robotics, such as continuously bending soft manipulators and mobile robots to explore unknown and uneven environments dynamically. Estimating their shape, which is the foundation of their state, is essential for establishing control. However, on-board sensor-based shape estimation remains difficult despite its importance, because tensegrity structures lack well-defined joints, which makes it challenging to use conventional angle sensors such as potentiometers or encoders for shape estimation. To our knowledge, no existing work has successfully achieved shape estimation using only onboard sensors such as Inertial Measurement Units (IMUs). This study addresses this issue by proposing a novel approach that uses energy minimization to estimate the shape. We validated our method through experiments on a simple Class 1 tensegrity structure, and the results show that the proposed algorithm can estimate the real-time shape of the structure using onboard sensors, even in the presence of external disturbances. 

**Abstract (ZH)**: tensegrity结构在机器人领域中的应用日益广泛，如连续弯曲的柔软 manipulator 和移动机器人用于动态探索未知和不平坦的环境。估计其形状是确定其状态的基础，并且对于建立控制至关重要。然而，基于机载传感器的形状估计仍然具有挑战性，尽管它非常重要，因为 tensegrity 结构缺乏明确的关节，使得使用传统的角度传感器（如电位计或编码器）进行形状估计变得困难。据我们所知，现有工作中尚无使用惯性测量单元（IMUs）等机载传感器成功实现形状估计的实例。本研究通过提出一种基于能量最小化的新型方法来解决这一问题。我们通过对一个简单的 Class 1 tensegrity 结构的实验验证了该方法，并结果表明，所提出的方法可以在存在外部干扰的情况下，使用机载传感器实时估计结构的形状。 

---
# Multi-goal Rapidly Exploring Random Tree with Safety and Dynamic Constraints for UAV Cooperative Path Planning 

**Title (ZH)**: 具有安全和动态约束的多目标快速探索随机树无人机协同路径规划 

**Authors**: Thu Hang Khuat, Duy-Nam Bui, Hoa TT. Nguyen, Mien L. Trinh, Minh T. Nguyen, Manh Duong Phung  

**Link**: [PDF](https://arxiv.org/pdf/2504.11823)  

**Abstract**: Cooperative path planning is gaining its importance due to the increasing demand on using multiple unmanned aerial vehicles (UAVs) for complex missions. This work addresses the problem by introducing a new algorithm named MultiRRT that extends the rapidly exploring random tree (RRT) to generate paths for a group of UAVs to reach multiple goal locations at the same time. We first derive the dynamics constraint of the UAV and include it in the problem formulation. MultiRRT is then developed, taking into account the cooperative requirements and safe constraints during its path-searching process. The algorithm features two new mechanisms, node reduction and Bezier interpolation, to ensure the feasibility and optimality of the paths generated. Importantly, the interpolated paths are proven to meet the safety and dynamics constraints imposed by obstacles and the UAVs. A number of simulations, comparisons, and experiments have been conducted to evaluate the performance of the proposed approach. The results show that MultiRRT can generate collision-free paths for multiple UAVs to reach their goals with better scores in path length and smoothness metrics than state-of-the-art RRT variants including Theta-RRT, FN-RRT, RRT*, and RRT*-Smart. The generated paths are also tested in practical flights with real UAVs to evaluate their validity for cooperative tasks. The source code of the algorithm is available at this https URL 

**Abstract (ZH)**: 基于多无人飞行器协同的路径规划 Método de planificación de trayectorias cooperativa para un grupo de vehículos aéreos no tripulados mediante un nuevo algoritmo MultiRRT 

---
# Steerable rolling of a 1-DoF robot using an internal pendulum 

**Title (ZH)**: 使用内部摆锤实现单自由度机器人可引导的滚动 

**Authors**: Christopher Y. Xu, Jack Yan, Kathleen Lum, Justin K. Yim  

**Link**: [PDF](https://arxiv.org/pdf/2504.11748)  

**Abstract**: We present ROCK (Rolling One-motor Controlled rocK), a 1 degree-of-freedom robot consisting of a round shell and an internal pendulum. An uneven shell surface enables steering by using only the movement of the pendulum, allowing for mechanically simple designs that may be feasible to scale to large quantities or small sizes. We train a control policy using reinforcement learning in simulation and deploy it onto the robot to complete a rectangular trajectory. 

**Abstract (ZH)**: ROCK (Rolling One-motor Controlled rocK): 一个由圆形外壳和内部摆动机构组成的单自由度机器人 

---
# RESPLE: Recursive Spline Estimation for LiDAR-Based Odometry 

**Title (ZH)**: 基于LiDAR的递归 spline 估计里程计(RESPLE) 

**Authors**: Ziyu Cao, William Talbot, Kailai Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.11580)  

**Abstract**: We present a novel recursive Bayesian estimation framework for continuous-time six-DoF dynamic motion estimation using B-splines. The state vector consists of a recurrent set of position control points and orientation control point increments, enabling a straightforward modification of the iterated extended Kalman filter without involving the error-state formulation. The resulting recursive spline estimator (RESPLE) provides a versatile, pragmatic and lightweight solution for motion estimation and is further exploited for direct LiDAR-based odometry, supporting integration of one or multiple LiDARs and an IMU. We conduct extensive real-world benchmarking based on public datasets and own experiments, covering aerial, wheeled, legged, and wearable platforms operating in indoor, urban, wild environments with diverse LiDARs. RESPLE-based solutions achieve superior estimation accuracy and robustness over corresponding state-of-the-art systems, while attaining real-time performance. Notably, our LiDAR-only variant outperforms existing LiDAR-inertial systems in scenarios without significant LiDAR degeneracy, and showing further improvements when additional LiDAR and inertial sensors are incorporated for more challenging conditions. We release the source code and own experimental datasets at this https URL . 

**Abstract (ZH)**: 一种基于B样条的新型递归贝叶斯连续时间六自由度动态运动估计框架 

---
# Probabilistic Task Parameterization of Tool-Tissue Interaction via Sparse Landmarks Tracking in Robotic Surgery 

**Title (ZH)**: 基于稀疏地标跟踪的机器人手术中工具-组织交互的概率任务参数化 

**Authors**: Yiting Wang, Yunxin Fan, Fei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11495)  

**Abstract**: Accurate modeling of tool-tissue interactions in robotic surgery requires precise tracking of deformable tissues and integration of surgical domain knowledge. Traditional methods rely on labor-intensive annotations or rigid assumptions, limiting flexibility. We propose a framework combining sparse keypoint tracking and probabilistic modeling that propagates expert-annotated landmarks across endoscopic frames, even with large tissue deformations. Clustered tissue keypoints enable dynamic local transformation construction via PCA, and tool poses, tracked similarly, are expressed relative to these frames. Embedding these into a Task-Parameterized Gaussian Mixture Model (TP-GMM) integrates data-driven observations with labeled clinical expertise, effectively predicting relative tool-tissue poses and enhancing visual understanding of robotic surgical motions directly from video data. 

**Abstract (ZH)**: 精确 modeling of 工具-组织交互在机器人手术中的建模需要精确跟踪可变形组织并整合手术领域知识。传统方法依赖于劳动密集型注释或刚性假设，限制了灵活性。我们提出了一种结合稀疏关键点跟踪和概率建模的框架，该框架能够在大量组织变形的情况下，将专家标注的关键点传播到内镜帧中。聚类组织关键点通过PCA_enable动态局部变换的构造，并以类似方式跟踪的工具姿态相对于这些框架进行表达。将这些内容嵌入到任务参数化高斯混合模型（TP-GMM）中，将数据驱动的观察与标记的临床专业知识整合起来，有效地预测相对工具-组织姿态并直接从视频数据中增强对手术运动的视觉理解。 

---
