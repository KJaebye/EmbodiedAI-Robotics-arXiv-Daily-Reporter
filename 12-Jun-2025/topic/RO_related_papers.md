# eFlesh: Highly customizable Magnetic Touch Sensing using Cut-Cell Microstructures 

**Title (ZH)**: eFlesh: 高度可定制的切割细胞微结构磁触感技术 

**Authors**: Venkatesh Pattabiraman, Zizhou Huang, Daniele Panozzo, Denis Zorin, Lerrel Pinto, Raunaq Bhirangi  

**Link**: [PDF](https://arxiv.org/pdf/2506.09994)  

**Abstract**: If human experience is any guide, operating effectively in unstructured environments -- like homes and offices -- requires robots to sense the forces during physical interaction. Yet, the lack of a versatile, accessible, and easily customizable tactile sensor has led to fragmented, sensor-specific solutions in robotic manipulation -- and in many cases, to force-unaware, sensorless approaches. With eFlesh, we bridge this gap by introducing a magnetic tactile sensor that is low-cost, easy to fabricate, and highly customizable. Building an eFlesh sensor requires only four components: a hobbyist 3D printer, off-the-shelf magnets (<$5), a CAD model of the desired shape, and a magnetometer circuit board. The sensor is constructed from tiled, parameterized microstructures, which allow for tuning the sensor's geometry and its mechanical response. We provide an open-source design tool that converts convex OBJ/STL files into 3D-printable STLs for fabrication. This modular design framework enables users to create application-specific sensors, and to adjust sensitivity depending on the task. Our sensor characterization experiments demonstrate the capabilities of eFlesh: contact localization RMSE of 0.5 mm, and force prediction RMSE of 0.27 N for normal force and 0.12 N for shear force. We also present a learned slip detection model that generalizes to unseen objects with 95% accuracy, and visuotactile control policies that improve manipulation performance by 40% over vision-only baselines -- achieving 91% average success rate for four precise tasks that require sub-mm accuracy for successful completion. All design files, code and the CAD-to-eFlesh STL conversion tool are open-sourced and available on this https URL. 

**Abstract (ZH)**: 基于eFlesh的低成本可定制磁性触觉传感器及其应用 

---
# Locomotion on Constrained Footholds via Layered Architectures and Model Predictive Control 

**Title (ZH)**: 基于分层架构和模型预测控制的受限制支撑点上的运动控制 

**Authors**: Zachary Olkin, Aaron D. Ames  

**Link**: [PDF](https://arxiv.org/pdf/2506.09979)  

**Abstract**: Computing stabilizing and optimal control actions for legged locomotion in real time is difficult due to the nonlinear, hybrid, and high dimensional nature of these robots. The hybrid nature of the system introduces a combination of discrete and continuous variables which causes issues for numerical optimal control. To address these challenges, we propose a layered architecture that separates the choice of discrete variables and a smooth Model Predictive Controller (MPC). The layered formulation allows for online flexibility and optimality without sacrificing real-time performance through a combination of gradient-free and gradient-based methods. The architecture leverages a sampling-based method for determining discrete variables, and a classical smooth MPC formulation using these fixed discrete variables. We demonstrate the results on a quadrupedal robot stepping over gaps and onto terrain with varying heights. In simulation, we demonstrate the controller on a humanoid robot for gap traversal. The layered approach is shown to be more optimal and reliable than common heuristic-based approaches and faster to compute than pure sampling methods. 

**Abstract (ZH)**: 实时计算支撑腿式移动的稳定和最优控制动作因系统非线性、混合性和高维性而具有挑战性。为应对这些挑战，我们提出了一种分层架构，该架构将离散变量的选择与平滑模型预测控制（MPC）分开。分层表示法通过结合无导数方法和导数方法，在不牺牲实时性能的情况下提供在线灵活性和最优性。该架构利用基于采样的方法确定离散变量，并使用固定离散变量的经典平滑MPC表示法。我们在一个四足机器人跨越缺口和踏上不同高度地形的实验中展示了该方法。在模拟中，我们在一个类人机器人上展示了控制器进行跨越缺口的控制。分层方法在最优性和可靠性方面优于常用的经验方法，并且计算速度比纯粹基于采样的方法更快。 

---
# Fluoroscopic Shape and Pose Tracking of Catheters with Custom Radiopaque Markers 

**Title (ZH)**: 带有定制放射不透明标志的导管透视形状和姿态跟踪 

**Authors**: Jared Lawson, Rohan Chitale, Nabil Simaan  

**Link**: [PDF](https://arxiv.org/pdf/2506.09934)  

**Abstract**: Safe navigation of steerable and robotic catheters in the cerebral vasculature requires awareness of the catheters shape and pose. Currently, a significant perception burden is placed on interventionalists to mentally reconstruct and predict catheter motions from biplane fluoroscopy images. Efforts to track these catheters are limited to planar segmentation or bulky sensing instrumentation, which are incompatible with microcatheters used in neurointervention. In this work, a catheter is equipped with custom radiopaque markers arranged to enable simultaneous shape and pose estimation under biplane fluoroscopy. A design measure is proposed to guide the arrangement of these markers to minimize sensitivity to marker tracking uncertainty. This approach was deployed for microcatheters smaller than 2mm OD navigating phantom vasculature with shape tracking errors less than 1mm and catheter roll errors below 40 degrees. This work can enable steerable catheters to autonomously navigate under biplane imaging. 

**Abstract (ZH)**: 基于双平面成像的可操控和机器人导管在脑血管内的安全导航需要对导管的形状和姿态有所认识。目前，介入医师需通过心理重构和预测从双平面透视图像中推断导管运动，面临显著的感知负担。当前用于跟踪此类导管的努力仅限于平面分割或笨重的传感装置，这些方法不适用于神经介入中常用的微导管。本研究中，导管配备了自定义的放射-opacity标记，以便在双平面透视下同时进行形状和姿态估计。提出了一种设计措施来指导这些标记的排列，以最小化标记跟踪不确定性的影响。该方法在OD小于2mm的微导管在仿真血管中导航时表现出优异的形状跟踪误差（小于1mm）和导管滚转误差（低于40度）性能，从而使得可操控导管能够在双平面成像下实现自主导航。 

---
# From Theory to Practice: Advancing Multi-Robot Path Planning Algorithms and Applications 

**Title (ZH)**: 从理论到实践：推动多机器人路径规划算法及其应用的发展 

**Authors**: Teng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.09914)  

**Abstract**: The labeled MRPP (Multi-Robot Path Planning) problem involves routing robots from start to goal configurations efficiently while avoiding collisions. Despite progress in solution quality and runtime, its complexity and industrial relevance continue to drive research.
This dissertation introduces scalable MRPP methods with provable guarantees and practical heuristics. First, we study dense MRPP on 2D grids, relevant to warehouse and parcel systems. We propose the Rubik Table method, achieving $(1 + \delta)$-optimal makespan (with $\delta \in (0, 0.5]$) for up to $\frac{m_1 m_2}{2}$ robots, solving large instances efficiently and setting a new theoretical benchmark.
Next, we address real-world MRPP. We design optimal layouts for structured environments (e.g., warehouses, parking systems) and propose a puzzle-based system for dense, deadlock-free autonomous vehicle parking. We also extend MRPP to Reeds-Shepp robots, introducing motion primitives and smoothing techniques to ensure feasible, efficient paths under nonholonomic constraints. Simulations and real-world tests validate the approach in urban driving and robotic transport scenarios. 

**Abstract (ZH)**: 带标注的多机器人路径规划问题涉及在避免碰撞的情况下，高效地将机器人从起始配置引导至目标配置。尽管在解决方案质量和运行时间方面取得了进展，但其复杂性和工业相关性仍继续推动研究。

本论文介绍了具有可验证保证和实用启发式的可扩展多机器人路径规划方法。首先，我们研究了密集型2D网格上的多机器人路径规划，这与仓库和包裹系统相关。我们提出了Rubik Table方法，实现了最多$\frac{m_1 m_2}{2}$个机器人在$(1 + \delta)$-最优最短工期（$\delta \in (0, 0.5]$）下的解，有效解决大规模实例，并建立了新的理论基准。
接下来，我们解决了实际的多机器人路径规划问题。我们为结构化环境设计了最优布局（例如，仓库、停车系统），并提出了一种基于拼图的系统，用于密集、无死锁的自主车辆停车。我们还扩展了多机器人路径规划到Reeds-Shepp机器人，引入了运动原语和平滑技术，以确保在非完整约束下可行且高效的路径。仿真实验和实际测试验证了在城市驾驶和机器人运输场景中的方法。 

---
# Aucamp: An Underwater Camera-Based Multi-Robot Platform with Low-Cost, Distributed, and Robust Localization 

**Title (ZH)**: Aucamp：一种基于水下摄像头的低成本、分布式和稳健的多机器人平台 

**Authors**: Jisheng Xu, Ding Lin, Pangkit Fong, Chongrong Fang, Xiaoming Duan, Jianping He  

**Link**: [PDF](https://arxiv.org/pdf/2506.09876)  

**Abstract**: This paper introduces an underwater multi-robot platform, named Aucamp, characterized by cost-effective monocular-camera-based sensing, distributed protocol and robust orientation control for localization. We utilize the clarity feature to measure the distance, present the monocular imaging model, and estimate the position of the target object. We achieve global positioning in our platform by designing a distributed update protocol. The distributed algorithm enables the perception process to simultaneously cover a broader range, and greatly improves the accuracy and robustness of the positioning. Moreover, the explicit dynamics model of the robot in our platform is obtained, based on which, we propose a robust orientation control framework. The control system ensures that the platform maintains a balanced posture for each robot, thereby ensuring the stability of the localization system. The platform can swiftly recover from an forced unstable state to a stable horizontal posture. Additionally, we conduct extensive experiments and application scenarios to evaluate the performance of our platform. The proposed new platform may provide support for extensive marine exploration by underwater sensor networks. 

**Abstract (ZH)**: 一种基于单目摄像头传感的分布式协议和 robust 方向控制的水下多机器人平台：Aucamp 

---
# VAULT: A Mobile Mapping System for ROS 2-based Autonomous Robots 

**Title (ZH)**: VAULT：基于ROS 2的自主机器人移动 mapping 系统 

**Authors**: Miguel Á. González-Santamarta, Francisco J. Rodríguez-Lera, Vicente Matellán-Olivera  

**Link**: [PDF](https://arxiv.org/pdf/2506.09583)  

**Abstract**: Localization plays a crucial role in the navigation capabilities of autonomous robots, and while indoor environments can rely on wheel odometry and 2D LiDAR-based mapping, outdoor settings such as agriculture and forestry, present unique challenges that necessitate real-time localization and consistent mapping. Addressing this need, this paper introduces the VAULT prototype, a ROS 2-based mobile mapping system (MMS) that combines various sensors to enable robust outdoor and indoor localization. The proposed solution harnesses the power of Global Navigation Satellite System (GNSS) data, visual-inertial odometry (VIO), inertial measurement unit (IMU) data, and the Extended Kalman Filter (EKF) to generate reliable 3D odometry. To further enhance the localization accuracy, Visual SLAM (VSLAM) is employed, resulting in the creation of a comprehensive 3D point cloud map. By leveraging these sensor technologies and advanced algorithms, the prototype offers a comprehensive solution for outdoor localization in autonomous mobile robots, enabling them to navigate and map their surroundings with confidence and precision. 

**Abstract (ZH)**: 自主机器人在导航能力中，定位起着关键作用。虽然室内环境可以依赖轮式里程计和基于2D LiDAR的制图，但如农业和林业等户外环境则提出了独特挑战，需要实时定位和持续建图。为应对这一需求，本文介绍了基于ROS 2的移动建图系统（MMS）VAULT原型，该系统结合了多种传感器以实现 robust 的户外和室内定位。所提出的方法利用全球导航卫星系统（GNSS）数据、视觉惯性里程计（VIO）、惯性测量单元（IMU）数据以及扩展卡尔曼滤波器（EKF）生成可靠的3D里程计。为了进一步提高定位准确性，还采用了视觉 SLAM（VSLAM），生成了全面的3D点云地图。通过利用这些传感器技术和先进算法，该原型为自主移动机器人的户外定位提供了全面的解决方案，使它们能够在陌生环境中导航和高效建图。 

---
# Tightly-Coupled LiDAR-IMU-Leg Odometry with Online Learned Leg Kinematics Incorporating Foot Tactile Information 

**Title (ZH)**: 紧耦合LiDAR-IMU-腿部里程计融合，结合足部触觉信息的在线学习腿部运动学 

**Authors**: Taku Okawara, Kenji Koide, Aoki Takanose, Shuji Oishi, Masashi Yokozuka, Kentaro Uno, Kazuya Yoshida  

**Link**: [PDF](https://arxiv.org/pdf/2506.09548)  

**Abstract**: In this letter, we present tightly coupled LiDAR-IMU-leg odometry, which is robust to challenging conditions such as featureless environments and deformable terrains. We developed an online learning-based leg kinematics model named the neural leg kinematics model, which incorporates tactile information (foot reaction force) to implicitly express the nonlinear dynamics between robot feet and the ground. Online training of this model enhances its adaptability to weight load changes of a robot (e.g., assuming delivery or transportation tasks) and terrain conditions. According to the \textit{neural adaptive leg odometry factor} and online uncertainty estimation of the leg kinematics model-based motion predictions, we jointly solve online training of this kinematics model and odometry estimation on a unified factor graph to retain the consistency of both. The proposed method was verified through real experiments using a quadruped robot in two challenging situations: 1) a sandy beach, representing an extremely featureless area with a deformable terrain, and 2) a campus, including multiple featureless areas and terrain types of asphalt, gravel (deformable terrain), and grass. Experimental results showed that our odometry estimation incorporating the \textit{neural leg kinematics model} outperforms state-of-the-art works. Our project page is available for further details: this https URL 

**Abstract (ZH)**: 本文介绍了紧耦合LiDAR-IMU腿部里程计，该方法在无特征环境和可变形地形等挑战条件下表现出高度鲁棒性。文中开发了一种基于在线学习的腿动力学模型——神经腿动力学模型，该模型整合了触觉信息（足反应力）以隐式表达机器人足部与地面之间的非线性动力学。该模型的在线训练增强了其对机器人载荷变化（例如执行递送或运输任务）和地形条件的适应性。根据神经自适应腿部里程计因子和基于腿动力学模型的运动预测的在线不确定性估计，文中在统一因子图上联合解决了该动力学模型的在线训练和里程计估计，以保持两者的一致性。通过在四足机器人上进行的实际试验验证了本方法，试验环境包括两个具有挑战性的场景：1）一个沙滩，代表一个极度无特征的可变形地形区域；2）一个校园，包括多个无特征区域和不同类型的地形，如沥青、碎石（可变形地形）和草地。实验结果表明，结合神经腿动力学模型的里程计估计优于现有最先进的方法。我们的项目页面可提供更多详情：this https URL。 

---
# Advances on Affordable Hardware Platforms for Human Demonstration Acquisition in Agricultural Applications 

**Title (ZH)**: 面向农业应用的人体示范获取的经济硬件平台进展 

**Authors**: Alberto San-Miguel-Tello, Gennaro Scarati, Alejandro Hernández, Mario Cavero-Vidal, Aakash Maroti, Néstor García  

**Link**: [PDF](https://arxiv.org/pdf/2506.09494)  

**Abstract**: This paper presents advances on the Universal Manipulation Interface (UMI), a low-cost hand-held gripper for robot Learning from Demonstration (LfD), for complex in-the-wild scenarios found in agricultural settings. The focus is on improving the acquisition of suitable samples with minimal additional setup. Firstly, idle times and user's cognitive load are reduced through the extraction of individual samples from a continuous demonstration considering task events. Secondly, reliability on the generation of task sample's trajectories is increased through the combination on-board inertial measurements and external visual marker localization usage using Extended Kalman Filtering (EKF). Results are presented for a fruit harvesting task, outperforming the default pipeline. 

**Abstract (ZH)**: 这篇论文介绍了通用操作接口（UMI）的进步，UMI是一种低成本手持夹持器，用于机器人从示范学习（LfD），适用于农业环境中发现的复杂实地场景。重点是通过考虑任务事件从连续示范中提取个体样本，减少空闲时间并降低用户的认知负荷，以及通过结合机载惯性测量和外部分辨率视觉标记定位，并使用扩展卡尔曼滤波（EKF）来提高任务样本轨迹生成的可靠性。结果表明，该方法在水果采摘任务中优于默认流水线。 

---
# Design of an innovative robotic surgical instrument for circular stapling 

**Title (ZH)**: 设计一种创新的圆周吻合 Robotics手术器械 

**Authors**: Paul Tucan, Nadim Al Hajjar, Calin Vaida, Alexandru Pusca, Tiberiu Antal, Corina Radu, Daniel Jucan, Adrian Pisla, Damien Chablat, Doina Pisla  

**Link**: [PDF](https://arxiv.org/pdf/2506.09444)  

**Abstract**: Esophageal cancer remains a highly aggressive malignancy with low survival rates, requiring advanced surgical interventions like esophagectomy. Traditional manual techniques, including circular staplers, face challenges such as limited precision, prolonged recovery times, and complications like leaks and tissue misalignment. This paper presents a novel robotic circular stapler designed to enhance the dexterity in confined spaces, improve tissue alignment, and reduce post-operative risks. Integrated with a cognitive robot that serves as a surgeon's assistant, the surgical stapler uses three actuators to perform anvil motion, cutter/stapler motion and allows a 75-degree bending of the cartridge (distal tip). Kinematic analysis is used to compute the stapler tip's position, ensuring synchronization with a robotic system. 

**Abstract (ZH)**: 食管癌 remains 一种高度侵袭性的恶性肿瘤，伴有较低的生存率，需要先进的手术干预如食管切除术。传统的手工技术，包括圆形吻合器，面临着精确度有限、恢复时间长以及漏口和组织错位等并发症的挑战。本文介绍了一种新型的机器人圆形吻合器，旨在在有限空间中提高机动性、改善组织对齐并降低术后风险。该吻合器集成了作为外科医生助手的认知机器人，使用三个执行器进行铆钉头运动、切割/吻合器运动，并允许枪 cartridge (远端尖端) 75度弯曲。使用运动分析来计算吻合器尖端的位置，确保与机器人系统同步。 

---
# Hearing the Slide: Acoustic-Guided Constraint Learning for Fast Non-Prehensile Transport 

**Title (ZH)**: 听滑动声：基于声学的约束学习快速非抓取运输 

**Authors**: Yuemin Mao, Bardienus P. Duisterhof, Moonyoung Lee, Jeffrey Ichnowski  

**Link**: [PDF](https://arxiv.org/pdf/2506.09169)  

**Abstract**: Object transport tasks are fundamental in robotic automation, emphasizing the importance of efficient and secure methods for moving objects. Non-prehensile transport can significantly improve transport efficiency, as it enables handling multiple objects simultaneously and accommodating objects unsuitable for parallel-jaw or suction grasps. Existing approaches incorporate constraints based on the Coulomb friction model, which is imprecise during fast motions where inherent mechanical vibrations occur. Imprecise constraints can cause transported objects to slide or even fall off the tray. To address this limitation, we propose a novel method to learn a friction model using acoustic sensing that maps a tray's motion profile to a dynamically conditioned friction coefficient. This learned model enables an optimization-based motion planner to adjust the friction constraint at each control step according to the planned motion at that step. In experiments, we generate time-optimized trajectories for a UR5e robot to transport various objects with constraints using both the standard Coulomb friction model and the learned friction model. Results suggest that the learned friction model reduces object displacement by up to 86.0% compared to the baseline, highlighting the effectiveness of acoustic sensing in learning real-world friction constraints. 

**Abstract (ZH)**: 基于声学感知学习的摩擦模型在机器人物体运输任务中的应用 

---
# Adaptive event-triggered robust tracking control of soft robots 

**Title (ZH)**: 软体机器人自适应事件触发鲁棒跟踪控制 

**Authors**: Renjie Ma, Ziyao Qu, Zhijian Hu, Dong Zhao, Marios M. Polycarpou  

**Link**: [PDF](https://arxiv.org/pdf/2506.09523)  

**Abstract**: Soft robots manufactured with flexible materials can be highly compliant and adaptive to their surroundings, which facilitates their application in areas such as dexterous manipulation and environmental exploration. This paper aims at investigating the tracking control problem for soft robots under uncertainty such as unmodeled dynamics and external disturbance. First, we establish a novel switching function and design the compensated tracking error dynamics by virtue of the command filter. Then, based on the backstepping methodology, the virtual controllers and the adaptive logic estimating the supremum of uncertainty impacts are developed for synthesizing an event-triggered control strategy. In addition, the uniformed finite-time stability certification is derived for different scenarios of the switching function. Finally, we perform a case study of a soft robot to illustrate the effectiveness of the proposed control algorithm. 

**Abstract (ZH)**: 基于柔性材料制造的软机器人可以高度顺应环境，适用于灵巧操作和环境探索等领域。本文旨在研究在未建模动态和外部干扰等不确定性条件下软机器人的跟踪控制问题。首先，我们建立了新的切换函数，并利用命令滤波器设计补偿跟踪误差动力学。然后，基于回步设计方法，开发了虚拟控制器和自适应逻辑来估计不确定性影响的上界，并据此合成了一种事件触发控制策略。此外，还推导了不同场景下切换函数的一致有限时间稳定性认证。最后，通过软机器人的案例研究来说明所提出控制算法的有效性。 

---
