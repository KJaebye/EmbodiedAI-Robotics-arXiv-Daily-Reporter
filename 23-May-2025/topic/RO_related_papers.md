# UAV See, UGV Do: Aerial Imagery and Virtual Teach Enabling Zero-Shot Ground Vehicle Repeat 

**Title (ZH)**: UAV 观察，UGV 复制：基于航空影像和虚拟示教的零样本地面车辆重复任务 

**Authors**: Desiree Fisker, Alexander Krawciw, Sven Lilge, Melissa Greeff, Timothy D. Barfoot  

**Link**: [PDF](https://arxiv.org/pdf/2505.16912)  

**Abstract**: This paper presents Virtual Teach and Repeat (VirT&R): an extension of the Teach and Repeat (T&R) framework that enables GPS-denied, zero-shot autonomous ground vehicle navigation in untraversed environments. VirT&R leverages aerial imagery captured for a target environment to train a Neural Radiance Field (NeRF) model so that dense point clouds and photo-textured meshes can be extracted. The NeRF mesh is used to create a high-fidelity simulation of the environment for piloting an unmanned ground vehicle (UGV) to virtually define a desired path. The mission can then be executed in the actual target environment by using NeRF-derived point cloud submaps associated along the path and an existing LiDAR Teach and Repeat (LT&R) framework. We benchmark the repeatability of VirT&R on over 12 km of autonomous driving data using physical markings that allow a sim-to-real lateral path-tracking error to be obtained and compared with LT&R. VirT&R achieved measured root mean squared errors (RMSE) of 19.5 cm and 18.4 cm in two different environments, which are slightly less than one tire width (24 cm) on the robot used for testing, and respective maximum errors were 39.4 cm and 47.6 cm. This was done using only the NeRF-derived teach map, demonstrating that VirT&R has similar closed-loop path-tracking performance to LT&R but does not require a human to manually teach the path to the UGV in the actual environment. 

**Abstract (ZH)**: Virtual Teach and Repeat (VirT&R): 无GPS环境下的零样本自主地面车辆导航扩展方法 

---
# Joint Magnetometer-IMU Calibration via Maximum A Posteriori Estimation 

**Title (ZH)**: 基于最大后验估计的磁强计-IMU 联合标定 

**Authors**: Chuan Huang, Gustaf Hendeby, Isaac Skog  

**Link**: [PDF](https://arxiv.org/pdf/2505.16662)  

**Abstract**: This paper presents a new approach for jointly calibrating magnetometers and inertial measurement units, focusing on improving calibration accuracy and computational efficiency. The proposed method formulates the calibration problem as a maximum a posteriori estimation problem, treating both the calibration parameters and orientation trajectory of the sensors as unknowns. This formulation enables efficient optimization with closed-form derivatives. The method is compared against two state-of-the-art approaches in terms of computational complexity and estimation accuracy. Simulation results demonstrate that the proposed method achieves lower root mean square error in calibration parameters while maintaining competitive computational efficiency. Further validation through real-world experiments confirms the practical benefits of our approach: it effectively reduces position drift in a magnetic field-aided inertial navigation system by more than a factor of two on most datasets. Moreover, the proposed method calibrated 30 magnetometers in less than 2 minutes. The contributions include a new calibration method, an analysis of existing methods, and a comprehensive empirical evaluation. Datasets and algorithms are made publicly available to promote reproducible research. 

**Abstract (ZH)**: 本文提出了一种新的方法，用于同时校准磁力计和惯性测量单元，重点关注提高校准精度和计算效率。所提出的方法将校准问题形式化为最大后验估计问题，将传感器的校准参数和姿态轨迹均视为未知数。这种形式化使得可以通过闭式导数进行高效优化。该方法在计算复杂性和估计准确性方面与两种最先进的方法进行了比较。仿真结果表明，所提出的方法在保持竞争力的计算效率的同时，校准参数的均方根误差较低。通过实际实验进一步验证了该方法的实际优势：它在大多数数据集上使磁场辅助惯性导航系统的位置漂移降低了两倍以上。此外，所提出的方法在不到2分钟内完成了30个磁力计的校准。本研究的贡献包括一种新的校准方法、一种对现有方法的分析以及全面的经验性评估。所有数据集和算法均已公开发布，以促进可重复研究。 

---
# Monitoring Electrostatic Adhesion Forces via Acoustic Pressure 

**Title (ZH)**: 通过声压监测静电吸附力 

**Authors**: Huacen Wang, Jiarui Zou, Zeju Zheng, Hongqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16609)  

**Abstract**: Electrostatic adhesion is widely used in mobile robotics, haptics, and robotic end effectors for its adaptability to diverse substrates and low energy consumption. Force sensing is important for feedback control, interaction, and monitoring in the EA system. However, EA force monitoring often relies on bulky and expensive sensors, increasing the complexity and weight of the entire system. This paper presents an acoustic-pressure-based method to monitor EA forces without contacting the adhesion pad. When the EA pad is driven by a bipolar square-wave voltage to adhere a conductive object, periodic acoustic pulses arise from the EA system. We employed a microphone to capture these acoustic pressure signals and investigate the influence of peak pressure values. Results show that the peak value of acoustic pressure increased with the mass and contact area of the adhered object, as well as with the amplitude and frequency of the driving voltage. We applied this technique to mass estimation of various objects and simultaneous monitoring of two EA systems. Then, we integrated this technique into an EA end effector that enables monitoring the change of adhered object mass during transport. The proposed technique offers a low-cost, non-contact, and multi-object monitoring solution for EA end effectors in handling tasks. 

**Abstract (ZH)**: 基于声压的无接触EA力监测方法及其在.End Effector处理任务中的多对象监测应用 

---
# SpineWave: Harnessing Fish Rigid-Flexible Spinal Kinematics for Enhancing Biomimetic Robotic Locomotion 

**Title (ZH)**: 脊柱波：利用鱼类刚柔脊柱运动学增强生物仿生机器人运动性能 

**Authors**: Qu He, Weikun Li, Guangmin Dai, Hao Chen, Qimeng Liu, Xiaoqing Tian, Jie You, Weicheng Cui, Michael S. Triantafyllou, Dixia Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.16453)  

**Abstract**: Fish have endured millions of years of evolution, and their distinct rigid-flexible body structures offer inspiration for overcoming challenges in underwater robotics, such as limited mobility, high energy consumption, and adaptability. This paper introduces SpineWave, a biomimetic robotic fish featuring a fish-spine-like rigid-flexible transition structure. The structure integrates expandable fishbone-like ribs and adjustable magnets, mimicking the stretch and recoil of fish muscles to balance rigidity and flexibility. In addition, we employed an evolutionary algorithm to optimize the hydrodynamics of the robot, achieving significant improvements in swimming performance. Real-world tests demonstrated robustness and potential for environmental monitoring, underwater exploration, and industrial inspection. These tests established SpineWave as a transformative platform for aquatic robotics. 

**Abstract (ZH)**: 鱼类经过数百万年的进化，其独特的刚柔并济的身体结构为克服水下机器人领域中的有限移动性、高能耗和适应能力等挑战提供了灵感。本文介绍了一种名为SpineWave的仿生水下机器人鱼，该机器人鱼采用了类似鱼类脊椎的刚柔过渡结构，结合可扩展的鱼骨状肋骨和可调节的磁铁，模拟鱼类肌肉的拉伸和回弹，实现刚性和柔性的平衡。此外，我们还采用进化算法优化了机器人的流体力学性能，显著提高了其游泳性能。实地测试表明，SpineWave具有较强的环境监测、水下探索和工业检测的潜力，奠定了其在水下机器人领域的革新平台地位。 

---
# TacCompress: A Benchmark for Multi-Point Tactile Data Compression in Dexterous Manipulation 

**Title (ZH)**: TacCompress: 多点触觉数据在灵巧操作中的压缩基准 

**Authors**: Yang Li, Yan Zhao, Zhengxue Cheng, Hengdi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16289)  

**Abstract**: Though robotic dexterous manipulation has progressed substantially recently, challenges like in-hand occlusion still necessitate fine-grained tactile perception, leading to the integration of more tactile sensors into robotic hands. Consequently, the increased data volume imposes substantial bandwidth pressure on signal transmission from the hand's controller. However, the acquisition and compression of multi-point tactile signals based on the dexterous hands' physical structures have not been thoroughly explored. In this paper, our contributions are twofold. First, we introduce a Multi-Point Tactile Dataset for Dexterous Hand Grasping (Dex-MPTD). This dataset captures tactile signals from multiple contact sensors across various objects and grasping poses, offering a comprehensive benchmark for advancing dexterous robotic manipulation research. Second, we investigate both lossless and lossy compression on Dex-MPTD by converting tactile data into images and applying six lossless and five lossy image codecs for efficient compression. Experimental results demonstrate that tactile data can be losslessly compressed to as low as 0.0364 bits per sub-sample (bpss), achieving approximately 200$\times$ compression ratio compared to the raw tactile data. Efficient lossy compressors like HM and VTM can achieve about 1000x data reductions while preserving acceptable data fidelity. The exploration of lossy compression also reveals that screen-content-targeted coding tools outperform general-purpose codecs in compressing tactile data. 

**Abstract (ZH)**: 多点触觉数据集用于灵巧手抓取（Dex-MPTD）及其压缩研究 

---
# Manipulating Elasto-Plastic Objects With 3D Occupancy and Learning-Based Predictive Control 

**Title (ZH)**: 基于3D占据表示和学习导向的预测控制的弹性-塑性物体操纵 

**Authors**: Zhen Zhang, Xiangyu Chu, Yunxi Tang, Lulu Zhao, Jing Huang, Zhongliang Jiang, K. W. Samuel Au  

**Link**: [PDF](https://arxiv.org/pdf/2505.16249)  

**Abstract**: Manipulating elasto-plastic objects remains a significant challenge due to severe self-occlusion, difficulties of representation, and complicated dynamics. This work proposes a novel framework for elasto-plastic object manipulation with a quasi-static assumption for motions, leveraging 3D occupancy to represent such objects, a learned dynamics model trained with 3D occupancy, and a learning-based predictive control algorithm to address these challenges effectively. We build a novel data collection platform to collect full spatial information and propose a pipeline for generating a 3D occupancy dataset. To infer the 3D occupancy during manipulation, an occupancy prediction network is trained with multiple RGB images supervised by the generated dataset. We design a deep neural network empowered by a 3D convolution neural network (CNN) and a graph neural network (GNN) to predict the complex deformation with the inferred 3D occupancy results. A learning-based predictive control algorithm is introduced to plan the robot actions, incorporating a novel shape-based action initialization module specifically designed to improve the planner efficiency. The proposed framework in this paper can successfully shape the elasto-plastic objects into a given goal shape and has been verified in various experiments both in simulation and the real world. 

**Abstract (ZH)**: 基于拟静止假设的弹性塑性物体操作新型框架 

---
# Tactile-based Reinforcement Learning for Adaptive Grasping under Observation Uncertainties 

**Title (ZH)**: 基于触觉的强化学习在观测不确定性下的自适应抓取 

**Authors**: Xiao Hu, Yang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2505.16167)  

**Abstract**: Robotic manipulation in industrial scenarios such as construction commonly faces uncertain observations in which the state of the manipulating object may not be accurately captured due to occlusions and partial observables. For example, object status estimation during pipe assembly, rebar installation, and electrical installation can be impacted by observation errors. Traditional vision-based grasping methods often struggle to ensure robust stability and adaptability. To address this challenge, this paper proposes a tactile simulator that enables a tactile-based adaptive grasping method to enhance grasping robustness. This approach leverages tactile feedback combined with the Proximal Policy Optimization (PPO) reinforcement learning algorithm to dynamically adjust the grasping posture, allowing adaptation to varying grasping conditions under inaccurate object state estimations. Simulation results demonstrate that the proposed method effectively adapts grasping postures, thereby improving the success rate and stability of grasping tasks. 

**Abstract (ZH)**: 工业场景下（如建筑施工）的机器人操作常面临不确定的观测问题，由于遮挡和部分可观测性，操作对象的状态可能无法准确捕捉。例如，在管道组装、钢筋安装和电气安装过程中，观测误差会影响物体状态估计。传统的基于视觉的抓取方法往往难以确保抓取的 robust 稳定性和适应性。为应对这一挑战，本文提出一种触觉模拟器，以增强抓取的 robust 性。该方法利用触觉反馈结合 Proximal Policy Optimization (PPO) 强化学习算法动态调整抓取姿态，适应不同不准确物体状态估计下的抓取条件。仿真结果表明，所提方法能够有效调整抓取姿态，从而提高抓取任务的成功率和稳定性。 

---
# Event-based Reconfiguration Control for Time-varying Formation of Robot Swarms in Narrow Spaces 

**Title (ZH)**: 基于事件的重构控制方法研究：狭窄空间中时间变化的机器人 swarm 形态控制 

**Authors**: Duy-Nam Bui, Manh Duong Phung, Hung Pham Duy  

**Link**: [PDF](https://arxiv.org/pdf/2505.16087)  

**Abstract**: This study proposes an event-based reconfiguration control to navigate a robot swarm through challenging environments with narrow passages such as valleys, tunnels, and corridors. The robot swarm is modeled as an undirected graph, where each node represents a robot capable of collecting real-time data on the environment and the states of other robots in the formation. This data serves as the input for the controller to provide dynamic adjustments between the desired and straight-line configurations. The controller incorporates a set of behaviors, designed using artificial potential fields, to meet the requirements of goal-oriented motion, formation maintenance, tailgating, and collision avoidance. The stability of the formation control is guaranteed via the Lyapunov theorem. Simulation and comparison results show that the proposed controller not only successfully navigates the robot swarm through narrow spaces but also outperforms other established methods in key metrics including the success rate, heading order, speed, travel time, and energy efficiency. Software-in-the-loop tests have also been conducted to validate the controller's applicability in practical scenarios. The source code of the controller is available at this https URL. 

**Abstract (ZH)**: 基于事件的重构控制方法在狭窄通道环境下的机器人集群导航 

---
# WaveTouch: Active Tactile Sensing Using Vibro-Feedback for Classification of Variable Stiffness and Infill Density Objects 

**Title (ZH)**: WaveTouch: 基于振动反馈的活性触觉传感用于变刚度和填充密度物体分类 

**Authors**: Danissa Sandykbayeva, Valeriya Kostyukova, Aditya Shekhar Nittala, Zhanat Kappassov, Bakhtiyar Orazbayev  

**Link**: [PDF](https://arxiv.org/pdf/2505.16062)  

**Abstract**: The perception and recognition of the surroundings is one of the essential tasks for a robot. With preliminary knowledge about a target object, it can perform various manipulation tasks such as rolling motion, palpation, and force control. Minimizing possible damage to the sensing system and testing objects during manipulation are significant concerns that persist in existing research solutions. To address this need, we designed a new type of tactile sensor based on the active vibro-feedback for object stiffness classification. With this approach, the classification can be performed during the gripping process, enabling the robot to quickly estimate the appropriate level of gripping force required to avoid damaging or dropping the object. This contrasts with passive vibration sensing, which requires to be triggered by object movement and is often inefficient for establishing a secure grip. The main idea is to observe the received changes in artificially injected vibrations that propagate through objects with different physical properties and molecular structures. The experiments with soft subjects demonstrated higher absorption of the received vibrations, while the opposite is true for the rigid subjects that not only demonstrated low absorption but also enhancement of the vibration signal. 

**Abstract (ZH)**: 基于主动振动反馈的触觉传感器及其在物体刚度分类中的应用 

---
# Proactive Hierarchical Control Barrier Function-Based Safety Prioritization in Close Human-Robot Interaction Scenarios 

**Title (ZH)**: 主动分层控制约束函数基于的安全优先级在近距离人机交互场景中 

**Authors**: Patanjali Maithania, Aliasghar Araba, Farshad Khorramia, Prashanth Krishnamurthya  

**Link**: [PDF](https://arxiv.org/pdf/2505.16055)  

**Abstract**: In collaborative human-robot environments, the unpredictable and dynamic nature of human motion can lead to situations where collisions become unavoidable. In such cases, it is essential for the robotic system to proactively mitigate potential harm through intelligent control strategies. This paper presents a hierarchical control framework based on Control Barrier Functions (CBFs) designed to ensure safe and adaptive operation of autonomous robotic manipulators during close-proximity human-robot interaction. The proposed method introduces a relaxation variable that enables real-time prioritization of safety constraints, allowing the robot to dynamically manage collision risks based on the criticality of different parts of the human body. A secondary constraint mechanism is incorporated to resolve infeasibility by increasing the priority of imminent threats. The framework is experimentally validated on a Franka Research 3 robot equipped with a ZED2i AI camera for real-time human pose and body detection. Experimental results confirm that the CBF-based controller, integrated with depth sensing, facilitates responsive and safe human-robot collaboration, while providing detailed risk analysis and maintaining robust performance in highly dynamic settings. 

**Abstract (ZH)**: 基于控制屏障函数的层次控制框架：在动态人体运动环境中实现自主机器人操作的安全与适应性 

---
