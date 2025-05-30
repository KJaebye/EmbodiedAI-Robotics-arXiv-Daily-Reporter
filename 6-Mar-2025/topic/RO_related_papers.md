# Adaptive Negative Damping Control for User-Dependent Multi-Terrain Walking Assistance with a Hip Exoskeleton 

**Title (ZH)**: 用户依赖的多地形行走辅助髋部外骨骼自适应负阻尼控制 

**Authors**: Giulia Ramella, Auke Ijspeert, Mohamed Bouri  

**Link**: [PDF](https://arxiv.org/pdf/2503.03662)  

**Abstract**: Hip exoskeletons are known for their versatility in assisting users across varied scenarios. However, current assistive strategies often lack the flexibility to accommodate for individual walking patterns and adapt to diverse locomotion environments. In this work, we present a novel control strategy that adapts the mechanical impedance of the human-exoskeleton system. We design the hip assistive torques as an adaptive virtual negative damping, which is able to inject energy into the system while allowing the users to remain in control and contribute voluntarily to the movements. Experiments with five healthy subjects demonstrate that our controller reduces the metabolic cost of walking compared to free walking (average reduction of 7.2%), and it preserves the lower-limbs kinematics. Additionally, our method achieves minimal power losses from the exoskeleton across the entire gait cycle (less than 2% negative mechanical power out of the total power), ensuring synchronized action with the users' movements. Moreover, we use Bayesian Optimization to adapt the assistance strength and allow for seamless adaptation and transitions across multi-terrain environments. Our strategy achieves efficient power transmission under all conditions. Our approach demonstrates an individualized, adaptable, and straightforward controller for hip exoskeletons, advancing the development of viable, adaptive, and user-dependent control laws. 

**Abstract (ZH)**: 髋部外骨骼在多种场景下提供助力具有灵活性的优势，但当前的助力策略往往缺乏适应个体步行模式和不同运动环境的灵活性。本工作中，我们提出了一种新的控制策略，以适应人类-外骨骼系统中的机械阻抗。我们设计了髋部助力扭力作为适应性的虚拟负阻尼，能够在注入能量的同时允许用户保持控制并自愿贡献于动作。五名健康受试者的实验结果表明，与自由行走相比，我们的控制器降低了行走的代谢成本（平均降低7.2%），同时保持了下肢运动学。此外，我们的方法在整步行周期内实现了最小的外骨骼功率损失（总功率的不到2%负机械功率），确保与用户的动作同步。我们使用贝叶斯优化来适应助力强度，并实现跨多地形环境的无缝适应和过渡。该策略在所有条件下均实现高效的功率传输。我们的方法证明了个性化、适应性和简易性髋部外骨骼控制器的有效性，促进了可行、适应性和用户自依赖控制律的发展。 

---
# Motion Planning and Control with Unknown Nonlinear Dynamics through Predicted Reachability 

**Title (ZH)**: 基于预测可达性的未知非线性动力学的运动规划与控制 

**Authors**: Zhiquan Zhang, Gokul Puthumanaillam, Manav Vora, Melkior Ornik  

**Link**: [PDF](https://arxiv.org/pdf/2503.03633)  

**Abstract**: Autonomous motion planning under unknown nonlinear dynamics presents significant challenges. An agent needs to continuously explore the system dynamics to acquire its properties, such as reachability, in order to guide system navigation adaptively. In this paper, we propose a hybrid planning-control framework designed to compute a feasible trajectory toward a target. Our approach involves partitioning the state space and approximating the system by a piecewise affine (PWA) system with constrained control inputs. By abstracting the PWA system into a directed weighted graph, we incrementally update the existence of its edges via affine system identification and reach control theory, introducing a predictive reachability condition by exploiting prior information of the unknown dynamics. Heuristic weights are assigned to edges based on whether their existence is certain or remains indeterminate. Consequently, we propose a framework that adaptively collects and analyzes data during mission execution, continually updates the predictive graph, and synthesizes a controller online based on the graph search outcomes. We demonstrate the efficacy of our approach through simulation scenarios involving a mobile robot operating in unknown terrains, with its unknown dynamics abstracted as a single integrator model. 

**Abstract (ZH)**: 在未知非线性动态下的自主运动规划面临重大挑战。本文提出了一种混合规划-控制框架，用于计算目标方向的可行轨迹。该方法通过将状态空间分区，并采用具有受限控制输入的分段线性-affine (PWA) 系统近似系统，逐步通过仿射系统识别和可达控制理论更新PWA系统的边的存在性，引入基于未知动力学先验信息的预测可达性条件。根据边的存在性是确定的还是不确定的，给边分配启发式权重。因此，本文提出了一种框架，在任务执行过程中适应性地收集和分析数据，持续更新预测图，并根据图搜索结果在线综合控制器。通过涉及未知地形中移动机器人操作的仿真场景，将未知动态抽象为单积分器模型，证明了该方法的有效性。 

---
# Coordinated Trajectories for Non-stop Flying Carriers Holding a Cable-Suspended Load 

**Title (ZH)**: 协调航迹规划以实现连续飞行的缆索悬挂载荷运载器 

**Authors**: Chiara Gabellieri, Antonio Franchi  

**Link**: [PDF](https://arxiv.org/pdf/2503.03481)  

**Abstract**: Multirotor UAVs have been typically considered for aerial manipulation, but their scarce endurance prevents long-lasting manipulation tasks. This work demonstrates that the non-stop flights of three or more carriers are compatible with holding a constant pose of a cable-suspended load, thus potentially enabling aerial manipulation with energy-efficient non-stop carriers. It also presents an algorithm for generating the coordinated non-stop trajectories. The proposed method builds upon two pillars: (1)~the choice of $n$ special linearly independent directions of internal forces within the $3n-6$-dimensional nullspace of the grasp matrix of the load, chosen as the edges of a Hamiltonian cycle on the graph that connects the cable attachment points on the load. Adjacent pairs of directions are used to generate $n$ forces evolving on distinct 2D affine subspaces, despite the attachment points being generically in 3D; (2)~the construction of elliptical trajectories within these subspaces by mapping, through appropriate graph coloring, each edge of the Hamiltonian cycle to a periodic coordinate while ensuring that no adjacent coordinates exhibit simultaneous zero derivatives. Combined with conditions for load statics and attachment point positions, these choices ensure that each of the $n$ force trajectories projects onto the corresponding cable constraint sphere with non-zero tangential velocity, enabling perpetual motion of the carriers while the load is still. The theoretical findings are validated through simulations and laboratory experiments with non-stopping multirotor UAVs. 

**Abstract (ZH)**: 基于连续飞行的多旋翼无人机实现高效空中操作的研究 

---
# Tiny Lidars for Manipulator Self-Awareness: Sensor Characterization and Initial Localization Experiments 

**Title (ZH)**: 小尺寸激光雷达用于 manipulator 自我意识：传感器特性化和初步定位实验 

**Authors**: Giammarco Caroleo, Alessandro Albini, Daniele De Martini, Timothy D. Barfoot, Perla Maiolino  

**Link**: [PDF](https://arxiv.org/pdf/2503.03449)  

**Abstract**: For several tasks, ranging from manipulation to inspection, it is beneficial for robots to localize a target object in their surroundings. In this paper, we propose an approach that utilizes coarse point clouds obtained from miniaturized VL53L5CX Time-of-Flight (ToF) sensors (tiny lidars) to localize a target object in the robot's workspace. We first conduct an experimental campaign to calibrate the dependency of sensor readings on relative range and orientation to targets. We then propose a probabilistic sensor model that is validated in an object pose estimation task using a Particle Filter (PF). The results show that the proposed sensor model improves the performance of the localization of the target object with respect to two baselines: one that assumes measurements are free from uncertainty and one in which the confidence is provided by the sensor datasheet. 

**Abstract (ZH)**: 本文提出了一种利用微型VL53L5CX飞行时间（ToF）传感器（小型激光雷达）获取的粗略点云来在机器人工作空间中定位目标物体的方法。我们首先进行实验以校准传感器读数对目标相对距离和方向的依赖性。然后，我们提出了一种概率传感器模型，并通过粒子滤波器（PF）在物体姿态估计任务中进行了验证。结果表明，提出的传感器模型相比于两种基线方法提高了目标物体定位性能：一种假设测量值无不确定性，另一种则通过传感器数据表提供置信度。 

---
# Navigating Intelligence: A Survey of Google OR-Tools and Machine Learning for Global Path Planning in Autonomous Vehicles 

**Title (ZH)**: 智能导航：面向自主车辆全球路径规划的Google OR-Tools和机器学习综述 

**Authors**: Alexandre Benoit, Pedram Asef  

**Link**: [PDF](https://arxiv.org/pdf/2503.03338)  

**Abstract**: We offer a new in-depth investigation of global path planning (GPP) for unmanned ground vehicles, an autonomous mining sampling robot named ROMIE. GPP is essential for ROMIE's optimal performance, which is translated into solving the traveling salesman problem, a complex graph theory challenge that is crucial for determining the most effective route to cover all sampling locations in a mining field. This problem is central to enhancing ROMIE's operational efficiency and competitiveness against human labor by optimizing cost and time. The primary aim of this research is to advance GPP by developing, evaluating, and improving a cost-efficient software and web application. We delve into an extensive comparison and analysis of Google operations research (OR)-Tools optimization algorithms. Our study is driven by the goal of applying and testing the limits of OR-Tools capabilities by integrating Reinforcement Learning techniques for the first time. This enables us to compare these methods with OR-Tools, assessing their computational effectiveness and real-world application efficiency. Our analysis seeks to provide insights into the effectiveness and practical application of each technique. Our findings indicate that Q-Learning stands out as the optimal strategy, demonstrating superior efficiency by deviating only 1.2% on average from the optimal solutions across our datasets. 

**Abstract (ZH)**: 针对自主采矿采样机器人ROMIE的全局路径规划研究：基于旅行推销员问题的优化算法比较与分析 

---
# Supervised Visual Docking Network for Unmanned Surface Vehicles Using Auto-labeling in Real-world Water Environments 

**Title (ZH)**: 基于自动标注的监督视觉对接网络在实际水环境中的应用研究 

**Authors**: Yijie Chu, Ziniu Wu, Yong Yue, Eng Gee Lim, Paolo Paoletti, Xiaohui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.03282)  

**Abstract**: Unmanned Surface Vehicles (USVs) are increasingly applied to water operations such as environmental monitoring and river-map modeling. It faces a significant challenge in achieving precise autonomous docking at ports or stations, still relying on remote human control or external positioning systems for accuracy and safety which limits the full potential of human-out-of-loop deployment for this http URL paper introduces a novel supervised learning pipeline with the auto-labeling technique for USVs autonomous visual docking. Firstly, we designed an auto-labeling data collection pipeline that appends relative pose and image pair to the dataset. This step does not require conventional manual labeling for supervised learning. Secondly, the Neural Dock Pose Estimator (NDPE) is proposed to achieve relative dock pose prediction without the need for hand-crafted feature engineering, camera calibration, and peripheral markers. Moreover, The NDPE can accurately predict the relative dock pose in real-world water environments, facilitating the implementation of Position-Based Visual Servo (PBVS) and low-level motion controllers for efficient and autonomous this http URL show that the NDPE is robust to the disturbance of the distance and the USV velocity. The effectiveness of our proposed solution is tested and validated in real-world water environments, reflecting its capability to handle real-world autonomous docking tasks. 

**Abstract (ZH)**: 无人驾驶水面车辆（USVs）在水体操作如环境监测和河流地图建模中应用日益广泛。它在实现精确自主靠泊方面面临重大挑战，仍依赖远程人工控制或外部定位系统以确保准确性和安全性，这限制了无人监管部署的潜力。本文介绍了一种新的监督学习管道，结合自标注技术实现USVs自主视觉靠泊。首先，我们设计了一种自标注数据采集管道，附加相对位姿和图像对到数据集中。这一步骤无需传统的手动标注即可进行监督学习。其次，我们提出了神经靠泊位姿估计器（NDPE），以实现无需手工特征工程、相机校准和辅助标记的靠泊位姿预测。此外，NDPE可以在真实水体环境中准确预测相对靠泊位姿，促进基于位置的视觉伺服（PBVS）和低级运动控制器的实现，以实现高效和自主的靠泊操作。实验表明，NDPE对距离和USV速度的干扰具有鲁棒性。我们的提出的解决方案在真实水体环境中进行测试和验证，展示了其处理真实世界自主靠泊任务的能力。 

---
# STORM: Spatial-Temporal Iterative Optimization for Reliable Multicopter Trajectory Generation 

**Title (ZH)**: STORM：空间-时间迭代优化方法在可靠多旋翼飞行轨迹生成中的应用 

**Authors**: Jinhao Zhang, Zhexuan Zhou, Wenlong Xia, Youmin Gong, Jie Mei  

**Link**: [PDF](https://arxiv.org/pdf/2503.03252)  

**Abstract**: Efficient and safe trajectory planning plays a critical role in the application of quadrotor unmanned aerial vehicles. Currently, the inherent trade-off between constraint compliance and computational efficiency enhancement in UAV trajectory optimization problems has not been sufficiently addressed. To enhance the performance of UAV trajectory optimization, we propose a spatial-temporal iterative optimization framework. Firstly, B-splines are utilized to represent UAV trajectories, with rigorous safety assurance achieved through strict enforcement of constraints on control points. Subsequently, a set of QP-LP subproblems via spatial-temporal decoupling and constraint linearization is derived. Finally, an iterative optimization strategy incorporating guidance gradients is employed to obtain high-performance UAV trajectories in different scenarios. Both simulation and real-world experimental results validate the efficiency and high-performance of the proposed optimization framework in generating safe and fast trajectories. Our source codes will be released for community reference at this https URL 

**Abstract (ZH)**: 高效的时空迭代优化框架在四旋翼无人机轨迹规划中的应用：缓解约束遵守与计算效率之间的固有trade-off 

---
# AirExo-2: Scaling up Generalizable Robotic Imitation Learning with Low-Cost Exoskeletons 

**Title (ZH)**: AirExo-2：低成本外骨骼助力可泛化机器人imitation learning规模化应用 

**Authors**: Hongjie Fang, Chenxi Wang, Yiming Wang, Jingjing Chen, Shangning Xia, Jun Lv, Zihao He, Xiyan Yi, Yunhan Guo, Xinyu Zhan, Lixin Yang, Weiming Wang, Cewu Lu, Hao-Shu Fang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03081)  

**Abstract**: Scaling up imitation learning for real-world applications requires efficient and cost-effective demonstration collection methods. Current teleoperation approaches, though effective, are expensive and inefficient due to the dependency on physical robot platforms. Alternative data sources like in-the-wild demonstrations can eliminate the need for physical robots and offer more scalable solutions. However, existing in-the-wild data collection devices have limitations: handheld devices offer restricted in-hand camera observation, while whole-body devices often require fine-tuning with robot data due to action inaccuracies. In this paper, we propose AirExo-2, a low-cost exoskeleton system for large-scale in-the-wild demonstration collection. By introducing the demonstration adaptor to transform the collected in-the-wild demonstrations into pseudo-robot demonstrations, our system addresses key challenges in utilizing in-the-wild demonstrations for downstream imitation learning in real-world environments. Additionally, we present RISE-2, a generalizable policy that integrates 2D and 3D perceptions, outperforming previous imitation learning policies in both in-domain and out-of-domain tasks, even with limited demonstrations. By leveraging in-the-wild demonstrations collected and transformed by the AirExo-2 system, without the need for additional robot demonstrations, RISE-2 achieves comparable or superior performance to policies trained with teleoperated data, highlighting the potential of AirExo-2 for scalable and generalizable imitation learning. Project page: this https URL 

**Abstract (ZH)**: 大范围在野演示收集的低成本exo骨架系统AirExo-2及其应用于可扩展和普适性模仿学习的RISE-2策略 

---
# MochiSwarm: A testbed for robotic blimps in realistic environments 

**Title (ZH)**: MochiSwarm: 一种用于现实环境中的机器人气球的实验平台 

**Authors**: Jiawei Xu, Thong Vu, Diego S. D'Antonio, David Saldaña  

**Link**: [PDF](https://arxiv.org/pdf/2503.03077)  

**Abstract**: Testing aerial robots in tasks such as pickup-and-delivery and surveillance significantly benefits from high energy efficiency and scalability of the deployed robotic system. This paper presents MochiSwarm, an open-source testbed of light-weight robotic blimps, ready for multi-robot operation without external localization. We introduce the system design in hardware, software, and perception, which capitalizes on modularity, low cost, and light weight. The hardware allows for rapid modification, which enables the integration of additional sensors to enhance autonomy for different scenarios. The software framework supports different actuation models and communication between the base station and multiple blimps. The detachable perception module allows independent blimps to perform tasks that involve detection and autonomous actuation. We showcase a differential-drive module as an example, of which the autonomy is enabled by visual servoing using the perception module. A case study of pickup-and-delivery tasks with up to 12 blimps highlights the autonomy of the MochiSwarm without external infrastructures. 

**Abstract (ZH)**: 测试诸如收发物品和 surveillance 等任务的空中机器人显著受益于部署机器人系统的高度能效和可扩展性。本文介绍了 MochiSwarm，一个开源的轻量级空中无人机测试平台，无需外部定位即可进行多机器人操作。我们在硬件、软件和感知方面介绍了系统设计，注重模块化、低成本和轻量化。硬件允许快速修改，便于集成额外传感器以增强不同场景下的自主性。软件框架支持不同的驱动模型，并在基站与多个无人机之间实现通信。可拆卸的感知模块使独立无人机能够执行涉及检测和自主操作的任务。我们以一个差分驱动模块为例，该模块的自主性通过使用感知模块实现的视觉伺服技术启用。长达12个无人机的收发物品任务案例研究表明，MochiSwarm 在没有外部基础设施的情况下具备自主性。 

---
# Physically-Feasible Reactive Synthesis for Terrain-Adaptive Locomotion via Trajectory Optimization and Symbolic Repair 

**Title (ZH)**: 基于轨迹优化和符号修复的地貌自适应 locomotion 的物理可行反应合成 

**Authors**: Ziyi Zhou, Qian Meng, Hadas Kress-Gazit, Ye Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.03071)  

**Abstract**: We propose an integrated planning framework for quadrupedal locomotion over dynamically changing, unforeseen terrains. Existing approaches either rely on heuristics for instantaneous foothold selection--compromising safety and versatility--or solve expensive trajectory optimization problems with complex terrain features and long time horizons. In contrast, our framework leverages reactive synthesis to generate correct-by-construction controllers at the symbolic level, and mixed-integer convex programming (MICP) for dynamic and physically feasible footstep planning for each symbolic transition. We use a high-level manager to reduce the large state space in synthesis by incorporating local environment information, improving synthesis scalability. To handle specifications that cannot be met due to dynamic infeasibility, and to minimize costly MICP solves, we leverage a symbolic repair process to generate only necessary symbolic transitions. During online execution, re-running the MICP with real-world terrain data, along with runtime symbolic repair, bridges the gap between offline synthesis and online execution. We demonstrate, in simulation, our framework's capabilities to discover missing locomotion skills and react promptly in safety-critical environments, such as scattered stepping stones and rebars. 

**Abstract (ZH)**: 一种基于反应合成的四足机器人动态环境适应性规划框架 

---
# ArticuBot: Learning Universal Articulated Object Manipulation Policy via Large Scale Simulation 

**Title (ZH)**: ArticuBot: 通过大规模模拟学习通用articulated物体操作策略 

**Authors**: Yufei Wang, Ziyu Wang, Mino Nakura, Pratik Bhowal, Chia-Liang Kuo, Yi-Ting Chen, Zackory Erickson, David Held  

**Link**: [PDF](https://arxiv.org/pdf/2503.03045)  

**Abstract**: This paper presents ArticuBot, in which a single learned policy enables a robotics system to open diverse categories of unseen articulated objects in the real world. This task has long been challenging for robotics due to the large variations in the geometry, size, and articulation types of such objects. Our system, Articubot, consists of three parts: generating a large number of demonstrations in physics-based simulation, distilling all generated demonstrations into a point cloud-based neural policy via imitation learning, and performing zero-shot sim2real transfer to real robotics systems. Utilizing sampling-based grasping and motion planning, our demonstration generalization pipeline is fast and effective, generating a total of 42.3k demonstrations over 322 training articulated objects. For policy learning, we propose a novel hierarchical policy representation, in which the high-level policy learns the sub-goal for the end-effector, and the low-level policy learns how to move the end-effector conditioned on the predicted goal. We demonstrate that this hierarchical approach achieves much better object-level generalization compared to the non-hierarchical version. We further propose a novel weighted displacement model for the high-level policy that grounds the prediction into the existing 3D structure of the scene, outperforming alternative policy representations. We show that our learned policy can zero-shot transfer to three different real robot settings: a fixed table-top Franka arm across two different labs, and an X-Arm on a mobile base, opening multiple unseen articulated objects across two labs, real lounges, and kitchens. Videos and code can be found on our project website: this https URL. 

**Abstract (ZH)**: 本文介绍了ArticuBot，这是一种单一学习策略使机器人系统能够在现实世界中打开多样化类别的未见过的关节对象的方法。由于这类对象在几何形状、尺寸和关节类型上存在大量变化，因此机器人长期以来一直难以完成这项任务。我们的系统Articubot由三部分组成：在物理基础上的模拟中生成大量演示，通过模拟学习将所有生成的演示总结为基于点云的神经策略，以及执行零样本模拟到现实机器人系统的转移。利用基于采样的抓取和运动规划，我们的演示泛化管道快速且有效，共生成了42300个演示数据，涵盖了322个训练中的关节对象。在策略学习方面，我们提出了一种新型层次策略表示，在这种表示中，高层策略学习末端执行器的目标子项，而低层策略学习在预测目标条件下如何移动末端执行器。我们证明，这种层次方法在对象级别泛化方面显著优于非层次版本。此外，我们还为高层策略提出了一种新的加权位移模型，将预测与场景的现有3D结构联系起来，优于其他策略表示。我们展示了我们的学习策略能够在三个不同的现实机器人设置中实现零样本转移：在两个不同实验室中的固定桌面Franka手臂，以及移动基座上的X-Arm，打开两个实验室、真实休息区和厨房中的多个未见过的关节对象。更多视频和代码请参见我们的项目网站：this https URL。 

---
