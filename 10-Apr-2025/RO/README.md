# UAV Position Estimation using a LiDAR-based 3D Object Detection Method 

**Title (ZH)**: 基于LiDAR的3D物体检测方法的无人机位置估计算法 

**Authors**: Uthman Olawoye, Jason N. Gross  

**Link**: [PDF](https://arxiv.org/pdf/2504.07028)  

**Abstract**: This paper explores the use of applying a deep learning approach for 3D object detection to compute the relative position of an Unmanned Aerial Vehicle (UAV) from an Unmanned Ground Vehicle (UGV) equipped with a LiDAR sensor in a GPS-denied environment. This was achieved by evaluating the LiDAR sensor's data through a 3D detection algorithm (PointPillars). The PointPillars algorithm incorporates a column voxel point-cloud representation and a 2D Convolutional Neural Network (CNN) to generate distinctive point-cloud features representing the object to be identified, in this case, the UAV. The current localization method utilizes point-cloud segmentation, Euclidean clustering, and predefined heuristics to obtain the relative position of the UAV. Results from the two methods were then compared to a reference truth solution. 

**Abstract (ZH)**: 本文探索了采用深度学习方法进行三维物体检测，以计算GPS受限制环境中配备LiDAR传感器的无人地面车（UGV）相对于无人 aerial 车辆（UAV）的相对位置。通过使用PointPillars算法评估LiDAR传感器的数据来实现这一目标。PointPillars算法结合了柱体体素点云表示和二维卷积神经网络（CNN），以生成用于识别物体（本例中为UAV）的特征点云表示。当前的定位方法利用点云分割、欧几里得聚类和预定义启发式规则来获取UAV的相对位置。然后将两种方法的结果与参考真实解进行比较。 

---
# Leveraging GCN-based Action Recognition for Teleoperation in Daily Activity Assistance 

**Title (ZH)**: 基于GCN的行动识别在日常活动辅助远程操作中的应用 

**Authors**: Thomas M. Kwok, Jiaan Li, Yue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.07001)  

**Abstract**: Caregiving of older adults is an urgent global challenge, with many older adults preferring to age in place rather than enter residential care. However, providing adequate home-based assistance remains difficult, particularly in geographically vast regions. Teleoperated robots offer a promising solution, but conventional motion-mapping teleoperation imposes unnatural movement constraints on operators, leading to muscle fatigue and reduced usability. This paper presents a novel teleoperation framework that leverages action recognition to enable intuitive remote robot control. Using our simplified Spatio-Temporal Graph Convolutional Network (S-ST-GCN), the system recognizes human actions and executes corresponding preset robot trajectories, eliminating the need for direct motion synchronization. A finite-state machine (FSM) is integrated to enhance reliability by filtering out misclassified actions. Our experiments demonstrate that the proposed framework enables effortless operator movement while ensuring accurate robot execution. This proof-of-concept study highlights the potential of teleoperation with action recognition for enabling caregivers to remotely assist older adults during activities of daily living (ADLs). Future work will focus on improving the S-ST-GCN's recognition accuracy and generalization, integrating advanced motion planning techniques to further enhance robotic autonomy in older adult care, and conducting a user study to evaluate the system's telepresence and ease of control. 

**Abstract (ZH)**: 老年人居家照护是一个紧迫的全球挑战，许多老年人更倾向于原地养老而非入住护理院。然而，在地理面积广阔的地区，提供充足的居家支持仍然困难。远程操作机器人提供了一种有前景的解决方案，但传统的运动映射远程操作在操作员身上施加了不自然的运动限制，导致肌肉疲劳并降低了使用性。本文提出了一种新的远程操作框架，利用动作识别使远程机器人控制更加直观。通过我们简化的时空图卷积网络（S-ST-GCN），系统识别人类动作并执行相应的预设机器人轨迹，从而消除直接运动同步的需要。集成一个有限状态机（FSM）以通过过滤错误分类的动作来提高可靠性。我们的实验表明，所提出框架使操作员的运动更加轻松，同时确保机器人执行的准确性。这项概念验证研究突显了动作识别远程操作的潜力，使照护者能够在日常生活中远程协助老年人。未来的工作将专注于提高S-ST-GCN的动作识别准确性和泛化能力，整合先进的运动规划技术以进一步增强老年人照护中的机器人自主性，并进行用户研究以评估系统的在场感和可控性。 

---
# RayFronts: Open-Set Semantic Ray Frontiers for Online Scene Understanding and Exploration 

**Title (ZH)**: RayFronts：开集语义射线前沿及其在在线场景理解与探索中的应用 

**Authors**: Omar Alama, Avigyan Bhattacharya, Haoyang He, Seungchan Kim, Yuheng Qiu, Wenshan Wang, Cherie Ho, Nikhil Keetha, Sebastian Scherer  

**Link**: [PDF](https://arxiv.org/pdf/2504.06994)  

**Abstract**: Open-set semantic mapping is crucial for open-world robots. Current mapping approaches either are limited by the depth range or only map beyond-range entities in constrained settings, where overall they fail to combine within-range and beyond-range observations. Furthermore, these methods make a trade-off between fine-grained semantics and efficiency. We introduce RayFronts, a unified representation that enables both dense and beyond-range efficient semantic mapping. RayFronts encodes task-agnostic open-set semantics to both in-range voxels and beyond-range rays encoded at map boundaries, empowering the robot to reduce search volumes significantly and make informed decisions both within & beyond sensory range, while running at 8.84 Hz on an Orin AGX. Benchmarking the within-range semantics shows that RayFronts's fine-grained image encoding provides 1.34x zero-shot 3D semantic segmentation performance while improving throughput by 16.5x. Traditionally, online mapping performance is entangled with other system components, complicating evaluation. We propose a planner-agnostic evaluation framework that captures the utility for online beyond-range search and exploration, and show RayFronts reduces search volume 2.2x more efficiently than the closest online baselines. 

**Abstract (ZH)**: 开放集语义映射对于开放世界机器人至关重要。RayFronts：统一表示支持稠密和远距离高效语义映射及其应用壯观分析与评价框架 

---
# Two by Two: Learning Multi-Task Pairwise Objects Assembly for Generalizable Robot Manipulation 

**Title (ZH)**: 两两学习：面向通用机器人 manipulation 的多任务成对对象装配 

**Authors**: Yu Qi, Yuanchen Ju, Tianming Wei, Chi Chu, Lawson L.S. Wong, Huazhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06961)  

**Abstract**: 3D assembly tasks, such as furniture assembly and component fitting, play a crucial role in daily life and represent essential capabilities for future home robots. Existing benchmarks and datasets predominantly focus on assembling geometric fragments or factory parts, which fall short in addressing the complexities of everyday object interactions and assemblies. To bridge this gap, we present 2BY2, a large-scale annotated dataset for daily pairwise objects assembly, covering 18 fine-grained tasks that reflect real-life scenarios, such as plugging into sockets, arranging flowers in vases, and inserting bread into toasters. 2BY2 dataset includes 1,034 instances and 517 pairwise objects with pose and symmetry annotations, requiring approaches that align geometric shapes while accounting for functional and spatial relationships between objects. Leveraging the 2BY2 dataset, we propose a two-step SE(3) pose estimation method with equivariant features for assembly constraints. Compared to previous shape assembly methods, our approach achieves state-of-the-art performance across all 18 tasks in the 2BY2 dataset. Additionally, robot experiments further validate the reliability and generalization ability of our method for complex 3D assembly tasks. 

**Abstract (ZH)**: 3D 组装任务，如家具组装和零部件装配，在日常生活中发挥着重要作用，并代表了未来家庭机器人的必备能力。现有的基准和数据集主要集中在组装几何碎片或工厂零部件上，未能解决日常生活物体交互和组装的复杂性。为解决这一问题，我们提出了 2BY2，一个大规模标注的日常成对物体组装数据集，涵盖了18项细致任务，反映现实场景，如插头插入插座、花瓶中插花和将面包插入烤面包机中。2BY2 数据集包含1,034个实例和517个成对物体，附有姿态和对称性注释，需要同时对齐几何形状并考虑物体之间功能和空间关系的方法。借助2BY2数据集，我们提出了一种基于变换不变特征的两步 SE(3) 姿态估计方法，用于组装约束。与之前的形状组装方法相比，我们的方法在2BY2数据集的18项任务中均实现了最先进的性能。此外，机器人实验进一步验证了我们方法在复杂3D组装任务中的可靠性和泛化能力。 

---
# GraspClutter6D: A Large-scale Real-world Dataset for Robust Perception and Grasping in Cluttered Scenes 

**Title (ZH)**: GraspClutter6D：嘈杂场景中鲁棒感知与抓取的大规模真实世界数据集 

**Authors**: Seunghyeok Back, Joosoon Lee, Kangmin Kim, Heeseon Rho, Geonhyup Lee, Raeyoung Kang, Sangbeom Lee, Sangjun Noh, Youngjin Lee, Taeyeop Lee, Kyoobin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.06866)  

**Abstract**: Robust grasping in cluttered environments remains an open challenge in robotics. While benchmark datasets have significantly advanced deep learning methods, they mainly focus on simplistic scenes with light occlusion and insufficient diversity, limiting their applicability to practical scenarios. We present GraspClutter6D, a large-scale real-world grasping dataset featuring: (1) 1,000 highly cluttered scenes with dense arrangements (14.1 objects/scene, 62.6\% occlusion), (2) comprehensive coverage across 200 objects in 75 environment configurations (bins, shelves, and tables) captured using four RGB-D cameras from multiple viewpoints, and (3) rich annotations including 736K 6D object poses and 9.3B feasible robotic grasps for 52K RGB-D images. We benchmark state-of-the-art segmentation, object pose estimation, and grasping detection methods to provide key insights into challenges in cluttered environments. Additionally, we validate the dataset's effectiveness as a training resource, demonstrating that grasping networks trained on GraspClutter6D significantly outperform those trained on existing datasets in both simulation and real-world experiments. The dataset, toolkit, and annotation tools are publicly available on our project website: this https URL. 

**Abstract (ZH)**: 在拥堵环境中的鲁棒抓取仍然是机器人技术中的一个开放挑战。我们提出了GraspClutter6D数据集，该数据集包含：(1) 1,000个高度拥堵场景，包含密集布置的对象（每场景14.1个物体，62.6%遮挡）；(2) 跨75种环境配置（箱子、货架和桌子）的200个物体的全面覆盖，使用四台RGB-D相机从多个视角捕捉；(3) 丰富的注解，包括73.6万6D物体姿态和52万张RGB-D图像的93亿个可行机器人抓取。我们对比最先进的分割、物体姿态估计和抓取检测方法，以提供拥堵环境中的挑战洞见。此外，我们验证了该数据集作为训练资源的有效性，表明在GraspClutter6D上训练的抓取网络在模拟和真实世界实验中显著优于现有数据集上的训练网络。该数据集、工具包和标注工具已在我们的项目网站上公开：这个链接https://。 

---
# Developing Modular Grasping and Manipulation Pipeline Infrastructure to Streamline Performance Benchmarking 

**Title (ZH)**: 开发模块化抓取与操作流水线基础设施以简化性能基准测试 

**Authors**: Brian Flynn, Kostas Bekris, Berk Calli, Aaron Dollar, Adam Norton, Yu Sun, Holly Yanco  

**Link**: [PDF](https://arxiv.org/pdf/2504.06819)  

**Abstract**: The robot manipulation ecosystem currently faces issues with integrating open-source components and reproducing results. This limits the ability of the community to benchmark and compare the performance of different solutions to one another in an effective manner, instead relying on largely holistic evaluations. As part of the COMPARE Ecosystem project, we are developing modular grasping and manipulation pipeline infrastructure in order to streamline performance benchmarking. The infrastructure will be used towards the establishment of standards and guidelines for modularity and improved open-source development and benchmarking. This paper provides a high-level overview of the architecture of the pipeline infrastructure, experiments conducted to exercise it during development, and future work to expand its modularity. 

**Abstract (ZH)**: 机器人操作生态系统目前面临开源组件集成和结果重现的问题。这限制了社区将不同解决方案进行有效基准测试和性能比较的能力，而是依赖于整体评价。作为COMPARE生态系统项目的一部分，我们正在开发模块化的抓取和操作管道基础设施，以简化性能基准测试。该基础设施将用于建立模块化标准和改进开源开发与基准测试的指南。本文提供管道基础设施架构的高层次概述，在开发过程中进行的实验，以及扩展其模块性的未来工作。 

---
# Towards Efficient Roadside LiDAR Deployment: A Fast Surrogate Metric Based on Entropy-Guided Visibility 

**Title (ZH)**: 基于熵引导可见性的一种快速代理指标 toward 有效路边LiDAR部署 

**Authors**: Yuze Jiang, Ehsan Javanmardi, Manabu Tsukada, Hiroshi Esaki  

**Link**: [PDF](https://arxiv.org/pdf/2504.06772)  

**Abstract**: The deployment of roadside LiDAR sensors plays a crucial role in the development of Cooperative Intelligent Transport Systems (C-ITS). However, the high cost of LiDAR sensors necessitates efficient placement strategies to maximize detection performance. Traditional roadside LiDAR deployment methods rely on expert insight, making them time-consuming. Automating this process, however, demands extensive computation, as it requires not only visibility evaluation but also assessing detection performance across different LiDAR placements. To address this challenge, we propose a fast surrogate metric, the Entropy-Guided Visibility Score (EGVS), based on information gain to evaluate object detection performance in roadside LiDAR configurations. EGVS leverages Traffic Probabilistic Occupancy Grids (TPOG) to prioritize critical areas and employs entropy-based calculations to quantify the information captured by LiDAR beams. This eliminates the need for direct detection performance evaluation, which typically requires extensive labeling and computational resources. By integrating EGVS into the optimization process, we significantly accelerate the search for optimal LiDAR configurations. Experimental results using the AWSIM simulator demonstrate that EGVS strongly correlates with Average Precision (AP) scores and effectively predicts object detection performance. This approach offers a computationally efficient solution for roadside LiDAR deployment, facilitating scalable smart infrastructure development. 

**Abstract (ZH)**: 路边LiDAR传感器的部署对于协同智能交通系统（C-ITS）的发展至关重要。然而，LiDAR传感器的高成本促使需要高效的布放策略以最大化检测性能。传统的路边LiDAR部署方法依赖于专家的经验，使得过程耗时。然而，自动化这一过程需要大量的计算，因为它不仅需要进行可见性评估，还需要评估不同LiDAR布放方案的检测性能。为应对这一挑战，我们提出了一种快速的代理指标——基于信息增益的熵引导可见度分数（EGVS），用于评估路边LiDAR配置中的目标检测性能。EGVS利用交通概率占据网格（TPOG）来优先考虑关键区域，并采用基于熵的计算来量化LiDAR光束捕获的信息。这可以消除对直接检测性能评估的需要，后者通常需要大量的标签和计算资源。通过将EGVS集成到优化过程中，显著加速了寻找最优LiDAR配置的过程。使用AWSIM仿真器进行的实验结果表明，EGVS与平均精度（AP）分数高度相关，并能有效预测目标检测性能。这种方法提供了一种计算高效的目标检测解决方案，支持可扩展的智能基础设施开发。 

---
# Interactive Expressive Motion Generation Using Dynamic Movement Primitives 

**Title (ZH)**: 基于动态运动部件的互动表达性运动生成 

**Authors**: Till Hielscher, Andreas Bulling, Kai O. Arras  

**Link**: [PDF](https://arxiv.org/pdf/2504.06735)  

**Abstract**: Our goal is to enable social robots to interact autonomously with humans in a realistic, engaging, and expressive manner. The 12 Principles of Animation [1] are a well-established framework animators use to create movements that make characters appear convincing, dynamic, and emotionally expressive. This paper proposes a novel approach that leverages Dynamic Movement Primitives (DMPs) to implement key animation principles, providing a learnable, explainable, modulable, online adaptable and composable model for automatic expressive motion generation. DMPs, originally developed for general imitation learning in robotics and grounded in a spring-damper system design, offer mathematical properties that make them particularly suitable for this task. Specifically, they enable modulation of the intensities of individual principles and facilitate the decomposition of complex, expressive motion sequences into learnable and parametrizable primitives. We present the mathematical formulation of the parameterized animation principles and demonstrate the effectiveness of our framework through experiments and application on three robotic platforms with different kinematic configurations, in simulation, on actual robots and in a user study. Our results show that the approach allows for creating diverse and nuanced expressions using a single base model. 

**Abstract (ZH)**: 我们的目标是使社会机器人能够以现实、引人入胜和富有表现力的方式自主与人类互动。动画的12条原则[1]是一个经过验证的框架，动画师利用这一框架创造出使角色显得可信、动态和情感表达丰富的动作。本文提出了一种新颖的方法，利用动态运动本原（DMPs）实现关键的动画原则，提供了一个可学习、可解释、可模块化、在线可适应和可组合的自动表现性动作生成模型。DMPs最初是为了一般机器人模仿学习而开发的，并基于弹簧阻尼系统设计，具有数学性质使其特别适合这一任务。具体而言，它们使个体原则的强度调整成为可能，并使复杂的、表现性的动作序列能被分解为可学习和参数化的本原。本文展示了参数化动画原则的数学描述，并通过在不同 cinématique配置的三台机器人平台上进行的实验和应用演示了本框架的有效性。我们的结果表明，该方法能够利用单一基本模型创建多样化且细腻的表情。 

---
# Learning global control of underactuated systems with Model-Based Reinforcement Learning 

**Title (ZH)**: 基于模型的强化学习在欠驱动系统的全局控制学习中应用 

**Authors**: Niccolò Turcato, Marco Calì, Alberto Dalla Libera, Giulio Giacomuzzo, Ruggero Carli, Diego Romeres  

**Link**: [PDF](https://arxiv.org/pdf/2504.06721)  

**Abstract**: This short paper describes our proposed solution for the third edition of the "AI Olympics with RealAIGym" competition, held at ICRA 2025. We employed Monte-Carlo Probabilistic Inference for Learning Control (MC-PILCO), an MBRL algorithm recognized for its exceptional data efficiency across various low-dimensional robotic tasks, including cart-pole, ball \& plate, and Furuta pendulum systems. MC-PILCO optimizes a system dynamics model using interaction data, enabling policy refinement through simulation rather than direct system data optimization. This approach has proven highly effective in physical systems, offering greater data efficiency than Model-Free (MF) alternatives. Notably, MC-PILCO has previously won the first two editions of this competition, demonstrating its robustness in both simulated and real-world environments. Besides briefly reviewing the algorithm, we discuss the most critical aspects of the MC-PILCO implementation in the tasks at hand: learning a global policy for the pendubot and acrobot systems. 

**Abstract (ZH)**: 这篇短文描述了我们为2025年ICRA举办的第三次“RealAIGym AI奥运”竞赛提出的方法，采用了蒙特卡洛概率推理学习控制（MC-PILCO）算法，这是一种被广泛认可的数据效率优异的模型随机制导策略学习算法，适用于各类低维度机器人任务，包括平衡杆、球与板以及Furuta摆系统。MC-PILCO通过对交互数据优化系统动力学模型，能够在仿真中进行策略优化，而不是直接使用系统数据进行优化。这种方法在物理系统中显示出了高度的有效性，其数据效率超过了无模型方法（MBRL）。值得一提的是，MC-PILCO之前已经在该竞赛的前两届比赛中获胜，证明了其在模拟和实际环境中的稳健性。除了简要回顾算法外，本文还讨论了MC-PILCO在当前任务中实现的关键方面：学习pendubot和acrobot系统的全局策略。 

---
# Ice-Breakers, Turn-Takers and Fun-Makers: Exploring Robots for Groups with Teenagers 

**Title (ZH)**: 破冰者、轮替说话者和活跃分子：探索适合青少年群体的机器人 

**Authors**: Sarah Gillet, Katie Winkle, Giulia Belgiovine, Iolanda Leite  

**Link**: [PDF](https://arxiv.org/pdf/2504.06718)  

**Abstract**: Successful, enjoyable group interactions are important in public and personal contexts, especially for teenagers whose peer groups are important for self-identity and self-esteem. Social robots seemingly have the potential to positively shape group interactions, but it seems difficult to effect such impact by designing robot behaviors solely based on related (human interaction) literature. In this article, we take a user-centered approach to explore how teenagers envisage a social robot "group assistant". We engaged 16 teenagers in focus groups, interviews, and robot testing to capture their views and reflections about robots for groups. Over the course of a two-week summer school, participants co-designed the action space for such a robot and experienced working with/wizarding it for 10+ hours. This experience further altered and deepened their insights into using robots as group assistants. We report results regarding teenagers' views on the applicability and use of a robot group assistant, how these expectations evolved throughout the study, and their repeat interactions with the robot. Our results indicate that each group moves on a spectrum of need for the robot, reflected in use of the robot more (or less) for ice-breaking, turn-taking, and fun-making as the situation demanded. 

**Abstract (ZH)**: 成功的愉快群体互动在公共和个人情境中至关重要，尤其是对青少年而言，他们的同龄人群组对其自我认同和自尊心至关重要。社会机器人似乎有能力正面塑造群体互动，但仅凭借相关的人际互动文献设计机器人的行为似乎难以实现这一影响。在本文中，我们采取用户中心的方法，探索青少年如何看待社会机器人“群组助手”。我们邀请了16名青少年参与焦点小组讨论、访谈和机器人测试，以捕捉他们对用于群体的机器人的观点和反思。在一个为期两周的暑期学校期间，参与者共同设计了此类机器人的行为空间，并与之/指挥其工作了10小时以上。这种经历进一步改变了并深化了他们对利用机器人作为群组助手的看法。我们报告了关于青少年对机器人群组助手的应用性和使用方式的看法、这些期望在整个研究过程中的演变以及他们与机器人重复互动的结果。我们的结果显示，每个小组对机器人的需求在程度上存在差异，反映在其在破冰、轮流发言和创造乐趣等方面对机器人的使用程度随着情况的变化而变化。 

---
# SDHN: Skewness-Driven Hypergraph Networks for Enhanced Localized Multi-Robot Coordination 

**Title (ZH)**: SDHN：基于偏度的超图网络以增强局部多机器人协调 

**Authors**: Delin Zhao, Yanbo Shan, Chang Liu, Shenghang Lin, Yingxin Shou, Bin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06684)  

**Abstract**: Multi-Agent Reinforcement Learning is widely used for multi-robot coordination, where simple graphs typically model pairwise interactions. However, such representations fail to capture higher-order collaborations, limiting effectiveness in complex tasks. While hypergraph-based approaches enhance cooperation, existing methods often generate arbitrary hypergraph structures and lack adaptability to environmental uncertainties. To address these challenges, we propose the Skewness-Driven Hypergraph Network (SDHN), which employs stochastic Bernoulli hyperedges to explicitly model higher-order multi-robot interactions. By introducing a skewness loss, SDHN promotes an efficient structure with Small-Hyperedge Dominant Hypergraph, allowing robots to prioritize localized synchronization while still adhering to the overall information, similar to human coordination. Extensive experiments on Moving Agents in Formation and Robotic Warehouse tasks validate SDHN's effectiveness, demonstrating superior performance over state-of-the-art baselines. 

**Abstract (ZH)**: 基于偏度驱动的超图网络：促进多机器人高效协调 

---
# Setup-Invariant Augmented Reality for Teaching by Demonstration with Surgical Robots 

**Title (ZH)**: 基于示教机器人的手术机器人教学中不变设置的增强现实方法 

**Authors**: Alexandre Banks, Richard Cook, Septimiu E. Salcudean  

**Link**: [PDF](https://arxiv.org/pdf/2504.06677)  

**Abstract**: Augmented reality (AR) is an effective tool in robotic surgery education as it combines exploratory learning with three-dimensional guidance. However, existing AR systems require expert supervision and do not account for differences in the mentor and mentee robot configurations. To enable novices to train outside the operating room while receiving expert-informed guidance, we present dV-STEAR: an open-source system that plays back task-aligned expert demonstrations without assuming identical setup joint positions between expert and novice. Pose estimation was rigorously quantified, showing a registration error of 3.86 (SD=2.01)mm. In a user study (N=24), dV-STEAR significantly improved novice performance on tasks from the Fundamentals of Laparoscopic Surgery. In a single-handed ring-over-wire task, dV-STEAR increased completion speed (p=0.03) and reduced collision time (p=0.01) compared to dry-lab training alone. During a pick-and-place task, it improved success rates (p=0.004). Across both tasks, participants using dV-STEAR exhibited significantly more balanced hand use and reported lower frustration levels. This work presents a novel educational tool implemented on the da Vinci Research Kit, demonstrates its effectiveness in teaching novices, and builds the foundation for further AR integration into robot-assisted surgery. 

**Abstract (ZH)**: 增强现实（AR）是机器人手术教育的有效工具，因为它将探索性学习与三维指导结合在一起。然而，现有的AR系统需要专家监督，并且不考虑指导者和被指导者机器人配置的区别。为了使新手能够在手术室外接受专家指导进行训练，我们提出了dV-STEAR：一个开源系统，能够在假设专家和新手之间的关节位置不完全相同的情况下，播放任务对齐的专家演示。姿态估计的准确度得到了严格量化，注册误差为3.86（SD=2.01）毫米。在用户研究（N=24）中，dV-STEAR显著提高了新手在《腹腔镜手术基础》任务中的表现。在单手穿环任务中，与仅进行干练室训练相比，dV-STEAR提高了完成速度（p=0.03）并减少了碰撞时间（p=0.01）。在抓取并放置任务中，它提高了成功率（p=0.004）。在两项任务中，使用dV-STEAR的参与者表现出更平衡的手部使用，并报告了较低的挫败感水平。本研究在达芬奇研究套件上实现了一种新颖的教育工具，展示了其在教授新手方面的效果，并为增强现实进一步集成到机器人辅助手术中奠定了基础。 

---
# Dynamic Residual Safe Reinforcement Learning for Multi-Agent Safety-Critical Scenarios Decision-Making 

**Title (ZH)**: 多Agent安全关键场景决策中的动态残差安全强化学习 

**Authors**: Kaifeng Wang, Yinsong Chen, Qi Liu, Xueyuan Li, Xin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2504.06670)  

**Abstract**: In multi-agent safety-critical scenarios, traditional autonomous driving frameworks face significant challenges in balancing safety constraints and task performance. These frameworks struggle to quantify dynamic interaction risks in real-time and depend heavily on manual rules, resulting in low computational efficiency and conservative strategies. To address these limitations, we propose a Dynamic Residual Safe Reinforcement Learning (DRS-RL) framework grounded in a safety-enhanced networked Markov decision process. It's the first time that the weak-to-strong theory is introduced into multi-agent decision-making, enabling lightweight dynamic calibration of safety boundaries via a weak-to-strong safety correction paradigm. Based on the multi-agent dynamic conflict zone model, our framework accurately captures spatiotemporal coupling risks among heterogeneous traffic participants and surpasses the static constraints of conventional geometric rules. Moreover, a risk-aware prioritized experience replay mechanism mitigates data distribution bias by mapping risk to sampling probability. Experimental results reveal that the proposed method significantly outperforms traditional RL algorithms in safety, efficiency, and comfort. Specifically, it reduces the collision rate by up to 92.17%, while the safety model accounts for merely 27% of the main model's parameters. 

**Abstract (ZH)**: 在多Agent关键安全场景中，传统自主驾驶框架在平衡安全约束与任务性能方面面临重大挑战。这些框架难以实时量化动态交互风险，并且高度依赖手动规则，导致计算效率低和保守策略。为解决这些问题，我们提出了一种基于增强安全网络马尔可夫决策过程的动态剩余安全强化学习（DRS-RL）框架。这是首次将“弱到强”理论引入多Agent决策，通过“弱到强”安全修正范式实现轻量级动态安全边界校准。基于多Agent动态冲突区域模型，我们的框架准确捕捉异质交通参与者之间的空间时间耦合风险，超越了传统几何规则的静态约束。此外，一种风险意识优先经验重播机制通过将风险映射到采样概率来减轻数据分布偏差。实验结果表明，所提出的方法在安全、效率和舒适性方面显著优于传统RL算法。具体而言，碰撞率最多可降低92.17%，而安全模型仅占主要模型参数的27%。 

---
# RAMBO: RL-augmented Model-based Optimal Control for Whole-body Loco-manipulation 

**Title (ZH)**: RAMBO：基于模型的强化学习增强全身 locomanoipulation 最优控制 

**Authors**: Jin Cheng, Dongho Kang, Gabriele Fadini, Guanya Shi, Stelian Coros  

**Link**: [PDF](https://arxiv.org/pdf/2504.06662)  

**Abstract**: Loco-manipulation -- coordinated locomotion and physical interaction with objects -- remains a major challenge for legged robots due to the need for both accurate force interaction and robustness to unmodeled dynamics. While model-based controllers provide interpretable dynamics-level planning and optimization, they are limited by model inaccuracies and computational cost. In contrast, learning-based methods offer robustness while struggling with precise modulation of interaction forces. We introduce RAMBO -- RL-Augmented Model-Based Optimal Control -- a hybrid framework that integrates model-based reaction force optimization using a simplified dynamics model and a feedback policy trained with reinforcement learning. The model-based module generates feedforward torques by solving a quadratic program, while the policy provides feedback residuals to enhance robustness in control execution. We validate our framework on a quadruped robot across a diverse set of real-world loco-manipulation tasks -- such as pushing a shopping cart, balancing a plate, and holding soft objects -- in both quadrupedal and bipedal walking. Our experiments demonstrate that RAMBO enables precise manipulation while achieving robust and dynamic locomotion, surpassing the performance of policies trained with end-to-end scheme. In addition, our method enables flexible trade-off between end-effector tracking accuracy with compliance. 

**Abstract (ZH)**: 基于RL增强的模型导向最优控制：腿部机器人协调运动与物体物理交互的研究 

---
# Domain-Conditioned Scene Graphs for State-Grounded Task Planning 

**Title (ZH)**: 领域条件化的场景图用于状态导向的任务规划 

**Authors**: Jonas Herzog, Jiangpin Liu, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06661)  

**Abstract**: Recent robotic task planning frameworks have integrated large multimodal models (LMMs) such as GPT-4V. To address grounding issues of such models, it has been suggested to split the pipeline into perceptional state grounding and subsequent state-based planning. As we show in this work, the state grounding ability of LMM-based approaches is still limited by weaknesses in granular, structured, domain-specific scene understanding. To address this shortcoming, we develop a more structured state grounding framework that features a domain-conditioned scene graph as its scene representation. We show that such representation is actionable in nature as it is directly mappable to a symbolic state in classical planning languages such as PDDL. We provide an instantiation of our state grounding framework where the domain-conditioned scene graph generation is implemented with a lightweight vision-language approach that classifies domain-specific predicates on top of domain-relevant object detections. Evaluated across three domains, our approach achieves significantly higher state estimation accuracy and task planning success rates compared to the previous LMM-based approaches. 

**Abstract (ZH)**: 基于大型多模态模型的近期机器人任务规划框架已经整合了如GPT-4V等大型多模态模型。为了应对这些模型的对接地问题，建议将管道拆分为感知状态对接地和后续基于状态的规划。正如我们在本文中所展示的，基于LMM的方法在细粒度、结构化和领域特定场景理解方面的局限性仍然限制了其状态对接地能力。为解决这一不足，我们开发了一种更具结构化的状态对接地框架，其场景表示为核心任务条件化的场景图。我们展示了这种表示本质上是可操作的，因为可以直接映射到经典规划语言（如PDDL）中的符号状态。我们提供了一种状态对接地框架的具体实现，其中核心任务条件化的场景图生成使用了轻量级的视觉语言方法，在与核心任务相关的目标检测基础上分类领域特定谓词。在三个领域进行评估，我们的方法在状态估计准确性和任务规划成功率方面显著优于之前的LMM方法。 

---
# Collision avoidance from monocular vision trained with novel view synthesis 

**Title (ZH)**: 基于新颖视角合成训练的单目视觉碰撞避免 

**Authors**: Valentin Tordjman--Levavasseur, Stéphane Caron  

**Link**: [PDF](https://arxiv.org/pdf/2504.06651)  

**Abstract**: Collision avoidance can be checked in explicit environment models such as elevation maps or occupancy grids, yet integrating such models with a locomotion policy requires accurate state estimation. In this work, we consider the question of collision avoidance from an implicit environment model. We use monocular RGB images as inputs and train a collisionavoidance policy from photorealistic images generated by 2D Gaussian splatting. We evaluate the resulting pipeline in realworld experiments under velocity commands that bring the robot on an intercept course with obstacles. Our results suggest that RGB images can be enough to make collision-avoidance decisions, both in the room where training data was collected and in out-of-distribution environments. 

**Abstract (ZH)**: 基于隐式环境模型的碰撞 avoidance 从单目 RGB 图像中学习 avoidance 策略 

---
# Design and use of devices to assist movement of the upper limb: review of the literature 

**Title (ZH)**: 辅助上肢运动的装置的设计与应用：文献综述 

**Authors**: Charlotte Le Goff, Pauline Coignard, Christine Azevedo-Coste, Franck Geffard, Charles Fattal  

**Link**: [PDF](https://arxiv.org/pdf/2504.06640)  

**Abstract**: This article explores assistive devices for upper limb movement in people with disabilities through a systematic review based on the PRISMA methodology. The studied devices encompass technologies ranging from orthoses to advanced robotics, aiming to compensate for or supplement motor impairments. The results highlight the diversity of applications (rehabilitation, daily living activities), targeted body segments (distal, proximal, or global), as well as control mechanisms and interfaces used. However, despite the variety of promising prototypes, few devices are commercially available, limiting their real impact on end users. Existing technologies, while effective in improving functional autonomy and quality of life, still face challenges in terms of ergonomics, cost, and portability. In conclusion, this article emphasizes the importance of a user-centered approach and proposes avenues for the development of innovative, modular, and accessible assistive devices. 

**Abstract (ZH)**: 本文通过基于PRISMA方法的系统评价，探讨了残疾人上肢运动辅助设备，涵盖了从矫形器到先进机器人等技术，旨在补偿或补充运动功能障碍。研究结果强调了应用多样性（康复、日常活动）、目标身体部位（远端、近端或全身）以及使用的控制机制和接口，尽管存在许多有前景的原型，但可供商业使用的装置较少，限制了其对终端用户的实际影响。现有技术虽然在提高功能性自主性和生活质量方面有效，但仍面临人机工程学、成本和便携性等方面的挑战。综上所述，本文强调了以用户为中心的方法的重要性，并提出了开发创新、模块化和可访问的辅助设备的途径。 

---
# Overcoming Dynamic Environments: A Hybrid Approach to Motion Planning for Manipulators 

**Title (ZH)**: 克服动态环境： manipulator 运动规划的混合方法 

**Authors**: Ho Minh Quang Ngo, Dac Dang Khoa Nguyen, Dinh Tung Le, Gavin Paul  

**Link**: [PDF](https://arxiv.org/pdf/2504.06596)  

**Abstract**: Robotic manipulators operating in dynamic and uncertain environments require efficient motion planning to navigate obstacles while maintaining smooth trajectories. Velocity Potential Field (VPF) planners offer real-time adaptability but struggle with complex constraints and local minima, leading to suboptimal performance in cluttered spaces. Traditional approaches rely on pre-planned trajectories, but frequent recomputation is computationally expensive. This study proposes a hybrid motion planning approach, integrating an improved VPF with a Sampling-Based Motion Planner (SBMP). The SBMP ensures optimal path generation, while VPF provides real-time adaptability to dynamic obstacles. This combination enhances motion planning efficiency, stability, and computational feasibility, addressing key challenges in uncertain environments such as warehousing and surgical robotics. 

**Abstract (ZH)**: 机器人操作员在动态和不确定环境中的运动规划需要高效地导航障碍物并保持平滑的轨迹。本文提出了一种集成改进的 Velocity Potential Field (VPF) 和基于采样的运动规划器 (SBMP) 的混合运动规划方法。SBMP 确保路径生成最优，而 VPF 提供对动态障碍物的实时适应性。这种组合增强了运动规划的效率、稳定性和计算可行性，解决了仓储和外科手术机器人等不确定环境中的关键挑战。 

---
# A Multi-Modal Interaction Framework for Efficient Human-Robot Collaborative Shelf Picking 

**Title (ZH)**: 多模态交互框架-efficient人类-机器人协作货架拾取 

**Authors**: Abhinav Pathak, Kalaichelvi Venkatesan, Tarek Taha, Rajkumar Muthusamy  

**Link**: [PDF](https://arxiv.org/pdf/2504.06593)  

**Abstract**: The growing presence of service robots in human-centric environments, such as warehouses, demands seamless and intuitive human-robot collaboration. In this paper, we propose a collaborative shelf-picking framework that combines multimodal interaction, physics-based reasoning, and task division for enhanced human-robot teamwork.
The framework enables the robot to recognize human pointing gestures, interpret verbal cues and voice commands, and communicate through visual and auditory feedback. Moreover, it is powered by a Large Language Model (LLM) which utilizes Chain of Thought (CoT) and a physics-based simulation engine for safely retrieving cluttered stacks of boxes on shelves, relationship graph for sub-task generation, extraction sequence planning and decision making. Furthermore, we validate the framework through real-world shelf picking experiments such as 1) Gesture-Guided Box Extraction, 2) Collaborative Shelf Clearing and 3) Collaborative Stability Assistance. 

**Abstract (ZH)**: 服务机器人在以人为中心的环境中（如仓库）的应用日益增多，要求实现无缝且直观的人机协作。本文提出了一种结合多模态交互、物理推理和任务分配的协作式货架拣选框架，以增强人机团队合作。该框架使机器人能够识别人类的手势、解释口头提示和语音命令，并通过视觉和听觉反馈进行沟通。此外，该框架由大型语言模型（LLM）驱动，利用思维链（CoT）和物理仿真引擎安全地从货架上提取杂乱的纸箱堆，使用子任务生成关系图进行提取序列规划和决策。我们通过实际的货架拣选实验，如1）手势引导的纸箱提取，2）协作式货架清理，3）协作式稳定性辅助，验证了该框架。 

---
# Sim-to-Real of Humanoid Locomotion Policies via Joint Torque Space Perturbation Injection 

**Title (ZH)**: 基于关节扭矩空间扰动注入的类人行走策略的Sim-to-Real转化 

**Authors**: Woohyun Cha, Junhyeok Cha, Jaeyong Shin, Donghyeon Kim, Jaeheung Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.06585)  

**Abstract**: This paper proposes a novel alternative to existing sim-to-real methods for training control policies with simulated experiences. Prior sim-to-real methods for legged robots mostly rely on the domain randomization approach, where a fixed finite set of simulation parameters is randomized during training. Instead, our method adds state-dependent perturbations to the input joint torque used for forward simulation during the training phase. These state-dependent perturbations are designed to simulate a broader range of reality gaps than those captured by randomizing a fixed set of simulation parameters. Experimental results show that our method enables humanoid locomotion policies that achieve greater robustness against complex reality gaps unseen in the training domain. 

**Abstract (ZH)**: 本文提出了一种新的替代现有 sim-to-real 方法的方案，用于使用模拟经验训练控制策略。现有的用于腿足机器人的 sim-to-real 方法主要依赖于领域随机化方法，在训练过程中随机化固定的一组仿真参数。相比之下，我们的方法在训练阶段通过对输入关节扭矩添加状态依赖的扰动来模拟更广泛的现实差距。实验结果表明，我们的方法能够使类人行走策略在训练领域未见的复杂现实差距中表现出更强的鲁棒性。 

---
# CAFE-AD: Cross-Scenario Adaptive Feature Enhancement for Trajectory Planning in Autonomous Driving 

**Title (ZH)**: CAFE-AD：跨场景自适应特征增强在自动驾驶路径规划中的应用 

**Authors**: Junrui Zhang, Chenjie Wang, Jie Peng, Haoyu Li, Jianmin Ji, Yu Zhang, Yanyong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06584)  

**Abstract**: Imitation learning based planning tasks on the nuPlan dataset have gained great interest due to their potential to generate human-like driving behaviors. However, open-loop training on the nuPlan dataset tends to cause causal confusion during closed-loop testing, and the dataset also presents a long-tail distribution of scenarios. These issues introduce challenges for imitation learning. To tackle these problems, we introduce CAFE-AD, a Cross-Scenario Adaptive Feature Enhancement for Trajectory Planning in Autonomous Driving method, designed to enhance feature representation across various scenario types. We develop an adaptive feature pruning module that ranks feature importance to capture the most relevant information while reducing the interference of noisy information during training. Moreover, we propose a cross-scenario feature interpolation module that enhances scenario information to introduce diversity, enabling the network to alleviate over-fitting in dominant scenarios. We evaluate our method CAFE-AD on the challenging public nuPlan Test14-Hard closed-loop simulation benchmark. The results demonstrate that CAFE-AD outperforms state-of-the-art methods including rule-based and hybrid planners, and exhibits the potential in mitigating the impact of long-tail distribution within the dataset. Additionally, we further validate its effectiveness in real-world environments. The code and models will be made available at this https URL. 

**Abstract (ZH)**: 基于模仿学习的nuPlan数据集上的规划任务引起了广泛关注，因其潜在的人类驾驶行为生成能力。然而，对nuPlan数据集进行开环训练在闭环测试中往往会引发因果混淆问题，且数据集也呈现出场景的长尾分布。这些问题给模仿学习带来了挑战。为解决这些问题，我们提出了一种名为CAFE-AD的方法，即跨场景自适应特征增强用于自主驾驶的轨迹规划，旨在跨不同类型的场景增强特征表示。我们开发了一种自适应特征剪枝模块，用于评估特征的重要性和排名，从而在训练过程中捕获最相关的信息并减少噪声信息的干扰。此外，我们提出了一种跨场景特征插值模块，用于增强场景信息，引入多样性，使网络能够在主流场景中减轻过拟合。我们在具有挑战性的公共nuPlan Test14-Hard闭环仿真基准上评估了我们的方法CAFE-AD。结果表明，CAFE-AD在抵消数据集内长尾分布的影响方面优于现有的基于规则和混合规划的方法，并且在真实环境中的有效性得到了进一步验证。代码和模型将在以下链接处公开。 

---
# ASHiTA: Automatic Scene-grounded HIerarchical Task Analysis 

**Title (ZH)**: ASHiTA: 自动场景grounded层次任务分析 

**Authors**: Yun Chang, Leonor Fermoselle, Duy Ta, Bernadette Bucher, Luca Carlone, Jiuguang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06553)  

**Abstract**: While recent work in scene reconstruction and understanding has made strides in grounding natural language to physical 3D environments, it is still challenging to ground abstract, high-level instructions to a 3D scene. High-level instructions might not explicitly invoke semantic elements in the scene, and even the process of breaking a high-level task into a set of more concrete subtasks, a process called hierarchical task analysis, is environment-dependent. In this work, we propose ASHiTA, the first framework that generates a task hierarchy grounded to a 3D scene graph by breaking down high-level tasks into grounded subtasks. ASHiTA alternates LLM-assisted hierarchical task analysis, to generate the task breakdown, with task-driven 3D scene graph construction to generate a suitable representation of the environment. Our experiments show that ASHiTA performs significantly better than LLM baselines in breaking down high-level tasks into environment-dependent subtasks and is additionally able to achieve grounding performance comparable to state-of-the-art methods. 

**Abstract (ZH)**: 基于3D场景图的高阶任务分解和环境接地方法 

---
# OPAL: Encoding Causal Understanding of Physical Systems for Robot Learning 

**Title (ZH)**: OPAL: 编码物理系统因果理解的机器人学习方法 

**Authors**: Daniel Tcheurekdjian, Joshua Klasmeier, Tom Cooney, Christopher McCann, Tyler Fenstermaker  

**Link**: [PDF](https://arxiv.org/pdf/2504.06538)  

**Abstract**: We present OPAL (Operant Physical Agent with Language), a novel vision-language-action architecture that introduces topological constraints to flow matching for robotic control. To do so, we further introduce topological attention. Our approach models action sequences as topologically-structured representations with non-trivial constraints. Experimental results across 10 complex manipulation tasks demonstrate OPAL's superior performance compared to previous approaches, including Octo, OpenVLA, and ${\pi}$0.
Our architecture achieves significant improvements in zero-shot performance without requiring task-specific fine-tuning, while reducing inference computational requirements by 42%. The theoretical guarantees provided by our topological approach result in more coherent long-horizon action sequences. Our results highlight the potential of constraining the search space of learning problems in robotics by deriving from fundamental physical laws, and the possibility of using topological attention to embed causal understanding into transformer architectures. 

**Abstract (ZH)**: OPAL：基于拓扑约束的视知觉行动架构及其在机器人控制中的应用 

---
# Controller Distillation Reduces Fragile Brain-Body Co-Adaptation and Enables Migrations in MAP-Elites 

**Title (ZH)**: 控制器蒸馏降低脆弱的脑体共适应性并 enables 导引迁移在 MAP-Elites 中 

**Authors**: Alican Mertan, Nick Cheney  

**Link**: [PDF](https://arxiv.org/pdf/2504.06523)  

**Abstract**: Brain-body co-optimization suffers from fragile co-adaptation where brains become over-specialized for particular bodies, hindering their ability to transfer well to others. Evolutionary algorithms tend to discard such low-performing solutions, eliminating promising morphologies. Previous work considered applying MAP-Elites, where niche descriptors are based on morphological features, to promote better search over morphology space. In this work, we show that this approach still suffers from fragile co-adaptation: where a core mechanism of MAP-Elites, creating stepping stones through solutions that migrate from one niche to another, is disrupted. We suggest that this disruption occurs because the body mutations that move an offspring to a new morphological niche break the robots' fragile brain-body co-adaptation and thus significantly decrease the performance of those potential solutions -- reducing their likelihood of outcompeting an existing elite in that new niche. We utilize a technique, we call Pollination, that periodically replaces the controllers of certain solutions with a distilled controller with better generalization across morphologies to reduce fragile brain-body co-adaptation and thus promote MAP-Elites migrations. Pollination increases the success of body mutations and the number of migrations, resulting in better quality-diversity metrics. We believe we develop important insights that could apply to other domains where MAP-Elites is used. 

**Abstract (ZH)**: 脑体协同优化受到脆弱共适应性的困扰，其中大脑对特定身体过度专业化，妨碍了其向其他身体转移的能力。进化算法倾向于丢弃这些低性能的解决方案，消除有前途的形态学。先前的研究考虑将MAP-Elites应用于其中，形态学特征作为生态位描述符，以促进更好的形态学空间搜索。在本工作中，我们表明这种方法仍然受到脆弱共适应性的困扰：其中MAP-Elites的核心机制——通过从一个生态位迁移到另一个生态位来创建踏脚石——被干扰。我们认为这种干扰是因为使后代迁移到新形态学生态位的身体变异打破了机器人的脆弱脑体共适应，从而显著降低了这些潜在解决方案的性能——减少了它们战胜现有精英的可能性。我们使用了一种称为Pollination的技术，定期用具有更好跨形态学泛化的精简控制器替换某些解决方案的控制单元，以减少脆弱的脑体共适应，并促进MAP-Elites迁移。Pollination增加了身体变异的成功率和迁移次数，从而提高了质量多样性指标。我们认为我们开发了重要的见解，这些见解可能应用于其他使用MAP-Elites的领域。 

---
# Safe Navigation in Uncertain Crowded Environments Using Risk Adaptive CVaR Barrier Functions 

**Title (ZH)**: 在不确定拥挤环境中基于风险自适应CVaR屏障函数的安全导航 

**Authors**: Xinyi Wang, Taekyung Kim, Bardh Hoxha, Georgios Fainekos, Dimitra Panagou  

**Link**: [PDF](https://arxiv.org/pdf/2504.06513)  

**Abstract**: Robot navigation in dynamic, crowded environments poses a significant challenge due to the inherent uncertainties in the obstacle model. In this work, we propose a risk-adaptive approach based on the Conditional Value-at-Risk Barrier Function (CVaR-BF), where the risk level is automatically adjusted to accept the minimum necessary risk, achieving a good performance in terms of safety and optimization feasibility under uncertainty. Additionally, we introduce a dynamic zone-based barrier function which characterizes the collision likelihood by evaluating the relative state between the robot and the obstacle. By integrating risk adaptation with this new function, our approach adaptively expands the safety margin, enabling the robot to proactively avoid obstacles in highly dynamic environments. Comparisons and ablation studies demonstrate that our method outperforms existing social navigation approaches, and validate the effectiveness of our proposed framework. 

**Abstract (ZH)**: 动态拥挤环境下机器人导航由于障碍模型固有的不确定性而面临重大挑战。本文提出了一种基于条件风险值屏障函数（CVaR-BF）的风险自适应方法，其中风险水平自动调整以接受最低必要的风险，在不确定性条件下实现了良好的安全性和优化可行性。此外，我们引入了一种基于动态区域的屏障函数，通过评估机器人与障碍物的相对状态来表征碰撞 likelihood。通过将风险适应与这种新函数集成，我们的方法能够自适应地扩大安全边际，在高度动态环境中使机器人能够主动避开障碍。比较和消除研究证明了我们方法优于现有的社会导航方法，并验证了我们提出框架的有效性。 

---
# Holistic Fusion: Task- and Setup-Agnostic Robot Localization and State Estimation with Factor Graphs 

**Title (ZH)**: 全局融合：基于因子图的任务和设置无关的机器人定位与状态估计 

**Authors**: Julian Nubert, Turcan Tuna, Jonas Frey, Cesar Cadena, Katherine J. Kuchenbecker, Shehryar Khattak, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2504.06479)  

**Abstract**: Seamless operation of mobile robots in challenging environments requires low-latency local motion estimation (e.g., dynamic maneuvers) and accurate global localization (e.g., wayfinding). While most existing sensor-fusion approaches are designed for specific scenarios, this work introduces a flexible open-source solution for task- and setup-agnostic multimodal sensor fusion that is distinguished by its generality and usability. Holistic Fusion formulates sensor fusion as a combined estimation problem of i) the local and global robot state and ii) a (theoretically unlimited) number of dynamic context variables, including automatic alignment of reference frames; this formulation fits countless real-world applications without any conceptual modifications. The proposed factor-graph solution enables the direct fusion of an arbitrary number of absolute, local, and landmark measurements expressed with respect to different reference frames by explicitly including them as states in the optimization and modeling their evolution as random walks. Moreover, local smoothness and consistency receive particular attention to prevent jumps in the robot state belief. HF enables low-latency and smooth online state estimation on typical robot hardware while simultaneously providing low-drift global localization at the IMU measurement rate. The efficacy of this released framework is demonstrated in five real-world scenarios on three robotic platforms, each with distinct task requirements. 

**Abstract (ZH)**: 无缝操作于具有挑战性的环境中的移动机器人需要低延迟的局部运动估计（例如动态机动）和准确的全局定位（例如路径规划）。本文提出了一种适用于任务和设置无关的多模态传感器融合的灵活开源解决方案，该方案以其通用性和易用性而区别于大多数现有传感器融合方法。Holistic Fusion将传感器融合表述为局部和全球机器人状态以及动态上下文变量（理论上有无限个）的综合估计问题，包括参考帧的自动对齐；这种表述形式无需任何概念上的修改即可适应无数的实际应用。提出的因子图解决方案能够直接融合任意数量的绝对、局部和地标测量，并通过显式地将它们作为状态包含在优化中来表示，并将它们的发展建模为随机游走。此外，局部平滑性和一致性受到特别关注，以防止机器人状态信念中的突变。Holistic Fusion能够在典型的机器人硬件上实现低延迟和平滑的在线状态估计，同时提供以IMU测量速率计算的低漂移全局定位。该发布框架的有效性在三个具有不同任务要求的机器人平台上通过五个实际场景进行了演示。 

---
# Agent-Arena: A General Framework for Evaluating Control Algorithms 

**Title (ZH)**: Agent-Arena: 一个评估控制算法的一般框架 

**Authors**: Halid Abdulrahim Kadi, Kasim Terzić  

**Link**: [PDF](https://arxiv.org/pdf/2504.06468)  

**Abstract**: Robotic research is inherently challenging, requiring expertise in diverse environments and control algorithms. Adapting algorithms to new environments often poses significant difficulties, compounded by the need for extensive hyper-parameter tuning in data-driven methods. To address these challenges, we present Agent-Arena, a Python framework designed to streamline the integration, replication, development, and testing of decision-making policies across a wide range of benchmark environments. Unlike existing frameworks, Agent-Arena is uniquely generalised to support all types of control algorithms and is adaptable to both simulation and real-robot scenarios. Please see our GitHub repository this https URL. 

**Abstract (ZH)**: 机器人研究本质上具有挑战性，需要在多样化的环境和控制算法方面具备专业知识。将算法适应新环境往往存在显著困难，并且数据驱动方法中需要进行广泛的超参数调优。为了解决这些挑战，我们提出了一种名为Agent-Arena的Python框架，该框架旨在简化决策政策在广泛基准环境中的集成、复制、开发和测试。与现有框架不同，Agent-Arena具有高度通用性，能够支持所有类型的控制算法，并且可以适应仿真和真实robots的场景。请查阅我们的GitHub仓库：this https URL。 

---
# Classifying Subjective Time Perception in a Multi-robot Control Scenario Using Eye-tracking Information 

**Title (ZH)**: 基于眼动信息的多机器人控制场景中主观时间感知分类 

**Authors**: Till Aust, Julian Kaduk, Heiko Hamann  

**Link**: [PDF](https://arxiv.org/pdf/2504.06442)  

**Abstract**: As automation and mobile robotics reshape work environments, rising expectations for productivity increase cognitive demands on human operators, leading to potential stress and cognitive overload. Accurately assessing an operator's mental state is critical for maintaining performance and well-being. We use subjective time perception, which can be altered by stress and cognitive load, as a sensitive, low-latency indicator of well-being and cognitive strain. Distortions in time perception can affect decision-making, reaction times, and overall task effectiveness, making it a valuable metric for adaptive human-swarm interaction systems.
We study how human physiological signals can be used to estimate a person's subjective time perception in a human-swarm interaction scenario as example. A human operator needs to guide and control a swarm of small mobile robots. We obtain eye-tracking data that is classified for subjective time perception based on questionnaire data. Our results show that we successfully estimate a person's time perception from eye-tracking data. The approach can profit from individual-based pretraining using only 30 seconds of data. In future work, we aim for robots that respond to human operator needs by automatically classifying physiological data in a closed control loop. 

**Abstract (ZH)**: 随着自动化和移动机器人重塑工作环境，不断提高的工作生产效率期望增加了人类操作者的认知需求，可能导致压力和认知过载。准确评估操作员的心理状态对于维持性能和福祉至关重要。我们使用受压力和认知负荷影响的主观时间感知作为敏感且低延迟的福祉和认知压力指标。时间感知的扭曲会影响决策、反应时间和整体任务效果，使其成为适应性人-群交互系统中重要的指标。
我们研究了在人-群交互场景中如何使用人类生理信号估计个人的主观时间感知。一个操作员需要引导和控制一群小型移动机器人。我们基于问卷数据对眼球追踪数据进行了分类，以估计主观时间感知。结果显示，我们成功地从眼球追踪数据中估计了人的时间感知。该方法可以通过个体预先训练，在仅使用30秒数据的情况下受益。未来的工作目标是开发能够自动分类生理数据并根据闭环控制响应人类操作员需求的机器人。 

---
# DBaS-Log-MPPI: Efficient and Safe Trajectory Optimization via Barrier States 

**Title (ZH)**: DBaS-Log-MPPI：基于障碍状态的高效安全轨迹优化 

**Authors**: Fanxin Wang, Haolong Jiang, Chuyuan Tao, Wenbin Wan, Yikun Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.06437)  

**Abstract**: Optimizing trajectory costs for nonlinear control systems remains a significant challenge. Model Predictive Control (MPC), particularly sampling-based approaches such as the Model Predictive Path Integral (MPPI) method, has recently demonstrated considerable success by leveraging parallel computing to efficiently evaluate numerous trajectories. However, MPPI often struggles to balance safe navigation in constrained environments with effective exploration in open spaces, leading to infeasibility in cluttered conditions. To address these limitations, we propose DBaS-Log-MPPI, a novel algorithm that integrates Discrete Barrier States (DBaS) to ensure safety while enabling adaptive exploration with enhanced feasibility. Our method is efficiently validated through three simulation missions and one real-world experiment, involving a 2D quadrotor and a ground vehicle navigating through cluttered obstacles. We demonstrate that our algorithm surpasses both Vanilla MPPI and Log-MPPI, achieving higher success rates, lower tracking errors, and a conservative average speed. 

**Abstract (ZH)**: 基于DBaS的日志MPPI算法在非线性控制系统轨迹成本优化中的应用 

---
# Extended Version: Multi-Robot Motion Planning with Cooperative Localization 

**Title (ZH)**: 扩展版本：具有合作定位的多机器人运动规划 

**Authors**: Anne Theurkauf, Nisar Ahmed, Morteza Lahijanian  

**Link**: [PDF](https://arxiv.org/pdf/2504.06429)  

**Abstract**: We consider the uncertain multi-robot motion planning (MRMP) problem with cooperative localization (CL-MRMP), under both motion and measurement noise, where each robot can act as a sensor for its nearby teammates. We formalize CL-MRMP as a chance-constrained motion planning problem, and propose a safety-guaranteed algorithm that explicitly accounts for robot-robot correlations. Our approach extends a sampling-based planner to solve CL-MRMP while preserving probabilistic completeness. To improve efficiency, we introduce novel biasing techniques. We evaluate our method across diverse benchmarks, demonstrating its effectiveness in generating motion plans, with significant performance gains from biasing strategies. 

**Abstract (ZH)**: 具有协同定位的不确定多机器人运动规划问题（CL-MRMP）：考虑运动和测量噪声，其中每个机器人可以作为其附近队友的传感器，并将其形式化为机会约束运动规划问题，提出一种保证安全性的算法，明确考虑机器人之间的关联性。同时引入新的偏差技术以提高效率，并通过多样化的基准测试验证了其有效性，偏差策略带来了显著的性能提升。 

---
# Automated Fabrication of Magnetic Soft Microrobots 

**Title (ZH)**: 自动制造磁性软微机器人 

**Authors**: Kaitlyn Clancy, Siwen Xie, Griffin Smith, Onaizah Onaizah  

**Link**: [PDF](https://arxiv.org/pdf/2504.06370)  

**Abstract**: The advent of 3D printing has revolutionized many industries and has had similar improvements for soft robots. However, many challenges persist for these functional devices. Magnetic soft robots require the addition of magnetic particles that must be correctly oriented. There is a significant gap in the automated fabrication of 3D geometric structures with 3D magnetization direction. A fully automated 3D printer was designed to improve accuracy, speed, and reproducibility. This design was able to achieve a circular spot size (voxels) of 1.6mm in diameter. An updated optical system can improve the resolution to a square spot size of 50$\mu$m by 50$\mu$m. The new system achieves higher resolution designs as shown through magneto-mechanical simulations. Various microrobots including 'worm', 'gripper' and 'zipper' designs are evaluated with the new spot size. 

**Abstract (ZH)**: 3D打印的兴起已革新诸多行业，并对软机器人产生了类似的影响。然而，这些功能性装置仍面临诸多挑战。磁性软机器人需要添加正确定向的磁性颗粒。在采用3D磁化方向的3D几何结构的自动化制造方面存在显著差距。设计了一种完全自动化的3D打印机以提高精度、速度和可重复性。该设计实现了直径为1.6毫米的圆形斑点尺寸（体素）。更新的光学系统可将分辨率提高到50微米×50微米的正方形斑点尺寸。新系统通过磁机械模拟实现了更高分辨率的设计。各种微机器人，包括“worm”、“gripper”和“zipper”设计，均在新的斑点尺寸下进行了评估。 

---
# Neural Motion Simulator: Pushing the Limit of World Models in Reinforcement Learning 

**Title (ZH)**: 神经运动模拟器：在强化学习中推动世界模型的极限 

**Authors**: Chenjie Hao, Weyl Lu, Yifan Xu, Yubei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.07095)  

**Abstract**: An embodied system must not only model the patterns of the external world but also understand its own motion dynamics. A motion dynamic model is essential for efficient skill acquisition and effective planning. In this work, we introduce the neural motion simulator (MoSim), a world model that predicts the future physical state of an embodied system based on current observations and actions. MoSim achieves state-of-the-art performance in physical state prediction and provides competitive performance across a range of downstream tasks. This works shows that when a world model is accurate enough and performs precise long-horizon predictions, it can facilitate efficient skill acquisition in imagined worlds and even enable zero-shot reinforcement learning. Furthermore, MoSim can transform any model-free reinforcement learning (RL) algorithm into a model-based approach, effectively decoupling physical environment modeling from RL algorithm development. This separation allows for independent advancements in RL algorithms and world modeling, significantly improving sample efficiency and enhancing generalization capabilities. Our findings highlight that world models for motion dynamics is a promising direction for developing more versatile and capable embodied systems. 

**Abstract (ZH)**: 一种基于体态的系统不仅需要模�arto外部世界的模式，还需要理解其自身的运动动力学。运动动力学模型对于高效的技能获得和有效的规划至关重要。在此工作中，我们引入了神经运动模拟器（MoSim），这是一种世界模型，能够在当前观测和动作的基础上预测体态系统的未来物理状态。MoSim 在物理状态预测方面达到了最先进的性能，并在一系列下游任务中提供了竞争力的性能。这项工作证明，当世界模型足够准确并且能够进行精确的长时间预测时，它可以在想象的世界中促进高效的技能获得，并且甚至能够实现零样本强化学习。此外，MoSim 可以将任何无模型的强化学习（RL）算法转换为基于模型的方法，有效解耦物理环境建模和RL算法开发。这种分离促进了RL算法和世界建模的独立进步，显著提高了样本效率并增强了泛化能力。我们的研究结果表明，用于运动动力学的世界模型是开发更加多功能的体态系统的有前途的方向。 

---
# Adaptive Human-Robot Collaborative Missions using Hybrid Task Planning 

**Title (ZH)**: 适应性的人机协作任务规划研究 

**Authors**: Gricel Vázquez, Alexandros Evangelidis, Sepeedeh Shahbeigi, Simos Gerasimou  

**Link**: [PDF](https://arxiv.org/pdf/2504.06746)  

**Abstract**: Producing robust task plans in human-robot collaborative missions is a critical activity in order to increase the likelihood of these missions completing successfully. Despite the broad research body in the area, which considers different classes of constraints and uncertainties, its applicability is confined to relatively simple problems that can be comfortably addressed by the underpinning mathematically-based or heuristic-driven solver engines. In this paper, we introduce a hybrid approach that effectively solves the task planning problem by decomposing it into two intertwined parts, starting with the identification of a feasible plan and followed by its uncertainty augmentation and verification yielding a set of Pareto optimal plans. To enhance its robustness, adaptation tactics are devised for the evolving system requirements and agents' capabilities. We demonstrate our approach through an industrial case study involving workers and robots undertaking activities within a vineyard, showcasing the benefits of our hybrid approach both in the generation of feasible solutions and scalability compared to native planners. 

**Abstract (ZH)**: 在人类与机器人协作任务中生成鲁棒的任务规划是提高这些任务成功完成几率的关键活动。尽管该领域已有广泛的研究，考虑了不同的约束类别和不确定性，其适用范围仍然局限于相对简单的问题，这些问题可以由基于数学的方法或启发式驱动的求解引擎舒适地解决。本文介绍了一种有效的混合方法，通过将任务规划问题分解为两个相互交织的部分来解决该问题，首先识别可行的计划，然后对其进行不确定性的增强和验证，生成一组 Pareto 优化的计划。为了增强其鲁棒性，我们为不断变化的系统需求和代理的能力设计了适应策略。我们通过涉及工人和机器人在葡萄园中执行活动的工业案例研究，展示了我们混合方法在生成可行解决方案和可扩展性方面的优势，与原生规划器相比。 

---
# Data-driven Fuzzy Control for Time-Optimal Aggressive Trajectory Following 

**Title (ZH)**: 数据驱动的模糊控制以实现时间最优激进轨迹跟踪 

**Authors**: August Phelps, Juan Augusto Paredes Salazar, Ankit Goel  

**Link**: [PDF](https://arxiv.org/pdf/2504.06500)  

**Abstract**: Optimal trajectories that minimize a user-defined cost function in dynamic systems require the solution of a two-point boundary value problem. The optimization process yields an optimal control sequence that depends on the initial conditions and system parameters. However, the optimal sequence may result in undesirable behavior if the system's initial conditions and parameters are erroneous. This work presents a data-driven fuzzy controller synthesis framework that is guided by a time-optimal trajectory for multicopter tracking problems. In particular, we consider an aggressive maneuver consisting of a mid-air flip and generate a time-optimal trajectory by numerically solving the two-point boundary value problem. A fuzzy controller consisting of a stabilizing controller near hover conditions and an autoregressive moving average (ARMA) controller, trained to mimic the time-optimal aggressive trajectory, is constructed using the Takagi-Sugeno fuzzy framework. 

**Abstract (ZH)**: 多旋翼飞行器跟踪问题中由最优轨迹引导的数据驱动模糊控制器合成框架 

---
# Comparing Self-Disclosure Themes and Semantics to a Human, a Robot, and a Disembodied Agent 

**Title (ZH)**: 比较自我披露主题和语义向人类、机器人和 disembodied 代理人的传达差异 

**Authors**: Sophie Chiang, Guy Laban, Emily S. Cross, Hatice Gunes  

**Link**: [PDF](https://arxiv.org/pdf/2504.06374)  

**Abstract**: As social robots and other artificial agents become more conversationally capable, it is important to understand whether the content and meaning of self-disclosure towards these agents changes depending on the agent's embodiment. In this study, we analysed conversational data from three controlled experiments in which participants self-disclosed to a human, a humanoid social robot, and a disembodied conversational agent. Using sentence embeddings and clustering, we identified themes in participants' disclosures, which were then labelled and explained by a large language model. We subsequently assessed whether these themes and the underlying semantic structure of the disclosures varied by agent embodiment. Our findings reveal strong consistency: thematic distributions did not significantly differ across embodiments, and semantic similarity analyses showed that disclosures were expressed in highly comparable ways. These results suggest that while embodiment may influence human behaviour in human-robot and human-agent interactions, people tend to maintain a consistent thematic focus and semantic structure in their disclosures, whether speaking to humans or artificial interlocutors. 

**Abstract (ZH)**: 随着社会机器人和其他人工代理的会话能力不断增强，了解这些代理的体现形式如何影响人们对其的自我披露内容和含义的变化变得尤为重要。本研究分析了三次控制实验中的对话数据，参与者分别对人类、类人社会机器人和无体对话代理进行了自我披露。通过句嵌入和聚类分析，我们识别了参与者披露的主题，并由大型语言模型进行了标记和解释。随后，我们评估了这些主题及其披露的语义结构是否因代理的体现形式而异。研究发现，主题分布表现出较强的一致性，语义相似性分析表明，披露内容以高度一致的方式表达。这些结果表明，尽管体现形式可能会影响人类与机器人和代理之间的互动行为，但在与人类或人工对话伙伴交流时，人们倾向于保持一致的主题焦点和语义结构。 

---
