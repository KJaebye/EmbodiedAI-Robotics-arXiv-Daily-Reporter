# EMMA: Scaling Mobile Manipulation via Egocentric Human Data 

**Title (ZH)**: EMMA: 通过第一人称人体数据扩展移动 manipulation 技术 

**Authors**: Lawrence Y. Zhu, Pranav Kuppili, Ryan Punamiya, Patcharapong Aphiwetsa, Dhruv Patel, Simar Kareer, Sehoon Ha, Danfei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04443)  

**Abstract**: Scaling mobile manipulation imitation learning is bottlenecked by expensive mobile robot teleoperation. We present Egocentric Mobile MAnipulation (EMMA), an end-to-end framework training mobile manipulation policies from human mobile manipulation data with static robot data, sidestepping mobile teleoperation. To accomplish this, we co-train human full-body motion data with static robot data. In our experiments across three real-world tasks, EMMA demonstrates comparable performance to baselines trained on teleoperated mobile robot data (Mobile ALOHA), achieving higher or equivalent task performance in full task success. We find that EMMA is able to generalize to new spatial configurations and scenes, and we observe positive performance scaling as we increase the hours of human data, opening new avenues for scalable robotic learning in real-world environments. Details of this project can be found at this https URL. 

**Abstract (ZH)**: 移动 manipulation 模仿学习的扩展受制于昂贵的移动机器人遥操作。我们提出了第一人称移动 manipulation (EMMA) 框架，该框架从人类移动 manipulation 数据和静态机器人数据中端到端训练移动 manipulation 策略，避开移动遥操作。为了实现这一目标，我们同时训练人类全身运动数据和静态机器人数据。在我们的三次实际任务实验中，EMMA 在全任务成功率方面展示了与基于遥操作移动机器人数据（Mobile ALOHA）训练的基础模型相当或更好的性能。我们发现 EMMA 能够泛化到新的空间配置和场景，并观察到随着人类数据小时数的增加，性能呈现积极的扩展趋势，为实际环境中的可扩展机器人学习开辟了新途径。详细内容请参见 <https://www.example.com>。 

---
# DEXOP: A Device for Robotic Transfer of Dexterous Human Manipulation 

**Title (ZH)**: DEXOP: 一种用于机器人化灵巧人类操作转移的装置 

**Authors**: Hao-Shu Fang, Branden Romero, Yichen Xie, Arthur Hu, Bo-Ruei Huang, Juan Alvarez, Matthew Kim, Gabriel Margolis, Kavya Anbarasu, Masayoshi Tomizuka, Edward Adelson, Pulkit Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2509.04441)  

**Abstract**: We introduce perioperation, a paradigm for robotic data collection that sensorizes and records human manipulation while maximizing the transferability of the data to real robots. We implement this paradigm in DEXOP, a passive hand exoskeleton designed to maximize human ability to collect rich sensory (vision + tactile) data for diverse dexterous manipulation tasks in natural environments. DEXOP mechanically connects human fingers to robot fingers, providing users with direct contact feedback (via proprioception) and mirrors the human hand pose to the passive robot hand to maximize the transfer of demonstrated skills to the robot. The force feedback and pose mirroring make task demonstrations more natural for humans compared to teleoperation, increasing both speed and accuracy. We evaluate DEXOP across a range of dexterous, contact-rich tasks, demonstrating its ability to collect high-quality demonstration data at scale. Policies learned with DEXOP data significantly improve task performance per unit time of data collection compared to teleoperation, making DEXOP a powerful tool for advancing robot dexterity. Our project page is at this https URL. 

**Abstract (ZH)**: perioperation：一种在手术期间用于机器人数据收集的范式，通过传感器化和记录人类操作，同时最大化数据向真实机器人转移的效率 

---
# OVGrasp: Open-Vocabulary Grasping Assistance via Multimodal Intent Detection 

**Title (ZH)**: OVGrasp: 基于多模态意图检测的开放词汇抓取辅助 

**Authors**: Chen Hu, Shan Luo, Letizia Gionfrida  

**Link**: [PDF](https://arxiv.org/pdf/2509.04324)  

**Abstract**: Grasping assistance is essential for restoring autonomy in individuals with motor impairments, particularly in unstructured environments where object categories and user intentions are diverse and unpredictable. We present OVGrasp, a hierarchical control framework for soft exoskeleton-based grasp assistance that integrates RGB-D vision, open-vocabulary prompts, and voice commands to enable robust multimodal interaction. To enhance generalization in open environments, OVGrasp incorporates a vision-language foundation model with an open-vocabulary mechanism, allowing zero-shot detection of previously unseen objects without retraining. A multimodal decision-maker further fuses spatial and linguistic cues to infer user intent, such as grasp or release, in multi-object scenarios. We deploy the complete framework on a custom egocentric-view wearable exoskeleton and conduct systematic evaluations on 15 objects across three grasp types. Experimental results with ten participants demonstrate that OVGrasp achieves a grasping ability score (GAS) of 87.00%, outperforming state-of-the-art baselines and achieving improved kinematic alignment with natural hand motion. 

**Abstract (ZH)**: 基于软外骨骼的层级控制框架OVGrasp：结合RGB-D视觉、开放词汇提示和语音指令实现稳健多模态交互 

---
# Lightweight Kinematic and Static Modeling of Cable-Driven Continuum Robots via Actuation-Space Energy Formulation 

**Title (ZH)**: 基于驱动空间能量公式的一种轻量化连续缆驱动机器人运动学与静态建模方法 

**Authors**: Ke Wu, Yuhao Wang, Kevin Henry, Cesare Stefanini, Gang Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.04119)  

**Abstract**: Continuum robots, inspired by octopus arms and elephant trunks, combine dexterity with intrinsic compliance, making them well suited for unstructured and confined environments. Yet their continuously deformable morphology poses challenges for motion planning and control, calling for accurate but lightweight models. We propose the Lightweight Actuation Space Energy Modeling (LASEM) framework for cable driven continuum robots, which formulates actuation potential energy directly in actuation space. LASEM yields an analytical forward model derived from geometrically nonlinear beam and rod theories via Hamilton's principle, while avoiding explicit modeling of cable backbone contact. It accepts both force and displacement inputs, thereby unifying kinematic and static formulations. Assuming the friction is neglected, the framework generalizes to nonuniform geometries, arbitrary cable routings, distributed loading and axial extensibility, while remaining computationally efficient for real-time use. Numerical simulations validate its accuracy, and a semi-analytical iterative scheme is developed for inverse kinematics. To address discretization in practical robots, LASEM further reformulates the functional minimization as a numerical optimization, which also naturally incorporates cable potential energy without explicit contact modeling. 

**Abstract (ZH)**: 基于缆索驱动连续机器人的轻量级 actuation 空间能量模型框架（LASEM） 

---
# Cloud-Assisted Remote Control for Aerial Robots: From Theory to Proof-of-Concept Implementation 

**Title (ZH)**: 云辅助远程控制的空中机器人技术：从理论到概念验证实现 

**Authors**: Achilleas Santi Seisa, Viswa Narayanan Sankaranarayanan, Gerasimos Damigos, Sumeet Gajanan Satpute, George Nikolakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2509.04095)  

**Abstract**: Cloud robotics has emerged as a promising technology for robotics applications due to its advantages of offloading computationally intensive tasks, facilitating data sharing, and enhancing robot coordination. However, integrating cloud computing with robotics remains a complex challenge due to network latency, security concerns, and the need for efficient resource management. In this work, we present a scalable and intuitive framework for testing cloud and edge robotic systems. The framework consists of two main components enabled by containerized technology: (a) a containerized cloud cluster and (b) the containerized robot simulation environment. The system incorporates two endpoints of a User Datagram Protocol (UDP) tunnel, enabling bidirectional communication between the cloud cluster container and the robot simulation environment, while simulating realistic network conditions. To achieve this, we consider the use case of cloud-assisted remote control for aerial robots, while utilizing Linux-based traffic control to introduce artificial delay and jitter, replicating variable network conditions encountered in practical cloud-robot deployments. 

**Abstract (ZH)**: 云 robotics 作为一种具有卸载计算密集型任务、促进数据共享以及增强机器人协调优势的技术，在机器人应用中展现出令人期待的前景。然而，将云计算与机器人技术整合仍是一项复杂的挑战，主要由于网络延迟、安全问题以及高效资源管理的需求。在本文中，我们提出了一种可扩展且直观的框架，用于测试云和边缘机器人系统。该框架由两种主要组件构成，基于容器化技术实现：（a）容器化云集群和（b）容器化机器人仿真环境。该系统包括用户数据报协议（UDP）隧道的两个端点，允许云集群容器与机器人仿真环境之间的双向通信，并模拟实际网络条件。为了实现这一目标，我们以云辅助远程控制固定翼机器人作为应用场景，利用基于 Linux 的流量控制引入人工延迟和抖动，以模拟实际云-机器人部署中遇到的可变网络条件。 

---
# Object-Reconstruction-Aware Whole-body Control of Mobile Manipulators 

**Title (ZH)**: 基于物体重建的全尺寸机器人 manipulator 控制 

**Authors**: Fatih Dursun, Bruno Vilhena Adorno, Simon Watson, Wei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2509.04094)  

**Abstract**: Object reconstruction and inspection tasks play a crucial role in various robotics applications. Identifying paths that reveal the most unknown areas of the object becomes paramount in this context, as it directly affects efficiency, and this problem is known as the view path planning problem. Current methods often use sampling-based path planning techniques, evaluating potential views along the path to enhance reconstruction performance. However, these methods are computationally expensive as they require evaluating several candidate views on the path. To this end, we propose a computationally efficient solution that relies on calculating a focus point in the most informative (unknown) region and having the robot maintain this point in the camera field of view along the path. We incorporated this strategy into the whole-body control of a mobile manipulator employing a visibility constraint without the need for an additional path planner. We conducted comprehensive and realistic simulations using a large dataset of 114 diverse objects of varying sizes from 57 categories to compare our method with a sampling-based planning strategy using Bayesian data analysis. Furthermore, we performed real-world experiments with an 8-DoF mobile manipulator to demonstrate the proposed method's performance in practice. Our results suggest that there is no significant difference in object coverage and entropy. In contrast, our method is approximately nine times faster than the baseline sampling-based method in terms of the average time the robot spends between views. 

**Abstract (ZH)**: 对象重建和检验任务在各种机器人应用中起着至关重要的作用。在这种情况下，识别出能够揭示对象最多未知区域的路径变得至关重要，这直接影响效率，这一问题被称为视图路径规划问题。当前的方法通常使用基于采样的路径规划技术，评估路径上的潜在视图以提高重建性能。然而，这些方法计算成本高昂，因为它们需要在路径上评估多个候选视图。为此，我们提出了一种计算效率更高的解决方案，该方案依赖于在最有信息量（未知）区域计算焦点点，并让机器人在路径上保持该点在相机视野内。我们将此策略整合到使用可见性约束的移动操作器的全身控制中，无需额外的路径规划器。我们使用包含114个不同类别和不同尺寸的大规模对象数据集（57个类别）进行了全面和现实的模拟，通过贝叶斯数据分析将我们的方法与基于采样的规划策略进行了比较。此外，我们在一个8自由度的移动操作器上进行了实地实验，以展示所提出方法的实际性能。结果显示，在对象覆盖范围和熵方面没有显著差异，而在平均视图之间机器人所花费的时间上，我们的方法比基线的基于采样的方法快约九倍。 

---
# Keypoint-based Diffusion for Robotic Motion Planning on the NICOL Robot 

**Title (ZH)**: 基于关键点的扩散模型在NICOL机器人运动规划中的应用 

**Authors**: Lennart Clasmeier, Jan-Gerrit Habekost, Connor Gäde, Philipp Allgeuer, Stefan Wermter  

**Link**: [PDF](https://arxiv.org/pdf/2509.04076)  

**Abstract**: We propose a novel diffusion-based action model for robotic motion planning. Commonly, established numerical planning approaches are used to solve general motion planning problems, but have significant runtime requirements. By leveraging the power of deep learning, we are able to achieve good results in a much smaller runtime by learning from a dataset generated by these planners. While our initial model uses point cloud embeddings in the input to predict keypoint-based joint sequences in its output, we observed in our ablation study that it remained challenging to condition the network on the point cloud embeddings. We identified some biases in our dataset and refined it, which improved the model's performance. Our model, even without the use of the point cloud encodings, outperforms numerical models by an order of magnitude regarding the runtime, while reaching a success rate of up to 90% of collision free solutions on the test set. 

**Abstract (ZH)**: 基于扩散的机器人运动规划动作模型 

---
# Solving Robotics Tasks with Prior Demonstration via Exploration-Efficient Deep Reinforcement Learning 

**Title (ZH)**: 基于探索高效深度强化学习的 Robotics 任务通过先验演示的解决方法 

**Authors**: Chengyandan Shen, Christoffer Sloth  

**Link**: [PDF](https://arxiv.org/pdf/2509.04069)  

**Abstract**: This paper proposes an exploration-efficient Deep Reinforcement Learning with Reference policy (DRLR) framework for learning robotics tasks that incorporates demonstrations. The DRLR framework is developed based on an algorithm called Imitation Bootstrapped Reinforcement Learning (IBRL). We propose to improve IBRL by modifying the action selection module. The proposed action selection module provides a calibrated Q-value, which mitigates the bootstrapping error that otherwise leads to inefficient exploration. Furthermore, to prevent the RL policy from converging to a sub-optimal policy, SAC is used as the RL policy instead of TD3. The effectiveness of our method in mitigating bootstrapping error and preventing overfitting is empirically validated by learning two robotics tasks: bucket loading and open drawer, which require extensive interactions with the environment. Simulation results also demonstrate the robustness of the DRLR framework across tasks with both low and high state-action dimensions, and varying demonstration qualities. To evaluate the developed framework on a real-world industrial robotics task, the bucket loading task is deployed on a real wheel loader. The sim2real results validate the successful deployment of the DRLR framework. 

**Abstract (ZH)**: 基于参考策略的探索高效的深度强化学习框架（DRLR）：结合演示的学习机器人任务 

---
# Balancing Signal and Variance: Adaptive Offline RL Post-Training for VLA Flow Models 

**Title (ZH)**: 信号与方差的平衡：适用于VLA流模型的自适应离线RL后训练 

**Authors**: Hongyin Zhang, Shiyuan Zhang, Junxi Jin, Qixin Zeng, Yifan Qiao, Hongchao Lu, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04063)  

**Abstract**: Vision-Language-Action (VLA) models based on flow matching have shown excellent performance in general-purpose robotic manipulation tasks. However, the action accuracy of these models on complex downstream tasks is unsatisfactory. One important reason is that these models rely solely on the post-training paradigm of imitation learning, which makes it difficult to have a deeper understanding of the distribution properties of data quality, which is exactly what Reinforcement Learning (RL) excels at. In this paper, we theoretically propose an offline RL post-training objective for VLA flow models and induce an efficient and feasible offline RL fine-tuning algorithm -- Adaptive Reinforced Flow Matching (ARFM). By introducing an adaptively adjusted scaling factor in the VLA flow model loss, we construct a principled bias-variance trade-off objective function to optimally control the impact of RL signal on flow loss. ARFM adaptively balances RL advantage preservation and flow loss gradient variance control, resulting in a more stable and efficient fine-tuning process. Extensive simulation and real-world experimental results show that ARFM exhibits excellent generalization, robustness, few-shot learning, and continuous learning performance. 

**Abstract (ZH)**: 基于流匹配的Vision-Language-Action (VLA) 模型在通用机器人操作任务中表现出色，但在复杂下游任务中的动作精度不令人满意。一个重要的原因是这些模型仅依赖于模仿学习的后训练范式，这使得它们难以深入理解数据质量分布特性，而这正是强化学习（RL）的优势所在。本文从理论上为VLA流模型提出了一个离线RL后训练目标，并设计了一个高效可行的离线RL微调算法——自适应强化流匹配（ARFM）。通过在VLA流模型损失中引入自适应调整的缩放因子，我们构建了一个原理上的偏置-方差权衡目标函数，以最优控制RL信号对流损失的影响。ARFM自适应平衡了RL优势保留和流损失梯度方差控制，从而实现更稳定和高效的微调过程。广泛的仿真和真实世界实验结果表明，ARFM表现出出色的泛化能力、鲁棒性、少样本学习能力和持续学习性能。 

---
# Integrated Wheel Sensor Communication using ESP32 -- A Contribution towards a Digital Twin of the Road System 

**Title (ZH)**: 基于ESP32的集成轮传感器通信研究——构建道路系统的数字孪生体的贡献 

**Authors**: Ventseslav Yordanov, Simon Schäfer, Alexander Mann, Stefan Kowalewski, Bassam Alrifaee, Lutz Eckstein  

**Link**: [PDF](https://arxiv.org/pdf/2509.04061)  

**Abstract**: While current onboard state estimation methods are adequate for most driving and safety-related applications, they do not provide insights into the interaction between tires and road surfaces. This paper explores a novel communication concept for efficiently transmitting integrated wheel sensor data from an ESP32 microcontroller. Our proposed approach utilizes a publish-subscribe system, surpassing comparable solutions in the literature regarding data transmission volume. We tested this approach on a drum tire test rig with our prototype sensors system utilizing a diverse selection of sample frequencies between 1 Hz and 32 000 Hz to demonstrate the efficacy of our communication concept. The implemented prototype sensor showcases minimal data loss, approximately 0.1 % of the sampled data, validating the reliability of our developed communication system. This work contributes to advancing real-time data acquisition, providing insights into optimizing integrated wheel sensor communication. 

**Abstract (ZH)**: 当前车载状态估计方法虽适用于大多数驾驶和安全相关应用，但未揭示轮胎与路面交互的信息。本文探索了一种新的通信概念，用于高效传输ESP32微控制器集成车轮传感器数据。我们提出的方法利用发布-订阅系统，数据传输量超过文献中 comparable 的解决方案。我们利用不同采样频率（1 Hz 至 32000 Hz）在鼓式轮胎测试台上测试了该方法，以证明我们提出的通信概念的有效性。实现的原型传感器数据显示了极小的数据丢失，约为 0.1% 的采样数据，验证了我们开发的通信系统的可靠性。该工作推动了实时数据采集的发展，提供了优化集成车轮传感器通信的见解。 

---
# FPC-VLA: A Vision-Language-Action Framework with a Supervisor for Failure Prediction and Correction 

**Title (ZH)**: FPC-VLA：带监督的视觉-语言-动作框架，用于故障预测与纠正 

**Authors**: Yifan Yang, Zhixiang Duan, Tianshi Xie, Fuyu Cao, Pinxi Shen, Peili Song, Piaopiao Jin, Guokang Sun, Shaoqing Xu, Yangwei You, Jingtai Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04018)  

**Abstract**: Robotic manipulation is a fundamental component of automation. However, traditional perception-planning pipelines often fall short in open-ended tasks due to limited flexibility, while the architecture of a single end-to-end Vision-Language-Action (VLA) offers promising capabilities but lacks crucial mechanisms for anticipating and recovering from failure. To address these challenges, we propose FPC-VLA, a dual-model framework that integrates VLA with a supervisor for failure prediction and correction. The supervisor evaluates action viability through vision-language queries and generates corrective strategies when risks arise, trained efficiently without manual labeling. A similarity-guided fusion module further refines actions by leveraging past predictions. Evaluation results on multiple simulation platforms (SIMPLER and LIBERO) and robot embodiments (WidowX, Google Robot, Franka) show that FPC-VLA outperforms state-of-the-art models in both zero-shot and fine-tuned settings. By activating the supervisor only at keyframes, our approach significantly increases task success rates with minimal impact on execution time. Successful real-world deployments on diverse, long-horizon tasks confirm FPC-VLA's strong generalization and practical utility for building more reliable autonomous systems. 

**Abstract (ZH)**: 机器人 manipulation 是自动化的基础组成部分。然而，传统的感知-规划管道在开放任务中常常由于缺乏灵活性而表现不佳，而单一的端到端视觉-语言-行动 (VLA) 架构虽然提供了有前景的能力，但缺乏预见和从失败中恢复的关键机制。为了解决这些挑战，我们提出了一种名为 FPC-VLA 的双模型框架，该框架将 VLA 与故障预测和纠正的监督机制相结合。监督机制通过视觉-语言查询评估行动的有效性，并在风险出现时生成纠正策略，通过高效训练且无需手动标注。一个基于相似性引导的融合模块进一步通过利用过往预测来细化行动。在多个仿真平台 (SIMPLER 和 LIBERO) 和机器人硬件 (WidowX、Google Robot、Franka) 上的评估结果显示，FPC-VLA 在零样本和微调设置中均优于现有最先进的模型。只有在关键帧上激活监督机制，我们的方法显著提高了任务成功率，且 minimal 影响执行时间。在多样化的长期任务上的成功实际部署证实了 FPC-VLA 强大的泛化能力和实际应用价值，对于构建更可靠的自主系统具有重要意义。 

---
# Odometry Calibration and Pose Estimation of a 4WIS4WID Mobile Wall Climbing Robot 

**Title (ZH)**: 4WIS4WID移动攀墙机器人里程计标定与位姿估计 

**Authors**: Branimir Ćaran, Vladimir Milić, Marko Švaco, Bojan Jerbić  

**Link**: [PDF](https://arxiv.org/pdf/2509.04016)  

**Abstract**: This paper presents the design of a pose estimator for a four wheel independent steer four wheel independent drive (4WIS4WID) wall climbing mobile robot, based on the fusion of multimodal measurements, including wheel odometry, visual odometry, and an inertial measurement unit (IMU) data using Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF). The pose estimator is a critical component of wall climbing mobile robots, as their operational environment involves carrying precise measurement equipment and maintenance tools in construction, requiring information about pose on the building at the time of measurement. Due to the complex geometry and material properties of building facades, the use of traditional localization sensors such as laser, ultrasonic, or radar is often infeasible for wall-climbing robots. Moreover, GPS-based localization is generally unreliable in these environments because of signal degradation caused by reinforced concrete and electromagnetic interference. Consequently, robot odometry remains the primary source of velocity and position information, despite being susceptible to drift caused by both systematic and non-systematic errors. The calibrations of the robot's systematic parameters were conducted using nonlinear optimization and Levenberg-Marquardt methods as Newton-Gauss and gradient-based model fitting methods, while Genetic algorithm and Particle swarm were used as stochastic-based methods for kinematic parameter calibration. Performance and results of the calibration methods and pose estimators were validated in detail with experiments on the experimental mobile wall climbing robot. 

**Abstract (ZH)**: 基于多模态测量融合的四轮独立转向和驱动壁 climbing 移动机器人姿态估计算法设计 

---
# Reactive In-Air Clothing Manipulation with Confidence-Aware Dense Correspondence and Visuotactile Affordance 

**Title (ZH)**: 基于置信感知密集对应与视触知能的空中服装 Manipulation 

**Authors**: Neha Sunil, Megha Tippur, Arnau Saumell, Edward Adelson, Alberto Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2509.03889)  

**Abstract**: Manipulating clothing is challenging due to complex configurations, variable material dynamics, and frequent self-occlusion. Prior systems often flatten garments or assume visibility of key features. We present a dual-arm visuotactile framework that combines confidence-aware dense visual correspondence and tactile-supervised grasp affordance to operate directly on crumpled and suspended garments. The correspondence model is trained on a custom, high-fidelity simulated dataset using a distributional loss that captures cloth symmetries and generates correspondence confidence estimates. These estimates guide a reactive state machine that adapts folding strategies based on perceptual uncertainty. In parallel, a visuotactile grasp affordance network, self-supervised using high-resolution tactile feedback, determines which regions are physically graspable. The same tactile classifier is used during execution for real-time grasp validation. By deferring action in low-confidence states, the system handles highly occluded table-top and in-air configurations. We demonstrate our task-agnostic grasp selection module in folding and hanging tasks. Moreover, our dense descriptors provide a reusable intermediate representation for other planning modalities, such as extracting grasp targets from human video demonstrations, paving the way for more generalizable and scalable garment manipulation. 

**Abstract (ZH)**: 基于视觉触觉的双臂操控框架：操作褶皱和悬挂衣物的新方法 

---
# Learning Multi-Stage Pick-and-Place with a Legged Mobile Manipulator 

**Title (ZH)**: 基于腿式移动 manipulator 的多阶段抓取放置学习 

**Authors**: Haichao Zhang, Haonan Yu, Le Zhao, Andrew Choi, Qinxun Bai, Yiqing Yang, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.03859)  

**Abstract**: Quadruped-based mobile manipulation presents significant challenges in robotics due to the diversity of required skills, the extended task horizon, and partial observability. After presenting a multi-stage pick-and-place task as a succinct yet sufficiently rich setup that captures key desiderata for quadruped-based mobile manipulation, we propose an approach that can train a visuo-motor policy entirely in simulation, and achieve nearly 80\% success in the real world. The policy efficiently performs search, approach, grasp, transport, and drop into actions, with emerged behaviors such as re-grasping and task chaining. We conduct an extensive set of real-world experiments with ablation studies highlighting key techniques for efficient training and effective sim-to-real transfer. Additional experiments demonstrate deployment across a variety of indoor and outdoor environments. Demo videos and additional resources are available on the project page: this https URL. 

**Abstract (ZH)**: 基于四足机器人移动操作的抓取与放置任务展示了显著的挑战，由于所需技能的多样性、延展的任务时间范围以及部分可观测性。在提出了一种简洁但足够丰富的多阶段抓取与放置任务设定以捕捉四足机器人移动操作的关键需求后，我们提出了一种完全在仿真环境中训练视觉-运动策略的方法，并在现实世界中实现了近80%的成功率。该策略高效地执行搜索、接近、抓取、运输及释放动作，展现出重抓取和任务链等行为。我们进行了广泛的现实世界实验，并通过消融研究强调了高效训练和有效仿真到现实世界迁移的关键技术。此外，实验还展示了在多种室内外环境中的部署能力。更多实验视频和资源可在项目页面获取：this https URL。 

---
# INGRID: Intelligent Generative Robotic Design Using Large Language Models 

**Title (ZH)**: INGRID: 智能生成机器人设计using大规模语言模型 

**Authors**: Guanglu Jia, Ceng Zhang, Gregory S. Chirikjian  

**Link**: [PDF](https://arxiv.org/pdf/2509.03842)  

**Abstract**: The integration of large language models (LLMs) into robotic systems has accelerated progress in embodied artificial intelligence, yet current approaches remain constrained by existing robotic architectures, particularly serial mechanisms. This hardware dependency fundamentally limits the scope of robotic intelligence. Here, we present INGRID (Intelligent Generative Robotic Design), a framework that enables the automated design of parallel robotic mechanisms through deep integration with reciprocal screw theory and kinematic synthesis methods. We decompose the design challenge into four progressive tasks: constraint analysis, kinematic joint generation, chain construction, and complete mechanism design. INGRID demonstrates the ability to generate novel parallel mechanisms with both fixed and variable mobility, discovering kinematic configurations not previously documented in the literature. We validate our approach through three case studies demonstrating how INGRID assists users in designing task-specific parallel robots based on desired mobility requirements. By bridging the gap between mechanism theory and machine learning, INGRID enables researchers without specialized robotics training to create custom parallel mechanisms, thereby decoupling advances in robotic intelligence from hardware constraints. This work establishes a foundation for mechanism intelligence, where AI systems actively design robotic hardware, potentially transforming the development of embodied AI systems. 

**Abstract (ZH)**: 基于递相定律和运动合成方法的深度集成的平行机构智能设计框架：INGRID 

---
# Real-Time Buoyancy Estimation for AUV Simulations Using Convex Hull-Based Submerged Volume Calculation 

**Title (ZH)**: 基于凸包分析的潜体体积计算的自主水下机器人实时浮力估计 

**Authors**: Ad-Deen Mahbub, Md Ragib Shaharear  

**Link**: [PDF](https://arxiv.org/pdf/2509.03804)  

**Abstract**: Accurate real-time buoyancy modeling is essential for high-fidelity Autonomous Underwater Vehicle (AUV) simulations, yet NVIDIA Isaac Sim lacks a native buoyancy system, requiring external solutions for precise underwater physics. This paper presents a novel convex hull-based approach to dynamically compute the submerged volume of an AUV in real time. By extracting mesh geometry from the simulation environment and calculating the hull portion intersecting the water level along the z-axis, our method enhances accuracy over traditional geometric approximations. A cross-sectional area extension reduces computational overhead, enabling efficient buoyant force updates that adapt to orientation, depth, and sinusoidal wave fluctuations (+-0.3 m). Tested on a custom AUV design for SAUVC 2025, this approach delivers real-time performance and scalability, improving simulation fidelity for underwater robotics research without precomputed hydrodynamic models. 

**Abstract (ZH)**: 精确的实时浮力建模对于高保真自主水下车辆(AUV)模拟至关重要，然而NVIDIA Isaac Sim缺乏原生浮力系统，需要外部解决方案来实现精确的水下物理仿真。本文提出了一种基于凸包的新型方法，以实时动态计算AUV在水下的体积。通过从仿真环境提取网格几何并计算沿z轴与水位相交的凸包部分，该方法提高了准确性，超过了传统的几何近似方法。通过扩展横截面积来减少计算开销，实现高效的浮力力更新，适应不同姿态、深度和竖直波浪波动（±0.3米）。在2025年SAUVC定制AUV设计上测试，该方法提供了实时性能和扩展性，提升了水下机器人研究的仿真保真度，无需预先计算水动力模型。 

---
# Low-Cost Open-Source Ambidextrous Robotic Hand with 23 Direct-Drive servos for American Sign Language Alphabet 

**Title (ZH)**: 低成本开源双足机器人手，配备23个直接驱动 servos 以实现美国手语字母表识别 

**Authors**: Kelvin Daniel Gonzalez Amador  

**Link**: [PDF](https://arxiv.org/pdf/2509.03690)  

**Abstract**: Accessible communication through sign language is vital for deaf communities, 1 yet robotic solutions are often costly and limited. This study presents VulcanV3, a low- 2 cost, open-source, 3D-printed ambidextrous robotic hand capable of reproducing the full 3 American Sign Language (ASL) alphabet (52 signs for right- and left-hand configurations). 4 The system employs 23 direct-drive servo actuators for precise finger and wrist movements, 5 controlled by an Arduino Mega with dual PCA9685 modules. Unlike most humanoid upper- 6 limb systems, which rarely employ direct-drive actuation, VulcanV3 achieves complete ASL 7 coverage with a reversible design. All CAD files and code are released under permissive 8 open-source licenses to enable replication. Empirical tests confirmed accurate reproduction 9 of all 52 ASL handshapes, while a participant study (n = 33) achieved 96.97% recognition 10 accuracy, improving to 98.78% after video demonstration. VulcanV3 advances assistive 11 robotics by combining affordability, full ASL coverage, and ambidexterity in an openly 12 shared platform, contributing to accessible communication technologies and inclusive 13 innovation. 

**Abstract (ZH)**: 通过手势语言实现的无障碍沟通对聋人社区至关重要，然而现有的机器人解决方案往往成本较高且功能有限。本研究介绍了VulcanV3，一种低成本、开源、3D打印的左右手兼用机器人手，能够再现完整的美国手语字母表（52个手势，包括右手和左手配置）。该系统采用23个直接驱动伺服驱动器，由Arduino Mega控制并配备双PCA9685模块进行精确的手指和手腕运动控制。与大多数类人上肢系统不同，VulcanV3通过可逆设计实现了完整的美国手语覆盖。所有CAD文件和代码均采用宽松的开源许可发布，以供复制。实证测试证实了对所有52个ASL手势的准确再现，而在参与者研究中（n = 33），识别准确率为96.97%，示范视频后提高至98.78%。VulcanV3通过结合经济性、全面覆盖美国手语以及左右手兼用来推动无障碍机器人技术的发展，促进了无障碍沟通技术和包容性创新。 

---
# Efficient Virtuoso: A Latent Diffusion Transformer Model for Goal-Conditioned Trajectory Planning 

**Title (ZH)**: 高效维鲁佐士：一个用于目标条件轨迹规划的潜扩散变换器模型 

**Authors**: Antonio Guillen-Perez  

**Link**: [PDF](https://arxiv.org/pdf/2509.03658)  

**Abstract**: The ability to generate a diverse and plausible distribution of future trajectories is a critical capability for autonomous vehicle planning systems. While recent generative models have shown promise, achieving high fidelity, computational efficiency, and precise control remains a significant challenge. In this paper, we present the \textbf{Efficient Virtuoso}, a conditional latent diffusion model for goal-conditioned trajectory planning. Our approach introduces a novel two-stage normalization pipeline that first scales trajectories to preserve their geometric aspect ratio and then normalizes the resulting PCA latent space to ensure a stable training target. The denoising process is performed efficiently in this low-dimensional latent space by a simple MLP denoiser, which is conditioned on a rich scene context fused by a powerful Transformer-based StateEncoder. We demonstrate that our method achieves state-of-the-art performance on the Waymo Open Motion Dataset, reaching a \textbf{minADE of 0.25}. Furthermore, through a rigorous ablation study on goal representation, we provide a key insight: while a single endpoint goal can resolve strategic ambiguity, a richer, multi-step sparse route is essential for enabling the precise, high-fidelity tactical execution that mirrors nuanced human driving behavior. 

**Abstract (ZH)**: 生成多样且合理的未来轨迹分布能力是自主车辆规划系统的一项关键能力。尽管近期生成模型展示了潜力，但实现高度保真度、计算效率和精确控制仍面临重大挑战。本文提出了一种条件潜扩散模型\textbf{Efficient Virtuoso}，用于目标导向的轨迹规划。我们的方法引入了一种新颖的两阶段规范化流水线，首先按比例缩放轨迹以保持其几何方面比例，然后对得到的PCA潜空间进行规范化以确保稳定的训练目标。去噪过程通过一个简单的MLP去噪器在低维潜空间中高效地进行，该去噪器由强大的Transformer基于StateEncoder融合丰富的场景语境进行条件化。我们证明，我们的方法在Waymo Open Motion Dataset上达到了最先进的性能，最小平均偏离误差(minADE)为0.25。此外，通过严格的关于目标表示的消融研究，我们提供了一个关键洞察：虽然单个终点目标可以解决策略上的模糊性，但更丰富、多步骤的稀疏路径对于实现可精确且高度保真的战术执行至关重要，这种执行行为类似于微妙的人类驾驶行为。 

---
# Cooperative Grasping for Collective Object Transport in Constrained Environments 

**Title (ZH)**: 约束环境下协作抓取与集体对象运输 

**Authors**: David Alvear, George Turkiyyah, Shinkyu Park  

**Link**: [PDF](https://arxiv.org/pdf/2509.03638)  

**Abstract**: We propose a novel framework for decision-making in cooperative grasping for two-robot object transport in constrained environments. The core of the framework is a Conditional Embedding (CE) model consisting of two neural networks that map grasp configuration information into an embedding space. The resulting embedding vectors are then used to identify feasible grasp configurations that allow two robots to collaboratively transport an object. To ensure generalizability across diverse environments and object geometries, the neural networks are trained on a dataset comprising a range of environment maps and object shapes. We employ a supervised learning approach with negative sampling to ensure that the learned embeddings effectively distinguish between feasible and infeasible grasp configurations. Evaluation results across a wide range of environments and objects in simulations demonstrate the model's ability to reliably identify feasible grasp configurations. We further validate the framework through experiments on a physical robotic platform, confirming its practical applicability. 

**Abstract (ZH)**: 一种用于受限环境下双机器人物体搬运协同抓取决策的新颖框架 

---
# Self-Organizing Aerial Swarm Robotics for Resilient Load Transportation : A Table-Mechanics-Inspired Approach 

**Title (ZH)**: 自组织空中 swarm 机器人用于稳健负载运输：一种基于桌 faced 力学的方法 

**Authors**: Quan Quan, Jiwen Xu, Runxiao Liu, Yi Ding, Jiaxing Che, Kai-Yuan Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.03563)  

**Abstract**: In comparison with existing approaches, which struggle with scalability, communication dependency, and robustness against dynamic failures, cooperative aerial transportation via robot swarms holds transformative potential for logistics and disaster response. Here, we present a physics-inspired cooperative transportation approach for flying robot swarms that imitates the dissipative mechanics of table-leg load distribution. By developing a decentralized dissipative force model, our approach enables autonomous formation stabilization and adaptive load allocation without the requirement of explicit communication. Based on local neighbor robots and the suspended payload, each robot dynamically adjusts its position. This is similar to energy-dissipating table leg reactions. The stability of the resultant control system is rigorously proved. Simulations demonstrate that the tracking errors of the proposed approach are 20%, 68%, 55.5%, and 21.9% of existing approaches under the cases of capability variation, cable uncertainty, limited vision, and payload variation, respectively. In real-world experiments with six flying robots, the cooperative aerial transportation system achieved a 94% success rate under single-robot failure, disconnection events, 25% payload variation, and 40% cable length uncertainty, demonstrating strong robustness under outdoor winds up to Beaufort scale 4. Overall, this physics-inspired approach bridges swarm intelligence and mechanical stability principles, offering a scalable framework for heterogeneous aerial systems to collectively handle complex transportation tasks in communication-constrained environments. 

**Abstract (ZH)**: 基于物理原理的飞行机器人 swarm 合作运输方法：桥接群智与机械稳定性原理以应对通信受限环境中的复杂运输任务 

---
# SAFE--MA--RRT: Multi-Agent Motion Planning with Data-Driven Safety Certificates 

**Title (ZH)**: SAFE--MA--RRT: 基于数据驱动安全证书的多agents运动规划 

**Authors**: Babak Esmaeili, Hamidreza Modares  

**Link**: [PDF](https://arxiv.org/pdf/2509.04413)  

**Abstract**: This paper proposes a fully data-driven motion-planning framework for homogeneous linear multi-agent systems that operate in shared, obstacle-filled workspaces without access to explicit system models. Each agent independently learns its closed-loop behavior from experimental data by solving convex semidefinite programs that generate locally invariant ellipsoids and corresponding state-feedback gains. These ellipsoids, centered along grid-based waypoints, certify the dynamic feasibility of short-range transitions and define safe regions of operation. A sampling-based planner constructs a tree of such waypoints, where transitions are allowed only when adjacent ellipsoids overlap, ensuring invariant-to-invariant transitions and continuous safety. All agents expand their trees simultaneously and are coordinated through a space-time reservation table that guarantees inter-agent safety by preventing simultaneous occupancy and head-on collisions. Each successful edge in the tree is equipped with its own local controller, enabling execution without re-solving optimization problems at runtime. The resulting trajectories are not only dynamically feasible but also provably safe with respect to both environmental constraints and inter-agent collisions. Simulation results demonstrate the effectiveness of the approach in synthesizing synchronized, safe trajectories for multiple agents under shared dynamics and constraints, using only data and convex optimization tools. 

**Abstract (ZH)**: 本文提出了一种完全基于数据的运动规划框架，适用于在共享且充满障碍的工作空间中运作的同构线性多智能体系统，且无需显式系统模型。每个智能体独立通过求解凸半定规划问题学习其闭环行为，并生成局部不变椭球和相应的状态反馈增益。这些椭球在基于网格的方式点中心定位，验证了短距离过渡的动态可行性，并定义了操作的安全区域。基于采样的规划器构建了一棵树状的路径点集合，仅允许当相邻椭球重叠时进行过渡，确保从不变集到不变集的过渡和连续安全性。所有智能体同时扩展其树状结构，并通过时空预留表进行协调，确保智能体间的互操作安全，避免同时占据和迎面碰撞。树状结构中的每个成功边都配备了本地控制器，使得执行时无需在运行时重新求解优化问题。生成的轨迹不仅具有动态可行性，而且在环境约束和智能体间碰撞方面具有可证明的安全性。仿真结果表明，该方法能够利用仅有的数据和凸优化工具，在共享动力学和约束条件下综合同步且安全的多智能体轨迹。 

---
# Leveraging Equivariances and Symmetries in the Control Barrier Function Synthesis 

**Title (ZH)**: 利用控制障碍函数合成中的不变性和对称性 

**Authors**: Adrian Wiltz, Dimos V. Dimarogonas  

**Link**: [PDF](https://arxiv.org/pdf/2509.04399)  

**Abstract**: The synthesis of Control Barrier Functions (CBFs) often involves demanding computations or a meticulous construction. However, structural properties of the system dynamics and constraints have the potential to mitigate these challenges. In this paper, we explore how equivariances in the dynamics, loosely speaking a form of symmetry, can be leveraged in the CBF synthesis. Although CBFs are generally not inherently symmetric, we show how equivariances in the dynamics and symmetries in the constraints induce symmetries in CBFs derived through reachability analysis. This insight allows us to infer their CBF values across the entire domain from their values on a subset, leading to significant computational savings. Interestingly, equivariances can be even leveraged to the CBF synthesis for non-symmetric constraints. Specifically, we show how a partially known CBF can be leveraged together with equivariances to construct a CBF for various new constraints. Throughout the paper, we provide examples illustrating the theoretical findings. Furthermore, a numerical study investigates the computational gains from invoking equivariances into the CBF synthesis. 

**Abstract (ZH)**: 基于动力学对称性的控制障碍函数合成研究 

---
# Privacy Perceptions in Robot-Assisted Well-Being Coaching: Examining the Roles of Information Transparency, User Control, and Proactivity 

**Title (ZH)**: 机器人辅助福祉教练中的隐私感知：探讨信息透明度、用户控制和主动性的作用 

**Authors**: Atikkhan Faridkhan Nilgar, Manuel Dietrich, Kristof Van Laerhoven  

**Link**: [PDF](https://arxiv.org/pdf/2509.04358)  

**Abstract**: Social robots are increasingly recognized as valuable supporters in the field of well-being coaching. They can function as independent coaches or provide support alongside human coaches, and healthcare professionals. In coaching interactions, these robots often handle sensitive information shared by users, making privacy a relevant issue. Despite this, little is known about the factors that shape users' privacy perceptions. This research aims to examine three key factors systematically: (1) the transparency about information usage, (2) the level of specific user control over how the robot uses their information, and (3) the robot's behavioral approach - whether it acts proactively or only responds on demand. Our results from an online study (N = 200) show that even when users grant the robot general access to personal data, they additionally expect the ability to explicitly control how that information is interpreted and shared during sessions. Experimental conditions that provided such control received significantly higher ratings for perceived privacy appropriateness and trust. Compared to user control, the effects of transparency and proactivity on privacy appropriateness perception were low, and we found no significant impact. The results suggest that merely informing users or proactive sharing is insufficient without accompanying user control. These insights underscore the need for further research on mechanisms that allow users to manage robots' information processing and sharing, especially when social robots take on more proactive roles alongside humans. 

**Abstract (ZH)**: 社会机器人在福祉教练领域的应用日益受到认可，它们可以作为独立教练存在，也可以作为人类教练和医疗专业人员的支持者发挥作用。在教练互动中，这些机器人经常处理用户分享的敏感信息，因此隐私问题成为一个相关议题。尽管如此，对塑造用户隐私感知因素的研究还相对不足。本研究旨在系统地考察三个关键因素：（1）信息使用透明度，（2）用户对其信息如何被机器人使用的具体控制程度，以及（3）机器人的行为方式——它是否主动或仅在需求时响应。在线研究（N=200）的结果显示，即使用户授予机器人访问个人数据的通用权限，他们也期望能够在会话中明确控制这些信息的解释和分享方式。提供这种控制的实验条件在感知隐私适当性和信任度方面获得了显著更高的评分。相比之下，透明度和主动性的效应对感知隐私适当性感知影响较低，我们未发现显著影响。结果表明，仅告知用户或主动分享不足以解决问题。这些见解强调了需要进一步研究允许用户管理机器人信息处理和分享的机制，尤其是在社会机器人承担更多主动角色时。 

---
# SRWToolkit: An Open Source Wizard of Oz Toolkit to Create Social Robotic Avatars 

**Title (ZH)**: SRWToolkit: 一个开源的社会机器人avatar创建Wizard of Oz工具-kit 

**Authors**: Atikkhan Faridkhan Nilgar, Kristof Van Laerhoven, Ayub Kinoti  

**Link**: [PDF](https://arxiv.org/pdf/2509.04356)  

**Abstract**: We present SRWToolkit, an open-source Wizard of Oz toolkit designed to facilitate the rapid prototyping of social robotic avatars powered by local large language models (LLMs). Our web-based toolkit enables multimodal interaction through text input, button-activated speech, and wake-word command. The toolkit offers real-time configuration of avatar appearance, behavior, language, and voice via an intuitive control panel. In contrast to prior works that rely on cloud-based LLM services, SRWToolkit emphasizes modularity and ensures on-device functionality through local LLM inference. In our small-scale user study ($n=11$), participants created and interacted with diverse robotic roles (hospital receptionist, mathematics teacher, and driving assistant), which demonstrated positive outcomes in the toolkit's usability, trust, and user experience. The toolkit enables rapid and efficient development of robot characters customized to researchers' needs, supporting scalable research in human-robot interaction. 

**Abstract (ZH)**: SRWToolkit：一个用于基于本地大型语言模型的社会机器人avatar快速原型设计的开源Wizard of Oz工具包 

---
# Compatibility of Multiple Control Barrier Functions for Constrained Nonlinear Systems 

**Title (ZH)**: 受限非线性系统的多个控制障碍函数兼容性研究 

**Authors**: Max H. Cohen, Eugene Lavretsky, Aaron D. Ames  

**Link**: [PDF](https://arxiv.org/pdf/2509.04220)  

**Abstract**: Control barrier functions (CBFs) are a powerful tool for the constrained control of nonlinear systems; however, the majority of results in the literature focus on systems subject to a single CBF constraint, making it challenging to synthesize provably safe controllers that handle multiple state constraints. This paper presents a framework for constrained control of nonlinear systems subject to box constraints on the systems' vector-valued outputs using multiple CBFs. Our results illustrate that when the output has a vector relative degree, the CBF constraints encoding these box constraints are compatible, and the resulting optimization-based controller is locally Lipschitz continuous and admits a closed-form expression. Additional results are presented to characterize the degradation of nominal tracking objectives in the presence of safety constraints. Simulations of a planar quadrotor are presented to demonstrate the efficacy of the proposed framework. 

**Abstract (ZH)**: 基于多个控制障碍函数的非线性系统受限输出区间约束控制框架 

---
# YOLO Ensemble for UAV-based Multispectral Defect Detection in Wind Turbine Components 

**Title (ZH)**: 基于无人机的风力涡轮机部件多光谱缺陷检测的YOLO集成方法 

**Authors**: Serhii Svystun, Pavlo Radiuk, Oleksandr Melnychenko, Oleg Savenko, Anatoliy Sachenko  

**Link**: [PDF](https://arxiv.org/pdf/2509.04156)  

**Abstract**: Unmanned aerial vehicles (UAVs) equipped with advanced sensors have opened up new opportunities for monitoring wind power plants, including blades, towers, and other critical components. However, reliable defect detection requires high-resolution data and efficient methods to process multispectral imagery. In this research, we aim to enhance defect detection accuracy through the development of an ensemble of YOLO-based deep learning models that integrate both visible and thermal channels. We propose an ensemble approach that integrates a general-purpose YOLOv8 model with a specialized thermal model, using a sophisticated bounding box fusion algorithm to combine their predictions. Our experiments show this approach achieves a mean Average Precision (mAP@.5) of 0.93 and an F1-score of 0.90, outperforming a standalone YOLOv8 model, which scored an mAP@.5 of 0.91. These findings demonstrate that combining multiple YOLO architectures with fused multispectral data provides a more reliable solution, improving the detection of both visual and thermal defects. 

**Abstract (ZH)**: 装备有先进传感器的无人机为监测风力发电机组叶片、塔架及其他关键组件提供了新机会。然而，可靠的缺陷检测需要高分辨率数据和高效的多光谱图像处理方法。本研究旨在通过开发将可见光和热成像通道结合的YOLO系列深度学习模型集成体来提高缺陷检测精度。我们提出了一种集成通用YOLOv8模型和专用热成像模型的方法，使用复杂的边界框融合算法结合其预测结果。实验结果表明，该方法在mAP@0.5上的平均精度为0.93，F1分数为0.90，优于单独使用YOLOv8模型，后者在mAP@0.5上的得分为0.91。这些发现表明，将多个YOLO架构与融合多光谱数据相结合能提供更可靠的方法，提高对视觉和热成像缺陷的检测。 

---
# Memory Optimization for Convex Hull Support Point Queries 

**Title (ZH)**: 凸包支撑点查询的内存优化 

**Authors**: Michael Greer  

**Link**: [PDF](https://arxiv.org/pdf/2509.03753)  

**Abstract**: This paper evaluates several improvements to the memory layout of convex hulls to improve computation times for support point queries. The support point query is a fundamental part of common collision algorithms, and the work presented achieves a significant speedup depending on the number of vertices of the convex hull. 

**Abstract (ZH)**: 本文评估了几种改进凸包内存布局的方法，以提高支持点查询的计算时间。支持点查询是常见碰撞算法的基本组成部分，所提出的工作根据凸包的顶点数实现了显著的速度提升。 

---
# Avoidance of an unexpected obstacle without reinforcement learning: Why not using advanced control-theoretic tools? 

**Title (ZH)**: 避免意外障碍而不使用强化学习：为什么不用高级控制理论工具？ 

**Authors**: Cédric Join, Michel Fliess  

**Link**: [PDF](https://arxiv.org/pdf/2509.03721)  

**Abstract**: This communication on collision avoidance with unexpected obstacles is motivated by some critical appraisals on reinforcement learning (RL) which "requires ridiculously large numbers of trials to learn any new task" (Yann LeCun). We use the classic Dubins' car in order to replace RL with flatness-based control, combined with the HEOL feedback setting, and the latest model-free predictive control approach. The two approaches lead to convincing computer experiments where the results with the model-based one are only slightly better. They exhibit a satisfactory robustness with respect to randomly generated mismatches/disturbances, which become excellent in the model-free case. Those properties would have been perhaps difficult to obtain with today's popular machine learning techniques in AI. Finally, we should emphasize that our two methods require a low computational burden. 

**Abstract (ZH)**: 基于意外障碍的避碰问题中的碰撞避免通信：基于光滑性控制的HEOL反馈与模型自由预测控制方法 

---
