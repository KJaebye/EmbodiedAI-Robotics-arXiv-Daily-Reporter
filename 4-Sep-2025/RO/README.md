# Can the Waymo Open Motion Dataset Support Realistic Behavioral Modeling? A Validation Study with Naturalistic Trajectories 

**Title (ZH)**: Waymo开放运动数据集能否支持现实行为建模？一种利用自然轨迹验证的研究 

**Authors**: Yanlin Zhang, Sungyong Chung, Nachuan Li, Dana Monzer, Hani S. Mahmassani, Samer H. Hamdar, Alireza Talebpour  

**Link**: [PDF](https://arxiv.org/pdf/2509.03515)  

**Abstract**: The Waymo Open Motion Dataset (WOMD) has become a popular resource for data-driven modeling of autonomous vehicles (AVs) behavior. However, its validity for behavioral analysis remains uncertain due to proprietary post-processing, the absence of error quantification, and the segmentation of trajectories into 20-second clips. This study examines whether WOMD accurately captures the dynamics and interactions observed in real-world AV operations. Leveraging an independently collected naturalistic dataset from Level 4 AV operations in Phoenix, Arizona (PHX), we perform comparative analyses across three representative urban driving scenarios: discharging at signalized intersections, car-following, and lane-changing behaviors. For the discharging analysis, headways are manually extracted from aerial video to ensure negligible measurement error. For the car-following and lane-changing cases, we apply the Simulation-Extrapolation (SIMEX) method to account for empirically estimated error in the PHX data and use Dynamic Time Warping (DTW) distances to quantify behavioral differences. Results across all scenarios consistently show that behavior in PHX falls outside the behavioral envelope of WOMD. Notably, WOMD underrepresents short headways and abrupt decelerations. These findings suggest that behavioral models calibrated solely on WOMD may systematically underestimate the variability, risk, and complexity of naturalistic driving. Caution is therefore warranted when using WOMD for behavior modeling without proper validation against independently collected data. 

**Abstract (ZH)**: Waymo 开放运动数据集 (WOMD) 在自主车辆 (AV) 行为驱动建模中的适用性分析：基于亚利桑那州凤凰城 (Phoenix, AZ) 的 Level 4 AV 实际运营数据的比较研究 

---
# Real-Time Instrument Planning and Perception for Novel Measurements of Dynamic Phenomena 

**Title (ZH)**: 实时仪器规划与感知以进行动态现象的新颖测量 

**Authors**: Itai Zilberstein, Alberto Candela, Steve Chien  

**Link**: [PDF](https://arxiv.org/pdf/2509.03500)  

**Abstract**: Advancements in onboard computing mean remote sensing agents can employ state-of-the-art computer vision and machine learning at the edge. These capabilities can be leveraged to unlock new rare, transient, and pinpoint measurements of dynamic science phenomena. In this paper, we present an automated workflow that synthesizes the detection of these dynamic events in look-ahead satellite imagery with autonomous trajectory planning for a follow-up high-resolution sensor to obtain pinpoint measurements. We apply this workflow to the use case of observing volcanic plumes. We analyze classification approaches including traditional machine learning algorithms and convolutional neural networks. We present several trajectory planning algorithms that track the morphological features of a plume and integrate these algorithms with the classifiers. We show through simulation an order of magnitude increase in the utility return of the high-resolution instrument compared to baselines while maintaining efficient runtimes. 

**Abstract (ZH)**: 车载计算的进步使得遥感代理能够在边缘运用最先进的计算机视觉和机器学习技术。这些能力可以被利用来解锁新的稀有、瞬态和精准的动态科学现象测量。本文提出了一种自动化工作流，该工作流结合了前瞻性卫星图像中动态事件的检测与后续高分辨率传感器的自主轨迹规划，以获取精准测量。我们以观测火山灰柱为例应用这一工作流。我们分析了包括传统机器学习算法和卷积神经网络在内的分类方法。我们提出了几种轨迹规划算法，这些算法追踪灰柱的形态特征，并将这些算法与分类器结合。通过模拟，我们展示了与基准相比，高分辨率仪器的利用率提高了数倍，同时保持了高效的运行时间。 

---
# Cost-Optimized Systems Engineering for IoT-Enabled Robot Nurse in Infectious Pandemic Management 

**Title (ZH)**: 物联网赋能感染性疫情管理中成本优化的护理机器人系统工程 

**Authors**: Md Mhamud Hussen Sifat, Md Maruf, Md Rokunuzzaman  

**Link**: [PDF](https://arxiv.org/pdf/2509.03436)  

**Abstract**: The utilization of robotic technology has gained traction in healthcare facilities due to progress in the field that enables time and cost savings, minimizes waste, and improves patient care. Digital healthcare technologies that leverage automation, such as robotics and artificial intelligence, have the potential to enhance the sustainability and profitability of healthcare systems in the long run. However, the recent COVID-19 pandemic has amplified the need for cyber-physical robots to automate check-ups and medication administration. A robot nurse is controlled by the Internet of Things (IoT) and can serve as an automated medical assistant while also allowing supervisory control based on custom commands. This system helps reduce infection risk and improves outcomes in pandemic settings. This research presents a test case with a nurse robot that can assess a patient's health status and take action accordingly. We also evaluate the system's performance in medication administration, health-status monitoring, and life-cycle considerations. 

**Abstract (ZH)**: 机器人技术在医疗设施中的应用由于能够节省时间与成本、减少浪费并改善患者护理而逐渐受到重视。利用自动化技术的数字 healthcare 技术，如机器人和人工智能，有潜力在长期内提高 healthcare 系统的可持续性和盈利能力。然而，近期的 COVID-19 大流行放大了对用于自动化检查和药物管理的网络物理机器人的需求。一种由物联网（IoT）控制的机器人护士可以作为自动化医疗助手，并允许基于自定义命令的监督控制。该系统有助于减少感染风险并在大流行环境中改善结果。本研究介绍了一个护士机器人测试案例，该机器人能够评估患者健康状况并相应采取行动。我们还评估了该系统在药物管理、健康状况监测和生命周期考虑方面的性能。 

---
# Parallel-Constraint Model Predictive Control: Exploiting Parallel Computation for Improving Safety 

**Title (ZH)**: 并行约束模型预测控制：利用并行计算提高安全性 

**Authors**: Elias Fontanari, Gianni Lunardi, Matteo Saveriano, Andrea Del Prete  

**Link**: [PDF](https://arxiv.org/pdf/2509.03261)  

**Abstract**: Ensuring constraint satisfaction is a key requirement for safety-critical systems, which include most robotic platforms. For example, constraints can be used for modeling joint position/velocity/torque limits and collision avoidance. Constrained systems are often controlled using Model Predictive Control, because of its ability to naturally handle constraints, relying on numerical optimization. However, ensuring constraint satisfaction is challenging for nonlinear systems/constraints. A well-known tool to make controllers safe is the so-called control-invariant set (a.k.a. safe set). In our previous work, we have shown that safety can be improved by letting the safe-set constraint recede along the MPC horizon. In this paper, we push that idea further by exploiting parallel computation to improve safety. We solve several MPC problems at the same time, where each problem instantiates the safe-set constraint at a different time step along the horizon. Finally, the controller can select the best solution according to some user-defined criteria. We validated this idea through extensive simulations with a 3-joint robotic arm, showing that significant improvements can be achieved in terms of safety and performance, even using as little as 4 computational cores. 

**Abstract (ZH)**: 确保约束满足是安全关键系统的关键要求，这些系统包括大多数机器人平台。例如，约束可用于建模关节位置/速度/扭矩限制和避碰。对于非线性系统/约束，确保约束满足具有挑战性。一种使控制器安全的已知工具是所谓的不变控制集（又称安全集）。在我们之前的工作中，我们已经展示了通过让安全集约束沿MPC时间轴退化来提高安全性。在本文中，我们通过利用并行计算进一步推进了这一理念，通过同时求解多个MPC问题，每个问题在时间轴上的不同时间步长实例化安全集约束。最后，控制器可以根据一些用户定义的标准选择最佳解决方案。我们通过广泛的仿真（使用一个3关节机器人臂）验证了这一理念，即使仅使用4个计算内核，也能在安全性和性能方面取得显著改进。 

---
# Vibration Damping in Underactuated Cable-suspended Artwork -- Flying Belt Motion Control 

**Title (ZH)**: 无动作冗余的悬索艺术品减振——飞行带运动控制 

**Authors**: Martin Goubej, Lauria Clarke, Martin Hrabačka, David Tolar  

**Link**: [PDF](https://arxiv.org/pdf/2509.03238)  

**Abstract**: This paper presents a comprehensive refurbishment of the interactive robotic art installation Standards and Double Standards by Rafael Lozano-Hemmer. The installation features an array of belts suspended from the ceiling, each actuated by stepper motors and dynamically oriented by a vision-based tracking system that follows the movements of exhibition visitors. The original system was limited by oscillatory dynamics, resulting in torsional and pendulum-like vibrations that constrained rotational speed and reduced interactive responsiveness. To address these challenges, the refurbishment involved significant upgrades to both hardware and motion control algorithms. A detailed mathematical model of the flying belt system was developed to accurately capture its dynamic behavior, providing a foundation for advanced control design. An input shaping method, formulated as a convex optimization problem, was implemented to effectively suppress vibrations, enabling smoother and faster belt movements. Experimental results demonstrate substantial improvements in system performance and audience interaction. This work exemplifies the integration of robotics, control engineering, and interactive art, offering new solutions to technical challenges in real-time motion control and vibration damping for large-scale kinetic installations. 

**Abstract (ZH)**: 本文对Rafael Lozano-Hemmer的互动机器人艺术装置《标准与双重标准》进行了全面修复。该装置由悬挂于天花板下的传送带阵列组成，每条传送带由步进电机驱动，并通过基于视觉的跟踪系统动态定向，该系统跟踪展览参观者的运动。原始系统受限于振荡动态，导致扭曲和摆动般的振动，限制了旋转速度并降低了交互响应性。为解决这些挑战，修复工作包括硬件和运动控制算法的重大升级。开发了飞带系统的详细数学模型，以准确捕捉其动态行为，为先进的控制设计提供了基础。实施了一种作为凸优化问题形式的输入 shaping 方法，有效地抑制了振动，使传送带的运动更加平滑和快速。实验结果表明系统性能和观众互动有了显著改进。这项工作体现了机器人学、控制工程和互动艺术的整合，提供了实时运动控制和大型动态装置振动抑制的技术解决方案。 

---
# Exploring persuasive Interactions with generative social robots: An experimental framework 

**Title (ZH)**: 探索生成式社会机器人中的说服性互动：一个实验框架 

**Authors**: Stephan Vonschallen, Larissa Julia Corina Finsler, Theresa Schmiedel, Friederike Eyssel  

**Link**: [PDF](https://arxiv.org/pdf/2509.03231)  

**Abstract**: Integrating generative AI such as large language models into social robots has improved their ability to engage in natural, human-like communication. This study presents a method to examine their persuasive capabilities. We designed an experimental framework focused on decision making and tested it in a pilot that varied robot appearance and self-knowledge. Using qualitative analysis, we evaluated interaction quality, persuasion effectiveness, and the robot's communicative strategies. Participants generally experienced the interaction positively, describing the robot as competent, friendly, and supportive, while noting practical limits such as delayed responses and occasional speech-recognition errors. Persuasiveness was highly context dependent and shaped by robot behavior: participants responded well to polite, reasoned suggestions and expressive gestures, but emphasized the need for more personalized, context-aware arguments and clearer social roles. These findings suggest that generative social robots can influence user decisions, but their effectiveness depends on communicative nuance and contextual relevance. We propose refinements to the framework to further study persuasive dynamics between robots and human users. 

**Abstract (ZH)**: 将生成式AI如大型语言模型集成到社交机器人中，提高了它们进行自然、人类般沟通的能力。本研究提出了一种方法来考察它们的说服能力。我们设计了一个以决策制定为中心的实验框架，并在一项试点研究中测试了该框架，该试点研究变换了机器人的外观和自我认知。通过定性分析，我们评估了互动质量、说服效果以及机器人的沟通策略。参与者普遍对互动体验持积极态度，描述机器人表现出色、friendly且支持性强，同时指出了实用性限制，如响应延迟和偶尔的语音识别错误。说服力的高度依赖于上下文和机器人行为：参与者对有礼貌、有条理的建议和表达性手势反应良好，但强调需要更具个性化、上下文相关的论据和更清晰的社会角色。这些发现表明，生成式社交机器人能够影响用户决策，但其有效性取决于沟通的细微差别和上下文相关性。我们提出了框架的改进方案，以便进一步研究机器人与人类用户间的说服动态。 

---
# The Role of Embodiment in Intuitive Whole-Body Teleoperation for Mobile Manipulation 

**Title (ZH)**: 整体身体遥操作中的知觉 embodiment作用研究 

**Authors**: Sophia Bianchi Moyen, Rickmer Krohn, Sophie Lueth, Kay Pompetzki, Jan Peters, Vignesh Prasad, Georgia Chalvatzaki  

**Link**: [PDF](https://arxiv.org/pdf/2509.03222)  

**Abstract**: Intuitive Teleoperation interfaces are essential for mobile manipulation robots to ensure high quality data collection while reducing operator workload. A strong sense of embodiment combined with minimal physical and cognitive demands not only enhances the user experience during large-scale data collection, but also helps maintain data quality over extended periods. This becomes especially crucial for challenging long-horizon mobile manipulation tasks that require whole-body coordination. We compare two distinct robot control paradigms: a coupled embodiment integrating arm manipulation and base navigation functions, and a decoupled embodiment treating these systems as separate control entities. Additionally, we evaluate two visual feedback mechanisms: immersive virtual reality and conventional screen-based visualization of the robot's field of view. These configurations were systematically assessed across a complex, multi-stage task sequence requiring integrated planning and execution. Our results show that the use of VR as a feedback modality increases task completion time, cognitive workload, and perceived effort of the teleoperator. Coupling manipulation and navigation leads to a comparable workload on the user as decoupling the embodiments, while preliminary experiments suggest that data acquired by coupled teleoperation leads to better imitation learning performance. Our holistic view on intuitive teleoperation interfaces provides valuable insight into collecting high-quality, high-dimensional mobile manipulation data at scale with the human operator in mind. Project website:this https URL 

**Abstract (ZH)**: 直观远程操作界面对于移动操控机器人在减少操作员工作负担的同时确保高质量数据采集至关重要。强烈的身体代入感结合最小的物理和认知需求不仅能增强大规模数据采集过程中的用户体验，还能在长时间内维持数据质量。这对于需要全身协调的挑战性长时 Horizon 移动操控任务尤其重要。我们比较了两种不同的机器人控制 paradigm：结合手臂操控和底盘导航功能的耦合身体代入感以及将这些系统视为独立控制实体的解耦身体代入感。此外，我们评估了两种视觉反馈机制：沉浸式虚拟现实和基于屏幕的传统机器人视野可视化。这些配置在一项复杂的多阶段任务序列中进行了系统性评估，该任务序列要求集成规划和执行。研究结果表明，使用 VR 作为反馈模态会增加任务完成时间、认知负担和远程操作员的感知努力。结合操控和导航在用户中产生的工作量与解耦身体代入感相当，初步实验表明，耦合远程操控获取的数据有助于更好的模仿学习性能。我们全面的直观远程操作界面观点提供了有关如何在注重人类操作员的情况下收集高质量、高维度的移动操控数据的宝贵见解。项目网站：this https URL。 

---
# Efficient Active Training for Deep LiDAR Odometry 

**Title (ZH)**: 高效主动训练用于深度激光雷达里程计 

**Authors**: Beibei Zhou, Zhiyuan Zhang, Zhenbo Song, Jianhui Guo, Hui Kong  

**Link**: [PDF](https://arxiv.org/pdf/2509.03211)  

**Abstract**: Robust and efficient deep LiDAR odometry models are crucial for accurate localization and 3D reconstruction, but typically require extensive and diverse training data to adapt to diverse environments, leading to inefficiencies. To tackle this, we introduce an active training framework designed to selectively extract training data from diverse environments, thereby reducing the training load and enhancing model generalization. Our framework is based on two key strategies: Initial Training Set Selection (ITSS) and Active Incremental Selection (AIS). ITSS begins by breaking down motion sequences from general weather into nodes and edges for detailed trajectory analysis, prioritizing diverse sequences to form a rich initial training dataset for training the base model. For complex sequences that are difficult to analyze, especially under challenging snowy weather conditions, AIS uses scene reconstruction and prediction inconsistency to iteratively select training samples, refining the model to handle a wide range of real-world scenarios. Experiments across datasets and weather conditions validate our approach's effectiveness. Notably, our method matches the performance of full-dataset training with just 52\% of the sequence volume, demonstrating the training efficiency and robustness of our active training paradigm. By optimizing the training process, our approach sets the stage for more agile and reliable LiDAR odometry systems, capable of navigating diverse environmental conditions with greater precision. 

**Abstract (ZH)**: 鲁棒且高效的深度LiDAR odomatry模型对于精确定位和3D重建至关重要，但通常需要大量的多样训练数据以适应不同的环境，导致效率低下。为解决这一问题，我们提出了一种主动训练框架，旨在从多样环境中选择性地提取训练数据，从而减少训练负担并增强模型的泛化能力。该框架基于两种关键策略：初始训练集选择（ITSS）和主动增量选择（AIS）。ITSS通过将通常天气下的运动序列分解成节点和边，进行详细的轨迹分析，优先选择多样化的序列以形成丰富的初始训练数据集，用于训练基础模型。对于难以分析的复杂序列，特别是在恶劣雪天条件下，AIS利用场景重建和预测一致性进行迭代选择训练样本，逐步优化模型以处理各种实际场景。实验结果表明，我们的方法在仅使用52%的数据序列体积的情况下，达到了与全数据集训练相当的效果，展示了我们主动训练框架的训练效率和鲁棒性。通过优化训练过程，我们的方法为更灵活可靠的LiDAR odomatry系统奠定了基础，使其能够在更广泛且多变的环境条件下实现更精确的导航。 

---
# Forbal: Force Balanced 2-5 Degree of Freedom Robot Manipulator Built from a Five Bar Linkage 

**Title (ZH)**: Forbal：基于五杆机构的二至五自由度力平衡机器人 manipulator 

**Authors**: Yash Vyas, Matteo Bottin  

**Link**: [PDF](https://arxiv.org/pdf/2509.03119)  

**Abstract**: A force balanced manipulator design based on the closed chain planar five bar linkage is developed and experimentally validated. We present 2 variants as a modular design: Forbal-2, a planar 2-DOF manipulator, and its extension to 5-DOF spatial motion called Forbal-5. The design considerations in terms of geometric, kinematic, and dynamic design that fulfill the force balance conditions while maximizing workspace are discussed. Then, the inverse kinematics of both variants are derived from geometric principles.
We validate the improvements from force balancing the manipulator through comparative experiments with counter mass balanced and unbalanced configurations. The results show how the balanced configuration yields a reduction in the average reaction moments of up to 66\%, a reduction of average joint torques of up to 79\%, as well as a noticeable reduction in position error for Forbal-2. For Forbal-5, which has a higher end effector payload mass, the joint torques are reduced up to 84\% for the balanced configuration. Experimental results validate that the balanced manipulator design is suitable for applications where the reduction of joint torques and reaction forces/moments helps achieve millimeter level precision. 

**Abstract (ZH)**: 基于闭链平面五连杆的力平衡 manipulator 设计及实验验证：Forbal-2 和 Forbal-5 的模块化设计与逆运动学分析及其力平衡配置的优势验证 

---
# Uncertainty-aware Test-Time Training (UT$^3$) for Efficient On-the-fly Domain Adaptive Dense Regression 

**Title (ZH)**: 面向高效实时域自适应密集回归的不确定性感知测试时训练（UT$^3$） 

**Authors**: Uddeshya Upadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2509.03012)  

**Abstract**: Deep neural networks (DNNs) are increasingly being used in autonomous systems. However, DNNs do not generalize well to domain shift. Adapting to a continuously evolving environment is a safety-critical challenge inevitably faced by all autonomous systems deployed to the real world. Recent work on test-time training proposes methods that adapt to a new test distribution on the fly by optimizing the DNN model for each test input using self-supervision. However, these techniques result in a sharp increase in inference time as multiple forward and backward passes are required for a single test sample (for test-time training) before finally making the prediction based on the fine-tuned features. This is undesirable for real-world robotics applications where these models may be deployed to resource constraint hardware with strong latency requirements. In this work, we propose a new framework (called UT$^3$) that leverages test-time training for improved performance in the presence of continuous domain shift while also decreasing the inference time, making it suitable for real-world applications. Our method proposes an uncertainty-aware self-supervision task for efficient test-time training that leverages the quantified uncertainty to selectively apply the training leading to sharp improvements in the inference time while performing comparably to standard test-time training protocol. Our proposed protocol offers a continuous setting to identify the selected keyframes, allowing the end-user to control how often to apply test-time training. We demonstrate the efficacy of our method on a dense regression task - monocular depth estimation. 

**Abstract (ZH)**: 基于连续领域转移的高效测试时训练框架UT$^3$ 

---
# CTBC: Contact-Triggered Blind Climbing for Wheeled Bipedal Robots with Instruction Learning and Reinforcement Learning 

**Title (ZH)**: 基于接触触发的轮式双足机器人盲爬行指令学习与强化学习方法（CTBC：Contact-Triggered Blind Climbing for Wheeled Bipedal Robots with Instruction Learning and Reinforcement Learning） 

**Authors**: Rankun Li, Hao Wang, Qi Li, Zhuo Han, Yifei Chu, Linqi Ye, Wende Xie, Wenlong Liao  

**Link**: [PDF](https://arxiv.org/pdf/2509.02986)  

**Abstract**: In recent years, wheeled bipedal robots have gained increasing attention due to their advantages in mobility, such as high-speed locomotion on flat terrain. However, their performance on complex environments (e.g., staircases) remains inferior to that of traditional legged robots. To overcome this limitation, we propose a general contact-triggered blind climbing (CTBC) framework for wheeled bipedal robots. Upon detecting wheel-obstacle contact, the robot triggers a leg-lifting motion to overcome the obstacle. By leveraging a strongly-guided feedforward trajectory, our method enables the robot to rapidly acquire agile leg-lifting skills, significantly enhancing its capability to traverse unstructured terrains. The approach has been experimentally validated and successfully deployed on LimX Dynamics' wheeled bipedal robot, Tron1. Real-world tests demonstrate that Tron1 can reliably climb obstacles well beyond its wheel radius using only proprioceptive feedback. 

**Abstract (ZH)**: recent years, 轮式双足机器人由于其在移动性上的优势，如在平坦地面上的高速运动，受到了越来越多的关注。然而，它们在复杂环境（如楼梯）中的表现仍逊于传统腿式机器人。为克服这一局限，我们提出了一种适用于轮式双足机器人的通用接触触发盲攀爬（CTBC）框架。在检测到车轮与障碍物接触时，机器人会触发一个抬腿动作以克服障碍物。通过利用强引导的前瞻轨迹，我们的方法使机器人能够迅速掌握敏捷的抬腿技能，显著增强其穿越非结构化地形的能力。该方法已在 LimX Dynamics 的轮式双足机器人 Tron1 上通过实验验证并成功部署。实地测试表明，Tron1 可以仅利用本体感受反馈可靠地攀爬远超过其车轮半径的障碍物。 

---
# DUViN: Diffusion-Based Underwater Visual Navigation via Knowledge-Transferred Depth Features 

**Title (ZH)**: DUViN：基于扩散的知识转移深度特征 underwater视觉导航 

**Authors**: Jinghe Yang, Minh-Quan Le, Mingming Gong, Ye Pu  

**Link**: [PDF](https://arxiv.org/pdf/2509.02983)  

**Abstract**: Autonomous underwater navigation remains a challenging problem due to limited sensing capabilities and the difficulty of constructing accurate maps in underwater environments. In this paper, we propose a Diffusion-based Underwater Visual Navigation policy via knowledge-transferred depth features, named DUViN, which enables vision-based end-to-end 4-DoF motion control for underwater vehicles in unknown environments. DUViN guides the vehicle to avoid obstacles and maintain a safe and perception awareness altitude relative to the terrain without relying on pre-built maps. To address the difficulty of collecting large-scale underwater navigation datasets, we propose a method that ensures robust generalization under domain shifts from in-air to underwater environments by leveraging depth features and introducing a novel model transfer strategy. Specifically, our training framework consists of two phases: we first train the diffusion-based visual navigation policy on in-air datasets using a pre-trained depth feature extractor. Secondly, we retrain the extractor on an underwater depth estimation task and integrate the adapted extractor into the trained navigation policy from the first step. Experiments in both simulated and real-world underwater environments demonstrate the effectiveness and generalization of our approach. The experimental videos are available at this https URL. 

**Abstract (ZH)**: 基于知识迁移深度特征的自主水下视觉导航政策：DUViN 

---
# IL-SLAM: Intelligent Line-assisted SLAM Based on Feature Awareness for Dynamic Environments 

**Title (ZH)**: 基于特征awareness的智能线协助SLAM用于动态环境 

**Authors**: Haolan Zhang, Thanh Nguyen Canh, Chenghao Li, Ruidong Yang, Yonghoon Ji, Nak Young Chong  

**Link**: [PDF](https://arxiv.org/pdf/2509.02972)  

**Abstract**: Visual Simultaneous Localization and Mapping (SLAM) plays a crucial role in autonomous systems. Traditional SLAM methods, based on static environment assumptions, struggle to handle complex dynamic environments. Recent dynamic SLAM systems employ geometric constraints and deep learning to remove dynamic features, yet this creates a new challenge: insufficient remaining point features for subsequent SLAM processes. Existing solutions address this by continuously introducing additional line and plane features to supplement point features, achieving robust tracking and pose estimation. However, current methods continuously introduce additional features regardless of necessity, causing two problems: unnecessary computational overhead and potential performance degradation from accumulated low-quality additional features and noise. To address these issues, this paper proposes a feature-aware mechanism that evaluates whether current features are adequate to determine if line feature support should be activated. This decision mechanism enables the system to introduce line features only when necessary, significantly reducing computational complexity of additional features while minimizing the introduction of low-quality features and noise. In subsequent processing, the introduced line features assist in obtaining better initial camera poses through tracking, local mapping, and loop closure, but are excluded from global optimization to avoid potential negative impacts from low-quality additional features in long-term process. Extensive experiments on TUM datasets demonstrate substantial improvements in both ATE and RPE metrics compared to ORB-SLAM3 baseline and superior performance over other dynamic SLAM and multi-feature methods. 

**Abstract (ZH)**: 视觉 simultaneous localization and mapping (SLAM) 在自主系统中发挥着关键作用。传统的 SLAM 方法基于静态环境假设，难以处理复杂的动态环境。最近的动态 SLAM 系统采用几何约束和深度学习来移除动态特征，但这也带来了一个新的挑战：剩余的点特征不足，不足以支持后续的 SLAM 过程。现有的解决方案通过不断引入额外的线性和平面特征来补充点特征，实现稳健的跟踪和姿态估计。然而，当前的方法在不需要时也不断引入额外特征，这导致了两个问题：不必要的计算开销和由于低质量的额外特征和噪声累积而导致的潜在性能下降。为了应对这些问题，本文提出了一种特征感知机制，该机制评估当前特征是否足够，以确定是否激活线特征支持。这种决策机制能使系统仅在必要时引入线特征，显著减少了额外特征的计算复杂性，同时减少了低质量特征和噪声的引入。在后续处理中，引入的线特征通过跟踪、局部地图构建和环视闭合辅助获得更好的初始相机姿态，但在全局优化中被排除，以避免长期过程中低质量额外特征的潜在负面影响。在 TUM 数据集上的 extensive 实验表明，与 ORB-SLAM3 基准相比，该方法在 ATE 和 RPE 指标上取得了显著的改进，并且在与其他动态 SLAM 和多特征方法的性能比较中表现出优越性。 

---
# Generalizable Skill Learning for Construction Robots with Crowdsourced Natural Language Instructions, Composable Skills Standardization, and Large Language Model 

**Title (ZH)**: 基于群众 sourced 自然语言指令的通用技能学习、可组合技能标准化及大规模语言模型 

**Authors**: Hongrui Yu, Vineet R. Kamat, Carol C. Menassa  

**Link**: [PDF](https://arxiv.org/pdf/2509.02876)  

**Abstract**: The quasi-repetitive nature of construction work and the resulting lack of generalizability in programming construction robots presents persistent challenges to the broad adoption of robots in the construction industry. Robots cannot achieve generalist capabilities as skills learnt from one domain cannot readily transfer to another work domain or be directly used to perform a different set of tasks. Human workers have to arduously reprogram their scene-understanding, path-planning, and manipulation components to enable the robots to perform alternate work tasks. The methods presented in this paper resolve a significant proportion of such reprogramming workload by proposing a generalizable learning architecture that directly teaches robots versatile task-performance skills through crowdsourced online natural language instructions. A Large Language Model (LLM), a standardized and modularized hierarchical modeling approach, and Building Information Modeling-Robot sematic data pipeline are developed to address the multi-task skill transfer problem. The proposed skill standardization scheme and LLM-based hierarchical skill learning framework were tested with a long-horizon drywall installation experiment using a full-scale industrial robotic manipulator. The resulting robot task learning scheme achieves multi-task reprogramming with minimal effort and high quality. 

**Abstract (ZH)**: 建筑工作的准重复性质及其导致的程序设计建筑机器人的一般化缺乏性给建筑行业广泛应用机器人带来了持续的挑战。机器人无法实现通用能力，因为从一个领域学到的技能不能轻易转移到另一个工作领域或直接用于执行不同的任务集合。工人不得不费力重新编程机器人的场景理解、路径规划和操作组件，以使机器人能够执行不同的工作任务。本文提出的 метод解决了相当一部分这种再编程工作量，通过提出一种通用的学习架构，直接利用众包在线自然语言指令教授机器人多样化的任务执行技能。开发了大规模语言模型（LLM）、标准化和模块化的分层建模方法以及基于Building Information Modeling-Robot语义数据管道，以解决多任务技能转移问题。提出的能力标准化方案和基于LLM的层次技能学习框架在一项长期干燥墙安装实验中使用全尺寸工业机器人操作器进行了测试。所提出的方法实现了少 Effort 高质量的多任务再编程。 

---
# Robotic 3D Flower Pose Estimation for Small-Scale Urban Farms 

**Title (ZH)**: 小型城市农场中基于机器人的三维花朵姿态估计 

**Authors**: Harsh Muriki, Hong Ray Teo, Ved Sengupta, Ai-Ping Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.02870)  

**Abstract**: The small scale of urban farms and the commercial availability of low-cost robots (such as the FarmBot) that automate simple tending tasks enable an accessible platform for plant phenotyping. We have used a FarmBot with a custom camera end-effector to estimate strawberry plant flower pose (for robotic pollination) from acquired 3D point cloud models. We describe a novel algorithm that translates individual occupancy grids along orthogonal axes of a point cloud to obtain 2D images corresponding to the six viewpoints. For each image, 2D object detection models for flowers are used to identify 2D bounding boxes which can be converted into the 3D space to extract flower point clouds. Pose estimation is performed by fitting three shapes (superellipsoids, paraboloids and planes) to the flower point clouds and compared with manually labeled ground truth. Our method successfully finds approximately 80% of flowers scanned using our customized FarmBot platform and has a mean flower pose error of 7.7 degrees, which is sufficient for robotic pollination and rivals previous results. All code will be made available at this https URL. 

**Abstract (ZH)**: 小型城市农场和商业化的低成本机器人（如FarmBot）实现简单养护任务的自动化，为植物表型分析提供了一个可访问的平台。我们使用具有自定义相机末端执行器的FarmBot，从获取的3D点云模型中估算草莓植株花的姿态（用于机器人授粉）。我们描述了一种新颖的算法，该算法将点云沿正交轴的单个占用网格转换为对应于六个视点的2D图像。对于每张图像，使用2D物体检测模型检测花朵，并将其2D边界框转换到3D空间以提取花朵点云。姿态估计通过将三个形状（超椭球、抛物面和平面）拟合到花朵点云并与手动标注的真实标注进行比较。我们的方法成功地在自定义的FarmBot平台上扫描约80%的花朵，并且平均花朵姿态误差为7.7度，足以用于机器人授粉，且优于先前的结果。所有代码将于以下网址公开：https://this-url。 

---
# Multi-Embodiment Locomotion at Scale with extreme Embodiment Randomization 

**Title (ZH)**: 大规模极端体征随机化下的多体征运动学习 

**Authors**: Nico Bohlinger, Jan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2509.02815)  

**Abstract**: We present a single, general locomotion policy trained on a diverse collection of 50 legged robots. By combining an improved embodiment-aware architecture (URMAv2) with a performance-based curriculum for extreme Embodiment Randomization, our policy learns to control millions of morphological variations. Our policy achieves zero-shot transfer to unseen real-world humanoid and quadruped robots. 

**Abstract (ZH)**: 我们提出了一种通用的运动策略，该策略在50种腿足机器人组成的多样化集合上进行训练。通过结合改进的体态意识架构(URMAv2)与基于性能的极端体态随机化课程，我们的策略学会了控制百万种形态学变体。该策略实现了对未见过的真实世界人形和四足机器人的零样本迁移。 

---
# Improving the Resilience of Quadrotors in Underground Environments by Combining Learning-based and Safety Controllers 

**Title (ZH)**: 结合学习型和安全控制器以提高地下环境中四旋翼无人机的鲁棒性 

**Authors**: Isaac Ronald Ward, Mark Paral, Kristopher Riordan, Mykel J. Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2509.02808)  

**Abstract**: Autonomously controlling quadrotors in large-scale subterranean environments is applicable to many areas such as environmental surveying, mining operations, and search and rescue. Learning-based controllers represent an appealing approach to autonomy, but are known to not generalize well to `out-of-distribution' environments not encountered during training. In this work, we train a normalizing flow-based prior over the environment, which provides a measure of how far out-of-distribution the quadrotor is at any given time. We use this measure as a runtime monitor, allowing us to switch between a learning-based controller and a safe controller when we are sufficiently out-of-distribution. Our methods are benchmarked on a point-to-point navigation task in a simulated 3D cave environment based on real-world point cloud data from the DARPA Subterranean Challenge Final Event Dataset. Our experimental results show that our combined controller simultaneously possesses the liveness of the learning-based controller (completing the task quickly) and the safety of the safety controller (avoiding collision). 

**Abstract (ZH)**: 自主控制大型地下环境中四旋翼无人机 applicability to 环境调研、 Mining 操作和搜索救援。基于学习的控制器代表了一种有吸引力的自治方法，但它们 known 不擅长将已训练环境以外的 `out-of-distribution' 环境进行推广。在本工作中，我们训练了一个环境的正規化流先验分布，提供了四旋翼无人机在任何给定时间偏离正常分布的程度。我们使用此度量作为运行业务监控器，允许我们在充分偏离正常分布时切换到学习基于的控制器和安全控制器。我们的方法在基于真实世界点云数据的 DARPA Subterranean 挑战最终赛事数据集构建的模拟 3D 洞穴环境中，针对点对点导航任务进行了基准测试。我们的实验结果表明，我们的组合控制器同时具备基于学习的控制器的生命力（快速完成任务）和安全控制器的安全性（避免碰撞）。 

---
# A Digital Twin for Robotic Post Mortem Tissue Sampling using Virtual Reality 

**Title (ZH)**: 基于虚拟现实的机器人尸检组织采样数字孪生系统 

**Authors**: Maximilian Neidhardt, Ludwig Bosse, Vidas Raudonis, Kristina Allgoewer, Axel Heinemann, Benjamin Ondruschka, Alexander Schlaefer  

**Link**: [PDF](https://arxiv.org/pdf/2509.02760)  

**Abstract**: Studying tissue samples obtained during autopsies is the gold standard when diagnosing the cause of death and for understanding disease pathophysiology. Recently, the interest in post mortem minimally invasive biopsies has grown which is a less destructive approach in comparison to an open autopsy and reduces the risk of infection. While manual biopsies under ultrasound guidance are more widely performed, robotic post mortem biopsies have been recently proposed. This approach can further reduce the risk of infection for physicians. However, planning of the procedure and control of the robot need to be efficient and usable. We explore a virtual reality setup with a digital twin to realize fully remote planning and control of robotic post mortem biopsies. The setup is evaluated with forensic pathologists in a usability study for three interaction methods. Furthermore, we evaluate clinical feasibility and evaluate the system with three human cadavers. Overall, 132 needle insertions were performed with an off-axis needle placement error of 5.30+-3.25 mm. Tissue samples were successfully biopsied and histopathologically verified. Users reported a very intuitive needle placement approach, indicating that the system is a promising, precise, and low-risk alternative to conventional approaches. 

**Abstract (ZH)**: 研究尸检过程中获取的组织样本是诊断死亡原因和理解疾病病理生理学的金标准。最近，对死后微创活组织检查的兴趣增长，这是一种比开放尸检破坏性较小的方法，可减少感染风险。虽然在超声引导下的手动活检更为常见，但近期提出了死后使用机器人进行活检的方法，该方法进一步降低了医生的感染风险。然而，该程序的规划和机器人控制需要高效且易用。我们探索了基于虚拟现实和数字孪生的全远程规划和控制机器人死后活检的设置。该设置在可用性研究中使用法医病理学家评估了三种交互方法。此外，我们评估了临床可行性，并使用三具人类遗体评估了系统。总共进行了132次针刺，轴外针放置误差为5.30±3.25 mm。成功获取了组织样本并进行了组织病理学验证。用户报告了非常直观的针刺放置方法，表明该系统是传统方法的一种有前途、精确且低风险的替代方案。 

---
# The Impact of Adaptive Emotional Alignment on Mental State Attribution and User Empathy in HRI 

**Title (ZH)**: 自适应情绪共鸣对社会机器人交互中心理状态归因和用户同理心的影响 

**Authors**: Giorgia Buracchio, Ariele Callegari, Massimo Donini, Cristina Gena, Antonio Lieto, Alberto Lillo, Claudio Mattutino, Alessandro Mazzei, Linda Pigureddu, Manuel Striani, Fabiana Vernero  

**Link**: [PDF](https://arxiv.org/pdf/2509.02749)  

**Abstract**: The paper presents an experiment on the effects of adaptive emotional alignment between agents, considered a prerequisite for empathic communication, in Human-Robot Interaction (HRI). Using the NAO robot, we investigate the impact of an emotionally aligned, empathic, dialogue on these aspects: (i) the robot's persuasive effectiveness, (ii) the user's communication style, and (iii) the attribution of mental states and empathy to the robot. In an experiment with 42 participants, two conditions were compared: one with neutral communication and another where the robot provided responses adapted to the emotions expressed by the users. The results show that emotional alignment does not influence users' communication styles or have a persuasive effect. However, it significantly influences attribution of mental states to the robot and its perceived empathy 

**Abstract (ZH)**: 适应性情感一致性对人机互动中情感共鸣对话影响的实验研究：以NAO机器人为例 

---
# Acrobotics: A Generalist Approahc To Quadrupedal Robots' Parkour 

**Title (ZH)**: 机械臂术：四足机器人过障碍的通才方法 

**Authors**: Guillaume Gagné-Labelle, Vassil Atanassov, Ioannis Havoutis  

**Link**: [PDF](https://arxiv.org/pdf/2509.02727)  

**Abstract**: Climbing, crouching, bridging gaps, and walking up stairs are just a few of the advantages that quadruped robots have over wheeled robots, making them more suitable for navigating rough and unstructured terrain. However, executing such manoeuvres requires precise temporal coordination and complex agent-environment interactions. Moreover, legged locomotion is inherently more prone to slippage and tripping, and the classical approach of modeling such cases to design a robust controller thus quickly becomes impractical. In contrast, reinforcement learning offers a compelling solution by enabling optimal control through trial and error. We present a generalist reinforcement learning algorithm for quadrupedal agents in dynamic motion scenarios. The learned policy rivals state-of-the-art specialist policies trained using a mixture of experts approach, while using only 25% as many agents during training. Our experiments also highlight the key components of the generalist locomotion policy and the primary factors contributing to its success. 

**Abstract (ZH)**: 四足机器人在攀爬、蹲伏、跨越障碍和上下楼梯等方面具有明显优势，使其更适合于穿越崎岖不平和未结构化的地形。然而，执行这些动作需要精确的时间协调和复杂的代理-环境交互。此外，腿足运动更容易发生打滑和绊倒，经典的通过建模此类情况来设计稳健控制器的方法很快变得不切实际。相比之下，强化学习通过试错提供了令人信服的解决方案，以实现最优控制。我们提出了一个适用于动态运动场景的通用强化学习算法。所学习的策略在训练中仅使用专家混合方法训练的专业策略的约25%的代理，就能与最先进的专业策略相媲美。我们的实验还强调了通用运动策略的关键组成部分和主要成功因素。 

---
# sam-llm: interpretable lane change trajectoryprediction via parametric finetuning 

**Title (ZH)**: SAM-LLM: 基于参数微调的可解释车道变更轨迹预测 

**Authors**: Zhuo Cao, Yunxiao Shi, Min Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.03462)  

**Abstract**: This work introduces SAM-LLM, a novel hybrid architecture that bridges the gap between the contextual reasoning of Large Language Models (LLMs) and the physical precision of kinematic lane change models for autonomous driving. The system is designed for interpretable lane change trajectory prediction by finetuning an LLM to output the core physical parameters of a trajectory model instead of raw coordinates. For lane-keeping scenarios, the model predicts discrete coordinates, but for lane change maneuvers, it generates the parameters for an enhanced Sinusoidal Acceleration Model (SAM), including lateral displacement, maneuver duration, initial lateral velocity, and longitudinal velocity change. This parametric approach yields a complete, continuous, and physically plausible trajectory model that is inherently interpretable and computationally efficient, achieving an 80% reduction in output size compared to coordinate-based methods. The SAM-LLM achieves a state-of-the-art overall intention prediction accuracy of 98.73%, demonstrating performance equivalent to traditional LLM predictors while offering significant advantages in explainability and resource efficiency. 

**Abstract (ZH)**: SAM-LLM：一种结合大规模语言模型上下文推理与动力学车道变更模型物理精度的新型混合架构 

---
# SmartPoser: Arm Pose Estimation with a Smartphone and Smartwatch Using UWB and IMU Data 

**Title (ZH)**: SmartPoser：基于UWB和IMU数据的智能手机和智能手表臂部姿态估计 

**Authors**: Nathan DeVrio, Vimal Mollyn, Chris Harrison  

**Link**: [PDF](https://arxiv.org/pdf/2509.03451)  

**Abstract**: The ability to track a user's arm pose could be valuable in a wide range of applications, including fitness, rehabilitation, augmented reality input, life logging, and context-aware assistants. Unfortunately, this capability is not readily available to consumers. Systems either require cameras, which carry privacy issues, or utilize multiple worn IMUs or markers. In this work, we describe how an off-the-shelf smartphone and smartwatch can work together to accurately estimate arm pose. Moving beyond prior work, we take advantage of more recent ultra-wideband (UWB) functionality on these devices to capture absolute distance between the two devices. This measurement is the perfect complement to inertial data, which is relative and suffers from drift. We quantify the performance of our software-only approach using off-the-shelf devices, showing it can estimate the wrist and elbow joints with a \hl{median positional error of 11.0~cm}, without the user having to provide training data. 

**Abstract (ZH)**: 一款智能手机和智能手表协同工作的方法可以准确估计手臂姿态 

---
# EclipseTouch: Touch Segmentation on Ad Hoc Surfaces using Worn Infrared Shadow Casting 

**Title (ZH)**: EclipseTouch：穿戴式红外投影像素分割adhoc表面触摸操作 

**Authors**: Vimal Mollyn, Nathan DeVrio, Chris Harrison  

**Link**: [PDF](https://arxiv.org/pdf/2509.03430)  

**Abstract**: The ability to detect touch events on uninstrumented, everyday surfaces has been a long-standing goal for mixed reality systems. Prior work has shown that virtual interfaces bound to physical surfaces offer performance and ergonomic benefits over tapping at interfaces floating in the air. A wide variety of approaches have been previously developed, to which we contribute a new headset-integrated technique called \systemname. We use a combination of a computer-triggered camera and one or more infrared emitters to create structured shadows, from which we can accurately estimate hover distance (mean error of 6.9~mm) and touch contact (98.0\% accuracy). We discuss how our technique works across a range of conditions, including surface material, interaction orientation, and environmental lighting. 

**Abstract (ZH)**: 混合现实系统中对未标记日常表面的触觉事件检测能力一直是长期目标。我们贡献了一种新的头戴式集成技术\systename，该技术利用计算机触发的摄像头和一个或多个红外发射器来创建结构化阴影，从而可以准确估计悬浮距离（平均误差6.9毫米）和触觉接触（准确率98.0%）。我们讨论了该技术在多种条件下的工作情况，包括表面材质、交互方向和环境照明。 

---
# ANNIE: Be Careful of Your Robots 

**Title (ZH)**: ANNIE: 注意你的机器人 

**Authors**: Yiyang Huang, Zixuan Wang, Zishen Wan, Yapeng Tian, Haobo Xu, Yinhe Han, Yiming Gan  

**Link**: [PDF](https://arxiv.org/pdf/2509.03383)  

**Abstract**: The integration of vision-language-action (VLA) models into embodied AI (EAI) robots is rapidly advancing their ability to perform complex, long-horizon tasks in humancentric environments. However, EAI systems introduce critical security risks: a compromised VLA model can directly translate adversarial perturbations on sensory input into unsafe physical actions. Traditional safety definitions and methodologies from the machine learning community are no longer sufficient. EAI systems raise new questions, such as what constitutes safety, how to measure it, and how to design effective attack and defense mechanisms in physically grounded, interactive settings. In this work, we present the first systematic study of adversarial safety attacks on embodied AI systems, grounded in ISO standards for human-robot interactions. We (1) formalize a principled taxonomy of safety violations (critical, dangerous, risky) based on physical constraints such as separation distance, velocity, and collision boundaries; (2) introduce ANNIEBench, a benchmark of nine safety-critical scenarios with 2,400 video-action sequences for evaluating embodied safety; and (3) ANNIE-Attack, a task-aware adversarial framework with an attack leader model that decomposes long-horizon goals into frame-level perturbations. Our evaluation across representative EAI models shows attack success rates exceeding 50% across all safety categories. We further demonstrate sparse and adaptive attack strategies and validate the real-world impact through physical robot experiments. These results expose a previously underexplored but highly consequential attack surface in embodied AI systems, highlighting the urgent need for security-driven defenses in the physical AI era. Code is available at this https URL. 

**Abstract (ZH)**: 视觉-语言-行动模型在体化人工智能机器人中的整合正迅速提高其在以人为中心环境中执行复杂长时间任务的能力。然而，体化人工智能系统引入了关键的安全风险：被篡改的视觉-语言-行动模型可以直接将感官输入的恶意干扰转化为不安全的物理行动。传统的机器学习领域的安全定义和方法已经不足以应对这一问题。体化人工智能系统提出了新的问题，如安全的定义、衡量方法以及如何在物理上接地的互动环境中设计有效的攻击和防御机制。在本项工作中，我们基于ISO人类-机器人交互标准，提出了体化人工智能系统中对抗性安全攻击的第一个系统性研究。我们(1)基于物理约束（如距离、速度和碰撞边界）制定了一个原则性的安全违规分类体系（关键性、危险性和风险性）；(2)引入了ANNIEBench，一个包含9个关键安全场景和2400个视频-行动序列的基准测试集，用于评估体化安全性；(3)提出了ANNIE-Attack，一个任务感知的对抗性框架，具有一个分解长时间目标的攻击领导者模型，逐帧进行干扰。我们在代表性体化人工智能模型上的评估显示，所有安全类别的攻击成功率超过50%。我们进一步展示了稀疏和适应性的攻击策略，并通过物理机器人实验验证了其实际影响。这些结果揭示了在体化人工智能系统中一个之前未被充分探索但也极具严重性的攻击面，突显了在物理人工智能时代安全驱动防御的迫切需求。代码可在以下链接获取。 

---
# Dependency Chain Analysis of ROS 2 DDS QoS Policies: From Lifecycle Tutorial to Static Verification 

**Title (ZH)**: ROS 2 DDS QoS策略的依赖链分析：从生命周期教程到静态验证 

**Authors**: Sanghoon Lee, Junha Kang, Kyung-Joon Park  

**Link**: [PDF](https://arxiv.org/pdf/2509.03381)  

**Abstract**: Robot Operating System 2 (ROS 2) relies on the Data Distribution Service (DDS), which offers more than 20 Quality of Service (QoS) policies governing availability, reliability, and resource usage. Yet ROS 2 users lack clear guidance on safe policy combinations and validation processes prior to deployment, which often leads to trial-and-error tuning and unexpected runtime failures. To address these challenges, we analyze DDS Publisher-Subscriber communication over a life cycle divided into Discovery, Data Exchange, and Disassociation, and provide a user oriented tutorial explaining how 16 QoS policies operate in each phase. Building on this analysis, we derive a QoS dependency chain that formalizes inter-policy relationships and classifies 41 dependency violation rules, capturing constraints that commonly cause communication failures in practice. Finally, we introduce QoS Guard, a ROS 2 package that statically validates DDS XML profiles offline, flags conflicts, and enables safe, predeployment tuning without establishing a live ROS 2 session. Together, these contributions give ROS 2 users both conceptual insight and a concrete tool that enables early detection of misconfigurations, improving the reliability and resource efficiency of ROS 2 based robotic systems. 

**Abstract (ZH)**: ROS 2中的Data Distribution Service (DDS)提供了超过20种服务质量（QoS）策略， governs可用性、可靠性和资源使用。然而，ROS 2用户缺乏明确的策略组合和部署前验证指导，这通常导致试错调整和意外的运行时故障。为解决这些挑战，我们按生命周期划分为发现、数据交换和脱离阶段分析DDS发布者-订阅者通信，并提供面向用户的教程解释每个阶段中如何运行16种QoS策略。基于这一分析，我们推导出一种QoS依赖链，明确了策略之间的关系，并分类出41条依赖违规规则，捕捉实践中常见的通信故障限制条件。最后，我们介绍了QoS Guard，这是一个ROS 2包，可以在离线情况下静态验证DDS XML配置文件，标记冲突，并在无需建立实时ROS 2会话的情况下实现安全的部署前调整。这些贡献为ROS 2用户提供了概念性的见解和一个具体的工具，以便早期检测配置错误，提高基于ROS 2的机器人系统的可靠性和资源效率。 

---
# AI Safety Assurance in Electric Vehicles: A Case Study on AI-Driven SOC Estimation 

**Title (ZH)**: 电动汽车中的人工智能安全保障：基于人工智能驱动的SOC估算案例研究 

**Authors**: Martin Skoglund, Fredrik Warg, Aria Mirzai, Anders Thorsen, Karl Lundgren, Peter Folkesson, Bastian Havers-zulka  

**Link**: [PDF](https://arxiv.org/pdf/2509.03270)  

**Abstract**: Integrating Artificial Intelligence (AI) technology in electric vehicles (EV) introduces unique challenges for safety assurance, particularly within the framework of ISO 26262, which governs functional safety in the automotive domain. Traditional assessment methodologies are not geared toward evaluating AI-based functions and require evolving standards and practices. This paper explores how an independent assessment of an AI component in an EV can be achieved when combining ISO 26262 with the recently released ISO/PAS 8800, whose scope is AI safety for road vehicles. The AI-driven State of Charge (SOC) battery estimation exemplifies the process. Key features relevant to the independent assessment of this extended evaluation approach are identified. As part of the evaluation, robustness testing of the AI component is conducted using fault injection experiments, wherein perturbed sensor inputs are systematically introduced to assess the component's resilience to input variance. 

**Abstract (ZH)**: 将人工智能技术集成到电动汽车中，为ISO 26262框架下的功能安全保证带来了独特的挑战。本文探讨了如何结合ISO 26262和近期发布的ISO/PAS 8800来独立评估电动汽车中的人工智能组件，后者专注于道路车辆的人工智能安全。以基于人工智能的电池荷电状态（SOC）估算为例，识别了这种扩展评估方法中关键的相关特性。作为评估的一部分，通过故障注入实验，对人工智能组件的鲁棒性进行了测试，系统地引入扰动传感器输入以评估其对输入变化的抗扰性。 

---
# Decentralised self-organisation of pivoting cube ensembles using geometric deep learning 

**Title (ZH)**: 使用几何深度学习实现 pivoting 立方体编队的去中心化自我组织 

**Authors**: Nadezhda Dobreva, Emmanuel Blazquez, Jai Grover, Dario Izzo, Yuzhen Qin, Dominik Dold  

**Link**: [PDF](https://arxiv.org/pdf/2509.03140)  

**Abstract**: We present a decentralized model for autonomous reconfiguration of homogeneous pivoting cube modular robots in two dimensions. Each cube in the ensemble is controlled by a neural network that only gains information from other cubes in its local neighborhood, trained using reinforcement learning. Furthermore, using geometric deep learning, we include the grid symmetries of the cube ensemble in the neural network architecture. We find that even the most localized versions succeed in reconfiguring to the target shape, although reconfiguration happens faster the more information about the whole ensemble is available to individual cubes. Near-optimal reconfiguration is achieved with only nearest neighbor interactions by using multiple information passing between cubes, allowing them to accumulate more global information about the ensemble. Compared to standard neural network architectures, using geometric deep learning approaches provided only minor benefits. Overall, we successfully demonstrate mostly local control of a modular self-assembling system, which is transferable to other space-relevant systems with different action spaces, such as sliding cube modular robots and CubeSat swarms. 

**Abstract (ZH)**: 我们提出了一种去中心化模型，用于自主重构二维环境中的同构 pivot 立方模块机器人。每个立方体由只能从其局部邻域内的其他立方体获取信息的神经网络控制，并使用强化学习进行训练。此外，我们利用几何深度学习，将立方体集合的空间对称性纳入神经网络架构。我们发现，即使是最局域化的版本也能成功重构为目标形状，尽管个体立方体可以获得的整个集合的信息越多，重构过程会变得更快速。通过在立方体之间使用多种信息传递，仅通过最近邻交互即可实现接近最优的重构，使它们能够积累更多的集合全局信息。与标准神经网络架构相比，使用几何深度学习方法仅提供了一些次要益处。总体而言，我们成功地展示了模块化自组装系统的局部控制，该控制方式可转移应用于其他具有不同动作空间的空间相关系统，例如滑动立方模块机器人和CubeSat群。 

---
# Population-aware Online Mirror Descent for Mean-Field Games with Common Noise by Deep Reinforcement Learning 

**Title (ZH)**: 基于种群感知的在线镜像下降方法用于具有公共噪声的大规模动态博弈的深度强化学习 

**Authors**: Zida Wu, Mathieu Lauriere, Matthieu Geist, Olivier Pietquin, Ankur Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2509.03030)  

**Abstract**: Mean Field Games (MFGs) offer a powerful framework for studying large-scale multi-agent systems. Yet, learning Nash equilibria in MFGs remains a challenging problem, particularly when the initial distribution is unknown or when the population is subject to common noise. In this paper, we introduce an efficient deep reinforcement learning (DRL) algorithm designed to achieve population-dependent Nash equilibria without relying on averaging or historical sampling, inspired by Munchausen RL and Online Mirror Descent. The resulting policy is adaptable to various initial distributions and sources of common noise. Through numerical experiments on seven canonical examples, we demonstrate that our algorithm exhibits superior convergence properties compared to state-of-the-art algorithms, particularly a DRL version of Fictitious Play for population-dependent policies. The performance in the presence of common noise underscores the robustness and adaptability of our approach. 

**Abstract (ZH)**: MFGs基于Munchausen RL和在线镜像下降的自适应深度 reinforcement学习算法及应用 

---
# VendiRL: A Framework for Self-Supervised Reinforcement Learning of Diversely Diverse Skills 

**Title (ZH)**: VendiRL：一种自我监督的强化学习框架，用于学习多样性的技能 

**Authors**: Erik M. Lintunen  

**Link**: [PDF](https://arxiv.org/pdf/2509.02930)  

**Abstract**: In self-supervised reinforcement learning (RL), one of the key challenges is learning a diverse set of skills to prepare agents for unknown future tasks. Despite impressive advances, scalability and evaluation remain prevalent issues. Regarding scalability, the search for meaningful skills can be obscured by high-dimensional feature spaces, where relevant features may vary across downstream task domains. For evaluating skill diversity, defining what constitutes "diversity" typically requires a hard commitment to a specific notion of what it means for skills to be diverse, potentially leading to inconsistencies in how skill diversity is understood, making results across different approaches hard to compare, and leaving many forms of diversity unexplored. To address these issues, we adopt a measure of sample diversity that translates ideas from ecology to machine learning -- the Vendi Score -- allowing the user to specify and evaluate any desired form of diversity. We demonstrate how this metric facilitates skill evaluation and introduce VendiRL, a unified framework for learning diversely diverse sets of skills. Given distinct similarity functions, VendiRL motivates distinct forms of diversity, which could support skill-diversity pretraining in new and richly interactive environments where optimising for various forms of diversity may be desirable. 

**Abstract (ZH)**: 在自监督强化学习中，一个关键挑战是学习一组多样的技能以应对未知的未来任务。尽管取得了显著进展，但可扩展性和评估仍然是普遍问题。在可扩展性方面，有意义技能的搜索可能会被高维特征空间所混淆，其中相关特征在下游任务域中可能会有所不同。对于评估技能多样性，定义什么是“多样性”通常需要对技能多样性的具体含义作出硬性承诺，可能导致对技能多样性的理解不一致，使得不同方法的结果难以比较，并且忽略了多种多样性的形式。为解决这些问题，我们采用了一个样本多样性的度量方法——维迪分数——该方法将生态学的思想引入到了机器学习中，允许用户指定和评估任何所需的多样性形式。我们展示了如何通过该度量方法促进技能评估，并介绍了VendiRL，这是一种统一框架，用于学习多样性和多样化的技能集。给定不同的相似度函数，VendiRL 可以促进不同形式的多样性的出现，这在优化多种多样性形式的环境中可能支持技能多样性的预训练。 

---
# Approximate constrained stochastic optimal control via parameterized input inference 

**Title (ZH)**: 参数化输入推断实现近似约束随机最优控制 

**Authors**: Shahbaz P Qadri Syed, He Bai  

**Link**: [PDF](https://arxiv.org/pdf/2509.02922)  

**Abstract**: Approximate methods to solve stochastic optimal control (SOC) problems have received significant interest from researchers in the past decade. Probabilistic inference approaches to SOC have been developed to solve nonlinear quadratic Gaussian problems. In this work, we propose an Expectation-Maximization (EM) based inference procedure to generate state-feedback controls for constrained SOC problems. We consider the inequality constraints for the state and controls and also the structural constraints for the controls. We employ barrier functions to address state and control constraints. We show that the expectation step leads to smoothing of the state-control pair while the the maximization step on the non-zero subsets of the control parameters allows inference of structured stochastic optimal controllers. We demonstrate the effectiveness of the algorithm on unicycle obstacle avoidance, four-unicycle formation control, and quadcopter navigation in windy environment examples. In these examples, we perform an empirical study on the parametric effect of barrier functions on the state constraint satisfaction. We also present a comparative study of smoothing algorithms on the performance of the proposed approach. 

**Abstract (ZH)**: 基于EM方法求解约束随机最优控制问题的近似方法 

---
# 2nd Place Solution for CVPR2024 E2E Challenge: End-to-End Autonomous Driving Using Vision Language Model 

**Title (ZH)**: CVPR2024 E2E 挑战赛的亚军解决方案：端到端自主驾驶视觉语言模型 

**Authors**: Zilong Guo, Yi Luo, Long Sha, Dongxu Wang, Panqu Wang, Chenyang Xu, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02659)  

**Abstract**: End-to-end autonomous driving has drawn tremendous attention recently. Many works focus on using modular deep neural networks to construct the end-to-end archi-tecture. However, whether using powerful large language models (LLM), especially multi-modality Vision Language Models (VLM) could benefit the end-to-end driving tasks remain a question. In our work, we demonstrate that combining end-to-end architectural design and knowledgeable VLMs yield impressive performance on the driving tasks. It is worth noting that our method only uses a single camera and is the best camera-only solution across the leaderboard, demonstrating the effectiveness of vision-based driving approach and the potential for end-to-end driving tasks. 

**Abstract (ZH)**: 端到端自主驾驶近年来引起了广泛关注。许多研究工作集中在使用模块化深度神经网络构建端到端架构。然而，是否使用强大的大型语言模型（LLM），尤其是多模态视觉语言模型（VLM），能提升端到端驾驶任务的表现仍是一个问题。在我们的工作中，我们展示了将端到端架构设计与 knowledgeable VLM 结合起来在驾驶任务中取得了显著性能。值得注意的是，我们的方法仅使用一个摄像头，并在排行榜上达到了最佳摄像头-only 解决方案，这表明基于视觉的驾驶方法的有效性以及端到端驾驶任务的潜力。 

---
# Who Owns The Robot?: Four Ethical and Socio-technical Questions about Wellbeing Robots in the Real World through Community Engagement 

**Title (ZH)**: 谁拥有机器人？通过社区参与探索现实世界中福祉机器人伦理与社会技术问题的四个视角 

**Authors**: Minja Axelsson, Jiaee Cheong, Rune Nyrup, Hatice Gunes  

**Link**: [PDF](https://arxiv.org/pdf/2509.02624)  

**Abstract**: Recent studies indicate that robotic coaches can play a crucial role in promoting wellbeing. However, the real-world deployment of wellbeing robots raises numerous ethical and socio-technical questions and concerns. To explore these questions, we undertake a community-centered investigation to examine three different communities' perspectives on using robotic wellbeing coaches in real-world environments. We frame our work as an anticipatory ethical investigation, which we undertake to better inform the development of robotic technologies with communities' opinions, with the ultimate goal of aligning robot development with public interest. We conducted workshops with three communities who are under-represented in robotics development: 1) members of the public at a science festival, 2) women computer scientists at a conference, and 3) humanities researchers interested in history and philosophy of science. In the workshops, we collected qualitative data using the Social Robot Co-Design Canvas on Ethics. We analysed the collected qualitative data with Thematic Analysis, informed by notes taken during workshops. Through our analysis, we identify four themes regarding key ethical and socio-technical questions about the real-world use of wellbeing robots. We group participants' insights and discussions around these broad thematic questions, discuss them in light of state-of-the-art literature, and highlight areas for future investigation. Finally, we provide the four questions as a broad framework that roboticists can and should use during robotic development and deployment, in order to reflect on the ethics and socio-technical dimensions of their robotic applications, and to engage in dialogue with communities of robot users. The four questions are: 1) Is the robot safe and how can we know that?, 2) Who is the robot built for and with?, 3) Who owns the robot and the data?, and 4) Why a robot?. 

**Abstract (ZH)**: 近期的研究表明，机器人教练可以在提高福祉方面发挥关键作用。然而，福祉机器人的实际部署引发了诸多伦理和社会技术层面的问题与担忧。为了探索这些问题，我们开展了一项以社区为中心的研究，旨在考察三个不同社区对在实际环境中使用机器人福祉教练的看法。我们将工作定位于前瞻性伦理调查，旨在更好地根据社区意见指导机器人技术的发展，最终目标是使机器人开发与公众利益相契合。我们与三位代表机器人开发不足领域的社区成员进行了工作坊：1）科学节的公众成员，2）女性计算机科学家会议上的成员，3）对科学和哲学有兴趣的人文研究人员。在工作坊中，我们使用伦理方面的社会机器人共同设计画布收集定性数据。通过分析收集到的定性数据，我们识别出关于福祉机器人实际使用中关键伦理和社会技术问题的四个主题。我们将参与者对这些问题的见解和讨论归类为广泛的主题问题，结合最新文献讨论这些问题，并指明未来研究的领域。最后，我们提供了这四个问题作为机器人开发和部署的广泛框架，机器人科学家在开发和部署机器人时可以使用这些框架，以反思其机器人应用的伦理和社会技术维度，并与机器人用户的社群进行对话讨论。这四个问题是：1）机器人是否安全，我们如何知道？2）机器人是为谁建造的？又是与谁一起构建的？3）机器人及其数据的归属权归属谁？4）为什么要使用机器人？ 

---
# Separation of Three or More Autonomous Mobile Models under Hierarchical Schedulers 

**Title (ZH)**: 三种及以上自主移动模型在分层调度器下的分离方法 

**Authors**: Shota Naito, Tsukasa Ninomiya, Koichi Wada  

**Link**: [PDF](https://arxiv.org/pdf/2508.19805)  

**Abstract**: Understanding the computational power of mobile robot systems is a fundamental challenge in distributed computing. While prior work has focused on pairwise separations between models, we explore how robot capabilities, light observability, and scheduler synchrony interact in more complex ways.
We first show that the Exponential Times Expansion (ETE) problem is solvable only in the strongest model -- fully-synchronous robots with full mutual lights ($\mathcal{LUMT}^F$). We then introduce the Hexagonal Edge Traversal (HET) and TAR(d)* problems to demonstrate how internal memory and lights interact with synchrony: under weak synchrony, internal memory alone is insufficient, while full synchrony can substitute for both lights and memory.
In the asynchronous setting, we classify problems such as LP-MLCv, VEC, and ZCC to show fine-grained separations between $\mathcal{FSTA}$ and $\mathcal{FCOM}$ robots. We also analyze Vertex Traversal Rendezvous (VTR) and Leave Place Convergence (LP-Cv), illustrating the limitations of internal memory in symmetric settings.
These results extend the known separation map of 14 canonical robot models, revealing structural phenomena only visible through higher-order comparisons. Our work provides new impossibility criteria and deepens the understanding of how observability, memory, and synchrony collectively shape the computational power of mobile robots. 

**Abstract (ZH)**: 理解移动机器人系统的计算能力是分布式计算中的基本挑战。在先前的工作主要关注模型之间的成对分离时，我们探讨了机器人能力、光线可观察性和调度同步性在更复杂交互中的作用。 

---
