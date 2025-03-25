# Autonomous Generation of Sub-goals for Lifelong Learning in Robots 

**Title (ZH)**: 机器人终身学习中自主子目标生成 

**Authors**: Emanuel Fallas Hernández, Sergio Martínez Alonso, Alejandro Romero, Jose A. Becerra Permuy, Richard J. Duro  

**Link**: [PDF](https://arxiv.org/pdf/2503.18914)  

**Abstract**: One of the challenges of open-ended learning in robots is the need to autonomously discover goals and learn skills to achieve them. However, when in lifelong learning settings, it is always desirable to generate sub-goals with their associated skills, without relying on explicit reward, as steppingstones to a goal. This allows sub-goals and skills to be reused to facilitate achieving other goals. This work proposes a two-pronged approach for sub-goal generation to address this challenge: a top-down approach, where sub-goals are hierarchically derived from general goals using intrinsic motivations to discover them, and a bottom-up approach, where sub-goal chains emerge from making latent relationships between goals and perceptual classes that were previously learned in different domains explicit. These methods help the robot to autonomously generate and chain sub-goals as a way to achieve more general goals. Additionally, they create more abstract representations of goals, helping to reduce sub-goal duplication and make the learning of skills more efficient. Implemented within an existing cognitive architecture for lifelong open-ended learning and tested with a real robot, our approach enhances the robot's ability to discover and achieve goals, generate sub-goals in an efficient manner, generalize learned skills, and operate in dynamic and unknown environments without explicit intermediate rewards. 

**Abstract (ZH)**: 一种面向终身学习的机器人子目标生成方法：自上而下与自下而上的双管齐下 approach 

---
# RoboEngine: Plug-and-Play Robot Data Augmentation with Semantic Robot Segmentation and Background Generation 

**Title (ZH)**: RoboEngine: 插拔式机器人数据增强结合语义机器人分割和背景生成 

**Authors**: Chengbo Yuan, Suraj Joshi, Shaoting Zhu, Hang Su, Hang Zhao, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.18738)  

**Abstract**: Visual augmentation has become a crucial technique for enhancing the visual robustness of imitation learning. However, existing methods are often limited by prerequisites such as camera calibration or the need for controlled environments (e.g., green screen setups). In this work, we introduce RoboEngine, the first plug-and-play visual robot data augmentation toolkit. For the first time, users can effortlessly generate physics- and task-aware robot scenes with just a few lines of code. To achieve this, we present a novel robot scene segmentation dataset, a generalizable high-quality robot segmentation model, and a fine-tuned background generation model, which together form the core components of the out-of-the-box toolkit. Using RoboEngine, we demonstrate the ability to generalize robot manipulation tasks across six entirely new scenes, based solely on demonstrations collected from a single scene, achieving a more than 200% performance improvement compared to the no-augmentation baseline. All datasets, model weights, and the toolkit will be publicly released. 

**Abstract (ZH)**: 视觉增强已成为提升模仿学习视觉鲁棒性的重要技术。然而，现有方法常常受限于摄像头标定或需要受控环境（例如绿幕设置）。在此工作中，我们引入了RoboEngine，这是首个即插即用的视觉机器人数据增强工具包。首次实现了用户只需几行代码即可轻松生成物理和任务感知的机器人场景。为此，我们提出了一个全新的机器人场景分割数据集、一个泛化的高质量机器人分割模型以及一个fine-tuned背景生成模型，这些构成了该开箱即用工具包的核心组件。使用RoboEngine，我们展示了在单个场景演示的基础上，能够在六个全新场景中泛化机器人操作任务的能力，相比于无增强基线，性能提升超过200%。所有数据集、模型权重和工具包将公开发布。 

---
# Efficient Continual Adaptation of Pretrained Robotic Policy with Online Meta-Learned Adapters 

**Title (ZH)**: 高效的预训练机器人策略的在线元学习适配 

**Authors**: Ruiqi Zhu, Endong Sun, Guanhe Huang, Oya Celiktutan  

**Link**: [PDF](https://arxiv.org/pdf/2503.18684)  

**Abstract**: Continual adaptation is essential for general autonomous agents. For example, a household robot pretrained with a repertoire of skills must still adapt to unseen tasks specific to each household. Motivated by this, building upon parameter-efficient fine-tuning in language models, prior works have explored lightweight adapters to adapt pretrained policies, which can preserve learned features from the pretraining phase and demonstrate good adaptation performances. However, these approaches treat task learning separately, limiting knowledge transfer between tasks. In this paper, we propose Online Meta-Learned adapters (OMLA). Instead of applying adapters directly, OMLA can facilitate knowledge transfer from previously learned tasks to current learning tasks through a novel meta-learning objective. Extensive experiments in both simulated and real-world environments demonstrate that OMLA can lead to better adaptation performances compared to the baseline methods. The project link: this https URL. 

**Abstract (ZH)**: 持续适应对于通用自主代理至关重要。例如，预训练掌握一系列技能的家用机器人仍需要适应每个家庭特有的未见过的任务。受此启发，基于语言模型中的参数高效微调，先前的工作探索了轻量级适配器以适应预训练策略，这些适配器可以在保持预训练阶段学到的特征的同时表现出良好的适应性能。然而，这些方法将任务学习分开处理，限制了任务之间的知识迁移。在本文中，我们提出了一种在线元学习适配器(OMLA)。OMLA 不直接应用适配器，而是通过一个新的元学习目标促进先前学习任务的知识向当前学习任务的迁移。在模拟和真实环境中的广泛实验表明，OMLA 可以比基线方法获得更好的适应性能。项目链接：this https URL。 

---
# Parental Guidance: Efficient Lifelong Learning through Evolutionary Distillation 

**Title (ZH)**: 家长指导：通过进化蒸馏实现高效终身学习 

**Authors**: Octi Zhang, Quanquan Peng, Rosario Scalise, Bryon Boots  

**Link**: [PDF](https://arxiv.org/pdf/2503.18531)  

**Abstract**: Developing robotic agents that can perform well in diverse environments while showing a variety of behaviors is a key challenge in AI and robotics. Traditional reinforcement learning (RL) methods often create agents that specialize in narrow tasks, limiting their adaptability and diversity. To overcome this, we propose a preliminary, evolution-inspired framework that includes a reproduction module, similar to natural species reproduction, balancing diversity and specialization. By integrating RL, imitation learning (IL), and a coevolutionary agent-terrain curriculum, our system evolves agents continuously through complex tasks. This approach promotes adaptability, inheritance of useful traits, and continual learning. Agents not only refine inherited skills but also surpass their predecessors. Our initial experiments show that this method improves exploration efficiency and supports open-ended learning, offering a scalable solution where sparse reward coupled with diverse terrain environments induces a multi-task setting. 

**Abstract (ZH)**: 基于进化启发的框架：通过集成强化学习、模仿学习和协同进化代理-地形课程以促进多样性和适应性 

---
# P3Nav: A Unified Framework for Embodied Navigation Integrating Perception, Planning, and Prediction 

**Title (ZH)**: P3Nav：一种融合感知、规划和预测的统一 embodied 导航框架 

**Authors**: Yufeng Zhong, Chengjian Feng, Feng Yan, Fanfan Liu, Liming Zheng, Lin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.18525)  

**Abstract**: In language-guided visual navigation, agents locate target objects in unseen environments using natural language instructions. For reliable navigation in unfamiliar scenes, agents must possess strong perception, planning, and prediction capabilities. Additionally, when agents revisit previously explored areas during long-term navigation, they may retain irrelevant and redundant historical perceptions, leading to suboptimal results. In this work, we introduce \textbf{P3Nav}, a unified framework that integrates \textbf{P}erception, \textbf{P}lanning, and \textbf{P}rediction capabilities through \textbf{Multitask Collaboration} on navigation and embodied question answering (EQA) tasks, thereby enhancing navigation performance. Furthermore, P3Nav employs an \textbf{Adaptive 3D-aware History Sampling} strategy to effectively and efficiently utilize historical observations. By leveraging the large language models (LLM), P3Nav comprehends diverse commands and complex visual scenes, resulting in appropriate navigation actions. P3Nav achieves a 75\% success rate in object goal navigation on the $\mathrm{CHORES}$-$\mathbb{S}$ benchmark, setting a new state-of-the-art performance. 

**Abstract (ZH)**: 在语言引导的视觉导航中，代理使用自然语言指令在未见环境中定位目标对象，需要具备强大的感知、规划和预测能力。在长期导航过程中，当代理重新访问之前探索的区域时，可能会保留无关和冗余的历史感知，导致次优结果。本文提出了一种统一框架P3Nav，通过导航和具身问答任务的多任务协作整合感知、规划和预测能力，从而提升导航性能。此外，P3Nav采用自适应3D感知采样策略，有效高效地利用历史观察。通过利用大规模语言模型（LLM），P3Nav能够理解和处理多样的指令和复杂的视觉场景，进而采取合适的导航动作。在$\mathrm{CHORES}$-$\mathbb{S}$基准测试中，P3Nav实现75%的对象目标导航成功率达到新的state-of-the-art性能。 

---
# Reinforcement Learning for Adaptive Planner Parameter Tuning: A Perspective on Hierarchical Architecture 

**Title (ZH)**: 适应性规划参数调优的强化学习方法：分层架构视角 

**Authors**: Lu Wangtao, Wei Yufei, Xu Jiadong, Jia Wenhao, Li Liang, Xiong Rong, Wang Yue  

**Link**: [PDF](https://arxiv.org/pdf/2503.18366)  

**Abstract**: Automatic parameter tuning methods for planning algorithms, which integrate pipeline approaches with learning-based techniques, are regarded as promising due to their stability and capability to handle highly constrained environments. While existing parameter tuning methods have demonstrated considerable success, further performance improvements require a more structured approach. In this paper, we propose a hierarchical architecture for reinforcement learning-based parameter tuning. The architecture introduces a hierarchical structure with low-frequency parameter tuning, mid-frequency planning, and high-frequency control, enabling concurrent enhancement of both upper-layer parameter tuning and lower-layer control through iterative training. Experimental evaluations in both simulated and real-world environments show that our method surpasses existing parameter tuning approaches. Furthermore, our approach achieves first place in the Benchmark for Autonomous Robot Navigation (BARN) Challenge. 

**Abstract (ZH)**: 基于强化学习的分层参数调优方法：通过迭代训练同时增强高层参数调优和低层控制 

---
# Vision-Guided Loco-Manipulation with a Snake Robot 

**Title (ZH)**: 基于视觉引导的蛇形机器人移动物体操作 

**Authors**: Adarsh Salagame, Sasank Potluri, Keshav Bharadwaj Vaidyanathan, Kruthika Gangaraju, Eric Sihite, Milad Ramezani, Alireza Ramezani  

**Link**: [PDF](https://arxiv.org/pdf/2503.18308)  

**Abstract**: This paper presents the development and integration of a vision-guided loco-manipulation pipeline for Northeastern University's snake robot, COBRA. The system leverages a YOLOv8-based object detection model and depth data from an onboard stereo camera to estimate the 6-DOF pose of target objects in real time. We introduce a framework for autonomous detection and control, enabling closed-loop loco-manipulation for transporting objects to specified goal locations. Additionally, we demonstrate open-loop experiments in which COBRA successfully performs real-time object detection and loco-manipulation tasks. 

**Abstract (ZH)**: 东北大学蛇形机器人COBRA的视觉引导移动操作管道的开发与集成：基于YOLOv8的目标检测模型与机载立体相机深度数据的六自由度姿态估计与自主检测控制框架 

---
# Ground Penetrating Radar-Assisted Multimodal Robot Odometry Using Subsurface Feature Matrix 

**Title (ZH)**: 基于Subsurface特征矩阵的地面穿透雷达辅助多模态机器人里程计 

**Authors**: Haifeng Li, Jiajun Guo, Xuanxin Fan, Dezhen Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.18301)  

**Abstract**: Localization of robots using subsurface features observed by ground-penetrating radar (GPR) enhances and adds robustness to common sensor modalities, as subsurface features are less affected by weather, seasons, and surface changes. We introduce an innovative multimodal odometry approach using inputs from GPR, an inertial measurement unit (IMU), and a wheel encoder. To efficiently address GPR signal noise, we introduce an advanced feature representation called the subsurface feature matrix (SFM). The SFM leverages frequency domain data and identifies peaks within radar scans. Additionally, we propose a novel feature matching method that estimates GPR displacement by aligning SFMs. The integrations from these three input sources are consolidated using a factor graph approach to achieve multimodal robot odometry. Our method has been developed and evaluated with the CMU-GPR public dataset, demonstrating improvements in accuracy and robustness with real-time performance in robotic odometry tasks. 

**Abstract (ZH)**: 利用地面穿透雷达（GPR）观测到的地下特征进行机器人定位增强了常见传感器模态的精度和鲁棒性，因为地下特征较少受天气、季节和表面变化的影响。我们介绍了一种创新的多模态里程计方法，使用来自GPR、惯性测量单元（IMU）和轮编码器的输入。为有效处理GPR信号噪声，我们提出了一种高级特征表示方法，称为地下特征矩阵（SFM）。SFM 利用频域数据并识别雷达扫描中的峰值。此外，我们提出了一种新颖的特征匹配方法，通过对齐SFM 来估计GPR位移。这些三种输入来源的整合通过因子图方法实现多模态机器人里程计。我们的方法已在CMU-GPR 公开数据集上进行开发和评估，展示了在机器人导航任务中实时性能下的精度和鲁棒性提高。 

---
# Learning Orientation Field for OSM-Guided Autonomous Navigation 

**Title (ZH)**: OSM引导下的学习方向场自主导航 

**Authors**: Yuming Huang, Wei Gao, Zhiyuan Zhang, Maani Ghaffari, Dezhen Song, Cheng-Zhong Xu, Hui Kong  

**Link**: [PDF](https://arxiv.org/pdf/2503.18276)  

**Abstract**: OpenStreetMap (OSM) has gained popularity recently in autonomous navigation due to its public accessibility, lower maintenance costs, and broader geographical coverage. However, existing methods often struggle with noisy OSM data and incomplete sensor observations, leading to inaccuracies in trajectory planning. These challenges are particularly evident in complex driving scenarios, such as at intersections or facing occlusions. To address these challenges, we propose a robust and explainable two-stage framework to learn an Orientation Field (OrField) for robot navigation by integrating LiDAR scans and OSM routes. In the first stage, we introduce the novel representation, OrField, which can provide orientations for each grid on the map, reasoning jointly from noisy LiDAR scans and OSM routes. To generate a robust OrField, we train a deep neural network by encoding a versatile initial OrField and output an optimized OrField. Based on OrField, we propose two trajectory planners for OSM-guided robot navigation, called Field-RRT* and Field-Bezier, respectively, in the second stage by improving the Rapidly Exploring Random Tree (RRT) algorithm and Bezier curve to estimate the trajectories. Thanks to the robustness of OrField which captures both global and local information, Field-RRT* and Field-Bezier can generate accurate and reliable trajectories even in challenging conditions. We validate our approach through experiments on the SemanticKITTI dataset and our own campus dataset. The results demonstrate the effectiveness of our method, achieving superior performance in complex and noisy conditions. Our code for network training and real-world deployment is available at this https URL. 

**Abstract (ZH)**: 基于LiDAR扫描和OSM路线的鲁棒可解释两阶段框架用于机器人导航 

---
# GI-SLAM: Gaussian-Inertial SLAM 

**Title (ZH)**: GI-SLAM: 高斯-惯性SLAM 

**Authors**: Xulang Liu, Ning Tan  

**Link**: [PDF](https://arxiv.org/pdf/2503.18275)  

**Abstract**: 3D Gaussian Splatting (3DGS) has recently emerged as a powerful representation of geometry and appearance for dense Simultaneous Localization and Mapping (SLAM). Through rapid, differentiable rasterization of 3D Gaussians, many 3DGS SLAM methods achieve near real-time rendering and accelerated training. However, these methods largely overlook inertial data, witch is a critical piece of information collected from the inertial measurement unit (IMU). In this paper, we present GI-SLAM, a novel gaussian-inertial SLAM system which consists of an IMU-enhanced camera tracking module and a realistic 3D Gaussian-based scene representation for mapping. Our method introduces an IMU loss that seamlessly integrates into the deep learning framework underpinning 3D Gaussian Splatting SLAM, effectively enhancing the accuracy, robustness and efficiency of camera tracking. Moreover, our SLAM system supports a wide range of sensor configurations, including monocular, stereo, and RGBD cameras, both with and without IMU integration. Our method achieves competitive performance compared with existing state-of-the-art real-time methods on the EuRoC and TUM-RGBD datasets. 

**Abstract (ZH)**: 基于惯性的3D高斯SLAM (GI-SLAM): 一种结合惯性数据的3D高斯映射系统 

---
# Decentralized Navigation of a Cable-Towed Load using Quadrupedal Robot Team via MARL 

**Title (ZH)**: 基于多-agent reinforcement学习的电缆牵引负载四足机器人团队分散导航 

**Authors**: Wen-Tse Chen, Minh Nguyen, Zhongyu Li, Guo Ning Sue, Koushil Sreenath  

**Link**: [PDF](https://arxiv.org/pdf/2503.18221)  

**Abstract**: This work addresses the challenge of enabling a team of quadrupedal robots to collaboratively tow a cable-connected load through cluttered and unstructured environments while avoiding obstacles. Leveraging cables allows the multi-robot system to navigate narrow spaces by maintaining slack when necessary. However, this introduces hybrid physical interactions due to alternating taut and slack states, with computational complexity that scales exponentially as the number of agents increases. To tackle these challenges, we developed a scalable and decentralized system capable of dynamically coordinating a variable number of quadrupedal robots while managing the hybrid physical interactions inherent in the load-towing task. At the core of this system is a novel multi-agent reinforcement learning (MARL)-based planner, designed for decentralized coordination. The MARL-based planner is trained using a centralized training with decentralized execution (CTDE) framework, enabling each robot to make decisions autonomously using only local (ego) observations. To accelerate learning and ensure effective collaboration across varying team sizes, we introduce a tailored training curriculum for MARL. Experimental results highlight the flexibility and scalability of the framework, demonstrating successful deployment with one to four robots in real-world scenarios and up to twelve robots in simulation. The decentralized planner maintains consistent inference times, regardless of the team size. Additionally, the proposed system demonstrates robustness to environment perturbations and adaptability to varying load weights. This work represents a step forward in achieving flexible and efficient multi-legged robotic collaboration in complex and real-world environments. 

**Abstract (ZH)**: 这项工作解决了让一组四足机器人在杂乱和未结构化的环境中协作拖曳电缆连接负载并避免障碍物的挑战。利用电缆允许多机器人系统通过在必要时保持松弛来导航狭窄空间，但这也引入了由于交替的紧绷和松弛状态而产生的混合物理交互，随着代理数量的增加，计算复杂性呈指数级增长。为了应对这些挑战，我们开发了一种可扩展且去中心化的系统，能够在管理负载拖曳任务中固有的混合物理交互的同时，动态协调数量可变的四足机器人。该系统的核心是一个专为去中心化协调设计的新型多代理强化学习（MARL）规划器。使用集中训练与分散执行（CTDE）框架训练基于MARL的规划器，使每个机器人仅使用局部（自我）观察就能自主做出决策。为了加快学习速度并确保在不同团队规模下有效的协作，我们引入了一种针对MARL的定制化训练课程。实验结果强调了该框架的灵活性和可扩展性，在实际场景中成功部署了一到四台机器人，并在模拟中最多使用了十二台机器人。去中心化规划器保持了不受团队规模影响的一致推理时间。此外，所提出的系统对环境扰动具有鲁棒性，并能适应不同的负载重量。这项工作代表了实现复杂和真实环境中灵活高效的多腿机器人协作的一个进步。 

---
# Extended Visibility of Autonomous Vehicles via Optimized Cooperative Perception under Imperfect Communication 

**Title (ZH)**: 基于 imperfect 通信的优化协同感知下自主车辆的扩展感知范围 

**Authors**: Ahmad Sarlak, Rahul Amin, Abolfazl Razi  

**Link**: [PDF](https://arxiv.org/pdf/2503.18192)  

**Abstract**: Autonomous Vehicles (AVs) rely on individual perception systems to navigate safely. However, these systems face significant challenges in adverse weather conditions, complex road geometries, and dense traffic scenarios. Cooperative Perception (CP) has emerged as a promising approach to extending the perception quality of AVs by jointly processing shared camera feeds and sensor readings across multiple vehicles. This work presents a novel CP framework designed to optimize vehicle selection and networking resource utilization under imperfect communications. Our optimized CP formation considers critical factors such as the helper vehicles' spatial position, visual range, motion blur, and available communication budgets. Furthermore, our resource optimization module allocates communication channels while adjusting power levels to maximize data flow efficiency between the ego and helper vehicles, considering realistic models of modern vehicular communication systems, such as LTE and 5G NR-V2X. We validate our approach through extensive experiments on pedestrian detection in challenging scenarios, using synthetic data generated by the CARLA simulator. The results demonstrate that our method significantly improves upon the perception quality of individual AVs with about 10% gain in detection accuracy. This substantial gain uncovers the unleashed potential of CP to enhance AV safety and performance in complex situations. 

**Abstract (ZH)**: 自主驾驶车辆（AVs）依赖于个体感知系统来确保安全导航。然而，在恶劣天气条件、复杂道路几何形状和密集交通场景下，这些系统面临着显著挑战。协作感知（CP）作为一种联合处理多辆车辆之间共享摄像头馈送和传感器读数的方法，已 emerged as a promising approach to extending the perception quality of AVs. 本研究提出了一个新颖的CP框架，旨在在不完美的通信条件下优化车辆选择和网络资源利用。我们在合作感知组织中考虑了助行车的空间位置、视距范围、运动模糊和可用的通信预算等关键因素。此外，我们的资源优化模块通过调整功率级别来分配通信信道，以最大限度地提高主车与助行车之间数据流的效率，同时考虑到现代车辆通信系统的现实模型，如LTE和5G NR-V2X。我们通过使用CARLA模拟器生成的合成数据，对在挑战性场景下的人行检测进行了广泛的实验验证。结果表明，我们的方法在检测准确性上提高了约10%，显著提升了单个AV的感知质量。这一重要增益揭示了CP在复杂情况下增强AV安全性和性能的巨大潜力。 

---
# Unraveling the Effects of Synthetic Data on End-to-End Autonomous Driving 

**Title (ZH)**: 解析合成数据对端到端自动驾驶的影响 

**Authors**: Junhao Ge, Zuhong Liu, Longteng Fan, Yifan Jiang, Jiaqi Su, Yiming Li, Zhejun Zhang, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.18108)  

**Abstract**: End-to-end (E2E) autonomous driving (AD) models require diverse, high-quality data to perform well across various driving scenarios. However, collecting large-scale real-world data is expensive and time-consuming, making high-fidelity synthetic data essential for enhancing data diversity and model robustness. Existing driving simulators for synthetic data generation have significant limitations: game-engine-based simulators struggle to produce realistic sensor data, while NeRF-based and diffusion-based methods face efficiency challenges. Additionally, recent simulators designed for closed-loop evaluation provide limited interaction with other vehicles, failing to simulate complex real-world traffic dynamics. To address these issues, we introduce SceneCrafter, a realistic, interactive, and efficient AD simulator based on 3D Gaussian Splatting (3DGS). SceneCrafter not only efficiently generates realistic driving logs across diverse traffic scenarios but also enables robust closed-loop evaluation of end-to-end models. Experimental results demonstrate that SceneCrafter serves as both a reliable evaluation platform and a efficient data generator that significantly improves end-to-end model generalization. 

**Abstract (ZH)**: 基于3D高斯点绘制的端到端自动驾驶实时互动模拟器：SceneCrafter 

---
# Optimizing Navigation And Chemical Application in Precision Agriculture With Deep Reinforcement Learning And Conditional Action Tree 

**Title (ZH)**: 基于深度强化学习和条件动作树的精准农业导航与化学应用优化 

**Authors**: Mahsa Khosravi, Zhanhong Jiang, Joshua R Waite, Sarah Jonesc, Hernan Torres, Arti Singh, Baskar Ganapathysubramanian, Asheesh Kumar Singh, Soumik Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2503.17985)  

**Abstract**: This paper presents a novel reinforcement learning (RL)-based planning scheme for optimized robotic management of biotic stresses in precision agriculture. The framework employs a hierarchical decision-making structure with conditional action masking, where high-level actions direct the robot's exploration, while low-level actions optimize its navigation and efficient chemical spraying in affected areas. The key objectives of optimization include improving the coverage of infected areas with limited battery power and reducing chemical usage, thus preventing unnecessary spraying of healthy areas of the field. Our numerical experimental results demonstrate that the proposed method, Hierarchical Action Masking Proximal Policy Optimization (HAM-PPO), significantly outperforms baseline practices, such as LawnMower navigation + indiscriminate spraying (Carpet Spray), in terms of yield recovery and resource efficiency. HAM-PPO consistently achieves higher yield recovery percentages and lower chemical costs across a range of infection scenarios. The framework also exhibits robustness to observation noise and generalizability under diverse environmental conditions, adapting to varying infection ranges and spatial distribution patterns. 

**Abstract (ZH)**: 一种基于强化学习的分级行动遮罩规划方案，用于精确农业中生物胁迫的优化机器人管理 

---
# GS-LTS: 3D Gaussian Splatting-Based Adaptive Modeling for Long-Term Service Robots 

**Title (ZH)**: 基于高斯点表示的自适应建模方法用于长周期服务机器人 

**Authors**: Bin Fu, Jialin Li, Bin Zhang, Ruiping Wang, Xilin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.17733)  

**Abstract**: 3D Gaussian Splatting (3DGS) has garnered significant attention in robotics for its explicit, high fidelity dense scene representation, demonstrating strong potential for robotic applications. However, 3DGS-based methods in robotics primarily focus on static scenes, with limited attention to the dynamic scene changes essential for long-term service robots. These robots demand sustained task execution and efficient scene updates-challenges current approaches fail to meet. To address these limitations, we propose GS-LTS (Gaussian Splatting for Long-Term Service), a 3DGS-based system enabling indoor robots to manage diverse tasks in dynamic environments over time. GS-LTS detects scene changes (e.g., object addition or removal) via single-image change detection, employs a rule-based policy to autonomously collect multi-view observations, and efficiently updates the scene representation through Gaussian editing. Additionally, we propose a simulation-based benchmark that automatically generates scene change data as compact configuration scripts, providing a standardized, user-friendly evaluation benchmark. Experimental results demonstrate GS-LTS's advantages in reconstruction, navigation, and superior scene updates-faster and higher quality than the image training baseline-advancing 3DGS for long-term robotic operations. Code and benchmark are available at: this https URL. 

**Abstract (ZH)**: 基于3D高斯斑点图（3DGS）的长期服务机器人三维场景动态管理方法 

---
# RAIDER: Tool-Equipped Large Language Model Agent for Robotic Action Issue Detection, Explanation and Recovery 

**Title (ZH)**: RAIDER: 工具有配备的大语言模型代理剂，用于机器人行动问题检测、解释与恢复 

**Authors**: Silvia Izquierdo-Badiola, Carlos Rizzo, Guillem Alenyà  

**Link**: [PDF](https://arxiv.org/pdf/2503.17703)  

**Abstract**: As robots increasingly operate in dynamic human-centric environments, improving their ability to detect, explain, and recover from action-related issues becomes crucial. Traditional model-based and data-driven techniques lack adaptability, while more flexible generative AI methods struggle with grounding extracted information to real-world constraints. We introduce RAIDER, a novel agent that integrates Large Language Models (LLMs) with grounded tools for adaptable and efficient issue detection and explanation. Using a unique "Ground, Ask& Answer, Issue" procedure, RAIDER dynamically generates context-aware precondition questions and selects appropriate tools for resolution, achieving targeted information gathering. Our results within a simulated household environment surpass methods relying on predefined models, full scene descriptions, or standalone trained models. Additionally, RAIDER's explanations enhance recovery success, including cases requiring human interaction. Its modular architecture, featuring self-correction mechanisms, enables straightforward adaptation to diverse scenarios, as demonstrated in a real-world human-assistive task. This showcases RAIDER's potential as a versatile agentic AI solution for robotic issue detection and explanation, while addressing the problem of grounding generative AI for its effective application in embodied agents. Project website: this https URL 

**Abstract (ZH)**: 随着机器人在以人类为中心的动态环境中操作越来越频繁，提高其检测、解释和从与行动相关的问题中恢复的能力变得至关重要。传统的基于模型和数据驱动的方法缺乏适应性，而更具灵活性的生成AI方法则难以将提取的信息与现实世界的约束相结合。我们引入了RAIDER，这是一种将大型语言模型（LLMs）与接地工具集成的新型代理，以实现适应性和高效的问题检测与解释。通过一种独特的“接地、询问与回答、问题”程序，RAIDER动态生成上下文感知的先置条件问题，并选择适当的工具进行解决，从而实现有针对性的信息收集。我们的模拟家庭环境中的结果优于依赖预定义模型、完整场景描述或独立训练模型的方法。此外，RAIDER的解释提高了恢复成功率，包括需要人类交互的情况。其模块化架构，包含自我修正机制，使其能够轻松适应各种场景，正如在实际的人辅助任务中的演示所示。这展示了RAIDER作为机器人问题检测与解释的多用途代理AI解决方案的潜力，同时也解决了生成AI在实际体态代理中有效应用时的接地问题。项目网站：this https URL。 

---
# Computationally and Sample Efficient Safe Reinforcement Learning Using Adaptive Conformal Prediction 

**Title (ZH)**: 使用自适应一致预测的计算高效且样本高效的安全强化学习 

**Authors**: Hao Zhou, Yanze Zhang, Wenhao Luo  

**Link**: [PDF](https://arxiv.org/pdf/2503.17678)  

**Abstract**: Safety is a critical concern in learning-enabled autonomous systems especially when deploying these systems in real-world scenarios. An important challenge is accurately quantifying the uncertainty of unknown models to generate provably safe control policies that facilitate the gathering of informative data, thereby achieving both safe and optimal policies. Additionally, the selection of the data-driven model can significantly impact both the real-time implementation and the uncertainty quantification process. In this paper, we propose a provably sample efficient episodic safe learning framework that remains robust across various model choices with quantified uncertainty for online control tasks. Specifically, we first employ Quadrature Fourier Features (QFF) for kernel function approximation of Gaussian Processes (GPs) to enable efficient approximation of unknown dynamics. Then the Adaptive Conformal Prediction (ACP) is used to quantify the uncertainty from online observations and combined with the Control Barrier Functions (CBF) to characterize the uncertainty-aware safe control constraints under learned dynamics. Finally, an optimism-based exploration strategy is integrated with ACP-based CBFs for safe exploration and near-optimal safe nonlinear control. Theoretical proofs and simulation results are provided to demonstrate the effectiveness and efficiency of the proposed framework. 

**Abstract (ZH)**: 学习驱动的自主系统在实际应用场景中，安全性是一个关键问题。一个重要的挑战是准确量化未知模型的不确定性，以生成可以证明的安全控制策略，这些策略既能收集有价值的数据，又能实现安全和最优的控制。此外，数据驱动模型的选择对实时实施和不确定性量化过程都有显著影响。在本文中，我们提出了一种 Provably 样本高效的经验安全学习框架，该框架在各种模型选择下具有量化不确定性，适用于在线控制任务的鲁棒性。具体而言，我们首先采用 Quadrature Fourier Features (QFF) 对高斯过程 (GPs) 的核函数进行逼近，以实现未知动力学的有效逼近。然后使用自适应置信预测 (ACP) 来量化在线观测中的不确定性，并结合控制障碍函数 (CBF) 来表征基于学习动态的不确定性感知安全控制约束。最后，我们将基于 ACP 的 CBF 与基于乐观探索策略结合，实现安全探索和接近最优的安全非线性控制。提供的理论证明和仿真结果表明了所提出框架的有效性和效率。 

---
# Transferable Latent-to-Latent Locomotion Policy for Efficient and Versatile Motion Control of Diverse Legged Robots 

**Title (ZH)**: 可迁移的潜空间到潜空间运动政策：用于多样化-legged机器人高效且多功能的运动控制 

**Authors**: Ziang Zheng, Guojian Zhan, Bin Shuai, Shengtao Qin, Jiangtao Li, Tao Zhang, Shengbo Eben Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.17626)  

**Abstract**: Reinforcement learning (RL) has demonstrated remarkable capability in acquiring robot skills, but learning each new skill still requires substantial data collection for training. The pretrain-and-finetune paradigm offers a promising approach for efficiently adapting to new robot entities and tasks. Inspired by the idea that acquired knowledge can accelerate learning new tasks with the same robot and help a new robot master a trained task, we propose a latent training framework where a transferable latent-to-latent locomotion policy is pretrained alongside diverse task-specific observation encoders and action decoders. This policy in latent space processes encoded latent observations to generate latent actions to be decoded, with the potential to learn general abstract motion skills. To retain essential information for decision-making and control, we introduce a diffusion recovery module that minimizes information reconstruction loss during pretrain stage. During fine-tune stage, the pretrained latent-to-latent locomotion policy remains fixed, while only the lightweight task-specific encoder and decoder are optimized for efficient adaptation. Our method allows a robot to leverage its own prior experience across different tasks as well as the experience of other morphologically diverse robots to accelerate adaptation. We validate our approach through extensive simulations and real-world experiments, demonstrating that the pretrained latent-to-latent locomotion policy effectively generalizes to new robot entities and tasks with improved efficiency. 

**Abstract (ZH)**: 强化学习（RL）在获取机器人技能方面展示了非凡的能力，但学习每个新技能仍然需要大量数据进行训练。预训练和微调范式为高效适应新机器人实体和任务提供了有前景的方法。受获取的知识可以加快使用同一机器人学习新任务并帮助新的机器人掌握已训练任务的想法启发，我们提出了一种潜在训练框架，其中可迁移的潜在到潜在运动策略与多种任务特定观察编码器和动作解码器一起进行预训练。该策略在潜在空间中处理编码的潜在观察以生成潜在动作，具有学习通用抽象运动技能的潜力。为保留决策和控制所需的必要信息，我们引入了一种扩散恢复模块，在预训练阶段最小化信息重构损失。在微调阶段，预训练的潜在到潜在运动策略保持固定，仅对轻量级的任务特定编码器和解码器进行优化，以实现高效的适应。该方法允许机器人利用其在不同任务中的先验经验以及不同形态的其他机器人的经验来加速适应。我们通过广泛的仿真和实际实验验证了该方法，证明了预训练的潜在到潜在运动策略可以高效地泛化到新的机器人实体和任务中。 

---
# LLM-Drone: Aerial Additive Manufacturing with Drones Planned Using Large Language Models 

**Title (ZH)**: 基于大型语言模型规划的无人机增材制造：LLM-Drone 

**Authors**: Akshay Raman, Chad Merrill, Abraham George, Amir Barati Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2503.17566)  

**Abstract**: Additive manufacturing (AM) has transformed the production landscape by enabling the precision creation of complex geometries. However, AM faces limitations when applied to challenging environments, such as elevated surfaces and remote locations. Aerial additive manufacturing, facilitated by drones, presents a solution to these challenges. However, despite advances in methods for the planning, control, and localization of drones, the accuracy of these methods is insufficient to run traditional feedforward extrusion-based additive manufacturing processes (such as Fused Deposition Manufacturing). Recently, the emergence of LLMs has revolutionized various fields by introducing advanced semantic reasoning and real-time planning capabilities. This paper proposes the integration of LLMs with aerial additive manufacturing to assist with the planning and execution of construction tasks, granting greater flexibility and enabling a feed-back based design and construction system. Using the semantic understanding and adaptability of LLMs, we can overcome the limitations of drone based systems by dynamically generating and adapting building plans on site, ensuring efficient and accurate construction even in constrained environments. Our system is able to design and build structures given only a semantic prompt and has shown success in understanding the spatial environment despite tight planning constraints. Our method's feedback system enables replanning using the LLM if the manufacturing process encounters unforeseen errors, without requiring complicated heuristics or evaluation functions. Combining the semantic planning with automatic error correction, our system achieved a 90% build accuracy, converting simple text prompts to build structures. 

**Abstract (ZH)**: 基于无人机的增量制造：通过大规模语言模型实现精确规划与执行 

---
# Aether: Geometric-Aware Unified World Modeling 

**Title (ZH)**: 以太：几何 Awareness 统一世界建模 

**Authors**: Aether Team, Haoyi Zhu, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Chunhua Shen, Jiangmiao Pang, Tong He  

**Link**: [PDF](https://arxiv.org/pdf/2503.18945)  

**Abstract**: The integration of geometric reconstruction and generative modeling remains a critical challenge in developing AI systems capable of human-like spatial reasoning. This paper proposes Aether, a unified framework that enables geometry-aware reasoning in world models by jointly optimizing three core capabilities: (1) 4D dynamic reconstruction, (2) action-conditioned video prediction, and (3) goal-conditioned visual planning. Through task-interleaved feature learning, Aether achieves synergistic knowledge sharing across reconstruction, prediction, and planning objectives. Building upon video generation models, our framework demonstrates unprecedented synthetic-to-real generalization despite never observing real-world data during training. Furthermore, our approach achieves zero-shot generalization in both action following and reconstruction tasks, thanks to its intrinsic geometric modeling. Remarkably, even without real-world data, its reconstruction performance far exceeds that of domain-specific models. Additionally, Aether leverages a geometry-informed action space to seamlessly translate predictions into actions, enabling effective autonomous trajectory planning. We hope our work inspires the community to explore new frontiers in physically-reasonable world modeling and its applications. 

**Abstract (ZH)**: 几何重建与生成建模的集成在开发具备人类空间推理能力的AI系统中仍是一项关键挑战。本文提出Aether，这是一种统一框架，通过联合优化三项核心能力来实现世界模型中的几何感知推理：（1）4D动态重建，（2）动作条件下的视频预测，以及（3）目标条件下的视觉规划。通过任务交错特征学习，Aether 实现了重建、预测和规划目标间的协同知识共享。在基于视频生成模型的基础上，我们的框架在从未见过真实世界数据的情况下，展示了前所未有的合成到现实的泛化能力。此外，由于其内在的几何建模特性，我们的方法在动作跟随和重建任务中实现了零样本泛化。令人惊讶的是，即使没有真实世界数据，其重建性能也远超领域特定模型。同时，Aether 利用几何启发的动作空间，无缝地将预测转化为行动，支持有效的自主轨迹规划。我们希望我们的工作能够激发社区探索物理合理的世界建模及其应用的全新领域。 

---
# AdaWorld: Learning Adaptable World Models with Latent Actions 

**Title (ZH)**: AdaWorld: 学习具有潜在动作的世界模型 

**Authors**: Shenyuan Gao, Siyuan Zhou, Yilun Du, Jun Zhang, Chuang Gan  

**Link**: [PDF](https://arxiv.org/pdf/2503.18938)  

**Abstract**: World models aim to learn action-controlled prediction models and have proven essential for the development of intelligent agents. However, most existing world models rely heavily on substantial action-labeled data and costly training, making it challenging to adapt to novel environments with heterogeneous actions through limited interactions. This limitation can hinder their applicability across broader domains. To overcome this challenge, we propose AdaWorld, an innovative world model learning approach that enables efficient adaptation. The key idea is to incorporate action information during the pretraining of world models. This is achieved by extracting latent actions from videos in a self-supervised manner, capturing the most critical transitions between frames. We then develop an autoregressive world model that conditions on these latent actions. This learning paradigm enables highly adaptable world models, facilitating efficient transfer and learning of new actions even with limited interactions and finetuning. Our comprehensive experiments across multiple environments demonstrate that AdaWorld achieves superior performance in both simulation quality and visual planning. 

**Abstract (ZH)**: AdaWorld: An Action-Driven World Model for Efficient Adaptation 

---
# Bootstrapped Model Predictive Control 

**Title (ZH)**: 基于自助模型预测控制 

**Authors**: Yuhang Wang, Hanwei Guo, Sizhe Wang, Long Qian, Xuguang Lan  

**Link**: [PDF](https://arxiv.org/pdf/2503.18871)  

**Abstract**: Model Predictive Control (MPC) has been demonstrated to be effective in continuous control tasks. When a world model and a value function are available, planning a sequence of actions ahead of time leads to a better policy. Existing methods typically obtain the value function and the corresponding policy in a model-free manner. However, we find that such an approach struggles with complex tasks, resulting in poor policy learning and inaccurate value estimation. To address this problem, we leverage the strengths of MPC itself. In this work, we introduce Bootstrapped Model Predictive Control (BMPC), a novel algorithm that performs policy learning in a bootstrapped manner. BMPC learns a network policy by imitating an MPC expert, and in turn, uses this policy to guide the MPC process. Combined with model-based TD-learning, our policy learning yields better value estimation and further boosts the efficiency of MPC. We also introduce a lazy reanalyze mechanism, which enables computationally efficient imitation learning. Our method achieves superior performance over prior works on diverse continuous control tasks. In particular, on challenging high-dimensional locomotion tasks, BMPC significantly improves data efficiency while also enhancing asymptotic performance and training stability, with comparable training time and smaller network sizes. Code is available at this https URL. 

**Abstract (ZH)**: Bootstraped Model Predictive Control (BMPC): A Bootstrapped Approach for Policy Learning in Continuous Control Tasks 

---
# A Robot-Led Intervention for Emotion Regulation: From Expression to Reappraisal 

**Title (ZH)**: 机器人引导的情绪调节干预：从表达到重评 

**Authors**: Guy Laban, Julie Wang, Hatice Gunes  

**Link**: [PDF](https://arxiv.org/pdf/2503.18243)  

**Abstract**: Emotion regulation is a crucial skill for managing emotions in everyday life, yet finding a constructive and accessible method to support these processes remains challenging due to their cognitive demands. In this study, we explore how regular interactions with a social robot, conducted in a structured yet familiar environment within university halls and departments, can provide effective support for emotion regulation through cognitive reappraisal. Twenty-one students participated in a five-session study at a university hall or department, where the robot facilitated structured conversations, encouraging the students to reinterpret emotionally charged situations that they shared with the robot. Quantitative and qualitative results indicate significant improvements in emotion self-regulation, with participants reporting better understanding and control of their emotions. The intervention led to significant changes in constructive emotion regulation tendencies and positive effects on mood and sentiment after each session. The findings also demonstrate that repeated interactions with the robot encouraged greater emotional expressiveness, including longer speech disclosures, increased use of affective language, and heightened facial arousal. Notably, expressiveness followed structured patterns aligned with the reappraisal process, with expression peaking during key reappraisal moments, particularly when participants were prompted to reinterpret negative experiences. The qualitative feedback further highlighted how the robot fostered introspection and provided a supportive space for discussing emotions, enabling participants to confront long-avoided emotional challenges. These findings demonstrate the potential of robots to effectively assist in emotion regulation in familiar environments, offering both emotional support and cognitive guidance. 

**Abstract (ZH)**: 社交机器人在大学教室和系部结构化环境中通过认知重评提供情绪调节的有效支持 

---
# PhysTwin: Physics-Informed Reconstruction and Simulation of Deformable Objects from Videos 

**Title (ZH)**: PhysTwin: 带有物理约束的可变形物体视频重构与模拟 

**Authors**: Hanxiao Jiang, Hao-Yu Hsu, Kaifeng Zhang, Hsin-Ni Yu, Shenlong Wang, Yunzhu Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.17973)  

**Abstract**: Creating a physical digital twin of a real-world object has immense potential in robotics, content creation, and XR. In this paper, we present PhysTwin, a novel framework that uses sparse videos of dynamic objects under interaction to produce a photo- and physically realistic, real-time interactive virtual replica. Our approach centers on two key components: (1) a physics-informed representation that combines spring-mass models for realistic physical simulation, generative shape models for geometry, and Gaussian splats for rendering; and (2) a novel multi-stage, optimization-based inverse modeling framework that reconstructs complete geometry, infers dense physical properties, and replicates realistic appearance from videos. Our method integrates an inverse physics framework with visual perception cues, enabling high-fidelity reconstruction even from partial, occluded, and limited viewpoints. PhysTwin supports modeling various deformable objects, including ropes, stuffed animals, cloth, and delivery packages. Experiments show that PhysTwin outperforms competing methods in reconstruction, rendering, future prediction, and simulation under novel interactions. We further demonstrate its applications in interactive real-time simulation and model-based robotic motion planning. 

**Abstract (ZH)**: 创建现实世界物体的物理数字孪生在机器人学、内容创作和XR领域具有巨大的潜力。本文我们提出PhysTwin，一种新颖的框架，利用交互下动态对象的稀疏视频生成高照片和物理真实感的实时互动虚拟复制品。该方法的核心在于两个关键组件：（1）一种物理信息驱动的表示，结合弹簧质量模型进行真实的物理模拟、生成形状模型进行几何建模以及高斯斑点进行渲染；（2）一种新颖的多阶段优化反向建模框架，用于重建完整的几何结构、推断密集的物理属性以及从视频中复制真实的外观。该方法将物理反向框架与视觉感知线索结合起来，能够在部分、被遮挡和有限视角的情况下实现高保真重建。PhysTwin 支持建模各种可变形对象，包括绳索、填充动物玩具、布料和送货包裹。实验表明，PhysTwin 在重建、渲染、未来预测和新交互下的模拟方面优于竞争对手的方法。我们进一步展示了其在实时互动模拟和基于模型的机器人运动规划中的应用。 

---
# Adaptive Koopman Model Predictive Control of Simple Serial Robots 

**Title (ZH)**: 简单串联机器人基于Koopman模型预测控制的自适应控制 

**Authors**: Adriano del Río, Christoph Stoeffler  

**Link**: [PDF](https://arxiv.org/pdf/2503.17902)  

**Abstract**: Approximating nonlinear systems as linear ones is a common workaround to apply control tools tailored for linear systems. This motivates our present work where we developed a data-driven model predictive controller (MPC) based on the Koopman operator framework, allowing the embedding of nonlinear dynamics in a higher dimensional, but linear function space. The controller, termed adaptive Koopman model predictive control (KMPC), uses online closed-loop feedback to learn and incrementally update a linear representation of nonlinear system dynamics, without the prior knowledge of a model. Adaptive KMPC differs from most other Koopman-based control frameworks that aim to identify high-validity-range models in advance and then enter closed-loop control without further model adaptations. To validate the controller, trajectory tracking experiments are conducted with 1R and 2R robots under force disturbances and changing model parameters. We compare the controller to classical linearization MPC and Koopman-based MPC without model updates, denoted static KMPC. The results show that adaptive KMPC can, opposed to static KMPC, generalize over unforeseen force disturbances and can, opposed to linearization MPC, handle varying dynamic parameters, while using a small set of basis functions to approximate the Koopman operator. 

**Abstract (ZH)**: 基于Koopman算子框架的自适应Koopman模型预测控制 

---
# The case for delegated AI autonomy for Human AI teaming in healthcare 

**Title (ZH)**: 为医疗健康领域的人机协作委托人工智能自主权辩护 

**Authors**: Yan Jia, Harriet Evans, Zoe Porter, Simon Graham, John McDermid, Tom Lawton, David Snead, Ibrahim Habli  

**Link**: [PDF](https://arxiv.org/pdf/2503.18778)  

**Abstract**: In this paper we propose an advanced approach to integrating artificial intelligence (AI) into healthcare: autonomous decision support. This approach allows the AI algorithm to act autonomously for a subset of patient cases whilst serving a supportive role in other subsets of patient cases based on defined delegation criteria. By leveraging the complementary strengths of both humans and AI, it aims to deliver greater overall performance than existing human-AI teaming models. It ensures safe handling of patient cases and potentially reduces clinician review time, whilst being mindful of AI tool limitations. After setting the approach within the context of current human-AI teaming models, we outline the delegation criteria and apply them to a specific AI-based tool used in histopathology. The potential impact of the approach and the regulatory requirements for its successful implementation are then discussed. 

**Abstract (ZH)**: 在本文中，我们提出了一种将人工智能（AI）先进地整合到医疗保健中的方法：自主决策支持。该方法允许AI算法在一组患者案例中自主行动，而在其他患者案例中则基于定义的授权标准发挥支持作用。通过利用人类和AI的互补优势，该方法旨在比现有的人机团队模式提供更好的整体性能。该方法确保安全处理患者案例，并可能减少 clinicians 的审阅时间，同时考虑到AI工具的局限性。在将该方法置于当前人机团队模式的背景下之后，我们阐述了授权标准，并将其应用于组织病理学中的一种特定AI工具。随后讨论了该方法的影响及其成功实施所需的监管要求。 

---
# Metacognition in Content-Centric Computational Cognitive C4 Modeling 

**Title (ZH)**: 内容为中心的元认知计算认知C4建模 

**Authors**: Sergei Nirenburg, Marjorie McShane, Sanjay Oruganti  

**Link**: [PDF](https://arxiv.org/pdf/2503.17822)  

**Abstract**: For AI agents to emulate human behavior, they must be able to perceive, meaningfully interpret, store, and use large amounts of information about the world, themselves, and other agents. Metacognition is a necessary component of all of these processes. In this paper, we briefly a) introduce content-centric computational cognitive (C4) modeling for next-generation AI agents; b) review the long history of developing C4 agents at RPI's LEIA (Language-Endowed Intelligent Agents) Lab; c) discuss our current work on extending LEIAs' cognitive capabilities to cognitive robotic applications developed using a neuro symbolic processing model; and d) sketch plans for future developments in this paradigm that aim to overcome underappreciated limitations of currently popular, LLM-driven methods in AI. 

**Abstract (ZH)**: 为了使AI代理仿真人行为，它们必须能够感知、有意义地解释、存储和使用大量关于世界、自身和其他代理的信息。元认知是所有这些过程的必要组成部分。本文简要介绍了a)下一代AI代理的内容中心计算认知（C4）建模；b) RPI的LEIA（语言赋能智能代理）实验室在发展C4代理方面的长期历史；c) 利用神经符号处理模型扩展LEIA的认知能力以应用于认知机器人应用的研究现状；以及d) 计划在这一范式中未来的发展，旨在克服当前流行的由大语言模型驱动的方法尚未充分认识到的局限性。 

---
# MEPNet: Medical Entity-balanced Prompting Network for Brain CT Report Generation 

**Title (ZH)**: MEPNet：医学实体平衡提示网络用于脑CT报告生成 

**Authors**: Xiaodan Zhang, Yanzhao Shi, Junzhong Ji, Chengxin Zheng, Liangqiong Qu  

**Link**: [PDF](https://arxiv.org/pdf/2503.17784)  

**Abstract**: The automatic generation of brain CT reports has gained widespread attention, given its potential to assist radiologists in diagnosing cranial diseases. However, brain CT scans involve extensive medical entities, such as diverse anatomy regions and lesions, exhibiting highly inconsistent spatial patterns in 3D volumetric space. This leads to biased learning of medical entities in existing methods, resulting in repetitiveness and inaccuracy in generated reports. To this end, we propose a Medical Entity-balanced Prompting Network (MEPNet), which harnesses the large language model (LLM) to fairly interpret various entities for accurate brain CT report generation. By introducing the visual embedding and the learning status of medical entities as enriched clues, our method prompts the LLM to balance the learning of diverse entities, thereby enhancing reports with comprehensive findings. First, to extract visual embedding of entities, we propose Knowledge-driven Joint Attention to explore and distill entity patterns using both explicit and implicit medical knowledge. Then, a Learning Status Scorer is designed to evaluate the learning of entity visual embeddings, resulting in unique learning status for individual entities. Finally, these entity visual embeddings and status are elaborately integrated into multi-modal prompts, to guide the text generation of LLM. This process allows LLM to self-adapt the learning process for biased-fitted entities, thereby covering detailed findings in generated reports. We conduct experiments on two brain CT report generation benchmarks, showing the effectiveness in clinical accuracy and text coherence. 

**Abstract (ZH)**: 基于医学实体平衡的脑CT报告自动生成网络（MEPNet）：利用大型语言模型公平解读医学实体以生成准确的脑CT报告 

---
# Adventurer: Exploration with BiGAN for Deep Reinforcement Learning 

**Title (ZH)**: Adventurer: 使用BiGAN进行深度强化学习的探索 

**Authors**: Yongshuai Liu, Xin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.18612)  

**Abstract**: Recent developments in deep reinforcement learning have been very successful in learning complex, previously intractable problems. Sample efficiency and local optimality, however, remain significant challenges. To address these challenges, novelty-driven exploration strategies have emerged and shown promising potential. Unfortunately, no single algorithm outperforms all others in all tasks and most of them struggle with tasks with high-dimensional and complex observations. In this work, we propose Adventurer, a novelty-driven exploration algorithm that is based on Bidirectional Generative Adversarial Networks (BiGAN), where BiGAN is trained to estimate state novelty. Intuitively, a generator that has been trained on the distribution of visited states should only be able to generate a state coming from the distribution of visited states. As a result, novel states using the generator to reconstruct input states from certain latent representations would lead to larger reconstruction errors. We show that BiGAN performs well in estimating state novelty for complex observations. This novelty estimation method can be combined with intrinsic-reward-based exploration. Our empirical results show that Adventurer produces competitive results on a range of popular benchmark tasks, including continuous robotic manipulation tasks (e.g. Mujoco robotics) and high-dimensional image-based tasks (e.g. Atari games). 

**Abstract (ZH)**: Recent developments in deep reinforcement learning have been very successful in learning complex, previously intractable problems. Sample efficiency and local optimality, however, remain significant challenges. To address these challenges, novelty-driven exploration strategies have emerged and shown promising potential. Unfortunately, no single algorithm outperforms all others in all tasks and most of them struggle with tasks with high-dimensional and complex observations. In this work, we propose Adventurer, a novelty-driven exploration algorithm based on Bidirectional Generative Adversarial Networks (BiGAN), where BiGAN is trained to estimate state novelty. 

---
# Galaxy Walker: Geometry-aware VLMs For Galaxy-scale Understanding 

**Title (ZH)**: 银河探索者：几何感知的大视图语言模型在银河规模理解中的应用 

**Authors**: Tianyu Chen, Xingcheng Fu, Yisen Gao, Haodong Qian, Yuecen Wei, Kun Yan, Haoyi Zhou, Jianxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.18578)  

**Abstract**: Modern vision-language models (VLMs) develop patch embedding and convolution backbone within vector space, especially Euclidean ones, at the very founding. When expanding VLMs to a galaxy scale for understanding astronomical phenomena, the integration of spherical space for planetary orbits and hyperbolic spaces for black holes raises two formidable challenges. a) The current pre-training model is confined to Euclidean space rather than a comprehensive geometric embedding. b) The predominant architecture lacks suitable backbones for anisotropic physical geometries. In this paper, we introduced Galaxy-Walker, a geometry-aware VLM, for the universe-level vision understanding tasks. We proposed the geometry prompt that generates geometry tokens by random walks across diverse spaces on a multi-scale physical graph, along with a geometry adapter that compresses and reshapes the space anisotropy in a mixture-of-experts manner. Extensive experiments demonstrate the effectiveness of our approach, with Galaxy-Walker achieving state-of-the-art performance in both galaxy property estimation ($R^2$ scores up to $0.91$) and morphology classification tasks (up to $+0.17$ F1 improvement in challenging features), significantly outperforming both domain-specific models and general-purpose VLMs. 

**Abstract (ZH)**: 现代视觉-语言模型在构建patches嵌入和卷积骨干时主要局限于欧几里得向量空间。当将视觉-语言模型扩展到银河系尺度以理解天文学现象时，行星轨道的球面空间和黑洞的双曲空间集成带来了两大挑战。a) 当前的预训练模型局限于欧几里得空间，而非全面的几何嵌入。b) 主要的架构缺乏适合各向异性物理几何结构的骨干网络。本文介绍了宇宙级视觉理解任务中的几何感知视觉-语言模型Galaxy-Walker。我们提出了几何提示，通过在多尺度物理图上进行随机漫步生成几何标记，并提出了一种几何适配器，以混合专家的方式压缩和重塑空间的各向异性。大量实验表明了该方法的有效性，Galaxy-Walker在星系属性估计（$R^2$分数最高可达0.91）和形态分类任务（在挑战性特征上F1分数提高0.17）中均取得了最佳性能，显著优于专门领域模型和通用视觉-语言模型。 

---
# Adaptive Multi-Fidelity Reinforcement Learning for Variance Reduction in Engineering Design Optimization 

**Title (ZH)**: 工程设计优化中基于方差减少的自适应多保真强化学习 

**Authors**: Akash Agrawal, Christopher McComb  

**Link**: [PDF](https://arxiv.org/pdf/2503.18229)  

**Abstract**: Multi-fidelity Reinforcement Learning (RL) frameworks efficiently utilize computational resources by integrating analysis models of varying accuracy and costs. The prevailing methodologies, characterized by transfer learning, human-inspired strategies, control variate techniques, and adaptive sampling, predominantly depend on a structured hierarchy of models. However, this reliance on a model hierarchy can exacerbate variance in policy learning when the underlying models exhibit heterogeneous error distributions across the design space. To address this challenge, this work proposes a novel adaptive multi-fidelity RL framework, in which multiple heterogeneous, non-hierarchical low-fidelity models are dynamically leveraged alongside a high-fidelity model to efficiently learn a high-fidelity policy. Specifically, low-fidelity policies and their experience data are adaptively used for efficient targeted learning, guided by their alignment with the high-fidelity policy. The effectiveness of the approach is demonstrated in an octocopter design optimization problem, utilizing two low-fidelity models alongside a high-fidelity simulator. The results demonstrate that the proposed approach substantially reduces variance in policy learning, leading to improved convergence and consistent high-quality solutions relative to traditional hierarchical multi-fidelity RL methods. Moreover, the framework eliminates the need for manually tuning model usage schedules, which can otherwise introduce significant computational overhead. This positions the framework as an effective variance-reduction strategy for multi-fidelity RL, while also mitigating the computational and operational burden of manual fidelity scheduling. 

**Abstract (ZH)**: 多保真度强化学习框架：通过动态利用非层次异质低保真度模型与高保真度模型高效学习高保真度策略 

---
# Unseen from Seen: Rewriting Observation-Instruction Using Foundation Models for Augmenting Vision-Language Navigation 

**Title (ZH)**: 未见之于所见：利用基础模型重写观察指令以增强视觉语言导航 

**Authors**: Ziming Wei, Bingqian Lin, Yunshuang Nie, Jiaqi Chen, Shikui Ma, Hang Xu, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2503.18065)  

**Abstract**: Data scarcity is a long-standing challenge in the Vision-Language Navigation (VLN) field, which extremely hinders the generalization of agents to unseen environments. Previous works primarily rely on additional simulator data or web-collected images/videos to improve the generalization. However, the simulator environments still face limited diversity, and the web-collected data often requires extensive labor to remove the noise. In this paper, we propose a Rewriting-driven AugMentation (RAM) paradigm for VLN, which directly creates the unseen observation-instruction pairs via rewriting human-annotated training data. Benefiting from our rewriting mechanism, new observation-instruction can be obtained in both simulator-free and labor-saving manners to promote generalization. Specifically, we first introduce Object-Enriched Observation Rewriting, where we combine Vision-Language Models (VLMs) and Large Language Models (LLMs) to derive rewritten object-enriched scene descriptions, enabling observation synthesis with diverse objects and spatial layouts via Text-to-Image Generation Models (T2IMs). Then, we propose Observation-Contrast Instruction Rewriting, which generates observation-aligned rewritten instructions by requiring LLMs to reason the difference between original and new observations. We further develop a mixing-then-focusing training strategy with a random observation cropping scheme, effectively enhancing data distribution diversity while suppressing augmentation data noise during training. Experiments on both the discrete environments (R2R, REVERIE, and R4R datasets) and continuous environments (R2R-CE dataset) show the superior performance and impressive generalization ability of our method. Code is available at this https URL. 

**Abstract (ZH)**: 基于重写驱动的扩增 paradigm在视觉语言导航中的应用：直接通过重写人类标注训练数据生成未见过的观察-指令对 

---
# Aligning Foundation Model Priors and Diffusion-Based Hand Interactions for Occlusion-Resistant Two-Hand Reconstruction 

**Title (ZH)**: 面向 Occlusion-抵抗的双手重建：对齐基础模型先验与扩散机制的手部交互 

**Authors**: Gaoge Han, Yongkang Cheng, Zhe Chen, Shaoli Huang, Tongliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.17788)  

**Abstract**: Two-hand reconstruction from monocular images faces persistent challenges due to complex and dynamic hand postures and occlusions, causing significant difficulty in achieving plausible interaction alignment. Existing approaches struggle with such alignment issues, often resulting in misalignment and penetration artifacts. To tackle this, we propose a novel framework that attempts to precisely align hand poses and interactions by synergistically integrating foundation model-driven 2D priors with diffusion-based interaction refinement for occlusion-resistant two-hand reconstruction. First, we introduce a Fusion Alignment Encoder that learns to align fused multimodal priors keypoints, segmentation maps, and depth cues from foundation models during training. This provides robust structured guidance, further enabling efficient inference without foundation models at test time while maintaining high reconstruction accuracy. Second, we employ a two-hand diffusion model explicitly trained to transform interpenetrated poses into plausible, non-penetrated interactions, leveraging gradient-guided denoising to correct artifacts and ensure realistic spatial relations. Extensive evaluations demonstrate that our method achieves state-of-the-art performance on InterHand2.6M, FreiHAND, and HIC datasets, significantly advancing occlusion handling and interaction robustness. 

**Abstract (ZH)**: 单目图像中的双手重建由于复杂多变的手部姿态和遮挡持续面临挑战，导致实现可信赖的交互对齐显著困难。现有方法难以解决这些问题，常导致对齐不准和穿透伪影。为应对这一挑战，我们提出了一种新颖框架，通过将基础模型驱动的二维先验与基于扩散的交互精修结合，协同实现对抗遮挡的双手重建并精确对齐手部姿态和交互。首先，我们引入了一种融合对齐编码器，在训练过程中学习对融合的多模态先验关键点、分割图和深度提示进行对齐，从而提供稳健的结构指导，并在测试时无需基础模型就能高效进行推理，同时保持高重建精度。其次，我们采用了一种明确训练的双手扩散模型，专门用于将交错的手部姿态转化为可信的、无穿透的交互，利用梯度指导去噪来纠正伪影并确保真实的空间关系。广泛评估表明，我们的方法在InterHand2.6M、FreiHAND和HIC数据集上达到了最先进的性能，显著提升了遮挡处理能力和交互鲁棒性。 

---
# Lifelong Evolution of Swarms 

**Title (ZH)**: 终身演化群体 

**Authors**: Lorenzo Leuzzi, Simon Jones, Sabine Hauert, Davide Bacciu, Andrea Cossu  

**Link**: [PDF](https://arxiv.org/pdf/2503.17763)  

**Abstract**: Adapting to task changes without forgetting previous knowledge is a key skill for intelligent systems, and a crucial aspect of lifelong learning. Swarm controllers, however, are typically designed for specific tasks, lacking the ability to retain knowledge across changing tasks. Lifelong learning, on the other hand, focuses on individual agents with limited insights into the emergent abilities of a collective like a swarm. To address this gap, we introduce a lifelong evolutionary framework for swarms, where a population of swarm controllers is evolved in a dynamic environment that incrementally presents novel tasks. This requires evolution to find controllers that quickly adapt to new tasks while retaining knowledge of previous ones, as they may reappear in the future. We discover that the population inherently preserves information about previous tasks, and it can reuse it to foster adaptation and mitigate forgetting. In contrast, the top-performing individual for a given task catastrophically forgets previous tasks. To mitigate this phenomenon, we design a regularization process for the evolutionary algorithm, reducing forgetting in top-performing individuals. Evolving swarms in a lifelong fashion raises fundamental questions on the current state of deep lifelong learning and on the robustness of swarm controllers in dynamic environments. 

**Abstract (ZH)**: 适应任务变化而不遗忘先前知识是智能系统的一项关键技能，也是终身学习的一个重要方面。然而，群体控制器通常设计用于特定任务，缺乏跨任务保留知识的能力。相比之下，终身学习侧重于个体代理，对其群体如群组所展现出的新兴能力了解有限。为解决这一差距，我们引入了一个群组的终身进化框架，在这一框架中，群体中的群组控制器在动态环境中逐步呈现新的任务进行进化。这就要求进化找到既能快速适应新任务又能保留先前知识的控制器，因为这些知识可能会在未来重新出现。我们发现，群体本身会固有地保存关于先前任务的信息，并可以重新利用这些信息促进适应和减轻遗忘。相比之下，特定任务的最佳个体会灾难性地遗忘先前任务。为减轻这一现象，我们为进化算法设计了一个正则化过程，减少最佳个体的遗忘。以终身方式演化群组引发了当前深度终身学习状态和群组控制器在动态环境中的鲁棒性方面的基本问题。 

---
# On The Sample Complexity Bounds In Bilevel Reinforcement Learning 

**Title (ZH)**: bilevel reinforcement learning中的样本复杂度界研究 

**Authors**: Mudit Gaur, Amrit Singh Bedi, Raghu Pasupathu, Vaneet Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2503.17644)  

**Abstract**: Bilevel reinforcement learning (BRL) has emerged as a powerful mathematical framework for studying generative AI alignment and related problems. While several principled algorithmic frameworks have been proposed, key theoretical foundations, particularly those related to sample complexity, remain underexplored. Understanding and deriving tight sample complexity bounds are crucial for bridging the gap between theory and practice, guiding the development of more efficient algorithms. In this work, we present the first sample complexity result for BRL, achieving a bound of $\epsilon^{-4}$. This result extends to standard bilevel optimization problems, providing an interesting theoretical contribution with practical implications. To address the computational challenges associated with hypergradient estimation in bilevel optimization, we develop a first-order Hessian-free algorithm that does not rely on costly hypergradient computations. By leveraging matrix-free techniques and constrained optimization methods, our approach ensures scalability and practicality. Our findings pave the way for improved methods in AI alignment and other fields reliant on bilevel optimization. 

**Abstract (ZH)**: bilevel强化学习（BRL）已成为研究生成式AI对齐及相关问题的强大数学框架。虽然已经提出了若干规范化的算法框架，但关键的理论基础，尤其是样本复杂度方面，仍欠研究。理解并推导出紧凑的样本复杂度界对于弥合理论与实践之间的差距、指导更高效算法的发展至关重要。在本文中，我们首次为BRL提供了样本复杂度结果，达到了$\epsilon^{-4}$的界限。该结果扩展到了标准的 bilevel优化问题，提供了具有实际意义的理论贡献。为了应对bilevel优化中超梯度估计的计算挑战，我们开发了一种无需进行昂贵的超梯度计算的一阶Hessian-free算法。通过利用矩阵自由技术与约束优化方法，我们的方法确保了可扩展性和实用性。我们的发现为AI对齐方法以及其他依赖于bilevel优化的领域改进方法奠定了基础。 

---
# PRIMAL: Physically Reactive and Interactive Motor Model for Avatar Learning 

**Title (ZH)**: PRIMAL: 物理反应性和交互性的运动模型用于角色学习 

**Authors**: Yan Zhang, Yao Feng, Alpár Cseke, Nitin Saini, Nathan Bajandas, Nicolas Heron, Michael J. Black  

**Link**: [PDF](https://arxiv.org/pdf/2503.17544)  

**Abstract**: To build a motor system of the interactive avatar, it is essential to develop a generative motion model drives the body to move through 3D space in a perpetual, realistic, controllable, and responsive manner. Although motion generation has been extensively studied, most methods do not support ``embodied intelligence'' due to their offline setting, slow speed, limited motion lengths, or unnatural movements. To overcome these limitations, we propose PRIMAL, an autoregressive diffusion model that is learned with a two-stage paradigm, inspired by recent advances in foundation models. In the pretraining stage, the model learns motion dynamics from a large number of sub-second motion segments, providing ``motor primitives'' from which more complex motions are built. In the adaptation phase, we employ a ControlNet-like adaptor to fine-tune the motor control for semantic action generation and spatial target reaching. Experiments show that physics effects emerge from our training. Given a single-frame initial state, our model not only generates unbounded, realistic, and controllable motion, but also enables the avatar to be responsive to induced impulses in real time. In addition, we can effectively and efficiently adapt our base model to few-shot personalized actions and the task of spatial control. Evaluations show that our proposed method outperforms state-of-the-art baselines. We leverage the model to create a real-time character animation system in Unreal Engine that is highly responsive and natural. Code, models, and more results are available at: this https URL 

**Abstract (ZH)**: 构建互动虚拟角色的运动系统，需要开发一个生成运动模型，该模型能够以持久的、真实的、可控的和响应的方式驱动身体在三维空间中移动。尽管运动生成已经被广泛研究，但由于大多数方法受限于离线设置、速度慢、运动长度有限或动作不自然等问题，这些方法大多不支持“具有体能智能”的互动。为克服这些限制，我们提出了一种自回归扩散模型PRIMAL，该模型采用两阶段学习范式，受到基础模型近期进展的启发。在预训练阶段，模型从大量的亚秒级运动片段中学习运动动力学，提供“运动基元”，从中构建更复杂的动作。在适应阶段，我们采用类似于ControlNet的适配器来 fine-tune 运动控制，以实现语义动作生成和空间目标追踪。实验显示，我们的训练过程中自然现象得以涌现。给定初始单帧状态，我们的模型不仅能生成无界限的、真实的、可控制的运动，还能使虚拟角色实时响应诱导的外力。此外，我们能够有效且高效地将基础模型适配到少量示例的个性化动作以及空间控制任务。评估结果显示，我们提出的方法优于现有最先进的基线方法。我们利用该模型在Unreal Engine中建立了一个高响应性和自然性的实时角色动画系统。更多代码、模型和实验结果请参见：this https URL。 

---
# Debugging and Runtime Analysis of Neural Networks with VLMs (A Case Study) 

**Title (ZH)**: 使用VLMs进行神经网络的调试与运行时分析：一个案例研究 

**Authors**: Boyue Caroline Hu, Divya Gopinath, Corina S. Pasareanu, Nina Narodytska, Ravi Mangal, Susmit Jha  

**Link**: [PDF](https://arxiv.org/pdf/2503.17416)  

**Abstract**: Debugging of Deep Neural Networks (DNNs), particularly vision models, is very challenging due to the complex and opaque decision-making processes in these networks. In this paper, we explore multi-modal Vision-Language Models (VLMs), such as CLIP, to automatically interpret the opaque representation space of vision models using natural language. This in turn, enables a semantic analysis of model behavior using human-understandable concepts, without requiring costly human annotations. Key to our approach is the notion of semantic heatmap, that succinctly captures the statistical properties of DNNs in terms of the concepts discovered with the VLM and that are computed off-line using a held-out data set. We show the utility of semantic heatmaps for fault localization -- an essential step in debugging -- in vision models. Our proposed technique helps localize the fault in the network (encoder vs head) and also highlights the responsible high-level concepts, by leveraging novel differential heatmaps, which summarize the semantic differences between the correct and incorrect behaviour of the analyzed DNN. We further propose a lightweight runtime analysis to detect and filter-out defects at runtime, thus improving the reliability of the analyzed DNNs. The runtime analysis works by measuring and comparing the similarity between the heatmap computed for a new (unseen) input and the heatmaps computed a-priori for correct vs incorrect DNN behavior. We consider two types of defects: misclassifications and vulnerabilities to adversarial attacks. We demonstrate the debugging and runtime analysis on a case study involving a complex ResNet-based classifier trained on the RIVAL10 dataset. 

**Abstract (ZH)**: 基于多模态视觉语言模型的深度神经网络调试 

---
