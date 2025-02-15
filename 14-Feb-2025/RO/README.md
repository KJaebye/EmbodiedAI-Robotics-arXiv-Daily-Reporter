# DexTrack: Towards Generalizable Neural Tracking Control for Dexterous Manipulation from Human References 

**Title (ZH)**: DexTrack: 向着从人类参考中实现通用化灵巧 manipulation 控制的神经追踪控制 

**Authors**: Xueyi Liu, Jianibieke Adalibieke, Qianwei Han, Yuzhe Qin, Li Yi  

**Link**: [PDF](https://arxiv.org/pdf/2502.09614)  

**Abstract**: We address the challenge of developing a generalizable neural tracking controller for dexterous manipulation from human references. This controller aims to manage a dexterous robot hand to manipulate diverse objects for various purposes defined by kinematic human-object interactions. Developing such a controller is complicated by the intricate contact dynamics of dexterous manipulation and the need for adaptivity, generalizability, and robustness. Current reinforcement learning and trajectory optimization methods often fall short due to their dependence on task-specific rewards or precise system models. We introduce an approach that curates large-scale successful robot tracking demonstrations, comprising pairs of human references and robot actions, to train a neural controller. Utilizing a data flywheel, we iteratively enhance the controller's performance, as well as the number and quality of successful tracking demonstrations. We exploit available tracking demonstrations and carefully integrate reinforcement learning and imitation learning to boost the controller's performance in dynamic environments. At the same time, to obtain high-quality tracking demonstrations, we individually optimize per-trajectory tracking by leveraging the learned tracking controller in a homotopy optimization method. The homotopy optimization, mimicking chain-of-thought, aids in solving challenging trajectory tracking problems to increase demonstration diversity. We showcase our success by training a generalizable neural controller and evaluating it in both simulation and real world. Our method achieves over a 10% improvement in success rates compared to leading baselines. The project website with animated results is available at this https URL. 

**Abstract (ZH)**: 我们探讨了从人类参考中开发一种通用可移植的灵巧操作跟踪控制器的挑战。该控制器旨在管理灵巧的机器人手进行各种由运动学人机物体交互定义的目的性操作。由于灵巧操作中的复杂接触动力学和需要具备适应性、可移植性和鲁棒性，开发此类控制器具有很大复杂性。现有的强化学习和轨迹优化方法往往由于依赖于特定任务的奖励或精确的系统模型而力有未逮。我们提出了一种方法，通过收集包含人类参考和机器人动作配对的大型成功机器人跟踪演示，来训练神经控制器。利用数据飞轮，我们迭代地提升控制器的性能以及成功跟踪演示的数量和质量。我们利用可用的跟踪演示，精心整合强化学习和模仿学习，以增强控制器在动态环境中的表现。同时，为了获得高质量的跟踪演示，我们通过在同伦优化方法中使用所学习的跟踪控制器，独立优化每个轨迹的跟踪。同伦优化模仿链式思考，有助于解决复杂的轨迹跟踪问题，从而增加演示的多样性。通过在模拟和现实世界中训练和评估通用可移植的神经控制器，我们展示了我们的成功。与领先基准相比，我们的方法成功率达提高了超过10%。项目网站及动画结果展示可访问此链接：这个https URL。 

---
# Real-Time Fast Marching Tree for Mobile Robot Motion Planning in Dynamic Environments 

**Title (ZH)**: 实时快速推进树在动态环境中的移动机器人路径规划 

**Authors**: Jefferson Silveira, Kleber Cabral, Sidney Givigi, Joshua A. Marshall  

**Link**: [PDF](https://arxiv.org/pdf/2502.09556)  

**Abstract**: This paper proposes the Real-Time Fast Marching Tree (RT-FMT), a real-time planning algorithm that features local and global path generation, multiple-query planning, and dynamic obstacle avoidance. During the search, RT-FMT quickly looks for the global solution and, in the meantime, generates local paths that can be used by the robot to start execution faster. In addition, our algorithm constantly rewires the tree to keep branches from forming inside the dynamic obstacles and to maintain the tree root near the robot, which allows the tree to be reused multiple times for different goals. Our algorithm is based on the planners Fast Marching Tree (FMT*) and Real-time Rapidly-Exploring Random Tree (RT-RRT*). We show via simulations that RT-FMT outperforms RT- RRT* in both execution cost and arrival time, in most cases. Moreover, we also demonstrate via simulation that it is worthwhile taking the local path before the global path is available in order to reduce arrival time, even though there is a small possibility of taking an inferior path. 

**Abstract (ZH)**: 实时代价快速推进树（RT-FMT）：一种结合局部和全局路径规划、多查询规划及动态避障的实时规划算法 

---
# Variable Stiffness for Robust Locomotion through Reinforcement Learning 

**Title (ZH)**: 基于强化学习的稳健运动的可变刚度 

**Authors**: Dario Spoljaric, Yashuai Yan, Dongheui Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.09436)  

**Abstract**: Reinforcement-learned locomotion enables legged robots to perform highly dynamic motions but often accompanies time-consuming manual tuning of joint stiffness. This paper introduces a novel control paradigm that integrates variable stiffness into the action space alongside joint positions, enabling grouped stiffness control such as per-joint stiffness (PJS), per-leg stiffness (PLS) and hybrid joint-leg stiffness (HJLS). We show that variable stiffness policies, with grouping in per-leg stiffness (PLS), outperform position-based control in velocity tracking and push recovery. In contrast, HJLS excels in energy efficiency. Furthermore, our method showcases robust walking behaviour on diverse outdoor terrains by sim-to-real transfer, although the policy is sorely trained on a flat floor. Our approach simplifies design by eliminating per-joint stiffness tuning while keeping competitive results with various metrics. 

**Abstract (ZH)**: 强化学习驱动的变刚度腿式机器人运动控制：整合关节位置与变刚度的动作空间以实现高效行走行为 

---
# Robot Pouring: Identifying Causes of Spillage and Selecting Alternative Action Parameters Using Probabilistic Actual Causation 

**Title (ZH)**: 机器人倾倒：基于概率实际因果关系识别洒漏原因和选择替代动作参数 

**Authors**: Jaime Maldonado, Jonas Krumme, Christoph Zetzsche, Vanessa Didelez, Kerstin Schill  

**Link**: [PDF](https://arxiv.org/pdf/2502.09395)  

**Abstract**: In everyday life, we perform tasks (e.g., cooking or cleaning) that involve a large variety of objects and goals. When confronted with an unexpected or unwanted outcome, we take corrective actions and try again until achieving the desired result. The reasoning performed to identify a cause of the observed outcome and to select an appropriate corrective action is a crucial aspect of human reasoning for successful task execution. Central to this reasoning is the assumption that a factor is responsible for producing the observed outcome. In this paper, we investigate the use of probabilistic actual causation to determine whether a factor is the cause of an observed undesired outcome. Furthermore, we show how the actual causation probabilities can be used to find alternative actions to change the outcome. We apply the probabilistic actual causation analysis to a robot pouring task. When spillage occurs, the analysis indicates whether a task parameter is the cause and how it should be changed to avoid spillage. The analysis requires a causal graph of the task and the corresponding conditional probability distributions. To fulfill these requirements, we perform a complete causal modeling procedure (i.e., task analysis, definition of variables, determination of the causal graph structure, and estimation of conditional probability distributions) using data from a realistic simulation of the robot pouring task, covering a large combinatorial space of task parameters. Based on the results, we discuss the implications of the variables' representation and how the alternative actions suggested by the actual causation analysis would compare to the alternative solutions proposed by a human observer. The practical use of the analysis of probabilistic actual causation to select alternative action parameters is demonstrated. 

**Abstract (ZH)**: 日常生活中的任务执行涉及多种对象和目标，当遇到意外或不想看到的结果时，我们会采取纠正措施并再次尝试，直到达到预期结果。识别观察到的结果原因并选择适当的纠正措施是人类成功执行任务的重要推理方面。这项推理的核心假设是某个因素导致了观察到的结果。在本文中，我们研究了使用概率实际因果关系来确定某个因素是否是观察到的不良结果的原因。此外，我们展示了如何使用实际因果关系的概率来寻找改变结果的替代行动。我们将概率实际因果关系分析应用于机器人倒水任务。当出现溢出时，分析可以表明任务参数是否是原因及其如何改变以避免溢出。该分析需要任务的因果图及其相应的条件概率分布。为满足这些要求，我们使用机器人倒水任务的现实仿真数据，进行了完整的因果建模流程（即任务分析、变量定义、因果图结构确定和条件概率分布估计），覆盖了大量的任务参数组合空间。基于这些结果，我们讨论了变量表示的含义以及由实际因果关系分析建议的替代行动与人类观察者提出的替代解决方案的对比。展示了一种实际应用概率实际因果关系分析来选择替代行动参数的方法。 

---
# Generalizable Reinforcement Learning with Biologically Inspired Hyperdimensional Occupancy Grid Maps for Exploration and Goal-Directed Path Planning 

**Title (ZH)**: 基于生物启发的高维占用网格图的可泛化强化学习及其在探索与目标导向路径规划中的应用 

**Authors**: Shay Snyder, Ryan Shea, Andrew Capodieci, David Gorsich, Maryam Parsa  

**Link**: [PDF](https://arxiv.org/pdf/2502.09393)  

**Abstract**: Real-time autonomous systems utilize multi-layer computational frameworks to perform critical tasks such as perception, goal finding, and path planning. Traditional methods implement perception using occupancy grid mapping (OGM), segmenting the environment into discretized cells with probabilistic information. This classical approach is well-established and provides a structured input for downstream processes like goal finding and path planning algorithms. Recent approaches leverage a biologically inspired mathematical framework known as vector symbolic architectures (VSA), commonly known as hyperdimensional computing, to perform probabilistic OGM in hyperdimensional space. This approach, VSA-OGM, provides native compatibility with spiking neural networks, positioning VSA-OGM as a potential neuromorphic alternative to conventional OGM. However, for large-scale integration, it is essential to assess the performance implications of VSA-OGM on downstream tasks compared to established OGM methods. This study examines the efficacy of VSA-OGM against a traditional OGM approach, Bayesian Hilbert Maps (BHM), within reinforcement learning based goal finding and path planning frameworks, across a controlled exploration environment and an autonomous driving scenario inspired by the F1-Tenth challenge. Our results demonstrate that VSA-OGM maintains comparable learning performance across single and multi-scenario training configurations while improving performance on unseen environments by approximately 47%. These findings highlight the increased generalizability of policy networks trained with VSA-OGM over BHM, reinforcing its potential for real-world deployment in diverse environments. 

**Abstract (ZH)**: 实时自主系统利用多层计算框架执行关键任务，如感知、目标定位和路径规划。传统方法使用占用网格映射（OGM）进行感知，将环境分割为具有概率信息的离散单元格。这一经典方法已被广泛认可，并为下游处理如目标定位和路径规划算法提供结构化的输入。近期的方法利用一种生物启发的数学框架，即向量符号架构（VSA），通常称为超维计算（Hyperdimensional Computing），在超维空间中执行概率OGM。该方法VSA-OGM与脉冲神经网络有天然兼容性，将VSA-OGM定位为经典OGM的潜在神经形态替代方案。然而，为了实现大规模集成，评估VSA-OGM在下游任务中的性能影响及其与传统OGM方法相比的重要性是必不可少的。本研究在基于强化学习的目标定位和路径规划框架中，对比了VSA-OGM与传统OGM方法（贝叶斯希爾伯特映射BHM）的效能，研究在受控探索环境和基于F1-Tenth挑战的自主驾驶场景中的表现。实验结果显示，VSA-OGM在单场景和多场景训练配置中保持了相似的学习性能，并在未见过的环境中提高了约47%的性能。这些发现突显了使用VSA-OGM训练的策略网络具有更强的通用性，增强了其在不同环境中的实际部署潜力。 

---
# S$^2$-Diffusion: Generalizing from Instance-level to Category-level Skills in Robot Manipulation 

**Title (ZH)**: S$^2$-Diffusion: 从实例级到类别级技能的泛化在机器人操作中 

**Authors**: Quantao Yang, Michael C. Welle, Danica Kragic, Olov Andersson  

**Link**: [PDF](https://arxiv.org/pdf/2502.09389)  

**Abstract**: Recent advances in skill learning has propelled robot manipulation to new heights by enabling it to learn complex manipulation tasks from a practical number of demonstrations. However, these skills are often limited to the particular action, object, and environment \textit{instances} that are shown in the training data, and have trouble transferring to other instances of the same category. In this work we present an open-vocabulary Spatial-Semantic Diffusion policy (S$^2$-Diffusion) which enables generalization from instance-level training data to category-level, enabling skills to be transferable between instances of the same category. We show that functional aspects of skills can be captured via a promptable semantic module combined with a spatial representation. We further propose leveraging depth estimation networks to allow the use of only a single RGB camera. Our approach is evaluated and compared on a diverse number of robot manipulation tasks, both in simulation and in the real world. Our results show that S$^2$-Diffusion is invariant to changes in category-irrelevant factors as well as enables satisfying performance on other instances within the same category, even if it was not trained on that specific instance. Full videos of all real-world experiments are available in the supplementary material. 

**Abstract (ZH)**: 最近在技能学习方面的进展使机器人的操作达到了新的高度，使其能够通过少量示范学习复杂的操作任务。然而，这些技能通常局限于训练数据中展示的具体动作、物体和环境实例，并且难以转移到同一类别的其他实例。本工作中，我们提出了一种开放词汇量的空间语义扩散策略（S$^2$-Diffusion），使技能可以从实例级别的训练数据推广到类别级别，从而使同一类别不同实例间的技能转移成为可能。我们表明，可以通过一个可提示的语义模块结合空间表示来捕获技能的功能方面。我们进一步提出利用深度估计网络，仅使用单个RGB相机即可。我们的方法在多种机器人操作任务上进行了评估和比较，包括模拟环境和真实世界环境。我们的结果显示，S$^2$-Diffusion对与类别无关的因素变化保持不变，并且即使未针对特定实例进行训练，也在同一类别的其他实例上实现了令人满意的表现。所有真实世界实验的完整视频可在补充材料中找到。 

---
# TRIFFID: Autonomous Robotic Aid For Increasing First Responders Efficiency 

**Title (ZH)**: TRIFFID: 自主机器人辅助以提高首Responder效率 

**Authors**: Jorgen Cani, Panagiotis Koletsis, Konstantinos Foteinos, Ioannis Kefaloukos, Lampros Argyriou, Manolis Falelakis, Iván Del Pino, Angel Santamaria-Navarro, Martin Čech, Ondřej Severa, Alessandro Umbrico, Francesca Fracasso, AndreA Orlandini, Dimitrios Drakoulis, Evangelos Markakis, Georgios Th. Papadopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2502.09379)  

**Abstract**: The increasing complexity of natural disaster incidents demands innovative technological solutions to support first responders in their efforts. This paper introduces the TRIFFID system, a comprehensive technical framework that integrates unmanned ground and aerial vehicles with advanced artificial intelligence functionalities to enhance disaster response capabilities across wildfires, urban floods, and post-earthquake search and rescue missions. By leveraging state-of-the-art autonomous navigation, semantic perception, and human-robot interaction technologies, TRIFFID provides a sophisticated system com- posed of the following key components: hybrid robotic platform, centralized ground station, custom communication infrastructure, and smartphone application. The defined research and development activities demonstrate how deep neural networks, knowledge graphs, and multimodal information fusion can enable robots to autonomously navigate and analyze disaster environ- ments, reducing personnel risks and accelerating response times. The proposed system enhances emergency response teams by providing advanced mission planning, safety monitoring, and adaptive task execution capabilities. Moreover, it ensures real- time situational awareness and operational support in complex and risky situations, facilitating rapid and precise information collection and coordinated actions. 

**Abstract (ZH)**: 自然灾害事件复杂性的增加要求创新的技术解决方案以支持一线救援人员的努力。本文介绍了TRIFFID系统，这是一种综合技术框架，将无人驾驶地面和航空车辆与高级人工智能功能集成，以增强针对野火、城市洪水和地震后的搜索与救援任务的灾害响应能力。通过利用先进的自主导航、语义感知和人机交互技术，TRIFFID提供了一个由以下关键组成部分组成的复杂系统：混合机器人平台、集中地面站、定制通信基础设施和智能手机应用程序。定义的研究和开发活动展示了深度神经网络、知识图谱和多模态信息融合如何使机器人能够自主导航和分析灾害环境，减少人员风险并加快响应时间。所提出的系统通过提供先进的任务规划、安全监控和自适应任务执行能力，增强了应急响应团队。此外，该系统确保在复杂和危险情况下实现实时态势感知和操作支持，促进快速和精确的信息收集与协调行动。 

---
# GEVRM: Goal-Expressive Video Generation Model For Robust Visual Manipulation 

**Title (ZH)**: GEVRM：目标表达型视频生成模型在稳健视觉操控中的应用 

**Authors**: Hongyin Zhang, Pengxiang Ding, Shangke Lyu, Ying Peng, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.09268)  

**Abstract**: With the rapid development of embodied artificial intelligence, significant progress has been made in vision-language-action (VLA) models for general robot decision-making. However, the majority of existing VLAs fail to account for the inevitable external perturbations encountered during deployment. These perturbations introduce unforeseen state information to the VLA, resulting in inaccurate actions and consequently, a significant decline in generalization performance. The classic internal model control (IMC) principle demonstrates that a closed-loop system with an internal model that includes external input signals can accurately track the reference input and effectively offset the disturbance. We propose a novel closed-loop VLA method GEVRM that integrates the IMC principle to enhance the robustness of robot visual manipulation. The text-guided video generation model in GEVRM can generate highly expressive future visual planning goals. Simultaneously, we evaluate perturbations by simulating responses, which are called internal embeddings and optimized through prototype contrastive learning. This allows the model to implicitly infer and distinguish perturbations from the external environment. The proposed GEVRM achieves state-of-the-art performance on both standard and perturbed CALVIN benchmarks and shows significant improvements in realistic robot tasks. 

**Abstract (ZH)**: 随着具身人工智能的快速发展，视觉-语言-动作（VLA）模型在通用机器人决策制定方面的进展显著。然而，现有的大多数VLA未能考虑部署过程中不可避免的外部干扰。这些干扰引入了未预见的状态信息，导致动作不准确，从而显著降低了泛化性能。经典的内部模型控制（IMC）原理表明，包含外部输入信号的内部模型闭环系统能够准确跟踪参考输入并有效抵消干扰。我们提出了一种结合IMC原理的新型闭环VLA方法GEVRM，以增强机器人视觉操作的鲁棒性。GEVRM中的文本指导视频生成模型可以生成高度表达性的未来视觉规划目标。同时，我们通过模拟响应评估干扰，这些响应称为内部嵌入，并通过原型对比学习进行优化。这使模型能够隐式推断和区分来自外部环境的干扰。所提出的GEVRM在标准和干扰的CALVIN基准上均实现了最先进的性能，并在现实机器人任务中显示出显著改进。 

---
# Safety Evaluation of Human Arm Operations Using IMU Sensors with a Spring-Damper-Mass Predictive Model 

**Title (ZH)**: 基于弹簧-阻尼-质量预测模型的惯性传感器在人体手臂操作安全性评估 

**Authors**: Musab Zubair Inamdar, Seyed Amir Tafrishi  

**Link**: [PDF](https://arxiv.org/pdf/2502.09241)  

**Abstract**: This paper presents a novel approach to real-time safety monitoring in human-robot collaborative manufacturing environments through a wrist-mounted Inertial Measurement Unit (IMU) system integrated with a Predictive Safety Model (PSM). The proposed system extends previous PSM implementations through the adaptation of a spring-damper-mass model specifically optimized for wrist motions, employing probabilistic safety assessment through impedance-based computations. We analyze our proposed impedance-based safety approach with frequency domain methods, establishing quantitative safety thresholds through comprehensive comparative analysis. Experimental validation across three manufacturing tasks - tool manipulation, visual inspection, and pick-and-place operations. Results show robust performance across diverse manufacturing scenarios while maintaining computational efficiency through optimized parameter selection. This work establishes a foundation for future developments in adaptive risk assessment in real-time for human-robot collaborative manufacturing environments. 

**Abstract (ZH)**: 本文提出了一种通过集成预测安全性模型（PSM）的手腕佩戴式惯性测量单元（IMU）系统进行实时安全性监控的新型方法。该提出的系统通过适应一种专门针对手腕运动优化的弹簧阻尼质量模型，扩展了之前的PSM实现，并通过基于阻抗的计算进行概率安全性评估。我们使用频域方法分析了基于阻抗的安全性方法，并通过全面的比较分析建立了量化安全门槛。在三种制造任务（工具操作、视觉检查和取放操作）中进行了实验验证，结果显示该系统在多种制造场景下表现出稳健的性能，同时通过优化参数选择保持了计算效率。本工作为未来在人机协作制造环境中进行实时自适应风险评估奠定了基础。 

---
# OpenBench: A New Benchmark and Baseline for Semantic Navigation in Smart Logistics 

**Title (ZH)**: OpenBench：智能物流中的语义导航新基准和基线 

**Authors**: Junhui Wang, Dongjie Huo, Zehui Xu, Yongliang Shi, Yimin Yan, Yuanxin Wang, Chao Gao, Yan Qiao, Guyue Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.09238)  

**Abstract**: The increasing demand for efficient last-mile delivery in smart logistics underscores the role of autonomous robots in enhancing operational efficiency and reducing costs. Traditional navigation methods, which depend on high-precision maps, are resource-intensive, while learning-based approaches often struggle with generalization in real-world scenarios. To address these challenges, this work proposes the Openstreetmap-enhanced oPen-air sEmantic Navigation (OPEN) system that combines foundation models with classic algorithms for scalable outdoor navigation. The system uses off-the-shelf OpenStreetMap (OSM) for flexible map representation, thereby eliminating the need for extensive pre-mapping efforts. It also employs Large Language Models (LLMs) to comprehend delivery instructions and Vision-Language Models (VLMs) for global localization, map updates, and house number recognition. To compensate the limitations of existing benchmarks that are inadequate for assessing last-mile delivery, this work introduces a new benchmark specifically designed for outdoor navigation in residential areas, reflecting the real-world challenges faced by autonomous delivery systems. Extensive experiments in simulated and real-world environments demonstrate the proposed system's efficacy in enhancing navigation efficiency and reliability. To facilitate further research, our code and benchmark are publicly available. 

**Abstract (ZH)**: 基于OpenStreetMap增强的户外语义导航（OPEN）系统：面向自主最后英里交付的可扩展导航方法 

---
# A Machine Learning Approach to Sensor Substitution for Non-Prehensile Manipulation 

**Title (ZH)**: 一种用于非抓握操作的传感器替代的机器学习方法 

**Authors**: Idil Ozdamar, Doganay Sirintuna, Arash Ajoudani  

**Link**: [PDF](https://arxiv.org/pdf/2502.09180)  

**Abstract**: Mobile manipulators are increasingly deployed in complex environments, requiring diverse sensors to perceive and interact with their surroundings. However, equipping every robot with every possible sensor is often impractical due to cost and physical constraints. A critical challenge arises when robots with differing sensor capabilities need to collaborate or perform similar tasks. For example, consider a scenario where a mobile manipulator equipped with high-resolution tactile skin is skilled at non-prehensile manipulation tasks like pushing. If this robot needs to be replaced or augmented by a robot lacking such tactile sensing, the learned manipulation policies become inapplicable. This paper addresses the problem of sensor substitution in non-prehensile manipulation. We propose a novel machine learning-based framework that enables a robot with a limited sensor set (e.g., LiDAR or RGB-D camera) to effectively perform tasks previously reliant on a richer sensor suite (e.g., tactile skin). Our approach learns a mapping between the available sensor data and the information provided by the substituted sensor, effectively synthesizing the missing sensory input. Specifically, we demonstrate the efficacy of our framework by training a model to substitute tactile skin data for the task of non-prehensile pushing using a mobile manipulator. We show that a manipulator equipped only with LiDAR or RGB-D can, after training, achieve comparable and sometimes even better pushing performance to a mobile base utilizing direct tactile feedback. 

**Abstract (ZH)**: 移动 manipulator 在复杂环境中部署日益增多，需要多种传感器来感知和交互。然而，为每台机器人配备所有可能的传感器往往由于成本和物理限制而 impractical。当具有不同传感器能力的机器人需要协作或执行类似任务时，一个关键挑战随之而来。例如，考虑一个高分辨率触觉皮肤装备在移动 manipulator 上，擅长非拾取式操作任务如推动的情况。如果这台机器人被其他缺少触觉传感器的机器人替换或增强，学到的操作策略将变得不适用。本论文解决了非拾取式操作中的传感器替代问题。我们提出一种基于机器学习的新颖框架，使具有有限传感器集的机器人（如 LiDAR 或 RGB-D 相机）能够有效执行依赖于更丰富传感器套件的任务（如触觉皮肤）。我们的方法学习可用传感器数据与被替代传感器提供的信息之间的映射，有效合成缺失的感觉输入。具体而言，我们通过训练模型将触觉皮肤数据替换为移动 manipulator 上的非拾取式推动任务，展示了该框架的有效性。结果显示，仅配备 LiDAR 或 RGB-D 的 manipulator 经过训练后，能够与直接触觉反馈的移动基座相媲美，甚至在某些情况下性能更优异。 

---
# LimSim Series: An Autonomous Driving Simulation Platform for Validation and Enhancement 

**Title (ZH)**: LimSim系列：一种用于验证和增强的 Autonomous Driving Simulation Platform 

**Authors**: Daocheng Fu, Naiting Zhong, Xu Han, Pinlong Cai, Licheng Wen, Song Mao, Botian Shi, Yu Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2502.09170)  

**Abstract**: Closed-loop simulation environments play a crucial role in the validation and enhancement of autonomous driving systems (ADS). However, certain challenges warrant significant attention, including balancing simulation accuracy with duration, reconciling functionality with practicality, and establishing comprehensive evaluation mechanisms. This paper addresses these challenges by introducing the LimSim Series, a comprehensive simulation platform designed to support the rapid deployment and efficient iteration of ADS. The LimSim Series integrates multi-type information from road networks, employs human-like decision-making and planning algorithms for background vehicles, and introduces the concept of the Area of Interest (AoI) to optimize computational resources. The platform offers a variety of baseline algorithms and user-friendly interfaces, facilitating flexible validation of multiple technical pipelines. Additionally, the LimSim Series incorporates multi-dimensional evaluation metrics, delivering thorough insights into system performance, thus enabling researchers to promptly identify issues for further improvements. Experiments demonstrate that the LimSim Series is compatible with modular, end-to-end, and VLM-based knowledge-driven systems. It can assist in the iteration and updating of ADS by evaluating performance across various scenarios. The code of the LimSim Series is released at: this https URL. 

**Abstract (ZH)**: 闭-loop仿真环境在自主驾驶系统（ADS）的验证与提升中发挥着关键作用，然而，某些挑战需要引起重视，包括平衡仿真准确性和时间长度、协调功能性和实用性以及建立全面的评估机制。本文通过引入LimSim系列综合仿真平台来应对这些挑战，该平台旨在支持ADS的快速部署和高效迭代。LimSim系列整合了多种道路网络信息，采用了人类类似的决策和规划算法来模拟背景车辆，并引入了兴趣区域（Area of Interest，AoI）的概念来优化计算资源。该平台提供了多种基本算法和用户友好的界面，便于灵活验证多个技术管线。此外，LimSim系列还整合了多维度的评估指标，提供了系统性能的全面洞察，从而帮助研究人员及时发现需要改进的问题。实验表明，LimSim系列兼容模块化、端到端和基于VLM的知识驱动系统，可以辅助在各种场景下评估和更新ADS。LimSim系列的代码发布在：this https URL。 

---
# MTDP: Modulated Transformer Diffusion Policy Model 

**Title (ZH)**: MTDP: 调制变换扩散策略模型 

**Authors**: Qianhao Wang, Yinqian Sun, Enmeng Lu, Qian Zhang, Yi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2502.09029)  

**Abstract**: Recent research on robot manipulation based on Behavior Cloning (BC) has made significant progress. By combining diffusion models with BC, diffusion policiy has been proposed, enabling robots to quickly learn manipulation tasks with high success rates. However, integrating diffusion policy with high-capacity Transformer presents challenges, traditional Transformer architectures struggle to effectively integrate guiding conditions, resulting in poor performance in manipulation tasks when using Transformer-based models. In this paper, we investigate key architectural designs of Transformers and improve the traditional Transformer architecture by proposing the Modulated Transformer Diffusion Policy (MTDP) model for diffusion policy. The core of this model is the Modulated Attention module we proposed, which more effectively integrates the guiding conditions with the main input, improving the generative model's output quality and, consequently, increasing the robot's task success rate. In six experimental tasks, MTDP outperformed existing Transformer model architectures, particularly in the Toolhang experiment, where the success rate increased by 12\%. To verify the generality of Modulated Attention, we applied it to the UNet architecture to construct Modulated UNet Diffusion Policy model (MUDP), which also achieved higher success rates than existing UNet architectures across all six experiments. The Diffusion Policy uses Denoising Diffusion Probabilistic Models (DDPM) as the diffusion model. Building on this, we also explored Denoising Diffusion Implicit Models (DDIM) as the diffusion model, constructing the MTDP-I and MUDP-I model, which nearly doubled the generation speed while maintaining performance. 

**Abstract (ZH)**: 基于行为克隆的机器人操作研究：摩爾 Filed 变形器扩散策略（MTDP）模型 

---
# SkyRover: A Modular Simulator for Cross-Domain Pathfinding 

**Title (ZH)**: SkyRover: 跨域路径finding的模块化模拟器 

**Authors**: Wenhui Ma, Wenhao Li, Bo Jin, Changhong Lu, Xiangfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.08969)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) and Automated Guided Vehicles (AGVs) increasingly collaborate in logistics, surveillance, inspection tasks and etc. However, existing simulators often focus on a single domain, limiting cross-domain study. This paper presents the SkyRover, a modular simulator for UAV-AGV multi-agent pathfinding (MAPF). SkyRover supports realistic agent dynamics, configurable 3D environments, and convenient APIs for external solvers and learning methods. By unifying ground and aerial operations, it facilitates cross-domain algorithm design, testing, and benchmarking. Experiments highlight SkyRover's capacity for efficient pathfinding and high-fidelity simulations in UAV-AGV coordination. Project is available at this https URL. 

**Abstract (ZH)**: 无人驾驶航空车辆(UAVs)和自动引导车(AGVs)在物流、 surveillance、检测等任务中逐渐实现协同合作。然而，现有的模拟器往往专注于单一领域，限制了跨领域的研究。本文介绍了SkyRover，一个支持多智能体路径规划(UAV-AGV MAPF)的模块化模拟器。SkyRover支持现实的智能体动力学、可配置的3D环境，并提供了对外部求解器和学习方法的便捷接口。通过统一地面和空中操作，它有助于跨领域的算法设计、测试和基准测试。实验突显了SkyRover在UAV-AGV协调中的高效路径规划和高保真模拟能力。项目可在以下链接访问：this https URL。 

---
# Training Trajectory Predictors Without Ground-Truth Data 

**Title (ZH)**: 不使用_ground-truth_数据训练轨迹预测器 

**Authors**: Mikolaj Kliniewski, Jesse Morris, Ian R. Manchester, Viorela Ila  

**Link**: [PDF](https://arxiv.org/pdf/2502.08957)  

**Abstract**: This paper presents a framework capable of accurately and smoothly estimating position, heading, and velocity. Using this high-quality input, we propose a system based on Trajectron++, able to consistently generate precise trajectory predictions. Unlike conventional models that require ground-truth data for training, our approach eliminates this dependency. Our analysis demonstrates that poor quality input leads to noisy and unreliable predictions, which can be detrimental to navigation modules. We evaluate both input data quality and model output to illustrate the impact of input noise. Furthermore, we show that our estimation system enables effective training of trajectory prediction models even with limited data, producing robust predictions across different environments. Accurate estimations are crucial for deploying trajectory prediction models in real-world scenarios, and our system ensures meaningful and reliable results across various application contexts. 

**Abstract (ZH)**: 本文提出了一种能够准确且平滑地估计位置、航向和速度的框架。利用高质量的输入，我们提出了一种基于Trajectron++的系统，能够一致地生成精确的轨迹预测。与需要地面真实数据进行训练的传统模型不同，我们的方法消除了对这种数据的依赖。我们的分析表明，低质量的输入会导致噪声大且不可靠的预测，这对导航模块可能有负面影响。我们评估了输入数据质量和模型输出，以阐明输入噪声的影响。此外，我们展示了我们的估计系统即使在数据有限的情况下也能有效训练轨迹预测模型，产生在不同环境中稳健的预测。准确的估计对于在真实世界场景中部署轨迹预测模型至关重要，我们的系统确保在各种应用上下文中产生有意义且可靠的结果。 

---
# 3D-Grounded Vision-Language Framework for Robotic Task Planning: Automated Prompt Synthesis and Supervised Reasoning 

**Title (ZH)**: 基于3D场景的视觉语言机器人任务规划框架：自动提示合成与监督推理 

**Authors**: Guoqin Tang, Qingxuan Jia, Zeyuan Huang, Gang Chen, Ning Ji, Zhipeng Yao  

**Link**: [PDF](https://arxiv.org/pdf/2502.08903)  

**Abstract**: Vision-language models (VLMs) have achieved remarkable success in scene understanding and perception tasks, enabling robots to plan and execute actions adaptively in dynamic environments. However, most multimodal large language models lack robust 3D scene localization capabilities, limiting their effectiveness in fine-grained robotic operations. Additionally, challenges such as low recognition accuracy, inefficiency, poor transferability, and reliability hinder their use in precision tasks. To address these limitations, we propose a novel framework that integrates a 2D prompt synthesis module by mapping 2D images to point clouds, and incorporates a small language model (SLM) for supervising VLM outputs. The 2D prompt synthesis module enables VLMs, trained on 2D images and text, to autonomously extract precise 3D spatial information without manual intervention, significantly enhancing 3D scene understanding. Meanwhile, the SLM supervises VLM outputs, mitigating hallucinations and ensuring reliable, executable robotic control code generation. Our framework eliminates the need for retraining in new environments, thereby improving cost efficiency and operational robustness. Experimental results that the proposed framework achieved a 96.0\% Task Success Rate (TSR), outperforming other methods. Ablation studies demonstrated the critical role of both the 2D prompt synthesis module and the output supervision module (which, when removed, caused a 67\% TSR drop). These findings validate the framework's effectiveness in improving 3D recognition, task planning, and robotic task execution. 

**Abstract (ZH)**: Vision-Language模型在场景理解与感知任务中取得了显著成功，使机器人能够在动态环境中适配地计划和执行动作。然而，大多数多模态大型语言模型缺乏 robust 的3D场景定位能力，限制了它们在精确机器人操作中的有效性。此外，低识别准确性、低效性、较差的迁移性和可靠性等挑战阻碍了它们在精密任务中的应用。为了应对这些局限性，我们提出了一种新颖的框架，该框架通过将2D图像映射到点云来集成一个2D提示合成模块，并结合一个小语言模型（SLM）监督VLM输出。2D提示合成模块使经过2D图像和文本训练的VLM能够在不需要人工干预的情况下自主提取精确的3D空间信息，显著增强了3D场景理解。同时，SLM监督VLM输出，减轻幻觉并确保可靠的、可执行的机器人控制代码生成。该框架消除了对新环境重训练的需求，从而提高了成本效率和操作稳健性。实验结果显示，所提出框架实现了96.0%的任务成功率（TSR），优于其他方法。消融研究证明了2D提示合成模块和输出监督模块的双重重要性（当移除其中一个时，TSR下降67%）。这些发现验证了该框架在提高3D识别、任务规划和机器人任务执行方面的有效性。 

---
# MuJoCo Playground 

**Title (ZH)**: MuJoCo playground 

**Authors**: Kevin Zakka, Baruch Tabanpour, Qiayuan Liao, Mustafa Haiderbhai, Samuel Holt, Jing Yuan Luo, Arthur Allshire, Erik Frey, Koushil Sreenath, Lueder A. Kahrs, Carmelo Sferrazza, Yuval Tassa, Pieter Abbeel  

**Link**: [PDF](https://arxiv.org/pdf/2502.08844)  

**Abstract**: We introduce MuJoCo Playground, a fully open-source framework for robot learning built with MJX, with the express goal of streamlining simulation, training, and sim-to-real transfer onto robots. With a simple "pip install playground", researchers can train policies in minutes on a single GPU. Playground supports diverse robotic platforms, including quadrupeds, humanoids, dexterous hands, and robotic arms, enabling zero-shot sim-to-real transfer from both state and pixel inputs. This is achieved through an integrated stack comprising a physics engine, batch renderer, and training environments. Along with video results, the entire framework is freely available at this http URL 

**Abstract (ZH)**: MuJoCo Playground：一个基于MJX构建的完全开源的机器人学习框架，旨在简化模拟、训练和从模拟到实际机器人的转移过程 

---
# ClipRover: Zero-shot Vision-Language Exploration and Target Discovery by Mobile Robots 

**Title (ZH)**: ClipRover：零样本视觉-语言探索与目标发现的移动机器人方法 

**Authors**: Yuxuan Zhang, Adnan Abdullah, Sanjeev J. Koppal, Md Jahidul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2502.08791)  

**Abstract**: Vision-language navigation (VLN) has emerged as a promising paradigm, enabling mobile robots to perform zero-shot inference and execute tasks without specific pre-programming. However, current systems often separate map exploration and path planning, with exploration relying on inefficient algorithms due to limited (partially observed) environmental information. In this paper, we present a novel navigation pipeline named ''ClipRover'' for simultaneous exploration and target discovery in unknown environments, leveraging the capabilities of a vision-language model named CLIP. Our approach requires only monocular vision and operates without any prior map or knowledge about the target. For comprehensive evaluations, we design the functional prototype of a UGV (unmanned ground vehicle) system named ''Rover Master'', a customized platform for general-purpose VLN tasks. We integrate and deploy the ClipRover pipeline on Rover Master to evaluate its throughput, obstacle avoidance capability, and trajectory performance across various real-world scenarios. Experimental results demonstrate that ClipRover consistently outperforms traditional map traversal algorithms and achieves performance comparable to path-planning methods that depend on prior map and target knowledge. Notably, ClipRover offers real-time active navigation without requiring pre-captured candidate images or pre-built node graphs, addressing key limitations of existing VLN pipelines. 

**Abstract (ZH)**: 基于视觉语言的同步探索与目标发现：ClipRover导航管道 

---
# Acoustic Wave Manipulation Through Sparse Robotic Actuation 

**Title (ZH)**: 通过稀疏机器人驱动实现声波操控 

**Authors**: Tristan Shah, Noam Smilovich, Samer Gerges, Feruza Amirkulova, Stas Tiomkin  

**Link**: [PDF](https://arxiv.org/pdf/2502.08784)  

**Abstract**: Recent advancements in robotics, control, and machine learning have facilitated progress in the challenging area of object manipulation. These advancements include, among others, the use of deep neural networks to represent dynamics that are partially observed by robot sensors, as well as effective control using sparse control signals. In this work, we explore a more general problem: the manipulation of acoustic waves, which are partially observed by a robot capable of influencing the waves through spatially sparse actuators. This problem holds great potential for the design of new artificial materials, ultrasonic cutting tools, energy harvesting, and other applications. We develop an efficient data-driven method for robot learning that is applicable to either focusing scattered acoustic energy in a designated region or suppressing it, depending on the desired task. The proposed method is better in terms of a solution quality and computational complexity as compared to a state-of-the-art learning based method for manipulation of dynamical systems governed by partial differential equations. Furthermore our proposed method is competitive with a classical semi-analytical method in acoustics research on the demonstrated tasks. We have made the project code publicly available, along with a web page featuring video demonstrations: this https URL. 

**Abstract (ZH)**: 近期机器人学、控制理论和机器学习的发展促进了物体操纵这一挑战性领域的进步。这些进展包括使用深度神经网络来表示部分被机器人传感器观测的动力学，以及通过稀疏控制信号实现有效的控制。在这项工作中，我们探讨了一个更广泛的问题：通过能通过空间稀疏执行器影响声波的机器人操纵声波，这些声波仅部分被机器人观测。该问题在设计新型人工材料、超声切割工具、能量采集等领域方面具有巨大潜力。我们开发了一种高效的基于数据的方法用于机器人学习，该方法可以根据所需任务将散射的声能聚集到指定区域或抑制它。所提出的方法在解决方案质量和计算复杂性方面优于用于由偏微分方程支配的动力学系统操纵的一种最先进的基于学习的方法。此外，在展示的任务上，我们的方法在声学研究方面与一种经典的半解析方法具有竞争力。我们已经将项目代码公开，并在网站上提供了视频演示：这个链接https://example.com。 

---
# Bilevel Learning for Bilevel Planning 

**Title (ZH)**: bilevel学习用于 bilevel 规划 

**Authors**: Bowen Li, Tom Silver, Sebastian Scherer, Alexander Gray  

**Link**: [PDF](https://arxiv.org/pdf/2502.08697)  

**Abstract**: A robot that learns from demonstrations should not just imitate what it sees -- it should understand the high-level concepts that are being demonstrated and generalize them to new tasks. Bilevel planning is a hierarchical model-based approach where predicates (relational state abstractions) can be leveraged to achieve compositional generalization. However, previous bilevel planning approaches depend on predicates that are either hand-engineered or restricted to very simple forms, limiting their scalability to sophisticated, high-dimensional state spaces. To address this limitation, we present IVNTR, the first bilevel planning approach capable of learning neural predicates directly from demonstrations. Our key innovation is a neuro-symbolic bilevel learning framework that mirrors the structure of bilevel planning. In IVNTR, symbolic learning of the predicate "effects" and neural learning of the predicate "functions" alternate, with each providing guidance for the other. We evaluate IVNTR in six diverse robot planning domains, demonstrating its effectiveness in abstracting various continuous and high-dimensional states. While most existing approaches struggle to generalize (with <35% success rate), our IVNTR achieves an average of 77% success rate on unseen tasks. Additionally, we showcase IVNTR on a mobile manipulator, where it learns to perform real-world mobile manipulation tasks and generalizes to unseen test scenarios that feature new objects, new states, and longer task horizons. Our findings underscore the promise of learning and planning with abstractions as a path towards high-level generalization. 

**Abstract (ZH)**: 一种能直接从演示中学习神经谓词的机器人不应该只是模仿所见的动作，而应该理解展示的高层概念并将其应用于新任务。IVNTR：一种能从演示中学习神经谓词的 bilevel 计划方法 

---
# LIR-LIVO: A Lightweight,Robust LiDAR/Vision/Inertial Odometry with Illumination-Resilient Deep Features 

**Title (ZH)**: LIR-LIVO：一种抗照度变化的轻量级、稳健的LiDAR/视觉/惯性里程计 

**Authors**: Shujie Zhou, Zihao Wang, Xinye Dai, Weiwei Song, Shengfeng Gu  

**Link**: [PDF](https://arxiv.org/pdf/2502.08676)  

**Abstract**: In this paper, we propose LIR-LIVO, a lightweight and robust LiDAR-inertial-visual odometry system designed for challenging illumination and degraded environments. The proposed method leverages deep learning-based illumination-resilient features and LiDAR-Inertial-Visual Odometry (LIVO). By incorporating advanced techniques such as uniform depth distribution of features enabled by depth association with LiDAR point clouds and adaptive feature matching utilizing Superpoint and LightGlue, LIR-LIVO achieves state-of-the-art (SOTA) accuracy and robustness with low computational cost. Experiments are conducted on benchmark datasets, including NTU-VIRAL, Hilti'22, and R3LIVE-Dataset. The corresponding results demonstrate that our proposed method outperforms other SOTA methods on both standard and challenging datasets. Particularly, the proposed method demonstrates robust pose estimation under poor ambient lighting conditions in the Hilti'22 dataset. The code of this work is publicly accessible on GitHub to facilitate advancements in the robotics community. 

**Abstract (ZH)**: 基于LiDAR-惯性-视觉里程计的轻量级鲁棒系统LIR-LIVO：针对复杂光照和降级环境的设计与实现 

---
# Motion Forecasting for Autonomous Vehicles: A Survey 

**Title (ZH)**: 自主驾驶车辆的运动预测：一个文献综述 

**Authors**: Jianxin Shi, Jinhao Chen, Yuandong Wang, Li Sun, Chunyang Liu, Wei Xiong, Tianyu Wo  

**Link**: [PDF](https://arxiv.org/pdf/2502.08664)  

**Abstract**: In recent years, the field of autonomous driving has attracted increasingly significant public interest. Accurately forecasting the future behavior of various traffic participants is essential for the decision-making of Autonomous Vehicles (AVs). In this paper, we focus on both scenario-based and perception-based motion forecasting for AVs. We propose a formal problem formulation for motion forecasting and summarize the main challenges confronting this area of research. We also detail representative datasets and evaluation metrics pertinent to this field. Furthermore, this study classifies recent research into two main categories: supervised learning and self-supervised learning, reflecting the evolving paradigms in both scenario-based and perception-based motion forecasting. In the context of supervised learning, we thoroughly examine and analyze each key element of the methodology. For self-supervised learning, we summarize commonly adopted techniques. The paper concludes and discusses potential research directions, aiming to propel progress in this vital area of AV technology. 

**Abstract (ZH)**: 近年来，自动驾驶领域吸引了日益显著的公众关注。准确预测各种交通参与者的未来行为是自动驾驶车辆决策制定的关键。本文专注于自动驾驶车辆的场景基于和感知基于的运动预测。我们提出了一种形式化的问题表述，总结了该研究领域面临的主耍挑战，并详细介绍了相关的代表性数据集和评价指标。此外，本研究将最近的研究分类为监督学习和自监督学习两大类，反映了这两种运动预测方法中的演进范式。在监督学习的背景下，我们详细 examination and analysis了方法论中的每个关键要素。对于自监督学习，我们总结了常用的技术。文章在讨论潜在的研究方向的同时进行总结，旨在推动这一关键领域的技术进步。 

---
# Deployment-friendly Lane-changing Intention Prediction Powered by Brain-inspired Spiking Neural Networks 

**Title (ZH)**: 基于脑启发脉冲神经网络的部署friendly变道意图预测 

**Authors**: Junjie Yang, Shuqi Shen, Hui Zhong, Qiming Zhang, Hongliang Lu, Hai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.08659)  

**Abstract**: Accurate and real-time prediction of surrounding vehicles' lane-changing intentions is a critical challenge in deploying safe and efficient autonomous driving systems in open-world scenarios. Existing high-performing methods remain hard to deploy due to their high computational cost, long training times, and excessive memory requirements. Here, we propose an efficient lane-changing intention prediction approach based on brain-inspired Spiking Neural Networks (SNN). By leveraging the event-driven nature of SNN, the proposed approach enables us to encode the vehicle's states in a more efficient manner. Comparison experiments conducted on HighD and NGSIM datasets demonstrate that our method significantly improves training efficiency and reduces deployment costs while maintaining comparable prediction accuracy. Particularly, compared to the baseline, our approach reduces training time by 75% and memory usage by 99.9%. These results validate the efficiency and reliability of our method in lane-changing predictions, highlighting its potential for safe and efficient autonomous driving systems while offering significant advantages in deployment, including reduced training time, lower memory usage, and faster inference. 

**Abstract (ZH)**: 基于脑启发突触神经网络的准确实时周边车辆变道意图预测方法 

---
# Analyzable Parameters Dominated Vehicle Platoon Dynamics Modeling and Analysis: A Physics-Encoded Deep Learning Approach 

**Title (ZH)**: 可分析参数主导的车辆车队动力学建模与分析：一种物理编码深度学习方法 

**Authors**: Hao Lyu, Yanyong Guo, Pan Liu, Shuo Feng, Weilin Ren, Quansheng Yue  

**Link**: [PDF](https://arxiv.org/pdf/2502.08658)  

**Abstract**: Recently, artificial intelligence (AI)-enabled nonlinear vehicle platoon dynamics modeling plays a crucial role in predicting and optimizing the interactions between vehicles. Existing efforts lack the extraction and capture of vehicle behavior interaction features at the platoon scale. More importantly, maintaining high modeling accuracy without losing physical analyzability remains to be solved. To this end, this paper proposes a novel physics-encoded deep learning network, named PeMTFLN, to model the nonlinear vehicle platoon dynamics. Specifically, an analyzable parameters encoded computational graph (APeCG) is designed to guide the platoon to respond to the driving behavior of the lead vehicle while ensuring local stability. Besides, a multi-scale trajectory feature learning network (MTFLN) is constructed to capture platoon following patterns and infer the physical parameters required for APeCG from trajectory data. The human-driven vehicle trajectory datasets (HIGHSIM) were used to train the proposed PeMTFLN. The trajectories prediction experiments show that PeMTFLN exhibits superior compared to the baseline models in terms of predictive accuracy in speed and gap. The stability analysis result shows that the physical parameters in APeCG is able to reproduce the platoon stability in real-world condition. In simulation experiments, PeMTFLN performs low inference error in platoon trajectories generation. Moreover, PeMTFLN also accurately reproduces ground-truth safety statistics. The code of proposed PeMTFLN is open source. 

**Abstract (ZH)**: 近期，基于人工智能的非线性车辆编队动力学建模在预测和优化车辆间交互方面发挥着关键作用。现有研究表明，在编队规模上提取和捕获车辆行为交互特征仍存在不足。更重要的是，保持高建模准确性同时不失去物理可分析性的问题尚未解决。为解决上述问题，本文提出了一种新的物理编码深度学习网络——PeMTFLN，用于建模非线性车辆编队动力学。具体而言，设计了一种可分析参数编码计算图（APeCG），以引导编队响应领头车辆的驾驶行为，同时保证局部稳定性。此外，构建了多尺度轨迹特征学习网络（MTFLN），以捕获编队跟随模式并从轨迹数据中推断出APeCG所需的物理参数。采用人工驾驶车辆轨迹数据集（HIGHSIM）对提出的PeMTFLN进行训练。轨迹预测实验结果显示，PeMTFLN在速度和间距预测准确性方面优于基线模型。稳定性分析结果表明，APeCG中的物理参数能够再现真实世界条件下的编队稳定性。在仿真实验中，PeMTFLN在编队轨迹生成中的推理误差较低，并准确再现了地面真实的安全统计数据。提出的PeMTFLN代码开源。 

---
# Rolling Ahead Diffusion for Traffic Scene Simulation 

**Title (ZH)**: Rolling Ahead Diffusion 交通场景模拟 

**Authors**: Yunpeng Liu, Matthew Niedoba, William Harvey, Adam Scibior, Berend Zwartsenberg, Frank Wood  

**Link**: [PDF](https://arxiv.org/pdf/2502.09587)  

**Abstract**: Realistic driving simulation requires that NPCs not only mimic natural driving behaviors but also react to the behavior of other simulated agents. Recent developments in diffusion-based scenario generation focus on creating diverse and realistic traffic scenarios by jointly modelling the motion of all the agents in the scene. However, these traffic scenarios do not react when the motion of agents deviates from their modelled trajectories. For example, the ego-agent can be controlled by a stand along motion planner. To produce reactive scenarios with joint scenario models, the model must regenerate the scenario at each timestep based on new observations in a Model Predictive Control (MPC) fashion. Although reactive, this method is time-consuming, as one complete possible future for all NPCs is generated per simulation step. Alternatively, one can utilize an autoregressive model (AR) to predict only the immediate next-step future for all NPCs. Although faster, this method lacks the capability for advanced planning. We present a rolling diffusion based traffic scene generation model which mixes the benefits of both methods by predicting the next step future and simultaneously predicting partially noised further future steps at the same time. We show that such model is efficient compared to diffusion model based AR, achieving a beneficial compromise between reactivity and computational efficiency. 

**Abstract (ZH)**: 基于滚动扩散的交通场景生成模型：混合高效预测方法 

---
# A Survey of Reinforcement Learning for Optimization in Automation 

**Title (ZH)**: 自动化中强化学习优化综述 

**Authors**: Ahmad Farooq, Kamran Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2502.09417)  

**Abstract**: Reinforcement Learning (RL) has become a critical tool for optimization challenges within automation, leading to significant advancements in several areas. This review article examines the current landscape of RL within automation, with a particular focus on its roles in manufacturing, energy systems, and robotics. It discusses state-of-the-art methods, major challenges, and upcoming avenues of research within each sector, highlighting RL's capacity to solve intricate optimization challenges. The paper reviews the advantages and constraints of RL-driven optimization methods in automation. It points out prevalent challenges encountered in RL optimization, including issues related to sample efficiency and scalability; safety and robustness; interpretability and trustworthiness; transfer learning and meta-learning; and real-world deployment and integration. It further explores prospective strategies and future research pathways to navigate these challenges. Additionally, the survey includes a comprehensive list of relevant research papers, making it an indispensable guide for scholars and practitioners keen on exploring this domain. 

**Abstract (ZH)**: reinforcement learning (RL) 已成为自动化领域优化挑战的关键工具，推动了多个领域的重大进展。本文综述了 RL 在自动化领域的当前格局，特别聚焦于其在制造、能源系统和机器人领域的角色。文章讨论了每个领域的最新方法、主要挑战和未来研究方向，突出了 RL 解决复杂优化挑战的能力。论文回顾了 RL 驱动的优化方法在自动化领域的优势与限制。指出了 RL 优化过程中遇到的常见挑战，包括样本效率和可扩展性问题；安全性和鲁棒性问题；可解释性和可靠性问题；迁移学习和元学习问题；以及实际部署和集成问题。进一步探讨了应对这些挑战的潜在策略和未来研究路径。此外，调研还包括了一篇相关的研究论文综述列表，成为对该领域感兴趣的研究人员和实践者的必备指南。 

---
# A Deep Inverse-Mapping Model for a Flapping Robotic Wing 

**Title (ZH)**: 一种摆动机器人翅膀的深度逆映射模型 

**Authors**: Hadar Sharvit, Raz Karl, Tsevi Beatus  

**Link**: [PDF](https://arxiv.org/pdf/2502.09378)  

**Abstract**: In systems control, the dynamics of a system are governed by modulating its inputs to achieve a desired outcome. For example, to control the thrust of a quad-copter propeller the controller modulates its rotation rate, relying on a straightforward mapping between the input rotation rate and the resulting thrust. This mapping can be inverted to determine the rotation rate needed to generate a desired thrust. However, in complex systems, such as flapping-wing robots where intricate fluid motions are involved, mapping inputs (wing kinematics) to outcomes (aerodynamic forces) is nontrivial and inverting this mapping for real-time control is computationally impractical. Here, we report a machine-learning solution for the inverse mapping of a flapping-wing system based on data from an experimental system we have developed. Our model learns the input wing motion required to generate a desired aerodynamic force outcome. We used a sequence-to-sequence model tailored for time-series data and augmented it with a novel adaptive-spectrum layer that implements representation learning in the frequency domain. To train our model, we developed a flapping wing system that simultaneously measures the wing's aerodynamic force and its 3D motion using high-speed cameras. We demonstrate the performance of our system on an additional open-source dataset of a flapping wing in a different flow regime. Results show superior performance compared with more complex state-of-the-art transformer-based models, with 11% improvement on the test datasets median loss. Moreover, our model shows superior inference time, making it practical for onboard robotic control. Our open-source data and framework may improve modeling and real-time control of systems governed by complex dynamics, from biomimetic robots to biomedical devices. 

**Abstract (ZH)**: 基于数据驱动的扑翼系统逆映射的机器学习解决方案 

---
# Moving Matter: Efficient Reconfiguration of Tile Arrangements by a Single Active Robot 

**Title (ZH)**: 移动物质：单个活性机器人高效重构砖块排列的方法 

**Authors**: Aaron T. Becker, Sándor P. Fekete, Jonas Friemel, Ramin Kosfeld, Peter Kramer, Harm Kube, Christian Rieck, Christian Scheffer, Arne Schmidt  

**Link**: [PDF](https://arxiv.org/pdf/2502.09299)  

**Abstract**: We consider the problem of reconfiguring a two-dimensional connected grid arrangement of passive building blocks from a start configuration to a goal configuration, using a single active robot that can move on the tiles, remove individual tiles from a given location and physically move them to a new position by walking on the remaining configuration. The objective is to determine a reconfiguration schedule that minimizes the overall makespan, while ensuring that the tile configuration remains connected. We provide both negative and positive results. (1) We present a generalized version of the problem, parameterized by weighted costs for moving with or without tiles, and show that this is NP-complete. (2) We give a polynomial-time constant-factor approximation algorithm for the case of disjoint start and target bounding boxes. In addition, our approach yields optimal carry distance for 2-scaled instances. 

**Abstract (ZH)**: 二维连接网格Arrange-to-Make-Span问题的重构研究：具有单个活性机器人在不连接状态下移动和重新定位个体单元格的配置优化 

---
# LLM-Driven Augmented Reality Puppeteer: Controller-Free Voice-Commanded Robot Teleoperation 

**Title (ZH)**: LLM驱动的增强现实傀儡师：无控制器语音指令的机器人远程操作 

**Authors**: Yuchong Zhang, Bastian Orthmann, Michael C. Welle, Jonne Van Haastregt, Danica Kragic  

**Link**: [PDF](https://arxiv.org/pdf/2502.09142)  

**Abstract**: The integration of robotics and augmented reality (AR) presents transformative opportunities for advancing human-robot interaction (HRI) by improving usability, intuitiveness, and accessibility. This work introduces a controller-free, LLM-driven voice-commanded AR puppeteering system, enabling users to teleoperate a robot by manipulating its virtual counterpart in real time. By leveraging natural language processing (NLP) and AR technologies, our system -- prototyped using Meta Quest 3 -- eliminates the need for physical controllers, enhancing ease of use while minimizing potential safety risks associated with direct robot operation. A preliminary user demonstration successfully validated the system's functionality, demonstrating its potential for safer, more intuitive, and immersive robotic control. 

**Abstract (ZH)**: 机器人与增强现实（AR）的集成为人类-机器人交互（HRI）的进步提供了变革性机会，通过提高易用性、直观性和可访问性。本工作介绍了一种基于语言模型（LLM）驱动的无需控制器、通过语音命令操控的AR puppeteering系统，使用户能够实时操控机器人并通过操控其虚拟对应物进行远程操作。通过利用自然语言处理（NLP）和AR技术，我们的系统（基于Meta Quest 3原型）消除了物理控制器的需求，增强了易用性并降低了直接操作机器人可能带来的安全风险。初步用户演示成功验证了系统的功能，展示了其在更安全、更直观和更具沉浸感的机器人控制方面的潜力。 

---
