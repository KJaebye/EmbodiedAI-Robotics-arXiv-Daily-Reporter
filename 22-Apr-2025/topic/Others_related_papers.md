# Neural ATTF: A Scalable Solution to Lifelong Multi-Agent Path Planning 

**Title (ZH)**: 神经ATTF：面向终身多agent路径规划的可扩展解决方案 

**Authors**: Kushal Shah, Jihyun Park, Seung-Kyum Choi  

**Link**: [PDF](https://arxiv.org/pdf/2504.15130)  

**Abstract**: Multi-Agent Pickup and Delivery (MAPD) is a fundamental problem in robotics, particularly in applications such as warehouse automation and logistics. Existing solutions often face challenges in scalability, adaptability, and efficiency, limiting their applicability in dynamic environments with real-time planning requirements. This paper presents Neural ATTF (Adaptive Task Token Framework), a new algorithm that combines a Priority Guided Task Matching (PGTM) Module with Neural STA* (Space-Time A*), a data-driven path planning method. Neural STA* enhances path planning by enabling rapid exploration of the search space through guided learned heuristics and ensures collision avoidance under dynamic constraints. PGTM prioritizes delayed agents and dynamically assigns tasks by prioritizing agents nearest to these tasks, optimizing both continuity and system throughput. Experimental evaluations against state-of-the-art MAPD algorithms, including TPTS, CENTRAL, RMCA, LNS-PBS, and LNS-wPBS, demonstrate the superior scalability, solution quality, and computational efficiency of Neural ATTF. These results highlight the framework's potential for addressing the critical demands of complex, real-world multi-agent systems operating in high-demand, unpredictable settings. 

**Abstract (ZH)**: 多代理拣取与配送（MAPD）是机器人领域的一个基本问题，特别是在仓储自动化和物流等领域应用广泛。现有解决方案在可扩展性、适应性和效率方面常常面临挑战，限制了它们在具有实时规划要求的动态环境中的应用。本文提出了一种新的算法——神经ATTF（自适应任务标记框架），该算法结合了优先级引导任务匹配（PGTM）模块和基于数据的路径规划方法神经STA*。神经STA*通过引导学习启发式快速探索搜索空间，并在动态约束条件下确保避障。PGTM优先处理延迟的代理，并动态分配任务，通过优先处理最近这些任务的代理来优化连续性和系统吞吐量。与当前最先进的MAPD算法TPTS、CENTRAL、RMCA、LNS-PBS和LNS-wPBS的实验对比表明，神经ATTF在可扩展性、解的质量和计算效率方面表现出优越性。这些结果突显了该框架在处理高需求、不可预测环境下的复杂多代理系统关键需求方面的潜力。 

---
# FERMI: Flexible Radio Mapping with a Hybrid Propagation Model and Scalable Autonomous Data Collection 

**Title (ZH)**: FERMI: 灵活的无线电地图构建方法结合混合传播模型和可扩展的自主数据收集 

**Authors**: Yiming Luo, Yunfei Wang, Hongming Chen, Chengkai Wu, Ximin Lyu, Jinni Zhou, Jun Ma, Fu Zhang, Boyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.14862)  

**Abstract**: Communication is fundamental for multi-robot collaboration, with accurate radio mapping playing a crucial role in predicting signal strength between robots. However, modeling radio signal propagation in large and occluded environments is challenging due to complex interactions between signals and obstacles. Existing methods face two key limitations: they struggle to predict signal strength for transmitter-receiver pairs not present in the training set, while also requiring extensive manual data collection for modeling, making them impractical for large, obstacle-rich scenarios. To overcome these limitations, we propose FERMI, a flexible radio mapping framework. FERMI combines physics-based modeling of direct signal paths with a neural network to capture environmental interactions with radio signals. This hybrid model learns radio signal propagation more efficiently, requiring only sparse training data. Additionally, FERMI introduces a scalable planning method for autonomous data collection using a multi-robot team. By increasing parallelism in data collection and minimizing robot travel costs between regions, overall data collection efficiency is significantly improved. Experiments in both simulation and real-world scenarios demonstrate that FERMI enables accurate signal prediction and generalizes well to unseen positions in complex environments. It also supports fully autonomous data collection and scales to different team sizes, offering a flexible solution for creating radio maps. Our code is open-sourced at this https URL. 

**Abstract (ZH)**: 基于物理学的建模与神经网络结合的灵活无线地图构建框架 

---
# A Complete and Bounded-Suboptimal Algorithm for a Moving Target Traveling Salesman Problem with Obstacles in 3D 

**Title (ZH)**: 一个完整的且在3D中有障碍物的移动目标旅行商问题的有界次优算法 

**Authors**: Anoop Bhat, Geordan Gutow, Bhaskar Vundurthy, Zhongqiang Ren, Sivakumar Rathinam, Howie Choset  

**Link**: [PDF](https://arxiv.org/pdf/2504.14680)  

**Abstract**: The moving target traveling salesman problem with obstacles (MT-TSP-O) seeks an obstacle-free trajectory for an agent that intercepts a given set of moving targets, each within specified time windows, and returns to the agent's starting position. Each target moves with a constant velocity within its time windows, and the agent has a speed limit no smaller than any target's speed. We present FMC*-TSP, the first complete and bounded-suboptimal algorithm for the MT-TSP-O, and results for an agent whose configuration space is $\mathbb{R}^3$. Our algorithm interleaves a high-level search and a low-level search, where the high-level search solves a generalized traveling salesman problem with time windows (GTSP-TW) to find a sequence of targets and corresponding time windows for the agent to visit. Given such a sequence, the low-level search then finds an associated agent trajectory. To solve the low-level planning problem, we develop a new algorithm called FMC*, which finds a shortest path on a graph of convex sets (GCS) via implicit graph search and pruning techniques specialized for problems with moving targets. We test FMC*-TSP on 280 problem instances with up to 40 targets and demonstrate its smaller median runtime than a baseline based on prior work. 

**Abstract (ZH)**: 带障碍的移动目标旅行商问题（MT-TSP-O）寻求一条无障碍路径，使代理拦截一组在指定时间窗口内移动的目标，并返回代理的起始位置。每个目标在时间窗口内以恒定速度移动，代理的速度限制不低于任何目标的速度。我们提出了第一个完整的且近乎最优的算法FMC*-TSP，该算法适用于配置空间为$\mathbb{R}^3$的代理，并解决了包含移动目标的一系列规划问题。该算法交替进行高层次搜索和低层次搜索，高层次搜索求解带时间窗口的一般化旅行商问题（GTSP-TW），以找到代理访问的目标序列及其对应的时间窗口。给定这样一个序列，低层次搜索则寻找相应的代理路径。为解决低层次规划问题，我们开发了一种新的算法FMC*，该算法通过特定于移动目标问题的隐式图搜索和修剪技术，在凸集图（GCS）上寻找最短路径。我们对最多包含40个目标的280个问题实例测试了FMC*-TSP，并展示了其中位数运行时间比基于先前工作的基线算法更短。 

---
# SG-Reg: Generalizable and Efficient Scene Graph Registration 

**Title (ZH)**: SG-Reg: 具有普遍适用性和高效性的场景图对齐方法 

**Authors**: Chuhao Liu, Zhijian Qiao, Jieqi Shi, Ke Wang, Peize Liu, Shaojie Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.14440)  

**Abstract**: This paper addresses the challenges of registering two rigid semantic scene graphs, an essential capability when an autonomous agent needs to register its map against a remote agent, or against a prior map. The hand-crafted descriptors in classical semantic-aided registration, or the ground-truth annotation reliance in learning-based scene graph registration, impede their application in practical real-world environments. To address the challenges, we design a scene graph network to encode multiple modalities of semantic nodes: open-set semantic feature, local topology with spatial awareness, and shape feature. These modalities are fused to create compact semantic node features. The matching layers then search for correspondences in a coarse-to-fine manner. In the back-end, we employ a robust pose estimator to decide transformation according to the correspondences. We manage to maintain a sparse and hierarchical scene representation. Our approach demands fewer GPU resources and fewer communication bandwidth in multi-agent tasks. Moreover, we design a new data generation approach using vision foundation models and a semantic mapping module to reconstruct semantic scene graphs. It differs significantly from previous works, which rely on ground-truth semantic annotations to generate data. We validate our method in a two-agent SLAM benchmark. It significantly outperforms the hand-crafted baseline in terms of registration success rate. Compared to visual loop closure networks, our method achieves a slightly higher registration recall while requiring only 52 KB of communication bandwidth for each query frame. Code available at: \href{this http URL}{this http URL}. 

**Abstract (ZH)**: 本文解决了两个刚性语义场景图注册的挑战，这是当自主代理需要将其地图与远程代理或先验地图进行注册时所需的基本能力。经典的语义辅助注册中的手工设计描述符，以及基于学习的场景图注册中对地面truth标注的依赖，在实际现实环境中限制了它们的应用。为应对这些挑战，我们设计了一个场景图网络来编码语义节点的多种模态：开放式语义特征、局部拓扑结构以及空间意识和形状特征。这些模态融合生成紧凑的语义节点特征。匹配层随后以粗到细的方式搜索对应关系。在后端，我们采用鲁棒的姿态估计器根据对应关系决定变换。我们能够保持稀疏且分层的场景表示。我们的方法在多代理任务中需要较少的GPU资源和较少的通信带宽。此外，我们设计了一种新的数据生成方法，利用视觉基础模型和语义映射模块重建语义场景图。这种方法与以往依赖地面truth语义标注生成数据的方法显著不同。我们在一个双代理SLAM基准中验证了我们的方法，其在注册成功率方面显著优于手工设计的基线。与视觉环视闭合网络相比，我们的方法在注册召回率上略有提高，同时每次查询帧只需要52 KB的通信带宽。代码可在以下链接获取：[this http URL]。 

---
# MILUV: A Multi-UAV Indoor Localization dataset with UWB and Vision 

**Title (ZH)**: MILUV: 一种基于UWB和视觉的多无人机室内定位数据集 

**Authors**: Mohammed Ayman Shalaby, Syed Shabbir Ahmed, Nicholas Dahdah, Charles Champagne Cossette, Jerome Le Ny, James Richard Forbes  

**Link**: [PDF](https://arxiv.org/pdf/2504.14376)  

**Abstract**: This paper introduces MILUV, a Multi-UAV Indoor Localization dataset with UWB and Vision measurements. This dataset comprises 217 minutes of flight time over 36 experiments using three quadcopters, collecting ultra-wideband (UWB) ranging data such as the raw timestamps and channel-impulse response data, vision data from a stereo camera and a bottom-facing monocular camera, inertial measurement unit data, height measurements from a laser rangefinder, magnetometer data, and ground-truth poses from a motion-capture system. The UWB data is collected from up to 12 transceivers affixed to mobile robots and static tripods in both line-of-sight and non-line-of-sight conditions. The UAVs fly at a maximum speed of 4.418 m/s in an indoor environment with visual fiducial markers as features. MILUV is versatile and can be used for a wide range of applications beyond localization, but the primary purpose of MILUV is for testing and validating multi-robot UWB- and vision-based localization algorithms. The dataset can be downloaded at this https URL. A development kit is presented alongside the MILUV dataset, which includes benchmarking algorithms such as visual-inertial odometry, UWB-based localization using an extended Kalman filter, and classification of CIR data using machine learning approaches. The development kit can be found at this https URL, and is supplemented with a website available at this https URL. 

**Abstract (ZH)**: MILUV：一种基于UWB和视觉的多无人机室内定位数据集 

---
# Task Matters: Investigating Human Questioning Behavior in Different Household Service for Learning by Asking Robots 

**Title (ZH)**: 任务不同，提问行为有异：探究不同家庭服务任务中机器人学习中的提问行为 

**Authors**: Yuanda Hu, Hou Jiani, Zhang Junyu, Yate Ge, Xiaohua Sun, Weiwei Guo  

**Link**: [PDF](https://arxiv.org/pdf/2504.13916)  

**Abstract**: Learning by Asking (LBA) enables robots to identify knowledge gaps during task execution and acquire the missing information by asking targeted questions. However, different tasks often require different types of questions, and how to adapt questioning strategies accordingly remains underexplored. This paper investigates human questioning behavior in two representative household service tasks: a Goal-Oriented task (refrigerator organization) and a Process-Oriented task (cocktail mixing). Through a human-human study involving 28 participants, we analyze the questions asked using a structured framework that encodes each question along three dimensions: acquired knowledge, cognitive process, and question form. Our results reveal that participants adapt both question types and their temporal ordering based on task structure. Goal-Oriented tasks elicited early inquiries about user preferences, while Process-Oriented tasks led to ongoing, parallel questioning of procedural steps and preferences. These findings offer actionable insights for developing task-sensitive questioning strategies in LBA-enabled robots for more effective and personalized human-robot collaboration. 

**Abstract (ZH)**: 学习提问（LBA）使机器人能够在执行任务时识别知识缺口，并通过提出针对性的问题来获取缺失的信息。然而，不同任务往往需要不同类型的问题，如何相应地调整提问策略仍待探索。本文在两个代表性家庭服务任务中研究了人类的提问行为：目标导向任务（冰箱整理）和过程导向任务（调制鸡尾酒）。通过涉及28名参与者的实验研究，我们使用结构化的框架来分析每个问题在三个维度上的编码：获取的知识、认知过程和问题形式。研究结果表明，参与者根据任务结构调整了提问类型及其时间顺序。目标导向任务引发了关于用户偏好的早期询问，而过程导向任务则导致了关于程序步骤和偏好的持续并行提问。这些发现为在LBA使能的机器人中开发更有效和个性化的任务敏感提问策略提供了可操作的见解。 

---
# V2P Collision Warnings for Distracted Pedestrians: A Comparative Study with Traditional Auditory Alerts 

**Title (ZH)**: 面向分心行人的V2P碰撞警告：与传统听觉警报的比较研究 

**Authors**: Novel Certad, Enrico Del Re, Joshua Varughese, Cristina Olaverri-Monreal  

**Link**: [PDF](https://arxiv.org/pdf/2504.13906)  

**Abstract**: This study assesses a Vehicle-to-Pedestrian (V2P) collision warning system compared to conventional vehicle-issued auditory alerts in a real-world scenario simulating a vehicle on a fixed track, characterized by limited maneuverability and the need for timely pedestrian response. The results from analyzing speed variations show that V2P warnings are particularly effective for pedestrians distracted by phone use (gaming or listening to music), highlighting the limitations of auditory alerts in noisy environments. The findings suggest that V2P technology offers a promising approach to improving pedestrian safety in urban areas 

**Abstract (ZH)**: 本研究评估了一种Vehicle-to-Pedestrian（V2P）碰撞预警系统在实际场景中的效果，该场景模拟了车辆在固定轨道上行驶的情况，具有有限的机动性和及时行人响应的需求。分析速度变化的结果表明，V2P警告特别适用于分散了注意力的手机使用者（如玩游戏或听音乐），突显了噪音环境中听觉警报的局限性。研究发现，V2P技术为改善城市区域的行人安全提供了有前景的方法。 

---
# Skeleton-Based Transformer for Classification of Errors and Better Feedback in Low Back Pain Physical Rehabilitation Exercises 

**Title (ZH)**: 基于骨架的变压器在低背部疼痛物理康复练习中错误分类与更好反馈的研究 

**Authors**: Aleksa Marusic, Sao Mai Nguyen, Adriana Tapus  

**Link**: [PDF](https://arxiv.org/pdf/2504.13866)  

**Abstract**: Physical rehabilitation exercises suggested by healthcare professionals can help recovery from various musculoskeletal disorders and prevent re-injury. However, patients' engagement tends to decrease over time without direct supervision, which is why there is a need for an automated monitoring system. In recent years, there has been great progress in quality assessment of physical rehabilitation exercises. Most of them only provide a binary classification if the performance is correct or incorrect, and a few provide a continuous score. This information is not sufficient for patients to improve their performance. In this work, we propose an algorithm for error classification of rehabilitation exercises, thus making the first step toward more detailed feedback to patients. We focus on skeleton-based exercise assessment, which utilizes human pose estimation to evaluate motion. Inspired by recent algorithms for quality assessment during rehabilitation exercises, we propose a Transformer-based model for the described classification. Our model is inspired by the HyperFormer method for human action recognition, and adapted to our problem and dataset. The evaluation is done on the KERAAL dataset, as it is the only medical dataset with clear error labels for the exercises, and our model significantly surpasses state-of-the-art methods. Furthermore, we bridge the gap towards better feedback to the patients by presenting a way to calculate the importance of joints for each exercise. 

**Abstract (ZH)**: 自动监测系统建议的物理康复锻炼对于各种肌肉骨骼疾病的恢复和防止再次受伤有益。然而，在缺乏直接监督的情况下，患者参与度往往会随着时间的推移而下降，因此需要一个自动监测系统。近年来，在物理康复锻炼的质量评估方面取得了很大进展。大多数方法仅提供二元分类，即表现正确或错误，少数方法提供连续评分。这些信息对于患者提高表现并不充分。在这项工作中，我们提出了一种康复锻炼错误分类算法，从而迈出了为患者提供更详细反馈的第一步。我们重点关注基于骨架的锻炼评估，利用人体姿态估计来评估动作。受康复锻炼质量评估最新算法的启发，我们提出了一种基于Transformer的模型来实现描述的分类。我们的模型受到HyperFormer方法用于人类动作识别的启发，并针对我们的问题和数据集进行了调整。评估在KERAAL数据集上进行，因为它是唯一一个具有明确锻炼错误标签的医疗数据集，我们的模型显著超过了最先进的方法。此外，我们通过提出一种计算每个锻炼关节重要性的方法，进一步缩小了向患者提供更好反馈的差距。 

---
# FlowReasoner: Reinforcing Query-Level Meta-Agents 

**Title (ZH)**: FlowReasoner: 强化查询级别元代理 

**Authors**: Hongcheng Gao, Yue Liu, Yufei He, Longxu Dou, Chao Du, Zhijie Deng, Bryan Hooi, Min Lin, Tianyu Pang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15257)  

**Abstract**: This paper proposes a query-level meta-agent named FlowReasoner to automate the design of query-level multi-agent systems, i.e., one system per user query. Our core idea is to incentivize a reasoning-based meta-agent via external execution feedback. Concretely, by distilling DeepSeek R1, we first endow the basic reasoning ability regarding the generation of multi-agent systems to FlowReasoner. Then, we further enhance it via reinforcement learning (RL) with external execution feedback. A multi-purpose reward is designed to guide the RL training from aspects of performance, complexity, and efficiency. In this manner, FlowReasoner is enabled to generate a personalized multi-agent system for each user query via deliberative reasoning. Experiments on both engineering and competition code benchmarks demonstrate the superiority of FlowReasoner. Remarkably, it surpasses o1-mini by 10.52% accuracy across three benchmarks. The code is available at this https URL. 

**Abstract (ZH)**: 本文提出了一种查询级元代理FlowReasoner，用于自动化查询级多代理系统的设计，即每个用户查询一个系统。我们的核心思想是通过外部执行反馈激励基于推理的元代理。具体而言，通过精炼DeepSeek R1，我们首先为FlowReasoner赋予了生成多代理系统的基本推理能力，然后进一步通过强化学习（RL）和外部执行反馈对其进行增强。设计了一个多用途奖励来从性能、复杂性和效率方面指导RL训练。通过这种方式，FlowReasoner能够通过审慎推理为每个用户查询生成个性化多代理系统。在工程和竞赛代码基准上的实验展示了FlowReasoner的优势。值得注意的是，它在三个基准上的准确率比o1-mini高10.52%。代码可在以下链接获取。 

---
# SuoiAI: Building a Dataset for Aquatic Invertebrates in Vietnam 

**Title (ZH)**: SuoiAI: 建立越南水生无脊椎动物数据集 

**Authors**: Tue Vo, Lakshay Sharma, Tuan Dinh, Khuong Dinh, Trang Nguyen, Trung Phan, Minh Do, Duong Vu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15252)  

**Abstract**: Understanding and monitoring aquatic biodiversity is critical for ecological health and conservation efforts. This paper proposes SuoiAI, an end-to-end pipeline for building a dataset of aquatic invertebrates in Vietnam and employing machine learning (ML) techniques for species classification. We outline the methods for data collection, annotation, and model training, focusing on reducing annotation effort through semi-supervised learning and leveraging state-of-the-art object detection and classification models. Our approach aims to overcome challenges such as data scarcity, fine-grained classification, and deployment in diverse environmental conditions. 

**Abstract (ZH)**: 理解与监测水生生物多样性对于生态健康和保护工作至关重要。本文提出了一种端到端的管道SuoiAI，用于构建越南水生无脊椎动物的数据集，并采用机器学习技术进行物种分类。我们概述了数据收集、标注和模型训练的方法，重点关注通过半监督学习减少标注努力，并利用最先进的对象检测和分类模型。我们的方法旨在克服数据稀缺、精细分类以及在多种环境条件下部署的挑战。 

---
# Position: Bayesian Statistics Facilitates Stakeholder Participation in Evaluation of Generative AI 

**Title (ZH)**: 位置：贝叶斯统计促进生成式人工智能评价中的利益相关者参与 

**Authors**: Yanan Long  

**Link**: [PDF](https://arxiv.org/pdf/2504.15211)  

**Abstract**: The evaluation of Generative AI (GenAI) systems plays a critical role in public policy and decision-making, yet existing methods are often limited by reliance on benchmark-driven, point-estimate comparisons that fail to capture uncertainty and broader societal impacts. This paper argues for the use of Bayesian statistics as a principled framework to address these challenges. Bayesian methods enable the integration of domain expertise through prior elicitation, allow for continuous learning from new data, and provide robust uncertainty quantification via posterior inference. We demonstrate how Bayesian inference can be applied to GenAI evaluation, particularly in incorporating stakeholder perspectives to enhance fairness, transparency, and reliability. Furthermore, we discuss Bayesian workflows as an iterative process for model validation and refinement, ensuring robust assessments of GenAI systems in dynamic, real-world contexts. 

**Abstract (ZH)**: 基于贝叶斯统计的方法在评估生成式人工智能系统中的应用：克服现有方法的局限以应对公共政策和决策中的挑战 

---
# Behavioral Universe Network (BUN): A Behavioral Information-Based Framework for Complex Systems 

**Title (ZH)**: 行为宇宙网络(BUN):一种基于行为信息的复杂系统框架 

**Authors**: Wei Zhou, Ailiya Borjigin, Cong He  

**Link**: [PDF](https://arxiv.org/pdf/2504.15146)  

**Abstract**: Modern digital ecosystems feature complex, dynamic interactions among autonomous entities across diverse domains. Traditional models often separate agents and objects, lacking a unified foundation to capture their interactive behaviors. This paper introduces the Behavioral Universe Network (BUN), a theoretical framework grounded in the Agent-Interaction-Behavior (AIB) formalism. BUN treats subjects (active agents), objects (resources), and behaviors (operations) as first-class entities, all governed by a shared Behavioral Information Base (BIB). We detail the AIB core concepts and demonstrate how BUN leverages information-driven triggers, semantic enrichment, and adaptive rules to coordinate multi-agent systems. We highlight key benefits: enhanced behavior analysis, strong adaptability, and cross-domain interoperability. We conclude by positioning BUN as a promising foundation for next-generation digital governance and intelligent applications. 

**Abstract (ZH)**: 现代数字生态系统特征在于跨多个领域中自主实体之间的复杂动态交互。传统模型常常将代理和对象分开，缺乏一个统一的基础来捕捉它们的交互行为。本文介绍了一种基于代理-交互-行为（AIB）形式主义的理论框架——行为宇宙网络（BUN）。BUN将主体（活跃的代理）、对象（资源）和行为（操作）视为一级实体，并均由共享的行为信息库（BIB）管理。我们详细阐述了AIB的核心概念，并展示了BUN如何利用信息驱动的触发机制、语义增强和适应性规则来协调多代理系统。我们强调了BUN的关键优势：增强的行为分析能力、强大的适应性和跨领域的互操作性。最后，我们将BUN定位为下一代数字治理和智能应用的有前途的基础。 

---
# Mitigating Degree Bias in Graph Representation Learning with Learnable Structural Augmentation and Structural Self-Attention 

**Title (ZH)**: 使用可学习的结构增强和结构自注意力减轻图表示学习中的度偏差 

**Authors**: Van Thuy Hoang, Hyeon-Ju Jeon, O-Joun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.15075)  

**Abstract**: Graph Neural Networks (GNNs) update node representations through message passing, which is primarily based on the homophily principle, assuming that adjacent nodes share similar features. However, in real-world graphs with long-tailed degree distributions, high-degree nodes dominate message passing, causing a degree bias where low-degree nodes remain under-represented due to inadequate messages. The main challenge in addressing degree bias is how to discover non-adjacent nodes to provide additional messages to low-degree nodes while reducing excessive messages for high-degree nodes. Nevertheless, exploiting non-adjacent nodes to provide valuable messages is challenging, as it could generate noisy information and disrupt the original graph structures. To solve it, we propose a novel Degree Fairness Graph Transformer, named DegFairGT, to mitigate degree bias by discovering structural similarities between non-adjacent nodes through learnable structural augmentation and structural self-attention. Our key idea is to exploit non-adjacent nodes with similar roles in the same community to generate informative edges under our augmentation, which could provide informative messages between nodes with similar roles while ensuring that the homophily principle is maintained within the community. To enable DegFairGT to learn such structural similarities, we then propose a structural self-attention to capture the similarities between node pairs. To preserve global graph structures and prevent graph augmentation from hindering graph structure, we propose a Self-Supervised Learning task to preserve p-step transition probability and regularize graph augmentation. Extensive experiments on six datasets showed that DegFairGT outperformed state-of-the-art baselines in degree fairness analysis, node classification, and node clustering tasks. 

**Abstract (ZH)**: Degree公平图变换器（DegFairGT）：通过发现非相邻节点的结构性相似性来减轻度偏差 

---
# AlignRAG: An Adaptable Framework for Resolving Misalignments in Retrieval-Aware Reasoning of RAG 

**Title (ZH)**: AlignRAG: 一种用于解决RAG检索aware推理中不一致性的可适应框架 

**Authors**: Jiaqi Wei, Hao Zhou, Xiang Zhang, Di Zhang, Zijie Qiu, Wei Wei, Jinzhe Li, Wanli Ouyang, Siqi Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.14858)  

**Abstract**: Retrieval-augmented generation (RAG) has emerged as a foundational paradigm for knowledge-grounded text generation. However, existing RAG pipelines often fail to ensure that the reasoning trajectories align with the evidential constraints imposed by retrieved content. In this paper, we reframe RAG as a problem of retrieval-aware reasoning and identify a core challenge: reasoning misalignment-the mismatch between a model's reasoning trajectory and the retrieved evidence. To address this challenge, we propose AlignRAG, a novel test-time framework that mitigates reasoning misalignment through iterative Critique-Driven Alignment (CDA) steps. In contrast to prior approaches that rely on static training or post-hoc selection, AlignRAG actively refines reasoning trajectories during inference by enforcing fine-grained alignment with evidence. Our framework introduces a new paradigm for retrieval-aware reasoning by: (1) constructing context-rich training corpora; (2) generating contrastive critiques from preference-aware reasoning trajectories; (3) training a dedicated \textit{Critic Language Model (CLM)} to identify reasoning misalignments; and (4) applying CDA steps to optimize reasoning trajectories iteratively. Empirical results demonstrate that AlignRAG consistently outperforms all baselines and could integrate as a plug-and-play module into existing RAG pipelines without further changes. By reconceptualizing RAG as a structured reasoning trajectory and establishing the test-time framework for correcting reasoning misalignments in RAG, AlignRAG provides practical advancements for retrieval-aware generation. 

**Abstract (ZH)**: 检索增强生成（RAG）已成为基于知识的文本生成的基础范式。然而，现有的RAG流水线往往无法确保推理轨迹与检索内容施加的证据约束保持一致。在这个论文中，我们将RAG重新框架化为一种检索意识推理问题，并识别出一个核心挑战：推理错位——模型的推理轨迹与检索证据之间的不匹配。为了解决这一挑战，我们提出了AlignRAG，这是一种新颖的测试时框架，通过迭代的质疑驱动校准（CDA）步骤来缓解推理错位。与依赖于静态训练或事后选择的先前方法不同，AlignRAG在推断过程中主动通过细粒度证据校准来精炼推理轨迹。我们的框架通过以下方式引入了检索意识推理的新范式：（1）构建丰富的上下文训练语料库；（2）从偏好意识推理轨迹中生成对比性批评；（3）训练专用的批评语言模型（CLM）以识别推理错位；（4）应用CDA步骤以迭代优化推理轨迹。实验证明，AlignRAG始终优于所有基线，在无需进一步修改的情况下可以无缝集成到现有的RAG流水线中。通过将RAG重新构想为有结构的推理轨迹，并建立纠正RAG中推理错位的测试时框架，AlignRAG为检索意识生成提供了实用的进步。 

---
# Consensus in Motion: A Case of Dynamic Rationality of Sequential Learning in Probability Aggregation 

**Title (ZH)**: 共识在motion：概率聚合中序贯学习动态理性案例研究 

**Authors**: Polina Gordienko, Christoph Jansen, Thomas Augustin, Martin Rechenauer  

**Link**: [PDF](https://arxiv.org/pdf/2504.14624)  

**Abstract**: We propose a framework for probability aggregation based on propositional probability logic. Unlike conventional judgment aggregation, which focuses on static rationality, our model addresses dynamic rationality by ensuring that collective beliefs update consistently with new information. We show that any consensus-compatible and independent aggregation rule on a non-nested agenda is necessarily linear. Furthermore, we provide sufficient conditions for a fair learning process, where individuals initially agree on a specified subset of propositions known as the common ground, and new information is restricted to this shared foundation. This guarantees that updating individual judgments via Bayesian conditioning-whether performed before or after aggregation-yields the same collective belief. A distinctive feature of our framework is its treatment of sequential decision-making, which allows new information to be incorporated progressively through multiple stages while maintaining the established common ground. We illustrate our findings with a running example in a political scenario concerning healthcare and immigration policies. 

**Abstract (ZH)**: 我们提出了一种基于命题概率逻辑的概率聚合框架。不同于侧重静态理性的传统判断聚合，我们的模型通过确保集体信念在获得新信息后能一致更新，来处理动态理性。我们证明，对于非嵌套议程上的任何共识兼容且独立的聚合规则，必定是线性的。此外，我们提供了确保公平学习过程的充分条件，在这种过程中，个体最初在一组特定命题上达成共识，这些命题被称为共同基础，而新的信息仅限于这一共享基础之上。这保证了通过贝叶斯条件化更新个体判断——无论是聚合前还是聚合后——都能得出相同的集体信念。我们的框架的一个独特之处在于其对顺序决策的处理，这使新的信息可以通过多个阶段逐步融入，同时保持已建立的共同基础。我们通过一个关于医疗政策和移民政策的政治场景实例来说明这些发现。 

---
# UFO2: The Desktop AgentOS 

**Title (ZH)**: UFO2：桌面代理操作系统 

**Authors**: Chaoyun Zhang, He Huang, Chiming Ni, Jian Mu, Si Qin, Shilin He, Lu Wang, Fangkai Yang, Pu Zhao, Chao Du, Liqun Li, Yu Kang, Zhao Jiang, Suzhen Zheng, Rujia Wang, Jiaxu Qian, Minghua Ma, Jian-Guang Lou, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14603)  

**Abstract**: Recent Computer-Using Agents (CUAs), powered by multimodal large language models (LLMs), offer a promising direction for automating complex desktop workflows through natural language. However, most existing CUAs remain conceptual prototypes, hindered by shallow OS integration, fragile screenshot-based interaction, and disruptive execution.
We present UFO2, a multiagent AgentOS for Windows desktops that elevates CUAs into practical, system-level automation. UFO2 features a centralized HostAgent for task decomposition and coordination, alongside a collection of application-specialized AppAgent equipped with native APIs, domain-specific knowledge, and a unified GUI--API action layer. This architecture enables robust task execution while preserving modularity and extensibility. A hybrid control detection pipeline fuses Windows UI Automation (UIA) with vision-based parsing to support diverse interface styles. Runtime efficiency is further enhanced through speculative multi-action planning, reducing per-step LLM overhead. Finally, a Picture-in-Picture (PiP) interface enables automation within an isolated virtual desktop, allowing agents and users to operate concurrently without interference.
We evaluate UFO2 across over 20 real-world Windows applications, demonstrating substantial improvements in robustness and execution accuracy over prior CUAs. Our results show that deep OS integration unlocks a scalable path toward reliable, user-aligned desktop automation. 

**Abstract (ZH)**: Recent Computer-Using Agents (CUAs) Powered by Multimodal Large Language Models for Robust Desktop Workflow Automation via Natural Language: The UFO2 Multiagent AgentOS for Windows Desktops 

---
# Toward the Axiomatization of Intelligence: Structure, Time, and Existence 

**Title (ZH)**: 智能的公理化：结构、时间与存在 

**Authors**: Kei Itoh  

**Link**: [PDF](https://arxiv.org/pdf/2504.14596)  

**Abstract**: This study aims to construct an axiomatic definition of intelligence within a meta-framework that defines the method of definition, addressing intelligence as an inherently naive and polysemous concept. Initially, we formalize a set-theoretic representation of the universe as the domain wherein intelligence exists and characterize intelligence as a structure that involves temporal evolution and interaction with other sets. Starting from a naive definition of intelligence as "an entity possessing structures for externally inputting, internally processing, and externally outputting information or matter," we axiomatically reformulate it within this set-theoretical depiction of the universe. Applying this axiomatic definition, we compare and interpret three examples -- Hebbian non-optimized neural networks (NNs), backpropagation-optimized NNs, and biological reflexive systems -- in terms of their intelligence, structural properties, and biological plausibility. Furthermore, by extending our definition into a categorical framework, we introduce two categories, "Time Category" and "Intelligence Category," along with the functorial relationships between them, demonstrating the potential to represent changes and mimicry relationships among intelligent systems abstractly. Additionally, since intelligence, as defined herein, functions effectively only when accompanied by temporal interactions, we introduce the concept of "activity" and explore how activity-based conditions influence classifications and interpretations of intelligence. Finally, we suggest that our definitional methodology is not limited to intelligence alone, but can be similarly applied to other concepts, such as consciousness and emotion, advocating for their formal reinterpretation through the same procedural steps: defining a universal representation, selecting naive definitions, and axiomatic formalization. 

**Abstract (ZH)**: 本研究旨在在元框架中构建智能的公理化定义，该框架定义了定义方法，以应对智能这一先天模糊和多义的概念。首先，我们形式化一个集合表示的宇宙，作为智能存在的领域，并将智能视为涉及时间演化和与其他集合交互的结构。从“智能是具有处理外部输入、内部处理和输出信息或物质的结构的实体”的朴素定义出发，我们在这种集合表示的宇宙中公理化地重新定义它。通过这种公理化定义，我们比较并解释三种示例——递推非优化神经网络（NNs）、反向传播优化NNs和生物反射系统——在智能、结构特性和生物可行性方面的差异。此外，通过将定义扩展到范畴框架，我们引入了两个范畴“时间范畴”和“智能范畴”，以及它们之间的函子关系，展示了如何抽象地表示智能系统的变化及其模仿关系。此外，由于在此定义下，智能的有效作用仅限于伴随时间交互时，我们引入了“活动”的概念，并探讨基于活动条件如何影响智能的分类和解释。最后，我们建议我们的定义方法不仅限于智能，还可以类似地应用于其他概念，如意识和情绪，倡导通过相同的程序步骤对其形式重解释：定义普遍表示、选择朴素定义和公理化形式化。 

---
# Learning from Reasoning Failures via Synthetic Data Generation 

**Title (ZH)**: 通过合成数据生成学习推理失败 

**Authors**: Gabriela Ben Melech Stan, Estelle Aflalo, Avinash Madasu, Vasudev Lal, Phillip Howard  

**Link**: [PDF](https://arxiv.org/pdf/2504.14523)  

**Abstract**: Training models on synthetic data has emerged as an increasingly important strategy for improving the performance of generative AI. This approach is particularly helpful for large multimodal models (LMMs) due to the relative scarcity of high-quality paired image-text data compared to language-only data. While a variety of methods have been proposed for generating large multimodal datasets, they do not tailor the synthetic data to address specific deficiencies in the reasoning abilities of LMMs which will be trained with the generated dataset. In contrast, humans often learn in a more efficient manner by seeking out examples related to the types of reasoning where they have failed previously. Inspired by this observation, we propose a new approach for synthetic data generation which is grounded in the analysis of an existing LMM's reasoning failures. Our methodology leverages frontier models to automatically analyze errors produced by a weaker LMM and propose new examples which can be used to correct the reasoning failure via additional training, which are then further filtered to ensure high quality. We generate a large multimodal instruction tuning dataset containing over 553k examples using our approach and conduct extensive experiments demonstrating its utility for improving the performance of LMMs on multiple downstream tasks. Our results show that models trained on our synthetic data can even exceed the performance of LMMs trained on an equivalent amount of additional real data, demonstrating the high value of generating synthetic data targeted to specific reasoning failure modes in LMMs. We will make our dataset and code publicly available. 

**Abstract (ZH)**: 基于现有大型多模态模型推理缺陷分析的合成数据生成方法 

---
# Seeing Through Risk: A Symbolic Approximation of Prospect Theory 

**Title (ZH)**: 透过风险：prospect理论的符号approximation 

**Authors**: Ali Arslan Yousaf, Umair Rehman, Muhammad Umair Danish  

**Link**: [PDF](https://arxiv.org/pdf/2504.14448)  

**Abstract**: We propose a novel symbolic modeling framework for decision-making under risk that merges interpretability with the core insights of Prospect Theory. Our approach replaces opaque utility curves and probability weighting functions with transparent, effect-size-guided features. We mathematically formalize the method, demonstrate its ability to replicate well-known framing and loss-aversion phenomena, and provide an end-to-end empirical validation on synthetic datasets. The resulting model achieves competitive predictive performance while yielding clear coefficients mapped onto psychological constructs, making it suitable for applications ranging from AI safety to economic policy analysis. 

**Abstract (ZH)**: 我们提出一种将可解释性与期望效用理论核心见解相结合的新型符号建模框架，用于风险下的决策制定。该方法用透明的影响效应引导特征替换不透明的效用曲线和概率权重函数。我们对方法进行了数学形式化，展示了其重现著名框架效应和损失规避现象的能力，并在合成数据集上提供了端到端的经验验证。该模型在保持竞争力的预测性能的同时，能够映射清晰的系数到心理构建，适用于从AI安全到经济政策分析等广泛应用。 

---
# The Geometry of Self-Verification in a Task-Specific Reasoning Model 

**Title (ZH)**: 任务特定推理模型中自我验证的几何学 

**Authors**: Andrew Lee, Lihao Sun, Chris Wendler, Fernanda Viégas, Martin Wattenberg  

**Link**: [PDF](https://arxiv.org/pdf/2504.14379)  

**Abstract**: How do reasoning models verify their own answers? We study this question by training a model using DeepSeek R1's recipe on the CountDown task. We leverage the fact that preference tuning leads to mode collapse, resulting in a model that always produces highly structured and easily parse-able chain-of-thought sequences. With this setup, we do a top-down and bottom-up analysis to reverse-engineer how the model verifies its outputs. Our top-down analysis reveals Gated Linear Unit (GLU) weights encoding verification-related tokens, such as ``success'' or ``incorrect'', which activate according to the correctness of the model's reasoning steps. Our bottom-up analysis reveals that ``previous-token heads'' are mainly responsible for model verification. Our analyses meet in the middle: drawing inspiration from inter-layer communication channels, we use the identified GLU vectors to localize as few as three attention heads that can disable model verification, pointing to a necessary component of a potentially larger verification circuit. 

**Abstract (ZH)**: 如何推理模型验证自己的答案？我们通过使用DeepSeek R1的配方在CountDown任务上训练模型来研究这一问题。我们利用偏好调整会导致模式坍缩的事实，从而获得一个始终产生高度结构化和易于解析的推理链的模型。在这种设置下，我们从上到下和从下到上的分析来反向工程模型如何验证其输出。我们的从上到下分析揭示了门线性单元（GLU）权重编码验证相关的标记，如“成功”或“错误”，这些权重会根据模型推理步骤的正确性激活。我们的从下到上分析揭示了“前一标记头部”主要负责模型验证。我们的分析在中间相遇：从层间通信通道中汲取灵感，我们利用识别出的GLU向量定位三个可以禁用模型验证的注意力头，这指向了一个潜在更大验证电路中的必要组件。 

---
# Mathematical Programming Models for Exact and Interpretable Formulation of Neural Networks 

**Title (ZH)**: 数学规划模型以精确和可解释的方式表述神经网络 

**Authors**: Masoud Ataei, Edrin Hasaj, Jacob Gipp, Sepideh Forouzi  

**Link**: [PDF](https://arxiv.org/pdf/2504.14356)  

**Abstract**: This paper presents a unified mixed-integer programming framework for training sparse and interpretable neural networks. We develop exact formulations for both fully connected and convolutional architectures by modeling nonlinearities such as ReLU activations through binary variables and encoding structural sparsity via filter- and layer-level pruning constraints. The resulting models integrate parameter learning, architecture selection, and structural regularization within a single optimization problem, yielding globally optimal solutions with respect to a composite objective that balances prediction accuracy, weight sparsity, and architectural compactness. The mixed-integer programming formulation accommodates piecewise-linear operations, including max pooling and activation gating, and permits precise enforcement of logic-based or domain-specific constraints. By incorporating considerations of interpretability, sparsity, and verifiability directly into the training process, the proposed framework bridges a range of research areas including explainable artificial intelligence, symbolic reasoning, and formal verification. 

**Abstract (ZH)**: 本文提出了一种统一的混合整数规划框架，用于训练稀疏且可解释的神经网络。我们通过使用二进制变量建模非线性激活（如ReLU）并在滤波器级和层级剪枝约束中编码结构稀疏性，为全连接和卷积架构开发了精确的形式化模型。由此产生的模型在单一优化问题中整合了参数学习、架构选择和结构正则化，根据综合目标函数（平衡预测准确性、权重稀疏性和架构紧凑性）获得全局最优解。混合整数规划形式化模型支持分段线性操作，包括最大池化和激活门控，并允许精确施加基于逻辑或特定领域的约束。通过直接将可解释性、稀疏性以及验证性考虑纳入训练过程，所提出的框架跨越了可解释人工智能、符号推理和形式验证等多个研究领域。 

---
# FAIRGAME: a Framework for AI Agents Bias Recognition using Game Theory 

**Title (ZH)**: FAIRGAME：基于博弈论的AI代理偏见识别框架 

**Authors**: Alessio Buscemi, Daniele Proverbio, Alessandro Di Stefano, Anh Han, German Castignani, Pietro Di Liò  

**Link**: [PDF](https://arxiv.org/pdf/2504.14325)  

**Abstract**: Letting AI agents interact in multi-agent applications adds a layer of complexity to the interpretability and prediction of AI outcomes, with profound implications for their trustworthy adoption in research and society. Game theory offers powerful models to capture and interpret strategic interaction among agents, but requires the support of reproducible, standardized and user-friendly IT frameworks to enable comparison and interpretation of results. To this end, we present FAIRGAME, a Framework for AI Agents Bias Recognition using Game Theory. We describe its implementation and usage, and we employ it to uncover biased outcomes in popular games among AI agents, depending on the employed Large Language Model (LLM) and used language, as well as on the personality trait or strategic knowledge of the agents. Overall, FAIRGAME allows users to reliably and easily simulate their desired games and scenarios and compare the results across simulation campaigns and with game-theoretic predictions, enabling the systematic discovery of biases, the anticipation of emerging behavior out of strategic interplays, and empowering further research into strategic decision-making using LLM agents. 

**Abstract (ZH)**: 基于博弈论的AI代理偏见识别框架FAIRGAME 

---
# RadioDiff-Inverse: Diffusion Enhanced Bayesian Inverse Estimation for ISAC Radio Map Construction 

**Title (ZH)**: RadioDiff-Inverse: 基于反向传播的扩散增强贝叶斯逆估计算法用于ISAC雷达地图构建 

**Authors**: Xiucheng Wang, Zhongsheng Fang, Nan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.14298)  

**Abstract**: Radio maps (RMs) are essential for environment-aware communication and sensing, providing location-specific wireless channel information. Existing RM construction methods often rely on precise environmental data and base station (BS) locations, which are not always available in dynamic or privacy-sensitive environments. While sparse measurement techniques reduce data collection, the impact of noise in sparse data on RM accuracy is not well understood. This paper addresses these challenges by formulating RM construction as a Bayesian inverse problem under coarse environmental knowledge and noisy sparse measurements. Although maximum a posteriori (MAP) filtering offers an optimal solution, it requires a precise prior distribution of the RM, which is typically unavailable. To solve this, we propose RadioDiff-Inverse, a diffusion-enhanced Bayesian inverse estimation framework that uses an unconditional generative diffusion model to learn the RM prior. This approach not only reconstructs the spatial distribution of wireless channel features but also enables environmental structure perception, such as building outlines, and location of BS just relay on pathloss, through integrated sensing and communication (ISAC). Remarkably, RadioDiff-Inverse is training-free, leveraging a pre-trained model from Imagenet without task-specific fine-tuning, which significantly reduces the training cost of using generative large model in wireless networks. Experimental results demonstrate that RadioDiff-Inverse achieves state-of-the-art performance in accuracy of RM construction and environmental reconstruction, and robustness against noisy sparse sampling. 

**Abstract (ZH)**: 基于粗略环境知识和噪请求测的数据的无线电地图构建的扩散增强贝叶斯逆问题方法 

---
# CHAINSFORMER: Numerical Reasoning on Knowledge Graphs from a Chain Perspective 

**Title (ZH)**: CHAINSFORMER：从链的角度进行知识图上的数值推理 

**Authors**: Ze Zhao, Bin Lu, Xiaoying Gan, Gu Tang, Luoyi Fu, Xinbing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14282)  

**Abstract**: Reasoning over Knowledge Graphs (KGs) plays a pivotal role in knowledge graph completion or question answering systems, providing richer and more accurate triples and attributes. As numerical attributes become increasingly essential in characterizing entities and relations in KGs, the ability to reason over these attributes has gained significant importance. Existing graph-based methods such as Graph Neural Networks (GNNs) and Knowledge Graph Embeddings (KGEs), primarily focus on aggregating homogeneous local neighbors and implicitly embedding diverse triples. However, these approaches often fail to fully leverage the potential of logical paths within the graph, limiting their effectiveness in exploiting the reasoning process. To address these limitations, we propose ChainsFormer, a novel chain-based framework designed to support numerical reasoning. Chainsformer not only explicitly constructs logical chains but also expands the reasoning depth to multiple hops. Specially, we introduces Relation-Attribute Chains (RA-Chains), a specialized logic chain, to model sequential reasoning patterns. ChainsFormer captures the step-by-step nature of multi-hop reasoning along RA-Chains by employing sequential in-context learning. To mitigate the impact of noisy chains, we propose a hyperbolic affinity scoring mechanism that selects relevant logic chains in a variable-resolution space. Furthermore, ChainsFormer incorporates an attention-based numerical reasoner to identify critical reasoning paths, enhancing both reasoning accuracy and transparency. Experimental results demonstrate that ChainsFormer significantly outperforms state-of-the-art methods, achieving up to a 20.0% improvement in performance. The implementations are available at this https URL. 

**Abstract (ZH)**: 基于知识图谱的链推理：一种新型链基框架支持数值推理 

---
# ProtPainter: Draw or Drag Protein via Topology-guided Diffusion 

**Title (ZH)**: ProtPainter: 通过拓扑引导扩散进行蛋白质绘制或拖拽 

**Authors**: Zhengxi Lu, Shizhuo Cheng, Yuru Jiang, Yan Zhang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14274)  

**Abstract**: Recent advances in protein backbone generation have achieved promising results under structural, functional, or physical constraints. However, existing methods lack the flexibility for precise topology control, limiting navigation of the backbone space. We present ProtPainter, a diffusion-based approach for generating protein backbones conditioned on 3D curves. ProtPainter follows a two-stage process: curve-based sketching and sketch-guided backbone generation. For the first stage, we propose CurveEncoder, which predicts secondary structure annotations from a curve to parametrize sketch generation. For the second stage, the sketch guides the generative process in Denoising Diffusion Probabilistic Modeling (DDPM) to generate backbones. During this process, we further introduce a fusion scheduling scheme, Helix-Gating, to control the scaling factors. To evaluate, we propose the first benchmark for topology-conditioned protein generation, introducing Protein Restoration Task and a new metric, self-consistency Topology Fitness (scTF). Experiments demonstrate ProtPainter's ability to generate topology-fit (scTF > 0.8) and designable (scTM > 0.5) backbones, with drawing and dragging tasks showcasing its flexibility and versatility. 

**Abstract (ZH)**: Recent Advances in Protein Backbone Generation Based on 3D Curves: Introducing ProtPainter 

---
# Rethinking Traffic Flow Forecasting: From Transition to Generatation 

**Title (ZH)**: 重新思考交通流量预测：从转换到生成 

**Authors**: Li Shijiao, Ma Zhipeng, He Huajun, Chen Haiyue  

**Link**: [PDF](https://arxiv.org/pdf/2504.14248)  

**Abstract**: Traffic flow prediction plays an important role in Intelligent Transportation Systems in traffic management and urban planning. There have been extensive successful works in this area. However, these approaches focus only on modelling the flow transition and ignore the flow generation process, which manifests itself in two ways: (i) The models are based on Markovian assumptions, ignoring the multi-periodicity of the flow generation in nodes. (ii) The same structure is designed to encode both the transition and generation processes, ignoring the differences between them. To address these problems, we propose an Effective Multi-Branch Similarity Transformer for Traffic Flow Prediction, namely EMBSFormer. Through data analysis, we find that the factors affecting traffic flow include node-level traffic generation and graph-level traffic transition, which describe the multi-periodicity and interaction pattern of nodes, respectively. Specifically, to capture traffic generation patterns, we propose a similarity analysis module that supports multi-branch encoding to dynamically expand significant cycles. For traffic transition, we employ a temporal and spatial self-attention mechanism to maintain global node interactions, and use GNN and time conv to model local node interactions, respectively. Model performance is evaluated on three real-world datasets on both long-term and short-term prediction tasks. Experimental results show that EMBSFormer outperforms baselines on both tasks. Moreover, compared to models based on flow transition modelling (e.g. GMAN, 513k), the variant of EMBSFormer(93K) only uses 18\% of the parameters, achieving the same performance. 

**Abstract (ZH)**: 有效多分支相似性变压器在交通流预测中的应用：EMBSFormer 

---
# Assessing AI-Generated Questions' Alignment with Cognitive Frameworks in Educational Assessment 

**Title (ZH)**: 评估AI生成问题与认知框架在教育评估中的一致性 

**Authors**: Antoun Yaacoub, Jérôme Da-Rugna, Zainab Assaghir  

**Link**: [PDF](https://arxiv.org/pdf/2504.14232)  

**Abstract**: This study evaluates the integration of Bloom's Taxonomy into OneClickQuiz, an Artificial Intelligence (AI) driven plugin for automating Multiple-Choice Question (MCQ) generation in Moodle. Bloom's Taxonomy provides a structured framework for categorizing educational objectives into hierarchical cognitive levels. Our research investigates whether incorporating this taxonomy can improve the alignment of AI-generated questions with specific cognitive objectives. We developed a dataset of 3691 questions categorized according to Bloom's levels and employed various classification models-Multinomial Logistic Regression, Naive Bayes, Linear Support Vector Classification (SVC), and a Transformer-based model (DistilBERT)-to evaluate their effectiveness in categorizing questions. Our results indicate that higher Bloom's levels generally correlate with increased question length, Flesch-Kincaid Grade Level (FKGL), and Lexical Density (LD), reflecting the increased complexity of higher cognitive demands. Multinomial Logistic Regression showed varying accuracy across Bloom's levels, performing best for "Knowledge" and less accurately for higher-order levels. Merging higher-level categories improved accuracy for complex cognitive tasks. Naive Bayes and Linear SVC also demonstrated effective classification for lower levels but struggled with higher-order tasks. DistilBERT achieved the highest performance, significantly improving classification of both lower and higher-order cognitive levels, achieving an overall validation accuracy of 91%. This study highlights the potential of integrating Bloom's Taxonomy into AI-driven assessment tools and underscores the advantages of advanced models like DistilBERT for enhancing educational content generation. 

**Abstract (ZH)**: 本研究评估了将布卢姆 taxonomy 整合到 OneClickQuiz 中的效果，OneClickQuiz 是一个基于人工智能 (AI) 的插件，用于在 Moodle 中自动化生成多项选择题 (MCQ)。布卢姆 taxonomy 提供了一种结构化的框架，用于将教育目标按层次的认知水平进行分类。我们的研究探讨了将这一分类法整合到 AI 生成的问题中是否能够改善其与特定认知目标的对齐程度。我们开发了一个包含 3691 道题的数据集，这些题目根据布卢姆的层次进行了分类，并使用了多种分类模型—多项式 Logistic 回归、朴素贝叶斯、线性支持向量分类 (SVC) 以及基于变换器的模型（DistilBERT）—来评估它们在分类问题方面的有效性。研究结果表明，较高的布卢姆层次通常与较长的问题长度、Flesch-Kincaid 阅读级别（FKGL）和词汇密度（LD）相关，反映了较高层次的认知需求增加了复杂性。多项式 Logistic 回归在不同布卢姆层次上显示出了不同准确度，对于“知识”层次表现最优，而对于较高层次则表现较差。合并较高层次的类别能够提高复杂认知任务的准确度。朴素贝叶斯和线性 SVC 在较低层次上也展示了良好的分类效果，但在较高层次的任务上却表现出挑战。DistilBERT 达到了最高的性能，显著提高了对较低和较高层次认知水平的分类效果，总体验证准确率达到 91%。本研究突显了在 AI 驱动的评估工具中整合布卢姆 taxonomy 的潜力，并强调了如 DistilBERT 这种高级模型在增强教育内容生成方面的优势。 

---
# Pets: General Pattern Assisted Architecture For Time Series Analysis 

**Title (ZH)**: 宠物：时间序列分析的通用模式辅助架构 

**Authors**: Xiangkai Ma, Xiaobin Hong, Wenzhong Li, Sanglu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14209)  

**Abstract**: Time series analysis has found widespread applications in areas such as weather forecasting, anomaly detection, and healthcare. However, real-world sequential data often exhibit a superimposed state of various fluctuation patterns, including hourly, daily, and monthly frequencies. Traditional decomposition techniques struggle to effectively disentangle these multiple fluctuation patterns from the seasonal components, making time series analysis challenging. Surpassing the existing multi-period decoupling paradigms, this paper introduces a novel perspective based on energy distribution within the temporal-spectrum space. By adaptively quantifying observed sequences into continuous frequency band intervals, the proposed approach reconstructs fluctuation patterns across diverse periods without relying on domain-specific prior knowledge. Building upon this innovative strategy, we propose Pets, an enhanced architecture that is adaptable to arbitrary model structures. Pets integrates a Fluctuation Pattern Assisted (FPA) module and a Context-Guided Mixture of Predictors (MoP). The FPA module facilitates information fusion among diverse fluctuation patterns by capturing their dependencies and progressively modeling these patterns as latent representations at each layer. Meanwhile, the MoP module leverages these compound pattern representations to guide and regulate the reconstruction of distinct fluctuations hierarchically. Pets achieves state-of-the-art performance across various tasks, including forecasting, imputation, anomaly detection, and classification, while demonstrating strong generalization and robustness. 

**Abstract (ZH)**: 时间序列分析在天气预报、异常检测和医疗健康等领域找到了广泛的应用。然而，现实世界的序列数据通常表现出多种波动模式的叠加，包括小时、日和月的频率。传统的分解技术难以有效分离这些多周期的波动模式，使时间序列分析变得具有挑战性。超越现有的多周期解耦范式，本文提出了一种基于时间频谱空间内能量分布的新视角。通过适应性地将观测序列量化的到连续的频率带区间，所提出的方法在不需要领域特定先验知识的情况下，重构了不同周期的波动模式。在此创新策略的基础上，我们提出了Pets模型，该模型具备任意模型结构的适应性。Pets模型结合了波动模式辅助（FPA）模块和上下文引导的预测混合物（MoP）。FPA模块通过捕捉不同波动模式之间的依赖关系，并逐层建模这些模式为潜在表示来促进信息融合。同时，MoP模块利用这些综合模式表示来指导和调节不同层次波动的重建。在包括预测、插补、异常检测和分类在内的多种任务中，Pets模型取得了最先进的性能，同时展示了良好的泛化能力和鲁棒性。 

---
# Adaptation Method for Misinformation Identification 

**Title (ZH)**: 错误信息识别的适应性方法 

**Authors**: Yangping Chen, Weijie Shi, Mengze Li, Yue Cui, Hao Chen, Jia Zhu, Jiajie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14171)  

**Abstract**: Multimodal fake news detection plays a crucial role in combating online misinformation. Unfortunately, effective detection methods rely on annotated labels and encounter significant performance degradation when domain shifts exist between training (source) and test (target) data. To address the problems, we propose ADOSE, an Active Domain Adaptation (ADA) framework for multimodal fake news detection which actively annotates a small subset of target samples to improve detection performance. To identify various deceptive patterns in cross-domain settings, we design multiple expert classifiers to learn dependencies across different modalities. These classifiers specifically target the distinct deception patterns exhibited in fake news, where two unimodal classifiers capture knowledge errors within individual modalities while one cross-modal classifier identifies semantic inconsistencies between text and images. To reduce annotation costs from the target domain, we propose a least-disagree uncertainty selector with a diversity calculator for selecting the most informative samples. The selector leverages prediction disagreement before and after perturbations by multiple classifiers as an indicator of uncertain samples, whose deceptive patterns deviate most from source domains. It further incorporates diversity scores derived from multi-view features to ensure the chosen samples achieve maximal coverage of target domain features. The extensive experiments on multiple datasets show that ADOSE outperforms existing ADA methods by 2.72\% $\sim$ 14.02\%, indicating the superiority of our model. 

**Abstract (ZH)**: 多模态假新闻检测在打击在线虚假信息中发挥着 crucial 作用。为了解决标签标注和领域偏移带来的性能下降问题，我们提出了 ADOSE，一个用于多模态假新闻检测的主动领域自适应（Active Domain Adaptation, ADA）框架，该框架能够积极标注目标数据集中的小部分样本以提高检测性能。为在跨域设置中识别各种欺骗性模式，我们设计了多个专家分类器来学习不同模态之间的依赖关系。这些分类器专门针对假新闻中展现的不同欺骗模式，其中两个单模态分类器捕获各单一模态内的知识错误，而一个跨模态分类器识别文本和图像之间的语义不一致。为了降低目标领域标注成本，我们提出了一种最少分歧不确定性选择器与多样性计算器，用于选择最具信息量的样本。该选择器利用多个分类器在扰动前后预测分歧作为不确定样本的指标，这些样本的欺骗性模式与源领域差异最大。此外，该选择器结合了多视图特征衍生的多样性分数，确保所选样本最大程度覆盖目标领域特征。在多个数据集上的广泛实验表明，ADOSE 在性能上优于现有 ADA 方法 2.72% ~ 14.02%，表明了我们模型的优势。 

---
# Bayesian Principles Improve Prompt Learning In Vision-Language Models 

**Title (ZH)**: 贝叶斯原则提升视觉-语言模型的提示学习效果 

**Authors**: Mingyu Kim, Jongwoo Ko, Mijung Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.14123)  

**Abstract**: Prompt learning is a popular fine-tuning method for vision-language models due to its efficiency. It requires a small number of additional learnable parameters while significantly enhancing performance on target tasks. However, most existing methods suffer from overfitting to fine-tuning data, yielding poor generalizability. To address this, we propose a new training objective function based on a Bayesian learning principle to balance adaptability and generalizability. We derive a prior over the logits, where the mean function is parameterized by the pre-trained model, while the posterior corresponds to the fine-tuned model. This objective establishes a balance by allowing the fine-tuned model to adapt to downstream tasks while remaining close to the pre-trained model. 

**Abstract (ZH)**: 基于贝叶斯学习原理的前景学习目标函数：平衡适配性和泛化性 

---
# Linking forward-pass dynamics in Transformers and real-time human processing 

**Title (ZH)**: 连接Transformer在前向传播中的动态与实时人类处理 

**Authors**: Jennifer Hu, Michael A. Lepori, Michael Franke  

**Link**: [PDF](https://arxiv.org/pdf/2504.14107)  

**Abstract**: Modern AI models are increasingly being used as theoretical tools to study human cognition. One dominant approach is to evaluate whether human-derived measures (such as offline judgments or real-time processing) are predicted by a model's output: that is, the end-product of forward pass(es) through the network. At the same time, recent advances in mechanistic interpretability have begun to reveal the internal processes that give rise to model outputs, raising the question of whether models and humans might arrive at outputs using similar "processing strategies". Here, we investigate the link between real-time processing in humans and "layer-time" dynamics in Transformer models. Across five studies spanning domains and modalities, we test whether the dynamics of computation in a single forward pass of pre-trained Transformers predict signatures of processing in humans, above and beyond properties of the model's output probability distribution. We consistently find that layer-time dynamics provide additional predictive power on top of output measures. Our results suggest that Transformer processing and human processing may be facilitated or impeded by similar properties of an input stimulus, and this similarity has emerged through general-purpose objectives such as next-token prediction or image recognition. Our work suggests a new way of using AI models to study human cognition: not just as a black box mapping stimuli to responses, but potentially also as explicit processing models. 

**Abstract (ZH)**: 现代AI模型 increasingly being used as theoretical tools to study human cognition: Investigating the Link between Real-time Processing in Humans and "Layer-time" Dynamics in Transformer Models 

---
# Birds of a Different Feather Flock Together: Exploring Opportunities and Challenges in Animal-Human-Machine Teaming 

**Title (ZH)**: 志不同道不合者不为朋：探索动物-人类-机器协同中的机遇与挑战 

**Authors**: Myke C. Cohen, David A. Grimm, Reuth Mirsky, Xiaoyun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2504.13973)  

**Abstract**: Animal-Human-Machine (AHM) teams are a type of hybrid intelligence system wherein interactions between a human, AI-enabled machine, and animal members can result in unique capabilities greater than the sum of their parts. This paper calls for a systematic approach to studying the design of AHM team structures to optimize performance and overcome limitations in various applied settings. We consider the challenges and opportunities in investigating the synergistic potential of AHM team members by introducing a set of dimensions of AHM team functioning to effectively utilize each member's strengths while compensating for individual weaknesses. Using three representative examples of such teams -- security screening, search-and-rescue, and guide dogs -- the paper illustrates how AHM teams can tackle complex tasks. We conclude with open research directions that this multidimensional approach presents for studying hybrid human-AI systems beyond AHM teams. 

**Abstract (ZH)**: 动物-人类-机器（AHM）团队是一种混合智能系统，其中人类、AI赋能的机器和动物成员之间的交互可以产生超出各自部分总和的独特能力。本文呼吁采用系统方法研究AHM团队结构的设计，以优化各种实际应用场景中的性能并克服限制。我们通过介绍AHM团队运作的一系列维度来探讨成员之间协同潜力的机会和挑战，有效利用每个成员的优势并弥补个体的不足。以安全筛查、搜索与救援和导盲犬三个代表性的团队为例，本文展示了AHM团队如何应对复杂的任务。最后，本文讨论了这一多维方法为研究超越AHM团队的混合人机系统所带来的开放研究方向。 

---
# Evaluation and Incident Prevention in an Enterprise AI Assistant 

**Title (ZH)**: 企业级AI助手中的评估与事件预防 

**Authors**: Akash V. Maharaj, David Arbour, Daniel Lee, Uttaran Bhattacharya, Anup Rao, Austin Zane, Avi Feller, Kun Qian, Yunyao Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.13924)  

**Abstract**: Enterprise AI Assistants are increasingly deployed in domains where accuracy is paramount, making each erroneous output a potentially significant incident. This paper presents a comprehensive framework for monitoring, benchmarking, and continuously improving such complex, multi-component systems under active development by multiple teams. Our approach encompasses three key elements: (1) a hierarchical ``severity'' framework for incident detection that identifies and categorizes errors while attributing component-specific error rates, facilitating targeted improvements; (2) a scalable and principled methodology for benchmark construction, evaluation, and deployment, designed to accommodate multiple development teams, mitigate overfitting risks, and assess the downstream impact of system modifications; and (3) a continual improvement strategy leveraging multidimensional evaluation, enabling the identification and implementation of diverse enhancement opportunities. By adopting this holistic framework, organizations can systematically enhance the reliability and performance of their AI Assistants, ensuring their efficacy in critical enterprise environments. We conclude by discussing how this multifaceted evaluation approach opens avenues for various classes of enhancements, paving the way for more robust and trustworthy AI systems. 

**Abstract (ZH)**: 企业AI助手在准确性至关重要的领域中越来越普及，每一次错误输出都可能构成重大事件。本文提出了一种全面的框架，用于监控、基准测试并持续改进由多个团队在积极开发中的复杂多组件系统。我们的方法包括三个关键要素：(1)一种分层的“严重性”框架，用于检测和分类错误，并赋予组件特定的错误率，从而实现有针对性的改进；(2)一种可扩展且原则性的基准构建、评估和部署方法，旨在适应多个开发团队、缓解过拟合风险并评估系统修改的下游影响；(3)一种基于多维度评估的持续改进策略，能够识别和实施各种改进机会。通过采用这一综合框架，组织可以系统地提高其AI助手的可靠性和性能，确保其在关键的企业环境中有效。我们最后讨论了这种多方面评估方法如何为各种类型的改进开辟途径，推动更可靠和可信的AI系统的发展。 

---
# The Model Counting Competitions 2021-2023 

**Title (ZH)**: 2021-2023年模型计数比赛 

**Authors**: Johannes K. Fichte, Markus Hecher  

**Link**: [PDF](https://arxiv.org/pdf/2504.13842)  

**Abstract**: Modern society is full of computational challenges that rely on probabilistic reasoning, statistics, and combinatorics. Interestingly, many of these questions can be formulated by encoding them into propositional formulas and then asking for its number of models. With a growing interest in practical problem-solving for tasks that involve model counting, the community established the Model Counting (MC) Competition in fall of 2019 with its first iteration in 2020. The competition aims at advancing applications, identifying challenging benchmarks, fostering new solver development, and enhancing existing solvers for model counting problems and their variants. The first iteration, brought together various researchers, identified challenges, and inspired numerous new applications. In this paper, we present a comprehensive overview of the 2021-2023 iterations of the Model Counting Competition. We detail its execution and outcomes. The competition comprised four tracks, each focusing on a different variant of the model counting problem. The first track centered on the model counting problem (MC), which seeks the count of models for a given propositional formula. The second track challenged developers to submit programs capable of solving the weighted model counting problem (WMC). The third track was dedicated to projected model counting (PMC). Finally, we initiated a track that combined projected and weighted model counting (PWMC). The competition continued with a high level of participation, with seven to nine solvers submitted in various different version and based on quite diverging techniques. 

**Abstract (ZH)**: 现代社会充满了依赖概率推理、统计和组合数学的计算挑战。许多问题可以通过将它们编码为命题公式，然后询问其模型数量来进行表述。随着对涉及模型计数的任务的实用问题解决日益感兴趣，社区在2019年秋季建立了模型计数（MC）竞赛，并在2020年进行了首次迭代。竞赛旨在促进应用、识别具有挑战性的基准、推动新的求解器开发，并提高现有模型计数问题及其变体求解器的性能。第一次迭代汇聚了各种研究人员，确定了挑战，并激励了众多新的应用。在本文中，我们对2021-2023年的模型计数竞赛进行了全面概述，详细说明了其执行和结果。竞赛包括四个赛道，每个赛道专注于模型计数问题的不同变体。第一赛道专注于模型计数问题（MC），旨在求解给定命题公式的模型数量。第二赛道挑战开发者提交能够解决加权模型计数问题（WMC）的程序。第三赛道专门用于投影模型计数（PMC）。最后，我们启动了一个结合投影和加权模型计数的赛道（PWMC）。竞赛继续保持着高水平的参与度，参赛者提交了七到九种不同版本的求解器，基于相当不同的技术。 

---
# Roll the dice & look before you leap: Going beyond the creative limits of next-token prediction 

**Title (ZH)**: 掷骰子再迈步：超越下一个-token 预测的创造性限制 

**Authors**: Vaishnavh Nagarajan, Chen Henry Wu, Charles Ding, Aditi Raghunathan  

**Link**: [PDF](https://arxiv.org/pdf/2504.15266)  

**Abstract**: We design a suite of minimal algorithmic tasks that are a loose abstraction of open-ended real-world tasks. This allows us to cleanly and controllably quantify the creative limits of the present-day language model. Much like real-world tasks that require a creative, far-sighted leap of thought, our tasks require an implicit, open-ended stochastic planning step that either (a) discovers new connections in an abstract knowledge graph (like in wordplay, drawing analogies, or research) or (b) constructs new patterns (like in designing math problems or new proteins). In these tasks, we empirically and conceptually argue how next-token learning is myopic and memorizes excessively; comparatively, multi-token approaches, namely teacherless training and diffusion models, excel in producing diverse and original output. Secondly, in our tasks, we find that to elicit randomness from the Transformer without hurting coherence, it is better to inject noise right at the input layer (via a method we dub hash-conditioning) rather than defer to temperature sampling from the output layer. Thus, our work offers a principled, minimal test-bed for analyzing open-ended creative skills, and offers new arguments for going beyond next-token learning and softmax-based sampling. We make part of the code available under this https URL 

**Abstract (ZH)**: 我们设计了一套最小算法任务，作为开放性现实世界任务的松散抽象，以便清晰可控地量化当今语言模型的创造性极限。就像现实世界任务需要远见卓识的创造性思维跳跃一样，我们的任务需要一种隐式的、开放性的随机规划步骤，这种步骤要么（a）在抽象知识图中发现新的联系（如在文字游戏、类比或研究中），要么（b）构建新的模式（如在设计数学问题或新型蛋白质中）。在这些任务中，我们从经验上和概念上论证了下一个token的学习是短视的且过度记忆；相比之下，多token方法，即无教师训练和扩散模型，在产生多样性和原创性输出方面更为出色。其次，在我们的任务中，我们发现，如果不损害连贯性，从Transformer中注入噪声以诱发随机性（我们称之为哈希条件化的方法）比在输出层使用温度采样更好。因此，我们的工作提供了一个分析开放性创造性技能的原理性、最小化测试平台，并为超越下一个token学习和softmax基于采样提供了新的论据。我们部分代码在此处提供。 

---
# M$^2$AD: Multi-Sensor Multi-System Anomaly Detection through Global Scoring and Calibrated Thresholding 

**Title (ZH)**: M$^2$AD: 多传感器多系统异常检测通过全局评分和校准阈值方法 

**Authors**: Sarah Alnegheimish, Zelin He, Matthew Reimherr, Akash Chandrayan, Abhinav Pradhan, Luca D'Angelo  

**Link**: [PDF](https://arxiv.org/pdf/2504.15225)  

**Abstract**: With the widespread availability of sensor data across industrial and operational systems, we frequently encounter heterogeneous time series from multiple systems. Anomaly detection is crucial for such systems to facilitate predictive maintenance. However, most existing anomaly detection methods are designed for either univariate or single-system multivariate data, making them insufficient for these complex scenarios. To address this, we introduce M$^2$AD, a framework for unsupervised anomaly detection in multivariate time series data from multiple systems. M$^2$AD employs deep models to capture expected behavior under normal conditions, using the residuals as indicators of potential anomalies. These residuals are then aggregated into a global anomaly score through a Gaussian Mixture Model and Gamma calibration. We theoretically demonstrate that this framework can effectively address heterogeneity and dependencies across sensors and systems. Empirically, M$^2$AD outperforms existing methods in extensive evaluations by 21% on average, and its effectiveness is demonstrated on a large-scale real-world case study on 130 assets in Amazon Fulfillment Centers. Our code and results are available at this https URL. 

**Abstract (ZH)**: 随着工业和运营系统中传感器数据的广泛可用，我们经常遇到来自多个系统的异构时间序列数据。多系统多变量时间序列数据的无监督异常检测对于这些系统促进预测性维护至关重要。然而，大多数现有的异常检测方法都是为单变量数据或单系统多变量数据设计的，不足以处理这些复杂场景。为了解决这一问题，我们提出了M$^2$AD框架，用于多系统多变量时间序列数据的无监督异常检测。M$^2$AD利用深度模型捕获正常条件下的预期行为，并使用残差作为潜在异常的指示符。这些残差通过高斯混合模型和Gamma校准聚合为全局异常分数。我们从理论上证明，该框架可以有效地处理传感器和系统之间的异构性和依赖性。实验上，M$^2$AD在广泛评价中平均优于现有方法21%，其有效性在Amazon Fulfillment Centers 130个资产的大规模实际案例研究中得到了验证。我们的代码和结果可访问此链接。 

---
# A Causal Convolutional Low-rank Representation Model for Imputation of Water Quality Data 

**Title (ZH)**: 一种因果卷积低秩表示模型用于水质数据插补 

**Authors**: Xin Liao, Bing Yang, Tan Dongli, Cai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15209)  

**Abstract**: The monitoring of water quality is a crucial part of environmental protection, and a large number of monitors are widely deployed to monitor water quality. Due to unavoidable factors such as data acquisition breakdowns, sensors and communication failures, water quality monitoring data suffers from missing values over time, resulting in High-Dimensional and Sparse (HDS) Water Quality Data (WQD). The simple and rough filling of the missing values leads to inaccurate results and affects the implementation of relevant measures. Therefore, this paper proposes a Causal convolutional Low-rank Representation (CLR) model for imputing missing WQD to improve the completeness of the WQD, which employs a two-fold idea: a) applying causal convolutional operation to consider the temporal dependence of the low-rank representation, thus incorporating temporal information to improve the imputation accuracy; and b) implementing a hyperparameters adaptation scheme to automatically adjust the best hyperparameters during model training, thereby reducing the tedious manual adjustment of hyper-parameters. Experimental studies on three real-world water quality datasets demonstrate that the proposed CLR model is superior to some of the existing state-of-the-art imputation models in terms of imputation accuracy and time cost, as well as indicating that the proposed model provides more reliable decision support for environmental monitoring. 

**Abstract (ZH)**: 高维稀疏水质数据的因果卷积低秩表示缺失值填充模型 

---
# Breast density in MRI: an AI-based quantification and relationship to assessment in mammography 

**Title (ZH)**: 基于AI的MRI乳腺密度定量及其与 mammography 评估的相关性 

**Authors**: Yaqian Chen, Lin Li, Hanxue Gu, Haoyu Dong, Derek L. Nguyen, Allan D. Kirk, Maciej A. Mazurowski, E. Shelley Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15192)  

**Abstract**: Mammographic breast density is a well-established risk factor for breast cancer. Recently there has been interest in breast MRI as an adjunct to mammography, as this modality provides an orthogonal and highly quantitative assessment of breast tissue. However, its 3D nature poses analytic challenges related to delineating and aggregating complex structures across slices. Here, we applied an in-house machine-learning algorithm to assess breast density on normal breasts in three MRI datasets. Breast density was consistent across different datasets (0.104 - 0.114). Analysis across different age groups also demonstrated strong consistency across datasets and confirmed a trend of decreasing density with age as reported in previous studies. MR breast density was correlated with mammographic breast density, although some notable differences suggest that certain breast density components are captured only on MRI. Future work will determine how to integrate MR breast density with current tools to improve future breast cancer risk prediction. 

**Abstract (ZH)**: 乳腺密度是乳腺癌的一个公认的风险因素。近年来，乳腺MRI作为一种辅助于乳腺X线摄影的技术引起了研究兴趣，因其提供了与乳腺组织高度定量的三维评估。然而，其三维性质带来了在不同切片中界定和聚合复杂结构的分析挑战。我们应用一种内部开发的机器学习算法评估了三个MRI数据集中正常乳腺的密度。乳腺密度在不同数据集中保持一致（0.104 - 0.114）。不同年龄组的分析也显示了数据集间的强一致性，并证实了随年龄增长密度下降的趋势，这与先前的研究一致。MRI乳腺密度与乳腺X线摄影乳腺密度相关，尽管一些显著差异表明某些乳腺密度成分仅在MRI上捕获。未来的工作将确定如何将MRI乳腺密度与现有工具结合，以改善未来的乳腺癌风险预测。 

---
# Existing Industry Practice for the EU AI Act's General-Purpose AI Code of Practice Safety and Security Measures 

**Title (ZH)**: 欧盟AI法案通用人工智能行为准则的安全与安全措施现有行业实践 

**Authors**: Lily Stelling, Mick Yang, Rokas Gipiškis, Leon Staufer, Ze Shen Chin, Siméon Campos, Michael Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.15181)  

**Abstract**: This report provides a detailed comparison between the measures proposed in the EU AI Act's General-Purpose AI (GPAI) Code of Practice (Third Draft) and current practices adopted by leading AI companies. As the EU moves toward enforcing binding obligations for GPAI model providers, the Code of Practice will be key to bridging legal requirements with concrete technical commitments. Our analysis focuses on the draft's Safety and Security section which is only relevant for the providers of the most advanced models (Commitments II.1-II.16) and excerpts from current public-facing documents quotes that are relevant to each individual measure.
We systematically reviewed different document types - including companies' frontier safety frameworks and model cards - from over a dozen companies, including OpenAI, Anthropic, Google DeepMind, Microsoft, Meta, Amazon, and others. This report is not meant to be an indication of legal compliance nor does it take any prescriptive viewpoint about the Code of Practice or companies' policies. Instead, it aims to inform the ongoing dialogue between regulators and GPAI model providers by surfacing evidence of precedent. 

**Abstract (ZH)**: 欧盟AI法案的通用人工智能(GPAI)行为守则（第三草案）提出的措施与领先AI公司现有做法的详细对比：从安全与安全性的角度探讨草案中的承诺及其相关公开文件摘录 

---
# C2RUST-BENCH: A Minimized, Representative Dataset for C-to-Rust Transpilation Evaluation 

**Title (ZH)**: C2RUST-BENCH：用于C到Rust转换评估的最小化且具代表性的数据集 

**Authors**: Melih Sirlanci, Carter Yagemann, Zhiqiang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15144)  

**Abstract**: Despite the effort in vulnerability detection over the last two decades, memory safety vulnerabilities continue to be a critical problem. Recent reports suggest that the key solution is to migrate to memory-safe languages. To this end, C-to-Rust transpilation becomes popular to resolve memory-safety issues in C programs. Recent works propose C-to-Rust transpilation frameworks; however, a comprehensive evaluation dataset is missing. Although one solution is to put together a large enough dataset, this increases the analysis time in automated frameworks as well as in manual efforts for some cases. In this work, we build a method to select functions from a large set to construct a minimized yet representative dataset to evaluate the C-to-Rust transpilation. We propose C2RUST-BENCH that contains 2,905 functions, which are representative of C-to-Rust transpilation, selected from 15,503 functions of real-world programs. 

**Abstract (ZH)**: 尽管在过去二十年中付出了努力进行漏洞检测，内存安全性漏洞仍然是一个关键问题。近期的报告显示，解决这一问题的关键是迁移到内存安全语言。为此，C到Rust的转换越来越流行，以解决C程序中的内存安全问题。近年来，提出了C-to-Rust转换框架，但缺乏一个全面的评估数据集。虽然可以通过构建足够大的数据集来解决这一问题，但这会增加自动化框架以及某些情况下手动努力的分析时间。在本工作中，我们构建了一种方法，从大量函数中选择函数来构建一个既能代表C-to-Rust转换，又具有最小性的数据集。我们提出了C2RUST-BENCH，包含了2,905个函数，这些函数是从15,503个真实程序函数中选取的，具有代表C-to-Rust转换的特点。 

---
# NeuGaze: Reshaping the future BCI 

**Title (ZH)**: NeuGaze: 重塑未来的脑机接口 

**Authors**: Yiqian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15101)  

**Abstract**: Traditional brain-computer interfaces (BCIs), reliant on costly electroencephalography or invasive implants, struggle with complex human-computer interactions due to setup complexity and limited precision. We present NeuGaze, a novel webcam-based system that leverages eye gaze, head movements, and facial expressions to enable intuitive, real-time control using only a standard 30 Hz webcam, often pre-installed in laptops. Requiring minimal calibration, NeuGaze achieves performance comparable to conventional inputs, supporting precise cursor navigation, key triggering via an efficient skill wheel, and dynamic gaming interactions, such as defeating formidable opponents in first-person games. By harnessing preserved neck-up functionalities in motor-impaired individuals, NeuGaze eliminates the need for specialized hardware, offering a low-cost, accessible alternative to BCIs. This paradigm empowers diverse applications, from assistive technology to entertainment, redefining human-computer interaction for motor-impaired users. Project is at \href{this https URL}{this http URL}. 

**Abstract (ZH)**: 传统脑机接口（BCIs）依赖于昂贵的脑电图或侵入性植入物，由于设置复杂性高和精度有限，难以实现复杂的计算机交互。我们提出NeuGaze，一种新型的基于网络摄像头的系统，通过利用眼球注视、头部运动和面部表情，仅使用标准30 Hz网络摄像头（常预装在笔记本电脑中）实现直观的实时控制。NeuGaze无需大量校准即可达到与传统输入相当的性能，支持精确的鼠标导航、通过高效的技能轮实现键触发，以及动态的游戏交互，如在第一人称游戏中击败强大的对手。通过利用运动受损个体保留的颈部以上功能，NeuGaze消除了对专用硬件的需求，提供了一种低成本、易访问的BCI替代方案。这一范式为从辅助技术到娱乐的各种应用赋能，重新定义了运动受损用户的人机交互。项目详情请参见this http URL。 

---
# Fast-Slow Co-advancing Optimizer: Toward Harmonious Adversarial Training of GAN 

**Title (ZH)**: 快速-缓慢协同优化器：迈向GAN和谐对抗训练 

**Authors**: Lin Wang, Xiancheng Wang, Rui Wang, Zhibo Zhang, Minghang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.15099)  

**Abstract**: Up to now, the training processes of typical Generative Adversarial Networks (GANs) are still particularly sensitive to data properties and hyperparameters, which may lead to severe oscillations, difficulties in convergence, or even failures to converge, especially when the overall variances of the training sets are large. These phenomena are often attributed to the training characteristics of such networks. Aiming at the problem, this paper develops a new intelligent optimizer, Fast-Slow Co-advancing Optimizer (FSCO), which employs reinforcement learning in the training process of GANs to make training easier. Specifically, this paper allows the training step size to be controlled by an agent to improve training stability, and makes the training process more intelligent with variable learning rates, making GANs less sensitive to step size. Experiments have been conducted on three benchmark datasets to verify the effectiveness of the developed FSCO. 

**Abstract (ZH)**: 到目前为止，典型的生成对抗网络（GANs）的训练过程仍然特别依赖于数据属性和超参数，这可能导致严重的振荡、收敛困难，甚至无法收敛，特别是在训练集总体方差较大时。这些现象通常归因于此类网络的训练特性。为了解决这一问题，本文开发了一种新的智能优化器——快速-缓慢协同优化器（FSCO），该优化器在GANs的训练过程中采用强化学习来提高训练稳定性。具体而言，本文通过允许训练步长由代理控制来提高训练稳定性，并通过可变学习率使训练过程更加智能，从而使GANs对步长不那么敏感。实验已在三个基准数据集上进行，以验证所开发的FSCO的有效性。 

---
# Federated Latent Factor Model for Bias-Aware Recommendation with Privacy-Preserving 

**Title (ZH)**: 联邦潜因素模型：具有隐私保护的偏倚感知推荐 

**Authors**: Junxiang Gao, Yixin Ran, Jia Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.15090)  

**Abstract**: A recommender system (RS) aims to provide users with personalized item recommendations, enhancing their overall experience. Traditional RSs collect and process all user data on a central server. However, this centralized approach raises significant privacy concerns, as it increases the risk of data breaches and privacy leakages, which are becoming increasingly unacceptable to privacy-sensitive users. To address these privacy challenges, federated learning has been integrated into RSs, ensuring that user data remains secure. In centralized RSs, the issue of rating bias is effectively addressed by jointly analyzing all users' raw interaction data. However, this becomes a significant challenge in federated RSs, as raw data is no longer accessible due to privacy-preserving constraints. To overcome this problem, we propose a Federated Bias-Aware Latent Factor (FBALF) model. In FBALF, training bias is explicitly incorporated into every local model's loss function, allowing for the effective elimination of rating bias without compromising data privacy. Extensive experiments conducted on three real-world datasets demonstrate that FBALF achieves significantly higher recommendation accuracy compared to other state-of-the-art federated RSs. 

**Abstract (ZH)**: 一种推荐系统（RS）旨在为用户提供个性化项目推荐，提升用户的整体体验。传统的RS在中央服务器上收集和处理所有用户数据。然而，这种集中式方法引发了重大的隐私 concern，增加了数据泄露和隐私泄漏的风险，这些风险越来越不被重视用户的接受。为了解决这些隐私挑战，已经将联邦学习整合到RS中，确保用户数据的安全性。在集中式的RS中，通过联合分析所有用户的原始交互数据，有效地解决了评分偏差问题。但在联邦RS中，由于隐私保护的限制，原始数据不再可访问，这成为了一个重大挑战。为了克服这个问题，我们提出了一种联邦感知偏差的潜在因子模型（FBALF）。在FBALF中，训练偏差被明确地纳入每个本地模型的损失函数中，可以在不牺牲数据隐私的情况下有效消除评分偏差。在对三个真实世界数据集进行的广泛实验中，FBALF在推荐准确性上显著高于其他最先进的联邦RS。 

---
# Mining Characteristics of Vulnerable Smart Contracts Across Lifecycle Stages 

**Title (ZH)**: 跨生命周期阶段脆弱智能合约的挖掘特性研究 

**Authors**: Hongli Peng, Xiaoqi Li, Wenkai Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.15063)  

**Abstract**: Smart contracts are the cornerstone of decentralized applications and financial protocols, which extend the application of digital currency transactions. The applications and financial protocols introduce significant security challenges, resulting in substantial economic losses. Existing solutions predominantly focus on code vulnerabilities within smart contracts, accounting for only 50% of security incidents. Therefore, a more comprehensive study of security issues related to smart contracts is imperative. The existing empirical research realizes the static analysis of smart contracts from the perspective of the lifecycle and gives the corresponding measures for each stage. However, they lack the characteristic analysis of vulnerabilities in each stage and the distinction between the vulnerabilities. In this paper, we present the first empirical study on the security of smart contracts throughout their lifecycle, including deployment and execution, upgrade, and destruction stages. It delves into the security issues at each stage and provides at least seven feature descriptions. Finally, utilizing these seven features, five machine-learning classification models are used to identify vulnerabilities at different stages. The classification results reveal that vulnerable contracts exhibit distinct transaction features and ego network properties at various stages. 

**Abstract (ZH)**: 智能合约是去中心化应用和金融协议的基石，扩展了数字货币交易的应用范围。这些应用和金融协议引入了重要的安全挑战，导致了巨大的经济损失。现有的解决方案主要集中在智能合约的代码漏洞上，占所有安全事件的50%。因此，对智能合约相关安全问题进行更加全面的研究显得尤为重要。现有的实证研究从生命周期的角度实现了智能合约的静态分析，并为每个阶段提供相应的措施。然而，它们缺乏对每个阶段漏洞特征的分析以及对漏洞之间的区别。在本文中，我们首次从部署和执行、升级和销毁等阶段对智能合约的安全性进行实证研究，并深入探讨了每个阶段的安全问题，提供了至少七个特征描述。最终，利用这七个特征，使用了五种机器学习分类模型来识别不同阶段的漏洞。分类结果表明，易受攻击的合约在各个阶段表现出不同的交易特征和ego网络属性。 

---
# OPO: Making Decision-Focused Data Acquisition Decisions 

**Title (ZH)**: OPO: 面向决策的数据采集决策 

**Authors**: Egon Peršak, Miguel F. Anjos  

**Link**: [PDF](https://arxiv.org/pdf/2504.15062)  

**Abstract**: We propose a model for making data acquisition decisions for variables in contextual stochastic optimisation problems. Data acquisition decisions are typically treated as separate and fixed. We explore problem settings in which the acquisition of contextual variables is costly and consequently constrained. The data acquisition problem is often solved heuristically for proxy objectives such as coverage. The more intuitive objective is the downstream decision quality as a result of data acquisition decisions. The whole pipeline can be characterised as an optimise-then-predict-then-optimise (OPO) problem. Analogously, much recent research has focused on how to integrate prediction and optimisation (PO) in the form of decision-focused learning. We propose leveraging differentiable optimisation to extend the integration to data acquisition. We solve the data acquisition problem with well-defined constraints by learning a surrogate linear objective function. We demonstrate an application of this model on a shortest path problem for which we first have to set a drone reconnaissance strategy to capture image segments serving as inputs to a model that predicts travel costs. We ablate the problem with a number of training modalities and demonstrate that the differentiable optimisation approach outperforms random search strategies. 

**Abstract (ZH)**: 我们提出了一种模型，用于在基于上下文的随机优化问题中为变量的数据采集决策制定策略。数据采集决策通常被视为独立且固定的。我们探讨了数据采集成本较高且因此受到限制的问题设置。数据采集问题通常通过代理目标（如覆盖率）的启发式方法来解决。更直观的目标是由于数据采集决策而导致的下游决策质量。整个流程可以被描述为优化-预测-再优化（OPO）问题。类似地，近期许多研究集中在如何将预测和优化（PO）整合为决策导向学习。我们提议利用可微优化来扩展这种整合到数据采集中。我们通过学习一个可微的线性目标函数来解决具有明确约束的数据采集问题。我们在此模型上应用了一个最短路径问题，首先需要设定无人机侦察策略以捕捉作为预测旅行成本模型输入的图像片段。我们通过多种训练方式消融研究，证明了可微优化方法优于随机搜索策略。 

---
# VeLU: Variance-enhanced Learning Unit for Deep Neural Networks 

**Title (ZH)**: VeLU：方差增强的学习单元用于深度神经网络 

**Authors**: Ashkan Shakarami, Yousef Yeganeh, Azade Farshad, Lorenzo Nicolè, Stefano Ghidoni, Nassir Navab  

**Link**: [PDF](https://arxiv.org/pdf/2504.15051)  

**Abstract**: Activation functions are fundamental in deep neural networks and directly impact gradient flow, optimization stability, and generalization. Although ReLU remains standard because of its simplicity, it suffers from vanishing gradients and lacks adaptability. Alternatives like Swish and GELU introduce smooth transitions, but fail to dynamically adjust to input statistics. We propose VeLU, a Variance-enhanced Learning Unit as an activation function that dynamically scales based on input variance by integrating ArcTan-Sin transformations and Wasserstein-2 regularization, effectively mitigating covariate shifts and stabilizing optimization. Extensive experiments on ViT_B16, VGG19, ResNet50, DenseNet121, MobileNetV2, and EfficientNetB3 confirm VeLU's superiority over ReLU, ReLU6, Swish, and GELU on six vision benchmarks. The codes of VeLU are publicly available on GitHub. 

**Abstract (ZH)**: VeLU：一种基于输入方差动态缩放的激活函数及其在优化稳定性和泛化能力上的改进 

---
# Beyond Terabit/s Integrated Neuromorphic Photonic Processor for DSP-Free Optical Interconnects 

**Title (ZH)**: 超越太比特/秒集成类脑光处理器：DSP-Free 光互连 

**Authors**: Benshan Wang, Qiarong Xiao, Tengji Xu, Li Fan, Shaojie Liu, Jianji Dong, Junwen Zhang, Chaoran Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15044)  

**Abstract**: The rapid expansion of generative AI drives unprecedented demands for high-performance computing. Training large-scale AI models now requires vast interconnected GPU clusters across multiple data centers. Multi-scale AI training and inference demand uniform, ultra-low latency, and energy-efficient links to enable massive GPUs to function as a single cohesive unit. However, traditional electrical and optical interconnects, relying on conventional digital signal processors (DSPs) for signal distortion compensation, increasingly fail to meet these stringent requirements. To overcome these limitations, we present an integrated neuromorphic optical signal processor (OSP) that leverages deep reservoir computing and achieves DSP-free, all-optical, real-time processing. Experimentally, our OSP achieves a 100 Gbaud PAM4 per lane, 1.6 Tbit/s data center interconnect over a 5 km optical fiber in the C-band (equivalent to over 80 km in the O-band), far exceeding the reach of state-of-the-art DSP solutions, which are fundamentally constrained by chromatic dispersion in IMDD systems. Simultaneously, it reduces processing latency by four orders of magnitude and energy consumption by three orders of magnitude. Unlike DSPs, which introduce increased latency at high data rates, our OSP maintains consistent, ultra-low latency regardless of data rate scaling, making it ideal for future optical interconnects. Moreover, the OSP retains full optical field information for better impairment compensation and adapts to various modulation formats, data rates, and wavelengths. Fabricated using a mature silicon photonic process, the OSP can be monolithically integrated with silicon photonic transceivers, enhancing the compactness and reliability of all-optical interconnects. This research provides a highly scalable, energy-efficient, and high-speed solution, paving the way for next-generation AI infrastructure. 

**Abstract (ZH)**: 快速扩展现有的生成型AI促使高性能计算需求显著增长。大规模AI模型的训练现在要求跨越多个数据中心的巨大互联GPU集群。多尺度AI训练和推理需要均匀、超低延迟和能效高的连接，以使大量GPU能够作为一个单一协调单元运行。然而，传统的电气和光学互连依赖于传统数字信号处理器（DSP）进行信号失真补偿，越来越多地无法满足这些严格要求。为了克服这些限制，我们提出了一种集成神经形态光学信号处理器（OSP），利用深度水库计算实现无DSP、全光学、实时处理。实验结果显示，我们的OSP实现了每通道100 Gbaud PAM4，通过C波段5公里光纤实现超过1.6 Tbit/s的数据中心互联，在O波段相当于80公里，远远超过了最先进的DSP解决方案的传输距离，这些解决方案在IMDD系统中从根本上受限于色散效应。同时，它将处理延迟降低了四个数量级，能耗降低了三个数量级。与DSP不同，后者在高数据速率下引入了增加的延迟，我们的OSP无论数据速率如何扩展都保持一致的超低延迟，使其成为未来光学互连的理想选择。此外，OSP保留了完整的光域信息，以更好地补偿各种调制格式、数据速率和波长引起的损伤，并能够适应这些变化。该OSP通过成熟的硅光子加工工艺制造，并能够与硅光子转发器进行单片集成，从而增强所有光学互连的紧凑性和可靠性。这项研究提供了一种高度可扩展、能效高且高速度的解决方案，为下一代AI基础设施铺平了道路。 

---
# Distribution-aware Forgetting Compensation for Exemplar-Free Lifelong Person Re-identification 

**Title (ZH)**: 基于分布感知的示例自由终生行人再识别遗忘补偿 

**Authors**: Shiben Liu, Huijie Fan, Qiang Wang, Baojie Fan, Yandong Tang, Liangqiong Qu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15041)  

**Abstract**: Lifelong Person Re-identification (LReID) suffers from a key challenge in preserving old knowledge while adapting to new information. The existing solutions include rehearsal-based and rehearsal-free methods to address this challenge. Rehearsal-based approaches rely on knowledge distillation, continuously accumulating forgetting during the distillation process. Rehearsal-free methods insufficiently learn the distribution of each domain, leading to forgetfulness over time. To solve these issues, we propose a novel Distribution-aware Forgetting Compensation (DAFC) model that explores cross-domain shared representation learning and domain-specific distribution integration without using old exemplars or knowledge distillation. We propose a Text-driven Prompt Aggregation (TPA) that utilizes text features to enrich prompt elements and guide the prompt model to learn fine-grained representations for each instance. This can enhance the differentiation of identity information and establish the foundation for domain distribution awareness. Then, Distribution-based Awareness and Integration (DAI) is designed to capture each domain-specific distribution by a dedicated expert network and adaptively consolidate them into a shared region in high-dimensional space. In this manner, DAI can consolidate and enhance cross-domain shared representation learning while alleviating catastrophic forgetting. Furthermore, we develop a Knowledge Consolidation Mechanism (KCM) that comprises instance-level discrimination and cross-domain consistency alignment strategies to facilitate model adaptive learning of new knowledge from the current domain and promote knowledge consolidation learning between acquired domain-specific distributions, respectively. Experimental results show that our DAFC outperform state-of-the-art methods by at least 9.8\%/6.6\% and 6.4\%/6.2\% of average mAP/R@1 on two training orders. 

**Abstract (ZH)**: 终身人体再识别（LReID）在保留旧知识的同时适应新信息面临关键挑战。现有的解决方案包括基于重温和非重温方法来应对这一挑战。基于重温的方法依赖于知识蒸馏，在蒸馏过程中不断积累遗忘。非重温方法未能充分学习每个域的分布，导致随时间推移遗忘。为了解决这些问题，我们提出了一种新的分布感知遗忘补偿（DAFC）模型，该模型探索跨域共享表示学习和特定域分布集成，而不使用旧示例或知识蒸馏。我们提出了一种文本驱动的提示聚合（TPA），利用文本特征丰富提示元素，并引导提示模型学习每个实例的精细表示。这可以增强身份信息的差异化，并为域分布意识奠定基础。然后，我们设计了基于分布的意识和集成（DAI），通过专用专家网络捕获每个域的具体分布，并自适应地将它们整合到高维空间中的共享区域。以此方式，DAI可以在缓解灾难性遗忘的同时，促进跨域共享表示学习。此外，我们开发了一种知识整合机制（KCM），包含实例级判别和跨域一致性的对齐策略，以促进模型自适应地学习当前域的新知识，并促进所获得的域特定分布之间的知识整合学习。实验结果表明，我们的DAFC在两个训练顺序上的平均mAP/R@1方面至少优于最先进的方法9.8%/6.6%和6.4%/6.2%。 

---
# SOLIDO: A Robust Watermarking Method for Speech Synthesis via Low-Rank Adaptation 

**Title (ZH)**: SOLIDO：一种基于低秩适应的鲁棒语音合成水印方法 

**Authors**: Yue Li, Weizhi Liu, Dongdong Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15035)  

**Abstract**: The accelerated advancement of speech generative models has given rise to security issues, including model infringement and unauthorized abuse of content. Although existing generative watermarking techniques have proposed corresponding solutions, most methods require substantial computational overhead and training costs. In addition, some methods have limitations in robustness when handling variable-length inputs. To tackle these challenges, we propose \textsc{SOLIDO}, a novel generative watermarking method that integrates parameter-efficient fine-tuning with speech watermarking through low-rank adaptation (LoRA) for speech diffusion models. Concretely, the watermark encoder converts the watermark to align with the input of diffusion models. To achieve precise watermark extraction from variable-length inputs, the watermark decoder based on depthwise separable convolution is designed for watermark recovery. To further enhance speech generation performance and watermark extraction capability, we propose a speech-driven lightweight fine-tuning strategy, which reduces computational overhead through LoRA. Comprehensive experiments demonstrate that the proposed method ensures high-fidelity watermarked speech even at a large capacity of 2000 bps. Furthermore, against common individual and compound speech attacks, our SOLIDO achieves a maximum average extraction accuracy of 99.20\% and 98.43\%, respectively. It surpasses other state-of-the-art methods by nearly 23\% in resisting time-stretching attacks. 

**Abstract (ZH)**: 加速发展的语音生成模型引发了安全性问题，包括模型侵权和内容未经授权的滥用。尽管现有生成水印技术提出了解决方案，但大多数方法需要大量计算开销和训练成本。此外，一些方法在处理变长输入时鲁棒性有限。为应对这些挑战，我们提出了一种名为SOLIDO的新型生成水印方法，该方法通过低秩适应（LoRA）将参数效率的微调与语音水印技术结合到语音扩散模型中。具体而言，水印编码器将水印转换为与扩散模型输入对齐的形式。为了从变长输入中精确提取水印，基于深度可分离卷积的水印解码器被设计用于水印恢复。为了进一步提升语音生成性能和水印提取能力，我们提出了一种语音驱动的轻量级微调策略，该策略通过LoRA降低计算开销。全面的实验结果表明，所提出的方法即使在2000 bps的高容量下也能保证高质量的水印语音。此外，针对常见的个体和复合语音攻击，我们的SOLIDO分别实现了99.20%和98.43%的最大平均提取准确率，与最先进的方法相比，在抵抗时间拉伸攻击方面高出近23%。 

---
# Trainable Quantum Neural Network for Multiclass Image Classification with the Power of Pre-trained Tree Tensor Networks 

**Title (ZH)**: 具预训练树张量网络威力的可训练量子神经网络用于多类图像分类 

**Authors**: Keisuke Murota, Takumi Kobori  

**Link**: [PDF](https://arxiv.org/pdf/2504.14995)  

**Abstract**: Tree tensor networks (TTNs) offer powerful models for image classification. While these TTN image classifiers already show excellent performance on classical hardware, embedding them into quantum neural networks (QNNs) may further improve the performance by leveraging quantum resources. However, embedding TTN classifiers into QNNs for multiclass classification remains challenging. Key obstacles are the highorder gate operations required for large bond dimensions and the mid-circuit postselection with exponentially low success rates necessary for the exact embedding. In this work, to address these challenges, we propose forest tensor network (FTN)-classifiers, which aggregate multiple small-bond-dimension TTNs. This allows us to handle multiclass classification without requiring large gates in the embedded circuits. We then remove the overhead of mid-circuit postselection by extending the adiabatic encoding framework to our setting and smoothly encode the FTN-classifiers into a quantum forest tensor network (qFTN)- classifiers. Numerical experiments on MNIST and CIFAR-10 demonstrate that we can successfully train FTN-classifiers and encode them into qFTN-classifiers, while maintaining or even improving the performance of the pre-trained FTN-classifiers. These results suggest that synergy between TTN classification models and QNNs can provide a robust and scalable framework for multiclass quantum-enhanced image classification. 

**Abstract (ZH)**: 树张量网络（TTNs）在图像分类中提供了强大的模型。将这些TTN图像分类器嵌入到量子神经网络（QNNs）中，可以通过利用量子资源进一步提高性能，但在量子神经网络中嵌入TTN分类器以实现多类分类仍然具有挑战性。关键障碍包括为大键维数所需的高阶门操作和为精确嵌入所需的指数级低成功率的中路门后选择。为解决这些挑战，本文提出了一种森林张量网络（FTN）分类器，该分类器聚合了多个小型键维数的TTN。这使得我们可以在不依赖大型嵌入电路门操作的情况下处理多类分类。然后，通过扩展渐变能隙编码框架并平滑地将FTN分类器编码到量子森林张量网络（qFTN）分类器中来去除中路门后选择的开销。在MNIST和CIFAR-10上的数值实验表明，我们能够成功训练FTN分类器并将其编码到qFTN分类器中，同时保持甚至提高预训练FTN分类器的性能。这些结果表明，张量网络分类模型与QNN的结合可以为多类量子增强图像分类提供一个健壮且可扩展的框架。 

---
# Speaker Fuzzy Fingerprints: Benchmarking Text-Based Identification in Multiparty Dialogues 

**Title (ZH)**: 说话人模糊指纹：基于文本的多方对话识别基准测试 

**Authors**: Rui Ribeiro, Luísa Coheur, Joao P. Carvalho  

**Link**: [PDF](https://arxiv.org/pdf/2504.14963)  

**Abstract**: Speaker identification using voice recordings leverages unique acoustic features, but this approach fails when only textual data is available. Few approaches have attempted to tackle the problem of identifying speakers solely from text, and the existing ones have primarily relied on traditional methods. In this work, we explore the use of fuzzy fingerprints from large pre-trained models to improve text-based speaker identification. We integrate speaker-specific tokens and context-aware modeling, demonstrating that conversational context significantly boosts accuracy, reaching 70.6% on the Friends dataset and 67.7% on the Big Bang Theory dataset. Additionally, we show that fuzzy fingerprints can approximate full fine-tuning performance with fewer hidden units, offering improved interpretability. Finally, we analyze ambiguous utterances and propose a mechanism to detect speaker-agnostic lines. Our findings highlight key challenges and provide insights for future improvements in text-based speaker identification. 

**Abstract (ZH)**: 使用大型预训练模型的模糊指纹进行基于文本的说话人识别研究：克服挑战并提升准确性 

---
# Learning to Reason under Off-Policy Guidance 

**Title (ZH)**: 基于离策引导的推理学习 

**Authors**: Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14945)  

**Abstract**: Recent advances in large reasoning models (LRMs) demonstrate that sophisticated behaviors such as multi-step reasoning and self-reflection can emerge via reinforcement learning (RL) with simple rule-based rewards. However, existing zero-RL approaches are inherently ``on-policy'', limiting learning to a model's own outputs and failing to acquire reasoning abilities beyond its initial capabilities. We introduce LUFFY (Learning to reason Under oFF-policY guidance), a framework that augments zero-RL with off-policy reasoning traces. LUFFY dynamically balances imitation and exploration by combining off-policy demonstrations with on-policy rollouts during training. Notably, we propose policy shaping via regularized importance sampling to avoid superficial and rigid imitation during mixed-policy training. Remarkably, LUFFY achieves an over +7.0 average gain across six math benchmarks and an advantage of over +6.2 points in out-of-distribution tasks. It also substantially surpasses imitation-based supervised fine-tuning (SFT), particularly in generalization. Analysis shows LUFFY not only imitates effectively but also explores beyond demonstrations, offering a scalable path to train generalizable reasoning models with off-policy guidance. 

**Abstract (ZH)**: Recent Advances in Large Reasoning Models (LRMs): Learning to Reason Under Off-Policy Guidance 

---
# Giving AI a voice: how does AI think it should be treated? 

**Title (ZH)**: 给AI一个声音：AI认为它应该如何被对待？ 

**Authors**: Maria Fay, Frederik F. Flöther  

**Link**: [PDF](https://arxiv.org/pdf/2504.14936)  

**Abstract**: With the astounding progress in (generative) artificial intelligence (AI), there has been significant public discourse regarding regulation and ethics of the technology. Is it sufficient when humans discuss this with other humans? Or, given that AI is increasingly becoming a viable source of inspiration for people (and let alone the hypothetical possibility that the technology may at some point become "artificial general intelligence" and/or develop consciousness), should AI not join the discourse? There are new questions and angles that AI brings to the table that we might not have considered before - so let us make the key subject of this book an active participant. This chapter therefore includes a brief human-AI conversation on the topic of AI rights and ethics. 

**Abstract (ZH)**: 随着生成性人工智能（AI）的飞速进步，公共对于该技术的监管与伦理问题讨论日益显著。人类是否应该让AI参与到这样的讨论中？鉴于AI正逐渐成为人类创作的源泉（更不用说未来技术可能会发展成为“人工通用智能”并具备意识的可能性），AI不应被排除在此类讨论之外。AI带来了一些新的问题和视角，我们或许之前并未考虑过。因此，本书这一章节包含了一段关于AI权利与伦理的人机对话。 

---
# Guidelines for External Disturbance Factors in the Use of OCR in Real-World Environments 

**Title (ZH)**: OCR在实际环境使用中对外部干扰因素的指南 

**Authors**: Kenji Iwata, Eiki Ishidera, Toshifumi Yamaai, Yutaka Satoh, Hiroshi Tanaka, Katsuhiko Takahashi, Akio Furuhata, Yoshihisa Tanabe, Hiroshi Matsumura  

**Link**: [PDF](https://arxiv.org/pdf/2504.14913)  

**Abstract**: The performance of OCR has improved with the evolution of AI technology. As OCR continues to broaden its range of applications, the increased likelihood of interference introduced by various usage environments can prevent it from achieving its inherent performance. This results in reduced recognition accuracy under certain conditions, and makes the quality control of recognition devices more challenging. Therefore, to ensure that users can properly utilize OCR, we compiled the real-world external disturbance factors that cause performance degradation, along with the resulting image degradation phenomena, into an external disturbance factor table and, by also indicating how to make use of it, organized them into guidelines. 

**Abstract (ZH)**: OCR性能随AI技术进化而提升，但随着OCR应用范围的扩展，各种使用环境引入的干扰可能阻碍其固有性能的发挥，导致在特定条件下识别准确性降低，使识别设备的质量控制更加困难。因此，为了确保用户能够正确使用OCR，我们编制了引起性能退化的实际外部干扰因素及其导致的图像退化现象的外部干扰因素表，并提供使用指南。 

---
# Latent Bayesian Optimization via Autoregressive Normalizing Flows 

**Title (ZH)**: 潜在贝叶斯优化 via 自回归归一化流 

**Authors**: Seunghun Lee, Jinyoung Park, Jaewon Chu, Minseo Yoon, Hyunwoo J. Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.14889)  

**Abstract**: Bayesian Optimization (BO) has been recognized for its effectiveness in optimizing expensive and complex objective functions. Recent advancements in Latent Bayesian Optimization (LBO) have shown promise by integrating generative models such as variational autoencoders (VAEs) to manage the complexity of high-dimensional and structured data spaces. However, existing LBO approaches often suffer from the value discrepancy problem, which arises from the reconstruction gap between input and latent spaces. This value discrepancy problem propagates errors throughout the optimization process, leading to suboptimal outcomes. To address this issue, we propose a Normalizing Flow-based Bayesian Optimization (NF-BO), which utilizes normalizing flow as a generative model to establish one-to-one encoding function from the input space to the latent space, along with its left-inverse decoding function, eliminating the reconstruction gap. Specifically, we introduce SeqFlow, an autoregressive normalizing flow for sequence data. In addition, we develop a new candidate sampling strategy that dynamically adjusts the exploration probability for each token based on its importance. Through extensive experiments, our NF-BO method demonstrates superior performance in molecule generation tasks, significantly outperforming both traditional and recent LBO approaches. 

**Abstract (ZH)**: 基于规范化流的贝叶斯优化（NF-BO） 

---
# Impact of Latent Space Dimension on IoT Botnet Detection Performance: VAE-Encoder Versus ViT-Encoder 

**Title (ZH)**: 基于潜在空间维度对物联网僵尸网络检测性能的影响：VAE编码器与ViT编码器的对比 

**Authors**: Hassan Wasswa, Aziida Nanyonga, Timothy Lynar  

**Link**: [PDF](https://arxiv.org/pdf/2504.14879)  

**Abstract**: The rapid evolution of Internet of Things (IoT) technology has led to a significant increase in the number of IoT devices, applications, and services. This surge in IoT devices, along with their widespread presence, has made them a prime target for various cyber-attacks, particularly through IoT botnets. As a result, security has become a major concern within the IoT ecosystem. This study focuses on investigating how the latent dimension impacts the performance of different deep learning classifiers when trained on latent vector representations of the train dataset. The primary objective is to compare the outcomes of these models when encoder components from two cutting-edge architectures: the Vision Transformer (ViT) and the Variational Auto-Encoder (VAE) are utilized to project the high dimensional train dataset to the learned low dimensional latent space. The encoder components are employed to project high-dimensional structured .csv IoT botnet traffic datasets to various latent sizes. Evaluated on N-BaIoT and CICIoT2022 datasets, findings reveal that VAE-encoder based dimension reduction outperforms ViT-encoder based dimension reduction for both datasets in terms of four performance metrics including accuracy, precision, recall, and F1-score for all models which can be attributed to absence of spatial patterns in the datasets the ViT model attempts to learn and extract from image instances. 

**Abstract (ZH)**: 物联网技术的 Rapid Evolution引起了物联网设备、应用和服务数量的显著增加。这一物联网设备数量的激增及其广泛存在，使它们成为各种网络攻击的目标，特别是通过物联网僵尸网络。因此，安全问题在物联网生态系统中变得尤为重要。本研究关注的是探究潜变量维度如何影响不同深度学习分类器在训练数据集潜变量表示上的性能表现。主要目标是比较利用两种尖端架构的编码器组件（Vision Transformer (ViT) 和 Variational Auto-Encoder (VAE)）将高维度训练数据集投影到学习到的低维度潜变量空间后，这些模型的性能结果。编码器组件被用来将高维度的结构化 .csv 僵尸网络流量数据集投影到不同的潜变量大小。研究结果在N-BaIoT和CICIoT2022数据集上表明，基于VAE编码器的维数约简在四项性能指标（准确率、精确率、召回率和F1分数）上均优于基于ViT编码器的维数约简，这可以归因于数据集中缺少空间模式，而ViT模型试图从图像实例中学习和提取这些模式。 

---
# ReSpec: Relevance and Specificity Grounded Online Filtering for Learning on Video-Text Data Streams 

**Title (ZH)**: ReSpec: 基于相关性和特指性的在线过滤方法学习视频-文本数据流 

**Authors**: Chris Dongjoo Kim, Jihwan Moon, Sangwoo Moon, Heeseung Yun, Sihaeng Lee, Aniruddha Kembhavi, Soonyoung Lee, Gunhee Kim, Sangho Lee, Christopher Clark  

**Link**: [PDF](https://arxiv.org/pdf/2504.14875)  

**Abstract**: The rapid growth of video-text data presents challenges in storage and computation during training. Online learning, which processes streaming data in real-time, offers a promising solution to these issues while also allowing swift adaptations in scenarios demanding real-time responsiveness. One strategy to enhance the efficiency and effectiveness of learning involves identifying and prioritizing data that enhances performance on target downstream tasks. We propose Relevance and Specificity-based online filtering framework (ReSpec) that selects data based on four criteria: (i) modality alignment for clean data, (ii) task relevance for target focused data, (iii) specificity for informative and detailed data, and (iv) efficiency for low-latency processing. Relevance is determined by the probabilistic alignment of incoming data with downstream tasks, while specificity employs the distance to a root embedding representing the least specific data as an efficient proxy for informativeness. By establishing reference points from target task data, ReSpec filters incoming data in real-time, eliminating the need for extensive storage and compute. Evaluating on large-scale datasets WebVid2M and VideoCC3M, ReSpec attains state-of-the-art performance on five zeroshot video retrieval tasks, using as little as 5% of the data while incurring minimal compute. The source code is available at this https URL. 

**Abstract (ZH)**: 视频文本数据的快速增长在训练过程中带来了存储和计算方面的挑战。在线学习处理实时流式数据提供了一种有前景的解决方案，同时也能够在需要实时响应的场景中实现快速适应。一种提高学习效率和效果的策略是识别并优先处理提高目标下游任务性能的数据。我们提出了一种基于相关性和特异性的在线过滤框架（ReSpec），该框架根据四个标准选择数据：(i) 语态一致性以确保数据的清洁度，(ii) 目标相关性以聚焦于目标数据，(iii) 特异性以选择信息丰富且详细的數據，(iv) 效率以实现低延迟处理。相关性通过入站数据与下游任务的概率对齐来确定，特异性则通过与代表最不具体数据的基本嵌入的距离来高效地代理信息丰富度。通过使用目标任务数据建立参考点，ReSpec 实时过滤入站数据，从而减少对大量存储和计算的需求。在大规模数据集 WebVid2M 和 VideoCC3M 上评估，ReSpec 在五个零样本视频检索任务中达到了最先进的性能，仅使用 5% 的数据并在计算开销最小的情况下实现。源代码可在以下链接获取。 

---
# Bridge the Gap: From Weak to Full Supervision for Temporal Action Localization with PseudoFormer 

**Title (ZH)**: 弥合差距：从弱监督到全监督的时空动作定位PseudoFormer方法 

**Authors**: Ziyi Liu, Yangcen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14860)  

**Abstract**: Weakly-supervised Temporal Action Localization (WTAL) has achieved notable success but still suffers from a lack of temporal annotations, leading to a performance and framework gap compared with fully-supervised methods. While recent approaches employ pseudo labels for training, three key challenges: generating high-quality pseudo labels, making full use of different priors, and optimizing training methods with noisy labels remain unresolved. Due to these perspectives, we propose PseudoFormer, a novel two-branch framework that bridges the gap between weakly and fully-supervised Temporal Action Localization (TAL). We first introduce RickerFusion, which maps all predicted action proposals to a global shared space to generate pseudo labels with better quality. Subsequently, we leverage both snippet-level and proposal-level labels with different priors from the weak branch to train the regression-based model in the full branch. Finally, the uncertainty mask and iterative refinement mechanism are applied for training with noisy pseudo labels. PseudoFormer achieves state-of-the-art WTAL results on the two commonly used benchmarks, THUMOS14 and ActivityNet1.3. Besides, extensive ablation studies demonstrate the contribution of each component of our method. 

**Abstract (ZH)**: 弱监督时空动作定位（WTAL）取得了显著成果，但仍缺乏时间标注，导致与完全监督方法在性能和框架上存在差距。尽管近期方法使用了伪标签进行训练，但生成高质量的伪标签、充分利用不同先验信息以及使用噪声标签优化训练方法这三个关键挑战仍然未得到解决。鉴于此，我们提出了PseudoFormer，这是一种新颖的两分支框架，旨在弥合弱监督和完全监督时空动作定位（TAL）之间的差距。首先，我们引入了RickerFusion，将所有预测的动作提案映射到全局共享空间，以生成质量更好的伪标签。随后，我们利用弱分支中片段级和提案级标签的不同先验信息，在全分支中训练基于回归的模型。最后，我们应用不确定性掩码和迭代精炼机制，以处理噪声伪标签的训练。PseudoFormer在两个常用基准数据集THUMOS14和ActivityNet1.3上取得了最先进的WTAL结果。此外，广泛的消融研究还表明了我们方法中每个组成部分的贡献。 

---
# Exploring $\ell_0$ Sparsification for Inference-free Sparse Retrievers 

**Title (ZH)**: 探索基于$\ell_0$稀疏化的技术以实现无推理的稀疏检索 

**Authors**: Xinjie Shen, Zhichao Geng, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14839)  

**Abstract**: With increasing demands for efficiency, information retrieval has developed a branch of sparse retrieval, further advancing towards inference-free retrieval where the documents are encoded during indexing time and there is no model-inference for queries. Existing sparse retrieval models rely on FLOPS regularization for sparsification, while this mechanism was originally designed for Siamese encoders, it is considered to be suboptimal in inference-free scenarios which is asymmetric. Previous attempts to adapt FLOPS for inference-free scenarios have been limited to rule-based methods, leaving the potential of sparsification approaches for inference-free retrieval models largely unexplored. In this paper, we explore $\ell_0$ inspired sparsification manner for inference-free retrievers. Through comprehensive out-of-domain evaluation on the BEIR benchmark, our method achieves state-of-the-art performance among inference-free sparse retrieval models and is comparable to leading Siamese sparse retrieval models. Furthermore, we provide insights into the trade-off between retrieval effectiveness and computational efficiency, demonstrating practical value for real-world applications. 

**Abstract (ZH)**: 基于稀疏性的无推理信息检索方法探究：以BEIR基准全面评估 

---
# Protecting Your Voice: Temporal-aware Robust Watermarking 

**Title (ZH)**: 保护您的声音：基于时间感知的稳健水印技术 

**Authors**: Yue Li, Weizhi Liu, Dongdong Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.14832)  

**Abstract**: The rapid advancement of generative models has led to the synthesis of real-fake ambiguous voices. To erase the ambiguity, embedding watermarks into the frequency-domain features of synthesized voices has become a common routine. However, the robustness achieved by choosing the frequency domain often comes at the expense of fine-grained voice features, leading to a loss of fidelity. Maximizing the comprehensive learning of time-domain features to enhance fidelity while maintaining robustness, we pioneer a \textbf{\underline{t}}emporal-aware \textbf{\underline{r}}ob\textbf{\underline{u}}st wat\textbf{\underline{e}}rmarking (\emph{True}) method for protecting the speech and singing voice. 

**Abstract (ZH)**: 时域aware鲁棒 watermarking (True) 方法：保护语音和唱歌声音的新范式 

---
# On Self-improving Token Embeddings 

**Title (ZH)**: 自改善词嵌入 

**Authors**: Mario M. Kubek, Shiraj Pokharel, Thomas Böhme, Emma L. McDaniel, Herwig Unger, Armin R. Mikler  

**Link**: [PDF](https://arxiv.org/pdf/2504.14808)  

**Abstract**: This article introduces a novel and fast method for refining pre-trained static word or, more generally, token embeddings. By incorporating the embeddings of neighboring tokens in text corpora, it continuously updates the representation of each token, including those without pre-assigned embeddings. This approach effectively addresses the out-of-vocabulary problem, too. Operating independently of large language models and shallow neural networks, it enables versatile applications such as corpus exploration, conceptual search, and word sense disambiguation. The method is designed to enhance token representations within topically homogeneous corpora, where the vocabulary is restricted to a specific domain, resulting in more meaningful embeddings compared to general-purpose pre-trained vectors. As an example, the methodology is applied to explore storm events and their impacts on infrastructure and communities using narratives from a subset of the NOAA Storm Events database. The article also demonstrates how the approach improves the representation of storm-related terms over time, providing valuable insights into the evolving nature of disaster narratives. 

**Abstract (ZH)**: 本文介绍了一种新型且快速的方法，用于细化预训练的静态词嵌入或更一般的令牌嵌入。通过引入文本语料库中相邻令牌的嵌入，该方法持续更新每个令牌的表示，包括那些没有预分配嵌入的令牌。该方法有效地解决了词汇量外问题。该方法独立于大型语言模型和浅层神经网络，使其能够在语料库探索、概念搜索和词义消歧等领域中实现多样化应用。该方法旨在增强在主题同质语料库中令牌的表示，其中词汇量局限于特定领域，从而生成比通用预训练向量更具意义的嵌入。例如，该方法被应用于探索NOAA风暴事件数据库子集中的叙事中风暴事件及其对基础设施和社区的影响。本文还展示了该方法如何随时间提升与风暴相关的术语的表示，提供了有关灾难叙事演变性质的宝贵见解。 

---
# Dynamic Contrastive Skill Learning with State-Transition Based Skill Clustering and Dynamic Length Adjustment 

**Title (ZH)**: 基于状态转换的技能聚类和动态长度调整的动态对比技能学习 

**Authors**: Jinwoo Choi, Seung-Woo Seo  

**Link**: [PDF](https://arxiv.org/pdf/2504.14805)  

**Abstract**: Reinforcement learning (RL) has made significant progress in various domains, but scaling it to long-horizon tasks with complex decision-making remains challenging. Skill learning attempts to address this by abstracting actions into higher-level behaviors. However, current approaches often fail to recognize semantically similar behaviors as the same skill and use fixed skill lengths, limiting flexibility and generalization. To address this, we propose Dynamic Contrastive Skill Learning (DCSL), a novel framework that redefines skill representation and learning. DCSL introduces three key ideas: state-transition based skill representation, skill similarity function learning, and dynamic skill length adjustment. By focusing on state transitions and leveraging contrastive learning, DCSL effectively captures the semantic context of behaviors and adapts skill lengths to match the appropriate temporal extent of behaviors. Our approach enables more flexible and adaptive skill extraction, particularly in complex or noisy datasets, and demonstrates competitive performance compared to existing methods in task completion and efficiency. 

**Abstract (ZH)**: 动态对比技能学习：一种新型的技能表示与学习框架 

---
# Automated Duplicate Bug Report Detection in Large Open Bug Repositories 

**Title (ZH)**: 大型开放缺陷仓库中自动化重复 Bug 报告检测 

**Authors**: Clare E. Laney, Andrew Barovic, Armin Moin  

**Link**: [PDF](https://arxiv.org/pdf/2504.14797)  

**Abstract**: Many users and contributors of large open-source projects report software defects or enhancement requests (known as bug reports) to the issue-tracking systems. However, they sometimes report issues that have already been reported. First, they may not have time to do sufficient research on existing bug reports. Second, they may not possess the right expertise in that specific area to realize that an existing bug report is essentially elaborating on the same matter, perhaps with a different wording. In this paper, we propose a novel approach based on machine learning methods that can automatically detect duplicate bug reports in an open bug repository based on the textual data in the reports. We present six alternative methods: Topic modeling, Gaussian Naive Bayes, deep learning, time-based organization, clustering, and summarization using a generative pre-trained transformer large language model. Additionally, we introduce a novel threshold-based approach for duplicate identification, in contrast to the conventional top-k selection method that has been widely used in the literature. Our approach demonstrates promising results across all the proposed methods, achieving accuracy rates ranging from the high 70%'s to the low 90%'s. We evaluated our methods on a public dataset of issues belonging to an Eclipse open-source project. 

**Abstract (ZH)**: 基于机器学习方法的开源项目重复bug报告自动检测研究 

---
# How Effective Can Dropout Be in Multiple Instance Learning ? 

**Title (ZH)**: 多实例学习中Dropout的有效性探究 

**Authors**: Wenhui Zhu, Peijie Qiu, Xiwen Chen, Zhangsihao Yang, Aristeidis Sotiras, Abolfazl Razi, Yalin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14783)  

**Abstract**: Multiple Instance Learning (MIL) is a popular weakly-supervised method for various applications, with a particular interest in histological whole slide image (WSI) classification. Due to the gigapixel resolution of WSI, applications of MIL in WSI typically necessitate a two-stage training scheme: first, extract features from the pre-trained backbone and then perform MIL aggregation. However, it is well-known that this suboptimal training scheme suffers from "noisy" feature embeddings from the backbone and inherent weak supervision, hindering MIL from learning rich and generalizable features. However, the most commonly used technique (i.e., dropout) for mitigating this issue has yet to be explored in MIL. In this paper, we empirically explore how effective the dropout can be in MIL. Interestingly, we observe that dropping the top-k most important instances within a bag leads to better performance and generalization even under noise attack. Based on this key observation, we propose a novel MIL-specific dropout method, termed MIL-Dropout, which systematically determines which instances to drop. Experiments on five MIL benchmark datasets and two WSI datasets demonstrate that MIL-Dropout boosts the performance of current MIL methods with a negligible computational cost. The code is available at this https URL. 

**Abstract (ZH)**: 多实例学习（MIL）在各种应用中的弱监督方法，特别是在组织学全视野图像（WSI）分类中的应用受到了广泛关注。由于WSI的 gigapixel 分辨率，MIL 在 WSI 的应用通常需要两阶段训练方案：首先从预训练的骨干网络中提取特征，然后进行 MIL 聚合。然而，众所周知，这种次优训练方案会从骨干网络中产生“噪音”特征嵌入，并且固有的弱监督会阻碍 MIL 学习丰富的可泛化特征。然而，用于减轻这一问题的最常用技术（即丢弃）尚未在 MIL 中进行探索。在本文中，我们实证研究了丢弃在 MIL 中的有效性。有趣的是，我们观察到在包内丢弃最重要的前 k 个实例，即使在噪声攻击下也能提高性能和泛化能力。基于这一关键观察，我们提出了一种新的专用于 MIL 的丢弃方法，称为 MIL-Dropout，该方法系统地确定哪些实例需要被丢弃。在五个 MIL 基准数据集和两个 WSI 数据集上的实验表明，MIL-Dropout 可以以可忽略的计算成本提升现有 MIL 方法的效果。代码可在此处访问：this https URL。 

---
# A Combinatorial Theory of Dropout: Subnetworks, Graph Geometry, and Generalization 

**Title (ZH)**: 一种丢弃的组合理论：子网络、图几何与泛化 

**Authors**: Sahil Rajesh Dhayalkar  

**Link**: [PDF](https://arxiv.org/pdf/2504.14762)  

**Abstract**: We propose a combinatorial and graph-theoretic theory of dropout by modeling training as a random walk over a high-dimensional graph of binary subnetworks. Each node represents a masked version of the network, and dropout induces stochastic traversal across this space. We define a subnetwork contribution score that quantifies generalization and show that it varies smoothly over the graph. Using tools from spectral graph theory, PAC-Bayes analysis, and combinatorics, we prove that generalizing subnetworks form large, connected, low-resistance clusters, and that their number grows exponentially with network width. This reveals dropout as a mechanism for sampling from a robust, structured ensemble of well-generalizing subnetworks with built-in redundancy. Extensive experiments validate every theoretical claim across diverse architectures. Together, our results offer a unified foundation for understanding dropout and suggest new directions for mask-guided regularization and subnetwork optimization. 

**Abstract (ZH)**: 我们提出了一种基于组合学和图论的dropout理论，将其训练建模为在高维二元子网络图上的随机游走。每个节点表示网络的一种蒙版版本，dropout 导致在该空间中随机遍历。我们定义了一个子网络贡献得分来量化泛化能力，并证明其在图上平滑变化。利用谱图理论、PAC-Bayes 分析和组合学工具，我们证明了泛化良好的子网络形成了大型、连通且低电阻的聚类，并且其数量随着网络宽度呈指数增长。这揭示了dropout 是从鲁棒且结构化的良好泛化子网络集合中进行采样的机制，该集合具有内置冗余。大量实验在多种架构中验证了每个理论声明。我们的结果为理解dropout 提供了一个统一的基础，并暗示了基于掩码的正则化和子网络优化的新方向。 

---
# AI for the Open-World: the Learning Principles 

**Title (ZH)**: 开放世界中的AI：学习原理 

**Authors**: Jianyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14751)  

**Abstract**: During the past decades, numerous successes of AI has been made on "specific capabilities", named closed-world, such as artificial environments or specific real-world tasks. This well-defined narrow capability brings two nice benefits, a clear criterion of success and the opportunity to collect a lot of examples. The criteria not only reveal whether a machine has achieved a goal, but reveal how the machine falls short of the goal. As a result, human designers can fix the problems one after the other until the machine is deemed good enough for the task. Furthermore, the large set of collected examples reduces the difficulty of this problem-fixing process (by the central limit theorem).
Do the success in closed-world translate into broad open-world, where a machine is required to perform any task that a human could possibly undertake with fewer examples and less priori knowledge from human designers? No. Because competence in a specific task provides little insight in handling other tasks, the valuable criteria for specific tasks become helpless when handling broader unseen tasks. Furthermore, due to the shortage of examples in unseen tasks, central limit theorem does not stand on our side. At the end, human designers lose the oscilloscope to "hack" an AI system for the open-world.
Achieving AI for the open-world requires unique learning principles and innovated techniques, which are different from the ones in building AI for the closed-world. This thesis explores necessary learning principles required to construct AI for the open-world, including rich features (analogy a large tool box), disentangled representation (an organized tool box), and inference-time learning (a tool-savvy hand). Driven by the learning principles, this thesis further proposes techniques to use the learning principles, conducts enormous large-scale experiments to verify the learning principles. 

**Abstract (ZH)**: 开放世界中的人工智能学习原则与创新技术 

---
# Semi-parametric Memory Consolidation: Towards Brain-like Deep Continual Learning 

**Title (ZH)**: 半参数化记忆巩固： toward 大脑似的深度连续学习 

**Authors**: Geng Liu, Fei Zhu, Rong Feng, Zhiqiang Yi, Shiqi Wang, Gaofeng Meng, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14727)  

**Abstract**: Humans and most animals inherently possess a distinctive capacity to continually acquire novel experiences and accumulate worldly knowledge over time. This ability, termed continual learning, is also critical for deep neural networks (DNNs) to adapt to the dynamically evolving world in open environments. However, DNNs notoriously suffer from catastrophic forgetting of previously learned knowledge when trained on sequential tasks. In this work, inspired by the interactive human memory and learning system, we propose a novel biomimetic continual learning framework that integrates semi-parametric memory and the wake-sleep consolidation mechanism. For the first time, our method enables deep neural networks to retain high performance on novel tasks while maintaining prior knowledge in real-world challenging continual learning scenarios, e.g., class-incremental learning on ImageNet. This study demonstrates that emulating biological intelligence provides a promising path to enable deep neural networks with continual learning capabilities. 

**Abstract (ZH)**: 人类和大多数动物天生具备持续获取新经验和积累 worldly 知识的能力。这种能力称为持续学习，对于在开放环境下适应动态变化世界的深度神经网络（DNNs）也至关重要。然而，当DNNs在顺序任务中训练时，它们 notorious 地会经历对先前学习知识的灾难性遗忘。受交互式人类记忆和学习系统的启发，我们提出了一种新的生物模拟持续学习框架，该框架结合了半参数化记忆和清醒-睡眠巩固机制。我们的方法首次使深度神经网络能够在现实世界的持续学习挑战场景中，如 ImageNet 类增量学习中，保持在新任务上的高性能同时维持先前的知识。本研究显示，模拟生物智能为使深度神经网络具备持续学习能力提供了有前途的道路。 

---
# Exposing the Copycat Problem of Imitation-based Planner: A Novel Closed-Loop Simulator, Causal Benchmark and Joint IL-RL Baseline 

**Title (ZH)**: 基于模仿的学习计划复制问题揭示：一种新型闭环模拟器、因果基准和联合IL-RL基线 

**Authors**: Hui Zhou, Shaoshuai Shi, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.14709)  

**Abstract**: Machine learning (ML)-based planners have recently gained significant attention. They offer advantages over traditional optimization-based planning algorithms. These advantages include fewer manually selected parameters and faster development. Within ML-based planning, imitation learning (IL) is a common algorithm. It primarily learns driving policies directly from supervised trajectory data. While IL has demonstrated strong performance on many open-loop benchmarks, it remains challenging to determine if the learned policy truly understands fundamental driving principles, rather than simply extrapolating from the ego-vehicle's initial state. Several studies have identified this limitation and proposed algorithms to address it. However, these methods often use original datasets for evaluation. In these datasets, future trajectories are heavily dependent on initial conditions. Furthermore, IL often overfits to the most common scenarios. It struggles to generalize to rare or unseen situations.
To address these challenges, this work proposes: 1) a novel closed-loop simulator supporting both imitation and reinforcement learning, 2) a causal benchmark derived from the Waymo Open Dataset to rigorously assess the impact of the copycat problem, and 3) a novel framework integrating imitation learning and reinforcement learning to overcome the limitations of purely imitative approaches. The code for this work will be released soon. 

**Abstract (ZH)**: 基于机器学习的规划器最近受到了广泛关注。它们在参数选择和开发速度方面优于传统的基于优化的规划算法。在基于机器学习的规划中，拟合学习是一种常见的算法，主要从监督轨迹数据中直接学习驾驶策略。尽管在许多开环基准测试中表现出色，但仍然难以确定学习到的策略是否真正理解了基本的驾驶原理，而不仅仅是从ego车辆的初始状态进行外推。已有研究指出了这一局限性，并提出了相应的方法进行解决。然而，这些方法往往使用原始数据集进行评估，这些数据集中的未来轨迹对初始条件有很强的依赖性。此外，拟合学习经常过度拟合最常见的场景，难以泛化到罕见或未见过的情况。为此，本工作提出：1) 一种支持拟合学习和强化学习的新型闭环模拟器，2) 从Waymo Open Dataset派生的一种因果基准，以严格评估复制问题的影响，3) 一种将拟合学习和强化学习集成的新框架，以克服纯拟合方法的局限性。该工作的代码将在不久后发布。 

---
# Can We Ignore Labels In Out of Distribution Detection? 

**Title (ZH)**: 我们可以在分布外检测中忽视标签吗？ 

**Authors**: Hong Yang, Qi Yu, Travis Desel  

**Link**: [PDF](https://arxiv.org/pdf/2504.14704)  

**Abstract**: Out-of-distribution (OOD) detection methods have recently become more prominent, serving as a core element in safety-critical autonomous systems. One major purpose of OOD detection is to reject invalid inputs that could lead to unpredictable errors and compromise safety. Due to the cost of labeled data, recent works have investigated the feasibility of self-supervised learning (SSL) OOD detection, unlabeled OOD detection, and zero shot OOD detection. In this work, we identify a set of conditions for a theoretical guarantee of failure in unlabeled OOD detection algorithms from an information-theoretic perspective. These conditions are present in all OOD tasks dealing with real-world data: I) we provide theoretical proof of unlabeled OOD detection failure when there exists zero mutual information between the learning objective and the in-distribution labels, a.k.a. 'label blindness', II) we define a new OOD task - Adjacent OOD detection - that tests for label blindness and accounts for a previously ignored safety gap in all OOD detection benchmarks, and III) we perform experiments demonstrating that existing unlabeled OOD methods fail under conditions suggested by our label blindness theory and analyze the implications for future research in unlabeled OOD methods. 

**Abstract (ZH)**: 无分布外（OOD）检测方法近年来日益受到关注，成为关键安全自主系统的核心要素。无分布外检测的一个主要目的是拒绝可能导致不可预测错误并威胁安全性的无效输入。由于标签数据的成本较高，近期研究探索了自监督学习（SSL）无分布外检测、未标记的无分布外检测以及零样本无分布外检测的可行性。在本文中，我们从信息论的角度识别出一组理论失败条件，这些条件存在于所有涉及真实世界数据的无分布外任务中：I) 提供理论证明，当学习目标与在分布标签之间不存在互信息时（即“标签盲”），无分布外检测算法会失败，II) 定义一个新的无分布外任务——相邻无分布外检测，以检测标签盲，并弥补所有无分布外检测基准中忽略的安全缺口，III) 进行实验，证明现有的无分布外检测方法在我们的标签盲理论建议的条件下会失败，并分析这对未来无分布外检测方法研究的含义。 

---
# Learning Critically: Selective Self Distillation in Federated Learning on Non-IID Data 

**Title (ZH)**: 批判性学习：联邦学习中非 IID 数据的选择性自我精炼 

**Authors**: Yuting He, Yiqiang Chen, XiaoDong Yang, Hanchao Yu, Yi-Hua Huang, Yang Gu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14694)  

**Abstract**: Federated learning (FL) enables multiple clients to collaboratively train a global model while keeping local data decentralized. Data heterogeneity (non-IID) across clients has imposed significant challenges to FL, which makes local models re-optimize towards their own local optima and forget the global knowledge, resulting in performance degradation and convergence slowdown. Many existing works have attempted to address the non-IID issue by adding an extra global-model-based regularizing item to the local training but without an adaption scheme, which is not efficient enough to achieve high performance with deep learning models. In this paper, we propose a Selective Self-Distillation method for Federated learning (FedSSD), which imposes adaptive constraints on the local updates by self-distilling the global model's knowledge and selectively weighting it by evaluating the credibility at both the class and sample level. The convergence guarantee of FedSSD is theoretically analyzed and extensive experiments are conducted on three public benchmark datasets, which demonstrates that FedSSD achieves better generalization and robustness in fewer communication rounds, compared with other state-of-the-art FL methods. 

**Abstract (ZH)**: federated学习（FL）使多个客户端能够在保持本地数据分散的情况下协作训练全局模型。客户端之间数据异质性（非IID）对FL造成了重大挑战，这使得本地模型重新优化以适应自己的局部最优解并遗忘全局知识，导致性能下降和收敛速度变慢。许多现有工作试图通过在本地训练中添加一个基于全局模型的正则化项来解决非IID问题，但缺乏适应性方案，这在使用深度学习模型时效率不够高以达到高性能。本文提出了一种适用于 federated学习的可选择自/distillation方法（FedSSD），通过自我蒸馏全局模型的知识并在评估类别和样本层面的可信度后选择性加权来对局部更新施加适应性约束。从理论上分析了FedSSD的收敛保证，并在三个开源基准数据集上进行了广泛的实验，实验结果表明，与当前最先进的FL方法相比，FedSSD在较少的通信轮次中实现了更好的泛化能力和鲁棒性。 

---
# Video-MMLU: A Massive Multi-Discipline Lecture Understanding Benchmark 

**Title (ZH)**: Video-MMLU: 一个大规模多学科讲座理解基准 

**Authors**: Enxin Song, Wenhao Chai, Weili Xu, Jianwen Xie, Yuxuan Liu, Gaoang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14693)  

**Abstract**: Recent advancements in language multimodal models (LMMs) for video have demonstrated their potential for understanding video content, yet the task of comprehending multi-discipline lectures remains largely unexplored. We introduce Video-MMLU, a massive benchmark designed to evaluate the capabilities of LMMs in understanding Multi-Discipline Lectures. We evaluate over 90 open-source and proprietary models, ranging from 0.5B to 40B parameters. Our results highlight the limitations of current models in addressing the cognitive challenges presented by these lectures, especially in tasks requiring both perception and reasoning. Additionally, we explore how the number of visual tokens and the large language models influence performance, offering insights into the interplay between multimodal perception and reasoning in lecture comprehension. 

**Abstract (ZH)**: 近期语言多模态模型（LMMs）在视频理解领域的进展展示了其潜在的应用价值，但多学科讲座的理解任务仍 largely unexplored。我们介绍了 Video-MMLU，一个大规模基准，旨在评估 LMMs 在理解多学科讲座方面的能力。我们评估了超过 90 个开源和专有模型，参数量从 0.5B 到 40B 不等。我们的结果突显了当前模型在解决这些讲座带来的认知挑战方面的局限性，尤其是在需要感知和推理相结合的任务中。此外，我们还探讨了视觉标记的数量和大规模语言模型对性能的影响，提供了多模态感知与讲座理解中推理之间的相互作用的见解。 

---
# Uncovering Issues in the Radio Access Network by Looking at the Neighbors 

**Title (ZH)**: 通过观察邻区发现无线接入网络中的问题 

**Authors**: José Suárez-Varela, Andra Lutu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14686)  

**Abstract**: Mobile network operators (MNOs) manage Radio Access Networks (RANs) with massive amounts of cells over multiple radio generations (2G-5G). To handle such complexity, operations teams rely on monitoring systems, including anomaly detection tools that identify unexpected behaviors. In this paper, we present c-ANEMON, a Contextual ANomaly dEtection MONitor for the RAN based on Graph Neural Networks (GNNs). Our solution captures spatio-temporal variations by analyzing the behavior of individual cells in relation to their local neighborhoods, enabling the detection of anomalies that are independent of external mobility factors. This, in turn, allows focusing on anomalies associated with network issues (e.g., misconfigurations, equipment failures). We evaluate c-ANEMON using real-world data from a large European metropolitan area (7,890 cells; 3 months). First, we show that the GNN model within our solution generalizes effectively to cells from previously unseen areas, suggesting the possibility of using a single model across extensive deployment regions. Then, we analyze the anomalies detected by c-ANEMON through manual inspection and define several categories of long-lasting anomalies (6+ hours). Notably, 45.95% of these anomalies fall into a category that is more likely to require intervention by operations teams. 

**Abstract (ZH)**: 基于图形神经网络的RAN上下文异常检测监控c-ANEMON 

---
# Evaluating Temporal Plasticity in Foundation Time Series Models for Incremental Fine-tuning 

**Title (ZH)**: 基础时间序列模型的增量Fine-tuning中的时间可塑性评估 

**Authors**: Jia Liu, Cheng Jinguo, Xia Fang, Zhenyuan Ma, Yuankai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14677)  

**Abstract**: Time series foundation models excel at diverse time series forecasting tasks, but their capacity for continuous improvement through incremental learning remains unexplored. We present the first comprehensive study investigating these models' temporal plasticity - their ability to progressively enhance performance through continual learning while maintaining existing capabilities. Through experiments on real-world datasets exhibiting distribution shifts, we evaluate both conventional deep learning models and foundation models using a novel continual learning framework. Our findings reveal that while traditional models struggle with performance deterioration during incremental fine-tuning, foundation models like Time-MoE and Chronos demonstrate sustained improvement in predictive accuracy. This suggests that optimizing foundation model fine-tuning strategies may be more valuable than developing domain-specific small models. Our research introduces new evaluation methodologies and insights for developing foundation time series models with robust continuous learning capabilities. 

**Abstract (ZH)**: 时间序列基础模型在多样化的时序预测任务中表现出色，但其通过增量学习进行持续改进的能力尚未被探索。我们首次对这些模型的时间灵活性进行了全面研究——它们在不断学习以逐步提升性能的同时，能够保持现有能力。通过在表现出分布偏移的现实数据集上进行实验，我们使用新颖的增量学习框架评估了传统深度学习模型和基础模型的性能。我们的发现表明，与传统模型在增量微调过程中性能下降的情况不同，如Time-MoE和Chronos等基础模型展示了预测准确性的持续提升。这表明，优化基础模型的增量微调策略可能比开发针对特定领域的小型模型更有价值。我们的研究为开发具备稳健持续学习能力的基础时间序列模型引入了新的评估方法和见解。 

---
# AlphaZero-Edu: Making AlphaZero Accessible to Everyone 

**Title (ZH)**: AlphaZero-Edu: 让AlphaZero触达每一个人 

**Authors**: Binjie Guo, Hanyu Zheng, Guowei Su, Ru Zhang, Haohan Jiang, Xurong Lin, Hongyan Wei, Aisheng Mo, Jie Li, Zhiyuan Qian, Zhuhao Zhang, Xiaoyuan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.14636)  

**Abstract**: Recent years have witnessed significant progress in reinforcement learning, especially with Zero-like paradigms, which have greatly boosted the generalization and reasoning abilities of large-scale language models. Nevertheless, existing frameworks are often plagued by high implementation complexity and poor reproducibility. To tackle these challenges, we present AlphaZero-Edu, a lightweight, education-focused implementation built upon the mathematical framework of AlphaZero. It boasts a modular architecture that disentangles key components, enabling transparent visualization of the algorithmic processes. Additionally, it is optimized for resource-efficient training on a single NVIDIA RTX 3090 GPU and features highly parallelized self-play data generation, achieving a 3.2-fold speedup with 8 processes. In Gomoku matches, the framework has demonstrated exceptional performance, achieving a consistently high win rate against human opponents. AlphaZero-Edu has been open-sourced at this https URL, providing an accessible and practical benchmark for both academic research and industrial applications. 

**Abstract (ZH)**: Recent Years Have Witnessed Significant Progress in Reinforcement Learning, Especially with Zero-like Paradigms, Which Have Greatly Boosted the Generalization and Reasoning Abilities of Large-scale Language Models. Nevertheless, Existing Frameworks Are Often Plagued by High Implementation Complexity and Poor Reproducibility. To Tackle These Challenges, We Present AlphaZero-Edu, a Lightweight, Education-Focused Implementation Built Upon the Mathematical Framework of AlphaZero. It Boasts a Modular Architecture That Disentangles Key Components, Enabling Transparent Visualization of the Algorithmic Processes. Additionally, It Is Optimized for Resource-Efficient Training on a Single NVIDIA RTX 3090 GPU and Features Highly Parallelized Self-Play Data Generation, Achieving a 3.2-Fold Speedup With 8 Processes. In Gomoku Matches, the Framework Has Demonstrated Exceptional Performance, Achieving a Consistently High Win Rate Against Human Opponents. AlphaZero-Edu Has Been Open-Sourced at This https URL, Providing an Accessible and Practical Benchmark for Both Academic Research and Industrial Applications. 

---
# Towards Optimal Circuit Generation: Multi-Agent Collaboration Meets Collective Intelligence 

**Title (ZH)**: 向着最优电路生成：多智能体协作与集体智能的融合 

**Authors**: Haiyan Qin, Jiahao Feng, Xiaotong Feng, Wei W. Xing, Wang Kang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14625)  

**Abstract**: Large language models (LLMs) have transformed code generation, yet their application in hardware design produces gate counts 38\%--1075\% higher than human designs. We present CircuitMind, a multi-agent framework that achieves human-competitive efficiency through three key innovations: syntax locking (constraining generation to basic logic gates), retrieval-augmented generation (enabling knowledge-driven design), and dual-reward optimization (balancing correctness with efficiency). To evaluate our approach, we introduce TC-Bench, the first gate-level benchmark harnessing collective intelligence from the TuringComplete ecosystem -- a competitive circuit design platform with hundreds of thousands of players. Experiments show CircuitMind enables 55.6\% of model implementations to match or exceed top-tier human experts in composite efficiency metrics. Most remarkably, our framework elevates the 14B Phi-4 model to outperform both GPT-4o mini and Gemini 2.0 Flash, achieving efficiency comparable to the top 25\% of human experts without requiring specialized training. These innovations establish a new paradigm for hardware optimization where collaborative AI systems leverage collective human expertise to achieve optimal circuit designs. Our model, data, and code are open-source at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）已在代码生成领域产生革命性影响，但在硬件设计中的应用却导致门电路数量比人工设计高出38%至1075%。我们提出了CircuitMind多智能体框架，通过三项关键创新实现与人类竞争的效率：语法锁定（约束生成到基本逻辑门）、检索增强生成（实现知识驱动设计）和双重奖励优化（平衡正确性和效率）。为了评估我们的方法，我们引入了TC-Bench基准测试，这是首个利用图灵完备生态系统集体智慧的门级基准测试——一个拥有数十万参赛者的竞争性电路设计平台。实验结果显示，CircuitMind使55.6%的模型实现能够匹配或超越顶级人工专家的综合效率指标。尤为令人瞩目的是，我们的框架将14B Phi-4模型提升到在效率上优于GPT-4o mini和Gemini 2.0 Flash，无需专门训练即可达到顶级人工专家前25%的效率水平。这些创新确立了一个新的硬件优化范式，即协作式AI系统利用集体人类专业知识来实现最佳电路设计。我们的模型、数据和代码已开源。 

---
# On Dimension-Free Transformer: An Application of STP to AI 

**Title (ZH)**: 维度无关的变压器：STP在AI中的应用 

**Authors**: Daizhan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.14514)  

**Abstract**: The matrix expressions for every parts of a transformer are firstly described. Based on semi-tensor product (STP) of matrices the hypervectors are reconsidered and the linear transformation over hypervectors is constructed by using projection. Its properties and calculating formulas are obtained. Using projection-based transformation of hypervector (PBTH), the framework of dimension-free transformer (DFT) is proposed by verifying each linear transformation in a transformer and replacing it by a proper PBTH, which allows the inputs and outputs being of arbitrary dimensions. Using balanced information about all entries, DFT must be more efficient in dealing with signals. 

**Abstract (ZH)**: 变压器中每一部分的矩阵表达式首先被描述。基于矩阵的半张量积（STP），重新考虑了超向量，并通过投影构建了超向量的线性变换，获得了其性质和计算公式。利用基于投影的超向量变换（PBTH），通过验证变压器中的每一线性变换并用适当的PBTH替换，提出了维度无关变压器（DFT）的框架，使得输入和输出可以具有任意维度。利用所有条目平衡的信息，DFT在处理信号时必然更高效。 

---
# LBM-GNN: Graph Neural Network Enhanced Lattice Boltzmann Method 

**Title (ZH)**: LBM-GNN： lattice Boltzmann 方法增强的图神经网络 

**Authors**: Yue Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.14494)  

**Abstract**: In this paper, we present LBM-GNN, a novel approach that enhances the traditional Lattice Boltzmann Method (LBM) with Graph Neural Networks (GNNs). We apply this method to fluid dynamics simulations, demonstrating improved stability and accuracy compared to standard LBM implementations. The method is validated using benchmark problems such as the Taylor-Green vortex, focusing on accuracy, conservation properties, and performance across different Reynolds numbers and grid resolutions. Our results indicate that GNN-enhanced LBM can maintain better conservation properties while improving numerical stability at higher Reynolds numbers. 

**Abstract (ZH)**: LBM-GNN：通过图神经网络增强的晶格玻尔兹曼方法及其在流体动力学模拟中的应用 

---
# FinSage: A Multi-aspect RAG System for Financial Filings Question Answering 

**Title (ZH)**: FinSage: 一种多方面语料库检索系统用于财务报表问答 

**Authors**: Xinyu Wang, Jijun Chi, Zhenghan Tai, Tung Sum Thomas Kwok, Muzhi Li, Zhuhong Li, Hailin He, Yuchen Hua, Peng Lu, Suyuchen Wang, Yihong Wu, Jerry Huang, Ling Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.14493)  

**Abstract**: Leveraging large language models in real-world settings often entails a need to utilize domain-specific data and tools in order to follow the complex regulations that need to be followed for acceptable use. Within financial sectors, modern enterprises increasingly rely on Retrieval-Augmented Generation (RAG) systems to address complex compliance requirements in financial document workflows. However, existing solutions struggle to account for the inherent heterogeneity of data (e.g., text, tables, diagrams) and evolving nature of regulatory standards used in financial filings, leading to compromised accuracy in critical information extraction. We propose the FinSage framework as a solution, utilizing a multi-aspect RAG framework tailored for regulatory compliance analysis in multi-modal financial documents. FinSage introduces three innovative components: (1) a multi-modal pre-processing pipeline that unifies diverse data formats and generates chunk-level metadata summaries, (2) a multi-path sparse-dense retrieval system augmented with query expansion (HyDE) and metadata-aware semantic search, and (3) a domain-specialized re-ranking module fine-tuned via Direct Preference Optimization (DPO) to prioritize compliance-critical content. Extensive experiments demonstrate that FinSage achieves an impressive recall of 92.51% on 75 expert-curated questions derived from surpasses the best baseline method on the FinanceBench question answering datasets by 24.06% in accuracy. Moreover, FinSage has been successfully deployed as financial question-answering agent in online meetings, where it has already served more than 1,200 people. 

**Abstract (ZH)**: 利用大规模语言模型在实际应用场景中往往需要使用领域特定的数据和工具以遵循复杂的合规要求。在金融领域，现代企业越来越多地依赖检索增强生成（RAG）系统来解决金融文档流程中的复杂合规要求。然而，现有的解决方案难以应对数据的内在异质性（例如，文本、表格、图表）和监管标准的不断发展变化，这导致关键信息提取的准确性受到影响。我们提出FinSage框架作为解决方案，利用一个针对多模态金融文件中合规分析的多方面RAG框架。FinSage引入了三个创新组件：（1）一个多模态预处理流水线，统一多种数据格式并生成片段级元数据摘要；（2）一个增强查询扩展（HyDE）和元数据意识语义搜索的多路径稀疏密集检索系统；（3）一个通过直接偏好优化（DPO）微调的领域专用重排模块，优先处理合规关键内容。广泛实验表明，FinSage在75个专家策划的问题上实现了92.51%的召回率，在FinanceBench问答数据集上比最佳基线方法的准确性高出24.06%。此外，FinSage已被成功部署为在线会议中的金融问答代理，已经为超过1,200人提供了服务。 

---
# Planet as a Brain: Towards Internet of AgentSites based on AIOS Server 

**Title (ZH)**: 行星作为大脑：基于AIOS服务器的代理站点互联网owards Internet of AgentSites based on AIOS Server 

**Authors**: Xiang Zhang, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14411)  

**Abstract**: The internet is undergoing a historical transformation from the "Internet of Websites" to the "Internet of AgentSites." While traditional Websites served as the foundation for information hosting and dissemination, a new frontier is emerging where AgentSites serve as the hubs of the internet, where each AgentSite hosts one or more AI agents that receive tasks, address them, and deliver actionable solutions, marking a significant shift in the digital landscape and representing the next generation of online ecosystems. Under this vision, AIOS, the AI Agent Operating System, serves as the server for the development, deployment and execution of AI agents, which is a fundamental infrastructure for the Internet of Agentsites.
In this paper, we introduce AIOS Server, a runtime framework to host agents and enable global-scale collaboration among decentralized agents. AIOS Server provides a communication protocol leveraging the Model Context Protocol (MCP) and JSON-RPC to enable agent-agent or human-agent interactions. Each AIOS node operates as a server to host and execute agents, while supporting peer-to-peer coordination without reliance on centralized orchestration. Based on AIOS Server, we further present the world's first practically deployed Internet of Agentsites (AIOS-IoA), including AgentHub for agent registration and discovery and AgentChat for interactive communication, at this https URL. The agent discovery mechanism based on Distributed Hash Tables (DHT) and a Gossip protocol serves as the search engine for the internet of agentsites. This work provides a practical foundation for building the Internet of Agentsites-a new paradigm where autonomous agents become first-class citizens of the web. The implementation is available at this https URL and will be integrated into the AIOS main branch at this https URL. 

**Abstract (ZH)**: 互联网正在经历从“网站网”到“智能站点网”的历史转变。传统网站作为信息托管和传播的基础，一个新的前沿领域正逐渐兴起，智能站点成为互联网的枢纽，每个智能站点托管一个或多个AI代理，接收任务、解决任务并提供可执行的解决方案，标志着数字景观的重大转变，并代表着下一代在线生态系统。在此愿景下，AIOS（AI代理操作系统）作为开发、部署和执行AI代理的服务器，是智能站点网的基础基础设施。

在本文中，我们介绍了AIOS Server，这是一个运行时框架，用于托管代理并促进去中心化代理的全球规模协作。AIOS Server利用模型上下文协议（MCP）和JSON-RPC提供通信协议，以支持代理与代理或人与代理之间的交互。每个AIOS节点作为一个服务器来托管和执行代理，并支持去中心化的协调机制而无需依赖集中式编排。基于AIOS Server，我们进一步介绍了第一个实际部署的智能站点网（AIOS-IoA），包括AgentHub（智能站点注册和发现）和AgentChat（互动通信），详情请参见此https://链接。基于分布式哈希表（DHT）和Gossip协议的智能站点发现机制作为智能站点网的搜索引擎。本项工作为构建智能站点网提供了实际的基础，使自治代理成为网络中的头等公民。相关实现可在此https://链接获取，并将集成到AIOS主分支中。 

---
# Data Augmentation Using Neural Acoustic Fields With Retrieval-Augmented Pre-training 

**Title (ZH)**: 基于检索增强预训练的神经声学场数据增强方法 

**Authors**: Christopher Ick, Gordon Wichern, Yoshiki Masuyama, François G. Germain, Jonathan Le Roux  

**Link**: [PDF](https://arxiv.org/pdf/2504.14409)  

**Abstract**: This report details MERL's system for room impulse response (RIR) estimation submitted to the Generative Data Augmentation Workshop at ICASSP 2025 for Augmenting RIR Data (Task 1) and Improving Speaker Distance Estimation (Task 2). We first pre-train a neural acoustic field conditioned by room geometry on an external large-scale dataset in which pairs of RIRs and the geometries are provided. The neural acoustic field is then adapted to each target room by using the enrollment data, where we leverage either the provided room geometries or geometries retrieved from the external dataset, depending on availability. Lastly, we predict the RIRs for each pair of source and receiver locations specified by Task 1, and use these RIRs to train the speaker distance estimation model in Task 2. 

**Abstract (ZH)**: MERL的房间冲激响应估计系统：用于ICASSP 2025生成数据增强工作坊的房间冲激响应数据增强（任务1）和演讲者距离估计改进（任务2）。 

---
# ScholarMate: A Mixed-Initiative Tool for Qualitative Knowledge Work and Information Sensemaking 

**Title (ZH)**: ScholarMate: 一种混合主动型工具，用于定性知识工作和信息意义构建 

**Authors**: Runlong Ye, Patrick Yung Kang Lee, Matthew Varona, Oliver Huang, Carolina Nobre  

**Link**: [PDF](https://arxiv.org/pdf/2504.14406)  

**Abstract**: Synthesizing knowledge from large document collections is a critical yet increasingly complex aspect of qualitative research and knowledge work. While AI offers automation potential, effectively integrating it into human-centric sensemaking workflows remains challenging. We present ScholarMate, an interactive system designed to augment qualitative analysis by unifying AI assistance with human oversight. ScholarMate enables researchers to dynamically arrange and interact with text snippets on a non-linear canvas, leveraging AI for theme suggestions, multi-level summarization, and contextual naming, while ensuring transparency through traceability to source documents. Initial pilot studies indicated that users value this mixed-initiative approach, finding the balance between AI suggestions and direct manipulation crucial for maintaining interpretability and trust. We further demonstrate the system's capability through a case study analyzing 24 papers. By balancing automation with human control, ScholarMate enhances efficiency and supports interpretability, offering a valuable approach for productive human-AI collaboration in demanding sensemaking tasks common in knowledge work. 

**Abstract (ZH)**: 从大规模文档集合中合成知识是定性研究和知识工作中一个关键但日益复杂的方面。尽管人工智能提供了自动化潜力，将其有效集成到以人文为中心的意义建构工作流程中仍然具有挑战性。我们介绍了ScholarMate，这是一个交互系统，旨在通过统一人工智能辅助和人类监督来增强定性分析。ScholarMate使研究者能够动态地在非线性画布上排列和交互文本片段，利用人工智能进行主题建议、多级总结和上下文命名，同时通过溯源保持透明度。初步的试点研究显示，用户欣赏这种混合主动的方法，认为在人工智能建议和直接操作之间找到平衡对保持解释性和信任至关重要。我们通过分析24篇论文的案例研究进一步展示了该系统的功能。通过平衡自动化和人类控制，ScholarMate提高了效率并支持了解释性，为知识工作中常见的意义建构任务提供了有价值的人机协作方法。 

---
# Learning Enhanced Structural Representations with Block-Based Uncertainties for Ocean Floor Mapping 

**Title (ZH)**: 基于块基础不确定性学习增强的结构表示用于海底测绘 

**Authors**: Jose Marie Antonio Minoza  

**Link**: [PDF](https://arxiv.org/pdf/2504.14372)  

**Abstract**: Accurate ocean modeling and coastal hazard prediction depend on high-resolution bathymetric data; yet, current worldwide datasets are too coarse for exact numerical simulations. While recent deep learning advances have improved earth observation data resolution, existing methods struggle with the unique challenges of producing detailed ocean floor maps, especially in maintaining physical structure consistency and quantifying uncertainties. This work presents a novel uncertainty-aware mechanism using spatial blocks to efficiently capture local bathymetric complexity based on block-based conformal prediction. Using the Vector Quantized Variational Autoencoder (VQ-VAE) architecture, the integration of this uncertainty quantification framework yields spatially adaptive confidence estimates while preserving topographical features via discrete latent representations. With smaller uncertainty widths in well-characterized areas and appropriately larger bounds in areas of complex seafloor structures, the block-based design adapts uncertainty estimates to local bathymetric complexity. Compared to conventional techniques, experimental results over several ocean regions show notable increases in both reconstruction quality and uncertainty estimation reliability. This framework increases the reliability of bathymetric reconstructions by preserving structural integrity while offering spatially adaptive uncertainty estimates, so opening the path for more solid climate modeling and coastal hazard assessment. 

**Abstract (ZH)**: 高精度海底地形建模与沿海灾害预测依赖于高分辨率 bathymetric 数据；然而，当前全球数据集的分辨率尚不足以进行精确的数值模拟。尽管近年来深度学习技术提高了地球观测数据的分辨率，但现有方法在生成详细的海底地形图时面临着独特挑战，尤其是在保持物理结构一致性和量化不确定性方面。本文提出了一种新的不确定性感知机制，利用基于区块的 conforme 预测，结合区块结构有效地捕获局部海底地形的复杂性。通过采用向量量化变分自编码器（VQ-VAE）架构，该不确定性量化框架提供了空间自适应的置信度估计，同时通过离散的潜在表示保留地形特征。在特征描述良好的区域，通过减小不确定性范围；而在复杂海底结构的区域，则适当扩大不确定性范围。区块化设计使不确定性估计能够适应局部海底地形复杂性。与传统技术相比，多个海洋区域的实验结果表明，该框架在重建质量和不确定性估计可靠性方面均有显著提升。该框架通过保留结构性完整性并提供空间自适应的不确定性估计，提高了海底地形重建的可靠性，为更加坚实的气候建模和沿海灾害评估铺平了道路。 

---
# Expanding the Generative AI Design Space through Structured Prompting and Multimodal Interfaces 

**Title (ZH)**: 通过结构化提示和多模态界面扩展生成式AI设计空间 

**Authors**: Nimisha Karnatak, Adrien Baranes, Rob Marchant, Huinan Zeng, Tríona Butler, Kristen Olson  

**Link**: [PDF](https://arxiv.org/pdf/2504.14320)  

**Abstract**: Text-based prompting remains the dominant interaction paradigm in generative AI, yet it often results in a high-friction experience for novice users, such as small business owners (SBOs), attempting to articulate creative or domain-specific goals for advertising. To investigate this challenge, we conducted a study with six SBOs in the United Kingdom, focusing on their advertising practices and perceptions and usage of AI tools in this context. Our findings surfaced two persistent breakdowns in current generative AI systems: first, the cognitive burden of prompt engineering, as users struggled to translate abstract creative goals into effective textual inputs; and second, the frequent generation of generic outputs that failed to align with users' articulated brand vision. To address these issues, we developed ACAI (AI Co-Creation for Advertising and Inspiration), a multimodal, GenAI-powered advertisement creation tool designed to support novice designers by reimagining the prompt interface. ACAI features a structured, panel-based interface composed of three modules: the Branding Panel, the Audience & Goals Panel, and the Inspiration Board Panel to provide SBOs with outputs that align with their creative vision by reducing prompt ambiguity. This work contributes to HCI research on generative systems by showing how structured interfaces can foreground user-defined context to improve both alignment and promptability in novice workflows. 

**Abstract (ZH)**: 基于文本的提示仍然是生成式AI的主要交互范式，但对于试图为广告 articulated 创造性或领域特定目标的小型企业主（SBOs）来说，往往会带来高摩擦的体验。为了研究这一挑战，我们在英国对六名SBOs进行了研究，关注他们在广告活动中的实践、观念及其对AI工具的使用。我们的研究发现当前生成式AI系统中存在的两大持续性问题：首先，提示工程的认知负担，用户在努力将抽象的创意目标转化为有效的文本输入；其次，生成的输出经常与用户阐明的品牌愿景不一致。为了解决这些问题，我们开发了ACAI（用于广告和灵感的人工智能联合创造），这是一种多模态的、由生成式AI驱动的广告创作工具，旨在通过重塑提示界面来支持初学者设计师。ACAI具有一结构化的面板式界面，由品牌板块、目标与受众板块以及灵感板板块三部分组成，通过减少提示的模糊性，为SBOs提供与其创意愿景相一致的输出。这项工作为生成系统的人机交互研究做出了贡献，展示了结构化界面如何将用户定义的上下文置于首位，以改善初学者工作流中的匹配度和提示能力。 

---
# Learning to Score 

**Title (ZH)**: 学习打分 

**Authors**: Yogev Kriger, Shai Fine  

**Link**: [PDF](https://arxiv.org/pdf/2504.14302)  

**Abstract**: Common machine learning settings range from supervised tasks, where accurately labeled data is accessible, through semi-supervised and weakly-supervised tasks, where target labels are scant or noisy, to unsupervised tasks where labels are unobtainable. In this paper we study a scenario where the target labels are not available but additional related information is at hand. This information, referred to as Side Information, is either correlated with the unknown labels or imposes constraints on the feature space. We formulate the problem as an ensemble of three semantic components: representation learning, side information and metric learning. The proposed scoring model is advantageous for multiple use-cases. For example, in the healthcare domain it can be used to create a severity score for diseases where the symptoms are known but the criteria for the disease progression are not well defined. We demonstrate the utility of the suggested scoring system on well-known benchmark data-sets and bio-medical patient records. 

**Abstract (ZH)**: 常见的机器学习设置包括监督任务、半监督任务、弱监督任务和无监督任务。在监督任务中，有准确标注的数据；半监督和弱监督任务中，目标标签稀少或噪声较大；无监督任务中，标签不可获得。在本文中，我们研究目标标签不可获得但有额外相关信息的情况。这种信息称为辅助信息，它要么与未知标签相关，要么约束特征空间。我们将问题形式化为三个语义组件的组合：表示学习、辅助信息和度量学习。所提出的成绩模型适用于多种应用场景。例如，在医疗领域，它可以用来为已知症状但疾病进展标准不明确的疾病创建严重程度评分。我们通过众所周知的标准数据集和生物医学患者记录展示了所建议评分系统的有效性。 

---
# Learning and Generating Diverse Residential Load Patterns Using GAN with Weakly-Supervised Training and Weight Selection 

**Title (ZH)**: 使用弱监督训练和权重选择的GAN学习和生成多样化的住宅负荷模式 

**Authors**: Xinyu Liang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14300)  

**Abstract**: The scarcity of high-quality residential load data can pose obstacles for decarbonizing the residential sector as well as effective grid planning and operation. The above challenges have motivated research into generating synthetic load data, but existing methods faced limitations in terms of scalability, diversity, and similarity. This paper proposes a Generative Adversarial Network-based Synthetic Residential Load Pattern (RLP-GAN) generation model, a novel weakly-supervised GAN framework, leveraging an over-complete autoencoder to capture dependencies within complex and diverse load patterns and learn household-level data distribution at scale. We incorporate a model weight selection method to address the mode collapse problem and generate load patterns with high diversity. We develop a holistic evaluation method to validate the effectiveness of RLP-GAN using real-world data of 417 households. The results demonstrate that RLP-GAN outperforms state-of-the-art models in capturing temporal dependencies and generating load patterns with higher similarity to real data. Furthermore, we have publicly released the RLP-GAN generated synthetic dataset, which comprises one million synthetic residential load pattern profiles. 

**Abstract (ZH)**: 基于生成对抗网络的合成住宅负荷模式生成模型（RLP-GAN）：一种新颖的弱监督GAN框架 

---
# Decomposition-based multi-scale transformer framework for time series anomaly detection 

**Title (ZH)**: 基于分解的多尺度变换器框架的时间序列异常检测 

**Authors**: Wenxin Zhang, Cuicui Luo  

**Link**: [PDF](https://arxiv.org/pdf/2504.14206)  

**Abstract**: Time series anomaly detection is crucial for maintaining stable systems. Existing methods face two main challenges. First, it is difficult to directly model the dependencies of diverse and complex patterns within the sequences. Second, many methods that optimize parameters using mean squared error struggle with noise in the time series, leading to performance deterioration. To address these challenges, we propose a transformer-based framework built on decomposition (TransDe) for multivariate time series anomaly detection. The key idea is to combine the strengths of time series decomposition and transformers to effectively learn the complex patterns in normal time series data. A multi-scale patch-based transformer architecture is proposed to exploit the representative dependencies of each decomposed component of the time series. Furthermore, a contrastive learn paradigm based on patch operation is proposed, which leverages KL divergence to align the positive pairs, namely the pure representations of normal patterns between different patch-level views. A novel asynchronous loss function with a stop-gradient strategy is further introduced to enhance the performance of TransDe effectively. It can avoid time-consuming and labor-intensive computation costs in the optimization process. Extensive experiments on five public datasets are conducted and TransDe shows superiority compared with twelve baselines in terms of F1 score. Our code is available at this https URL. 

**Abstract (ZH)**: 基于分解的Transformer框架在多变元时间序列异常检测中的应用 

---
# Dual-channel Heterophilic Message Passing for Graph Fraud Detection 

**Title (ZH)**: 双重通道异类消息传递用于图欺诈检测 

**Authors**: Wenxin Zhang, Jingxing Zhong, Guangzhen Yao, Renda Han, Xiaojian Lin, Zeyu Zhang, Cuicui Luo  

**Link**: [PDF](https://arxiv.org/pdf/2504.14205)  

**Abstract**: Fraudulent activities have significantly increased across various domains, such as e-commerce, online review platforms, and social networks, making fraud detection a critical task. Spatial Graph Neural Networks (GNNs) have been successfully applied to fraud detection tasks due to their strong inductive learning capabilities. However, existing spatial GNN-based methods often enhance the graph structure by excluding heterophilic neighbors during message passing to align with the homophilic bias of GNNs. Unfortunately, this approach can disrupt the original graph topology and increase uncertainty in predictions. To address these limitations, this paper proposes a novel framework, Dual-channel Heterophilic Message Passing (DHMP), for fraud detection. DHMP leverages a heterophily separation module to divide the graph into homophilic and heterophilic subgraphs, mitigating the low-pass inductive bias of traditional GNNs. It then applies shared weights to capture signals at different frequencies independently and incorporates a customized sampling strategy for training. This allows nodes to adaptively balance the contributions of various signals based on their labels. Extensive experiments on three real-world datasets demonstrate that DHMP outperforms existing methods, highlighting the importance of separating signals with different frequencies for improved fraud detection. The code is available at this https URL. 

**Abstract (ZH)**: 欺诈活动在电子商务、在线评价平台和社会网络等领域显著增加，使欺诈检测成为一项关键任务。空间图神经网络（GNNs）由于其强大的归纳学习能力，在欺诈检测任务中取得了成功应用。然而，现有的基于空间GNN的方法常常在消息传递过程中通过排除异ophilic邻居来增强图结构，以符合GNN的同ophilic偏见。不幸的是，这种方法可能会破坏原始图拓扑结构，增加预测的不确定性。为了解决这些局限性，本文提出了一种新的框架，双通道异ophilic消息传递（DHMP）方法，用于欺诈检测。DHMP利用一个异ophilic分离模块将图划分为同ophilic和异ophilic子图，以减轻传统GNN的低通归纳偏差。然后，它使用共享权重独立捕获不同频率的信号，并结合一种定制的采样策略进行训练。这使节点能够根据其标签适应性地平衡各种信号的贡献。在三个真实世界数据集上的广泛实验表明，DHMP优于现有方法，突出了分离不同频率信号以提高欺诈检测效果的重要性。代码可在此处获取。 

---
# DConAD: A Differencing-based Contrastive Representation Learning Framework for Time Series Anomaly Detection 

**Title (ZH)**: DConAD：一种基于差异对比的时序异常检测表示学习框架 

**Authors**: Wenxin Zhang, Xiaojian Lin, Wenjun Yu, Guangzhen Yao, jingxiang Zhong, Yu Li, Renda Han, Songcheng Xu, Hao Shi, Cuicui Luo  

**Link**: [PDF](https://arxiv.org/pdf/2504.14204)  

**Abstract**: Time series anomaly detection holds notable importance for risk identification and fault detection across diverse application domains. Unsupervised learning methods have become popular because they have no requirement for labels. However, due to the challenges posed by the multiplicity of abnormal patterns, the sparsity of anomalies, and the growth of data scale and complexity, these methods often fail to capture robust and representative dependencies within the time series for identifying anomalies. To enhance the ability of models to capture normal patterns of time series and avoid the retrogression of modeling ability triggered by the dependencies on high-quality prior knowledge, we propose a differencing-based contrastive representation learning framework for time series anomaly detection (DConAD). Specifically, DConAD generates differential data to provide additional information about time series and utilizes transformer-based architecture to capture spatiotemporal dependencies, which enhances the robustness of unbiased representation learning ability. Furthermore, DConAD implements a novel KL divergence-based contrastive learning paradigm that only uses positive samples to avoid deviation from reconstruction and deploys the stop-gradient strategy to compel convergence. Extensive experiments on five public datasets show the superiority and effectiveness of DConAD compared with nine baselines. The code is available at this https URL. 

**Abstract (ZH)**: 基于差分的对比表示学习框架：时间序列异常检测（DConAD） 

---
# Personalized News Recommendation with Multi-granularity Candidate-aware User Modeling 

**Title (ZH)**: 多粒度候选意识用户建模的个性化新闻推荐 

**Authors**: Qiang Li, Xinze Lin, Shenghao Lv, Faliang Huang, Xiangju Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.14130)  

**Abstract**: Matching candidate news with user interests is crucial for personalized news recommendations. Most existing methods can represent a user's reading interests through a single profile based on clicked news, which may not fully capture the diversity of user interests. Although some approaches incorporate candidate news or topic information, they remain insufficient because they neglect the multi-granularity relatedness between candidate news and user interests. To address this, this study proposed a multi-granularity candidate-aware user modeling framework that integrated user interest features across various levels of granularity. It consisted of two main components: candidate news encoding and user modeling. A news textual information extractor and a knowledge-enhanced entity information extractor can capture candidate news features, and word-level, entity-level, and news-level candidate-aware mechanisms can provide a comprehensive representation of user interests. Extensive experiments on a real-world dataset demonstrated that the proposed model could significantly outperform baseline models. 

**Abstract (ZH)**: 基于多粒度候选新闻的用户建模框架：捕获用户兴趣的多样性 

---
# Exploring Language Patterns of Prompts in Text-to-Image Generation and Their Impact on Visual Diversity 

**Title (ZH)**: 探索文本生成图像中提示的语言模式及其对视觉多样性的影响 

**Authors**: Maria-Teresa De Rosa Palmini, Eva Cetinic  

**Link**: [PDF](https://arxiv.org/pdf/2504.14125)  

**Abstract**: Following the initial excitement, Text-to-Image (TTI) models are now being examined more critically. While much of the discourse has focused on biases and stereotypes embedded in large-scale training datasets, the sociotechnical dynamics of user interactions with these models remain underexplored. This study examines the linguistic and semantic choices users make when crafting prompts and how these choices influence the diversity of generated outputs. Analyzing over six million prompts from the Civiverse dataset on the CivitAI platform across seven months, we categorize users into three groups based on their levels of linguistic experimentation: consistent repeaters, occasional repeaters, and non-repeaters. Our findings reveal that as user participation grows over time, prompt language becomes increasingly homogenized through the adoption of popular community tags and descriptors, with repeated prompts comprising 40-50% of submissions. At the same time, semantic similarity and topic preferences remain relatively stable, emphasizing common subjects and surface aesthetics. Using Vendi scores to quantify visual diversity, we demonstrate a clear correlation between lexical similarity in prompts and the visual similarity of generated images, showing that linguistic repetition reinforces less diverse representations. These findings highlight the significant role of user-driven factors in shaping AI-generated imagery, beyond inherent model biases, and underscore the need for tools and practices that encourage greater linguistic and thematic experimentation within TTI systems to foster more inclusive and diverse AI-generated content. 

**Abstract (ZH)**: Text-to-Image模型：用户互动中的语言与语义选择及其对生成输出多样性的影响 

---
# Longitudinal Study on Social and Emotional Use of AI Conversational Agent 

**Title (ZH)**: longitudinal研究：社会情感使用中的AI对话代理 

**Authors**: Mohit Chandra, Javier Hernandez, Gonzalo Ramos, Mahsa Ershadi, Ananya Bhattacharjee, Judith Amores, Ebele Okoli, Ann Paradiso, Shahed Warreth, Jina Suh  

**Link**: [PDF](https://arxiv.org/pdf/2504.14112)  

**Abstract**: Development in digital technologies has continuously reshaped how individuals seek and receive social and emotional support. While online platforms and communities have long served this need, the increased integration of general-purpose conversational AI into daily lives has introduced new dynamics in how support is provided and experienced. Existing research has highlighted both benefits (e.g., wider access to well-being resources) and potential risks (e.g., over-reliance) of using AI for support seeking. In this five-week, exploratory study, we recruited 149 participants divided into two usage groups: a baseline usage group (BU, n=60) that used the internet and AI as usual, and an active usage group (AU, n=89) encouraged to use one of four commercially available AI tools (Microsoft Copilot, Google Gemini, PI AI, ChatGPT) for social and emotional interactions. Our analysis revealed significant increases in perceived attachment towards AI (32.99 percentage points), perceived AI empathy (25.8 p.p.), and motivation to use AI for entertainment (22.90 p.p.) among the AU group. We also observed that individual differences (e.g., gender identity, prior AI usage) influenced perceptions of AI empathy and attachment. Lastly, the AU group expressed higher comfort in seeking personal help, managing stress, obtaining social support, and talking about health with AI, indicating potential for broader emotional support while highlighting the need for safeguards against problematic usage. Overall, our exploratory findings underscore the importance of developing consumer-facing AI tools that support emotional well-being responsibly, while empowering users to understand the limitations of these tools. 

**Abstract (ZH)**: 数字技术的发展不断重塑个体寻求和接收社会和情感支持的方式。在线平台和社区长期服务于这一需求，而日常生活中的通用对话型AI集成增加，引入了支持提供和体验的新动态。现有研究强调了使用AI寻求支持的益处（如更广泛的福祉资源获取）和潜在风险（如过度依赖）。在为期五周的探索性研究中，我们招募了149名参与者，分为两组使用群体：常规使用组（BU，n=60）继续正常使用互联网和AI，活跃使用组（AU，n=89）被鼓励使用四种商用可用的AI工具（Microsoft Copilot、Google Gemini、PI AI、ChatGPT）进行社交和情感互动。我们的分析显示，活跃使用组（AU组）对AI的情感依附感知提高了32.99个百分点，感知到的AI同理心提高25.8个百分点，以及使用AI进行娱乐的动力提高22.90个百分点。我们还发现，个体差异（如性别认同、先前的AI使用情况）影响了对AI同理心和情感依附的感知。最后，活跃使用组表示在寻求个人帮助、管理压力、获取社交支持和与AI讨论健康方面更为自在，这表明AI有可能提供更广泛的情感支持，但也突显了防范有害使用的需求。总体而言，我们的探索性发现强调了负责任地开发面向消费者的情感福祉支持AI工具的重要性，同时赋予用户了解这些工具局限性的能力。 

---
# Amplify Initiative: Building A Localized Data Platform for Globalized AI 

**Title (ZH)**: Amplify倡议：构建面向全球AI的本地化数据平台 

**Authors**: Qazi Mamunur Rashid, Erin van Liemt, Tiffany Shih, Amber Ebinama, Karla Barrios Ramos, Madhurima Maji, Aishwarya Verma, Charu Kalia, Jamila Smith-Loud, Joyce Nakatumba-Nabende, Rehema Baguma, Andrew Katumba, Chodrine Mutebi, Jagen Marvin, Eric Peter Wairagala, Mugizi Bruce, Peter Oketta, Lawrence Nderu, Obichi Obiajunwa, Abigail Oppong, Michael Zimba, Data Authors  

**Link**: [PDF](https://arxiv.org/pdf/2504.14105)  

**Abstract**: Current AI models often fail to account for local context and language, given the predominance of English and Western internet content in their training data. This hinders the global relevance, usefulness, and safety of these models as they gain more users around the globe. Amplify Initiative, a data platform and methodology, leverages expert communities to collect diverse, high-quality data to address the limitations of these models. The platform is designed to enable co-creation of datasets, provide access to high-quality multilingual datasets, and offer recognition to data authors. This paper presents the approach to co-creating datasets with domain experts (e.g., health workers, teachers) through a pilot conducted in Sub-Saharan Africa (Ghana, Kenya, Malawi, Nigeria, and Uganda). In partnership with local researchers situated in these countries, the pilot demonstrated an end-to-end approach to co-creating data with 155 experts in sensitive domains (e.g., physicians, bankers, anthropologists, human and civil rights advocates). This approach, implemented with an Android app, resulted in an annotated dataset of 8,091 adversarial queries in seven languages (e.g., Luganda, Swahili, Chichewa), capturing nuanced and contextual information related to key themes such as misinformation and public interest topics. This dataset in turn can be used to evaluate models for their safety and cultural relevance within the context of these languages. 

**Abstract (ZH)**: 当前的AI模型往往未能考虑到地方语境和语言，因为训练数据中占主导地位的是英语和西方互联网内容。这阻碍了这些模型在全球范围内的相关性、实用性和安全性，特别是在它们获得越来越多的全球用户之后。Amplify Initiative，一个数据平台和方法，通过利用专家社区收集多样性和高质量的数据来解决这些模型的局限性。该平台旨在促进与领域专家的合作数据集创建，提供高质量多语言数据集的访问权限，并为数据作者提供认可。本文介绍了一种通过在撒哈拉以南非洲地区（加纳、肯尼亚、马拉维、尼日利亚和乌干达）进行试点研究与领域专家（如医护人员、教师等）合作创建数据集的方法。与当地研究人员合作，试点展示了从155位专家（如医生、银行家、人类学家、人权倡导者）敏感领域中端到端合作创建数据集的方法。通过Android应用实施的方法，生成了一个包含8,091个 adversarial 查询的标注数据集，涉及七种语言（如卢干达语、斯瓦希里语、奇切瓦语），捕捉到与误导信息和公众兴趣主题相关的细微和情境信息。这个数据集可用于评估模型在这些语言背景下的安全性和文化相关性。 

---
# 6G WavesFM: A Foundation Model for Sensing, Communication, and Localization 

**Title (ZH)**: 6G WavesFM：感测、通信与定位的础模型 

**Authors**: Ahmed Aboulfotouh, Elsayed Mohammed, Hatem Abou-Zeid  

**Link**: [PDF](https://arxiv.org/pdf/2504.14100)  

**Abstract**: This paper introduces WavesFM, a novel Wireless Foundation Model (WFM) framework, capable of supporting a wide array of communication, sensing, and localization tasks. Our proposed architecture combines a shared Vision Transformer (ViT) backbone with task-specific multi-layer perceptron (MLP) heads and incorporates Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning. This design promotes full parameter sharing across tasks, significantly reducing the computational and memory footprint without sacrificing performance. The model processes both image-like wireless modalities, such as spectrograms and channel state information (CSI), and in-phase and quadrature (IQ) signals arranged as orthogonal frequency-division multiplexing (OFDM) resource grids. We demonstrate the strong generalization capabilities of WavesFM through extensive experiments on four downstream tasks: Fifth Generation New Radio (5G NR) positioning; multiple-input multiple-output OFDM (MIMO-OFDM) channel estimation; human activity sensing; and radio-frequency (RF) signal classification. Compared to supervised baselines trained individually, our approach achieves superior performance while sharing 80% of its parameters across tasks. Furthermore, we show that pretraining on domain-relevant data not only boosts performance but also accelerates convergence, reducing training time by up to 5x. These results demonstrate that our unified WFM can support diverse tasks and deliver significant gains in both performance and efficiency, highlighting the transformative potential of foundation models to drive AI-native paradigms in future sixth-generation (6G) networks. 

**Abstract (ZH)**: 本文介绍了WavesFM，一种新型的无线基础模型（WFM）框架，支持广泛的通信、感测和定位任务。我们提出的设计结合了共享的Vision Transformer（ViT）骨干网络和特定任务的多层感知机（MLP）头部，并采用了低秩适应（LoRA）进行参数高效的微调。这一设计促进了跨任务的全参数共享，显著减少了计算和内存开销，同时不牺牲性能。该模型处理了包括频谱图和信道状态信息（CSI）在内的图像-like无线模态，以及按正交频分复用（OFDM）资源网格排列的同相和正交（IQ）信号。我们通过在四个下游任务上进行广泛的实验展示了WavesFM的强大泛化能力：第五代新型无线电（5G NR）定位；多输入多输出OFDM（MIMO-OFDM）信道估计；人体活动感测；射频（RF）信号分类。相较于单独训练的监督基线方法，我们的方法在共享80%参数的情况下实现了更优的性能。此外，我们还展示了在相关领域数据上的预训练不仅能提升性能，还能加速收敛，最多可减少5倍的训练时间。这些结果表明，我们的统一WFM可以支持多种任务，并在性能和效率方面取得显著提升，突显了基础模型在下一代（6G）网络中驱动AI原生范式的潜在变革性潜力。 

---
# Enhancing Math Learning in an LMS Using AI-Driven Question Recommendations 

**Title (ZH)**: 使用AI驱动的问题推荐增强LMS中的数学学习 

**Authors**: Justus Råmunddal  

**Link**: [PDF](https://arxiv.org/pdf/2504.14098)  

**Abstract**: This paper presents an AI-driven approach to enhance math learning in a modern Learning Management System (LMS) by recommending similar math questions. Deep embeddings for math questions are generated using Meta's Llama-3.2-11B-Vision-Instruct model, and three recommendation methods-cosine similarity, Self-Organizing Maps (SOM), and Gaussian Mixture Models (GMM)-are applied to identify similar questions. User interaction data, including session durations, response times, and correctness, are used to evaluate the methods. Our findings suggest that while cosine similarity produces nearly identical question matches, SOM yields higher user satisfaction whereas GMM generally underperforms, indicating that introducing variety to a certain degree may enhance engagement and thereby potential learning outcomes until variety is no longer balanced reasonably, which our data about the implementations of all three methods demonstrate. 

**Abstract (ZH)**: 基于AI驱动的方法在现代学习管理系统中通过推荐相似数学问题来增强数学学习 

---
# Leakage and Interpretability in Concept-Based Models 

**Title (ZH)**: 基于概念的模型中的泄漏与可解释性 

**Authors**: Enrico Parisini, Tapabrata Chakraborti, Chris Harbron, Ben D. MacArthur, Christopher R. S. Banerji  

**Link**: [PDF](https://arxiv.org/pdf/2504.14094)  

**Abstract**: Concept Bottleneck Models aim to improve interpretability by predicting high-level intermediate concepts, representing a promising approach for deployment in high-risk scenarios. However, they are known to suffer from information leakage, whereby models exploit unintended information encoded within the learned concepts. We introduce an information-theoretic framework to rigorously characterise and quantify leakage, and define two complementary measures: the concepts-task leakage (CTL) and interconcept leakage (ICL) scores. We show that these measures are strongly predictive of model behaviour under interventions and outperform existing alternatives in robustness and reliability. Using this framework, we identify the primary causes of leakage and provide strong evidence that Concept Embedding Models exhibit substantial leakage regardless of the hyperparameters choice. Finally, we propose practical guidelines for designing concept-based models to reduce leakage and ensure interpretability. 

**Abstract (ZH)**: 概念瓶颈模型旨在通过预测高层中间概念来提高可解释性，并被视为在高风险场景中部署的一种有前景的方法。然而，它们known to suffer from information leakage，即模型利用了在学习概念中编码的未预期信息。我们引入了一种信息论框架来严格地表征和量化这种泄漏，并定义了两种互补的度量标准：概念任务泄漏 (CTL) 分数和概念间泄漏 (ICL) 分数。我们证明了这些度量标准在干预下的模型行为预测能力强，并且在稳健性和可靠性方面优于现有替代方法。使用该框架，我们确定了泄漏的主要原因，并提供了强有力的证据，表明概念嵌入模型在任何超参数选择下都表现出显著的泄漏。最后，我们提出了实用指南，以减少泄漏并确保基于概念的模型的可解释性。 

---
# Evaluating Human-AI Interaction via Usability, User Experience and Acceptance Measures for MMM-C: A Creative AI System for Music Composition 

**Title (ZH)**: 基于MMM-C的音乐创作创意人工智能系统的人机交互评价：易用性、用户体验和接受度指标的研究 

**Authors**: Renaud Bougueng Tchemeube, Jeff Ens, Cale Plut, Philippe Pasquier, Maryam Safi, Yvan Grabit, Jean-Baptiste Rolland  

**Link**: [PDF](https://arxiv.org/pdf/2504.14071)  

**Abstract**: With the rise of artificial intelligence (AI), there has been increasing interest in human-AI co-creation in a variety of artistic domains including music as AI-driven systems are frequently able to generate human-competitive artifacts. Now, the implications of such systems for musical practice are being investigated. We report on a thorough evaluation of the user adoption of the Multi-Track Music Machine (MMM) as a co-creative AI tool for music composers. To do this, we integrate MMM into Cubase, a popular Digital Audio Workstation (DAW) by Steinberg, by producing a "1-parameter" plugin interface named MMM-Cubase (MMM-C), which enables human-AI co-composition. We contribute a methodological assemblage as a 3-part mixed method study measuring usability, user experience and technology acceptance of the system across two groups of expert-level composers: hobbyists and professionals. Results show positive usability and acceptance scores. Users report experiences of novelty, surprise and ease of use from using the system, and limitations on controllability and predictability of the interface when generating music. Findings indicate no significant difference between the two user groups. 

**Abstract (ZH)**: 随着人工智能（AI）的发展，对多种艺术领域包括音乐中的人机共创的兴趣日益增加，因为AI驱动的系统经常能够生成与人类竞争的艺术品。现在，正在研究此类系统对音乐实践的影响。我们报告了对多轨音乐机器（MMM）作为音乐作曲人协作型AI工具的用户采用进行全面评估的结果。为此，我们通过开发一个名为MMM-Cubase（MMM-C）的一参数插件接口，将MMM集成到Steinberg公司的流行数字音频工作站（DAW）Cubase中，从而实现人机协作创作。我们采用一种综合了三种混合方法的研究方法，从两个专家级作曲家群体：业余和专业作曲家的角度，测量系统的易用性、用户体验和技术接受度。结果显示积极的易用性和接纳性评分。用户报告了使用该系统时的新颖性、惊喜和易用性体验，同时也指出了生成音乐时接口的可控性和可预测性限制。研究发现，两个用户群体之间不存在显著差异。 

---
# A CMOS Probabilistic Computing Chip With In-situ hardware Aware Learning 

**Title (ZH)**: 一种具有就地硬件感知学习功能的CMOS概率计算芯片 

**Authors**: Jinesh Jhonsa, William Whitehead, David McCarthy, Shuvro Chowdhury, Kerem Camsari, Luke Theogarajan  

**Link**: [PDF](https://arxiv.org/pdf/2504.14070)  

**Abstract**: This paper demonstrates a probabilistic bit physics inspired solver with 440 spins configured in a Chimera graph, occupying an area of 0.44 mm^2. Area efficiency is maximized through a current-mode implementation of the neuron update circuit, standard cell design for analog blocks pitch-matched to digital blocks, and a shared power supply for both digital and analog components. Process variation related mismatches introduced by this approach are effectively mitigated using a hardware aware contrastive divergence algorithm during training. We validate the chip's ability to perform probabilistic computing tasks such as modeling logic gates and full adders, as well as optimization tasks such as MaxCut, demonstrating its potential for AI and machine learning applications. 

**Abstract (ZH)**: 基于概率比特物理的Chipira图模拟器设计与实现：440个自旋元件的面积效率最大化 

---
# Sentiment Analysis of Airbnb Reviews: Exploring Their Impact on Acceptance Rates and Pricing Across Multiple U.S. Regions 

**Title (ZH)**: Airbnb评价的情感分析：探索其对多个美国地区接受率和定价的影响 

**Authors**: Ali Safari  

**Link**: [PDF](https://arxiv.org/pdf/2504.14053)  

**Abstract**: This research examines whether Airbnb guests' positive and negative comments influence acceptance rates and rental prices across six U.S. regions: Rhode Island, Broward County, Chicago, Dallas, San Diego, and Boston. Thousands of reviews were collected and analyzed using Natural Language Processing (NLP) to classify sentiments as positive or negative, followed by statistical testing (t-tests and basic correlations) on the average scores. The findings reveal that over 90 percent of reviews in each region are positive, indicating that having additional reviews does not significantly enhance prices. However, listings with predominantly positive feedback exhibit slightly higher acceptance rates, suggesting that sentiment polarity, rather than the sheer volume of reviews, is a more critical factor for host success. Additionally, budget listings often gather extensive reviews while maintaining competitive pricing, whereas premium listings sustain higher prices with fewer but highly positive reviews. These results underscore the importance of sentiment quality over quantity in shaping guest behavior and pricing strategies in an overwhelmingly positive review environment. 

**Abstract (ZH)**: 本研究考察了 Airbnb 客人正面和负面评论是否影响六大美地区域的接受率和租金价格：罗德岛、布劳沃德县、芝加哥、达拉斯、圣地亚哥和波士顿。收集并分析了数千条评论，使用自然语言处理（NLP）将情感分类为正面或负面，随后通过统计检验（t 检验和基本相关性分析）对平均得分进行分析。研究发现，每个地区的评论中有超过 90% 是正面的，表明额外的评论对提升价格影响不大。然而，主要正面反馈的房源展现出略微更高的接受率，表明情绪极性而不是评论数量是影响房东成功的关键因素。此外，经济型房源通常会积累大量评论同时保持竞争力价格，而高端房源则凭借少量但高度正面的评论维持较高价格。这些结果强调，在大量正面评论的环境中，情感质量比数量对客人行为和定价策略的影响更为重要。 

---
# A synthetic dataset of French electric load curves with temperature conditioning 

**Title (ZH)**: 带有温度条件的法电负荷曲线合成数据集 

**Authors**: Tahar Nabil, Ghislain Agoua, Pierre Cauchois, Anne De Moliner, Benoît Grossin  

**Link**: [PDF](https://arxiv.org/pdf/2504.14046)  

**Abstract**: The undergoing energy transition is causing behavioral changes in electricity use, e.g. with self-consumption of local generation, or flexibility services for demand control. To better understand these changes and the challenges they induce, accessing individual smart meter data is crucial. Yet this is personal data under the European GDPR. A widespread use of such data requires thus to create synthetic realistic and privacy-preserving samples. This paper introduces a new synthetic load curve dataset generated by conditional latent diffusion. We also provide the contracted power, time-of-use plan and local temperature used for generation. Fidelity, utility and privacy of the dataset are thoroughly evaluated, demonstrating its good quality and thereby supporting its interest for energy modeling applications. 

**Abstract (ZH)**: 正在进行的能源转型正在改变 electricity 使用行为，例如通过本地发电的自我消费或需求控制的柔性服务。为了更好地理解和应对这些变化及其带来的挑战，访问个体智能电表数据至关重要。然而，这些数据属于个人数据且受欧盟GDPR保护。因此，广泛使用这些数据需要创建合成的、具现实性和隐私保护的数据样本。本文介绍了一种由条件潜在扩散生成的新合成负荷曲线数据集，并提供了用于生成的功率合同、时间-of-use 计划和当地温度。对数据集的忠实度、效用和隐私进行了详尽评估，展示了其良好的质量，从而支持其在能源建模应用中的应用价值。 

---
# Causal pieces: analysing and improving spiking neural networks piece by piece 

**Title (ZH)**: 因果性组件：逐个分析与提升脉冲神经网络 

**Authors**: Dominik Dold, Philipp Christian Petersen  

**Link**: [PDF](https://arxiv.org/pdf/2504.14015)  

**Abstract**: We introduce a novel concept for spiking neural networks (SNNs) derived from the idea of "linear pieces" used to analyse the expressiveness and trainability of artificial neural networks (ANNs). We prove that the input domain of SNNs decomposes into distinct causal regions where its output spike times are locally Lipschitz continuous with respect to the input spike times and network parameters. The number of such regions - which we call "causal pieces" - is a measure of the approximation capabilities of SNNs. In particular, we demonstrate in simulation that parameter initialisations which yield a high number of causal pieces on the training set strongly correlate with SNN training success. Moreover, we find that feedforward SNNs with purely positive weights exhibit a surprisingly high number of causal pieces, allowing them to achieve competitive performance levels on benchmark tasks. We believe that causal pieces are not only a powerful and principled tool for improving SNNs, but might also open up new ways of comparing SNNs and ANNs in the future. 

**Abstract (ZH)**: 我们介绍了一种源自“线性片段”思想的新型突触神经网络（SNN）概念，用于分析和训练人工神经网络（ANN）。我们证明SNN的输入域分解为不同的因果区域，在这些区域中，输出尖锋时间对输入尖锋时间和网络参数的局部Lipschitz连续。这样的区域数量——我们称为“因果片段”——是SNN逼近能力的度量。特别地，在模拟中我们证明，能够在训练集上产生高数量因果片段的参数初始化与SNN训练成功高度相关。此外，我们发现具有全正权重的前向SNN显示出惊人的高数量因果片段，使它们能够在基准任务上达到竞争力的性能水平。我们认为，因果片段不仅是改进SNN的一种强大且基于原理的工具，还可能在未来为比较SNN和ANN开辟新的途径。 

---
# PC-DeepNet: A GNSS Positioning Error Minimization Framework Using Permutation-Invariant Deep Neural Network 

**Title (ZH)**: PC-DeepNet：一种使用排列不变深度神经网络的GNSS定位误差最小化框架 

**Authors**: M. Humayun Kabir, Md. Ali Hasan, Md. Shafiqul Islam, Kyeongjun Ko, Wonjae Shin  

**Link**: [PDF](https://arxiv.org/pdf/2504.13990)  

**Abstract**: Global navigation satellite systems (GNSS) face significant challenges in urban and sub-urban areas due to non-line-of-sight (NLOS) propagation, multipath effects, and low received power levels, resulting in highly non-linear and non-Gaussian measurement error distributions. In light of this, conventional model-based positioning approaches, which rely on Gaussian error approximations, struggle to achieve precise localization under these conditions. To overcome these challenges, we put forth a novel learning-based framework, PC-DeepNet, that employs a permutation-invariant (PI) deep neural network (DNN) to estimate position corrections (PC). This approach is designed to ensure robustness against changes in the number and/or order of visible satellite measurements, a common issue in GNSS systems, while leveraging NLOS and multipath indicators as features to enhance positioning accuracy in challenging urban and sub-urban environments. To validate the performance of the proposed framework, we compare the positioning error with state-of-the-art model-based and learning-based positioning methods using two publicly available datasets. The results confirm that proposed PC-DeepNet achieves superior accuracy than existing model-based and learning-based methods while exhibiting lower computational complexity compared to previous learning-based approaches. 

**Abstract (ZH)**: 全球导航卫星系统（GNSS）在城市和亚城市区域面临着由非视距（NLOS）传播、多路径效应和接收到的信号功率低所导致的显著挑战，这导致了高度非线性和非高斯测量误差分布。鉴于此，依赖高斯误差近似的传统基于模型的定位方法在这些条件下难以实现精确定位。为克服这些挑战，我们提出了一种新型基于学习的框架PC-DeepNet，该框架采用不变置换（PI）深度神经网络（DNN）来估计位置校正（PC）。该方法旨在确保在可见卫星测量数量和/或顺序发生变化时的鲁棒性，这一问题是GNSS系统的常见问题，同时利用非视距和多路径指示器作为特征来提升在具有挑战性的城市和亚城市环境中的定位精度。为了验证所提出框架的性能，我们使用两个公开可用的数据集与最先进的基于模型和基于学习的定位方法进行定位误差比较。结果表明，所提出的PC-DeepNet在精度上优于现有基于模型和基于学习的方法，并且相比之前的基于学习的方法具有更低的计算复杂度。 

---
# Entropy Rectifying Guidance for Diffusion and Flow Models 

**Title (ZH)**: 熵校正指导下的扩散与流模型 

**Authors**: Tariq Berrada Ifriqi, Adriana Romero-Soriano, Michal Drozdzal, Jakob Verbeek, Karteek Alahari  

**Link**: [PDF](https://arxiv.org/pdf/2504.13987)  

**Abstract**: Guidance techniques are commonly used in diffusion and flow models to improve image quality and consistency for conditional generative tasks such as class-conditional and text-to-image generation. In particular, classifier-free guidance (CFG) -- the most widely adopted guidance technique -- contrasts conditional and unconditional predictions to improve the generated images. This results, however, in trade-offs across quality, diversity and consistency, improving some at the expense of others. While recent work has shown that it is possible to disentangle these factors to some extent, such methods come with an overhead of requiring an additional (weaker) model, or require more forward passes per sampling step. In this paper, we propose Entropy Rectifying Guidance (ERG), a simple and effective guidance mechanism based on inference-time changes in the attention mechanism of state-of-the-art diffusion transformer architectures, which allows for simultaneous improvements over image quality, diversity and prompt consistency. ERG is more general than CFG and similar guidance techniques, as it extends to unconditional sampling. ERG results in significant improvements in various generation tasks such as text-to-image, class-conditional and unconditional image generation. We also show that ERG can be seamlessly combined with other recent guidance methods such as CADS and APG, further boosting generation performance. 

**Abstract (ZH)**: 熵矫正引导（ERG）：一种基于推断时注意机制改变的简单有效引导机制 

---
# On the redundancy of short and heterogeneous sequences of belief revisions 

**Title (ZH)**: 关于信念修订的短异质序列的冗余性 

**Authors**: Paolo Liberatore  

**Link**: [PDF](https://arxiv.org/pdf/2504.13986)  

**Abstract**: Forgetting a specific belief revision episode may not erase information because the other revisions may provide the same information or allow to deduce it. Whether it does was proved coNP-hard for sequence of two arbitrary lexicographic revision or arbitrarily long lexicographic Horn revision. A polynomial algorithm is presented for the case of two Horn revision. Heterogeneous sequences of revisions were proved to belong in Delta2. Their previously proved coNP-hardness is enhanced by a proof of NP-hardness. 

**Abstract (ZH)**: 遗忘特定的信念修订事件未必会消除信息，因为其他修订可能会提供相同的信息或允许推导出该信息。对于任意两个字典序修订或任意长的字典序Horn修订序列，该问题已被证明为coNP-hard。对于两个Horn修订的情况，提出了一种多项式时间算法。不同类型的修订序列被证明属于Δ²。通过证明NP-hardness，增强并扩展了先前证明的coNP-hardness。 

---
# CacheFormer: High Attention-Based Segment Caching 

**Title (ZH)**: CacheFormer：基于高注意力的 segment 缓存 

**Authors**: Sushant Singh, Ausif Mahmood  

**Link**: [PDF](https://arxiv.org/pdf/2504.13981)  

**Abstract**: Efficiently handling long contexts in transformer-based language models with low perplexity is an active area of research. Numerous recent approaches like Linformer, Longformer, Performer, and Structured state space models (SSMs)., have not fully resolved this problem. All these models strive to reduce the quadratic time complexity of the attention mechanism while minimizing the loss in quality due to the effective compression of the long context. Inspired by the cache and virtual memory principle in computers, where in case of a cache miss, not only the needed data is retrieved from the memory, but the adjacent data is also obtained, we apply this concept to handling long contexts by dividing it into small segments. In our design, we retrieve the nearby segments in an uncompressed form when high segment-level attention occurs at the compressed level. Our en-hancements for handling long context include aggregating four attention mechanisms consisting of short sliding window attention, long compressed segmented attention, dynamically retrieving top k high attention uncompressed segments, and overlapping segments in long segment attention to avoid segment fragmentation. These enhancements result in an architecture that outperforms ex-isting SOTA architectures with an average perplexity improvement of 8.5% over similar model sizes. 

**Abstract (ZH)**: 基于变压器的语言模型高效处理长上下文并保持低 perplexity 是一个活跃的研究领域。尽管 Linformer、Longformer、Performer 和结构化状态空间模型 (SSMs) 等众多近期方法有所尝试，但仍未完全解决这一问题。所有这些模型都致力于减轻注意力机制的二次时间复杂度，同时尽可能减少由于有效压缩长上下文而导致的质量损失。受计算机中的缓存和虚拟内存原理启发，当发生缓存缺失时，不仅会从内存中检索所需的数据，还会获取相邻的数据，我们在此原理基础上，通过将长上下文分割为小段，来处理长上下文。在我们的设计中，在压缩级别出现高段级注意力时，会以不压缩的形式检索附近的段。为了处理长上下文，我们增强了四种注意力机制，包括短滑动窗口注意力、长压缩分割注意力、动态检索高注意力不压缩段以及长段注意力中的重叠段，以避免段落碎片化。这些增强提升了架构性能，相较于同等模型规模，平均 perplexity 改进幅度为 8.5%。 

---
# Framework, Standards, Applications and Best practices of Responsible AI : A Comprehensive Survey 

**Title (ZH)**: 负责任人工智能的框架、标准、应用及最佳实践综述 

**Authors**: Thippa Reddy Gadekallu, Kapal Dev, Sunder Ali Khowaja, Weizheng Wang, Hailin Feng, Kai Fang, Sharnil Pandya, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13979)  

**Abstract**: Responsible Artificial Intelligence (RAI) is a combination of ethics associated with the usage of artificial intelligence aligned with the common and standard frameworks. This survey paper extensively discusses the global and national standards, applications of RAI, current technology and ongoing projects using RAI, and possible challenges in implementing and designing RAI in the industries and projects based on AI. Currently, ethical standards and implementation of RAI are decoupled which caters each industry to follow their own standards to use AI ethically. Many global firms and government organizations are taking necessary initiatives to design a common and standard framework. Social pressure and unethical way of using AI forces the RAI design rather than implementation. 

**Abstract (ZH)**: 负责任的人工智能（RAI）：伦理与全球及国家标准、应用、技术及挑战的研究 

---
# Gas Station of the Future: A Perspective on AI/ML and IoT in Retail Downstream 

**Title (ZH)**: 未来加油站：AI/ML与IoT在零售下游领域的视角 

**Authors**: Wrick Talukdar  

**Link**: [PDF](https://arxiv.org/pdf/2504.13976)  

**Abstract**: The gas station of the future is poised to transform from a simple fuel dispensing center into an intelligent retail hub, driven by advancements in Artificial Intelligence (AI), Machine Learning (ML), and the Internet of Things (IoT). This paper explores how technology is reshaping the retail downstream sector while briefly addressing the upstream and midstream segments. By leveraging AI/ML for predictive analytics, dynamic pricing, personalized customer engagement, and IoT for real-time monitoring and automation, the future gas station will redefine the fuel retail experience. Additionally, this paper incorporates statistics, AI/ML core technical concepts, mathematical formulations, case studies, and a proposed framework for a fully autonomous gas station. 

**Abstract (ZH)**: 未来加油站即将从简单的燃油供应中心转变为由人工智能、机器学习和物联网驱动的智能零售枢纽，论文探讨技术如何重塑零售下游产业，并简要涉及上游和中期产业。通过利用人工智能/机器学习进行预测分析、动态定价、个性化客户服务以及物联网进行实时监控和自动化，未来的加油站将重新定义燃油零售体验。此外，本文还包含统计数据、人工智能/机器学习核心技术概念、数学公式、案例研究及一个全自主加油站的拟议框架。 

---
# Enhancing Stroke Diagnosis in the Brain Using a Weighted Deep Learning Approach 

**Title (ZH)**: 使用加权深度学习方法增强脑卒中诊断 

**Authors**: Yao Zhiwan, Reza Zarrab, Jean Dubois  

**Link**: [PDF](https://arxiv.org/pdf/2504.13974)  

**Abstract**: A brain stroke occurs when blood flow to a part of the brain is disrupted, leading to cell death. Traditional stroke diagnosis methods, such as CT scans and MRIs, are costly and time-consuming. This study proposes a weighted voting ensemble (WVE) machine learning model that combines predictions from classifiers like random forest, Deep Learning, and histogram-based gradient boosting to predict strokes more effectively. The model achieved 94.91% accuracy on a private dataset, enabling early risk assessment and prevention. Future research could explore optimization techniques to further enhance accuracy. 

**Abstract (ZH)**: 脑卒中发生时，脑部某部分的血液供应被中断，导致细胞死亡。传统的大脑中风诊断方法，如CT扫描和MRI，成本高且耗时。本研究提出了一种加权投票集成(WVE)机器学习模型，该模型结合了随机森林、深度学习和直方图基梯度提升等分类器的预测，以更有效地预测中风。该模型在私有数据集上实现了94.91%的准确率，能够进行早期风险评估和预防。未来的研究可以探索优化技术以进一步提高准确率。 

---
# The Future of Internet of Things and Multimodal Language Models in 6G Networks: Opportunities and Challenges 

**Title (ZH)**: 6G网络中物联网与多模态语言模型的未来：机遇与挑战 

**Authors**: Abdelrahman Soliman  

**Link**: [PDF](https://arxiv.org/pdf/2504.13971)  

**Abstract**: Based on recent trends in artificial intelligence and IoT research. The cooperative potential of integrating the Internet of Things (IoT) and Multimodal Language Models (MLLMs) is presented in this survey paper for future 6G systems. It focuses on the applications of this integration in different fields, such as healthcare, agriculture, and smart cities, and investigates the four pillars of IoT integration, such as sensors, communication, processing, and security. The paper provides a comprehensive description of IoT and MLLM technologies and applications, addresses the role of multimodality in each pillar, and concludes with an overview of the most significant challenges and directions for future research. The general survey is a roadmap for researchers interested in tracing the application areas of MLLMs and IoT, highlighting the potential and challenges in this rapidly growing field. The survey recognizes the need to deal with data availability, computational expense, privacy, and real-time processing to harness the complete potential of IoT, MLLM, and 6G technology 

**Abstract (ZH)**: 基于人工智能和物联网研究的最新趋势，本文综述了将物联网(IoT)与多模态语言模型(MLLMs)集成的协同潜力，为未来的6G系统提供参考。本文集中探讨了这种集成在不同领域（如医疗保健、农业和智慧城市）的应用，并研究了物联网集成的四大支柱，即传感器、通信、处理和安全。文章全面描述了物联网和多模态语言模型的技术和应用，分析了在每个支柱中多模态的作用，并总结了这一快速发展的领域中最具挑战性的问题和未来研究方向。综述为有兴趣跟踪多模态语言模型和物联网应用领域的研究人员提供了一条 roadmap，并强调了利用物联网、多模态语言模型和6G技术全部潜力时所面临的机遇与挑战。 

---
# Tinker Tales: Interactive Storytelling Framework for Early Childhood Narrative Development and AI Literacy 

**Title (ZH)**: 玩转故事：面向幼儿叙事发展与人工智能 literacy 的交互式叙事框架 

**Authors**: Nayoung Choi, Peace Cyebukayire, Jinho D. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2504.13969)  

**Abstract**: This paper presents Tinker Tales, an interactive storytelling framework in the format of a board game, designed to support both narrative development and AI literacy in early childhood. The framework integrates tangible and speech-based interactions with AI through NFC chip-attached pawns and tokens, along with a speaker and microphone. Children select and define key story elements-such as characters, places, items, and emotions-using the pawns and tokens, providing further details to the AI and receiving proper assistance, similar to how adults prompt AI for specific tasks (e.g., writing). For evaluation, several game sessions were simulated with a child AI agent, and the quality and safety of the generated stories were assessed from various perspectives. This work highlights the potential of combining physical and digital elements in AI literacy, offering a safe and engaging way for children to learn how to effectively collaborate with AI. 

**Abstract (ZH)**: Tinker Tales：一种板游戏式互动叙事框架，支持儿童早期的叙事发展与AI素养 

---
# CONTINA: Confidence Interval for Traffic Demand Prediction with Coverage Guarantee 

**Title (ZH)**: CONTINA：带有覆盖保证的交通需求预测置信区间 

**Authors**: Chao Yang, Xiannan Huang, Shuhan Qiu, Yan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.13961)  

**Abstract**: Accurate short-term traffic demand prediction is critical for the operation of traffic systems. Besides point estimation, the confidence interval of the prediction is also of great importance. Many models for traffic operations, such as shared bike rebalancing and taxi dispatching, take into account the uncertainty of future demand and require confidence intervals as the input. However, existing methods for confidence interval modeling rely on strict assumptions, such as unchanging traffic patterns and correct model specifications, to guarantee enough coverage. Therefore, the confidence intervals provided could be invalid, especially in a changing traffic environment. To fill this gap, we propose an efficient method, CONTINA (Conformal Traffic Intervals with Adaptation) to provide interval predictions that can adapt to external changes. By collecting the errors of interval during deployment, the method can adjust the interval in the next step by widening it if the errors are too large or shortening it otherwise. Furthermore, we theoretically prove that the coverage of the confidence intervals provided by our method converges to the target coverage level. Experiments across four real-world datasets and prediction models demonstrate that the proposed method can provide valid confidence intervals with shorter lengths. Our method can help traffic management personnel develop a more reasonable and robust operation plan in practice. And we release the code, model and dataset in \href{ this https URL}{ Github}. 

**Abstract (ZH)**: 准确的短时交通需求预测对于交通系统运营至关重要。除了点估计外，预测的置信区间同样十分重要。许多交通运营模型，如共享自行车重新平衡和出租车调度，都会考虑未来需求的不确定性，并要求输入置信区间。然而，现有的置信区间建模方法依赖于严格的假设，如交通模式不变和模型规格正确，以保证足够的覆盖范围。因此，提供的置信区间可能是无效的，尤其是在不断变化的交通环境中。为此，我们提出了一种高效的 方法，CONTINA（Conformal Traffic Intervals with Adaptation），以提供能够适应外部变化的区间预测。通过收集部署期间的误差，该方法可以在下一次调整区间时，如果误差过大则使其变宽，否则使其变窄。此外，我们从理论上证明了由我们方法提供的置信区间的覆盖范围将收敛到目标覆盖水平。实验结果表明，本方法可以在较短的区间长度下提供有效的置信区间。该方法有助于交通管理人员在实践中制定更为合理和稳健的运营计划。我们已将代码、模型和数据集发布在GitHub上。 

---
# AI Safety Should Prioritize the Future of Work 

**Title (ZH)**: AI安全应关注工作未来 

**Authors**: Sanchaita Hazra, Bodhisattwa Prasad Majumder, Tuhin Chakrabarty  

**Link**: [PDF](https://arxiv.org/pdf/2504.13959)  

**Abstract**: Current efforts in AI safety prioritize filtering harmful content, preventing manipulation of human behavior, and eliminating existential risks in cybersecurity or biosecurity. While pressing, this narrow focus overlooks critical human-centric considerations that shape the long-term trajectory of a society. In this position paper, we identify the risks of overlooking the impact of AI on the future of work and recommend comprehensive transition support towards the evolution of meaningful labor with human agency. Through the lens of economic theories, we highlight the intertemporal impacts of AI on human livelihood and the structural changes in labor markets that exacerbate income inequality. Additionally, the closed-source approach of major stakeholders in AI development resembles rent-seeking behavior through exploiting resources, breeding mediocrity in creative labor, and monopolizing innovation. To address this, we argue in favor of a robust international copyright anatomy supported by implementing collective licensing that ensures fair compensation mechanisms for using data to train AI models. We strongly recommend a pro-worker framework of global AI governance to enhance shared prosperity and economic justice while reducing technical debt. 

**Abstract (ZH)**: 当前在AI安全性方面的努力主要集中在过滤有害内容、防止操纵人类行为以及消除网络安全或生物安全中的生存风险。虽然这些是紧迫的问题，但这种狭窄的焦点忽视了塑造社会长期轨迹的关键的人本因素。在本文中，我们识别了忽视AI对未来工作影响的风险，并建议全面的支持向有意义劳动演变的过渡，其中包含人类自主权。通过经济学理论的视角，我们强调了AI对未来生计的跨时期影响以及劳动市场结构变化对收入不平等的加剧。此外，主要AI开发利益相关者采取的封闭源代码方法类似于通过利用资源、在创造性劳动中培养平庸和垄断创新来寻求租金的行为。为此，我们主张建立一个基于集体许可的坚实国际版权体系，以确保公平的补偿机制，并训练AI模型使用数据。我们强烈建议建立一个有利于工人的全球AI治理体系，以促进共享繁荣和经济正义，同时减少技术债务。 

---
# Naming is framing: How cybersecurity's language problems are repeating in AI governance 

**Title (ZH)**: 命名即框架：网络安全语言问题如何在AI治理中重演 

**Authors**: Liane Potter  

**Link**: [PDF](https://arxiv.org/pdf/2504.13957)  

**Abstract**: Language is not neutral; it frames understanding, structures power, and shapes governance. This paper argues that misnomers like cybersecurity and artificial intelligence (AI) are more than semantic quirks; they carry significant governance risks by obscuring human agency, inflating expectations, and distorting accountability. Drawing on lessons from cybersecurity's linguistic pitfalls, such as the 'weakest link' narrative, this paper highlights how AI discourse is falling into similar traps with metaphors like 'alignment,' 'black box,' and 'hallucination.' These terms embed adversarial, mystifying, or overly technical assumptions into governance structures. In response, the paper advocates for a language-first approach to AI governance: one that interrogates dominant metaphors, foregrounds human roles, and co-develops a lexicon that is precise, inclusive, and reflexive. This paper contends that linguistic reform is not peripheral to governance but central to the construction of transparent, equitable, and anticipatory regulatory frameworks. 

**Abstract (ZH)**: 语言不是中性的；它塑造理解、结构权力并影响治理。本文认为，诸如网络安全和人工智能（AI）之类的术语不仅仅是语义上的怪癖；它们通过模糊人类agency、夸大期望和扭曲问责制带来了重要的治理风险。本文借鉴网络安全语言陷阱的经验教训，如“最弱环节”叙事，指出AI话语正在落入类似陷阱，使用诸如“对齐”、“黑箱”和“幻觉”之类的隐喻。这些术语将对抗性、迷惑性的或过度技术化的假设嵌入到治理结构中。为此，本文提倡在AI治理中采取语言先行的方法：这种方法质疑主导隐喻，强调人类的作用，并共同开发一个精确、包容和反思性的词汇表。本文认为，语言改革对于建构透明、公平和前瞻性的治理框架是核心而非边缘问题。 

---
# Thousand Voices of Trauma: A Large-Scale Synthetic Dataset for Modeling Prolonged Exposure Therapy Conversations 

**Title (ZH)**: 千声创伤：模型 prolonged exposure 治疗对话的大规模合成数据集 

**Authors**: Suhas BN, Dominik Mattioli, Saeed Abdullah, Rosa I. Arriaga, Chris W. Wiese, Andrew M. Sherrill  

**Link**: [PDF](https://arxiv.org/pdf/2504.13955)  

**Abstract**: The advancement of AI systems for mental health support is hindered by limited access to therapeutic conversation data, particularly for trauma treatment. We present Thousand Voices of Trauma, a synthetic benchmark dataset of 3,000 therapy conversations based on Prolonged Exposure therapy protocols for Post-traumatic Stress Disorder (PTSD). The dataset comprises 500 unique cases, each explored through six conversational perspectives that mirror the progression of therapy from initial anxiety to peak distress to emotional processing. We incorporated diverse demographic profiles (ages 18-80, M=49.3, 49.4% male, 44.4% female, 6.2% non-binary), 20 trauma types, and 10 trauma-related behaviors using deterministic and probabilistic generation methods. Analysis reveals realistic distributions of trauma types (witnessing violence 10.6%, bullying 10.2%) and symptoms (nightmares 23.4%, substance abuse 20.8%). Clinical experts validated the dataset's therapeutic fidelity, highlighting its emotional depth while suggesting refinements for greater authenticity. We also developed an emotional trajectory benchmark with standardized metrics for evaluating model responses. This privacy-preserving dataset addresses critical gaps in trauma-focused mental health data, offering a valuable resource for advancing both patient-facing applications and clinician training tools. 

**Abstract (ZH)**: 基于长期暴露疗法协议的大规模合成创伤治疗对话数据集： thousand voices of trauma 

---
# Generative System Dynamics in Recurrent Neural Networks 

**Title (ZH)**: 生成系统动力学在递归神经网络中的应用 

**Authors**: Michele Casoni, Tommaso Guidi, Alessandro Betti, Stefano Melacci, Marco Gori  

**Link**: [PDF](https://arxiv.org/pdf/2504.13951)  

**Abstract**: In this study, we investigate the continuous time dynamics of Recurrent Neural Networks (RNNs), focusing on systems with nonlinear activation functions. The objective of this work is to identify conditions under which RNNs exhibit perpetual oscillatory behavior, without converging to static fixed points. We establish that skew-symmetric weight matrices are fundamental to enable stable limit cycles in both linear and nonlinear configurations. We further demonstrate that hyperbolic tangent-like activation functions (odd, bounded, and continuous) preserve these oscillatory dynamics by ensuring motion invariants in state space. Numerical simulations showcase how nonlinear activation functions not only maintain limit cycles, but also enhance the numerical stability of the system integration process, mitigating those instabilities that are commonly associated with the forward Euler method. The experimental results of this analysis highlight practical considerations for designing neural architectures capable of capturing complex temporal dependencies, i.e., strategies for enhancing memorization skills in recurrent models. 

**Abstract (ZH)**: 本研究探讨了具有非线性激活函数的循环神经网络（RNN）的连续时间动力学，重点研究了在系统中实现持久振荡行为而不收敛于静态固定点的条件。我们证明了反对称权重矩阵是实现线性与非线性配置下稳定极限环的关键。进一步研究表明，类似双曲正切的激活函数（奇函数、有界且连续）通过在状态空间中保持运动不变量，来维护这些振荡动力学。数值模拟展示了非线性激活函数不仅能够维持极限环，还能增强系统积分过程的数值稳定性，减轻与向前欧拉方法相关的那些不稳定现象。本分析的实验结果强调了设计能够捕捉复杂时间依赖性的神经架构的实用考虑，即增强循环模型记忆能力的策略。 

---
# Open-Medical-R1: How to Choose Data for RLVR Training at Medicine Domain 

**Title (ZH)**: Open-Medical-R1: 如何在医疗领域选择数据进行RLVR训练 

**Authors**: Zhongxi Qiu, Zhang Zhang, Yan Hu, Heng Li, Jiang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.13950)  

**Abstract**: This paper explores optimal data selection strategies for Reinforcement Learning with Verified Rewards (RLVR) training in the medical domain. While RLVR has shown exceptional potential for enhancing reasoning capabilities in large language models, most prior implementations have focused on mathematics and logical puzzles, with limited exploration of domain-specific applications like medicine. We investigate four distinct data sampling strategies from MedQA-USMLE: random sampling (baseline), and filtering using Phi-4, Gemma-3-27b-it, and Gemma-3-12b-it models. Using Gemma-3-12b-it as our base model and implementing Group Relative Policy Optimization (GRPO), we evaluate performance across multiple benchmarks including MMLU, GSM8K, MMLU-Pro, and CMMLU. Our findings demonstrate that models trained on filtered data generally outperform those trained on randomly selected samples. Notably, training on self-filtered samples (using Gemma-3-12b-it for filtering) achieved superior performance in medical domains but showed reduced robustness across different benchmarks, while filtering with larger models from the same series yielded better overall robustness. These results provide valuable insights into effective data organization strategies for RLVR in specialized domains and highlight the importance of thoughtful data selection in achieving optimal performance. You can access our repository (this https URL) to get the codes. 

**Abstract (ZH)**: 这篇论文探讨了在医学领域使用验证奖励强化学习（RLVR）训练时的最优数据选择策略。尽管RLVR在增强大型语言模型的推理能力方面表现出出色的潜力，但大多数之前的实现主要集中于数学和逻辑谜题，对医学等特定领域的应用探索有限。我们从MedQA-USMLE中研究了四种不同的数据采样策略：随机采样（基线）、以及使用Phi-4、Gemma-3-27b-it和Gemma-3-12b-it模型进行过滤。以Gemma-3-12b-it作为基础模型并采用组相对策略优化（GRPO），我们在MMLU、GSM8K、MMLU-Pro和CMMLU等多个基准上评估了性能。研究结果显示，使用过滤数据训练的模型通常优于随机选择样本训练的模型。值得注意的是，使用Gemma-3-12b-it进行自我过滤的样本在医学领域取得了更好的性能，但在不同基准上的鲁棒性较差，而使用该系列更大模型进行过滤则表现出更好的整体鲁棒性。这些结果为RLVR在专门领域的有效数据组织策略提供了宝贵见解，并强调了在实现最佳性能时进行精心数据选择的重要性。您可以访问我们的仓库（点击这里）获取代码。 

---
# On Revealing the Hidden Problem Structure in Real-World and Theoretical Problems Using Walsh Coefficient Influence 

**Title (ZH)**: 基于沃尔什系数影响在现实世界和理论问题中揭示隐藏问题结构的研究 

**Authors**: M. W. Przewozniczek, F. Chicano, R. Tinós, J. Nalepa, B. Ruszczak, A. M. Wijata  

**Link**: [PDF](https://arxiv.org/pdf/2504.13949)  

**Abstract**: Gray-box optimization employs Walsh decomposition to obtain non-linear variable dependencies and utilize them to propose masks of variables that have a joint non-linear influence on fitness value. These masks significantly improve the effectiveness of variation operators. In some problems, all variables are non-linearly dependent, making the aforementioned masks useless. We analyze the features of the real-world instances of such problems and show that many of their dependencies may have noise-like origins. Such noise-caused dependencies are irrelevant to the optimization process and can be ignored. To identify them, we propose extending the use of Walsh decomposition by measuring variable dependency strength that allows the construction of the weighted dynamic Variable Interaction Graph (wdVIG). wdVIGs adjust the dependency strength to mixed individuals. They allow the filtering of irrelevant dependencies and re-enable using dependency-based masks by variation operators. We verify the wdVIG potential on a large benchmark suite. For problems with noise, the wdVIG masks can improve the optimizer's effectiveness. If all dependencies are relevant for the optimization, i.e., the problem is not noised, the influence of wdVIG masks is similar to that of state-of-the-art structures of this kind. 

**Abstract (ZH)**: 灰盒优化采用沃尔什分解获取非线性变量依赖关系，并利用这些依赖关系提出具有联合非线性影响于适应值的变量掩码。这些掩码显著提高了变异操作的有效性。在某些问题中，所有变量都呈非线性依赖，使得上述掩码变得无用。我们分析了此类问题的实际实例特征，并显示其中许多依赖可能是噪声引起的。这些由噪声引起的依赖与优化过程无关，并且可以忽略。为了识别它们，我们建议通过测量变量依赖强度来扩展沃尔什分解的应用，从而构建加权动态变量交互图（wdVIG）。wdVIG 调整依赖强度以适应混合个体，并允许过滤无关依赖并重新启用基于依赖的掩码。我们通过大型基准套件验证了 wdVIG 的潜在价值。对于具有噪声的问题，wdVIG 掩码可以提高优化器的效果。如果所有依赖都对优化相关，即问题无噪声，wdVIG 掩码的影响与此类最先进的结构相当。 

---
# Using customized GPT to develop prompting proficiency in architectural AI-generated images 

**Title (ZH)**: 使用定制化GPT提升 architectural AI生成图像的提示技巧 

**Authors**: Juan David Salazar Rodriguez, Sam Conrad Joyce, Julfendi Julfendi  

**Link**: [PDF](https://arxiv.org/pdf/2504.13948)  

**Abstract**: This research investigates the use of customized GPT models to enhance prompting proficiency among architecture students when generating AI-driven images. Prompt engineering is increasingly essential in architectural education due to the widespread adoption of generative AI tools. This study utilized a mixed-methods experimental design involving architecture students divided into three distinct groups: a control group receiving no structured support, a second group provided with structured prompting guides, and a third group supported by both structured guides and interactive AI personas. Students engaged in reverse engineering tasks, first guessing provided image prompts and then generating their own prompts, aiming to boost critical thinking and prompting skills. Variables examined included time spent prompting, word count, prompt similarity, and concreteness. Quantitative analysis involved correlation assessments between these variables and a one-way ANOVA to evaluate differences across groups. While several correlations showed meaningful relationships, not all were statistically significant. ANOVA results indicated statistically significant improvements in word count, similarity, and concreteness, especially in the group supported by AI personas and structured prompting guides. Qualitative feedback complemented these findings, revealing enhanced confidence and critical thinking skills in students. These results suggest tailored GPT interactions substantially improve students' ability to communicate architectural concepts clearly and effectively. 

**Abstract (ZH)**: 本研究探讨了定制GPT模型在提升建筑学生生成AI驱动图像时的提示 proficiency 方面的应用。由于生成型AI工具的广泛应用，提示工程在建筑教育中变得越来越重要。本研究采用混合方法实验设计，将建筑学生分为三组：一组为对照组，未提供结构化支持；第二组提供了结构化提示指南；第三组则同时得到了结构化指南和互动AI角色的支持。学生进行了逆向工程任务，首先猜测提供的图像提示，然后生成自己的提示，旨在提升批判性思维和提示技巧。研究变量包括提示时间、字数、提示相似度和具体性。定量分析包括变量间的相关性评估和单因素方差分析（ANOVA）以评估组间差异。虽然一些相关性显示出有意义的关系，但并非所有都是统计显著的。ANOVA结果表明，在得到AI角色和结构化提示指南支持的组中，字数、相似度和具体性有统计显著的提升。定性反馈补充了这些发现，显示学生自信心和批判性思维能力有所提升。这些结果表明，个性化GPT交互显著提高了学生清晰有效地传达建筑概念的能力。 

---
# Intelligence of Things: A Spatial Context-Aware Control System for Smart Devices 

**Title (ZH)**: 物联网中的智能：一种基于空间上下文的智能设备控制系統 

**Authors**: Sukanth Kalivarathan, Muhmmad Abrar Raja Mohamed, Aswathy Ravikumar, S Harini  

**Link**: [PDF](https://arxiv.org/pdf/2504.13942)  

**Abstract**: This paper introduces Intelligence of Things (INOT), a novel spatial context-aware control system that enhances smart home automation through intuitive spatial reasoning. Current smart home systems largely rely on device-specific identifiers, limiting user interaction to explicit naming conventions rather than natural spatial references. INOT addresses this limitation through a modular architecture that integrates Vision Language Models with IoT control systems to enable natural language commands with spatial context (e.g., "turn on the light near the window"). The system comprises key components including an Onboarding Inference Engine, Zero-Shot Device Detection, Spatial Topology Inference, and Intent-Based Command Synthesis. A comprehensive user study with 15 participants demonstrated INOT's significant advantages over conventional systems like Google Home Assistant, with users reporting reduced cognitive workload (NASA-TLX scores decreased by an average of 13.17 points), higher ease-of-use ratings, and stronger preference (14 out of 15 participants). By eliminating the need to memorize device identifiers and enabling context-aware spatial commands, INOT represents a significant advancement in creating more intuitive and accessible smart home control systems. 

**Abstract (ZH)**: 基于事物的智能（INOT）：一种通过直观的空间推理增强智能家居自动化的新颖空间感知控制系统 

---
# Hashigo: A Next Generation Sketch Interactive System for Japanese Kanji 

**Title (ZH)**: Hashigo：下一代日语漢字交互绘图系统 

**Authors**: Paul Taele, Tracy Hammond  

**Link**: [PDF](https://arxiv.org/pdf/2504.13940)  

**Abstract**: Language students can increase their effectiveness in learning written Japanese by mastering the visual structure and written technique of Japanese kanji. Yet, existing kanji handwriting recognition systems do not assess the written technique sufficiently enough to discourage students from developing bad learning habits. In this paper, we describe our work on Hashigo, a kanji sketch interactive system which achieves human instructor-level critique and feedback on both the visual structure and written technique of students' sketched kanji. This type of automated critique and feedback allows students to target and correct specific deficiencies in their sketches that, if left untreated, are detrimental to effective long-term kanji learning. 

**Abstract (ZH)**: 语言学習者可以通过掌握日语漢字的視覺結構和書寫技巧來提高寫作日语的有效性。然而，現有的漢字書寫認知系統在評估書寫技巧方面并不充分，無法阻止學習者養成不良的學習習慣。本文介紹了我們對Hashigo的研究工作，这是一个漢字草稿交互系統，可以實現類似人類教練的評估和反饋，針對學生草書漢字的視覺結構和書寫技巧進行評価和指導。這種自動化的評估和反饋使學員能夠즉시纠正在草稿中出现的特定缺陷，這些缺陷如果不予治療，將對長期有效的漢字學習産生負面影響。 

---
# A Multi-Layered Research Framework for Human-Centered AI: Defining the Path to Explainability and Trust 

**Title (ZH)**: 以人为本的AI多层研究框架：通往可解释性和信任的道路 

**Authors**: Chameera De Silva, Thilina Halloluwa, Dhaval Vyas  

**Link**: [PDF](https://arxiv.org/pdf/2504.13926)  

**Abstract**: The integration of Artificial Intelligence (AI) into high-stakes domains such as healthcare, finance, and autonomous systems is often constrained by concerns over transparency, interpretability, and trust. While Human-Centered AI (HCAI) emphasizes alignment with human values, Explainable AI (XAI) enhances transparency by making AI decisions more understandable. However, the lack of a unified approach limits AI's effectiveness in critical decision-making scenarios. This paper presents a novel three-layered framework that bridges HCAI and XAI to establish a structured explainability paradigm. The framework comprises (1) a foundational AI model with built-in explainability mechanisms, (2) a human-centered explanation layer that tailors explanations based on cognitive load and user expertise, and (3) a dynamic feedback loop that refines explanations through real-time user interaction. The framework is evaluated across healthcare, finance, and software development, demonstrating its potential to enhance decision-making, regulatory compliance, and public trust. Our findings advance Human-Centered Explainable AI (HCXAI), fostering AI systems that are transparent, adaptable, and ethically aligned. 

**Abstract (ZH)**: 将人工智能集成到医疗、金融和自主系统等高 stakes 领域常常受限于透明度、可解释性和信任方面的担忧。以人为本的人工智能（HCAI）强调与人类价值观的契合，可解释的人工智能（XAI）通过使人工智能决策更具可理解性来增强透明度。然而，缺乏统一的方法限制了人工智能在关键决策场景中的有效性。本文提出了一种新颖的三层框架，将HCAI和XAI相结合，建立结构化的可解释性范式。该框架包括（1）具有内置可解释性机制的基础人工智能模型，（2）以人为本的解释层，根据认知负荷和用户专业知识定制解释，以及（3）通过实时用户交互优化解释的动态反馈循环。该框架在医疗、金融和软件开发等领域进行了评估，证明了其在增强决策、合规性和公众信任方面的潜力。我们的研究推进了以人为本的可解释人工智能（HCXAI），促进了透明、适应性强且伦理上一致的人工智能系统。 

---
# Modeling the quantum-like dynamics of human reliability ratings in Human-AI interactions by interaction dependent Hamiltonians 

**Title (ZH)**: 基于相互作用依赖哈密顿量的人类可靠性评级的量子似动态建模：人类-人工智能交互中的应用 

**Authors**: Johan van der Meer, Pamela Hoyte, Luisa Roeder, Peter Bruza  

**Link**: [PDF](https://arxiv.org/pdf/2504.13918)  

**Abstract**: As our information environments become ever more powered by artificial intelligence (AI), the phenomenon of trust in a human's interactions with this intelligence is becoming increasingly pertinent. For example, in the not too distant future, there will be teams of humans and intelligent robots involved in dealing with the repercussions of high-risk disaster situations such as hurricanes, earthquakes, or nuclear accidents. Even in such conditions of high uncertainty, humans and intelligent machines will need to engage in shared decision making, and trust is fundamental to the effectiveness of these interactions. A key challenge in modeling the dynamics of this trust is to provide a means to incorporate sensitivity to fluctuations in human trust judgments. In this article, we explore the ability of Quantum Random Walk models to model the dynamics of trust in human-AI interactions, and to integrate a sensitivity to fluctuations in participant trust judgments based on the nature of the interaction with the AI. We found that using empirical parameters to inform the use of different Hamiltonians can provide a promising means to model the evolution of trust in Human-AI interactions. 

**Abstract (ZH)**: 随着我们的信息环境日益依赖人工智能（AI），人类与这一智能互动中的信任现象变得 increasingly pertinent。例如，在不远的将来，人类和智能机器人将组成团队应对飓风、地震或核事故等高风险灾难情况的后果。即使在这种高度不确定的条件下，人类和智能机器也需要进行共享决策，而信任是这些互动有效性的基础。建模这种信任动态的一个关键挑战是提供一种方法，以敏感性地反映人类信任判断的变化。在本文中，我们探讨了量子随机游走模型在建模人类与AI互动中的信任动态方面的能力，并基于与AI互动的性质整合了参与者信任判断波动的敏感性。我们发现，使用实证参数来指导使用不同的哈密顿量是一种有前途的方法，用以建模人类与AI互动中信任的发展。 

---
# AI-Assisted Conversational Interviewing: Effects on Data Quality and User Experience 

**Title (ZH)**: AI辅助对话式访谈：对数据质量与用户体验的影响 

**Authors**: Soubhik Barari, Jarret Angbazo, Natalie Wang, Leah M. Christian, Elizabeth Dean, Zoe Slowinski, Brandon Sepulvado  

**Link**: [PDF](https://arxiv.org/pdf/2504.13908)  

**Abstract**: Standardized surveys scale efficiently but sacrifice depth, while conversational interviews improve response quality at the cost of scalability and consistency. This study bridges the gap between these methods by introducing a framework for AI-assisted conversational interviewing. To evaluate this framework, we conducted a web survey experiment where 1,800 participants were randomly assigned to text-based conversational AI agents, or "textbots", to dynamically probe respondents for elaboration and interactively code open-ended responses. We assessed textbot performance in terms of coding accuracy, response quality, and respondent experience. Our findings reveal that textbots perform moderately well in live coding even without survey-specific fine-tuning, despite slightly inflated false positive errors due to respondent acquiescence bias. Open-ended responses were more detailed and informative, but this came at a slight cost to respondent experience. Our findings highlight the feasibility of using AI methods to enhance open-ended data collection in web surveys. 

**Abstract (ZH)**: 标准化调查可以高效扩展但牺牲深度，而对话式访谈可以提高响应质量但牺牲扩展性和一致性。本研究通过引入AI辅助对话式访谈框架弥合了这两种方法之间的差距。为评估该框架，我们进行了一项网络调查实验，随机将1,800名参与者分配给基于文本的对话式AI代理，即“文本机器人”，以动态探询受访者并互动编码开放式回答。我们从编码准确性、响应质量和受访者体验三个方面评估了文本机器人的表现。研究结果显示，即使没有特定调查的微调，文本机器人在实时编码中表现适度良好，但由于受访者 acquiescence 偏差导致的轻微虚假积极错误有所增加。开放式回答更加详尽和信息丰富，但这对受访者的体验造成了一定的影响。我们的研究结果表明，使用AI方法增强网络调查中的开放式数据收集具有可行性。 

---
# Generative Framework for Personalized Persuasion: Inferring Causal, Counterfactual, and Latent Knowledge 

**Title (ZH)**: 个性化说服的生成框架：因果推理、反事实推理和潜在知识inferencing 

**Authors**: Donghuo Zeng, Roberto Legaspi, Yuewen Sun, Xinshuai Dong, Kazushi Ikeda, Peter Spirtes, Kun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13904)  

**Abstract**: We hypothesize that optimal system responses emerge from adaptive strategies grounded in causal and counterfactual knowledge. Counterfactual inference allows us to create hypothetical scenarios to examine the effects of alternative system responses. We enhance this process through causal discovery, which identifies the strategies informed by the underlying causal structure that govern system behaviors. Moreover, we consider the psychological constructs and unobservable noises that might be influencing user-system interactions as latent factors. We show that these factors can be effectively estimated. We employ causal discovery to identify strategy-level causal relationships among user and system utterances, guiding the generation of personalized counterfactual dialogues. We model the user utterance strategies as causal factors, enabling system strategies to be treated as counterfactual actions. Furthermore, we optimize policies for selecting system responses based on counterfactual data. Our results using a real-world dataset on social good demonstrate significant improvements in persuasive system outcomes, with increased cumulative rewards validating the efficacy of causal discovery in guiding personalized counterfactual inference and optimizing dialogue policies for a persuasive dialogue system. 

**Abstract (ZH)**: 我们假设最优系统响应源自基于因果和反事实知识的适应性策略。反事实推理允许我们创建假设情景以检验替代系统响应的影响。我们通过因果发现这一过程加以增强，该过程识别出受底层因果结构指导的策略，以规范系统行为。此外，我们考虑可能影响用户-系统交互的心理构念和不可观测噪声作为潜在因素。我们证明了这些因素可以有效地进行估计。我们利用因果发现识别用户和系统声明中的策略级因果关系，指导个性化反事实对话的生成。我们将用户声明策略建模为因果因素，使系统策略能够被视为反事实行动。此外，我们基于反事实数据优化选择系统响应的策略。使用关于社会公益的现实世界数据集的实验结果表明，在具有增加累积奖励的情况下，因果发现能够显著改善说服系统的效果，并验证了其在引导个性化反事实推断和优化具有说服力对话系统对话策略方面的有效性。 

---
# Supporting Students' Reading and Cognition with AI 

**Title (ZH)**: 使用AI支持学生的阅读与认知 

**Authors**: Yue Fu, Alexis Hiniker  

**Link**: [PDF](https://arxiv.org/pdf/2504.13900)  

**Abstract**: With the rapid adoption of AI tools in learning contexts, it is vital to understand how these systems shape users' reading processes and cognitive engagement. We collected and analyzed text from 124 sessions with AI tools, in which students used these tools to support them as they read assigned readings for an undergraduate course. We categorized participants' prompts to AI according to Bloom's Taxonomy of educational objectives -- Remembering, Understanding, Applying, Analyzing, Evaluating. Our results show that ``Analyzing'' and ``Evaluating'' are more prevalent in users' second and third prompts within a single usage session, suggesting a shift toward higher-order thinking. However, in reviewing users' engagement with AI tools over several weeks, we found that users converge toward passive reading engagement over time. Based on these results, we propose design implications for future AI reading-support systems, including structured scaffolds for lower-level cognitive tasks (e.g., recalling terms) and proactive prompts that encourage higher-order thinking (e.g., analyzing, applying, evaluating). Additionally, we advocate for adaptive, human-in-the-loop features that allow students and instructors to tailor their reading experiences with AI, balancing efficiency with enriched cognitive engagement. Our paper expands the dialogue on integrating AI into academic reading, highlighting both its potential benefits and challenges. 

**Abstract (ZH)**: 随着AI工具在学习情境中的迅速采用，了解这些系统如何塑造用户的阅读过程和认知参与变得至关重要。我们收集并分析了124个使用AI工具的会话文本，这些学生在这些工具的支持下阅读了一门本科课程的指定读物。我们将用户对AI的提示按照布卢姆教育目标分类学进行分类——记忆、理解、应用、分析、评价。结果显示，“分析”和“评价”类型的提示在单次使用会话中的第二和第三个提示中更为常见，这表明了一种向高层次思考的转变。然而，在审查用户在数周内与AI工具的互动时，我们发现用户逐渐转向了被动的阅读参与。基于这些结果，我们提出了未来AI阅读支持系统的设想，包括为低层次认知任务提供结构化的支架（例如，回忆术语）以及促进高层次思考的主动提示（例如，分析、应用、评价）。此外，我们提倡具备适应性和人类在环功能的设计，让学生和教师能够根据需要调整他们的阅读体验，平衡效率与丰富的认知参与。我们的论文扩展了将AI整合到学术阅读中的对话，强调了其潜在的利弊。 

---
# Predicting Satisfaction of Counterfactual Explanations from Human Ratings of Explanatory Qualities 

**Title (ZH)**: 从解释质量的人类评价预测反事实解释的满意度 

**Authors**: Marharyta Domnich, Rasmus Moorits Veski, Julius Välja, Kadi Tulver, Raul Vicente  

**Link**: [PDF](https://arxiv.org/pdf/2504.13899)  

**Abstract**: Counterfactual explanations are a widely used approach in Explainable AI, offering actionable insights into decision-making by illustrating how small changes to input data can lead to different outcomes. Despite their importance, evaluating the quality of counterfactual explanations remains an open problem. Traditional quantitative metrics, such as sparsity or proximity, fail to fully account for human preferences in explanations, while user studies are insightful but not scalable. Moreover, relying only on a single overall satisfaction rating does not lead to a nuanced understanding of why certain explanations are effective or not. To address this, we analyze a dataset of counterfactual explanations that were evaluated by 206 human participants, who rated not only overall satisfaction but also seven explanatory criteria: feasibility, coherence, complexity, understandability, completeness, fairness, and trust. Modeling overall satisfaction as a function of these criteria, we find that feasibility (the actionability of suggested changes) and trust (the belief that the changes would lead to the desired outcome) consistently stand out as the strongest predictors of user satisfaction, though completeness also emerges as a meaningful contributor. Crucially, even excluding feasibility and trust, other metrics explain 58% of the variance, highlighting the importance of additional explanatory qualities. Complexity appears independent, suggesting more detailed explanations do not necessarily reduce satisfaction. Strong metric correlations imply a latent structure in how users judge quality, and demographic background significantly shapes ranking patterns. These insights inform the design of counterfactual algorithms that adapt explanatory qualities to user expertise and domain context. 

**Abstract (ZH)**: 基于行为的反事实解释质量评估：用户满意度与解释质量准则分析 

---
# Maestoso: An Intelligent Educational Sketching Tool for Learning Music Theory 

**Title (ZH)**: Maestoso: 一种智能音乐理论绘图学习工具 

**Authors**: Paul Taele, Laura Barreto, Tracy Hammond  

**Link**: [PDF](https://arxiv.org/pdf/2504.13889)  

**Abstract**: Learning music theory not only has practical benefits for musicians to write, perform, understand, and express music better, but also for both non-musicians to improve critical thinking, math analytical skills, and music appreciation. However, current external tools applicable for learning music theory through writing when human instruction is unavailable are either limited in feedback, lacking a written modality, or assuming already strong familiarity of music theory concepts. In this paper, we describe Maestoso, an educational tool for novice learners to learn music theory through sketching practice of quizzed music structures. Maestoso first automatically recognizes students' sketched input of quizzed concepts, then relies on existing sketch and gesture recognition techniques to automatically recognize the input, and finally generates instructor-emulated feedback. From our evaluations, we demonstrate that Maestoso performs reasonably well on recognizing music structure elements and that novice students can comfortably grasp introductory music theory in a single session. 

**Abstract (ZH)**: 学习音乐理论不仅对音乐家提高作曲、表演、理解与表现音乐的技能具有实际益处，也对非音乐家提高批判性思维、数学分析能力和音乐鉴赏能力有益。然而，当前可用于在缺乏人类指导的情况下通过写作学习音乐理论的外部工具要么反馈有限，要么缺少书面表达模式，要么假定使用者对音乐理论概念已有较强的熟悉度。本文介绍了Maestoso，这是一种面向初学者的教育工具，通过练习测验过的音乐结构草图来学习音乐理论。Maestoso 首先自动识别学生草绘的测验概念输入，然后依赖现有的草图和手势识别技术自动识别输入，并最终生成类似教师的反馈。从我们的评估中可以看出，Maestoso 在识别音乐结构元素方面表现合理，且初学者可以在单次会话中舒适地掌握初步的音乐理论知识。 

---
# Kanji Workbook: A Writing-Based Intelligent Tutoring System for Learning Proper Japanese Kanji Writing Technique with Instructor-Emulated Assessment 

**Title (ZH)**: kanji 工作坊：一种基于书写的人工助手智能辅导系统，用于学习正确的日语汉字书写技巧并模拟教师评估 

**Authors**: Paul Taele, Jung In Koh, Tracy Hammond  

**Link**: [PDF](https://arxiv.org/pdf/2504.13888)  

**Abstract**: Kanji script writing is a skill that is often introduced to novice Japanese foreign language students for achieving Japanese writing mastery, but often poses difficulties to students with primarily English fluency due to their its vast differences with written English. Instructors often introduce various pedagogical methods -- such as visual structure and written techniques -- to assist students in kanji study, but may lack availability providing direct feedback on students' writing outside of class. Current educational applications are also limited due to lacking richer instructor-emulated feedback. We introduce Kanji Workbook, a writing-based intelligent tutoring system for students to receive intelligent assessment that emulates human instructor feedback. Our interface not only leverages students' computing devices for allowing them to learn, practice, and review the writing of prompted characters from their course's kanji script lessons, but also provides a diverse set of writing assessment metrics -- derived from instructor interviews and classroom observation insights -- through intelligent scoring and visual animations. We deployed our interface onto novice- and intermediate-level university courses over an entire academic year, and observed that interface users on average achieved higher course grades than their peers and also reacted positively to our interface's various features. 

**Abstract (ZH)**: 日文漢字书写是一种常常被外语日语初学者用于掌握日文书写的技能，但由于它与英文书写的巨大差异，往往给以英语为主导语言的学生带来困难。教师常常采用各种教学方法——如视觉结构和书写技术——来帮助学生学习汉字，但由于缺乏课后直接反馈，这些方法可能效果有限。当前的教育应用也受限于缺乏更丰富的人工模拟反馈。我们介绍了一款名为“汉字工作簿”的基于书写的人工智能辅导系统，以帮助学生获得模拟教师反馈的智能评估。我们的界面不仅利用学生的计算设备，让学生能够学习、练习和复习课程中所学的汉字书写，还提供了一组多样化的书写评估指标——这些指标是从教师访谈和课堂观察中提取的，通过智能评分和可视化动画实现。我们在整个学年中将该界面部署到了初学者和中级水平的大学课程中，并观察到使用该界面的学生平均获得了更高的课程成绩，同时也对该界面的各种功能表现出积极的反应。 

---
# New care pathways for supporting transitional care from hospitals to home using AI and personalized digital assistance 

**Title (ZH)**: 利用AI和个人化数字辅助支持从医院向家庭过渡护理的新路径 

**Authors**: Ionut Anghel, Tudor Cioara, Roberta Bevilacqua, Federico Barbarossa, Terje Grimstad, Riitta Hellman, Arnor Solberg, Lars Thomas Boye, Ovidiu Anchidin, Ancuta Nemes, Camilla Gabrielsen  

**Link**: [PDF](https://arxiv.org/pdf/2504.13877)  

**Abstract**: Transitional care may play a vital role for the sustainability of Europe future healthcare system, offering solutions for relocating patient care from hospital to home therefore addressing the growing demand for medical care as the population is ageing. However, to be effective, it is essential to integrate innovative Information and Communications Technology technologies to ensure that patients with comorbidities experience a smooth and coordinated transition from hospitals or care centers to home, thereby reducing the risk of rehospitalization. In this paper, we present an overview of the integration of Internet of Things, artificial intelligence, and digital assistance technologies with traditional care pathways to address the challenges and needs of healthcare systems in Europe. We identify the current gaps in transitional care and define the technology mapping to enhance the care pathways, aiming to improve patient outcomes, safety, and quality of life avoiding hospital readmissions. Finally, we define the trial setup and evaluation methodology needed to provide clinical evidence that supports the positive impact of technology integration on patient care and discuss the potential effects on the healthcare system. 

**Abstract (ZH)**: 过渡期护理可能在欧洲未来医疗保健系统可持续性中发挥关键作用，通过将患者护理从医院转移到家庭，从而应对人口老龄化带来的日益增长的医疗服务需求。然而，为了有效实施，必须整合创新的信息化和通信技术，以确保共病患者能够顺利且协调地从医院或护理中心转移到家中，从而降低再次入院的风险。本文概述了将物联网、人工智能和数字辅助技术与传统护理路径结合以应对欧洲医疗保健系统面临的挑战和需求。我们确定了过渡期护理中的现有差距，并制定了技术规划以增强护理路径，旨在提高患者结果、安全性和生活质量，避免重新入院。最后，我们定义了试验设计和评估方法，以提供支持技术整合对患者护理产生积极影响的临床证据，并讨论了对医疗保健系统潜在影响。 

---
# Using Generative AI Personas Increases Collective Diversity in Human Ideation 

**Title (ZH)**: 使用生成式AI人格增加集体 ideation 多样性 

**Authors**: Yun Wan, Yoram M Kalman  

**Link**: [PDF](https://arxiv.org/pdf/2504.13868)  

**Abstract**: This study challenges the widely-reported tradeoff between generative AI's (GenAI) contribution to creative outcomes and decreased diversity of these outcomes. We modified the design of such a study, by Doshi and Hauser (2024), in which participants wrote short stories either aided or unaided by GenAI plot ideas[1]. In the modified study, plot ideas were generated through ten unique GenAI "personas" with diverse traits (e.g. cultural backgrounds, thinking styles, genre preferences), creating a pool of 300 story plots. While plot ideas from any individual persona showed high similarity (average cosine similarity of 0.92), ideas across different personas exhibited substantial variation (average similarity of 0.20). When human participants wrote stories based on these diverse plot ideas, their collective outputs maintained the same level of diversity as stories written without GenAI assistance, effectively eliminating the diversity reduction observed in [1]. Traditional text analytics further revealed that GenAI-assisted stories featured greater diversity in descriptive and emotional language compared to purely human-generated stories without GenAI assistance. Our findings demonstrate that introducing diversity at the AI input stage through distinct personas can preserve and potentially enhance the collective diversity of human creative outputs when collaborating with GenAI. 

**Abstract (ZH)**: 本研究挑战了广泛报道的生成式AI（GenAI）对创造性成果贡献与这些成果多样性降低之间的权衡关系。通过修改Doshi和Hauser（2024）的研究设计，参与者使用或未使用GenAI情节创意来撰写短篇故事。在修改后的研究中，通过十种具有不同特质的独特GenAI“角色”生成情节创意，创建了300个故事梗概池。虽然任何单一角色的情节创意显示出高相似性（平均余弦相似度为0.92），但不同角色之间的情节创意展现了显著的差异性（平均相似度为0.20）。当人类参与者基于这些多样性的情节创意撰写故事时，集体产出与未使用GenAI辅助撰写的故事多样性相当，有效消除了[Doshi和Hauser的研究]中观察到的多样性减少现象。传统文本分析进一步表明，使用GenAI辅助的故事在描述性和情感语言上显示出了比完全由人类生成的故事更多的多样性。我们的研究结果表明，在AI输入阶段通过不同角色引入多样性可以保留并可能增强与GenAI合作时人类创造性产出的集体多样性。 

---
# The Effect of Explainable AI-based Decision Support on Human Task Performance: A Meta-Analysis 

**Title (ZH)**: 基于可解释AI的决策支持对人类任务绩效的影响：一篇元分析 

**Authors**: Felix Haag  

**Link**: [PDF](https://arxiv.org/pdf/2504.13858)  

**Abstract**: The desirable properties of explanations in information systems have fueled the demands for transparency in artificial intelligence (AI) outputs. To address these demands, the field of explainable AI (XAI) has put forth methods that can support human decision-making by explaining AI outputs. However, current empirical works present inconsistent findings on whether such explanations help to improve users' task performance in decision support systems (DSS). In this paper, we conduct a meta-analysis to explore how XAI affects human performance in classification tasks. Our results show an improvement in task performance through XAI-based decision support, though explanations themselves are not the decisive driver for this improvement. The analysis reveals that the studies' risk of bias moderates the effect of explanations in AI, while the explanation type appears to play only a negligible role. Our findings contribute to the human computer interaction field by enhancing the understanding of human-XAI collaboration in DSS. 

**Abstract (ZH)**: 信息系统的解释特性推动了对人工智能输出透明度的需求。为了应对这一需求，可解释人工智能(XAI)领域提出了支持人类决策的方法，通过解释人工智能输出来辅助决策。然而，当前的实证研究在决策支持系统(DSS)中这些解释是否能改善用户任务性能方面结果不一。本文通过元分析探讨XAI如何影响人类在分类任务中的绩效。结果显示，基于XAI的决策支持可以提高任务绩效，但解释本身并不是这种改进的关键驱动因素。分析发现，研究的风险偏倚调节了解释在人工智能中的效果，而解释类型似乎起到了非常次要的作用。我们的研究结果在人机交互领域增强了对人类与XAI协作的理解。 

---
# Towards Balancing Preference and Performance through Adaptive Personalized Explainability 

**Title (ZH)**: 通过自适应个性化可解释性实现偏好与性能的平衡 

**Authors**: Andrew Silva, Pradyumna Tambwekar, Mariah Schrum, Matthew Gombolay  

**Link**: [PDF](https://arxiv.org/pdf/2504.13856)  

**Abstract**: As robots and digital assistants are deployed in the real world, these agents must be able to communicate their decision-making criteria to build trust, improve human-robot teaming, and enable collaboration. While the field of explainable artificial intelligence (xAI) has made great strides to enable such communication, these advances often assume that one xAI approach is ideally suited to each problem (e.g., decision trees to explain how to triage patients in an emergency or feature-importance maps to explain radiology reports). This fails to recognize that users have diverse experiences or preferences for interaction modalities. In this work, we present two user-studies set in a simulated autonomous vehicle (AV) domain. We investigate (1) population-level preferences for xAI and (2) personalization strategies for providing robot explanations. We find significant differences between xAI modes (language explanations, feature-importance maps, and decision trees) in both preference (p < 0.01) and performance (p < 0.05). We also observe that a participant's preferences do not always align with their performance, motivating our development of an adaptive personalization strategy to balance the two. We show that this strategy yields significant performance gains (p < 0.05), and we conclude with a discussion of our findings and implications for xAI in human-robot interactions. 

**Abstract (ZH)**: 随着机器人和数字助手在现实世界中的应用，这些代理必须能够沟通其决策标准以建立信任、提高人机协同作战能力并促进合作。虽然可解释人工智能（xAI）领域的研究已经取得了显著进展以实现这种沟通，这些进步往往假设每种问题都有一个最理想的方法（例如，使用决策树解释紧急情况下的病人分诊过程，或使用特征重要性图解释放射学报告）。这种方法未能认识到用户在交互方式上存在多样化的经验或偏好。在本研究中，我们在模拟的自动驾驶车辆（AV）领域进行了两项用户研究。我们探讨了（1）关于xAI的整体偏好以及（2）提供机器人解释的个性化策略。我们发现，在偏好（p < 0.01）和性能（p < 0.05）方面，xAI模式（语言解释、特征重要性图和决策树）之间存在显著差异。我们还观察到，参与者的态度与其表现并不总是相符，这促使我们开发了一种适应性的个性化策略来平衡两者。我们展示了这种方法在性能方面取得了显著成效（p < 0.05），并就我们发现的结果及其对人机交互中xAI的含义进行了讨论。 

---
# GenShin:geometry-enhanced structural graph embodies binding pose can better predicting compound-protein interaction affinity 

**Title (ZH)**: GenShin：几何增强的结构性蛋白质图更好地预测化合物-蛋白质相互作用亲和力 

**Authors**: Pingfei Zhu, Chenyang Zhao, Haishi Zhao, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13853)  

**Abstract**: AI-powered drug discovery typically relies on the successful prediction of compound-protein interactions, which are pivotal for the evaluation of designed compound molecules in structure-based drug design and represent a core challenge in the field.
However, accurately predicting compound-protein affinity via regression models usually requires adequate-binding pose, which are derived from costly and complex experimental methods or time-consuming simulations with docking software. In response, we have introduced the GenShin model, which constructs a geometry-enhanced structural graph module that separately extracts additional features from proteins and compounds. Consequently, it attains an accuracy on par with mainstream models in predicting compound-protein affinities, while eliminating the need for adequate-binding pose as input. Our experimental findings demonstrate that the GenShin model vastly outperforms other models that rely on non-input docking conformations, achieving, or in some cases even exceeding, the performance of those requiring adequate-binding pose. Further experiments indicate that our GenShin model is more robust to inadequate-binding pose, affirming its higher suitability for real-world drug discovery scenarios. We hope our work will inspire more endeavors to bridge the gap between AI models and practical drug discovery challenges. 

**Abstract (ZH)**: AI赋能的药物发现通常依赖于成功预测化合物-蛋白质相互作用，这是基于结构的药物设计中评估设计的化合物分子的关键，也是该领域的核心挑战。然而，通过回归模型准确预测化合物-蛋白质亲和力通常需要充分结合的姿态，这些姿态是从昂贵且复杂的实验方法或耗时的对接软件模拟中获得的。为应对这一挑战，我们引入了GenShin模型，该模型构建了一个增强几何结构图模块，分别从蛋白质和化合物中提取额外特征。因此，它在预测化合物-蛋白质亲和力的准确性方面与主流模型相当，同时消除了对充分结合的姿态作为输入的需求。我们的实验结果表明，GenShin模型在依赖非输入对接构象的其他模型中表现远远优于后者，在某些情况下甚至超过了需要充分结合姿态的模型。进一步的实验表明，我们的GenShin模型在不充分结合姿态方面更具鲁棒性，证实了其在实际药物发现场景中的更高适用性。我们希望我们的工作能够激励更多努力缩小AI模型与实际药物发现挑战之间的差距。 

---
# From Interaction to Collaboration: How Hybrid Intelligence Enhances Chatbot Feedback 

**Title (ZH)**: 从交互到协作：混合智能如何增强聊天机器人反馈 

**Authors**: Janet Rafner, Ryan Q. Guloy, Eden W. Wen, Catherine M. Chiodo, Jacob Sherson  

**Link**: [PDF](https://arxiv.org/pdf/2504.13848)  

**Abstract**: Generative AI (GenAI) chatbots are becoming increasingly integrated into virtual assistant technologies, yet their success hinges on the ability to gather meaningful user feedback to improve interaction quality, system outcomes, and overall user acceptance. Successful chatbot interactions can enable organizations to build long-term relationships with their customers and users, supporting customer loyalty and furthering the organization's goals. This study explores the impact of two distinct narratives and feedback collection mechanisms on user engagement and feedback behavior: a standard AI-focused interaction versus a hybrid intelligence (HI) framed interaction. Initial findings indicate that while small-scale survey measures allowed for no significant differences in user willingness to leave feedback, use the system, or trust the system, participants exposed to the HI narrative statistically significantly provided more detailed feedback. These initial findings offer insights into designing effective feedback systems for GenAI virtual assistants, balancing user effort with system improvement potential. 

**Abstract (ZH)**: 生成式人工智能（GenAI）聊天机器人日益融入虚拟助手技术，但其成功取决于收集有意义的用户反馈的能力，以提高交互质量、系统效果和整体用户接受度。成功的聊天机器人交互可以帮助组织与其客户和用户建立长期关系，支持客户忠诚度并实现组织目标。本研究探讨了两种不同叙事和反馈收集机制对用户参与度和反馈行为的影响：标准AI导向的交互与混合智能（HI）框架的交互。初步结果显示，虽然小型调查措施未发现用户愿意留反馈、使用系统或信任系统的显著差异，但接受HI叙事的参与者在提供详细反馈方面显著更为活跃。这些初步结果为设计有效的GenAI虚拟助手反馈系统提供了见解，平衡了用户努力与系统改进潜力。 

---
