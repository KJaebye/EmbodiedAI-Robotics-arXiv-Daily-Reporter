# Adapt Your Body: Mitigating Proprioception Shifts in Imitation Learning 

**Title (ZH)**: 适应你的身体：减轻 imitation 学习中的本体感觉移位 

**Authors**: Fuhang Kuang, Jiacheng You, Yingdong Hu, Tong Zhang, Chuan Wen, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.23944)  

**Abstract**: Imitation learning models for robotic tasks typically rely on multi-modal inputs, such as RGB images, language, and proprioceptive states. While proprioception is intuitively important for decision-making and obstacle avoidance, simply incorporating all proprioceptive states leads to a surprising degradation in imitation learning performance. In this work, we identify the underlying issue as the proprioception shift problem, where the distributions of proprioceptive states diverge significantly between training and deployment. To address this challenge, we propose a domain adaptation framework that bridges the gap by utilizing rollout data collected during deployment. Using Wasserstein distance, we quantify the discrepancy between expert and rollout proprioceptive states and minimize this gap by adding noise to both sets of states, proportional to the Wasserstein distance. This strategy enhances robustness against proprioception shifts by aligning the training and deployment distributions. Experiments on robotic manipulation tasks demonstrate the efficacy of our method, enabling the imitation policy to leverage proprioception while mitigating its adverse effects. Our approach outperforms the naive solution which discards proprioception, and other baselines designed to address distributional shifts. 

**Abstract (ZH)**: 基于 proprioception 变迁问题的机器人任务模仿学习模型 

---
# World4Omni: A Zero-Shot Framework from Image Generation World Model to Robotic Manipulation 

**Title (ZH)**: World4Omni：从图像生成世界模型到机器人操作的零样本框架 

**Authors**: Haonan Chen, Bangjun Wang, Jingxiang Guo, Tianrui Zhang, Yiwen Hou, Xuchuan Huang, Chenrui Tie, Lin Shao  

**Link**: [PDF](https://arxiv.org/pdf/2506.23919)  

**Abstract**: Improving data efficiency and generalization in robotic manipulation remains a core challenge. We propose a novel framework that leverages a pre-trained multimodal image-generation model as a world model to guide policy learning. By exploiting its rich visual-semantic representations and strong generalization across diverse scenes, the model generates open-ended future state predictions that inform downstream manipulation. Coupled with zero-shot low-level control modules, our approach enables general-purpose robotic manipulation without task-specific training. Experiments in both simulation and real-world environments demonstrate that our method achieves effective performance across a wide range of manipulation tasks with no additional data collection or fine-tuning. Supplementary materials are available on our website: this https URL. 

**Abstract (ZH)**: 提高机器人操作中的数据效率和泛化能力仍然是一个核心挑战。我们提出了一种新的框架，该框架利用预训练的多模态图像生成模型作为世界模型来指导策略学习。通过利用其丰富的视觉语义表示和在多种场景下的强大泛化能力，该模型生成开放式的未来状态预测，以指导后续的操作。结合零样本低级控制模块，我们的方法能够在无需特定任务训练的情况下实现通用的机器人操作。在模拟和真实环境中的实验表明，我们的方法能够跨越一系列操作任务实现有效的性能，无需额外的数据收集或微调。更多资料请参见我们的网站：this https URL。 

---
# Multi-Timescale Hierarchical Reinforcement Learning for Unified Behavior and Control of Autonomous Driving 

**Title (ZH)**: 面向自主驾驶统一行为与控制的多时尺度层次 reinforcement 学习 

**Authors**: Guizhe Jin, Zhuoren Li, Bo Leng, Ran Yu, Lu Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2506.23771)  

**Abstract**: Reinforcement Learning (RL) is increasingly used in autonomous driving (AD) and shows clear advantages. However, most RL-based AD methods overlook policy structure design. An RL policy that only outputs short-timescale vehicle control commands results in fluctuating driving behavior due to fluctuations in network outputs, while one that only outputs long-timescale driving goals cannot achieve unified optimality of driving behavior and control. Therefore, we propose a multi-timescale hierarchical reinforcement learning approach. Our approach adopts a hierarchical policy structure, where high- and low-level RL policies are unified-trained to produce long-timescale motion guidance and short-timescale control commands, respectively. Therein, motion guidance is explicitly represented by hybrid actions to capture multimodal driving behaviors on structured road and support incremental low-level extend-state updates. Additionally, a hierarchical safety mechanism is designed to ensure multi-timescale safety. Evaluation in simulator-based and HighD dataset-based highway multi-lane scenarios demonstrates that our approach significantly improves AD performance, effectively increasing driving efficiency, action consistency and safety. 

**Abstract (ZH)**: 基于多时间尺度层次强化学习的自主驾驶方法 

---
# Motion Tracking with Muscles: Predictive Control of a Parametric Musculoskeletal Canine Model 

**Title (ZH)**: 基于肌肉的运动跟踪：参量化犬类肌肉骨骼模型的预测控制 

**Authors**: Vittorio La Barbera, Steven Bohez, Leonard Hasenclever, Yuval Tassa, John R. Hutchinson  

**Link**: [PDF](https://arxiv.org/pdf/2506.23768)  

**Abstract**: We introduce a novel musculoskeletal model of a dog, procedurally generated from accurate 3D muscle meshes. Accompanying this model is a motion capture-based locomotion task compatible with a variety of control algorithms, as well as an improved muscle dynamics model designed to enhance convergence in differentiable control frameworks. We validate our approach by comparing simulated muscle activation patterns with experimentally obtained electromyography (EMG) data from previous canine locomotion studies. This work aims to bridge gaps between biomechanics, robotics, and computational neuroscience, offering a robust platform for researchers investigating muscle actuation and neuromuscular this http URL plan to release the full model along with the retargeted motion capture clips to facilitate further research and development. 

**Abstract (ZH)**: 一种基于准确3D肌肉网格 procedurally 生成的狗的运动学模型及其配套的运动捕捉运动任务和改进的肌肉动力学模型：填补生物力学、机器人学和计算神经科学之间的差距 

---
# PAC Bench: Do Foundation Models Understand Prerequisites for Executing Manipulation Policies? 

**Title (ZH)**: PAC Bench: 基础模型理解执行操纵策略的前提理解吗？ 

**Authors**: Atharva Gundawar, Som Sagar, Ransalu Senanayake  

**Link**: [PDF](https://arxiv.org/pdf/2506.23725)  

**Abstract**: Vision-Language Models (VLMs) are increasingly pivotal for generalist robot manipulation, enabling tasks such as physical reasoning, policy generation, and failure detection. However, their proficiency in these high-level applications often assumes a deep understanding of low-level physical prerequisites, a capability that remains largely unverified. For robots to perform actions reliably, they must comprehend intrinsic object properties (e.g., material, weight), action affordances (e.g., graspable, stackable), and physical constraints (e.g., stability, reachability, or an object's state, such as being closed). Despite the widespread use of VLMs in manipulation tasks, we argue that off-the-shelf models may lack this granular, physically grounded understanding, as such prerequisites are often overlooked during training.
To address this critical gap, we introduce PAC Bench, a comprehensive benchmark designed to systematically evaluate VLMs on their understanding of core Properties, Affordances, and Constraints (PAC) from a task executability perspective. PAC Bench features a diverse dataset with over 30,000 annotations, comprising 673 real-world images (115 object classes, 15 property types, and 1 to 3 affordances defined per class), 100 real-world humanoid-view scenarios, and 120 unique simulated constraint scenarios across four tasks.
Our evaluations reveal significant gaps in the ability of current VLMs to grasp fundamental physical concepts, highlighting limitations in their suitability for reliable robot manipulation and pointing to key areas for targeted research. PAC Bench also serves as a standardized benchmark for rigorously evaluating physical reasoning in VLMs and guiding the development of more robust, physically grounded models for robotic applications.
Project Page: this https URL 

**Abstract (ZH)**: 视觉-语言模型（VLMs）在通用机器人操作中的作用日益重要，能够实现物理推理、策略生成和故障检测等任务。然而，它们在这些高层次应用中的熟练程度往往假设了对低层次物理前提的深刻理解，而这种能力尚未得到充分验证。为了使机器人能够可靠地执行动作，它们必须理解对象的内在属性（例如，材料、重量）、动作可能性（例如，可抓取性、可堆叠性）以及物理约束（例如，稳定性、可接近性或物体状态，如关闭状态）。尽管视觉-语言模型在操作任务中被广泛应用，我们认为，现成的模型可能缺乏这一精细的物理相关理解，因为这些前提条件在训练过程中常常被忽视。

为了解决这一关键问题，我们提出了PAC Bench，这是一个全面的基准测试工具，旨在从任务可执行性的角度系统评估视觉-语言模型对核心属性、可能性和约束（PAC）的理解能力。PAC Bench 包含一个多样化的数据集，共有超过30,000个注释，包括673张真实世界图片（115个物体类别、15种属性类型，每类1至3种可能性定义）、100个真实世界的人形视角场景和四个任务中涉及的120种独特的模拟约束场景。

我们的评估揭示了当前视觉-语言模型在掌握基本物理概念方面存在显著差距，强调了它们在可靠机器人操作中的适用限制，并指出了需要重点研究的关键领域。PAC Bench 还为严格评估视觉-语言模型中的物理推理提供了一个标准化基准，并指导了更为稳健和物理相关模型的开发，以应用于机器人应用。 

---
# Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop 

**Title (ZH)**: CVPR 2025 MEIS研讨会中RoboTwin双臂协作挑战赛对可泛化双臂操作的基准测试 

**Authors**: Tianxing Chen, Kaixuan Wang, Zhaohui Yang, Yuhao Zhang, Zanxin Chen, Baijun Chen, Wanxi Dong, Ziyuan Liu, Dong Chen, Tianshuo Yang, Haibao Yu, Xiaokang Yang, Yusen Qin, Zhiqiang Xie, Yao Mu, Ping Luo, Tian Nian, Weiliang Deng, Yiheng Ge, Yibin Liu, Zixuan Li, Dehui Wang, Zhixuan Liang, Haohui Xie, Rijie Zeng, Yunfei Ge, Peiqing Cong, Guannan He, Zhaoming Han, Ruocheng Yin, Jingxiang Guo, Lunkai Lin, Tianling Xu, Hongzhe Bi, Xuewu Lin, Tianwei Lin, Shujie Luo, Keyu Li, Ziyan Zhao, Ke Fan, Heyang Xu, Bo Peng, Wenlong Gao, Dongjiang Li, Feng Jin, Hui Shen, Jinming Li, Chaowei Cui, Yuchen, Yaxin Peng, Lingdong Zeng, Wenlong Dong, Tengfei Li, Weijie Ke, Jun Chen, Erdemt Bao, Tian Lan, Tenglong Liu, Jin Yang, Huiping Zhuang, Baozhi Jia, Shuai Zhang, Zhengfeng Zou, Fangheng Guan, Tianyi Jia, Ke Zhou, Hongjiu Zhang, Yating Han, Cheng Fang, Yixian Zou, Chongyang Xu, Qinglun Zhang, Shen Cheng, Xiaohe Wang, Ping Tan, Haoqiang Fan, Shuaicheng Liu, Jiaheng Chen, Chuxuan Huang, Chengliang Lin, Kaijun Luo, Boyu Yue, Yi Liu, Jinyu Chen, Zichang Tan, Liming Deng, Shuo Xu, Zijian Cai, Shilong Yin, Hao Wang, Hongshan Liu, Tianyang Li, Long Shi, Ran Xu, Huilin Xu, Zhengquan Zhang, Congsheng Xu, Jinchang Yang, Feng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23351)  

**Abstract**: Embodied Artificial Intelligence (Embodied AI) is an emerging frontier in robotics, driven by the need for autonomous systems that can perceive, reason, and act in complex physical environments. While single-arm systems have shown strong task performance, collaborative dual-arm systems are essential for handling more intricate tasks involving rigid, deformable, and tactile-sensitive objects. To advance this goal, we launched the RoboTwin Dual-Arm Collaboration Challenge at the 2nd MEIS Workshop, CVPR 2025. Built on the RoboTwin Simulation platform (1.0 and 2.0) and the AgileX COBOT-Magic Robot platform, the competition consisted of three stages: Simulation Round 1, Simulation Round 2, and a final Real-World Round. Participants totally tackled 17 dual-arm manipulation tasks, covering rigid, deformable, and tactile-based scenarios. The challenge attracted 64 global teams and over 400 participants, producing top-performing solutions like SEM and AnchorDP3 and generating valuable insights into generalizable bimanual policy learning. This report outlines the competition setup, task design, evaluation methodology, key findings and future direction, aiming to support future research on robust and generalizable bimanual manipulation policies. The Challenge Webpage is available at this https URL. 

**Abstract (ZH)**: embodied人工智能(Embodied Artificial Intelligence)是机器人领域的新兴前沿，驱动这一领域发展的是对于能够在复杂物理环境中感知、推理和行动的自主系统的需要。尽管单臂系统在任务执行中表现出强大的性能，但对于涉及刚性、可变形和触觉敏感物体的复杂任务，协作双臂系统是必不可少的。为了进一步推动这一目标，我们在2025年CVPR的第2届MEIS研讨会上发起了RoboTwin双臂协作挑战赛。基于RoboTwin模拟平台（版本1.0和2.0）和AgileX COBOT-Magic机器人平台，该挑战赛分为三个阶段：模拟首轮、模拟次轮和最终的现实世界轮。参与者共计完成了17项双臂操作任务，涵盖刚性、可变形和基于触觉的场景。此次挑战吸引了来自全球的64支团队和超过400名参与者，产生了如SEM和AnchorDP3等顶级解决方案，并为双臂控制策略的可泛化学习提供了宝贵见解。本报告概述了挑战赛的设置、任务设计、评估方法、关键发现及未来方向，旨在支持未来关于稳健且可泛化的双臂操作策略的研究。挑战网页地址为：this https URL。 

---
# Simplifying Data-Driven Modeling of the Volume-Flow-Pressure Relationship in Hydraulic Soft Robotic Actuators 

**Title (ZH)**: 基于液压软体驱动器中的体积-流量-压力关系的数据驱动建模简化方法 

**Authors**: Sang-Yoep Lee, Leonardo Zamora Yanez, Jacob Rogatinsky, Vi T. Vo, Tanvi Shingade, Tommaso Ranzani  

**Link**: [PDF](https://arxiv.org/pdf/2506.23326)  

**Abstract**: Soft robotic systems are known for their flexibility and adaptability, but traditional physics-based models struggle to capture their complex, nonlinear behaviors. This study explores a data-driven approach to modeling the volume-flow-pressure relationship in hydraulic soft actuators, focusing on low-complexity models with high accuracy. We perform regression analysis on a stacked balloon actuator system using exponential, polynomial, and neural network models with or without autoregressive inputs. The results demonstrate that simpler models, particularly multivariate polynomials, effectively predict pressure dynamics with fewer parameters. This research offers a practical solution for real-time soft robotics applications, balancing model complexity and computational efficiency. Moreover, the approach may benefit various techniques that require explicit analytical models. 

**Abstract (ZH)**: 数据驱动方法在液压软执行器的体积-流量-压力关系建模中的应用：关注低复杂度高精度模型 

---
# ParticleFormer: A 3D Point Cloud World Model for Multi-Object, Multi-Material Robotic Manipulation 

**Title (ZH)**: ParticleFormer: 多对象、多材料机器人 manipulation 的三维点云世界模型 

**Authors**: Suning Huang, Qianzhong Chen, Xiaohan Zhang, Jiankai Sun, Mac Schwager  

**Link**: [PDF](https://arxiv.org/pdf/2506.23126)  

**Abstract**: 3D world models (i.e., learning-based 3D dynamics models) offer a promising approach to generalizable robotic manipulation by capturing the underlying physics of environment evolution conditioned on robot actions. However, existing 3D world models are primarily limited to single-material dynamics using a particle-based Graph Neural Network model, and often require time-consuming 3D scene reconstruction to obtain 3D particle tracks for training. In this work, we present ParticleFormer, a Transformer-based point cloud world model trained with a hybrid point cloud reconstruction loss, supervising both global and local dynamics features in multi-material, multi-object robot interactions. ParticleFormer captures fine-grained multi-object interactions between rigid, deformable, and flexible materials, trained directly from real-world robot perception data without an elaborate scene reconstruction. We demonstrate the model's effectiveness both in 3D scene forecasting tasks, and in downstream manipulation tasks using a Model Predictive Control (MPC) policy. In addition, we extend existing dynamics learning benchmarks to include diverse multi-material, multi-object interaction scenarios. We validate our method on six simulation and three real-world experiments, where it consistently outperforms leading baselines by achieving superior dynamics prediction accuracy and less rollout error in downstream visuomotor tasks. Experimental videos are available at this https URL. 

**Abstract (ZH)**: 基于Transformer的粒子点云世界模型：ParticleFormer及其在多材料多物体机器人交互中的应用 

---
# Learning Motion Skills with Adaptive Assistive Curriculum Force in Humanoid Robots 

**Title (ZH)**: 基于自适应辅助课程力的学习类人机器人运动技能 

**Authors**: Zhanxiang Cao, Yang Zhang, Buqing Nie, Huangxuan Lin, Haoyang Li, Yue Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.23125)  

**Abstract**: Learning policies for complex humanoid tasks remains both challenging and compelling. Inspired by how infants and athletes rely on external support--such as parental walkers or coach-applied guidance--to acquire skills like walking, dancing, and performing acrobatic flips, we propose A2CF: Adaptive Assistive Curriculum Force for humanoid motion learning. A2CF trains a dual-agent system, in which a dedicated assistive force agent applies state-dependent forces to guide the robot through difficult initial motions and gradually reduces assistance as the robot's proficiency improves. Across three benchmarks--bipedal walking, choreographed dancing, and backflip--A2CF achieves convergence 30% faster than baseline methods, lowers failure rates by over 40%, and ultimately produces robust, support-free policies. Real-world experiments further demonstrate that adaptively applied assistive forces significantly accelerate the acquisition of complex skills in high-dimensional robotic control. 

**Abstract (ZH)**: 适应性辅助 Curriculum 力量学习复杂人形任务策略 

---
# Minimizing Acoustic Noise: Enhancing Quiet Locomotion for Quadruped Robots in Indoor Applications 

**Title (ZH)**: 最小化噪声振动：提高室内应用中四足机器人静音运动的表现 

**Authors**: Zhanxiang Cao, Buqing Nie, Yang Zhang, Yue Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.23114)  

**Abstract**: Recent advancements in quadruped robot research have significantly improved their ability to traverse complex and unstructured outdoor environments. However, the issue of noise generated during locomotion is generally overlooked, which is critically important in noise-sensitive indoor environments, such as service and healthcare settings, where maintaining low noise levels is essential. This study aims to optimize the acoustic noise generated by quadruped robots during locomotion through the development of advanced motion control algorithms. To achieve this, we propose a novel approach that minimizes noise emissions by integrating optimized gait design with tailored control strategies. This method achieves an average noise reduction of approximately 8 dBA during movement, thereby enhancing the suitability of quadruped robots for deployment in noise-sensitive indoor environments. Experimental results demonstrate the effectiveness of this approach across various indoor settings, highlighting the potential of quadruped robots for quiet operation in noise-sensitive environments. 

**Abstract (ZH)**: 最近四足机器人研究的进展显著提高了它们在复杂和未结构化的户外环境中的通过能力。然而，运动过程中产生的噪音问题通常被忽视，这对于敏感性室内环境（如服务和医疗保健场所）尤其重要，在这些环境中保持低噪音水平至关重要。本研究旨在通过开发先进的运动控制算法来优化四足机器人在运动过程中产生的声学噪音。为此，我们提出了一种新颖的方法，通过将优化的步态设计与定制的控制策略相结合来最小化噪音排放。这种方法在运动过程中平均减少了约8 dBA的噪音，从而提高了四足机器人在敏感性室内环境中的适用性。实验结果证明了该方法在各种室内环境中的有效性，突显了四足机器人在敏感性噪音环境中的安静运行潜力。 

---
# Scenario-Based Hierarchical Reinforcement Learning for Automated Driving Decision Making 

**Title (ZH)**: 基于场景的分层强化学习在自动驾驶决策中的应用 

**Authors**: M. Youssef Abdelhamid, Lennart Vater, Zlatan Ajanovic  

**Link**: [PDF](https://arxiv.org/pdf/2506.23023)  

**Abstract**: Developing decision-making algorithms for highly automated driving systems remains challenging, since these systems have to operate safely in an open and complex environments. Reinforcement Learning (RL) approaches can learn comprehensive decision policies directly from experience and already show promising results in simple driving tasks. However, current approaches fail to achieve generalizability for more complex driving tasks and lack learning efficiency. Therefore, we present Scenario-based Automated Driving Reinforcement Learning (SAD-RL), the first framework that integrates Reinforcement Learning (RL) of hierarchical policy in a scenario-based environment. A high-level policy selects maneuver templates that are evaluated and executed by a low-level control logic. The scenario-based environment allows to control the training experience for the agent and to explicitly introduce challenging, but rate situations into the training process. Our experiments show that an agent trained using the SAD-RL framework can achieve safe behaviour in easy as well as challenging situations efficiently. Our ablation studies confirmed that both HRL and scenario diversity are essential for achieving these results. 

**Abstract (ZH)**: 基于场景的自动驾驶强化学习（SAD-RL）框架：实现复杂驾驶任务的安全高效决策 

---
# Hierarchical Vision-Language Planning for Multi-Step Humanoid Manipulation 

**Title (ZH)**: 层级视觉语言规划用于多步 humanoid 操作规划 

**Authors**: André Schakkal, Ben Zandonati, Zhutian Yang, Navid Azizan  

**Link**: [PDF](https://arxiv.org/pdf/2506.22827)  

**Abstract**: Enabling humanoid robots to reliably execute complex multi-step manipulation tasks is crucial for their effective deployment in industrial and household environments. This paper presents a hierarchical planning and control framework designed to achieve reliable multi-step humanoid manipulation. The proposed system comprises three layers: (1) a low-level RL-based controller responsible for tracking whole-body motion targets; (2) a mid-level set of skill policies trained via imitation learning that produce motion targets for different steps of a task; and (3) a high-level vision-language planning module that determines which skills should be executed and also monitors their completion in real-time using pretrained vision-language models (VLMs). Experimental validation is performed on a Unitree G1 humanoid robot executing a non-prehensile pick-and-place task. Over 40 real-world trials, the hierarchical system achieved a 72.5% success rate in completing the full manipulation sequence. These experiments confirm the feasibility of the proposed hierarchical system, highlighting the benefits of VLM-based skill planning and monitoring for multi-step manipulation scenarios. See this https URL for video demonstrations of the policy rollout. 

**Abstract (ZH)**: 使类人机器人可靠地执行复杂多步操作任务对于其在工业和家庭环境中的有效部署至关重要。本文提出了一种分层规划与控制框架，旨在实现可靠的多步类人操作。所提出的系统包括三层结构：（1）一个基于RL的低层控制器，负责跟踪全身运动目标；（2）一套通过模仿学习训练的中间层技能策略，为任务的不同步骤生成运动目标；以及（3）一个高层的视觉-语言规划模块，确定应执行哪些技能并在实时中使用预训练的视觉-语言模型（VLMs）监控其完成情况。实验验证在Unitree G1类人机器人执行一个非抓取式捡取放置任务中进行。在超过40次真实世界试验中，分层系统在完成整个操作序列方面的成功率达到72.5%。这些实验证明了所提出的分层系统的可行性，突显了基于VLM的技能规划和监控在多步操作场景中的优势。有关策略展开的视频演示，请参考链接：见此链接。 

---
# Unsupervised Discovery of Behavioral Primitives from Sensorimotor Dynamic Functional Connectivity 

**Title (ZH)**: 无监督发现传感运动动态功能连接中的行为 primitives 

**Authors**: Fernando Diaz Ledezma, Valentin Marcel, Matej Hoffmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.22473)  

**Abstract**: The movements of both animals and robots give rise to streams of high-dimensional motor and sensory information. Imagine the brain of a newborn or the controller of a baby humanoid robot trying to make sense of unprocessed sensorimotor time series. Here, we present a framework for studying the dynamic functional connectivity between the multimodal sensory signals of a robotic agent to uncover an underlying structure. Using instantaneous mutual information, we capture the time-varying functional connectivity (FC) between proprioceptive, tactile, and visual signals, revealing the sensorimotor relationships. Using an infinite relational model, we identified sensorimotor modules and their evolving connectivity. To further interpret these dynamic interactions, we employed non-negative matrix factorization, which decomposed the connectivity patterns into additive factors and their corresponding temporal coefficients. These factors can be considered the agent's motion primitives or movement synergies that the agent can use to make sense of its sensorimotor space and later for behavior selection. In the future, the method can be deployed in robot learning as well as in the analysis of human movement trajectories or brain signals. 

**Abstract (ZH)**: 机器人代理的多模态感官信号之间的动态功能连接研究 

---
# Conversations with Andrea: Visitors' Opinions on Android Robots in a Museum 

**Title (ZH)**: Andrea互动交流：访客对博物馆中Android机器人观点的研究 

**Authors**: Marcel Heisler, Christian Becker-Asano  

**Link**: [PDF](https://arxiv.org/pdf/2506.22466)  

**Abstract**: The android robot Andrea was set up at a public museum in Germany for six consecutive days to have conversations with visitors, fully autonomously. No specific context was given, so visitors could state their opinions regarding possible use-cases in structured interviews, without any bias. Additionally the 44 interviewees were asked for their general opinions of the robot, their reasons (not) to interact with it and necessary improvements for future use. The android's voice and wig were changed between different days of operation to give varying cues regarding its gender. This did not have a significant impact on the positive overall perception of the robot. Most visitors want the robot to provide information about exhibits in the future, while opinions on other roles, like a receptionist, were both wanted and explicitly not wanted by different visitors. Speaking more languages (than only English) and faster response times were the improvements most desired. These findings from the interviews are in line with an analysis of the system logs, which revealed, that after chitchat and personal questions, most of the 4436 collected requests asked for information related to the museum and to converse in a different language. The valuable insights gained from these real-world interactions are now used to improve the system to become a useful real-world application. 

**Abstract (ZH)**: 安卓机器人Andrea在德国一家公共博物馆连续六天与访客进行自主对话的实验：访客的意见和反馈研究及系统改进 

---
# A Survey on Vision-Language-Action Models for Autonomous Driving 

**Title (ZH)**: 自动驾驶领域的视觉-语言-行动模型综述 

**Authors**: Sicong Jiang, Zilin Huang, Kangan Qian, Ziang Luo, Tianze Zhu, Yang Zhong, Yihong Tang, Menglin Kong, Yunlong Wang, Siwen Jiao, Hao Ye, Zihao Sheng, Xin Zhao, Tuopu Wen, Zheng Fu, Sikai Chen, Kun Jiang, Diange Yang, Seongjin Choi, Lijun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.24044)  

**Abstract**: The rapid progress of multimodal large language models (MLLM) has paved the way for Vision-Language-Action (VLA) paradigms, which integrate visual perception, natural language understanding, and control within a single policy. Researchers in autonomous driving are actively adapting these methods to the vehicle domain. Such models promise autonomous vehicles that can interpret high-level instructions, reason about complex traffic scenes, and make their own decisions. However, the literature remains fragmented and is rapidly expanding. This survey offers the first comprehensive overview of VLA for Autonomous Driving (VLA4AD). We (i) formalize the architectural building blocks shared across recent work, (ii) trace the evolution from early explainer to reasoning-centric VLA models, and (iii) compare over 20 representative models according to VLA's progress in the autonomous driving domain. We also consolidate existing datasets and benchmarks, highlighting protocols that jointly measure driving safety, accuracy, and explanation quality. Finally, we detail open challenges - robustness, real-time efficiency, and formal verification - and outline future directions of VLA4AD. This survey provides a concise yet complete reference for advancing interpretable socially aligned autonomous vehicles. Github repo is available at \href{this https URL}{SicongJiang/Awesome-VLA4AD}. 

**Abstract (ZH)**: 多模态大型语言模型的快速发展为视觉-语言-行动（VLA）范式铺平了道路，这些范式将视觉感知、自然语言理解和控制集成为一个单一的策略。自动驾驶领域的研究人员正在积极将这些方法应用于车辆领域。此类模型有望实现能够解释高级指令、推理复杂交通场景并自主作出决策的自动驾驶车辆。然而，该领域文献仍然 fragmented 并且发展迅速。本文综述首次全面概述了面向自动驾驶的视觉-语言-行动（VLA4AD）。我们（i）正式化了近期工作中共有的架构构建块，（ii）追溯从早期解释器到以推理为中心的VLA模型的发展过程，（iii）根据VLA在自动驾驶领域的进展比较了20多个代表模型。我们还整合了现有的数据集和基准测试，突出了同时衡量驾驶安全性、准确性和解释质量的协议。最后，我们详细阐述了开放性挑战——鲁棒性、实时效率和形式验证，并概述了VLA4AD的未来方向。本文综述为推进解释性社会对齐的自动驾驶车辆提供了简洁而完整的参考。相关代码库可访问：\href{this https URL}{SicongJiang/Awesome-VLA4AD}。 

---
# StyleDrive: Towards Driving-Style Aware Benchmarking of End-To-End Autonomous Driving 

**Title (ZH)**: StyleDrive：面向驾驶风格的端到端自动驾驶基准测试研究 

**Authors**: Ruiyang Hao, Bowen Jing, Haibao Yu, Zaiqing Nie  

**Link**: [PDF](https://arxiv.org/pdf/2506.23982)  

**Abstract**: While personalization has been explored in traditional autonomous driving systems, it remains largely overlooked in end-to-end autonomous driving (E2EAD), despite its growing prominence. This gap is critical, as user-aligned behavior is essential for trust, comfort, and widespread adoption of autonomous vehicles. A core challenge is the lack of large-scale real-world datasets annotated with diverse and fine-grained driving preferences, hindering the development and evaluation of personalized E2EAD models. In this work, we present the first large-scale real-world dataset enriched with annotations capturing diverse driving preferences, establishing a foundation for personalization in E2EAD. We extract static environmental features from real-world road topology and infer dynamic contextual cues using a fine-tuned visual language model (VLM), enabling consistent and fine-grained scenario construction. Based on these scenarios, we derive objective preference annotations through behavioral distribution analysis and rule-based heuristics. To address the inherent subjectivity of driving style, we further employ the VLM to generate subjective annotations by jointly modeling scene semantics and driver behavior. Final high-quality labels are obtained through a human-in-the-loop verification process that fuses both perspectives. Building on this dataset, we propose the first benchmark for evaluating personalized E2EAD models. We assess several state-of-the-art models with and without preference conditioning, demonstrating that incorporating personalized preferences results in behavior more aligned with human driving. Our work lays the foundation for personalized E2EAD by providing a standardized platform to systematically integrate human preferences into data-driven E2EAD systems, catalyzing future research in human-centric autonomy. 

**Abstract (ZH)**: 尽管个性化在传统自动驾驶系统中已经得到了探索，但在端到端自动驾驶（E2EAD）中仍被很大程度上忽视，尽管其重要性日益凸显。这一差距至关重要，因为与用户需求一致的行为是建立信任、提高舒适度和推动自动驾驶车辆广泛应用的基础。核心挑战在于缺乏包含多样化和细粒度驾驶偏好标注的大规模真实世界数据集，阻碍了个性化E2EAD模型的开发与评估。在这项工作中，我们提出了首个包含多样化驾驶偏好标注的大规模真实世界数据集，为E2EAD中的个性化奠定了基础。我们从现实道路拓扑中提取静态环境特征，并利用微调后的视觉语言模型（VLM）推断动态上下文线索，实现一致且细粒度的场景构建。基于这些场景，我们通过对行为分布分析和基于规则的启发式方法推导出客观的偏好标注。为解决驾驶风格固有的主观性问题，我们进一步利用VLM生成主观标注，通过场景语义与驾驶行为的同时建模来实现。最终高质量的标签通过结合两方面的视角的人工验证过程获得。基于此数据集，我们提出了首个评估个性化E2EAD模型的标准基准。我们评估了几种最先进的模型，有和没有偏好调整的情况，证明了引入个性化偏好使得行为更加符合人类驾驶。我们的工作为个性化E2EAD奠定了基础，提供了一个标准化平台，系统地将人类偏好整合到数据驱动的E2EAD系统中，推动了面向人类的自働性研究的进步。 

---
# Towards foundational LiDAR world models with efficient latent flow matching 

**Title (ZH)**: 面向高效潜在流匹配的基本LiDAR世界模型研究 

**Authors**: Tianran Liu, Shengwen Zhao, Nicholas Rhinehart  

**Link**: [PDF](https://arxiv.org/pdf/2506.23434)  

**Abstract**: LiDAR-based world models offer more structured and geometry-aware representations than their image-based counterparts. However, existing LiDAR world models are narrowly trained; each model excels only in the domain for which it was built. Can we develop LiDAR world models that exhibit strong transferability across multiple domains? We conduct the first systematic domain transfer study across three demanding scenarios: (i) outdoor to indoor generalization, (ii) sparse-beam \& dense-beam adaptation, and (iii) non-semantic to semantic transfer. Given different amounts of fine-tuning data, our experiments show that a single pre-trained model can achieve up to 11% absolute improvement (83\% relative) over training from scratch and outperforms training from scratch in 30/36 of our comparisons. This transferability of dynamic learning significantly reduces the reliance on manually annotated data for semantic occupancy forecasting: our method exceed the previous semantic occupancy forecasting models with only 5% of the labeled training data required by prior models. We also observed inefficiencies of current LiDAR world models, mainly through their under-compression of LiDAR data and inefficient training objectives. To address this, we propose a latent conditional flow matching (CFM)-based frameworks that achieves state-of-the-art reconstruction accuracy using only half the training data and a compression ratio 6 times higher than that of prior methods. Our model achieves SOTA performance on future-trajectory-conditioned semantic occupancy forecasting while being 23x more computationally efficient (a 28x FPS speedup); and achieves SOTA performance on semantic occupancy forecasting while being 2x more computationally efficient (a 1.1x FPS speedup). 

**Abstract (ZH)**: 基于LiDAR的 world models 在多域迁移性方面的研究 

---
# RoboScape: Physics-informed Embodied World Model 

**Title (ZH)**: RoboScape: 物理驱动的实体世界模型 

**Authors**: Yu Shang, Xin Zhang, Yinzhou Tang, Lei Jin, Chen Gao, Wei Wu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.23135)  

**Abstract**: World models have become indispensable tools for embodied intelligence, serving as powerful simulators capable of generating realistic robotic videos while addressing critical data scarcity challenges. However, current embodied world models exhibit limited physical awareness, particularly in modeling 3D geometry and motion dynamics, resulting in unrealistic video generation for contact-rich robotic scenarios. In this paper, we present RoboScape, a unified physics-informed world model that jointly learns RGB video generation and physics knowledge within an integrated framework. We introduce two key physics-informed joint training tasks: temporal depth prediction that enhances 3D geometric consistency in video rendering, and keypoint dynamics learning that implicitly encodes physical properties (e.g., object shape and material characteristics) while improving complex motion modeling. Extensive experiments demonstrate that RoboScape generates videos with superior visual fidelity and physical plausibility across diverse robotic scenarios. We further validate its practical utility through downstream applications including robotic policy training with generated data and policy evaluation. Our work provides new insights for building efficient physics-informed world models to advance embodied intelligence research. The code is available at: this https URL. 

**Abstract (ZH)**: 世界模型已成为体态智能不可或缺的工具，作为强大的模拟器，能够生成逼真的机器人视频，同时解决关键的数据稀缺挑战。然而，当前的体态世界模型在建模3D几何和运动动力学方面表现出有限的物理意识，导致在涉及大量接触的机器人场景中生成不现实的视频。在本文中，我们提出RoboScape，这是一种统一的物理知情世界模型，在集成框架中联合学习RGB视频生成和物理知识。我们介绍了两个关键的物理知情联合训练任务：时间深度预测，以增强视频渲染中的3D几何一致性；关键点动力学学习，隐式编码物理属性（如物体形状和材料特性），同时改进复杂的运动建模。广泛的实验表明，RoboScape能够在多种机器人场景中生成具有卓越视觉保真度和物理可信度的视频。我们进一步通过下游应用，包括使用生成数据训练机器人策略和评估策略，验证了其实际效用。我们的工作为构建高效的物理知情世界模型以推进体态智能研究提供了新的见解。代码availability: this https URL。 

---
# SoMi-ToM: Evaluating Multi-Perspective Theory of Mind in Embodied Social Interactions 

**Title (ZH)**: SoMi-ToM: 评估具身社会互动中的多视角理论OfMind 

**Authors**: Xianzhe Fan, Xuhui Zhou, Chuanyang Jin, Kolby Nottingham, Hao Zhu, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2506.23046)  

**Abstract**: Humans continuously infer the states, goals, and behaviors of others by perceiving their surroundings in dynamic, real-world social interactions. However, most Theory of Mind (ToM) benchmarks only evaluate static, text-based scenarios, which have a significant gap compared to real interactions. We propose the SoMi-ToM benchmark, designed to evaluate multi-perspective ToM in embodied multi-agent complex social interactions. This benchmark is based on rich multimodal interaction data generated by the interaction environment SoMi, covering diverse crafting goals and social relationships. Our framework supports multi-level evaluation: (1) first-person evaluation provides multimodal (visual, dialogue, action, etc.) input from a first-person perspective during a task for real-time state inference, (2) third-person evaluation provides complete third-person perspective video and text records after a task for goal and behavior inference. This evaluation method allows for a more comprehensive examination of a model's ToM capabilities from both the subjective immediate experience and the objective global observation. We constructed a challenging dataset containing 35 third-person perspective videos, 363 first-person perspective images, and 1225 expert-annotated multiple-choice questions (three options). On this dataset, we systematically evaluated the performance of human subjects and several state-of-the-art large vision-language models (LVLMs). The results show that LVLMs perform significantly worse than humans on SoMi-ToM: the average accuracy gap between humans and models is 40.1% in first-person evaluation and 26.4% in third-person evaluation. This indicates that future LVLMs need to further improve their ToM capabilities in embodied, complex social interactions. 

**Abstract (ZH)**: 基于多视角心智理论的SoMi-ToM基准：评估动态真实世界多智能体社会交互中的心智理论能力 

---
# RoboPearls: Editable Video Simulation for Robot Manipulation 

**Title (ZH)**: RoboPearls: 可编辑视频仿真用于机器人 manipulation 

**Authors**: Tao Tang, Likui Zhang, Youpeng Wen, Kaidong Zhang, Jia-Wang Bian, xia zhou, Tianyi Yan, Kun Zhan, Peng Jia, Hefeng Wu, Liang Lin, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22756)  

**Abstract**: The development of generalist robot manipulation policies has seen significant progress, driven by large-scale demonstration data across diverse environments. However, the high cost and inefficiency of collecting real-world demonstrations hinder the scalability of data acquisition. While existing simulation platforms enable controlled environments for robotic learning, the challenge of bridging the sim-to-real gap remains. To address these challenges, we propose RoboPearls, an editable video simulation framework for robotic manipulation. Built on 3D Gaussian Splatting (3DGS), RoboPearls enables the construction of photo-realistic, view-consistent simulations from demonstration videos, and supports a wide range of simulation operators, including various object manipulations, powered by advanced modules like Incremental Semantic Distillation (ISD) and 3D regularized NNFM Loss (3D-NNFM). Moreover, by incorporating large language models (LLMs), RoboPearls automates the simulation production process in a user-friendly manner through flexible command interpretation and execution. Furthermore, RoboPearls employs a vision-language model (VLM) to analyze robotic learning issues to close the simulation loop for performance enhancement. To demonstrate the effectiveness of RoboPearls, we conduct extensive experiments on multiple datasets and scenes, including RLBench, COLOSSEUM, Ego4D, Open X-Embodiment, and a real-world robot, which demonstrate our satisfactory simulation performance. 

**Abstract (ZH)**: RoboPearls：可编辑视频模拟框架促进通用机器人操作政策开发 

---
# Innovative Research on IoT Architecture and Robotic Operating Platforms: Applications of Large Language Models and Generative AI 

**Title (ZH)**: 物联网架构与机器人操作系统方面的创新研究：大型语言模型和生成式人工智能的应用 

**Authors**: Huiwen Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.22477)  

**Abstract**: This paper introduces an innovative design for robotic operating platforms, underpinned by a transformative Internet of Things (IoT) architecture, seamlessly integrating cutting-edge technologies such as large language models (LLMs), generative AI, edge computing, and 5G networks. The proposed platform aims to elevate the intelligence and autonomy of IoT systems and robotics, enabling them to make real-time decisions and adapt dynamically to changing environments. Through a series of compelling case studies across industries including smart manufacturing, healthcare, and service sectors, this paper demonstrates the substantial potential of IoT-enabled robotics to optimize operational workflows, enhance productivity, and deliver innovative, scalable solutions. By emphasizing the roles of LLMs and generative AI, the research highlights how these technologies drive the evolution of intelligent robotics and IoT, shaping the future of industry-specific advancements. The findings not only showcase the transformative power of these technologies but also offer a forward-looking perspective on their broader societal and industrial implications, positioning them as catalysts for next-generation automation and technological convergence. 

**Abstract (ZH)**: 本文介绍了基于颠覆性物联网架构的创新机器人操作平台设计，该架构无缝整合了大型语言模型（LLMs）、生成AI、边缘计算和5G网络等前沿技术。所提出的平台旨在提升物联网系统和机器人的人工智能和自主性，使其能够实时决策并动态适应不断变化的环境。通过跨越智能制造、医疗保健和服务行业的一系列引人入胜的案例研究，本文展示了物联网赋能机器人在优化操作工作流程、提升生产力和提供创新可扩展解决方案方面的巨大潜力。强调了大型语言模型和生成AI的作用，研究突显了这些技术如何驱动智能机器人和物联网的演变，塑造了特定行业进步的未来。这些发现不仅展示了这些技术的变革力量，还提供了对未来社会和工业影响的前瞻性视角，将它们定位为下一代自动化和技术创新融合的催化剂。 

---
# Constructing Non-Markovian Decision Process via History Aggregator 

**Title (ZH)**: 通过历史聚合构建非马尔可夫决策过程 

**Authors**: Yongyi Wang, Wenxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.24026)  

**Abstract**: In the domain of algorithmic decision-making, non-Markovian dynamics manifest as a significant impediment, especially for paradigms such as Reinforcement Learning (RL), thereby exerting far-reaching consequences on the advancement and effectiveness of the associated systems. Nevertheless, the existing benchmarks are deficient in comprehensively assessing the capacity of decision algorithms to handle non-Markovian dynamics. To address this deficiency, we have devised a generalized methodology grounded in category theory. Notably, we established the category of Markov Decision Processes (MDP) and the category of non-Markovian Decision Processes (NMDP), and proved the equivalence relationship between them. This theoretical foundation provides a novel perspective for understanding and addressing non-Markovian dynamics. We further introduced non-Markovianity into decision-making problem settings via the History Aggregator for State (HAS). With HAS, we can precisely control the state dependency structure of decision-making problems in the time series. Our analysis demonstrates the effectiveness of our method in representing a broad range of non-Markovian dynamics. This approach facilitates a more rigorous and flexible evaluation of decision algorithms by testing them in problem settings where non-Markovian dynamics are explicitly constructed. 

**Abstract (ZH)**: 在算法决策领域，非马尔可夫动力学表现为一个显著的障碍，尤其是在强化学习（RL）等范式中，从而对相关系统的进步和有效性产生了深远影响。然而，现有的基准在全面评估决策算法处理非马尔可夫动力学的能力方面存在不足。为解决这一不足，我们基于范畴论提出了一个通用的方法论。我们建立了马尔可夫决策过程（MDP）范畴和非马尔可夫决策过程（NMDP）范畴，并证明了它们之间的等价关系。这一理论基础提供了理解并应对非马尔可夫动力学的新视角。我们还通过状态历史聚合器（HAS）将非马尔可夫性引入决策问题设置中。借助HAS，我们可以在时间序列中精确控制决策问题的状态依赖结构。我们的分析证明了该方法在表示广泛范围的非马尔可夫动力学方面的有效性。这种方法通过在明确构建非马尔可夫动力学的问题设置中测试决策算法，促进了更为严谨和灵活的评估。 

---
# Industrial brain: a human-like autonomous neuro-symbolic cognitive decision-making system 

**Title (ZH)**: 工业大脑：一种类人自主神经符号认知决策系统 

**Authors**: Junping Wang, Bicheng Wang, Yibo Xuea, Yuan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.23926)  

**Abstract**: Resilience non-equilibrium measurement, the ability to maintain fundamental functionality amidst failures and errors, is crucial for scientific management and engineering applications of industrial chain. The problem is particularly challenging when the number or types of multiple co-evolution of resilience (for example, randomly placed) are extremely chaos. Existing end-to-end deep learning ordinarily do not generalize well to unseen full-feld reconstruction of spatiotemporal co-evolution structure, and predict resilience of network topology, especially in multiple chaos data regimes typically seen in real-world applications. To address this challenge, here we propose industrial brain, a human-like autonomous cognitive decision-making and planning framework integrating higher-order activity-driven neuro network and CT-OODA symbolic reasoning to autonomous plan resilience directly from observational data of global variable. The industrial brain not only understands and model structure of node activity dynamics and network co-evolution topology without simplifying assumptions, and reveal the underlying laws hidden behind complex networks, but also enabling accurate resilience prediction, inference, and planning. Experimental results show that industrial brain significantly outperforms resilience prediction and planning methods, with an accurate improvement of up to 10.8\% over GoT and OlaGPT framework and 11.03\% over spectral dimension reduction. It also generalizes to unseen topologies and dynamics and maintains robust performance despite observational disturbances. Our findings suggest that industrial brain addresses an important gap in resilience prediction and planning for industrial chain. 

**Abstract (ZH)**: 工业链韧性非平衡测量：一种类人自主认知决策和规划框架 

---
# Self-correcting Reward Shaping via Language Models for Reinforcement Learning Agents in Games 

**Title (ZH)**: 基于语言模型的自纠正奖励塑造方法在游戏中的应用 

**Authors**: António Afonso, Iolanda Leite, Alessandro Sestini, Florian Fuchs, Konrad Tollmar, Linus Gisslén  

**Link**: [PDF](https://arxiv.org/pdf/2506.23626)  

**Abstract**: Reinforcement Learning (RL) in games has gained significant momentum in recent years, enabling the creation of different agent behaviors that can transform a player's gaming experience. However, deploying RL agents in production environments presents two key challenges: (1) designing an effective reward function typically requires an RL expert, and (2) when a game's content or mechanics are modified, previously tuned reward weights may no longer be optimal. Towards the latter challenge, we propose an automated approach for iteratively fine-tuning an RL agent's reward function weights, based on a user-defined language based behavioral goal. A Language Model (LM) proposes updated weights at each iteration based on this target behavior and a summary of performance statistics from prior training rounds. This closed-loop process allows the LM to self-correct and refine its output over time, producing increasingly aligned behavior without the need for manual reward engineering. We evaluate our approach in a racing task and show that it consistently improves agent performance across iterations. The LM-guided agents show a significant increase in performance from $9\%$ to $74\%$ success rate in just one iteration. We compare our LM-guided tuning against a human expert's manual weight design in the racing task: by the final iteration, the LM-tuned agent achieved an $80\%$ success rate, and completed laps in an average of $855$ time steps, a competitive performance against the expert-tuned agent's peak $94\%$ success, and $850$ time steps. 

**Abstract (ZH)**: 基于语言模型的强化学习代理奖励函数自动化调优方法 

---
# Data Augmentation for Cognitive Behavioral Therapy: Leveraging ERNIE Language Models using Artificial Intelligence 

**Title (ZH)**: 基于ERNIE语言模型的数据增强认知行为疗法：利用人工智能技术 

**Authors**: Bosubabu Sambana, Kondreddygari Archana, Suram Indhra Sena Reddy, Shaik Meethaigar Jameer Basha, Shaik Karishma  

**Link**: [PDF](https://arxiv.org/pdf/2506.23503)  

**Abstract**: Cognitive Behavioral Therapy (CBT) is a proven approach for addressing the irrational thought patterns associated with mental health disorders, but its effectiveness relies on accurately identifying cognitive pathways to provide targeted treatment. In today's digital age, individuals often express negative emotions on social media, where they may reveal cognitive distortions, and in severe cases, exhibit suicidal tendencies. However, there is a significant gap in methodologies designed to analyze these cognitive pathways, which could be critical for psychotherapists aiming to deliver timely and effective interventions in online environments. Cognitive Behavioral Therapy (CBT) framework leveraging acceptance, commitment and data augmentation to categorize and address both textual and visual content as positive or negative. Specifically, the system employs BERT, RoBERTa for Sentiment Analysis and T5, PEGASUS for Text Summarization, mT5 for Text Translation in Multiple Languages focusing on detecting negative emotions and cognitive distortions within social media data. While existing models are primarily designed to identify negative thoughts, the proposed system goes beyond this by predicting additional negative side effects and other potential mental health disorders likes Phobias, Eating Disorders. This enhancement allows for a more comprehensive understanding and intervention strategy, offering psychotherapists a powerful tool for early detection and treatment of various psychological issues. 

**Abstract (ZH)**: 基于接纳、承诺和数据增强的认知行为疗法（CBT）框架：通过分析文本和视觉内容识别和应对消极情绪和认知扭曲 

---
# GATSim: Urban Mobility Simulation with Generative Agents 

**Title (ZH)**: GATSim: 城市移动性仿真with生成型代理 

**Authors**: Qi Liu, Can Li, Wanjing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.23306)  

**Abstract**: Traditional agent-based urban mobility simulations rely on rigid rule-based systems that fail to capture the complexity, adaptability, and behavioral diversity characteristic of human travel decision-making. Recent advances in large language models and AI agent technology offer opportunities to create agents with reasoning capabilities, persistent memory, and adaptive learning mechanisms. We propose GATSim (Generative-Agent Transport Simulation), a novel framework that leverages these advances to create generative agents with rich behavioral characteristics for urban mobility simulation. Unlike conventional approaches, GATSim agents possess diverse socioeconomic attributes, individual lifestyles, and evolving preferences that shape their mobility decisions through psychologically-informed memory systems, tool usage capabilities, and lifelong learning mechanisms. The main contributions of this study include: (1) a comprehensive architecture combining an urban mobility foundation model with agent cognitive systems and transport simulation environment, (2) a fully functional prototype implementation, and (3) systematic validation demonstrating that generative agents produce believable travel behaviors. Through designed reflection processes, generative agents in this study can transform specific travel experiences into generalized insights, enabling realistic behavioral adaptation over time with specialized mechanisms for activity planning and real-time reactive behaviors tailored to urban mobility contexts. Experiments show that generative agents perform competitively with human annotators in mobility scenarios while naturally producing macroscopic traffic evolution patterns. The code for the prototype system is shared at this https URL. 

**Abstract (ZH)**: 基于生成智能体的城市交通仿真：一种结合丰富行为特征的新型框架 

---
# Bridging Ethical Principles and Algorithmic Methods: An Alternative Approach for Assessing Trustworthiness in AI Systems 

**Title (ZH)**: 伦理原则与算法方法的桥梁：评估人工智能系统可信性的替代方法 

**Authors**: Michael Papademas, Xenia Ziouvelou, Antonis Troumpoukis, Vangelis Karkaletsis  

**Link**: [PDF](https://arxiv.org/pdf/2506.22774)  

**Abstract**: Artificial Intelligence (AI) technology epitomizes the complex challenges posed by human-made artifacts, particularly those widely integrated into society and exert significant influence, highlighting potential benefits and their negative consequences. While other technologies may also pose substantial risks, AI's pervasive reach makes its societal effects especially profound. The complexity of AI systems, coupled with their remarkable capabilities, can lead to a reliance on technologies that operate beyond direct human oversight or understanding. To mitigate the risks that arise, several theoretical tools and guidelines have been developed, alongside efforts to create technological tools aimed at safeguarding Trustworthy AI. The guidelines take a more holistic view of the issue but fail to provide techniques for quantifying trustworthiness. Conversely, while technological tools are better at achieving such quantification, they lack a holistic perspective, focusing instead on specific aspects of Trustworthy AI. This paper aims to introduce an assessment method that combines the ethical components of Trustworthy AI with the algorithmic processes of PageRank and TrustRank. The goal is to establish an assessment framework that minimizes the subjectivity inherent in the self-assessment techniques prevalent in the field by introducing algorithmic criteria. The application of our approach indicates that a holistic assessment of an AI system's trustworthiness can be achieved by providing quantitative insights while considering the theoretical content of relevant guidelines. 

**Abstract (ZH)**: 人工智能（AI）技术体现了由人类制造的复杂挑战，特别是一些广泛融入社会并发挥重大影响的技术，凸显了其潜在益处及其负面影响。尽管其他技术也可能带来重大风险，但AI的广泛影响使其对社会的影响尤为深远。AI系统的复杂性与其卓越的能力相结合，可能导致对超出直接人类监督或理解的技术系统的依赖。为了缓解由此产生的风险，已开发出一些理论工具和指导原则，并努力创建旨在保障可信AI的技术工具。指导原则提供了更加整体的观点，但未能提供量化可信度的技术。相反，虽然技术工具在实现这种量化方面效果更好，但它们缺乏整体视角，而是专注于可信AI的特定方面。本文旨在引入一种结合可信AI的伦理要素与PageRank和TrustRank算法过程的评估方法。目标是通过引入算法标准来最小化当前领域中常用的自我评估技术中的主观性，建立一个评估框架。应用我们的方法表明，通过考虑相关指导原则的理论内容，可以实现对AI系统可信度的整体评估，同时提供定量洞察。 

---
# Bridging Physical and Digital Worlds: Embodied Large AI for Future Wireless Systems 

**Title (ZH)**: 物理世界与数字世界交融：面向未来无线系统的具身大规模AI 

**Authors**: Xinquan Wang, Fenghao Zhu, Zhaohui Yang, Chongwen Huang, Xiaoming Chen, Zhaoyang Zhang, Sami Muhaidat, Mérouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2506.24009)  

**Abstract**: Large artificial intelligence (AI) models offer revolutionary potential for future wireless systems, promising unprecedented capabilities in network optimization and performance. However, current paradigms largely overlook crucial physical interactions. This oversight means they primarily rely on offline datasets, leading to difficulties in handling real-time wireless dynamics and non-stationary environments. Furthermore, these models often lack the capability for active environmental probing. This paper proposes a fundamental paradigm shift towards wireless embodied large AI (WELAI), moving from passive observation to active embodiment. We first identify key challenges faced by existing models, then we explore the design principles and system structure of WELAI. Besides, we outline prospective applications in next-generation wireless. Finally, through an illustrative case study, we demonstrate the effectiveness of WELAI and point out promising research directions for realizing adaptive, robust, and autonomous wireless systems. 

**Abstract (ZH)**: 大型人工智能模型为未来无线系统带来了革命性的潜力，有望在网络优化和性能方面实现前所未有的能力。然而，当前的范式很大程度上忽视了关键的物理交互。这一忽视意味着它们主要依赖于离线数据集，这导致了在处理实时无线动态和非稳态环境时的困难。此外，这些模型通常缺乏主动环境探测的能力。本文提出了一种向无线嵌入式大型人工智能（WELAI）的基本范式转变，从被动观察转向主动体化。首先，我们识别现有模型面临的关键挑战，然后探讨WELAI的设计原理和系统结构。此外，我们概述了WELAI在下一代无线系统中的潜在应用。最后，通过一个示例性案例研究，我们展示了WELAI的有效性，并指出了实现适应性强、稳健且自主的无线系统的研究方向。 

---
# Reinforcement Learning for Synchronised Flow Control in a Dual-Gate Resin Infusion System 

**Title (ZH)**: 双门树脂灌注系统中同步流控制的强化学习方法 

**Authors**: Miguel Camacho-Sánchez, Fernando García-Torres, Jesper John Lisegaard, Rocío del Amor, Sankhya Mohanty, Valery Naranjo  

**Link**: [PDF](https://arxiv.org/pdf/2506.23923)  

**Abstract**: Resin infusion (RI) and resin transfer moulding (RTM) are critical processes for the manufacturing of high-performance fibre-reinforced polymer composites, particularly for large-scale applications such as wind turbine blades. Controlling the resin flow dynamics in these processes is critical to ensure the uniform impregnation of the fibre reinforcements, thereby preventing residual porosities and dry spots that impact the consequent structural integrity of the final component. This paper presents a reinforcement learning (RL) based strategy, established using process simulations, for synchronising the different resin flow fronts in an infusion scenario involving two resin inlets and a single outlet. Using Proximal Policy Optimisation (PPO), our approach addresses the challenge of managing the fluid dynamics in a partially observable environment. The results demonstrate the effectiveness of the RL approach in achieving an accurate flow convergence, highlighting its potential towards improving process control and product quality in composites manufacturing. 

**Abstract (ZH)**: 树脂灌注（RI）和树脂传输模塑（RTM）是制造高性能纤维增强聚合物复合材料的关键工艺，尤其适用于大型应用如风力涡轮机叶片。控制这些工艺中的树脂流动动力学对于确保纤维增强材料的均匀浸润、防止残留孔隙和干燥区域，从而保证最终组件的结构完整性至关重要。本文提出了一种基于过程模拟的强化学习（RL）策略，用于同步涉及两个树脂入口和一个出口的灌注场景中的不同树脂流动前沿。使用近端策略优化（PPO），该方法解决了在部分可观测环境中管理流体动力学的挑战。结果表明，基于RL的方法在实现准确的流体汇聚方面具有有效性，突显了其在复合材料制造过程中改进工艺控制和产品质量方面的潜力。 

---
# Towards the "Digital Me": A vision of authentic Conversational Agents powered by personal Human Digital Twins 

**Title (ZH)**: “数字我”的愿景：由个性化人类数字双子驱动的可信对话代理 

**Authors**: Lluís C. Coll, Martin W. Lauer-Schmaltz, Philip Cash, John P. Hansen, Anja Maier  

**Link**: [PDF](https://arxiv.org/pdf/2506.23826)  

**Abstract**: Human Digital Twins (HDTs) have traditionally been conceptualized as data-driven models designed to support decision-making across various domains. However, recent advancements in conversational AI open new possibilities for HDTs to function as authentic, interactive digital counterparts of individuals. This paper introduces a novel HDT system architecture that integrates large language models with dynamically updated personal data, enabling it to mirror an individual's conversational style, memories, and behaviors. To achieve this, our approach implements context-aware memory retrieval, neural plasticity-inspired consolidation, and adaptive learning mechanisms, creating a more natural and evolving digital persona. The resulting system does not only replicate an individual's unique conversational style depending on who they are speaking with, but also enriches responses with dynamically captured personal experiences, opinions, and memories. While this marks a significant step toward developing authentic virtual counterparts, it also raises critical ethical concerns regarding privacy, accountability, and the long-term implications of persistent digital identities. This study contributes to the field of HDTs by describing our novel system architecture, demonstrating its capabilities, and discussing future directions and emerging challenges to ensure the responsible and ethical development of HDTs. 

**Abstract (ZH)**: 人类数字双胞胎（HDTs）传统上被视为一种数据驱动的模型，旨在支持跨各个领域的决策制定。然而，近期对话式AI的发展为HDTs的功能提供了新的可能性，使其能够作为个体的真实互动数字对应物。本文介绍了一种新型的HDT系统架构，该架构将大型语言模型与动态更新的个人数据相结合，使其能够反映个体的对话风格、记忆和行为。为了实现这一点，我们的方法实现了基于上下文的记忆检索、受神经可塑性启发的整合以及自适应学习机制，从而创造了一个更加自然和演化中的数字人格。由此产生的系统不仅根据对话对象来复制个体独特的对话风格，还通过动态捕捉的个人经历、观点和记忆丰富了回应。虽然这标志着向开发真实虚拟对应物迈出了一大步，但也引发了隐私、问责制以及持久数字身份长期影响的关键伦理问题。本文为HDTs领域贡献了我们的新型系统架构描述、展示了其功能，并讨论了未来方向和新兴挑战，以确保HDTs的责任和伦理发展。 

---
# Online Human Action Detection during Escorting 

**Title (ZH)**: 在线护送期间的人类动作检测 

**Authors**: Siddhartha Mondal, Avik Mitra, Chayan Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2506.23573)  

**Abstract**: The deployment of robot assistants in large indoor spaces has seen significant growth, with escorting tasks becoming a key application. However, most current escorting robots primarily rely on navigation-focused strategies, assuming that the person being escorted will follow without issue. In crowded environments, this assumption often falls short, as individuals may struggle to keep pace, become obstructed, get distracted, or need to stop unexpectedly. As a result, conventional robotic systems are often unable to provide effective escorting services due to their limited understanding of human movement dynamics. To address these challenges, an effective escorting robot must continuously detect and interpret human actions during the escorting process and adjust its movement accordingly. However, there is currently no existing dataset designed specifically for human action detection in the context of escorting. Given that escorting often occurs in crowded environments, where other individuals may enter the robot's camera view, the robot also needs to identify the specific human it is escorting (the subject) before predicting their actions. Since no existing model performs both person re-identification and action prediction in real-time, we propose a novel neural network architecture that can accomplish both tasks. This enables the robot to adjust its speed dynamically based on the escortee's movements and seamlessly resume escorting after any disruption. In comparative evaluations against strong baselines, our system demonstrates superior efficiency and effectiveness, showcasing its potential to significantly improve robotic escorting services in complex, real-world scenarios. 

**Abstract (ZH)**: 大型室内空间中陪伴机器人部署的显著增长使其辅助任务成为关键应用。然而，当前大多数陪伴机器人主要依赖于专注于导航的策略，假设被陪伴者会毫无问题地跟随。在拥挤的环境中，这一假设往往难以满足，因为个人可能会难以保持步伐、被阻挡、分心或需要突然停下。因此，由于传统机器人系统对人类运动动态的理解有限，它们往往无法提供有效的陪伴服务。为应对这些挑战，一个有效的陪伴机器人必须在陪伴过程中连续检测和解释人类行为，并相应地调整其运动。然而，目前尚无专门设计用于陪伴场景中的人类动作检测的数据集。鉴于陪伴往往发生在拥挤的环境中，其他个体可能会进入机器人的摄像头视图，机器人在预测被陪伴者的行为之前还需要识别出特定的目标个体（主体）。由于现有模型无法在实时环境中同时进行人员再识别和动作预测，我们提出了一种新的神经网络架构，可以同时完成这两项任务。这使机器人能够根据被陪伴者的动作动态调整其速度，并在任何中断后无缝恢复陪伴服务。在与强大基线系统的比较评估中，我们的系统显示出更高的效率和有效性，突显了其在复杂现实场景中显著改善机器人陪伴服务的潜力。 

---
# Objective-Free Local Learning and Emergent Language Structure in Thinking Machines 

**Title (ZH)**: 基于目标的局部学习与思考机器中 Emergent 语言结构的涌现 

**Authors**: P. Myles Eugenio  

**Link**: [PDF](https://arxiv.org/pdf/2506.23293)  

**Abstract**: We present a neuro-symbolic framework for generative language modeling based on local, event-driven emergent learning. At its core is a hierarchical Hopfield memory chain acting as a compositional short-term memory and dynamic tokenizer (retokenizer). Rather than relying on predefined tokens or supervision, the model builds structure from scratch, learning symbol sequences as multi-scale representations. It constructs projection tensors that bind co-occurring features into hierarchical tokens, introducing redundancy (i.e an emergent gauge structure) and enabling compression of local activations into long-range dependencies. Curiously, we find that the retokenizer can filter natural language patterns from noise, generating synthetic languages with coherent internal morphology -- quantifiably the same as human language. Language is learned in a local (Hebbian) fashion, where model constraints dictate allowed emergent structure, and new information is retained in alignment with this structure. The absence of a global objective enables a form of plasticity not found in conventional language models, allowing the system to generalize beyond its initial inference class -- even without explicit data. We demonstrate that briefly activating a new neuron during inference binds distributed multi-scale token features into a symbolic embedding. These emergent embedding neurons act as long-term memory and support a key-value mechanism for compositional inference and generalization. This architecture provides a methodological foundation for studying how symbolic structure can emerge from local neural learning. It offers a new pathway for building scalable, interpretable neuro-symbolic systems -- where tokens, grammar, and reasoning arise as compressed memory traces within a Hopfield hierarchy. This approach advances the development of neuromorphic architectures for generative language models. 

**Abstract (ZH)**: 基于局部事件驱动 emergent 学习的神经符号生成语言模型框架 

---
# Unleashing Embodied Task Planning Ability in LLMs via Reinforcement Learning 

**Title (ZH)**: 通过强化学习释放大型语言模型的实体任务规划能力 

**Authors**: Zhaoye Fei, Li Ji, Siyin Wang, Junhao Shi, Jingjing Gong, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23127)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, yet they face significant challenges in embodied task planning scenarios that require continuous environmental understanding and action generation. Existing approaches generate open-loop action scripts based on static knowledge, making it difficult to learn causal relationships between actions and environmental feedback, particularly in partially observable environments. We introduce Embodied Planner-R1, a novel outcome-driven reinforcement learning framework that enables LLMs to develop interactive capabilities through autonomous exploration with minimal supervision. Our framework incorporates three key innovations: (1) Without human annotations, we employ pure reinforcement learning with group rollout, incorporating in-environment interaction through parallel exploration; (2) completion-driven sparse reward; and (3) Interactive Policy Optimization (IPO) for efficient learning from grouped trajectories. Across two challenging text-based Embodied planning benchmarks, Embodied Planner-R1 achieves impressive completion rates of 97.78% on ALFWorld and 79.92% on ScienceWorld, surpassing prior methods by a large margin, and suffers only a -3.66% drop in previously unseen environments, evidencing strong generalization. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种任务中展现了 remarkable 的能力，但在需要持续环境理解和行动生成的嵌入式任务规划场景中面临显著挑战。现有方法基于静态知识生成开环行动脚本，难以学习行动与环境反馈之间的因果关系，尤其是在部分可观测环境中。我们引入了 Embodied Planner-R1，这是一种新型的结果驱动强化学习框架，能够让大型语言模型通过最少的监督自主探索来发展交互能力。我们的框架包含三个关键创新：（1）无需人工标注，我们采用群体回放的纯强化学习方法，并通过并行探索在环境中进行交互；（2）基于完成任务的稀疏奖励；（3）交互策略优化（IPO）以高效学习分组轨迹。在两个具有挑战性的基于文本的嵌入式规划基准测试中，Embodied Planner-R1 在 ALFWorld 中实现了 97.78% 的完成率，在 ScienceWorld 中实现了 79.92% 的完成率，显著优于之前的方法，并且在未见过的环境中仅表现出 -3.66% 的下降，证明了强大的泛化能力。 

---
# Curious Causality-Seeking Agents Learn Meta Causal World 

**Title (ZH)**: 好奇的因果探索智能体学习元因果世界 

**Authors**: Zhiyu Zhao, Haoxuan Li, Haifeng Zhang, Jun Wang, Francesco Faccio, Jürgen Schmidhuber, Mengyue Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23068)  

**Abstract**: When building a world model, a common assumption is that the environment has a single, unchanging underlying causal rule, like applying Newton's laws to every situation. In reality, what appears as a drifting causal mechanism is often the manifestation of a fixed underlying mechanism seen through a narrow observational window. This brings about a problem that, when building a world model, even subtle shifts in policy or environment states can alter the very observed causal mechanisms. In this work, we introduce the \textbf{Meta-Causal Graph} as world models, a minimal unified representation that efficiently encodes the transformation rules governing how causal structures shift across different latent world states. A single Meta-Causal Graph is composed of multiple causal subgraphs, each triggered by meta state, which is in the latent state space. Building on this representation, we introduce a \textbf{Causality-Seeking Agent} whose objectives are to (1) identify the meta states that trigger each subgraph, (2) discover the corresponding causal relationships by agent curiosity-driven intervention policy, and (3) iteratively refine the Meta-Causal Graph through ongoing curiosity-driven exploration and agent experiences. Experiments on both synthetic tasks and a challenging robot arm manipulation task demonstrate that our method robustly captures shifts in causal dynamics and generalizes effectively to previously unseen contexts. 

**Abstract (ZH)**: 元因果图作为世界模型：一种统一表示因果结构变换规则的最小化模型 

---
# Offline Reinforcement Learning for Mobility Robustness Optimization 

**Title (ZH)**: 离线强化学习在移动性稳健性优化中的应用 

**Authors**: Pegah Alizadeh, Anastasios Giovanidis, Pradeepa Ramachandra, Vasileios Koutsoukis, Osama Arouk  

**Link**: [PDF](https://arxiv.org/pdf/2506.22793)  

**Abstract**: In this work we revisit the Mobility Robustness Optimisation (MRO) algorithm and study the possibility of learning the optimal Cell Individual Offset tuning using offline Reinforcement Learning. Such methods make use of collected offline datasets to learn the optimal policy, without further exploration. We adapt and apply a sequence-based method called Decision Transformers as well as a value-based method called Conservative Q-Learning to learn the optimal policy for the same target reward as the vanilla rule-based MRO. The same input features related to failures, ping-pongs, and other handover issues are used. Evaluation for realistic New Radio networks with 3500 MHz carrier frequency on a traffic mix including diverse user service types and a specific tunable cell-pair shows that offline-RL methods outperform rule-based MRO, offering up to 7% improvement. Furthermore, offline-RL can be trained for diverse objective functions using the same available dataset, thus offering operational flexibility compared to rule-based methods. 

**Abstract (ZH)**: 基于离线强化学习的CELL INDIVIDUAL OFFSET调优研究 

---
# Unifying Biomedical Vision-Language Expertise: Towards a Generalist Foundation Model via Multi-CLIP Knowledge Distillation 

**Title (ZH)**: 统一生物医学视觉-语言专长：通过多CLIP知识蒸馏 toward 通用基础模型 

**Authors**: Shansong Wang, Zhecheng Jin, Mingzhe Hu, Mojtaba Safari, Feng Zhao, Chih-Wei Chang, Richard LJ Qiu, Justin Roper, David S. Yu, Xiaofeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22567)  

**Abstract**: CLIP models pretrained on natural images with billion-scale image-text pairs have demonstrated impressive capabilities in zero-shot classification, cross-modal retrieval, and open-ended visual answering. However, transferring this success to biomedicine is hindered by the scarcity of large-scale biomedical image-text corpora, the heterogeneity of image modalities, and fragmented data standards across institutions. These limitations hinder the development of a unified and generalizable biomedical foundation model trained from scratch. To overcome this, we introduce MMKD-CLIP, a generalist biomedical foundation model developed via Multiple Medical CLIP Knowledge Distillation. Rather than relying on billion-scale raw data, MMKD-CLIP distills knowledge from nine state-of-the-art domain-specific or generalist biomedical CLIP models, each pretrained on millions of biomedical image-text pairs. Our two-stage training pipeline first performs CLIP-style pretraining on over 2.9 million biomedical image-text pairs from 26 image modalities, followed by feature-level distillation using over 19.2 million feature pairs extracted from teacher models. We evaluate MMKD-CLIP on 58 diverse biomedical datasets, encompassing over 10.8 million biomedical images across nine image modalities. The evaluation spans six core task types: zero-shot classification, linear probing, cross-modal retrieval, visual question answering, survival prediction, and cancer diagnosis. MMKD-CLIP consistently outperforms all teacher models while demonstrating remarkable robustness and generalization across image domains and task settings. These results underscore that multi-teacher knowledge distillation is a scalable and effective paradigm for building high-performing biomedical foundation models under the practical constraints of real-world data availability. 

**Abstract (ZH)**: MMKD-CLIP：基于多医学CLIP知识精炼的一般ist生物医药基础模型 

---
# Seamless Interaction: Dyadic Audiovisual Motion Modeling and Large-Scale Dataset 

**Title (ZH)**: 无缝交互：二元音频视觉运动建模与大规模数据集 

**Authors**: Vasu Agrawal, Akinniyi Akinyemi, Kathryn Alvero, Morteza Behrooz, Julia Buffalini, Fabio Maria Carlucci, Joy Chen, Junming Chen, Zhang Chen, Shiyang Cheng, Praveen Chowdary, Joe Chuang, Antony D'Avirro, Jon Daly, Ning Dong, Mark Duppenthaler, Cynthia Gao, Jeff Girard, Martin Gleize, Sahir Gomez, Hongyu Gong, Srivathsan Govindarajan, Brandon Han, Sen He, Denise Hernandez, Yordan Hristov, Rongjie Huang, Hirofumi Inaguma, Somya Jain, Raj Janardhan, Qingyao Jia, Christopher Klaiber, Dejan Kovachev, Moneish Kumar, Hang Li, Yilei Li, Pavel Litvin, Wei Liu, Guangyao Ma, Jing Ma, Martin Ma, Xutai Ma, Lucas Mantovani, Sagar Miglani, Sreyas Mohan, Louis-Philippe Morency, Evonne Ng, Kam-Woh Ng, Tu Anh Nguyen, Amia Oberai, Benjamin Peloquin, Juan Pino, Jovan Popovic, Omid Poursaeed, Fabian Prada, Alice Rakotoarison, Alexander Richard, Christophe Ropers, Safiyyah Saleem, Vasu Sharma, Alex Shcherbyna, Jia Shen, Jie Shen, Anastasis Stathopoulos, Anna Sun, Paden Tomasello, Tuan Tran, Arina Turkatenko, Bo Wan, Chao Wang, Jeff Wang, Mary Williamson, Carleigh Wood, Tao Xiang, Yilin Yang, Julien Yao, Chen Zhang, Jiemin Zhang, Xinyue Zhang, Jason Zheng, Pavlo Zhyzheria, Jan Zikes, Michael Zollhoefer  

**Link**: [PDF](https://arxiv.org/pdf/2506.22554)  

**Abstract**: Human communication involves a complex interplay of verbal and nonverbal signals, essential for conveying meaning and achieving interpersonal goals. To develop socially intelligent AI technologies, it is crucial to develop models that can both comprehend and generate dyadic behavioral dynamics. To this end, we introduce the Seamless Interaction Dataset, a large-scale collection of over 4,000 hours of face-to-face interaction footage from over 4,000 participants in diverse contexts. This dataset enables the development of AI technologies that understand dyadic embodied dynamics, unlocking breakthroughs in virtual agents, telepresence experiences, and multimodal content analysis tools. We also develop a suite of models that utilize the dataset to generate dyadic motion gestures and facial expressions aligned with human speech. These models can take as input both the speech and visual behavior of their interlocutors. We present a variant with speech from an LLM model and integrations with 2D and 3D rendering methods, bringing us closer to interactive virtual agents. Additionally, we describe controllable variants of our motion models that can adapt emotional responses and expressivity levels, as well as generating more semantically-relevant gestures. Finally, we discuss methods for assessing the quality of these dyadic motion models, which are demonstrating the potential for more intuitive and responsive human-AI interactions. 

**Abstract (ZH)**: 人类沟通涉及言语和非言语信号的复杂交互，对于传达意义和实现人际目标至关重要。为了开发社会智能人工智能技术，必须开发既能理解又能生成双边行为动态的模型。为此，我们引入了无缝交互数据集，该数据集包含来自4000多名参与者超过4000小时多元情境下的面对面互动视频。该数据集促成了能够理解双边 embodiable 动态的AI技术的发展，开启了虚拟代理、远程在场体验和多模态内容分析工具的重大突破。我们还开发了一套利用该数据集生成与人类言语相匹配的双边运动姿势和面部表情的模型。这些模型可以输入对话双方的语音和视觉行为。我们提出了一种使用语言模型语音变体，并与2D和3D渲染方法集成的版本，使我们更接近互动虚拟代理。此外，我们描述了可控制的运动模型变体，可以适应情感反应和表达程度，并生成更具语义相关性的手势。最后，我们讨论了评估这些双边运动模型质量的方法，这些方法展示了更直观和响应式的以人为中心的人机交互的潜力。 

---
# Visual-Semantic Knowledge Conflicts in Operating Rooms: Synthetic Data Curation for Surgical Risk Perception in Multimodal Large Language Models 

**Title (ZH)**: 手术室中的视觉-语义知识冲突：多模态大型语言模型中手术风险感知的合成数据整理 

**Authors**: Weiyi Zhao, Xiaoyu Tan, Liang Liu, Sijia Li, Youwei Song, Xihe Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.22500)  

**Abstract**: Surgical risk identification is critical for patient safety and reducing preventable medical errors. While multimodal large language models (MLLMs) show promise for automated operating room (OR) risk detection, they often exhibit visual-semantic knowledge conflicts (VS-KC), failing to identify visual safety violations despite understanding textual rules. To address this, we introduce a dataset comprising over 34,000 synthetic images generated by diffusion models, depicting operating room scenes containing entities that violate established safety rules. These images were created to alleviate data scarcity and examine MLLMs vulnerabilities. In addition, the dataset includes 214 human-annotated images that serve as a gold-standard reference for validation. This comprehensive dataset, spanning diverse perspectives, stages, and configurations, is designed to expose and study VS-KC. Fine-tuning on OR-VSKC significantly improves MLLMs' detection of trained conflict entities and generalizes well to new viewpoints for these entities, but performance on untrained entity types remains poor, highlighting learning specificity and the need for comprehensive training. The main contributions of this work include: (1) a data generation methodology tailored for rule-violation scenarios; (2) the release of the OR-VSKC dataset and its associated benchmark as open-source resources; and (3) an empirical analysis of violation-sensitive knowledge consistency in representative MLLMs. The dataset and appendix are available at this https URL. 

**Abstract (ZH)**: 手术风险识别对于保障患者安全和减少可预防的医疗错误至关重要。虽然多模态大规模语言模型（MLLMs）在自动化手术室（OR）风险检测方面显示出潜力，但它们往往表现出视觉-语义知识冲突（VS-KC），即使理解了文本规则也无法识别视觉安全违规。为此，我们引入了一个包含超过34,000张由扩散模型生成的合成图像的数据集，这些图像描绘了包含违反既有安全规则的实体的手术室场景，旨在缓解数据稀缺并考察MLLMs的脆弱性。此外，该数据集还包括214张由人工标注的图像，作为验证的黄金标准参考。该全面的数据集覆盖了多角度、多阶段和多配置，旨在揭示和研究VS-KC。对OR-VSKC进行微调显著提高了MLLMs对训练冲突实体的检测能力，并且在这些实体的新视角上泛化良好，但对未训练实体类型的性能仍然不佳，突出了学习特定性以及综合训练的需要。本工作的主要贡献包括：（1）一种针对规则违规场景的数据生成方法；（2）发布OR-VSKC数据集及其相关基准作为开源资源；（3）对代表性MLLMs在违规敏感知识一致性方面的实证分析。该数据集及其附录可在以下链接获取：this https URL。 

---
# Hierarchical Adversarially-Resilient Multi-Agent Reinforcement Learning for Cyber-Physical Systems Security 

**Title (ZH)**: 基于博弈鲁棒性的层次化多智能体强化学习在 cyber-物理系统安全中的应用 

**Authors**: Saad Alqithami  

**Link**: [PDF](https://arxiv.org/pdf/2506.22445)  

**Abstract**: Cyber-Physical Systems play a critical role in the infrastructure of various sectors, including manufacturing, energy distribution, and autonomous transportation systems. However, their increasing connectivity renders them highly vulnerable to sophisticated cyber threats, such as adaptive and zero-day attacks, against which traditional security methods like rule-based intrusion detection and single-agent reinforcement learning prove insufficient. To overcome these challenges, this paper introduces a novel Hierarchical Adversarially-Resilient Multi-Agent Reinforcement Learning (HAMARL) framework. HAMARL employs a hierarchical structure consisting of local agents dedicated to subsystem security and a global coordinator that oversees and optimizes comprehensive, system-wide defense strategies. Furthermore, the framework incorporates an adversarial training loop designed to simulate and anticipate evolving cyber threats, enabling proactive defense adaptation. Extensive experimental evaluations conducted on a simulated industrial IoT testbed indicate that HAMARL substantially outperforms traditional multi-agent reinforcement learning approaches, significantly improving attack detection accuracy, reducing response times, and ensuring operational continuity. The results underscore the effectiveness of combining hierarchical multi-agent coordination with adversarially-aware training to enhance the resilience and security of next-generation CPS. 

**Abstract (ZH)**: 基于物理系统的网络安全：一种新型分级对抗鲁棒多智能体强化学习框架在工业物联网中的应用 

---
