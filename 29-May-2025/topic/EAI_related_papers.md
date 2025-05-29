# FastTD3: Simple, Fast, and Capable Reinforcement Learning for Humanoid Control 

**Title (ZH)**: FastTD3：简单、快速且能力强的类人机器人控制 reinforcement 学习方法 

**Authors**: Younggyo Seo, Carmelo Sferrazza, Haoran Geng, Michal Nauman, Zhao-Heng Yin, Pieter Abbeel  

**Link**: [PDF](https://arxiv.org/pdf/2505.22642)  

**Abstract**: Reinforcement learning (RL) has driven significant progress in robotics, but its complexity and long training times remain major bottlenecks. In this report, we introduce FastTD3, a simple, fast, and capable RL algorithm that significantly speeds up training for humanoid robots in popular suites such as HumanoidBench, IsaacLab, and MuJoCo Playground. Our recipe is remarkably simple: we train an off-policy TD3 agent with several modifications -- parallel simulation, large-batch updates, a distributional critic, and carefully tuned hyperparameters. FastTD3 solves a range of HumanoidBench tasks in under 3 hours on a single A100 GPU, while remaining stable during training. We also provide a lightweight and easy-to-use implementation of FastTD3 to accelerate RL research in robotics. 

**Abstract (ZH)**: 快速TD3算法：一种简单、快速且强大的强化学习方法及其在机器人领域的应用 

---
# LabUtopia: High-Fidelity Simulation and Hierarchical Benchmark for Scientific Embodied Agents 

**Title (ZH)**: LabUtopia: 高保真模拟与科学体现代理分层基准测试 

**Authors**: Rui Li, Zixuan Hu, Wenxi Qu, Jinouwen Zhang, Zhenfei Yin, Sha Zhang, Xuantuo Huang, Hanqing Wang, Tai Wang, Jiangmiao Pang, Wanli Ouyang, Lei Bai, Wangmeng Zuo, Ling-Yu Duan, Dongzhan Zhou, Shixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22634)  

**Abstract**: Scientific embodied agents play a crucial role in modern laboratories by automating complex experimental workflows. Compared to typical household environments, laboratory settings impose significantly higher demands on perception of physical-chemical transformations and long-horizon planning, making them an ideal testbed for advancing embodied intelligence. However, its development has been long hampered by the lack of suitable simulator and benchmarks. In this paper, we address this gap by introducing LabUtopia, a comprehensive simulation and benchmarking suite designed to facilitate the development of generalizable, reasoning-capable embodied agents in laboratory settings. Specifically, it integrates i) LabSim, a high-fidelity simulator supporting multi-physics and chemically meaningful interactions; ii) LabScene, a scalable procedural generator for diverse scientific scenes; and iii) LabBench, a hierarchical benchmark spanning five levels of complexity from atomic actions to long-horizon mobile manipulation. LabUtopia supports 30 distinct tasks and includes more than 200 scene and instrument assets, enabling large-scale training and principled evaluation in high-complexity environments. We demonstrate that LabUtopia offers a powerful platform for advancing the integration of perception, planning, and control in scientific-purpose agents and provides a rigorous testbed for exploring the practical capabilities and generalization limits of embodied intelligence in future research. 

**Abstract (ZH)**: 科学具身代理在现代实验室中通过自动化复杂实验流程扮演关键角色。与典型的家庭环境相比，实验室环境对物理-化学转换的感知和长期规划提出了更高的要求，使其成为推动具身智能发展的理想试验床。然而，其发展一直受限于合适的模拟器和基准测试的缺乏。本文通过引入LabUtopia——一个全面的模拟和基准测试套件——来解决这一问题，旨在促进实验室环境中可泛化的、具备推理能力的具身代理的发展。具体而言，LabUtopia 综合了：i) LabSim，一个支持多物理现象和化学意义交互的高保真模拟器；ii) LabScene，一个可扩展的程序生成器，用于生成多样的科学场景；以及 iii) LabBench，一个跨越五个复杂层次的层次化基准，从原子动作到长时间移动操作。LabUtopia 支持30个不同的任务，包括超过200个场景和仪器资产，能够实现大规模训练并在高复杂度环境中进行有原则的评估。我们证明，LabUtopia 提供了一个强大的平台，用于推进旨在科学目的的代理中感知、规划和控制的整合，并为探索未来研究中具身智能的实用性能力和泛化极限提供了严格的测试床。 

---
# From Strangers to Assistants: Fast Desire Alignment for Embodied Agent-User Adaptation 

**Title (ZH)**: 从陌生人到助手：快速欲望对齐以适应具身智能体-用户适应性 

**Authors**: Yuanfei Wang, Xinju Huang, Fangwei Zhong, Yaodong Yang, Yizhou Wang, Yuanpei Chen, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.22503)  

**Abstract**: While embodied agents have made significant progress in performing complex physical tasks, real-world applications demand more than pure task execution. The agents must collaborate with unfamiliar agents and human users, whose goals are often vague and implicit. In such settings, interpreting ambiguous instructions and uncovering underlying desires is essential for effective assistance. Therefore, fast and accurate desire alignment becomes a critical capability for embodied agents. In this work, we first develop a home assistance simulation environment HA-Desire that integrates an LLM-driven human user agent exhibiting realistic value-driven goal selection and communication. The ego agent must interact with this proxy user to infer and adapt to the user's latent desires. To achieve this, we present a novel framework FAMER for fast desire alignment, which introduces a desire-based mental reasoning mechanism to identify user intent and filter desire-irrelevant actions. We further design a reflection-based communication module that reduces redundant inquiries, and incorporate goal-relevant information extraction with memory persistence to improve information reuse and reduce unnecessary exploration. Extensive experiments demonstrate that our framework significantly enhances both task execution and communication efficiency, enabling embodied agents to quickly adapt to user-specific desires in complex embodied environments. 

**Abstract (ZH)**: 虽然嵌入式代理在执行复杂物理任务方面取得了显著进展，但在实际应用中，要求不仅仅是纯粹的任务执行。这些代理还必须与不熟悉的合作代理和人类用户进行协作，而这些用户的目的是往往模糊且隐含的。在这种环境中，理解模棱两可的指示和揭示潜在的欲望对于有效协助至关重要。因此，快速且准确的欲望对齐成为嵌入式代理的关键能力。在本文中，我们首先开发了一个家庭辅助模拟环境HA-Desire，该环境整合了一个由LLM驱动的人类用户代理，表现出基于现实价值的目标选择和沟通。主体代理必须与这个代理用户进行互动，以推断并适应用户的潜在欲望。为此，我们提出了一种新颖的框架FAMER，该框架引入了一种基于欲望的心理推理机制，以识别用户意图并过滤掉与欲望无关的动作。我们进一步设计了一种基于反思的通信模块，减少了冗余查询，并结合了与目标相关的信息提取与记忆持久化，以提高信息重用并减少不必要的探索。广泛的实验表明，我们的框架显著提高了任务执行和沟通效率，使嵌入式代理能够在复杂的身体环境中迅速适应用户的特定欲望。 

---
# ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation 

**Title (ZH)**: ForceVLA: 增强VLA模型的力感知MoE以实现接触丰富的操作 

**Authors**: Jiawen Yu, Hairuo Liu, Qiaojun Yu, Jieji Ren, Ce Hao, Haitong Ding, Guangyu Huang, Guofan Huang, Yan Song, Panpan Cai, Cewu Lu, Wenqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22159)  

**Abstract**: Vision-Language-Action (VLA) models have advanced general-purpose robotic manipulation by leveraging pretrained visual and linguistic representations. However, they struggle with contact-rich tasks that require fine-grained control involving force, especially under visual occlusion or dynamic uncertainty. To address these limitations, we propose \textbf{ForceVLA}, a novel end-to-end manipulation framework that treats external force sensing as a first-class modality within VLA systems. ForceVLA introduces \textbf{FVLMoE}, a force-aware Mixture-of-Experts fusion module that dynamically integrates pretrained visual-language embeddings with real-time 6-axis force feedback during action decoding. This enables context-aware routing across modality-specific experts, enhancing the robot's ability to adapt to subtle contact dynamics. We also introduce \textbf{ForceVLA-Data}, a new dataset comprising synchronized vision, proprioception, and force-torque signals across five contact-rich manipulation tasks. ForceVLA improves average task success by 23.2\% over strong $\pi_0$-based baselines, achieving up to 80\% success in tasks such as plug insertion. Our approach highlights the importance of multimodal integration for dexterous manipulation and sets a new benchmark for physically intelligent robotic control. Code and data will be released at this https URL. 

**Abstract (ZH)**: 基于视觉-语言-动作的力感知框架 (ForceVLA)：面向接触丰富任务的端到端 manipulation 方法 

---
# A simulation framework for autonomous lunar construction work 

**Title (ZH)**: 自主月球建设工作的仿真框架 

**Authors**: Mattias Linde, Daniel Lindmark, Sandra Ålstig, Martin Servin  

**Link**: [PDF](https://arxiv.org/pdf/2505.22091)  

**Abstract**: We present a simulation framework for lunar construction work involving multiple autonomous machines. The framework supports modelling of construction scenarios and autonomy solutions, execution of the scenarios in simulation, and analysis of work time and energy consumption throughout the construction project. The simulations are based on physics-based models for contacting multibody dynamics and deformable terrain, including vehicle-soil interaction forces and soil flow in real time. A behaviour tree manages the operational logic and error handling, which enables the representation of complex behaviours through a discrete set of simpler tasks in a modular hierarchical structure. High-level decision-making is separated from lower-level control algorithms, with the two connected via ROS2. Excavation movements are controlled through inverse kinematics and tracking controllers. The framework is tested and demonstrated on two different lunar construction scenarios. 

**Abstract (ZH)**: 一种用于月球建设项目中多自主机器施工的仿真框架 

---
# Learning Compositional Behaviors from Demonstration and Language 

**Title (ZH)**: 从示范和语言学习组合行为 

**Authors**: Weiyu Liu, Neil Nie, Ruohan Zhang, Jiayuan Mao, Jiajun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21981)  

**Abstract**: We introduce Behavior from Language and Demonstration (BLADE), a framework for long-horizon robotic manipulation by integrating imitation learning and model-based planning. BLADE leverages language-annotated demonstrations, extracts abstract action knowledge from large language models (LLMs), and constructs a library of structured, high-level action representations. These representations include preconditions and effects grounded in visual perception for each high-level action, along with corresponding controllers implemented as neural network-based policies. BLADE can recover such structured representations automatically, without manually labeled states or symbolic definitions. BLADE shows significant capabilities in generalizing to novel situations, including novel initial states, external state perturbations, and novel goals. We validate the effectiveness of our approach both in simulation and on real robots with a diverse set of objects with articulated parts, partial observability, and geometric constraints. 

**Abstract (ZH)**: 基于语言与演示的行为学习框架：长时Horizon机器人操作（BLADE） 

---
# DORAEMON: Decentralized Ontology-aware Reliable Agent with Enhanced Memory Oriented Navigation 

**Title (ZH)**: DORAEMON：去中心化本体意识可靠智能体及其增强记忆导向导航 

**Authors**: Tianjun Gu, Linfeng Li, Xuhong Wang, Chenghua Gong, Jingyu Gong, Zhizhong Zhang, Yuan Xie, Lizhuang Ma, Xin Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.21969)  

**Abstract**: Adaptive navigation in unfamiliar environments is crucial for household service robots but remains challenging due to the need for both low-level path planning and high-level scene understanding. While recent vision-language model (VLM) based zero-shot approaches reduce dependence on prior maps and scene-specific training data, they face significant limitations: spatiotemporal discontinuity from discrete observations, unstructured memory representations, and insufficient task understanding leading to navigation failures. We propose DORAEMON (Decentralized Ontology-aware Reliable Agent with Enhanced Memory Oriented Navigation), a novel cognitive-inspired framework consisting of Ventral and Dorsal Streams that mimics human navigation capabilities. The Dorsal Stream implements the Hierarchical Semantic-Spatial Fusion and Topology Map to handle spatiotemporal discontinuities, while the Ventral Stream combines RAG-VLM and Policy-VLM to improve decision-making. Our approach also develops Nav-Ensurance to ensure navigation safety and efficiency. We evaluate DORAEMON on the HM3D, MP3D, and GOAT datasets, where it achieves state-of-the-art performance on both success rate (SR) and success weighted by path length (SPL) metrics, significantly outperforming existing methods. We also introduce a new evaluation metric (AORI) to assess navigation intelligence better. Comprehensive experiments demonstrate DORAEMON's effectiveness in zero-shot autonomous navigation without requiring prior map building or pre-training. 

**Abstract (ZH)**: 自适应导航在 unfamiliar environments 中对于家庭服务机器人至关重要，但因其需要低级路径规划和高级场景理解而仍然具有挑战性。虽然基于视觉-语言模型（VLM）的零-shot 方法减少了对先验地图和场景特定训练数据的依赖，但它们面临显著的局限性：来自离散观测的时间空间不连续性、无结构的内存表示以及任务理解不足导致的导航失败。我们提出 DORAEMON（Decentralized Ontology-aware Reliable Agent with Enhanced Memory Oriented Navigation），一种新的认知启发式框架，由腹侧流和背侧流组成，模仿人类的导航能力。背侧流实现层次语义-空间融合和拓扑图以处理时间空间不连续性，而腹侧流结合了RAG-VLM和Policy-VLM以提高决策能力。我们的方法还开发了Nav-Ensurance以确保导航的安全性和效率。我们在HM3D、MP3D和GOAT数据集上评估了DORAEMON，其在成功率（SR）和路径长度加权成功率（SPL）指标上达到最佳性能，显著优于现有方法。我们还引入了一个新的评估指标（AORI）来更好地评估导航智能。综合实验表明，DORAEMON能够在无需构建先验地图或预训练的情况下实现零-shot 自主导航。 

---
# Mastering Agile Tasks with Limited Trials 

**Title (ZH)**: 掌握受限实验次数的敏捷任务 

**Authors**: Yihang Hu, Pingyue Sheng, Shengjie Wang, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.21916)  

**Abstract**: Embodied robots nowadays can already handle many real-world manipulation tasks. However, certain other real-world tasks (e.g., shooting a basketball into a hoop) are highly agile and require high execution precision, presenting additional challenges for methods primarily designed for quasi-static manipulation tasks. This leads to increased efforts in costly data collection, laborious reward design, or complex motion planning. Such tasks, however, are far less challenging for humans. Say a novice basketball player typically needs only $\sim$10 attempts to make their first successful shot, by roughly imitating a motion prior and then iteratively adjusting their motion based on the past outcomes. Inspired by this human learning paradigm, we propose the Adaptive Diffusion Action Plannin (ADAP) algorithm, a simple & scalable approach which iteratively refines its action plan by few real-world trials within a learned prior motion pattern, until reaching a specific goal. Experiments demonstrated that ADAP can learn and accomplish a wide range of goal-conditioned agile dynamic tasks with human-level precision and efficiency directly in real-world, such as throwing a basketball into the hoop in fewer than 10 trials. Project website:this https URL . 

**Abstract (ZH)**: 当前的具身机器人已经能够处理许多实际世界的操作任务。然而，某些其他实际世界的任务（如投篮入筐）要求极高的敏捷性和执行精度，给主要设计用于准静态操作任务的方法带来了额外挑战。这导致了在数据收集、奖励设计或复杂运动规划方面增加了更多成本和努力。然而，对于人类来说，这些任务远不具挑战性。一名初学者篮球运动员通常只需要大约10次尝试就能成功投进第一个球，通过大致模仿一个动作模式，然后根据过去的尝试结果进行迭代调整。受人类学习范式的启发，我们提出了一种自适应扩散动作规划（ADAP）算法，这是一种简单且可扩展的方法，它在学习到的先验动作模式内通过几次实际世界的试错来逐步细化其行动策略，直到达到特定目标。实验表明，ADAP可以直接在真实世界中学习并高效地完成一系列具有人类水平精度的敏捷动态任务，例如在不到10次尝试内投篮入筐。项目网址：this https URL。 

---
# Vision-Language-Action Model with Open-World Embodied Reasoning from Pretrained Knowledge 

**Title (ZH)**: 具有开放世界体.maxLength限制了我只能输出最基本的内容，因此标题翻译如下：

基于预训练知识的视觉-语言-行动模型与开放世界体化推理 

**Authors**: Zhongyi Zhou, Yichen Zhu, Junjie Wen, Chaomin Shen, Yi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21906)  

**Abstract**: Vision-language-action (VLA) models have emerged as the next generation of models in robotics. However, despite leveraging powerful pre-trained Vision-Language Models (VLMs), existing end-to-end VLA systems often lose key capabilities during fine-tuning as the model adapts to specific robotic tasks. We argue that a generalizable VLA model should retain and expand upon the VLM's core competencies: 1) Open-world embodied reasoning - the VLA should inherit the knowledge from VLM, i.e., recognize anything that the VLM can recognize, capable of solving math problems, possessing visual-spatial intelligence, 2) Reasoning following - effectively translating the open-world reasoning into actionable steps for the robot. In this work, we introduce ChatVLA-2, a novel mixture-of-expert VLA model coupled with a specialized three-stage training pipeline designed to preserve the VLM's original strengths while enabling actionable reasoning. To validate our approach, we design a math-matching task wherein a robot interprets math problems written on a whiteboard and picks corresponding number cards from a table to solve equations. Remarkably, our method exhibits exceptional mathematical reasoning and OCR capabilities, despite these abilities not being explicitly trained within the VLA. Furthermore, we demonstrate that the VLA possesses strong spatial reasoning skills, enabling it to interpret novel directional instructions involving previously unseen objects. Overall, our method showcases reasoning and comprehension abilities that significantly surpass state-of-the-art imitation learning methods such as OpenVLA, DexVLA, and pi-zero. This work represents a substantial advancement toward developing truly generalizable robotic foundation models endowed with robust reasoning capacities. 

**Abstract (ZH)**: Vision-语言-行动（VLA）模型已成为机器人领域的下一代模型。然而，尽管利用了强大的预训练视觉-语言模型（VLMs），现有的端到端VLA系统在微调过程中往往会失去一些关键能力，随着模型适应特定的机器人任务，这些能力往往会消失。我们认为，可泛化的VLA模型应当保留并扩展VLM的核心能力：1）开放世界体像是推理——VLA應继承VLM的知识，即识别VLM能识别的一切内容，能够解决数学问题，具备视觉-空间智能；2）跟随推理——有效将开放世界推理转化为可供机器人执行的操作步骤。在这项工作中，我们引入了ChatVLA-2，这是一种新颖的专家混合VLA模型，配备了专门的三阶段训练流程，旨在保留VLM的原始优势，同时使模型具备可执行的推理能力。为了验证我们的方法，我们设计了一个数学配对任务，其中机器人解读写在白板上的数学问题，并从桌子上选取相应数字卡片解决问题。令人惊讶的是，尽管这些能力未在VLA中显式训练，我们的方法仍表现出突出的数学推理能力和光学字符识别（OCR）能力。此外，我们展示了VLA具备强大的空间推理能力，能够解释涉及未见过物体的新型方向指令。总体而言，我们的方法展现了远超现有的模仿学习方法（如OpenVLA、DexVLA和pi-zero）的推理和理解能力。这项工作代表了朝着开发具备强大推理能力的真正可泛化机器人基础模型迈出的重要一步。 

---
# DexUMI: Using Human Hand as the Universal Manipulation Interface for Dexterous Manipulation 

**Title (ZH)**: DexUMI: 使用人类手部作为通用操作界面进行灵巧操作 

**Authors**: Mengda Xu, Han Zhang, Yifan Hou, Zhenjia Xu, Linxi Fan, Manuela Veloso, Shuran Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.21864)  

**Abstract**: We present DexUMI - a data collection and policy learning framework that uses the human hand as the natural interface to transfer dexterous manipulation skills to various robot hands. DexUMI includes hardware and software adaptations to minimize the embodiment gap between the human hand and various robot hands. The hardware adaptation bridges the kinematics gap using a wearable hand exoskeleton. It allows direct haptic feedback in manipulation data collection and adapts human motion to feasible robot hand motion. The software adaptation bridges the visual gap by replacing the human hand in video data with high-fidelity robot hand inpainting. We demonstrate DexUMI's capabilities through comprehensive real-world experiments on two different dexterous robot hand hardware platforms, achieving an average task success rate of 86%. 

**Abstract (ZH)**: DexUMI - 一种以人类手部作為自然接口，將靈巧操作技能轉移至各種機械手的数据采集与策略学习框架 

---
# MIND-Stack: Modular, Interpretable, End-to-End Differentiability for Autonomous Navigation 

**Title (ZH)**: MIND-Stack: 模块化、可解释、端到端可微分自主导航 

**Authors**: Felix Jahncke, Johannes Betz  

**Link**: [PDF](https://arxiv.org/pdf/2505.21734)  

**Abstract**: Developing robust, efficient navigation algorithms is challenging. Rule-based methods offer interpretability and modularity but struggle with learning from large datasets, while end-to-end neural networks excel in learning but lack transparency and modularity. In this paper, we present MIND-Stack, a modular software stack consisting of a localization network and a Stanley Controller with intermediate human interpretable state representations and end-to-end differentiability. Our approach enables the upstream localization module to reduce the downstream control error, extending its role beyond state estimation. Unlike existing research on differentiable algorithms that either lack modules of the autonomous stack to span from sensor input to actuator output or real-world implementation, MIND-Stack offers both capabilities. We conduct experiments that demonstrate the ability of the localization module to reduce the downstream control loss through its end-to-end differentiability while offering better performance than state-of-the-art algorithms. We showcase sim-to-real capabilities by deploying the algorithm on a real-world embedded autonomous platform with limited computation power and demonstrate simultaneous training of both the localization and controller towards one goal. While MIND-Stack shows good results, we discuss the incorporation of additional modules from the autonomous navigation pipeline in the future, promising even greater stability and performance in the next iterations of the framework. 

**Abstract (ZH)**: 开发稳健、高效的导航算法具有挑战性。基于规则的方法提供了可解释性和模块化，但难以从大型数据集中学习，而端到端神经网络在学习方面表现出色，但在透明性和模块化方面存在不足。在本文中，我们提出了一种模块化软件栈MIND-Stack，其中包括一个定位网络和斯坦利控制器，并且具有中间的人类可解释状态表示和端到端可微性。我们的方法使上游定位模块能够减少下游控制误差，使其角色超越了状态估计。与现有基于可微性算法的研究相比，MIND-Stack不仅涵盖了从传感器输入到执行器输出的自主栈模块，还能够在实际应用中实施。我们进行了实验，证明了定位模块通过其端到端可微性减少下游控制损失的能力，并展示了与最先进的算法相比更好的性能。我们通过在具有有限计算能力的实际嵌入式自主平台上部署算法展示了其从仿真到现实的 capabilities，并展示了同时对定位和控制器进行训练以实现一个目标的可能性。虽然MIND-Stack显示了良好的结果，但我们讨论了未来将自主导航管道中的其他模块纳入的可能性，这将在框架的后续迭代中实现更加稳定和高性能。 

---
# Real-World Deployment of Cloud Autonomous Mobility System Using 5G Networks for Outdoor and Indoor Environments 

**Title (ZH)**: 基于5G网络的云自主移动系统在室内外环境中的实际部署 

**Authors**: Yufeng Yang, Minghao Ning, Keqi Shu, Aladdin Saleh, Ehsan Hashemi, Amir Khajepour  

**Link**: [PDF](https://arxiv.org/pdf/2505.21676)  

**Abstract**: The growing complexity of both outdoor and indoor mobility systems demands scalable, cost-effective, and reliable perception and communication frameworks. This work presents the real-world deployment and evaluation of a Cloud Autonomous Mobility (CAM) system that leverages distributed sensor nodes connected via 5G networks, which integrates LiDAR- and camera-based perception at infrastructure units, cloud computing for global information fusion, and Ultra-Reliable Low Latency Communications (URLLC) to enable real-time situational awareness and autonomous operation. The CAM system is deployed in two distinct environments: a dense urban roundabout and a narrow indoor hospital corridor. Field experiments show improved traffic monitoring, hazard detection, and asset management capabilities. The paper also discusses practical deployment challenges and shares key insights for scaling CAM systems. The results highlight the potential of cloud-based infrastructure perception to advance both outdoor and indoor intelligent transportation systems. 

**Abstract (ZH)**: 不断增长的室内外移动系统复杂性要求具备扩展性、成本效益和可靠性的感知与通信框架。本文介绍了基于5G网络连接分布式传感器节点的Cloud Autonomous Mobility (CAM)系统的实际部署与评估，该系统集成了基础设施单位的LiDAR-和相机感知、基于云的全球信息融合以及超可靠低时延通信（URLLC），以实现实时态势感知和自主运行。CAM系统在两个不同的环境中部署：一个密集的城市立交桥和一个狭窄的室内医院走廊。现场实验展示了改进的交通监控、危险检测和资产管理能力。本文还讨论了实际部署中的挑战，并分享了扩展CAM系统的关键见解。研究结果突显了基于云的基础设施感知在推进室内外智能交通运输系统方面的发展潜力。 

---
# Convergent Functions, Divergent Forms 

**Title (ZH)**: 收敛函数，发散形式 

**Authors**: Hyeonseong Jeon, Ainaz Eftekhar, Aaron Walsman, Kuo-Hao Zeng, Ali Farhadi, Ranjay Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2505.21665)  

**Abstract**: We introduce LOKI, a compute-efficient framework for co-designing morphologies and control policies that generalize across unseen tasks. Inspired by biological adaptation -- where animals quickly adjust to morphological changes -- our method overcomes the inefficiencies of traditional evolutionary and quality-diversity algorithms. We propose learning convergent functions: shared control policies trained across clusters of morphologically similar designs in a learned latent space, drastically reducing the training cost per design. Simultaneously, we promote divergent forms by replacing mutation with dynamic local search, enabling broader exploration and preventing premature convergence. The policy reuse allows us to explore 780$\times$ more designs using 78% fewer simulation steps and 40% less compute per design. Local competition paired with a broader search results in a diverse set of high-performing final morphologies. Using the UNIMAL design space and a flat-terrain locomotion task, LOKI discovers a rich variety of designs -- ranging from quadrupeds to crabs, bipedals, and spinners -- far more diverse than those produced by prior work. These morphologies also transfer better to unseen downstream tasks in agility, stability, and manipulation domains (e.g., 2$\times$ higher reward on bump and push box incline tasks). Overall, our approach produces designs that are both diverse and adaptable, with substantially greater sample efficiency than existing co-design methods. (Project website: this https URL) 

**Abstract (ZH)**: LOKI：一种高效的框架，用于跨未见任务协同设计形态和控制策略 

---
# PartInstruct: Part-level Instruction Following for Fine-grained Robot Manipulation 

**Title (ZH)**: Part级指令跟随：用于精细机器人操作的部件级指令遵循 

**Authors**: Yifan Yin, Zhengtao Han, Shivam Aarya, Jianxin Wang, Shuhang Xu, Jiawei Peng, Angtian Wang, Alan Yuille, Tianmin Shu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21652)  

**Abstract**: Fine-grained robot manipulation, such as lifting and rotating a bottle to display the label on the cap, requires robust reasoning about object parts and their relationships with intended tasks. Despite recent advances in training general-purpose robot manipulation policies guided by language instructions, there is a notable lack of large-scale datasets for fine-grained manipulation tasks with part-level instructions and diverse 3D object instances annotated with part-level labels. In this work, we introduce PartInstruct, the first large-scale benchmark for training and evaluating fine-grained robot manipulation models using part-level instructions. PartInstruct comprises 513 object instances across 14 categories, each annotated with part-level information, and 1302 fine-grained manipulation tasks organized into 16 task classes. Our training set consists of over 10,000 expert demonstrations synthesized in a 3D simulator, where each demonstration is paired with a high-level task instruction, a chain of base part-based skill instructions, and ground-truth 3D information about the object and its parts. Additionally, we designed a comprehensive test suite to evaluate the generalizability of learned policies across new states, objects, and tasks. We evaluated several state-of-the-art robot manipulation approaches, including end-to-end vision-language policy learning and bi-level planning models for robot manipulation on our benchmark. The experimental results reveal that current models struggle to robustly ground part concepts and predict actions in 3D space, and face challenges when manipulating object parts in long-horizon tasks. 

**Abstract (ZH)**: 细粒度机器人操作的鲁棒部分级推理及其大规模数据集PartInstruct 

---
# Fast and Cost-effective Speculative Edge-Cloud Decoding with Early Exits 

**Title (ZH)**: 快速且经济高效的 speculative 边缘-云解码早期退出方法 

**Authors**: Yeshwanth Venkatesha, Souvik Kundu, Priyadarshini Panda  

**Link**: [PDF](https://arxiv.org/pdf/2505.21594)  

**Abstract**: Large Language Models (LLMs) enable various applications on edge devices such as smartphones, wearables, and embodied robots. However, their deployment often depends on expensive cloud-based APIs, creating high operational costs, which limit access for smaller organizations and raise sustainability concerns. Certain LLMs can be deployed on-device, offering a cost-effective solution with reduced latency and improved privacy. Yet, limited computing resources constrain the size and accuracy of models that can be deployed, necessitating a collaborative design between edge and cloud. We propose a fast and cost-effective speculative edge-cloud decoding framework with a large target model on the server and a small draft model on the device. By introducing early exits in the target model, tokens are generated mid-verification, allowing the client to preemptively draft subsequent tokens before final verification, thus utilizing idle time and enhancing parallelism between edge and cloud. Using an NVIDIA Jetson Nano (client) and an A100 GPU (server) with Vicuna-68M (draft) and Llama2-7B (target) models, our method achieves up to a 35% reduction in latency compared to cloud-based autoregressive decoding, with an additional 11% improvement from preemptive drafting. To demonstrate real-world applicability, we deploy our method on the Unitree Go2 quadruped robot using Vision-Language Model (VLM) based control, achieving a 21% speedup over traditional cloud-based autoregressive decoding. These results demonstrate the potential of our framework for real-time LLM and VLM applications on resource-constrained edge devices. 

**Abstract (ZH)**: 面向边缘设备的快速和成本-effective 推测边缘-云解码框架：基于服务器端大型目标模型和设备端小型草稿模型 

---
# CogAD: Cognitive-Hierarchy Guided End-to-End Autonomous Driving 

**Title (ZH)**: CogAD：基于认知层级的端到端自主驾驶 

**Authors**: Zhennan Wang, Jianing Teng, Canqun Xiang, Kangliang Chen, Xing Pan, Lu Deng, Weihao Gu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21581)  

**Abstract**: While end-to-end autonomous driving has advanced significantly, prevailing methods remain fundamentally misaligned with human cognitive principles in both perception and planning. In this paper, we propose CogAD, a novel end-to-end autonomous driving model that emulates the hierarchical cognition mechanisms of human drivers. CogAD implements dual hierarchical mechanisms: global-to-local context processing for human-like perception and intent-conditioned multi-mode trajectory generation for cognitively-inspired planning. The proposed method demonstrates three principal advantages: comprehensive environmental understanding through hierarchical perception, robust planning exploration enabled by multi-level planning, and diverse yet reasonable multi-modal trajectory generation facilitated by dual-level uncertainty modeling. Extensive experiments on nuScenes and Bench2Drive demonstrate that CogAD achieves state-of-the-art performance in end-to-end planning, exhibiting particular superiority in long-tail scenarios and robust generalization to complex real-world driving conditions. 

**Abstract (ZH)**: 端到端自动驾驶中的认知驱动模型CogAD 

---
# Spot-On: A Mixed Reality Interface for Multi-Robot Cooperation 

**Title (ZH)**: Spot-On: 一种用于多机器人协同的混合现实界面 

**Authors**: Tim Engelbracht, Petar Lukovic, Tjark Behrens, Kai Lascheit, René Zurbrügg, Marc Pollefeys, Hermann Blum, Zuria Bauer  

**Link**: [PDF](https://arxiv.org/pdf/2505.22539)  

**Abstract**: Recent progress in mixed reality (MR) and robotics is enabling increasingly sophisticated forms of human-robot collaboration. Building on these developments, we introduce a novel MR framework that allows multiple quadruped robots to operate in semantically diverse environments via a MR interface. Our system supports collaborative tasks involving drawers, swing doors, and higher-level infrastructure such as light switches. A comprehensive user study verifies both the design and usability of our app, with participants giving a "good" or "very good" rating in almost all cases. Overall, our approach provides an effective and intuitive framework for MR-based multi-robot collaboration in complex, real-world scenarios. 

**Abstract (ZH)**: 最近混合现实(MR)和机器人技术的发展使人类与机器人更加复杂的协作成为可能。在此基础上，我们提出了一种新型的MR框架，该框架通过MR接口使多个四足机器人能够在具有语义差异的环境中协同操作。该系统支持涉及抽屉、摆门以及更高层次的基础设施如开关的协作任务。全面的用户研究验证了该应用程序的设计和可用性，参与者中几乎所有人都给予了“良好”或“非常好”的评价。总体而言，我们的方法为基于MR的多机器人协作在复杂的真实场景中提供了有效且直观的框架。 

---
# GeoDrive: 3D Geometry-Informed Driving World Model with Precise Action Control 

**Title (ZH)**: GeoDrive: 基于精确几何信息的三维驾驶世界模型与行动控制 

**Authors**: Anthony Chen, Wenzhao Zheng, Yida Wang, Xueyang Zhang, Kun Zhan, Peng Jia, Kurt Keutzer, Shangbang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22421)  

**Abstract**: Recent advancements in world models have revolutionized dynamic environment simulation, allowing systems to foresee future states and assess potential actions. In autonomous driving, these capabilities help vehicles anticipate the behavior of other road users, perform risk-aware planning, accelerate training in simulation, and adapt to novel scenarios, thereby enhancing safety and reliability. Current approaches exhibit deficiencies in maintaining robust 3D geometric consistency or accumulating artifacts during occlusion handling, both critical for reliable safety assessment in autonomous navigation tasks. To address this, we introduce GeoDrive, which explicitly integrates robust 3D geometry conditions into driving world models to enhance spatial understanding and action controllability. Specifically, we first extract a 3D representation from the input frame and then obtain its 2D rendering based on the user-specified ego-car trajectory. To enable dynamic modeling, we propose a dynamic editing module during training to enhance the renderings by editing the positions of the vehicles. Extensive experiments demonstrate that our method significantly outperforms existing models in both action accuracy and 3D spatial awareness, leading to more realistic, adaptable, and reliable scene modeling for safer autonomous driving. Additionally, our model can generalize to novel trajectories and offers interactive scene editing capabilities, such as object editing and object trajectory control. 

**Abstract (ZH)**: 近期世界模型的发展革新了动态环境的模拟，使系统能够预见未来状态并评估潜在行动。在自动驾驶领域，这些能力帮助车辆预判其他道路使用者的行为，进行风险意识规划，加速模拟训练，并适应新的场景，从而提升安全性和可靠性。现有方法在保持稳健的3D几何一致性或处理遮挡时积累伪影，这些都是可靠的安全评估的关键。为解决这一问题，我们提出了GeoDrive，这是一种将稳健的3D几何条件显式集成到驾驶世界模型中的方法，以增强空间理解和行动可控性。具体而言，我们首先从输入帧中提取3D表示，然后基于用户指定的ego车辆轨迹获取其2D渲染。为实现动态建模，我们在训练期间提出了一个动态编辑模块，通过编辑车辆的位置来增强渲染效果。实验结果表明，我们的方法在动作准确性和3D空间意识方面显著优于现有模型，从而为更安全的自动驾驶提供了更为逼真、适应性强且可靠的场景建模。此外，我们的模型能够泛化到新的轨迹，并提供了交互式场景编辑能力，如对象编辑和对象轨迹控制。 

---
# A Provable Approach for End-to-End Safe Reinforcement Learning 

**Title (ZH)**: 可验证的方法实现端到端安全强化学习 

**Authors**: Akifumi Wachi, Kohei Miyaguchi, Takumi Tanabe, Rei Sato, Youhei Akimoto  

**Link**: [PDF](https://arxiv.org/pdf/2505.21852)  

**Abstract**: A longstanding goal in safe reinforcement learning (RL) is a method to ensure the safety of a policy throughout the entire process, from learning to operation. However, existing safe RL paradigms inherently struggle to achieve this objective. We propose a method, called Provably Lifetime Safe RL (PLS), that integrates offline safe RL with safe policy deployment to address this challenge. Our proposed method learns a policy offline using return-conditioned supervised learning and then deploys the resulting policy while cautiously optimizing a limited set of parameters, known as target returns, using Gaussian processes (GPs). Theoretically, we justify the use of GPs by analyzing the mathematical relationship between target and actual returns. We then prove that PLS finds near-optimal target returns while guaranteeing safety with high probability. Empirically, we demonstrate that PLS outperforms baselines both in safety and reward performance, thereby achieving the longstanding goal to obtain high rewards while ensuring the safety of a policy throughout the lifetime from learning to operation. 

**Abstract (ZH)**: 证明可全程保障安全的强化学习（Provably Lifetime Safe RL） 

---
# Cognitively-Inspired Emergent Communication via Knowledge Graphs for Assisting the Visually Impaired 

**Title (ZH)**: 基于知识图谱的认知启发式 Emergent 通信辅助视障人士 

**Authors**: Ruxiao Chen, Dezheng Han, Wenjie Han, Shuaishuai Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.22087)  

**Abstract**: Assistive systems for visually impaired individuals must deliver rapid, interpretable, and adaptive feedback to facilitate real-time navigation. Current approaches face a trade-off between latency and semantic richness: natural language-based systems provide detailed guidance but are too slow for dynamic scenarios, while emergent communication frameworks offer low-latency symbolic languages but lack semantic depth, limiting their utility in tactile modalities like vibration. To address these limitations, we introduce a novel framework, Cognitively-Inspired Emergent Communication via Knowledge Graphs (VAG-EC), which emulates human visual perception and cognitive mapping. Our method constructs knowledge graphs to represent objects and their relationships, incorporating attention mechanisms to prioritize task-relevant entities, thereby mirroring human selective attention. This structured approach enables the emergence of compact, interpretable, and context-sensitive symbolic languages. Extensive experiments across varying vocabulary sizes and message lengths demonstrate that VAG-EC outperforms traditional emergent communication methods in Topographic Similarity (TopSim) and Context Independence (CI). These findings underscore the potential of cognitively grounded emergent communication as a fast, adaptive, and human-aligned solution for real-time assistive technologies. Code is available at this https URL. 

**Abstract (ZH)**: 基于知识图谱的认知启发式新兴通信（VAG-EC）：促进视障个体的实时导航 

---
# Reinforced Reasoning for Embodied Planning 

**Title (ZH)**: 强化推理在体域规划中的应用 

**Authors**: Di Wu, Jiaxin Fan, Junzhe Zang, Guanbo Wang, Wei Yin, Wenhao Li, Bo Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.22050)  

**Abstract**: Embodied planning requires agents to make coherent multi-step decisions based on dynamic visual observations and natural language goals. While recent vision-language models (VLMs) excel at static perception tasks, they struggle with the temporal reasoning, spatial understanding, and commonsense grounding needed for planning in interactive environments. In this work, we introduce a reinforcement fine-tuning framework that brings R1-style reasoning enhancement into embodied planning. We first distill a high-quality dataset from a powerful closed-source model and perform supervised fine-tuning (SFT) to equip the model with structured decision-making priors. We then design a rule-based reward function tailored to multi-step action quality and optimize the policy via Generalized Reinforced Preference Optimization (GRPO). Our approach is evaluated on Embench, a recent benchmark for interactive embodied tasks, covering both in-domain and out-of-domain scenarios. Experimental results show that our method significantly outperforms models of similar or larger scale, including GPT-4o-mini and 70B+ open-source baselines, and exhibits strong generalization to unseen environments. This work highlights the potential of reinforcement-driven reasoning to advance long-horizon planning in embodied AI. 

**Abstract (ZH)**: 基于感知的规划需要代理基于动态视觉观察和自然语言目标做出连贯的多步决策。尽管近期的跨模态模型在静态感知任务上表现出色，但在需要规划的时间推理、空间理解以及常识绑定的交互环境中，它们仍然存在困难。在本工作中，我们引入了一种强化微调框架，将R1风格的推理增强引入到基于感知的规划中。我们首先从一个强大的闭源模型中提取高质量的数据集，并进行监督微调（SFT）以赋予模型结构化的决策先验。然后，我们设计了一个基于规则的奖励函数，以适应多步行动质量的优化，并通过广义强化偏好优化（GRPO）优化策略。我们的方法在Embench上进行了评估，这是一个最近提出的交互式基于感知任务基准，涵盖了领域内和领域外场景。实验结果表明，我们的方法显著优于类似规模或更大规模的模型，包括GPT-4o-mini和70B+开源基线，并且在未见过的环境中表现出强大的泛化能力。本工作突显了基于强化学习的推理对于推进基于感知的AI长期规划的潜在价值。 

---
# Efficiently Enhancing General Agents With Hierarchical-categorical Memory 

**Title (ZH)**: 高效增强通用代理的层级分类记忆 

**Authors**: Changze Qiao, Mingming Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.22006)  

**Abstract**: With large language models (LLMs) demonstrating remarkable capabilities, there has been a surge in research on leveraging LLMs to build general-purpose multi-modal agents. However, existing approaches either rely on computationally expensive end-to-end training using large-scale multi-modal data or adopt tool-use methods that lack the ability to continuously learn and adapt to new environments. In this paper, we introduce EHC, a general agent capable of learning without parameter updates. EHC consists of a Hierarchical Memory Retrieval (HMR) module and a Task-Category Oriented Experience Learning (TOEL) module. The HMR module facilitates rapid retrieval of relevant memories and continuously stores new information without being constrained by memory capacity. The TOEL module enhances the agent's comprehension of various task characteristics by classifying experiences and extracting patterns across different categories. Extensive experiments conducted on multiple standard datasets demonstrate that EHC outperforms existing methods, achieving state-of-the-art performance and underscoring its effectiveness as a general agent for handling complex multi-modal tasks. 

**Abstract (ZH)**: 基于大规模语言模型的通用多模态代理学习方法：EHC 

---
# 3DLLM-Mem: Long-Term Spatial-Temporal Memory for Embodied 3D Large Language Model 

**Title (ZH)**: 3DLLM-Mem：嵌入式3D大型语言模型的长期空时记忆 

**Authors**: Wenbo Hu, Yining Hong, Yanjun Wang, Leison Gao, Zibu Wei, Xingcheng Yao, Nanyun Peng, Yonatan Bitton, Idan Szpektor, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22657)  

**Abstract**: Humans excel at performing complex tasks by leveraging long-term memory across temporal and spatial experiences. In contrast, current Large Language Models (LLMs) struggle to effectively plan and act in dynamic, multi-room 3D environments. We posit that part of this limitation is due to the lack of proper 3D spatial-temporal memory modeling in LLMs. To address this, we first introduce 3DMem-Bench, a comprehensive benchmark comprising over 26,000 trajectories and 2,892 embodied tasks, question-answering and captioning, designed to evaluate an agent's ability to reason over long-term memory in 3D environments. Second, we propose 3DLLM-Mem, a novel dynamic memory management and fusion model for embodied spatial-temporal reasoning and actions in LLMs. Our model uses working memory tokens, which represents current observations, as queries to selectively attend to and fuse the most useful spatial and temporal features from episodic memory, which stores past observations and interactions. Our approach allows the agent to focus on task-relevant information while maintaining memory efficiency in complex, long-horizon environments. Experimental results demonstrate that 3DLLM-Mem achieves state-of-the-art performance across various tasks, outperforming the strongest baselines by 16.5% in success rate on 3DMem-Bench's most challenging in-the-wild embodied tasks. 

**Abstract (ZH)**: 人类通过利用跨时空经验的长期记忆来执行复杂任务。相比之下，当前的大规模语言模型（LLMs）在动态的多房间3D环境中计划和执行行动方面能力有限。我们认为这一限制部分原因是由于LLMs缺少适当的3D空间-时间记忆建模。为此，我们首先引入了3DMem-Bench，这是一个包含超过26,000条轨迹和2,892个具身任务、问答和描述的全面基准，旨在评估代理在3D环境中的长期记忆推理能力。其次，我们提出了3DLLM-Mem，这是一种新型的动态记忆管理与融合模型，用于LLMs中的具身空间-时间推理和执行动作。我们的模型使用表示当前观测的工怍记忆标记作为查询，选择性地关注并融合 episodic 记忆中最有用的空间和时间特征，episodic 记忆存储了过去的观测和交互。我们的方法使代理能够聚焦于与任务相关的信息，同时在复杂的长期环境中保持记忆效率。实验结果表明，3DLLM-Mem 在多项任务中达到了最先进的性能，在3DMem-Bench最具挑战性的现实世界具身任务中，成功率为16.5%的提升超过了最强基线模型。 

---
# Universal Visuo-Tactile Video Understanding for Embodied Interaction 

**Title (ZH)**: 面向实体交互的通用视听触视频理解 

**Authors**: Yifan Xie, Mingyang Li, Shoujie Li, Xingting Li, Guangyu Chen, Fei Ma, Fei Richard Yu, Wenbo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.22566)  

**Abstract**: Tactile perception is essential for embodied agents to understand physical attributes of objects that cannot be determined through visual inspection alone. While existing approaches have made progress in visual and language modalities for physical understanding, they fail to effectively incorporate tactile information that provides crucial haptic feedback for real-world interaction. In this paper, we present VTV-LLM, the first multi-modal large language model for universal Visuo-Tactile Video (VTV) understanding that bridges the gap between tactile perception and natural language. To address the challenges of cross-sensor and cross-modal integration, we contribute VTV150K, a comprehensive dataset comprising 150,000 video frames from 100 diverse objects captured across three different tactile sensors (GelSight Mini, DIGIT, and Tac3D), annotated with four fundamental tactile attributes (hardness, protrusion, elasticity, and friction). We develop a novel three-stage training paradigm that includes VTV enhancement for robust visuo-tactile representation, VTV-text alignment for cross-modal correspondence, and text prompt finetuning for natural language generation. Our framework enables sophisticated tactile reasoning capabilities including feature assessment, comparative analysis, scenario-based decision making and so on. Experimental evaluations demonstrate that VTV-LLM achieves superior performance in tactile video understanding tasks, establishing a foundation for more intuitive human-machine interaction in tactile domains. 

**Abstract (ZH)**: 触觉感知对于理解视觉检查无法确定的物体物理属性是 essential 的，对于具身智能体来说至关重要。虽然现有方法在视觉和语言模态下的物理理解方面取得了进展，但它们未能有效整合提供关键触觉反馈的触觉信息，以促进实际交互。本文介绍了 VTV-LLM，这是首个用于通用视觉-触觉视频 (VTV) 理解的大规模多模态语言模型，它弥合了触觉感知与自然语言之间的差距。为了解决跨传感器和跨模态整合的挑战，我们贡献了 VTV150K 数据集，该数据集包含来自 100 个不同物体的 150,000 个视频帧，这些物体跨越了三个不同的触觉传感器（GelSight Mini、DIGIT 和 Tac3D），并标注了四种基本的触觉属性（硬度、突出度、弹性、摩擦力）。我们开发了一种新颖的三阶段训练范式，包括 VTV 增强以实现稳健的视觉-触觉表示、VTV 文本对齐以实现跨模态对应，以及文本提示微调以实现自然语言生成。我们的框架 enables 复杂的触觉推理能力，包括特征评估、比较分析、基于场景的决策等。实验评估表明，VTV-LLM 在触觉视频理解任务中表现出优越的性能，为触觉领域更直观的人机交互奠定了基础。 

---
# Training RL Agents for Multi-Objective Network Defense Tasks 

**Title (ZH)**: 训练面向多目标网络防御任务的RL代理 

**Authors**: Andres Molina-Markham, Luis Robaina, Sean Steinle, Akash Trivedi, Derek Tsui, Nicholas Potteiger, Lauren Brandt, Ransom Winder, Ahmed Ridley  

**Link**: [PDF](https://arxiv.org/pdf/2505.22531)  

**Abstract**: Open-ended learning (OEL) -- which emphasizes training agents that achieve broad capability over narrow competency -- is emerging as a paradigm to develop artificial intelligence (AI) agents to achieve robustness and generalization. However, despite promising results that demonstrate the benefits of OEL, applying OEL to develop autonomous agents for real-world cybersecurity applications remains a challenge.
We propose a training approach, inspired by OEL, to develop autonomous network defenders. Our results demonstrate that like in other domains, OEL principles can translate into more robust and generalizable agents for cyber defense. To apply OEL to network defense, it is necessary to address several technical challenges. Most importantly, it is critical to provide a task representation approach over a broad universe of tasks that maintains a consistent interface over goals, rewards and action spaces. This way, the learning agent can train with varying network conditions, attacker behaviors, and defender goals while being able to build on previously gained knowledge.
With our tools and results, we aim to fundamentally impact research that applies AI to solve cybersecurity problems. Specifically, as researchers develop gyms and benchmarks for cyber defense, it is paramount that they consider diverse tasks with consistent representations, such as those we propose in our work. 

**Abstract (ZH)**: 开放性学习（OEL）——强调训练能够在广泛能力而非狭窄专长上取得成就的智能体——正在成为开发具备稳健性和泛化能力的人工智能代理的范式。然而，尽管开放性学习展示了显著的好处，将其应用于开发用于现实世界网络安全应用的自主代理仍面临挑战。 

---
# SOReL and TOReL: Two Methods for Fully Offline Reinforcement Learning 

**Title (ZH)**: SOReL 和 TOReL：两种完全离线强化学习方法 

**Authors**: Mattie Fellows, Clarisse Wibault, Uljad Berdica, Johannes Forkel, Jakob N. Foerster, Michael A. Osborne  

**Link**: [PDF](https://arxiv.org/pdf/2505.22442)  

**Abstract**: Sample efficiency remains a major obstacle for real world adoption of reinforcement learning (RL): success has been limited to settings where simulators provide access to essentially unlimited environment interactions, which in reality are typically costly or dangerous to obtain. Offline RL in principle offers a solution by exploiting offline data to learn a near-optimal policy before deployment. In practice, however, current offline RL methods rely on extensive online interactions for hyperparameter tuning, and have no reliable bound on their initial online performance. To address these two issues, we introduce two algorithms. Firstly, SOReL: an algorithm for safe offline reinforcement learning. Using only offline data, our Bayesian approach infers a posterior over environment dynamics to obtain a reliable estimate of the online performance via the posterior predictive uncertainty. Crucially, all hyperparameters are also tuned fully offline. Secondly, we introduce TOReL: a tuning for offline reinforcement learning algorithm that extends our information rate based offline hyperparameter tuning methods to general offline RL approaches. Our empirical evaluation confirms SOReL's ability to accurately estimate regret in the Bayesian setting whilst TOReL's offline hyperparameter tuning achieves competitive performance with the best online hyperparameter tuning methods using only offline data. Thus, SOReL and TOReL make a significant step towards safe and reliable offline RL, unlocking the potential for RL in the real world. Our implementations are publicly available: this https URL\_torel. 

**Abstract (ZH)**: 安全的离线强化学习：SOReL和TOReL算法 

---
# Voice CMS: updating the knowledge base of a digital assistant through conversation 

**Title (ZH)**: Voice CMS：通过对话更新数字助理的知识库 

**Authors**: Grzegorz Wolny, Michał Szczerbak  

**Link**: [PDF](https://arxiv.org/pdf/2505.22303)  

**Abstract**: In this study, we propose a solution based on a multi-agent LLM architecture and a voice user interface (VUI) designed to update the knowledge base of a digital assistant. Its usability is evaluated in comparison to a more traditional graphical content management system (CMS), with a focus on understanding the relationship between user preferences and the complexity of the information being provided. The findings demonstrate that, while the overall usability of the VUI is rated lower than the graphical interface, it is already preferred by users for less complex tasks. Furthermore, the quality of content entered through the VUI is comparable to that achieved with the graphical interface, even for highly complex tasks. Obtained qualitative results suggest that a hybrid interface combining the strengths of both approaches could address the key challenges identified during the experiment, such as reducing cognitive load through graphical feedback while maintaining the intuitive nature of voice-based interactions. This work highlights the potential of conversational interfaces as a viable and effective method for knowledge management in specific business contexts. 

**Abstract (ZH)**: 本研究提出一种基于多代理语言模型架构和语音用户界面（VUI）的解决方案，旨在更新数字助手的知识库。并与传统的图形内容管理系统（CMS）进行 usability 评估，重点关注用户偏好与提供信息复杂性之间的关系。研究发现，尽管 VUI 的整体 usability 评分低于图形界面，但它已在复杂度较低的任务中更受用户青睐。此外，通过 VUI 输入的内容质量与图形界面相当，即使对于高度复杂任务也是如此。获得的定性结果表明，结合两种方法优点的混合界面可能解决实验中识别的关键挑战，如通过图形反馈减轻认知负担同时维持基于语音交互的直观性。本研究突显了对话界面在特定商业背景下作为知识管理可行且有效方法的潜力。 

---
# Flexible Tool Selection through Low-dimensional Attribute Alignment of Vision and Language 

**Title (ZH)**: 视觉与语言低维度属性对齐驱动的灵活工具选择 

**Authors**: Guangfu Hao, Haojie Wen, Liangxuna Guo, Yang Chen, Yanchao Bi, Shan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.22146)  

**Abstract**: Flexible tool selection reflects a complex cognitive ability that distinguishes humans from other species, yet computational models that capture this ability remain underdeveloped. We developed a framework using low-dimensional attribute representations to bridge visual tool perception and linguistic task understanding. We constructed a comprehensive dataset (ToolNet) containing 115 common tools labeled with 13 carefully designed attributes spanning physical, functional, and psychological properties, paired with natural language scenarios describing tool usage. Visual encoders (ResNet or ViT) extract attributes from tool images while fine-tuned language models (GPT-2, LLaMA, DeepSeek) derive required attributes from task descriptions. Our approach achieves 74% accuracy in tool selection tasks-significantly outperforming direct tool matching (20%) and smaller multimodal models (21%-58%), while approaching performance of much larger models like GPT-4o (73%) with substantially fewer parameters. Ablation studies revealed that manipulation-related attributes (graspability, hand-relatedness, elongation) consistently prove most critical across modalities. This work provides a parameter-efficient, interpretable solution that mimics human-like tool cognition, advancing both cognitive science understanding and practical applications in tool selection tasks. 

**Abstract (ZH)**: 灵活工具选择反映了一种复杂的认知能力，这种能力使人类与其他物种区分开来，但能够捕获这一能力的计算模型仍不够发达。我们提出了一种使用低维属性表示的框架，以连接视觉工具感知和语言任务理解。我们构建了一个包含115种常用工具的数据集（ToolNet），这些工具被标记了13个精心设计的属性，涵盖了物理、功能和心理属性，并配以自然语言场景描述工具的使用。视觉编码器（ResNet或ViT）从工具图像中提取属性，微调的语言模型（GPT-2、LLaMA、DeepSeek）从任务描述中推导出所需属性。我们的方法在工具选择任务中的准确率达到74%，显著优于直接工具匹配（20%）和较小的多模态模型（21%-58%），同时参数量较少的情况下接近如GPT-4o（73%）等更大模型的性能。消融研究显示，与操作相关的属性（握持性、与手的相关性、长度）在各个模态中始终证明是最关键的。本工作提供了一种参数高效、可解释的解决方案，能够模拟人类似的工具认知，既推进了认知科学的理解，又在工具选择任务的实际应用中取得了进展。 

---
# VRAG-RL: Empower Vision-Perception-Based RAG for Visually Rich Information Understanding via Iterative Reasoning with Reinforcement Learning 

**Title (ZH)**: VRAG-RL: 基于视觉感知的图形聚合模型通过强化学习迭代推理理解丰富视觉信息 

**Authors**: Qiuchen Wang, Ruixue Ding, Yu Zeng, Zehui Chen, Lin Chen, Shihang Wang, Pengjun Xie, Fei Huang, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.22019)  

**Abstract**: Effectively retrieving, reasoning and understanding visually rich information remains a challenge for RAG methods. Traditional text-based methods cannot handle visual-related information. On the other hand, current vision-based RAG approaches are often limited by fixed pipelines and frequently struggle to reason effectively due to the insufficient activation of the fundamental capabilities of models. As RL has been proven to be beneficial for model reasoning, we introduce VRAG-RL, a novel RL framework tailored for complex reasoning across visually rich information. With this framework, VLMs interact with search engines, autonomously sampling single-turn or multi-turn reasoning trajectories with the help of visual perception tokens and undergoing continual optimization based on these samples. Our approach highlights key limitations of RL in RAG domains: (i) Prior Multi-modal RAG approaches tend to merely incorporate images into the context, leading to insufficient reasoning token allocation and neglecting visual-specific perception; and (ii) When models interact with search engines, their queries often fail to retrieve relevant information due to the inability to articulate requirements, thereby leading to suboptimal performance. To address these challenges, we define an action space tailored for visually rich inputs, with actions including cropping and scaling, allowing the model to gather information from a coarse-to-fine perspective. Furthermore, to bridge the gap between users' original inquiries and the retriever, we employ a simple yet effective reward that integrates query rewriting and retrieval performance with a model-based reward. Our VRAG-RL optimizes VLMs for RAG tasks using specially designed RL strategies, aligning the model with real-world applications. The code is available at \hyperlink{this https URL}{this https URL}. 

**Abstract (ZH)**: 有效地检索、推理和理解富含视觉信息的内容仍然是RAG方法的一个挑战。传统的基于文本的方法无法处理视觉相关信息。另一方面，当前基于视觉的RAG方法往往受限于固定的管道，并且经常由于模型基本能力激活不足而难以有效推理。鉴于强化学习（RL）已被证明有助于模型推理，我们提出了VRAG-RL，这是一种专为处理富含视觉信息的复杂推理而设计的新型RL框架。通过此框架，视觉语言模型能够与搜索引擎自主交互，通过视觉感知令牌的帮助自动采样单轮或多轮的推理轨迹，并基于这些样本进行持续优化。我们的方法突出了RL在RAG领域中的关键局限性：(i) 前沿的多模态RAG方法倾向于仅将图像纳入上下文，导致推理令牌分配不足且忽视了视觉特有的感知；(ii) 当模型与搜索引擎交互时，由于无法准确表达需求，其查询往往难以检索到相关的信息，从而导致性能欠佳。为应对这些挑战，我们定义了一个针对富含视觉信息的输入的动作空间，包括裁剪和缩放等动作，使模型能够从粗略到精细的视角收集信息。此外，为了弥合用户原始询问与检索器之间的差距，我们采用了简单有效的奖励机制，该机制结合了查询重写和检索性能与基于模型的奖励。我们的VRAG-RL利用特别设计的RL策略优化视觉语言模型，使其与实际应用相契合。代码可在 <https://this https URL> 获取。 

---
# Learning World Models for Interactive Video Generation 

**Title (ZH)**: 学习世界模型进行互动视频生成 

**Authors**: Taiye Chen, Xun Hu, Zihan Ding, Chi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.21996)  

**Abstract**: Foundational world models must be both interactive and preserve spatiotemporal coherence for effective future planning with action choices. However, present models for long video generation have limited inherent world modeling capabilities due to two main challenges: compounding errors and insufficient memory mechanisms. We enhance image-to-video models with interactive capabilities through additional action conditioning and autoregressive framework, and reveal that compounding error is inherently irreducible in autoregressive video generation, while insufficient memory mechanism leads to incoherence of world models. We propose video retrieval augmented generation (VRAG) with explicit global state conditioning, which significantly reduces long-term compounding errors and increases spatiotemporal consistency of world models. In contrast, naive autoregressive generation with extended context windows and retrieval-augmented generation prove less effective for video generation, primarily due to the limited in-context learning capabilities of current video models. Our work illuminates the fundamental challenges in video world models and establishes a comprehensive benchmark for improving video generation models with internal world modeling capabilities. 

**Abstract (ZH)**: 基础世界模型必须兼具交互性和时空连贯性，以有效进行带有行动选择的未来规划。然而，当前用于长视频生成的模型由于两大主要挑战——累积误差和记忆机制不足——而具有有限的世界建模能力。我们通过添加动作条件和自回归框架来增强图像到视频模型的交互能力，并揭示了自回归视频生成中累积误差是固有无法减少的，而记忆机制不足导致世界模型不连贯。我们提出了带有显式全局状态条件的视频检索增强生成（VRAG），显著减少了长期累积误差并提高了世界模型的时空一致性。相比之下，扩展上下文窗口的朴素自回归生成和检索增强生成在视频生成中效果较差，主要是由于当前视频模型的有限上下文学习能力。我们的工作揭示了视频世界模型的基本挑战，并建立了一个全面的基准，用于改进具有内在世界建模能力的视频生成模型。 

---
# Deep Reinforcement Learning Agents are not even close to Human Intelligence 

**Title (ZH)**: 深层强化学习代理远未达到人类智能水平 

**Authors**: Quentin Delfosse, Jannis Blüml, Fabian Tatai, Théo Vincent, Bjarne Gregori, Elisabeth Dillies, Jan Peters, Constantin Rothkopf, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2505.21731)  

**Abstract**: Deep reinforcement learning (RL) agents achieve impressive results in a wide variety of tasks, but they lack zero-shot adaptation capabilities. While most robustness evaluations focus on tasks complexifications, for which human also struggle to maintain performances, no evaluation has been performed on tasks simplifications. To tackle this issue, we introduce HackAtari, a set of task variations of the Arcade Learning Environments. We use it to demonstrate that, contrary to humans, RL agents systematically exhibit huge performance drops on simpler versions of their training tasks, uncovering agents' consistent reliance on shortcuts. Our analysis across multiple algorithms and architectures highlights the persistent gap between RL agents and human behavioral intelligence, underscoring the need for new benchmarks and methodologies that enforce systematic generalization testing beyond static evaluation protocols. Training and testing in the same environment is not enough to obtain agents equipped with human-like intelligence. 

**Abstract (ZH)**: 深度强化学习（RL）代理在各种任务中取得了令人印象深刻的成果，但它们缺乏零样本适应能力。虽然大多数鲁棒性评估集中在人类也难以维持性能的任务复杂化上，但尚未对任务简化进行评估。为解决这一问题，我们引入了HackAtari，即 Arcade Learning Environments 的一系列任务变体。我们使用它来展示，与人类不同，RL代理在任务简化版本中系统地表现出巨大的性能下降，揭示了代理对捷径的一贯依赖。我们在多种算法和架构上的分析强调了RL代理与人类行为智能之间持续存在的差距，突出了需要新的基准和方法，以确保超出静态评估协议的系统泛化测试。在相同环境中训练和测试并不足以获得具备人类智能的代理。 

---
# AITEE -- Agentic Tutor for Electrical Engineering 

**Title (ZH)**: AITEE —— 电气工程领域赋能式导师 

**Authors**: Christopher Knievel, Alexander Bernhardt, Christian Bernhardt  

**Link**: [PDF](https://arxiv.org/pdf/2505.21582)  

**Abstract**: Intelligent tutoring systems combined with large language models offer a promising approach to address students' diverse needs and promote self-efficacious learning. While large language models possess good foundational knowledge of electrical engineering basics, they remain insufficiently capable of addressing specific questions about electrical circuits. In this paper, we present AITEE, an agent-based tutoring system for electrical engineering designed to accompany students throughout their learning process, offer individualized support, and promote self-directed learning. AITEE supports both hand-drawn and digital circuits through an adapted circuit reconstruction process, enabling natural interaction with students. Our novel graph-based similarity measure identifies relevant context from lecture materials through a retrieval augmented generation approach, while parallel Spice simulation further enhances accuracy in applying solution methodologies. The system implements a Socratic dialogue to foster learner autonomy through guided questioning. Experimental evaluations demonstrate that AITEE significantly outperforms baseline approaches in domain-specific knowledge application, with even medium-sized LLM models showing acceptable performance. Our results highlight the potential of agentic tutors to deliver scalable, personalized, and effective learning environments for electrical engineering education. 

**Abstract (ZH)**: 智能 tutoring 系统结合大语言模型为满足学生多样化需求和促进自主学习提供了有希望的方法。虽然大语言模型在电气工程基础知识方面具有良好的基础，但在处理关于电气电路的特定问题方面仍存在不足。在本文中，我们介绍了 AITEE，这是一种基于代理的电气工程辅导系统，旨在陪伴学生整个学习过程，提供个性化支持，并促进自主学习。AITEE 通过适应性的电路重建过程支持手绘和数字电路，实现自然的学生交互。我们提出的一种基于图的相似性度量通过检索增强生成方法识别相关上下文，并与并行Spice仿真结合进一步提高解决方案方法的应用准确性。系统采用苏格拉底式对话来通过引导性提问培养学习者的自主性。实验评估表明，AITEE 在特定领域知识应用方面显著优于基线方法，即使中型规模的LLM模型也表现出可接受的性能。我们的结果突显了代理式辅导在为电气工程教育提供可扩展、个性化和有效学习环境方面的潜力。 

---
# Collaborative Agentic AI Needs Interoperability Across Ecosystems 

**Title (ZH)**: 跨生态系统的代理性协作AI需要互操作性 

**Authors**: Rishi Sharma, Martijn de Vos, Pradyumna Chari, Ramesh Raskar, Anne-Marie Kermarrec  

**Link**: [PDF](https://arxiv.org/pdf/2505.21550)  

**Abstract**: Collaborative agentic AI is projected to transform entire industries by enabling AI-powered agents to autonomously perceive, plan, and act within digital environments. Yet, current solutions in this field are all built in isolation, and we are rapidly heading toward a landscape of fragmented, incompatible ecosystems. In this position paper, we argue that interoperability, achieved by the adoption of minimal standards, is essential to ensure open, secure, web-scale, and widely-adopted agentic ecosystems. To this end, we devise a minimal architectural foundation for collaborative agentic AI, named Web of Agents, which is composed of four components: agent-to-agent messaging, interaction interoperability, state management, and agent discovery. Web of Agents adopts existing standards and reuses existing infrastructure where possible. With Web of Agents, we take the first but critical step toward interoperable agentic systems and offer a pragmatic path forward before ecosystem fragmentation becomes the norm. 

**Abstract (ZH)**: 协作代理人工智能预计通过使AI驱动的代理能够在数字环境中自主感知、规划和行动来颠覆整个行业。然而，目前该领域内的解决方案都是独立构建的，我们正迅速走向一个由碎片化且不兼容的生态系统构成的景观。在这种立场论文中，我们认为通过采用最小标准来实现互操作性是确保开放、安全、Web规模且广泛采用的代理生态系统的关键。为此，我们提出了一个协作代理人工智能的最小架构基础，称为代理网络，它由四个组件组成：代理间消息传递、交互互操作性、状态管理以及代理发现。代理网络采用现有标准并在可能的情况下重用现有基础设施。通过代理网络，我们迈出了实现互操作代理系统的关键一步，并在生态系统碎片化成为常态之前提供了一条实际可行的道路。 

---
