# Hume: Introducing System-2 Thinking in Visual-Language-Action Model 

**Title (ZH)**: 休姆：在视觉-语言-行动模型中引入系统二型思考 

**Authors**: Haoming Song, Delin Qu, Yuanqi Yao, Qizhi Chen, Qi Lv, Yiwen Tang, Modi Shi, Guanghui Ren, Maoqing Yao, Bin Zhao, Dong Wang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.21432)  

**Abstract**: Humans practice slow thinking before performing actual actions when handling complex tasks in the physical world. This thinking paradigm, recently, has achieved remarkable advancement in boosting Large Language Models (LLMs) to solve complex tasks in digital domains. However, the potential of slow thinking remains largely unexplored for robotic foundation models interacting with the physical world. In this work, we propose Hume: a dual-system Vision-Language-Action (VLA) model with value-guided System-2 thinking and cascaded action denoising, exploring human-like thinking capabilities of Vision-Language-Action models for dexterous robot control. System 2 of Hume implements value-Guided thinking by extending a Vision-Language-Action Model backbone with a novel value-query head to estimate the state-action value of predicted actions. The value-guided thinking is conducted by repeat sampling multiple action candidates and selecting one according to state-action value. System 1 of Hume is a lightweight reactive visuomotor policy that takes System 2 selected action and performs cascaded action denoising for dexterous robot control. At deployment time, System 2 performs value-guided thinking at a low frequency while System 1 asynchronously receives the System 2 selected action candidate and predicts fluid actions in real time. We show that Hume outperforms the existing state-of-the-art Vision-Language-Action models across multiple simulation benchmark and real-robot deployments. 

**Abstract (ZH)**: 人类在处理物理世界中的复杂任务时会先进行慢思考再采取实际行动。这种思维范式近期已在增强大型语言模型解决数字领域复杂任务方面取得了显著进展。然而，慢思考在与物理世界互动的机器人基础模型中的潜力尚未被充分探索。在本文中，我们提出Hume：一种具有价值导向的System-2思考和级联动作去噪的双系统视觉-语言-动作（VLA）模型，旨在探索视觉-语言-动作模型在灵巧机器人控制中的类人类思维能力。Hume的System 2通过扩展视觉-语言-动作模型骨干网络并添加一个新颖的价值查询头来实现价值导向的思考，以估算预测动作的状态-动作值。价值导向的思考通过多次采样动作候选并根据状态-动作值选择一个动作来实现。Hume的System 1是一个轻量级的反应式视知觉运动策略，它接受System 2选择的动作候选并进行级联动作去噪，以实现灵巧机器人控制。在部署阶段，System 2以低频率执行价值导向的思考，而System 1异步接收System 2选择的动作候选并实时预测连贯动作。实验表明，Hume在多个仿真基准测试和实际机器人部署中优于现有的最先进的视觉-语言-动作模型。 

---
# EquAct: An SE(3)-Equivariant Multi-Task Transformer for Open-Loop Robotic Manipulation 

**Title (ZH)**: EquAct: 一种针对开放环机器人操作的SE(3)不变多任务变压器 

**Authors**: Xupeng Zhu, Yu Qi, Yizhe Zhu, Robin Walters, Robert Platt  

**Link**: [PDF](https://arxiv.org/pdf/2505.21351)  

**Abstract**: Transformer architectures can effectively learn language-conditioned, multi-task 3D open-loop manipulation policies from demonstrations by jointly processing natural language instructions and 3D observations. However, although both the robot policy and language instructions inherently encode rich 3D geometric structures, standard transformers lack built-in guarantees of geometric consistency, often resulting in unpredictable behavior under SE(3) transformations of the scene. In this paper, we leverage SE(3) equivariance as a key structural property shared by both policy and language, and propose EquAct-a novel SE(3)-equivariant multi-task transformer. EquAct is theoretically guaranteed to be SE(3) equivariant and consists of two key components: (1) an efficient SE(3)-equivariant point cloud-based U-net with spherical Fourier features for policy reasoning, and (2) SE(3)-invariant Feature-wise Linear Modulation (iFiLM) layers for language conditioning. To evaluate its spatial generalization ability, we benchmark EquAct on 18 RLBench simulation tasks with both SE(3) and SE(2) scene perturbations, and on 4 physical tasks. EquAct performs state-of-the-art across these simulation and physical tasks. 

**Abstract (ZH)**: Transformer架构可以通过联合处理自然语言指令和3D观察，有效学习基于演示的多任务3D开放环操作策略。然而，尽管机器人的策略和自然语言指令本身就包含了丰富的3D几何结构，标准的变压器缺乏内在的几何一致性保证，常常导致在场景SE(3)变换下产生不可预测的行为。在这项研究中，我们利用SE(3)不变性作为策略和语言共同拥有的关键结构特性，提出了一种新的SE(3)-不变的多任务变压器——EquAct。EquAct在理论上保证了SE(3)不变性，并包含两个关键组件：(1)基于点云的SE(3)-不变U-net，结合球面傅里叶特征进行策略推理；(2)用于语言条件的SE(3)-不变特征归一化线性调制(iFiLM)层。为了评估其空间泛化能力，我们在包含SE(3)和SE(2)场景扰动的18个RLBench仿真任务以及4个物理任务上对EquAct进行了基准测试。EquAct在这些仿真和物理任务上表现出了最先进的性能。 

---
# Object-Centric Action-Enhanced Representations for Robot Visuo-Motor Policy Learning 

**Title (ZH)**: 基于对象中心的动作增强表示方法用于机器人视觉-运动策略学习 

**Authors**: Nikos Giannakakis, Argyris Manetas, Panagiotis P. Filntisis, Petros Maragos, George Retsinas  

**Link**: [PDF](https://arxiv.org/pdf/2505.20962)  

**Abstract**: Learning visual representations from observing actions to benefit robot visuo-motor policy generation is a promising direction that closely resembles human cognitive function and perception. Motivated by this, and further inspired by psychological theories suggesting that humans process scenes in an object-based fashion, we propose an object-centric encoder that performs semantic segmentation and visual representation generation in a coupled manner, unlike other works, which treat these as separate processes. To achieve this, we leverage the Slot Attention mechanism and use the SOLV model, pretrained in large out-of-domain datasets, to bootstrap fine-tuning on human action video data. Through simulated robotic tasks, we demonstrate that visual representations can enhance reinforcement and imitation learning training, highlighting the effectiveness of our integrated approach for semantic segmentation and encoding. Furthermore, we show that exploiting models pretrained on out-of-domain datasets can benefit this process, and that fine-tuning on datasets depicting human actions -- although still out-of-domain -- , can significantly improve performance due to close alignment with robotic tasks. These findings show the capability to reduce reliance on annotated or robot-specific action datasets and the potential to build on existing visual encoders to accelerate training and improve generalizability. 

**Abstract (ZH)**: 从观察动作中学习视觉表示以提高机器人视听运动策略生成：一种类人认知功能和感知的方向 

---
# G-DReaM: Graph-conditioned Diffusion Retargeting across Multiple Embodiments 

**Title (ZH)**: G-DReaM：基于图的扩散重定向跨多个载体模型 

**Authors**: Zhefeng Cao, Ben Liu, Sen Li, Wei Zhang, Hua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.20857)  

**Abstract**: Motion retargeting for specific robot from existing motion datasets is one critical step in transferring motion patterns from human behaviors to and across various robots. However, inconsistencies in topological structure, geometrical parameters as well as joint correspondence make it difficult to handle diverse embodiments with a unified retargeting architecture. In this work, we propose a novel unified graph-conditioned diffusion-based motion generation framework for retargeting reference motions across diverse embodiments. The intrinsic characteristics of heterogeneous embodiments are represented with graph structure that effectively captures topological and geometrical features of different robots. Such a graph-based encoding further allows for knowledge exploitation at the joint level with a customized attention mechanisms developed in this work. For lacking ground truth motions of the desired embodiment, we utilize an energy-based guidance formulated as retargeting losses to train the diffusion model. As one of the first cross-embodiment motion retargeting methods in robotics, our experiments validate that the proposed model can retarget motions across heterogeneous embodiments in a unified manner. Moreover, it demonstrates a certain degree of generalization to both diverse skeletal structures and similar motion patterns. 

**Abstract (ZH)**: 基于现有运动数据集的特定机器人运动移植是将人类行为的运动模式转移到和跨多个机器人的重要步骤。然而，拓扑结构、几何参数以及关节对应的一致性问题使得统一的移植架构难以处理多样化的机器人实体。在这项工作中，我们提出了一种新颖的统一图条件扩散驱动的运动生成框架，用于在多样化的机器人实体之间移植参考运动。异构实体的内在特征通过图结构来表示，有效捕捉不同机器人的拓扑和几何特征。基于图的编码还允许在本工作中开发的定制注意力机制下在关节级别进行知识利用。对于目标实体缺乏真实运动的情况，我们利用被表述为移植损失的能量导向方法来训练扩散模型。作为机器人中第一个跨实体运动移植方法之一，我们的实验验证了提出模型能够以统一的方式移植跨异构实体的运动，并且其在多样化的骨骼结构以及相似的运动模式方面展现了一定程度的泛化能力。 

---
# Learning Unified Force and Position Control for Legged Loco-Manipulation 

**Title (ZH)**: 统一腿式Manipulation的力和位置控制学习 

**Authors**: Peiyuan Zhi, Peiyang Li, Jianqin Yin, Baoxiong Jia, Siyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20829)  

**Abstract**: Robotic loco-manipulation tasks often involve contact-rich interactions with the environment, requiring the joint modeling of contact force and robot position. However, recent visuomotor policies often focus solely on learning position or force control, overlooking their co-learning. In this work, we propose the first unified policy for legged robots that jointly models force and position control learned without reliance on force sensors. By simulating diverse combinations of position and force commands alongside external disturbance forces, we use reinforcement learning to learn a policy that estimates forces from historical robot states and compensates for them through position and velocity adjustments. This policy enables a wide range of manipulation behaviors under varying force and position inputs, including position tracking, force application, force tracking, and compliant interactions. Furthermore, we demonstrate that the learned policy enhances trajectory-based imitation learning pipelines by incorporating essential contact information through its force estimation module, achieving approximately 39.5% higher success rates across four challenging contact-rich manipulation tasks compared to position-control policies. Extensive experiments on both a quadrupedal manipulator and a humanoid robot validate the versatility and robustness of the proposed policy across diverse scenarios. 

**Abstract (ZH)**: 基于接触的腿式机器人操控任务往往需要同时建模接触力和机器人位置，但近期的视觉运动策略往往仅 focuses on 学习位置或力控制，忽略了两者的同时学习。在此项工作中，我们提出了第一个无需依赖力传感器即可同时建模力和位置控制的统一策略。通过模拟位置和力命令以及外部干扰力的各种组合，我们利用强化学习来学习一个能够从历史机器人状态中估计力并通过对位置和速度的调整来补偿这些力的策略。该策略可以在不同力和位置输入下实现广泛的操控行为，包括位置跟踪、力应用、力跟踪和顺应交互。此外，我们展示了所学策略通过其力估计模块增加的关键接触信息改进了基于轨迹的模仿学习管道，相比位置控制策略，在四个具有挑战性的基于接触的操控任务中实现了大约 39.5% 的更高成功率。广泛的实验表明，所提出的策略在不同场景下具有广泛的适用性和鲁棒性。 

---
# GET: Goal-directed Exploration and Targeting for Large-Scale Unknown Environments 

**Title (ZH)**: GET: 目标导向的探索与瞄准在大规模未知环境中 

**Authors**: Lanxiang Zheng, Ruidong Mei, Mingxin Wei, Hao Ren, Hui Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.20828)  

**Abstract**: Object search in large-scale, unstructured environments remains a fundamental challenge in robotics, particularly in dynamic or expansive settings such as outdoor autonomous exploration. This task requires robust spatial reasoning and the ability to leverage prior experiences. While Large Language Models (LLMs) offer strong semantic capabilities, their application in embodied contexts is limited by a grounding gap in spatial reasoning and insufficient mechanisms for memory integration and decision this http URL address these challenges, we propose GET (Goal-directed Exploration and Targeting), a framework that enhances object search by combining LLM-based reasoning with experience-guided exploration. At its core is DoUT (Diagram of Unified Thought), a reasoning module that facilitates real-time decision-making through a role-based feedback loop, integrating task-specific criteria and external memory. For repeated tasks, GET maintains a probabilistic task map based on a Gaussian Mixture Model, allowing for continual updates to object-location priors as environments this http URL conducted in real-world, large-scale environments demonstrate that GET improves search efficiency and robustness across multiple LLMs and task settings, significantly outperforming heuristic and LLM-only baselines. These results suggest that structured LLM integration provides a scalable and generalizable approach to embodied decision-making in complex environments. 

**Abstract (ZH)**: 大规模、非结构化环境中的物体搜索仍然是机器人领域的一项基本挑战，特别是在动态或广阔的户外自主探索环境中。为了解决这些挑战，我们提出了一种名为GET（目标导向探索与瞄准）的框架，该框架通过结合基于LLM的推理与经验引导的探索来增强物体搜索能力。GET的核心是DoUT（统一思维图示），这是一种通过基于角色的反馈循环促进实时决策的推理模块，能够集成任务特定标准和外部记忆。在重复任务中，GET基于高斯混合模型维护一个概率任务地图，能够随着环境变化不断更新对象位置的先验知识。在真实世界的大规模环境中进行的实验表明，GET在多个LLM和任务设置中改善了搜索效率和鲁棒性，显著优于启发式和仅LLM基线。这些结果表明，结构化LLM集成为复杂环境中的体感决策提供了一种可扩展和通用的方法。 

---
# Spatial RoboGrasp: Generalized Robotic Grasping Control Policy 

**Title (ZH)**: 空间RoboGrasp：通用化机器人抓取控制策略 

**Authors**: Yiqi Huang, Travis Davies, Jiahuan Yan, Jiankai Sun, Xiang Chen, Luhui Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20814)  

**Abstract**: Achieving generalizable and precise robotic manipulation across diverse environments remains a critical challenge, largely due to limitations in spatial perception. While prior imitation-learning approaches have made progress, their reliance on raw RGB inputs and handcrafted features often leads to overfitting and poor 3D reasoning under varied lighting, occlusion, and object conditions. In this paper, we propose a unified framework that couples robust multimodal perception with reliable grasp prediction. Our architecture fuses domain-randomized augmentation, monocular depth estimation, and a depth-aware 6-DoF Grasp Prompt into a single spatial representation for downstream action planning. Conditioned on this encoding and a high-level task prompt, our diffusion-based policy yields precise action sequences, achieving up to 40% improvement in grasp success and 45% higher task success rates under environmental variation. These results demonstrate that spatially grounded perception, paired with diffusion-based imitation learning, offers a scalable and robust solution for general-purpose robotic grasping. 

**Abstract (ZH)**: 跨多种环境实现可泛化且精确的机器人 manipulation 仍然是一个关键挑战，主要由于空间感知的限制。尽管先前的模仿学习方法取得了一定进展，但它们依赖于原始 RGB 输入和手工制作的特征，往往导致在不同光照条件、遮挡和物体状态下发生过拟合和三维推理能力差。本文提出了一种统一框架，将健壮的多模态感知与可靠的抓取预测相结合。我们的架构将领域随机化增强、单目深度估计和深度感知的 6 自由度抓取提示融合到单一的空间表示中，用于下游动作规划。在给定此编码和高层任务提示的条件下，我们的基于扩散的策略生成精确的动作序列，在环境变化下抓取成功率提高了 40%，任务成功率提高了 45%。这些结果表明，基于空间感知的模仿学习提供了一种可扩展且鲁棒的通用机器人抓取解决方案。 

---
# Learning Generalizable Robot Policy with Human Demonstration Video as a Prompt 

**Title (ZH)**: 基于人类演示视频提示的学习可泛化的机器人策略 

**Authors**: Xiang Zhu, Yichen Liu, Hezhong Li, Jianyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.20795)  

**Abstract**: Recent robot learning methods commonly rely on imitation learning from massive robotic dataset collected with teleoperation. When facing a new task, such methods generally require collecting a set of new teleoperation data and finetuning the policy. Furthermore, the teleoperation data collection pipeline is also tedious and expensive. Instead, human is able to efficiently learn new tasks by just watching others do. In this paper, we introduce a novel two-stage framework that utilizes human demonstrations to learn a generalizable robot policy. Such policy can directly take human demonstration video as a prompt and perform new tasks without any new teleoperation data and model finetuning at all. In the first stage, we train video generation model that captures a joint representation for both the human and robot demonstration video data using cross-prediction. In the second stage, we fuse the learned representation with a shared action space between human and robot using a novel prototypical contrastive loss. Empirical evaluations on real-world dexterous manipulation tasks show the effectiveness and generalization capabilities of our proposed method. 

**Abstract (ZH)**: 利用人类演示学习可泛化的机器人策略的新型两阶段框架 

---
# FM-Planner: Foundation Model Guided Path Planning for Autonomous Drone Navigation 

**Title (ZH)**: FM-Planner: 基于基础模型的自主无人机导航路径规划 

**Authors**: Jiaping Xiao, Cheng Wen Tsao, Yuhang Zhang, Mir Feroskhan  

**Link**: [PDF](https://arxiv.org/pdf/2505.20783)  

**Abstract**: Path planning is a critical component in autonomous drone operations, enabling safe and efficient navigation through complex environments. Recent advances in foundation models, particularly large language models (LLMs) and vision-language models (VLMs), have opened new opportunities for enhanced perception and intelligent decision-making in robotics. However, their practical applicability and effectiveness in global path planning remain relatively unexplored. This paper proposes foundation model-guided path planners (FM-Planner) and presents a comprehensive benchmarking study and practical validation for drone path planning. Specifically, we first systematically evaluate eight representative LLM and VLM approaches using standardized simulation scenarios. To enable effective real-time navigation, we then design an integrated LLM-Vision planner that combines semantic reasoning with visual perception. Furthermore, we deploy and validate the proposed path planner through real-world experiments under multiple configurations. Our findings provide valuable insights into the strengths, limitations, and feasibility of deploying foundation models in real-world drone applications and providing practical implementations in autonomous flight. Project site: this https URL. 

**Abstract (ZH)**: 路径规划是自主无人机操作中的关键组件，能够确保无人机在复杂环境中的安全高效导航。最近基础模型的进步，特别是大规模语言模型（LLMs）和视觉语言模型（VLMs），为机器人领域的增强感知和智能决策开辟了新的机会。然而，它们在全局路径规划中的实际应用和有效性仍相对未被充分探索。本文提出了基于基础模型的路径规划器（FM-Planner），并进行了无人机路径规划的全面基准测试和实际验证。具体来说，我们首先系统地评估了八个代表性的LLM和VLM方法，使用标准化的模拟场景。为了实现有效的实时导航，我们设计了结合语义推理和视觉感知的集成LLM-视觉规划器。此外，我们通过多种配置下的实际实验部署并验证了所提路径规划器。我们的研究结果为在实际无人机应用中部署基础模型提供了有价值的见解，并为自主飞行提供了实用实现。项目网址：这个 https URL。 

---
# Interactive OT Gym: A Reinforcement Learning-Based Interactive Optical tweezer (OT)-Driven Microrobotics Simulation Platform 

**Title (ZH)**: 交互式OT健身房：基于强化学习的互动光学捕获（OT）驱动的微机器人模拟平台 

**Authors**: Zongcai Tan amd Dandan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20751)  

**Abstract**: Optical tweezers (OT) offer unparalleled capabilities for micromanipulation with submicron precision in biomedical applications. However, controlling conventional multi-trap OT to achieve cooperative manipulation of multiple complex-shaped microrobots in dynamic environments poses a significant challenge. To address this, we introduce Interactive OT Gym, a reinforcement learning (RL)-based simulation platform designed for OT-driven microrobotics. Our platform supports complex physical field simulations and integrates haptic feedback interfaces, RL modules, and context-aware shared control strategies tailored for OT-driven microrobot in cooperative biological object manipulation tasks. This integration allows for an adaptive blend of manual and autonomous control, enabling seamless transitions between human input and autonomous operation. We evaluated the effectiveness of our platform using a cell manipulation task. Experimental results show that our shared control system significantly improves micromanipulation performance, reducing task completion time by approximately 67% compared to using pure human or RL control alone and achieving a 100% success rate. With its high fidelity, interactivity, low cost, and high-speed simulation capabilities, Interactive OT Gym serves as a user-friendly training and testing environment for the development of advanced interactive OT-driven micromanipulation systems and control algorithms. For more details on the project, please see our website this https URL 

**Abstract (ZH)**: 光学 tweezers (OT) 在生物医学应用中提供了前所未有的 micron 级精度微操作能力。然而，控制传统的多陷阱 OT 在动态环境中实现复杂形状微机器人协同操作仍然面临重大挑战。为应对这一挑战，我们引入了 Interactive OT Gym，一个基于强化学习 (RL) 的仿真平台，旨在支持 OT 驱动的微机器人技术。该平台支持复杂的物理场仿真，并集成了触觉反馈接口、RL 模块和面向 OT 驱动微机器人的上下文感知共享控制策略，专为合作生物物体操作任务设计。该集成允许手动控制和自主控制的适应性结合，使得人类输入和自主操作之间的无缝过渡成为可能。我们使用细胞操作任务评估了该平台的有效性。实验结果表明，我们的共享控制系统显著提高了微操作性能，与仅使用纯人类或 RL 控制相比，任务完成时间减少了约 67%，并且成功率达到 100%。凭借其高保真度、交互性、低成本和高速仿真能力，Interactive OT Gym 成为开发高级互动 OT 驱动微操作系统和控制算法的用户友好型训练和测试环境。 

---
# ManiTaskGen: A Comprehensive Task Generator for Benchmarking and Improving Vision-Language Agents on Embodied Decision-Making 

**Title (ZH)**: ManiTaskGen：全面的任务生成器，用于基准测试和提升在体感决策方面视觉-语言代理的表现 

**Authors**: Liu Dai, Haina Wang, Weikang Wan, Hao Su  

**Link**: [PDF](https://arxiv.org/pdf/2505.20726)  

**Abstract**: Building embodied agents capable of accomplishing arbitrary tasks is a core objective towards achieving embodied artificial general intelligence (E-AGI). While recent work has advanced such general robot policies, their training and evaluation are often limited to tasks within specific scenes, involving restricted instructions and scenarios. Existing benchmarks also typically rely on manual annotation of limited tasks in a few scenes. We argue that exploring the full spectrum of feasible tasks within any given scene is crucial, as they provide both extensive benchmarks for evaluation and valuable resources for agent improvement. Towards this end, we introduce ManiTaskGen, a novel system that automatically generates comprehensive, diverse, feasible mobile manipulation tasks for any given scene. The generated tasks encompass both process-based, specific instructions (e.g., "move object from X to Y") and outcome-based, abstract instructions (e.g., "clear the table"). We apply ManiTaskGen to both simulated and real-world scenes, demonstrating the validity and diversity of the generated tasks. We then leverage these tasks to automatically construct benchmarks, thoroughly evaluating the embodied decision-making capabilities of agents built upon existing vision-language models (VLMs). Furthermore, we propose a simple yet effective method that utilizes ManiTaskGen tasks to enhance embodied decision-making. Overall, this work presents a universal task generation framework for arbitrary scenes, facilitating both benchmarking and improvement of embodied decision-making agents. 

**Abstract (ZH)**: 构建能够在任意任务中胜任的具身智能代理是实现具身人工智能（E-AGI）的核心目标。虽然近期的工作推进了通用机器人政策的发展，但其训练和评估往往局限于特定场景内的特定任务和场景。现有的基准测试通常依赖于对少数场景中有限任务的手动标注。我们认为，在任何给定场景中探索可行任务的全谱至关重要，因为这为评估提供了广泛的基准，并为代理改进提供了宝贵的资源。为此，我们介绍了ManiTaskGen，这是一种新型系统，能够自动为任何给定场景生成全面、多样、可行的移动操控任务。生成的任务既包括基于过程的具体指令（例如，“将物体从X移动到Y”），也包括基于结果的抽象指令（例如，“清理桌子”）。我们将ManiTaskGen应用于模拟和真实场景，展示了生成任务的有效性和多样性。然后利用这些任务自动构建基准测试，全面评估基于现有视觉-语言模型（VLMs）构建的代理的具身决策能力。此外，我们提出了一个简单有效的方法，利用ManiTaskGen任务来增强具身决策能力。总体而言，本工作提出了一种用于任意场景的通用任务生成框架，促进了具身决策代理的基准测试和改进。 

---
# Gait-Conditioned Reinforcement Learning with Multi-Phase Curriculum for Humanoid Locomotion 

**Title (ZH)**: 基于步态条件的多阶段 Curriculum 强化学习在类人行走中的应用 

**Authors**: Tianhu Peng, Lingfan Bao, CHengxu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.20619)  

**Abstract**: We present a unified gait-conditioned reinforcement learning framework that enables humanoid robots to perform standing, walking, running, and smooth transitions within a single recurrent policy. A compact reward routing mechanism dynamically activates gait-specific objectives based on a one-hot gait ID, mitigating reward interference and supporting stable multi-gait learning. Human-inspired reward terms promote biomechanically natural motions, such as straight-knee stance and coordinated arm-leg swing, without requiring motion capture data. A structured curriculum progressively introduces gait complexity and expands command space over multiple phases. In simulation, the policy successfully achieves robust standing, walking, running, and gait transitions. On the real Unitree G1 humanoid, we validate standing, walking, and walk-to-stand transitions, demonstrating stable and coordinated locomotion. This work provides a scalable, reference-free solution toward versatile and naturalistic humanoid control across diverse modes and environments. 

**Abstract (ZH)**: 我们提出了一种统一的步态条件增强学习框架，使类人机器人能够在单一递归策略中执行站立、行走、跑步及平滑过渡。紧凑的奖励路由机制基于一热步态ID动态激活特定步态目标，减轻了奖励干扰，支持稳定的多步态学习。灵感源于人类的奖励术语促进了生物力学自然的运动模式，如直膝站立和协调的臂腿摆动，无需使用运动捕捉数据。结构化的课程设计逐步引入步态复杂性并扩展命令空间，跨越多个阶段。在模拟中，该策略成功实现了稳健的站立、行走、跑步及步态过渡。在真实的Unitree G1类人机器人上，我们验证了站立、行走及行走至站立的过渡，展示了稳定且协调的行走。这项工作为跨多种模式和环境的多功能且自然的类人机器人控制提供了可扩展且无需参考的解决方案。 

---
# CoRI: Synthesizing Communication of Robot Intent for Physical Human-Robot Interaction 

**Title (ZH)**: CoRI: 合成机器人在物理人机交互中意图的通信 

**Authors**: Junxiang Wang, Emek Barış Küçüktabak, Rana Soltani Zarrin, Zackory Erickson  

**Link**: [PDF](https://arxiv.org/pdf/2505.20537)  

**Abstract**: Clear communication of robot intent fosters transparency and interpretability in physical human-robot interaction (pHRI), particularly during assistive tasks involving direct human-robot contact. We introduce CoRI, a pipeline that automatically generates natural language communication of a robot's upcoming actions directly from its motion plan and visual perception. Our pipeline first processes the robot's image view to identify human poses and key environmental features. It then encodes the planned 3D spatial trajectory (including velocity and force) onto this view, visually grounding the path and its dynamics. CoRI queries a vision-language model with this visual representation to interpret the planned action within the visual context before generating concise, user-directed statements, without relying on task-specific information. Results from a user study involving robot-assisted feeding, bathing, and shaving tasks across two different robots indicate that CoRI leads to statistically significant difference in communication clarity compared to a baseline communication strategy. Specifically, CoRI effectively conveys not only the robot's high-level intentions but also crucial details about its motion and any collaborative user action needed. 

**Abstract (ZH)**: 清晰传达机器人意图促进物理人机交互（pHRI）的透明度和可解释性，特别是在涉及直接人机接触的辅助任务中。我们引入了CoRI管道，该管道能够直接从机器人的运动计划和视觉感知中自动生成自然语言的沟通内容。该管道首先处理机器人的图像视图以识别人类姿态和关键环境特征。然后，将计划中的3D空间轨迹（包括速度和力）编码到这一视图中，从而在视觉上将路径及其动态联系起来。CoRI 使用这一视觉表示查询视觉语言模型，以在视觉上下文中解释计划的动作，随后生成简洁、面向用户的陈述，而无需依赖特定任务的信息。来自一项用户研究的结果表明，CoRI 在机器人辅助喂食、洗澡和刮胡任务中显著提高了沟通清晰度，特别是在有效地传达机器人的高层次意图和关键运动细节方面。 

---
# Embodied AI with Foundation Models for Mobile Service Robots: A Systematic Review 

**Title (ZH)**: 基于基础模型的移动服务机器人具身AI：一项系统回顾 

**Authors**: Matthew Lisondra, Beno Benhabib, Goldie Nejat  

**Link**: [PDF](https://arxiv.org/pdf/2505.20503)  

**Abstract**: Rapid advancements in foundation models, including Large Language Models, Vision-Language Models, Multimodal Large Language Models, and Vision-Language-Action Models have opened new avenues for embodied AI in mobile service robotics. By combining foundation models with the principles of embodied AI, where intelligent systems perceive, reason, and act through physical interactions, robots can improve understanding, adapt to, and execute complex tasks in dynamic real-world environments. However, embodied AI in mobile service robots continues to face key challenges, including multimodal sensor fusion, real-time decision-making under uncertainty, task generalization, and effective human-robot interactions (HRI). In this paper, we present the first systematic review of the integration of foundation models in mobile service robotics, identifying key open challenges in embodied AI and examining how foundation models can address them. Namely, we explore the role of such models in enabling real-time sensor fusion, language-conditioned control, and adaptive task execution. Furthermore, we discuss real-world applications in the domestic assistance, healthcare, and service automation sectors, demonstrating the transformative impact of foundation models on service robotics. We also include potential future research directions, emphasizing the need for predictive scaling laws, autonomous long-term adaptation, and cross-embodiment generalization to enable scalable, efficient, and robust deployment of foundation models in human-centric robotic systems. 

**Abstract (ZH)**: 基础模型的快速发展，包括大型语言模型、 vision-language模型、多模态大型语言模型以及vision-language-action模型，为移动服务机器人中的具身人工智能打开了新的途径。通过将基础模型与具身人工智能的原则相结合，即智能系统通过物理交互来感知、推理和行动，机器人可以在动态真实环境中提高理解能力、适应环境并执行复杂任务。然而，移动服务机器人中的具身人工智能仍然面临多重挑战，包括多模态传感器融合、不确定性的实时决策、任务泛化以及有效的机器人-人类交互（HRI）。在本文中，我们首次系统地回顾了基础模型在移动服务机器人中的集成，确定了具身人工智能中的关键开放挑战，并探讨了基础模型如何应对这些挑战。具体而言，我们探讨了这些模型在实现实时传感器融合、基于语言的控制以及适应性任务执行中的作用。此外，我们讨论了在家庭辅助、医疗服务和服务业自动化领域的实际应用，展示了基础模型对服务机器人的变革性影响。我们还提出了潜在的未来研究方向，强调预测性扩展规律、自主长期适应和跨具身的一般化的重要性，以实现面向人类的机器人系统中基础模型的大规模、高效和稳健部署。 

---
# OSVI-WM: One-Shot Visual Imitation for Unseen Tasks using World-Model-Guided Trajectory Generation 

**Title (ZH)**: OSVI-WM：基于世界模型指导轨迹生成的单次视觉模仿以应对未见任务 

**Authors**: Raktim Gautam Goswami, Prashanth Krishnamurthy, Yann LeCun, Farshad Khorrami  

**Link**: [PDF](https://arxiv.org/pdf/2505.20425)  

**Abstract**: Visual imitation learning enables robotic agents to acquire skills by observing expert demonstration videos. In the one-shot setting, the agent generates a policy after observing a single expert demonstration without additional fine-tuning. Existing approaches typically train and evaluate on the same set of tasks, varying only object configurations, and struggle to generalize to unseen tasks with different semantic or structural requirements. While some recent methods attempt to address this, they exhibit low success rates on hard test tasks that, despite being visually similar to some training tasks, differ in context and require distinct responses. Additionally, most existing methods lack an explicit model of environment dynamics, limiting their ability to reason about future states. To address these limitations, we propose a novel framework for one-shot visual imitation learning via world-model-guided trajectory generation. Given an expert demonstration video and the agent's initial observation, our method leverages a learned world model to predict a sequence of latent states and actions. This latent trajectory is then decoded into physical waypoints that guide the agent's execution. Our method is evaluated on two simulated benchmarks and three real-world robotic platforms, where it consistently outperforms prior approaches, with over 30% improvement in some cases. 

**Abstract (ZH)**: 视觉模仿学习使机器人代理通过观察专家演示视频来获取技能。在单次学习设置中，代理在观察单次专家演示后生成策略，无需额外微调。现有方法通常在相同的任务集上进行训练和评估，仅改变对象配置，难以泛化到具有不同语义或结构要求的未见过的任务。尽管一些最新方法试图解决这一问题，但在涉及视觉上相似但上下文和所需响应不同的困难测试任务上，它们的表现成功率较低。此外，大多数现有方法缺乏环境动力学的显式模型，限制了它们对未来状态的推理能力。为解决这些问题，我们提出了一种基于世界模型引导轨迹生成的单次视觉模仿学习新框架。给定专家演示视频和代理的初始观察，我们的方法利用学习到的世界模型预测一序列潜在状态和动作。该潜在轨迹随后被解码为物理航点，指导代理的执行。我们在两个模拟基准和三个真实世界机器人平台上进行了评估，结果显示该方法在所有测试中均优于先前方法，在某些情况下提高了30%以上。 

---
# Robot Operation of Home Appliances by Reading User Manuals 

**Title (ZH)**: 基于阅读用户手册控制家用电器的机器人操作方法 

**Authors**: Jian Zhang, Hanbo Zhang, Anxing Xiao, David Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20424)  

**Abstract**: Operating home appliances, among the most common tools in every household, is a critical capability for assistive home robots. This paper presents ApBot, a robot system that operates novel household appliances by "reading" their user manuals. ApBot faces multiple challenges: (i) infer goal-conditioned partial policies from their unstructured, textual descriptions in a user manual document, (ii) ground the policies to the appliance in the physical world, and (iii) execute the policies reliably over potentially many steps, despite compounding errors. To tackle these challenges, ApBot constructs a structured, symbolic model of an appliance from its manual, with the help of a large vision-language model (VLM). It grounds the symbolic actions visually to control panel elements. Finally, ApBot closes the loop by updating the model based on visual feedback. Our experiments show that across a wide range of simulated and real-world appliances, ApBot achieves consistent and statistically significant improvements in task success rate, compared with state-of-the-art large VLMs used directly as control policies. These results suggest that a structured internal representations plays an important role in robust robot operation of home appliances, especially, complex ones. 

**Abstract (ZH)**: 操作家用电器：一种通过“阅读”用户手册操控新型家用电器的机器人系统 

---
# CoDA: Coordinated Diffusion Noise Optimization for Whole-Body Manipulation of Articulated Objects 

**Title (ZH)**: CoDA: 协调扩散噪声优化以 Manipulate 整体 articulated 对象 

**Authors**: Huaijin Pi, Zhi Cen, Zhiyang Dou, Taku Komura  

**Link**: [PDF](https://arxiv.org/pdf/2505.21437)  

**Abstract**: Synthesizing whole-body manipulation of articulated objects, including body motion, hand motion, and object motion, is a critical yet challenging task with broad applications in virtual humans and robotics. The core challenges are twofold. First, achieving realistic whole-body motion requires tight coordination between the hands and the rest of the body, as their movements are interdependent during manipulation. Second, articulated object manipulation typically involves high degrees of freedom and demands higher precision, often requiring the fingers to be placed at specific regions to actuate movable parts. To address these challenges, we propose a novel coordinated diffusion noise optimization framework. Specifically, we perform noise-space optimization over three specialized diffusion models for the body, left hand, and right hand, each trained on its own motion dataset to improve generalization. Coordination naturally emerges through gradient flow along the human kinematic chain, allowing the global body posture to adapt in response to hand motion objectives with high fidelity. To further enhance precision in hand-object interaction, we adopt a unified representation based on basis point sets (BPS), where end-effector positions are encoded as distances to the same BPS used for object geometry. This unified representation captures fine-grained spatial relationships between the hand and articulated object parts, and the resulting trajectories serve as targets to guide the optimization of diffusion noise, producing highly accurate interaction motion. We conduct extensive experiments demonstrating that our method outperforms existing approaches in motion quality and physical plausibility, and enables various capabilities such as object pose control, simultaneous walking and manipulation, and whole-body generation from hand-only data. 

**Abstract (ZH)**: 合成具关节对象的整体操控动作，包括身体运动、手部运动和物体运动，是虚拟人类和机器人领域广泛应用中一个关键而富有挑战的任务。核心挑战主要有两点。首先，实现真实的身体运动需要手部与身体其他部分之间的紧密协调，因为它们在操控过程中相互依赖。其次，具关节对象的操控通常涉及高自由度，并要求更高的精度，往往需要手指置于特定区域以驱动可动部分。为应对这些挑战，我们提出了一种新的协调扩散噪声优化框架。具体而言，我们在专门针对身体、左手和右手的三个扩散模型的空间中进行噪声优化，每个模型均在其自身的运动数据集上训练，以提高泛化能力。通过沿着人体运动链的梯度流动自然实现协调，使整体身体姿态能够高保真地适应手部运动目标。为了进一步提高手部与对象交互的精度，我们采用基于基点集（BPS）的统一表示法，其中末端执行器位置编码为相对于用于对象几何的同一BPS的距离。这种统一表示法捕捉了手部与具关节对象部件之间细微的空间关系，生成的轨迹作为优化扩散噪声的目标，产生高精度的交互动作。我们进行大量实验表明，我们的方法在动作质量和物理可行性方面优于现有方法，并能够实现多种能力，如对象姿态控制、同时行走和操控以及仅凭手部数据生成全部身体动作。 

---
# ControlTac: Force- and Position-Controlled Tactile Data Augmentation with a Single Reference Image 

**Title (ZH)**: ControlTac: 基于单张参考图像的力控制和位置控制触觉数据增强 

**Authors**: Dongyu Luo, Kelin Yu, Amir-Hossein Shahidzadeh, Cornelia Fermüller, Yiannis Aloimonos  

**Link**: [PDF](https://arxiv.org/pdf/2505.20498)  

**Abstract**: Vision-based tactile sensing has been widely used in perception, reconstruction, and robotic manipulation. However, collecting large-scale tactile data remains costly due to the localized nature of sensor-object interactions and inconsistencies across sensor instances. Existing approaches to scaling tactile data, such as simulation and free-form tactile generation, often suffer from unrealistic output and poor transferability to downstream this http URL address this, we propose ControlTac, a two-stage controllable framework that generates realistic tactile images conditioned on a single reference tactile image, contact force, and contact position. With those physical priors as control input, ControlTac generates physically plausible and varied tactile images that can be used for effective data augmentation. Through experiments on three downstream tasks, we demonstrate that ControlTac can effectively augment tactile datasets and lead to consistent gains. Our three real-world experiments further validate the practical utility of our approach. Project page: this https URL. 

**Abstract (ZH)**: 基于视觉的触觉感知在感知、重建和机器人操作中得到了广泛应用。然而，由于传感器-物体交互的局部性质以及传感器实例间的不一致性，收集大规模触觉数据仍然是 costly 的。现有的触觉数据扩展方法，如仿真和自由形式触觉生成，常面临输出不现实和下游任务适应性差的问题。为解决这些问题，我们提出了一种两阶段可控框架——ControlTac，该框架可以根据一个参考触觉图像、接触力和接触位置生成现实的触觉图像。通过物理先验作为控制输入，ControlTac 生成了物理上合理且多变的触觉图像，可用于有效的数据增强。通过在三个下游任务上的实验，我们证明了 ControlTac 可以有效增强触觉数据集，并取得一致的改进。我们三个实际实验进一步验证了该方法的实用价值。项目页面：this https URL。 

---
# Diagnosing and Resolving Cloud Platform Instability with Multi-modal RAG LLMs 

**Title (ZH)**: 使用多模态RAG大语言模型诊断和解决云平台不稳定问题 

**Authors**: Yifan Wang, Kenneth P. Birman  

**Link**: [PDF](https://arxiv.org/pdf/2505.21419)  

**Abstract**: Today's cloud-hosted applications and services are complex systems, and a performance or functional instability can have dozens or hundreds of potential root causes. Our hypothesis is that by combining the pattern matching capabilities of modern AI tools with a natural multi-modal RAG LLM interface, problem identification and resolution can be simplified. ARCA is a new multi-modal RAG LLM system that targets this domain. Step-wise evaluations show that ARCA outperforms state-of-the-art alternatives. 

**Abstract (ZH)**: 今天托管在云上的应用程序和服务是复杂的系统，性能或功能不稳定可能有数十甚至数百个潜在的根本原因。我们假设通过结合现代AI工具的模式匹配能力以及自然多模态RAG语言模型接口，问题的识别和解决可以得到简化。ARCA是一种针对这一领域的新型多模态RAG语言模型系统。逐步评估表明，ARCA优于现有最先进的替代方案。 

---
# Assured Autonomy with Neuro-Symbolic Perception 

**Title (ZH)**: 确保自主性的神经符号感知 

**Authors**: R. Spencer Hallyburton, Miroslav Pajic  

**Link**: [PDF](https://arxiv.org/pdf/2505.21322)  

**Abstract**: Many state-of-the-art AI models deployed in cyber-physical systems (CPS), while highly accurate, are simply pattern-matchers.~With limited security guarantees, there are concerns for their reliability in safety-critical and contested domains. To advance assured AI, we advocate for a paradigm shift that imbues data-driven perception models with symbolic structure, inspired by a human's ability to reason over low-level features and high-level context. We propose a neuro-symbolic paradigm for perception (NeuSPaPer) and illustrate how joint object detection and scene graph generation (SGG) yields deep scene understanding.~Powered by foundation models for offline knowledge extraction and specialized SGG algorithms for real-time deployment, we design a framework leveraging structured relational graphs that ensures the integrity of situational awareness in autonomy. Using physics-based simulators and real-world datasets, we demonstrate how SGG bridges the gap between low-level sensor perception and high-level reasoning, establishing a foundation for resilient, context-aware AI and advancing trusted autonomy in CPS. 

**Abstract (ZH)**: 先进的AI模型在 cyber-物理系统(CPS)中的应用虽高度准确，但本质上只是模式匹配器。受限于有限的安全保障，其在安全关键和对抗领域中的可靠性存在担忧。为推进值得信赖的AI，我们提倡一种范式转变，即在数据驱动的感知模型中融入符号结构，借鉴人类在低级特征和高级语境上的推理能力。我们提出了一种神经符号感知范式（NeuSPaPer），并通过结合物体检测和场景图生成（SGG）实现深层次场景理解。借助基础模型进行离线知识提取和专门的SGG算法实现实时部署，我们设计了一种框架，利用结构化关系图确保自主系统情境意识的完整性。利用物理仿真器和真实世界数据集，我们展示了SGG如何弥合传感器低级感知与高级推理之间的差距，为 resilient、情境感知的AI奠定基础，并推进CPS中的可信自主性。 

---
# XBOUND: Exploring the Capability Boundaries of Device-Control Agents through Trajectory Tree Exploration 

**Title (ZH)**: XBOUND：通过轨迹树探索设备控制代理的能力边界 

**Authors**: Shaoqing Zhang, Kehai Chen, Zhuosheng Zhang, Rumei Li, Rongxiang Weng, Yang Xiang, Liqiang Nie, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21279)  

**Abstract**: Recent advancements in vision-language models (VLMs) have spurred increased interest in Device-Control Agents (DC agents), such as utilizing in-the-wild device control to manage graphical user interfaces. Conventional methods for assessing the capabilities of DC agents, such as computing step-wise action accuracy and overall task success rates, provide a macroscopic view of DC agents' performance; however, they fail to offer microscopic insights into potential errors that may occur in real-world applications. Conducting a finer-grained performance evaluation of DC agents presents significant challenges. This study introduces a new perspective on evaluation methods for DC agents by proposing the XBOUND evaluation method, which employs the calculation of a novel Explore Metric to delineate the capability boundaries of DC agents. Compared to previous evaluation methods, XBOUND focuses on individual states to assess the proficiency of DC agents in mastering these states. Furthermore, we have developed a ``pseudo'' episode tree dataset derived from Android Control test data. Utilizing this dataset and XBOUND, we comprehensively evaluate the OS-Atlas and UI-TARS series, examining both the overall and specific performance across five common tasks. Additionally, we select representative cases to highlight the current deficiencies and limitations inherent in both series. Code is available at this https URL. 

**Abstract (ZH)**: 最近视觉-语言模型的发展促进了设备控制代理的研究，如利用野外设备控制管理图形用户界面。传统的方法通过计算逐步动作准确率和整体任务成功率来评估设备控制代理的能力，提供了宏观视角但无法提供潜在错误的微观见解。对设备控制代理进行细粒度性能评估面临重大挑战。本文通过提出XBOUND评估方法引入了一种新的评估方法视角，该方法通过计算一种新颖的探索度量来界定设备控制代理的能力边界。与以往的方法相比，XBOUND着眼于个体状态来评估代理掌握这些状态的能力。此外，我们基于Android Control测试数据开发了一个“伪”情景树数据集。利用此数据集和XBOUND，我们全面评估了OS-Atlas和UI-TARS系列，在五个常见任务中考察了整体和特定性能，并选择了代表性案例突出两个系列中存在的当前不足和局限性。代码可在以下链接获取。 

---
# Reinforcement Learning-based Sequential Route Recommendation for System-Optimal Traffic Assignment 

**Title (ZH)**: 基于强化学习的顺序路线推荐方法用于系统最优交通分配 

**Authors**: Leizhen Wang, Peibo Duan, Cheng Lyu, Zhenliang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.20889)  

**Abstract**: Modern navigation systems and shared mobility platforms increasingly rely on personalized route recommendations to improve individual travel experience and operational efficiency. However, a key question remains: can such sequential, personalized routing decisions collectively lead to system-optimal (SO) traffic assignment? This paper addresses this question by proposing a learning-based framework that reformulates the static SO traffic assignment problem as a single-agent deep reinforcement learning (RL) task. A central agent sequentially recommends routes to travelers as origin-destination (OD) demands arrive, to minimize total system travel time. To enhance learning efficiency and solution quality, we develop an MSA-guided deep Q-learning algorithm that integrates the iterative structure of traditional traffic assignment methods into the RL training process. The proposed approach is evaluated on both the Braess and Ortuzar-Willumsen (OW) networks. Results show that the RL agent converges to the theoretical SO solution in the Braess network and achieves only a 0.35% deviation in the OW network. Further ablation studies demonstrate that the route action set's design significantly impacts convergence speed and final performance, with SO-informed route sets leading to faster learning and better outcomes. This work provides a theoretically grounded and practically relevant approach to bridging individual routing behavior with system-level efficiency through learning-based sequential assignment. 

**Abstract (ZH)**: 基于学习的框架：个人化路径推荐与系统最优交通分配的统一 

---
# Project Riley: Multimodal Multi-Agent LLM Collaboration with Emotional Reasoning and Voting 

**Title (ZH)**: Project Riley：具有情绪推理和投票的多模态多agents LLM协作 

**Authors**: Ana Rita Ortigoso, Gabriel Vieira, Daniel Fuentes, Luis Frazão, Nuno Costa, António Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2505.20521)  

**Abstract**: This paper presents Project Riley, a novel multimodal and multi-model conversational AI architecture oriented towards the simulation of reasoning influenced by emotional states. Drawing inspiration from Pixar's Inside Out, the system comprises five distinct emotional agents - Joy, Sadness, Fear, Anger, and Disgust - that engage in structured multi-round dialogues to generate, criticise, and iteratively refine responses. A final reasoning mechanism synthesises the contributions of these agents into a coherent output that either reflects the dominant emotion or integrates multiple perspectives. The architecture incorporates both textual and visual large language models (LLMs), alongside advanced reasoning and self-refinement processes. A functional prototype was deployed locally in an offline environment, optimised for emotional expressiveness and computational efficiency. From this initial prototype, another one emerged, called Armando, which was developed for use in emergency contexts, delivering emotionally calibrated and factually accurate information through the integration of Retrieval-Augmented Generation (RAG) and cumulative context tracking. The Project Riley prototype was evaluated through user testing, in which participants interacted with the chatbot and completed a structured questionnaire assessing three dimensions: Emotional Appropriateness, Clarity and Utility, and Naturalness and Human-likeness. The results indicate strong performance in structured scenarios, particularly with respect to emotional alignment and communicative clarity. 

**Abstract (ZH)**: 项目里利：一种面向情绪影响推理模拟的新型多模态多模型对话AI架构 

---
# Challenges for artificial cognitive systems 

**Title (ZH)**: 人工认知系统面临的挑战 

**Authors**: Antoni Gomila, Vincent C. Müller  

**Link**: [PDF](https://arxiv.org/pdf/2505.20339)  

**Abstract**: The declared goal of this paper is to fill this gap: "... cognitive systems research needs questions or challenges that define progress. The challenges are not (yet more) predictions of the future, but a guideline to what are the aims and what would constitute progress." -- the quotation being from the project description of EUCogII, the project for the European Network for Cognitive Systems within which this formulation of the 'challenges' was originally developed (this http URL). So, we stick out our neck and formulate the challenges for artificial cognitive systems. These challenges are articulated in terms of a definition of what a cognitive system is: a system that learns from experience and uses its acquired knowledge (both declarative and practical) in a flexible manner to achieve its own goals. 

**Abstract (ZH)**: 本文的目标是填补这一空白：“……认知系统研究需要能够定义进步的问题或挑战。这些挑战不是对未来的新预测，而是指导人们了解目标以及何为进步的标准。”——引自EUCogII项目描述，该项目是欧洲认知系统网络的一部分，在其中首次提出了这些“挑战”的表述（详见链接）。因此，我们提出了人工认知系统的挑战。这些挑战基于对认知系统定义的阐述：一种能够从经验中学习并在灵活运用已获知识（包括显性和实用性知识）以实现自身目标方面表现出色的系统。 

---
# ViewSpatial-Bench: Evaluating Multi-perspective Spatial Localization in Vision-Language Models 

**Title (ZH)**: ViewSpatial-Bench: 评估视觉语言模型的多视角空间定位能力 

**Authors**: Dingming Li, Hongxing Li, Zixuan Wang, Yuchen Yan, Hang Zhang, Siqi Chen, Guiyang Hou, Shengpei Jiang, Wenqi Zhang, Yongliang Shen, Weiming Lu, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21500)  

**Abstract**: Vision-language models (VLMs) have demonstrated remarkable capabilities in understanding and reasoning about visual content, but significant challenges persist in tasks requiring cross-viewpoint understanding and spatial reasoning. We identify a critical limitation: current VLMs excel primarily at egocentric spatial reasoning (from the camera's perspective) but fail to generalize to allocentric viewpoints when required to adopt another entity's spatial frame of reference. We introduce ViewSpatial-Bench, the first comprehensive benchmark designed specifically for multi-viewpoint spatial localization recognition evaluation across five distinct task types, supported by an automated 3D annotation pipeline that generates precise directional labels. Comprehensive evaluation of diverse VLMs on ViewSpatial-Bench reveals a significant performance disparity: models demonstrate reasonable performance on camera-perspective tasks but exhibit reduced accuracy when reasoning from a human viewpoint. By fine-tuning VLMs on our multi-perspective spatial dataset, we achieve an overall performance improvement of 46.24% across tasks, highlighting the efficacy of our approach. Our work establishes a crucial benchmark for spatial intelligence in embodied AI systems and provides empirical evidence that modeling 3D spatial relationships enhances VLMs' corresponding spatial comprehension capabilities. 

**Abstract (ZH)**: Vision-language模型（VLMs）在理解和推理视觉内容方面展示了显著的能力，但在要求跨视角理解和空间推理的任务中仍面临重大挑战。我们识别出一个关键限制：当前的VLMs 主要擅长以自我中心的空间推理（从相机的角度），但在需要采用其他实体的空间参照框架时无法有效泛化到他觉中心视角。我们引入了ViewSpatial-Bench，这是第一个专为跨视角空间定位识别评估设计的综合性基准，支持自动化的3D注释流水线以生成精确的方向标签。对ViewSpatial-Bench上各种VLMs的全面评估揭示了一个显著的性能差异：模型在以相机视角进行任务时表现出合理的性能，但在从人类视角进行推理时却表现出较低的准确性。通过在我们多元视角空间数据集上微调VLMs，我们实现了跨任务总体性能提升46.24%，突显了我们方法的有效性。我们的工作为体态人工智能系统中的空间智能建立了关键基准，并提供了实验证据，表明建模3D空间关系能够增强VLMs相应的空间理解能力。 

---
# Active-O3: Empowering Multimodal Large Language Models with Active Perception via GRPO 

**Title (ZH)**: Active-O3: 通过GRPO增强多模态大型语言模型的主动感知能力 

**Authors**: Muzhi Zhu, Hao Zhong, Canyu Zhao, Zongze Du, Zheng Huang, Mingyu Liu, Hao Chen, Cheng Zou, Jingdong Chen, Ming Yang, Chunhua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.21457)  

**Abstract**: Active vision, also known as active perception, refers to the process of actively selecting where and how to look in order to gather task-relevant information. It is a critical component of efficient perception and decision-making in humans and advanced embodied agents. Recently, the use of Multimodal Large Language Models (MLLMs) as central planning and decision-making modules in robotic systems has gained extensive attention. However, despite the importance of active perception in embodied intelligence, there is little to no exploration of how MLLMs can be equipped with or learn active perception capabilities. In this paper, we first provide a systematic definition of MLLM-based active perception tasks. We point out that the recently proposed GPT-o3 model's zoom-in search strategy can be regarded as a special case of active perception; however, it still suffers from low search efficiency and inaccurate region selection. To address these issues, we propose ACTIVE-O3, a purely reinforcement learning based training framework built on top of GRPO, designed to equip MLLMs with active perception capabilities. We further establish a comprehensive benchmark suite to evaluate ACTIVE-O3 across both general open-world tasks, such as small-object and dense object grounding, and domain-specific scenarios, including small object detection in remote sensing and autonomous driving, as well as fine-grained interactive segmentation. In addition, ACTIVE-O3 also demonstrates strong zero-shot reasoning abilities on the V* Benchmark, without relying on any explicit reasoning data. We hope that our work can provide a simple codebase and evaluation protocol to facilitate future research on active perception in MLLMs. 

**Abstract (ZH)**: 基于多模态大规模语言模型的主动视觉：一种基于GRPO的纯强化学习训练框架 

---
# A Framework for Adversarial Analysis of Decision Support Systems Prior to Deployment 

**Title (ZH)**: 决策支持系统部署前的对抗分析框架 

**Authors**: Brett Bissey, Kyle Gatesman, Walker Dimon, Mohammad Alam, Luis Robaina, Joseph Weissman  

**Link**: [PDF](https://arxiv.org/pdf/2505.21414)  

**Abstract**: This paper introduces a comprehensive framework designed to analyze and secure decision-support systems trained with Deep Reinforcement Learning (DRL), prior to deployment, by providing insights into learned behavior patterns and vulnerabilities discovered through simulation. The introduced framework aids in the development of precisely timed and targeted observation perturbations, enabling researchers to assess adversarial attack outcomes within a strategic decision-making context. We validate our framework, visualize agent behavior, and evaluate adversarial outcomes within the context of a custom-built strategic game, CyberStrike. Utilizing the proposed framework, we introduce a method for systematically discovering and ranking the impact of attacks on various observation indices and time-steps, and we conduct experiments to evaluate the transferability of adversarial attacks across agent architectures and DRL training algorithms. The findings underscore the critical need for robust adversarial defense mechanisms to protect decision-making policies in high-stakes environments. 

**Abstract (ZH)**: 本文提出了一种全面框架，旨在在部署前分析和保障基于深度强化学习(DRL)训练的决策支持系统，通过模拟提供对学习行为模式和发现的漏洞的洞见。该引入的框架有助于开发精确的时间和目标观察扰动，使研究人员能够在战略决策背景下评估对抗攻击的效果。我们在自建的战略游戏CyberStrike中验证了该框架，可视化了智能体的行为，并评估了对抗攻击的结果。利用提出的框架，我们介绍了一种系统地发现和排列攻击对不同观察指标和时间步影响的方法，并进行了实验以评估对抗攻击在智能体架构和DRL训练算法之间的可移植性。研究结果强调了在高危环境中保护决策策略的抗逆向防御机制的迫切需要。 

---
# A domain adaptation neural network for digital twin-supported fault diagnosis 

**Title (ZH)**: 基于数字孪生支持的域适应神经网络故障诊断 

**Authors**: Zhenling Chen, Haiwei Fu, Zhiguo Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.21046)  

**Abstract**: Digital twins offer a promising solution to the lack of sufficient labeled data in deep learning-based fault diagnosis by generating simulated data for model training. However, discrepancies between simulation and real-world systems can lead to a significant drop in performance when models are applied in real scenarios. To address this issue, we propose a fault diagnosis framework based on Domain-Adversarial Neural Networks (DANN), which enables knowledge transfer from simulated (source domain) to real-world (target domain) data. We evaluate the proposed framework using a publicly available robotics fault diagnosis dataset, which includes 3,600 sequences generated by a digital twin model and 90 real sequences collected from physical systems. The DANN method is compared with commonly used lightweight deep learning models such as CNN, TCN, Transformer, and LSTM. Experimental results show that incorporating domain adaptation significantly improves the diagnostic performance. For example, applying DANN to a baseline CNN model improves its accuracy from 70.00% to 80.22% on real-world test data, demonstrating the effectiveness of domain adaptation in bridging the sim-to-real gap. 

**Abstract (ZH)**: 数字孪生提供的模拟数据可为基于深度学习的故障诊断问题提供充足的标记数据，并通过域对抗神经网络（DANN）实现模拟数据向真实数据的知识迁移，从而改善模型在实际场景中的性能。实验结果表明，引入域适应显著提高了诊断性能。例如，将DANN应用于基础的CNN模型，其在真实数据上的准确率从70.00%提高到80.22%，证明了域适应在弥合仿真实与真实场景差距方面的有效性。 

---
# Revisiting Multi-Agent World Modeling from a Diffusion-Inspired Perspective 

**Title (ZH)**: 从扩散启发的角度 revisiting 多智能体世界建模 

**Authors**: Yang Zhang, Xinran Li, Jianing Ye, Delin Qu, Shuang Qiu, Chongjie Zhang, Xiu Li, Chenjia Bai  

**Link**: [PDF](https://arxiv.org/pdf/2505.20922)  

**Abstract**: World models have recently attracted growing interest in Multi-Agent Reinforcement Learning (MARL) due to their ability to improve sample efficiency for policy learning. However, accurately modeling environments in MARL is challenging due to the exponentially large joint action space and highly uncertain dynamics inherent in multi-agent systems. To address this, we reduce modeling complexity by shifting from jointly modeling the entire state-action transition dynamics to focusing on the state space alone at each timestep through sequential agent modeling. Specifically, our approach enables the model to progressively resolve uncertainty while capturing the structured dependencies among agents, providing a more accurate representation of how agents influence the state. Interestingly, this sequential revelation of agents' actions in a multi-agent system aligns with the reverse process in diffusion models--a class of powerful generative models known for their expressiveness and training stability compared to autoregressive or latent variable models. Leveraging this insight, we develop a flexible and robust world model for MARL using diffusion models. Our method, Diffusion-Inspired Multi-Agent world model (DIMA), achieves state-of-the-art performance across multiple multi-agent control benchmarks, significantly outperforming prior world models in terms of final return and sample efficiency, including MAMuJoCo and Bi-DexHands. DIMA establishes a new paradigm for constructing multi-agent world models, advancing the frontier of MARL research. 

**Abstract (ZH)**: 基于扩散模型的Multi-Agent世界模型（DIMA）：多智能体控制benchmark上的最新进展 

---
# VLM Can Be a Good Assistant: Enhancing Embodied Visual Tracking with Self-Improving Visual-Language Models 

**Title (ZH)**: 基于视觉语言模型的自我提升能力可以成为增强現實视觉跟踪的好助手 

**Authors**: Kui Wu, Shuhang Xu, Hao Chen, Churan Wang, Zhoujun Li, Yizhou Wang, Fangwei Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2505.20718)  

**Abstract**: We introduce a novel self-improving framework that enhances Embodied Visual Tracking (EVT) with Visual-Language Models (VLMs) to address the limitations of current active visual tracking systems in recovering from tracking failure. Our approach combines the off-the-shelf active tracking methods with VLMs' reasoning capabilities, deploying a fast visual policy for normal tracking and activating VLM reasoning only upon failure detection. The framework features a memory-augmented self-reflection mechanism that enables the VLM to progressively improve by learning from past experiences, effectively addressing VLMs' limitations in 3D spatial reasoning. Experimental results demonstrate significant performance improvements, with our framework boosting success rates by $72\%$ with state-of-the-art RL-based approaches and $220\%$ with PID-based methods in challenging environments. This work represents the first integration of VLM-based reasoning to assist EVT agents in proactive failure recovery, offering substantial advances for real-world robotic applications that require continuous target monitoring in dynamic, unstructured environments. Project website: this https URL. 

**Abstract (ZH)**: 我们引入了一种新型自我改进框架，通过结合视觉语言模型（VLMs）的能力来增强沉浸式视觉跟踪（EVT），以解决当前主动视觉跟踪系统在恢复跟踪失败方面存在的局限性。该方法将现成的主动跟踪方法与VLMs的推理能力相结合，在正常跟踪时部署快速视觉策略，并仅在检测到失败时激活VLM推理。该框架拥有一个增强的记忆自我反省机制，使VLM能够通过学习过往经验逐步改进，有效解决了VLM在三维空间推理方面的局限性。实验结果表明，与基于强化学习（RL）的方法相比，我们的框架在具有挑战性的环境中将成功率提升了72%，与基于PID的方法相比提升了220%。这项工作代表了VLM基于推理首次应用于帮助EVT代理实现主动故障恢复的整合，为需要在动态、非结构化环境中持续目标监控的实际机器人应用提供了显著的进步。项目网站：这个 https URL。 

---
# VSCBench: Bridging the Gap in Vision-Language Model Safety Calibration 

**Title (ZH)**: VSCBench: 桥接视觉-语言模型安全性标定的差距 

**Authors**: Jiahui Geng, Qing Li, Zongxiong Chen, Yuxia Wang, Derui Zhu, Zhuohan Xie, Chenyang Lyu, Xiuying Chen, Preslav Nakov, Fakhri Karray  

**Link**: [PDF](https://arxiv.org/pdf/2505.20362)  

**Abstract**: The rapid advancement of vision-language models (VLMs) has brought a lot of attention to their safety alignment. However, existing methods have primarily focused on model undersafety, where the model responds to hazardous queries, while neglecting oversafety, where the model refuses to answer safe queries. In this paper, we introduce the concept of $\textit{safety calibration}$, which systematically addresses both undersafety and oversafety. Specifically, we present $\textbf{VSCBench}$, a novel dataset of 3,600 image-text pairs that are visually or textually similar but differ in terms of safety, which is designed to evaluate safety calibration across image-centric and text-centric scenarios. Based on our benchmark, we evaluate safety calibration across eleven widely used VLMs. Our extensive experiments revealed major issues with both undersafety and oversafety. We further investigated four approaches to improve the model's safety calibration. We found that even though some methods effectively calibrated the models' safety problems, these methods also lead to the degradation of models' utility. This trade-off underscores the urgent need for advanced calibration methods, and our benchmark provides a valuable tool for evaluating future approaches. Our code and data are available at this https URL. 

**Abstract (ZH)**: 视觉语言模型的安全校准：VSCBench及其应用 

---
# Decision Flow Policy Optimization 

**Title (ZH)**: 决策流策略优化 

**Authors**: Jifeng Hu, Sili Huang, Siyuan Guo, Zhaogeng Liu, Li Shen, Lichao Sun, Hechang Chen, Yi Chang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2505.20350)  

**Abstract**: In recent years, generative models have shown remarkable capabilities across diverse fields, including images, videos, language, and decision-making. By applying powerful generative models such as flow-based models to reinforcement learning, we can effectively model complex multi-modal action distributions and achieve superior robotic control in continuous action spaces, surpassing the limitations of single-modal action distributions with traditional Gaussian-based policies. Previous methods usually adopt the generative models as behavior models to fit state-conditioned action distributions from datasets, with policy optimization conducted separately through additional policies using value-based sample weighting or gradient-based updates. However, this separation prevents the simultaneous optimization of multi-modal distribution fitting and policy improvement, ultimately hindering the training of models and degrading the performance. To address this issue, we propose Decision Flow, a unified framework that integrates multi-modal action distribution modeling and policy optimization. Specifically, our method formulates the action generation procedure of flow-based models as a flow decision-making process, where each action generation step corresponds to one flow decision. Consequently, our method seamlessly optimizes the flow policy while capturing multi-modal action distributions. We provide rigorous proofs of Decision Flow and validate the effectiveness through extensive experiments across dozens of offline RL environments. Compared with established offline RL baselines, the results demonstrate that our method achieves or matches the SOTA performance. 

**Abstract (ZH)**: 近年来，生成模型在图像、视频、语言和决策等领域展示了 remarkable 的能力。通过将基于流的生成模型等强大生成模型应用到强化学习中，我们可以有效地建模复杂的多模态动作分布，并在连续动作空间中实现优胜的机器人控制，超越了基于单一模态动作分布的传统高斯策略的限制。先前的方法通常将生成模型作为行为模型，用于拟合数据集中的状态条件动作分布，并通过额外的策略进行独立的策略优化，使用基于价值的样本加权或基于梯度的更新。然而，这种分离阻碍了多模态分布拟合和策略改进的同时优化，最终影响模型的训练并降低性能。为了解决这一问题，我们提出了 Decision Flow，这是一种统一框架，将多模态动作分布建模和策略优化结合起来。具体而言，我们的方法将基于流的模型的动作生成过程表述为一个流决策过程，其中每个动作生成步骤对应一个流决策。因此，我们的方法能够无缝优化流策略并捕获多模态动作分布。我们提供了 Decision Flow 的严格证明，并通过在多个离线 RL 环境中的广泛实验证明了其有效性。与现有的离线 RL 基准相比，结果表明我们的方法达到了或匹配了 SOTA 性能。 

---
# SpatialLLM: From Multi-modality Data to Urban Spatial Intelligence 

**Title (ZH)**: SpatialLLM：从多模态数据到城市空间智能 

**Authors**: Jiabin Chen, Haiping Wang, Jinpeng Li, Yuan Liu, Zhen Dong, Bisheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12703)  

**Abstract**: We propose SpatialLLM, a novel approach advancing spatial intelligence tasks in complex urban scenes. Unlike previous methods requiring geographic analysis tools or domain expertise, SpatialLLM is a unified language model directly addressing various spatial intelligence tasks without any training, fine-tuning, or expert intervention. The core of SpatialLLM lies in constructing detailed and structured scene descriptions from raw spatial data to prompt pre-trained LLMs for scene-based analysis. Extensive experiments show that, with our designs, pretrained LLMs can accurately perceive spatial distribution information and enable zero-shot execution of advanced spatial intelligence tasks, including urban planning, ecological analysis, traffic management, etc. We argue that multi-field knowledge, context length, and reasoning ability are key factors influencing LLM performances in urban analysis. We hope that SpatialLLM will provide a novel viable perspective for urban intelligent analysis and management. The code and dataset are available at this https URL. 

**Abstract (ZH)**: 我们提出SpatialLLM，一种在复杂城市场景中推进空间智能任务的新方法。与以往需要地理分析工具或专业领域知识的方法不同，SpatialLLM 是一个统一的语言模型，可以直接处理各种空间智能任务，无需任何训练、微调或专家干预。SpatialLLM 的核心在于从原始空间数据构建详细的结构化场景描述，以触发预训练的大规模语言模型进行基于场景的分析。广泛的实验结果显示，通过我们的设计，预训练的语言模型能够准确感知空间分布信息，并实现诸如城市规划、生态分析、交通管理等高级空间智能任务的零样本执行。我们认为，多领域知识、上下文长度和推理能力是影响大规模语言模型在城市分析中表现的关键因素。我们希望SpatialLLM能够为城市智能分析和管理提供一种新颖可行的视角。相关代码和数据集可在以下网址获取。 

---
