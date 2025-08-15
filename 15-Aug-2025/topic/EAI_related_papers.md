# CVIRO: A Consistent and Tightly-Coupled Visual-Inertial-Ranging Odometry on Lie Groups 

**Title (ZH)**: CVIRO：基于李群的一致且紧密耦合的视觉-惯性-测距 одometry 

**Authors**: Yizhi Zhou, Ziwei Kang, Jiawei Xia, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10867)  

**Abstract**: Ultra Wideband (UWB) is widely used to mitigate drift in visual-inertial odometry (VIO) systems. Consistency is crucial for ensuring the estimation accuracy of a UWBaided VIO system. An inconsistent estimator can degrade localization performance, where the inconsistency primarily arises from two main factors: (1) the estimator fails to preserve the correct system observability, and (2) UWB anchor positions are assumed to be known, leading to improper neglect of calibration uncertainty. In this paper, we propose a consistent and tightly-coupled visual-inertial-ranging odometry (CVIRO) system based on the Lie group. Our method incorporates the UWB anchor state into the system state, explicitly accounting for UWB calibration uncertainty and enabling the joint and consistent estimation of both robot and anchor states. Furthermore, observability consistency is ensured by leveraging the invariant error properties of the Lie group. We analytically prove that the CVIRO algorithm naturally maintains the system's correct unobservable subspace, thereby preserving estimation consistency. Extensive simulations and experiments demonstrate that CVIRO achieves superior localization accuracy and consistency compared to existing methods. 

**Abstract (ZH)**: 基于李群的一致且紧密耦合的视觉-惯性-测距定位系统（C Vishion-Inertial-Ranging Odometry, CVIRO） 

---
# The SET Perceptual Factors Framework: Towards Assured Perception for Autonomous Systems 

**Title (ZH)**: SET感知因素框架：迈向自主系统可信赖感知的研究 

**Authors**: Troi Williams  

**Link**: [PDF](https://arxiv.org/pdf/2508.10798)  

**Abstract**: Future autonomous systems promise significant societal benefits, yet their deployment raises concerns about safety and trustworthiness. A key concern is assuring the reliability of robot perception, as perception seeds safe decision-making. Failures in perception are often due to complex yet common environmental factors and can lead to accidents that erode public trust. To address this concern, we introduce the SET (Self, Environment, and Target) Perceptual Factors Framework. We designed the framework to systematically analyze how factors such as weather, occlusion, or sensor limitations negatively impact perception. To achieve this, the framework employs SET State Trees to categorize where such factors originate and SET Factor Trees to model how these sources and factors impact perceptual tasks like object detection or pose estimation. Next, we develop Perceptual Factor Models using both trees to quantify the uncertainty for a given task. Our framework aims to promote rigorous safety assurances and cultivate greater public understanding and trust in autonomous systems by offering a transparent and standardized method for identifying, modeling, and communicating perceptual risks. 

**Abstract (ZH)**: 未来自主系统有望为社会带来巨大益处，但部署引发了关于安全与可信度的关注。一个关键的关注点是如何保证机器人的感知可靠性 以及如何确保感知的安全决策。感知中的失败往往由于复杂且常见的环境因素所致 并且会导致因信任度下降而引发的事故。为应对这一关切 我们介绍了SET（自我·环境·验证）感知因素框架。该框架旨在系统性分析感知问题 如如遮挡等、低识别率等等和其他限制因素如何负面影响感知。为了分析此类问题 该框架使用了SET STEM树来来分类导致感知问题的根本原因 并使用SET因素树来评估感知任务的性能和这些因素如何影响感知任务 如如目标检测和 �推荐阅读点 � עם估计。该框架使用了SET STM模型 来衡量一种感知任务中的不确定性。我们的目标是提供严格感知保障 以通过提供透明且标准化的方法来模（型请感知风险和沟通感知不确定性来来促进对和和信任。 

---
# Biasing Frontier-Based Exploration with Saliency Areas 

**Title (ZH)**: 基于显著区域引导前沿探索 

**Authors**: Matteo Luperto, Valerii Stakanov, Giacomo Boracchi, Nicola Basilico, Francesco Amigoni  

**Link**: [PDF](https://arxiv.org/pdf/2508.10689)  

**Abstract**: Autonomous exploration is a widely studied problem where a robot incrementally builds a map of a previously unknown environment. The robot selects the next locations to reach using an exploration strategy. To do so, the robot has to balance between competing objectives, like exploring the entirety of the environment, while being as fast as possible. Most exploration strategies try to maximise the explored area to speed up exploration; however, they do not consider that parts of the environment are more important than others, as they lead to the discovery of large unknown areas. We propose a method that identifies \emph{saliency areas} as those areas that are of high interest for exploration, by using saliency maps obtained from a neural network that, given the current map, implements a termination criterion to estimate whether the environment can be considered fully-explored or not. We use saliency areas to bias some widely used exploration strategies, showing, with an extensive experimental campaign, that this knowledge can significantly influence the behavior of the robot during exploration. 

**Abstract (ZH)**: 自主探索是广泛研究的一个问题，其中机器人通过增量构建以前未知环境的地图。机器人使用探索策略来选择下一个要访问的位置。为此，机器人需要在诸如全面探索环境与尽可能快速之间平衡竞争目标。大多数探索策略试图最大化探索的区域以加快探索过程；然而，它们没有考虑环境中某些部分比其他部分更重要，因为这些部分会发现大面积未知区域。我们提出了一种方法，通过使用神经网络从当前地图生成的可引起终止条件的显著性图来识别显著性区域，这些区域对探索具有高兴趣。我们利用显著性区域偏置一些广泛使用的探索策略，并通过广泛的经验实验表明，这种知识可以显著影响机器人在探索过程中的行为。 

---
# An Open-Source User-Friendly Interface for Simulating Magnetic Soft Robots using Simulation Open Framework Architecture (SOFA) 

**Title (ZH)**: 使用Simulation Open Framework Architecture (SOFA)模拟磁软机器人的开源用户友好界面 

**Authors**: Carla Wehner, Finn Schubert, Heiko Hellkamp, Julius Hahnewald, Kilian Scheafer, Muhammad Bilal Khan, Oliver Gutfleisch  

**Link**: [PDF](https://arxiv.org/pdf/2508.10686)  

**Abstract**: Soft robots, particularly magnetic soft robots, require specialized simulation tools to accurately model their deformation under external magnetic fields. However, existing platforms often lack dedicated support for magnetic materials, making them difficult to use for researchers at different expertise levels. This work introduces an open-source, user-friendly simulation interface using the Simulation Open Framework Architecture (SOFA), specifically designed to model magnetic soft robots. The tool enables users to define material properties, apply magnetic fields, and observe resulting deformations in real time. By integrating intuitive controls and stress analysis capabilities, it aims to bridge the gap between theoretical modeling and practical design. Four benchmark models - a beam, three- and four-finger grippers, and a butterfly - demonstrate its functionality. The software's ease of use makes it accessible to both beginners and advanced researchers. Future improvements will refine accuracy through experimental validation and comparison with industry-standard finite element solvers, ensuring realistic and predictive simulations of magnetic soft robots. 

**Abstract (ZH)**: 软体机器人，尤其是磁性软体机器人，需要专门的仿真工具来准确模拟其在外加磁场下的变形。现有平台往往缺乏专门支持磁性材料的功能，使得研究人员难以使用。本文介绍了使用Simulation Open Framework Architecture (SOFA)开发的一种开源、用户友好的仿真接口，专门用于模拟磁性软体机器人。该工具允许用户定义材料属性、应用磁场并实时观察变形结果。通过集成直观的控制和应力分析能力，它旨在缩小理论建模与实际设计之间的差距。四个基准模型（一根梁、三指和四指夹爪以及一只蝴蝶）演示了其功能。该软件易于使用，使其对初学者和高级研究人员都具有可访问性。未来改进将通过实验验证和与工业标准有限元求解器的比较来提高准确性，以确保对磁性软体机器人的真实性和预测性仿真。 

---
# MLM: Learning Multi-task Loco-Manipulation Whole-Body Control for Quadruped Robot with Arm 

**Title (ZH)**: MLM：学习四足机器人带臂多任务位姿操作全身控制 

**Authors**: Xin Liu, Bida Ma, Chenkun Qi, Yan Ding, Zhaxizhuoma, Guorong Zhang, Pengan Chen, Kehui Liu, Zhongjie Jia, Chuyue Guan, Yule Mo, Jiaqi Liu, Feng Gao, Jiangwei Zhong, Bin Zhao, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.10538)  

**Abstract**: Whole-body loco-manipulation for quadruped robots with arm remains a challenging problem, particularly in achieving multi-task control. To address this, we propose MLM, a reinforcement learning framework driven by both real-world and simulation data. It enables a six-DoF robotic arm--equipped quadruped robot to perform whole-body loco-manipulation for multiple tasks autonomously or under human teleoperation. To address the problem of balancing multiple tasks during the learning of loco-manipulation, we introduce a trajectory library with an adaptive, curriculum-based sampling mechanism. This approach allows the policy to efficiently leverage real-world collected trajectories for learning multi-task loco-manipulation. To address deployment scenarios with only historical observations and to enhance the performance of policy execution across tasks with different spatial ranges, we propose a Trajectory-Velocity Prediction policy network. It predicts unobservable future trajectories and velocities. By leveraging extensive simulation data and curriculum-based rewards, our controller achieves whole-body behaviors in simulation and zero-shot transfer to real-world deployment. Ablation studies in simulation verify the necessity and effectiveness of our approach, while real-world experiments on the Go2 robot with an Airbot robotic arm demonstrate the policy's good performance in multi-task execution. 

**Abstract (ZH)**: 四足机器人配备手臂的全身 locomotion-manipulation 与多任务控制：基于现实与仿真数据的强化学习框架 MLM 

---
# MASH: Cooperative-Heterogeneous Multi-Agent Reinforcement Learning for Single Humanoid Robot Locomotion 

**Title (ZH)**: MASH: 合作异构多智能体强化学习在单个类人机器人运动中的应用 

**Authors**: Qi Liu, Xiaopeng Zhang, Mingshan Tan, Shuaikang Ma, Jinliang Ding, Yanjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.10423)  

**Abstract**: This paper proposes a novel method to enhance locomotion for a single humanoid robot through cooperative-heterogeneous multi-agent deep reinforcement learning (MARL). While most existing methods typically employ single-agent reinforcement learning algorithms for a single humanoid robot or MARL algorithms for multi-robot system tasks, we propose a distinct paradigm: applying cooperative-heterogeneous MARL to optimize locomotion for a single humanoid robot. The proposed method, multi-agent reinforcement learning for single humanoid locomotion (MASH), treats each limb (legs and arms) as an independent agent that explores the robot's action space while sharing a global critic for cooperative learning. Experiments demonstrate that MASH accelerates training convergence and improves whole-body cooperation ability, outperforming conventional single-agent reinforcement learning methods. This work advances the integration of MARL into single-humanoid-robot control, offering new insights into efficient locomotion strategies. 

**Abstract (ZH)**: 一种基于合作异构多智能体深度强化学习的单人形机器人运动增强方法 

---
# CorrectNav: Self-Correction Flywheel Empowers Vision-Language-Action Navigation Model 

**Title (ZH)**: CorrectNav: 自校正飞轮赋能视觉-语言-动作导航模型 

**Authors**: Zhuoyuan Yu, Yuxing Long, Zihan Yang, Chengyan Zeng, Hongwei Fan, Jiyao Zhang, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2508.10416)  

**Abstract**: Existing vision-and-language navigation models often deviate from the correct trajectory when executing instructions. However, these models lack effective error correction capability, hindering their recovery from errors. To address this challenge, we propose Self-correction Flywheel, a novel post-training paradigm. Instead of considering the model's error trajectories on the training set as a drawback, our paradigm emphasizes their significance as a valuable data source. We have developed a method to identify deviations in these error trajectories and devised innovative techniques to automatically generate self-correction data for perception and action. These self-correction data serve as fuel to power the model's continued training. The brilliance of our paradigm is revealed when we re-evaluate the model on the training set, uncovering new error trajectories. At this time, the self-correction flywheel begins to spin. Through multiple flywheel iterations, we progressively enhance our monocular RGB-based VLA navigation model CorrectNav. Experiments on R2R-CE and RxR-CE benchmarks show CorrectNav achieves new state-of-the-art success rates of 65.1% and 69.3%, surpassing prior best VLA navigation models by 8.2% and 16.4%. Real robot tests in various indoor and outdoor environments demonstrate \method's superior capability of error correction, dynamic obstacle avoidance, and long instruction following. 

**Abstract (ZH)**: 自纠正飞轮：一种新的后训练范式 

---
# Large Model Empowered Embodied AI: A Survey on Decision-Making and Embodied Learning 

**Title (ZH)**: 大型模型赋能的 embodied AI: 一项关于决策与体态学习的综述 

**Authors**: Wenlong Liang, Rui Zhou, Yang Ma, Bing Zhang, Songlin Li, Yijia Liao, Ping Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10399)  

**Abstract**: Embodied AI aims to develop intelligent systems with physical forms capable of perceiving, decision-making, acting, and learning in real-world environments, providing a promising way to Artificial General Intelligence (AGI). Despite decades of explorations, it remains challenging for embodied agents to achieve human-level intelligence for general-purpose tasks in open dynamic environments. Recent breakthroughs in large models have revolutionized embodied AI by enhancing perception, interaction, planning and learning. In this article, we provide a comprehensive survey on large model empowered embodied AI, focusing on autonomous decision-making and embodied learning. We investigate both hierarchical and end-to-end decision-making paradigms, detailing how large models enhance high-level planning, low-level execution, and feedback for hierarchical decision-making, and how large models enhance Vision-Language-Action (VLA) models for end-to-end decision making. For embodied learning, we introduce mainstream learning methodologies, elaborating on how large models enhance imitation learning and reinforcement learning in-depth. For the first time, we integrate world models into the survey of embodied AI, presenting their design methods and critical roles in enhancing decision-making and learning. Though solid advances have been achieved, challenges still exist, which are discussed at the end of this survey, potentially as the further research directions. 

**Abstract (ZH)**: 具身AI旨在开发具备感知、决策、行动和学习能力的物理形态智能系统，为人工通用智能（AGI）提供了有希望的方式。尽管经过了数十年的探索，具身智能体在开放动态环境中实现通用任务的人类水平智能仍具有挑战性。近年来，大规模模型的突破性进展通过增强感知、交互、规划和学习，革命性地推动了具身AI的发展。在本文中，我们对大规模模型赋能的具身AI进行了全面调研，重点关注自主决策和具身学习。我们探讨了分层和端到端决策-making范式，详细说明了大规模模型如何增强分层决策的高层规划、低层执行和反馈，以及如何增强视觉-语言-行动（VLA）模型的端到端决策。对于具身学习，我们介绍了主流的学习方法，并深入阐述了大规模模型如何增强模仿学习和强化学习。这是第一次将世界模型整合到具身AI的综述中，介绍了其设计方法及其在增强决策和学习方面的关键作用。尽管取得了扎实的进步，但仍存在挑战，这些挑战将在本文结尾讨论，作为进一步研究的方向。 

---
# A Semantic-Aware Framework for Safe and Intent-Integrative Assistance in Upper-Limb Exoskeletons 

**Title (ZH)**: 基于语义的认知框架：上肢外骨骼安全且意图整合的辅助方法 

**Authors**: Yu Chen, Shu Miao, Chunyu Wu, Jingsong Mu, Bo OuYang, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.10378)  

**Abstract**: Upper-limb exoskeletons are primarily designed to provide assistive support by accurately interpreting and responding to human intentions. In home-care scenarios, exoskeletons are expected to adapt their assistive configurations based on the semantic information of the task, adjusting appropriately in accordance with the nature of the object being manipulated. However, existing solutions often lack the ability to understand task semantics or collaboratively plan actions with the user, limiting their generalizability. To address this challenge, this paper introduces a semantic-aware framework that integrates large language models into the task planning framework, enabling the delivery of safe and intent-integrative assistance. The proposed approach begins with the exoskeleton operating in transparent mode to capture the wearer's intent during object grasping. Once semantic information is extracted from the task description, the system automatically configures appropriate assistive parameters. In addition, a diffusion-based anomaly detector is used to continuously monitor the state of human-robot interaction and trigger real-time replanning in response to detected anomalies. During task execution, online trajectory refinement and impedance control are used to ensure safety and regulate human-robot interaction. Experimental results demonstrate that the proposed method effectively aligns with the wearer's cognition, adapts to semantically varying tasks, and responds reliably to anomalies. 

**Abstract (ZH)**: 上肢外骨骼主要通过准确解读和响应人类意图来提供辅助支持。在家庭护理场景中，外骨骼期望根据任务的语义信息调整其辅助配置，根据所操作对象的性质适时调整。然而，现有解决方案往往缺乏理解任务语义或与用户协作规划动作的能力，限制了其通用性。为应对这一挑战，本文引入了一种语义感知框架，将大型语言模型整合到任务规划框架中，从而实现安全且意图整合的辅助。该提出的方案首先在外骨骼透明模式下运行，以捕捉佩戴者在抓取物体时的意图。一旦从任务描述中提取出语义信息，系统将自动配置适当的辅助参数。此外，采用基于扩散的异常检测器持续监控人机交互状态，并在检测到异常时触发实时重新规划。在任务执行过程中，通过在线轨迹优化和阻抗控制确保安全并调节人机交互。实验结果表明，所提出的方法能够有效地与佩戴者的认知相契合，适应语义变化的任务，并可靠地响应异常。 

---
# Few-shot Vision-based Human Activity Recognition with MLLM-based Visual Reinforcement Learning 

**Title (ZH)**: 基于MLLM的视觉强化学习在少样本视觉人体活动识别中的应用 

**Authors**: Wenqi Zheng, Yutaka Arakawa  

**Link**: [PDF](https://arxiv.org/pdf/2508.10371)  

**Abstract**: Reinforcement learning in large reasoning models enables learning from feedback on their outputs, making it particularly valuable in scenarios where fine-tuning data is limited. However, its application in multi-modal human activity recognition (HAR) domains remains largely underexplored. Our work extends reinforcement learning to the human activity recognition domain with multimodal large language models. By incorporating visual reinforcement learning in the training process, the model's generalization ability on few-shot recognition can be greatly improved. Additionally, visual reinforcement learning can enhance the model's reasoning ability and enable explainable analysis in the inference stage. We name our few-shot human activity recognition method with visual reinforcement learning FAVOR. Specifically, our approach first utilizes a multimodal large language model (MLLM) to generate multiple candidate responses for the human activity image, each containing reasoning traces and final answers. These responses are then evaluated using reward functions, and the MLLM model is subsequently optimized using the Group Relative Policy Optimization (GRPO) algorithm. In this way, the MLLM model can be adapted to human activity recognition with only a few samples. Extensive experiments on four human activity recognition datasets and five different settings demonstrate the superiority of the proposed method. 

**Abstract (ZH)**: 基于视觉强化学习的大规模语言模型在多模态人类活动识别中的 few-shot 训练方法 

---
# BEASST: Behavioral Entropic Gradient based Adaptive Source Seeking for Mobile Robots 

**Title (ZH)**: BEASST: 基于行为熵梯度自适应源寻求的移动机器人算法 

**Authors**: Donipolo Ghimire, Aamodh Suresh, Carlos Nieto-Granda, Solmaz S. Kia  

**Link**: [PDF](https://arxiv.org/pdf/2508.10363)  

**Abstract**: This paper presents BEASST (Behavioral Entropic Gradient-based Adaptive Source Seeking for Mobile Robots), a novel framework for robotic source seeking in complex, unknown environments. Our approach enables mobile robots to efficiently balance exploration and exploitation by modeling normalized signal strength as a surrogate probability of source location. Building on Behavioral Entropy(BE) with Prelec's probability weighting function, we define an objective function that adapts robot behavior from risk-averse to risk-seeking based on signal reliability and mission urgency. The framework provides theoretical convergence guarantees under unimodal signal assumptions and practical stability under bounded disturbances. Experimental validation across DARPA SubT and multi-room scenarios demonstrates that BEASST consistently outperforms state-of-the-art methods, achieving 15% reduction in path length and 20% faster source localization through intelligent uncertainty-driven navigation that dynamically transitions between aggressive pursuit and cautious exploration. 

**Abstract (ZH)**: 基于行为熵梯度的自适应源搜索框架BEASST：移动机器人在复杂未知环境中的源搜索新方法 

---
# ReconVLA: Reconstructive Vision-Language-Action Model as Effective Robot Perceiver 

**Title (ZH)**: ReconVLA：重建视觉-语言-动作模型作为有效的机器人感知器 

**Authors**: Wenxuan Song, Ziyang Zhou, Han Zhao, Jiayi Chen, Pengxiang Ding, Haodong Yan, Yuxin Huang, Feilong Tang, Donglin Wang, Haoang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.10333)  

**Abstract**: Recent advances in Vision-Language-Action (VLA) models have enabled robotic agents to integrate multimodal understanding with action execution. However, our empirical analysis reveals that current VLAs struggle to allocate visual attention to target regions. Instead, visual attention is always dispersed. To guide the visual attention grounding on the correct target, we propose ReconVLA, a reconstructive VLA model with an implicit grounding paradigm. Conditioned on the model's visual outputs, a diffusion transformer aims to reconstruct the gaze region of the image, which corresponds to the target manipulated objects. This process prompts the VLA model to learn fine-grained representations and accurately allocate visual attention, thus effectively leveraging task-specific visual information and conducting precise manipulation. Moreover, we curate a large-scale pretraining dataset comprising over 100k trajectories and 2 million data samples from open-source robotic datasets, further boosting the model's generalization in visual reconstruction. Extensive experiments in simulation and the real world demonstrate the superiority of our implicit grounding method, showcasing its capabilities of precise manipulation and generalization. Our project page is this https URL. 

**Abstract (ZH)**: Recent advances in 视觉-语言-行动 (VLA) 模型使机器人代理能够整合多模态理解和行动执行。然而，我们的实证分析表明，当前的VLA模型在分配视觉注意力到目标区域时面临困难，视觉注意力总是分散的。为引导视觉注意力正确地聚焦到目标上，我们提出了一种具有隐式地标化范式的重建VLA模型——ReconVLA。基于模型的视觉输出，扩散变换器旨在重建图像的目标注视区域，对应于被操作的目标物体。这一过程促使VLA模型学习精细的表示并准确分配视觉注意力，从而有效利用任务相关的视觉信息并进行精确操作。此外，我们编制了一个包含超过10万个轨迹和200万个数据样本的大规模预训练数据集，进一步提升了模型在视觉重建方面的泛化能力。在仿真和真实世界中的广泛实验显示了我们隐式地标化方法的优越性，展示了其精确操作和泛化的能力。项目页面见此链接：https URL。 

---
# WiFi-based Global Localization in Large-Scale Environments Leveraging Structural Priors from osmAG 

**Title (ZH)**: 基于OSMAG结构先验的大型环境WiFi全局定位 

**Authors**: Xu Ma, Jiajie Zhang, Fujing Xie, Sören Schwertfeger  

**Link**: [PDF](https://arxiv.org/pdf/2508.10144)  

**Abstract**: Global localization is essential for autonomous robotics, especially in indoor environments where the GPS signal is denied. We propose a novel WiFi-based localization framework that leverages ubiquitous wireless infrastructure and the OpenStreetMap Area Graph (osmAG) for large-scale indoor environments. Our approach integrates signal propagation modeling with osmAG's geometric and topological priors. In the offline phase, an iterative optimization algorithm localizes WiFi Access Points (APs) by modeling wall attenuation, achieving a mean localization error of 3.79 m (35.3\% improvement over trilateration). In the online phase, real-time robot localization uses the augmented osmAG map, yielding a mean error of 3.12 m in fingerprinted areas (8.77\% improvement over KNN fingerprinting) and 3.83 m in non-fingerprinted areas (81.05\% improvement). Comparison with a fingerprint-based method shows that our approach is much more space efficient and achieves superior localization accuracy, especially for positions where no fingerprint data are available. Validated across a complex 11,025 &m^2& multi-floor environment, this framework offers a scalable, cost-effective solution for indoor robotic localization, solving the kidnapped robot problem. The code and dataset are available at this https URL. 

**Abstract (ZH)**: 全球定位对于自主机器人至关重要，特别是在GPS信号被拒绝的室内环境中。我们提出了一种基于WiFi的新型定位框架，该框架利用了普遍存在的无线基础设施和OpenStreetMap区域图（osmAG）来应对大规模室内环境的定位问题。该方法将信号传播建模与osmAG的几何和拓扑先验相结合。在离线阶段，通过迭代优化算法定位WiFi接入点（AP），通过建模墙壁衰减，实现了均值定位误差3.79米（比三角测量提高了35.3%）。在线阶段，实时机器人定位使用增强的osmAG地图，指纹匹配区域的均值误差为3.12米（比KNN指纹识别提高了8.77%），非指纹匹配区域的均值误差为3.83米（比KNN指纹识别提高了81.05%）。与基于指纹的方法相比，我们的方法在空间效率和定位准确性方面表现出优越性，特别是在没有指纹数据的情况下。该框架已在复杂的大规模多层环境中得到验证，提供了一种可扩展且成本效益高的室内机器人定位解决方案，解决了被绑架的机器人问题。代码和数据集可在以下链接获取。 

---
# Agentic AI Frameworks: Architectures, Protocols, and Design Challenges 

**Title (ZH)**: 代理性人工智能框架：架构、协议与设计挑战 

**Authors**: Hana Derouiche, Zaki Brahmi, Haithem Mazeni  

**Link**: [PDF](https://arxiv.org/pdf/2508.10146)  

**Abstract**: The emergence of Large Language Models (LLMs) has ushered in a transformative paradigm in artificial intelligence, Agentic AI, where intelligent agents exhibit goal-directed autonomy, contextual reasoning, and dynamic multi-agent coordination. This paper provides a systematic review and comparative analysis of leading Agentic AI frameworks, including CrewAI, LangGraph, AutoGen, Semantic Kernel, Agno, Google ADK, and MetaGPT, evaluating their architectural principles, communication mechanisms, memory management, safety guardrails, and alignment with service-oriented computing paradigms. Furthermore, we identify key limitations, emerging trends, and open challenges in the field. To address the issue of agent communication, we conduct an in-depth analysis of protocols such as the Contract Net Protocol (CNP), Agent-to-Agent (A2A), Agent Network Protocol (ANP), and Agora. Our findings not only establish a foundational taxonomy for Agentic AI systems but also propose future research directions to enhance scalability, robustness, and interoperability. This work serves as a comprehensive reference for researchers and practitioners working to advance the next generation of autonomous AI systems. 

**Abstract (ZH)**: 大型语言模型(Large Language Models)的出现推动了人工智能领域的范式转变，即自主人工智能(Agentic AI)，其中智能代理表现出目标导向的自主性、情境推理和动态多代理协调。本文对CrewAI、LangGraph、AutoGen、Semantic Kernel、Agno、Google ADK和MetaGPT等领先自主人工智能框架进行了系统审查和比较分析，评估了它们的架构原理、通信机制、内存管理、安全护栏以及与面向服务计算范式的契合度。此外，我们还识别出该领域的关键局限性、新兴趋势和开放式挑战。为了解决代理通信问题，我们深入分析了合同网协议(CNP)、代理间通信(A2A)、代理网络协议(ANP)和Agora等协议。我们的研究不仅建立了自主人工智能系统的分类框架，还提出了未来的研究方向以增强可伸缩性、鲁棒性和互操作性。本研究为致力于推进下一代自主人工智能系统的研究人员和实践者提供了全面的参考。 

---
# Amazon Nova AI Challenge -- Trusted AI: Advancing secure, AI-assisted software development 

**Title (ZH)**: Amazon Nova AI挑战赛 —— 可信赖AI：推动安全的AI辅助软件开发 

**Authors**: Sattvik Sahai, Prasoon Goyal, Michael Johnston, Anna Gottardi, Yao Lu, Lucy Hu, Luke Dai, Shaohua Liu, Samyuth Sagi, Hangjie Shi, Desheng Zhang, Lavina Vaz, Leslie Ball, Maureen Murray, Rahul Gupta, Shankar Ananthakrishna  

**Link**: [PDF](https://arxiv.org/pdf/2508.10108)  

**Abstract**: AI systems for software development are rapidly gaining prominence, yet significant challenges remain in ensuring their safety. To address this, Amazon launched the Trusted AI track of the Amazon Nova AI Challenge, a global competition among 10 university teams to drive advances in secure AI. In the challenge, five teams focus on developing automated red teaming bots, while the other five create safe AI assistants. This challenge provides teams with a unique platform to evaluate automated red-teaming and safety alignment methods through head-to-head adversarial tournaments where red teams have multi-turn conversations with the competing AI coding assistants to test their safety alignment. Along with this, the challenge provides teams with a feed of high quality annotated data to fuel iterative improvement. Throughout the challenge, teams developed state-of-the-art techniques, introducing novel approaches in reasoning-based safety alignment, robust model guardrails, multi-turn jail-breaking, and efficient probing of large language models (LLMs). To support these efforts, the Amazon Nova AI Challenge team made substantial scientific and engineering investments, including building a custom baseline coding specialist model for the challenge from scratch, developing a tournament orchestration service, and creating an evaluation harness. This paper outlines the advancements made by university teams and the Amazon Nova AI Challenge team in addressing the safety challenges of AI for software development, highlighting this collaborative effort to raise the bar for AI safety. 

**Abstract (ZH)**: AI系统在软件开发中的应用迅速增长，但确保其安全性的挑战仍然存在。为应对这一挑战，亚马逊发起了亚马逊诺瓦AI挑战中的可信AI赛道，这是一个由10所大学团队参与的全球竞赛，旨在推动安全AI的发展。在此次挑战中，五支队伍专注于开发自动化红队机器人，而另外五支队伍则创建安全的AI助手。此挑战为团队提供了独特的平台，通过头对头的 adversarial 对抗 tournament，红队与竞争的AI编码助手进行多轮对话以测试其安全性对齐。此外，挑战还为团队提供了高质量标注数据流，以供迭代改进。在整个挑战过程中，团队研发了最先进的技术，引入了基于推理的安全对齐、稳健模型护栏、多轮逃狱以及高效探索大型语言模型的新方法。为了支持这些努力，亚马逊诺瓦AI挑战团队进行了大量的科学和工程投资，包括从零开始构建定制化的挑战专用编码专家模型，开发tournament管弦服务，以及创建评估框架。本文概述了大学团队和亚马逊诺瓦AI挑战团队在应对AI软件开发安全性挑战方面取得的进步，强调了这一合作努力以提高AI安全性标准。 

---
# EgoCross: Benchmarking Multimodal Large Language Models for Cross-Domain Egocentric Video Question Answering 

**Title (ZH)**: EgoCross：跨域主观视角多模态语言模型跨模态问答基准测试 

**Authors**: Yanjun Li, Yuqian Fu, Tianwen Qian, Qi'ao Xu, Silong Dai, Danda Pani Paudel, Luc Van Gool, Xiaoling Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10729)  

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) have significantly pushed the frontier of egocentric video question answering (EgocentricQA). However, existing benchmarks and studies are mainly limited to common daily activities such as cooking and cleaning. In contrast, real-world deployment inevitably encounters domain shifts, where target domains differ substantially in both visual style and semantic content. To bridge this gap, we introduce \textbf{EgoCross}, a comprehensive benchmark designed to evaluate the cross-domain generalization of MLLMs in EgocentricQA. EgoCross covers four diverse and challenging domains, including surgery, industry, extreme sports, and animal perspective, representing realistic and high-impact application scenarios. It comprises approximately 1,000 QA pairs across 798 video clips, spanning four key QA tasks: prediction, recognition, localization, and counting. Each QA pair provides both OpenQA and CloseQA formats to support fine-grained evaluation. Extensive experiments show that most existing MLLMs, whether general-purpose or egocentric-specialized, struggle to generalize to domains beyond daily life, highlighting the limitations of current models. Furthermore, we conduct several pilot studies, \eg, fine-tuning and reinforcement learning, to explore potential improvements. We hope EgoCross and our accompanying analysis will serve as a foundation for advancing domain-adaptive, robust egocentric video understanding. Data and codes will be released at: \href{this https URL}{this https URL.} 

**Abstract (ZH)**: 近期多模态大型语言模型（MLLMs）的进展显著推动了第一人称视频问答（EgocentricQA）的前沿。然而，现有的基准和研究主要集中在烹饪和清洁等日常活动中。相比之下，实际部署不可避免地会遇到领域偏移问题，目标领域在视觉风格和语义内容上存在显著差异。为了解决这一问题，我们引入了EgoCross，这是一个全面的基准，旨在评估MLLMs在EgocentricQA中的跨领域泛化能力。EgoCross涵盖了四个多样且具有挑战性的领域，包括手术、工业、极限运动和动物视角，代表了具有现实意义和高影响的应用场景。它包括约1,000组QA对，横跨798个视频片段，涵盖四个关键的QA任务：预测、识别、定位和计数。每组QA对都提供了开放式问答（OpenQA）和封闭式问答（CloseQA）格式，以支持精细评估。大量实验表明，大多数现有MLLMs，无论是通用的还是专门针对第一人称视角的，都难以泛化到日常生活之外的领域，突显了当前模型的局限性。此外，我们还进行了几项初步研究，例如微调和强化学习，以探索潜在的改进方法。我们希望EgoCross及其伴随的分析能够为推进适应性强的第一人称视频理解提供基础。数据和代码将在以下链接发布：\href{this https URL}{this https URL.} 

---
# Med-GLIP: Advancing Medical Language-Image Pre-training with Large-scale Grounded Dataset 

**Title (ZH)**: Med-GLIP: 基于大规模grounded数据集的医学语言-图像预训练 

**Authors**: Ziye Deng, Ruihan He, Jiaxiang Liu, Yuan Wang, Zijie Meng, Songtao Jiang, Yong Xie, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10528)  

**Abstract**: Medical image grounding aims to align natural language phrases with specific regions in medical images, serving as a foundational task for intelligent diagnosis, visual question answering (VQA), and automated report generation (MRG). However, existing research is constrained by limited modality coverage, coarse-grained annotations, and the absence of a unified, generalizable grounding framework. To address these challenges, we construct a large-scale medical grounding dataset Med-GLIP-5M comprising over 5.3 million region-level annotations across seven imaging modalities, covering diverse anatomical structures and pathological findings. The dataset supports both segmentation and grounding tasks with hierarchical region labels, ranging from organ-level boundaries to fine-grained lesions. Based on this foundation, we propose Med-GLIP, a modality-aware grounding framework trained on Med-GLIP-5M. Rather than relying on explicitly designed expert modules, Med-GLIP implicitly acquires hierarchical semantic understanding from diverse training data -- enabling it to recognize multi-granularity structures, such as distinguishing lungs from pneumonia lesions. Extensive experiments demonstrate that Med-GLIP consistently outperforms state-of-the-art baselines across multiple grounding benchmarks. Furthermore, integrating its spatial outputs into downstream tasks, including medical VQA and report generation, leads to substantial performance gains. Our dataset will be released soon. 

**Abstract (ZH)**: 医学图像接地旨在将自然语言短语与医学图像中的特定区域对齐，作为智能诊断、视觉问答（VQA）和自动化报告生成（MRG）的基础任务。然而，现有研究受限于模态覆盖有限、标注粗糙以及缺乏统一可泛化的接地框架。为应对这些挑战，我们构建了一个名为Med-GLIP-5M的大规模医学接地数据集，包含超过530万张区域级标注，涵盖了七种成像模态，全面覆盖了多样的解剖结构和病理发现。该数据集支持分割和接地任务，提供了从器官级边界到细微病灶的层次化区域标签。基于这一基础，我们提出Med-GLIP，这是一种基于Med-GLIP-5M训练的模态感知接地框架。Med-GLIP 不依赖于显式设计的专家模块，而是从多样化的训练数据中隐式获取层次语义理解，使其能够识别多粒度结构，如区分肺部与肺炎病灶。广泛实验表明，Med-GLIP 在多个接地基准测试中一贯优于现有最先进的基线。进一步将其实空间输出集成到下游任务，如医学VQA和报告生成中，可实现显著的性能提升。我们的数据集将很快发布。 

---
# Securing Agentic AI: Threat Modeling and Risk Analysis for Network Monitoring Agentic AI System 

**Title (ZH)**: 保障自主人工智能安全：网络监控自主人工智能系统威胁建模与风险分析 

**Authors**: Pallavi Zambare, Venkata Nikhil Thanikella, Ying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10043)  

**Abstract**: When combining Large Language Models (LLMs) with autonomous agents, used in network monitoring and decision-making systems, this will create serious security issues. In this research, the MAESTRO framework consisting of the seven layers threat modeling architecture in the system was used to expose, evaluate, and eliminate vulnerabilities of agentic AI. The prototype agent system was constructed and implemented, using Python, LangChain, and telemetry in WebSockets, and deployed with inference, memory, parameter tuning, and anomaly detection modules. Two practical threat cases were confirmed as follows: (i) resource denial of service by traffic replay denial-of-service, and (ii) memory poisoning by tampering with the historical log file maintained by the agent. These situations resulted in measurable levels of performance degradation, i.e. telemetry updates were delayed, and computational loads were increased, as a result of poor system adaptations. It was suggested to use a multilayered defense-in-depth approach with memory isolation, validation of planners and anomaly response systems in real-time. These findings verify that MAESTRO is viable in operational threat mapping, prospective risk scoring, and the basis of the resilient system design. The authors bring attention to the importance of the enforcement of memory integrity, paying attention to the adaptation logic monitoring, and cross-layer communication protection that guarantee the agentic AI reliability in adversarial settings. 

**Abstract (ZH)**: 当将大型语言模型（LLMs）与用于网络监控和决策系统的自主代理相结合时，会创建出严重安全问题。 

---
# Cognitive Cybersecurity for Artificial Intelligence: Guardrail Engineering with CCS-7 

**Title (ZH)**: 人工智能领域的认知网络安全：基于CCS-7的护栏工程 

**Authors**: Yuksel Aydin  

**Link**: [PDF](https://arxiv.org/pdf/2508.10033)  

**Abstract**: Language models exhibit human-like cognitive vulnerabilities, such as emotional framing, that escape traditional behavioral alignment. We present CCS-7 (Cognitive Cybersecurity Suite), a taxonomy of seven vulnerabilities grounded in human cognitive security research. To establish a human benchmark, we ran a randomized controlled trial with 151 participants: a "Think First, Verify Always" (TFVA) lesson improved cognitive security by +7.9% overall. We then evaluated TFVA-style guardrails across 12,180 experiments on seven diverse language model architectures. Results reveal architecture-dependent risk patterns: some vulnerabilities (e.g., identity confusion) are almost fully mitigated, while others (e.g., source interference) exhibit escalating backfire, with error rates increasing by up to 135% in certain models. Humans, in contrast, show consistent moderate improvement. These findings reframe cognitive safety as a model-specific engineering problem: interventions effective in one architecture may fail, or actively harm, another, underscoring the need for architecture-aware cognitive safety testing before deployment. 

**Abstract (ZH)**: 语言模型表现出类似人类的认知脆弱性，如情绪框架效应，这些脆弱性超出了传统的行为对齐。我们提出了认知网络安全套件CCS-7（Cognitive Cybersecurity Suite），该套件基于人类认知安全研究，包含七种漏洞分类。为了建立人类基准，我们进行了随机对照试验，共有151名参与者：一项“先思考，后验证”（TFVA）课程总体上提高了认知安全性7.9%。然后，我们评估了TFVA风格的防护措施在七种不同语言模型架构的12,180次实验中。结果显示，不同架构的风险模式各不相同：某些漏洞（如身份混淆）几乎完全得到缓解，而其他漏洞（如来源干扰）则表现出增强的回火现象，某些模型中的错误率最高可增加135%。相比之下，人类在各方面都表现出一致的适度改进。这些发现将认知安全性重新定义为模型特定的工程问题：一项在一种架构中有效的干预措施可能在另一种架构中失效，甚至造成损害，从而凸显了在部署前进行架构意识认知安全性测试的必要性。 

---
