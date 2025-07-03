# AC-DiT: Adaptive Coordination Diffusion Transformer for Mobile Manipulation 

**Title (ZH)**: 自适应协调扩散变换器：面向移动操作的任务适应协调扩散变换器 

**Authors**: Sixiang Chen, Jiaming Liu, Siyuan Qian, Han Jiang, Lily Li, Renrui Zhang, Zhuoyang Liu, Chenyang Gu, Chengkai Hou, Pengwei Wang, Zhongyuan Wang, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.01961)  

**Abstract**: Recently, mobile manipulation has attracted increasing attention for enabling language-conditioned robotic control in household tasks. However, existing methods still face challenges in coordinating mobile base and manipulator, primarily due to two limitations. On the one hand, they fail to explicitly model the influence of the mobile base on manipulator control, which easily leads to error accumulation under high degrees of freedom. On the other hand, they treat the entire mobile manipulation process with the same visual observation modality (e.g., either all 2D or all 3D), overlooking the distinct multimodal perception requirements at different stages during mobile manipulation. To address this, we propose the Adaptive Coordination Diffusion Transformer (AC-DiT), which enhances mobile base and manipulator coordination for end-to-end mobile manipulation. First, since the motion of the mobile base directly influences the manipulator's actions, we introduce a mobility-to-body conditioning mechanism that guides the model to first extract base motion representations, which are then used as context prior for predicting whole-body actions. This enables whole-body control that accounts for the potential impact of the mobile base's motion. Second, to meet the perception requirements at different stages of mobile manipulation, we design a perception-aware multimodal conditioning strategy that dynamically adjusts the fusion weights between various 2D visual images and 3D point clouds, yielding visual features tailored to the current perceptual needs. This allows the model to, for example, adaptively rely more on 2D inputs when semantic information is crucial for action prediction, while placing greater emphasis on 3D geometric information when precise spatial understanding is required. We validate AC-DiT through extensive experiments on both simulated and real-world mobile manipulation tasks. 

**Abstract (ZH)**: 近期，移动操作在使机器人能够在家庭任务中实现基于语言的控制方面引起了越来越多的关注。然而，现有方法在协调移动底座和 manipulator 时仍面临挑战，主要由于两个限制。一方面，它们未能明确建模移动底座对 manipulator 控制的影响，容易导致在高自由度下产生误差累积。另一方面，它们用相同的视觉观察模态（例如，全都是 2D 或全都是 3D）对待整个移动操作过程，忽略了移动操作不同阶段的多模态感知需求差异。为了解决这一问题，我们提出了自适应协调扩散变换器（AC-DiT），它增强了移动底座和 manipulator 的协调，以实现端到端的移动操作。首先，由于移动底座的运动直接影响 manipulator 的动作，我们引入了一种移动性到身体的条件机制，指导模型首先提取底座运动表示，并将其用作全局动作预测的上下文先验，从而使整个身体控制能够考虑到移动底座运动的潜在影响。其次，为了满足移动操作不同阶段的感知需求，我们设计了一种感知增强的多模态条件策略，动态调整各种 2D 视觉图像和 3D 点云之间的融合权重，以生成适应当前感知需求的视觉特征。这使模型能够，在语义信息对于动作预测至关重要时，更多地依赖 2D 输入；而在需要精确空间理解时，则更多地强调 3D 几何信息。我们通过在模拟和真实世界的移动操作任务中的广泛实验验证了 AC-DiT。 

---
# TypeTele: Releasing Dexterity in Teleoperation by Dexterous Manipulation Types 

**Title (ZH)**: TypeTele: 通过灵巧操作类型释放遥控操作中的灵巧性 

**Authors**: Yuhao Lin, Yi-Lin Wei, Haoran Liao, Mu Lin, Chengyi Xing, Hao Li, Dandan Zhang, Mark Cutkosky, Wei-Shi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.01857)  

**Abstract**: Dexterous teleoperation plays a crucial role in robotic manipulation for real-world data collection and remote robot control. Previous dexterous teleoperation mostly relies on hand retargeting to closely mimic human hand postures. However, these approaches may fail to fully leverage the inherent dexterity of dexterous hands, which can execute unique actions through their structural advantages compared to human hands. To address this limitation, we propose TypeTele, a type-guided dexterous teleoperation system, which enables dexterous hands to perform actions that are not constrained by human motion patterns. This is achieved by introducing dexterous manipulation types into the teleoperation system, allowing operators to employ appropriate types to complete specific tasks. To support this system, we build an extensible dexterous manipulation type library to cover comprehensive dexterous postures used in manipulation tasks. During teleoperation, we employ a MLLM (Multi-modality Large Language Model)-assisted type retrieval module to identify the most suitable manipulation type based on the specific task and operator commands. Extensive experiments of real-world teleoperation and imitation learning demonstrate that the incorporation of manipulation types significantly takes full advantage of the dexterous robot's ability to perform diverse and complex tasks with higher success rates. 

**Abstract (ZH)**: 精确灵巧遥控操作在真实世界数据收集和远程机器人控制中的作用不可或缺。以往的灵巧遥控操作主要依赖手部目标变换来密切模仿人类手部姿态。然而，这些方法可能无法充分利用灵巧手固有的灵活性，灵巧手利用其结构优势可以执行人类手无法完成的独特动作。为解决这一局限，我们提出了TypeTele，一种类型引导的灵巧遥控操作系统，使灵巧手能够执行不受人类运动模式限制的动作。这通过在遥控操作系统中引入灵巧操作类型来实现，允许操作员根据任务需求选择合适的类型。为此，我们构建了一个可扩展的灵巧操作类型库，以涵盖任务中使用的全面灵巧姿态。在遥控操作过程中，我们采用一个多模态大型语言模型辅助的操作类型检索模块，根据特定任务和操作员指令识别最合适的操作类型。实验结果表明，引入操作类型显著提高了灵巧机器人执行多样化和复杂任务的成功率。 

---
# MoIRA: Modular Instruction Routing Architecture for Multi-Task Robotics 

**Title (ZH)**: MoIRA：多任务机器人分模块指令路由架构 

**Authors**: Dmytro Kuzmenko, Nadiya Shvai  

**Link**: [PDF](https://arxiv.org/pdf/2507.01843)  

**Abstract**: Mixture-of-Experts (MoE) approaches have recently gained traction in robotics applications due to their ability to dynamically allocate computational resources and specialize sub-networks for distinct tasks or environmental contexts, enabling more efficient decision-making. Such systems often comprise sparsely activated experts combined under a single monolithic architecture and require a well-configured internal routing mechanism, which does not allow for selective low-level expert and router customization and requires additional training. We propose MoIRA, an architecture-agnostic modular MoE framework designed to coordinate existing experts with an external text-based router. MoIRA incorporates two zero-shot routing options: embedding-based similarity and prompt-driven language model inference. In our experiments, we choose large Vision-Language-Action models, gr00t-N1 and $\pi_0$, as the underlying experts, and train low-rank adapters for low-overhead inference. We evaluate MoIRA on various GR1 Humanoid tasks and LIBERO Spatial and Goal benchmarks, where it consistently outperforms generalist models and competes with other MoE pipelines. Additionally, we analyse the robustness of the proposed approach to the variations of the instructions. While relying solely on textual descriptions of tasks and experts, MoIRA demonstrates the practical viability of modular deployment with precise, low-effort routing and provides an alternative, scalable foundation for future multi-expert robotic systems. 

**Abstract (ZH)**: MoIRA：一种面向文本路由的模块化Mixture-of-Experts架构 

---
# Dynamic System Model Generation for Online Fault Detection and Diagnosis of Robotic Systems 

**Title (ZH)**: 基于机器人系统在线故障检测与诊断的动态系统模型生成 

**Authors**: Johannes Kohl, Georg Muck, Georg Jäger, Sebastian Zug  

**Link**: [PDF](https://arxiv.org/pdf/2507.01550)  

**Abstract**: With the rapid development of more complex robots, Fault Detection and Diagnosis (FDD) becomes increasingly harder. Especially the need for predetermined models and historic data is problematic because they do not encompass the dynamic and fast-changing nature of such systems. To this end, we propose a concept that actively generates a dynamic system model at runtime and utilizes it to locate root causes. The goal is to be applicable to all kinds of robotic systems that share a similar software design. Additionally, it should exhibit minimal overhead and enhance independence from expert attention. 

**Abstract (ZH)**: 随着更加复杂的机器人迅速发展，故障检测与诊断（FDD）变得 increasingly 更加困难。尤其是对预定义模型和历史数据的需求问题，因为这些模型和数据无法涵盖此类系统的动态和快速变化特性。为此，我们提出了一种概念，在运行时主动生成动态系统模型并利用该模型来定位根本原因。目标是使其适用于所有具有相似软件设计的各种机器人系统。此外，它应具备最小的开销并增强独立于专家关注的能力。 

---
# BioMARS: A Multi-Agent Robotic System for Autonomous Biological Experiments 

**Title (ZH)**: BioMARS：一种进行自主生物实验的多agent机器人系统 

**Authors**: Yibo Qiu, Zan Huang, Zhiyu Wang, Handi Liu, Yiling Qiao, Yifeng Hu, Shu'ang Sun, Hangke Peng, Ronald X Xu, Mingzhai Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.01485)  

**Abstract**: Large language models (LLMs) and vision-language models (VLMs) have the potential to transform biological research by enabling autonomous experimentation. Yet, their application remains constrained by rigid protocol design, limited adaptability to dynamic lab conditions, inadequate error handling, and high operational complexity. Here we introduce BioMARS (Biological Multi-Agent Robotic System), an intelligent platform that integrates LLMs, VLMs, and modular robotics to autonomously design, plan, and execute biological experiments. BioMARS uses a hierarchical architecture: the Biologist Agent synthesizes protocols via retrieval-augmented generation; the Technician Agent translates them into executable robotic pseudo-code; and the Inspector Agent ensures procedural integrity through multimodal perception and anomaly detection. The system autonomously conducts cell passaging and culture tasks, matching or exceeding manual performance in viability, consistency, and morphological integrity. It also supports context-aware optimization, outperforming conventional strategies in differentiating retinal pigment epithelial cells. A web interface enables real-time human-AI collaboration, while a modular backend allows scalable integration with laboratory hardware. These results highlight the feasibility of generalizable, AI-driven laboratory automation and the transformative role of language-based reasoning in biological research. 

**Abstract (ZH)**: 生物多智能体机器人系统（BioMARS）：基于大规模语言模型和视觉语言模型的自主生物实验平台 

---
# TriVLA: A Unified Triple-System-Based Unified Vision-Language-Action Model for General Robot Control 

**Title (ZH)**: TriVLA: 一种基于三元系统的一体化视觉-语言-动作模型用于通用机器人控制 

**Authors**: Zhenyang Liu, Yongchong Gu, Sixiao Zheng, Xiangyang Xue, Yanwei Fu  

**Link**: [PDF](https://arxiv.org/pdf/2507.01424)  

**Abstract**: Recent advancements in vision-language models (VLMs) for common-sense reasoning have led to the development of vision-language-action (VLA) models, enabling robots to perform generalized manipulation. Although existing autoregressive VLA methods design a specific architecture like dual-system to leverage large-scale pretrained knowledge, they tend to capture static information, often neglecting the dynamic aspects vital for embodied tasks. To this end, we propose TriVLA, a unified Vision-Language-Action model with a triple-system architecture for general robot control. The vision-language module (System 2) interprets the environment through vision and language instructions. The dynamics perception module (System 3) inherently produces visual representations that encompass both current static information and predicted future dynamics, thereby providing valuable guidance for policy learning. TriVLA utilizes pre-trained VLM model and fine-tunes pre-trained video foundation model on robot datasets along with internet human manipulation data. The subsequent policy learning module (System 1) generates fluid motor actions in real time. Experimental evaluation demonstrates that TriVLA operates at approximately 36 Hz and surpasses state-of-the-art imitation learning baselines on standard simulation benchmarks as well as challenging real-world manipulation tasks. 

**Abstract (ZH)**: 近期视觉语言模型在常识推理领域的进展推动了视觉语言动作模型的发展，使机器人能够执行通用的操控任务。虽然现有的自回归视觉语言动作方法设计了如双系统等特定架构以利用大规模预训练知识，但它们往往捕捉静态信息，忽视了执行体态任务所需的动力学方面。为此，我们提出了TriVLA——一种用于通用机器人控制的三系统统一视觉语言动作模型。视觉语言模块（系统2）通过视觉和语言指令解释环境。动力学感知模块（系统3）固有地产生包含当前静态信息和预测未来动力学的视觉表示，从而为策略学习提供有价值的指导。TriVLA 利用预训练的视觉语言模型，并在机器人数据集和网络上预训练的视频基础模型上进行微调。随后的策略学习模块（系统1）实时生成流畅的运动动作。实验评估表明，TriVLA 可以在约36 Hz的频率下运行，并在标准仿真基准测试以及具有挑战性的实际操控任务中均超过了最先进的模仿学习基线方法。 

---
# VLAD: A VLM-Augmented Autonomous Driving Framework with Hierarchical Planning and Interpretable Decision Process 

**Title (ZH)**: VLAD：一种基于层级规划和可解释决策过程的VLM增强自动驾驶框架 

**Authors**: Cristian Gariboldi, Hayato Tokida, Ken Kinjo, Yuki Asada, Alexander Carballo  

**Link**: [PDF](https://arxiv.org/pdf/2507.01284)  

**Abstract**: Recent advancements in open-source Visual Language Models (VLMs) such as LLaVA, Qwen-VL, and Llama have catalyzed extensive research on their integration with diverse systems. The internet-scale general knowledge encapsulated within these models presents significant opportunities for enhancing autonomous driving perception, prediction, and planning capabilities. In this paper we propose VLAD, a vision-language autonomous driving model, which integrates a fine-tuned VLM with VAD, a state-of-the-art end-to-end system. We implement a specialized fine-tuning approach using custom question-answer datasets designed specifically to improve the spatial reasoning capabilities of the model. The enhanced VLM generates high-level navigational commands that VAD subsequently processes to guide vehicle operation. Additionally, our system produces interpretable natural language explanations of driving decisions, thereby increasing transparency and trustworthiness of the traditionally black-box end-to-end architecture. Comprehensive evaluation on the real-world nuScenes dataset demonstrates that our integrated system reduces average collision rates by 31.82% compared to baseline methodologies, establishing a new benchmark for VLM-augmented autonomous driving systems. 

**Abstract (ZH)**: recent advancements in open-source视觉语言模型(VLMs)如LLaVA、Qwen-VL和Llama推动了其与不同系统的集成研究。这些模型中蕴含的大规模通用知识为自主驾驶感知、预测和规划能力的提升提供了重大机会。在本文中，我们提出了一种视觉语言自主驾驶模型(VLAD)，该模型将细调的VLM与最先进的端到端系统VAD集成。我们采用专门设计的问题-回答数据集实现了一种定制的细调方法，以提高模型的空间推理能力。增强的VLM生成高层次的导航命令，VAD随后处理这些命令以指导车辆操作。此外，我们的系统生成可解释的自然语言驾驶决策解释，从而增加传统黑盒端到端架构的透明性和可信度。在现实世界nuScenes数据集上的全面评估表明，与基线方法相比，我们集成的系统将平均碰撞率降低了31.82%，建立了VLM增强自主驾驶系统的新的基准。 

---
# Jump-Start Reinforcement Learning with Self-Evolving Priors for Extreme Monopedal Locomotion 

**Title (ZH)**: 自适应先验强化学习加速极端单腿运动控制 

**Authors**: Ziang Zheng, Guojian Zhan, Shiqi Liu, Yao Lyu, Tao Zhang, Shengbo Eben Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.01243)  

**Abstract**: Reinforcement learning (RL) has shown great potential in enabling quadruped robots to perform agile locomotion. However, directly training policies to simultaneously handle dual extreme challenges, i.e., extreme underactuation and extreme terrains, as in monopedal hopping tasks, remains highly challenging due to unstable early-stage interactions and unreliable reward feedback. To address this, we propose JumpER (jump-start reinforcement learning via self-evolving priors), an RL training framework that structures policy learning into multiple stages of increasing complexity. By dynamically generating self-evolving priors through iterative bootstrapping of previously learned policies, JumpER progressively refines and enhances guidance, thereby stabilizing exploration and policy optimization without relying on external expert priors or handcrafted reward shaping. Specifically, when integrated with a structured three-stage curriculum that incrementally evolves action modality, observation space, and task objective, JumpER enables quadruped robots to achieve robust monopedal hopping on unpredictable terrains for the first time. Remarkably, the resulting policy effectively handles challenging scenarios that traditional methods struggle to conquer, including wide gaps up to 60 cm, irregularly spaced stairs, and stepping stones with distances varying from 15 cm to 35 cm. JumpER thus provides a principled and scalable approach for addressing locomotion tasks under the dual challenges of extreme underactuation and extreme terrains. 

**Abstract (ZH)**: 基于自演进先验的跳跃强化学习（JumpER）：双极端挑战下四足机器人敏捷运动的学习框架 

---
# 2024 NASA SUITS Report: LLM-Driven Immersive Augmented Reality User Interface for Robotics and Space Exploration 

**Title (ZH)**: 2024 NASA SUITS报告：由LLM驱动的沉浸式增强现实用户界面在机器人学和太空探索中的应用 

**Authors**: Kathy Zhuang, Zixun Huang, Yukun Song, Rui Li, Yinuo Zhou, Allen Y. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.01206)  

**Abstract**: As modern computing advances, new interaction paradigms have emerged, particularly in Augmented Reality (AR), which overlays virtual interfaces onto physical objects. This evolution poses challenges in machine perception, especially for tasks like 3D object pose estimation in complex, dynamic environments. Our project addresses critical issues in human-robot interaction within mobile AR, focusing on non-intrusive, spatially aware interfaces. We present URSA, an LLM-driven immersive AR system developed for NASA's 2023-2024 SUITS challenge, targeting future spaceflight needs such as the Artemis missions. URSA integrates three core technologies: a head-mounted AR device (e.g., HoloLens) for intuitive visual feedback, voice control powered by large language models for hands-free interaction, and robot tracking algorithms that enable accurate 3D localization in dynamic settings. To enhance precision, we leverage digital twin localization technologies, using datasets like DTTD-Mobile and specialized hardware such as the ZED2 camera for real-world tracking under noise and occlusion. Our system enables real-time robot control and monitoring via an AR interface, even in the absence of ground-truth sensors--vital for hazardous or remote operations. Key contributions include: (1) a non-intrusive AR interface with LLM-based voice input; (2) a ZED2-based dataset tailored for non-rigid robotic bodies; (3) a Local Mission Control Console (LMCC) for mission visualization; (4) a transformer-based 6DoF pose estimator (DTTDNet) optimized for depth fusion and real-time tracking; and (5) end-to-end integration for astronaut mission support. This work advances digital twin applications in robotics, offering scalable solutions for both aerospace and industrial domains. 

**Abstract (ZH)**: 现代计算的进步催生了新的交互范式，尤其是在增强现实（AR）领域，AR将虚拟界面叠加在物理对象上。这一演变在复杂、动态环境下提出了机器感知的挑战，特别是对于3D物体姿态估计等任务。我们的项目针对移动AR中的人机交互，重点关注非侵入性和空间感知的界面。我们介绍了URSA，一个由大规模语言模型驱动的沉浸式AR系统，为NASA 2023-2024年SUITS挑战设计，旨在满足如阿尔忒弥斯任务等未来太空飞行需求。URSA结合了三项核心技术：头戴式AR设备（如HoloLens）提供直观的视觉反馈、基于大规模语言模型的声音控制实现无手操作、以及能够让机器人在动态环境中精确3D定位的算法。为了提高精度，我们利用数字孪生定位技术，并使用如DTTD-Mobile等数据集以及ZED2相机等专用硬件，在噪声和遮挡条件下进行现实世界跟踪。我们的系统允许通过AR界面进行实时机器人控制和监控，即便没有地面真实传感器——这对于危险或偏远操作至关重要。主要贡献包括：（1）基于大规模语言模型的非侵入性AR界面，支持语音输入；（2）针对非刚性机器人的ZED2数据集；（3）用于任务可视化的地方使命控制台（LMCC）；（4）基于变压器的6自由度姿态估计器（DTTDNet），优化深度融合和实时跟踪；（5）端到端集成以支持宇航员任务。这项工作推进了机器人领域的数字孪生应用，提供了适用于航空航天和工业领域的可扩展解决方案。 

---
# SonoGym: High Performance Simulation for Challenging Surgical Tasks with Robotic Ultrasound 

**Title (ZH)**: SonoGym：高性能机器人超声挑战手术任务模拟 

**Authors**: Yunke Ao, Masoud Moghani, Mayank Mittal, Manish Prajapat, Luohong Wu, Frederic Giraud, Fabio Carrillo, Andreas Krause, Philipp Fürnstahl  

**Link**: [PDF](https://arxiv.org/pdf/2507.01152)  

**Abstract**: Ultrasound (US) is a widely used medical imaging modality due to its real-time capabilities, non-invasive nature, and cost-effectiveness. Robotic ultrasound can further enhance its utility by reducing operator dependence and improving access to complex anatomical regions. For this, while deep reinforcement learning (DRL) and imitation learning (IL) have shown potential for autonomous navigation, their use in complex surgical tasks such as anatomy reconstruction and surgical guidance remains limited -- largely due to the lack of realistic and efficient simulation environments tailored to these tasks. We introduce SonoGym, a scalable simulation platform for complex robotic ultrasound tasks that enables parallel simulation across tens to hundreds of environments. Our framework supports realistic and real-time simulation of US data from CT-derived 3D models of the anatomy through both a physics-based and a generative modeling approach. Sonogym enables the training of DRL and recent IL agents (vision transformers and diffusion policies) for relevant tasks in robotic orthopedic surgery by integrating common robotic platforms and orthopedic end effectors. We further incorporate submodular DRL -- a recent method that handles history-dependent rewards -- for anatomy reconstruction and safe reinforcement learning for surgery. Our results demonstrate successful policy learning across a range of scenarios, while also highlighting the limitations of current methods in clinically relevant environments. We believe our simulation can facilitate research in robot learning approaches for such challenging robotic surgery applications. Dataset, codes, and videos are publicly available at this https URL. 

**Abstract (ZH)**: 超声波成像（US）因其实时能力、非侵入性和低成本而广泛应用于医学影像领域。机器人超声波成像可通过降低操作者依赖性和提高对复杂解剖区域的访问性进一步增强其应用价值。为此，尽管深度强化学习（DRL）和 imitation 学习（IL）在自主导航方面显示出潜力，但在如解剖重建和手术指导等复杂手术任务中的应用仍然有限——主要是由于缺乏针对这些任务的现实和高效的模拟环境。我们引入了 SonoGym，这是一种适用于复杂机器人超声波任务的可扩展模拟平台，可在数十到数百个环境中实现并行模拟。我们的框架通过基于物理和生成建模的方法，支持从来自 CT 标记的 3D 解剖模型实时模拟超声波数据。SonoGym 允许通过集成通用机器人平台和骨科末端执行器，为机器人骨科手术中的相关任务训练 DRL 和近期的 IL 代理（如视觉转换器和扩散策略）。我们进一步引入了处理历史依赖性奖励的次模强化学习（submodular DRL）方法，以及用于手术的强化学习安全性。我们的结果表明，无论是在何种场景下均成功地学习了策略，同时也指出了当前方法在临床相关环境中的局限性。我们相信，该模拟可以促进机器人学习方法在如此具有挑战性的机器人手术应用中的研究。有关数据集、代码和视频可在以下网址获取。 

---
# VISTA: Open-Vocabulary, Task-Relevant Robot Exploration with Online Semantic Gaussian Splatting 

**Title (ZH)**: VISTA：具有在线语义高斯点云化任务相关机器人探索的开放式词汇表方法 

**Authors**: Keiko Nagami, Timothy Chen, Javier Yu, Ola Shorinwa, Maximilian Adang, Carlyn Dougherty, Eric Cristofalo, Mac Schwager  

**Link**: [PDF](https://arxiv.org/pdf/2507.01125)  

**Abstract**: We present VISTA (Viewpoint-based Image selection with Semantic Task Awareness), an active exploration method for robots to plan informative trajectories that improve 3D map quality in areas most relevant for task completion. Given an open-vocabulary search instruction (e.g., "find a person"), VISTA enables a robot to explore its environment to search for the object of interest, while simultaneously building a real-time semantic 3D Gaussian Splatting reconstruction of the scene. The robot navigates its environment by planning receding-horizon trajectories that prioritize semantic similarity to the query and exploration of unseen regions of the environment. To evaluate trajectories, VISTA introduces a novel, efficient viewpoint-semantic coverage metric that quantifies both the geometric view diversity and task relevance in the 3D scene. On static datasets, our coverage metric outperforms state-of-the-art baselines, FisherRF and Bayes' Rays, in computation speed and reconstruction quality. In quadrotor hardware experiments, VISTA achieves 6x higher success rates in challenging maps, compared to baseline methods, while matching baseline performance in less challenging maps. Lastly, we show that VISTA is platform-agnostic by deploying it on a quadrotor drone and a Spot quadruped robot. Open-source code will be released upon acceptance of the paper. 

**Abstract (ZH)**: 基于视角的语义任务感知图像选择方法VISTA：用于提高与任务完成最相关的区域三维地图质量的主动探索方法 

---
# Environment-Aware and Human-Cooperative Swing Control for Lower-Limb Prostheses in Diverse Obstacle Scenarios 

**Title (ZH)**: 环境感知和人类协同摆动控制在多样障碍场景下的下肢假肢 

**Authors**: Haosen Xing, Haoran Ma, Sijin Zhang, Hartmut Geyer  

**Link**: [PDF](https://arxiv.org/pdf/2507.01111)  

**Abstract**: Current control strategies for powered lower limb prostheses often lack awareness of the environment and the user's intended interactions with it. This limitation becomes particularly apparent in complex terrains. Obstacle negotiation, a critical scenario exemplifying such challenges, requires both real-time perception of obstacle geometry and responsiveness to user intention about when and where to step over or onto, to dynamically adjust swing trajectories. We propose a novel control strategy that fuses environmental awareness and human cooperativeness: an on-board depth camera detects obstacles ahead of swing phase, prompting an elevated early-swing trajectory to ensure clearance, while late-swing control defers to natural biomechanical cues from the user. This approach enables intuitive stepping strategies without requiring unnatural movement patterns. Experiments with three non-amputee participants demonstrated 100 percent success across more than 150 step-overs and 30 step-ons with randomly placed obstacles of varying heights (4-16 cm) and distances (15-70 cm). By effectively addressing obstacle navigation -- a gateway challenge for complex terrain mobility -- our system demonstrates adaptability to both environmental constraints and user intentions, with promising applications across diverse locomotion scenarios. 

**Abstract (ZH)**: Powered Lower Limb Prostheses的当前控制策略通常缺乏对环境和用户意图的意识。特别是在复杂地形中，这一限制尤为明显。障碍物穿越，这一关键场景展示了此类挑战，需要实时感知障碍物几何形状并响应用户关于何时及何处跨越或踏上障碍物的意图，以动态调整摆腿轨迹。我们提出了一种融合环境意识和人类合作性的新型控制策略：车载深度相机在摆腿阶段前检测到障碍物，提示较高的早期摆腿轨迹以确保安全，而晚期摆腿控制则遵循用户的自然生物力学线索。这种方法使用户能够进行直观的步态策略，而无需采用不自然的运动模式。三项非截肢参与者的实验表明，超过150次跨越和30次踏上不同高度（4-16 cm）和距离（15-70 cm）的随机障碍物均成功。通过有效解决障碍物导航这一复杂地形移动的关键挑战，该系统展示了对环境约束和用户意图的适应能力，并在多种移动场景中具有广泛应用前景。 

---
# TD-MPC-Opt: Distilling Model-Based Multi-Task Reinforcement Learning Agents 

**Title (ZH)**: TD-MPC-Opt: 基于模型的多任务强化学习代理萃取方法 

**Authors**: Dmytro Kuzmenko, Nadiya Shvai  

**Link**: [PDF](https://arxiv.org/pdf/2507.01823)  

**Abstract**: We present a novel approach to knowledge transfer in model-based reinforcement learning, addressing the critical challenge of deploying large world models in resource-constrained environments. Our method efficiently distills a high-capacity multi-task agent (317M parameters) into a compact model (1M parameters) on the MT30 benchmark, significantly improving performance across diverse tasks. Our distilled model achieves a state-of-the-art normalized score of 28.45, surpassing the original 1M parameter model score of 18.93. This improvement demonstrates the ability of our distillation technique to capture and consolidate complex multi-task knowledge. We further optimize the distilled model through FP16 post-training quantization, reducing its size by $\sim$50\%. Our approach addresses practical deployment limitations and offers insights into knowledge representation in large world models, paving the way for more efficient and accessible multi-task reinforcement learning systems in robotics and other resource-constrained applications. Code available at this https URL. 

**Abstract (ZH)**: 一种基于模型的强化学习中的知识转移新方法：在资源受限环境中部署大型世界模型的关键挑战 

---
# What does really matter in image goal navigation? 

**Title (ZH)**: 图像目标导航中真正重要的因素是什么？ 

**Authors**: Gianluca Monaci, Philippe Weinzaepfel, Christian Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2507.01667)  

**Abstract**: Image goal navigation requires two different skills: firstly, core navigation skills, including the detection of free space and obstacles, and taking decisions based on an internal representation; and secondly, computing directional information by comparing visual observations to the goal image. Current state-of-the-art methods either rely on dedicated image-matching, or pre-training of computer vision modules on relative pose estimation. In this paper, we study whether this task can be efficiently solved with end-to-end training of full agents with RL, as has been claimed by recent work. A positive answer would have impact beyond Embodied AI and allow training of relative pose estimation from reward for navigation alone. In a large study we investigate the effect of architectural choices like late fusion, channel stacking, space-to-depth projections and cross-attention, and their role in the emergence of relative pose estimators from navigation training. We show that the success of recent methods is influenced up to a certain extent by simulator settings, leading to shortcuts in simulation. However, we also show that these capabilities can be transferred to more realistic setting, up to some extend. We also find evidence for correlations between navigation performance and probed (emerging) relative pose estimation performance, an important sub skill. 

**Abstract (ZH)**: 图像目标导航需要两种不同的技能：首先，核心导航技能，包括空旷空间和障碍物的检测以及基于内部表示的决策；其次，通过将视觉观察与目标图像进行比较来计算方向信息。当前最先进的方法要么依赖于专用的图像匹配，要么在相对位姿估计上进行计算机视觉模块的预训练。在这项研究中，我们探讨是否可以通过端到端的强化学习训练完整的代理来高效地解决这一任务，正如近期的一些工作所声称的那样。肯定的答案将对有代理人工智能产生影响，并允许仅通过导航奖励来训练相对位姿估计。在一项大规模的研究中，我们调查了诸如晚期融合、通道堆叠、空间到深度投影和交叉注意力等架构选择的效果及其在导航训练中促进相对位姿估计器出现的作用。我们证明了最近方法的成功在一定程度上受到模拟设置的影响，导致模拟中的捷径。但是，我们还表明，这些能力可以在一定程度上转移到更现实的设置中。我们还发现了导航性能与探测（新兴）相对位姿估计性能之间相关性的证据，这是一个重要的子技能。 

---
# RALLY: Role-Adaptive LLM-Driven Yoked Navigation for Agentic UAV Swarms 

**Title (ZH)**: RALLY：角色适配的基于LLM的联动导航以实现自主无人机群行动 

**Authors**: Ziyao Wang, Rongpeng Li, Sizhao Li, Yuming Xiang, Haiping Wang, Zhifeng Zhao, Honggang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.01378)  

**Abstract**: Intelligent control of Unmanned Aerial Vehicles (UAVs) swarms has emerged as a critical research focus, and it typically requires the swarm to navigate effectively while avoiding obstacles and achieving continuous coverage over multiple mission targets. Although traditional Multi-Agent Reinforcement Learning (MARL) approaches offer dynamic adaptability, they are hindered by the semantic gap in numerical communication and the rigidity of homogeneous role structures, resulting in poor generalization and limited task scalability. Recent advances in Large Language Model (LLM)-based control frameworks demonstrate strong semantic reasoning capabilities by leveraging extensive prior knowledge. However, due to the lack of online learning and over-reliance on static priors, these works often struggle with effective exploration, leading to reduced individual potential and overall system performance. To address these limitations, we propose a Role-Adaptive LLM-Driven Yoked navigation algorithm RALLY. Specifically, we first develop an LLM-driven semantic decision framework that uses structured natural language for efficient semantic communication and collaborative reasoning. Afterward, we introduce a dynamic role-heterogeneity mechanism for adaptive role switching and personalized decision-making. Furthermore, we propose a Role-value Mixing Network (RMIX)-based assignment strategy that integrates LLM offline priors with MARL online policies to enable semi-offline training of role selection strategies. Experiments in the Multi-Agent Particle Environment (MPE) environment and a Software-In-The-Loop (SITL) platform demonstrate that RALLY outperforms conventional approaches in terms of task coverage, convergence speed, and generalization, highlighting its strong potential for collaborative navigation in agentic multi-UAV systems. 

**Abstract (ZH)**: 智能控制无人机群 swarm 的智能控制已 emerge 作为关键研究重点，通常要求群组有效导航同时避开障碍物并在多个任务目标上实现持续覆盖。尽管传统的多智能体强化学习（MARL）方法具有动态适应性优势，但由于数值通信中的语义差距和同质角色结构的僵化，导致泛化性能差且任务扩展性受限。基于大型语言模型（LLM）的控制框架的最新进展展示了强大的语义推理能力，通过利用大量先验知识。然而，由于缺乏在线学习且过度依赖静态先验，这些工作在有效的探索方面常遇到困难，导致个体潜力和整体系统性能降低。为解决这些限制，我们提出了一种基于角色自适应 LLM 驱动同步导航算法 RALLY。具体而言，我们首先开发了一种使用结构化自然语言进行高效语义通信和协作推理的 LLM 驱动语义决策框架；随后引入了动态角色异质性机制以实现自适应角色切换和个人化决策；此外，我们提出了一种基于角色价值混合网络（RMIX）的任务分配策略，该策略整合 LLM 离线先验与 MARL 在线策略，以实现角色选择策略的部分离线训练。在多智能体粒子环境（MPE）环境和软件在环（SITL）平台上的实验表明，RALLY 在任务覆盖、收敛速度和泛化性能方面优于传统方法，突显了其在代理多无人机系统中协同导航的强大潜力。 

---
# SpecCLIP: Aligning and Translating Spectroscopic Measurements for Stars 

**Title (ZH)**: SpecCLIP: 对比和翻译光谱测量以匹配恒星 

**Authors**: Xiaosheng Zhao, Yang Huang, Guirong Xue, Xiao Kong, Jifeng Liu, Xiaoyu Tang, Timothy C. Beers, Yuan-Sen Ting, A-Li Luo  

**Link**: [PDF](https://arxiv.org/pdf/2507.01939)  

**Abstract**: In recent years, large language models (LLMs) have transformed natural language understanding through vast datasets and large-scale parameterization. Inspired by this success, we present SpecCLIP, a foundation model framework that extends LLM-inspired methodologies to stellar spectral analysis. Stellar spectra, akin to structured language, encode rich physical and chemical information about stars. By training foundation models on large-scale spectral datasets, our goal is to learn robust and informative embeddings that support diverse downstream applications. As a proof of concept, SpecCLIP involves pre-training on two spectral types--LAMOST low-resolution and Gaia XP--followed by contrastive alignment using the CLIP (Contrastive Language-Image Pre-training) framework, adapted to associate spectra from different instruments. This alignment is complemented by auxiliary decoders that preserve spectrum-specific information and enable translation (prediction) between spectral types, with the former achieved by maximizing mutual information between embeddings and input spectra. The result is a cross-spectrum framework enabling intrinsic calibration and flexible applications across instruments. We demonstrate that fine-tuning these models on moderate-sized labeled datasets improves adaptability to tasks such as stellar-parameter estimation and chemical-abundance determination. SpecCLIP also enhances the accuracy and precision of parameter estimates benchmarked against external survey data. Additionally, its similarity search and cross-spectrum prediction capabilities offer potential for anomaly detection. Our results suggest that contrastively trained foundation models enriched with spectrum-aware decoders can advance precision stellar spectroscopy. 

**Abstract (ZH)**: 基于对比学习的大型语言模型扩展框架：SpecCLIP在恒星光谱分析中的应用 

---
# Towards culturally-appropriate conversational AI for health in the majority world: An exploratory study with citizens and professionals in Latin America 

**Title (ZH)**: 面向大多数国家的健康对话AI的文化适应性研究：拉丁美洲公民和专业人士的探索性研究 

**Authors**: Dorian Peters, Fernanda Espinoza, Marco da Re, Guido Ivetta, Luciana Benotti, Rafael A. Calvo  

**Link**: [PDF](https://arxiv.org/pdf/2507.01719)  

**Abstract**: There is justifiable interest in leveraging conversational AI (CAI) for health across the majority world, but to be effective, CAI must respond appropriately within culturally and linguistically diverse contexts. Therefore, we need ways to address the fact that current LLMs exclude many lived experiences globally. Various advances are underway which focus on top-down approaches and increasing training data. In this paper, we aim to complement these with a bottom-up locally-grounded approach based on qualitative data collected during participatory workshops in Latin America. Our goal is to construct a rich and human-centred understanding of: a) potential areas of cultural misalignment in digital health; b) regional perspectives on chatbots for health and c)strategies for creating culturally-appropriate CAI; with a focus on the understudied Latin American context. Our findings show that academic boundaries on notions of culture lose meaning at the ground level and technologies will need to engage with a broader framework; one that encapsulates the way economics, politics, geography and local logistics are entangled in cultural experience. To this end, we introduce a framework for 'Pluriversal Conversational AI for Health' which allows for the possibility that more relationality and tolerance, rather than just more data, may be called for. 

**Abstract (ZH)**: 在全世界范围内利用对话人工智能（CAI）促进健康是有充分理由的，但为了有效，CAI 必须在多元的文化和语言背景下做出恰当的响应。因此，我们需要解决当前语言模型排除全球许多生活体验的问题。尽管存在各种以自上而下方法和增加训练数据为主的进展，我们在本文中旨在通过在拉丁美洲参与式工作坊中收集的定性数据，采用自下而上的当地扎根方法来进行补充。我们的目标是构建一个丰富的人本中心的理解：a) 数字健康中的潜在文化错位区域；b) 区域内的聊天机器人在健康领域的视角；c) 创造适合文化背景的CAI的策略；重点关注相对较少研究的拉丁美洲背景。我们的发现表明，关于文化的学术边界在基层失去了意义，技术需要与更广泛的框架相结合；这一框架涵盖了经济、政治、地理和当地物流与文化体验交织的方式。为了实现这一目标，我们介绍了“整体对话人工智能（CAI）促进健康”框架，该框架允许更多的关系性和包容性，而不仅仅是更多的数据。 

---
# Quantum-Assisted Automatic Path-Planning for Robotic Quality Inspection in Industry 4.0 

**Title (ZH)**: 量子辅助自动路径规划的工业4.0机器人质量检测研究 

**Authors**: Eneko Osaba, Estibaliz Garrote, Pablo Miranda-Rodriguez, Alessia Ciacco, Itziar Cabanes, Aitziber Mancisidor  

**Link**: [PDF](https://arxiv.org/pdf/2507.01462)  

**Abstract**: This work explores the application of hybrid quantum-classical algorithms to optimize robotic inspection trajectories derived from Computer-Aided Design (CAD) models in industrial settings. By modeling the task as a 3D variant of the Traveling Salesman Problem, incorporating incomplete graphs and open-route constraints, this study evaluates the performance of two D-Wave-based solvers against classical methods such as GUROBI and Google OR-Tools. Results across five real-world cases demonstrate competitive solution quality with significantly reduced computation times, highlighting the potential of quantum approaches in automation under Industry 4.0. 

**Abstract (ZH)**: 本研究探讨了混合量子-经典算法在工业环境中基于计算机辅助设计（CAD）模型优化机器人检测轨迹的应用。通过将任务建模为三维旅行商问题的变体，并纳入不完整图和开路约束，本文评估了两种D-Wave基于的求解器与GUROBI和Google OR-Tools等经典方法的性能。在五种实际案例中的结果表明，量子方法在解决方案质量上具有竞争力，同时大幅减少了计算时间，突显了在工业4.0背景下自动化中量子方法的潜力。 

---
# Distributional Soft Actor-Critic with Diffusion Policy 

**Title (ZH)**: 分布软actor-critic结合扩散策略 

**Authors**: Tong Liu, Yinuo Wang, Xujie Song, Wenjun Zou, Liangfa Chen, Likun Wang, Bin Shuai, Jingliang Duan, Shengbo Eben Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.01381)  

**Abstract**: Reinforcement learning has been proven to be highly effective in handling complex control tasks. Traditional methods typically use unimodal distributions, such as Gaussian distributions, to model the output of value distributions. However, unimodal distribution often and easily causes bias in value function estimation, leading to poor algorithm performance. This paper proposes a distributional reinforcement learning algorithm called DSAC-D (Distributed Soft Actor Critic with Diffusion Policy) to address the challenges of estimating bias in value functions and obtaining multimodal policy representations. A multimodal distributional policy iteration framework that can converge to the optimal policy was established by introducing policy entropy and value distribution function. A diffusion value network that can accurately characterize the distribution of multi peaks was constructed by generating a set of reward samples through reverse sampling using a diffusion model. Based on this, a distributional reinforcement learning algorithm with dual diffusion of the value network and the policy network was derived. MuJoCo testing tasks demonstrate that the proposed algorithm not only learns multimodal policy, but also achieves state-of-the-art (SOTA) performance in all 9 control tasks, with significant suppression of estimation bias and total average return improvement of over 10\% compared to existing mainstream algorithms. The results of real vehicle testing show that DSAC-D can accurately characterize the multimodal distribution of different driving styles, and the diffusion policy network can characterize multimodal trajectories. 

**Abstract (ZH)**: 分布式软演员批评与扩散策略算法（DSAC-D）：针对价值函数偏差估计和多模态策略表示的分布强化学习算法 

---
# Reasoner for Real-World Event Detection: Scaling Reinforcement Learning via Adaptive Perplexity-Aware Sampling Strategy 

**Title (ZH)**: 面向真实世界事件检测的推理器：通过自适应困惑度感知采样策略扩展强化学习 

**Authors**: Xiaoyun Zhang, Jingqing Ruan, Xing Ma, Yawen Zhu, Jiansong Chen, Ke Zeng, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2507.01327)  

**Abstract**: Detecting abnormal events in real-world customer service dialogues is highly challenging due to the complexity of business data and the dynamic nature of customer interactions. Moreover, models must demonstrate strong out-of-domain (OOD) generalization to enable rapid adaptation across different business scenarios and maximize commercial value. In this work, we propose a novel Adaptive Perplexity-Aware Reinforcement Learning (APARL) framework that leverages the advanced reasoning capabilities of large language models for abnormal event detection. APARL introduces a dual-loop dynamic curriculum learning architecture, enabling the model to progressively focus on more challenging samples as its proficiency increases. This design effectively addresses performance bottlenecks and significantly enhances OOD transferability. Extensive evaluations on food delivery dialogue tasks show that our model achieves significantly enhanced adaptability and robustness, attaining the highest F1 score with an average improvement of 17.19\%, and an average improvement of 9.59\% in OOD transfer tests. This method provides a superior solution for industrial deployment of anomaly detection models, contributing to improved operational efficiency and commercial benefits. 

**Abstract (ZH)**: 在现实世界客户服务对话中检测异常事件具有高度挑战性，由于商业数据的复杂性和客户交互的动态性。此外，模型必须表现出强大的领域外（OOD）泛化能力，以实现不同业务场景下的快速适应并最大化商业价值。在本文中，我们提出了一种新颖的自适应困惑度意识强化学习（APARL）框架，利用大规模语言模型的高级推理能力进行异常事件检测。APARL引入了一种双环动态课程学习架构，使模型能够随着其能力的提升逐步关注更具挑战性的样本。该设计有效解决了性能瓶颈并显著提升了领域外泛化能力。在食物配送对话任务上的广泛评估表明，我们的模型在适应性和鲁棒性方面取得了显著增强，平均F1分数提高了17.19%，领域外泛化测试的平均提高率为9.59%。该方法为异常检测模型的工业部署提供了卓越的解决方案，有助于提高运营效率和商业收益。 

---
# Geometry-aware 4D Video Generation for Robot Manipulation 

**Title (ZH)**: 面向几何aware的4D视频生成在机器人操作中 

**Authors**: Zeyi Liu, Shuang Li, Eric Cousineau, Siyuan Feng, Benjamin Burchfiel, Shuran Song  

**Link**: [PDF](https://arxiv.org/pdf/2507.01099)  

**Abstract**: Understanding and predicting the dynamics of the physical world can enhance a robot's ability to plan and interact effectively in complex environments. While recent video generation models have shown strong potential in modeling dynamic scenes, generating videos that are both temporally coherent and geometrically consistent across camera views remains a significant challenge. To address this, we propose a 4D video generation model that enforces multi-view 3D consistency of videos by supervising the model with cross-view pointmap alignment during training. This geometric supervision enables the model to learn a shared 3D representation of the scene, allowing it to predict future video sequences from novel viewpoints based solely on the given RGB-D observations, without requiring camera poses as inputs. Compared to existing baselines, our method produces more visually stable and spatially aligned predictions across multiple simulated and real-world robotic datasets. We further show that the predicted 4D videos can be used to recover robot end-effector trajectories using an off-the-shelf 6DoF pose tracker, supporting robust robot manipulation and generalization to novel camera viewpoints. 

**Abstract (ZH)**: 理解并预测物理世界的动力学可以增强机器人在复杂环境中的计划和交互能力。虽然近年来的视频生成模型在建模动态场景方面展现出了强大的潜力，但生成在不同摄像机视角下时空一致且几何一致的视频仍然是一个重大挑战。为了应对这一挑战，我们提出了一种4D视频生成模型，在训练过程中通过监督模型的跨视图点图对齐来强制多视图三维一致性。这种几何监督使模型能够学习场景的共享三维表示，从而仅根据给定的RGB-D观察结果预测新的视角下的未来视频序列，不需要输入摄像机姿态。与现有基线方法相比，我们的方法在多个模拟和真实世界机器人数据集中产生了更加视觉稳定且空间对齐的预测。进一步研究表明，预测的4D视频可以使用现成的6DoF姿态追踪器来恢复机器人末端执行器轨迹，支持鲁棒的机器人操作并在新的摄像机视角上泛化。 

---
