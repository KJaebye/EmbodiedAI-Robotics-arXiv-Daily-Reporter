# InstructVLA: Vision-Language-Action Instruction Tuning from Understanding to Manipulation 

**Title (ZH)**: InstructVLA：从理解到操作的视觉-语言-动作指令调优 

**Authors**: Shuai Yang, Hao Li, Yilun Chen, Bin Wang, Yang Tian, Tai Wang, Hanqing Wang, Feng Zhao, Yiyi Liao, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17520)  

**Abstract**: To operate effectively in the real world, robots must integrate multimodal reasoning with precise action generation. However, existing vision-language-action (VLA) models often sacrifice one for the other, narrow their abilities to task-specific manipulation data, and suffer catastrophic forgetting of pre-trained vision-language capabilities. To bridge this gap, we introduce InstructVLA, an end-to-end VLA model that preserves the flexible reasoning of large vision-language models (VLMs) while delivering leading manipulation performance. InstructVLA introduces a novel training paradigm, Vision-Language-Action Instruction Tuning (VLA-IT), which employs multimodal training with mixture-of-experts adaptation to jointly optimize textual reasoning and action generation on both standard VLM corpora and a curated 650K-sample VLA-IT dataset. On in-domain SimplerEnv tasks, InstructVLA achieves 30.5% improvement over SpatialVLA. To evaluate generalization, we introduce SimplerEnv-Instruct, an 80-task benchmark requiring closed-loop control and high-level instruction understanding, where it outperforms a fine-tuned OpenVLA by 92% and an action expert aided by GPT-4o by 29%. Additionally, InstructVLA surpasses baseline VLMs on multimodal tasks and exhibits inference-time scaling by leveraging textual reasoning to boost manipulation performance in both simulated and real-world settings. These results demonstrate InstructVLA's potential for bridging intuitive and steerable human-robot interaction with efficient policy learning. 

**Abstract (ZH)**: 基于指令的多模态视觉-语言-动作模型：融合灵活推理与精准操作能力 

---
# The Wilhelm Tell Dataset of Affordance Demonstrations 

**Title (ZH)**: William Tell数据集中的功能演示 

**Authors**: Rachel Ringe, Mihai Pomarlan, Nikolaos Tsiogkas, Stefano De Giorgis, Maria Hedblom, Rainer Malaka  

**Link**: [PDF](https://arxiv.org/pdf/2507.17401)  

**Abstract**: Affordances - i.e. possibilities for action that an environment or objects in it provide - are important for robots operating in human environments to perceive. Existing approaches train such capabilities on annotated static images or shapes. This work presents a novel dataset for affordance learning of common household tasks. Unlike previous approaches, our dataset consists of video sequences demonstrating the tasks from first- and third-person perspectives, along with metadata about the affordances that are manifested in the task, and is aimed towards training perception systems to recognize affordance manifestations. The demonstrations were collected from several participants and in total record about seven hours of human activity. The variety of task performances also allows studying preparatory maneuvers that people may perform for a task, such as how they arrange their task space, which is also relevant for collaborative service robots. 

**Abstract (ZH)**: 环境提供的功能（即环境或其中物体提供的行动可能性）对于机器人在人类环境中操作非常重要。现有的方法通过标注静态图像或形状来训练这些能力。本研究提出了一种新的数据集，用于学习常见家庭任务的功能。不同于先前的方法，我们的数据集包括从第一人称和第三人称视角展示任务的视频序列，以及任务中表现的功能的元数据，并旨在训练感知系统以识别功能的体现。演示从多个参与者处收集，总共记录了大约七小时的人类活动。任务执行的多样性还允许研究人们为任务可能进行的预备动作，如他们如何规划任务空间，这也对协作服务机器人很重要。 

---
# Language-Conditioned Open-Vocabulary Mobile Manipulation with Pretrained Models 

**Title (ZH)**: 基于预训练模型的语言条件化开放词汇移动操作 

**Authors**: Shen Tan, Dong Zhou, Xiangyu Shao, Junqiao Wang, Guanghui Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.17379)  

**Abstract**: Open-vocabulary mobile manipulation (OVMM) that involves the handling of novel and unseen objects across different workspaces remains a significant challenge for real-world robotic applications. In this paper, we propose a novel Language-conditioned Open-Vocabulary Mobile Manipulation framework, named LOVMM, incorporating the large language model (LLM) and vision-language model (VLM) to tackle various mobile manipulation tasks in household environments. Our approach is capable of solving various OVMM tasks with free-form natural language instructions (e.g. "toss the food boxes on the office room desk to the trash bin in the corner", and "pack the bottles from the bed to the box in the guestroom"). Extensive experiments simulated in complex household environments show strong zero-shot generalization and multi-task learning abilities of LOVMM. Moreover, our approach can also generalize to multiple tabletop manipulation tasks and achieve better success rates compared to other state-of-the-art methods. 

**Abstract (ZH)**: 面向开放词汇的移动 manipulation (OVMM)：一种结合大规模语言模型和视觉语言模型的语言条件化开放词汇移动 manipulation 框架 

---
# An Exploratory Study on Human-Robot Interaction using Semantics-based Situational Awareness 

**Title (ZH)**: 基于语义的情境意识探索性研究：人机交互 

**Authors**: Tianshu Ruan, Aniketh Ramesh, Rustam Stolkin, Manolis Chiou  

**Link**: [PDF](https://arxiv.org/pdf/2507.17376)  

**Abstract**: In this paper, we investigate the impact of high-level semantics (evaluation of the environment) on Human-Robot Teams (HRT) and Human-Robot Interaction (HRI) in the context of mobile robot deployments. Although semantics has been widely researched in AI, how high-level semantics can benefit the HRT paradigm is underexplored, often fuzzy, and intractable. We applied a semantics-based framework that could reveal different indicators of the environment (i.e. how much semantic information exists) in a mock-up disaster response mission. In such missions, semantics are crucial as the HRT should handle complex situations and respond quickly with correct decisions, where humans might have a high workload and stress. Especially when human operators need to shift their attention between robots and other tasks, they will struggle to build Situational Awareness (SA) quickly. The experiment suggests that the presented semantics: 1) alleviate the perceived workload of human operators; 2) increase the operator's trust in the SA; and 3) help to reduce the reaction time in switching the level of autonomy when needed. Additionally, we find that participants with higher trust in the system are encouraged by high-level semantics to use teleoperation mode more. 

**Abstract (ZH)**: 基于高阶语义对移动机器人部署环境下人机团队和人机交互的影响研究 

---
# Mobile Manipulation with Active Inference for Long-Horizon Rearrangement Tasks 

**Title (ZH)**: 基于主动推断的移动操作 dài 期重构任务 

**Authors**: Corrado Pezzato, Ozan Çatal, Toon Van de Maele, Riddhi J. Pitliya, Tim Verbelen  

**Link**: [PDF](https://arxiv.org/pdf/2507.17338)  

**Abstract**: Despite growing interest in active inference for robotic control, its application to complex, long-horizon tasks remains untested. We address this gap by introducing a fully hierarchical active inference architecture for goal-directed behavior in realistic robotic settings. Our model combines a high-level active inference model that selects among discrete skills realized via a whole-body active inference controller. This unified approach enables flexible skill composition, online adaptability, and recovery from task failures without requiring offline training. Evaluated on the Habitat Benchmark for mobile manipulation, our method outperforms state-of-the-art baselines across the three long-horizon tasks, demonstrating for the first time that active inference can scale to the complexity of modern robotics benchmarks. 

**Abstract (ZH)**: 尽管人们对在机器人控制中应用主动推断表现出浓厚兴趣，但其在复杂、长时 horizon 任务中的应用尚未得到验证。我们通过引入一种面向现实机器人环境的目标导向行为全层级主动推断架构来填补这一空白。该模型结合了高层主动推断模型和全身主动推断控制器实现的离散技能选择，这种统一的方法能够实现灵活的技能组合、在线适应性并在不需离线训练的情况下从任务失败中恢复。在 Habitat 移动操作基准测试中，我们的方法在三个长时 horizon 任务上的表现优于最新基线，首次证明了主动推断可以扩展到现代机器人基准的复杂性。 

---
# VLA-Touch: Enhancing Vision-Language-Action Models with Dual-Level Tactile Feedback 

**Title (ZH)**: VLA-触觉: 通过双层触觉反馈提升视觉-语言-动作模型 

**Authors**: Jianxin Bi, Kevin Yuchen Ma, Ce Hao, Mike Zheng Shou, Harold Soh  

**Link**: [PDF](https://arxiv.org/pdf/2507.17294)  

**Abstract**: Tactile feedback is generally recognized to be crucial for effective interaction with the physical world. However, state-of-the-art Vision-Language-Action (VLA) models lack the ability to interpret and use tactile signals, limiting their effectiveness in contact-rich tasks. Incorporating tactile feedback into these systems is challenging due to the absence of large multi-modal datasets. We present VLA-Touch, an approach that enhances generalist robot policies with tactile sensing \emph{without fine-tuning} the base VLA. Our method introduces two key innovations: (1) a pipeline that leverages a pretrained tactile-language model that provides semantic tactile feedback for high-level task planning, and (2) a diffusion-based controller that refines VLA-generated actions with tactile signals for contact-rich manipulation. Through real-world experiments, we demonstrate that our dual-level integration of tactile feedback improves task planning efficiency while enhancing execution precision. Code is open-sourced at \href{this https URL}{this URL}. 

**Abstract (ZH)**: 触觉反馈通常被认定为与物理世界有效交互的关键。然而，最先进的视觉-语言-动作（VLA）模型缺乏解释和利用触觉信号的能力，限制了其在接触密集型任务中的有效性。由于缺乏大规模多模态数据集，将触觉反馈纳入这些系统具有挑战性。我们提出了VLA-Touch方法，该方法在不微调基VLA模型的情况下，增强通用机器人策略的触觉感知能力。我们的方法引入了两项关键创新：（1）一个利用预训练触觉-语言模型的流水线，该模型为高层任务规划提供语义触觉反馈；（2）一种基于扩散的控制器，该控制器使用触觉信号细化VLA生成的动作，以适应接触密集型操作。通过实际实验，我们展示了我们的多级触觉反馈集成提高了任务规划效率并增强了执行精度。代码已在<此网址>开源。 

---
# Prolonging Tool Life: Learning Skillful Use of General-purpose Tools through Lifespan-guided Reinforcement Learning 

**Title (ZH)**: 延长工具寿命：通过生命周期导向的强化学习学习高效使用通用工具的方法 

**Authors**: Po-Yen Wu, Cheng-Yu Kuo, Yuki Kadokawa, Takamitsu Matsubara  

**Link**: [PDF](https://arxiv.org/pdf/2507.17275)  

**Abstract**: In inaccessible environments with uncertain task demands, robots often rely on general-purpose tools that lack predefined usage strategies. These tools are not tailored for particular operations, making their longevity highly sensitive to how they are used. This creates a fundamental challenge: how can a robot learn a tool-use policy that both completes the task and prolongs the tool's lifespan? In this work, we address this challenge by introducing a reinforcement learning (RL) framework that incorporates tool lifespan as a factor during policy optimization. Our framework leverages Finite Element Analysis (FEA) and Miner's Rule to estimate Remaining Useful Life (RUL) based on accumulated stress, and integrates the RUL into the RL reward to guide policy learning toward lifespan-guided behavior. To handle the fact that RUL can only be estimated after task execution, we introduce an Adaptive Reward Normalization (ARN) mechanism that dynamically adjusts reward scaling based on estimated RULs, ensuring stable learning signals. We validate our method across simulated and real-world tool use tasks, including Object-Moving and Door-Opening with multiple general-purpose tools. The learned policies consistently prolong tool lifespan (up to 8.01x in simulation) and transfer effectively to real-world settings, demonstrating the practical value of learning lifespan-guided tool use strategies. 

**Abstract (ZH)**: 在难以进入的环境中且任务需求不确定的情况下，机器人通常依赖于缺乏预定义使用策略的一般用途工具。这些工具未针对特定操作进行定制，使其使用寿命高度依赖于使用方式。这引发了一个基本挑战：机器人如何学习一种既能完成任务又能延长工具寿命的工具使用策略？在这项工作中，我们通过引入一个将工具寿命纳入策略优化过程的强化学习（RL）框架来应对这一挑战。该框架利用有限元分析（FEA）和 Miner’s Rule 来基于积累的应力估算剩余使用寿命（RUL），并将RUL整合到RL奖励中，以引导政策学习朝着寿命导向的行为。为处理在任务执行后才能估算RUL的情况，我们引入了一种自适应奖励归一化（ARN）机制，该机制根据估计的RUL动态调整奖励缩放，确保学习信号的稳定性。我们在模拟和真实世界工具使用任务中验证了该方法，包括涉及多种一般用途工具的Object-Moving和Door-Opening任务。学习到的策略一致地延长了工具寿命（在模拟中最多延长8.01倍），并在现实世界环境中有效转移，展示了学习寿命导向的工具使用策略的实用价值。 

---
# Towards Human-level Intelligence via Human-like Whole-Body Manipulation 

**Title (ZH)**: 通过类人全身 Manipulation 追求人类水平的智能 

**Authors**: Guang Gao, Jianan Wang, Jinbo Zuo, Junnan Jiang, Jingfan Zhang, Xianwen Zeng, Yuejiang Zhu, Lianyang Ma, Ke Chen, Minhua Sheng, Ruirui Zhang, Zhaohui An  

**Link**: [PDF](https://arxiv.org/pdf/2507.17141)  

**Abstract**: Building general-purpose intelligent robots has long been a fundamental goal of robotics. A promising approach is to mirror the evolutionary trajectory of humans: learning through continuous interaction with the environment, with early progress driven by the imitation of human behaviors. Achieving this goal presents three core challenges: (1) designing safe robotic hardware with human-level physical capabilities; (2) developing an intuitive and scalable whole-body teleoperation interface for data collection; and (3) creating algorithms capable of learning whole-body visuomotor policies from human demonstrations. To address these challenges in a unified framework, we propose Astribot Suite, a robot learning suite for whole-body manipulation aimed at general daily tasks across diverse environments. We demonstrate the effectiveness of our system on a wide range of activities that require whole-body coordination, extensive reachability, human-level dexterity, and agility. Our results show that Astribot's cohesive integration of embodiment, teleoperation interface, and learning pipeline marks a significant step towards real-world, general-purpose whole-body robotic manipulation, laying the groundwork for the next generation of intelligent robots. 

**Abstract (ZH)**: 构建通用智能机器人一直是机器人学的基本目标。一种有前景的方法是模仿人类的进化轨迹：通过持续与环境互动学习，早期进展通过模仿人类行为驱动。实现这一目标面临三大核心挑战：（1）设计具有人类级别物理能力的安全机器人硬件；（2）开发直观且可扩展的全身远程操作界面以收集数据；（3）创建能够从人类示范学习全身视听运动策略的算法。为了在一个统一框架中应对这些挑战，我们提出Astribot Suite，一个面向通用日常任务的全身操纵机器人学习套件，适用于多种环境。我们展示了我们的系统在多种需要全身协调、广泛可达性、人类级灵巧性和敏捷性的活动中具有有效性。我们的结果表明，Astribot 有效地整合了本体、远程操作界面和学习管道，标志着向实用化、通用化全身机器人操纵迈出的重要一步，为新一代智能机器人奠定了基础。 

---
# Deformable Cluster Manipulation via Whole-Arm Policy Learning 

**Title (ZH)**: 基于整臂策略学习的可变形簇集操作 

**Authors**: Jayadeep Jacob, Wenzheng Zhang, Houston Warren, Paulo Borges, Tirthankar Bandyopadhyay, Fabio Ramos  

**Link**: [PDF](https://arxiv.org/pdf/2507.17085)  

**Abstract**: Manipulating clusters of deformable objects presents a substantial challenge with widespread applicability, but requires contact-rich whole-arm interactions. A potential solution must address the limited capacity for realistic model synthesis, high uncertainty in perception, and the lack of efficient spatial abstractions, among others. We propose a novel framework for learning model-free policies integrating two modalities: 3D point clouds and proprioceptive touch indicators, emphasising manipulation with full body contact awareness, going beyond traditional end-effector modes. Our reinforcement learning framework leverages a distributional state representation, aided by kernel mean embeddings, to achieve improved training efficiency and real-time inference. Furthermore, we propose a novel context-agnostic occlusion heuristic to clear deformables from a target region for exposure tasks. We deploy the framework in a power line clearance scenario and observe that the agent generates creative strategies leveraging multiple arm links for de-occlusion. Finally, we perform zero-shot sim-to-real policy transfer, allowing the arm to clear real branches with unknown occlusion patterns, unseen topology, and uncertain dynamics. 

**Abstract (ZH)**: 一种基于3D点云和本体感觉指示的新型无模型策略框架：面向全触觉接触的可变形对象 manipulation，及其在电力线清障中的应用 

---
# Multi-agent Reinforcement Learning for Robotized Coral Reef Sample Collection 

**Title (ZH)**: 基于多代理 reinforcement 学习的机器人化珊瑚礁样本采集 

**Authors**: Daniel Correa, Tero Kaarlela, Jose Fuentes, Paulo Padrao, Alain Duran, Leonardo Bobadilla  

**Link**: [PDF](https://arxiv.org/pdf/2507.16941)  

**Abstract**: This paper presents a reinforcement learning (RL) environment for developing an autonomous underwater robotic coral sampling agent, a crucial coral reef conservation and research task. Using software-in-the-loop (SIL) and hardware-in-the-loop (HIL), an RL-trained artificial intelligence (AI) controller is developed using a digital twin (DT) in simulation and subsequently verified in physical experiments. An underwater motion capture (MOCAP) system provides real-time 3D position and orientation feedback during verification testing for precise synchronization between the digital and physical domains. A key novelty of this approach is the combined use of a general-purpose game engine for simulation, deep RL, and real-time underwater motion capture for an effective zero-shot sim-to-real strategy. 

**Abstract (ZH)**: 本文提出了一种强化学习（RL）环境，用于开发自主水下机器人珊瑚采样代理，这是珊瑚礁保护与研究中的一项关键任务。通过软件在环（SIL）和硬件在环（HIL）的方法，使用数字孪生（DT）在仿真中开发了一个RL训练的AI控制器，并在物理实验中进行了验证。基于水下运动捕捉（MOCAP）系统提供了验证测试过程中的实时3D位置和姿态反馈，以实现数字域和物理域的精确同步。该方法的关键 novelty 是结合使用通用游戏引擎进行仿真、深度RL和实时水下运动捕捉，以实现有效的零样本仿真实现策略。 

---
# AquaChat: An LLM-Guided ROV Framework for Adaptive Inspection of Aquaculture Net Pens 

**Title (ZH)**: AquaChat：一种基于LLM的ROV框架，用于适应性检查水产养殖网箱 

**Authors**: Waseem Akram, Muhayy Ud Din, Abdelhaleem Saad, Irfan Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2507.16841)  

**Abstract**: Inspection of aquaculture net pens is essential for maintaining the structural integrity, biosecurity, and operational efficiency of fish farming systems. Traditional inspection approaches rely on pre-programmed missions or manual control, offering limited adaptability to dynamic underwater conditions and user-specific demands. In this study, we propose AquaChat, a novel Remotely Operated Vehicle (ROV) framework that integrates Large Language Models (LLMs) for intelligent and adaptive net pen inspection. The system features a multi-layered architecture: (1) a high-level planning layer that interprets natural language user commands using an LLM to generate symbolic task plans; (2) a mid-level task manager that translates plans into ROV control sequences; and (3) a low-level motion control layer that executes navigation and inspection tasks with precision. Real-time feedback and event-triggered replanning enhance robustness in challenging aquaculture environments. The framework is validated through experiments in both simulated and controlled aquatic environments representative of aquaculture net pens. Results demonstrate improved task flexibility, inspection accuracy, and operational efficiency. AquaChat illustrates the potential of integrating language-based AI with marine robotics to enable intelligent, user-interactive inspection systems for sustainable aquaculture operations. 

**Abstract (ZH)**: 水产养殖网笼的检查对于保持养殖系统的结构完整性、生物安全性和运营效率至关重要。传统的检查方法依赖于预编程的任务或手动控制，对动态的水下条件和用户特定需求适应性有限。本研究提出了一种名为AquaChat的新型遥控潜水器（ROV）框架，该框架结合了大型语言模型（LLMs）实现智能和自适应的网笼检查。该系统具有多层架构：（1）高层规划层，使用LLM解析自然语言用户命令生成符号任务计划；（2）中间任务管理器，将计划转换为ROV控制序列；（3）低层运动控制层，精确执行导航和检查任务。实时反馈和事件触发的重新规划增强了在挑战性水产养殖环境中的鲁棒性。该框架通过在代表水产养殖网笼的模拟和受控水生环境中进行的实验得到了验证。结果表明，任务的灵活性、检查的准确性以及运营效率均有所提高。AquaChat展示了将基于语言的AI与海洋机器人结合以实现智能、用户互动的检查系统对可持续水产养殖运营的潜力。 

---
# PRIX: Learning to Plan from Raw Pixels for End-to-End Autonomous Driving 

**Title (ZH)**: PRIX: 从原始像素中学习规划以实现端到端自主驾驶 

**Authors**: Maciej K. Wozniak, Lianhang Liu, Yixi Cai, Patric Jensfelt  

**Link**: [PDF](https://arxiv.org/pdf/2507.17596)  

**Abstract**: While end-to-end autonomous driving models show promising results, their practical deployment is often hindered by large model sizes, a reliance on expensive LiDAR sensors and computationally intensive BEV feature representations. This limits their scalability, especially for mass-market vehicles equipped only with cameras. To address these challenges, we propose PRIX (Plan from Raw Pixels). Our novel and efficient end-to-end driving architecture operates using only camera data, without explicit BEV representation and forgoing the need for LiDAR. PRIX leverages a visual feature extractor coupled with a generative planning head to predict safe trajectories from raw pixel inputs directly. A core component of our architecture is the Context-aware Recalibration Transformer (CaRT), a novel module designed to effectively enhance multi-level visual features for more robust planning. We demonstrate through comprehensive experiments that PRIX achieves state-of-the-art performance on the NavSim and nuScenes benchmarks, matching the capabilities of larger, multimodal diffusion planners while being significantly more efficient in terms of inference speed and model size, making it a practical solution for real-world deployment. Our work is open-source and the code will be at this https URL. 

**Abstract (ZH)**: 基于原始像素的计划：端到端驾驶架构 PRIX 

---
# From Scan to Action: Leveraging Realistic Scans for Embodied Scene Understanding 

**Title (ZH)**: 从扫描到行动：利用现实扫描进行体域场景理解 

**Authors**: Anna-Maria Halacheva, Jan-Nico Zaech, Sombit Dey, Luc Van Gool, Danda Pani Paudel  

**Link**: [PDF](https://arxiv.org/pdf/2507.17585)  

**Abstract**: Real-world 3D scene-level scans offer realism and can enable better real-world generalizability for downstream applications. However, challenges such as data volume, diverse annotation formats, and tool compatibility limit their use. This paper demonstrates a methodology to effectively leverage these scans and their annotations. We propose a unified annotation integration using USD, with application-specific USD flavors. We identify challenges in utilizing holistic real-world scan datasets and present mitigation strategies. The efficacy of our approach is demonstrated through two downstream applications: LLM-based scene editing, enabling effective LLM understanding and adaptation of the data (80% success), and robotic simulation, achieving an 87% success rate in policy learning. 

**Abstract (ZH)**: 现实世界场景级三维扫描提供了真实的视觉效果，并能促进下游应用中的更好泛化能力。然而，数据量大、标注格式多样以及工具兼容性差等问题限制了其应用。本文提出了一种有效利用这些扫描及其标注的方法论。我们使用USD进行统一标注集成，并开发了适用于不同应用场景的USD变体。我们识别了利用整体现实世界扫描数据集面临的挑战，并提出了相应的缓解策略。通过两个下游应用，展示了我们方法的有效性：基于LLM的场景编辑，使LLM能够有效理解并适应数据（成功率为80%），以及机器人模拟，实现了在策略学习中87%的成功率。 

---
# VLM-Guided Visual Place Recognition for Planet-Scale Geo-Localization 

**Title (ZH)**: 行星规模地理定位的VLM引导视觉地点识别 

**Authors**: Sania Waheed, Na Min An, Michael Milford, Sarvapali D. Ramchurn, Shoaib Ehsan  

**Link**: [PDF](https://arxiv.org/pdf/2507.17455)  

**Abstract**: Geo-localization from a single image at planet scale (essentially an advanced or extreme version of the kidnapped robot problem) is a fundamental and challenging task in applications such as navigation, autonomous driving and disaster response due to the vast diversity of locations, environmental conditions, and scene variations. Traditional retrieval-based methods for geo-localization struggle with scalability and perceptual aliasing, while classification-based approaches lack generalization and require extensive training data. Recent advances in vision-language models (VLMs) offer a promising alternative by leveraging contextual understanding and reasoning. However, while VLMs achieve high accuracy, they are often prone to hallucinations and lack interpretability, making them unreliable as standalone solutions. In this work, we propose a novel hybrid geo-localization framework that combines the strengths of VLMs with retrieval-based visual place recognition (VPR) methods. Our approach first leverages a VLM to generate a prior, effectively guiding and constraining the retrieval search space. We then employ a retrieval step, followed by a re-ranking mechanism that selects the most geographically plausible matches based on feature similarity and proximity to the initially estimated coordinates. We evaluate our approach on multiple geo-localization benchmarks and show that it consistently outperforms prior state-of-the-art methods, particularly at street (up to 4.51%) and city level (up to 13.52%). Our results demonstrate that VLM-generated geographic priors in combination with VPR lead to scalable, robust, and accurate geo-localization systems. 

**Abstract (ZH)**: 从单张图像进行全球规模的地理定位（本质上是被劫持机器人问题的高级或极端版本）是导航、自动驾驶和灾害响应等领域中的一个基础且具有挑战性的任务，由于地理位置、环境条件和场景变化的广泛多样性。传统的基于检索的地理定位方法在可扩展性和感知性混叠方面存在困难，而基于分类的方法缺乏泛化能力且需要大量的训练数据。最近在视觉-语言模型（VLMs）方面的进展提供了一种有前途的替代方案，通过利用上下文理解和推理。然而，尽管VLMs在准确性方面表现出色，但它们往往容易产生幻觉且缺乏可解释性，使其作为独立解决方案不可靠。在本文中，我们提出了一种新颖的混合地理定位框架，将VLMs的优势与基于检索的视觉场所识别（VPR）方法结合起来。我们的方法首先利用VLM生成先验知识，有效地引导和限制检索搜索空间。随后采用检索步骤，并通过一个重新排名机制，根据特征相似性和与最初估计坐标的空间接近性，选择最地理上合理的一系列匹配。我们在多个地理定位基准上评估了我们的方法，并表明它在街道级别（高达4.51%）和城市级别（高达13.52%）上一致地优于先前的最佳方法。我们的结果表明，VLM生成的地理先验与VPR相结合，可以实现可扩展、稳健且准确的地理定位系统。 

---
# PIG-Nav: Key Insights for Pretrained Image Goal Navigation Models 

**Title (ZH)**: PIG-Nav：预训练图像目标导航模型的关键见解 

**Authors**: Jiansong Wan, Chengming Zhou, Jinkua Liu, Xiangge Huang, Xiaoyu Chen, Xiaohan Yi, Qisen Yang, Baiting Zhu, Xin-Qiang Cai, Lixing Liu, Rushuai Yang, Chuheng Zhang, Sherif Abdelfattah, Hayong Shin, Pushi Zhang, Li Zhao, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2507.17220)  

**Abstract**: Recent studies have explored pretrained (foundation) models for vision-based robotic navigation, aiming to achieve generalizable navigation and positive transfer across diverse environments while enhancing zero-shot performance in unseen settings. In this work, we introduce PIG-Nav (Pretrained Image-Goal Navigation), a new approach that further investigates pretraining strategies for vision-based navigation models and contributes in two key areas. Model-wise, we identify two critical design choices that consistently improve the performance of pretrained navigation models: (1) integrating an early-fusion network structure to combine visual observations and goal images via appropriately pretrained Vision Transformer (ViT) image encoder, and (2) introducing suitable auxiliary tasks to enhance global navigation representation learning, thus further improving navigation performance. Dataset-wise, we propose a novel data preprocessing pipeline for efficiently labeling large-scale game video datasets for navigation model training. We demonstrate that augmenting existing open navigation datasets with diverse gameplay videos improves model performance. Our model achieves an average improvement of 22.6% in zero-shot settings and a 37.5% improvement in fine-tuning settings over existing visual navigation foundation models in two complex simulated environments and one real-world environment. These results advance the state-of-the-art in pretrained image-goal navigation models. Notably, our model maintains competitive performance while requiring significantly less fine-tuning data, highlighting its potential for real-world deployment with minimal labeled supervision. 

**Abstract (ZH)**: Recent Studies on Pretrained Models for Vision-Based Robotic Navigation: Introducing PIG-Nav (Pretrained Image-Goal Navigation) 

---
# Evaluating Uncertainty and Quality of Visual Language Action-enabled Robots 

**Title (ZH)**: 评估视觉语言驱动机器人中的不确定性和质量 

**Authors**: Pablo Valle, Chengjie Lu, Shaukat Ali, Aitor Arrieta  

**Link**: [PDF](https://arxiv.org/pdf/2507.17049)  

**Abstract**: Visual Language Action (VLA) models are a multi-modal class of Artificial Intelligence (AI) systems that integrate visual perception, natural language understanding, and action planning to enable agents to interpret their environment, comprehend instructions, and perform embodied tasks autonomously. Recently, significant progress has been made to advance this field. These kinds of models are typically evaluated through task success rates, which fail to capture the quality of task execution and the mode's confidence in its decisions. In this paper, we propose eight uncertainty metrics and five quality metrics specifically designed for VLA models for robotic manipulation tasks. We assess their effectiveness through a large-scale empirical study involving 908 successful task executions from three state-of-the-art VLA models across four representative robotic manipulation tasks. Human domain experts manually labeled task quality, allowing us to analyze the correlation between our proposed metrics and expert judgments. The results reveal that several metrics show moderate to strong correlation with human assessments, highlighting their utility for evaluating task quality and model confidence. Furthermore, we found that some of the metrics can discriminate between high-, medium-, and low-quality executions from unsuccessful tasks, which can be interesting when test oracles are not available. Our findings challenge the adequacy of current evaluation practices that rely solely on binary success rates and pave the way for improved real-time monitoring and adaptive enhancement of VLA-enabled robotic systems. 

**Abstract (ZH)**: 视觉语言行动（VLA）模型是一种多模态的人工智能系统，通过整合视觉感知、自然语言理解和行动规划，使智能体能够解释其环境、理解指令并执行自主的实体任务。近年来，该领域取得了显著进步。这类模型通常通过任务成功率来评估，但无法捕捉任务执行的质量和模型决策的置信度。本文提出八项不确定性度量和五项质量度量，专门用于评估视觉语言行动模型在机器人操作任务中的表现。我们通过涵盖三个先进VLA模型在四类代表性机器人操作任务中908次成功执行任务的大型实证研究，评估其有效性。人类领域专家手工标注了任务质量，使我们能够分析我们提出的度量与专家判断之间的相关性。结果显示，多项度量与人类评估之间存在中等到较强的关联性，突显了其评估任务质量和模型置信度的实用性。此外，我们发现某些度量能够区分成功的高、中、低质量执行，这在测试判据不可用时可能非常有趣。我们的研究挑战了当前仅依赖二元成功率的评估实践，并为改进视觉语言行动增强的机器人系统的实时监控和自适应改进开辟了新途径。 

---
# Hierarchical Reinforcement Learning Framework for Adaptive Walking Control Using General Value Functions of Lower-Limb Sensor Signals 

**Title (ZH)**: 基于下肢传感器信号通用价值函数的分层级强化学习适应性行走控制框架 

**Authors**: Sonny T. Jones, Grange M. Simpson, Patrick M. Pilarski, Ashley N. Dalrymple  

**Link**: [PDF](https://arxiv.org/pdf/2507.16983)  

**Abstract**: Rehabilitation technology is a natural setting to study the shared learning and decision-making of human and machine agents. In this work, we explore the use of Hierarchical Reinforcement Learning (HRL) to develop adaptive control strategies for lower-limb exoskeletons, aiming to enhance mobility and autonomy for individuals with motor impairments. Inspired by prominent models of biological sensorimotor processing, our investigated HRL approach breaks down the complex task of exoskeleton control adaptation into a higher-level framework for terrain strategy adaptation and a lower-level framework for providing predictive information; this latter element is implemented via the continual learning of general value functions (GVFs). GVFs generated temporal abstractions of future signal values from multiple wearable lower-limb sensors, including electromyography, pressure insoles, and goniometers. We investigated two methods for incorporating actual and predicted sensor signals into a policy network with the intent to improve the decision-making capacity of the control system of a lower-limb exoskeleton during ambulation across varied terrains. As a key result, we found that the addition of predictions made from GVFs increased overall network accuracy. Terrain-specific performance increases were seen while walking on even ground, uneven ground, up and down ramps, and turns, terrains that are often misclassified without predictive information. This suggests that predictive information can aid decision-making during uncertainty, e.g., on terrains that have a high chance of being misclassified. This work, therefore, contributes new insights into the nuances of HRL and the future development of exoskeletons to facilitate safe transitioning and traversing across different walking environments. 

**Abstract (ZH)**: 康复技术是研究人类和机器智能共享学习与决策的有效环境。本文探讨了层次强化学习（HRL）在开发下肢外骨骼适应控制策略中的应用，旨在增强运动障碍个体的行动能力和自主性。受生物传感器运动处理模型启发，我们研究的HRL方法将外骨骼控制适应的复杂任务分解为一个更高层次的地形策略适应框架和一个较低层次的提供预测信息框架；后者通过持续学习一般价值函数（GVFs）实现。GVFs从包括肌电图、压力内底和关节角度传感器在内的多种可穿戴下肢传感器中生成了未来信号值的时间抽象。我们研究了两种方法，将实际和预测的传感器信号纳入策略网络中，旨在提高下肢外骨骼行走过程中复杂地形环境中控制系统决策能力。一个关键结果表明，GVFs预测的引入提高了整体网络的准确性。在平坦地面、不平地面、斜坡上上下下及转弯等地形上行走时，特定地形性能的提升表明预测信息可以在不确定性环境中帮助决策，特别是在高概率误分类的地形上。因此，本文为HRL的细微机制以及未来外骨骼的发展提供了新的见解，以促进在不同行走环境中的安全过渡和穿越。 

---
# Symbiotic Agents: A Novel Paradigm for Trustworthy AGI-driven Networks 

**Title (ZH)**: 共生代理：一种 trustworthy AGI 驱动网络的新范式 

**Authors**: Ilias Chatzistefanidis, Navid Nikaein  

**Link**: [PDF](https://arxiv.org/pdf/2507.17695)  

**Abstract**: Large Language Model (LLM)-based autonomous agents are expected to play a vital role in the evolution of 6G networks, by empowering real-time decision-making related to management and service provisioning to end-users. This shift facilitates the transition from a specialized intelligence approach, where artificial intelligence (AI) algorithms handle isolated tasks, to artificial general intelligence (AGI)-driven networks, where agents possess broader reasoning capabilities and can manage diverse network functions. In this paper, we introduce a novel agentic paradigm that combines LLMs with real-time optimization algorithms towards Trustworthy AI, defined as symbiotic agents. Optimizers at the LLM's input-level provide bounded uncertainty steering for numerically precise tasks, whereas output-level optimizers supervised by the LLM enable adaptive real-time control. We design and implement two novel agent types including: (i) Radio Access Network optimizers, and (ii) multi-agent negotiators for Service-Level Agreements (SLAs). We further propose an end-to-end architecture for AGI networks and evaluate it on a 5G testbed capturing channel fluctuations from moving vehicles. Results show that symbiotic agents reduce decision errors fivefold compared to standalone LLM-based agents, while smaller language models (SLM) achieve similar accuracy with a 99.9% reduction in GPU resource overhead and in near-real-time loops of 82 ms. A multi-agent demonstration for collaborative RAN on the real-world testbed highlights significant flexibility in service-level agreement and resource allocation, reducing RAN over-utilization by approximately 44%. Drawing on our findings and open-source implementations, we introduce the symbiotic paradigm as the foundation for next-generation, AGI-driven networks-systems designed to remain adaptable, efficient, and trustworthy even as LLMs advance. 

**Abstract (ZH)**: 基于大型语言模型的自主代理在6G网络演进中的作用：可信赖人工智能的共生代理 Paradigm 

---
# Yume: An Interactive World Generation Model 

**Title (ZH)**: 梦境：一个互动世界生成模型 

**Authors**: Xiaofeng Mao, Shaoheng Lin, Zhen Li, Chuanhao Li, Wenshuo Peng, Tong He, Jiangmiao Pang, Mingmin Chi, Yu Qiao, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17744)  

**Abstract**: Yume aims to use images, text, or videos to create an interactive, realistic, and dynamic world, which allows exploration and control using peripheral devices or neural signals. In this report, we present a preview version of \method, which creates a dynamic world from an input image and allows exploration of the world using keyboard actions. To achieve this high-fidelity and interactive video world generation, we introduce a well-designed framework, which consists of four main components, including camera motion quantization, video generation architecture, advanced sampler, and model acceleration. First, we quantize camera motions for stable training and user-friendly interaction using keyboard inputs. Then, we introduce the Masked Video Diffusion Transformer~(MVDT) with a memory module for infinite video generation in an autoregressive manner. After that, training-free Anti-Artifact Mechanism (AAM) and Time Travel Sampling based on Stochastic Differential Equations (TTS-SDE) are introduced to the sampler for better visual quality and more precise control. Moreover, we investigate model acceleration by synergistic optimization of adversarial distillation and caching mechanisms. We use the high-quality world exploration dataset \sekai to train \method, and it achieves remarkable results in diverse scenes and applications. All data, codebase, and model weights are available on this https URL. Yume will update monthly to achieve its original goal. Project page: this https URL. 

**Abstract (ZH)**: Yume旨在使用图像、文本或视频创建一个交互式、逼真且动态的世界，允许用户通过外围设备或神经信号进行探索和控制。在本报告中，我们展示了\method的预览版本，该版本可以从输入图像中生成动态世界，并允许用户通过键盘操作进行世界探索。为了实现这一高保真度和交互式的视频生成，我们提出了一种精心设计的框架，包括相机运动量化、视频生成架构、高级采样器和模型加速四个主要组成部分。首先，我们对相机运动进行量化，以实现稳定的训练和用户友好的键盘输入交互。然后，我们引入了具有记忆模块的Masked Video Diffusion Transformer (MVDT)，以进行自回归方式的无限视频生成。接着，我们引入了训练免费的抗伪影机制（AAM）和基于随机微分方程的时间旅行采样（TTS-SDE），以提高视觉质量并实现更精确的控制。此外，我们通过对抗蒸馏和缓存机制的协同优化来研究模型加速。我们使用高质量的世界探索数据集\sekai对\method进行了训练，并在多种场景和应用中取得了显著成果。所有数据、代码库和模型权重均可通过该链接下载。Yume将每月更新以实现其原始目标。项目页面：该链接。 

---
# Reality Proxy: Fluid Interactions with Real-World Objects in MR via Abstract Representations 

**Title (ZH)**: 现实代理：通过抽象表示在MR中与真实世界物体进行流体交互 

**Authors**: Xiaoan Liu, Difan Jia, Xianhao Carton Liu, Mar Gonzalez-Franco, Chen Zhu-Tian  

**Link**: [PDF](https://arxiv.org/pdf/2507.17248)  

**Abstract**: Interacting with real-world objects in Mixed Reality (MR) often proves difficult when they are crowded, distant, or partially occluded, hindering straightforward selection and manipulation. We observe that these difficulties stem from performing interaction directly on physical objects, where input is tightly coupled to their physical constraints. Our key insight is to decouple interaction from these constraints by introducing proxies-abstract representations of real-world objects. We embody this concept in Reality Proxy, a system that seamlessly shifts interaction targets from physical objects to their proxies during selection. Beyond facilitating basic selection, Reality Proxy uses AI to enrich proxies with semantic attributes and hierarchical spatial relationships of their corresponding physical objects, enabling novel and previously cumbersome interactions in MR - such as skimming, attribute-based filtering, navigating nested groups, and complex multi object selections - all without requiring new gestures or menu systems. We demonstrate Reality Proxy's versatility across diverse scenarios, including office information retrieval, large-scale spatial navigation, and multi-drone control. An expert evaluation suggests the system's utility and usability, suggesting that proxy-based abstractions offer a powerful and generalizable interaction paradigm for future MR systems. 

**Abstract (ZH)**: 在混合现实（MR）中与拥挤、遥远或部分遮挡的实际物体互动往往很困难，这阻碍了直接的选择和操作。我们观察到这些困难源于直接在物理对象上进行交互，其中输入紧密耦合于物理约束。我们的关键洞察是通过引入代理——现实世界物体的抽象表示，来解耦交互与这些约束。我们提出了Reality Proxy系统，该系统在选择过程中无缝地将交互目标从物理对象转移到其代理。超越基本的选择，Reality Proxy利用AI为代理添加对应的物理对象的语义属性和层次空间关系，这在MR中促成了一些以前难以实现的新颖交互，例如浏览、基于属性的筛选、导航嵌套组和复杂的多对象选择，而无需新的手势或菜单系统。我们展示了Reality Proxy在多种场景中的灵活性，包括办公室信息检索、大规模空间导航和多无人机控制。专家评估表明该系统的实用性和易用性，表明基于代理的抽象为未来的MR系统提供了一种强大且通用的交互范式。 

---
# LLM Meets the Sky: Heuristic Multi-Agent Reinforcement Learning for Secure Heterogeneous UAV Networks 

**Title (ZH)**: LLM 接触天空：启发式多代理强化学习在安全异构无人机网络中的应用 

**Authors**: Lijie Zheng, Ji He, Shih Yu Chang, Yulong Shen, Dusit Niyato  

**Link**: [PDF](https://arxiv.org/pdf/2507.17188)  

**Abstract**: This work tackles the physical layer security (PLS) problem of maximizing the secrecy rate in heterogeneous UAV networks (HetUAVNs) under propulsion energy constraints. Unlike prior studies that assume uniform UAV capabilities or overlook energy-security trade-offs, we consider a realistic scenario where UAVs with diverse payloads and computation resources collaborate to serve ground terminals in the presence of eavesdroppers. To manage the complex coupling between UAV motion and communication, we propose a hierarchical optimization framework. The inner layer uses a semidefinite relaxation (SDR)-based S2DC algorithm combining penalty functions and difference-of-convex (d.c.) programming to solve the secrecy precoding problem with fixed UAV positions. The outer layer introduces a Large Language Model (LLM)-guided heuristic multi-agent reinforcement learning approach (LLM-HeMARL) for trajectory optimization. LLM-HeMARL efficiently incorporates expert heuristics policy generated by the LLM, enabling UAVs to learn energy-aware, security-driven trajectories without the inference overhead of real-time LLM calls. The simulation results show that our method outperforms existing baselines in secrecy rate and energy efficiency, with consistent robustness across varying UAV swarm sizes and random seeds. 

**Abstract (ZH)**: 本研究解决了动力能量约束下异构无人机网络（HetUAVNs）物理层安全（PLS）问题，旨在最大化保密率。不同于以往研究中假设无人机能力均匀或忽视能量－安全权衡，我们考虑了一个现实场景，在该场景中，配备不同载荷和计算资源的无人机协同工作，为地面终端服务，并同时对抗窃听者。为了处理无人机运动与通信之间的复杂耦合，我们提出了一种分层优化框架。内层采用基于半定 relaxation (SDR) 的带有惩罚函数和凸二次锥（d.c.）编程的 S2DC 算法来解决固定无人机位置下的保密前向编码问题。外层引入了一种由大型语言模型（LLM）引导的启发式多代理强化学习方法（LLM-HeMARL）进行轨迹优化。LLM-HeMARL 有效地整合了由 LLM 生成的专家启发式策略，使无人机能够学习能量意识强且安全导向的轨迹，而无需进行实时 LLM 推理的开销。仿真结果表明，与现有基准方法相比，我们的方法在保密率和能效方面更具优势，并且在不同的无人机群大小和随机种子下表现出一致的鲁棒性。 

---
# Advancing Robustness in Deep Reinforcement Learning with an Ensemble Defense Approach 

**Title (ZH)**: 基于集成防御方法提升深度强化学习的 robustness 

**Authors**: Adithya Mohan, Dominik Rößle, Daniel Cremers, Torsten Schön  

**Link**: [PDF](https://arxiv.org/pdf/2507.17070)  

**Abstract**: Recent advancements in Deep Reinforcement Learning (DRL) have demonstrated its applicability across various domains, including robotics, healthcare, energy optimization, and autonomous driving. However, a critical question remains: How robust are DRL models when exposed to adversarial attacks? While existing defense mechanisms such as adversarial training and distillation enhance the resilience of DRL models, there remains a significant research gap regarding the integration of multiple defenses in autonomous driving scenarios specifically. This paper addresses this gap by proposing a novel ensemble-based defense architecture to mitigate adversarial attacks in autonomous driving. Our evaluation demonstrates that the proposed architecture significantly enhances the robustness of DRL models. Compared to the baseline under FGSM attacks, our ensemble method improves the mean reward from 5.87 to 18.38 (over 213% increase) and reduces the mean collision rate from 0.50 to 0.09 (an 82% decrease) in the highway scenario and merge scenario, outperforming all standalone defense strategies. 

**Abstract (ZH)**: Recent advancements in深度强化学习(DRL)已经证明其在机器人学、医疗保健、能源优化和自主驾驶等各个领域中的适用性。然而，一个关键问题仍然存在：当暴露于恶意攻击时，DRL模型的稳健性如何？虽然现有的防御机制如恶意训练和蒸馏可以增强DRL模型的抗攻击性，但目前在自主驾驶场景中集成多种防御机制尚存在显著的研究空白。本文通过提出一种新型集成防御架构，旨在缓解自主驾驶场景中的恶意攻击，从而填补这一空白。我们的评估结果表明，该提出的方法显著增强了DRL模型的稳健性。在高架场景和并线场景中，与基线方法相比，在FGSM攻击下，我们的集成方法将平均奖励从5.87提高到18.38（提高了213%），并将平均碰撞率从0.50降低到0.09（降低了82%），优于所有单一防御策略。 

---
# Pixels, Patterns, but No Poetry: To See The World like Humans 

**Title (ZH)**: 像素、模式，但没有诗意：以人类方式看待世界 

**Authors**: Hongcheng Gao, Zihao Huang, Lin Xu, Jingyi Tang, Xinhao Li, Yue Liu, Haoyang Li, Taihang Hu, Minhua Lin, Xinlong Yang, Ge Wu, Balong Bi, Hongyu Chen, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16863)  

**Abstract**: Achieving human-like perception and reasoning in Multimodal Large Language Models (MLLMs) remains a central challenge in artificial intelligence. While recent research has primarily focused on enhancing reasoning capabilities in MLLMs, a fundamental question persists: Can Multimodal Large Language Models truly perceive the world as humans do? This paper shifts focus from reasoning to perception. Rather than constructing benchmarks specifically for reasoning, we introduce the Turing Eye Test (TET), a challenging perception-oriented benchmark comprising four diagnostic tasks that evaluate MLLMs' performance on synthetic images that humans process intuitively. Our findings reveal that state-of-the-art MLLMs exhibit catastrophic failures on our perceptual tasks trivial for humans. Both in-context learning and training on language backbone-effective for previous benchmarks-fail to improve performance on our tasks, while fine-tuning the vision tower enables rapid adaptation, suggesting that our benchmark poses challenges for vision tower generalization rather than for the knowledge and reasoning capabilities of the language backbone-a key gap between current MLLMs and human perception. We release a representative subset of TET tasks in this version, and will introduce more diverse tasks and methods to enhance visual generalization in future work. 

**Abstract (ZH)**: 在多模态大语言模型（MLLMs）中实现类似人类的感知与推理仍然是人工智能中的一个核心挑战。尽管近期的研究主要集中在增强MLLMs的推理能力上，但一个基本问题仍然存在：多模态大语言模型真能像人类一样感知世界吗？本文将研究重点从推理转向感知。我们构建了图灵眼部测试（TET），这是一个具有四个诊断任务的具有挑战性的感知导向基准，用于评估MLLMs在人类直观处理的合成图像上的表现。我们的发现表明，最先进的MLLMs在人类认为简单的感知任务上表现出灾难性的失败。上下文学习和语言骨干训练——这在过去的一些基准测试中有效——未能提高我们在任务上的表现，而微调视觉塔则能够迅速适应，这表明我们的基准测试对视觉塔的泛化提出了挑战，而不是对语言骨干的知识和推理能力——这是当前MLLMs与人类感知之间的关键差距。我们在此版本中释放了TET任务的一个代表性子集，并将在未来的工作中引入更多样化的任务和方法以增强视觉泛化能力。 

---
