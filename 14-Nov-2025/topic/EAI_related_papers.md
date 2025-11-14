# Robot Crash Course: Learning Soft and Stylized Falling 

**Title (ZH)**: 机器人入门教程：学习柔和和风格化的跌落 

**Authors**: Pascal Strauch, David Müller, Sammy Christen, Agon Serifi, Ruben Grandia, Espen Knoop, Moritz Bächer  

**Link**: [PDF](https://arxiv.org/pdf/2511.10635)  

**Abstract**: Despite recent advances in robust locomotion, bipedal robots operating in the real world remain at risk of falling. While most research focuses on preventing such events, we instead concentrate on the phenomenon of falling itself. Specifically, we aim to reduce physical damage to the robot while providing users with control over a robot's end pose. To this end, we propose a robot agnostic reward function that balances the achievement of a desired end pose with impact minimization and the protection of critical robot parts during reinforcement learning. To make the policy robust to a broad range of initial falling conditions and to enable the specification of an arbitrary and unseen end pose at inference time, we introduce a simulation-based sampling strategy of initial and end poses. Through simulated and real-world experiments, our work demonstrates that even bipedal robots can perform controlled, soft falls. 

**Abstract (ZH)**: 尽管在稳健运动方面取得了近期进展，但在现实世界中运行的双足机器人仍然面临摔倒的风险。大多数研究集中在预防此类事件，而我们则将重点放在摔倒现象本身。具体而言，我们旨在在强化学习过程中平衡期望结束姿态的实现、冲击最小化以及保护关键机器人部件，从而减少对机器人的物理损伤并提供用户对机器人结束姿态的控制。为此，我们提出了一种机器人无关的奖励函数，该函数在期望结束姿态的实现与冲击最小化以及保护关键机器人部件之间取得平衡。为了使策略能够应对广泛的初始摔倒条件，并允许在推理时指定任意和未见过的结束姿态，我们引入了一种基于仿真的初始姿态和结束姿态采样策略。通过模拟和真实世界实验，我们的工作证明了即使双足机器人也能执行可控的、软着陆的摔倒。 

---
# Learning a Thousand Tasks in a Day 

**Title (ZH)**: 每天学习一千种任务 

**Authors**: Kamil Dreczkowski, Pietro Vitiello, Vitalis Vosylius, Edward Johns  

**Link**: [PDF](https://arxiv.org/pdf/2511.10110)  

**Abstract**: Humans are remarkably efficient at learning tasks from demonstrations, but today's imitation learning methods for robot manipulation often require hundreds or thousands of demonstrations per task. We investigate two fundamental priors for improving learning efficiency: decomposing manipulation trajectories into sequential alignment and interaction phases, and retrieval-based generalisation. Through 3,450 real-world rollouts, we systematically study this decomposition. We compare different design choices for the alignment and interaction phases, and examine generalisation and scaling trends relative to today's dominant paradigm of behavioural cloning with a single-phase monolithic policy. In the few-demonstrations-per-task regime (<10 demonstrations), decomposition achieves an order of magnitude improvement in data efficiency over single-phase learning, with retrieval consistently outperforming behavioural cloning for both alignment and interaction. Building on these insights, we develop Multi-Task Trajectory Transfer (MT3), an imitation learning method based on decomposition and retrieval. MT3 learns everyday manipulation tasks from as little as a single demonstration each, whilst also generalising to novel object instances. This efficiency enables us to teach a robot 1,000 distinct everyday tasks in under 24 hours of human demonstrator time. Through 2,200 additional real-world rollouts, we reveal MT3's capabilities and limitations across different task families. Videos of our experiments can be found on at this https URL. 

**Abstract (ZH)**: 人类在从演示中学任务方面表现出惊人的效率，但当前用于机器人操作的模仿学习方法往往需要每项任务成百上千次的演示。我们研究了提高学习效率的两个基本先验知识：将操作轨迹分解为序列对齐和交互阶段，并采用基于检索的泛化。通过3,450个真实世界的试验，我们系统地研究了这种分解。我们将不同对齐和交互阶段设计选择进行了比较，并考察了与当前主导的行为克隆单一阶段整体策略相比的泛化和扩展趋势。在每项任务使用少量演示(<10次演示)的范围内，分解在数据效率上实现了数量级的提升，检索在对齐和交互阶段均持续优于行为克隆。基于这些见解，我们开发了基于分解和检索的多任务轨迹转移（MT3）模仿学习方法。MT3可以从每项任务一个演示开始学习日常操作任务，并泛化到新的对象实例。这种效率使我们能够在不到24小时的人类演示时间内教会机器人1,000个不同的日常任务。通过2,200个额外的真实世界试验，我们揭示了MT3在不同任务家族中的能力和局限性。我们的实验视频可在以下链接找到：这个 https URL。 

---
# Opinion: Towards Unified Expressive Policy Optimization for Robust Robot Learning 

**Title (ZH)**: 意见：面向鲁棒机器人学习的统一表达性策略优化途径 

**Authors**: Haidong Huang, Haiyue Zhu. Jiayu Song, Xixin Zhao, Yaohua Zhou, Jiayi Zhang, Yuze Zhai, Xiaocong Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.10087)  

**Abstract**: Offline-to-online reinforcement learning (O2O-RL) has emerged as a promising paradigm for safe and efficient robotic policy deployment but suffers from two fundamental challenges: limited coverage of multimodal behaviors and distributional shifts during online adaptation. We propose UEPO, a unified generative framework inspired by large language model pretraining and fine-tuning strategies. Our contributions are threefold: (1) a multi-seed dynamics-aware diffusion policy that efficiently captures diverse modalities without training multiple models; (2) a dynamic divergence regularization mechanism that enforces physically meaningful policy diversity; and (3) a diffusion-based data augmentation module that enhances dynamics model generalization. On the D4RL benchmark, UEPO achieves +5.9\% absolute improvement over Uni-O4 on locomotion tasks and +12.4\% on dexterous manipulation, demonstrating strong generalization and scalability. 

**Abstract (ZH)**: 离线到在线强化学习（O2O-RL）作为一种有前途的机器人策略部署范式已出现，但面临两个根本挑战：多模态行为覆盖有限和在线适应过程中的分布偏移。我们提出了UEPO，这是一种受大规模语言模型预训练和微调策略启发的统一生成框架。我们的贡献包括：（1）一种多种子动力学感知扩散策略，能在不训练多个模型的情况下高效捕捉多种模态；（2）一种动态偏差正则化机制，强制执行物理上合理的策略多样性；以及（3）一种基于扩散的数据增强模块，提高动力学模型的一般化能力。在D4RL基准测试中，UEPO在移动任务上实现了相对于Uni-O4的5.9%的绝对改进，在灵巧操作任务上实现了12.4%的改进，展示了强大的一般化能力和可扩展性。 

---
# Physics-informed Machine Learning for Static Friction Modeling in Robotic Manipulators Based on Kolmogorov-Arnold Networks 

**Title (ZH)**: 基于柯尔莫哥洛夫-阿诺德网络的机器人 manipulator 静摩擦建模的物理约束机器学习方法 

**Authors**: Yizheng Wang, Timon Rabczuk, Yinghua Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.10079)  

**Abstract**: Friction modeling plays a crucial role in achieving high-precision motion control in robotic operating systems. Traditional static friction models (such as the Stribeck model) are widely used due to their simple forms; however, they typically require predefined functional assumptions, which poses significant challenges when dealing with unknown functional structures. To address this issue, this paper proposes a physics-inspired machine learning approach based on the Kolmogorov Arnold Network (KAN) for static friction modeling of robotic joints. The method integrates spline activation functions with a symbolic regression mechanism, enabling model simplification and physical expression extraction through pruning and attribute scoring, while maintaining both high prediction accuracy and interpretability. We first validate the method's capability to accurately identify key parameters under known functional models, and further demonstrate its robustness and generalization ability under conditions with unknown functional structures and noisy data. Experiments conducted on both synthetic data and real friction data collected from a six-degree-of-freedom industrial manipulator show that the proposed method achieves a coefficient of determination greater than 0.95 across various tasks and successfully extracts concise and physically meaningful friction expressions. This study provides a new perspective for interpretable and data-driven robotic friction modeling with promising engineering applicability. 

**Abstract (ZH)**: 摩擦建模在机器人操作系统实现高精度运动控制中起着至关重要的作用。传统的静态摩擦模型（如斯特布希模型）因其简单的形式而被广泛使用，但通常需要预先定义的功能假设，这在处理未知功能结构时构成了重大挑战。为了解决这一问题，本文提出了一种基于Kolmogorov Arnold Network (KAN)的物理启发式机器学习方法，用于机器人关节的静态摩擦建模。该方法结合了样条激活函数和符号回归机制，通过修剪和属性评分实现模型简化和物理表达提取，同时保持高预测精度和可解释性。我们首先验证了该方法在已知功能模型下准确识别关键参数的能力，并进一步展示了其在未知功能结构和噪声数据条件下的鲁棒性和泛化能力。在六自由度工业 manipulator 上合成数据和实际摩擦数据的实验表明，所提出的方法在各种任务中的决定系数均大于0.95，并成功提取了简洁且物理上有意义的摩擦表达式。本研究为可解释和数据驱动的机器人摩擦建模提供了新的视角，并具有潜在的工程应用价值。 

---
# DecARt Leg: Design and Evaluation of a Novel Humanoid Robot Leg with Decoupled Actuation for Agile Locomotion 

**Title (ZH)**: DecARt Leg：一种解耦驱动的人形机器人腿的设计与评估 

**Authors**: Egor Davydenko, Andrei Volchenkov, Vladimir Gerasimov, Roman Gorbachev  

**Link**: [PDF](https://arxiv.org/pdf/2511.10021)  

**Abstract**: In this paper, we propose a novel design of an electrically actuated robotic leg, called the DecARt (Decoupled Actuation Robot) Leg, aimed at performing agile locomotion. This design incorporates several new features, such as the use of a quasi-telescopic kinematic structure with rotational motors for decoupled actuation, a near-anthropomorphic leg appearance with a forward facing knee, and a novel multi-bar system for ankle torque transmission from motors placed above the knee. To analyze the agile locomotion capabilities of the design numerically, we propose a new descriptive metric, called the `Fastest Achievable Swing Time` (FAST), and perform a quantitative evaluation of the proposed design and compare it with other designs. Then we evaluate the performance of the DecARt Leg-based robot via extensive simulation and preliminary hardware experiments. 

**Abstract (ZH)**: 一种 decoupled actuation 电驱动机器人腿部设计：DecARt (Decoupled Actuation Robot) 腿，及其敏捷运动能力的分析与评估 

---
# Phantom Menace: Exploring and Enhancing the Robustness of VLA Models against Physical Sensor Attacks 

**Title (ZH)**: 虚假威胁：探索并增强VLA模型对抗物理传感器攻击的鲁棒性 

**Authors**: Xuancun Lu, Jiaxiang Chen, Shilin Xiao, Zizhi Jin, Zhangrui Chen, Hanwen Yu, Bohan Qian, Ruochen Zhou, Xiaoyu Ji, Wenyuan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.10008)  

**Abstract**: Vision-Language-Action (VLA) models revolutionize robotic systems by enabling end-to-end perception-to-action pipelines that integrate multiple sensory modalities, such as visual signals processed by cameras and auditory signals captured by microphones. This multi-modality integration allows VLA models to interpret complex, real-world environments using diverse sensor data streams. Given the fact that VLA-based systems heavily rely on the sensory input, the security of VLA models against physical-world sensor attacks remains critically underexplored.
To address this gap, we present the first systematic study of physical sensor attacks against VLAs, quantifying the influence of sensor attacks and investigating the defenses for VLA models. We introduce a novel ``Real-Sim-Real'' framework that automatically simulates physics-based sensor attack vectors, including six attacks targeting cameras and two targeting microphones, and validates them on real robotic systems. Through large-scale evaluations across various VLA architectures and tasks under varying attack parameters, we demonstrate significant vulnerabilities, with susceptibility patterns that reveal critical dependencies on task types and model designs. We further develop an adversarial-training-based defense that enhances VLA robustness against out-of-distribution physical perturbations caused by sensor attacks while preserving model performance. Our findings expose an urgent need for standardized robustness benchmarks and mitigation strategies to secure VLA deployments in safety-critical environments. 

**Abstract (ZH)**: 基于视觉-语言-动作（VLA）模型的物理传感器攻击研究 

---
# A Study on Enhancing the Generalization Ability of Visuomotor Policies via Data Augmentation 

**Title (ZH)**: 基于数据增强提升视动策略的泛化能力的研究 

**Authors**: Hanwen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.09932)  

**Abstract**: The generalization ability of visuomotor policy is crucial, as a good policy should be deployable across diverse scenarios. Some methods can collect large amounts of trajectory augmentation data to train more generalizable imitation learning policies, aimed at handling the random placement of objects on the scene's horizontal plane. However, the data generated by these methods still lack diversity, which limits the generalization ability of the trained policy. To address this, we investigate the performance of policies trained by existing methods across different scene layout factors via automate the data generation for those factors that significantly impact generalization. We have created a more extensively randomized dataset that can be efficiently and automatically generated with only a small amount of human demonstration. The dataset covers five types of manipulators and two types of grippers, incorporating extensive randomization factors such as camera pose, lighting conditions, tabletop texture, and table height across six manipulation tasks. We found that all of these factors influence the generalization ability of the policy. Applying any form of randomization enhances policy generalization, with diverse trajectories particularly effective in bridging visual gap. Notably, we investigated on low-cost manipulator the effect of the scene randomization proposed in this work on enhancing the generalization capability of visuomotor policies for zero-shot sim-to-real transfer. 

**Abstract (ZH)**: 视觉运动策略的泛化能力至关重要，因为一个好的策略应该可以在多种场景下部署。现有的方法可以通过收集大量的轨迹增广数据来训练更具泛化能力的模仿学习策略，以应对场景水平面上随机放置的对象。然而，这些方法生成的数据仍然缺乏多样性，这限制了训练策略的泛化能力。为了解决这一问题，我们通过自动化那些对泛化影响显著的场景布局因素的数据生成过程，研究现有方法训练策略在不同场景布局因素下的性能。我们创建了一个更为广泛随机化的数据集，该数据集可以仅通过少量的人类示范高效且自动地生成。数据集涵盖了五种类型的操纵器和两种类型的夹爪，并包含了广泛随机化因素，如摄像头姿态、光照条件、桌面纹理和桌子高度，这些因素贯穿六个操作任务。我们发现这些因素都会影响策略的泛化能力。任何形式的随机化都可以增强策略的泛化能力，尤其是多样化的轨迹特别有效于填补视觉差距。特别地，我们研究了在低成本操纵器上，本文提出的场景随机化对提高视觉运动策略零样本仿真实际转移能力的影响。 

---
# Baby Sophia: A Developmental Approach to Self-Exploration through Self-Touch and Hand Regard 

**Title (ZH)**: 婴儿索菲亚：一种通过自我触碰和手的关注进行自我探索的发展性方法 

**Authors**: Stelios Zarifis, Ioannis Chalkiadakis, Artemis Chardouveli, Vasiliki Moutzouri, Aggelos Sotirchos, Katerina Papadimitriou, Panagiotis Filntisis, Niki Efthymiou, Petros Maragos, Katerina Pastra  

**Link**: [PDF](https://arxiv.org/pdf/2511.09727)  

**Abstract**: Inspired by infant development, we propose a Reinforcement Learning (RL) framework for autonomous self-exploration in a robotic agent, Baby Sophia, using the BabyBench simulation environment. The agent learns self-touch and hand regard behaviors through intrinsic rewards that mimic an infant's curiosity-driven exploration of its own body. For self-touch, high-dimensional tactile inputs are transformed into compact, meaningful representations, enabling efficient learning. The agent then discovers new tactile contacts through intrinsic rewards and curriculum learning that encourage broad body coverage, balance, and generalization. For hand regard, visual features of the hands, such as skin-color and shape, are learned through motor babbling. Then, intrinsic rewards encourage the agent to perform novel hand motions, and follow its hands with its gaze. A curriculum learning setup from single-hand to dual-hand training allows the agent to reach complex visual-motor coordination. The results of this work demonstrate that purely curiosity-based signals, with no external supervision, can drive coordinated multimodal learning, imitating an infant's progression from random motor babbling to purposeful behaviors. 

**Abstract (ZH)**: 受婴儿发展启发，我们提出了一种基于强化学习（RL）的自主自我探索框架，用于机器人代理Baby Sophia，采用BabyBench模拟环境。代理通过模仿婴儿对自身身体的好奇探索，学会自我触碰和手部关注行为。在自我触碰方面，高维度的触觉输入被转换为紧凑且有意义的表示，从而实现高效的自我学习。代理通过内在奖励和促进全身覆盖、平衡与泛化的课程学习，发现新的触觉接触点。在手部关注方面，通过运动咿呀学语，代理学习手部的视觉特征，如肤色和形状。接着，内在奖励促使代理执行新颖的手部动作，并用目光跟随手的动作。从单手训练到双手训练的课程学习设置，使代理能够实现复杂的视听运动协调。本研究结果表明，仅基于好奇心的信号，无需外部监督，即可驱动协调的多模态学习，模仿婴儿从随意的运动咿呀学语到有目的的行为的进展。 

---
# SemanticVLA: Semantic-Aligned Sparsification and Enhancement for Efficient Robotic Manipulation 

**Title (ZH)**: 语义VLA：面向高效机器人操纵的语义对齐稀疏化与增强 

**Authors**: Wei Li, Renshan Zhang, Rui Shao, Zhijian Fang, Kaiwen Zhou, Zhuotao Tian, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2511.10518)  

**Abstract**: Vision-Language-Action (VLA) models have advanced in robotic manipulation, yet practical deployment remains hindered by two key limitations: 1) perceptual redundancy, where irrelevant visual inputs are processed inefficiently, and 2) superficial instruction-vision alignment, which hampers semantic grounding of actions. In this paper, we propose SemanticVLA, a novel VLA framework that performs Semantic-Aligned Sparsification and Enhancement for Efficient Robotic Manipulation. Specifically: 1) To sparsify redundant perception while preserving semantic alignment, Semantic-guided Dual Visual Pruner (SD-Pruner) performs: Instruction-driven Pruner (ID-Pruner) extracts global action cues and local semantic anchors in SigLIP; Spatial-aggregation Pruner (SA-Pruner) compacts geometry-rich features into task-adaptive tokens in DINOv2. 2) To exploit sparsified features and integrate semantics with spatial geometry, Semantic-complementary Hierarchical Fuser (SH-Fuser) fuses dense patches and sparse tokens across SigLIP and DINOv2 for coherent representation. 3) To enhance the transformation from perception to action, Semantic-conditioned Action Coupler (SA-Coupler) replaces the conventional observation-to-DoF approach, yielding more efficient and interpretable behavior modeling for manipulation tasks. Extensive experiments on simulation and real-world tasks show that SemanticVLA sets a new SOTA in both performance and efficiency. SemanticVLA surpasses OpenVLA on LIBERO benchmark by 21.1% in success rate, while reducing training cost and inference latency by 3.0-fold and this http URL is open-sourced and publicly available at this https URL 

**Abstract (ZH)**: 基于语义对齐的稀疏化与增强的视觉-语言-行动框架（Semantic-Aligned Sparsification and Enhancement for Efficient Robotic Manipulation） 

---
# MSGNav: Unleashing the Power of Multi-modal 3D Scene Graph for Zero-Shot Embodied Navigation 

**Title (ZH)**: MSGNav: 激发多模态3D场景图的零样本嵌入式导航潜力 

**Authors**: Xun Huang, Shijia Zhao, Yunxiang Wang, Xin Lu, Wanfa Zhang, Rongsheng Qu, Weixin Li, Yunhong Wang, Chenglu Wen  

**Link**: [PDF](https://arxiv.org/pdf/2511.10376)  

**Abstract**: Embodied navigation is a fundamental capability for robotic agents operating. Real-world deployment requires open vocabulary generalization and low training overhead, motivating zero-shot methods rather than task-specific RL training. However, existing zero-shot methods that build explicit 3D scene graphs often compress rich visual observations into text-only relations, leading to high construction cost, irreversible loss of visual evidence, and constrained vocabularies. To address these limitations, we introduce the Multi-modal 3D Scene Graph (M3DSG), which preserves visual cues by replacing textual relational edges with dynamically assigned images. Built on M3DSG, we propose MSGNav, a zero-shot navigation system that includes a Key Subgraph Selection module for efficient reasoning, an Adaptive Vocabulary Update module for open vocabulary support, and a Closed-Loop Reasoning module for accurate exploration reasoning. Additionally, we further identify the last-mile problem in zero-shot navigation - determining the feasible target location with a suitable final viewpoint, and propose a Visibility-based Viewpoint Decision module to explicitly resolve it. Comprehensive experimental results demonstrate that MSGNav achieves state-of-the-art performance on GOAT-Bench and HM3D-OVON datasets. The open-source code will be publicly available. 

**Abstract (ZH)**: 具身导航是机器人代理的基本能力。真实世界部署需要开放词汇的一般化和低训练开销，推动了零样本方法而非任务特定的强化学习训练。然而，现有的构建显式3D场景图的零样本方法往往会将丰富的视觉观察压缩为仅文本关系，导致高昂的构建成本、不可逆转的视觉证据丢失和受限的词汇量。为解决这些局限性，我们引入了多模态3D场景图（M3DSG），通过使用动态分配的图像替换文本关系边来保留视觉线索。基于M3DSG，我们提出了一种零样本导航系统MSGNav，该系统包括一个关键子图选择模块进行高效推理、一个自适应词汇更新模块支持开放词汇，并且包括一个闭环推理模块进行精确的探索推理。此外，我们进一步识别了零样本导航中的最后一步问题——确定一个合适的最终视点下的可行目标位置，并提出了一种基于视图可见性的视点决策模块明确解决这一问题。全面的实验结果表明，MSGNav在GOAT-Bench和HM3D-OVON数据集上达到了最先进的性能。开源代码将公开。 

---
# Balancing Centralized Learning and Distributed Self-Organization: A Hybrid Model for Embodied Morphogenesis 

**Title (ZH)**: 平衡集中学习与分布式自我组织：一种身体现生形态发生混合模型 

**Authors**: Takehiro Ishikawa  

**Link**: [PDF](https://arxiv.org/pdf/2511.10101)  

**Abstract**: We investigate how to couple a learnable brain-like'' controller to a cell-like'' Gray--Scott substrate to steer pattern formation with minimal effort. A compact convolutional policy is embedded in a differentiable PyTorch reaction--diffusion simulator, producing spatially smooth, bounded modulations of the feed and kill parameters ($\Delta F$, $\Delta K$) under a warm--hold--decay gain schedule. Training optimizes Turing-band spectral targets (FFT-based) while penalizing control effort ($\ell_1/\ell_2$) and instability. We compare three regimes: pure reaction--diffusion, NN-dominant, and a hybrid coupling. The hybrid achieves reliable, fast formation of target textures: 100% strict convergence in $\sim 165$ steps, matching cell-only spectral selectivity (0.436 vs.\ 0.434) while using $\sim 15\times$ less $\ell_1$ effort and $>200\times$ less $\ell_2$ power than NN-dominant control. An amplitude sweep reveals a non-monotonic Goldilocks'' zone ($A \approx 0.03$--$0.045$) that yields 100\% quasi convergence in 94--96 steps, whereas weaker or stronger gains fail to converge or degrade selectivity. These results quantify morphological computation: the controller seeds then cedes,'' providing brief, sparse nudges that place the system in the correct basin of attraction, after which local physics maintains the pattern. The study offers a practical recipe for building steerable, robust, and energy-efficient embodied systems that exploit an optimal division of labor between centralized learning and distributed self-organization. 

**Abstract (ZH)**: 探究如何将可学习的大脑样控制器与细胞样的Gray--Scott基质耦合以最少的努力引导模式形成 

---
# SlideBot: A Multi-Agent Framework for Generating Informative, Reliable, Multi-Modal Presentations 

**Title (ZH)**: SlideBot: 一种生成 informative、可靠且多模态演示文稿的多代理框架 

**Authors**: Eric Xie, Danielle Waterfield, Michael Kennedy, Aidong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.09804)  

**Abstract**: Large Language Models (LLMs) have shown immense potential in education, automating tasks like quiz generation and content summarization. However, generating effective presentation slides introduces unique challenges due to the complexity of multimodal content creation and the need for precise, domain-specific information. Existing LLM-based solutions often fail to produce reliable and informative outputs, limiting their educational value. To address these limitations, we introduce SlideBot - a modular, multi-agent slide generation framework that integrates LLMs with retrieval, structured planning, and code generation. SlideBot is organized around three pillars: informativeness, ensuring deep and contextually grounded content; reliability, achieved by incorporating external sources through retrieval; and practicality, which enables customization and iterative feedback through instructor collaboration. It incorporates evidence-based instructional design principles from Cognitive Load Theory (CLT) and the Cognitive Theory of Multimedia Learning (CTML), using structured planning to manage intrinsic load and consistent visual macros to reduce extraneous load and enhance dual-channel learning. Within the system, specialized agents collaboratively retrieve information, summarize content, generate figures, and format slides using LaTeX, aligning outputs with instructor preferences through interactive refinement. Evaluations from domain experts and students in AI and biomedical education show that SlideBot consistently enhances conceptual accuracy, clarity, and instructional value. These findings demonstrate SlideBot's potential to streamline slide preparation while ensuring accuracy, relevance, and adaptability in higher education. 

**Abstract (ZH)**: 大型语言模型（LLMs）在教育领域展示了巨大的潜力，自动化了如 Quiz 生成和内容总结等任务。然而，生成有效的演示文稿引入了独特的挑战，因为多模态内容创作的复杂性和对精确、领域特定信息的需求。现有的基于LLM的解决方案往往无法生成可靠且富有信息量的输出，限制了它们的教育价值。为了克服这些局限性，我们引入了SlideBot——一个模块化的多智能体演示文稿生成框架，将大型语言模型与检索、结构化规划和代码生成集成在一起。SlideBot围绕三个核心支柱组织：信息量，通过确保深入且情境相关的内容来实现；可靠性，通过检索外部资源来实现；以及实用性，通过教师合作实现定制化和迭代反馈。它结合了来自认知负载理论（CLT）和多媒体学习的认知理论（CTML）的基于证据的教学设计原则，使用结构化规划来管理内在负载，并通过一致的视觉宏减少辅助负载，从而增强双重通道学习。系统内的专业智能体协作检索信息、总结内容、生成图表，并使用LaTeX格式化幻灯片，通过互动细化与教师的偏好保持一致。来自人工智能和生物医学教育领域的领域专家和学生的评估表明，SlideBot在概念准确性、清晰度和教学价值方面始终表现出色。这些发现表明，SlideBot有能力简化幻灯片准备过程，同时确保高教中的准确、相关性和适应性。 

---
# Towards Emotionally Intelligent and Responsible Reinforcement Learning 

**Title (ZH)**: 面向情感智能和负责任的强化学习 

**Authors**: Garapati Keerthana, Manik Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2511.10573)  

**Abstract**: Personalized decision systems in healthcare and behavioral support often rely on static rule-based or engagement-maximizing heuristics that overlook users' emotional context and ethical constraints. Such approaches risk recommending insensitive or unsafe interventions, especially in domains involving serious mental illness, substance use disorders, or depression. To address this limitation, we propose a Responsible Reinforcement Learning (RRL) framework that integrates emotional and contextual understanding with ethical considerations into the sequential decision-making process. RRL formulates personalization as a Constrained Markov Decision Process (CMDP), where the agent optimizes engagement and adherence while ensuring emotional alignment and ethical safety. We introduce a multi-objective reward function that explicitly balances short-term behavioral engagement with long-term user well-being, and define an emotion-informed state representation that captures fluctuations in emotional readiness, affect, and risk. The proposed architecture can be instantiated with any RL algorithm (e.g., DQN, PPO) augmented with safety constraints or Lagrangian regularization. Conceptually, this framework operationalizes empathy and responsibility within machine learning policy optimization, bridging safe RL, affective computing and responsible AI. We discuss the implications of this approach for human-centric domains such as behavioral health, education, and digital therapeutics, and outline simulation-based validation paths for future empirical work. This paper aims to initiate a methodological conversation about ethically aligned reinforcement learning for emotionally aware and trustworthy personalization systems. 

**Abstract (ZH)**: 责任强化学习在医疗和行为支持中的个性化决策系统中整合情感和情境理解及伦理考虑的框架 

---
# AgentEvolver: Towards Efficient Self-Evolving Agent System 

**Title (ZH)**: AgentEvolver: 向Towards Efficient Self-Evolving Agent System方向的高效自演化智能体系统 

**Authors**: Yunpeng Zhai, Shuchang Tao, Cheng Chen, Anni Zou, Ziqian Chen, Qingxu Fu, Shinji Mai, Li Yu, Jiaji Deng, Zouying Cao, Zhaoyang Liu, Bolin Ding, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.10395)  

**Abstract**: Autonomous agents powered by large language models (LLMs) have the potential to significantly enhance human productivity by reasoning, using tools, and executing complex tasks in diverse environments. However, current approaches to developing such agents remain costly and inefficient, as they typically require manually constructed task datasets and reinforcement learning (RL) pipelines with extensive random exploration. These limitations lead to prohibitively high data-construction costs, low exploration efficiency, and poor sample utilization. To address these challenges, we present AgentEvolver, a self-evolving agent system that leverages the semantic understanding and reasoning capabilities of LLMs to drive autonomous agent learning. AgentEvolver introduces three synergistic mechanisms: (i) self-questioning, which enables curiosity-driven task generation in novel environments, reducing dependence on handcrafted datasets; (ii) self-navigating, which improves exploration efficiency through experience reuse and hybrid policy guidance; and (iii) self-attributing, which enhances sample efficiency by assigning differentiated rewards to trajectory states and actions based on their contribution. By integrating these mechanisms into a unified framework, AgentEvolver enables scalable, cost-effective, and continual improvement of agent capabilities. Preliminary experiments indicate that AgentEvolver achieves more efficient exploration, better sample utilization, and faster adaptation compared to traditional RL-based baselines. 

**Abstract (ZH)**: 由大规模语言模型驱动的自主代理有能力显著提升人类生产力，通过在多样化环境中进行推理、使用工具和执行复杂任务。然而，目前开发此类代理的方法仍然成本高且效率低，通常需要手动构建任务数据集和包含广泛随机探索的强化学习管道。这些限制导致数据构造成本高昂、探索效率低下和样本利用不佳。为应对这些挑战，我们提出了AgentEvolver自主进化代理系统，该系统利用大规模语言模型的语义理解和推理能力来推动自主代理学习。AgentEvolver引入了三种协同机制：（i）自我提问，使代理能够在新颖环境中通过好奇心驱动的任务生成减少对手工设计数据集的依赖；（ii）自我导航，通过经验重用和混合策略指导提高探索效率；（iii）自我归因，通过根据轨迹状态和动作的贡献赋予不同的奖励来提高样本效率。通过将这些机制整合到统一框架中，AgentEvolver实现了代理能力的可扩展、低成本和持续改进。初步实验表明，与传统的基于强化学习的基线方法相比，AgentEvolver实现了更有效的探索、更好的样本利用和更快的适应。 

---
# Heuristic Transformer: Belief Augmented In-Context Reinforcement Learning 

**Title (ZH)**: 启发式变换器：基于信念的上下文内强化学习 

**Authors**: Oliver Dippel, Alexei Lisitsa, Bei Peng  

**Link**: [PDF](https://arxiv.org/pdf/2511.10251)  

**Abstract**: Transformers have demonstrated exceptional in-context learning (ICL) capabilities, enabling applications across natural language processing, computer vision, and sequential decision-making. In reinforcement learning, ICL reframes learning as a supervised problem, facilitating task adaptation without parameter updates. Building on prior work leveraging transformers for sequential decision-making, we propose Heuristic Transformer (HT), an in-context reinforcement learning (ICRL) approach that augments the in-context dataset with a belief distribution over rewards to achieve better decision-making. Using a variational auto-encoder (VAE), a low-dimensional stochastic variable is learned to represent the posterior distribution over rewards, which is incorporated alongside an in-context dataset and query states as prompt to the transformer policy. We assess the performance of HT across the Darkroom, Miniworld, and MuJoCo environments, showing that it consistently surpasses comparable baselines in terms of both effectiveness and generalization. Our method presents a promising direction to bridge the gap between belief-based augmentations and transformer-based decision-making. 

**Abstract (ZH)**: transformers在上下文学习中的强化学习应用：Heuristic Transformer及其在Darkroom、Miniworld和MuJoCo环境中的表现 

---
# Improved Offline Reinforcement Learning via Quantum Metric Encoding 

**Title (ZH)**: 基于量子度量编码的改进离线强化学习 

**Authors**: Outongyi Lv, Yewei Yuan, Nana Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.10187)  

**Abstract**: Reinforcement learning (RL) with limited samples is common in real-world applications. However, offline RL performance under this constraint is often suboptimal. We consider an alternative approach to dealing with limited samples by introducing the Quantum Metric Encoder (QME). In this methodology, instead of applying the RL framework directly on the original states and rewards, we embed the states into a more compact and meaningful representation, where the structure of the encoding is inspired by quantum circuits. For classical data, QME is a classically simulable, trainable unitary embedding and thus serves as a quantum-inspired module, on a classical device. For quantum data in the form of quantum states, QME can be implemented directly on quantum hardware, allowing for training without measurement or re-encoding.
We evaluated QME on three datasets, each limited to 100 samples. We use Soft-Actor-Critic (SAC) and Implicit-Q-Learning (IQL), two well-known RL algorithms, to demonstrate the effectiveness of our approach. From the experimental results, we find that training offline RL agents on QME-embedded states with decoded rewards yields significantly better performance than training on the original states and rewards. On average across the three datasets, for maximum reward performance, we achieve a 116.2% improvement for SAC and 117.6% for IQL.
We further investigate the $\Delta$-hyperbolicity of our framework, a geometric property of the state space known to be important for the RL training efficacy. The QME-embedded states exhibit low $\Delta$-hyperbolicity, suggesting that the improvement after embedding arises from the modified geometry of the state space induced by QME. Thus, the low $\Delta$-hyperbolicity and the corresponding effectiveness of QME could provide valuable information for developing efficient offline RL methods under limited-sample conditions. 

**Abstract (ZH)**: 有限样本条件下基于量子度量编码的强化学习方法 

---
# EnvTrace: Simulation-Based Semantic Evaluation of LLM Code via Execution Trace Alignment -- Demonstrated at Synchrotron Beamlines 

**Title (ZH)**: EnvTrace：基于执行跟踪对大规模语言模型代码进行语义评估的仿真方法——以同步辐射束线为例 

**Authors**: Noah van der Vleuten, Anthony Flores, Shray Mathur, Max Rakitin, Thomas Hopkins, Kevin G. Yager, Esther H. R. Tsai  

**Link**: [PDF](https://arxiv.org/pdf/2511.09964)  

**Abstract**: Evaluating large language models (LLMs) for instrument control requires methods that go beyond standard, stateless algorithmic benchmarks, since the behavior of physical systems cannot be fully captured by unit tests alone. Here we introduce EnvTrace, a simulation-based method that evaluates execution traces to assess semantic code equivalence. EnvTrace is demonstrated with a beamline control-logic digital twin to facilitate the evaluation of instrument control code, with the digital twin itself also enabling the pre-execution validation of live experiments. Over 30 LLMs were evaluated using trace alignment to generate a multi-faceted score for functional correctness across key behavioral dimensions, showing that many top-tier models can approach human-level performance in rapid control-code generation. This is a first step toward a broader vision where LLMs and digital twins work symbiotically: LLMs providing intuitive control and agentic orchestration, and digital twins offering safe and high-fidelity environments, paving the way towards autonomous embodied AI. 

**Abstract (ZH)**: 评估大型语言模型（LLMs）在仪器控制中的性能要求方法超越传统的无状态算法基准测试，因为单一的单元测试无法全面捕捉物理系统的动态行为。我们介绍了EnvTrace，一种基于模拟的方法，通过评估执行轨迹来评估语义代码等价性。EnvTrace 使用一个束线控制逻辑数字孪生进行演示，以促进仪器控制代码的评估，并且数字孪生本身还能够对实时实验进行预执行验证。超过30个LLMs通过轨迹对齐进行了评估，生成了针对关键行为维度的功能正确性的多维度评分，结果显示许多顶级模型在快速控制代码生成方面可以达到或接近人类水平的表现。这是朝着更广泛愿景迈出的第一步，即LLMs和数字孪生能够共生：LLMs 提供直观控制和自主编排，而数字孪生提供安全和高保真环境，为自主具身AI铺平道路。 

---
# Harnessing Bounded-Support Evolution Strategies for Policy Refinement 

**Title (ZH)**: 基于有界支撑演化策略的策略精炼 

**Authors**: Ethan Hirschowitz, Fabio Ramos  

**Link**: [PDF](https://arxiv.org/pdf/2511.09923)  

**Abstract**: Improving competent robot policies with on-policy RL is often hampered by noisy, low-signal gradients. We revisit Evolution Strategies (ES) as a policy-gradient proxy and localize exploration with bounded, antithetic triangular perturbations, suitable for policy refinement. We propose Triangular-Distribution ES (TD-ES) which pairs bounded triangular noise with a centered-rank finite-difference estimator to deliver stable, parallelizable, gradient-free updates. In a two-stage pipeline -- PPO pretraining followed by TD-ES refinement -- this preserves early sample efficiency while enabling robust late-stage gains. Across a suite of robotic manipulation tasks, TD-ES raises success rates by 26.5% relative to PPO and greatly reduces variance, offering a simple, compute-light path to reliable refinement. 

**Abstract (ZH)**: 利用演化策略改进有能力的机器人策略：基于三角分布的演化策略（TD-ES）提高鲁棒性高效细调方法 

---
# SEBA: Sample-Efficient Black-Box Attacks on Visual Reinforcement Learning 

**Title (ZH)**: SEBA: 样本高效的黑盒攻击方法针对视觉强化学习 

**Authors**: Tairan Huang, Yulin Jin, Junxu Liu, Qingqing Ye, Haibo Hu  

**Link**: [PDF](https://arxiv.org/pdf/2511.09681)  

**Abstract**: Visual reinforcement learning has achieved remarkable progress in visual control and robotics, but its vulnerability to adversarial perturbations remains underexplored. Most existing black-box attacks focus on vector-based or discrete-action RL, and their effectiveness on image-based continuous control is limited by the large action space and excessive environment queries. We propose SEBA, a sample-efficient framework for black-box adversarial attacks on visual RL agents. SEBA integrates a shadow Q model that estimates cumulative rewards under adversarial conditions, a generative adversarial network that produces visually imperceptible perturbations, and a world model that simulates environment dynamics to reduce real-world queries. Through a two-stage iterative training procedure that alternates between learning the shadow model and refining the generator, SEBA achieves strong attack performance while maintaining efficiency. Experiments on MuJoCo and Atari benchmarks show that SEBA significantly reduces cumulative rewards, preserves visual fidelity, and greatly decreases environment interactions compared to prior black-box and white-box methods. 

**Abstract (ZH)**: 视觉增强学习在视觉控制和机器人技术中取得了显著进展，但其对对抗性扰动的脆弱性仍待深入探索。现有的大多数黑盒攻击集中在基于向量或离散动作的增强学习上，它们在基于图像的连续控制方面由于动作空间庞大和环境查询过多而效果有限。我们提出了一种样本高效框架SEBA，用于视觉RL代理的黑盒对抗性攻击。SEBA整合了一种阴影Q模型，用于在对抗条件下估计累计奖励；一种生成对抗网络，产生视觉上不可察觉的扰动；以及一种世界模型，用于模拟环境动力学以减少真实的环境查询。通过交替学习阴影模型和优化生成器的两阶段迭代训练过程，SEBA实现了强大的攻击性能并保持了高效性。在MuJoCo和Atari基准测试中的实验表明，SEBA明显减少了累计奖励，保持了视觉保真度，显著减少了环境交互，优于先前的黑盒和灰盒方法。 

---
# Optimistic Reinforcement Learning with Quantile Objectives 

**Title (ZH)**: 乐观强化学习与分位数目标 

**Authors**: Mohammad Alipour-Vaezi, Huaiyang Zhong, Kwok-Leung Tsui, Sajad Khodadadian  

**Link**: [PDF](https://arxiv.org/pdf/2511.09652)  

**Abstract**: Reinforcement Learning (RL) has achieved tremendous success in recent years. However, the classical foundations of RL do not account for the risk sensitivity of the objective function, which is critical in various fields, including healthcare and finance. A popular approach to incorporate risk sensitivity is to optimize a specific quantile of the cumulative reward distribution. In this paper, we develop UCB-QRL, an optimistic learning algorithm for the $\tau$-quantile objective in finite-horizon Markov decision processes (MDPs). UCB-QRL is an iterative algorithm in which, at each iteration, we first estimate the underlying transition probability and then optimize the quantile value function over a confidence ball around this estimate. We show that UCB-QRL yields a high-probability regret bound $\mathcal O\left((2/\kappa)^{H+1}H\sqrt{SATH\log(2SATH/\delta)}\right)$ in the episodic setting with $S$ states, $A$ actions, $T$ episodes, and $H$ horizons. Here, $\kappa>0$ is a problem-dependent constant that captures the sensitivity of the underlying MDP's quantile value. 

**Abstract (ZH)**: 强化学习（RL）近年来取得了 tremendous 成功。然而，经典的 RL 基础没有考虑到目标函数的风险敏感性，这在包括医疗保健和金融在内的多个领域至关重要。一种常见的结合风险敏感性的方法是优化奖励累积分布的特定分位数。本文我们开发了 UCB-QRL，这是一种针对有限时域马尔可夫决策过程（MDPs）中 $\tau$ 分位数目标的乐观学习算法。UCB-QRL 是一种迭代算法，在每次迭代中，我们首先估计基础转移概率，然后在该估计周围的置信球中优化分位数值函数。我们证明，在有 $S$ 个状态、$A$ 个动作、$T$ 次episode 和 $H$ 个时域的 episodic 设定中，UCB-QRL 可以得到高概率后悔上界 $\mathcal O\left((2/\kappa)^{H+1}H\sqrt{SATH\log(2SATH/\delta)}\right)$。这里，$\kappa>0$ 是一个与 MDP 的分位数值函数敏感性相关的依赖问题的常数。 

---
