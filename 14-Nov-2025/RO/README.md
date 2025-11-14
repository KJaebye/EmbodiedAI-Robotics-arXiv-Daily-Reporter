# Robot Crash Course: Learning Soft and Stylized Falling 

**Title (ZH)**: 机器人入门教程：学习柔和和风格化的跌落 

**Authors**: Pascal Strauch, David Müller, Sammy Christen, Agon Serifi, Ruben Grandia, Espen Knoop, Moritz Bächer  

**Link**: [PDF](https://arxiv.org/pdf/2511.10635)  

**Abstract**: Despite recent advances in robust locomotion, bipedal robots operating in the real world remain at risk of falling. While most research focuses on preventing such events, we instead concentrate on the phenomenon of falling itself. Specifically, we aim to reduce physical damage to the robot while providing users with control over a robot's end pose. To this end, we propose a robot agnostic reward function that balances the achievement of a desired end pose with impact minimization and the protection of critical robot parts during reinforcement learning. To make the policy robust to a broad range of initial falling conditions and to enable the specification of an arbitrary and unseen end pose at inference time, we introduce a simulation-based sampling strategy of initial and end poses. Through simulated and real-world experiments, our work demonstrates that even bipedal robots can perform controlled, soft falls. 

**Abstract (ZH)**: 尽管在稳健运动方面取得了近期进展，但在现实世界中运行的双足机器人仍然面临摔倒的风险。大多数研究集中在预防此类事件，而我们则将重点放在摔倒现象本身。具体而言，我们旨在在强化学习过程中平衡期望结束姿态的实现、冲击最小化以及保护关键机器人部件，从而减少对机器人的物理损伤并提供用户对机器人结束姿态的控制。为此，我们提出了一种机器人无关的奖励函数，该函数在期望结束姿态的实现与冲击最小化以及保护关键机器人部件之间取得平衡。为了使策略能够应对广泛的初始摔倒条件，并允许在推理时指定任意和未见过的结束姿态，我们引入了一种基于仿真的初始姿态和结束姿态采样策略。通过模拟和真实世界实验，我们的工作证明了即使双足机器人也能执行可控的、软着陆的摔倒。 

---
# Optimizing the flight path for a scouting Uncrewed Aerial Vehicle 

**Title (ZH)**: 优化侦察无人驾驶航空器的飞行路径 

**Authors**: Raghav Adhikari, Sachet Khatiwada, Suman Poudel  

**Link**: [PDF](https://arxiv.org/pdf/2511.10598)  

**Abstract**: Post-disaster situations pose unique navigation challenges. One of those challenges is the unstructured nature of the environment, which makes it hard to layout paths for rescue vehicles. We propose the use of Uncrewed Aerial Vehicle (UAV) in such scenario to perform reconnaissance across the environment. To accomplish this, we propose an optimization-based approach to plan a path for the UAV at optimal height where the sensors of the UAV can cover the most area and collect data with minimum uncertainty. 

**Abstract (ZH)**: 灾难后的情况提出了独特的导航挑战。其中一项挑战是环境的无结构特性，这使得为救援车辆规划路径变得困难。我们提议在 such 场景中使用无人机（UAV）进行环境 Reconnaissance。为此，我们提出了一种基于优化的方法，用于为无人机规划在最佳高度上的路径，以使无人机的传感器能够覆盖最大面积并以最小的不确定性收集数据。 

---
# From Fold to Function: Dynamic Modeling and Simulation-Driven Design of Origami Mechanisms 

**Title (ZH)**: 从折叠到功能： Origami机制的动态建模与仿真驱动设计 

**Authors**: Tianhui Han, Shashwat Singh, Sarvesh Patil, Zeynep Temel  

**Link**: [PDF](https://arxiv.org/pdf/2511.10580)  

**Abstract**: Origami-inspired mechanisms can transform flat sheets into functional three-dimensional dynamic structures that are lightweight, compact, and capable of complex motion. These properties make origami increasingly valuable in robotic and deployable systems. However, accurately simulating their folding behavior and interactions with the environment remains challenging. To address this, we present a design framework for origami mechanism simulation that utilizes MuJoCo's deformable-body capabilities. In our approach, origami sheets are represented as graphs of interconnected deformable elements with user-specified constraints such as creases and actuation, defined through an intuitive graphical user interface (GUI). This framework allows users to generate physically consistent simulations that capture both the geometric structure of origami mechanisms and their interactions with external objects and surfaces. We demonstrate our method's utility through a case study on an origami catapult, where design parameters are optimized in simulation using the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) and validated experimentally on physical prototypes. The optimized structure achieves improved throwing performance, illustrating how our system enables rapid, simulation-driven origami design, optimization, and analysis. 

**Abstract (ZH)**: origami-inspired 机制可以将扁平板材转化为轻巧、紧凑且具备复杂运动功能的三维动态结构。这些特性使其在机器人和可展系统中越来越有价值。然而，准确模拟其折叠行为及其与环境的相互作用仍然具有挑战性。为此，我们提出了一种利用 MuJoCo 可变形体能力的 origami 机制仿真设计框架。在我们的方法中，origami 表面通过包含用户指定约束（如折痕和驱动）的相互连接的可变形元件表示，并通过直观的图形用户界面（GUI）进行定义。该框架允许用户生成物理上一致的仿真，既能捕捉 origami 机制的几何结构，又能反映其与外部物体和表面的相互作用。我们通过一个 origami 弹射器案例研究演示了该方法的实用性，其中利用 Covariance Matrix Adaptation Evolution Strategy (CMA-ES) 在仿真中优化设计参数，并在物理原型上进行实验验证。优化后的结构实现了更好的投掷性能，展示了我们的系统如何实现快速、基于仿真的 origami 设计、优化和分析。 

---
# Improving dependability in robotized bolting operations 

**Title (ZH)**: 提高机器人化锚喷作业的可靠性 

**Authors**: Lorenzo Pagliara, Violeta Redondo, Enrico Ferrentino, Manuel Ferre, Pasquale Chiacchio  

**Link**: [PDF](https://arxiv.org/pdf/2511.10448)  

**Abstract**: Bolting operations are critical in industrial assembly and in the maintenance of scientific facilities, requiring high precision and robustness to faults. Although robotic solutions have the potential to improve operational safety and effectiveness, current systems still lack reliable autonomy and fault management capabilities. To address this gap, we propose a control framework for dependable robotized bolting tasks and instantiate it on a specific robotic system. The system features a control architecture ensuring accurate driving torque control and active compliance throughout the entire operation, enabling safe interaction even under fault conditions. By designing a multimodal human-robot interface (HRI) providing real-time visualization of relevant system information and supporting seamless transitions between automatic and manual control, we improve operator situation awareness and fault detection capabilities. A high-level supervisor (SV) coordinates the execution and manages transitions between control modes, ensuring consistency with the supervisory control (SVC) paradigm, while preserving the human operator's authority. The system is validated in a representative bolting operation involving pipe flange joining, under several fault conditions. The results demonstrate improved fault detection capabilities, enhanced operator situational awareness, and accurate and compliant execution of the bolting operation. However, they also reveal the limitations of relying on a single camera to achieve full situational awareness. 

**Abstract (ZH)**: 机器人化紧固作业的控制框架及其在特定机器人系统中的实现：面向可靠的故障管理与操作安全性 

---
# LongComp: Long-Tail Compositional Zero-Shot Generalization for Robust Trajectory Prediction 

**Title (ZH)**: 长尾组合零-shot泛化：面向稳健轨迹预测 

**Authors**: Benjamin Stoler, Jonathan Francis, Jean Oh  

**Link**: [PDF](https://arxiv.org/pdf/2511.10411)  

**Abstract**: Methods for trajectory prediction in Autonomous Driving must contend with rare, safety-critical scenarios that make reliance on real-world data collection alone infeasible. To assess robustness under such conditions, we propose new long-tail evaluation settings that repartition datasets to create challenging out-of-distribution (OOD) test sets. We first introduce a safety-informed scenario factorization framework, which disentangles scenarios into discrete ego and social contexts. Building on analogies to compositional zero-shot image-labeling in Computer Vision, we then hold out novel context combinations to construct challenging closed-world and open-world settings. This process induces OOD performance gaps in future motion prediction of 5.0% and 14.7% in closed-world and open-world settings, respectively, relative to in-distribution performance for a state-of-the-art baseline. To improve generalization, we extend task-modular gating networks to operate within trajectory prediction models, and develop an auxiliary, difficulty-prediction head to refine internal representations. Our strategies jointly reduce the OOD performance gaps to 2.8% and 11.5% in the two settings, respectively, while still improving in-distribution performance. 

**Abstract (ZH)**: 自主驾驶中轨迹预测方法必须应对罕见的安全关键场景，这使得依赖真实的实地数据收集不可行。为了在这些条件下评估鲁棒性，我们提出了新的长尾评估设置，重新分配数据集以创建具有挑战性的离分布测试集。首先，我们引入了一种基于安全的信息场景分解框架，将场景分解为离散的自我和社交背景。在此基础上，借鉴计算机视觉中组合零样本图像标记的类比，我们保留新的上下文组合来构建具有挑战性的闭世界和开世界设置。这一过程在闭世界和开世界设置中分别引起了5.0%和14.7%的离分布性能差距相对于内部分布性能，对于一个最先进的基线。为了提高泛化能力，我们扩展了任务模块门控网络在轨迹预测模型中的操作，并开发了一个辅助的难度预测头部来细化内部表示。我们的策略分别将两种设置中的离分布性能差距减少到2.8%和11.5%，同时仍然提高内部分布性能。 

---
# nuPlan-R: A Closed-Loop Planning Benchmark for Autonomous Driving via Reactive Multi-Agent Simulation 

**Title (ZH)**: nuPlan-R：基于反应式多Agent模拟的自主驾驶闭环规划基准测试 

**Authors**: Mingxing Peng, Ruoyu Yao, Xusen Guo, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2511.10403)  

**Abstract**: Recent advances in closed-loop planning benchmarks have significantly improved the evaluation of autonomous vehicles. However, existing benchmarks still rely on rule-based reactive agents such as the Intelligent Driver Model (IDM), which lack behavioral diversity and fail to capture realistic human interactions, leading to oversimplified traffic dynamics. To address these limitations, we present nuPlan-R, a new reactive closed-loop planning benchmark that integrates learning-based reactive multi-agent simulation into the nuPlan framework. Our benchmark replaces the rule-based IDM agents with noise-decoupled diffusion-based reactive agents and introduces an interaction-aware agent selection mechanism to ensure both realism and computational efficiency. Furthermore, we extend the benchmark with two additional metrics to enable a more comprehensive assessment of planning performance. Extensive experiments demonstrate that our reactive agent model produces more realistic, diverse, and human-like traffic behaviors, leading to a benchmark environment that better reflects real-world interactive driving. We further reimplement a collection of rule-based, learning-based, and hybrid planning approaches within our nuPlan-R benchmark, providing a clearer reflection of planner performance in complex interactive scenarios and better highlighting the advantages of learning-based planners in handling complex and dynamic scenarios. These results establish nuPlan-R as a new standard for fair, reactive, and realistic closed-loop planning evaluation. We will open-source the code for the new benchmark. 

**Abstract (ZH)**: 近期闭环规划基准的进展显著提高了自主车辆的评估标准。然而，现有基准仍然依赖于基于规则的反应式代理，如智能驾驶员模型（IDM），这些代理缺乏行为多样性，无法捕捉真实的人类交互，导致交通动力学过于简化。为了应对这些限制，我们提出nuPlan-R，这是一个新的基于学习的反应式闭环规划基准，将基于学习的多代理反应式仿真集成到nuPlan框架中。我们的基准用去噪声的扩散基反应式代理取代了基于规则的IDM代理，并引入了一种基于交互感知的代理选择机制，以确保真实性和计算效率。此外，我们还扩展了基准，引入了两个额外的评估指标，以实现对规划性能的更全面评估。大量实验表明，我们提出的反应式代理模型生成了更加真实、多样且人类似的行为，从而构建了一个更能反映真实交互驾驶场景的基准环境。我们进一步在nuPlan-R基准中实现了一组基于规则的、基于学习的和混合规划方法，以更清晰地反映规划器在复杂交互场景中的性能，并更突出基于学习的规划器在处理复杂和动态场景中的优势。这些结果确立了nuPlan-R作为公平、反应式和现实的闭环规划评估新标准的地位。我们将开源新基准的代码。 

---
# RoboBenchMart: Benchmarking Robots in Retail Environment 

**Title (ZH)**: RoboBenchMart: 零售环境中的机器人基准测试 

**Authors**: Konstantin Soshin, Alexander Krapukhin, Andrei Spiridonov, Denis Shepelev, Gregorii Bukhtuev, Andrey Kuznetsov, Vlad Shakhuro  

**Link**: [PDF](https://arxiv.org/pdf/2511.10276)  

**Abstract**: Most existing robotic manipulation benchmarks focus on simplified tabletop scenarios, typically involving a stationary robotic arm interacting with various objects on a flat surface. To address this limitation, we introduce RoboBenchMart, a more challenging and realistic benchmark designed for dark store environments, where robots must perform complex manipulation tasks with diverse grocery items. This setting presents significant challenges, including dense object clutter and varied spatial configurations -- with items positioned at different heights, depths, and in close proximity. By targeting the retail domain, our benchmark addresses a setting with strong potential for near-term automation impact. We demonstrate that current state-of-the-art generalist models struggle to solve even common retail tasks. To support further research, we release the RoboBenchMart suite, which includes a procedural store layout generator, a trajectory generation pipeline, evaluation tools and fine-tuned baseline models. 

**Abstract (ZH)**: RoboBenchMart：针对暗储区环境的更具挑战性和现实性的机器人操作基准 

---
# Learning a Thousand Tasks in a Day 

**Title (ZH)**: 每天学习一千种任务 

**Authors**: Kamil Dreczkowski, Pietro Vitiello, Vitalis Vosylius, Edward Johns  

**Link**: [PDF](https://arxiv.org/pdf/2511.10110)  

**Abstract**: Humans are remarkably efficient at learning tasks from demonstrations, but today's imitation learning methods for robot manipulation often require hundreds or thousands of demonstrations per task. We investigate two fundamental priors for improving learning efficiency: decomposing manipulation trajectories into sequential alignment and interaction phases, and retrieval-based generalisation. Through 3,450 real-world rollouts, we systematically study this decomposition. We compare different design choices for the alignment and interaction phases, and examine generalisation and scaling trends relative to today's dominant paradigm of behavioural cloning with a single-phase monolithic policy. In the few-demonstrations-per-task regime (<10 demonstrations), decomposition achieves an order of magnitude improvement in data efficiency over single-phase learning, with retrieval consistently outperforming behavioural cloning for both alignment and interaction. Building on these insights, we develop Multi-Task Trajectory Transfer (MT3), an imitation learning method based on decomposition and retrieval. MT3 learns everyday manipulation tasks from as little as a single demonstration each, whilst also generalising to novel object instances. This efficiency enables us to teach a robot 1,000 distinct everyday tasks in under 24 hours of human demonstrator time. Through 2,200 additional real-world rollouts, we reveal MT3's capabilities and limitations across different task families. Videos of our experiments can be found on at this https URL. 

**Abstract (ZH)**: 人类在从演示中学任务方面表现出惊人的效率，但当前用于机器人操作的模仿学习方法往往需要每项任务成百上千次的演示。我们研究了提高学习效率的两个基本先验知识：将操作轨迹分解为序列对齐和交互阶段，并采用基于检索的泛化。通过3,450个真实世界的试验，我们系统地研究了这种分解。我们将不同对齐和交互阶段设计选择进行了比较，并考察了与当前主导的行为克隆单一阶段整体策略相比的泛化和扩展趋势。在每项任务使用少量演示(<10次演示)的范围内，分解在数据效率上实现了数量级的提升，检索在对齐和交互阶段均持续优于行为克隆。基于这些见解，我们开发了基于分解和检索的多任务轨迹转移（MT3）模仿学习方法。MT3可以从每项任务一个演示开始学习日常操作任务，并泛化到新的对象实例。这种效率使我们能够在不到24小时的人类演示时间内教会机器人1,000个不同的日常任务。通过2,200个额外的真实世界试验，我们揭示了MT3在不同任务家族中的能力和局限性。我们的实验视频可在以下链接找到：这个 https URL。 

---
# Opinion: Towards Unified Expressive Policy Optimization for Robust Robot Learning 

**Title (ZH)**: 观点：面向鲁棒机器人学习的统一表达性策略优化 

**Authors**: Haidong Huang, Haiyue Zhu. Jiayu Song, Xixin Zhao, Yaohua Zhou, Jiayi Zhang, Yuze Zhai, Xiaocong Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.10087)  

**Abstract**: Offline-to-online reinforcement learning (O2O-RL) has emerged as a promising paradigm for safe and efficient robotic policy deployment but suffers from two fundamental challenges: limited coverage of multimodal behaviors and distributional shifts during online adaptation. We propose UEPO, a unified generative framework inspired by large language model pretraining and fine-tuning strategies. Our contributions are threefold: (1) a multi-seed dynamics-aware diffusion policy that efficiently captures diverse modalities without training multiple models; (2) a dynamic divergence regularization mechanism that enforces physically meaningful policy diversity; and (3) a diffusion-based data augmentation module that enhances dynamics model generalization. On the D4RL benchmark, UEPO achieves +5.9\% absolute improvement over Uni-O4 on locomotion tasks and +12.4\% on dexterous manipulation, demonstrating strong generalization and scalability. 

**Abstract (ZH)**: 基于离线到在线强化学习（O2O-RL）已 emerge 作为一种在安全和高效地部署机器人策略方面的有前途的范式，但面临两大根本挑战：多模态行为的有限覆盖和在线适应过程中的分布偏移。我们提出了 UEPO，这是一种受大规模语言模型预训练和微调策略启发的统一生成框架。我们的贡献包括：（1）一个多种子动力学感知的扩散策略，能够在不训练多个模型的情况下高效捕捉多样性；（2）一种动态差异正则化机制，以确保物理上合理的策略多样性；（3）一种基于扩散的数据增强模块，以增强动力学模型泛化能力。在 D4RL 基准测试中，UEPO 在运动任务上比 Uni-O4 实现了 +5.9% 的绝对改进，在灵巧操作上实现了 +12.4% 的改进，证明了其强大的泛化能力和可扩展性。 

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

**Title (ZH)**: phantom威胁：探索和增强VLA模型Against物理传感器攻击的鲁棒性 

**Authors**: Xuancun Lu, Jiaxiang Chen, Shilin Xiao, Zizhi Jin, Zhangrui Chen, Hanwen Yu, Bohan Qian, Ruochen Zhou, Xiaoyu Ji, Wenyuan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.10008)  

**Abstract**: Vision-Language-Action (VLA) models revolutionize robotic systems by enabling end-to-end perception-to-action pipelines that integrate multiple sensory modalities, such as visual signals processed by cameras and auditory signals captured by microphones. This multi-modality integration allows VLA models to interpret complex, real-world environments using diverse sensor data streams. Given the fact that VLA-based systems heavily rely on the sensory input, the security of VLA models against physical-world sensor attacks remains critically underexplored.
To address this gap, we present the first systematic study of physical sensor attacks against VLAs, quantifying the influence of sensor attacks and investigating the defenses for VLA models. We introduce a novel ``Real-Sim-Real'' framework that automatically simulates physics-based sensor attack vectors, including six attacks targeting cameras and two targeting microphones, and validates them on real robotic systems. Through large-scale evaluations across various VLA architectures and tasks under varying attack parameters, we demonstrate significant vulnerabilities, with susceptibility patterns that reveal critical dependencies on task types and model designs. We further develop an adversarial-training-based defense that enhances VLA robustness against out-of-distribution physical perturbations caused by sensor attacks while preserving model performance. Our findings expose an urgent need for standardized robustness benchmarks and mitigation strategies to secure VLA deployments in safety-critical environments. 

**Abstract (ZH)**: 基于视觉-语言-行动(VLA)模型的物理传感器攻击研究：框架、影响与防御 

---
# Audio-VLA: Adding Contact Audio Perception to Vision-Language-Action Model for Robotic Manipulation 

**Title (ZH)**: 音频-视觉行动模型中加入接触音频感知的研究：针对机器人操作的任务 

**Authors**: Xiangyi Wei, Haotian Zhang, Xinyi Cao, Siyu Xie, Weifeng Ge, Yang Li, Changbo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.09958)  

**Abstract**: The Vision-Language-Action models (VLA) have achieved significant advances in robotic manipulation recently. However, vision-only VLA models create fundamental limitations, particularly in perceiving interactive and manipulation dynamic processes. This paper proposes Audio-VLA, a multimodal manipulation policy that leverages contact audio to perceive contact events and dynamic process feedback. Audio-VLA overcomes the vision-only constraints of VLA models. Additionally, this paper introduces the Task Completion Rate (TCR) metric to systematically evaluate dynamic operational processes. Audio-VLA employs pre-trained DINOv2 and SigLIP as visual encoders, AudioCLIP as the audio encoder, and Llama2 as the large language model backbone. We apply LoRA fine-tuning to these pre-trained modules to achieve robust cross-modal understanding of both visual and acoustic inputs. A multimodal projection layer aligns features from different modalities into the same feature space. Moreover RLBench and LIBERO simulation environments are enhanced by adding collision-based audio generation to provide realistic sound feedback during object interactions. Since current robotic manipulation evaluations focus on final outcomes rather than providing systematic assessment of dynamic operational processes, the proposed TCR metric measures how well robots perceive dynamic processes during manipulation, creating a more comprehensive evaluation metric. Extensive experiments on LIBERO, RLBench, and two real-world tasks demonstrate Audio-VLA's superior performance over vision-only comparative methods, while the TCR metric effectively quantifies dynamic process perception capabilities. 

**Abstract (ZH)**: 基于音频的视觉语言动作模型（Audio-VLA）及其动态操作过程评价方法 

---
# A Study on Enhancing the Generalization Ability of Visuomotor Policies via Data Augmentation 

**Title (ZH)**: 基于数据增强提升视动策略的泛化能力的研究 

**Authors**: Hanwen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.09932)  

**Abstract**: The generalization ability of visuomotor policy is crucial, as a good policy should be deployable across diverse scenarios. Some methods can collect large amounts of trajectory augmentation data to train more generalizable imitation learning policies, aimed at handling the random placement of objects on the scene's horizontal plane. However, the data generated by these methods still lack diversity, which limits the generalization ability of the trained policy. To address this, we investigate the performance of policies trained by existing methods across different scene layout factors via automate the data generation for those factors that significantly impact generalization. We have created a more extensively randomized dataset that can be efficiently and automatically generated with only a small amount of human demonstration. The dataset covers five types of manipulators and two types of grippers, incorporating extensive randomization factors such as camera pose, lighting conditions, tabletop texture, and table height across six manipulation tasks. We found that all of these factors influence the generalization ability of the policy. Applying any form of randomization enhances policy generalization, with diverse trajectories particularly effective in bridging visual gap. Notably, we investigated on low-cost manipulator the effect of the scene randomization proposed in this work on enhancing the generalization capability of visuomotor policies for zero-shot sim-to-real transfer. 

**Abstract (ZH)**: 视觉运动策略的泛化能力至关重要，因为一个好的策略应该可以在多种场景下部署。现有的方法可以通过收集大量的轨迹增广数据来训练更具泛化能力的模仿学习策略，以应对场景水平面上随机放置的对象。然而，这些方法生成的数据仍然缺乏多样性，这限制了训练策略的泛化能力。为了解决这一问题，我们通过自动化那些对泛化影响显著的场景布局因素的数据生成过程，研究现有方法训练策略在不同场景布局因素下的性能。我们创建了一个更为广泛随机化的数据集，该数据集可以仅通过少量的人类示范高效且自动地生成。数据集涵盖了五种类型的操纵器和两种类型的夹爪，并包含了广泛随机化因素，如摄像头姿态、光照条件、桌面纹理和桌子高度，这些因素贯穿六个操作任务。我们发现这些因素都会影响策略的泛化能力。任何形式的随机化都可以增强策略的泛化能力，尤其是多样化的轨迹特别有效于填补视觉差距。特别地，我们研究了在低成本操纵器上，本文提出的场景随机化对提高视觉运动策略零样本仿真实际转移能力的影响。 

---
# PuffyBot: An Untethered Shape Morphing Robot for Multi-environment Locomotion 

**Title (ZH)**: PuffyBot：一个无绳形态变化机器人，适用于多环境运动 

**Authors**: Shashwat Singh, Zilin Si, Zeynep Temel  

**Link**: [PDF](https://arxiv.org/pdf/2511.09885)  

**Abstract**: Amphibians adapt their morphologies and motions to accommodate movement in both terrestrial and aquatic environments. Inspired by these biological features, we present PuffyBot, an untethered shape morphing robot capable of changing its body morphology to navigate multiple environments. Our robot design leverages a scissor-lift mechanism driven by a linear actuator as its primary structure to achieve shape morphing. The transformation enables a volume change from 255.00 cm3 to 423.75 cm3, modulating the buoyant force to counteract a downward force of 3.237 N due to 330 g mass of the robot. A bell-crank linkage is integrated with the scissor-lift mechanism, which adjusts the servo-actuated limbs by 90 degrees, allowing a seamless transition between crawling and swimming modes. The robot is fully waterproof, using thermoplastic polyurethane (TPU) fabric to ensure functionality in aquatic environments. The robot can operate untethered for two hours with an onboard battery of 1000 mA h. Our experimental results demonstrate multi-environment locomotion, including crawling on the land, crawling on the underwater floor, swimming on the water surface, and bimodal buoyancy adjustment to submerge underwater or resurface. These findings show the potential of shape morphing to create versatile and energy efficient robotic platforms suitable for diverse environments. 

**Abstract (ZH)**: 两栖动物适应其形态和运动以适应陆地和水生环境的变化。受此生物特征的启发，我们提出了PuffyBot，一种无需缆绳的形态变化机器人，能够改变其身体形态以导航多个环境。我们的机器人设计利用由线性执行器驱动的剪式升降机构作为主要结构，实现形态变化。这种变换使得体积从255.00 cm³变化到423.75 cm³，调节浮力以抵消330 g机器人质量产生的3.237 N下向力。摆杆连杆与剪式升降机构集成，通过90度调整伺服驱动的肢体，实现爬行模式和游泳模式之间的无缝转换。机器人完全防水，使用热塑性聚氨酯（TPU）织物确保其在水生环境中的功能。机器人配备1000 mA h的内置电池，可以在未缆绳束缚的情况下运行两小时。我们的实验结果展示了多环境移动能力，包括在陆地上爬行、在水下地板上爬行、在水面游泳以及双模式浮力调节以潜入水中或浮出水面。这些发现表明形态变化有潜力创建多功能且能效高的机器人平台，适用于多变的环境。 

---
# Provably Safe Stein Variational Clarity-Aware Informative Planning 

**Title (ZH)**: 可验证安全的Stein变分清晰度意识信息性规划 

**Authors**: Kaleb Ben Naveed, Utkrisht Sahai, Anouck Girard, Dimitra Panagou  

**Link**: [PDF](https://arxiv.org/pdf/2511.09836)  

**Abstract**: Autonomous robots are increasingly deployed for information-gathering tasks in environments that vary across space and time. Planning informative and safe trajectories in such settings is challenging because information decays when regions are not revisited. Most existing planners model information as static or uniformly decaying, ignoring environments where the decay rate varies spatially; those that model non-uniform decay often overlook how it evolves along the robot's motion, and almost all treat safety as a soft penalty. In this paper, we address these challenges. We model uncertainty in the environment using clarity, a normalized representation of differential entropy from our earlier work that captures how information improves through new measurements and decays over time when regions are not revisited. Building on this, we present Stein Variational Clarity-Aware Informative Planning, a framework that embeds clarity dynamics within trajectory optimization and enforces safety through a low-level filtering mechanism based on our earlier gatekeeper framework for safety verification. The planner performs Bayesian inference-based learning via Stein variational inference, refining a distribution over informative trajectories while filtering each nominal Stein informative trajectory to ensure safety. Hardware experiments and simulations across environments with varying decay rates and obstacles demonstrate consistent safety and reduced information deficits. 

**Abstract (ZH)**: 自主机器人在时空变化环境中进行信息收集任务的应用日益增多。在这些环境中规划既能提供信息又不失安全性轨迹是一项具有挑战性的任务，因为未重新访问的区域中的信息会逐渐衰减。现有的大多数规划器假设信息是静态的或均匀衰减的，忽略了衰减率在不同区域之间变化的环境；而那些建模非均匀衰减的规划器往往忽略了这种衰减如何随机器人运动而演变，并几乎都将安全性视为一种软约束。本文针对上述挑战进行了研究。我们使用清晰度进行环境中的不确定性建模，这是一种以归一化差异熵为基础的表示方法，捕捉信息通过新测量得到改善并在未重新访问的区域中随时间衰减的过程。在此基础上，我们提出了Stein变分清晰度感知信息规划框架，该框架将清晰度动力学嵌入轨迹优化中，并通过基于我们之前的安全验证框架中的门keeper机制的低级滤波机制来确保安全性。规划器通过Stein变分推断进行贝叶斯推理学习，不断精炼具有信息性的轨迹分布，并对每个名义的Stein信息性轨迹进行滤波以确保安全性。跨越不同衰减率和障碍的环境的硬件实验和仿真表明，该方法能够实现一致的安全性和减少信息不足的现象。 

---
# A Robust Task-Level Control Architecture for Learned Dynamical Systems 

**Title (ZH)**: learned动态系统中鲁棒的任务级控制架构 

**Authors**: Eshika Pathak, Ahmed Aboudonia, Sandeep Banik, Naira Hovakimyan  

**Link**: [PDF](https://arxiv.org/pdf/2511.09790)  

**Abstract**: Dynamical system (DS)-based learning from demonstration (LfD) is a powerful tool for generating motion plans in the operation (`task') space of robotic systems. However, the realization of the generated motion plans is often compromised by a ''task-execution mismatch'', where unmodeled dynamics, persistent disturbances, and system latency cause the robot's actual task-space state to diverge from the desired motion trajectory. We propose a novel task-level robust control architecture, L1-augmented Dynamical Systems (L1-DS), that explicitly handles the task-execution mismatch in tracking a nominal motion plan generated by any DS-based LfD scheme. Our framework augments any DS-based LfD model with a nominal stabilizing controller and an L1 adaptive controller. Furthermore, we introduce a windowed Dynamic Time Warping (DTW)-based target selector, which enables the nominal stabilizing controller to handle temporal misalignment for improved phase-consistent tracking. We demonstrate the efficacy of our architecture on the LASA and IROS handwriting datasets. 

**Abstract (ZH)**: 基于动力学系统（DS）的学习从演示（LfD）生成机器人系统操作空间中的运动计划是一种强大工具。然而，生成的运动计划常常受到“任务执行不匹配”的影响，其中未建模的动力学、持续干扰和系统时延导致机器人的实际任务空间状态偏离期望的运动轨迹。我们提出了一种新颖的任务级鲁棒控制架构L1-增强动力学系统（L1-DS），以明确处理由任何基于DS的LfD方案生成的名义运动计划引起的任务执行不匹配问题。我们的框架将任何基于DS的LfD模型与一个名义稳态控制器和一个L1自适应控制器相结合。此外，我们引入了一个窗口化的动态时间规整（DTW）目标选择器，以使名义稳态控制器能够处理时间对齐问题，从而提高相位一致的跟踪效果。我们在LASA和IROS手写数据集上展示了我们架构的有效性。 

---
# Baby Sophia: A Developmental Approach to Self-Exploration through Self-Touch and Hand Regard 

**Title (ZH)**: 婴儿 Sophia：一种通过自我触摸和手的注意进行自我探索的发展方法 

**Authors**: Stelios Zarifis, Ioannis Chalkiadakis, Artemis Chardouveli, Vasiliki Moutzouri, Aggelos Sotirchos, Katerina Papadimitriou, Panagiotis Filntisis, Niki Efthymiou, Petros Maragos, Katerina Pastra  

**Link**: [PDF](https://arxiv.org/pdf/2511.09727)  

**Abstract**: Inspired by infant development, we propose a Reinforcement Learning (RL) framework for autonomous self-exploration in a robotic agent, Baby Sophia, using the BabyBench simulation environment. The agent learns self-touch and hand regard behaviors through intrinsic rewards that mimic an infant's curiosity-driven exploration of its own body. For self-touch, high-dimensional tactile inputs are transformed into compact, meaningful representations, enabling efficient learning. The agent then discovers new tactile contacts through intrinsic rewards and curriculum learning that encourage broad body coverage, balance, and generalization. For hand regard, visual features of the hands, such as skin-color and shape, are learned through motor babbling. Then, intrinsic rewards encourage the agent to perform novel hand motions, and follow its hands with its gaze. A curriculum learning setup from single-hand to dual-hand training allows the agent to reach complex visual-motor coordination. The results of this work demonstrate that purely curiosity-based signals, with no external supervision, can drive coordinated multimodal learning, imitating an infant's progression from random motor babbling to purposeful behaviors. 

**Abstract (ZH)**: 受婴儿发展启发，我们提出了一种用于自主自我探索的机器人代理Baby Sophia的强化学习（RL）框架，采用BabyBench模拟环境。代理通过模拟婴儿好奇驱动的身体探索，利用内在奖励学习自我触碰和手的关注行为。对于自我触碰，高维触觉输入被转换为紧凑且有意义的表示，从而实现高效的學習。代理然后通过内在奖励和鼓励广泛身体覆盖、平衡和泛化的课程学习来发现新的触觉接触。对于手的关注，通过运动咿呀学语学习手的视觉特征，如肤色和形状。随后，内在奖励促使代理执行新颖的手部动作，并用其视线跟随手部。从单手到双手训练的课程学习设置使得代理能够实现复杂的视觉-运动协调。这项工作的结果表明，纯粹基于好奇心的信号，无需外部监督，可以驱动协调的跨模态学习，模仿婴儿从随机运动咿呀学语到目标行为的进展。 

---
# A Shared-Autonomy Construction Robotic System for Overhead Works 

**Title (ZH)**: 一种用于高处作业的共享自治建筑机器人系统 

**Authors**: David Minkwan Kim, K. M. Brian Lee, Yong Hyeok Seo, Nikola Raicevic, Runfa Blark Li, Kehan Long, Chan Seon Yoon, Dong Min Kang, Byeong Jo Lim, Young Pyoung Kim, Nikolay Atanasov, Truong Nguyen, Se Woong Jun, Young Wook Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.09695)  

**Abstract**: We present the ongoing development of a robotic system for overhead work such as ceiling drilling. The hardware platform comprises a mobile base with a two-stage lift, on which a bimanual torso is mounted with a custom-designed drilling end effector and RGB-D cameras. To support teleoperation in dynamic environments with limited visibility, we use Gaussian splatting for online 3D reconstruction and introduce motion parameters to model moving objects. For safe operation around dynamic obstacles, we developed a neural configuration-space barrier approach for planning and control. Initial feasibility studies demonstrate the capability of the hardware in drilling, bolting, and anchoring, and the software in safe teleoperation in a dynamic environment. 

**Abstract (ZH)**: 我们呈现了一种用于顶面作业（如天花板钻孔）的机器人系统的发展进度。该硬件平台包括一个配备两级提升装置的移动基座，上面安装有一个配备定制设计的钻孔末端执行器和RGB-D相机的双臂躯干。为支持在有限视域的动态环境中的遥操作，我们采用了高斯点云法进行在线三维重建，并引入运动参数来建模移动对象。为确保在动态障碍物周围的安全操作，我们开发了一种基于神经配置空间屏障的方法进行规划与控制。初步可行性研究证明了该硬件在钻孔、螺接和锚定方面的能力，以及软件在动态环境中的安全遥操作能力。 

---
# ScaleADFG: Affordance-based Dexterous Functional Grasping via Scalable Dataset 

**Title (ZH)**: 基于功能性的灵巧抓取通过可扩展数据集的可承受性规模扩展方法 

**Authors**: Sizhe Wang, Yifan Yang, Yongkang Luo, Daheng Li, Wei Wei, Yan Zhang, Peiying Hu, Yunjin Fu, Haonan Duan, Jia Sun, Peng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.09602)  

**Abstract**: Dexterous functional tool-use grasping is essential for effective robotic manipulation of tools. However, existing approaches face significant challenges in efficiently constructing large-scale datasets and ensuring generalizability to everyday object scales. These issues primarily arise from size mismatches between robotic and human hands, and the diversity in real-world object scales. To address these limitations, we propose the ScaleADFG framework, which consists of a fully automated dataset construction pipeline and a lightweight grasp generation network. Our dataset introduce an affordance-based algorithm to synthesize diverse tool-use grasp configurations without expert demonstrations, allowing flexible object-hand size ratios and enabling large robotic hands (compared to human hands) to grasp everyday objects effectively. Additionally, we leverage pre-trained models to generate extensive 3D assets and facilitate efficient retrieval of object affordances. Our dataset comprising five object categories, each containing over 1,000 unique shapes with 15 scale variations. After filtering, the dataset includes over 60,000 grasps for each 2 dexterous robotic hands. On top of this dataset, we train a lightweight, single-stage grasp generation network with a notably simple loss design, eliminating the need for post-refinement. This demonstrates the critical importance of large-scale datasets and multi-scale object variant for effective training. Extensive experiments in simulation and on real robot confirm that the ScaleADFG framework exhibits strong adaptability to objects of varying scales, enhancing functional grasp stability, diversity, and generalizability. Moreover, our network exhibits effective zero-shot transfer to real-world objects. Project page is available at this https URL 

**Abstract (ZH)**: 灵巧功能工具使用抓取对于有效进行工具的机器人操作至关重要。然而，现有的方法在高效构建大规模数据集和确保日常物体尺度的一般化方面面临重大挑战。这些问题主要源于机器人手和人类手的尺寸不匹配以及现实世界中物体尺度的多样性。为了解决这些限制，我们提出了ScaleADFG框架，该框架包括一个完全自动的数据集构建管道和一个轻量级的抓取生成网络。我们的数据集采用基于功能的方法合成了各种工具使用抓取配置，无需专家演示，允许灵活的物体-手尺寸比，并使大型机器人手能够有效地抓取日常物品。此外，我们利用预训练模型生成了大量的3D资产，并促进了物体功能的高效检索。我们的数据集包含五个物种类别，每个类别包含超过1,000种独特形状，有15种尺度变化。过滤后，数据集包括超过60,000个抓取，针对每只2只灵巧的机器人手。在此数据集基础上，我们训练了一个轻量级的单阶段抓取生成网络，具有显著简单的损失设计，消除了事后优化的需要。这表明大规模数据集和多尺度物体变体对于有效的训练至关重要。在模拟和现实机器人中的广泛实验表明，ScaleADFG框架在各种尺度的对象上表现出很强的适应性，提高了功能抓取的稳定性、多样性和一般化能力。此外，我们的网络在零样本转移到现实世界物体上表现出有效的能力。项目页面可通过此链接访问。 

---
# Safe Planning in Interactive Environments via Iterative Policy Updates and Adversarially Robust Conformal Prediction 

**Title (ZH)**: 通过迭代策略更新和对抗性稳健的 conformal 预测在交互环境中实现安全规划 

**Authors**: Omid Mirzaeedodangeh, Eliot Shekhtman, Nikolai Matni, Lars Lindemann  

**Link**: [PDF](https://arxiv.org/pdf/2511.10586)  

**Abstract**: Safe planning of an autonomous agent in interactive environments -- such as the control of a self-driving vehicle among pedestrians and human-controlled vehicles -- poses a major challenge as the behavior of the environment is unknown and reactive to the behavior of the autonomous agent. This coupling gives rise to interaction-driven distribution shifts where the autonomous agent's control policy may change the environment's behavior, thereby invalidating safety guarantees in existing work. Indeed, recent works have used conformal prediction (CP) to generate distribution-free safety guarantees using observed data of the environment. However, CP's assumption on data exchangeability is violated in interactive settings due to a circular dependency where a control policy update changes the environment's behavior, and vice versa. To address this gap, we propose an iterative framework that robustly maintains safety guarantees across policy updates by quantifying the potential impact of a planned policy update on the environment's behavior. We realize this via adversarially robust CP where we perform a regular CP step in each episode using observed data under the current policy, but then transfer safety guarantees across policy updates by analytically adjusting the CP result to account for distribution shifts. This adjustment is performed based on a policy-to-trajectory sensitivity analysis, resulting in a safe, episodic open-loop planner. We further conduct a contraction analysis of the system providing conditions under which both the CP results and the policy updates are guaranteed to converge. We empirically demonstrate these safety and convergence guarantees on a two-dimensional car-pedestrian case study. To the best of our knowledge, these are the first results that provide valid safety guarantees in such interactive settings. 

**Abstract (ZH)**: 自主代理在互动环境中的安全规划——例如在行人和人工控制车辆中的自动驾驶车辆控制——面临重大挑战，因为环境的行为是未知且对自主代理行为作出反应的。这种耦合导致了由互动驱动的分布偏移，其中自主代理的控制策略可能改变环境的行为，从而在现有工作中撤销了安全保证。实际上，近期的工作利用了一致预测（CP）生成基于观察到的环境数据的安全保证。然而，由于在互动设置中数据交换性的假设被违反——控制策略的更新改变了环境的行为，反之亦然——这导致了问题。为此，我们提出了一种迭代框架，通过量化计划策略更新对环境行为的潜在影响，稳健地保持跨策略更新的安全保证。我们通过对手抗性一致预测（CP）来实现这一目标，在每个时期使用当前策略下的观察数据执行常规CP步骤，然后通过分析调整CP结果以考虑分布偏移，跨策略更新传递安全保证。这种调整基于策略到轨迹的敏感性分析，从而实现一个安全的、分段的开环规划器。我们还进行了收缩分析，提供了确保一致预测结果和策略更新收敛的条件。我们通过一个二维汽车-行人案例研究，经验性地展示了这些安全性和收敛性保证。据我们所知，这是首次在这样的互动设置中提供有效安全保证的结果。 

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
# VISTA: A Vision and Intent-Aware Social Attention Framework for Multi-Agent Trajectory Prediction 

**Title (ZH)**: VISTA：一种视觉和意图感知的社会注意力框架用于多agent轨迹预测 

**Authors**: Stephane Da Silva Martins, Emanuel Aldea, Sylvie Le Hégarat-Mascle  

**Link**: [PDF](https://arxiv.org/pdf/2511.10203)  

**Abstract**: Multi-agent trajectory prediction is crucial for autonomous systems operating in dense, interactive environments. Existing methods often fail to jointly capture agents' long-term goals and their fine-grained social interactions, which leads to unrealistic multi-agent futures. We propose VISTA, a recursive goal-conditioned transformer for multi-agent trajectory forecasting. VISTA combines (i) a cross-attention fusion module that integrates long-horizon intent with past motion, (ii) a social-token attention mechanism for flexible interaction modeling across agents, and (iii) pairwise attention maps that make social influence patterns interpretable at inference time. Our model turns single-agent goal-conditioned prediction into a coherent multi-agent forecasting framework. Beyond standard displacement metrics, we evaluate trajectory collision rates as a measure of joint realism. On the high-density MADRAS benchmark and on SDD, VISTA achieves state-of-the-art accuracy and substantially fewer collisions. On MADRAS, it reduces the average collision rate of strong baselines from 2.14 to 0.03 percent, and on SDD it attains zero collisions while improving ADE, FDE, and minFDE. These results show that VISTA generates socially compliant, goal-aware, and interpretable trajectories, making it promising for safety-critical autonomous systems. 

**Abstract (ZH)**: 多代理轨迹预测对于在密集交互环境中运行的自主系统至关重要。现有的方法往往无法同时捕捉到代理的长期目标及其精细的社会互动，导致不现实的多代理未来轨迹。我们提出VISTA，一种递归目标条件变换器用于多代理轨迹预测。VISTA结合了(i)一种跨注意力融合模块，将长期意图与过去运动整合，(ii)一种社会标记注意力机制，用于灵活地建模代理间的互动，以及(iii)一对注意力图，使得社会影响模式在推理时可解释。我们的模型将单代理目标条件预测转化为一个一致的多代理预测框架。除了标准的位移度量之外，我们还通过轨迹碰撞率来衡量联合现实度。在高密度MADRAS基准测试和SDD上，VISTA达到了最先进的准确性和显著减少的碰撞数。在MADRAS上，它将强基线的平均碰撞率从2.14%降至0.03%，而在SDD上实现了零碰撞，并提高了ADE、FDE和minFDE。这些结果表明，VISTA生成了符合社会规范、目标意识和可解释的轨迹，使其适用于安全关键的自主系统。 

---
# Harnessing Bounded-Support Evolution Strategies for Policy Refinement 

**Title (ZH)**: 基于有界支持演化策略的策略精炼 

**Authors**: Ethan Hirschowitz, Fabio Ramos  

**Link**: [PDF](https://arxiv.org/pdf/2511.09923)  

**Abstract**: Improving competent robot policies with on-policy RL is often hampered by noisy, low-signal gradients. We revisit Evolution Strategies (ES) as a policy-gradient proxy and localize exploration with bounded, antithetic triangular perturbations, suitable for policy refinement. We propose Triangular-Distribution ES (TD-ES) which pairs bounded triangular noise with a centered-rank finite-difference estimator to deliver stable, parallelizable, gradient-free updates. In a two-stage pipeline -- PPO pretraining followed by TD-ES refinement -- this preserves early sample efficiency while enabling robust late-stage gains. Across a suite of robotic manipulation tasks, TD-ES raises success rates by 26.5% relative to PPO and greatly reduces variance, offering a simple, compute-light path to reliable refinement. 

**Abstract (ZH)**: 使用优先策略强化学习改进智能机器人策略常受制于噪声大、信号低的梯度。我们重新审视进化策略（ES）作为策略梯度近似，并使用有界反对称三角扰动局部化探索，适用于策略细化。我们提出三角分布进化策略（TD-ES），结合有界三角噪声与居中秩有限差分估计器，提供稳定、可并行、无梯度的更新。在两阶段流水线——PPO 预训练后跟随 TD-ES 精细调整——这种方法保持了早期样本效率的同时，使后期稳健收益成为可能。在一系列机器人操作任务中，TD-ES 的成功率相对 PPO 提高了 26.5%，大幅减少了方差，提供了一条简单、计算量小的可靠细化路径。 

---
# PALMS+: Modular Image-Based Floor Plan Localization Leveraging Depth Foundation Model 

**Title (ZH)**: PALMS+: 基于图像的楼层平面定位模块化方法利用深度基础模型 

**Authors**: Yunqian Cheng, Benjamin Princen, Roberto Manduchi  

**Link**: [PDF](https://arxiv.org/pdf/2511.09724)  

**Abstract**: Indoor localization in GPS-denied environments is crucial for applications like emergency response and assistive navigation. Vision-based methods such as PALMS enable infrastructure-free localization using only a floor plan and a stationary scan, but are limited by the short range of smartphone LiDAR and ambiguity in indoor layouts. We propose PALMS$+$, a modular, image-based system that addresses these challenges by reconstructing scale-aligned 3D point clouds from posed RGB images using a foundation monocular depth estimation model (Depth Pro), followed by geometric layout matching via convolution with the floor plan. PALMS$+$ outputs a posterior over the location and orientation, usable for direct or sequential localization. Evaluated on the Structured3D and a custom campus dataset consisting of 80 observations across four large campus buildings, PALMS$+$ outperforms PALMS and F3Loc in stationary localization accuracy -- without requiring any training. Furthermore, when integrated with a particle filter for sequential localization on 33 real-world trajectories, PALMS$+$ achieved lower localization errors compared to other methods, demonstrating robustness for camera-free tracking and its potential for infrastructure-free applications. Code and data are available at this https URL 

**Abstract (ZH)**: 在GPS受限环境下的室内定位对于应急响应和辅助导航等应用至关重要。基于视觉的方法如PALMS能够利用楼地板平面图和静态扫描实现无基础设施定位，但受智能手机LiDAR范围短和室内布局歧义的限制。我们提出PALMS$+$，一个模块化的基于图像系统，通过使用基础单目深度估计模型（Depth Pro）从布置的RGB图像中重建尺度对齐的3D点云，随后通过与楼地板平面图卷积实现几何布局匹配。PALMS$+$输出位置和朝向的后验概率，适用于直接或序列定位。在Structured3D和一个包含80次观测的自定义校园数据集上评估，PALMS$+$在静态定位准确性上优于PALMS和F3Loc，无需任何训练。此外，当与粒子滤波器结合进行序列定位时，在33条真实世界轨迹上，PALMS$+$的定位误差低于其他方法，展示了其无需摄像头的跟踪鲁棒性和无基础设施应用的潜力。代码和数据可在此处获取。 

---
