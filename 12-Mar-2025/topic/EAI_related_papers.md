# Generating Robot Constitutions & Benchmarks for Semantic Safety 

**Title (ZH)**: 生成机器人宪法与语义安全基准 

**Authors**: Pierre Sermanet, Anirudha Majumdar, Alex Irpan, Dmitry Kalashnikov, Vikas Sindhwani  

**Link**: [PDF](https://arxiv.org/pdf/2503.08663)  

**Abstract**: Until recently, robotics safety research was predominantly about collision avoidance and hazard reduction in the immediate vicinity of a robot. Since the advent of large vision and language models (VLMs), robots are now also capable of higher-level semantic scene understanding and natural language interactions with humans. Despite their known vulnerabilities (e.g. hallucinations or jail-breaking), VLMs are being handed control of robots capable of physical contact with the real world. This can lead to dangerous behaviors, making semantic safety for robots a matter of immediate concern. Our contributions in this paper are two fold: first, to address these emerging risks, we release the ASIMOV Benchmark, a large-scale and comprehensive collection of datasets for evaluating and improving semantic safety of foundation models serving as robot brains. Our data generation recipe is highly scalable: by leveraging text and image generation techniques, we generate undesirable situations from real-world visual scenes and human injury reports from hospitals. Secondly, we develop a framework to automatically generate robot constitutions from real-world data to steer a robot's behavior using Constitutional AI mechanisms. We propose a novel auto-amending process that is able to introduce nuances in written rules of behavior; this can lead to increased alignment with human preferences on behavior desirability and safety. We explore trade-offs between generality and specificity across a diverse set of constitutions of different lengths, and demonstrate that a robot is able to effectively reject unconstitutional actions. We measure a top alignment rate of 84.3% on the ASIMOV Benchmark using generated constitutions, outperforming no-constitution baselines and human-written constitutions. Data is available at this http URL 

**Abstract (ZH)**: 直到最近，机器人安全研究主要集中在避免碰撞和减少机器人周围区域的潜在危险。随着大型视觉和语言模型（VLMs）的出现，机器人现在也能够进行更高层次的语义场景理解，并与人类进行自然语言交互。尽管这些模型存在已知的漏洞（例如幻觉或逃逸），但它们现在正在控制能够与真实世界进行物理互动的机器人。这可能导致危险的行为，使机器人的语义安全性成为一个迫切需要关注的问题。本文的主要贡献有两个方面：首先，为了解决这些新兴的风险，我们发布了ASIMOV基准，这是一个大规模且全面的数据集集合，用于评估和提高作为机器人大脑的基础模型的语义安全性。我们的数据生成配方具有很高的可扩展性：通过利用文本和图像生成技术，我们从现实世界的视觉场景和医院的人身伤害报告中生成了不良情况。其次，我们开发了一个框架，可以从现实世界数据自动生成机器人的宪法，使用宪法AI机制引导机器人的行为。我们提出了一种新颖的自动修正过程，能够引入行为规则中的细微差异；这可以提高机器人的行为偏好和安全性的匹配度。我们探讨了不同长度宪法的一般性和特异性之间的权衡，并证明机器人能够有效拒绝不符合宪法的行为。使用生成的宪法在ASIMOV基准上的顶级对齐率为84.3%，优于无宪法基线和人类撰写的宪法。数据可通过以下链接获取。 

---
# Cross-Embodiment Robotic Manipulation Synthesis via Guided Demonstrations through CycleVAE and Human Behavior Transformer 

**Title (ZH)**: 通过CycleVAE和人类行为变换器引导的跨体态机器人操作合成 

**Authors**: Apan Dastider, Hao Fang, Mingjie Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.08622)  

**Abstract**: Cross-embodiment robotic manipulation synthesis for complicated tasks is challenging, partially due to the scarcity of paired cross-embodiment datasets and the impediment of designing intricate controllers. Inspired by robotic learning via guided human expert demonstration, we here propose a novel cross-embodiment robotic manipulation algorithm via CycleVAE and human behavior transformer. First, we utilize unsupervised CycleVAE together with a bidirectional subspace alignment algorithm to align latent motion sequences between cross-embodiments. Second, we propose a casual human behavior transformer design to learn the intrinsic motion dynamics of human expert demonstrations. During the test case, we leverage the proposed transformer for the human expert demonstration generation, which will be aligned using CycleVAE for the final human-robotic manipulation synthesis. We validated our proposed algorithm through extensive experiments using a dexterous robotic manipulator with the robotic hand. Our results successfully generate smooth trajectories across intricate tasks, outperforming prior learning-based robotic motion planning algorithms. These results have implications for performing unsupervised cross-embodiment alignment and future autonomous robotics design. Complete video demonstrations of our experiments can be found in this https URL. 

**Abstract (ZH)**: 基于循环VAE和人类行为变换器的跨体态机器人操作合成算法 

---
# EMMOE: A Comprehensive Benchmark for Embodied Mobile Manipulation in Open Environments 

**Title (ZH)**: EMMOE: 一种全面的开放环境体态移动操作基准 

**Authors**: Dongping Li, Tielong Cai, Tianci Tang, Wenhao Chai, Katherine Rose Driggs-Campbell, Gaoang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08604)  

**Abstract**: Developing autonomous home robots controlled by natural language has long been a pursuit of human. While advancements in large language models (LLMs) and embodied intelligence make this goal closer, several challenges persist: the lack of a unified benchmark for more complex robot tasks, limited evaluation methods and metrics, data incompatibility between LLMs and mobile manipulation trajectories. To address these issues, we introduce Embodied Mobile Manipulation in Open Environments (EMMOE), which requires agents to interpret user instructions and execute long-horizon everyday tasks in continuous space. EMMOE seamlessly integrates high-level and low-level embodied tasks into a unified framework, along with three new metrics for more diverse assessment. Additionally, we collect EMMOE-100, which features in various task attributes, detailed process annotations, re-plans after failures, and two sub-datasets for LLM training. Furthermore, we design HomieBot, a sophisticated agent system consists of LLM with Direct Preference Optimization (DPO), light weighted navigation and manipulation models, and multiple error detection mechanisms. Finally, we demonstrate HomieBot's performance and the evaluation of different models and policies. 

**Abstract (ZH)**: 基于自然语言控制的家庭自主机器人长期是人类的追求。尽管大规模语言模型（LLMs）和体态智能的进步使得这一目标更近了一步，但仍存在一些挑战：缺乏针对更复杂机器人任务的统一基准、有限的评估方法和指标、LLMs与移动操控轨迹之间的数据不兼容。为应对这些挑战，我们引入了开放环境中的体态移动操控（EMMOE），要求智能体解析用户指令并执行持续空间中的长期日常任务。EMMOE 将高层和低层体态任务无缝集成到一个统一框架中，并引入了三项新的评估指标。此外，我们收集了EMMOE-100数据集，该数据集涵盖了各种任务属性、详细的流程注解、失败后的重新规划，并为LLM训练提供了两个子数据集。我们还设计了HomieBot，这是一种包含直接偏好优化（DPO）的大规模语言模型、轻量级导航和操作模型以及多种错误检测机制的高级代理系统。最后，我们展示了HomieBot的性能，并对不同模型和策略进行了评估。 

---
# Proc4Gem: Foundation models for physical agency through procedural generation 

**Title (ZH)**: Proc4Gem: 基于程序生成的物理代理基础模型 

**Authors**: Yixin Lin, Jan Humplik, Sandy H. Huang, Leonard Hasenclever, Francesco Romano, Stefano Saliceti, Daniel Zheng, Jose Enrique Chen, Catarina Barros, Adrian Collister, Matt Young, Adil Dostmohamed, Ben Moran, Ken Caluwaerts, Marissa Giustina, Joss Moore, Kieran Connell, Francesco Nori, Nicolas Heess, Steven Bohez, Arunkumar Byravan  

**Link**: [PDF](https://arxiv.org/pdf/2503.08593)  

**Abstract**: In robot learning, it is common to either ignore the environment semantics, focusing on tasks like whole-body control which only require reasoning about robot-environment contacts, or conversely to ignore contact dynamics, focusing on grounding high-level movement in vision and language. In this work, we show that advances in generative modeling, photorealistic rendering, and procedural generation allow us to tackle tasks requiring both. By generating contact-rich trajectories with accurate physics in semantically-diverse simulations, we can distill behaviors into large multimodal models that directly transfer to the real world: a system we call Proc4Gem. Specifically, we show that a foundation model, Gemini, fine-tuned on only simulation data, can be instructed in language to control a quadruped robot to push an object with its body to unseen targets in unseen real-world environments. Our real-world results demonstrate the promise of using simulation to imbue foundation models with physical agency. Videos can be found at our website: this https URL 

**Abstract (ZH)**: 在机器人学习中，通常要么忽略环境语义，专注于仅需处理机器人-环境接触的整体现体控制任务，要么忽略接触动力学，专注于将高层次运动嵌入到视觉和语言中。在这项工作中，我们展示了生成模型的进步、真实感渲染和程序生成技术允许我们应对需要结合这两种处理的任务。通过在语义多样化的模拟中生成富含接触的真实物理轨迹，我们可以提炼出能直接转移到现实世界中的大规模多模态模型：我们称之为Proc4Gem的系统。具体而言，我们展示了仅在模拟数据上微调的基础模型Gemini，可以通过语言指令控制四足机器人将其身体用来推动物体到未见的真实环境中的目标。我们的现实世界结果证明了使用模拟来赋予基础模型物理自主性的潜力。更多信息和视频请访问我们的网站：this https URL 

---
# MoE-Loco: Mixture of Experts for Multitask Locomotion 

**Title (ZH)**: MoE-Loco: 专家混合的多任务运动控制 

**Authors**: Runhan Huang, Shaoting Zhu, Yilun Du, Hang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.08564)  

**Abstract**: We present MoE-Loco, a Mixture of Experts (MoE) framework for multitask locomotion for legged robots. Our method enables a single policy to handle diverse terrains, including bars, pits, stairs, slopes, and baffles, while supporting quadrupedal and bipedal gaits. Using MoE, we mitigate the gradient conflicts that typically arise in multitask reinforcement learning, improving both training efficiency and performance. Our experiments demonstrate that different experts naturally specialize in distinct locomotion behaviors, which can be leveraged for task migration and skill composition. We further validate our approach in both simulation and real-world deployment, showcasing its robustness and adaptability. 

**Abstract (ZH)**: MoE-Loco：一种用于腿足机器人多任务运动的专家混合框架 

---
# TLA: Tactile-Language-Action Model for Contact-Rich Manipulation 

**Title (ZH)**: TLA：触觉-语言-行动模型用于接触丰富的操作 

**Authors**: Peng Hao, Chaofan Zhang, Dingzhe Li, Xiaoge Cao, Xiaoshuai Hao, Shaowei Cui, Shuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08548)  

**Abstract**: Significant progress has been made in vision-language models. However, language-conditioned robotic manipulation for contact-rich tasks remains underexplored, particularly in terms of tactile sensing. To address this gap, we introduce the Tactile-Language-Action (TLA) model, which effectively processes sequential tactile feedback via cross-modal language grounding to enable robust policy generation in contact-intensive scenarios. In addition, we construct a comprehensive dataset that contains 24k pairs of tactile action instruction data, customized for fingertip peg-in-hole assembly, providing essential resources for TLA training and evaluation. Our results show that TLA significantly outperforms traditional imitation learning methods (e.g., diffusion policy) in terms of effective action generation and action accuracy, while demonstrating strong generalization capabilities by achieving over 85\% success rate on previously unseen assembly clearances and peg shapes. We publicly release all data and code in the hope of advancing research in language-conditioned tactile manipulation skill learning. Project website: this https URL 

**Abstract (ZH)**: 视觉-语言模型取得了显著进展。然而，面向接触丰富任务的语言条件机器人操作，尤其是在触觉感知方面，仍待深入探索。为填补这一空白，我们提出了触觉-语言-动作（TLA）模型，该模型通过跨模态语言grounding有效地处理顺序触觉反馈，以在接触密集场景中生成稳健的策略。此外，我们构建了一个包含24000对触觉动作指令数据的综合数据集，专门用于指尖孔装配，为TLA的训练与评估提供了重要资源。实验结果表明，TLA在有效动作生成和动作准确性方面显著优于传统的模仿学习方法（如扩散策略），并通过实现超过85%的装配成功率展示了强大的泛化能力，适用于未见过的装配间隙和钉子形状。我们公开发布了所有数据和代码，以推动语言条件触觉操作技能学习的研究。项目网站：这个 https URL。 

---
# Hybrid Deep Reinforcement Learning for Radio Tracer Localisation in Robotic-assisted Radioguided Surgery 

**Title (ZH)**: 混合深度强化学习在机器人辅助放射引导手术中射频示踪物定位中的应用 

**Authors**: Hanyi Zhang, Kaizhong Deng, Zhaoyang Jacopo Hu, Baoru Huang, Daniel S. Elson  

**Link**: [PDF](https://arxiv.org/pdf/2503.08492)  

**Abstract**: Radioguided surgery, such as sentinel lymph node biopsy, relies on the precise localization of radioactive targets by non-imaging gamma/beta detectors. Manual radioactive target detection based on visual display or audible indication of gamma level is highly dependent on the ability of the surgeon to track and interpret the spatial information. This paper presents a learning-based method to realize the autonomous radiotracer detection in robot-assisted surgeries by navigating the probe to the radioactive target. We proposed novel hybrid approach that combines deep reinforcement learning (DRL) with adaptive robotic scanning. The adaptive grid-based scanning could provide initial direction estimation while the DRL-based agent could efficiently navigate to the target utilising historical data. Simulation experiments demonstrate a 95% success rate, and improved efficiency and robustness compared to conventional techniques. Real-world evaluation on the da Vinci Research Kit (dVRK) further confirms the feasibility of the approach, achieving an 80% success rate in radiotracer detection. This method has the potential to enhance consistency, reduce operator dependency, and improve procedural accuracy in radioguided surgeries. 

**Abstract (ZH)**: 基于学习的机器人辅助手术中自主放射性标记物检测方法 

---
# PhysVLM: Enabling Visual Language Models to Understand Robotic Physical Reachability 

**Title (ZH)**: PhysVLM: 使视觉语言模型理解机器人物理可达性 

**Authors**: Weijie Zhou, Manli Tao, Chaoyang Zhao, Haiyun Guo, Honghui Dong, Ming Tang, Jinqiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08481)  

**Abstract**: Understanding the environment and a robot's physical reachability is crucial for task execution. While state-of-the-art vision-language models (VLMs) excel in environmental perception, they often generate inaccurate or impractical responses in embodied visual reasoning tasks due to a lack of understanding of robotic physical reachability. To address this issue, we propose a unified representation of physical reachability across diverse robots, i.e., Space-Physical Reachability Map (S-P Map), and PhysVLM, a vision-language model that integrates this reachability information into visual reasoning. Specifically, the S-P Map abstracts a robot's physical reachability into a generalized spatial representation, independent of specific robot configurations, allowing the model to focus on reachability features rather than robot-specific parameters. Subsequently, PhysVLM extends traditional VLM architectures by incorporating an additional feature encoder to process the S-P Map, enabling the model to reason about physical reachability without compromising its general vision-language capabilities. To train and evaluate PhysVLM, we constructed a large-scale multi-robot dataset, Phys100K, and a challenging benchmark, EQA-phys, which includes tasks for six different robots in both simulated and real-world environments. Experimental results demonstrate that PhysVLM outperforms existing models, achieving a 14\% improvement over GPT-4o on EQA-phys and surpassing advanced embodied VLMs such as RoboMamba and SpatialVLM on the RoboVQA-val and OpenEQA benchmarks. Additionally, the S-P Map shows strong compatibility with various VLMs, and its integration into GPT-4o-mini yields a 7.1\% performance improvement. 

**Abstract (ZH)**: 理解环境和机器人物理可达性对于任务执行至关重要。尽管最先进的视觉-语言模型（VLMs）在环境感知方面表现出色，但在涉及机器人物理可达性的体感视觉推理任务中，它们常常生成不准确或不现实的响应，这是因为缺乏对机器人物理可达性的理解。为了解决这一问题，我们提出了一种跨不同机器人的一致表示物理可达性的方法，即空间-物理可达性图（S-P Map），以及一种集成该可达性信息的视觉-语言模型PhysVLM。具体而言，S-P Map将机器人的物理可达性抽象为一种泛化的空间表示，独立于特定的机器人配置，使模型能够专注于可达性特征而非机器人特定参数。随后，PhysVLM通过引入额外的功能编码器来处理S-P Map，从而能够在不牺牲其一般视觉-语言能力的情况下推理解析物理可达性。为了训练和评估PhysVLM，我们构建了一个大规模的多机器人数据集Phys100K以及一个具有挑战性的基准EQA-phys，其中包括六个不同机器人在模拟和真实环境中的任务。实验结果表明，PhysVLM优于现有模型，在EQA-phys基准上比GPT-4o提高了14%的性能，并在RoboVQA-val和OpenEQA基准上超越了高级的体感VLM如RoboMamba和SpatialVLM。此外，S-P Map与多种VLM具有很强的兼容性，将其集成到GPT-4o-mini中可实现7.1%的性能提升。 

---
# Soft Actor-Critic-based Control Barrier Adaptation for Robust Autonomous Navigation in Unknown Environments 

**Title (ZH)**: 基于Soft Actor-Critic的控制障碍自适应方法以实现未知环境中的 robust 自主导航 

**Authors**: Nicholas Mohammad, Nicola Bezzo  

**Link**: [PDF](https://arxiv.org/pdf/2503.08479)  

**Abstract**: Motion planning failures during autonomous navigation often occur when safety constraints are either too conservative, leading to deadlocks, or too liberal, resulting in collisions. To improve robustness, a robot must dynamically adapt its safety constraints to ensure it reaches its goal while balancing safety and performance measures. To this end, we propose a Soft Actor-Critic (SAC)-based policy for adapting Control Barrier Function (CBF) constraint parameters at runtime, ensuring safe yet non-conservative motion. The proposed approach is designed for a general high-level motion planner, low-level controller, and target system model, and is trained in simulation only. Through extensive simulations and physical experiments, we demonstrate that our framework effectively adapts CBF constraints, enabling the robot to reach its final goal without compromising safety. 

**Abstract (ZH)**: 自主导航过程中运动规划失败常常发生在安全约束过于保守导致死锁或过于宽松导致碰撞的情况下。为提高鲁棒性，机器人必须动态调整其安全约束，以确保在平衡安全与性能的前提下达到目标。为此，我们提出了一种基于Soft Actor-Critic (SAC)的策略，用于在运行时调整控制障碍函数(CBF)约束参数，确保运动既安全又不保守。所提出的方法适用于一般的高层运动规划器、低层控制器和目标系统模型，并仅在仿真中进行训练。通过广泛的仿真和物理实验，我们证明了该框架有效适应CBF约束，使机器人能够在不牺牲安全性的前提下达到最终目标。 

---
# LLM-Pack: Intuitive Grocery Handling for Logistics Applications 

**Title (ZH)**: LLM-Pack：面向物流应用的直观 grocery 处理方法 

**Authors**: Yannik Blei, Michael Krawez, Tobias Jülg, Pierre Krack, Florian Walter, Wolfram Burgard  

**Link**: [PDF](https://arxiv.org/pdf/2503.08445)  

**Abstract**: Robotics and automation are increasingly influential in logistics but remain largely confined to traditional warehouses. In grocery retail, advancements such as cashier-less supermarkets exist, yet customers still manually pick and pack groceries. While there has been a substantial focus in robotics on the bin picking problem, the task of packing objects and groceries has remained largely untouched. However, packing grocery items in the right order is crucial for preventing product damage, e.g., heavy objects should not be placed on top of fragile ones. However, the exact criteria for the right packing order are hard to define, in particular given the huge variety of objects typically found in stores. In this paper, we introduce LLM-Pack, a novel approach for grocery packing. LLM-Pack leverages language and vision foundation models for identifying groceries and generating a packing sequence that mimics human packing strategy. LLM-Pack does not require dedicated training to handle new grocery items and its modularity allows easy upgrades of the underlying foundation models. We extensively evaluate our approach to demonstrate its performance. We will make the source code of LLMPack publicly available upon the publication of this manuscript. 

**Abstract (ZH)**: 机器人技术和自动化在物流中日益影响力增大，但仍主要局限于传统仓库。在超市零售中，虽然无收银员超市等进步已存在，但客户仍需手动挑选和包装商品。虽然在机器人领域针对箱内物件拾取的问题已有大量研究，但如何打包物件和商品的问题则被相对忽视。然而，正确顺序包装商品对于防止商品损坏至关重要，例如，重物不应放在易碎物品上。然而，针对正确包装顺序的确切标准难以定义，尤其是在面对商店中常见的众多不同种类的商品时。本文介绍了一种新的商超打包方法——LLM-Pack。LLM-Pack 利用语言和视觉基础模型来识别商品并生成模仿人类打包策略的打包序列。LLM-Pack 不需要专门的训练来处理新的商品类型，其模块化设计允许对基础模型进行简便升级。我们进行了广泛评估以展示其性能。本文发表后，我们将公开 LLMPack 的源代码。 

---
# Gait in Eight: Efficient On-Robot Learning for Omnidirectional Quadruped Locomotion 

**Title (ZH)**: 八足：高效机器人上学习的全向四足运动控制 

**Authors**: Nico Bohlinger, Jonathan Kinzel, Daniel Palenicek, Lukasz Antczak, Jan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2503.08375)  

**Abstract**: On-robot Reinforcement Learning is a promising approach to train embodiment-aware policies for legged robots. However, the computational constraints of real-time learning on robots pose a significant challenge. We present a framework for efficiently learning quadruped locomotion in just 8 minutes of raw real-time training utilizing the sample efficiency and minimal computational overhead of the new off-policy algorithm CrossQ. We investigate two control architectures: Predicting joint target positions for agile, high-speed locomotion and Central Pattern Generators for stable, natural gaits. While prior work focused on learning simple forward gaits, our framework extends on-robot learning to omnidirectional locomotion. We demonstrate the robustness of our approach in different indoor and outdoor environments. 

**Abstract (ZH)**: 基于机器人强化学习是一种有望训练腿部机器人感知体魄政策的方法。然而，机器人实时学习的计算约束构成了一个重大挑战。我们提出了一种框架，利用新型离策略算法CrossQ的样本高效性和最小计算开销，在仅仅8分钟的原始实时训练中高效学习四足行走。我们研究了两种控制架构：预测关节目标位置以实现敏捷的高速行走和中央模式生成器以实现稳定、自然的步伐。虽然先前的工作集中在学习简单的前向步伐上，我们的框架将机器人上的学习扩展到了全方位行走。我们展示了该方法在不同室内外环境中的鲁棒性。 

---
# DG16M: A Large-Scale Dataset for Dual-Arm Grasping with Force-Optimized Grasps 

**Title (ZH)**: DG16M：用于力优化 grasp 的大型双臂抓取数据集 

**Authors**: Md Faizal Karim, Mohammed Saad Hashmi, Shreya Bollimuntha, Mahesh Reddy Tapeti, Gaurav Singh, Nagamanikandan Govindan, K Madhava Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2503.08358)  

**Abstract**: Dual-arm robotic grasping is crucial for handling large objects that require stable and coordinated manipulation. While single-arm grasping has been extensively studied, datasets tailored for dual-arm settings remain scarce. We introduce a large-scale dataset of 16 million dual-arm grasps, evaluated under improved force-closure constraints. Additionally, we develop a benchmark dataset containing 300 objects with approximately 30,000 grasps, evaluated in a physics simulation environment, providing a better grasp quality assessment for dual-arm grasp synthesis methods. Finally, we demonstrate the effectiveness of our dataset by training a Dual-Arm Grasp Classifier network that outperforms the state-of-the-art methods by 15\%, achieving higher grasp success rates and improved generalization across objects. 

**Abstract (ZH)**: 双臂机器人抓取对于处理需要稳定协调操作的大对象至关重要。虽然单臂抓取已经被广泛研究，但适用于双臂环境的数据集仍然稀缺。我们引入了一个包含1600万组双臂抓取的大规模数据集，并在改进的力闭合约束下进行了评估。此外，我们还开发了一个包含300个物体、约3万个抓取的数据集，并在物理仿真环境中进行评估，为双臂抓取合成方法提供了更好的抓取质量评估。最终，我们通过训练一个双臂抓取分类网络展示了该数据集的有效性，该网络在性能上比现有的最佳方法高出15%，并在不同物体上的成功率和泛化能力方面表现更优。 

---
# LiPS: Large-Scale Humanoid Robot Reinforcement Learning with Parallel-Series Structures 

**Title (ZH)**: LiPS: 大规模 humanoid 机器人强化学习的并行级联结构方法 

**Authors**: Qiang Zhang, Gang Han, Jingkai Sun, Wen Zhao, Jiahang Cao, Jiaxu Wang, Hao Cheng, Lingfeng Zhang, Yijie Guo, Renjing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.08349)  

**Abstract**: In recent years, research on humanoid robots has garnered significant attention, particularly in reinforcement learning based control algorithms, which have achieved major breakthroughs. Compared to traditional model-based control algorithms, reinforcement learning based algorithms demonstrate substantial advantages in handling complex tasks. Leveraging the large-scale parallel computing capabilities of GPUs, contemporary humanoid robots can undergo extensive parallel training in simulated environments. A physical simulation platform capable of large-scale parallel training is crucial for the development of humanoid robots. As one of the most complex robot forms, humanoid robots typically possess intricate mechanical structures, encompassing numerous series and parallel mechanisms. However, many reinforcement learning based humanoid robot control algorithms currently employ open-loop topologies during training, deferring the conversion to series-parallel structures until the sim2real phase. This approach is primarily due to the limitations of physics engines, as current GPU-based physics engines often only support open-loop topologies or have limited capabilities in simulating multi-rigid-body closed-loop topologies. For enabling reinforcement learning-based humanoid robot control algorithms to train in large-scale parallel environments, we propose a novel training method LiPS. By incorporating multi-rigid-body dynamics modeling in the simulation environment, we significantly reduce the sim2real gap and the difficulty of converting to parallel structures during model deployment, thereby robustly supporting large-scale reinforcement learning for humanoid robots. 

**Abstract (ZH)**: 基于强化学习的人形机器人控制算法研究与LiPS训练方法 

---
# Trinity: A Modular Humanoid Robot AI System 

**Title (ZH)**: 三位一体：一种模块化人形机器人AI系统 

**Authors**: Jingkai Sun, Qiang Zhang, Gang Han, Wen Zhao, Zhe Yong, Yan He, Jiaxu Wang, Jiahang Cao, Yijie Guo, Renjing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.08338)  

**Abstract**: In recent years, research on humanoid robots has garnered increasing attention. With breakthroughs in various types of artificial intelligence algorithms, embodied intelligence, exemplified by humanoid robots, has been highly anticipated. The advancements in reinforcement learning (RL) algorithms have significantly improved the motion control and generalization capabilities of humanoid robots. Simultaneously, the groundbreaking progress in large language models (LLM) and visual language models (VLM) has brought more possibilities and imagination to humanoid robots. LLM enables humanoid robots to understand complex tasks from language instructions and perform long-term task planning, while VLM greatly enhances the robots' understanding and interaction with their environment. This paper introduces \textcolor{magenta}{Trinity}, a novel AI system for humanoid robots that integrates RL, LLM, and VLM. By combining these technologies, Trinity enables efficient control of humanoid robots in complex environments. This innovative approach not only enhances the capabilities but also opens new avenues for future research and applications of humanoid robotics. 

**Abstract (ZH)**: 近年来，关于人形机器人的研究引起了越来越多的关注。随着各种类型的人工智能算法的突破，以人形机器人为代表的体现智能受到了高度期待。强化学习（RL）算法的进步显著提高了人形机器人的运动控制能力和泛化能力。同时，大规模语言模型（LLM）和视觉语言模型（VLM）的突破性进展为人形机器人带来了更多的可能性和想象空间。大规模语言模型使人类形机器人能够从语言指令中理解复杂任务并进行长期任务规划，而视觉语言模型极大地增强了机器人对环境的理解和交互能力。本文介绍了一种新颖的人形机器人AI系统Trinity，该系统结合了RL、LLM和VLM。通过结合这些技术，Trinity能够有效地控制人形机器人在复杂环境中的行为。这一创新方法不仅增强了功能，还为未来人形机器人研究和应用开辟了新的途径。 

---
# KiteRunner: Language-Driven Cooperative Local-Global Navigation Policy with UAV Mapping in Outdoor Environments 

**Title (ZH)**: 风筝巡飞者：基于语言驱动的户外环境空中无人机局部-全局导航策略与_mapping_协作 

**Authors**: Shibo Huang, Chenfan Shi, Jian Yang, Hanlin Dong, Jinpeng Mi, Ke Li, Jianfeng Zhang, Miao Ding, Peidong Liang, Xiong You, Xian Wei  

**Link**: [PDF](https://arxiv.org/pdf/2503.08330)  

**Abstract**: Autonomous navigation in open-world outdoor environments faces challenges in integrating dynamic conditions, long-distance spatial reasoning, and semantic understanding. Traditional methods struggle to balance local planning, global planning, and semantic task execution, while existing large language models (LLMs) enhance semantic comprehension but lack spatial reasoning capabilities. Although diffusion models excel in local optimization, they fall short in large-scale long-distance navigation. To address these gaps, this paper proposes KiteRunner, a language-driven cooperative local-global navigation strategy that combines UAV orthophoto-based global planning with diffusion model-driven local path generation for long-distance navigation in open-world scenarios. Our method innovatively leverages real-time UAV orthophotography to construct a global probability map, providing traversability guidance for the local planner, while integrating large models like CLIP and GPT to interpret natural language instructions. Experiments demonstrate that KiteRunner achieves 5.6% and 12.8% improvements in path efficiency over state-of-the-art methods in structured and unstructured environments, respectively, with significant reductions in human interventions and execution time. 

**Abstract (ZH)**: 自主导航在开放世界户外环境中的挑战在于集成动态条件、长距离空间推理和语义理解。传统方法难以平衡局部规划、全局规划和语义任务执行，而现有的大型语言模型（LLMs）增强了语义理解但缺乏空间推理能力。尽管扩散模型在局部优化方面表现出色，但在大规模长距离导航方面仍有不足。为了解决这些差距，本文提出KiteRunner，一种基于语言驱动的合作局部-全局导航策略，结合了基于无人机正射摄影的全局规划和基于扩散模型的局部路径生成，适用于开放世界场景中的长距离导航。我们的方法创新性地利用实时无人机正射摄影构建全局概率图，为局部规划器提供通行性指导，同时整合像CLIP和GPT这样的大型模型来解释自然语言指令。实验结果表明，与最先进的方法相比，KiteRunner在结构化和非结构化环境中分别实现了5.6%和12.8%的路径效率提升，显著减少了人工干预和执行时间。 

---
# Reasoning in visual navigation of end-to-end trained agents: a dynamical systems approach 

**Title (ZH)**: 基于动力系统方法的端到端训练代理视觉导航推理 

**Authors**: Steeven Janny, Hervé Poirier, Leonid Antsfeld, Guillaume Bono, Gianluca Monaci, Boris Chidlovskii, Francesco Giuliari, Alessio Del Bue, Christian Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2503.08306)  

**Abstract**: Progress in Embodied AI has made it possible for end-to-end-trained agents to navigate in photo-realistic environments with high-level reasoning and zero-shot or language-conditioned behavior, but benchmarks are still dominated by simulation. In this work, we focus on the fine-grained behavior of fast-moving real robots and present a large-scale experimental study involving \numepisodes{} navigation episodes in a real environment with a physical robot, where we analyze the type of reasoning emerging from end-to-end training. In particular, we study the presence of realistic dynamics which the agent learned for open-loop forecasting, and their interplay with sensing. We analyze the way the agent uses latent memory to hold elements of the scene structure and information gathered during exploration. We probe the planning capabilities of the agent, and find in its memory evidence for somewhat precise plans over a limited horizon. Furthermore, we show in a post-hoc analysis that the value function learned by the agent relates to long-term planning. Put together, our experiments paint a new picture on how using tools from computer vision and sequential decision making have led to new capabilities in robotics and control. An interactive tool is available at this http URL. 

**Abstract (ZH)**: 基于感官的真实环境中的实时机器人精细行为研究：端到端训练代理在实际环境中的高层推理与感知交互分析 

---
# General-Purpose Aerial Intelligent Agents Empowered by Large Language Models 

**Title (ZH)**: 大型语言模型赋能的通用 aerial 智能代理 

**Authors**: Ji Zhao, Xiao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.08302)  

**Abstract**: The emergence of large language models (LLMs) opens new frontiers for unmanned aerial vehicle (UAVs), yet existing systems remain confined to predefined tasks due to hardware-software co-design challenges. This paper presents the first aerial intelligent agent capable of open-world task execution through tight integration of LLM-based reasoning and robotic autonomy. Our hardware-software co-designed system addresses two fundamental limitations: (1) Onboard LLM operation via an edge-optimized computing platform, achieving 5-6 tokens/sec inference for 14B-parameter models at 220W peak power; (2) A bidirectional cognitive architecture that synergizes slow deliberative planning (LLM task planning) with fast reactive control (state estimation, mapping, obstacle avoidance, and motion planning). Validated through preliminary results using our prototype, the system demonstrates reliable task planning and scene understanding in communication-constrained environments, such as sugarcane monitoring, power grid inspection, mine tunnel exploration, and biological observation applications. This work establishes a novel framework for embodied aerial artificial intelligence, bridging the gap between task planning and robotic autonomy in open environments. 

**Abstract (ZH)**: 大语言模型的出现为无人驾驶航空器开启了新的前沿领域，但现有系统仍受限于硬件软件协同设计的挑战。本文提出了一种通过将基于大语言模型的推理与机器人自主性紧密结合而实现的首个适用于开放世界任务执行的空中智能代理。我们的硬件软件协同设计系统解决了两个根本限制：（1）通过边缘优化计算平台实现机载大语言模型运行，以220W的峰值功率实现每秒5-6个词的推理，适用于14B参数模型；（2）一种双向认知架构，将慢速的反思性计划（基于大语言模型的任务规划）与快速的反应性控制（状态估计、制图、避障和路径规划）协同起来。通过我们原型的初步结果进行验证，该系统在通信受限环境中展示了可靠的任务规划和场景理解能力，适用于甘蔗监控、电力网络巡检、矿井隧道勘探和生物观察等应用。本文建立了一种新的框架，为开放环境中的任务规划与机器人自主性的结合提供了新的途径。 

---
# Distillation-PPO: A Novel Two-Stage Reinforcement Learning Framework for Humanoid Robot Perceptive Locomotion 

**Title (ZH)**: Distillation-PPO：一种新型双阶段强化学习框架，用于类人机器人感知行走 

**Authors**: Qiang Zhang, Gang Han, Jingkai Sun, Wen Zhao, Chenghao Sun, Jiahang Cao, Jiaxu Wang, Yijie Guo, Renjing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.08299)  

**Abstract**: In recent years, humanoid robots have garnered significant attention from both academia and industry due to their high adaptability to environments and human-like characteristics. With the rapid advancement of reinforcement learning, substantial progress has been made in the walking control of humanoid robots. However, existing methods still face challenges when dealing with complex environments and irregular terrains. In the field of perceptive locomotion, existing approaches are generally divided into two-stage methods and end-to-end methods. Two-stage methods first train a teacher policy in a simulated environment and then use distillation techniques, such as DAgger, to transfer the privileged information learned as latent features or actions to the student policy. End-to-end methods, on the other hand, forgo the learning of privileged information and directly learn policies from a partially observable Markov decision process (POMDP) through reinforcement learning. However, due to the lack of supervision from a teacher policy, end-to-end methods often face difficulties in training and exhibit unstable performance in real-world applications. This paper proposes an innovative two-stage perceptive locomotion framework that combines the advantages of teacher policies learned in a fully observable Markov decision process (MDP) to regularize and supervise the student policy. At the same time, it leverages the characteristics of reinforcement learning to ensure that the student policy can continue to learn in a POMDP, thereby enhancing the model's upper bound. Our experimental results demonstrate that our two-stage training framework achieves higher training efficiency and stability in simulated environments, while also exhibiting better robustness and generalization capabilities in real-world applications. 

**Abstract (ZH)**: 近年来，由于人形机器人在环境适应性和类人特征方面的高度适应性，它们在学术界和工业界获得了广泛关注。随着强化学习的迅速发展，人在环控制能力在人形机器人行走控制方面取得了显著进展。然而，现有方法在处理复杂环境和不规则地形时仍面临挑战。在感知行走领域，现有方法通常被分成两阶段方法和端到端方法。两阶段方法首先在仿真环境中训练一个教师策略，然后使用蒸馏技术，如DAgger，将从潜在特征或行动中学到的优势信息转移到学生策略。而端到端方法则省去了学习优势信息的步骤，直接通过强化学习从部分可观测马尔可夫决策过程（POMDP）中学习策略。然而，由于缺乏教师策略的监督，端到端方法往往在训练中遇到困难，并且在实际应用中表现出不稳定的性能。本文提出了一种创新的两阶段感知行走框架，该框架结合了在完全可观测马尔可夫决策过程（MDP）中学习的教师策略的优势，以正则化和监督学生策略。同时，该框架利用强化学习的特性，确保学生策略可以在POMDP中继续学习，从而提高模型的上界。实验结果表明，我们的两阶段训练框架在仿真环境中实现了更高的训练效率和稳定性，同时也表现出更好的鲁棒性和泛化能力。 

---
# Multitask Reinforcement Learning for Quadcopter Attitude Stabilization and Tracking using Graph Policy 

**Title (ZH)**: 基于图策略的四旋翼姿态稳定与跟踪多任务强化学习 

**Authors**: Yu Tang Liu, Afonso Vale, Aamir Ahmad, Rodrigo Ventura, Meysam Basiri  

**Link**: [PDF](https://arxiv.org/pdf/2503.08259)  

**Abstract**: Quadcopter attitude control involves two tasks: smooth attitude tracking and aggressive stabilization from arbitrary states. Although both can be formulated as tracking problems, their distinct state spaces and control strategies complicate a unified reward function. We propose a multitask deep reinforcement learning framework that leverages parallel simulation with IsaacGym and a Graph Convolutional Network (GCN) policy to address both tasks effectively. Our multitask Soft Actor-Critic (SAC) approach achieves faster, more reliable learning and higher sample efficiency than single-task methods. We validate its real-world applicability by deploying the learned policy - a compact two-layer network with 24 neurons per layer - on a Pixhawk flight controller, achieving 400 Hz control without extra computational resources. We provide our code at this https URL\_UAV/. 

**Abstract (ZH)**: 四旋翼飞行器姿态控制涉及两任务：平滑的姿态跟踪和从任意状态的激进稳定化。尽管两者都可以形式化为跟踪问题，但其不同的状态空间和控制策略使统一的奖励函数变得复杂。我们提出了一种多任务深度强化学习框架，该框架利用IsaacGym并行模拟和图卷积网络（GCN）策略有效地解决这两个任务。我们的多任务Soft Actor-Critic（SAC）方法在样本效率和学习速度上优于单任务方法。我们通过将学习到的策略（一个包含两层，每层24个神经元的紧凑网络）部署到Pixhawk飞行控制器上，在不额外占用计算资源的情况下实现了400 Hz控制，验证了其实用性。代码发布在this <https://UAV/> URL\_UAV/。 

---
# Investigating the Effectiveness of a Socratic Chain-of-Thoughts Reasoning Method for Task Planning in Robotics, A Case Study 

**Title (ZH)**: 探究苏格拉底式连锁思维推理方法在机器人任务规划中的有效性：一个案例研究 

**Authors**: Veronica Bot, Zheyuan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.08174)  

**Abstract**: Large language models (LLMs) have demonstrated unprecedented capability in reasoning with natural language. Coupled with this development is the emergence of embodied AI in robotics. Despite showing promise for verbal and written reasoning tasks, it remains unknown whether LLMs are capable of navigating complex spatial tasks with physical actions in the real world. To this end, it is of interest to investigate applying LLMs to robotics in zero-shot learning scenarios, and in the absence of fine-tuning - a feat which could significantly improve human-robot interaction, alleviate compute cost, and eliminate low-level programming tasks associated with robot tasks.
To explore this question, we apply GPT-4(Omni) with a simulated Tiago robot in Webots engine for an object search task. We evaluate the effectiveness of three reasoning strategies based on Chain-of-Thought (CoT) sub-task list generation with the Socratic method (SocraCoT) (in order of increasing rigor): (1) Non-CoT/Non-SocraCoT, (2) CoT only, and (3) SocraCoT. Performance was measured in terms of the proportion of tasks successfully completed and execution time (N = 20). Our preliminary results show that when combined with chain-of-thought reasoning, the Socratic method can be used for code generation for robotic tasks that require spatial awareness. In extension of this finding, we propose EVINCE-LoC; a modified EVINCE method that could further enhance performance in highly complex and or dynamic testing scenarios. 

**Abstract (ZH)**: 大规模语言模型在自然语言推理方面展示了前所未有的能力。伴随着这一发展，机器人领域的嵌入式AI也出现了。虽然大规模语言模型在语言和书面推理任务中显示出前景，但在现实世界中是否能够通过物理动作导航复杂的空间任务仍不清楚。为此，有必要研究在零样本学习场景中将大规模语言模型应用于机器人技术的可能性，且无需微调——这一成就将显著改善人机交互，降低计算成本，并消除与机器人任务相关的低级编程任务。

为了探索这一问题，我们使用Webots引擎中的模拟Tiago机器人和GPT-4（全知）进行物体搜索任务。基于链式思维（CoT）子任务列表生成和苏格拉底方法（SocraCoT）的逐步严谨性，我们评估了三种推理策略的有效性：（1）非CoT/非SocraCoT，（2）仅CoT，（3）SocraCoT。性能通过成功完成任务的比例和执行时间（N = 20）进行衡量。初步结果显示，将链式思维推理与苏格拉底方法结合使用，可以用于生成需要空间意识的机器人任务的代码。在此基础上，我们提出了一种改进的EVINCE-LoC方法，可以在高度复杂或动态测试场景中进一步提高性能。 

---
# LATMOS: Latent Automaton Task Model from Observation Sequences 

**Title (ZH)**: LATMOS: 从观测序列推导的潜在自动机任务模型 

**Authors**: Weixiao Zhan, Qiyue Dong, Eduardo Sebastián, Nikolay Atanasov  

**Link**: [PDF](https://arxiv.org/pdf/2503.08090)  

**Abstract**: Robot task planning from high-level instructions is an important step towards deploying fully autonomous robot systems in the service sector. Three key aspects of robot task planning present challenges yet to be resolved simultaneously, namely, (i) factorization of complex tasks specifications into simpler executable subtasks, (ii) understanding of the current task state from raw observations, and (iii) planning and verification of task executions. To address these challenges, we propose LATMOS, an automata-inspired task model that, given observations from correct task executions, is able to factorize the task, while supporting verification and planning operations. LATMOS combines an observation encoder to extract the features from potentially high-dimensional observations with automata theory to learn a sequential model that encapsulates an automaton with symbols in the latent feature space. We conduct extensive evaluations in three task model learning setups: (i) abstract tasks described by logical formulas, (ii) real-world human tasks described by videos and natural language prompts and (iii) a robot task described by image and state observations. The results demonstrate the improved plan generation and verification capabilities of LATMOS across observation modalities and tasks. 

**Abstract (ZH)**: 基于高层指令的机器人任务规划是部署服务领域完全自主机器人系统的重要步骤。为了应对这一挑战，我们提出了一种受自动机启发的任务模型LATMOS，该模型能够根据正确的任务执行观察结果分解任务，同时支持验证和规划操作。LATMOS结合了一种观测编码器从潜在高维度观测中提取特征，并利用自动机理论学习一个在潜空间符号中封装自动机的序列模型。我们在三种任务模型学习设置中进行了广泛的评估：（i）由逻辑公式描述的抽象任务，（ii）由视频和自然语言提示描述的现实世界人类任务，以及（iii）由图像和状态观测描述的机器人任务。结果表明，LATMOS在不同观测模式和任务中的改进计划生成和验证能力。 

---
# Instruction-Augmented Long-Horizon Planning: Embedding Grounding Mechanisms in Embodied Mobile Manipulation 

**Title (ZH)**: 基于指令增强的长时规划：嵌入式实体移动 manipulative 机器人的 grounding 机制 

**Authors**: Fangyuan Wang, Shipeng Lyu, Peng Zhou, Anqing Duan, Guodong Guo, David Navarro-Alarcon  

**Link**: [PDF](https://arxiv.org/pdf/2503.08084)  

**Abstract**: Enabling humanoid robots to perform long-horizon mobile manipulation planning in real-world environments based on embodied perception and comprehension abilities has been a longstanding challenge. With the recent rise of large language models (LLMs), there has been a notable increase in the development of LLM-based planners. These approaches either utilize human-provided textual representations of the real world or heavily depend on prompt engineering to extract such representations, lacking the capability to quantitatively understand the environment, such as determining the feasibility of manipulating objects. To address these limitations, we present the Instruction-Augmented Long-Horizon Planning (IALP) system, a novel framework that employs LLMs to generate feasible and optimal actions based on real-time sensor feedback, including grounded knowledge of the environment, in a closed-loop interaction. Distinct from prior works, our approach augments user instructions into PDDL problems by leveraging both the abstract reasoning capabilities of LLMs and grounding mechanisms. By conducting various real-world long-horizon tasks, each consisting of seven distinct manipulatory skills, our results demonstrate that the IALP system can efficiently solve these tasks with an average success rate exceeding 80%. Our proposed method can operate as a high-level planner, equipping robots with substantial autonomy in unstructured environments through the utilization of multi-modal sensor inputs. 

**Abstract (ZH)**: 基于嵌入式感知与理解能力的大型语言模型驱动的人形机器人长期移动操作规划实现 

---
# MoRE: Unlocking Scalability in Reinforcement Learning for Quadruped Vision-Language-Action Models 

**Title (ZH)**: MoRE: 解锁 quadruped 视听行动模型在强化学习中的可扩展性 

**Authors**: Han Zhao, Wenxuan Song, Donglin Wang, Xinyang Tong, Pengxiang Ding, Xuelian Cheng, Zongyuan Ge  

**Link**: [PDF](https://arxiv.org/pdf/2503.08007)  

**Abstract**: Developing versatile quadruped robots that can smoothly perform various actions and tasks in real-world environments remains a significant challenge. This paper introduces a novel vision-language-action (VLA) model, mixture of robotic experts (MoRE), for quadruped robots that aim to introduce reinforcement learning (RL) for fine-tuning large-scale VLA models with a large amount of mixed-quality data. MoRE integrates multiple low-rank adaptation modules as distinct experts within a dense multi-modal large language model (MLLM), forming a sparse-activated mixture-of-experts model. This design enables the model to effectively adapt to a wide array of downstream tasks. Moreover, we employ a reinforcement learning-based training objective to train our model as a Q-function after deeply exploring the structural properties of our tasks. Effective learning from automatically collected mixed-quality data enhances data efficiency and model performance. Extensive experiments demonstrate that MoRE outperforms all baselines across six different skills and exhibits superior generalization capabilities in out-of-distribution scenarios. We further validate our method in real-world scenarios, confirming the practicality of our approach and laying a solid foundation for future research on multi-task learning in quadruped robots. 

**Abstract (ZH)**: 基于视觉-语言-动作的混合机器人专家模型MoRE：一种适用于 quadruped 机器人的 reinforcement learning 方法 

---
# QLIO: Quantized LiDAR-Inertial Odometry 

**Title (ZH)**: QLIO: 量化LiDAR-惯性里程计 

**Authors**: Boyang Lou, Shenghai Yuan, Jianfei Yang, Wenju Su, Yingjian Zhang, Enwen Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.07949)  

**Abstract**: LiDAR-Inertial Odometry (LIO) is widely used for autonomous navigation, but its deployment on Size, Weight, and Power (SWaP)-constrained platforms remains challenging due to the computational cost of processing dense point clouds. Conventional LIO frameworks rely on a single onboard processor, leading to computational bottlenecks and high memory demands, making real-time execution difficult on embedded systems. To address this, we propose QLIO, a multi-processor distributed quantized LIO framework that reduces computational load and bandwidth consumption while maintaining localization accuracy. QLIO introduces a quantized state estimation pipeline, where a co-processor pre-processes LiDAR measurements, compressing point-to-plane residuals before transmitting only essential features to the host processor. Additionally, an rQ-vector-based adaptive resampling strategy intelligently selects and compresses key observations, further reducing computational redundancy. Real-world evaluations demonstrate that QLIO achieves a 14.1% reduction in per-observation residual data while preserving localization accuracy. Furthermore, we release an open-source implementation to facilitate further research and real-world deployment. These results establish QLIO as an efficient and scalable solution for real-time autonomous systems operating under computational and bandwidth constraints. 

**Abstract (ZH)**: 基于多处理器分布式量化算法的LiDAR-惯性里程计（QLIO） 

---
# Learning Gentle Grasping Using Vision, Sound, and Touch 

**Title (ZH)**: 使用视觉、声音和触觉学习温和抓取 

**Authors**: Ken Nakahara, Roberto Calandra  

**Link**: [PDF](https://arxiv.org/pdf/2503.07926)  

**Abstract**: In our daily life, we often encounter objects that are fragile and can be damaged by excessive grasping force, such as fruits. For these objects, it is paramount to grasp gently -- not using the maximum amount of force possible, but rather the minimum amount of force necessary. This paper proposes using visual, tactile, and auditory signals to learn to grasp and regrasp objects stably and gently. Specifically, we use audio signals as an indicator of gentleness during the grasping, and then train end-to-end an action-conditional model from raw visuo-tactile inputs that predicts both the stability and the gentleness of future grasping candidates, thus allowing the selection and execution of the most promising action. Experimental results on a multi-fingered hand over 1,500 grasping trials demonstrated that our model is useful for gentle grasping by validating the predictive performance (3.27\% higher accuracy than the vision-only variant) and providing interpretations of their behavior. Finally, real-world experiments confirmed that the grasping performance with the trained multi-modal model outperformed other baselines (17\% higher rate for stable and gentle grasps than vision-only). Our approach requires neither tactile sensor calibration nor analytical force modeling, drastically reducing the engineering effort to grasp fragile objects. Dataset and videos are available at this https URL. 

**Abstract (ZH)**: 使用视觉、触觉和听觉信号学习稳定而轻柔地抓取和再抓取物体的方法 

---
# Intelligent Framework for Human-Robot Collaboration: Safety, Dynamic Ergonomics, and Adaptive Decision-Making 

**Title (ZH)**: 智能人机协作框架：安全、动态人机工程学与适应性决策-making 

**Authors**: Francesco Iodice, Elena De Momi, Arash Ajoudani  

**Link**: [PDF](https://arxiv.org/pdf/2503.07901)  

**Abstract**: The integration of collaborative robots into industrial environments has improved productivity, but has also highlighted significant challenges related to operator safety and ergonomics. This paper proposes an innovative framework that integrates advanced visual perception technologies, real-time ergonomic monitoring, and Behaviour Tree (BT)-based adaptive decision-making. Unlike traditional methods, which often operate in isolation or statically, our approach combines deep learning models (YOLO11 and SlowOnly), advanced tracking (Unscented Kalman Filter) and dynamic ergonomic assessments (OWAS), offering a modular, scalable and adaptive system. Experimental results show that the framework outperforms previous methods in several aspects: accuracy in detecting postures and actions, adaptivity in managing human-robot interactions, and ability to reduce ergonomic risk through timely robotic interventions. In particular, the visual perception module showed superiority over YOLOv9 and YOLOv8, while real-time ergonomic monitoring eliminated the limitations of static analysis. Adaptive role management, made possible by the Behaviour Tree, provided greater responsiveness than rule-based systems, making the framework suitable for complex industrial scenarios. Our system demonstrated a 92.5\% accuracy in grasping intention recognition and successfully classified ergonomic risks with real-time responsiveness (average latency of 0.57 seconds), enabling timely robotic 

**Abstract (ZH)**: 协作机器人集成到工业环境中的整合提高了生产效率，但也凸显了操作员安全和人体工程学方面的重要挑战。本文提出了一种创新框架，该框架结合了先进的视觉感知技术、实时人体工程学监测和基于行为树（BT）的自适应决策。与通常孤立或静态运作的传统方法不同，本方法将深入学习模型（YOLO11和SlowOnly）、高级跟踪（无迹卡尔曼滤波器）和动态人体工程学评估（OWAS）相结合，提供了一个模块化、可扩展和自适应的系统。实验结果表明，该框架在多个方面优于以往方法：姿势和动作检测的准确性、管理人机交互的自适应性以及通过及时的机器人干预降低人体工程学风险的能力。特别是视觉感知模块优于YOLOv9和YOLOv8，实时人体工程学监测克服了静态分析的限制。行为树支持的自适应角色管理提供比基于规则的系统更高的响应性，使框架适合复杂工业场景。我们的系统表现出92.5%的抓取意图识别准确率，并能够通过实时响应（平均延迟0.57秒）成功分类人体工程学风险，实现及时的机器人干预。 

---
# RoboCopilot: Human-in-the-loop Interactive Imitation Learning for Robot Manipulation 

**Title (ZH)**: RoboCopilot: 人工在环的交互式模仿学习在机器人 manipulation 中的应用 

**Authors**: Philipp Wu, Yide Shentu, Qiayuan Liao, Ding Jin, Menglong Guo, Koushil Sreenath, Xingyu Lin, Pieter Abbeel  

**Link**: [PDF](https://arxiv.org/pdf/2503.07771)  

**Abstract**: Learning from human demonstration is an effective approach for learning complex manipulation skills. However, existing approaches heavily focus on learning from passive human demonstration data for its simplicity in data collection. Interactive human teaching has appealing theoretical and practical properties, but they are not well supported by existing human-robot interfaces. This paper proposes a novel system that enables seamless control switching between human and an autonomous policy for bi-manual manipulation tasks, enabling more efficient learning of new tasks. This is achieved through a compliant, bilateral teleoperation system. Through simulation and hardware experiments, we demonstrate the value of our system in an interactive human teaching for learning complex bi-manual manipulation skills. 

**Abstract (ZH)**: 从人类示范学习是一种学习复杂操作技能的有效方法。然而，现有的方法主要关注从被动的人类示范数据中学习，因为这种数据收集方式简单。交互式的人机教学具有诱人的理论和实用特性，但现有的人机接口并未充分支持这一点。本文提出了一种新型系统，该系统能够在双手操作任务中无缝切换人为控制和自主策略控制，从而更有效地学习新任务。通过这种合规的双边遥操作系统得以实现。通过仿真和硬件实验，我们证明了该系统在交互式人机教学中学习复杂双手操作技能中的价值。 

---
# V-Max: Making RL practical for Autonomous Driving 

**Title (ZH)**: V-Max：使强化学习在自动驾驶中变得更加实用 

**Authors**: Valentin Charraut, Thomas Tournaire, Waël Doulazmi, Thibault Buhet  

**Link**: [PDF](https://arxiv.org/pdf/2503.08388)  

**Abstract**: Learning-based decision-making has the potential to enable generalizable Autonomous Driving (AD) policies, reducing the engineering overhead of rule-based approaches. Imitation Learning (IL) remains the dominant paradigm, benefiting from large-scale human demonstration datasets, but it suffers from inherent limitations such as distribution shift and imitation gaps. Reinforcement Learning (RL) presents a promising alternative, yet its adoption in AD remains limited due to the lack of standardized and efficient research frameworks. To this end, we introduce V-Max, an open research framework providing all the necessary tools to make RL practical for AD. V-Max is built on Waymax, a hardware-accelerated AD simulator designed for large-scale experimentation. We extend it using ScenarioNet's approach, enabling the fast simulation of diverse AD datasets. V-Max integrates a set of observation and reward functions, transformer-based encoders, and training pipelines. Additionally, it includes adversarial evaluation settings and an extensive set of evaluation metrics. Through a large-scale benchmark, we analyze how network architectures, observation functions, training data, and reward shaping impact RL performance. 

**Abstract (ZH)**: 基于学习的决策制定有望使自主驾驶（AD）策略具备通用性，减少基于规则的方法的工程开销。模拟学习（IL）仍然是主导范式，得益于大规模的人类示范数据集，但面临分布偏移和模仿差距等固有的局限性。强化学习（RL）提供了一种有前景的替代方案，但由于缺乏标准化和高效的研兖框架，其在AD中的应用仍受限。为此，我们介绍了V-Max，一个开放的研究框架，提供所有必要的工具使RL在AD中实用。V-Max基于Waymax构建，Waymax是一个硬件加速的AD仿真器，适用于大规模实验。我们使用ScenarioNet的方法对其进行扩展，使其能够快速模拟多样化的AD数据集。V-Max集成了观测函数和奖励函数、基于变压器的编码器以及训练管道。此外，它还包括对抗性评估设置和一组广泛的评估指标。通过大规模基准测试，我们分析了网络架构、观测函数、训练数据和奖懆塑形对RL性能的影响。 

---
# DexGrasp Anything: Towards Universal Robotic Dexterous Grasping with Physics Awareness 

**Title (ZH)**: 基于物理意识的全能机器人灵巧抓取：DexGrasp Anything 

**Authors**: Yiming Zhong, Qi Jiang, Jingyi Yu, Yuexin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.08257)  

**Abstract**: A dexterous hand capable of grasping any object is essential for the development of general-purpose embodied intelligent robots. However, due to the high degree of freedom in dexterous hands and the vast diversity of objects, generating high-quality, usable grasping poses in a robust manner is a significant challenge. In this paper, we introduce DexGrasp Anything, a method that effectively integrates physical constraints into both the training and sampling phases of a diffusion-based generative model, achieving state-of-the-art performance across nearly all open datasets. Additionally, we present a new dexterous grasping dataset containing over 3.4 million diverse grasping poses for more than 15k different objects, demonstrating its potential to advance universal dexterous grasping. The code of our method and our dataset will be publicly released soon. 

**Abstract (ZH)**: 一种能够抓取任意物体的灵巧手对于通用 embodiment 智能机器人的发展至关重要。然而，由于灵巧手的大自由度和物体的极大多样性，以稳健的方式生成高质量且实用的抓取姿态是一个重大挑战。本文介绍了一种方法——DexGrasp Anything，该方法在基于扩散的生成模型的训练和采样阶段有效集成物理约束，实现了在几乎所有公开数据集上的最佳性能。此外，我们还提出了一种新的灵巧抓取数据集，包含超过340万种不同的抓取姿态，涉及超过15,000种不同的物体，展示了其在推动通用灵巧抓取方面的能力。我们的方法代码和数据集将在不久的将来公开发布。 

---
# HASARD: A Benchmark for Vision-Based Safe Reinforcement Learning in Embodied Agents 

**Title (ZH)**: HASARD：基于视觉的体态智能体安全强化学习基准测试 

**Authors**: Tristan Tomilin, Meng Fang, Mykola Pechenizkiy  

**Link**: [PDF](https://arxiv.org/pdf/2503.08241)  

**Abstract**: Advancing safe autonomous systems through reinforcement learning (RL) requires robust benchmarks to evaluate performance, analyze methods, and assess agent competencies. Humans primarily rely on embodied visual perception to safely navigate and interact with their surroundings, making it a valuable capability for RL agents. However, existing vision-based 3D benchmarks only consider simple navigation tasks. To address this shortcoming, we introduce \textbf{HASARD}, a suite of diverse and complex tasks to $\textbf{HA}$rness $\textbf{SA}$fe $\textbf{R}$L with $\textbf{D}$oom, requiring strategic decision-making, comprehending spatial relationships, and predicting the short-term future. HASARD features three difficulty levels and two action spaces. An empirical evaluation of popular baseline methods demonstrates the benchmark's complexity, unique challenges, and reward-cost trade-offs. Visualizing agent navigation during training with top-down heatmaps provides insight into a method's learning process. Incrementally training across difficulty levels offers an implicit learning curriculum. HASARD is the first safe RL benchmark to exclusively target egocentric vision-based learning, offering a cost-effective and insightful way to explore the potential and boundaries of current and future safe RL methods. The environments and baseline implementations are open-sourced at this https URL. 

**Abstract (ZH)**: 通过强化学习推动安全自主系统的进步需要稳健的基准来评估性能、分析方法和评估代理能力。由于人类主要依赖具身视觉感知来安全导航和与周围环境交互，这是一项对强化学习代理有价值的技能。然而，现有的基于视觉的三维基准仅考虑简单的导航任务。为解决这一不足，我们引入了HASARD，这是一个用于利用Doom的多样且复杂的任务套件，要求战略决策、理解空间关系和预测短期未来。HASARD 包含三个难度级别和两个行动空间。对流行基线方法的经验性评估展示了该基准的复杂性、独特的挑战和奖励与成本之间的权衡。在训练期间可视化代理导航的过程中的鸟瞰热图为方法的学习过程提供了见解。逐步跨难度级别训练提供了隐式的学习课程。HASARD 是首个专门针对自视点视觉感知学习的安全强化学习基准，提供了一种经济高效且具有洞察力的方式，以探索当前和未来安全强化学习方法的潜力和界限。环境和基线实现开源于此 <https://>。 

---
# POp-GS: Next Best View in 3D-Gaussian Splatting with P-Optimality 

**Title (ZH)**: POp-GS: 基于P-最优性的3D高斯点云最佳视角选择 

**Authors**: Joey Wilson, Marcelino Almeida, Sachit Mahajan, Martin Labrie, Maani Ghaffari, Omid Ghasemalizadeh, Min Sun, Cheng-Hao Kuo, Arnab Sen  

**Link**: [PDF](https://arxiv.org/pdf/2503.07819)  

**Abstract**: In this paper, we present a novel algorithm for quantifying uncertainty and information gained within 3D Gaussian Splatting (3D-GS) through P-Optimality. While 3D-GS has proven to be a useful world model with high-quality rasterizations, it does not natively quantify uncertainty. Quantifying uncertainty in parameters of 3D-GS is necessary to understand the information gained from acquiring new images as in active perception, or identify redundant images which can be removed from memory due to resource constraints in online 3D-GS SLAM. We propose to quantify uncertainty and information gain in 3D-GS by reformulating the problem through the lens of optimal experimental design, which is a classical solution to measuring information gain. By restructuring information quantification of 3D-GS through optimal experimental design, we arrive at multiple solutions, of which T-Optimality and D-Optimality perform the best quantitatively and qualitatively as measured on two popular datasets. Additionally, we propose a block diagonal approximation of the 3D-GS uncertainty, which provides a measure of correlation for computing more accurate information gain, at the expense of a greater computation cost. 

**Abstract (ZH)**: 基于P-优化的一种新型3D高斯点表示中不确定性与信息增益的量化方法 

---
# DriveTransformer: Unified Transformer for Scalable End-to-End Autonomous Driving 

**Title (ZH)**: DriveTransformer: 统一的端到端可扩展自主驾驶Transformer模型 

**Authors**: Xiaosong Jia, Junqi You, Zhiyuan Zhang, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2503.07656)  

**Abstract**: End-to-end autonomous driving (E2E-AD) has emerged as a trend in the field of autonomous driving, promising a data-driven, scalable approach to system design. However, existing E2E-AD methods usually adopt the sequential paradigm of perception-prediction-planning, which leads to cumulative errors and training instability. The manual ordering of tasks also limits the system`s ability to leverage synergies between tasks (for example, planning-aware perception and game-theoretic interactive prediction and planning). Moreover, the dense BEV representation adopted by existing methods brings computational challenges for long-range perception and long-term temporal fusion. To address these challenges, we present DriveTransformer, a simplified E2E-AD framework for the ease of scaling up, characterized by three key features: Task Parallelism (All agent, map, and planning queries direct interact with each other at each block), Sparse Representation (Task queries direct interact with raw sensor features), and Streaming Processing (Task queries are stored and passed as history information). As a result, the new framework is composed of three unified operations: task self-attention, sensor cross-attention, temporal cross-attention, which significantly reduces the complexity of system and leads to better training stability. DriveTransformer achieves state-of-the-art performance in both simulated closed-loop benchmark Bench2Drive and real world open-loop benchmark nuScenes with high FPS. 

**Abstract (ZH)**: 端到端自动驾驶（E2E-AD）：DriveTransformer框架及其应用 

---
# Real-Time Detection of Robot Failures Using Gaze Dynamics in Collaborative Tasks 

**Title (ZH)**: 使用凝视动力学进行协作任务中实时机器人故障检测 

**Authors**: Ramtin Tabatabaei, Vassilis Kostakos, Wafa Johal  

**Link**: [PDF](https://arxiv.org/pdf/2503.07622)  

**Abstract**: Detecting robot failures during collaborative tasks is crucial for maintaining trust in human-robot interactions. This study investigates user gaze behaviour as an indicator of robot failures, utilising machine learning models to distinguish between non-failure and two types of failures: executional and decisional. Eye-tracking data were collected from 26 participants collaborating with a robot on Tangram puzzle-solving tasks. Gaze metrics, such as average gaze shift rates and the probability of gazing at specific areas of interest, were used to train machine learning classifiers, including Random Forest, AdaBoost, XGBoost, SVM, and CatBoost. The results show that Random Forest achieved 90% accuracy for detecting executional failures and 80% for decisional failures using the first 5 seconds of failure data. Real-time failure detection was evaluated by segmenting gaze data into intervals of 3, 5, and 10 seconds. These findings highlight the potential of gaze dynamics for real-time error detection in human-robot collaboration. 

**Abstract (ZH)**: 检测协作任务中机器人的故障对于维护人机交互中的信任至关重要。本研究调查了用户注视行为作为机器人故障指标的应用，并利用机器学习模型区分非故障和两种类型的故障：执行故障和决策故障。研究人员从26名参与者完成七巧板拼图任务时与机器人协作的数据中收集了眼动追踪数据。使用平均注视转换率和注视特定区域的概率等注视指标训练了包括随机森林、AdaBoost、XGBoost、SVM和CatBoost在内的机器学习分类器。结果显示，随机森林在使用故障发生后最初5秒的数据时，对于检测执行故障的准确率为90%，对于检测决策故障的准确率为80%。通过将注视数据分割成3秒、5秒和10秒的区间以评估实时故障检测性能。这些发现强调了注视动力学在人机协作中实时错误检测的潜力。 

---
# Seeing and Reasoning with Confidence: Supercharging Multimodal LLMs with an Uncertainty-Aware Agentic Framework 

**Title (ZH)**: 基于不确定性意识代理框架增强的多模态大语言模型的感知与推理 

**Authors**: Zhuo Zhi, Chen Feng, Adam Daneshmend, Mine Orlu, Andreas Demosthenous, Lu Yin, Da Li, Ziquan Liu, Miguel R. D. Rodrigues  

**Link**: [PDF](https://arxiv.org/pdf/2503.08308)  

**Abstract**: Multimodal large language models (MLLMs) show promise in tasks like visual question answering (VQA) but still face challenges in multimodal reasoning. Recent works adapt agentic frameworks or chain-of-thought (CoT) reasoning to improve performance. However, CoT-based multimodal reasoning often demands costly data annotation and fine-tuning, while agentic approaches relying on external tools risk introducing unreliable output from these tools. In this paper, we propose Seeing and Reasoning with Confidence (SRICE), a training-free multimodal reasoning framework that integrates external vision models with uncertainty quantification (UQ) into an MLLM to address these challenges. Specifically, SRICE guides the inference process by allowing MLLM to autonomously select regions of interest through multi-stage interactions with the help of external tools. We propose to use a conformal prediction-based approach to calibrate the output of external tools and select the optimal tool by estimating the uncertainty of an MLLM's output. Our experiment shows that the average improvement of SRICE over the base MLLM is 4.6% on five datasets and the performance on some datasets even outperforms fine-tuning-based methods, revealing the significance of ensuring reliable tool use in an MLLM agent. 

**Abstract (ZH)**: 基于自信推理的多模态 reasoning 体系结构（SRICE） 

---
# Efficient Neural Clause-Selection Reinforcement 

**Title (ZH)**: 高效神经子句选择强化学习 

**Authors**: Martin Suda  

**Link**: [PDF](https://arxiv.org/pdf/2503.07792)  

**Abstract**: Clause selection is arguably the most important choice point in saturation-based theorem proving. Framing it as a reinforcement learning (RL) task is a way to challenge the human-designed heuristics of state-of-the-art provers and to instead automatically evolve -- just from prover experiences -- their potentially optimal replacement. In this work, we present a neural network architecture for scoring clauses for clause selection that is powerful yet efficient to evaluate. Following RL principles to make design decisions, we integrate the network into the Vampire theorem prover and train it from successful proof attempts. An experiment on the diverse TPTP benchmark finds the neurally guided prover improve over a baseline strategy, from which it initially learns--in terms of the number of in-training-unseen problems solved under a practically relevant, short CPU instruction limit--by 20%. 

**Abstract (ZH)**: 基于饱和定理证明的句法选择是最重要的决策点。将其作为一个强化学习任务可以挑战最先进的证明器设计的人工启发式方法，并通过证明器的经验自动进化可能最优的替代方法。本文提出了一种高效评价的神经网络架构用于句法选择分数计算。遵循强化学习原理进行设计决策，我们将网络集成到Vampire证明器中，并从成功的证明尝试中对其进行训练。一项针对TPTP基准的实验发现，神经引导的证明器在缩短的CPU指令限制下，相较于基准策略，在解决训练中未见过的问题数量上提高了20%。 

---
# Sensemaking in Novel Environments: How Human Cognition Can Inform Artificial Agents 

**Title (ZH)**: 新环境中意义构建：人类认知如何指导人工代理 

**Authors**: Robert E. Patterson, Regina Buccello-Stout, Mary E. Frame, Anna M. Maresca, Justin Nelson, Barbara Acker-Mills, Erica Curtis, Jared Culbertson, Kevin Schmidt, Scott Clouse, Steve Rogers  

**Link**: [PDF](https://arxiv.org/pdf/2503.07783)  

**Abstract**: One of the most vital cognitive skills to possess is the ability to make sense of objects, events, and situations in the world. In the current paper, we offer an approach for creating artificially intelligent agents with the capacity for sensemaking in novel environments. Objectives: to present several key ideas: (1) a novel unified conceptual framework for sensemaking (which includes the existence of sign relations embedded within and across frames); (2) interaction among various content-addressable, distributed-knowledge structures via shared attributes (whose net response would represent a synthesized object, event, or situation serving as a sign for sensemaking in a novel environment). Findings: we suggest that attributes across memories can be shared and recombined in novel ways to create synthesized signs, which can denote certain outcomes in novel environments (i.e., sensemaking). 

**Abstract (ZH)**: 掌握对世界中的物体、事件和情境进行理解的认知能力是极其重要的。本文提出了一个创造能够在新型环境中进行理解的人工智能代理的方法。目标：提出几个关键概念：（1）一种新颖统一的理解概念框架（其中包括嵌套和跨域的关系符号）；（2）通过共享属性来促进各种内容可寻址、分布式知识结构之间的交互（这些交互的综合响应将代表一种合成的目标、事件或情况，作为新型环境中的符号，用于理解）。发现：我们建议，在记忆中的属性可以以新颖的方式共享和重组以生成合成符号，这些符号能够表示新型环境中的某些结果（即理解）。 

---
# A Grid Cell-Inspired Structured Vector Algebra for Cognitive Maps 

**Title (ZH)**: 基于网格细胞启发的结构化向量代数认知地图 

**Authors**: Sven Krausse, Emre Neftci, Friedrich T. Sommer, Alpha Renner  

**Link**: [PDF](https://arxiv.org/pdf/2503.08608)  

**Abstract**: The entorhinal-hippocampal formation is the mammalian brain's navigation system, encoding both physical and abstract spaces via grid cells. This system is well-studied in neuroscience, and its efficiency and versatility make it attractive for applications in robotics and machine learning. While continuous attractor networks (CANs) successfully model entorhinal grid cells for encoding physical space, integrating both continuous spatial and abstract spatial computations into a unified framework remains challenging. Here, we attempt to bridge this gap by proposing a mechanistic model for versatile information processing in the entorhinal-hippocampal formation inspired by CANs and Vector Symbolic Architectures (VSAs), a neuro-symbolic computing framework. The novel grid-cell VSA (GC-VSA) model employs a spatially structured encoding scheme with 3D neuronal modules mimicking the discrete scales and orientations of grid cell modules, reproducing their characteristic hexagonal receptive fields. In experiments, the model demonstrates versatility in spatial and abstract tasks: (1) accurate path integration for tracking locations, (2) spatio-temporal representation for querying object locations and temporal relations, and (3) symbolic reasoning using family trees as a structured test case for hierarchical relationships. 

**Abstract (ZH)**: 内嗅-海马复合体是哺乳动物大脑的导航系统，通过网格细胞编码物理和抽象空间。该系统在神经科学中被广泛研究，其高效性和灵活性使其在机器人技术和机器学习领域具有吸引力。虽然连续吸引网络（CANs）成功地模拟了内嗅网格细胞以编码物理空间，但将连续空间计算和抽象空间计算统一流程化仍具有挑战性。在这里，我们通过提出受CANs和向量象征架构（VSAs）启发的多功能信息处理机制模型，尝试弥合这一差距。该新颖的网格细胞VSA（GC-VSA）模型采用空间结构化的编码方案，使用3D神经模块模拟网格细胞模块的离散尺度和方向，再现其特征性的六边形感受野。在实验中，该模型在空间和抽象任务中展示了多功能性：（1）准确的位置路径整合，（2）时空表示以查询物体位置和时间关系，以及（3）使用家族树作为分层关系的结构化测试案例进行符号推理。 

---
# GTR: Guided Thought Reinforcement Prevents Thought Collapse in RL-based VLM Agent Training 

**Title (ZH)**: GTR：引导性思考强化防止基于RL的VLM代理培训中思考崩溃 

**Authors**: Tong Wei, Yijun Yang, Junliang Xing, Yuanchun Shi, Zongqing Lu, Deheng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2503.08525)  

**Abstract**: Reinforcement learning with verifiable outcome rewards (RLVR) has effectively scaled up chain-of-thought (CoT) reasoning in large language models (LLMs). Yet, its efficacy in training vision-language model (VLM) agents for goal-directed action reasoning in visual environments is less established. This work investigates this problem through extensive experiments on complex card games, such as 24 points, and embodied tasks from ALFWorld. We find that when rewards are based solely on action outcomes, RL fails to incentivize CoT reasoning in VLMs, instead leading to a phenomenon we termed thought collapse, characterized by a rapid loss of diversity in the agent's thoughts, state-irrelevant and incomplete reasoning, and subsequent invalid actions, resulting in negative rewards. To counteract thought collapse, we highlight the necessity of process guidance and propose an automated corrector that evaluates and refines the agent's reasoning at each RL step. This simple and scalable GTR (Guided Thought Reinforcement) framework trains reasoning and action simultaneously without the need for dense, per-step human labeling. Our experiments demonstrate that GTR significantly enhances the performance and generalization of the LLaVA-7b model across various visual environments, achieving 3-5 times higher task success rates compared to SoTA models with notably smaller model sizes. 

**Abstract (ZH)**: 验证性奖励学习（RLVR）在大型语言模型（LLMs）中有效扩展了链式思考（CoT）推理。然而，其在视觉语言模型（VLM）代理在视觉环境中的目标导向行动推理训练方面的有效性尚不明确。本工作通过在24点等复杂纸牌游戏以及ALFWorld中的实体任务上进行广泛的实验，探讨了这一问题。我们发现，当奖励仅基于行动结果时，RL无法激励VLM中的链式思考推理，反而导致我们称之为思维崩溃的现象，表现为代理思维的迅速同质化、与状态无关及不完整的推理，以及随后的无效行动，导致负奖励。为对抗思维崩溃，我们强调过程指导的必要性，并提出了一种自动化校正器，能够在每个RL步骤中评估和改进代理的推理。这种简单且可扩展的引导思考强化学习（GTR）框架能够在无需密集、逐步的人工标注的情况下同时训练推理和行动。我们的实验结果表明，GTR显著提升了LLaVA-7b模型在各种视觉环境中的性能和泛化能力，与性能相当但模型规模显著较小的最新模型相比，任务成功率提高了3-5倍。 

---
# Evaluating Interpretable Reinforcement Learning by Distilling Policies into Programs 

**Title (ZH)**: 将策略提炼为程序以评估可解释的强化学习 

**Authors**: Hector Kohler, Quentin Delfosse, Waris Radji, Riad Akrour, Philippe Preux  

**Link**: [PDF](https://arxiv.org/pdf/2503.08322)  

**Abstract**: There exist applications of reinforcement learning like medicine where policies need to be ''interpretable'' by humans. User studies have shown that some policy classes might be more interpretable than others. However, it is costly to conduct human studies of policy interpretability. Furthermore, there is no clear definition of policy interpretabiliy, i.e., no clear metrics for interpretability and thus claims depend on the chosen definition. We tackle the problem of empirically evaluating policies interpretability without humans. Despite this lack of clear definition, researchers agree on the notions of ''simulatability'': policy interpretability should relate to how humans understand policy actions given states. To advance research in interpretable reinforcement learning, we contribute a new methodology to evaluate policy interpretability. This new methodology relies on proxies for simulatability that we use to conduct a large-scale empirical evaluation of policy interpretability. We use imitation learning to compute baseline policies by distilling expert neural networks into small programs. We then show that using our methodology to evaluate the baselines interpretability leads to similar conclusions as user studies. We show that increasing interpretability does not necessarily reduce performances and can sometimes increase them. We also show that there is no policy class that better trades off interpretability and performance across tasks making it necessary for researcher to have methodologies for comparing policies interpretability. 

**Abstract (ZH)**: 存在医学等应用领域的强化学习中，策略需要“可解释”给人类。虽然用户研究显示某些策略类别可能比其他类别更容易解释，但由于进行策略可解释性的用户研究成本高，且目前缺乏明确的策略可解释性定义，使得解释性依赖于所选择的定义。我们致力于通过不依赖人类的研究方法来实证评估策略的可解释性。尽管缺乏明确的定义，研究人员一致认为“可模拟性”的概念：策略可解释性应与人类理解给定状态下的策略行动相关。为了促进可解释强化学习的研究，我们贡献了一种新的评估策略可解释性的方法。该方法依赖于模拟性的代理指标，并通过大规模实证研究评估策略的可解释性。我们使用模仿学习将专家神经网络提炼为小型程序来计算基线策略，展示了使用我们的方法评估这些基线策略的可解释性，所得结论与用户研究相似。我们证明增加可解释性不一定降低性能，有时甚至可以提高性能。我们还证明，在各类任务中没有策略类别能够在可解释性和性能之间取得更好的权衡，因此研究人员需要具有比较策略可解释性的方法。 

---
# Toward Stable World Models: Measuring and Addressing World Instability in Generative Environments 

**Title (ZH)**: 朝向稳定的世界模型：衡量与解决生成环境中世界不稳定性的方法 

**Authors**: Soonwoo Kwon, Jin-Young Kim, Hyojun Go, Kyungjune Baek  

**Link**: [PDF](https://arxiv.org/pdf/2503.08122)  

**Abstract**: We present a novel study on enhancing the capability of preserving the content in world models, focusing on a property we term World Stability. Recent diffusion-based generative models have advanced the synthesis of immersive and realistic environments that are pivotal for applications such as reinforcement learning and interactive game engines. However, while these models excel in quality and diversity, they often neglect the preservation of previously generated scenes over time--a shortfall that can introduce noise into agent learning and compromise performance in safety-critical settings. In this work, we introduce an evaluation framework that measures world stability by having world models perform a sequence of actions followed by their inverses to return to their initial viewpoint, thereby quantifying the consistency between the starting and ending observations. Our comprehensive assessment of state-of-the-art diffusion-based world models reveals significant challenges in achieving high world stability. Moreover, we investigate several improvement strategies to enhance world stability. Our results underscore the importance of world stability in world modeling and provide actionable insights for future research in this domain. 

**Abstract (ZH)**: 一种提升世界模型内容保持能力的新型研究：着眼于我们称为世界稳定性的特性 

---
# Provable Zero-Shot Generalization in Offline Reinforcement Learning 

**Title (ZH)**: 可验证的零样本泛化在离线强化学习中 

**Authors**: Zhiyong Wang, Chen Yang, John C.S. Lui, Dongruo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.07988)  

**Abstract**: In this work, we study offline reinforcement learning (RL) with zero-shot generalization property (ZSG), where the agent has access to an offline dataset including experiences from different environments, and the goal of the agent is to train a policy over the training environments which performs well on test environments without further interaction. Existing work showed that classical offline RL fails to generalize to new, unseen environments. We propose pessimistic empirical risk minimization (PERM) and pessimistic proximal policy optimization (PPPO), which leverage pessimistic policy evaluation to guide policy learning and enhance generalization. We show that both PERM and PPPO are capable of finding a near-optimal policy with ZSG. Our result serves as a first step in understanding the foundation of the generalization phenomenon in offline reinforcement learning. 

**Abstract (ZH)**: 在本工作中，我们研究了具有零-shot泛化能力（ZSG）的离线强化学习（RL），其中智能体可以访问包含不同环境经验的离线数据集，并且智能体的目标是训练一个在训练环境上的策略能够在测试环境上表现良好，而无需进一步交互。现有研究表明，经典离线RL无法泛化到新的未见过的环境。我们提出悲观经验风险最小化（PERM）和悲观近端策略优化（PPPO），利用悲观策略评估来指导策略学习并增强泛化能力。我们证明了PERM和PPPO都能够通过ZSG找到接近最优的策略。我们的结果为进一步理解离线强化学习中的泛化现象奠定了基础。 

---
# Safety Guardrails for LLM-Enabled Robots 

**Title (ZH)**: LLM驱动机器人安全守rail 

**Authors**: Zachary Ravichandran, Alexander Robey, Vijay Kumar, George J. Pappas, Hamed Hassani  

**Link**: [PDF](https://arxiv.org/pdf/2503.07885)  

**Abstract**: Although the integration of large language models (LLMs) into robotics has unlocked transformative capabilities, it has also introduced significant safety concerns, ranging from average-case LLM errors (e.g., hallucinations) to adversarial jailbreaking attacks, which can produce harmful robot behavior in real-world settings. Traditional robot safety approaches do not address the novel vulnerabilities of LLMs, and current LLM safety guardrails overlook the physical risks posed by robots operating in dynamic real-world environments. In this paper, we propose RoboGuard, a two-stage guardrail architecture to ensure the safety of LLM-enabled robots. RoboGuard first contextualizes pre-defined safety rules by grounding them in the robot's environment using a root-of-trust LLM, which employs chain-of-thought (CoT) reasoning to generate rigorous safety specifications, such as temporal logic constraints. RoboGuard then resolves potential conflicts between these contextual safety specifications and a possibly unsafe plan using temporal logic control synthesis, which ensures safety compliance while minimally violating user preferences. Through extensive simulation and real-world experiments that consider worst-case jailbreaking attacks, we demonstrate that RoboGuard reduces the execution of unsafe plans from 92% to below 2.5% without compromising performance on safe plans. We also demonstrate that RoboGuard is resource-efficient, robust against adaptive attacks, and significantly enhanced by enabling its root-of-trust LLM to perform CoT reasoning. These results underscore the potential of RoboGuard to mitigate the safety risks and enhance the reliability of LLM-enabled robots. 

**Abstract (ZH)**: RoboGuard：一种确保大语言模型赋能机器人安全的两阶段护栏架构 

---
