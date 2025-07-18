# ROS Help Desk: GenAI Powered, User-Centric Framework for ROS Error Diagnosis and Debugging 

**Title (ZH)**: ROS Help Desk：以用户为中心的基于GenAI的ROS错误诊断与调试框架 

**Authors**: Kavindie Katuwandeniya, Samith Rajapaksha Jayasekara Widhanapathirana  

**Link**: [PDF](https://arxiv.org/pdf/2507.07846)  

**Abstract**: As the robotics systems increasingly integrate into daily life, from smart home assistants to the new-wave of industrial automation systems (Industry 4.0), there's an increasing need to bridge the gap between complex robotic systems and everyday users. The Robot Operating System (ROS) is a flexible framework often utilised in writing robot software, providing tools and libraries for building complex robotic systems. However, ROS's distributed architecture and technical messaging system create barriers for understanding robot status and diagnosing errors. This gap can lead to extended maintenance downtimes, as users with limited ROS knowledge may struggle to quickly diagnose and resolve system issues. Moreover, this deficit in expertise often delays proactive maintenance and troubleshooting, further increasing the frequency and duration of system interruptions. ROS Help Desk provides intuitive error explanations and debugging support, dynamically customized to users of varying expertise levels. It features user-centric debugging tools that simplify error diagnosis, implements proactive error detection capabilities to reduce downtime, and integrates multimodal data processing for comprehensive system state understanding across multi-sensor data (e.g., lidar, RGB). Testing qualitatively and quantitatively with artificially induced errors demonstrates the system's ability to proactively and accurately diagnose problems, ultimately reducing maintenance time and fostering more effective human-robot collaboration. 

**Abstract (ZH)**: 随着机器人系统越来越多地融入日常生活中，从智能家庭助手到工业4.0新一代自动化系统， bridges the gap between复杂的机器人系统和普通用户的需求不断增加。机器人操作系统（ROS）是一个灵活的框架，常用于编写机器人软件，提供构建复杂机器人系统的工具和库。然而，ROS的分布式架构和技术消息系统会为理解机器人状态和诊断错误设置障碍。这一差距可能导致维护时间延长，因为知识有限的用户可能难以快速诊断和解决问题。此外，这种专业知识的不足往往会延迟主动维护和故障排除，从而增加系统中断的频率和持续时间。ROS Help Desk提供了直观的错误解释和调试支持，可以根据用户的不同专业水平动态定制。它配备了以用户为中心的调试工具，简化了错误诊断过程，实现了主动错误检测能力以减少停机时间，并整合了多模态数据处理，以便全面理解多传感器数据（如激光雷达、RGB）下的系统状态。通过引入人工诱导的错误进行定性和定量测试，展示了该系统主动且准确地诊断问题的能力，最终减少了维护时间并促进了更有效的机器人协作。 

---
# Perceptual Distortions and Autonomous Representation Learning in a Minimal Robotic System 

**Title (ZH)**: 感知失真与最小机器人系统中的自主表征学习 

**Authors**: David Warutumo, Ciira wa Maina  

**Link**: [PDF](https://arxiv.org/pdf/2507.07845)  

**Abstract**: Autonomous agents, particularly in the field of robotics, rely on sensory information to perceive and navigate their environment. However, these sensory inputs are often imperfect, leading to distortions in the agent's internal representation of the world. This paper investigates the nature of these perceptual distortions and how they influence autonomous representation learning using a minimal robotic system. We utilize a simulated two-wheeled robot equipped with distance sensors and a compass, operating within a simple square environment. Through analysis of the robot's sensor data during random exploration, we demonstrate how a distorted perceptual space emerges. Despite these distortions, we identify emergent structures within the perceptual space that correlate with the physical environment, revealing how the robot autonomously learns a structured representation for navigation without explicit spatial information. This work contributes to the understanding of embodied cognition, minimal agency, and the role of perception in self-generated navigation strategies in artificial life. 

**Abstract (ZH)**: 自主代理，特别是在机器人领域，依赖于感官信息来感知和导航其环境。然而，这些感官输入往往是不完美的，导致代理内部对世界的表征出现失真。本文通过一个简易的轮式机器人系统研究了这些感知失真的本质及其如何影响自主表征学习。我们利用一个装备有距离传感器和罗盘的模拟两轮机器人，在一个简单的方形环境中操作。通过对机器人在随机探索过程中的传感器数据进行分析，我们展示了如何出现一个失真的感知空间。尽管存在这些失真，我们发现感知空间中存在与物理环境相关的新兴结构，揭示了在没有显性空间信息的情况下，机器人如何自主学习一个结构化导航表示。本文为了解体态认知、最小代理以及感知在人工生命中自我生成导航策略中的作用做出了贡献。 

---
# Beyond Robustness: Learning Unknown Dynamic Load Adaptation for Quadruped Locomotion on Rough Terrain 

**Title (ZH)**: 超越鲁棒性：学习未知动态负载适应以实现粗糙地形上的四足运动 

**Authors**: Leixin Chang, Yuxuan Nai, Hua Chen, Liangjing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.07825)  

**Abstract**: Unknown dynamic load carrying is one important practical application for quadruped robots. Such a problem is non-trivial, posing three major challenges in quadruped locomotion control. First, how to model or represent the dynamics of the load in a generic manner. Second, how to make the robot capture the dynamics without any external sensing. Third, how to enable the robot to interact with load handling the mutual effect and stabilizing the load. In this work, we propose a general load modeling approach called load characteristics modeling to capture the dynamics of the load. We integrate this proposed modeling technique and leverage recent advances in Reinforcement Learning (RL) based locomotion control to enable the robot to infer the dynamics of load movement and interact with the load indirectly to stabilize it and realize the sim-to-real deployment to verify its effectiveness in real scenarios. We conduct extensive comparative simulation experiments to validate the effectiveness and superiority of our proposed method. Results show that our method outperforms other methods in sudden load resistance, load stabilizing and locomotion with heavy load on rough terrain. \href{this https URL}{Project Page}. 

**Abstract (ZH)**: 未知动力负载承载是四足机器人的一项重要实际应用。这个问题并不简单，在四足机器人运动控制中提出了三大挑战。首先，如何以通用方式建模或表示负载的动力学。其次，如何让机器人在没有任何外部传感的情况下捕捉到负载的动力学。第三，如何使机器人能够处理负载交互效应并稳定负载。在本文中，我们提出了一种称为负载特性建模的通用负载建模方法，以捕捉负载的动力学。我们结合了所提出建模技术，并利用基于强化学习（RL）的运动控制最近的进展，使机器人能够推断负载移动的动力学并通过间接方式与负载互动以稳定它，并实现从仿真到现实世界的部署，以验证其在实际场景中的有效性。我们进行了大量的比较仿真实验来验证我们提出方法的有效性和优越性。结果表明，我们的方法在突然负载抵抗、负载稳定和在粗糙地形上承载重负载方面优于其他方法。点击此处查看项目页面。 

---
# UniTracker: Learning Universal Whole-Body Motion Tracker for Humanoid Robots 

**Title (ZH)**: UniTracker: 学习通用全身运动跟踪器用于类人机器人 

**Authors**: Kangning Yin, Weishuai Zeng, Ke Fan, Zirui Wang, Qiang Zhang, Zheng Tian, Jingbo Wang, Jiangmiao Pang, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.07356)  

**Abstract**: Humanoid robots must achieve diverse, robust, and generalizable whole-body control to operate effectively in complex, human-centric environments. However, existing methods, particularly those based on teacher-student frameworks often suffer from a loss of motion diversity during policy distillation and exhibit limited generalization to unseen behaviors. In this work, we present UniTracker, a simplified yet powerful framework that integrates a Conditional Variational Autoencoder (CVAE) into the student policy to explicitly model the latent diversity of human motion. By leveraging a learned CVAE prior, our method enables the student to retain expressive motion characteristics while improving robustness and adaptability under partial observations. The result is a single policy capable of tracking a wide spectrum of whole-body motions with high fidelity and stability. Comprehensive experiments in both simulation and real-world deployments demonstrate that UniTracker significantly outperforms MLP-based DAgger baselines in motion quality, generalization to unseen references, and deployment robustness, offering a practical and scalable solution for expressive humanoid control. 

**Abstract (ZH)**: 人形机器人必须在复杂的人本中心环境中实现多样、 robust 和普适的整体身体控制。然而，现有的方法，尤其是基于教师-学生框架的方法，在策略蒸馏过程中常常丧失运动多样性，并且对未见过的行为表现出有限的泛化能力。在本文中，我们提出了一种简化但强大的 UniTracker 框架，将条件变分自编码器（CVAE）集成到学生策略中，以明确建模人类运动的潜在多样性。通过利用学习到的 CVAE 先验，我们的方法使学生能够在保持表达性运动特征的同时，提高在部分观测条件下的鲁棒性和适应性。结果是单一策略能够以高保真度和稳定性跟踪广泛的全身运动。在仿真和真实世界部署中的全面实验表明，UniTracker 在运动质量、对未见过的参考的泛化能力和部署鲁棒性方面显著优于基于MLP的DAgger基线，提供了一种实用且可扩展的表达性人形控制解决方案。 

---
# Classifying Emergence in Robot Swarms: An Observer-Dependent Approach 

**Title (ZH)**: 基于观察者依赖的方法划分机器人 swarm 中的涌现现象 

**Authors**: Ricardo Vega, Cameron Nowzari  

**Link**: [PDF](https://arxiv.org/pdf/2507.07315)  

**Abstract**: Emergence and swarms are widely discussed topics, yet no consensus exists on their formal definitions. This lack of agreement makes it difficult not only for new researchers to grasp these concepts, but also for experts who may use the same terms to mean different things. Many attempts have been made to objectively define 'swarm' or 'emergence,' with recent work highlighting the role of the external observer. Still, several researchers argue that once an observer's vantage point (e.g., scope, resolution, context) is established, the terms can be made objective or measured quantitatively. In this note, we propose a framework to discuss these ideas rigorously by separating externally observable states from latent, unobservable ones. This allows us to compare and contrast existing definitions of swarms and emergence on common ground. We argue that these concepts are ultimately subjective-shaped less by the system itself than by the perception and tacit knowledge of the observer. Specifically, we suggest that a 'swarm' is not defined by its group behavior alone, but by the process generating that behavior. Our broader goal is to support the design and deployment of robotic swarm systems, highlighting the critical distinction between multi-robot systems and true swarms. 

**Abstract (ZH)**: Emergence和群集广泛讨论但缺乏正式定义的共识：外部观测视角下探讨群集和涌现的概念框架 

---
# LangNavBench: Evaluation of Natural Language Understanding in Semantic Navigation 

**Title (ZH)**: LangNavBench: 语义导航中的自然语言理解评估 

**Authors**: Sonia Raychaudhuri, Enrico Cancelli, Tommaso Campari, Lamberto Ballan, Manolis Savva, Angel X. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2507.07299)  

**Abstract**: Recent progress in large vision-language models has driven improvements in language-based semantic navigation, where an embodied agent must reach a target object described in natural language. Despite these advances, we still lack a clear, language-focused benchmark for testing how well such agents ground the words in their instructions. We address this gap with LangNav, an open-set dataset specifically created to test an agent's ability to locate objects described at different levels of detail, from broad category names to fine attributes and object-object relations. Every description in LangNav was manually checked, yielding a lower error rate than existing lifelong- and semantic-navigation datasets. On top of LangNav we build LangNavBench, a benchmark that measures how well current semantic-navigation methods understand and act on these descriptions while moving toward their targets. LangNavBench allows us to systematically compare models on their handling of attributes, spatial and relational cues, and category hierarchies, offering the first thorough, language-centric evaluation of embodied navigation systems. We also present Multi-Layered Feature Map (MLFM), a method that builds a queryable multi-layered semantic map, particularly effective when dealing with small objects or instructions involving spatial relations. MLFM outperforms state-of-the-art mapping-based navigation baselines on the LangNav dataset. 

**Abstract (ZH)**: Recent进展在大规模多模态模型方面推动了基于语言的语义导航的进步，其中实体代理必须根据自然语言描述到达目标物体。尽管取得了这些进展，我们仍缺乏一个清晰的语言导向基准来测试这些代理如何理解指令中的词语。我们通过建立LangNav，一个开放集数据集来填补这一空白，该数据集专门用于测试代理识别不同详细程度描述对象的能力，从广泛的类别名称到细微的属性和物体间关系。LangNav 中的每个描述都经过人工检查，导致错误率低于现有的终身语义导航数据集。在LangNav基础上，我们构建了LangNavBench，该基准用于衡量当前语义导航方法在向目标移动过程中理解并执行这些描述的效果。LangNavBench 允许我们系统地比较模型在处理属性、空间和关系线索以及类别层次结构方面的能力，提供第一个全面的语言导向评估方法体系。我们还介绍了多层特征图（MLFM）方法，这是一种构建可查询多层语义图的方法，特别适用于处理小型物体或涉及空间关系的指令。MLFM 在LangNav数据集上的表现优于最先进的基于映射的导航基准。 

---
# SURPRISE3D: A Dataset for Spatial Understanding and Reasoning in Complex 3D Scenes 

**Title (ZH)**: SURPRISE3D: 一个用于复杂三维场景空间理解与推理的数据集 

**Authors**: Jiaxin Huang, Ziwen Li, Hanlve Zhang, Runnan Chen, Xiao He, Yandong Guo, Wenping Wang, Tongliang Liu, Mingming Gong  

**Link**: [PDF](https://arxiv.org/pdf/2507.07781)  

**Abstract**: The integration of language and 3D perception is critical for embodied AI and robotic systems to perceive, understand, and interact with the physical world. Spatial reasoning, a key capability for understanding spatial relationships between objects, remains underexplored in current 3D vision-language research. Existing datasets often mix semantic cues (e.g., object name) with spatial context, leading models to rely on superficial shortcuts rather than genuinely interpreting spatial relationships. To address this gap, we introduce S\textsc{urprise}3D, a novel dataset designed to evaluate language-guided spatial reasoning segmentation in complex 3D scenes. S\textsc{urprise}3D consists of more than 200k vision language pairs across 900+ detailed indoor scenes from ScanNet++ v2, including more than 2.8k unique object classes. The dataset contains 89k+ human-annotated spatial queries deliberately crafted without object name, thereby mitigating shortcut biases in spatial understanding. These queries comprehensively cover various spatial reasoning skills, such as relative position, narrative perspective, parametric perspective, and absolute distance reasoning. Initial benchmarks demonstrate significant challenges for current state-of-the-art expert 3D visual grounding methods and 3D-LLMs, underscoring the necessity of our dataset and the accompanying 3D Spatial Reasoning Segmentation (3D-SRS) benchmark suite. S\textsc{urprise}3D and 3D-SRS aim to facilitate advancements in spatially aware AI, paving the way for effective embodied interaction and robotic planning. The code and datasets can be found in this https URL. 

**Abstract (ZH)**: 语言与三维感知的整合对于体现式AI和机器人系统感知、理解和与物理世界交互至关重要。当前三维视觉-语言研究中基于语言指导的三维场景空间推理分割评价数据集鲜有探索。现有数据集常常将语义线索（例如，物体名称）与空间上下文混合，导致模型依赖表面化捷径而非真正理解空间关系。为解决这一问题，我们引入了S\textsc{urprise}3D，这是一个旨在评估复杂三维场景中基于语言指导的空间推理分割的新颖数据集。S\textsc{urprise}3D 包含超过20万的视觉语言对，跨越超过900个详细室内场景（来源于ScanNet++ v2），包括超过2800个独特的物体类别。该数据集包含超过8.9万个人工标注的空间查询，这些查询刻意未包含物体名称，从而减少了空间理解中的捷径偏差。这些查询全面涵盖了各种空间推理技能，例如相对位置、叙述视角、参数视角以及绝对距离推理。初步基准测试表明，当前最先进的专家级三维视觉定位方法和三维语言-模型在S\textsc{urprise}3D上的表现存在显著挑战，强调了我们数据集和伴随的三维空间推理分割基准套件（3D-SRS）的必要性。S\textsc{urprise}3D 和 3D-SRS 力求推动空间感知智能的发展，铺就有效体现交互和机器人规划的道路。代码和数据集可在以下链接找到。 

---
# Pluri-perspectivism in Human-robot Co-creativity with Older Adults 

**Title (ZH)**: 人类与机器人共同创意中的多元主义视角与老年人合作 

**Authors**: Marianne Bossema, Rob Saunders, Aske Plaat, Somaya Ben Allouch  

**Link**: [PDF](https://arxiv.org/pdf/2507.07550)  

**Abstract**: This position paper explores pluriperspectivism as a core element of human creative experience and its relevance to humanrobot cocreativity We propose a layered fivedimensional model to guide the design of cocreative behaviors and the analysis of interaction dynamics This model is based on literature and results from an interview study we conducted with 10 visual artists and 8 arts educators examining how pluriperspectivism supports creative practice The findings of this study provide insight in how robots could enhance human creativity through adaptive contextsensitive behavior demonstrating the potential of pluriperspectivism This paper outlines future directions for integrating pluriperspectivism with visionlanguage models VLMs to support context sensitivity in cocreative robots 

**Abstract (ZH)**: 这篇立场论文探讨了多元透视主义作为人类创造性体验核心要素的地位及其与人机协同创造的相关性。我们提出了一种分层五维模型，以指导协同创造行为的设计以及交互动力学的分析。该模型基于文献和我们对10名视觉艺术家和8名艺术教育者的访谈研究结果，探讨了多元透视主义如何支持创造性实践。本研究的发现提供了关于机器人如何通过适应性上下文敏感行为增强人类创造力的见解，并展示了多元透视主义的潜力。本文概述了将多元透视主义与视觉语言模型VLM结合以支持协同创造机器人上下文敏感性的未来方向。 

---
# AI Should Sense Better, Not Just Scale Bigger: Adaptive Sensing as a Paradigm Shift 

**Title (ZH)**: AI 应该更智能地感知，而不仅仅是规模扩大：自适应感知作为一种范式转变 

**Authors**: Eunsu Baek, Keondo Park, Jeonggil Ko, Min-hwan Oh, Taesik Gong, Hyung-Sin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.07820)  

**Abstract**: Current AI advances largely rely on scaling neural models and expanding training datasets to achieve generalization and robustness. Despite notable successes, this paradigm incurs significant environmental, economic, and ethical costs, limiting sustainability and equitable access. Inspired by biological sensory systems, where adaptation occurs dynamically at the input (e.g., adjusting pupil size, refocusing vision)--we advocate for adaptive sensing as a necessary and foundational shift. Adaptive sensing proactively modulates sensor parameters (e.g., exposure, sensitivity, multimodal configurations) at the input level, significantly mitigating covariate shifts and improving efficiency. Empirical evidence from recent studies demonstrates that adaptive sensing enables small models (e.g., EfficientNet-B0) to surpass substantially larger models (e.g., OpenCLIP-H) trained with significantly more data and compute. We (i) outline a roadmap for broadly integrating adaptive sensing into real-world applications spanning humanoid, healthcare, autonomous systems, agriculture, and environmental monitoring, (ii) critically assess technical and ethical integration challenges, and (iii) propose targeted research directions, such as standardized benchmarks, real-time adaptive algorithms, multimodal integration, and privacy-preserving methods. Collectively, these efforts aim to transition the AI community toward sustainable, robust, and equitable artificial intelligence systems. 

**Abstract (ZH)**: 当前的人工智能进展主要依赖于扩大神经模型规模和扩大训练数据集以实现泛化和鲁棒性。尽管取得了显著的成功，但这种范式产生了重大的环境、经济和伦理成本，限制了可持续性和公平的访问。受生物感觉系统的启发，在输入端动态发生适应（例如，调整瞳孔大小，调节焦距）——我们提倡将适应性感知作为必要的基础性转变。适应性感知在输入端主动调节传感器参数（例如，曝光、灵敏度、多模态配置），显著缓解了 covariate 偏移并提高了效率。最近的研究表明，适应性感知使小型模型（例如，EfficientNet-B0）能够在大大较少的数据和计算资源下超过大规模模型（例如，OpenCLIP-H）。我们将（i）概述将适应性感知广泛整合到类人、医疗保健、自主系统、农业和环境监测等实际应用中的 roadmap，（ii）批判性评估技术与伦理整合的挑战，并（iii）提出有针对性的研究方向，例如标准化基准、实时适应算法、多模态整合和隐私保护方法。这些努力共同旨在使人工智能社区向可持续的、稳健的和公平的人工智能系统转变。 

---
# MoSE: Skill-by-Skill Mixture-of-Expert Learning for Autonomous Driving 

**Title (ZH)**: MoSE：自主驾驶中的技能级专家混合学习 

**Authors**: Lu Xu, Jiaqian Yu, Xiongfeng Peng, Yiwei Chen, Weiming Li, Jaewook Yoo, Sunghyun Chunag, Dongwook Lee, Daehyun Ji, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.07818)  

**Abstract**: Recent studies show large language models (LLMs) and vision language models (VLMs) trained using web-scale data can empower end-to-end autonomous driving systems for a better generalization and interpretation. Specifically, by dynamically routing inputs to specialized subsets of parameters, the Mixture-of-Experts (MoE) technique enables general LLMs or VLMs to achieve substantial performance improvements while maintaining computational efficiency. However, general MoE models usually demands extensive training data and complex optimization. In this work, inspired by the learning process of human drivers, we propose a skill-oriented MoE, called MoSE, which mimics human drivers' learning process and reasoning process, skill-by-skill and step-by-step. We propose a skill-oriented routing mechanism that begins with defining and annotating specific skills, enabling experts to identify the necessary driving competencies for various scenarios and reasoning tasks, thereby facilitating skill-by-skill learning. Further align the driving process to multi-step planning in human reasoning and end-to-end driving models, we build a hierarchical skill dataset and pretrain the router to encourage the model to think step-by-step. Unlike multi-round dialogs, MoSE integrates valuable auxiliary tasks (e.g.\ description, reasoning, planning) in one single forward process without introducing any extra computational cost. With less than 3B sparsely activated parameters, our model outperforms several 8B+ parameters on CODA AD corner case reasoning task. Compared to existing methods based on open-source models and data, our approach achieves state-of-the-art performance with significantly reduced activated model size (at least by $62.5\%$) with a single-turn conversation. 

**Abstract (ZH)**: Recent Studies Show Large Language Models and Vision Language Models Trained with Web-Scale Data Can Empower End-to-End Autonomous Driving Systems for Better Generalization and Interpretation 

---
# StarDojo: Benchmarking Open-Ended Behaviors of Agentic Multimodal LLMs in Production-Living Simulations with Stardew Valley 

**Title (ZH)**: StarDojo: 在 Stardew Valley 生存模拟中评估代理多模态大语言模型的开放-ended 行为 

**Authors**: Weihao Tan, Changjiu Jiang, Yu Duan, Mingcong Lei, Jiageng Li, Yitian Hong, Xinrun Wang, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2507.07445)  

**Abstract**: Autonomous agents navigating human society must master both production activities and social interactions, yet existing benchmarks rarely evaluate these skills simultaneously. To bridge this gap, we introduce StarDojo, a novel benchmark based on Stardew Valley, designed to assess AI agents in open-ended production-living simulations. In StarDojo, agents are tasked to perform essential livelihood activities such as farming and crafting, while simultaneously engaging in social interactions to establish relationships within a vibrant community. StarDojo features 1,000 meticulously curated tasks across five key domains: farming, crafting, exploration, combat, and social interactions. Additionally, we provide a compact subset of 100 representative tasks for efficient model evaluation. The benchmark offers a unified, user-friendly interface that eliminates the need for keyboard and mouse control, supports all major operating systems, and enables the parallel execution of multiple environment instances, making it particularly well-suited for evaluating the most capable foundation agents, powered by multimodal large language models (MLLMs). Extensive evaluations of state-of-the-art MLLMs agents demonstrate substantial limitations, with the best-performing model, GPT-4.1, achieving only a 12.7% success rate, primarily due to challenges in visual understanding, multimodal reasoning and low-level manipulation. As a user-friendly environment and benchmark, StarDojo aims to facilitate further research towards robust, open-ended agents in complex production-living environments. 

**Abstract (ZH)**: 自主导航人类社会的智能代理必须掌握生产活动和社会交往技能，而现有基准很少同时评估这些技能。为弥补这一差距，我们提出StarDojo，一个基于Stardew Valley的新颖基准，旨在评估AI代理在开放式的生产生活模拟中的能力。在StarDojo中，代理被指派执行如耕作和制作等基本生活活动，同时参与社交互动以在充满活力的社区中建立关系。StarDojo包含跨五大关键领域的1000个精心筛选的任务：耕作、制作、探索、战斗和社会交往。此外，我们还提供一个包含100个代表性任务的紧凑子集，以便高效地评估模型。该基准提供了一个统一且用户友好的界面，无需键盘和鼠标控制，支持所有主要操作系统，并允许多个环境实例并行执行，使其特别适用于评估由多模态大规模语言模型（MLLMs）驱动的最强大基础代理。对最先进的MLLMs代理的广泛评估表明，性能最佳的模型GPT-4.1的成功率仅为12.7%，主要原因是视觉理解、多模态推理和低级操作的挑战。作为一个用户友好的环境和基准，StarDojo旨在促进对在复杂生产生活环境中稳健、开放性代理的研究。 

---
# Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought 

**Title (ZH)**: 基于推理增强的多模态链式思维解码 

**Authors**: Shin'ya Yamaguchi, Kosuke Nishida, Daiki Chijiwa  

**Link**: [PDF](https://arxiv.org/pdf/2507.07685)  

**Abstract**: Large vision-language models (LVLMs) have demonstrated remarkable capabilities by integrating pre-trained vision encoders with large language models (LLMs). Similar to single-modal LLMs, chain-of-thought (CoT) prompting has been adapted for LVLMs to enhance multi-modal reasoning by generating intermediate rationales based on visual and textual inputs. While CoT is assumed to improve grounding and accuracy in LVLMs, our experiments reveal a key challenge: existing LVLMs often ignore the contents of generated rationales in CoT reasoning. To address this, we re-formulate multi-modal CoT reasoning as a KL-constrained reward maximization focused on rationale-conditional log-likelihood. As the optimal solution, we propose rationale-enhanced decoding (RED), a novel plug-and-play inference-time decoding strategy. RED harmonizes visual and rationale information by multiplying distinct image-conditional and rationale-conditional next token distributions. Extensive experiments show that RED consistently and significantly improves reasoning over standard CoT and other decoding methods across multiple benchmarks and LVLMs. Our work offers a practical and effective approach to improve both the faithfulness and accuracy of CoT reasoning in LVLMs, paving the way for more reliable rationale-grounded multi-modal systems. 

**Abstract (ZH)**: 大型多模态语言模型（LVLMs）通过结合预训练的视觉编码器和大型语言模型（LLMs）展现出了非凡的能力。类似于单一模态的LLMs，chain-of-thought（CoT）提示也被用于LVLMs，通过基于视觉和文本输入生成中间推理来增强多模态推理。尽管CoT被认为是提高LVLMs中的语义关联和准确性的方法，但我们的实验揭示了一个关键挑战：现有的LVLMs往往忽视CoT推理中生成的推理内容。为了解决这个问题，我们将多模态CoT推理重新表述为一个基于推理条件对数似然的KL约束下的奖励最大化问题。作为最优解，我们提出了推理增强解码（RED），这是一种新颖的即插即用的推理时解码策略。RED通过乘以不同的图像条件和推理条件的下一个词分布，协调视觉和推理信息。广泛的实验证明，RED在多个基准和LVLMs上一致且显著地改善了标准CoT和其它解码方法的推理能力。我们的工作提供了一种实用且有效的途径，以提高LVLMs中CoT推理的真实性和准确性，为更可靠的基于推理的多模态系统铺平了道路。 

---
# ArchiveGPT: A human-centered evaluation of using a vision language model for image cataloguing 

**Title (ZH)**: ArchiveGPT：基于视觉语言模型的图像分类目录构建的人本评估 

**Authors**: Line Abele, Gerrit Anders, Tolgahan Aydın, Jürgen Buder, Helen Fischer, Dominik Kimmel, Markus Huff  

**Link**: [PDF](https://arxiv.org/pdf/2507.07551)  

**Abstract**: The accelerating growth of photographic collections has outpaced manual cataloguing, motivating the use of vision language models (VLMs) to automate metadata generation. This study examines whether Al-generated catalogue descriptions can approximate human-written quality and how generative Al might integrate into cataloguing workflows in archival and museum collections. A VLM (InternVL2) generated catalogue descriptions for photographic prints on labelled cardboard mounts with archaeological content, evaluated by archive and archaeology experts and non-experts in a human-centered, experimental framework. Participants classified descriptions as AI-generated or expert-written, rated quality, and reported willingness to use and trust in AI tools. Classification performance was above chance level, with both groups underestimating their ability to detect Al-generated descriptions. OCR errors and hallucinations limited perceived quality, yet descriptions rated higher in accuracy and usefulness were harder to classify, suggesting that human review is necessary to ensure the accuracy and quality of catalogue descriptions generated by the out-of-the-box model, particularly in specialized domains like archaeological cataloguing. Experts showed lower willingness to adopt AI tools, emphasizing concerns on preservation responsibility over technical performance. These findings advocate for a collaborative approach where AI supports draft generation but remains subordinate to human verification, ensuring alignment with curatorial values (e.g., provenance, transparency). The successful integration of this approach depends not only on technical advancements, such as domain-specific fine-tuning, but even more on establishing trust among professionals, which could both be fostered through a transparent and explainable AI pipeline. 

**Abstract (ZH)**: 摄影收藏的快速增长已超出手工目录编排的能力，促使使用视觉语言模型（VLMs）自动化元数据生成。本研究探讨AI生成的目录描述能否接近人类撰写的质量，以及生成式AI如何融入档案和博物馆收藏的目录编排工作流程中。视觉语言模型（InternVL2）为带有考古内容的标Labels纸质照片生成目录描述，并由档案和考古专家及非专家在以人类为中心的实验框架下进行评估。参与者将描述分类为AI生成或专家撰写，评估质量，并报告使用和信任AI工具的意愿。分类性能超过随机水平，两组均低估了检测AI生成描述的能力。OCR错误和幻觉限制了感知质量，但准确性较高且更具实用性的描述更难以分类，表明需要人类审核以确保迁移到现成模型生成的目录描述的准确性和质量，特别是在考古目录编目等专业化领域。专家对采用AI工具的意愿较低，强调保护责任而非技术性能。这些发现倡导一种协作方式，在这种方式中，AI支持草稿生成但仍次于人类验证，以确保与策展价值（如来源、透明度）的对齐。这一方法的成功整合不仅取决于技术进步（如领域特定微调），更取决于在专业人士之间建立信任，而透明且可解释的AI管道可以促进这种信任的建立。 

---
# Toward Real-World Chinese Psychological Support Dialogues: CPsDD Dataset and a Co-Evolving Multi-Agent System 

**Title (ZH)**: 面向现实世界中文心理支持对话：CPsDD数据集与共生演化多Agent系统 

**Authors**: Yuanchen Shi, Longyin Zhang, Fang Kong  

**Link**: [PDF](https://arxiv.org/pdf/2507.07509)  

**Abstract**: The growing need for psychological support due to increasing pressures has exposed the scarcity of relevant datasets, particularly in non-English languages. To address this, we propose a framework that leverages limited real-world data and expert knowledge to fine-tune two large language models: Dialog Generator and Dialog Modifier. The Generator creates large-scale psychological counseling dialogues based on predefined paths, which guide system response strategies and user interactions, forming the basis for effective support. The Modifier refines these dialogues to align with real-world data quality. Through both automated and manual review, we construct the Chinese Psychological support Dialogue Dataset (CPsDD), containing 68K dialogues across 13 groups, 16 psychological problems, 13 causes, and 12 support focuses. Additionally, we introduce the Comprehensive Agent Dialogue Support System (CADSS), where a Profiler analyzes user characteristics, a Summarizer condenses dialogue history, a Planner selects strategies, and a Supporter generates empathetic responses. The experimental results of the Strategy Prediction and Emotional Support Conversation (ESC) tasks demonstrate that CADSS achieves state-of-the-art performance on both CPsDD and ESConv datasets. 

**Abstract (ZH)**: 由于不断增加的压力导致对心理支持的需求增长，暴露了相关数据集的稀缺性，尤其是在非英语语言中。为解决这一问题，我们提出了一种框架，利用有限的真实世界数据和专家知识对两种大型语言模型：对话生成器和对话修改器进行微调。生成器根据预定义的路径创建大规模的心理咨询对话，指导系统应答策略和用户交互，为基础有效支持提供支撑。修改器则调整这些对话以符合真实世界数据的质量。通过自动化和人工审核，我们构建了包含68,000对话的中文心理支持对话数据集（CPsDD），涵盖了13个组别、16种心理问题、13种原因和12种支持重点。此外，我们还引入了全面代理对话支持系统（CADSS），其中分析器分析用户特征，摘要器压缩对话历史，规划器选择策略，支持者生成同理心回应。策略预测和情感支持对话（ESC）任务的实验结果表明，CADSS在CPsDD和ESConv数据集上的性能达到最新水平。 

---
# Behave Your Motion: Habit-preserved Cross-category Animal Motion Transfer 

**Title (ZH)**: 保持行为：习惯保留的跨类别动物动作转移 

**Authors**: Zhimin Zhang, Bi'an Du, Caoyuan Ma, Zheng Wang, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.07394)  

**Abstract**: Animal motion embodies species-specific behavioral habits, making the transfer of motion across categories a critical yet complex task for applications in animation and virtual reality. Existing motion transfer methods, primarily focused on human motion, emphasize skeletal alignment (motion retargeting) or stylistic consistency (motion style transfer), often neglecting the preservation of distinct habitual behaviors in animals. To bridge this gap, we propose a novel habit-preserved motion transfer framework for cross-category animal motion. Built upon a generative framework, our model introduces a habit-preservation module with category-specific habit encoder, allowing it to learn motion priors that capture distinctive habitual characteristics. Furthermore, we integrate a large language model (LLM) to facilitate the motion transfer to previously unobserved species. To evaluate the effectiveness of our approach, we introduce the DeformingThings4D-skl dataset, a quadruped dataset with skeletal bindings, and conduct extensive experiments and quantitative analyses, which validate the superiority of our proposed model. 

**Abstract (ZH)**: 动物运动蕴含了物种特有的行为习惯，使得跨类别运动转移成为了动画和虚拟现实应用中一个关键而复杂的任务。现有的运动转移方法主要关注人类运动，侧重于骨骼对齐（运动目标映射）或风格一致性（运动风格转移），往往忽视了动物特有习惯行为的保持。为填补这一空白，我们提出了一种新的保留习惯行为的跨类别动物运动转移框架。基于生成框架，我们的模型引入了一个习惯保持模块，包含类别特定的习惯编码器，使其能够学习能够捕捉独特习惯特征的运动先验。此外，我们集成了一个大规模语言模型（LLM）以促进对未观察物种的运动转移。为了评估我们方法的有效性，我们引入了DeformingThings4D-skl数据集，这是一个带有骨骼绑定的四足动物数据集，并进行了广泛的实验和定量分析，验证了我们提出模型的优越性。 

---
# Multi-level Mixture of Experts for Multimodal Entity Linking 

**Title (ZH)**: 多层专家混合模型用于多模态实体链接 

**Authors**: Zhiwei Hu, Víctor Gutiérrez-Basulto, Zhiliang Xiang, Ru Li, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2507.07108)  

**Abstract**: Multimodal Entity Linking (MEL) aims to link ambiguous mentions within multimodal contexts to associated entities in a multimodal knowledge base. Existing approaches to MEL introduce multimodal interaction and fusion mechanisms to bridge the modality gap and enable multi-grained semantic matching. However, they do not address two important problems: (i) mention ambiguity, i.e., the lack of semantic content caused by the brevity and omission of key information in the mention's textual context; (ii) dynamic selection of modal content, i.e., to dynamically distinguish the importance of different parts of modal information. To mitigate these issues, we propose a Multi-level Mixture of Experts (MMoE) model for MEL. MMoE has four components: (i) the description-aware mention enhancement module leverages large language models to identify the WikiData descriptions that best match a mention, considering the mention's textual context; (ii) the multimodal feature extraction module adopts multimodal feature encoders to obtain textual and visual embeddings for both mentions and entities; (iii)-(iv) the intra-level mixture of experts and inter-level mixture of experts modules apply a switch mixture of experts mechanism to dynamically and adaptively select features from relevant regions of information. Extensive experiments demonstrate the outstanding performance of MMoE compared to the state-of-the-art. MMoE's code is available at: this https URL. 

**Abstract (ZH)**: 多模态实体链接（MEL）旨在将多模态上下文中的模糊提及与多模态知识库中的相关实体进行链接。现有的MEL方法引入了多模态交互和融合机制以弥合模态鸿沟，并实现多粒度语义匹配。然而，它们未能解决两个重要问题：（i）提及模糊性，即提及文本上下文中关键信息遗漏导致的语义内容不足；（ii）模态内容的动态选择，即区分不同模态信息部分的重要性。为缓解这些问题，我们提出了一种多层专家混合（MMoE）模型用于MEL。MMoE模型由四个组件组成：（i）描述感知的提及增强模块利用大语言模型识别与提及最匹配的WikiData描述，同时考虑提及的文本上下文；（ii）多模态特征提取模块采用多模态特征编码器为提及和实体获取文本和视觉嵌入；（iii）（iv）在同一层和跨层的专家混合模块中应用切换专家混合机制，以动态和适配地选择来自相关信息区域的特征。广泛实验证明，与现有方法相比，MMoE表现出色。MMoE的代码可在以下网址获取：this https URL。 

---
