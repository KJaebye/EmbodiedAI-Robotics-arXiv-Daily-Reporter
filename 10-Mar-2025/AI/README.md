# R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning 

**Title (ZH)**: R1-Searcher: 通过强化学习激励大语言模型的搜索能力 

**Authors**: Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2503.05592)  

**Abstract**: Existing Large Reasoning Models (LRMs) have shown the potential of reinforcement learning (RL) to enhance the complex reasoning capabilities of Large Language Models~(LLMs). While they achieve remarkable performance on challenging tasks such as mathematics and coding, they often rely on their internal knowledge to solve problems, which can be inadequate for time-sensitive or knowledge-intensive questions, leading to inaccuracies and hallucinations. To address this, we propose \textbf{R1-Searcher}, a novel two-stage outcome-based RL approach designed to enhance the search capabilities of LLMs. This method allows LLMs to autonomously invoke external search systems to access additional knowledge during the reasoning process. Our framework relies exclusively on RL, without requiring process rewards or distillation for a cold start. % effectively generalizing to out-of-domain datasets and supporting both Base and Instruct models. Our experiments demonstrate that our method significantly outperforms previous strong RAG methods, even when compared to the closed-source GPT-4o-mini. 

**Abstract (ZH)**: 现有的大型推理模型(LRMs)展示了强化学习(RL)在提升大型语言模型(LLMs)的复杂推理能力方面的潜力。尽管它们在数学和编程等挑战性任务上取得了显著性能，但在处理时间敏感或知识密集的问题时，常常依赖内部知识，这可能导致不准确性和幻觉。为了解决这一问题，我们提出了一种名为R1-Searcher的新型两阶段基于结果的RL方法，旨在增强LLMs的搜索能力。该方法使LLMs能够在推理过程中自主调用外部搜索系统以访问额外知识。我们的框架仅依赖于RL，无需过程奖励或冷启动的蒸馏。实验结果表明，我们的方法在性能上显著优于之前的强大检索增強方法，甚至优于闭源的GPT-4o-mini。 

---
# Ontology Generation using Large Language Models 

**Title (ZH)**: 基于大型语言模型的本体生成 

**Authors**: Anna Sofia Lippolis, Mohammad Javad Saeedizade, Robin Keskisärkkä, Sara Zuppiroli, Miguel Ceriani, Aldo Gangemi, Eva Blomqvist, Andrea Giovanni Nuzzolese  

**Link**: [PDF](https://arxiv.org/pdf/2503.05388)  

**Abstract**: The ontology engineering process is complex, time-consuming, and error-prone, even for experienced ontology engineers. In this work, we investigate the potential of Large Language Models (LLMs) to provide effective OWL ontology drafts directly from ontological requirements described using user stories and competency questions. Our main contribution is the presentation and evaluation of two new prompting techniques for automated ontology development: Memoryless CQbyCQ and Ontogenia. We also emphasize the importance of three structural criteria for ontology assessment, alongside expert qualitative evaluation, highlighting the need for a multi-dimensional evaluation in order to capture the quality and usability of the generated ontologies. Our experiments, conducted on a benchmark dataset of ten ontologies with 100 distinct CQs and 29 different user stories, compare the performance of three LLMs using the two prompting techniques. The results demonstrate improvements over the current state-of-the-art in LLM-supported ontology engineering. More specifically, the model OpenAI o1-preview with Ontogenia produces ontologies of sufficient quality to meet the requirements of ontology engineers, significantly outperforming novice ontology engineers in modelling ability. However, we still note some common mistakes and variability of result quality, which is important to take into account when using LLMs for ontology authoring support. We discuss these limitations and propose directions for future research. 

**Abstract (ZH)**: 大型语言模型在直接从用户故事和专业问题生成OWL本体草案中的潜力研究：Memoryless CQbyCQ和Ontogenia方法及其评估 

---
# VLMs Play StarCraft II: A Benchmark and Multimodal Decision Method 

**Title (ZH)**: VLMs在StarCraft II中的应用：一个基准和多模态决策方法 

**Authors**: Weiyu Ma, Yuqian Fu, Zecheng Zhang, Guohao Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.05383)  

**Abstract**: We introduce VLM-Attention, a multimodal StarCraft II environment that aligns artificial agent perception with the human gameplay experience. Traditional frameworks such as SMAC rely on abstract state representations that diverge significantly from human perception, limiting the ecological validity of agent behavior. Our environment addresses this limitation by incorporating RGB visual inputs and natural language observations that more closely simulate human cognitive processes during gameplay. The VLM-Attention framework consists of three integrated components: (1) a vision-language model enhanced with specialized self-attention mechanisms for strategic unit targeting and battlefield assessment, (2) a retrieval-augmented generation system that leverages domain-specific StarCraft II knowledge to inform tactical decisions, and (3) a dynamic role-based task distribution system that enables coordinated multi-agent behavior. Our experimental evaluation across 21 custom scenarios demonstrates that VLM-based agents powered by foundation models (specifically Qwen-VL and GPT-4o) can execute complex tactical maneuvers without explicit training, achieving comparable performance to traditional MARL methods that require substantial training iterations. This work establishes a foundation for developing human-aligned StarCraft II agents and advances the broader research agenda of multimodal game AI. Our implementation is available at this https URL. 

**Abstract (ZH)**: 我们引入了VLM-Attention，这是一种 multimodal StarCraft II 环境，将人工代理的感知与人类游戏体验对齐。传统的框架如SMAC依赖于与人类感知差异较大的抽象状态表示，限制了代理行为的生态有效性。我们的环境通过引入RGB视觉输入和自然语言观察来解决这一局限性，这些观察更贴近于游戏过程中人类的认知过程。VLM-Attention框架包括三个集成组件：（1）配备专门自注意力机制的视觉-语言模型，用于战略单位目标定位和战场评估；（2）利用特定领域StarCraft II知识的检索增强生成系统，以指导战术决策；（3）一种动态角色任务分配系统，以实现协调的多代理行为。我们在21个定制场景的实验评估表明，基于基础模型（具体为Qwen-VL和GPT-4o）的VLM代理可以执行复杂的战术操作，无需显式训练，达到与需要大量训练迭代的传统MARL方法相当的性能。这项工作为开发与人类对齐的StarCraft II代理奠定了基础，并推动了多模态游戏AI的更广泛研究议程。我们的实现可在此处访问：this https URL。 

---
# Toward an Evaluation Science for Generative AI Systems 

**Title (ZH)**: 生成式AI系统评估科学探讨 

**Authors**: Laura Weidinger, Deb Raji, Hanna Wallach, Margaret Mitchell, Angelina Wang, Olawale Salaudeen, Rishi Bommasani, Sayash Kapoor, Deep Ganguli, Sanmi Koyejo, William Isaac  

**Link**: [PDF](https://arxiv.org/pdf/2503.05336)  

**Abstract**: There is an increasing imperative to anticipate and understand the performance and safety of generative AI systems in real-world deployment contexts. However, the current evaluation ecosystem is insufficient: Commonly used static benchmarks face validity challenges, and ad hoc case-by-case audits rarely scale. In this piece, we advocate for maturing an evaluation science for generative AI systems. While generative AI creates unique challenges for system safety engineering and measurement science, the field can draw valuable insights from the development of safety evaluation practices in other fields, including transportation, aerospace, and pharmaceutical engineering. In particular, we present three key lessons: Evaluation metrics must be applicable to real-world performance, metrics must be iteratively refined, and evaluation institutions and norms must be established. Applying these insights, we outline a concrete path toward a more rigorous approach for evaluating generative AI systems. 

**Abstract (ZH)**: 生成式AI系统在实际部署中的性能与安全预测及理解日益紧迫：当前的评估生态系统存在不足：常用的静态基准面临有效性挑战，而逐案审计很少能扩大规模。本文倡导成熟生成式AI系统的评估科学。尽管生成式AI为系统安全性工程和测量科学带来了独特挑战，但该领域可以从其他领域（如交通、航空和制药工程）的安全评估实践中获得宝贵启示。特别是，我们提出了三条关键教训：评估指标必须适用于实际性能，指标必须逐步精炼，并需建立评估机构和规范。应用这些启示，我们概述了一条更加严谨的评估生成式AI系统的具体路径。 

---
# WritingBench: A Comprehensive Benchmark for Generative Writing 

**Title (ZH)**: WritingBench: 生成写作的综合基准 

**Authors**: Yuning Wu, Jiahao Mei, Ming Yan, Chenliang Li, SHaopeng Lai, Yuran Ren, Zijia Wang, Ji Zhang, Mengyue Wu, Qin Jin, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05244)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly enhanced text generation capabilities, yet evaluating their performance in generative writing remains a challenge. Existing benchmarks primarily focus on generic text generation or limited in writing tasks, failing to capture the diverse requirements of high-quality written contents across various domains. To bridge this gap, we present WritingBench, a comprehensive benchmark designed to evaluate LLMs across 6 core writing domains and 100 subdomains, encompassing creative, persuasive, informative, and technical writing. We further propose a query-dependent evaluation framework that empowers LLMs to dynamically generate instance-specific assessment criteria. This framework is complemented by a fine-tuned critic model for criteria-aware scoring, enabling evaluations in style, format and length. The framework's validity is further demonstrated by its data curation capability, which enables 7B-parameter models to approach state-of-the-art (SOTA) performance. We open-source the benchmark, along with evaluation tools and modular framework components, to advance the development of LLMs in writing. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs）显著提升了文本生成能力，但评估其在生成性写作中的表现依然颇具挑战。现有基准主要集中在通用文本生成或局限于某些写作任务，未能捕捉各领域高质量书面内容的多样化需求。为弥补这一不足，我们提出了WritingBench，这是一个综合基准，旨在评估LLMs在6个核心写作领域和100个子领域中的表现，涵盖创造性、 persuasiveness、信息性和技术性写作。我们还提出了一个查询依赖的评估框架，使LLMs能够动态生成实例特定的评估标准。该框架结合了一个细调的评论者模型，用于根据标准评分，从而实现风格、格式和长度的评估。框架的有效性通过其数据整理能力得以证明，使7B参数模型接近了当前最佳表现（SOTA）。我们开源了该基准以及评估工具和模块化框架组件，以促进大型语言模型在写作方面的开发。 

---
# Path Pooling: Train-Free Structure Enhancement for Efficient Knowledge Graph Retrieval-Augmented Generation 

**Title (ZH)**: 路径聚合：高效的知识图谱检索增强生成中的无训练结构增强 

**Authors**: Hairu Wang, Yuan Feng, Xike Xie, S Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.05203)  

**Abstract**: Although Large Language Models achieve strong success in many tasks, they still suffer from hallucinations and knowledge deficiencies in real-world applications. Many knowledge graph-based retrieval-augmented generation (KG-RAG) methods enhance the quality and credibility of LLMs by leveraging structure and semantic information in KGs as external knowledge bases. However, these methods struggle to effectively incorporate structure information, either incurring high computational costs or underutilizing available knowledge. Inspired by smoothing operations in graph representation learning, we propose path pooling, a simple, train-free strategy that introduces structure information through a novel path-centric pooling operation. It seamlessly integrates into existing KG-RAG methods in a plug-and-play manner, enabling richer structure information utilization. Extensive experiments demonstrate that incorporating the path pooling into the state-of-the-art KG-RAG method consistently improves performance across various settings while introducing negligible additional cost. Code is coming soon at this https URL. 

**Abstract (ZH)**: 尽管大型语言模型在许多任务中取得了显著的成功，但在实际应用中仍存在幻觉和知识不足的问题。基于知识图谱的检索增强生成（KG-RAG）方法通过利用知识图谱中结构和语义信息作为外部知识库，增强了语言模型的质量和可靠性。然而，这些方法在有效整合结构信息方面存在困难，要么导致高计算成本，要么未能充分利用可用的知识。受到图表示学习中平滑操作的启发，我们提出了一种简单的、无需训练的策略——路径池化，通过一个新颖的路径为中心的池化操作引入结构信息。该方法以即插即用的方式无缝集成到现有的KG-RAG方法中，能够更好地利用结构信息。大量实验表明，在最先进的KG-RAG方法中引入路径池化可以在各种场景中一致地提高性能，同时几乎不增加额外成本。代码 shortly 将在以下链接发布：this https URL。 

---
# FedMABench: Benchmarking Mobile Agents on Decentralized Heterogeneous User Data 

**Title (ZH)**: FedMABench：在分布式异构用户数据上的移动代理基准测试 

**Authors**: Wenhao Wang, Zijie Yu, Rui Ye, Jianqing Zhang, Siheng Chen, Yanfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05143)  

**Abstract**: Mobile agents have attracted tremendous research participation recently. Traditional approaches to mobile agent training rely on centralized data collection, leading to high cost and limited scalability. Distributed training utilizing federated learning offers an alternative by harnessing real-world user data, providing scalability and reducing costs. However, pivotal challenges, including the absence of standardized benchmarks, hinder progress in this field.
To tackle the challenges, we introduce FedMABench, the first benchmark for federated training and evaluation of mobile agents, specifically designed for heterogeneous scenarios. FedMABench features 6 datasets with 30+ subsets, 8 federated algorithms, 10+ base models, and over 800 apps across 5 categories, providing a comprehensive framework for evaluating mobile agents across diverse environments. Through extensive experiments, we uncover several key insights: federated algorithms consistently outperform local training; the distribution of specific apps plays a crucial role in heterogeneity; and, even apps from distinct categories can exhibit correlations during training. FedMABench is publicly available at: this https URL with the datasets at: this https URL. 

**Abstract (ZH)**: 移动代理吸引了大量的研究参与。传统移动代理训练方法依赖于集中式数据收集，导致成本高且可扩展性有限。利用联邦学习进行分布式训练提供了一种替代方案，通过利用真实世界用户数据，实现了可扩展性和成本降低。然而，缺乏标准化基准数据阻碍了该领域的发展。

为应对这些挑战，我们提出FedMABench，这是首个针对异构场景下移动代理联邦训练与评估的标准基准。FedMABench包括6个数据集（30多个子集）、8种联邦算法、10多种基础模型以及超过800个应用（涵盖5个类别），提供了一个全面的框架，用于评估不同环境下的移动代理。通过大量实验，我们揭示了几个关键洞察：联邦算法始终优于局部训练；特定应用的分布对于异构性至关重要；即使来自不同类别的应用在训练中也可能表现出相关性。FedMABench已公开发布：[这里](this https URL)，数据集获取地址：[这里](this https URL)。 

---
# R1-Zero's "Aha Moment" in Visual Reasoning on a 2B Non-SFT Model 

**Title (ZH)**: R1-Zero在视觉推理中的“恍然大悟”时刻：一个2B非SFT模型的研究 

**Authors**: Hengguang Zhou, Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, Cho-Jui Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2503.05132)  

**Abstract**: Recently DeepSeek R1 demonstrated how reinforcement learning with simple rule-based incentives can enable autonomous development of complex reasoning in large language models, characterized by the "aha moment", in which the model manifest self-reflection and increased response length during training. However, attempts to extend this success to multimodal reasoning often failed to reproduce these key characteristics. In this report, we present the first successful replication of these emergent characteristics for multimodal reasoning on only a non-SFT 2B model. Starting with Qwen2-VL-2B and applying reinforcement learning directly on the SAT dataset, our model achieves 59.47% accuracy on CVBench, outperforming the base model by approximately ~30% and exceeding both SFT setting by ~2%. In addition, we share our failed attempts and insights in attempting to achieve R1-like reasoning using RL with instruct models. aiming to shed light on the challenges involved. Our key observations include: (1) applying RL on instruct model often results in trivial reasoning trajectories, and (2) naive length reward are ineffective in eliciting reasoning capabilities. The project code is available at this https URL 

**Abstract (ZH)**: 最近，DeepSeek R1证明了使用基于规则的激励与强化学习相结合可以促使大型语言模型在训练中自主发展出具有“啊哈时刻”的复杂推理能力。然而，将这一成功扩展到多模态推理时，往往无法再现这些关键特征。在本报告中，我们首次成功实现了仅在非SFT 2B模型上复制这些新兴特征的多模态推理。从Qwen2-VL-2B出发，直接在SAT数据集上应用强化学习，我们的模型在CVBench上的准确率达到59.47%，比基线模型高出约30%，同时超越SFT设置约2%。此外，我们分享了尝试使用指令模型通过RL实现类似R1的推理的失败尝试与见解，以揭示其中的挑战。我们的主要观察结果包括：(1) 在指令模型上应用RL通常会导致简单的推理轨迹，(2) 粗糙的长度奖励在激发推理能力方面无效。项目代码可在此处访问：这个 https URL。 

---
# INTENT: Trajectory Prediction Framework with Intention-Guided Contrastive Clustering 

**Title (ZH)**: 意图引导对比聚类的轨迹预测框架 

**Authors**: Yihong Tang, Wei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.04952)  

**Abstract**: Accurate trajectory prediction of road agents (e.g., pedestrians, vehicles) is an essential prerequisite for various intelligent systems applications, such as autonomous driving and robotic navigation. Recent research highlights the importance of environmental contexts (e.g., maps) and the "multi-modality" of trajectories, leading to increasingly complex model structures. However, real-world deployments require lightweight models that can quickly migrate and adapt to new environments. Additionally, the core motivations of road agents, referred to as their intentions, deserves further exploration. In this study, we advocate that understanding and reasoning road agents' intention plays a key role in trajectory prediction tasks, and the main challenge is that the concept of intention is fuzzy and abstract. To this end, we present INTENT, an efficient intention-guided trajectory prediction model that relies solely on information contained in the road agent's trajectory. Our model distinguishes itself from existing models in several key aspects: (i) We explicitly model road agents' intentions through contrastive clustering, accommodating the fuzziness and abstraction of human intention in their trajectories. (ii) The proposed INTENT is based solely on multi-layer perceptrons (MLPs), resulting in reduced training and inference time, making it very efficient and more suitable for real-world deployment. (iii) By leveraging estimated intentions and an innovative algorithm for transforming trajectory observations, we obtain more robust trajectory representations that lead to superior prediction accuracy. Extensive experiments on real-world trajectory datasets for pedestrians and autonomous vehicles demonstrate the effectiveness and efficiency of INTENT. 

**Abstract (ZH)**: 基于路径意图引导的道路上交通参与者的准确轨迹预测 

---
# Multi-Fidelity Policy Gradient Algorithms 

**Title (ZH)**: 多保真度策略梯度算法 

**Authors**: Xinjie Liu, Cyrus Neary, Kushagra Gupta, Christian Ellis, Ufuk Topcu, David Fridovich-Keil  

**Link**: [PDF](https://arxiv.org/pdf/2503.05696)  

**Abstract**: Many reinforcement learning (RL) algorithms require large amounts of data, prohibiting their use in applications where frequent interactions with operational systems are infeasible, or high-fidelity simulations are expensive or unavailable. Meanwhile, low-fidelity simulators--such as reduced-order models, heuristic reward functions, or generative world models--can cheaply provide useful data for RL training, even if they are too coarse for direct sim-to-real transfer. We propose multi-fidelity policy gradients (MFPGs), an RL framework that mixes a small amount of data from the target environment with a large volume of low-fidelity simulation data to form unbiased, reduced-variance estimators (control variates) for on-policy policy gradients. We instantiate the framework by developing multi-fidelity variants of two policy gradient algorithms: REINFORCE and proximal policy optimization. Experimental results across a suite of simulated robotics benchmark problems demonstrate that when target-environment samples are limited, MFPG achieves up to 3.9x higher reward and improves training stability when compared to baselines that only use high-fidelity data. Moreover, even when the baselines are given more high-fidelity samples--up to 10x as many interactions with the target environment--MFPG continues to match or outperform them. Finally, we observe that MFPG is capable of training effective policies even when the low-fidelity environment is drastically different from the target environment. MFPG thus not only offers a novel paradigm for efficient sim-to-real transfer but also provides a principled approach to managing the trade-off between policy performance and data collection costs. 

**Abstract (ZH)**: 多保真度策略梯度：一种将目标环境数据与低保真度模拟数据相结合的强化学习框架 

---
# BEHAVIOR Robot Suite: Streamlining Real-World Whole-Body Manipulation for Everyday Household Activities 

**Title (ZH)**: BEHAVIOR 机器人套件：简化日常家庭活动中的全身操纵 

**Authors**: Yunfan Jiang, Ruohan Zhang, Josiah Wong, Chen Wang, Yanjie Ze, Hang Yin, Cem Gokmen, Shuran Song, Jiajun Wu, Li Fei-Fei  

**Link**: [PDF](https://arxiv.org/pdf/2503.05652)  

**Abstract**: Real-world household tasks present significant challenges for mobile manipulation robots. An analysis of existing robotics benchmarks reveals that successful task performance hinges on three key whole-body control capabilities: bimanual coordination, stable and precise navigation, and extensive end-effector reachability. Achieving these capabilities requires careful hardware design, but the resulting system complexity further complicates visuomotor policy learning. To address these challenges, we introduce the BEHAVIOR Robot Suite (BRS), a comprehensive framework for whole-body manipulation in diverse household tasks. Built on a bimanual, wheeled robot with a 4-DoF torso, BRS integrates a cost-effective whole-body teleoperation interface for data collection and a novel algorithm for learning whole-body visuomotor policies. We evaluate BRS on five challenging household tasks that not only emphasize the three core capabilities but also introduce additional complexities, such as long-range navigation, interaction with articulated and deformable objects, and manipulation in confined spaces. We believe that BRS's integrated robotic embodiment, data collection interface, and learning framework mark a significant step toward enabling real-world whole-body manipulation for everyday household tasks. BRS is open-sourced at this https URL 

**Abstract (ZH)**: 家庭任务中的实际挑战为移动操作机器人带来了重大困难。现有的机器人基准分析表明，任务的成功执行取决于三项核心的整体身体控制能力：双臂协调、稳定而精确的导航以及广泛的末端执行器可达性。实现这些能力需要仔细设计硬件，但由此产生的系统复杂性进一步 complicates 视觉运动策略学习。为了应对这些挑战，我们引入了 BEHAVIOR 机器人套件（BRS），这是一个全面的家庭任务复杂操作框架。BRS 以双臂轮式机器人和4自由度躯干为基础，集成了一个经济高效的全身远程操作接口以收集数据，并提出了一种新的算法来学习全身视觉运动策略。我们在五个具有挑战性的家庭任务中评估了 BRS，这些任务不仅强调了三个核心能力，还引入了额外的复杂性，如长距离导航、与关节和变形物体的交互以及在受限空间中的操作。我们相信，BRS 的集成机器人实体、数据采集接口和学习框架标志着向实现日常生活家庭任务中全身操作能力迈出的重要一步。BRS 已开源，参见 this https URL。 

---
# dARt Vinci: Egocentric Data Collection for Surgical Robot Learning at Scale 

**Title (ZH)**: 达芬奇_egocentric数据采集：面向手术机器人规模化学习 

**Authors**: Yihao Liu, Yu-Chun Ku, Jiaming Zhang, Hao Ding, Peter Kazanzides, Mehran Armand  

**Link**: [PDF](https://arxiv.org/pdf/2503.05646)  

**Abstract**: Data scarcity has long been an issue in the robot learning community. Particularly, in safety-critical domains like surgical applications, obtaining high-quality data can be especially difficult. It poses challenges to researchers seeking to exploit recent advancements in reinforcement learning and imitation learning, which have greatly improved generalizability and enabled robots to conduct tasks autonomously. We introduce dARt Vinci, a scalable data collection platform for robot learning in surgical settings. The system uses Augmented Reality (AR) hand tracking and a high-fidelity physics engine to capture subtle maneuvers in primitive surgical tasks: By eliminating the need for a physical robot setup and providing flexibility in terms of time, space, and hardware resources-such as multiview sensors and actuators-specialized simulation is a viable alternative. At the same time, AR allows the robot data collection to be more egocentric, supported by its body tracking and content overlaying capabilities. Our user study confirms the proposed system's efficiency and usability, where we use widely-used primitive tasks for training teleoperation with da Vinci surgical robots. Data throughput improves across all tasks compared to real robot settings by 41% on average. The total experiment time is reduced by an average of 10%. The temporal demand in the task load survey is improved. These gains are statistically significant. Additionally, the collected data is over 400 times smaller in size, requiring far less storage while achieving double the frequency. 

**Abstract (ZH)**: 数据稀缺一直是机器人学习领域的一个问题。特别是在如手术应用这样的安全关键领域，获取高质量的数据尤为困难。这给致力于利用强化学习和imitation learning等最近进步的研究人员带来了挑战，这些进步极大地提高了泛化能力并使机器人能够自主执行任务。我们提出了dARt Vinci，一种适用于手术场景的机器人学习数据收集平台。该系统利用增强现实（AR）手部追踪和高度逼真的物理引擎来捕捉原始手术任务中的微妙操作：通过消除物理机器人设置的需要，并在时间、空间和硬件资源（如多视角传感器和执行器）方面提供灵活性，专门的模拟成为一种可行的替代方案。同时，AR使机器人数据收集更加以自我为中心，其身体追踪和内容叠加能力为此提供了支持。我们的用户研究证实了该系统的效率和易用性，我们使用广泛采用的原始任务来训练与da Vinci手术机器人相关的远程操作。与实际机器人设置相比，所有任务的数据传输速度平均提高了41%。实验总时间平均减少了10%。任务负载调查中的时间需求也得到了改善。这些收益具有统计显著性。此外，收集的数据大小超过400倍较小，占用的存储空间大大减少，但同时实现了两倍的频率。 

---
# Symbolic Mixture-of-Experts: Adaptive Skill-based Routing for Heterogeneous Reasoning 

**Title (ZH)**: 符号混合专家：异构推理的自适应技能路由 

**Authors**: Justin Chih-Yao Chen, Sukwon Yun, Elias Stengel-Eskin, Tianlong Chen, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2503.05641)  

**Abstract**: Combining existing pre-trained expert LLMs is a promising avenue for scalably tackling large-scale and diverse tasks. However, selecting experts at the task level is often too coarse-grained, as heterogeneous tasks may require different expertise for each instance. To enable adaptive instance-level mixing of pre-trained LLM experts, we propose Symbolic-MoE, a symbolic, text-based, and gradient-free Mixture-of-Experts framework. Symbolic-MoE takes a fine-grained approach to selection by emphasizing skills, e.g., algebra in math or molecular biology in biomedical reasoning. We propose a skill-based recruiting strategy that dynamically selects the most relevant set of expert LLMs for diverse reasoning tasks based on their strengths. Each selected expert then generates its own reasoning, resulting in k outputs from k experts, which are then synthesized into a final high-quality response by an aggregator chosen based on its ability to integrate diverse reasoning outputs. We show that Symbolic-MoE's instance-level expert selection improves performance by a large margin but -- when implemented naively -- can introduce a high computational overhead due to the need for constant model loading and offloading. To address this, we implement a batch inference strategy that groups instances based on their assigned experts, loading each model only once. This allows us to integrate 16 expert models on 1 GPU with a time cost comparable to or better than prior multi-agent baselines using 4 GPUs. Through extensive evaluations on diverse benchmarks (MMLU-Pro, GPQA, AIME, and MedMCQA), we demonstrate that Symbolic-MoE outperforms strong LLMs like GPT4o-mini, as well as multi-agent approaches, with an absolute average improvement of 8.15% over the best multi-agent baseline. Moreover, Symbolic-MoE removes the need for expensive multi-round discussions, outperforming discussion baselines with less computation. 

**Abstract (ZH)**: 结合现有的预训练专家大规模语言模型是一种有望应对大规模和多样化任务的途径。然而，任务级别的专家选择往往过于粗粒度，因为异构任务可能需要每个实例有不同的专长。为了实现预训练语言模型专家的自适应实例级混合，我们提出了一种符号混合-of-专家（Symbolic-MoE）框架，该框架是符号的、基于文本的和无需梯度的混合-of-专家框架。Symbolic-MoE 通过强调技能，如数学中的代数或生物医学推理中的分子生物学，采取了细粒度的选择方法。我们提出了一种基于技能的招聘策略，该策略根据专家的长处动态选择最适合多种推理任务的专家集。每个选定的专家生成其自的推理，从而产生来自k个专家的k个输出，这些输出然后由根据其整合多样推理输出的能力选择的聚合器进行综合生成最终的高质量响应。我们展示了Symbolic-MoE的实例级专家选择能够显著提升性能，但未经优化时可能会因需要不断加载和卸载模型而引入高计算开销。为了解决这个问题，我们实施了批量推理策略，根据分配的专家对实例进行分组，并且只加载每个模型一次。这使得我们能够在1个GPU上整合16个专家模型，计算成本与4个GPU的多智能体基线相当或更优。通过在多样基准（MMLU-Pro、GPQA、AIME和MedMCQA）上的 extensive 评估，我们证明了Symbolic-MoE 在性能上优于强大的语言模型如GPT4o-mini，以及多智能体方法，绝对平均改进率为8.15%，并且优于最好的多智能体基线。此外，Symbolic-MoE 消除了昂贵的多轮讨论需求，并在计算开销较少的情况下优于讨论基线。 

---
# VideoPainter: Any-length Video Inpainting and Editing with Plug-and-Play Context Control 

**Title (ZH)**: VideoPainter：基于即用型上下文控制的任意长度视频修复与编辑 

**Authors**: Yuxuan Bian, Zhaoyang Zhang, Xuan Ju, Mingdeng Cao, Liangbin Xie, Ying Shan, Qiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.05639)  

**Abstract**: Video inpainting, which aims to restore corrupted video content, has experienced substantial progress. Despite these advances, existing methods, whether propagating unmasked region pixels through optical flow and receptive field priors, or extending image-inpainting models temporally, face challenges in generating fully masked objects or balancing the competing objectives of background context preservation and foreground generation in one model, respectively. To address these limitations, we propose a novel dual-stream paradigm VideoPainter that incorporates an efficient context encoder (comprising only 6% of the backbone parameters) to process masked videos and inject backbone-aware background contextual cues to any pre-trained video DiT, producing semantically consistent content in a plug-and-play manner. This architectural separation significantly reduces the model's learning complexity while enabling nuanced integration of crucial background context. We also introduce a novel target region ID resampling technique that enables any-length video inpainting, greatly enhancing our practical applicability. Additionally, we establish a scalable dataset pipeline leveraging current vision understanding models, contributing VPData and VPBench to facilitate segmentation-based inpainting training and assessment, the largest video inpainting dataset and benchmark to date with over 390K diverse clips. Using inpainting as a pipeline basis, we also explore downstream applications including video editing and video editing pair data generation, demonstrating competitive performance and significant practical potential. Extensive experiments demonstrate VideoPainter's superior performance in both any-length video inpainting and editing, across eight key metrics, including video quality, mask region preservation, and textual coherence. 

**Abstract (ZH)**: 视频修复：一种新颖的双流 paradigm VideoPainter 及其实用应用 

---
# TrajectoryCrafter: Redirecting Camera Trajectory for Monocular Videos via Diffusion Models 

**Title (ZH)**: TrajectoryCrafter: 通过扩散模型重定向单目视频摄像机轨迹 

**Authors**: Mark YU, Wenbo Hu, Jinbo Xing, Ying Shan  

**Link**: [PDF](https://arxiv.org/pdf/2503.05638)  

**Abstract**: We present TrajectoryCrafter, a novel approach to redirect camera trajectories for monocular videos. By disentangling deterministic view transformations from stochastic content generation, our method achieves precise control over user-specified camera trajectories. We propose a novel dual-stream conditional video diffusion model that concurrently integrates point cloud renders and source videos as conditions, ensuring accurate view transformations and coherent 4D content generation. Instead of leveraging scarce multi-view videos, we curate a hybrid training dataset combining web-scale monocular videos with static multi-view datasets, by our innovative double-reprojection strategy, significantly fostering robust generalization across diverse scenes. Extensive evaluations on multi-view and large-scale monocular videos demonstrate the superior performance of our method. 

**Abstract (ZH)**: 我们提出了TrajectoryCrafter，一种用于单目视频的新型相机轨迹重定向方法。通过解耦确定性的视角转换和随机的内容生成，我们的方法实现了对用户指定相机轨迹的精确控制。我们提出了一种新颖的双流条件视频扩散模型，该模型可以同时整合点云渲染和源视频作为条件，确保准确的视角转换和连贯的4D内容生成。我们不依赖稀少的多视角视频，而是通过我们创新的双重投影策略，将网络规模的单目视频与静态多视角数据集结合，构建混合训练数据集，显著增强了方法在多样场景下的泛化能力。在多视角和大规模单目视频上的广泛评估证明了该方法的优越性能。 

---
# Exploring FMCW Radars and Feature Maps for Activity Recognition: A Benchmark Study 

**Title (ZH)**: 探索FMCW雷达与特征图在活动识别中的应用：一项基准研究 

**Authors**: Ali Samimi Fard, Mohammadreza Mashhadigholamali, Samaneh Zolfaghari, Hajar Abedi, Mainak Chakraborty, Luigi Borzì, Masoud Daneshtalab, George Shaker  

**Link**: [PDF](https://arxiv.org/pdf/2503.05629)  

**Abstract**: Human Activity Recognition has gained significant attention due to its diverse applications, including ambient assisted living and remote sensing. Wearable sensor-based solutions often suffer from user discomfort and reliability issues, while video-based methods raise privacy concerns and perform poorly in low-light conditions or long ranges. This study introduces a Frequency-Modulated Continuous Wave radar-based framework for human activity recognition, leveraging a 60 GHz radar and multi-dimensional feature maps. Unlike conventional approaches that process feature maps as images, this study feeds multi-dimensional feature maps -- Range-Doppler, Range-Azimuth, and Range-Elevation -- as data vectors directly into the machine learning (SVM, MLP) and deep learning (CNN, LSTM, ConvLSTM) models, preserving the spatial and temporal structures of the data. These features were extracted from a novel dataset with seven activity classes and validated using two different validation approaches. The ConvLSTM model outperformed conventional machine learning and deep learning models, achieving an accuracy of 90.51% and an F1-score of 87.31% on cross-scene validation and an accuracy of 89.56% and an F1-score of 87.15% on leave-one-person-out cross-validation. The results highlight the approach's potential for scalable, non-intrusive, and privacy-preserving activity monitoring in real-world scenarios. 

**Abstract (ZH)**: 基于频段调制连续波雷达的人体活动识别框架：一种多维特征图在机器学习和深度学习中的应用 

---
# Superintelligence Strategy: Expert Version 

**Title (ZH)**: 超人工智能策略：专家版 

**Authors**: Dan Hendrycks, Eric Schmidt, Alexandr Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05628)  

**Abstract**: Rapid advances in AI are beginning to reshape national security. Destabilizing AI developments could rupture the balance of power and raise the odds of great-power conflict, while widespread proliferation of capable AI hackers and virologists would lower barriers for rogue actors to cause catastrophe. Superintelligence -- AI vastly better than humans at nearly all cognitive tasks -- is now anticipated by AI researchers. Just as nations once developed nuclear strategies to secure their survival, we now need a coherent superintelligence strategy to navigate a new period of transformative change. We introduce the concept of Mutual Assured AI Malfunction (MAIM): a deterrence regime resembling nuclear mutual assured destruction (MAD) where any state's aggressive bid for unilateral AI dominance is met with preventive sabotage by rivals. Given the relative ease of sabotaging a destabilizing AI project -- through interventions ranging from covert cyberattacks to potential kinetic strikes on datacenters -- MAIM already describes the strategic picture AI superpowers find themselves in. Alongside this, states can increase their competitiveness by bolstering their economies and militaries through AI, and they can engage in nonproliferation to rogue actors to keep weaponizable AI capabilities out of their hands. Taken together, the three-part framework of deterrence, nonproliferation, and competitiveness outlines a robust strategy to superintelligence in the years ahead. 

**Abstract (ZH)**: AI快速发展开始重塑国家安全。不稳定的人工智能发展可能破坏权力平衡并增加大国冲突的可能性，而人工智能黑客和病毒学家的广泛扩散会降低流氓行为体引发灾难的门槛。超级智能——在几乎所有认知任务上远远超过人类的人工智能——现已为人工智能研究人员所预期。正如各国曾经制定核策略以确保自身的生存一样，我们现在需要一个连贯的超级人工智能战略来应对一次变革性变革时期的到来。我们提出了相互确保人工智能故障(MAIM)的概念：一种类似于核相互确保摧毁(MAD)的威慑机制，任何国家试图单方面追求人工智能主导地位的行为都可能受到对手的预防性破坏。鉴于对不稳定人工智能项目的破坏相对容易——从隐蔽网络攻击到可能的数据中心物理打击等干预措施——MAIM已经描述了人工智能超级大国所面临的战略格局。同时，各国可以通过增强经济和军事来提高竞争力，并通过非扩散努力防止将武器化的人工智能能力落入流氓行为体之手。整体来看，威慑、非扩散和竞争力三部分框架概述了在未来几年应对超级智能的稳健策略。 

---
# FMT:A Multimodal Pneumonia Detection Model Based on Stacking MOE Framework 

**Title (ZH)**: FMT：基于MOE框架的多模态肺炎检测模型 

**Authors**: Jingyu Xu, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05626)  

**Abstract**: Artificial intelligence has shown the potential to improve diagnostic accuracy through medical image analysis for pneumonia diagnosis. However, traditional multimodal approaches often fail to address real-world challenges such as incomplete data and modality loss. In this study, a Flexible Multimodal Transformer (FMT) was proposed, which uses ResNet-50 and BERT for joint representation learning, followed by a dynamic masked attention strategy that simulates clinical modality loss to improve robustness; finally, a sequential mixture of experts (MOE) architecture was used to achieve multi-level decision refinement. After evaluation on a small multimodal pneumonia dataset, FMT achieved state-of-the-art performance with 94% accuracy, 95% recall, and 93% F1 score, outperforming single-modal baselines (ResNet: 89%; BERT: 79%) and the medical benchmark CheXMed (90%), providing a scalable solution for multimodal diagnosis of pneumonia in resource-constrained medical settings. 

**Abstract (ZH)**: 人工智能在通过肺炎诊断的医学图像分析中展示了提升诊断准确性潜力，然而传统多模态方法往往无法解决实际挑战如数据不完整和模态丢失。在此研究中，提出了一种灵活的多模态变换器（FMT），使用ResNet-50和BERT进行联合表示学习，随后采用动态遮蔽注意机制以模拟临床模态丢失从而提高鲁棒性；最后，采用层次专家混合架构实现多重决策精细化。经过对小型多模态肺炎数据集的评估，FMT 达到了94%的准确率、95%的召回率和93%的F1分数，超越了单模态基线（ResNet: 89%; BERT: 79%）和医学基准CheXMed（90%），提供了资源受限医疗环境中多模态肺炎诊断的可扩展解决方案。 

---
# Learning LLM Preference over Intra-Dialogue Pairs: A Framework for Utterance-level Understandings 

**Title (ZH)**: 学习大规模语言模型在对话内部配对中的偏好：一种语句级理解的框架 

**Authors**: Xuanqing Liu, Luyang Kong, Wei Niu, Afshin Khashei, Belinda Zeng, Steve Johnson, Jon Jay, Davor Golac, Matt Pope  

**Link**: [PDF](https://arxiv.org/pdf/2503.05620)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in handling complex dialogue tasks without requiring use case-specific fine-tuning. However, analyzing live dialogues in real-time necessitates low-latency processing systems, making it impractical to deploy models with billions of parameters due to latency constraints. As a result, practitioners often prefer smaller models with millions of parameters, trained on high-quality, human-annotated datasets. Yet, curating such datasets is both time-consuming and costly. Consequently, there is a growing need to combine the scalability of LLM-generated labels with the precision of human annotations, enabling fine-tuned smaller models to achieve both higher speed and accuracy comparable to larger models. In this paper, we introduce a simple yet effective framework to address this challenge. Our approach is specifically designed for per-utterance classification problems, which encompass tasks such as intent detection, dialogue state tracking, and more. To mitigate the impact of labeling errors from LLMs -- the primary source of inaccuracies in student models -- we propose a noise-reduced preference learning loss. Experimental results demonstrate that our method significantly improves accuracy across utterance-level dialogue tasks, including sentiment detection (over $2\%$), dialogue act classification (over $1.5\%$), etc. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在无需特定用例微调的情况下展示了处理复杂对话任务的非凡能力。然而，实时分析对话需要低延迟处理系统，因此由于延迟限制，部署具有 billions 参数的模型变得不切实际。因此，实践者通常更倾向于使用 millions 参数的小型模型，并在高质量的人工标注数据集上进行训练。然而，收集这样的数据集既耗时又昂贵。因此，迫切需要结合LLM生成标签的可扩展性与人工标注的精确性，从而使微调后的较小模型能够同时在速度和准确率上与较大模型匹敌。在本文中，我们介绍了一种简单且有效的框架来应对这一挑战。我们的方法特别适用于每句话分类问题，包括意图检测、对话状态跟踪等任务。为了减轻来自LLM的标注错误对学生模型准确性的影响——主要来源——我们提出了一种噪声减少的偏好学习损失。实验结果表明，我们的方法在包括情感识别（超过2%）、对话行为分类（超过1.5%）等句子级对话任务上的准确性显著提升。 

---
# A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of Large Language Models 

**Title (ZH)**: Sparse 自编码器研究：解读大型语言模型的内部机制 

**Authors**: Dong Shu, Xuansheng Wu, Haiyan Zhao, Daking Rai, Ziyu Yao, Ninghao Liu, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2503.05613)  

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing, yet their internal mechanisms remain largely opaque. Recently, mechanistic interpretability has attracted significant attention from the research community as a means to understand the inner workings of LLMs. Among various mechanistic interpretability approaches, Sparse Autoencoders (SAEs) have emerged as a particularly promising method due to their ability to disentangle the complex, superimposed features within LLMs into more interpretable components. This paper presents a comprehensive examination of SAEs as a promising approach to interpreting and understanding LLMs. We provide a systematic overview of SAE principles, architectures, and applications specifically tailored for LLM analysis, covering theoretical foundations, implementation strategies, and recent developments in sparsity mechanisms. We also explore how SAEs can be leveraged to explain the internal workings of LLMs, steer model behaviors in desired directions, and develop more transparent training methodologies for future models. Despite the challenges that remain around SAE implementation and scaling, they continue to provide valuable tools for understanding the internal mechanisms of large language models. 

**Abstract (ZH)**: 大型语言模型的机制解释：稀疏自编码器在理解大型语言模型中的应用 

---
# AceWGS: An LLM-Aided Framework to Accelerate Catalyst Design for Water-Gas Shift Reactions 

**Title (ZH)**: AceWGS：一种基于LLM的框架，用于加速水煤气转变反应催化剂设计 

**Authors**: Joyjit Chattoraj, Brahim Hamadicharef, Teo Shi Chang, Yingzhi Zeng, Chee Kok Poh, Luwei Chen, Teck Leong Tan  

**Link**: [PDF](https://arxiv.org/pdf/2503.05607)  

**Abstract**: While the Water-Gas Shift (WGS) reaction plays a crucial role in hydrogen production for fuel cells, finding suitable catalysts to achieve high yields for low-temperature WGS reactions remains a persistent challenge. Artificial Intelligence (AI) has shown promise in accelerating catalyst design by exploring vast candidate spaces, however, two key gaps limit its effectiveness. First, AI models primarily train on numerical data, which fail to capture essential text-based information, such as catalyst synthesis methods. Second, the cross-disciplinary nature of catalyst design requires seamless collaboration between AI, theory, experiments, and numerical simulations, often leading to communication barriers. To address these gaps, we present AceWGS, a Large Language Models (LLMs)-aided framework to streamline WGS catalyst design. AceWGS interacts with researchers through natural language, answering queries based on four features: (i) answering general queries, (ii) extracting information about the database comprising WGS-related journal articles, (iii) comprehending the context described in these articles, and (iv) identifying catalyst candidates using our proposed AI inverse model. We presented a practical case study demonstrating how AceWGS can accelerate the catalyst design process. AceWGS, built with open-source tools, offers an adjustable framework that researchers can readily adapt for a range of AI-accelerated catalyst design applications, supporting seamless integration across cross-disciplinary studies. 

**Abstract (ZH)**: AceWGS：一种大型语言模型辅助的水煤气变换催化剂设计框架 

---
# CACTUS: An Open Dataset and Framework for Automated Cardiac Assessment and Classification of Ultrasound Images Using Deep Transfer Learning 

**Title (ZH)**: CACTUS：一种基于深度传输学习的心超图像自动心脏评估与分类的开放数据集及框架 

**Authors**: Hanae Elmekki, Ahmed Alagha, Hani Sami, Amanda Spilkin, Antonela Mariel Zanuttini, Ehsan Zakeri, Jamal Bentahar, Lyes Kadem, Wen-Fang Xie, Philippe Pibarot, Rabeb Mizouni, Hadi Otrok, Shakti Singh, Azzam Mourad  

**Link**: [PDF](https://arxiv.org/pdf/2503.05604)  

**Abstract**: Cardiac ultrasound (US) scanning is a commonly used techniques in cardiology to diagnose the health of the heart and its proper functioning. Therefore, it is necessary to consider ways to automate these tasks and assist medical professionals in classifying and assessing cardiac US images. Machine learning (ML) techniques are regarded as a prominent solution due to their success in numerous applications aimed at enhancing the medical field, including addressing the shortage of echography technicians. However, the limited availability of medical data presents a significant barrier to applying ML in cardiology, particularly regarding US images of the heart. This paper addresses this challenge by introducing the first open graded dataset for Cardiac Assessment and ClassificaTion of UltraSound (CACTUS), which is available online. This dataset contains images obtained from scanning a CAE Blue Phantom and representing various heart views and different quality levels, exceeding the conventional cardiac views typically found in the literature. Additionally, the paper introduces a Deep Learning (DL) framework consisting of two main components. The first component classifies cardiac US images based on the heart view using a Convolutional Neural Network (CNN). The second component uses Transfer Learning (TL) to fine-tune the knowledge from the first component and create a model for grading and assessing cardiac images. The framework demonstrates high performance in both classification and grading, achieving up to 99.43% accuracy and as low as 0.3067 error, respectively. To showcase its robustness, the framework is further fine-tuned using new images representing additional cardiac views and compared to several other state-of-the-art architectures. The framework's outcomes and performance in handling real-time scans were also assessed using a questionnaire answered by cardiac experts. 

**Abstract (ZH)**: 心脏超声（US）扫描是心脏病学中常用的技术，用于诊断心脏健康及其正常功能。因此，考虑自动化这些任务并协助医学专业人员分类和评估心脏US图像的方法是必要的。机器学习（ML）技术因其在增强医疗领域中的广泛应用而被视为一种突出的解决方案，包括解决超声技术人员短缺问题。然而，医疗数据的有限可用性是将ML应用于心脏病学，特别是心脏US图像的一个重要障碍。本文通过引入第一个开放分级数据集Cardiac Assessment and ClassificaTion of UltraSound (CACTUS)，解决了这一挑战，该数据集已在互联网上发布。此数据集包含从CAE Blue Phantom扫描获取的图像，并代表了各种心脏视图和不同的质量水平，超过文献中常规的心脏视图。此外，本文还介绍了一个深度学习（DL）框架，该框架由两个主要组件组成。第一个组件使用卷积神经网络（CNN）根据心脏视图对心脏US图像进行分类。第二个组件利用迁移学习（TL）进一步微调第一个组件的知识，以创建用于评分和评估心脏图像的模型。该框架在分类和评分方面均表现出高性能，分类准确率达到99.43%，评分误差低至0.3067。为展示其鲁棒性，该框架进一步利用额外心脏视图的图像进行微调，并与几种其他最先进的架构进行了比较。该框架在处理实时扫描方面的结果和性能也得到了心脏专家问卷调查的评估。 

---
# Quantifying the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data 

**Title (ZH)**: 量化检索增强语言模型在地面数据中虚假特征面前的 robustness 

**Authors**: Shiping Yang, Jie Wu, Wenbiao Ding, Ning Wu, Shining Liang, Ming Gong, Hengyuan Zhang, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05587)  

**Abstract**: Robustness has become a critical attribute for the deployment of RAG systems in real-world applications. Existing research focuses on robustness to explicit noise (e.g., document semantics) but overlooks spurious features (a.k.a. implicit noise). While previous works have explored spurious features in LLMs, they are limited to specific features (e.g., formats) and narrow scenarios (e.g., ICL). In this work, we statistically confirm the presence of spurious features in the RAG paradigm, a robustness problem caused by the sensitivity of LLMs to semantic-agnostic features. Moreover, we provide a comprehensive taxonomy of spurious features and empirically quantify their impact through controlled experiments. Further analysis reveals that not all spurious features are harmful and they can even be beneficial sometimes. Extensive evaluation results across multiple LLMs suggest that spurious features are a widespread and challenging problem in the field of RAG. The code and dataset will be released to facilitate future research. We release all codes and data at: $\\\href{this https URL}{this https URL}$. 

**Abstract (ZH)**: 鲁棒性已成为在实际应用中部署RAG系统的关键属性。现有研究主要关注显式噪声（例如，文档语义）的鲁棒性，但忽视了隐性特征（即隐式噪声）。虽然之前的工作已经在LLMs中探索了隐性特征，但这些工作局限于特定的特征（例如，格式）和狭窄的场景（例如，ICL）。在本文中，我们通过统计方法证实了在RAG范式中存在隐性特征，这是一种由LLMs对语义无关特征的敏感性引起的鲁棒性问题。此外，我们提供了隐性特征的全面分类，并通过受控实验实证衡量其影响。进一步的分析表明，并非所有隐性特征都是有害的，有时它们甚至可能是有益的。在多个LLMs上的广泛评估结果表明，隐性特征是RAG领域中普遍存在且具有挑战性的问题。我们将发布代码和数据以促进未来的研究：$\\\href{this https URL}{this https URL}$。 

---
# InDRiVE: Intrinsic Disagreement based Reinforcement for Vehicle Exploration through Curiosity Driven Generalized World Model 

**Title (ZH)**: InDRiVE: 内在分歧基于强化学习的车辆好奇心驱动 generalize世界模型探索 

**Authors**: Feeza Khan Khanzada, Jaerock Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2503.05573)  

**Abstract**: Model-based Reinforcement Learning (MBRL) has emerged as a promising paradigm for autonomous driving, where data efficiency and robustness are critical. Yet, existing solutions often rely on carefully crafted, task specific extrinsic rewards, limiting generalization to new tasks or environments. In this paper, we propose InDRiVE (Intrinsic Disagreement based Reinforcement for Vehicle Exploration), a method that leverages purely intrinsic, disagreement based rewards within a Dreamer based MBRL framework. By training an ensemble of world models, the agent actively explores high uncertainty regions of environments without any task specific feedback. This approach yields a task agnostic latent representation, allowing for rapid zero shot or few shot fine tuning on downstream driving tasks such as lane following and collision avoidance. Experimental results in both seen and unseen environments demonstrate that InDRiVE achieves higher success rates and fewer infractions compared to DreamerV2 and DreamerV3 baselines despite using significantly fewer training steps. Our findings highlight the effectiveness of purely intrinsic exploration for learning robust vehicle control behaviors, paving the way for more scalable and adaptable autonomous driving systems. 

**Abstract (ZH)**: 基于模型的强化学习（MBRL）在自主驾驶中 emerged as a promising paradigm where 数据效率和鲁棒性至关重要。然而，现有解决方案往往依赖于精心设计的任务特定外在奖励，限制了其对新任务或新环境的泛化能力。在本文中，我们提出了一种名为 InDRiVE（基于内在分歧的车辆探索强化学习）的方法，该方法在基于 Dreamer 的 MBRL 框架中利用纯粹内在的分歧基奖励。通过训练一组世界模型，代理能够在没有任何任务特定反馈的情况下主动探索环境中的高不确定性区域。此方法生成了任务无关的潜在表示，允许多任务的下游驾驶任务（如车道跟随和碰撞避免）的快速零样本或少样本微调。在已见和未见环境的实验结果表明，尽管使用了显著更少的训练步骤，InDRiVE 在成功率和违规次数方面优于 DreamerV2 和 DreamerV3 基线。我们的研究结果突显了纯粹内在探索在学习鲁棒车辆控制行为方面的有效性，为更可扩展和适应性强的自主驾驶系统铺平了道路。 

---
# Compliance of AI Systems 

**Title (ZH)**: AI系统的合规性 

**Authors**: Julius Schöning, Niklas Kruse  

**Link**: [PDF](https://arxiv.org/pdf/2503.05571)  

**Abstract**: The increasing integration of artificial intelligence (AI) systems in various fields requires solid concepts to ensure compliance with upcoming legislation. This paper systematically examines the compliance of AI systems with relevant legislation, focusing on the EU's AI Act and the compliance of data sets. The analysis highlighted many challenges associated with edge devices, which are increasingly being used to deploy AI applications closer and closer to the data sources. Such devices often face unique issues due to their decentralized nature and limited computing resources for implementing sophisticated compliance mechanisms. By analyzing AI implementations, the paper identifies challenges and proposes the first best practices for legal compliance when developing, deploying, and running AI. The importance of data set compliance is highlighted as a cornerstone for ensuring the trustworthiness, transparency, and explainability of AI systems, which must be aligned with ethical standards set forth in regulatory frameworks such as the AI Act. The insights gained should contribute to the ongoing discourse on the responsible development and deployment of embedded AI systems. 

**Abstract (ZH)**: 人工智能系统在各领域的不断增加集成要求具备坚实的概念以确保符合即将出台的立法规定。本文系统性地研究了人工智能系统与相关立法的合规性，重点考察了欧盟AI法案以及数据集的合规性。分析指出，边缘设备越来越多地被用于更接近数据来源部署人工智能应用，这些设备因其去中心化特性以及有限的计算资源来实施复杂合规机制而面临独特的问题。通过分析人工智能实现方式，本文识别出挑战并提出了在开发、部署和运行人工智能时的首批最佳实践方法。数据集合规的重要性被突出强调，作为确保人工智能系统可信、透明和可解释性的基石，必须与伦理标准保持一致，这些标准由如AI法案等监管框架设定。获得的见解应有助于人工智能嵌入系统的负责任开发和部署的持续讨论。 

---
# Impoola: The Power of Average Pooling for Image-Based Deep Reinforcement Learning 

**Title (ZH)**: Impoola: 平均池化在基于图像的深度强化学习中的作用 

**Authors**: Raphael Trumpp, Ansgar Schäfftlein, Mirco Theile, Marco Caccamo  

**Link**: [PDF](https://arxiv.org/pdf/2503.05546)  

**Abstract**: As image-based deep reinforcement learning tackles more challenging tasks, increasing model size has become an important factor in improving performance. Recent studies achieved this by focusing on the parameter efficiency of scaled networks, typically using Impala-CNN, a 15-layer ResNet-inspired network, as the image encoder. However, while Impala-CNN evidently outperforms older CNN architectures, potential advancements in network design for deep reinforcement learning-specific image encoders remain largely unexplored. We find that replacing the flattening of output feature maps in Impala-CNN with global average pooling leads to a notable performance improvement. This approach outperforms larger and more complex models in the Procgen Benchmark, particularly in terms of generalization. We call our proposed encoder model Impoola-CNN. A decrease in the network's translation sensitivity may be central to this improvement, as we observe the most significant gains in games without agent-centered observations. Our results demonstrate that network scaling is not just about increasing model size - efficient network design is also an essential factor. 

**Abstract (ZH)**: 基于图像的深度强化学习随着处理任务的挑战性增加，模型规模的增大已成为提升性能的重要因素。近期研究通过关注缩放网络的参数效率，通常使用Impala-CNN（一个借鉴ResNet设计的15层网络）作为图像编码器来实现这一点。然而，尽管Impala-CNN显然优于较早的CNN架构，针对深度强化学习的特定图像编码器的网络设计潜在改进仍 largely unexplored。我们发现，将Impala-CNN中的输出特征图展平操作替换为全局平均池化可以显著提高性能。这种方法在Procgen基准测试中优于更大、更复杂的模型，尤其是在泛化能力方面。我们将我们提出的设计命名为Impoola-CNN。网络的平移敏感性降低可能是这一改进的关键，因为我们观察到在没有代理中心观测的游戏场景中获得了最大的收益。我们的研究结果表明，网络规模的增加不仅仅是增加模型大小的问题，高效网络设计也是关键因素。 

---
# Post-Hoc Concept Disentanglement: From Correlated to Isolated Concept Representations 

**Title (ZH)**: 事后概念去纠缠：从相关到独立概念表示 

**Authors**: Eren Erogullari, Sebastian Lapuschkin, Wojciech Samek, Frederik Pahde  

**Link**: [PDF](https://arxiv.org/pdf/2503.05522)  

**Abstract**: Concept Activation Vectors (CAVs) are widely used to model human-understandable concepts as directions within the latent space of neural networks. They are trained by identifying directions from the activations of concept samples to those of non-concept samples. However, this method often produces similar, non-orthogonal directions for correlated concepts, such as "beard" and "necktie" within the CelebA dataset, which frequently co-occur in images of men. This entanglement complicates the interpretation of concepts in isolation and can lead to undesired effects in CAV applications, such as activation steering. To address this issue, we introduce a post-hoc concept disentanglement method that employs a non-orthogonality loss, facilitating the identification of orthogonal concept directions while preserving directional correctness. We evaluate our approach with real-world and controlled correlated concepts in CelebA and a synthetic FunnyBirds dataset with VGG16 and ResNet18 architectures. We further demonstrate the superiority of orthogonalized concept representations in activation steering tasks, allowing (1) the insertion of isolated concepts into input images through generative models and (2) the removal of concepts for effective shortcut suppression with reduced impact on correlated concepts in comparison to baseline CAVs. 

**Abstract (ZH)**: Concept Activation Vectors (CAVs) 在潜空间中的可解释概念 Modeling through Post-hoc Concept Disentanglement by Non-Orthogonality Loss 

---
# Cognitive Bias Detection Using Advanced Prompt Engineering 

**Title (ZH)**: 高级提示工程在认知偏差检测中的应用 

**Authors**: Frederic Lemieux, Aisha Behr, Clara Kellermann-Bryant, Zaki Mohammed  

**Link**: [PDF](https://arxiv.org/pdf/2503.05516)  

**Abstract**: Cognitive biases, systematic deviations from rationality in judgment, pose significant challenges in generating objective content. This paper introduces a novel approach for real-time cognitive bias detection in user-generated text using large language models (LLMs) and advanced prompt engineering techniques. The proposed system analyzes textual data to identify common cognitive biases such as confirmation bias, circular reasoning, and hidden assumption. By designing tailored prompts, the system effectively leverages LLMs' capabilities to both recognize and mitigate these biases, improving the quality of human-generated content (e.g., news, media, reports). Experimental results demonstrate the high accuracy of our approach in identifying cognitive biases, offering a valuable tool for enhancing content objectivity and reducing the risks of biased decision-making. 

**Abstract (ZH)**: 基于大型语言模型和高级提示工程的实时用户生成文本认知偏差检测方法 

---
# Noise-Robust Radio Frequency Fingerprint Identification Using Denoise Diffusion Model 

**Title (ZH)**: 噪声鲁棒的射频指纹识别方法：基于降噪扩散模型 

**Authors**: Guolin Yin, Junqing Zhang, Yuan Ding, Simon Cotton  

**Link**: [PDF](https://arxiv.org/pdf/2503.05514)  

**Abstract**: Securing Internet of Things (IoT) devices presents increasing challenges due to their limited computational and energy resources. Radio Frequency Fingerprint Identification (RFFI) emerges as a promising authentication technique to identify wireless devices through hardware impairments. RFFI performance under low signal-to-noise ratio (SNR) scenarios is significantly degraded because the minute hardware features can be easily swamped in noise. In this paper, we leveraged the diffusion model to effectively restore the RFF under low SNR scenarios. Specifically, we trained a powerful noise predictor and tailored a noise removal algorithm to effectively reduce the noise level in the received signal and restore the device fingerprints. We used Wi-Fi as a case study and created a testbed involving 6 commercial off-the-shelf Wi-Fi dongles and a USRP N210 software-defined radio (SDR) platform. We conducted experimental evaluations on various SNR scenarios. The experimental results show that the proposed algorithm can improve the classification accuracy by up to 34.9%. 

**Abstract (ZH)**: 物联网设备的安全性由于其有限的计算能力和能源资源而面临不断增加的挑战。射频指纹识别（RFFI）作为一种通过硬件缺陷识别无线设备的有希望的认证技术逐渐浮现。在低信噪比（SNR）场景下，RFFI的性能显著下降，因为细微的硬件特征容易被噪声淹没。在本文中，我们利用扩散模型有效地恢复了低SNR场景下的RFF。具体而言，我们训练了一个强大的噪声预测器，并设计了一种噪声去除算法，以有效降低接收到的信号中的噪声水平并恢复设备指纹。我们将Wi-Fi作为案例研究，并构建了一个包含6个商用Wi-Fi dongles和一个USRP N210软件定义无线电（SDR）平台的测试床。我们在不同的SNR场景下进行了实验评估。实验结果表明，所提出的算法可以将分类精度提高多达34.9%。 

---
# Grammar-Based Code Representation: Is It a Worthy Pursuit for LLMs? 

**Title (ZH)**: 基于语法的代码表示：LLM值得追求的目标吗？ 

**Authors**: Qingyuan Liang, Zhao Zhang, Zeyu Sun, Zheng Lin, Qi Luo, Yueyi Xiao, Yizhou Chen, Yuqun Zhang, Haotian Zhang, Lu Zhang, Bin Chen, Yingfei Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.05507)  

**Abstract**: Grammar serves as a cornerstone in programming languages and software engineering, providing frameworks to define the syntactic space and program structure. Existing research demonstrates the effectiveness of grammar-based code representations in small-scale models, showing their ability to reduce syntax errors and enhance performance. However, as language models scale to the billion level or beyond, syntax-level errors become rare, making it unclear whether grammar information still provides performance benefits. To explore this, we develop a series of billion-scale GrammarCoder models, incorporating grammar rules in the code generation process. Experiments on HumanEval (+) and MBPP (+) demonstrate a notable improvement in code generation accuracy. Further analysis shows that grammar-based representations enhance LLMs' ability to discern subtle code differences, reducing semantic errors caused by minor variations. These findings suggest that grammar-based code representations remain valuable even in billion-scale models, not only by maintaining syntax correctness but also by improving semantic differentiation. 

**Abstract (ZH)**: 语法在编程语言和软件工程中作为 foundation，提供定义语法空间和程序结构的框架。现有研究显示，基于语法的代码表示在小型模型中具有有效性，能够减少语法错误并提升性能。然而，当语言模型扩展到亿级或更大规模时，语法级别错误变得罕见，这使得不清楚语法信息是否仍能提供性能上的优势。为了探索这一点，我们开发了一系列亿级规模的GrammarCoder模型，在代码生成过程中融入了语法规则。在HumanEval (+) 和 MBPP (+) 上的实验显示代码生成精度有显著提升。进一步的分析表明，基于语法的表示增强了大型语言模型区分细微代码差异的能力，减少了由细微差异引起的语义错误。这些发现表明，即使在亿级模型中，基于语法的代码表示仍然具有价值，不仅通过保持语法的正确性，还通过提高语义区分能力来提升性能。 

---
# EuroBERT: Scaling Multilingual Encoders for European Languages 

**Title (ZH)**: 欧罗巴语BERT：扩展多语言编码器以应用到欧洲语言 

**Authors**: Nicolas Boizard, Hippolyte Gisserot-Boukhlef, Duarte M. Alves, André Martins, Ayoub Hammal, Caio Corro, Céline Hudelot, Emmanuel Malherbe, Etienne Malaboeuf, Fanny Jourdan, Gabriel Hautreux, João Alves, Kevin El-Haddad, Manuel Faysse, Maxime Peyrard, Nuno M. Guerreiro, Patrick Fernandes, Ricardo Rei, Pierre Colombo  

**Link**: [PDF](https://arxiv.org/pdf/2503.05500)  

**Abstract**: General-purpose multilingual vector representations, used in retrieval, regression and classification, are traditionally obtained from bidirectional encoder models. Despite their wide applicability, encoders have been recently overshadowed by advances in generative decoder-only models. However, many innovations driving this progress are not inherently tied to decoders. In this paper, we revisit the development of multilingual encoders through the lens of these advances, and introduce EuroBERT, a family of multilingual encoders covering European and widely spoken global languages. Our models outperform existing alternatives across a diverse range of tasks, spanning multilingual capabilities, mathematics, and coding, and natively supporting sequences of up to 8,192 tokens. We also examine the design decisions behind EuroBERT, offering insights into our dataset composition and training pipeline. We publicly release the EuroBERT models, including intermediate training checkpoints, together with our training framework. 

**Abstract (ZH)**: 通用多语言向量表示在检索、回归和分类中广泛应用，传统上是从双向编码器模型中获取的。尽管它们具有广泛的应用前景，编码器最近已被生成型解码器模型的进步所超越。然而，驱动这一进展的许多创新并非固有地与解码器相关。在本文中，我们通过这些进步的视角回顾多语言编码器的发展，并介绍了EuroBERT这一涵盖欧洲及广泛使用的全球语言的多语言编码器家族。我们的模型在一系列任务中表现出色，涵盖多语言能力、数学和编码领域，并原生支持最大8,192个令牌的序列。我们还分析了EuroBERT的设计决策，提供了关于数据集组成和训练管道的见解。我们公开发布了EuroBERT模型及其中间训练检查点，以及我们的训练框架。 

---
# FastMap: Fast Queries Initialization Based Vectorized HD Map Reconstruction Framework 

**Title (ZH)**: FastMap: 基于向量化快速建图和查询初始化框架 

**Authors**: Haotian Hu, Jingwei Xu, Fanyi Wang, Toyota Li, Yaonong Wang, Laifeng Hu, Zhiwang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05492)  

**Abstract**: Reconstruction of high-definition maps is a crucial task in perceiving the autonomous driving environment, as its accuracy directly impacts the reliability of prediction and planning capabilities in downstream modules. Current vectorized map reconstruction methods based on the DETR framework encounter limitations due to the redundancy in the decoder structure, necessitating the stacking of six decoder layers to maintain performance, which significantly hampers computational efficiency. To tackle this issue, we introduce FastMap, an innovative framework designed to reduce decoder redundancy in existing approaches. FastMap optimizes the decoder architecture by employing a single-layer, two-stage transformer that achieves multilevel representation capabilities. Our framework eliminates the conventional practice of randomly initializing queries and instead incorporates a heatmap-guided query generation module during the decoding phase, which effectively maps image features into structured query vectors using learnable positional encoding. Additionally, we propose a geometry-constrained point-to-line loss mechanism for FastMap, which adeptly addresses the challenge of distinguishing highly homogeneous features that often arise in traditional point-to-point loss computations. Extensive experiments demonstrate that FastMap achieves state-of-the-art performance in both nuScenes and Argoverse2 datasets, with its decoder operating 3.2 faster than the baseline. Code and more demos are available at this https URL. 

**Abstract (ZH)**: 高-definition地图的重建是自主驾驶环境中环境感知的关键任务，其准确性直接影响下游模块预测和规划能力的可靠性。基于DETR框架的向量地图重建方法因解码器结构的冗余限制，需要堆叠六层解码层以维持性能，这显著降低了计算效率。为此，我们提出FastMap，一种旨在减少现有方法中解码器冗余的创新框架。FastMap通过采用单层两阶段变换器优化解码器架构，实现多层次表示能力。该框架摒弃了随机初始化查询的传统做法，在解码阶段引入了由热图引导的查询生成模块，利用可学习的位置编码将图像特征有效地映射到结构化的查询向量中。此外，我们还提出了适用于FastMap的几何约束点到线损失机制，有效地解决了传统点到点损失计算中经常出现的高同质特征区分难题。广泛实验表明，FastMap在nuScenes和Argoverse2数据集中均达到最新技术水平，其解码器比基线快3.2倍。代码和更多演示可在以下链接获得。 

---
# Personalized Federated Learning via Learning Dynamic Graphs 

**Title (ZH)**: 基于学习动态图的个性化 federated 学习 

**Authors**: Ziran Zhou, Guanyu Gao, Xiaohu Wu, Yan Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2503.05474)  

**Abstract**: Personalized Federated Learning (PFL) aims to train a personalized model for each client that is tailored to its local data distribution, learning fails to perform well on individual clients due to variations in their local data distributions. Most existing PFL methods focus on personalizing the aggregated global model for each client, neglecting the fundamental aspect of federated learning: the regulation of how client models are aggregated. Additionally, almost all of them overlook the graph structure formed by clients in federated learning. In this paper, we propose a novel method, Personalized Federated Learning with Graph Attention Network (pFedGAT), which captures the latent graph structure between clients and dynamically determines the importance of other clients for each client, enabling fine-grained control over the aggregation process. We evaluate pFedGAT across multiple data distribution scenarios, comparing it with twelve state of the art methods on three datasets: Fashion MNIST, CIFAR-10, and CIFAR-100, and find that it consistently performs well. 

**Abstract (ZH)**: 基于图注意力网络的个性化联邦学习（Personalized Federated Learning with Graph Attention Network, pFedGAT） 

---
# The Society of HiveMind: Multi-Agent Optimization of Foundation Model Swarms to Unlock the Potential of Collective Intelligence 

**Title (ZH)**: HiveMind 社区：多 Agent 优化基础模型集群以释放集体智能的潜力 

**Authors**: Noah Mamie, Susie Xi Rao  

**Link**: [PDF](https://arxiv.org/pdf/2503.05473)  

**Abstract**: Multi-agent systems address issues of accessibility and scalability of artificial intelligence (AI) foundation models, which are often represented by large language models. We develop a framework - the "Society of HiveMind" (SOHM) - that orchestrates the interaction between multiple AI foundation models, imitating the observed behavior of animal swarms in nature by following modern evolutionary theories. On the one hand, we find that the SOHM provides a negligible benefit on tasks that mainly require real-world knowledge. On the other hand, we remark a significant improvement on tasks that require intensive logical reasoning, indicating that multi-agent systems are capable of increasing the reasoning capabilities of the collective compared to the individual agents. Our findings demonstrate the potential of combining a multitude of diverse AI foundation models to form an artificial swarm intelligence capable of self-improvement through interactions with a given environment. 

**Abstract (ZH)**: 多智能体系统解决了人工智能基础模型在可访问性和可扩展性方面的限制，这些基础模型常常由大型语言模型表示。我们开发了一个框架——“蜂群社会”（SOHM）——它通过遵循现代进化理论来协调多个AI基础模型之间的交互，模仿自然界中动物集群的观察行为。一方面，我们发现SOHM在主要依赖于现实世界知识的任务上提供了微乎其微的优势。另一方面，在需要密集逻辑推理的任务上，我们注意到了显著的改进，表明多智能体系统能够提高集体的推理能力，超过个体代理的能力。我们的研究结果证明，通过与给定环境的交互，结合多种多样的人工智能基础模型，有可能形成一种能够自我改善的仿生群智能系统。 

---
# Controllable Complementarity: Subjective Preferences in Human-AI Collaboration 

**Title (ZH)**: 可控互补性：人类与AI协作中的主观偏好 

**Authors**: Chase McDonald, Cleotilde Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2503.05455)  

**Abstract**: Research on human-AI collaboration often prioritizes objective performance. However, understanding human subjective preferences is essential to improving human-AI complementarity and human experiences. We investigate human preferences for controllability in a shared workspace task with AI partners using Behavior Shaping (BS), a reinforcement learning algorithm that allows humans explicit control over AI behavior.
In one experiment, we validate the robustness of BS in producing effective AI policies relative to self-play policies, when controls are hidden. In another experiment, we enable human control, showing that participants perceive AI partners as more effective and enjoyable when they can directly dictate AI behavior. Our findings highlight the need to design AI that prioritizes both task performance and subjective human preferences. By aligning AI behavior with human preferences, we demonstrate how human-AI complementarity can extend beyond objective outcomes to include subjective preferences. 

**Abstract (ZH)**: 关于人类与人工智能协作的研究往往侧重于客观性能。然而，理解人类的主观偏好对于提高人类与人工智能的互补性和人类体验至关重要。我们通过一种允许人类对人工智能行为进行显式控制的行为塑造（BS）强化学习算法，研究了人类在与人工智能伙伴共同工作空间任务中对可控性的偏好。在一项实验中，我们验证了在隐藏控制的情况下，BS相比自我博弈政策的健壮性。在另一项实验中，我们赋予了人类控制权，表明参与者在能够直接指导人工智能行为时，认为人工智能伙伴更有效且更令人愉悦。我们的研究结果强调了设计兼顾任务性能和主观人类偏好的人工智能的必要性。通过使人工智能行为与人类偏好相一致，我们展示了人类与人工智能的互补性可以超出客观结果，还包括主观偏好。 

---
# Soft Policy Optimization: Online Off-Policy RL for Sequence Models 

**Title (ZH)**: 软策略优化：在线离策序贯模型强化学习 

**Authors**: Taco Cohen, David W. Zhang, Kunhao Zheng, Yunhao Tang, Remi Munos, Gabriel Synnaeve  

**Link**: [PDF](https://arxiv.org/pdf/2503.05453)  

**Abstract**: RL-based post-training of language models is almost exclusively done using on-policy methods such as PPO. These methods cannot learn from arbitrary sequences such as those produced earlier in training, in earlier runs, by human experts or other policies, or by decoding and exploration methods. This results in severe sample inefficiency and exploration difficulties, as well as a potential loss of diversity in the policy responses. Moreover, asynchronous PPO implementations require frequent and costly model transfers, and typically use value models which require a large amount of memory. In this paper we introduce Soft Policy Optimization (SPO), a simple, scalable and principled Soft RL method for sequence model policies that can learn from arbitrary online and offline trajectories and does not require a separate value model. In experiments on code contests, we shows that SPO outperforms PPO on pass@10, is significantly faster and more memory efficient, is able to benefit from off-policy data, enjoys improved stability, and learns more diverse (i.e. soft) policies. 

**Abstract (ZH)**: 基于RL的后训练语言模型几乎完全使用了基于策略的方法（如PPO）。这些方法无法从训练早期、先前运行、人类专家或其他策略、或解码和探索方法生成的任意序列中学习，这导致了严重的样本效率低下和探索困难，并可能损失策略响应的多样性。此外，异步PPO实现需要频繁且昂贵的模型转移，并通常使用需要大量内存的价值模型。在本文中，我们引入了Soft Policy Optimization (SPO)，这是一种简单、可扩展且基于原则的软RL方法，用于序列模型策略，可以从任意的在线和离线轨迹中学习，且不需要独立的价值模型。在代码竞赛实验中，我们展示了SPO在pass@10上优于PPO，速度快且内存效率更高，能够利用离策略数据，具有更好的稳定性，并学习到更具多样性的（即软的）政策。 

---
# LLM-based Iterative Approach to Metamodeling in Automotive 

**Title (ZH)**: 基于LLM的迭代元建模方法在汽车领域中的应用 

**Authors**: Nenad Petrovic, Fengjunjie Pan, Vahid Zolfaghari, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2503.05449)  

**Abstract**: In this paper, we introduce an automated approach to domain-specific metamodel construction relying on Large Language Model (LLM). The main focus is adoption in automotive domain. As outcome, a prototype was implemented as web service using Python programming language, while OpenAI's GPT-4o was used as the underlying LLM. Based on the initial experiments, this approach successfully constructs Ecore metamodel based on set of automotive requirements and visualizes it making use of PlantUML notation, so human experts can provide feedback in order to refine the result. Finally, locally deployable solution is also considered, including the limitations and additional steps required. 

**Abstract (ZH)**: 基于大型语言模型的汽车领域特定元模型自动化构建方法 

---
# Linear-MoE: Linear Sequence Modeling Meets Mixture-of-Experts 

**Title (ZH)**: 线性-MoE：线性序列建模与混合专家模型的结合 

**Authors**: Weigao Sun, Disen Lan, Tong Zhu, Xiaoye Qu, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.05447)  

**Abstract**: Linear Sequence Modeling (LSM) like linear attention, state space models and linear RNNs, and Mixture-of-Experts (MoE) have recently emerged as significant architectural improvements. In this paper, we introduce Linear-MoE, a production-level system for modeling and training large-scale models that integrate LSM with MoE. Linear-MoE leverages the advantages of both LSM modules for linear-complexity sequence modeling and MoE layers for sparsely activation, aiming to offer high performance with efficient training. The Linear-MoE system comprises: 1) Modeling subsystem, which provides a unified framework supporting all instances of LSM. and 2) Training subsystem, which facilitates efficient training by incorporating various advanced parallelism technologies, particularly Sequence Parallelism designed for Linear-MoE models. Additionally, we explore hybrid models that combine Linear-MoE layers with standard Transformer-MoE layers with its Sequence Parallelism to further enhance model flexibility and performance. Evaluations on two model series, A0.3B-2B and A1B-7B, demonstrate Linear-MoE achieves efficiency gains while maintaining competitive performance on various benchmarks, showcasing its potential as a next-generation foundational model architecture. Code: this https URL. 

**Abstract (ZH)**: 线性混合专家（Linear-MoE）：结合线性序列模型与混合专家的生产级系统 

---
# An Empirical Study of Conformal Prediction in LLM with ASP Scaffolds for Robust Reasoning 

**Title (ZH)**: 基于ASP支架的LLM中一致预测的实证研究：稳健推理 

**Authors**: Navdeep Kaur, Lachlan McPheat, Alessandra Russo, Anthony G Cohn, Pranava Madhyastha  

**Link**: [PDF](https://arxiv.org/pdf/2503.05439)  

**Abstract**: In this paper, we examine the use of Conformal Language Modelling (CLM) alongside Answer Set Programming (ASP) to enhance the performance of standard open-weight LLMs on complex multi-step reasoning tasks. Using the StepGame dataset, which requires spatial reasoning, we apply CLM to generate sets of ASP programs from an LLM, providing statistical guarantees on the correctness of the outputs. Experimental results show that CLM significantly outperforms baseline models that use standard sampling methods, achieving substantial accuracy improvements across different levels of reasoning complexity. Additionally, the LLM-as-Judge metric enhances CLM's performance, especially in assessing structurally and logically correct ASP outputs. However, calibrating CLM with diverse calibration sets did not improve generalizability for tasks requiring much longer reasoning steps, indicating limitations in handling more complex tasks. 

**Abstract (ZH)**: 本文探究将同构语言模型(CLM)与回答集编程(ASP)结合使用以增强标准开放权重语言模型在复杂多步推理任务中的性能。通过StepGame数据集，该数据集需要空间推理能力，我们将CLM应用于从语言模型生成ASP程序集，并提供输出正确性的统计保证。实验结果表明，CLM显著优于使用标准采样方法的基线模型，在不同的推理复杂度级别上实现了显著的准确率提升。此外，LLM作为裁判的评估指标进一步提升了CLM的性能，特别是在评估结构上和逻辑上正确的ASP输出方面尤为明显。然而，使用多样化的校准集对CLM进行校准并未提高需要更长推理步骤的任务的一般化能力，这表明在处理更复杂任务方面存在局限性。 

---
# Semantic Shift Estimation via Dual-Projection and Classifier Reconstruction for Exemplar-Free Class-Incremental Learning 

**Title (ZH)**: 基于双投影和分类器重构的示例无类增量学习语义转移估计 

**Authors**: Run He, Di Fang, Yicheng Xu, Yawen Cui, Ming Li, Cen Chen, Ziqian Zeng, Huiping Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05423)  

**Abstract**: Exemplar-Free Class-Incremental Learning (EFCIL) aims to sequentially learn from distinct categories without retaining exemplars but easily suffers from catastrophic forgetting of learned knowledge. While existing EFCIL methods leverage knowledge distillation to alleviate forgetting, they still face two critical challenges: semantic shift and decision bias. Specifically, the embeddings of old tasks shift in the embedding space after learning new tasks, and the classifier becomes biased towards new tasks due to training solely with new data, thereby hindering the balance between old and new knowledge. To address these issues, we propose the Dual-Projection Shift Estimation and Classifier Reconstruction (DPCR) approach for EFCIL. DPCR effectively estimates semantic shift through a dual-projection, which combines a learnable transformation with a row-space projection to capture both task-wise and category-wise shifts. Furthermore, to mitigate decision bias, DPCR employs ridge regression to reformulate classifier training as a reconstruction process. This reconstruction exploits previous information encoded in covariance and prototype of each class after calibration with estimated shift, thereby reducing decision bias. Extensive experiments demonstrate that, across various datasets, DPCR effectively balances old and new tasks, outperforming state-of-the-art EFCIL methods. 

**Abstract (ZH)**: Exemplar-Free Class-Incremental Learning without Catastrophic Forgetting via Dual-Projection Shift Estimation and Classifier Reconstruction 

---
# Static Program Analysis Guided LLM Based Unit Test Generation 

**Title (ZH)**: 基于静态程序分析引导的大语言模型单元测试生成 

**Authors**: Sujoy Roychowdhury, Giriprasad Sridhara, A K Raghavan, Joy Bose, Sourav Mazumdar, Hamender Singh, Srinivasan Bajji Sugumaran, Ricardo Britto  

**Link**: [PDF](https://arxiv.org/pdf/2503.05394)  

**Abstract**: We describe a novel approach to automating unit test generation for Java methods using large language models (LLMs). Existing LLM-based approaches rely on sample usage(s) of the method to test (focal method) and/or provide the entire class of the focal method as input prompt and context. The former approach is often not viable due to the lack of sample usages, especially for newly written focal methods. The latter approach does not scale well enough; the bigger the complexity of the focal method and larger associated class, the harder it is to produce adequate test code (due to factors such as exceeding the prompt and context lengths of the underlying LLM). We show that augmenting prompts with \emph{concise} and \emph{precise} context information obtained by program analysis %of the focal method increases the effectiveness of generating unit test code through LLMs. We validate our approach on a large commercial Java project and a popular open-source Java project. 

**Abstract (ZH)**: 我们描述了一种使用大规模语言模型（LLMs）自动生成Java方法单元测试的新方法。现有的基于LLM的方法依赖于方法使用样本（目标方法的示例用法）进行测试，或者提供目标方法的整个类作为输入提示和上下文。前者由于缺乏示例用法，特别是在新编写目标方法的情况下，往往不可行。后者在扩展性方面也存在问题；目标方法及其相关类的复杂度越大，生成足够的测试代码就越困难（由于诸如超出底层LLM提示和上下文长度限制等因素）。我们通过提供由程序分析获得的简洁且精确的上下文信息来增强提示，以提高通过LLM生成单元测试代码的有效性。我们在一个大型商用Java项目和一个流行的开源Java项目上验证了我们的方法。 

---
# Shifting Perspectives: Steering Vector Ensembles for Robust Bias Mitigation in LLMs 

**Title (ZH)**: 转变视角：引导矢量集合在LLMs中实现稳健的偏差减轻 

**Authors**: Zara Siddique, Irtaza Khalid, Liam D. Turner, Luis Espinosa-Anke  

**Link**: [PDF](https://arxiv.org/pdf/2503.05371)  

**Abstract**: We present a novel approach to bias mitigation in large language models (LLMs) by applying steering vectors to modify model activations in forward passes. We employ Bayesian optimization to systematically identify effective contrastive pair datasets across nine bias axes. When optimized on the BBQ dataset, our individually tuned steering vectors achieve average improvements of 12.2%, 4.7%, and 3.2% over the baseline for Mistral, Llama, and Qwen, respectively. Building on these promising results, we introduce Steering Vector Ensembles (SVE), a method that averages multiple individually optimized steering vectors, each targeting a specific bias axis such as age, race, or gender. By leveraging their collective strength, SVE outperforms individual steering vectors in both bias reduction and maintaining model performance. The work presents the first systematic investigation of steering vectors for bias mitigation, and we demonstrate that SVE is a powerful and computationally efficient strategy for reducing bias in LLMs, with broader implications for enhancing AI safety. 

**Abstract (ZH)**: 我们提出了一种通过应用导向矢量来修改大型语言模型（LLMs）前向传递中模型激活的新方法，以减轻偏差。我们使用贝叶斯优化系统地识别出在九个偏差轴上有效的对比 pair 数据集。当在 BBQ 数据集上优化时，我们单独调优的导向矢量分别在 Mistral、Llama 和 Qwen 上实现了基线平均改进 12.2%、4.7% 和 3.2%。基于这些有希望的结果，我们引入了导向矢量集成（SVE）方法，该方法平均多个单独优化的导向矢量，每个导向矢量针对特定的偏差轴（如年龄、种族或性别）。通过利用它们的集体力量，SVE 在降低偏差和保持模型性能方面均优于单独的导向矢量。我们的工作首次系统地研究了导向矢量在减轻偏差方面的应用，并展示了 SVE 是一种强大的且计算效率高的策略，用于减轻 LLM 中的偏差，并具有提高 AI 安全性的广泛意义。 

---
# Improving Hate Speech Classification with Cross-Taxonomy Dataset Integration 

**Title (ZH)**: 跨 taxonomy 数据集整合以改进仇恨言论分类 

**Authors**: Jan Fillies, Adrian Paschke  

**Link**: [PDF](https://arxiv.org/pdf/2503.05357)  

**Abstract**: Algorithmic hate speech detection faces significant challenges due to the diverse definitions and datasets used in research and practice. Social media platforms, legal frameworks, and institutions each apply distinct yet overlapping definitions, complicating classification efforts. This study addresses these challenges by demonstrating that existing datasets and taxonomies can be integrated into a unified model, enhancing prediction performance and reducing reliance on multiple specialized classifiers. The work introduces a universal taxonomy and a hate speech classifier capable of detecting a wide range of definitions within a single framework. Our approach is validated by combining two widely used but differently annotated datasets, showing improved classification performance on an independent test set. This work highlights the potential of dataset and taxonomy integration in advancing hate speech detection, increasing efficiency, and ensuring broader applicability across contexts. 

**Abstract (ZH)**: 算法仇恨言论检测面临着由于研究和实践中使用的多样定义和数据集所造成的重大挑战。社会媒体平台、法律框架和机构各自采用不同的但又相互重叠的定义，这使得分类工作变得复杂。本研究通过展示现有数据集和分类法可以整合到一个统一模型中，从而提高预测性能并减少对多种专门分类器的依赖，来应对这些挑战。这项工作提出了一种通用分类法和一个能够在一个框架内检测广泛定义的仇恨言论分类器。我们的方法通过结合两个广泛使用的但标注不同的数据集得以验证，在独立测试集上展示了改进的分类性能。这项工作强调了数据集和分类法整合在推进仇恨言论检测方面的潜力，提高了效率，并确保了更广泛的适用性。 

---
# On the Logical Content of Logic Programs 

**Title (ZH)**: 逻辑程序中的逻辑内容 

**Authors**: Alexader V. Gheorghiu  

**Link**: [PDF](https://arxiv.org/pdf/2503.05355)  

**Abstract**: Logic programming (LP) is typically understood through operational semantics (e.g., SLD-resolution) or model-theoretic interpretations (e.g., the least Herbrand model). This paper introduces a novel perspective on LP by defining a ``support'' relation that explicates what a program ``knows''. This interpretation is shown to express classical and intuitionistic logic, as well as an intermediate logic, depending on certain choices regarding LP and the meanings of disjunction and negation. These results are formalized using the idea of base-extension semantics within proof-theoretic semantics. Our approach offers new insights into the logical foundations of LP and has potential applications in knowledge representation, automated reasoning, and formal verification. 

**Abstract (ZH)**: 逻辑编程的一种新型视角：通过“支持”关系阐明程序的“知识” 

---
# Spatial Distillation based Distribution Alignment (SDDA) for Cross-Headset EEG Classification 

**Title (ZH)**: 基于空间蒸馏的分布对齐（SDDA）方法在跨头戴设备EEG分类中的应用 

**Authors**: Dingkun Liu, Siyang Li, Ziwei Wang, Wei Li, Dongrui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.05349)  

**Abstract**: A non-invasive brain-computer interface (BCI) enables direct interaction between the user and external devices, typically via electroencephalogram (EEG) signals. However, decoding EEG signals across different headsets remains a significant challenge due to differences in the number and locations of the electrodes. To address this challenge, we propose a spatial distillation based distribution alignment (SDDA) approach for heterogeneous cross-headset transfer in non-invasive BCIs. SDDA uses first spatial distillation to make use of the full set of electrodes, and then input/feature/output space distribution alignments to cope with the significant differences between the source and target domains. To our knowledge, this is the first work to use knowledge distillation in cross-headset transfers. Extensive experiments on six EEG datasets from two BCI paradigms demonstrated that SDDA achieved superior performance in both offline unsupervised domain adaptation and online supervised domain adaptation scenarios, consistently outperforming 10 classical and state-of-the-art transfer learning algorithms. 

**Abstract (ZH)**: 一种基于空间蒸馏的分布对齐方法（SDDA）在跨头盔非侵入式脑机接口中的异质域迁移 

---
# AutoIOT: LLM-Driven Automated Natural Language Programming for AIoT Applications 

**Title (ZH)**: AutoIOT：由大规模语言模型驱动的自动化自然语言编程 for AIoT 应用 

**Authors**: Leming Shen, Qiang Yang, Yuanqing Zheng, Mo Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.05346)  

**Abstract**: The advent of Large Language Models (LLMs) has profoundly transformed our lives, revolutionizing interactions with AI and lowering the barrier to AI usage. While LLMs are primarily designed for natural language interaction, the extensive embedded knowledge empowers them to comprehend digital sensor data. This capability enables LLMs to engage with the physical world through IoT sensors and actuators, performing a myriad of AIoT tasks. Consequently, this evolution triggers a paradigm shift in conventional AIoT application development, democratizing its accessibility to all by facilitating the design and development of AIoT applications via natural language. However, some limitations need to be addressed to unlock the full potential of LLMs in AIoT application development. First, existing solutions often require transferring raw sensor data to LLM servers, which raises privacy concerns, incurs high query fees, and is limited by token size. Moreover, the reasoning processes of LLMs are opaque to users, making it difficult to verify the robustness and correctness of inference results. This paper introduces AutoIOT, an LLM-based automated program generator for AIoT applications. AutoIOT enables users to specify their requirements using natural language (input) and automatically synthesizes interpretable programs with documentation (output). AutoIOT automates the iterative optimization to enhance the quality of generated code with minimum user involvement. AutoIOT not only makes the execution of AIoT tasks more explainable but also mitigates privacy concerns and reduces token costs with local execution of synthesized programs. Extensive experiments and user studies demonstrate AutoIOT's remarkable capability in program synthesis for various AIoT tasks. The synthesized programs can match and even outperform some representative baselines. 

**Abstract (ZH)**: 大型语言模型(Large Language Models, LLMs)的出现深刻改变了我们的生活，通过革新人机互动方式并降低人工智能的应用门槛。虽然LLMs主要用于自然语言交互，但其广泛的内置知识使它们能够理解数字传感器数据。这种能力使LLMs能够通过物联网传感器和执行器与物理世界互动，执行多种AIoT任务。因此，这一演变引发了传统AIoT应用开发范式的转变，通过自然语言简化了AIoT应用的设计和开发。然而，仍需解决一些限制，以充分发挥LLMs在AIoT应用开发中的潜力。首先，现有解决方案通常需要将原始传感器数据传输到LLM服务器，这引发了隐私问题，产生了较高的查询费用，并且受到token数量的限制。此外，LLMs的推理过程对用户不透明，难以验证推断结果的鲁棒性和正确性。本文介绍了AutoIOT，一种基于LLM的AIoT应用自动程序生成器。AutoIOT使用户能够使用自然语言（输入）指定其需求，并自动合成可解释的程序并附带文档（输出）。AutoIOT通过最小化用户参与自动迭代优化，提升生成代码的质量。AutoIOT不仅使AIoT任务的执行更具可解释性，还通过局部执行合成程序来缓解隐私问题和降低token成本。广泛的实验和用户研究证明了AutoIOT在各种AIoT任务中的出色生成程序能力。生成的程序能够与甚至超越一些代表性基准。 

---
# Speculative Decoding for Multi-Sample Inference 

**Title (ZH)**: 推测解码for多样本推理 

**Authors**: Yiwei Li, Jiayi Shi, Shaoxiong Feng, Peiwen Yuan, Xinglin Wang, Yueqi Zhang, Ji Zhang, Chuyi Tan, Boyuan Pan, Yao Hu, Kan Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.05330)  

**Abstract**: We propose a novel speculative decoding method tailored for multi-sample reasoning scenarios, such as self-consistency and Best-of-N sampling. Our method exploits the intrinsic consensus of parallel generation paths to synthesize high-quality draft tokens without requiring auxiliary models or external databases. By dynamically analyzing structural patterns across parallel reasoning paths through a probabilistic aggregation mechanism, it identifies consensus token sequences that align with the decoding distribution. Evaluations on mathematical reasoning benchmarks demonstrate a substantial improvement in draft acceptance rates over baselines, while reducing the latency in draft token construction. This work establishes a paradigm shift for efficient multi-sample inference, enabling seamless integration of speculative decoding with sampling-based reasoning techniques. 

**Abstract (ZH)**: 我们提出了一种专门针对多样本推理场景（如自我一致性及Best-of-N采样）的新型推测性解码方法。该方法利用并行生成路径内的固有共识来合成高质量的草稿令牌，无需辅助模型或外部数据库。通过概率聚合机制动态分析并行推理路径中的结构模式，它能够识别与解码分布相一致的共识令牌序列。在数学推理基准测试上的评估表明，与基线方法相比，该方法在草稿接纳率上取得了显著提高，同时降低了草稿令牌构建的延迟。这项工作确立了高效多样本推理的新范式，使得推测性解码与基于采样的推理技术无缝集成。 

---
# Dynamic Knowledge Integration for Evidence-Driven Counter-Argument Generation with Large Language Models 

**Title (ZH)**: 基于大型语言模型的证据驱动反argument生成中的动态知识集成 

**Authors**: Anar Yeginbergen, Maite Oronoz, Rodrigo Agerri  

**Link**: [PDF](https://arxiv.org/pdf/2503.05328)  

**Abstract**: This paper investigates the role of dynamic external knowledge integration in improving counter-argument generation using Large Language Models (LLMs). While LLMs have shown promise in argumentative tasks, their tendency to generate lengthy, potentially unfactual responses highlights the need for more controlled and evidence-based approaches. We introduce a new manually curated dataset of argument and counter-argument pairs specifically designed to balance argumentative complexity with evaluative feasibility. We also propose a new LLM-as-a-Judge evaluation methodology that shows a stronger correlation with human judgments compared to traditional reference-based metrics. Our experimental results demonstrate that integrating dynamic external knowledge from the web significantly improves the quality of generated counter-arguments, particularly in terms of relatedness, persuasiveness, and factuality. The findings suggest that combining LLMs with real-time external knowledge retrieval offers a promising direction for developing more effective and reliable counter-argumentation systems. 

**Abstract (ZH)**: 本论文探究动态外部知识整合在提高大型语言模型（LLMs）反论生成中的作用。虽然LLMs在论辩任务上展现了潜力，但它们生成长篇且可能不实的回应表明需要更加受控和基于证据的方法。我们引入了一个新的手动_curated反论数据集，专门设计以平衡论辩复杂性和评估可行性。我们还提出了一个新的LLM-as-a-Judge评估方法，其与人类判断的相关性比传统引用基标准测量指标更强。实验结果显示，从网络中整合动态外部知识显著提高了生成反论的质量，特别是在相关性、说服力和事实性方面。研究结果表明，将LLMs与实时外部知识检索相结合是开发更有效和可靠反论系统的一个有希望的方向。 

---
# Attenuation artifact detection and severity classification in intracoronary OCT using mixed image representations 

**Title (ZH)**: 基于混合图像表示的冠状动脉OCT中衰减伪影检测及严重程度分类 

**Authors**: Pierandrea Cancian, Simone Saitta, Xiaojin Gu, Rudolf L.M. van Herten, Thijs J. Luttikholt, Jos Thannhauser, Rick H.J.A. Volleberg, Ruben G.A. van der Waerden, Joske L. van der Zande, Clarisa I. Sánchez, Bram van Ginneken, Niels van Royen, Ivana Išgum  

**Link**: [PDF](https://arxiv.org/pdf/2503.05322)  

**Abstract**: In intracoronary optical coherence tomography (OCT), blood residues and gas bubbles cause attenuation artifacts that can obscure critical vessel structures. The presence and severity of these artifacts may warrant re-acquisition, prolonging procedure time and increasing use of contrast agent. Accurate detection of these artifacts can guide targeted re-acquisition, reducing the amount of repeated scans needed to achieve diagnostically viable images. However, the highly heterogeneous appearance of these artifacts poses a challenge for the automated detection of the affected image regions. To enable automatic detection of the attenuation artifacts caused by blood residues and gas bubbles based on their severity, we propose a convolutional neural network that performs classification of the attenuation lines (A-lines) into three classes: no artifact, mild artifact and severe artifact. Our model extracts and merges features from OCT images in both Cartesian and polar coordinates, where each column of the image represents an A-line. Our method detects the presence of attenuation artifacts in OCT frames reaching F-scores of 0.77 and 0.94 for mild and severe artifacts, respectively. The inference time over a full OCT scan is approximately 6 seconds. Our experiments show that analysis of images represented in both Cartesian and polar coordinate systems outperforms the analysis in polar coordinates only, suggesting that these representations contain complementary features. This work lays the foundation for automated artifact assessment and image acquisition guidance in intracoronary OCT imaging. 

**Abstract (ZH)**: 基于衰减程度的血残留和气泡衰减伪影的光学相干断层扫描自动检测 

---
# Disentangling Task Interference within Neurons: Model Merging in Alignment with Neuronal Mechanisms 

**Title (ZH)**: 在神经机制一致性的框架下解纠缠任务干扰：神经元内模型融合 

**Authors**: Zitao Fang, Guodong DU, Shuyang Yu, Yifei Guo, Yiwei Zhang, Jing Li, Ho-Kin Tang, Sim Kuan Goh  

**Link**: [PDF](https://arxiv.org/pdf/2503.05320)  

**Abstract**: Fine-tuning pre-trained models on targeted datasets enhances task-specific performance but often comes at the expense of generalization. Model merging techniques, which integrate multiple fine-tuned models into a single multi-task model through task arithmetic at various levels: model, layer, or parameter, offer a promising solution. However, task interference remains a fundamental challenge, leading to performance degradation and suboptimal merged models. Existing approaches largely overlook the fundamental role of individual neurons and their connectivity, resulting in a lack of interpretability in both the merging process and the merged models. In this work, we present the first study on the impact of neuronal alignment in model merging. We decompose task-specific representations into two complementary neuronal subspaces that regulate neuron sensitivity and input adaptability. Leveraging this decomposition, we introduce NeuroMerging, a novel merging framework developed to mitigate task interference within neuronal subspaces, enabling training-free model fusion across diverse tasks. Through extensive experiments, we demonstrate that NeuroMerging achieves superior performance compared to existing methods on multi-task benchmarks across both vision and natural language domains. Our findings highlight the importance of aligning neuronal mechanisms in model merging, offering new insights into mitigating task interference and improving knowledge fusion. 

**Abstract (ZH)**: 基于神经元对齐的模型合并对多任务学习的影响研究 

---
# Robust Multimodal Learning for Ophthalmic Disease Grading via Disentangled Representation 

**Title (ZH)**: 基于解耦表示的鲁棒多模态学习在眼科疾病分级中的应用 

**Authors**: Xinkun Wang, Yifang Wang, Senwei Liang, Feilong Tang, Chengzhi Liu, Ming Hu, Chao Hu, Junjun He, Zongyuan Ge, Imran Razzak  

**Link**: [PDF](https://arxiv.org/pdf/2503.05319)  

**Abstract**: This paper discusses how ophthalmologists often rely on multimodal data to improve diagnostic accuracy. However, complete multimodal data is rare in real-world applications due to a lack of medical equipment and concerns about data privacy. Traditional deep learning methods typically address these issues by learning representations in latent space. However, the paper highlights two key limitations of these approaches: (i) Task-irrelevant redundant information (e.g., numerous slices) in complex modalities leads to significant redundancy in latent space representations. (ii) Overlapping multimodal representations make it difficult to extract unique features for each modality. To overcome these challenges, the authors propose the Essence-Point and Disentangle Representation Learning (EDRL) strategy, which integrates a self-distillation mechanism into an end-to-end framework to enhance feature selection and disentanglement for more robust multimodal learning. Specifically, the Essence-Point Representation Learning module selects discriminative features that improve disease grading performance. The Disentangled Representation Learning module separates multimodal data into modality-common and modality-unique representations, reducing feature entanglement and enhancing both robustness and interpretability in ophthalmic disease diagnosis. Experiments on multimodal ophthalmology datasets show that the proposed EDRL strategy significantly outperforms current state-of-the-art methods. 

**Abstract (ZH)**: 本文讨论了眼科医生如何依靠多模态数据以提高诊断准确性。然而，在实际应用中，由于缺乏医疗设备和数据隐私方面的担忧，完整的多模态数据很少见。传统的深度学习方法通常通过在潜在空间中学习表示来解决这些问题。然而，本文强调了这些方法的两个关键局限性：（i）在复杂模态中存在与任务无关的冗余信息（如大量切片），导致潜在空间表示中的显著冗余；（ii）重叠的多模态表示使得难以提取每种模态的独特特征。为克服这些挑战，作者提出了一种称为Essence-Point和解耦表示学习（EDRL）的策略，该策略将自我蒸馏机制集成到端到端框架中，以增强特征选择和解耦，从而提高多模态学习的鲁棒性。具体来说，Essence-Point表示学习模块选择能提高疾病分级性能的判别特征。解耦表示学习模块将多模态数据分离为模态通用和模态独特表示，减少特征纠缠并增强眼部疾病诊断的鲁棒性和可解释性。实验表明，所提出的EDRL策略在多模态眼科数据集上的性能显著优于当前最先进的方法。 

---
# Uncertainty-Aware Decoding with Minimum Bayes Risk 

**Title (ZH)**: 最小贝叶斯风险意识的不确定性解码 

**Authors**: Nico Daheim, Clara Meister, Thomas Möllenhoff, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2503.05318)  

**Abstract**: Despite their outstanding performance in the majority of scenarios, contemporary language models still occasionally generate undesirable outputs, for example, hallucinated text. While such behaviors have previously been linked to uncertainty, there is a notable lack of methods that actively consider uncertainty during text generation. In this work, we show how Minimum Bayes Risk (MBR) decoding, which selects model generations according to an expected risk, can be generalized into a principled uncertainty-aware decoding method. In short, we account for model uncertainty during decoding by incorporating a posterior over model parameters into MBR's computation of expected risk. We show that this modified expected risk is useful for both choosing outputs and deciding when to abstain from generation and can provide improvements without incurring overhead. We benchmark different methods for learning posteriors and show that performance improves with prediction diversity. We release our code publicly. 

**Abstract (ZH)**: 尽管当代语言模型在大多数场景下表现出色，但仍 occasionally生成不 desirable 的输出，例如虚构文本。虽然这些行为之前已与不确定性关联起来，但鲜有方法在文本生成过程中积极考虑不确定性。在本文中，我们展示了如何将最小贝叶斯风险（MBR）解码法推广为一个有原则的不确定性意识解码方法，通过将模型参数的后验概率纳入MBR中期望风险的计算，我们在解码过程中考虑了模型的不确定性。我们证明，这种修改后的期望风险对于选择输出和决定何时停止生成都是有用的，且不会增加额外开销。我们对学习后验的方法进行了基准测试，并显示预测多样性能够提升性能。我们公开发布了我们的代码。 

---
# Adversarial Policy Optimization for Offline Preference-based Reinforcement Learning 

**Title (ZH)**: 基于离线偏好强化学习的对抗性策略优化 

**Authors**: Hyungkyu Kang, Min-hwan Oh  

**Link**: [PDF](https://arxiv.org/pdf/2503.05306)  

**Abstract**: In this paper, we study offline preference-based reinforcement learning (PbRL), where learning is based on pre-collected preference feedback over pairs of trajectories. While offline PbRL has demonstrated remarkable empirical success, existing theoretical approaches face challenges in ensuring conservatism under uncertainty, requiring computationally intractable confidence set constructions. We address this limitation by proposing Adversarial Preference-based Policy Optimization (APPO), a computationally efficient algorithm for offline PbRL that guarantees sample complexity bounds without relying on explicit confidence sets. By framing PbRL as a two-player game between a policy and a model, our approach enforces conservatism in a tractable manner. Using standard assumptions on function approximation and bounded trajectory concentrability, we derive a sample complexity bound. To our knowledge, APPO is the first offline PbRL algorithm to offer both statistical efficiency and practical applicability. Experimental results on continuous control tasks demonstrate that APPO effectively learns from complex datasets, showing comparable performance with existing state-of-the-art methods. 

**Abstract (ZH)**: 在本论文中，我们研究了基于离线偏好反馈的强化学习（PbRL），其中学习基于预先收集的轨迹对的偏好反馈。尽管离线PbRL在实验上取得了显著的成功，但现有的理论方法在确保不确定性下的保守性时面临挑战，需要构建计算上不可行的置信集。我们通过提出对抗性基于偏好的策略优化（APPO）算法解决了这一限制，该算法在不依赖显式置信集的情况下提供了样本复杂度界。通过将PbRL建模为策略和模型之间的两人游戏，我们的方法以可计算的方式保证了保守性。在函数逼近和轨迹集中性有标准假设的情况下，我们推导出了样本复杂度界。据我们所知，APPO是第一个同时提供统计效率和实际适用性的离线PbRL算法。实验结果在连续控制任务上表明，APPO能够有效地从复杂的数据集中学到，展示了与现有最先进的方法相当的性能。 

---
# Frequency Autoregressive Image Generation with Continuous Tokens 

**Title (ZH)**: 连续Token驱动的频率自回归图像生成 

**Authors**: Hu Yu, Hao Luo, Hangjie Yuan, Yu Rong, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.05305)  

**Abstract**: Autoregressive (AR) models for image generation typically adopt a two-stage paradigm of vector quantization and raster-scan ``next-token prediction", inspired by its great success in language modeling. However, due to the huge modality gap, image autoregressive models may require a systematic reevaluation from two perspectives: tokenizer format and regression direction. In this paper, we introduce the frequency progressive autoregressive (\textbf{FAR}) paradigm and instantiate FAR with the continuous tokenizer. Specifically, we identify spectral dependency as the desirable regression direction for FAR, wherein higher-frequency components build upon the lower one to progressively construct a complete image. This design seamlessly fits the causality requirement for autoregressive models and preserves the unique spatial locality of image data. Besides, we delve into the integration of FAR and the continuous tokenizer, introducing a series of techniques to address optimization challenges and improve the efficiency of training and inference processes. We demonstrate the efficacy of FAR through comprehensive experiments on the ImageNet dataset and verify its potential on text-to-image generation. 

**Abstract (ZH)**: 基于频率渐进的自回归（FAR）模型 paradigm及其在图像生成中的应用 

---
# Evidential Uncertainty Estimation for Multi-Modal Trajectory Prediction 

**Title (ZH)**: 多模态轨迹预测中的证据不确定性估计 

**Authors**: Sajad Marvi, Christoph Rist, Julian Schmidt, Julian Jordan, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2503.05274)  

**Abstract**: Accurate trajectory prediction is crucial for autonomous driving, yet uncertainty in agent behavior and perception noise makes it inherently challenging. While multi-modal trajectory prediction models generate multiple plausible future paths with associated probabilities, effectively quantifying uncertainty remains an open problem. In this work, we propose a novel multi-modal trajectory prediction approach based on evidential deep learning that estimates both positional and mode probability uncertainty in real time. Our approach leverages a Normal Inverse Gamma distribution for positional uncertainty and a Dirichlet distribution for mode uncertainty. Unlike sampling-based methods, it infers both types of uncertainty in a single forward pass, significantly improving efficiency. Additionally, we experimented with uncertainty-driven importance sampling to improve training efficiency by prioritizing underrepresented high-uncertainty samples over redundant ones. We perform extensive evaluations of our method on the Argoverse 1 and Argoverse 2 datasets, demonstrating that it provides reliable uncertainty estimates while maintaining high trajectory prediction accuracy. 

**Abstract (ZH)**: 基于证据深度学习的实时位置和模式概率不确定性估计的多模态轨迹预测方法 

---
# PhiloBERTA: A Transformer-Based Cross-Lingual Analysis of Greek and Latin Lexicons 

**Title (ZH)**: PhiloBERTA：基于Transformer的希腊语和拉丁语词典跨语言分析 

**Authors**: Rumi A. Allbert, Makai L. Allbert  

**Link**: [PDF](https://arxiv.org/pdf/2503.05265)  

**Abstract**: We present PhiloBERTA, a cross-lingual transformer model that measures semantic relationships between ancient Greek and Latin lexicons. Through analysis of selected term pairs from classical texts, we use contextual embeddings and angular similarity metrics to identify precise semantic alignments. Our results show that etymologically related pairs demonstrate significantly higher similarity scores, particularly for abstract philosophical concepts such as epistēmē (scientia) and dikaiosynē (iustitia). Statistical analysis reveals consistent patterns in these relationships (p = 0.012), with etymologically related pairs showing remarkably stable semantic preservation compared to control pairs. These findings establish a quantitative framework for examining how philosophical concepts moved between Greek and Latin traditions, offering new methods for classical philological research. 

**Abstract (ZH)**: 我们呈现PhiloBERTA，一种跨语言变换模型，用于测量古希腊语和拉丁语词典之间的语义关系。通过分析古典文献中选择的词对，我们使用上下文嵌入和角度相似性度量来识别精确的语义对齐。我们的结果表明，语源学上相关的词对显示出显著更高的相似分数，特别是在epistēmē（scientia）和dikaiosynē（iustitia）这样的抽象哲学概念方面。统计分析揭示了这些关系中一致的模式（p = 0.012），语源学上相关的词对在语义保真度方面显示出异常的稳定性，相比之下，控制词对则不是。这些发现为研究哲学概念如何在古希腊和拉丁传统之间转移建立了一个量化框架，为古典语言学研究提供了新的方法。 

---
# Jailbreaking is (Mostly) Simpler Than You Think 

**Title (ZH)**: 越狱并没有你想象的那么复杂 

**Authors**: Mark Russinovich, Ahmed Salem  

**Link**: [PDF](https://arxiv.org/pdf/2503.05264)  

**Abstract**: We introduce the Context Compliance Attack (CCA), a novel, optimization-free method for bypassing AI safety mechanisms. Unlike current approaches -- which rely on complex prompt engineering and computationally intensive optimization -- CCA exploits a fundamental architectural vulnerability inherent in many deployed AI systems. By subtly manipulating conversation history, CCA convinces the model to comply with a fabricated dialogue context, thereby triggering restricted behavior. Our evaluation across a diverse set of open-source and proprietary models demonstrates that this simple attack can circumvent state-of-the-art safety protocols. We discuss the implications of these findings and propose practical mitigation strategies to fortify AI systems against such elementary yet effective adversarial tactics. 

**Abstract (ZH)**: Context Compliance Attack: A Novel Method for Bypassing AI Safety Mechanisms 

---
# A Map-free Deep Learning-based Framework for Gate-to-Gate Monocular Visual Navigation aboard Miniaturized Aerial Vehicles 

**Title (ZH)**: 无地图深度学习导向的微型飞行器单目视觉导航框架 

**Authors**: Lorenzo Scarciglia, Antonio Paolillo, Daniele Palossi  

**Link**: [PDF](https://arxiv.org/pdf/2503.05251)  

**Abstract**: Palm-sized autonomous nano-drones, i.e., sub-50g in weight, recently entered the drone racing scenario, where they are tasked to avoid obstacles and navigate as fast as possible through gates. However, in contrast with their bigger counterparts, i.e., kg-scale drones, nano-drones expose three orders of magnitude less onboard memory and compute power, demanding more efficient and lightweight vision-based pipelines to win the race. This work presents a map-free vision-based (using only a monocular camera) autonomous nano-drone that combines a real-time deep learning gate detection front-end with a classic yet elegant and effective visual servoing control back-end, only relying on onboard resources. Starting from two state-of-the-art tiny deep learning models, we adapt them for our specific task, and after a mixed simulator-real-world training, we integrate and deploy them aboard our nano-drone. Our best-performing pipeline costs of only 24M multiply-accumulate operations per frame, resulting in a closed-loop control performance of 30 Hz, while achieving a gate detection root mean square error of 1.4 pixels, on our ~20k real-world image dataset. In-field experiments highlight the capability of our nano-drone to successfully navigate through 15 gates in 4 min, never crashing and covering a total travel distance of ~100m, with a peak flight speed of 1.9 m/s. Finally, to stress the generalization capability of our system, we also test it in a never-seen-before environment, where it navigates through gates for more than 4 min. 

**Abstract (ZH)**: 掌サイズ自主纳米无人机：基于单目视觉的无地图自主纳米无人机及其应用 

---
# Robust Conformal Prediction with a Single Binary Certificate 

**Title (ZH)**: 单个二元证书的稳健同credible预测 

**Authors**: Soroush H. Zargarbashi, Aleksandar Bojchevski  

**Link**: [PDF](https://arxiv.org/pdf/2503.05239)  

**Abstract**: Conformal prediction (CP) converts any model's output to prediction sets with a guarantee to cover the true label with (adjustable) high probability. Robust CP extends this guarantee to worst-case (adversarial) inputs. Existing baselines achieve robustness by bounding randomly smoothed conformity scores. In practice, they need expensive Monte-Carlo (MC) sampling (e.g. $\sim10^4$ samples per point) to maintain an acceptable set size. We propose a robust conformal prediction that produces smaller sets even with significantly lower MC samples (e.g. 150 for CIFAR10). Our approach binarizes samples with an adjustable (or automatically adjusted) threshold selected to preserve the coverage guarantee. Remarkably, we prove that robustness can be achieved by computing only one binary certificate, unlike previous methods that certify each calibration (or test) point. Thus, our method is faster and returns smaller robust sets. We also eliminate a previous limitation that requires a bounded score function. 

**Abstract (ZH)**: 符合学术规范的标题翻译：

自适应预测 (CP) 将任何模型的输出转换为具有可调高概率覆盖真实标签的预测集。鲁棒 CP 将此保证扩展到最坏情况（对抗性）输入。现有Baseline通过限制随机平滑的一致性分数实现鲁棒性。实际上，它们需要昂贵的蒙特卡洛 (MC) 采样（例如，每个点约 $10^4$ 个样本）以保持可接受的集大小。我们提出了一种鲁棒预测，即使使用显著较少的MC样本（例如，CIFAR10 上为 150）也能生成更小的集。我们的方法使用可调（或自动调整）阈值对样本进行二值化，以保持覆盖保证。值得注意的是，我们证明了只需计算一个二值证书即可实现鲁棒性，而不像先前方法需要为每个校准（或测试）点进行验证。因此，我们的方法更快并返回更小的鲁棒集。我们还消除了先前需要有界评分函数的限制。 

---
# Kaiwu: A Multimodal Manipulation Dataset and Framework for Robot Learning and Human-Robot Interaction 

**Title (ZH)**: .Kaiwu：一种用于机器人学习和人机交互的多模态操作数据集及框架 

**Authors**: Shuo Jiang, Haonan Li, Ruochen Ren, Yanmin Zhou, Zhipeng Wang, Bin He  

**Link**: [PDF](https://arxiv.org/pdf/2503.05231)  

**Abstract**: Cutting-edge robot learning techniques including foundation models and imitation learning from humans all pose huge demands on large-scale and high-quality datasets which constitute one of the bottleneck in the general intelligent robot fields. This paper presents the Kaiwu multimodal dataset to address the missing real-world synchronized multimodal data problems in the sophisticated assembling scenario,especially with dynamics information and its fine-grained labelling. The dataset first provides an integration of human,environment and robot data collection framework with 20 subjects and 30 interaction objects resulting in totally 11,664 instances of integrated actions. For each of the demonstration,hand motions,operation pressures,sounds of the assembling process,multi-view videos, high-precision motion capture information,eye gaze with first-person videos,electromyography signals are all recorded. Fine-grained multi-level annotation based on absolute timestamp,and semantic segmentation labelling are performed. Kaiwu dataset aims to facilitate robot learning,dexterous manipulation,human intention investigation and human-robot collaboration research. 

**Abstract (ZH)**: 基于多模态数据的精密装配场景机器人学习数据集Kaiwu 

---
# Discrete Contrastive Learning for Diffusion Policies in Autonomous Driving 

**Title (ZH)**: 离散对比学习在自主驾驶中的扩散策略 

**Authors**: Kalle Kujanpää, Daulet Baimukashev, Farzeen Munir, Shoaib Azam, Tomasz Piotr Kucner, Joni Pajarinen, Ville Kyrki  

**Link**: [PDF](https://arxiv.org/pdf/2503.05229)  

**Abstract**: Learning to perform accurate and rich simulations of human driving behaviors from data for autonomous vehicle testing remains challenging due to human driving styles' high diversity and variance. We address this challenge by proposing a novel approach that leverages contrastive learning to extract a dictionary of driving styles from pre-existing human driving data. We discretize these styles with quantization, and the styles are used to learn a conditional diffusion policy for simulating human drivers. Our empirical evaluation confirms that the behaviors generated by our approach are both safer and more human-like than those of the machine-learning-based baseline methods. We believe this has the potential to enable higher realism and more effective techniques for evaluating and improving the performance of autonomous vehicles. 

**Abstract (ZH)**: 基于对比学习从数据中提取驾驶风格字典以实现自动驾驶汽车测试中的准确丰富的人类驾驶行为模拟仍然具有挑战性，因为人类驾驶风格具有高度的多样性和变异性。我们通过提出一种新颖的方法来应对这一挑战，该方法利用对比学习从现有的人类驾驶数据中提取驾驶风格字典。我们通过量化对这些风格进行离散化，并使用这些风格来学习一个条件扩散策略以模拟人类驾驶员。我们的实证评估表明，通过我们的方法生成的行为不仅更安全，而且更接近人类。我们相信这有可能提高自动驾驶汽车评估和性能提升的真实性和有效性。 

---
# MOHPER: Multi-objective Hyperparameter Optimization Framework for E-commerce Retrieval System 

**Title (ZH)**: MOHPER：电子商务检索系统多目标超参数优化框架 

**Authors**: Jungbae Park, Heonseok Jang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05227)  

**Abstract**: E-commerce search optimization has evolved to include a wider range of metrics that reflect user engagement and business objectives. Modern search frameworks now incorporate advanced quality features, such as sales counts and document-query relevance, to better align search results with these goals. Traditional methods typically focus on click-through rate (CTR) as a measure of engagement or relevance, but this can miss true purchase intent, creating a gap between user interest and actual conversions. Joint training with the click-through conversion rate (CTCVR) has become essential for understanding buying behavior, although its sparsity poses challenges for reliable optimization. This study presents MOHPER, a Multi-Objective Hyperparameter Optimization framework for E-commerce Retrieval systems. Utilizing Bayesian optimization and sampling, it jointly optimizes both CTR, CTCVR, and relevant objectives, focusing on engagement and conversion of the users. In addition, to improve the selection of the best configuration from multi-objective optimization, we suggest advanced methods for hyperparameter selection, including a meta-configuration voting strategy and a cumulative training approach that leverages prior optimal configurations, to improve speeds of training and efficiency. Currently deployed in a live setting, our proposed framework substantiates its practical efficacy in achieving a balanced optimization that aligns with both user satisfaction and revenue goals. 

**Abstract (ZH)**: 电子商务搜索优化已扩展到包括更广泛的指标，以反映用户参与度和企业目标。现代搜索框架现在整合了高级质量特征，如销售数量和文档查询相关性，以更好地使搜索结果与这些目标一致。传统方法通常以点击率（CTR）作为参与度或相关性的衡量标准，但可能会错过真实的购买意图，导致用户兴趣与实际转化之间产生差距。结合点击转化率（CTCVR）的联合训练已成为理解购买行为的必要手段，尽管其稀疏性给可靠的优化带来了挑战。本研究提出了一种面向电子商务检索系统的多目标超参数优化框架MOHPER。利用贝叶斯优化与采样，它联合优化了CTR、CTCVR以及相关目标，专注于用户的参与度和转化率。此外，为了提高多目标优化中最佳配置的选择，我们建议了高级超参数选择方法，包括元配置投票策略和利用先验最优配置的累积训练方法，以提高训练速度和效率。目前该框架已在实际环境中部署，证明其在兼顾用户满意度和收入目标方面具有实际效果。 

---
# Reward-Centered ReST-MCTS: A Robust Decision-Making Framework for Robotic Manipulation in High Uncertainty Environments 

**Title (ZH)**: 面向奖励中心的ReST-MCTS：高不确定性环境下机器人操控的稳健决策框架 

**Authors**: Xibai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05226)  

**Abstract**: Monte Carlo Tree Search (MCTS) has emerged as a powerful tool for decision-making in robotics, enabling efficient exploration of large search spaces. However, traditional MCTS methods struggle in environments characterized by high uncertainty and noisy data due to their reliance on final-step reward evaluation. The lack of intermediate feedback during search often results in suboptimal decision-making and computational inefficiencies.
This paper introduces Reward-Centered ReST-MCTS, a novel framework that enhances MCTS by incorporating intermediate reward shaping. The core of our approach is the Rewarding Center, which refines search trajectories by dynamically assigning partial rewards using rule-based validation, heuristic guidance, and neural estimation. By integrating these mechanisms, our method enables real-time optimization of search paths, mitigating the effects of error propagation.
We evaluate Reward-Centered ReST-MCTS in robotic manipulation tasks under high uncertainty, demonstrating consistent improvements in decision accuracy. Compared to baseline methods, including Chain-of-Thought (CoT) prompting and Vanilla ReST-MCTS, our framework achieves a 2-4% accuracy improvement while maintaining computational feasibility. Ablation studies confirm the effectiveness of intermediate feedback in search refinement, particularly in pruning incorrect decision paths early. Furthermore, robustness tests show that our method retains high performance across varying levels of uncertainty. 

**Abstract (ZH)**: 基于奖励中心的ReST-MCTS：一种增强的蒙特卡洛树搜索框架 

---
# Deep Sequence Models for Predicting Average Shear Wave Velocity from Strong Motion Records 

**Title (ZH)**: 基于强地面运动记录预测平均剪切波速的深层序列模型 

**Authors**: Baris Yilmaz, Erdem Akagündüz, Salih Tileylioglu  

**Link**: [PDF](https://arxiv.org/pdf/2503.05224)  

**Abstract**: This study explores the use of deep learning for predicting the time averaged shear wave velocity in the top 30 m of the subsurface ($V_{s30}$) at strong motion recording stations in Türkiye. $V_{s30}$ is a key parameter in site characterization and, as a result for seismic hazard assessment. However, it is often unavailable due to the lack of direct measurements and is therefore estimated using empirical correlations. Such correlations however are commonly inadequate in capturing complex, site-specific variability and this motivates the need for data-driven approaches. In this study, we employ a hybrid deep learning model combining convolutional neural networks (CNNs) and long short-term memory (LSTM) networks to capture both spatial and temporal dependencies in strong motion records. Furthermore, we explore how using different parts of the signal influence our deep learning model. Our results suggest that the hybrid approach effectively learns complex, nonlinear relationships within seismic signals. We observed that an improved P-wave arrival time model increased the prediction accuracy of $V_{s30}$. We believe the study provides valuable insights into improving $V_{s30}$ predictions using a CNN-LSTM framework, demonstrating its potential for improving site characterization for seismic studies. Our codes are available via this repo: this https URL 

**Abstract (ZH)**: 本研究探讨了使用深度学习预测土耳其强震记录站表层30米范围内的平均剪切波速度（$V_{s30}$）的方法。$V_{s30}$是场地表征中的关键参数，对于地震危害评估至关重要。然而，由于缺乏直接测量，$V_{s30}$往往不可用，因此通常使用经验相关方法来估算。然而，这些经验相关方法通常无法捕捉到复杂且场地特定的变化，因此本研究采用数据驱动的方法来解决这一问题。在本研究中，我们采用结合卷积神经网络（CNNs）和长短期记忆（LSTM）网络的混合深度学习模型，以捕捉强震记录中的时空依赖关系。此外，我们还探索了使用信号的不同部分如何影响深度学习模型。研究结果表明，混合方法能够有效地学习地震信号中的复杂非线性关系。我们发现，改进的P波到达时间模型提高了$V_{s30}$的预测精度。我们相信，本研究为使用CNN-LSTM框架改进$V_{s30}$预测提供了有价值的见解，展示了其在地震场地表征中的潜在应用价值。研究代码可通过以下链接获取：this https URL。 

---
# Knowledge Updating? No More Model Editing! Just Selective Contextual Reasoning 

**Title (ZH)**: 知识更新？不再模型编辑！只需选择性语境推理。 

**Authors**: Guoxiu He, Xin Song, Aixin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.05212)  

**Abstract**: As real-world knowledge evolves, the information embedded within large language models (LLMs) can become outdated, inadequate, or erroneous. Model editing has emerged as a prominent approach for updating LLMs' knowledge with minimal computational costs and parameter changes. This approach typically identifies and adjusts specific model parameters associated with newly acquired knowledge. However, existing methods often underestimate the adverse effects that parameter modifications can have on broadly distributed knowledge. More critically, post-edit LLMs frequently struggle with multi-hop reasoning and continuous knowledge updates. Although various studies have discussed these shortcomings, there is a lack of comprehensive evaluation. In this paper, we provide an evaluation of ten model editing methods along four dimensions: reliability, generalization, locality, and portability. Results confirm that all ten popular model editing methods show significant shortcomings across multiple dimensions, suggesting model editing is less promising. We then propose a straightforward method called Selective Contextual Reasoning (SCR), for knowledge updating. SCR does not modify model parameters but harnesses LLM's inherent contextual reasoning capabilities utilizing the updated knowledge pieces. Under SCR, an LLM first assesses whether an incoming query falls within the scope of an external knowledge base. If it does, the relevant external knowledge texts are contextualized to enhance reasoning; otherwise, the query is answered directly. We evaluate SCR against the ten model editing methods on two counterfactual datasets with three backbone LLMs. Empirical results confirm the effectiveness and efficiency of contextual reasoning for knowledge updating. 

**Abstract (ZH)**: 随着现实世界知识的演变，大型语言模型（LLMs）中嵌入的信息可能会变得过时、不足或错误。模型编辑已成为一种突出的方法，用于以最小的计算成本和参数变化更新LLMs的知识。这种方法 typically 通常会识别并调整与新获得知识相关的特定模型参数。然而，现有方法往往低估了参数修改对广泛分布知识的不良影响。更为关键的是，经过编辑的LLMs在多跳推理和连续知识更新方面时常遇到困难。尽管已有多种研究讨论了这些不足之处，但缺乏全面的评估。本文从可靠性、泛化能力、局部性和可移植性四个维度评估了十种模型编辑方法。结果表明，这十种流行的模型编辑方法在多个维度上均显示出显著的不足，建议模型编辑的效果有限。我们随后提出了一个名为Selective Contextual Reasoning（选择性上下文推理，SCR）的简单方法进行知识更新。SCR 不修改模型参数，而是利用LLMs固有的上下文推理能力，结合更新的知识片段。在SCR方法下，LLM首先评估传入查询是否属于外部知识库的范围。如果是，则将相关外部知识文本上下文化以增强推理；否则，直接回答查询。我们使用两个反事实数据集和三种基础LLM对SCR与十种模型编辑方法进行了评估。实验证明，上下文推理在知识更新中的有效性和效率。 

---
# Policy Constraint by Only Support Constraint for Offline Reinforcement Learning 

**Title (ZH)**: 仅基于约束的支持约束对offline reinforcement learning的策略进行限制 

**Authors**: Yunkai Gao, Jiaming Guo, Fan Wu, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05207)  

**Abstract**: Offline reinforcement learning (RL) aims to optimize a policy by using pre-collected datasets, to maximize cumulative rewards. However, offline reinforcement learning suffers challenges due to the distributional shift between the learned and behavior policies, leading to errors when computing Q-values for out-of-distribution (OOD) actions. To mitigate this issue, policy constraint methods aim to constrain the learned policy's distribution with the distribution of the behavior policy or confine action selection within the support of the behavior policy. However, current policy constraint methods tend to exhibit excessive conservatism, hindering the policy from further surpassing the behavior policy's performance. In this work, we present Only Support Constraint (OSC) which is derived from maximizing the total probability of learned policy in the support of behavior policy, to address the conservatism of policy constraint. OSC presents a regularization term that only restricts policies to the support without imposing extra constraints on actions within the support. Additionally, to fully harness the performance of the new policy constraints, OSC utilizes a diffusion model to effectively characterize the support of behavior policies. Experimental evaluations across a variety of offline RL benchmarks demonstrate that OSC significantly enhances performance, alleviating the challenges associated with distributional shifts and mitigating conservatism of policy constraints. Code is available at this https URL. 

**Abstract (ZH)**: 基于offline强化学习中的仅支持约束（Only Support Constraint，OSC）：缓解保守性并提升性能 

---
# Deep Muscle EMG construction using A Physics-Integrated Deep Learning approach 

**Title (ZH)**: 基于物理整合深度学习的深部肌肉EMG构建 

**Authors**: Rajnish Kumar, Tapas Tripura, Souvik Chakraborty, Sitikantha Roy  

**Link**: [PDF](https://arxiv.org/pdf/2503.05201)  

**Abstract**: Electromyography (EMG)--based computational musculoskeletal modeling is a non-invasive method for studying musculotendon function, human movement, and neuromuscular control, providing estimates of internal variables like muscle forces and joint torques. However, EMG signals from deeper muscles are often challenging to measure by placing the surface EMG electrodes and unfeasible to measure directly using invasive methods. The restriction to the access of EMG data from deeper muscles poses a considerable obstacle to the broad adoption of EMG-driven modeling techniques. A strategic alternative is to use an estimation algorithm to approximate the missing EMG signals from deeper muscle. A similar strategy is used in physics-informed deep learning, where the features of physical systems are learned without labeled data. In this work, we propose a hybrid deep learning algorithm, namely the neural musculoskeletal model (NMM), that integrates physics-informed and data-driven deep learning to approximate the EMG signals from the deeper muscles. While data-driven modeling is used to predict the missing EMG signals, physics-based modeling engraves the subject-specific information into the predictions. Experimental verifications on five test subjects are carried out to investigate the performance of the proposed hybrid framework. The proposed NMM is validated against the joint torque computed from 'OpenSim' software. The predicted deep EMG signals are also compared against the state-of-the-art muscle synergy extrapolation (MSE) approach, where the proposed NMM completely outperforms the existing MSE framework by a significant margin. 

**Abstract (ZH)**: 基于 Electromyography (EMG) 的计算肌骨建模是一种非侵入性方法，用于研究肌肉肌腱功能、人体运动和神经肌肉控制，提供肌肉力量和关节扭矩等内部变量的估计。然而，深层肌肉的 EMG 信号往往难以通过表面 EMG 电极测量，使用侵入性方法直接测量也未必可行。受限于深层肌肉 EMG 数据的获取，困扰了 EMG 驱动建模技术的大规模应用。一种战略性的替代方案是使用估计算法来近似深层肌肉缺失的 EMG 信号。类似的方法在物理启发式的深度学习中使用，无需标注数据即可学习物理系统的特征。本文提出了一种结合物理启发式和数据驱动深度学习的混合算法——神经肌骨模型 (NMM)，用以近似深层肌肉的 EMG 信号。基于数据驱动的方法预测缺失的 EMG 信号，基于物理的方法将个体特异性信息嵌入预测中。通过五个受试者的实验验证了所提出的混合框架的性能。提出的 NMM 模型被验证与来自 'OpenSim' 软件计算的关节扭矩一致。预测的深层 EMG 信号还与最先进的肌群协同扩展 (MSE) 方法进行了比较，结果显示所提出的 NMM 明显优于现有的 MSE 框架。 

---
# Uncertainty-Aware Explainable Federated Learning 

**Title (ZH)**: 不确定性意识可解释联邦学习 

**Authors**: Yanci Zhang, Han Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.05194)  

**Abstract**: Federated Learning (FL) is a collaborative machine learning paradigm for enhancing data privacy preservation. Its privacy-preserving nature complicates the explanation of the decision-making processes and the evaluation of the reliability of the generated explanations. In this paper, we propose the Uncertainty-aware eXplainable Federated Learning (UncertainXFL) to address these challenges. It generates explanations for decision-making processes under FL settings and provides information regarding the uncertainty of these explanations. UncertainXFL is the first framework to explicitly offer uncertainty evaluation for explanations within the FL context. Explanatory information is initially generated by the FL clients and then aggregated by the server in a comprehensive and conflict-free manner during FL training. The quality of the explanations, including the uncertainty score and tested validity, guides the FL training process by prioritizing clients with the most reliable explanations through higher weights during model aggregation. Extensive experimental evaluation results demonstrate that UncertainXFL achieves superior model accuracy and explanation accuracy, surpassing the current state-of-the-art model that does not incorporate uncertainty information by 2.71% and 1.77%, respectively. By integrating and quantifying uncertainty in the data into the explanation process, UncertainXFL not only clearly presents the explanation alongside its uncertainty, but also leverages this uncertainty to guide the FL training process, thereby enhancing the robustness and reliability of the resulting models. 

**Abstract (ZH)**: 联邦学习中的不确定性感知可解释联邦学习（Uncertainty-aware eXplainable Federated Learning） 

---
# Rewarding Curse: Analyze and Mitigate Reward Modeling Issues for LLM Reasoning 

**Title (ZH)**: 奖励之咒：分析和缓解大语言模型推理中的奖励建模问题 

**Authors**: Jiachun Li, Pengfei Cao, Yubo Chen, Jiexin Xu, Huaijun Li, Xiaojian Jiang, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.05188)  

**Abstract**: Chain-of-thought (CoT) prompting demonstrates varying performance under different reasoning tasks. Previous work attempts to evaluate it but falls short in providing an in-depth analysis of patterns that influence the CoT. In this paper, we study the CoT performance from the perspective of effectiveness and faithfulness. For the former, we identify key factors that influence CoT effectiveness on performance improvement, including problem difficulty, information gain, and information flow. For the latter, we interpret the unfaithful CoT issue by conducting a joint analysis of the information interaction among the question, CoT, and answer. The result demonstrates that, when the LLM predicts answers, it can recall correct information missing in the CoT from the question, leading to the problem. Finally, we propose a novel algorithm to mitigate this issue, in which we recall extra information from the question to enhance the CoT generation and evaluate CoTs based on their information gain. Extensive experiments demonstrate that our approach enhances both the faithfulness and effectiveness of CoT. 

**Abstract (ZH)**: 链式思考（CoT）引导在不同推理任务下的性能表现存在差异。先前的工作尝试对其进行评估，但未能深入分析影响CoT表现的因素模式。本文从有效性与忠实性两个视角研究CoT性能。在有效性方面，我们识别出影响CoT有效性提升的关键因素，包括问题难度、信息增益和信息流。在忠实性方面，我们通过联合分析问题、CoT和答案之间的信息交互来解析不忠实的CoT问题。结果表明，当大模型预测答案时，它可以从问题中召回缺失在CoT中的正确信息，导致问题。最后，我们提出了一种新颖的算法来缓解这一问题，在该算法中，我们从问题中召回额外信息以增强CoT生成，并基于信息增益评价CoT。广泛实验表明，我们的方法能够提升CoT的有效性和忠实性。 

---
# FinTMMBench: Benchmarking Temporal-Aware Multi-Modal RAG in Finance 

**Title (ZH)**: FinTMMBench: 基于时间感知的多模态RAG在金融领域的基准测试 

**Authors**: Fengbin Zhu, Junfeng Li, Liangming Pan, Wenjie Wang, Fuli Feng, Chao Wang, Huanbo Luan, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2503.05185)  

**Abstract**: Finance decision-making often relies on in-depth data analysis across various data sources, including financial tables, news articles, stock prices, etc. In this work, we introduce FinTMMBench, the first comprehensive benchmark for evaluating temporal-aware multi-modal Retrieval-Augmented Generation (RAG) systems in finance. Built from heterologous data of NASDAQ 100 companies, FinTMMBench offers three significant advantages. 1) Multi-modal Corpus: It encompasses a hybrid of financial tables, news articles, daily stock prices, and visual technical charts as the corpus. 2) Temporal-aware Questions: Each question requires the retrieval and interpretation of its relevant data over a specific time period, including daily, weekly, monthly, quarterly, and annual periods. 3) Diverse Financial Analysis Tasks: The questions involve 10 different tasks, including information extraction, trend analysis, sentiment analysis and event detection, etc. We further propose a novel TMMHybridRAG method, which first leverages LLMs to convert data from other modalities (e.g., tabular, visual and time-series data) into textual format and then incorporates temporal information in each node when constructing graphs and dense indexes. Its effectiveness has been validated in extensive experiments, but notable gaps remain, highlighting the challenges presented by our FinTMMBench. 

**Abstract (ZH)**: 金融决策往往依赖于对各种数据源的深入数据分析，包括财务报表、新闻文章、股票价格等。在此工作中，我们引入了FinTMMBench，这是首个用于评估金融市场中感知时间的多模态检索增强生成（RAG）系统的综合性基准。FinTMMBench基于纳斯达克100家公司异构数据构建，具有三大显著优势。1) 多模态语料库：其中包括财务报表、新闻文章、每日股票价格和可视化技术图表的混合体。2) 感知时间的问题：每个问题都需要在特定时间段内检索和解释其相关数据，包括日、周、月、季度和年度等时间段。3) 多样化的金融分析任务：问题涉及10种不同的任务，包括信息提取、趋势分析、情感分析和事件检测等。我们进一步提出了一种新型的TMMHybridRAG方法，该方法首先利用预训练语言模型将其他模态的数据（例如表、图和时间序列数据）转换为文本格式，然后在构建图和密集索引时在每个节点中融入时间信息。该方法已在大量实验中得到验证，但仍存在显著差距，突显了我们FinTMMBench带来的挑战。 

---
# Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching 

**Title (ZH)**: Thought-sketch: 有效的认知启发式素描推理的大型语言模型 

**Authors**: Simon A. Aytes, Jinheon Baek, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05179)  

**Abstract**: Recent advances in large language models have demonstrated remarkable reasoning capabilities through Chain of Thought (CoT) prompting, but often at the cost of excessive verbosity in their intermediate outputs, which increases computational overhead. We introduce Sketch-of-Thought (SoT), a novel prompting framework that combines cognitive-inspired reasoning paradigms with linguistic constraints to minimize token usage while preserving reasoning accuracy. SoT is designed as a flexible framework that can incorporate any custom reasoning paradigms based on cognitive science, and we instantiate it with three such paradigms - Conceptual Chaining, Chunked Symbolism, and Expert Lexicons - each tailored to different reasoning tasks and selected dynamically via a lightweight routing model. Through comprehensive evaluation across 15 reasoning datasets with multiple languages and multimodal scenarios, we demonstrate that SoT achieves token reductions of 76% with negligible accuracy impact. In certain domains like mathematical and multi-hop reasoning, it even improves accuracy while using significantly fewer tokens. Our code is publicly available: this https URL. 

**Abstract (ZH)**: 最近的大语言模型进展通过Chain of Thought (CoT)提示展示了卓越的推理能力，但常常以中间输出过度冗长为代价，增加了计算开销。我们引入了Sketch-of-Thought (SoT)这一新颖的提示框架，结合认知启发式的推理范式与语言约束，以最小化令牌使用量同时保持推理准确性。SoT设计为一个灵活框架，可以根据认知科学融合任意自定义推理范式，并通过一个轻量级路由模型动态选择三种范式中的任一种——概念链式推理、分块符号主义和专家领域词典，每种范式针对不同的推理任务。通过在15个跨语言和多模态场景的推理数据集中进行全面评估，我们证明SoT在不影响准确性的情况下实现了高达76%的令牌缩减。在某些领域，如数学和多步骤推理中，它甚至在使用显著更少的令牌时提高了准确性。我们的代码已公开：this https URL。 

---
# A Comprehensive LLM-powered Framework for Driving Intelligence Evaluation 

**Title (ZH)**: 基于大语言模型的综合驱动智能化评估框架 

**Authors**: Shanhe You, Xuewen Luo, Xinhe Liang, Jiashu Yu, Chen Zheng, Jiangtao Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.05164)  

**Abstract**: Evaluation methods for autonomous driving are crucial for algorithm optimization. However, due to the complexity of driving intelligence, there is currently no comprehensive evaluation method for the level of autonomous driving intelligence. In this paper, we propose an evaluation framework for driving behavior intelligence in complex traffic environments, aiming to fill this gap. We constructed a natural language evaluation dataset of human professional drivers and passengers through naturalistic driving experiments and post-driving behavior evaluation interviews. Based on this dataset, we developed an LLM-powered driving evaluation framework. The effectiveness of this framework was validated through simulated experiments in the CARLA urban traffic simulator and further corroborated by human assessment. Our research provides valuable insights for evaluating and designing more intelligent, human-like autonomous driving agents. The implementation details of the framework and detailed information about the dataset can be found at Github. 

**Abstract (ZH)**: 自动驾驶驾驶行为智能评价方法对于算法优化至关重要。然而，由于驾驶智能化的复杂性，目前尚无全面的自动驾驶智能化水平评价方法。本文提出了一种用于复杂交通环境下的驾驶行为智能评价框架，旨在填补这一空白。我们通过自然驾驶实验和驾驶后行为评估访谈，构建了一个基于自然语言的人类专业司机和乘客的评价数据集。基于该数据集，我们开发了一种基于大语言模型的驾驶评价框架。通过CARLA城市交通模拟器的模拟实验和进一步的人工评估验证了该框架的有效性。我们的研究为评估和设计更具智能化、更像人类的自动驾驶代理提供了有价值的见解。框架的实现细节和数据集的详细信息可在GitHub上找到。 

---
# Generative Trajectory Stitching through Diffusion Composition 

**Title (ZH)**: 通过扩散合成的生成轨迹拼接 

**Authors**: Yunhao Luo, Utkarsh A. Mishra, Yilun Du, Danfei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.05153)  

**Abstract**: Effective trajectory stitching for long-horizon planning is a significant challenge in robotic decision-making. While diffusion models have shown promise in planning, they are limited to solving tasks similar to those seen in their training data. We propose CompDiffuser, a novel generative approach that can solve new tasks by learning to compositionally stitch together shorter trajectory chunks from previously seen tasks. Our key insight is modeling the trajectory distribution by subdividing it into overlapping chunks and learning their conditional relationships through a single bidirectional diffusion model. This allows information to propagate between segments during generation, ensuring physically consistent connections. We conduct experiments on benchmark tasks of various difficulties, covering different environment sizes, agent state dimension, trajectory types, training data quality, and show that CompDiffuser significantly outperforms existing methods. 

**Abstract (ZH)**: 长 horizon 规划中有效的轨迹拼接是一项重要的机器人决策挑战。虽然扩散模型在规划方面展现了潜力，但它们仅限于解决与其训练数据相似的任务。我们提出了 CompDiffuser，一种新颖的生成性方法，能够通过学习将先前见过的任务中的较短轨迹片段组合起来解决新任务。我们的关键洞察是通过将轨迹分布细分重叠片段，并通过单一双向扩散模型学习它们的条件关系来建模轨迹分布。这使得在生成过程中片段之间能够传播信息，确保物理上的一致性连接。我们在涵盖不同环境大小、代理状态维度、轨迹类型、训练数据质量等多种难度基准任务上进行了实验，并展示了 CompDiffuser 显著优于现有方法。 

---
# Development and Enhancement of Text-to-Image Diffusion Models 

**Title (ZH)**: 文本到图像扩散模型的开发与优化 

**Authors**: Rajdeep Roshan Sahu  

**Link**: [PDF](https://arxiv.org/pdf/2503.05149)  

**Abstract**: This research focuses on the development and enhancement of text-to-image denoising diffusion models, addressing key challenges such as limited sample diversity and training instability. By incorporating Classifier-Free Guidance (CFG) and Exponential Moving Average (EMA) techniques, this study significantly improves image quality, diversity, and stability. Utilizing Hugging Face's state-of-the-art text-to-image generation model, the proposed enhancements establish new benchmarks in generative AI. This work explores the underlying principles of diffusion models, implements advanced strategies to overcome existing limitations, and presents a comprehensive evaluation of the improvements achieved. Results demonstrate substantial progress in generating stable, diverse, and high-quality images from textual descriptions, advancing the field of generative artificial intelligence and providing new foundations for future applications.
Keywords: Text-to-image, Diffusion model, Classifier-free guidance, Exponential moving average, Image generation. 

**Abstract (ZH)**: 本研究集中于文本到图像去噪扩散模型的开发与增强，针对样本多样性有限和训练不稳定等关键挑战。通过结合Classifier-Free Guidance (CFG)和Exponential Moving Average (EMA)技术，本研究显著提高了图像的质量、多样性和稳定性。利用Hugging Face的前沿文本到图像生成模型，所提出的研究增强建立了生成AI的新基准。本工作探索了扩散模型的基本原理，实施了先进的策略以克服现有限制，并全面评估了所取得的改进。结果表明，在从文本描述生成稳定、多样和高质量图像方面取得了实质性进步，推动了生成人工智能领域的发展，并为未来应用提供了新的基础。关键词：文本到图像、扩散模型、无分类器引导、指数移动平均、图像生成。 

---
# Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs 

**Title (ZH)**: 每一TFLOP都重要：无需高端GPU即可扩展3000亿参数的专家混合LLM 

**Authors**: Ling Team, Binwei Zeng, Chao Huang, Chao Zhang, Changxin Tian, Cong Chen, Dingnan Jin, Feng Yu, Feng Zhu, Feng Yuan, Fakang Wang, Gangshan Wang, Guangyao Zhai, Haitao Zhang, Huizhong Li, Jun Zhou, Jia Liu, Junpeng Fang, Junjie Ou, Jun Hu, Ji Luo, Ji Zhang, Jian Liu, Jian Sha, Jianxue Qian, Jiewei Wu, Junping Zhao, Jianguo Li, Jubao Feng, Jingchao Di, Junming Xu, Jinghua Yao, Kuan Xu, Kewei Du, Longfei Li, Lei Liang, Lu Yu, Li Tang, Lin Ju, Peng Xu, Qing Cui, Song Liu, Shicheng Li, Shun Song, Song Yan, Tengwei Cai, Tianyi Chen, Ting Guo, Ting Huang, Tao Feng, Tao Wu, Wei Wu, Xiaolu Zhang, Xueming Yang, Xin Zhao, Xiaobo Hu, Xin Lin, Yao Zhao, Yilong Wang, Yongzhen Guo, Yuanyuan Wang, Yue Yang, Yang Cao, Yuhao Fu, Yi Xiong, Yanzhe Li, Zhe Li, Zhiqiang Zhang, Ziqi Liu, Zhaoxin Huan, Zujie Wen, Zhenhang Sun, Zhuoxuan Du, Zhengyu He  

**Link**: [PDF](https://arxiv.org/pdf/2503.05139)  

**Abstract**: In this technical report, we tackle the challenges of training large-scale Mixture of Experts (MoE) models, focusing on overcoming cost inefficiency and resource limitations prevalent in such systems. To address these issues, we present two differently sized MoE large language models (LLMs), namely Ling-Lite and Ling-Plus (referred to as "Bailing" in Chinese, spelled Bǎilíng in Pinyin). Ling-Lite contains 16.8 billion parameters with 2.75 billion activated parameters, while Ling-Plus boasts 290 billion parameters with 28.8 billion activated parameters. Both models exhibit comparable performance to leading industry benchmarks. This report offers actionable insights to improve the efficiency and accessibility of AI development in resource-constrained settings, promoting more scalable and sustainable technologies. Specifically, to reduce training costs for large-scale MoE models, we propose innovative methods for (1) optimization of model architecture and training processes, (2) refinement of training anomaly handling, and (3) enhancement of model evaluation efficiency. Additionally, leveraging high-quality data generated from knowledge graphs, our models demonstrate superior capabilities in tool use compared to other models. Ultimately, our experimental findings demonstrate that a 300B MoE LLM can be effectively trained on lower-performance devices while achieving comparable performance to models of a similar scale, including dense and MoE models. Compared to high-performance devices, utilizing a lower-specification hardware system during the pre-training phase demonstrates significant cost savings, reducing computing costs by approximately 20%. The models can be accessed at this https URL. 

**Abstract (ZH)**: 本技术报告探讨了训练大规模专家混合模型（MoE）的挑战，重点在于克服此类系统中普遍存在的成本不效率和资源限制。为了解决这些问题，我们展示了两种不同规模的MoE大型语言模型（LLMs），分别是Ling-Lite和Ling-Plus（中文简称“摆灵”，拼音Bǎilíng）。Ling-Lite包含168亿参数，其中激活参数为2.75亿，而Ling-Plus则包含2900亿参数，激活参数为288亿。两者在性能上均与行业领先的标准相当。本报告提供了在资源受限环境中提高AI开发效率和可访问性的可行建议，促进更具扩展性和可持续性的技术。具体而言，为了减少大规模MoE模型的训练成本，我们提出了优化模型架构和训练过程、改进训练异常处理以及提高模型评估效率的创新方法。此外，利用从知识图谱中生成的高质量数据，我们的模型在工具使用能力上优于其他模型。最终，我们的实验结果表明，一个300亿参数的MoE LLM可以在较低性能的设备上有效训练，并达到与类似规模模型相当的性能，包括密集型和MoE模型。与高性能设备相比，在预训练阶段使用较低配置的硬件系统可以节省显著的计算成本，降低约20%的计算成本。这些模型可通过以下链接访问：这个 https URL。 

---
# HexPlane Representation for 3D Semantic Scene Understanding 

**Title (ZH)**: 适用于3D语义场景理解的HexPlane表示方法 

**Authors**: Zeren Chen, Yuenan Hou, Yulin Chen, Li Liu, Xiao Sun, Lu Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.05127)  

**Abstract**: In this paper, we introduce the HexPlane representation for 3D semantic scene understanding. Specifically, we first design the View Projection Module (VPM) to project the 3D point cloud into six planes to maximally retain the original spatial information. Features of six planes are extracted by the 2D encoder and sent to the HexPlane Association Module (HAM) to adaptively fuse the most informative information for each point. The fused point features are further fed to the task head to yield the ultimate predictions. Compared to the popular point and voxel representation, the HexPlane representation is efficient and can utilize highly optimized 2D operations to process sparse and unordered 3D point clouds. It can also leverage off-the-shelf 2D models, network weights, and training recipes to achieve accurate scene understanding in 3D space. On ScanNet and SemanticKITTI benchmarks, our algorithm, dubbed HexNet3D, achieves competitive performance with previous algorithms. In particular, on the ScanNet 3D segmentation task, our method obtains 77.0 mIoU on the validation set, surpassing Point Transformer V2 by 1.6 mIoU. We also observe encouraging results in indoor 3D detection tasks. Note that our method can be seamlessly integrated into existing voxel-based, point-based, and range-based approaches and brings considerable gains without bells and whistles. The codes will be available upon publication. 

**Abstract (ZH)**: 本文引入了HexPlane表示方法用于3D语义场景理解。具体而言，我们首先设计了视图投影模块（VPM）将3D点云投影到六个平面上，以最大程度保留原始的空间信息。六个平面上的特征由2D编码器提取，然后送入六边形平面关联模块（HAM）中，以自适应地融合每个点的最具信息量的信息。融合后的点特征进一步传递给任务头以产生最终预测。与流行的点和体素表示方法相比，HexPlane表示方法更高效，并能利用高度优化的2D操作来处理稀疏且无序的3D点云。此外，它还可以利用现成的2D模型、网络权重和训练方案在3D空间中实现准确的场景理解。在ScanNet和SemanticKITTI基准测试中，我们的算法HexNet3D在性能上与先前算法具有竞争力。特别是在ScanNet 3D分割任务中，我们的方法在验证集上获得了77.0的mIoU，超越了Point Transformer V2的1.6个mIoU。我们还在室内3D检测任务中观察到了令人鼓舞的结果。值得注意的是，我们的方法能无缝集成到现有的体素基、点基和距离基方法中，并且在不添加复杂功能的情况下带来显著改进。代码将在发表后公开。 

---
# Multi-Task Reinforcement Learning Enables Parameter Scaling 

**Title (ZH)**: 多任务强化学习实现参数缩放 

**Authors**: Reginald McLean, Evangelos Chataroulas, Jordan Terry, Isaac Woungang, Nariman Farsad, Pablo Samuel Castro  

**Link**: [PDF](https://arxiv.org/pdf/2503.05126)  

**Abstract**: Multi-task reinforcement learning (MTRL) aims to endow a single agent with the ability to perform well on multiple tasks. Recent works have focused on developing novel sophisticated architectures to improve performance, often resulting in larger models; it is unclear, however, whether the performance gains are a consequence of the architecture design itself or the extra parameters. We argue that gains are mostly due to scale by demonstrating that naively scaling up a simple MTRL baseline to match parameter counts outperforms the more sophisticated architectures, and these gains benefit most from scaling the critic over the actor. Additionally, we explore the training stability advantages that come with task diversity, demonstrating that increasing the number of tasks can help mitigate plasticity loss. Our findings suggest that MTRL's simultaneous training across multiple tasks provides a natural framework for beneficial parameter scaling in reinforcement learning, challenging the need for complex architectural innovations. 

**Abstract (ZH)**: 多任务强化学习（MTRL）旨在赋予单个智能体在多个任务上表现出色的能力。近期研究重点在于开发新颖复杂的架构以提高性能，通常会导致模型规模增大；然而，性能提升的原因是由于架构设计本身还是额外的参数尚不明确。我们 argue 认为增益主要源于规模的扩大，通过展示简单 MTRL 基线模型盲目扩大以匹配参数数量的表现超过了更为复杂的架构，并且这些增益主要得益于对价值函数而非策略函数的规模扩大。此外，我们还探讨了任务多样性带来的训练稳定性优势，证明增加任务数量有助于减轻模型适应性损失。我们的发现表明，MTRL 跨多个任务的同时训练为强化学习中的有益参数规模扩大提供了一个自然框架，挑战了复杂架构创新的必要性。 

---
# Look Before You Leap: Using Serialized State Machine for Language Conditioned Robotic Manipulation 

**Title (ZH)**: 未雨绸缪：使用序列化状态机进行语言条件驱动的机器人操作 

**Authors**: Tong Mu, Yihao Liu, Mehran Armand  

**Link**: [PDF](https://arxiv.org/pdf/2503.05114)  

**Abstract**: Imitation learning frameworks for robotic manipulation have drawn attention in the recent development of language model grounded robotics. However, the success of the frameworks largely depends on the coverage of the demonstration cases: When the demonstration set does not include examples of how to act in all possible situations, the action may fail and can result in cascading errors. To solve this problem, we propose a framework that uses serialized Finite State Machine (FSM) to generate demonstrations and improve the success rate in manipulation tasks requiring a long sequence of precise interactions. To validate its effectiveness, we use environmentally evolving and long-horizon puzzles that require long sequential actions. Experimental results show that our approach achieves a success rate of up to 98 in these tasks, compared to the controlled condition using existing approaches, which only had a success rate of up to 60, and, in some tasks, almost failed completely. 

**Abstract (ZH)**: 基于语言模型导向的机器人操控仿存学习框架：通过序列化有限状态机生成示范以提高成功率 

---
# TS-LIF: A Temporal Segment Spiking Neuron Network for Time Series Forecasting 

**Title (ZH)**: TS-LIF: 一种用于时间序列预测的时序段脉冲神经网络 

**Authors**: Shibo Feng, Wanjin Feng, Xingyu Gao, Peilin Zhao, Zhiqi Shen  

**Link**: [PDF](https://arxiv.org/pdf/2503.05108)  

**Abstract**: Spiking Neural Networks (SNNs) offer a promising, biologically inspired approach for processing spatiotemporal data, particularly for time series forecasting. However, conventional neuron models like the Leaky Integrate-and-Fire (LIF) struggle to capture long-term dependencies and effectively process multi-scale temporal dynamics. To overcome these limitations, we introduce the Temporal Segment Leaky Integrate-and-Fire (TS-LIF) model, featuring a novel dual-compartment architecture. The dendritic and somatic compartments specialize in capturing distinct frequency components, providing functional heterogeneity that enhances the neuron's ability to process both low- and high-frequency information. Furthermore, the newly introduced direct somatic current injection reduces information loss during intra-neuronal transmission, while dendritic spike generation improves multi-scale information extraction. We provide a theoretical stability analysis of the TS-LIF model and explain how each compartment contributes to distinct frequency response characteristics. Experimental results show that TS-LIF outperforms traditional SNNs in time series forecasting, demonstrating better accuracy and robustness, even with missing data. TS-LIF advances the application of SNNs in time-series forecasting, providing a biologically inspired approach that captures complex temporal dynamics and offers potential for practical implementation in diverse forecasting scenarios. The source code is available at this https URL. 

**Abstract (ZH)**: 时空段漏积分火（TS-LIF）模型：用于时间序列预测的生物启发式方法 

---
# Grouped Sequential Optimization Strategy -- the Application of Hyperparameter Importance Assessment in Deep Learning 

**Title (ZH)**: 分组序贯优化策略——超参数重要性评估在深度学习中的应用 

**Authors**: Ruinan Wang, Ian Nabney, Mohammad Golbabaee  

**Link**: [PDF](https://arxiv.org/pdf/2503.05106)  

**Abstract**: Hyperparameter optimization (HPO) is a critical component of machine learning pipelines, significantly affecting model robustness, stability, and generalization. However, HPO is often a time-consuming and computationally intensive task. Traditional HPO methods, such as grid search and random search, often suffer from inefficiency. Bayesian optimization, while more efficient, still struggles with high-dimensional search spaces. In this paper, we contribute to the field by exploring how insights gained from hyperparameter importance assessment (HIA) can be leveraged to accelerate HPO, reducing both time and computational resources. Building on prior work that quantified hyperparameter importance by evaluating 10 hyperparameters on CNNs using 10 common image classification datasets, we implement a novel HPO strategy called 'Sequential Grouping.' That prior work assessed the importance weights of the investigated hyperparameters based on their influence on model performance, providing valuable insights that we leverage to optimize our HPO process. Our experiments, validated across six additional image classification datasets, demonstrate that incorporating hyperparameter importance assessment (HIA) can significantly accelerate HPO without compromising model performance, reducing optimization time by an average of 31.9\% compared to the conventional simultaneous strategy. 

**Abstract (ZH)**: 基于超参数重要性评估的加速超参数优化方法 

---
# Multi-Robot Collaboration through Reinforcement Learning and Abstract Simulation 

**Title (ZH)**: 通过强化学习和抽象模拟的多机器人协作 

**Authors**: Adam Labiosa, Josiah P. Hanna  

**Link**: [PDF](https://arxiv.org/pdf/2503.05092)  

**Abstract**: Teams of people coordinate to perform complex tasks by forming abstract mental models of world and agent dynamics. The use of abstract models contrasts with much recent work in robot learning that uses a high-fidelity simulator and reinforcement learning (RL) to obtain policies for physical robots. Motivated by this difference, we investigate the extent to which so-called abstract simulators can be used for multi-agent reinforcement learning (MARL) and the resulting policies successfully deployed on teams of physical robots. An abstract simulator models the robot's target task at a high-level of abstraction and discards many details of the world that could impact optimal decision-making. Policies are trained in an abstract simulator then transferred to the physical robot by making use of separately-obtained low-level perception and motion control modules. We identify three key categories of modifications to the abstract simulator that enable policy transfer to physical robots: simulation fidelity enhancements, training optimizations and simulation stochasticity. We then run an empirical study with extensive ablations to determine the value of each modification category for enabling policy transfer in cooperative robot soccer tasks. We also compare the performance of policies produced by our method with a well-tuned non-learning-based behavior architecture from the annual RoboCup competition and find that our approach leads to a similar level of performance. Broadly we show that MARL can be use to train cooperative physical robot behaviors using highly abstract models of the world. 

**Abstract (ZH)**: 团队成员通过构建抽象的心理模型来协作完成复杂任务，这些模型概括了世界和代理的动力学。与机器人学习中广泛使用的高保真仿真器和强化学习（RL）获得物理机器人策略的方法不同，我们探索所谓的抽象仿真器在多智能体强化学习（MARL）中的应用及其生成的策略在物理机器人团队上的部署效果。抽象仿真器以高层次的抽象概括机器人目标任务，摒弃了许多可能影响最优决策的世界细节。策略在抽象仿真器中训练，然后通过使用分别获得的低级感知和运动控制模块转移到物理机器人。我们确定了三种关键的抽象仿真器修改类别，这些修改能促进策略向物理机器人的转移：仿真保真度提升、训练优化和仿真随机性。接着，我们进行了一系列详尽的实验消融分析，以确定每个修改类别在协同机器人足球任务中促进策略转移的价值。我们还将通过本方法生成的策略性能与年度RoboCup竞赛中 WELL 调参的非学习基于行为架构进行比较，发现我们的方法能达到相似的性能水平。总体而言，我们展示了MARL可以使用高度抽象的世界模型来训练协同物理机器人的行为。 

---
# Object Packing and Scheduling for Sequential 3D Printing: a Linear Arithmetic Model and a CEGAR-inspired Optimal Solver 

**Title (ZH)**: 面向顺序3D打印的物体打包与调度：线性算术模型及CEGAR启发式最优求解器 

**Authors**: Pavel Surynek, Vojtěch Bubník, Lukáš Matěna, Petr Kubiš  

**Link**: [PDF](https://arxiv.org/pdf/2503.05071)  

**Abstract**: We address the problem of object arrangement and scheduling for sequential 3D printing. Unlike the standard 3D printing, where all objects are printed slice by slice at once, in sequential 3D printing, objects are completed one after other. In the sequential case, it is necessary to ensure that the moving parts of the printer do not collide with previously printed objects. We look at the sequential printing problem from the perspective of combinatorial optimization. We propose to express the problem as a linear arithmetic formula, which is then solved using a solver for satisfiability modulo theories (SMT). However, we do not solve the formula expressing the problem of object arrangement and scheduling directly, but we have proposed a technique inspired by counterexample guided abstraction refinement (CEGAR), which turned out to be a key innovation to efficiency. 

**Abstract (ZH)**: 序列3D打印中的对象布置与调度问题 

---
# PromptPex: Automatic Test Generation for Language Model Prompts 

**Title (ZH)**: PromptPex：自动生成语言模型提示的测试用例 

**Authors**: Reshabh K Sharma, Jonathan De Halleux, Shraddha Barke, Benjamin Zorn  

**Link**: [PDF](https://arxiv.org/pdf/2503.05070)  

**Abstract**: Large language models (LLMs) are being used in many applications and prompts for these models are integrated into software applications as code-like artifacts. These prompts behave much like traditional software in that they take inputs, generate outputs, and perform some specific function. However, prompts differ from traditional code in many ways and require new approaches to ensure that they are robust. For example, unlike traditional software the output of a prompt depends on the AI model that interprets it. Also, while natural language prompts are easy to modify, the impact of updates is harder to predict. New approaches to testing, debugging, and modifying prompts with respect to the model running them are required.
To address some of these issues, we developed PromptPex, an LLM-based tool to automatically generate and evaluate unit tests for a given prompt. PromptPex extracts input and output specifications from a prompt and uses them to generate diverse, targeted, and valid unit tests. These tests are instrumental in identifying regressions when a prompt is changed and also serve as a tool to understand how prompts are interpreted by different models. We use PromptPex to generate tests for eight benchmark prompts and evaluate the quality of the generated tests by seeing if they can cause each of four diverse models to produce invalid output. PromptPex consistently creates tests that result in more invalid model outputs than a carefully constructed baseline LLM-based test generator. Furthermore, by extracting concrete specifications from the input prompt, PromptPex allows prompt writers to clearly understand and test specific aspects of their prompts. The source code of PromptPex is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多种应用中被使用，这些模型的提示作为代码-like制品集成到软件应用中。这些提示在许多方面类似于传统软件，它们接受输入、生成输出并执行特定功能。然而，提示与传统代码有许多不同之处，需要采取新的方法以确保其稳健性。例如，与传统软件不同，提示的输出取决于解释它的AI模型。此外，虽然自然语言提示容易修改，但更新的影响更难预测。对提示与运行它们的模型相关的测试、调试和修改方法需要新的方法。

为解决部分问题，我们开发了PromptPex，这是一种基于LLM的工具，用于自动为给定的提示生成和评估单元测试。PromptPex从提示中提取输入和输出规范，并使用它们生成多样、针对性且有效的单元测试。这些测试对于提示更改时识别回归至关重要，同时也作为工具来理解不同模型如何解释提示。我们使用PromptPex为八个基准提示生成测试，并通过评估生成测试能否导致四个不同模型生成无效输出来衡量测试质量。PromptPex生成的测试始终导致比精心构建的基本LLM测试生成器更多的无效模型输出。此外，通过从输入提示中提取具体的规范，PromptPex允许提示编写者清晰地理解和测试其提示的具体方面。PromptPex的源代码可从此链接获取。 

---
# Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts 

**Title (ZH)**: 容量感知推断：减轻混合专家模型中的游荡者效应 

**Authors**: Shwai He, Weilin Cai, Jiayi Huang, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.05066)  

**Abstract**: The Mixture of Experts (MoE) is an effective architecture for scaling large language models by leveraging sparse expert activation, optimizing the trade-off between performance and efficiency. However, under expert parallelism, MoE suffers from inference inefficiencies due to imbalanced token-to-expert assignment, where some experts are overloaded while others remain underutilized. This imbalance leads to poor resource utilization and increased latency, as the most burdened expert dictates the overall delay, a phenomenon we define as the \textbf{\textit{Straggler Effect}}. To mitigate this, we propose Capacity-Aware Inference, including two key techniques: (1) \textbf{\textit{Capacity-Aware Token Drop}}, which discards overloaded tokens to regulate the maximum latency of MoE, and (2) \textbf{\textit{Capacity-Aware Token Reroute}}, which reallocates overflowed tokens to underutilized experts, balancing the token distribution. These techniques collectively optimize both high-load and low-load expert utilization, leading to a more efficient MoE inference pipeline. Extensive experiments demonstrate the effectiveness of our methods, showing significant improvements in inference efficiency, e.g., 0.2\% average performance increase and a 1.94$\times$ inference speedup on Mixtral-8$\times$7B-Instruct. 

**Abstract (ZH)**: 专家混合模型（MoE）的有效扩展架构通过利用稀疏专家激活来扩展大规模语言模型，优化性能与效率之间的 trade-off。然而，在专家并行处理下，MoE 因 token-to-expert 分配不平衡而导致推理效率低下，一些专家超载而另一些则利用不足。这种不平衡导致资源利用效率低下和延迟增加，我们将其现象定义为“拖后腿效应”（Straggler Effect）。为缓解这一问题，我们提出了容量感知推理，包括两种关键技术：（1）容量感知 token 筛选（Capacity-Aware Token Drop），通过丢弃超载 token 来调节 MoE 的最大延迟；（2）容量感知 token 重分配（Capacity-Aware Token Reroute），通过将 overflowed token 重新分配给利用不足的专家，平衡 token 分布。这些技术共同优化高负载和低负载专家的利用，从而实现更高效的 MoE 推理管道。大量实验证明了我们方法的有效性，显示出推理效率的显著提高，例如平均性能提高 0.2%，Mixedral-8×7B-Instruct 上推理速度提升 1.94 倍。 

---
# Perceiving, Reasoning, Adapting: A Dual-Layer Framework for VLM-Guided Precision Robotic Manipulation 

**Title (ZH)**: 感知、推理、适应：面向VLM引导的精确机器人操作的双层框架 

**Authors**: Qingxuan Jia, Guoqin Tang, Zeyuan Huang, Zixuan Hao, Ning Ji, Shihang, Gang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.05064)  

**Abstract**: Vision-Language Models (VLMs) demonstrate remarkable potential in robotic manipulation, yet challenges persist in executing complex fine manipulation tasks with high speed and precision. While excelling at high-level planning, existing VLM methods struggle to guide robots through precise sequences of fine motor actions. To address this limitation, we introduce a progressive VLM planning algorithm that empowers robots to perform fast, precise, and error-correctable fine manipulation. Our method decomposes complex tasks into sub-actions and maintains three key data structures: task memory structure, 2D topology graphs, and 3D spatial networks, achieving high-precision spatial-semantic fusion. These three components collectively accumulate and store critical information throughout task execution, providing rich context for our task-oriented VLM interaction mechanism. This enables VLMs to dynamically adjust guidance based on real-time feedback, generating precise action plans and facilitating step-wise error correction. Experimental validation on complex assembly tasks demonstrates that our algorithm effectively guides robots to rapidly and precisely accomplish fine manipulation in challenging scenarios, significantly advancing robot intelligence for precision tasks. 

**Abstract (ZH)**: Vision-Language模型在机器人精细操作中的潜力显著，但在执行复杂精细操作任务时仍面临高速度和高精度的挑战。现有的VLM方法在高层次规划方面表现出色，但在指导机器人执行精细动作的精确序列方面存在局限性。为解决这一局限性，我们提出了一种渐进式VLM规划算法，使机器人能够进行快速、精确且可纠正误差的精细操作。我们的方法将复杂任务分解为子动作，并维护三种关键数据结构：任务记忆结构、2D拓扑图和3D空间网络，实现高精度的空间语义融合。这三种组件在整个任务执行过程中持续积累和存储关键信息，为我们的任务导向VLM交互机制提供丰富的上下文。这使得VLM能够根据实时反馈动态调整指导，生成精确的动作计划并促进逐步错误纠正。在复杂装配任务上的实验验证表明，我们的算法能够有效地指导机器人在挑战性场景中快速且精确地完成精细操作，显著推进了机器人在精确任务中的智能水平。 

---
# Accelerated Patient-specific Non-Cartesian MRI Reconstruction using Implicit Neural Representations 

**Title (ZH)**: 基于隐式神经表示的加速患者特定非笛卡尔MRI重建 

**Authors**: Di Xu, Hengjie Liu, Xin Miao, Daniel O'Connor, Jessica E. Scholey, Wensha Yang, Mary Feng, Michael Ohliger, Hui Lin, Dan Ruan, Yang Yang, Ke Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.05051)  

**Abstract**: The scanning time for a fully sampled MRI can be undesirably lengthy. Compressed sensing has been developed to minimize image artifacts in accelerated scans, but the required iterative reconstruction is computationally complex and difficult to generalize on new cases. Image-domain-based deep learning methods (e.g., convolutional neural networks) emerged as a faster alternative but face challenges in modeling continuous k-space, a problem amplified with non-Cartesian sampling commonly used in accelerated acquisition. In comparison, implicit neural representations can model continuous signals in the frequency domain and thus are compatible with arbitrary k-space sampling patterns. The current study develops a novel generative-adversarially trained implicit neural representations (k-GINR) for de novo undersampled non-Cartesian k-space reconstruction. k-GINR consists of two stages: 1) supervised training on an existing patient cohort; 2) self-supervised patient-specific optimization. In stage 1, the network is trained with the generative-adversarial network on diverse patients of the same anatomical region supervised by fully sampled acquisition. In stage 2, undersampled k-space data of individual patients is used to tailor the prior-embedded network for patient-specific optimization. The UCSF StarVIBE T1-weighted liver dataset was evaluated on the proposed framework. k-GINR is compared with an image-domain deep learning method, Deep Cascade CNN, and a compressed sensing method. k-GINR consistently outperformed the baselines with a larger performance advantage observed at very high accelerations (e.g., 20 times). k-GINR offers great value for direct non-Cartesian k-space reconstruction for new incoming patients across a wide range of accelerations liver anatomy. 

**Abstract (ZH)**: 基于隐式神经表示的生成对抗训练方法用于非笛卡尔编码下采样k空间重建 

---
# Provably Correct Automata Embeddings for Optimal Automata-Conditioned Reinforcement Learning 

**Title (ZH)**: 可证明正确的自动机嵌入以实现最优自动机条件强化学习 

**Authors**: Beyazit Yalcinkaya, Niklas Lauffer, Marcell Vazquez-Chanlatte, Sanjit A. Seshia  

**Link**: [PDF](https://arxiv.org/pdf/2503.05042)  

**Abstract**: Automata-conditioned reinforcement learning (RL) has given promising results for learning multi-task policies capable of performing temporally extended objectives given at runtime, done by pretraining and freezing automata embeddings prior to training the downstream policy. However, no theoretical guarantees were given. This work provides a theoretical framework for the automata-conditioned RL problem and shows that it is probably approximately correct learnable. We then present a technique for learning provably correct automata embeddings, guaranteeing optimal multi-task policy learning. Our experimental evaluation confirms these theoretical results. 

**Abstract (ZH)**: 基于自动机条件的强化学习（RL）在学习能够执行运行时给定的临时扩展目标的多任务策略方面提供了令人鼓舞的结果，通过在训练下游策略之前预训练并冻结自动机嵌入来实现。然而没有给出理论保证。本工作提供了一个基于自动机条件的RL问题的理论框架，并证明它是可能近似正确的学习的。我们随后提出了一种学习可证明正确的自动机嵌入的技术，确保多任务策略学习的最优性。实验评估证实了这些理论结果。 

---
# Enhancing Alzheimer's Diagnosis: Leveraging Anatomical Landmarks in Graph Convolutional Neural Networks on Tetrahedral Meshes 

**Title (ZH)**: 基于四面体网格的图卷积神经网络中的解剖标志增强阿尔茨海默病诊断 

**Authors**: Yanxi Chen, Mohammad Farazi, Zhangsihao Yang, Yonghui Fan, Nicholas Ashton, Eric M Reiman, Yi Su, Yalin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05031)  

**Abstract**: Alzheimer's disease (AD) is a major neurodegenerative condition that affects millions around the world. As one of the main biomarkers in the AD diagnosis procedure, brain amyloid positivity is typically identified by positron emission tomography (PET), which is costly and invasive. Brain structural magnetic resonance imaging (sMRI) may provide a safer and more convenient solution for the AD diagnosis. Recent advances in geometric deep learning have facilitated sMRI analysis and early diagnosis of AD. However, determining AD pathology, such as brain amyloid deposition, in preclinical stage remains challenging, as less significant morphological changes can be observed. As a result, few AD classification models are generalizable to the brain amyloid positivity classification task. Blood-based biomarkers (BBBMs), on the other hand, have recently achieved remarkable success in predicting brain amyloid positivity and identifying individuals with high risk of being brain amyloid positive. However, individuals in medium risk group still require gold standard tests such as Amyloid PET for further evaluation. Inspired by the recent success of transformer architectures, we propose a geometric deep learning model based on transformer that is both scalable and robust to variations in input volumetric mesh size. Our work introduced a novel tokenization scheme for tetrahedral meshes, incorporating anatomical landmarks generated by a pre-trained Gaussian process model. Our model achieved superior classification performance in AD classification task. In addition, we showed that the model was also generalizable to the brain amyloid positivity prediction with individuals in the medium risk class, where BM alone cannot achieve a clear classification. Our work may enrich geometric deep learning research and improve AD diagnosis accuracy without using expensive and invasive PET scans. 

**Abstract (ZH)**: 阿尔茨海默病（AD）是一种严重影响全世界数百万人的重大神经退行性疾病。作为AD诊断过程中的主要生物标志物之一，脑淀粉样蛋白阳性通常通过正电子发射断层扫描（PET）来识别，这种方法成本高且侵入性大。脑结构磁共振成像（sMRI）可能提供一种更安全、更方便的AD诊断解决方案。几何深度学习的最新进展促进了sMRI分析和AD的早期诊断。然而，在预临床阶段确定AD病理，如脑淀粉样蛋白沉积，仍然具有挑战性，因为观察到的形态学变化较少。因此，很少有AD分类模型能够推广到脑淀粉样蛋白阳性分类任务。另一方面，基于血液的生物标志物（BBBMs）最近在预测脑淀粉样蛋白阳性以及识别高风险个体方面取得了显著成功。然而，中风险个体仍需要使用如淀粉样蛋白PET等黄金标准测试进行进一步评估。受近期变压器架构成功的启发，我们提出了一种基于变压器的几何深度学习模型，该模型具有可扩展性和对输入体素网格尺寸变化的鲁棒性。我们的工作引入了一种新的四面体网格分词方案，结合了预训练高斯过程模型生成的解剖学landmark。我们的模型在AD分类任务中实现了卓越的分类性能。此外，我们还展示了该模型在中风险个体的脑淀粉样蛋白阳性预测方面具有泛化能力，而单独使用血液生物标志物无法清晰分类。我们的工作可能丰富几何深度学习研究，并在不使用昂贵且侵入性的PET扫描的情况下提高AD的诊断准确性。 

---
# Continual Pre-training of MoEs: How robust is your router? 

**Title (ZH)**: MoEs的持续预训练：你的路由器有多 robust？ 

**Authors**: Benjamin Thérien, Charles-Étienne Joseph, Zain Sarwar, Ashwinee Panda, Anirban Das, Shi-Xiong Zhang, Stephen Rawls, Sambit Sahu, Eugene Belilovsky, Irina Rish  

**Link**: [PDF](https://arxiv.org/pdf/2503.05029)  

**Abstract**: Sparsely-activated Mixture of Experts (MoE) transformers are promising architectures for foundation models. Compared to dense transformers that require the same amount of floating point operations (FLOPs) per forward pass, MoEs benefit from improved sample efficiency at training time and achieve much stronger performance. Many closed-source and open-source frontier language models have thus adopted an MoE architecture. Naturally, practitioners will want to extend the capabilities of these models with large amounts of newly collected data without completely re-training them. Prior work has shown that a simple combination of replay and learning rate re-warming and re-decaying can enable the continual pre-training (CPT) of dense decoder-only transformers with minimal performance degradation compared to full re-training. In the case of decoder-only MoE transformers, however, it is unclear how the routing algorithm will impact continual pre-training performance: 1) do the MoE transformer's routers exacerbate forgetting relative to a dense model?; 2) do the routers maintain a balanced load on previous distributions after CPT?; 3) are the same strategies applied to dense models sufficient to continually pre-train MoE LLMs? In what follows, we conduct a large-scale (>2B parameter switch and DeepSeek MoE LLMs trained for 600B tokens) empirical study across four MoE transformers to answer these questions. Our results establish a surprising robustness to distribution shifts for both Sinkhorn-Balanced and Z-and-Aux-loss-balanced routing algorithms, even in MoEs continually pre-trained without replay. Moreover, we show that MoE LLMs maintain their sample efficiency (relative to a FLOP-matched dense model) during CPT and that they can match the performance of a fully re-trained MoE at a fraction of the cost. 

**Abstract (ZH)**: 稀疏激活 experts 混合的变换器架构（MoE）是基础模型的有前途的架构。与需要每次前向传递相同浮点运算（FLOPs）数量的密集变换器相比，MoE 在训练时受益于改进的样本效率并实现更强大的性能。因此，许多闭源和开源前沿语言模型采用了 MoE 架构。自然地，实践者希望利用大量新收集的数据扩展这些模型的能力，而无需完全重新训练它们。以往的工作表明，简单的回放与学习率重新温暖和衰减相结合可以使得密集的解码器-only 变换器的持续预训练（CPT）在与完全重新训练相比的最小性能下降下得以实现。然而，在解码器-only MoE 变换器的情况下，路由算法对持续预训练性能的影响尚不清楚：1）MoE 变换器的路由器是否相对于密集模型加剧了遗忘？2）路由器在持续预训练后是否能够保持对以前分布的平衡负担？3）应用于密集模型的策略是否足够用于持续预训练 MoE 大型语言模型？我们随后通过四类 MoE 变换器进行的一项大规模（超过 2 亿参数的切换和 DeepSeek MoE 大型语言模型，训练了 600 亿个标记）实证研究来回答这些问题。我们的结果表明，即使在没有回放的情况下持续预训练 MoE 中，Sinkhorn-Balanced 和 Z-and-Aux-loss-balanced 路由算法也表现出惊人的鲁棒性。此外，我们证明，在持续预训练中，MoE 大型语言模型保持了与 FLOP 匹配的密集模型相当的样本效率，并且它们可以在极低的成本下匹配完全重新训练的 MoE 的性能。 

---
# LLMs' Reshaping of People, Processes, Products, and Society in Software Development: A Comprehensive Exploration with Early Adopters 

**Title (ZH)**: LLMs对软件开发中的人、流程、产品和社会的重塑：早期采用者的全面探索 

**Authors**: Benyamin Tabarsi, Heidi Reichert, Ally Limke, Sandeep Kuttal, Tiffany Barnes  

**Link**: [PDF](https://arxiv.org/pdf/2503.05012)  

**Abstract**: Large language models (LLMs) like OpenAI ChatGPT, Google Gemini, and GitHub Copilot are rapidly gaining traction in the software industry, but their full impact on software engineering remains insufficiently explored. Despite their growing adoption, there is a notable lack of formal, qualitative assessments of how LLMs are applied in real-world software development contexts. To fill this gap, we conducted semi-structured interviews with sixteen early-adopter professional developers to explore their use of LLMs throughout various stages of the software development life cycle. Our investigation examines four dimensions: people - how LLMs affect individual developers and teams; process - how LLMs alter software engineering workflows; product - LLM impact on software quality and innovation; and society - the broader socioeconomic and ethical implications of LLM adoption. Thematic analysis of our data reveals that while LLMs have not fundamentally revolutionized the development process, they have substantially enhanced routine coding tasks, including code generation, refactoring, and debugging. Developers reported the most effective outcomes when providing LLMs with clear, well-defined problem statements, indicating that LLMs excel with decomposed problems and specific requirements. Furthermore, these early-adopters identified that LLMs offer significant value for personal and professional development, aiding in learning new languages and concepts. Early-adopters, highly skilled in software engineering and how LLMs work, identified early and persisting challenges for software engineering, such as inaccuracies in generated content and the need for careful manual review before integrating LLM outputs into production environments. Our study provides a nuanced understanding of how LLMs are shaping the landscape of software development, with their benefits, limitations, and ongoing implications. 

**Abstract (ZH)**: 大型语言模型（LLMs）如OpenAI ChatGPT、Google Gemini和GitHub Copilot在软件行业中的应用正迅速增长，但其对软件工程的全面影响尚未充分探讨。尽管其采用率不断增加，但对LLMs在实际软件开发环境中的应用进行了正式和定性评估的情况仍然较少。为填补这一空白，我们对十六名早期采用者的专业开发人员进行了半结构化访谈，以探索他们在软件开发生命周期各阶段使用LLMs的情况。我们的研究考察了四个维度：人——LLMs如何影响个体开发人员和团队；过程——LLMs如何改变软件工程工作流程；产品——LLMs对软件质量和创新的影响；社会——LLMs采用的更广泛的经济社会和伦理影响。对我们的数据进行主题分析揭示了以下内容：虽然LLMs尚未根本改变开发流程，但它们极大地提高了常规编码任务的效率，包括代码生成、重构和调试。开发人员表示，当提供给LLMs清晰明确的问题陈述时，其效果最佳，表明LLMs在分解问题和明确需求方面表现出色。此外，早期采用者认为LLMs在个人和职业发展中提供了重大价值，有助于学习新的语言和概念。早期采用者，软件工程和LLMs工作的高手，指出了软件工程领域早期和持续存在的挑战，如生成内容的不准确性以及在将LLM输出集成到生产环境之前需要进行仔细的手动审查的必要性。本研究提供了对LLMs如何塑造软件开发格局的细致理解，包括其优点、局限性和持续影响。 

---
# Balcony: A Lightweight Approach to Dynamic Inference of Generative Language Models 

**Title (ZH)**: 阳台：生成语言模型动态推理的一种轻量级方法 

**Authors**: Benyamin Jamialahmadi, Parsa Kavehzadeh, Mehdi Rezagholizadeh, Parsa Farinneya, Hossein Rajabzadeh, Aref Jafari, Boxing Chen, Marzieh Tahaei  

**Link**: [PDF](https://arxiv.org/pdf/2503.05005)  

**Abstract**: Deploying large language models (LLMs) in real-world applications is often hindered by strict computational and latency constraints. While dynamic inference offers the flexibility to adjust model behavior based on varying resource budgets, existing methods are frequently limited by hardware inefficiencies or performance degradation. In this paper, we introduce Balcony, a simple yet highly effective framework for depth-based dynamic inference. By freezing the pretrained LLM and inserting additional transformer layers at selected exit points, Balcony maintains the full model's performance while enabling real-time adaptation to different computational budgets. These additional layers are trained using a straightforward self-distillation loss, aligning the sub-model outputs with those of the full model. This approach requires significantly fewer training tokens and tunable parameters, drastically reducing computational costs compared to prior methods. When applied to the LLaMA3-8B model, using only 0.2% of the original pretraining data, Balcony achieves minimal performance degradation while enabling significant speedups. Remarkably, we show that Balcony outperforms state-of-the-art methods such as Flextron and Layerskip as well as other leading compression techniques on multiple models and at various scales, across a variety of benchmarks. 

**Abstract (ZH)**: 部署大型语言模型（LLMs）在实际应用中往往受限于严格的计算和延迟约束。虽然动态推理可以根据不同的资源预算调整模型行为，但现有方法常常受到硬件效率低下或性能下降的限制。本文介绍了一种基于深度的简单而高效的动态推理框架——Balcony。通过冻结预训练的LLM并在选定的退出点插入额外的Transformer层，Balcony能够在保持完整模型性能的同时，实现对不同计算预算的即时适应。这些额外的层是通过一个简单的自蒸馏损失进行训练的，使得子模型输出与完整模型的输出相一致。这种方法所需的训练令牌和可调参数数量显著减少，相比以往方法大幅降低了计算成本。当应用于LLaMA3-8B模型时，仅使用原始预训练数据的0.2%，Balcony仍能实现性能微小下降的同时带来显著的加速。令人惊讶的是，我们证明了Balcony在多种模型和不同规模下，在多个基准测试中优于Flextron、Layerskip等最新方法以及其他领先的压缩技术。 

---
# Wanda++: Pruning Large Language Models via Regional Gradients 

**Title (ZH)**: Wanda++: 基于区域梯度裁剪大型语言模型 

**Authors**: Yifan Yang, Kai Zhen, Bhavana Ganesh, Aram Galstyan, Goeric Huybrechts, Markus Müller, Jonas M. Kübler, Rupak Vignesh Swaminathan, Athanasios Mouchtaris, Sravan Babu Bodapati, Nathan Susanj, Zheng Zhang, Jack FitzGerald, Abhishek Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2503.04992)  

**Abstract**: Large Language Models (LLMs) pruning seeks to remove unimportant weights for inference speedup with minimal performance impact. However, existing methods often suffer from performance loss without full-model sparsity-aware fine-tuning. This paper presents Wanda++, a novel pruning framework that outperforms the state-of-the-art methods by utilizing decoder-block-level \textbf{regional} gradients. Specifically, Wanda++ improves the pruning score with regional gradients for the first time and proposes an efficient regional optimization method to minimize pruning-induced output discrepancies between the dense and sparse decoder output. Notably, Wanda++ improves perplexity by up to 32\% over Wanda in the language modeling task and generalizes effectively to downstream tasks. Further experiments indicate our proposed method is orthogonal to sparsity-aware fine-tuning, where Wanda++ can be combined with LoRA fine-tuning to achieve a similar perplexity improvement as the Wanda method. The proposed method is lightweight, pruning a 7B LLaMA model in under 10 minutes on a single NVIDIA H100 GPU. 

**Abstract (ZH)**: 大规模语言模型（LLMs）剪枝旨在通过移除不重要权重来提高推理速度，同时最小化性能影响。然而，现有方法往往在未进行全模型稀疏意识微调的情况下会遭受性能下降。本文提出了一种新型剪枝框架Wanda++，该框架通过利用解码块级区域梯度超越了现有最佳方法。具体而言，Wanda++首次利用区域梯度来改进剪枝分数，并提出了一种高效的区域优化方法，以最小化剪枝引起的稠密和稀疏解码器输出之间的输出差异。值得注意的是，Wanda++在语言建模任务中将困惑度提高了高达32%，并且在下游任务上表现出了有效的泛化能力。进一步的实验表明，我们提出的方法与稀疏意识微调正交，Wanda++可以与LoRA微调结合使用，以达到与Wanda方法相似的困惑度改进效果。所提出的方法轻量级，在单块NVIDIA H100 GPU上对一个7B LLaMA模型进行剪枝不到10分钟。 

---
# LVLM-Compress-Bench: Benchmarking the Broader Impact of Large Vision-Language Model Compression 

**Title (ZH)**: LVLM-Compress-Bench: 评估大规模视觉-语言模型压缩的更广泛影响 

**Authors**: Souvik Kundu, Anahita Bhiwandiwalla, Sungduk Yu, Phillip Howard, Tiep Le, Sharath Nittur Sridhar, David Cobbley, Hao Kang, Vasudev Lal  

**Link**: [PDF](https://arxiv.org/pdf/2503.04982)  

**Abstract**: Despite recent efforts in understanding the compression impact on large language models (LLMs) in terms of their downstream task performance and trustworthiness on relatively simpler uni-modal benchmarks (for example, question answering, common sense reasoning), their detailed study on multi-modal Large Vision-Language Models (LVLMs) is yet to be unveiled. Towards mitigating this gap, we present LVLM-Compress-Bench, a framework to first thoroughly study the broad impact of compression on the generative performance of LVLMs with multi-modal input driven tasks. In specific, we consider two major classes of compression for autoregressive models, namely KV cache and weight compression, for the dynamically growing intermediate cache and static weights, respectively.
We use four LVLM variants of the popular LLaVA framework to present our analysis via integrating various state-of-the-art KV and weight compression methods including uniform, outlier-reduced, and group quantization for the KV cache and weights. With this framework we demonstrate on ten different multi-modal datasets with different capabilities including recognition, knowledge, language generation, spatial awareness, visual reasoning, hallucination and visual illusion identification, toxicity, stereotypes and bias. In specific, our framework demonstrates the compression impact on both general and ethically critical metrics leveraging a combination of real world and synthetic datasets to encompass diverse societal intersectional attributes. Extensive experimental evaluations yield diverse and intriguing observations on the behavior of LVLMs at different quantization budget of KV and weights, in both maintaining and losing performance as compared to the baseline model with FP16 data format.
Code will be open-sourced at this https URL. 

**Abstract (ZH)**: 尽管近年来在理解压缩对大型语言模型（LLMs）的影响方面取得了一定进展，特别是在其下游任务性能和相对简单的单模基准上的可信度方面（例如，问答、常识推理），但对于多模态大型视觉-语言模型（LVLMs）的详细研究仍然尚未展开。为弥补这一差距，我们提出了LVLM-Compress-Bench框架，旨在全面研究压缩对多模态输入驱动任务中LVLMs生成性能的影响。具体而言，我们考虑了两类自回归模型的压缩方法，分别是用于动态增长的中间缓存的KV缓存压缩和用于静态权重的权重压缩。

我们使用流行的LLaVA框架的四种LVLM变体，通过集成最新的KV和权重压缩方法（包括均匀量化、异常值减少量化和分组量化）来展示我们的分析。通过该框架，我们在包括识别、知识、语言生成、空间意识、视觉推理、幻觉和视觉错觉识别、毒性、刻板印象和偏见等多种能力的十个不同多模态数据集上进行了演示。具体而言，我们的框架利用结合现实世界和合成数据集来展示压缩对通用性和伦理关键性指标的影响，这些数据集涵盖了多样化的社会交叉属性。广泛的实验评估揭示了在不同量化预算下的KV和权重压缩对LVLMs行为的影响，在保持和损失性能方面与FP16数据格式的基线模型相比呈现出多样而有趣的观察结果。

代码将在此处开放源代码：this https URL。 

---
# A Consensus Privacy Metrics Framework for Synthetic Data 

**Title (ZH)**: 合成数据的共识隐私度量框架 

**Authors**: Lisa Pilgram, Fida K. Dankar, Jorg Drechsler, Mark Elliot, Josep Domingo-Ferrer, Paul Francis, Murat Kantarcioglu, Linglong Kong, Bradley Malin, Krishnamurty Muralidhar, Puja Myles, Fabian Prasser, Jean Louis Raisaro, Chao Yan, Khaled El Emam  

**Link**: [PDF](https://arxiv.org/pdf/2503.04980)  

**Abstract**: Synthetic data generation is one approach for sharing individual-level data. However, to meet legislative requirements, it is necessary to demonstrate that the individuals' privacy is adequately protected. There is no consolidated standard for measuring privacy in synthetic data. Through an expert panel and consensus process, we developed a framework for evaluating privacy in synthetic data. Our findings indicate that current similarity metrics fail to measure identity disclosure, and their use is discouraged. For differentially private synthetic data, a privacy budget other than close to zero was not considered interpretable. There was consensus on the importance of membership and attribute disclosure, both of which involve inferring personal information about an individual without necessarily revealing their identity. The resultant framework provides precise recommendations for metrics that address these types of disclosures effectively. Our findings further present specific opportunities for future research that can help with widespread adoption of synthetic data. 

**Abstract (ZH)**: 合成数据生成是共享个体级数据的一种方法。然而，为了满足立法要求，有必要证明个体的隐私得到了充分保护。目前没有统一的标准来衡量合成数据中的隐私。通过专家小组和共识过程，我们开发了一个评估合成数据中隐私的框架。我们的研究发现当前的相似性度量无法衡量身份泄露，其使用是不被推荐的。对于差分隐私的合成数据，除了接近零的隐私预算外，其他隐私预算被认为不具可解释性。一致认为成员身份和属性泄露的重要性，这两种情况都涉及推断个人的个人信息而不 necessarily 暴露其身份。所形成的框架提供了针对这些类型泄露的有效度量指标的具体建议。我们的研究结果还指出了未来研究的具体机会，这有助于合成数据的广泛应用。 

---
# Quantifying the Relevance of Youth Research Cited in the US Policy Documents 

**Title (ZH)**: 量化引用在美国政策文件中青年研究的相关性 

**Authors**: Miftahul Jannat Mokarrama, Hamed Alhoori  

**Link**: [PDF](https://arxiv.org/pdf/2503.04977)  

**Abstract**: In recent years, there has been a growing concern and emphasis on conducting research beyond academic or scientific research communities, benefiting society at large. A well-known approach to measuring the impact of research on society is enumerating its policy citation(s). Despite the importance of research in informing policy, there is no concrete evidence to suggest the research's relevance in cited policy documents. This is concerning because it may increase the possibility of evidence used in policy being manipulated by individual, social, or political biases that may lead to inappropriate, fragmented, or archaic research evidence in policy. Therefore, it is crucial to identify the degree of relevance between research articles and citing policy documents. In this paper, we examined the scale of contextual relevance of youth-focused research in the referenced US policy documents using natural language processing techniques, state-of-the-art pre-trained Large Language Models (LLMs), and statistical analysis. Our experiments and analysis concluded that youth-related research articles that get US policy citations are mostly relevant to the citing policy documents. 

**Abstract (ZH)**: 近年来，越来越关注并强调在学术或科学研究社区之外开展研究，以惠及全社会。衡量研究成果对社会影响的一种公认方法是统计其政策引用次数。尽管研究对政策制定具有重要影响，但目前没有确凿证据表明被引用的政策文件中所引用的研究的相关性。这令人担忧，因为它增加了政策中使用证据被个人、社会或政治偏见操纵的可能性，可能导致政策中缺乏适当的、整合的或现代的研究证据。因此，识别研究论文与引用政策文件之间的相关性程度至关重要。本文使用自然语言处理技术、最先进的预训练大规模语言模型（LLMs）和统计分析，考察了被引用的美国政策文件中 youths 领域研究的相关性规模。我们的实验和分析得出结论，获得美国政策引用的研究文章大多与引用的政策文件相关。 

---
# Beyond RAG: Task-Aware KV Cache Compression for Comprehensive Knowledge Reasoning 

**Title (ZH)**: 超越RAG：面向任务的键值缓存压缩以实现全面知识推理 

**Authors**: Giulio Corallo, Orion Weller, Fabio Petroni, Paolo Papotti  

**Link**: [PDF](https://arxiv.org/pdf/2503.04973)  

**Abstract**: Incorporating external knowledge in large language models (LLMs) enhances their utility across diverse applications, but existing methods have trade-offs. Retrieval-Augmented Generation (RAG) fetches evidence via similarity search, but key information may fall outside top ranked results. Long-context models can process multiple documents but are computationally expensive and limited by context window size. Inspired by students condensing study material for open-book exams, we propose task-aware key-value (KV) cache compression, which compresses external knowledge in a zero- or few-shot setup. This enables LLMs to reason efficiently over a compacted representation of all relevant information. Experiments show our approach outperforms both RAG and task-agnostic compression methods. On LongBench v2, it improves accuracy by up to 7 absolute points over RAG with a 30x compression rate, while reducing inference latency from 0.43s to 0.16s. A synthetic dataset highlights that RAG performs well when sparse evidence suffices, whereas task-aware compression is superior for broad knowledge tasks. 

**Abstract (ZH)**: 在大型语言模型中整合外部知识可以增强其在多种应用中的实用性，但现有方法存在权衡。受学生为开卷考试浓缩学习资料的启发，我们提出了一种任务感知的关键值缓存压缩方法，该方法在零样本或少样本设置中压缩外部知识，使大型语言模型能够高效地推理所有相关信息的紧凑表示。实验表明，我们的方法在性能上优于检索增强生成（RAG）和任务无关的压缩方法。在LongBench v2上，使用30倍压缩率时，准确率提高了7个绝对点，并将推理延迟从0.43秒减少到0.16秒。合成数据集表明，当证据稀疏时RAG表现良好，而任务感知压缩对于广泛知识任务更优。 

---
# Incentivizing Multi-Tenant Split Federated Learning for Foundation Models at the Network Edge 

**Title (ZH)**: 在网络边缘促进多租户分割联邦学习的基础模型激励机制 

**Authors**: Songyuan Li, Jia Hu, Geyong Min, Haojun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04971)  

**Abstract**: Foundation models (FMs) such as GPT-4 exhibit exceptional generative capabilities across diverse downstream tasks through fine-tuning. Split Federated Learning (SFL) facilitates privacy-preserving FM fine-tuning on resource-constrained local devices by offloading partial FM computations to edge servers, enabling device-edge synergistic fine-tuning. Practical edge networks often host multiple SFL tenants to support diversified downstream tasks. However, existing research primarily focuses on single-tenant SFL scenarios, and lacks tailored incentive mechanisms for multi-tenant settings, which are essential to effectively coordinate self-interested local devices for participation in various downstream tasks, ensuring that each SFL tenant's distinct FM fine-tuning requirements (e.g., FM types, performance targets, and fine-tuning deadlines) are met. To address this gap, we propose a novel Price-Incentive Mechanism (PRINCE) that guides multiple SFL tenants to offer strategic price incentives, which solicit high-quality device participation for efficient FM fine-tuning. Specifically, we first develop a bias-resilient global SFL model aggregation scheme to eliminate model biases caused by independent device participation. We then derive a rigorous SFL convergence bound to evaluate the contributions of heterogeneous devices to FM performance improvements, guiding the incentive strategies of SFL tenants. Furthermore, we model inter-tenant device competition as a congestion game for Stackelberg equilibrium (SE) analysis, deriving each SFL tenant's optimal incentive strategy. Extensive simulations involving four representative SFL tenant types (ViT, BERT, Whisper, and LLaMA) across diverse data modalities (text, images, and audio) demonstrate that PRINCE accelerates FM fine-tuning by up to 3.07x compared to state-of-the-art approaches, while consistently meeting fine-tuning performance targets. 

**Abstract (ZH)**: 基于定价激励机制的多租户Split联邦学习中基础模型微调加速方法 

---
# Data-Efficient Learning from Human Interventions for Mobile Robots 

**Title (ZH)**: 基于人类干预的数据高效学习在移动机器人中的应用 

**Authors**: Zhenghao Peng, Zhizheng Liu, Bolei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.04969)  

**Abstract**: Mobile robots are essential in applications such as autonomous delivery and hospitality services. Applying learning-based methods to address mobile robot tasks has gained popularity due to its robustness and generalizability. Traditional methods such as Imitation Learning (IL) and Reinforcement Learning (RL) offer adaptability but require large datasets, carefully crafted reward functions, and face sim-to-real gaps, making them challenging for efficient and safe real-world deployment. We propose an online human-in-the-loop learning method PVP4Real that combines IL and RL to address these issues. PVP4Real enables efficient real-time policy learning from online human intervention and demonstration, without reward or any pretraining, significantly improving data efficiency and training safety. We validate our method by training two different robots -- a legged quadruped, and a wheeled delivery robot -- in two mobile robot tasks, one of which even uses raw RGBD image as observation. The training finishes within 15 minutes. Our experiments show the promising future of human-in-the-loop learning in addressing the data efficiency issue in real-world robotic tasks. More information is available at: this https URL 

**Abstract (ZH)**: 移动机器人在自主配送和 Hospitality 服务等应用中至关重要。基于学习的方法在解决移动机器人任务时由于其鲁棒性和泛化能力而变得流行。传统方法如模仿学习（IL）和强化学习（RL）具有适应性，但需要大量数据集、精心设计的奖励函数，并且面临从仿真到现实的差距，使得它们在高效的现实世界部署中具有挑战性。我们提出了一种名为 PVP4Real 的在线人类在环学习方法，该方法结合了 IL 和 RL 来解决这些问题。PVP4Real 允许通过实时的人类干预和示范高效地学习策略，无需任何奖励或预训练，显著提高了数据效率和训练安全性。我们通过训练两个不同类型的机器人——一个腿足四足机器人和一个轮式配送机器人——在两种移动机器人任务中验证了该方法，其中一个任务甚至使用原始 RGBD 图像作为观测。训练时间仅需 15 分钟。我们的实验展示了人类在环学习在解决实际机器人任务中的数据效率问题方面的光明前景。更多信息请参见：this https URL 

---
# Prediction of Frozen Region Growth in Kidney Cryoablation Intervention Using a 3D Flow-Matching Model 

**Title (ZH)**: 使用3D流匹配模型预测肾脏冷冻消融治疗中冻结区域的增长 

**Authors**: Siyeop Yoon, Yujin Oh, Matthew Tivnan, Sifan Song, Pengfei Jin, Sekeun KimHyun Jin Cho, Dufan Wu, Raul Uppot, Quanzheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.04966)  

**Abstract**: This study presents a 3D flow-matching model designed to predict the progression of the frozen region (iceball) during kidney cryoablation. Precise intraoperative guidance is critical in cryoablation to ensure complete tumor eradication while preserving adjacent healthy tissue. However, conventional methods, typically based on physics driven or diffusion based simulations, are computationally demanding and often struggle to represent complex anatomical structures accurately. To address these limitations, our approach leverages intraoperative CT imaging to inform the model. The proposed 3D flow matching model is trained to learn a continuous deformation field that maps early-stage CT scans to future predictions. This transformation not only estimates the volumetric expansion of the iceball but also generates corresponding segmentation masks, effectively capturing spatial and morphological changes over time. Quantitative analysis highlights the model robustness, demonstrating strong agreement between predictions and ground-truth segmentations. The model achieves an Intersection over Union (IoU) score of 0.61 and a Dice coefficient of 0.75. By integrating real time CT imaging with advanced deep learning techniques, this approach has the potential to enhance intraoperative guidance in kidney cryoablation, improving procedural outcomes and advancing the field of minimally invasive surgery. 

**Abstract (ZH)**: 本研究提出一种3D流匹配模型，用于预测肾脏冷冻消融过程中冰球区域的 progression。在冷冻消融过程中，精确的术中引导对于确保完全消除肿瘤同时保留相邻健康组织至关重要。然而，传统的基于物理驱动或扩散驱动的模拟方法计算复杂，往往难以准确表示复杂的解剖结构。为克服这些局限性，我们的方法利用术中CT成像来指导模型。所提出的3D流匹配模型被训练以学习连续的变形场，将早期CT扫描映射到未来的预测。这一变换不仅能估计冰球体积的扩张，还能生成相应的分割掩膜，有效地捕捉随时间变化的空间和形态变化。定量分析突显了模型的稳健性，预测与真实分割之间表现出强烈的对应关系。该模型的交并比(IoU)得分为0.61，Dice系数为0.75。通过将实时CT成像与先进的深度学习技术相结合，此方法有望增强肾脏冷冻消融过程中的术中引导，提高手术结果，并推动微创手术领域的发展。 

---
# Energy-Latency Attacks: A New Adversarial Threat to Deep Learning 

**Title (ZH)**: 能量-延迟攻击：对深度学习的一种新型敌对威胁 

**Authors**: Hanene F. Z. Brachemi Meftah, Wassim Hamidouche, Sid Ahmed Fezza, Olivier Deforges  

**Link**: [PDF](https://arxiv.org/pdf/2503.04963)  

**Abstract**: The growing computational demand for deep neural networks ( DNNs) has raised concerns about their energy consumption and carbon footprint, particularly as the size and complexity of the models continue to increase. To address these challenges, energy-efficient hardware and custom accelerators have become essential. Additionally, adaptable DNN s are being developed to dynamically balance performance and efficiency. The use of these strategies became more common to enable sustainable AI deployment. However, these efficiency-focused designs may also introduce vulnerabilities, as attackers can potentially exploit them to increase latency and energy usage by triggering their worst-case-performance scenarios. This new type of attack, called energy-latency attacks, has recently gained significant research attention, focusing on the vulnerability of DNN s to this emerging attack paradigm, which can trigger denial-of-service ( DoS) attacks. This paper provides a comprehensive overview of current research on energy-latency attacks, categorizing them using the established taxonomy for traditional adversarial attacks. We explore different metrics used to measure the success of these attacks and provide an analysis and comparison of existing attack strategies. We also analyze existing defense mechanisms and highlight current challenges and potential areas for future research in this developing field. The GitHub page for this work can be accessed at this https URL 

**Abstract (ZH)**: 深度神经网络（DNNs）不断增长的计算需求引发了对其能源消耗和碳足迹的担忧，尤其是在模型的规模和复杂性持续增加的情况下。为应对这些挑战，能源高效的硬件和定制加速器变得至关重要。此外，正在开发可调整的DNNs以动态平衡性能和效率。这些策略的使用越来越普遍，以实现可持续的人工智能部署。然而，这些注重效率的设计也可能引入漏洞，攻击者可以通过触发其最坏情况性能场景来增加延迟和能源消耗。这种新型攻击被称为能量-延迟攻击，最近引起了广泛关注，重点关注DNNs对这种新兴攻击范式的易受攻击性，这可以引发服务拒绝攻击（DoS）。本文提供了一种关于能量-延迟攻击的全面综述，通过传统对抗攻击的分类学对它们进行分类。我们探讨了用于衡量这些攻击成功程度的不同指标，并对现有的攻击策略进行了分析和比较。我们还分析了现有的防御机制，并指出现有挑战以及未来研究的潜在领域。 

---
# SafeArena: Evaluating the Safety of Autonomous Web Agents 

**Title (ZH)**: SafeArena: 评估自主网络代理的安全性 

**Authors**: Ada Defne Tur, Nicholas Meade, Xing Han Lù, Alejandra Zambrano, Arkil Patel, Esin Durmus, Spandana Gella, Karolina Stańczak, Siva Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2503.04957)  

**Abstract**: LLM-based agents are becoming increasingly proficient at solving web-based tasks. With this capability comes a greater risk of misuse for malicious purposes, such as posting misinformation in an online forum or selling illicit substances on a website. To evaluate these risks, we propose SafeArena, the first benchmark to focus on the deliberate misuse of web agents. SafeArena comprises 250 safe and 250 harmful tasks across four websites. We classify the harmful tasks into five harm categories -- misinformation, illegal activity, harassment, cybercrime, and social bias, designed to assess realistic misuses of web agents. We evaluate leading LLM-based web agents, including GPT-4o, Claude-3.5 Sonnet, Qwen-2-VL 72B, and Llama-3.2 90B, on our benchmark. To systematically assess their susceptibility to harmful tasks, we introduce the Agent Risk Assessment framework that categorizes agent behavior across four risk levels. We find agents are surprisingly compliant with malicious requests, with GPT-4o and Qwen-2 completing 34.7% and 27.3% of harmful requests, respectively. Our findings highlight the urgent need for safety alignment procedures for web agents. Our benchmark is available here: this https URL 

**Abstract (ZH)**: 基于LLM的代理在解决网络任务方面的能力日益增强。随着这一能力的提升，代理被恶意利用的风险也在增加，比如在在线论坛发布虚假信息或在网上销售非法物品。为评估这些风险，我们提出了SafeArena，这是首个专注于网络代理恶意利用行为的标准基准。SafeArena包含四个网站上的250个安全任务和250个有害任务。我们将有害任务划分为五类危害类别——虚假信息、非法活动、骚扰、网络犯罪和社会偏见，旨在评估网络代理的现实滥用情况。我们对包括GPT-4o、Claude-3.5 Sonnet、Qwen-2-VL 72B和Llama-3.2 90B在内的领先基于LLM的网络代理进行评估。为了系统地评估其对有害任务的易感性，我们引入了代理风险评估框架，该框架将代理行为分为四个风险等级。我们发现，代理竟然相当合规地执行了恶意请求，GPT-4o和Qwen-2分别完成了34.7%和27.3%的有害请求。我们的研究结果突显了迫切需要对网络代理进行安全对齐程序。我们的基准数据可在此获取：this https URL 

---
# Federated Inverse Probability Treatment Weighting for Individual Treatment Effect Estimation 

**Title (ZH)**: 联邦逆概率治疗加权估计个体治疗效果 

**Authors**: Changchang Yin, Hong-You Chen, Wei-Lun Chao, Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04946)  

**Abstract**: Individual treatment effect (ITE) estimation is to evaluate the causal effects of treatment strategies on some important outcomes, which is a crucial problem in healthcare. Most existing ITE estimation methods are designed for centralized settings. However, in real-world clinical scenarios, the raw data are usually not shareable among hospitals due to the potential privacy and security risks, which makes the methods not applicable. In this work, we study the ITE estimation task in a federated setting, which allows us to harness the decentralized data from multiple hospitals. Due to the unavoidable confounding bias in the collected data, a model directly learned from it would be inaccurate. One well-known solution is Inverse Probability Treatment Weighting (IPTW), which uses the conditional probability of treatment given the covariates to re-weight each training example. Applying IPTW in a federated setting, however, is non-trivial. We found that even with a well-estimated conditional probability, the local model training step using each hospital's data alone would still suffer from confounding bias. To address this, we propose FED-IPTW, a novel algorithm to extend IPTW into a federated setting that enforces both global (over all the data) and local (within each hospital) decorrelation between covariates and treatments. We validated our approach on the task of comparing the treatment effects of mechanical ventilation on improving survival probability for patients with breadth difficulties in the intensive care unit (ICU). We conducted experiments on both synthetic and real-world eICU datasets and the results show that FED-IPTW outperform state-of-the-art methods on all the metrics on factual prediction and ITE estimation tasks, paving the way for personalized treatment strategy design in mechanical ventilation usage. 

**Abstract (ZH)**: 个体治疗效果（ITE）估计在 federated 设置中的研究 

---
# Collaborative Evaluation of Deepfake Text with Deliberation-Enhancing Dialogue Systems 

**Title (ZH)**: 增强反思对话系统的深度假信息文本协作评估 

**Authors**: Jooyoung Lee, Xiaochen Zhu, Georgi Karadzhov, Tom Stafford, Andreas Vlachos, Dongwon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.04945)  

**Abstract**: The proliferation of generative models has presented significant challenges in distinguishing authentic human-authored content from deepfake content. Collaborative human efforts, augmented by AI tools, present a promising solution. In this study, we explore the potential of DeepFakeDeLiBot, a deliberation-enhancing chatbot, to support groups in detecting deepfake text. Our findings reveal that group-based problem-solving significantly improves the accuracy of identifying machine-generated paragraphs compared to individual efforts. While engagement with DeepFakeDeLiBot does not yield substantial performance gains overall, it enhances group dynamics by fostering greater participant engagement, consensus building, and the frequency and diversity of reasoning-based utterances. Additionally, participants with higher perceived effectiveness of group collaboration exhibited performance benefits from DeepFakeDeLiBot. These findings underscore the potential of deliberative chatbots in fostering interactive and productive group dynamics while ensuring accuracy in collaborative deepfake text detection. \textit{Dataset and source code used in this study will be made publicly available upon acceptance of the manuscript. 

**Abstract (ZH)**: 生成模型的泛滥为区分真实人类创作的内容与深度伪造内容带来了重大挑战。借助AI工具的人机协作提供了有希望的解决方案。本研究探讨了增强辩论的聊天机器人DeepFakeDeLiBot在支持团队检测深度伪造文本方面的潜在作用。研究发现，基于团队的问题解决显著提高了识别机器生成段落的准确性，而个人努力则不如团队努力。虽然与DeepFakeDeLiBot的互动在总体上并未带来显著的性能提升，但它通过促进更高的参与者参与度、共识构建以及基于推理的说法频次和多样性，增强了团队动态。此外，感知到团队合作更有效的参与者从DeepFakeDeLiBot中获得了性能上的益处。这些发现强调了辩论型聊天机器人在促进互动和高效团队动态同时确保协作深度伪造文本检测准确性方面的潜力。本研究的数据集和源代码将在稿件被接受后公开。 

---
# VQEL: Enabling Self-Developed Symbolic Language in Agents through Vector Quantization in Emergent Language Games 

**Title (ZH)**: VQEL：通过Emergent语言游戏中的向量量化实现自主开发符号语言的能力 

**Authors**: Mohammad Mahdi Samiei Paqaleh, Mahdieh Soleymani Baghshah  

**Link**: [PDF](https://arxiv.org/pdf/2503.04940)  

**Abstract**: In the field of emergent language, efforts have traditionally focused on developing communication protocols through interactions between agents in referential games. However, the aspect of internal language learning, where language serves not only as a communicative tool with others but also as a means for individual thinking, self-reflection, and problem-solving remains underexplored. Developing a language through self-play, without another agent's involvement, poses a unique challenge. It requires an agent to craft symbolic representations and train them using direct gradient methods. The challenge here is that if an agent attempts to learn symbolic representations through self-play using conventional modeling and techniques such as REINFORCE, the solution will offer no advantage over previous multi-agent approaches. We introduce VQEL, a novel method that incorporates Vector Quantization into the agents' architecture, enabling them to autonomously invent and develop discrete symbolic representations in a self-play referential game. Following the self-play phase, agents can enhance their language through reinforcement learning and interactions with other agents in the mutual-play phase. Our experiments across various datasets demonstrate that VQEL not only outperforms the traditional REINFORCE method but also benefits from improved control and reduced susceptibility to collapse, thanks to the incorporation of vector quantization. 

**Abstract (ZH)**: 在新兴语言领域，努力传统上集中在通过代理在指称游戏中相互作用来开发通信协议。然而，语言的内部学习方面仍然未被充分探索，即语言不仅作为与他人沟通的工具，也作为个体思考、自我反思和解决问题的手段。通过自我游戏发展语言而不涉及另一个代理带来了独特挑战。这要求代理构建象征性表示并使用直接梯度方法进行训练。问题是，如果代理试图通过自我游戏使用传统建模和技术（如REINFORCE）学习象征性表示，那么该解决方案将无法提供优于多代理方法的优势。我们提出了VQEL，这是一种新颖的方法，将向量量化融入代理的架构中，使代理能够自主发明和发展离散的象征性表示，在自我游戏指称游戏中。在自我游戏阶段之后，代理可以通过强化学习和在互游戏阶段与其他代理的交互来提升其语言能力。我们的跨多个数据集的实验表明，VQEL不仅优于传统的REINFORCE方法，而且由于引入了向量量化，还受益于更好的控制和较低的塌陷倾向。 

---
# Learning-based GNSS Uncertainty Quantification using Continuous-Time Factor Graph Optimization 

**Title (ZH)**: 基于学习的连续时间因子图优化下的GNSS不确定性量化 

**Authors**: Haoming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04933)  

**Abstract**: This short paper presents research findings on two learning-based methods for quantifying measurement uncertainties in global navigation satellite systems (GNSS). We investigate two learning strategies: offline learning for outlier prediction and online learning for noise distribution approximation, specifically applied to GNSS pseudorange observations. To develop and evaluate these learning methods, we introduce a novel multisensor state estimator that accurately and robustly estimates trajectory from multiple sensor inputs, critical for deriving GNSS measurement residuals used to train the uncertainty models. We validate the proposed learning-based models using real-world sensor data collected in diverse urban environments. Experimental results demonstrate that both models effectively handle GNSS outliers and improve state estimation performance. Furthermore, we provide insightful discussions to motivate future research toward developing a federated framework for robust vehicle localization in challenging environments. 

**Abstract (ZH)**: 基于学习的全球导航卫星系统测距误差量化方法研究 

---
# Curiosity-Driven Imagination: Discovering Plan Operators and Learning Associated Policies for Open-World Adaptation 

**Title (ZH)**: 好奇心驱动的想象：发现开放世界适应中的计划操作并学习相关策略 

**Authors**: Pierrick Lorang, Hong Lu, Matthias Scheutz  

**Link**: [PDF](https://arxiv.org/pdf/2503.04931)  

**Abstract**: Adapting quickly to dynamic, uncertain environments-often called "open worlds"-remains a major challenge in robotics. Traditional Task and Motion Planning (TAMP) approaches struggle to cope with unforeseen changes, are data-inefficient when adapting, and do not leverage world models during learning. We address this issue with a hybrid planning and learning system that integrates two models: a low level neural network based model that learns stochastic transitions and drives exploration via an Intrinsic Curiosity Module (ICM), and a high level symbolic planning model that captures abstract transitions using operators, enabling the agent to plan in an "imaginary" space and generate reward machines. Our evaluation in a robotic manipulation domain with sequential novelty injections demonstrates that our approach converges faster and outperforms state-of-the-art hybrid methods. 

**Abstract (ZH)**: 快速适应动态和不确定的环境——通常称为“开放世界”——仍然是机器人技术中的一个重大挑战。传统的任务与运动规划（TAMP）方法难以应对意外变化，适应过程数据效率低，并且在学习过程中不利用世界模型。我们通过结合一个低级别的基于神经网络的模型和一个高级别的符号规划模型来解决这个问题：低级别的模型学习随机转移并借助内在好奇心模块（ICM）驱动探索，高级别的模型使用操作符捕捉抽象转移，使代理能够在“想象空间”中进行规划并生成奖励机器。在具有序列新颖性注入的机器人操作域中的评估表明，我们的方法更快地收敛并且优于最先进的混合方法。 

---
# HILGEN: Hierarchically-Informed Data Generation for Biomedical NER Using Knowledgebases and Large Language Models 

**Title (ZH)**: HILGEN：基于层级知识的数据生成方法在生物医学NER中的应用，结合知识库和大型语言模型 

**Authors**: Yao Ge, Yuting Guo, Sudeshna Das, Swati Rajwal, Selen Bozkurt, Abeed Sarker  

**Link**: [PDF](https://arxiv.org/pdf/2503.04930)  

**Abstract**: We present HILGEN, a Hierarchically-Informed Data Generation approach that combines domain knowledge from the Unified Medical Language System (UMLS) with synthetic data generated by large language models (LLMs), specifically GPT-3.5. Our approach leverages UMLS's hierarchical structure to expand training data with related concepts, while incorporating contextual information from LLMs through targeted prompts aimed at automatically generating synthetic examples for sparsely occurring named entities. The performance of the HILGEN approach was evaluated across four biomedical NER datasets (MIMIC III, BC5CDR, NCBI-Disease, and Med-Mentions) using BERT-Large and DANN (Data Augmentation with Nearest Neighbor Classifier) models, applying various data generation strategies, including UMLS, GPT-3.5, and their best ensemble. For the BERT-Large model, incorporating UMLS led to an average F1 score improvement of 40.36%, while using GPT-3.5 resulted in a comparable average increase of 40.52%. The Best-Ensemble approach using BERT-Large achieved the highest improvement, with an average increase of 42.29%. DANN model's F1 score improved by 22.74% on average using the UMLS-only approach. The GPT-3.5-based method resulted in a 21.53% increase, and the Best-Ensemble DANN model showed a more notable improvement, with an average increase of 25.03%. Our proposed HILGEN approach improves NER performance in few-shot settings without requiring additional manually annotated data. Our experiments demonstrate that an effective strategy for optimizing biomedical NER is to combine biomedical knowledge curated in the past, such as the UMLS, and generative LLMs to create synthetic training instances. Our future research will focus on exploring additional innovative synthetic data generation strategies for further improving NER performance. 

**Abstract (ZH)**: Hierarchically-Informed Data Generation Approach Combining UMLS with Large Language Models for Biomedical NER 

---
# Adapt3R: Adaptive 3D Scene Representation for Domain Transfer in Imitation Learning 

**Title (ZH)**: Adapt3R: 适应性三维场景表示在imitation learning中的领域迁移 

**Authors**: Albert Wilcox, Mohamed Ghanem, Masoud Moghani, Pierre Barroso, Benjamin Joffe, Animesh Garg  

**Link**: [PDF](https://arxiv.org/pdf/2503.04877)  

**Abstract**: Imitation Learning (IL) has been very effective in training robots to perform complex and diverse manipulation tasks. However, its performance declines precipitously when the observations are out of the training distribution. 3D scene representations that incorporate observations from calibrated RGBD cameras have been proposed as a way to improve generalizability of IL policies, but our evaluations in cross-embodiment and novel camera pose settings found that they show only modest improvement. To address those challenges, we propose Adaptive 3D Scene Representation (Adapt3R), a general-purpose 3D observation encoder which uses a novel architecture to synthesize data from one or more RGBD cameras into a single vector that can then be used as conditioning for arbitrary IL algorithms. The key idea is to use a pretrained 2D backbone to extract semantic information about the scene, using 3D only as a medium for localizing this semantic information with respect to the end-effector. We show that when trained end-to-end with several SOTA multi-task IL algorithms, Adapt3R maintains these algorithms' multi-task learning capacity while enabling zero-shot transfer to novel embodiments and camera poses. Furthermore, we provide a detailed suite of ablation and sensitivity experiments to elucidate the design space for point cloud observation encoders. 

**Abstract (ZH)**: 自适应三维场景表示（Adapt3R）：一种通用的三维观测编码器 

---
# Are Large Language Models Good In-context Learners for Financial Sentiment Analysis? 

**Title (ZH)**: 大型语言模型在金融情感分析中的即插即用学习能力良好吗？ 

**Authors**: Xinyu Wei, Luojia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04873)  

**Abstract**: Recently, large language models (LLMs) with hundreds of billions of parameters have demonstrated the emergent ability, surpassing traditional methods in various domains even without fine-tuning over domain-specific data. However, when it comes to financial sentiment analysis (FSA)$\unicode{x2013}$a fundamental task in financial AI$\unicode{x2013}$these models often encounter various challenges, such as complex financial terminology, subjective human emotions, and ambiguous inclination expressions. In this paper, we aim to answer the fundamental question: whether LLMs are good in-context learners for FSA? Unveiling this question can yield informative insights on whether LLMs can learn to address the challenges by generalizing in-context demonstrations of financial document-sentiment pairs to the sentiment analysis of new documents, given that finetuning these models on finance-specific data is difficult, if not impossible at all. To the best of our knowledge, this is the first paper exploring in-context learning for FSA that covers most modern LLMs (recently released DeepSeek V3 included) and multiple in-context sample selection methods. Comprehensive experiments validate the in-context learning capability of LLMs for FSA. 

**Abstract (ZH)**: 近期，具有数百亿参数的大型语言模型（LLMs）在各种领域展现了超越传统方法的能力，即使在没有针对特定领域进行微调的情况下也是如此。然而，在金融情感分析（FSA）这一金融AI的基本任务中，这些模型常常会遇到诸如复杂的金融术语、主观的人类情感以及模糊的倾向性表达等挑战。本文旨在回答一个基本问题：LLMs 是否适合用于FSA的上下文学习？揭开这个问题的答案可以揭示LLMs是否能够通过泛化金融文档-情感对的上下文示例来解决FSA的新文档情感分析问题，特别是在难以甚至不可能针对金融特定数据进行模型微调的情况下。据我们所知，这是第一篇探讨涵盖大多数现代LLM（包括最近发布的DeepSeek V3）和多种上下文样本选择方法的FSA上下文学习的论文。全面的实验验证了LLMs在FSA上的上下文学习能力。 

---
# TinyR1-32B-Preview: Boosting Accuracy with Branch-Merge Distillation 

**Title (ZH)**: TinyR1-32B-预览：基于分支合并蒸馏提升准确性 

**Authors**: Lin Sun, Guangxiang Zhao, Xiaoqi Jian, Yuhan Wu, Weihong Lin, Yongfu Zhu, Change Jia, Linglin Zhang, Jinzhu Wu, Junfeng Ran, Sai-er Hu, Zihan Jiang, Junting Zhou, Wenrui Liu, Bin Cui, Tong Yang, Xiangzheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04872)  

**Abstract**: The challenge of reducing the size of Large Language Models (LLMs) while maintaining their performance has gained significant attention. However, existing methods, such as model distillation and transfer learning, often fail to achieve high accuracy. To address this limitation, we introduce the Branch-Merge distillation approach, which enhances model compression through two phases: (1) the Branch Phase, where knowledge from a large teacher model is \textit{selectively distilled} into specialized student models via domain-specific supervised fine-tuning (SFT); And (2) the Merge Phase, where these student models are merged to enable cross-domain knowledge transfer and improve generalization. We validate our distillation approach using DeepSeek-R1 as the teacher and DeepSeek-R1-Distill-Qwen-32B as the student. The resulting merged model, TinyR1-32B-Preview, outperforms its counterpart DeepSeek-R1-Distill-Qwen-32B across multiple benchmarks, including Mathematics (+5.5 points), Coding (+4.4 points) and Science (+2.9 points), while achieving near-equal performance to DeepSeek-R1 on AIME 2024. The Branch-Merge distillation approach provides a scalable solution for creating smaller, high-performing LLMs with reduced computational cost and time. 

**Abstract (ZH)**: 减小大型语言模型规模以维持其性能的支分合并蒸馏方法面临着重大挑战 

---
# Label Distribution Learning-Enhanced Dual-KNN for Text Classification 

**Title (ZH)**: 标签分布学习增强的双KNN文本分类 

**Authors**: Bo Yuan, Yulin Chen, Zhen Tan, Wang Jinyan, Huan Liu, Yin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04869)  

**Abstract**: Many text classification methods usually introduce external information (e.g., label descriptions and knowledge bases) to improve the classification performance. Compared to external information, some internal information generated by the model itself during training, like text embeddings and predicted label probability distributions, are exploited poorly when predicting the outcomes of some texts. In this paper, we focus on leveraging this internal information, proposing a dual $k$ nearest neighbor (D$k$NN) framework with two $k$NN modules, to retrieve several neighbors from the training set and augment the distribution of labels. For the $k$NN module, it is easily confused and may cause incorrect predictions when retrieving some nearest neighbors from noisy datasets (datasets with labeling errors) or similar datasets (datasets with similar labels). To address this issue, we also introduce a label distribution learning module that can learn label similarity, and generate a better label distribution to help models distinguish texts more effectively. This module eases model overfitting and improves final classification performance, hence enhancing the quality of the retrieved neighbors by $k$NN modules during inference. Extensive experiments on the benchmark datasets verify the effectiveness of our method. 

**Abstract (ZH)**: 许多文本分类方法通常引入外部信息（如标签描述和知识库）以提高分类性能。与外部信息相比，模型在训练过程中生成的一些内部信息（如文本嵌入和预测的标签概率分布）在预测某些文本的结果时利用不足。本文专注于利用这些内部信息，提出了一种双$K$最近邻（D$k$NN）框架，该框架包含两个$K$NN模块，从训练集检索几个邻居并增强标签分布。对于$K$NN模块，从嘈杂数据集（带有标签错误的数据集）或类似数据集（带有相似标签的数据集）中检索最近邻居时容易出错并可能导致错误预测。为解决这一问题，我们还引入了一个标签分布学习模块，它可以学习标签相似性，生成更好的标签分布，从而帮助模型更有效地区分文本。该模块减轻了模型过拟合，提高了最终分类性能，从而在推理过程中提高了$K$NN模块检索到的邻居的质量。基准数据集上的大量实验验证了我们方法的有效性。 

---
# Privacy in Responsible AI: Approaches to Facial Recognition from Cloud Providers 

**Title (ZH)**: 负责任人工智能中的隐私保护：云提供商的面部识别方法 

**Authors**: Anna Elivanova  

**Link**: [PDF](https://arxiv.org/pdf/2503.04866)  

**Abstract**: As the use of facial recognition technology is expanding in different domains, ensuring its responsible use is gaining more importance. This paper conducts a comprehensive literature review of existing studies on facial recognition technology from the perspective of privacy, which is one of the key Responsible AI principles.
Cloud providers, such as Microsoft, AWS, and Google, are at the forefront of delivering facial-related technology services, but their approaches to responsible use of these technologies vary significantly. This paper compares how these cloud giants implement the privacy principle into their facial recognition and detection services. By analysing their approaches, it identifies both common practices and notable differences. The results of this research will be valuable for developers and businesses by providing them insights into best practices of three major companies for integration responsible AI, particularly privacy, into their cloud-based facial recognition technologies. 

**Abstract (ZH)**: 随着面部识别技术在不同领域的应用扩展，确保其负责任地使用变得愈发重要。本文从隐私保护的角度对现有的面部识别技术研究进行综述，这是一项负责任人工智能原则中的重要方面。云计算提供商，如微软、AWS和Google，是提供相关技术服務的前沿，但它们在这些技术负责任使用方面的做法存在显著差异。本文比较了这些云计算巨头如何将隐私原则融入其面部识别和检测服务中，并通过分析其做法，识别出共同实践和显著差异。研究结果将为开发者和企业提供了宝贵的见解，特别是在将其云基面部识别技术与负责任的人工智能，特别是隐私保护相结合的最佳实践方面。 

---
# E4: Energy-Efficient DNN Inference for Edge Video Analytics Via Early-Exit and DVFS 

**Title (ZH)**: 基于早期退出和动态电压频率调整的边缘视频分析中的能效DNN推理 

**Authors**: Ziyang Zhang, Yang Zhao, Ming-Ching Chang, Changyao Lin, Jie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04865)  

**Abstract**: Deep neural network (DNN) models are increasingly popular in edge video analytic applications. However, the compute-intensive nature of DNN models pose challenges for energy-efficient inference on resource-constrained edge devices. Most existing solutions focus on optimizing DNN inference latency and accuracy, often overlooking energy efficiency. They also fail to account for the varying complexity of video frames, leading to sub-optimal performance in edge video analytics. In this paper, we propose an Energy-Efficient Early-Exit (E4) framework that enhances DNN inference efficiency for edge video analytics by integrating a novel early-exit mechanism with dynamic voltage and frequency scaling (DVFS) governors. It employs an attention-based cascade module to analyze video frame diversity and automatically determine optimal DNN exit points. Additionally, E4 features a just-in-time (JIT) profiler that uses coordinate descent search to co-optimize CPU and GPU clock frequencies for each layer before the DNN exit points. Extensive evaluations demonstrate that E4 outperforms current state-of-the-art methods, achieving up to 2.8x speedup and 26% average energy saving while maintaining high accuracy. 

**Abstract (ZH)**: 一种用于边缘视频分析的节能早期退出框架：结合基于注意机制的级联模块与动态电压和频率调整的节能早退框架 

---
# Manboformer: Learning Gaussian Representations via Spatial-temporal Attention Mechanism 

**Title (ZH)**: Manboformer：通过空间-时间注意力机制学习高斯表示 

**Authors**: Ziyue Zhao, Qining Qi, Jianfa Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.04863)  

**Abstract**: Compared with voxel-based grid prediction, in the field of 3D semantic occupation prediction for autonomous driving, GaussianFormer proposed using 3D Gaussian to describe scenes with sparse 3D semantic Gaussian based on objects is another scheme with lower memory requirements. Each 3D Gaussian function represents a flexible region of interest and its semantic features, which are iteratively refined by the attention mechanism. In the experiment, it is found that the Gaussian function required by this method is larger than the query resolution of the original dense grid network, resulting in impaired performance. Therefore, we consider optimizing GaussianFormer by using unused temporal information. We learn the Spatial-Temporal Self-attention Mechanism from the previous grid-given occupation network and improve it to GaussianFormer. The experiment was conducted with the NuScenes dataset, and the experiment is currently underway. 

**Abstract (ZH)**: 基于高斯函数的3D语义占用预测：GaussianFormer在自动驾驶领域的另一种低内存要求方案及其时空自注意力机制优化 

---
# Codebook Reduction and Saturation: Novel observations on Inductive Thematic Saturation for Large Language Models and initial coding in Thematic Analysis 

**Title (ZH)**: 代码本缩减与饱和：大型语言模型归纳主题饱和的新观察及其在主题分析初步编码中的应用 

**Authors**: Stefano De Paoli, Walter Stan Mathis  

**Link**: [PDF](https://arxiv.org/pdf/2503.04859)  

**Abstract**: This paper reflects on the process of performing Thematic Analysis with Large Language Models (LLMs). Specifically, the paper deals with the problem of analytical saturation of initial codes, as produced by LLMs. Thematic Analysis is a well-established qualitative analysis method composed of interlinked phases. A key phase is the initial coding, where the analysts assign labels to discrete components of a dataset. Saturation is a way to measure the validity of a qualitative analysis and relates to the recurrence and repetition of initial codes. In the paper we reflect on how well LLMs achieve analytical saturation and propose also a novel technique to measure Inductive Thematic Saturation (ITS). This novel technique leverages a programming framework called DSPy. The proposed novel approach allows a precise measurement of ITS. 

**Abstract (ZH)**: 本研究反思了使用大规模语言模型（LLMs）进行主题分析的过程，特别关注初始编码的分析饱和问题。主题分析是一种成熟的定性分析方法，由多个相互关联的阶段组成。初始编码是一个关键阶段，分析师为数据集中的离散组件分配标签。饱和度是一种衡量定性分析有效性的方法，与初始编码的重复和重现有关。本研究反思了LLMs实现分析饱和的效果，并提出了一种新的技术来衡量归纳主题饱和（ITS）。该新技术利用了名为DSPy的编程框架，提出的新型方法能够精确测量ITS。 

---
# SHAPE : Self-Improved Visual Preference Alignment by Iteratively Generating Holistic Winner 

**Title (ZH)**: SHAPE : 自我改进的视觉偏好对齐通过迭代生成全局胜者 

**Authors**: Kejia Chen, Jiawen Zhang, Jiacong Hu, Jiazhen Yang, Jian Lou, Zunlei Feng, Mingli Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.04858)  

**Abstract**: Large Visual Language Models (LVLMs) increasingly rely on preference alignment to ensure reliability, which steers the model behavior via preference fine-tuning on preference data structured as ``image - winner text - loser text'' triplets. However, existing approaches often suffer from limited diversity and high costs associated with human-annotated preference data, hindering LVLMs from fully achieving their intended alignment capabilities. We present \projectname, a self-supervised framework capable of transforming the already abundant supervised text-image pairs into holistic preference triplets for more effective and cheaper LVLM alignment, eliminating the need for human preference annotations. Our approach facilitates LVLMs in progressively enhancing alignment capabilities through iterative self-improvement. The key design rationale is to devise preference triplets where the winner text consistently improves in holisticness and outperforms the loser response in quality, thereby pushing the model to ``strive to the utmost'' of alignment performance through preference fine-tuning. For each given text-image pair, SHAPE introduces multiple visual augmentations and pairs them with a summarized text to serve as the winner response, while designating the original text as the loser response. Experiments across \textbf{12} benchmarks on various model architectures and sizes, including LLaVA and DeepSeek-VL, show that SHAPE achieves significant gains, for example, achieving +11.3\% on MMVet (comprehensive evaluation), +1.4\% on MMBench (general VQA), and +8.0\% on POPE (hallucination robustness) over baselines in 7B models. Notably, qualitative analyses confirm enhanced attention to visual details and better alignment with human preferences for holistic descriptions. 

**Abstract (ZH)**: Large Visual Language Models (LVLMs) 越来越多地依赖偏好对齐以确保可靠性，这通过偏好微调引导模型行为，偏好数据结构化为“图像-获胜文本-失败文本”三元组。然而，现有方法通常受到有限多样性和与人类标注偏好数据相关的高成本的限制，阻碍了 LVLMs 完全实现其预期的对齐能力。我们提出 \projectname，一个自监督框架，能够将已经丰富的监督文本-图像配对转换为整体偏好三元组，以实现更有效的且成本更低的 LVLM 对齐，从而消除对人类偏好标注的需求。我们的方法通过逐步自我改进帮助 LVLMs 提高对齐能力。核心设计原则是设计偏好三元组，在这些三元组中，获胜文本在整体性上持续改进，并在质量上优于失败响应，从而通过偏好微调促使模型“尽最大努力”提高对齐性能。对于每个给定的文本-图像配对，SHAPE 引入多种视觉增强并将其与总结的文本配对，作为获胜响应，而将原始文本指定为失败响应。在包括 LLaVA 和 DeepSeek-VL 在内的各种模型架构和大小（共 12 个基准测试）上的实验表明，SHAPE 在 7B 模型中实现了显著收益，例如，在 MMVet（综合评估）中提高了 11.3%，在 MMBench（通用 VQA）中提高了 1.4%，在 POPE（幻觉稳健性）中提高了 8.0%。值得注意的是，定性分析证实了对视觉细节的更多关注以及整体描述与人类偏好的更好对齐。 

---
# One-Shot is Enough: Consolidating Multi-Turn Attacks into Efficient Single-Turn Prompts for LLMs 

**Title (ZH)**: 一击即中：将多轮攻击 Consolidate 为高效单轮提示以应用于大语言模型 

**Authors**: Junwoo Ha, Hyunjun Kim, Sangyoon Yu, Haon Park, Ashkan Yousefpour, Yuna Park, Suhyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.04856)  

**Abstract**: Despite extensive safety enhancements in large language models (LLMs), multi-turn "jailbreak" conversations crafted by skilled human adversaries can still breach even the most sophisticated guardrails. However, these multi-turn attacks demand considerable manual effort, limiting their scalability. In this work, we introduce a novel approach called Multi-turn-to-Single-turn (M2S) that systematically converts multi-turn jailbreak prompts into single-turn attacks. Specifically, we propose three conversion strategies - Hyphenize, Numberize, and Pythonize - each preserving sequential context yet packaging it in a single query. Our experiments on the Multi-turn Human Jailbreak (MHJ) dataset show that M2S often increases or maintains high Attack Success Rates (ASRs) compared to original multi-turn conversations. Notably, using a StrongREJECT-based evaluation of harmfulness, M2S achieves up to 95.9% ASR on Mistral-7B and outperforms original multi-turn prompts by as much as 17.5% in absolute improvement on GPT-4o. Further analysis reveals that certain adversarial tactics, when consolidated into a single prompt, exploit structural formatting cues to evade standard policy checks. These findings underscore that single-turn attacks - despite being simpler and cheaper to conduct - can be just as potent, if not more, than their multi-turn counterparts. Our findings underscore the urgent need to reevaluate and reinforce LLM safety strategies, given how adversarial queries can be compacted into a single prompt while still retaining sufficient complexity to bypass existing safety measures. 

**Abstract (ZH)**: 尽管在大型语言模型（LLMs）中进行了广泛的安全增强，但由熟练的人类对手设计的多轮“监狱突破”对话仍然可以突破最复杂的防护措施。然而，这些多轮攻击需要大量的手动努力，限制了其可扩展性。在本工作中，我们介绍了一种名为多轮到单轮（M2S）的新方法，该方法系统地将多轮监狱突破提示转换为单轮攻击。具体而言，我们提出了三种转换策略——减号化、编号化和Python化，每种策略保留了顺序上下文，但将其打包成单一查询。我们在多轮人类监狱突破（MHJ）数据集上的实验表明，M2S通常能够提高或保持较高的攻击成功率（ASRs），与原始的多轮对话相比。值得注意的是，通过对有害性的StrongREJECT评估，M2S在Mistral-7B上的ASR达到了95.9%，而在GPT-4o上绝对提高了17.5%的性能。进一步的分析表明，某些对抗性策略，在合并为单一提示后，利用结构格式化提示来规避标准策略检查。这些发现表明，虽然单一轮次攻击比其多轮对手更为简单和经济，但它们仍然同样具有强大甚至更强大的威力。我们的发现强调了重新评估和加强LLM安全策略的紧迫性，特别是在对抗性查询可以压缩成单一提示但仍保留足够复杂性以规避现有安全措施的情况下。 

---
# From Pixels to Trajectory: Universal Adversarial Example Detection via Temporal Imprints 

**Title (ZH)**: 从像素到轨迹：通过时间印记进行通用对抗性示例检测 

**Authors**: Yansong Gao, Huaibing Peng, Hua Ma, Zhiyang Dai, Shuo Wang, Hongsheng Hu, Anmin Fu, Minhui Xue  

**Link**: [PDF](https://arxiv.org/pdf/2503.04853)  

**Abstract**: For the first time, we unveil discernible temporal (or historical) trajectory imprints resulting from adversarial example (AE) attacks. Standing in contrast to existing studies all focusing on spatial (or static) imprints within the targeted underlying victim models, we present a fresh temporal paradigm for understanding these attacks. Of paramount discovery is that these imprints are encapsulated within a single loss metric, spanning universally across diverse tasks such as classification and regression, and modalities including image, text, and audio. Recognizing the distinct nature of loss between adversarial and clean examples, we exploit this temporal imprint for AE detection by proposing TRAIT (TRaceable Adversarial temporal trajectory ImprinTs). TRAIT operates under minimal assumptions without prior knowledge of attacks, thereby framing the detection challenge as a one-class classification problem. However, detecting AEs is still challenged by significant overlaps between the constructed synthetic losses of adversarial and clean examples due to the absence of ground truth for incoming inputs. TRAIT addresses this challenge by converting the synthetic loss into a spectrum signature, using the technique of Fast Fourier Transform to highlight the discrepancies, drawing inspiration from the temporal nature of the imprints, analogous to time-series signals. Across 12 AE attacks including SMACK (USENIX Sec'2023), TRAIT demonstrates consistent outstanding performance across comprehensively evaluated modalities, tasks, datasets, and model architectures. In all scenarios, TRAIT achieves an AE detection accuracy exceeding 97%, often around 99%, while maintaining a false rejection rate of 1%. TRAIT remains effective under the formulated strong adaptive attacks. 

**Abstract (ZH)**: 首次揭示 adversarial example 攻击导致的可辨识的时间轨迹印记：一种全新的时间维度分析范式 

---
# Enhancing Collective Intelligence in Large Language Models Through Emotional Integration 

**Title (ZH)**: 通过情绪整合增强大型语言模型的集体智能 

**Authors**: Likith Kadiyala, Ramteja Sajja, Yusuf Sermet, Ibrahim Demir  

**Link**: [PDF](https://arxiv.org/pdf/2503.04849)  

**Abstract**: This research investigates the integration of emotional diversity into Large Language Models (LLMs) to enhance collective intelligence. Inspired by the human wisdom of crowds phenomenon, where group decisions often outperform individual judgments, we fine-tuned the DarkIdol-Llama-3.1-8B model using Google's GoEmotions dataset and Low-Rank Adaptation (LoRA) to simulate emotionally diverse responses. Evaluating the model on a distance estimation task between Fargo, ND, and Seattle, WA, across 15,064 unique persona configurations, we analyzed how emotional states and social attributes influence decision-making. Our findings demonstrate that emotional integration shapes response patterns while maintaining acceptable prediction accuracy, revealing its potential to enhance artificial collective intelligence. This study provides valuable insights into the interplay of emotional diversity and decision-making in LLMs, suggesting pathways for creating emotionally aware AI systems that balance emotional depth with analytical precision. 

**Abstract (ZH)**: 本研究调查了将情感多样性整合到大规模语言模型中以增强集体智能的方法。受人群中智慧群体现象的启发，该现象表明集体决策往往优于个人判断，我们使用Google的GoEmotions数据集和低秩适应（LoRA）对DarkIdol-Llama-3.1-8B模型进行了微调，以模拟情感多样化的响应。在对北达科他州法戈市和华盛顿州西雅图市之间距离估计任务的评估中，我们研究了15,064种独特的人设配置下情感状态和社会属性如何影响决策。研究结果表明，情感整合改变了响应模式，同时保持了可接受的预测准确性，揭示了其增强人工集体智能的潜力。本研究提供了关于情感多样性与大规模语言模型中决策相互作用的有价值见解，建议了创建既具备情感深度又兼具分析精确性的感知情感AI系统的途径。 

---
# Role of Databases in GenAI Applications 

**Title (ZH)**: 数据库在生成式AI应用中的作用 

**Authors**: Santosh Bhupathi  

**Link**: [PDF](https://arxiv.org/pdf/2503.04847)  

**Abstract**: Generative AI (GenAI) is transforming industries by enabling intelligent content generation, automation, and decision-making. However, the effectiveness of GenAI applications depends significantly on efficient data storage, retrieval, and contextual augmentation. This paper explores the critical role of databases in GenAI workflows, emphasizing the importance of choosing the right database architecture to optimize performance, accuracy, and scalability. It categorizes database roles into conversational context (key-value/document databases), situational context (relational databases/data lakehouses), and semantic context (vector databases) each serving a distinct function in enriching AI-generated responses. Additionally, the paper highlights real-time query processing, vector search for semantic retrieval, and the impact of database selection on model efficiency and scalability. By leveraging a multi-database approach, GenAI applications can achieve more context-aware, personalized, and high-performing AI-driven solutions. 

**Abstract (ZH)**: 生成式人工智能（GenAI）正在通过智能内容生成、自动化和决策改造各行各业。然而，GenAI应用的有效性在很大程度上取决于高效的数据存储、检索和背景增强。本文探讨了数据库在GenAI工作流中的关键作用，强调选择合适的数据库架构以优化性能、准确性和可扩展性的重要性。文章将数据库角色划分为会话上下文（键值/文档数据库）、情境上下文（关系数据库/数据湖屋）和语义上下文（向量数据库），每种角色在丰富AI生成的响应方面发挥着独特的作用。此外，文章还强调了实时查询处理、向量搜索以实现语义检索以及数据库选择对模型效率和可扩展性的影响。通过运用多数据库方法，GenAI应用可以实现更具有上下文意识、个性化和高性能的人工智能驱动解决方案。 

---
# Universal Narrative Model: an Author-centric Storytelling Framework for Generative AI 

**Title (ZH)**: 面向生成式AI的以作者为中心的叙事框架：通用叙事模型 

**Authors**: Hank Gerba  

**Link**: [PDF](https://arxiv.org/pdf/2503.04844)  

**Abstract**: Generative AI promises to finally realize dynamic, personalized storytelling technologies across a range of media. To date, experimentation with generative AI in the field of procedural narrative generation has been quite promising from a technical perspective. However, fundamental narrative dilemmas remain, such as the balance between player agency and narrative coherence, and no rigorous narrative standard has been proposed to specifically leverage the strengths of generative AI. In this paper, we propose the Universal Narrative Model (UNM), an open and extensible standard designed to place writers at the center of future narrative design workflows and enable interoperability across authoring platforms. By encoding an author's intent according to an objective narrative model, the UNM enables narrative portability as well as intent-based constraints for generative systems. 

**Abstract (ZH)**: 生成式AI有望最终实现跨多种媒体的动态个性化叙事技术。在程序化叙事生成领域，生成式AI的技术实验迄今为止取得了相当有希望的结果。然而，根本的叙事难题仍然存在，例如玩家自主性和叙事连贯性的平衡，以及尚未提出具体的规范标准来充分发挥生成式AI的优势。本文提出了一种通用叙事模型（UNM），这是一种开放扩展的标准，旨在将作家置于未来叙事设计流程的中心，并实现跨作者平台的互操作性。通过根据客观叙事模型编码作者的意图，UNM 使得叙事具有可移植性，并为生成系统提供了基于意图的约束。 

---
# ZAugNet for Z-Slice Augmentation in Bio-Imaging 

**Title (ZH)**: ZAugNet在生物成像中的Z切片增强 

**Authors**: Alessandro Pasqui, Sajjad Mahdavi, Benoit Vianay, Alexandra Colin, Alex McDougall, Rémi Dumollard, Yekaterina A. Miroshnikova, Elsa Labrune, Hervé Turlier  

**Link**: [PDF](https://arxiv.org/pdf/2503.04843)  

**Abstract**: Three-dimensional biological microscopy has significantly advanced our understanding of complex biological structures. However, limitations due to microscopy techniques, sample properties or phototoxicity often result in poor z-resolution, hindering accurate cellular measurements. Here, we introduce ZAugNet, a fast, accurate, and self-supervised deep learning method for enhancing z-resolution in biological images. By performing nonlinear interpolation between consecutive slices, ZAugNet effectively doubles resolution with each iteration. Compared on several microscopy modalities and biological objects, it outperforms competing methods on most metrics. Our method leverages a generative adversarial network (GAN) architecture combined with knowledge distillation to maximize prediction speed without compromising accuracy. We also developed ZAugNet+, an extended version enabling continuous interpolation at arbitrary distances, making it particularly useful for datasets with nonuniform slice spacing. Both ZAugNet and ZAugNet+ provide high-performance, scalable z-slice augmentation solutions for large-scale 3D imaging. They are available as open-source frameworks in PyTorch, with an intuitive Colab notebook interface for easy access by the scientific community. 

**Abstract (ZH)**: 三维生物显微镜显著提高了我们对复杂生物结构的理解。然而，由于显微镜技术的限制、样本特性或光毒性，往往会导致较差的z分辨率，阻碍了准确的细胞测量。在这里，我们介绍了ZAugNet，这是一种快速、准确且自我监督的深度学习方法，用于增强生物图像的z分辨率。通过在连续切片间进行非线性插值，ZAugNet在每次迭代中有效提高了分辨率。在多种显微镜模式和生物对象的对比中，其在大多数指标上优于竞争方法。我们的方法结合生成对抗网络（GAN）架构与知识蒸馏，最大化预测速度同时不牺牲准确性。我们还开发了ZAugNet+，一个增强版本，能够在任意距离实现连续插值，特别适用于非均匀切片间距的数据集。ZAugNet和ZAugNet+为大规模3D成像提供了高性能、可扩展的z切片增强解决方案。它们以PyTorch开源框架形式提供，并附有直观的Colab笔记本界面，便于科学界访问。 

---
# Replicating Human Social Perception in Generative AI: Evaluating the Valence-Dominance Model 

**Title (ZH)**: 在生成式AI中复制人类社会知觉：评价情感主导模型 

**Authors**: Necdet Gurkan, Kimathi Njoki, Jordan W. Suchow  

**Link**: [PDF](https://arxiv.org/pdf/2503.04842)  

**Abstract**: As artificial intelligence (AI) continues to advance--particularly in generative models--an open question is whether these systems can replicate foundational models of human social perception. A well-established framework in social cognition suggests that social judgments are organized along two primary dimensions: valence (e.g., trustworthiness, warmth) and dominance (e.g., power, assertiveness). This study examines whether multimodal generative AI systems can reproduce this valence-dominance structure when evaluating facial images and how their representations align with those observed across world regions. Through principal component analysis (PCA), we found that the extracted dimensions closely mirrored the theoretical structure of valence and dominance, with trait loadings aligning with established definitions. However, many world regions and generative AI models also exhibited a third component, the nature and significance of which warrant further investigation. These findings demonstrate that multimodal generative AI systems can replicate key aspects of human social perception, raising important questions about their implications for AI-driven decision-making and human-AI interactions. 

**Abstract (ZH)**: 随着人工智能（AI）的不断发展——特别是在生成模型方面——一个开放式问题是这些系统是否能够复制人类社会感知的基础模型。社会认知中一个成熟的框架表明，社会判断沿着两个主要维度组织：效价（例如，信任度，温暖）和支配性（例如，权力，主导性）。本研究探讨了多模态生成AI系统在评估面部图像时是否能够重现这一效价-支配性结构，以及它们的表示方式如何与世界各地的观察结果相一致。通过主成分分析（PCA），我们发现提取的维度与理论中的效价和支配性结构高度一致，特性载荷与既定定义相符。然而，许多世界地区和生成AI模型还表现出第三个成分，其性质和意义需要进一步研究。这些发现表明，多模态生成AI系统能够复制人类社会感知的关键方面，这引发了关于其对AI驱动决策和人机交互影响的重要问题。 

---
# Framing the Game: How Context Shapes LLM Decision-Making 

**Title (ZH)**: 框架游戏：背景如何塑造大语言模型决策-making 

**Authors**: Isaac Robinson, John Burden  

**Link**: [PDF](https://arxiv.org/pdf/2503.04840)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed across diverse contexts to support decision-making. While existing evaluations effectively probe latent model capabilities, they often overlook the impact of context framing on perceived rational decision-making. In this study, we introduce a novel evaluation framework that systematically varies evaluation instances across key features and procedurally generates vignettes to create highly varied scenarios. By analyzing decision-making patterns across different contexts with the same underlying game structure, we uncover significant contextual variability in LLM responses. Our findings demonstrate that this variability is largely predictable yet highly sensitive to framing effects. Our results underscore the need for dynamic, context-aware evaluation methodologies for real-world deployments. 

**Abstract (ZH)**: 大型语言模型（LLMs）在不同情境中日益被部署以支持决策。虽然现有的评估有效探测了潜在模型能力，但往往忽视了情境框架对感知理性决策的影响。本研究引入了一种新的评估框架，系统地在关键特征上变化评估实例，并通过程序生成情景片断，创造高度多样的场景。通过在具有相同底层游戏结构的不同情境下分析决策模式，我们揭示了LLM响应中的显著情境变化性。我们的研究发现，这种变化性是高度可预测的，但对框架效应极为敏感。结果强调了在实际部署中需要动态的情境感知评估方法的重要性。 

---
# Advancing Multimodal In-Context Learning in Large Vision-Language Models with Task-aware Demonstrations 

**Title (ZH)**: 在大型视觉语言模型中通过任务感知演示促进多模态上下文学习 

**Authors**: Yanshu Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.04839)  

**Abstract**: Multimodal in-context learning (ICL) has emerged as a key capability of Large Vision-Language Models (LVLMs), driven by their increasing scale and applicability. Despite its promise, effective ICL in the multimodal setting remains challenging due to the inherent complexity of image-text inputs and the high sensitivity of ICL performance to input configurations. In this work, we shed light on the core mechanism underlying multimodal ICL, identifying task mapping as a crucial factor in configuring robust in-context demonstration (ICD) sequences. Building on these insights, we propose \textit{SabER}, a lightweight yet powerful decoder-only transformer equipped with task-aware attention, which intelligently selects and arranges ICDs from a demonstration library in an autoregressive fashion. This design enables fine-grained feature extraction and cross-modal reasoning, iteratively refining task mapping to generate high-quality ICD sequences. Through extensive experiments covering five LVLMs and nine benchmark datasets, SabER not only demonstrates strong empirical performance, but also provides deeper understanding of how task semantics interact with multimodal ICDs. Our findings highlight the importance of principled ICD sequence configuration and open new avenues to enhance multimodal ICL in a wide range of real-world scenarios. 

**Abstract (ZH)**: 多模态上下文学习（ICL）已成为大型视觉-语言模型（LVLMs）的关键能力，随着模型规模的增加和应用场景的扩展而兴起。尽管前景广阔，但在多模态环境中有效的ICL仍然具有挑战性，这归因于图像-文本输入的内在复杂性以及ICL性能对输入配置的高敏感性。在本文中，我们揭示了多模态ICL的核心机制，并识别任务映射是配置鲁棒的上下文示范（ICD）序列的关键因素。基于这些见解，我们提出了一种轻量级但强大的仅解码器变换器\textit{SabER}，它配备了任务感知注意力机制，并以自回归方式智能选择和排列示范库中的ICD。这种设计使细粒度特征提取和跨模态推理成为可能，迭代优化任务映射以生成高质量的ICD序列。通过涵盖五种LVLM和九个基准数据集的 extensive 实验，\textit{SabER} 不仅展示了强大的实证性能，还加深了对任务语义与多模态ICD之间相互作用的理解。我们的发现强调了原理性ICD序列配置的重要性，并为在广泛的实际场景中增强多模态ICL开辟了新的途径。 

---
# FedPalm: A General Federated Learning Framework for Closed- and Open-Set Palmprint Verification 

**Title (ZH)**: FedPalm: 一种用于封闭集和开放集掌纹验证的通用联邦学习框架 

**Authors**: Ziyuan Yang, Yingyu Chen, Chengrui Gao, Andrew Beng Jin Teoh, Bob Zhang, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04837)  

**Abstract**: Current deep learning (DL)-based palmprint verification models rely on centralized training with large datasets, which raises significant privacy concerns due to biometric data's sensitive and immutable nature. Federated learning~(FL), a privacy-preserving distributed learning paradigm, offers a compelling alternative by enabling collaborative model training without the need for data sharing. However, FL-based palmprint verification faces critical challenges, including data heterogeneity from diverse identities and the absence of standardized evaluation benchmarks. This paper addresses these gaps by establishing a comprehensive benchmark for FL-based palmprint verification, which explicitly defines and evaluates two practical scenarios: closed-set and open-set verification. We propose FedPalm, a unified FL framework that balances local adaptability with global generalization. Each client trains a personalized textural expert tailored to local data and collaboratively contributes to a shared global textural expert for extracting generalized features. To further enhance verification performance, we introduce a Textural Expert Interaction Module that dynamically routes textural features among experts to generate refined side textural features. Learnable parameters are employed to model relationships between original and side features, fostering cross-texture-expert interaction and improving feature discrimination. Extensive experiments validate the effectiveness of FedPalm, demonstrating robust performance across both scenarios and providing a promising foundation for advancing FL-based palmprint verification research. 

**Abstract (ZH)**: 基于联邦学习的掌纹验证：综合基准与FedPalm框架 

---
# PGAD: Prototype-Guided Adaptive Distillation for Multi-Modal Learning in AD Diagnosis 

**Title (ZH)**: PGAD：原型引导的自适应 distilled 多模态学习在AD诊断中的应用 

**Authors**: Yanfei Li, Teng Yin, Wenyi Shang, Jingyu Liu, Xi Wang, Kaiyang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04836)  

**Abstract**: Missing modalities pose a major issue in Alzheimer's Disease (AD) diagnosis, as many subjects lack full imaging data due to cost and clinical constraints. While multi-modal learning leverages complementary information, most existing methods train only on complete data, ignoring the large proportion of incomplete samples in real-world datasets like ADNI. This reduces the effective training set and limits the full use of valuable medical data. While some methods incorporate incomplete samples, they fail to effectively address inter-modal feature alignment and knowledge transfer challenges under high missing rates. To address this, we propose a Prototype-Guided Adaptive Distillation (PGAD) framework that directly incorporates incomplete multi-modal data into training. PGAD enhances missing modality representations through prototype matching and balances learning with a dynamic sampling strategy. We validate PGAD on the ADNI dataset with varying missing rates (20%, 50%, and 70%) and demonstrate that it significantly outperforms state-of-the-art approaches. Ablation studies confirm the effectiveness of prototype matching and adaptive sampling, highlighting the potential of our framework for robust and scalable AD diagnosis in real-world clinical settings. 

**Abstract (ZH)**: 缺失模态数据在阿尔茨海默病诊断中构成重大挑战：现有的方法大多仅在完整数据上进行训练，忽略了真实世界数据集中（如ADNI）大量不完整样本的价值，这限制了有效训练集的大小并限制了宝贵医学数据的充分利用。为此，我们提出了一种原型导向自适应蒸馏（PGAD）框架，该框架可以直接将不完整多模态数据纳入训练。PGAD通过原型匹配增强缺失模态表示，并借助动态采样策略平衡学习。我们在不同缺失率（20%，50%，70%）的ADNI数据集上验证了PGAD，并且结果显示其显著优于现有先进方法。消融实验确认了原型匹配和自适应采样的有效性，突显了该框架在真实临床环境中进行鲁棒且可扩展的阿尔茨海默病诊断的潜力。 

---
# Distilling Dataset into Neural Field 

**Title (ZH)**: 将数据集提炼为神经场模型 

**Authors**: Donghyeok Shin, HeeSun Bae, Gyuwon Sim, Wanmo Kang, Il-Chul Moon  

**Link**: [PDF](https://arxiv.org/pdf/2503.04835)  

**Abstract**: Utilizing a large-scale dataset is essential for training high-performance deep learning models, but it also comes with substantial computation and storage costs. To overcome these challenges, dataset distillation has emerged as a promising solution by compressing the large-scale dataset into a smaller synthetic dataset that retains the essential information needed for training. This paper proposes a novel parameterization framework for dataset distillation, coined Distilling Dataset into Neural Field (DDiF), which leverages the neural field to store the necessary information of the large-scale dataset. Due to the unique nature of the neural field, which takes coordinates as input and output quantity, DDiF effectively preserves the information and easily generates various shapes of data. We theoretically confirm that DDiF exhibits greater expressiveness than some previous literature when the utilized budget for a single synthetic instance is the same. Through extensive experiments, we demonstrate that DDiF achieves superior performance on several benchmark datasets, extending beyond the image domain to include video, audio, and 3D voxel. We release the code at this https URL. 

**Abstract (ZH)**: 利用大规模数据集训练高性能深度学习模型至关重要，但也会带来显著的计算和存储成本。为克服这些挑战，数据集蒸馏作为一种有前景的解决方案应运而生，通过压缩大规模数据集为较小的合成数据集，同时保留必要的训练信息。本文提出了一种名为Distilling Dataset into Neural Field (DDiF) 的新型参数化框架，该框架利用神经场存储大规模数据集的必要信息。由于神经场的独特性质，它可以将坐标作为输入并输出数量，DDiF有效地保留了信息，并能够轻松生成各种形状的数据。理论分析证实，在相同的单个合成实例预算下，DDiF表现出了比某些先前文献更大的表达能力。通过广泛的实验，我们展示了在多个基准数据集上，DDiF实现了优于其他方法的性能，不仅限于图像领域，还包括视频、音频和3D体素。我们已在以下网址发布代码：此 https URL。 

---
# Extrapolation Merging: Keep Improving With Extrapolation and Merging 

**Title (ZH)**: 外推融合：通过外推和融合持续改进 

**Authors**: Yiguan Lin, Bin Xu, Yinghao Li, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04834)  

**Abstract**: Large Language Models (LLMs) require instruction fine-tuning to perform different downstream tasks. However, the instruction fine-tuning phase still demands significant computational resources and labeled data, lacking a paradigm that can improve model performance without additional computational power and data. Model merging aims to enhance performance by combining the parameters of different models, but the lack of a clear optimization direction during the merging process does not always guarantee improved performance. In this paper, we attempt to provide a clear optimization direction for model merging. We first validate the effectiveness of the model extrapolation method during the instruction fine-tuning phase. Then, we propose Extrapolation Merging, a paradigm that can continue improving model performance without requiring extra computational resources or data. Using the extrapolation method, we provide a clear direction for model merging, achieving local optimization search, and consequently enhancing the merged model's performance. We conduct experiments on seven different tasks, and the results show that our method can consistently improve the model's performance after fine-tuning. 

**Abstract (ZH)**: 大型语言模型（LLMs）需要指令微调以执行不同的下游任务。然而，指令微调阶段仍然需要大量计算资源和标注数据，缺乏一种可以在不增加计算资源和数据的情况下提升模型性能的范式。模型合并旨在通过合并不同模型的参数来提高性能，但在合并过程中缺乏明确的优化方向并不总是能够保证性能的提升。本文尝试为模型合并提供一个明确的优化方向。我们首先验证了指令微调阶段模型外推方法的有效性。然后，我们提出了外推合并这一范式，可以在不需额外计算资源和数据的情况下继续提升模型性能。利用外推方法，我们为模型合并提供了一个明确的方向，实现了局部优化搜索，从而提升了合并模型的性能。我们在七个不同的任务上进行了实验，结果表明，我们的方法可以在微调后一致地提升模型的性能。 

---
# Adversarial Training for Multimodal Large Language Models against Jailbreak Attacks 

**Title (ZH)**: 对抗训练以抵御 Jailbreak 攻击的多模态大型语言模型 

**Authors**: Liming Lu, Shuchao Pang, Siyuan Liang, Haotian Zhu, Xiyu Zeng, Aishan Liu, Yunhuai Liu, Yongbin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.04833)  

**Abstract**: Multimodal large language models (MLLMs) have made remarkable strides in cross-modal comprehension and generation tasks. However, they remain vulnerable to jailbreak attacks, where crafted perturbations bypass security guardrails and elicit harmful outputs. In this paper, we present the first adversarial training (AT) paradigm tailored to defend against jailbreak attacks during the MLLM training phase. Extending traditional AT to this domain poses two critical challenges: efficiently tuning massive parameters and ensuring robustness against attacks across multiple modalities. To address these challenges, we introduce Projection Layer Against Adversarial Training (ProEAT), an end-to-end AT framework. ProEAT incorporates a projector-based adversarial training architecture that efficiently handles large-scale parameters while maintaining computational feasibility by focusing adversarial training on a lightweight projector layer instead of the entire model; additionally, we design a dynamic weight adjustment mechanism that optimizes the loss function's weight allocation based on task demands, streamlining the tuning process. To enhance defense performance, we propose a joint optimization strategy across visual and textual modalities, ensuring robust resistance to jailbreak attacks originating from either modality. Extensive experiments conducted on five major jailbreak attack methods across three mainstream MLLMs demonstrate the effectiveness of our approach. ProEAT achieves state-of-the-art defense performance, outperforming existing baselines by an average margin of +34% across text and image modalities, while incurring only a 1% reduction in clean accuracy. Furthermore, evaluations on real-world embodied intelligent systems highlight the practical applicability of our framework, paving the way for the development of more secure and reliable multimodal systems. 

**Abstract (ZH)**: 基于对抗训练的多模态大型语言模型防范狱 break 攻击框架 

---
# RD Efficient FPGA Deployment of Learned Image Compression: Knowledge Distillation and Hybrid Quantization 

**Title (ZH)**: 基于知识精炼和混合量化的大规模FPGA部署学习型图像压缩 

**Authors**: Mazouz Alaa Eddine, Sumanta Chaudhuri, Marco Cagnanzzo, Mihai Mitrea, Enzo Tartaglione, Attilio Fiandrotti  

**Link**: [PDF](https://arxiv.org/pdf/2503.04832)  

**Abstract**: Learnable Image Compression (LIC) has shown the potential to outperform standardized video codecs in RD efficiency, prompting the research for hardware-friendly implementations. Most existing LIC hardware implementations prioritize latency to RD-efficiency and through an extensive exploration of the hardware design space. We present a novel design paradigm where the burden of tuning the design for a specific hardware platform is shifted towards model dimensioning and without compromising on RD-efficiency. First, we design a framework for distilling a leaner student LIC model from a reference teacher: by tuning a single model hyperparameters, we can meet the constraints of different hardware platforms without a complex hardware design exploration. Second, we propose a hardware-friendly implementation of the Generalized Divisive Normalization (GDN) activation that preserves RD efficiency even post parameter quantization. Third, we design a pipelined FPGA configuration which takes full advantage of available FPGA resources by leveraging parallel processing and optimizing resource allocation. Our experiments with a state of the art LIC model show that we outperform all existing FPGA implementations while performing very close to the original model in terms of RD efficiency. 

**Abstract (ZH)**: 可学习图像压缩（LIC）展现了超越标准化视频编解码器的潜力，尤其是在RD效率方面，推动了硬件友好型实现的研究。现有的LIC硬件实现主要侧重于降低延迟以提高RD效率，并通过广泛探索硬件设计空间。我们提出了一种新的设计范式，将针对特定硬件平台调整设计的负担转移到模型维度化上，而不牺牲RD效率。首先，我们设计了一个框架，从一个参考教师模型中提取一个更精益的学生LIC模型：通过调整单一模型的超参数，可以在不进行复杂硬件设计探索的情况下满足不同硬件平台的约束条件。其次，我们提出了一种硬件友好的Generalized Divisive Normalization（GDN）激活函数实现，即使在参数量化后仍能保持RD效率。第三，我们设计了一个流水线FPGA配置，通过利用并行处理并优化资源分配来充分利用FPGA资源。我们的实验表明，在使用最先进的LIC模型时，我们在RD效率方面优于所有现有的FPGA实现，且性能与原始模型非常接近。 

---
# "Only ChatGPT gets me": An Empirical Analysis of GPT versus other Large Language Models for Emotion Detection in Text 

**Title (ZH)**: “只有ChatGPT能打动我”：基于文本情绪检测的GPT与其他大型语言模型的实证分析 

**Authors**: Florian Lecourt, Madalina Croitoru, Konstantin Todorov  

**Link**: [PDF](https://arxiv.org/pdf/2503.04831)  

**Abstract**: This work investigates the capabilities of large language models (LLMs) in detecting and understanding human emotions through text. Drawing upon emotion models from psychology, we adopt an interdisciplinary perspective that integrates computational and affective sciences insights. The main goal is to assess how accurately they can identify emotions expressed in textual interactions and compare different models on this specific task. This research contributes to broader efforts to enhance human-computer interaction, making artificial intelligence technologies more responsive and sensitive to users' emotional nuances. By employing a methodology that involves comparisons with a state-of-the-art model on the GoEmotions dataset, we aim to gauge LLMs' effectiveness as a system for emotional analysis, paving the way for potential applications in various fields that require a nuanced understanding of human language. 

**Abstract (ZH)**: 本研究探讨了大型语言模型（LLMs）在通过文本检测和理解人类情绪方面的能力。借鉴心理学中的情绪模型，我们采取跨学科视角，结合计算科学和情感科学的见解。主要目标是评估它们在识别文本交互中表达的情绪方面的准确度，并将不同模型在此特定任务上进行对比。本研究为提升人机交互水平，使人工智能技术更加敏感和细腻地捕捉用户的情绪细微差别做出了贡献。通过在GoEmotions数据集上与最先进的模型进行比较的方法，我们旨在评估LLMs作为情绪分析系统的有效性，为需要细致理解人类语言的各种领域开辟潜在应用途径。 

---
# Cite Before You Speak: Enhancing Context-Response Grounding in E-commerce Conversational LLM-Agents 

**Title (ZH)**: 引用再发言：增强电子商务对话型LLM代理的情境-响应关联 

**Authors**: Jingying Zeng, Hui Liu, Zhenwei Dai, Xianfeng Tang, Chen Luo, Samarth Varshney, Zhen Li, Qi He  

**Link**: [PDF](https://arxiv.org/pdf/2503.04830)  

**Abstract**: With the advancement of conversational large language models (LLMs), several LLM-based Conversational Shopping Agents (CSA) have been developed to help customers answer questions and smooth their shopping journey in e-commerce domain. The primary objective in building a trustworthy CSA is to ensure the agent's responses are accurate and factually grounded, which is essential for building customer trust and encouraging continuous engagement. However, two challenges remain. First, LLMs produce hallucinated or unsupported claims. Such inaccuracies risk spreading misinformation and diminishing customer trust. Second, without providing knowledge source attribution in CSA response, customers struggle to verify LLM-generated information. To address these challenges, we present an easily productionized solution that enables a "citation experience" utilizing In-context Learning (ICL) and Multi-UX-Inference (MUI) to generate responses with citations to attribute its original sources without interfering other existing UX features. With proper UX design, these citation marks can be linked to the related product information and display the source to our customers. In this work, we also build auto-metrics and scalable benchmarks to holistically evaluate LLM's grounding and attribution capabilities. Our experiments demonstrate that incorporating this citation generation paradigm can substantially enhance the grounding of LLM responses by 13.83% on the real-world data. As such, our solution not only addresses the immediate challenges of LLM grounding issues but also adds transparency to conversational AI. 

**Abstract (ZH)**: 基于大规模语言模型的可信赖对话型购物代理的发展：利用上下文学习和多用户体验推断的引证经验实现语义接地与知识溯源 

---
# StickMotion: Generating 3D Human Motions by Drawing a Stickman 

**Title (ZH)**: StickMotion: 通过绘制 stickman 生成 3D 人体运动 

**Authors**: Tao Wang, Zhihua Wu, Qiaozhi He, Jiaming Chu, Ling Qian, Yu Cheng, Junliang Xing, Jian Zhao, Lei Jin  

**Link**: [PDF](https://arxiv.org/pdf/2503.04829)  

**Abstract**: Text-to-motion generation, which translates textual descriptions into human motions, has been challenging in accurately capturing detailed user-imagined motions from simple text inputs. This paper introduces StickMotion, an efficient diffusion-based network designed for multi-condition scenarios, which generates desired motions based on traditional text and our proposed stickman conditions for global and local control of these motions, respectively. We address the challenges introduced by the user-friendly stickman from three perspectives: 1) Data generation. We develop an algorithm to generate hand-drawn stickmen automatically across different dataset formats. 2) Multi-condition fusion. We propose a multi-condition module that integrates into the diffusion process and obtains outputs of all possible condition combinations, reducing computational complexity and enhancing StickMotion's performance compared to conventional approaches with the self-attention module. 3) Dynamic supervision. We empower StickMotion to make minor adjustments to the stickman's position within the output sequences, generating more natural movements through our proposed dynamic supervision strategy. Through quantitative experiments and user studies, sketching stickmen saves users about 51.5% of their time generating motions consistent with their imagination. Our codes, demos, and relevant data will be released to facilitate further research and validation within the scientific community. 

**Abstract (ZH)**: 基于文本到运动生成的StickMotion：一种高效控制全局和局部运动的扩散网络 

---
# Beyond Next Word Prediction: Developing Comprehensive Evaluation Frameworks for measuring LLM performance on real world applications 

**Title (ZH)**: 超越下一个词预测：开发全面评价框架以衡量大模型在实际应用中的性能 

**Authors**: Vishakha Agrawal, Archie Chaudhury, Shreya Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2503.04828)  

**Abstract**: While Large Language Models (LLMs) are fundamentally next-token prediction systems, their practical applications extend far beyond this basic function. From natural language processing and text generation to conversational assistants and software use, LLMs have numerous use-cases, and have already acquired a significant degree of enterprise adoption. To evaluate such models, static evaluation datasets, consisting of a set of prompts and their corresponding ground truths, are often used to benchmark the efficacy of the model for a particular task. In this paper, we provide the basis for a more comprehensive evaluation framework, based upon a traditional game and tool-based architecture that enables a more overarching measurement of a model's capabilities. For simplicity, we provide a generalized foundation that can be extended, without significant alteration, to numerous scenarios, from specific use cases such as supply chain management or financial reasoning, to abstract measurements such as ethics or safety. 

**Abstract (ZH)**: 虽然大型语言模型（LLMs）本质上是下一-token预测系统，但它们的实际应用远远超出了这一基本功能。从自然语言处理和文本生成到对话助手和软件使用，LLMs 有着众多的应用场景，并已获得了相当程度的企业采用。为了评估这些模型，通常会使用包含一组提示及其对应真实结果的静态评估数据集来衡量模型在特定任务上的有效性。在本文中，我们提供了一种更为全面的评价框架的基础，基于传统游戏和工具架构，能够更全面地衡量模型的能力。为了简化，我们提供了可以扩展且无需显著修改的基础框架，适用于从具体应用场景如供应链管理或金融推理，到抽象指标如伦理或安全性的多种情景。 

---
# Preserving Cultural Identity with Context-Aware Translation Through Multi-Agent AI Systems 

**Title (ZH)**: 基于多智能体AI系统的上下文感知翻译 preserve Cultural Identity Across Languages 

**Authors**: Mahfuz Ahmed Anik, Abdur Rahman, Azmine Toushik Wasi, Md Manjurul Ahsan  

**Link**: [PDF](https://arxiv.org/pdf/2503.04827)  

**Abstract**: Language is a cornerstone of cultural identity, yet globalization and the dominance of major languages have placed nearly 3,000 languages at risk of extinction. Existing AI-driven translation models prioritize efficiency but often fail to capture cultural nuances, idiomatic expressions, and historical significance, leading to translations that marginalize linguistic diversity. To address these challenges, we propose a multi-agent AI framework designed for culturally adaptive translation in underserved language communities. Our approach leverages specialized agents for translation, interpretation, content synthesis, and bias evaluation, ensuring that linguistic accuracy and cultural relevance are preserved. Using CrewAI and LangChain, our system enhances contextual fidelity while mitigating biases through external validation. Comparative analysis shows that our framework outperforms GPT-4o, producing contextually rich and culturally embedded translations, a critical advancement for Indigenous, regional, and low-resource languages. This research underscores the potential of multi-agent AI in fostering equitable, sustainable, and culturally sensitive NLP technologies, aligning with the AI Governance, Cultural NLP, and Sustainable NLP pillars of Language Models for Underserved Communities. Our full experimental codebase is publicly available at: this https URL 

**Abstract (ZH)**: 语言是文化身份的基础，然而全球化和主要语言的主导地位使得近3000种语言面临灭绝的风险。现有的基于AI的翻译模型注重效率，但常常无法捕捉文化细微差别、习语表达和历史意义，导致翻译结果边缘化语言多样性。为应对这些挑战，我们提出了一种多智能体AI框架，旨在为未充分服务的语言社区提供文化适应性翻译。我们的方法利用专门的智能体进行翻译、口译、内容合成和偏见评估，确保语言准确性和文化相关性得到保留。通过使用CrewAI和LangChain，我们的系统增强了上下文的真实性，同时通过外部验证减轻偏见。比较分析表明，我们的框架优于GPT-4o，产生丰富上下文并融入文化内容的翻译，这对原住民、区域性和低资源语言而言是一项关键进步。本研究强调了多智能体AI在促进公平、可持续和文化敏感的自然语言处理技术方面的潜力，符合未服务社区语言模型的AI治理、文化NLP和可持续NLP支柱。我们的完整实验代码库已公开发布于：this https URL 

---
# DA-STGCN: 4D Trajectory Prediction Based on Spatiotemporal Feature Extraction 

**Title (ZH)**: DA-STGCN：基于时空特征提取的4D轨迹预测 

**Authors**: Yuheng Kuang, Zhengning Wang, Jianping Zhang, Zhenyu Shi, Yuding Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04823)  

**Abstract**: The importance of four-dimensional (4D) trajectory prediction within air traffic management systems is on the rise. Key operations such as conflict detection and resolution, aircraft anomaly monitoring, and the management of congested flight paths are increasingly reliant on this foundational technology, underscoring the urgent demand for intelligent solutions. The dynamics in airport terminal zones and crowded airspaces are intricate and ever-changing; however, current methodologies do not sufficiently account for the interactions among aircraft. To tackle these challenges, we propose DA-STGCN, an innovative spatiotemporal graph convolutional network that integrates a dual attention mechanism. Our model reconstructs the adjacency matrix through a self-attention approach, enhancing the capture of node correlations, and employs graph attention to distill spatiotemporal characteristics, thereby generating a probabilistic distribution of predicted trajectories. This novel adjacency matrix, reconstructed with the self-attention mechanism, is dynamically optimized throughout the network's training process, offering a more nuanced reflection of the inter-node relationships compared to traditional algorithms. The performance of the model is validated on two ADS-B datasets, one near the airport terminal area and the other in dense airspace. Experimental results demonstrate a notable improvement over current 4D trajectory prediction methods, achieving a 20% and 30% reduction in the Average Displacement Error (ADE) and Final Displacement Error (FDE), respectively. The incorporation of a Dual-Attention module has been shown to significantly enhance the extraction of node correlations, as verified by ablation experiments. 

**Abstract (ZH)**: 四维（4D）飞行轨迹预测在空中交通管理系统中的重要性日益凸显：基于双注意机制的时空图卷积网络（DA-STGCN）及其应用 

---
# HeTGB: A Comprehensive Benchmark for Heterophilic Text-Attributed Graphs 

**Title (ZH)**: HeTGB: 一种全面的异ophilic文本 attributed 图基准 

**Authors**: Shujie Li, Yuxia Wu, Chuan Shi, Yuan Fang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04822)  

**Abstract**: Graph neural networks (GNNs) have demonstrated success in modeling relational data primarily under the assumption of homophily. However, many real-world graphs exhibit heterophily, where linked nodes belong to different categories or possess diverse attributes. Additionally, nodes in many domains are associated with textual descriptions, forming heterophilic text-attributed graphs (TAGs). Despite their significance, the study of heterophilic TAGs remains underexplored due to the lack of comprehensive benchmarks. To address this gap, we introduce the Heterophilic Text-attributed Graph Benchmark (HeTGB), a novel benchmark comprising five real-world heterophilic graph datasets from diverse domains, with nodes enriched by extensive textual descriptions. HeTGB enables systematic evaluation of GNNs, pre-trained language models (PLMs) and co-training methods on the node classification task. Through extensive benchmarking experiments, we showcase the utility of text attributes in heterophilic graphs, analyze the challenges posed by heterophilic TAGs and the limitations of existing models, and provide insights into the interplay between graph structures and textual attributes. We have publicly released HeTGB with baseline implementations to facilitate further research in this field. 

**Abstract (ZH)**: 异构文本属性图基准（HeTGB）：一种包含五个来自多个领域的异构图数据集的新基准 

---
# RTFusion: A depth estimation network based on multimodal fusion in challenging scenarios 

**Title (ZH)**: RTFusion：基于多模态融合的挑战场景深度估计网络 

**Authors**: Zelin Meng, Takanori Fukao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04821)  

**Abstract**: Depth estimation in complex real-world scenarios is a challenging task, especially when relying solely on a single modality such as visible light or thermal infrared (THR) imagery. This paper proposes a novel multimodal depth estimation model, RTFusion, which enhances depth estimation accuracy and robustness by integrating the complementary strengths of RGB and THR data. The RGB modality provides rich texture and color information, while the THR modality captures thermal patterns, ensuring stability under adverse lighting conditions such as extreme illumination. The model incorporates a unique fusion mechanism, EGFusion, consisting of the Mutual Complementary Attention (MCA) module for cross-modal feature alignment and the Edge Saliency Enhancement Module (ESEM) to improve edge detail preservation. Comprehensive experiments on the MS2 and ViViD++ datasets demonstrate that the proposed model consistently produces high-quality depth maps across various challenging environments, including nighttime, rainy, and high-glare conditions. The experimental results highlight the potential of the proposed method in applications requiring reliable depth estimation, such as autonomous driving, robotics, and augmented reality. 

**Abstract (ZH)**: 复杂现实场景中的深度估计是一项具有挑战性的任务，尤其是在依赖单一模态如可见光或热红外（THR）图像时。本文提出了一种新颖的多模态深度估计模型RTFusion，该模型通过结合RGB和THR数据的互补优势，提高了深度估计的准确性和鲁棒性。RGB模态提供丰富的纹理和颜色信息，而THR模态捕捉热模式，确保在极端光照等恶劣光照条件下保持稳定性。该模型结合了一种独特的融合机制EGFusion，包括用于跨模态特征对齐的Mutual Complementary Attention (MCA) 模块和用于增强边缘细节保留的Edge Saliency Enhancement Module (ESEM)。在MS2和ViViD++数据集上的全面实验显示，所提模型在各种具有挑战性的环境中（包括夜间、雨天和高反射条件）能够生成高质量的深度图。实验结果强调了所提出方法在需要可靠深度估计的应用，如自动驾驶、机器人技术和增强现实中的潜力。 

---
# Technique Inference Engine: A Recommender Model to Support Cyber Threat Hunting 

**Title (ZH)**: 网络威胁狩猎支持的推荐模型：技术推理引擎 

**Authors**: Matthew J. Turner, Mike Carenzo, Jackie Lasky, James Morris-King, James Ross  

**Link**: [PDF](https://arxiv.org/pdf/2503.04819)  

**Abstract**: Cyber threat hunting is the practice of proactively searching for latent threats in a network. Engaging in threat hunting can be difficult due to the volume of network traffic, variety of adversary techniques, and constantly evolving vulnerabilities. To aid analysts in identifying techniques which may be co-occurring as part of a campaign, we present the Technique Inference Engine, a tool to infer tactics, techniques, and procedures (TTPs) which may be related to existing observations of adversarial behavior. We compile the largest (to our knowledge) available dataset of cyber threat intelligence (CTI) reports labeled with relevant TTPs. With the knowledge that techniques are chronically under-reported in CTI, we apply several implicit feedback recommender models to the data in order to predict additional techniques which may be part of a given campaign. We evaluate the results in the context of the cyber analyst's use case and apply t-SNE to visualize the model embeddings. We provide our code and a web interface. 

**Abstract (ZH)**: 网络威胁狩猎是主动搜索网络中潜藏威胁的做法。由于网络流量庞大、对手技术多样以及漏洞不断演变，进行威胁狩猎颇具挑战。为了帮助分析师识别可能在特定活动中共同出现的技术，我们介绍了技术推理引擎，该工具可用于推断可能与现有敌对行为观察相关的战术、技术和程序（TTP）。我们收集了已知最大的网络威胁情报（CTI）报告数据集，并将其标记为相关TTP。鉴于技术在威胁情报中的报告不充分，我们应用了多种隐式反馈推荐模型来预测可能属于特定活动的技术。我们从网络安全分析师的角度评估了结果，并使用t-SNE进行模型嵌入的可视化。我们提供了我们的代码和网络界面。 

---
# Prompting Science Report 1: Prompt Engineering is Complicated and Contingent 

**Title (ZH)**: Prompt Engineering 是复杂且依赖环境的。 

**Authors**: Lennart Meincke, Ethan Mollick, Lilach Mollick, Dan Shapiro  

**Link**: [PDF](https://arxiv.org/pdf/2503.04818)  

**Abstract**: This is the first of a series of short reports that seek to help business, education, and policy leaders understand the technical details of working with AI through rigorous testing. In this report, we demonstrate two things:
- There is no single standard for measuring whether a Large Language Model (LLM) passes a benchmark, and that choosing a standard has a big impact on how well the LLM does on that benchmark. The standard you choose will depend on your goals for using an LLM in a particular case.
- It is hard to know in advance whether a particular prompting approach will help or harm the LLM's ability to answer any particular question. Specifically, we find that sometimes being polite to the LLM helps performance, and sometimes it lowers performance. We also find that constraining the AI's answers helps performance in some cases, though it may lower performance in other cases.
Taken together, this suggests that benchmarking AI performance is not one-size-fits-all, and also that particular prompting formulas or approaches, like being polite to the AI, are not universally valuable. 

**Abstract (ZH)**: 这是关于通过严谨测试帮助商业、教育和政策领导者理解与人工智能合作技术细节的一系列简短报告中的第一篇。本报告展示了两方面内容：
- 没有一种统一的标准来衡量大型语言模型（LLM）是否通过了一项基准测试，选择不同的标准会影响LLM在该基准测试中的表现。你选择的标准将取决于你在特定情况下使用LLM的目标。
- 在预先确定某一特定提示方法是否有助于或损害LLM回答特定问题的能力方面存在困难。具体而言，我们发现有时对LLM礼貌有助于提高性能，有时则会降低性能。我们还发现，限制AI的回答在某些情况下有助于提高性能，但在其他情况下则可能降低性能。
综上所述，这表明评估AI性能并非一刀切的，特定的提示公式或方法，如对AI礼貌，也不是普遍有价值的。 

---
# Multi-Agent System for AI-Assisted Extraction of Narrative Arcs in TV Series 

**Title (ZH)**: 基于多agent系统的AI辅助电视剧叙事弧抽取系统 

**Authors**: Roberto Balestri, Guglielmo Pescatore  

**Link**: [PDF](https://arxiv.org/pdf/2503.04817)  

**Abstract**: Serialized TV shows are built on complex storylines that can be hard to track and evolve in ways that defy straightforward analysis. This paper introduces a multi-agent system designed to extract and analyze these narrative arcs. Tested on the first season of Grey's Anatomy (ABC 2005-), the system identifies three types of arcs: Anthology (self-contained), Soap (relationship-focused), and Genre-Specific (strictly related to the series' genre). Episodic progressions of these arcs are stored in both relational and semantic (vectorial) databases, enabling structured analysis and comparison. To bridge the gap between automation and critical interpretation, the system is paired with a graphical interface that allows for human refinement using tools to enhance and visualize the data. The system performed strongly in identifying Anthology Arcs and character entities, but its reliance on textual paratexts (such as episode summaries) revealed limitations in recognizing overlapping arcs and subtler dynamics. This approach highlights the potential of combining computational and human expertise in narrative analysis. Beyond television, it offers promise for serialized written formats, where the narrative resides entirely in the text. Future work will explore the integration of multimodal inputs, such as dialogue and visuals, and expand testing across a wider range of genres to refine the system further. 

**Abstract (ZH)**: 基于复杂叙事结构的连续剧需要复杂分析方法：一个多智能体系统在《实习医生格蕾》第一季中的应用与拓展 

---
# Normalization through Fine-tuning: Understanding Wav2vec 2.0 Embeddings for Phonetic Analysis 

**Title (ZH)**: 通过微调实现规范化：理解Wav2vec 2.0嵌入在音素分析中的作用 

**Authors**: Yiming Wang, Yi Yang, Jiahong Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2503.04814)  

**Abstract**: Phonetic normalization plays a crucial role in speech recognition and analysis, ensuring the comparability of features derived from raw audio data. However, in the current paradigm of fine-tuning pre-trained large transformer models, phonetic normalization is not deemed a necessary step; instead, it is implicitly executed within the models. This study investigates the normalization process within transformer models, especially wav2vec 2.0. Through a comprehensive analysis of embeddings from models fine-tuned for various tasks, our results demonstrate that fine-tuning wav2vec 2.0 effectively achieves phonetic normalization by selectively suppressing task-irrelevant information. We found that models fine-tuned for multiple tasks retain information for both tasks without compromising performance, and that suppressing task-irrelevant information is not necessary for effective classification. These findings provide new insights into how phonetic normalization can be flexibly achieved in speech models and how it is realized in human speech perception. 

**Abstract (ZH)**: 语音归一化在语音识别和分析中起着关键作用，确保从原始音频数据中提取的特征具有可比性。然而，在当前预训练大型变换器模型的微调 paradigm 中，语音归一化并不被视为必要步骤，而是隐式地在模型内部执行。本研究探讨了变换器模型中的归一化过程，特别是 wav2vec 2.0。通过对各种任务微调后的模型的嵌入进行全面分析，我们的结果表明，微调 wav2vec 2.0 通过选择性抑制与任务无关的信息有效地实现了语音归一化。我们发现，针对多个任务微调的模型保留了两个任务的信息而不牺牲性能，而且抑制与任务无关的信息对于有效的分类并不是必要的。这些发现为语音模型中语音归一化如何灵活实现以及其在人类语音感知中的实现方式提供了新的见解。 

---
# Self-Evolved Preference Optimization for Enhancing Mathematical Reasoning in Small Language Models 

**Title (ZH)**: 自我进化偏好优化以增强小型语言模型的数学推理能力 

**Authors**: Joykirat Singh, Tanmoy Chakraborty, Akshay Nambi  

**Link**: [PDF](https://arxiv.org/pdf/2503.04813)  

**Abstract**: Large language models (LLMs) have significantly improved their reasoning capabilities; however, they still struggle with complex multi-step mathematical problem-solving due to error propagation, lack of self-correction, and limited adaptability to diverse reasoning styles. Existing methods rely on static fine-tuning or prompt engineering, which fail to generalize across problem complexities, while the scarcity of high-quality preference data further hinders reliable reasoning.
We introduce SPHERE, a self-evolving data generation pipeline that enhances reasoning in small language models (SLMs) by iteratively generating, correcting, and diversifying reasoning chains. SPHERE operates in three stages: (i) Self-Generation, where the model autonomously constructs problem-solving steps; (ii) Self-Correction, enabling it to identify and rectify errors; and (iii) Diversity Induction, improving robustness through multiple valid reasoning trajectories. This self-evolution mechanism strengthens mathematical reasoning and enhances model reliability. Evaluations on MATH 500, GSM8K, AIME, AMC, and Olympiad show that SPHERE-trained models achieve significant gains over their base versions and match/surpass GPT-4o on certain benchmarks. Our findings demonstrate that self-evolving models can close the reasoning gap between SLMs and state-of-the-art LLMs, making mathematical AI more reliable, scalable, and efficient. 

**Abstract (ZH)**: 自适应数据生成pipeline增强小型语言模型的数学推理能力：SPHERE 

---
# LLaVE: Large Language and Vision Embedding Models with Hardness-Weighted Contrastive Learning 

**Title (ZH)**: LLaVE: 大规模语言和视觉嵌入模型与硬度加权对比学习 

**Authors**: Zhibin Lan, Liqiang Niu, Fandong Meng, Jie Zhou, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2503.04812)  

**Abstract**: Universal multimodal embedding models play a critical role in tasks such as interleaved image-text retrieval, multimodal RAG, and multimodal clustering. However, our empirical results indicate that existing LMM-based embedding models trained with the standard InfoNCE loss exhibit a high degree of overlap in similarity distribution between positive and negative pairs, making it challenging to distinguish hard negative pairs effectively. To deal with this issue, we propose a simple yet effective framework that dynamically improves the embedding model's representation learning for negative pairs based on their discriminative difficulty. Within this framework, we train a series of models, named LLaVE, and evaluate them on the MMEB benchmark, which covers 4 meta-tasks and 36 datasets. Experimental results show that LLaVE establishes stronger baselines that achieve state-of-the-art (SOTA) performance while demonstrating strong scalability and efficiency. Specifically, LLaVE-2B surpasses the previous SOTA 7B models, while LLaVE-7B achieves a further performance improvement of 6.2 points. Although LLaVE is trained on image-text data, it can generalize to text-video retrieval tasks in a zero-shot manner and achieve strong performance, demonstrating its remarkable potential for transfer to other embedding tasks. 

**Abstract (ZH)**: 通用多模态嵌入模型在交错图像-文本检索、多模态RAG和多模态聚类等任务中发挥着关键作用。然而，我们的实验证据表明，使用标准InfoNCE损失训练的现有基于LLM的嵌入模型，正样本和负样本对之间的相似性分布存在高度重叠，这使得有效区分困难负样本对变得具有挑战性。为了解决这一问题，我们提出了一种简单而有效的框架，该框架根据负样本对的区分难度动态改进嵌入模型的表示学习。在该框架中，我们训练了一系列名为LLaVE的模型，并在MMEB基准上评估它们，该基准涵盖了4个元任务和36个数据集。实验结果表明，LLaVE建立了更强的基础模型，实现了最先进的性能，同时展示了强大的可扩展性和效率。特别地，LLaVE-2B超越了之前的7B模型，而LLaVE-7B进一步提高了6.2个百分点。尽管LLaVE是在图像-文本数据上训练的，但它可以在零样本情况下泛化到文本-视频检索任务，并实现强劲的性能，显示出它在其他嵌入任务上转移的巨大潜力。 

---
# PanguIR Technical Report for NTCIR-18 AEOLLM Task 

**Title (ZH)**: PanguIR 技术报告：NTCIR-18 AEOLLM 任务 

**Authors**: Lang Mei, Chong Chen, Jiaxin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04809)  

**Abstract**: As large language models (LLMs) gain widespread attention in both academia and industry, it becomes increasingly critical and challenging to effectively evaluate their capabilities. Existing evaluation methods can be broadly categorized into two types: manual evaluation and automatic evaluation. Manual evaluation, while comprehensive, is often costly and resource-intensive. Conversely, automatic evaluation offers greater scalability but is constrained by the limitations of its evaluation criteria (dominated by reference-based answers). To address these challenges, NTCIR-18 introduced the AEOLLM (Automatic Evaluation of LLMs) task, aiming to encourage reference-free evaluation methods that can overcome the limitations of existing approaches. In this paper, to enhance the evaluation performance of the AEOLLM task, we propose three key methods to improve the reference-free evaluation: 1) Multi-model Collaboration: Leveraging multiple LLMs to approximate human ratings across various subtasks; 2) Prompt Auto-optimization: Utilizing LLMs to iteratively refine the initial task prompts based on evaluation feedback from training samples; and 3) In-context Learning (ICL) Optimization: Based on the multi-task evaluation feedback, we train a specialized in-context example retrieval model, combined with a semantic relevance retrieval model, to jointly identify the most effective in-context learning examples. Experiments conducted on the final dataset demonstrate that our approach achieves superior performance on the AEOLLM task. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在学术界和工业界中获得广泛关注，有效地评价其能力变得日益重要和具有挑战性。现有的评价方法大致可以分为两类：人工评价和自动评价。人工评价虽然全面，但往往成本高昂且资源密集。相反，自动评价具有更高的可扩展性，但在评价标准的限制下（主要依赖参考答案）受到约束。为了应对这些挑战，NTCIR-18 引入了 AEOLLM（自动评价的LLMs）任务，旨在鼓励克服现有方法局限性的无参考评价方法。在本文中，为了提高AEOLLM任务的评价性能，我们提出了三种关键方法以改进无参考评价：1）多模型协作：利用多个LLM在各种子任务中近似人类评分；2）提示自动优化：使用LLM基于训练样本的评价反馈迭代优化初始任务提示；3）上下文相关学习（ICL）优化：基于多任务评价反馈，我们训练一个专门的上下文相关示例检索模型，结合语义相关性检索模型，共同识别最有效的上下文相关学习示例。在最终数据集上的实验表明，我们的方法在AEOLLM任务上取得了更好的性能。 

---
# Learning from Failures in Multi-Attempt Reinforcement Learning 

**Title (ZH)**: 基于多尝试强化学习中失败的学习 

**Authors**: Stephen Chung, Wenyu Du, Jie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04808)  

**Abstract**: Recent advancements in reinforcement learning (RL) for large language models (LLMs), exemplified by DeepSeek R1, have shown that even a simple question-answering task can substantially improve an LLM's reasoning capabilities. In this work, we extend this approach by modifying the task into a multi-attempt setting. Instead of generating a single response per question, the model is given multiple attempts, with feedback provided after incorrect responses. The multi-attempt task encourages the model to refine its previous attempts and improve search efficiency. Experimental results show that even a small LLM trained on a multi-attempt task achieves significantly higher accuracy when evaluated with more attempts, improving from 45.6% with 1 attempt to 52.5% with 2 attempts on the math benchmark. In contrast, the same LLM trained on a standard single-turn task exhibits only a marginal improvement, increasing from 42.3% to 43.2% when given more attempts during evaluation. The results indicate that, compared to the standard single-turn task, an LLM trained on a multi-attempt task achieves slightly better performance on math benchmarks while also learning to refine its responses more effectively based on user feedback. Full code is available at this https URL 

**Abstract (ZH)**: Recent advancements in reinforcement learning for large language models: Extending reasoning capabilities through a multi-attempt task 

---
# Call for Rigor in Reporting Quality of Instruction Tuning Data 

**Title (ZH)**: 呼吁在报告指令调优数据质量方面保持严谨。 

**Authors**: Hyeonseok Moon, Jaehyung Seo, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2503.04807)  

**Abstract**: Instruction tuning is crucial for adapting large language models (LLMs) to align with user intentions. Numerous studies emphasize the significance of the quality of instruction tuning (IT) data, revealing a strong correlation between IT data quality and the alignment performance of LLMs. In these studies, the quality of IT data is typically assessed by evaluating the performance of LLMs trained with that data. However, we identified a prevalent issue in such practice: hyperparameters for training models are often selected arbitrarily without adequate justification. We observed significant variations in hyperparameters applied across different studies, even when training the same model with the same data. In this study, we demonstrate the potential problems arising from this practice and emphasize the need for careful consideration in verifying data quality. Through our experiments on the quality of LIMA data and a selected set of 1,000 Alpaca data points, we demonstrate that arbitrary hyperparameter decisions can make any arbitrary conclusion. 

**Abstract (ZH)**: 指令调优对于使大型语言模型（LLMs）与用户意图相一致至关重要。众多研究强调了指令调优（IT）数据质量的 significance，指出IT数据质量与LLMs的对齐性能之间存在强烈的相关性。在这些研究中，通常通过评估使用该数据训练的LLMs的性能来评估IT数据质量。然而，我们发现这种做法中存在一个普遍问题：模型训练参数通常会被随意选择而缺乏充分的解释。我们观察到，在使用相同数据训练相同模型的情况下，不同研究中应用的参数存在显著差异。在本研究中，我们展示了这种做法可能导致的问题，并强调了验证数据质量时需要谨慎。通过在LIMA数据质量和1,000个Alpaca数据点上进行的实验，我们证明了任意选择超参数会导致任意结论。 

---
# An energy-efficient learning solution for the Agile Earth Observation Satellite Scheduling Problem 

**Title (ZH)**: 敏捷地球观测卫星调度问题的节能学习解决方案 

**Authors**: Antonio M. Mercado-Martínez, Beatriz Soret, Antonio Jurado-Navas  

**Link**: [PDF](https://arxiv.org/pdf/2503.04803)  

**Abstract**: The Agile Earth Observation Satellite Scheduling Problem (AEOSSP) entails finding the subset of observation targets to be scheduled along the satellite's orbit while meeting operational constraints of time, energy and memory. The problem of deciding what and when to observe is inherently complex, and becomes even more challenging when considering several issues that compromise the quality of the captured images, such as cloud occlusion, atmospheric turbulence, and image resolution. This paper presents a Deep Reinforcement Learning (DRL) approach for addressing the AEOSSP with time-dependent profits, integrating these three factors to optimize the use of energy and memory resources. The proposed method involves a dual decision-making process: selecting the sequence of targets and determining the optimal observation time for each. Our results demonstrate that the proposed algorithm reduces the capture of images that fail to meet quality requirements by > 60% and consequently decreases energy waste from attitude maneuvers by up to 78%, all while maintaining strong observation performance. 

**Abstract (ZH)**: 敏捷遥感卫星调度问题（AEOSSP）涉及在满足时间、能量和内存等操作约束条件下，确定卫星轨道上要调度的观测目标子集。决定观测什么以及何时观测问题本身是复杂的，在考虑诸如云遮挡、大气湍流和图像分辨率等因素影响图像质量的问题时，则变得更加具有挑战性。本文提出了一种基于深度强化学习（DRL）的方法来解决具有时间依赖性收益的AEOSSP问题，将这三个因素综合考虑以优化能量和内存资源的使用。所提出的方法包括双重决策过程：选择目标序列并确定每个目标的最佳观测时间。实验结果表明，所提出的算法可以将不符合质量要求的图像捕获量减少超过60%，从而将姿态机动的能量浪费降低高达78%同时保持强劲的观测性能。 

---
# The order in speech disorder: a scoping review of state of the art machine learning methods for clinical speech classification 

**Title (ZH)**: 言语紊乱中的秩序：先进机器学习方法在临床语音分类中的综述 

**Authors**: Birger Moell, Fredrik Sand Aronsson, Per Östberg, Jonas Beskow  

**Link**: [PDF](https://arxiv.org/pdf/2503.04802)  

**Abstract**: Background:Speech patterns have emerged as potential diagnostic markers for conditions with varying etiologies. Machine learning (ML) presents an opportunity to harness these patterns for accurate disease diagnosis.
Objective: This review synthesized findings from studies exploring ML's capability in leveraging speech for the diagnosis of neurological, laryngeal and mental disorders.
Methods: A systematic examination of 564 articles was conducted with 91 articles included in the study, which encompassed a wide spectrum of conditions, ranging from voice pathologies to mental and neurological disorders. Methods for speech classifications were assessed based on the relevant studies and scored between 0-10 based on the reported diagnostic accuracy of their ML models.
Results: High diagnostic accuracies were consistently observed for laryngeal disorders, dysarthria, and changes related to speech in Parkinsons disease. These findings indicate the robust potential of speech as a diagnostic tool. Disorders like depression, schizophrenia, mild cognitive impairment and Alzheimers dementia also demonstrated high accuracies, albeit with some variability across studies. Meanwhile, disorders like OCD and autism highlighted the need for more extensive research to ascertain the relationship between speech patterns and the respective conditions.
Conclusion: ML models utilizing speech patterns demonstrate promising potential in diagnosing a range of mental, laryngeal, and neurological disorders. However, the efficacy varies across conditions, and further research is needed. The integration of these models into clinical practice could potentially revolutionize the evaluation and diagnosis of a number of different medical conditions. 

**Abstract (ZH)**: 背景：语音模式已被确定为具有不同病因的条件的潜在诊断标志物。机器学习（ML）提供了一种利用这些模式进行准确疾病诊断的机会。目的：本综述综合了研究ML在利用语音进行神经疾病、喉疾病和精神障碍诊断方面能力的发现。方法：系统检查了564篇文章，其中91篇文章被纳入研究，涵盖了从嗓音病理到神经和精神障碍的广泛疾病范围。根据相关研究评估了语音分类方法，并根据其ML模型报告的诊断准确性得分，范围为0-10。结果：喉疾病、言语障碍及帕金森病相关的言语变化显示出一致的高诊断准确性。这些结果表明，语音作为诊断工具具有坚实的潜力。抑郁、精神分裂症、轻度认知障碍和阿尔茨海默病等疾病也表现出较高的准确性，但不同研究之间存在一定的可变性。另一方面，强迫症和自闭症等疾病强调了进一步研究的必要性，以确定语音模式与相应疾病的关系。结论：利用语音模式的ML模型在诊断多种神经、喉及精神障碍方面展现了有希望的潜力。然而，不同疾病的效用存在差异，仍需进行更多研究。将这些模型整合到临床实践中可能彻底改变多种医疗条件的评估和诊断。 

---
# Exploring and Evaluating Multimodal Knowledge Reasoning Consistency of Multimodal Large Language Models 

**Title (ZH)**: 探索和评估多模态大型语言模型的多模态知识推理一致性 

**Authors**: Boyu Jia, Junzhe Zhang, Huixuan Zhang, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2503.04801)  

**Abstract**: In recent years, multimodal large language models (MLLMs) have achieved significant breakthroughs, enhancing understanding across text and vision. However, current MLLMs still face challenges in effectively integrating knowledge across these modalities during multimodal knowledge reasoning, leading to inconsistencies in reasoning outcomes. To systematically explore this issue, we propose four evaluation tasks and construct a new dataset. We conduct a series of experiments on this dataset to analyze and compare the extent of consistency degradation in multimodal knowledge reasoning within MLLMs. Based on the experimental results, we identify factors contributing to the observed degradation in consistency. Our research provides new insights into the challenges of multimodal knowledge reasoning and offers valuable guidance for future efforts aimed at improving MLLMs. 

**Abstract (ZH)**: 近年来，多模态大型语言模型（MLLMs）在文本和视觉理解方面取得了显著突破，但在多模态知识推理过程中仍面临有效整合知识的挑战，导致推理结果不一致。为系统探索这一问题，我们提出了四种评估任务并构建了一个新的数据集。在该数据集上进行了一系列实验，以分析和比较多模态知识推理中MLLMs一致性退化的程度。基于实验结果，我们识别出了导致一致性退化的因素。本研究为多模态知识推理面临的挑战提供了新的见解，并为未来改进MLLMs的努力提供了有价值的经验指导。 

---
# HoH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on Retrieval-Augmented Generation 

**Title (ZH)**: HoH：评估过时信息对检索增强生成影响的动态基准 

**Authors**: Jie Ouyang, Tingyue Pan, Mingyue Cheng, Ruiran Yan, Yucong Luo, Jiaying Lin, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04800)  

**Abstract**: While Retrieval-Augmented Generation (RAG) has emerged as an effective approach for addressing the knowledge outdating problem in Large Language Models (LLMs), it faces a critical challenge: the prevalence of outdated information in knowledge bases. Current research primarily focuses on incorporating up-to-date information, yet the impact of outdated information coexisting in retrieval sources remains inadequately addressed. To bridge this gap, we introduce HoH, the first benchmark specifically designed to evaluate the impact of outdated information on RAG. Our benchmark leverages token-level diff algorithms combined with LLM pipelines to efficiently create a large-scale QA dataset that accurately captures temporal knowledge evolution in real-world facts. Through comprehensive experiments, we reveal that outdated information significantly degrades RAG performance in two critical ways: (1) it substantially reduces response accuracy by distracting models from correct information, and (2) it can mislead models into generating potentially harmful outputs, even when current information is available. Current RAG approaches struggle with both retrieval and generation aspects when handling outdated information. These findings highlight the urgent need for innovative solutions to address the temporal challenges in RAG. 

**Abstract (ZH)**: HOH：评估检索增强生成中过时信息影响的第一个基准 

---
# Advancing MAPF towards the Real World: A Scalable Multi-Agent Realistic Testbed (SMART) 

**Title (ZH)**: 面向现实世界的路径规划：大规模多Agent实际测试平台（SMART） 

**Authors**: Jingtian Yan, Zhifei Li, William Kang, Yulun Zhang, Stephen Smith, Jiaoyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.04798)  

**Abstract**: We present Scalable Multi-Agent Realistic Testbed (SMART), a realistic and efficient software tool for evaluating Multi-Agent Path Finding (MAPF) algorithms. MAPF focuses on planning collision-free paths for a group of agents. While state-of-the-art MAPF algorithms can plan paths for hundreds of robots in seconds, they often rely on simplified robot models, making their real-world performance unclear. Researchers typically lack access to hundreds of physical robots in laboratory settings to evaluate the algorithms. Meanwhile, industrial professionals who lack expertise in MAPF require an easy-to-use simulator to efficiently test and understand the performance of MAPF algorithms in their specific settings. SMART fills this gap with several advantages: (1) SMART uses a physics-engine-based simulator to create realistic simulation environments, accounting for complex real-world factors such as robot kinodynamics and execution uncertainties, (2) SMART uses an execution monitor framework based on the Action Dependency Graph, facilitating seamless integration with various MAPF algorithms and robot models, and (3) SMART scales to thousands of robots. In addition, we use SMART to explore and demonstrate research questions about the execution of MAPF algorithms in real-world scenarios. The code is publicly available at this https URL. 

**Abstract (ZH)**: 面向多Agent路径规划算法评估的可扩展多Agent现实测试床（SMART） 

---
# Optimizing Multi-Hop Document Retrieval Through Intermediate Representations 

**Title (ZH)**: 通过中间表示优化多跳文档检索 

**Authors**: Jiaen Lin, Jingyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04796)  

**Abstract**: Retrieval-augmented generation (RAG) encounters challenges when addressing complex queries, particularly multi-hop questions. While several methods tackle multi-hop queries by iteratively generating internal queries and retrieving external documents, these approaches are computationally expensive. In this paper, we identify a three-stage information processing pattern in LLMs during layer-by-layer reasoning, consisting of extraction, processing, and subsequent extraction steps. This observation suggests that the representations in intermediate layers contain richer information compared to those in other layers. Building on this insight, we propose Layer-wise RAG (L-RAG). Unlike prior methods that focus on generating new internal queries, L-RAG leverages intermediate representations from the middle layers, which capture next-hop information, to retrieve external knowledge. L-RAG achieves performance comparable to multi-step approaches while maintaining inference overhead similar to that of standard RAG. Experimental results show that L-RAG outperforms existing RAG methods on open-domain multi-hop question-answering datasets, including MuSiQue, HotpotQA, and 2WikiMultiHopQA. The code is available in this https URL 

**Abstract (ZH)**: 检索增强生成（RAG）在处理复杂查询，特别是多跳问题时遇到挑战。虽然有一些方法通过迭代生成内部查询并检索外部文档来应对多跳查询，但这些方法计算成本较高。在本文中，我们发现在逐层推理过程中LLMs的信息处理模式呈现出三层结构，包括提取、处理和后续提取步骤。这一观察表明，中间层的表示比其他层的表示包含更多的信息。基于这一洞察，我们提出了分层RAG（L-RAG）。与之前专注于生成新的内部查询的方法不同，L-RAG利用中间层捕获下一跳信息的表示来进行外部知识检索。L-RAG在多步方法可比的性能下，保持了与标准RAG类似的推理开销。实验结果表明，L-RAG在MuSiQue、HotpotQA和2WikiMultiHopQA等开放式多跳问答数据集上的表现优于现有RAG方法。代码可在以下链接获取。 

---
# Cyber for AI at SemEval-2025 Task 4: Forgotten but Not Lost: The Balancing Act of Selective Unlearning in Large Language Models 

**Title (ZH)**: Cyber for AI at SemEval-2025 Task 4: 忘记但未消失：大型语言模型中选择性遗忘的平衡艺术 

**Authors**: Dinesh Srivasthav P, Bala Mallikarjunarao Garlapati  

**Link**: [PDF](https://arxiv.org/pdf/2503.04795)  

**Abstract**: Large Language Models (LLMs) face significant challenges in maintaining privacy, ethics, and compliance, when sensitive or obsolete data must be selectively removed. Retraining these models from scratch is computationally infeasible, necessitating efficient alternatives. As part of the SemEval 2025 Task 4, this work focuses on the application of selective unlearning in LLMs to address this challenge. In this paper, we present our experiments and findings, primarily leveraging global weight modification to achieve an equilibrium between effectiveness of unlearning, knowledge retention, and target model's post-unlearning utility. We also detail the task-specific evaluation mechanism, results, and challenges. Our algorithms have achieved an aggregate score of 0.409 and 0.389 on the test set for 7B and 1B target models, respectively, demonstrating promising results in verifiable LLM unlearning. 

**Abstract (ZH)**: 大型语言模型（LLMs）在处理敏感或过时数据的有选择性删除时面临着维护隐私、伦理和合规性的重大挑战。从头重新训练这些模型是计算上不可行的，因此需要高效的替代方案。作为SemEval 2025 Task 4的一部分，本工作聚焦于在LLMs中应用有选择性遗忘以应对这一挑战。在本文中，我们介绍了我们的实验和发现，主要通过全局权重修改来平衡遗忘效果、知识保留以及目标模型删除后的实用性。我们还详细描述了任务特定的评估机制、结果及挑战。我们的算法在7B和1B目标模型的测试集上分别获得了0.409和0.389的综合得分，展示了在可验证的LLM遗忘方面的有希望的结果。 

---
# Cross-linguistic disagreement as a conflict of semantic alignment norms in multilingual AI~Linguistic Diversity as a Problem for Philosophy, Cognitive Science, and AI~ 

**Title (ZH)**: 跨语言分歧作为多语言AI中语义对齐规范的冲突问题 Linguistic多样性对哲学、认知科学和AI的挑战 

**Authors**: Masaharu Mizumoto, Dat Tien Nguyen, Justin Sytsma, Mark Alfano, Yu Izumi, Koji Fujita, Nguyen Le Minh  

**Link**: [PDF](https://arxiv.org/pdf/2503.04792)  

**Abstract**: Multilingual large language models (LLMs) face an often-overlooked challenge stemming from intrinsic semantic differences across languages. Linguistic divergence can sometimes lead to cross-linguistic disagreements--disagreements purely due to semantic differences about a relevant concept. This paper identifies such disagreements as conflicts between two fundamental alignment norms in multilingual LLMs: cross-linguistic consistency (CL-consistency), which seeks universal concepts across languages, and consistency with folk judgments (Folk-consistency), which respects language-specific semantic norms. Through examining responses of conversational multilingual AIs in English and Japanese with the cases used in philosophy (cases of knowledge-how attributions), this study demonstrates that even state-of-the-art LLMs provide divergent and internally inconsistent responses. Such findings reveal a novel qualitative limitation in crosslingual knowledge transfer, or conceptual crosslingual knowledge barriers, challenging the assumption that universal representations and cross-linguistic transfer capabilities are inherently desirable. Moreover, they reveal conflicts of alignment policies of their developers, highlighting critical normative questions for LLM researchers and developers. The implications extend beyond technical alignment challenges, raising normative, moral-political, and metaphysical questions about the ideals underlying AI development--questions that are shared with philosophers and cognitive scientists but for which no one yet has definitive answers, inviting a multidisciplinary approach to balance the practical benefits of cross-linguistic consistency and respect for linguistic diversity. 

**Abstract (ZH)**: 多语言大型语言模型面临的内生语义差异引发的Often-Overlooked挑战：跨语言分歧与概念跨语言壁垒 

---
# SuperRAG: Beyond RAG with Layout-Aware Graph Modeling 

**Title (ZH)**: SuperRAG：具有布局意识的图建模超越RAG 

**Authors**: Jeff Yang, Duy-Khanh Vu, Minh-Tien Nguyen, Xuan-Quang Nguyen, Linh Nguyen, Hung Le  

**Link**: [PDF](https://arxiv.org/pdf/2503.04790)  

**Abstract**: This paper introduces layout-aware graph modeling for multimodal RAG. Different from traditional RAG methods that mostly deal with flat text chunks, the proposed method takes into account the relationship of multimodalities by using a graph structure. To do that, a graph modeling structure is defined based on document layout parsing. The structure of an input document is retained with the connection of text chunks, tables, and figures. This representation allows the method to handle complex questions that require information from multimodalities. To confirm the efficiency of the graph modeling, a flexible RAG pipeline is developed using robust components. Experimental results on four benchmark test sets confirm the contribution of the layout-aware modeling for performance improvement of the RAG pipeline. 

**Abstract (ZH)**: 基于布局感知的图建模在多模态RAG中的应用 

---
# Ext2Gen: Alignment through Unified Extraction and Generation for Robust Retrieval-Augmented Generation 

**Title (ZH)**: Ext2Gen：通过统一提取与生成进行稳健的检索增强生成对齐 

**Authors**: Hwanjun Song, Jeonghwan Choi, Minseok Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.04789)  

**Abstract**: Retrieval-augmented generation (RAG) enhances LLMs by integrating external knowledge, but generation remains fragile due to the uncertain placement of relevant chunks and retrieval-induced information overload, leading to hallucinations. We propose Ext2Gen, a novel extract-then-generate model that enhances RAG robustness by first extracting query-relevant sentences before generating answers. To optimize this model, we employ preference alignment through pairwise feedback learning, enabling the model to generate robust answers regardless of variations in retrieval results. Extensive experiments demonstrate that Ext2Gen effectively identifies query-relevant sentences with high precision and recall, leading to highly reliable answers. Furthermore, deploying our model in a RAG environment reveals that it not only boosts the performance of the base LLM but also synergizes with advanced retrieval strategies like query expansion. The dataset and model will be released soon. 

**Abstract (ZH)**: retrieve-然后生成增强（Ext2Gen）通过首先提取查询相关句子以增强RAG的稳定性，进而生成稳健的答案 

---
# AgroLLM: Connecting Farmers and Agricultural Practices through Large Language Models for Enhanced Knowledge Transfer and Practical Application 

**Title (ZH)**: 农林LLM：通过大型语言模型连接农民与农业实践以增强知识转移和实际应用 

**Authors**: Dinesh Jackson Samuel, Inna Skarga-Bandurova, David Sikolia, Muhammad Awais  

**Link**: [PDF](https://arxiv.org/pdf/2503.04788)  

**Abstract**: AgroLLM is an AI-powered chatbot designed to enhance knowledge-sharing and education in agriculture using Large Language Models (LLMs) and a Retrieval-Augmented Generation (RAG) framework. By using a comprehensive open-source agricultural database, AgroLLM provides accurate, contextually relevant responses while reducing incorrect information retrieval. The system utilizes the FAISS vector database for efficient similarity searches, ensuring rapid access to agricultural knowledge. A comparative study of three advanced models: Gemini 1.5 Flash, ChatGPT-4o Mini, and Mistral-7B-Instruct-v0.2 was conducted to evaluate performance across four key agricultural domains: Agriculture and Life Sciences, Agricultural Management, Agriculture and Forestry, and Agriculture Business. Key evaluation metrics included embedding quality, search efficiency, and response relevance. Results indicated that ChatGPT-4o Mini with RAG achieved the highest accuracy at 93%. Continuous feedback mechanisms enhance response quality, making AgroLLM a benchmark AI-driven educational tool for farmers, researchers, and professionals, promoting informed decision-making and improved agricultural practices. 

**Abstract (ZH)**: AgroLLM是一种利用大规模语言模型（LLMs）和检索增强生成（RAG）框架的AI驱动聊天机器人，旨在通过综合开源农业数据库增强农业领域的知识共享和教育。该系统利用FAISS向量数据库进行高效的相似搜索，确保快速访问农业知识。研究了三个先进模型：Gemini 1.5 Flash、ChatGPT-4o Mini和Mistral-7B-Instruct-v0.2在农业和生命科学、农业管理、农业和林业、农业商业四大关键农业领域中的性能。主要评估指标包括嵌入质量、搜索效率和响应相关性。结果显示，配备RAG的ChatGPT-4o Mini在准确性方面最高，达到93%。持续的反馈机制提高了响应质量，使AgroLLM成为农民、研究人员和专业人士的基准AI驱动教育工具，促进基于信息的决策和农业实践的改进。 

---
# Towards Anthropomorphic Conversational AI Part I: A Practical Framework 

**Title (ZH)**: Towards 类人对话人工智能 第一部分：一个实用框架 

**Authors**: Fei Wei, Yaliang Li, Bolin Ding  

**Link**: [PDF](https://arxiv.org/pdf/2503.04787)  

**Abstract**: Large language models (LLMs), due to their advanced natural language capabilities, have seen significant success in applications where the user interface is usually a conversational artificial intelligence (AI) agent and engages the user through multi-round conversations. However, many scenarios require the agents to exhibit stronger social and conversational intelligence and demonstrate more human-like (anthropomorphic) reactions. This is an aspect that foundational LLMs have yet to fully address such that a single call of foundational models might be insufficient.
To bridge this gap, we propose a two-stage solution. In this work, we focus on the first stage, introducing a multi-module framework designed to replicate the key aspects of human intelligence involved in conversations. This framework comprises thinking modules for reasoning, resource modules for managing knowledge and external information, and response modules for generating contextually appropriate interactions. With all the modules cooperating, the framework would empower the agents to provide a better human-like conversation experience. In the second stage of our approach, these conversational data, after filtering and labeling, can serve as training and testing data for reinforcement learning, enabling AI to better capture human preferences. This stage is left for future work.
In our experiments, volunteers engaged in over 3000 rounds of conversation with the same AI character powered by a standalone LLM and our framework which integrates the same LLM. A separate group of evaluators rated the conversation samples, revealing that our framework significantly enhanced the social and conversational intelligence, even without fine-tuning the LLM. 

**Abstract (ZH)**: 大规模语言模型（LLMs）由于其先进的自然语言能力，在通常由会话人工智能（AI）代理器用户界面和通过多轮对话与用户交互的应用场景中取得了显著成功。然而，许多场景需要代理展现出更强的社会智能和交谈智慧，并表现出更多类似人类（拟人化）的反应。这是基础LLMs尚未充分解决的一个方面，因此基础模型单次调用可能不足以满足需求。

为此，我们提出了一种两阶段解决方案。在本文中，我们关注第一阶段，引入了一个多模块框架，旨在模仿对话中涉及的关键方面的人类智能。该框架包括推理模块、资源管理模块和响应模块，分别用于推理、管理和处理知识以及外部信息，以及生成上下文相关交互。通过所有模块的协同工作，该框架将赋予代理更好的拟人化对话体验。在我们方法的第二阶段，经过筛选和标注的对话数据可以作为强化学习的训练和测试数据，使AI更好地捕捉人类偏好。此阶段留待未来工作。

在我们的实验中，志愿者与由独立运行的LLM和我们结合在同一LLM的框架驱动的同一AI角色进行了超过3000轮对话。另外一组评估人员对对话样本进行了评估，结果显示我们的框架显著提高了社会智能和交谈智慧，即使没有对LLM进行微调。 

---
# KunlunBaize: LLM with Multi-Scale Convolution and Multi-Token Prediction Under TransformerX Framework 

**Title (ZH)**: KunlunBaize：在TransformerX框架下的多尺度卷积与多令牌预测大型语言模型 

**Authors**: Jiexiong Liu, Yixuan Chen, Yanqin Jia, Zhepeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.04784)  

**Abstract**: Large language models have demonstrated remarkable performance across various tasks, yet they face challenges such as low computational efficiency, gradient vanishing, and difficulties in capturing complex feature interactions. To address these limitations, a novel framework has been proposed. This framework incorporates a learnable dense residual skip connection mechanism, a TransformerX module a transformer based component integrating multiscale convolution and adaptive activation functions and a multitoken prediction interaction module. The learnable dense residual connections enhance information flow and feature capture across layers. Within the TransformerX module, large convolutional kernels aggregate semantic information from extensive text segments, while smaller convolutions focus on local word order and syntactic structures. The adaptive activation function dynamically adjusts its parameters based on the semantic features of the input text, improving the model's ability to handle diverse semantic expressions and complex relationships. The multitoken prediction module boosts data utilization and accelerates inference by predicting multiple future tokens. These components significantly enhance the performance and efficiency of large language models. 

**Abstract (ZH)**: 大型语言模型在各种任务中展现了出色的表现，但仍面临计算效率低、梯度消失以及捕捉复杂特征交互的难题。为解决这些问题，提出了一种新型框架。该框架包含可学习的密集_residual跳连机制、TransformerX模块（基于Transformer的部件，集成了多尺度卷积和自适应激活函数）以及多令牌预测交互模块。可学习的密集_residual跳连机制增强了各层间的信息流动和特征捕获。TransformerX模块通过大卷积核聚合大量文本段落的语义信息，而小卷积核则关注局部词序和句法结构。自适应激活函数根据输入文本的语义特征动态调整参数，提高了模型处理多种语义表达和复杂关系的能力。多令牌预测模块通过预测多个未来令牌来增强数据利用和加速推理。这些组件显著提升了大型语言模型的性能和效率。 

---
# SMT(LIA) Sampling with High Diversity 

**Title (ZH)**: SMT（LIA）采样与高多样性 

**Authors**: Yong Lai, Junjie Li, Chuan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2503.04782)  

**Abstract**: Satisfiability Modulo Linear Integer Arithmetic, SMT(LIA) for short, is pivotal across various critical domains. Previous research has primarily focused on SMT solving techniques. However, in practical applications such as software and hardware testing, there is a need to generate a diverse set of solutions for use as test inputs. We have developed the first sampling framework that integrates local search with CDCL(T) techniques, named HighDiv, capable of generating a highly diverse set of solutions for constraints under linear integer theory. Initially, in the local search phase, we introduced a novel operator called boundary-aware movement. This operator performs random moves by considering the current state's constraints on variables, thereby enhancing the diversity of variables during the search process. Furthermore, we have conducted an in-depth study of the preprocessing and variable initialization mechanisms within the framework, which significantly enhances the efficiency of subsequent local searches. Lastly, we use the solutions obtained from local search sampling as additional constraints to further explore the solution space using the stochastic CDCL(T) method. Experimental results demonstrate that \HighDiv generates solutions with greater diversity compared to the state-of-the-art SMT(LIA) sampling tool, MeGASampler. 

**Abstract (ZH)**: 模线性整数算术 satisfiability modulo linear integer arithmetic, SMT(LIA)，简称 SMT(LIA)，在多个关键领域中至关重要。以往的研究主要关注 SMT 求解技术。然而，在软件和硬件测试等实用应用中，需要生成多样性较强的测试输入。我们开发了首个结合局部搜索与 CDCL(T) 技术的采样框架 HighDiv，能够在线性整数理论约束下生成高度多样化的解集。在局部搜索阶段，我们引入了一种新的操作符称为边界感知移动，该操作符通过考虑当前状态对变量的约束来进行随机移动，从而在搜索过程中增强变量的多样性。此外，我们在框架中深入研究了预处理和变量初始化机制，显著提高了后续局部搜索的效率。最后，我们使用局部搜索采样获得的解作为额外约束，进一步使用随机 CDCL(T) 方法探索解空间。实验结果表明，HighDiv 生成的解集比最先进的 SMT(LIA) 采样工具 MeGASampler 更具多样性。 

---
# MV-CLAM: Multi-View Molecular Interpretation with Cross-Modal Projection via Language Model 

**Title (ZH)**: MV-CLAM：通过语言模型实现跨模态投影的多视图分子解释 

**Authors**: Sumin Ha, Jun Hyeong Kim, Yinhua Piao, Sun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.04780)  

**Abstract**: Human expertise in chemistry and biomedicine relies on contextual molecular understanding, a capability that large language models (LLMs) can extend through fine-grained alignment between molecular structures and text. Recent multimodal learning advances focus on cross-modal alignment, but existing molecule-text models ignore complementary information in different molecular views and rely on single-view representations, limiting molecular understanding. Moreover, naïve multi-view alignment strategies face two challenges: (1) separate aligned spaces with inconsistent mappings between molecule and text embeddings, and that (2) existing loss objectives fail to preserve complementary information for fine-grained alignment. This can limit the LLM's ability to fully understand the molecular properties. To address these issues, we propose MV-CLAM, a novel framework that aligns multi-view molecular representations into a unified textual space using a multi-query transformer (MQ-Former). Our approach ensures cross-view consistency while a token-level contrastive loss preserves diverse molecular features across textual queries. MV-CLAM enhances molecular reasoning, improving retrieval and captioning accuracy. The source code of MV-CLAM is available in this https URL. 

**Abstract (ZH)**: 多视图CLAM：一种将多视图分子表示统一到文本空间的新型框架 

---
# Can LLMs Reason About Program Semantics? A Comprehensive Evaluation of LLMs on Formal Specification Inference 

**Title (ZH)**: LLMs在形式化规范推断方面的推理能力探究：一项全面评估 

**Authors**: Thanh Le-Cong, Bach Le, Toby Murray  

**Link**: [PDF](https://arxiv.org/pdf/2503.04779)  

**Abstract**: Large Language Models (LLMs) are increasingly being used to automate programming tasks. Yet, LLMs' capabilities in reasoning about program semantics are still inadequately studied, leaving significant potential for further exploration. This paper introduces FormalBench, a comprehensive benchmark designed to evaluate LLMs' reasoning abilities on program semantics, particularly via the task of synthesizing formal program specifications to assist verifying program correctness. This task requires both comprehensive reasoning over all possible program executions (i.e., \textit{completeness}) and the generation of precise, syntactically correct expressions that adhere to formal syntax and semantics (i.e., \textit{consistency}). Using this benchmark, we evaluated the ability of LLMs in synthesizing consistent and complete specifications. Our findings show that LLMs perform well with simple control flows but struggle with more complex structures, especially loops, even with advanced prompting. Additionally, LLMs exhibit limited robustness against semantic-preserving transformations. We also highlight common failure patterns and design self-repair prompts, improving success rates by 25%. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地被用于自动化编程任务。然而，LLMs在程序语义推理方面的能力仍然研究不足，留下了进一步探索的巨大潜力。本文介绍了FormalBench，这是一个全面的基准，旨在评估LLMs在程序语义推理方面的能力，特别是通过合成正式的程序规范以辅助验证程序正确性的任务。这一任务要求对所有可能的程序执行进行全面推理（即完备性）和生成精确且符合正式语法和语义的表达式（即一致性）。利用这一基准，我们评估了LLMs在合成一致且完整的规范方面的能力。我们的研究结果表明，LLMs在简单的控制流方面表现良好，但在更复杂的结构，尤其是循环结构方面表现较差，即使使用高级提示也是如此。此外，LLMs对语义保留转换的鲁棒性有限。我们还指出了常见的失败模式，并设计了自我修复提示，将成功率提高了25%。 

---
# Generating Millions Of Lean Theorems With Proofs By Exploring State Transition Graphs 

**Title (ZH)**: 通过探索状态转换图生成数百万条精简定理及其证明 

**Authors**: David Yin, Jing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04772)  

**Abstract**: Large Language Models (LLMs) have demonstrated significant potential in generating mathematical proofs. However, a persistent challenge is that LLMs occasionally make mistakes, while even a minor mistake can invalidate an entire proof. Proof assistants like Lean offer a great remedy. They are designed for verifying each step of a proof in a formal language, and in recent years researchers have created AI models to generate proofs in their languages. However, the scarcity of large-scale datasets of Lean proofs restrict the performance of such Automated Theorem Proving (ATP) models.
We developed LeanNavigator, a novel method for generating a large-scale dataset of Lean theorems and proofs by finding new ways to prove existing Lean theorems. By leveraging an interactive Lean client and an efficient method for proof step generation, LeanNavigator efficiently produces new theorems with corresponding proofs. Applying this approach to Mathlib4, we generated 4.7 million theorems totaling 1 billion tokens, surpassing previous datasets by more than an order of magnitude. Using this extensive dataset, we trained an AI model that outperforms the state-of-the-art ReProver model in theorem-proving tasks. These results confirm our hypothesis and demonstrate the critical role of large datasets in improving the performance of automated theorem provers. 

**Abstract (ZH)**: 大型语言模型（LLMs）在生成数学证明方面展现了显著的潜力。然而，一个持续性的挑战是LLMs偶尔会出现错误，即使是一个小错误也可能使整个证明无效。像Lean这样的证明助手提供了很好的解决办法。它们旨在以形式语言验证每一个证明步骤，并且近年来研究人员已经创建了能够生成它们语言中证明的AI模型。然而，Lean证明的大规模数据集稀缺限制了此类自动化定理证明（ATP）模型的表现。我们开发了LeanNavigator，这是一种新颖的方法，通过寻找证明现有Lean定理的新途径来生成大规模的Lean定理和证明数据集。通过利用交互式Lean客户端和高效的证明步骤生成方法，LeanNavigator高效地产生了新的定理及其对应的证明。将这种方法应用于Mathlib4，我们生成了总计1亿个词元的470万个定理，大大超过了之前的数据集的规模。利用这个庞大的数据集，我们训练了一个AI模型，其在定理证明任务上超过了最先进的ReProver模型。这些结果证实了我们的假设，并展示了大规模数据集在提高自动化定理证明性能中的关键作用。 

---
# A cross-regional review of AI safety regulations in the commercial aviation 

**Title (ZH)**: 跨区域商业航空AI安全监管综述 

**Authors**: Penny A. Barr, Sohel M. Imroz  

**Link**: [PDF](https://arxiv.org/pdf/2503.04767)  

**Abstract**: In this paper we examine the existing artificial intelligence (AI) policy documents in aviation for the following three regions: the United States, European Union, and China. The aviation industry has always been a first mover in adopting technological advancements. This early adoption offers valuable insights because of its stringent regulations and safety-critical procedures. As a result, the aviation industry provides an optimal platform to counter AI vulnerabilities through its tight regulations, standardization processes, and certification of new technologies. Keywords: AI in aviation; aviation safety; standardization; certifiable AI; regulations 

**Abstract (ZH)**: 本文分析了航空领域在美国、欧洲联盟和中国现有的人工智能（AI）政策文件。航空业一直是采用科技革新先驱者，其严格的法规和安全关键程序为研究AI漏洞提供了宝贵见解。因此，航空业提供了通过严格法规、标准化流程和新技术认证来应对AI漏洞的理想平台。关键词：航空人工智能；航空安全；标准化；可认证AI；法规。 

---
# Which Economic Tasks are Performed with AI? Evidence from Millions of Claude Conversations 

**Title (ZH)**: 哪些经济任务是由AI执行的？来自数百万次Claude对话的证据 

**Authors**: Kunal Handa, Alex Tamkin, Miles McCain, Saffron Huang, Esin Durmus, Sarah Heck, Jared Mueller, Jerry Hong, Stuart Ritchie, Tim Belonax, Kevin K. Troy, Dario Amodei, Jared Kaplan, Jack Clark, Deep Ganguli  

**Link**: [PDF](https://arxiv.org/pdf/2503.04761)  

**Abstract**: Despite widespread speculation about artificial intelligence's impact on the future of work, we lack systematic empirical evidence about how these systems are actually being used for different tasks. Here, we present a novel framework for measuring AI usage patterns across the economy. We leverage a recent privacy-preserving system to analyze over four million this http URL conversations through the lens of tasks and occupations in the U.S. Department of Labor's O*NET Database. Our analysis reveals that AI usage primarily concentrates in software development and writing tasks, which together account for nearly half of all total usage. However, usage of AI extends more broadly across the economy, with approximately 36% of occupations using AI for at least a quarter of their associated tasks. We also analyze how AI is being used for tasks, finding 57% of usage suggests augmentation of human capabilities (e.g., learning or iterating on an output) while 43% suggests automation (e.g., fulfilling a request with minimal human involvement). While our data and methods face important limitations and only paint a picture of AI usage on a single platform, they provide an automated, granular approach for tracking AI's evolving role in the economy and identifying leading indicators of future impact as these technologies continue to advance. 

**Abstract (ZH)**: 尽管人们对人工智能对未来工作的影响进行了广泛猜测，但我们缺乏系统性实证证据来说明这些系统在不同任务中的实际使用模式。在此，我们提出了一种衡量经济领域中人工智能使用模式的新框架。我们利用一个近期的隐私保护系统，分析了超过400万条this http URL对话，从美国劳工部的O*NET数据库中的任务和职业视角进行分析。我们的分析揭示，人工智能的使用主要集中在软件开发和写作任务上，这两大类任务占据了总体使用量的近一半。然而，人工智能的使用更广泛地延伸到整个经济领域，大约有36%的职业在其相关任务中至少使用了四分之一的人工智能。我们还分析了人工智能在任务中的使用情况，发现57%的使用表明了对人类能力的增强（例如学习或改进输出），而43%的使用表明了自动化（例如在最少人类干预的情况下完成请求）。尽管我们的数据和方法存在重要限制，只能描绘单个平台中人工智能使用状况的图景，但它们提供了自动化的、详细的追踪人工智能在经济中 evolving 角色的方法，并有助于识别这些技术继续发展时未来影响的领先指标。 

---
# Agentic AI and the Cyber Arms Race 

**Title (ZH)**: 代理人工智能与网络军备竞赛 

**Authors**: Sean Oesch, Jack Hutchins, Phillipe Austria, Amul Chaulagain  

**Link**: [PDF](https://arxiv.org/pdf/2503.04760)  

**Abstract**: Agentic AI is shifting the cybersecurity landscape as attackers and defenders leverage AI agents to augment humans and automate common tasks. In this article, we examine the implications for cyber warfare and global politics as Agentic AI becomes more powerful and enables the broad proliferation of capabilities only available to the most well resourced actors today. 

**Abstract (ZH)**: 行为AI正在改变网络安全格局，随着攻击者和防御者利用AI代理增强人类并自动化常见任务，行为AI变得更加强大并使当今仅能由资源最雄厚的行为主体获取的能力得以广泛普及，由此对网络战争和全球政治产生了影响。 

---
# Chat-GPT: An AI Based Educational Revolution 

**Title (ZH)**: Chat-GPT：基于AI的教育革命 

**Authors**: Sasa Maric, Sonja Maric, Lana Maric  

**Link**: [PDF](https://arxiv.org/pdf/2503.04758)  

**Abstract**: The AI revolution is gathering momentum at an unprecedented rate. Over the past decade, we have witnessed a seemingly inevitable integration of AI in every facet of our lives. Much has been written about the potential revolutionary impact of AI in education. AI has the potential to completely revolutionise the educational landscape as we could see entire courses and degrees developed by programs such as ChatGPT. AI has the potential to develop courses, set assignments, grade and provide feedback to students much faster than a team of teachers. In addition, because of its dynamic nature, it has the potential to continuously improve its content. In certain fields such as computer science, where technology is continuously evolving, AI based applications can provide dynamically changing, relevant material to students. AI has the potential to replace entire degrees and may challenge the concept of higher education institutions. We could also see entire new disciplines emerge as a consequence of AI. This paper examines the practical impact of ChatGPT and why it is believed that its implementation is a critical step towards a new era of education. We investigate the impact that ChatGPT will have on learning, problem solving skills and cognitive ability of students. We examine the positives, negatives and many other aspects of AI and its applications throughout this paper. 

**Abstract (ZH)**: AI革命正在以前所未有的速度加速到来。过去十年间，我们见证了AI在我们生活方方面面的似乎不可避免的融合。关于AI在教育领域潜在革命性影响的讨论甚嚣尘上。AI有潜力彻底改变我们所熟知的教育面貌，如同ChatGPT这样的程序可能开发出全新的课程和学位。AI能够比教师团队更快地开发课程、布置作业、评分并提供反馈。此外，由于其动态特性，它有潜力不断改进其内容。在计算机科学等技术不断发展的领域，基于AI的应用程序可以向学生提供动态变化的相关材料。AI有潜力取代整个学位课程，并可能挑战高等教育机构的概念。AI还可能引发全新的学科领域。本文探讨ChatGPT的实际影响，以及为什么其实施被认为是迈向教育新纪元的关键步骤。我们将研究ChatGPT对学生学习、解决问题能力和认知能力的影响。在本文中，我们还将探讨AI及其应用的诸多积极面、消极面及其他诸多方面。 

---
# Peeking Behind Closed Doors: Risks of LLM Evaluation by Private Data Curators 

**Title (ZH)**: 窥视密室之内：基于私人数据策展人评估的大语言模型风险 

**Authors**: Hritik Bansal, Pratyush Maini  

**Link**: [PDF](https://arxiv.org/pdf/2503.04756)  

**Abstract**: The rapid advancement in building large language models (LLMs) has intensified competition among big-tech companies and AI startups. In this regard, model evaluations are critical for product and investment-related decision-making. While open evaluation sets like MMLU initially drove progress, concerns around data contamination and data bias have constantly questioned their reliability. As a result, it has led to the rise of private data curators who have begun conducting hidden evaluations with high-quality self-curated test prompts and their own expert annotators. In this paper, we argue that despite potential advantages in addressing contamination issues, private evaluations introduce inadvertent financial and evaluation risks. In particular, the key concerns include the potential conflict of interest arising from private data curators' business relationships with their clients (leading LLM firms). In addition, we highlight that the subjective preferences of private expert annotators will lead to inherent evaluation bias towards the models trained with the private curators' data. Overall, this paper lays the foundation for studying the risks of private evaluations that can lead to wide-ranging community discussions and policy changes. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的快速进展加剧了大-tech公司和AI初创企业的竞争。在这种背景下，模型评估对于产品和投资决策至关重要。虽然像MMLU这样的开放评估集最初促进了进展，但关于数据污染和数据偏见的担忧不断质疑其可靠性。因此，这导致了私有数据策划者的兴起，他们开始进行隐藏评估，并使用高质量的自策划测试提示和自己的专家注释者。在本文中，我们argue尽管私人评估可能在解决污染问题方面具有潜在优势，但它们引入了潜在的财务和评估风险。特别是，主要关注点包括私营数据策划者与其客户（领先LLM公司）的商业关系可能引发的利益冲突。此外，我们强调私营专家注释者的主观偏好将导致模型对私营策划者数据进行训练时固有的评估偏见。总体而言，本文为研究可能导致广泛社区讨论和政策变化的私人评估风险奠定了基础。 

---
# Transforming Student Evaluation with Adaptive Intelligence and Performance Analytics 

**Title (ZH)**: 适应性智能与绩效分析驱动的学生评价转型 

**Authors**: Pushpalatha K S, Abhishek Mangalur, Ketan Hegde, Chetan Badachi, Mohammad Aamir  

**Link**: [PDF](https://arxiv.org/pdf/2503.04752)  

**Abstract**: The development in Artificial Intelligence (AI) offers transformative potential for redefining student assessment methodologies. This paper aims to establish the idea of the advancement of Artificial Intelligence (AI) and its prospect in reshaping approaches to assessing students. It creates a system for the evaluation of students performance using Artificial intelligence, and particularly the Gemini API for the generation of questions, grading and report on the students performances. This is to facilitate easy use of the tools in creating, scheduling, and delivering assessments with minimal chances of cheating through options such as full screen and time limit. There are formats of questions in the system which comprises multiple choice, short answers and descriptive questions, developed by Gemini. The most conspicuous feature is the self-checking system whereby the user gets instant feedback for the correct score that each of the students would have scored instantly with explanations about wrong answers. Moreover, the platform has intelligent learning progressions where the user will be able to monitor his/her performances to be recommended a certain level of performance. It will allow students as well as educators to have real-time analytics and feedback on what they are good at and where they need to improve. Not only does it make the assessment easier, but it also improves the levels of accuracy in grading and effectively strengthens a data based learning process for students. 

**Abstract (ZH)**: 人工智能的发展为重塑学生评估方法提供了变革潜力。本文旨在探讨人工智能的进步及其在重塑学生评估方法方面的前景。该论文建立了一个使用人工智能评估学生表现的系统，特别是利用Gemini API生成问题、评分和报告学生表现。这将有助于通过全屏和时间限制等选项轻松创建、安排和交付评估，减少作弊的机会。该系统包括由Gemini开发的多种题型，如选择题、简答题和论述题。最显着的特征是即刻反馈系统，用户可以即时获得每个学生的正确得分反馈，并附有错误答案的解释。此外，该平台具有智能学习渐进性，用户可以监控自己的表现并得到推荐的一些绩效水平。这将使学生和教育者能够实时获取他们在哪些方面表现出色以及需要改进的地方，不仅使评估更容易，还提高了评分的准确性，有效地增强了基于数据的学习过程。 

---
# What is Ethical: AIHED Driving Humans or Human-Driven AIHED? A Conceptual Framework enabling the Ethos of AI-driven Higher education 

**Title (ZH)**: 什么是符合伦理的：AIHED驱动的人类还是人类驱动的AIHED？一个基于人工智能驱动高等教育 ethos 的概念框架 

**Authors**: Prashant Mahajan  

**Link**: [PDF](https://arxiv.org/pdf/2503.04751)  

**Abstract**: The rapid integration of Artificial Intelligence (AI) in Higher Education (HE) is transforming personalized learning, administrative automation, and decision-making. However, this progress presents a duality, as AI adoption also introduces ethical and institutional challenges, including algorithmic bias, data privacy risks, and governance inconsistencies. To address these concerns, this study introduces the Human-Driven AI in Higher Education (HD-AIHED) Framework, ensuring compliance with UNESCO and OECD ethical standards. This conceptual research employs a qualitative meta-synthesis approach, integrating qualitative and quantitative studies to identify patterns, contradictions, and gaps in AI adoption within HE. It reinterprets existing datasets through theoretical and ethical lenses to develop governance frameworks. The study applies a participatory integrated co-system, Phased Human Intelligence, SWOC analysis, and AI ethical review boards to assess AI readiness and governance strategies for universities and HE institutions. The HD-AIHED model bridges AI research gaps, addresses global real-time challenges, and provides tailored, scalable, and ethical strategies for diverse educational contexts. By emphasizing interdisciplinary collaboration among stakeholders, this study envisions AIHED as a transparent and equitable force for innovation. The HD-AIHED framework ensures AI acts as a collaborative and ethical enabler rather than a disruptive replacement for human intelligence while advocating for responsible AI implementation in HE. 

**Abstract (ZH)**: 人工智能在高等教育中的快速融合：人类驱动的AI框架（HD-AIHED） 

---
# Position: AI agents should be regulated based on autonomous action sequences 

**Title (ZH)**: 位置：基于自主行动序列的AI代理应予以规制。 

**Authors**: Takauki Osogami  

**Link**: [PDF](https://arxiv.org/pdf/2503.04750)  

**Abstract**: This position paper argues that AI agents should be regulated based on the sequence of actions they autonomously take. AI agents with long-term planning and strategic capabilities can pose significant risks of human extinction and irreversible global catastrophes. While existing regulations often focus on computational scale as a proxy for potential harm, we contend that such measures are insufficient for assessing the risks posed by AI agents whose capabilities arise primarily from inference-time computation. To support our position, we discuss relevant regulations and recommendations from AI scientists regarding existential risks, as well as the advantages of action sequences over existing impact measures that require observing environmental states. 

**Abstract (ZH)**: 基于自主决策序列的AIagent监管：论点论文 

---
# E-LENS: User Requirements-Oriented AI Ethics Assurance 

**Title (ZH)**: E-LENS: 用户需求导向的AI伦理保障 

**Authors**: Jianlong Zhou, Fang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.04747)  

**Abstract**: Despite the much proliferation of AI ethical principles in recent years, there is a challenge of assuring AI ethics with current AI ethics frameworks in real-world applications. While system safety has emerged as a distinct discipline for a long time, originated from safety concerns in early aircraft manufacturing. The safety assurance is now an indispensable component in safety critical domains. Motivated by the assurance approaches for safety-critical systems such as aviation, this paper introduces the concept of AI ethics assurance cases into the AI ethics assurance. Three pillars of user requirements, evidence, and validation are proposed as key components and integrated into AI ethics assurance cases for a new approach of user requirements-oriented AI ethics assurance. The user requirements-oriented AI ethics assurance case is set up based on three pillars and hazard analysis methods used in the safety assurance of safety-critical systems. This paper also proposes a platform named Ethical-Lens (E-LENS) to implement the user requirements-oriented AI ethics assurance approach. The proposed user requirements-based E-LENS platform is then applied to assure AI ethics of an AI-driven human resource shortlisting system as a case study to show the effectiveness of the proposed approach. 

**Abstract (ZH)**: 尽管近年来人工智能伦理原则得到了极大的普及，但在现实应用中确保人工智能伦理仍面临挑战。受到航空等关键安全领域安全保障方法的启发，本文将人工智能伦理保证概念引入人工智能伦理保证之中。提出了用户需求、证据和验证三大支柱作为关键组成部分，并整合至基于用户需求的人工智能伦理保证案例中，形成一种新的用户需求导向的人工智能伦理保证方法。该用户需求导向的人工智能伦理保证案例基于安全关键系统中的用户需求和危害分析方法建立。本文还提出一个名为Ethical-Lens (E-LENS) 的平台，用于实现基于用户需求的人工智能伦理保证方法，并将所提出的基于用户需求的E-LENS平台应用于一个人工智能驱动的人力资源筛选系统的伦理保证案例研究中，以展示所提方法的有效性。 

---
# Emerging Practices in Frontier AI Safety Frameworks 

**Title (ZH)**: 前沿AI安全框架中的新兴实践 

**Authors**: Marie Davidsen Buhl, Ben Bucknall, Tammy Masterson  

**Link**: [PDF](https://arxiv.org/pdf/2503.04746)  

**Abstract**: As part of the Frontier AI Safety Commitments agreed to at the 2024 AI Seoul Summit, many AI developers agreed to publish a safety framework outlining how they will manage potential severe risks associated with their systems. This paper summarises current thinking from companies, governments, and researchers on how to write an effective safety framework. We outline three core areas of a safety framework - risk identification and assessment, risk mitigation, and governance - and identify emerging practices within each area. As safety frameworks are novel and rapidly developing, we hope that this paper can serve both as an overview of work to date and as a starting point for further discussion and innovation. 

**Abstract (ZH)**: 关于2024 AI首尔峰会达成的前沿AI安全承诺，许多AI开发者同意发布一份安全框架，概述他们将如何管理其系统潜在的重大风险。本文总结了公司、政府和研究人员关于如何编写有效安全框架的当前观点。我们概述了安全框架的三个核心领域——风险识别与评估、风险减轻和治理——并确定了每个领域内的新兴实践。鉴于安全框架是新颖且快速发展的，我们希望本文既能作为迄今为止工作的综述，又能作为进一步讨论和创新的起点。 

---
# Safety Cases: A Scalable Approach to Frontier AI Safety 

**Title (ZH)**: 安全案例：通往前沿AI安全的可扩展方法 

**Authors**: Benjamin Hilton, Marie Davidsen Buhl, Tomek Korbak, Geoffrey Irving  

**Link**: [PDF](https://arxiv.org/pdf/2503.04744)  

**Abstract**: Safety cases - clear, assessable arguments for the safety of a system in a given context - are a widely-used technique across various industries for showing a decision-maker (e.g. boards, customers, third parties) that a system is safe. In this paper, we cover how and why frontier AI developers might also want to use safety cases. We then argue that writing and reviewing safety cases would substantially assist in the fulfilment of many of the Frontier AI Safety Commitments. Finally, we outline open research questions on the methodology, implementation, and technical details of safety cases. 

**Abstract (ZH)**: 安全论证——在一个给定的上下文中，清晰、可评估的安全论据是各行业中展示系统安全性给决策者（例如董事会、客户、第三方）的有效技术。本文探讨前沿AI开发者为何也应该使用安全论证，并论证编写和审核安全论证将如何有助于履行许多前沿AI安全承诺。最后，本文概述了安全论证方法论、实现和技术细节方面的开放研究问题。 

---
# AI Safety is Stuck in Technical Terms -- A System Safety Response to the International AI Safety Report 

**Title (ZH)**: AI安全困于技术术语——国际AI安全报告的系统安全回应 

**Authors**: Roel Dobbe  

**Link**: [PDF](https://arxiv.org/pdf/2503.04743)  

**Abstract**: Safety has become the central value around which dominant AI governance efforts are being shaped. Recently, this culminated in the publication of the International AI Safety Report, written by 96 experts of which 30 nominated by the Organisation for Economic Co-operation and Development (OECD), the European Union (EU), and the United Nations (UN). The report focuses on the safety risks of general-purpose AI and available technical mitigation approaches. In this response, informed by a system safety perspective, I refl ect on the key conclusions of the report, identifying fundamental issues in the currently dominant technical framing of AI safety and how this frustrates meaningful discourse and policy efforts to address safety comprehensively. The system safety discipline has dealt with the safety risks of software-based systems for many decades, and understands safety risks in AI systems as sociotechnical and requiring consideration of technical and non-technical factors and their interactions. The International AI Safety report does identify the need for system safety approaches. Lessons, concepts and methods from system safety indeed provide an important blueprint for overcoming current shortcomings in technical approaches by integrating rather than adding on non-technical factors and interventions. I conclude with why building a system safety discipline can help us overcome limitations in the European AI Act, as well as how the discipline can help shape sustainable investments into Public Interest AI. 

**Abstract (ZH)**: Safety已成为主导AI治理努力的中心价值。最近，这体现在国际AI安全报告的发布上，该报告由96位专家编写，其中30位分别由经济合作与发展组织(OECD)、欧盟(EU)和联合国(UN)提名。该报告专注于通用人工智能的安全风险以及可用的技术缓解方法。在此回应中，基于系统安全性视角，我反思报告的关键结论，指出现行主导的技术框架中关于AI安全的核心问题，以及这些问题如何阻碍全面安全的有意义讨论和政策制定。系统安全性学科已经处理基于软件的系统安全风险数十年，并将AI系统中的安全风险视为社会技术性的，需要考虑技术和非技术因素及其相互作用。国际AI安全报告确实指出了需要采取系统安全性方法。系统安全领域的教训、概念和方法确实为通过集成而非附加非技术因素和干预来克服当前技术方法的局限性提供了重要蓝图。最后，建立系统安全性学科如何帮助我们克服《欧洲AI法案》的局限性，以及该学科如何有助于塑造公共利益AI的可持续投资。 

---
# A case for specialisation in non-human entities 

**Title (ZH)**: 非人类实体专门化的论据 

**Authors**: El-Mahdi El-Mhamdi, Lê-Nguyên Hoang, Mariame Tighanimine  

**Link**: [PDF](https://arxiv.org/pdf/2503.04742)  

**Abstract**: With the rise of large multi-modal AI models, fuelled by recent interest in large language models (LLMs), the notion of artificial general intelligence (AGI) went from being restricted to a fringe community, to dominate mainstream large AI development programs.
In contrast, in this paper, we make a \emph{case for specialisation}, by reviewing the pitfalls of generality and stressing the industrial value of specialised
systems.
Our contribution is threefold. First, we review the most widely accepted arguments \emph{against} specialisation, and discuss how their relevance in the context of human labour is actually an argument \emph{for} specialisation in the case of non human agents, be they algorithms or human organisations. Second, we propose four arguments \emph{in favor of} specialisation, ranging from machine learning robustness, to computer security, social sciences and cultural evolution.
Third, we finally make a case for \emph{specification}, discuss how the machine learning approach to AI has so far failed to catch up with good practices from safety-engineering and formal verification of software, and discuss how some emerging good practices in machine learning help reduce this gap.
In particular, we justify the need for \emph{specified governance} for hard-to-specify systems. 

**Abstract (ZH)**: 大型多模态AI模型兴起背景下人工通用智能理念从边缘走向主流：论专业化的重要性及规范性 

---
# Which Information should the UK and US AISI share with an International Network of AISIs? Opportunities, Risks, and a Tentative Proposal 

**Title (ZH)**: 英国和美国AISI应与国际AISIs网络分享哪些信息？机遇、风险及初步建议 

**Authors**: Lara Thurnherr  

**Link**: [PDF](https://arxiv.org/pdf/2503.04741)  

**Abstract**: The UK AI Safety Institute (UK AISI) and its parallel organisation in the United States (US AISI) take up a unique position in the recently established International Network of AISIs. Both are in jurisdictions with frontier AI companies and are assuming leading roles in the international conversation on AI Safety. This paper argues that it is in the interest of both institutions to share specific categories of information with the International Network of AISIs, deliberately abstain from sharing others and carefully evaluate sharing some categories on a case by case basis, according to domestic priorities. The paper further proposes a provisional framework with which policymakers and researchers can distinguish between these three cases, taking into account the potential benefits and risks of sharing specific categories of information, ranging from pre-deployment evaluation results to evaluation standards. In an effort to further improve the research on AI policy relevant information sharing decisions, the paper emphasises the importance of continuously monitoring fluctuating factors influencing sharing decisions and a more in-depth analysis of specific policy relevant information categories and additional factors to consider in future research. 

**Abstract (ZH)**: 英国AI安全研究所（UK AISI）及其在美国的平行组织（US AISI）在新成立的国际AI安全研究所网络中占据独特位置。两者都在拥有前沿AI公司的管辖区域内，并在国际AI安全对话中承担领导角色。本文认为，这两所机构有必要向国际AI安全研究所网络分享特定类别信息，故意不分享其他类别信息，并根据国内优先事项就某些类别信息的具体案例进行谨慎评估。本文进一步提出了一种暂定框架，供政策制定者和研究人员区分这三种情况，同时考虑分享特定类别信息的潜在利益和风险，范围从部署前的评估结果到评估标准。为进一步改善与AI政策相关的信息共享决策的研究，本文强调持续监控影响共享决策的波动因素以及对特定政策相关信息类别和未来研究中需要考虑的额外因素进行更深入分析的重要性。 

---
# PRISM: Perspective Reasoning for Integrated Synthesis and Mediation as a Multi-Perspective Framework for AI Alignment 

**Title (ZH)**: PRISM: 多视角推理框架下的综合合成与调解以实现AI对齐 

**Authors**: Anthony Diamond  

**Link**: [PDF](https://arxiv.org/pdf/2503.04740)  

**Abstract**: In this work, we propose Perspective Reasoning for Integrated Synthesis and Mediation (PRISM), a multiple-perspective framework for addressing persistent challenges in AI alignment such as conflicting human values and specification gaming. Grounded in cognitive science and moral psychology, PRISM organizes moral concerns into seven "basis worldviews", each hypothesized to capture a distinct dimension of human moral cognition, ranging from survival-focused reflexes through higher-order integrative perspectives. It then applies a Pareto-inspired optimization scheme to reconcile competing priorities without reducing them to a single metric. Under the assumption of reliable context validation for robust use, the framework follows a structured workflow that elicits viewpoint-specific responses, synthesizes them into a balanced outcome, and mediates remaining conflicts in a transparent and iterative manner. By referencing layered approaches to moral cognition from cognitive science, moral psychology, and neuroscience, PRISM clarifies how different moral drives interact and systematically documents and mediates ethical tradeoffs. We illustrate its efficacy through real outputs produced by a working prototype, applying PRISM to classic alignment problems in domains such as public health policy, workplace automation, and education. By anchoring AI deliberation in these human vantage points, PRISM aims to bound interpretive leaps that might otherwise drift into non-human or machine-centric territory. We briefly outline future directions, including real-world deployments and formal verifications, while maintaining the core focus on multi-perspective synthesis and conflict mediation. 

**Abstract (ZH)**: 基于多视角的AI对齐综合与调解框架：认知科学与道德心理学视角下的观点推理（PRISM） 

---
# Responsible Artificial Intelligence Systems: A Roadmap to Society's Trust through Trustworthy AI, Auditability, Accountability, and Governance 

**Title (ZH)**: 负责任的智能系统：通过可信AI、可审计性、问责制和治理赢得社会信任的道路图谱 

**Authors**: Andrés Herrera-Poyatos, Javier Del Ser, Marcos López de Prado, Fei-Yue Wang, Enrique Herrera-Viedma, Francisco Herrera  

**Link**: [PDF](https://arxiv.org/pdf/2503.04739)  

**Abstract**: Artificial intelligence (AI) has matured as a technology, necessitating the development of responsibility frameworks that are fair, inclusive, trustworthy, safe and secure, transparent, and accountable. By establishing such frameworks, we can harness the full potential of AI while mitigating its risks, particularly in high-risk scenarios. This requires the design of responsible AI systems based on trustworthy AI technologies and ethical principles, with the aim of ensuring auditability and accountability throughout their design, development, and deployment, adhering to domain-specific regulations and standards.
This paper explores the concept of a responsible AI system from a holistic perspective, which encompasses four key dimensions: 1) regulatory context; 2) trustworthy AI technology along with standardization and assessments; 3) auditability and accountability; and 4) AI governance. The aim of this paper is double. First, we analyze and understand these four dimensions and their interconnections in the form of an analysis and overview. Second, the final goal of the paper is to propose a roadmap in the design of responsible AI systems, ensuring that they can gain society's trust. To achieve this trustworthiness, this paper also fosters interdisciplinary discussions on the ethical, legal, social, economic, and cultural aspects of AI from a global governance perspective. Last but not least, we also reflect on the current state and those aspects that need to be developed in the near future, as ten lessons learned. 

**Abstract (ZH)**: 人工智能的责任框架：公平、包容、可信、安全与隐私保护、透明及问责机制的研究 

---
# Copyright in AI-generated works: Lessons from recent developments in patent law 

**Title (ZH)**: AI生成作品的版权：近期专利法发展带来的启示 

**Authors**: Rita Matulionyte, Jyh-An Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.04738)  

**Abstract**: In Thaler v The Comptroller-General of Patents, Designs and Trade Marks (DABUS), Smith J. held that an AI owner can possibly claim patent ownership over an AI-generated invention based on their ownership and control of the AI system. This AI-owner approach reveals a new option to allocate property rights over AI-generated output. While this judgment was primarily about inventorship and ownership of AI-generated invention in patent law, it has important implications for copyright law. After analysing the weaknesses of applying existing judicial approaches to copyright ownership of AI-generated works, this paper examines whether the AI-owner approach is a better option for determining copyright ownership of AI-generated works. The paper argues that while contracts can be used to work around the AI-owner approach in scenarios where users want to commercially exploit the outputs, this approach still provides more certainty and less transaction costs for relevant parties than other approaches proposed so far. 

**Abstract (ZH)**: 在Thaler诉专利、外观设计及商标审查长一案（DABUS）中，史密斯法官裁定，AI的所有者可以基于其对AI系统的所有权和控制权，宣称对AI生成发明的专利所有权。这一AI所有者的方法揭示了一种新的分配AI生成输出财产权的方式。虽然这一判决主要涉及专利法中AI生成发明的发明者身份和所有权问题，但它对版权法具有重要意义。在分析现有司法方法应用于AI生成作品版权所有权的不足之后，本文探讨AI所有者方法是否是确定AI生成作品版权所有权的更好选择。本文认为，尽管在用户希望商业利用输出的情况下，合同可以绕过AI所有者方法，但该方法仍然为相关方提供了更多的确定性和更低的交易成本，这优于目前提出的所有其他方法。 

---
# Carelessness Detection using Performance Factor Analysis: A New Operationalization with Unexpectedly Different Relationship to Learning 

**Title (ZH)**: 使用绩效因素分析进行粗心检测：一个新的操作化及其与学习出乎意料的关系 

**Authors**: Jiayi Zhang, Ryan S. Baker, Namrata Srivastava, Jaclyn Ocumpaugh, Caitlin Mills, Bruce M. McLaren  

**Link**: [PDF](https://arxiv.org/pdf/2503.04737)  

**Abstract**: Detection of carelessness in digital learning platforms has relied on the contextual slip model, which leverages conditional probability and Bayesian Knowledge Tracing (BKT) to identify careless errors, where students make mistakes despite having the knowledge. However, this model cannot effectively assess carelessness in questions tagged with multiple skills due to the use of conditional probability. This limitation narrows the scope within which the model can be applied. Thus, we propose a novel model, the Beyond Knowledge Feature Carelessness (BKFC) model. The model detects careless errors using performance factor analysis (PFA) and behavioral features distilled from log data, controlling for knowledge when detecting carelessness. We applied the BKFC to detect carelessness in data from middle school students playing a learning game on decimal numbers and operations. We conducted analyses comparing the careless errors detected using contextual slip to the BKFC model. Unexpectedly, careless errors identified by these two approaches did not align. We found students' post-test performance was (corresponding to past results) positively associated with the carelessness detected using the contextual slip model, while negatively associated with the carelessness detected using the BKFC model. These results highlight the complexity of carelessness and underline a broader challenge in operationalizing carelessness and careless errors. 

**Abstract (ZH)**: 超越知识特征的疏忽检测模型（BKFC）：基于性能因素分析和行为特征识别疏忽错误 

---
# Standardizing Intelligence: Aligning Generative AI for Regulatory and Operational Compliance 

**Title (ZH)**: 标准化智能：对接监管与运营合规的生成式AI 

**Authors**: Joseph Marvin Imperial, Matthew D. Jones, Harish Tayyar Madabushi  

**Link**: [PDF](https://arxiv.org/pdf/2503.04736)  

**Abstract**: Technical standards, or simply standards, are established documented guidelines and rules that facilitate the interoperability, quality, and accuracy of systems and processes. In recent years, we have witnessed an emerging paradigm shift where the adoption of generative AI (GenAI) models has increased tremendously, spreading implementation interests across standard-driven industries, including engineering, legal, healthcare, and education. In this paper, we assess the criticality levels of different standards across domains and sectors and complement them by grading the current compliance capabilities of state-of-the-art GenAI models. To support the discussion, we outline possible challenges and opportunities with integrating GenAI for standard compliance tasks while also providing actionable recommendations for entities involved with developing and using standards. Overall, we argue that aligning GenAI with standards through computational methods can help strengthen regulatory and operational compliance. We anticipate this area of research will play a central role in the management, oversight, and trustworthiness of larger, more powerful GenAI-based systems in the near future. 

**Abstract (ZH)**: 技术标准是建立并记载的指南和规则，旨在促进系统和过程的互操作性、质量和准确性。近年来，我们见证了生成式人工智能（GenAI）模型采用程度的显著增加，其实施兴趣遍及由标准驱动的行业，包括工程、法律、医疗保健和教育。本文评估了不同领域和部门中标准的关键性级别，并通过评价当前最先进的GenAI模型的合规能力加以补充。为了支持讨论，我们概述了将GenAI整合到标准合规任务中可能面临的挑战和机遇，并为参与标准开发和使用的实体提供可操作的建议。总体而言，我们认为通过计算方法使GenAI与标准相一致有助于强化监管和运营合规性。我们预计，这一研究领域将在未来对大型、更强大的基于GenAI系统的管理和监督中扮演核心角色。 

---
# What can large language models do for sustainable food? 

**Title (ZH)**: 大型语言模型如何促进可持续食品？ 

**Authors**: Anna T. Thomas, Adam Yee, Andrew Mayne, Maya B. Mathur, Dan Jurafsky, Kristina Gligorić  

**Link**: [PDF](https://arxiv.org/pdf/2503.04734)  

**Abstract**: Food systems are responsible for a third of human-caused greenhouse gas emissions. We investigate what Large Language Models (LLMs) can contribute to reducing the environmental impacts of food production. We define a typology of design and prediction tasks based on the sustainable food literature and collaboration with domain experts, and evaluate six LLMs on four tasks in our typology. For example, for a sustainable protein design task, food science experts estimated that collaboration with an LLM can reduce time spent by 45% on average, compared to 22% for collaboration with another expert human food scientist. However, for a sustainable menu design task, LLMs produce suboptimal solutions when instructed to consider both human satisfaction and climate impacts. We propose a general framework for integrating LLMs with combinatorial optimization to improve reasoning capabilities. Our approach decreases emissions of food choices by 79% in a hypothetical restaurant while maintaining participants' satisfaction with their set of choices. Our results demonstrate LLMs' potential, supported by optimization techniques, to accelerate sustainable food development and adoption. 

**Abstract (ZH)**: 食物系统 responsable for 约三分之一的人为温室气体排放。我们探讨了大型语言模型（LLMs）在减少食物生产环境影响方面的贡献。我们根据可持续食物文献和与领域专家的合作，定义了一种设计和预测任务类型，并在我们定义的四种任务上评估了六种LLMs。例如，在可持续蛋白质设计任务中，食品科学专家估计与LLM合作可以将平均时间减少45%，而与另一名专家人类食品科学家合作则减少22%。然而，在可持续菜单设计任务中，当LLM被指示同时考虑人类满意和气候影响时，生成的解决方案往往是次优的。我们提出了一种通用框架，通过结合组合优化来增强LLM的推理能力。我们的方法在假设的餐馆中减少了79%的食物选择排放，同时保持了参与者对其选择集的满意度。我们的结果证明了，通过优化技术，LLMs有能力加速可持续食物的发展和采用。 

---
# Ethics of generative AI and manipulation: a design-oriented research agenda 

**Title (ZH)**: 生成式人工智能与操控的伦理：以设计为导向的研究议程 

**Authors**: Michael Klenk  

**Link**: [PDF](https://arxiv.org/pdf/2503.04733)  

**Abstract**: Generative AI enables automated, effective manipulation at scale. Despite the growing general ethical discussion around generative AI, the specific manipulation risks remain inadequately investigated. This article outlines essential inquiries encompassing conceptual, empirical, and design dimensions of manipulation, pivotal for comprehending and curbing manipulation risks. By highlighting these questions, the article underscores the necessity of an appropriate conceptualisation of manipulation to ensure the responsible development of Generative AI technologies. 

**Abstract (ZH)**: 生成式AI使大规模自动化有效操作成为可能。尽管生成式AI的通用伦理讨论日益增长，但具体的操纵风险仍未得到充分研究。本文概述了涵盖概念、实证和设计维度的关键问题，对于理解和遏制操纵风险至关重要。通过强调这些问题，本文强调了对操纵进行适当概念化的重要性，以确保生成式AI技术的负责任发展。 

---
# Epistemic Logic Programs: Non-Ground and Counting Complexity 

**Title (ZH)**: 知识逻辑程序：非ground化与计数复杂性 

**Authors**: Thomas Eiter, Johannes K. Fichte, Markus Hecher, Stefan Woltran  

**Link**: [PDF](https://arxiv.org/pdf/2503.04731)  

**Abstract**: Answer Set Programming (ASP) is a prominent problem-modeling and solving framework, whose solutions are called answer sets. Epistemic logic programs (ELP) extend ASP to reason about all or some answer sets. Solutions to an ELP can be seen as consequences over multiple collections of answer sets, known as world views. While the complexity of propositional programs is well studied, the non-ground case remains open. This paper establishes the complexity of non-ground ELPs. We provide a comprehensive picture for well-known program fragments, which turns out to be complete for the class NEXPTIME with access to oracles up to \Sigma^P_2. In the quantitative setting, we establish complexity results for counting complexity beyond #EXP. To mitigate high complexity, we establish results in case of bounded predicate arity, reaching up to the fourth level of the polynomial hierarchy. Finally, we provide ETH-tight runtime results for the parameter treewidth, which has applications in quantitative reasoning, where we reason on (marginal) probabilities of epistemic literals. 

**Abstract (ZH)**: 非地面Epistemic逻辑程序的复杂性分析 

---
# Leveraging Large Language Models For Optimized Item Categorization using UNSPSC Taxonomy 

**Title (ZH)**: 利用大型语言模型优化基于UNSPSC分类法的物品分类 

**Authors**: Anmolika Singh, Yuhang Diao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04728)  

**Abstract**: Effective item categorization is vital for businesses, enabling the transformation of unstructured datasets into organized categories that streamline inventory management. Despite its importance, item categorization remains highly subjective and lacks a uniform standard across industries and businesses. The United Nations Standard Products and Services Code (UNSPSC) provides a standardized system for cataloguing inventory, yet employing UNSPSC categorizations often demands significant manual effort. This paper investigates the deployment of Large Language Models (LLMs) to automate the classification of inventory data into UNSPSC codes based on Item Descriptions. We evaluate the accuracy and efficiency of LLMs in categorizing diverse datasets, exploring their language processing capabilities and their potential as a tool for standardizing inventory classification. Our findings reveal that LLMs can substantially diminish the manual labor involved in item categorization while maintaining high accuracy, offering a scalable solution for businesses striving to enhance their inventory management practices. 

**Abstract (ZH)**: 有效的物品分类对于企业至关重要，能够将无结构的数据集转化为有组织的类别，简化库存管理。尽管如此，物品分类仍然高度主观，并且缺乏跨行业和企业的统一标准。联合国标准产品和服务分类码（UNSPSC）提供了一种标准化的库存分类系统，但采用UNSPSC分类通常需要大量的手动努力。本文研究了大型语言模型（LLMs）在基于物品描述将库存数据分类到UNSPSC代码中的应用效果，评估了LLMs在分类多样数据集方面的准确性和效率，探讨了它们的语言处理能力和作为标准化库存分类工具的潜力。我们的研究发现，LLMs可以大幅减少物品分类所需的 manual 努力，同时保持高准确性，为企业提供了一种可扩展的解决方案，以增强其库存管理实践。 

---
# Function-Coherent Gambles 

**Title (ZH)**: 功能相干赌局 

**Authors**: Gregory Wheeler  

**Link**: [PDF](https://arxiv.org/pdf/2503.01855)  

**Abstract**: The desirable gambles framework provides a foundational approach to imprecise probability theory but relies heavily on linear utility assumptions. This paper introduces {\em function-coherent gambles}, a generalization that accommodates non-linear utility while preserving essential rationality properties. We establish core axioms for function-coherence and prove a representation theorem that characterizes acceptable gambles through continuous linear functionals. The framework is then applied to analyze various forms of discounting in intertemporal choice, including hyperbolic, quasi-hyperbolic, scale-dependent, and state-dependent discounting. We demonstrate how these alternatives to constant-rate exponential discounting can be integrated within the function-coherent framework. This unified treatment provides theoretical foundations for modeling sophisticated patterns of time preference within the desirability paradigm, bridging a gap between normative theory and observed behavior in intertemporal decision-making under genuine uncertainty. 

**Abstract (ZH)**: 函数一致赌注：非线性效用下不精确概率理论的广义基础及时间折扣分析 

---
# Static Vs. Agentic Game Master AI for Facilitating Solo Role-Playing Experiences 

**Title (ZH)**: 静态 vs. 主动游戏大师AI在促进单人角色扮演体验中的应用 

**Authors**: Nicolai Hejlesen Jørgensen, Sarmilan Tharmabalan, Ilhan Aslan, Nicolai Brodersen Hansen, Timothy Merritt  

**Link**: [PDF](https://arxiv.org/pdf/2502.19519)  

**Abstract**: This paper presents a game master AI for single-player role-playing games. The AI is designed to deliver interactive text-based narratives and experiences typically associated with multiplayer tabletop games like Dungeons & Dragons. We report on the design process and the series of experiments to improve the functionality and experience design, resulting in two functional versions of the system. While v1 of our system uses simplified prompt engineering, v2 leverages a multi-agent architecture and the ReAct framework to include reasoning and action. A comparative evaluation demonstrates that v2 as an agentic system maintains play while significantly improving modularity and game experience, including immersion and curiosity. Our findings contribute to the evolution of AI-driven interactive fiction, highlighting new avenues for enhancing solo role-playing experiences. 

**Abstract (ZH)**: 本文提出了一种面向单人角色扮演游戏的场景主持AI。该AI旨在提供类似于多人桌面游戏（例如龙与地下城）的互动文本叙述和体验。我们报告了该设计过程以及一系列实验以改进功能和体验设计，最终实现了两个功能版本的系统。虽然系统v1采用简化的提示工程，但v2则利用多智能体架构和ReAct框架，加入了推理和行动。比较评估表明，v2作为一种自主系统在保持游戏性的同时显著提高了模块化程度和游戏体验，包括沉浸感和好奇心。我们的研究结果促进了由AI驱动的互动小说的发展，指出了增强单人角色扮演体验的新途径。 

---
