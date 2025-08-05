# COLLAGE: Adaptive Fusion-based Retrieval for Augmented Policy Learning 

**Title (ZH)**: 拼贴：自适应融合检索以增强策略学习 

**Authors**: Sateesh Kumar, Shivin Dass, Georgios Pavlakos, Roberto Martín-Martín  

**Link**: [PDF](https://arxiv.org/pdf/2508.01131)  

**Abstract**: In this work, we study the problem of data retrieval for few-shot imitation learning: selecting data from a large dataset to train a performant policy for a specific task, given only a few target demonstrations. Prior methods retrieve data using a single-feature distance heuristic, assuming that the best demonstrations are those that most closely resemble the target examples in visual, semantic, or motion space. However, this approach captures only a subset of the relevant information and can introduce detrimental demonstrations, e.g., retrieving data from unrelated tasks due to similar scene layouts, or selecting similar motions from tasks with divergent goals. We present COLLAGE, a method for COLLective data AGgrEgation in few-shot imitation learning that uses an adaptive late fusion mechanism to guide the selection of relevant demonstrations based on a task-specific combination of multiple cues. COLLAGE follows a simple, flexible, and efficient recipe: it assigns weights to subsets of the dataset that are pre-selected using a single feature (e.g., appearance, shape, or language similarity), based on how well a policy trained on each subset predicts actions in the target demonstrations. These weights are then used to perform importance sampling during policy training, sampling data more densely or sparsely according to estimated relevance. COLLAGE is general and feature-agnostic, allowing it to combine any number of subsets selected by any retrieval heuristic, and to identify which subsets provide the greatest benefit for the target task. In extensive experiments, COLLAGE outperforms state-of-the-art retrieval and multi-task learning approaches by 5.1% in simulation across 10 tasks, and by 16.6% in the real world across 6 tasks, where we perform retrieval from the large-scale DROID dataset. More information at this https URL . 

**Abstract (ZH)**: 面向少样本模仿学习的数据检索问题：选择特定任务性能良好的数据方法，仅给定少量目标演示。先前方法使用单特征距离启发式检索数据，假设最佳演示是那些在视觉、语义或运动空间中最接近目标示例的。然而，这种方法仅捕获了一部分相关信息，并可能会引入不利的演示，例如，由于场景布局相似性，从不相关的任务检索数据；或者在目标任务具有不同目标的情景下选择相似的动作。我们提出了COLLAGE，一种面向少样本模仿学习的集合数据聚合方法，利用自适应后期融合机制，基于特定任务的多个线索组合引导相关演示的选择。COLLAGE 遵循一个简单、灵活且高效的配方：它根据训练于每个子集上的策略对目标演示中动作的预测能力，为这些子集分配权重。然后，在策略训练期间使用这些权重进行重要性采样，根据估计的相关性密度更大或更小地采样数据。COLLAGE 是通用且特征无关的，允许它结合任意数量由任何检索启发式选择的子集，并确定哪些子集对目标任务提供了最大的好处。在广泛的实验中，COLLAGE 在模拟环境中在 10 个任务上的性能优于最先进的检索和多任务学习方法 5.1%，在现实世界中在 6 个任务上的性能优于 16.6%，其中我们在大规模 DROID 数据集中执行检索。更多详细信息请访问此链接。 

---
# MUTE-DSS: A Digital-Twin-Based Decision Support System for Minimizing Underwater Radiated Noise in Ship Voyage Planning 

**Title (ZH)**: MUTE-DSS: 基于数字孪生的海底辐射噪声最小化船舶航程规划决策支持系统 

**Authors**: Akash Venkateshwaran, Indu Kant Deo, Rajeev K. Jaiman  

**Link**: [PDF](https://arxiv.org/pdf/2508.01907)  

**Abstract**: We present a novel MUTE-DSS, a digital-twin-based decision support system for minimizing underwater radiated noise (URN) during ship voyage planning. It is a ROS2-centric framework that integrates state-of-the-art acoustic models combining a semi-empirical reference spectrum for near-field modeling with 3D ray tracing for propagation losses for far-field modeling, offering real-time computation of the ship noise signature, alongside a data-driven Southern resident killer whale distribution model. The proposed DSS performs a two-stage optimization pipeline: Batch Informed Trees for collision-free ship routing and a genetic algorithm for adaptive ship speed profiling under voyage constraints that minimizes cumulative URN exposure to marine mammals. The effectiveness of MUTE-DSS is demonstrated through case studies of ships operating between the Strait of Georgia and the Strait of Juan de Fuca, comparing optimized voyages against baseline trajectories derived from automatic identification system data. Results show substantial reductions in noise exposure level, up to 7.14 dB, corresponding to approximately an 80.68% reduction in a simplified scenario, and an average 4.90 dB reduction, corresponding to approximately a 67.6% reduction in a more realistic dynamic setting. These results illustrate the adaptability and practical utility of the proposed decision support system. 

**Abstract (ZH)**: 基于数字孪生的MUTE-DSS决策支持系统：用于 ship 航行规划期间最小化水下辐射噪声（URN） 

---
# T2S: Tokenized Skill Scaling for Lifelong Imitation Learning 

**Title (ZH)**: T2S: 分词技能扩展for 全生命周期imitation learning 

**Authors**: Hongquan Zhang, Jingyu Gong, Zhizhong Zhang, Xin Tan, Yanyun Qu, Yuan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2508.01167)  

**Abstract**: The main challenge in lifelong imitation learning lies in the balance between mitigating catastrophic forgetting of previous skills while maintaining sufficient capacity for acquiring new ones. However, current approaches typically address these aspects in isolation, overlooking their internal correlation in lifelong skill acquisition. We address this limitation with a unified framework named Tokenized Skill Scaling (T2S). Specifically, by tokenizing the model parameters, the linear parameter mapping of the traditional transformer is transformed into cross-attention between input and learnable tokens, thereby enhancing model scalability through the easy extension of new tokens. Additionally, we introduce language-guided skill scaling to transfer knowledge across tasks efficiently and avoid linearly growing parameters. Extensive experiments across diverse tasks demonstrate that T2S: 1) effectively prevents catastrophic forgetting (achieving an average NBT of 1.0% across the three LIBERO task suites), 2) excels in new skill scaling with minimal increases in trainable parameters (needing only 8.0% trainable tokens in an average of lifelong tasks), and 3) enables efficient knowledge transfer between tasks (achieving an average FWT of 77.7% across the three LIBERO task suites), offering a promising solution for lifelong imitation learning. 

**Abstract (ZH)**: lifelong imitation学习的主要挑战在于在减轻先前技能灾难性遗忘的同时维持足够容量以获取新技能之间的平衡。然而，当前方法通常将这两方面分离处理，忽略了它们在终身技能获取中的内在关联。我们通过一个名为Tokenized Skill Scaling (T2S)的统一框架来克服这一限制。具体而言，通过模型参数的-token化-，传统的变压器的线性参数映射被转换为输入和可学习标记之间的交叉注意，从而通过新标记的易于扩展性增强模型的可扩展性。此外，我们引入了由语言指导的技能缩放，以高效地跨任务转移知识并避免参数线性增长。广泛的跨多种任务的实验证明了T2S：1) 有效防止了灾难性遗忘（在三个LIBERO任务套件中平均NBT为1.0%），2) 在新技能缩放方面表现优异，需要的可训练参数小幅增加（平均终身任务中仅为8.0%的可训练标记），3) 使任务之间的知识转移变得高效（在三个LIBERO任务套件中平均FWT为77.7%），提供了一个有前景的终身模仿学习解决方案。 

---
# Actionable Counterfactual Explanations Using Bayesian Networks and Path Planning with Applications to Environmental Quality Improvement 

**Title (ZH)**: 基于贝叶斯网络和路径规划的可操作反事实解释及其在环境质量改善中的应用 

**Authors**: Enrique Valero-Leal, Pedro Larrañaga, Concha Bielza  

**Link**: [PDF](https://arxiv.org/pdf/2508.02634)  

**Abstract**: Counterfactual explanations study what should have changed in order to get an alternative result, enabling end-users to understand machine learning mechanisms with counterexamples. Actionability is defined as the ability to transform the original case to be explained into a counterfactual one. We develop a method for actionable counterfactual explanations that, unlike predecessors, does not directly leverage training data. Rather, data is only used to learn a density estimator, creating a search landscape in which to apply path planning algorithms to solve the problem and masking the endogenous data, which can be sensitive or private. We put special focus on estimating the data density using Bayesian networks, demonstrating how their enhanced interpretability is useful in high-stakes scenarios in which fairness is raising concern. Using a synthetic benchmark comprised of 15 datasets, our proposal finds more actionable and simpler counterfactuals than the current state-of-the-art algorithms. We also test our algorithm with a real-world Environmental Protection Agency dataset, facilitating a more efficient and equitable study of policies to improve the quality of life in United States of America counties. Our proposal captures the interaction of variables, ensuring equity in decisions, as policies to improve certain domains of study (air, water quality, etc.) can be detrimental in others. In particular, the sociodemographic domain is often involved, where we find important variables related to the ongoing housing crisis that can potentially have a severe negative impact on communities. 

**Abstract (ZH)**: 基于反事实解释的研究探讨了为了获得替代结果需要发生哪些变化，从而帮助最终用户理解机器学习机制。行动性被定义为将原始待解释案例转换为反事实案例的能力。我们开发了一种行动性反事实解释方法，与之前的算法不同，该方法不直接利用训练数据，而是仅使用数据来学习密度估计器，在其中应用路径规划算法解决问题，并屏蔽可能敏感或私有的内生数据。我们特别关注使用贝叶斯网络估计数据密度，并展示了其增强的可解释性在高风险场景中如何在公平性受到关注时发挥作用。通过由15个数据集组成的合成基准测试，我们的提案在可操作性和简洁性方面找到了当前最先进的算法所未能发现的反事实解释。我们还将算法应用于现实世界的环境保护局数据集，促进了美国各县生活质量改善政策的更高效和公平的研究。我们的提案捕获了变量之间的交互作用，确保决策公平，因为旨在改善某些研究领域（空气质量、水质等）的政策可能在其他领域产生负面影响。特别是在社会经济领域，我们发现与持续的住房危机相关的关键变量，这些变量可能会对社区产生严重负面影响。 

---
# What Is Your AI Agent Buying? Evaluation, Implications and Emerging Questions for Agentic E-Commerce 

**Title (ZH)**: 你的AI代理在买什么？关于代理型电子商务的评估、影响与新兴问题 

**Authors**: Amine Allouah, Omar Besbes, Josué D Figueroa, Yash Kanoria, Akshit Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2508.02630)  

**Abstract**: Online marketplaces will be transformed by autonomous AI agents acting on behalf of consumers. Rather than humans browsing and clicking, vision-language-model (VLM) agents can parse webpages, evaluate products, and transact. This raises a fundamental question: what do AI agents buy, and why? We develop ACES, a sandbox environment that pairs a platform-agnostic VLM agent with a fully programmable mock marketplace to study this question. We first conduct basic rationality checks in the context of simple tasks, and then, by randomizing product positions, prices, ratings, reviews, sponsored tags, and platform endorsements, we obtain causal estimates of how frontier VLMs actually shop. Models show strong but heterogeneous position effects: all favor the top row, yet different models prefer different columns, undermining the assumption of a universal "top" rank. They penalize sponsored tags and reward endorsements. Sensitivities to price, ratings, and reviews are directionally human-like but vary sharply in magnitude across models. Motivated by scenarios where sellers use AI agents to optimize product listings, we show that a seller-side agent that makes minor tweaks to product descriptions, targeting AI buyer preferences, can deliver substantial market-share gains if AI-mediated shopping dominates. We also find that modal product choices can differ across models and, in some cases, demand may concentrate on a few select products, raising competition questions. Together, our results illuminate how AI agents may behave in e-commerce settings and surface concrete seller strategy, platform design, and regulatory questions in an AI-mediated ecosystem. 

**Abstract (ZH)**: 在线市场将由代表消费者行动的自主AI代理重构。这些代理可以解析网页、评估产品并完成交易，而不仅仅是人力浏览和点击。这引发了一个基本问题：AI代理会购买什么，并且为什么会这样？我们开发了ACES，一种沙盒环境，将一个平台无关的VLM代理与一个完全可编程的模拟市场配对，以研究这一问题。我们首先在简单任务的情境中进行基本的理性检验，然后通过随机化产品位置、价格、评分、评论、付费标签和平台背书，我们获得了前沿VLM实际上购物方式的因果估计。模型显示了强烈但各不相同的位次效应：所有模型都偏好第一行，但不同的模型偏好不同的列，这削弱了普遍存在“顶级”排名的假设。它们会惩罚付费标签并奖励背书。对价格、评分和评论的敏感性表现为人类般的方向，但在不同模型中的幅度差异巨大。受卖家使用AI代理优化产品列表场景的启发，我们展示了如果AI中介购物占主导地位，一个针对AI买家偏好的产品描述微调的卖家代理可以显著提升市场份额。我们还发现，主流产品选择在不同模型之间可能有所不同，在某些情况下，需求可能会集中在少数几种产品上，这提出了竞争方面的疑问。我们的研究结果阐明了AI代理在电子商务环境中的行为方式，并揭示了AI中介生态系统中的具体卖家策略、平台设计和监管问题。 

---
# HealthFlow: A Self-Evolving AI Agent with Meta Planning for Autonomous Healthcare Research 

**Title (ZH)**: 健康流：一种基于元规划的自我进化AI医疗代理 

**Authors**: Yinghao Zhu, Yifan Qi, Zixiang Wang, Lei Gu, Dehao Sui, Haoran Hu, Xichen Zhang, Ziyi He, Liantao Ma, Lequan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02621)  

**Abstract**: The efficacy of AI agents in healthcare research is hindered by their reliance on static, predefined strategies. This creates a critical limitation: agents can become better tool-users but cannot learn to become better strategic planners, a crucial skill for complex domains like healthcare. We introduce HealthFlow, a self-evolving AI agent that overcomes this limitation through a novel meta-level evolution mechanism. HealthFlow autonomously refines its own high-level problem-solving policies by distilling procedural successes and failures into a durable, strategic knowledge base. To anchor our research and facilitate reproducible evaluation, we introduce EHRFlowBench, a new benchmark featuring complex, realistic health data analysis tasks derived from peer-reviewed clinical research. Our comprehensive experiments demonstrate that HealthFlow's self-evolving approach significantly outperforms state-of-the-art agent frameworks. This work marks a necessary shift from building better tool-users to designing smarter, self-evolving task-managers, paving the way for more autonomous and effective AI for scientific discovery. 

**Abstract (ZH)**: AI代理在医疗研究中的有效性受制于其对静态、预定义策略的依赖。这造成本质上的一个关键限制：代理可以变得更好的工具使用者，但不能学习成为更好的战略规划者，这对像医疗这样的复杂领域至关重要。我们提出了HealthFlow，这是一种通过新颖的元级进化机制克服这一限制的自适应进化AI代理。HealthFlow自主地通过提炼程序上的成功与失败提炼出一个持久的战略知识库，以优化其高级问题解决策略。为支撑我们的研究并促进可再现评估，我们引入了EHRFlowBench，这是一个包含来自同行评审临床研究的复杂、现实的健康数据分析任务的新基准。我们的综合实验表明，HealthFlow的自适应进化方法显著优于现有最先进的代理框架。这项工作标志着从构建更好的工具使用者转向设计更智能、自适应进化的任务管理者的重要转变，为更自主和有效的科学发现AI铺平了道路。 

---
# CABENCH: Benchmarking Composable AI for Solving Complex Tasks through Composing Ready-to-Use Models 

**Title (ZH)**: CABENCH: 通过组合即用型模型评估可组合AI解决复杂任务的能力 

**Authors**: Tung-Thuy Pham, Duy-Quan Luong, Minh-Quan Duong, Trung-Hieu Nguyen, Thu-Trang Nguyen, Son Nguyen, Hieu Dinh Vo  

**Link**: [PDF](https://arxiv.org/pdf/2508.02427)  

**Abstract**: Composable AI offers a scalable and effective paradigm for tackling complex AI tasks by decomposing them into sub-tasks and solving each sub-task using ready-to-use well-trained models. However, systematically evaluating methods under this setting remains largely unexplored. In this paper, we introduce CABENCH, the first public benchmark comprising 70 realistic composable AI tasks, along with a curated pool of 700 models across multiple modalities and domains. We also propose an evaluation framework to enable end-to-end assessment of composable AI solutions. To establish initial baselines, we provide human-designed reference solutions and compare their performance with two LLM-based approaches. Our results illustrate the promise of composable AI in addressing complex real-world problems while highlighting the need for methods that can fully unlock its potential by automatically generating effective execution pipelines. 

**Abstract (ZH)**: 可组合AI通过分解复杂AI任务并使用已训练模型解决子任务，提供了一种可扩展和有效的范式。然而，在这种设置下系统性地评估方法仍未得到充分探索。本文介绍了CABENCH，这是首个包含70个现实可组合AI任务的公开基准，同时还收录了来自多个模态和领域中的700个模型。我们还提出了一种评估框架，以实现端到端评估可组合AI解决方案。为建立初始基准，我们提供了人工设计的参考解决方案，并将其性能与两种基于LLM的方法进行了比较。我们的结果展示了可组合AI在解决复杂现实世界问题方面的潜力，同时也突显了通过自动生成有效执行管道来充分利用其全部潜力的必要性。 

---
# FinWorld: An All-in-One Open-Source Platform for End-to-End Financial AI Research and Deployment 

**Title (ZH)**: FinWorld: 一站式开源金融AI研究与部署平台 

**Authors**: Wentao Zhang, Yilei Zhao, Chuqiao Zong, Xinrun Wang, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2508.02292)  

**Abstract**: Financial AI holds great promise for transforming modern finance, with the potential to support a wide range of tasks such as market forecasting, portfolio management, quantitative trading, and automated analysis. However, existing platforms remain limited in task coverage, lack robust multimodal data integration, and offer insufficient support for the training and deployment of large language models (LLMs). In response to these limitations, we present FinWorld, an all-in-one open-source platform that provides end-to-end support for the entire financial AI workflow, from data acquisition to experimentation and deployment. FinWorld distinguishes itself through native integration of heterogeneous financial data, unified support for diverse AI paradigms, and advanced agent automation, enabling seamless development and deployment. Leveraging data from 2 representative markets, 4 stock pools, and over 800 million financial data points, we conduct comprehensive experiments on 4 key financial AI tasks. These experiments systematically evaluate deep learning and reinforcement learning algorithms, with particular emphasis on RL-based finetuning for LLMs and LLM Agents. The empirical results demonstrate that FinWorld significantly enhances reproducibility, supports transparent benchmarking, and streamlines deployment, thereby providing a strong foundation for future research and real-world applications. Code is available at Github~\footnote{this https URL}. 

**Abstract (ZH)**: 金融AI具有极大潜力以重塑现代金融，有望支持包括市场预测、投资组合管理、量化交易和自动分析等一系列任务。然而，现有平台在任务覆盖范围、多模态数据整合 robust multimodal data integration 和大规模语言模型（LLMs）的训练与部署支持方面仍存在局限性。针对这些局限性，我们提出了一个一站式开源平台 FinWorld，该平台从数据获取到实验和部署为整个金融AI工作流程提供端到端支持。FinWorld 通过原生整合异构金融数据、统一支持多种AI范式以及先进的代理自动化，实现了无缝开发与部署。通过利用两个代表性市场、四个股票池以及超过 8 亿个金融数据点的数据，我们在四个关键的金融AI任务上进行了全面实验。这些实验系统地评估了深度学习和强化学习算法，并特别侧重于基于RL的LLM微调和LLM代理。实验证明，FinWorld 显著增强了可重复性，支持透明基准测试，并简化了部署，从而为未来的研究和实际应用提供了坚实基础。代码可在 Github 上获取。 

---
# A Message Passing Realization of Expected Free Energy Minimization 

**Title (ZH)**: 预期自由能最小化的一种消息传递实现 

**Authors**: Wouter W. L. Nuijten, Mykola Lukashchuk, Thijs van de Laar, Bert de Vries  

**Link**: [PDF](https://arxiv.org/pdf/2508.02197)  

**Abstract**: We present a message passing approach to Expected Free Energy (EFE) minimization on factor graphs, based on the theory introduced in arXiv:2504.14898. By reformulating EFE minimization as Variational Free Energy minimization with epistemic priors, we transform a combinatorial search problem into a tractable inference problem solvable through standard variational techniques. Applying our message passing method to factorized state-space models enables efficient policy inference. We evaluate our method on environments with epistemic uncertainty: a stochastic gridworld and a partially observable Minigrid task. Agents using our approach consistently outperform conventional KL-control agents on these tasks, showing more robust planning and efficient exploration under uncertainty. In the stochastic gridworld environment, EFE-minimizing agents avoid risky paths, while in the partially observable minigrid setting, they conduct more systematic information-seeking. This approach bridges active inference theory with practical implementations, providing empirical evidence for the efficiency of epistemic priors in artificial agents. 

**Abstract (ZH)**: 基于arXiv:2504.14898中提出理论的消息传递方法在因子图上最小化预期自由能的研究 

---
# Reconsidering Overthinking: Penalizing Internal and External Redundancy in CoT Reasoning 

**Title (ZH)**: 重新审视过度思考：在共情推理中惩罚内部和外部冗余 

**Authors**: Jialiang Hong, Taihang Zhen, Kai Chen, Jiaheng Liu, Wenpeng Zhu, Jing Huo, Yang Gao, Depeng Wang, Haitao Wan, Xi Yang, Boyan Wang, Fanyu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2508.02178)  

**Abstract**: Large Reasoning Models (LRMs) often produce excessively verbose reasoning traces, a phenomenon known as overthinking, which hampers both efficiency and interpretability. Prior works primarily address this issue by reducing response length, without fully examining the underlying semantic structure of the reasoning process. In this paper, we revisit overthinking by decomposing it into two distinct forms: internal redundancy, which consists of low-contribution reasoning steps within the first correct solution (FCS), and external redundancy, which refers to unnecessary continuation after the FCS. To mitigate both forms, we propose a dual-penalty reinforcement learning framework. For internal redundancy, we adopt a sliding-window semantic analysis to penalize low-gain reasoning steps that contribute little toward reaching the correct answer. For external redundancy, we penalize its proportion beyond the FCS to encourage earlier termination. Our method significantly compresses reasoning traces with minimal accuracy loss, and generalizes effectively to out-of-domain tasks such as question answering and code generation. Crucially, we find that external redundancy can be safely removed without degrading performance, whereas internal redundancy must be reduced more cautiously to avoid impairing correctness. These findings suggest that our method not only improves reasoning efficiency but also enables implicit, semantic-aware control over Chain-of-Thought length, paving the way for more concise and interpretable LRMs. 

**Abstract (ZH)**: 大型推理模型中的过度推理现象通常表现为生成冗长的推理痕迹，这影响了效率和可解释性。以往工作主要通过减少响应长度来缓解这一问题，但未充分探究推理过程的语义结构。本文重新审视过度推理，将其分解为两种形式：内部冗余，即在首次正确解（FCS）内的低贡献推理步骤；外部冗余，即FCS之后不必要的继续推理。为此，我们提出了一种双罚 reinforcement 学习框架。对于内部冗余，我们采用滑动窗口语义分析来惩罚对达到正确答案贡献较小的低收益推理步骤；对于外部冗余，我们惩罚其FCS外的比例以鼓励更早终止。我们的方法在几乎不损失准确性的前提下显著压缩了推理痕迹，并能有效泛化到领域外任务如问答和代码生成。重要的是，我们发现外部冗余可以安全移除而不影响性能，而内部冗余则需谨慎减少以避免影响正确性。这些发现表明，我们的方法不仅能提高推理效率，还能实现对推理链长度的隐式、语义感知控制，为更简洁和可解释的大型推理模型铺平了道路。 

---
# Dynamic Context Adaptation for Consistent Role-Playing Agents with Retrieval-Augmented Generations 

**Title (ZH)**: 动态上下文适应以实现一致性角色扮演代理的检索增强生成 

**Authors**: Jeiyoon Park, Yongshin Han, Minseop Kim, Kisu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02016)  

**Abstract**: We propose AMADEUS, which is composed of Adaptive Context-aware Text Splitter (ACTS), Guided Selection (GS), and Attribute Extractor (AE). ACTS finds an optimal chunk length and hierarchical contexts for each character. AE identifies a character's general attributes from the chunks retrieved by GS and uses these attributes as a final context to maintain robust persona consistency even when answering out of knowledge questions. To facilitate the development and evaluation of RAG-based RPAs, we construct CharacterRAG, a role-playing dataset that consists of persona documents for 15 distinct fictional characters totaling 976K written characters, and 450 question and answer pairs. We find that our framework effectively models not only the knowledge possessed by characters, but also various attributes such as personality. 

**Abstract (ZH)**: 我们提出AMADEUS，其由自适应上下文感知文本分割器（ACTS）、引导选择（GS）和属性 extractor（AE）组成。我们构建了CharacterRAG，一个角色扮演数据集，包含15个不同虚构角色的persona文档共计976K汉字，以及450组问题和答案对。我们发现，该框架不仅有效地建模了角色的知识，还涵盖了诸如个性等多种属性。 

---
# Multi-turn Natural Language to Graph Query Language Translation 

**Title (ZH)**: 多轮自然语言到图形查询语言翻译 

**Authors**: Yuanyuan Liang, Lei Pan, Tingyu Xie, Yunshi Lan, Weining Qian  

**Link**: [PDF](https://arxiv.org/pdf/2508.01871)  

**Abstract**: In recent years, research on transforming natural language into graph query language (NL2GQL) has been increasing. Most existing methods focus on single-turn transformation from NL to GQL. In practical applications, user interactions with graph databases are typically multi-turn, dynamic, and context-dependent. While single-turn methods can handle straightforward queries, more complex scenarios often require users to iteratively adjust their queries, investigate the connections between entities, or request additional details across multiple dialogue turns. Research focused on single-turn conversion fails to effectively address multi-turn dialogues and complex context dependencies. Additionally, the scarcity of high-quality multi-turn NL2GQL datasets further hinders the progress of this field. To address this challenge, we propose an automated method for constructing multi-turn NL2GQL datasets based on Large Language Models (LLMs) , and apply this method to develop the MTGQL dataset, which is constructed from a financial market graph database and will be publicly released for future research. Moreover, we propose three types of baseline methods to assess the effectiveness of multi-turn NL2GQL translation, thereby laying a solid foundation for future research. 

**Abstract (ZH)**: 近年来，将自然语言转换为图形查询语言（NL2GQL）的研究不断增加。大多数现有方法专注于从自然语言单步转换为图形查询语言。在实际应用中，用户与图形数据库的交互通常是多轮的、动态的且依赖于上下文。虽然单轮方法可以处理简单的查询，但在更复杂的情景中，用户通常需要迭代调整查询、探索实体之间的联系或在多轮对话中请求更多细节。专注于单轮转换的研究无法有效解决多轮对话和复杂的上下文依赖性。此外，高质量的多轮NL2GQL数据集的稀缺性进一步阻碍了该领域的发展。为应对这一挑战，我们提出了基于大型语言模型（LLMs）的自动化方法以构建多轮NL2GQL数据集，并应用该方法开发了MTGQL数据集，该数据集基于金融市场的图形数据库，并将公开发布供未来研究使用。此外，我们提出了三种基线方法来评估多轮NL2GQL翻译的有效性，从而为未来研究奠定坚实基础。 

---
# ProKG-Dial: Progressive Multi-Turn Dialogue Construction with Domain Knowledge Graphs 

**Title (ZH)**: 渐进步进式多轮对话构建结合领域知识图谱 

**Authors**: Yuanyuan Liang, Xiaoman Wang, Tingyu Xie, Lei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2508.01869)  

**Abstract**: Current large language models (LLMs) excel at general NLP tasks but often lack domain specific precision in professional settings. Building a high quality domain specific multi turn dialogue dataset is essential for developing specialized conversational systems. However, existing methods such as manual annotation, simulated human LLM interactions, and role based LLM dialogues are resource intensive or suffer from limitations in dialogue quality and domain coverage. To address these challenges, we introduce ProKG Dial, a progressive framework for constructing knowledge intensive multi turn dialogue datasets using domain specific knowledge graphs (KGs). ProKG Dial leverages the structured nature of KGs to encode complex domain knowledge and relationships, providing a solid foundation for generating meaningful and coherent dialogues. Specifically, ProKG Dial begins by applying community detection to partition the KG into semantically cohesive subgraphs. For each subgraph, the framework incrementally generates a series of questions and answers centered around a target entity, ensuring relevance and coverage. A rigorous filtering step is employed to maintain high dialogue quality. We validate ProKG Dial on a medical knowledge graph by evaluating the generated dialogues in terms of diversity, semantic coherence, and entity coverage. Furthermore, we fine tune a base LLM on the resulting dataset and benchmark it against several baselines. Both automatic metrics and human evaluations demonstrate that ProKG Dial substantially improves dialogue quality and domain specific performance, highlighting its effectiveness and practical utility. 

**Abstract (ZH)**: 当前的大规模语言模型在通用自然语言处理任务上表现出色，但在专业环境中往往缺乏领域-specific的精准度。构建高质量的领域特定多轮对话数据集对于开发专门化的对话系统至关重要。然而，现有方法如手动标注、模拟人类与大型语言模型的交互以及基于角色的大型语言模型对话，要么资源密集，要么在对话质量和领域覆盖上存在局限。为应对这些挑战，我们引入了ProKG Dial，这是一种渐进框架，用于利用领域特定知识图谱（KGs）构建知识密集型的多轮对话数据集。ProKG Dial 利用KG的结构化特性来编码复杂的领域知识和关系，为生成有意义且连贯的对话提供了坚实的基础。具体而言，ProKG Dial 首先应用社区检测将KG划分为语义上统一的子图。对于每个子图，框架逐步生成围绕目标实体的问题和答案，确保相关性和覆盖范围。采用严格的过滤步骤来维持高对话质量。我们通过评估生成的对话在多样性、语义连贯性和实体覆盖方面的表现，对ProKG Dial在医学知识图谱上的有效性进行了验证。进一步地，我们在所得数据集上微调了一个基础的大规模语言模型，并将其与几种基线进行对比。自动评价指标和人工评估均表明，ProKG Dial 显著提高了对话质量和领域特定性能，突显了其有效性和实用价值。 

---
# Reasoning Systems as Structured Processes: Foundations, Failures, and Formal Criteria 

**Title (ZH)**: 结构化的过程视角下推理系统的基础、失败与正式标准 

**Authors**: Saleh Nikooroo, Thomas Engel  

**Link**: [PDF](https://arxiv.org/pdf/2508.01763)  

**Abstract**: This paper outlines a general formal framework for reasoning systems, intended to support future analysis of inference architectures across domains. We model reasoning systems as structured tuples comprising phenomena, explanation space, inference and generation maps, and a principle base. The formulation accommodates logical, algorithmic, and learning-based reasoning processes within a unified structural schema, while remaining agnostic to any specific reasoning algorithm or logic system. We survey basic internal criteria--including coherence, soundness, and completeness-and catalog typical failure modes such as contradiction, incompleteness, and non-convergence. The framework also admits dynamic behaviors like iterative refinement and principle evolution. The goal of this work is to establish a foundational structure for representing and comparing reasoning systems, particularly in contexts where internal failure, adaptation, or fragmentation may arise. No specific solution architecture is proposed; instead, we aim to support future theoretical and practical investigations into reasoning under structural constraint. 

**Abstract (ZH)**: 本文提出了一种通用的形式化框架，旨在支持跨领域推理架构的未来分析。我们将推理系统建模为包含现象、解释空间、推理和生成映射以及原则基础的结构化元组。该表述在统一的结构化框架内包容逻辑、算法和基于学习的推理过程，同时对任何特定的推理算法或逻辑系统保持中立。我们概述了基本的内在标准，包括一致性、稳健性和完备性，并记录了典型失败模式，如矛盾、不完备性和非收敛性。该框架还允许动态行为，如迭代细化和原则进化。本文的目标是在可能存在内部失败、适应或碎片化的情况下，建立表示和比较推理系统的基础结构。我们没有提出特定的解决方案架构，而是旨在支持未来在结构约束下进行推理的理论和实践研究。 

---
# Implementing Cumulative Functions with Generalized Cumulative Constraints 

**Title (ZH)**: 实现累积函数的一般累积约束 

**Authors**: Pierre Schaus, Charles Thomas, Roger Kameugne  

**Link**: [PDF](https://arxiv.org/pdf/2508.01751)  

**Abstract**: Modeling scheduling problems with conditional time intervals and cumulative functions has become a common approach when using modern commercial constraint programming solvers. This paradigm enables the modeling of a wide range of scheduling problems, including those involving producers and consumers. However, it is unavailable in existing open-source solvers and practical implementation details remain undocumented. In this work, we present an implementation of this modeling approach using a single, generic global constraint called the Generalized Cumulative. We also introduce a novel time-table filtering algorithm designed to handle tasks defined on conditional time-intervals. Experimental results demonstrate that this approach, combined with the new filtering algorithm, performs competitively with existing solvers enabling the modeling of producer and consumer scheduling problems and effectively scales to large problems. 

**Abstract (ZH)**: 基于条件时间间隔和累积函数的调度问题建模已成为使用现代商业约束编程求解器的一种常见方法。本工作介绍了使用一个通用全局约束——广义累积——实现这种建模方法，并引入了一种新的时间表过滤算法以处理基于条件时间间隔的任务。实验结果表明，该方法结合新的过滤算法，在建模 producers 和 consumers 的调度问题时具有竞争力，并能够有效处理大规模问题。 

---
# Bayes-Entropy Collaborative Driven Agents for Research Hypotheses Generation and Optimization 

**Title (ZH)**: 贝叶斯熵协作驱动代理用于研究假设生成与优化 

**Authors**: Shiyang Duan, Yuan Tian, Qi Bing, Xiaowei Shao  

**Link**: [PDF](https://arxiv.org/pdf/2508.01746)  

**Abstract**: The exponential growth of scientific knowledge has made the automated generation of scientific hypotheses that combine novelty, feasibility, and research value a core challenge. Existing methods based on large language models fail to systematically model the inherent in hypotheses or incorporate the closed-loop feedback mechanisms crucial for refinement. This paper proposes a multi-agent collaborative framework called HypoAgents, which for the first time integrates Bayesian reasoning with an information entropy-driven search mechanism across three stages-hypotheses generation, evidence validation, and hypotheses Refinement-to construct an iterative closed-loop simulating scientists' cognitive processes. Specifically, the framework first generates an initial set of hypotheses through diversity sampling and establishes prior beliefs based on a composite novelty-relevance-feasibility (N-R-F) score. It then employs etrieval-augmented generation (RAG) to gather external literature evidence, updating the posterior probabilities of hypotheses using Bayes' theorem. Finally, it identifies high-uncertainty hypotheses using information entropy $H = - \sum {{p_i}\log {p_i}}$ and actively refines them, guiding the iterative optimization of the hypothesis set toward higher quality and confidence. Experimental results on the ICLR 2025 conference real-world research question dataset (100 research questions) show that after 12 optimization iterations, the average ELO score of generated hypotheses improves by 116.3, surpassing the benchmark of real paper abstracts by 17.8, while the framework's overall uncertainty, as measured by Shannon entropy, decreases significantly by 0.92. This study presents an interpretable probabilistic reasoning framework for automated scientific discovery, substantially improving the quality and reliability of machine-generated research hypotheses. 

**Abstract (ZH)**: 科学知识的指数增长使得结合新颖性、可行性和研究价值的自动科学假设生成成为核心挑战。现有基于大语言模型的方法未能系统地建模假设中的固有属性或整合关键的闭环反馈机制以进行优化。本文提出了一种多智能体协作框架HypoAgents，首次将贝叶斯推理与信息熵驱动的搜索机制整合到三个阶段——假设生成、证据验证和假设优化中，构建了一个模拟科学家认知过程的迭代闭环。具体而言，该框架首先通过多样性的抽样生成初始假设集，并基于复合新颖性-相关性-可行性（N-R-F）评分建立先验信念。然后使用检索增强生成（RAG）收集外部文献证据，并使用贝叶斯定理更新假设的后验概率。最后，利用信息熵 $H = - \sum {{p_i}\log {p_i}}$ 识别高不确定性假设并主动优化它们，从而引导假设集的迭代优化以获得更高的质量和可信度。在ICLR 2025会议真实世界研究问题数据集中（包含100个研究问题）的实验结果显示，在12次优化迭代后，生成假设的平均ELO评分提高了116.3，比现实论文摘要基准高出17.8，同时框架整体不确定性，以香农熵衡量，显著降低了0.92。本研究提出了一种可解释的概率推理框架，显著提高了机器生成研究假设的质量和可靠性。 

---
# DeepVIS: Bridging Natural Language and Data Visualization Through Step-wise Reasoning 

**Title (ZH)**: DeepVIS: 通过逐步推理连接自然语言与数据可视化 

**Authors**: Zhihao Shuai, Boyan Li, Siyu Yan, Yuyu Luo, Weikai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01700)  

**Abstract**: Although data visualization is powerful for revealing patterns and communicating insights, creating effective visualizations requires familiarity with authoring tools and often disrupts the analysis flow. While large language models show promise for automatically converting analysis intent into visualizations, existing methods function as black boxes without transparent reasoning processes, which prevents users from understanding design rationales and refining suboptimal outputs. To bridge this gap, we propose integrating Chain-of-Thought (CoT) reasoning into the Natural Language to Visualization (NL2VIS) pipeline. First, we design a comprehensive CoT reasoning process for NL2VIS and develop an automatic pipeline to equip existing datasets with structured reasoning steps. Second, we introduce nvBench-CoT, a specialized dataset capturing detailed step-by-step reasoning from ambiguous natural language descriptions to finalized visualizations, which enables state-of-the-art performance when used for model fine-tuning. Third, we develop DeepVIS, an interactive visual interface that tightly integrates with the CoT reasoning process, allowing users to inspect reasoning steps, identify errors, and make targeted adjustments to improve visualization outcomes. Quantitative benchmark evaluations, two use cases, and a user study collectively demonstrate that our CoT framework effectively enhances NL2VIS quality while providing insightful reasoning steps to users. 

**Abstract (ZH)**: 尽管数据可视化在揭示模式和传达洞见方面具有强大功能，但创建有效的可视化往往需要熟悉作者工具，并且往往会中断分析流程。虽然大规模语言模型显示出将分析意图自动转换为可视化图的潜力，但现有方法作为黑盒子运行，缺乏透明的推理过程，这使得用户无法理解设计理由并改进不理想的输出。为了解决这一问题，我们提出了将链式思考（CoT）推理集成到自然语言到可视化（NL2VIS）管道中的方法。首先，我们设计了一套全面的CoT推理过程并开发了一个自动管道，用于为现有数据集添加结构化的推理步骤。其次，我们引入了nvBench-CoT，这是一种专门的数据集，可以捕捉从模糊自然语言描述到最终可视化的过程中的详细步骤，这使得在模型微调时可以实现最先进的性能。第三，我们开发了DeepVIS，这是一种与CoT推理过程紧密集成的交互式可视化界面，允许用户检查推理步骤、识别错误并进行针对性调整以改进可视化结果。定量基准评估、两个使用案例和用户研究共同证明了我们的CoT框架在提高NL2VIS质量的同时为用户提供了解释性的推理步骤。 

---
# SURE-Med: Systematic Uncertainty Reduction for Enhanced Reliability in Medical Report Generation 

**Title (ZH)**: SURE-Med: 系统性不确定性减少以提高医学报告生成的可靠性 

**Authors**: Yuhang Gu, Xingyu Hu, Yuyu Fan, Xulin Yan, Longhuan Xu, Peng peng  

**Link**: [PDF](https://arxiv.org/pdf/2508.01693)  

**Abstract**: Automated medical report generation (MRG) holds great promise for reducing the heavy workload of radiologists. However, its clinical deployment is hindered by three major sources of uncertainty. First, visual uncertainty, caused by noisy or incorrect view annotations, compromises feature extraction. Second, label distribution uncertainty, stemming from long-tailed disease prevalence, biases models against rare but clinically critical conditions. Third, contextual uncertainty, introduced by unverified historical reports, often leads to factual hallucinations. These challenges collectively limit the reliability and clinical trustworthiness of MRG systems. To address these issues, we propose SURE-Med, a unified framework that systematically reduces uncertainty across three critical dimensions: visual, distributional, and contextual. To mitigate visual uncertainty, a Frontal-Aware View Repair Resampling module corrects view annotation errors and adaptively selects informative features from supplementary views. To tackle label distribution uncertainty, we introduce a Token Sensitive Learning objective that enhances the modeling of critical diagnostic sentences while reweighting underrepresented diagnostic terms, thereby improving sensitivity to infrequent conditions. To reduce contextual uncertainty, our Contextual Evidence Filter validates and selectively incorporates prior information that aligns with the current image, effectively suppressing hallucinations. Extensive experiments on the MIMIC-CXR and IU-Xray benchmarks demonstrate that SURE-Med achieves state-of-the-art performance. By holistically reducing uncertainty across multiple input modalities, SURE-Med sets a new benchmark for reliability in medical report generation and offers a robust step toward trustworthy clinical decision support. 

**Abstract (ZH)**: 自动化医学报告生成（MRG）在减轻放射科医生工作负担方面大有前景。然而，其临床部署受到三大不确定性来源的阻碍。首先，由噪声或错误的视角注释引起的视觉不确定性会损害特征提取。其次，由长尾疾病分布引起的标签分布不确定性会使模型偏向罕见但临床上至关重要的条件。第三，由未经验证的历史报告引入的上下文不确定性往往会导致事实性幻觉。这些挑战共同限制了MRG系统的可靠性和临床可信度。为应对这些问题，我们提出了一种统一框架SURE-Med，系统地减少了这三个关键维度上的不确定性：视觉、分布性和上下文不确定性。为减轻视觉不确定性，通过引入前景意识视角修复重采样模块来纠正视角注释错误，并自适应地从补充视角中选择具有信息性的特征。为应对标签分布不确定性，我们引入了一种敏感标记学习目标，该目标提升了对关键诊断句子的建模能力，同时重新加权未充分代表的诊断术语，从而提高了对不常见条件的敏感性。为减少上下文不确定性，我们的上下文证据过滤器验证并选择性地整合与当前图像一致的先验信息，有效抑制了幻觉现象。在MIMIC-CXR和IU-Xray基准测试上的广泛实验表明，SURE-Med取得了最先进的性能。通过对多种输入模态的整体不确定性减少，SURE-Med为医学报告生成的可靠性设定了新的基准，并为可靠的临床决策支持迈出了坚实的一步。 

---
# One Subgoal at a Time: Zero-Shot Generalization to Arbitrary Linear Temporal Logic Requirements in Multi-Task Reinforcement Learning 

**Title (ZH)**: 一次子目标一步：多任务强化学习中对任意线性时序逻辑要求的零样本泛化 

**Authors**: Zijian Guo, İlker Işık, H. M. Sabbir Ahmad, Wenchao Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.01561)  

**Abstract**: Generalizing to complex and temporally extended task objectives and safety constraints remains a critical challenge in reinforcement learning (RL). Linear temporal logic (LTL) offers a unified formalism to specify such requirements, yet existing methods are limited in their abilities to handle nested long-horizon tasks and safety constraints, and cannot identify situations when a subgoal is not satisfiable and an alternative should be sought. In this paper, we introduce GenZ-LTL, a method that enables zero-shot generalization to arbitrary LTL specifications. GenZ-LTL leverages the structure of Büchi automata to decompose an LTL task specification into sequences of reach-avoid subgoals. Contrary to the current state-of-the-art method that conditions on subgoal sequences, we show that it is more effective to achieve zero-shot generalization by solving these reach-avoid problems \textit{one subgoal at a time} through proper safe RL formulations. In addition, we introduce a novel subgoal-induced observation reduction technique that can mitigate the exponential complexity of subgoal-state combinations under realistic assumptions. Empirical results show that GenZ-LTL substantially outperforms existing methods in zero-shot generalization to unseen LTL specifications. 

**Abstract (ZH)**: 广义化到复杂的和时序扩展的任务目标及安全约束仍然是强化学习（RL）中的一个关键挑战。GenZ-LTL：一种实现任意LTL规范零样本广义化的方法 

---
# Empowering Tabular Data Preparation with Language Models: Why and How? 

**Title (ZH)**: 利用语言模型赋能表格数据准备：为什么以及如何实现？ 

**Authors**: Mengshi Chen, Yuxiang Sun, Tengchao Li, Jianwei Wang, Kai Wang, Xuemin Lin, Ying Zhang, Wenjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01556)  

**Abstract**: Data preparation is a critical step in enhancing the usability of tabular data and thus boosts downstream data-driven tasks. Traditional methods often face challenges in capturing the intricate relationships within tables and adapting to the tasks involved. Recent advances in Language Models (LMs), especially in Large Language Models (LLMs), offer new opportunities to automate and support tabular data preparation. However, why LMs suit tabular data preparation (i.e., how their capabilities match task demands) and how to use them effectively across phases still remain to be systematically explored. In this survey, we systematically analyze the role of LMs in enhancing tabular data preparation processes, focusing on four core phases: data acquisition, integration, cleaning, and transformation. For each phase, we present an integrated analysis of how LMs can be combined with other components for different preparation tasks, highlight key advancements, and outline prospective pipelines. 

**Abstract (ZH)**: 语言模型在增强表格数据准备过程中的作用及其应用：一项涵盖数据获取、集成、清洗和转换四个核心阶段的系统调研 

---
# WinkTPG: An Execution Framework for Multi-Agent Path Finding Using Temporal Reasoning 

**Title (ZH)**: WinkTPG：基于时间推理的多Agent路径规划执行框架 

**Authors**: Jingtian Yan, Stephen F. Smith, Jiaoyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.01495)  

**Abstract**: Planning collision-free paths for a large group of agents is a challenging problem with numerous real-world applications. While recent advances in Multi-Agent Path Finding (MAPF) have shown promising progress, standard MAPF algorithms rely on simplified kinodynamic models, preventing agents from directly following the generated MAPF plan. To bridge this gap, we propose kinodynamic Temporal Plan Graph Planning (kTPG), a multi-agent speed optimization algorithm that efficiently refines a MAPF plan into a kinodynamically feasible plan while accounting for uncertainties and preserving collision-freeness. Building on kTPG, we propose Windowed kTPG (WinkTPG), a MAPF execution framework that incrementally refines MAPF plans using a window-based mechanism, dynamically incorporating agent information during execution to reduce uncertainty. Experiments show that WinkTPG can generate speed profiles for up to 1,000 agents in 1 second and improves solution quality by up to 51.7% over existing MAPF execution methods. 

**Abstract (ZH)**: 大规模群体代理的碰撞免费路径规划是一个具有诸多实际应用的研究挑战。虽然最近在多代理路径finding (MAPF) 方面取得了有前景的进展，但标准的MAPF算法依赖于简化的动力学模型，导致代理无法直接遵循生成的MAPF计划。为解决这一问题，我们提出了动力学时序计划图规划（kTPG），这是一种多代理速度优化算法，可以高效地将MAPF计划细化为动力学可行的计划，同时考虑不确定性并保持碰撞免费。基于kTPG，我们提出了窗口化kTPG（WinkTPG），这是一种通过窗口机制逐步细化MAPF计划的多代理路径finding执行框架，在执行过程中动态整合代理信息以降低不确定性。实验结果显示，WinkTPG可以在1秒内为多达1,000个代理生成速度轮廓，并比现有MAPF执行方法提高了解决方案质量高达51.7%。 

---
# CARGO: A Co-Optimization Framework for EV Charging and Routing in Goods Delivery Logistics 

**Title (ZH)**: CARGO: 面向货物配送物流的电动汽车充电与路由联合优化框架 

**Authors**: Arindam Khanda, Anurag Satpathy, Amit Jha, Sajal K. Das  

**Link**: [PDF](https://arxiv.org/pdf/2508.01476)  

**Abstract**: With growing interest in sustainable logistics, electric vehicle (EV)-based deliveries offer a promising alternative for urban distribution. However, EVs face challenges due to their limited battery capacity, requiring careful planning for recharging. This depends on factors such as the charging point (CP) availability, cost, proximity, and vehicles' state of charge (SoC). We propose CARGO, a framework addressing the EV-based delivery route planning problem (EDRP), which jointly optimizes route planning and charging for deliveries within time windows. After proving the problem's NP-hardness, we propose a mixed integer linear programming (MILP)-based exact solution and a computationally efficient heuristic method. Using real-world datasets, we evaluate our methods by comparing the heuristic to the MILP solution, and benchmarking it against baseline strategies, Earliest Deadline First (EDF) and Nearest Delivery First (NDF). The results show up to 39% and 22% reductions in the charging cost over EDF and NDF, respectively, while completing comparable deliveries. 

**Abstract (ZH)**: 基于电动汽车的城市配送路径与充电优化研究 

---
# $R^2$-CoD: Understanding Text-Graph Complementarity in Relational Reasoning via Knowledge Co-Distillation 

**Title (ZH)**: $R^2$-CoD: 通过知识协同提炼理解文本-图形在关系推理中的互补性 

**Authors**: Zhen Wu, Ritam Dutt, Luke M. Breitfeller, Armineh Nourbakhsh, Siddharth Parekh, Carolyn Rosé  

**Link**: [PDF](https://arxiv.org/pdf/2508.01475)  

**Abstract**: Relational reasoning lies at the core of many NLP tasks, drawing on complementary signals from text and graphs. While prior research has investigated how to leverage this dual complementarity, a detailed and systematic understanding of text-graph interplay and its effect on hybrid models remains underexplored. We take an analysis-driven approach to investigate text-graph representation complementarity via a unified architecture that supports knowledge co-distillation (CoD). We explore five tasks involving relational reasoning that differ in how text and graph structures encode the information needed to solve that task. By tracking how these dual representations evolve during training, we uncover interpretable patterns of alignment and divergence, and provide insights into when and why their integration is beneficial. 

**Abstract (ZH)**: 关系推理是许多自然语言处理任务的核心，它依赖于文本和图互补信号的利用。尽管此前的研究探讨了如何利用这种双重互补性，但文本-图相互作用的详细系统理解及其对混合模型的影响仍待进一步探索。我们通过支持知识共提炼（CoD）的统一架构，采用分析驱动的方法来研究文本-图表示的互补性。我们探索了五个涉及关系推理的任务，这些任务在文本和图结构如何编码解决任务所需信息方面存在差异。通过跟踪这些双重表示在训练过程中如何演变，我们发现可解释的对齐和偏离模式，并提供了关于在何时以及为何整合它们是有益的见解。 

---
# Relation-Aware LNN-Transformer for Intersection-Centric Next-Step Prediction 

**Title (ZH)**: 基于关系的LNN- Transformer用于以交叉口为中心的下一步预测 

**Authors**: Zhehong Ren, Tianluo Zhang, Yiheng Lu, Yushen Liang, Promethee Spathis  

**Link**: [PDF](https://arxiv.org/pdf/2508.01368)  

**Abstract**: Next-step location prediction plays a pivotal role in modeling human mobility, underpinning applications from personalized navigation to strategic urban planning. However, approaches that assume a closed world - restricting choices to a predefined set of points of interest (POIs) - often fail to capture exploratory or target-agnostic behavior and the topological constraints of urban road networks. Hence, we introduce a road-node-centric framework that represents road-user trajectories on the city's road-intersection graph, thereby relaxing the closed-world constraint and supporting next-step forecasting beyond fixed POI sets. To encode environmental context, we introduce a sector-wise directional POI aggregation that produces compact features capturing distance, bearing, density and presence cues. By combining these cues with structural graph embeddings, we obtain semantically grounded node representations. For sequence modeling, we integrate a Relation-Aware LNN-Transformer - a hybrid of a Continuous-time Forgetting Cell CfC-LNN and a bearing-biased self-attention module - to capture both fine-grained temporal dynamics and long-range spatial dependencies. Evaluated on city-scale road-user trajectories, our model outperforms six state-of-the-art baselines by up to 17 percentage points in accuracy at one hop and 10 percentage points in MRR, and maintains high resilience under noise, losing only 2.4 percentage points in accuracy at one under 50 meter GPS perturbation and 8.9 percentage points in accuracy at one hop under 25 percent POI noise. 

**Abstract (ZH)**: 基于道路节点的下一步位置预测框架：超越固定兴趣点集的路网用户轨迹建模 

---
# Idempotent Equilibrium Analysis of Hybrid Workflow Allocation: A Mathematical Schema for Future Work 

**Title (ZH)**: 混合工作流分配中的幂等均衡分析：面向未来工作的数学框架 

**Authors**: Faruk Alpay, Bugra Kilictas, Taylan Alpay, Hamdi Alakkad  

**Link**: [PDF](https://arxiv.org/pdf/2508.01323)  

**Abstract**: The rapid advance of large-scale AI systems is reshaping how work is divided between people and machines. We formalise this reallocation as an iterated task-delegation map and show that--under broad, empirically grounded assumptions--the process converges to a stable idempotent equilibrium in which every task is performed by the agent (human or machine) with enduring comparative advantage. Leveraging lattice-theoretic fixed-point tools (Tarski and Banach), we (i) prove existence of at least one such equilibrium and (ii) derive mild monotonicity conditions that guarantee uniqueness. In a stylised continuous model the long-run automated share takes the closed form $x^* = \alpha / (\alpha + \beta)$, where $\alpha$ captures the pace of automation and $\beta$ the rate at which new, human-centric tasks appear; hence full automation is precluded whenever $\beta > 0$. We embed this analytic result in three complementary dynamical benchmarks--a discrete linear update, an evolutionary replicator dynamic, and a continuous Beta-distributed task spectrum--each of which converges to the same mixed equilibrium and is reproducible from the provided code-free formulas. A 2025-to-2045 simulation calibrated to current adoption rates projects automation rising from approximately 10% of work to approximately 65%, leaving a persistent one-third of tasks to humans. We interpret that residual as a new profession of workflow conductor: humans specialise in assigning, supervising and integrating AI modules rather than competing with them. Finally, we discuss implications for skill development, benchmark design and AI governance, arguing that policies which promote "centaur" human-AI teaming can steer the economy toward the welfare-maximising fixed point. 

**Abstract (ZH)**: 大规模AI系统的迅速发展正在重新塑造人类和机器之间的劳动分工。我们这种重新分配形式化为迭代的任务委派映射，并表明在广泛的经验基础上，该过程收敛到一个稳定的恒等平衡点，在此点上每个任务都由具有持久比较优势的代理（人类或机器）执行。利用格论不动点工具（塔斯基和巴纳赫），我们（i）证明至少存在一个这样的平衡点，并（ii）推导出保证唯一性的温和单调条件。在一个简化的连续模型中，长期的自动化份额以闭形式 $x^* = \alpha / (\alpha + \beta)$ 表示，其中 $\alpha$ 表示自动化速度，$\beta$ 表示新的人类中心任务出现的速率；因此，只要 $\beta > 0$，就排除了完全自动化的可能性。我们将这一分析结果嵌入三个互补的动力学基准中——离散线性更新、进化复制动态和连续的Beta分布任务谱——每个基准都收敛到相同的混合平衡点，并且可以从提供的代码自由公式中重现。根据当前采用率调整后的2025年至2045年模拟预测，自动化从大约10%的工作份额上升到大约65%，留下大约三分之一的任务给人类。我们解释这个剩余部分为一种新的职业——工作流程指挥官：人类专门负责分配、监督和整合AI模块，而不是与之竞争。最后，我们讨论了技能发展、基准设计和AI治理的影响，认为促进“人马合一”的人机团队策略能够引导经济向福利最大化不动点发展。 

---
# Multi-TW: Benchmarking Multimodal Models on Traditional Chinese Question Answering in Taiwan 

**Title (ZH)**: 多模态模型在台湾传统中文问答任务上的基准测试 

**Authors**: Jui-Ming Yao, Bing-Cheng Xie, Sheng-Wei Peng, Hao-Yuan Chen, He-Rong Zheng, Bing-Jia Tan, Peter Shaojui Wang, Shun-Feng Su  

**Link**: [PDF](https://arxiv.org/pdf/2508.01274)  

**Abstract**: Multimodal Large Language Models (MLLMs) process visual, acoustic, and textual inputs, addressing the limitations of single-modality LLMs. However, existing benchmarks often overlook tri-modal evaluation in Traditional Chinese and do not consider inference latency. To address this, we introduce Multi-TW, the first Traditional Chinese benchmark for evaluating the performance and latency of any-to-any multimodal models. Multi-TW includes 900 multiple-choice questions (image and text, audio and text pairs) sourced from official proficiency tests developed with the Steering Committee for the Test of Proficiency-Huayu (SC-TOP). We evaluated various any-to-any models and vision-language models (VLMs) with audio transcription. Our results show that closed-source models generally outperform open-source ones across modalities, although open-source models can perform well in audio tasks. End-to-end any-to-any pipelines offer clear latency advantages compared to VLMs using separate audio transcription. Multi-TW presents a comprehensive view of model capabilities and highlights the need for Traditional Chinese fine-tuning and efficient multimodal architectures. 

**Abstract (ZH)**: Multimodal Large Language Models (MLLMs) 处理视觉、声学和文本输入，解决单模态 LLMs 的局限性。然而，现有的基准在传统中文方面通常忽视了三模态评估，也没有考虑推理延迟。为解决这一问题，我们引入 Multi-TW，这是首个用于评估任意到任意的多模态模型性能和延迟的传统中文基准。Multi-TW 包含 900 个多选题（图像和文本、声学和文本配对），这些问题源自由 Test of Proficiency-Huayu 指导委员会（SC-TOP）开发的官方 proficiency 测试。我们评估了各种任意到任意模型和视觉语言模型（VLMs）的声学转录。结果显示，闭源模型在各模态中通常优于开源模型，尽管开源模型在声学任务中也能表现出色。端到端的任意到任意管道与使用单独声学转录的 VLMs 相比，在延迟方面具有明显的优势。Multi-TW 提供了模型能力的全面视角，并强调了传统中文微调和高效多模态架构的需求。 

---
# Calibrated Prediction Set in Fault Detection with Risk Guarantees via Significance Tests 

**Title (ZH)**: 故障检测中具有风险保证的校准预测集通过显著性检验 

**Authors**: Mingchen Mei, Yi Li, YiYao Qian, Zijun Jia  

**Link**: [PDF](https://arxiv.org/pdf/2508.01208)  

**Abstract**: Fault detection is crucial for ensuring the safety and reliability of modern industrial systems. However, a significant scientific challenge is the lack of rigorous risk control and reliable uncertainty quantification in existing diagnostic models, particularly when facing complex scenarios such as distributional shifts. To address this issue, this paper proposes a novel fault detection method that integrates significance testing with the conformal prediction framework to provide formal risk guarantees. The method transforms fault detection into a hypothesis testing task by defining a nonconformity measure based on model residuals. It then leverages a calibration dataset to compute p-values for new samples, which are used to construct prediction sets mathematically guaranteed to contain the true label with a user-specified probability, $1-\alpha$. Fault classification is subsequently performed by analyzing the intersection of the constructed prediction set with predefined normal and fault label sets. Experimental results on cross-domain fault diagnosis tasks validate the theoretical properties of our approach. The proposed method consistently achieves an empirical coverage rate at or above the nominal level ($1-\alpha$), demonstrating robustness even when the underlying point-prediction models perform poorly. Furthermore, the results reveal a controllable trade-off between the user-defined risk level ($\alpha$) and efficiency, where higher risk tolerance leads to smaller average prediction set sizes. This research contributes a theoretically grounded framework for fault detection that enables explicit risk control, enhancing the trustworthiness of diagnostic systems in safety-critical applications and advancing the field from simple point predictions to informative, uncertainty-aware outputs. 

**Abstract (ZH)**: 基于显著性检验与同态预测框架的故障检测方法研究 

---
# A Survey on Agent Workflow -- Status and Future 

**Title (ZH)**: 智能体工作流综述——现状与未来 

**Authors**: Chaojia Yu, Zihan Cheng, Hanwen Cui, Yishuo Gao, Zexu Luo, Yijin Wang, Hangbin Zheng, Yong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.01186)  

**Abstract**: In the age of large language models (LLMs), autonomous agents have emerged as a powerful paradigm for achieving general intelligence. These agents dynamically leverage tools, memory, and reasoning capabilities to accomplish user-defined goals. As agent systems grow in complexity, agent workflows-structured orchestration frameworks-have become central to enabling scalable, controllable, and secure AI behaviors. This survey provides a comprehensive review of agent workflow systems, spanning academic frameworks and industrial implementations. We classify existing systems along two key dimensions: functional capabilities (e.g., planning, multi-agent collaboration, external API integration) and architectural features (e.g., agent roles, orchestration flows, specification languages). By comparing over 20 representative systems, we highlight common patterns, potential technical challenges, and emerging trends. We further address concerns related to workflow optimization strategies and security. Finally, we outline open problems such as standardization and multimodal integration, offering insights for future research at the intersection of agent design, workflow infrastructure, and safe automation. 

**Abstract (ZH)**: 在大规模语言模型时代，自主代理已成为实现通用智能的强大范式。这些代理动态利用工具、记忆和推理能力以实现用户定义的目标。随着代理系统的复杂性增加，代理工作流——结构化的编排框架——已成为实现可扩展、可控和安全的AI行为的关键。本文综述了代理工作流系统，涵盖了学术框架和工业实现。我们根据两个关键维度对现有系统进行分类：功能能力（如规划、多代理协作、外部API集成）和架构特性（如代理角色、编排流程、规范语言）。通过比较超过20个代表性系统，我们突显了共同模式、潜在的技术挑战以及新兴趋势。我们进一步探讨了工作流优化策略和安全性的相关问题。最后，我们概述了标准化和多模态集成等开放问题，为代理设计、工作流基础设施和安全自动化交叉领域的未来研究提供了见解。 

---
# H2C: Hippocampal Circuit-inspired Continual Learning for Lifelong Trajectory Prediction in Autonomous Driving 

**Title (ZH)**: H2C：基于海马神经回路的终身轨迹预测持续学习方法 

**Authors**: Yunlong Lin, Zirui Li, Guodong Du, Xiaocong Zhao, Cheng Gong, Xinwei Wang, Chao Lu, Jianwei Gong  

**Link**: [PDF](https://arxiv.org/pdf/2508.01158)  

**Abstract**: Deep learning (DL) has shown state-of-the-art performance in trajectory prediction, which is critical to safe navigation in autonomous driving (AD). However, most DL-based methods suffer from catastrophic forgetting, where adapting to a new distribution may cause significant performance degradation in previously learned ones. Such inability to retain learned knowledge limits their applicability in the real world, where AD systems need to operate across varying scenarios with dynamic distributions. As revealed by neuroscience, the hippocampal circuit plays a crucial role in memory replay, effectively reconstructing learned knowledge based on limited resources. Inspired by this, we propose a hippocampal circuit-inspired continual learning method (H2C) for trajectory prediction across varying scenarios. H2C retains prior knowledge by selectively recalling a small subset of learned samples. First, two complementary strategies are developed to select the subset to represent learned knowledge. Specifically, one strategy maximizes inter-sample diversity to represent the distinctive knowledge, and the other estimates the overall knowledge by equiprobable sampling. Then, H2C updates via a memory replay loss function calculated by these selected samples to retain knowledge while learning new data. Experiments based on various scenarios from the INTERACTION dataset are designed to evaluate H2C. Experimental results show that H2C reduces catastrophic forgetting of DL baselines by 22.71% on average in a task-free manner, without relying on manually informed distributional shifts. The implementation is available at this https URL. 

**Abstract (ZH)**: 基于海马电路的持续学习方法（H2C）用于跨场景的轨迹预测 

---
# Multispin Physics of AI Tipping Points and Hallucinations 

**Title (ZH)**: 多自旋物理：AI临界点和幻觉 

**Authors**: Neil F. Johnson, Frank Yingjie Huo  

**Link**: [PDF](https://arxiv.org/pdf/2508.01097)  

**Abstract**: Output from generative AI such as ChatGPT, can be repetitive and biased. But more worrying is that this output can mysteriously tip mid-response from good (correct) to bad (misleading or wrong) without the user noticing. In 2024 alone, this reportedly caused $67 billion in losses and several deaths. Establishing a mathematical mapping to a multispin thermal system, we reveal a hidden tipping instability at the scale of the AI's 'atom' (basic Attention head). We derive a simple but essentially exact formula for this tipping point which shows directly the impact of a user's prompt choice and the AI's training bias. We then show how the output tipping can get amplified by the AI's multilayer architecture. As well as helping improve AI transparency, explainability and performance, our results open a path to quantifying users' AI risk and legal liabilities. 

**Abstract (ZH)**: 生成式AI如ChatGPT的输出可能存在重复和偏见问题，更令人担忧的是，这种输出可能会神秘地在用户不知情的情况下从良好（正确）转变为不良（误导或错误）。2024年，这 reportedly 导致了670亿美元的损失和数人死亡。通过建立数学映射到多自旋热系统，我们揭示了在AI的“原子”（基本注意力头）尺度上存在一个隐藏的翻转不稳定性。我们推导出一个简单但基本上精确的公式来描述这一翻转点，它直接展示了用户提示选择和AI训练偏见的影响。我们还展示了AI多层架构如何放大输出翻转。除了提高AI的透明度、可解释性和性能，我们的结果还开辟了一条量化用户AI风险和法律责任的途径。 

---
# gpuRDF2vec -- Scalable GPU-based RDF2vec 

**Title (ZH)**: gpuRDF2vec——基于GPU的可扩展RDF2vec 

**Authors**: Martin Böckling, Heiko Paulheim  

**Link**: [PDF](https://arxiv.org/pdf/2508.01073)  

**Abstract**: Generating Knowledge Graph (KG) embeddings at web scale remains challenging. Among existing techniques, RDF2vec combines effectiveness with strong scalability. We present gpuRDF2vec, an open source library that harnesses modern GPUs and supports multi-node execution to accelerate every stage of the RDF2vec pipeline. Extensive experiments on both synthetically generated graphs and real-world benchmarks show that gpuRDF2vec achieves up to a substantial speedup over the currently fastest alternative, i.e., jRDF2vec. In a single-node setup, our walk-extraction phase alone outperforms pyRDF2vec, SparkKGML, and jRDF2vec by a substantial margin using random walks on large/ dense graphs, and scales very well to longer walks, which typically lead to better quality embeddings. Our implementation of gpuRDF2vec enables practitioners and researchers to train high-quality KG embeddings on large-scale graphs within practical time budgets and builds on top of Pytorch Lightning for the scalable word2vec implementation. 

**Abstract (ZH)**: 在Web规模下生成知识图谱（KG）嵌入仍然具有挑战性。我们介绍了gpuRDF2vec，这是一个开源库，利用现代GPU并支持多节点执行以加速RDF2vec管道中的每个阶段。在合成生成的图和真实基准上的广泛实验表明，gpuRDF2vec相比于当前最快的替代方案jRDF2vec实现了显著的速度提升。在单节点设置中，仅抽取走相比较pyRDF2vec、SparkKGML和jRDF2vec在大规模/密集图上使用随机走的方法取得了显著的优势，并且能够很好地扩展到更长的走，通常会导致更好的质量嵌入。我们对gpuRDF2vec的实现使从业者和研究者能够在实际的时间预算内训练大规模图上的高质量KG嵌入，并基于Pytorch Lightning构建以实现可扩展的word2vec实现。 

---
# ff4ERA: A new Fuzzy Framework for Ethical Risk Assessment in AI 

**Title (ZH)**: ff4ERA：一种新的模糊框架用于AI伦理风险评估 

**Authors**: Abeer Dyoub, Ivan Letteri, Francesca A. Lisi  

**Link**: [PDF](https://arxiv.org/pdf/2508.00899)  

**Abstract**: The emergence of Symbiotic AI (SAI) introduces new challenges to ethical decision-making as it deepens human-AI collaboration. As symbiosis grows, AI systems pose greater ethical risks, including harm to human rights and trust. Ethical Risk Assessment (ERA) thus becomes crucial for guiding decisions that minimize such risks. However, ERA is hindered by uncertainty, vagueness, and incomplete information, and morality itself is context-dependent and imprecise. This motivates the need for a flexible, transparent, yet robust framework for ERA. Our work supports ethical decision-making by quantitatively assessing and prioritizing multiple ethical risks so that artificial agents can select actions aligned with human values and acceptable risk levels. We introduce ff4ERA, a fuzzy framework that integrates Fuzzy Logic, the Fuzzy Analytic Hierarchy Process (FAHP), and Certainty Factors (CF) to quantify ethical risks via an Ethical Risk Score (ERS) for each risk type. The final ERS combines the FAHP-derived weight, propagated CF, and risk level. The framework offers a robust mathematical approach for collaborative ERA modeling and systematic, step-by-step analysis. A case study confirms that ff4ERA yields context-sensitive, ethically meaningful risk scores reflecting both expert input and sensor-based evidence. Risk scores vary consistently with relevant factors while remaining robust to unrelated inputs. Local sensitivity analysis shows predictable, mostly monotonic behavior across perturbations, and global Sobol analysis highlights the dominant influence of expert-defined weights and certainty factors, validating the model design. Overall, the results demonstrate ff4ERA ability to produce interpretable, traceable, and risk-aware ethical assessments, enabling what-if analyses and guiding designers in calibrating membership functions and expert judgments for reliable ethical decision support. 

**Abstract (ZH)**: 共生人工智能(SAI)的兴起为伦理决策提出了新的挑战，随着人机协作的加深，AI系统带来的伦理风险增加，包括对人权和信任的危害。因此，伦理风险评估(ERA)变得至关重要，以指导减少这些风险的决策。然而，ERA受制于不确定性、模糊性和信息不完整，而道德本身也是情境依赖和不精确的。这促使我们需要一个灵活、透明且稳健的ERA框架。我们的工作通过定量评估和优先级排序多种伦理风险，支持AI代理选择与人类价值观和可接受风险水平一致的行为。我们引入了ff4ERA，这是一个结合了模糊逻辑、模糊分析层次过程(FAHP)和确信因子(CF)的框架，用于通过伦理风险评分(ERS)量化每种风险类型的风险。最终的ERS结合了FAHP得出的权重、传播的CF和风险水平。该框架提供了协作ERA建模和系统化、按步骤分析的稳健数学方法。案例研究证实，ff4ERA能够生成情境敏感、合乎伦理意义的风险评分，既能反映专家输入，又能反映传感器数据。风险评分与相关因素一致变化，对不相关输入保持稳健。局部灵敏度分析显示在扰动下可预测的、大体单调的行为，并且全局Sobol分析突出了专家定义的权重和确信因子的主导影响，验证了模型设计。总体而言，这些结果表明ff4ERA能够生成可解释、可追溯且风险意识强的伦理评估，支持假设场景分析，并指导设计者校准隶属函数和专家判断，以提供可靠的伦理决策支持。 

---
# A Formal Framework for the Definition of 'State': Hierarchical Representation and Meta-Universe Interpretation 

**Title (ZH)**: 形式化的“状态”定义框架：层级表示与元宇宙解释 

**Authors**: Kei Itoh  

**Link**: [PDF](https://arxiv.org/pdf/2508.00853)  

**Abstract**: This study aims to reinforce the theoretical foundation for diverse systems--including the axiomatic definition of intelligence--by introducing a mathematically rigorous and unified formal structure for the concept of 'state,' which has long been used without consensus or formal clarity. First, a 'hierarchical state grid' composed of two axes--state depth and mapping hierarchy--is proposed to provide a unified notational system applicable across mathematical, physical, and linguistic domains. Next, the 'Intermediate Meta-Universe (IMU)' is introduced to enable explicit descriptions of definers (ourselves) and the languages we use, thereby allowing conscious meta-level operations while avoiding self-reference and logical inconsistency. Building on this meta-theoretical foundation, this study expands inter-universal theory beyond mathematics to include linguistic translation and agent integration, introducing the conceptual division between macrocosm-inter-universal and microcosm-inter-universal operations for broader expressivity. Through these contributions, this paper presents a meta-formal logical framework--grounded in the principle of definition = state--that spans time, language, agents, and operations, providing a mathematically robust foundation applicable to the definition of intelligence, formal logic, and scientific theory at large. 

**Abstract (ZH)**: 本研究旨在通过引入严格且统一的形式结构来强化多样系统的基础理论——包括智能的公理定义——该结构为长期缺乏共识和形式清晰的概念“状态”提供支持。首先，提出了一种由状态深度和映射层次构成的“层级状态格”，以提供适用于数学、物理和语言领域的一致符号系统。接着，引入“中介元宇宙（IMU）”，以明确描述定义者及其使用的语言，从而允许元层次操作并避免自我引用和逻辑不一致。在此元理论基础上，本研究将跨宇宙理论扩展至语言翻译和代理集成领域，引入宏观跨宇宙和微观跨宇宙操作的概念，以扩展表达能力。通过这些贡献，本文提出了一种根植于定义=状态原则的元形式逻辑框架，该框架涵盖了时间、语言、代理和操作，提供了一种适用于智能、形式逻辑和科学理论整体的数学稳健基础。 

---
# An Efficient Continuous-Time MILP for Integrated Aircraft Hangar Scheduling and Layout 

**Title (ZH)**: 一种高效的连续时间混合整数线性规划方法，用于飞机库集成调度与布局规划 

**Authors**: Shayan Farhang Pazhooh, Hossein Shams Shemirani  

**Link**: [PDF](https://arxiv.org/pdf/2508.02640)  

**Abstract**: Efficient management of aircraft maintenance hangars is a critical operational challenge, involving complex, interdependent decisions regarding aircraft scheduling and spatial allocation. This paper introduces a novel continuous-time mixed-integer linear programming (MILP) model to solve this integrated spatio-temporal problem. By treating time as a continuous variable, our formulation overcomes the scalability limitations of traditional discrete-time approaches. The performance of the exact model is benchmarked against a constructive heuristic, and its practical applicability is demonstrated through a custom-built visualization dashboard. Computational results are compelling: the model solves instances with up to 25 aircraft to proven optimality, often in mere seconds, and for large-scale cases of up to 40 aircraft, delivers high-quality solutions within known optimality gaps. In all tested scenarios, the resulting solutions consistently and significantly outperform the heuristic, which highlights the framework's substantial economic benefits and provides valuable managerial insights into the trade-off between solution time and optimality. 

**Abstract (ZH)**: 高效的机库管理是关键的操作挑战，涉及复杂的、相互依赖的飞机排程和空间分配决策。本文引入了一种新的连续时间混合整数线性规划（MILP）模型来解决这一集成的空间-时间问题。通过将时间视为连续变量，我们的建模方法克服了传统离散时间方法的可扩展性限制。精确模型的性能与构造性启发式方法进行了基准测试，并通过自建的可视化仪表板展示了其实用性。计算结果表明：该模型在几秒钟内即可解决多达25架飞机的实例，并且在多达40架飞机的大规模情况下，能够在已知的最优性差距内提供高质量的解决方案。在所有测试场景中，所得解始终且显著优于启发式方法，这突显了该框架的显著经济效益，并提供了有关解的时间与最优性之间权衡关系的宝贵管理见解。 

---
# AutoML-Med: A Framework for Automated Machine Learning in Medical Tabular Data 

**Title (ZH)**: AutoML-Med：一种医疗表格数据的自动化机器学习框架 

**Authors**: Riccardo Francia, Maurizio Leone, Giorgio Leonardi, Stefania Montani, Marzio Pennisi, Manuel Striani, Sandra D'Alfonso  

**Link**: [PDF](https://arxiv.org/pdf/2508.02625)  

**Abstract**: Medical datasets are typically affected by issues such as missing values, class imbalance, a heterogeneous feature types, and a high number of features versus a relatively small number of samples, preventing machine learning models from obtaining proper results in classification and regression tasks. This paper introduces AutoML-Med, an Automated Machine Learning tool specifically designed to address these challenges, minimizing user intervention and identifying the optimal combination of preprocessing techniques and predictive models. AutoML-Med's architecture incorporates Latin Hypercube Sampling (LHS) for exploring preprocessing methods, trains models using selected metrics, and utilizes Partial Rank Correlation Coefficient (PRCC) for fine-tuned optimization of the most influential preprocessing steps. Experimental results demonstrate AutoML-Med's effectiveness in two different clinical settings, achieving higher balanced accuracy and sensitivity, which are crucial for identifying at-risk patients, compared to other state-of-the-art tools. AutoML-Med's ability to improve prediction results, especially in medical datasets with sparse data and class imbalance, highlights its potential to streamline Machine Learning applications in healthcare. 

**Abstract (ZH)**: 医学数据集通常受到缺失值、类别不平衡、异质特征类型以及特征数量远多于样本数量等问题的影响，这会阻碍机器学习模型在分类和回归任务中获得适当的结果。本文介绍了AutoML-Med，一种针对这些挑战而设计的自动化机器学习工具，旨在最小化用户干预并识别最优的预处理技术与预测模型组合。AutoML-Med的架构包括拉丁超立方采样（LHS）以探索预处理方法，使用选定的评估指标训练模型，并通过部分秩相关系数（PRCC）对影响最大的预处理步骤进行精细化优化。实验结果表明，与现有的先进工具相比，AutoML-Med 在两种不同的临床设置中更有效地提高了平衡准确率和敏感性，这对于识别高风险患者至关重要。AutoML-Med在处理稀疏数据和类别不平衡的医学数据集时的能力提升，突显了其在医疗健康领域机器学习应用中简化流程的潜力。 

---
# Entity Representation Learning Through Onsite-Offsite Graph for Pinterset Ads 

**Title (ZH)**: 基于现场-离场图的实体表示学习在Pinterest广告中的应用 

**Authors**: Jiayin Jin, Zhimeng Pan, Yang Tang, Jiarui Feng, Kungang Li, Chongyuan Xiang, Jiacheng Li, Runze Su, Siping Ji, Han Sun, Ling Leng, Prathibha Deshikachar  

**Link**: [PDF](https://arxiv.org/pdf/2508.02609)  

**Abstract**: Graph Neural Networks (GNN) have been extensively applied to industry recommendation systems, as seen in models like GraphSage\cite{GraphSage}, TwHIM\cite{TwHIM}, LiGNN\cite{LiGNN} etc. In these works, graphs were constructed based on users' activities on the platforms, and various graph models were developed to effectively learn node embeddings. In addition to users' onsite activities, their offsite conversions are crucial for Ads models to capture their shopping interest. To better leverage offsite conversion data and explore the connection between onsite and offsite activities, we constructed a large-scale heterogeneous graph based on users' onsite ad interactions and opt-in offsite conversion activities. Furthermore, we introduced TransRA (TransR\cite{TransR} with Anchors), a novel Knowledge Graph Embedding (KGE) model, to more efficiently integrate graph embeddings into Ads ranking models. However, our Ads ranking models initially struggled to directly incorporate Knowledge Graph Embeddings (KGE), and only modest gains were observed during offline experiments. To address this challenge, we employed the Large ID Embedding Table technique and innovated an attention based KGE finetuning approach within the Ads ranking models. As a result, we observed a significant AUC lift in Click-Through Rate (CTR) and Conversion Rate (CVR) prediction models. Moreover, this framework has been deployed in Pinterest's Ads Engagement Model and contributed to $2.69\%$ CTR lift and $1.34\%$ CPC reduction. We believe the techniques presented in this paper can be leveraged by other large-scale industrial models. 

**Abstract (ZH)**: 图神经网络（GNN）在工业推荐系统中的广泛应用：基于用户平台活动的大规模异构图构建与广告 ranking 模型中的知识图嵌入融合研究 

---
# Explainable AI for Automated User-specific Feedback in Surgical Skill Acquisition 

**Title (ZH)**: 可解释的人工智能在手术技能获取中自动实现用户特定反馈 

**Authors**: Catalina Gomez, Lalithkumar Seenivasan, Xinrui Zou, Jeewoo Yoon, Sirui Chu, Ariel Leong, Patrick Kramer, Yu-Chun Ku, Jose L. Porras, Alejandro Martin-Gomez, Masaru Ishii, Mathias Unberath  

**Link**: [PDF](https://arxiv.org/pdf/2508.02593)  

**Abstract**: Traditional surgical skill acquisition relies heavily on expert feedback, yet direct access is limited by faculty availability and variability in subjective assessments. While trainees can practice independently, the lack of personalized, objective, and quantitative feedback reduces the effectiveness of self-directed learning. Recent advances in computer vision and machine learning have enabled automated surgical skill assessment, demonstrating the feasibility of automatic competency evaluation. However, it is unclear whether such Artificial Intelligence (AI)-driven feedback can contribute to skill acquisition. Here, we examine the effectiveness of explainable AI (XAI)-generated feedback in surgical training through a human-AI study. We create a simulation-based training framework that utilizes XAI to analyze videos and extract surgical skill proxies related to primitive actions. Our intervention provides automated, user-specific feedback by comparing trainee performance to expert benchmarks and highlighting deviations from optimal execution through understandable proxies for actionable guidance. In a prospective user study with medical students, we compare the impact of XAI-guided feedback against traditional video-based coaching on task outcomes, cognitive load, and trainees' perceptions of AI-assisted learning. Results showed improved cognitive load and confidence post-intervention. While no differences emerged between the two feedback types in reducing performance gaps or practice adjustments, trends in the XAI group revealed desirable effects where participants more closely mimicked expert practice. This work encourages the study of explainable AI in surgical education and the development of data-driven, adaptive feedback mechanisms that could transform learning experiences and competency assessment. 

**Abstract (ZH)**: 传统外科技能获取高度依赖专家反馈，但由于师资 availability 限制和主观评估的差异性，直接访问受限。尽管学员可以独立练习，但在缺乏个性化、客观和定量反馈的情况下，自我导向学习的效果受到影响。近期计算机视觉和机器学习的进步使自动化外科技能评估成为可能，展示了自动技能评估的可行性。然而，尚不清楚此类基于人工智能（AI）的反馈是否能促进技能获取。在这里，我们通过一项人机研究，评估可解释AI（XAI）生成反馈在外科训练中的有效性。我们创建了一个基于模拟的培训框架，利用XAI分析视频，提取与基本动作相关的外科技能代理指标。我们的干预措施通过可理解的反馈代理提供自动化的、用户特定的反馈，将学员的表现与专家基准进行比较，并通过突出显示与最优执行的偏差来提供可操作的指导。在一项前瞻性用户研究中，我们比较了XAI引导反馈与基于视频的传统教练对任务结果、认知负荷以及学员对AI辅助学习的感知的影响。结果表明，干预后认知负荷和信心有所提高。虽然两种反馈类型在减少性能差距或实践调整方面没有显现出差异，但XAI组的趋势显示，参与者模仿专家实践的效果更为理想。这项工作鼓励在外科教育中研究可解释AI，并开发数据驱动、自适应的反馈机制，从而改变学习体验和技能评估方式。 

---
# Parameter-Efficient Routed Fine-Tuning: Mixture-of-Experts Demands Mixture of Adaptation Modules 

**Title (ZH)**: 参数高效路由微调：专家混合需求适配模块混合 

**Authors**: Yilun Liu, Yunpu Ma, Yuetian Lu, Shuo Chen, Zifeng Ding, Volker Tresp  

**Link**: [PDF](https://arxiv.org/pdf/2508.02587)  

**Abstract**: Mixture-of-Experts (MoE) benefits from a dynamic routing mechanism among their specialized experts, which existing Parameter- Efficient Fine-Tuning (PEFT) strategies fail to leverage. This motivates us to investigate whether adaptation modules themselves should incorporate routing mechanisms to align with MoE's multi-expert architecture. We analyze dynamics of core components when applying PEFT to MoE language models and examine how different routing strategies affect adaptation effectiveness. Extensive experiments adapting OLMoE-1B-7B and Mixtral-8x7B on various commonsense and math reasoning tasks validate the performance and efficiency of our routed approach. We identify the optimal configurations for different scenarios and provide empirical analyses with practical insights to facilitate better PEFT and MoE applications. 

**Abstract (ZH)**: Mixture-of-Experts (MoE)受益于其专门专家之间的动态路由机制，而现有的参数高效微调（PEFT）策略未能利用这一点。这促使我们 investigation 是否应将路由机制纳入适应模块，以与MoE的多专家架构相一致。我们分析了在MoE语言模型中应用PEFT时核心组件的动力学，并研究了不同路由策略如何影响适应效果。针对OLMoE-1B-7B和Mixtral-8x7B在各种常识和数学推理任务上的广泛实验验证了我们所提出的路由方法的性能和效率。我们确定了不同场景下的最优配置，并提供了实用见解的实证分析，以促进更好的PEFT和MoE应用。 

---
# Dynamic Feature Selection based on Rule-based Learning for Explainable Classification with Uncertainty Quantification 

**Title (ZH)**: 基于规则学习的动态特征选择方法及其在解释性分类中的不确定性量化 

**Authors**: Javier Fumanal-Idocin, Raquel Fernandez-Peralta, Javier Andreu-Perez  

**Link**: [PDF](https://arxiv.org/pdf/2508.02566)  

**Abstract**: Dynamic feature selection (DFS) offers a compelling alternative to traditional, static feature selection by adapting the selected features to each individual sample. Unlike classical methods that apply a uniform feature set, DFS customizes feature selection per sample, providing insight into the decision-making process for each case. DFS is especially significant in settings where decision transparency is key, i.e., clinical decisions; however, existing methods use opaque models, which hinder their applicability in real-life scenarios. This paper introduces a novel approach leveraging a rule-based system as a base classifier for the DFS process, which enhances decision interpretability compared to neural estimators. We also show how this method provides a quantitative measure of uncertainty for each feature query and can make the feature selection process computationally lighter by constraining the feature search space. We also discuss when greedy selection of conditional mutual information is equivalent to selecting features that minimize the difference with respect to the global model predictions. Finally, we demonstrate the competitive performance of our rule-based DFS approach against established and state-of-the-art greedy and RL methods, which are mostly considered opaque, compared to our explainable rule-based system. 

**Abstract (ZH)**: 基于规则系统的动态特征选择方法 

---
# Stakeholder Perspectives on Humanistic Implementation of Computer Perception in Healthcare: A Qualitative Study 

**Title (ZH)**: 计算机感知在医疗领域的人文实现：一项定性研究 

**Authors**: Kristin M. Kostick-Quenet, Meghan E. Hurley, Syed Ayaz, John Herrington, Casey Zampella, Julia Parish-Morris, Birkan Tunç, Gabriel Lázaro-Muñoz, J.S. Blumenthal-Barby, Eric A. Storch  

**Link**: [PDF](https://arxiv.org/pdf/2508.02550)  

**Abstract**: Computer perception (CP) technologies (digital phenotyping, affective computing and related passive sensing approaches) offer unprecedented opportunities to personalize healthcare, but provoke concerns about privacy, bias and the erosion of empathic, relationship-centered practice. A comprehensive understanding of perceived risks, benefits, and implementation challenges from those who design, deploy and experience these tools in real-world settings remains elusive. This study provides the first evidence-based account of key stakeholder perspectives on the relational, technical, and governance challenges raised by the integration of CP technologies into patient care. We conducted in-depth, semi-structured interviews with 102 stakeholders: adolescent patients and their caregivers, frontline clinicians, technology developers, and ethics, legal, policy or philosophy scholars. Transcripts underwent thematic analysis by a multidisciplinary team; reliability was enhanced through double coding and consensus adjudication. Stakeholders articulated seven interlocking concern domains: (1) trustworthiness and data integrity; (2) patient-specific relevance; (3) utility and workflow integration; (4) regulation and governance; (5) privacy and data protection; (6) direct and indirect patient harms; and (7) philosophical critiques of reductionism. To operationalize humanistic safeguards, we propose "personalized roadmaps": co-designed plans that predetermine which metrics will be monitored, how and when feedback is shared, thresholds for clinical action, and procedures for reconciling discrepancies between algorithmic inferences and lived experience. By translating these insights into personalized roadmaps, we offer a practical framework for developers, clinicians and policymakers seeking to harness continuous behavioral data while preserving the humanistic core of care. 

**Abstract (ZH)**: Computer感知技术(CP)（数字表型分析、情感计算及相关的被动感知方法）为个性化医疗保健提供了前所未有的机遇，但也引发了关于隐私、偏见和人际关系中心实践侵蚀的担忧。有关在实际应用环境中设计、部署和体验这些工具的参与者对感知风险、益处以及实施挑战的全面理解仍然匮乏。本文提供了首个基于证据的关键利益相关者对将计算机感知技术整合入患者护理所引发的关系、技术和治理挑战的观点描述。我们对102名参与者进行了深入半结构化访谈，包括青少年患者及其护理人员、一线临床医生、技术开发者以及伦理、法律、政策或哲学学者。研究结果通过对多学科团队进行主题分析获得，通过双重编码和共识裁定提升了可靠性。参与者概述了七个交织的关注领域：（1）可信度和数据完整性；（2）患者特定的相关性；（3）效用和工作流程整合；（4）监管和治理；（5）隐私和数据保护；（6）对患者的直接和间接伤害；以及（7）对还原论的哲学批评。为实现人文主义保障的有效实施，我们提出“个性化路线图”：由多方共同设计的计划，明确监测哪些指标、何时何地分享反馈、临床行动的阈值以及算法推断与生活体验之间的分歧解决程序。通过将这些见解转化为个性化路线图，我们提供了开发人员、临床医生和政策制定者在获取连续行为数据的同时保留关照核心的实用框架。 

---
# The KG-ER Conceptual Schema Language 

**Title (ZH)**: KG-ER 概念模式语言 

**Authors**: Enrico Franconi, Benoît Groz, Jan Hidders, Nina Pardal, Sławek Staworko, Jan Van den Bussche, Piotr Wieczorek  

**Link**: [PDF](https://arxiv.org/pdf/2508.02548)  

**Abstract**: We propose KG-ER, a conceptual schema language for knowledge graphs that describes the structure of knowledge graphs independently of their representation (relational databases, property graphs, RDF) while helping to capture the semantics of the information stored in a knowledge graph. 

**Abstract (ZH)**: 我们提出了一种名为KG-ER的概念模式语言，用于知识图谱，独立于知识图谱的表现形式（关系数据库、属性图、RDF）来描述知识图谱的结构，同时帮助捕获知识图谱中存储信息的语义。 

---
# What are you sinking? A geometric approach on attention sink 

**Title (ZH)**: 什么是你下沉的？一种注意力陷阱的几何方法 

**Authors**: Valeria Ruscio, Umberto Nanni, Fabrizio Silvestri  

**Link**: [PDF](https://arxiv.org/pdf/2508.02546)  

**Abstract**: Attention sink (AS) is a consistent pattern in transformer attention maps where certain tokens (often special tokens or positional anchors) disproportionately attract attention from other tokens. We show that in transformers, AS is not an architectural artifact, but it is the manifestation of a fundamental geometric principle: the establishment of reference frames that anchor representational spaces. We analyze several architectures and identify three distinct reference frame types, centralized, distributed, and bidirectional, that correlate with the attention sink phenomenon. We show that they emerge during the earliest stages of training as optimal solutions to the problem of establishing stable coordinate systems in high-dimensional spaces. We show the influence of architecture components, particularly position encoding implementations, on the specific type of reference frame. This perspective transforms our understanding of transformer attention mechanisms and provides insights for both architecture design and the relationship with AS. 

**Abstract (ZH)**: 注意力汇流（Attention Sink）是变压器注意力图中的一致模式，某些特定的标记（通常是特殊标记或位置锚点）会对其他标记的关注度不成比例地高。我们证明在变压器中，注意力汇流不是架构伪影，而是建立参考框架的基本几何原则的表现：这些框架将表征空间进行定位。我们分析了多个架构，并识别出三种不同的参考框架类型：集中型、分布式和双向型，它们与注意力汇流现象相关联。我们展示了这些参考框架在训练初期作为在高维空间中建立稳定坐标系统问题的最优解而出现。我们指出了架构组件，特别是位置编码实现方式，对特定类型参考框架的影响。这种视角重塑了我们对变压器注意力机制的理解，并提供了对架构设计以及与注意力汇流关系的见解。 

---
# Automatic Identification of Machine Learning-Specific Code Smells 

**Title (ZH)**: 自动识别机器学习特定的代码异味 

**Authors**: Peter Hamfelt, Ricardo Britto, Lincoln Rocha, Camilo Almendra  

**Link**: [PDF](https://arxiv.org/pdf/2508.02541)  

**Abstract**: Machine learning (ML) has rapidly grown in popularity, becoming vital to many industries. Currently, the research on code smells in ML applications lacks tools and studies that address the identification and validity of ML-specific code smells. This work investigates suitable methods and tools to design and develop a static code analysis tool (MLpylint) based on code smell criteria. This research employed the Design Science Methodology. In the problem identification phase, a literature review was conducted to identify ML-specific code smells. In solution design, a secondary literature review and consultations with experts were performed to select methods and tools for implementing the tool. We evaluated the tool on data from 160 open-source ML applications sourced from GitHub. We also conducted a static validation through an expert survey involving 15 ML professionals. The results indicate the effectiveness and usefulness of the MLpylint. We aim to extend our current approach by investigating ways to introduce MLpylint seamlessly into development workflows, fostering a more productive and innovative developer environment. 

**Abstract (ZH)**: 机器学习代码异味的研究：设计与开发基于代码异味标准的静态代码分析工具MLpylint 

---
# AIAP: A No-Code Workflow Builder for Non-Experts with Natural Language and Multi-Agent Collaboration 

**Title (ZH)**: AIAP：一种基于自然语言和多agent协作的无代码工作流构建器供非专家使用 

**Authors**: Hyunjn An, Yongwon Kim, Wonduk Seo, Joonil Park, Daye Kang, Changhoon Oh, Dokyun Kim, Seunghyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.02470)  

**Abstract**: While many tools are available for designing AI, non-experts still face challenges in clearly expressing their intent and managing system complexity. We introduce AIAP, a no-code platform that integrates natural language input with visual workflows. AIAP leverages a coordinated multi-agent system to decompose ambiguous user instructions into modular, actionable steps, hidden from users behind a unified interface. A user study involving 32 participants showed that AIAP's AI-generated suggestions, modular workflows, and automatic identification of data, actions, and context significantly improved participants' ability to develop services intuitively. These findings highlight that natural language-based visual programming significantly reduces barriers and enhances user experience in AI service design. 

**Abstract (ZH)**: 非专家友好的自然语言输入与可视化工作流集成的无代码平台AIAP：提高AI服务设计的直观性和用户体验 

---
# TreeRanker: Fast and Model-agnostic Ranking System for Code Suggestions in IDEs 

**Title (ZH)**: TreeRanker: 快速且模型无关的代码建议排名系统 

**Authors**: Daniele Cipollone, Egor Bogomolov, Arie van Deursen, Maliheh Izadi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02455)  

**Abstract**: Token-level code completion is one of the most critical features in modern Integrated Development Environments (IDEs). It assists developers by suggesting relevant identifiers and APIs during coding. While completions are typically derived from static analysis, their usefulness depends heavily on how they are ranked, as correct predictions buried deep in the list are rarely seen by users. Most current systems rely on hand-crafted heuristics or lightweight machine learning models trained on user logs, which can be further improved to capture context information and generalize across projects and coding styles. In this work, we propose a new scoring approach to ranking static completions using language models in a lightweight and model-agnostic way. Our method organizes all valid completions into a prefix tree and performs a single greedy decoding pass to collect token-level scores across the tree. This enables a precise token-aware ranking without needing beam search, prompt engineering, or model adaptations. The approach is fast, architecture-agnostic, and compatible with already deployed models for code completion. These findings highlight a practical and effective pathway for integrating language models into already existing tools within IDEs, and ultimately providing smarter and more responsive developer assistance. 

**Abstract (ZH)**: Token级别代码补全在现代集成开发环境（IDEs）中是一个最关键的功能之一。它通过在编码过程中建议相关的标识符和API来辅助开发人员。虽然补全通常是通过静态分析获得的，但它们的有用性极大地依赖于它们的排名方式，因为列表深处的正确预测很少被用户看到。现有的大多数系统依赖于手工编写的启发式规则或基于用户日志训练的轻量级机器学习模型，这些模型可以进一步改进以捕获上下文信息并在不同项目和编码风格之间进行泛化。在本工作中，我们提出了一种新的评分方法，使用语言模型以轻量级且模型无关的方式对静态补全进行排名。该方法将所有有效的补全组织成前缀树，并进行一次贪婪解码以收集树上的token级别分数。这种方法能够实现精确的token感知排名，而无需使用束搜索、提示工程或模型调整。该方法快速、架构无关，并且可以与已部署的代码补全模型兼容。这些发现强调了一条实用且有效的方法，即将语言模型集成到现有工具中，并最终为开发人员提供更智能和响应更快的帮助。 

---
# Dynamic Forgetting and Spatio-Temporal Periodic Interest Modeling for Local-Life Service Recommendation 

**Title (ZH)**: 动态遗忘与时空周期兴趣建模在本地生活服务推荐中的应用 

**Authors**: Zhaoyu Hu, Hao Guo, Yuan Tian, Erpeng Xue, Jianyang Wang, Xianyang Qi, Hongxiang Lin, Lei Wang, Sheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.02451)  

**Abstract**: In the context of the booming digital economy, recommendation systems, as a key link connecting users and numerous services, face challenges in modeling user behavior sequences on local-life service platforms, including the sparsity of long sequences and strong spatio-temporal dependence. Such challenges can be addressed by drawing an analogy to the forgetting process in human memory. This is because users' responses to recommended content follow the recency effect and the cyclicality of memory. By exploring this, this paper introduces the forgetting curve and proposes Spatio-Temporal periodic Interest Modeling (STIM) with long sequences for local-life service recommendation. STIM integrates three key components: a dynamic masking module based on the forgetting curve, which is used to extract both recent spatiotemporal features and periodic spatiotemporal features; a query-based mixture of experts (MoE) approach that can adaptively activate expert networks under different dynamic masks, enabling the collaborative modeling of time, location, and items; and a hierarchical multi-interest network unit, which captures multi-interest representations by modeling the hierarchical interactions between the shallow and deep semantics of users' recent behaviors. By introducing the STIM method, we conducted online A/B tests and achieved a 1.54\% improvement in gross transaction volume (GTV). In addition, extended offline experiments also showed improvements. STIM has been deployed in a large-scale local-life service recommendation system, serving hundreds of millions of daily active users in core application scenarios. 

**Abstract (ZH)**: 在蓬勃发展数字经济的背景下，推荐系统作为连接用户和众多服务的关键环节，在本地生活服务平台上面临着长序列稀疏性和时空强依赖性的挑战。通过借鉴人类记忆的遗忘过程，本文提出了一种基于遗忘曲线的空间时序周期兴趣建模（STIM）方法来应对这些挑战，并取得了1.54%的交易总额（GTV）提升。该方法包括三个关键组件：基于遗忘曲线的动态掩码模块，用于提取近期和周期性的时空特征；基于查询的专家混合模型（MoE），能够在不同动态掩码下自适应激活专家网络，实现时间、地点和物品的协同建模；以及层级多兴趣网络模块，通过建模用户近期行为浅层和深层语义的层级交互来捕捉多兴趣表示。STIM已在大规模本地生活服务推荐系统中部署，服务于数十亿日活跃用户的核心应用场景。 

---
# Emergence of Fair Leaders via Mediators in Multi-Agent Reinforcement Learning 

**Title (ZH)**: 多智能体强化学习中调解者促进公平领导者的涌现 

**Authors**: Akshay Dodwadmath, Setareh Maghsudi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02421)  

**Abstract**: Stackelberg games and their resulting equilibria have received increasing attention in the multi-agent reinforcement learning literature. Each stage of a traditional Stackelberg game involves a leader(s) acting first, followed by the followers. In situations where the roles of leader(s) and followers can be interchanged, the designated role can have considerable advantages, for example, in first-mover advantage settings. Then the question arises: Who should be the leader and when? A bias in the leader selection process can lead to unfair outcomes. This problem is aggravated if the agents are self-interested and care only about their goals and rewards. We formally define this leader selection problem and show its relation to fairness in agents' returns. Furthermore, we propose a multi-agent reinforcement learning framework that maximizes fairness by integrating mediators. Mediators have previously been used in the simultaneous action setting with varying levels of control, such as directly performing agents' actions or just recommending them. Our framework integrates mediators in the Stackelberg setting with minimal control (leader selection). We show that the presence of mediators leads to self-interested agents taking fair actions, resulting in higher overall fairness in agents' returns. 

**Abstract (ZH)**: Stackelberg博弈及其均衡在多agent强化学习领域的研究受到越来越多的关注。传统的Stackelberg博弈的每个阶段涉及领导者先行行动，随后是跟随者。当领导者和跟随者的角色可以互换时，指定的角色可能会带来显著的优势，例如在先手优势的情景中。那么一个问题出现了：谁应该成为领导者？什么时候？选择领导者的偏差可能导致不公平的结果。如果代理是自利的，只关心自己的目标和奖励，这个问题会更加严重。我们正式定义了这个领导者的选择问题，并展示了它与代理回报中的公平性之间的关系。此外，我们提出了一种通过整合调停者来最大化公平性的多agent强化学习框架。调停者在同时行动的设定中已被不同程度地使用，从直接执行代理的行动到仅推荐行动。我们的框架在极小控制的Stackelberg设定中整合了调停者。我们展示了调停者的存在导致自利代理采取公平行动，从而提高了代理回报中的总体公平性。 

---
# HGTS-Former: Hierarchical HyperGraph Transformer for Multivariate Time Series Analysis 

**Title (ZH)**: HGTS-Former：多层次超图变换器在多变量时间序列分析中的应用 

**Authors**: Xiao Wang, Hao Si, Fan Zhang, Xiaoya Zhou, Dengdi Sun, Wanli Lyu, Qingquan Yang, Jin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02411)  

**Abstract**: Multivariate time series analysis has long been one of the key research topics in the field of artificial intelligence. However, analyzing complex time series data remains a challenging and unresolved problem due to its high dimensionality, dynamic nature, and complex interactions among variables. Inspired by the strong structural modeling capability of hypergraphs, this paper proposes a novel hypergraph-based time series transformer backbone network, termed HGTS-Former, to address the multivariate coupling in time series data. Specifically, given the multivariate time series signal, we first normalize and embed each patch into tokens. Then, we adopt the multi-head self-attention to enhance the temporal representation of each patch. The hierarchical hypergraphs are constructed to aggregate the temporal patterns within each channel and fine-grained relations between different variables. After that, we convert the hyperedge into node features through the EdgeToNode module and adopt the feed-forward network to further enhance the output features. Extensive experiments conducted on two multivariate time series tasks and eight datasets fully validated the effectiveness of our proposed HGTS-Former. The source code will be released on this https URL. 

**Abstract (ZH)**: 基于超图的时间序列变压器骨干网络：HGTS-Former及其在多变量时间序列数据中的应用 

---
# Inference-time Scaling for Diffusion-based Audio Super-resolution 

**Title (ZH)**: 基于推断时缩放的扩散模型音频超分辨率推理 

**Authors**: Yizhu Jin, Zhen Ye, Zeyue Tian, Haohe Liu, Qiuqiang Kong, Yike Guo, Wei Xue  

**Link**: [PDF](https://arxiv.org/pdf/2508.02391)  

**Abstract**: Diffusion models have demonstrated remarkable success in generative tasks, including audio super-resolution (SR). In many applications like movie post-production and album mastering, substantial computational budgets are available for achieving superior audio quality. However, while existing diffusion approaches typically increase sampling steps to improve quality, the performance remains fundamentally limited by the stochastic nature of the sampling process, leading to high-variance and quality-limited outputs. Here, rather than simply increasing the number of sampling steps, we propose a different paradigm through inference-time scaling for SR, which explores multiple solution trajectories during the sampling process. Different task-specific verifiers are developed, and two search algorithms, including the random search and zero-order search for SR, are introduced. By actively guiding the exploration of the high-dimensional solution space through verifier-algorithm combinations, we enable more robust and higher-quality outputs. Through extensive validation across diverse audio domains (speech, music, sound effects) and frequency ranges, we demonstrate consistent performance gains, achieving improvements of up to 9.70% in aesthetics, 5.88% in speaker similarity, 15.20% in word error rate, and 46.98% in spectral distance for speech SR from 4kHz to 24kHz, showcasing the effectiveness of our approach. Audio samples are available at: this https URL. 

**Abstract (ZH)**: 扩散模型在音频超分辨率生成任务中取得了显著成功。在电影后期制作和专辑母版制作等应用中，可用大量的计算资源以实现卓越的音频质量。然而，现有扩散方法通常通过增加采样步骤来提高质量，但性能依然受限于采样过程的随机性，导致输出具有高方差和质量限制。本文不单纯增加采样步骤，而是提出了一种在推断时通过尺度调整进行超分辨率的全新范式，在采样过程中探索多种解决方案轨迹。为不同任务开发了特定的验证器，并引入了两种搜索算法，包括随机搜索和用于超分辨率的零阶搜索。通过验证器和算法的组合主动引导高维解决方案空间的探索，实现了更加稳健和高质量的输出。通过在多种音频域（语音、音乐、音效）和频率范围下进行广泛验证，我们展示了持续的性能提升，从4kHz到24kHz的语音超分辨率方面，分别在美学、说话人相似性、词错误率和频谱距离上取得高达9.70%、5.88%、15.20%和46.98%的改善，展示了我们方法的有效性。音频样本见：this https URL。 

---
# Flexible Automatic Identification and Removal (FAIR)-Pruner: An Efficient Neural Network Pruning Method 

**Title (ZH)**: FAIR修剪器：一种高效的神经网络修剪方法 

**Authors**: Chenqing Lin, Mostafa Hussien, Chengyao Yu, Mohamed Cheriet, Osama Abdelrahman, Ruixing Ming  

**Link**: [PDF](https://arxiv.org/pdf/2508.02291)  

**Abstract**: Neural network pruning is a critical compression technique that facilitates the deployment of large-scale neural networks on resource-constrained edge devices, typically by identifying and eliminating redundant or insignificant parameters to reduce computational and memory overhead. This paper proposes the Flexible Automatic Identification and Removal (FAIR)-Pruner, a novel method for neural network structured pruning. Specifically, FAIR-Pruner first evaluates the importance of each unit (e.g., neuron or channel) through the Utilization Score quantified by the Wasserstein distance. To reflect the performance degradation after unit removal, it then introduces the Reconstruction Error, which is computed via the Taylor expansion of the loss function. Finally, FAIR-Pruner identifies superfluous units with negligible impact on model performance by controlling the proposed Tolerance of Difference, which measures differences between unimportant units and those that cause performance degradation. A major advantage of FAIR-Pruner lies in its capacity to automatically determine the layer-wise pruning rates, which yields a more efficient subnetwork structure compared to applying a uniform pruning rate. Another advantage of the FAIR-Pruner is its great one-shot performance without post-pruning fine-tuning. Furthermore, with utilization scores and reconstruction errors, users can flexibly obtain pruned models under different pruning ratios. Comprehensive experimental validation on diverse benchmark datasets (e.g., ImageNet) and various neural network architectures (e.g., VGG) demonstrates that FAIR-Pruner achieves significant model compression while maintaining high accuracy. 

**Abstract (ZH)**: 灵活自动识别和去除（FAIR）剪枝器：一种新颖的神经网络结构化剪枝方法 

---
# Dialogue Systems Engineering: A Survey and Future Directions 

**Title (ZH)**: 对话系统工程：综述与未来方向 

**Authors**: Mikio Nakano, Hironori Takeuchi, Sadahiro Yoshikawa, Yoichi Matsuyama, Kazunori Komatani  

**Link**: [PDF](https://arxiv.org/pdf/2508.02279)  

**Abstract**: This paper proposes to refer to the field of software engineering related to the life cycle of dialogue systems as Dialogue Systems Engineering, and surveys this field while also discussing its future directions. With the advancement of large language models, the core technologies underlying dialogue systems have significantly progressed. As a result, dialogue system technology is now expected to be applied to solving various societal issues and in business contexts. To achieve this, it is important to build, operate, and continuously improve dialogue systems correctly and efficiently. Accordingly, in addition to applying existing software engineering knowledge, it is becoming increasingly important to evolve software engineering tailored specifically to dialogue systems. In this paper, we enumerate the knowledge areas of dialogue systems engineering based on those of software engineering, as defined in the Software Engineering Body of Knowledge (SWEBOK) Version 4.0, and survey each area. Based on this survey, we identify unexplored topics in each area and discuss the future direction of dialogue systems engineering. 

**Abstract (ZH)**: 本文提出将与对话系统生命周期相关的软件工程领域称为对话系统工程，并对该领域进行综述，同时讨论其未来方向。随着大型语言模型的发展，对话系统的核心技术有了显著进步，因此对话系统技术现在被期望应用于解决各种社会问题和商业情境中。为了实现这一目标，正确且高效地构建、运行和持续改进对话系统至关重要。因此，在现有软件工程知识的基础上，有必要针对对话系统演化出专门的软件工程方法。本文基于软件工程知识体系（SWEBOK）第4版的知识领域，列出对话系统工程的知识领域，并对每个领域进行综述。基于该综述，我们识别出每个领域的未探索话题，并讨论对话系统工程的未来方向。 

---
# CellForge: Agentic Design of Virtual Cell Models 

**Title (ZH)**: CellForge: 自主设计虚拟细胞模型 

**Authors**: Xiangru Tang, Zhuoyun Yu, Jiapeng Chen, Yan Cui, Daniel Shao, Weixu Wang, Fang Wu, Yuchen Zhuang, Wenqi Shi, Zhi Huang, Arman Cohan, Xihong Lin, Fabian Theis, Smita Krishnaswamy, Mark Gerstein  

**Link**: [PDF](https://arxiv.org/pdf/2508.02276)  

**Abstract**: Virtual cell modeling represents an emerging frontier at the intersection of artificial intelligence and biology, aiming to predict quantities such as responses to diverse perturbations quantitatively. However, autonomously building computational models for virtual cells is challenging due to the complexity of biological systems, the heterogeneity of data modalities, and the need for domain-specific expertise across multiple disciplines. Here, we introduce CellForge, an agentic system that leverages a multi-agent framework that transforms presented biological datasets and research objectives directly into optimized computational models for virtual cells. More specifically, given only raw single-cell multi-omics data and task descriptions as input, CellForge outputs both an optimized model architecture and executable code for training virtual cell models and inference. The framework integrates three core modules: Task Analysis for presented dataset characterization and relevant literature retrieval, Method Design, where specialized agents collaboratively develop optimized modeling strategies, and Experiment Execution for automated generation of code. The agents in the Design module are separated into experts with differing perspectives and a central moderator, and have to collaboratively exchange solutions until they achieve a reasonable consensus. We demonstrate CellForge's capabilities in single-cell perturbation prediction, using six diverse datasets that encompass gene knockouts, drug treatments, and cytokine stimulations across multiple modalities. CellForge consistently outperforms task-specific state-of-the-art methods. Overall, CellForge demonstrates how iterative interaction between LLM agents with differing perspectives provides better solutions than directly addressing a modeling challenge. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 虚拟细胞建模代表了人工智能与生物学交汇领域的新兴前沿，旨在定量预测不同扰动的响应。然而，由于生物系统的复杂性、数据模态的异质性以及多学科领域特定专业知识的需要，自主构建虚拟细胞的计算模型具有挑战性。这里我们介绍CellForge，这是一种代理系统，利用多代理框架将呈现的生物数据集和研究目标直接转换为优化的虚拟细胞计算模型。具体而言，仅给定原始单细胞多组学数据和任务描述作为输入，CellForge 输出优化的模型架构和用于训练虚拟细胞模型及推断的可执行代码。该框架集成三个核心模块：任务分析、方法设计和实验执行。设计模块中的代理被分为具有不同视角的专家和一个中心协调员，他们需要协作交换解决方案，直到达成合理的共识。我们通过六个涵盖基因敲除、药物治疗和细胞因子刺激等多种模态的单细胞扰动预测数据集，展示了CellForge的能力。CellForge在所有任务中均优于特定任务的最佳方法。总体而言，CellForge展示了迭代互动如何为多视角人工智能代理提供比直接解决建模挑战更好的解决方案。我们的代码可在以下网址公开获得。 

---
# Dynaword: From One-shot to Continuously Developed Datasets 

**Title (ZH)**: Dynaword: 从单次生成到持续发展的数据集 

**Authors**: Kenneth Enevoldsen, Kristian Nørgaard Jensen, Jan Kostkan, Balázs Szabó, Márton Kardos, Kirten Vad, Andrea Blasi Núñez, Gianluca Barmina, Jacob Nielsen, Rasmus Larsen, Peter Vahlstrup, Per Møldrup Dalum, Desmond Elliott, Lukas Galke, Peter Schneider-Kamp, Kristoffer Nielbo  

**Link**: [PDF](https://arxiv.org/pdf/2508.02271)  

**Abstract**: Large-scale datasets are foundational for research and development in natural language processing. However, current approaches face three key challenges: (1) reliance on ambiguously licensed sources restricting use, sharing, and derivative works; (2) static dataset releases that prevent community contributions and diminish longevity; and (3) quality assurance processes restricted to publishing teams rather than leveraging community expertise.
To address these limitations, we introduce two contributions: the Dynaword approach and Danish Dynaword. The Dynaword approach is a framework for creating large-scale, open datasets that can be continuously updated through community collaboration. Danish Dynaword is a concrete implementation that validates this approach and demonstrates its potential. Danish Dynaword contains over four times as many tokens as comparable releases, is exclusively openly licensed, and has received multiple contributions across industry and research. The repository includes light-weight tests to ensure data formatting, quality, and documentation, establishing a sustainable framework for ongoing community contributions and dataset evolution. 

**Abstract (ZH)**: 大规模语料库是自然语言处理研究与开发的基础。然而，当前的方法面临三大关键挑战：(1) 对于使用、共享和衍生工作的限制性许可限制；(2) 静态数据集发布阻碍社区贡献并减少其持久性；(3) 质量保证过程仅限于发布团队，未能利用社区专业知识。

为解决这些限制，我们提出了两项贡献：Dynaword方法和Danish Dynaword。Dynaword方法是一种可以通过社区协作持续更新的大规模开源数据集框架。Danish Dynaword是这一方法的具体实现，展示了其潜力。Danish Dynaword包含比可比发布版本多四倍的词元，完全是开放许可的，并且已收到来自工业和研究界的多次贡献。该仓库还包括轻量级测试以确保数据格式化、质量和文档化，从而建立一个可持续的框架，以促进持续的社区贡献和数据集的进化。 

---
# StutterCut: Uncertainty-Guided Normalised Cut for Dysfluency Segmentation 

**Title (ZH)**: StutterCut: 基于不确定性归一化切分的非流畅性分割 

**Authors**: Suhita Ghosh, Melanie Jouaiti, Jan-Ole Perschewski, Sebastian Stober  

**Link**: [PDF](https://arxiv.org/pdf/2508.02255)  

**Abstract**: Detecting and segmenting dysfluencies is crucial for effective speech therapy and real-time feedback. However, most methods only classify dysfluencies at the utterance level. We introduce StutterCut, a semi-supervised framework that formulates dysfluency segmentation as a graph partitioning problem, where speech embeddings from overlapping windows are represented as graph nodes. We refine the connections between nodes using a pseudo-oracle classifier trained on weak (utterance-level) labels, with its influence controlled by an uncertainty measure from Monte Carlo dropout. Additionally, we extend the weakly labelled FluencyBank dataset by incorporating frame-level dysfluency boundaries for four dysfluency types. This provides a more realistic benchmark compared to synthetic datasets. Experiments on real and synthetic datasets show that StutterCut outperforms existing methods, achieving higher F1 scores and more precise stuttering onset detection. 

**Abstract (ZH)**: 检测和分割语音不流畅是有效言语治疗和实时反馈的关键。然而，大多数方法仅在句级对不流畅进行分类。我们引入了StutterCut，一种半监督框架，将语音不流畅分割问题形式化为图划分问题，其中重叠窗口的语音嵌入表示为图节点。我们使用在弱（句级）标签上训练的伪 oracle 分类器来细化节点之间的连接，并通过蒙特卡洛弃权的不确定性测量来控制其影响。此外，我们通过为四种不流畅类型增加帧级不流畅边界，扩展了弱标记的FluencyBank数据集。这提供了比合成数据集更现实的基准。实验结果表明，StutterCut在真实和合成数据集上均优于现有方法，取得了更高的F1分数和更精确的重复语起始检测。 

---
# ByteGen: A Tokenizer-Free Generative Model for Orderbook Events in Byte Space 

**Title (ZH)**: ByteGen：一种基于字节空间的无分词生成模型用于订单簿事件 

**Authors**: Yang Li, Zhi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.02247)  

**Abstract**: Generative modeling of high-frequency limit order book (LOB) dynamics is a critical yet unsolved challenge in quantitative finance, essential for robust market simulation and strategy backtesting. Existing approaches are often constrained by simplifying stochastic assumptions or, in the case of modern deep learning models like Transformers, rely on tokenization schemes that affect the high-precision, numerical nature of financial data through discretization and binning. To address these limitations, we introduce ByteGen, a novel generative model that operates directly on the raw byte streams of LOB events. Our approach treats the problem as an autoregressive next-byte prediction task, for which we design a compact and efficient 32-byte packed binary format to represent market messages without information loss. The core novelty of our work is the complete elimination of feature engineering and tokenization, enabling the model to learn market dynamics from its most fundamental representation. We achieve this by adapting the H-Net architecture, a hybrid Mamba-Transformer model that uses a dynamic chunking mechanism to discover the inherent structure of market messages without predefined rules. Our primary contributions are: 1) the first end-to-end, byte-level framework for LOB modeling; 2) an efficient packed data representation; and 3) a comprehensive evaluation on high-frequency data. Trained on over 34 million events from CME Bitcoin futures, ByteGen successfully reproduces key stylized facts of financial markets, generating realistic price distributions, heavy-tailed returns, and bursty event timing. Our findings demonstrate that learning directly from byte space is a promising and highly flexible paradigm for modeling complex financial systems, achieving competitive performance on standard market quality metrics without the biases of tokenization. 

**Abstract (ZH)**: 高频率限价订单簿（LOB）动力学的生成建模是量化金融中一个关键但未解决的挑战，对于稳健的市场模拟和策略回测至关重要。现有方法通常受限于简化的随机假设，或者在使用如变换器等现代深度学习模型时依赖影响金融数据高精度数值性质的令牌化方案。为解决这些限制，我们提出了ByteGen，一种新型的生成模型，直接处理LOB事件的原始字节流。我们的方法将问题视为自回归下一个字节预测任务，并设计了一种紧凑高效的32字节打包二进制格式来表示市场消息而不损失信息。我们的核心创新在于完全消除特征工程和令牌化，使模型能够从最基本的数据表示中学习市场动态。通过适应H-Net架构，一种混合Mamba-变压器模型，该模型使用动态分块机制来发现市场消息的固有结构，而无需预定义规则。我们的主要贡献包括：1）首个端到端的字节级LOB建模框架；2）高效的数据打包表示；3）在高频数据上的全面评估。ByteGen在CME比特币期货超过3400万事件的训练下，成功重现了金融市场的关键统计事实，生成了真实的 price 分布、重尾收益和突发事件时间。我们的研究结果表明，直接从字节空间学习是一种有前景且高度灵活的方法，用于建模复杂金融系统，在标准市场质量度量上实现了与令牌化偏差的竞争性性能。 

---
# Fitness aligned structural modeling enables scalable virtual screening with AuroBind 

**Title (ZH)**: 基于适应度对齐的结构建模使AuroBind能够实现可扩展的虚拟筛选 

**Authors**: Zhongyue Zhang, Jiahua Rao, Jie Zhong, Weiqiang Bai, Dongxue Wang, Shaobo Ning, Lifeng Qiao, Sheng Xu, Runze Ma, Will Hua, Jack Xiaoyu Chen, Odin Zhang, Wei Lu, Hanyi Feng, He Yang, Xinchao Shi, Rui Li, Wanli Ouyang, Xinzhu Ma, Jiahao Wang, Jixian Zhang, Jia Duan, Siqi Sun, Jian Zhang, Shuangjia Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.02137)  

**Abstract**: Most human proteins remain undrugged, over 96% of human proteins remain unexploited by approved therapeutics. While structure-based virtual screening promises to expand the druggable proteome, existing methods lack atomic-level precision and fail to predict binding fitness, limiting translational impact. We present AuroBind, a scalable virtual screening framework that fine-tunes a custom atomic-level structural model on million-scale chemogenomic data. AuroBind integrates direct preference optimization, self-distillation from high-confidence complexes, and a teacher-student acceleration strategy to jointly predict ligand-bound structures and binding fitness. The proposed models outperform state-of-the-art models on structural and functional benchmarks while enabling 100,000-fold faster screening across ultra-large compound libraries. In a prospective screen across ten disease-relevant targets, AuroBind achieved experimental hit rates of 7-69%, with top compounds reaching sub-nanomolar to picomolar potency. For the orphan GPCRs GPR151 and GPR160, AuroBind identified both agonists and antagonists with success rates of 16-30%, and functional assays confirmed GPR160 modulation in liver and prostate cancer models. AuroBind offers a generalizable framework for structure-function learning and high-throughput molecular screening, bridging the gap between structure prediction and therapeutic discovery. 

**Abstract (ZH)**: 大多数人类蛋白质未被药物化，超过96%的人类蛋白质未被获批的治疗药物所利用。尽管基于结构的虚拟筛选有望扩大可药物化蛋白质组，现有方法缺乏原子级精度，无法预测配体结合亲和力，限制了其临床转化效果。我们提出了一种名为AuroBind的可扩展虚拟筛选框架，该框架在一个百万尺度的化学生物学数据集上对定制的原子级结构模型进行了微调。AuroBind结合了直接偏好优化、高置信度复合物的自蒸馏以及教师-学生加速策略，以联合预测配体结合结构和结合亲和力。所提出的模型在结构和功能基准测试中优于现有模型，同时能够对超大型化合物库进行高达100,000倍速度的筛选。在针对十个疾病相关靶点的前瞻性筛选中，AuroBind获得了7-69%的实验阳性率，其中最佳化合物达到次纳摩到皮摩的效力。对于_ORphan_ GPCRs GPR151和GPR160，AuroBind分别以16-30%的成功率发现了激动剂和拮抗剂，并且功能检测证实了GPR160在肝癌和前列腺癌模型中的调节作用。AuroBind提供了一种可推广的结构-功能学习框架和高通量分子筛选框架，缩短了结构预测与治疗发现之间的差距。 

---
# The Complexity of Extreme Climate Events on the New Zealand's Kiwifruit Industry 

**Title (ZH)**: 极端气候事件对新西兰猕猴桃行业的复杂性影响 

**Authors**: Boyuan Zheng, Victor W. Chu, Zhidong Li, Evan Webster, Ashley Rootsey  

**Link**: [PDF](https://arxiv.org/pdf/2508.02130)  

**Abstract**: Climate change has intensified the frequency and severity of extreme weather events, presenting unprecedented challenges to the agricultural industry worldwide. In this investigation, we focus on kiwifruit farming in New Zealand. We propose to examine the impacts of climate-induced extreme events, specifically frost, drought, extreme rainfall, and heatwave, on kiwifruit harvest yields. These four events were selected due to their significant impacts on crop productivity and their prevalence as recorded by climate monitoring institutions in the country. We employed Isolation Forest, an unsupervised anomaly detection method, to analyse climate history and recorded extreme events, alongside with kiwifruit yields. Our analysis reveals considerable variability in how different types of extreme event affect kiwifruit yields underscoring notable discrepancies between climatic extremes and individual farm's yield outcomes. Additionally, our study highlights critical limitations of current anomaly detection approaches, particularly in accurately identifying events such as frost. These findings emphasise the need for integrating supplementary features like farm management strategies with climate adaptation practices. Our further investigation will employ ensemble methods that consolidate nearby farms' yield data and regional climate station features to reduce variance, thereby enhancing the accuracy and reliability of extreme event detection and the formulation of response strategies. 

**Abstract (ZH)**: 气候变化加剧了极端天气事件的频率和 severity，给全球农业行业带来了前所未有的挑战。本研究以新西兰奇异果农场为例，旨在考察由气候变迁引发的极端事件，包括霜冻、干旱、极端降雨和热浪，对奇异果收获产量的影响。四种事件因其对作物产量的显著影响以及作为气象监测机构记录的常见事件而被选作研究对象。研究使用孤立森林算法，一种无监督异常检测方法，分析气候历史、记录的极端事件以及奇异果产量数据。研究发现，不同类型的极端事件对奇异果产量的影响存在显著差异，揭示了气候极端事件与单个农场产量结果之间的重大差异。此外，研究还突显了当前异常检测方法的关键局限性，特别是在准确识别霜冻等事件方面。这些发现强调了需要结合农场管理策略与气候适应实践的重要性。进一步的研究将采用集成方法，整合附近农场的产量数据和区域气象站特征，以减少变异，从而提高极端事件检测的准确性和可靠性，以及响应策略的制定。 

---
# Coward: Toward Practical Proactive Federated Backdoor Defense via Collision-based Watermark 

**Title (ZH)**: Coward: 基于碰撞标记的实用主动联邦后门防御方法 

**Authors**: Wenjie Li, Siying Gu, Yiming Li, Kangjie Chen, Zhili Chen, Tianwei Zhang, Shu-Tao Xia, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2508.02115)  

**Abstract**: Backdoor detection is currently the mainstream defense against backdoor attacks in federated learning (FL), where malicious clients upload poisoned updates that compromise the global model and undermine the reliability of FL deployments. Existing backdoor detection techniques fall into two categories, including passive and proactive ones, depending on whether the server proactively modifies the global model. However, both have inherent limitations in practice: passive defenses are vulnerable to common non-i.i.d. data distributions and random participation of FL clients, whereas current proactive defenses suffer inevitable out-of-distribution (OOD) bias because they rely on backdoor co-existence effects. To address these issues, we introduce a new proactive defense, dubbed Coward, inspired by our discovery of multi-backdoor collision effects, in which consecutively planted, distinct backdoors significantly suppress earlier ones. In general, we detect attackers by evaluating whether the server-injected, conflicting global watermark is erased during local training rather than retained. Our method preserves the advantages of proactive defenses in handling data heterogeneity (\ie, non-i.i.d. data) while mitigating the adverse impact of OOD bias through a revised detection mechanism. Extensive experiments on benchmark datasets confirm the effectiveness of Coward and its resilience to potential adaptive attacks. The code for our method would be available at this https URL. 

**Abstract (ZH)**: 基于多重后门碰撞效果的Coward主动防御机制：缓解联邦学习中的后门攻击 

---
# CRINN: Contrastive Reinforcement Learning for Approximate Nearest Neighbor Search 

**Title (ZH)**: CRINN: 对比增强学习近邻搜索近似算法 

**Authors**: Xiaoya Li, Xiaofei Sun, Albert Wang, Chris Shum, Jiwei Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.02091)  

**Abstract**: Approximate nearest-neighbor search (ANNS) algorithms have become increasingly critical for recent AI applications, particularly in retrieval-augmented generation (RAG) and agent-based LLM applications. In this paper, we present CRINN, a new paradigm for ANNS algorithms. CRINN treats ANNS optimization as a reinforcement learning problem where execution speed serves as the reward signal. This approach enables the automatic generation of progressively faster ANNS implementations while maintaining accuracy constraints. Our experimental evaluation demonstrates CRINN's effectiveness across six widely-used NNS benchmark datasets. When compared against state-of-the-art open-source ANNS algorithms, CRINN achieves best performance on three of them (GIST-960-Euclidean, MNIST-784-Euclidean, and GloVe-25-angular), and tied for first place on two of them (SIFT-128-Euclidean and GloVe-25-angular). The implications of CRINN's success reach well beyond ANNS optimization: It validates that LLMs augmented with reinforcement learning can function as an effective tool for automating sophisticated algorithmic optimizations that demand specialized knowledge and labor-intensive manual this http URL can be found at this https URL 

**Abstract (ZH)**: 近邻搜索(CRINN):一种新的近邻搜索算法范式 

---
# SSBD Ontology: A Two-Tier Approach for Interoperable Bioimaging Metadata 

**Title (ZH)**: SSBD本体：一种可互操作生物成像元数据的二层方法 

**Authors**: Yuki Yamagata, Koji Kyoda, Hiroya Itoga, Emi Fujisawa, Shuichi Onami  

**Link**: [PDF](https://arxiv.org/pdf/2508.02084)  

**Abstract**: Advanced bioimaging technologies have enabled the large-scale acquisition of multidimensional data, yet effective metadata management and interoperability remain significant challenges. To address these issues, we propose a new ontology-driven framework for the Systems Science of Biological Dynamics Database (SSBD) that adopts a two-tier architecture. The core layer provides a class-centric structure referencing existing biomedical ontologies, supporting both SSBD:repository -- which focuses on rapid dataset publication with minimal metadata -- and SSBD:database, which is enhanced with biological and imaging-related annotations. Meanwhile, the instance layer represents actual imaging dataset information as Resource Description Framework individuals that are explicitly linked to the core classes. This layered approach aligns flexible instance data with robust ontological classes, enabling seamless integration and advanced semantic queries. By coupling flexibility with rigor, the SSBD Ontology promotes interoperability, data reuse, and the discovery of novel biological mechanisms. Moreover, our solution aligns with the Recommended Metadata for Biological Images guidelines and fosters compatibility. Ultimately, our approach contributes to establishing a Findable, Accessible, Interoperable, and Reusable data ecosystem within the bioimaging community. 

**Abstract (ZH)**: 先进生物成像技术使得大规模获取多维数据成为可能，然而有效的元数据管理和互操作性仍然面临重大挑战。为应对这些问题，我们提出了一种新的本体驱动框架，用于生物动力学数据库系统科学（SSBD），采用两层架构。核心层提供以类为中心的结构，参考现有的生物医学本体，支持SSBD:repository——专注于快速数据集发布，最少元数据——以及SSBD:database，后者增加了生物学和成像相关的注释。与此同时，实例层将实际的成像数据集信息表示为与核心类明确链接的资源描述框架个体。这种分层方法使灵活的实例数据与坚固的本体类保持一致，从而实现无缝集成和高级语义查询。通过结合灵活性和严谨性，SSBD本体促进了互操作性、数据重用和新型生物机制的发现。此外，我们的解决方案符合生物图像推荐元数据指南，促进了兼容性。最终，我们的方法有助于建立生物成像社区内的可查找、可访问、可互操作和可重用的数据生态系统。 

---
# SpikeSTAG: Spatial-Temporal Forecasting via GNN-SNN Collaboration 

**Title (ZH)**: SpikeSTAG：基于GNN-SNN协作的时空预测 

**Authors**: Bang Hu, Changze Lv, Mingjie Li, Yunpeng Liu, Xiaoqing Zheng, Fengzhe Zhang, Wei cao, Fan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02069)  

**Abstract**: Spiking neural networks (SNNs), inspired by the spiking behavior of biological neurons, offer a distinctive approach for capturing the complexities of temporal data. However, their potential for spatial modeling in multivariate time-series forecasting remains largely unexplored. To bridge this gap, we introduce a brand new SNN architecture, which is among the first to seamlessly integrate graph structural learning with spike-based temporal processing for multivariate time-series forecasting. Specifically, we first embed time features and an adaptive matrix, eliminating the need for predefined graph structures. We then further learn sequence features through the Observation (OBS) Block. Building upon this, our Multi-Scale Spike Aggregation (MSSA) hierarchically aggregates neighborhood information through spiking SAGE layers, enabling multi-hop feature extraction while eliminating the need for floating-point operations. Finally, we propose a Dual-Path Spike Fusion (DSF) Block to integrate spatial graph features and temporal dynamics via a spike-gated mechanism, combining LSTM-processed sequences with spiking self-attention outputs, effectively improve the model accuracy of long sequence datasets. Experiments show that our model surpasses the state-of-the-art SNN-based iSpikformer on all datasets and outperforms traditional temporal models at long horizons, thereby establishing a new paradigm for efficient spatial-temporal modeling. 

**Abstract (ZH)**: 基于尖峰神经网络的多尺度尖峰聚合与双路径尖峰融合的时空模型 

---
# Enhancement of Quantum Semi-Supervised Learning via Improved Laplacian and Poisson Methods 

**Title (ZH)**: 改进拉普拉斯和泊松方法增强量子半监督学习 

**Authors**: Hamed Gholipour, Farid Bozorgnia, Hamzeh Mohammadigheymasi, Kailash Hambarde, Javier Mancilla, Hugo Proenca, Joao Neves, Moharram Challenger  

**Link**: [PDF](https://arxiv.org/pdf/2508.02054)  

**Abstract**: This paper develops a hybrid quantum approach for graph-based semi-supervised learning to enhance performance in scenarios where labeled data is scarce. We introduce two enhanced quantum models, the Improved Laplacian Quantum Semi-Supervised Learning (ILQSSL) and the Improved Poisson Quantum Semi-Supervised Learning (IPQSSL), that incorporate advanced label propagation strategies within variational quantum circuits. These models utilize QR decomposition to embed graph structure directly into quantum states, thereby enabling more effective learning in low-label settings. We validate our methods across four benchmark datasets like Iris, Wine, Heart Disease, and German Credit Card -- and show that both ILQSSL and IPQSSL consistently outperform leading classical semi-supervised learning algorithms, particularly under limited supervision. Beyond standard performance metrics, we examine the effect of circuit depth and qubit count on learning quality by analyzing entanglement entropy and Randomized Benchmarking (RB). Our results suggest that while some level of entanglement improves the model's ability to generalize, increased circuit complexity may introduce noise that undermines performance on current quantum hardware. Overall, the study highlights the potential of quantum-enhanced models for semi-supervised learning, offering practical insights into how quantum circuits can be designed to balance expressivity and stability. These findings support the role of quantum machine learning in advancing data-efficient classification, especially in applications constrained by label availability and hardware limitations. 

**Abstract (ZH)**: 一种基于图的半监督学习的混合量子方法：在标注数据稀少场景中提升性能 

---
# Epi$^2$-Net: Advancing Epidemic Dynamics Forecasting with Physics-Inspired Neural Networks 

**Title (ZH)**: Epi$^2$-Net：基于物理启发的神经网络推动传染病动力学预测 

**Authors**: Rui Sun, Chenghua Gong, Tianjun Gu, Yuhao Zheng, Jie Ding, Juyuan Zhang, Liming Pan, Linyuan Lü  

**Link**: [PDF](https://arxiv.org/pdf/2508.02049)  

**Abstract**: Advancing epidemic dynamics forecasting is vital for targeted interventions and safeguarding public health. Current approaches mainly fall into two categories: mechanism-based and data-driven models. Mechanism-based models are constrained by predefined compartmental structures and oversimplified system assumptions, limiting their ability to model complex real-world dynamics, while data-driven models focus solely on intrinsic data dependencies without physical or epidemiological constraints, risking biased or misleading representations. Although recent studies have attempted to integrate epidemiological knowledge into neural architectures, most of them fail to reconcile explicit physical priors with neural representations. To overcome these obstacles, we introduce Epi$^2$-Net, a Epidemic Forecasting Framework built upon Physics-Inspired Neural Networks. Specifically, we propose reconceptualizing epidemic transmission from the physical transport perspective, introducing the concept of neural epidemic transport. Further, we present a physic-inspired deep learning framework, and integrate physical constraints with neural modules to model spatio-temporal patterns of epidemic dynamics. Experiments on real-world datasets have demonstrated that Epi$^2$-Net outperforms state-of-the-art methods in epidemic forecasting, providing a promising solution for future epidemic containment. The code is available at: this https URL. 

**Abstract (ZH)**: 推进流行病动态预测对于针对性干预和保障公共卫生至关重要。当前的方法主要分为两类：机制基础模型和数据驱动模型。机制基础模型受限于预设的隔室结构和过于简化的系统假设，限制了其对复杂现实世界动态的建模能力，而数据驱动模型仅关注内在数据依赖性，缺乏物理或流行病学约束，可能导致偏见或误导性的表示。尽管最近的研究尝试将流行病学知识融入神经网络架构中，但大多数方法未能解决显式物理先验与神经表示之间的契合问题。为克服这些障碍，我们引入了Epi$^2$-Net，这是一种基于物理启发神经网络的流行病预测框架。具体而言，我们从物理传输的角度重新概念化流行病传播，引入了神经流行病传输的概念。进一步地，我们提出了一种基于物理启发的深度学习框架，将物理约束与神经模块集成，用于建模流行病动态的时空模式。实验结果表明，Epi$^2$-Net在流行病预测中优于现有方法，为未来的流行病控制提供了有希望的解决方案。代码可在此处访问：this https URL。 

---
# Graph Unlearning via Embedding Reconstruction -- A Range-Null Space Decomposition Approach 

**Title (ZH)**: 基于范围Null空间分解的图遗忘重构方法 

**Authors**: Hang Yin, Zipeng Liu, Xiaoyong Peng, Liyao Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02044)  

**Abstract**: Graph unlearning is tailored for GNNs to handle widespread and various graph structure unlearning requests, which remain largely unexplored. The GIF (graph influence function) achieves validity under partial edge unlearning, but faces challenges in dealing with more disturbing node unlearning. To avoid the overhead of retraining and realize the model utility of unlearning, we proposed a novel node unlearning method to reverse the process of aggregation in GNN by embedding reconstruction and to adopt Range-Null Space Decomposition for the nodes' interaction learning. Experimental results on multiple representative datasets demonstrate the SOTA performance of our proposed approach. 

**Abstract (ZH)**: 图去学习针对GNNs设计以处理广泛且多样的图结构去学习请求，这些请求目前尚未充分探索。 

---
# SpeechR: A Benchmark for Speech Reasoning in Large Audio-Language Models 

**Title (ZH)**: SpeechR：面向大规模音视语言模型的语音推理基准 

**Authors**: Wanqi Yang, Yanda Li, Yunchao Wei, Meng Fang, Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.02018)  

**Abstract**: Large audio-language models (LALMs) have achieved near-human performance in sentence-level transcription and emotion recognition. However, existing evaluations focus mainly on surface-level perception, leaving the capacity of models for contextual and inference-driven reasoning in speech-based scenarios insufficiently examined. To address this gap, we introduce SpeechR, a unified benchmark for evaluating reasoning over speech in large audio-language models. SpeechR evaluates models along three key dimensions: factual retrieval, procedural inference, and normative judgment. It includes three distinct evaluation formats. The multiple-choice version measures answer selection accuracy. The generative version assesses the coherence and logical consistency of reasoning chains. The acoustic-feature version investigates whether variations in stress and emotion affect reasoning performance. Evaluations on eleven state-of-the-art LALMs reveal that high transcription accuracy does not translate into strong reasoning capabilities. SpeechR establishes a structured benchmark for evaluating reasoning in spoken language, enabling more targeted analysis of model capabilities across diverse dialogue-based tasks. 

**Abstract (ZH)**: 大型音频语言模型（LALMs）已在句级转写和情绪识别方面实现了近乎人类的性能。然而，现有评估主要集中在表层感知上，使得模型在基于语音的情景中进行上下文和推理驱动的推理能力不足。为弥补这一空白，我们引入了SpeechR，这是一个统一的基准，用于评估大型音频语言模型在语音上的推理能力。SpeechR 从三个关键维度评估模型：事实检索、程序推理和规范判断。它包括三种不同的评估格式。多项选择版本衡量答案选择的准确性。生成版本评估推理链的连贯性和逻辑一致性。声学特征版本探讨语音中的重音和情绪变化是否影响推理性能。对十一种最先进的LALMs的评估表明，高转写准确性并不一定能转化为强大的推理能力。SpeechR 为评估口语中的推理能力建立了结构化的基准，使我们能够对模型在多种对话任务中的能力进行更针对性的分析。 

---
# DIRF: A Framework for Digital Identity Protection and Clone Governance in Agentic AI Systems 

**Title (ZH)**: DIRF：一种针对有能动性的AI系统中数字身份保护与克隆治理的框架 

**Authors**: Hammad Atta, Muhammad Zeeshan Baig, Yasir Mehmood, Nadeem Shahzad, Ken Huang, Muhammad Aziz Ul Haq, Muhammad Awais, Kamal Ahmed, Anthony Green  

**Link**: [PDF](https://arxiv.org/pdf/2508.01997)  

**Abstract**: The rapid advancement and widespread adoption of generative artificial intelligence (AI) pose significant threats to the integrity of personal identity, including digital cloning, sophisticated impersonation, and the unauthorized monetization of identity-related data. Mitigating these risks necessitates the development of robust AI-generated content detection systems, enhanced legal frameworks, and ethical guidelines. This paper introduces the Digital Identity Rights Framework (DIRF), a structured security and governance model designed to protect behavioral, biometric, and personality-based digital likeness attributes to address this critical need. Structured across nine domains and 63 controls, DIRF integrates legal, technical, and hybrid enforcement mechanisms to secure digital identity consent, traceability, and monetization. We present the architectural foundations, enforcement strategies, and key use cases supporting the need for a unified framework. This work aims to inform platform builders, legal entities, and regulators about the essential controls needed to enforce identity rights in AI-driven systems. 

**Abstract (ZH)**: 数字身份权利框架（DIRF）：保护行为、生物特征和基于个性的数字肖像属性的安全与治理模型 

---
# Controllable and Stealthy Shilling Attacks via Dispersive Latent Diffusion 

**Title (ZH)**: 可控且隐蔽的分流攻击通过分散潜在扩散实现 

**Authors**: Shutong Qiao, Wei Yuan, Junliang Yu, Tong Chen, Quoc Viet Hung Nguyen, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.01987)  

**Abstract**: Recommender systems (RSs) are now fundamental to various online platforms, but their dependence on user-contributed data leaves them vulnerable to shilling attacks that can manipulate item rankings by injecting fake users. Although widely studied, most existing attack models fail to meet two critical objectives simultaneously: achieving strong adversarial promotion of target items while maintaining realistic behavior to evade detection. As a result, the true severity of shilling threats that manage to reconcile the two objectives remains underappreciated. To expose this overlooked vulnerability, we present DLDA, a diffusion-based attack framework that can generate highly effective yet indistinguishable fake users by enabling fine-grained control over target promotion. Specifically, DLDA operates in a pre-aligned collaborative embedding space, where it employs a conditional latent diffusion process to iteratively synthesize fake user profiles with precise target item control. To evade detection, DLDA introduces a dispersive regularization mechanism that promotes variability and realism in generated behavioral patterns. Extensive experiments on three real-world datasets and five popular RS models demonstrate that, compared to prior attacks, DLDA consistently achieves stronger item promotion while remaining harder to detect. These results highlight that modern RSs are more vulnerable than previously recognized, underscoring the urgent need for more robust defenses. 

**Abstract (ZH)**: 基于扩散的推荐系统欺骗攻击框架（DLDA）：实现目标项的有效推广与自然行为伪装 

---
# Flow-Aware GNN for Transmission Network Reconfiguration via Substation Breaker Optimization 

**Title (ZH)**: 基于变电站断路器优化的流感知GNN传输网络重构 

**Authors**: Dekang Meng, Rabab Haider, Pascal van Hentenryck  

**Link**: [PDF](https://arxiv.org/pdf/2508.01951)  

**Abstract**: This paper introduces OptiGridML, a machine learning framework for discrete topology optimization in power grids. The task involves selecting substation breaker configurations that maximize cross-region power exports, a problem typically formulated as a mixed-integer program (MIP) that is NP-hard and computationally intractable for large networks. OptiGridML replaces repeated MIP solves with a two-stage neural architecture: a line-graph neural network (LGNN) that approximates DC power flows for a given network topology, and a heterogeneous GNN (HeteroGNN) that predicts breaker states under structural and physical constraints. A physics-informed consistency loss connects these components by enforcing Kirchhoff's law on predicted flows. Experiments on synthetic networks with up to 1,000 breakers show that OptiGridML achieves power export improvements of up to 18% over baseline topologies, while reducing inference time from hours to milliseconds. These results demonstrate the potential of structured, flow-aware GNNs for accelerating combinatorial optimization in physical networked systems. 

**Abstract (ZH)**: 基于电力网络的离散拓扑优化的OptiGridML机器学习框架 

---
# Inferring Reward Machines and Transition Machines from Partially Observable Markov Decision Processes 

**Title (ZH)**: 从部分可观测马尔可夫决策过程推断奖励机器和转移机器 

**Authors**: Yuly Wu, Jiamou Liu, Libo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01947)  

**Abstract**: Partially Observable Markov Decision Processes (POMDPs) are fundamental to many real-world applications. Although reinforcement learning (RL) has shown success in fully observable domains, learning policies from traces in partially observable environments remains challenging due to non-Markovian observations. Inferring an automaton to handle the non-Markovianity is a proven effective approach, but faces two limitations: 1) existing automaton representations focus only on reward-based non-Markovianity, leading to unnatural problem formulations; 2) inference algorithms face enormous computational costs. For the first limitation, we introduce Transition Machines (TMs) to complement existing Reward Machines (RMs). To develop a unified inference algorithm for both automata types, we propose the Dual Behavior Mealy Machine (DBMM) that subsumes both TMs and RMs. We then introduce DB-RPNI, a passive automata learning algorithm that efficiently infers DBMMs while avoiding the costly reductions required by prior work. We further develop optimization techniques and identify sufficient conditions for inferring the minimal correct automata. Experimentally, our inference method achieves speedups of up to three orders of magnitude over SOTA baselines. 

**Abstract (ZH)**: 部分可观测马尔可夫决策过程（POMDPs）在许多实际应用中是基础性的。尽管强化学习（RL）在完全可观测领域取得了成功，但在部分可观测环境中从轨迹学习策略由于非马尔可夫观察结果仍然具有挑战性。通过自动机推理非马尔可夫性是一种 proven 有效的办法，但面临两个限制：1）现有的自动机表示只关注基于奖励的非马尔可夫性，导致非自然的问题表述；2）推理算法面临巨大的计算成本。为了解决第一个限制，我们引入转换机（TMs）来补充现有的奖励机（RMs）。为了一致地为这两种自动机类型开发推理算法，我们提出了双行为梅利机（DBMM），它概括了TMs和RMs。然后，我们引入DB-RPNI，这是一种被动自动机学习算法，可以高效地推理DBMMs，同时避免了先前工作中所需的昂贵归约。我们还开发了优化技术，并确定了推理最少正确自动机的充分条件。实验结果显示，我们的推理方法在与最优基线方法相比时，性能提高了三个数量级。 

---
# Proactive Disentangled Modeling of Trigger-Object Pairings for Backdoor Defense 

**Title (ZH)**: 主动解耦建模触发-对象配对以防御后门攻击 

**Authors**: Kyle Stein, Andrew A. Mahyari, Guillermo Francia III, Eman El-Sheikh  

**Link**: [PDF](https://arxiv.org/pdf/2508.01932)  

**Abstract**: Deep neural networks (DNNs) and generative AI (GenAI) are increasingly vulnerable to backdoor attacks, where adversaries embed triggers into inputs to cause models to misclassify or misinterpret target labels. Beyond traditional single-trigger scenarios, attackers may inject multiple triggers across various object classes, forming unseen backdoor-object configurations that evade standard detection pipelines. In this paper, we introduce DBOM (Disentangled Backdoor-Object Modeling), a proactive framework that leverages structured disentanglement to identify and neutralize both seen and unseen backdoor threats at the dataset level. Specifically, DBOM factorizes input image representations by modeling triggers and objects as independent primitives in the embedding space through the use of Vision-Language Models (VLMs). By leveraging the frozen, pre-trained encoders of VLMs, our approach decomposes the latent representations into distinct components through a learnable visual prompt repository and prompt prefix tuning, ensuring that the relationships between triggers and objects are explicitly captured. To separate trigger and object representations in the visual prompt repository, we introduce the trigger-object separation and diversity losses that aids in disentangling trigger and object visual features. Next, by aligning image features with feature decomposition and fusion, as well as learned contextual prompt tokens in a shared multimodal space, DBOM enables zero-shot generalization to novel trigger-object pairings that were unseen during training, thereby offering deeper insights into adversarial attack patterns. Experimental results on CIFAR-10 and GTSRB demonstrate that DBOM robustly detects poisoned images prior to downstream training, significantly enhancing the security of DNN training pipelines. 

**Abstract (ZH)**: 深度神经网络（DNNs）和生成AI（GenAI）日益面临后门攻击的威胁，攻击者会将触发器嵌入到输入中以导致模型错误分类或误解释目标标签。除了传统的单一触发器场景外，攻击者可能在各类物体中注入多个触发器，形成未见的后门-物体配置，从而逃避标准检测管道。在本文中，我们引入了DBOM（分离后门-物体建模）这一先发制人的框架，利用结构化分离来识别并在数据集级别中中和可见和不可见的后门威胁。具体而言，DBOM 通过视觉语言模型（VLMs）在嵌入空间中将触发器和物体建模为独立的基本要素，从而对输入图像表示进行因子分解。借助VLMs的冷冻预训练编码器，我们的方法通过可学习的视觉提示库和提示前缀调优将潜在表示分解为不同的组成部分，确保捕获触发器和物体之间的关系。为了在视觉提示库中分离触发器和物体表示，我们引入了触发器-物体分离和多样性的损失，以帮助分离触发器和物体的视觉特征。接下来，通过将图像特征与特征分解和融合对齐，以及学习到的多模态共享上下文提示令牌，DBOM 使模型能够零样本泛化到训练过程中未见过的新颖触发器-物体配对，从而提供对抗攻击模式的深入洞察。实验结果表明，DBOM 能在下游训练之前 robust 地检测受污染图像，显著增强 DNN 训练管道的安全性。 

---
# Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning 

**Title (ZH)**: 使用无监督学习分解表示空间为可解释子空间 

**Authors**: Xinting Huang, Michael Hahn  

**Link**: [PDF](https://arxiv.org/pdf/2508.01916)  

**Abstract**: Understanding internal representations of neural models is a core interest of mechanistic interpretability. Due to its large dimensionality, the representation space can encode various aspects about inputs. To what extent are different aspects organized and encoded in separate subspaces? Is it possible to find these ``natural'' subspaces in a purely unsupervised way? Somewhat surprisingly, we can indeed achieve this and find interpretable subspaces by a seemingly unrelated training objective. Our method, neighbor distance minimization (NDM), learns non-basis-aligned subspaces in an unsupervised manner. Qualitative analysis shows subspaces are interpretable in many cases, and encoded information in obtained subspaces tends to share the same abstract concept across different inputs, making such subspaces similar to ``variables'' used by the model. We also conduct quantitative experiments using known circuits in GPT-2; results show a strong connection between subspaces and circuit variables. We also provide evidence showing scalability to 2B models by finding separate subspaces mediating context and parametric knowledge routing. Viewed more broadly, our findings offer a new perspective on understanding model internals and building circuits. 

**Abstract (ZH)**: 理解神经模型的内部表示是机制可解释性的核心兴趣。由于表示空间的高维度，它可以编码输入的各种方面。不同的方面在独立的子空间中组织和编码到什么程度？是否有可能以纯粹无监督的方式找到这些“自然”的子空间？令人惊讶的是，我们确实可以通过一个看似无关的训练目标来实现这一点，并在此过程中找到可解释的子空间。我们的方法，邻近距离最小化（NDM），以无监督的方式学习非基底对齐的子空间。定性分析表明，在许多情况下，子空间是可解释的，并且在获得的子空间中编码的信息在不同输入上往往共享相同的抽象概念，从而使这些子空间类似于模型中使用的“变量”。我们还使用GPT-2中的已知电路进行了定量实验；结果表明子空间与电路变量之间存在强烈联系。我们还提供了证据，证明可以通过找到分别介导上下文和参数知识路由的独立子空间来实现2B模型的可扩展性。更广泛地看，我们的发现为理解模型内部结构和构建电路提供了新的视角。 

---
# Complete Evasion, Zero Modification: PDF Attacks on AI Text Detection 

**Title (ZH)**: 完整的规避，零修改：针对AI文本检测的PDF攻击 

**Authors**: Aldan Creo  

**Link**: [PDF](https://arxiv.org/pdf/2508.01887)  

**Abstract**: AI-generated text detectors have become essential tools for maintaining content authenticity, yet their robustness against evasion attacks remains questionable. We present PDFuzz, a novel attack that exploits the discrepancy between visual text layout and extraction order in PDF documents. Our method preserves exact textual content while manipulating character positioning to scramble extraction sequences. We evaluate this approach against the ArguGPT detector using a dataset of human and AI-generated text. Our results demonstrate complete evasion: detector performance drops from (93.6 $\pm$ 1.4) % accuracy and 0.938 $\pm$ 0.014 F1 score to random-level performance ((50.4 $\pm$ 3.2) % accuracy, 0.0 F1 score) while maintaining perfect visual fidelity. Our work reveals a vulnerability in current detection systems that is inherent to PDF document structures and underscores the need for implementing sturdy safeguards against such attacks. We make our code publicly available at this https URL. 

**Abstract (ZH)**: 基于AI生成文本的PDFuzz攻击：利用PDF文档中可视化文本布局与提取顺序之间的差异，同时保持文本内容不变，扰乱提取顺序，以实现完全 evasion。 

---
# Counterfactual Reciprocal Recommender Systems for User-to-User Matching 

**Title (ZH)**: 用户间匹配的反事实互惠推荐系统 

**Authors**: Kazuki Kawamura, Takuma Udagawa, Kei Tateno  

**Link**: [PDF](https://arxiv.org/pdf/2508.01867)  

**Abstract**: Reciprocal recommender systems (RRS) in dating, gaming, and talent platforms require mutual acceptance for a match. Logged data, however, over-represents popular profiles due to past exposure policies, creating feedback loops that skew learning and fairness. We introduce Counterfactual Reciprocal Recommender Systems (CFRR), a causal framework to mitigate this bias. CFRR uses inverse propensity scored, self-normalized objectives. Experiments show CFRR improves NDCG@10 by up to 3.5% (e.g., from 0.459 to 0.475 on DBLP, from 0.299 to 0.307 on Synthetic), increases long-tail user coverage by up to 51% (from 0.504 to 0.763 on Synthetic), and reduces Gini exposure inequality by up to 24% (from 0.708 to 0.535 on Synthetic). CFRR offers a promising approach for more accurate and fair user-to-user matching. 

**Abstract (ZH)**: -counterfactual 双向推荐系统（CFRR）在 dating、gaming 和 talent 平台上需要相互接受才能匹配。然而，由于过去的曝光政策，记录数据过度代表了受欢迎的资料，从而造成了反馈循环，扭曲了学习和公平性。我们引入了 Counterfactual 双向推荐系统（CFRR），这是一种因果框架，用于减轻这种偏见。CFRR 使用逆概率加权和自规范化目标。实验表明，CFRR 可以提高 NDCG@10 最多 3.5%（例如，在 DBLP 上从 0.459 提高到 0.475，在 Synthetic 上从 0.299 提高到 0.307），增加长尾用户覆盖率最多 51%（在 Synthetic 上从 0.504 增加到 0.763），并减少 Gini � Exposures 不平等最多 24%（在 Synthetic 上从 0.708 减少到 0.535）。CFRR 提供了一种更准确和公平的用户与用户匹配的有前途的方法。 

---
# ACT-Tensor: Tensor Completion Framework for Financial Dataset Imputation 

**Title (ZH)**: ACT-Tensor: 金融数据集插补的张量完成框架 

**Authors**: Junyi Mo, Jiayu Li, Duo Zhang, Elynn Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.01861)  

**Abstract**: Missing data in financial panels presents a critical obstacle, undermining asset-pricing models and reducing the effectiveness of investment strategies. Such panels are often inherently multi-dimensional, spanning firms, time, and financial variables, which adds complexity to the imputation task. Conventional imputation methods often fail by flattening the data's multidimensional structure, struggling with heterogeneous missingness patterns, or overfitting in the face of extreme data sparsity. To address these limitations, we introduce an Adaptive, Cluster-based Temporal smoothing tensor completion framework (ACT-Tensor) tailored for severely and heterogeneously missing multi-dimensional financial data panels. ACT-Tensor incorporates two key innovations: a cluster-based completion module that captures cross-sectional heterogeneity by learning group-specific latent structures; and a temporal smoothing module that proactively removes short-lived noise while preserving slow-moving fundamental trends. Extensive experiments show that ACT-Tensor consistently outperforms state-of-the-art benchmarks in terms of imputation accuracy across a range of missing data regimes, including extreme sparsity scenarios. To assess its practical financial utility, we evaluate the imputed data with an asset-pricing pipeline tailored for tensor-structured financial data. Results show that ACT-Tensor not only reduces pricing errors but also significantly improves risk-adjusted returns of the constructed portfolio. These findings confirm that our method delivers highly accurate and informative imputations, offering substantial value for financial decision-making. 

**Abstract (ZH)**: 适应性基于聚类的时间光滑张量补全框架（ACT-Tensor）：应对多维金融面板数据的缺失 

---
# Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents 

**Title (ZH)**: Web-CogReasoner: 向导知识驱动的认知推理for Web代理 

**Authors**: Yuhan Guo, Cong Guo, Aiwen Sun, Hongliang He, Xinyu Yang, Yue Lu, Yingji Zhang, Xuntao Guo, Dong Zhang, Jianzhuang Liu, Jiang Duan, Yijia Xiao, Liangjian Wen, Hai-Ming Xu, Yong Dai  

**Link**: [PDF](https://arxiv.org/pdf/2508.01858)  

**Abstract**: Multimodal large-scale models have significantly advanced the development of web agents, enabling perception and interaction with digital environments akin to human cognition. In this paper, we argue that web agents must first acquire sufficient knowledge to effectively engage in cognitive reasoning. Therefore, we decompose a web agent's capabilities into two essential stages: knowledge content learning and cognitive processes. To formalize this, we propose Web-CogKnowledge Framework, categorizing knowledge as Factual, Conceptual, and Procedural. In this framework, knowledge content learning corresponds to the agent's processes of Memorizing and Understanding, which rely on the first two knowledge types, representing the "what" of learning. Conversely, cognitive processes correspond to Exploring, grounded in Procedural knowledge, defining the "how" of reasoning and action. To facilitate knowledge acquisition, we construct the Web-CogDataset, a structured resource curated from 14 real-world websites, designed to systematically instill core knowledge necessary for web agent. This dataset serves as the agent's conceptual grounding-the "nouns" upon which comprehension is built-as well as the basis for learning how to reason and act. Building on this foundation, we operationalize these processes through a novel knowledge-driven Chain-of-Thought (CoT) reasoning framework, developing and training our proposed agent, the Web-CogReasoner. Extensive experimentation reveals its significant superiority over existing models, especially in generalizing to unseen tasks where structured knowledge is decisive. To enable rigorous evaluation, we introduce the Web-CogBench, a comprehensive evaluation suite designed to assess and compare agent performance across the delineated knowledge domains and cognitive capabilities. Our code and data is open sourced at this https URL 

**Abstract (ZH)**: 多模态大规模模型显著推动了网络代理的发展，使其能够像人类认知一样感知和交互数字环境。在本文中，我们argue认为网络代理必须首先获得足够的知识才能有效进行认知推理。因此，我们将网络代理的能力分解为两个关键阶段：知识内容学习和认知过程。为了形式化这一点，我们提出了Web-CogKnowledge框架，将知识分类为事实性知识、概念性和程序性知识。在该框架中，知识内容学习对应于代理的记忆与理解过程，依赖于前两种知识类型，代表了学习的“是什么”。相反，认知过程基于程序性知识，对应于探索，定义了推理和行动的“怎么做”。为了促进知识获取，我们构建了Web-CogDataset，这是从14个真实网站中精心整理而成的结构化资源，旨在系统地传授网络代理所需的核心知识。该数据集作为代理的理解基础——即构建理解的“名词”，同时也是学习如何推理和行动的基础。在此基础上，我们通过一种新颖的知识驱动的Chain-of-Thought（CoT）推理框架来操作这些过程，开发并训练了我们所提出的Web-CogReasoner代理。广泛的实验表明，该代理在泛化能力方面显著优于现有模型，特别是在结构化知识至关重要的未知任务中。为了进行严格的评估，我们引入了Web-CogBench，这是一个全面的评估套件，旨在评估和比较代理在划分的知识领域和认知能力方面的性能。我们的代码和数据在此开源：this https URL。 

---
# Neural Predictive Control to Coordinate Discrete- and Continuous-Time Models for Time-Series Analysis with Control-Theoretical Improvements 

**Title (ZH)**: 基于神经预测控制的离散-连续时间模型协调方法及其控制理论改进的时间序列分析 

**Authors**: Haoran Li, Muhao Guo, Yang Weng, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2508.01833)  

**Abstract**: Deep sequence models have achieved notable success in time-series analysis, such as interpolation and forecasting. Recent advances move beyond discrete-time architectures like Recurrent Neural Networks (RNNs) toward continuous-time formulations such as the family of Neural Ordinary Differential Equations (Neural ODEs). Generally, they have shown that capturing the underlying dynamics is beneficial for generic tasks like interpolation, extrapolation, and classification. However, existing methods approximate the dynamics using unconstrained neural networks, which struggle to adapt reliably under distributional shifts. In this paper, we recast time-series problems as the continuous ODE-based optimal control problem. Rather than learning dynamics solely from data, we optimize control actions that steer ODE trajectories toward task objectives, bringing control-theoretical performance guarantees. To achieve this goal, we need to (1) design the appropriate control actions and (2) apply effective optimal control algorithms. As the actions should contain rich context information, we propose to employ the discrete-time model to process past sequences and generate actions, leading to a coordinate model to extract long-term temporal features to modulate short-term continuous dynamics. During training, we apply model predictive control to plan multi-step future trajectories, minimize a task-specific cost, and greedily select the optimal current action. We show that, under mild assumptions, this multi-horizon optimization leads to exponential convergence to infinite-horizon solutions, indicating that the coordinate model can gain robust and generalizable performance. Extensive experiments on diverse time-series datasets validate our method's superior generalization and adaptability compared to state-of-the-art baselines. 

**Abstract (ZH)**: 深度序列模型在时间序列分析、插值和预测方面取得了显著成果。最近的进步超越了离散时间架构如循环神经网络（RNNs），转向了神经常微分方程（Neural ODEs）等连续时间形式。通常，它们显示捕获潜在动态对于通用任务如插值、外推和分类是有益的。然而，现有方法使用无约束神经网络近似动态，这在分布转移下难以可靠适应。在本文中，我们将时间序列问题重新表述为基于连续微分方程的最优控制问题。我们不仅从数据中学习动态，还优化控制动作，使其引导常微分方程轨迹趋向任务目标，从而带来控制理论的性能保证。为了实现这一目标，我们需要（1）设计适当的控制动作，（2）应用有效的最优控制算法。由于动作应包含丰富的上下文信息，我们提出使用离散时间模型处理过去序列并生成动作，导致坐标模型提取长期时间特征以调节短期连续动态。在训练过程中，我们应用模型预测控制规划多步未来轨迹，最小化特定任务成本，并贪婪地选择当前最优动作。我们显示，假设较多，此多视窗优化可导致指数收敛至无限视窗解，表明坐标模型可以获得稳健且可泛化的性能。广泛的时间序列数据集实验验证了我们方法在泛化和适应性方面的优越性，优于最先进的基线方法。 

---
# AGENTICT$^2$S:Robust Text-to-SPARQL via Agentic Collaborative Reasoning over Heterogeneous Knowledge Graphs for the Circular Economy 

**Title (ZH)**: 基于代理协作推理的AGENTICT$^2$S：面向循环经济的 robust 文本到 SPARQL 转换 

**Authors**: Yang Zhao, Chengxiao Dai, Wei Zhuo, Tan Chuan Fu, Yue Xiu, Dusit Niyato, Jonathan Z. Low, Eugene Ho Hong Zhuang, Daren Zong Loong Tan  

**Link**: [PDF](https://arxiv.org/pdf/2508.01815)  

**Abstract**: Question answering over heterogeneous knowledge graphs (KGQA) involves reasoning across diverse schemas, incomplete alignments, and distributed data sources. Existing text-to-SPARQL approaches rely on large-scale domain-specific fine-tuning or operate within single-graph settings, limiting their generalizability in low-resource domains and their ability to handle queries spanning multiple graphs. These challenges are particularly relevant in domains such as the circular economy, where information about classifications, processes, and emissions is distributed across independently curated knowledge graphs (KGs). We present AgenticT$^2$S, a modular framework that decomposes KGQA into subtasks managed by specialized agents responsible for retrieval, query generation, and verification. A scheduler assigns subgoals to different graphs using weak-to-strong alignment strategies. A two-stage verifier detects structurally invalid and semantically underspecified queries through symbolic validation and counterfactual consistency checks. Experiments on real-world circular economy KGs demonstrate that AgenticT$^2$S improves execution accuracy by 17.3% and triple level F$_1$ by 25.4% over the best baseline, while reducing the average prompt length by 46.4%. These results demonstrate the benefits of agent-based schema-aware reasoning for scalable KGQA and support decision-making in sustainability domains through robust cross-graph reasoning. 

**Abstract (ZH)**: 基于异构知识图谱的问答（KGQA）涉及跨越多样模式、不完整对齐和分布式数据源的推理。现有的文本到SPARQL方法依赖于大规模领域特定的微调或限定在单个图的环境中，限制了其在资源贫乏领域中的普适性及其处理跨越多个图的查询的能力。这些挑战在循环经济等领域尤为相关，在这些领域中，关于分类、过程和排放的信息分布在独立维护的知识图谱（KGs）中。我们提出了一种模块化框架AgenticT$^2$S，将KGQA分解为由专门代理管理的子任务，这些代理负责检索、查询生成和验证。调度器使用从弱到强的对齐策略为不同的图分配子目标。两阶段验证器通过符号验证和反事实一致性检查检测结构上无效和语义上不明确的查询。实验证实在实际的循环经济KG上，AgenticT$^2$S将执行准确性提高了17.3%，三元组水平的F$_1$分数提高了25.4%，同时将平均提示长度减少了46.4%。这些结果表明基于代理的模式感知推理对于可扩展的KGQA的好处，并通过稳健的跨图推理支持可持续性领域中的决策制定。 

---
# HeQ: a Large and Diverse Hebrew Reading Comprehension Benchmark 

**Title (ZH)**: HeQ：一个大规模且多样的希伯来阅读理解基准 

**Authors**: Amir DN Cohen, Hilla Merhav, Yoav Goldberg, Reut Tsarfaty  

**Link**: [PDF](https://arxiv.org/pdf/2508.01812)  

**Abstract**: Current benchmarks for Hebrew Natural Language Processing (NLP) focus mainly on morpho-syntactic tasks, neglecting the semantic dimension of language understanding. To bridge this gap, we set out to deliver a Hebrew Machine Reading Comprehension (MRC) dataset, where MRC is to be realized as extractive Question Answering. The morphologically rich nature of Hebrew poses a challenge to this endeavor: the indeterminacy and non-transparency of span boundaries in morphologically complex forms lead to annotation inconsistencies, disagreements, and flaws in standard evaluation metrics.
To remedy this, we devise a novel set of guidelines, a controlled crowdsourcing protocol, and revised evaluation metrics that are suitable for the morphologically rich nature of the language. Our resulting benchmark, HeQ (Hebrew QA), features 30,147 diverse question-answer pairs derived from both Hebrew Wikipedia articles and Israeli tech news. Our empirical investigation reveals that standard evaluation metrics such as F1 scores and Exact Match (EM) are not appropriate for Hebrew (and other MRLs), and we propose a relevant enhancement.
In addition, our experiments show low correlation between models' performance on morpho-syntactic tasks and on MRC, which suggests that models designed for the former might underperform on semantics-heavy tasks. The development and exploration of HeQ illustrate some of the challenges MRLs pose in natural language understanding (NLU), fostering progression towards more and better NLU models for Hebrew and other MRLs. 

**Abstract (ZH)**: 当前用于_hebrew_自然语言处理(NLP)的基准主要集中在形态学和句法任务上，忽视了语言理解的语义维度。为了弥合这一差距，我们致力于提供一个希伯来机器阅读理解(MRC)数据集，其中MRC将实现为抽取式问答。希伯来语丰富的形态学特性给这一努力带来了挑战：形态学复杂形式中的跨度边界模糊和不透明导致了标注不一致、歧义和标准评价指标中的缺陷。

为了解决这个问题，我们设计了一套新的指南，一个受控的众包协议，以及适用于该语言丰富形态学特性的修订评价指标。我们得到的基准数据集HeQ (希伯来语QA)包含了30,147个来自希伯来维基百科文章和以色列科技新闻的多样化的问答对。我们的实证研究表明，标准评价指标如F1分数和精确匹配(EM)对于希伯来语（以及其他MRLs）并不合适，并提出了相关的改进方案。

此外，我们的实验表明，模型在形态学和句法任务上的表现与在MRC上的表现之间存在较低的相关性，这表明适用于前者的设计可能在语义密集型任务上表现不佳。HeQ的发展和探索揭示了MRLs在自然语言理解(NLU)中所面临的挑战，促进了对希伯来语和其他MRLs更先进、更高质量的NLU模型的发展。 

---
# Mitigating Persistent Client Dropout in Asynchronous Decentralized Federated Learning 

**Title (ZH)**: 缓解异步去中心化联邦学习中的持续客户端退出问题 

**Authors**: Ignacy Stępka, Nicholas Gisolfi, Kacper Trębacz, Artur Dubrawski  

**Link**: [PDF](https://arxiv.org/pdf/2508.01807)  

**Abstract**: We consider the problem of persistent client dropout in asynchronous Decentralized Federated Learning (DFL). Asynchronicity and decentralization obfuscate information about model updates among federation peers, making recovery from a client dropout difficult. Access to the number of learning epochs, data distributions, and all the information necessary to precisely reconstruct the missing neighbor's loss functions is limited. We show that obvious mitigations do not adequately address the problem and introduce adaptive strategies based on client reconstruction. We show that these strategies can effectively recover some performance loss caused by dropout. Our work focuses on asynchronous DFL with local regularization and differs substantially from that in the existing literature. We evaluate the proposed methods on tabular and image datasets, involve three DFL algorithms, and three data heterogeneity scenarios (iid, non-iid, class-focused non-iid). Our experiments show that the proposed adaptive strategies can be effective in maintaining robustness of federated learning, even if they do not reconstruct the missing client's data precisely. We also discuss the limitations and identify future avenues for tackling the problem of client dropout. 

**Abstract (ZH)**: 我们在异步去中心化联邦学习中的持久客户端辍学问题 

---
# Contrastive Multi-Task Learning with Solvent-Aware Augmentation for Drug Discovery 

**Title (ZH)**: 溶剂 Awareness 增强的对比多任务学习在药物发现中的应用 

**Authors**: Jing Lan, Hexiao Ding, Hongzhao Chen, Yufeng Jiang, Ng Nga Chun, Gerald W.Y. Cheng, Zongxi Li, Jing Cai, Liang-ting Lin, Jung Sun Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2508.01799)  

**Abstract**: Accurate prediction of protein-ligand interactions is essential for computer-aided drug discovery. However, existing methods often fail to capture solvent-dependent conformational changes and lack the ability to jointly learn multiple related tasks. To address these limitations, we introduce a pre-training method that incorporates ligand conformational ensembles generated under diverse solvent conditions as augmented input. This design enables the model to learn both structural flexibility and environmental context in a unified manner. The training process integrates molecular reconstruction to capture local geometry, interatomic distance prediction to model spatial relationships, and contrastive learning to build solvent-invariant molecular representations. Together, these components lead to significant improvements, including a 3.7% gain in binding affinity prediction, an 82% success rate on the PoseBusters Astex docking benchmarks, and an area under the curve of 97.1% in virtual screening. The framework supports solvent-aware, multi-task modeling and produces consistent results across benchmarks. A case study further demonstrates sub-angstrom docking accuracy with a root-mean-square deviation of 0.157 angstroms, offering atomic-level insight into binding mechanisms and advancing structure-based drug design. 

**Abstract (ZH)**: 准确预测蛋白质-配体相互作用对于计算机辅助药物发现至关重要。然而，现有方法往往难以捕捉溶剂依赖的构象变化，且缺乏联合学习多个相关任务的能力。为解决这些局限性，我们提出了一种预训练方法，该方法将不同溶剂条件下生成的配体构象ensemble作为增强输入纳入其中。该设计使模型能够以统一的方式学习结构灵活性和环境上下文。训练过程整合了分子重构以捕获局部几何结构、原子间距离预测以建模空间关系，以及对比学习以构建溶剂不变的分子表示。这些组成部分共同带来了显著的改进，包括3.7%的结合亲和力预测提升、PoseBusters Astex 锚定基准测试中82%的成功率，以及虚拟筛选中97.1%的曲线下面积。该框架支持溶剂感知的多任务建模，并在不同的基准测试中产生一致的结果。进一步的案例研究展示了亚埃级别的锚定准确性，根均方偏差为0.157埃，提供了Binding机制的原子级洞察，并推动了基于结构的药物设计。 

---
# RouteMark: A Fingerprint for Intellectual Property Attribution in Routing-based Model Merging 

**Title (ZH)**: RouteMark：基于路由合并模型中的知识产权归属指纹技术 

**Authors**: Xin He, Junxi Shen, Zhenheng Tang, Xiaowen Chu, Bo Li, Ivor W. Tsang, Yew-Soon Ong  

**Link**: [PDF](https://arxiv.org/pdf/2508.01784)  

**Abstract**: Model merging via Mixture-of-Experts (MoE) has emerged as a scalable solution for consolidating multiple task-specific models into a unified sparse architecture, where each expert is derived from a model fine-tuned on a distinct task. While effective for multi-task integration, this paradigm introduces a critical yet underexplored challenge: how to attribute and protect the intellectual property (IP) of individual experts after merging. We propose RouteMark, a framework for IP protection in merged MoE models through the design of expert routing fingerprints. Our key insight is that task-specific experts exhibit stable and distinctive routing behaviors under probing inputs. To capture these patterns, we construct expert-level fingerprints using two complementary statistics: the Routing Score Fingerprint (RSF), quantifying the intensity of expert activation, and the Routing Preference Fingerprint (RPF), characterizing the input distribution that preferentially activates each expert. These fingerprints are reproducible, task-discriminative, and lightweight to construct. For attribution and tampering detection, we introduce a similarity-based matching algorithm that compares expert fingerprints between a suspect and a reference (victim) model. Extensive experiments across diverse tasks and CLIP-based MoE architectures show that RouteMark consistently yields high similarity for reused experts and clear separation from unrelated ones. Moreover, it remains robust against both structural tampering (expert replacement, addition, deletion) and parametric tampering (fine-tuning, pruning, permutation), outperforming weight- and activation-based baseliness. Our work lays the foundation for RouteMark as a practical and broadly applicable framework for IP verification in MoE-based model merging. 

**Abstract (ZH)**: 基于Mixture-of-Experts (MoE)的模型合并中的知识产权保护：RouteMark框架 

---
# VAGPO: Vision-augmented Asymmetric Group Preference Optimization for the Routing Problems 

**Title (ZH)**: 基于视觉增强非对称群体偏好的路径优化方法 

**Authors**: Shiyan Liu, Bohan Tan, Yan Jin  

**Link**: [PDF](https://arxiv.org/pdf/2508.01774)  

**Abstract**: The routing problems such as the Traveling Salesman Problem (TSP) and the Capacitated Vehicle Routing Problem (CVRP) are well-known combinatorial optimization challenges with broad practical relevance. Recent data-driven optimization methods have made significant progress, yet they often face limitations in training efficiency and generalization to large-scale instances. In this paper, we propose a novel Vision-Augmented Asymmetric Group Preference Optimization (VAGPO) approach for solving the routing problems. By leveraging ResNet-based visual encoding and Transformer-based sequential modeling, VAGPO captures both spatial structure and temporal dependencies. Furthermore, we introduce an asymmetric group preference optimization strategy that significantly accelerates convergence compared to commonly used policy gradient methods. Experimental results on TSP and CVRP benchmarks show that the proposed VAGPO not only achieves highly competitive solution quality but also exhibits strong generalization to larger instances (up to 1000 nodes) without re-training, highlighting its effectiveness in both learning efficiency and scalability. 

**Abstract (ZH)**: 视觉增强非对称组偏好优化方法（VAGPO）求解路由问题 

---
# Semantically-Guided Inference for Conditional Diffusion Models: Enhancing Covariate Consistency in Time Series Forecasting 

**Title (ZH)**: 基于语义指导的条件扩散模型推理：增强时间序列预测中的协变量一致性 

**Authors**: Rui Ding, Hanyang Meng, Zeyang Zhang, Jielong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01761)  

**Abstract**: Diffusion models have demonstrated strong performance in time series forecasting, yet often suffer from semantic misalignment between generated trajectories and conditioning covariates, especially under complex or multimodal conditions. To address this issue, we propose SemGuide, a plug-and-play, inference-time method that enhances covariate consistency in conditional diffusion models. Our approach introduces a scoring network to assess the semantic alignment between intermediate diffusion states and future covariates. These scores serve as proxy likelihoods in a stepwise importance reweighting procedure, which progressively adjusts the sampling path without altering the original training process. The method is model-agnostic and compatible with any conditional diffusion framework. Experiments on real-world forecasting tasks show consistent gains in both predictive accuracy and covariate alignment, with especially strong performance under complex conditioning scenarios. 

**Abstract (ZH)**: 基于注释指导的条件差分模型语义一致性增强方法：应用于时间序列预测 

---
# Improving Noise Efficiency in Privacy-preserving Dataset Distillation 

**Title (ZH)**: 提高隐私保护数据蒸馏中的噪声效率 

**Authors**: Runkai Zheng, Vishnu Asutosh Dasu, Yinong Oliver Wang, Haohan Wang, Fernando De la Torre  

**Link**: [PDF](https://arxiv.org/pdf/2508.01749)  

**Abstract**: Modern machine learning models heavily rely on large datasets that often include sensitive and private information, raising serious privacy concerns. Differentially private (DP) data generation offers a solution by creating synthetic datasets that limit the leakage of private information within a predefined privacy budget; however, it requires a substantial amount of data to achieve performance comparable to models trained on the original data. To mitigate the significant expense incurred with synthetic data generation, Dataset Distillation (DD) stands out for its remarkable training and storage efficiency. This efficiency is particularly advantageous when integrated with DP mechanisms, curating compact yet informative synthetic datasets without compromising privacy. However, current state-of-the-art private DD methods suffer from a synchronized sampling-optimization process and the dependency on noisy training signals from randomly initialized networks. This results in the inefficient utilization of private information due to the addition of excessive noise. To address these issues, we introduce a novel framework that decouples sampling from optimization for better convergence and improves signal quality by mitigating the impact of DP noise through matching in an informative subspace. On CIFAR-10, our method achieves a \textbf{10.0\%} improvement with 50 images per class and \textbf{8.3\%} increase with just \textbf{one-fifth} the distilled set size of previous state-of-the-art methods, demonstrating significant potential to advance privacy-preserving DD. 

**Abstract (ZH)**: 现代机器学习模型高度依赖大規模数据集，这些数据集通常包含敏感和私人信息，引发了严重的隐私问题。不同隐私（DP）数据生成通过创建合成数据集来限制在预定义隐私预算内的私人信息泄露，但它需要大量数据以达到与原始数据训练模型相当的性能。为了缓解合成数据生成带来的显著成本，数据集蒸馏（DD）因其出色的训练和存储效率而脱颖而出。当与DP机制结合时，这种效率特别有利，可以生成紧凑且富有信息量的合成数据集，而不牺牲隐私。然而，当前最先进的私密DD方法面临同步采样-优化过程和对随机初始化网络噪声训练信号的依赖，导致由于过多添加噪声而导致私人信息的低效利用。为了应对这些问题，我们提出了一种新颖的框架，该框架解耦了采样和优化，通过匹配在信息子空间中降低DP噪声的影响来改善信号质量。在CIFAR-10上，我们的方法在每类50张图片的情况下实现了10.0%的改进，并且仅使用之前最先进的方法五分之一的蒸馏集合大小就实现了8.3%的提升，这表明在保护隐私的同时进行DD有很大前景。 

---
# Granular Concept Circuits: Toward a Fine-Grained Circuit Discovery for Concept Representations 

**Title (ZH)**: 细粒度概念电路：概念表示的精细电路发现 toward 细粒度概念电路：概念表示的精细电路发现 

**Authors**: Dahee Kwon, Sehyun Lee, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2508.01728)  

**Abstract**: Deep vision models have achieved remarkable classification performance by leveraging a hierarchical architecture in which human-interpretable concepts emerge through the composition of individual neurons across layers. Given the distributed nature of representations, pinpointing where specific visual concepts are encoded within a model remains a crucial yet challenging task. In this paper, we introduce an effective circuit discovery method, called Granular Concept Circuit (GCC), in which each circuit represents a concept relevant to a given query. To construct each circuit, our method iteratively assesses inter-neuron connectivity, focusing on both functional dependencies and semantic alignment. By automatically discovering multiple circuits, each capturing specific concepts within that query, our approach offers a profound, concept-wise interpretation of models and is the first to identify circuits tied to specific visual concepts at a fine-grained level. We validate the versatility and effectiveness of GCCs across various deep image classification models. 

**Abstract (ZH)**: 深层次的视觉模型通过利用层次结构，其中个体神经元在各层中的组合产生了可由人类解释的概念，从而实现了显著的分类性能。鉴于表示的分布性质，确定模型中特定视觉概念的编码位置仍然是一个关键而具有挑战性的问题。在本文中，我们介绍了一种有效电路发现方法，称为粒度概念电路（GCC），其中每条电路代表与给定查询相关的概念。为了构建每条电路，我们的方法逐次评估神经元间的连接性，重点在于功能依赖关系和语义对齐。通过自动发现多条电路，每条电路捕捉查询中特定的概念，我们的方法提供了概念层面的深刻解释，并且是首次能够在细粒度级别识别与特定视觉概念相关联的电路。我们在多种深层图像分类模型中验证了GCCs的通用性和有效性。 

---
# HateClipSeg: A Segment-Level Annotated Dataset for Fine-Grained Hate Video Detection 

**Title (ZH)**: HateClipSeg：一种细粒度仇恨视频检测的片段级标注数据集 

**Authors**: Han Wang, Zhuoran Wang, Roy Ka-Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.01712)  

**Abstract**: Detecting hate speech in videos remains challenging due to the complexity of multimodal content and the lack of fine-grained annotations in existing datasets. We present HateClipSeg, a large-scale multimodal dataset with both video-level and segment-level annotations, comprising over 11,714 segments labeled as Normal or across five Offensive categories: Hateful, Insulting, Sexual, Violence, Self-Harm, along with explicit target victim labels. Our three-stage annotation process yields high inter-annotator agreement (Krippendorff's alpha = 0.817). We propose three tasks to benchmark performance: (1) Trimmed Hateful Video Classification, (2) Temporal Hateful Video Localization, and (3) Online Hateful Video Classification. Results highlight substantial gaps in current models, emphasizing the need for more sophisticated multimodal and temporally aware approaches. The HateClipSeg dataset are publicly available at this https URL. 

**Abstract (ZH)**: 检测视频中的仇恨言论仍具有挑战性，由于多模态内容的复杂性和现有数据集中缺少细粒度注释。我们 presents HateClipSeg，一个大规模多模态数据集，包含视频级和片段级注释，共有超过11,714个片段被标记为Normal或五个冒犯类别中的一个：仇恨、侮辱、性相关、暴力、自我伤害，以及明确的目标受害者标签。我们的三阶段注释过程获得了较高的注释者间一致性（Krippendorff’s α = 0.817）。我们提出了三个基准任务：(1) 剪辑仇恨视频分类，(2) 时光轴上的仇恨视频本地化，(3) 实时仇恨视频分类。结果突显了当前模型中的巨大差距，强调了需要更复杂的多模态和时间感知方法的必要性。HateClipSeg数据集可在以下链接获取：this https URL。 

---
# From SHAP to Rules: Distilling Expert Knowledge from Post-hoc Model Explanations in Time Series Classification 

**Title (ZH)**: 从SHAP到规则：从时间序列分类模型后验解释中提炼专家知识 

**Authors**: Maciej Mozolewski, Szymon Bobek, Grzegorz J. Nalepa  

**Link**: [PDF](https://arxiv.org/pdf/2508.01687)  

**Abstract**: Explaining machine learning (ML) models for time series (TS) classification is challenging due to inherent difficulty in raw time series interpretation and doubled down by the high dimensionality. We propose a framework that converts numeric feature attributions from post-hoc, instance-wise explainers (e.g., LIME, SHAP) into structured, human-readable rules. These rules define intervals indicating when and where they apply, improving transparency. Our approach performs comparably to native rule-based methods like Anchor while scaling better to long TS and covering more instances. Rule fusion integrates rule sets through methods such as weighted selection and lasso-based refinement to balance coverage, confidence, and simplicity, ensuring all instances receive an unambiguous, metric-optimized rule. It enhances explanations even for a single explainer. We introduce visualization techniques to manage specificity-generalization trade-offs. By aligning with expert-system principles, our framework consolidates conflicting or overlapping explanations - often resulting from the Rashomon effect - into coherent and domain-adaptable insights. Experiments on UCI datasets confirm that the resulting rule-based representations improve interpretability, decision transparency, and practical applicability for TS classification. 

**Abstract (ZH)**: 基于时间序列分类的机器学习模型解释因原始时间序列解释的固有难度和高维性而具有挑战性。我们提出了一种框架，将后 hoc、实例级别的特征归因（例如，LIME、SHAP）转换为结构化、易于理解的规则。这些规则定义了指示其适用时间和范围的区间，提高了透明度。我们的方法在长时间序列和实例方面具有更好的扩展性，同时覆盖更多实例。规则融合通过加权选择和lasso基优化等方法整合规则集，以平衡覆盖范围、置信度和简洁性，确保所有实例都获得一个明确的、优化了的规则。即使对于单一解释器，这种方法也能增强解释。我们引入了可视化技术来管理具体性和通用性之间的权衡。通过与专家系统原则对齐，我们的框架将来自“拉什门辛效应”导致的矛盾或重叠解释整合为一致且适用于特定领域的见解。实验结果证实，基于规则的表示形式提高了时间序列分类的可解释性、决策透明度和实用适用性。 

---
# SPARTA: Advancing Sparse Attention in Spiking Neural Networks via Spike-Timing-Based Prioritization 

**Title (ZH)**: SPARTA: 基于脉冲时序优先级促进脉冲神经网络中稀疏注意机制的研究 

**Authors**: Minsuk Jang, Changick Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.01646)  

**Abstract**: Current Spiking Neural Networks (SNNs) underutilize the temporal dynamics inherent in spike-based processing, relying primarily on rate coding while overlooking precise timing information that provides rich computational cues. We propose SPARTA (Spiking Priority Attention with Resource-Adaptive Temporal Allocation), a framework that leverages heterogeneous neuron dynamics and spike-timing information to enable efficient sparse attention. SPARTA prioritizes tokens based on temporal cues, including firing patterns, spike timing, and inter-spike intervals, achieving 65.4% sparsity through competitive gating. By selecting only the most salient tokens, SPARTA reduces attention complexity from O(N^2) to O(K^2) with k << n, while maintaining high accuracy. Our method achieves state-of-the-art performance on DVS-Gesture (98.78%) and competitive results on CIFAR10-DVS (83.06%) and CIFAR-10 (95.3%), demonstrating that exploiting spike timing dynamics improves both computational efficiency and accuracy. 

**Abstract (ZH)**: 当前的脉冲神经网络(SNNs)未能充分利用基于脉冲的处理中固有的时间动态性，主要依赖速率编码而忽视了提供丰富计算线索的精确时间信息。我们提出了一种名为SPARTA（Spiking Priority Attention with Resource-Adaptive Temporal Allocation）的框架，该框架利用异质神经元动力学和脉冲时间信息以实现高效的稀疏注意。SPARTA基于时间线索（包括放电模式、脉冲时间以及脉冲间间隔）对token进行优先级排序，通过竞争性门控达到65.4%的稀疏度。通过仅选择最显著的token，SPARTA将注意力复杂度从O(N^2)降低到O(K^2)（其中k<<n），同时保持高准确性。我们的方法在DVS-Gesture（98.78%）和CIFAR10-DVS（83.06%）以及CIFAR-10（95.3%）上实现了最佳性能，证明了利用脉冲时间动态性能够同时提高计算效率和准确性。 

---
# Learning Unified System Representations for Microservice Tail Latency Prediction 

**Title (ZH)**: 面向微服务延迟预测的统一系统表示学习 

**Authors**: Wenzhuo Qian, Hailiang Zhao, Tianlv Chen, Jiayi Chen, Ziqi Wang, Kingsum Chow, Shuiguang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2508.01635)  

**Abstract**: Microservice architectures have become the de facto standard for building scalable cloud-native applications, yet their distributed nature introduces significant challenges in performance monitoring and resource management. Traditional approaches often rely on per-request latency metrics, which are highly sensitive to transient noise and fail to reflect the holistic behavior of complex, concurrent workloads. In contrast, window-level P95 tail latency provides a stable and meaningful signal that captures both system-wide trends and user-perceived performance degradation. We identify two key shortcomings in existing methods: (i) inadequate handling of heterogeneous data, where traffic-side features propagate across service dependencies and resource-side signals reflect localized bottlenecks, and (ii) the lack of principled architectural designs that effectively distinguish and integrate these complementary modalities. To address these challenges, we propose USRFNet, a deep learning network that explicitly separates and models traffic-side and resource-side features. USRFNet employs GNNs to capture service interactions and request propagation patterns, while gMLP modules independently model cluster resource dynamics. These representations are then fused into a unified system embedding to predict window-level P95 latency with high accuracy. We evaluate USRFNet on real-world microservice benchmarks under large-scale stress testing conditions, demonstrating substantial improvements in prediction accuracy over state-of-the-art baselines. 

**Abstract (ZH)**: 面向服务异构数据的分布式微服务架构窗口级别P95尾延迟预测方法及USRFNet网络设计 

---
# OpenMed NER: Open-Source, Domain-Adapted State-of-the-Art Transformers for Biomedical NER Across 12 Public Datasets 

**Title (ZH)**: OpenMed NER：面向12个公开数据集的开源、领域自适应最先进的变换器生物医学命名实体识别方法 

**Authors**: Maziyar Panahi  

**Link**: [PDF](https://arxiv.org/pdf/2508.01630)  

**Abstract**: Named-entity recognition (NER) is fundamental to extracting structured information from the >80% of healthcare data that resides in unstructured clinical notes and biomedical literature. Despite recent advances with large language models, achieving state-of-the-art performance across diverse entity types while maintaining computational efficiency remains a significant challenge. We introduce OpenMed NER, a suite of open-source, domain-adapted transformer models that combine lightweight domain-adaptive pre-training (DAPT) with parameter-efficient Low-Rank Adaptation (LoRA). Our approach performs cost-effective DAPT on a 350k-passage corpus compiled from ethically sourced, publicly available research repositories and de-identified clinical notes (PubMed, arXiv, and MIMIC-III) using DeBERTa-v3, PubMedBERT, and BioELECTRA backbones. This is followed by task-specific fine-tuning with LoRA, which updates less than 1.5% of model parameters. We evaluate our models on 12 established biomedical NER benchmarks spanning chemicals, diseases, genes, and species. OpenMed NER achieves new state-of-the-art micro-F1 scores on 10 of these 12 datasets, with substantial gains across diverse entity types. Our models advance the state-of-the-art on foundational disease and chemical benchmarks (e.g., BC5CDR-Disease, +2.70 pp), while delivering even larger improvements of over 5.3 and 9.7 percentage points on more specialized gene and clinical cell line corpora. This work demonstrates that strategically adapted open-source models can surpass closed-source solutions. This performance is achieved with remarkable efficiency: training completes in under 12 hours on a single GPU with a low carbon footprint (< 1.2 kg CO2e), producing permissively licensed, open-source checkpoints designed to help practitioners facilitate compliance with emerging data protection and AI regulations, such as the EU AI Act. 

**Abstract (ZH)**: 命名实体识别（NER）是从超过80%保存在未结构化临床笔记和生物医学文献中的医疗数据中提取结构化信息的基础。尽管大型语言模型取得了recent进展，但在多种实体类型上实现最先进的性能并保持计算效率仍然是一项重大挑战。我们介绍OpenMed NER，一个结合了轻量级领域适应预训练（DAPT）和参数高效低秩适应（LoRA）的开源领域适应变换器模型套件。我们的方法使用DeBERTa-v3、PubMedBERT和BioELECTRA主干，在包含伦理来源的公开可用研究repositories和去标识化临床笔记（PubMed、arXiv和MIMIC-III）的350k段落语料库上执行成本有效的大规模适应预训练。随后是使用LoRA的任务特定微调，更新少于1.5%的模型参数。我们在涵盖化学物质、疾病、基因和物种的12个现有生物医学NER基准测试上评估了我们的模型。OpenMed NER在这些12个数据集中的10个上实现了新的最先进的微F1分数，针对不同实体类型取得了显著改进。我们的模型在基础疾病和化学基准（例如BC5CDR-Disease，+2.70 pp）上推动了最先进的技术水平，同时在更专业化的基因和临床细胞系语料库上实现了超过5.3和9.7个百分点的更大改进。这项工作表明，战略性调整的开源模型可以超越封闭源解决方案。这种性能通过在一个GPU上在不到12小时内完成训练实现，并具有较低的碳足迹（<1.2 kg CO2e），生成了许可许可的开源检查点，旨在帮助从业者促进遵守新兴的数据保护和AI法规，如欧盟AI法案。 

---
# TCDiff: Triplex Cascaded Diffusion for High-fidelity Multimodal EHRs Generation with Incomplete Clinical Data 

**Title (ZH)**: TCDiff: 三重级联扩散模型用于生成具有不完整临床数据的高保真多模态EHR 

**Authors**: Yandong Yan, Chenxi Li, Yu Huang, Dexuan Xu, Jiaqi Zhu, Zhongyan Chai, Huamin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01615)  

**Abstract**: The scarcity of large-scale and high-quality electronic health records (EHRs) remains a major bottleneck in biomedical research, especially as large foundation models become increasingly data-hungry. Synthesizing substantial volumes of de-identified and high-fidelity data from existing datasets has emerged as a promising solution. However, existing methods suffer from a series of limitations: they struggle to model the intrinsic properties of heterogeneous multimodal EHR data (e.g., continuous, discrete, and textual modalities), capture the complex dependencies among them, and robustly handle pervasive data incompleteness. These challenges are particularly acute in Traditional Chinese Medicine (TCM). To this end, we propose TCDiff (Triplex Cascaded Diffusion Network), a novel EHR generation framework that cascades three diffusion networks to learn the features of real-world EHR data, formatting a multi-stage generative process: Reference Modalities Diffusion, Cross-Modal Bridging, and Target Modality Diffusion. Furthermore, to validate our proposed framework, besides two public datasets, we also construct and introduce TCM-SZ1, a novel multimodal EHR dataset for benchmarking. Experimental results show that TCDiff consistently outperforms state-of-the-art baselines by an average of 10% in data fidelity under various missing rate, while maintaining competitive privacy guarantees. This highlights the effectiveness, robustness, and generalizability of our approach in real-world healthcare scenarios. 

**Abstract (ZH)**: 电子健康记录(EHR)数据的稀缺性仍然是生物医学研究中的一个主要瓶颈，特别是随着大型基础模型变得越来越依赖数据。从现有数据集合成大量去标识且高保真的数据已 emerged 作为一种有前景的解决方案。然而，现有方法存在一系列限制：它们难以建模异质多模态 EHR 数据的本质特性（如连续型、离散型和文本模态），捕捉它们之间的复杂依赖关系，并且难以稳健地处理普遍存在的数据不完整性。这些挑战在中医药 (TCM) 中尤为严重。为此，我们提出了一种新颖的 EHR 生成框架 TCDiff（三重级联扩散网络），该框架级联三个扩散网络以学习现实世界 EHR 数据的特征，格式化一个多阶段生成过程：参考模态扩散、跨模态桥梁构建 和 目标模态扩散。此外，为了验证我们提出的方法，在两个公开数据集的基础上，我们还构建并引入了 TCM-SZ1，这是一个新型多模态 EHR 数据集用于基准测试。实验结果表明，无论在各种缺失率下，TCDiff 在数据保真度方面均比最先进的基线方法平均高出 10%，同时保持了竞争力的隐私保障。这突显了我们方法在实际医疗保健场景中的有效性、鲁棒性和通用性。 

---
# Drift-aware Collaborative Assistance Mixture of Experts for Heterogeneous Multistream Learning 

**Title (ZH)**: 具有漂移意识的协作辅助专家混合模型用于异构多流学习 

**Authors**: En Yu, Jie Lu, Kun Wang, Xiaoyu Yang, Guangquan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01598)  

**Abstract**: Learning from multiple data streams in real-world scenarios is fundamentally challenging due to intrinsic heterogeneity and unpredictable concept drifts. Existing methods typically assume homogeneous streams and employ static architectures with indiscriminate knowledge fusion, limiting generalizability in complex dynamic environments. To tackle this gap, we propose CAMEL, a dynamic \textbf{C}ollaborative \textbf{A}ssistance \textbf{M}ixture of \textbf{E}xperts \textbf{L}earning framework. It addresses heterogeneity by assigning each stream an independent system with a dedicated feature extractor and task-specific head. Meanwhile, a dynamic pool of specialized private experts captures stream-specific idiosyncratic patterns. Crucially, collaboration across these heterogeneous streams is enabled by a dedicated assistance expert. This expert employs a multi-head attention mechanism to distill and integrate relevant context autonomously from all other concurrent streams. It facilitates targeted knowledge transfer while inherently mitigating negative transfer from irrelevant sources. Furthermore, we propose an Autonomous Expert Tuner (AET) strategy, which dynamically manages expert lifecycles in response to drift. It instantiates new experts for emerging concepts (freezing prior ones to prevent catastrophic forgetting) and prunes obsolete ones. This expert-level plasticity provides a robust and efficient mechanism for online model capacity adaptation. Extensive experiments demonstrate CAMEL's superior generalizability across diverse multistreams and exceptional resilience against complex concept drifts. 

**Abstract (ZH)**: 多数据流在现实场景中的学习本质上极具挑战性，由于固有的异构性和不可预测的概念漂移。现有方法通常假设同质流，并采用静态架构和不分青红皂白的知识融合，限制了在复杂动态环境中的泛化能力。为弥补这一差距，我们提出了一种动态的协作专家混合学习框架CAMEL。该框架通过为每一流分配独立的系统，包含专用特征提取器和任务特定的头部，来应对异构性。同时，一个动态的特殊私有专家池捕捉流特定的异质模式。尤为重要的是，通过专用的协助专家，在这些异构流之间实现协作。该专家利用多头注意力机制自主从所有其他并行流中提炼和整合相关上下文，促进有针对性的知识转移，同时固有地减少来自无关源的负面影响。此外，我们提出了自主专家调谐器（AET）策略，该策略根据漂移动态管理专家生命周期。它为新兴概念实例化新专家（并冻结先前的专家以防灾难性遗忘），并淘汰过时的专家。这种专家级别的可塑性为在线模型容量适应提供了一个稳健且高效的机制。广泛实验表明，CAMEL在多种多流场景下的泛化能力和对抗复杂概念漂移的出色鲁棒性优于现有方法。 

---
# Censored Sampling for Topology Design: Guiding Diffusion with Human Preferences 

**Title (ZH)**: 基于裁剪采样的拓扑设计：以人类偏好引导扩散 

**Authors**: Euihyun Kim, Keun Park, Yeoneung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.01589)  

**Abstract**: Recent advances in denoising diffusion models have enabled rapid generation of optimized structures for topology optimization. However, these models often rely on surrogate predictors to enforce physical constraints, which may fail to capture subtle yet critical design flaws such as floating components or boundary discontinuities that are obvious to human experts. In this work, we propose a novel human-in-the-loop diffusion framework that steers the generative process using a lightweight reward model trained on minimal human feedback. Inspired by preference alignment techniques in generative modeling, our method learns to suppress unrealistic outputs by modulating the reverse diffusion trajectory using gradients of human-aligned rewards. Specifically, we collect binary human evaluations of generated topologies and train classifiers to detect floating material and boundary violations. These reward models are then integrated into the sampling loop of a pre-trained diffusion generator, guiding it to produce designs that are not only structurally performant but also physically plausible and manufacturable. Our approach is modular and requires no retraining of the diffusion model. Preliminary results show substantial reductions in failure modes and improved design realism across diverse test conditions. This work bridges the gap between automated design generation and expert judgment, offering a scalable solution to trustworthy generative design. 

**Abstract (ZH)**: Recent Advances in Denoising Diffusion Models for Topology Optimization: A Human-in-the-Loop Framework for Enhanced Design Realism and Manufacturability 

---
# Diffusion Models for Future Networks and Communications: A Comprehensive Survey 

**Title (ZH)**: 未来网络与通信中的扩散模型：一项全面综述 

**Authors**: Nguyen Cong Luong, Nguyen Duc Hai, Duc Van Le, Huy T. Nguyen, Thai-Hoc Vu, Thien Huynh-The, Ruichen Zhang, Nguyen Duc Duy Anh, Dusit Niyato, Marco Di Renzo, Dong In Kim, Quoc-Viet Pham  

**Link**: [PDF](https://arxiv.org/pdf/2508.01586)  

**Abstract**: The rise of Generative AI (GenAI) in recent years has catalyzed transformative advances in wireless communications and networks. Among the members of the GenAI family, Diffusion Models (DMs) have risen to prominence as a powerful option, capable of handling complex, high-dimensional data distribution, as well as consistent, noise-robust performance. In this survey, we aim to provide a comprehensive overview of the theoretical foundations and practical applications of DMs across future communication systems. We first provide an extensive tutorial of DMs and demonstrate how they can be applied to enhance optimizers, reinforcement learning and incentive mechanisms, which are popular approaches for problems in wireless networks. Then, we review and discuss the DM-based methods proposed for emerging issues in future networks and communications, including channel modeling and estimation, signal detection and data reconstruction, integrated sensing and communication, resource management in edge computing networks, semantic communications and other notable issues. We conclude the survey with highlighting technical limitations of DMs and their applications, as well as discussing future research directions. 

**Abstract (ZH)**: 近年来生成式AI（GenAI）的发展推动了无线通信和网络的转型性进步。在生成式AI家族中，扩散模型（DMs）因其能够处理复杂高维数据分布以及一致的抗噪性能而崭露头角。本文综述旨在提供扩散模型在future通信系统中的理论基础及其实际应用的全面概述。我们首先提供了扩散模型的深入教程，并展示了它们如何被用于优化器、强化学习和激励机制的增强，这些是无线网络中流行的方法。然后，我们回顾并讨论了用于解决未来网络和通信中新兴问题的基于扩散模型的方法，包括信道建模与估计、信号检测与数据重构、综合传感与通信、边缘计算网络中的资源管理、语义通信及其他重大问题。最后，我们强调了扩散模型及其应用的技术限制，并讨论了未来的研究方向。 

---
# Leveraging Machine Learning for Botnet Attack Detection in Edge-Computing Assisted IoT Networks 

**Title (ZH)**: 利用机器学习在辅助边缘计算的物联网网络中检测僵尸网络攻击 

**Authors**: Dulana Rupanetti, Naima Kaabouch  

**Link**: [PDF](https://arxiv.org/pdf/2508.01542)  

**Abstract**: The increase of IoT devices, driven by advancements in hardware technologies, has led to widespread deployment in large-scale networks that process massive amounts of data daily. However, the reliance on Edge Computing to manage these devices has introduced significant security vulnerabilities, as attackers can compromise entire networks by targeting a single IoT device. In light of escalating cybersecurity threats, particularly botnet attacks, this paper investigates the application of machine learning techniques to enhance security in Edge-Computing-Assisted IoT environments. Specifically, it presents a comparative analysis of Random Forest, XGBoost, and LightGBM -- three advanced ensemble learning algorithms -- to address the dynamic and complex nature of botnet threats. Utilizing a widely recognized IoT network traffic dataset comprising benign and malicious instances, the models were trained, tested, and evaluated for their accuracy in detecting and classifying botnet activities. Furthermore, the study explores the feasibility of deploying these models in resource-constrained edge and IoT devices, demonstrating their practical applicability in real-world scenarios. The results highlight the potential of machine learning to fortify IoT networks against emerging cybersecurity challenges. 

**Abstract (ZH)**: 物联网设备数量的不断增加，得益于硬件技术的进步，已在大规模网络中得到广泛部署，这些网络每天处理大量数据。然而，对边缘计算的依赖性管理这些设备已引入了显著的安全漏洞，攻击者可以通过攻击单一物联网设备来控制整个网络。鉴于网络安全威胁的升级，尤其是僵尸网络攻击，本文研究了机器学习技术在边缘计算辅助物联网环境中的应用，以增强安全性。具体而言，本文对随机森林、XGBoost和LightGBM三种先进的集成学习算法进行了比较分析，以应对僵尸网络威胁的动态和复杂性。利用一个包含正常和恶意实例的广泛认可的物联网网络流量数据集，对模型进行了训练、测试和评估，以检测和分类僵尸网络活动。此外，研究还探讨了在资源受限的边缘和物联网设备中部署这些模型的可行性，展示了其在实际场景中的实用性。研究结果突显了机器学习在增强物联网网络抵御新兴网络安全挑战方面的能力。 

---
# Revisiting Gossip Protocols: A Vision for Emergent Coordination in Agentic Multi-Agent Systems 

**Title (ZH)**: 重新审视闲谈协议：自主多agent系统中 emergent 协调的愿景 

**Authors**: Mansura Habiba, Nafiul I. Khan  

**Link**: [PDF](https://arxiv.org/pdf/2508.01531)  

**Abstract**: As agentic platforms scale, agents are evolving beyond static roles and fixed toolchains, creating a growing need for flexible, decentralized coordination. Today's structured communication protocols (e.g., direct agent-to-agent messaging) excel at reliability and task delegation, but they fall short in enabling emergent, swarm-like intelligence, where distributed agents continuously learn, adapt, and communicate to form collective cognition. This paper revisits gossip protocols, long valued in distributed systems for their fault tolerance and decentralization, and argues that they offer a missing layer for context-rich, adaptive communication in agentic AI. Gossip enables scalable, low-overhead dissemination of shared knowledge, but also raises unresolved challenges around semantic filtering, staleness, trustworthiness, and consistency in high-stakes environments. Rather than proposing a new framework, this work charts a research agenda for integrating gossip as a complementary substrate alongside structured protocols. We identify critical gaps in current agent-to-agent architectures, highlight where gossip could reshape assumptions about coordination, and outline open questions around intent propagation, knowledge decay, and peer-to-peer trust. Gossip is not a silver bullet, but overlooking it risks missing a key path toward resilient, reflexive, and self-organizing multi-agent systems. 

**Abstract (ZH)**: 随着代理平台的扩展，代理正在超越静态角色和固定工具链，创建对灵活且去中心化协调日益增长的需求。今天的结构化通信协议（例如，直接代理间通信）在可靠性和任务委派方面表现出色，但在促进分布式代理的连续学习、适应和交流，从而形成集体认知的涌现式智能方面仍显不足。本文重访在分布式系统中因其容错性和去中心化而长期受到重视的流言协议，并argue指出它们为富有语境的、适应性通信提供了缺失的一层。流言协议使共享知识的大规模、低开销传播成为可能，但也提出了在高危环境中围绕语义过滤、陈旧、可信性和一致性的未解决挑战。本文未提出新的框架，而是为将流言协议作为一种补充基础结构与结构化协议整合的研究议程制定了路线图。我们指出了当前代理间架构中的关键缺口，强调了流言协议如何重新塑造关于协调的假设，并概述了意图传播、知识衰退和点对点信任等方面存在的开放问题。流言协议并非灵丹妙药，但忽略它可能会错失通向稳健、反射性和自主多代理系统的关键路径。 

---
# The Vanishing Gradient Problem for Stiff Neural Differential Equations 

**Title (ZH)**: 刚性神经微分方程中的消失梯度问题 

**Authors**: Colby Fronk, Linda Petzold  

**Link**: [PDF](https://arxiv.org/pdf/2508.01519)  

**Abstract**: Gradient-based optimization of neural differential equations and other parameterized dynamical systems fundamentally relies on the ability to differentiate numerical solutions with respect to model parameters. In stiff systems, it has been observed that sensitivities to parameters controlling fast-decaying modes become vanishingly small during training, leading to optimization difficulties. In this paper, we show that this vanishing gradient phenomenon is not an artifact of any particular method, but a universal feature of all A-stable and L-stable stiff numerical integration schemes. We analyze the rational stability function for general stiff integration schemes and demonstrate that the relevant parameter sensitivities, governed by the derivative of the stability function, decay to zero for large stiffness. Explicit formulas for common stiff integration schemes are provided, which illustrate the mechanism in detail. Finally, we rigorously prove that the slowest possible rate of decay for the derivative of the stability function is $O(|z|^{-1})$, revealing a fundamental limitation: all A-stable time-stepping methods inevitably suppress parameter gradients in stiff regimes, posing a significant barrier for training and parameter identification in stiff neural ODEs. 

**Abstract (ZH)**: 基于梯度的神经微分方程和其他参数化动力系统的优化从根本上依赖于能够对模型参数求解数值解的灵敏度。在刚性系统中，已观察到控制快速衰减模式的参数的灵敏度在训练过程中变得微不足道，导致优化困难。在本文中，我们证明这种梯度消失现象并非任何特定方法的产物，而是所有A-稳定和L-稳定的刚性数值积分方案的普遍特征。我们分析了一般刚性积分方案的有理稳定性函数，并演示了由稳定性函数的导数确定的相关参数灵敏度在大刚性下衰减为零。提供了常见刚性积分方案的显式公式，详细说明了机制。最后，我们严格证明了稳定性函数导数的最慢衰减率是$O(|z|^{-1})$，揭示了一个基本限制：所有A-稳定的时步方法在刚性状态下不可避免地抑制参数梯度，对刚性神经常微分方程的训练和参数识别构成重大障碍。 

---
# ShrutiSense: Microtonal Modeling and Correction in Indian Classical Music 

**Title (ZH)**: ShrutiSense：印度古典音乐中的微分音建模与修正 

**Authors**: Rajarshi Ghosh, Jayanth Athipatla  

**Link**: [PDF](https://arxiv.org/pdf/2508.01498)  

**Abstract**: Indian classical music relies on a sophisticated microtonal system of 22 shrutis (pitch intervals), which provides expressive nuance beyond the 12-tone equal temperament system. Existing symbolic music processing tools fail to account for these microtonal distinctions and culturally specific raga grammars that govern melodic movement. We present ShrutiSense, a comprehensive symbolic pitch processing system designed for Indian classical music, addressing two critical tasks: (1) correcting westernized or corrupted pitch sequences, and (2) completing melodic sequences with missing values. Our approach employs complementary models for different tasks: a Shruti-aware finite-state transducer (FST) that performs contextual corrections within the 22-shruti framework and a grammar-constrained Shruti hidden Markov model (GC-SHMM) that incorporates raga-specific transition rules for contextual completions. Comprehensive evaluation on simulated data across five ragas demonstrates that ShrutiSense (FST model) achieves 91.3% shruti classification accuracy for correction tasks, with example sequences showing 86.7-90.0% accuracy at corruption levels of 0.2 to 0.4. The system exhibits robust performance under pitch noise up to +/-50 cents, maintaining consistent accuracy across ragas (90.7-91.8%), thus preserving the cultural authenticity of Indian classical music expression. 

**Abstract (ZH)**: 印度古典音乐依赖于一个基于22 shrutis（音高间隔）的复杂微音系统，提供了超越12平均_temperament系统的表达细腻之处。现有的符号音乐处理工具未能考虑到这些微音差异以及指导旋律运动的文化特定法则。我们提出了一种专为印度古典音乐设计的全面符号音高处理系统——ShrutiSense，该系统针对两个关键任务：（1）校正西方化或损坏的音高序列，以及（2）补全缺失值的旋律序列。我们的方法使用了不同的互补模型：一种aware finite-state transducer（FS筹资者，利用22-shruti框架内的上下文校正，以及一种语法约束的Shruti隐马尔可夫模型（GC-SHMM），该模型整合了特定于拉格的转换规则以进行上下文补全。在五种拉格模拟数据上的全面评估结果显示，ShrutiSense（FST模型）在纠正任务中的shruti分类准确率达到91.3%，在噪声水平为0.2至0.4的损坏序列中，示例序列的准确性范围为86.7%至90.0%。该系统在高达±50美分的音高噪声下表现出稳健性，并在不同拉格中保持了90.7%至91.8%的一致准确性，从而保持了印度古典音乐表达的文化真实性。 

---
# Translation-Equivariant Self-Supervised Learning for Pitch Estimation with Optimal Transport 

**Title (ZH)**: 基于最优传输的平移不变自监督学习在音高估计中的应用 

**Authors**: Bernardo Torres, Alain Riou, Gaël Richard, Geoffroy Peeters  

**Link**: [PDF](https://arxiv.org/pdf/2508.01493)  

**Abstract**: In this paper, we propose an Optimal Transport objective for learning one-dimensional translation-equivariant systems and demonstrate its applicability to single pitch estimation. Our method provides a theoretically grounded, more numerically stable, and simpler alternative for training state-of-the-art self-supervised pitch estimators. 

**Abstract (ZH)**: 本文提出了一个最优传输目标用于学习一维平移不变系统，并展示了其在单音高估计中的适用性。我们的方法提供了一种理论依据更可靠、数值稳定性更强且更简单的训练当前最先进自主监督音高估计器的替代方案。 

---
# A Large-Scale Benchmark of Cross-Modal Learning for Histology and Gene Expression in Spatial Transcriptomics 

**Title (ZH)**: 大规模跨模态学习在空间转录组学组织学和基因表达基准测试 

**Authors**: Rushin H. Gindra, Giovanni Palla, Mathias Nguyen, Sophia J. Wagner, Manuel Tran, Fabian J Theis, Dieter Saur, Lorin Crawford, Tingying Peng  

**Link**: [PDF](https://arxiv.org/pdf/2508.01490)  

**Abstract**: Spatial transcriptomics enables simultaneous measurement of gene expression and tissue morphology, offering unprecedented insights into cellular organization and disease mechanisms. However, the field lacks comprehensive benchmarks for evaluating multimodal learning methods that leverage both histology images and gene expression data. Here, we present HESCAPE, a large-scale benchmark for cross-modal contrastive pretraining in spatial transcriptomics, built on a curated pan-organ dataset spanning 6 different gene panels and 54 donors. We systematically evaluated state-of-the-art image and gene expression encoders across multiple pretraining strategies and assessed their effectiveness on two downstream tasks: gene mutation classification and gene expression prediction. Our benchmark demonstrates that gene expression encoders are the primary determinant of strong representational alignment, and that gene models pretrained on spatial transcriptomics data outperform both those trained without spatial data and simple baseline approaches. However, downstream task evaluation reveals a striking contradiction: while contrastive pretraining consistently improves gene mutation classification performance, it degrades direct gene expression prediction compared to baseline encoders trained without cross-modal objectives. We identify batch effects as a key factor that interferes with effective cross-modal alignment. Our findings highlight the critical need for batch-robust multimodal learning approaches in spatial transcriptomics. To accelerate progress in this direction, we release HESCAPE, providing standardized datasets, evaluation protocols, and benchmarking tools for the community 

**Abstract (ZH)**: 跨模态对比预训练在空间转录组学中的大规模基准：HESCAPE 

---
# PESTO: Real-Time Pitch Estimation with Self-supervised Transposition-equivariant Objective 

**Title (ZH)**: PESTO: 自监督移调不变目标的实时音高估计 

**Authors**: Alain Riou, Bernardo Torres, Ben Hayes, Stefan Lattner, Gaëtan Hadjeres, Gaël Richard, Geoffroy Peeters  

**Link**: [PDF](https://arxiv.org/pdf/2508.01488)  

**Abstract**: In this paper, we introduce PESTO, a self-supervised learning approach for single-pitch estimation using a Siamese architecture. Our model processes individual frames of a Variable-$Q$ Transform (VQT) and predicts pitch distributions. The neural network is designed to be equivariant to translations, notably thanks to a Toeplitz fully-connected layer. In addition, we construct pitch-shifted pairs by translating and cropping the VQT frames and train our model with a novel class-based transposition-equivariant objective, eliminating the need for annotated data. Thanks to this architecture and training objective, our model achieves remarkable performances while being very lightweight ($130$k parameters). Evaluations on music and speech datasets (MIR-1K, MDB-stem-synth, and PTDB) demonstrate that PESTO not only outperforms self-supervised baselines but also competes with supervised methods, exhibiting superior cross-dataset generalization. Finally, we enhance PESTO's practical utility by developing a streamable VQT implementation using cached convolutions. Combined with our model's low latency (less than 10 ms) and minimal parameter count, this makes PESTO particularly suitable for real-time applications. 

**Abstract (ZH)**: 基于Siamese架构的自监督单音高估计方法PESTO 

---
# Training Dynamics of the Cooldown Stage in Warmup-Stable-Decay Learning Rate Scheduler 

**Title (ZH)**: Warmup-Stable-Decay学习率调度中cooldown阶段的训练动态 

**Authors**: Aleksandr Dremov, Alexander Hägele, Atli Kosson, Martin Jaggi  

**Link**: [PDF](https://arxiv.org/pdf/2508.01483)  

**Abstract**: Learning rate scheduling is essential in transformer training, where the final annealing plays a crucial role in getting the best performance. However, the mechanisms behind this cooldown phase, with its characteristic drop in loss, remain poorly understood. To address this, we provide a comprehensive analysis focusing solely on the cooldown phase in the Warmup-Stable-Decay (WSD) learning rate scheduler. Our analysis reveals that different cooldown shapes reveal a fundamental bias-variance trade-off in the resulting models, with shapes that balance exploration and exploitation consistently outperforming alternatives. Similarly, we find substantial performance variations $\unicode{x2013}$ comparable to those from cooldown shape selection $\unicode{x2013}$ when tuning AdamW hyperparameters. Notably, we observe consistent improvements with higher values of $\beta_2$ during cooldown. From a loss landscape perspective, we provide visualizations of the landscape during cooldown, supporting the river valley loss perspective empirically. These findings offer practical recommendations for configuring the WSD scheduler in transformer training, emphasizing the importance of optimizing the cooldown phase alongside traditional hyperparameter tuning. 

**Abstract (ZH)**: 学习率调度在变压器训练中至关重要，其中最终降温在获得最佳性能中扮演关键角色。然而，这一冷却阶段的背后机制，尤其是其特征性的损失下降，仍知之甚少。为了解决这一问题，我们专注于Warmup-Stable-Decay (WSD) 学习率调度器中的冷却阶段，提供了全面的分析。分析表明，不同的冷却形状揭示了模型中基本的偏差-方差权衡，能够平衡探索与利用的形状始终优于其他替代方案。同样，我们发现性能变化显著，类似于通过调整AdamW 超参数时从冷却形状选择中得到的变化。值得注意的是，我们观察到在冷却阶段使用较高的 $\beta_2$ 值会带来一致性改进。从损失景观的角度来看，我们提供了冷却阶段损失景观的可视化，支持经验上的河流谷地损失观点。这些发现为在变压器训练中配置WSD调度器提供了实用建议，强调优化冷却阶段的重要性不仅限于传统的超参数调优。 

---
# Reconstructing Trust Embeddings from Siamese Trust Scores: A Direct-Sum Approach with Fixed-Point Semantics 

**Title (ZH)**: 从暹罗信任评分重构信任嵌入：固定点语义下的直和方法 

**Authors**: Faruk Alpay, Taylan Alpay, Bugra Kilictas  

**Link**: [PDF](https://arxiv.org/pdf/2508.01479)  

**Abstract**: We study the inverse problem of reconstructing high-dimensional trust embeddings from the one-dimensional Siamese trust scores that many distributed-security frameworks expose. Starting from two independent agents that publish time-stamped similarity scores for the same set of devices, we formalise the estimation task, derive an explicit direct-sum estimator that concatenates paired score series with four moment features, and prove that the resulting reconstruction map admits a unique fixed point under a contraction argument rooted in Banach theory. A suite of synthetic benchmarks (20 devices x 10 time steps) confirms that, even in the presence of Gaussian noise, the recovered embeddings preserve inter-device geometry as measured by Euclidean and cosine metrics; we complement these experiments with non-asymptotic error bounds that link reconstruction accuracy to score-sequence length. Beyond methodology, the paper demonstrates a practical privacy risk: publishing granular trust scores can leak latent behavioural information about both devices and evaluation models. We therefore discuss counter-measures -- score quantisation, calibrated noise, obfuscated embedding spaces -- and situate them within wider debates on transparency versus confidentiality in networked AI systems. All datasets, reproduction scripts and extended proofs accompany the submission so that results can be verified without proprietary code. 

**Abstract (ZH)**: 研究分布式安全框架中一维西梅森信任评分还原高维信任嵌入的逆问题：基于Banach理论的压缩映射证明及其合成基准验证 

---
# Fast and scalable retrosynthetic planning with a transformer neural network and speculative beam search 

**Title (ZH)**: 基于变压器神经网络和推测性束搜索的快速可扩展 retrosynthetic 规划 

**Authors**: Mikhail Andronov, Natalia Andronova, Michael Wand, Jürgen Schmidhuber, Djork-Arné Clevert  

**Link**: [PDF](https://arxiv.org/pdf/2508.01459)  

**Abstract**: AI-based computer-aided synthesis planning (CASP) systems are in demand as components of AI-driven drug discovery workflows. However, the high latency of such CASP systems limits their utility for high-throughput synthesizability screening in de novo drug design. We propose a method for accelerating multi-step synthesis planning systems that rely on SMILES-to-SMILES transformers as single-step retrosynthesis models. Our approach reduces the latency of SMILES-to-SMILES transformers powering multi-step synthesis planning in AiZynthFinder through speculative beam search combined with a scalable drafting strategy called Medusa. Replacing standard beam search with our approach allows the CASP system to solve 26\% to 86\% more molecules under the same time constraints of several seconds. Our method brings AI-based CASP systems closer to meeting the strict latency requirements of high-throughput synthesizability screening and improving general user experience. 

**Abstract (ZH)**: 基于AI的计算机辅助合成规划（CASP）系统在AI驱动的药物发现工作流程中需求旺盛。然而，这类CASP系统的高延迟限制了其在从头药物设计中进行高通量合成可及性筛查的实用性。我们提出了一种加速依赖于SMILES-to-SMILES转换器作为单步逆合成模型的多步合成规划系统的办法。我们的方法通过结合投机性 beam 搜索和一种可扩展的速记策略Medusa，减少了AiZynthFinder中SMILES-to-SMILES转换器驱动的多步合成规划的延迟。将标准beam搜索替换为我们的方法，允许CASP系统在相同几秒时间约束下解决26\%到86\%更多的分子。该方法使基于AI的CASP系统更接近满足高通量合成可及性筛查的严格延迟要求，并提高通用用户体验。 

---
# Capturing More: Learning Multi-Domain Representations for Robust Online Handwriting Verification 

**Title (ZH)**: 捕获更多：学习多域表示以实现稳健的在线手写验证 

**Authors**: Peirong Zhang, Kai Ding, Lianwen Jin  

**Link**: [PDF](https://arxiv.org/pdf/2508.01427)  

**Abstract**: In this paper, we propose SPECTRUM, a temporal-frequency synergistic model that unlocks the untapped potential of multi-domain representation learning for online handwriting verification (OHV). SPECTRUM comprises three core components: (1) a multi-scale interactor that finely combines temporal and frequency features through dual-modal sequence interaction and multi-scale aggregation, (2) a self-gated fusion module that dynamically integrates global temporal and frequency features via self-driven balancing. These two components work synergistically to achieve micro-to-macro spectral-temporal integration. (3) A multi-domain distance-based verifier then utilizes both temporal and frequency representations to improve discrimination between genuine and forged handwriting, surpassing conventional temporal-only approaches. Extensive experiments demonstrate SPECTRUM's superior performance over existing OHV methods, underscoring the effectiveness of temporal-frequency multi-domain learning. Furthermore, we reveal that incorporating multiple handwritten biometrics fundamentally enhances the discriminative power of handwriting representations and facilitates verification. These findings not only validate the efficacy of multi-domain learning in OHV but also pave the way for future research in multi-domain approaches across both feature and biometric domains. Code is publicly available at this https URL. 

**Abstract (ZH)**: 本研究提出SPECTRUM，一种时空协同模型，解锁多域在线手写验证中的潜在能力。SPECTRUM包含三个核心组件：(1) 多尺度交互器通过双模序列交互和多尺度聚合精细结合时空特征，(2) 自控融合模块通过自我驱动的平衡动态整合全局时空特征。这两个组件协同工作以实现从微观到宏观的时空频谱整合。(3) 多域基于距离的验证器利用时空特征提高 Genuine 和 Forgery 手写之间的鉴别能力，超越传统仅时空的方法。大量实验表明 SPECTRUM 在现有在线手写验证方法中的优越性能，突显时空多域学习的有效性。此外，我们揭示，整合多种手写生物特征从根本上增强了手写表示的鉴别力并促进了验证。这些发现不仅验证了多域学习在在线手写验证中的有效性，还为跨特征和生物特征领域的多域方法研究奠定了基础。代码在此处公开。 

---
# MedSynth: Realistic, Synthetic Medical Dialogue-Note Pairs 

**Title (ZH)**: MedSynth: 真实可信的合成医疗对话-笔记对 

**Authors**: Ahmad Rezaie Mianroodi, Amirali Rezaie, Niko Grisel Todorov, Cyril Rakovski, Frank Rudzicz  

**Link**: [PDF](https://arxiv.org/pdf/2508.01401)  

**Abstract**: Physicians spend significant time documenting clinical encounters, a burden that contributes to professional burnout. To address this, robust automation tools for medical documentation are crucial. We introduce MedSynth -- a novel dataset of synthetic medical dialogues and notes designed to advance the Dialogue-to-Note (Dial-2-Note) and Note-to-Dialogue (Note-2-Dial) tasks. Informed by an extensive analysis of disease distributions, this dataset includes over 10,000 dialogue-note pairs covering over 2000 ICD-10 codes. We demonstrate that our dataset markedly enhances the performance of models in generating medical notes from dialogues, and dialogues from medical notes. The dataset provides a valuable resource in a field where open-access, privacy-compliant, and diverse training data are scarce. Code is available at this https URL and the dataset is available at this https URL. 

**Abstract (ZH)**: 医生花费大量时间记录临床 Encounter，这一负担导致了职业倦怠。为了解决这个问题，需要强大的医疗文档自动化工具。我们介绍 MedSynth ——一个新颖的合成医疗对话与笔记数据集，旨在推进对话转笔记（Dial-2-Note）和笔记转对话（Note-2-Dial）任务。该数据集基于广泛的疾病分布分析，包含超过10,000个对话-笔记对，涵盖了超过2000个ICD-10编码。我们证明，该数据集显著提高了模型从对话生成医疗笔记以及从医疗笔记生成对话的性能。该数据集为一个稀缺开放访问、隐私合规和多样化训练数据的领域提供了宝贵的资源。代码可在以下网址获取，并且数据集可在以下网址获取。 

---
# Via Score to Performance: Efficient Human-Controllable Long Song Generation with Bar-Level Symbolic Notation 

**Title (ZH)**: 通过得分到性能：基于小节级符号记谱的人性化高效长乐曲生成 

**Authors**: Tongxi Wang, Yang Yu, Qing Wang, Junlang Qian  

**Link**: [PDF](https://arxiv.org/pdf/2508.01394)  

**Abstract**: Song generation is regarded as the most challenging problem in music AIGC; nonetheless, existing approaches have yet to fully overcome four persistent limitations: controllability, generalizability, perceptual quality, and duration. We argue that these shortcomings stem primarily from the prevailing paradigm of attempting to learn music theory directly from raw audio, a task that remains prohibitively difficult for current models. To address this, we present Bar-level AI Composing Helper (BACH), the first model explicitly designed for song generation through human-editable symbolic scores. BACH introduces a tokenization strategy and a symbolic generative procedure tailored to hierarchical song structure. Consequently, it achieves substantial gains in the efficiency, duration, and perceptual quality of song generation. Experiments demonstrate that BACH, with a small model size, establishes a new SOTA among all publicly reported song generation systems, even surpassing commercial solutions such as Suno. Human evaluations further confirm its superiority across multiple subjective metrics. 

**Abstract (ZH)**: 歌曲生成被认为是音乐AIGC中最具挑战性的问题；然而，现有方法尚未完全克服四大持久难题：可控性、普适性、感知质量以及时长。我们argue这些缺陷主要源于当前范式直接从原始音频中学习音乐理论的任务，这仍然是当前模型无法克服的难题。为此，我们提出了Bar-level AI Composing Helper (BACH)，这是首个专门通过可编辑符号谱进行歌曲生成的模型。BACH引入了一种标记化策略和针对层次化歌曲结构的符号生成流程，从而在歌曲生成的效率、时长和感知质量方面取得了显著提升。实验结果显示，尽管模型规模较小，BACH仍建立了所有已报道歌曲生成系统中的新SOTA，甚至超越了诸如Suno等商业解决方案。进一步的人类评估也证实了其在多个主观指标上的优越性。 

---
# Convergence Analysis of Aggregation-Broadcast in LoRA-enabled Federated Learning 

**Title (ZH)**: LoRA增强联邦学习中聚合-广播的收敛性分析 

**Authors**: Xin Chen, Shuaijun Chen, Omid Tavallaie, Nguyen Tran, Shuhuang Xiang, Albert Zomaya  

**Link**: [PDF](https://arxiv.org/pdf/2508.01348)  

**Abstract**: Federated Learning (FL) enables collaborative model training across decentralized data sources while preserving data privacy. However, the growing size of Machine Learning (ML) models poses communication and computation challenges in FL. Low-Rank Adaptation (LoRA) has recently been introduced into FL as an efficient fine-tuning method, reducing communication overhead by updating only a small number of trainable parameters. Despite its effectiveness, how to aggregate LoRA-updated local models on the server remains a critical and understudied problem. In this paper, we provide a unified convergence analysis for LoRA-based FL. We first categories the current aggregation method into two major type: Sum-Product (SP) and Product-Sum (PS). Then we formally define the Aggregation-Broadcast Operator (ABO) and derive a general convergence condition under mild assumptions. Furthermore, we present several sufficient conditions that guarantee convergence of the global model. These theoretical analyze offer a principled understanding of various aggregation strategies. Notably, we prove that the SP and PS aggregation methods both satisfy our convergence condition, but differ in their ability to achieve the optimal convergence rate. Extensive experiments on standard benchmarks validate our theoretical findings. 

**Abstract (ZH)**: 联邦学习（FL）能够在保护数据隐私的同时，跨分散的数据源进行协同模型训练。然而，机器学习（ML）模型的快速增长给FL带来了通信和计算上的挑战。低秩适应（LoRA） recently has被引入到FL中，作为一种有效的微调方法，通过仅更新少量可训练参数来减少通信开销。尽管LoRA非常有效，但在服务器端如何聚合LoRA更新的本地模型仍然是一个关键且研究不足的问题。在本文中，我们提供了LoRA基联邦学习的统一收敛性分析。我们首先将当前的聚合方法归纳为两大类：求和-积（SP）和积-求和（PS）。然后我们形式化定义了聚合-广播操作符（ABO），并在温和假设下推导出一般的收敛条件。此外，我们提出了若干保证全局模型收敛的充分条件。这些理论分析为各种聚合策略提供了基本原则的理解。值得注意的是，我们证明了SP和PS聚合方法都满足我们的收敛条件，但在实现最优收敛速率方面有所不同。广泛的实验证明了我们理论发现的有效性。 

---
# UEChecker: Detecting Unchecked External Call Vulnerabilities in DApps via Graph Analysis 

**Title (ZH)**: UEChecker: 通过图分析检测DApps中的未检查外部调用漏洞 

**Authors**: Dechao Kong, Xiaoqi Li, Wenkai Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.01343)  

**Abstract**: The increasing number of attacks on the contract layer of DApps has resulted in economic losses amounting to $66 billion. Vulnerabilities arise when contracts interact with external protocols without verifying the results of the calls, leading to exploit entry points such as flash loan attacks and reentrancy attacks. In this paper, we propose UEChecker, a deep learning-based tool that utilizes a call graph and a Graph Convolutional Network to detect unchecked external call vulnerabilities. We design the following components: An edge prediction module that reconstructs the feature representation of nodes and edges in the call graph; A node aggregation module that captures structural information from both the node itself and its neighbors, thereby enhancing feature representation between nodes and improving the model's understanding of the global graph structure; A Conformer Block module that integrates multi-head attention, convolutional modules, and feedforward neural networks to more effectively capture dependencies of different scales within the call graph, extending beyond immediate neighbors and enhancing the performance of vulnerability detection. Finally, we combine these modules with Graph Convolutional Network to detect unchecked external call vulnerabilities. By auditing the smart contracts of 608 DApps, our results show that our tool achieves an accuracy of 87.59% in detecting unchecked external call vulnerabilities. Furthermore, we compare our tool with GAT, LSTM, and GCN baselines, and in the comparison experiments, UEChecker consistently outperforms these models in terms of accuracy. 

**Abstract (ZH)**: DApp合约层攻击不断增加导致经济损失660亿美元，外部调用未验证漏洞引发闪贷攻击和重入攻击等exploit入口。本文提出UEChecker，一种基于深度学习的工具，利用调用图和图卷积网络检测未验证外部调用漏洞。我们设计了以下组件：边预测模块，重建调用图中节点和边的特征表示；节点聚合模块，捕获节点本身及其邻居的信息，增强节点间的特征表示并提升模型对全局图结构的理解；Conformer Block模块，结合多头注意力机制、卷积模块和前馈神经网络，更有效地捕捉调用图中不同尺度的依赖关系，超越直接邻居以提高漏洞检测性能。最后，我们将这些模块与图卷积网络结合，检测未验证的外部调用漏洞。通过审计608个DApp的智能合约，结果显示，我们的工具在检测未验证的外部调用漏洞方面达到了87.59%的准确率。此外，在与GAT、LSTM和GCN基准模型的比较实验中，UEChecker在准确率方面始终优于这些模型。 

---
# CoCoLIT: ControlNet-Conditioned Latent Image Translation for MRI to Amyloid PET Synthesis 

**Title (ZH)**: CoCoLIT: ControlNet条件下的潜空间图像转换用于MRI到淀粉样蛋白PET合成 

**Authors**: Alec Sargood, Lemuel Puglisi, James H. Cole, Neil P. Oxtoby, Daniele Ravì, Daniel C. Alexander  

**Link**: [PDF](https://arxiv.org/pdf/2508.01292)  

**Abstract**: Synthesizing amyloid PET scans from the more widely available and accessible structural MRI modality offers a promising, cost-effective approach for large-scale Alzheimer's Disease (AD) screening. This is motivated by evidence that, while MRI does not directly detect amyloid pathology, it may nonetheless encode information correlated with amyloid deposition that can be uncovered through advanced modeling. However, the high dimensionality and structural complexity of 3D neuroimaging data pose significant challenges for existing MRI-to-PET translation methods. Modeling the cross-modality relationship in a lower-dimensional latent space can simplify the learning task and enable more effective translation. As such, we present CoCoLIT (ControlNet-Conditioned Latent Image Translation), a diffusion-based latent generative framework that incorporates three main innovations: (1) a novel Weighted Image Space Loss (WISL) that improves latent representation learning and synthesis quality; (2) a theoretical and empirical analysis of Latent Average Stabilization (LAS), an existing technique used in similar generative models to enhance inference consistency; and (3) the introduction of ControlNet-based conditioning for MRI-to-PET translation. We evaluate CoCoLIT's performance on publicly available datasets and find that our model significantly outperforms state-of-the-art methods on both image-based and amyloid-related metrics. Notably, in amyloid-positivity classification, CoCoLIT outperforms the second-best method with improvements of +10.5% on the internal dataset and +23.7% on the external dataset. The code and models of our approach are available at this https URL. 

**Abstract (ZH)**: 从更广泛可用的结构性MRI模态合成淀粉样蛋白PET扫描为大规模阿尔茨海默病（AD）筛查提供了一种有前景且成本效益高的方法。 

---
# Exploitation Is All You Need... for Exploration 

**Title (ZH)**: 你需要的只是利用……而不是探索 

**Authors**: Micah Rentschler, Jesse Roberts  

**Link**: [PDF](https://arxiv.org/pdf/2508.01287)  

**Abstract**: Ensuring sufficient exploration is a central challenge when training meta-reinforcement learning (meta-RL) agents to solve novel environments. Conventional solutions to the exploration-exploitation dilemma inject explicit incentives such as randomization, uncertainty bonuses, or intrinsic rewards to encourage exploration. In this work, we hypothesize that an agent trained solely to maximize a greedy (exploitation-only) objective can nonetheless exhibit emergent exploratory behavior, provided three conditions are met: (1) Recurring Environmental Structure, where the environment features repeatable regularities that allow past experience to inform future choices; (2) Agent Memory, enabling the agent to retain and utilize historical interaction data; and (3) Long-Horizon Credit Assignment, where learning propagates returns over a time frame sufficient for the delayed benefits of exploration to inform current decisions. Through experiments in stochastic multi-armed bandits and temporally extended gridworlds, we observe that, when both structure and memory are present, a policy trained on a strictly greedy objective exhibits information-seeking exploratory behavior. We further demonstrate, through controlled ablations, that emergent exploration vanishes if either environmental structure or agent memory is absent (Conditions 1 & 2). Surprisingly, removing long-horizon credit assignment (Condition 3) does not always prevent emergent exploration-a result we attribute to the pseudo-Thompson Sampling effect. These findings suggest that, under the right prerequisites, exploration and exploitation need not be treated as orthogonal objectives but can emerge from a unified reward-maximization process. 

**Abstract (ZH)**: 确保充分探索是训练元强化学习（元-RL）代理解决新型环境时的主要挑战。传统解决方案通过注入显式的激励机制如随机化、不确定性奖励或内在奖励来促进探索。在本研究中，我们假设，只要满足三个条件，仅最大化贪婪（仅探索）目标的代理仍然可以表现出 Emergent 探索行为：（1）反复出现的环境结构，使得过去的经验能够对未来的选择提供指导；（2）代理记忆，使代理能够保留和利用历史交互数据；以及（3）长时延信用分配，使得学习能传播足够长时间范围内的回报，从而让探索的延迟收益能够影响当前决策。通过在随机多臂 bandit 和时间扩展的网格世界中的实验，我们观察到，在结构和记忆同时存在的情况下，严格遵循贪婪目标训练的策略会表现出信息寻求的探索行为。进一步通过受控的退化实验，我们证明，如果缺失环境结构或代理记忆（条件 1 和 2），Emergent 探索行为会消失。令人惊讶的是，移除长时延信用分配（条件 3）并不总是能防止 Emergent 探索行为，我们将这一结果归因于伪-Thompson 抽样效应。这些发现表明，在合适的前提条件下，探索和利用不一定要被视为正交的目标，而是可以从统一的奖励最大化过程中自然涌现。 

---
# Defending Against Beta Poisoning Attacks in Machine Learning Models 

**Title (ZH)**: 在机器学习模型中防御贝塔中毒攻击 

**Authors**: Nilufer Gulciftci, M. Emre Gursoy  

**Link**: [PDF](https://arxiv.org/pdf/2508.01276)  

**Abstract**: Poisoning attacks, in which an attacker adversarially manipulates the training dataset of a machine learning (ML) model, pose a significant threat to ML security. Beta Poisoning is a recently proposed poisoning attack that disrupts model accuracy by making the training dataset linearly nonseparable. In this paper, we propose four defense strategies against Beta Poisoning attacks: kNN Proximity-Based Defense (KPB), Neighborhood Class Comparison (NCC), Clustering-Based Defense (CBD), and Mean Distance Threshold (MDT). The defenses are based on our observations regarding the characteristics of poisoning samples generated by Beta Poisoning, e.g., poisoning samples have close proximity to one another, and they are centered near the mean of the target class. Experimental evaluations using MNIST and CIFAR-10 datasets demonstrate that KPB and MDT can achieve perfect accuracy and F1 scores, while CBD and NCC also provide strong defensive capabilities. Furthermore, by analyzing performance across varying parameters, we offer practical insights regarding defenses' behaviors under varying conditions. 

**Abstract (ZH)**: 针对Beta中毒攻击的防御策略：基于kNN邻近性的防御（KPB）、邻域类比较（NCC）、基于聚类的防御（CBD）和均值距离阈值（MDT） 

---
# Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models 

**Title (ZH)**: 多缓存增强原型学习：视觉-语言模型的测试时泛化优化 

**Authors**: Xinyu Chen, Haotian Zhai, Can Zhang, Xiupeng Shi, Ruirui Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.01225)  

**Abstract**: In zero-shot setting, test-time adaptation adjusts pre-trained models using unlabeled data from the test phase to enhance performance on unknown test distributions. Existing cache-enhanced TTA methods rely on a low-entropy criterion to select samples for prototype construction, assuming intra-class compactness. However, low-entropy samples may be unreliable under distribution shifts, and the resulting prototypes may not ensure compact intra-class distributions. This study identifies a positive correlation between cache-enhanced performance and intra-class compactness. Based on this observation, we propose a Multi-Cache enhanced Prototype-based Test-Time Adaptation (MCP) featuring three caches: an entropy cache for initializing prototype representations with low-entropy samples, an align cache for integrating visual and textual information to achieve compact intra-class distributions, and a negative cache for prediction calibration using high-entropy samples. We further developed MCP++, a framework incorporating cross-modal prototype alignment and residual learning, introducing prototype residual fine-tuning. Comparative and ablation experiments across 15 downstream tasks demonstrate that the proposed method and framework achieve state-of-the-art generalization performance. 

**Abstract (ZH)**: 零样本设置下，测试时适应使用测试阶段的未标注数据调整预训练模型，以增强未知测试分布上的性能。现有的缓存增强测试时适应方法依赖于低熵准则来选择用于原型构建的样本，并假设类内紧凑性。然而，分布转移下低熵样本可能不可靠，由此产生的原型无法确保类内紧凑分布。本研究发现，缓存增强性能与类内紧凑性之间存在正相关。基于这一观察，我们提出了一种名为多缓存增强基于原型的测试时适应（MCP）的方法，包括三个缓存：熵缓存用于使用低熵样本初始化原型表示，对齐缓存用于整合视觉和文本信息以实现类内紧凑分布，以及负缓存用于预测校准，使用高熵样本。我们进一步开发了MCP++框架，融合了跨模态原型对齐和残差学习，引入了原型残差微调。在15个下游任务上的对比和消融实验表明，所提出的方法和框架实现了最先进的泛化性能。 

---
# WebDS: An End-to-End Benchmark for Web-based Data Science 

**Title (ZH)**: WebDS：基于Web的数据科学端到端基准 

**Authors**: Ethan Hsu, Hong Meng Yam, Ines Bouissou, Aaron Murali John, Raj Thota, Josh Koe, Vivek Sarath Putta, G K Dharesan, Alexander Spangher, Shikhar Murty, Tenghao Huang, Christopher D. Manning  

**Link**: [PDF](https://arxiv.org/pdf/2508.01222)  

**Abstract**: A large portion of real-world data science tasks are complex and require multi-hop web-based interactions: finding appropriate data available on the internet, synthesizing real-time data of various modalities from different locations, and producing summarized analyses. Existing web benchmarks often focus on simplistic interactions, such as form submissions or e-commerce transactions, and often do not require diverse tool-using capabilities required for web based data science. Conversely, traditional data science benchmarks typically concentrate on static, often textually bound datasets and do not assess end-to-end workflows that encompass data acquisition, cleaning, analysis, and insight generation. In response, we introduce WebDS, the first end-to-end web-based data science benchmark. It comprises 870 web-based data science tasks across 29 diverse websites from structured government data portals to unstructured news media, challenging agents to perform complex, multi-step operations requiring the use of tools and heterogeneous data formats that better reflect the realities of modern data analytics. Evaluations of current SOTA LLM agents indicate significant performance gaps in accomplishing these tasks. For instance, Browser Use, which accomplishes 80% of tasks on Web Voyager, successfully completes only 15% of tasks in WebDS, which our analysis suggests is due to new failure modes like poor information grounding, repetitive behavior and shortcut-taking that agents performing WebDS' tasks display. By providing a more robust and realistic testing ground, WebDS sets the stage for significant advances in the development of practically useful LLM-based data science. 

**Abstract (ZH)**: WebDS：首个端到端的基于Web的数据科学基准 

---
# Oldie but Goodie: Re-illuminating Label Propagation on Graphs with Partially Observed Features 

**Title (ZH)**: 经典仍优良：利用部分观测特征重新照亮图上的标签传播 

**Authors**: Sukwon Yun, Xin Liu, Yunhak Oh, Junseok Lee, Tianlong Chen, Tsuyoshi Murata, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2508.01209)  

**Abstract**: In real-world graphs, we often encounter missing feature situations where a few or the majority of node features, e.g., sensitive information, are missed. In such scenarios, directly utilizing Graph Neural Networks (GNNs) would yield sub-optimal results in downstream tasks such as node classification. Despite the emergence of a few GNN-based methods attempting to mitigate its missing situation, when only a few features are available, they rather perform worse than traditional structure-based models. To this end, we propose a novel framework that further illuminates the potential of classical Label Propagation (Oldie), taking advantage of Feature Propagation, especially when only a partial feature is available. Now called by GOODIE, it takes a hybrid approach to obtain embeddings from the Label Propagation branch and Feature Propagation branch. To do so, we first design a GNN-based decoder that enables the Label Propagation branch to output hidden embeddings that align with those of the FP branch. Then, GOODIE automatically captures the significance of structure and feature information thanks to the newly designed Structure-Feature Attention. Followed by a novel Pseudo-Label contrastive learning that differentiates the contribution of each positive pair within pseudo-labels originating from the LP branch, GOODIE outputs the final prediction for the unlabeled nodes. Through extensive experiments, we demonstrate that our proposed model, GOODIE, outperforms the existing state-of-the-art methods not only when only a few features are available but also in abundantly available situations. Source code of GOODIE is available at: this https URL. 

**Abstract (ZH)**: 在现实世界的图中，我们经常会遇到节点特征缺失的情况，无论是少量还是多数节点的敏感信息丢失。在这种情况下，直接使用图神经网络（GNN）会在节点分类等下游任务中得到次优结果。尽管出现了一些基于GNN的方法试图解决这个问题，但在只有少量特征可用的情况下，它们的表现反而不如传统的基于结构的模型。为了解决这一问题，我们提出了一种新的框架，进一步突显了经典标签传播（Oldie）的传统潜力，利用特征传播，特别是在只有部分特征可用时。现在称为GOODIE，它采取混合方法从标签传播分支和特征传播分支中获得嵌入。为此，我们首先设计了一个基于GNN的解码器，使标签传播分支能够输出与特征传播分支一致的隐藏嵌入。然后，GOODIE通过新设计的结构-特征注意力自动捕捉结构和特征信息的重要性。接着，通过一种新的伪标签对比学习，区分标签传播分支伪标签中每个正样本对的贡献，GOODIE最终输出未标记节点的预测结果。通过广泛的实验证明，与现有最先进的方法相比，我们的模型GOODIE不仅在少量特征可用的情况下表现更优，在特征丰富的情况下也同样表现出色。GOODIE的源代码可在以下链接获取：this https URL。 

---
# BSL: A Unified and Generalizable Multitask Learning Platform for Virtual Drug Discovery from Design to Synthesis 

**Title (ZH)**: BSL：从设计到合成的虚拟药物发现多任务学习统一平台 

**Authors**: Kun Li, Zhennan Wu, Yida Xiong, Hongzhi Zhang, Longtao Hu, Zhonglie Liu, Junqi Zeng, Wenjie Wu, Mukun Chen, Jiameng Chen, Wenbin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01195)  

**Abstract**: Drug discovery is of great social significance in safeguarding human health, prolonging life, and addressing the challenges of major diseases. In recent years, artificial intelligence has demonstrated remarkable advantages in key tasks across bioinformatics and pharmacology, owing to its efficient data processing and data representation capabilities. However, most existing computational platforms cover only a subset of core tasks, leading to fragmented workflows and low efficiency. In addition, they often lack algorithmic innovation and show poor generalization to out-of-distribution (OOD) data, which greatly hinders the progress of drug discovery. To address these limitations, we propose Baishenglai (BSL), a deep learning-enhanced, open-access platform designed for virtual drug discovery. BSL integrates seven core tasks within a unified and modular framework, incorporating advanced technologies such as generative models and graph neural networks. In addition to achieving state-of-the-art (SOTA) performance on multiple benchmark datasets, the platform emphasizes evaluation mechanisms that focus on generalization to OOD molecular structures. Comparative experiments with existing platforms and baseline methods demonstrate that BSL provides a comprehensive, scalable, and effective solution for virtual drug discovery, offering both algorithmic innovation and high-precision prediction for real-world pharmaceutical research. In addition, BSL demonstrated its practical utility by discovering novel modulators of the GluN1/GluN3A NMDA receptor, successfully identifying three compounds with clear bioactivity in in-vitro electrophysiological assays. These results highlight BSL as a promising and comprehensive platform for accelerating biomedical research and drug discovery. The platform is accessible at this https URL. 

**Abstract (ZH)**: 人工智能增强的开放访问平台Baishenglai：用于虚拟药物发现的七大核心任务统一框架 

---
# Advancing the Foundation Model for Music Understanding 

**Title (ZH)**: 音乐理解基础模型的进展 

**Authors**: Yi Jiang, Wei Wang, Xianwen Guo, Huiyun Liu, Hanrui Wang, Youri Xu, Haoqi Gu, Zhongqian Xie, Chuanjiang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2508.01178)  

**Abstract**: The field of Music Information Retrieval (MIR) is fragmented, with specialized models excelling at isolated tasks. In this work, we challenge this paradigm by introducing a unified foundation model named MuFun for holistic music understanding. Our model features a novel architecture that jointly processes instrumental and lyrical content, and is trained on a large-scale dataset covering diverse tasks such as genre classification, music tagging, and question answering. To facilitate robust evaluation, we also propose a new benchmark for multi-faceted music understanding called MuCUE (Music Comprehensive Understanding Evaluation). Experiments show our model significantly outperforms existing audio large language models across the MuCUE tasks, demonstrating its state-of-the-art effectiveness and generalization ability. 

**Abstract (ZH)**: 音乐信息检索领域的研究支离破碎，专门的模型在单一任务上表现出色。本文通过引入全面音乐理解的统一基础模型MuFun来挑战这一范式。我们的模型具有新颖的架构，可以联合处理乐器和歌词内容，并在涵盖了流派分类、音乐标记和问答等多种任务的大规模数据集上进行训练。为了便于稳健的评估，我们还提出了一种新的多方面音乐理解基准MuCUE（音乐综合理解评估）。实验结果显示，我们的模型在MuCUE任务上显著优于现有音频大型语言模型，证明了其先进的有效性和泛化能力。 

---
# GeHirNet: A Gender-Aware Hierarchical Model for Voice Pathology Classification 

**Title (ZH)**: GeHirNet：一种考虑性别差异的层次模型用于语音病理分类 

**Authors**: Fan Wu, Kaicheng Zhao, Elgar Fleisch, Filipe Barata  

**Link**: [PDF](https://arxiv.org/pdf/2508.01172)  

**Abstract**: AI-based voice analysis shows promise for disease diagnostics, but existing classifiers often fail to accurately identify specific pathologies because of gender-related acoustic variations and the scarcity of data for rare diseases. We propose a novel two-stage framework that first identifies gender-specific pathological patterns using ResNet-50 on Mel spectrograms, then performs gender-conditioned disease classification. We address class imbalance through multi-scale resampling and time warping augmentation. Evaluated on a merged dataset from four public repositories, our two-stage architecture with time warping achieves state-of-the-art performance (97.63\% accuracy, 95.25\% MCC), with a 5\% MCC improvement over single-stage baseline. This work advances voice pathology classification while reducing gender bias through hierarchical modeling of vocal characteristics. 

**Abstract (ZH)**: 基于AI的声音分析在疾病诊断中展现出前景，但现有分类器常常因性别相关的声学变异和罕见疾病数据稀少而无法准确识别特定病理。我们提出一种新颖的两阶段框架，首先使用ResNet-50在梅尔频谱图上识别性别特异性病理模式，然后进行基于性别的疾病分类。我们通过多尺度重采样和时间扭曲增强解决类别不平衡问题。在四个公开数据仓库合并的数据集上评估，我们的两阶段架构结合时间扭曲实现了目前最佳性能（97.63%准确率，95.25%麦考利效能系数），相比单阶段基线提高了5%的麦考利效能系数。该工作通过层级建模声音特征推进了声音病理分类，同时减少了性别偏见。 

---
# Personalized Safety Alignment for Text-to-Image Diffusion Models 

**Title (ZH)**: 个性化安全对齐 для текст-к изображению диффузионных моделей 

**Authors**: Yu Lei, Jinbin Bai, Qingyu Shi, Aosong Feng, Kaidong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01151)  

**Abstract**: Text-to-image diffusion models have revolutionized visual content generation, but current safety mechanisms apply uniform standards that often fail to account for individual user preferences. These models overlook the diverse safety boundaries shaped by factors like age, mental health, and personal beliefs. To address this, we propose Personalized Safety Alignment (PSA), a framework that allows user-specific control over safety behaviors in generative models. PSA integrates personalized user profiles into the diffusion process, adjusting the model's behavior to match individual safety preferences while preserving image quality. We introduce a new dataset, Sage, which captures user-specific safety preferences and incorporates these profiles through a cross-attention mechanism. Experiments show that PSA outperforms existing methods in harmful content suppression and aligns generated content better with user constraints, achieving higher Win Rate and Pass Rate scores. Our code, data, and models are publicly available at this https URL. 

**Abstract (ZH)**: 个性化安全性对齐（PSA）：文本到图像扩散模型的用户特定安全性控制 

---
# Dataset Condensation with Color Compensation 

**Title (ZH)**: 颜色补偿下的数据集凝练 

**Authors**: Huyu Wu, Duo Su, Junjie Hou, Guang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.01139)  

**Abstract**: Dataset condensation always faces a constitutive trade-off: balancing performance and fidelity under extreme compression. Existing methods struggle with two bottlenecks: image-level selection methods (Coreset Selection, Dataset Quantization) suffer from inefficiency condensation, while pixel-level optimization (Dataset Distillation) introduces semantic distortion due to over-parameterization. With empirical observations, we find that a critical problem in dataset condensation is the oversight of color's dual role as an information carrier and a basic semantic representation unit. We argue that improving the colorfulness of condensed images is beneficial for representation learning. Motivated by this, we propose DC3: a Dataset Condensation framework with Color Compensation. After a calibrated selection strategy, DC3 utilizes the latent diffusion model to enhance the color diversity of an image rather than creating a brand-new one. Extensive experiments demonstrate the superior performance and generalization of DC3 that outperforms SOTA methods across multiple benchmarks. To the best of our knowledge, besides focusing on downstream tasks, DC3 is the first research to fine-tune pre-trained diffusion models with condensed datasets. The FID results prove that training networks with our high-quality datasets is feasible without model collapse or other degradation issues. Code and generated data will be released soon. 

**Abstract (ZH)**: 数据集凝练总是面临着一个固有的权衡：在极端压缩条件下平衡性能和保真度。现有方法面临两个瓶颈：图像级选择方法（聚簇选择、数据集量化）由于效率凝练问题而受阻，而像素级优化（数据集蒸馏）由于过度参数化引入了语义失真。通过实证观察，我们发现数据集凝练中的一个关键问题是忽视了颜色作为信息载体和基本语义表示单元的双重角色。我们认为，提高凝练后图像的色彩丰富度有助于表示学习。受此启发，我们提出了DC3：一种具有色彩补偿的数据集凝练框架。在经过校准的选择策略后，DC3利用潜在扩散模型增强图像的色彩多样性，而非创造一个新的图像。广泛的经验表明，DC3在多个基准上表现出优越的性能和泛化能力，且优于当前最先进的方法。据我们所知，除了关注下游任务外，DC3还是第一个研究通过使用凝练后的数据集微调预训练扩散模型的工作。FID结果证明了使用高质量数据集训练网络的可行性，且不会出现模型崩溃或其他退化问题。代码和生成的数据将尽快发布。 

---
# Towards Bridging Review Sparsity in Recommendation with Textual Edge Graph Representation 

**Title (ZH)**: 基于文本边图表示的推荐中应对评价稀疏性的桥梁构建 

**Authors**: Leyao Wang, Xutao Mao, Xuhui Zhan, Yuying Zhao, Bo Ni, Ryan A. Rossi, Nesreen K. Ahmed, Tyler Derr  

**Link**: [PDF](https://arxiv.org/pdf/2508.01128)  

**Abstract**: Textual reviews enrich recommender systems with fine-grained preference signals and enhanced explainability. However, in real-world scenarios, users rarely leave reviews, resulting in severe sparsity that undermines the effectiveness of existing models. A natural solution is to impute or generate missing reviews to enrich the data. However, conventional imputation techniques -- such as matrix completion and LLM-based augmentation -- either lose contextualized semantics by embedding texts into vectors, or overlook structural dependencies among user-item interactions. To address these shortcomings, we propose TWISTER (ToWards Imputation on Sparsity with Textual Edge Graph Representation), a unified framework that imputes missing reviews by jointly modeling semantic and structural signals. Specifically, we represent user-item interactions as a Textual-Edge Graph (TEG), treating reviews as edge attributes. To capture relational context, we construct line-graph views and employ a large language model as a graph-aware aggregator. For each interaction lacking a textual review, our model aggregates the neighborhood's natural-language representations to generate a coherent and personalized review. Experiments on the Amazon and Goodreads datasets show that TWISTER consistently outperforms traditional numeric, graph-based, and LLM baselines, delivering higher-quality imputed reviews and, more importantly, enhanced recommendation performance. In summary, TWISTER generates reviews that are more helpful, authentic, and specific, while smoothing structural signals for improved recommendations. 

**Abstract (ZH)**: 文本评论丰富了推荐系统中的细粒度偏好信号和增强了可解释性。然而，在实际场景中，用户很少留下评论，导致严重的数据稀疏性，损害了现有模型的效果。一个自然的解决方案是通过填充或生成缺失的评论来丰富数据。然而，传统的填充技术——如矩阵完成和基于LLM的增强——要么通过将文本嵌入向量中失去上下文化的语义，要么忽略了用户-物品交互的结构依赖性。为了应对这些不足，我们提出了一种统一框架TWISTER（针对文本边图表示的稀疏填充），该框架通过联合建模语义和结构信号来填充缺失评论。具体而言，我们将用户-物品交互表示为文本边图（TEG），将评论视为边的属性。为了捕捉关系上下文，我们构建了线图视图并采用大型语言模型作为图感知聚合器。对于每个缺失文本评论的交互，我们的模型通过聚合邻域的自然语言表示来生成连贯且个性化的评论。实验结果表明，TWISTER在亚马逊和Goodreads数据集上的表现始终优于传统的数值、图基和LLM基线模型，不仅生成了高质量的填充评论，而且更重要的是提高了推荐性能。总之，TWISTER生成的评论更具有帮助性、真实性且具体性，同时平滑了结构信号以改进推荐。 

---
# TensoMeta-VQC: A Tensor-Train-Guided Meta-Learning Framework for Robust and Scalable Variational Quantum Computing 

**Title (ZH)**: TensoMeta-VQC：一种基于张量 train 的元学习框架，用于稳健且可扩展的变量子计算 

**Authors**: Jun Qi, Chao-Han Yang, Pin-Yu Chen, Min-Hsiu Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2508.01116)  

**Abstract**: Variational Quantum Computing (VQC) faces fundamental barriers in scalability, primarily due to barren plateaus and quantum noise sensitivity. To address these challenges, we introduce TensoMeta-VQC, a novel tensor-train (TT)-guided meta-learning framework designed to improve the robustness and scalability of VQC significantly. Our framework fully delegates the generation of quantum circuit parameters to a classical TT network, effectively decoupling optimization from quantum hardware. This innovative parameterization mitigates gradient vanishing, enhances noise resilience through structured low-rank representations, and facilitates efficient gradient propagation. Based on Neural Tangent Kernel and statistical learning theory, our rigorous theoretical analyses establish strong guarantees on approximation capability, optimization stability, and generalization performance. Extensive empirical results across quantum dot classification, Max-Cut optimization, and molecular quantum simulation tasks demonstrate that TensoMeta-VQC consistently achieves superior performance and robust noise tolerance, establishing it as a principled pathway toward practical and scalable VQC on near-term quantum devices. 

**Abstract (ZH)**: 张量元学习引导的量子电路参数化变分量子computing (TensoMeta-VQC): 一种提高变分量子computing稳健性与可扩展性的新框架 

---
# Protecting Student Mental Health with a Context-Aware Machine Learning Framework for Stress Monitoring 

**Title (ZH)**: 基于上下文感知的机器学习框架对学生压力监测及其心理健康保护 

**Authors**: Md Sultanul Islam Ovi, Jamal Hossain, Md Raihan Alam Rahi, Fatema Akter  

**Link**: [PDF](https://arxiv.org/pdf/2508.01105)  

**Abstract**: Student mental health is an increasing concern in academic institutions, where stress can severely impact well-being and academic performance. Traditional assessment methods rely on subjective surveys and periodic evaluations, offering limited value for timely intervention. This paper introduces a context-aware machine learning framework for classifying student stress using two complementary survey-based datasets covering psychological, academic, environmental, and social factors. The framework follows a six-stage pipeline involving preprocessing, feature selection (SelectKBest, RFECV), dimensionality reduction (PCA), and training with six base classifiers: SVM, Random Forest, Gradient Boosting, XGBoost, AdaBoost, and Bagging. To enhance performance, we implement ensemble strategies, including hard voting, soft voting, weighted voting, and stacking. Our best models achieve 93.09% accuracy with weighted hard voting on the Student Stress Factors dataset and 99.53% with stacking on the Stress and Well-being dataset, surpassing previous benchmarks. These results highlight the potential of context-integrated, data-driven systems for early stress detection and underscore their applicability in real-world academic settings to support student well-being. 

**Abstract (ZH)**: 基于上下文的机器学习框架在互补调查数据集上的学生压力分类：早期检测和实际应用 

---
# Cross-Domain Web Information Extraction at Pinterest 

**Title (ZH)**: 跨域Pinterest网页信息提取 

**Authors**: Michael Farag, Patrick Halina, Andrey Zaytsev, Alekhya Munagala, Imtihan Ahmed, Junhao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01096)  

**Abstract**: The internet offers a massive repository of unstructured information, but it's a significant challenge to convert this into a structured format. At Pinterest, the ability to accurately extract structured product data from e-commerce websites is essential to enhance user experiences and improve content distribution. In this paper, we present Pinterest's system for attribute extraction, which achieves remarkable accuracy and scalability at a manageable cost. Our approach leverages a novel webpage representation that combines structural, visual, and text modalities into a compact form, optimizing it for small model learning. This representation captures each visible HTML node with its text, style and layout information. We show how this allows simple models such as eXtreme Gradient Boosting (XGBoost) to extract attributes more accurately than much more complex Large Language Models (LLMs) such as Generative Pre-trained Transformer (GPT). Our results demonstrate a system that is highly scalable, processing over 1,000 URLs per second, while being 1000 times more cost-effective than the cheapest GPT alternatives. 

**Abstract (ZH)**: 互联网提供了大量未结构化的信息库，但将其转换为结构化格式是一项重大挑战。在Pinterest，从电子商务网站准确提取结构化产品数据的能力对于提升用户体验和改善内容分发至关重要。在本文中，我们介绍了Pinterest的属性提取系统，该系统在可管理的代价下实现了显著的准确性和可扩展性。我们的方法利用了一种新颖的网页表示形式，该表示形式将结构化、视觉和文本模态整合为紧凑的形式，优化了小型模型的学习。这种表示形式捕捉了每个可见的HTML节点及其文本、样式和布局信息。我们展示了这种表示方法如何使简单的模型，如极端梯度增强（XGBoost），能够比复杂的大型语言模型（LLMs），如生成预训练变换器（GPT）提取属性更加准确。我们的结果表明，该系统具有高度的可扩展性，每秒可以处理超过1,000个URL，同时比最便宜的GPT替代方案的成本低1000倍。 

---
# Provably Secure Retrieval-Augmented Generation 

**Title (ZH)**: 可验证安全的检索增强生成 

**Authors**: Pengcheng Zhou, Yinglun Feng, Zhongliang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01084)  

**Abstract**: Although Retrieval-Augmented Generation (RAG) systems have been widely applied, the privacy and security risks they face, such as data leakage and data poisoning, have not been systematically addressed yet. Existing defense strategies primarily rely on heuristic filtering or enhancing retriever robustness, which suffer from limited interpretability, lack of formal security guarantees, and vulnerability to adaptive attacks. To address these challenges, this paper proposes the first provably secure framework for RAG systems(SAG). Our framework employs a pre-storage full-encryption scheme to ensure dual protection of both retrieved content and vector embeddings, guaranteeing that only authorized entities can access the data. Through formal security proofs, we rigorously verify the scheme's confidentiality and integrity under a computational security model. Extensive experiments across multiple benchmark datasets demonstrate that our framework effectively resists a range of state-of-the-art attacks. This work establishes a theoretical foundation and practical paradigm for verifiably secure RAG systems, advancing AI-powered services toward formally guaranteed security. 

**Abstract (ZH)**: 虽然检索增强生成（RAG）系统已被广泛应用，但它们面临的数据泄露和数据投毒等隐私和安全风险尚未系统性解决。现有防御策略主要依赖启发式过滤或增强检索器的鲁棒性，存在可解释性有限、缺乏正式安全保证以及容易受到适应性攻击的缺点。为应对这些挑战，本文提出了首个可验证安全的RAG系统框架（SAG）。该框架采用预存储全加密方案，确保检索内容和向量嵌入的双重保护，保证只有授权实体可以访问数据。通过形式化安全证明，我们在计算安全模型下严格验证了该方案的机密性和完整性。在多个基准数据集上进行的广泛实验表明，该框架有效抵御了一系列最先进的攻击。本研究为可验证安全的RAG系统奠定了理论基础和实践范式，推动了基于人工智能的服务向正式保证的安全迈进。 

---
# The Lattice Geometry of Neural Network Quantization -- A Short Equivalence Proof of GPTQ and Babai's algorithm 

**Title (ZH)**: 神经网络量化晶格几何——GPTQ与Babai算法的简短等价证明 

**Authors**: Johann Birnick  

**Link**: [PDF](https://arxiv.org/pdf/2508.01077)  

**Abstract**: We explain how data-driven quantization of a linear unit in a neural network corresponds to solving the closest vector problem for a certain lattice generated by input data. We prove that the GPTQ algorithm is equivalent to Babai's well-known nearest-plane algorithm. We furthermore provide geometric intuition for both algorithms. Lastly, we note the consequences of these results, in particular hinting at the possibility for using lattice basis reduction for better quantization. 

**Abstract (ZH)**: 我们解释了神经网络中基于数据的线性单元量化如何对应于某些由输入数据生成的格中最近向量问题的求解。我们证明GPTQ算法等价于Babai著名的最近平面算法。此外，我们为两种算法提供了几何直观。最后，我们指出这些结果的后果， particularly 暗示使用格基减少方法可能实现更好的量化。 

---
# Expressive Power of Graph Transformers via Logic 

**Title (ZH)**: 图变换器的逻辑表达能力 

**Authors**: Veeti Ahvonen, Maurice Funk, Damian Heiman, Antti Kuusisto, Carsten Lutz  

**Link**: [PDF](https://arxiv.org/pdf/2508.01067)  

**Abstract**: Transformers are the basis of modern large language models, but relatively little is known about their precise expressive power on graphs. We study the expressive power of graph transformers (GTs) by Dwivedi and Bresson (2020) and GPS-networks by Rampásek et al. (2022), both under soft-attention and average hard-attention. Our study covers two scenarios: the theoretical setting with real numbers and the more practical case with floats. With reals, we show that in restriction to vertex properties definable in first-order logic (FO), GPS-networks have the same expressive power as graded modal logic (GML) with the global modality. With floats, GPS-networks turn out to be equally expressive as GML with the counting global modality. The latter result is absolute, not restricting to properties definable in a background logic. We also obtain similar characterizations for GTs in terms of propositional logic with the global modality (for reals) and the counting global modality (for floats). 

**Abstract (ZH)**: 图变换器和GPS网络在图上的精确表征能力研究：基于软注意和平均硬注意 

---
# Connectivity Management in Satellite-Aided Vehicular Networks with Multi-Head Attention-Based State Estimation 

**Title (ZH)**: 基于多头注意力机制状态估计的卫星辅助 vehicular 网络连通性管理 

**Authors**: Ibrahim Althamary, Chen-Fu Chou, Chih-Wei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01060)  

**Abstract**: Managing connectivity in integrated satellite-terrestrial vehicular networks is critical for 6G, yet is challenged by dynamic conditions and partial observability. This letter introduces the Multi-Agent Actor-Critic with Satellite-Aided Multi-head self-attention (MAAC-SAM), a novel multi-agent reinforcement learning framework that enables vehicles to autonomously manage connectivity across Vehicle-to-Satellite (V2S), Vehicle-to-Infrastructure (V2I), and Vehicle-to-Vehicle (V2V) links. Our key innovation is the integration of a multi-head attention mechanism, which allows for robust state estimation even with fluctuating and limited information sharing among vehicles. The framework further leverages self-imitation learning (SIL) and fingerprinting to improve learning efficiency and real-time decisions. Simulation results, based on realistic SUMO traffic models and 3GPP-compliant configurations, demonstrate that MAAC-SAM outperforms state-of-the-art terrestrial and satellite-assisted baselines by up to 14% in transmission utility and maintains high estimation accuracy across varying vehicle densities and sharing levels. 

**Abstract (ZH)**: 基于卫星辅助多头自注意力的多agentActor-Critic框架（MAAC-SAM）在综合卫星-地面-vehicular网络中的连通性管理 

---
# A Deep Reinforcement Learning-Based TCP Congestion Control Algorithm: Design, Simulation, and Evaluation 

**Title (ZH)**: 基于深度强化学习的TCP拥塞控制算法：设计、仿真与评估 

**Authors**: Efe Ağlamazlar, Emirhan Eken, Harun Batur Geçici  

**Link**: [PDF](https://arxiv.org/pdf/2508.01047)  

**Abstract**: This paper presents a novel TCP congestion control algorithm based on Deep Reinforcement Learning. The proposed approach utilizes Deep Q-Networks to optimize the congestion window (cWnd) by observing key network parameters and taking real-time actions. The algorithm is trained and evaluated within the NS-3 network simulator using the OpenGym interface. The results demonstrate significant improvements over traditional TCP New Reno in terms of latency and throughput, with better adaptability to changing network conditions. This study emphasizes the potential of reinforcement learning techniques for solving complex congestion control problems in modern networks. 

**Abstract (ZH)**: 基于深度强化学习的新型TCP拥塞控制算法 

---
# AutoSIGHT: Automatic Eye Tracking-based System for Immediate Grading of Human experTise 

**Title (ZH)**: AutoSIGHT：基于自动眼动追踪的即时评估人类专业知识系统 

**Authors**: Byron Dowling, Jozef Probcin, Adam Czajka  

**Link**: [PDF](https://arxiv.org/pdf/2508.01015)  

**Abstract**: Can we teach machines to assess the expertise of humans solving visual tasks automatically based on eye tracking features? This paper proposes AutoSIGHT, Automatic System for Immediate Grading of Human experTise, that classifies expert and non-expert performers, and builds upon an ensemble of features extracted from eye tracking data while the performers were solving a visual task. Results on the task of iris Presentation Attack Detection (PAD) used for this study show that with a small evaluation window of just 5 seconds, AutoSIGHT achieves an average average Area Under the ROC curve performance of 0.751 in subject-disjoint train-test regime, indicating that such detection is viable. Furthermore, when a larger evaluation window of up to 30 seconds is available, the Area Under the ROC curve (AUROC) increases to 0.8306, indicating the model is effectively leveraging more information at a cost of slightly delayed decisions. This work opens new areas of research on how to incorporate the automatic weighing of human and machine expertise into human-AI pairing setups, which need to react dynamically to nonstationary expertise distribution between the human and AI players (e.g. when the experts need to be replaced, or the task at hand changes rapidly). Along with this paper, we offer the eye tracking data used in this study collected from 6 experts and 53 non-experts solving iris PAD visual task. 

**Abstract (ZH)**: 基于眼动特征自动评估人类解决视觉任务专家水平的能力：AutoSIGHT自动即时评分系统 

---
# On Some Tunable Multi-fidelity Bayesian Optimization Frameworks 

**Title (ZH)**: 一些可调多 fidelity 贝叶斯优化框架 

**Authors**: Arjun Manoj, Anastasia S. Georgiou, Dimitris G. Giovanis, Themistoklis P. Sapsis, Ioannis G. Kevrekidis  

**Link**: [PDF](https://arxiv.org/pdf/2508.01013)  

**Abstract**: Multi-fidelity optimization employs surrogate models that integrate information from varying levels of fidelity to guide efficient exploration of complex design spaces while minimizing the reliance on (expensive) high-fidelity objective function evaluations. To advance Gaussian Process (GP)-based multi-fidelity optimization, we implement a proximity-based acquisition strategy that simplifies fidelity selection by eliminating the need for separate acquisition functions at each fidelity level. We also enable multi-fidelity Upper Confidence Bound (UCB) strategies by combining them with multi-fidelity GPs rather than the standard GPs typically used. We benchmark these approaches alongside other multi-fidelity acquisition strategies (including fidelity-weighted approaches) comparing their performance, reliance on high-fidelity evaluations, and hyperparameter tunability in representative optimization tasks. The results highlight the capability of the proximity-based multi-fidelity acquisition function to deliver consistent control over high-fidelity usage while maintaining convergence efficiency. Our illustrative examples include multi-fidelity chemical kinetic models, both homogeneous and heterogeneous (dynamic catalysis for ammonia production). 

**Abstract (ZH)**: 基于邻近度的多保真度获取策略促进了高斯过程（GP）为基础的多保真度优化，通过结合多保真度GP而非标准GP来实现多保真度Upper Confidence Bound（UCB）策略，并在代表性优化任务中与其他多保真度获取策略（包括保真度加权方法）进行了基准测试，比较了它们的性能、对高保真度评估的依赖性和超参数调节性。结果强调了基于邻近度的多保真度获取函数在控制高保真度使用方面的一致能力和保持收敛效率的能力。示例包括多保真度化学动力学模型，包括均相和非均相（氨生产的动态催化）。 

---
# v-PuNNs: van der Put Neural Networks for Transparent Ultrametric Representation Learning 

**Title (ZH)**: v-PuNNs：van der Put神经网络的透明超度量表示学习 

**Authors**: Gnankan Landry Regis N'guessan  

**Link**: [PDF](https://arxiv.org/pdf/2508.01010)  

**Abstract**: Conventional deep learning models embed data in Euclidean space $\mathbb{R}^d$, a poor fit for strictly hierarchical objects such as taxa, word senses, or file systems. We introduce van der Put Neural Networks (v-PuNNs), the first architecture whose neurons are characteristic functions of p-adic balls in $\mathbb{Z}_p$. Under our Transparent Ultrametric Representation Learning (TURL) principle every weight is itself a p-adic number, giving exact subtree semantics. A new Finite Hierarchical Approximation Theorem shows that a depth-K v-PuNN with $\sum_{j=0}^{K-1}p^{\,j}$ neurons universally represents any K-level tree. Because gradients vanish in this discrete space, we propose Valuation-Adaptive Perturbation Optimization (VAPO), with a fast deterministic variant (HiPaN-DS) and a moment-based one (HiPaN / Adam-VAPO). On three canonical benchmarks our CPU-only implementation sets new state-of-the-art: WordNet nouns (52,427 leaves) 99.96% leaf accuracy in 16 min; GO molecular-function 96.9% leaf / 100% root in 50 s; NCBI Mammalia Spearman $\rho = -0.96$ with true taxonomic distance. The learned metric is perfectly ultrametric (zero triangle violations), and its fractal and information-theoretic properties are analyzed. Beyond classification we derive structural invariants for quantum systems (HiPaQ) and controllable generative codes for tabular data (Tab-HiPaN). v-PuNNs therefore bridge number theory and deep learning, offering exact, interpretable, and efficient models for hierarchical data. 

**Abstract (ZH)**: 范德普尔神经网络：p-进球函数神经网络及其在严格层次结构数据表示中的应用 

---
# Generative AI Adoption in Postsecondary Education, AI Hype, and ChatGPT's Launch 

**Title (ZH)**: _generative AI在高等教育中的应用、AI hype及其ChatGPT的发布_ 

**Authors**: Isabel Pedersen  

**Link**: [PDF](https://arxiv.org/pdf/2508.01003)  

**Abstract**: The rapid integration of generative artificial intelligence (AI) into postsecondary education and many other sectors resulted in a global reckoning with this new technology. This paper contributes to the study of the multifaceted influence of generative AI, with a particular focus on OpenAI's ChatGPT within academic settings during the first six months after the release in three specific ways. First, it scrutinizes the rise of ChatGPT as a transformative event construed through a study of mainstream discourses exhibiting AI hype. Second, it discusses the perceived implications of generative AI for writing, teaching, and learning through the lens of critical discourse analysis and critical AI studies. Third, it encourages the necessity for best practices in the adoption of generative AI technologies in education. 

**Abstract (ZH)**: 生成式人工智能在高等教育及其他领域的快速集成引起了全球对这一新技术的反思。本文通过对主流话语中人工智能 hype 的研究，探讨 ChatGPT 在学术环境中的变革性影响，同时从批判话语分析和批判人工智能研究的视角讨论生成式人工智能对写作、教学和学习的潜在影响，并呼吁教育中采用生成式人工智能技术的最佳实践。 

---
# ThermoCycleNet: Stereo-based Thermogram Labeling for Model Transition to Cycling 

**Title (ZH)**: ThermoCycleNet：基于立体视觉的热图标签生成以实现模型过渡到循环使用 

**Authors**: Daniel Andrés López, Vincent Weber, Severin Zentgraf, Barlo Hillen, Perikles Simon, Elmar Schömer  

**Link**: [PDF](https://arxiv.org/pdf/2508.00974)  

**Abstract**: Infrared thermography is emerging as a powerful tool in sports medicine, allowing assessment of thermal radiation during exercise and analysis of anatomical regions of interest, such as the well-exposed calves. Building on our previous advanced automatic annotation method, we aimed to transfer the stereo- and multimodal-based labeling approach from treadmill running to ergometer cycling. Therefore, the training of the semantic segmentation network with automatic labels and fine-tuning on high-quality manually annotated images has been examined and compared in different data set combinations. The results indicate that fine-tuning with a small fraction of manual data is sufficient to improve the overall performance of the deep neural network. Finally, combining automatically generated labels with small manually annotated data sets accelerates the adaptation of deep neural networks to new use cases, such as the transition from treadmill to bicycle. 

**Abstract (ZH)**: 红外热成像技术在体育医学中的应用正逐渐成为一种强大的工具，允许评估运动过程中的热辐射，并分析感兴趣的解剖区域，如暴露良好的小腿。基于我们之前先进的自动注释方法，我们旨在将基于立体和多模态的标注方法从跑台跑步转移到功率自行车骑行中。因此，使用自动标签训练语义分割网络，并在高质量的手动注释图像上进行微调，不同数据集组合下的性能已经进行了比较。结果表明，使用少量的手动数据进行微调足以提高深度神经网络的整体性能。最后，将自动生成的标签与少量手动标注的数据集结合使用，可以加快深度神经网络对新应用场景的适应，如从跑台转变为自行车。 

---
# Generative AI as a Geopolitical Factor in Industry 5.0: Sovereignty, Access, and Control 

**Title (ZH)**: 生成式AI作为 geopolitics 因素在 Industry 5.0 中的作用：主权、访问与控制 

**Authors**: Azmine Toushik Wasi, Enjamamul Haque Eram, Sabrina Afroz Mitu, Md Manjurul Ahsan  

**Link**: [PDF](https://arxiv.org/pdf/2508.00973)  

**Abstract**: Industry 5.0 marks a new phase in industrial evolution, emphasizing human-centricity, sustainability, and resilience through the integration of advanced technologies. Within this evolving landscape, Generative AI (GenAI) and autonomous systems are not only transforming industrial processes but also emerging as pivotal geopolitical instruments. We examine strategic implications of GenAI in Industry 5.0, arguing that these technologies have become national assets central to sovereignty, access, and global influence. As countries compete for AI supremacy, growing disparities in talent, computational infrastructure, and data access are reshaping global power hierarchies and accelerating the fragmentation of the digital economy. The human-centric ethos of Industry 5.0, anchored in collaboration between humans and intelligent systems, increasingly conflicts with the autonomy and opacity of GenAI, raising urgent governance challenges related to meaningful human control, dual-use risks, and accountability. We analyze how these dynamics influence defense strategies, industrial competitiveness, and supply chain resilience, including the geopolitical weaponization of export controls and the rise of data sovereignty. Our contribution synthesizes technological, economic, and ethical perspectives to propose a comprehensive framework for navigating the intersection of GenAI and geopolitics. We call for governance models that balance national autonomy with international coordination while safeguarding human-centric values in an increasingly AI-driven world. 

**Abstract (ZH)**: Industry 5.0标志着工业演化的全新阶段，强调以人为中心、可持续性和韧性，并通过先进技術名称的整合实现。在这一不断演进的背景中，生成式人工智能（GenAI）和自主系统不仅正在重塑工业流程，还逐渐成为关键的地缘政治工具。我们探讨GenAI在Industry 5.0中的战略影响，认为这些技术已成为国家安全、获取和全球影响力的核心资产。随着各国在人工智能领域的竞争加剧，人才、计算基础设施和数据访问方面的差异正在重塑全球权力结构，并加速了数字经济的碎片化。Industry 5.0以人为本的宗旨，根植于人类与智能系统的合作，在与GenAI的自主性和不透明性之间产生了日益激烈的冲突，这引发了与有意义的人类控制、双重用途风险和问责制相关的紧迫治理挑战。我们分析这些动态如何影响国防策略、工业竞争力和供应链韧性，包括出口管制的地缘政治武器化以及数据主权的兴起。我们的贡献综合了技术、经济和伦理视角，提出了一种全面框架，用于应对GenAI和地缘政治交汇处的挑战。我们呼吁建立一种治理模式，能够在保障人类中心价值观的前提下，在日益依赖人工智能的世界中维护国家自主性与国际协调的平衡。 

---
# AI-Educational Development Loop (AI-EDL): A Conceptual Framework to Bridge AI Capabilities with Classical Educational Theories 

**Title (ZH)**: AI教育发展循环（AI-EDL）：一种将AI能力与传统教育理论相结合的概念框架 

**Authors**: Ning Yu, Jie Zhang, Sandeep Mitra, Rebecca Smith, Adam Rich  

**Link**: [PDF](https://arxiv.org/pdf/2508.00970)  

**Abstract**: This study introduces the AI-Educational Development Loop (AI-EDL), a theory-driven framework that integrates classical learning theories with human-in-the-loop artificial intelligence (AI) to support reflective, iterative learning. Implemented in EduAlly, an AI-assisted platform for writing-intensive and feedback-sensitive tasks, the framework emphasizes transparency, self-regulated learning, and pedagogical oversight. A mixed-methods study was piloted at a comprehensive public university to evaluate alignment between AI-generated feedback, instructor evaluations, and student self-assessments; the impact of iterative revision on performance; and student perceptions of AI feedback. Quantitative results demonstrated statistically significant improvement between first and second attempts, with agreement between student self-evaluations and final instructor grades. Qualitative findings indicated students valued immediacy, specificity, and opportunities for growth that AI feedback provided. These findings validate the potential to enhance student learning outcomes through developmentally grounded, ethically aligned, and scalable AI feedback systems. The study concludes with implications for future interdisciplinary applications and refinement of AI-supported educational technologies. 

**Abstract (ZH)**: 基于人类在环的AI教育发展循环（AI-EDL）框架：经典的教育理论与人工智能的整合以支持反思性的迭代学习 

---
# Enhancing material behavior discovery using embedding-oriented Physically-Guided Neural Networks with Internal Variables 

**Title (ZH)**: 基于内部变量的嵌入导向物理引导神经网络材料行为发现增强 

**Authors**: Rubén Muñoz-Sierra, Manuel Doblaré, Jacobo Ayensa-Jiménez  

**Link**: [PDF](https://arxiv.org/pdf/2508.00959)  

**Abstract**: Physically Guided Neural Networks with Internal Variables are SciML tools that use only observable data for training and and have the capacity to unravel internal state relations. They incorporate physical knowledge both by prescribing the model architecture and using loss regularization, thus endowing certain specific neurons with a physical meaning as internal state variables. Despite their potential, these models face challenges in scalability when applied to high-dimensional data such as fine-grid spatial fields or time-evolving systems. In this work, we propose some enhancements to the PGNNIV framework that address these scalability limitations through reduced-order modeling techniques. Specifically, we introduce alternatives to the original decoder structure using spectral decomposition, POD, and pretrained autoencoder-based mappings. These surrogate decoders offer varying trade-offs between computational efficiency, accuracy, noise tolerance, and generalization, while improving drastically the scalability. Additionally, we integrate model reuse via transfer learning and fine-tuning strategies to exploit previously acquired knowledge, supporting efficient adaptation to novel materials or configurations, and significantly reducing training time while maintaining or improving model performance. To illustrate these various techniques, we use a representative case governed by the nonlinear diffusion equation, using only observable data. Results demonstrate that the enhanced PGNNIV framework successfully identifies the underlying constitutive state equations while maintaining high predictive accuracy. It also improves robustness to noise, mitigates overfitting, and reduces computational demands. The proposed techniques can be tailored to various scenarios depending on data availability, resources, and specific modeling objectives, overcoming scalability challenges in all the scenarios. 

**Abstract (ZH)**: 物理指导的神经网络结合内部变量：SciML工具，仅使用可观测数据进行训练，并具备解析内部状态关系的能力。通过减缩阶建模技术解决高维数据下的可扩展性限制，提出改进的PGNNIV框架。 

---
# Learning Unified User Quantized Tokenizers for User Representation 

**Title (ZH)**: 统一用户量化词元化器学习用户表示 

**Authors**: Chuan He, Yang Chen, Wuliang Huang, Tianyi Zheng, Jianhu Chen, Bin Dou, Yice Luo, Yun Zhu, Baokun Wang, Yongchao Liu, Xing Fu, Yu Cheng, Chuntao Hong, Weiqiang Wang, Xin-Wei Yao  

**Link**: [PDF](https://arxiv.org/pdf/2508.00956)  

**Abstract**: Multi-source user representation learning plays a critical role in enabling personalized services on web platforms (e.g., Alipay). While prior works have adopted late-fusion strategies to combine heterogeneous data sources, they suffer from three key limitations: lack of unified representation frameworks, scalability and storage issues in data compression, and inflexible cross-task generalization. To address these challenges, we propose U^2QT (Unified User Quantized Tokenizers), a novel framework that integrates cross-domain knowledge transfer with early fusion of heterogeneous domains. Our framework employs a two-stage architecture: first, a causal Q-Former projects domain-specific features into a shared causal representation space to preserve inter-modality dependencies; second, a multi-view RQ-VAE discretizes causal embeddings into compact tokens through shared and source-specific codebooks, enabling efficient storage while maintaining semantic coherence. Experimental results showcase U^2QT's advantages across diverse downstream tasks, outperforming task-specific baselines in future behavior prediction and recommendation tasks while achieving efficiency gains in storage and computation. The unified tokenization framework enables seamless integration with language models and supports industrial-scale applications. 

**Abstract (ZH)**: 多源用户表示学习在web平台（如支付宝）上实现个性化服务中扮演着关键角色。尽管先前的工作采用晚融合策略结合异构数据源，但它们面临三个关键限制：缺乏统一表示框架、数据压缩中的可扩展性和存储问题，以及跨任务的灵活性差。为了解决这些挑战，我们提出了一种名为U^2QT（统一用户量化分词器）的新框架，该框架将跨域知识转移与异构领域早期融合相结合。该框架采用两阶段架构：首先，因果Q-Former将领域特定特征投影到共享因果表示空间，以保留跨模态依赖性；其次，多视图RQ-VAE通过共享和来源特定的代码本将因果嵌入离散化为紧凑的令牌，从而实现高效的存储同时保持语义连贯性。实验结果展示出U^2QT在多种下游任务中的优势，在未来行为预测和推荐任务中优于任务特定的基线，并实现在存储和计算上的效率提升。统一的令牌化框架能够无缝集成到语言模型中，并支持工业规模的应用。 

---
# Trusted Routing for Blockchain-Empowered UAV Networks via Multi-Agent Deep Reinforcement Learning 

**Title (ZH)**: 基于多代理深度强化学习的区块链赋能无人机网络可信路由 

**Authors**: Ziye Jia, Sijie He, Qiuming Zhu, Wei Wang, Qihui Wu, Zhu Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.00938)  

**Abstract**: Due to the high flexibility and versatility, unmanned aerial vehicles (UAVs) are leveraged in various fields including surveillance and disaster this http URL, in UAV networks, routing is vulnerable to malicious damage due to distributed topologies and high dynamics. Hence, ensuring the routing security of UAV networks is challenging. In this paper, we characterize the routing process in a time-varying UAV network with malicious nodes. Specifically, we formulate the routing problem to minimize the total delay, which is an integer linear programming and intractable to solve. Then, to tackle the network security issue, a blockchain-based trust management mechanism (BTMM) is designed to dynamically evaluate trust values and identify low-trust UAVs. To improve traditional practical Byzantine fault tolerance algorithms in the blockchain, we propose a consensus UAV update mechanism. Besides, considering the local observability, the routing problem is reformulated into a decentralized partially observable Markov decision process. Further, a multi-agent double deep Q-network based routing algorithm is designed to minimize the total delay. Finally, simulations are conducted with attacked UAVs and numerical results show that the delay of the proposed mechanism decreases by 13.39$\%$, 12.74$\%$, and 16.6$\%$ than multi-agent proximal policy optimal algorithms, multi-agent deep Q-network algorithms, and methods without BTMM, respectively. 

**Abstract (ZH)**: 基于区块链的信任管理机制及其在受攻击的时变无人机网络中路由算法的研究 

---
# Maximize margins for robust splicing detection 

**Title (ZH)**: 最大化边距以实现稳健的剪接检测 

**Authors**: Julien Simon de Kergunic, Rony Abecidan, Patrick Bas, Vincent Itier  

**Link**: [PDF](https://arxiv.org/pdf/2508.00897)  

**Abstract**: Despite recent progress in splicing detection, deep learning-based forensic tools remain difficult to deploy in practice due to their high sensitivity to training conditions. Even mild post-processing applied to evaluation images can significantly degrade detector performance, raising concerns about their reliability in operational contexts. In this work, we show that the same deep architecture can react very differently to unseen post-processing depending on the learned weights, despite achieving similar accuracy on in-distribution test data. This variability stems from differences in the latent spaces induced by training, which affect how samples are separated internally. Our experiments reveal a strong correlation between the distribution of latent margins and a detector's ability to generalize to post-processed images. Based on this observation, we propose a practical strategy for building more robust detectors: train several variants of the same model under different conditions, and select the one that maximizes latent margins. 

**Abstract (ZH)**: 尽管在剪接检测方面取得了最近的进步，基于深度学习的法医工具由于对训练条件的高度敏感性，在实际部署中仍然面临挑战。即使对评估图像进行轻度后处理也可能显著降低检测器性能，这引起了对其在实际操作环境中可靠性的担忧。在本文中，我们证明了在实现类似准确度的情况下，相同的深度架构在面对未见过的后处理时会根据所学习的权重表现出截然不同的反应。这种变化性源自于训练过程中诱导的潜在空间差异，这些差异影响样本内部的分离方式。我们的实验揭示了潜在边界分布与检测器对后处理图像的泛化能力之间存在密切关系。基于这一观察，我们提出了构建更为稳健的检测器的实用策略：在不同的条件下训练同一模型的多种变体，并选择能使潜在边界最大化的一个。 

---
# HoneyImage: Verifiable, Harmless, and Stealthy Dataset Ownership Verification for Image Models 

**Title (ZH)**: 蜂蜜图像：可验证、无害且隐蔽的图像模型数据集所有权验证 

**Authors**: Zhihao Zhu, Jiale Han, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00892)  

**Abstract**: Image-based AI models are increasingly deployed across a wide range of domains, including healthcare, security, and consumer applications. However, many image datasets carry sensitive or proprietary content, raising critical concerns about unauthorized data usage. Data owners therefore need reliable mechanisms to verify whether their proprietary data has been misused to train third-party models. Existing solutions, such as backdoor watermarking and membership inference, face inherent trade-offs between verification effectiveness and preservation of data integrity. In this work, we propose HoneyImage, a novel method for dataset ownership verification in image recognition models. HoneyImage selectively modifies a small number of hard samples to embed imperceptible yet verifiable traces, enabling reliable ownership verification while maintaining dataset integrity. Extensive experiments across four benchmark datasets and multiple model architectures show that HoneyImage consistently achieves strong verification accuracy with minimal impact on downstream performance while maintaining imperceptible. The proposed HoneyImage method could provide data owners with a practical mechanism to protect ownership over valuable image datasets, encouraging safe sharing and unlocking the full transformative potential of data-driven AI. 

**Abstract (ZH)**: 基于图像的AI模型已在医疗、安全和消费者应用等多个领域得到广泛应用。然而，许多图像数据集包含敏感或专有内容，引发了未经授权使用数据的严重关切。因此，数据所有者需要可靠的机制来验证其专有数据是否被误用于训练第三方模型。现有解决方案，如后门水印和成员推理，存在验证效果与数据完整性保持之间的固有权衡。在此工作中，我们提出HoneyImage，一种用于图像识别模型数据集所有权验证的新方法。HoneyImage选择性地修改少量难以解决的样本，嵌入不可感知但可验证的痕迹，从而实现可靠的所有权验证，同时保持数据集的完整性。在四个基准数据集和多种模型架构上进行的 extensive 实验表明，HoneyImage 在对下游性能影响最小的情况下，能够实现稳定的验证准确性，并保持不可感知性。所提出的HoneyImage方法可为数据所有者提供保护其有价值图像数据所有权的实际机制，促进安全共享并充分发挥数据驱动AI的变革潜力。 

---
# Accelerating multiparametric quantitative MRI using self-supervised scan-specific implicit neural representation with model reinforcement 

**Title (ZH)**: 利用自监督扫描特异性隐式神经表示与模型强化加速多参数定量磁共振成像 

**Authors**: Ruimin Feng, Albert Jang, Xingxin He, Fang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00891)  

**Abstract**: Purpose: To develop a self-supervised scan-specific deep learning framework for reconstructing accelerated multiparametric quantitative MRI (qMRI).
Methods: We propose REFINE-MORE (REference-Free Implicit NEural representation with MOdel REinforcement), combining an implicit neural representation (INR) architecture with a model reinforcement module that incorporates MR physics constraints. The INR component enables informative learning of spatiotemporal correlations to initialize multiparametric quantitative maps, which are then further refined through an unrolled optimization scheme enforcing data consistency. To improve computational efficiency, REFINE-MORE integrates a low-rank adaptation strategy that promotes rapid model convergence. We evaluated REFINE-MORE on accelerated multiparametric quantitative magnetization transfer imaging for simultaneous estimation of free water spin-lattice relaxation, tissue macromolecular proton fraction, and magnetization exchange rate, using both phantom and in vivo brain data.
Results: Under 4x and 5x accelerations on in vivo data, REFINE-MORE achieved superior reconstruction quality, demonstrating the lowest normalized root-mean-square error and highest structural similarity index compared to baseline methods and other state-of-the-art model-based and deep learning approaches. Phantom experiments further showed strong agreement with reference values, underscoring the robustness and generalizability of the proposed framework. Additionally, the model adaptation strategy improved reconstruction efficiency by approximately fivefold.
Conclusion: REFINE-MORE enables accurate and efficient scan-specific multiparametric qMRI reconstruction, providing a flexible solution for high-dimensional, accelerated qMRI applications. 

**Abstract (ZH)**: 目的：开发一种自我监督的扫描特定深度学习框架，用于重建加速的多参数定量磁共振成像（qMRI）。 

---
# FECT: Factuality Evaluation of Interpretive AI-Generated Claims in Contact Center Conversation Transcripts 

**Title (ZH)**: FECT：接触中心对话转录中解释性AI生成声明的事实性评估 

**Authors**: Hagyeong Shin, Binoy Robin Dalal, Iwona Bialynicka-Birula, Navjot Matharu, Ryan Muir, Xingwei Yang, Samuel W. K. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2508.00889)  

**Abstract**: Large language models (LLMs) are known to hallucinate, producing natural language outputs that are not grounded in the input, reference materials, or real-world knowledge. In enterprise applications where AI features support business decisions, such hallucinations can be particularly detrimental. LLMs that analyze and summarize contact center conversations introduce a unique set of challenges for factuality evaluation, because ground-truth labels often do not exist for analytical interpretations about sentiments captured in the conversation and root causes of the business problems. To remedy this, we first introduce a \textbf{3D} -- \textbf{Decompose, Decouple, Detach} -- paradigm in the human annotation guideline and the LLM-judges' prompt to ground the factuality labels in linguistically-informed evaluation criteria. We then introduce \textbf{FECT}, a novel benchmark dataset for \textbf{F}actuality \textbf{E}valuation of Interpretive AI-Generated \textbf{C}laims in Contact Center Conversation \textbf{T}ranscripts, labeled under our 3D paradigm. Lastly, we report our findings from aligning LLM-judges on the 3D paradigm. Overall, our findings contribute a new approach for automatically evaluating the factuality of outputs generated by an AI system for analyzing contact center conversations. 

**Abstract (ZH)**: 大型语言模型（LLMs） Known to Hallucinate, Introducing Challenges for Factuality Evaluation in Enterprise Applications: A 3D Paradigm for Factuality Assessment in Interpretive AI-Generated Claims from Contact Center Conversations 

---
# Multi-Grained Temporal-Spatial Graph Learning for Stable Traffic Flow Forecasting 

**Title (ZH)**: 多粒度时空图学习方法在稳定交通流预测中的应用 

**Authors**: Zhenan Lin, Yuni Lai, Wai Lun Lo, Richard Tai-Chiu Hsung, Harris Sik-Ho Tsang, Xiaoyu Xue, Kai Zhou, Yulin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00884)  

**Abstract**: Time-evolving traffic flow forecasting are playing a vital role in intelligent transportation systems and smart cities. However, the dynamic traffic flow forecasting is a highly nonlinear problem with complex temporal-spatial dependencies. Although the existing methods has provided great contributions to mine the temporal-spatial patterns in the complex traffic networks, they fail to encode the globally temporal-spatial patterns and are prone to overfit on the pre-defined geographical correlations, and thus hinder the model's robustness on the complex traffic environment. To tackle this issue, in this work, we proposed a multi-grained temporal-spatial graph learning framework to adaptively augment the globally temporal-spatial patterns obtained from a crafted graph transformer encoder with the local patterns from the graph convolution by a crafted gated fusion unit with residual connection techniques. Under these circumstances, our proposed model can mine the hidden global temporal-spatial relations between each monitor stations and balance the relative importance of local and global temporal-spatial patterns. Experiment results demonstrate the strong representation capability of our proposed method and our model consistently outperforms other strong baselines on various real-world traffic networks. 

**Abstract (ZH)**: 时空演化交通流预测在智能交通系统和智慧城市中发挥着重要作用。然而，动态交通流预测是一个高度非线性问题，涉及复杂的时空依赖关系。尽管现有的方法在挖掘复杂交通网络中的时空模式方面做出了巨大贡献，但它们无法编码全局时空模式，并且容易过度拟合预定义的地理关联，从而阻碍了模型在复杂交通环境中的 robustness。为了解决这一问题，本文提出了一种多粒度时空图学习框架，通过一个精心设计的门控融合单元结合残差连接技术，动态增强从构造的图变换器编码器中获得的全局时空模式与图卷积中的局部模式。在这种情况下，我们的模型可以揭示每个监控站之间的隐藏全局时空关系，并平衡局部和全局时空模式的相对重要性。实验结果表明，我们提出的方法具有强大的表示能力，并且在各种实际交通网络中始终优于其他强大的基线方法。 

---
# Reproducibility of Machine Learning-Based Fault Detection and Diagnosis for HVAC Systems in Buildings: An Empirical Study 

**Title (ZH)**: 基于机器学习的 HVAC 系统建筑故障检测与诊断的可重复性研究：一项实证研究 

**Authors**: Adil Mukhtar, Michael Hadwiger, Franz Wotawa, Gerald Schweiger  

**Link**: [PDF](https://arxiv.org/pdf/2508.00880)  

**Abstract**: Reproducibility is a cornerstone of scientific research, enabling independent verification and validation of empirical findings. The topic gained prominence in fields such as psychology and medicine, where concerns about non - replicable results sparked ongoing discussions about research practices. In recent years, the fast-growing field of Machine Learning (ML) has become part of this discourse, as it faces similar concerns about transparency and reliability. Some reproducibility issues in ML research are shared with other fields, such as limited access to data and missing methodological details. In addition, ML introduces specific challenges, including inherent nondeterminism and computational constraints. While reproducibility issues are increasingly recognized by the ML community and its major conferences, less is known about how these challenges manifest in applied disciplines. This paper contributes to closing this gap by analyzing the transparency and reproducibility standards of ML applications in building energy systems. The results indicate that nearly all articles are not reproducible due to insufficient disclosure across key dimensions of reproducibility. 72% of the articles do not specify whether the dataset used is public, proprietary, or commercially available. Only two papers share a link to their code - one of which was broken. Two-thirds of the publications were authored exclusively by academic researchers, yet no significant differences in reproducibility were observed compared to publications with industry-affiliated authors. These findings highlight the need for targeted interventions, including reproducibility guidelines, training for researchers, and policies by journals and conferences that promote transparency and reproducibility. 

**Abstract (ZH)**: 可重复性是科学研究的基石，能使独立的验证和验证实证发现成为可能。该话题在心理学和医学等领域中因其关于不可重复结果的担忧而变得尤为重要，引发了关于研究方法的持续讨论。近年来，快速发展的机器学习（ML）领域也加入了这一讨论，因为它面临着透明度和可靠性的相似关切。ML研究中的某些可重复性问题与其他领域共享，例如数据访问有限和方法学细节缺失。此外，ML还带来了特定的挑战，包括固有的非确定性和计算约束。虽然ML社区及其主要会议越来越认识到这些问题，但人们对这些挑战在应用学科中的表现知之甚少。本文通过分析建筑能源系统中机器学习应用的透明度和可重复性标准，旨在缩小这一差距。研究结果表明，几乎所有文章都无法实现可重复性，因为在关键的可重复性维度上披露不足。72%的文章没有明确说明所使用的数据集是公开的、专有的还是商业可用的。只有两篇论文共享了其代码的链接，其中一个是无效链接。三分之二的出版物由学术研究人员独撰，但与有工业关联作者的出版物相比，没有观察到显著的可重复性差异。这些发现突显了需要有针对性的干预措施，包括可重复性指南、研究人员培训以及促进透明度和可重复性的期刊和会议政策。 

---
# GNN-ASE: Graph-Based Anomaly Detection and Severity Estimation in Three-Phase Induction Machines 

**Title (ZH)**: 基于图的三相感应电机异常检测与严重程度估计 

**Authors**: Moutaz Bellah Bentrad, Adel Ghoggal, Tahar Bahi, Abderaouf Bahi  

**Link**: [PDF](https://arxiv.org/pdf/2508.00879)  

**Abstract**: The diagnosis of induction machines has traditionally relied on model-based methods that require the development of complex dynamic models, making them difficult to implement and computationally expensive. To overcome these limitations, this paper proposes a model-free approach using Graph Neural Networks (GNNs) for fault diagnosis in induction machines. The focus is on detecting multiple fault types -- including eccentricity, bearing defects, and broken rotor bars -- under varying severity levels and load conditions. Unlike traditional approaches, raw current and vibration signals are used as direct inputs, eliminating the need for signal preprocessing or manual feature extraction. The proposed GNN-ASE model automatically learns and extracts relevant features from raw inputs, leveraging the graph structure to capture complex relationships between signal types and fault patterns. It is evaluated for both individual fault detection and multi-class classification of combined fault conditions. Experimental results demonstrate the effectiveness of the proposed model, achieving 92.5\% accuracy for eccentricity defects, 91.2\% for bearing faults, and 93.1\% for broken rotor bar detection. These findings highlight the model's robustness and generalization capability across different operational scenarios. The proposed GNN-based framework offers a lightweight yet powerful solution that simplifies implementation while maintaining high diagnostic performance. It stands as a promising alternative to conventional model-based diagnostic techniques for real-world induction machine monitoring and predictive maintenance. 

**Abstract (ZH)**: 基于图神经网络的感应电机故障诊断方法 

---
# Satellite Connectivity Prediction for Fast-Moving Platforms 

**Title (ZH)**: 快移动平台卫星连接性预测 

**Authors**: Chao Yan, Babak Mafakheri  

**Link**: [PDF](https://arxiv.org/pdf/2508.00877)  

**Abstract**: Satellite connectivity is gaining increased attention as the demand for seamless internet access, especially in transportation and remote areas, continues to grow. For fast-moving objects such as aircraft, vehicles, or trains, satellite connectivity is critical due to their mobility and frequent presence in areas without terrestrial coverage. Maintaining reliable connectivity in these cases requires frequent switching between satellite beams, constellations, or orbits. To enhance user experience and address challenges like long switching times, Machine Learning (ML) algorithms can analyze historical connectivity data and predict network quality at specific locations. This allows for proactive measures, such as network switching before connectivity issues arise. In this paper, we analyze a real dataset of communication between a Geostationary Orbit (GEO) satellite and aircraft over multiple flights, using ML to predict signal quality. Our prediction model achieved an F1 score of 0.97 on the test data, demonstrating the accuracy of machine learning in predicting signal quality during flight. By enabling seamless broadband service, including roaming between different satellite constellations and providers, our model addresses the need for real-time predictions of signal quality. This approach can further be adapted to automate satellite and beam-switching mechanisms to improve overall communication efficiency. The model can also be retrained and applied to any moving object with satellite connectivity, using customized datasets, including connected vehicles and trains. 

**Abstract (ZH)**: 卫星通信正因对无缝互联网接入需求的持续增长而备受关注，尤其是在交通运输和偏远地区。对于如飞机、车辆或列车等快速移动的对象，卫星通信因其移动性及在陆地覆盖不足区域的频繁存在而至关重要。在这种情况下，保持可靠的连接需要频繁切换卫星波束、星座或轨道。为了提升用户体验并应对如长切换时间等问题，机器学习（ML）算法可以分析历史连接数据并预测特定位置的网络质量。这使得可以在出现连接问题之前采取主动措施，如网络切换。本文分析了多架航班中地球静止轨道（GEO）卫星与飞机之间的通信数据集，使用机器学习来预测信号质量。我们的预测模型在测试数据上的F1分数达到了0.97，证明了机器学习在飞行过程中预测信号质量的准确性。通过提供无缝宽带服务，包括不同卫星星座和提供商之间的漫游，我们的模型满足了实时预测信号质量的需求。该方法还可进一步调整以自动化卫星和波束切换机制，从而提高整体通信效率。该模型还可以通过定制数据集重新训练并应用于任何具有卫星连接的移动对象，包括连接车辆和列车。 

---
# Patents as Knowledge Artifacts: An Information Science Perspective on Global Innovation 

**Title (ZH)**: 专利作为知识 artefact：信息科学视角下的全球创新 

**Authors**: M. S. Rajeevan, B. Mini Devi  

**Link**: [PDF](https://arxiv.org/pdf/2508.00871)  

**Abstract**: In an age of fast-paced technological change, patents have evolved into not only legal mechanisms of intellectual property, but also structured storage containers of knowledge full of metadata, categories, and formal innovation. This chapter proposes to reframe patents in the context of information science, by focusing on patents as knowledge artifacts, and by seeing patents as fundamentally tied to the global movement of scientific and technological knowledge. With a focus on three areas, the inventions of AIs, biotech patents, and international competition with patents, this work considers how new technologies are challenging traditional notions of inventorship, access, and moral this http URL chapter provides a critical analysis of AI's implications for patent authorship and prior art searches, ownership issues arising from proprietary claims in biotechnology to ethical dilemmas, and the problem of using patents for strategic advantage in a global context of innovation competition. In this analysis, the chapter identified the importance of organizing information, creating metadata standards about originality, implementing retrieval systems to access previous works, and ethical contemplation about patenting unseen relationships in innovation ecosystems. Ultimately, the chapter called for a collaborative, transparent, and ethically-based approach in managing knowledge in the patenting environment highlighting the role for information professionals and policy to contribute to access equity in innovation. 

**Abstract (ZH)**: 在快速技术变革的时代，专利不仅演化为知识产权的法律机制，也成为充满元数据、分类和正式创新的知识结构存储容器。本章建议从信息科学的视角重新审视专利，将专利视为知识 artefact，并将其视为与全球科学与技术知识传播基本相关的核心要素。本章以人工智能发明、生物技术专利和国际专利竞争这三个领域为重点，探讨新技术如何挑战传统的发明人身份、访问权和道德标准问题。本章对人工智能对专利作者身份和现有技术搜索的影响、因生物技术的专有主张引发的所有权问题及伦理困境，以及在全球创新竞争背景下的专利战略优势问题进行了批判性分析。本章强调了组织信息、制定原创性元数据标准、实施检索系统以访问先前作品，以及对创新生态系统中未见关系进行专利申请的伦理思考的重要性。最终，本章倡导在专利环境中采取协作、透明和基于伦理的方法来管理知识，并强调信息专业人员和政策在促进创新访问公平方面的作用。 

---
# Better Recommendations: Validating AI-generated Subject Terms Through LOC Linked Data Service 

**Title (ZH)**: 更好的推荐：通过LOC链接数据服务验证AI生成的主题词 

**Authors**: Kwok Leong Tang, Yi Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00867)  

**Abstract**: This article explores the integration of AI-generated subject terms into library cataloging, focusing on validation through the Library of Congress Linked Data Service. It examines the challenges of traditional subject cataloging under the Library of Congress Subject Headings system, including inefficiencies and cataloging backlogs. While generative AI shows promise in expediting cataloging workflows, studies reveal significant limitations in the accuracy of AI-assigned subject headings. The article proposes a hybrid approach combining AI technology with human validation through LOC Linked Data Service, aiming to enhance the precision, efficiency, and overall quality of metadata creation in library cataloging practices. 

**Abstract (ZH)**: 本文探讨将AI生成的主题词纳入图书馆分类法的整合，重点通过美国国会图书馆关联数据服务进行验证。文章考察了使用美国国会图书馆主题词表系统传统主题分类的挑战，包括效率低下和分类积压问题。虽然生成式AI有潜力加速分类流程，但研究表明，AI分配的主题词准确性存在显著限制。本文提议结合AI技术与通过LOC关联数据服务进行的人工验证的混合方法，旨在提高图书馆分类实践中元数据创建的精确性、效率和整体质量。 

---
# Deploying Geospatial Foundation Models in the Real World: Lessons from WorldCereal 

**Title (ZH)**: 在现实世界部署地理空间基础模型：WorldCereal的启示 

**Authors**: Christina Butsko, Kristof Van Tricht, Gabriel Tseng, Giorgia Milli, David Rolnick, Ruben Cartuyvels, Inbal Becker Reshef, Zoltan Szantoi, Hannah Kerner  

**Link**: [PDF](https://arxiv.org/pdf/2508.00858)  

**Abstract**: The increasing availability of geospatial foundation models has the potential to transform remote sensing applications such as land cover classification, environmental monitoring, and change detection. Despite promising benchmark results, the deployment of these models in operational settings is challenging and rare. Standardized evaluation tasks often fail to capture real-world complexities relevant for end-user adoption such as data heterogeneity, resource constraints, and application-specific requirements. This paper presents a structured approach to integrate geospatial foundation models into operational mapping systems. Our protocol has three key steps: defining application requirements, adapting the model to domain-specific data and conducting rigorous empirical testing. Using the Presto model in a case study for crop mapping, we demonstrate that fine-tuning a pre-trained model significantly improves performance over conventional supervised methods. Our results highlight the model's strong spatial and temporal generalization capabilities. Our protocol provides a replicable blueprint for practitioners and lays the groundwork for future research to operationalize foundation models in diverse remote sensing applications. Application of the protocol to the WorldCereal global crop-mapping system showcases the framework's scalability. 

**Abstract (ZH)**: 地理空间基础模型的日益可用有望变革土地覆盖分类、环境监测和变化检测等遥感应用。尽管基准测试结果充满 promise，但在实际操作环境中的部署仍具挑战性和罕见性。标准评估任务往往未能捕捉到影响最终用户采用的实际复杂性，如数据异质性、资源限制和应用特定要求。本文提出了一种结构化方法，将地理空间基础模型集成到操作性制图系统中。我们的协议包含三个关键步骤：定义应用要求、适应领域特定数据以及进行严格的实证测试。通过在作物制图案例研究中使用 Presto 模型，我们证明了对预训练模型进行微调显著优于传统监督方法。我们的结果突显了该模型在空间和时间上的强泛化能力。我们的协议为实践者提供了一个可复制的范本，并为未来研究在多种遥感应用中实现基础模型奠定了基础。将该协议应用于全球作物制图系统 WorldCereal 展示了该框架的可扩展性。 

---
# EthicAlly: a Prototype for AI-Powered Research Ethics Support for the Social Sciences and Humanities 

**Title (ZH)**: EthicAlly：社会科学与人文科学领域的AI驱动研究伦理支持原型 

**Authors**: Steph Grohmann  

**Link**: [PDF](https://arxiv.org/pdf/2508.00856)  

**Abstract**: In biomedical science, review by a Research Ethics Committee (REC) is an indispensable way of protecting human subjects from harm. However, in social science and the humanities, mandatory ethics compliance has long been met with scepticism as biomedical models of ethics can map poorly onto methodologies involving complex socio-political and cultural considerations. As a result, tailored ethics training and support as well as access to RECs with the necessary expertise is lacking in some areas, including parts of Europe and low- and middle-income countries. This paper suggests that Generative AI can meaningfully contribute to closing these gaps, illustrating this claim by presenting EthicAlly, a proof-of-concept prototype for an AI-powered ethics support system for social science and humanities researchers. Drawing on constitutional AI technology and a collaborative prompt development methodology, EthicAlly provides structured ethics assessment that incorporates both universal ethics principles and contextual and interpretive considerations relevant to most social science research. In supporting researchers in ethical research design and preparation for REC submission, this kind of system can also contribute to easing the burden on institutional RECs, without attempting to automate or replace human ethical oversight. 

**Abstract (ZH)**: 生成式AI在填补社会科学和人文科学伦理合规缺口中的潜在贡献：EthicAlly原型概览 

---
# Inclusive Review on Advances in Masked Human Face Recognition Technologies 

**Title (ZH)**: 包容性综述：遮罩人脸 Recognition 技术进展 

**Authors**: Ali Haitham Abdul Amir, Zainab N. Nemer  

**Link**: [PDF](https://arxiv.org/pdf/2508.00841)  

**Abstract**: Masked Face Recognition (MFR) is an increasingly important area in biometric recognition technologies, especially with the widespread use of masks as a result of the COVID-19 pandemic. This development has created new challenges for facial recognition systems due to the partial concealment of basic facial features. This paper aims to provide a comprehensive review of the latest developments in the field, with a focus on deep learning techniques, especially convolutional neural networks (CNNs) and twin networks (Siamese networks), which have played a pivotal role in improving the accuracy of covering face recognition. The paper discusses the most prominent challenges, which include changes in lighting, different facial positions, partial concealment, and the impact of mask types on the performance of systems. It also reviews advanced technologies developed to overcome these challenges, including data enhancement using artificial databases and multimedia methods to improve the ability of systems to generalize. In addition, the paper highlights advance in deep network design, feature extraction techniques, evaluation criteria, and data sets used in this area. Moreover, it reviews the various applications of masked face recognition in the fields of security and medicine, highlighting the growing importance of these systems in light of recurrent health crises and increasing security threats. Finally, the paper focuses on future research trends such as developing more efficient algorithms and integrating multimedia technologies to improve the performance of recognition systems in real-world environments and expand their applications. 

**Abstract (ZH)**: 掩码面部识别（MFR）是生物特征识别技术中日益重要的领域，尤其是由于COVID-19疫情广泛使用口罩所致。这一发展为面部识别系统带来了新的挑战，因为基本面部特征部分被遮挡。本文旨在全面概述该领域的最新进展，重点关注深度学习技术，特别是卷积神经网络（CNNs）和孪生网络（Siamese网络），这些技术在提高遮挡面部识别精度方面发挥了关键作用。本文讨论了最显著的挑战，包括光照变化、不同面部位置、部分遮挡以及不同口罩类型对系统性能的影响。此外，本文还回顾了为克服这些挑战而开发的先进技术，包括使用人工数据库的数据增强以及多媒体方法以提高系统泛化能力。文中还强调了该领域深度网络设计的发展、特征提取技术、评估标准和数据集的进展。此外，本文还回顾了掩码面部识别在安全和医疗领域的各种应用，突显了这些系统在反复出现的健康危机和不断增加的安全威胁背景下日益重要的作用。最后，本文集中在未来研究趋势上，如开发更高效的算法和整合多媒体技术，以改善实际环境中的识别系统性能并扩展其应用领域。 

---
# PCS Workflow for Veridical Data Science in the Age of AI 

**Title (ZH)**: AI时代可信数据科学的PCS工作流 

**Authors**: Zachary T. Rewolinski, Bin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00835)  

**Abstract**: Data science is a pillar of artificial intelligence (AI), which is transforming nearly every domain of human activity, from the social and physical sciences to engineering and medicine. While data-driven findings in AI offer unprecedented power to extract insights and guide decision-making, many are difficult or impossible to replicate. A key reason for this challenge is the uncertainty introduced by the many choices made throughout the data science life cycle (DSLC). Traditional statistical frameworks often fail to account for this uncertainty. The Predictability-Computability-Stability (PCS) framework for veridical (truthful) data science offers a principled approach to addressing this challenge throughout the DSLC. This paper presents an updated and streamlined PCS workflow, tailored for practitioners and enhanced with guided use of generative AI. We include a running example to display the PCS framework in action, and conduct a related case study which showcases the uncertainty in downstream predictions caused by judgment calls in the data cleaning stage. 

**Abstract (ZH)**: 数据科学是人工智能（AI）的支柱，正在几乎每一个领域的人类活动中产生变革，从社会科学和物理科学到工程和医学。尽管AI中的数据驱动发现提供了前所未有的力量来提取洞察并指导决策，但许多发现难以复制或根本无法复制。这一挑战的关键原因是在数据科学生命周期（DSLC）中的众多选择引入了不确定性。传统的统计框架往往未能考虑到这种不确定性。揭示性（真实）数据科学的可预报性-可计算性-稳定性（PCS）框架提供了一种原则性的方法，以在整个数据科学生命周期中应对这一挑战。本文介绍了更新和完善后的PCS工作流程，针对实践工作者，并增强了生成式AI的指导使用。我们提供了一个示例案例来展示PCS框架的实际应用，并进行了一项相关案例研究，展示了数据清洗阶段判断决策导致的下游预测中的不确定性。 

---
# Bike-Bench: A Bicycle Design Benchmark for Generative Models with Objectives and Constraints 

**Title (ZH)**: Bike-Bench：面向生成模型的目标与约束自行车设计基准 

**Authors**: Lyle Regenwetter, Yazan Abu Obaideh, Fabien Chiotti, Ioanna Lykourentzou, Faez Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2508.00830)  

**Abstract**: We introduce Bike-Bench, an engineering design benchmark for evaluating generative models on problems with multiple real-world objectives and constraints. As generative AI's reach continues to grow, evaluating its capability to understand physical laws, human guidelines, and hard constraints grows increasingly important. Engineering product design lies at the intersection of these difficult tasks, providing new challenges for AI capabilities. Bike-Bench evaluates AI models' capability to generate designs that not only resemble the dataset, but meet specific performance objectives and constraints. To do so, Bike-Bench quantifies a variety of human-centered and multiphysics performance characteristics, such as aerodynamics, ergonomics, structural mechanics, human-rated usability, and similarity to subjective text or image prompts. Supporting the benchmark are several datasets of simulation results, a dataset of 10K human-rated bicycle assessments, and a synthetically-generated dataset of 1.4M designs, each with a parametric, CAD/XML, SVG, and PNG representation. Bike-Bench is uniquely configured to evaluate tabular generative models, LLMs, design optimization, and hybrid algorithms side-by-side. Our experiments indicate that LLMs and tabular generative models fall short of optimization and optimization-augmented generative models in both validity and optimality scores, suggesting significant room for improvement. We hope Bike-Bench, a first-of-its-kind benchmark, will help catalyze progress in generative AI for constrained multi-objective engineering design problems. Code, data, and other resources are published at this http URL. 

**Abstract (ZH)**: 我们介绍Bike-Bench：用于评价生成模型在具有多个现实世界目标和约束的问题上的工程设计基准。 

---
# A Schema.org Mapping for Brazilian Legal Norms: Toward Interoperable Legal Graphs and Open Government Data 

**Title (ZH)**: 巴西法律规范的Schema.org 映射：面向互操作法律图谱和开放政府数据的研究 

**Authors**: Hudson de Martim  

**Link**: [PDF](https://arxiv.org/pdf/2508.00827)  

**Abstract**: Open Government Data (OGD) initiatives aim to enhance transparency and public participation by making government data openly accessible. However, structuring legal norms for machine readability remains a critical challenge for advancing Legal Tech applications such as Legal Knowledge Graphs (LKGs). Focusing on the this http URL portal initiative by the Brazilian National Congress, we propose a unified mapping of Brazilian legislation to the this http URL vocabulary via JSON-LD and Linked Data. Our approach covers both the conceptual "Norm" entity (mapped to sdo:Legislation) and its digital publications or manifestations (mapped to sdo:LegislationObject). We detail key properties for each type, providing concrete examples and considering URN identifiers (per the LexML standard), multilingual support, versioning in the Official Journal, and inter-norm relationships (e.g., citations and references). Our structured schema improves the quality and interoperability of Brazilian legal data, fosters integration within the global OGD ecosystem, and facilitates the creation of a wor 

**Abstract (ZH)**: 开放政府数据（OGD）倡议旨在通过使政府数据公开 accessible 提高透明度和公众参与度。然而，为机器可读性制定法律规范仍然是推进如法律知识图谱（LKGs）之类的法律科技应用的关键挑战。基于巴西联邦议会的 this http URL 项目，我们提出了一种将巴西立法统一映射到 this http URL 词汇表的方法，通过 JSON-LD 和 Linked Data。我们的方法涵盖了概念性的“Norm”实体（映射到 sdo:Legislation）及其数字出版物或表现形式（映射到 sdo:LegislationObject）。我们为每种类型详细列出了关键属性，提供了具体的示例，并考虑了 LexML 标准的 URN 标识符、多语言支持、官方公报中的版本控制以及相互之间的关系（如引用和参考）。我们的结构化模式提高了巴西法律数据的质量和互操作性，促进了与全球 OGD 生态系统的整合，并促进了法律知识图谱的创建。 

---
# Towards Actionable Pedagogical Feedback: A Multi-Perspective Analysis of Mathematics Teaching and Tutoring Dialogue 

**Title (ZH)**: 面向可行的教学反馈：数学教学与辅导对话的多视角分析 

**Authors**: Jannatun Naim, Jie Cao, Fareen Tasneem, Jennifer Jacobs, Brent Milne, James Martin, Tamara Sumner  

**Link**: [PDF](https://arxiv.org/pdf/2505.07161)  

**Abstract**: Effective feedback is essential for refining instructional practices in mathematics education, and researchers often turn to advanced natural language processing (NLP) models to analyze classroom dialogues from multiple perspectives. However, utterance-level discourse analysis encounters two primary challenges: (1) multifunctionality, where a single utterance may serve multiple purposes that a single tag cannot capture, and (2) the exclusion of many utterances from domain-specific discourse move classifications, leading to their omission in feedback. To address these challenges, we proposed a multi-perspective discourse analysis that integrates domain-specific talk moves with dialogue act (using the flattened multi-functional SWBD-MASL schema with 43 tags) and discourse relation (applying Segmented Discourse Representation Theory with 16 relations). Our top-down analysis framework enables a comprehensive understanding of utterances that contain talk moves, as well as utterances that do not contain talk moves. This is applied to two mathematics education datasets: TalkMoves (teaching) and SAGA22 (tutoring). Through distributional unigram analysis, sequential talk move analysis, and multi-view deep dive, we discovered meaningful discourse patterns, and revealed the vital role of utterances without talk moves, demonstrating that these utterances, far from being mere fillers, serve crucial functions in guiding, acknowledging, and structuring classroom discourse. These insights underscore the importance of incorporating discourse relations and dialogue acts into AI-assisted education systems to enhance feedback and create more responsive learning environments. Our framework may prove helpful for providing human educator feedback, but also aiding in the development of AI agents that can effectively emulate the roles of both educators and students. 

**Abstract (ZH)**: 有效的反馈对于数学教育中的教学实践改进至关重要，研究人员常借助先进的自然语言处理（NLP）模型从多角度分析课堂对话。然而，话语单元层面的分析面临两大挑战：（1）多功能性，即一个话语单元可能包含多个无法用单一标签捕捉的目的；（2）许多话语单元因专属领域的话语移动分类排除在外，导致其在反馈中被忽略。为应对这些挑战，我们提出了一种多视角话语分析框架，该框架结合了专属领域的对话动作（采用扁平化的多功能性SWBD-MASL方案，包含43个标签）和话语关系（应用分段的话语表示理论，包含16种关系）。自上而下的分析框架能够全面理解包含话语移动和不包含话语移动的话语单元。我们将其应用于两个数学教育数据集：TalkMoves（教学）和SAGA22（辅导）。通过分布分析、序列对话动作分析和多视角深入分析，我们发现了有意义的话语模式，并揭示了不包含话语移动的话语单元的关键作用，证明这些话语单元远非仅仅是填充物，而是引导、认可和结构化课堂对话的重要手段。这些见解强调了将话语关系和对话动作纳入AI辅助教育系统以提升反馈并创造更具响应性的学习环境的重要性。我们的框架不仅有助于提供给人类教育者的反馈，还能协助开发能够有效模仿教育者和学生角色的AI代理。 

---
# Enhancing Talk Moves Analysis in Mathematics Tutoring through Classroom Teaching Discourse 

**Title (ZH)**: 通过课堂教学 discourse 提高数学辅导中谈话移动分析 

**Authors**: Jie Cao, Abhijit Suresh, Jennifer Jacobs, Charis Clevenger, Amanda Howard, Chelsea Brown, Brent Milne, Tom Fischaber, Tamara Sumner, James H. Martin  

**Link**: [PDF](https://arxiv.org/pdf/2412.13395)  

**Abstract**: Human tutoring interventions play a crucial role in supporting student learning, improving academic performance, and promoting personal growth. This paper focuses on analyzing mathematics tutoring discourse using talk moves - a framework of dialogue acts grounded in Accountable Talk theory. However, scaling the collection, annotation, and analysis of extensive tutoring dialogues to develop machine learning models is a challenging and resource-intensive task. To address this, we present SAGA22, a compact dataset, and explore various modeling strategies, including dialogue context, speaker information, pretraining datasets, and further fine-tuning. By leveraging existing datasets and models designed for classroom teaching, our results demonstrate that supplementary pretraining on classroom data enhances model performance in tutoring settings, particularly when incorporating longer context and speaker information. Additionally, we conduct extensive ablation studies to underscore the challenges in talk move modeling. 

**Abstract (ZH)**: 人类辅导干预在支持学生学习、提高学术成绩和促进个人成长中发挥着重要作用。本文focus于利用对话动作——基于可问责对话理论的框架——分析数学辅导对话。然而，扩展收集、标注和分析大量辅导对话以开发机器学习模型是一项具有挑战性和资源密集的任务。为了解决这一问题，我们提出了SAGA22紧凑型数据集，并探索了包括对话上下文、演讲者信息、预训练数据集和进一步微调在内的各种建模策略。通过利用为课堂教学设计的现有数据集和模型，我们的结果表明，在辅导环境中，补充课堂数据的预训练可以提高模型性能，尤其是在结合更长的上下文和演讲者信息时。此外，我们进行了广泛的消融研究以强调谈话动作建模中的挑战。 

---
# Observing Dialogue in Therapy: Categorizing and Forecasting Behavioral Codes 

**Title (ZH)**: 在治疗中观察对话：行为编码的分类与预测 

**Authors**: Jie Cao, Michael Tanana, Zac E. Imel, Eric Poitras, David C. Atkins, Vivek Srikumar  

**Link**: [PDF](https://arxiv.org/pdf/1907.00326)  

**Abstract**: Automatically analyzing dialogue can help understand and guide behavior in domains such as counseling, where interactions are largely mediated by conversation. In this paper, we study modeling behavioral codes used to asses a psychotherapy treatment style called Motivational Interviewing (MI), which is effective for addressing substance abuse and related problems. Specifically, we address the problem of providing real-time guidance to therapists with a dialogue observer that (1) categorizes therapist and client MI behavioral codes and, (2) forecasts codes for upcoming utterances to help guide the conversation and potentially alert the therapist. For both tasks, we define neural network models that build upon recent successes in dialogue modeling. Our experiments demonstrate that our models can outperform several baselines for both tasks. We also report the results of a careful analysis that reveals the impact of the various network design tradeoffs for modeling therapy dialogue. 

**Abstract (ZH)**: 自动分析对话有助于理解并引导涉及咨询等领域的行为，其中对话主要由交流驱动。本文研究了用于评估一种有效的心理治疗风格（动机访谈）的行为代码建模方法，该风格适用于处理物质滥用及相关问题。具体来说，我们提出了一个对话观察者以实现实时指导，该观察者可以（1）分类治疗师和来访者使用的动机访谈行为代码，（2）预测未来言论的行为代码以帮助引导对话，并可能提醒治疗师。在两个任务中，我们定义了基于最近对话建模成功经验的神经网络模型。实验表明，我们的模型在两个任务上均优于若干基准。我们还报告了对各种网络设计权衡的仔细分析结果，揭示了其对治疗对话建模的影响。 

---
