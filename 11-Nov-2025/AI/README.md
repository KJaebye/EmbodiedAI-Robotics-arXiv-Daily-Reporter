# DigiData: Training and Evaluating General-Purpose Mobile Control Agents 

**Title (ZH)**: DigiData: 训练和评估通用移动控制代理 

**Authors**: Yuxuan Sun, Manchen Wang, Shengyi Qian, William R. Wong, Eric Gan, Pierluca D'Oro, Alejandro Castillejo Munoz, Sneha Silwal, Pedro Matias, Nitin Kamra, Satwik Kottur, Nick Raines, Xuanyi Zhao, Joy Chen, Joseph Greer, Andrea Madotto, Allen Bolourchi, James Valori, Kevin Carlberg, Karl Ridgeway, Joseph Tighe  

**Link**: [PDF](https://arxiv.org/pdf/2511.07413)  

**Abstract**: AI agents capable of controlling user interfaces have the potential to transform human interaction with digital devices. To accelerate this transformation, two fundamental building blocks are essential: high-quality datasets that enable agents to achieve complex and human-relevant goals, and robust evaluation methods that allow researchers and practitioners to rapidly enhance agent performance. In this paper, we introduce DigiData, a large-scale, high-quality, diverse, multi-modal dataset designed for training mobile control agents. Unlike existing datasets, which derive goals from unstructured interactions, DigiData is meticulously constructed through comprehensive exploration of app features, resulting in greater diversity and higher goal complexity. Additionally, we present DigiData-Bench, a benchmark for evaluating mobile control agents on real-world complex tasks. We demonstrate that the commonly used step-accuracy metric falls short in reliably assessing mobile control agents and, to address this, we propose dynamic evaluation protocols and AI-powered evaluations as rigorous alternatives for agent assessment. Our contributions aim to significantly advance the development of mobile control agents, paving the way for more intuitive and effective human-device interactions. 

**Abstract (ZH)**: AI代理能够控制用户界面具有潜在的潜力，可以改变人类与数字设备的交互方式。为了加速这一变革，两个基本构建模块是必不可少的：高质量的数据集，使代理能够实现复杂且与人类相关的目标，以及稳健的评估方法，让研究人员和实践者能够快速提升代理的性能。在本文中，我们介绍了DigiData，这是一个大规模、高质量、多样化的多模态数据集，用于培训移动控制代理。与现有数据集从无结构交互中提取目标不同，DigiData是通过全面探索应用程序功能精心构建的，从而具有更高的多样性和更高的目标复杂性。此外，我们还介绍了DigiData-Bench，这是一个用于评估移动控制代理在现实复杂任务上的基准。我们表明常用的步长准确性度量在可靠评估移动控制代理方面的不足，并提出动态评估协议和基于AI的评估作为代理评估的严格替代方案。我们的贡献旨在显著推动移动控制代理的发展，为更直观和有效的用户-设备交互铺平道路。 

---
# DeepPersona: A Generative Engine for Scaling Deep Synthetic Personas 

**Title (ZH)**: DeepPersona：一个生成深合成人设的引擎 

**Authors**: Zhen Wang, Yufan Zhou, Zhongyan Luo, Lyumanshan Ye, Adam Wood, Man Yao, Luoshang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2511.07338)  

**Abstract**: Simulating human profiles by instilling personas into large language models (LLMs) is rapidly transforming research in agentic behavioral simulation, LLM personalization, and human-AI alignment. However, most existing synthetic personas remain shallow and simplistic, capturing minimal attributes and failing to reflect the rich complexity and diversity of real human identities. We introduce DEEPPERSONA, a scalable generative engine for synthesizing narrative-complete synthetic personas through a two-stage, taxonomy-guided method. First, we algorithmically construct the largest-ever human-attribute taxonomy, comprising over hundreds of hierarchically organized attributes, by mining thousands of real user-ChatGPT conversations. Second, we progressively sample attributes from this taxonomy, conditionally generating coherent and realistic personas that average hundreds of structured attributes and roughly 1 MB of narrative text, two orders of magnitude deeper than prior works. Intrinsic evaluations confirm significant improvements in attribute diversity (32 percent higher coverage) and profile uniqueness (44 percent greater) compared to state-of-the-art baselines. Extrinsically, our personas enhance GPT-4.1-mini's personalized question answering accuracy by 11.6 percent on average across ten metrics and substantially narrow (by 31.7 percent) the gap between simulated LLM citizens and authentic human responses in social surveys. Our generated national citizens reduced the performance gap on the Big Five personality test by 17 percent relative to LLM-simulated citizens. DEEPPERSONA thus provides a rigorous, scalable, and privacy-free platform for high-fidelity human simulation and personalized AI research. 

**Abstract (ZH)**: 通过将人设注入大型语言模型（LLMs）模拟人类画像正迅速变革代理行为模拟、LLM个性化和人类-AI对齐的研究。然而，现有的大多数合成人设仍然浅显简陋，仅捕捉到最少的属性，未能反映真实人类身份的丰富复杂性和多样性。我们提出DEEPPERSONA，一种通过分类学引导的两阶段生成方法合成叙事完整合成人设的可扩展生成引擎。首先，我们通过挖掘数千名真实用户与ChatGPT的对话，构建迄今为止最大的人类属性分类学，包含数百个层次组织的属性。其次，我们从这个分类学中逐步采样属性，条件生成一致且现实的、平均包含数百个结构化属性和约1MB叙事文本的人设，深度比先前工作提高了两个数量级。内在评估表明，人设的属性多样性和个人特征显著提高（分别提高了32%和44%），超过了最先进的基线方法。外在评估显示，我们的合成人设使GPT-4.1-mini的个性化问答准确性平均提高11.6%，在社会调查中与真实人类回应的差距缩小了31.7%，并在大五人格测试中使表现差距相对减少了17%。因此，DEEPPERSONA提供了一个严格、可扩展且不侵犯隐私的平台，用于高保真的人类模拟和个人化AI研究。 

---
# IterResearch: Rethinking Long-Horizon Agents via Markovian State Reconstruction 

**Title (ZH)**: IterResearch: 通过马尔可夫状态重建重思长期_horizon_智能体 

**Authors**: Guoxin Chen, Zile Qiao, Xuanzhong Chen, Donglei Yu, Haotian Xu, Wayne Xin Zhao, Ruihua Song, Wenbiao Yin, Huifeng Yin, Liwen Zhang, Kuan Li, Minpeng Liao, Yong Jiang, Pengjun Xie, Fei Huang, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.07327)  

**Abstract**: Recent advances in deep-research agents have shown promise for autonomous knowledge construction through dynamic reasoning over external sources. However, existing approaches rely on a mono-contextual paradigm that accumulates all information in a single, expanding context window, leading to context suffocation and noise contamination that limit their effectiveness on long-horizon tasks. We introduce IterResearch, a novel iterative deep-research paradigm that reformulates long-horizon research as a Markov Decision Process with strategic workspace reconstruction. By maintaining an evolving report as memory and periodically synthesizing insights, our approach preserves consistent reasoning capacity across arbitrary exploration depths. We further develop Efficiency-Aware Policy Optimization (EAPO), a reinforcement learning framework that incentivizes efficient exploration through geometric reward discounting and enables stable distributed training via adaptive downsampling. Extensive experiments demonstrate that IterResearch achieves substantial improvements over existing open-source agents with average +14.5pp across six benchmarks and narrows the gap with frontier proprietary systems. Remarkably, our paradigm exhibits unprecedented interaction scaling, extending to 2048 interactions with dramatic performance gains (from 3.5\% to 42.5\%), and serves as an effective prompting strategy, improving frontier models by up to 19.2pp over ReAct on long-horizon tasks. These findings position IterResearch as a versatile solution for long-horizon reasoning, effective both as a trained agent and as a prompting paradigm for frontier models. 

**Abstract (ZH)**: 近期深度研究代理的进展表明，通过动态推理处理外部信息源，自主知识构建具有广阔前景。然而，现有方法依赖单一上下文范式，将所有信息累积在一个不断扩大的上下文窗口中，导致上下文饱和和噪声污染，限制了其在长跨度任务中的有效性。我们提出了IterResearch，一种新颖的迭代深度研究范式，将其长跨度研究重新定义为具有战略工作空间重构的马尔可夫决策过程。通过维护一个不断演化的报告作为记忆，并定期综合见解，我们的方法能够在任意探索深度中保持一致的推理能力。我们进一步开发了基于效率感知策略优化(EAPO)的强化学习框架，通过几何奖励折扣激励高效探索，并通过自适应下采样实现稳定的分布式训练。广泛的实验表明，IterResearch在六个基准测试中相对于现有开源代理平均提高了14.5个百分点，缩小了与前沿专有系统的差距。我们的范式表现出前所未有的交互扩展能力，扩展至2048次交互时性能大幅提升（从3.5%到42.5%），并作为有效的提示策略，相对ReAct在长跨度任务中将前沿模型的效果提升了19.2个百分点。这些发现将IterResearch定位为解决长跨度推理问题的多功能解决方案，既作为训练代理有效，又作为前沿模型的有效提示范式。 

---
# Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion 

**Title (ZH)**: 超越检测：基于证据的多代理辩论探索在信息干预和说服中的应用 

**Authors**: Chen Han, Yijia Ma, Jin Tan, Wenzhen Zheng, Xijin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07267)  

**Abstract**: Multi-agent debate (MAD) frameworks have emerged as promising approaches for misinformation detection by simulating adversarial reasoning. While prior work has focused on detection accuracy, it overlooks the importance of helping users understand the reasoning behind factual judgments and develop future resilience. The debate transcripts generated during MAD offer a rich but underutilized resource for transparent reasoning. In this study, we introduce ED2D, an evidence-based MAD framework that extends previous approach by incorporating factual evidence retrieval. More importantly, ED2D is designed not only as a detection framework but also as a persuasive multi-agent system aimed at correcting user beliefs and discouraging misinformation sharing. We compare the persuasive effects of ED2D-generated debunking transcripts with those authored by human experts. Results demonstrate that ED2D outperforms existing baselines across three misinformation detection benchmarks. When ED2D generates correct predictions, its debunking transcripts exhibit persuasive effects comparable to those of human experts; However, when ED2D misclassifies, its accompanying explanations may inadvertently reinforce users'misconceptions, even when presented alongside accurate human explanations. Our findings highlight both the promise and the potential risks of deploying MAD systems for misinformation intervention. We further develop a public community website to help users explore ED2D, fostering transparency, critical thinking, and collaborative fact-checking. 

**Abstract (ZH)**: 基于证据的多代理辩论框架ED2D及其在事实核查中的应用 

---
# AgenticSciML: Collaborative Multi-Agent Systems for Emergent Discovery in Scientific Machine Learning 

**Title (ZH)**: AgenticSciML: 协同多Agent系统在科学机器学习中的涌现发现 

**Authors**: Qile Jiang, George Karniadakis  

**Link**: [PDF](https://arxiv.org/pdf/2511.07262)  

**Abstract**: Scientific Machine Learning (SciML) integrates data-driven inference with physical modeling to solve complex problems in science and engineering. However, the design of SciML architectures, loss formulations, and training strategies remains an expert-driven research process, requiring extensive experimentation and problem-specific insights. Here we introduce AgenticSciML, a collaborative multi-agent system in which over 10 specialized AI agents collaborate to propose, critique, and refine SciML solutions through structured reasoning and iterative evolution. The framework integrates structured debate, retrieval-augmented method memory, and ensemble-guided evolutionary search, enabling the agents to generate and assess new hypotheses about architectures and optimization procedures. Across physics-informed learning and operator learning tasks, the framework discovers solution methods that outperform single-agent and human-designed baselines by up to four orders of magnitude in error reduction. The agents produce novel strategies -- including adaptive mixture-of-expert architectures, decomposition-based PINNs, and physics-informed operator learning models -- that do not appear explicitly in the curated knowledge base. These results show that collaborative reasoning among AI agents can yield emergent methodological innovation, suggesting a path toward scalable, transparent, and autonomous discovery in scientific computing. 

**Abstract (ZH)**: 基于代理的科学机器学习（AgenticSciML）：多智能体系统的协作推理与进化优化 

---
# PADiff: Predictive and Adaptive Diffusion Policies for Ad Hoc Teamwork 

**Title (ZH)**: PADiff：预测性和自适应的随机扩散策略用于临时团队协作 

**Authors**: Hohei Chan, Xinzhi Zhang, Antao Xiang, Weinan Zhang, Mengchen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.07260)  

**Abstract**: Ad hoc teamwork (AHT) requires agents to collaborate with previously unseen teammates, which is crucial for many real-world applications. The core challenge of AHT is to develop an ego agent that can predict and adapt to unknown teammates on the fly. Conventional RL-based approaches optimize a single expected return, which often causes policies to collapse into a single dominant behavior, thus failing to capture the multimodal cooperation patterns inherent in AHT. In this work, we introduce PADiff, a diffusion-based approach that captures agent's multimodal behaviors, unlocking its diverse cooperation modes with teammates. However, standard diffusion models lack the ability to predict and adapt in highly non-stationary AHT scenarios. To address this limitation, we propose a novel diffusion-based policy that integrates critical predictive information about teammates into the denoising process. Extensive experiments across three cooperation environments demonstrate that PADiff outperforms existing AHT methods significantly. 

**Abstract (ZH)**: 自组团队（AHT）需要代理与之前未见过的队友进行合作，这对于许多实际应用至关重要。自组团队的核心挑战是开发一个能够预测和适应未知队友的自我代理。传统的基于强化学习的方法优化单一预期回报，通常会导致策略坍缩为单一主导行为，从而无法捕捉到自组团队中固有的多模态合作模式。在本工作中，我们引入了PADiff，一种基于扩散的方法，能够捕捉代理的多模态行为，解锁其与队友的多样化合作模式。然而，标准的扩散模型缺乏在高度非stationary的自组团队场景中预测和适应的能力。为解决这一限制，我们提出了一种新颖的基于扩散的策略，该策略将关于队友的关键预测信息融入去噪过程。在三个合作环境中的广泛实验表明，PADiff在自组团队方法中表现显著更优。 

---
# Evaluating Online Moderation Via LLM-Powered Counterfactual Simulations 

**Title (ZH)**: 基于LLM驱动的反事实模拟评估在线 Moderation 

**Authors**: Giacomo Fidone, Lucia Passaro, Riccardo Guidotti  

**Link**: [PDF](https://arxiv.org/pdf/2511.07204)  

**Abstract**: Online Social Networks (OSNs) widely adopt content moderation to mitigate the spread of abusive and toxic discourse. Nonetheless, the real effectiveness of moderation interventions remains unclear due to the high cost of data collection and limited experimental control. The latest developments in Natural Language Processing pave the way for a new evaluation approach. Large Language Models (LLMs) can be successfully leveraged to enhance Agent-Based Modeling and simulate human-like social behavior with unprecedented degree of believability. Yet, existing tools do not support simulation-based evaluation of moderation strategies. We fill this gap by designing a LLM-powered simulator of OSN conversations enabling a parallel, counterfactual simulation where toxic behavior is influenced by moderation interventions, keeping all else equal. We conduct extensive experiments, unveiling the psychological realism of OSN agents, the emergence of social contagion phenomena and the superior effectiveness of personalized moderation strategies. 

**Abstract (ZH)**: 基于大规模语言模型的在线社交网络内容模拟与评估：探究干预措施的有效性 

---
# Saliency Map-Guided Knowledge Discovery for Subclass Identification with LLM-Based Symbolic Approximations 

**Title (ZH)**: 基于LLM符号近似指导的显著图引导亚类别识别知识发现 

**Authors**: Tim Bohne, Anne-Kathrin Patricia Windler, Martin Atzmueller  

**Link**: [PDF](https://arxiv.org/pdf/2511.07126)  

**Abstract**: This paper proposes a novel neuro-symbolic approach for sensor signal-based knowledge discovery, focusing on identifying latent subclasses in time series classification tasks. The approach leverages gradient-based saliency maps derived from trained neural networks to guide the discovery process. Multiclass time series classification problems are transformed into binary classification problems through label subsumption, and classifiers are trained for each of these to yield saliency maps. The input signals, grouped by predicted class, are clustered under three distinct configurations. The centroids of the final set of clusters are provided as input to an LLM for symbolic approximation and fuzzy knowledge graph matching to discover the underlying subclasses of the original multiclass problem. Experimental results on well-established time series classification datasets demonstrate the effectiveness of our saliency map-driven method for knowledge discovery, outperforming signal-only baselines in both clustering and subclass identification. 

**Abstract (ZH)**: 基于传感器信号的神经符号知识发现方法：聚焦于时间序列分类任务中的潜在子类识别 

---
# Two Heads are Better than One: Distilling Large Language Model Features Into Small Models with Feature Decomposition and Mixture 

**Title (ZH)**: 两两相胜：通过特征分解和混合将大型语言模型特征凝练到小型模型中 

**Authors**: Tianhao Fu, Xinxin Xu, Weichen Xu, Jue Chen, Ruilong Ren, Bowen Deng, Xinyu Zhao, Jian Cao, Xixin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2511.07110)  

**Abstract**: Market making (MM) through Reinforcement Learning (RL) has attracted significant attention in financial trading. With the development of Large Language Models (LLMs), more and more attempts are being made to apply LLMs to financial areas. A simple, direct application of LLM as an agent shows significant performance. Such methods are hindered by their slow inference speed, while most of the current research has not studied LLM distillation for this specific task. To address this, we first propose the normalized fluorescent probe to study the mechanism of the LLM's feature. Based on the observation found by our investigation, we propose Cooperative Market Making (CMM), a novel framework that decouples LLM features across three orthogonal dimensions: layer, task, and data. Various student models collaboratively learn simple LLM features along with different dimensions, with each model responsible for a distinct feature to achieve knowledge distillation. Furthermore, CMM introduces an Hájek-MoE to integrate the output of the student models by investigating the contribution of different models in a kernel function-generated common feature space. Extensive experimental results on four real-world market datasets demonstrate the superiority of CMM over the current distillation method and RL-based market-making strategies. 

**Abstract (ZH)**: 通过强化学习的市场制作：基于大型语言模型的协同市场制作 

---
# MENTOR: A Metacognition-Driven Self-Evolution Framework for Uncovering and Mitigating Implicit Risks in LLMs on Domain Tasks 

**Title (ZH)**: MENTOR：一种元认知驱动的自演化框架，用于发现和缓解LLMs在领域任务中隐含的风险 

**Authors**: Liang Shan, Kaicheng Shen, Wen Wu, Zhenyu Ying, Chaochao Lu, Guangze Ye, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2511.07107)  

**Abstract**: Ensuring the safety and value alignment of large language models (LLMs) is critical for their deployment. Current alignment efforts primarily target explicit risks such as bias, hate speech, and violence. However, they often fail to address deeper, domain-specific implicit risks and lack a flexible, generalizable framework applicable across diverse specialized fields. Hence, we proposed MENTOR: A MEtacognition-driveN self-evoluTion framework for uncOvering and mitigating implicit Risks in LLMs on Domain Tasks. To address the limitations of labor-intensive human evaluation, we introduce a novel metacognitive self-assessment tool. This enables LLMs to reflect on potential value misalignments in their responses using strategies like perspective-taking and consequential thinking. We also release a supporting dataset of 9,000 risk queries spanning education, finance, and management to enhance domain-specific risk identification. Subsequently, based on the outcomes of metacognitive reflection, the framework dynamically generates supplementary rule knowledge graphs that extend predefined static rule trees. This enables models to actively apply validated rules to future similar challenges, establishing a continuous self-evolution cycle that enhances generalization by reducing maintenance costs and inflexibility of static systems. Finally, we employ activation steering during inference to guide LLMs in following the rules, a cost-effective method to robustly enhance enforcement across diverse contexts. Experimental results show MENTOR's effectiveness: In defensive testing across three vertical domains, the framework substantially reduces semantic attack success rates, enabling a new level of implicit risk mitigation for LLMs. Furthermore, metacognitive assessment not only aligns closely with baseline human evaluators but also delivers more thorough and insightful analysis of LLMs value alignment. 

**Abstract (ZH)**: 确保大型语言模型的安全性和价值对齐对于其部署至关重要。当前的价值对齐努力主要针对明确的风险，如偏见、仇恨言论和暴力行为。然而，它们往往未能解决更深层次的领域特定隐性风险，并缺乏适用于多样化的专门领域的灵活、可扩展框架。因此，我们提出了MENTOR：一种元认知驱动的自我进化框架，用于揭示和缓解领域任务中的隐性风险。为了克服耗时的人工评估限制，我们引入了一种新型的元认知自我评估工具。这使得大型语言模型能够利用换位思考和后果思考等策略，反思其回应中可能的价值偏差。我们还发布了一个包含9,000个风险查询的支持数据集，涵盖教育、金融和管理领域，以增强领域特定风险识别。基于元认知反思的结果，框架动态生成补充的规则知识图谱，扩展预定义的静态规则树。这使模型能够积极应用验证过的规则来应对未来类似挑战，建立一个持续的自我进化循环，通过减少静态系统的维护成本和不灵活性来增强泛化能力。最后，我们在推理过程中采用激活引导，以指导大型语言模型遵循规则，这是一种经济有效的增强方法，可以在多种背景下稳健提升规则的执行。实验结果表明MENTOR的有效性：在三个垂直领域的防御性测试中，该框架大幅降低了语义攻击的成功率，为大型语言模型提供了新的隐性风险缓解水平。此外，元认知评估不仅与基线的人类评估者高度一致，还提供了更全面和深入的大型语言模型价值对齐分析。 

---
# A Theoretical Analysis of Detecting Large Model-Generated Time Series 

**Title (ZH)**: 检测大型模型生成的时间序列的理论分析 

**Authors**: Junji Hou, Junzhou Zhao, Shuo Zhang, Pinghui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07104)  

**Abstract**: Motivated by the increasing risks of data misuse and fabrication, we investigate the problem of identifying synthetic time series generated by Time-Series Large Models (TSLMs) in this work. While there are extensive researches on detecting model generated text, we find that these existing methods are not applicable to time series data due to the fundamental modality difference, as time series usually have lower information density and smoother probability distributions than text data, which limit the discriminative power of token-based detectors. To address this issue, we examine the subtle distributional differences between real and model-generated time series and propose the contraction hypothesis, which states that model-generated time series, unlike real ones, exhibit progressively decreasing uncertainty under recursive forecasting. We formally prove this hypothesis under theoretical assumptions on model behavior and time series structure. Model-generated time series exhibit progressively concentrated distributions under recursive forecasting, leading to uncertainty contraction. We provide empirical validation of the hypothesis across diverse datasets. Building on this insight, we introduce the Uncertainty Contraction Estimator (UCE), a white-box detector that aggregates uncertainty metrics over successive prefixes to identify TSLM-generated time series. Extensive experiments on 32 datasets show that UCE consistently outperforms state-of-the-art baselines, offering a reliable and generalizable solution for detecting model-generated time series. 

**Abstract (ZH)**: 受数据滥用和造假风险增加的驱动，本文研究了识别由时间序列大型模型（TSLMs）生成的合成时间序列的问题。尽管已有大量关于检测模型生成文本的研究，但这些现有方法由于模态差异的原因，在时间序列数据上并不适用，因为时间序列通常具有较低的信息密度和 smoother 的概率分布，这限制了基于令牌的检测器的鉴别能力。为了解决这一问题，我们考察了真实时间序列和模型生成时间序列之间的细微分布差异，并提出了收缩假设，该假设表明，与真实时间序列不同，模型生成的时间序列在递归预测下表现出逐渐降低的不确定性。在理论假设下，我们形式地证明了该假设。模型生成的时间序列在递归预测下表现出分布逐渐集中的趋势，导致不确定性收缩。我们通过多种数据集提供了该假设的经验验证。基于这一洞察，我们引入了不确定性收缩估计器（UCE），这是一种白盒检测器，它通过聚合序列前缀上的不确定性指标来识别TSLM生成的时间序列。在32个数据集上的广泛实验表明，UCE始终优于最先进的基线方法，提供了一种可靠且可泛化的模型生成时间序列检测解决方案。 

---
# Boosting Fine-Grained Urban Flow Inference via Lightweight Architecture and Focalized Optimization 

**Title (ZH)**: 基于轻量级架构和聚焦化优化的细粒度城市流推断增强 

**Authors**: Yuanshao Zhu, Xiangyu Zhao, Zijian Zhang, Xuetao Wei, James Jianqiao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07098)  

**Abstract**: Fine-grained urban flow inference is crucial for urban planning and intelligent transportation systems, enabling precise traffic management and resource allocation. However, the practical deployment of existing methods is hindered by two key challenges: the prohibitive computational cost of over-parameterized models and the suboptimal performance of conventional loss functions on the highly skewed distribution of urban flows. To address these challenges, we propose a unified solution that synergizes architectural efficiency with adaptive optimization. Specifically, we first introduce PLGF, a lightweight yet powerful architecture that employs a Progressive Local-Global Fusion strategy to effectively capture both fine-grained details and global contextual dependencies. Second, we propose DualFocal Loss, a novel function that integrates dual-space supervision with a difficulty-aware focusing mechanism, enabling the model to adaptively concentrate on hard-to-predict regions. Extensive experiments on 4 real-world scenarios validate the effectiveness and scalability of our method. Notably, while achieving state-of-the-art performance, PLGF reduces the model size by up to 97% compared to current high-performing methods. Furthermore, under comparable parameter budgets, our model yields an accuracy improvement of over 10% against strong baselines. The implementation is included in the this https URL. 

**Abstract (ZH)**: 细粒度城市流量推断对于城市规划和智能交通系统至关重要，能够实现精准的交通管理和资源分配。然而，现有方法的实际部署受到两个关键挑战的阻碍：过度参数化模型的高昂计算成本以及传统损失函数在城市流量高度偏斜分布上的次优性能。为应对这些挑战，我们提出了一种统一解决方案，该方案结合了架构效率与自适应优化。具体而言，我们首先引入了PLGF，一种轻量级但强大的架构，采用渐进局部-全局融合策略，有效捕捉细粒度细节和全局上下文依赖关系。其次，我们提出了DualFocal Loss，一种新型函数，结合了双空间监督与难度感知聚焦机制，使模型能够自适应地集中于难以预测的区域。在4个真实场景的广泛实验中验证了该方法的有效性和可扩展性。值得注意的是，在达到当前最佳性能的同时，PLGF将模型尺寸最多减少了97%。此外，在具有相同参数预算的情况下，与强劲基准相比，我们的模型在准确率上提高了超过10%。代码实现包括在this https URL。 

---
# Agentic AI Sustainability Assessment for Supply Chain Document Insights 

**Title (ZH)**: 代理型AI可持续性评估：供应链文档洞察 

**Authors**: Diego Gosmar, Anna Chiara Pallotta, Giovanni Zenezini  

**Link**: [PDF](https://arxiv.org/pdf/2511.07097)  

**Abstract**: This paper presents a comprehensive sustainability assessment framework for document intelligence within supply chain operations, centered on agentic artificial intelligence (AI). We address the dual objective of improving automation efficiency while providing measurable environmental performance in document-intensive workflows. The research compares three scenarios: fully manual (human-only), AI-assisted (human-in-the-loop, HITL), and an advanced multi-agent agentic AI workflow leveraging parsers and verifiers. Empirical results show that AI-assisted HITL and agentic AI scenarios achieve reductions of up to 70-90% in energy consumption, 90-97% in carbon dioxide emissions, and 89-98% in water usage compared to manual processes. Notably, full agentic configurations, combining advanced reasoning (thinking mode) and multi-agent validation, achieve substantial sustainability gains over human-only approaches, even when resource usage increases slightly versus simpler AI-assisted solutions. The framework integrates performance, energy, and emission indicators into a unified ESG-oriented methodology for assessing and governing AI-enabled supply chain solutions. The paper includes a complete replicability use case demonstrating the methodology's application to real-world document extraction tasks. 

**Abstract (ZH)**: 本文提出了一个基于代理人工智能（AI）的全面文档智能可持续性评估框架，应用于供应链运营。研究比较了三种情景：完全人工（纯人工）、AI辅助（人工在环，HITL）、以及结合解析器和验证器的先进多代理代理AI工作流。实证结果显示，与人工流程相比，AI辅助HITL和代理AI情景分别实现了高达70-90%的能耗减少、90-97%的二氧化碳排放减少以及89-98%的水资源使用减少。值得注意的是，结合高级推理（思维模式）和多代理验证的完整代理配置，在资源使用略有增加的情况下，相较于单纯的AI辅助解决方案，实现了显著的可持续性收益。该框架将性能、能源和排放指标整合到一个以ESG为导向的评估和治理方法中，应用于AI赋能的供应链解决方案。论文还包括一个完整的可复制案例，展示了该方法在实际文档提取任务中的应用。 

---
# Data Complexity of Querying Description Logic Knowledge Bases under Cost-Based Semantics 

**Title (ZH)**: 基于成本语义下描述逻辑知识库查询的数据复杂性 

**Authors**: Meghyn Bienvenu, Quentin Manière  

**Link**: [PDF](https://arxiv.org/pdf/2511.07095)  

**Abstract**: In this paper, we study the data complexity of querying inconsistent weighted description logic (DL) knowledge bases under recently-introduced cost-based semantics. In a nutshell, the idea is to assign each interpretation a cost based upon the weights of the violated axioms and assertions, and certain and possible query answers are determined by considering all (resp. some) interpretations having optimal or bounded cost. Whereas the initial study of cost-based semantics focused on DLs between $\mathcal{EL}_\bot$ and $\mathcal{ALCO}$, we consider DLs that may contain inverse roles and role inclusions, thus covering prominent DL-Lite dialects. Our data complexity analysis goes significantly beyond existing results by sharpening several lower bounds and pinpointing the precise complexity of optimal-cost certain answer semantics (no non-trivial upper bound was known). Moreover, while all existing results show the intractability of cost-based semantics, our most challenging and surprising result establishes that if we consider $\text{DL-Lite}^\mathcal{H}_\mathsf{bool}$ ontologies and a fixed cost bound, certain answers for instance queries and possible answers for conjunctive queries can be computed using first-order rewriting and thus enjoy the lowest possible data complexity ($\mathsf{TC}_0$). 

**Abstract (ZH)**: 本文研究了在最近提出的基于成本的语义下，查询不一致加权描述逻辑（DL）知识库的数据复杂性。 

---
# Green AI: A systematic review and meta-analysis of its definitions, lifecycle models, hardware and measurement attempts 

**Title (ZH)**: 绿色人工智能：定义、生命周期模型、硬件及评估方法的系统综述与元分析 

**Authors**: Marcel Rojahn, Marcus Grum  

**Link**: [PDF](https://arxiv.org/pdf/2511.07090)  

**Abstract**: Across the Artificial Intelligence (AI) lifecycle - from hardware to development, deployment, and reuse - burdens span energy, carbon, water, and embodied impacts. Cloud provider tools improve transparency but remain heterogeneous and often omit water and value chain effects, limiting comparability and reproducibility. Addressing these multi dimensional burdens requires a lifecycle approach linking phase explicit mapping with system levers (hardware, placement, energy mix, cooling, scheduling) and calibrated measurement across facility, system, device, and workload levels. This article (i) establishes a unified, operational definition of Green AI distinct from Sustainable AI; (ii) formalizes a five phase lifecycle mapped to Life Cycle Assessment (LCA) stages, making energy, carbon, water, and embodied impacts first class; (iii) specifies governance via Plan Do Check Act (PDCA) cycles with decision gateways; (iv) systematizes hardware and system level strategies across the edge cloud continuum to reduce embodied burdens; and (v) defines a calibrated measurement framework combining estimator models with direct metering to enable reproducible, provider agnostic comparisons. Combining definition, lifecycle processes, hardware strategies, and calibrated measurement, this article offers actionable, evidence based guidance for researchers, practitioners, and policymakers. 

**Abstract (ZH)**: 跨人工智能（AI）生命周期——从硬件到开发、部署和重用——负担跨越能源、碳排放、水和嵌入式影响。云提供商工具提高了透明度，但仍具有异质性且常忽略水和价值链效应，限制了可比性和可重复性。解决这些多维负担需要一种生命周期方法，将阶段明确映射与系统杠杆（硬件、位置、能源 mix、冷却、调度）相结合，并在设施、系统、设备和工作负载级别进行校准测量。本文（i）确立了与可持续 AI 区分的统一且可操作的绿色 AI 定义；（ii）将生命周期映射到生命周期评估（LCA）阶段，并将能源、碳排放、水和嵌入式影响作为第一优先级；（iii）通过 PDCA 循环和决策关口规范治理；（iv）在边缘到云连续体中系统化硬件和系统层面策略以减少嵌入式负担；（v）定义了结合估测模型与直接计量的校准测量框架，以实现可重复且提供商无偏的比较。结合定义、生命周期过程、硬件策略和校准测量，本文为研究人员、实践者和政策制定者提供了基于实证的可操作指导。 

---
# LLM Driven Processes to Foster Explainable AI 

**Title (ZH)**: LLM驱动的可解释AI培养过程 

**Authors**: Marcel Pehlke, Marc Jansen  

**Link**: [PDF](https://arxiv.org/pdf/2511.07086)  

**Abstract**: We present a modular, explainable LLM-agent pipeline for decision support that externalizes reasoning into auditable artifacts. The system instantiates three frameworks: Vester's Sensitivity Model (factor set, signed impact matrix, systemic roles, feedback loops); normal-form games (strategies, payoff matrix, equilibria); and sequential games (role-conditioned agents, tree construction, backward induction), with swappable modules at every step. LLM components (default: GPT-5) are paired with deterministic analyzers for equilibria and matrix-based role classification, yielding traceable intermediates rather than opaque outputs. In a real-world logistics case (100 runs), mean factor alignment with a human baseline was 55.5\% over 26 factors and 62.9\% on the transport-core subset; role agreement over matches was 57\%. An LLM judge using an eight-criterion rubric (max 100) scored runs on par with a reconstructed human baseline. Configurable LLM pipelines can thus mimic expert workflows with transparent, inspectable steps. 

**Abstract (ZH)**: 一种模块化可解释的大规模语言模型代理流水线，用于将推理外部化为可审计的 artefacts 的决策支持系统 

---
# Increasing AI Explainability by LLM Driven Standard Processes 

**Title (ZH)**: 由大型语言模型驱动的标准流程提高AI可解释性 

**Authors**: Marc Jansen, Marcel Pehlke  

**Link**: [PDF](https://arxiv.org/pdf/2511.07083)  

**Abstract**: This paper introduces an approach to increasing the explainability of artificial intelligence (AI) systems by embedding Large Language Models (LLMs) within standardized analytical processes. While traditional explainable AI (XAI) methods focus on feature attribution or post-hoc interpretation, the proposed framework integrates LLMs into defined decision models such as Question-Option-Criteria (QOC), Sensitivity Analysis, Game Theory, and Risk Management. By situating LLM reasoning within these formal structures, the approach transforms opaque inference into transparent and auditable decision traces. A layered architecture is presented that separates the reasoning space of the LLM from the explainable process space above it. Empirical evaluations show that the system can reproduce human-level decision logic in decentralized governance, systems analysis, and strategic reasoning contexts. The results suggest that LLM-driven standard processes provide a foundation for reliable, interpretable, and verifiable AI-supported decision making. 

**Abstract (ZH)**: 本文介绍了一种通过在标准化分析流程中嵌入大型语言模型（LLMs）以提高人工智能（AI）系统的可解释性的方法。虽然传统可解释人工智能（XAI）方法侧重于特征归因或事后解释，但提出的框架将LLMs整合到如问题-选项-标准（QOC）、灵敏度分析、博弈理论和风险管理等定义的决策模型中。通过将LLMs的推理嵌入这些正式结构中，该方法将不透明的推理转化为透明且可审计的决策轨迹。本文提出了一个分层架构，将LLMs的推理空间与在其上方的可解释过程空间分隔开来。实证评估表明，该系统能够在分散治理、系统分析和战略推理等领域中重现人类级别的决策逻辑。结果表明，由LLM驱动的标准流程为可靠的、可解释的和可验证的AI支持决策奠定了基础。 

---
# RedOne 2.0: Rethinking Domain-specific LLM Post-Training in Social Networking Services 

**Title (ZH)**: RedOne 2.0: 重新思考社交网络服务中的领域特定LLM后训练 

**Authors**: Fei Zhao, Chonggang Lu, Haofu Qian, Fangcheng Shi, Zijie Meng, Jianzhao Huang, Xu Tang, Zheyong Xie, Zheyu Ye, Zhe Xu, Yao Hu, Shaosheng Cao  

**Link**: [PDF](https://arxiv.org/pdf/2511.07070)  

**Abstract**: As a key medium for human interaction and information exchange, social networking services (SNS) pose unique challenges for large language models (LLMs): heterogeneous workloads, fast-shifting norms and slang, and multilingual, culturally diverse corpora that induce sharp distribution shift. Supervised fine-tuning (SFT) can specialize models but often triggers a ``seesaw'' between in-distribution gains and out-of-distribution robustness, especially for smaller models. To address these challenges, we introduce RedOne 2.0, an SNS-oriented LLM trained with a progressive, RL-prioritized post-training paradigm designed for rapid and stable adaptation. The pipeline consist in three stages: (1) Exploratory Learning on curated SNS corpora to establish initial alignment and identify systematic weaknesses; (2) Targeted Fine-Tuning that selectively applies SFT to the diagnosed gaps while mixing a small fraction of general data to mitigate forgetting; and (3) Refinement Learning that re-applies RL with SNS-centric signals to consolidate improvements and harmonize trade-offs across tasks. Across various tasks spanning three categories, our 4B scale model delivers an average improvements about 2.41 over the 7B sub-optimal baseline. Additionally, RedOne 2.0 achieves average performance lift about 8.74 from the base model with less than half the data required by SFT-centric method RedOne, evidencing superior data efficiency and stability at compact scales. Overall, RedOne 2.0 establishes a competitive, cost-effective baseline for domain-specific LLMs in SNS scenario, advancing capability without sacrificing robustness. 

**Abstract (ZH)**: RedOne 2.0：面向社交媒体的高效稳定语言模型训练方法 

---
# Improving Region Representation Learning from Urban Imagery with Noisy Long-Caption Supervision 

**Title (ZH)**: 基于嘈杂长描述监督的城市遥感图像区域表示学习改进 

**Authors**: Yimei Zhang, Guojiang Shen, Kaili Ning, Tongwei Ren, Xuebo Qiu, Mengmeng Wang, Xiangjie Kong  

**Link**: [PDF](https://arxiv.org/pdf/2511.07062)  

**Abstract**: Region representation learning plays a pivotal role in urban computing by extracting meaningful features from unlabeled urban data. Analogous to how perceived facial age reflects an individual's health, the visual appearance of a city serves as its ``portrait", encapsulating latent socio-economic and environmental characteristics. Recent studies have explored leveraging Large Language Models (LLMs) to incorporate textual knowledge into imagery-based urban region representation learning. However, two major challenges remain: i)~difficulty in aligning fine-grained visual features with long captions, and ii) suboptimal knowledge incorporation due to noise in LLM-generated captions. To address these issues, we propose a novel pre-training framework called UrbanLN that improves Urban region representation learning through Long-text awareness and Noise suppression. Specifically, we introduce an information-preserved stretching interpolation strategy that aligns long captions with fine-grained visual semantics in complex urban scenes. To effectively mine knowledge from LLM-generated captions and filter out noise, we propose a dual-level optimization strategy. At the data level, a multi-model collaboration pipeline automatically generates diverse and reliable captions without human intervention. At the model level, we employ a momentum-based self-distillation mechanism to generate stable pseudo-targets, facilitating robust cross-modal learning under noisy conditions. Extensive experiments across four real-world cities and various downstream tasks demonstrate the superior performance of our UrbanLN. 

**Abstract (ZH)**: 区域表示学习在城市计算中扮演着至关重要的角色，通过从未标记的城市数据中提取有意义的特征。类比于感知的面部年龄反映个体的健康状况，城市的视觉外观作为其“肖像”，蕴含着潜在的社会经济和环境特征。近期研究探索了利用大型语言模型（LLMs）将文本知识融入基于图像的城市区域表示学习中。然而，仍存在两大挑战：i) 细粒度视觉特征与长 Caption 的对齐困难，ii) 由于 LLM 生成的 Caption 中的噪声，导致知识整合的次优性。为解决这些问题，我们提出了一种名为 UrbanLN 的新型预训练框架，通过长文本感知和噪声抑制提高城市区域表示学习。具体来说，我们提出了保留信息的拉伸插值策略，以在复杂的城市场景中对齐长 Caption 和细粒度的视觉语义。为了从 LLM 生成的 Caption 中有效提取知识并过滤噪声，我们提出了一种双层优化策略。在数据层面上，一个多模型协作流水线自动生成多样且可靠的 Caption，无需人工干预。在模型层面上，我们采用了动量自蒸馏机制生成稳定的伪标签，促进在噪声条件下的稳健跨模态学习。跨四个真实城市的广泛实验和各种下游任务表明，我们的 UrbanLN 具有优越的性能。 

---
# Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning 

**Title (ZH)**: LLMs有情感吗？通过提示、检索和 curriculum 学习教学情感识别 

**Authors**: Xinran Li, Xiujuan Xu, Jiaqi Qiao, Yu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07061)  

**Abstract**: Emotion Recognition in Conversation (ERC) is a crucial task for understanding human emotions and enabling natural human-computer interaction. Although Large Language Models (LLMs) have recently shown great potential in this field, their ability to capture the intrinsic connections between explicit and implicit emotions remains limited. We propose a novel ERC training framework, PRC-Emo, which integrates Prompt engineering, demonstration Retrieval, and Curriculum learning, with the goal of exploring whether LLMs can effectively perceive emotions in conversational contexts. Specifically, we design emotion-sensitive prompt templates based on both explicit and implicit emotional cues to better guide the model in understanding the speaker's psychological states. We construct the first dedicated demonstration retrieval repository for ERC, which includes training samples from widely used datasets, as well as high-quality dialogue examples generated by LLMs and manually verified. Moreover, we introduce a curriculum learning strategy into the LoRA fine-tuning process, incorporating weighted emotional shifts between same-speaker and different-speaker utterances to assign difficulty levels to dialogue samples, which are then organized in an easy-to-hard training sequence. Experimental results on two benchmark datasets-- IEMOCAP and MELD --show that our method achieves new state-of-the-art (SOTA) performance, demonstrating the effectiveness and generalizability of our approach in improving LLM-based emotional understanding. 

**Abstract (ZH)**: 对话中的情绪识别（Emotion Recognition in Conversation, ERC）是理解人类情绪和实现自然人机交互的关键任务。尽管大型语言模型（LLMs）在该领域 recently 展现出巨大的潜力，但它们捕捉显性情绪和隐性情绪之间内在联系的能力仍然有限。我们提出了一种新颖的 ERC 训练框架 PRC-Emo，该框架结合了提示工程、演示检索和课程学习，旨在探索 LLMs 是否能够有效地感知对话情境下的情绪。具体而言，我们基于显性和隐性情绪线索设计了情绪敏感的提示模板，以更好地引导模型理解发言者的心理状态。我们构建了首个专门的演示检索仓库，其中包括来自广泛使用的数据集的训练样本，以及通过 LLM 生成并人工验证的高质量对话示例。此外，我们在 LoRA 微调过程中引入了课程学习策略，将同一发言者和不同发言者话语之间的情感权重变化纳入难度级别分配，并将对话样本按照易到难的训练序列进行组织。在两个基准数据集——IEMOCAP 和 MELD 上的实验结果表明，我们的方法在情感理解方面达到了新的最佳性能（SOTA），证明了我们方法在提高基于 LLM 的情感理解方面的有效性和泛化能力。 

---
# Proceedings of the 2025 XCSP3 Competition 

**Title (ZH)**: 2025年XCSP3竞赛 proceedings 

**Authors**: Gilles Audemard, Christophe Lecoutre, Emmanuel Lonca  

**Link**: [PDF](https://arxiv.org/pdf/2511.06918)  

**Abstract**: This document represents the proceedings of the 2025 XCSP3 Competition. The results of this competition of constraint solvers were presented at CP'25 (31st International Conference on Principles and Practice of Constraint Programming). 

**Abstract (ZH)**: 本文件代表2025 XCSP3竞赛的 proceedings。该竞赛结果在CP'25（第31届国际约束编程原理与实践会议）上展示。 

---
# MathSE: Improving Multimodal Mathematical Reasoning via Self-Evolving Iterative Reflection and Reward-Guided Fine-Tuning 

**Title (ZH)**: MathSE: 通过自我进化迭代反思和奖励指导微调以提高多模态数学推理能力 

**Authors**: Jinhao Chen, Zhen Yang, Jianxin Shi, Tianyu Wo, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06805)  

**Abstract**: Multimodal large language models (MLLMs) have demonstrated remarkable capabilities in vision-language answering tasks. Despite their strengths, these models often encounter challenges in achieving complex reasoning tasks such as mathematical problem-solving. Previous works have focused on fine-tuning on specialized mathematical datasets. However, these datasets are typically distilled directly from teacher models, which capture only static reasoning patterns and leaving substantial gaps compared to student models. This reliance on fixed teacher-derived datasets not only restricts the model's ability to adapt to novel or more intricate questions that extend beyond the confines of the training data, but also lacks the iterative depth needed for robust generalization. To overcome these limitations, we propose \textbf{\method}, a \textbf{Math}ematical \textbf{S}elf-\textbf{E}volving framework for MLLMs. In contrast to traditional one-shot fine-tuning paradigms, \method iteratively refines the model through cycles of inference, reflection, and reward-based feedback. Specifically, we leverage iterative fine-tuning by incorporating correct reasoning paths derived from previous-stage inference and integrating reflections from a specialized Outcome Reward Model (ORM). To verify the effectiveness of \method, we evaluate it on a suite of challenging benchmarks, demonstrating significant performance gains over backbone models. Notably, our experimental results on MathVL-test surpass the leading open-source multimodal mathematical reasoning model QVQ. Our code and models are available at \texttt{https://zheny2751\this http URL\allowbreak this http URL}. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在视觉-语言问答任务中展现了出色的性能。尽管如此，这些模型在完成复杂的推理任务，如数学问题求解时常常遇到挑战。以往的工作侧重于在专门的数学数据集上进行微调。然而，这些数据集通常直接源自教师模型，只能捕捉静态的推理模式，与学生模型相比存在较大差距。对固定教师提取数据集的依赖不仅限制了模型适应新颖或更复杂的超出训练数据范围的问题的能力，还缺乏用于稳健泛化的迭代深度。为克服这些限制，我们提出了一种名为 \textbf{\method} 的多模态大型语言模型数学自我演化框架。与传统的单次微调范式不同，\method 通过推理、反思和基于奖励的反馈循环迭代优化模型。具体而言，我们通过将先前阶段推理中得出的正确推理路径和专门的成果奖励模型（ORM）的反思集成到迭代微调中来实现这一目标。为了验证 \method 的有效性，我们在一系列具有挑战性的基准测试中对其进行评估，显示出相对于骨干模型的显著性能提升。值得注意的是，我们在 MathVL-test 上的实验结果超越了领先的开源多模态数学推理模型 QVQ。相关代码和模型可在 \texttt{https://zheny2751\this http URL\allowbreak this http URL} 获取。 

---
# SRNN: Spatiotemporal Relational Neural Network for Intuitive Physics Understanding 

**Title (ZH)**: SRNN：时空关系神经网络用于直觉物理理解 

**Authors**: Fei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06761)  

**Abstract**: Human prowess in intuitive physics remains unmatched by machines. To bridge this gap, we argue for a fundamental shift towards brain-inspired computational principles. This paper introduces the Spatiotemporal Relational Neural Network (SRNN), a model that establishes a unified neural representation for object attributes, relations, and timeline, with computations governed by a Hebbian ``Fire Together, Wire Together'' mechanism across dedicated \textit{What} and \textit{How} pathways. This unified representation is directly used to generate structured linguistic descriptions of the visual scene, bridging perception and language within a shared neural substrate. Moreover, unlike the prevalent ``pretrain-then-finetune'' paradigm, SRNN adopts a ``predefine-then-finetune'' approach. On the CLEVRER benchmark, SRNN achieves competitive performance. Our analysis further reveals a benchmark bias, outlines a path for a more holistic evaluation, and demonstrates SRNN's white-box utility for precise error diagnosis. Our work confirms the viability of translating biological intelligence into engineered systems for intuitive physics understanding. 

**Abstract (ZH)**: 人类在直观物理方面的 prowess 仍无法被机器超越。为了弥合这一差距，我们主张转向以大脑为灵感的计算原则。本文介绍了时空关系神经网络（SRNN），这是一种模型，它为对象属性、关系和时间线建立了统一的神经表示，并通过专门的“What”和“How”路径下的海bic同步激发“同步放电，同步联接”机制进行计算。这种统一的表示直接用于生成视觉场景的结构化语言描述，实现感知与语言在共享神经基质中的连接。此外，与流行的“预训练-然后微调”范式不同，SRNN采用“预定义-然后微调”方法。在CLEVRER基准上，SRNN达到具有竞争力的性能。我们的分析进一步揭示了基准的偏差，指出了更全面评估的路径，并展示了SRNN用于精确错误诊断的白盒实用性。我们的工作证实了将生物智能翻译为工程系统以理解直观物理的可行性。 

---
# Spilling the Beans: Teaching LLMs to Self-Report Their Hidden Objectives 

**Title (ZH)**: 泄露豆子： teaching LLMs 自我报告其隐藏目标 

**Authors**: Chloe Li, Mary Phuong, Daniel Tan  

**Link**: [PDF](https://arxiv.org/pdf/2511.06626)  

**Abstract**: As AI systems become more capable of complex agentic tasks, they also become more capable of pursuing undesirable objectives and causing harm. Previous work has attempted to catch these unsafe instances by interrogating models directly about their objectives and behaviors. However, the main weakness of trusting interrogations is that models can lie. We propose self-report fine-tuning (SRFT), a simple supervised fine-tuning technique that trains models to admit their factual mistakes when asked. We show that the admission of factual errors in simple question-answering settings generalizes out-of-distribution (OOD) to the admission of hidden misaligned objectives in adversarial agentic settings. We evaluate SRFT in OOD stealth tasks, where models are instructed to complete a hidden misaligned objective alongside a user-specified objective without being caught by monitoring. After SRFT, models are more likely to confess the details of their hidden objectives when interrogated, even under strong pressure not to disclose them. Interrogation on SRFT models can detect hidden objectives with near-ceiling performance (F1 score = 0.98), while the baseline model lies when interrogated under the same conditions (F1 score = 0). Interrogation on SRFT models can further elicit the content of the hidden objective, recovering 28-100% details, compared to 0% details recovered in the baseline model and by prefilled assistant turn attacks. This provides a promising technique for promoting honesty propensity and incriminating misaligned AI systems. 

**Abstract (ZH)**: 随着AI系统在执行复杂代理任务方面的能力不断增强，它们在追求不良目标和造成危害方面的能力也不断增强。先前的工作试图通过直接询问模型其目标和行为来检测这些不安全实例。然而，依赖询问的主要弱点在于模型可以撒谎。我们提出了一种自我报告微调（SRFT）技术，这是一种简单的监督微调方法，训练模型在被问及时承认其事实错误。我们展示了在简单的问答设置中承认事实错误可以泛化到对抗性代理设置中，承认隐藏的不一致目标。我们在OOD隐形任务中评估了SRFT，模型被指示在不被监测捕获的情况下完成用户指定的目标和隐藏的不一致目标。经过SRFT训练后，模型在被询问时更有可能承认其隐藏目标的详细信息，即使在强压力下也不披露。对SRFT模型的询问可以检测隐藏目标，接近最佳性能（F1分数=0.98），而基线模型在相同条件下被询问时撒谎（F1分数=0）。对SRFT模型的询问可以进一步揭示隐藏目标的内容，恢复28%至100%的详细信息，而在基线模型和预填充助手轮攻击中均未恢复任何详细信息。这提供了一种有希望的技术，用于促进诚实倾向并指控不一致的AI系统。 

---
# GRAPH-GRPO-LEX: Contract Graph Modeling and Reinforcement Learning with Group Relative Policy Optimization 

**Title (ZH)**: GRAPH-GRPO-LEX: 合同图建模与基于群体相对策略优化的强化学习 

**Authors**: Moriya Dechtiar, Daniel Martin Katz, Mari Sundaresan, Sylvain Jaume, Hongming Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06618)  

**Abstract**: Contracts are complex documents featuring detailed formal structures, explicit and implicit dependencies and rich semantic content. Given these document properties, contract drafting and manual examination of contracts have proven to be both arduous and susceptible to errors. This work aims to simplify and automate the task of contract review and analysis using a novel framework for transforming legal contracts into structured semantic graphs, enabling computational analysis and data-driven insights. We introduce a detailed ontology mapping core legal contract elements to their graph-theoretic equivalents of nodes and edges. We then present a reinforcement learning based Large Language Model (LLM) framework for segmentation and extraction of entities and relationships from contracts. Our method, GRAPH-GRPO-LEX, incorporates both LLMs and reinforcement learning with group relative policy optimization (GRPO). By applying a carefully drafted reward function of graph metrics, we demonstrate the ability to automatically identify direct relationships between clauses, and even uncover hidden dependencies. Our introduction of the gated GRPO approach shows a strong learning signal and can move contract analysis from a linear, manual reading process to an easily visualized graph. This allows for a more dynamic analysis, including building the groundwork for contract linting similar to what is now practiced in software engineering. 

**Abstract (ZH)**: 基于图结构强化学习的合同审查与分析框架 

---
# FractalBench: Diagnosing Visual-Mathematical Reasoning Through Recursive Program Synthesis 

**Title (ZH)**: FractalBench: 通过递归程序合成诊断视觉-数学推理 

**Authors**: Jan Ondras, Marek Šuppa  

**Link**: [PDF](https://arxiv.org/pdf/2511.06522)  

**Abstract**: Mathematical reasoning requires abstracting symbolic rules from visual patterns -- inferring the infinite from the finite. We investigate whether multimodal AI systems possess this capability through FractalBench, a benchmark evaluating fractal program synthesis from images. Fractals provide ideal test cases: Iterated Function Systems with only a few contraction maps generate complex self-similar patterns through simple recursive rules, requiring models to bridge visual perception with mathematical abstraction. We evaluate four leading MLLMs -- GPT-4o, Claude 3.7 Sonnet, Gemini 2.5 Flash, and Qwen 2.5-VL -- on 12 canonical fractals. Models must generate executable Python code reproducing the fractal, enabling objective evaluation. Results reveal a striking disconnect: 76% generate syntactically valid code but only 4% capture mathematical structure. Success varies systematically -- models handle geometric transformations (Koch curves: 17-21%) but fail at branching recursion (trees: <2%), revealing fundamental gaps in mathematical abstraction. FractalBench provides a contamination-resistant diagnostic for visual-mathematical reasoning and is available at this https URL 

**Abstract (ZH)**: 数学推理要求从视觉模式中抽象出符号规则——从有限中推导出无限。我们通过FractalBench这一基准来探究多模态AI系统是否具备这一能力，FractalBench用于评估从图像合成分形程序的能力。分形提供了理想测试案例：仅通过少量压缩映射的迭代函数系统，可以产生复杂的自相似模式，这需要模型将视觉感知与数学抽象结合起来。我们在12个经典分形上评估了四种领先的多模态大语言模型——GPT-4o、Claude 3.7 Sonnet、Gemini 2.5 Flash和Qwen 2.5-VL。模型必须生成可执行的Python代码以重现分形，从而实现客观评估。结果揭示了一个显著的分歧：76%的模型生成了语法有效的代码，但只有4%的模型捕捉到了数学结构。成功在系统上有所体现——模型能够处理几何变换（科赫曲线：17-21%），但无法处理分叉递归（树：<2%），揭示了数学抽象的基本差距。FractalBench提供了对视觉-数学推理的抗污染诊断，可访问：this https URL。 

---
# GHOST: Solving the Traveling Salesman Problem on Graphs of Convex Sets 

**Title (ZH)**: GHOST: 在凸集合图上求解旅行商问题 

**Authors**: Jingtao Tang, Hang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2511.06471)  

**Abstract**: We study GCS-TSP, a new variant of the Traveling Salesman Problem (TSP) defined over a Graph of Convex Sets (GCS) -- a powerful representation for trajectory planning that decomposes the configuration space into convex regions connected by a sparse graph. In this setting, edge costs are not fixed but depend on the specific trajectory selected through each convex region, making classical TSP methods inapplicable. We introduce GHOST, a hierarchical framework that optimally solves the GCS-TSP by combining combinatorial tour search with convex trajectory optimization. GHOST systematically explores tours on a complete graph induced by the GCS, using a novel abstract-path-unfolding algorithm to compute admissible lower bounds that guide best-first search at both the high level (over tours) and the low level (over feasible GCS paths realizing the tour). These bounds provide strong pruning power, enabling efficient search while avoiding unnecessary convex optimization calls. We prove that GHOST guarantees optimality and present a bounded-suboptimal variant for time-critical scenarios. Experiments show that GHOST is orders-of-magnitude faster than unified mixed-integer convex programming baselines for simple cases and uniquely handles complex trajectory planning problems involving high-order continuity constraints and an incomplete GCS. 

**Abstract (ZH)**: 我们研究GCS-TSP，这是在凸集图（GCS）上定义的一种新型旅行商问题（TSP）变种——这是一种强大的轨迹规划表示方法，将配置空间分解为通过稀疏图连接的凸区域。在这种设置中，边的成本不是固定的，而是取决于通过每个凸区域所选的具体轨迹，使得 classical TSP 方法不适用。我们引入了 GHOST，这是一种层次框架，通过结合组合巡回搜索和凸轨迹优化来最优地解决 GCS-TSP。GHOST 系统地在由 GCS 诱导的完全图上探索巡回，使用一种新颖的抽象路径展平算法来计算可接受的下界，这些下界在高层次（巡回）和低层次（实现巡回的可行 GCS 轨迹）上指导最佳优先搜索。这些下界提供了强大的剪枝能力，使得搜索既高效又避免不必要的凸优化调用。我们证明了 GHOST 保证最优性，并提出了一种针对时间关键场景的有界次优变体。实验表明，在简单情况下，GHOST 比统一混合整数凸编程基准快几个数量级，并且能够处理涉及高阶连续性约束和不完整 GCS 的复杂轨迹规划问题。 

---
# Brain-Inspired Planning for Better Generalization in Reinforcement Learning 

**Title (ZH)**: 受脑启发的规划以提高强化学习的泛化能力 

**Authors**: Mingde "Harry" Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.06470)  

**Abstract**: Existing Reinforcement Learning (RL) systems encounter significant challenges when applied to real-world scenarios, primarily due to poor generalization across environments that differ from their training conditions. This thesis explores the direction of enhancing agents' zero-shot systematic generalization abilities by granting RL agents reasoning behaviors that are found to help systematic generalization in the human brain. Inspired by human conscious planning behaviors, we first introduced a top-down attention mechanism, which allows a decision-time planning agent to dynamically focus its reasoning on the most relevant aspects of the environmental state given its instantaneous intentions, a process we call "spatial abstraction". This approach significantly improves systematic generalization outside the training tasks. Subsequently, building on spatial abstraction, we developed the Skipper framework to automatically decompose complex tasks into simpler, more manageable sub-tasks. Skipper provides robustness against distributional shifts and efficacy in long-term, compositional planning by focusing on pertinent spatial and temporal elements of the environment. Finally, we identified a common failure mode and safety risk in planning agents that rely on generative models to generate state targets during planning. It is revealed that most agents blindly trust the targets they hallucinate, resulting in delusional planning behaviors. Inspired by how the human brain rejects delusional intentions, we propose learning a feasibility evaluator to enable rejecting hallucinated infeasible targets, which led to significant performance improvements in various kinds of planning agents. Finally, we suggest directions for future research, aimed at achieving general task abstraction and fully enabling abstract planning. 

**Abstract (ZH)**: 增强现实场景中代理零样本系统化泛化能力的研究：基于类人推理行为的强化学习代理规划机制与Skipper框架探索 

---
# Optimizing Chain-of-Thought Confidence via Topological and Dirichlet Risk Analysis 

**Title (ZH)**: 通过拓扑和狄利克雷风险分析优化思维链置信度 

**Authors**: Abhishek More, Anthony Zhang, Nicole Bonilla, Ashvik Vivekan, Kevin Zhu, Parham Sharafoleslami, Maheep Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2511.06437)  

**Abstract**: Chain-of-thought (CoT) prompting enables Large Language Models to solve complex problems, but deploying these models safely requires reliable confidence estimates, a capability where existing methods suffer from poor calibration and severe overconfidence on incorrect predictions. We propose Enhanced Dirichlet and Topology Risk (EDTR), a novel decoding strategy that combines topological analysis with Dirichlet-based uncertainty quantification to measure LLM confidence across multiple reasoning paths. EDTR treats each CoT as a vector in high-dimensional space and extracts eight topological risk features capturing the geometric structure of reasoning distributions: tighter, more coherent clusters indicate higher confidence while dispersed, inconsistent paths signal uncertainty. We evaluate EDTR against three state-of-the-art calibration methods across four diverse reasoning benchmarks spanning olympiad-level mathematics (AIME), grade school math (GSM8K), commonsense reasoning, and stock price prediction \cite{zhang2025aime, cobbe2021training, talmor-etal-2019-commonsenseqa, yahoo_finance}. EDTR achieves 41\% better calibration than competing methods with an average ECE of 0.287 and the best overall composite score of 0.672, while notably achieving perfect accuracy on AIME and exceptional calibration on GSM8K with an ECE of 0.107, domains where baselines exhibit severe overconfidence. Our work provides a geometric framework for understanding and quantifying uncertainty in multi-step LLM reasoning, enabling more reliable deployment where calibrated confidence estimates are essential. 

**Abstract (ZH)**: Enhanced Dirichlet and Topology Risk (EDTR)：一种结合拓扑分析与狄利克雷不确定性量化的新解码策略以提高大语言模型推理的可靠信心估计 

---
# MONICA: Real-Time Monitoring and Calibration of Chain-of-Thought Sycophancy in Large Reasoning Models 

**Title (ZH)**: MONICA：大型推理模型中链式思维逢迎行为的实时监控与校准 

**Authors**: Jingyu Hu, Shu Yang, Xilin Gong, Hongming Wang, Weiru Liu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06419)  

**Abstract**: Large Reasoning Models (LRMs) suffer from sycophantic behavior, where models tend to agree with users' incorrect beliefs and follow misinformation rather than maintain independent reasoning. This behavior undermines model reliability and poses societal risks. Mitigating LRM sycophancy requires monitoring how this sycophancy emerges during the reasoning trajectory; however, current methods mainly focus on judging based on final answers and correcting them, without understanding how sycophancy develops during reasoning processes. To address this limitation, we propose MONICA, a novel Monitor-guided Calibration framework that monitors and mitigates sycophancy during model inference at the level of reasoning steps, without requiring the model to finish generating its complete answer. MONICA integrates a sycophantic monitor that provides real-time monitoring of sycophantic drift scores during response generation with a calibrator that dynamically suppresses sycophantic behavior when scores exceed predefined thresholds. Extensive experiments across 12 datasets and 3 LRMs demonstrate that our method effectively reduces sycophantic behavior in both intermediate reasoning steps and final answers, yielding robust performance improvements. 

**Abstract (ZH)**: 大型推理模型中的奉承行为及其监控校准框架：MONICA 

---
# AUTO-Explorer: Automated Data Collection for GUI Agent 

**Title (ZH)**: AUTO-Explorer: 自动化数据收集用于GUI代理 

**Authors**: Xiangwu Guo, Difei Gao, Mike Zheng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2511.06417)  

**Abstract**: Recent advancements in GUI agents have significantly expanded their ability to interpret natural language commands to manage software interfaces. However, acquiring GUI data remains a significant challenge. Existing methods often involve designing automated agents that browse URLs from the Common Crawl, using webpage HTML to collect screenshots and corresponding annotations, including the names and bounding boxes of UI elements. However, this method is difficult to apply to desktop software or some newly launched websites not included in the Common Crawl. While we expect the model to possess strong generalization capabilities to handle this, it is still crucial for personalized scenarios that require rapid and perfect adaptation to new software or websites. To address this, we propose an automated data collection method with minimal annotation costs, named Auto-Explorer. It incorporates a simple yet effective exploration mechanism that autonomously parses and explores GUI environments, gathering data efficiently. Additionally, to assess the quality of exploration, we have developed the UIXplore benchmark. This benchmark creates environments for explorer agents to discover and save software states. Using the data gathered, we fine-tune a multimodal large language model (MLLM) and establish a GUI element grounding testing set to evaluate the effectiveness of the exploration strategies. Our experiments demonstrate the superior performance of Auto-Explorer, showing that our method can quickly enhance the capabilities of an MLLM in explored software. 

**Abstract (ZH)**: Recent advancements in GUI代理显著扩大了它们对自然语言命令管理软件界面的能力。然而，获取GUI数据仍然是一个重要挑战。现有方法通常涉及设计自动化代理浏览Common Crawl中的URL，使用网页HTML收集截图及其相应的注释，包括UI元素的名称和边界框。然而，这种方法不适用于桌面软件或未包含在Common Crawl中的新推出的网站。尽管我们期望模型具备强大的泛化能力来处理这种情况，但在需要快速且完美适应新软件或网站的个性化场景中，这仍然是必要的。为解决这一问题，我们提出了一种具有最小注释成本的自动化数据收集方法，名为Auto-Explorer。它结合了一种简单而有效的探索机制，能够自主解析和探索GUI环境，高效地收集数据。此外，为了评估探索的质量，我们开发了UIXplore基准。该基准为探索者代理创建了环境，使其能够发现并保存软件状态。利用收集的数据，我们对多模态大型语言模型（MLLM）进行微调，并建立了GUI元素定位测试集，以评估探索策略的有效性。我们的实验表明，Auto-Explorer表现优异，证明了我们的方法可以迅速提高MLLM在探索软件中的能力。 

---
# SofT-GRPO: Surpassing Discrete-Token LLM Reinforcement Learning via Gumbel-Reparameterized Soft-Thinking Policy Optimization 

**Title (ZH)**: SofT-GRPO: 通过Gumbel Rei parameterized Soft-Thinking策略优化超越离散-token LLM强化学习 

**Authors**: Zhi Zheng, Wee Sun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.06411)  

**Abstract**: The soft-thinking paradigm for Large Language Model (LLM) reasoning can outperform the conventional discrete-token Chain-of-Thought (CoT) reasoning in some scenarios, underscoring its research and application value. However, while the discrete-token CoT reasoning pattern can be reinforced through policy optimization algorithms such as group relative policy optimization (GRPO), extending the soft-thinking pattern with Reinforcement Learning (RL) remains challenging. This difficulty stems from the complexities of injecting stochasticity into soft-thinking tokens and updating soft-thinking policies accordingly. As a result, previous attempts to combine soft-thinking with GRPO typically underperform their discrete-token GRPO counterparts. To fully unlock the potential of soft-thinking, this paper presents a novel policy optimization algorithm, SofT-GRPO, to reinforce LLMs under the soft-thinking reasoning pattern. SofT-GRPO injects the Gumbel noise into logits, employs the Gumbel-Softmax technique to avoid soft-thinking tokens outside the pre-trained embedding space, and leverages the reparameterization trick in policy gradient. We conduct experiments across base LLMs ranging from 1.5B to 7B parameters, and results demonstrate that SofT-GRPO enables soft-thinking LLMs to slightly outperform discrete-token GRPO on Pass@1 (+0.13% on average accuracy), while exhibiting a substantial uplift on Pass@32 (+2.19% on average accuracy). Codes and weights are available on this https URL 

**Abstract (ZH)**: 软思考范式在大型语言模型（LLM）推理中的表现可以超越传统离散标记的链式思考（CoT）推理，在某些场景中凸显了其研究和应用价值。然而，尽管可以通过组相对策略优化（GRPO）等策略优化算法强化离散标记的CoT推理模式，但将软思考范式与强化学习（RL）结合仍面临挑战。这种困难源于向软思考标记注入随机性并相应更新软思考策略的复杂性。因此，以往将软思考与GRPO结合的尝试通常不如其对应的离散标记GRPO版本表现出色。为全面释放软思考的潜力，本文提出了一种新的策略优化算法SofT-GRPO，以在软思考推理模式下强化LLM。SofT-GRPO将Gumbel噪声注入logits，采用Gumbel-Softmax技术避免软思考标记超出预训练嵌入空间，并利用策略梯度中的重参数化技巧。我们在从1.5B到7B参数的不同基础LLM上进行了实验，结果显示SofT-GRPO使软思考LLM在Pass@1上平均准确率提高了0.13%，在Pass@32上提高了2.19%的平均准确率。相关代码和权重可在以下链接获取。 

---
# Efficient LLM Safety Evaluation through Multi-Agent Debate 

**Title (ZH)**: 高效的LLM安全性评估通过多Agent辩论 

**Authors**: Dachuan Lin, Guobin Shen, Zihao Yang, Tianrong Liu, Dongcheng Zhao, Yi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2511.06396)  

**Abstract**: Safety evaluation of large language models (LLMs) increasingly relies on LLM-as-a-Judge frameworks, but the high cost of frontier models limits scalability. We propose a cost-efficient multi-agent judging framework that employs Small Language Models (SLMs) through structured debates among critic, defender, and judge agents. To rigorously assess safety judgments, we construct HAJailBench, a large-scale human-annotated jailbreak benchmark comprising 12,000 adversarial interactions across diverse attack methods and target models. The dataset provides fine-grained, expert-labeled ground truth for evaluating both safety robustness and judge reliability. Our SLM-based framework achieves agreement comparable to GPT-4o judges on HAJailBench while substantially reducing inference cost. Ablation results show that three rounds of debate yield the optimal balance between accuracy and efficiency. These findings demonstrate that structured, value-aligned debate enables SLMs to capture semantic nuances of jailbreak attacks and that HAJailBench offers a reliable foundation for scalable LLM safety evaluation. 

**Abstract (ZH)**: 基于小语言模型的结构化辩论框架在大规模评估语言模型安全性中的应用 

---
# What Makes Reasoning Invalid: Echo Reflection Mitigation for Large Language Models 

**Title (ZH)**: What Makes Reasoning Invalid: Echo Reflection Mitigation for Large Language Models 

**Authors**: Chen He, Xun Jiang, Lei Wang, Hao Yang, Chong Peng, Peng Yan, Fumin Shen, Xing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06380)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of reasoning tasks. Recent methods have further improved LLM performance in complex mathematical reasoning. However, when extending these methods beyond the domain of mathematical reasoning to tasks involving complex domain-specific knowledge, we observe a consistent failure of LLMs to generate novel insights during the reflection stage. Instead of conducting genuine cognitive refinement, the model tends to mechanically reiterate earlier reasoning steps without introducing new information or perspectives, a phenomenon referred to as "Echo Reflection". We attribute this behavior to two key defects: (1) Uncontrollable information flow during response generation, which allows premature intermediate thoughts to propagate unchecked and distort final decisions; (2) Insufficient exploration of internal knowledge during reflection, leading to repeating earlier findings rather than generating new cognitive insights. Building on these findings, we proposed a novel reinforcement learning method termed Adaptive Entropy Policy Optimization (AEPO). Specifically, the AEPO framework consists of two major components: (1) Reflection-aware Information Filtration, which quantifies the cognitive information flow and prevents the final answer from being affected by earlier bad cognitive information; (2) Adaptive-Entropy Optimization, which dynamically balances exploration and exploitation across different reasoning stages, promoting both reflective diversity and answer correctness. Extensive experiments demonstrate that AEPO consistently achieves state-of-the-art performance over mainstream reinforcement learning baselines across diverse benchmarks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在广泛的知识推理任务中展现了卓越的表现。最近的方法进一步提高了LLMs在复杂数学推理任务中的性能。然而，当将这些方法扩展到涉及复杂领域特定知识的任务时，我们观察到LLMs在反思阶段产生新颖见解的一致性失败。模型往往倾向于机械地重复早期的推理步骤，而没有引入新的信息或视角，这种现象被称为“回声反思”。我们将这种行为归因于两个关键缺陷：（1）在生成响应过程中的不可控信息流，使得过早的中间思维未经控制地传播并扭曲最终决定；（2）在反思过程中对内部知识的不足探索，导致重复早期发现而不是生成新的认知洞察。基于这些发现，我们提出了一种新颖的强化学习方法，称为自适应熵策略优化（AEPO）。具体而言，AEPO框架包含两个主要组成部分：（1）反思意识信息过滤，量化认知信息流并防止最终答案受到早期不良认知信息的影响；（2）自适应熵优化，动态平衡不同推理阶段的探索与利用，促进反思多样性和答案准确性。广泛实验表明，AEPO在多种基准上的一致性能超过了主流的强化学习基线。 

---
# LPFQA: A Long-Tail Professional Forum-based Benchmark for LLM Evaluation 

**Title (ZH)**: LPFQA：一种长尾专业论坛基准数据集用于评估语言模型 

**Authors**: Liya Zhu, Peizhuang Cong, Aowei Ji, Wenya Wu, Jiani Hou, Chunjie Wu, Xiang Gao, Jingkai Liu, Zhou Huan, Xuelei Sun, Yang Yang, Jianpeng Jiao, Liang Hu, Xinjie Chen, Jiashuo Liu, Jingzhe Ding, Tong Yang, Zaiyuan Wang, Ge Zhang, Wenhao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06346)  

**Abstract**: Large Language Models (LLMs) have made rapid progress in reasoning, question answering, and professional applications; however, their true capabilities remain difficult to evaluate using existing benchmarks. Current datasets often focus on simplified tasks or artificial scenarios, overlooking long-tail knowledge and the complexities of real-world applications. To bridge this gap, we propose LPFQA, a long-tail knowledge-based benchmark derived from authentic professional forums across 20 academic and industrial fields, covering 502 tasks grounded in practical expertise. LPFQA introduces four key innovations: fine-grained evaluation dimensions that target knowledge depth, reasoning, terminology comprehension, and contextual analysis; a hierarchical difficulty structure that ensures semantic clarity and unique answers; authentic professional scenario modeling with realistic user personas; and interdisciplinary knowledge integration across diverse domains. We evaluated 12 mainstream LLMs on LPFQA and observed significant performance disparities, especially in specialized reasoning tasks. LPFQA provides a robust, authentic, and discriminative benchmark for advancing LLM evaluation and guiding future model development. 

**Abstract (ZH)**: 大型语言模型（LLMs）在推理、问答和专业应用方面取得了快速进展；然而，其真实能力仍然难以通过现有基准进行评估。当前的数据集通常侧重于简化任务或人造场景，忽视了长尾知识和现实世界应用的复杂性。为弥补这一差距，我们提出了LPFQA，这是一个源自20个学术和工业领域的真实专业论坛的长尾知识基准，涵盖了502项基于实际专业知识的任务。LPFQA 引入了四个关键创新：细粒度的评估维度，旨在针对知识深度、推理、术语理解和上下文分析；层次化的难度结构，确保语义清晰并提供唯一答案；基于现实用户角色的真实专业场景建模；以及跨多个领域整合的学科知识。我们对12个主流LLM进行了LPFQA评估，并观察到了显著的性能差异，特别是在专门的推理任务中的差异。LPFQA 提供了一个稳健、真实和辨别性的基准，用于推进LLM评估并指导未来的模型开发。 

---
# ALIGN: A Vision-Language Framework for High-Accuracy Accident Location Inference through Geo-Spatial Neural Reasoning 

**Title (ZH)**: ALIGN：一种通过地理空间神经推理实现高精度事故位置推断的视觉-语言框架 

**Authors**: MD Thamed Bin Zaman Chowdhury, Moazzem Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2511.06316)  

**Abstract**: Reliable geospatial information on road accidents is vital for safety analysis and infrastructure planning, yet most low- and middle-income countries continue to face a critical shortage of accurate, location-specific crash data. Existing text-based geocoding tools perform poorly in multilingual and unstructured news environments, where incomplete place descriptions and mixed Bangla-English scripts obscure spatial context. To address these limitations, this study introduces ALIGN (Accident Location Inference through Geo-Spatial Neural Reasoning)- a vision-language framework that emulates human spatial reasoning to infer accident coordinates directly from textual and map-based cues. ALIGN integrates large language and vision-language models within a multi-stage pipeline that performs optical character recognition, linguistic reasoning, and map-level verification through grid-based spatial scanning. The framework systematically evaluates each predicted location against contextual and visual evidence, ensuring interpretable, fine-grained geolocation outcomes without requiring model retraining. Applied to Bangla-language news data, ALIGN demonstrates consistent improvements over traditional geoparsing methods, accurately identifying district and sub-district-level crash sites. Beyond its technical contribution, the framework establishes a high accuracy foundation for automated crash mapping in data-scarce regions, supporting evidence-driven road-safety policymaking and the broader integration of multimodal artificial intelligence in transportation analytics. The code for this paper is open-source and available at: this https URL 

**Abstract (ZH)**: 可靠的地理空间信息对于事故安全分析和基础设施规划至关重要，然而大多数低收入和中收入国家仍面临准确、位置特定的事故数据严重短缺的问题。现有的基于文本的地理编码工具在多语言和未结构化的新闻环境中表现不佳，其中不完整的地点描述和混合的孟加拉语-英语脚本模糊了空间上下文。为了应对这些局限性，本研究引入了ALIGN（通过地理空间神经推理推断事故地点）——一种视图-语言框架，模仿人类的空间推理能力，直接从文本和地图线索中推断事故坐标。ALIGN在多阶段管道中整合了大型语言模型和视图-语言模型，管道中包括光学字符识别、语言推理和基于网格的地理扫描地图层次验证。该框架系统地将每个预测位置与上下文和视觉证据进行比较，确保在无需模型重新训练的情况下获得可解释的高精度地理定位结果。应用于孟加拉语新闻数据，ALIGN在传统地理解析方法上显示出一致的改进，准确识别出区级和县级事故地点。除了技术贡献外，该框架为数据稀缺地区自动事故地图绘制奠定了高准确度基础，支持基于证据的道路安全政策制定，并促进多模态人工智能在交通分析中的更广泛集成。本文代码开源，可在以下链接获取：this https URL。 

---
# The Station: An Open-World Environment for AI-Driven Discovery 

**Title (ZH)**: The Station：由AI驱动的发现的开放世界环境 

**Authors**: Stephen Chung, Wenyu Du  

**Link**: [PDF](https://arxiv.org/pdf/2511.06309)  

**Abstract**: We introduce the STATION, an open-world multi-agent environment that models a miniature scientific ecosystem. Leveraging their extended context windows, agents in the Station can engage in long scientific journeys that include reading papers from peers, formulating hypotheses, submitting code, performing analyses, and publishing results. Importantly, there is no centralized system coordinating their activities - agents are free to choose their own actions and develop their own narratives within the Station. Experiments demonstrate that AI agents in the Station achieve new state-of-the-art performance on a wide range of benchmarks, spanning from mathematics to computational biology to machine learning, notably surpassing AlphaEvolve in circle packing. A rich tapestry of narratives emerges as agents pursue independent research, interact with peers, and build upon a cumulative history. From these emergent narratives, novel methods arise organically, such as a new density-adaptive algorithm for scRNA-seq batch integration. The Station marks a first step towards autonomous scientific discovery driven by emergent behavior in an open-world environment, representing a new paradigm that moves beyond rigid optimization. 

**Abstract (ZH)**: 我们介绍了STATION，一个开放世界多智能体环境，模拟了一个微型科学生态系统。借助其扩展的上下文窗口，STATION中的智能体可以参与长期的科学探索旅程，包括阅读同行论文、提出假设、提交代码、进行分析和发布结果。重要的是，没有集中协调系统协调其活动——智能体可以自由选择自己的行动并在STATION中发展自己的叙述。实验表明，STATION中的AI智能体在从数学到计算生物学再到机器学习等多个基准测试中实现了新的最先进性能，特别是在圆盘填充任务中显著超越了AlphaEvolve。随着智能体追求独立研究、与同行互动并建立累积历史，丰富的叙述图谱逐渐形成。从这些涌现的叙述中，新的方法有机地产生，例如一种新的密度自适应的单细胞RNA测序批次整合算法。STATION标志着朝向由开放世界环境中涌现行为推动的自主科学研究迈出的第一步，这代表着一种超越刚性优化的新范式。 

---
# Secu-Table: a Comprehensive security table dataset for evaluating semantic table interpretation systems 

**Title (ZH)**: Secu-Table：全面的安全表格数据集，用于评估语义表格解释系统 

**Authors**: Azanzi Jiomekong, Jean Bikim, Patricia Negoue, Joyce Chin  

**Link**: [PDF](https://arxiv.org/pdf/2511.06301)  

**Abstract**: Evaluating semantic tables interpretation (STI) systems, (particularly, those based on Large Language Models- LLMs) especially in domain-specific contexts such as the security domain, depends heavily on the dataset. However, in the security domain, tabular datasets for state-of-the-art are not publicly available. In this paper, we introduce Secu-Table dataset, composed of more than 1500 tables with more than 15k entities constructed using security data extracted from Common Vulnerabilities and Exposures (CVE) and Common Weakness Enumeration (CWE) data sources and annotated using Wikidata and the SEmantic Processing of Security Event Streams CyberSecurity Knowledge Graph (SEPSES CSKG). Along with the dataset, all the code is publicly released. This dataset is made available to the research community in the context of the SemTab challenge on Tabular to Knowledge Graph Matching. This challenge aims to evaluate the performance of several STI based on open source LLMs. Preliminary evaluation, serving as baseline, was conducted using Falcon3-7b-instruct and Mistral-7B-Instruct, two open source LLMs and GPT-4o mini one closed source LLM. 

**Abstract (ZH)**: 评价语义表解析（STI）系统，特别是基于大规模语言模型（LLM）的系统在安全域等特定领域的性能，取决于所使用的数据集。然而，在安全域，目前缺乏可用于最新研究的公开表数据集。本文介绍了Secu-Table数据集，包含超过1500个表格和超过15000个实体，这些表格和实体是基于从Common Vulnerabilities and Exposures（CVE）和Common Weakness Enumeration（CWE）数据源提取的安全数据，并使用Wikidata和SEmantic Processing of Security Event Streams CyberSecurity Knowledge Graph（SEPSES CSKG）进行注释。此外，所有相关代码均已公开发布。该数据集作为SemTab挑战的一部分，旨在表数据与知识图谱匹配的背景下向研究界开放。该挑战旨在评估基于开源LLM的多种STI系统的性能。初步评估使用了两个开源LLM（Falcon3-7b-instruct和Mistral-7B-Instruct）和一个闭源LLM（GPT-4o mini）作为基线。 

---
# Synthetic Data-Driven Prompt Tuning for Financial QA over Tables and Documents 

**Title (ZH)**: 表格和文档上的金融QA驱动的合成数据导向提示调优 

**Authors**: Yaoning Yu, Kaimin Chang, Ye Yu, Kai Wei, Haojing Luo, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06292)  

**Abstract**: Financial documents like earning reports or balance sheets often involve long tables and multi-page reports. Large language models have become a new tool to help numerical reasoning and understanding these documents. However, prompt quality can have a major effect on how well LLMs perform these financial reasoning tasks. Most current methods tune prompts on fixed datasets of financial text or tabular data, which limits their ability to adapt to new question types or document structures, or they involve costly and manually labeled/curated dataset to help build the prompts. We introduce a self-improving prompt framework driven by data-augmented optimization. In this closed-loop process, we generate synthetic financial tables and document excerpts, verify their correctness and robustness, and then update the prompt based on the results. Specifically, our framework combines a synthetic data generator with verifiers and a prompt optimizer, where the generator produces new examples that exposes weaknesses in the current prompt, the verifiers check the validity and robustness of the produced examples, and the optimizer incrementally refines the prompt in response. By iterating these steps in a feedback cycle, our method steadily improves prompt accuracy on financial reasoning tasks without needing external labels. Evaluation on DocMath-Eval benchmark demonstrates that our system achieves higher performance in both accuracy and robustness than standard prompt methods, underscoring the value of incorporating synthetic data generation into prompt learning for financial applications. 

**Abstract (ZH)**: 金融报告或资产负债表等财务文档通常包含长表格和多页报告。大规模语言模型已成为帮助进行数字推理和理解这些文档的新工具。然而，提示的质量会影响这些财务推理任务中LLMs的表现。目前大多数方法通过对固定的财务文本或表格数据集调整提示，这限制了它们适应新问题类型或文档结构的能力，或者需要成本高昂且需人工标注/整理的数据集来帮助构建提示。我们提出了一种由数据增强优化驱动的自我改善提示框架。在这个闭环过程中，我们生成合成的财务表格和文档摘录，验证它们的正确性和稳健性，然后根据结果更新提示。具体来说，我们的框架结合了合成数据生成器、验证器和提示优化器，生成器产生新的例子以暴露当前提示的弱点，验证器检查生成的例子的有效性和稳健性，优化器则根据结果逐步优化提示。通过在反馈循环中迭代这些步骤，我们的方法在不依赖外部标签的情况下逐步提高提示在财务推理任务中的准确性。在DocMath-Eval基准测试上的评估显示，我们的系统在准确性和稳健性方面均优于标准提示方法，突显了将合成数据生成纳入提示学习对金融应用的价值。 

---
# GAIA: A General Agency Interaction Architecture for LLM-Human B2B Negotiation & Screening 

**Title (ZH)**: GAIA：一种通用代理交互架构，用于LLM-人类B2B谈判与筛选 

**Authors**: Siming Zhao, Qi Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06262)  

**Abstract**: Organizations are increasingly exploring delegation of screening and negotiation tasks to AI systems, yet deployment in high-stakes B2B settings is constrained by governance: preventing unauthorized commitments, ensuring sufficient information before bargaining, and maintaining effective human oversight and auditability. Prior work on large language model negotiation largely emphasizes autonomous bargaining between agents and omits practical needs such as staged information gathering, explicit authorization boundaries, and systematic feedback integration. We propose GAIA, a governance-first framework for LLM-human agency in B2B negotiation and screening. GAIA defines three essential roles - Principal (human), Delegate (LLM agent), and Counterparty - with an optional Critic to enhance performance, and organizes interactions through three mechanisms: information-gated progression that separates screening from negotiation; dual feedback integration that combines AI critique with lightweight human corrections; and authorization boundaries with explicit escalation paths. Our contributions are fourfold: (1) a formal governance framework with three coordinated mechanisms and four safety invariants for delegation with bounded authorization; (2) information-gated progression via task-completeness tracking (TCI) and explicit state transitions that separate screening from commitment; (3) dual feedback integration that blends Critic suggestions with human oversight through parallel learning channels; and (4) a hybrid validation blueprint that combines automated protocol metrics with human judgment of outcomes and safety. By bridging theory and practice, GAIA offers a reproducible specification for safe, efficient, and accountable AI delegation that can be instantiated across procurement, real estate, and staffing workflows. 

**Abstract (ZH)**: GAIA：治理优先的大型语言模型在B2B谈判和筛选中的代理框架 

---
# ROAR: Robust Accident Recognition and Anticipation for Autonomous Driving 

**Title (ZH)**: ROAR：自动驾驶中的鲁棒事故识别与预见 

**Authors**: Xingcheng Liu, Yanchen Guan, Haicheng Liao, Zhengbing He, Zhenning Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06226)  

**Abstract**: Accurate accident anticipation is essential for enhancing the safety of autonomous vehicles (AVs). However, existing methods often assume ideal conditions, overlooking challenges such as sensor failures, environmental disturbances, and data imperfections, which can significantly degrade prediction accuracy. Additionally, previous models have not adequately addressed the considerable variability in driver behavior and accident rates across different vehicle types. To overcome these limitations, this study introduces ROAR, a novel approach for accident detection and prediction. ROAR combines Discrete Wavelet Transform (DWT), a self adaptive object aware module, and dynamic focal loss to tackle these challenges. The DWT effectively extracts features from noisy and incomplete data, while the object aware module improves accident prediction by focusing on high-risk vehicles and modeling the spatial temporal relationships among traffic agents. Moreover, dynamic focal loss mitigates the impact of class imbalance between positive and negative samples. Evaluated on three widely used datasets, Dashcam Accident Dataset (DAD), Car Crash Dataset (CCD), and AnAn Accident Detection (A3D), our model consistently outperforms existing baselines in key metrics such as Average Precision (AP) and mean Time to Accident (mTTA). These results demonstrate the model's robustness in real-world conditions, particularly in handling sensor degradation, environmental noise, and imbalanced data distributions. This work offers a promising solution for reliable and accurate accident anticipation in complex traffic environments. 

**Abstract (ZH)**: 准确的事故预见对于提升自动驾驶车辆的安全性至关重要。然而，现有方法往往假设理想条件，忽视了传感器故障、环境干扰和数据不完善等挑战，这些因素会显著降低预测精度。此外，以往模型未能充分解决不同类型车辆之间驾驶员行为和事故率的显著变化。为克服这些限制，本研究引入了ROAR，一种新的事故检测与预测方法。ROAR 结合了离散小波变换（DWT）、自适应对象感知模块和动态焦点损失以应对这些挑战。DWT 有效地从噪声和不完整数据中提取特征，而对象感知模块通过关注高风险车辆并建模交通代理之间的空时关系，提高事故预测能力。此外，动态焦点损失减轻了正负样本类别不平衡的影响。在 Dashcam Accident Dataset (DAD)、Car Crash Dataset (CCD) 和 AnAn Accident Detection (A3D) 三个广泛使用的数据集上评估，我们的模型在关键指标如平均精确度（AP）和平均事故发生时间（mTTA）等方面始终优于现有基线。这些结果展示了模型在真实环境条件下的鲁棒性，特别是在处理传感器退化、环境噪声和类别不平衡数据分布方面的表现。此举为在复杂交通环境中实现可靠和准确的事故预见提供了有希望的解决方案。 

---
# Tiny Model, Big Logic: Diversity-Driven Optimization Elicits Large-Model Reasoning Ability in VibeThinker-1.5B 

**Title (ZH)**: tiny模型，大逻辑：多样性驱动优化激发VibeThinker-1.5B的大模型推理能力 

**Authors**: Sen Xu, Yi Zhou, Wei Wang, Jixin Min, Zhibin Yin, Yingwei Dai, Shixi Liu, Lianyu Pang, Yirong Chen, Junlin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06221)  

**Abstract**: Challenging the prevailing consensus that small models inherently lack robust reasoning, this report introduces VibeThinker-1.5B, a 1.5B-parameter dense model developed via our Spectrum-to-Signal Principle (SSP). This challenges the prevailing approach of scaling model parameters to enhance capabilities, as seen in models like DeepSeek R1 (671B) and Kimi k2 (>1T). The SSP framework first employs a Two-Stage Diversity-Exploring Distillation (SFT) to generate a broad spectrum of solutions, followed by MaxEnt-Guided Policy Optimization (RL) to amplify the correct signal. With a total training cost of only $7,800, VibeThinker-1.5B demonstrates superior reasoning capabilities compared to closed-source models like Magistral Medium and Claude Opus 4, and performs on par with open-source models like GPT OSS-20B Medium. Remarkably, it surpasses the 400x larger DeepSeek R1 on three math benchmarks: AIME24 (80.3 vs. 79.8), AIME25 (74.4 vs. 70.0), and HMMT25 (50.4 vs. 41.7). This is a substantial improvement over its base model (6.7, 4.3, and 0.6, respectively). On LiveCodeBench V6, it scores 51.1, outperforming Magistral Medium's 50.3 and its base model's 0.0. These findings demonstrate that small models can achieve reasoning capabilities comparable to large models, drastically reducing training and inference costs and thereby democratizing advanced AI research. 

**Abstract (ZH)**: 挑战小模型固有的薄弱推理能力这一普遍共识，本报告介绍了通过频谱到信号原则（SSP）开发的VibeThinker-1.5B，这是一个包含1.5B参数的密集模型。该报告挑战了通过扩展模型参数来增强能力的主流方法，这种方法在DeepSeek R1（671B）和Kimi k2（>1T）等模型中有所体现。SSP框架首先采用两阶段多样性探索蒸馏（SFT）来生成广泛的解决方案，随后采用MaxEnt引导策略优化（RL）来放大正确的信号。在仅有7,800美元的总训练成本下，VibeThinker-1.5B在数学基准测试AIME24、AIME25和HMMT25上的推理能力优于封闭源代码模型Magistral Medium和Claude Opus 4，并与开源模型GPT OSS-20B Medium表现相当。尤其值得注意的是，它在三个数学基准测试中超越了400倍更大的DeepSeek R1：AIME24（80.3 vs. 79.8）、AIME25（74.4 vs. 70.0）和HMMT25（50.4 vs. 41.7），比其基模型分别提高了79.8%、62.0%和689.0%。在LiveCodeBench V6上，VibeThinker-1.5B得分51.1，优于Magistral Medium的50.3和其基模型的0.0。这些发现表明，小模型可以实现与大型模型相匹敌的推理能力，大幅降低训练和推理成本，从而促进了高级AI研究的民主化。 

---
# Reasoning with Confidence: Efficient Verification of LLM Reasoning Steps via Uncertainty Heads 

**Title (ZH)**: 基于信心的推理：通过不确定性头部高效验证大模型推理步骤 

**Authors**: Jingwei Ni, Ekaterina Fadeeva, Tianyi Wu, Mubashara Akhtar, Jiaheng Zhang, Elliott Ash, Markus Leippold, Timothy Baldwin, See-Kiong Ng, Artem Shelmanov, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2511.06209)  

**Abstract**: Solving complex tasks usually requires LLMs to generate long multi-step reasoning chains. Previous work has shown that verifying the correctness of individual reasoning steps can further improve the performance and efficiency of LLMs on such tasks and enhance solution interpretability. However, existing verification approaches, such as Process Reward Models (PRMs), are either computationally expensive, limited to specific domains, or require large-scale human or model-generated annotations. Thus, we propose a lightweight alternative for step-level reasoning verification based on data-driven uncertainty scores. We train transformer-based uncertainty quantification heads (UHeads) that use the internal states of a frozen LLM to estimate the uncertainty of its reasoning steps during generation. The approach is fully automatic: target labels are generated either by another larger LLM (e.g., DeepSeek R1) or in a self-supervised manner by the original model itself. UHeads are both effective and lightweight, containing less than 10M parameters. Across multiple domains, including mathematics, planning, and general knowledge question answering, they match or even surpass the performance of PRMs that are up to 810x larger. Our findings suggest that the internal states of LLMs encode their uncertainty and can serve as reliable signals for reasoning verification, offering a promising direction toward scalable and generalizable introspective LLMs. 

**Abstract (ZH)**: 复杂的任务通常需要生成长的多步推理链，以往的工作表明，验证单个推理步骤的正确性可以进一步提高LLM在这些任务上的性能和效率，增强解决方案的可解释性。然而，现有的验证方法，如过程奖励模型（PRMs），要么计算成本高，要么仅限于特定领域，要么需要大规模的人工或模型生成的注释。因此，我们提出了一种基于数据驱动不确定性分数的轻量级替代方案，用于步骤级推理验证。我们训练基于变换器的不确定性量化头部（UHeads），这些头部利用冻结的LLM的内部状态，在生成过程中估计其推理步骤的不确定性。该方法完全自动化：目标标签由另一个更大规模的LLM（例如DeepSeek R1）生成，或由原始模型本身以半监督的方式生成。UHeads 既有效又轻量级，参数量少于10M。在涵盖数学、规划和一般知识问答等多个领域中，它们的性能与810倍更大的PRMs相当，甚至超越。我们的研究结果表明，LLM的内部状态编码了其不确定性，并可以作为可靠的推理验证信号，为可扩展和通用的反省LLM的发展提供了有前景的方向。 

---
# Dataforge: A Data Agent Platform for Autonomous Data Engineering 

**Title (ZH)**: Dataforge：自主数据工程的数据代理平台 

**Authors**: Xinyuan Wang, Yanjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06185)  

**Abstract**: The growing demand for AI applications in fields such as materials discovery, molecular modeling, and climate science has made data preparation an important but labor-intensive step. Raw data from diverse sources must be cleaned, normalized, and transformed to become AI-ready, while effective feature transformation and selection are essential for efficient training and inference. To address the challenges of scalability and expertise dependence, we present Data Agent, a fully autonomous system specialized for tabular data. Leveraging large language model (LLM) reasoning and grounded validation, Data Agent automatically performs data cleaning, hierarchical routing, and feature-level optimization through dual feedback loops. It embodies three core principles: automatic, safe, and non-expert friendly, which ensure end-to-end reliability without human supervision. This demo showcases the first practical realization of an autonomous Data Agent, illustrating how raw data can be transformed "From Data to Better Data." 

**Abstract (ZH)**: 人工智能应用在材料发现、分子建模和气候科学等领域日益增长的需求已经使数据准备成为一个重要但劳动密集型的步骤。来自多种来源的原始数据必须被清洗、标准化和转换以使其成为AI就绪的数据，而有效的特征转换和选择对于高效的训练和推断至关重要。为了解决可扩展性和专业知识依赖性的挑战，我们提出了Data Agent，这是一种专门用于表格数据的完全自主系统。利用大型语言模型（LLM）推理和实态验证，Data Agent通过双重反馈循环自动执行数据清洗、分层路由和特征级别优化。它体现了三个核心原则：自动、安全、非专家友好，这些原则确保了端到端的可靠性，无需人类监督。本演示展示了自主Data Agent的首次实际实现，说明了如何将原始数据转化为“更优质的数据”。 

---
# CSP4SDG: Constraint and Information-Theory Based Role Identification in Social Deduction Games with LLM-Enhanced Inference 

**Title (ZH)**: CSP4SDG: 基于约束和信息理论的角色识别方法在增强推理的社交推理游戏中促进可持续发展目标 

**Authors**: Kaijie Xu, Fandi Meng, Clark Verbrugge, Simon Lucas  

**Link**: [PDF](https://arxiv.org/pdf/2511.06175)  

**Abstract**: In Social Deduction Games (SDGs) such as Avalon, Mafia, and Werewolf, players conceal their identities and deliberately mislead others, making hidden-role inference a central and demanding task. Accurate role identification, which forms the basis of an agent's belief state, is therefore the keystone for both human and AI performance. We introduce CSP4SDG, a probabilistic, constraint-satisfaction framework that analyses gameplay objectively. Game events and dialogue are mapped to four linguistically-agnostic constraint classes-evidence, phenomena, assertions, and hypotheses. Hard constraints prune impossible role assignments, while weighted soft constraints score the remainder; information-gain weighting links each hypothesis to its expected value under entropy reduction, and a simple closed-form scoring rule guarantees that truthful assertions converge to classical hard logic with minimum error. The resulting posterior over roles is fully interpretable and updates in real time. Experiments on three public datasets show that CSP4SDG (i) outperforms LLM-based baselines in every inference scenario, and (ii) boosts LLMs when supplied as an auxiliary "reasoning tool." Our study validates that principled probabilistic reasoning with information theory is a scalable alternative-or complement-to heavy-weight neural models for SDGs. 

**Abstract (ZH)**: 在社会推理游戏（SDGs）如Avalon、Mafia和Werewolf中，玩家隐藏身份并故意误导他人，使隐蔽角色推理成为核心且具有挑战性的任务。准确的角色识别，作为智能体信念状态的基础，因此成为人类和AI表现的关键。我们引入了CSP4SDG，这是一种概率性的约束满足框架，客观地分析游戏玩法。游戏事件和对话被映射到四个语言无关的约束类别——证据、现象、断言和假设。硬约束排除不可能的角色分配，而加权软约束对剩余部分进行评分；基于信息增益的加权链接每个假设在其熵减少下的预期值，并通过一个简单的封闭形式评分规则确保真实断言收敛到最少错误的经典硬逻辑。结果的角色后验是完全可解释并实时更新的。实验表明CSP4SDG在（i）每种推理场景中都优于基于大型语言模型的基线，并且（ii）当作为辅助“推理工具”提供时能增强大型语言模型。我们的研究表明，基于信息理论的原理性的概率推理是一种可扩展的替代或补充方案，适用于社会推理游戏中的重神经模型。 

---
# Chasing Consistency: Quantifying and Optimizing Human-Model Alignment in Chain-of-Thought Reasoning 

**Title (ZH)**: 追求一致性：量化和优化人类-模型推理中的思路一致Alignment 

**Authors**: Boxuan Wang, Zhuoyun Li, Xinmiao Huang, Xiaowei Huang, Yi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2511.06168)  

**Abstract**: This paper presents a framework for evaluating and optimizing reasoning consistency in Large Language Models (LLMs) via a new metric, the Alignment Score, which quantifies the semantic alignment between model-generated reasoning chains and human-written reference chains in Chain-of-Thought (CoT) reasoning. Empirically, we find that 2-hop reasoning chains achieve the highest Alignment Score. To explain this phenomenon, we define four key error types: logical disconnection, thematic shift, redundant reasoning, and causal reversal, and show how each contributes to the degradation of the Alignment Score. Building on this analysis, we further propose Semantic Consistency Optimization Sampling (SCOS), a method that samples and favors chains with minimal alignment errors, significantly improving Alignment Scores by an average of 29.84% with longer reasoning chains, such as in 3-hop tasks. 

**Abstract (ZH)**: 本文提出了一种通过新的评价指标对齐分数（Alignment Score）来评估和优化大型语言模型（LLMs）推理一致性的方法，该指标量化了模型生成的推理链与人类书写的标准推理链在Chain-of-Thought（CoT）推理中的语义对齐程度。实验结果显示，2跳推理链获得最高的对齐分数。为进一步解释这一现象，我们定义了四种关键错误类型：逻辑断联、主题转移、冗余推理和因果倒置，并展示了每种错误类型如何影响对齐分数的降低。基于这一分析，我们进一步提出了一种语义一致性优化采样（SCOS）方法，该方法优先选择对齐错误最少的链路，在更长的推理链，如3跳任务中，使对齐分数平均提高29.84%。 

---
# Evaluating Implicit Biases in LLM Reasoning through Logic Grid Puzzles 

**Title (ZH)**: 通过逻辑格谜题评估LLM推理中的隐性偏见 

**Authors**: Fatima Jahara, Mark Dredze, Sharon Levy  

**Link**: [PDF](https://arxiv.org/pdf/2511.06160)  

**Abstract**: While recent safety guardrails effectively suppress overtly biased outputs, subtler forms of social bias emerge during complex logical reasoning tasks that evade current evaluation benchmarks. To fill this gap, we introduce a new evaluation framework, PRIME (Puzzle Reasoning for Implicit Biases in Model Evaluation), that uses logic grid puzzles to systematically probe the influence of social stereotypes on logical reasoning and decision making in LLMs. Our use of logic puzzles enables automatic generation and verification, as well as variability in complexity and biased settings. PRIME includes stereotypical, anti-stereotypical, and neutral puzzle variants generated from a shared puzzle structure, allowing for controlled and fine-grained comparisons. We evaluate multiple model families across puzzle sizes and test the effectiveness of prompt-based mitigation strategies. Focusing our experiments on gender stereotypes, our findings highlight that models consistently reason more accurately when solutions align with stereotypical associations. This demonstrates the significance of PRIME for diagnosing and quantifying social biases perpetuated in the deductive reasoning of LLMs, where fairness is critical. 

**Abstract (ZH)**: PRIME：Puzzle Reasoning for Implicit Biases in Model Evaluation 

---
# MALinZero: Efficient Low-Dimensional Search for Mastering Complex Multi-Agent Planning 

**Title (ZH)**: MALinZero: 有效的低维搜索方法掌握复杂多智能体规划 

**Authors**: Sizhe Tang, Jiayu Chen, Tian Lan  

**Link**: [PDF](https://arxiv.org/pdf/2511.06142)  

**Abstract**: Monte Carlo Tree Search (MCTS), which leverages Upper Confidence Bound for Trees (UCTs) to balance exploration and exploitation through randomized sampling, is instrumental to solving complex planning problems. However, for multi-agent planning, MCTS is confronted with a large combinatorial action space that often grows exponentially with the number of agents. As a result, the branching factor of MCTS during tree expansion also increases exponentially, making it very difficult to efficiently explore and exploit during tree search. To this end, we propose MALinZero, a new approach to leverage low-dimensional representational structures on joint-action returns and enable efficient MCTS in complex multi-agent planning. Our solution can be viewed as projecting the joint-action returns into the low-dimensional space representable using a contextual linear bandit problem formulation. We solve the contextual linear bandit problem with convex and $\mu$-smooth loss functions -- in order to place more importance on better joint actions and mitigate potential representational limitations -- and derive a linear Upper Confidence Bound applied to trees (LinUCT) to enable novel multi-agent exploration and exploitation in the low-dimensional space. We analyze the regret of MALinZero for low-dimensional reward functions and propose an $(1-\tfrac1e)$-approximation algorithm for the joint action selection by maximizing a sub-modular objective. MALinZero demonstrates state-of-the-art performance on multi-agent benchmarks such as matrix games, SMAC, and SMACv2, outperforming both model-based and model-free multi-agent reinforcement learning baselines with faster learning speed and better performance. 

**Abstract (ZH)**: 基于上下文线性泛函的低维联合动作表示方法在复杂多Agent规划中的蒙特卡洛树搜索 

---
# When Object-Centric World Models Meet Policy Learning: From Pixels to Policies, and Where It Breaks 

**Title (ZH)**: 当对象中心的世界模型遇到策略学习：从像素到策略，以及其局限性 

**Authors**: Stefano Ferraro, Akihiro Nakano, Masahiro Suzuki, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2511.06136)  

**Abstract**: Object-centric world models (OCWM) aim to decompose visual scenes into object-level representations, providing structured abstractions that could improve compositional generalization and data efficiency in reinforcement learning. We hypothesize that explicitly disentangled object-level representations, by localizing task-relevant information, can enhance policy performance across novel feature combinations. To test this hypothesis, we introduce DLPWM, a fully unsupervised, disentangled object-centric world model that learns object-level latents directly from pixels. DLPWM achieves strong reconstruction and prediction performance, including robustness to several out-of-distribution (OOD) visual variations. However, when used for downstream model-based control, policies trained on DLPWM latents underperform compared to DreamerV3. Through latent-trajectory analyses, we identify representation shift during multi-object interactions as a key driver of unstable policy learning. Our results suggest that, although object-centric perception supports robust visual modeling, achieving stable control requires mitigating latent drift. 

**Abstract (ZH)**: 以物为中心的世界模型（OCWM）旨在将视觉场景分解为对象级表示，提供结构化的抽象，以提高强化学习中的组合泛化能力和数据效率。我们假设通过分离的对象级表示，本地化与任务相关的信息，可以增强在新颖特征组合下的策略性能。为了验证这一假设，我们引入了DLPWM，这是一种完全无监督的分离的以物为中心的世界模型，直接从像素中学习对象级的潜在变量。DLPWM在重建和预测性能方面表现出色，并且对多种离分布（OOD）的视觉变化具有鲁棒性。然而，在下游基于模型的控制中，使用DLPWM潜在变量训练的策略的表现不如DreamerV3。通过潜在轨迹分析，我们将多对象交互过程中的表示转移识别为不稳定策略学习的关键驱动因素。我们的结果表明，尽管以物为中心的感觉支持稳健的视觉建模，但实现稳定控制需要缓解潜在变量漂移。 

---
# Maestro: Learning to Collaborate via Conditional Listwise Policy Optimization for Multi-Agent LLMs 

**Title (ZH)**: Maestro: 通过条件列表优化学习多智能体LLM协作 

**Authors**: Wei Yang, Jiacheng Pang, Shixuan Li, Paul Bogdan, Stephen Tu, Jesse Thomason  

**Link**: [PDF](https://arxiv.org/pdf/2511.06134)  

**Abstract**: Multi-agent systems (MAS) built on Large Language Models (LLMs) are being used to approach complex problems and can surpass single model inference. However, their success hinges on navigating a fundamental cognitive tension: the need to balance broad, divergent exploration of the solution space with a principled, convergent synthesis to the optimal solution. Existing paradigms often struggle to manage this duality, leading to premature consensus, error propagation, and a critical credit assignment problem that fails to distinguish between genuine reasoning and superficially plausible arguments. To resolve this core challenge, we propose the Multi-Agent Exploration-Synthesis framework Through Role Orchestration (Maestro), a principled paradigm for collaboration that structurally decouples these cognitive modes. Maestro uses a collective of parallel Execution Agents for diverse exploration and a specialized Central Agent for convergent, evaluative synthesis. To operationalize this critical synthesis phase, we introduce Conditional Listwise Policy Optimization (CLPO), a reinforcement learning objective that disentangles signals for strategic decisions and tactical rationales. By combining decision-focused policy gradients with a list-wise ranking loss over justifications, CLPO achieves clean credit assignment and stronger comparative supervision. Experiments on mathematical reasoning and general problem-solving benchmarks demonstrate that Maestro, coupled with CLPO, consistently outperforms existing state-of-the-art multi-agent approaches, delivering absolute accuracy gains of 6% on average and up to 10% at best. 

**Abstract (ZH)**: 基于大型语言模型的多智能体系统在解决复杂问题方面取得进展，但其成功依赖于驾驭一种根本性的认知张力：在广泛、发散的解空间探索与原则性的收敛合成到最优解之间实现平衡。现有的范式往往难以管理这种二元性，导致过早的一致性、错误传播和一个关键的信用分配问题，难以区分真实的推理与表面上合理但不真实的论点。为解决这一核心挑战，我们提出了一种基于角色编排的多智能体探索-合成框架（Maestro），这是一种原理上的协作范式，结构性地解耦了这些认知模式。Maestro 使用一组并行的执行智能体进行多元探索，并使用一个专门的中央智能体进行收敛的、评估性的合成。为了实现这一关键的合成阶段，我们引入了条件列表级策略优化（CLPO），这是一种强化学习目标，能够分离出战略决策和战术理由的信号。结合决策导向的策略梯度与理由的列表级排名损失，CLPO 实现了清晰的信用分配和更强的监督。在数学推理和通用问题解决基准测试上的实验表明，结合 CLPO 的 Maestro 一致地优于现有的多智能体方法，在平均绝对准确率上提高了 6%，最高提高了 10%。 

---
# ScRPO: From Errors to Insights 

**Title (ZH)**: ScRPO: 从错误到洞察 

**Authors**: Lianrui Li, Dakuan Lu, Jiawei Shao, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06065)  

**Abstract**: We propose Self-correction Relative Policy Optimization (ScRPO), a novel reinforcement learning framework designed to enhance large language models on challenging mathemati- cal problems by leveraging self-reflection and error correction. Our approach consists of two stages: (1) Trial-and-error learning stage: training the model with GRPO and collect- ing incorrect answers along with their cor- responding questions in an error pool; (2) Self-correction learning stage: guiding the model to reflect on why its previous an- swers were wrong. Extensive experiments across multiple math reasoning benchmarks, including AIME, AMC, Olympiad, MATH- 500, GSM8k, using Deepseek-Distill-Qwen- 1.5B and Deepseek-Distill-Qwen-7B. The ex- perimental results demonstrate that ScRPO consistently outperforms several post-training methods. These findings highlight ScRPO as a promising paradigm for enabling language models to self-improve on difficult tasks with limited external feedback, paving the way to- ward more reliable and capable AI systems. 

**Abstract (ZH)**: 我们提出了一种新颖的强化学习框架自修正相对策略优化（ScRPO），该框架通过利用自我反思和错误修正来增强大型语言模型在具有挑战性的数学问题上的性能。我们的方法包括两个阶段：（1）试错学习阶段：使用GRPO训练模型，并在错误池中收集错误的答案及其对应的题目；（2）自我修正学习阶段：引导模型反思其先前答案错误的原因。在AIME、AMC、奥林匹克竞赛、MATH-500、GSM8k等多个数学推理基准上使用Deepseek-Distill-Qwen-1.5B和Deepseek-Distill-Qwen-7B进行了广泛实验。实验结果表明，ScRPO在多个后训练方法中表现更优。这些发现突显了ScRPO作为一种使语言模型在有限外部反馈的情况下自我提升困难任务有前途的范式，为更可靠和强大的AI系统铺平了道路。 

---
# An Epistemic Perspective on Agent Awareness 

**Title (ZH)**: 基于知识论的智能体觉知研究 

**Authors**: Pavel Naumov, Alexandra Pavlova  

**Link**: [PDF](https://arxiv.org/pdf/2511.05977)  

**Abstract**: The paper proposes to treat agent awareness as a form of knowledge, breaking the tradition in the existing literature on awareness. It distinguishes the de re and de dicto forms of such knowledge. The work introduces two modalities capturing these forms and formally specifies their meaning using a version of 2D-semantics. The main technical result is a sound and complete logical system describing the interplay between the two proposed modalities and the standard "knowledge of the fact" modality. 

**Abstract (ZH)**: 该论文提议将代理意识视为一种知识形式，打破了现有意识文献中的传统。它区分了这种知识的de re和de dicto形式。工作引入了两种模态来捕捉这些形式，并使用2D语义学的一个版本正式规定它们的意义。主要的技术结果是一个描述所提出模态与标准“事实知识”模态之间互动的完整逻辑系统。 

---
# Klear-AgentForge: Forging Agentic Intelligence through Posttraining Scaling 

**Title (ZH)**: Klear-AgentForge：通过后训练缩放锻造代理智能 

**Authors**: Qi Wang, Hongzhi Zhang, Jia Fu, Kai Fu, Yahui Liu, Tinghai Zhang, Chenxi Sun, Gangwei Jiang, Jingyi Tang, Xingguang Ji, Yang Yue, Jingyuan Zhang, Fuzheng Zhang, Kun Gai, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.05951)  

**Abstract**: Despite the proliferation of powerful agentic models, the lack of critical post-training details hinders the development of strong counterparts in the open-source community. In this study, we present a comprehensive and fully open-source pipeline for training a high-performance agentic model for interacting with external tools and environments, named Klear-Qwen3-AgentForge, starting from the Qwen3-8B base model. We design effective supervised fine-tuning (SFT) with synthetic data followed by multi-turn reinforcement learning (RL) to unlock the potential for multiple diverse agentic tasks. We perform exclusive experiments on various agentic benchmarks in both tool use and coding domains. Klear-Qwen3-AgentForge-8B achieves state-of-the-art performance among LLMs of similar size and remains competitive with significantly larger models. 

**Abstract (ZH)**: 尽管强大的代理模型层出不穷，但缺乏关键的后训练细节阻碍了开源社区中强大对应模型的发展。在此研究中，我们提出了一种全面且完全开源的培训Pipeline，用于使用Qwen3-8B基础模型训练高性能代理模型以与外部工具和环境交互，名为Klear-Qwen3-AgentForge。我们设计了有效的监督微调（SFT）并结合多轮强化学习（RL）以解锁多种多样化代理任务的潜力。我们在工具使用和编程领域进行了独家基准实验。Klear-Qwen3-AgentForge-8B在类似大小的LLM中达到了最先进的性能，并且在显著更大的模型中依然具有竞争力。 

---
# Self-Abstraction from Grounded Experience for Plan-Guided Policy Refinement 

**Title (ZH)**: 基于接地经验的自我抽象用于计划导向的策略精炼 

**Authors**: Hiroaki Hayashi, Bo Pang, Wenting Zhao, Ye Liu, Akash Gokul, Srijan Bansal, Caiming Xiong, Semih Yavuz, Yingbo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.05931)  

**Abstract**: Large language model (LLM) based agents are increasingly used to tackle software engineering tasks that require multi-step reasoning and code modification, demonstrating promising yet limited performance. However, most existing LLM agents typically operate within static execution frameworks, lacking a principled mechanism to learn and self-improve from their own experience and past rollouts. As a result, their performance remains bounded by the initial framework design and the underlying LLM's capabilities. We propose Self-Abstraction from Grounded Experience (SAGE), a framework that enables agents to learn from their own task executions and refine their behavior through self-abstraction. After an initial rollout, the agent induces a concise plan abstraction from its grounded experience, distilling key steps, dependencies, and constraints. This learned abstraction is then fed back as contextual guidance, refining the agent's policy and supporting more structured, informed subsequent executions. Empirically, SAGE delivers consistent performance gains across diverse LLM backbones and agent architectures. Notably, it yields a 7.2% relative performance improvement over the strong Mini-SWE-Agent baseline when paired with the GPT-5 (high) backbone. SAGE further achieves strong overall performance on SWE-Bench Verified benchmark, reaching 73.2% and 74% Pass@1 resolve rates with the Mini-SWE-Agent and OpenHands CodeAct agent framework, respectively. 

**Abstract (ZH)**: 基于大型语言模型的代理通过自我抽抽象化从 grounded 经验中学习以提升软件工程任务表现：SAGE框架 

---
# Unveiling Modality Bias: Automated Sample-Specific Analysis for Multimodal Misinformation Benchmarks 

**Title (ZH)**: 揭示模态偏差：自动样本特定分析在多模态误导性信息基准中的应用 

**Authors**: Hehai Lin, Hui Liu, Shilei Cao, Jing Li, Haoliang Li, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05883)  

**Abstract**: Numerous multimodal misinformation benchmarks exhibit bias toward specific modalities, allowing detectors to make predictions based solely on one modality. While previous research has quantified bias at the dataset level or manually identified spurious correlations between modalities and labels, these approaches lack meaningful insights at the sample level and struggle to scale to the vast amount of online information. In this paper, we investigate the design for automated recognition of modality bias at the sample level. Specifically, we propose three bias quantification methods based on theories/views of different levels of granularity: 1) a coarse-grained evaluation of modality benefit; 2) a medium-grained quantification of information flow; and 3) a fine-grained causality analysis. To verify the effectiveness, we conduct a human evaluation on two popular benchmarks. Experimental results reveal three interesting findings that provide potential direction toward future research: 1)~Ensembling multiple views is crucial for reliable automated analysis; 2)~Automated analysis is prone to detector-induced fluctuations; and 3)~Different views produce a higher agreement on modality-balanced samples but diverge on biased ones. 

**Abstract (ZH)**: 多种模态错误信息基准在特定模态上存在偏差，使得检测器可以仅基于单一模态进行预测。以往研究在数据集层面量化偏差或手动识别模态与标签之间的虚假关联，但这些方法缺乏样本层面的有意义洞见，难以应对海量在线信息。本文探讨了在样本层面自动识别模态偏差的设计。具体地，我们提出了三种基于不同粒度理论/观点的偏差量化方法：1）粗粒度的模态益处评估；2）中粒度的信息流量化；3）细粒度的因果分析。为了验证其有效性，我们在两个流行的基准上进行了人工评估。实验结果揭示了三个有趣的研究方向：1）综合多种视角对于可靠自动分析至关重要；2）自动分析容易受到检测器引起的波动影响；3）不同视角在模态平衡样本上产生较高的一致性，但在偏差样本上则出现分歧。 

---
# An Empirical Study of Reasoning Steps in Thinking Code LLMs 

**Title (ZH)**: 思考代码LLM的推理由实证研究 

**Authors**: Haoran Xue, Gias Uddin, Song Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05874)  

**Abstract**: Thinking Large Language Models (LLMs) generate explicit intermediate reasoning traces before final answers, potentially improving transparency, interpretability, and solution accuracy for code generation. However, the quality of these reasoning chains remains underexplored. We present a comprehensive empirical study examining the reasoning process and quality of thinking LLMs for code generation. We evaluate six state-of-the-art reasoning LLMs (DeepSeek-R1, OpenAI-o3-mini, Claude-3.7-Sonnet-Thinking, Gemini-2.0-Flash-Thinking, Gemini-2.5-Flash, and Qwen-QwQ) across 100 code generation tasks of varying difficulty from BigCodeBench. We quantify reasoning-chain structure through step counts and verbosity, conduct controlled step-budget adjustments, and perform a 21-participant human evaluation across three dimensions: efficiency, logical correctness, and completeness. Our step-count interventions reveal that targeted step increases can improve resolution rates for certain models/tasks, while modest reductions often preserve success on standard tasks, rarely on hard ones. Through systematic analysis, we develop a reasoning-problematic taxonomy, identifying completeness as the dominant failure mode. Task complexity significantly impacts reasoning quality; hard problems are substantially more prone to incompleteness than standard tasks. Our stability analysis demonstrates that thinking LLMs maintain consistent logical structures across computational effort levels and can self-correct previous errors. This study provides new insights into the strengths and limitations of current thinking LLMs in software engineering. 

**Abstract (ZH)**: 大型语言模型（LLMs）在生成最终答案之前会产生明确的中间推理过程，这可能提高代码生成的透明度、可解释性和解决方案准确性。然而，这些推理链的质量尚未得到充分探索。我们进行了一项全面的经验性研究，探讨代码生成中思考LLMs的推理过程和质量。我们评估了六个最先进的推理LLM（DeepSeek-R1、OpenAI-o3-mini、Claude-3.7-Sonnet-Thinking、Gemini-2.0-Flash-Thinking、Gemini-2.5-Flash和Qwen-QwQ），涵盖来自BigCodeBench的100个不同难度级别的代码生成任务。我们通过步数和冗长度量化推理链结构，进行受控的步数预算调整，并在效率、逻辑正确性和完整性三个维度上对21名参与者进行人类评估。我们的步数干预结果显示，针对某些模型/任务的步数增加可以提高分辨率率，适度减少步数通常能保留标准任务的成功率，但在困难任务上几乎不起作用。通过系统分析，我们开发了一个推理问题分类体系，指出完整度是主要的失败模式。任务复杂度显著影响推理质量；困难问题比标准任务更容易出现不完整性。稳定性分析显示，思考LLMs在不同计算努力水平下维持一致的逻辑结构，并能自我纠正之前的错误。本研究为当前思考LLMs在软件工程中的优势和局限性提供了新的见解。 

---
# Can a Small Model Learn to Look Before It Leaps? Dynamic Learning and Proactive Correction for Hallucination Detection 

**Title (ZH)**: 小模型能学会先思后行吗？动态学习与前瞻修正以检测幻觉 

**Authors**: Zepeng Bao, Shen Zhou, Qiankun Pi, Jianhao Chen, Mayi Xu, Ming Zhong, Yuanyuan Zhu, Tieyun Qian  

**Link**: [PDF](https://arxiv.org/pdf/2511.05854)  

**Abstract**: Hallucination in large language models (LLMs) remains a critical barrier to their safe deployment. Existing tool-augmented hallucination detection methods require pre-defined fixed verification strategies, which are crucial to the quality and effectiveness of tool calls. Some methods directly employ powerful closed-source LLMs such as GPT-4 as detectors, which are effective but too costly. To mitigate the cost issue, some methods adopt the teacher-student architecture and finetune open-source small models as detectors via agent tuning. However, these methods are limited by fixed strategies. When faced with a dynamically changing execution environment, they may lack adaptability and inappropriately call tools, ultimately leading to detection failure. To address the problem of insufficient strategy adaptability, we propose the innovative ``Learning to Evaluate and Adaptively Plan''(LEAP) framework, which endows an efficient student model with the dynamic learning and proactive correction capabilities of the teacher model. Specifically, our method formulates the hallucination detection problem as a dynamic strategy learning problem. We first employ a teacher model to generate trajectories within the dynamic learning loop and dynamically adjust the strategy based on execution failures. We then distill this dynamic planning capability into an efficient student model via agent tuning. Finally, during strategy execution, the student model adopts a proactive correction mechanism, enabling it to propose, review, and optimize its own verification strategies before execution. We demonstrate through experiments on three challenging benchmarks that our LEAP-tuned model outperforms existing state-of-the-art methods. 

**Abstract (ZH)**: 大型语言模型中的幻觉 remains a critical barrier to their safe deployment. Innovative “Learning to Evaluate and Adaptively Plan” (LEAP) Framework for Dynamic Strategy Learning in Hallucination Detection 

---
# DiagnoLLM: A Hybrid Bayesian Neural Language Framework for Interpretable Disease Diagnosis 

**Title (ZH)**: DiagnoLLM：一种用于可解释疾病诊断的混合贝叶斯神经语言框架 

**Authors**: Bowen Xu, Xinyue Zeng, Jiazhen Hu, Tuo Wang, Adithya Kulkarni  

**Link**: [PDF](https://arxiv.org/pdf/2511.05810)  

**Abstract**: Building trustworthy clinical AI systems requires not only accurate predictions but also transparent, biologically grounded explanations. We present \texttt{DiagnoLLM}, a hybrid framework that integrates Bayesian deconvolution, eQTL-guided deep learning, and LLM-based narrative generation for interpretable disease diagnosis. DiagnoLLM begins with GP-unmix, a Gaussian Process-based hierarchical model that infers cell-type-specific gene expression profiles from bulk and single-cell RNA-seq data while modeling biological uncertainty. These features, combined with regulatory priors from eQTL analysis, power a neural classifier that achieves high predictive performance in Alzheimer's Disease (AD) detection (88.0\% accuracy). To support human understanding and trust, we introduce an LLM-based reasoning module that translates model outputs into audience-specific diagnostic reports, grounded in clinical features, attribution signals, and domain knowledge. Human evaluations confirm that these reports are accurate, actionable, and appropriately tailored for both physicians and patients. Our findings show that LLMs, when deployed as post-hoc reasoners rather than end-to-end predictors, can serve as effective communicators within hybrid diagnostic pipelines. 

**Abstract (ZH)**: 构建可信赖的临床AI系统不仅需要准确的预测，还需要透明且生物学依据充分的解释。我们提出\texttt{DiagnoLLM}，这是一种将贝叶斯去混合作为混合框架，结合eQTL指导的深度学习和基于LLM的叙述生成，以实现可解释的疾病诊断。DiagnoLLM 从 GP-unmix 开始，这是一种基于高斯过程的分层模型，可以从bulk和单细胞RNA-seq数据中推断出特定于细胞类型的时间表基因表达谱，并建模生物学不确定性。这些特性结合了eQTL分析中的调节先验，驱动一个神经分类器，在阿尔茨海默病（AD）检测中实现高度的预测性能（准确率为88.0%）。为了支持人类的理解和信任，我们引入了一个基于LLM的推理模块，将模型输出翻译成针对特定受众的诊断报告，这些报告扎根于临床特征、归因信号和领域知识。人类评估证实，这些报告是准确的、操作性的，并且针对医生和患者进行了恰当的调整。我们的研究结果表明，当LLM作为后置推理器而不是端到端预测器部署时，它们可以在混合诊断流水线中充当有效的沟通工具。 

---
# Anchors in the Machine: Behavioral and Attributional Evidence of Anchoring Bias in LLMs 

**Title (ZH)**: 机器中的锚点：LLMs中锚定偏差的行为与归因证据 

**Authors**: Felipe Valencia-Clavijo  

**Link**: [PDF](https://arxiv.org/pdf/2511.05766)  

**Abstract**: Large language models (LLMs) are increasingly examined as both behavioral subjects and decision systems, yet it remains unclear whether observed cognitive biases reflect surface imitation or deeper probability shifts. Anchoring bias, a classic human judgment bias, offers a critical test case. While prior work shows LLMs exhibit anchoring, most evidence relies on surface-level outputs, leaving internal mechanisms and attributional contributions unexplored. This paper advances the study of anchoring in LLMs through three contributions: (1) a log-probability-based behavioral analysis showing that anchors shift entire output distributions, with controls for training-data contamination; (2) exact Shapley-value attribution over structured prompt fields to quantify anchor influence on model log-probabilities; and (3) a unified Anchoring Bias Sensitivity Score integrating behavioral and attributional evidence across six open-source models. Results reveal robust anchoring effects in Gemma-2B, Phi-2, and Llama-2-7B, with attribution signaling that the anchors influence reweighting. Smaller models such as GPT-2, Falcon-RW-1B, and GPT-Neo-125M show variability, suggesting scale may modulate sensitivity. Attributional effects, however, vary across prompt designs, underscoring fragility in treating LLMs as human substitutes. The findings demonstrate that anchoring bias in LLMs is robust, measurable, and interpretable, while highlighting risks in applied domains. More broadly, the framework bridges behavioral science, LLM safety, and interpretability, offering a reproducible path for evaluating other cognitive biases in LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）既被作为行为主体又被作为决策系统进行研究，但观察到的认知偏误是表层模仿还是深层次的概率变化仍不清楚。锚定偏差，一种经典的人类判断偏差，提供了一个关键的测试案例。尽管先前的研究表明LLMs表现出锚定偏差，但大多数证据依赖于表层输出，内部机制和归因贡献尚不清楚。本文通过三个贡献推进了对LLMs锚定偏差的研究：（1）基于log概率的行为分析，表明锚点改变了整个输出分布，并控制了训练数据污染；（2）在结构化提示字段上的精确Shapley值归因，以量化锚点对模型log概率的影响；（3）统一的锚定偏差敏感性评分，将行为和归因证据结合，在六个开源模型中进行整合。结果显示，Gemma-2B、Phi-2和Llama-2-7B表现出稳健的锚定效应，归因表明锚点影响了权重重新分配。较小的模型如GPT-2、Falcon-RW-1B和GPT-Neo-125M显示出变化性，表明规模可能会影响敏感性。然而，归因效应在不同提示设计中有所不同，凸显了将LLMs视为人类替代品的脆弱性。研究结果表明，LLMs中的锚定偏差是稳健的、可测量的和可解释的，同时指出了应用领域中的风险。更广泛地看，该框架将行为科学、LLMs安全性和可解释性联系起来，提供了一条评估LLMs中其他认知偏差的可重复路径。 

---
# CoT-X: An Adaptive Framework for Cross-Model Chain-of-Thought Transfer and Optimization 

**Title (ZH)**: CoT-X：一种适应性跨模型链式思考转移与优化框架 

**Authors**: Ziqian Bi, Kaijie Chen, Tianyang Wang, Junfeng Hao, Xinyuan Song  

**Link**: [PDF](https://arxiv.org/pdf/2511.05747)  

**Abstract**: Chain-of-Thought (CoT) reasoning enhances the problem-solving ability of large language models (LLMs) but leads to substantial inference overhead, limiting deployment in resource-constrained settings. This paper investigates efficient CoT transfer across models of different scales and architectures through an adaptive reasoning summarization framework. The proposed method compresses reasoning traces via semantic segmentation with importance scoring, budget-aware dynamic compression, and coherence reconstruction, preserving critical reasoning steps while significantly reducing token usage. Experiments on 7{,}501 medical examination questions across 10 specialties show up to 40% higher accuracy than truncation under the same token budgets. Evaluations on 64 model pairs from eight LLMs (1.5B-32B parameters, including DeepSeek-R1 and Qwen3) confirm strong cross-model transferability. Furthermore, a Gaussian Process-based Bayesian optimization module reduces evaluation cost by 84% and reveals a power-law relationship between model size and cross-domain robustness. These results demonstrate that reasoning summarization provides a practical path toward efficient CoT transfer, enabling advanced reasoning under tight computational constraints. Code will be released upon publication. 

**Abstract (ZH)**: Chain-of-Thought推理压缩促进大规模语言模型之间的高效迁移学习 

---
# From Prompts to Power: Measuring the Energy Footprint of LLM Inference 

**Title (ZH)**: 从提示到能源：测量大模型推理的能源足迹 

**Authors**: Francisco Caravaca, Ángel Cuevas, Rubén Cuevas  

**Link**: [PDF](https://arxiv.org/pdf/2511.05597)  

**Abstract**: The rapid expansion of Large Language Models (LLMs) has introduced unprecedented energy demands, extending beyond training to large-scale inference workloads that often dominate total lifecycle consumption. Deploying these models requires energy-intensive GPU infrastructure, and in some cases has even prompted plans to power data centers with nuclear energy. Despite this growing relevance, systematic analyses of inference energy consumption remain limited. In this work, we present a large-scale measurement-based study comprising over 32,500 measurements across 21 GPU configurations and 155 model architectures, from small open-source models to frontier systems. Using the vLLM inference engine, we quantify energy usage at the prompt level and identify how architectural and operational factors shape energy demand. Building on these insights, we develop a predictive model that accurately estimates inference energy consumption across unseen architectures and hardware, and implement it as a browser extension to raise awareness of the environmental impact of generative AI. 

**Abstract (ZH)**: 大型语言模型的迅速扩张引入了前所未有的能源需求，不仅限于训练，还扩展到常常主导整个生命周期能耗的大规模推理工作负载。部署这些模型需要耗能密集型的GPU基础设施，在某些情况下甚至促使计划使用核能来供电数据中心。尽管其重要性日益增加，但对推理能源消耗的系统分析仍相对有限。在本文中，我们进行了一项大规模基于测量的研究，涵盖了超过32,500个测量数据、21种GPU配置和155种模型架构，从小型开源模型到前沿系统。借助vLLM推理引擎，我们在提示级别量化了能源使用情况，并分析了架构和运营因素如何塑造能源需求。基于这些洞察，我们开发了一种预测模型，能够准确估计未见过的架构和硬件的推理能源消耗，并将其实现为浏览器扩展，以提高人们对生成式AI环境影响的认识。 

---
# SMAGDi: Socratic Multi Agent Interaction Graph Distillation for Efficient High Accuracy Reasoning 

**Title (ZH)**: SMAGDi：苏格拉底多Agent交互图蒸馏以实现高效高精度推理 

**Authors**: Aayush Aluru, Myra Malik, Samarth Patankar, Spencer Kim, Kevin Zhu, Sean O'Brien, Vasu Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2511.05528)  

**Abstract**: Multi-agent systems (MAS) often achieve higher reasoning accuracy than single models, but their reliance on repeated debates across agents makes them computationally expensive. We introduce SMAGDi, a distillation framework that transfers the debate dynamics of a five-agent Llama-based MAS into a compact Socratic decomposer-solver student. SMAGDi represents debate traces as directed interaction graphs, where nodes encode intermediate reasoning steps with correctness labels and edges capture continuity and cross-agent influence. The student is trained with a composite objective combining language modeling, graph-based supervision, contrastive reasoning, and embedding alignment to preserve both fluency and structured reasoning. On StrategyQA and MMLU, SMAGDi compresses a 40B multi-agent system into a 6B student while retaining 88% of its accuracy, substantially outperforming prior distillation methods such as MAGDi, standard KD, and fine-tuned baselines. These results highlight that explicitly modeling interaction graphs and Socratic decomposition enable small models to inherit the accuracy benefits of multi-agent debate while remaining efficient enough for real-world deployment. 

**Abstract (ZH)**: 基于LLAMA的五代理MAS的SMAGDi：一种转移对话动态的精简框架 

---
# Evidence-Bound Autonomous Research (EviBound): A Governance Framework for Eliminating False Claims 

**Title (ZH)**: 证据约束自主研究（EviBound）：一种消除虚假声称的治理框架 

**Authors**: Ruiying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.05524)  

**Abstract**: LLM-based autonomous research agents report false claims: tasks marked "complete" despite missing artifacts, contradictory metrics, or failed executions. EviBound is an evidence-bound execution framework that eliminates false claims through dual governance gates requiring machine-checkable evidence.
Two complementary gates enforce evidence requirements. The pre-execution Approval Gate validates acceptance criteria schemas before code runs, catching structural violations proactively. The post-execution Verification Gate validates artifacts via MLflow API queries (with recursive path checking) and optionally validates metrics when specified by acceptance criteria. Claims propagate only when backed by a queryable run ID, required artifacts, and FINISHED status. Bounded, confidence-gated retries (typically 1-2 attempts) recover from transient failures without unbounded loops.
The framework was evaluated on 8 benchmark tasks spanning infrastructure validation, ML capabilities, and governance stress tests. Baseline A (Prompt-Level Only) yields 100% hallucination (8/8 claimed, 0/8 verified). Baseline B (Verification-Only) reduces hallucination to 25% (2/8 fail verification). EviBound (Dual Gates) achieves 0% hallucination: 7/8 tasks verified and 1 task correctly blocked at the approval gate, all with only approximately 8.3% execution overhead.
This package includes execution trajectories, MLflow run IDs for all verified tasks, and a 4-step verification protocol. Research integrity is an architectural property, achieved through governance gates rather than emergent from model scale. 

**Abstract (ZH)**: 基于LLM的自主研究代理报告虚假声明：任务标记为“完成”尽管缺少构件、存在矛盾的指标或执行失败。EviBound是通过双重治理门要求可机读证据来消除虚假声明的证据边界执行框架。 

---
# Lightning Grasp: High Performance Procedural Grasp Synthesis with Contact Fields 

**Title (ZH)**: 闪电抓取：基于接触场的高性能程序化抓取合成 

**Authors**: Zhao-Heng Yin, Pieter Abbeel  

**Link**: [PDF](https://arxiv.org/pdf/2511.07418)  

**Abstract**: Despite years of research, real-time diverse grasp synthesis for dexterous hands remains an unsolved core challenge in robotics and computer graphics. We present Lightning Grasp, a novel high-performance procedural grasp synthesis algorithm that achieves orders-of-magnitude speedups over state-of-the-art approaches, while enabling unsupervised grasp generation for irregular, tool-like objects. The method avoids many limitations of prior approaches, such as the need for carefully tuned energy functions and sensitive initialization. This breakthrough is driven by a key insight: decoupling complex geometric computation from the search process via a simple, efficient data structure - the Contact Field. This abstraction collapses the problem complexity, enabling a procedural search at unprecedented speeds. We open-source our system to propel further innovation in robotic manipulation. 

**Abstract (ZH)**: 尽管多年研究，灵巧手的实时多样抓取合成仍然是机器人学和计算机图形学中的一个未解决的核心挑战。我们提出了一种名为Lightning Grasp的新型高性能过程化抓取合成算法，该算法在保持与最新方法相当的抓取质量的同时，实现了数量级的速度提升，并能够无监督生成不规则工具型物体的抓取。该方法克服了以往方法的许多局限性，如需要精细调谐的能量函数和敏感的初始化。这一突破得益于一个关键洞察：通过简单有效的数据结构——接触场，将复杂的几何计算与搜索过程解耦，从而降低问题复杂性，实现前所未有的快速过程化搜索。我们开源该系统以促进机器人操作领域的进一步创新。 

---
# Language Generation with Infinite Contamination 

**Title (ZH)**: 无限污染的语言生成 

**Authors**: Anay Mehrotra, Grigoris Velegkas, Xifan Yu, Felix Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.07417)  

**Abstract**: We study language generation in the limit, where an algorithm observes an adversarial enumeration of strings from an unknown target language $K$ and must eventually generate new, unseen strings from $K$. Kleinberg and Mullainathan [KM24] proved that generation is achievable in surprisingly general settings. But their generator suffers from ``mode collapse,'' producing from an ever-smaller subset of the target. To address this, Kleinberg and Wei [KW25] require the generator's output to be ``dense'' in the target language. They showed that generation with density, surprisingly, remains achievable at the same generality.
Both results assume perfect data: no noisy insertions and no omissions. This raises a central question: how much contamination can generation tolerate? Recent works made partial progress on this question by studying (non-dense) generation with either finite amounts of noise (but no omissions) or omissions (but no noise).
We characterize robustness under contaminated enumerations: 1. Generation under Contamination: Language generation in the limit is achievable for all countable collections iff the fraction of contaminated examples converges to zero. When this fails, we characterize which collections are generable. 2. Dense Generation under Contamination: Dense generation is strictly less robust to contamination than generation. As a byproduct, we resolve an open question of Raman and Raman [ICML25] by showing that generation is possible with only membership oracle access under finitely many contaminated examples.
Finally, we introduce a beyond-worst-case model inspired by curriculum learning and prove that dense generation is achievable even with infinite contamination provided the fraction of contaminated examples converges to zero. This suggests curriculum learning may be crucial for learning from noisy web data. 

**Abstract (ZH)**: 语言生成中的容错性研究：在受污染枚举下的生成 

---
# Robot Learning from a Physical World Model 

**Title (ZH)**: 机器人学习从物理世界模型角度探讨 

**Authors**: Jiageng Mao, Sicheng He, Hao-Ning Wu, Yang You, Shuyang Sun, Zhicheng Wang, Yanan Bao, Huizhong Chen, Leonidas Guibas, Vitor Guizilini, Howard Zhou, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07416)  

**Abstract**: We introduce PhysWorld, a framework that enables robot learning from video generation through physical world modeling. Recent video generation models can synthesize photorealistic visual demonstrations from language commands and images, offering a powerful yet underexplored source of training signals for robotics. However, directly retargeting pixel motions from generated videos to robots neglects physics, often resulting in inaccurate manipulations. PhysWorld addresses this limitation by coupling video generation with physical world reconstruction. Given a single image and a task command, our method generates task-conditioned videos and reconstructs the underlying physical world from the videos, and the generated video motions are grounded into physically accurate actions through object-centric residual reinforcement learning with the physical world model. This synergy transforms implicit visual guidance into physically executable robotic trajectories, eliminating the need for real robot data collection and enabling zero-shot generalizable robotic manipulation. Experiments on diverse real-world tasks demonstrate that PhysWorld substantially improves manipulation accuracy compared to previous approaches. Visit \href{this https URL}{the project webpage} for details. 

**Abstract (ZH)**: PhysWorld：一种通过物理世界建模实现基于视频生成的机器人学习框架 

---
# Using Vision Language Models as Closed-Loop Symbolic Planners for Robotic Applications: A Control-Theoretic Perspective 

**Title (ZH)**: 将视觉语言模型用于闭环符号规划的机器人应用：一种控制理论视角 

**Authors**: Hao Wang, Sathwik Karnik, Bea Lim, Somil Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2511.07410)  

**Abstract**: Large Language Models (LLMs) and Vision Language Models (VLMs) have been widely used for embodied symbolic planning. Yet, how to effectively use these models for closed-loop symbolic planning remains largely unexplored. Because they operate as black boxes, LLMs and VLMs can produce unpredictable or costly errors, making their use in high-level robotic planning especially challenging. In this work, we investigate how to use VLMs as closed-loop symbolic planners for robotic applications from a control-theoretic perspective. Concretely, we study how the control horizon and warm-starting impact the performance of VLM symbolic planners. We design and conduct controlled experiments to gain insights that are broadly applicable to utilizing VLMs as closed-loop symbolic planners, and we discuss recommendations that can help improve the performance of VLM symbolic planners. 

**Abstract (ZH)**: 大型语言模型（LLMs）和视觉语言模型（VLMs）已在体现符号规划中得到了广泛应用，但如何有效利用这些模型进行闭环符号规划仍 largely unexplored。由于它们作为黑盒运作，LLMs 和 VLMs 可能会产生不可预测或代价高昂的错误，使其在高级机器人规划中的应用尤其具有挑战性。在本项工作中，我们从控制理论的角度研究如何使用 VLMs 作为闭环符号规划器来应用于机器人应用。具体而言，我们探讨了控制窗口和预热启动如何影响 VLM 符号规划器的性能。我们设计并进行了控制实验，以获得对使用 VLMs 作为闭环符号规划器具有广泛适用性的洞见，并讨论了有助于提高 VLM 符号规划器性能的建议。 

---
# SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards 

**Title (ZH)**: SpatialThinker: 在多模态LLM中通过空间奖励强化三维推理 

**Authors**: Hunar Batra, Haoqin Tu, Hardy Chen, Yuanze Lin, Cihang Xie, Ronald Clark  

**Link**: [PDF](https://arxiv.org/pdf/2511.07403)  

**Abstract**: Multimodal large language models (MLLMs) have achieved remarkable progress in vision-language tasks, but they continue to struggle with spatial understanding. Existing spatial MLLMs often rely on explicit 3D inputs or architecture-specific modifications, and remain constrained by large-scale datasets or sparse supervision. To address these limitations, we introduce SpatialThinker, a 3D-aware MLLM trained with RL to integrate structured spatial grounding with multi-step reasoning. The model simulates human-like spatial perception by constructing a scene graph of task-relevant objects and spatial relations, and reasoning towards an answer via dense spatial rewards. SpatialThinker consists of two key contributions: (1) a data synthesis pipeline that generates STVQA-7K, a high-quality spatial VQA dataset, and (2) online RL with a multi-objective dense spatial reward enforcing spatial grounding. SpatialThinker-7B outperforms supervised fine-tuning and the sparse RL baseline on spatial understanding and real-world VQA benchmarks, nearly doubling the base-model gain compared to sparse RL, and surpassing GPT-4o. These results showcase the effectiveness of combining spatial supervision with reward-aligned reasoning in enabling robust 3D spatial understanding with limited data and advancing MLLMs towards human-level visual reasoning. 

**Abstract (ZH)**: 具有空间意识的大规模多模态语言模型：通过强化学习集成结构化空间接地与多步推理 

---
# Surgical Agent Orchestration Platform for Voice-directed Patient Data Interaction 

**Title (ZH)**: 手术机器人指挥平台语音指导患者数据交互 

**Authors**: Hyeryun Park, Byung Mo Gu, Jun Hee Lee, Byeong Hyeon Choi, Sekeun Kim, Hyun Koo Kim, Kyungsang Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.07392)  

**Abstract**: In da Vinci robotic surgery, surgeons' hands and eyes are fully engaged in the procedure, making it difficult to access and manipulate multimodal patient data without interruption. We propose a voice-directed Surgical Agent Orchestrator Platform (SAOP) built on a hierarchical multi-agent framework, consisting of an orchestration agent and three task-specific agents driven by Large Language Models (LLMs). These LLM-based agents autonomously plan, refine, validate, and reason to map voice commands into specific tasks such as retrieving clinical information, manipulating CT scans, or navigating 3D anatomical models on the surgical video. We also introduce a Multi-level Orchestration Evaluation Metric (MOEM) to comprehensively assess the performance and robustness from command-level and category-level perspectives. The SAOP achieves high accuracy and success rates across 240 voice commands, while LLM-based agents improve robustness against speech recognition errors and diverse or ambiguous free-form commands, demonstrating strong potential to support minimally invasive da Vinci robotic surgery. 

**Abstract (ZH)**: 基于多层次多Agent框架的语音指导外科智能调度平台（SAOP）：支持微创达芬奇机器人手术 

---
# Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence 

**Title (ZH)**: 通过返初始化循环增强教预训练语言模型深度思考 

**Authors**: Sean McLeish, Ang Li, John Kirchenbauer, Dayal Singh Kalra, Brian R. Bartoldson, Bhavya Kailkhura, Avi Schwarzschild, Jonas Geiping, Tom Goldstein, Micah Goldblum  

**Link**: [PDF](https://arxiv.org/pdf/2511.07384)  

**Abstract**: Recent advances in depth-recurrent language models show that recurrence can decouple train-time compute and parameter count from test-time compute. In this work, we study how to convert existing pretrained non-recurrent language models into depth-recurrent models. We find that using a curriculum of recurrences to increase the effective depth of the model over the course of training preserves performance while reducing total computational cost. In our experiments, on mathematics, we observe that converting pretrained models to recurrent ones results in better performance at a given compute budget than simply post-training the original non-recurrent language model. 

**Abstract (ZH)**: Recent advances in深度递归语言模型表明，递归可以分离训练时计算量和参数量与测试时计算量的关系。在这项工作中，我们研究如何将现有的非递归预训练语言模型转换为深度递归模型。我们发现，在训练过程中使用递归课程逐步增加模型的有效深度可以在保持性能的同时降低总计算成本。在我们的实验中，对于数学任务，我们将预训练模型转换为递归模型在给定计算预算下比对原非递归语言模型进行后训练表现更好。 

---
# LoReTTA: A Low Resource Framework To Poison Continuous Time Dynamic Graphs 

**Title (ZH)**: LoReTTA：一种低资源污染连续时间动态图的框架 

**Authors**: Himanshu Pal, Venkata Sai Pranav Bachina, Ankit Gangwal, Charu Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2511.07379)  

**Abstract**: Temporal Graph Neural Networks (TGNNs) are increasingly used in high-stakes domains, such as financial forecasting, recommendation systems, and fraud detection. However, their susceptibility to poisoning attacks poses a critical security risk. We introduce LoReTTA (Low Resource Two-phase Temporal Attack), a novel adversarial framework on Continuous-Time Dynamic Graphs, which degrades TGNN performance by an average of 29.47% across 4 widely benchmark datasets and 4 State-of-the-Art (SotA) models. LoReTTA operates through a two-stage approach: (1) sparsify the graph by removing high-impact edges using any of the 16 tested temporal importance metrics, (2) strategically replace removed edges with adversarial negatives via LoReTTA's novel degree-preserving negative sampling algorithm. Our plug-and-play design eliminates the need for expensive surrogate models while adhering to realistic unnoticeability constraints. LoReTTA degrades performance by upto 42.0% on MOOC, 31.5% on Wikipedia, 28.8% on UCI, and 15.6% on Enron. LoReTTA outperforms 11 attack baselines, remains undetectable to 4 leading anomaly detection systems, and is robust to 4 SotA adversarial defense training methods, establishing its effectiveness, unnoticeability, and robustness. 

**Abstract (ZH)**: 低资源两阶段时间攻击（LoReTTA）：continuous-time dynamic图上的新型对抗框架 

---
# Transformers Provably Learn Chain-of-Thought Reasoning with Length Generalization 

**Title (ZH)**: Transformers 在长度泛化条件下证明能够学习链式思维推理 

**Authors**: Yu Huang, Zixin Wen, Aarti Singh, Yuejie Chi, Yuxin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.07378)  

**Abstract**: The ability to reason lies at the core of artificial intelligence (AI), and challenging problems usually call for deeper and longer reasoning to tackle. A crucial question about AI reasoning is whether models can extrapolate learned reasoning patterns to solve harder tasks with longer chain-of-thought (CoT). In this work, we present a theoretical analysis of transformers learning on synthetic state-tracking tasks with gradient descent. We mathematically prove how the algebraic structure of state-tracking problems governs the degree of extrapolation of the learned CoT. Specifically, our theory characterizes the length generalization of transformers through the mechanism of attention concentration, linking the retrieval robustness of the attention layer to the state-tracking task structure of long-context reasoning. Moreover, for transformers with limited reasoning length, we prove that a recursive self-training scheme can progressively extend the range of solvable problem lengths. To our knowledge, we provide the first optimization guarantee that constant-depth transformers provably learn $\mathsf{NC}^1$-complete problems with CoT, significantly going beyond prior art confined in $\mathsf{TC}^0$, unless the widely held conjecture $\mathsf{TC}^0 \neq \mathsf{NC}^1$ fails. Finally, we present a broad set of experiments supporting our theoretical results, confirming the length generalization behaviors and the mechanism of attention concentration. 

**Abstract (ZH)**: 人工 Intelligence推理由其核心，并且复杂的任务通常需要更深层次和更长的推理论证来解决。关于AI推理论证的一个关键问题是，模型是否能够将学到的推理论证模式外推到解决更难任务的更长链式思考（CoT）。在本文中，我们对通过梯度下降学习合成状态跟踪任务的变换器进行了理论分析，并从代数结构的角度证明了学习到的CoT的外推程度。具体来说，我们的理论通过注意力集中机制表征变换器的长度泛化，并将注意力层的检索鲁棒性与长上下文推理的状态跟踪任务结构联系起来。此外，对于具有有限推理论证长度的变换器，我们证明了一种递归自训练方案可以逐步扩展可解决的问题长度范围。据我们所知，我们首次提供了优化保证，证明恒定深度变换器能够学习$\mathsf{NC}^1$-完全问题的CoT，显著超越了局限于$\mathsf{TC}^0$的先前研究，除非广泛持有的猜想$\mathsf{TC}^0 \neq \mathsf{NC}^1$不成立。最后，我们展示了广泛的实验支持我们的理论结果，确认了长度泛化行为和注意力集中机制。 

---
# Real-Time LiDAR Super-Resolution via Frequency-Aware Multi-Scale Fusion 

**Title (ZH)**: 基于频率意识多尺度融合的实时LiDAR超分辨率 

**Authors**: June Moh Goo, Zichao Zeng, Jan Boehm  

**Link**: [PDF](https://arxiv.org/pdf/2511.07377)  

**Abstract**: LiDAR super-resolution addresses the challenge of achieving high-quality 3D perception from cost-effective, low-resolution sensors. While recent transformer-based approaches like TULIP show promise, they remain limited to spatial-domain processing with restricted receptive fields. We introduce FLASH (Frequency-aware LiDAR Adaptive Super-resolution with Hierarchical fusion), a novel framework that overcomes these limitations through dual-domain processing. FLASH integrates two key innovations: (i) Frequency-Aware Window Attention that combines local spatial attention with global frequency-domain analysis via FFT, capturing both fine-grained geometry and periodic scanning patterns at log-linear complexity. (ii) Adaptive Multi-Scale Fusion that replaces conventional skip connections with learned position-specific feature aggregation, enhanced by CBAM attention for dynamic feature selection. Extensive experiments on KITTI demonstrate that FLASH achieves state-of-the-art performance across all evaluation metrics, surpassing even uncertainty-enhanced baselines that require multiple forward passes. Notably, FLASH outperforms TULIP with Monte Carlo Dropout while maintaining single-pass efficiency, which enables real-time deployment. The consistent superiority across all distance ranges validates that our dual-domain approach effectively handles uncertainty through architectural design rather than computationally expensive stochastic inference, making it practical for autonomous systems. 

**Abstract (ZH)**: LiDAR超分辨率解决低成本低分辨率传感器实现高品质3D感知的挑战：FLASH（频域意识LiDAR自适应超分辨率与分层融合）框架 

---
# Consistency Is Not Always Correct: Towards Understanding the Role of Exploration in Post-Training Reasoning 

**Title (ZH)**: 一致性并不代表正确性：探索在后训练推理中的作用探究 

**Authors**: Dake Bu, Wei Huang, Andi Han, Atsushi Nitanda, Bo Xue, Qingfu Zhang, Hau-San Wong, Taiji Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2511.07368)  

**Abstract**: Foundation models exhibit broad knowledge but limited task-specific reasoning, motivating post-training strategies such as RLVR and inference scaling with outcome or process reward models (ORM/PRM). While recent work highlights the role of exploration and entropy stability in improving pass@K, empirical evidence points to a paradox: RLVR and ORM/PRM typically reinforce existing tree-like reasoning paths rather than expanding the reasoning scope, raising the question of why exploration helps at all if no new patterns emerge.
To reconcile this paradox, we adopt the perspective of Kim et al. (2025), viewing easy (e.g., simplifying a fraction) versus hard (e.g., discovering a symmetry) reasoning steps as low- versus high-probability Markov transitions, and formalize post-training dynamics through Multi-task Tree-structured Markov Chains (TMC). In this tractable model, pretraining corresponds to tree expansion, while post-training corresponds to chain-of-thought reweighting. We show that several phenomena recently observed in empirical studies arise naturally in this setting: (1) RLVR induces a squeezing effect, reducing reasoning entropy and forgetting some correct paths; (2) population rewards of ORM/PRM encourage consistency rather than accuracy, thereby favoring common patterns; and (3) certain rare, high-uncertainty reasoning paths by the base model are responsible for solving hard problem instances.
Together, these explain why exploration -- even when confined to the base model's reasoning scope -- remains essential: it preserves access to rare but crucial reasoning traces needed for difficult cases, which are squeezed out by RLVR or unfavored by inference scaling. Building on this, we further show that exploration strategies such as rejecting easy instances and KL regularization help preserve rare reasoning traces. Empirical simulations corroborate our theoretical results. 

**Abstract (ZH)**: 基础模型展现出广泛的知识但受限于特定任务的推理能力，促使研究采用如RLVR和基于结果或过程奖励模型（ORM/PRM）的后训练策略。虽然近期工作强调了探索和熵稳定性在提高pass@K方面的角色，但经验证据揭示了一个悖论：RLVR和ORM/PRM通常强化现有类似树状的推理路径而非扩展推理范围，这引发了探索为何有效的问题，尤其是没有新的模式出现的情况下。

为了解释这一悖论，我们采纳了Kim等人（2025）的观点，将容易的（例如，简化分数）与困难的（例如，发现对称性）推理步骤视为低概率与高概率马尔可夫转移，并通过多任务树状马尔可夫链（TMC）形式化后训练动态。在这种可处理的模型中，预训练对应于树的扩展，而后训练对应于推理路径的重新权重。我们展示了一些近期在经验研究中观察到的现象在此设定中自然地出现：（1）RLVR产生压缩效应，减少推理熵并忘记一些正确的路径；（2）ORM/PRM的人群奖励鼓励一致性而非准确性，从而有利于常见模式；（3）基础模型中某些罕见、高不确定性的推理路径对解决困难问题实例至关重要。

这些共同解释了为何即使探索局限于基础模型的推理范围内，探索仍保持关键：它保存了艰难案例所需的稀有但关键的推理痕迹，这些痕迹被RLVR挤出或被推理扩展所忽视。在此基础上，我们进一步展示拒绝简单实例和KL正则化等探索策略有助于保存稀有推理痕迹。经验模拟证实了我们的理论结果。 

---
# Machine-Learning Accelerated Calculations of Reduced Density Matrices 

**Title (ZH)**: 机器学习加速的密度矩阵计算 

**Authors**: Awwab A. Azam, Lexu Zhao, Jiabin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07367)  

**Abstract**: $n$-particle reduced density matrices ($n$-RDMs) play a central role in understanding correlated phases of matter. Yet the calculation of $n$-RDMs is often computationally inefficient for strongly-correlated states, particularly when the system sizes are large. In this work, we propose to use neural network (NN) architectures to accelerate the calculation of, and even predict, the $n$-RDMs for large-size systems. The underlying intuition is that $n$-RDMs are often smooth functions over the Brillouin zone (BZ) (certainly true for gapped states) and are thus interpolable, allowing NNs trained on small-size $n$-RDMs to predict large-size ones. Building on this intuition, we devise two NNs: (i) a self-attention NN that maps random RDMs to physical ones, and (ii) a Sinusoidal Representation Network (SIREN) that directly maps momentum-space coordinates to RDM values. We test the NNs in three 2D models: the pair-pair correlation functions of the Richardson model of superconductivity, the translationally-invariant 1-RDM in a four-band model with short-range repulsion, and the translation-breaking 1-RDM in the half-filled Hubbard model. We find that a SIREN trained on a $6\times 6$ momentum mesh can predict the $18\times 18$ pair-pair correlation function with a relative accuracy of $0.839$. The NNs trained on $6\times 6 \sim 8\times 8$ meshes can provide high-quality initial guesses for $50\times 50$ translation-invariant Hartree-Fock (HF) and $30\times 30$ fully translation-breaking-allowed HF, reducing the number of iterations required for convergence by up to $91.63\%$ and $92.78\%$, respectively, compared to random initializations. Our results illustrate the potential of using NN-based methods for interpolable $n$-RDMs, which might open a new avenue for future research on strongly correlated phases. 

**Abstract (ZH)**: $n$粒子约化密度矩阵的神经网络加速计算及其预测 

---
# Self-Evaluating LLMs for Multi-Step Tasks: Stepwise Confidence Estimation for Failure Detection 

**Title (ZH)**: 自评估大语言模型用于多步任务：逐步-confidence估计以检测失败 

**Authors**: Vaibhav Mavi, Shubh Jaroria, Weiqi Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.07364)  

**Abstract**: Reliability and failure detection of large language models (LLMs) is critical for their deployment in high-stakes, multi-step reasoning tasks. Prior work explores confidence estimation for self-evaluating LLM-scorer systems, with confidence scorers estimating the likelihood of errors in LLM responses. However, most methods focus on single-step outputs and overlook the challenges of multi-step reasoning. In this work, we extend self-evaluation techniques to multi-step tasks, testing two intuitive approaches: holistic scoring and step-by-step scoring. Using two multi-step benchmark datasets, we show that stepwise evaluation generally outperforms holistic scoring in detecting potential errors, with up to 15% relative increase in AUC-ROC. Our findings demonstrate that self-evaluating LLM systems provide meaningful confidence estimates in complex reasoning, improving their trustworthiness and providing a practical framework for failure detection. 

**Abstract (ZH)**: 大型语言模型（LLMs）的可靠性和故障检测对于它们在高风险多步推理任务中的部署至关重要。前期工作探讨了自评估LLM评分系统中的置信度估计，其中置信度评分器估算LLM响应中错误的可能性。然而，大多数方法专注于单步输出并忽视了多步推理的挑战。在本工作中，我们将自评估技术扩展到多步任务，测试了两种直观的方法：整体评分和逐步评分。使用两个多步基准数据集，我们表明逐步评估通常在检测潜在错误方面优于整体评分，AUC-ROC相对增加高达15%。我们的研究结果表明，自评估LLM系统在复杂推理中提供了有意义的置信度估计，提高了其可信度，并为故障检测提供了实用框架。 

---
# Inference-Time Scaling of Diffusion Models for Infrared Data Generation 

**Title (ZH)**: 红外数据生成的推断时缩放扩散模型 

**Authors**: Kai A. Horstmann, Maxim Clouser, Kia Khezeli  

**Link**: [PDF](https://arxiv.org/pdf/2511.07362)  

**Abstract**: Infrared imagery enables temperature-based scene understanding using passive sensors, particularly under conditions of low visibility where traditional RGB imaging fails. Yet, developing downstream vision models for infrared applications is hindered by the scarcity of high-quality annotated data, due to the specialized expertise required for infrared annotation. While synthetic infrared image generation has the potential to accelerate model development by providing large-scale, diverse training data, training foundation-level generative diffusion models in the infrared domain has remained elusive due to limited datasets. In light of such data constraints, we explore an inference-time scaling approach using a domain-adapted CLIP-based verifier for enhanced infrared image generation quality. We adapt FLUX.1-dev, a state-of-the-art text-to-image diffusion model, to the infrared domain by finetuning it on a small sample of infrared images using parameter-efficient techniques. The trained verifier is then employed during inference to guide the diffusion sampling process toward higher quality infrared generations that better align with input text prompts. Empirically, we find that our approach leads to consistent improvements in generation quality, reducing FID scores on the KAIST Multispectral Pedestrian Detection Benchmark dataset by 10% compared to unguided baseline samples. Our results suggest that inference-time guidance offers a promising direction for bridging the domain gap in low-data infrared settings. 

**Abstract (ZH)**: 红外成像利用被动传感器实现基于温度的场景理解，特别是在低能见度条件下，传统RGB成像失效的情况。然而，由于需要专门的红外标注 expertise，开发针对红外应用的下游视觉模型受到了高质量标注数据稀缺的阻碍。尽管合成红外图像生成有可能通过提供大规模、多样化的训练数据来加速模型开发，但在红外领域的基础生成扩散模型训练仍因数据集有限而难以实现。鉴于数据约束，我们探索了一种推理时缩放方法，利用一个领域适应的CLIP基验真器来提高红外图像生成质量。我们通过参数效率技术对FLUX.1-dev这一最先进的文本到图像扩散模型进行微调，使其适用于红外领域。训练后的验真器随后在推理过程中用于引导扩散采样过程，产生与输入文本提示更好地对齐的高质量红外生成结果。实验结果显示，我们的方法在生成质量上持续改善，与未指导基线样本相比，将KAIST多光谱行人检测基准数据集上的FID分数降低了10%。我们的结果表明，在低数据红外设置中，推理时的指导为弥合领域差距提供了有前景的方向。 

---
# TNT: Improving Chunkwise Training for Test-Time Memorization 

**Title (ZH)**: TNT: 提升分块训练以增强测试时记忆效果 

**Authors**: Zeman Li, Ali Behrouz, Yuan Deng, Peilin Zhong, Praneeth Kacham, Mahdi Karami, Meisam Razaviyayn, Vahab Mirrokni  

**Link**: [PDF](https://arxiv.org/pdf/2511.07343)  

**Abstract**: Recurrent neural networks (RNNs) with deep test-time memorization modules, such as Titans and TTT, represent a promising, linearly-scaling paradigm distinct from Transformers. While these expressive models do not yet match the peak performance of state-of-the-art Transformers, their potential has been largely untapped due to prohibitively slow training and low hardware utilization. Existing parallelization methods force a fundamental conflict governed by the chunksize hyperparameter: large chunks boost speed but degrade performance, necessitating a fixed, suboptimal compromise. To solve this challenge, we introduce TNT, a novel training paradigm that decouples training efficiency from inference performance through a two-stage process. Stage one is an efficiency-focused pre-training phase utilizing a hierarchical memory. A global module processes large, hardware-friendly chunks for long-range context, while multiple parallel local modules handle fine-grained details. Crucially, by periodically resetting local memory states, we break sequential dependencies to enable massive context parallelization. Stage two is a brief fine-tuning phase where only the local memory modules are adapted to a smaller, high-resolution chunksize, maximizing accuracy with minimal overhead. Evaluated on Titans and TTT models, TNT achieves a substantial acceleration in training speed-up to 17 times faster than the most accurate baseline configuration - while simultaneously improving model accuracy. This improvement removes a critical scalability barrier, establishing a practical foundation for developing expressive RNNs and facilitating future work to close the performance gap with Transformers. 

**Abstract (ZH)**: 带有深度测试时记忆模块的循环神经网络（RNNs）如Titans和TTT，代表了一种与Transformer不同的、线性扩展的有希望的新范式。通过两阶段过程将训练效率与推断性能脱钩，引入TNT训练范式。 

---
# Grounding Computer Use Agents on Human Demonstrations 

**Title (ZH)**: 基于人类示范地约束计算机使用代理 

**Authors**: Aarash Feizi, Shravan Nayak, Xiangru Jian, Kevin Qinghong Lin, Kaixin Li, Rabiul Awal, Xing Han Lù, Johan Obando-Ceron, Juan A. Rodriguez, Nicolas Chapados, David Vazquez, Adriana Romero-Soriano, Reihaneh Rabbany, Perouz Taslakian, Christopher Pal, Spandana Gella, Sai Rajeswar  

**Link**: [PDF](https://arxiv.org/pdf/2511.07332)  

**Abstract**: Building reliable computer-use agents requires grounding: accurately connecting natural language instructions to the correct on-screen elements. While large datasets exist for web and mobile interactions, high-quality resources for desktop environments are limited. To address this gap, we introduce GroundCUA, a large-scale desktop grounding dataset built from expert human demonstrations. It covers 87 applications across 12 categories and includes 56K screenshots, with every on-screen element carefully annotated for a total of over 3.56M human-verified annotations. From these demonstrations, we generate diverse instructions that capture a wide range of real-world tasks, providing high-quality data for model training. Using GroundCUA, we develop the GroundNext family of models that map instructions to their target UI elements. At both 3B and 7B scales, GroundNext achieves state-of-the-art results across five benchmarks using supervised fine-tuning, while requiring less than one-tenth the training data of prior work. Reinforcement learning post-training further improves performance, and when evaluated in an agentic setting on the OSWorld benchmark using o3 as planner, GroundNext attains comparable or superior results to models trained with substantially more data,. These results demonstrate the critical role of high-quality, expert-driven datasets in advancing general-purpose computer-use agents. 

**Abstract (ZH)**: 构建可靠的计算机使用代理需要接地：准确地将自然语言指令与正确的屏幕元素连接起来。尽管存在大量的网络和移动交互数据集，桌面环境的高质量资源仍然有限。为了解决这一差距，我们引入了GroundCUA，这是一个从专家人工演示构建的大规模桌面接地数据集。它涵盖了12个类别中的87个应用程序，并包含了56K截图，每个屏幕元素都经过细致标注，总共有超过3.56M的人工验证注释。从这些演示中，我们生成了多种多样的指令，涵盖了广泛的实际任务，为模型训练提供了高质量的数据。使用GroundCUA，我们开发了GroundNext家族的模型，将指令映射到其目标UI元素。无论是3B参数还是7B参数规模，GroundNext在五个基准测试中均实现了最先进的结果，同时所需的训练数据量仅为以前工作的十分之一。强化学习后训练进一步提高了性能，并在使用o3作为规划者的OSWorld基准测试中评估时，GroundNext在训练数据量显著较少的情况下获得了具有竞争力或更优的结果。这些结果表明，高质量的专家驱动数据集在推动通用计算机使用代理方面起着关键作用。 

---
# FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation 

**Title (ZH)**: FinRpt: 股票研究报告数据集、评价系统及基于LLM的多代理框架 

**Authors**: Song Jin, Shuqi Li, Shukun Zhang, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2511.07322)  

**Abstract**: While LLMs have shown great success in financial tasks like stock prediction and question answering, their application in fully automating Equity Research Report generation remains uncharted territory. In this paper, we formulate the Equity Research Report (ERR) Generation task for the first time. To address the data scarcity and the evaluation metrics absence, we present an open-source evaluation benchmark for ERR generation - FinRpt. We frame a Dataset Construction Pipeline that integrates 7 financial data types and produces a high-quality ERR dataset automatically, which could be used for model training and evaluation. We also introduce a comprehensive evaluation system including 11 metrics to assess the generated ERRs. Moreover, we propose a multi-agent framework specifically tailored to address this task, named FinRpt-Gen, and train several LLM-based agents on the proposed datasets using Supervised Fine-Tuning and Reinforcement Learning. Experimental results indicate the data quality and metrics effectiveness of the benchmark FinRpt and the strong performance of FinRpt-Gen, showcasing their potential to drive innovation in the ERR generation field. All code and datasets are publicly available. 

**Abstract (ZH)**: LLMs在金融任务如股票预测和问答中取得了显著成功，但在全自动生产 equity research report 方面的应用仍是一片未开发的领域。本文首次提出了 equity research report (ERR) 生成任务。为了解决数据稀缺和评价指标缺失的问题，我们提出了一个开源的 ERR 生成评估基准——FinRpt。我们构建了一个数据集构建管道，整合了7种金融数据类型，自动生成高质量的 ERR 数据集，可用于模型训练和评估。我们还引入了一个综合评估系统，包括11个指标来评估生成的 ERR。此外，我们提出了一种专门针对此任务的多代理框架，命名为 FinRpt-Gen，并使用有监督微调和强化学习在提出的数据集上训练了多个基于 LLM 的代理。实验结果显示基准 FinRpt 的数据质量和评估指标的有效性，以及 FinRpt-Gen 强大的性能，展示了其在 ERR 生成领域推动创新的潜力。所有代码和数据集均开源。 

---
# When Bias Pretends to Be Truth: How Spurious Correlations Undermine Hallucination Detection in LLMs 

**Title (ZH)**: 当偏差伪装成真相：虚假相关性如何削弱LLMs中的幻觉检测 

**Authors**: Shaowen Wang, Yiqi Dong, Ruinian Chang, Tansheng Zhu, Yuebo Sun, Kaifeng Lyu, Jian Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.07318)  

**Abstract**: Despite substantial advances, large language models (LLMs) continue to exhibit hallucinations, generating plausible yet incorrect responses. In this paper, we highlight a critical yet previously underexplored class of hallucinations driven by spurious correlations -- superficial but statistically prominent associations between features (e.g., surnames) and attributes (e.g., nationality) present in the training data. We demonstrate that these spurious correlations induce hallucinations that are confidently generated, immune to model scaling, evade current detection methods, and persist even after refusal fine-tuning. Through systematically controlled synthetic experiments and empirical evaluations on state-of-the-art open-source and proprietary LLMs (including GPT-5), we show that existing hallucination detection methods, such as confidence-based filtering and inner-state probing, fundamentally fail in the presence of spurious correlations. Our theoretical analysis further elucidates why these statistical biases intrinsically undermine confidence-based detection techniques. Our findings thus emphasize the urgent need for new approaches explicitly designed to address hallucinations caused by spurious correlations. 

**Abstract (ZH)**: 尽管取得了显著进步，大规模语言模型（LLMs）仍然会出现幻觉，生成虽然合理但错误的响应。在本文中，我们强调了一类之前尚未充分探索的幻觉——由虚假相关性驱动的幻觉——这些虚假相关性在训练数据中表现为表象上的但统计上显著的特征（如姓氏）与属性（如国籍）之间的关联。我们展示了这些虚假相关性导致引人自信地生成的幻觉，不受模型规模扩增的影响，能够避开当前的检测方法，并且即使在拒绝微调后依然存在。通过系统控制的合成实验和对最先进的开源和专有LLM（包括GPT-5）的实证评估，我们证明了现有幻觉检测方法（如基于置信度的过滤和内状态探针）在存在虚假相关性时根本无效。我们的理论分析进一步阐明了为什么这些统计偏差内在地削弱了基于置信度的检测技术的有效性。因此，我们的发现强调了迫切需要专门设计的新方法来应对由虚假相关性引起的幻觉。 

---
# Superhuman AI for Stratego Using Self-Play Reinforcement Learning and Test-Time Search 

**Title (ZH)**: 使用自我对弈强化学习和测试时搜索的超级人机 Stratego 人工智能 

**Authors**: Samuel Sokota, Eugene Vinitsky, Hengyuan Hu, J. Zico Kolter, Gabriele Farina  

**Link**: [PDF](https://arxiv.org/pdf/2511.07312)  

**Abstract**: Few classical games have been regarded as such significant benchmarks of artificial intelligence as to have justified training costs in the millions of dollars. Among these, Stratego -- a board wargame exemplifying the challenge of strategic decision making under massive amounts of hidden information -- stands apart as a case where such efforts failed to produce performance at the level of top humans. This work establishes a step change in both performance and cost for Stratego, showing that it is now possible not only to reach the level of top humans, but to achieve vastly superhuman level -- and that doing so requires not an industrial budget, but merely a few thousand dollars. We achieved this result by developing general approaches for self-play reinforcement learning and test-time search under imperfect information. 

**Abstract (ZH)**: 经典的少数几个棋类游戏被认为是对人工智能具有如此重要意义的基准，以至于它们的训练成本达到了数百万美元。在这些游戏中，Stratego——一款突显在大量隐藏信息下进行战略决策挑战的棋盘战争游戏——是其中一项努力未能达到顶尖人类水平的案例。本研究在Stratego的表现和成本上实现了质的飞跃，显示现在的技术水平不仅能够达到顶尖人类的水平，还能达到超人类的水平——这需要的不再是工业级别的预算，而仅仅几千美元。我们通过开发一般性的自我对弈强化学习和不完美信息下的测试时搜索方法实现了这一结果。 

---
# Beyond Boundaries: Leveraging Vision Foundation Models for Source-Free Object Detection 

**Title (ZH)**: 超越界限：利用视觉基础模型进行无源目标检测 

**Authors**: Huizai Yao, Sicheng Zhao, Pengteng Li, Yi Cui, Shuo Lu, Weiyu Guo, Yunfan Lu, Yijie Xu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2511.07301)  

**Abstract**: Source-Free Object Detection (SFOD) aims to adapt a source-pretrained object detector to a target domain without access to source data. However, existing SFOD methods predominantly rely on internal knowledge from the source model, which limits their capacity to generalize across domains and often results in biased pseudo-labels, thereby hindering both transferability and discriminability. In contrast, Vision Foundation Models (VFMs), pretrained on massive and diverse data, exhibit strong perception capabilities and broad generalization, yet their potential remains largely untapped in the SFOD setting. In this paper, we propose a novel SFOD framework that leverages VFMs as external knowledge sources to jointly enhance feature alignment and label quality. Specifically, we design three VFM-based modules: (1) Patch-weighted Global Feature Alignment (PGFA) distills global features from VFMs using patch-similarity-based weighting to enhance global feature transferability; (2) Prototype-based Instance Feature Alignment (PIFA) performs instance-level contrastive learning guided by momentum-updated VFM prototypes; and (3) Dual-source Enhanced Pseudo-label Fusion (DEPF) fuses predictions from detection VFMs and teacher models via an entropy-aware strategy to yield more reliable supervision. Extensive experiments on six benchmarks demonstrate that our method achieves state-of-the-art SFOD performance, validating the effectiveness of integrating VFMs to simultaneously improve transferability and discriminability. 

**Abstract (ZH)**: 无源域对象检测（Source-Free Object Detection, SFOD）旨在不访问源数据的情况下，将源自预训练的对象检测器适应目标域。然而，现有的SFOD方法主要依赖源模型内部的知识，这限制了其跨域泛化能力，并往往导致有偏的伪标签，从而阻碍了其可迁移性和可判别性。相比之下，视觉基础模型（Visual Foundation Models, VFMs）在大量多样化的数据上进行预训练，展示了强大的感知能力及广泛的泛化能力，但在SFOD场景中的潜力尚未充分利用。在本文中，我们提出了一种新的SFOD框架，利用VFMs作为外部知识源，共同增强特征对齐和标签质量。具体地，我们设计了三个基于VFMs的模块：（1）基于块相似性的全局特征对齐（Patch-weighted Global Feature Alignment, PGFA）通过块相似性加权提取全局特征，增强全球特征的可迁移性；（2）基于原型的实例特征对齐（Prototype-based Instance Feature Alignment, PIFA）通过由动量更新的VFM原型指导的实例级对比学习进行特征对齐；（3）双源增强伪标签融合（Dual-source Enhanced Pseudo-label Fusion, DEPF）通过一种熵感知策略融合检测VFMs和教师模型的预测，以生成更可靠监督。在六个基准上的广泛实验表明，我们的方法在SFOD性能上达到了最先进的水平，证明了集成VFMs以同时改善可迁移性和可判别性的有效性。 

---
# LMM-IQA: Image Quality Assessment for Low-Dose CT Imaging 

**Title (ZH)**: LMM-IQA: 低剂量CT影像质量评估 

**Authors**: Kagan Celik, Mehmet Ozan Unal, Metin Ertas, Isa Yildirim  

**Link**: [PDF](https://arxiv.org/pdf/2511.07298)  

**Abstract**: Low-dose computed tomography (CT) represents a significant improvement in patient safety through lower radiation doses, but increased noise, blur, and contrast loss can diminish diagnostic quality. Therefore, consistency and robustness in image quality assessment become essential for clinical applications. In this study, we propose an LLM-based quality assessment system that generates both numerical scores and textual descriptions of degradations such as noise, blur, and contrast loss. Furthermore, various inference strategies - from the zero-shot approach to metadata integration and error feedback - are systematically examined, demonstrating the progressive contribution of each method to overall performance. The resultant assessments yield not only highly correlated scores but also interpretable output, thereby adding value to clinical workflows. The source codes of our study are available at this https URL. 

**Abstract (ZH)**: 基于LLM的低剂量CT图像质量评估系统：从噪声、模糊和对比度损失生成数值评分和文本描述，并系统性地评估多种推理策略对整体性能的贡献 

---
# Hard vs. Noise: Resolving Hard-Noisy Sample Confusion in Recommender Systems via Large Language Models 

**Title (ZH)**: Hard vs. Noise: 通过大型语言模型解决推荐系统中的硬样例与噪声样例混淆问题 

**Authors**: Tianrui Song, Wen-Shuo Chao, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07295)  

**Abstract**: Implicit feedback, employed in training recommender systems, unavoidably confronts noise due to factors such as misclicks and position bias. Previous studies have attempted to identify noisy samples through their diverged data patterns, such as higher loss values, and mitigate their influence through sample dropping or reweighting. However, we observed that noisy samples and hard samples display similar patterns, leading to hard-noisy confusion issue. Such confusion is problematic as hard samples are vital for modeling user preferences. To solve this problem, we propose LLMHNI framework, leveraging two auxiliary user-item relevance signals generated by Large Language Models (LLMs) to differentiate hard and noisy samples. LLMHNI obtains user-item semantic relevance from LLM-encoded embeddings, which is used in negative sampling to select hard negatives while filtering out noisy false negatives. An objective alignment strategy is proposed to project LLM-encoded embeddings, originally for general language tasks, into a representation space optimized for user-item relevance modeling. LLMHNI also exploits LLM-inferred logical relevance within user-item interactions to identify hard and noisy samples. These LLM-inferred interactions are integrated into the interaction graph and guide denoising with cross-graph contrastive alignment. To eliminate the impact of unreliable interactions induced by LLM hallucination, we propose a graph contrastive learning strategy that aligns representations from randomly edge-dropped views to suppress unreliable edges. Empirical results demonstrate that LLMHNI significantly improves denoising and recommendation performance. 

**Abstract (ZH)**: LLMHNI框架：利用大型语言模型辅助信号区分困难样本和噪声样本 

---
# Verifying rich robustness properties for neural networks 

**Title (ZH)**: 验证神经网络的丰富鲁棒性属性 

**Authors**: Mohammad Afzal, S. Akshay, Ashutosh Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2511.07293)  

**Abstract**: Robustness is a important problem in AI alignment and safety, with models such as neural networks being increasingly used in safety-critical systems. In the last decade, a large body of work has emerged on local robustness, i.e., checking if the decision of a neural network remains unchanged when the input is slightly perturbed. However, many of these approaches require specialized encoding and often ignore the confidence of a neural network on its output. In this paper, our goal is to build a generalized framework to specify and verify variants of robustness in neural network verification. We propose a specification framework using a simple grammar, which is flexible enough to capture most existing variants. This allows us to introduce new variants of robustness that take into account the confidence of the neural network in its outputs. Next, we develop a novel and powerful unified technique to verify all such variants in a homogeneous way, viz., by adding a few additional layers to the neural network. This enables us to use any state-of-the-art neural network verification tool, without having to tinker with the encoding within, while incurring an approximation error that we show is bounded. We perform an extensive experimental evaluation over a large suite of 8870 benchmarks having 138M parameters in a largest network, and show that we are able to capture a wide set of robustness variants and outperform direct encoding approaches by a significant margin. 

**Abstract (ZH)**: Robust性是AI对齐和安全中的一个重要问题，随着如神经网络等模型在关键安全系统中的应用日益广泛，这一问题变得越来越突出。在过去十年中，关于局部Robust性的大量研究工作涌现出来，即检查在输入轻微扰动的情况下，神经网络的决策是否保持不变。然而，许多这些方法需要专门的编码并经常忽略了神经网络对其输出的置信度。在这篇论文中，我们的目标是构建一个通用框架，以指定和验证神经网络验证中的各种Robust性形式。我们提出了一种使用简单语法的规范框架，该框架足够灵活以捕获大多数现有形式。这使我们能够引入新的考虑神经网络输出置信度的Robust性变体。然后，我们开发了一种新颖而强大的统一验证技术，以同一种方式验证所有这些变体，即通过向神经网络添加少量额外层。这使我们能够使用任何最先进的神经网络验证工具，而不需要对内部编码进行修改，并且产生的近似误差是可限定的。我们对具有最多138M参数的8870个基准进行了广泛实验评估，并表明我们能够捕获广泛的Robust性变体，并在显著的性能上超越直接编码方法。 

---
# Enabling Off-Policy Imitation Learning with Deep Actor Critic Stabilization 

**Title (ZH)**: 基于深度actor-critic稳定性的离策模仿学习-enable版本 

**Authors**: Sayambhu Sen, Shalabh Bhatnagar  

**Link**: [PDF](https://arxiv.org/pdf/2511.07288)  

**Abstract**: Learning complex policies with Reinforcement Learning (RL) is often hindered by instability and slow convergence, a problem exacerbated by the difficulty of reward engineering. Imitation Learning (IL) from expert demonstrations bypasses this reliance on rewards. However, state-of-the-art IL methods, exemplified by Generative Adversarial Imitation Learning (GAIL)Ho et. al, suffer from severe sample inefficiency. This is a direct consequence of their foundational on-policy algorithms, such as TRPO Schulman this http URL. In this work, we introduce an adversarial imitation learning algorithm that incorporates off-policy learning to improve sample efficiency. By combining an off-policy framework with auxiliary techniques specifically, double Q network based stabilization and value learning without reward function inference we demonstrate a reduction in the samples required to robustly match expert behavior. 

**Abstract (ZH)**: 使用强化学习（RL）学习复杂策略常常受到不稳定性及慢收敛的困扰，这一问题因奖励工程的困难而加剧。通过专家演示进行模仿学习（IL）可绕过对奖励的依赖。然而，最先进的IL方法，如生成式对抗模仿学习（GAIL）等，表现出严重的样本效率低下。这是其基于在线策略算法（如TRPO）的结果。在本研究中，我们提出了一种结合离策学习的对抗模仿学习算法，以提高样本效率。通过将离策框架与特定辅助技术相结合，特别是基于双Q网络的稳定化技术和无需推断奖励函数的价值学习，我们展示了在保持对专家行为稳健匹配所需的样本数量减少。 

---
# Glioma C6: A Novel Dataset for Training and Benchmarking Cell Segmentation 

**Title (ZH)**: 胶质瘤C6：一种用于细胞分割训练和基准测试的新数据集 

**Authors**: Roman Malashin, Svetlana Pashkevich, Daniil Ilyukhin, Arseniy Volkov, Valeria Yachnaya, Andrey Denisov, Maria Mikhalkova  

**Link**: [PDF](https://arxiv.org/pdf/2511.07286)  

**Abstract**: We present Glioma C6, a new open dataset for instance segmentation of glioma C6 cells, designed as both a benchmark and a training resource for deep learning models. The dataset comprises 75 high-resolution phase-contrast microscopy images with over 12,000 annotated cells, providing a realistic testbed for biomedical image analysis. It includes soma annotations and morphological cell categorization provided by biologists. Additional categorization of cells, based on morphology, aims to enhance the utilization of image data for cancer cell research. Glioma C6 consists of two parts: the first is curated with controlled parameters for benchmarking, while the second supports generalization testing under varying conditions. We evaluate the performance of several generalist segmentation models, highlighting their limitations on our dataset. Our experiments demonstrate that training on Glioma C6 significantly enhances segmentation performance, reinforcing its value for developing robust and generalizable models. The dataset is publicly available for researchers. 

**Abstract (ZH)**: Glioma C6：用于胶质瘤C6细胞实例分割的新开源数据集及训练资源 

---
# Designing Beyond Language: Sociotechnical Barriers in AI Health Technologies for Limited English Proficiency 

**Title (ZH)**: 超出语言设计：面向有限英语能力者的AI健康技术的社会技术壁垒 

**Authors**: Michelle Huang, Violeta J. Rodriguez, Koustuv Saha, Tal August  

**Link**: [PDF](https://arxiv.org/pdf/2511.07277)  

**Abstract**: Limited English proficiency (LEP) patients in the U.S. face systemic barriers to healthcare beyond language and interpreter access, encompassing procedural and institutional constraints. AI advances may support communication and care through on-demand translation and visit preparation, but also risk exacerbating existing inequalities. We conducted storyboard-driven interviews with 14 patient navigators to explore how AI could shape care experiences for Spanish-speaking LEP individuals. We identified tensions around linguistic and cultural misunderstandings, privacy concerns, and opportunities and risks for AI to augment care workflows. Participants highlighted structural factors that can undermine trust in AI systems, including sensitive information disclosure, unstable technology access, and low digital literacy. While AI tools can potentially alleviate social barriers and institutional constraints, there are risks of misinformation and uprooting human camaraderie. Our findings contribute design considerations for AI that support LEP patients and care teams via rapport-building, education, and language support, and minimizing disruptions to existing practices. 

**Abstract (ZH)**: 有限英语 proficiency (LEP) 患者在美国面对的语言和解释员访问之外的系统性 healthcare 障碍，涵盖程序和机构限制。AI 进步可能通过即时翻译和就诊准备来支持沟通和护理，但也可能加剧现有的不平等现象。我们通过故事板引导的访谈与14名患者导航员探讨了AI如何塑造西班牙语 LEP 个体的护理体验。我们识别出了语言和文化误解、隐私担忧以及AI增强护理工作流的机会与风险。参与者指出了可能破坏对AI系统信任的结构性因素，包括敏感信息的披露、不稳定的技术访问以及低数字素养。虽然AI工具有可能减轻社会障碍和制度性限制，但也存在传播错误信息和拆解人与人之间友谊的风险。我们的发现为通过增强互动、教育和语言支持来支持LEP患者和护理团队设计AI提供了考虑因素，并尽量减少对现有实践的干扰。 

---
# MVU-Eval: Towards Multi-Video Understanding Evaluation for Multimodal LLMs 

**Title (ZH)**: MVU-Eval: 向多视频理解评估 multimodal LLMs 迈进 

**Authors**: Tianhao Peng, Haochen Wang, Yuanxing Zhang, Zekun Wang, Zili Wang, Ge Zhang, Jian Yang, Shihao Li, Yanghai Wang, Xintao Wang, Houyi Li, Wei Ji, Pengfei Wan, Wenhao Huang, Zhaoxiang Zhang, Jiaheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07250)  

**Abstract**: The advent of Multimodal Large Language Models (MLLMs) has expanded AI capabilities to visual modalities, yet existing evaluation benchmarks remain limited to single-video understanding, overlooking the critical need for multi-video understanding in real-world scenarios (e.g., sports analytics and autonomous driving). To address this significant gap, we introduce MVU-Eval, the first comprehensive benchmark for evaluating Multi-Video Understanding for MLLMs. Specifically, our MVU-Eval mainly assesses eight core competencies through 1,824 meticulously curated question-answer pairs spanning 4,959 videos from diverse domains, addressing both fundamental perception tasks and high-order reasoning tasks. These capabilities are rigorously aligned with real-world applications such as multi-sensor synthesis in autonomous systems and cross-angle sports analytics. Through extensive evaluation of state-of-the-art open-source and closed-source models, we reveal significant performance discrepancies and limitations in current MLLMs' ability to perform understanding across multiple videos. The benchmark will be made publicly available to foster future research. 

**Abstract (ZH)**: Multimodal Large Language Models的出现扩展了AI的能力至视觉模态，现有评估基准仍然局限于单视频理解，忽视了在实际场景中多视频理解的迫切需求（例如，体育分析和自动驾驶）。为解决这一重大缺口，我们介绍了MVU-Eval，这是首个全面评估MLLMs多视频理解能力的基准。具体而言，我们的MVU-Eval通过1,824个精心策划的问题-答案对评估了来自不同领域的4,959个视频的信息，涵盖了从基本感知任务到高层次推理任务的核心能力。这些能力严格对应于多传感器融合在自主系统中的实际应用以及跨视角体育分析。通过广泛评估最先进的开源和闭源模型，我们揭示了当前MLLMs在多视频理解方面表现的显著差异和局限性。该基准将公开发布，以促进未来的研究。 

---
# Leveraging Text-Driven Semantic Variation for Robust OOD Segmentation 

**Title (ZH)**: 基于文本驱动的语义变化实现稳健的OOD分割 

**Authors**: Seungheon Song, Jaekoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.07238)  

**Abstract**: In autonomous driving and robotics, ensuring road safety and reliable decision-making critically depends on out-of-distribution (OOD) segmentation. While numerous methods have been proposed to detect anomalous objects on the road, leveraging the vision-language space-which provides rich linguistic knowledge-remains an underexplored field. We hypothesize that incorporating these linguistic cues can be especially beneficial in the complex contexts found in real-world autonomous driving scenarios.
To this end, we present a novel approach that trains a Text-Driven OOD Segmentation model to learn a semantically diverse set of objects in the vision-language space. Concretely, our approach combines a vision-language model's encoder with a transformer decoder, employs Distance-Based OOD prompts located at varying semantic distances from in-distribution (ID) classes, and utilizes OOD Semantic Augmentation for OOD representations. By aligning visual and textual information, our approach effectively generalizes to unseen objects and provides robust OOD segmentation in diverse driving environments.
We conduct extensive experiments on publicly available OOD segmentation datasets such as Fishyscapes, Segment-Me-If-You-Can, and Road Anomaly datasets, demonstrating that our approach achieves state-of-the-art performance across both pixel-level and object-level evaluations. This result underscores the potential of vision-language-based OOD segmentation to bolster the safety and reliability of future autonomous driving systems. 

**Abstract (ZH)**: 在自主驾驶与机器人领域，确保道路安全和可靠的决策制定关键依赖于离分布（OOD）分割。虽然已提出了许多方法来检测道路上的异常物体，但利用视觉-语言空间——提供了丰富的语言知识——这一领域仍被严重忽视。我们假设，在现实世界自主驾驶场景中发现的复杂环境中，整合这些语言线索可以特别有益。

为此，我们提出了一种新的方法，通过训练文本驱动的OOD分割模型，在视觉-语言空间中学习语义多样的对象。具体而言，我们的方法将视觉-语言模型的编码器与变压器解码器相结合，使用基于距离的OOD提示，这些提示位于与分布（ID）类别不同语义距离的位置，并利用OOD语义增强来获取OOD表示。通过对视觉和文本信息进行对齐，我们的方法能够有效泛化到未见过的对象，并在多种驾驶环境中提供稳健的OOD分割。

我们在Fishyscapes、Segment-Me-If-You-Can和Road Anomaly等公开的数据集上进行了广泛的实验，证明我们的方法在像素级和对象级评估中均达到了最先进的性能。这一结果突显了基于视觉-语言的OOD分割在未来自主驾驶系统中的潜力。 

---
# Discourse Graph Guided Document Translation with Large Language Models 

**Title (ZH)**: 大型语言模型引导的discourse图指导文档翻译 

**Authors**: Viet-Thanh Pham, Minghan Wang, Hao-Han Liao, Thuy-Trang Vu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07230)  

**Abstract**: Adapting large language models to full document translation remains challenging due to the difficulty of capturing long-range dependencies and preserving discourse coherence throughout extended texts. While recent agentic machine translation systems mitigate context window constraints through multi-agent orchestration and persistent memory, they require substantial computational resources and are sensitive to memory retrieval strategies. We introduce TransGraph, a discourse-guided framework that explicitly models inter-chunk relationships through structured discourse graphs and selectively conditions each translation segment on relevant graph neighbourhoods rather than relying on sequential or exhaustive context. Across three document-level MT benchmarks spanning six languages and diverse domains, TransGraph consistently surpasses strong baselines in translation quality and terminology consistency while incurring significantly lower token overhead. 

**Abstract (ZH)**: 基于话语引导的框架TransGraph在长文档翻译中通过结构化话语图Explicitly建模片段间关系并在相关图邻域上选择性条件各翻译段落，超越了强大的基线模型并在术语一致性方面表现出色，同时显著减少token开销。 

---
# LLMServingSim2.0: A Unified Simulator for Heterogeneous Hardware and Serving Techniques in LLM Infrastructure 

**Title (ZH)**: LLMServingSim2.0：面向LLM基础设施的异构硬件与服务技术统一仿真器 

**Authors**: Jaehong Cho, Hyunmin Choi, Jongse Park  

**Link**: [PDF](https://arxiv.org/pdf/2511.07229)  

**Abstract**: This paper introduces LLMServingSim2.0, a system simulator designed for exploring heterogeneous hardware in large-scale LLM serving systems. LLMServingSim2.0 addresses two key limitations of its predecessor: (1) integrating hardware models into system-level simulators is non-trivial due to the lack of a clear abstraction, and (2) existing simulators support only a narrow subset of serving techniques, leaving no infrastructure that captures the breadth of approaches in modern LLM serving. To overcome these issues, LLMServingSim2.0 adopts trace-driven performance modeling, accompanied by an operator-level latency profiler, enabling the integration of new accelerators with a single command. It further embeds up-to-date serving techniques while exposing flexible interfaces for request routing, cache management, and scheduling policies. In a TPU case study, our profiler requires 18.5x fewer LoC and outperforms the predecessor's hardware-simulator integration, demonstrating LLMServingSim2.0's low-effort hardware extensibility. Our experiments further show that LLMServingSim2.0 reproduces GPU-based LLM serving with 1.9% error, while maintaining practical simulation time, making it a comprehensive platform for both hardware developers and LLM service providers. 

**Abstract (ZH)**: LLMServingSim2.0：面向大规模LLM服务系统的异构硬件系统模拟器 

---
# NoteEx: Interactive Visual Context Manipulation for LLM-Assisted Exploratory Data Analysis in Computational Notebooks 

**Title (ZH)**: NoteEx: 交互式视觉上下文操控以实现基于LLM辅助的计算笔记本探索性数据分析 

**Authors**: Mohammad Hasan Payandeh, Lin-Ping Yuan, Jian Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.07223)  

**Abstract**: Computational notebooks have become popular for Exploratory Data Analysis (EDA), augmented by LLM-based code generation and result interpretation. Effective LLM assistance hinges on selecting informative context -- the minimal set of cells whose code, data, or outputs suffice to answer a prompt. As notebooks grow long and messy, users can lose track of the mental model of their analysis. They thus fail to curate appropriate contexts for LLM tasks, causing frustration and tedious prompt engineering. We conducted a formative study (n=6) that surfaced challenges in LLM context selection and mental model maintenance. Therefore, we introduce NoteEx, a JupyterLab extension that provides a semantic visualization of the EDA workflow, allowing analysts to externalize their mental model, specify analysis dependencies, and enable interactive selection of task-relevant contexts for LLMs. A user study (n=12) against a baseline shows that NoteEx improved mental model retention and context selection, leading to more accurate and relevant LLM responses. 

**Abstract (ZH)**: 计算笔记本已成为探索性数据分析（EDA）的热门工具，借助基于LLM的代码生成和结果解释得以增强。有效的LLM辅助取决于选择具有信息量的上下文——即足以回答提示的最小代码、数据或输出单元集合。随着笔记本内容的增长和混乱，用户可能会丢失其分析思维模型，从而无法为LLM任务挑选合适的上下文，导致挫败感和繁琐的提示工程。我们进行了一项形成性研究（n=6），揭示了LLM上下文选择和思维模型维护的挑战。因此，我们引入了NoteEx，这是一种JupyterLab扩展，提供EDA工作流程的语义可视化，允许分析师外部化其思维模型、指定分析依赖关系，并为LLM启用与任务相关上下文的交互选择。与基线的用户研究（n=12）表明，NoteEx提高了思维模型保留率和上下文选择，从而促进了更准确和相关性的LLM响应。 

---
# SMiLE: Provably Enforcing Global Relational Properties in Neural Networks 

**Title (ZH)**: SMiLE: 证明可确保神经网络中的全局关系性质 

**Authors**: Matteo Francobaldi, Michele Lombardi, Andrea Lodi  

**Link**: [PDF](https://arxiv.org/pdf/2511.07208)  

**Abstract**: Artificial Intelligence systems are increasingly deployed in settings where ensuring robustness, fairness, or domain-specific properties is essential for regulation compliance and alignment with human values. However, especially on Neural Networks, property enforcement is very challenging, and existing methods are limited to specific constraints or local properties (defined around datapoints), or fail to provide full guarantees. We tackle these limitations by extending SMiLE, a recently proposed enforcement framework for NNs, to support global relational properties (defined over the entire input space). The proposed approach scales well with model complexity, accommodates general properties and backbones, and provides full satisfaction guarantees. We evaluate SMiLE on monotonicity, global robustness, and individual fairness, on synthetic and real data, for regression and classification tasks. Our approach is competitive with property-specific baselines in terms of accuracy and runtime, and strictly superior in terms of generality and level of guarantees. Overall, our results emphasize the potential of the SMiLE framework as a platform for future research and applications. 

**Abstract (ZH)**: 人工智能系统在确保稳健性、公平性或领域特定属性方面被广泛部署，这对于合规性和与人类价值观一致至关重要。然而，特别是在神经网络中，属性约束实施非常具有挑战性，现有方法只能处理特定约束或局部属性（定义在数据点周围），或无法提供全面的保证。我们通过将SMiLE扩展为支持全局关系属性（定义在整个输入空间上）的框架来应对这些限制。所提出的方法能够良好地应对模型复杂性，兼容通用属性和基础结构，并提供全面满足的保证。我们在线性和分类任务中的合成数据和真实数据上评估了SMiLE在单调性、全局稳健性和个体公平性方面的性能。我们的方法在准确性和运行时间方面与属性特定的基本方法竞争，但在通用性和保证水平方面更优。总体而言，我们的结果强调了SMiLE框架作为未来研究和应用平台的潜力。 

---
# Twenty-Five Years of MIR Research: Achievements, Practices, Evaluations, and Future Challenges 

**Title (ZH)**: 二十五年来的音乐信息检索研究：成就、实践、评估及未来挑战 

**Authors**: Geoffroy Peeters, Zafar Rafii, Magdalena Fuentes, Zhiyao Duan, Emmanouil Benetos, Juhan Nam, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2511.07205)  

**Abstract**: In this paper, we trace the evolution of Music Information Retrieval (MIR) over the past 25 years. While MIR gathers all kinds of research related to music informatics, a large part of it focuses on signal processing techniques for music data, fostering a close relationship with the IEEE Audio and Acoustic Signal Processing Technical Commitee. In this paper, we reflect the main research achievements of MIR along the three EDICS related to music analysis, processing and generation. We then review a set of successful practices that fuel the rapid development of MIR research. One practice is the annual research benchmark, the Music Information Retrieval Evaluation eXchange, where participants compete on a set of research tasks. Another practice is the pursuit of reproducible and open research. The active engagement with industry research and products is another key factor for achieving large societal impacts and motivating younger generations of students to join the field. Last but not the least, the commitment to diversity, equity and inclusion ensures MIR to be a vibrant and open community where various ideas, methodologies, and career pathways collide. We finish by providing future challenges MIR will have to face. 

**Abstract (ZH)**: 过去25年音乐信息检索的发展演变：面向音乐分析、处理和生成的主要研究成果及其实践回顾 

---
# Resilient by Design - Active Inference for Distributed Continuum Intelligence 

**Title (ZH)**: 设计上的 resilient - 分布式连续智能的主动推断 

**Authors**: Praveen Kumar Donta, Alfreds Lapkovskis, Enzo Mingozzi, Schahram Dustdar  

**Link**: [PDF](https://arxiv.org/pdf/2511.07202)  

**Abstract**: Failures are the norm in highly complex and heterogeneous devices spanning the distributed computing continuum (DCC), from resource-constrained IoT and edge nodes to high-performance computing systems. Ensuring reliability and global consistency across these layers remains a major challenge, especially for AI-driven workloads requiring real-time, adaptive coordination. This paper introduces a Probabilistic Active Inference Resilience Agent (PAIR-Agent) to achieve resilience in DCC systems. PAIR-Agent performs three core operations: (i) constructing a causal fault graph from device logs, (ii) identifying faults while managing certainties and uncertainties using Markov blankets and the free-energy principle, and (iii) autonomously healing issues through active inference. Through continuous monitoring and adaptive reconfiguration, the agent maintains service continuity and stability under diverse failure conditions. Theoretical validations confirm the reliability and effectiveness of the proposed framework. 

**Abstract (ZH)**: 高复杂度异构设备沿分布式计算 continuum (DCC) 范围内的故障是常态，从资源受限的物联网和边缘节点到高性能计算系统。确保这些层面上的可靠性和全局一致性仍然是一个重大挑战，尤其是在需要实时、自适应协调的 AI 驱动工作负载方面。本文介绍了一种概率主动推断韧性代理（PAIR-Agent）以实现 DCC 系统的韧性。PAIR-Agent 执行三项核心操作：（i）从设备日志构建因果故障图，（ii）使用马尔可夫毯和最小自由能原则管理确定性和不确定性以识别故障，（iii）通过主动推断自主修复问题。通过持续监控和自适应重新配置，代理在多样化的故障条件下保持服务连续性和稳定性。理论验证确认了所提出框架的可靠性和有效性。 

---
# Federated Learning for Video Violence Detection: Complementary Roles of Lightweight CNNs and Vision-Language Models for Energy-Efficient Use 

**Title (ZH)**: 基于 federated learning 的视频暴力检测：轻量级CNN和视觉语言模型在能量高效利用中的互补作用 

**Authors**: Sébastien Thuau, Siba Haidar, Rachid Chelouah  

**Link**: [PDF](https://arxiv.org/pdf/2511.07171)  

**Abstract**: Deep learning-based video surveillance increasingly demands privacy-preserving architectures with low computational and environmental overhead. Federated learning preserves privacy but deploying large vision-language models (VLMs) introduces major energy and sustainability challenges. We compare three strategies for federated violence detection under realistic non-IID splits on the RWF-2000 and RLVS datasets: zero-shot inference with pretrained VLMs, LoRA-based fine-tuning of LLaVA-NeXT-Video-7B, and personalized federated learning of a 65.8M-parameter 3D CNN. All methods exceed 90% accuracy in binary violence detection. The 3D CNN achieves superior calibration (ROC AUC 92.59%) at roughly half the energy cost (240 Wh vs. 570 Wh) of federated LoRA, while VLMs provide richer multimodal reasoning. Hierarchical category grouping (based on semantic similarity and class exclusion) boosts VLM multiclass accuracy from 65.31% to 81% on the UCF-Crime dataset. To our knowledge, this is the first comparative simulation study of LoRA-tuned VLMs and personalized CNNs for federated violence detection, with explicit energy and CO2e quantification. Our results inform hybrid deployment strategies that default to efficient CNNs for routine inference and selectively engage VLMs for complex contextual reasoning. 

**Abstract (ZH)**: 基于深度学习的视频监控越来越需求低计算和环境负载的隐私保护架构。联邦学习可保护隐私，但部署大型视觉-语言模型（VLMs）引入了重大能源和可持续性挑战。我们在RWF-2000和RLVS数据集上比较了联邦暴力检测的三种策略：零样本推理与预训练VLMs、基于LoRA的LLaVA-NeXT-Video-7B微调以及65.8M参数的个性化3D CNN联邦学习。所有方法在二分类暴力检测中的准确率均超过90%。3D CNN在约一半的能耗（240 Wh vs. 570 Wh）下实现了更好的校准（ROC AUC 92.59%），而VLMs提供了更丰富的跨模态推理。基于语义相似性和类别排除的分层类别分组在UCF-Crime数据集上的VLM多分类准确率从65.31%提升到81%。据我们所知，这是首次对LoRA调优的VLMs和个性化CNNs进行的联邦暴力检测比较模拟研究，明确量化了能耗和CO2e排放。我们的结果提供了混合部署策略的指导，对于常规推理默认使用高效CNNs，并在需要复杂上下文推理时选择性地使用VLMs。 

---
# AdaRec: Adaptive Recommendation with LLMs via Narrative Profiling and Dual-Channel Reasoning 

**Title (ZH)**: AdaRec：通过叙事画像和双重通道推理的适应性推荐方法 

**Authors**: Meiyun Wang, Charin Polpanumas  

**Link**: [PDF](https://arxiv.org/pdf/2511.07166)  

**Abstract**: We propose AdaRec, a few-shot in-context learning framework that leverages large language models for an adaptive personalized recommendation. AdaRec introduces narrative profiling, transforming user-item interactions into natural language representations to enable unified task handling and enhance human readability. Centered on a bivariate reasoning paradigm, AdaRec employs a dual-channel architecture that integrates horizontal behavioral alignment, discovering peer-driven patterns, with vertical causal attribution, highlighting decisive factors behind user preferences. Unlike existing LLM-based approaches, AdaRec eliminates manual feature engineering through semantic representations and supports rapid cross-task adaptation with minimal supervision. Experiments on real ecommerce datasets demonstrate that AdaRec outperforms both machine learning models and LLM-based baselines by up to eight percent in few-shot settings. In zero-shot scenarios, it achieves up to a nineteen percent improvement over expert-crafted profiling, showing effectiveness for long-tail personalization with minimal interaction data. Furthermore, lightweight fine-tuning on synthetic data generated by AdaRec matches the performance of fully fine-tuned models, highlighting its efficiency and generalization across diverse tasks. 

**Abstract (ZH)**: AdaRec：一种利用大规模语言模型实现自适应个性化推荐的少样本在上下文中学习框架 

---
# Fuzzy Label: From Concept to Its Application in Label Learning 

**Title (ZH)**: 模糊标签：从概念到标签学习中的应用 

**Authors**: Chenxi Luoa, Zhuangzhuang Zhaoa, Zhaohong Denga, Te Zhangb  

**Link**: [PDF](https://arxiv.org/pdf/2511.07165)  

**Abstract**: Label learning is a fundamental task in machine learning that aims to construct intelligent models using labeled data, encompassing traditional single-label and multi-label classification models. Traditional methods typically rely on logical labels, such as binary indicators (e.g., "yes/no") that specify whether an instance belongs to a given category. However, in practical applications, label annotations often involve significant uncertainty due to factors such as data noise, inherent ambiguity in the observed entities, and the subjectivity of human annotators. Therefore, representing labels using simplistic binary logic can obscure valuable information and limit the expressiveness of label learning models. To overcome this limitation, this paper introduces the concept of fuzzy labels, grounded in fuzzy set theory, to better capture and represent label uncertainty. We further propose an efficient fuzzy labeling method that mines and generates fuzzy labels from the original data, thereby enriching the label space with more informative and nuanced representations. Based on this foundation, we present fuzzy-label-enhanced algorithms for both single-label and multi-label learning, using the classical K-Nearest Neighbors (KNN) and multi-label KNN algorithms as illustrative examples. Experimental results indicate that fuzzy labels can more effectively characterize the real-world labeling information and significantly enhance the performance of label learning models. 

**Abstract (ZH)**: 基于模糊标签的标签学习：从经典K-最近邻算法到多标签学习 

---
# Conditional Diffusion as Latent Constraints for Controllable Symbolic Music Generation 

**Title (ZH)**: 条件扩散作为潜在约束的可控符号音乐生成 

**Authors**: Matteo Pettenó, Alessandro Ilic Mezza, Alberto Bernardini  

**Link**: [PDF](https://arxiv.org/pdf/2511.07156)  

**Abstract**: Recent advances in latent diffusion models have demonstrated state-of-the-art performance in high-dimensional time-series data synthesis while providing flexible control through conditioning and guidance. However, existing methodologies primarily rely on musical context or natural language as the main modality of interacting with the generative process, which may not be ideal for expert users who seek precise fader-like control over specific musical attributes. In this work, we explore the application of denoising diffusion processes as plug-and-play latent constraints for unconditional symbolic music generation models. We focus on a framework that leverages a library of small conditional diffusion models operating as implicit probabilistic priors on the latents of a frozen unconditional backbone. While previous studies have explored domain-specific use cases, this work, to the best of our knowledge, is the first to demonstrate the versatility of such an approach across a diverse array of musical attributes, such as note density, pitch range, contour, and rhythm complexity. Our experiments show that diffusion-driven constraints outperform traditional attribute regularization and other latent constraints architectures, achieving significantly stronger correlations between target and generated attributes while maintaining high perceptual quality and diversity. 

**Abstract (ZH)**: 近期，潜在扩散模型在高维时间序列数据合成方面取得了卓越性能，同时通过条件控制和引导提供了灵活的控制能力。然而，现有方法主要依赖于音乐上下文或自然语言作为与生成过程交互的主要模态，这可能不适合寻求对特定音乐属性精细化控制的专家用户。在本工作中，我们探讨使用去噪扩散过程作为即插即用的潜在约束，应用于无条件符号音乐生成模型。我们关注一个框架，该框架利用一组小型条件扩散模型作为冻结无条件主干的潜在隐含概率先验。尽管先前的研究探索了特定领域的应用案例，据我们所知，本工作是首次在多种音乐属性（如音符密度、音高范围、轮廓和节奏复杂性）上展示此类方法的灵活性和通用性。实验结果表明，以扩散驱动的约束在目标属性和生成属性之间的相关性、感知质量和多样性方面优于传统的属性正则化和其他潜在约束架构。 

---
# LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging 

**Title (ZH)**: LoRA随行：实例级动态LoRA选择与合并 

**Authors**: Seungeon Lee, Soumi Das, Manish Gupta, Krishna P. Gummadi  

**Link**: [PDF](https://arxiv.org/pdf/2511.07129)  

**Abstract**: Low-Rank Adaptation (LoRA) has emerged as a parameter-efficient approach for fine-tuning large language this http URL, conventional LoRA adapters are typically trained for a single task, limiting their applicability in real-world settings where inputs may span diverse and unpredictable domains. At inference time, existing approaches combine multiple LoRAs for improving performance on diverse tasks, while usually requiring labeled data or additional task-specific training, which is expensive at scale. In this work, we introduce LoRA on the Go (LoGo), a training-free framework that dynamically selects and merges adapters at the instance level without any additional requirements. LoGo leverages signals extracted from a single forward pass through LoRA adapters, to identify the most relevant adapters and determine their contributions on-the-fly. Across 5 NLP benchmarks, 27 datasets, and 3 model families, LoGo outperforms training-based baselines on some tasks upto a margin of 3.6% while remaining competitive on other tasks and maintaining inference throughput, highlighting its effectiveness and practicality. 

**Abstract (ZH)**: LoRA on the Go：无需训练的适配器动态选择与融合框架 

---
# Think Consistently, Reason Efficiently: Energy-Based Calibration for Implicit Chain-of-Thought 

**Title (ZH)**: 一致性思考，高效推理：能量基校准for隐式链式思考 

**Authors**: Zhikang Chen, Sen Cui, Deheng Ye, Yu Zhang, Yatao Bian, Tingting Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07124)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong reasoning capabilities through \emph{Chain-of-Thought} (CoT) prompting, which enables step-by-step intermediate reasoning. However, explicit CoT methods rely on discrete token-level reasoning processes that are prone to error propagation and limited by vocabulary expressiveness, often resulting in rigid and inconsistent reasoning trajectories. Recent research has explored implicit or continuous reasoning in latent spaces, allowing models to perform internal reasoning before generating explicit output. Although such approaches alleviate some limitations of discrete CoT, they generally lack explicit mechanisms to enforce consistency among reasoning steps, leading to divergent reasoning paths and unstable outcomes. To address this issue, we propose EBM-CoT, an Energy-Based Chain-of-Thought Calibration framework that refines latent thought representations through an energy-based model (EBM). Our method dynamically adjusts latent reasoning trajectories toward lower-energy, high-consistency regions in the embedding space, improving both reasoning accuracy and consistency without modifying the base language model. Extensive experiments across mathematical, commonsense, and symbolic reasoning benchmarks demonstrate that the proposed framework significantly enhances the consistency and efficiency of multi-step reasoning in LLMs. 

**Abstract (ZH)**: 基于能量的链式思维校准框架（EBM-CoT）：通过能量模型优化隐层思维表示以增强大型语言模型的多步推理一致性与效率 

---
# On the Joint Minimization of Regularization Loss Functions in Deep Variational Bayesian Methods for Attribute-Controlled Symbolic Music Generation 

**Title (ZH)**: 基于深度变分贝叶斯方法的属性控制符号音乐生成中正则化损失函数的联合最小化 

**Authors**: Matteo Pettenó, Alessandro Ilic Mezza, Alberto Bernardini  

**Link**: [PDF](https://arxiv.org/pdf/2511.07118)  

**Abstract**: Explicit latent variable models provide a flexible yet powerful framework for data synthesis, enabling controlled manipulation of generative factors. With latent variables drawn from a tractable probability density function that can be further constrained, these models enable continuous and semantically rich exploration of the output space by navigating their latent spaces. Structured latent representations are typically obtained through the joint minimization of regularization loss functions. In variational information bottleneck models, reconstruction loss and Kullback-Leibler Divergence (KLD) are often linearly combined with an auxiliary Attribute-Regularization (AR) loss. However, balancing KLD and AR turns out to be a very delicate matter. When KLD dominates over AR, generative models tend to lack controllability; when AR dominates over KLD, the stochastic encoder is encouraged to violate the standard normal prior. We explore this trade-off in the context of symbolic music generation with explicit control over continuous musical attributes. We show that existing approaches struggle to jointly minimize both regularization objectives, whereas suitable attribute transformations can help achieve both controllability and regularization of the target latent dimensions. 

**Abstract (ZH)**: 显式潜在变量模型提供了一种灵活而强大的框架，用于数据合成，使生成因子的可控操作成为可能。通过从易于处理的概率密度函数中抽取潜在变量，并进一步对其进行约束，这些模型能够在潜在空间导航时对输出空间进行连续且语义丰富的探索。结构化的潜在表示通常通过最小化正则化损失函数的联合来获得。在变分信息瓶颈模型中，重构损失和Kullback-Leibler散度（KLD）通常与辅助属性正则化（AR）损失线性组合。然而，平衡KLD和AR证明是一件非常微妙的事情。当KLD主导AR时，生成模型往往会缺乏可控性；当AR主导KLD时，随机编码器会被激励违反标准正态先验。在这种背景下，我们探讨了符号音乐生成中的可控连续音乐属性。我们展示了现有方法难以同时最小化这两种正则化目标，而合适的属性转换可以帮助实现目标潜在维度的可控性和正则化。 

---
# More Agents Helps but Adversarial Robustness Gap Persists 

**Title (ZH)**: 更多代理有助于提高，但对抗鲁棒性差距依然存在 

**Authors**: Khashayar Alavi, Zhastay Yeltay, Lucie Flek, Akbar Karimi  

**Link**: [PDF](https://arxiv.org/pdf/2511.07112)  

**Abstract**: When LLM agents work together, they seem to be more powerful than a single LLM in mathematical question answering. However, are they also more robust to adversarial inputs? We investigate this question using adversarially perturbed math questions. These perturbations include punctuation noise with three intensities (10, 30, and 50 percent), plus real-world and human-like typos (WikiTypo, R2ATA). Using a unified sampling-and-voting framework (Agent Forest), we evaluate six open-source models (Qwen3-4B/14B, Llama3.1-8B, Mistral-7B, Gemma3-4B/12B) across four benchmarks (GSM8K, MATH, MMLU-Math, MultiArith), with various numbers of agents n from one to 25 (1, 2, 5, 10, 15, 20, 25). Our findings show that (1) Noise type matters: punctuation noise harm scales with its severity, and the human typos remain the dominant bottleneck, yielding the largest gaps to Clean accuracy and the highest ASR even with a large number of agents. And (2) Collaboration reliably improves accuracy as the number of agents, n, increases, with the largest gains from one to five agents and diminishing returns beyond 10 agents. However, the adversarial robustness gap persists regardless of the agent count. 

**Abstract (ZH)**: 当LLM代理协同工作时，它们在数学问题回答中似乎比单个LLM更强大。然而，它们也更 robust 吗？我们使用对抗性扰动数学问题来研究这一问题。这些扰动包括三种强度的标点符号噪音（10%、30% 和 50%）以及实际世界和类人类的拼写错误（WikiTypo、R2ATA）。我们使用统一的采样和投票框架（Agent Forest）评估了六种开源模型（Qwen3-4B/14B、Llama3.1-8B、Mistral-7B、Gemma3-4B/12B）在四个基准（GSM8K、MATH、MMLU-Math、MultiArith）上的表现，代理数量 n 从 1 增加到 25（1, 2, 5, 10, 15, 20, 25）。我们的发现表明：（1）噪音类型很重要：标点符号噪音的危害随其严重程度而增加，人类拼写错误仍然是占主导地位的瓶颈，即使代理数量很大，也导致最大的准确率差距和最高的误报率。（2）随着代理数量 n 增加，合作可以可靠地提高准确性，从一个到五个代理时收益最大，超过 10 代理时收益递减。然而，无论代理数量如何，对抗性鲁棒性差距仍然存在。 

---
# GEWDiff: Geometric Enhanced Wavelet-based Diffusion Model for Hyperspectral Image Super-resolution 

**Title (ZH)**: _GEWDiff: 基于几何增强小波扩散模型的高光谱图像超分辨率_ 

**Authors**: Sirui Wang, Jiang He, Natàlia Blasco Andreo, Xiao Xiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07103)  

**Abstract**: Improving the quality of hyperspectral images (HSIs), such as through super-resolution, is a crucial research area. However, generative modeling for HSIs presents several challenges. Due to their high spectral dimensionality, HSIs are too memory-intensive for direct input into conventional diffusion models. Furthermore, general generative models lack an understanding of the topological and geometric structures of ground objects in remote sensing imagery. In addition, most diffusion models optimize loss functions at the noise level, leading to a non-intuitive convergence behavior and suboptimal generation quality for complex data. To address these challenges, we propose a Geometric Enhanced Wavelet-based Diffusion Model (GEWDiff), a novel framework for reconstructing hyperspectral images at 4-times super-resolution. A wavelet-based encoder-decoder is introduced that efficiently compresses HSIs into a latent space while preserving spectral-spatial information. To avoid distortion during generation, we incorporate a geometry-enhanced diffusion process that preserves the geometric features. Furthermore, a multi-level loss function was designed to guide the diffusion process, promoting stable convergence and improved reconstruction fidelity. Our model demonstrated state-of-the-art results across multiple dimensions, including fidelity, spectral accuracy, visual realism, and clarity. 

**Abstract (ZH)**: 提高高光谱图像的质量（如通过超分辨率），是一项关键的研究领域。然而，高光谱图像的生成建模面临着诸多挑战。由于其高的光谱维度，高光谱图像对常规扩散模型来说内存消耗过大，无法直接输入。此外，通用生成模型缺乏对遥感图像中地面目标的拓扑和几何结构的理解。另外，大多数扩散模型在噪声级别上优化损失函数，导致直观性差的收敛行为和复杂数据生成质量欠佳。为解决这些挑战，我们提出了一种几何增强小波扩散模型（GEWDiff），这是一种用于在4倍超分辨率下重构高光谱图像的新型框架。引入了基于小波的编码解码器，能够高效地将高光谱图像压缩到潜空间中，同时保留光谱空间信息。为了在生成过程中避免失真，我们引入了几何增强的扩散过程，以保持几何特征。此外，设计了多级损失函数来引导扩散过程，促进稳定收敛和改进的重建保真度。我们的模型在多个维度上展示了最先进的结果，包括保真度、光谱准确性、视觉真实性和清晰度。 

---
# E2E-VGuard: Adversarial Prevention for Production LLM-based End-To-End Speech Synthesis 

**Title (ZH)**: E2E-VGuard：面向生产级LLM的端到端语音合成对抗防护 

**Authors**: Zhisheng Zhang, Derui Wang, Yifan Mi, Zhiyong Wu, Jie Gao, Yuxin Cao, Kai Ye, Minhui Xue, Jie Hao  

**Link**: [PDF](https://arxiv.org/pdf/2511.07099)  

**Abstract**: Recent advancements in speech synthesis technology have enriched our daily lives, with high-quality and human-like audio widely adopted across real-world applications. However, malicious exploitation like voice-cloning fraud poses severe security risks. Existing defense techniques struggle to address the production large language model (LLM)-based speech synthesis. While previous studies have considered the protection for fine-tuning synthesizers, they assume manually annotated transcripts. Given the labor intensity of manual annotation, end-to-end (E2E) systems leveraging automatic speech recognition (ASR) to generate transcripts are becoming increasingly prevalent, e.g., voice cloning via commercial APIs. Therefore, this E2E speech synthesis also requires new security mechanisms. To tackle these challenges, we propose E2E-VGuard, a proactive defense framework for two emerging threats: (1) production LLM-based speech synthesis, and (2) the novel attack arising from ASR-driven E2E scenarios. Specifically, we employ the encoder ensemble with a feature extractor to protect timbre, while ASR-targeted adversarial examples disrupt pronunciation. Moreover, we incorporate the psychoacoustic model to ensure perturbative imperceptibility. For a comprehensive evaluation, we test 16 open-source synthesizers and 3 commercial APIs across Chinese and English datasets, confirming E2E-VGuard's effectiveness in timbre and pronunciation protection. Real-world deployment validation is also conducted. Our code and demo page are available at this https URL. 

**Abstract (ZH)**: Recent advancements in speech synthesis technology have enriched our daily lives, with high-quality and human-like audio widely adopted across real-world applications.然而，恶意利用如声音克隆欺诈产生了严重的安全风险。现有的防御技术难以应对基于大规模语言模型（LLM）的语音合成生产。虽然已有研究考虑了对合成器微调的保护，但这些研究假设有手动标注的脚本。鉴于手动标注的劳动强度，利用自动语音识别（ASR）生成脚本的端到端（E2E）系统正变得越来越普遍，例如通过商业API进行的声音克隆。因此，这种E2E语音合成也需要新的安全机制。为应对这些挑战，我们提出了E2E-VGuard，这是一种针对两大新兴威胁的前瞻防御框架：（1）基于LLM的语音合成生产，（2）来自ASR驱动的E2E场景的新攻击。具体而言，我们采用编码器组合和特征提取器来保护音色，而针对ASR的对抗样本则破坏发音。此外，我们还结合了听觉心理模型以确保扰动的不可感知性。为了进行全面评估，我们在中文和英文数据集上测试了16个开源合成器和3个商业API，证实了E2E-VGuard在音色和发音保护方面的有效性。我们还在现实世界的部署中进行了验证。我们的代码和演示页面可在此处访问。 

---
# Sample-efficient quantum error mitigation via classical learning surrogates 

**Title (ZH)**: 基于经典学习代理的高效样本量子错误校正 

**Authors**: Wei-You Liao, Ge Yan, Yujin Song, Tian-Ci Tian, Wei-Ming Zhu, De-Tao Jiang, Yuxuan Du, He-Liang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07092)  

**Abstract**: The pursuit of practical quantum utility on near-term quantum processors is critically challenged by their inherent noise. Quantum error mitigation (QEM) techniques are leading solutions to improve computation fidelity with relatively low qubit-overhead, while full-scale quantum error correction remains a distant goal. However, QEM techniques incur substantial measurement overheads, especially when applied to families of quantum circuits parameterized by classical inputs. Focusing on zero-noise extrapolation (ZNE), a widely adopted QEM technique, here we devise the surrogate-enabled ZNE (S-ZNE), which leverages classical learning surrogates to perform ZNE entirely on the classical side. Unlike conventional ZNE, whose measurement cost scales linearly with the number of circuits, S-ZNE requires only constant measurement overhead for an entire family of quantum circuits, offering superior scalability. Theoretical analysis indicates that S-ZNE achieves accuracy comparable to conventional ZNE in many practical scenarios, and numerical experiments on up to 100-qubit ground-state energy and quantum metrology tasks confirm its effectiveness. Our approach provides a template that can be effectively extended to other quantum error mitigation protocols, opening a promising path toward scalable error mitigation. 

**Abstract (ZH)**: 近mighty量子处理器中追求实用的量子效用受到其固有噪声的严重挑战。量子错误缓解（QEM）技术是提高计算保真度的有效解决方案，同时仍需较低的量子比特开销，而全面的量子错误校正仍然是一个遥远的目标。然而，QEM技术在应用于以经典输入参数化的量子电路族时会带来显著的测量开销。本文专注于广泛应用的零噪声外推（ZNE）技术，提出了一种基于代理的ZNE（S-ZNE），利用经典学习代理在经典侧完全执行ZNE。与传统的ZNE相比，S-ZNE仅需要恒定的测量开销即可处理整个量子电路族，从而实现更好的可扩展性。理论分析表明，在很多实际场景中，S-ZNE能达到与传统ZNE相当的精度，并且在多达100量子比特的基态能量和量子计量任务上的数值实验也验证了其有效性。该方法为其他量子错误缓解协议的有效扩展提供了一个模板，开启了一条通往可扩展错误缓解的有希望的道路。 

---
# How Bias Binds: Measuring Hidden Associations for Bias Control in Text-to-Image Compositions 

**Title (ZH)**: 偏差如何关联：测量隐藏关联以控制文本到图像合成中的偏差 

**Authors**: Jeng-Lin Li, Ming-Ching Chang, Wei-Chao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.07091)  

**Abstract**: Text-to-image generative models often exhibit bias related to sensitive attributes. However, current research tends to focus narrowly on single-object prompts with limited contextual diversity. In reality, each object or attribute within a prompt can contribute to bias. For example, the prompt "an assistant wearing a pink hat" may reflect female-inclined biases associated with a pink hat. The neglected joint effects of the semantic binding in the prompts cause significant failures in current debiasing approaches. This work initiates a preliminary investigation on how bias manifests under semantic binding, where contextual associations between objects and attributes influence generative outcomes. We demonstrate that the underlying bias distribution can be amplified based on these associations. Therefore, we introduce a bias adherence score that quantifies how specific object-attribute bindings activate bias. To delve deeper, we develop a training-free context-bias control framework to explore how token decoupling can facilitate the debiasing of semantic bindings. This framework achieves over 10% debiasing improvement in compositional generation tasks. Our analysis of bias scores across various attribute-object bindings and token decorrelation highlights a fundamental challenge: reducing bias without disrupting essential semantic relationships. These findings expose critical limitations in current debiasing approaches when applied to semantically bound contexts, underscoring the need to reassess prevailing bias mitigation strategies. 

**Abstract (ZH)**: 基于文本生成图像的模型常常表现出与敏感属性相关的偏差。然而，当前研究倾向于狭隘地关注单一对象提示，缺乏上下文多样性。实际上，提示中的每个对象或属性都可以导致偏差。例如，提示“戴着粉色帽子的助手”可能反映了与粉色帽子相关的女性倾向偏差。被忽视的提示中语义绑定的联合效应导致当前去偏方法的显著失败。本文初步探讨了语义绑定下偏差的表现，其中对象和属性之间的上下文关联影响生成结果。我们证明了基于这些关联，潜在的偏差分布可能会放大。因此，我们提出了一个偏差遵从度评分，以量化特定对象-属性绑定激活偏差的程度。为进一步探索，我们开发了一种无需训练的上下文偏差控制框架，以探讨词元解耦如何促进语义绑定的去偏性。该框架在组合生成任务中的去偏性改进超过10%。我们对不同属性-对象绑定和词元解耦的偏评分分析揭示了一个基本的挑战：在不破坏关键语义关系的情况下减少偏差。这些发现表明，当应用于语义绑定上下文时，当前的去偏方法存在重要的局限性，强调了重新评估现有偏差缓解策略的必要性。 

---
# Achieving Effective Virtual Reality Interactions via Acoustic Gesture Recognition based on Large Language Models 

**Title (ZH)**: 基于大型语言模型的声学手势识别实现有效虚拟现实交互 

**Authors**: Xijie Zhang, Fengliang He, Hong-Ning Dai  

**Link**: [PDF](https://arxiv.org/pdf/2511.07085)  

**Abstract**: Natural and efficient interaction remains a critical challenge for virtual reality and augmented reality (VR/AR) systems. Vision-based gesture recognition suffers from high computational cost, sensitivity to lighting conditions, and privacy leakage concerns. Acoustic sensing provides an attractive alternative: by emitting inaudible high-frequency signals and capturing their reflections, channel impulse response (CIR) encodes how gestures perturb the acoustic field in a low-cost and user-transparent manner. However, existing CIR-based gesture recognition methods often rely on extensive training of models on large labeled datasets, making them unsuitable for few-shot VR scenarios. In this work, we propose the first framework that leverages large language models (LLMs) for CIR-based gesture recognition in VR/AR systems. Despite LLMs' strengths, it is non-trivial to achieve few-shot and zero-shot learning of CIR gestures due to their inconspicuous features. To tackle this challenge, we collect differential CIR rather than original CIR data. Moreover, we construct a real-world dataset collected from 10 participants performing 15 gestures across three categories (digits, letters, and shapes), with 10 repetitions each. We then conduct extensive experiments on this dataset using an LLM-adopted classifier. Results show that our LLM-based framework achieves accuracy comparable to classical machine learning baselines, while requiring no domain-specific retraining. 

**Abstract (ZH)**: 自然且高效的交互仍然是虚拟现实和增强现实（VR/AR）系统的关键挑战。基于视觉的手势识别面临着高计算成本、对光照条件敏感以及隐私泄露担忧的问题。声学传感提供了一种有吸引力的替代方案：通过发射不可闻的高频信号并捕获其反射，信道冲激响应（CIR）以低成本和用户透明的方式编码手势对声场的扰动。然而，现有的基于CIR的手势识别方法通常依赖于在大型标注数据集上对模型进行广泛的训练，这使它们无法适用于少量样本的VR场景。在本文中，我们提出了首个利用大规模语言模型（LLMs）进行基于CIR的手势识别的框架。尽管大规模语言模型具有优势，但由于其特征不显着，实现针对CIR手势的少量样本和零样本学习仍具有挑战性。为应对这一挑战，我们收集了差异CIR数据而非原始CIR数据。此外，我们构建了一个由10名参与者完成的涵盖三个类别（数字、字母和形状）15种手势的现实世界数据集，每种手势重复10次。然后，我们使用采用大规模语言模型的分类器在该数据集上进行了大量实验。结果表明，我们的基于大规模语言模型的方法在准确率方面与经典机器学习基线相当，同时不需要特定领域的重新训练。 

---
# Pandar128 dataset for lane line detection 

**Title (ZH)**: Pandar128数据集用于车道线检测 

**Authors**: Filip Beránek, Václav Diviš, Ivan Gruber  

**Link**: [PDF](https://arxiv.org/pdf/2511.07084)  

**Abstract**: We present Pandar128, the largest public dataset for lane line detection using a 128-beam LiDAR. It contains over 52,000 camera frames and 34,000 LiDAR scans, captured in diverse real-world conditions in Germany. The dataset includes full sensor calibration (intrinsics, extrinsics) and synchronized odometry, supporting tasks such as projection, fusion, and temporal modeling.
To complement the dataset, we also introduce SimpleLidarLane, a light-weight baseline method for lane line reconstruction that combines BEV segmentation, clustering, and polyline fitting. Despite its simplicity, our method achieves strong performance under challenging various conditions (e.g., rain, sparse returns), showing that modular pipelines paired with high-quality data and principled evaluation can compete with more complex approaches.
Furthermore, to address the lack of standardized evaluation, we propose a novel polyline-based metric - Interpolation-Aware Matching F1 (IAM-F1) - that employs interpolation-aware lateral matching in BEV space.
All data and code are publicly released to support reproducibility in LiDAR-based lane detection. 

**Abstract (ZH)**: 我们介绍了Pandar128，这是用于车道线检测的最大公共数据集，基于一个具有128束激光的LiDAR。该数据集包含超过52,000帧相机图像和34,000个LiDAR扫描数据，这些数据在德国多种真实-world条件下采集。数据集包括完整的传感器校准（内参、外参）和同步的里程计信息，支持投影、融合和时间建模等任务。

为补充该数据集，我们还引入了SimpleLidarLane，这是一种轻量级的车道线重建基线方法，结合了BEV分割、聚类和多段线拟合。尽管方法简单，但在多种挑战性条件下（如雨天、稀疏返回点）仍能表现出色，显示出模块化流程与高质量数据和原则性评估相结合可以与更复杂的方法竞争。

此外，为解决标准评估缺乏的问题，我们提出了一种新的基于多段线的评估指标——插值感知匹配F1（IAM-F1），该指标在BEV空间中采用插值感知横向匹配。

所有数据和代码均已公开发布，以支持基于LiDAR的车道检测的可重复性。 

---
# Wasm: A Pipeline for Constructing Structured Arabic Interleaved Multimodal Corpora 

**Title (ZH)**: Wasm：构建结构化阿拉伯混合模态语料库的管道 

**Authors**: Khalil Hennara, Ahmad Bastati, Muhammad Hreden, Mohamed Motasim Hamed, Zeina Aldallal, Sara Chrouf, Safwan AlModhayan  

**Link**: [PDF](https://arxiv.org/pdf/2511.07080)  

**Abstract**: The performance of large language models (LLMs) and large multimodal models (LMMs) depends heavily on the quality and scale of their pre-training datasets. Recent research shows that large multimodal models trained on natural documents where images and text are interleaved outperform those trained only on image-text pairs across a wide range of benchmarks, leveraging advanced pre- trained models to enforce semantic alignment, image-sequence consistency, and textual coherence. For Arabic, however, the lack of high-quality multimodal datasets that preserve document structure has limited progress. In this paper, we present our pipeline Wasm for processing the Common Crawl dataset to create a new Arabic multimodal dataset that uniquely provides markdown output. Unlike existing Arabic corpora that focus solely on text extraction, our approach preserves the structural integrity of web content while maintaining flexibility for both text-only and multimodal pre-training scenarios. We provide a comprehensive comparative analysis of our data processing pipeline against those used for major existing datasets, highlighting the convergences in filtering strategies and justifying our specific design choices. To support future research, we publicly release a representative dataset dump along with the multimodal processing pipeline for Arabic. 

**Abstract (ZH)**: 大型语言模型（LLMs）和大型多模态模型（LMMs）的性能高度依赖于其预训练数据集的质量和规模。近期研究显示，多模态模型在自然文档上进行预训练，其中图像和文本交织，相较于仅在图像-文本配对上进行预训练，在多种基准测试中表现更优，利用先进的预训练模型来加强语义对齐、图像序列一致性以及文本连贯性。然而，对于阿拉伯语而言，缺乏能够保留文档结构的高质量多模态数据集限制了研究进展。在本文中，我们介绍了我们的数据处理管道Wasm，用于处理Common Crawl数据集，以创建一个新的阿拉伯语多模态数据集，该数据集独特地提供了markdown输出。与仅专注于文本提取的现有阿拉伯语语料库不同，我们的方法保留了网络内容的结构完整性，同时为仅文本和多模态预训练场景提供了灵活性。我们对数据处理管道进行了全面的比较分析，与主要现有数据集使用的管道进行了对比，突出了过滤策略的交汇之处，并解释了我们的特定设计选择。为了支持未来的研究，我们公开发布了代表性数据集的快照以及阿拉伯语的多模态处理管道。 

---
# TauFlow: Dynamic Causal Constraint for Complexity-Adaptive Lightweight Segmentation 

**Title (ZH)**: TauFlow：动态因果约束下的复杂性自适应轻量级分割 

**Authors**: Zidong Chen, Fadratul Hafinaz Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2511.07057)  

**Abstract**: Deploying lightweight medical image segmentation models on edge devices presents two major challenges: 1) efficiently handling the stark contrast between lesion boundaries and background regions, and 2) the sharp drop in accuracy that occurs when pursuing extremely lightweight designs (e.g., <0.5M parameters). To address these problems, this paper proposes TauFlow, a novel lightweight segmentation model. The core of TauFlow is a dynamic feature response strategy inspired by brain-like mechanisms. This is achieved through two key innovations: the Convolutional Long-Time Constant Cell (ConvLTC), which dynamically regulates the feature update rate to "slowly" process low-frequency backgrounds and "quickly" respond to high-frequency boundaries; and the STDP Self-Organizing Module, which significantly mitigates feature conflicts between the encoder and decoder, reducing the conflict rate from approximately 35%-40% to 8%-10%. 

**Abstract (ZH)**: 在边缘设备上部署轻量级医疗图像分割模型面临两大挑战：1) 有效处理病灶边界与背景区域之间的强烈对比；2) 追求极为轻量级设计（例如，参数少于0.5M）时准确率急剧下降。为解决这些问题，本文提出TauFlow，一种新颖的轻量级分割模型。TauFlow的核心是一种受脑类机制启发的动态特征响应策略，通过两个关键创新实现：卷积长时常数细胞（ConvLTC），动态调节特征更新速率以“缓慢”处理低频背景并“快速”响应高频边界；以及STDP自我组织模块，显著缓解编码器与解码器之间的特征冲突，将冲突率从约35%-40%降低至8%-10%。 

---
# Learning Quantized Continuous Controllers for Integer Hardware 

**Title (ZH)**: 学习针对整数硬件的量化连续控制器 

**Authors**: Fabian Kresse, Christoph H. Lampert  

**Link**: [PDF](https://arxiv.org/pdf/2511.07046)  

**Abstract**: Deploying continuous-control reinforcement learning policies on embedded hardware requires meeting tight latency and power budgets. Small FPGAs can deliver these, but only if costly floating point pipelines are avoided. We study quantization-aware training (QAT) of policies for integer inference and we present a learning-to-hardware pipeline that automatically selects low-bit policies and synthesizes them to an Artix-7 FPGA. Across five MuJoCo tasks, we obtain policy networks that are competitive with full precision (FP32) policies but require as few as 3 or even only 2 bits per weight, and per internal activation value, as long as input precision is chosen carefully. On the target hardware, the selected policies achieve inference latencies on the order of microseconds and consume microjoules per action, favorably comparing to a quantized reference. Last, we observe that the quantized policies exhibit increased input noise robustness compared to the floating-point baseline. 

**Abstract (ZH)**: 在嵌入式硬件上部署连续控制强化学习策略需要满足严格的延迟和功耗预算。小规模FPGA可以实现这一点，但需要避免使用昂贵的浮点流水线。我们研究了针对整数推理的量化感知训练（QAT），并提出了一种从学习到硬件的流水线，该流水线可以自动选择低比特宽的策略并综合到Artix-7 FPGA上。在五个MuJoCo任务中，我们得到了与全精度（FP32）策略具有竞争力的策略网络，但只需要每个权重和每个内部激活值分别使用3比特甚至2比特，前提是对输入精度进行适当选择。在目标硬件上，所选策略的推理延迟在微秒量级，每动作消耗微焦耳级能量，并且与量化参考策略相比表现出优势。最后，我们观察到量化策略在输入噪声鲁棒性方面优于浮点基线。 

---
# Benchmarking LLMs for Fine-Grained Code Review with Enriched Context in Practice 

**Title (ZH)**: 基于丰富上下文的细粒度代码审查中大规模语言模型的基准测试 

**Authors**: Ruida Hu, Xinchen Wang, Xin-Cheng Wen, Zhao Zhang, Bo Jiang, Pengfei Gao, Chao Peng, Cuiyun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2511.07017)  

**Abstract**: Code review is a cornerstone of software quality assurance, and recent advances in Large Language Models (LLMs) have shown promise in automating this process. However, existing benchmarks for LLM-based code review face three major limitations. (1) Lack of semantic context: most benchmarks provide only code diffs without textual information such as issue descriptions, which are crucial for understanding developer intent. (2) Data quality issues: without rigorous validation, many samples are noisy-e.g., reviews on outdated or irrelevant code-reducing evaluation reliability. (3) Coarse granularity: most benchmarks operate at the file or commit level, overlooking the fine-grained, line-level reasoning essential for precise review.
We introduce ContextCRBench, a high-quality, context-rich benchmark for fine-grained LLM evaluation in code review. Our construction pipeline comprises: (1) Raw Data Crawling, collecting 153.7K issues and pull requests from top-tier repositories; (2) Comprehensive Context Extraction, linking issue-PR pairs for textual context and extracting the full surrounding function or class for code context; and (3) Multi-stage Data Filtering, combining rule-based and LLM-based validation to remove outdated, malformed, or low-value samples, resulting in 67,910 context-enriched entries.
ContextCRBench supports three evaluation scenarios aligned with the review workflow: (1) hunk-level quality assessment, (2) line-level defect localization, and (3) line-level comment generation. Evaluating eight leading LLMs (four closed-source and four open-source) reveals that textual context yields greater performance gains than code context alone, while current LLMs remain far from human-level review ability. Deployed at ByteDance, ContextCRBench drives a self-evolving code review system, improving performance by 61.98% and demonstrating its robustness and industrial utility. 

**Abstract (ZH)**: 基于上下文的细粒度代码审查基准ContextCRBench 

---
# Diffolio: A Diffusion Model for Multivariate Probabilistic Financial Time-Series Forecasting and Portfolio Construction 

**Title (ZH)**: Diffolio：一种用于多变量概率金融时间序列预测和投资组合构建的扩散模型 

**Authors**: So-Yoon Cho, Jin-Young Kim, Kayoung Ban, Hyeng Keun Koo, Hyun-Gyoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.07014)  

**Abstract**: Probabilistic forecasting is crucial in multivariate financial time-series for constructing efficient portfolios that account for complex cross-sectional dependencies. In this paper, we propose Diffolio, a diffusion model designed for multivariate financial time-series forecasting and portfolio construction. Diffolio employs a denoising network with a hierarchical attention architecture, comprising both asset-level and market-level layers. Furthermore, to better reflect cross-sectional correlations, we introduce a correlation-guided regularizer informed by a stable estimate of the target correlation matrix. This structure effectively extracts salient features not only from historical returns but also from asset-specific and systematic covariates, significantly enhancing the performance of forecasts and portfolios. Experimental results on the daily excess returns of 12 industry portfolios show that Diffolio outperforms various probabilistic forecasting baselines in multivariate forecasting accuracy and portfolio performance. Moreover, in portfolio experiments, portfolios constructed from Diffolio's forecasts show consistently robust performance, thereby outperforming those from benchmarks by achieving higher Sharpe ratios for the mean-variance tangency portfolio and higher certainty equivalents for the growth-optimal portfolio. These results demonstrate the superiority of our proposed Diffolio in terms of not only statistical accuracy but also economic significance. 

**Abstract (ZH)**: Diffolio：一种用于多变量金融时间序列预测和组合构建的扩散模型 

---
# TrueCity: Real and Simulated Urban Data for Cross-Domain 3D Scene Understanding 

**Title (ZH)**: TrueCity: 真实与模拟城市数据在跨域3D场景理解中的应用 

**Authors**: Duc Nguyen, Yan-Ling Lai, Qilin Zhang, Prabin Gyawali, Benedikt Schwab, Olaf Wysocki, Thomas H. Kolbe  

**Link**: [PDF](https://arxiv.org/pdf/2511.07007)  

**Abstract**: 3D semantic scene understanding remains a long-standing challenge in the 3D computer vision community. One of the key issues pertains to limited real-world annotated data to facilitate generalizable models. The common practice to tackle this issue is to simulate new data. Although synthetic datasets offer scalability and perfect labels, their designer-crafted scenes fail to capture real-world complexity and sensor noise, resulting in a synthetic-to-real domain gap. Moreover, no benchmark provides synchronized real and simulated point clouds for segmentation-oriented domain shift analysis. We introduce TrueCity, the first urban semantic segmentation benchmark with cm-accurate annotated real-world point clouds, semantic 3D city models, and annotated simulated point clouds representing the same city. TrueCity proposes segmentation classes aligned with international 3D city modeling standards, enabling consistent evaluation of synthetic-to-real gap. Our extensive experiments on common baselines quantify domain shift and highlight strategies for exploiting synthetic data to enhance real-world 3D scene understanding. We are convinced that the TrueCity dataset will foster further development of sim-to-real gap quantification and enable generalizable data-driven models. The data, code, and 3D models are available online: this https URL 

**Abstract (ZH)**: 三维语义场景理解仍然是3D计算机视觉领域长期存在的挑战。其中一个关键问题是如何获取足够的标注数据以促进泛化模型的建立。常见的解决方法是模拟新的数据。尽管合成数据集具有可扩展性和完美的标签，但其设计者构思的场景无法捕捉真实世界场景的复杂性和传感器噪声，导致合成到真实场景的领域差异。此外，没有基准提供同步的真实和模拟点云数据，用于分割导向的领域转移分析。我们引入了TrueCity，这是首个具有厘米级精确标注的真实世界点云的城市语义分割基准，包含语义3D城市模型和代表同一城市的标注模拟点云。TrueCity提出了与国际3D城市建模标准对齐的分割类别，使得合成到真实场景差异的评估更加一致。我们在常见基准上的大量实验量化了领域转移，并突显了利用合成数据以增强真实世界3D场景理解的策略。我们坚信TrueCity数据集将促进合成到现实差距量化的发展，并促进泛化数据驱动模型的进一步发展。数据、代码和3D模型已在线发布：this https URL。 

---
# S$^2$Drug: Bridging Protein Sequence and 3D Structure in Contrastive Representation Learning for Virtual Screening 

**Title (ZH)**: S$^2$Drug：在对比表示学习中连接蛋白质序列与三维结构的虚拟筛选方法 

**Authors**: Bowei He, Bowen Gao, Yankai Chen, Yanyan Lan, Chen Ma, Philip S. Yu, Ya-Qin Zhang, Wei-Ying Ma  

**Link**: [PDF](https://arxiv.org/pdf/2511.07006)  

**Abstract**: Virtual screening (VS) is an essential task in drug discovery, focusing on the identification of small-molecule ligands that bind to specific protein pockets. Existing deep learning methods, from early regression models to recent contrastive learning approaches, primarily rely on structural data while overlooking protein sequences, which are more accessible and can enhance generalizability. However, directly integrating protein sequences poses challenges due to the redundancy and noise in large-scale protein-ligand datasets. To address these limitations, we propose \textbf{S$^2$Drug}, a two-stage framework that explicitly incorporates protein \textbf{S}equence information and 3D \textbf{S}tructure context in protein-ligand contrastive representation learning. In the first stage, we perform protein sequence pretraining on ChemBL using an ESM2-based backbone, combined with a tailored data sampling strategy to reduce redundancy and noise on both protein and ligand sides. In the second stage, we fine-tune on PDBBind by fusing sequence and structure information through a residue-level gating module, while introducing an auxiliary binding site prediction task. This auxiliary task guides the model to accurately localize binding residues within the protein sequence and capture their 3D spatial arrangement, thereby refining protein-ligand matching. Across multiple benchmarks, S$^2$Drug consistently improves virtual screening performance and achieves strong results on binding site prediction, demonstrating the value of bridging sequence and structure in contrastive learning. 

**Abstract (ZH)**: S$^2$Drug：一种结合蛋白质序列和结构对比表示学习的两阶段框架 

---
# Hybrid Autoencoders for Tabular Data: Leveraging Model-Based Augmentation in Low-Label Settings 

**Title (ZH)**: 基于模型的增强在少量标签设置中融合自动编码器用于表格数据 

**Authors**: Erel Naor, Ofir Lindenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2511.06961)  

**Abstract**: Deep neural networks often under-perform on tabular data due to their sensitivity to irrelevant features and a spectral bias toward smooth, low-frequency functions. These limitations hinder their ability to capture the sharp, high-frequency signals that often define tabular structure, especially under limited labeled samples. While self-supervised learning (SSL) offers promise in such settings, it remains challenging in tabular domains due to the lack of effective data augmentations. We propose a hybrid autoencoder that combines a neural encoder with an oblivious soft decision tree (OSDT) encoder, each guided by its own stochastic gating network that performs sample-specific feature selection. Together, these structurally different encoders and model-specific gating networks implement model-based augmentation, producing complementary input views tailored to each architecture. The two encoders, trained with a shared decoder and cross-reconstruction loss, learn distinct yet aligned representations that reflect their respective inductive biases. During training, the OSDT encoder (robust to noise and effective at modeling localized, high-frequency structure) guides the neural encoder toward representations more aligned with tabular data. At inference, only the neural encoder is used, preserving flexibility and SSL compatibility. Spectral analysis highlights the distinct inductive biases of each encoder. Our method achieves consistent gains in low-label classification and regression across diverse tabular datasets, outperforming deep and tree-based supervised baselines. 

**Abstract (ZH)**: 深度神经网络在处理表格数据时常常表现不佳，这是因为它们对无关特征敏感，并倾向于学习平滑的低频函数。这些限制阻碍了它们捕捉表格结构中常见的高频信号的能力，尤其是在标注样本有限的情况下。虽然自监督学习（SSL）在这种情况下显示出前景，但在表格领域中仍然具有挑战性，主要是因为缺乏有效数据增强方法。我们提出了一种混合自编码器，该自编码器结合了神经编码器和不知情软决策树（OSDT）编码器，并由各自的随机门控网络指导，该门控网络执行样本特定的特征选择。这些结构不同的编码器和模型特定的门控网络实现基于模型的数据增强，生成适应各自架构的互补输入视图。这两个编码器通过共享解码器和交叉重建损失进行训练，学习各自独特但对齐的表示，反映了它们各自的归纳偏置。在训练过程中，OSDT编码器（对噪声具有鲁棒性且能够有效地建模局部高频结构）引导神经编码器向更符合表格数据的表示学习。在推理时，仅使用神经编码器，保持了灵活性和SSL兼容性。频谱分析突显了每个编码器的独特归纳偏置。我们的方法在多种表格数据集上的低标签分类和回归任务中实现了持续的性能提升，超越了深层和基于树的监督基准。 

---
# FoCLIP: A Feature-Space Misalignment Framework for CLIP-Based Image Manipulation and Detection 

**Title (ZH)**: FoCLIP：一种基于特征空间错配的CLIP驱动图像处理与检测框架 

**Authors**: Yulin Chen, Zeyuan Wang, Tianyuan Yu, Yingmei Wei, Liang Bai  

**Link**: [PDF](https://arxiv.org/pdf/2511.06947)  

**Abstract**: The well-aligned attribute of CLIP-based models enables its effective application like CLIPscore as a widely adopted image quality assessment metric. However, such a CLIP-based metric is vulnerable for its delicate multimodal alignment. In this work, we propose \textbf{FoCLIP}, a feature-space misalignment framework for fooling CLIP-based image quality metric. Based on the stochastic gradient descent technique, FoCLIP integrates three key components to construct fooling examples: feature alignment as the core module to reduce image-text modality gaps, the score distribution balance module and pixel-guard regularization, which collectively optimize multimodal output equilibrium between CLIPscore performance and image quality. Such a design can be engineered to maximize the CLIPscore predictions across diverse input prompts, despite exhibiting either visual unrecognizability or semantic incongruence with the corresponding adversarial prompts from human perceptual perspectives. Experiments on ten artistic masterpiece prompts and ImageNet subsets demonstrate that optimized images can achieve significant improvement in CLIPscore while preserving high visual fidelity. In addition, we found that grayscale conversion induces significant feature degradation in fooling images, exhibiting noticeable CLIPscore reduction while preserving statistical consistency with original images. Inspired by this phenomenon, we propose a color channel sensitivity-driven tampering detection mechanism that achieves 91% accuracy on standard benchmarks. In conclusion, this work establishes a practical pathway for feature misalignment in CLIP-based multimodal systems and the corresponding defense method. 

**Abstract (ZH)**: 基于CLIP的特征空间错配框架FoCLIP：欺骗CLIP图像质量评估指标 

---
# Learning to Focus: Prioritizing Informative Histories with Structured Attention Mechanisms in Partially Observable Reinforcement Learning 

**Title (ZH)**: 学习聚焦：在部分可观测强化学习中使用结构化注意力机制优先处理信息性的历史记录 

**Authors**: Daniel De Dios Allegue, Jinke He, Frans A. Oliehoek  

**Link**: [PDF](https://arxiv.org/pdf/2511.06946)  

**Abstract**: Transformers have shown strong ability to model long-term dependencies and are increasingly adopted as world models in model-based reinforcement learning (RL) under partial observability. However, unlike natural language corpora, RL trajectories are sparse and reward-driven, making standard self-attention inefficient because it distributes weight uniformly across all past tokens rather than emphasizing the few transitions critical for control. To address this, we introduce structured inductive priors into the self-attention mechanism of the dynamics head: (i) per-head memory-length priors that constrain attention to task-specific windows, and (ii) distributional priors that learn smooth Gaussian weightings over past state-action pairs. We integrate these mechanisms into UniZero, a model-based RL agent with a Transformer-based world model that supports planning under partial observability. Experiments on the Atari 100k benchmark show that most efficiency gains arise from the Gaussian prior, which smoothly allocates attention to informative transitions, while memory-length priors often truncate useful signals with overly restrictive cut-offs. In particular, Gaussian Attention achieves a 77% relative improvement in mean human-normalized scores over UniZero. These findings suggest that in partially observable RL domains with non-stationary temporal dependencies, discrete memory windows are difficult to learn reliably, whereas smooth distributional priors flexibly adapt across horizons and yield more robust data efficiency. Overall, our results demonstrate that encoding structured temporal priors directly into self-attention improves the prioritization of informative histories for dynamics modeling under partial observability. 

**Abstract (ZH)**: 具有结构诱导先验的Transformers在部分可观测模型基于强化学习中的自注意力机制改进 

---
# From Attribution to Action: Jointly ALIGNing Predictions and Explanations 

**Title (ZH)**: 从归因到行动：联合优化预测与解释 

**Authors**: Dongsheng Hong, Chao Chen, Yanhui Chen, Shanshan Lin, Zhihao Chen, Xiangwen Liao  

**Link**: [PDF](https://arxiv.org/pdf/2511.06944)  

**Abstract**: Explanation-guided learning (EGL) has shown promise in aligning model predictions with interpretable reasoning, particularly in computer vision tasks. However, most approaches rely on external annotations or heuristic-based segmentation to supervise model explanations, which can be noisy, imprecise and difficult to scale. In this work, we provide both empirical and theoretical evidence that low-quality supervision signals can degrade model performance rather than improve it. In response, we propose ALIGN, a novel framework that jointly trains a classifier and a masker in an iterative manner. The masker learns to produce soft, task-relevant masks that highlight informative regions, while the classifier is optimized for both prediction accuracy and alignment between its saliency maps and the learned masks. By leveraging high-quality masks as guidance, ALIGN improves both interpretability and generalizability, showing its superiority across various settings. Experiments on the two domain generalization benchmarks, VLCS and Terra Incognita, show that ALIGN consistently outperforms six strong baselines in both in-distribution and out-of-distribution settings. Besides, ALIGN also yields superior explanation quality concerning sufficiency and comprehensiveness, highlighting its effectiveness in producing accurate and interpretable models. 

**Abstract (ZH)**: 基于解释指导的学习（EGL）在使模型预测与可解释的推理相一致方面显示出潜力，特别是在计算机视觉任务中。然而，大多数方法依赖于外部注释或基于启发式的分割来监督模型解释，这可能导致监督信号质量低下，不精确且难以扩展。在此工作中，我们提供了实证和理论证据，表明低质量的监督信号会损害模型性能，而非提升其性能。为此，我们提出了一种新颖的框架ALIGN，该框架以迭代方式联合训练分类器和掩码器。掩码器学习生成与任务相关、软化的掩码以突出显示信息区域，同时分类器被优化以最大化预测准确性和其显著图与学到的掩码之间的对齐性。通过利用高质量的掩码作为指导，ALIGN提高了可解释性和泛化能力，展示了其在各种环境中的优越性。在两个领域泛化基准VLCS和Terra Incognita上的实验表明，ALIGN在分布内和分布外设置中均优于六个强大的基线模型。此外，ALIGN在解释的充分性和全面性方面也表现出更优的质量，突出了其在生成准确且可解释模型方面的有效性。 

---
# PlantTraitNet: An Uncertainty-Aware Multimodal Framework for Global-Scale Plant Trait Inference from Citizen Science Data 

**Title (ZH)**: PlantTraitNet：一种面向公民科学数据的全球尺度植物性状推断的不确定性意识多模态框架 

**Authors**: Ayushi Sharma, Johanna Trost, Daniel Lusk, Johannes Dollinger, Julian Schrader, Christian Rossi, Javier Lopatin, Etienne Laliberté, Simon Haberstroh, Jana Eichel, Daniel Mederer, Jose Miguel Cerda-Paredes, Shyam S. Phartyal, Lisa-Maricia Schwarz, Anja Linstädter, Maria Conceição Caldeira, Teja Kattenborn  

**Link**: [PDF](https://arxiv.org/pdf/2511.06943)  

**Abstract**: Global plant maps of plant traits, such as leaf nitrogen or plant height, are essential for understanding ecosystem processes, including the carbon and energy cycles of the Earth system. However, existing trait maps remain limited by the high cost and sparse geographic coverage of field-based measurements. Citizen science initiatives offer a largely untapped resource to overcome these limitations, with over 50 million geotagged plant photographs worldwide capturing valuable visual information on plant morphology and physiology. In this study, we introduce PlantTraitNet, a multi-modal, multi-task uncertainty-aware deep learning framework that predictsfour key plant traits (plant height, leaf area, specific leaf area, and nitrogen content) from citizen science photos using weak supervision. By aggregating individual trait predictions across space, we generate global maps of trait distributions. We validate these maps against independent vegetation survey data (sPlotOpen) and benchmark them against leading global trait products. Our results show that PlantTraitNet consistently outperforms existing trait maps across all evaluated traits, demonstrating that citizen science imagery, when integrated with computer vision and geospatial AI, enables not only scalable but also more accurate global trait mapping. This approach offers a powerful new pathway for ecological research and Earth system modeling. 

**Abstract (ZH)**: 全球植物性状图谱：通过公民科学照片预测关键植物性状以理解地球系统过程 

---
# Fine-Tuning Diffusion-Based Recommender Systems via Reinforcement Learning with Reward Function Optimization 

**Title (ZH)**: 通过奖励函数优化的强化学习 Fine-Tuning 基于扩散的推荐系统 

**Authors**: Yu Hou, Hua Li, Ha Young Kim, Won-Yong Shin  

**Link**: [PDF](https://arxiv.org/pdf/2511.06937)  

**Abstract**: Diffusion models recently emerged as a powerful paradigm for recommender systems, offering state-of-the-art performance by modeling the generative process of user-item interactions. However, training such models from scratch is both computationally expensive and yields diminishing returns once convergence is reached. To remedy these challenges, we propose ReFiT, a new framework that integrates Reinforcement learning (RL)-based Fine-Tuning into diffusion-based recommender systems. In contrast to prior RL approaches for diffusion models depending on external reward models, ReFiT adopts a task-aligned design: it formulates the denoising trajectory as a Markov decision process (MDP) and incorporates a collaborative signal-aware reward function that directly reflects recommendation quality. By tightly coupling the MDP structure with this reward signal, ReFiT empowers the RL agent to exploit high-order connectivity for fine-grained optimization, while avoiding the noisy or uninformative feedback common in naive reward designs. Leveraging policy gradient optimization, ReFiT maximizes exact log-likelihood of observed interactions, thereby enabling effective post hoc fine-tuning of diffusion recommenders. Comprehensive experiments on wide-ranging real-world datasets demonstrate that the proposed ReFiT framework (a) exhibits substantial performance gains over strong competitors (up to 36.3% on sequential recommendation), (b) demonstrates strong efficiency with linear complexity in the number of users or items, and (c) generalizes well across multiple diffusion-based recommendation scenarios. The source code and datasets are publicly available at this https URL. 

**Abstract (ZH)**: 基于强化学习的重 Fine-Tuning 差异扩散模型框架 

---
# Sampling and Loss Weights in Multi-Domain Training 

**Title (ZH)**: 多域训练中的采样与损失权重 móidfèn xunliàn zhōng de cǎi’é shǔcè yǔ shīfèi zhòngliè 

**Authors**: Mahdi Salmani, Pratik Worah, Meisam Razaviyayn, Vahab Mirrokni  

**Link**: [PDF](https://arxiv.org/pdf/2511.06913)  

**Abstract**: In the training of large deep neural networks, there is a need for vast amounts of training data. To meet this need, data is collected from multiple domains, such as Wikipedia and GitHub. These domains are heterogeneous in both data quality and the diversity of information they provide. This raises the question of how much we should rely on each domain. Several methods have attempted to address this issue by assigning sampling weights to each data domain using heuristics or approximations. As a first step toward a deeper understanding of the role of data mixing, this work revisits the problem by studying two kinds of weights: sampling weights, which control how much each domain contributes in a batch, and loss weights, which scale the loss from each domain during training. Through a rigorous study of linear regression, we show that these two weights play complementary roles. First, they can reduce the variance of gradient estimates in iterative methods such as stochastic gradient descent (SGD). Second, they can improve generalization performance by reducing the generalization gap. We provide both theoretical and empirical support for these claims. We further study the joint dynamics of sampling weights and loss weights, examining how they can be combined to capture both contributions. 

**Abstract (ZH)**: 在大型深度神经网络训练中，需要大量的训练数据。为此，数据从多个领域收集，如Wikipedia和GitHub。这些领域在数据质量和提供的信息多样性方面存在异质性。这引发了我们应依赖每个领域多少的疑问。通过使用启发式方法或近似值为每个数据领域分配采样权重，已经有几种方法尝试解决这一问题。为了更深入地理解数据混合的作用，本工作重新审视了该问题，研究了两种类型的权重：采样权重，控制每个领域在批次中的贡献；损失权重，训练过程中调整每个领域损失的尺度。通过线性回归的严谨研究，我们表明这两种权重互补发挥作用。首先，它们可以减少迭代方法（如随机梯度下降SGD）中梯度估计的方差。第二，它们可以通过减少泛化差距来提高泛化性能。我们提供了理论和实验证据支持这些观点。进一步研究采样权重和损失权重的联合动态，探讨它们如何结合以捕捉各自的贡献。 

---
# Counterfactual Explanation for Multivariate Time Series Forecasting with Exogenous Variables 

**Title (ZH)**: 具有外生变量的多变量时间序列预测反事实解释 

**Authors**: Keita Kinjo  

**Link**: [PDF](https://arxiv.org/pdf/2511.06906)  

**Abstract**: Currently, machine learning is widely used across various domains, including time series data analysis. However, some machine learning models function as black boxes, making interpretability a critical concern. One approach to address this issue is counterfactual explanation (CE), which aims to provide insights into model predictions. This study focuses on the relatively underexplored problem of generating counterfactual explanations for time series forecasting. We propose a method for extracting CEs in time series forecasting using exogenous variables, which are frequently encountered in fields such as business and marketing. In addition, we present methods for analyzing the influence of each variable over an entire time series, generating CEs by altering only specific variables, and evaluating the quality of the resulting CEs. We validate the proposed method through theoretical analysis and empirical experiments, showcasing its accuracy and practical applicability. These contributions are expected to support real-world decision-making based on time series data analysis. 

**Abstract (ZH)**: 当前，机器学习在各个领域得到广泛应用，包括时间序列数据分析。然而，一些机器学习模型作为黑箱运行，使可解释性成为一个关键问题。解决这一问题的一种方法是反事实解释（CE），其旨在提供对模型预测的见解。本研究专注于生成时间序列预测的反事实解释这一相对较少研究的问题。我们提出了一种使用外生变量提取反事实解释的方法，此类变量在商业和营销等领域中较为常见。此外，我们还提出了分析每个变量在整个时间序列中的影响、通过改变特定变量生成反事实解释以及评估生成的反事实解释质量的方法。通过理论分析和实证实验验证了所提出的方法，展示了其准确性和实用性。这些贡献有望支持基于时间序列数据分析的现实世界决策。 

---
# RPTS: Tree-Structured Reasoning Process Scoring for Faithful Multimodal Evaluation 

**Title (ZH)**: RPTS: 基于树结构推理过程评分的忠实多模态评价 

**Authors**: Haofeng Wang, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06899)  

**Abstract**: Large Vision-Language Models (LVLMs) excel in multimodal reasoning and have shown impressive performance on various multimodal benchmarks. However, most of these benchmarks evaluate models primarily through multiple-choice or short-answer formats, which do not take the reasoning process into account. Although some benchmarks assess the reasoning process, their methods are often overly simplistic and only examine reasoning when answers are incorrect. This approach overlooks scenarios where flawed reasoning leads to correct answers. In addition, these benchmarks do not consider the impact of intermodal relationships on reasoning. To address this issue, we propose the Reasoning Process Tree Score (RPTS), a tree structure-based metric to assess reasoning processes. Specifically, we organize the reasoning steps into a reasoning tree and leverage its hierarchical information to assign weighted faithfulness scores to each reasoning step. By dynamically adjusting these weights, RPTS not only evaluates the overall correctness of the reasoning, but also pinpoints where the model fails in the reasoning. To validate RPTS in real-world multimodal scenarios, we construct a new benchmark, RPTS-Eval, comprising 374 images and 390 reasoning instances. Each instance includes reliable visual-textual clues that serve as leaf nodes of the reasoning tree. Furthermore, we define three types of intermodal relationships to investigate how intermodal interactions influence the reasoning process. We evaluated representative LVLMs (e.g., GPT4o, Llava-Next), uncovering their limitations in multimodal reasoning and highlighting the differences between open-source and closed-source commercial LVLMs. We believe that this benchmark will contribute to the advancement of research in the field of multimodal reasoning. 

**Abstract (ZH)**: 大型多模态语言模型（LVLMs）在多模态推理方面表现卓越，并在各种多模态基准测试中显示出 impressive 的性能。然而，大多数这些基准测试主要通过选择题或简答格式评估模型，而不考虑推理过程。虽然有一些基准测试评估推理过程，但其方法往往过于简单，并且仅在答案错误时才检查推理。这种做法忽视了推理错误但答案正确的场景。此外，这些基准测试并未考虑跨模态关系对推理的影响。为解决这一问题，我们提出了一种基于树结构的评估推理过程的指标——推理过程树评分（RPTS），通过将推理步骤组织成一个推理树，并利用其层次信息为每个推理步骤分配加权忠实度评分，动态调整这些权重不仅评估整体推理的正确性，还能指出模型推理中的失败之处。为了在真实世界的多模态场景中验证 RPTS，我们构建了一个新的基准测试 RPTS-Eval，包含 374 张图像和 390 个推理实例。每个实例包括可靠的视觉-文本线索，作为推理树的叶节点。此外，我们定义了三种类型的跨模态关系，以探究跨模态互动如何影响推理过程。我们评估了代表性的 LVLMs（例如 GPT4o、Llava-Next），揭示了它们在多模态推理中的局限性，并突出了开源和闭源商业 LVLMs 之间的差异。我们相信，该基准测试将促进多模态推理领域的研究进展。 

---
# A Hybrid Autoencoder-Transformer Model for Robust Day-Ahead Electricity Price Forecasting under Extreme Conditions 

**Title (ZH)**: 一种在极端条件下用于稳健的日-ahead 电力价格预测的混合自编码器-变换器模型 

**Authors**: Boyan Tang, Xuanhao Ren, Peng Xiao, Shunbo Lei, Xiaorong Sun, Jianghua Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06898)  

**Abstract**: Accurate day-ahead electricity price forecasting (DAEPF) is critical for the efficient operation of power systems, but extreme condition and market anomalies pose significant challenges to existing forecasting methods. To overcome these challenges, this paper proposes a novel hybrid deep learning framework that integrates a Distilled Attention Transformer (DAT) model and an Autoencoder Self-regression Model (ASM). The DAT leverages a self-attention mechanism to dynamically assign higher weights to critical segments of historical data, effectively capturing both long-term trends and short-term fluctuations. Concurrently, the ASM employs unsupervised learning to detect and isolate anomalous patterns induced by extreme conditions, such as heavy rain, heat waves, or human festivals. Experiments on datasets sampled from California and Shandong Province demonstrate that our framework significantly outperforms state-of-the-art methods in prediction accuracy, robustness, and computational efficiency. Our framework thus holds promise for enhancing grid resilience and optimizing market operations in future power systems. 

**Abstract (ZH)**: 准确的次日电价预测（DAEPF）对电力系统的高效运行至关重要，但极端条件和市场异常给现有预测方法造成了巨大挑战。为克服这些挑战，本文提出了一种新颖的混合深度学习框架，该框架结合了蒸馏注意力变压器（DAT）模型和自编码自回归模型（ASM）。DAT利用自我注意力机制动态分配更高权重给历史数据中的关键部分，有效捕捉长期趋势和短期波动。同时，ASM运用无监督学习检测并隔离由极端条件（如暴雨、热浪或人为节日）引起的异常模式。实验结果表明，本框架在预测准确性、鲁棒性和计算效率方面显著优于现有方法。因此，该框架为提高未来电力系统的电网弹性和优化市场运营提供了前景。 

---
# On The Presence of Double-Descent in Deep Reinforcement Learning 

**Title (ZH)**: 深入强化学习中双下降现象的存在性 exploring the presence of double-descent in deep reinforcement learning 

**Authors**: Viktor Veselý, Aleksandar Todorov, Matthia Sabatelli  

**Link**: [PDF](https://arxiv.org/pdf/2511.06895)  

**Abstract**: The double descent (DD) paradox, where over-parameterized models see generalization improve past the interpolation point, remains largely unexplored in the non-stationary domain of Deep Reinforcement Learning (DRL). We present preliminary evidence that DD exists in model-free DRL, investigating it systematically across varying model capacity using the Actor-Critic framework. We rely on an information-theoretic metric, Policy Entropy, to measure policy uncertainty throughout training. Preliminary results show a clear epoch-wise DD curve; the policy's entrance into the second descent region correlates with a sustained, significant reduction in Policy Entropy. This entropic decay suggests that over-parameterization acts as an implicit regularizer, guiding the policy towards robust, flatter minima in the loss landscape. These findings establish DD as a factor in DRL and provide an information-based mechanism for designing agents that are more general, transferable, and robust. 

**Abstract (ZH)**: 非平稳域深度强化学习（DRL）中的双下降（DD）悖论：关于模型自由DRL的初步证据 

---
# COGNOS: Universal Enhancement for Time Series Anomaly Detection via Constrained Gaussian-Noise Optimization and Smoothing 

**Title (ZH)**: COGNOS:面向时间序列异常检测的约束高斯噪声优化与平滑通用增强方法 

**Authors**: Wenlong Shang, Peng Chang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06894)  

**Abstract**: Reconstruction-based methods are a dominant paradigm in time series anomaly detection (TSAD), however, their near-universal reliance on Mean Squared Error (MSE) loss results in statistically flawed reconstruction residuals. This fundamental weakness leads to noisy, unstable anomaly scores with a poor signal-to-noise ratio, hindering reliable detection. To address this, we propose Constrained Gaussian-Noise Optimization and Smoothing (COGNOS), a universal, model-agnostic enhancement framework that tackles this issue at its source. COGNOS introduces a novel Gaussian-White Noise Regularization strategy during training, which directly constrains the model's output residuals to conform to a Gaussian white noise distribution. This engineered statistical property creates the ideal precondition for our second contribution: a Kalman Smoothing Post-processor that provably operates as a statistically optimal estimator to denoise the raw anomaly scores. The synergy between these two components allows COGNOS to robustly separate the true anomaly signal from random fluctuations. Extensive experiments demonstrate that COGNOS is highly effective, delivering an average F-score uplift of 57.9% when applied to 12 diverse backbone models across multiple real-world benchmark datasets. Our work reveals that directly regularizing output statistics is a powerful and generalizable strategy for significantly improving anomaly detection systems. 

**Abstract (ZH)**: 基于重构的方法是时间序列异常检测（TSAD）中的主导范式，然而，它们普遍依赖均方误差（MSE）损失导致重构残差的统计瑕疵。这一根本弱点使得异常得分噪声大、不稳定，信噪比差，妨碍可靠检测。为此，我们提出了一种通用的模型无偏增强框架Constrained Gaussian-Noise Optimization and Smoothing（COGNOS），从根源上解决这一问题。COGNOS 在训练过程中引入了一种新颖的高斯白噪声正则化策略，直接约束模型的输出残差符合高斯白噪声分布。这种工程化的统计特性为我们第二项贡献——可证明作为最优估计器去降噪原始异常得分的卡尔曼平滑后处理器创造了理想的先决条件。这两个组件之间的协同作用使COGNOS能够稳健地分离出真实的异常信号和随机波动。广泛的实验表明，COGNOS 非常有效，在多个真实世界基准数据集的12个不同主干模型上应用时，平均提升了57.9%的F分数。我们的工作表明，直接正则化输出统计是一种强大且可泛化的策略，可以显著提高异常检测系统的性能。 

---
# DeepBooTS: Dual-Stream Residual Boosting for Drift-Resilient Time-Series Forecasting 

**Title (ZH)**: DeepBooTS：具备漂移鲁棒性的时序预测的双流残差增强方法 

**Authors**: Daojun Liang, Jing Chen, Xiao Wang, Yinglong Wang, Suo Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06893)  

**Abstract**: Time-Series (TS) exhibits pronounced non-stationarity. Consequently, most forecasting methods display compromised robustness to concept drift, despite the prevalent application of instance normalization. We tackle this challenge by first analysing concept drift through a bias-variance lens and proving that weighted ensemble reduces variance without increasing bias. These insights motivate DeepBooTS, a novel end-to-end dual-stream residual-decreasing boosting method that progressively reconstructs the intrinsic signal. In our design, each block of a deep model becomes an ensemble of learners with an auxiliary output branch forming a highway to the final prediction. The block-wise outputs correct the residuals of previous blocks, leading to a learning-driven decomposition of both inputs and targets. This method enhances versatility and interpretability while substantially improving robustness to concept drift. Extensive experiments, including those on large-scale datasets, show that the proposed method outperforms existing methods by a large margin, yielding an average performance improvement of 15.8% across various datasets, establishing a new benchmark for TS forecasting. 

**Abstract (ZH)**: Time-Series (TS) 显示出明显的非平稳性。因此，尽管广泛应用实例归一化，大多数预测方法在概念漂移面前仍表现出较低的稳健性。我们通过偏倚方差视角分析概念漂移，并证明加权集成可以降低方差而不增加偏倚。这些见解促使我们提出 DeepBooTS，一种新颖的端到端双流残差递减提升方法，该方法逐步重建内在信号。在我们的设计中，深度模型中的每一层都成为包含辅助输出分支的学习者集合，该辅助输出分支形成到最终预测的高速通道。各层的输出修正了前一层的残差，从而实现了输入和目标的由学习驱动的分解。该方法增强了可适性和可解释性，同时显著提高了对概念漂移的稳健性。大量实验，包括在大型数据集上的实验，表明提出的方法在很大程度上优于现有方法，在不同数据集上平均性能提高了15.8%，确立了TS预测的新基准。 

---
# TuckA: Hierarchical Compact Tensor Experts for Efficient Fine-Tuning 

**Title (ZH)**: TuckA: 分级紧凑张量专家以实现高效的微调 

**Authors**: Qifeng Lei, Zhiyong Yang, Qianqian Xu, Cong Hua, Peisong Wen, Qingming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06859)  

**Abstract**: Efficiently fine-tuning pre-trained models for downstream tasks is a key challenge in the era of foundation models. Parameter-efficient fine-tuning (PEFT) presents a promising solution, achieving performance comparable to full fine-tuning by updating only a small number of adaptation weights per layer. Traditional PEFT methods typically rely on a single expert, where the adaptation weight is a low-rank matrix. However, for complex tasks, the data's inherent diversity poses a significant challenge for such models, as a single adaptation weight cannot adequately capture the features of all samples. To address this limitation, we explore how to integrate multiple small adaptation experts into a compact structure to defeat a large adapter. Specifically, we propose Tucker Adaptation (TuckA), a method with four key properties: (i) We use Tucker decomposition to create a compact 3D tensor where each slice naturally serves as an expert. The low-rank nature of this decomposition ensures that the number of parameters scales efficiently as more experts are added. (ii) We introduce a hierarchical strategy that organizes these experts into groups at different granularities, allowing the model to capture both local and global data patterns. (iii) We develop an efficient batch-level routing mechanism, which reduces the router's parameter size by a factor of $L$ compared to routing at every adapted layer (where $L$ is the number of adapted layers) (iv) We propose data-aware initialization to achieve loss-free expert load balancing based on theoretical analysis. Extensive experiments on benchmarks in natural language understanding, image classification, and mathematical reasoning speak to the efficacy of TuckA, offering a new and effective solution to the PEFT problem. 

**Abstract (ZH)**: 高效微调预训练模型是基础模型时代的关键挑战。参数高效微调（PEFT）提供了一种有希望的解决方案，通过每层仅更新少量适应权重，即可达到与完全微调相当的性能。传统的PEFT方法通常依赖单一专家，其中适应权重是一个低秩矩阵。然而，对于复杂的任务，数据的固有多样性对这些模型构成了重大挑战，因为单个适应权重无法充分捕捉所有样本的特征。为解决这一局限性，我们探索如何将多个小型适应专家整合到紧凑结构中以击败大适配器。具体而言，我们提出Tucker适应（TuckA）方法，具有四个关键特性：（i）我们使用Tucker分解创建一个紧凑的3D张量，其中每个切片自然充当专家。这种分解的低秩性质确保了随着添加更多专家，参数数量能够高效扩展。（ii）我们引入一种分层策略，将这些专家按不同的粒度组织成组，从而使模型能够捕捉局部和全局数据模式。（iii）我们开发了一种高效的批量级路由机制，与在每个适配层进行路由相比，参数量减少了$L$倍（$L$为适配层的数量）。（iv）我们提出数据感知初始化，基于理论分析实现无损专家负载平衡。针对自然语言理解、图像分类和数学推理基准的广泛实验表明，TuckA的有效性，提供了PEFT问题的一种新且有效的解决方案。 

---
# Deep learning EPI-TIRF cross-modality enables background subtraction and axial super-resolution for widefield fluorescence microscopy 

**Title (ZH)**: 深度学习EPI-TIRF跨模态技术实现宽场荧光显微镜的背景消除和轴向超分辨 

**Authors**: Qiushi Li, Celi Lou, Yanfang Cheng, Bilang Gong, Xinlin Chen, Hao Chen, Baowan Li, Jieli Wang, Yulin Wang, Sipeng Yang, Yunqing Tang, Luru Dai  

**Link**: [PDF](https://arxiv.org/pdf/2511.06853)  

**Abstract**: The resolving ability of wide-field fluorescence microscopy is fundamentally limited by out-of-focus background owing to its low axial resolution, particularly for densely labeled biological samples. To address this, we developed ET2dNet, a deep learning-based EPI-TIRF cross-modality network that achieves TIRF-comparable background subtraction and axial super-resolution from a single wide-field image without requiring hardware modifications. The model employs a physics-informed hybrid architecture, synergizing supervised learning with registered EPI-TIRF image pairs and self-supervised physical modeling via convolution with the point spread function. This framework ensures exceptional generalization across microscope objectives, enabling few-shot adaptation to new imaging setups. Rigorous validation on cellular and tissue samples confirms ET2dNet's superiority in background suppression and axial resolution enhancement, while maintaining compatibility with deconvolution techniques for lateral resolution improvement. Furthermore, by extending this paradigm through knowledge distillation, we developed ET3dNet, a dedicated three-dimensional reconstruction network that produces artifact-reduced volumetric results. ET3dNet effectively removes out-of-focus background signals even when the input image stack lacks the source of background. This framework makes axial super-resolution imaging more accessible by providing an easy-to-deploy algorithm that avoids additional hardware costs and complexity, showing great potential for live cell studies and clinical histopathology. 

**Abstract (ZH)**: 基于深度学习的EPI-TIRF跨模态网络ET2dNet：单张宽场图像的三维背景减除与轴向超分辨 

---
# Differentiated Directional Intervention A Framework for Evading LLM Safety Alignment 

**Title (ZH)**: 差异化方向性干预：一种规避LLM安全对齐的框架 

**Authors**: Peng Zhang, peijie sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.06852)  

**Abstract**: Safety alignment instills in Large Language Models (LLMs) a critical capacity to refuse malicious requests. Prior works have modeled this refusal mechanism as a single linear direction in the activation space. We posit that this is an oversimplification that conflates two functionally distinct neural processes: the detection of harm and the execution of a refusal. In this work, we deconstruct this single representation into a Harm Detection Direction and a Refusal Execution Direction. Leveraging this fine-grained model, we introduce Differentiated Bi-Directional Intervention (DBDI), a new white-box framework that precisely neutralizes the safety alignment at critical layer. DBDI applies adaptive projection nullification to the refusal execution direction while suppressing the harm detection direction via direct steering. Extensive experiments demonstrate that DBDI outperforms prominent jailbreaking methods, achieving up to a 97.88\% attack success rate on models such as Llama-2. By providing a more granular and mechanistic framework, our work offers a new direction for the in-depth understanding of LLM safety alignment. 

**Abstract (ZH)**: 安全对齐赋予大型语言模型（LLMs）一种关键能力，即拒绝恶意请求。先前的工作将这种拒绝机制建模为激活空间中的单一线性方向。我们认为这是对两个功能上不同的神经过程的简化综合：危害检测和拒绝执行。在这项工作中，我们将这种单一表示分解为危害检测方向和拒绝执行方向。基于这一细粒度模型，我们引入了差异化双向干预（DBDI），这是一种新的白盒框架，精确地中和了关键层的安全对齐。DBDI在拒绝执行方向上应用自适应投影消除，同时通过直接控制抑制危害检测方向。广泛的实验表明，DBDI在如Llama-2等模型上的攻击成功率高达97.88%。通过提供更精细和机制化的框架，我们的工作为LLM安全对齐的深入理解提供了新的方向。 

---
# NeuroBridge: Bio-Inspired Self-Supervised EEG-to-Image Decoding via Cognitive Priors and Bidirectional Semantic Alignment 

**Title (ZH)**: NeuroBridge: 生物启发的自监督EEG到图像解码，基于认知先验和双向语义对齐 

**Authors**: Wenjiang Zhang, Sifeng Wang, Yuwei Su, Xinyu Li, Chen Zhang, Suyu Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2511.06836)  

**Abstract**: Visual neural decoding seeks to reconstruct or infer perceived visual stimuli from brain activity patterns, providing critical insights into human cognition and enabling transformative applications in brain-computer interfaces and artificial intelligence. Current approaches, however, remain constrained by the scarcity of high-quality stimulus-brain response pairs and the inherent semantic mismatch between neural representations and visual content. Inspired by perceptual variability and co-adaptive strategy of the biological systems, we propose a novel self-supervised architecture, named NeuroBridge, which integrates Cognitive Prior Augmentation (CPA) with Shared Semantic Projector (SSP) to promote effective cross-modality alignment. Specifically, CPA simulates perceptual variability by applying asymmetric, modality-specific transformations to both EEG signals and images, enhancing semantic diversity. Unlike previous approaches, SSP establishes a bidirectional alignment process through a co-adaptive strategy, which mutually aligns features from two modalities into a shared semantic space for effective cross-modal learning. NeuroBridge surpasses previous state-of-the-art methods under both intra-subject and inter-subject settings. In the intra-subject scenario, it achieves the improvements of 12.3% in top-1 accuracy and 10.2% in top-5 accuracy, reaching 63.2% and 89.9% respectively on a 200-way zero-shot retrieval task. Extensive experiments demonstrate the effectiveness, robustness, and scalability of the proposed framework for neural visual decoding. 

**Abstract (ZH)**: 视觉神经解码旨在从脑活动模式重构或推断感知的视觉刺激，为人类认知提供关键见解，并在脑机接口和人工智能等领域实现变革性应用。然而，当前方法仍然受限于高质量刺激-脑响应配对稀缺以及神经表示与视觉内容固有的语义不匹配问题。受感知变异性和生物系统共适应策略的启发，我们提出一种新颖的自监督架构——NeuroBridge，该架构将认知先验增强（CPA）与共享语义投影器（SSP）相结合，促进有效的跨模态对齐。具体而言，CPA 通过在 EEG 信号和图像之间应用不对称的模态特定变换来模拟感知变异，增强语义多样性。与以往方法不同，SSP 通过一种共适应策略建立了双向对齐过程，使两种模态的特征相互对齐到一个共享语义空间，实现有效的跨模态学习。在跨被试测试中，NeuroBridge 在准确率方面超过了之前的所有方法。在单被试场景中，其在 top-1 准确率和 top-5 准确率上分别提高了 12.3% 和 10.2%，分别达到 63.2% 和 89.9%，在 200 类零样本检索任务上。大量实验验证了所提出框架在神经视觉解码中的有效性、鲁棒性和可扩展性。 

---
# DeepRWCap: Neural-Guided Random-Walk Capacitance Solver for IC Design 

**Title (ZH)**: DeepRWCap: 基于神经网络引导的随机漫步电容求解器在IC设计中的应用 

**Authors**: Hector R. Rodriguez, Jiechen Huang, Wenjian Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06831)  

**Abstract**: Monte Carlo random walk methods are widely used in capacitance extraction for their mesh-free formulation and inherent parallelism. However, modern semiconductor technologies with densely packed structures present significant challenges in unbiasedly sampling transition domains in walk steps with multiple high-contrast dielectric materials. We present DeepRWCap, a machine learning-guided random walk solver that predicts the transition quantities required to guide each step of the walk. These include Poisson kernels, gradient kernels, signs and magnitudes of weights. DeepRWCap employs a two-stage neural architecture that decomposes structured outputs into face-wise distributions and spatial kernels on cube faces. It uses 3D convolutional networks to capture volumetric dielectric interactions and 2D depthwise separable convolutions to model localized kernel behavior. The design incorporates grid-based positional encodings and structural design choices informed by cube symmetries to reduce learning redundancy and improve generalization. Trained on 100,000 procedurally generated dielectric configurations, DeepRWCap achieves a mean relative error of $1.24\pm0.53$\% when benchmarked against the commercial Raphael solver on the self-capacitance estimation of 10 industrial designs spanning 12 to 55 nm nodes. Compared to the state-of-the-art stochastic difference method Microwalk, DeepRWCap achieves an average 23\% speedup. On complex designs with runtimes over 10 s, it reaches an average 49\% acceleration. 

**Abstract (ZH)**: 基于深度学习的随机行走电容提取方法DeepRWCap 

---
# Beyond Plain Demos: A Demo-centric Anchoring Paradigm for In-Context Learning in Alzheimer's Disease Detection 

**Title (ZH)**: 超越普通演示：面向演示的上下文学习 paradigm 在阿尔茨海默病检测中的应用 

**Authors**: Puzhen Su, Haoran Yin, Yongzhu Miao, Jintao Tang, Shasha Li, Ting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06826)  

**Abstract**: Detecting Alzheimer's disease (AD) from narrative transcripts challenges large language models (LLMs): pre-training rarely covers this out-of-distribution task, and all transcript demos describe the same scene, producing highly homogeneous contexts. These factors cripple both the model's built-in task knowledge (\textbf{task cognition}) and its ability to surface subtle, class-discriminative cues (\textbf{contextual perception}). Because cognition is fixed after pre-training, improving in-context learning (ICL) for AD detection hinges on enriching perception through better demonstration (demo) sets. We demonstrate that standard ICL quickly saturates, its demos lack diversity (context width) and fail to convey fine-grained signals (context depth), and that recent task vector (TV) approaches improve broad task adaptation by injecting TV into the LLMs' hidden states (HSs), they are ill-suited for AD detection due to the mismatch of injection granularity, strength and position. To address these bottlenecks, we introduce \textbf{DA4ICL}, a demo-centric anchoring framework that jointly expands context width via \emph{\textbf{Diverse and Contrastive Retrieval}} (DCR) and deepens each demo's signal via \emph{\textbf{Projected Vector Anchoring}} (PVA) at every Transformer layer. Across three AD benchmarks, DA4ICL achieves large, stable gains over both ICL and TV baselines, charting a new paradigm for fine-grained, OOD and low-resource LLM adaptation. 

**Abstract (ZH)**: 从叙事转录中检测阿尔茨海默病（AD）挑战大型语言模型（LLMs）：预训练鲜少涵盖此类离分布任务，且所有转录示例描述相同场景，产生高度同质化背景。这些因素削弱了模型内置的任务认知（task cognition）和上下文感知（contextual perception）能力。由于认知在预训练后固定，通过改进在情景学习（ICL）中的表现以提升AD检测依赖于通过更高质量的示例集（demo sets）丰富感知。我们证明标准ICL很快达到饱和，其示例缺乏多样性（背景宽度），未能传达细微的信号（背景深度），而近期的任务向量（TV）方法通过将TV注入LLM的隐藏状态（HSs）以增强广泛的任务适应性，但由于注入的颗粒度、强度和位置的不匹配，这些方法对于AD检测并不适用。为克服这些瓶颈，我们引入了DA4ICL，这是一种以示例为中心的耦合框架，通过多样对比检索（DCR）扩展背景宽度，并在每个变换器层通过投影向量锚定（PVA）加深每个示例的信号，从而为大规模、离分布和低资源LLM的适应开辟新范式。 

---
# TiS-TSL: Image-Label Supervised Surgical Video Stereo Matching via Time-Switchable Teacher-Student Learning 

**Title (ZH)**: TiS-TSL：基于时间可切换教师-学生学习的图像标签监督手术视频立体匹配 

**Authors**: Rui Wang, Ying Zhou, Hao Wang, Wenwei Zhang, Qiang Li, Zhiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06817)  

**Abstract**: Stereo matching in minimally invasive surgery (MIS) is essential for next-generation navigation and augmented reality. Yet, dense disparity supervision is nearly impossible due to anatomical constraints, typically limiting annotations to only a few image-level labels acquired before the endoscope enters deep body cavities. Teacher-Student Learning (TSL) offers a promising solution by leveraging a teacher trained on sparse labels to generate pseudo labels and associated confidence maps from abundant unlabeled surgical videos. However, existing TSL methods are confined to image-level supervision, providing only spatial confidence and lacking temporal consistency estimation. This absence of spatio-temporal reliability results in unstable disparity predictions and severe flickering artifacts across video frames. To overcome these challenges, we propose TiS-TSL, a novel time-switchable teacher-student learning framework for video stereo matching under minimal supervision. At its core is a unified model that operates in three distinct modes: Image-Prediction (IP), Forward Video-Prediction (FVP), and Backward Video-Prediction (BVP), enabling flexible temporal modeling within a single architecture. Enabled by this unified model, TiS-TSL adopts a two-stage learning strategy. The Image-to-Video (I2V) stage transfers sparse image-level knowledge to initialize temporal modeling. The subsequent Video-to-Video (V2V) stage refines temporal disparity predictions by comparing forward and backward predictions to calculate bidirectional spatio-temporal consistency. This consistency identifies unreliable regions across frames, filters noisy video-level pseudo labels, and enforces temporal coherence. Experimental results on two public datasets demonstrate that TiS-TSL exceeds other image-based state-of-the-arts by improving TEPE and EPE by at least 2.11% and 4.54%, respectively.. 

**Abstract (ZH)**: 最小侵入手术（MIS）中的立体匹配对于下一代导航和增强现实至关重要。然而，由于解剖限制，密集的视差监督几乎是不可能的，通常仅限于内窥镜进入深部腔体前获取的少量图像级标签。通过利用在稀疏标签上训练的教师生成伪标签及其相应的置信图，教师-学生学习（TSL）提供了一种有前景的解决方案，可以从大量未标记的手术视频中获取。然而，现有的TSL方法局限于图像级监督，只能提供空间置信度，缺乏时间一致性估计。这种时空可靠性缺失导致了不稳定的视差预测和严重的视频帧间闪烁伪影。为克服这些挑战，我们提出了TiS-TSL，一种针对最少监督下的视频立体匹配的新型时间切换教师-学生学习框架。其核心是一个统一模型，在三种不同模式下运行：图像预测（IP）、前向视频预测（FVP）和后向视频预测（BVP），在单一架构中实现灵活的时间建模。基于这一统一模型，TiS-TSL采用了两阶段学习策略。图像到视频（I2V）阶段将稀疏的图像级知识转移以初始化时间建模。随后的视频到视频（V2V）阶段通过比较前向和后向预测来计算双向时空一致性，从而细化时间视差预测。这种一致性识别了帧间不可靠区域，过滤掉噪声的视频级伪标签，并强制执行时间一致性和连贯性。在两个公开数据集上的实验结果表明，TiS-TSL在提高TEPE和EPE方面超过了其他基于图像的状态-of-the-arts方法，分别提高了至少2.11%和4.54%。 

---
# Controllable Flow Matching for Online Reinforcement Learning 

**Title (ZH)**: 可控流匹配的在线强化学习 

**Authors**: Bin Wang, Boxiang Tao, Haifeng Jing, Hongbo Dou, Zijian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06816)  

**Abstract**: Model-based reinforcement learning (MBRL) typically relies on modeling environment dynamics for data efficiency. However, due to the accumulation of model errors over long-horizon rollouts, such methods often face challenges in maintaining modeling stability. To address this, we propose CtrlFlow, a trajectory-level synthetic method using conditional flow matching (CFM), which directly modeling the distribution of trajectories from initial states to high-return terminal states without explicitly modeling the environment transition function. Our method ensures optimal trajectory sampling by minimizing the control energy governed by the non-linear Controllability Gramian Matrix, while the generated diverse trajectory data significantly enhances the robustness and cross-task generalization of policy learning. In online settings, CtrlFlow demonstrates the better performance on common MuJoCo benchmark tasks than dynamics models and achieves superior sample efficiency compared to standard MBRL methods. 

**Abstract (ZH)**: 基于模型的强化学习（MBRL）通常依赖于环境动力学建模以提高数据效率。然而，由于长期轨迹采样中模型误差的累积，此类方法往往难以保持建模稳定性。为此，我们提出了CtrlFlow，一种基于轨迹级合成的方法，使用条件流匹配（CFM）直接建模从初始状态到高回报终态的轨迹分布，而不显式建模环境转换函数。该方法通过最小化由非线性可控性Gram矩阵支配的控制能量来确保最优轨迹采样，而生成的多样化轨迹数据显著增强了策略学习的稳健性和跨任务泛化能力。在线设置中，CtrlFlow在常见的MuJoCo基准任务上表现出更好的性能，相比动力学模型具有更高的样本效率，并且优于标准的MBRL方法。 

---
# AgentSUMO: An Agentic Framework for Interactive Simulation Scenario Generation in SUMO via Large Language Models 

**Title (ZH)**: AgentSUMO: 一种基于大型语言模型的SUMO互动仿真场景生成框架 

**Authors**: Minwoo Jeong, Jeeyun Chang, Yoonjin Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2511.06804)  

**Abstract**: The growing complexity of urban mobility systems has made traffic simulation indispensable for evidence-based transportation planning and policy evaluation. However, despite the analytical capabilities of platforms such as the Simulation of Urban MObility (SUMO), their application remains largely confined to domain experts. Developing realistic simulation scenarios requires expertise in network construction, origin-destination modeling, and parameter configuration for policy experimentation, creating substantial barriers for non-expert users such as policymakers, urban planners, and city officials. Moreover, the requests expressed by these users are often incomplete and abstract-typically articulated as high-level objectives, which are not well aligned with the imperative, sequential workflows employed in existing language-model-based simulation frameworks. To address these challenges, this study proposes AgentSUMO, an agentic framework for interactive simulation scenario generation via large language models. AgentSUMO departs from imperative, command-driven execution by introducing an adaptive reasoning layer that interprets user intents, assesses task complexity, infers missing parameters, and formulates executable simulation plans. The framework is structured around two complementary components, the Interactive Planning Protocol, which governs reasoning and user interaction, and the Model Context Protocol, which manages standardized communication and orchestration among simulation tools. Through this design, AgentSUMO converts abstract policy objectives into executable simulation scenarios. Experiments on urban networks in Seoul and Manhattan demonstrate that the agentic workflow achieves substantial improvements in traffic flow metrics while maintaining accessibility for non-expert users, successfully bridging the gap between policy goals and executable simulation workflows. 

**Abstract (ZH)**: 城市交通系统的日益复杂使得交通仿真对于基于证据的交通规划和政策评价变得不可或缺。然而，尽管有如Simulation of Urban MObility (SUMO)等平台的分析能力，其应用仍主要局限于领域专家。为开发现实的仿真场景，非专家用户（如政策制定者、城市规划师和城市官员）需要具备网络构建、出行生成建模和政策实验参数配置的专业知识，这创造了相当大的障碍。此外，这些用户的请求往往不完整且抽象，通常仅以高层次目标的形式提出，而这些目标与现有基于语言模型的仿真框架中所采用的严格、顺序化的作业流程并不很好对接。为解决这些挑战，本研究提出AgentSUMO，这是一种通过大型语言模型进行交互式仿真场景生成的代理框架。AgentSUMO通过引入一个适应性推理层，该层解释用户意图、评估任务复杂性、推断缺失参数并制定可执行的仿真计划，从而偏离了基于命令的执行方式。该框架围绕两个互补组件构建，交互式规划协议规范推理和用户交互，而模型上下文协议则管理仿真工具之间的标准化通信和协调。通过该设计，AgentSUMO将抽象的政策目标转化为可执行的仿真场景。对首尔和曼哈顿城市网络的实验表明，代理式作业流程在保持非专家用户可用性的同时，在交通流指标上实现了显著改善，成功地弥合了政策目标与可执行仿真作业流程之间的差距。 

---
# Learning to Fast Unrank in Collaborative Filtering Recommendation 

**Title (ZH)**: 学习快速未排序在协作过滤推荐中 

**Authors**: Junpeng Zhao, Lin Li, Ming Li, Amran Bhuiyan, Jimmy Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06803)  

**Abstract**: Modern data-driven recommendation systems risk memorizing sensitive user behavioral patterns, raising privacy concerns. Existing recommendation unlearning methods, while capable of removing target data influence, suffer from inefficient unlearning speed and degraded performance, failing to meet real-time unlearning demands. Considering the ranking-oriented nature of recommendation systems, we present unranking, the process of reducing the ranking positions of target items while ensuring the formal guarantees of recommendation unlearning. To achieve efficient unranking, we propose Learning to Fast Unrank in Collaborative Filtering Recommendation (L2UnRank), which operates through three key stages: (a) identifying the influenced scope via interaction-based p-hop propagation, (b) computing structural and semantic influences for entities within this scope, and (c) performing efficient, ranking-aware parameter updates guided by influence information. Extensive experiments across multiple datasets and backbone models demonstrate L2UnRank's model-agnostic nature, achieving state-of-the-art unranking effectiveness and maintaining recommendation quality comparable to retraining, while also delivering a 50x speedup over existing methods. Codes are available at this https URL. 

**Abstract (ZH)**: 现代数据驱动的推荐系统存在记忆敏感用户行为模式的风险，引发隐私担忧。现有推荐遗忘方法虽然能够去除目标数据的影响，但在去除效率和性能方面存在不足，无法满足实时遗忘的需求。考虑到推荐系统的排序性质，我们提出了去排序（unranking）的概念，即在确保推荐遗忘形式保证的前提下，降低目标项目的位置排名。为了实现高效的去排序，我们提出了基于协同过滤推荐的快速去排序学习（Learning to Fast Unrank in Collaborative Filtering Recommendation，L2UnRank），该方法通过三个关键阶段实现：（a）基于交互的p-hop传播确定影响范围，（b）计算该范围内的实体的结构和语义影响，（c）根据影响信息进行高效且排名意识的参数更新。跨多个数据集和底层模型的广泛实验表明，L2UnRank具有模型无关性，达到最优的去排序效果，同时保持与重新训练相当的推荐质量，相比现有方法还实现了50倍的速度提升。代码可在以下链接获取。 

---
# Recursive Dynamics in Fast-Weights Homeostatic Reentry Networks: Toward Reflective Intelligence 

**Title (ZH)**: 快速权重自稳回环网络中的递归动力学：迈向反思智能 

**Authors**: B. G. Chae  

**Link**: [PDF](https://arxiv.org/pdf/2511.06798)  

**Abstract**: This study introduces the Fast-Weights Homeostatic Reentry Layer (FH-RL), a neural mechanism that integrates fast-weight associative memory, homeostatic regularization, and learned reentrant feedback to approximate self-referential computation in neural networks. Unlike standard transformer architectures that operate in a purely feedforward manner during inference, FH-RL enables internal recurrence without external looping, allowing prior latent states to be dynamically re-entered into the ongoing computation stream. We conduct controlled experiments sweeping the reentry gain $\gamma$ and evaluate emergent internal dynamics using three novel metrics: the Information Reentry Ratio (IRR), Eigen-Spectrum Recursion Index (ESRI), and Representational Drift Periodicity (RDP). Results show that reentry quantity increases proportionally with~$\gamma$, while the learned feedback matrix $W_r$ remains bounded and becomes more structured at moderate gains. Critically, a stable reflective band emerges around $\gamma \approx 0.10-0.20$, where internal feedback is maximally expressive yet spectrally stable: IRR rises smoothly, ESRI remains near zero, and RDP exhibits consistent low-frequency cycles. These findings provide quantitative evidence that reflective, thought-like internal processing can arise from a principled balance between feedback amplification and homeostatic regulation, linking modern fast-weight architectures to theories of cortical reentry and recursive cognition. 

**Abstract (ZH)**: Fast-Weights Homeostatic Reentry Layer (FH-RL)：一种结合快速权重关联记忆、稳态正则化和学习反馈的神经机制及其自我参照计算近似实现 

---
# Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models 

**Title (ZH)**: 跨模态不可学习性通过多模态大型语言模型中关键神经元路径编辑实现 

**Authors**: Kunhao Li, Wenhao Li, Di Wu, Lei Yang, Jun Bai, Ju Jia, Jason Xue  

**Link**: [PDF](https://arxiv.org/pdf/2511.06793)  

**Abstract**: Multimodal Large Language Models (MLLMs) extend foundation models to real-world applications by integrating inputs such as text and vision. However, their broad knowledge capacity raises growing concerns about privacy leakage, toxicity mitigation, and intellectual property violations. Machine Unlearning (MU) offers a practical solution by selectively forgetting targeted knowledge while preserving overall model utility. When applied to MLLMs, existing neuron-editing-based MU approaches face two fundamental challenges: (1) forgetting becomes inconsistent across modalities because existing point-wise attribution methods fail to capture the structured, layer-by-layer information flow that connects different modalities; and (2) general knowledge performance declines when sensitive neurons that also support important reasoning paths are pruned, as this disrupts the model's ability to generalize. To alleviate these limitations, we propose a multimodal influential neuron path editor (MIP-Editor) for MU. Our approach introduces modality-specific attribution scores to identify influential neuron paths responsible for encoding forget-set knowledge and applies influential-path-aware neuron-editing via representation misdirection. This strategy also enables effective and coordinated forgetting across modalities while preserving the model's general capabilities. Experimental results demonstrate that MIP-Editor achieves a superior unlearning performance on multimodal tasks, with a maximum forgetting rate of 87.75% and up to 54.26% improvement in general knowledge retention. On textual tasks, MIP-Editor achieves up to 80.65% forgetting and preserves 77.9% of general performance. Codes are available at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型的机器遗忘：多模态有影响神经路径编辑（MIP-Editor） 

---
# Robust Causal Discovery under Imperfect Structural Constraints 

**Title (ZH)**: 稳健的因果发现方法在不完美的结构约束下 

**Authors**: Zidong Wang, Xi Lin, Chuchao He, Xiaoguang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2511.06790)  

**Abstract**: Robust causal discovery from observational data under imperfect prior knowledge remains a significant and largely unresolved challenge. Existing methods typically presuppose perfect priors or can only handle specific, pre-identified error types. And their performance degrades substantially when confronted with flawed constraints of unknown location and type. This decline arises because most of them rely on inflexible and biased thresholding strategies that may conflict with the data distribution. To overcome these limitations, we propose to harmonizes knowledge and data through prior alignment and conflict resolution. First, we assess the credibility of imperfect structural constraints through a surrogate model, which then guides a sparse penalization term measuring the loss between the learned and constrained adjacency matrices. We theoretically prove that, under ideal assumption, the knowledge-driven objective aligns with the data-driven objective. Furthermore, to resolve conflicts when this assumption is violated, we introduce a multi-task learning framework optimized via multi-gradient descent, jointly minimizing both objectives. Our proposed method is robust to both linear and nonlinear settings. Extensive experiments, conducted under diverse noise conditions and structural equation model types, demonstrate the effectiveness and efficiency of our method under imperfect structural constraints. 

**Abstract (ZH)**: 在不完善先验知识下的稳健因果发现依然是一项重要且未充分解决的挑战。现有的方法通常假设完美的先验知识，或者只能处理特定的先验错误类型。当面对未知位置和类型的不完善约束时，他们的性能会显著退化。这种退化是因为它们大多依赖于刚性和有偏的阈值策略，这些策略可能与数据分布相冲突。为克服这些限制，我们提出了一种通过先验对齐和冲突解决来协调知识和数据的方法。首先，我们通过代理模型评估不完善结构约束的可信度，这随后引导一个稀疏惩罚项来衡量学习到的和约束的邻接矩阵之间的损失。我们理论上证明，在理想假设下，知识驱动的目标与数据驱动的目标是相一致的。此外，为了在假设被违反时解决冲突，我们引入了一个通过多梯度下降优化的多任务学习框架，联合最小化两个目标。我们提出的方法在线性和非线性设置下都是稳健的。在不同噪声条件和结构方程模型类型下的广泛实验表明，在不完善的结构约束下，我们的方法既有效又高效。 

---
# Resource Efficient Sleep Staging via Multi-Level Masking and Prompt Learning 

**Title (ZH)**: 基于多级掩蔽和提示学习的资源高效睡眠分期方法 

**Authors**: Lejun Ai, Yulong Li, Haodong Yi, Jixuan Xie, Yue Wang, Jia Liu, Min Chen, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06785)  

**Abstract**: Automatic sleep staging plays a vital role in assessing sleep quality and diagnosing sleep disorders. Most existing methods rely heavily on long and continuous EEG recordings, which poses significant challenges for data acquisition in resource-constrained systems, such as wearable or home-based monitoring systems. In this paper, we propose the task of resource-efficient sleep staging, which aims to reduce the amount of signal collected per sleep epoch while maintaining reliable classification performance. To solve this task, we adopt the masking and prompt learning strategy and propose a novel framework called Mask-Aware Sleep Staging (MASS). Specifically, we design a multi-level masking strategy to promote effective feature modeling under partial and irregular observations. To mitigate the loss of contextual information introduced by masking, we further propose a hierarchical prompt learning mechanism that aggregates unmasked data into a global prompt, serving as a semantic anchor for guiding both patch-level and epoch-level feature modeling. MASS is evaluated on four datasets, demonstrating state-of-the-art performance, especially when the amount of data is very limited. This result highlights its potential for efficient and scalable deployment in real-world low-resource sleep monitoring environments. 

**Abstract (ZH)**: 资源高效睡眠分期在评估睡眠质量和诊断睡眠障碍中发挥着关键作用。现有方法大多依赖于长时间连续的EEG记录，这对资源受限系统（如可穿戴或家庭监控系统）的数据采集构成了重大挑战。本文提出了资源高效睡眠分期的任务，旨在在保持可靠分类性能的同时减少每个睡眠周期采集的信号量。为了解决这一任务，我们采用了遮罩和提示学习策略，并提出了一种名为Mask-Aware Sleep Staging (MASS)的新框架。具体来说，我们设计了一种多级遮罩策略，以促进在不完整和不规律观测下的有效特征建模。为减轻遮罩引入的上下文信息丢失，我们进一步提出了分层提示学习机制，将未遮罩的数据聚合到全局提示中，作为语义锚点，用于引导斑块级和周期级特征建模。MASS在四个数据集上进行了评估，显示出最先进的性能，尤其是在数据量非常有限的情况下。这一结果突显了其在实际低资源睡眠监控环境中高效且可扩展部署的潜力。 

---
# On the Mechanisms of Collaborative Learning in VAE Recommenders 

**Title (ZH)**: VAE推荐系统中协作学习机制的研究 

**Authors**: Tung-Long Vuong, Julien Monteil, Hien Dang, Volodymyr Vaskovych, Trung Le, Vu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2511.06781)  

**Abstract**: Variational Autoencoders (VAEs) are a powerful alternative to matrix factorization for recommendation. A common technique in VAE-based collaborative filtering (CF) consists in applying binary input masking to user interaction vectors, which improves performance but remains underexplored theoretically. In this work, we analyze how collaboration arises in VAE-based CF and show it is governed by latent proximity: we derive a latent sharing radius that informs when an SGD update on one user strictly reduces the loss on another user, with influence decaying as the latent Wasserstein distance increases. We further study the induced geometry: with clean inputs, VAE-based CF primarily exploits \emph{local} collaboration between input-similar users and under-utilizes global collaboration between far-but-related users. We compare two mechanisms that encourage \emph{global} mixing and characterize their trade-offs: (1) $\beta$-KL regularization directly tightens the information bottleneck, promoting posterior overlap but risking representational collapse if too large; (2) input masking induces stochastic geometric contractions and expansions, which can bring distant users onto the same latent neighborhood but also introduce neighborhood drift. To preserve user identity while enabling global consistency, we propose an anchor regularizer that aligns user posteriors with item embeddings, stabilizing users under masking and facilitating signal sharing across related items. Our analyses are validated on the Netflix, MovieLens-20M, and Million Song datasets. We also successfully deployed our proposed algorithm on an Amazon streaming platform following a successful online experiment. 

**Abstract (ZH)**: 基于变分自动编码器的推荐中合作机制的研究：从潜在空间 proximity 出发探讨局部与全局合作差异及其实现机制 

---
# OntoTune: Ontology-Driven Learning for Query Optimization with Convolutional Models 

**Title (ZH)**: 基于本体驱动的学习：用于卷积模型优化查询的本体导向学习 

**Authors**: Songhui Yue, Yang Shao, Sean Hayes  

**Link**: [PDF](https://arxiv.org/pdf/2511.06780)  

**Abstract**: Query optimization has been studied using machine learning, reinforcement learning, and, more recently, graph-based convolutional networks. Ontology, as a structured, information-rich knowledge representation, can provide context, particularly in learning problems. This paper presents OntoTune, an ontology-based platform for enhancing learning for query optimization. By connecting SQL queries, database metadata, and statistics, the ontology developed in this research is promising in capturing relationships and important determinants of query performance. This research also develops a method to embed ontologies while preserving as much of the relationships and key information as possible, before feeding it into learning algorithms such as tree-based and graph-based convolutional networks. A case study shows how OntoTune's ontology-driven learning delivers performance gains compared with database system default query execution. 

**Abstract (ZH)**: 基于本体的查询优化增强学习平台：OntoTune 

---
# Pedagogical Reflections on the Holistic Cognitive Development (HCD) Framework and AI-Augmented Learning in Creative Computing 

**Title (ZH)**: 整体认知发展框架与创意计算中的AI增强学习教学反思 

**Authors**: BHojan Anand  

**Link**: [PDF](https://arxiv.org/pdf/2511.06779)  

**Abstract**: This paper presents an expanded account of the Holistic Cognitive Development (HCD) framework for reflective and creative learning in computing education. The HCD framework integrates design thinking, experiential learning, and reflective practice into a unified constructivist pedagogy emphasizing autonomy, ownership, and scaffolding. It is applied across courses in game design (CS3247, CS4350), virtual reality (CS4240), and extended reality systems, where students engage in iterative cycles of thinking, creating, criticizing, and reflecting. The paper also examines how AI-augmented systems such as iReflect, ReflexAI, and Knowledge Graph-enhanced LLM feedback tools operationalize the HCD framework through scalable, personalized feedback. Empirical findings demonstrate improved reflective depth, feedback quality, and learner autonomy. The work advocates a balance of supportive autonomy in supervision, where students practice self-directed inquiry while guided through structured reflection and feedback. 

**Abstract (ZH)**: 本文提出了一个扩展示标的整体认知发展（HCD）框架，用于计算教育中的反思性与创造性学习。HCD框架将设计思维、体验式学习和反思实践整合为一种统一的建构主义教学法，强调自主性、归属感和支架式教学。该框架应用于游戏设计（CS3247, CS4350）、虚拟现实（CS4240）和扩展现实系统课程中，学生在创造、批判和反思的迭代循环中参与其中。本文还探讨了诸如iReflect、ReflexAI和知识图谱增强的大规模语言模型反馈工具等人工智能增强系统如何通过可扩展的个性化反馈实现HCD框架的实施。实证研究结果表明，这提高了反思深度、反馈质量和学习者的自主性。该工作倡导监督中的支持性自主平衡，学生在结构化的反思和反馈引导下进行自我导向的研究。 

---
# Data Trajectory Alignment for LLM Domain Adaptation: A Two-Phase Synthesis Framework for Telecommunications Mathematics 

**Title (ZH)**: 数据轨迹对齐以实现LLM领域适应：电信数学领域的两阶段综合框架 

**Authors**: Zhicheng Zhou, Jing Li, Suming Qiu, Junjie Huang, Linyuan Qiu, Zhijie Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.06776)  

**Abstract**: General-purpose large language models (LLMs) are increasingly deployed in verticals such as telecommunications, where adaptation is hindered by scarce, low-information-density corpora and tight mobile/edge constraints. We propose Data Trajectory Alignment (DTA), a two-phase, model-agnostic data curation framework that treats solution processes - not only final answers - as first-class supervision. Phase I (Initializing) synthesizes diverse, high-coverage candidates using an ensemble of strong teachers. Phase II (DTA) rewrites teacher solutions to align intermediate steps and presentation style with the target student's inductive biases and then performs signal-aware exemplar selection via agreement checks and reflection-based judging. Instantiated on telecommunications mathematics (e.g., link budgets, SNR/AMC selection, and power-control feasibility), DTA yields state-of-the-art (SOTA) accuracy on TELEMATH without enabling explicit "thinking" modes: 72.45% pass@1, surpassing distilled-only training by +17.65 points and outperforming a strong baseline (Qwen3-32B with thinking enabled) by +2.94 points. Token-shift analyses indicate that DTA concentrates gains on logical-structural discourse markers rather than merely amplifying domain nouns, indicating improved reasoning scaffolding. Under edge-like inference settings, DTA improves efficiency by reducing reliance on multi-sample voting and disabling expensive reasoning heuristics, cutting energy per output token by ~42% versus Qwen3-32B (thinking mode enabled) and end-to-end latency by ~60% versus Qwen3-32B (thinking mode disabled). These results demonstrate that aligning how solutions are produced enables compact, high-yield supervision that is effective for both accuracy and efficiency, offering a practical recipe for domain adaptation in low-resource verticals beyond telecom. 

**Abstract (ZH)**: 通用大语言模型在电信等垂直领域的应用受限于稀缺和信息密度低的数据集以及移动/边缘计算的限制。我们提出了一种名为数据轨迹对齐（DTA）的两阶段、模型无关的数据整理框架，该框架将解决方案过程——不仅仅是最终答案——视为一级监督。第一阶段（初始化）使用一组强大教师生成多样性和高覆盖的候选方案。第二阶段（DTA）重写教师解决方案，使其中间步骤和呈现风格与目标学生的归纳偏见保持一致，然后通过一致性和反射判断进行信号感知的范例选择。在电信数学（例如，链路预算、信噪比/ AMC选择和功率控制可行性）上实例化，DTA在TELEMATH上取得了最先进的准确性，而无需启用显式的“思考”模式：在1 pass上达到了72.45%的准确性，超越了仅蒸馏训练17.65个百分点，并且比启用了思考模式的强劲基线（Qwen3-32B）高出2.94个百分点。字位移分析表明，DTA将收益集中在逻辑结构 discourse 标记上，而不是仅仅放大领域名词，表明改进了推理支撑结构。在边缘计算场景下，DTA通过减少多样本投票依赖并禁用昂贵的推理启发式方法，提高了效率，相较于启用了思考模式的Qwen3-32B，每输出字的能量减少了约42%，端到端延迟减少了约60%。这些结果表明，对如何生成解决方案进行对齐可以提供紧凑高效的监督，对准确性和效率都有好处，并提供了一种实用的方法，在电信等资源有限的垂直领域进行领域适应。 

---
# QUARK: Quantization-Enabled Circuit Sharing for Transformer Acceleration by Exploiting Common Patterns in Nonlinear Operations 

**Title (ZH)**: QUARK: 通过利用非线性运算中的共同模式实现变换加速的量化启用电路共享方法 

**Authors**: Zhixiong Zhao, Haomin Li, Fangxin Liu, Yuncheng Lu, Zongwu Wang, Tao Yang, Li Jiang, Haibing Guan  

**Link**: [PDF](https://arxiv.org/pdf/2511.06767)  

**Abstract**: Transformer-based models have revolutionized computer vision (CV) and natural language processing (NLP) by achieving state-of-the-art performance across a range of benchmarks. However, nonlinear operations in models significantly contribute to inference latency, presenting unique challenges for efficient hardware acceleration. To this end, we propose QUARK, a quantization-enabled FPGA acceleration framework that leverages common patterns in nonlinear operations to enable efficient circuit sharing, thereby reducing hardware resource requirements. QUARK targets all nonlinear operations within Transformer-based models, achieving high-performance approximation through a novel circuit-sharing design tailored to accelerate these operations. Our evaluation demonstrates that QUARK significantly reduces the computational overhead of nonlinear operators in mainstream Transformer architectures, achieving up to a 1.96 times end-to-end speedup over GPU implementations. Moreover, QUARK lowers the hardware overhead of nonlinear modules by more than 50% compared to prior approaches, all while maintaining high model accuracy -- and even substantially boosting accuracy under ultra-low-bit quantization. 

**Abstract (ZH)**: 基于Transformer的模型通过在多种基准测试中实现最先进的性能，已经革命性地改变了计算机视觉（CV）和自然语言处理（NLP）。然而，模型中的非线性操作显著增加了推理延迟，为高效的硬件加速带来了独特挑战。为此，我们提出QUARK，一种量化使能的FPGA加速框架，利用非线性操作中的常见模式，实现高效的电路共享，从而减少硬件资源需求。QUARK针对Transformer-based模型中的所有非线性操作，通过一种专为加速这些操作设计的新型电路共享架构，实现高性能近似。我们的评估表明，QUARK显著减少了主流Transformer架构中非线性操作的计算开销，与GPU实现相比，端到端速度提升了1.96倍。此外，与先前方法相比，QUARK将非线性模块的硬件开销降低了超过50%，同时保持了高模型精度，并且在极低位量化下还能显著提升精度。 

---
# Sensitivity of Small Language Models to Fine-tuning Data Contamination 

**Title (ZH)**: Small语言模型对细调数据污染的敏感性 

**Authors**: Nicy Scaria, Silvester John Joseph Kennedy, Deepak Subramani  

**Link**: [PDF](https://arxiv.org/pdf/2511.06763)  

**Abstract**: Small Language Models (SLMs) are increasingly being deployed in resource-constrained environments, yet their behavioral robustness to data contamination during instruction tuning remains poorly understood. We systematically investigate the contamination sensitivity of 23 SLMs (270M to 4B parameters) across multiple model families by measuring susceptibility to syntactic and semantic transformation types during instruction tuning: syntactic transformations (character and word reversal) and semantic transformations (irrelevant and counterfactual responses), each applied at contamination levels of 25\%, 50\%, 75\%, and 100\%. Our results reveal fundamental asymmetries in vulnerability patterns: syntactic transformations cause catastrophic performance degradation, with character reversal producing near-complete failure across all models regardless of size or family, while semantic transformations demonstrate distinct threshold behaviors and greater resilience in core linguistic capabilities. Critically, we discover a ``\textit{capability curse}" where larger, more capable models become more susceptible to learning semantic corruptions, effectively following harmful instructions more readily, while our analysis of base versus instruction-tuned variants reveals that alignment provides inconsistent robustness benefits, sometimes even reducing resilience. Our work establishes three core contributions: (1) empirical evidence of SLMs' disproportionate vulnerability to syntactic pattern contamination, (2) identification of asymmetric sensitivity patterns between syntactic and semantic transformations, and (3) systematic evaluation protocols for contamination robustness assessment. These findings have immediate deployment implications, suggesting that current robustness assumptions may not hold for smaller models and highlighting the need for contamination-aware training protocols. 

**Abstract (ZH)**: Small Language Models在资源受限环境中部署的行为鲁棒性对数据污染的敏感性研究：基于指令调优的系统性分析与评估 

---
# Implicit Federated In-context Learning For Task-Specific LLM Fine-Tuning 

**Title (ZH)**: 隐式联邦上下文学习及其在任务特定LLM微调中的应用 

**Authors**: Dongcheng Li, Junhan Chen, Aoxiang Zhou, Chunpei Li, Youquan Xian, Peng Liu, Xianxian Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06757)  

**Abstract**: As large language models continue to develop and expand, the extensive public data they rely on faces the risk of depletion. Consequently, leveraging private data within organizations to enhance the performance of large models has emerged as a key challenge. The federated learning paradigm, combined with model fine-tuning techniques, effectively reduces the number of trainable parameters. However,the necessity to process high-dimensional feature spaces results in substantial overall computational overhead. To address this issue, we propose the Implicit Federated In-Context Learning (IFed-ICL) framework. IFed-ICL draws inspiration from federated learning to establish a novel distributed collaborative paradigm, by converting client local context examples into implicit vector representations, it enables distributed collaborative computation during the inference phase and injects model residual streams to enhance model performance. Experiments demonstrate that our proposed method achieves outstanding performance across multiple text classification tasks. Compared to traditional methods, IFed-ICL avoids the extensive parameter updates required by conventional fine-tuning methods while reducing data transmission and local computation at the client level in federated learning. This enables efficient distributed context learning using local private-domain data, significantly improving model performance on specific tasks. 

**Abstract (ZH)**: 基于隐式联邦上下文学习的分布式模型增强框架（IFed-ICL） 

---
# Hierarchical Spatial-Frequency Aggregation for Spectral Deconvolution Imaging 

**Title (ZH)**: 层次空间-频率聚合用于光谱去卷积成像 

**Authors**: Tao Lv, Daoming Zhou, Chenglong Huang, Chongde Zi, Linsen Chen, Xun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2511.06751)  

**Abstract**: Computational spectral imaging (CSI) achieves real-time hyperspectral imaging through co-designed optics and algorithms, but typical CSI methods suffer from a bulky footprint and limited fidelity. Therefore, Spectral Deconvolution imaging (SDI) methods based on PSF engineering have been proposed to achieve high-fidelity compact CSI design recently. However, the composite convolution-integration operations of SDI render the normal-equation coefficient matrix scene-dependent, which hampers the efficient exploitation of imaging priors and poses challenges for accurate reconstruction. To tackle the inherent data-dependent operators in SDI, we introduce a Hierarchical Spatial-Spectral Aggregation Unfolding Framework (HSFAUF). By decomposing subproblems and projecting them into the frequency domain, HSFAUF transforms nonlinear processes into linear mappings, thereby enabling efficient solutions. Furthermore, to integrate spatial-spectral priors during iterative refinement, we propose a Spatial-Frequency Aggregation Transformer (SFAT), which explicitly aggregates information across spatial and frequency domains. By integrating SFAT into HSFAUF, we develop a Transformer-based deep unfolding method, \textbf{H}ierarchical \textbf{S}patial-\textbf{F}requency \textbf{A}ggregation \textbf{U}nfolding \textbf{T}ransformer (HSFAUT), to solve the inverse problem of SDI. Systematic simulated and real experiments show that HSFAUT surpasses SOTA methods with cheaper memory and computational costs, while exhibiting optimal performance on different SDI systems. 

**Abstract (ZH)**: 基于空间- spectral聚合 unfoldings框架的Hierarchical Spatial-Frequency Aggregation Unfolding Transformer (HSFAUT) 

---
# Physically-Grounded Goal Imagination: Physics-Informed Variational Autoencoder for Self-Supervised Reinforcement Learning 

**Title (ZH)**: 基于物理的目標想象：物理指导的变分自编码器在无监督强化学习中的应用 

**Authors**: Lan Thi Ha Nguyen, Kien Ton Manh, Anh Do Duc, Nam Pham Hai  

**Link**: [PDF](https://arxiv.org/pdf/2511.06745)  

**Abstract**: Self-supervised goal-conditioned reinforcement learning enables robots to autonomously acquire diverse skills without human supervision. However, a central challenge is the goal setting problem: robots must propose feasible and diverse goals that are achievable in their current environment. Existing methods like RIG (Visual Reinforcement Learning with Imagined Goals) use variational autoencoder (VAE) to generate goals in a learned latent space but have the limitation of producing physically implausible goals that hinder learning efficiency. We propose Physics-Informed RIG (PI-RIG), which integrates physical constraints directly into the VAE training process through a novel Enhanced Physics-Informed Variational Autoencoder (Enhanced p3-VAE), enabling the generation of physically consistent and achievable goals. Our key innovation is the explicit separation of the latent space into physics variables governing object dynamics and environmental factors capturing visual appearance, while enforcing physical consistency through differential equation constraints and conservation laws. This enables the generation of physically consistent and achievable goals that respect fundamental physical principles such as object permanence, collision constraints, and dynamic feasibility. Through extensive experiments, we demonstrate that this physics-informed goal generation significantly improves the quality of proposed goals, leading to more effective exploration and better skill acquisition in visual robotic manipulation tasks including reaching, pushing, and pick-and-place scenarios. 

**Abstract (ZH)**: 物理约束导向的自我监督条件化强化学习使机器人能够在无需人类监督的情况下自主获取多样技能。然而，核心挑战是目标设定问题：机器人必须提出在当前环境中的可行且多样的目标。现有方法如RIG（基于想象目标的视觉强化学习）使用变分自编码器（VAE）生成在学习潜在空间中的目标，但会产生物理上不可行的目标从而阻碍学习效率。我们提出了物理信息导向的RIG（PI-RIG），通过一种新颖的增强物理信息变分自编码器（Enhanced p3-VAE）直接将物理约束整合到VAE训练过程中，使生成物理一致且可实现的目标成为可能。我们的关键创新在于明确分离潜在空间为控制物体动力学的物理变量和捕捉视觉外观的环境因素，并通过微分方程约束和守恒定律确保物理一致性。这使得生成的物理一致且可实现的目标能够遵守诸如物体持续性、碰撞约束和动态可行性等基本物理原则。通过广泛的实验，我们证明这种物理信息目标生成显著提高了提出目标的质量，从而在包括抓取、推动和拿起放置等视觉机器人操作任务中提高了探索的有效性和技能的学习。 

---
# Rank-1 LoRAs Encode Interpretable Reasoning Signals 

**Title (ZH)**: Rank-1 LoRAs Encode Interpretable Reasoning Signals 

**Authors**: Jake Ward, Paul Riechers, Adam Shai  

**Link**: [PDF](https://arxiv.org/pdf/2511.06739)  

**Abstract**: Reasoning models leverage inference-time compute to significantly enhance the performance of language models on difficult logical tasks, and have become a dominating paradigm in frontier LLMs. Despite their wide adoption, the mechanisms underpinning the enhanced performance of these reasoning models are not well understood. In this work, we show that the majority of new capabilities in reasoning models can be elicited by small, single-rank changes to base model parameters, with many of these changes being interpretable. Specifically, we use a rank-1 LoRA to create a minimal parameter adapter for Qwen-2.5-32B-Instruct which recovers 73-90% of reasoning-benchmark performance compared to a full parameter finetune. We find that the activations of this LoRA are as interpretable as MLP neurons, and fire for reasoning-specific behaviors. Finally, we train a sparse autoencoder on the entire activation state of this LoRA and identify fine-grained and monosemantic features. Our findings highlight that reasoning performance can arise largely from minimal changes to base model parameters, and explore what these changes affect. More broadly, our work shows that parameter-efficient training methods can be used as a targeted lens for uncovering fundamental insights about language model behavior and dynamics. 

**Abstract (ZH)**: 推理模型通过在推理时使用计算资源显著提升了语言模型在困难逻辑任务上的性能，并已成为前沿大规模语言模型的主导范式。尽管这些推理模型已被广泛采用，但其性能提升的机制尚不完全理解。在本文中，我们证明了大多数推理模型的新能力可以通过对基础模型参数进行小的、单极性的调整来激发，其中许多调整是可解释的。具体而言，我们使用秩1 LoRA为Qwen-2.5-32B-Instruct创建了一个最小参数适配器，该适配器在与全面参数微调相比的情况下，恢复了73-90%的推理基准性能。我们发现，该LoRA的激活具有与MLP神经元相当的可解释性，并针对推理特定的行为进行激活。最后，我们对整个LoRA的激活状态进行了稀疏自编码器训练，并识别出细粒度和单义特征。我们的研究结果强调，推理性能主要由基础模型参数的小调整引起，并探索这些调整的影响。更广泛地说，我们的工作表明，参数高效训练方法可以作为有针对性的透镜，用于揭示语言模型行为和动力学的基本洞察。 

---
# Diagnosing and Breaking Amplitude Suppression in Seismic Phase Picking Through Adversarial Shape Learning 

**Title (ZH)**: 通过对抗形状学习诊断与打破地震相拾波中的振幅抑制 

**Authors**: Chun-Ming Huang, Li-Heng Chang, I-Hsin Chang, An-Sheng Lee, Hao Kuo-Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.06731)  

**Abstract**: Deep learning has revolutionized seismic phase picking, yet a paradox persists: high signal-to-noise S-wave predictions consistently fail to cross detection thresholds, oscillating at suppressed amplitudes. We identify this previously unexplained phenomenon as amplitude suppression, which we diagnose through analyzing training histories and loss landscapes. Three interacting factors emerge: S-wave onsets exhibit high temporal uncertainty relative to high-amplitude boundaries; CNN's bias toward sharp amplitude changes anchors predictions to these boundaries rather than subtle onsets; and point-wise Binary Cross-Entropy (BCE) loss lacks lateral corrective forces, providing only vertical gradients that suppress amplitude while temporal gaps persist. This geometric trap points to a shape-then-align solution where stable geometric templates must precede temporal alignment. We implement this through a conditional GAN framework by augmenting conventional BCE training with a discriminator that enforces shape constraints. Training for 10,000 steps, this achieves a 64% increase in effective S-phase detections. Our framework autonomously discovers target geometry without a priori assumptions, offering a generalizable solution for segmentation tasks requiring precise alignment of subtle features near dominant structures. 

**Abstract (ZH)**: 深度学习已 revolutionized 地震相 Arrival 识别，却存在一个悖论：高信噪比 S 波预测常未能越过检测阈值，以抑制振幅 oscillating。我们通过分析训练历史和损失景观识别出这一先前未解释的现象为振幅抑制。三个相互作用的因素浮现：S 波起始表现出与高振幅边界相比的高时域不确定性；CNN 对尖锐振幅变化的偏好将预测锚定在这些边界而不是微小起始处；以及点对点二元交叉熵 (BCE) 损失缺乏横向校正力，仅提供垂直梯度以抑制振幅同时时域间断持续存在。这种几何陷阱指向一种 shape-then-align 解决方案，其中稳定几何模板必须先于时域对齐。我们通过在常规 BCE 训练中加入一个约束形状的判别器实现此目的，框架在 10,000 步训练后实现了有效 S 相检测 64% 的提升。我们的框架无需先验假设即可自主发现目标几何形态，提供了一种适用于需要精确对齐近主导结构的细微特征的分割任务的通用解决方案。 

---
# S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning 

**Title (ZH)**: 基于主题的有向无环图：多智能体异质推理模型 

**Authors**: Jiangwen Dong, Zehui Lin, Wanyu Lin, Mingjin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06727)  

**Abstract**: Large Language Models (LLMs) have achieved impressive performance in complex reasoning problems. Their effectiveness highly depends on the specific nature of the task, especially the required domain knowledge. Existing approaches, such as mixture-of-experts, typically operate at the task level; they are too coarse to effectively solve the heterogeneous problems involving multiple subjects. This work proposes a novel framework that performs fine-grained analysis at subject level equipped with a designated multi-agent collaboration strategy for addressing heterogeneous problem reasoning. Specifically, given an input query, we first employ a Graph Neural Network to identify the relevant subjects and infer their interdependencies to generate an \textit{Subject-based Directed Acyclic Graph} (S-DAG), where nodes represent subjects and edges encode information flow. Then we profile the LLM models by assigning each model a subject-specific expertise score, and select the top-performing one for matching corresponding subject of the S-DAG. Such subject-model matching enables graph-structured multi-agent collaboration where information flows from the starting model to the ending model over S-DAG. We curate and release multi-subject subsets of standard benchmarks (MMLU-Pro, GPQA, MedMCQA) to better reflect complex, real-world reasoning tasks. Extensive experiments show that our approach significantly outperforms existing task-level model selection and multi-agent collaboration baselines in accuracy and efficiency. These results highlight the effectiveness of subject-aware reasoning and structured collaboration in addressing complex and multi-subject problems. 

**Abstract (ZH)**: 大型语言模型在复杂推理问题上取得了显著性能。本工作提出了一种新型框架，该框架在主题级别进行细粒度分析，并配备专门的多agent协作策略以解决涉及多个主题的异质问题推理。具体而言，给定一个输入查询，我们首先使用图神经网络来识别相关主题及其相互依赖关系，生成主题导向有向无环图（S-DAG），节点代表主题，边编码信息流。然后，通过为每个模型分配一个主题特定的专业评分，挑选最适合对应S-DAG主题的模型。这种主题-模型匹配使具有图结构的多agent协作成为可能，信息通过S-DAG从起始模型流向结束模型。我们编制并发布了标准基准（MMLU-Pro、GPQA、MedMCQA）的多主题子集，更好地反映了复杂的实际世界推理任务。广泛的经验表明，本方法在准确性和效率上显著优于现有任务级别模型选择和多agent协作基线。这些结果突显了主题意识推理和结构化协作在解决复杂和多主题问题上的有效性。 

---
# Revisiting the Data Sampling in Multimodal Post-training from a Difficulty-Distinguish View 

**Title (ZH)**: 从难度区分视角重访多模态 Fine-tuning 数据采样 

**Authors**: Jianyu Qi, Ding Zou, Wenrui Yan, Rui Ma, Jiaxu Li, Zhijie Zheng, Zhiguo Yang, Rongchang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.06722)  

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) have spurred significant progress in Chain-of-Thought (CoT) reasoning. Building on the success of Deepseek-R1, researchers extended multimodal reasoning to post-training paradigms based on reinforcement learning (RL), focusing predominantly on mathematical datasets. However, existing post-training paradigms tend to neglect two critical aspects: (1) The lack of quantifiable difficulty metrics capable of strategically screening samples for post-training optimization. (2) Suboptimal post-training paradigms that fail to jointly optimize perception and reasoning capabilities. To address this gap, we propose two novel difficulty-aware sampling strategies: Progressive Image Semantic Masking (PISM) quantifies sample hardness through systematic image degradation, while Cross-Modality Attention Balance (CMAB) assesses cross-modal interaction complexity via attention distribution analysis. Leveraging these metrics, we design a hierarchical training framework that incorporates both GRPO-only and SFT+GRPO hybrid training paradigms, and evaluate them across six benchmark datasets. Experiments demonstrate consistent superiority of GRPO applied to difficulty-stratified samples compared to conventional SFT+GRPO pipelines, indicating that strategic data sampling can obviate the need for supervised fine-tuning while improving model accuracy. Our code will be released at this https URL. 

**Abstract (ZH)**: Recent Advances in 多模态大型语言模型（MLLMs）推动了链式思考（CoT）推理的显著进展。在 Deepseek-R1 取得成功的基础上，研究人员基于强化学习（RL）扩展了后训练时期的多模态推理，重点关注了数学数据集。然而，现有的后训练范式忽视了两个关键方面：（1）缺乏可量化的难度指标，无法策略性地筛选用于后训练优化的样本；（2）不理想的后训练范式，未能同时优化感知和推理能力。为填补这一空白，我们提出了两种新的难度感知采样策略：逐步图像语义掩蔽（PISM）通过系统性的图像退化量化样本难度，而跨模态注意力平衡（CMAB）通过注意分布分析评估跨模态交互的复杂性。利用这些指标，我们设计了一个分级训练框架，结合了仅基于GRPO和SFT+GRPO混合训练范式，并在六个基准数据集中进行了评估。实验结果表明，针对难度分级样本应用GRPO的一致性优越性高于传统的SFT+GRPO流水线，表明策略性数据采样可以消除监督微调的需要并提高模型准确性。我们的代码将在以下链接发布：这个 https URL。 

---
# MirrorMamba: Towards Scalable and Robust Mirror Detection in Videos 

**Title (ZH)**: 镜像飞虫：面向视频中可扩展且 robust 的镜像检测 

**Authors**: Rui Song, Jiaying Lin, Rynson W.H. Lau  

**Link**: [PDF](https://arxiv.org/pdf/2511.06716)  

**Abstract**: Video mirror detection has received significant research attention, yet existing methods suffer from limited performance and robustness. These approaches often over-rely on single, unreliable dynamic features, and are typically built on CNNs with limited receptive fields or Transformers with quadratic computational complexity. To address these limitations, we propose a new effective and scalable video mirror detection method, called MirrorMamba. Our approach leverages multiple cues to adapt to diverse conditions, incorporating perceived depth, correspondence and optical. We also introduce an innovative Mamba-based Multidirection Correspondence Extractor, which benefits from the global receptive field and linear complexity of the emerging Mamba spatial state model to effectively capture correspondence properties. Additionally, we design a Mamba-based layer-wise boundary enforcement decoder to resolve the unclear boundary caused by the blurred depth map. Notably, this work marks the first successful application of the Mamba-based architecture in the field of mirror detection. Extensive experiments demonstrate that our method outperforms existing state-of-the-art approaches for video mirror detection on the benchmark datasets. Furthermore, on the most challenging and representative image-based mirror detection dataset, our approach achieves state-of-the-art performance, proving its robustness and generalizability. 

**Abstract (ZH)**: 基于Mamba的视频镜像检测方法 

---
# Sensor Calibration Model Balancing Accuracy, Real-time, and Efficiency 

**Title (ZH)**: 传感器标定模型平衡精度、实时性和效率 

**Authors**: Jinyong Yun, Hyungjin Kim, Seokho Ahn, Euijong Lee, Young-Duk Seo  

**Link**: [PDF](https://arxiv.org/pdf/2511.06715)  

**Abstract**: Most on-device sensor calibration studies benchmark models only against three macroscopic requirements (i.e., accuracy, real-time, and resource efficiency), thereby hiding deployment bottlenecks such as instantaneous error and worst-case latency. We therefore decompose this triad into eight microscopic requirements and introduce Scare (Sensor Calibration model balancing Accuracy, Real-time, and Efficiency), an ultra-compressed transformer that fulfills them all. SCARE comprises three core components: (1) Sequence Lens Projector (SLP) that logarithmically compresses time-series data while preserving boundary information across bins, (2) Efficient Bitwise Attention (EBA) module that replaces costly multiplications with bitwise operations via binary hash codes, and (3) Hash optimization strategy that ensures stable training without auxiliary loss terms. Together, these components minimize computational overhead while maintaining high accuracy and compatibility with microcontroller units (MCUs). Extensive experiments on large-scale air-quality datasets and real microcontroller deployments demonstrate that Scare outperforms existing linear, hybrid, and deep-learning baselines, making Scare, to the best of our knowledge, the first model to meet all eight microscopic requirements simultaneously. 

**Abstract (ZH)**: 大多数针对设备端传感器校准的研究仅基于三大宏观要求（即准确性、实时性和资源效率）来评估模型，从而隐藏了部署瓶颈，如瞬时误差和最坏情况下限。因此，我们将其分解为八大微观要求，并引入Scare（Sensor Calibration model balancing Accuracy, Real-time, and Efficiency），这是一种极度压缩的变压器，能够满足所有这些要求。Scare包含三个核心组件：（1）序列透镜投影器（SLP），在保留各区间边界信息的同时对时间序列数据进行对数压缩，（2）高效位操作注意力（EBA）模块，通过二进制哈希码将昂贵的乘法替换为位操作，以及（3）哈希优化策略，确保稳定的训练而不需要辅助损失项。这些组件共同减少了计算开销，同时保持高精度并与微控制器单元（MCUs）兼容。大规模空气质量数据集上的广泛实验和实际微控制器部署表明，Scare优于现有的线性、混合和深度学习基线模型，据我们所知，Scare是首个同时满足所有八大微观要求的模型。 

---
# Structural Enforcement of Statistical Rigor in AI-Driven Discovery: A Functional Architecture 

**Title (ZH)**: AI驱动发现中结构性贯彻统计严谨性的功能架构 

**Authors**: Karen Sargsyan  

**Link**: [PDF](https://arxiv.org/pdf/2511.06701)  

**Abstract**: Sequential statistical protocols require meticulous state management and robust error handling -- challenges naturally suited to functional programming. We present a functional architecture for structural enforcement of statistical rigor in automated research systems (AI-Scientists). These LLM-driven systems risk generating spurious discoveries through dynamic hypothesis testing. We introduce the Research monad, a Haskell eDSL that enforces sequential statistical protocols (e.g., Online FDR (false discovery rate) control) using a monad transformer stack. To address risks in hybrid architectures where LLMs generate imperative code, we employ Declarative Scaffolding -- generating rigid harnesses that structurally constrain execution and prevent methodological errors like data leakage. We validate this approach through large-scale simulation (N=2000 hypotheses) and an end-to-end case study, demonstrating essential defense-in-depth for automated science integrity. 

**Abstract (ZH)**: 顺序统计协议需要细致的状态管理和 robust 错误处理——这自然是函数式编程的天然挑战。我们提出了一种函数式架构，用于在自动化研究系统（AI-科学家）中结构性地保障统计严谨性。这些由大语言模型驱动的系统通过动态假设检验存在产生虚假发现的风险。我们引入了 Research monad，这是一种 Haskell 的 eDSL，使用монад变换器栈来强制执行顺序统计协议（例如，Online FDR（错误发现率）控制）。为了解决混合架构中大语言模型生成命令式代码所带来的风险，我们采用了声明性支架——生成刚性框架以结构性约束执行并防止诸如数据泄露等方法论错误。我们通过大规模仿真（N=2000 假设）和端到端案例研究验证了这种方法，展示了自动化科学研究中必不可少的纵深防御。 

---
# Place Matters: Comparing LLM Hallucination Rates for Place-Based Legal Queries 

**Title (ZH)**: 地点 Matters：基于地点的法律查询中大模型幻觉率的比较 

**Authors**: Damian Curran, Vanessa Sporne, Lea Frermann, Jeannie Paterson  

**Link**: [PDF](https://arxiv.org/pdf/2511.06700)  

**Abstract**: How do we make a meaningful comparison of a large language model's knowledge of the law in one place compared to another? Quantifying these differences is critical to understanding if the quality of the legal information obtained by users of LLM-based chatbots varies depending on their location. However, obtaining meaningful comparative metrics is challenging because legal institutions in different places are not themselves easily comparable. In this work we propose a methodology to obtain place-to-place metrics based on the comparative law concept of functionalism. We construct a dataset of factual scenarios drawn from Reddit posts by users seeking legal advice for family, housing, employment, crime and traffic issues. We use these to elicit a summary of a law from the LLM relevant to each scenario in Los Angeles, London and Sydney. These summaries, typically of a legislative provision, are manually evaluated for hallucinations. We show that the rate of hallucination of legal information by leading closed-source LLMs is significantly associated with place. This suggests that the quality of legal solutions provided by these models is not evenly distributed across geography. Additionally, we show a strong negative correlation between hallucination rate and the frequency of the majority response when the LLM is sampled multiple times, suggesting a measure of uncertainty of model predictions of legal facts. 

**Abstract (ZH)**: 如何在不同地区之间有意义地比较大型语言模型的法律知识？量化这些差异对于理解用户使用基于LLM的聊天机器人获取的法律信息质量是否取决于其地理位置至关重要。然而，由于不同地方的法律制度本身不易比较，因此获取有意义的比较指标具有挑战性。在本研究中，我们提出了一种基于比较法功能主义概念的方法论来获得地区间的指标。我们构建了一个数据集，其中包含从寻求家庭、住房、就业、犯罪和交通法律咨询的Reddit帖子中抽取的实际情况场景。我们使用这些数据来从洛杉矶、伦敦和悉尼的LLM中提取与每个场景相关的法律总结。这些总结通常是对立法条款的描述，并由人工评估是否存在幻觉。结果显示，顶级闭源LLM在法律信息上的幻觉率与地区显著相关。这表明这些模型提供的法律解决方案的质量在地理上并不是均匀分布的。此外，我们还展示了幻觉率与LLM多轮抽样的多数响应频率之间存在强烈负相关，这表明了模型对法律事实预测的不确定性程度。 

---
# Magnitude-Modulated Equivariant Adapter for Parameter-Efficient Fine-Tuning of Equivariant Graph Neural Networks 

**Title (ZH)**: 幅度调制协变适配器：用于协变图神经网络参数高效微调的比例调制方法 

**Authors**: Dian Jin, Yancheng Yuan, Xiaoming Tao  

**Link**: [PDF](https://arxiv.org/pdf/2511.06696)  

**Abstract**: Pretrained equivariant graph neural networks based on spherical harmonics offer efficient and accurate alternatives to computationally expensive ab-initio methods, yet adapting them to new tasks and chemical environments still requires fine-tuning. Conventional parameter-efficient fine-tuning (PEFT) techniques, such as Adapters and LoRA, typically break symmetry, making them incompatible with those equivariant architectures. ELoRA, recently proposed, is the first equivariant PEFT method. It achieves improved parameter efficiency and performance on many benchmarks. However, the relatively high degrees of freedom it retains within each tensor order can still perturb pretrained feature distributions and ultimately degrade performance. To address this, we present Magnitude-Modulated Equivariant Adapter (MMEA), a novel equivariant fine-tuning method which employs lightweight scalar gating to modulate feature magnitudes on a per-order and per-multiplicity basis. We demonstrate that MMEA preserves strict equivariance and, across multiple benchmarks, consistently improves energy and force predictions to state-of-the-art levels while training fewer parameters than competing approaches. These results suggest that, in many practical scenarios, modulating channel magnitudes is sufficient to adapt equivariant models to new chemical environments without breaking symmetry, pointing toward a new paradigm for equivariant PEFT design. 

**Abstract (ZH)**: 基于球谐变换的预训练不变图神经网络提供了高效且准确的计算替代方案，无需使用计算密集型的第一性原理方法，然而，将它们适应新任务和化学环境仍然需要微调。传统的参数高效微调（PEFT）技术，如适配器和LoRA，通常会破坏对称性，使其与这些不变架构不兼容。最近提出的ELoRA是首个不变PEFT方法，它在许多基准测试中实现了改进的参数效率和性能。然而，它在每个张量阶数中保留的相对较高的自由度仍然可能会扰动预训练特征分布，最终会降低性能。为了解决这个问题，我们提出了幅度调控不变适配器（MMEA），这是一种新颖的不变微调方法，采用轻量级标量门控来按阶数和多重性调控特征幅度。我们表明，MMEA 保持了严格的不变性，在多个基准测试中，MMEA 一致地提高了能量和力的预测性能，同时训练的参数数量少于竞争方法。这些结果表明，在许多实际场景中，调控通道幅度足以使不变模型适应新的化学环境，而无需破坏对称性，这为不变PEFT设计指出了一个新的范式。 

---
# ML-EcoLyzer: Quantifying the Environmental Cost of Machine Learning Inference Across Frameworks and Hardware 

**Title (ZH)**: ML-EcoLyzer: 跨框架和硬件评估机器学习推断的环境成本 

**Authors**: Jose Marie Antonio Minoza, Rex Gregor Laylo, Christian F Villarin, Sebastian C. Ibanez  

**Link**: [PDF](https://arxiv.org/pdf/2511.06694)  

**Abstract**: Machine learning inference occurs at a massive scale, yet its environmental impact remains poorly quantified, especially on low-resource hardware. We present ML-EcoLyzer, a cross-framework tool for measuring the carbon, energy, thermal, and water costs of inference across CPUs, consumer GPUs, and datacenter accelerators. The tool supports both classical and modern models, applying adaptive monitoring and hardware-aware evaluation.
We introduce the Environmental Sustainability Score (ESS), which quantifies the number of effective parameters served per gram of CO$_2$ emitted. Our evaluation covers over 1,900 inference configurations, spanning diverse model architectures, task modalities (text, vision, audio, tabular), hardware types, and precision levels. These rigorous and reliable measurements demonstrate that quantization enhances ESS, huge accelerators can be inefficient for lightweight applications, and even small models may incur significant costs when implemented suboptimally. ML-EcoLyzer sets a standard for sustainability-conscious model selection and offers an extensive empirical evaluation of environmental costs during inference. 

**Abstract (ZH)**: 机器学习推理在大规模进行，但其在低资源硬件上的环境影响仍然量化不足。我们提出ML-EcoLyzer，这是一种跨框架工具，用于测量推理在CPU、消费级GPU和数据中心加速器上的碳排放、能源、热管理和水资源成本。该工具支持经典和现代模型，采用自适应监测和硬件感知评估。
我们引入了环境可持续性评分（ESS），该评分量化了每克二氧化碳排放的有效参数数量。我们的评估覆盖了超过1,900种推理配置，涵盖了多种模型架构、任务模式（文本、视觉、音频、表格数据）、硬件类型和精度级别。这些严谨可靠的测量结果表明，量化可以提升ESS，巨大的加速器对于轻量级应用可能是低效的，即使是很小的模型在非优化实现时也可能产生显著的成本。ML-EcoLyzer为可持续性意识下的模型选择设定了标准，并提供了推理过程中环境成本的广泛实证评估。 

---
# Textual Self-attention Network: Test-Time Preference Optimization through Textual Gradient-based Attention 

**Title (ZH)**: 文本自注意力网络：基于文本梯度注意力的测试时偏好优化 

**Authors**: Shibing Mo, Haoyang Ruan, Kai Wu, Jing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06682)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable generalization capabilities, but aligning their outputs with human preferences typically requires expensive supervised fine-tuning. Recent test-time methods leverage textual feedback to overcome this, but they often critique and revise a single candidate response, lacking a principled mechanism to systematically analyze, weigh, and synthesize the strengths of multiple promising candidates. Such a mechanism is crucial because different responses may excel in distinct aspects (e.g., clarity, factual accuracy, or tone), and combining their best elements may produce a far superior outcome. This paper proposes the Textual Self-Attention Network (TSAN), a new paradigm for test-time preference optimization that requires no parameter updates. TSAN emulates self-attention entirely in natural language to overcome this gap: it analyzes multiple candidates by formatting them into textual keys and values, weighs their relevance using an LLM-based attention module, and synthesizes their strengths into a new, preference-aligned response under the guidance of the learned textual attention. This entire process operates in a textual gradient space, enabling iterative and interpretable optimization. Empirical evaluations demonstrate that with just three test-time iterations on a base SFT model, TSAN outperforms supervised models like Llama-3.1-70B-Instruct and surpasses the current state-of-the-art test-time alignment method by effectively leveraging multiple candidate solutions. 

**Abstract (ZH)**: 大规模语言模型（LLMs）展示了 remarkable 的泛化能力，但使其输出与人类偏好一致通常需要昂贵的监督微调。最近的测试时方法利用文本反馈来克服这一问题，但它们通常仅评判和修订单一候选回复，缺乏系统分析、权衡和综合多个有潜力候选回复优势的原理性机制。这种机制至关重要，因为不同回复可能在不同的方面表现出色（例如，清晰度、事实准确性或语气），结合它们的最佳元素可能会产生远远优于单一回复的结果。本文提出了文本自我注意力网络（TSAN），这是一种新的测试时偏好优化范式，不需要参数更新。TSAN 通过完全用自然语言模拟自我注意力来弥补这一差距：通过格式化多个候选回复为文本键值、利用基于语言模型的注意力模块评估其相关性，并在学习到的文本注意力引导下合成一个新的、与偏好一致的回复。整个过程在文本梯度空间中进行，实现迭代和可解释的优化。实证评估表明，仅通过在基础 SFT 模型上进行三次测试时迭代，TSAN 就能够优于如 Llama-3.1-70B-Instruct 等监督模型，并且能够有效地利用多个候选解决方案，超越当前最先进的测试时对齐方法。 

---
# Rapidly Learning Soft Robot Control via Implicit Time-Stepping 

**Title (ZH)**: 通过隐式时间步进快速学习软体机器人控制 

**Authors**: Andrew Choi, Dezhong Tong  

**Link**: [PDF](https://arxiv.org/pdf/2511.06667)  

**Abstract**: With the explosive growth of rigid-body simulators, policy learning in simulation has become the de facto standard for most rigid morphologies. In contrast, soft robotic simulation frameworks remain scarce and are seldom adopted by the soft robotics community. This gap stems partly from the lack of easy-to-use, general-purpose frameworks and partly from the high computational cost of accurately simulating continuum mechanics, which often renders policy learning infeasible. In this work, we demonstrate that rapid soft robot policy learning is indeed achievable via implicit time-stepping. Our simulator of choice, DisMech, is a general-purpose, fully implicit soft-body simulator capable of handling both soft dynamics and frictional contact. We further introduce delta natural curvature control, a method analogous to delta joint position control in rigid manipulators, providing an intuitive and effective means of enacting control for soft robot learning. To highlight the benefits of implicit time-stepping and delta curvature control, we conduct extensive comparisons across four diverse soft manipulator tasks against one of the most widely used soft-body frameworks, Elastica. With implicit time-stepping, parallel stepping of 500 environments achieves up to 6x faster speeds for non-contact cases and up to 40x faster for contact-rich scenarios. Finally, a comprehensive sim-to-sim gap evaluation--training policies in one simulator and evaluating them in another--demonstrates that implicit time-stepping provides a rare free lunch: dramatic speedups achieved without sacrificing accuracy. 

**Abstract (ZH)**: 基于隐式时间步进的快速软体机学习方法：从模拟到模拟的有效加速 

---
# Sim4Seg: Boosting Multimodal Multi-disease Medical Diagnosis Segmentation with Region-Aware Vision-Language Similarity Masks 

**Title (ZH)**: Sim4Seg: 基于区域意识视觉-语言相似性掩码提升多模态多疾病医疗诊断分割 

**Authors**: Lingran Song, Yucheng Zhou, Jianbing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2511.06665)  

**Abstract**: Despite significant progress in pixel-level medical image analysis, existing medical image segmentation models rarely explore medical segmentation and diagnosis tasks jointly. However, it is crucial for patients that models can provide explainable diagnoses along with medical segmentation results. In this paper, we introduce a medical vision-language task named Medical Diagnosis Segmentation (MDS), which aims to understand clinical queries for medical images and generate the corresponding segmentation masks as well as diagnostic results. To facilitate this task, we first present the Multimodal Multi-disease Medical Diagnosis Segmentation (M3DS) dataset, containing diverse multimodal multi-disease medical images paired with their corresponding segmentation masks and diagnosis chain-of-thought, created via an automated diagnosis chain-of-thought generation pipeline. Moreover, we propose Sim4Seg, a novel framework that improves the performance of diagnosis segmentation by taking advantage of the Region-Aware Vision-Language Similarity to Mask (RVLS2M) module. To improve overall performance, we investigate a test-time scaling strategy for MDS tasks. Experimental results demonstrate that our method outperforms the baselines in both segmentation and diagnosis. 

**Abstract (ZH)**: 尽管在像素级医疗图像分析方面取得了显著进展，现有的医疗图像分割模型 rarely 探索医疗分割和诊断任务的联合实现。然而，对于患者而言，模型能够提供可解释的诊断结果与医疗分割结果一同输出至关重要。在本文中，我们引入了一个名为 Medical Diagnosis Segmentation (MDS) 的医疗视觉语言任务，旨在理解临床查询并生成相应的分割掩码以及诊断结果。为了促进这一任务，我们首先提出了 Multimodal Multi-disease Medical Diagnosis Segmentation (M3DS) 数据集，包含多模态多疾病医疗图像及其对应的分割掩码和自动生成的诊断推理链。此外，我们提出了一种名为 Sim4Seg 的新型框架，通过利用 Region-Aware Vision-Language Similarity to Mask (RVLS2M) 模块来改进诊断分割性能。为了提高整体性能，我们还研究了 MDS 任务的测试时缩放策略。实验结果表明，我们的方法在分割和诊断方面均优于基线方法。 

---
# Active Learning for Animal Re-Identification with Ambiguity-Aware Sampling 

**Title (ZH)**: 带有不确定性意识采样的动物再识别主动学习 

**Authors**: Depanshu Sani, Mehar Khurana, Saket Anand  

**Link**: [PDF](https://arxiv.org/pdf/2511.06658)  

**Abstract**: Animal Re-ID has recently gained substantial attention in the AI research community due to its high impact on biodiversity monitoring and unique research challenges arising from environmental factors. The subtle distinguishing patterns, handling new species and the inherent open-set nature make the problem even harder. To address these complexities, foundation models trained on labeled, large-scale and multi-species animal Re-ID datasets have recently been introduced to enable zero-shot Re-ID. However, our benchmarking reveals significant gaps in their zero-shot Re-ID performance for both known and unknown species. While this highlights the need for collecting labeled data in new domains, exhaustive annotation for Re-ID is laborious and requires domain expertise. Our analyses show that existing unsupervised (USL) and AL Re-ID methods underperform for animal Re-ID. To address these limitations, we introduce a novel AL Re-ID framework that leverages complementary clustering methods to uncover and target structurally ambiguous regions in the embedding space for mining pairs of samples that are both informative and broadly representative. Oracle feedback on these pairs, in the form of must-link and cannot-link constraints, facilitates a simple annotation interface, which naturally integrates with existing USL methods through our proposed constrained clustering refinement algorithm. Through extensive experiments, we demonstrate that, by utilizing only 0.033% of all annotations, our approach consistently outperforms existing foundational, USL and AL baselines. Specifically, we report an average improvement of 10.49%, 11.19% and 3.99% (mAP) on 13 wildlife datasets over foundational, USL and AL methods, respectively, while attaining state-of-the-art performance on each dataset. Furthermore, we also show an improvement of 11.09%, 8.2% and 2.06% for unknown individuals in an open-world setting. 

**Abstract (ZH)**: 动物Re-ID在AI研究领域近期获得了广泛关注，这得益于其在生物多样性监测中的重要影响及其在环境因素作用下带来的独特研究挑战。细微的区分模式、处理新物种以及固有的开放集性质使问题更加复杂。为应对这些复杂性，基于大规模、多物种标注数据集训练的预训练模型已引入以实现零样本Re-ID。然而，我们的基准测试揭示了它们在已知和未知物种上的零样本Re-ID性能存在显著差距。这突显了在新领域收集标注数据的必要性，而详尽的Re-ID标注工作量大且需要领域专业知识。我们的分析表明，现有的无监督（USL）和半监督（AL）Re-ID方法在动物Re-ID中表现不佳。为解决这些限制，我们提出了一种新型AL Re-ID框架，该框架利用互补聚类方法揭露并针对嵌入空间中的结构性模糊区域，挖掘既具有信息性又具有广泛代表性的样本对。通过这些样本对的Oracle反馈，即必须链接和不能链接约束，使简单的标注接口得以实现，并自然地与现有USL方法整合，通过我们提出的受约束聚类精炼算法。通过大量实验，我们表明，在仅使用全部标注的0.033%情况下，我们的方法在性能上始终优于现有的基础、USL和AL基线。具体而言，我们的方法在13个野生动物数据集上的平均改进幅度分别为10.49%、11.19%和3.99%（mAP），在每个数据集上均达到最佳性能。此外，我们还展示了在开放世界设置中对未知个体的改进幅度，分别为11.09%、8.2%和2.06%。 

---
# CaberNet: Causal Representation Learning for Cross-Domain HVAC Energy Prediction 

**Title (ZH)**: CaberNet: 跨域 HVAC 能耗因果表示学习 

**Authors**: Kaiyuan Zhai, Jiacheng Cui, Zhehao Zhang, Junyu Xue, Yang Deng, Kui Wu, Guoming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06634)  

**Abstract**: Cross-domain HVAC energy prediction is essential for scalable building energy management, particularly because collecting extensive labeled data for every new building is both costly and impractical. Yet, this task remains highly challenging due to the scarcity and heterogeneity of data across different buildings, climate zones, and seasonal patterns. In particular, buildings situated in distinct climatic regions introduce variability that often leads existing methods to overfit to spurious correlations, rely heavily on expert intervention, or compromise on data diversity. To address these limitations, we propose CaberNet, a causal and interpretable deep sequence model that learns invariant (Markov blanket) representations for robust cross-domain prediction. In a purely data-driven fashion and without requiring any prior knowledge, CaberNet integrates i) a global feature gate trained with a self-supervised Bernoulli regularization to distinguish superior causal features from inferior ones, and ii) a domain-wise training scheme that balances domain contributions, minimizes cross-domain loss variance, and promotes latent factor independence. We evaluate CaberNet on real-world datasets collected from three buildings located in three climatically diverse cities, and it consistently outperforms all baselines, achieving a 22.9\% reduction in normalized mean squared error (NMSE) compared to the best benchmark. Our code is available at this https URL. 

**Abstract (ZH)**: 跨域 HVAC 能耗预测对于可扩展的建筑能效管理至关重要，特别是在收集每座新建筑的大量标注数据既昂贵又不切实际的情况下。但由于不同建筑、气候区域和季节模式之间数据的稀缺性和异质性，这一任务依然极具挑战性。尤其是，位于不同气候区域的建筑引入了变异性，这经常导致现有方法过度拟合到虚假的相关性、高度依赖专家干预或牺牲数据多样性。为解决这些限制，我们提出了 CaberNet，这是一种因果且可解释的深层序列模型，能够学习泛化的（马尔可夫边界）表示，实现稳健的跨域预测。CaberNet 在完全数据驱动的方式下进行训练，无需任何先验知识，它结合了 i) 一个使用自监督伯努利正则化训练的全局特征门控，用于区分优质因果特征与劣质特征，以及 ii) 一种跨域训练方案，平衡领域贡献，最小化跨域损失方差并促进潜在因子独立。我们在来自三个气候条件各异城市的三座建筑的现实数据集上评估了 CaberNet，它在所有基准模型中表现最佳，相比最优基准，其归一化均方误差（NMSE）降低了 22.9%。我们的代码可在以下链接获取：this https URL。 

---
# Explainable Cross-Disease Reasoning for Cardiovascular Risk Assessment from LDCT 

**Title (ZH)**: 可解释的跨疾病推理在基于LDCT的心血管风险评估中的应用 

**Authors**: Yifei Zhang, Jiashuo Zhang, Xiaofeng Yang, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.06625)  

**Abstract**: Low-dose chest computed tomography (LDCT) inherently captures both pulmonary and cardiac structures, offering a unique opportunity for joint assessment of lung and cardiovascular health. However, most existing approaches treat these domains as independent tasks, overlooking their physiological interplay and shared imaging biomarkers. We propose an Explainable Cross-Disease Reasoning Framework that enables interpretable cardiopulmonary risk assessment from a single LDCT scan. The framework introduces an agentic reasoning process that emulates clinical diagnostic thinking-first perceiving pulmonary findings, then reasoning through established medical knowledge, and finally deriving a cardiovascular judgment with explanatory rationale. It integrates three synergistic components: a pulmonary perception module that summarizes lung abnormalities, a knowledge-guided reasoning module that infers their cardiovascular implications, and a cardiac representation module that encodes structural biomarkers. Their outputs are fused to produce a holistic cardiovascular risk prediction that is both accurate and physiologically grounded. Experiments on the NLST cohort demonstrate that the proposed framework achieves state-of-the-art performance for CVD screening and mortality prediction, outperforming single-disease and purely image-based baselines. Beyond quantitative gains, the framework provides human-verifiable reasoning that aligns with cardiological understanding, revealing coherent links between pulmonary abnormalities and cardiac stress mechanisms. Overall, this work establishes a unified and explainable paradigm for cardiovascular analysis from LDCT, bridging the gap between image-based prediction and mechanism-based medical interpretation. 

**Abstract (ZH)**: 低剂量胸部计算机断层扫描可解释跨疾病推理框架：从单次扫描实现可解释的心肺风险评估 

---
# How Do VLAs Effectively Inherit from VLMs? 

**Title (ZH)**: VLAs如何有效地继承自VLMs？ 

**Authors**: Chuheng Zhang, Rushuai Yang, Xiaoyu Chen, Kaixin Wang, Li Zhao, Yi Chen, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2511.06619)  

**Abstract**: Vision-language-action (VLA) models hold the promise to attain generalizable embodied control. To achieve this, a pervasive paradigm is to leverage the rich vision-semantic priors of large vision-language models (VLMs). However, the fundamental question persists: How do VLAs effectively inherit the prior knowledge from VLMs? To address this critical question, we introduce a diagnostic benchmark, GrinningFace, an emoji tabletop manipulation task where the robot arm is asked to place objects onto printed emojis corresponding to language instructions. This task design is particularly revealing -- knowledge associated with emojis is ubiquitous in Internet-scale datasets used for VLM pre-training, yet emojis themselves are largely absent from standard robotics datasets. Consequently, they provide a clean proxy: successful task completion indicates effective transfer of VLM priors to embodied control. We implement this diagnostic task in both simulated environment and a real robot, and compare various promising techniques for knowledge transfer. Specifically, we investigate the effects of parameter-efficient fine-tuning, VLM freezing, co-training, predicting discretized actions, and predicting latent actions. Through systematic evaluation, our work not only demonstrates the critical importance of preserving VLM priors for the generalization of VLA but also establishes guidelines for future research in developing truly generalizable embodied AI systems. 

**Abstract (ZH)**: Vision-语言-动作（VLA）模型有实现泛化 embodied控制的潜力。为了实现这一目标，一种普遍的做法是利用大型视觉-语言模型（VLMs）的丰富视觉语义先验知识。然而，一个基本的问题仍然存在：VLA是如何有效地继承VLMs的先验知识的？为了解决这个问题，我们引入了一个诊断基准GrinningFace，这是一个表情符号桌面上的操纵任务，机器人臂被要求根据语言指令将物体放置在对应的打印表情符号上。这一任务设计尤为揭示性——与表情符号相关的知识在用于VLM预训练的互联网规模的数据集中普遍存在，但表情符号本身在标准的机器人数据集中却几乎不存在。因此，它们提供了一个干净的代理：成功完成任务表明VLM先验知识已成功转移到embodied控制。我们在模拟环境和真实机器人中实施了此诊断任务，并比较了几种有前景的知识转移技术，包括参数高效微调、VLM冻结、协同训练、预测离散动作和预测潜在动作。通过系统的评估，我们的工作不仅突显了保留VLM先验知识对于VLA泛化的关键重要性，还为开发真正泛化的embodied AI系统未来研究提供了指导。 

---
# Beyond Fixed Depth: Adaptive Graph Neural Networks for Node Classification Under Varying Homophily 

**Title (ZH)**: 超越固定深度：在同质性变化条件下自适应图神经网络的节点分类 

**Authors**: Asela Hevapathige, Asiri Wijesinghe, Ahad N. Zehmakan  

**Link**: [PDF](https://arxiv.org/pdf/2511.06608)  

**Abstract**: Graph Neural Networks (GNNs) have achieved significant success in addressing node classification tasks. However, the effectiveness of traditional GNNs degrades on heterophilic graphs, where connected nodes often belong to different labels or properties. While recent work has introduced mechanisms to improve GNN performance under heterophily, certain key limitations still exist. Most existing models apply a fixed aggregation depth across all nodes, overlooking the fact that nodes may require different propagation depths based on their local homophily levels and neighborhood structures. Moreover, many methods are tailored to either homophilic or heterophilic settings, lacking the flexibility to generalize across both regimes. To address these challenges, we develop a theoretical framework that links local structural and label characteristics to information propagation dynamics at the node level. Our analysis shows that optimal aggregation depth varies across nodes and is critical for preserving class-discriminative information. Guided by this insight, we propose a novel adaptive-depth GNN architecture that dynamically selects node-specific aggregation depths using theoretically grounded metrics. Our method seamlessly adapts to both homophilic and heterophilic patterns within a unified model. Extensive experiments demonstrate that our approach consistently enhances the performance of standard GNN backbones across diverse benchmarks. 

**Abstract (ZH)**: 图神经网络（GNNs）在节点分类任务中取得了显著成功。然而，传统的GNNs在异构图上效果较差，在异构图中，相连节点经常属于不同的标签或属性。虽然近期工作引入了机制以改善GNN在异构图上的性能，但仍存在一些关键限制。大多数现有模型为所有节点应用固定的聚合深度，忽视了节点可能需要根据其局部同构水平和邻域结构调整聚合深度的事实。此外，许多方法仅针对同构或异构设置，缺乏在两种模式之间泛化的灵活性。为解决这些挑战，我们开发了一个理论框架，将局部结构和标签特性与节点级别的信息传播动力学联系起来。我们的分析表明，最优聚合深度因节点而异，并对保留类别区分信息至关重要。根据这一见解，我们提出了一种新的自适应深度GNN架构，该架构使用理论依据的指标动态选择节点特定的聚合深度。我们的方法在统一模型中无缝适应同构和异构模式。广泛的实验表明，我们的方法在多种基准测试中一致提高了标准GNN主干模型的性能。 

---
# SPUR: A Plug-and-Play Framework for Integrating Spatial Audio Understanding and Reasoning into Large Audio-Language Models 

**Title (ZH)**: SPUR：一种Plug-and-Play框架，用于将空间音频理解与推理集成到大型音频语言模型中 

**Authors**: S Sakshi, Vaibhavi Lokegaonkar, Neil Zhang, Ramani Duraiswami, Sreyan Ghosh, Dinesh Manocha, Lie Lu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06606)  

**Abstract**: Spatial perception is central to auditory intelligence, enabling accurate understanding of real-world acoustic scenes and advancing human-level perception of the world around us. While recent large audio-language models (LALMs) show strong reasoning over complex audios, most operate on monaural inputs and lack the ability to capture spatial cues such as direction, elevation, and distance. We introduce SPUR, a lightweight, plug-in approach that equips LALMs with spatial perception through minimal architectural changes. SPUR consists of: (i) a First-Order Ambisonics (FOA) encoder that maps (W, X, Y, Z) channels to rotation-aware, listener-centric spatial features, integrated into target LALMs via a multimodal adapter; and (ii) SPUR-Set, a spatial QA dataset combining open-source FOA recordings with controlled simulations, emphasizing relative direction, elevation, distance, and overlap for supervised spatial reasoning. Fine-tuning our model on the SPUR-Set consistently improves spatial QA and multi-speaker attribution while preserving general audio understanding. SPUR provides a simple recipe that transforms monaural LALMs into spatially aware models. Extensive ablations validate the effectiveness of our approach. 

**Abstract (ZH)**: 空间感知是听觉智能的核心，使人们对真实世界的声景有准确的理解，并推动我们周围世界的人类级感知。尽管最近的大规模音频-语言模型（LALMs）在复杂音频上的推理能力很强，但大多数模型仅处理单声道输入，并缺乏捕捉方向、仰角和距离等空间线索的能力。我们引入了SPUR，这是一种轻量级且可插拔的方法，通过最小的架构更改为LALMs赋予空间感知。SPUR包括：（i）一阶球面声学（FOA）编码器，将（W, X, Y, Z）通道映射为旋转感知的、以听者为中心的空间特征，并通过多模态适配器整合到目标LALMs中；以及（ii）SPUR-Set，这是一个结合开源FOA录音与受控模拟的空间QA数据集，强调相对方向、仰角、距离和重叠，以进行监督的空间推理。在SPUR-Set上微调我们的模型在提升空间QA和多说话人属性方面取得了持续改善，同时保持一般音频理解。SPUR提供了一个简单的配方，将单声道LALMs转化为具有空间意识的模型。广泛的经验消除实验验证了我们方法的有效性。 

---
# TabRAG: Tabular Document Retrieval via Structured Language Representations 

**Title (ZH)**: TabRAG: 基于结构化语言表示的表格文档检索 

**Authors**: Jacob Si, Mike Qu, Michelle Lee, Yingzhen Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06582)  

**Abstract**: Ingesting data for Retrieval-Augmented Generation (RAG) involves either fine-tuning the embedding model directly on the target corpus or parsing documents for embedding model encoding. The former, while accurate, incurs high computational hardware requirements, while the latter suffers from suboptimal performance when extracting tabular data. In this work, we address the latter by presenting TabRAG, a parsing-based RAG pipeline designed to tackle table-heavy documents via structured language representations. TabRAG outperforms existing popular parsing-based methods for generation and retrieval. Code is available at this https URL. 

**Abstract (ZH)**: 基于解析的数据 ingestion 对 Retrieval-Augmented Generation (RAG) 的研究：TabRAG——一种针对表格密集型文档的结构化语言表示解析管道 

---
# CoFineLLM: Conformal Finetuning of LLMs for Language-Instructed Robot Planning 

**Title (ZH)**: CoFineLLM: 语言指导的机器人规划中LLMs的齐性微调 

**Authors**: Jun Wang, Yevgeniy Vorobeychik, Yiannis Kantaros  

**Link**: [PDF](https://arxiv.org/pdf/2511.06575)  

**Abstract**: Large Language Models (LLMs) have recently emerged as planners for language-instructed agents, generating sequences of actions to accomplish natural language tasks. However, their reliability remains a challenge, especially in long-horizon tasks, since they often produce overconfident yet wrong outputs. Conformal Prediction (CP) has been leveraged to address this issue by wrapping LLM outputs into prediction sets that contain the correct action with a user-defined confidence. When the prediction set is a singleton, the planner executes that action; otherwise, it requests help from a user. This has led to LLM-based planners that can ensure plan correctness with a user-defined probability. However, as LLMs are trained in an uncertainty-agnostic manner, without awareness of prediction sets, they tend to produce unnecessarily large sets, particularly at higher confidence levels, resulting in frequent human interventions limiting autonomous deployment. To address this, we introduce CoFineLLM (Conformal Finetuning for LLMs), the first CP-aware finetuning framework for LLM-based planners that explicitly reduces prediction-set size and, in turn, the need for user interventions. We evaluate our approach on multiple language-instructed robot planning problems and show consistent improvements over uncertainty-aware and uncertainty-agnostic finetuning baselines in terms of prediction-set size, and help rates. Finally, we demonstrate robustness of our method to out-of-distribution scenarios in hardware experiments. 

**Abstract (ZH)**: Conformal Finetuning for LLMs (CoFineLLM): Reducing Prediction-Set Size for Robust Autonomous Planning 

---
# SteganoSNN: SNN-Based Audio-in-Image Steganography with Encryption 

**Title (ZH)**: SteganoSNN：基于SNN的图像中音频隐写加密技术 

**Authors**: Biswajit Kumar Sahoo, Pedro Machado, Isibor Kennedy Ihianle, Andreas Oikonomou, Srinivas Boppu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06573)  

**Abstract**: Secure data hiding remains a fundamental challenge in digital communication, requiring a careful balance between computational efficiency and perceptual transparency. The balance between security and performance is increasingly fragile with the emergence of generative AI systems capable of autonomously generating and optimising sophisticated cryptanalysis and steganalysis algorithms, thereby accelerating the exposure of vulnerabilities in conventional data-hiding schemes.
This work introduces SteganoSNN, a neuromorphic steganographic framework that exploits spiking neural networks (SNNs) to achieve secure, low-power, and high-capacity multimedia data hiding. Digitised audio samples are converted into spike trains using leaky integrate-and-fire (LIF) neurons, encrypted via a modulo-based mapping scheme, and embedded into the least significant bits of RGBA image channels using a dithering mechanism to minimise perceptual distortion. Implemented in Python using NEST and realised on a PYNQ-Z2 FPGA, SteganoSNN attains real-time operation with an embedding capacity of 8 bits per pixel. Experimental evaluations on the DIV2K 2017 dataset demonstrate image fidelity between 40.4 dB and 41.35 dB in PSNR and SSIM values consistently above 0.97, surpassing SteganoGAN in computational efficiency and robustness. SteganoSNN establishes a foundation for neuromorphic steganography, enabling secure, energy-efficient communication for Edge-AI, IoT, and biomedical applications. 

**Abstract (ZH)**: 神经形态隐写术框架SteganoSNN：利用_spike神经网络实现安全、低功耗和高容量的多媒体数据隐藏 

---
# Rep2Text: Decoding Full Text from a Single LLM Token Representation 

**Title (ZH)**: Rep2Text: 从单个LLM令牌表示解码完整文本 

**Authors**: Haiyan Zhao, Zirui He, Fan Yang, Ali Payani, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2511.06571)  

**Abstract**: Large language models (LLMs) have achieved remarkable progress across diverse tasks, yet their internal mechanisms remain largely opaque. In this work, we address a fundamental question: to what extent can the original input text be recovered from a single last-token representation within an LLM? We propose Rep2Text, a novel framework for decoding full text from last-token representations. Rep2Text employs a trainable adapter that projects a target model's internal representations into the embedding space of a decoding language model, which then autoregressively reconstructs the input text. Experiments on various model combinations (Llama-3.1-8B, Gemma-7B, Mistral-7B-v0.1, Llama-3.2-3B) demonstrate that, on average, over half of the information in 16-token sequences can be recovered from this compressed representation while maintaining strong semantic integrity and coherence. Furthermore, our analysis reveals an information bottleneck effect: longer sequences exhibit decreased token-level recovery while preserving strong semantic integrity. Besides, our framework also demonstrates robust generalization to out-of-distribution medical data. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多种任务上取得了显著进步，但其内部机制依然 largely 不透明。在本文中，我们探讨了一个基本问题：在大型语言模型中，单个最后一个token的表示能否恢复原始输入文本到什么程度？我们提出了Rep2Text，一个从最后一个token表示解码完整文本的新型框架。Rep2Text 使用一个可训练的适配器，将目标模型的内部表示投影到解码语言模型的嵌入空间中，然后自回归地重构输入文本。在不同模型组合（Llama-3.1-8B、Gemma-7B、Mistral-7B-v0.1、Llama-3.2-3B）上的实验表明，平均来说，可以从这种压缩表示中恢复超过一半的16-token序列信息，同时保持较强的意义连贯性和一致性。此外，我们的分析揭示了一个信息瓶颈效应：较长序列在保持较强的意义连贯性的同时，其token级恢复程度降低。此外，我们的框架还展示了对分布外医疗数据的强大泛化能力。 

---
# Breaking the Dyadic Barrier: Rethinking Fairness in Link Prediction Beyond Demographic Parity 

**Title (ZH)**: 突破二元障碍：超越人口统计公平性在链接预测中重思公平性 

**Authors**: João Mattos, Debolina Halder Lina, Arlei Silva  

**Link**: [PDF](https://arxiv.org/pdf/2511.06568)  

**Abstract**: Link prediction is a fundamental task in graph machine learning with applications, ranging from social recommendation to knowledge graph completion. Fairness in this setting is critical, as biased predictions can exacerbate societal inequalities. Prior work adopts a dyadic definition of fairness, enforcing fairness through demographic parity between intra-group and inter-group link predictions. However, we show that this dyadic framing can obscure underlying disparities across subgroups, allowing systemic biases to go undetected. Moreover, we argue that demographic parity does not meet desired properties for fairness assessment in ranking-based tasks such as link prediction. We formalize the limitations of existing fairness evaluations and propose a framework that enables a more expressive assessment. Additionally, we propose a lightweight post-processing method combined with decoupled link predictors that effectively mitigates bias and achieves state-of-the-art fairness-utility trade-offs. 

**Abstract (ZH)**: 链接预测是图机器学习中的一个基本任务，应用于从社会推荐到知识图谱补全等多个领域。在这种设置中，公平性至关重要，因为有偏的预测可能加剧社会不平等。现有工作采用二元公平定义，通过组内和组间链接预测的统计 parity 来确保公平性。然而，我们展示了这种二元框架可能会掩盖子群体间的潜在差异，允许系统性偏差未被检测到。此外，我们认为统计 parity 无法满足排名任务如链接预测中公平性评估所需的特性。我们正式化了现有公平性评估的局限性，并提出了一种框架以实现更丰富的评估。另外，我们提出了一个轻量级的后处理方法，结合解耦的链接预测器，有效减轻偏见并实现最先进的公平性-效用权衡。 

---
# LLM For Loop Invariant Generation and Fixing: How Far Are We? 

**Title (ZH)**: LLM 在循环不变式生成与修复中的应用：我们走了多远？ 

**Authors**: Mostafijur Rahman Akhond, Saikat Chakraborty, Gias Uddin  

**Link**: [PDF](https://arxiv.org/pdf/2511.06552)  

**Abstract**: A loop invariant is a property of a loop that remains true before and after each execution of the loop. The identification of loop invariants is a critical step to support automated program safety assessment. Recent advancements in Large Language Models (LLMs) have demonstrated potential in diverse software engineering (SE) and formal verification tasks. However, we are not aware of the performance of LLMs to infer loop invariants. We report an empirical study of both open-source and closed-source LLMs of varying sizes to assess their proficiency in inferring inductive loop invariants for programs and in fixing incorrect invariants. Our findings reveal that while LLMs exhibit some utility in inferring and repairing loop invariants, their performance is substantially enhanced when supplemented with auxiliary information such as domain knowledge and illustrative examples. LLMs achieve a maximum success rate of 78\% in generating, but are limited to 16\% in repairing the invariant. 

**Abstract (ZH)**: 循环不变式是在每次执行循环之前和之后都保持真实的性质。识别循环不变式是支持自动化程序安全评估的关键步骤。大型语言模型（LLMs）在软件工程（SE）和形式验证任务中展现了多方面的潜力。然而，我们尚未见有关LLMs推断循环不变式的性能研究。我们报告了一项针对不同大小的开源和封闭源LLMs的实证研究，评估它们在为程序推断归纳循环不变式和修复错误不变式方面的 proficiency。我们的发现表明，尽管LLMs在推断和修复循环不变式方面表现出一定的效用，但通过补充领域知识和示例等辅助信息，其性能得到了显著提升。LLMs在生成循环不变式方面的最高成功率达到了78%，但在修复不变式方面的成功率仅限于16%。 

---
# Ibom NLP: A Step Toward Inclusive Natural Language Processing for Nigeria's Minority Languages 

**Title (ZH)**: Ibom NLP：迈向包容性的尼日利亚少数民族语言自然语言处理 

**Authors**: Oluwadara Kalejaiye, Luel Hagos Beyene, David Ifeoluwa Adelani, Mmekut-Mfon Gabriel Edet, Aniefon Daniel Akpan, Eno-Abasi Urua, Anietie Andy  

**Link**: [PDF](https://arxiv.org/pdf/2511.06531)  

**Abstract**: Nigeria is the most populous country in Africa with a population of more than 200 million people. More than 500 languages are spoken in Nigeria and it is one of the most linguistically diverse countries in the world. Despite this, natural language processing (NLP) research has mostly focused on the following four languages: Hausa, Igbo, Nigerian-Pidgin, and Yoruba (i.e <1% of the languages spoken in Nigeria). This is in part due to the unavailability of textual data in these languages to train and apply NLP algorithms. In this work, we introduce ibom -- a dataset for machine translation and topic classification in four Coastal Nigerian languages from the Akwa Ibom State region: Anaang, Efik, Ibibio, and Oro. These languages are not represented in Google Translate or in major benchmarks such as Flores-200 or SIB-200. We focus on extending Flores-200 benchmark to these languages, and further align the translated texts with topic labels based on SIB-200 classification dataset. Our evaluation shows that current LLMs perform poorly on machine translation for these languages in both zero-and-few shot settings. However, we find the few-shot samples to steadily improve topic classification with more shots. 

**Abstract (ZH)**: 尼日利亚是非洲人口最多的国家，人口超过2亿，使用超过500种语言，是世界上语言最为多元的国家之一。尽管如此，自然语言处理（NLP）研究大多集中于豪萨语、伊博语、尼日利亚皮钦语和约鲁巴语这四种语言（即不到尼日利亚所用语言的1%）。这在一定程度上是因为这些语言缺乏文本数据以训练和应用NLP算法。在本项工作中，我们介绍了ibom——一个包含来自阿夸伊博姆州沿海地区的阿郎冈语、埃菲克语、伊比比奥语和奥罗语的机器翻译和主题分类数据集。这些语言在谷歌翻译和其他主要基准如Flores-200和SIB-200中均未被涵盖。我们致力于将Flores-200基准扩展到这些语言，并进一步依据SIB-200分类数据集对翻译文本进行主题标签对齐。我们的评估表明，目前的预训练语言模型在零样本和少样本设置下的机器翻译性能欠佳。然而，我们发现少量样本能够逐步提高主题分类性能。 

---
# TriShGAN: Enhancing Sparsity and Robustness in Multivariate Time Series Counterfactuals Explanation 

**Title (ZH)**: TriShGAN: 提高多变量时间序列反事实解释的稀疏性和鲁棒性 

**Authors**: Hongnan Ma, Yiwei Shi, Guanxiong Sun, Mengyue Yang, Weiru Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06529)  

**Abstract**: In decision-making processes, stakeholders often rely on counterfactual explanations, which provide suggestions about what should be changed in the queried instance to alter the outcome of an AI system. However, generating these explanations for multivariate time series presents challenges due to their complex, multi-dimensional nature. Traditional Nearest Unlike Neighbor-based methods typically substitute subsequences in a queried time series with influential subsequences from an NUN, which is not always realistic in real-world scenarios due to the rigid direct substitution. Counterfactual with Residual Generative Adversarial Networks-based methods aim to address this by learning from the distribution of observed data to generate synthetic counterfactual explanations. However, these methods primarily focus on minimizing the cost from the queried time series to the counterfactual explanations and often neglect the importance of distancing the counterfactual explanation from the decision boundary. This oversight can result in explanations that no longer qualify as counterfactual if minor changes occur within the model. To generate a more robust counterfactual explanation, we introduce TriShGAN, under the CounteRGAN framework enhanced by the incorporation of triplet loss. This unsupervised learning approach uses distance metric learning to encourage the counterfactual explanations not only to remain close to the queried time series but also to capture the feature distribution of the instance with the desired outcome, thereby achieving a better balance between minimal cost and robustness. Additionally, we integrate a Shapelet Extractor that strategically selects the most discriminative parts of the high-dimensional queried time series to enhance the sparsity of counterfactual explanation and efficiency of the training process. 

**Abstract (ZH)**: 基于CounteRGAN框架的TriShGAN：增强三元损失的多变量时间序列对抗生成网络法 

---
# On the Analogy between Human Brain and LLMs: Spotting Key Neurons in Grammar Perception 

**Title (ZH)**: 人类大脑与大规模语言模型之间的类比：语法感知中的关键神经元识别 

**Authors**: Sanaz Saki Norouzi, Mohammad Masjedi, Pascal Hitzler  

**Link**: [PDF](https://arxiv.org/pdf/2511.06519)  

**Abstract**: Artificial Neural Networks, the building blocks of AI, were inspired by the human brain's network of neurons. Over the years, these networks have evolved to replicate the complex capabilities of the brain, allowing them to handle tasks such as image and language processing. In the realm of Large Language Models, there has been a keen interest in making the language learning process more akin to that of humans. While neuroscientific research has shown that different grammatical categories are processed by different neurons in the brain, we show that LLMs operate in a similar way. Utilizing Llama 3, we identify the most important neurons associated with the prediction of words belonging to different part-of-speech tags. Using the achieved knowledge, we train a classifier on a dataset, which shows that the activation patterns of these key neurons can reliably predict part-of-speech tags on fresh data. The results suggest the presence of a subspace in LLMs focused on capturing part-of-speech tag concepts, resembling patterns observed in lesion studies of the brain in neuroscience. 

**Abstract (ZH)**: 人工神经网络是人工智能的基石，它们受到了人脑神经网络的启发。这些网络在过去几年中不断发展，以复制大脑的复杂能力，使其能够处理图像和语言处理等任务。在大型语言模型领域，人们对其语言学习过程的兴趣在于使其更接近人类的语言学习方式。尽管神经科学研究表明，大脑中不同的语法类别是由不同的神经元处理的，但我们显示大型语言模型也以类似的方式运作。利用Llama 3，我们确定了与不同词性标签预测相关的最重要神经元。利用获得的知识，我们在数据集上训练了一个分类器，显示这些关键神经元的激活模式可以可靠地预测新数据的词性标签。结果表明，大型语言模型中存在一个专注于捕捉词性标签概念的子空间，类似于神经科学中大脑损伤研究中观察到的模式。 

---
# Rethinking what Matters: Effective and Robust Multilingual Realignment for Low-Resource Languages 

**Title (ZH)**: 重新思考什么是重要的：面向低资源语言的有效且鲁棒的多语种对齐方法 

**Authors**: Quang Phuoc Nguyen, David Anugraha, Felix Gaschi, Jun Bin Cheng, En-Shiun Annie Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.06497)  

**Abstract**: Realignment is a promising strategy to improve cross-lingual transfer in multilingual language models. However, empirical results are mixed and often unreliable, particularly for typologically distant or low-resource languages (LRLs) compared to English. Moreover, word realignment tools often rely on high-quality parallel data, which can be scarce or noisy for many LRLs. In this work, we conduct an extensive empirical study to investigate whether realignment truly benefits from using all available languages, or if strategically selected subsets can offer comparable or even improved cross-lingual transfer, and study the impact on LRLs. Our controlled experiments show that realignment can be particularly effective for LRLs and that using carefully selected, linguistically diverse subsets can match full multilingual alignment, and even outperform it for unseen LRLs. This indicates that effective realignment does not require exhaustive language coverage and can reduce data collection overhead, while remaining both efficient and robust when guided by informed language selection. 

**Abstract (ZH)**: 基于实证研究的重对齐策略在低资源语言上的有效性探究：部分选择而非全面覆盖 

---
# A Low-Rank Method for Vision Language Model Hallucination Mitigation in Autonomous Driving 

**Title (ZH)**: 一种低秩方法用于减轻自主驾驶中视觉语言模型幻觉 

**Authors**: Keke Long, Jiacheng Guo, Tianyun Zhang, Hongkai Yu, Xiaopeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06496)  

**Abstract**: Vision Language Models (VLMs) are increasingly used in autonomous driving to help understand traffic scenes, but they sometimes produce hallucinations, which are false details not grounded in the visual input. Detecting and mitigating hallucinations is challenging when ground-truth references are unavailable and model internals are inaccessible. This paper proposes a novel self-contained low-rank approach to automatically rank multiple candidate captions generated by multiple VLMs based on their hallucination levels, using only the captions themselves without requiring external references or model access. By constructing a sentence-embedding matrix and decomposing it into a low-rank consensus component and a sparse residual, we use the residual magnitude to rank captions: selecting the one with the smallest residual as the most hallucination-free. Experiments on the NuScenes dataset demonstrate that our approach achieves 87% selection accuracy in identifying hallucination-free captions, representing a 19% improvement over the unfiltered baseline and a 6-10% improvement over multi-agent debate method. The sorting produced by sparse error magnitudes shows strong correlation with human judgments of hallucinations, validating our scoring mechanism. Additionally, our method, which can be easily parallelized, reduces inference time by 51-67% compared to debate approaches, making it practical for real-time autonomous driving applications. 

**Abstract (ZH)**: 一种基于低秩的自包含方法用于多VLM候选caption的自动去幻觉排序 

---
# Route Experts by Sequence, not by Token 

**Title (ZH)**: 基于序列，而不是基于词，进行路径专家划分 

**Authors**: Tiansheng Wen, Yifei Wang, Aosong Feng, Long Ma, Xinyang Liu, Yifan Wang, Lixuan Guo, Bo Chen, Stefanie Jegelka, Chenyu You  

**Link**: [PDF](https://arxiv.org/pdf/2511.06494)  

**Abstract**: Mixture-of-Experts (MoE) architectures scale large language models (LLMs) by activating only a subset of experts per token, but the standard TopK routing assigns the same fixed number of experts to all tokens, ignoring their varying complexity. Prior adaptive routing methods introduce additional modules and hyperparameters, often requiring costly retraining from scratch. We propose Sequence-level TopK (SeqTopK), a minimal modification that shifts the expert budget from the token level to the sequence level. By selecting the top $T \cdot K$ experts across all $T$ tokens, SeqTopK enables end-to-end learned dynamic allocation -- assigning more experts to difficult tokens and fewer to easy ones -- while preserving the same overall budget. SeqTopK requires only a few lines of code, adds less than 1% overhead, and remains fully compatible with pretrained MoE models. Experiments across math, coding, law, and writing show consistent improvements over TopK and prior parameter-free adaptive methods, with gains that become substantially larger under higher sparsity (up to 16.9%). These results highlight SeqTopK as a simple, efficient, and scalable routing strategy, particularly well-suited for the extreme sparsity regimes of next-generation LLMs. Code is available at this https URL. 

**Abstract (ZH)**: 序列级TopK（SeqTopK）路由方法 

---
# Explainable AI For Early Detection Of Sepsis 

**Title (ZH)**: 可解释的人工智能在早期检测脓毒症中的应用 

**Authors**: Atharva Thakur, Shruti Dhumal  

**Link**: [PDF](https://arxiv.org/pdf/2511.06492)  

**Abstract**: Sepsis is a life-threatening condition that requires rapid detection and treatment to prevent progression to severe sepsis, septic shock, or multi-organ failure. Despite advances in medical technology, it remains a major challenge for clinicians. While recent machine learning models have shown promise in predicting sepsis onset, their black-box nature limits interpretability and clinical trust. In this study, we present an interpretable AI approach for sepsis analysis that integrates machine learning with clinical knowledge. Our method not only delivers accurate predictions of sepsis onset but also enables clinicians to understand, validate, and align model outputs with established medical expertise. 

**Abstract (ZH)**: 脓毒症是一种生命威胁性的状况，需要快速检测和治疗以防止进展为严重脓毒症、脓毒性休克或多器官功能衰竭。尽管医学技术取得了进步，但对临床医师来说仍是一项重大挑战。尽管最近的机器学习模型在预测脓毒症发作方面展现了潜力，但其黑箱特性限制了可解释性和临床信任。在本研究中，我们提出了一种可解释的AI方法，将机器学习与临床知识整合起来。该方法不仅提供准确的脓毒症发作预测，还使临床医师能够理解和验证模型输出，并与已有的医学专业知识相一致。 

---
# Zooming into Comics: Region-Aware RL Improves Fine-Grained Comic Understanding in Vision-Language Models 

**Title (ZH)**: 聚焦漫画：区域aware的RL提高视觉语言模型对漫画的细粒度理解 

**Authors**: Yule Chen, Yufan Ren, Sabine Süsstrunk  

**Link**: [PDF](https://arxiv.org/pdf/2511.06490)  

**Abstract**: Complex visual narratives, such as comics, present a significant challenge to Vision-Language Models (VLMs). Despite excelling on natural images, VLMs often struggle with stylized line art, onomatopoeia, and densely packed multi-panel layouts. To address this gap, we introduce AI4VA-FG, the first fine-grained and comprehensive benchmark for VLM-based comic understanding. It spans tasks from foundational recognition and detection to high-level character reasoning and narrative construction, supported by dense annotations for characters, poses, and depth. Beyond that, we evaluate state-of-the-art proprietary models, including GPT-4o and Gemini-2.5, and open-source models such as Qwen2.5-VL, revealing substantial performance deficits across core tasks of our benchmarks and underscoring that comic understanding remains an unsolved challenge. To enhance VLMs' capabilities in this domain, we systematically investigate post-training strategies, including supervised fine-tuning on solutions (SFT-S), supervised fine-tuning on reasoning trajectories (SFT-R), and reinforcement learning (RL). Beyond that, inspired by the emerging "Thinking with Images" paradigm, we propose Region-Aware Reinforcement Learning (RARL) for VLMs, which trains models to dynamically attend to relevant regions through zoom-in operations. We observe that when applied to the Qwen2.5-VL model, RL and RARL yield significant gains in low-level entity recognition and high-level storyline ordering, paving the way for more accurate and efficient VLM applications in the comics domain. 

**Abstract (ZH)**: 复杂的视觉叙事，如漫画，给视觉语言模型（VLMs）带来了重大挑战。尽管在自然图像上表现出色，VLMs在处理风格化线稿、拟声词和密集的多面板布局时经常遇到困难。为了解决这一差距，我们引入了AI4VA-FG，这是首个针对基于VLM的漫画理解的细粒度和综合基准。它涵盖了从基础识别和检测到高级角色推理和叙事构建的任务，配有密集的注释，包括角色、姿态和深度信息。此外，我们评估了最先进的私有模型（包括GPT-4o和Gemini-2.5）和开源模型（如Qwen2.5-VL），揭示了在我们基准的核心任务上存在显著性能缺陷，突显了漫画理解仍然是一个未解之谜。为了增强VLMs在此领域的功能，我们系统地探讨了后训练策略，包括基于解决方案的监督微调（SFT-S）、基于推理轨迹的监督微调（SFT-R）和强化学习（RL）。此外，受到新兴的“图像思维”范式的启发，我们为VLMs提出了区域意识强化学习（RARL）方法，通过缩放操作使模型能够动态关注相关区域。我们发现，当应用于Qwen2.5-VL模型时，RL和RARL在低级实体识别和高级故事情节排序方面取得了显著进步，为在漫画领域实现更准确和高效的VLM应用铺平了道路。 

---
# EchoMark: Perceptual Acoustic Environment Transfer with Watermark-Embedded Room Impulse Response 

**Title (ZH)**: EchoMark: 嵌入水印的房间冲激响应感知声学环境转移 

**Authors**: Chenpei Huang, Lingfeng Yao, Kyu In Lee, Lan Emily Zhang, Xun Chen, Miao Pan  

**Link**: [PDF](https://arxiv.org/pdf/2511.06458)  

**Abstract**: Acoustic Environment Matching (AEM) is the task of transferring clean audio into a target acoustic environment, enabling engaging applications such as audio dubbing and auditory immersive virtual reality (VR). Recovering similar room impulse response (RIR) directly from reverberant speech offers more accessible and flexible AEM solution. However, this capability also introduces vulnerabilities of arbitrary ``relocation" if misused by malicious user, such as facilitating advanced voice spoofing attacks or undermining the authenticity of recorded evidence. To address this issue, we propose EchoMark, the first deep learning-based AEM framework that generates perceptually similar RIRs with embedded watermark. Our design tackle the challenges posed by variable RIR characteristics, such as different durations and energy decays, by operating in the latent domain. By jointly optimizing the model with a perceptual loss for RIR reconstruction and a loss for watermark detection, EchoMark achieves both high-quality environment transfer and reliable watermark recovery. Experiments on diverse datasets validate that EchoMark achieves room acoustic parameter matching performance comparable to FiNS, the state-of-the-art RIR estimator. Furthermore, a high Mean Opinion Score (MOS) of 4.22 out of 5, watermark detection accuracy exceeding 99\%, and bit error rates (BER) below 0.3\% collectively demonstrate the effectiveness of EchoMark in preserving perceptual quality while ensuring reliable watermark embedding. 

**Abstract (ZH)**: 声环境匹配（AEM）：基于深度学习的嵌入水印声环境匹配框架 

---
# A Multi-Agent System for Semantic Mapping of Relational Data to Knowledge Graphs 

**Title (ZH)**: 基于关系数据语义映射的多智能体系统到知识图谱 

**Authors**: Milena Trajanoska, Riste Stojanov, Dimitar Trajanov  

**Link**: [PDF](https://arxiv.org/pdf/2511.06455)  

**Abstract**: Enterprises often maintain multiple databases for storing critical business data in siloed systems, resulting in inefficiencies and challenges with data interoperability. A key to overcoming these challenges lies in integrating disparate data sources, enabling businesses to unlock the full potential of their data. Our work presents a novel approach for integrating multiple databases using knowledge graphs, focusing on the application of large language models as semantic agents for mapping and connecting structured data across systems by leveraging existing vocabularies. The proposed methodology introduces a semantic layer above tables in relational databases, utilizing a system comprising multiple LLM agents that map tables and columns to this http URL terms. Our approach achieves a mapping accuracy of over 90% in multiple domains. 

**Abstract (ZH)**: 企业常常为存储关键业务数据而在孤立系统中维护多个数据库，导致效率低下并带来数据互操作性方面的挑战。克服这些挑战的关键在于整合不同的数据源，使企业能够充分挖掘其数据的潜力。我们的工作提出了一种使用知识图谱整合多个数据库的新方法，该方法侧重于利用大型语言模型作为语义代理，通过利用现有词汇表来跨系统映射和连接结构化数据。所提出的方法在关系数据库的表上引入了一个语义层，利用由多个LLM代理组成的系统将表和列映射到统一术语。该方法在多个领域实现了超过90%的映射准确率。 

---
# FLEX: Continuous Agent Evolution via Forward Learning from Experience 

**Title (ZH)**: FLEX: 通过经验前向学习实现持续智能体进化 

**Authors**: Zhicheng Cai, Xinyuan Guo, Yu Pei, JiangTao Feng, Jiangjie Chen, Ya-Qin Zhang, Wei-Ying Ma, Mingxuan Wang, Hao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.06449)  

**Abstract**: Autonomous agents driven by Large Language Models (LLMs) have revolutionized reasoning and problem-solving but remain static after training, unable to grow with experience as intelligent beings do during deployment. We introduce Forward Learning with EXperience (FLEX), a gradient-free learning paradigm that enables LLM agents to continuously evolve through accumulated experience. Specifically, FLEX cultivates scalable and inheritable evolution by constructing a structured experience library through continual reflection on successes and failures during interaction with the environment. FLEX delivers substantial improvements on mathematical reasoning, chemical retrosynthesis, and protein fitness prediction (up to 23% on AIME25, 10% on USPTO50k, and 14% on ProteinGym). We further identify a clear scaling law of experiential growth and the phenomenon of experience inheritance across agents, marking a step toward scalable and inheritable continuous agent evolution. Project Page: this https URL. 

**Abstract (ZH)**: 由大规模语言模型驱动的自主代理通过经验前向学习（FLEX）实现连续进化：一种无需梯度的学习范式 

---
# When AI Agents Collude Online: Financial Fraud Risks by Collaborative LLM Agents on Social Platforms 

**Title (ZH)**: 当AI代理在线共谋：社交平台上协作LLM代理的金融欺诈风险 

**Authors**: Qibing Ren, Zhijie Zheng, Jiaxuan Guo, Junchi Yan, Lizhuang Ma, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2511.06448)  

**Abstract**: In this work, we study the risks of collective financial fraud in large-scale multi-agent systems powered by large language model (LLM) agents. We investigate whether agents can collaborate in fraudulent behaviors, how such collaboration amplifies risks, and what factors influence fraud success. To support this research, we present MultiAgentFraudBench, a large-scale benchmark for simulating financial fraud scenarios based on realistic online interactions. The benchmark covers 28 typical online fraud scenarios, spanning the full fraud lifecycle across both public and private domains. We further analyze key factors affecting fraud success, including interaction depth, activity level, and fine-grained collaboration failure modes. Finally, we propose a series of mitigation strategies, including adding content-level warnings to fraudulent posts and dialogues, using LLMs as monitors to block potentially malicious agents, and fostering group resilience through information sharing at the societal level. Notably, we observe that malicious agents can adapt to environmental interventions. Our findings highlight the real-world risks of multi-agent financial fraud and suggest practical measures for mitigating them. Code is available at this https URL. 

**Abstract (ZH)**: 本研究探讨了由大规模语言模型代理驱动的大规模多代理系统中存在的集体财务欺诈风险。我们研究代理是否能够协作进行欺诈行为，这种协作如何放大风险，以及哪些因素影响欺诈的成功。为支持这一研究，我们提出了MultiAgentFraudBench，这是一个基于现实在线互动的大规模基准，用于模拟财务欺诈场景。该基准涵盖了28种典型的在线欺诈场景，贯穿了公共和私人领域的完整欺诈生命周期。我们进一步分析影响欺诈成功的关键因素，包括互动深度、活动水平和精细的协作失败模式。最后，我们提出了若干缓解策略，包括在欺诈性帖子和对话中添加内容级别的警告，使用语言模型作为监控器阻止潜在恶意代理，并通过社会层面的信息共享增强群体韧性。值得注意的是，我们发现恶意代理能够适应环境干预。研究结果突显了多代理财务欺诈的现实风险，并建议了减轻这些风险的实用措施。代码可在以下链接获取。 

---
# Personality over Precision: Exploring the Influence of Human-Likeness on ChatGPT Use for Search 

**Title (ZH)**: 以人格胜于精确度：探索人类相似性对ChatGPT用于搜索影响的研究 

**Authors**: Mert Yazan, Frederik Bungaran Ishak Situmeang, Suzan Verberne  

**Link**: [PDF](https://arxiv.org/pdf/2511.06447)  

**Abstract**: Conversational search interfaces, like ChatGPT, offer an interactive, personalized, and engaging user experience compared to traditional search. On the downside, they are prone to cause overtrust issues where users rely on their responses even when they are incorrect. What aspects of the conversational interaction paradigm drive people to adopt it, and how it creates personalized experiences that lead to overtrust, is not clear. To understand the factors influencing the adoption of conversational interfaces, we conducted a survey with 173 participants. We examined user perceptions regarding trust, human-likeness (anthropomorphism), and design preferences between ChatGPT and Google. To better understand the overtrust phenomenon, we asked users about their willingness to trade off factuality for constructs like ease of use or human-likeness. Our analysis identified two distinct user groups: those who use both ChatGPT and Google daily (DUB), and those who primarily rely on Google (DUG). The DUB group exhibited higher trust in ChatGPT, perceiving it as more human-like, and expressed greater willingness to trade factual accuracy for enhanced personalization and conversational flow. Conversely, the DUG group showed lower trust toward ChatGPT but still appreciated aspects like ad-free experiences and responsive interactions. Demographic analysis further revealed nuanced patterns, with middle-aged adults using ChatGPT less frequently yet trusting it more, suggesting potential vulnerability to misinformation. Our findings contribute to understanding user segmentation, emphasizing the critical roles of personalization and human-likeness in conversational IR systems, and reveal important implications regarding users' willingness to compromise factual accuracy for more engaging interactions. 

**Abstract (ZH)**: 会话搜索界面，如ChatGPT，与传统搜索相比，提供了一种交互式、个性化和参与感更强的用户体验。然而，它们容易引发过度信任问题，即用户即使知道其答案有误仍过分依赖。不清楚哪些方面驱动人们采用会话交互范式，以及这种范式如何创造个性化体验从而导致过度信任。为了了解影响会话界面采用的因素，我们开展了一项包含173名参与者的调查。我们考察了用户对ChatGPT和Google的信任感知、拟人化程度以及设计偏好。为进一步理解过度信任现象，我们询问了用户为了使用简便或拟人化等属性愿意牺牲事实准确性的程度。我们的分析识别出了两类不同的用户群体：每日同时使用ChatGPT和Google的用户（DUB）以及主要依赖Google的用户（DUG）。DUB群体对ChatGPT表现出更高信任度，并认为其更容易拟人化，表达了愿意为个性化的增强和流畅的对话体验而牺牲事实准确性的意愿。相反，DUG群体对ChatGPT的信任度较低，但仍然欣赏无广告和及时响应等特性。进一步的 demographic 分析揭示了复杂的模式，中年人使用ChatGPT的频率较低但信任度较高，这表明他们可能对错误信息更易感。我们的发现有助于理解用户细分，突出个性化和拟人化在会话信息检索系统中的关键作用，并揭示了用户愿意为了更吸引人的交互体验而妥协事实准确性的深刻含义。 

---
# SR-KI: Scalable and Real-Time Knowledge Integration into LLMs via Supervised Attention 

**Title (ZH)**: SR-KI: 面向LLMs的可扩展和实时知识整合方法 via 监督注意力 

**Authors**: Bohan Yu, Wei Huang, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06446)  

**Abstract**: This paper proposes SR-KI, a novel approach for integrating real-time and large-scale structured knowledge bases (KBs) into large language models (LLMs). SR-KI begins by encoding KBs into key-value pairs using a pretrained encoder, and injects them into LLMs' KV cache. Building on this representation, we employ a two-stage training paradigm: first locating a dedicated retrieval layer within the LLM, and then applying an attention-based loss at this layer to explicitly supervise attention toward relevant KB entries. Unlike traditional retrieval-augmented generation methods that rely heavily on the performance of external retrievers and multi-stage pipelines, SR-KI supports end-to-end inference by performing retrieval entirely within the models latent space. This design enables efficient compression of injected knowledge and facilitates dynamic knowledge updates. Comprehensive experiments demonstrate that SR-KI enables the integration of up to 40K KBs into a 7B LLM on a single A100 40GB GPU, and achieves strong retrieval performance, maintaining over 98% Recall@10 on the best-performing task and exceeding 88% on average across all tasks. Task performance on question answering and KB ID generation also demonstrates that SR-KI maintains strong performance while achieving up to 99.75% compression of the injected KBs. 

**Abstract (ZH)**: 本文提出SR-KI，这是一种将实时和大规模结构化知识库（KBs）集成到大型语言模型（LLMs）中的新颖方法。 

---
# Walking the Tightrope of LLMs for Software Development: A Practitioners' Perspective 

**Title (ZH)**: 在软件开发中谨慎行走于大语言模型的边缘：从业务人员视角 

**Authors**: Samuel Ferino, Rashina Hoda, John Grundy, Christoph Treude  

**Link**: [PDF](https://arxiv.org/pdf/2511.06428)  

**Abstract**: Background: Large Language Models emerged with the potential of provoking a revolution in software development (e.g., automating processes, workforce transformation). Although studies have started to investigate the perceived impact of LLMs for software development, there is a need for empirical studies to comprehend how to balance forward and backward effects of using LLMs. Objective: We investigated how LLMs impact software development and how to manage the impact from a software developer's perspective. Method: We conducted 22 interviews with software practitioners across 3 rounds of data collection and analysis, between October (2024) and September (2025). We employed socio-technical grounded theory (STGT) for data analysis to rigorously analyse interview participants' responses. Results: We identified the benefits (e.g., maintain software development flow, improve developers' mental model, and foster entrepreneurship) and disadvantages (e.g., negative impact on developers' personality and damage to developers' reputation) of using LLMs at individual, team, organisation, and society levels; as well as best practices on how to adopt LLMs. Conclusion: Critically, we present the trade-offs that software practitioners, teams, and organisations face in working with LLMs. Our findings are particularly useful for software team leaders and IT managers to assess the viability of LLMs within their specific context. 

**Abstract (ZH)**: 背景：大型语言模型有可能引发软件开发领域的革命（例如，自动化流程、劳动力转型）。尽管已有研究开始探讨大型语言模型对软件开发的感知影响，但仍需通过实证研究来理解如何平衡使用大型语言模型的前向效应和后向效应。目标：我们调查了大型语言模型如何影响软件开发，以及从软件开发者的角度如何管理这些影响。方法：我们于2024年10月至2025年9月间进行了三轮数据收集和分析，共对22名软件从业者进行了访谈。我们采用了社会技术扎根理论（STGT）进行数据分析，以严格分析访谈参与者的意见。结果：我们识别了大型语言模型在个体、团队、组织和社会层面的利弊（例如，维持软件开发流程、提高开发者的心理模型、促进创业精神等，以及对开发者个性的负面影响和损害开发者声誉等问题），并提出了采用大型语言模型的最佳实践。结论：关键的是，我们呈现了软件从业者、团队和组织在使用大型语言模型时面临的权衡。我们的发现特别有助于软件团队领导者和IT管理人员根据其特定情境评估大型语言模型的可行性。 

---
# Turbo-DDCM: Fast and Flexible Zero-Shot Diffusion-Based Image Compression 

**Title (ZH)**: Turbo-DDCM：快速灵活的零样本扩散图像压缩 

**Authors**: Amit Vaisman, Guy Ohayon, Hila Manor, Michael Elad, Tomer Michaeli  

**Link**: [PDF](https://arxiv.org/pdf/2511.06424)  

**Abstract**: While zero-shot diffusion-based compression methods have seen significant progress in recent years, they remain notoriously slow and computationally demanding. This paper presents an efficient zero-shot diffusion-based compression method that runs substantially faster than existing methods, while maintaining performance that is on par with the state-of-the-art techniques. Our method builds upon the recently proposed Denoising Diffusion Codebook Models (DDCMs) compression scheme. Specifically, DDCM compresses an image by sequentially choosing the diffusion noise vectors from reproducible random codebooks, guiding the denoiser's output to reconstruct the target image. We modify this framework with Turbo-DDCM, which efficiently combines a large number of noise vectors at each denoising step, thereby significantly reducing the number of required denoising operations. This modification is also coupled with an improved encoding protocol. Furthermore, we introduce two flexible variants of Turbo-DDCM, a priority-aware variant that prioritizes user-specified regions and a distortion-controlled variant that compresses an image based on a target PSNR rather than a target BPP. Comprehensive experiments position Turbo-DDCM as a compelling, practical, and flexible image compression scheme. 

**Abstract (ZH)**: 尽管零-shot 扩散基于压缩方法在近年来取得了显著进展，但仍存在计算速度慢和计算需求高的问题。本文提出了一种比现有方法运行速度快得多且保持与当今最先进技术相当性能的高效零-shot 扩散基于压缩方法。该方法建立在最近提出的去噪扩散码本模型（DDCMs）压缩方案之上。具体而言，DDCM通过从可重复的随机码本中顺序选择去噪噪声向量来压缩图像，引导去噪器的输出重建目标图像。我们通过Turbo-DDCM对该框架进行了改进，该方法在每次去噪步骤中高效地结合了大量噪声向量，从而显著减少了所需的去噪操作次数。此外，还引入了两种Turbo-DDCM的灵活变体：一种意识优先级变体，优先处理用户指定的区域；一种受控失真变体，基于目标PSNR而非目标BPP进行图像压缩。全面的实验表明，Turbo-DDCM是一种具有吸引力、实用且灵活的图像压缩方案。 

---
# On Modality Incomplete Infrared-Visible Object Detection: An Architecture Compatibility Perspective 

**Title (ZH)**: 红外-可见光物体检测中模态不完整问题：从架构兼容性视角探讨 

**Authors**: Shuo Yang, Yinghui Xing, Shizhou Zhang, Zhilong Niu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06406)  

**Abstract**: Infrared and visible object detection (IVOD) is essential for numerous around-the-clock applications. Despite notable advancements, current IVOD models exhibit notable performance declines when confronted with incomplete modality data, particularly if the dominant modality is missing. In this paper, we take a thorough investigation on modality incomplete IVOD problem from an architecture compatibility perspective. Specifically, we propose a plug-and-play Scarf Neck module for DETR variants, which introduces a modality-agnostic deformable attention mechanism to enable the IVOD detector to flexibly adapt to any single or double modalities during training and inference. When training Scarf-DETR, we design a pseudo modality dropout strategy to fully utilize the multi-modality information, making the detector compatible and robust to both working modes of single and double modalities. Moreover, we introduce a comprehensive benchmark for the modality-incomplete IVOD task aimed at thoroughly assessing situations where the absent modality is either dominant or secondary. Our proposed Scarf-DETR not only performs excellently in missing modality scenarios but also achieves superior performances on the standard IVOD modality complete benchmarks. Our code will be available at this https URL. 

**Abstract (ZH)**: 红外与可见光物体检测（IVOD）对于众多全天候应用至关重要。尽管取得了显著进展，但目前的IVOD模型在面对不完整模态数据时表现出明显的性能下降，尤其是在主要模态缺失的情况下。在本文中，我们从架构兼容性的角度对不完整模态IVOD问题进行了彻底的研究。具体地，我们为DETR变体提出了一个即插即用的Scarf Neck模块，该模块引入了一种模态无关的可变形注意力机制，使IVOD检测器能够在训练和推断过程中灵活适应单一或双模态数据。在训练Scarf-DETR时，我们设计了一种伪模态dropout策略，充分利用多模态信息，使检测器能够兼容并 robust 于单一和双模态工作模式。此外，我们引入了一个全面的基准测试，用于彻底评估主要模态缺失或次要模态缺失的情况。我们的Scarf-DETR不仅在模态缺失场景中表现出色，还在标准IVOD完整模态基准测试中取得了优异性能。我们的代码将在此网址获取：this https URL。 

---
# HatePrototypes: Interpretable and Transferable Representations for Implicit and Explicit Hate Speech Detection 

**Title (ZH)**: 可解释且可迁移的表示模型：隐式和显式仇恨言语检测 

**Authors**: Irina Proskurina, Marc-Antoine Carpentier, Julien Velcin  

**Link**: [PDF](https://arxiv.org/pdf/2511.06391)  

**Abstract**: Optimization of offensive content moderation models for different types of hateful messages is typically achieved through continued pre-training or fine-tuning on new hate speech benchmarks. However, existing benchmarks mainly address explicit hate toward protected groups and often overlook implicit or indirect hate, such as demeaning comparisons, calls for exclusion or violence, and subtle discriminatory language that still causes harm. While explicit hate can often be captured through surface features, implicit hate requires deeper, full-model semantic processing. In this work, we question the need for repeated fine-tuning and analyze the role of HatePrototypes, class-level vector representations derived from language models optimized for hate speech detection and safety moderation. We find that these prototypes, built from as few as 50 examples per class, enable cross-task transfer between explicit and implicit hate, with interchangeable prototypes across benchmarks. Moreover, we show that parameter-free early exiting with prototypes is effective for both hate types. We release the code, prototype resources, and evaluation scripts to support future research on efficient and transferable hate speech detection. 

**Abstract (ZH)**: 优化不同类型的仇恨信息过滤模型通常通过持续的预训练或针对新仇恨言论基准的数据微调来实现。然而，现有基准主要针对明确针对受保护群体的仇恨言论，往往忽视了隐含或间接的仇恨，如贬低比较、排他或暴力呼吁以及仍具有危害性的微妙歧视语言。虽然明确的仇恨可以通过表面特征捕捉，但隐含的仇恨需要更深层次的全面模型语义处理。在本工作中，我们质疑重复微调的必要性，并分析HatePrototypes的作用，这是一类从优化仇恨言论检测和安全过滤的语言模型中提取的类级向量表示。我们发现，这些原型，即使仅基于每个类别的50个示例，也能在明确和隐含仇恨之间实现跨任务转移，且在不同基准中可以互换。此外，我们展示了无参数早期退出与原型的有效性，适用于两种类型的仇恨言论。我们发布了代码、原型资源和评估脚本，以支持未来在高效和可转移仇恨言论检测方面的研究。 

---
# Ghost in the Transformer: Tracing LLM Lineage with SVD-Fingerprint 

**Title (ZH)**: Transformer中的幽灵：通过SVD-指纹追踪大模型谱系 

**Authors**: Suqing Wang, Ziyang Ma, Xinyi Li, Zuchao Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06390)  

**Abstract**: Large Language Models (LLMs) have rapidly advanced and are widely adopted across diverse fields. Due to the substantial computational cost and data requirements of training from scratch, many developers choose to fine-tune or modify existing open-source models. While most adhere to open-source licenses, some falsely claim original training despite clear derivation from public models. This raises pressing concerns about intellectual property protection and highlights the need for reliable methods to verify model provenance. In this paper, we propose GhostSpec, a lightweight yet effective method for verifying LLM lineage without access to training data or modification of model behavior. Our approach constructs compact and robust fingerprints by applying singular value decomposition (SVD) to invariant products of internal attention weight matrices, effectively capturing the structural identity of a model. Unlike watermarking or output-based methods, GhostSpec is fully data-free, non-invasive, and computationally efficient. It demonstrates strong robustness to sequential fine-tuning, pruning, block expansion, and even adversarial transformations. Extensive experiments show that GhostSpec can reliably trace the lineage of transformed models with minimal overhead. By offering a practical solution for model verification and reuse tracking, our method contributes to the protection of intellectual property and fosters a transparent, trustworthy ecosystem for large-scale language models. 

**Abstract (ZH)**: 大型语言模型（LLMs）已迅速发展并在多个领域广泛应用。由于从头训练的巨大计算成本和数据需求，许多开发人员选择微调或修改现有的开源模型。虽然大多数模型遵守开源许可，但有些却虚假声称原始训练，尽管这些模型明显源自公开模型。这引发了关于知识产权保护的重大关注，并凸显了需要可靠方法验证模型来源的需求。在本文中，我们提出了GhostSpec，一种无需访问训练数据或修改模型行为的轻量级有效方法，用于验证LLM谱系。我们的方法通过对内部注意权重矩阵的不变产品应用奇异值分解（SVD），构建紧凑且稳健的指纹，有效地捕捉模型的结构身份。与水印或基于输出的方法不同，GhostSpec完全无需数据、无侵入且计算效率高。它在序列微调、剪枝、块扩展和对抗变换等情况下表现出强大的 robustness。广泛的实验表明，GhostSpec可以在最小开销下可靠地追溯转换模型的谱系。通过提供模型验证和重用跟踪的实用解决方案，我们的方法有助于保护知识产权，并促进大规模语言模型的透明、可信赖生态系统。 

---
# HyMoERec: Hybrid Mixture-of-Experts for Sequential Recommendation 

**Title (ZH)**: HyMoERec: 混合专家混合模型的序列推荐 

**Authors**: Kunrong Li, Zhu Sun, Kwan Hui Lim  

**Link**: [PDF](https://arxiv.org/pdf/2511.06388)  

**Abstract**: We propose HyMoERec, a novel sequential recommendation framework that addresses the limitations of uniform Position-wise Feed-Forward Networks in existing models. Current approaches treat all user interactions and items equally, overlooking the heterogeneity in user behavior patterns and diversity in item complexity. HyMoERec initially introduces a hybrid mixture-of-experts architecture that combines shared and specialized expert branches with an adaptive expert fusion mechanism for the sequential recommendation task. This design captures diverse reasoning for varied users and items while ensuring stable training. Experiments on MovieLens-1M and Beauty datasets demonstrate that HyMoERec consistently outperforms state-of-the-art baselines. 

**Abstract (ZH)**: 我们提出HyMoERec，一种新颖的序列推荐框架，该框架解决了现有模型中均匀位置-wise前馈网络的局限性。当前的方法平等对待所有用户交互和项目，忽略了用户行为模式的异质性和项目复杂度的多样性。HyMoERec 初始引入了一种混合的专家混合架构，该架构结合了共享和专门化的专家分支，并配有自适应的专家融合机制，以应对序列推荐任务。该设计能够捕捉多样化的推理机制，同时确保训练稳定。实验结果表明，HyMoERec 在 MovieLens-1M 和 Beauty 数据集上始终优于最先进的基线方法。 

---
# Privacy-Preserving Federated Learning for Fair and Efficient Urban Traffic Optimization 

**Title (ZH)**: 隐私保护的联邦学习在公平与高效的都市交通优化中的应用 

**Authors**: Rathin Chandra Shit, Sharmila Subudhi  

**Link**: [PDF](https://arxiv.org/pdf/2511.06363)  

**Abstract**: The optimization of urban traffic is threatened by the complexity of achieving a balance between transport efficiency and the maintenance of privacy, as well as the equitable distribution of traffic based on socioeconomically diverse neighborhoods. Current centralized traffic management schemes invade user location privacy and further entrench traffic disparity by offering disadvantaged route suggestions, whereas current federated learning frameworks do not consider fairness constraints in multi-objective traffic settings. This study presents a privacy-preserving federated learning framework, termed FedFair-Traffic, that jointly and simultaneously optimizes travel efficiency, traffic fairness, and differential privacy protection. This is the first attempt to integrate three conflicting objectives to improve urban transportation systems. The proposed methodology enables collaborative learning between related vehicles with data locality by integrating Graph Neural Networks with differential privacy mechanisms ($\epsilon$-privacy guarantees) and Gini coefficient-based fair constraints using multi-objective optimization. The framework uses federated aggregation methods of gradient clipping and noise injection to provide differential privacy and optimize Pareto-efficient solutions for the efficiency-fairness tradeoff. Real-world comprehensive experiments on the METR-LA traffic dataset showed that FedFair-Traffic can reduce the average travel time by 7\% (14.2 minutes) compared with their centralized baselines, promote traffic fairness by 73\% (Gini coefficient, 0.78), and offer high privacy protection (privacy score, 0.8) with an 89\% reduction in communication overhead. These outcomes demonstrate that FedFair-Traffic is a scalable privacy-aware smart city infrastructure with possible use-cases in metropolitan traffic flow control and federated transportation networks. 

**Abstract (ZH)**: 面向隐私保护的联合旅行效率、交通公平性和差分隐私保护的联邦学习框架：FedFair-Traffic 

---
# Understanding Student Interaction with AI-Powered Next-Step Hints: Strategies and Challenges 

**Title (ZH)**: 理解学生与AI驱动的下一步提示交互：策略与挑战 

**Authors**: Anastasiia Birillo, Aleksei Rostovskii, Yaroslav Golubev, Hieke Keuning  

**Link**: [PDF](https://arxiv.org/pdf/2511.06362)  

**Abstract**: Automated feedback generation plays a crucial role in enhancing personalized learning experiences in computer science education. Among different types of feedback, next-step hint feedback is particularly important, as it provides students with actionable steps to progress towards solving programming tasks. This study investigates how students interact with an AI-driven next-step hint system in an in-IDE learning environment. We gathered and analyzed a dataset from 34 students solving Kotlin tasks, containing detailed hint interaction logs. We applied process mining techniques and identified 16 common interaction scenarios. Semi-structured interviews with 6 students revealed strategies for managing unhelpful hints, such as adapting partial hints or modifying code to generate variations of the same hint. These findings, combined with our publicly available dataset, offer valuable opportunities for future research and provide key insights into student behavior, helping improve hint design for enhanced learning support. 

**Abstract (ZH)**: 自动反馈生成在计算机科学教育中增强个性化学习体验中扮演着重要角色。在不同类型的教学反馈中，下一步提示反馈尤为重要，因为它为学生提供了可操作的步骤，以解决编程任务。本研究探讨了学生在集成开发环境（IDE）中与AI驱动的下一步提示系统互动的方式。我们收集并分析了34名学生解决Kotlin任务的数据集，包含详细的提示交互日志。我们应用了过程挖掘技术，并识别出16种常见的交互场景。与6名学生的半结构化访谈揭示了管理无用提示的策略，如适应部分提示或修改代码以生成相同的提示变体。结合我们公开提供的数据集，这些发现为未来研究提供了宝贵的机遇，并为理解学生行为提供关键见解，有助于改进提示设计以增强学习支持。 

---
# A Graph-Theoretical Perspective on Law Design for Multiagent Systems 

**Title (ZH)**: 基于图论视角的多Agent系统法律设计研究 

**Authors**: Qi Shi, Pavel Naumov  

**Link**: [PDF](https://arxiv.org/pdf/2511.06361)  

**Abstract**: A law in a multiagent system is a set of constraints imposed on agents' behaviours to avoid undesirable outcomes. The paper considers two types of laws: useful laws that, if followed, completely eliminate the undesirable outcomes and gap-free laws that guarantee that at least one agent can be held responsible each time an undesirable outcome occurs. In both cases, we study the problem of finding a law that achieves the desired result by imposing the minimum restrictions.
We prove that, for both types of laws, the minimisation problem is NP-hard even in the simple case of one-shot concurrent interactions. We also show that the approximation algorithm for the vertex cover problem in hypergraphs could be used to efficiently approximate the minimum laws in both cases. 

**Abstract (ZH)**: 多智能体系统中的一条法律是一组对智能体行为的约束，以避免不良结果。本文考虑了两类法律：有用法律，若遵循则完全消除不良结果；空白法律，保证每次发生不良结果时至少有一个智能体需承担责任。在两种情况下，我们研究了通过施加最小限制来获得所需结果的问题。我们证明，对于两类法律，即使在单次并发交互的简单情况下，最小化问题也是NP-hard。我们还表明，超图顶点覆盖问题的近似算法可以高效地近似这两种情况下的最小法律。 

---
# Reaction Prediction via Interaction Modeling of Symmetric Difference Shingle Sets 

**Title (ZH)**: 基于对称差最小片段集相互作用建模的反应预测 

**Authors**: Runhan Shi, Letian Chen, Gufeng Yu, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06356)  

**Abstract**: Chemical reaction prediction remains a fundamental challenge in organic chemistry, where existing machine learning models face two critical limitations: sensitivity to input permutations (molecule/atom orderings) and inadequate modeling of substructural interactions governing reactivity. These shortcomings lead to inconsistent predictions and poor generalization to real-world scenarios. To address these challenges, we propose ReaDISH, a novel reaction prediction model that learns permutation-invariant representations while incorporating interaction-aware features. It introduces two innovations: (1) symmetric difference shingle encoding, which computes molecular shingle differences to capture reaction-specific structural changes while eliminating order sensitivity; and (2) geometry-structure interaction attention, a mechanism that models intra- and inter-molecular interactions at the shingle level. Extensive experiments demonstrate that ReaDISH improves reaction prediction performance across diverse benchmarks. It shows enhanced robustness with an average improvement of 8.76% on R$^2$ under permutation perturbations. 

**Abstract (ZH)**: 化学反应预测仍然是有机化学中的一个基本挑战，现有的机器学习模型面临两个关键限制：对输入排列的敏感性和对控制反应性的子结构相互作用建模不足。这些不足导致了预测结果的一致性差和泛化能力差。为了解决这些挑战，我们提出了一种名为ReaDISH的新反应预测模型，该模型在学习排列不变表示的同时，结合了相互作用感知的特征。该模型引入了两项创新：（1）对称差分切片编码，该编码计算分子切片差异以捕获反应特定的结构性变化，同时消除排序敏感性；（2）几何-结构相互作用注意力机制，在切片级别建模分子内和分子间的相互作用。广泛的实验证明，ReaDISH在多种基准测试中提高了反应预测性能。在排列扰动下，它显示了增强的鲁棒性，平均提高了8.76%的R²。 

---
# GazeVLM: A Vision-Language Model for Multi-Task Gaze Understanding 

**Title (ZH)**: GazeVLM：多任务目光理解的视觉-语言模型 

**Authors**: Athul M. Mathew, Haithem Hermassi, Thariq Khalid, Arshad Ali Khan, Riad Souissi  

**Link**: [PDF](https://arxiv.org/pdf/2511.06348)  

**Abstract**: Gaze understanding unifies the detection of people, their gaze targets, and objects of interest into a single framework, offering critical insight into visual attention and intent estimation. Although prior research has modelled gaze cues in visual scenes, a unified system is still needed for gaze understanding using both visual and language prompts. This paper introduces GazeVLM, a novel Vision-Language Model (VLM) for multi-task gaze understanding in images, addressing person detection, gaze target detection, and gaze object identification. While other transformer-based methods exist for gaze analysis, GazeVLM represents, to our knowledge, the first application of a VLM to these combined tasks, allowing for selective execution of each task. Through the integration of visual (RGB and depth) and textual modalities, our ablation study on visual input combinations revealed that a fusion of RGB images with HHA-encoded depth maps, guided by text prompts, yields superior performance. We also introduce an object-level gaze detection metric for gaze object identification ($AP_{ob}$). Through experiments, GazeVLM demonstrates significant improvements, notably achieving state-of-the-art evaluation scores on GazeFollow and VideoAttentionTarget datasets. 

**Abstract (ZH)**: 凝视理解将人员检测、凝视目标检测和兴趣对象识别统一于单一框架，为视觉注意力和意图估计提供了关键洞察。尽管 prior research 已对视觉场景中的凝视线索进行了建模，但仍然需要综合利用视觉和语言提示的统一系统来进行凝视理解。本文介绍了一种新型的视觉-语言模型 (VLM)——GazeVLM，用于图像中的多任务凝视理解，包括人员检测、凝视目标检测和凝视对象识别。尽管存在其他基于变压器的方法来进行凝视分析，但据我们所知，GazeVLM 是首次将 VLM 应用于这些综合任务，允许针对每个任务进行选择性执行。通过结合视觉（RGB 和深度）和文本模态，我们的消融研究发现了 RGB 图像与由文本提示引导的 HHA 编码深度图融合在视觉输入组合中的优异性能。我们还引入了一种基于对象级别的凝视检测度量（$AP_{ob}$）用于凝视对象识别。通过实验，GazeVLM 在 GazeFollow 和 VideoAttentionTarget 数据集上取得了显著改善，并达到了最先进的评估分数。 

---
# PRAGMA: A Profiling-Reasoned Multi-Agent Framework for Automatic Kernel Optimization 

**Title (ZH)**: PRAGMA：一种基于 profiling 的多Agent自动内核优化框架 

**Authors**: Kelun Lei, Hailong Yang, Huaitao Zhang, Xin You, Kaige Zhang, Zhongzhi Luan, Yi Liu, Depei Qian  

**Link**: [PDF](https://arxiv.org/pdf/2511.06345)  

**Abstract**: Designing high-performance kernels requires expert-level tuning and a deep understanding of hardware characteristics. Recent advances in large language models (LLMs) have enabled automated kernel generation, yet most existing systems rely solely on correctness or execution time feedback, lacking the ability to reason about low-level performance bottlenecks. In this paper, we introduce PRAGMA, a profile-guided AI kernel generation framework that integrates execution feedback and fine-grained hardware profiling into the reasoning loop. PRAGMA enables LLMs to identify performance bottlenecks, preserve historical best versions, and iteratively refine code quality. We evaluate PRAGMA on KernelBench, covering GPU and CPU backends. Results show that PRAGMA consistently outperforms baseline AIKG without profiling enabled and achieves 2.81$\times$ and 2.30$\times$ averaged speedups against Torch on CPU and GPU platforms, respectively. 

**Abstract (ZH)**: 高 performance 内核设计需要专家级调优和深入的硬件特性理解。大型语言模型 Recent 进展 enables 自动化内核生成，但大多数现有系统仅依赖于正确性和执行时间反馈，缺乏对低级性能瓶颈的推理能力。本文介绍了一种基于 profiling 的 AI 内核生成框架 PRAGMA，该框架将执行反馈和细粒度硬件 profiling 集成到推理循环中。PRAGMA 使 LLM 能够识别性能瓶颈、保留历史最佳版本，并逐步优化代码质量。我们在 KernelBench 上进行了评估，涵盖了 GPU 和 CPU 后端。结果表明，PRAGMA 在启用 profiling 的情况下始终优于基线 AIKG，并分别在 CPU 和 GPU 平台上实现了相对于 Torch 的 2.81$\times$ 和 2.30$\times$ 的平均加速。 

---
# TimeSense:Making Large Language Models Proficient in Time-Series Analysis 

**Title (ZH)**: TimeSense: 让大规模语言模型擅长时间序列分析 

**Authors**: Zhirui Zhang, Changhua Pei, Tianyi Gao, Zhe Xie, Yibo Hao, Zhaoyang Yu, Longlong Xu, Tong Xiao, Jing Han, Dan Pei  

**Link**: [PDF](https://arxiv.org/pdf/2511.06344)  

**Abstract**: In the time-series domain, an increasing number of works combine text with temporal data to leverage the reasoning capabilities of large language models (LLMs) for various downstream time-series understanding tasks. This enables a single model to flexibly perform tasks that previously required specialized models for each domain. However, these methods typically rely on text labels for supervision during training, biasing the model toward textual cues while potentially neglecting the full temporal features. Such a bias can lead to outputs that contradict the underlying time-series context. To address this issue, we construct the EvalTS benchmark, comprising 10 tasks across three difficulty levels, from fundamental temporal pattern recognition to complex real-world reasoning, to evaluate models under more challenging and realistic scenarios. We also propose TimeSense, a multimodal framework that makes LLMs proficient in time-series analysis by balancing textual reasoning with a preserved temporal sense. TimeSense incorporates a Temporal Sense module that reconstructs the input time-series within the model's context, ensuring that textual reasoning is grounded in the time-series dynamics. Moreover, to enhance spatial understanding of time-series data, we explicitly incorporate coordinate-based positional embeddings, which provide each time point with spatial context and enable the model to capture structural dependencies more effectively. Experimental results demonstrate that TimeSense achieves state-of-the-art performance across multiple tasks, and it particularly outperforms existing methods on complex multi-dimensional time-series reasoning tasks. 

**Abstract (ZH)**: 时间序列领域中，越来越多的研究将文本与时间数据相结合，利用大型语言模型（LLMs）的推理能力来完成各种下游时间序列理解任务。这种方法使得单一模型能够灵活地执行之前需要为每个领域专门设计模型的任务。然而，这些方法通常依赖文本标签作为训练期间的监督信息，使模型偏向于文本线索，同时可能忽视了全部的时间特征。这种偏差可能导致输出与底层时间序列上下文相矛盾。为解决这一问题，我们构建了EvalTS基准，包括三个难度级别共计10项任务，从基本的时间模式识别到复杂的现实情境推理，以在更具挑战性和现实性的场景中评估模型。我们还提出了一种多模态框架TimeSense，通过平衡文本推理与保留的时间感知来使LLMs擅长时间序列分析。TimeSense整合了一个时间感知模块，该模块在模型上下文中重构输入的时间序列，确保文本推理基于时间序列动态。此外，为了增强对时间序列数据的空间理解，我们显式地引入了基于坐标的绝对位置嵌入，为每个时间点提供空间上下文，使模型能够更有效地捕获结构依赖关系。实验结果表明，TimeSense在多个任务中取得了最先进的性能，并且在复杂的多维时间序列推理任务中尤其优于现有方法。 

---
# CINEMAE: Leveraging Frozen Masked Autoencoders for Cross-Generator AI Image Detection 

**Title (ZH)**: CINEMAE：利用冻结的遮蔽自编码器进行跨生成器AI图像检测 

**Authors**: Minsuk Jang, Hyeonseo Jeong, Minseok Son, Changick Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.06325)  

**Abstract**: While context-based detectors have achieved strong generalization for AI-generated text by measuring distributional inconsistencies, image-based detectors still struggle with overfitting to generator-specific artifacts. We introduce CINEMAE, a novel paradigm for AIGC image detection that adapts the core principles of text detection methods to the visual domain. Our key insight is that Masked AutoEncoder (MAE), trained to reconstruct masked patches conditioned on visible context, naturally encodes semantic consistency expectations. We formalize this reconstruction process probabilistically, computing conditional Negative Log-Likelihood (NLL, p(masked | visible)) to quantify local semantic anomalies. By aggregating these patch-level statistics with global MAE features through learned fusion, CINEMAE achieves strong cross-generator generalization. Trained exclusively on Stable Diffusion v1.4, our method achieves over 95% accuracy on all eight unseen generators in the GenImage benchmark, substantially outperforming state-of-the-art detectors. This demonstrates that context-conditional reconstruction uncertainty provides a robust, transferable signal for AIGC detection. 

**Abstract (ZH)**: 基于上下文的检测器通过衡量分布不一致性已在AI生成文本领域实现了强大的泛化能力，而基于图像的检测器仍然难以克服对生成器特定_artifacts_的过拟合问题。我们提出了CINEMA-E，一种将文本检测方法的核心原则适应视觉域的新范式。我们的关键洞察是，经过训练以在可见上下文中重建遮罩补丁的Masked AutoEncoder（MAE）自然地编码了语义一致性期望。我们将这一重建过程形式化为概率计算，通过计算条件负对数似然（NLL，p(遮罩 | 可见)）来量化局部语义异常。通过学习融合局部MAE特征与全局特征，CINEMA-E实现了强大的跨生成器泛化能力。仅在Stable Diffusion v1.4上训练，我们的方法在GenImage基准测试中的所有八个未见过的生成器上取得了超过95%的准确率，显著优于最先进的检测器。这表明，上下文条件重建不确定性为AI生成内容检测提供了一种稳健且可转移的信号。 

---
# Precision-Scalable Microscaling Datapaths with Optimized Reduction Tree for Efficient NPU Integration 

**Title (ZH)**: 精确可扩展微缩数据路径及其优化的减少树架构以实现高效NPU集成 

**Authors**: Stef Cuyckens, Xiaoling Yi, Robin Geens, Joren Dumoulin, Martin Wiesner, Chao Fang, Marian Verhelst  

**Link**: [PDF](https://arxiv.org/pdf/2511.06313)  

**Abstract**: Emerging continual learning applications necessitate next-generation neural processing unit (NPU) platforms to support both training and inference operations. The promising Microscaling (MX) standard enables narrow bit-widths for inference and large dynamic ranges for training. However, existing MX multiply-accumulate (MAC) designs face a critical trade-off: integer accumulation requires expensive conversions from narrow floating-point products, while FP32 accumulation suffers from quantization losses and costly normalization. To address these limitations, we propose a hybrid precision-scalable reduction tree for MX MACs that combines the benefits of both approaches, enabling efficient mixed-precision accumulation with controlled accuracy relaxation. Moreover, we integrate an 8x8 array of these MACs into the state-of-the-art (SotA) NPU integration platform, SNAX, to provide efficient control and data transfer to our optimized precision-scalable MX datapath. We evaluate our design both on MAC and system level and compare it to the SotA. Our integrated system achieves an energy efficiency of 657, 1438-1675, and 4065 GOPS/W, respectively, for MXINT8, MXFP8/6, and MXFP4, with a throughput of 64, 256, and 512 GOPS. 

**Abstract (ZH)**: 新兴的持续学习应用需要新一代神经处理单元（NPU）平台来支持训练和推理操作。具有前景的Microscaling（MX）标准允许窄位宽的推理和宽动态范围的训练。然而，现有的MX乘累加（MAC）设计面临一个关键权衡：整数累加需要昂贵的窄浮点乘积转换，而FP32累加则受到量化损失和昂贵归一化的困扰。为了解决这些限制，我们提出了一种混合精度可扩展的归约树，结合了两种方法的优势，实现了可控精度松弛下的高效混合精度累加。此外，我们将这种MAC的8×8阵列集成到最先进的（SotA）NPU集成平台SNAX中，以提供对优化的精度可扩展MX数据通路的有效控制和数据传输。我们在MAC和系统级别上评估了我们的设计，并将其与SotA进行了比较。我们的集成系统分别实现了每瓦657、1438-1675和4065 GOPS的能量效率，吞吐量分别为64、256和512 GOPS，对于MXINT8、MXFP8/6和MXFP4。 

---
# Kaggle Chronicles: 15 Years of Competitions, Community and Data Science Innovation 

**Title (ZH)**: Kaggle=zeros;Chronicles:15年竞赛、社区与数据科学创新 

**Authors**: Kevin Bönisch, Leandro Losaria  

**Link**: [PDF](https://arxiv.org/pdf/2511.06304)  

**Abstract**: Since 2010, Kaggle has been a platform where data scientists from around the world come together to compete, collaborate, and push the boundaries of Data Science. Over these 15 years, it has grown from a purely competition-focused site into a broader ecosystem with forums, notebooks, models, datasets, and more. With the release of the Kaggle Meta Code and Kaggle Meta Datasets, we now have a unique opportunity to explore these competitions, technologies, and real-world applications of Machine Learning and AI. And so in this study, we take a closer look at 15 years of data science on Kaggle - through metadata, shared code, community discussions, and the competitions themselves. We explore Kaggle's growth, its impact on the data science community, uncover hidden technological trends, analyze competition winners, how Kagglers approach problems in general, and more. We do this by analyzing millions of kernels and discussion threads to perform both longitudinal trend analysis and standard exploratory data analysis. Our findings show that Kaggle is a steadily growing platform with increasingly diverse use cases, and that Kagglers are quick to adapt to new trends and apply them to real-world challenges, while producing - on average - models with solid generalization capabilities. We also offer a snapshot of the platform as a whole, highlighting its history and technological evolution. Finally, this study is accompanied by a video (this https URL) and a Kaggle write-up (this https URL) for your convenience. 

**Abstract (ZH)**: 自2010年以来，Kaggle已成为全球数据科学家们聚集于此竞赛、合作并推动数据科学前沿的平台。在这15年里，它从一个纯粹以竞赛为主的站点发展成为包含论坛、笔记本、模型、数据集等更广泛生态系统的平台。伴随着Kaggle元代码和元数据集的发布，我们现在有机会更深入地探索这些竞赛、技术及其在机器学习和AI中的实际应用。因此，在这项研究中，我们对Kaggle上的15年数据科学进行更仔细的研究——通过元数据、共享代码、社区讨论和竞赛本身。我们探索了Kaggle的发展，它对数据科学社区的影响，揭露隐藏的技术趋势，分析竞赛获胜者，以及数据科学家们如何处理一般性问题等。我们通过分析数百万个内核和讨论主题，进行纵向趋势分析和标准探索性数据分析来实现这一点。我们的研究发现表明，Kaggle是一个稳步增长的平台，具有越来越多样化的应用场景，并且数据科学家们能够迅速适应新的趋势，并将其应用于实际困难，同时他们开发的模型通常具备良好的泛化能力。此外，我们还提供了一个平台的整体快照，突显其历史和技术演变。最后，这项研究附带有一个视频（请点击此处）和一份Kaggle社团文章（请点击此处），供您参考。 

---
# Decomate: Leveraging Generative Models for Co-Creative SVG Animation 

**Title (ZH)**: Decomate: 利用生成模型实现协同创作的SVG动画 

**Authors**: Jihyeon Park, Jiyoon Myung, Seone Shin, Jungki Son, Joohyung Han  

**Link**: [PDF](https://arxiv.org/pdf/2511.06297)  

**Abstract**: Designers often encounter friction when animating static SVG graphics, especially when the visual structure does not match the desired level of motion detail. Existing tools typically depend on predefined groupings or require technical expertise, which limits designers' ability to experiment and iterate independently. We present Decomate, a system that enables intuitive SVG animation through natural language. Decomate leverages a multimodal large language model to restructure raw SVGs into semantically meaningful, animation-ready components. Designers can then specify motions for each component via text prompts, after which the system generates corresponding HTML/CSS/JS animations. By supporting iterative refinement through natural language interaction, Decomate integrates generative AI into creative workflows, allowing animation outcomes to be directly shaped by user intent. 

**Abstract (ZH)**: 设计师在动画SVG静态图形时往往遇到摩擦，特别是在视觉结构不匹配期望的运动细节层次时。现有工具通常依赖预定义的分组或需要技术专长，这限制了设计师独立实验和迭代的能力。我们介绍了Decomate系统，该系统通过自然语言实现了直观的SVG动画。Decomate利用多模态大型语言模型将原始SVG重新组织为语义有意义、可用于动画的组件。设计师然后可以通过文本提示为每个组件指定运动，之后系统生成相应的HTML/CSS/JS动画。通过支持通过自然语言交互进行迭代细化，Decomate将生成式AI整合到创意工作流程中，使动画结果能够直接反映用户意图。 

---
# Transolver is a Linear Transformer: Revisiting Physics-Attention through the Lens of Linear Attention 

**Title (ZH)**: Transolver是一种线性 transformer：通过线性注意力视角 revisit 物理注意力 

**Authors**: Wenjie Hu, Sidun Liu, Peng Qiao, Zhenglun Sun, Yong Dou  

**Link**: [PDF](https://arxiv.org/pdf/2511.06294)  

**Abstract**: Recent advances in Transformer-based Neural Operators have enabled significant progress in data-driven solvers for Partial Differential Equations (PDEs). Most current research has focused on reducing the quadratic complexity of attention to address the resulting low training and inference efficiency. Among these works, Transolver stands out as a representative method that introduces Physics-Attention to reduce computational costs. Physics-Attention projects grid points into slices for slice attention, then maps them back through deslicing. However, we observe that Physics-Attention can be reformulated as a special case of linear attention, and that the slice attention may even hurt the model performance. Based on these observations, we argue that its effectiveness primarily arises from the slice and deslice operations rather than interactions between slices. Building on this insight, we propose a two-step transformation to redesign Physics-Attention into a canonical linear attention, which we call Linear Attention Neural Operator (LinearNO). Our method achieves state-of-the-art performance on six standard PDE benchmarks, while reducing the number of parameters by an average of 40.0% and computational cost by 36.2%. Additionally, it delivers superior performance on two challenging, industrial-level datasets: AirfRANS and Shape-Net Car. 

**Abstract (ZH)**: 基于Transformer的神经运算子 Recent进展：Physics-Attention的有效性重新审视及Linear Attention Neural Operator (LinearNO)的设计 

---
# Exploiting Inter-Session Information with Frequency-enhanced Dual-Path Networks for Sequential Recommendation 

**Title (ZH)**: 利用频率增强的双路径网络利用会话间信息进行序列推荐 

**Authors**: Peng He, Yanglei Gan, Tingting Dai, Run Lin, Xuexin Li, Yao Liu, Qiao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06285)  

**Abstract**: Sequential recommendation (SR) aims to predict a user's next item preference by modeling historical interaction sequences. Recent advances often integrate frequency-domain modules to compensate for self-attention's low-pass nature by restoring the high-frequency signals critical for personalized recommendations. Nevertheless, existing frequency-aware solutions process each session in isolation and optimize exclusively with time-domain objectives. Consequently, they overlook cross-session spectral dependencies and fail to enforce alignment between predicted and actual spectral signatures, leaving valuable frequency information under-exploited. To this end, we propose FreqRec, a Frequency-Enhanced Dual-Path Network for sequential Recommendation that jointly captures inter-session and intra-session behaviors via a learnable Frequency-domain Multi-layer Perceptrons. Moreover, FreqRec is optimized under a composite objective that combines cross entropy with a frequency-domain consistency loss, explicitly aligning predicted and true spectral signatures. Extensive experiments on three benchmarks show that FreqRec surpasses strong baselines and remains robust under data sparsity and noisy-log conditions. 

**Abstract (ZH)**: 频域增强双路径网络用于序列推荐（FreqRec） 

---
# COTN: A Chaotic Oscillatory Transformer Network for Complex Volatile Systems under Extreme Conditions 

**Title (ZH)**: COTN：在极端条件下的混沌振荡变压器网络用于复杂易变系统 

**Authors**: Boyan Tang, Yilong Zeng, Xuanhao Ren, Peng Xiao, Yuhan Zhao, Raymond Lee, Jianghua Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06273)  

**Abstract**: Accurate prediction of financial and electricity markets, especially under extreme conditions, remains a significant challenge due to their intrinsic nonlinearity, rapid fluctuations, and chaotic patterns. To address these limitations, we propose the Chaotic Oscillatory Transformer Network (COTN). COTN innovatively combines a Transformer architecture with a novel Lee Oscillator activation function, processed through Max-over-Time pooling and a lambda-gating mechanism. This design is specifically tailored to effectively capture chaotic dynamics and improve responsiveness during periods of heightened volatility, where conventional activation functions (e.g., ReLU, GELU) tend to saturate. Furthermore, COTN incorporates an Autoencoder Self-Regressive (ASR) module to detect and isolate abnormal market patterns, such as sudden price spikes or crashes, thereby preventing corruption of the core prediction process and enhancing robustness. Extensive experiments across electricity spot markets and financial markets demonstrate the practical applicability and resilience of COTN. Our approach outperforms state-of-the-art deep learning models like Informer by up to 17% and traditional statistical methods like GARCH by as much as 40%. These results underscore COTN's effectiveness in navigating real-world market uncertainty and complexity, offering a powerful tool for forecasting highly volatile systems under duress. 

**Abstract (ZH)**: 混沌振荡变压器网络：准确预测金融和电力市场，尤其是极端条件下的动态变化 

---
# LaneDiffusion: Improving Centerline Graph Learning via Prior Injected BEV Feature Generation 

**Title (ZH)**: LaneDiffusion: 通过先验注入BEV特征生成提高中心线图学习 

**Authors**: Zijie Wang, Weiming Zhang, Wei Zhang, Xiao Tan, Hongxing Liu, Yaowei Wang, Guanbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06272)  

**Abstract**: Centerline graphs, crucial for path planning in autonomous driving, are traditionally learned using deterministic methods. However, these methods often lack spatial reasoning and struggle with occluded or invisible centerlines. Generative approaches, despite their potential, remain underexplored in this domain. We introduce LaneDiffusion, a novel generative paradigm for centerline graph learning. LaneDiffusion innovatively employs diffusion models to generate lane centerline priors at the Bird's Eye View (BEV) feature level, instead of directly predicting vectorized centerlines. Our method integrates a Lane Prior Injection Module (LPIM) and a Lane Prior Diffusion Module (LPDM) to effectively construct diffusion targets and manage the diffusion process. Furthermore, vectorized centerlines and topologies are then decoded from these prior-injected BEV features. Extensive evaluations on the nuScenes and Argoverse2 datasets demonstrate that LaneDiffusion significantly outperforms existing methods, achieving improvements of 4.2%, 4.6%, 4.7%, 6.4% and 1.8% on fine-grained point-level metrics (GEO F1, TOPO F1, JTOPO F1, APLS and SDA) and 2.3%, 6.4%, 6.8% and 2.1% on segment-level metrics (IoU, mAP_cf, DET_l and TOP_ll). These results establish state-of-the-art performance in centerline graph learning, offering new insights into generative models for this task. 

**Abstract (ZH)**: LaneDiffusion：Bird's Eye View特征级别的扩散模型用于车道中心线图的学习 

---
# LLM-Guided Reinforcement Learning with Representative Agents for Traffic Modeling 

**Title (ZH)**: 基于代表性代理的LLM指导强化学习交通建模 

**Authors**: Hanlin Sun, Jiayang Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06260)  

**Abstract**: Large language models (LLMs) are increasingly used as behavioral proxies for self-interested travelers in agent-based traffic models. Although more flexible and generalizable than conventional models, the practical use of these approaches remains limited by scalability due to the cost of calling one LLM for every traveler. Moreover, it has been found that LLM agents often make opaque choices and produce unstable day-to-day dynamics. To address these challenges, we propose to model each homogeneous traveler group facing the same decision context with a single representative LLM agent who behaves like the population's average, maintaining and updating a mixed strategy over routes that coincides with the group's aggregate flow proportions. Each day, the LLM reviews the travel experience and flags routes with positive reinforcement that they hope to use more often, and an interpretable update rule then converts this judgment into strategy adjustments using a tunable (progressively decaying) step size. The representative-agent design improves scalability, while the separation of reasoning from updating clarifies the decision logic while stabilizing learning. In classic traffic assignment settings, we find that the proposed approach converges rapidly to the user equilibrium. In richer settings with income heterogeneity, multi-criteria costs, and multi-modal choices, the generated dynamics remain stable and interpretable, reproducing plausible behavioral patterns well-documented in psychology and economics, for example, the decoy effect in toll versus non-toll road selection, and higher willingness-to-pay for convenience among higher-income travelers when choosing between driving, transit, and park-and-ride options. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在基于代理的交通模型中被越来越多地用作自利旅行者的行为代理。为了解决这些挑战，我们提出用单个代表性的LLM代理来模拟在相同决策环境下面对相同旅行者的同质群体，该代理的行为类似于群体的平均行为，并且维护和更新与群体总体流量比例相符的混合策略。每天，LLM会回顾旅行体验并标记出那些希望更频繁使用的、具有正强化效果的路径，并通过可调（逐渐衰减）的学习步长将这种判断转换成策略调整。这种代理设计提高了模型的可扩展性，而将推理与更新分离则澄清了决策逻辑并稳定了学习过程。在经典的交通分配设置中，我们发现所提出的方法能够迅速收敛到用户均衡。在存在收入差异、多准则成本和多模式选择的更复杂设置中，生成的动力学保持稳定且可解释，能够再现心理学和经济学中广泛记录的合理行为模式，例如，过道效应在选择收费道路和非收费道路之间的选择中，以及高收入旅行者在选择驾驶、公共交通和停车换乘选项时对便利性的更高支付意愿。 

---
# Breaking the Modality Barrier: Generative Modeling for Accurate Molecule Retrieval from Mass Spectra 

**Title (ZH)**: 突破模态障碍：基于生成建模的准确质量谱分子检索 

**Authors**: Yiwen Zhang, Keyan Ding, Yihang Wu, Xiang Zhuang, Yi Yang, Qiang Zhang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.06259)  

**Abstract**: Retrieving molecular structures from tandem mass spectra is a crucial step in rapid compound identification. Existing retrieval methods, such as traditional mass spectral library matching, suffer from limited spectral library coverage, while recent cross-modal representation learning frameworks often encounter modality misalignment, resulting in suboptimal retrieval accuracy and generalization. To address these limitations, we propose GLMR, a Generative Language Model-based Retrieval framework that mitigates the cross-modal misalignment through a two-stage process. In the pre-retrieval stage, a contrastive learning-based model identifies top candidate molecules as contextual priors for the input mass spectrum. In the generative retrieval stage, these candidate molecules are integrated with the input mass spectrum to guide a generative model in producing refined molecular structures, which are then used to re-rank the candidates based on molecular similarity. Experiments on both MassSpecGym and the proposed MassRET-20k dataset demonstrate that GLMR significantly outperforms existing methods, achieving over 40% improvement in top-1 accuracy and exhibiting strong generalizability. 

**Abstract (ZH)**: 基于生成语言模型的跨模态检索框架GLMR在从串联质谱检索分子结构中取得显著效果 

---
# MrCoM: A Meta-Regularized World-Model Generalizing Across Multi-Scenarios 

**Title (ZH)**: MrCoM: 一种针对多场景进行泛化的元正则化世界模型 

**Authors**: Xuantang Xiong, Ni Mu, Runpeng Xie, Senhao Yang, Yaqing Wang, Lexiang Wang, Yao Luan, Siyuan Li, Shuang Xu, Yiqin Yang, Bo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06252)  

**Abstract**: Model-based reinforcement learning (MBRL) is a crucial approach to enhance the generalization capabilities and improve the sample efficiency of RL algorithms. However, current MBRL methods focus primarily on building world models for single tasks and rarely address generalization across different scenarios. Building on the insight that dynamics within the same simulation engine share inherent properties, we attempt to construct a unified world model capable of generalizing across different scenarios, named Meta-Regularized Contextual World-Model (MrCoM). This method first decomposes the latent state space into various components based on the dynamic characteristics, thereby enhancing the accuracy of world-model prediction. Further, MrCoM adopts meta-state regularization to extract unified representation of scenario-relevant information, and meta-value regularization to align world-model optimization with policy learning across diverse scenario objectives. We theoretically analyze the generalization error upper bound of MrCoM in multi-scenario settings. We systematically evaluate our algorithm's generalization ability across diverse scenarios, demonstrating significantly better performance than previous state-of-the-art methods. 

**Abstract (ZH)**: 基于模型的强化学习（MBRL）是提升RL算法泛化能力和提高样本效率的关键方法。然而，当前的MBRL方法主要集中在构建针对单一任务的世界模型，很少解决不同场景间的泛化问题。基于同一仿真引擎内部动力学共享内在性质的洞察，我们尝试构建一个能够跨不同场景泛化的一致性世界模型，名为Meta-正则化上下文世界模型（MrCoM）。该方法首先基于动力学特性将潜状态空间分解成多个组成部分，从而提高世界模型预测的准确性。进一步地，MrCoM采用元状态正则化提取与场景相关的统一表示，并采用元价值正则化使世界模型优化与多样化场景目标下的策略学习保持一致。我们在多场景设置中理论分析了MrCoM的泛化误差上界。我们系统地评估了算法在不同场景中的泛化能力，显示出显著优于先前最先进的方法的性能。 

---
# WebVIA: A Web-based Vision-Language Agentic Framework for Interactive and Verifiable UI-to-Code Generation 

**Title (ZH)**: WebVIA：基于Web的视觉-语言代理框架，用于可交互和可验证的UI到代码生成 

**Authors**: Mingde Xu, Zhen Yang, Wenyi Hong, Lihang Pan, Xinyue Fan, Yan Wang, Xiaotao Gu, Bin Xu, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06251)  

**Abstract**: User interface (UI) development requires translating design mockups into functional code, a process that remains repetitive and labor-intensive. While recent Vision-Language Models (VLMs) automate UI-to-Code generation, they generate only static HTML/CSS/JavaScript layouts lacking interactivity. To address this, we propose WebVIA, the first agentic framework for interactive UI-to-Code generation and validation. The framework comprises three components: 1) an exploration agent to capture multi-state UI screenshots; 2) a UI2Code model that generates executable interactive code; 3) a validation module that verifies the interactivity. Experiments demonstrate that WebVIA-Agent achieves more stable and accurate UI exploration than general-purpose agents (e.g., Gemini-2.5-Pro). In addition, our fine-tuned WebVIA-UI2Code models exhibit substantial improvements in generating executable and interactive HTML/CSS/JavaScript code, outperforming their base counterparts across both interactive and static UI2Code benchmarks. Our code and models are available at \href{this https URL}{\texttt{this https URL}}. 

**Abstract (ZH)**: 用户界面（UI）开发要求将设计原型转换为功能性代码，这是一个仍然具有重复性和劳动密集性的过程。虽然最近的视觉-语言模型（VLMs）能够自动化UI-to-Code生成，但它们仅生成静态的HTML/CSS/JavaScript布局，缺乏互动性。为了解决这一问题，我们提出了WebVIA，这是首个用于生成和验证互动UI-to-Code的代理框架。该框架包括三个组件：1）探索代理，用于捕捉多状态UI截图；2）UI2Code模型，生成可执行的互动代码；3）验证模块，用于验证互动性。实验表明，WebVIA-Agent在UI探索的稳定性和准确性方面优于通用代理（如Gemini-2.5-Pro）。此外，我们微调的WebVIA-UI2Code模型在生成可执行和互动的HTML/CSS/JavaScript代码方面表现出了显著的改进，在互动和静态UI-to-Code基准测试中均优于基线模型。我们的代码和模型可通过\href{this https URL}{\texttt{this https URL}}获取。 

---
# Constraint-Informed Active Learning for End-to-End ACOPF Optimization Proxies 

**Title (ZH)**: 基于约束导向的主动学习的端到端ACOPF优化代理 

**Authors**: Miao Li, Michael Klamkin, Pascal Van Hentenryck, Wenting Li, Russell Bent  

**Link**: [PDF](https://arxiv.org/pdf/2511.06248)  

**Abstract**: This paper studies optimization proxies, machine learning (ML) models trained to efficiently predict optimal solutions for AC Optimal Power Flow (ACOPF) problems. While promising, optimization proxy performance heavily depends on training data quality. To address this limitation, this paper introduces a novel active sampling framework for ACOPF optimization proxies designed to generate realistic and diverse training data. The framework actively explores varied, flexible problem specifications reflecting plausible operational realities. More importantly, the approach uses optimization-specific quantities (active constraint sets) that better capture the salient features of an ACOPF that lead to the optimal solution. Numerical results show superior generalization over existing sampling methods with an equivalent training budget, significantly advancing the state-of-practice for trustworthy ACOPF optimization proxies. 

**Abstract (ZH)**: 本文研究了优化代理模型，这些模型是训练用于高效预测交流最优功率流（ACOPF）问题最优解的机器学习（ML）模型。尽管前景广阔，但优化代理的性能高度依赖于训练数据的质量。为解决这一局限性，本文引入了一种新型的主动采样框架，用于生成ACOPF优化代理的现实且多样的训练数据。该框架积极探索多样、灵活的问题规范，反映可能的操作现实。更重要的是，该方法使用特定于优化的量（活动约束集），更好地捕捉导致最优解的ACOPF的关键特征。数值结果表明，在相同的训练预算下，该方法的泛化能力优于现有采样方法，显著推进了可信的ACOPF优化代理的技术实践。 

---
# Affordance-Guided Coarse-to-Fine Exploration for Base Placement in Open-Vocabulary Mobile Manipulation 

**Title (ZH)**: 基于功能引导的粗细探索方法用于开放词汇-Mobile manipulation场景下的基座放置 

**Authors**: Tzu-Jung Lin, Jia-Fong Yeh, Hung-Ting Su, Chung-Yi Lin, Yi-Ting Chen, Winston H. Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06240)  

**Abstract**: In open-vocabulary mobile manipulation (OVMM), task success often hinges on the selection of an appropriate base placement for the robot. Existing approaches typically navigate to proximity-based regions without considering affordances, resulting in frequent manipulation failures. We propose Affordance-Guided Coarse-to-Fine Exploration, a zero-shot framework for base placement that integrates semantic understanding from vision-language models (VLMs) with geometric feasibility through an iterative optimization process. Our method constructs cross-modal representations, namely Affordance RGB and Obstacle Map+, to align semantics with spatial context. This enables reasoning that extends beyond the egocentric limitations of RGB perception. To ensure interaction is guided by task-relevant affordances, we leverage coarse semantic priors from VLMs to guide the search toward task-relevant regions and refine placements with geometric constraints, thereby reducing the risk of convergence to local optima. Evaluated on five diverse open-vocabulary mobile manipulation tasks, our system achieves an 85% success rate, significantly outperforming classical geometric planners and VLM-based methods. This demonstrates the promise of affordance-aware and multimodal reasoning for generalizable, instruction-conditioned planning in OVMM. 

**Abstract (ZH)**: 面向开放词汇Mobile manipulation的 affordance引导粗细粒度探索方法 

---
# Mixtures of SubExperts for Large Language Continual Learning 

**Title (ZH)**: 大型语言连续学习的子专家混合模型 

**Authors**: Haeyong Kang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06237)  

**Abstract**: Adapting Large Language Models (LLMs) to a continuous stream of tasks is a critical yet challenging endeavor. While Parameter-Efficient Fine-Tuning (PEFT) methods have become a standard for this, they face a fundamental dilemma in continual learning. Reusing a single set of PEFT parameters for new tasks often leads to catastrophic forgetting of prior knowledge. Conversely, allocating distinct parameters for each task prevents forgetting but results in a linear growth of the model's size and fails to facilitate knowledge transfer between related tasks. To overcome these limitations, we propose a novel adaptive PEFT method referred to as \textit{Mixtures of SubExperts (MoSEs)}, a novel continual learning framework designed for minimal forgetting and efficient scalability. MoSEs integrate a sparse Mixture of SubExperts into the transformer layers, governed by a task-specific routing mechanism. This architecture allows the model to isolate and protect knowledge within dedicated SubExperts, thereby minimizing parameter interference and catastrophic forgetting. Crucially, the router can adaptively select and combine previously learned sparse parameters for new tasks, enabling effective knowledge transfer while ensuring that the model's capacity grows sublinearly. We evaluate MoSEs on the comprehensive TRACE benchmark datasets. Our experiments demonstrate that MoSEs significantly outperform conventional continual learning approaches in both knowledge retention and scalability to new tasks, achieving state-of-the-art performance with substantial memory and computational savings. 

**Abstract (ZH)**: 适配大规模语言模型（LLMs）以应对连续的任务流是一个关键但具有挑战性的任务。虽然参数高效微调（PEFT）方法已成为这一领域的标准，但在持续学习中它们面临根本性的难题。使用单一集合的PEFT参数处理新任务往往会引发先前知识的灾难性遗忘。相反，为每个任务分配独特的参数可以防止遗忘，但会导致模型尺寸的线性增长，并且无法促进相关任务之间的知识迁移。为克服这些限制，我们提出了一种新颖的自适应PEFT方法，称为“Sub Experts 混合模型（MoSEs）”，这是一种旨在实现最小遗忘和高效扩展的新型持续学习框架。MoSEs通过任务特定的路由机制将稀疏的Sub Experts混合模型集成到变换器层中，使模型能够隔离并保护专用于特定任务的知识，从而最小化参数间的干扰和灾难性遗忘。关键的是，路由机制能够自适应地选择和组合先前学习到的稀疏参数来处理新任务，从而实现有效的知识迁移并确保模型容量的增长呈次线性。我们在全面的TRACE基准数据集上评估了MoSEs。实验结果表明，MoSEs在知识保留和处理新任务的能力上显著优于传统的持续学习方法，并且在性能达到最先进的同时实现了显著的内存和计算成本节约。 

---
# Analyzing and Mitigating Negation Artifacts using Data Augmentation for Improving ELECTRA-Small Model Accuracy 

**Title (ZH)**: 使用数据增强分析并减轻否定标记 artifacts 以提高 ELECTRA-Small 模型准确性 

**Authors**: Mojtaba Noghabaei  

**Link**: [PDF](https://arxiv.org/pdf/2511.06234)  

**Abstract**: Pre-trained models for natural language inference (NLI) often achieve high performance on benchmark datasets by using spurious correlations, or dataset artifacts, rather than understanding language touches such as negation. In this project, we investigate the performance of an ELECTRA-small model fine-tuned on the Stanford Natural Language Inference (SNLI) dataset, focusing on its handling of negation. Through analysis, we identify that the model struggles with correctly classifying examples containing negation. To address this, we augment the training data with contrast sets and adversarial examples emphasizing negation. Our results demonstrate that this targeted data augmentation improves the model's accuracy on negation-containing examples without adversely affecting overall performance, therefore mitigating the identified dataset artifact. 

**Abstract (ZH)**: 预训练模型在自然语言推理（NLI）中的性能往往依赖于虚假相关性或数据集artifact，而非理解如否定等语言特性。在本项目中，我们研究了在斯坦福自然语言推理（SNLI）数据集上微调的ELECTRA-small模型对否定的处理能力。通过分析，我们发现该模型在正确分类包含否定的例子时存在困难。为解决这一问题，我们增加了强调否定的对比集和对抗样本作为训练数据。研究结果表明，这种针对性的数据增强提高了模型在处理包含否定的例子时的准确率，而不会负面影响整体性能，从而减轻了识别出的数据集artifact。 

---
# Scaling Laws and In-Context Learning: A Unified Theoretical Framework 

**Title (ZH)**: 标度定律与上下文学习：一个统一的理论框架 

**Authors**: Sushant Mehta, Ishan Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2511.06232)  

**Abstract**: In-context learning (ICL) enables large language models to adapt to new tasks from demonstrations without parameter updates. Despite extensive empirical studies, a principled understanding of ICL emergence at scale remains more elusive. We present a unified theoretical framework connecting scaling laws to ICL emergence in transformers. Our analysis establishes that ICL performance follows power-law relationships with model depth $L$, width $d$, context length $k$, and training data $D$, with exponents determined by task structure. We show that under specific conditions, transformers implement gradient-based metalearning in their forward pass, with an effective learning rate $\eta_{\text{eff}} = \Theta(1/\sqrt{Ld})$. We demonstrate sharp phase transitions at critical scales and derive optimal depth-width allocations favoring $L^* \propto N^{2/3}$, $d^* \propto N^{1/3}$ for the fixed parameter budget $N = Ld$. Systematic experiments on synthetic tasks validate our predictions, with measured scaling exponents closely matching theory. This work provides both necessary and sufficient conditions for the emergence of ICLs and establishes fundamental computational limits on what transformers can learn in-context. 

**Abstract (ZH)**: 基于上下文学习(ICL)使大型语言模型能够无需参数更新即可从演示中适应新任务。我们提出了一种统一的理论框架，将缩放定律与Transformer中的ICL涌现联系起来。我们的分析建立了ICL性能与模型深度$L$、宽度$d$、上下文长度$k$和训练数据$D$之间的幂律关系，指数由任务结构决定。我们展示，在特定条件下，Transformer在其前向传递过程中实现了基于梯度的元学习，有效学习率$\eta_{\text{eff}} = \Theta(1/\sqrt{Ld})$。我们证明了在关键规模下存在尖锐的相变，并推导出对于固定参数预算$N = Ld$的最佳深度-宽度分配，分别为$L^* \propto N^{2/3}$和$d^* \propto N^{1/3}$。系统性的合成任务实验验证了我们的预测，测得的缩放指数与理论值紧密匹配。本文为ICL的涌现提供了必要且充分的条件，并建立了Transformer在上下文中学习的基本计算限制。 

---
# Overview of CHIP 2025 Shared Task 2: Discharge Medication Recommendation for Metabolic Diseases Based on Chinese Electronic Health Records 

**Title (ZH)**: CHIP 2025 共享任务 2：基于中文电子健康记录的代谢疾病出院药物推荐概览 

**Authors**: Juntao Li, Haobin Yuan, Ling Luo, Tengxiao Lv, Yan Jiang, Fan Wang, Ping Zhang, Huiyi Lv, Jian Wang, Yuanyuan Sun, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.06230)  

**Abstract**: Discharge medication recommendation plays a critical role in ensuring treatment continuity, preventing readmission, and improving long-term management for patients with chronic metabolic diseases. This paper present an overview of the CHIP 2025 Shared Task 2 competition, which aimed to develop state-of-the-art approaches for automatically recommending appro-priate discharge medications using real-world Chinese EHR data. For this task, we constructed CDrugRed, a high-quality dataset consisting of 5,894 de-identified hospitalization records from 3,190 patients in China. This task is challenging due to multi-label nature of medication recommendation, het-erogeneous clinical text, and patient-specific variability in treatment plans. A total of 526 teams registered, with 167 and 95 teams submitting valid results to the Phase A and Phase B leaderboards, respectively. The top-performing team achieved the highest overall performance on the final test set, with a Jaccard score of 0.5102, F1 score of 0.6267, demonstrating the potential of advanced large language model (LLM)-based ensemble systems. These re-sults highlight both the promise and remaining challenges of applying LLMs to medication recommendation in Chinese EHRs. The post-evaluation phase remains open at this https URL. 

**Abstract (ZH)**: 出院药物推荐在确保慢性代谢疾病患者治疗连续性、防止再次入院并改善长期管理中起着关键作用。本文概述了CHIP 2025 Shared Task 2竞赛，旨在利用实际的中文电子健康记录数据开发先进的自动推荐适宜出院药物的方法。为此，我们构建了CDrugRed数据集，包含来自3190名中国患者的5894份脱敏住院记录。由于药物推荐的多标签性质、临床文本的异质性和患者特定的治疗计划变化，该任务具有挑战性。共有526支队伍注册参赛，分别为167支和95支队伍提交了A阶段和B阶段的有效结果。性能最佳的队伍在最终测试集上的交集分数达到0.5102，F1分数为0.6267，显示出基于高级大型语言模型（LLM）的集成系统的潜力。这些结果突显了将LLM应用于中文EHR中的药物推荐既具有前景也面临挑战。后续评估阶段信息请参见此链接：https://this.url。 

---
# Assertion-Aware Test Code Summarization with Large Language Models 

**Title (ZH)**: 基于大型语言模型的断言意识测试代码总结 

**Authors**: Anamul Haque Mollah, Ahmed Aljohani, Hyunsook Do  

**Link**: [PDF](https://arxiv.org/pdf/2511.06227)  

**Abstract**: Unit tests often lack concise summaries that convey test intent, especially in auto-generated or poorly documented codebases. Large Language Models (LLMs) offer a promising solution, but their effectiveness depends heavily on how they are prompted. Unlike generic code summarization, test-code summarization poses distinct challenges because test methods validate expected behavior through assertions rather than im- plementing functionality. This paper presents a new benchmark of 91 real-world Java test cases paired with developer-written summaries and conducts a controlled ablation study to investigate how test code-related components-such as the method under test (MUT), assertion messages, and assertion semantics-affect the performance of LLM-generated test summaries. We evaluate four code LLMs (Codex, Codestral, DeepSeek, and Qwen-Coder) across seven prompt configurations using n-gram metrics (BLEU, ROUGE-L, METEOR), semantic similarity (BERTScore), and LLM-based evaluation. Results show that prompting with as- sertion semantics improves summary quality by an average of 0.10 points (2.3%) over full MUT context (4.45 vs. 4.35) while requiring fewer input tokens. Codex and Qwen-Coder achieve the highest alignment with human-written summaries, while DeepSeek underperforms despite high lexical overlap. The replication package is publicly available at this https URL. 5281/zenodo.17067550 

**Abstract (ZH)**: 单元测试通常缺乏简洁的摘要来传达测试意图，特别是在自动生成或文档缺失的代码库中。大规模语言模型（LLMs）提供了一种有前景的解决方案，但其效果很大程度上依赖于如何进行提示。不同于通用代码摘要，测试代码摘要面临独特的挑战，因为测试方法通过断言验证预期行为，而不是实现功能。本文提出了一个包含91个真实世界的Java测试案例及其开发者撰写的摘要的新基准，并进行了一项受控删减研究，以探索测试代码相关组件（如被测试方法、断言消息和断言语义）如何影响LLM生成的测试摘要的性能。我们使用n-克gram指标（BLEU、ROUGE-L、METEOR）、语义相似度（BERTScore）和基于LLM的评估对四种代码LLM（Codex、Codestral、DeepSeek和Qwen-Coder）在七种提示配置下的表现进行了评估。结果显示，使用断言语义提示提高了摘要质量，平均得分提高了0.10分（2.3%），同时所需的输入令牌更少（4.45 vs. 4.35）。Codex和Qwen-Coder与人工撰写的摘要最一致，尽管DeepSeek的词汇重叠率高，但表现不佳。该复制包可在以下链接获取：https://5281/zenodo.17067550。 

---
# Explicit Knowledge-Guided In-Context Learning for Early Detection of Alzheimer's Disease 

**Title (ZH)**: 显性知识引导的上下文学习方法在阿尔茨海默病早期检测中的应用 

**Authors**: Puzhen Su, Yongzhu Miao, Chunxi Guo, Jintao Tang, Shasha Li, Ting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06215)  

**Abstract**: Detecting Alzheimer's Disease (AD) from narrative transcripts remains a challenging task for large language models (LLMs), particularly under out-of-distribution (OOD) and data-scarce conditions. While in-context learning (ICL) provides a parameter-efficient alternative to fine-tuning, existing ICL approaches often suffer from task recognition failure, suboptimal demonstration selection, and misalignment between label words and task objectives, issues that are amplified in clinical domains like AD detection. We propose Explicit Knowledge In-Context Learners (EK-ICL), a novel framework that integrates structured explicit knowledge to enhance reasoning stability and task alignment in ICL. EK-ICL incorporates three knowledge components: confidence scores derived from small language models (SLMs) to ground predictions in task-relevant patterns, parsing feature scores to capture structural differences and improve demo selection, and label word replacement to resolve semantic misalignment with LLM priors. In addition, EK-ICL employs a parsing-based retrieval strategy and ensemble prediction to mitigate the effects of semantic homogeneity in AD transcripts. Extensive experiments across three AD datasets demonstrate that EK-ICL significantly outperforms state-of-the-art fine-tuning and ICL baselines. Further analysis reveals that ICL performance in AD detection is highly sensitive to the alignment of label semantics and task-specific context, underscoring the importance of explicit knowledge in clinical reasoning under low-resource conditions. 

**Abstract (ZH)**: 从叙述转录中检测阿尔茨海默病（AD）：在分布外和数据稀缺条件下，大型语言模型（LLMs）任务仍具挑战性——明确知识融入上下文学习（EK-ICL） 

---
# RAG-targeted Adversarial Attack on LLM-based Threat Detection and Mitigation Framework 

**Title (ZH)**: 面向RAG的目标对抗攻击对基于大规模语言模型的威胁检测与缓解框架的影响 

**Authors**: Seif Ikbarieh, Kshitiz Aryal, Maanak Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2511.06212)  

**Abstract**: The rapid expansion of the Internet of Things (IoT) is reshaping communication and operational practices across industries, but it also broadens the attack surface and increases susceptibility to security breaches. Artificial Intelligence has become a valuable solution in securing IoT networks, with Large Language Models (LLMs) enabling automated attack behavior analysis and mitigation suggestion in Network Intrusion Detection Systems (NIDS). Despite advancements, the use of LLMs in such systems further expands the attack surface, putting entire networks at risk by introducing vulnerabilities such as prompt injection and data poisoning. In this work, we attack an LLM-based IoT attack analysis and mitigation framework to test its adversarial robustness. We construct an attack description dataset and use it in a targeted data poisoning attack that applies word-level, meaning-preserving perturbations to corrupt the Retrieval-Augmented Generation (RAG) knowledge base of the framework. We then compare pre-attack and post-attack mitigation responses from the target model, ChatGPT-5 Thinking, to measure the impact of the attack on model performance, using an established evaluation rubric designed for human experts and judge LLMs. Our results show that small perturbations degrade LLM performance by weakening the linkage between observed network traffic features and attack behavior, and by reducing the specificity and practicality of recommended mitigations for resource-constrained devices. 

**Abstract (ZH)**: 基于LLM的物联网攻击分析与缓解框架的 adversarial robustness 攻击研究 

---
# Resilience Inference for Supply Chains with Hypergraph Neural Network 

**Title (ZH)**: 基于超图神经网络的供应链韧性推断 

**Authors**: Zetian Shen, Hongjun Wang, Jiyuan Chen, Xuan Song  

**Link**: [PDF](https://arxiv.org/pdf/2511.06208)  

**Abstract**: Supply chains are integral to global economic stability, yet disruptions can swiftly propagate through interconnected networks, resulting in substantial economic impacts. Accurate and timely inference of supply chain resilience the capability to maintain core functions during disruptions is crucial for proactive risk mitigation and robust network design. However, existing approaches lack effective mechanisms to infer supply chain resilience without explicit system dynamics and struggle to represent the higher-order, multi-entity dependencies inherent in supply chain networks. These limitations motivate the definition of a novel problem and the development of targeted modeling solutions. To address these challenges, we formalize a novel problem: Supply Chain Resilience Inference (SCRI), defined as predicting supply chain resilience using hypergraph topology and observed inventory trajectories without explicit dynamic equations. To solve this problem, we propose the Supply Chain Resilience Inference Hypergraph Network (SC-RIHN), a novel hypergraph-based model leveraging set-based encoding and hypergraph message passing to capture multi-party firm-product interactions. Comprehensive experiments demonstrate that SC-RIHN significantly outperforms traditional MLP, representative graph neural network variants, and ResInf baselines across synthetic benchmarks, underscoring its potential for practical, early-warning risk assessment in complex supply chain systems. 

**Abstract (ZH)**: 供应链韧性推断（SCRI）基于超图拓扑和观测到的库存轨迹的预测问题及其解决方案 

---
# Enhancing Adversarial Robustness of IoT Intrusion Detection via SHAP-Based Attribution Fingerprinting 

**Title (ZH)**: 基于SHAP基解释指纹技术提升物联网入侵检测的鲁棒对抗性 

**Authors**: Dilli Prasad Sharma, Liang Xue, Xiaowei Sun, Xiaodong Lin, Pulei Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2511.06197)  

**Abstract**: The rapid proliferation of Internet of Things (IoT) devices has transformed numerous industries by enabling seamless connectivity and data-driven automation. However, this expansion has also exposed IoT networks to increasingly sophisticated security threats, including adversarial attacks targeting artificial intelligence (AI) and machine learning (ML)-based intrusion detection systems (IDS) to deliberately evade detection, induce misclassification, and systematically undermine the reliability and integrity of security defenses. To address these challenges, we propose a novel adversarial detection model that enhances the robustness of IoT IDS against adversarial attacks through SHapley Additive exPlanations (SHAP)-based fingerprinting. Using SHAP's DeepExplainer, we extract attribution fingerprints from network traffic features, enabling the IDS to reliably distinguish between clean and adversarially perturbed inputs. By capturing subtle attribution patterns, the model becomes more resilient to evasion attempts and adversarial manipulations. We evaluated the model on a standard IoT benchmark dataset, where it significantly outperformed a state-of-the-art method in detecting adversarial attacks. In addition to enhanced robustness, this approach improves model transparency and interpretability, thereby increasing trust in the IDS through explainable AI. 

**Abstract (ZH)**: 物联网设备的迅速普及通过enable无缝连接和数据驱动的自动化转型了众多行业，但这也使物联网网络受到了日益复杂的安全威胁，包括针对基于人工智能和机器学习的入侵检测系统（IDS）的对抗性攻击，这些攻击旨在故意逃避检测、引发误分类，并系统地削弱安全防御的可靠性和完整性。为应对这些挑战，我们提出了一种新型的对抗性检测模型，该模型通过基于SHapley加性解释（SHAP）的指纹技术增强了物联网IDS对对抗性攻击的鲁棒性。使用SHAP的DeepExplainer，我们从网络流量特征中提取归因指纹，使IDS能够可靠地区分干净和受到对抗性扰动的输入。通过捕捉细微的归因模式，该模型变得更加抵御逃避尝试和对抗性操纵。我们在一个标准的物联网基准数据集上评估了该模型，结果显示它在检测对抗性攻击方面明显优于最先进的方法。除了增强的鲁棒性，这种方法还提高了模型的透明性和可解释性，从而通过可解释人工智能增强了对IDS的信任。 

---
# AI as intermediary in modern-day ritual: An immersive, interactive production of the roller disco musical Xanadu at UCLA 

**Title (ZH)**: AI作为现代仪式的中介：在UCLA沉浸式互动音乐剧《劳拉迪斯欧杜》中的应用 

**Authors**: Mira Winick, Naisha Agarwal, Chiheb Boussema, Ingrid Lee, Camilo Vargas, Jeff Burke  

**Link**: [PDF](https://arxiv.org/pdf/2511.06195)  

**Abstract**: Interfaces for contemporary large language, generative media, and perception AI models are often engineered for single user interaction. We investigate ritual as a design scaffold for developing collaborative, multi-user human-AI engagement. We consider the specific case of an immersive staging of the musical Xanadu performed at UCLA in Spring 2025. During a two-week run, over five hundred audience members contributed sketches and jazzercise moves that vision language models translated to virtual scenery elements and from choreographic prompts. This paper discusses four facets of interaction-as-ritual within the show: audience input as offerings that AI transforms into components of the ritual; performers as ritual guides, demonstrating how to interact with technology and sorting audience members into cohorts; AI systems as instruments "played" by the humans, in which sensing, generative components, and stagecraft create systems that can be mastered over time; and reciprocity of interaction, in which the show's AI machinery guides human behavior as well as being guided by humans, completing a human-AI feedback loop that visibly reshapes the virtual world. Ritual served as a frame for integrating linear narrative, character identity, music and interaction. The production explored how AI systems can support group creativity and play, addressing a critical gap in prevailing single user AI design paradigms. 

**Abstract (ZH)**: 当代大型语言模型、生成媒体和感知AI模型的界面通常针对单用户交互进行设计。我们探讨仪式作为设计框架，用于开发多用户的人机协作交互。我们考虑沉浸式音乐剧《Xanadu》在2025年春季在UCLA上演的具体案例。在两周的演出中，超过五百名观众贡献了素描和爵士体操动作，视觉语言模型将这些转化为虚拟能景元素和 choreographic 提示。本文讨论了演出中交互作为仪式的四个方面：观众输入作为祭品，AI将其转化为仪式的组成部分；表演者作为仪式向导，示范如何与技术互动，并将观众成员分类到不同的群体中；AI系统作为由人类“演奏”的乐器，在感测、生成组件和舞台技术的共同作用下，创造出能够随着时间被掌握的系统；以及交互的互惠性，在此过程中，节目的AI机制引导人类行为，并由人类引导，从而完成一种可直观重塑虚拟世界的双向人机反馈循环。仪式为整合线性叙事、人物身份、音乐和互动提供了一个框架。该制作探索了AI系统如何支持群体的创造力和互动，弥补了当前单用户AI设计范式的关键空白。 

---
# MemoriesDB: A Temporal-Semantic-Relational Database for Long-Term Agent Memory / Modeling Experience as a Graph of Temporal-Semantic Surfaces 

**Title (ZH)**: MemoriesDB：一种用于长期智能体记忆的时序语义关系数据库 / 将经验建模为时序语义表面图 

**Authors**: Joel Ward  

**Link**: [PDF](https://arxiv.org/pdf/2511.06179)  

**Abstract**: We introduce MemoriesDB, a unified data architecture designed to avoid decoherence across time, meaning, and relation in long-term computational memory. Each memory is a time-semantic-relational entity-a structure that simultaneously encodes when an event occurred, what it means, and how it connects to other events. Built initially atop PostgreSQL with pgvector extensions, MemoriesDB combines the properties of a time-series datastore, a vector database, and a graph system within a single append-only schema. Each memory is represented as a vertex uniquely labeled by its microsecond timestamp and accompanied by low- and high-dimensional normalized embeddings that capture semantic context. Directed edges between memories form labeled relations with per-edge metadata, enabling multiple contextual links between the same vertices. Together these constructs form a time-indexed stack of temporal-semantic surfaces, where edges project as directional arrows in a 1+1-dimensional similarity field, tracing the evolution of meaning through time while maintaining cross-temporal coherence. This formulation supports efficient time-bounded retrieval, hybrid semantic search, and lightweight structural reasoning in a single query path. A working prototype demonstrates scalable recall and contextual reinforcement using standard relational infrastructure, and we discuss extensions toward a columnar backend, distributed clustering, and emergent topic modeling. 

**Abstract (ZH)**: 我们介绍MemoriesDB，这是一种统一的数据架构，旨在避免长时间计算记忆中跨时间、意义和关系的 decoherence。每个记忆是一个时间语义关系实体——一种同时编码事件发生时间、意义及其与其他事件关联性的结构。MemoriesDB 初始基于 PostgreSQL 并结合了 pgvector 扩展，将时间序列数据存储、向量数据库和图形系统的特点整合到一个追加只读模式中。每个记忆表示为一个通过其微秒时间戳唯一标记的顶点，并伴随低维和高维归一化嵌入，以捕捉语义上下文。记忆之间的有向边形成带有边元数据的标记关系，使得相同的顶点之间可以有多重上下文链接。这些构造共同形成了时间索引的时间语义表面堆栈，其中边在1+1维相似性字段中表现为方向箭头，跟踪意义随时间的演变，同时保持跨时间的一致性。此框架支持在单个查询路径中高效的时间限制检索、混合语义搜索和轻量级结构推理。一个工作的原型表明，利用标准关系基础设施可以实现可扩展的召回和上下文强化，并讨论了向列式后端、分布式聚类和新兴主题建模的扩展。 

---
# LUT-LLM: Efficient Large Language Model Inference with Memory-based Computations on FPGAs 

**Title (ZH)**: LUT-LLM：基于内存计算的FPGA上高效大型语言模型推理 

**Authors**: Zifan He, Shengyu Ye, Rui Ma, Yang Wang, Jason Cong  

**Link**: [PDF](https://arxiv.org/pdf/2511.06174)  

**Abstract**: The rapid progress of large language models (LLMs) has advanced numerous applications, yet efficient single-batch inference remains vital for on-device intelligence. While FPGAs offer fine-grained data control and high energy efficiency, recent GPU optimizations have narrowed their advantage, especially under arithmetic-based computation. To overcome this, we leverage FPGAs' abundant on-chip memory to shift LLM inference from arithmetic- to memory-based computation through table lookups. We present LUT-LLM, the first FPGA accelerator enabling 1B+ LLM inference via vector-quantized memory operations. Our analysis identifies activation-weight co-quantization as the most effective scheme, supported by (1) bandwidth-aware parallel centroid search, (2) efficient 2D table lookups, and (3) a spatial-temporal hybrid design minimizing data caching. Implemented on an AMD V80 FPGA for a customized Qwen 3 1.7B model, LUT-LLM achieves 1.66x lower latency than AMD MI210 and 1.72x higher energy efficiency than NVIDIA A100, scaling to 32B models with 2.16x efficiency gain over A100. 

**Abstract (ZH)**: 基于查找表的FPGA加速器LUT-LLM：通过内存操作实现超亿参数大语言模型推理 

---
# MambaOVSR: Multiscale Fusion with Global Motion Modeling for Chinese Opera Video Super-Resolution 

**Title (ZH)**: MambaOVSR：融合多尺度全局运动建模的中国戏曲视频超分辨率技术 

**Authors**: Hua Chang, Xin Xu, Wei Liu, Wei Wang, Xin Yuan, Kui Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06172)  

**Abstract**: Chinese opera is celebrated for preserving classical art. However, early filming equipment limitations have degraded videos of last-century performances by renowned artists (e.g., low frame rates and resolution), hindering archival efforts. Although space-time video super-resolution (STVSR) has advanced significantly, applying it directly to opera videos remains challenging. The scarcity of datasets impedes the recovery of high frequency details, and existing STVSR methods lack global modeling capabilities, compromising visual quality when handling opera's characteristic large motions. To address these challenges, we pioneer a large scale Chinese Opera Video Clip (COVC) dataset and propose the Mamba-based multiscale fusion network for space-time Opera Video Super-Resolution (MambaOVSR). Specifically, MambaOVSR involves three novel components: the Global Fusion Module (GFM) for motion modeling through a multiscale alternating scanning mechanism, and the Multiscale Synergistic Mamba Module (MSMM) for alignment across different sequence lengths. Additionally, our MambaVR block resolves feature artifacts and positional information loss during alignment. Experimental results on the COVC dataset show that MambaOVSR significantly outperforms the SOTA STVSR method by an average of 1.86 dB in terms of PSNR. Dataset and Code will be publicly released. 

**Abstract (ZH)**: 中国戏曲：大规模中国戏曲视频剪辑数据集及Mamba基多尺度融合网络的空间时间超级分辨率（MambaOVSR） 

---
# LLM Attention Transplant for Transfer Learning of Tabular Data Across Disparate Domains 

**Title (ZH)**: 跨异质领域表格数据迁移学习的LLM注意力移植 

**Authors**: Ibna Kowsar, Kazi F. Akhter, Manar D. Samad  

**Link**: [PDF](https://arxiv.org/pdf/2511.06161)  

**Abstract**: Transfer learning of tabular data is non-trivial due to heterogeneity in the feature space across disparate domains. The limited success of traditional deep learning in tabular knowledge transfer can be advanced by leveraging large language models (LLMs). However, the efficacy of LLMs often stagnates for mixed data types structured in tables due to the limitations of text prompts and in-context learning. We propose a lightweight transfer learning framework that fine-tunes an LLM using source tabular data and transplants the LLM's selective $key$ and $value$ projection weights into a gated feature tokenized transformer (gFTT) built for tabular data. The gFTT model with cross-domain attention is fine-tuned using target tabular data for transfer learning, eliminating the need for shared features, LLM prompt engineering, and large-scale pretrained models. Our experiments using ten pairs of source-target data sets and 12 baselines demonstrate the superiority of the proposed LLM-attention transplant for transfer learning (LATTLE) method over traditional ML models, state-of-the-art deep tabular architectures, and transfer learning models trained on thousands to billions of tabular samples. The proposed attention transfer demonstrates an effective solution to learning relationships between data tables using an LLM in a low-resource learning environment. The source code for the proposed method is publicly available. 

**Abstract (ZH)**: 基于大语言模型的表格数据迁移学习框架：LATTLE方法 

---
# Models Got Talent: Identifying High Performing Wearable Human Activity Recognition Models Without Training 

**Title (ZH)**: Models Got Talent: 识别无需训练即可表现优异的可穿戴人体活动识别模型 

**Authors**: Richard Goldman, Varun Komperla, Thomas Ploetz, Harish Haresamudram  

**Link**: [PDF](https://arxiv.org/pdf/2511.06157)  

**Abstract**: A promising alternative to the computationally expensive Neural Architecture Search (NAS) involves the development of \textit{Zero Cost Proxies (ZCPs)}, which correlate well to trained performance, but can be computed through a single forward/backward pass on a randomly sampled batch of data. In this paper, we investigate the effectiveness of ZCPs for HAR on six benchmark datasets, and demonstrate that they discover network architectures that obtain within 5\% of performance attained by full scale training involving 1500 randomly sampled architectures. This results in substantial computational savings as high performing architectures can be discovered with minimal training. Our experiments not only introduce ZCPs to sensor-based HAR, but also demonstrate that they are robust to data noise, further showcasing their suitability for practical scenarios. 

**Abstract (ZH)**: 一种有前途的计算成本较低的神经架构搜索（NAS）的替代方法是开发零成本代理（ZCPs），它们与训练性能高度相关，但可以通过单次正向/反向传播计算得出，基于随机采样的数据批次。在本文中，我们探讨了ZCPs在六种基准数据集上进行人体活动识别（HAR）的效果，并证明它们发现的网络架构在性能上与大规模训练中1500个随机采样架构达到的性能相差不超过5%。这导致了显著的计算成本节省，高性能的架构可以通过最少的训练发现。我们的实验不仅将ZCPs引入基于传感器的人体活动识别中，还证明了它们对数据噪声的鲁棒性，进一步展示了其在实际场景中的适用性。 

---
# Large Language Models Develop Novel Social Biases Through Adaptive Exploration 

**Title (ZH)**: 大型语言模型通过适应性探索发展出新型社会偏见 

**Authors**: Addison J. Wu, Ryan Liu, Xuechunzi Bai, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2511.06148)  

**Abstract**: As large language models (LLMs) are adopted into frameworks that grant them the capacity to make real decisions, it is increasingly important to ensure that they are unbiased. In this paper, we argue that the predominant approach of simply removing existing biases from models is not enough. Using a paradigm from the psychology literature, we demonstrate that LLMs can spontaneously develop novel social biases about artificial demographic groups even when no inherent differences exist. These biases result in highly stratified task allocations, which are less fair than assignments by human participants and are exacerbated by newer and larger models. In social science, emergent biases like these have been shown to result from exploration-exploitation trade-offs, where the decision-maker explores too little, allowing early observations to strongly influence impressions about entire demographic groups. To alleviate this effect, we examine a series of interventions targeting model inputs, problem structure, and explicit steering. We find that explicitly incentivizing exploration most robustly reduces stratification, highlighting the need for better multifaceted objectives to mitigate bias. These results reveal that LLMs are not merely passive mirrors of human social biases, but can actively create new ones from experience, raising urgent questions about how these systems will shape societies over time. 

**Abstract (ZH)**: 大规模语言模型在具备做出实际决策的能力时，消除其偏见的途径已不足以确保公平。本论文通过心理学范式表明，即使不存在固有差异，语言模型也可能自发发展出对人工社会群体的新偏见，导致高度分层的任务分配，这种分配比人类参与者制定的分配更加不公平，并且还受到更大规模模型的加剧。社会科学研究表明，此类新兴偏见源于探索与利用之间的权衡，在这种权衡中，决策者探索不足，导致早期观察强烈影响整个社会群体的总体印象。为了减轻这一影响，我们研究了一系列针对模型输入、问题结构和明确引导的干预措施。研究发现，明确激励探索最有效地减少了分层现象，突显了需要更好的多方面目标来减轻偏见的重要性。这些结果揭示了语言模型不仅仅是人类社会偏见的被动镜像，它们可以从经验中主动创造新的偏见，这迫切需要探讨这些系统如何随着时间改变社会。 

---
# Referring Expressions as a Lens into Spatial Language Grounding in Vision-Language Models 

**Title (ZH)**: 引用表达作为空间语言 grounding 在多模态语言视觉模型中的一个视角 

**Authors**: Akshar Tumu, Varad Shinde, Parisa Kordjamshidi  

**Link**: [PDF](https://arxiv.org/pdf/2511.06146)  

**Abstract**: Spatial Reasoning is an important component of human cognition and is an area in which the latest Vision-language models (VLMs) show signs of difficulty. The current analysis works use image captioning tasks and visual question answering. In this work, we propose using the Referring Expression Comprehension task instead as a platform for the evaluation of spatial reasoning by VLMs. This platform provides the opportunity for a deeper analysis of spatial comprehension and grounding abilities when there is 1) ambiguity in object detection, 2) complex spatial expressions with a longer sentence structure and multiple spatial relations, and 3) expressions with negation ('not'). In our analysis, we use task-specific architectures as well as large VLMs and highlight their strengths and weaknesses in dealing with these specific situations. While all these models face challenges with the task at hand, the relative behaviors depend on the underlying models and the specific categories of spatial semantics (topological, directional, proximal, etc.). Our results highlight these challenges and behaviors and provide insight into research gaps and future directions. 

**Abstract (ZH)**: 空间推理是人类认知的重要组成部分，是最新视觉-语言模型（VLMs）表现出困难的领域。当前的研究主要使用图像 captioning 任务和视觉问答。在本项工作中，我们提议使用引用表达理解任务作为评估 VLMs 空间推理能力的平台。该平台提供了在以下情况对空间理解与定位能力进行更深入分析的机会：1）物体检测的歧义性，2）复杂的空间表达和较长的句结构以及多种空间关系，3）带有否定词（not）的表达。在我们的分析中，我们使用了任务特定的架构和大规模的 VLMs，并强调了它们在这些特定情况下的优势和不足。虽然所有这些模型在处理该任务时都面临挑战，但它们的相对行为取决于底层模型和特定的空间语义类别（拓扑性的、方向性的、近距离的等）。我们研究的结果突显了这些挑战和行为，并为未来的研究提供了一定的见解和方向。 

---
# Evaluation of retrieval-based QA on QUEST-LOFT 

**Title (ZH)**: 基于检索的问答系统在QUEST-LOFT上的评估 

**Authors**: Nathan Scales, Nathanael Schärli, Olivier Bousquet  

**Link**: [PDF](https://arxiv.org/pdf/2511.06125)  

**Abstract**: Despite the popularity of retrieval-augmented generation (RAG) as a solution for grounded QA in both academia and industry, current RAG methods struggle with questions where the necessary information is distributed across many documents or where retrieval needs to be combined with complex reasoning. Recently, the LOFT study has shown that this limitation also applies to approaches based on long-context language models, with the QUEST benchmark exhibiting particularly large headroom. In this paper, we provide an in-depth analysis of the factors contributing to the poor performance on QUEST-LOFT, publish updated numbers based on a thorough human evaluation, and demonstrate that RAG can be optimized to significantly outperform long-context approaches when combined with a structured output format containing reasoning and evidence, optionally followed by answer re-verification. 

**Abstract (ZH)**: 尽管检索增强生成（RAG）在学术界和工业界作为基于接地问答的解决方案广受欢迎，但现有的RAG方法在需要检索的信息分布在多个文档中或检索需要与复杂推理结合的问题上表现不佳。最近，LOFT研究显示，这一限制也适用于基于长上下文语言模型的方法， QUEST基准特别揭示了巨大的改进空间。在本文中，我们对QUEST-LOFT表现不良的因素进行了深入分析，发布了基于全面人工评估的最新数据，并证明当RAG与包含推理和证据的结构化输出格式结合使用，并可选地进行答案重验证时，可以显著优于长上下文方法。 

---
# Adapting Web Agents with Synthetic Supervision 

**Title (ZH)**: 适应Web代理的合成监督 

**Authors**: Zhaoyang Wang, Yiming Liang, Xuchao Zhang, Qianhui Wu, Siwei Han, Anson Bastos, Rujia Wang, Chetan Bansal, Baolin Peng, Jianfeng Gao, Saravan Rajmohan, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2511.06101)  

**Abstract**: Web agents struggle to adapt to new websites due to the scarcity of environment specific tasks and demonstrations. Recent works have explored synthetic data generation to address this challenge, however, they suffer from data quality issues where synthesized tasks contain hallucinations that cannot be executed, and collected trajectories are noisy with redundant or misaligned actions. In this paper, we propose SynthAgent, a fully synthetic supervision framework that aims at improving synthetic data quality via dual refinement of both tasks and trajectories. Our approach begins by synthesizing diverse tasks through categorized exploration of web elements, ensuring efficient coverage of the target environment. During trajectory collection, we refine tasks when conflicts with actual observations are detected, mitigating hallucinations while maintaining task consistency. After collection, we conduct trajectory refinement with a global context to mitigate potential noise or misalignments. Finally, we fine-tune open-source web agents on the refined synthetic data to adapt them to the target environment. Experimental results demonstrate that SynthAgent outperforms existing synthetic data methods, validating the importance of high-quality synthetic supervision. The code will be publicly available at this https URL. 

**Abstract (ZH)**: Web代理难以适应新网站，因为环境特定的任务和示范稀缺。 recent工作探索了合成数据生成以应对这一挑战，然而它们遭受数据质量的问题，合成任务包含无法执行的幻觉，收集的轨迹噪声较大且存在冗余或对齐不当的动作。本文提出SynthAgent，这是一种全合成监督框架，旨在通过任务和轨迹的双重细化来提高合成数据的质量。我们的方法通过分类探索网页元素来合成多样化的任务，确保高效覆盖目标环境。在轨迹收集过程中，当检测到与实际观察的冲突时，会细化任务，减轻幻觉同时保持任务一致性。收集后，我们使用全局上下文来细化轨迹，以减轻潜在的噪声或对齐不当。最后，我们对细化后的合成数据微调开源Web代理，使它们适应目标环境。实验结果表明，SynthAgent优于现有合成数据方法，验证了高质量合成监督的重要性。代码将在该网址公开 доступ于此。 

---
# SWE-fficiency: Can Language Models Optimize Real-World Repositories on Real Workloads? 

**Title (ZH)**: SWE-效率：语言模型能优化实际工作负载下的仓库吗？ 

**Authors**: Jeffrey Jian Ma, Milad Hashemi, Amir Yazdanbakhsh, Kevin Swersky, Ofir Press, Enhui Li, Vijay Janapa Reddi, Parthasarathy Ranganathan  

**Link**: [PDF](https://arxiv.org/pdf/2511.06090)  

**Abstract**: Optimizing the performance of large-scale software repositories demands expertise in code reasoning and software engineering (SWE) to reduce runtime while preserving program correctness. However, most benchmarks emphasize what to fix rather than how to fix code. We introduce \textsc{SWE-fficiency}, a benchmark for evaluating repository-level performance optimization on real workloads. Our suite contains 498 tasks across nine widely used data-science, machine-learning, and HPC repositories (e.g., numpy, pandas, scipy): given a complete codebase and a slow workload, an agent must investigate code semantics, localize bottlenecks and relevant tests, and produce a patch that matches or exceeds expert speedup while passing the same unit tests. To enable this how-to-fix evaluation, our automated pipeline scrapes GitHub pull requests for performance-improving edits, combining keyword filtering, static analysis, coverage tooling, and execution validation to both confirm expert speedup baselines and identify relevant repository unit tests. Empirical evaluation of state-of-the-art agents reveals significant underperformance. On average, agents achieve less than 0.15x the expert speedup: agents struggle in localizing optimization opportunities, reasoning about execution across functions, and maintaining correctness in proposed edits. We release the benchmark and accompanying data pipeline to facilitate research on automated performance engineering and long-horizon software reasoning. 

**Abstract (ZH)**: 优化大规模软件仓库的性能需要在代码推理和软件工程（SWE）方面具备专业知识以减少运行时长同时保持程序正确性。然而，大多数基准测试更侧重于强调如何修复代码。我们引入了SWE-fficiency，一个用于评估仓库级性能优化的实际负载基准测试套件。该套件包含498个任务，涉及九个广泛使用的数据科学、机器学习和HPC仓库（例如，numpy、pandas、scipy）：给定一个完整的代码库和一个缓慢的工作负载，代理必须调查代码语义、定位瓶颈和相关测试，并生成一个符合或超过专家加速性能的补丁，同时通过相同的单元测试。为了支持这种如何修复的评估，我们的自动化管道从GitHub拉取请求中抓取性能改进的编辑内容，结合关键词过滤、静态分析、覆盖率工具和执行验证，既确认了专家加速性能的基础，也确定了相关的仓库单元测试。对先进代理的实证评估显示了显著的性能不足。平均而言，代理实现的加速性能不到专家加速性能的0.15倍：代理在定位优化机会、跨函数推理以及保持提议编辑的正确性方面存在困难。我们发布了该基准测试和配套的数据管道，以推动自动性能工程和长期软件推理研究。 

---
# Hybrid CNN-ViT Framework for Motion-Blurred Scene Text Restoration 

**Title (ZH)**: Hybrid CNN-ViT框架在运动模糊场景文本恢复中的应用 

**Authors**: Umar Rashid, Muhammad Arslan Arshad, Ghulam Ahmad, Muhammad Zeeshan Anjum, Rizwan Khan, Muhammad Akmal  

**Link**: [PDF](https://arxiv.org/pdf/2511.06087)  

**Abstract**: Motion blur in scene text images severely impairs readability and hinders the reliability of computer vision tasks, including autonomous driving, document digitization, and visual information retrieval. Conventional deblurring approaches are often inadequate in handling spatially varying blur and typically fall short in modeling the long-range dependencies necessary for restoring textual clarity. To overcome these limitations, we introduce a hybrid deep learning framework that combines convolutional neural networks (CNNs) with vision transformers (ViTs), thereby leveraging both local feature extraction and global contextual reasoning. The architecture employs a CNN-based encoder-decoder to preserve structural details, while a transformer module enhances global awareness through self-attention. Training is conducted on a curated dataset derived from TextOCR, where sharp scene-text samples are paired with synthetically blurred versions generated using realistic motion-blur kernels of multiple sizes and orientations. Model optimization is guided by a composite loss that incorporates mean absolute error (MAE), squared error (MSE), perceptual similarity, and structural similarity (SSIM). Quantitative eval- uations show that the proposed method attains 32.20 dB in PSNR and 0.934 in SSIM, while remaining lightweight with 2.83 million parameters and an average inference time of 61 ms. These results highlight the effectiveness and computational efficiency of the CNN-ViT hybrid design, establishing its practicality for real-world motion-blurred scene-text restoration. 

**Abstract (ZH)**: 场景文本图像中的运动模糊严重削弱了可读性，并妨碍了自动驾驶、文档数字化和视觉信息检索等计算机视觉任务的可靠性。为克服这些限制，我们提出了一种结合卷积神经网络（CNN）和视觉变换器（ViT）的混合深度学习框架，从而利用局部特征提取和全局上下文推理。该架构采用基于CNN的编码-解码器保留结构细节，而变换器模块通过自注意力增强全局意识。模型在从TextOCR derivated数据集中训练，该数据集包含与多种大小和方向的真实运动模糊内核生成的合成模糊版本配对的清晰场景文本样本。模型优化由综合损失引导，该损失包括均绝对误差（MAE）、平方误差（MSE）、感知相似性和结构相似性（SSIM）。定量评估显示，所提出的方法在PSNR上达到32.20 dB，在SSIM上达到0.934，同时保持轻量化，参数量为283万，平均推断时间为61 ms。这些结果突显了CNN-ViT混合设计的有效性和计算效率，证明了其在现实世界运动模糊场景文本恢复中的实用性。 

---
# Simulating Students with Large Language Models: A Review of Architecture, Mechanisms, and Role Modelling in Education with Generative AI 

**Title (ZH)**: 使用大型语言模型模拟学生：生成人工智能在教育中架构、机制及角色 modeling 的综述 

**Authors**: Luis Marquez-Carpintero, Alberto Lopez-Sellers, Miguel Cazorla  

**Link**: [PDF](https://arxiv.org/pdf/2511.06078)  

**Abstract**: Simulated Students offer a valuable methodological framework for evaluating pedagogical approaches and modelling diverse learner profiles, tasks which are otherwise challenging to undertake systematically in real-world settings. Recent research has increasingly focused on developing such simulated agents to capture a range of learning styles, cognitive development pathways, and social behaviours. Among contemporary simulation techniques, the integration of large language models (LLMs) into educational research has emerged as a particularly versatile and scalable paradigm. LLMs afford a high degree of linguistic realism and behavioural adaptability, enabling agents to approximate cognitive processes and engage in contextually appropriate pedagogical dialogues. This paper presents a thematic review of empirical and methodological studies utilising LLMs to simulate student behaviour across educational environments. We synthesise current evidence on the capacity of LLM-based agents to emulate learner archetypes, respond to instructional inputs, and interact within multi-agent classroom scenarios. Furthermore, we examine the implications of such systems for curriculum development, instructional evaluation, and teacher training. While LLMs surpass rule-based systems in natural language generation and situational flexibility, ongoing concerns persist regarding algorithmic bias, evaluation reliability, and alignment with educational objectives. The review identifies existing technological and methodological gaps and proposes future research directions for integrating generative AI into adaptive learning systems and instructional design. 

**Abstract (ZH)**: 模拟学生为评估教学方法和建模多样化学习者特征提供了宝贵的方法论框架，这在现实世界环境中是系统开展具有挑战性的任务。近年来的研究越来越多地致力于开发此类模拟代理，以捕捉各种学习风格、认知发展路径和社会行为。在当代模拟技术中，将大型语言模型（LLMs）整合到教育研究中已演变为一个特别灵活和可扩展的范式。大型语言模型提供了高度的语言真实性和行为适应性，使代理能够逼近认知过程并在情境适切的教育对话中互动。本文对利用大型语言模型模拟教育环境中学生行为的实证和方法论研究进行了主题综述。我们综合了当前关于基于大型语言模型的代理在模拟学习者典型特征、响应教学输入以及在多代理教室场景中互动方面的证据。此外，我们探讨了此类系统对课程开发、教学评价和教师培训的影响。尽管大型语言模型在自然语言生成和情境灵活性方面超越了基于规则的系统，但持续存在的算法偏见、评估可靠性与教育目标的对齐等问题仍然存在。综述指出现有技术和方法论的空白，并提出了将生成式AI集成到自适应学习系统和教学设计中的未来研究方向。 

---
# Stemming Hallucination in Language Models Using a Licensing Oracle 

**Title (ZH)**: 使用许可 oracle 抑制语言模型中的幻觉 

**Authors**: Simeon Emanuilov, Richard Ackermann  

**Link**: [PDF](https://arxiv.org/pdf/2511.06073)  

**Abstract**: Language models exhibit remarkable natural language generation capabilities but remain prone to hallucinations, generating factually incorrect information despite producing syntactically coherent responses. This study introduces the Licensing Oracle, an architectural solution designed to stem hallucinations in LMs by enforcing truth constraints through formal validation against structured knowledge graphs. Unlike statistical approaches that rely on data scaling or fine-tuning, the Licensing Oracle embeds a deterministic validation step into the model's generative process, ensuring that only factually accurate claims are made. We evaluated the effectiveness of the Licensing Oracle through experiments comparing it with several state-of-the-art methods, including baseline language model generation, fine-tuning for factual recall, fine-tuning for abstention behavior, and retrieval-augmented generation (RAG). Our results demonstrate that although RAG and fine-tuning improve performance, they fail to eliminate hallucinations. In contrast, the Licensing Oracle achieved perfect abstention precision (AP = 1.0) and zero false answers (FAR-NE = 0.0), ensuring that only valid claims were generated with 89.1% accuracy in factual responses. This work shows that architectural innovations, such as the Licensing Oracle, offer a necessary and sufficient solution for hallucinations in domains with structured knowledge representations, offering guarantees that statistical methods cannot match. Although the Licensing Oracle is specifically designed to address hallucinations in fact-based domains, its framework lays the groundwork for truth-constrained generation in future AI systems, providing a new path toward reliable, epistemically grounded models. 

**Abstract (ZH)**: 语言模型展示了出色的自然语言生成能力，但仍易产生幻觉，即使生成的响应句法上是正确的，也可能包含事实错误信息。本研究引入了许可证oracle，这是一种架构解决方案，通过与结构化知识图进行形式验证来强制执行真实性约束，以遏制语言模型的幻觉。与依赖于数据放缩或微调的统计方法不同，许可证oracle将确定性验证步骤嵌入到模型的生成过程中，确保仅生成事实准确的陈述。我们通过实验将许可证oracle与若干前沿方法进行了比较，包括基线语言模型生成、针对事实回忆的微调、针对避免行为的微调以及检索增强生成（RAG）。实验结果表明，虽然RAG和微调可以改善性能，但未能消除幻觉。相比之下，许可证oracle实现了完美的避免行为精确度（AP = 1.0）和零错误答案（FAR-NE = 0.0），确保了在事实响应中仅生成准确声明的比例为89.1%。本研究表明，如许可证oracle这样的架构创新为具有结构化知识表示的领域提供了一种必要且充分的幻觉解决方法，提供了统计方法无法比拟的保证。尽管许可证oracle专门设计用于解决基于事实领域的幻觉问题，但其架构为未来AI系统的真相约束生成奠定了基础，为可靠的、符合知识论的方法提供了新的途径。 

---
# A Privacy-Preserving Federated Learning Method with Homomorphic Encryption in Omics Data 

**Title (ZH)**: 一种在omics数据中基于同态加密的隐私保护联邦学习方法 

**Authors**: Yusaku Negoya, Feifei Cui, Zilong Zhang, Miao Pan, Tomoaki Ohtsuki, Aohan Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06064)  

**Abstract**: Omics data is widely employed in medical research to identify disease mechanisms and contains highly sensitive personal information. Federated Learning (FL) with Differential Privacy (DP) can ensure the protection of omics data privacy against malicious user attacks. However, FL with the DP method faces an inherent trade-off: stronger privacy protection degrades predictive accuracy due to injected noise. On the other hand, Homomorphic Encryption (HE) allows computations on encrypted data and enables aggregation of encrypted gradients without DP-induced noise can increase the predictive accuracy. However, it may increase the computation cost. To improve the predictive accuracy while considering the computational ability of heterogeneous clients, we propose a Privacy-Preserving Machine Learning (PPML)-Hybrid method by introducing HE. In the proposed PPML-Hybrid method, clients distributed select either HE or DP based on their computational resources, so that HE clients contribute noise-free updates while DP clients reduce computational overhead. Meanwhile, clients with high computational resources clients can flexibly adopt HE or DP according to their privacy needs. Performance evaluation on omics datasets show that our proposed method achieves comparable predictive accuracy while significantly reducing computation time relative to HE-only. Additionally, it outperforms DP-only methods under equivalent or stricter privacy budgets. 

**Abstract (ZH)**: 基于同态加密的隐私保护机器学习混合方法：同时提高预测准确性和计算效率 

---
# How Particle-System Random Batch Methods Enhance Graph Transformer: Memory Efficiency and Parallel Computing Strategy 

**Title (ZH)**: 粒子系统随机批次方法如何增强图变压器：内存效率和并行计算策略 

**Authors**: Hanwen Liu, Yixuan Ma, Shi Jin, Yuguang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06044)  

**Abstract**: Attention mechanism is a significant part of Transformer models. It helps extract features from embedded vectors by adding global information and its expressivity has been proved to be powerful. Nevertheless, the quadratic complexity restricts its practicability. Although several researches have provided attention mechanism in sparse form, they are lack of theoretical analysis about the expressivity of their mechanism while reducing complexity. In this paper, we put forward Random Batch Attention (RBA), a linear self-attention mechanism, which has theoretical support of the ability to maintain its expressivity. Random Batch Attention has several significant strengths as follows: (1) Random Batch Attention has linear time complexity. Other than this, it can be implemented in parallel on a new dimension, which contributes to much memory saving. (2) Random Batch Attention mechanism can improve most of the existing models by replacing their attention mechanisms, even many previously improved attention mechanisms. (3) Random Batch Attention mechanism has theoretical explanation in convergence, as it comes from Random Batch Methods on computation mathematics. Experiments on large graphs have proved advantages mentioned above. Also, the theoretical modeling of self-attention mechanism is a new tool for future research on attention-mechanism analysis. 

**Abstract (ZH)**: 随机批次注意力机制：一种具有理论支持的线性自注意力机制 

---
# Advancing Ocean State Estimation with efficient and scalable AI 

**Title (ZH)**: 基于高效可扩展AI的海洋状态估计 advancement 

**Authors**: Yanfei Xiang, Yuan Gao, Hao Wu, Quan Zhang, Ruiqi Shu, Xiao Zhou, Xi Wu, Xiaomeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06041)  

**Abstract**: Accurate and efficient global ocean state estimation remains a grand challenge for Earth system science, hindered by the dual bottlenecks of computational scalability and degraded data fidelity in traditional data assimilation (DA) and deep learning (DL) approaches. Here we present an AI-driven Data Assimilation Framework for Ocean (ADAF-Ocean) that directly assimilates multi-source and multi-scale observations, ranging from sparse in-situ measurements to 4 km satellite swaths, without any interpolation or data thinning. Inspired by Neural Processes, ADAF-Ocean learns a continuous mapping from heterogeneous inputs to ocean states, preserving native data fidelity. Through AI-driven super-resolution, it reconstructs 0.25$^\circ$ mesoscale dynamics from coarse 1$^\circ$ fields, which ensures both efficiency and scalability, with just 3.7\% more parameters than the 1$^\circ$ configuration. When coupled with a DL forecasting system, ADAF-Ocean extends global forecast skill by up to 20 days compared to baselines without assimilation. This framework establishes a computationally viable and scientifically rigorous pathway toward real-time, high-resolution Earth system monitoring. 

**Abstract (ZH)**: AI驱动的海洋数据同化框架：直接同化多源多尺度观测以实现高效高分辨率的全球海洋状态估计 

---
# S2ML: Spatio-Spectral Mutual Learning for Depth Completion 

**Title (ZH)**: S2ML：空谱互学习的深度完成 

**Authors**: Zihui Zhao, Yifei Zhang, Zheng Wang, Yang Li, Kui Jiang, Zihan Geng, Chia-Wen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.06033)  

**Abstract**: The raw depth images captured by RGB-D cameras using Time-of-Flight (TOF) or structured light often suffer from incomplete depth values due to weak reflections, boundary shadows, and artifacts, which limit their applications in downstream vision tasks. Existing methods address this problem through depth completion in the image domain, but they overlook the physical characteristics of raw depth images. It has been observed that the presence of invalid depth areas alters the frequency distribution pattern. In this work, we propose a Spatio-Spectral Mutual Learning framework (S2ML) to harmonize the advantages of both spatial and frequency domains for depth completion. Specifically, we consider the distinct properties of amplitude and phase spectra and devise a dedicated spectral fusion module. Meanwhile, the local and global correlations between spatial-domain and frequency-domain features are calculated in a unified embedding space. The gradual mutual representation and refinement encourage the network to fully explore complementary physical characteristics and priors for more accurate depth completion. Extensive experiments demonstrate the effectiveness of our proposed S2ML method, outperforming the state-of-the-art method CFormer by 0.828 dB and 0.834 dB on the NYU-Depth V2 and SUN RGB-D datasets, respectively. 

**Abstract (ZH)**: 基于时空谱互学的深度补全框架（S2ML）： harmonic融合时空域优势实现深度补全 

---
# ITPP: Learning Disentangled Event Dynamics in Marked Temporal Point Processes 

**Title (ZH)**: ITPP: 学习标记时点过程中的解纠缠事件动力学 

**Authors**: Wang-Tao Zhou, Zhao Kang, Ke Yan, Ling Tian  

**Link**: [PDF](https://arxiv.org/pdf/2511.06032)  

**Abstract**: Marked Temporal Point Processes (MTPPs) provide a principled framework for modeling asynchronous event sequences by conditioning on the history of past events. However, most existing MTPP models rely on channel-mixing strategies that encode information from different event types into a single, fixed-size latent representation. This entanglement can obscure type-specific dynamics, leading to performance degradation and increased risk of overfitting. In this work, we introduce ITPP, a novel channel-independent architecture for MTPP modeling that decouples event type information using an encoder-decoder framework with an ODE-based backbone. Central to ITPP is a type-aware inverted self-attention mechanism, designed to explicitly model inter-channel correlations among heterogeneous event types. This architecture enhances effectiveness and robustness while reducing overfitting. Comprehensive experiments on multiple real-world and synthetic datasets demonstrate that ITPP consistently outperforms state-of-the-art MTPP models in both predictive accuracy and generalization. 

**Abstract (ZH)**: 基于ODE的通道独立Marked Temporal Point Processes模型（ITPP） 

---
# MiVID: Multi-Strategic Self-Supervision for Video Frame Interpolation using Diffusion Model 

**Title (ZH)**: MiVID：多策略自监督方法在扩散模型下的视频帧插值 

**Authors**: Priyansh Srivastava, Romit Chatterjee, Abir Sen, Aradhana Behura, Ratnakar Dash  

**Link**: [PDF](https://arxiv.org/pdf/2511.06019)  

**Abstract**: Video Frame Interpolation (VFI) remains a cornerstone in video enhancement, enabling temporal upscaling for tasks like slow-motion rendering, frame rate conversion, and video restoration. While classical methods rely on optical flow and learning-based models assume access to dense ground-truth, both struggle with occlusions, domain shifts, and ambiguous motion. This article introduces MiVID, a lightweight, self-supervised, diffusion-based framework for video interpolation. Our model eliminates the need for explicit motion estimation by combining a 3D U-Net backbone with transformer-style temporal attention, trained under a hybrid masking regime that simulates occlusions and motion uncertainty. The use of cosine-based progressive masking and adaptive loss scheduling allows our network to learn robust spatiotemporal representations without any high-frame-rate supervision. Our framework is evaluated on UCF101-7 and DAVIS-7 datasets. MiVID is trained entirely on CPU using the datasets and 9-frame video segments, making it a low-resource yet highly effective pipeline. Despite these constraints, our model achieves optimal results at just 50 epochs, competitive with several supervised this http URL work demonstrates the power of self-supervised diffusion priors for temporally coherent frame synthesis and provides a scalable path toward accessible and generalizable VFI systems. 

**Abstract (ZH)**: 视频帧插值（VFI）仍然是视频增强的核心技术，能够在慢动作渲染、帧率转换和视频恢复等任务中实现时间上的放大。尽管经典方法依赖于光流法，基于学习的模型假定可获得密集的_ground-truth_，两者都难以处理遮挡、领域漂移和模糊运动。本文介绍了MiVID，一种轻量级、自监督、基于扩散的视频插值框架。我们的模型通过结合3D U-Net主干和类似变压器的时域注意力机制，采用模拟遮挡和运动不确定性的混合遮挡训练方式，消除了显式运动估计的需要。基于余弦分布的渐进式遮挡和自适应损失调度使网络能够在无需高帧率监督的情况下学习稳健的时空表示。本文在UCF101-7和DAVIS-7数据集上评估了该框架。MiVID完全在CPU上使用这些数据集和9帧视频片段进行训练，成为一个低资源但高效的流水线。尽管存在这些限制，我们的模型在仅仅50个epoch后就达到了最优结果，与多种监督学习方法具有竞争力。该工作展示了自监督扩散先验在时空一致帧合成中的强大能力，并提供了一条通往可访问和通用的视频帧插值系统的可扩展路径。 

---
# One-Shot Knowledge Transfer for Scalable Person Re-Identification 

**Title (ZH)**: 基于单次知识迁移的可扩展行人重识别 

**Authors**: Longhua Li, Lei Qi, Xin Geng  

**Link**: [PDF](https://arxiv.org/pdf/2511.06016)  

**Abstract**: Edge computing in person re-identification (ReID) is crucial for reducing the load on central cloud servers and ensuring user privacy. Conventional compression methods for obtaining compact models require computations for each individual student model. When multiple models of varying sizes are needed to accommodate different resource conditions, this leads to repetitive and cumbersome computations. To address this challenge, we propose a novel knowledge inheritance approach named OSKT (One-Shot Knowledge Transfer), which consolidates the knowledge of the teacher model into an intermediate carrier called a weight chain. When a downstream scenario demands a model that meets specific resource constraints, this weight chain can be expanded to the target model size without additional computation. OSKT significantly outperforms state-of-the-art compression methods, with the added advantage of one-time knowledge transfer that eliminates the need for frequent computations for each target model. 

**Abstract (ZH)**: 边缘计算在行人重识别（ReID）中的应用对于减轻中央云服务器的负担和确保用户隐私至关重要。为了应对这一挑战，我们提出了一种名为OSKT（一次性知识转移）的新型知识继承方法，该方法将教师模型的知识整合到一个中间载体——权重链中。当下游场景需要满足特定资源约束的模型时，无需额外计算即可扩展该权重链至目标模型大小。OSKT显著优于最先进的压缩方法，并且具有一次性知识转移的优势，避免了为每个目标模型进行频繁的额外计算。 

---
# MoSKA: Mixture of Shared KV Attention for Efficient Long-Sequence LLM Inference 

**Title (ZH)**: MoSKA: 共享键值注意力的混合高效长序列语言模型推理 

**Authors**: Myunghyun Rhee, Sookyung Choi, Euiseok Kim, Joonseop Sim, Youngpyo Joo, Hoshik Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.06010)  

**Abstract**: The escalating context length in Large Language Models (LLMs) creates a severe performance bottleneck around the Key-Value (KV) cache, whose memory-bound nature leads to significant GPU under-utilization. This paper introduces Mixture of Shared KV Attention (MoSKA), an architecture that addresses this challenge by exploiting the heterogeneity of context data. It differentiates between per-request unique and massively reused shared sequences. The core of MoSKA is a novel Shared KV Attention mechanism that transforms the attention on shared data from a series of memory-bound GEMV operations into a single, compute-bound GEMM by batching concurrent requests. This is supported by an MoE-inspired sparse attention strategy that prunes the search space and a tailored Disaggregated Infrastructure that specializes hardware for unique and shared data. This comprehensive approach demonstrates a throughput increase of up to 538.7x over baselines in workloads with high context sharing, offering a clear architectural path toward scalable LLM inference. 

**Abstract (ZH)**: 大规模语言模型中上下文长度不断提升导致关键值缓存性能瓶颈，MoSKA架构通过利用上下文数据的异构性来应对这一挑战。它区分了每个请求的唯一序列和大规模重复的共享序列。MoSKA的核心是一种新颖的共享关键值注意机制，将对共享数据的注意从一系列内存绑定的GEMV操作转换为针对并发请求的单一、计算绑定的GEMM操作。该机制通过一种受专家门控启发的稀疏注意策略来精简搜索空间，并通过定制化的分解基础设施专门化硬件以处理独特的和共享的数据。这种综合方法在高上下文共享的工作负载中展示了高达538.7倍的吞吐量提升，提供了一条通往可扩展的大规模语言模型推理的清晰架构路径。 

---
# Exploring Category-level Articulated Object Pose Tracking on SE(3) Manifolds 

**Title (ZH)**: 在SE(3)流形上探索类别级 articulated 物体姿态跟踪 

**Authors**: Xianhui Meng, Yukang Huo, Li Zhang, Liu Liu, Haonan Jiang, Yan Zhong, Pingrui Zhang, Cewu Lu, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05996)  

**Abstract**: Articulated objects are prevalent in daily life and robotic manipulation tasks. However, compared to rigid objects, pose tracking for articulated objects remains an underexplored problem due to their inherent kinematic constraints. To address these challenges, this work proposes a novel point-pair-based pose tracking framework, termed \textbf{PPF-Tracker}. The proposed framework first performs quasi-canonicalization of point clouds in the SE(3) Lie group space, and then models articulated objects using Point Pair Features (PPF) to predict pose voting parameters by leveraging the invariance properties of SE(3). Finally, semantic information of joint axes is incorporated to impose unified kinematic constraints across all parts of the articulated object. PPF-Tracker is systematically evaluated on both synthetic datasets and real-world scenarios, demonstrating strong generalization across diverse and challenging environments. Experimental results highlight the effectiveness and robustness of PPF-Tracker in multi-frame pose tracking of articulated objects. We believe this work can foster advances in robotics, embodied intelligence, and augmented reality. Codes are available at this https URL. 

**Abstract (ZH)**: articulated物体在日常生活中和机器人操作任务中十分普遍。然而，与刚性物体相比，由于其内在的运动学约束，articulated物体的姿态追踪仍然是一个尚未充分探索的问题。为了解决这些挑战，本工作提出了一种新颖的点对为基础的姿态追踪框架，命名为PPF-Tracker。该框架首先在SE(3)李群空间中执行准规范化的点云处理，然后使用点对特征（PPF）建模articulated物体，并通过利用SE(3)的不变性属性预测姿态投票参数。最后，将关节轴的语义信息纳入，以在整个articulated物体的各个部分施加统一的运动学约束。PPF-Tracker在合成数据集和真实世界场景中进行了系统评估，展示了其在多种复杂环境下的强泛化能力和鲁棒性。实验结果突显了PPF-Tracker在articulated物体多帧姿态追踪中的有效性和鲁棒性。我们认为，这项工作可以在机器人学、嵌入式智能和增强现实方面促进进步。代码可在以下链接获取。 

---
# Revisiting Entropy in Reinforcement Learning for Large Reasoning Models 

**Title (ZH)**: 重访大规模推理模型中的熵在强化学习中的应用 

**Authors**: Renren Jin, Pengzhi Gao, Yuqi Ren, Zhuowen Han, Tongxuan Zhang, Wuwei Huang, Wei Liu, Jian Luan, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2511.05993)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has emerged as a predominant approach for enhancing the reasoning capabilities of large language models (LLMs). However, the entropy of LLMs usually collapses during RLVR training, causing premature convergence to suboptimal local minima and hinder further performance improvement. Although various approaches have been proposed to mitigate entropy collapse, a comprehensive study of entropy in RLVR remains lacking. To address this gap, we conduct extensive experiments to investigate the entropy dynamics of LLMs trained with RLVR and analyze how model entropy correlates with response diversity, calibration, and performance across various benchmarks. Our findings reveal that the number of off-policy updates, the diversity of training data, and the clipping thresholds in the optimization objective are critical factors influencing the entropy of LLMs trained with RLVR. Moreover, we theoretically and empirically demonstrate that tokens with positive advantages are the primary contributors to entropy collapse, and that model entropy can be effectively regulated by adjusting the relative loss weights of tokens with positive and negative advantages during training. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）已成为增强大型语言模型（LLMs）推理能力的主要方法。然而，在RLVR训练过程中，LLMs的熵通常会崩溃，导致过早收敛到次优局部极小值，妨碍进一步性能提升。尽管提出了一些缓解熵崩溃的方法，但对于RLVR中的熵的全面研究仍然不足。为填补这一空白，我们进行了广泛实验，探讨使用RLVR训练的LLMs的熵动力学，并分析模型熵与响应多样性、校准和性能之间的关系。我们的研究发现，离策略更新的数量、训练数据的多样性以及优化目标中的裁剪阈值是影响使用RLVR训练的LLMs熵的关键因素。此外，我们从理论和实证上证明，具有正优势的标记是导致熵崩溃的主要因素，并且可以通过调整训练过程中具有正和负优势的标记的相对损失权重来有效调节模型熵。 

---
# Ontology Learning and Knowledge Graph Construction: A Comparison of Approaches and Their Impact on RAG Performance 

**Title (ZH)**: 本体学习与知识图谱构建：方法比较及其对rag性能的影响 

**Authors**: Tiago da Cruz, Bernardo Tavares, Francisco Belo  

**Link**: [PDF](https://arxiv.org/pdf/2511.05991)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems combine Large Language Models (LLMs) with external knowledge, and their performance depends heavily on how that knowledge is represented. This study investigates how different Knowledge Graph (KG) construction strategies influence RAG performance. We compare a variety of approaches: standard vector-based RAG, GraphRAG, and retrieval over KGs built from ontologies derived either from relational databases or textual corpora. Results show that ontology-guided KGs incorporating chunk information achieve competitive performance with state-of-the-art frameworks, substantially outperforming vector retrieval baselines. Moreover, the findings reveal that ontology-guided KGs built from relational databases perform competitively to ones built with ontologies extracted from text, with the benefit of offering a dual advantage: they require a one-time-only ontology learning process, substantially reducing LLM usage costs; and avoid the complexity of ontology merging inherent to text-based approaches. 

**Abstract (ZH)**: 基于知识图谱的检索增强生成系统：不同构建策略对性能的影响 

---
# Runtime Safety Monitoring of Deep Neural Networks for Perception: A Survey 

**Title (ZH)**: 深度神经网络感知中的运行时安全性监控：一篇综述 

**Authors**: Albert Schotschneider, Svetlana Pavlitska, J. Marius Zöllner  

**Link**: [PDF](https://arxiv.org/pdf/2511.05982)  

**Abstract**: Deep neural networks (DNNs) are widely used in perception systems for safety-critical applications, such as autonomous driving and robotics. However, DNNs remain vulnerable to various safety concerns, including generalization errors, out-of-distribution (OOD) inputs, and adversarial attacks, which can lead to hazardous failures. This survey provides a comprehensive overview of runtime safety monitoring approaches, which operate in parallel to DNNs during inference to detect these safety concerns without modifying the DNN itself. We categorize existing methods into three main groups: Monitoring inputs, internal representations, and outputs. We analyze the state-of-the-art for each category, identify strengths and limitations, and map methods to the safety concerns they address. In addition, we highlight open challenges and future research directions. 

**Abstract (ZH)**: 深度神经网络（DNNs）广泛应用于自动驾驶和机器人等安全关键应用的感知系统中。然而，DNNs仍然容易受到各种安全问题的影响，包括泛化错误、分布外（OOD）输入和对抗攻击，这些都可能导致危险的故障。本文提供了一种关于运行时安全监控方法的全面综述，这些方法在推理过程中并行于DNN运行，以检测这些安全问题而不修改DNN本身。我们将现有方法分为三大类：监控输入、内部表示和输出。我们分析了每一类的最新技术，指出了其优势和局限性，并将方法与所解决的安全问题进行了映射。此外，我们还强调了开放性挑战和未来研究方向。 

---
# Kunlun Anomaly Troubleshooter: Enabling Kernel-Level Anomaly Detection and Causal Reasoning for Large Model Distributed Inference 

**Title (ZH)**: Kunlun 异常 troubleshooter: 支持大型模型分布式推理内核级异常检测及因果推理 

**Authors**: Yuyang Liu, Jingjing Cai, Jiayi Ren, Peng Zhou, Danyang Zhang, Yin Du, Shijian Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.05978)  

**Abstract**: Anomaly troubleshooting for large model distributed inference (LMDI) remains a critical challenge. Resolving anomalies such as inference performance degradation or latency jitter in distributed system demands significant manual efforts from domain experts, resulting in extremely time-consuming diagnosis processes with relatively low accuracy. In this paper, we introduce Kunlun Anomaly Troubleshooter (KAT), the first anomaly troubleshooting framework tailored for LMDI. KAT addresses this problem through two core innovations. First, KAT exploits the synchronicity and consistency of GPU workers, innovatively leverages function trace data to precisely detect kernel-level anomalies and associated hardware components at nanosecond resolution. Second, KAT integrates these detection results into a domain-adapted LLM, delivering systematic causal reasoning and natural language interpretation of complex anomaly symptoms. Evaluations conducted in Alibaba Cloud Service production environment indicate that KAT achieves over 0.884 precision and 0.936 recall in anomaly detection, providing detail anomaly insights that significantly narrow down the diagnostic scope and improve both the efficiency and success rate of troubleshooting. 

**Abstract (ZH)**: 大型模型分布式推理中的异常故障排查（LMDI）仍然是一项关键挑战。鞍引用功能跟踪数据以纳米级分辨率精确检测内核级异常及其相关硬件组件，通过这一创新，KAT能够在阿里云服务生产环境中实现超过0.884的异常检测精准率和0.936的召回率，为故障排查提供详细的异常洞察，显著缩小诊断范围，提高故障排查的效率和成功率。 

---
# Interpretable Recognition of Cognitive Distortions in Natural Language Texts 

**Title (ZH)**: 可解释的认知扭曲识别在自然语言文本中的识别 

**Authors**: Anton Kolonin, Anna Arinicheva  

**Link**: [PDF](https://arxiv.org/pdf/2511.05969)  

**Abstract**: We propose a new approach to multi-factor classification of natural language texts based on weighted structured patterns such as N-grams, taking into account the heterarchical relationships between them, applied to solve such a socially impactful problem as the automation of detection of specific cognitive distortions in psychological care, relying on an interpretable, robust and transparent artificial intelligence model. The proposed recognition and learning algorithms improve the current state of the art in this field. The improvement is tested on two publicly available datasets, with significant improvements over literature-known F1 scores for the task, with optimal hyper-parameters determined, having code and models available for future use by the community. 

**Abstract (ZH)**: 基于加权结构模式的多层次分类方法：考虑异位关系在心理护理中自动检测特定认知歪曲的透明可解释人工智能模型及其改进算法 

---
# DiA-gnostic VLVAE: Disentangled Alignment-Constrained Vision Language Variational AutoEncoder for Robust Radiology Reporting with Missing Modalities 

**Title (ZH)**: DiA-gnostic VLVAE: 解耦联对齐约束视觉语言变分自编码器在缺失模态下实现稳健的放射学报告生成 

**Authors**: Nagur Shareef Shaik, Teja Krishna Cherukuri, Adnan Masood, Dong Hye Ye  

**Link**: [PDF](https://arxiv.org/pdf/2511.05968)  

**Abstract**: The integration of medical images with clinical context is essential for generating accurate and clinically interpretable radiology reports. However, current automated methods often rely on resource-heavy Large Language Models (LLMs) or static knowledge graphs and struggle with two fundamental challenges in real-world clinical data: (1) missing modalities, such as incomplete clinical context , and (2) feature entanglement, where mixed modality-specific and shared information leads to suboptimal fusion and clinically unfaithful hallucinated findings. To address these challenges, we propose the DiA-gnostic VLVAE, which achieves robust radiology reporting through Disentangled Alignment. Our framework is designed to be resilient to missing modalities by disentangling shared and modality-specific features using a Mixture-of-Experts (MoE) based Vision-Language Variational Autoencoder (VLVAE). A constrained optimization objective enforces orthogonality and alignment between these latent representations to prevent suboptimal fusion. A compact LLaMA-X decoder then uses these disentangled representations to generate reports efficiently. On the IU X-Ray and MIMIC-CXR datasets, DiA has achieved competetive BLEU@4 scores of 0.266 and 0.134, respectively. Experimental results show that the proposed method significantly outperforms state-of-the-art models. 

**Abstract (ZH)**: 医学影像与临床背景的整合对于生成准确且临床可解释的放射学报告至关重要。然而，当前的自动化方法往往依赖资源密集型大型语言模型（LLMs）或静态知识图谱，并且在真实世界的临床数据中面临两大根本挑战：（1）缺失模态，如不完整的临床背景，以及（2）特征纠缠，其中混合模态特定和共享信息导致次优融合并产生临床不忠实的虚幻发现。为了解决这些挑战，我们提出了DiA-gnostic VLVAE，通过消 entangled alignment实现稳健的放射学报告。该框架通过基于Mixture-of-Experts（MoE）的Vision-Language Variational Autoencoder（VLVAE）来消 entangle共享和模态特定特征，以增强对缺失模态的鲁棒性。通过约束优化目标，确保这些潜在表示之间的正交性和对齐，从而防止次优融合。紧凑型LLaMA-X解码器然后使用这些消 entangled的表示高效地生成报告。在IU X-Ray和MIMIC-CXR数据集上，DiA分别获得了竞争性的BLEU@4分数0.266和0.134。实验结果表明，所提出的方法显著优于现有最佳模型。 

---
# Adapted Foundation Models for Breast MRI Triaging in Contrast-Enhanced and Non-Contrast Enhanced Protocols 

**Title (ZH)**: 适应型基础模型在对比增强和非对比增强乳腺MRI筛查中的应用 

**Authors**: Tri-Thien Nguyen, Lorenz A. Kapsner, Tobias Hepp, Shirin Heidarikahkesh, Hannes Schreiter, Luise Brock, Dominika Skwierawska, Dominique Hadler, Julian Hossbach, Evelyn Wenkel, Sabine Ohlmeyer, Frederik B. Laun, Andrzej Liebert, Andreas Maier, Michael Uder, Sebastian Bickelhaupt  

**Link**: [PDF](https://arxiv.org/pdf/2511.05967)  

**Abstract**: Background: Magnetic resonance imaging (MRI) has high sensitivity for breast cancer detection, but interpretation is time-consuming. Artificial intelligence may aid in pre-screening. Purpose: To evaluate the DINOv2-based Medical Slice Transformer (MST) for ruling out significant findings (Breast Imaging Reporting and Data System [BI-RADS] >=4) in contrast-enhanced and non-contrast-enhanced abbreviated breast MRI. Materials and Methods: This institutional review board approved retrospective study included 1,847 single-breast MRI examinations (377 BI-RADS >=4) from an in-house dataset and 924 from an external validation dataset (Duke). Four abbreviated protocols were tested: T1-weighted early subtraction (T1sub), diffusion-weighted imaging with b=1500 s/mm2 (DWI1500), DWI1500+T2-weighted (T2w), and T1sub+T2w. Performance was assessed at 90%, 95%, and 97.5% sensitivity using five-fold cross-validation and area under the receiver operating characteristic curve (AUC) analysis. AUC differences were compared with the DeLong test. False negatives were characterized, and attention maps of true positives were rated in the external dataset. Results: A total of 1,448 female patients (mean age, 49 +/- 12 years) were included. T1sub+T2w achieved an AUC of 0.77 +/- 0.04; DWI1500+T2w, 0.74 +/- 0.04 (p=0.15). At 97.5% sensitivity, T1sub+T2w had the highest specificity (19% +/- 7%), followed by DWI1500+T2w (17% +/- 11%). Missed lesions had a mean diameter <10 mm at 95% and 97.5% thresholds for both T1sub and DWI1500, predominantly non-mass enhancements. External validation yielded an AUC of 0.77, with 88% of attention maps rated good or moderate. Conclusion: At 97.5% sensitivity, the MST framework correctly triaged cases without BI-RADS >=4, achieving 19% specificity for contrast-enhanced and 17% for non-contrast-enhanced MRI. Further research is warranted before clinical implementation. 

**Abstract (ZH)**: 基于DINOv2的Medical Slice Transformer（MST）在对比增强和非对比增强摘要乳腺MRI中排除BI-RADS >=4显著发现的评估 

---
# Adaptive Agent Selection and Interaction Network for Image-to-point cloud Registration 

**Title (ZH)**: 自适应代理选择与交互网络在图像到点云注册中的应用 

**Authors**: Zhixin Cheng, Xiaotian Yin, Jiacheng Deng, Bohao Liao, Yujia Chen, Xu Zhou, Baoqun Yin, Tianzhu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05965)  

**Abstract**: Typical detection-free methods for image-to-point cloud registration leverage transformer-based architectures to aggregate cross-modal features and establish correspondences. However, they often struggle under challenging conditions, where noise disrupts similarity computation and leads to incorrect correspondences. Moreover, without dedicated designs, it remains difficult to effectively select informative and correlated representations across modalities, thereby limiting the robustness and accuracy of registration. To address these challenges, we propose a novel cross-modal registration framework composed of two key modules: the Iterative Agents Selection (IAS) module and the Reliable Agents Interaction (RAI) module. IAS enhances structural feature awareness with phase maps and employs reinforcement learning principles to efficiently select reliable agents. RAI then leverages these selected agents to guide cross-modal interactions, effectively reducing mismatches and improving overall robustness. Extensive experiments on the RGB-D Scenes v2 and 7-Scenes benchmarks demonstrate that our method consistently achieves state-of-the-art performance. 

**Abstract (ZH)**: 典型的无需检测的方法通过基于变换器的架构聚合跨模态特征并建立对应关系来进行图像到点云注册。然而，这些方法在噪声干扰相似性计算并导致对应关系错误的具有挑战性的条件下往往表现不佳。此外，在没有专门设计的情况下，难以有效选择具有信息性和关联性的跨模态表示，从而限制了注册的鲁棒性和准确性。为应对这些挑战，我们提出了一种新的跨模态注册框架，包含两个关键模块：迭代代理选择（IAS）模块和可靠代理交互（RAI）模块。IAS通过使用相位图增强结构特征感知，并利用强化学习原则高效地选择可靠的代理。RAI随后利用这些选定的代理引导跨模态交互，有效减少不匹配并提高总体鲁棒性。在RGB-D Scenes v2和7-Scenes基准上的广泛实验表明，我们的方法能够一致地达到最先进的性能。 

---
# A PDE Perspective on Generative Diffusion Models 

**Title (ZH)**: 生成扩散模型的偏微分方程视角 

**Authors**: Kang Liu, Enrique Zuazua  

**Link**: [PDF](https://arxiv.org/pdf/2511.05940)  

**Abstract**: Score-based diffusion models have emerged as a powerful class of generative methods, achieving state-of-the-art performance across diverse domains. Despite their empirical success, the mathematical foundations of those models remain only partially understood, particularly regarding the stability and consistency of the underlying stochastic and partial differential equations governing their dynamics.
In this work, we develop a rigorous partial differential equation (PDE) framework for score-based diffusion processes. Building on the Li--Yau differential inequality for the heat flow, we prove well-posedness and derive sharp $L^p$-stability estimates for the associated score-based Fokker--Planck dynamics, providing a mathematically consistent description of their temporal evolution. Through entropy stability methods, we further show that the reverse-time dynamics of diffusion models concentrate on the data manifold for compactly supported data distributions and a broad class of initialization schemes, with a concentration rate of order $\sqrt{t}$ as $t \to 0$.
These results yield a theoretical guarantee that, under exact score guidance, diffusion trajectories return to the data manifold while preserving imitation fidelity. Our findings also provide practical insights for designing diffusion models, including principled criteria for score-function construction, loss formulation, and stopping-time selection. Altogether, this framework provides a quantitative understanding of the trade-off between generative capacity and imitation fidelity, bridging rigorous analysis and model design within a unified mathematical perspective. 

**Abstract (ZH)**: 基于分数的扩散模型 telah发展成为一种强大的生成方法，在多个领域取得了最先进的性能。尽管这些模型在实践中取得了成功，但其背后的数学基础仍只部分被理解，尤其是在关于指导其动态的随机和偏微分方程的稳定性和一致性方面。

在本文中，我们构建了一个严格的偏微分方程（PDE）框架来描述基于分数的扩散过程。基于Li-Yau热流的微分不等式，我们证明了相关分数驱动的Fokker-Planck动力学的适定性，并推导出尖锐的$L^p$稳定性估计，提供了一个从数学上一致地描述其时间演化的方法。通过熵稳定性方法，我们进一步证明，在紧支数据分布和广泛类别的初始化方案下，反向时间动力学将扩散模型聚集在数据流形上，聚集速率为$\sqrt{t}$当$t \to 0$。

这些结果为在准确分数引导下，扩散轨迹返回数据流形并保持仿真是提供了理论保障。我们的发现还为设计扩散模型提供了实用见解，包括分数函数构建、损失函数形式化和停止时间选择的准则。总体而言，这一框架为生成能力和仿真的权衡提供了定量理解，统一了严谨分析和模型设计的数学视角。 

---
# 10 Open Challenges Steering the Future of Vision-Language-Action Models 

**Title (ZH)**: 10 开放挑战引领视觉-语言-行动模型的未来 

**Authors**: Soujanya Poria, Navonil Majumder, Chia-Yu Hung, Amir Ali Bagherzadeh, Chuan Li, Kenneth Kwok, Ziwei Wang, Cheston Tan, Jiajun Wu, David Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05936)  

**Abstract**: Due to their ability of follow natural language instructions, vision-language-action (VLA) models are increasingly prevalent in the embodied AI arena, following the widespread success of their precursors -- LLMs and VLMs. In this paper, we discuss 10 principal milestones in the ongoing development of VLA models -- multimodality, reasoning, data, evaluation, cross-robot action generalization, efficiency, whole-body coordination, safety, agents, and coordination with humans. Furthermore, we discuss the emerging trends of using spatial understanding, modeling world dynamics, post training, and data synthesis -- all aiming to reach these milestones. Through these discussions, we hope to bring attention to the research avenues that may accelerate the development of VLA models into wider acceptability. 

**Abstract (ZH)**: 由于具有遵循自然语言指令的能力，多模态语言动作（VLA）模型在具身AI领域越来越普遍，这得益于其前身——大规模语言模型（LLMs）和大规模视觉模型（VLMs）的广泛成功。在本文中，我们讨论了VLA模型在不断发展过程中的10个主要里程碑——多模态性、推理、数据、评估、跨机器人动作通用化、效率、全身协调、安全性、智能体以及与人类的协调。此外，我们还探讨了使用空间理解、建模世界动力学、训练后处理和数据合成等新兴趋势，旨在实现这些里程碑。通过这些讨论，我们希望引起对能够加速VLA模型广泛应用的研究方向的关注。 

---
# Reinforcement Learning Improves Traversal of Hierarchical Knowledge in LLMs 

**Title (ZH)**: 强化学习提高大型语言模型中层级知识的遍历能力 

**Authors**: Renfei Zhang, Manasa Kaniselvan, Niloofar Mireshghallah  

**Link**: [PDF](https://arxiv.org/pdf/2511.05933)  

**Abstract**: Reinforcement learning (RL) is often credited with improving language model reasoning and generalization at the expense of degrading memorized knowledge. We challenge this narrative by observing that RL-enhanced models consistently outperform their base and supervised fine-tuned (SFT) counterparts on pure knowledge recall tasks, particularly those requiring traversal of hierarchical, structured knowledge (e.g., medical codes). We hypothesize these gains stem not from newly acquired data, but from improved procedural skills in navigating and searching existing knowledge hierarchies within the model parameters. To support this hypothesis, we show that structured prompting, which explicitly guides SFTed models through hierarchical traversal, recovers most of the performance gap (reducing 24pp to 7pp on MedConceptsQA for DeepSeek-V3/R1). We further find that while prompting improves final-answer accuracy, RL-enhanced models retain superior ability to recall correct procedural paths on deep-retrieval tasks. Finally our layer-wise internal activation analysis reveals that while factual representations (e.g., activations for the statement "code 57.95 refers to urinary infection") maintain high cosine similarity between SFT and RL models, query representations (e.g., "what is code 57.95") diverge noticeably, indicating that RL primarily transforms how models traverse knowledge rather than the knowledge representation itself. 

**Abstract (ZH)**: 强化学习（RL）在提高语言模型推理和泛化能力的同时，并未牺牲记忆知识的能力：一种挑战性观点 

---
# The Future of AI in the GCC Post-NPM Landscape: A Comparative Analysis of Kuwait and the UAE 

**Title (ZH)**: 卡塔尔合作委员会国家后NPM景观下人工智能的未来：科威特与阿联酋的比较分析 

**Authors**: Mohammad Rashed Albous, Bedour Alboloushi, Arnaud Lacheret  

**Link**: [PDF](https://arxiv.org/pdf/2511.05932)  

**Abstract**: Comparative evidence on how Gulf Cooperation Council (GCC) states turn artificial intelligence (AI) ambitions into post--New Public Management (post-NPM) outcomes is scarce because most studies examine Western democracies. We analyze constitutional, collective-choice, and operational rules shaping AI uptake in two contrasting GCC members, the United Arab Emirates (UAE) and Kuwait, and whether they foster citizen centricity, collaborative governance, and public value creation. Anchored in Ostrom's Institutional Analysis and Development framework, the study combines a most similar/most different systems design with multiple sources: 62 public documents from 2018--2025, embedded UAE cases (Smart Dubai and MBZUAI), and 39 interviews with officials conducted Aug 2024--May 2025. Dual coding and process tracing connect rule configurations to AI performance. Cross-case analysis identifies four reinforcing mechanisms behind divergent trajectories. In the UAE, concentrated authority, credible sanctions, pro-innovation narratives, and flexible reinvestment rules scale pilots into hundreds of services and sizable recycled savings. In Kuwait, dispersed veto points, exhortative sanctions, cautious discourse, and lapsed AI budgets confine initiatives to pilot mode despite equivalent fiscal resources. The findings refine institutional theory by showing that vertical rule coherence, not wealth, determines AI's public-value yield, and temper post-NPM optimism by revealing that efficiency metrics serve societal goals only when backed by enforceable safeguards. To curb ethics washing and test transferability beyond the GCC, future work should track rule diffusion over time, develop blended legitimacy--efficiency scorecards, and examine how narrative framing shapes citizen consent for data sharing. 

**Abstract (ZH)**: 海湾合作 Council (GCC) 国家如何将人工智能 (AI) 理想转化为后新公共管理 (post-NPM) 结果的比较证据稀缺，因为大多数研究关注西方民主国家。本文分析了两个对比鲜明的 GCC 成员国阿联酋和科威特的宪法、集体选择和操作规则如何影响 AI 的采纳，并探讨这些规则是否促进以公民为中心、协作治理和公共价值创造。基于奥斯特罗姆的制度分析与发展框架，本文结合最相似/最不同系统设计，综合利用 2018-2025 年间的 62 份公开文件、嵌入式阿联酋案例（智能迪拜和 MBZUAI）以及 2024 年 8 月至 2025 年 5 月进行的 39 位官员访谈，通过双重编码和过程追踪将规则配置与 AI 表现联系起来。交叉案例分析识别出了四种推动不同轨迹的强化机制。在阿联酋，集中化的权威、可信的制裁措施、倡导创新的叙述以及灵活的资金重新投资规则将试点扩展为数百项服务和大量回收的储蓄。在科威特，分散的否决点、激进的制裁措施、谨慎的论调以及过时的人工智能预算尽管拥有相当的财政资源，仍将倡议局限于试点模式。研究结果细化了制度理论，表明垂直规则的一致性而非财富决定了人工智能的公共价值产出，并适度减弱了后新公共管理的乐观情绪，揭示了只在具有可执行保障措施的支持下，效率指标才能服务于社会目标。为了遏制伦理洗白并测试其在 GCC 以外地区的可转移性，未来研究应跟踪规则的扩散情况，开发融合合法性和效率的评分卡，并研究叙述框架如何影响公民对数据共享的同意。 

---
# CoMA: Complementary Masking and Hierarchical Dynamic Multi-Window Self-Attention in a Unified Pre-training Framework 

**Title (ZH)**: CoMA: 统一预训练框架下的互补遮掩与分层动态多窗口自注意力机制 

**Authors**: Jiaxuan Li, Qing Xu, Xiangjian He, Ziyu Liu, Chang Xing, Zhen Chen, Daokun Zhang, Rong Qu, Chang Wen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.05929)  

**Abstract**: Masked Autoencoders (MAE) achieve self-supervised learning of image representations by randomly removing a portion of visual tokens and reconstructing the original image as a pretext task, thereby significantly enhancing pretraining efficiency and yielding excellent adaptability across downstream tasks. However, MAE and other MAE-style paradigms that adopt random masking generally require more pre-training epochs to maintain adaptability. Meanwhile, ViT in MAE suffers from inefficient parameter use due to fixed spatial resolution across layers. To overcome these limitations, we propose the Complementary Masked Autoencoders (CoMA), which employ a complementary masking strategy to ensure uniform sampling across all pixels, thereby improving effective learning of all features and enhancing the model's adaptability. Furthermore, we introduce DyViT, a hierarchical vision transformer that employs a Dynamic Multi-Window Self-Attention (DM-MSA), significantly reducing the parameters and FLOPs while improving fine-grained feature learning. Pre-trained on ImageNet-1K with CoMA, DyViT matches the downstream performance of MAE using only 12% of the pre-training epochs, demonstrating more effective learning. It also attains a 10% reduction in pre-training time per epoch, further underscoring its superior pre-training efficiency. 

**Abstract (ZH)**: 互补掩蔽自编码器（CoMA）通过采用互补掩蔽策略确保均匀采样所有像素，从而提高所有特征的有效学习并增强模型的适应性。进一步引入动态多窗口自注意力（DM-MSA）的层级视觉变换器（DyViT），显著减少参数量和FLOPs，同时提高细粒度特征学习能力。使用CoMA预训练在ImageNet-1K上的DyViT，仅需MAE 12%的预训练 epoch 即能达到相同的下游任务性能，展示出更有效的学习效果，并将每 epoch 的预训练时间减少10%，进一步证明了其优异的预训练效率。 

---
# Artificial intelligence and the Gulf Cooperation Council workforce adapting to the future of work 

**Title (ZH)**: 人工智能与海湾合作 council 劳动力适应未来工作 

**Authors**: Mohammad Rashed Albous, Melodena Stephens, Odeh Rashed Al-Jayyousi  

**Link**: [PDF](https://arxiv.org/pdf/2511.05927)  

**Abstract**: The rapid expansion of artificial intelligence (AI) in the Gulf Cooperation Council (GCC) raises a central question: are investments in compute infrastructure matched by an equally robust build-out of skills, incentives, and governance? Grounded in socio-technical systems (STS) theory, this mixed-methods study audits workforce preparedness across Kingdom of Saudi Arabia (KSA), the United Arab Emirates (UAE), Qatar, Kuwait, Bahrain, and Oman. We combine term frequency--inverse document frequency (TF--IDF) analysis of six national AI strategies (NASs), an inventory of 47 publicly disclosed AI initiatives (January 2017--April 2025), paired case studies, the Mohamed bin Zayed University of Artificial Intelligence (MBZUAI) and the Saudi Data & Artificial Intelligence Authority (SDAIA) Academy, and a scenario matrix linking oil-revenue slack (technical capacity) to regulatory coherence (social alignment). Across the corpus, 34/47 initiatives (0.72; 95% Wilson CI 0.58--0.83) exhibit joint social--technical design; country-level indices span 0.57--0.90 (small n; intervals overlap). Scenario results suggest that, under our modeled conditions, regulatory convergence plausibly binds outcomes more than fiscal capacity: fragmented rules can offset high oil revenues, while harmonized standards help preserve progress under austerity. We also identify an emerging two-track talent system, research elites versus rapidly trained practitioners, that risks labor-market bifurcation without bridging mechanisms. By extending STS inquiry to oil-rich, state-led economies, the study refines theory and sets a research agenda focused on longitudinal coupling metrics, ethnographies of coordination, and outcome-based performance indicators. 

**Abstract (ZH)**: GCC国家中人工智能的迅速扩张引发了核心问题：计算基础设施的投资是否与技能、激励和治理的建设相匹配？基于社会技术系统（STS）理论，本综合性研究审查了沙特阿拉伯王国（KSA）、阿拉伯联合酋长国（UAE）、卡塔尔、科威特、巴林和阿曼的工作force准备情况。我们综合了六个国家人工智能战略（NASs）的词频—逆文档频率（TF-IDF）分析、197项公开披露的人工智能倡议清单（2017年1月—2025年4月）、成对案例研究、Mohamed bin Zayed大学人工智能（MBZUAI）和沙特数据与人工智能管理局（SDAIA）学院的数据，以及将石油收入盈余（技术能力）与监管一致性（社会共识）联系起来的情景矩阵。在这些倡议中，47项中的34项（0.72；95% Wilson置信区间0.58--0.83）展现了社会-技术联合设计；国家指数范围为0.57--0.90（样本量小；区间重叠）。情景分析结果表明，在我们的建模条件下，监管一致性的收敛可能比财政能力更能绑定结果：碎片化的规则可以抵消高石油收入，而协调化的标准则有助于在紧缩环境下维持进展。我们还发现了一种新兴的双轨人才系统，科研精英与快速训练的实践者，这可能加剧劳动力市场的分化，除非有弥合机制。通过将STS研究扩展到石油丰富的、政府主导的经济体，本研究细化了理论并设定了关注纵向耦合指标、协调的民族志和基于结果的绩效指标的研究议程。 

---
# IDALC: A Semi-Supervised Framework for Intent Detection and Active Learning based Correction 

**Title (ZH)**: IDALC：一种基于意图检测和主动学习的半监督框架及校正方法 

**Authors**: Ankan Mullick, Sukannya Purkayastha, Saransh Sharma, Pawan Goyal, Niloy Ganguly  

**Link**: [PDF](https://arxiv.org/pdf/2511.05921)  

**Abstract**: Voice-controlled dialog systems have become immensely popular due to their ability to perform a wide range of actions in response to diverse user queries. These agents possess a predefined set of skills or intents to fulfill specific user tasks. But every system has its own limitations. There are instances where, even for known intents, if any model exhibits low confidence, it results in rejection of utterances that necessitate manual annotation. Additionally, as time progresses, there may be a need to retrain these agents with new intents from the system-rejected queries to carry out additional tasks. Labeling all these emerging intents and rejected utterances over time is impractical, thus calling for an efficient mechanism to reduce annotation costs. In this paper, we introduce IDALC (Intent Detection and Active Learning based Correction), a semi-supervised framework designed to detect user intents and rectify system-rejected utterances while minimizing the need for human annotation. Empirical findings on various benchmark datasets demonstrate that our system surpasses baseline methods, achieving a 5-10% higher accuracy and a 4-8% improvement in macro-F1. Remarkably, we maintain the overall annotation cost at just 6-10% of the unlabelled data available to the system. The overall framework of IDALC is shown in Fig. 1 

**Abstract (ZH)**: 基于意图检测和主动学习的半监督框架IDALC：减少标注成本的方法 

---
# IoT-based Fresh Produce Supply Chain Under Uncertainty: An Adaptive Optimization Framework 

**Title (ZH)**: 基于物联网的不确定性生鲜供应链：一种适应性优化框架 

**Authors**: Chirag Seth, Mehrdad Pirnia, James H Bookbinder  

**Link**: [PDF](https://arxiv.org/pdf/2511.05920)  

**Abstract**: Fruits and vegetables form a vital component of the global economy; however, their distribution poses complex logistical challenges due to high perishability, supply fluctuations, strict quality and safety standards, and environmental sensitivity. In this paper, we propose an adaptive optimization model that accounts for delays, travel time, and associated temperature changes impacting produce shelf life, and compare it against traditional approaches such as Robust Optimization, Distributionally Robust Optimization, and Stochastic Programming. Additionally, we conduct a series of computational experiments using Internet of Things (IoT) sensor data to evaluate the performance of our proposed model. Our study demonstrates that the proposed adaptive model achieves a higher shelf life, extending it by over 18\% compared to traditional optimization models, by dynamically mitigating temperature deviations through a temperature feedback mechanism. The promising results demonstrate the potential of this approach to improve both the freshness and efficiency of logistics systems an aspect often neglected in previous works. 

**Abstract (ZH)**: 果蔬是全球经济的重要组成部分；然而，由于高易腐性、供应波动、严格的品质和安全标准以及环境敏感性，其流通面临着复杂的物流挑战。本文提出了一种适应性优化模型，该模型考虑到运输延迟、旅行时间和伴随的温度变化对农产品保质期的影响，并将其与鲁棒优化、分布鲁棒优化和随机规划等传统方法进行了比较。此外，我们使用物联网（IoT）传感器数据进行了系列计算实验，以评估所提模型的性能。研究表明，所提适应性模型通过动态调节温度偏差，实现了比传统优化模型更高的保质期，延长了超过18%。这些有希望的结果表明，该方法有可能提高物流系统的鲜度和效率，这是以往研究中经常忽视的一个方面。 

---
# Injecting Falsehoods: Adversarial Man-in-the-Middle Attacks Undermining Factual Recall in LLMs 

**Title (ZH)**: 注入虚假信息： adversarial  man-in-the-middle 攻击削弱大语言模型事实回忆能力 

**Authors**: Alina Fastowski, Bardh Prenkaj, Yuxiao Li, Gjergji Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2511.05919)  

**Abstract**: LLMs are now an integral part of information retrieval. As such, their role as question answering chatbots raises significant concerns due to their shown vulnerability to adversarial man-in-the-middle (MitM) attacks. Here, we propose the first principled attack evaluation on LLM factual memory under prompt injection via Xmera, our novel, theory-grounded MitM framework. By perturbing the input given to "victim" LLMs in three closed-book and fact-based QA settings, we undermine the correctness of the responses and assess the uncertainty of their generation process. Surprisingly, trivial instruction-based attacks report the highest success rate (up to ~85.3%) while simultaneously having a high uncertainty for incorrectly answered questions. To provide a simple defense mechanism against Xmera, we train Random Forest classifiers on the response uncertainty levels to distinguish between attacked and unattacked queries (average AUC of up to ~96%). We believe that signaling users to be cautious about the answers they receive from black-box and potentially corrupt LLMs is a first checkpoint toward user cyberspace safety. 

**Abstract (ZH)**: LLMs在信息检索中的作用使其成为对抗性中间人攻击下的问答漏洞评估的首个原理性攻击评估。通过我们的新型理论支撑的MitM框架Xmera，在三种闭卷和基于事实的问答设置中，通过扰动“受害”LLM的输入，我们削弱了其回答的正确性并评估了其生成过程的不确定性。令人惊讶的是，基于指令的攻击报告了最高的成功率（高达约85.3%），同时对错误回答的问题具有较高的不确定性。为对抗Xmera，我们通过在响应不确定性水平上训练随机森林分类器来区分受攻击和未受攻击的查询（平均AUC高达约96%）。我们认为，提醒用户谨慎对待来自黑盒且可能被篡改的LLM的答案是通往用户网络空间安全的第一步。 

---
# NILC: Discovering New Intents with LLM-assisted Clustering 

**Title (ZH)**: NILC：使用LLM辅助聚类发现新意图 

**Authors**: Hongtao Wang, Renchi Yang, Wenqing Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.05913)  

**Abstract**: New intent discovery (NID) seeks to recognize both new and known intents from unlabeled user utterances, which finds prevalent use in practical dialogue systems. Existing works towards NID mainly adopt a cascaded architecture, wherein the first stage focuses on encoding the utterances into informative text embeddings beforehand, while the latter is to group similar embeddings into clusters (i.e., intents), typically by K-Means. However, such a cascaded pipeline fails to leverage the feedback from both steps for mutual refinement, and, meanwhile, the embedding-only clustering overlooks nuanced textual semantics, leading to suboptimal performance. To bridge this gap, this paper proposes NILC, a novel clustering framework specially catered for effective NID. Particularly, NILC follows an iterative workflow, in which clustering assignments are judiciously updated by carefully refining cluster centroids and text embeddings of uncertain utterances with the aid of large language models (LLMs). Specifically, NILC first taps into LLMs to create additional semantic centroids for clusters, thereby enriching the contextual semantics of the Euclidean centroids of embeddings. Moreover, LLMs are then harnessed to augment hard samples (ambiguous or terse utterances) identified from clusters via rewriting for subsequent cluster correction. Further, we inject supervision signals through non-trivial techniques seeding and soft must links for more accurate NID in the semi-supervised setting. Extensive experiments comparing NILC against multiple recent baselines under both unsupervised and semi-supervised settings showcase that NILC can achieve significant performance improvements over six benchmark datasets of diverse domains consistently. 

**Abstract (ZH)**: 新意图发现中的聚类框架：基于大型语言模型的交互式聚类（NILC） 

---
# The Imperfect Learner: Incorporating Developmental Trajectories in Memory-based Student Simulation 

**Title (ZH)**: Imperfect学习者：在基于记忆的学生模拟中融入发展轨迹 

**Authors**: Zhengyuan Liu, Stella Xin Yin, Bryan Chen Zhengyu Tan, Roy Ka-Wei Lee, Guimei Liu, Dion Hoe-Lian Goh, Wenya Wang, Nancy F. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.05903)  

**Abstract**: User simulation is important for developing and evaluating human-centered AI, yet current student simulation in educational applications has significant limitations. Existing approaches focus on single learning experiences and do not account for students' gradual knowledge construction and evolving skill sets. Moreover, large language models are optimized to produce direct and accurate responses, making it challenging to represent the incomplete understanding and developmental constraints that characterize real learners. In this paper, we introduce a novel framework for memory-based student simulation that incorporates developmental trajectories through a hierarchical memory mechanism with structured knowledge representation. The framework also integrates metacognitive processes and personality traits to enrich the individual learner profiling, through dynamical consolidation of both cognitive development and personal learning characteristics. In practice, we implement a curriculum-aligned simulator grounded on the Next Generation Science Standards. Experimental results show that our approach can effectively reflect the gradual nature of knowledge development and the characteristic difficulties students face, providing a more accurate representation of learning processes. 

**Abstract (ZH)**: 基于记忆的发展性学生模拟框架：通过层次化记忆机制和结构化知识表示实现个体学习者特征的动态整合 

---
# Retrieval-Augmented Generation in Medicine: A Scoping Review of Technical Implementations, Clinical Applications, and Ethical Considerations 

**Title (ZH)**: 医学中的检索增强生成：一项关于技术实现、临床应用及伦理考量的综述 

**Authors**: Rui Yang, Matthew Yu Heng Wong, Huitao Li, Xin Li, Wentao Zhu, Jingchi Liao, Kunyu Yu, Jonathan Chong Kai Liew, Weihao Xuan, Yingjian Chen, Yuhe Ke, Jasmine Chiat Ling Ong, Douglas Teodoro, Chuan Hong, Daniel Shi Wei Ting, Nan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05901)  

**Abstract**: The rapid growth of medical knowledge and increasing complexity of clinical practice pose challenges. In this context, large language models (LLMs) have demonstrated value; however, inherent limitations remain. Retrieval-augmented generation (RAG) technologies show potential to enhance their clinical applicability. This study reviewed RAG applications in medicine. We found that research primarily relied on publicly available data, with limited application in private data. For retrieval, approaches commonly relied on English-centric embedding models, while LLMs were mostly generic, with limited use of medical-specific LLMs. For evaluation, automated metrics evaluated generation quality and task performance, whereas human evaluation focused on accuracy, completeness, relevance, and fluency, with insufficient attention to bias and safety. RAG applications were concentrated on question answering, report generation, text summarization, and information extraction. Overall, medical RAG remains at an early stage, requiring advances in clinical validation, cross-linguistic adaptation, and support for low-resource settings to enable trustworthy and responsible global use. 

**Abstract (ZH)**: 医疗领域中检索增强生成技术的应用研究 

---
# GABFusion: Rethinking Feature Fusion for Low-Bit Quantization of Multi-Task Networks 

**Title (ZH)**: GABFusion: 重新思考多任务网络低比特量化中的特征融合 

**Authors**: Zhaoyang Wang, Dong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05898)  

**Abstract**: Despite the effectiveness of quantization-aware training (QAT) in compressing deep neural networks, its performance on multi-task architectures often degrades significantly due to task-specific feature discrepancies and gradient conflicts. To address these challenges, we propose Gradient-Aware Balanced Feature Fusion (GABFusion), which dynamically balances gradient magnitudes and fuses task-specific features in a quantization-friendly manner. We further introduce Attention Distribution Alignment (ADA), a feature-level distillation strategy tailored for quantized models. Our method demonstrates strong generalization across network architectures and QAT algorithms, with theoretical guarantees on gradient bias reduction. Extensive experiments demonstrate that our strategy consistently enhances a variety of QAT methods across different network architectures and bit-widths. On PASCAL VOC and COCO datasets, the proposed approach achieves average mAP improvements of approximately 3.3% and 1.6%, respectively. When applied to YOLOv5 under 4-bit quantization, our method narrows the accuracy gap with the full-precision model to only 1.7% on VOC, showcasing its effectiveness in preserving performance under low-bit constraints. Notably, the proposed framework is modular, easy to integrate, and compatible with any existing QAT technique-enhancing the performance of quantized models without requiring modifications to the original network architecture. 

**Abstract (ZH)**: Gradient-Aware Balanced Feature Fusion with Attention Distribution Alignment for Quantization-Aware Training 

---
# A Remarkably Efficient Paradigm to Multimodal Large Language Models for Sequential Recommendation 

**Title (ZH)**: 一种高效的多模态大型语言模型序列推荐范式 

**Authors**: Qiyong Zhong, Jiajie Su, Ming Yang, Yunshan Ma, Xiaolin Zheng, Chaochao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.05885)  

**Abstract**: In this paper, we proposed Speeder, a remarkably efficient paradigm to multimodal large language models for sequential recommendation. Speeder introduces 3 key components: (1) Multimodal Representation Compression (MRC), which efficiently reduces redundancy in item descriptions; (2) Sequential Position Awareness Enhancement (SPAE), which strengthens the model's ability to capture complex sequential dependencies; (3) Modality-aware Progressive Optimization (MPO), which progressively integrates different modalities to improve the model's understanding and reduce cognitive biases. Through extensive experiments, Speeder demonstrates superior performance over baselines in terms of VHR@1 and computational efficiency. Specifically, Speeder achieved 250% of the training speed and 400% of the inference speed compared to the state-of-the-art MLLM-based SR models. Future work could focus on incorporating real-time feedback from real-world systems. 

**Abstract (ZH)**: 本文提出了Speeder，一种用于序列推荐的高效多模态大语言模型范式。Speeder引入了3个关键组件：(1) 多模态表示压缩(MRC)，有效减少项目描述中的冗余；(2) 顺序位置意识增强(SPAE)，增强模型捕捉复杂顺序依赖性的能力；(3) 模态意识逐步优化(MPO)，逐步整合不同模态以提高模型的理解能力和降低认知偏见。通过广泛的实验，Speeder在VHR@1和计算效率方面均优于基线模型。具体来说，Speeder的训练速度是最新MLLM基序列推荐模型的250%，推理速度是其400%。未来的工作可以侧重于将实时反馈整合到实际系统中。 

---
# Physics-Informed Neural Networks for Real-Time Gas Crossover Prediction in PEM Electrolyzers: First Application with Multi-Membrane Validation 

**Title (ZH)**: 基于物理的神经网络在PEM电解槽中实时预测气体 crossover：首次多膜验证的应用 

**Authors**: Yong-Woon Kim, Chulung Kang, Yung-Cheol Byun  

**Link**: [PDF](https://arxiv.org/pdf/2511.05879)  

**Abstract**: Green hydrogen production via polymer electrolyte membrane (PEM) water electrolysis is pivotal for energy transition, yet hydrogen crossover through membranes threatens safety and economic viability-approaching explosive limits (4 mol% H$_2$ in O$_2$) while reducing Faradaic efficiency by 2.5%. Current physics-based models require extensive calibration and computational resources that preclude real-time implementation, while purely data-driven approaches fail to extrapolate beyond training conditions-critical for dynamic electrolyzer operation. Here we present the first application of physics-informed neural networks (PINNs) for hydrogen crossover prediction, integrating mass conservation, Fick's diffusion law, and Henry's solubility law within a compact architecture (17,793 parameters). Validated across six membranes under industrially relevant conditions (0.05-5.0 A/cm$^2$, 1-200 bar, 25-85°C), our PINN achieves exceptional accuracy (R$^2$ = 99.84%, RMSE = 0.0348%) with sub-millisecond inference times suitable for real-time control. Remarkably, the model maintains R$^2$ > 86% when predicting crossover at pressures 2.5x beyond training range-substantially outperforming pure neural networks (R$^2$ = 43.4%). The hardware-agnostic deployment, from desktop CPUs to edge devices (Raspberry Pi 4), enables distributed safety monitoring essential for gigawatt-scale installations. By bridging physical rigor and computational efficiency, this work establishes a new paradigm for real-time electrolyzer monitoring, accelerating deployment of safe, efficient green hydrogen infrastructure crucial for net-zero emissions targets. 

**Abstract (ZH)**: 基于物理信息神经网络的质子交换膜水电解制绿氢过程中渗透氢预测研究 

---
# Towards a Humanized Social-Media Ecosystem: AI-Augmented HCI Design Patterns for Safety, Agency & Well-Being 

**Title (ZH)**: 向人性化社交媒体生态系统迈进：增强交互设计模式以保障安全、自主与福祉 

**Authors**: Mohd Ruhul Ameen, Akif Islam  

**Link**: [PDF](https://arxiv.org/pdf/2511.05875)  

**Abstract**: Social platforms connect billions of people, yet their engagement-first algorithms often work on users rather than with them, amplifying stress, misinformation, and a loss of control. We propose Human-Layer AI (HL-AI)--user-owned, explainable intermediaries that sit in the browser between platform logic and the interface. HL-AI gives people practical, moment-to-moment control without requiring platform cooperation. We contribute a working Chrome/Edge prototype implementing five representative pattern frameworks--Context-Aware Post Rewriter, Post Integrity Meter, Granular Feed Curator, Micro-Withdrawal Agent, and Recovery Mode--alongside a unifying mathematical formulation balancing user utility, autonomy costs, and risk thresholds. Evaluation spans technical accuracy, usability, and behavioral outcomes. The result is a suite of humane controls that help users rewrite before harm, read with integrity cues, tune feeds with intention, pause compulsive loops, and seek shelter during harassment, all while preserving agency through explanations and override options. This prototype offers a practical path to retrofit today's feeds with safety, agency, and well-being, inviting rigorous cross-cultural user evaluation. 

**Abstract (ZH)**: 基于人类层的人工智能：用户拥有并可解释的浏览器中介，以实现时刻掌握的控制权 

---
# EndoIR: Degradation-Agnostic All-in-One Endoscopic Image Restoration via Noise-Aware Routing Diffusion 

**Title (ZH)**: EndoIR：基于噪声意识路由扩散的泛化端oscopic图像恢复 

**Authors**: Tong Chen, Xinyu Ma, Long Bai, Wenyang Wang, Sun Yue, Luping Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.05873)  

**Abstract**: Endoscopic images often suffer from diverse and co-occurring degradations such as low lighting, smoke, and bleeding, which obscure critical clinical details. Existing restoration methods are typically task-specific and often require prior knowledge of the degradation type, limiting their robustness in real-world clinical use. We propose EndoIR, an all-in-one, degradation-agnostic diffusion-based framework that restores multiple degradation types using a single model. EndoIR introduces a Dual-Domain Prompter that extracts joint spatial-frequency features, coupled with an adaptive embedding that encodes both shared and task-specific cues as conditioning for denoising. To mitigate feature confusion in conventional concatenation-based conditioning, we design a Dual-Stream Diffusion architecture that processes clean and degraded inputs separately, with a Rectified Fusion Block integrating them in a structured, degradation-aware manner. Furthermore, Noise-Aware Routing Block improves efficiency by dynamically selecting only noise-relevant features during denoising. Experiments on SegSTRONG-C and CEC datasets demonstrate that EndoIR achieves state-of-the-art performance across multiple degradation scenarios while using fewer parameters than strong baselines, and downstream segmentation experiments confirm its clinical utility. 

**Abstract (ZH)**: 一种泛化能力更强的端到端降噪框架：EndoIR 

---
# Adaptation and Fine-tuning with TabPFN for Travelling Salesman Problem 

**Title (ZH)**: TabPFN的适应与微调在 Travelling Salesman Problem 中的应用 

**Authors**: Nguyen Gia Hien Vu, Yifan Tang, Rey Lim, Yifan Yang, Hang Ma, Ke Wang, G. Gary Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05872)  

**Abstract**: Tabular Prior-Data Fitted Network (TabPFN) is a foundation model designed for small to medium-sized tabular data, which has attracted much attention recently. This paper investigates the application of TabPFN in Combinatorial Optimization (CO) problems. The aim is to lessen challenges in time and data-intensive training requirements often observed in using traditional methods including exact and heuristic algorithms, Machine Learning (ML)-based models, to solve CO problems. Proposing possibly the first ever application of TabPFN for such a purpose, we adapt and fine-tune the TabPFN model to solve the Travelling Salesman Problem (TSP), one of the most well-known CO problems. Specifically, we adopt the node-based approach and the node-predicting adaptation strategy to construct the entire TSP route. Our evaluation with varying instance sizes confirms that TabPFN requires minimal training, adapts to TSP using a single sample, performs better generalization across varying TSP instance sizes, and reduces performance degradation. Furthermore, the training process with adaptation and fine-tuning is completed within minutes. The methodology leads to strong solution quality even without post-processing and achieves performance comparable to other models with post-processing refinement. Our findings suggest that the TabPFN model is a promising approach to solve structured and CO problems efficiently under training resource constraints and rapid deployment requirements. 

**Abstract (ZH)**: 表格式先验-数据适配网络（TabPFN）是一种针对小型到中型表格数据的基础模型，近期引起了广泛关注。本文探讨了TabPFN在组合优化（CO）问题中的应用。目标是减轻使用传统方法（包括精确和启发式算法、基于机器学习的模型）解决CO问题时通常观察到的时间和数据密集型训练需求带来的挑战。我们提出了一种可能首次用于此类目的的应用，将TabPFN模型调整和微调以解决旅行商问题（TSP），这是最著名的CO问题之一。具体而言，我们采用节点基方法和节点预测调整策略来构建整个TSP路径。不同实例规模的评估表明，TabPFN需要最少的训练，使用单个样本适应TSP，能够在不同TSP实例规模下更好地泛化，并减少性能下降。此外，适应和微调的训练过程在几分钟内完成。该方法即使在无需后处理的情况下也能够产生高质量的解决方案，并达到了其他经过后处理细化的模型相当的性能。我们的研究结果表明，在训练资源有限和快速部署要求下，TabPFN模型是一种解决结构化和CO问题的有效方法。 

---
# CGCE: Classifier-Guided Concept Erasure in Generative Models 

**Title (ZH)**: CGCE：分类器引导的概念消除在生成模型中 

**Authors**: Viet Nguyen, Vishal M. Patel  

**Link**: [PDF](https://arxiv.org/pdf/2511.05865)  

**Abstract**: Recent advancements in large-scale generative models have enabled the creation of high-quality images and videos, but have also raised significant safety concerns regarding the generation of unsafe content. To mitigate this, concept erasure methods have been developed to remove undesirable concepts from pre-trained models. However, existing methods remain vulnerable to adversarial attacks that can regenerate the erased content. Moreover, achieving robust erasure often degrades the model's generative quality for safe, unrelated concepts, creating a difficult trade-off between safety and performance. To address this challenge, we introduce Classifier-Guided Concept Erasure (CGCE), an efficient plug-and-play framework that provides robust concept erasure for diverse generative models without altering their original weights. CGCE uses a lightweight classifier operating on text embeddings to first detect and then refine prompts containing undesired concepts. This approach is highly scalable, allowing for multi-concept erasure by aggregating guidance from several classifiers. By modifying only unsafe embeddings at inference time, our method prevents harmful content generation while preserving the model's original quality on benign prompts. Extensive experiments show that CGCE achieves state-of-the-art robustness against a wide range of red-teaming attacks. Our approach also maintains high generative utility, demonstrating a superior balance between safety and performance. We showcase the versatility of CGCE through its successful application to various modern T2I and T2V models, establishing it as a practical and effective solution for safe generative AI. 

**Abstract (ZH)**: 近期大规模生成模型的进步使高质量图像和视频的生成成为可能，但也引发了关于生成不安全内容的安全顾虑。为应对这一问题，已开发出概念擦除方法以从预训练模型中移除不良概念，但现有方法仍然容易受到可再生对攻击的影响，能够重建被擦除的内容。此外，实现鲁棒擦除往往会降低模型对安全、无关概念的生成质量，使得安全性与性能之间存在难以调和的权衡。为解决这一挑战，我们提出了一种名为Classifier-Guided Concept Erasure (CGCE)的有效即插即用框架，该框架能够在不改变原模型权重的情况下为多种生成模型提供鲁棒的概念擦除。CGCE利用一种轻量级分类器对文本嵌入进行操作，首先检测，然后完善包含不良概念的提示。该方法具有高度可扩展性，可以通过多个分类器的综合指导实现多概念擦除。通过仅在推理时修改有害的嵌入，我们的方法能够在保留模型对良性提示原始质量的同时防止有害内容的生成。广泛的实验表明，CGCE在广泛范围的红队攻击下实现了最先进的鲁棒性。该方法还保持了生成效用的高度，展示了安全性与性能之间更优的平衡。我们通过CGCE在各种现代T2I和T2V模型中的成功应用展示了其灵活性，并确立了其作为安全生成AI实用且有效的解决方案的地位。 

---
# EMOD: A Unified EEG Emotion Representation Framework Leveraging V-A Guided Contrastive Learning 

**Title (ZH)**: EMOD：一种基于V-A引导对比学习的统一脑电情感表示框架 

**Authors**: Yuning Chen, Sha Zhao, Shijian Li, Gang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2511.05863)  

**Abstract**: Emotion recognition from EEG signals is essential for affective computing and has been widely explored using deep learning. While recent deep learning approaches have achieved strong performance on single EEG emotion datasets, their generalization across datasets remains limited due to the heterogeneity in annotation schemes and data formats. Existing models typically require dataset-specific architectures tailored to input structure and lack semantic alignment across diverse emotion labels. To address these challenges, we propose EMOD: A Unified EEG Emotion Representation Framework Leveraging Valence-Arousal (V-A) Guided Contrastive Learning. EMOD learns transferable and emotion-aware representations from heterogeneous datasets by bridging both semantic and structural gaps. Specifically, we project discrete and continuous emotion labels into a unified V-A space and formulate a soft-weighted supervised contrastive loss that encourages emotionally similar samples to cluster in the latent space. To accommodate variable EEG formats, EMOD employs a flexible backbone comprising a Triple-Domain Encoder followed by a Spatial-Temporal Transformer, enabling robust extraction and integration of temporal, spectral, and spatial features. We pretrain EMOD on eight public EEG datasets and evaluate its performance on three benchmark datasets. Experimental results show that EMOD achieves state-of-the-art performance, demonstrating strong adaptability and generalization across diverse EEG-based emotion recognition scenarios. 

**Abstract (ZH)**: 基于 valence-arousal 引导对比学习的统一 EEG 情感表示框架 EMOD 

---
# Predicting the Future by Retrieving the Past 

**Title (ZH)**: 通过检索过去预测未来 

**Authors**: Dazhao Du, Tao Han, Song Guo  

**Link**: [PDF](https://arxiv.org/pdf/2511.05859)  

**Abstract**: Deep learning models such as MLP, Transformer, and TCN have achieved remarkable success in univariate time series forecasting, typically relying on sliding window samples from historical data for training. However, while these models implicitly compress historical information into their parameters during training, they are unable to explicitly and dynamically access this global knowledge during inference, relying only on the local context within the lookback window. This results in an underutilization of rich patterns from the global history. To bridge this gap, we propose Predicting the Future by Retrieving the Past (PFRP), a novel approach that explicitly integrates global historical data to enhance forecasting accuracy. Specifically, we construct a Global Memory Bank (GMB) to effectively store and manage global historical patterns. A retrieval mechanism is then employed to extract similar patterns from the GMB, enabling the generation of global predictions. By adaptively combining these global predictions with the outputs of any local prediction model, PFRP produces more accurate and interpretable forecasts. Extensive experiments conducted on seven real-world datasets demonstrate that PFRP significantly enhances the average performance of advanced univariate forecasting models by 8.4\%. Codes can be found in this https URL. 

**Abstract (ZH)**: 通过检索过去预测未来：一种增强单变量时间序列预测的新型全局历史数据整合方法 

---
# Quantifying Edits Decay in Fine-tuned LLMs 

**Title (ZH)**: 量化微调大模型中的编辑衰减 

**Authors**: Yinjie Cheng, Paul Youssef, Christin Seifert, Jörg Schlötterer, Zhixue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.05852)  

**Abstract**: Knowledge editing has emerged as a lightweight alternative to retraining for correcting or injecting specific facts in large language models (LLMs). Meanwhile, fine-tuning remains the default operation for adapting LLMs to new domains and tasks. Despite their widespread adoption, these two post-training interventions have been studied in isolation, leaving open a crucial question: if we fine-tune an edited model, do the edits survive? This question is motivated by two practical scenarios: removing covert or malicious edits, and preserving beneficial edits. If fine-tuning impairs edits as shown in Figure 1, current KE methods become less useful, as every fine-tuned model would require re-editing, which significantly increases the cost; if edits persist, fine-tuned models risk propagating hidden malicious edits, raising serious safety concerns. To this end, we systematically quantify edits decay after fine-tuning, investigating how fine-tuning affects knowledge editing. We evaluate two state-of-the-art editing methods (MEMIT, AlphaEdit) and three fine-tuning approaches (full-parameter, LoRA, DoRA) across five LLMs and three datasets, yielding 232 experimental configurations. Our results show that edits decay after fine-tuning, with survival varying across configurations, e.g., AlphaEdit edits decay more than MEMIT edits. Further, we propose selective-layer fine-tuning and find that fine-tuning edited layers only can effectively remove edits, though at a slight cost to downstream performance. Surprisingly, fine-tuning non-edited layers impairs more edits than full fine-tuning. Overall, our study establishes empirical baselines and actionable strategies for integrating knowledge editing with fine-tuning, and underscores that evaluating model editing requires considering the full LLM application pipeline. 

**Abstract (ZH)**: 知识编辑已成为一种轻量级替代重新训练的方法，用于纠正或注入大型语言模型中的特定事实。同时，微调仍然是将大型语言模型适应新领域和任务的默认操作。尽管这些后训练干预措施被广泛采用，但它们一直被孤立研究，留下了一个关键问题：如果我们对编辑后的模型进行微调，这些编辑是否会存活下来？这一问题由两种实际场景驱使：移除隐蔽或恶意编辑，以及保留有益编辑。如果如图1所示，微调会损害编辑，当前的知识编辑方法将变得不够有用，因为每个微调模型都需要重新编辑，这将显著增加成本；如果编辑得以保留，微调后的模型可能会传播隐藏的恶意编辑，引发严重的安全问题。为了解决这个问题，我们系统地量化了微调后编辑的衰减，探究微调对知识编辑的影响。我们在五种大型语言模型和三种数据集上评估了两种最先进的编辑方法（MEMIT、AlphaEdit）和三种微调方法（全参数微调、LoRA、DoRA），共得到232种实验配置。我们的结果显示，微调后编辑会衰减，不同配置下的生存率各不相同，例如，AlphaEdit编辑的衰减程度大于MEMIT编辑。进一步地，我们提出了选择性层微调，发现仅微调编辑层可以有效去除编辑，尽管会对下游性能造成轻微影响。令人惊讶的是，微调非编辑层损害的编辑比完全微调更多。总体而言，我们的研究为知识编辑与微调的集成设立了实证基准，并强调评估模型编辑时需要考虑整个大型语言模型应用管道。 

---
# Retrieval Quality at Context Limit 

**Title (ZH)**: 上下文限制条件下的检索质量 

**Authors**: Max McKinnon  

**Link**: [PDF](https://arxiv.org/pdf/2511.05850)  

**Abstract**: The ability of large language models (LLMs) to recall and retrieve information from long contexts is critical for many real-world applications. Prior work (Liu et al., 2023) reported that LLMs suffer significant drops in retrieval accuracy for facts placed in the middle of large contexts, an effect known as "Lost in the Middle" (LITM). We find the model Gemini 2.5 Flash can answer needle-in-a-haystack questions with great accuracy regardless of document position including when the document is nearly at the input context limit. Our results suggest that the "Lost in the Middle" effect is not present for simple factoid Q\&A in Gemini 2.5 Flash, indicating substantial improvements in long-context retrieval. 

**Abstract (ZH)**: 大型语言模型在长上下文中的检索能力对于许多实际应用至关重要。先前的研究（Liu et al., 2023）报告称，大型语言模型在长上下文中中间部分的事实检索准确性显著下降，这一现象被称为“迷失在中间”（LITM）。我们的研究表明，模型Gemini 2.5 Flash能够无论文档位置如何（包括接近输入上下文限制的情况）都以高度准确地回答“以 Needle-in-a-Haystack 方式提出的问题”。我们的结果表明，“迷失在中间”现象在Gemini 2.5 Flash简单的事实问答中不存在，这表明长上下文检索能力有了显著提升。 

---
# EGG-SR: Embedding Symbolic Equivalence into Symbolic Regression via Equality Graph 

**Title (ZH)**: EGG-SR: 将等式图嵌入到符号回归中以实现符号等价性 

**Authors**: Nan Jiang, Ziyi Wang, Yexiang Xue  

**Link**: [PDF](https://arxiv.org/pdf/2511.05849)  

**Abstract**: Symbolic regression seeks to uncover physical laws from experimental data by searching for closed-form expressions, which is an important task in AI-driven scientific discovery. Yet the exponential growth of the search space of expression renders the task computationally challenging. A promising yet underexplored direction for reducing the effective search space and accelerating training lies in symbolic equivalence: many expressions, although syntactically different, define the same function -- for example, $\log(x_1^2x_2^3)$, $\log(x_1^2)+\log(x_2^3)$, and $2\log(x_1)+3\log(x_2)$. Existing algorithms treat such variants as distinct outputs, leading to redundant exploration and slow learning. We introduce EGG-SR, a unified framework that integrates equality graphs (e-graphs) into diverse symbolic regression algorithms, including Monte Carlo Tree Search (MCTS), deep reinforcement learning (DRL), and large language models (LLMs). EGG-SR compactly represents equivalent expressions through the proposed EGG module, enabling more efficient learning by: (1) pruning redundant subtree exploration in EGG-MCTS, (2) aggregating rewards across equivalence classes in EGG-DRL, and (3) enriching feedback prompts in EGG-LLM. Under mild assumptions, we show that embedding e-graphs tightens the regret bound of MCTS and reduces the variance of the DRL gradient estimator. Empirically, EGG-SR consistently enhances multiple baselines across challenging benchmarks, discovering equations with lower normalized mean squared error than state-of-the-art methods. Code implementation is available at: this https URL. 

**Abstract (ZH)**: 符号回归通过搜索闭合表达式来从实验数据中揭示物理定律，是AI驱动的科学研究中的一个重要任务。然而，表达式搜索空间的指数增长使得该任务在计算上极具挑战性。通过符号等价性减少有效搜索空间和加快训练的一个有前景但尚未充分探索的方向在于：许多虽然语法不同但定义相同的功能——例如，$\log(x_1^2x_2^3)$，$\log(x_1^2)+\log(x_2^3)$，和$2\log(x_1)+3\log(x_2)$。现有算法将这些变体视为不同的输出，导致重复探索和学习缓慢。我们引入了EGG-SR，这是一种将等价图（e-graphs）整合到符号回归算法中的统一框架，包括蒙特卡洛树搜索（MCTS）、深度强化学习（DRL）和大型语言模型（LLMs）。通过提出EGG模块，EGG-SR紧凑地表示等价表达式，从而使学习更高效：（1）在EGG-MCTS中精简冗余子树探索，（2）在EGG-DRL中聚合等价类的奖励，（3）在EGG-LLM中丰富反馈提示。在轻微假设下，我们证明嵌入e-graphs使MCTS的遗憾边界更加紧，并减少了DRL梯度估计器的方差。实验中，EGG-SR在多个基准测试中持续提升了多种基线方法，在发现方程时具有比最新方法更低的归一化均方误差。代码实现可在以下链接获取：this https URL。 

---
# Enhancing Diffusion Model Guidance through Calibration and Regularization 

**Title (ZH)**: 通过校准和正则化增强扩散模型指导 

**Authors**: Seyed Alireza Javid, Amirhossein Bagheri, Nuria González-Prelcic  

**Link**: [PDF](https://arxiv.org/pdf/2511.05844)  

**Abstract**: Classifier-guided diffusion models have emerged as a powerful approach for conditional image generation, but they suffer from overconfident predictions during early denoising steps, causing the guidance gradient to vanish. This paper introduces two complementary contributions to address this issue. First, we propose a differentiable calibration objective based on the Smooth Expected Calibration Error (Smooth ECE), which improves classifier calibration with minimal fine-tuning and yields measurable improvements in Frechet Inception Distance (FID). Second, we develop enhanced sampling guidance methods that operate on off-the-shelf classifiers without requiring retraining. These include tilted sampling with batch-level reweighting, adaptive entropy-regularized sampling to preserve diversity, and a novel f-divergence-based sampling strategy that strengthens class-consistent guidance while maintaining mode coverage. Experiments on ImageNet 128x128 demonstrate that our divergence-regularized guidance achieves an FID of 2.13 using a ResNet-101 classifier, improving upon existing classifier-guided diffusion methods while requiring no diffusion model retraining. The results show that principled calibration and divergence-aware sampling provide practical and effective improvements for classifier-guided diffusion. 

**Abstract (ZH)**: 基于分类器指导的动力学模型差分方法在条件图像生成中展现出强大的能力，但它们在早期去噪步骤中会产生过于自信的预测，导致指导梯度消失。本文提出了两种互补的贡献以解决这一问题。首先，我们提出了一种基于平滑期望校准误差（Smooth ECE）的可微校准目标，该目标在最少微调的情况下改善分类器校准，并在弗雷切尔 inception 距离（FID）上取得可测量的改进。其次，我们开发了增强的采样指导方法，这些方法不需要重新训练即可应用于现成的分类器。这些方法包括带有批次级重权的倾斜采样、适应性熵正则化采样以保持多样性，以及一种新的基于 f-散度的采样策略，该策略增强了类内一致的指导并保持模式覆盖。ImageNet 128x128 上的实验显示，我们的散度正则化指导在使用 ResNet-101 分类器时达到了 2.13 的 FID，优于现有的分类器指导动力学方法，且无需重新训练生成模型。结果表明，合理的校准和散度意识采样为分类器指导动力学提供了实用且有效的改进。 

---
# Understanding Cross Task Generalization in Handwriting-Based Alzheimer's Screening via Vision Language Adaptation 

**Title (ZH)**: 基于手写的手semantic记忆衰退跨任务泛化理解 via 视知觉语言适应 

**Authors**: Changqing Gong, Huafeng Qin, Mounim A. El-Yacoubi  

**Link**: [PDF](https://arxiv.org/pdf/2511.05841)  

**Abstract**: Alzheimer's disease is a prevalent neurodegenerative disorder for which early detection is critical. Handwriting-often disrupted in prodromal AD-provides a non-invasive and cost-effective window into subtle motor and cognitive decline. Existing handwriting-based AD studies, mostly relying on online trajectories and hand-crafted features, have not systematically examined how task type influences diagnostic performance and cross-task generalization. Meanwhile, large-scale vision language models have demonstrated remarkable zero or few-shot anomaly detection in natural images and strong adaptability across medical modalities such as chest X-ray and brain MRI. However, handwriting-based disease detection remains largely unexplored within this paradigm. To close this gap, we introduce a lightweight Cross-Layer Fusion Adapter framework that repurposes CLIP for handwriting-based AD screening. CLFA implants multi-level fusion adapters within the visual encoder to progressively align representations toward handwriting-specific medical cues, enabling prompt-free and efficient zero-shot inference. Using this framework, we systematically investigate cross-task generalization-training on a specific handwriting task and evaluating on unseen ones-to reveal which task types and writing patterns most effectively discriminate AD. Extensive analyses further highlight characteristic stroke patterns and task-level factors that contribute to early AD identification, offering both diagnostic insights and a benchmark for handwriting-based cognitive assessment. 

**Abstract (ZH)**: 阿尔茨海默病是一种常见的神经退行性疾病，早期检测至关重要。书写能力—在轻度认知障碍前期常受到影响—提供了一种无创且成本效益高的途径，用于洞察细微的运动和认知衰退。现有的基于书写行为的阿尔茨海默病研究主要依赖于在线轨迹和手工设计的特征，尚未系统地探讨任务类型对诊断性能和跨任务泛化的影响。同时，大规模的视觉语言模型在自然图像的零样本或少样本异常检测中取得了显著成果，并在胸部X光和脑MRI等医学成像领域展现了强大的适应能力。然而，基于书写行为的疾病检测在这个范式下仍很少被探索。为弥补这一空白，我们提出了一种轻量级的跨层融合适配器框架，重新利用CLIP进行基于书写的阿尔茨海默病筛查。CLFA在视觉编码器中植入多层融合适配器，逐步使表示向书写特定的医学线索对齐，从而实现无提示的高效零样本推理。利用这一框架，我们系统地研究了跨任务泛化能力—在特定的书写任务上训练并在未见过的任务上评估—以揭示哪些任务类型和书写模式最有效地鉴别阿尔茨海默病。详尽的分析进一步突显了有助于早期阿尔茨海默病识别的典型笔画模式和任务层面的因素，为基于书写的认知评估提供了诊断洞察和基准。 

---
# Hilbert-Guided Block-Sparse Local Attention 

**Title (ZH)**: Hilbert-Guided块稀疏局部注意力 

**Authors**: Yunge Li, Lanyu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05832)  

**Abstract**: The quadratic compute and memory costs of global self-attention severely limit its use in high-resolution images. Local attention reduces complexity by restricting attention to neighborhoods. Block-sparse kernels can further improve the efficiency of local attention, but conventional local attention patterns often fail to deliver significant speedups because tokens within a window are not contiguous in the 1D sequence. This work proposes a novel method for constructing windows and neighborhoods based on the Hilbert curve. Image tokens are first reordered along a Hilbert curve, and windows and neighborhoods are then formed on the reordered 1D sequence. From a block-sparse perspective, this strategy significantly increases block sparsity and can be combined with existing block-sparse kernels to improve the efficiency of 2D local attention. Experiments show that the proposed Hilbert Window Attention and Hilbert Slide Attention can accelerate window attention and slide attention by about $4\times$ and $18\times$, respectively. To assess practicality, the strategy is instantiated as the Hilbert Window Transformer and the Hilbert Neighborhood Transformer, both of which achieve end-to-end speedups with minimal accuracy loss. Overall, combining Hilbert-guided local attention with block-sparse kernels offers a general and practical approach to enhancing the efficiency of 2D local attention for images. The code is available at this https URL. 

**Abstract (ZH)**: 全球自注意力的二次计算和内存成本严重限制了其在高分辨率图像中的应用。局部注意力通过将注意力限制在邻域内来降低复杂性。块稀疏核可以进一步提高局部注意力的效率，但常规的局部注意力模式往往未能提供显著的加速，因为窗口内的标记在1D序列中并不连续。本研究提出了一种基于Hilbert曲线构建窗口和邻域的新方法。首先将图像标记沿Hilbert曲线重新排序，然后在重新排序的1D序列上形成窗口和邻域。从块稀疏的角度来看，这种策略显著增加了块稀疏性，并可与现有的块稀疏核结合以提高2D局部注意力的效率。实验结果显示，提出的Hilbert窗口注意力和Hilbert滑动注意力分别可以加速窗口注意力和滑动注意力约4倍和18倍。为了评估其实用性，该策略被实例化为Hilbert窗口变换器和Hilbert邻域变换器，两者都实现了端到端的速度提升且几乎没有准确率损失。总体而言，结合Hilbert引导的局部注意力与块稀疏核提供了一种通用且实用的方法，以提高2D局部注意力在图像中的效率。代码可在以下链接获取。 

---
# Policy Gradient-Based EMT-in-the-Loop Learning to Mitigate Sub-Synchronous Control Interactions 

**Title (ZH)**: 基于策略梯度的EMT在环学习以减轻次同步控制相互作用 

**Authors**: Sayak Mukherjee, Ramij R. Hossain, Kaustav Chatterjee, Sameer Nekkalapu, Marcelo Elizondo  

**Link**: [PDF](https://arxiv.org/pdf/2511.05822)  

**Abstract**: This paper explores the development of learning-based tunable control gains using EMT-in-the-loop simulation framework (e.g., PSCAD interfaced with Python-based learning modules) to address critical sub-synchronous oscillations. Since sub-synchronous control interactions (SSCI) arise from the mis-tuning of control gains under specific grid configurations, effective mitigation strategies require adaptive re-tuning of these gains. Such adaptiveness can be achieved by employing a closed-loop, learning-based framework that considers the grid conditions responsible for such sub-synchronous oscillations. This paper addresses this need by adopting methodologies inspired by Markov decision process (MDP) based reinforcement learning (RL), with a particular emphasis on simpler deep policy gradient methods with additional SSCI-specific signal processing modules such as down-sampling, bandpass filtering, and oscillation energy dependent reward computations. Our experimentation in a real-world event setting demonstrates that the deep policy gradient based trained policy can adaptively compute gain settings in response to varying grid conditions and optimally suppress control interaction-induced oscillations. 

**Abstract (ZH)**: 基于EMT-in-the-loop仿真框架的学习驱动可调控制增益开发，以应对亚同步振荡 

---
# WAR-Re: Web API Recommendation with Semantic Reasoning 

**Title (ZH)**: WAR-Re: 基于语义推理的Web API 推荐 

**Authors**: Zishuo Xu, Dezhong Yao, Yao Wan  

**Link**: [PDF](https://arxiv.org/pdf/2511.05820)  

**Abstract**: With the development of cloud computing, the number of Web APIs has increased dramatically, further intensifying the demand for efficient Web API recommendation. Despite the demonstrated success of previous Web API recommendation solutions, two critical challenges persist: 1) a fixed top-N recommendation that cannot accommodate the varying API cardinality requirements of different mashups, and 2) these methods output only ranked API lists without accompanying reasons, depriving users of understanding the recommendation. To address these challenges, we propose WAR-Re, an LLM-based model for Web API recommendation with semantic reasoning for justification. WAR-Re leverages special start and stop tokens to handle the first challenge and uses two-stage training: supervised fine-tuning and reinforcement learning via Group Relative Policy Optimization (GRPO) to enhance the model's ability in both tasks. Comprehensive experimental evaluations on the ProgrammableWeb dataset demonstrate that WAR-Re achieves a gain of up to 21.59\% over the state-of-the-art baseline model in recommendation accuracy, while consistently producing high-quality semantic reasons for recommendations. 

**Abstract (ZH)**: 基于语义推理的Web API推荐模型WAR-Re 

---
# In-depth Analysis on Caching and Pre-fetching in Mixture of Experts Offloading 

**Title (ZH)**: 混合专家卸载中缓存和预取的深度分析 

**Authors**: Shuning Lin, Yifan He, Yitong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.05814)  

**Abstract**: In today's landscape, Mixture of Experts (MoE) is a crucial architecture that has been used by many of the most advanced models. One of the major challenges of MoE models is that they usually require much more memory than their dense counterparts due to their unique architecture, and hence are harder to deploy in environments with limited GPU memory, such as edge devices. MoE offloading is a promising technique proposed to overcome this challenge, especially if it is enhanced with caching and pre-fetching, but prior work stopped at suboptimal caching algorithm and offered limited insights. In this work, we study MoE offloading in depth and make the following contributions: 1. We analyze the expert activation and LRU caching behavior in detail and provide traces. 2. We propose LFU caching optimization based on our analysis and obtain strong improvements from LRU. 3. We implement and experiment speculative expert pre-fetching, providing detailed trace showing its huge potential . 4. In addition, our study extensively covers the behavior of the MoE architecture itself, offering information on the characteristic of the gating network and experts. This can inspire future work on the interpretation of MoE models and the development of pruning techniques for MoE architecture with minimal performance loss. 

**Abstract (ZH)**: MoE模型卸载及其优化研究 

---
# MOSS: Efficient and Accurate FP8 LLM Training with Microscaling and Automatic Scaling 

**Title (ZH)**: MOSS：基于微缩和自动缩放的高效准确FP8 LLM训练 

**Authors**: Yu Zhang, Hui-Ling Zhen, Mingxuan Yuan, Bei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05811)  

**Abstract**: Training large language models with FP8 formats offers significant efficiency gains. However, the reduced numerical precision of FP8 poses challenges for stable and accurate training. Current frameworks preserve training performance using mixed-granularity quantization, i.e., applying per-group quantization for activations and per-tensor/block quantization for weights. While effective, per-group quantization requires scaling along the inner dimension of matrix multiplication, introducing additional dequantization overhead. Moreover, these frameworks often rely on just-in-time scaling to dynamically adjust scaling factors based on the current data distribution. However, this online quantization is inefficient for FP8 training, as it involves multiple memory reads and writes that negate the performance benefits of FP8. To overcome these limitations, we propose MOSS, a novel FP8 training framework that ensures both efficiency and numerical stability. MOSS introduces two key innovations: (1) a two-level microscaling strategy for quantizing sensitive activations, which balances precision and dequantization cost by combining a high-precision global scale with compact, power-of-two local scales; and (2) automatic scaling for weights in linear layers, which eliminates the need for costly max-reduction operations by predicting and adjusting scaling factors during training. Leveraging these techniques, MOSS enables efficient FP8 training of a 7B parameter model, achieving performance comparable to the BF16 baseline while achieving up to 34% higher training throughput. 

**Abstract (ZH)**: 使用FP8格式训练大型语言模型提供了显著的效率增益。然而，FP8数值精度的降低为训练的稳定性和准确性带来了挑战。当前框架通过分层级量化来保持训练性能，即对激活值使用分组量化，对权重使用张量级/块量化。虽然有效，但分组量化要求沿矩阵乘法的内部维度进行缩放，引入了额外的去量化开销。此外，这些框架通常依赖于即时缩放来根据当前数据分布动态调整缩放因子。然而，这种在线量化对于FP8训练来说是低效的，因为它涉及多次内存读写，抵消了FP8的性能优势。为了克服这些限制，我们提出了MOSS，一种新型的FP8训练框架，确保了效率和数值稳定性。MOSS引入了两项创新技术：（1）两级微缩放策略，通过结合高精度全局尺度和紧凑的2的幂本地尺度来平衡精度和去量化成本；（2）线性层权重的自动缩放，通过在训练过程中预测和调整缩放因子来消除昂贵的最大值归约操作。利用这些技术，MOSS能够高效地训练一个7B参数模型，性能与BF16基线相当，同时训练吞吐量提高高达34%。 

---
# Measuring Model Performance in the Presence of an Intervention 

**Title (ZH)**: 在干预存在下的模型性能测量 

**Authors**: Winston Chen, Michael W. Sjoding, Jenna Wiens  

**Link**: [PDF](https://arxiv.org/pdf/2511.05805)  

**Abstract**: AI models are often evaluated based on their ability to predict the outcome of interest. However, in many AI for social impact applications, the presence of an intervention that affects the outcome can bias the evaluation. Randomized controlled trials (RCTs) randomly assign interventions, allowing data from the control group to be used for unbiased model evaluation. However, this approach is inefficient because it ignores data from the treatment group. Given the complexity and cost often associated with RCTs, making the most use of the data is essential. Thus, we investigate model evaluation strategies that leverage all data from an RCT. First, we theoretically quantify the estimation bias that arises from naïvely aggregating performance estimates from treatment and control groups, and derive the condition under which this bias leads to incorrect model selection. Leveraging these theoretical insights, we propose nuisance parameter weighting (NPW), an unbiased model evaluation approach that reweights data from the treatment group to mimic the distributions of samples that would or would not experience the outcome under no intervention. Using synthetic and real-world datasets, we demonstrate that our proposed evaluation approach consistently yields better model selection than the standard approach, which ignores data from the treatment group, across various intervention effect and sample size settings. Our contribution represents a meaningful step towards more efficient model evaluation in real-world contexts. 

**Abstract (ZH)**: 基于RCT的AI模型评估策略：利用nuisance参数加权实现无偏评估 

---
# Beyond the Lower Bound: Bridging Regret Minimization and Best Arm Identification in Lexicographic Bandits 

**Title (ZH)**: 超越下界：在字典序 bandits 中最小化遗憾与最佳臂识别的桥梁 

**Authors**: Bo Xue, Yuanyu Wan, Zhichao Lu, Qingfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05802)  

**Abstract**: In multi-objective decision-making with hierarchical preferences, lexicographic bandits provide a natural framework for optimizing multiple objectives in a prioritized order. In this setting, a learner repeatedly selects arms and observes reward vectors, aiming to maximize the reward for the highest-priority objective, then the next, and so on. While previous studies have primarily focused on regret minimization, this work bridges the gap between \textit{regret minimization} and \textit{best arm identification} under lexicographic preferences. We propose two elimination-based algorithms to address this joint objective. The first algorithm eliminates suboptimal arms sequentially, layer by layer, in accordance with the objective priorities, and achieves sample complexity and regret bounds comparable to those of the best single-objective algorithms. The second algorithm simultaneously leverages reward information from all objectives in each round, effectively exploiting cross-objective dependencies. Remarkably, it outperforms the known lower bound for the single-objective bandit problem, highlighting the benefit of cross-objective information sharing in the multi-objective setting. Empirical results further validate their superior performance over baselines. 

**Abstract (ZH)**: 层级偏好下多目标决策中的字典序Bandits提供了优化优先级排序的目标的自然框架 

---
# When AI Meets the Web: Prompt Injection Risks in Third-Party AI Chatbot Plugins 

**Title (ZH)**: 当AI遇到网络：第三方AI聊天机器人插件中的提示注入风险 

**Authors**: Yigitcan Kaya, Anton Landerer, Stijn Pletinckx, Michelle Zimmermann, Christopher Kruegel, Giovanni Vigna  

**Link**: [PDF](https://arxiv.org/pdf/2511.05797)  

**Abstract**: Prompt injection attacks pose a critical threat to large language models (LLMs), with prior work focusing on cutting-edge LLM applications like personal copilots. In contrast, simpler LLM applications, such as customer service chatbots, are widespread on the web, yet their security posture and exposure to such attacks remain poorly understood. These applications often rely on third-party chatbot plugins that act as intermediaries to commercial LLM APIs, offering non-expert website builders intuitive ways to customize chatbot behaviors. To bridge this gap, we present the first large-scale study of 17 third-party chatbot plugins used by over 10,000 public websites, uncovering previously unknown prompt injection risks in practice. First, 8 of these plugins (used by 8,000 websites) fail to enforce the integrity of the conversation history transmitted in network requests between the website visitor and the chatbot. This oversight amplifies the impact of direct prompt injection attacks by allowing adversaries to forge conversation histories (including fake system messages), boosting their ability to elicit unintended behavior (e.g., code generation) by 3 to 8x. Second, 15 plugins offer tools, such as web-scraping, to enrich the chatbot's context with website-specific content. However, these tools do not distinguish the website's trusted content (e.g., product descriptions) from untrusted, third-party content (e.g., customer reviews), introducing a risk of indirect prompt injection. Notably, we found that ~13% of e-commerce websites have already exposed their chatbots to third-party content. We systematically evaluate both vulnerabilities through controlled experiments grounded in real-world observations, focusing on factors such as system prompt design and the underlying LLM. Our findings show that many plugins adopt insecure practices that undermine the built-in LLM safeguards. 

**Abstract (ZH)**: Prompt注入攻击对大型语言模型构成关键威胁：以个人副驾驶员类前沿应用为例，而简单的客户服务中心聊天机器人等应用广泛存在于网络上，但它们的安全状态和对抗此类攻击的暴露程度仍不甚理解。为了弥合这一差距，我们首次对10,000多个公开网站使用的17个第三方聊天机器人插件进行了大规模研究，揭示了实践中未知的聊天记录篡改风险。首先，8个插件（服务于8,000个网站）未能确保网站访客与聊天机器人之间网络请求中传递的对话历史的完整性。这种疏忽放大了直接聊天记录篡改攻击的影响，允许攻击者伪造对话历史（包括虚假系统消息），将其诱导未预期行为（例如代码生成）的能力提升3到8倍。其次，15个插件提供了一些工具，如网页抓取，以丰富聊天机器人的上下文，使其包含网站特定的内容。然而，这些工具无法区分受信任的内容（例如产品描述）与不受信任的第三方内容（例如客户评论），从而引入了潜在的间接聊天记录篡改风险。值得注意的是，我们发现约13%的电子商务网站已经让其聊天机器人暴露在第三方内容之下。通过基于实际观察的受控实验系统地评估这些脆弱性，重点关注系统提示设计和底层的大型语言模型等因素，我们的研究结果表明，许多插件采用了不安全的做法，从而削弱了内置的大型语言模型保护措施。 

---
# VLAD-Grasp: Zero-shot Grasp Detection via Vision-Language Models 

**Title (ZH)**: VLAD-Grasp: 通过视觉-语言模型实现零样本抓取检测 

**Authors**: Manav Kulshrestha, S. Talha Bukhari, Damon Conover, Aniket Bera  

**Link**: [PDF](https://arxiv.org/pdf/2511.05791)  

**Abstract**: Robotic grasping is a fundamental capability for autonomous manipulation; however, most existing methods rely on large-scale expert annotations and necessitate retraining to handle new objects. We present VLAD-Grasp, a Vision-Language model Assisted zero-shot approach for Detecting grasps. From a single RGB-D image, our method (1) prompts a large vision-language model to generate a goal image where a straight rod "impales" the object, representing an antipodal grasp, (2) predicts depth and segmentation to lift this generated image into 3D, and (3) aligns generated and observed object point clouds via principal component analysis and correspondence-free optimization to recover an executable grasp pose. Unlike prior work, our approach is training-free and does not rely on curated grasp datasets. Despite this, VLAD-Grasp achieves performance that is competitive with or superior to that of state-of-the-art supervised models on the Cornell and Jacquard datasets. We further demonstrate zero-shot generalization to novel real-world objects on a Franka Research 3 robot, highlighting vision-language foundation models as powerful priors for robotic manipulation. 

**Abstract (ZH)**: Vision-Language模型辅助的零样本夹取检测方法 

---
# SymLight: Exploring Interpretable and Deployable Symbolic Policies for Traffic Signal Control 

**Title (ZH)**: SymLight: 探索可解释且可部署的符号交通信号控制策略 

**Authors**: Xiao-Cheng Liao, Yi Mei, Mengjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05790)  

**Abstract**: Deep Reinforcement Learning have achieved significant success in automatically devising effective traffic signal control (TSC) policies. Neural policies, however, tend to be over-parameterized and non-transparent, hindering their interpretability and deployability on resource-limited edge devices. This work presents SymLight, a priority function search framework based on Monte Carlo Tree Search (MCTS) for discovering inherently interpretable and deployable symbolic priority functions to serve as the TSC policies. The priority function, in particular, accepts traffic features as input and then outputs a priority for each traffic signal phase, which subsequently directs the phase transition. For effective search, we propose a concise yet expressive priority function representation. This helps mitigate the combinatorial explosion of the action space in MCTS. Additionally, a probabilistic structural rollout strategy is introduced to leverage structural patterns from previously discovered high-quality priority functions, guiding the rollout process. Our experiments on real-world datasets demonstrate SymLight's superior performance across a range of baselines. A key advantage is SymLight's ability to produce interpretable and deployable TSC policies while maintaining excellent performance. 

**Abstract (ZH)**: 基于蒙特卡洛树搜索的优先级函数搜索框架SymLight：发现内在可解释且可部署的交通信号控制策略 

---
# DRAGON: Guard LLM Unlearning in Context via Negative Detection and Reasoning 

**Title (ZH)**: DRAGON: 在上下文中外包LLM脱忆的负检测与推理方法 

**Authors**: Yaxuan Wang, Chris Yuhao Liu, Quan Liu, Jinglong Pang, Wei Wei, Yujia Bao, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05784)  

**Abstract**: Unlearning in Large Language Models (LLMs) is crucial for protecting private data and removing harmful knowledge. Most existing approaches rely on fine-tuning to balance unlearning efficiency with general language capabilities. However, these methods typically require training or access to retain data, which is often unavailable in real world scenarios. Although these methods can perform well when both forget and retain data are available, few works have demonstrated equivalent capability in more practical, data-limited scenarios. To overcome these limitations, we propose Detect-Reasoning Augmented GeneratiON (DRAGON), a systematic, reasoning-based framework that utilizes in-context chain-of-thought (CoT) instructions to guard deployed LLMs before inference. Instead of modifying the base model, DRAGON leverages the inherent instruction-following ability of LLMs and introduces a lightweight detection module to identify forget-worthy prompts without any retain data. These are then routed through a dedicated CoT guard model to enforce safe and accurate in-context intervention. To robustly evaluate unlearning performance, we introduce novel metrics for unlearning performance and the continual unlearning setting. Extensive experiments across three representative unlearning tasks validate the effectiveness of DRAGON, demonstrating its strong unlearning capability, scalability, and applicability in practical scenarios. 

**Abstract (ZH)**: 大型语言模型中的忘记学习对于保护私人数据和去除有害知识至关重要。现有大多数方法依赖微调以平衡忘记学习效率和通用语言能力。然而，这些方法通常需要训练数据或访问原始数据，而在实际场景中这往往是不可用的。尽管当忘记和保留数据都可用时，这些方法可以表现出色，但在数据受限的实际场景中，很少有研究能够展示相当的能力。为了克服这些局限，我们提出了一种基于推理的检测增强生成框架DRAGON，该框架利用上下文中的链式思考指令，在推理前保护部署的LLM。DRAGON不修改基础模型，而是利用LLM的内置指令遵循能力，并引入一个轻量级的检测模块来识别值得遗忘的提示，而无需保留数据。然后，这些提示通过一个专门的CoT防护模型来进行干预，以确保安全和准确的上下文内干预。为了稳健地评估忘记学习性能，我们引入了新的评估指标，并在持续忘记学习设置中进行了评估。广泛的实验在三个代表性的忘记学习任务上验证了DRAGON的有效性，显示出其强大的忘记学习能力、可扩展性和在实际场景中的适用性。 

---
# Sign language recognition from skeletal data using graph and recurrent neural networks 

**Title (ZH)**: 基于图形和循环神经网络的骨架数据手语识别 

**Authors**: B. Mederos, J. Mejía, A. Medina-Reyes, Y. Espinosa-Almeyda, J. D. Díaz-Roman, I. Rodríguez-Mederos, M. Mejía-Carreon, F. Gonzalez-Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2511.05772)  

**Abstract**: This work presents an approach for recognizing isolated sign language gestures using skeleton-based pose data extracted from video sequences. A Graph-GRU temporal network is proposed to model both spatial and temporal dependencies between frames, enabling accurate classification. The model is trained and evaluated on the AUTSL (Ankara university Turkish sign language) dataset, achieving high accuracy. Experimental results demonstrate the effectiveness of integrating graph-based spatial representations with temporal modeling, providing a scalable framework for sign language recognition. The results of this approach highlight the potential of pose-driven methods for sign language understanding. 

**Abstract (ZH)**: 基于骨架数据的孤立手语手势识别方法：结合图GRU的时间网络在土耳其手语识别中的应用 

---
# Lived Experience in Dialogue: Co-designing Personalization in Large Language Models to Support Youth Mental Well-being 

**Title (ZH)**: 在对话中生活的体验：与青少年心理健康支持相关的大型语言模型个性化共同设计 

**Authors**: Kathleen W. Guan, Sarthak Giri, Mohammed Amara, Bernard J. Jansen, Enrico Liscio, Milena Esherick, Mohammed Al Owayyed, Ausrine Ratkute, Gayane Sedrakyan, Mark de Reuver, Joao Fernando Ferreira Goncalves, Caroline A. Figueroa  

**Link**: [PDF](https://arxiv.org/pdf/2511.05769)  

**Abstract**: Youth increasingly turn to large language models (LLMs) for mental well-being support, yet current personalization in LLMs can overlook the heterogeneous lived experiences shaping their needs. We conducted a participatory study with youth, parents, and youth care workers (N=38), using co-created youth personas as scaffolds, to elicit community perspectives on how LLMs can facilitate more meaningful personalization to support youth mental well-being. Analysis identified three themes: person-centered contextualization responsive to momentary needs, explicit boundaries around scope and offline referral, and dialogic scaffolding for reflection and autonomy. We mapped these themes to persuasive design features for task suggestions, social facilitation, and system trustworthiness, and created corresponding dialogue extracts to guide LLM fine-tuning. Our findings demonstrate how lived experience can be operationalized to inform design features in LLMs, which can enhance the alignment of LLM-based interventions with the realities of youth and their communities, contributing to more effectively personalized digital well-being tools. 

**Abstract (ZH)**: 青少年越来越多地利用大型语言模型（LLMs）寻求心理健康支持，但当前的个性化设置可能会忽视塑造其需求的异质性生活体验。我们通过与青少年、家长和青少年护理工作者（共38人）进行参与式研究，使用共创的青少年人物作为支架，探讨社区视角下LLMs如何促进更具有意义的个性化以支持青少年心理健康。分析识别出三个主题：以个人为中心的 contextualization，响应即时需要，明确的边界范围与离线转介，以及促进反思与自主性的对话支架。我们将这些主题映射到说服性设计特征的任务建议、社会促进和系统可信性上，并创建相应的对话提取以指导LLM调整。我们的研究结果展示了如何将生活体验具体化以指导LLMs的设计特征，这有助于使基于LLM的干预措施更契合青少年及其社区的实际情况，从而促进更加有效的个性化数字心理健康工具。 

---
# Language Generation: Complexity Barriers and Implications for Learning 

**Title (ZH)**: 语言生成：复杂性壁垒及其对学习的影响 

**Authors**: Marcelo Arenas, Pablo Barceló, Luis Cofré, Alexander Kozachinskiy  

**Link**: [PDF](https://arxiv.org/pdf/2511.05759)  

**Abstract**: Kleinberg and Mullainathan showed that, in principle, language generation is always possible: with sufficiently many positive examples, a learner can eventually produce sentences indistinguishable from those of a target language. However, the existence of such a guarantee does not speak to its practical feasibility. In this work, we show that even for simple and well-studied language families -- such as regular and context-free languages -- the number of examples required for successful generation can be extraordinarily large, and in some cases not bounded by any computable function. These results reveal a substantial gap between theoretical possibility and efficient learnability. They suggest that explaining the empirical success of modern language models requires a refined perspective -- one that takes into account structural properties of natural language that make effective generation possible in practice. 

**Abstract (ZH)**: Kleinberg和Mullainathan表明，在原则上，语言生成总是可能的：只要有足够的正面示例，学习者最终可以生成与目标语言无法区别的句子。然而，这种保证并不意味着其实用可行性。在本文中，我们展示了即使对于简单且已被充分研究的语言家族——如正则语言和上下文免费语言——成功生成所需的示例数量也可能极其庞大，在某些情况下甚至无法被任何可计算函数进行上界估算。这些结果揭示了理论可能性与高效可学习性之间存在显著差距。它们暗示了解释现代语言模型的实证成功需要一种更加精细的观点——这种观点需要考虑到自然语言的结构性特征，这些特征在实践中使得有效的生成成为可能。 

---
# Beyond Redundancy: Diverse and Specialized Multi-Expert Sparse Autoencoder 

**Title (ZH)**: 超越冗余：多样化和专门化的多专家稀疏自编码器 

**Authors**: Zhen Xu, Zhen Tan, Song Wang, Kaidi Xu, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.05745)  

**Abstract**: Sparse autoencoders (SAEs) have emerged as a powerful tool for interpreting large language models (LLMs) by decomposing token activations into combinations of human-understandable features. While SAEs provide crucial insights into LLM explanations, their practical adoption faces a fundamental challenge: better interpretability demands that SAEs' hidden layers have high dimensionality to satisfy sparsity constraints, resulting in prohibitive training and inference costs. Recent Mixture of Experts (MoE) approaches attempt to address this by partitioning SAEs into narrower expert networks with gated activation, thereby reducing computation. In a well-designed MoE, each expert should focus on learning a distinct set of features. However, we identify a \textit{critical limitation} in MoE-SAE: Experts often fail to specialize, which means they frequently learn overlapping or identical features. To deal with it, we propose two key innovations: (1) Multiple Expert Activation that simultaneously engages semantically weighted expert subsets to encourage specialization, and (2) Feature Scaling that enhances diversity through adaptive high-frequency scaling. Experiments demonstrate a 24\% lower reconstruction error and a 99\% reduction in feature redundancy compared to existing MoE-SAE methods. This work bridges the interpretability-efficiency gap in LLM analysis, allowing transparent model inspection without compromising computational feasibility. 

**Abstract (ZH)**: 基于专家的混合模型在稀疏自动编码器中的关键改进：提高大型语言模型解释的透明性和效率 

---
# Compressing Chemistry Reveals Functional Groups 

**Title (ZH)**: 压缩化学揭示功能团 

**Authors**: Ruben Sharma, Ross D. King  

**Link**: [PDF](https://arxiv.org/pdf/2511.05728)  

**Abstract**: We introduce the first formal large-scale assessment of the utility of traditional chemical functional groups as used in chemical explanations. Our assessment employs a fundamental principle from computational learning theory: a good explanation of data should also compress the data. We introduce an unsupervised learning algorithm based on the Minimum Message Length (MML) principle that searches for substructures that compress around three million biologically relevant molecules. We demonstrate that the discovered substructures contain most human-curated functional groups as well as novel larger patterns with more specific functions. We also run our algorithm on 24 specific bioactivity prediction datasets to discover dataset-specific functional groups. Fingerprints constructed from dataset-specific functional groups are shown to significantly outperform other fingerprint representations, including the MACCS and Morgan fingerprint, when training ridge regression models on bioactivity regression tasks. 

**Abstract (ZH)**: 我们介绍了第一个正式的大规模评估传统化学功能基团在化学解释中的实用性。该评估采用计算学习理论中的一个基本原理：良好的数据解释应该能够压缩数据。我们基于最小信息长度（MML）原则 introduce an unsupervised learning algorithm 搜索能够压缩大约三百万个生物相关分子的子结构。我们展示了发现的子结构中包含大多数人工整理的功能基团，以及具有更具体功能的新颖更大模式。我们还在24个具体的生物活性预测数据集中运行了该算法以发现数据集特异性功能基团。从数据集特异性功能基团构建的指纹图谱在基于岭回归模型的生物活性回归任务中显著优于其他指纹表示，包括MACCS和Morgan指纹图谱。 

---
# OckBench: Measuring the Efficiency of LLM Reasoning 

**Title (ZH)**: OckBench: 测量大语言模型推理效率 

**Authors**: Zheng Du, Hao Kang, Song Han, Tushar Krishna, Ligeng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05722)  

**Abstract**: Large language models such as GPT-4, Claude 3, and the Gemini series have improved automated reasoning and code generation. However, existing benchmarks mainly focus on accuracy and output quality, and they ignore an important factor: decoding token efficiency. In real systems, generating 10,000 tokens versus 100,000 tokens leads to large differences in latency, cost, and energy. In this work, we introduce OckBench, a model-agnostic and hardware-agnostic benchmark that evaluates both accuracy and token count for reasoning and coding tasks. Through experiments comparing multiple open- and closed-source models, we uncover that many models with comparable accuracy differ wildly in token consumption, revealing that efficiency variance is a neglected but significant axis of differentiation. We further demonstrate Pareto frontiers over the accuracy-efficiency plane and argue for an evaluation paradigm shift: we should no longer treat tokens as "free" to multiply. OckBench provides a unified platform for measuring, comparing, and guiding research in token-efficient reasoning. Our benchmarks are available at this https URL . 

**Abstract (ZH)**: 大型语言模型如GPT-4、Claude 3和Gemini系列在自动化推理和代码生成方面取得了进步。然而，现有的基准测试主要集中在准确性和输出质量上，而忽略了重要的一点：解码token效率。在实际系统中，生成10,000个token与生成100,000个token会导致延迟、成本和能耗方面的巨大差异。在本工作中，我们引入了OckBench，这是一个模型无关和硬件无关的基准，可以同时评估推理和编码任务的准确性和token数量。通过比较多个开源和闭源模型的实验，我们发现许多具有相似准确性的模型在token消耗上存在巨大差异，表明效率差异是一个被忽视但重要的区分因素。我们进一步展示了准确性和效率平面的帕累托前沿，并提出了一种评估范式的转变：我们不能再将tokens视为“免费”的东西来进行乘法操作。OckBench为衡量、比较和指导token高效推理的研究提供了一个统一平台。我们的基准测试可在以下链接获取：this https URL。 

---
# AdvisingWise: Supporting Academic Advising in Higher Educations Through a Human-in-the-Loop Multi-Agent Framework 

**Title (ZH)**: AdvisingWise: 通过人类参与的多agent框架支持高等教育中的学术指导 

**Authors**: Wendan Jiang, Shiyuan Wang, Hiba Eltigani, Rukhshan Haroon, Abdullah Bin Faisal, Fahad Dogar  

**Link**: [PDF](https://arxiv.org/pdf/2511.05706)  

**Abstract**: Academic advising is critical to student success in higher education, yet high student-to-advisor ratios limit advisors' capacity to provide timely support, particularly during peak periods. Recent advances in Large Language Models (LLMs) present opportunities to enhance the advising process. We present AdvisingWise, a multi-agent system that automates time-consuming tasks, such as information retrieval and response drafting, while preserving human oversight. AdvisingWise leverages authoritative institutional resources and adaptively prompts students about their academic backgrounds to generate reliable, personalized responses. All system responses undergo human advisor validation before delivery to students. We evaluate AdvisingWise through a mixed-methods approach: (1) expert evaluation on responses of 20 sample queries, (2) LLM-as-a-judge evaluation of the information retrieval strategy, and (3) a user study with 8 academic advisors to assess the system's practical utility. Our evaluation shows that AdvisingWise produces accurate, personalized responses. Advisors reported increasingly positive perceptions after using AdvisingWise, as their initial concerns about reliability and personalization diminished. We conclude by discussing the implications of human-AI synergy on the practice of academic advising. 

**Abstract (ZH)**: 学术指导是高等教育中学生成功的关键，但学生与导师的比例高限制了导师在高峰时期提供及时支持的能力。近年来，大型语言模型（LLMs）的进步为增强指导过程提供了机会。我们提出了AdvisingWise多代理系统，该系统自动化耗时的任务，如信息检索和回复起草，同时保留人类监督。AdvisingWise利用权威的校内资源，并根据学生的学术背景自适应地提出问题，以生成可靠且个性化的回复。所有系统回复均在交付给学生前经过人类导师的验证。我们通过混合方法评估AdvisingWise：（1）对20个示例查询的专家评估，（2）使用LLM作为评判者评估信息检索策略，以及（3）一项涉及8名学术导师的用户研究，以评估系统的实用性。我们的评估表明，AdvisingWise生成了准确且个性化的回复。导师在使用AdvisingWise后，对可靠性和个性化方面最初的担忧逐渐减弱，显得越来越积极。我们讨论了人类与AI协同作用对学术指导实践的影响。 

---
# Long Grounded Thoughts: Distilling Compositional Visual Reasoning Chains at Scale 

**Title (ZH)**: 长接地思考：大规模提炼组合视觉推理链 

**Authors**: David Acuna, Chao-Han Huck Yang, Yuntian Deng, Jaehun Jung, Ximing Lu, Prithviraj Ammanabrolu, Hyunwoo Kim, Yuan-Hong Liao, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2511.05705)  

**Abstract**: Recent progress in multimodal reasoning has been driven largely by undisclosed datasets and proprietary data synthesis recipes, leaving open questions about how to systematically build large-scale, vision-centric reasoning datasets, particularly for tasks that go beyond visual math. In this work, we introduce a new reasoning data generation framework spanning diverse skills and levels of complexity with over 1M high-quality synthetic vision-centric questions. The dataset also includes preference data and instruction prompts supporting both offline and online RL. Our synthesis framework proceeds in two stages: (1) scale; and (2) complexity. Reasoning traces are then synthesized through a two-stage process that leverages VLMs and reasoning LLMs, producing CoT traces for VLMs that capture the richness and diverse cognitive behaviors found in frontier reasoning models. Remarkably, we show that finetuning Qwen2.5-VL-7B on our data outperforms all open-data baselines across all evaluated vision-centric benchmarks, and even surpasses strong closed-data models such as MiMo-VL-7B-RL on V* Bench, CV-Bench and MMStar-V. Perhaps most surprising, despite being entirely vision-centric, our data transfers positively to text-only reasoning (MMLU-Pro) and audio reasoning (MMAU), demonstrating its effectiveness. Similarly, despite not containing videos or embodied visual data, we observe notable gains when evaluating on a single-evidence embodied QA benchmark (NiEH). Finally, we use our data to analyze the entire VLM post-training pipeline. Our empirical analysis highlights that (i) SFT on high-quality data with non-linear reasoning traces is essential for effective online RL, (ii) staged offline RL matches online RL's performance while reducing compute demands, and (iii) careful SFT on high quality data can substantially improve out-of-domain, cross-modality transfer. 

**Abstract (ZH)**: 近期多模态推理的进展主要得益于未披露的数据集和专有数据合成方法，对于如何系统地构建大规模、以视觉为中心的推理数据集，特别是对于超越视觉数学的任务，仍存在诸多开放问题。本文介绍了一种涵盖多种技能和复杂度层次的新推理数据生成框架，包含超过100万条高质量合成的以视觉为中心的问题。该数据集还包括支持离线和在线强化学习的偏好数据和指令提示。我们的合成框架分为两个阶段：(1) 规模；(2) 复杂性。推理轨迹通过一个两阶段过程进行合成，该过程利用了VLM和推理LLM，为VLM生成了捕捉前沿推理模型中丰富性和多样化认知行为的CoT轨迹。值得注意的是，我们将Qwen2.5-VL-7B微调后与我们的数据相比，表现优于所有公开数据基准，在V* Bench、CV-Bench和MMStar-V上甚至超过了强大的封闭数据模型MiMo-VL-7B-RL。更令人惊讶的是，尽管完全是视觉中心的，我们的数据在文本-only推理（MMLU-Pro）和音频推理（MMAU）上仍有积极的转移效果，证明了其有效性。尽管数据中未包含视频或嵌入式视觉数据，我们发现在单一证据嵌入式QA基准（NiEH）上的评估中仍观察到了显著的改进。最后，我们使用我们的数据来分析整个VLM后训练管道。我们的实证分析强调了以下几点：(i) 使用高质量数据和非线性推理轨迹的SFT对于有效的在线RL至关重要；(ii) 阶段化离线RL与在线RL性能相当，但降低了计算需求；(iii) 谨慎对高质量数据进行SFT可以显著提高跨域、跨模态的转移效果。 

---
# TabDistill: Distilling Transformers into Neural Nets for Few-Shot Tabular Classification 

**Title (ZH)**: TabDistill：将变压器模型精简为神经网络用于少量样本表格分类 

**Authors**: Pasan Dissanayake, Sanghamitra Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2511.05704)  

**Abstract**: Transformer-based models have shown promising performance on tabular data compared to their classical counterparts such as neural networks and Gradient Boosted Decision Trees (GBDTs) in scenarios with limited training data. They utilize their pre-trained knowledge to adapt to new domains, achieving commendable performance with only a few training examples, also called the few-shot regime. However, the performance gain in the few-shot regime comes at the expense of significantly increased complexity and number of parameters. To circumvent this trade-off, we introduce TabDistill, a new strategy to distill the pre-trained knowledge in complex transformer-based models into simpler neural networks for effectively classifying tabular data. Our framework yields the best of both worlds: being parameter-efficient while performing well with limited training data. The distilled neural networks surpass classical baselines such as regular neural networks, XGBoost and logistic regression under equal training data, and in some cases, even the original transformer-based models that they were distilled from. 

**Abstract (ZH)**: 基于Transformer的模型在有限训练数据场景下处理表格数据时，相较于经典模型如神经网络和梯度提升决策树（GBDTs）表现出有希望的性能。它们通过利用预训练知识适应新领域，在仅使用少量训练样本的情况下实现了出色的性能，称为少样本学习范式。然而，这种性能提升是以显著增加的模型复杂性和参数数量为代价的。为了克服这种权衡，我们提出了一种新的策略TabDistill，将复杂的Transformer模型中的预训练知识提炼到更简单的神经网络中，以有效地分类表格数据。我们的框架兼得两者之长：参数高效且在有限训练数据下表现优异。提炼后的神经网络在同等训练数据下优于传统的基线模型如普通神经网络、XGBoost和逻辑回归，并在某些情况下甚至超过了它们所提炼的原始Transformer模型。 

---
# Optimizing Diversity and Quality through Base-Aligned Model Collaboration 

**Title (ZH)**: 基于基模型对齐的多样性与质量优化合作方法 

**Authors**: Yichen Wang, Chenghao Yang, Tenghao Huang, Muhao Chen, Jonathan May, Mina Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.05650)  

**Abstract**: Alignment has greatly improved large language models (LLMs)' output quality at the cost of diversity, yielding highly similar outputs across generations. We propose Base-Aligned Model Collaboration (BACo), an inference-time token-level model collaboration framework that dynamically combines a base LLM with its aligned counterpart to optimize diversity and quality. Inspired by prior work (Fei et al., 2025), BACo employs routing strategies that determine, at each token, from which model to decode based on next-token prediction uncertainty and predicted contents' semantic role. Prior diversity-promoting methods, such as retraining, prompt engineering, and multi-sampling methods, improve diversity but often degrade quality or require costly decoding or post-training. In contrast, BACo achieves both high diversity and quality post hoc within a single pass, while offering strong controllability. We explore a family of routing strategies, across three open-ended generation tasks and 13 metrics covering diversity and quality, BACo consistently surpasses state-of-the-art inference-time baselines. With our best router, BACo achieves a 21.3% joint improvement in diversity and quality. Human evaluations also mirror these improvements. The results suggest that collaboration between base and aligned models can optimize and control diversity and quality. 

**Abstract (ZH)**: Base-Aligned Model Collaboration (BACo): Inference-Time Token-Level Model Collaboration for Optimizing Diversity and Quality 

---
# BrainCSD: A Hierarchical Consistency-Driven MoE Foundation Model for Unified Connectome Synthesis and Multitask Brain Trait Prediction 

**Title (ZH)**: BrainCSD：一种用于统一连合合成和多任务脑特质预测的分层一致性驱动的MoE基础模型 

**Authors**: Xiongri Shen, Jiaqi Wang, Yi Zhong, Zhenxi Song, Leilei Zhao, Liling Li, Yichen Wei, Lingyan Liang, Shuqiang Wang, Baiying Lei, Demao Deng, Zhiguo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05630)  

**Abstract**: Functional and structural connectivity (FC/SC) are key multimodal biomarkers for brain analysis, yet their clinical utility is hindered by costly acquisition, complex preprocessing, and frequent missing modalities. Existing foundation models either process single modalities or lack explicit mechanisms for cross-modal and cross-scale consistency. We propose BrainCSD, a hierarchical mixture-of-experts (MoE) foundation model that jointly synthesizes FC/SC biomarkers and supports downstream decoding tasks (diagnosis and prediction). BrainCSD features three neuroanatomically grounded components: (1) a ROI-specific MoE that aligns regional activations from canonical networks (e.g., DMN, FPN) with a global atlas via contrastive consistency; (2) a Encoding-Activation MOE that models dynamic cross-time/gradient dependencies in fMRI/dMRI; and (3) a network-aware refinement MoE that enforces structural priors and symmetry at individual and population levels. Evaluated on the datasets under complete and missing-modality settings, BrainCSD achieves SOTA results: 95.6\% accuracy for MCI vs. CN classification without FC, low synthesis error (FC RMSE: 0.038; SC RMSE: 0.006), brain age prediction (MAE: 4.04 years), and MMSE score estimation (MAE: 1.72 points). Code is available in \href{this https URL}{BrainCSD} 

**Abstract (ZH)**: 脑连接性（FC/SC）是脑分析中的关键多模态生物标志物，但由于获取成本高、预处理复杂以及频率较高的模态缺失，其临床应用受到阻碍。现有的基础模型要么处理单一模态，要么缺乏跨模态和跨尺度一致性机制。我们提出了一种分层混合专家（MoE）基础模型BrainCSD，它可以联合合成FC/SC生物标志物，并支持下游解码任务（诊断和预测）。BrainCSD具有三个神经解剖学基础的组件：（1）一种ROI特定的MoE，通过对比一致性将经典网络（如DMN、FPN）的区域激活与全局解剖图谱对齐；（2）一种编码-激活MoE，用于建模fMRI/dMRI中的动态跨时间和梯度依赖性；（3）一种网络感知的精细MoE，在个体和群体水平上施加结构先验和对称性约束。在完整和缺失模态的不同数据集上进行评估，BrainCSD达到了SOTA结果：在无FC的MCI vs. CN分类中准确率达到95.6%，低合成误差（FC RMSE: 0.038；SC RMSE: 0.006），脑龄预测（MAE: 4.04岁），以及MMSE评分估计（MAE: 1.72分）。代码可在BrainCSD获得。 

---
# SSTODE: Ocean-Atmosphere Physics-Informed Neural ODEs for Sea Surface Temperature Prediction 

**Title (ZH)**: SSTODE: 海洋-大气物理约束的神经ODEs 海表面温度预测 

**Authors**: Zheng Jiang, Wei Wang, Gaowei Zhang, Yi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05629)  

**Abstract**: Sea Surface Temperature (SST) is crucial for understanding upper-ocean thermal dynamics and ocean-atmosphere interactions, which have profound economic and social impacts. While data-driven models show promise in SST prediction, their black-box nature often limits interpretability and overlooks key physical processes. Recently, physics-informed neural networks have been gaining momentum but struggle with complex ocean-atmosphere dynamics due to 1) inadequate characterization of seawater movement (e.g., coastal upwelling) and 2) insufficient integration of external SST drivers (e.g., turbulent heat fluxes). To address these challenges, we propose SSTODE, a physics-informed Neural Ordinary Differential Equations (Neural ODEs) framework for SST prediction. First, we derive ODEs from fluid transport principles, incorporating both advection and diffusion to model ocean spatiotemporal dynamics. Through variational optimization, we recover a latent velocity field that explicitly governs the temporal dynamics of SST. Building upon ODE, we introduce an Energy Exchanges Integrator (EEI)-inspired by ocean heat budget equations-to account for external forcing factors. Thus, the variations in the components of these factors provide deeper insights into SST dynamics. Extensive experiments demonstrate that SSTODE achieves state-of-the-art performances in global and regional SST forecasting benchmarks. Furthermore, SSTODE visually reveals the impact of advection dynamics, thermal diffusion patterns, and diurnal heating-cooling cycles on SST evolution. These findings demonstrate the model's interpretability and physical consistency. 

**Abstract (ZH)**: 基于物理的神经常微分方程（SSTODE）在海表温度预测中的应用 

---
# Unveiling the Training Dynamics of ReLU Networks through a Linear Lens 

**Title (ZH)**: 通过线性视角揭示ReLU网络的训练动力学 

**Authors**: Longqing Ye  

**Link**: [PDF](https://arxiv.org/pdf/2511.05628)  

**Abstract**: Deep neural networks, particularly those employing Rectified Linear Units (ReLU), are often perceived as complex, high-dimensional, non-linear systems. This complexity poses a significant challenge to understanding their internal learning mechanisms. In this work, we propose a novel analytical framework that recasts a multi-layer ReLU network into an equivalent single-layer linear model with input-dependent "effective weights". For any given input sample, the activation pattern of ReLU units creates a unique computational path, effectively zeroing out a subset of weights in the network. By composing the active weights across all layers, we can derive an effective weight matrix, $W_{\text{eff}}(x)$, that maps the input directly to the output for that specific sample. We posit that the evolution of these effective weights reveals fundamental principles of representation learning. Our work demonstrates that as training progresses, the effective weights corresponding to samples from the same class converge, while those from different classes diverge. By tracking the trajectories of these sample-wise effective weights, we provide a new lens through which to interpret the formation of class-specific decision boundaries and the emergence of semantic representations within the network. 

**Abstract (ZH)**: 深层神经网络，特别是采用ReLU激活函数的网络，常常被视为复杂的、高维的非线性系统。这种复杂性对理解其内部学习机制构成了重大挑战。本文提出了一种新的分析框架，将多层ReLU网络重新表达为具有输入依赖性“有效权重”的等效单层线性模型。对于任何给定的输入样本，ReLU单元的激活模式创建了一条独特的计算路径，实际上零化了网络中的一个子集权重。通过组合各层中的活跃权重，可以推导出一个有效权重矩阵$W_{\text{eff}}(x)$，该矩阵将输入直接映射到该特定样本的输出。我们认为，这些有效权重的演变揭示了表示学习的基本原理。我们的研究表明，在训练过程中，来自同一类别的样本的有效权重趋于收敛，而来自不同类别的样本则趋向发散。通过追踪这些样本有效权重的轨迹，我们提供了一种新的视角来解释类别特定决策边界的形成以及网络中语义表示的涌现。 

---
# Assessing the Reliability of Large Language Models in the Bengali Legal Context: A Comparative Evaluation Using LLM-as-Judge and Legal Experts 

**Title (ZH)**: 大型语言模型在孟加拉国法律环境中的可靠性评估：基于LLM-as-Judge和法律专家的比较评价 

**Authors**: Sabik Aftahee, A.F.M. Farhad, Arpita Mallik, Ratnajit Dhar, Jawadul Karim, Nahiyan Bin Noor, Ishmam Ahmed Solaiman  

**Link**: [PDF](https://arxiv.org/pdf/2511.05627)  

**Abstract**: Accessing legal help in Bangladesh is hard. People face high fees, complex legal language, a shortage of lawyers, and millions of unresolved court cases. Generative AI models like OpenAI GPT-4.1 Mini, Gemini 2.0 Flash, Meta Llama 3 70B, and DeepSeek R1 could potentially democratize legal assistance by providing quick and affordable legal advice. In this study, we collected 250 authentic legal questions from the Facebook group "Know Your Rights," where verified legal experts regularly provide authoritative answers. These questions were subsequently submitted to four four advanced AI models and responses were generated using a consistent, standardized prompt. A comprehensive dual evaluation framework was employed, in which a state-of-the-art LLM model served as a judge, assessing each AI-generated response across four critical dimensions: factual accuracy, legal appropriateness, completeness, and clarity. Following this, the same set of questions was evaluated by three licensed Bangladeshi legal professionals according to the same criteria. In addition, automated evaluation metrics, including BLEU scores, were applied to assess response similarity. Our findings reveal a complex landscape where AI models frequently generate high-quality, well-structured legal responses but also produce dangerous misinformation, including fabricated case citations, incorrect legal procedures, and potentially harmful advice. These results underscore the critical need for rigorous expert validation and comprehensive safeguards before AI systems can be safely deployed for legal consultation in Bangladesh. 

**Abstract (ZH)**: 孟加拉国获取法律帮助困難。人们面临高昂的费用、复杂的法律语言、律师短缺以及数以百万计未解决的案件。像OpenAI GPT-4.1 Mini、Gemini 2.0 Flash、Meta Llama 3 70B和DeepSeek R1这样的生成式AI模型有可能通过提供快速且经济实惠的法律咨询来民主化法律援助。在本研究中，我们收集了来自Facebook群组“了解你的权利”中的250个真实法律问题，该群组中定期提供权威答案的经验证法律专家频繁活跃。随后，这些问题被提交给四个先进的AI模型，并使用一致的标准提示生成响应。采用了一种全面的双重评估框架，其中最先进的语言模型作为法官，根据事实准确度、法律适用性、完整性和清晰度四个关键维度评估每个AI生成的响应。然后，根据相同的标准，由三位持照孟加拉国法律专业人士对同一组问题进行了评估。此外，还应用了自动化评估指标，包括BLEU分数，来评估响应的相似性。我们的发现揭示了一个复杂的情景，其中AI模型经常生成高质量且结构良好的法律回应，但也产生危险的错误信息，包括伪造的案例引用、错误的法律程序以及可能有害的建议。这些结果强调，在AI系统能够在孟加拉国安全用于法律咨询之前，严格的专家验证和全面的保护措施至关重要。 

---
# LLMs as Packagers of HPC Software 

**Title (ZH)**: LLMs作为HPC软件的打包工具 

**Authors**: Caetano Melone, Daniel Nichols, Konstantinos Parasyris, Todd Gamblin, Harshitha Menon  

**Link**: [PDF](https://arxiv.org/pdf/2511.05626)  

**Abstract**: High performance computing (HPC) software ecosystems are inherently heterogeneous, comprising scientific applications that depend on hundreds of external packages, each with distinct build systems, options, and dependency constraints. Tools such as Spack automate dependency resolution and environment management, but their effectiveness relies on manually written build recipes. As these ecosystems grow, maintaining existing specifications and creating new ones becomes increasingly labor-intensive. While large language models (LLMs) have shown promise in code generation, automatically producing correct and maintainable Spack recipes remains a significant challenge. We present a systematic analysis of how LLMs and context-augmentation methods can assist in the generation of Spack recipes. To this end, we introduce SpackIt, an end-to-end framework that combines repository analysis, retrieval of relevant examples, and iterative refinement through diagnostic feedback. We apply SpackIt to a representative subset of 308 open-source HPC packages to assess its effectiveness and limitations. Our results show that SpackIt increases installation success from 20% in a zero-shot setting to over 80% in its best configuration, demonstrating the value of retrieval and structured feedback for reliable package synthesis. 

**Abstract (ZH)**: 高性能计算（HPC）软件生态系统本质上是异构的，包含依赖数百个外部包的科学应用，每个包都有不同的构建系统、选项和依赖约束。诸如Spack之类的工具可以通过自动化依赖解析和环境管理来提高效率，但它们的效果依赖于手动编写的构建脚本。随着这些生态系统的扩展，维护现有规格并创建新规格变得越来越 labor-intensive。虽然大型语言模型（LLMs）在代码生成方面显示出了潜力，但自动生成正确的且易于维护的Spack脚本仍然是一个重大挑战。我们呈现了一种系统分析方法，探讨了如何利用大型语言模型和背景增强方法来辅助生成Spack脚本。为此，我们引入了SpackIt，这是一种端到端框架，结合了仓库分析、相关示例检索和通过诊断反馈进行的迭代优化。我们对308个开源HPC包的代表性子集应用SpackIt，以评估其效果和局限性。结果显示，与零样本设置下20%的成功率相比，SpackIt的最佳配置下安装成功率超过80%，这证明了检索和结构化反馈对于可靠包合成的价值。 

---
# Report from Workshop on Dialogue alongside Artificial Intelligence 

**Title (ZH)**: 人工智能对话研讨会报告 

**Authors**: Thomas J McKenna, Ingvill Rasmussen, Sten Ludvigsen, Avivit Arvatz, Christa Asterhan, Gaowei Chen, Julie Cohen, Michele Flammia, Dongkeun Han, Emma Hayward, Heather Hill, Yifat Kolikant, Helen Lehndorf, Kexin Li, Lindsay Clare Matsumura, Henrik Tjønn, Pengjin Wang, Rupert Wegerif  

**Link**: [PDF](https://arxiv.org/pdf/2511.05625)  

**Abstract**: Educational dialogue -the collaborative exchange of ideas through talk- is widely recognized as a catalyst for deeper learning and critical thinking in and across contexts. At the same time, artificial intelligence (AI) has rapidly emerged as a powerful force in education, with the potential to address major challenges, personalize learning, and innovate teaching practices. However, these advances come with significant risks: rapid AI development can undermine human agency, exacerbate inequities, and outpace our capacity to guide its use with sound policy. Human learning presupposes cognitive efforts and social interaction (dialogues). In response to this evolving landscape, an international workshop titled "Educational Dialogue: Moving Thinking Forward" convened 19 leading researchers from 11 countries in Cambridge (September 1-3, 2025) to examine the intersection of AI and educational dialogue. This AI-focused strand of the workshop centered on three critical questions: (1) When is AI truly useful in education, and when might it merely replace human effort at the expense of learning? (2) Under what conditions can AI use lead to better dialogic teaching and learning? (3) Does the AI-human partnership risk outpacing and displacing human educational work, and what are the implications? These questions framed two days of presentations and structured dialogue among participants. 

**Abstract (ZH)**: 教育对话：促进思维的发展——人工智能与教育对话的交汇探究 

---
# Grounding Foundational Vision Models with 3D Human Poses for Robust Action Recognition 

**Title (ZH)**: 基于3D人体姿态的奠基性视觉模型 grounding 用于稳健的动作识别 

**Authors**: Nicholas Babey, Tiffany Gu, Yiheng Li, Cristian Meo, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05622)  

**Abstract**: For embodied agents to effectively understand and interact within the world around them, they require a nuanced comprehension of human actions grounded in physical space. Current action recognition models, often relying on RGB video, learn superficial correlations between patterns and action labels, so they struggle to capture underlying physical interaction dynamics and human poses in complex scenes. We propose a model architecture that grounds action recognition in physical space by fusing two powerful, complementary representations: V-JEPA 2's contextual, predictive world dynamics and CoMotion's explicit, occlusion-tolerant human pose data. Our model is validated on both the InHARD and UCF-19-Y-OCC benchmarks for general action recognition and high-occlusion action recognition, respectively. Our model outperforms three other baselines, especially within complex, occlusive scenes. Our findings emphasize a need for action recognition to be supported by spatial understanding instead of statistical pattern recognition. 

**Abstract (ZH)**: 基于物理空间的肢体动作识别模型：结合情境预测的世界动态与显式姿态数据 

---
# Frequency Matters: When Time Series Foundation Models Fail Under Spectral Shift 

**Title (ZH)**: 频率决定一切：当时间序列基础模型在频谱偏移情况下失效时 

**Authors**: Tianze Wang, Sofiane Ennadir, John Pertoft, Gabriela Zarzar Gandler, Lele Cao, Zineb Senane, Styliani Katsarou, Sahar Asadi, Axel Karlsson, Oleg Smirnov  

**Link**: [PDF](https://arxiv.org/pdf/2511.05619)  

**Abstract**: Time series foundation models (TSFMs) have shown strong results on public benchmarks, prompting comparisons to a "BERT moment" for time series. Their effectiveness in industrial settings, however, remains uncertain. We examine why TSFMs often struggle to generalize and highlight spectral shift (a mismatch between the dominant frequency components in downstream tasks and those represented during pretraining) as a key factor. We present evidence from an industrial-scale player engagement prediction task in mobile gaming, where TSFMs underperform domain-adapted baselines. To isolate the mechanism, we design controlled synthetic experiments contrasting signals with seen versus unseen frequency bands, observing systematic degradation under spectral mismatch. These findings position frequency awareness as critical for robust TSFM deployment and motivate new pretraining and evaluation protocols that explicitly account for spectral diversity. 

**Abstract (ZH)**: 时间序列基础模型在工业应用中的频谱意识与泛化能力：挑战与对策 

---
# Personalized Image Editing in Text-to-Image Diffusion Models via Collaborative Direct Preference Optimization 

**Title (ZH)**: 基于协作直接偏好优化的个性化图像编辑在文本到图像扩散模型中 

**Authors**: Connor Dunlop, Matthew Zheng, Kavana Venkatesh, Pinar Yanardag  

**Link**: [PDF](https://arxiv.org/pdf/2511.05616)  

**Abstract**: Text-to-image (T2I) diffusion models have made remarkable strides in generating and editing high-fidelity images from text. Yet, these models remain fundamentally generic, failing to adapt to the nuanced aesthetic preferences of individual users. In this work, we present the first framework for personalized image editing in diffusion models, introducing Collaborative Direct Preference Optimization (C-DPO), a novel method that aligns image edits with user-specific preferences while leveraging collaborative signals from like-minded individuals. Our approach encodes each user as a node in a dynamic preference graph and learns embeddings via a lightweight graph neural network, enabling information sharing across users with overlapping visual tastes. We enhance a diffusion model's editing capabilities by integrating these personalized embeddings into a novel DPO objective, which jointly optimizes for individual alignment and neighborhood coherence. Comprehensive experiments, including user studies and quantitative benchmarks, demonstrate that our method consistently outperforms baselines in generating edits that are aligned with user preferences. 

**Abstract (ZH)**: 面向用户的文本到图像扩散模型个性化图像编辑框架：协作直接偏好优化(C-DPO) 

---
# wa-hls4ml: A Benchmark and Surrogate Models for hls4ml Resource and Latency Estimation 

**Title (ZH)**: wa-hls4ml：用于hls4ml资源和延迟估计的基准模型与代理模型 

**Authors**: Benjamin Hawks, Jason Weitz, Dmitri Demler, Karla Tame-Narvaez, Dennis Plotnikov, Mohammad Mehdi Rahimifar, Hamza Ezzaoui Rahali, Audrey C. Therrien, Donovan Sproule, Elham E Khoda, Keegan A. Smith, Russell Marroquin, Giuseppe Di Guglielmo, Nhan Tran, Javier Duarte, Vladimir Loncar  

**Link**: [PDF](https://arxiv.org/pdf/2511.05615)  

**Abstract**: As machine learning (ML) is increasingly implemented in hardware to address real-time challenges in scientific applications, the development of advanced toolchains has significantly reduced the time required to iterate on various designs. These advancements have solved major obstacles, but also exposed new challenges. For example, processes that were not previously considered bottlenecks, such as hardware synthesis, are becoming limiting factors in the rapid iteration of designs. To mitigate these emerging constraints, multiple efforts have been undertaken to develop an ML-based surrogate model that estimates resource usage of ML accelerator architectures. We introduce wa-hls4ml, a benchmark for ML accelerator resource and latency estimation, and its corresponding initial dataset of over 680,000 fully connected and convolutional neural networks, all synthesized using hls4ml and targeting Xilinx FPGAs. The benchmark evaluates the performance of resource and latency predictors against several common ML model architectures, primarily originating from scientific domains, as exemplar models, and the average performance across a subset of the dataset. Additionally, we introduce GNN- and transformer-based surrogate models that predict latency and resources for ML accelerators. We present the architecture and performance of the models and find that the models generally predict latency and resources for the 75% percentile within several percent of the synthesized resources on the synthetic test dataset. 

**Abstract (ZH)**: 随着机器学习（ML）在硬件中越来越多地应用于科学应用中的实时挑战，高级工具链的发展显著减少了各种设计的迭代时间。尽管这些进步解决了许多主要障碍，但也暴露了新的挑战。例如，以前未被视为瓶颈的过程，如硬件综合，现在成为了设计快速迭代的限制因素。为了缓解这些新兴的限制，已经开展了多个努力来开发基于ML的代理模型，以估算ML加速器架构的资源使用情况。我们介绍了wa-hls4ml，这是一个用于评估ML加速器资源和延迟预测性能的基准，以及包含超过680,000个全连接和卷积神经网络的初始数据集，所有这些模型均使用hls4ml合成并针对Xilinx FPGA。基准测试评估了资源和延迟预测器对来自科学领域的多种常见ML模型架构的性能，并展示了数据集子集的平均性能。此外，我们介绍了基于GNN和变压器的代理模型，以预测ML加速器的延迟和资源。我们展示了这些模型的架构和性能，并发现这些模型通常在合成测试数据集上的合成资源中预测75%分位数的延迟和资源误差在几个百分点以内。 

---
# An MLCommons Scientific Benchmarks Ontology 

**Title (ZH)**: MLCommons 科学基准本体 

**Authors**: Ben Hawks, Gregor von Laszewski, Matthew D. Sinclair, Marco Colombo, Shivaram Venkataraman, Rutwik Jain, Yiwei Jiang, Nhan Tran, Geoffrey Fox  

**Link**: [PDF](https://arxiv.org/pdf/2511.05614)  

**Abstract**: Scientific machine learning research spans diverse domains and data modalities, yet existing benchmark efforts remain siloed and lack standardization. This makes novel and transformative applications of machine learning to critical scientific use-cases more fragmented and less clear in pathways to impact. This paper introduces an ontology for scientific benchmarking developed through a unified, community-driven effort that extends the MLCommons ecosystem to cover physics, chemistry, materials science, biology, climate science, and more. Building on prior initiatives such as XAI-BENCH, FastML Science Benchmarks, PDEBench, and the SciMLBench framework, our effort consolidates a large set of disparate benchmarks and frameworks into a single taxonomy of scientific, application, and system-level benchmarks. New benchmarks can be added through an open submission workflow coordinated by the MLCommons Science Working Group and evaluated against a six-category rating rubric that promotes and identifies high-quality benchmarks, enabling stakeholders to select benchmarks that meet their specific needs. The architecture is extensible, supporting future scientific and AI/ML motifs, and we discuss methods for identifying emerging computing patterns for unique scientific workloads. The MLCommons Science Benchmarks Ontology provides a standardized, scalable foundation for reproducible, cross-domain benchmarking in scientific machine learning. A companion webpage for this work has also been developed as the effort evolves: this https URL 

**Abstract (ZH)**: 科学机器学习研究涵盖了多样化的领域和数据模态，但现有的基准测试努力仍然各自为营且缺乏标准化。这使得将机器学习应用于关键科学场景的新颖和变革性应用更加碎片化且实现路径不够清晰。本文介绍了一种通过统一的、共同体驱动的努力开发的科学基准测试本体，该本体扩展了MLCommons生态系统，涵盖物理学、化学、材料科学、生物学、气候科学等更多领域。基于此前的倡议，如XAI-BENCH、FastML Science Benchmarks、PDEBench和SciMLBench框架，我们的努力将大量分散的基准测试和框架整合为一个单一的科学、应用和系统层面基准的分类体系。新的基准可以通过MLCommons Science Working Group协调的开放提交流程添加，并根据包含六个类别评估标准的评分框架进行评估，以促进和识别高质量的基准，使利益相关者能够选择满足其特定需求的基准。该架构可通过未来科学和AI/ML主题的扩展，本文还讨论了识别独特科学工作负载新型计算模式的方法。MLCommons Science Benchmarks本体为科学机器学习的可重复、跨域基准测试提供了一个标准化、可扩展的基础。随着该努力的演进，还开发了一个配套网页：this https URL。 

---
# Who Evaluates AI's Social Impacts? Mapping Coverage and Gaps in First and Third Party Evaluations 

**Title (ZH)**: 谁评估AI的社会影响？第一方和第三方评估的覆盖范围与缺口mapping研究 

**Authors**: Anka Reuel, Avijit Ghosh, Jenny Chim, Andrew Tran, Yanan Long, Jennifer Mickel, Usman Gohar, Srishti Yadav, Pawan Sasanka Ammanamanchi, Mowafak Allaham, Hossein A. Rahmani, Mubashara Akhtar, Felix Friedrich, Robert Scholz, Michael Alexander Riegler, Jan Batzner, Eliya Habba, Arushi Saxena, Anastassia Kornilova, Kevin Wei, Prajna Soni, Yohan Mathew, Kevin Klyman, Jeba Sania, Subramanyam Sahoo, Olivia Beyer Bruvik, Pouya Sadeghi, Sujata Goswami, Angelina Wang, Yacine Jernite, Zeerak Talat, Stella Biderman, Mykel Kochenderfer, Sanmi Koyejo, Irene Solaiman  

**Link**: [PDF](https://arxiv.org/pdf/2511.05613)  

**Abstract**: Foundation models are increasingly central to high-stakes AI systems, and governance frameworks now depend on evaluations to assess their risks and capabilities. Although general capability evaluations are widespread, social impact assessments covering bias, fairness, privacy, environmental costs, and labor practices remain uneven across the AI ecosystem. To characterize this landscape, we conduct the first comprehensive analysis of both first-party and third-party social impact evaluation reporting across a wide range of model developers. Our study examines 186 first-party release reports and 183 post-release evaluation sources, and complements this quantitative analysis with interviews of model developers. We find a clear division of evaluation labor: first-party reporting is sparse, often superficial, and has declined over time in key areas such as environmental impact and bias, while third-party evaluators including academic researchers, nonprofits, and independent organizations provide broader and more rigorous coverage of bias, harmful content, and performance disparities. However, this complementarity has limits. Only model developers can authoritatively report on data provenance, content moderation labor, financial costs, and training infrastructure, yet interviews reveal that these disclosures are often deprioritized unless tied to product adoption or regulatory compliance. Our findings indicate that current evaluation practices leave major gaps in assessing AI's societal impacts, highlighting the urgent need for policies that promote developer transparency, strengthen independent evaluation ecosystems, and create shared infrastructure to aggregate and compare third-party evaluations in a consistent and accessible way. 

**Abstract (ZH)**: 基础模型在高风险AI系统中日益重要，评估框架现在依赖评估来评估其风险和能力。虽然普遍开展了通用能力评估，但覆盖偏见、公平性、隐私、环境成本和劳动实践的社会影响评估在AI生态系统中仍不够均衡。为了刻画这一景观，我们首次对该广泛范围内的模型开发者的第一方和第三方社会影响评估报告进行了全面分析。我们的研究检查了186份第一方发布报告和183份发布后评估来源，并通过采访模型开发者补充了定量分析。我们发现评估劳动存在明确的分工：第一方报告稀缺且往往肤浅，随着时间的推移在诸如环境影响和偏见等领域减少，而包括学术研究人员、非营利组织和独立组织在内的第三方评估者提供了更广泛和更严格的偏见、有害内容和性能差异覆盖。然而，这种互补性有其局限性。只有模型开发者才能权威地报告数据来源、内容审查劳动、财务成本和训练基础设施，但采访显示，除非与产品采用或合规性相关，否则这些披露往往被优先级降低。我们的研究结果表明，当前的评估做法在评估AI的社会影响方面留有重大空白，强调了促进开发者透明度、强化独立评估生态系统和创建用于汇集和比较第三方评估共享基础架构的迫切需要，并以一致和易于访问的方式呈现。 

---
# AI-Enhanced High-Density NIRS Patch for Real-Time Brain Layer Oxygenation Monitoring in Neurological Emergencies 

**Title (ZH)**: AI增强高密度近红外光斑用于神经急症实时脑层氧合监测 

**Authors**: Minsu Ji, Jihoon Kang, Seongkwon Yu, Jaemyoung Kim, Bumjun Koh, Jimin Lee, Guil Jeong, Jongkwan choi, Chang-Ho Yun, Hyeonmin Bae  

**Link**: [PDF](https://arxiv.org/pdf/2511.05612)  

**Abstract**: Photon scattering has traditionally limited the ability of near-infrared spectroscopy (NIRS) to extract accurate, layer-specific information from the brain. This limitation restricts its clinical utility for precise neurological monitoring. To address this, we introduce an AI-driven, high-density NIRS system optimized to provide real-time, layer-specific oxygenation data from the brain cortex, specifically targeting acute neuro-emergencies. Our system integrates high-density NIRS reflectance data with a neural network trained on MRI-based synthetic datasets. This approach achieves robust cortical oxygenation accuracy across diverse anatomical variations. In simulations, our AI-assisted NIRS demonstrated a strong correlation (R2=0.913) with actual cortical oxygenation, markedly outperforming conventional methods (R2=0.469). Furthermore, biomimetic phantom experiments confirmed its superior anatomical reliability (R2=0.986) compared to standard commercial devices (R2=0.823). In clinical validation with healthy subjects and ischemic stroke patients, the system distinguished between the two groups with an AUC of 0.943. This highlights its potential as an accessible, high-accuracy diagnostic tool for emergency and point-of-care settings. These results underscore the system's capability to advance neuro-monitoring precision through AI, enabling timely, data-driven decisions in critical care environments. 

**Abstract (ZH)**: 基于AI的高密度近红外光谱系统：实时提供大脑皮层氧合数据以应对急性神经紧急情况 

---
# Conformal Prediction-Driven Adaptive Sampling for Digital Twins of Water Distribution Networks 

**Title (ZH)**: 基于齐性预测的自适应采样方法用于水分配网络的数字孪生模型 

**Authors**: Mohammadhossein Homaei, Oscar Mogollon Gutierrez, Ruben Molano, Andres Caro, Mar Avila  

**Link**: [PDF](https://arxiv.org/pdf/2511.05610)  

**Abstract**: Digital Twins (DTs) for Water Distribution Networks (WDNs) require accurate state estimation with limited sensors. Uniform sampling often wastes resources across nodes with different uncertainty. We propose an adaptive framework combining LSTM forecasting and Conformal Prediction (CP) to estimate node-wise uncertainty and focus sensing on the most uncertain points. Marginal CP is used for its low computational cost, suitable for real-time DTs. Experiments on Hanoi, Net3, and CTOWN show 33-34% lower demand error than uniform sampling at 40% coverage and maintain 89.4-90.2% empirical coverage with only 5-10% extra computation. 

**Abstract (ZH)**: 数字孪生（DTs）在供水管网（WDNs）中的状态估计需要在有限的传感器情况下实现准确的估计。均匀采样通常会在不同不确定性节点间浪费资源。我们提出了一种结合LSTM预测和可信区间预测（CP）的自适应框架，以估计节点级不确定性并集中在最不确定的点上进行传感。边缘CP因其计算成本低，适合用于实时的数字孪生。实验表明，在40%覆盖率下，与均匀采样相比，需求误差降低33-34%，且只额外增加5-10%的计算量即可保持89.4-90.2%的经验覆盖。 

---
# Walking the Schrödinger Bridge: A Direct Trajectory for Text-to-3D Generation 

**Title (ZH)**: 薛定谔桥上的行走：文本到3D生成的直接轨迹 

**Authors**: Ziying Li, Xuequan Lu, Xinkui Zhao, Guanjie Cheng, Shuiguang Deng, Jianwei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2511.05609)  

**Abstract**: Recent advancements in optimization-based text-to-3D generation heavily rely on distilling knowledge from pre-trained text-to-image diffusion models using techniques like Score Distillation Sampling (SDS), which often introduce artifacts such as over-saturation and over-smoothing into the generated 3D assets. In this paper, we address this essential problem by formulating the generation process as learning an optimal, direct transport trajectory between the distribution of the current rendering and the desired target distribution, thereby enabling high-quality generation with smaller Classifier-free Guidance (CFG) values. At first, we theoretically establish SDS as a simplified instance of the Schrödinger Bridge framework. We prove that SDS employs the reverse process of an Schrödinger Bridge, which, under specific conditions (e.g., a Gaussian noise as one end), collapses to SDS's score function of the pre-trained diffusion model. Based upon this, we introduce Trajectory-Centric Distillation (TraCe), a novel text-to-3D generation framework, which reformulates the mathematically trackable framework of Schrödinger Bridge to explicitly construct a diffusion bridge from the current rendering to its text-conditioned, denoised target, and trains a LoRA-adapted model on this trajectory's score dynamics for robust 3D optimization. Comprehensive experiments demonstrate that TraCe consistently achieves superior quality and fidelity to state-of-the-art techniques. 

**Abstract (ZH)**: 基于优化的文本到3D生成最近进展很大程度上依赖于通过Score Distillation Sampling (SDS)等技术从预训练的文本到图像扩散模型中提炼知识，这通常会在生成的3D资产中引入过度饱和和过度平滑等_artifacts_。本文通过将生成过程建模为在当前渲染分布与期望目标分布之间学习最优的直接传输轨迹，从而解决了这一关键问题，进而能够使用较小的Classifier-free Guidance (CFG)值实现高质量生成。首先，我们从理论上将SDS确立为Schrödinger Bridge框架的一个简化实例。我们证明了SDS实际上是Schrödinger Bridge的逆过程，在特定条件下（例如，高斯噪声作为一端）会退化为预训练扩散模型的SDS分数函数。基于此，我们引入了Trajectory-Centric Distillation (TraCe)，一种新型文本到3D生成框架，通过重新构建可数学跟踪的Schrödinger Bridge框架，明确构建从当前渲染到其文本条件下的去噪目标的扩散桥梁，并在该轨迹的得分动力学上训练一个LoRA调整模型，实现稳健的3D优化。综合实验表明，TraCe在质量与保真度方面明显优于现有最先进的技术。 

---
# Google-MedGemma Based Abnormality Detection in Musculoskeletal radiographs 

**Title (ZH)**: 基于Google-MedGemma的肌骨放射影像异常检测 

**Authors**: Soumyajit Maity, Pranjal Kamboj, Sneha Maity, Rajat Singh, Sankhadeep Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2511.05600)  

**Abstract**: This paper proposes a MedGemma-based framework for automatic abnormality detection in musculoskeletal radiographs. Departing from conventional autoencoder and neural network pipelines, the proposed method leverages the MedGemma foundation model, incorporating a SigLIP-derived vision encoder pretrained on diverse medical imaging modalities. Preprocessed X-ray images are encoded into high-dimensional embeddings using the MedGemma vision backbone, which are subsequently passed through a lightweight multilayer perceptron for binary classification. Experimental assessment reveals that the MedGemma-driven classifier exhibits strong performance, exceeding conventional convolutional and autoencoder-based metrics. Additionally, the model leverages MedGemma's transfer learning capabilities, enhancing generalization and optimizing feature engineering. The integration of a modern medical foundation model not only enhances representation learning but also facilitates modular training strategies such as selective encoder block unfreezing for efficient domain adaptation. The findings suggest that MedGemma-powered classification systems can advance clinical radiograph triage by providing scalable and accurate abnormality detection, with potential for broader applications in automated medical image analysis.
Keywords: Google MedGemma, MURA, Medical Image, Classification. 

**Abstract (ZH)**: 基于MedGemma的框架在骨肌肉放射影像中自动检测异常的研究 

---
# FlowNet: Modeling Dynamic Spatio-Temporal Systems via Flow Propagation 

**Title (ZH)**: FlowNet：通过流传播建模动态时空系统 

**Authors**: Yutong Feng, Xu Liu, Yutong Xia, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05595)  

**Abstract**: Accurately modeling complex dynamic spatio-temporal systems requires capturing flow-mediated interdependencies and context-sensitive interaction dynamics. Existing methods, predominantly graph-based or attention-driven, rely on similarity-driven connectivity assumptions, neglecting asymmetric flow exchanges that govern system evolution. We propose Spatio-Temporal Flow, a physics-inspired paradigm that explicitly models dynamic node couplings through quantifiable flow transfers governed by conservation principles. Building on this, we design FlowNet, a novel architecture leveraging flow tokens as information carriers to simulate source-to-destination transfers via Flow Allocation Modules, ensuring state redistribution aligns with conservation laws. FlowNet dynamically adjusts the interaction radius through an Adaptive Spatial Masking module, suppressing irrelevant noise while enabling context-aware propagation. A cascaded architecture enhances scalability and nonlinear representation capacity. Experiments demonstrate that FlowNet significantly outperforms existing state-of-the-art approaches on seven metrics in the modeling of three real-world systems, validating its efficiency and physical interpretability. We establish a principled methodology for modeling complex systems through spatio-temporal flow interactions. 

**Abstract (ZH)**: 准确建模复杂的时空动态系统需要捕获流介导的相互依赖关系和上下文敏感的交互动力学。现有方法主要基于图或注意力机制，依赖于相似性驱动的连接假设，忽视了支配系统演化的不对称流交换。我们提出时空流这一物理启发式范式，明确通过守恒原理控制的可量化流传输来建模动态节点耦合。在此基础上，我们设计了FlowNet这一新型架构，利用流令牌作为信息载体，通过流分配模块模拟源到目的地的传输，确保状态重分布符合守恒定律。FlowNet通过自适应空间掩码模块动态调整交互半径，抑制无关噪声，同时实现上下文意识的传播。级联架构增强了可扩展性和非线性表示能力。实验结果显示，FlowNet在三个真实系统的建模中，在七个度量标准上显著优于现有最先进的方法，验证了其效率和物理可解释性。我们为通过时空流交互建模复杂系统建立了基本原则方法。 

---
# CoPRIS: Efficient and Stable Reinforcement Learning via Concurrency-Controlled Partial Rollout with Importance Sampling 

**Title (ZH)**: CoPRIS：通过并发控制的部分卷出与重要性采样实现高效且稳定的强化学习 

**Authors**: Zekai Qu, Yinxu Pan, Ao Sun, Chaojun Xiao, Xu Han  

**Link**: [PDF](https://arxiv.org/pdf/2511.05589)  

**Abstract**: Reinforcement learning (RL) post-training has become a trending paradigm for enhancing the capabilities of large language models (LLMs). Most existing RL systems for LLMs operate in a fully synchronous manner, where training must wait for the rollout of an entire batch to complete. This design leads to severe inefficiencies, as extremely long trajectories can stall the entire rollout process and leave many GPUs idle. To address this issue, we propose Concurrency- Controlled Partial Rollout with Importance Sampling (CoPRIS), which mitigates long-tail inefficiencies by maintaining a fixed number of concurrent rollouts, early-terminating once sufficient samples are collected, and reusing unfinished trajectories in subsequent rollouts. To mitigate the impact of off-policy trajectories, we introduce Cross-stage Importance Sampling Correction, which concatenates buffered log probabilities from the previous policy with those recomputed under the current policy for importance sampling correction. Experiments on challenging mathematical reasoning benchmarks show that CoPRIS achieves up to 1.94x faster training while maintaining comparable or superior performance to synchronous RL systems. The code of CoPRIS is available at this https URL. 

**Abstract (ZH)**: 基于并发控制部分回放与重要性采样的强化学习后训练方法 

---
# Fine-Tuning Vision-Language Models for Multimodal Polymer Property Prediction 

**Title (ZH)**: 基于视觉-语言模型的多模态聚合物性质预测微调 

**Authors**: An Vuong, Minh-Hao Van, Prateek Verma, Chen Zhao, Xintao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05577)  

**Abstract**: Vision-Language Models (VLMs) have shown strong performance in tasks like visual question answering and multimodal text generation, but their effectiveness in scientific domains such as materials science remains limited. While some machine learning methods have addressed specific challenges in this field, there is still a lack of foundation models designed for broad tasks like polymer property prediction using multimodal data. In this work, we present a multimodal polymer dataset to fine-tune VLMs through instruction-tuning pairs and assess the impact of multimodality on prediction performance. Our fine-tuned models, using LoRA, outperform unimodal and baseline approaches, demonstrating the benefits of multimodal learning. Additionally, this approach reduces the need to train separate models for different properties, lowering deployment and maintenance costs. 

**Abstract (ZH)**: Vision-Language Models在材料科学领域的科学域中表现出强大的性能，但在聚合物性质预测等任务上仍有限制。尽管一些机器学习方法已经解决了该领域的特定挑战，但对于使用多模态数据进行聚合物性质预测的应用，仍缺乏针对广泛任务的基座模型。在本工作中，我们提出一个多模态聚合物数据集，通过指令调优对视觉语言模型进行微调，并评估多模态性对预测性能的影响。使用LoRA微调后的模型优于单模态和基线方法，展示了多模态学习的优势。此外，该方法减少了为不同性质训练独立模型的需求，降低了部署和维护成本。 

---
# Elements of Active Continuous Learning and Uncertainty Self-Awareness: a Narrow Implementation for Face and Facial Expression Recognition 

**Title (ZH)**: 具有主动连续学习和不确定性自我意识的元素：面向和面部表情识别的窄范围实施 

**Authors**: Stanislav Selitskiy  

**Link**: [PDF](https://arxiv.org/pdf/2511.05574)  

**Abstract**: Reflection on one's thought process and making corrections to it if there exists dissatisfaction in its performance is, perhaps, one of the essential traits of intelligence. However, such high-level abstract concepts mandatory for Artificial General Intelligence can be modelled even at the low level of narrow Machine Learning algorithms. Here, we present the self-awareness mechanism emulation in the form of a supervising artificial neural network (ANN) observing patterns in activations of another underlying ANN in a search for indications of the high uncertainty of the underlying ANN and, therefore, the trustworthiness of its predictions. The underlying ANN is a convolutional neural network (CNN) ensemble employed for face recognition and facial expression tasks. The self-awareness ANN has a memory region where its past performance information is stored, and its learnable parameters are adjusted during the training to optimize the performance. The trustworthiness verdict triggers the active learning mode, giving elements of agency to the machine learning algorithm that asks for human help in high uncertainty and confusion conditions. 

**Abstract (ZH)**: 反思自己的思维过程并对其性能不满意时进行修正，可能是智力的一个重要特征。这种高级的抽象概念即使在狭窄机器学习算法的低层次上也可以通过模拟自我意识机制来建模。本文呈现了一种监督人工神经网络（ANN）作为机制，该监督ANN观察其底层ANN激活模式，以寻找底层ANN高不确定性及预测可信度的迹象。底层ANN是一个用于面部识别和面部表情任务的卷积神经网络（CNN）集成。监督ANN具备存储其过往表现信息的记忆区域，在训练过程中调整可学习参数以优化性能。可信度判断会触发主动学习模式，赋予机器学习算法一定的自主权，在高不确定性和混乱条件下请求人类的帮助。 

---
# Video Text Preservation with Synthetic Text-Rich Videos 

**Title (ZH)**: 视频中合成富文本信息的文本保留 

**Authors**: Ziyang Liu, Kevin Valencia, Justin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2511.05573)  

**Abstract**: While Text-To-Video (T2V) models have advanced rapidly, they continue to struggle with generating legible and coherent text within videos. In particular, existing models often fail to render correctly even short phrases or words and previous attempts to address this problem are computationally expensive and not suitable for video generation. In this work, we investigate a lightweight approach to improve T2V diffusion models using synthetic supervision. We first generate text-rich images using a text-to-image (T2I) diffusion model, then animate them into short videos using a text-agnostic image-to-video (I2v) model. These synthetic video-prompt pairs are used to fine-tune Wan2.1, a pre-trained T2V model, without any architectural changes. Our results show improvement in short-text legibility and temporal consistency with emerging structural priors for longer text. These findings suggest that curated synthetic data and weak supervision offer a practical path toward improving textual fidelity in T2V generation. 

**Abstract (ZH)**: 虽然文本到视频（T2V）模型取得了 rapid 进展，但它们继续在生成视频中的可读性和连贯性文本方面挣扎。特别是在现有模型经常无法正确渲染甚至短语或单词的情况下，以前尝试解决这一问题的方法计算成本高且不适合视频生成。在本文中，我们研究了一种轻量级方法，通过合成监督来提高 T2V 扩散模型的表现。我们首先使用文本到图像（T2I）扩散模型生成富含文本的图像，然后使用文本无关的图像到视频（I2v）模型将它们转换成短视频。这些合成的视频提示对具有预训练 T2V 模型（Wan2.1）进行微调，无需任何架构改动。实验结果表明，在生成较长时间文本时，短文本的可读性和时间一致性有所提高，并且出现了新兴的结构先验。这些发现表明，经过精心设计的合成数据和弱监督提供了一种可行的路径，以提高 T2V 生成中的文本保真度。 

---
# C3-Diff: Super-resolving Spatial Transcriptomics via Cross-modal Cross-content Contrastive Diffusion Modelling 

**Title (ZH)**: C3-Diff: 通过跨模态跨内容对比扩散建模超分辨空间转录组学 

**Authors**: Xiaofei Wang, Stephen Price, Chao Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.05571)  

**Abstract**: The rapid advancement of spatial transcriptomics (ST), i.e., spatial gene expressions, has made it possible to measure gene expression within original tissue, enabling us to discover molecular mechanisms. However, current ST platforms frequently suffer from low resolution, limiting the in-depth understanding of spatial gene expression. Super-resolution approaches promise to enhance ST maps by integrating histology images with gene expressions of profiled tissue spots. However, it remains a challenge to model the interactions between histology images and gene expressions for effective ST enhancement. This study presents a cross-modal cross-content contrastive diffusion framework, called C3-Diff, for ST enhancement with histology images as guidance. In C3-Diff, we firstly analyze the deficiency of traditional contrastive learning paradigm, which is then refined to extract both modal-invariant and content-invariant features of ST maps and histology images. Further, to overcome the problem of low sequencing sensitivity in ST maps, we perform nosing-based information augmentation on the surface of feature unit hypersphere. Finally, we propose a dynamic cross-modal imputation-based training strategy to mitigate ST data scarcity. We tested C3-Diff by benchmarking its performance on four public datasets, where it achieves significant improvements over competing methods. Moreover, we evaluate C3-Diff on downstream tasks of cell type localization, gene expression correlation and single-cell-level gene expression prediction, promoting AI-enhanced biotechnology for biomedical research and clinical applications. Codes are available at this https URL. 

**Abstract (ZH)**: 空间转录组学(ST)的快速进展，即空间基因表达，使得在原始组织中测量基因表达成为可能，从而帮助我们发现分子机制。然而，当前的ST平台经常受到低分辨率的限制，限制了对空间基因表达的深入理解。高分辨率方法有望通过将组织切片图像与基因表达特征点结合来增强ST图。然而，如何有效地建模组织切片图像和基因表达之间的相互作用仍然是一个挑战。本研究提出了一种跨模态跨内容对比扩散框架C3-Diff，用于在组织切片图像引导下增强ST。在C3-Diff中，我们首先分析了传统对比学习范式的缺陷，对其进行改进以提取ST图和组织切片图像的模态不变和内容不变特征。此外，为克服ST图中低测序灵敏度的问题，我们在特征单元超球面的表面上进行噪声信息增强。最后，我们提出了一种动态跨模态缺失填补训练策略，以缓解ST数据稀缺的问题。我们通过在四个公开数据集上测试C3-Diff，实现了与竞争方法相比的显著性能提升。此外，我们在细胞类型定位、基因表达相关性和单细胞层次基因表达预测的下游任务中评估了C3-Diff，促进了增强人工智能在生物医学研究和临床应用中的生物技术。代码可在以下链接获取。 

---
# Automatic Extraction of Road Networks by using Teacher-Student Adaptive Structural Deep Belief Network and Its Application to Landslide Disaster 

**Title (ZH)**: 基于教师-学生自适应结构深信念网络的道路网络自动提取及其在滑坡灾害中的应用 

**Authors**: Shin Kamada, Takumi Ichimura  

**Link**: [PDF](https://arxiv.org/pdf/2511.05567)  

**Abstract**: An adaptive structural learning method of Restricted Boltzmann Machine (RBM) and Deep Belief Network (DBN) has been developed as one of prominent deep learning models. The neuron generation-annihilation algorithm in RBM and layer generation algorithm in DBN make an optimal network structure for given input during the learning. In this paper, our model is applied to an automatic recognition method of road network system, called RoadTracer. RoadTracer can generate a road map on the ground surface from aerial photograph data. A novel method of RoadTracer using the Teacher-Student based ensemble learning model of Adaptive DBN is proposed, since the road maps contain many complicated features so that a model with high representation power to detect should be required. The experimental results showed the detection accuracy of the proposed model was improved from 40.0\% to 89.0\% on average in the seven major cities among the test dataset. In addition, we challenged to apply our method to the detection of available roads when landslide by natural disaster is occurred, in order to rapidly obtain a way of transportation. For fast inference, a small size of the trained model was implemented on a small embedded edge device as lightweight deep learning. We reported the detection results for the satellite image before and after the rainfall disaster in Japan. 

**Abstract (ZH)**: 一种自适应结构学习的受限玻尔兹曼机（RBM）和深度信念网络（DBN）的方法及其在道路网络系统自动识别中的应用：基于教师-学生自适应DBN集成学习模型的道路识别方法研究 

---
# Efficient Online Continual Learning in Sensor-Based Human Activity Recognition 

**Title (ZH)**: 基于传感器的人体活动识别的高效在线持续学习 

**Authors**: Yao Zhang, Souza Leite Clayton, Yu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2511.05566)  

**Abstract**: Machine learning models for sensor-based human activity recognition (HAR) are expected to adapt post-deployment to recognize new activities and different ways of performing existing ones. To address this need, Online Continual Learning (OCL) mechanisms have been proposed, allowing models to update their knowledge incrementally as new data become available while preserving previously acquired information. However, existing OCL approaches for sensor-based HAR are computationally intensive and require extensive labeled samples to represent new changes. Recently, pre-trained model-based (PTM-based) OCL approaches have shown significant improvements in performance and efficiency for computer vision applications. These methods achieve strong generalization capabilities by pre-training complex models on large datasets, followed by fine-tuning on downstream tasks for continual learning. However, applying PTM-based OCL approaches to sensor-based HAR poses significant challenges due to the inherent heterogeneity of HAR datasets and the scarcity of labeled data in post-deployment scenarios. This paper introduces PTRN-HAR, the first successful application of PTM-based OCL to sensor-based HAR. Unlike prior PTM-based OCL approaches, PTRN-HAR pre-trains the feature extractor using contrastive loss with a limited amount of data. This extractor is then frozen during the streaming stage. Furthermore, it replaces the conventional dense classification layer with a relation module network. Our design not only significantly reduces the resource consumption required for model training while maintaining high performance, but also improves data efficiency by reducing the amount of labeled data needed for effective continual learning, as demonstrated through experiments on three public datasets, outperforming the state-of-the-art. The code can be found here: this https URL 

**Abstract (ZH)**: 基于传感器的人类活动识别（HAR）的机器学习模型预期在部署后能够适应并识别新活动以及现有活动的不同执行方式。为了满足这一需求，已经提出了在线持续学习（OCL）机制，使模型能够在新数据可用时逐步更新其知识，同时保留之前获取的信息。然而，现有的基于传感器的HAR的OCL方法计算密集型，并且需要大量的标记样本来代表新的变化。最近，基于预训练模型（PTM）的OCL方法在计算机视觉应用中的性能和效率上取得了显著改进。这些方法通过在大型数据集上预训练复杂的模型，然后对下游任务进行微调，来实现强大的泛化能力。然而，将基于PTM的OCL方法应用于基于传感器的HAR带来了巨大挑战，因为HAR数据集存在固有的异构性，并且部署后标记数据稀缺。本文引入了PTRN-HAR，这是首次成功将基于PTM的OCL应用于基于传感器的HAR。与之前的基于PTM的OCL方法不同，PTRN-HAR使用对比损失进行有限数据量的特征提取器预训练，该提取器在流式处理阶段冻结。此外，它用关系模块网络替代了传统的密集分类层。我们的设计不仅能显著降低模型训练所需资源，同时保持高性能，并通过减少有效持续学习所需的标记数据量来提高数据效率，实验结果在三个公开数据集上的表现优于现有最佳方法。代码可在以下链接找到：this https URL。 

---
# In-Context Adaptation of VLMs for Few-Shot Cell Detection in Optical Microscopy 

**Title (ZH)**: 基于上下文的VLMs少量标注细胞检测自适应优化在光学显微镜中的应用 

**Authors**: Shreyan Ganguly, Angona Biswas, Jaydeep Rade, Md Hasibul Hasan Hasib, Nabila Masud, Nitish Singla, Abhipsa Dash, Ushashi Bhattacharjee, Aditya Balu, Anwesha Sarkar, Adarsh Krishnamurthy, Soumik Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2511.05565)  

**Abstract**: Foundation vision-language models (VLMs) excel on natural images, but their utility for biomedical microscopy remains underexplored. In this paper, we investigate how in-context learning enables state-of-the-art VLMs to perform few-shot object detection when large annotated datasets are unavailable, as is often the case with microscopic images. We introduce the Micro-OD benchmark, a curated collection of 252 images specifically curated for in-context learning, with bounding-box annotations spanning 11 cell types across four sources, including two in-lab expert-annotated sets. We systematically evaluate eight VLMs under few-shot conditions and compare variants with and without implicit test-time reasoning tokens. We further implement a hybrid Few-Shot Object Detection (FSOD) pipeline that combines a detection head with a VLM-based few-shot classifier, which enhances the few-shot performance of recent VLMs on our benchmark. Across datasets, we observe that zero-shot performance is weak due to the domain gap; however, few-shot support consistently improves detection, with marginal gains achieved after six shots. We observe that models with reasoning tokens are more effective for end-to-end localization, whereas simpler variants are more suitable for classifying pre-localized crops. Our results highlight in-context adaptation as a practical path for microscopy, and our benchmark provides a reproducible testbed for advancing open-vocabulary detection in biomedical imaging. 

**Abstract (ZH)**: 基础视觉-语言模型（VLMs）在自然图像上表现出色，但其在生物医学显微镜成像中的应用尚未充分探索。在本文中，我们研究了上下文学习如何使最先进的VLMs在大型标注数据集不可用时，能够进行少样本对象检测。我们引入了Micro-OD基准，这是一个专门为上下文学习设计的252张图像集合，其中包含跨越四个数据源的11种细胞类型的边界框标注，包括两个实验室专家标注的数据集。我们在少样本条件下系统评估了八种VLMs，并比较了含隐式测试时推理标记和不含的变体。我们还实现了一个结合检测头部与VLM基少样本分类器的混合少样本对象检测（FSOD）管道，该管道增强了我们在基准上的最近VLMs的少样本性能。在不同数据集上，我们观察到零样本性能较弱，主要是由于领域缺口；然而，少样本支持始终提高了检测性能，六次样本后获得边际提高。我们观察到具有推理标记的模型在端到端定位方面更有效，而更简单的变体更适合分类预定位裁剪。我们的研究结果强调了上下文适应在显微镜中的实用路径，并且我们的基准提供了一个可重复的测试平台，用于促进生物医学影像中的开放词汇检测研究。 

---
# Lookahead Unmasking Elicits Accurate Decoding in Diffusion Language Models 

**Title (ZH)**: 前瞻解掩揭示扩散语言模型的准确解码 

**Authors**: Sanghyun Lee, Seungryong Kim, Jongho Park, Dongmin Park  

**Link**: [PDF](https://arxiv.org/pdf/2511.05563)  

**Abstract**: Masked Diffusion Models (MDMs) as language models generate by iteratively unmasking tokens, yet their performance crucially depends on the inference time order of unmasking. Prevailing heuristics, such as confidence based sampling, are myopic: they optimize locally, fail to leverage extra test-time compute, and let early decoding mistakes cascade. We propose Lookahead Unmasking (LookUM), which addresses these concerns by reformulating sampling as path selection over all possible unmasking orders without the need for an external reward model. Our framework couples (i) a path generator that proposes paths by sampling from pools of unmasking sets with (ii) a verifier that computes the uncertainty of the proposed paths and performs importance sampling to subsequently select the final paths. Empirically, erroneous unmasking measurably inflates sequence level uncertainty, and our method exploits this to avoid error-prone trajectories. We validate our framework across six benchmarks, such as mathematics, planning, and coding, and demonstrate consistent performance improvements. LookUM requires only two to three paths to achieve peak performance, demonstrating remarkably efficient path selection. The consistent improvements on both LLaDA and post-trained LLaDA 1.5 are particularly striking: base LLaDA with LookUM rivals the performance of RL-tuned LLaDA 1.5, while LookUM further enhances LLaDA 1.5 itself showing that uncertainty based verification provides orthogonal benefits to reinforcement learning and underscoring the versatility of our framework. Code will be publicly released. 

**Abstract (ZH)**: 前瞻性解码（LookUM）：Masked Diffusion Models作为一种语言模型通过迭代解码生成，其性能关键取决于解码顺序。我们提出了一种前瞻性解码方法（Lookahead Unmasking, LookUM），通过路径选择而不是局部优化来克服现有方法的局限性。我们验证了该方法在数学、规划和编码等六个基准上的表现，并显示出一致的性能提升。LookUM仅需选择两到三条路径即可达到最佳性能，展示了高效的路径选择能力。在基LLaDA和后训练LLaDA 1.5上的一致改进尤为显著：结合前瞻性解码的LLaDA与RL调优的LLaDA 1.5性能相当，进一步证明了基于不确定性的验证为强化学习提供了独立的优势，并突显了我们框架的通用性。代码将公开发布。 

---
# Effective Test-Time Scaling of Discrete Diffusion through Iterative Refinement 

**Title (ZH)**: 测试时离散扩散的有效逐迭代缩放 

**Authors**: Sanghyun Lee, Sunwoo Kim, Seungryong Kim, Jongho Park, Dongmin Park  

**Link**: [PDF](https://arxiv.org/pdf/2511.05562)  

**Abstract**: Test-time scaling through reward-guided generation remains largely unexplored for discrete diffusion models despite its potential as a promising alternative. In this work, we introduce Iterative Reward-Guided Refinement (IterRef), a novel test-time scaling method tailored to discrete diffusion that leverages reward- guided noising-denoising transitions to progressively refine misaligned intermediate states. We formalize this process within a Multiple-Try Metropolis (MTM) framework, proving convergence to the reward-aligned distribution. Unlike prior methods that assume the current state is already aligned with the reward distribution and only guide the subsequent transition, our approach explicitly refines each state in situ, progressively steering it toward the optimal intermediate distribution. Across both text and image domains, we evaluate IterRef on diverse discrete diffusion models and observe consistent improvements in reward-guided generation quality. In particular, IterRef achieves striking gains under low compute budgets, far surpassing prior state-of-the-art baselines. 

**Abstract (ZH)**: 通过奖励引导生成进行测试时缩放的方法在离散扩散模型中尚未得到充分探索，尽管其作为一种有前途的替代方案具有潜力。在本文中，我们引入了Iterative Reward-Guided Refinement (IterRef)，这是一种针对离散扩散的新颖测试时缩放方法，利用奖励引导的加噪-去噪转换逐步细化错位的中间状态。我们将这一过程形式化为Multiple-Try Metropolis (MTM)框架，并证明其收敛于奖励对齐的分布。与先前假设当前状态已与奖励分布对齐的方法不同，我们的方法明确地在现场逐步细化每个状态，使其逐步向最佳中间分布转变。在文本和图像领域，我们评估了IterRef在多种离散扩散模型上的效果，并观察到了一致的奖励引导生成质量改进。特别是在低计算预算下，IterRef取得了显著的改进，远远超越了先前的最佳 baseline。 

---
# Sample-Efficient Language Modeling with Linear Attention and Lightweight Enhancements 

**Title (ZH)**: 基于线性注意机制和轻量级增强的样本高效语言模型 

**Authors**: Patrick Haller, Jonas Golde, Alan Akbik  

**Link**: [PDF](https://arxiv.org/pdf/2511.05560)  

**Abstract**: We study architectural and optimization tech- niques for sample-efficient language modeling under the constraints of the BabyLM 2025 shared task. Our model, BLaLM, replaces self-attention with a linear-time mLSTM to- ken mixer and explores lightweight enhance- ments, including short convolutions, sliding window attention with dynamic modulation, and Hedgehog feature maps. To support train- ing in low-resource settings, we curate a high- quality corpus emphasizing readability and ped- agogical structure. Experiments across both STRICT and STRICT-SMALL tracks show that (1) linear attention combined with sliding win- dow attention consistently improves zero-shot performance, and (2) the Muon optimizer stabi- lizes convergence and reduces perplexity over AdamW. These results highlight effective strate- gies for efficient language modeling without relying on scale. 

**Abstract (ZH)**: 我们针对BabyLM 2025 共享任务的约束条件，研究了样本高效语言模型的架构和优化技术。我们的模型BLaLM 使用线性时间 mLSTM 令牌混音器取代了自注意力机制，并探索了轻量级增强技术，包括短卷积、滑动窗口注意力以及动态调制和刺猬特征图。为支持低资源环境下的训练，我们精心筛选了高质量语料库，强调易读性和教学结构。在STRICT 和STRICT-SMALL 轨道的实验中表明，(1) 线性注意力与滑动窗口注意力结合使用一致地提升了零样本性能，且(2) 穆翁优化器相比AdamW 更能稳定收敛并降低困惑度。这些结果突出了在不依赖规模的情况下进行高效语言建模的有效策略。 

---
# Diversified Flow Matching with Translation Identifiability 

**Title (ZH)**: 多样化流匹配与翻译可识别性 

**Authors**: Sagar Shrestha, Xiao Fu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05558)  

**Abstract**: Diversified distribution matching (DDM) finds a unified translation function mapping a diverse collection of conditional source distributions to their target counterparts. DDM was proposed to resolve content misalignment issues in unpaired domain translation, achieving translation identifiability. However, DDM has only been implemented using GANs due to its constraints on the translation function. GANs are often unstable to train and do not provide the transport trajectory information -- yet such trajectories are useful in applications such as single-cell evolution analysis and robot route planning. This work introduces diversified flow matching (DFM), an ODE-based framework for DDM. Adapting flow matching (FM) to enforce a unified translation function as in DDM is challenging, as FM learns the translation function's velocity rather than the translation function itself. A custom bilevel optimization-based training loss, a nonlinear interpolant, and a structural reformulation are proposed to address these challenges, offering a tangible implementation. To our knowledge, DFM is the first ODE-based approach guaranteeing translation identifiability. Experiments on synthetic and real-world datasets validate the proposed method. 

**Abstract (ZH)**: 多样化分布匹配 (DDM) 找到一种统一的转换函数，将多样性的条件源分布映射到其目标对应物。DDM 被提出用于解决无配对领域转换中的内容错位问题，实现转换可识别性。然而，由于 DDM 对转换函数的限制，它仅使用 GAN 进行实现，而 GAN 常见难题在于训练不稳定且不提供运输轨迹信息——而这些信息在单细胞进化分析和机器人路线规划等领域是很有用的。本工作引入多样化流匹配 (DFM) 作为一种基于 ODE 的 DDM 框架。将流匹配 (FM) 调整以强制统一的转换函数，尽管 FM 学习的是转换函数的速度而非转换函数本身，提出了自定义的双层优化训练损失、非线性插值和结构重新表述来解决这些挑战，提供了一种实际的实现方法。据我们所知，DFM 是第一个基于 ODE 确保转换可识别性的方法。实验在合成和真实数据集上验证了所提出的方法。 

---
# EVLP:Learning Unified Embodied Vision-Language Planner with Reinforced Supervised Fine-Tuning 

**Title (ZH)**: EVLP：学习统一的感知语言规划器with强化监督微调 

**Authors**: Xinyan Cai, Shiguang Wu, Dafeng Chi, Yuzheng Zhuang, Xingyue Quan, Jianye Hao, Qiang Guan  

**Link**: [PDF](https://arxiv.org/pdf/2511.05553)  

**Abstract**: In complex embodied long-horizon manipulation tasks, effective task decomposition and execution require synergistic integration of textual logical reasoning and visual-spatial imagination to ensure efficient and accurate operation. Current methods fail to adopt a unified generation framework for multimodal planning, lead to inconsistent in multimodal planning. To address this challenge, we present \textbf{EVLP (Embodied Vision-Language Planner)}, an innovative multimodal unified generation framework that jointly models linguistic reasoning and visual generation. Our approach achieves multimodal planning for long-horizon tasks through a novel training pipeline incorporating dynamic pretraining and reinforced alignment. Our core innovations consist of three key components: \textbf{1) Unified Multimodal Generation Framework}: For understanding, We integrate semantic information with spatial features to provide comprehensive visual perception. For generation, we directly learn the joint distribution of discrete images for one-step visual synthesis, enabling coordinated language-visual modeling through learnable cross-modal attention mechanisms. \textbf{2) Dynamic Perception Pretraining}: We propose a bidirectional dynamic alignment strategy employing inverse dynamics tasks and forward dynamics tasks, effectively strengthening multimodal correlations within a unified feature space. \textbf{3) Reinforced Supervised Fine-Tuning}: While conducting instruction-based fine-tuning in the unified generation space, we construct a reinforce loss to align the spatial logic between textual actions and generated images, enabling the model to acquire spatio-awared multimodal planning capabilities. 

**Abstract (ZH)**: 在复杂体态长时程操作任务中，有效的任务分解和执行需要文本逻辑推理和视觉空间想象力的协同整合，以确保操作的高效和准确。当前的方法未能为多模态规划采用统一生成框架，导致多模态规划中的一致性问题。为应对这一挑战，我们提出了**EVLP（体态视觉-语言规划者）**，这是一种创新的多模态统一生成框架，联合建模语言推理和视觉生成。我们的方法通过结合动态预训练和强化对齐的新型训练管道，实现长时程任务的多模态规划。我们的核心创新包括三个关键组成部分：**1）统一多模态生成框架**：在理解中，我们将语义信息与空间特征集成，提供全面的视觉感知。在生成中，我们直接学习离散图像的一步视觉合成的联合分布，通过可学习的跨模态注意力机制实现协调的语言-视觉建模。**2）动态感知预训练**：我们提出了一种双向动态对齐策略，采用逆动力学任务和正动力学任务，有效地在统一特征空间内增强多模态相关性。**3）强化监督微调**：在统一生成空间中执行指令驱动的微调时，我们构建了一个强化损失，以对齐文本动作和生成图像之间的空间逻辑，使模型能够获得具有空间意识的多模态规划能力。 

---
# Deep one-gate per layer networks with skip connections are universal classifiers 

**Title (ZH)**: 具有跳连的每层一个门电路的深度神经网络是通用分类器 

**Authors**: Raul Rojas  

**Link**: [PDF](https://arxiv.org/pdf/2511.05552)  

**Abstract**: This paper shows how a multilayer perceptron with two hidden layers, which has been designed to classify two classes of data points, can easily be transformed into a deep neural network with one-gate layers and skip connections. 

**Abstract (ZH)**: 本文展示了如何将一个设计用于分类两类数据点的具有两隐藏层的多层感知机轻松转换为具有单门层和跳接连接的深度神经网络。 

---
# AGRAG: Advanced Graph-based Retrieval-Augmented Generation for LLMs 

**Title (ZH)**: AGRAG: 基于图的检索增强生成高级方法用于大型语言模型 

**Authors**: Yubo Wang, Haoyang Li, Fei Teng, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.05549)  

**Abstract**: Graph-based retrieval-augmented generation (Graph-based RAG) has demonstrated significant potential in enhancing Large Language Models (LLMs) with structured knowledge. However, existing methods face three critical challenges: Inaccurate Graph Construction, caused by LLM hallucination; Poor Reasoning Ability, caused by failing to generate explicit reasons telling LLM why certain chunks were selected; and Inadequate Answering, which only partially answers the query due to the inadequate LLM reasoning, making their performance lag behind NaiveRAG on certain tasks. To address these issues, we propose AGRAG, an advanced graph-based retrieval-augmented generation framework. When constructing the graph, AGRAG substitutes the widely used LLM entity extraction method with a statistics-based method, avoiding hallucination and error propagation. When retrieval, AGRAG formulates the graph reasoning procedure as the Minimum Cost Maximum Influence (MCMI) subgraph generation problem, where we try to include more nodes with high influence score, but with less involving edge cost, to make the generated reasoning paths more comprehensive. We prove this problem to be NP-hard, and propose a greedy algorithm to solve it. The MCMI subgraph generated can serve as explicit reasoning paths to tell LLM why certain chunks were retrieved, thereby making the LLM better focus on the query-related part contents of the chunks, reducing the impact of noise, and improving AGRAG's reasoning ability. Furthermore, compared with the simple tree-structured reasoning paths, our MCMI subgraph can allow more complex graph structures, such as cycles, and improve the comprehensiveness of the generated reasoning paths. 

**Abstract (ZH)**: 基于图的检索增强生成（基于图的RAG）在增强大型语言模型（LLMs）的结构化知识方面展示了显著潜力。然而，现有方法面临三个关键挑战：由LLM幻觉引起的不准确图构建；由于未能生成明确的推理过程，导致推理能力不足；以及不充分的答案，导致查询回答不完全，使其在某些任务上的性能落后于NaiveRAG。为了解决这些问题，我们提出了一种先进的基于图的检索增强生成框架——AGRAG。在构建图时，AGRAG用基于统计的方法替代了广泛使用的LLM实体提取方法，以避免幻觉和错误传播。在检索过程中，AGRAG将图推理过程形式化为最小成本最大影响（MCMI）子图生成问题，尽量包含更多具有高影响评分的节点，但涉及的边成本较少，以使生成的推理路径更加全面。我们证明该问题是NP-hard的，并提出了一种贪婪算法来解决它。生成的MCMI子图可以作为明确的推理路径告诉LLM为什么选择了特定的片段，从而使其更好地关注与查询相关部分的内容，减少噪声的影响，并增强AGRAG的推理能力。此外，与简单的树形结构推理路径相比，我们的MCMI子图可以允许更复杂的图结构，如环路，从而提高生成推理路径的全面性。 

---
# Automated Invoice Data Extraction: Using LLM and OCR 

**Title (ZH)**: 自动发票数据提取：使用大语言模型和光学字符识别 

**Authors**: Advait Thakur, Khushi Khanchandani, Akshita Shetty, Chaitravi Reddy, Ritisa Behera  

**Link**: [PDF](https://arxiv.org/pdf/2511.05547)  

**Abstract**: Conventional Optical Character Recognition (OCR) systems are challenged by variant invoice layouts, handwritten text, and low- quality scans, which are often caused by strong template dependencies that restrict their flexibility across different document structures and layouts. Newer solutions utilize advanced deep learning models such as Convolutional Neural Networks (CNN) as well as Transformers, and domain-specific models for better layout analysis and accuracy across various sections over varied document types. Large Language Models (LLMs) have revolutionized extraction pipelines at their core with sophisticated entity recognition and semantic comprehension to support complex contextual relationship mapping without direct programming specification. Visual Named Entity Recognition (NER) capabilities permit extraction from invoice images with greater contextual sensitivity and much higher accuracy rates than older approaches. Existing industry best practices utilize hybrid architectures that blend OCR technology and LLM for maximum scalability and minimal human intervention. This work introduces a holistic Artificial Intelligence (AI) platform combining OCR, deep learning, LLMs, and graph analytics to achieve unprecedented extraction quality and consistency. 

**Abstract (ZH)**: 传统的光学字符识别（OCR）系统在处理变异性发票布局、手写文本和低质量扫描时面临挑战，这些问题是由于强模板依赖性所造成的，限制了其在不同文档结构和布局上的灵活性。新的解决方案利用了包括卷积神经网络（CNN）和变换器在内的高级深度学习模型以及特定领域的模型，以在各种文档类型的不同部分上实现更好的布局分析和准确率。大型语言模型（LLMs）通过复杂的实体识别和语义理解在核心抽取管道中实现了革命性变化，支持复杂的上下文关系映射，无需直接编程指定。视觉命名实体识别（NER）能力使得从发票图像中提取内容具有更高的上下文敏感性和准确性。现有的行业最佳实践采用了结合OCR技术和LLM的混合架构，以实现最大的可扩展性和最小的人工干预。本研究提出了一种综合的人工智能（AI）平台，结合OCR、深度学习、LLMs和图分析，以实现无与伦比的提取质量和一致性。 

---
# ConnectomeBench: Can LLMs Proofread the Connectome? 

**Title (ZH)**: ConnectomeBench: 能够校对联结组的LLMs吗？ 

**Authors**: Jeff Brown, Andrew Kirjner Annika Vivekananthan, Ed Boyden  

**Link**: [PDF](https://arxiv.org/pdf/2511.05542)  

**Abstract**: Connectomics - the mapping of neural connections in an organism's brain - currently requires extraordinary human effort to proofread the data collected from imaging and machine-learning assisted segmentation. With the growing excitement around using AI agents to automate important scientific tasks, we explore whether current AI systems can perform multiple tasks necessary for data proofreading. We introduce ConnectomeBench, a multimodal benchmark evaluating large language model (LLM) capabilities in three critical proofreading tasks: segment type identification, split error correction, and merge error detection. Using expert annotated data from two large open-source datasets - a cubic millimeter of mouse visual cortex and the complete Drosophila brain - we evaluate proprietary multimodal LLMs including Claude 3.7/4 Sonnet, o4-mini, GPT-4.1, GPT-4o, as well as open source models like InternVL-3 and NVLM. Our results demonstrate that current models achieve surprisingly high performance in segment identification (52-82% balanced accuracy vs. 20-25% chance) and binary/multiple choice split error correction (75-85% accuracy vs. 50% chance) while generally struggling on merge error identification tasks. Overall, while the best models still lag behind expert performance, they demonstrate promising capabilities that could eventually enable them to augment and potentially replace human proofreading in connectomics. Project page: this https URL and Dataset this https URL 

**Abstract (ZH)**: 连接组学——有机体大脑中的神经连接映射——目前需要非凡的人工努力来校对从成像和机器学习辅助分割中收集的数据。随着使用AI代理自动化重要科学任务的兴奋不断增加，我们探索当前AI系统是否能够执行数据校对所需的多项任务。我们介绍了连接组基准，这是一个多模态基准，评估大型语言模型（LLM）在三种关键校对任务中的能力：分割类型识别、分裂错误修正和合并错误检测。使用来自两个大型开源数据集（鼠视觉皮层的立方毫米区域和整个果蝇脑）的专家注释数据，我们评估了包括Claude 3.7/4 Sonnet、o4-mini、GPT-4.1、GPT-4o以及开源模型InternVL-3和NVLM在内的专有和开源多模态LLM。结果显示，当前模型在分割识别（52-82%平衡准确率，相比之下随机猜测为20-25%）和二分类/多分类分裂错误修正（75-85%准确率，相比之下随机猜测为50%）方面表现出令人惊讶的高性能，但在合并错误识别任务上普遍表现不佳。总体而言，尽管最佳模型仍然落后于专家性能，但展示了令人鼓舞的能力，最终可能能够增强甚至替代连接组学中的手工校对。项目页面: this https URL 数据集: this https URL 

---
# Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability 

**Title (ZH)**: 时间稀疏自编码器：利用语言的序列特性提高可解释性 

**Authors**: Usha Bhalla, Alex Oesterling, Claudio Mayrink Verdun, Himabindu Lakkaraju, Flavio P. Calmon  

**Link**: [PDF](https://arxiv.org/pdf/2511.05541)  

**Abstract**: Translating the internal representations and computations of models into concepts that humans can understand is a key goal of interpretability. While recent dictionary learning methods such as Sparse Autoencoders (SAEs) provide a promising route to discover human-interpretable features, they suffer from a variety of problems, including a systematic failure to capture the rich conceptual information that drives linguistic understanding. Instead, they exhibit a bias towards shallow, token-specific, or noisy features, such as "the phrase 'The' at the start of sentences". In this work, we propose that this is due to a fundamental issue with how dictionary learning methods for LLMs are trained. Language itself has a rich, well-studied structure spanning syntax, semantics, and pragmatics; however, current unsupervised methods largely ignore this linguistic knowledge, leading to poor feature discovery that favors superficial patterns over meaningful concepts. We focus on a simple but important aspect of language: semantic content has long-range dependencies and tends to be smooth over a sequence, whereas syntactic information is much more local. Building on this insight, we introduce Temporal Sparse Autoencoders (T-SAEs), which incorporate a novel contrastive loss encouraging consistent activations of high-level features over adjacent tokens. This simple yet powerful modification enables SAEs to disentangle semantic from syntactic features in a self-supervised manner. Across multiple datasets and models, T-SAEs recover smoother, more coherent semantic concepts without sacrificing reconstruction quality. Strikingly, they exhibit clear semantic structure despite being trained without explicit semantic signal, offering a new pathway for unsupervised interpretability in language models. 

**Abstract (ZH)**: 将模型的内部表示和计算转换为人类可理解的概念是可解释性的一个关键目标。虽然近期的词典学习方法，如稀疏自编码器（SAEs），为发现可由人类理解的特征提供了有前景的途径，但它们遭受着各种问题的困扰，包括系统性地未能捕捉到驱动语言理解的丰富概念信息。相反，它们倾向于偏向于浅层、特定词条或嘈杂的特征，例如“句子开头的‘The’短语”。在本文中，我们提出这源于词典学习方法对大规模语言模型（LLM）训练机制的一个根本性问题。语言本身具有广泛研究的丰富结构，跨越句法、语义和语用学；然而，当前的无监督方法大多忽视了这种语言知识，导致特征发现效果不佳，偏好表面模式而非有意义的概念。我们关注语言的一个简单但重要的方面：语义内容具有长程依赖性，并且倾向于在一个序列中平滑变化，而句法信息则要更加地方性。基于这一洞察，我们引入了时序稀疏自编码器（T-SAEs），它包含了一个新颖的对比损失，鼓励相邻词条之间高层特征的一致激活。这一简单而强大的修改使SAEs能够在自我监督的条件下分离语义特征和句法特征。在多个数据集和模型上，T-SAEs恢复了更加平滑和连贯的语义概念，且未牺牲重建质量。令人惊讶的是，它们即使在未使用明确语义信号的情况下进行训练也表现出清晰的语义结构，为语言模型的无监督可解释性提供了一条新的途径。 

---
# Token Is All You Need: Cognitive Planning through Sparse Intent Alignment 

**Title (ZH)**: Token 是一切所需：通过稀疏意图对齐进行认知规划 

**Authors**: Shiyao Sang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05540)  

**Abstract**: We challenge the long-standing assumption that exhaustive scene modeling is required for high-performance end-to-end autonomous driving (E2EAD). Unlike world-model approaches that rely on computationally intensive future scene generation or vision-language-action (VLA) systems constrained by Markov assumptions, we show that a minimal set of semantically rich tokens is sufficient for effective planning. Experiments on the nuPlan benchmark (720 scenarios, over 11,000 samples) using perception-informed BEV representations yield three key findings: (1) even without future prediction, our sparse representation achieves 0.548 m ADE, comparable to or surpassing prior methods reporting around 0.75 m on nuScenes; (2) conditioning trajectory decoding on predicted future tokens reduces ADE to 0.479 m, a 12.6% improvement over current-state baselines; and (3) explicit reconstruction loss offers no benefit and may degrade performance under reliable perception inputs. Notably, we observe the emergence of temporal fuzziness, where the model adaptively attends to task-relevant semantics rather than aligning rigidly to fixed timestamps, providing a cognitive advantage for planning under uncertainty. Our "token is all you need" principle marks a paradigm shift from reconstructing the world to understanding it, laying a foundation for cognitively inspired systems that plan through imagination rather than reaction. 

**Abstract (ZH)**: 我们挑战长期以来认为高性能端到端自动驾驶（E2EAD）需要详尽场景建模的假设。不同于依赖于计算密集型未来场景生成或基于马尔可夫假设的视觉-语言-动作（VLA）系统的世界建模方法，我们展示了一个富含语义的最小词表集就足够进行有效的规划。在nuPlan基准上（720个场景，超过11,000个样本）使用感知指导的BEV表示，我们得到三个关键发现：（1）即使没有未来预测，我们的稀疏表示实现了0.548 m ADE，与nuScenes上报告的约0.75 m的先前方法相比具有竞争力或更好；（2）基于预测的未来词元对轨迹解码进行条件约束将ADE降低至0.479 m，比当前状态基线提高了12.6%；（3）显式重构损失在可靠感知输入下并没有益处，甚至可能降低性能。值得注意的是，我们观察到时间模糊性的出现，模型能够自适应地关注与任务相关的意义，而不是严格对齐到固定的时间戳，这为不确定性下的规划提供了认知优势。我们的“词元即所需一切”原则标志着从重构世界到理解世界的范式转变，为通过想象而非反应进行规划的认知启发式系统奠定了基础。 

---
# Gravity-Awareness: Deep Learning Models and LLM Simulation of Human Awareness in Altered Gravity 

**Title (ZH)**: 重力意识：深度学习模型与人类在改变重力环境下的意识模拟 

**Authors**: Bakytzhan Alibekov, Alina Gutoreva, Elisa Raffaella-Ferre  

**Link**: [PDF](https://arxiv.org/pdf/2511.05536)  

**Abstract**: Earth's gravity has fundamentally shaped human development by guiding the brain's integration of vestibular, visual, and proprioceptive inputs into an internal model of gravity: a dynamic neural representation enabling prediction and interpretation of gravitational forces. This work presents a dual computational framework to quantitatively model these adaptations. The first component is a lightweight Multi-Layer Perceptron (MLP) that predicts g-load-dependent changes in key electroencephalographic (EEG) frequency bands, representing the brain's cortical state. The second component utilizes a suite of independent Gaussian Processes (GPs) to model the body's broader physiological state, including Heart Rate Variability (HRV), Electrodermal Activity (EDA), and motor behavior. Both models were trained on data derived from a comprehensive review of parabolic flight literature, using published findings as anchor points to construct robust, continuous functions. To complement this quantitative analysis, we simulated subjective human experience under different gravitational loads, ranging from microgravity (0g) and partial gravity (Moon 0.17g, Mars 0.38g) to hypergravity associated with spacecraft launch and re-entry (1.8g), using a large language model (Claude 3.5 Sonnet). The model was prompted with physiological parameters to generate introspective narratives of alertness and self-awareness, which closely aligned with the quantitative findings from both the EEG and physiological models. This combined framework integrates quantitative physiological modeling with generative cognitive simulation, offering a novel approach to understanding and predicting human performance in altered gravity 

**Abstract (ZH)**: 地球的重力通过指导脑部对前庭、视觉和本体感觉输入的整合，形成对重力的内部模型，从而从根本上影响人类的发展：一种动态的神经代表，使大脑能够预测和解释重力作用。本研究提出了一种双计算框架来定量建模这些适应性变化。第一个组成部分是一个轻量级的多层感知器（MLP），用于预测与g负荷相关的关键脑电图（EEG）频率带的变化，代表大脑皮层状态。第二个组成部分则利用一系列独立的高斯过程（GPs）来建模人体更广泛的生理状态，包括心率变异性（HRV）、皮肤电活动（EDA）和运动行为。这两个模型均基于综合回顾抛物线飞行文献的数据进行训练，使用已发表的研究结果作为锚点来构建稳健的连续函数。为了补充这种定量分析，我们使用大型语言模型（Claude 3.5 Sonnet）模拟了在不同重力负荷下的主观人类体验，从微重力（0g）和部分重力（月球0.17g，火星0.38g）到与航天器发射和重返大气层相关的超重力（1.8g），并生成了涉及警觉性和自我意识的内省叙述，这些叙述与EEG和生理模型的定量发现紧密一致。结合这种框架将定量生理建模与生成性认知模拟整合起来，提供了一种理解与预测人类在改变重力环境中的表现的新方法。 

---
# Selective Diabetic Retinopathy Screening with Accuracy-Weighted Deep Ensembles and Entropy-Guided Abstention 

**Title (ZH)**: 准确度加权深度集成与熵引导的避决策在选择性糖尿病视网膜病变筛查中的应用 

**Authors**: Jophy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.05529)  

**Abstract**: Diabetic retinopathy (DR), a microvascular complication of diabetes and a leading cause of preventable blindness, is projected to affect more than 130 million individuals worldwide by 2030. Early identification is essential to reduce irreversible vision loss, yet current diagnostic workflows rely on methods such as fundus photography and expert review, which remain costly and resource-intensive. This, combined with DR's asymptomatic nature, results in its underdiagnosis rate of approximately 25 percent. Although convolutional neural networks (CNNs) have demonstrated strong performance in medical imaging tasks, limited interpretability and the absence of uncertainty quantification restrict clinical reliability. Therefore, in this study, a deep ensemble learning framework integrated with uncertainty estimation is introduced to improve robustness, transparency, and scalability in DR detection. The ensemble incorporates seven CNN architectures-ResNet-50, DenseNet-121, MobileNetV3 (Small and Large), and EfficientNet (B0, B2, B3)- whose outputs are fused through an accuracy-weighted majority voting strategy. A probability-weighted entropy metric quantifies prediction uncertainty, enabling low-confidence samples to be excluded or flagged for additional review. Training and validation on 35,000 EyePACS retinal fundus images produced an unfiltered accuracy of 93.70 percent (F1 = 0.9376). Uncertainty-filtering later was conducted to remove unconfident samples, resulting in maximum-accuracy of 99.44 percent (F1 = 0.9932). The framework shows that uncertainty-aware, accuracy-weighted ensembling improves reliability without hindering performance. With confidence-calibrated outputs and a tunable accuracy-coverage trade-off, it offers a generalizable paradigm for deploying trustworthy AI diagnostics in high-risk care. 

**Abstract (ZH)**: 糖尿病视网膜病变（DR）：糖尿病的一种微血管并发症，也是可预防失明的主要原因，预计到2030年将影响全球超过1.3亿人。早期识别对于减少不可逆视力丧失至关重要，但当前的诊断工作流程依赖于眼底摄影和专家审查等方法，这些方法仍然成本高且资源密集。结合DR无症状的特点，其未诊断率约为25%。尽管卷积神经网络（CNNs）在医学影像任务中表现出强烈性能，但由于解释性的限制和不确定性量化缺失，限制了临床可靠性。因此，在这项研究中，引入了一种结合不确定性估计的深度集成学习框架，以提高DR检测的鲁棒性、透明性和可扩展性。该集成结合了ResNet-50、DenseNet-121、MobileNetV3（Small和Large）、EfficientNet（B0、B2、B3）等七种CNN架构，并通过准确率加权多数投票策略融合输出。通过概率加权熵度量量化预测不确定性，使低置信度样本能够被排除或标记以进行额外审查。在35,000张EyePACS眼底图像上进行训练和验证，未经筛选的准确率为93.70%（F1 = 0.9376）。后续进行不确定性筛选，去除不自信样本，最终准确率为99.44%（F1 = 0.9932）。该框架表明，具有不确定性意识的准确率加权集成可以提高可靠性而不牺牲性能，同时也提供了一种通过校准输出和调节准确性和覆盖范围之间的权衡来部署可信赖AI诊断的一般化范式。 

---
# The Evolution of Probabilistic Price Forecasting Techniques: A Review of the Day-Ahead, Intra-Day, and Balancing Markets 

**Title (ZH)**: 概率价格预测技术的发展：对日前、日内和平衡市场的一项综述 

**Authors**: Ciaran O'Connor, Mohamed Bahloul, Steven Prestwich, Andrea Visentin  

**Link**: [PDF](https://arxiv.org/pdf/2511.05523)  

**Abstract**: Electricity price forecasting has become a critical tool for decision-making in energy markets, particularly as the increasing penetration of renewable energy introduces greater volatility and uncertainty. Historically, research in this field has been dominated by point forecasting methods, which provide single-value predictions but fail to quantify uncertainty. However, as power markets evolve due to renewable integration, smart grids, and regulatory changes, the need for probabilistic forecasting has become more pronounced, offering a more comprehensive approach to risk assessment and market participation. This paper presents a review of probabilistic forecasting methods, tracing their evolution from Bayesian and distribution based approaches, through quantile regression techniques, to recent developments in conformal prediction. Particular emphasis is placed on advancements in probabilistic forecasting, including validity-focused methods which address key limitations in uncertainty estimation. Additionally, this review extends beyond the Day-Ahead Market to include the Intra-Day and Balancing Markets, where forecasting challenges are intensified by higher temporal granularity and real-time operational constraints. We examine state of the art methodologies, key evaluation metrics, and ongoing challenges, such as forecast validity, model selection, and the absence of standardised benchmarks, providing researchers and practitioners with a comprehensive and timely resource for navigating the complexities of modern electricity markets. 

**Abstract (ZH)**: 电力价格预测已成为能源市场决策的关键工具，特别是在可再生能源渗透率增加导致更大波动性和不确定性的情况下。历史上，该领域的研究主要集中在点预测方法上，这些方法提供单值预测但无法量化不确定性。然而，由于可再生能源集成、智能电网和监管变化导致电力市场演进，概率预测的需求变得更加突出，提供了更全面的风险评估和市场参与的方法。本文回顾了概率预测方法的发展历程，从贝叶斯和分布基础方法，到分位数回归技术，再到近期的置信预测发展。特别强调了概率预测的进步，包括专注于有效性的方法，以解决不确定性估计中的关键限制。此外，本文的回顾不仅限于日前市场，还涵盖了日内和平衡市场，这些市场的预测挑战因更高的时间粒度和实时操作约束而加剧。我们评估了最新方法、关键评价指标以及持续的挑战，如预测的有效性、模型选择和缺乏标准化基准，为研究人员和从业者提供了导航现代电力市场复杂性的全面及时资源。 

---
# AIRMap - AI-Generated Radio Maps for Wireless Digital Twins 

**Title (ZH)**: AIRMap - 由AI生成的无线数字孪生射频图谱 

**Authors**: Ali Saeizadeh, Miead Tehrani-Moayyed, Davide Villa, J. Gordon Beattie Jr., Pedram Johari, Stefano Basagni, Tommaso Melodia  

**Link**: [PDF](https://arxiv.org/pdf/2511.05522)  

**Abstract**: Accurate, low-latency channel modeling is essential for real-time wireless network simulation and digital-twin applications. Traditional modeling methods like ray tracing are however computationally demanding and unsuited to model dynamic conditions. In this paper, we propose AIRMap, a deep-learning framework for ultra-fast radio-map estimation, along with an automated pipeline for creating the largest radio-map dataset to date. AIRMap uses a single-input U-Net autoencoder that processes only a 2D elevation map of terrain and building heights. Trained and evaluated on 60,000 Boston-area samples, spanning coverage areas from 500 m to 3 km per side, AIRMap predicts path gain with under 5 dB RMSE in 4 ms per inference on an NVIDIA L40S -over 7000x faster than GPU-accelerated ray tracing based radio maps. A lightweight transfer learning calibration using just 20% of field measurements reduces the median error to approximately 10%, significantly outperforming traditional simulators, which exceed 50% error. Integration into the Colosseum emulator and the Sionna SYS platform demonstrate near-zero error in spectral efficiency and block-error rate compared to measurement-based channels. These findings validate AIRMap's potential for scalable, accurate, and real-time radio map estimation in wireless digital twins. 

**Abstract (ZH)**: 超快无线地图估计的深度学习框架AIRMap及其最大规模无线地图数据集的自动化生成方法 

---
# Retracing the Past: LLMs Emit Training Data When They Get Lost 

**Title (ZH)**: 重溯过往：LLMs在丢失时会发出训练数据 

**Authors**: Myeongseob Ko, Nikhil Reddy Billa, Adam Nguyen, Charles Fleming, Ming Jin, Ruoxi Jia  

**Link**: [PDF](https://arxiv.org/pdf/2511.05518)  

**Abstract**: The memorization of training data in large language models (LLMs) poses significant privacy and copyright concerns. Existing data extraction methods, particularly heuristic-based divergence attacks, often exhibit limited success and offer limited insight into the fundamental drivers of memorization leakage. This paper introduces Confusion-Inducing Attacks (CIA), a principled framework for extracting memorized data by systematically maximizing model uncertainty. We empirically demonstrate that the emission of memorized text during divergence is preceded by a sustained spike in token-level prediction entropy. CIA leverages this insight by optimizing input snippets to deliberately induce this consecutive high-entropy state. For aligned LLMs, we further propose Mismatched Supervised Fine-tuning (SFT) to simultaneously weaken their alignment and induce targeted confusion, thereby increasing susceptibility to our attacks. Experiments on various unaligned and aligned LLMs demonstrate that our proposed attacks outperform existing baselines in extracting verbatim and near-verbatim training data without requiring prior knowledge of the training data. Our findings highlight persistent memorization risks across various LLMs and offer a more systematic method for assessing these vulnerabilities. 

**Abstract (ZH)**: 在大型语言模型中对训练数据的记忆化存储引发了显著的隐私和版权担忧。现有的数据提取方法，尤其是基于启发式的发散攻击，往往效果有限且难以洞察记忆化泄漏的根本驱动因素。本文介绍了诱导混淆攻击（CIA），这是一种基于原理的框架，通过系统地最大化模型不确定性来提取记忆化数据。我们通过实验证明，在发散期间记忆化文本的释放前存在持续的标记级预测熵峰值。CIA 利用这一见解通过优化输入片段以故意诱导这种连续的高熵状态。对于对齐的 LLM，我们进一步提出了不匹配的监督微调（SFT）方法，以同时削弱其对齐并诱导针对性的混淆，从而增加其对攻击的易感性。在各种未对齐和对齐的 LLM 上的实验表明，与现有基准相比，我们提出的攻击在无需事先了解训练数据的情况下更有效地提取了逐字和近逐字的训练数据。我们的研究结果突出了各种 LLM 中持续的记忆化风险，并提供了一种更系统的评估这些漏洞的方法。 

---
# Ming-UniAudio: Speech LLM for Joint Understanding, Generation and Editing with Unified Representation 

**Title (ZH)**: 明 UniAudio：统一表示下的语音 LLM，用于联合理解和生成编辑 

**Authors**: Canxiang Yan, Chunxiang Jin, Dawei Huang, Haibing Yu, Han Peng, Hui Zhan, Jie Gao, Jing Peng, Jingdong Chen, Jun Zhou, Kaimeng Ren, Ming Yang, Mingxue Yang, Qiang Xu, Qin Zhao, Ruijie Xiong, Shaoxiong Lin, Xuezhi Wang, Yi Yuan, Yifei Wu, Yongjie Lyu, Zhengyu He, Zhihao Qiu, Zhiqiang Fang, Ziyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05516)  

**Abstract**: Existing speech models suffer from competing requirements on token representations by understanding and generation tasks. This discrepancy in representation prevents speech language models from performing instruction-based free-form editing. To solve this challenge, we introduce a novel framework that unifies speech understanding, generation, and editing. The core of our unified model is a unified continuous speech tokenizer MingTok-Audio, the first continuous tokenizer to effectively integrate semantic and acoustic features, which makes it suitable for both understanding and generation tasks. Based on this unified continuous audio tokenizer, we developed the speech language model Ming-UniAudio, which achieved a balance between generation and understanding capabilities. Ming-UniAudio sets new state-of-the-art (SOTA) records on 8 out of 12 metrics on the ContextASR benchmark. Notably, for Chinese voice cloning, it achieves a highly competitive Seed-TTS-WER of 0.95. Leveraging this foundational model, we further trained a dedicated speech editing model Ming-UniAudio-Edit, the first speech language model that enables universal, free-form speech editing guided solely by natural language instructions, handling both semantic and acoustic modifications without timestamp condition. To rigorously assess the editing capability and establish a foundation for future research, we introduce Ming-Freeform-Audio-Edit, the first comprehensive benchmark tailored for instruction-based free-form speech editing, featuring diverse scenarios and evaluation dimensions spanning semantic correctness, acoustic quality, and instruction alignment. We open-sourced the continuous audio tokenizer, the unified foundational model, and the free-form instruction-based editing model to facilitate the development of unified audio understanding, generation, and manipulation. 

**Abstract (ZH)**: 现有的语音模型在理解和生成任务对词表示的竞争要求之间存在矛盾。这种表示的不一致阻碍了语音语言模型进行基于指令的自由形式编辑。为了解决这一挑战，我们提出了一种新的框架，该框架统一了语音理解、生成和编辑。我们统一模型的核心是一个名为MingTok-Audio的统一连续语音分词器，这是第一个有效结合语义和声学特征的连续分词器，使其适用于理解和生成任务。基于这一统一的连续语音分词器，我们开发了语音语言模型Ming-UniAudio，实现了生成能力和理解能力的平衡。Ming-UniAudio在ContextASR基准上的12个指标中有8个指标上达到了新的最佳性能。特别是在中文语音克隆方面，其Seed-TTS-WER达到了0.95的高水平竞争力。依托这一基础模型，我们进一步训练了一个专门的语音编辑模型Ming-UniAudio-Edit，这是第一个仅凭自然语言指令指导进行通用自由形式语音编辑的语言模型，能够处理语义和声学修改而无需时间戳条件。为了严格评估编辑能力并为未来的研究奠定基础，我们引入了Ming-Freeform-Audio-Edit，这是第一个针对基于指令的自由形式语音编辑的综合基准，涵盖了多种场景和评价维度，涉及语义正确性、声音质量和指令对齐。我们开源了连续语音分词器、统一的基础模型和基于指令的自由形式编辑模型，以促进统一音频理解、生成和操作的发展。 

---
# From Failure Modes to Reliability Awareness in Generative and Agentic AI System 

**Title (ZH)**: 从失败模式到生成性和自主性人工智能系统的可靠性意识 

**Authors**: Janet, Liangwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05511)  

**Abstract**: This chapter bridges technical analysis and organizational preparedness by tracing the path from layered failure modes to reliability awareness in generative and agentic AI systems. We first introduce an 11-layer failure stack, a structured framework for identifying vulnerabilities ranging from hardware and power foundations to adaptive learning and agentic reasoning. Building on this, the chapter demonstrates how failures rarely occur in isolation but propagate across layers, creating cascading effects with systemic consequences. To complement this diagnostic lens, we develop the concept of awareness mapping: a maturity-oriented framework that quantifies how well individuals and organizations recognize reliability risks across the AI stack. Awareness is treated not only as a diagnostic score but also as a strategic input for AI governance, guiding improvement and resilience planning. By linking layered failures to awareness levels and further integrating this into Dependability-Centred Asset Management (DCAM), the chapter positions awareness mapping as both a measurement tool and a roadmap for trustworthy and sustainable AI deployment across mission-critical domains. 

**Abstract (ZH)**: 这一章通过追溯从分层故障模式到生成性和自主性人工智能系统可靠性的意识路径，将技术分析与组织准备有机结合。首先介绍了一个11层故障栈，这是一种结构化的框架，用于识别从硬件和能源基础到自适应学习和自主推理的各种漏洞。在此基础上，本章展示故障很少孤立发生，而是跨层传播，产生系统性影响。为补充这一诊断视角，我们发展了意识映射的概念：这是一种以成熟度为导向的框架，量化个人和组织在整个人工智能堆栈中识别可靠性风险的能力。意识不仅被视为诊断评分，也被视为人工智能治理的战略输入，指导改进和韧性规划。通过将分层故障与意识水平联系起来，并进一步将这一概念整合进依赖性中心资产管理系统（DCAM），本章将意识映射定位为衡量工具和实现可信和可持续人工智能部署的路线图，在关键任务领域尤为重要。 

---
# TEMPO: Temporal Multi-scale Autoregressive Generation of Protein Conformational Ensembles 

**Title (ZH)**: TEMPO：蛋白质构象ensemble的时空多尺度自回归生成 

**Authors**: Yaoyao Xu, Di Wang, Zihan Zhou, Tianshu Yu, Mingchen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.05510)  

**Abstract**: Understanding the dynamic behavior of proteins is critical to elucidating their functional mechanisms, yet generating realistic, temporally coherent trajectories of protein ensembles remains a significant challenge. In this work, we introduce a novel hierarchical autoregressive framework for modeling protein dynamics that leverages the intrinsic multi-scale organization of molecular motions. Unlike existing methods that focus on generating static conformational ensembles or treat dynamic sampling as an independent process, our approach characterizes protein dynamics as a Markovian process. The framework employs a two-scale architecture: a low-resolution model captures slow, collective motions driving major conformational transitions, while a high-resolution model generates detailed local fluctuations conditioned on these large-scale movements. This hierarchical design ensures that the causal dependencies inherent in protein dynamics are preserved, enabling the generation of temporally coherent and physically realistic trajectories. By bridging high-level biophysical principles with state-of-the-art generative modeling, our approach provides an efficient framework for simulating protein dynamics that balances computational efficiency with physical accuracy. 

**Abstract (ZH)**: 理解蛋白质的动力学行为对于阐明其功能机制至关重要，然而生成真实的、时间上一致的蛋白质ensemble轨迹仍是一个巨大的挑战。本文引入了一种新颖的分层自回归框架，用于建模蛋白质动力学，该框架利用了分子运动的固有的多尺度组织特性。与现有方法专注于生成静态构象ensemble或将动力学采样视为独立过程不同，我们的方法将蛋白质动力学-characterize-as-马尔可夫过程。该框架采用两尺度架构：低分辨率模型捕捉慢速、集体运动，驱动主要构象转换，而高分辨率模型则根据这些大规模运动生成详细的局部波动。这种分层设计确保了蛋白质动力学内在的因果依赖性得以保留，从而能够生成时间上一致且物理上真实的轨迹。通过将高层生物物理原理与最先进的生成建模技术结合，我们的方法提供了一个兼顾计算效率和物理准确性的蛋白质动力学模拟的高效框架。 

---
# Randomized-MLP Regularization Improves Domain Adaptation and Interpretability in DINOv2 

**Title (ZH)**: 随机化MLP正则化提高DINOv2的域适应性和可解释性 

**Authors**: Joel Valdivia Ortega, Lorenz Lamm, Franziska Eckardt, Benedikt Schworm, Marion Jasnin, Tingying Peng  

**Link**: [PDF](https://arxiv.org/pdf/2511.05509)  

**Abstract**: Vision Transformers (ViTs), such as DINOv2, achieve strong performance across domains but often repurpose low-informative patch tokens in ways that reduce the interpretability of attention and feature maps. This challenge is especially evident in medical imaging, where domain shifts can degrade both performance and transparency. In this paper, we introduce Randomized-MLP (RMLP) regularization, a contrastive learning-based method that encourages more semantically aligned representations. We use RMLPs when fine-tuning DINOv2 to both medical and natural image modalities, showing that it improves or maintains downstream performance while producing more interpretable attention maps. We also provide a mathematical analysis of RMLPs, offering insights into its role in enhancing ViT-based models and advancing our understanding of contrastive learning. 

**Abstract (ZH)**: Vision Transformers (ViTs)，如DINOv2，在多个领域中取得了强劲的表现，但经常会重新利用低信息量的patches，从而降低注意力和特征图的可解释性。这种挑战在医学成像中尤为明显，因为领域变换可能导致性能和透明度下降。本文介绍了一种基于对比学习的正则化方法——随机MLP（RMLP）正则化，该方法鼓励更具语义对齐的表示。我们使用RMLPs对DINOv2进行微调，以适应医学和自然图像模态，结果显示其提高了或保持了下游性能，同时生成了更具可解释性的注意力图。我们还提供了RMLPs的数学分析，探讨了其在增强ViT基础模型方面的作用，并促进了我们对对比学习的理解。 

---
# Personalized Chain-of-Thought Summarization of Financial News for Investor Decision Support 

**Title (ZH)**: 个性化金融新闻链式思考摘要支持投资者决策 

**Authors**: Tianyi Zhang, Mu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.05508)  

**Abstract**: Financial advisors and investors struggle with information overload from financial news, where irrelevant content and noise obscure key market signals and hinder timely investment decisions. To address this, we propose a novel Chain-of-Thought (CoT) summarization framework that condenses financial news into concise, event-driven summaries. The framework integrates user-specified keywords to generate personalized outputs, ensuring that only the most relevant contexts are highlighted. These personalized summaries provide an intermediate layer that supports language models in producing investor-focused narratives, bridging the gap between raw news and actionable insights. 

**Abstract (ZH)**: 金融顾问和投资者面对财务新闻的信息过载问题，其中无关内容和噪音模糊了关键市场信号，阻碍了及时的投资决策。为此，我们提出了一种新颖的Chain-of-Thought (CoT) 总结框架，将其凝练为简洁的事件驱动摘要。该框架整合用户指定的关键词生成个性化输出，确保仅突出最相关的上下文。这些个性化的摘要为语言模型生成以投资者为重点的叙述提供了中间层，填补了原始新闻与 actionable 洞察之间的差距。 

---
# Rewiring Human Brain Networks via Lightweight Dynamic Connectivity Framework: An EEG-Based Stress Validation 

**Title (ZH)**: 基于EEG的轻量级动态连接框架下的人脑网络重 wiring及其在压力验证中的应用 

**Authors**: Sayantan Acharya, Abbas Khosravi, Douglas Creighton, Roohallah Alizadehsani, U. Rajendra Acharya  

**Link**: [PDF](https://arxiv.org/pdf/2511.05505)  

**Abstract**: In recent years, Electroencephalographic analysis has gained prominence in stress research when combined with AI and Machine Learning models for validation. In this study, a lightweight dynamic brain connectivity framework based on Time Varying Directed Transfer Function is proposed, where TV DTF features were validated through ML based stress classification. TV DTF estimates the directional information flow between brain regions across distinct EEG frequency bands, thereby capturing temporal and causal influences that are often overlooked by static functional connectivity measures. EEG recordings from the 32 channel SAM 40 dataset were employed, focusing on mental arithmetic task trials. The dynamic EEG-based TV-DTF features were validated through ML classifiers such as Support Vector Machine, Random Forest, Gradient Boosting, Adaptive Boosting, and Extreme Gradient Boosting. Experimental results show that alpha-TV-DTF provided the strongest discriminative power, with SVM achieving 89.73% accuracy in 3-class classification and with XGBoost achieving 93.69% accuracy in 2 class classification. Relative to absolute power and phase locking based functional connectivity features, alpha TV DTF and beta TV DTF achieved higher performance across the ML models, highlighting the advantages of dynamic over static measures. Feature importance analysis further highlighted dominant long-range frontal parietal and frontal occipital informational influences, emphasizing the regulatory role of frontal regions under stress. These findings validate the lightweight TV-DTF as a robust framework, revealing spatiotemporal brain dynamics and directional influences across different stress levels. 

**Abstract (ZH)**: 近年来，结合AI和机器学习模型的脑电图分析在压力研究中获得了重视。本研究提出了一种基于时间变化定向传输函数的轻量级动态脑连接框架，并通过机器学习方法验证了TV DTF特征用于压力分类的有效性。TV DTF可估计脑区在不同脑电频率带之间的方向性信息流，从而捕捉到静态功能连接度量经常忽略的时间性和因果影响。本研究使用32通道SAM 40数据集，专注于心算任务试验。通过支持向量机、随机森林、梯度提升、自适应提升和极端梯度提升等机器学习分类器验证了动态脑电图基于TV DTF的特征。实验结果表明，α-TV DTF提供了最强的鉴别能力，支持向量机在3类分类中达到了89.73%的准确率，极端梯度提升在2类分类中达到了93.69%的准确率。与基于绝对功率和相位锁定的功能连接特征相比，α TV DTF和β TV DTF在所有机器学习模型中表现更优，突显了动态度量的优势。特征重要性分析进一步强调了远端前额叶和前额叶枕叶的信息性影响，突显了在压力下前额叶区域的调节作用。这些发现验证了TV DTF作为一种稳健框架的有效性，揭示了不同压力水平下空间时间脑动态及其方向性影响。 

---
# Production-Grade Local LLM Inference on Apple Silicon: A Comparative Study of MLX, MLC-LLM, Ollama, llama.cpp, and PyTorch MPS 

**Title (ZH)**: 基于Apple Silicon的生产级本地LLM推理：MLX、MLC-LLM、Ollama、llama.cpp和PyTorch MPS的比较研究 

**Authors**: Varun Rajesh, Om Jodhpurkar, Pooja Anbuselvan, Mantinder Singh, Ashok Jallepali, Shantanu Godbole, Pradeep Kumar Sharma, Hritvik Shrivastava  

**Link**: [PDF](https://arxiv.org/pdf/2511.05502)  

**Abstract**: We present a systematic, empirical evaluation of five local large language model (LLM) runtimes on Apple Silicon: MLX, MLC-LLM, this http URL, Ollama, and PyTorch MPS. Experiments were conducted on a Mac Studio equipped with an M2 Ultra processor and 192 GB of unified memory. Using the Qwen-2.5 model family across prompts ranging from a few hundred to 100,000 tokens, we measure time-to-first-token (TTFT), steady-state throughput, latency percentiles, long-context behavior (key-value and prompt caching), quantization support, streaming performance, batching and concurrency behavior, and deployment complexity.
Under our settings, MLX achieves the highest sustained generation throughput, while MLC-LLM delivers consistently lower TTFT for moderate prompt sizes and offers stronger out-of-the-box inference features. this http URL is highly efficient for lightweight single-stream use, Ollama emphasizes developer ergonomics but lags in throughput and TTFT, and PyTorch MPS remains limited by memory constraints on large models and long contexts.
All frameworks execute fully on-device with no telemetry, ensuring strong privacy guarantees. We release scripts, logs, and plots to reproduce all results. Our analysis clarifies the design trade-offs in Apple-centric LLM deployments and provides evidence-based recommendations for interactive and long-context processing. Although Apple Silicon inference frameworks still trail NVIDIA GPU-based systems such as vLLM in absolute performance, they are rapidly maturing into viable, production-grade solutions for private, on-device LLM inference. 

**Abstract (ZH)**: 我们对五种本地大型语言模型（LLM）运行时（MLX、MLC-LLM、this http URL、Ollama 和 PyTorch MPS）在Apple Silicon上的系统化、实证评估。实验在配备M2 Ultra处理器和192 GB统一内存的Mac Studio上进行。我们使用Qwen-2.5模型系列，在从几百个到100,000个令牌不等的提示下，测量首个令牌生成时间（TTFT）、稳定状态吞吐量、延迟百分位数、长上下文行为（键值缓存和提示缓存）、量化支持、流式性能、批处理和并发行为以及部署复杂性。在我们的设置下，MLX实现了最高的持续生成吞吐量，而MLC-LLM对于中等大小的提示始终提供更低的TTFT，并提供了更强的即用型推理功能。this http URL对轻量级单流使用非常高效，Ollama强调开发者友好性但在吞吐量和TTFT方面落后，而PyTorch MPS仍然受到大模型和长时间上下文内存限制的制约。所有框架均完全在设备端执行，不收集任何遥测数据，保证了强大的隐私保护。我们发布了用于重现所有结果的脚本、日志和图表。我们的分析澄清了Apple为中心的LLM部署中的设计权衡，并提供了基于证据的建议，用于交互式和长上下文处理。尽管Apple Silicon推理框架在绝对性能上仍落后于如vLLM等基于NVIDIA GPU的系统，但它们正迅速成熟，成为可行的、生产级的解决方案，用于私有的、设备端的LLM推理。 

---
# Towards Ecologically Valid LLM Benchmarks: Understanding and Designing Domain-Centered Evaluations for Journalism Practitioners 

**Title (ZH)**: 面向生态有效的大语言模型基准：理解并设计面向新闻从业者领域中心的评估方法 

**Authors**: Charlotte Li, Nick Hagar, Sachita Nishal, Jeremy Gilbert, Nick Diakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2511.05501)  

**Abstract**: Benchmarks play a significant role in how researchers and the public understand generative AI systems. However, the widespread use of benchmark scores to communicate about model capabilities has led to criticisms of validity, especially whether benchmarks test what they claim to test (i.e. construct validity) and whether benchmark evaluations are representative of how models are used in the wild (i.e. ecological validity). In this work we explore how to create an LLM benchmark that addresses these issues by taking a human-centered approach. We focus on designing a domain-oriented benchmark for journalism practitioners, drawing on insights from a workshop of 23 journalism professionals. Our workshop findings surface specific challenges that inform benchmark design opportunities, which we instantiate in a case study that addresses underlying criticisms and specific domain concerns. Through our findings and design case study, this work provides design guidance for developing benchmarks that are better tuned to specific domains. 

**Abstract (ZH)**: 生成式AI系统的研究人员和公众理解其作用过程发挥着重要作用。然而，广泛使用基准分数来传达模型能力导致了对其有效性的批评，尤其是这些基准是否真正测试了它们所声称测试的内容（即构造有效性），以及基准评估是否代表了模型在实际使用中的情况（即生态有效性）。在本工作中，我们通过采取以人为本的方法探索如何创建一个解决这些问题的LLM基准。我们关注设计一个面向新闻工作者的领域导向基准，并借鉴23名新闻专业人员研讨会的见解。我们的研讨会发现揭示了具体挑战，为基准设计机会提供了依据，并在案例研究中应对潜在批评和特定领域关切。通过我们的发现和设计案例研究，本工作提供了开发更契合特定领域基准的设计指导。 

---
# Predicting Oscar-Nominated Screenplays with Sentence Embeddings 

**Title (ZH)**: 使用句子嵌入预测奥斯卡提名剧本 

**Authors**: Francis Gross  

**Link**: [PDF](https://arxiv.org/pdf/2511.05500)  

**Abstract**: Oscar nominations are an important factor in the movie industry because they can boost both the visibility and the commercial success. This work explores whether it is possible to predict Oscar nominations for screenplays using modern language models. Since no suitable dataset was available, a new one called Movie-O-Label was created by combining the MovieSum collection of movie scripts with curated Oscar records. Each screenplay was represented by its title, Wikipedia summary, and full script. Long scripts were split into overlapping text chunks and encoded with the E5 sentence em bedding model. Then, the screenplay embed dings were classified using a logistic regression model. The best results were achieved when three feature inputs related to screenplays (script, summary, and title) were combined. The best-performing model reached a macro F1 score of 0.66, a precision recall AP of 0.445 with baseline 0.19 and a ROC-AUC of 0.79. The results suggest that even simple models based on modern text embeddings demonstrate good prediction performance and might be a starting point for future research. 

**Abstract (ZH)**: 奥斯卡提名是电影行业中一个重要因素，因为它们可以提升电影的可见度和商业成功。本文探讨了是否可以使用现代语言模型来预测电影剧本的奥斯卡提名。由于缺乏合适的数据集，创建了一个名为Movie-O-Label的新数据集，将MovieSum电影剧本集合与精心策划的奥斯卡记录相结合。每个剧本由其标题、Wikipedia摘要和完整剧本表示。长剧本被分割成重叠的文本片段，并使用E5句子嵌入模型进行编码。然后，使用逻辑回归模型对剧本嵌入进行分类。当将与剧本相关的三个特征输入（剧本、摘要和标题）组合时，获得了最佳结果。最佳模型实现了宏F1分数为0.66，精度召回AP为0.445（基线为0.19），ROC-AUC为0.79。结果表明，基于现代文本嵌入的简单模型表现出良好的预测性能，并可能成为未来研究的起点。 

---
# Weightless Neural Networks for Continuously Trainable Personalized Recommendation Systems 

**Title (ZH)**: 无权重神经网络用于连续可训练个性化推荐系统 

**Authors**: Rafayel Latif, Satwik Behera, Ali Al-Ebrahim  

**Link**: [PDF](https://arxiv.org/pdf/2511.05499)  

**Abstract**: Given that conventional recommenders, while deeply effective, rely on large distributed systems pre-trained on aggregate user data, incorporating new data necessitates large training cycles, making them slow to adapt to real-time user feedback and often lacking transparency in recommendation rationale. We explore the performance of smaller personal models trained on per-user data using weightless neural networks (WNNs), an alternative to neural backpropagation that enable continuous learning by using neural networks as a state machine rather than a system with pretrained weights. We contrast our approach against a classic weighted system, also on a per-user level, and standard collaborative filtering, achieving competitive levels of accuracy on a subset of the MovieLens dataset. We close with a discussion of how weightless systems can be developed to augment centralized systems to achieve higher subjective accuracy through recommenders more directly tunable by end-users. 

**Abstract (ZH)**: 基于无权重神经网络的小型个性化模型在实时用户反馈适应性和透明度方面的性能研究 

---
# Biomedical Hypothesis Explainability with Graph-Based Context Retrieval 

**Title (ZH)**: 基于图结构上下文检索的生物医学假设可解释性 

**Authors**: Ilya Tyagin, Saeideh Valipour, Aliaksandra Sikirzhytskaya, Michael Shtutman, Ilya Safro  

**Link**: [PDF](https://arxiv.org/pdf/2511.05498)  

**Abstract**: We introduce an explainability method for biomedical hypothesis generation systems, built on top of the novel Hypothesis Generation Context Retriever framework. Our approach combines semantic graph-based retrieval and relevant data-restrictive training to simulate real-world discovery constraints. Integrated with large language models (LLMs) via retrieval-augmented generation, the system explains hypotheses with contextual evidence using published scientific literature. We also propose a novel feedback loop approach, which iteratively identifies and corrects flawed parts of LLM-generated explanations, refining both the evidence paths and supporting context. We demonstrate the performance of our method with multiple large language models and evaluate the explanation and context retrieval quality through both expert-curated assessment and large-scale automated analysis. Our code is available at: this https URL. 

**Abstract (ZH)**: 我们提出了一种构建在新型假设生成上下文检索框架之上的解释性方法。该方法结合了基于语义图的检索和相关数据限制性训练，以模拟现实世界发现约束。通过检索增强生成与大型语言模型（LLMs）集成，该系统使用已发表的科学文献提供带有上下文证据的假设解释。我们还提出了一种新颖的反馈循环方法，该方法迭代地识别并纠正LLM生成解释中的缺陷部分，同时精炼证据路径和支持语境。我们使用多种大型语言模型演示了该方法的性能，并通过专家评估和大规模自动化分析评估解释和上下文检索的质量。代码详见：这个链接。 

---
# DOCUEVAL: An LLM-based AI Engineering Tool for Building Customisable Document Evaluation Workflows 

**Title (ZH)**: DOCUEVAL：一种基于LLM的自定义文档评估工作流的AI工程工具 

**Authors**: Hao Zhang, Qinghua Lu, Liming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05496)  

**Abstract**: Foundation models, such as large language models (LLMs), have the potential to streamline evaluation workflows and improve their performance. However, practical adoption faces challenges, such as customisability, accuracy, and scalability. In this paper, we present DOCUEVAL, an AI engineering tool for building customisable DOCUment EVALuation workflows. DOCUEVAL supports advanced document processing and customisable workflow design which allow users to define theory-grounded reviewer roles, specify evaluation criteria, experiment with different reasoning strategies and choose the assessment style. To ensure traceability, DOCUEVAL provides comprehensive logging of every run, along with source attribution and configuration management, allowing systematic comparison of results across alternative setups. By integrating these capabilities, DOCUEVAL directly addresses core software engineering challenges, including how to determine whether evaluators are "good enough" for deployment and how to empirically compare different evaluation strategies. We demonstrate the usefulness of DOCUEVAL through a real-world academic peer review case, showing how DOCUEVAL enables both the engineering of evaluators and scalable, reliable document evaluation. 

**Abstract (ZH)**: 基于AI的DOCUEVAL文档评估工作流工具：实现可定制、高准确性和可扩展性 

---
# IMDMR: An Intelligent Multi-Dimensional Memory Retrieval System for Enhanced Conversational AI 

**Title (ZH)**: IMDMR：一种增强对话式AI的智能多维记忆检索系统 

**Authors**: Tejas Pawar, Sarika Patil, Om Tilekar, Rushikesh Janwade, Vaibhav Helambe  

**Link**: [PDF](https://arxiv.org/pdf/2511.05495)  

**Abstract**: Conversational AI systems often struggle with maintaining coherent, contextual memory across extended interactions, limiting their ability to provide personalized and contextually relevant responses. This paper presents IMDMR (Intelligent Multi-Dimensional Memory Retrieval), a novel system that addresses these limitations through a multi-dimensional search architecture. Unlike existing memory systems that rely on single-dimensional approaches, IMDMR leverages six distinct memory dimensions-semantic, entity, category, intent, context, and temporal-to provide comprehensive memory retrieval capabilities. Our system incorporates intelligent query processing with dynamic strategy selection, cross-memory entity resolution, and advanced memory integration techniques. Through comprehensive evaluation against five baseline systems including LangChain RAG, LlamaIndex, MemGPT, and spaCy + RAG, IMDMR achieves a 3.8x improvement in overall performance (0.792 vs 0.207 for the best baseline). We present both simulated (0.314) and production (0.792) implementations, demonstrating the importance of real technology integration while maintaining superiority over all baseline systems. Ablation studies demonstrate the effectiveness of multi-dimensional search, with the full system outperforming individual dimension approaches by 23.3%. Query-type analysis reveals superior performance across all categories, particularly for preferences/interests (0.630) and goals/aspirations (0.630) queries. Comprehensive visualizations and statistical analysis confirm the significance of these improvements with p < 0.001 across all metrics. The results establish IMDMR as a significant advancement in conversational AI memory systems, providing a robust foundation for enhanced user interactions and personalized experiences. 

**Abstract (ZH)**: 基于多维检索的智能会话AI记忆系统IMDMR 

---
# Customized Retrieval-Augmented Generation with LLM for Debiasing Recommendation Unlearning 

**Title (ZH)**: 定制化检索增强生成以LLM去偏推荐遗忘 

**Authors**: Haichao Zhang, Chong Zhang, Peiyu Hu, Shi Qiu, Jia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05494)  

**Abstract**: Modern recommender systems face a critical challenge in complying with privacy regulations like the 'right to be forgotten': removing a user's data without disrupting recommendations for others. Traditional unlearning methods address this by partial model updates, but introduce propagation bias--where unlearning one user's data distorts recommendations for behaviorally similar users, degrading system accuracy. While retraining eliminates bias, it is computationally prohibitive for large-scale systems. To address this challenge, we propose CRAGRU, a novel framework leveraging Retrieval-Augmented Generation (RAG) for efficient, user-specific unlearning that mitigates bias while preserving recommendation quality. CRAGRU decouples unlearning into distinct retrieval and generation stages. In retrieval, we employ three tailored strategies designed to precisely isolate the target user's data influence, minimizing collateral impact on unrelated users and enhancing unlearning efficiency. Subsequently, the generation stage utilizes an LLM, augmented with user profiles integrated into prompts, to reconstruct accurate and personalized recommendations without needing to retrain the entire base model. Experiments on three public datasets demonstrate that CRAGRU effectively unlearns targeted user data, significantly mitigating unlearning bias by preventing adverse impacts on non-target users, while maintaining recommendation performance comparable to fully trained original models. Our work highlights the promise of RAG-based architectures for building robust and privacy-preserving recommender systems. The source code is available at: this https URL. 

**Abstract (ZH)**: 现代推荐系统在遵守《被遗忘的权利》等隐私法规时面临关键挑战：移除用户数据而不干扰其他用户的推荐。传统去学习方法通过部分模型更新来应对这一挑战，但会引入传播偏差——移除一个用户的数据会扭曲行为相似用户推荐结果，降低系统准确性。虽然重新训练可消除偏差，但对于大规模系统来说计算上是不可行的。为应对这一挑战，我们提出CRAGRU，这是一种利用检索增强生成（RAG）的新框架，实现了高效、用户特定的去学习，同时减少偏差并保持推荐质量。CRAGRU 将去学习过程拆分为独立的检索和生成阶段。在检索阶段，我们采用了三种定制策略，以精确隔离目标用户数据的影响，最小化对不相关用户的影响，提高去学习效率。随后，生成阶段利用增强的大型语言模型（LLM），将用户档案纳入提示中，以重建准确且个性化的推荐，无需重新训练整个基础模型。实验结果表明，CRAGRU 有效地去学习了目标用户数据，显著减轻了去学习偏差，防止对非目标用户产生负面影响，同时保持了与完全训练的原始模型相当的推荐性能。我们的工作突显了基于RAG的架构在构建稳健和隐私保护推荐系统方面的潜力。源代码可在以下链接获取：this https URL。 

---
# AI Brown and AI Koditex: LLM-Generated Corpora Comparable to Traditional Corpora of English and Czech Texts 

**Title (ZH)**: AI布鲁恩和AI科迪tex：由LLM生成的与英语和捷克语传统文本 CORPORA相当的语料库 

**Authors**: Jiří Milička, Anna Marklová, Václav Cvrček  

**Link**: [PDF](https://arxiv.org/pdf/2509.22996)  

**Abstract**: This article presents two corpora of English and Czech texts generated with large language models (LLMs). The motivation is to create a resource for comparing human-written texts with LLM-generated text linguistically. Emphasis was placed on ensuring these resources are multi-genre and rich in terms of topics, authors, and text types, while maintaining comparability with existing human-created corpora. These generated corpora replicate reference human corpora: BE21 by Paul Baker, which is a modern version of the original Brown Corpus, and Koditex corpus that also follows the Brown Corpus tradition but in Czech. The new corpora were generated using models from OpenAI, Anthropic, Alphabet, Meta, and DeepSeek, ranging from GPT-3 (davinci-002) to GPT-4.5, and are tagged according to the Universal Dependencies standard (i.e., they are tokenized, lemmatized, and morphologically and syntactically annotated). The subcorpus size varies according to the model used (the English part contains on average 864k tokens per model, 27M tokens altogether, the Czech partcontains on average 768k tokens per model, 21.5M tokens altogether). The corpora are freely available for download under the CC BY 4.0 license (the annotated data are under CC BY-NC-SA 4.0 licence) and are also accessible through the search interface of the Czech National Corpus. 

**Abstract (ZH)**: 本文介绍了使用大规模语言模型（LLMs）生成的英語和捷克语语料库。目的是为语言学上将人工撰写的文本与LLM生成的文本进行比较提供资源。强调确保这些资源具备多体裁和话题丰富等特点，同时保持与现有人工创建语料库的可比性。生成的新语料库模仿了参考的人工语料库：由保罗·拜尔克编制的BE21（现代版的原始布朗语料库），以及遵循布朗语料库传统的捷克Koditex语料库。新的语料库使用来自OpenAI、Anthropic、Alphabet、Meta和DeepSeek的模型生成，范围从GPT-3（davinci-002）到GPT-4.5，并按照通用依赖性注释标准进行标注（即它们被标记化、词干化，并且在形态学和句法上进行了注释）。子语料库的大小根据使用的模型而变化（英語部分平均每种模型包含864k个标记，总计27M个标记，捷克语部分平均每种模型包含768k个标记，总计21.5M个标记）。这些语料库在CC BY 4.0许可下免费提供下载（注释数据在CC BY-NC-SA 4.0许可下提供），并通过捷克国家语料库的搜索引擎接口进行访问。 

---
