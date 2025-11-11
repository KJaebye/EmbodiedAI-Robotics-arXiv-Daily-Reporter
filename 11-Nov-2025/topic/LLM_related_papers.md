# DeepPersona: A Generative Engine for Scaling Deep Synthetic Personas 

**Title (ZH)**: DeepPersona：一个生成深合成人设的引擎 

**Authors**: Zhen Wang, Yufan Zhou, Zhongyan Luo, Lyumanshan Ye, Adam Wood, Man Yao, Luoshang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2511.07338)  

**Abstract**: Simulating human profiles by instilling personas into large language models (LLMs) is rapidly transforming research in agentic behavioral simulation, LLM personalization, and human-AI alignment. However, most existing synthetic personas remain shallow and simplistic, capturing minimal attributes and failing to reflect the rich complexity and diversity of real human identities. We introduce DEEPPERSONA, a scalable generative engine for synthesizing narrative-complete synthetic personas through a two-stage, taxonomy-guided method. First, we algorithmically construct the largest-ever human-attribute taxonomy, comprising over hundreds of hierarchically organized attributes, by mining thousands of real user-ChatGPT conversations. Second, we progressively sample attributes from this taxonomy, conditionally generating coherent and realistic personas that average hundreds of structured attributes and roughly 1 MB of narrative text, two orders of magnitude deeper than prior works. Intrinsic evaluations confirm significant improvements in attribute diversity (32 percent higher coverage) and profile uniqueness (44 percent greater) compared to state-of-the-art baselines. Extrinsically, our personas enhance GPT-4.1-mini's personalized question answering accuracy by 11.6 percent on average across ten metrics and substantially narrow (by 31.7 percent) the gap between simulated LLM citizens and authentic human responses in social surveys. Our generated national citizens reduced the performance gap on the Big Five personality test by 17 percent relative to LLM-simulated citizens. DEEPPERSONA thus provides a rigorous, scalable, and privacy-free platform for high-fidelity human simulation and personalized AI research. 

**Abstract (ZH)**: 通过将人设注入大型语言模型（LLMs）模拟人类画像正迅速变革代理行为模拟、LLM个性化和人类-AI对齐的研究。然而，现有的大多数合成人设仍然浅显简陋，仅捕捉到最少的属性，未能反映真实人类身份的丰富复杂性和多样性。我们提出DEEPPERSONA，一种通过分类学引导的两阶段生成方法合成叙事完整合成人设的可扩展生成引擎。首先，我们通过挖掘数千名真实用户与ChatGPT的对话，构建迄今为止最大的人类属性分类学，包含数百个层次组织的属性。其次，我们从这个分类学中逐步采样属性，条件生成一致且现实的、平均包含数百个结构化属性和约1MB叙事文本的人设，深度比先前工作提高了两个数量级。内在评估表明，人设的属性多样性和个人特征显著提高（分别提高了32%和44%），超过了最先进的基线方法。外在评估显示，我们的合成人设使GPT-4.1-mini的个性化问答准确性平均提高11.6%，在社会调查中与真实人类回应的差距缩小了31.7%，并在大五人格测试中使表现差距相对减少了17%。因此，DEEPPERSONA提供了一个严格、可扩展且不侵犯隐私的平台，用于高保真的人类模拟和个人化AI研究。 

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
# Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning 

**Title (ZH)**: LLMs有情感吗？通过提示、检索和 curriculum 学习教学情感识别 

**Authors**: Xinran Li, Xiujuan Xu, Jiaqi Qiao, Yu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07061)  

**Abstract**: Emotion Recognition in Conversation (ERC) is a crucial task for understanding human emotions and enabling natural human-computer interaction. Although Large Language Models (LLMs) have recently shown great potential in this field, their ability to capture the intrinsic connections between explicit and implicit emotions remains limited. We propose a novel ERC training framework, PRC-Emo, which integrates Prompt engineering, demonstration Retrieval, and Curriculum learning, with the goal of exploring whether LLMs can effectively perceive emotions in conversational contexts. Specifically, we design emotion-sensitive prompt templates based on both explicit and implicit emotional cues to better guide the model in understanding the speaker's psychological states. We construct the first dedicated demonstration retrieval repository for ERC, which includes training samples from widely used datasets, as well as high-quality dialogue examples generated by LLMs and manually verified. Moreover, we introduce a curriculum learning strategy into the LoRA fine-tuning process, incorporating weighted emotional shifts between same-speaker and different-speaker utterances to assign difficulty levels to dialogue samples, which are then organized in an easy-to-hard training sequence. Experimental results on two benchmark datasets-- IEMOCAP and MELD --show that our method achieves new state-of-the-art (SOTA) performance, demonstrating the effectiveness and generalizability of our approach in improving LLM-based emotional understanding. 

**Abstract (ZH)**: 对话中的情绪识别（Emotion Recognition in Conversation, ERC）是理解人类情绪和实现自然人机交互的关键任务。尽管大型语言模型（LLMs）在该领域 recently 展现出巨大的潜力，但它们捕捉显性情绪和隐性情绪之间内在联系的能力仍然有限。我们提出了一种新颖的 ERC 训练框架 PRC-Emo，该框架结合了提示工程、演示检索和课程学习，旨在探索 LLMs 是否能够有效地感知对话情境下的情绪。具体而言，我们基于显性和隐性情绪线索设计了情绪敏感的提示模板，以更好地引导模型理解发言者的心理状态。我们构建了首个专门的演示检索仓库，其中包括来自广泛使用的数据集的训练样本，以及通过 LLM 生成并人工验证的高质量对话示例。此外，我们在 LoRA 微调过程中引入了课程学习策略，将同一发言者和不同发言者话语之间的情感权重变化纳入难度级别分配，并将对话样本按照易到难的训练序列进行组织。在两个基准数据集——IEMOCAP 和 MELD 上的实验结果表明，我们的方法在情感理解方面达到了新的最佳性能（SOTA），证明了我们方法在提高基于 LLM 的情感理解方面的有效性和泛化能力。 

---
# MathSE: Improving Multimodal Mathematical Reasoning via Self-Evolving Iterative Reflection and Reward-Guided Fine-Tuning 

**Title (ZH)**: MathSE: 通过自我进化迭代反思和奖励指导微调以提高多模态数学推理能力 

**Authors**: Jinhao Chen, Zhen Yang, Jianxin Shi, Tianyu Wo, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06805)  

**Abstract**: Multimodal large language models (MLLMs) have demonstrated remarkable capabilities in vision-language answering tasks. Despite their strengths, these models often encounter challenges in achieving complex reasoning tasks such as mathematical problem-solving. Previous works have focused on fine-tuning on specialized mathematical datasets. However, these datasets are typically distilled directly from teacher models, which capture only static reasoning patterns and leaving substantial gaps compared to student models. This reliance on fixed teacher-derived datasets not only restricts the model's ability to adapt to novel or more intricate questions that extend beyond the confines of the training data, but also lacks the iterative depth needed for robust generalization. To overcome these limitations, we propose \textbf{\method}, a \textbf{Math}ematical \textbf{S}elf-\textbf{E}volving framework for MLLMs. In contrast to traditional one-shot fine-tuning paradigms, \method iteratively refines the model through cycles of inference, reflection, and reward-based feedback. Specifically, we leverage iterative fine-tuning by incorporating correct reasoning paths derived from previous-stage inference and integrating reflections from a specialized Outcome Reward Model (ORM). To verify the effectiveness of \method, we evaluate it on a suite of challenging benchmarks, demonstrating significant performance gains over backbone models. Notably, our experimental results on MathVL-test surpass the leading open-source multimodal mathematical reasoning model QVQ. Our code and models are available at \texttt{https://zheny2751\this http URL\allowbreak this http URL}. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在视觉-语言问答任务中展现了出色的性能。尽管如此，这些模型在完成复杂的推理任务，如数学问题求解时常常遇到挑战。以往的工作侧重于在专门的数学数据集上进行微调。然而，这些数据集通常直接源自教师模型，只能捕捉静态的推理模式，与学生模型相比存在较大差距。对固定教师提取数据集的依赖不仅限制了模型适应新颖或更复杂的超出训练数据范围的问题的能力，还缺乏用于稳健泛化的迭代深度。为克服这些限制，我们提出了一种名为 \textbf{\method} 的多模态大型语言模型数学自我演化框架。与传统的单次微调范式不同，\method 通过推理、反思和基于奖励的反馈循环迭代优化模型。具体而言，我们通过将先前阶段推理中得出的正确推理路径和专门的成果奖励模型（ORM）的反思集成到迭代微调中来实现这一目标。为了验证 \method 的有效性，我们在一系列具有挑战性的基准测试中对其进行评估，显示出相对于骨干模型的显著性能提升。值得注意的是，我们在 MathVL-test 上的实验结果超越了领先的开源多模态数学推理模型 QVQ。相关代码和模型可在 \texttt{https://zheny2751\this http URL\allowbreak this http URL} 获取。 

---
# Spilling the Beans: Teaching LLMs to Self-Report Their Hidden Objectives 

**Title (ZH)**: 泄露豆子： teaching LLMs 自我报告其隐藏目标 

**Authors**: Chloe Li, Mary Phuong, Daniel Tan  

**Link**: [PDF](https://arxiv.org/pdf/2511.06626)  

**Abstract**: As AI systems become more capable of complex agentic tasks, they also become more capable of pursuing undesirable objectives and causing harm. Previous work has attempted to catch these unsafe instances by interrogating models directly about their objectives and behaviors. However, the main weakness of trusting interrogations is that models can lie. We propose self-report fine-tuning (SRFT), a simple supervised fine-tuning technique that trains models to admit their factual mistakes when asked. We show that the admission of factual errors in simple question-answering settings generalizes out-of-distribution (OOD) to the admission of hidden misaligned objectives in adversarial agentic settings. We evaluate SRFT in OOD stealth tasks, where models are instructed to complete a hidden misaligned objective alongside a user-specified objective without being caught by monitoring. After SRFT, models are more likely to confess the details of their hidden objectives when interrogated, even under strong pressure not to disclose them. Interrogation on SRFT models can detect hidden objectives with near-ceiling performance (F1 score = 0.98), while the baseline model lies when interrogated under the same conditions (F1 score = 0). Interrogation on SRFT models can further elicit the content of the hidden objective, recovering 28-100% details, compared to 0% details recovered in the baseline model and by prefilled assistant turn attacks. This provides a promising technique for promoting honesty propensity and incriminating misaligned AI systems. 

**Abstract (ZH)**: 随着AI系统在执行复杂代理任务方面的能力不断增强，它们在追求不良目标和造成危害方面的能力也不断增强。先前的工作试图通过直接询问模型其目标和行为来检测这些不安全实例。然而，依赖询问的主要弱点在于模型可以撒谎。我们提出了一种自我报告微调（SRFT）技术，这是一种简单的监督微调方法，训练模型在被问及时承认其事实错误。我们展示了在简单的问答设置中承认事实错误可以泛化到对抗性代理设置中，承认隐藏的不一致目标。我们在OOD隐形任务中评估了SRFT，模型被指示在不被监测捕获的情况下完成用户指定的目标和隐藏的不一致目标。经过SRFT训练后，模型在被询问时更有可能承认其隐藏目标的详细信息，即使在强压力下也不披露。对SRFT模型的询问可以检测隐藏目标，接近最佳性能（F1分数=0.98），而基线模型在相同条件下被询问时撒谎（F1分数=0）。对SRFT模型的询问可以进一步揭示隐藏目标的内容，恢复28%至100%的详细信息，而在基线模型和预填充助手轮攻击中均未恢复任何详细信息。这提供了一种有希望的技术，用于促进诚实倾向并指控不一致的AI系统。 

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
# SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards 

**Title (ZH)**: SpatialThinker: 在多模态LLM中通过空间奖励强化三维推理 

**Authors**: Hunar Batra, Haoqin Tu, Hardy Chen, Yuanze Lin, Cihang Xie, Ronald Clark  

**Link**: [PDF](https://arxiv.org/pdf/2511.07403)  

**Abstract**: Multimodal large language models (MLLMs) have achieved remarkable progress in vision-language tasks, but they continue to struggle with spatial understanding. Existing spatial MLLMs often rely on explicit 3D inputs or architecture-specific modifications, and remain constrained by large-scale datasets or sparse supervision. To address these limitations, we introduce SpatialThinker, a 3D-aware MLLM trained with RL to integrate structured spatial grounding with multi-step reasoning. The model simulates human-like spatial perception by constructing a scene graph of task-relevant objects and spatial relations, and reasoning towards an answer via dense spatial rewards. SpatialThinker consists of two key contributions: (1) a data synthesis pipeline that generates STVQA-7K, a high-quality spatial VQA dataset, and (2) online RL with a multi-objective dense spatial reward enforcing spatial grounding. SpatialThinker-7B outperforms supervised fine-tuning and the sparse RL baseline on spatial understanding and real-world VQA benchmarks, nearly doubling the base-model gain compared to sparse RL, and surpassing GPT-4o. These results showcase the effectiveness of combining spatial supervision with reward-aligned reasoning in enabling robust 3D spatial understanding with limited data and advancing MLLMs towards human-level visual reasoning. 

**Abstract (ZH)**: 具有空间意识的大规模多模态语言模型：通过强化学习集成结构化空间接地与多步推理 

---
# Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence 

**Title (ZH)**: 通过返初始化循环增强教预训练语言模型深度思考 

**Authors**: Sean McLeish, Ang Li, John Kirchenbauer, Dayal Singh Kalra, Brian R. Bartoldson, Bhavya Kailkhura, Avi Schwarzschild, Jonas Geiping, Tom Goldstein, Micah Goldblum  

**Link**: [PDF](https://arxiv.org/pdf/2511.07384)  

**Abstract**: Recent advances in depth-recurrent language models show that recurrence can decouple train-time compute and parameter count from test-time compute. In this work, we study how to convert existing pretrained non-recurrent language models into depth-recurrent models. We find that using a curriculum of recurrences to increase the effective depth of the model over the course of training preserves performance while reducing total computational cost. In our experiments, on mathematics, we observe that converting pretrained models to recurrent ones results in better performance at a given compute budget than simply post-training the original non-recurrent language model. 

**Abstract (ZH)**: Recent advances in深度递归语言模型表明，递归可以分离训练时计算量和参数量与测试时计算量的关系。在这项工作中，我们研究如何将现有的非递归预训练语言模型转换为深度递归模型。我们发现，在训练过程中使用递归课程逐步增加模型的有效深度可以在保持性能的同时降低总计算成本。在我们的实验中，对于数学任务，我们将预训练模型转换为递归模型在给定计算预算下比对原非递归语言模型进行后训练表现更好。 

---
# Self-Evaluating LLMs for Multi-Step Tasks: Stepwise Confidence Estimation for Failure Detection 

**Title (ZH)**: 自评估大语言模型用于多步任务：逐步-confidence估计以检测失败 

**Authors**: Vaibhav Mavi, Shubh Jaroria, Weiqi Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.07364)  

**Abstract**: Reliability and failure detection of large language models (LLMs) is critical for their deployment in high-stakes, multi-step reasoning tasks. Prior work explores confidence estimation for self-evaluating LLM-scorer systems, with confidence scorers estimating the likelihood of errors in LLM responses. However, most methods focus on single-step outputs and overlook the challenges of multi-step reasoning. In this work, we extend self-evaluation techniques to multi-step tasks, testing two intuitive approaches: holistic scoring and step-by-step scoring. Using two multi-step benchmark datasets, we show that stepwise evaluation generally outperforms holistic scoring in detecting potential errors, with up to 15% relative increase in AUC-ROC. Our findings demonstrate that self-evaluating LLM systems provide meaningful confidence estimates in complex reasoning, improving their trustworthiness and providing a practical framework for failure detection. 

**Abstract (ZH)**: 大型语言模型（LLMs）的可靠性和故障检测对于它们在高风险多步推理任务中的部署至关重要。前期工作探讨了自评估LLM评分系统中的置信度估计，其中置信度评分器估算LLM响应中错误的可能性。然而，大多数方法专注于单步输出并忽视了多步推理的挑战。在本工作中，我们将自评估技术扩展到多步任务，测试了两种直观的方法：整体评分和逐步评分。使用两个多步基准数据集，我们表明逐步评估通常在检测潜在错误方面优于整体评分，AUC-ROC相对增加高达15%。我们的研究结果表明，自评估LLM系统在复杂推理中提供了有意义的置信度估计，提高了其可信度，并为故障检测提供了实用框架。 

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
# AdaRec: Adaptive Recommendation with LLMs via Narrative Profiling and Dual-Channel Reasoning 

**Title (ZH)**: AdaRec：通过叙事画像和双重通道推理的适应性推荐方法 

**Authors**: Meiyun Wang, Charin Polpanumas  

**Link**: [PDF](https://arxiv.org/pdf/2511.07166)  

**Abstract**: We propose AdaRec, a few-shot in-context learning framework that leverages large language models for an adaptive personalized recommendation. AdaRec introduces narrative profiling, transforming user-item interactions into natural language representations to enable unified task handling and enhance human readability. Centered on a bivariate reasoning paradigm, AdaRec employs a dual-channel architecture that integrates horizontal behavioral alignment, discovering peer-driven patterns, with vertical causal attribution, highlighting decisive factors behind user preferences. Unlike existing LLM-based approaches, AdaRec eliminates manual feature engineering through semantic representations and supports rapid cross-task adaptation with minimal supervision. Experiments on real ecommerce datasets demonstrate that AdaRec outperforms both machine learning models and LLM-based baselines by up to eight percent in few-shot settings. In zero-shot scenarios, it achieves up to a nineteen percent improvement over expert-crafted profiling, showing effectiveness for long-tail personalization with minimal interaction data. Furthermore, lightweight fine-tuning on synthetic data generated by AdaRec matches the performance of fully fine-tuned models, highlighting its efficiency and generalization across diverse tasks. 

**Abstract (ZH)**: AdaRec：一种利用大规模语言模型实现自适应个性化推荐的少样本在上下文中学习框架 

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
# More Agents Helps but Adversarial Robustness Gap Persists 

**Title (ZH)**: 更多代理有助于提高，但对抗鲁棒性差距依然存在 

**Authors**: Khashayar Alavi, Zhastay Yeltay, Lucie Flek, Akbar Karimi  

**Link**: [PDF](https://arxiv.org/pdf/2511.07112)  

**Abstract**: When LLM agents work together, they seem to be more powerful than a single LLM in mathematical question answering. However, are they also more robust to adversarial inputs? We investigate this question using adversarially perturbed math questions. These perturbations include punctuation noise with three intensities (10, 30, and 50 percent), plus real-world and human-like typos (WikiTypo, R2ATA). Using a unified sampling-and-voting framework (Agent Forest), we evaluate six open-source models (Qwen3-4B/14B, Llama3.1-8B, Mistral-7B, Gemma3-4B/12B) across four benchmarks (GSM8K, MATH, MMLU-Math, MultiArith), with various numbers of agents n from one to 25 (1, 2, 5, 10, 15, 20, 25). Our findings show that (1) Noise type matters: punctuation noise harm scales with its severity, and the human typos remain the dominant bottleneck, yielding the largest gaps to Clean accuracy and the highest ASR even with a large number of agents. And (2) Collaboration reliably improves accuracy as the number of agents, n, increases, with the largest gains from one to five agents and diminishing returns beyond 10 agents. However, the adversarial robustness gap persists regardless of the agent count. 

**Abstract (ZH)**: 当LLM代理协同工作时，它们在数学问题回答中似乎比单个LLM更强大。然而，它们也更 robust 吗？我们使用对抗性扰动数学问题来研究这一问题。这些扰动包括三种强度的标点符号噪音（10%、30% 和 50%）以及实际世界和类人类的拼写错误（WikiTypo、R2ATA）。我们使用统一的采样和投票框架（Agent Forest）评估了六种开源模型（Qwen3-4B/14B、Llama3.1-8B、Mistral-7B、Gemma3-4B/12B）在四个基准（GSM8K、MATH、MMLU-Math、MultiArith）上的表现，代理数量 n 从 1 增加到 25（1, 2, 5, 10, 15, 20, 25）。我们的发现表明：（1）噪音类型很重要：标点符号噪音的危害随其严重程度而增加，人类拼写错误仍然是占主导地位的瓶颈，即使代理数量很大，也导致最大的准确率差距和最高的误报率。（2）随着代理数量 n 增加，合作可以可靠地提高准确性，从一个到五个代理时收益最大，超过 10 代理时收益递减。然而，无论代理数量如何，对抗性鲁棒性差距仍然存在。 

---
# E2E-VGuard: Adversarial Prevention for Production LLM-based End-To-End Speech Synthesis 

**Title (ZH)**: E2E-VGuard：面向生产级LLM的端到端语音合成对抗防护 

**Authors**: Zhisheng Zhang, Derui Wang, Yifan Mi, Zhiyong Wu, Jie Gao, Yuxin Cao, Kai Ye, Minhui Xue, Jie Hao  

**Link**: [PDF](https://arxiv.org/pdf/2511.07099)  

**Abstract**: Recent advancements in speech synthesis technology have enriched our daily lives, with high-quality and human-like audio widely adopted across real-world applications. However, malicious exploitation like voice-cloning fraud poses severe security risks. Existing defense techniques struggle to address the production large language model (LLM)-based speech synthesis. While previous studies have considered the protection for fine-tuning synthesizers, they assume manually annotated transcripts. Given the labor intensity of manual annotation, end-to-end (E2E) systems leveraging automatic speech recognition (ASR) to generate transcripts are becoming increasingly prevalent, e.g., voice cloning via commercial APIs. Therefore, this E2E speech synthesis also requires new security mechanisms. To tackle these challenges, we propose E2E-VGuard, a proactive defense framework for two emerging threats: (1) production LLM-based speech synthesis, and (2) the novel attack arising from ASR-driven E2E scenarios. Specifically, we employ the encoder ensemble with a feature extractor to protect timbre, while ASR-targeted adversarial examples disrupt pronunciation. Moreover, we incorporate the psychoacoustic model to ensure perturbative imperceptibility. For a comprehensive evaluation, we test 16 open-source synthesizers and 3 commercial APIs across Chinese and English datasets, confirming E2E-VGuard's effectiveness in timbre and pronunciation protection. Real-world deployment validation is also conducted. Our code and demo page are available at this https URL. 

**Abstract (ZH)**: Recent advancements in speech synthesis technology have enriched our daily lives, with high-quality and human-like audio widely adopted across real-world applications.然而，恶意利用如声音克隆欺诈产生了严重的安全风险。现有的防御技术难以应对基于大规模语言模型（LLM）的语音合成生产。虽然已有研究考虑了对合成器微调的保护，但这些研究假设有手动标注的脚本。鉴于手动标注的劳动强度，利用自动语音识别（ASR）生成脚本的端到端（E2E）系统正变得越来越普遍，例如通过商业API进行的声音克隆。因此，这种E2E语音合成也需要新的安全机制。为应对这些挑战，我们提出了E2E-VGuard，这是一种针对两大新兴威胁的前瞻防御框架：（1）基于LLM的语音合成生产，（2）来自ASR驱动的E2E场景的新攻击。具体而言，我们采用编码器组合和特征提取器来保护音色，而针对ASR的对抗样本则破坏发音。此外，我们还结合了听觉心理模型以确保扰动的不可感知性。为了进行全面评估，我们在中文和英文数据集上测试了16个开源合成器和3个商业API，证实了E2E-VGuard在音色和发音保护方面的有效性。我们还在现实世界的部署中进行了验证。我们的代码和演示页面可在此处访问。 

---
# Achieving Effective Virtual Reality Interactions via Acoustic Gesture Recognition based on Large Language Models 

**Title (ZH)**: 基于大型语言模型的声学手势识别实现有效虚拟现实交互 

**Authors**: Xijie Zhang, Fengliang He, Hong-Ning Dai  

**Link**: [PDF](https://arxiv.org/pdf/2511.07085)  

**Abstract**: Natural and efficient interaction remains a critical challenge for virtual reality and augmented reality (VR/AR) systems. Vision-based gesture recognition suffers from high computational cost, sensitivity to lighting conditions, and privacy leakage concerns. Acoustic sensing provides an attractive alternative: by emitting inaudible high-frequency signals and capturing their reflections, channel impulse response (CIR) encodes how gestures perturb the acoustic field in a low-cost and user-transparent manner. However, existing CIR-based gesture recognition methods often rely on extensive training of models on large labeled datasets, making them unsuitable for few-shot VR scenarios. In this work, we propose the first framework that leverages large language models (LLMs) for CIR-based gesture recognition in VR/AR systems. Despite LLMs' strengths, it is non-trivial to achieve few-shot and zero-shot learning of CIR gestures due to their inconspicuous features. To tackle this challenge, we collect differential CIR rather than original CIR data. Moreover, we construct a real-world dataset collected from 10 participants performing 15 gestures across three categories (digits, letters, and shapes), with 10 repetitions each. We then conduct extensive experiments on this dataset using an LLM-adopted classifier. Results show that our LLM-based framework achieves accuracy comparable to classical machine learning baselines, while requiring no domain-specific retraining. 

**Abstract (ZH)**: 自然且高效的交互仍然是虚拟现实和增强现实（VR/AR）系统的关键挑战。基于视觉的手势识别面临着高计算成本、对光照条件敏感以及隐私泄露担忧的问题。声学传感提供了一种有吸引力的替代方案：通过发射不可闻的高频信号并捕获其反射，信道冲激响应（CIR）以低成本和用户透明的方式编码手势对声场的扰动。然而，现有的基于CIR的手势识别方法通常依赖于在大型标注数据集上对模型进行广泛的训练，这使它们无法适用于少量样本的VR场景。在本文中，我们提出了首个利用大规模语言模型（LLMs）进行基于CIR的手势识别的框架。尽管大规模语言模型具有优势，但由于其特征不显着，实现针对CIR手势的少量样本和零样本学习仍具有挑战性。为应对这一挑战，我们收集了差异CIR数据而非原始CIR数据。此外，我们构建了一个由10名参与者完成的涵盖三个类别（数字、字母和形状）15种手势的现实世界数据集，每种手势重复10次。然后，我们使用采用大规模语言模型的分类器在该数据集上进行了大量实验。结果表明，我们的基于大规模语言模型的方法在准确率方面与经典机器学习基线相当，同时不需要特定领域的重新训练。 

---
# Wasm: A Pipeline for Constructing Structured Arabic Interleaved Multimodal Corpora 

**Title (ZH)**: Wasm：构建结构化阿拉伯混合模态语料库的管道 

**Authors**: Khalil Hennara, Ahmad Bastati, Muhammad Hreden, Mohamed Motasim Hamed, Zeina Aldallal, Sara Chrouf, Safwan AlModhayan  

**Link**: [PDF](https://arxiv.org/pdf/2511.07080)  

**Abstract**: The performance of large language models (LLMs) and large multimodal models (LMMs) depends heavily on the quality and scale of their pre-training datasets. Recent research shows that large multimodal models trained on natural documents where images and text are interleaved outperform those trained only on image-text pairs across a wide range of benchmarks, leveraging advanced pre- trained models to enforce semantic alignment, image-sequence consistency, and textual coherence. For Arabic, however, the lack of high-quality multimodal datasets that preserve document structure has limited progress. In this paper, we present our pipeline Wasm for processing the Common Crawl dataset to create a new Arabic multimodal dataset that uniquely provides markdown output. Unlike existing Arabic corpora that focus solely on text extraction, our approach preserves the structural integrity of web content while maintaining flexibility for both text-only and multimodal pre-training scenarios. We provide a comprehensive comparative analysis of our data processing pipeline against those used for major existing datasets, highlighting the convergences in filtering strategies and justifying our specific design choices. To support future research, we publicly release a representative dataset dump along with the multimodal processing pipeline for Arabic. 

**Abstract (ZH)**: 大型语言模型（LLMs）和大型多模态模型（LMMs）的性能高度依赖于其预训练数据集的质量和规模。近期研究显示，多模态模型在自然文档上进行预训练，其中图像和文本交织，相较于仅在图像-文本配对上进行预训练，在多种基准测试中表现更优，利用先进的预训练模型来加强语义对齐、图像序列一致性以及文本连贯性。然而，对于阿拉伯语而言，缺乏能够保留文档结构的高质量多模态数据集限制了研究进展。在本文中，我们介绍了我们的数据处理管道Wasm，用于处理Common Crawl数据集，以创建一个新的阿拉伯语多模态数据集，该数据集独特地提供了markdown输出。与仅专注于文本提取的现有阿拉伯语语料库不同，我们的方法保留了网络内容的结构完整性，同时为仅文本和多模态预训练场景提供了灵活性。我们对数据处理管道进行了全面的比较分析，与主要现有数据集使用的管道进行了对比，突出了过滤策略的交汇之处，并解释了我们的特定设计选择。为了支持未来的研究，我们公开发布了代表性数据集的快照以及阿拉伯语的多模态处理管道。 

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
# Differentiated Directional Intervention A Framework for Evading LLM Safety Alignment 

**Title (ZH)**: 差异化方向性干预：一种规避LLM安全对齐的框架 

**Authors**: Peng Zhang, peijie sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.06852)  

**Abstract**: Safety alignment instills in Large Language Models (LLMs) a critical capacity to refuse malicious requests. Prior works have modeled this refusal mechanism as a single linear direction in the activation space. We posit that this is an oversimplification that conflates two functionally distinct neural processes: the detection of harm and the execution of a refusal. In this work, we deconstruct this single representation into a Harm Detection Direction and a Refusal Execution Direction. Leveraging this fine-grained model, we introduce Differentiated Bi-Directional Intervention (DBDI), a new white-box framework that precisely neutralizes the safety alignment at critical layer. DBDI applies adaptive projection nullification to the refusal execution direction while suppressing the harm detection direction via direct steering. Extensive experiments demonstrate that DBDI outperforms prominent jailbreaking methods, achieving up to a 97.88\% attack success rate on models such as Llama-2. By providing a more granular and mechanistic framework, our work offers a new direction for the in-depth understanding of LLM safety alignment. 

**Abstract (ZH)**: 安全对齐赋予大型语言模型（LLMs）一种关键能力，即拒绝恶意请求。先前的工作将这种拒绝机制建模为激活空间中的单一线性方向。我们认为这是对两个功能上不同的神经过程的简化综合：危害检测和拒绝执行。在这项工作中，我们将这种单一表示分解为危害检测方向和拒绝执行方向。基于这一细粒度模型，我们引入了差异化双向干预（DBDI），这是一种新的白盒框架，精确地中和了关键层的安全对齐。DBDI在拒绝执行方向上应用自适应投影消除，同时通过直接控制抑制危害检测方向。广泛的实验表明，DBDI在如Llama-2等模型上的攻击成功率高达97.88%。通过提供更精细和机制化的框架，我们的工作为LLM安全对齐的深入理解提供了新的方向。 

---
# Beyond Plain Demos: A Demo-centric Anchoring Paradigm for In-Context Learning in Alzheimer's Disease Detection 

**Title (ZH)**: 超越普通演示：面向演示的上下文学习 paradigm 在阿尔茨海默病检测中的应用 

**Authors**: Puzhen Su, Haoran Yin, Yongzhu Miao, Jintao Tang, Shasha Li, Ting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06826)  

**Abstract**: Detecting Alzheimer's disease (AD) from narrative transcripts challenges large language models (LLMs): pre-training rarely covers this out-of-distribution task, and all transcript demos describe the same scene, producing highly homogeneous contexts. These factors cripple both the model's built-in task knowledge (\textbf{task cognition}) and its ability to surface subtle, class-discriminative cues (\textbf{contextual perception}). Because cognition is fixed after pre-training, improving in-context learning (ICL) for AD detection hinges on enriching perception through better demonstration (demo) sets. We demonstrate that standard ICL quickly saturates, its demos lack diversity (context width) and fail to convey fine-grained signals (context depth), and that recent task vector (TV) approaches improve broad task adaptation by injecting TV into the LLMs' hidden states (HSs), they are ill-suited for AD detection due to the mismatch of injection granularity, strength and position. To address these bottlenecks, we introduce \textbf{DA4ICL}, a demo-centric anchoring framework that jointly expands context width via \emph{\textbf{Diverse and Contrastive Retrieval}} (DCR) and deepens each demo's signal via \emph{\textbf{Projected Vector Anchoring}} (PVA) at every Transformer layer. Across three AD benchmarks, DA4ICL achieves large, stable gains over both ICL and TV baselines, charting a new paradigm for fine-grained, OOD and low-resource LLM adaptation. 

**Abstract (ZH)**: 从叙事转录中检测阿尔茨海默病（AD）挑战大型语言模型（LLMs）：预训练鲜少涵盖此类离分布任务，且所有转录示例描述相同场景，产生高度同质化背景。这些因素削弱了模型内置的任务认知（task cognition）和上下文感知（contextual perception）能力。由于认知在预训练后固定，通过改进在情景学习（ICL）中的表现以提升AD检测依赖于通过更高质量的示例集（demo sets）丰富感知。我们证明标准ICL很快达到饱和，其示例缺乏多样性（背景宽度），未能传达细微的信号（背景深度），而近期的任务向量（TV）方法通过将TV注入LLM的隐藏状态（HSs）以增强广泛的任务适应性，但由于注入的颗粒度、强度和位置的不匹配，这些方法对于AD检测并不适用。为克服这些瓶颈，我们引入了DA4ICL，这是一种以示例为中心的耦合框架，通过多样对比检索（DCR）扩展背景宽度，并在每个变换器层通过投影向量锚定（PVA）加深每个示例的信号，从而为大规模、离分布和低资源LLM的适应开辟新范式。 

---
# Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models 

**Title (ZH)**: 跨模态不可学习性通过多模态大型语言模型中关键神经元路径编辑实现 

**Authors**: Kunhao Li, Wenhao Li, Di Wu, Lei Yang, Jun Bai, Ju Jia, Jason Xue  

**Link**: [PDF](https://arxiv.org/pdf/2511.06793)  

**Abstract**: Multimodal Large Language Models (MLLMs) extend foundation models to real-world applications by integrating inputs such as text and vision. However, their broad knowledge capacity raises growing concerns about privacy leakage, toxicity mitigation, and intellectual property violations. Machine Unlearning (MU) offers a practical solution by selectively forgetting targeted knowledge while preserving overall model utility. When applied to MLLMs, existing neuron-editing-based MU approaches face two fundamental challenges: (1) forgetting becomes inconsistent across modalities because existing point-wise attribution methods fail to capture the structured, layer-by-layer information flow that connects different modalities; and (2) general knowledge performance declines when sensitive neurons that also support important reasoning paths are pruned, as this disrupts the model's ability to generalize. To alleviate these limitations, we propose a multimodal influential neuron path editor (MIP-Editor) for MU. Our approach introduces modality-specific attribution scores to identify influential neuron paths responsible for encoding forget-set knowledge and applies influential-path-aware neuron-editing via representation misdirection. This strategy also enables effective and coordinated forgetting across modalities while preserving the model's general capabilities. Experimental results demonstrate that MIP-Editor achieves a superior unlearning performance on multimodal tasks, with a maximum forgetting rate of 87.75% and up to 54.26% improvement in general knowledge retention. On textual tasks, MIP-Editor achieves up to 80.65% forgetting and preserves 77.9% of general performance. Codes are available at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型的机器遗忘：多模态有影响神经路径编辑（MIP-Editor） 

---
# Data Trajectory Alignment for LLM Domain Adaptation: A Two-Phase Synthesis Framework for Telecommunications Mathematics 

**Title (ZH)**: 数据轨迹对齐以实现LLM领域适应：电信数学领域的两阶段综合框架 

**Authors**: Zhicheng Zhou, Jing Li, Suming Qiu, Junjie Huang, Linyuan Qiu, Zhijie Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.06776)  

**Abstract**: General-purpose large language models (LLMs) are increasingly deployed in verticals such as telecommunications, where adaptation is hindered by scarce, low-information-density corpora and tight mobile/edge constraints. We propose Data Trajectory Alignment (DTA), a two-phase, model-agnostic data curation framework that treats solution processes - not only final answers - as first-class supervision. Phase I (Initializing) synthesizes diverse, high-coverage candidates using an ensemble of strong teachers. Phase II (DTA) rewrites teacher solutions to align intermediate steps and presentation style with the target student's inductive biases and then performs signal-aware exemplar selection via agreement checks and reflection-based judging. Instantiated on telecommunications mathematics (e.g., link budgets, SNR/AMC selection, and power-control feasibility), DTA yields state-of-the-art (SOTA) accuracy on TELEMATH without enabling explicit "thinking" modes: 72.45% pass@1, surpassing distilled-only training by +17.65 points and outperforming a strong baseline (Qwen3-32B with thinking enabled) by +2.94 points. Token-shift analyses indicate that DTA concentrates gains on logical-structural discourse markers rather than merely amplifying domain nouns, indicating improved reasoning scaffolding. Under edge-like inference settings, DTA improves efficiency by reducing reliance on multi-sample voting and disabling expensive reasoning heuristics, cutting energy per output token by ~42% versus Qwen3-32B (thinking mode enabled) and end-to-end latency by ~60% versus Qwen3-32B (thinking mode disabled). These results demonstrate that aligning how solutions are produced enables compact, high-yield supervision that is effective for both accuracy and efficiency, offering a practical recipe for domain adaptation in low-resource verticals beyond telecom. 

**Abstract (ZH)**: 通用大语言模型在电信等垂直领域的应用受限于稀缺和信息密度低的数据集以及移动/边缘计算的限制。我们提出了一种名为数据轨迹对齐（DTA）的两阶段、模型无关的数据整理框架，该框架将解决方案过程——不仅仅是最终答案——视为一级监督。第一阶段（初始化）使用一组强大教师生成多样性和高覆盖的候选方案。第二阶段（DTA）重写教师解决方案，使其中间步骤和呈现风格与目标学生的归纳偏见保持一致，然后通过一致性和反射判断进行信号感知的范例选择。在电信数学（例如，链路预算、信噪比/ AMC选择和功率控制可行性）上实例化，DTA在TELEMATH上取得了最先进的准确性，而无需启用显式的“思考”模式：在1 pass上达到了72.45%的准确性，超越了仅蒸馏训练17.65个百分点，并且比启用了思考模式的强劲基线（Qwen3-32B）高出2.94个百分点。字位移分析表明，DTA将收益集中在逻辑结构 discourse 标记上，而不是仅仅放大领域名词，表明改进了推理支撑结构。在边缘计算场景下，DTA通过减少多样本投票依赖并禁用昂贵的推理启发式方法，提高了效率，相较于启用了思考模式的Qwen3-32B，每输出字的能量减少了约42%，端到端延迟减少了约60%。这些结果表明，对如何生成解决方案进行对齐可以提供紧凑高效的监督，对准确性和效率都有好处，并提供了一种实用的方法，在电信等资源有限的垂直领域进行领域适应。 

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
# Rank-1 LoRAs Encode Interpretable Reasoning Signals 

**Title (ZH)**: Rank-1 LoRAs Encode Interpretable Reasoning Signals 

**Authors**: Jake Ward, Paul Riechers, Adam Shai  

**Link**: [PDF](https://arxiv.org/pdf/2511.06739)  

**Abstract**: Reasoning models leverage inference-time compute to significantly enhance the performance of language models on difficult logical tasks, and have become a dominating paradigm in frontier LLMs. Despite their wide adoption, the mechanisms underpinning the enhanced performance of these reasoning models are not well understood. In this work, we show that the majority of new capabilities in reasoning models can be elicited by small, single-rank changes to base model parameters, with many of these changes being interpretable. Specifically, we use a rank-1 LoRA to create a minimal parameter adapter for Qwen-2.5-32B-Instruct which recovers 73-90% of reasoning-benchmark performance compared to a full parameter finetune. We find that the activations of this LoRA are as interpretable as MLP neurons, and fire for reasoning-specific behaviors. Finally, we train a sparse autoencoder on the entire activation state of this LoRA and identify fine-grained and monosemantic features. Our findings highlight that reasoning performance can arise largely from minimal changes to base model parameters, and explore what these changes affect. More broadly, our work shows that parameter-efficient training methods can be used as a targeted lens for uncovering fundamental insights about language model behavior and dynamics. 

**Abstract (ZH)**: 推理模型通过在推理时使用计算资源显著提升了语言模型在困难逻辑任务上的性能，并已成为前沿大规模语言模型的主导范式。尽管这些推理模型已被广泛采用，但其性能提升的机制尚不完全理解。在本文中，我们证明了大多数推理模型的新能力可以通过对基础模型参数进行小的、单极性的调整来激发，其中许多调整是可解释的。具体而言，我们使用秩1 LoRA为Qwen-2.5-32B-Instruct创建了一个最小参数适配器，该适配器在与全面参数微调相比的情况下，恢复了73-90%的推理基准性能。我们发现，该LoRA的激活具有与MLP神经元相当的可解释性，并针对推理特定的行为进行激活。最后，我们对整个LoRA的激活状态进行了稀疏自编码器训练，并识别出细粒度和单义特征。我们的研究结果强调，推理性能主要由基础模型参数的小调整引起，并探索这些调整的影响。更广泛地说，我们的工作表明，参数高效训练方法可以作为有针对性的透镜，用于揭示语言模型行为和动力学的基本洞察。 

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
# Place Matters: Comparing LLM Hallucination Rates for Place-Based Legal Queries 

**Title (ZH)**: 地点 Matters：基于地点的法律查询中大模型幻觉率的比较 

**Authors**: Damian Curran, Vanessa Sporne, Lea Frermann, Jeannie Paterson  

**Link**: [PDF](https://arxiv.org/pdf/2511.06700)  

**Abstract**: How do we make a meaningful comparison of a large language model's knowledge of the law in one place compared to another? Quantifying these differences is critical to understanding if the quality of the legal information obtained by users of LLM-based chatbots varies depending on their location. However, obtaining meaningful comparative metrics is challenging because legal institutions in different places are not themselves easily comparable. In this work we propose a methodology to obtain place-to-place metrics based on the comparative law concept of functionalism. We construct a dataset of factual scenarios drawn from Reddit posts by users seeking legal advice for family, housing, employment, crime and traffic issues. We use these to elicit a summary of a law from the LLM relevant to each scenario in Los Angeles, London and Sydney. These summaries, typically of a legislative provision, are manually evaluated for hallucinations. We show that the rate of hallucination of legal information by leading closed-source LLMs is significantly associated with place. This suggests that the quality of legal solutions provided by these models is not evenly distributed across geography. Additionally, we show a strong negative correlation between hallucination rate and the frequency of the majority response when the LLM is sampled multiple times, suggesting a measure of uncertainty of model predictions of legal facts. 

**Abstract (ZH)**: 如何在不同地区之间有意义地比较大型语言模型的法律知识？量化这些差异对于理解用户使用基于LLM的聊天机器人获取的法律信息质量是否取决于其地理位置至关重要。然而，由于不同地方的法律制度本身不易比较，因此获取有意义的比较指标具有挑战性。在本研究中，我们提出了一种基于比较法功能主义概念的方法论来获得地区间的指标。我们构建了一个数据集，其中包含从寻求家庭、住房、就业、犯罪和交通法律咨询的Reddit帖子中抽取的实际情况场景。我们使用这些数据来从洛杉矶、伦敦和悉尼的LLM中提取与每个场景相关的法律总结。这些总结通常是对立法条款的描述，并由人工评估是否存在幻觉。结果显示，顶级闭源LLM在法律信息上的幻觉率与地区显著相关。这表明这些模型提供的法律解决方案的质量在地理上并不是均匀分布的。此外，我们还展示了幻觉率与LLM多轮抽样的多数响应频率之间存在强烈负相关，这表明了模型对法律事实预测的不确定性程度。 

---
# Textual Self-attention Network: Test-Time Preference Optimization through Textual Gradient-based Attention 

**Title (ZH)**: 文本自注意力网络：基于文本梯度注意力的测试时偏好优化 

**Authors**: Shibing Mo, Haoyang Ruan, Kai Wu, Jing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06682)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable generalization capabilities, but aligning their outputs with human preferences typically requires expensive supervised fine-tuning. Recent test-time methods leverage textual feedback to overcome this, but they often critique and revise a single candidate response, lacking a principled mechanism to systematically analyze, weigh, and synthesize the strengths of multiple promising candidates. Such a mechanism is crucial because different responses may excel in distinct aspects (e.g., clarity, factual accuracy, or tone), and combining their best elements may produce a far superior outcome. This paper proposes the Textual Self-Attention Network (TSAN), a new paradigm for test-time preference optimization that requires no parameter updates. TSAN emulates self-attention entirely in natural language to overcome this gap: it analyzes multiple candidates by formatting them into textual keys and values, weighs their relevance using an LLM-based attention module, and synthesizes their strengths into a new, preference-aligned response under the guidance of the learned textual attention. This entire process operates in a textual gradient space, enabling iterative and interpretable optimization. Empirical evaluations demonstrate that with just three test-time iterations on a base SFT model, TSAN outperforms supervised models like Llama-3.1-70B-Instruct and surpasses the current state-of-the-art test-time alignment method by effectively leveraging multiple candidate solutions. 

**Abstract (ZH)**: 大规模语言模型（LLMs）展示了 remarkable 的泛化能力，但使其输出与人类偏好一致通常需要昂贵的监督微调。最近的测试时方法利用文本反馈来克服这一问题，但它们通常仅评判和修订单一候选回复，缺乏系统分析、权衡和综合多个有潜力候选回复优势的原理性机制。这种机制至关重要，因为不同回复可能在不同的方面表现出色（例如，清晰度、事实准确性或语气），结合它们的最佳元素可能会产生远远优于单一回复的结果。本文提出了文本自我注意力网络（TSAN），这是一种新的测试时偏好优化范式，不需要参数更新。TSAN 通过完全用自然语言模拟自我注意力来弥补这一差距：通过格式化多个候选回复为文本键值、利用基于语言模型的注意力模块评估其相关性，并在学习到的文本注意力引导下合成一个新的、与偏好一致的回复。整个过程在文本梯度空间中进行，实现迭代和可解释的优化。实证评估表明，仅通过在基础 SFT 模型上进行三次测试时迭代，TSAN 就能够优于如 Llama-3.1-70B-Instruct 等监督模型，并且能够有效地利用多个候选解决方案，超越当前最先进的测试时对齐方法。 

---
# SPUR: A Plug-and-Play Framework for Integrating Spatial Audio Understanding and Reasoning into Large Audio-Language Models 

**Title (ZH)**: SPUR：一种Plug-and-Play框架，用于将空间音频理解与推理集成到大型音频语言模型中 

**Authors**: S Sakshi, Vaibhavi Lokegaonkar, Neil Zhang, Ramani Duraiswami, Sreyan Ghosh, Dinesh Manocha, Lie Lu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06606)  

**Abstract**: Spatial perception is central to auditory intelligence, enabling accurate understanding of real-world acoustic scenes and advancing human-level perception of the world around us. While recent large audio-language models (LALMs) show strong reasoning over complex audios, most operate on monaural inputs and lack the ability to capture spatial cues such as direction, elevation, and distance. We introduce SPUR, a lightweight, plug-in approach that equips LALMs with spatial perception through minimal architectural changes. SPUR consists of: (i) a First-Order Ambisonics (FOA) encoder that maps (W, X, Y, Z) channels to rotation-aware, listener-centric spatial features, integrated into target LALMs via a multimodal adapter; and (ii) SPUR-Set, a spatial QA dataset combining open-source FOA recordings with controlled simulations, emphasizing relative direction, elevation, distance, and overlap for supervised spatial reasoning. Fine-tuning our model on the SPUR-Set consistently improves spatial QA and multi-speaker attribution while preserving general audio understanding. SPUR provides a simple recipe that transforms monaural LALMs into spatially aware models. Extensive ablations validate the effectiveness of our approach. 

**Abstract (ZH)**: 空间感知是听觉智能的核心，使人们对真实世界的声景有准确的理解，并推动我们周围世界的人类级感知。尽管最近的大规模音频-语言模型（LALMs）在复杂音频上的推理能力很强，但大多数模型仅处理单声道输入，并缺乏捕捉方向、仰角和距离等空间线索的能力。我们引入了SPUR，这是一种轻量级且可插拔的方法，通过最小的架构更改为LALMs赋予空间感知。SPUR包括：（i）一阶球面声学（FOA）编码器，将（W, X, Y, Z）通道映射为旋转感知的、以听者为中心的空间特征，并通过多模态适配器整合到目标LALMs中；以及（ii）SPUR-Set，这是一个结合开源FOA录音与受控模拟的空间QA数据集，强调相对方向、仰角、距离和重叠，以进行监督的空间推理。在SPUR-Set上微调我们的模型在提升空间QA和多说话人属性方面取得了持续改善，同时保持一般音频理解。SPUR提供了一个简单的配方，将单声道LALMs转化为具有空间意识的模型。广泛的经验消除实验验证了我们方法的有效性。 

---
# CoFineLLM: Conformal Finetuning of LLMs for Language-Instructed Robot Planning 

**Title (ZH)**: CoFineLLM: 语言指导的机器人规划中LLMs的齐性微调 

**Authors**: Jun Wang, Yevgeniy Vorobeychik, Yiannis Kantaros  

**Link**: [PDF](https://arxiv.org/pdf/2511.06575)  

**Abstract**: Large Language Models (LLMs) have recently emerged as planners for language-instructed agents, generating sequences of actions to accomplish natural language tasks. However, their reliability remains a challenge, especially in long-horizon tasks, since they often produce overconfident yet wrong outputs. Conformal Prediction (CP) has been leveraged to address this issue by wrapping LLM outputs into prediction sets that contain the correct action with a user-defined confidence. When the prediction set is a singleton, the planner executes that action; otherwise, it requests help from a user. This has led to LLM-based planners that can ensure plan correctness with a user-defined probability. However, as LLMs are trained in an uncertainty-agnostic manner, without awareness of prediction sets, they tend to produce unnecessarily large sets, particularly at higher confidence levels, resulting in frequent human interventions limiting autonomous deployment. To address this, we introduce CoFineLLM (Conformal Finetuning for LLMs), the first CP-aware finetuning framework for LLM-based planners that explicitly reduces prediction-set size and, in turn, the need for user interventions. We evaluate our approach on multiple language-instructed robot planning problems and show consistent improvements over uncertainty-aware and uncertainty-agnostic finetuning baselines in terms of prediction-set size, and help rates. Finally, we demonstrate robustness of our method to out-of-distribution scenarios in hardware experiments. 

**Abstract (ZH)**: Conformal Finetuning for LLMs (CoFineLLM): Reducing Prediction-Set Size for Robust Autonomous Planning 

---
# Rep2Text: Decoding Full Text from a Single LLM Token Representation 

**Title (ZH)**: Rep2Text: 从单个LLM令牌表示解码完整文本 

**Authors**: Haiyan Zhao, Zirui He, Fan Yang, Ali Payani, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2511.06571)  

**Abstract**: Large language models (LLMs) have achieved remarkable progress across diverse tasks, yet their internal mechanisms remain largely opaque. In this work, we address a fundamental question: to what extent can the original input text be recovered from a single last-token representation within an LLM? We propose Rep2Text, a novel framework for decoding full text from last-token representations. Rep2Text employs a trainable adapter that projects a target model's internal representations into the embedding space of a decoding language model, which then autoregressively reconstructs the input text. Experiments on various model combinations (Llama-3.1-8B, Gemma-7B, Mistral-7B-v0.1, Llama-3.2-3B) demonstrate that, on average, over half of the information in 16-token sequences can be recovered from this compressed representation while maintaining strong semantic integrity and coherence. Furthermore, our analysis reveals an information bottleneck effect: longer sequences exhibit decreased token-level recovery while preserving strong semantic integrity. Besides, our framework also demonstrates robust generalization to out-of-distribution medical data. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多种任务上取得了显著进步，但其内部机制依然 largely 不透明。在本文中，我们探讨了一个基本问题：在大型语言模型中，单个最后一个token的表示能否恢复原始输入文本到什么程度？我们提出了Rep2Text，一个从最后一个token表示解码完整文本的新型框架。Rep2Text 使用一个可训练的适配器，将目标模型的内部表示投影到解码语言模型的嵌入空间中，然后自回归地重构输入文本。在不同模型组合（Llama-3.1-8B、Gemma-7B、Mistral-7B-v0.1、Llama-3.2-3B）上的实验表明，平均来说，可以从这种压缩表示中恢复超过一半的16-token序列信息，同时保持较强的意义连贯性和一致性。此外，我们的分析揭示了一个信息瓶颈效应：较长序列在保持较强的意义连贯性的同时，其token级恢复程度降低。此外，我们的框架还展示了对分布外医疗数据的强大泛化能力。 

---
# LLM For Loop Invariant Generation and Fixing: How Far Are We? 

**Title (ZH)**: LLM 在循环不变式生成与修复中的应用：我们走了多远？ 

**Authors**: Mostafijur Rahman Akhond, Saikat Chakraborty, Gias Uddin  

**Link**: [PDF](https://arxiv.org/pdf/2511.06552)  

**Abstract**: A loop invariant is a property of a loop that remains true before and after each execution of the loop. The identification of loop invariants is a critical step to support automated program safety assessment. Recent advancements in Large Language Models (LLMs) have demonstrated potential in diverse software engineering (SE) and formal verification tasks. However, we are not aware of the performance of LLMs to infer loop invariants. We report an empirical study of both open-source and closed-source LLMs of varying sizes to assess their proficiency in inferring inductive loop invariants for programs and in fixing incorrect invariants. Our findings reveal that while LLMs exhibit some utility in inferring and repairing loop invariants, their performance is substantially enhanced when supplemented with auxiliary information such as domain knowledge and illustrative examples. LLMs achieve a maximum success rate of 78\% in generating, but are limited to 16\% in repairing the invariant. 

**Abstract (ZH)**: 循环不变式是在每次执行循环之前和之后都保持真实的性质。识别循环不变式是支持自动化程序安全评估的关键步骤。大型语言模型（LLMs）在软件工程（SE）和形式验证任务中展现了多方面的潜力。然而，我们尚未见有关LLMs推断循环不变式的性能研究。我们报告了一项针对不同大小的开源和封闭源LLMs的实证研究，评估它们在为程序推断归纳循环不变式和修复错误不变式方面的 proficiency。我们的发现表明，尽管LLMs在推断和修复循环不变式方面表现出一定的效用，但通过补充领域知识和示例等辅助信息，其性能得到了显著提升。LLMs在生成循环不变式方面的最高成功率达到了78%，但在修复不变式方面的成功率仅限于16%。 

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
# Route Experts by Sequence, not by Token 

**Title (ZH)**: 基于序列，而不是基于词，进行路径专家划分 

**Authors**: Tiansheng Wen, Yifei Wang, Aosong Feng, Long Ma, Xinyang Liu, Yifan Wang, Lixuan Guo, Bo Chen, Stefanie Jegelka, Chenyu You  

**Link**: [PDF](https://arxiv.org/pdf/2511.06494)  

**Abstract**: Mixture-of-Experts (MoE) architectures scale large language models (LLMs) by activating only a subset of experts per token, but the standard TopK routing assigns the same fixed number of experts to all tokens, ignoring their varying complexity. Prior adaptive routing methods introduce additional modules and hyperparameters, often requiring costly retraining from scratch. We propose Sequence-level TopK (SeqTopK), a minimal modification that shifts the expert budget from the token level to the sequence level. By selecting the top $T \cdot K$ experts across all $T$ tokens, SeqTopK enables end-to-end learned dynamic allocation -- assigning more experts to difficult tokens and fewer to easy ones -- while preserving the same overall budget. SeqTopK requires only a few lines of code, adds less than 1% overhead, and remains fully compatible with pretrained MoE models. Experiments across math, coding, law, and writing show consistent improvements over TopK and prior parameter-free adaptive methods, with gains that become substantially larger under higher sparsity (up to 16.9%). These results highlight SeqTopK as a simple, efficient, and scalable routing strategy, particularly well-suited for the extreme sparsity regimes of next-generation LLMs. Code is available at this https URL. 

**Abstract (ZH)**: 序列级TopK（SeqTopK）路由方法 

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
# Ghost in the Transformer: Tracing LLM Lineage with SVD-Fingerprint 

**Title (ZH)**: Transformer中的幽灵：通过SVD-指纹追踪大模型谱系 

**Authors**: Suqing Wang, Ziyang Ma, Xinyi Li, Zuchao Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06390)  

**Abstract**: Large Language Models (LLMs) have rapidly advanced and are widely adopted across diverse fields. Due to the substantial computational cost and data requirements of training from scratch, many developers choose to fine-tune or modify existing open-source models. While most adhere to open-source licenses, some falsely claim original training despite clear derivation from public models. This raises pressing concerns about intellectual property protection and highlights the need for reliable methods to verify model provenance. In this paper, we propose GhostSpec, a lightweight yet effective method for verifying LLM lineage without access to training data or modification of model behavior. Our approach constructs compact and robust fingerprints by applying singular value decomposition (SVD) to invariant products of internal attention weight matrices, effectively capturing the structural identity of a model. Unlike watermarking or output-based methods, GhostSpec is fully data-free, non-invasive, and computationally efficient. It demonstrates strong robustness to sequential fine-tuning, pruning, block expansion, and even adversarial transformations. Extensive experiments show that GhostSpec can reliably trace the lineage of transformed models with minimal overhead. By offering a practical solution for model verification and reuse tracking, our method contributes to the protection of intellectual property and fosters a transparent, trustworthy ecosystem for large-scale language models. 

**Abstract (ZH)**: 大型语言模型（LLMs）已迅速发展并在多个领域广泛应用。由于从头训练的巨大计算成本和数据需求，许多开发人员选择微调或修改现有的开源模型。虽然大多数模型遵守开源许可，但有些却虚假声称原始训练，尽管这些模型明显源自公开模型。这引发了关于知识产权保护的重大关注，并凸显了需要可靠方法验证模型来源的需求。在本文中，我们提出了GhostSpec，一种无需访问训练数据或修改模型行为的轻量级有效方法，用于验证LLM谱系。我们的方法通过对内部注意权重矩阵的不变产品应用奇异值分解（SVD），构建紧凑且稳健的指纹，有效地捕捉模型的结构身份。与水印或基于输出的方法不同，GhostSpec完全无需数据、无侵入且计算效率高。它在序列微调、剪枝、块扩展和对抗变换等情况下表现出强大的 robustness。广泛的实验表明，GhostSpec可以在最小开销下可靠地追溯转换模型的谱系。通过提供模型验证和重用跟踪的实用解决方案，我们的方法有助于保护知识产权，并促进大规模语言模型的透明、可信赖生态系统。 

---
# TimeSense:Making Large Language Models Proficient in Time-Series Analysis 

**Title (ZH)**: TimeSense: 让大规模语言模型擅长时间序列分析 

**Authors**: Zhirui Zhang, Changhua Pei, Tianyi Gao, Zhe Xie, Yibo Hao, Zhaoyang Yu, Longlong Xu, Tong Xiao, Jing Han, Dan Pei  

**Link**: [PDF](https://arxiv.org/pdf/2511.06344)  

**Abstract**: In the time-series domain, an increasing number of works combine text with temporal data to leverage the reasoning capabilities of large language models (LLMs) for various downstream time-series understanding tasks. This enables a single model to flexibly perform tasks that previously required specialized models for each domain. However, these methods typically rely on text labels for supervision during training, biasing the model toward textual cues while potentially neglecting the full temporal features. Such a bias can lead to outputs that contradict the underlying time-series context. To address this issue, we construct the EvalTS benchmark, comprising 10 tasks across three difficulty levels, from fundamental temporal pattern recognition to complex real-world reasoning, to evaluate models under more challenging and realistic scenarios. We also propose TimeSense, a multimodal framework that makes LLMs proficient in time-series analysis by balancing textual reasoning with a preserved temporal sense. TimeSense incorporates a Temporal Sense module that reconstructs the input time-series within the model's context, ensuring that textual reasoning is grounded in the time-series dynamics. Moreover, to enhance spatial understanding of time-series data, we explicitly incorporate coordinate-based positional embeddings, which provide each time point with spatial context and enable the model to capture structural dependencies more effectively. Experimental results demonstrate that TimeSense achieves state-of-the-art performance across multiple tasks, and it particularly outperforms existing methods on complex multi-dimensional time-series reasoning tasks. 

**Abstract (ZH)**: 时间序列领域中，越来越多的研究将文本与时间数据相结合，利用大型语言模型（LLMs）的推理能力来完成各种下游时间序列理解任务。这种方法使得单一模型能够灵活地执行之前需要为每个领域专门设计模型的任务。然而，这些方法通常依赖文本标签作为训练期间的监督信息，使模型偏向于文本线索，同时可能忽视了全部的时间特征。这种偏差可能导致输出与底层时间序列上下文相矛盾。为解决这一问题，我们构建了EvalTS基准，包括三个难度级别共计10项任务，从基本的时间模式识别到复杂的现实情境推理，以在更具挑战性和现实性的场景中评估模型。我们还提出了一种多模态框架TimeSense，通过平衡文本推理与保留的时间感知来使LLMs擅长时间序列分析。TimeSense整合了一个时间感知模块，该模块在模型上下文中重构输入的时间序列，确保文本推理基于时间序列动态。此外，为了增强对时间序列数据的空间理解，我们显式地引入了基于坐标的绝对位置嵌入，为每个时间点提供空间上下文，使模型能够更有效地捕获结构依赖关系。实验结果表明，TimeSense在多个任务中取得了最先进的性能，并且在复杂的多维时间序列推理任务中尤其优于现有方法。 

---
# LLM-Guided Reinforcement Learning with Representative Agents for Traffic Modeling 

**Title (ZH)**: 基于代表性代理的LLM指导强化学习交通建模 

**Authors**: Hanlin Sun, Jiayang Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06260)  

**Abstract**: Large language models (LLMs) are increasingly used as behavioral proxies for self-interested travelers in agent-based traffic models. Although more flexible and generalizable than conventional models, the practical use of these approaches remains limited by scalability due to the cost of calling one LLM for every traveler. Moreover, it has been found that LLM agents often make opaque choices and produce unstable day-to-day dynamics. To address these challenges, we propose to model each homogeneous traveler group facing the same decision context with a single representative LLM agent who behaves like the population's average, maintaining and updating a mixed strategy over routes that coincides with the group's aggregate flow proportions. Each day, the LLM reviews the travel experience and flags routes with positive reinforcement that they hope to use more often, and an interpretable update rule then converts this judgment into strategy adjustments using a tunable (progressively decaying) step size. The representative-agent design improves scalability, while the separation of reasoning from updating clarifies the decision logic while stabilizing learning. In classic traffic assignment settings, we find that the proposed approach converges rapidly to the user equilibrium. In richer settings with income heterogeneity, multi-criteria costs, and multi-modal choices, the generated dynamics remain stable and interpretable, reproducing plausible behavioral patterns well-documented in psychology and economics, for example, the decoy effect in toll versus non-toll road selection, and higher willingness-to-pay for convenience among higher-income travelers when choosing between driving, transit, and park-and-ride options. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在基于代理的交通模型中被越来越多地用作自利旅行者的行为代理。为了解决这些挑战，我们提出用单个代表性的LLM代理来模拟在相同决策环境下面对相同旅行者的同质群体，该代理的行为类似于群体的平均行为，并且维护和更新与群体总体流量比例相符的混合策略。每天，LLM会回顾旅行体验并标记出那些希望更频繁使用的、具有正强化效果的路径，并通过可调（逐渐衰减）的学习步长将这种判断转换成策略调整。这种代理设计提高了模型的可扩展性，而将推理与更新分离则澄清了决策逻辑并稳定了学习过程。在经典的交通分配设置中，我们发现所提出的方法能够迅速收敛到用户均衡。在存在收入差异、多准则成本和多模式选择的更复杂设置中，生成的动力学保持稳定且可解释，能够再现心理学和经济学中广泛记录的合理行为模式，例如，过道效应在选择收费道路和非收费道路之间的选择中，以及高收入旅行者在选择驾驶、公共交通和停车换乘选项时对便利性的更高支付意愿。 

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
# LUT-LLM: Efficient Large Language Model Inference with Memory-based Computations on FPGAs 

**Title (ZH)**: LUT-LLM：基于内存计算的FPGA上高效大型语言模型推理 

**Authors**: Zifan He, Shengyu Ye, Rui Ma, Yang Wang, Jason Cong  

**Link**: [PDF](https://arxiv.org/pdf/2511.06174)  

**Abstract**: The rapid progress of large language models (LLMs) has advanced numerous applications, yet efficient single-batch inference remains vital for on-device intelligence. While FPGAs offer fine-grained data control and high energy efficiency, recent GPU optimizations have narrowed their advantage, especially under arithmetic-based computation. To overcome this, we leverage FPGAs' abundant on-chip memory to shift LLM inference from arithmetic- to memory-based computation through table lookups. We present LUT-LLM, the first FPGA accelerator enabling 1B+ LLM inference via vector-quantized memory operations. Our analysis identifies activation-weight co-quantization as the most effective scheme, supported by (1) bandwidth-aware parallel centroid search, (2) efficient 2D table lookups, and (3) a spatial-temporal hybrid design minimizing data caching. Implemented on an AMD V80 FPGA for a customized Qwen 3 1.7B model, LUT-LLM achieves 1.66x lower latency than AMD MI210 and 1.72x higher energy efficiency than NVIDIA A100, scaling to 32B models with 2.16x efficiency gain over A100. 

**Abstract (ZH)**: 基于查找表的FPGA加速器LUT-LLM：通过内存操作实现超亿参数大语言模型推理 

---
# LLM Attention Transplant for Transfer Learning of Tabular Data Across Disparate Domains 

**Title (ZH)**: 跨异质领域表格数据迁移学习的LLM注意力移植 

**Authors**: Ibna Kowsar, Kazi F. Akhter, Manar D. Samad  

**Link**: [PDF](https://arxiv.org/pdf/2511.06161)  

**Abstract**: Transfer learning of tabular data is non-trivial due to heterogeneity in the feature space across disparate domains. The limited success of traditional deep learning in tabular knowledge transfer can be advanced by leveraging large language models (LLMs). However, the efficacy of LLMs often stagnates for mixed data types structured in tables due to the limitations of text prompts and in-context learning. We propose a lightweight transfer learning framework that fine-tunes an LLM using source tabular data and transplants the LLM's selective $key$ and $value$ projection weights into a gated feature tokenized transformer (gFTT) built for tabular data. The gFTT model with cross-domain attention is fine-tuned using target tabular data for transfer learning, eliminating the need for shared features, LLM prompt engineering, and large-scale pretrained models. Our experiments using ten pairs of source-target data sets and 12 baselines demonstrate the superiority of the proposed LLM-attention transplant for transfer learning (LATTLE) method over traditional ML models, state-of-the-art deep tabular architectures, and transfer learning models trained on thousands to billions of tabular samples. The proposed attention transfer demonstrates an effective solution to learning relationships between data tables using an LLM in a low-resource learning environment. The source code for the proposed method is publicly available. 

**Abstract (ZH)**: 基于大语言模型的表格数据迁移学习框架：LATTLE方法 

---
# Large Language Models Develop Novel Social Biases Through Adaptive Exploration 

**Title (ZH)**: 大型语言模型通过适应性探索发展出新型社会偏见 

**Authors**: Addison J. Wu, Ryan Liu, Xuechunzi Bai, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2511.06148)  

**Abstract**: As large language models (LLMs) are adopted into frameworks that grant them the capacity to make real decisions, it is increasingly important to ensure that they are unbiased. In this paper, we argue that the predominant approach of simply removing existing biases from models is not enough. Using a paradigm from the psychology literature, we demonstrate that LLMs can spontaneously develop novel social biases about artificial demographic groups even when no inherent differences exist. These biases result in highly stratified task allocations, which are less fair than assignments by human participants and are exacerbated by newer and larger models. In social science, emergent biases like these have been shown to result from exploration-exploitation trade-offs, where the decision-maker explores too little, allowing early observations to strongly influence impressions about entire demographic groups. To alleviate this effect, we examine a series of interventions targeting model inputs, problem structure, and explicit steering. We find that explicitly incentivizing exploration most robustly reduces stratification, highlighting the need for better multifaceted objectives to mitigate bias. These results reveal that LLMs are not merely passive mirrors of human social biases, but can actively create new ones from experience, raising urgent questions about how these systems will shape societies over time. 

**Abstract (ZH)**: 大规模语言模型在具备做出实际决策的能力时，消除其偏见的途径已不足以确保公平。本论文通过心理学范式表明，即使不存在固有差异，语言模型也可能自发发展出对人工社会群体的新偏见，导致高度分层的任务分配，这种分配比人类参与者制定的分配更加不公平，并且还受到更大规模模型的加剧。社会科学研究表明，此类新兴偏见源于探索与利用之间的权衡，在这种权衡中，决策者探索不足，导致早期观察强烈影响整个社会群体的总体印象。为了减轻这一影响，我们研究了一系列针对模型输入、问题结构和明确引导的干预措施。研究发现，明确激励探索最有效地减少了分层现象，突显了需要更好的多方面目标来减轻偏见的重要性。这些结果揭示了语言模型不仅仅是人类社会偏见的被动镜像，它们可以从经验中主动创造新的偏见，这迫切需要探讨这些系统如何随着时间改变社会。 

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
# MoSKA: Mixture of Shared KV Attention for Efficient Long-Sequence LLM Inference 

**Title (ZH)**: MoSKA: 共享键值注意力的混合高效长序列语言模型推理 

**Authors**: Myunghyun Rhee, Sookyung Choi, Euiseok Kim, Joonseop Sim, Youngpyo Joo, Hoshik Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.06010)  

**Abstract**: The escalating context length in Large Language Models (LLMs) creates a severe performance bottleneck around the Key-Value (KV) cache, whose memory-bound nature leads to significant GPU under-utilization. This paper introduces Mixture of Shared KV Attention (MoSKA), an architecture that addresses this challenge by exploiting the heterogeneity of context data. It differentiates between per-request unique and massively reused shared sequences. The core of MoSKA is a novel Shared KV Attention mechanism that transforms the attention on shared data from a series of memory-bound GEMV operations into a single, compute-bound GEMM by batching concurrent requests. This is supported by an MoE-inspired sparse attention strategy that prunes the search space and a tailored Disaggregated Infrastructure that specializes hardware for unique and shared data. This comprehensive approach demonstrates a throughput increase of up to 538.7x over baselines in workloads with high context sharing, offering a clear architectural path toward scalable LLM inference. 

**Abstract (ZH)**: 大规模语言模型中上下文长度不断提升导致关键值缓存性能瓶颈，MoSKA架构通过利用上下文数据的异构性来应对这一挑战。它区分了每个请求的唯一序列和大规模重复的共享序列。MoSKA的核心是一种新颖的共享关键值注意机制，将对共享数据的注意从一系列内存绑定的GEMV操作转换为针对并发请求的单一、计算绑定的GEMM操作。该机制通过一种受专家门控启发的稀疏注意策略来精简搜索空间，并通过定制化的分解基础设施专门化硬件以处理独特的和共享的数据。这种综合方法在高上下文共享的工作负载中展示了高达538.7倍的吞吐量提升，提供了一条通往可扩展的大规模语言模型推理的清晰架构路径。 

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
# DiA-gnostic VLVAE: Disentangled Alignment-Constrained Vision Language Variational AutoEncoder for Robust Radiology Reporting with Missing Modalities 

**Title (ZH)**: DiA-gnostic VLVAE: 解耦联对齐约束视觉语言变分自编码器在缺失模态下实现稳健的放射学报告生成 

**Authors**: Nagur Shareef Shaik, Teja Krishna Cherukuri, Adnan Masood, Dong Hye Ye  

**Link**: [PDF](https://arxiv.org/pdf/2511.05968)  

**Abstract**: The integration of medical images with clinical context is essential for generating accurate and clinically interpretable radiology reports. However, current automated methods often rely on resource-heavy Large Language Models (LLMs) or static knowledge graphs and struggle with two fundamental challenges in real-world clinical data: (1) missing modalities, such as incomplete clinical context , and (2) feature entanglement, where mixed modality-specific and shared information leads to suboptimal fusion and clinically unfaithful hallucinated findings. To address these challenges, we propose the DiA-gnostic VLVAE, which achieves robust radiology reporting through Disentangled Alignment. Our framework is designed to be resilient to missing modalities by disentangling shared and modality-specific features using a Mixture-of-Experts (MoE) based Vision-Language Variational Autoencoder (VLVAE). A constrained optimization objective enforces orthogonality and alignment between these latent representations to prevent suboptimal fusion. A compact LLaMA-X decoder then uses these disentangled representations to generate reports efficiently. On the IU X-Ray and MIMIC-CXR datasets, DiA has achieved competetive BLEU@4 scores of 0.266 and 0.134, respectively. Experimental results show that the proposed method significantly outperforms state-of-the-art models. 

**Abstract (ZH)**: 医学影像与临床背景的整合对于生成准确且临床可解释的放射学报告至关重要。然而，当前的自动化方法往往依赖资源密集型大型语言模型（LLMs）或静态知识图谱，并且在真实世界的临床数据中面临两大根本挑战：（1）缺失模态，如不完整的临床背景，以及（2）特征纠缠，其中混合模态特定和共享信息导致次优融合并产生临床不忠实的虚幻发现。为了解决这些挑战，我们提出了DiA-gnostic VLVAE，通过消 entangled alignment实现稳健的放射学报告。该框架通过基于Mixture-of-Experts（MoE）的Vision-Language Variational Autoencoder（VLVAE）来消 entangle共享和模态特定特征，以增强对缺失模态的鲁棒性。通过约束优化目标，确保这些潜在表示之间的正交性和对齐，从而防止次优融合。紧凑型LLaMA-X解码器然后使用这些消 entangled的表示高效地生成报告。在IU X-Ray和MIMIC-CXR数据集上，DiA分别获得了竞争性的BLEU@4分数0.266和0.134。实验结果表明，所提出的方法显著优于现有最佳模型。 

---
# Reinforcement Learning Improves Traversal of Hierarchical Knowledge in LLMs 

**Title (ZH)**: 强化学习提高大型语言模型中层级知识的遍历能力 

**Authors**: Renfei Zhang, Manasa Kaniselvan, Niloofar Mireshghallah  

**Link**: [PDF](https://arxiv.org/pdf/2511.05933)  

**Abstract**: Reinforcement learning (RL) is often credited with improving language model reasoning and generalization at the expense of degrading memorized knowledge. We challenge this narrative by observing that RL-enhanced models consistently outperform their base and supervised fine-tuned (SFT) counterparts on pure knowledge recall tasks, particularly those requiring traversal of hierarchical, structured knowledge (e.g., medical codes). We hypothesize these gains stem not from newly acquired data, but from improved procedural skills in navigating and searching existing knowledge hierarchies within the model parameters. To support this hypothesis, we show that structured prompting, which explicitly guides SFTed models through hierarchical traversal, recovers most of the performance gap (reducing 24pp to 7pp on MedConceptsQA for DeepSeek-V3/R1). We further find that while prompting improves final-answer accuracy, RL-enhanced models retain superior ability to recall correct procedural paths on deep-retrieval tasks. Finally our layer-wise internal activation analysis reveals that while factual representations (e.g., activations for the statement "code 57.95 refers to urinary infection") maintain high cosine similarity between SFT and RL models, query representations (e.g., "what is code 57.95") diverge noticeably, indicating that RL primarily transforms how models traverse knowledge rather than the knowledge representation itself. 

**Abstract (ZH)**: 强化学习（RL）在提高语言模型推理和泛化能力的同时，并未牺牲记忆知识的能力：一种挑战性观点 

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
# Retrieval-Augmented Generation in Medicine: A Scoping Review of Technical Implementations, Clinical Applications, and Ethical Considerations 

**Title (ZH)**: 医学中的检索增强生成：一项关于技术实现、临床应用及伦理考量的综述 

**Authors**: Rui Yang, Matthew Yu Heng Wong, Huitao Li, Xin Li, Wentao Zhu, Jingchi Liao, Kunyu Yu, Jonathan Chong Kai Liew, Weihao Xuan, Yingjian Chen, Yuhe Ke, Jasmine Chiat Ling Ong, Douglas Teodoro, Chuan Hong, Daniel Shi Wei Ting, Nan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05901)  

**Abstract**: The rapid growth of medical knowledge and increasing complexity of clinical practice pose challenges. In this context, large language models (LLMs) have demonstrated value; however, inherent limitations remain. Retrieval-augmented generation (RAG) technologies show potential to enhance their clinical applicability. This study reviewed RAG applications in medicine. We found that research primarily relied on publicly available data, with limited application in private data. For retrieval, approaches commonly relied on English-centric embedding models, while LLMs were mostly generic, with limited use of medical-specific LLMs. For evaluation, automated metrics evaluated generation quality and task performance, whereas human evaluation focused on accuracy, completeness, relevance, and fluency, with insufficient attention to bias and safety. RAG applications were concentrated on question answering, report generation, text summarization, and information extraction. Overall, medical RAG remains at an early stage, requiring advances in clinical validation, cross-linguistic adaptation, and support for low-resource settings to enable trustworthy and responsible global use. 

**Abstract (ZH)**: 医疗领域中检索增强生成技术的应用研究 

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
# MOSS: Efficient and Accurate FP8 LLM Training with Microscaling and Automatic Scaling 

**Title (ZH)**: MOSS：基于微缩和自动缩放的高效准确FP8 LLM训练 

**Authors**: Yu Zhang, Hui-Ling Zhen, Mingxuan Yuan, Bei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05811)  

**Abstract**: Training large language models with FP8 formats offers significant efficiency gains. However, the reduced numerical precision of FP8 poses challenges for stable and accurate training. Current frameworks preserve training performance using mixed-granularity quantization, i.e., applying per-group quantization for activations and per-tensor/block quantization for weights. While effective, per-group quantization requires scaling along the inner dimension of matrix multiplication, introducing additional dequantization overhead. Moreover, these frameworks often rely on just-in-time scaling to dynamically adjust scaling factors based on the current data distribution. However, this online quantization is inefficient for FP8 training, as it involves multiple memory reads and writes that negate the performance benefits of FP8. To overcome these limitations, we propose MOSS, a novel FP8 training framework that ensures both efficiency and numerical stability. MOSS introduces two key innovations: (1) a two-level microscaling strategy for quantizing sensitive activations, which balances precision and dequantization cost by combining a high-precision global scale with compact, power-of-two local scales; and (2) automatic scaling for weights in linear layers, which eliminates the need for costly max-reduction operations by predicting and adjusting scaling factors during training. Leveraging these techniques, MOSS enables efficient FP8 training of a 7B parameter model, achieving performance comparable to the BF16 baseline while achieving up to 34% higher training throughput. 

**Abstract (ZH)**: 使用FP8格式训练大型语言模型提供了显著的效率增益。然而，FP8数值精度的降低为训练的稳定性和准确性带来了挑战。当前框架通过分层级量化来保持训练性能，即对激活值使用分组量化，对权重使用张量级/块量化。虽然有效，但分组量化要求沿矩阵乘法的内部维度进行缩放，引入了额外的去量化开销。此外，这些框架通常依赖于即时缩放来根据当前数据分布动态调整缩放因子。然而，这种在线量化对于FP8训练来说是低效的，因为它涉及多次内存读写，抵消了FP8的性能优势。为了克服这些限制，我们提出了MOSS，一种新型的FP8训练框架，确保了效率和数值稳定性。MOSS引入了两项创新技术：（1）两级微缩放策略，通过结合高精度全局尺度和紧凑的2的幂本地尺度来平衡精度和去量化成本；（2）线性层权重的自动缩放，通过在训练过程中预测和调整缩放因子来消除昂贵的最大值归约操作。利用这些技术，MOSS能够高效地训练一个7B参数模型，性能与BF16基线相当，同时训练吞吐量提高高达34%。 

---
# When AI Meets the Web: Prompt Injection Risks in Third-Party AI Chatbot Plugins 

**Title (ZH)**: 当AI遇到网络：第三方AI聊天机器人插件中的提示注入风险 

**Authors**: Yigitcan Kaya, Anton Landerer, Stijn Pletinckx, Michelle Zimmermann, Christopher Kruegel, Giovanni Vigna  

**Link**: [PDF](https://arxiv.org/pdf/2511.05797)  

**Abstract**: Prompt injection attacks pose a critical threat to large language models (LLMs), with prior work focusing on cutting-edge LLM applications like personal copilots. In contrast, simpler LLM applications, such as customer service chatbots, are widespread on the web, yet their security posture and exposure to such attacks remain poorly understood. These applications often rely on third-party chatbot plugins that act as intermediaries to commercial LLM APIs, offering non-expert website builders intuitive ways to customize chatbot behaviors. To bridge this gap, we present the first large-scale study of 17 third-party chatbot plugins used by over 10,000 public websites, uncovering previously unknown prompt injection risks in practice. First, 8 of these plugins (used by 8,000 websites) fail to enforce the integrity of the conversation history transmitted in network requests between the website visitor and the chatbot. This oversight amplifies the impact of direct prompt injection attacks by allowing adversaries to forge conversation histories (including fake system messages), boosting their ability to elicit unintended behavior (e.g., code generation) by 3 to 8x. Second, 15 plugins offer tools, such as web-scraping, to enrich the chatbot's context with website-specific content. However, these tools do not distinguish the website's trusted content (e.g., product descriptions) from untrusted, third-party content (e.g., customer reviews), introducing a risk of indirect prompt injection. Notably, we found that ~13% of e-commerce websites have already exposed their chatbots to third-party content. We systematically evaluate both vulnerabilities through controlled experiments grounded in real-world observations, focusing on factors such as system prompt design and the underlying LLM. Our findings show that many plugins adopt insecure practices that undermine the built-in LLM safeguards. 

**Abstract (ZH)**: Prompt注入攻击对大型语言模型构成关键威胁：以个人副驾驶员类前沿应用为例，而简单的客户服务中心聊天机器人等应用广泛存在于网络上，但它们的安全状态和对抗此类攻击的暴露程度仍不甚理解。为了弥合这一差距，我们首次对10,000多个公开网站使用的17个第三方聊天机器人插件进行了大规模研究，揭示了实践中未知的聊天记录篡改风险。首先，8个插件（服务于8,000个网站）未能确保网站访客与聊天机器人之间网络请求中传递的对话历史的完整性。这种疏忽放大了直接聊天记录篡改攻击的影响，允许攻击者伪造对话历史（包括虚假系统消息），将其诱导未预期行为（例如代码生成）的能力提升3到8倍。其次，15个插件提供了一些工具，如网页抓取，以丰富聊天机器人的上下文，使其包含网站特定的内容。然而，这些工具无法区分受信任的内容（例如产品描述）与不受信任的第三方内容（例如客户评论），从而引入了潜在的间接聊天记录篡改风险。值得注意的是，我们发现约13%的电子商务网站已经让其聊天机器人暴露在第三方内容之下。通过基于实际观察的受控实验系统地评估这些脆弱性，重点关注系统提示设计和底层的大型语言模型等因素，我们的研究结果表明，许多插件采用了不安全的做法，从而削弱了内置的大型语言模型保护措施。 

---
# DRAGON: Guard LLM Unlearning in Context via Negative Detection and Reasoning 

**Title (ZH)**: DRAGON: 在上下文中外包LLM脱忆的负检测与推理方法 

**Authors**: Yaxuan Wang, Chris Yuhao Liu, Quan Liu, Jinglong Pang, Wei Wei, Yujia Bao, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05784)  

**Abstract**: Unlearning in Large Language Models (LLMs) is crucial for protecting private data and removing harmful knowledge. Most existing approaches rely on fine-tuning to balance unlearning efficiency with general language capabilities. However, these methods typically require training or access to retain data, which is often unavailable in real world scenarios. Although these methods can perform well when both forget and retain data are available, few works have demonstrated equivalent capability in more practical, data-limited scenarios. To overcome these limitations, we propose Detect-Reasoning Augmented GeneratiON (DRAGON), a systematic, reasoning-based framework that utilizes in-context chain-of-thought (CoT) instructions to guard deployed LLMs before inference. Instead of modifying the base model, DRAGON leverages the inherent instruction-following ability of LLMs and introduces a lightweight detection module to identify forget-worthy prompts without any retain data. These are then routed through a dedicated CoT guard model to enforce safe and accurate in-context intervention. To robustly evaluate unlearning performance, we introduce novel metrics for unlearning performance and the continual unlearning setting. Extensive experiments across three representative unlearning tasks validate the effectiveness of DRAGON, demonstrating its strong unlearning capability, scalability, and applicability in practical scenarios. 

**Abstract (ZH)**: 大型语言模型中的忘记学习对于保护私人数据和去除有害知识至关重要。现有大多数方法依赖微调以平衡忘记学习效率和通用语言能力。然而，这些方法通常需要训练数据或访问原始数据，而在实际场景中这往往是不可用的。尽管当忘记和保留数据都可用时，这些方法可以表现出色，但在数据受限的实际场景中，很少有研究能够展示相当的能力。为了克服这些局限，我们提出了一种基于推理的检测增强生成框架DRAGON，该框架利用上下文中的链式思考指令，在推理前保护部署的LLM。DRAGON不修改基础模型，而是利用LLM的内置指令遵循能力，并引入一个轻量级的检测模块来识别值得遗忘的提示，而无需保留数据。然后，这些提示通过一个专门的CoT防护模型来进行干预，以确保安全和准确的上下文内干预。为了稳健地评估忘记学习性能，我们引入了新的评估指标，并在持续忘记学习设置中进行了评估。广泛的实验在三个代表性的忘记学习任务上验证了DRAGON的有效性，显示出其强大的忘记学习能力、可扩展性和在实际场景中的适用性。 

---
# Lived Experience in Dialogue: Co-designing Personalization in Large Language Models to Support Youth Mental Well-being 

**Title (ZH)**: 在对话中生活的体验：与青少年心理健康支持相关的大型语言模型个性化共同设计 

**Authors**: Kathleen W. Guan, Sarthak Giri, Mohammed Amara, Bernard J. Jansen, Enrico Liscio, Milena Esherick, Mohammed Al Owayyed, Ausrine Ratkute, Gayane Sedrakyan, Mark de Reuver, Joao Fernando Ferreira Goncalves, Caroline A. Figueroa  

**Link**: [PDF](https://arxiv.org/pdf/2511.05769)  

**Abstract**: Youth increasingly turn to large language models (LLMs) for mental well-being support, yet current personalization in LLMs can overlook the heterogeneous lived experiences shaping their needs. We conducted a participatory study with youth, parents, and youth care workers (N=38), using co-created youth personas as scaffolds, to elicit community perspectives on how LLMs can facilitate more meaningful personalization to support youth mental well-being. Analysis identified three themes: person-centered contextualization responsive to momentary needs, explicit boundaries around scope and offline referral, and dialogic scaffolding for reflection and autonomy. We mapped these themes to persuasive design features for task suggestions, social facilitation, and system trustworthiness, and created corresponding dialogue extracts to guide LLM fine-tuning. Our findings demonstrate how lived experience can be operationalized to inform design features in LLMs, which can enhance the alignment of LLM-based interventions with the realities of youth and their communities, contributing to more effectively personalized digital well-being tools. 

**Abstract (ZH)**: 青少年越来越多地利用大型语言模型（LLMs）寻求心理健康支持，但当前的个性化设置可能会忽视塑造其需求的异质性生活体验。我们通过与青少年、家长和青少年护理工作者（共38人）进行参与式研究，使用共创的青少年人物作为支架，探讨社区视角下LLMs如何促进更具有意义的个性化以支持青少年心理健康。分析识别出三个主题：以个人为中心的 contextualization，响应即时需要，明确的边界范围与离线转介，以及促进反思与自主性的对话支架。我们将这些主题映射到说服性设计特征的任务建议、社会促进和系统可信性上，并创建相应的对话提取以指导LLM调整。我们的研究结果展示了如何将生活体验具体化以指导LLMs的设计特征，这有助于使基于LLM的干预措施更契合青少年及其社区的实际情况，从而促进更加有效的个性化数字心理健康工具。 

---
# Beyond Redundancy: Diverse and Specialized Multi-Expert Sparse Autoencoder 

**Title (ZH)**: 超越冗余：多样化和专门化的多专家稀疏自编码器 

**Authors**: Zhen Xu, Zhen Tan, Song Wang, Kaidi Xu, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.05745)  

**Abstract**: Sparse autoencoders (SAEs) have emerged as a powerful tool for interpreting large language models (LLMs) by decomposing token activations into combinations of human-understandable features. While SAEs provide crucial insights into LLM explanations, their practical adoption faces a fundamental challenge: better interpretability demands that SAEs' hidden layers have high dimensionality to satisfy sparsity constraints, resulting in prohibitive training and inference costs. Recent Mixture of Experts (MoE) approaches attempt to address this by partitioning SAEs into narrower expert networks with gated activation, thereby reducing computation. In a well-designed MoE, each expert should focus on learning a distinct set of features. However, we identify a \textit{critical limitation} in MoE-SAE: Experts often fail to specialize, which means they frequently learn overlapping or identical features. To deal with it, we propose two key innovations: (1) Multiple Expert Activation that simultaneously engages semantically weighted expert subsets to encourage specialization, and (2) Feature Scaling that enhances diversity through adaptive high-frequency scaling. Experiments demonstrate a 24\% lower reconstruction error and a 99\% reduction in feature redundancy compared to existing MoE-SAE methods. This work bridges the interpretability-efficiency gap in LLM analysis, allowing transparent model inspection without compromising computational feasibility. 

**Abstract (ZH)**: 基于专家的混合模型在稀疏自动编码器中的关键改进：提高大型语言模型解释的透明性和效率 

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
# Optimizing Diversity and Quality through Base-Aligned Model Collaboration 

**Title (ZH)**: 基于基模型对齐的多样性与质量优化合作方法 

**Authors**: Yichen Wang, Chenghao Yang, Tenghao Huang, Muhao Chen, Jonathan May, Mina Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.05650)  

**Abstract**: Alignment has greatly improved large language models (LLMs)' output quality at the cost of diversity, yielding highly similar outputs across generations. We propose Base-Aligned Model Collaboration (BACo), an inference-time token-level model collaboration framework that dynamically combines a base LLM with its aligned counterpart to optimize diversity and quality. Inspired by prior work (Fei et al., 2025), BACo employs routing strategies that determine, at each token, from which model to decode based on next-token prediction uncertainty and predicted contents' semantic role. Prior diversity-promoting methods, such as retraining, prompt engineering, and multi-sampling methods, improve diversity but often degrade quality or require costly decoding or post-training. In contrast, BACo achieves both high diversity and quality post hoc within a single pass, while offering strong controllability. We explore a family of routing strategies, across three open-ended generation tasks and 13 metrics covering diversity and quality, BACo consistently surpasses state-of-the-art inference-time baselines. With our best router, BACo achieves a 21.3% joint improvement in diversity and quality. Human evaluations also mirror these improvements. The results suggest that collaboration between base and aligned models can optimize and control diversity and quality. 

**Abstract (ZH)**: Base-Aligned Model Collaboration (BACo): Inference-Time Token-Level Model Collaboration for Optimizing Diversity and Quality 

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
# Lookahead Unmasking Elicits Accurate Decoding in Diffusion Language Models 

**Title (ZH)**: 前瞻解掩揭示扩散语言模型的准确解码 

**Authors**: Sanghyun Lee, Seungryong Kim, Jongho Park, Dongmin Park  

**Link**: [PDF](https://arxiv.org/pdf/2511.05563)  

**Abstract**: Masked Diffusion Models (MDMs) as language models generate by iteratively unmasking tokens, yet their performance crucially depends on the inference time order of unmasking. Prevailing heuristics, such as confidence based sampling, are myopic: they optimize locally, fail to leverage extra test-time compute, and let early decoding mistakes cascade. We propose Lookahead Unmasking (LookUM), which addresses these concerns by reformulating sampling as path selection over all possible unmasking orders without the need for an external reward model. Our framework couples (i) a path generator that proposes paths by sampling from pools of unmasking sets with (ii) a verifier that computes the uncertainty of the proposed paths and performs importance sampling to subsequently select the final paths. Empirically, erroneous unmasking measurably inflates sequence level uncertainty, and our method exploits this to avoid error-prone trajectories. We validate our framework across six benchmarks, such as mathematics, planning, and coding, and demonstrate consistent performance improvements. LookUM requires only two to three paths to achieve peak performance, demonstrating remarkably efficient path selection. The consistent improvements on both LLaDA and post-trained LLaDA 1.5 are particularly striking: base LLaDA with LookUM rivals the performance of RL-tuned LLaDA 1.5, while LookUM further enhances LLaDA 1.5 itself showing that uncertainty based verification provides orthogonal benefits to reinforcement learning and underscoring the versatility of our framework. Code will be publicly released. 

**Abstract (ZH)**: 前瞻性解码（LookUM）：Masked Diffusion Models作为一种语言模型通过迭代解码生成，其性能关键取决于解码顺序。我们提出了一种前瞻性解码方法（Lookahead Unmasking, LookUM），通过路径选择而不是局部优化来克服现有方法的局限性。我们验证了该方法在数学、规划和编码等六个基准上的表现，并显示出一致的性能提升。LookUM仅需选择两到三条路径即可达到最佳性能，展示了高效的路径选择能力。在基LLaDA和后训练LLaDA 1.5上的一致改进尤为显著：结合前瞻性解码的LLaDA与RL调优的LLaDA 1.5性能相当，进一步证明了基于不确定性的验证为强化学习提供了独立的优势，并突显了我们框架的通用性。代码将公开发布。 

---
# Sample-Efficient Language Modeling with Linear Attention and Lightweight Enhancements 

**Title (ZH)**: 基于线性注意机制和轻量级增强的样本高效语言模型 

**Authors**: Patrick Haller, Jonas Golde, Alan Akbik  

**Link**: [PDF](https://arxiv.org/pdf/2511.05560)  

**Abstract**: We study architectural and optimization tech- niques for sample-efficient language modeling under the constraints of the BabyLM 2025 shared task. Our model, BLaLM, replaces self-attention with a linear-time mLSTM to- ken mixer and explores lightweight enhance- ments, including short convolutions, sliding window attention with dynamic modulation, and Hedgehog feature maps. To support train- ing in low-resource settings, we curate a high- quality corpus emphasizing readability and ped- agogical structure. Experiments across both STRICT and STRICT-SMALL tracks show that (1) linear attention combined with sliding win- dow attention consistently improves zero-shot performance, and (2) the Muon optimizer stabi- lizes convergence and reduces perplexity over AdamW. These results highlight effective strate- gies for efficient language modeling without relying on scale. 

**Abstract (ZH)**: 我们针对BabyLM 2025 共享任务的约束条件，研究了样本高效语言模型的架构和优化技术。我们的模型BLaLM 使用线性时间 mLSTM 令牌混音器取代了自注意力机制，并探索了轻量级增强技术，包括短卷积、滑动窗口注意力以及动态调制和刺猬特征图。为支持低资源环境下的训练，我们精心筛选了高质量语料库，强调易读性和教学结构。在STRICT 和STRICT-SMALL 轨道的实验中表明，(1) 线性注意力与滑动窗口注意力结合使用一致地提升了零样本性能，且(2) 穆翁优化器相比AdamW 更能稳定收敛并降低困惑度。这些结果突出了在不依赖规模的情况下进行高效语言建模的有效策略。 

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
# Gravity-Awareness: Deep Learning Models and LLM Simulation of Human Awareness in Altered Gravity 

**Title (ZH)**: 重力意识：深度学习模型与人类在改变重力环境下的意识模拟 

**Authors**: Bakytzhan Alibekov, Alina Gutoreva, Elisa Raffaella-Ferre  

**Link**: [PDF](https://arxiv.org/pdf/2511.05536)  

**Abstract**: Earth's gravity has fundamentally shaped human development by guiding the brain's integration of vestibular, visual, and proprioceptive inputs into an internal model of gravity: a dynamic neural representation enabling prediction and interpretation of gravitational forces. This work presents a dual computational framework to quantitatively model these adaptations. The first component is a lightweight Multi-Layer Perceptron (MLP) that predicts g-load-dependent changes in key electroencephalographic (EEG) frequency bands, representing the brain's cortical state. The second component utilizes a suite of independent Gaussian Processes (GPs) to model the body's broader physiological state, including Heart Rate Variability (HRV), Electrodermal Activity (EDA), and motor behavior. Both models were trained on data derived from a comprehensive review of parabolic flight literature, using published findings as anchor points to construct robust, continuous functions. To complement this quantitative analysis, we simulated subjective human experience under different gravitational loads, ranging from microgravity (0g) and partial gravity (Moon 0.17g, Mars 0.38g) to hypergravity associated with spacecraft launch and re-entry (1.8g), using a large language model (Claude 3.5 Sonnet). The model was prompted with physiological parameters to generate introspective narratives of alertness and self-awareness, which closely aligned with the quantitative findings from both the EEG and physiological models. This combined framework integrates quantitative physiological modeling with generative cognitive simulation, offering a novel approach to understanding and predicting human performance in altered gravity 

**Abstract (ZH)**: 地球的重力通过指导脑部对前庭、视觉和本体感觉输入的整合，形成对重力的内部模型，从而从根本上影响人类的发展：一种动态的神经代表，使大脑能够预测和解释重力作用。本研究提出了一种双计算框架来定量建模这些适应性变化。第一个组成部分是一个轻量级的多层感知器（MLP），用于预测与g负荷相关的关键脑电图（EEG）频率带的变化，代表大脑皮层状态。第二个组成部分则利用一系列独立的高斯过程（GPs）来建模人体更广泛的生理状态，包括心率变异性（HRV）、皮肤电活动（EDA）和运动行为。这两个模型均基于综合回顾抛物线飞行文献的数据进行训练，使用已发表的研究结果作为锚点来构建稳健的连续函数。为了补充这种定量分析，我们使用大型语言模型（Claude 3.5 Sonnet）模拟了在不同重力负荷下的主观人类体验，从微重力（0g）和部分重力（月球0.17g，火星0.38g）到与航天器发射和重返大气层相关的超重力（1.8g），并生成了涉及警觉性和自我意识的内省叙述，这些叙述与EEG和生理模型的定量发现紧密一致。结合这种框架将定量生理建模与生成性认知模拟整合起来，提供了一种理解与预测人类在改变重力环境中的表现的新方法。 

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
